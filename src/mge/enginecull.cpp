// EngineCull — see enginecull.h for why this module exists and what stage it is at.
//
// This is msoc's cullShowBody with every occlusion branch deleted. D0 established
// that MSOC runs permanently in its zero-occluder fast path for us, so the pieces
// that fall away are: occluder rasterisation, the occluder property cache, the
// threadpool, drain slots and verdicts, the temporal cache, TestRect, debug tints,
// the traversal distance gate, and the mask snapshot. What is left is the engine's
// own traversal plus a deferral point.
//
// Deferral is NOT an occlusion leftover. It is what lets the traversal run at
// BeginScene(0) — earlier than the engine's natural CullShow — so MGE's produce
// kickoff gets a CURRENT-frame visible set. That timing is a correctness
// requirement, not a nicety: buildFrustumVisibleSet feeds these keys straight into
// GeometryCache::ensureLive as raw NiTriShape*, and current-frame-ness is the only
// thing guaranteeing they are still alive (renderdepth.cpp:246). A one-frame-stale
// feed would be use-after-free, which is why the cheap "just observe the engine"
// design was rejected — see the re-weigh in tasks/msoc-detour-absorb.md.

// MGE-side consumer prelude — must precede the SharedSE NI headers (it sets
// SE_IS_MGE / SE_TARGETS_MW and does the near/far undef NICamera.h needs). Also
// where windows.h comes from, for GetModuleHandleA below.
#include "mge_se_prelude.h"

// SharedSE NI types.
#include "NIAVObject.h"
#include "NICamera.h"
#include "NIGeometry.h"
#include "NINode.h"
#include "NIPoint4.h"
#include "NIProperty.h"
#include "NIRTTIDefines.h"

// se::memory::genJumpUnprotected — the prologue patch. MWSE-provided, which is
// what keeps the detour inside PRIME DIRECTIVE 6.
#include "MemoryUtil.h"

#include <unordered_map>
#include <vector>

#include "enginecull.h"
#include "configuration.h"
#include "datahandler_view.h"
#include "scenegraph.h"
#include "worldcontroller_view.h"
#include "support/log.h"

namespace MGE::EngineCull {

namespace {

// NiAVObject::CullShow. The address comes from our own msoc-plugin source, which
// took it from MWSE's published symbols — no disassembly of our own.
constexpr uintptr_t kCullShowAddr = 0x6EB480;

// ---------------------------------------------------------------------------
// Per-frame state
// ---------------------------------------------------------------------------

// True while the traversal is collecting instead of displaying. Mirrors msoc's
// g_msocActive, and carries the same contract: when FALSE this file's traversal
// must behave identically to the engine's own CullShow. That equivalence is what
// makes the prologue patch survivable for the six non-main-scene callers.
//
// Leaking this true is the one catastrophic failure mode in the module — every
// CullShow in the process would collect and nothing would ever draw — so it is
// only ever set through DeferGuard below.
bool g_deferring = false;

bool g_installed = false;

bool g_skipOwnedOpaque = false;
bool g_skipOwnedAlpha  = false;

// Recursion depth. Interior nodes display() into their children, which re-enter
// this body through the patched prologue, so "outer entry" (depth 1) is the only
// way to tell one of CullShow's seven callers from the traversal's own descent.
// msoc used !g_msocActive for the same job; that only works while a build is
// active, and the root capture below has to run on frames where none is.
int g_callDepth = 0;

// The EXACT NiAVObject* the engine hands its top-level world-camera CullShow.
// NOT worldObjectRoot — an ancestor that contains it. Captured from the engine's
// own call (we cannot synthesise it: traversing a wider root would collect
// leaves the engine's pass never displays, a narrower one would miss whole
// subtrees, and either way the display-pass identity match below stops firing).
// Only ever pointer-compared, never dereferenced, so a root freed by a
// worldspace change is caught without a crash.
NI::AVObject* g_topLevelRoot = nullptr;

// Set by classifyNow() when it has collected leaves that the engine's own
// top-level pass still has to draw. Consumed by that pass, cleared by
// beginFrame() — a frame that ends with it still set drew no world geometry.
bool g_displayPending = false;

// World-camera outer entries since beginFrame(). Guards the "engine already
// handled this scene" case (msoc's isTopLevelFiresThisScene).
unsigned g_mainCamFires = 0;

// Where the drawn set goes. Registered once at init; null just means the feed
// is dropped, which degrades to the frustum fallback rather than misbehaving.
FnVisibleGeom g_visibleGeomCb = nullptr;

// Roots latched once per frame by beginFrame(). Null means UNKNOWN, and the
// coverage classifier must treat unknown as "not safe to skip" — a null sky root
// must never let sky leaves fall into AlphaCovered.
NI::Node*     g_landscapeRoot = nullptr;
NI::AVObject* g_skyRoot       = nullptr;
NI::AVObject* g_rainRoot      = nullptr;
NI::AVObject* g_snowRoot      = nullptr;
NI::AVObject* g_stormRoot     = nullptr;

Stats g_stats{};

// Leaves collected by the deferring traversal, in engine draw order.
//
// A flat pointer vector rather than msoc's {shape, camera} pairs: one classify
// pass has exactly one camera, so carrying it per leaf was redundant. The payoff
// is that this vector IS the feed's wire format — DistantLand::updateVisibleSet
// reinterprets the pointers as cache keys — so the sink gets g_pending.data()
// with no copy, where msoc had to build a parallel g_visCallbackNodes array.
std::vector<NI::AVObject*> g_pending;
NI::Camera* g_deferCamera = nullptr;

// Memoised coverage class per leaf.
//
// HAZARD: the key is a raw NiTriShape*, and MW recycles those aggressively — the
// geometry cache was already burned by exactly this (a recycled pointer serving a
// different mesh's data). Coverage is a function of the leaf's properties and its
// root membership, both stable while the cell is loaded, so the cache is wiped on
// cell change and never trusted across one. Do not "optimise" that wipe away.
std::unordered_map<const void*, uint8_t> g_coverageCache;

// ---------------------------------------------------------------------------
// NI::Camera culling-plane accessors
// ---------------------------------------------------------------------------
// countCullingPlanes is always 6 for the world camera, and the
// usedCullingPlanesBitfield sits immediately past the inline cullingPlanes[6]
// array — upstream SharedSE labels that field unknown_*, so it is reached by
// offset from a field that IS named rather than by a fresh magic constant.

constexpr int kCullingPlanes = 6;

inline const NI::Point4* cullingPlane(NI::Camera* cam, int i) {
    return &cam->cullingPlanes[i];
}

inline uint32_t* usedPlanesMask(NI::Camera* cam) {
    auto* base = reinterpret_cast<char*>(&cam->cullingPlanes[0]);
    return reinterpret_cast<uint32_t*>(base + sizeof(NI::Point4) * kCullingPlanes);
}

// ---------------------------------------------------------------------------
// Coverage classification
// ---------------------------------------------------------------------------
// Ported from msoc OccluderClassify.cpp. Every rule below is a claim about what
// MGE's proxy rejects and what the Forge host redraws; the two must stay in sync
// or a skipped leaf becomes a hole. Rules, with the reason each exists:
//
// OpaqueRedundant (skippable under kOwnedOpaque) — byte-identical to MGE's own
// isCoveredOpaque + isLandSplat reject gate (distantland.cpp:1276):
//   alpha blend ON   -> not opaque-covered; falls through to the alpha rules.
//                       (Alpha TEST alone stays covered-opaque -> skippable.)
//   z-write OFF      -> not covered-opaque -> keep. Property flag bit 1 = write
//                       enable; an absent ZBuffer property means engine default
//                       z-write ON.
//   no base texture  -> stage0 samples no texture -> not covered -> keep.
//   any decal map    -> decal passes are separate blended DIPs the proxy does
//                       NOT reject -> keep the whole leaf.
// Dark/detail/glow stay skippable: on 8-stage hardware MW folds them into the
// single covered-opaque pass.
//
// AlphaCovered (skippable under kOwnedAlpha) — strict subset of the host's alpha
// pass predicate (buildGeometryDrawLists: !isSky && !isSkinned && d3dTexture &&
// !isLandscape && blendEnable && !(dark|detail|glow)):
//   dark/detail/glow -> host skips multi-map blends -> keep.
//   decal map        -> host draws base-only, decal DIPs are separate -> keep.
//   material Power 99999 -> MW's water plane, whose DIP is what TRIGGERS the
//                       water stage. Skipping it kills water outright.
//   skinInstance     -> host routes skinned to the skinned pipeline -> keep.
//   NiParticles      -> AT3 set; MW still draws particles/VFX -> keep.
//   under weather    -> rain/snow are TriShapes (sgTriRain, snowflake clones),
//                       NOT particles, so the particle rule misses them, and the
//                       cache never walks those subtrees. They ride the AT3
//                       capture, which needs the engine display -> keep.
//   under sky root   -> MW's scene-0 sky/cloud DIPs feed the sky capture -> keep.
//   under landscape  -> host draws landscape via the static path; splat DIP
//                       rejection is the proxy's job -> keep.
// The blended walk cannot early-out: sky/landscape membership needs the full
// parent chain, because those roots sit ABOVE the property-carrying nodes.

Coverage classifyUncached(NI::AVObject* obj) {
    bool alphaResolved = false, zResolved = false, texResolved = false, matResolved = false;
    bool blended = false, zWriteOff = false;
    bool hasBaseTexture = false, hasDecal = false, hasMultiMap = false;
    bool waterMaterial = false;
    bool underSky = false, underLandscape = false, underWeather = false;

    for (NI::AVObject* cur = obj; cur; cur = cur->parentNode) {
        if (g_skyRoot && cur == g_skyRoot) underSky = true;
        if (g_landscapeRoot && cur == reinterpret_cast<NI::AVObject*>(g_landscapeRoot)) underLandscape = true;
        if ((g_rainRoot  && cur == g_rainRoot)
         || (g_snowRoot  && cur == g_snowRoot)
         || (g_stormRoot && cur == g_stormRoot)) underWeather = true;

        for (auto* node = &cur->propertyNode; node && node->data; node = node->next) {
            const auto type = node->data->getType();
            if (!alphaResolved && type == NI::PropertyType::Alpha) {
                blended = (node->data->flags & NI::AlphaProperty::ALPHA_MASK) != 0;
                alphaResolved = true;
            } else if (!zResolved && type == NI::PropertyType::ZBuffer) {
                zWriteOff = (node->data->flags & 0x2) == 0;
                zResolved = true;
            } else if (!texResolved && type == NI::PropertyType::Texturing) {
                auto* tex = static_cast<NI::TexturingProperty*>(node->data);
                // The map getters index maps[] unguarded (TArray::at throws), so
                // check size before every slot. getDecalCount guards internally.
                using MapType = NI::TexturingProperty::MapType;
                const auto mapCount = tex->maps.size();
                auto mapPresent = [&](MapType t) {
                    if (mapCount <= size_t(t)) return false;
                    const auto* m = tex->maps[size_t(t)];
                    return m && m->texture;
                };
                hasBaseTexture = mapPresent(MapType::BASE);
                hasMultiMap = mapPresent(MapType::DARK) || mapPresent(MapType::DETAIL)
                           || mapPresent(MapType::GLOW);
                hasDecal = tex->getDecalCount() > 0;
                texResolved = true;
            } else if (!matResolved && type == NI::PropertyType::Material) {
                waterMaterial = static_cast<NI::MaterialProperty*>(node->data)->shininess == 99999.0f;
                matResolved = true;
            }
        }

        // Opaque early-out: once the three opaque-relevant properties are resolved
        // and the leaf is NOT blended, no ancestor root can change the verdict.
        // Blended leaves must walk to the root for sky/landscape membership.
        if (alphaResolved && zResolved && texResolved && !blended) break;
    }

    if (blended) {
        if (underSky || underLandscape || underWeather || waterMaterial
            || hasMultiMap || hasDecal || !hasBaseTexture) {
            return Coverage::EngineDraws;
        }
        // Deferred leaves are NiTriBasedGeom by construction (see cullShowBody).
        auto* geom = static_cast<NI::Geometry*>(obj);
        if (geom->skinInstance) return Coverage::EngineDraws;
        if (obj->isInstanceOfType(NI::RTTIStaticPtr::NiParticles)) return Coverage::EngineDraws;
        return Coverage::AlphaCovered;
    }
    if (zWriteOff) return Coverage::EngineDraws;
    return (hasBaseTexture && !hasDecal) ? Coverage::OpaqueRedundant : Coverage::EngineDraws;
}

// ---------------------------------------------------------------------------
// The traversal
// ---------------------------------------------------------------------------

// Unflips exactly the culling-plane ignore bits this invocation set, mirroring the
// engine's LABEL_10 cleanup. msoc calls its equivalent by hand on each of four
// exit paths; this is RAII instead, because the prologue patch leaves no original
// to fall back to and a single missed unflip corrupts culling state for every
// later traversal in the frame. Equivalent by construction: restoring with no bits
// set is a no-op, which is exactly what the pre-frustum-test early return wants.
class IgnoreBitGuard {
    uint32_t* mask;
    uint32_t  setBits[4] = {0, 0, 0, 0};

public:
    explicit IgnoreBitGuard(uint32_t* m) : mask(m) {}
    ~IgnoreBitGuard() {
        for (int j = 0; j < kCullingPlanes; ++j) {
            const uint32_t jbit = 1u << (j & 0x1F);
            if (jbit & setBits[j >> 5]) {
                mask[j >> 5] &= ~jbit;
            }
        }
    }
    IgnoreBitGuard(const IgnoreBitGuard&) = delete;
    IgnoreBitGuard& operator=(const IgnoreBitGuard&) = delete;

    void flip(int word, uint32_t bit) {
        mask[word] |= bit;
        setBits[word] |= bit;
    }
};

// Depth bookkeeping for every entry; the collecting flag for the classify pass
// only. Both are RAII for the same reason the ignore bits are: there is no
// original CullShow left to fall back to if one of them leaks.
struct DepthGuard {
    DepthGuard()  { ++g_callDepth; }
    ~DepthGuard() { --g_callDepth; }
};

struct DeferGuard {
    DeferGuard(NI::Camera* camera) { g_deferCamera = camera; g_deferring = true; }
    ~DeferGuard() { g_deferring = false; }
};

// ---------------------------------------------------------------------------
// Top-level root identity
// ---------------------------------------------------------------------------

// Is `candidate` the engine's world render root — i.e. on the live parent chain
// of worldObjectRoot? Walks the FRESH chain from DataHandler (always current)
// and pointer-compares; `candidate` is never dereferenced, so a stale capture
// answers false instead of faulting.
bool isRenderRootOf(NI::AVObject* candidate, void* dh) {
    NI::AVObject* p = MGE::DataHandlerView::worldObjectRoot(dh);
    for (int i = 0; i < 32 && p; ++i) {
        if (p == candidate) return true;
        p = p->parentNode;
    }
    return false;
}

// Outer CullShow entry outside the classify pass. Two jobs, and it is the only
// place either happens:
//
//  * Mode B — hand the engine's own top-level pass the leaves classifyNow()
//    already collected, instead of letting it re-traverse. Matched on ROOT
//    IDENTITY (self == g_topLevelRoot), which is strictly tighter than msoc's
//    "first main-camera fire this scene": a sky or first-person pass that
//    happened to share the world camera cannot consume the world's leaves.
//  * Capture the root for the next frame's classify, on the engine's own pass.
//
// Returns true when the entry was fully handled and must not traverse.
bool handleTopLevelEntry(NI::AVObject* self, NI::Camera* camera) {
    if (camera != MGE::WorldControllerView::worldCamera()) return false;
    ++g_mainCamFires;

    if (g_displayPending) {
        if (self != g_topLevelRoot) return false;   // a different main-camera pass
        g_displayPending = false;
        displayDeferred();
        return true;
    }

    // Capture only while we have no root. Re-capturing every scene (msoc's rule)
    // risks latching a NARROWER root mid-session — a later main-camera pass on
    // some subtree also sits on worldObjectRoot's chain — and once that happens
    // the engine's real top-level pass stops matching, so it traverses and draws
    // the world while the narrow pass ALSO displays our collected leaves. Double
    // draw. classifyNow() is the sole invalidator: it nulls the root the moment
    // its staleness check fails, and the next pass through here re-captures.
    if (!g_topLevelRoot) {
        void* dh = MGE::SceneGraph::getDataHandler();
        if (dh && isRenderRootOf(self, dh)) {
            g_topLevelRoot = self;
            NI::AVObject* wor = MGE::DataHandlerView::worldObjectRoot(dh);
            LOG::logline(">> [enginecull] captured top-level render root %p (worldObjectRoot=%p)%s",
                         (void*)self, (void*)wor,
                         (self == wor) ? " - IS worldObjectRoot, expected an ancestor" : "");
        }
    }
    return false;
}

void fireVisibleGeomFeed() {
    g_stats.fed = static_cast<uint32_t>(g_pending.size());
    if (!g_visibleGeomCb) return;
    static_assert(sizeof(NI::AVObject*) == sizeof(void*), "feed is a flat pointer array");
    g_visibleGeomCb(reinterpret_cast<void* const*>(g_pending.data()), nullptr,
                    static_cast<int>(g_pending.size()));
}

}  // namespace

// The engine-faithful body. Every one of NiAVObject::CullShow's seven direct
// callers lands here once installed, so this must be correct for the sky pass,
// the first-person subtree and NiCamera::Click, not just the main world scene.
// With g_deferring false it is behaviourally identical to the engine's.
void __fastcall cullShowBody(NI::AVObject* self, void* /*edx*/, NI::Camera* camera) {
    DepthGuard depth;

    // Outer entry, and not the classify pass driving itself: this is one of the
    // engine's own seven call sites, so run the phase machine before traversing.
    if (g_callDepth == 1 && !g_deferring && handleTopLevelEntry(self, camera)) {
        return;
    }

    if (g_deferring) ++g_stats.recursiveCalls;

    if (self->getAppCulled()) {
        if (g_deferring) ++g_stats.appCulled;
        return;
    }

    // Hierarchical frustum test, mirroring the engine's loop. A plane the bound
    // is entirely inside gets marked ignorable for the whole subtree below (and
    // unflipped on the way out by the guard).
    uint32_t* mask = usedPlanesMask(camera);
    IgnoreBitGuard guard(mask);

    const float boundRadius = self->worldBoundRadius;
    for (int i = kCullingPlanes - 1; i >= 0; --i) {
        const uint32_t bit = 1u << (i & 0x1F);
        const int word = i >> 5;
        if ((bit & mask[word]) != 0) continue;

        const auto* plane = cullingPlane(camera, i);
        const float d = plane->x * self->worldBoundOrigin.x
                      + plane->y * self->worldBoundOrigin.y
                      + plane->z * self->worldBoundOrigin.z
                      - plane->w;
        if (d <= -boundRadius) {
            if (g_deferring) ++g_stats.frustumCulled;
            return;
        }
        if (d >= boundRadius) {
            guard.flip(word, bit);
        }
    }

    if (g_deferring) {
        // Geometry leaves are collected, not displayed. Interior NiNodes fall
        // through to display(), which recurses into children and re-enters this
        // body through the detour — that recursion IS the traversal.
        if (self->isInstanceOfType(NI::RTTIStaticPtr::NiTriBasedGeom)) {
            g_pending.push_back(self);
            ++g_stats.deferred;
            return;
        }
    }

    self->vTable.asAVObject->display(self, camera);
}

void beginFrame(int ownedFlags) {
    // Previous frame's numbers, published before they are cleared. This is the
    // only place that always runs, so it is also where a classify with no
    // matching display surfaces.
    if (g_displayPending) ++g_stats.missedDisplays;
    if (Configuration.LogDistantPipeline) {
        static unsigned s_n = 0;
        static uint32_t s_missed = 0;
        s_missed += g_stats.missedDisplays;
        if (++s_n % 300 == 0) {
            LOG::logline("-- [enginecull] deferred=%u fed=%u displayed=%u skipOpaque=%u skipAlpha=%u "
                         "nodes=%u appCulled=%u frustumCulled=%u missedDisplays=%u/300",
                         g_stats.deferred, g_stats.fed, g_stats.displayed,
                         g_stats.ownedOpaqueSkipped, g_stats.ownedAlphaSkipped,
                         g_stats.recursiveCalls, g_stats.appCulled, g_stats.frustumCulled,
                         s_missed);
            s_missed = 0;
        }
    }

    // Re-arm the classify/display pair. Dropping leaves the engine never drew is
    // deliberate: they are raw NiTriShape* whose only liveness guarantee was
    // being current-frame, so carrying them forward is exactly the use-after-free
    // this whole design exists to avoid.
    g_displayPending = false;
    g_pending.clear();
    g_mainCamFires = 0;

    g_skipOwnedOpaque = (ownedFlags & kOwnedOpaque) != 0;
    g_skipOwnedAlpha  = (ownedFlags & kOwnedAlpha) != 0;

    void* dh = MGE::SceneGraph::getDataHandler();

    // Coverage is stable while a cell is loaded (it is a function of the leaf's
    // properties and its root membership) but the memo is keyed on raw
    // NiTriShape*, which MW recycles. Wipe on cell change — same lifetime msoc
    // gave its equivalent cache. Done here rather than through a cell-change hook
    // because no such hook exists in mgecore and inventing one for this would be
    // more machinery than the check.
    static const void* s_lastCell = nullptr;
    const void* cell = dh ? MGE::DataHandlerView::currentCell(dh) : nullptr;
    if (cell != s_lastCell) {
        s_lastCell = cell;
        invalidateCoverageCache();
    }

    g_landscapeRoot = dh ? MGE::DataHandlerView::worldLandscapeRoot(dh) : nullptr;

    g_skyRoot   = reinterpret_cast<NI::AVObject*>(MGE::WorldControllerView::sgSkyRoot());
    g_rainRoot  = reinterpret_cast<NI::AVObject*>(MGE::WorldControllerView::sgRainRoot());
    g_snowRoot  = reinterpret_cast<NI::AVObject*>(MGE::WorldControllerView::sgSnowRoot());
    g_stormRoot = reinterpret_cast<NI::AVObject*>(MGE::WorldControllerView::sgStormRoot());

    g_stats = Stats{};
}

Coverage classifyCoverage(NI::AVObject* obj) {
    auto it = g_coverageCache.find(obj);
    if (it != g_coverageCache.end()) {
        return static_cast<Coverage>(it->second);
    }
    const Coverage c = classifyUncached(obj);
    g_coverageCache.emplace(obj, static_cast<uint8_t>(c));
    return c;
}

void invalidateCoverageCache() {
    g_coverageCache.clear();
}

// Display the collected leaves, dropping the ones the Forge side already covers.
// This is msoc's buildDisplayPartition + displaySurvivors collapsed into one pass:
// the two-phase split existed so an occlusion classify could run between them, and
// there is no occlusion classify any more.
//
// The proxy's per-DIP reject gate remains the correctness belt — a leaf we fail to
// skip costs a rejected DIP, a leaf we skip wrongly is a missing object. That
// asymmetry is why every doubtful case in classifyUncached returns EngineDraws.
void displayDeferred() {
    const bool skipping = g_skipOwnedOpaque || g_skipOwnedAlpha;
    for (NI::AVObject* shape : g_pending) {
        if (skipping) {
            const Coverage c = classifyCoverage(shape);
            if (g_skipOwnedOpaque && c == Coverage::OpaqueRedundant) {
                ++g_stats.ownedOpaqueSkipped;
                continue;
            }
            if (g_skipOwnedAlpha && c == Coverage::AlphaCovered) {
                ++g_stats.ownedAlphaSkipped;
                continue;
            }
        }
        ++g_stats.displayed;
        shape->vTable.asAVObject->display(shape, g_deferCamera);
    }
    g_pending.clear();
}

// The early classify: run the traversal at BeginScene(0) instead of waiting for
// the engine, so MGE's produce kickoff gets a CURRENT-frame drawn set.
//
// Every guard below returns rather than degrading, and a return means MGE falls
// back to its own frustum cull for the frame — correct, just ~7 ms slower. The
// one thing that must never happen is collecting leaves without arming the
// display, so g_displayPending is set last, only on the success path.
int classifyNow(void* cameraIn) {
    if (!g_installed)     return 12;
    if (g_deferring)      return 1;
    if (!g_topLevelRoot)  return 3;   // engine hasn't rendered once yet
    if (g_displayPending) return 4;
    if (g_mainCamFires)   return 5;   // engine already ran its pass this scene

    NI::Camera* mainCamera = MGE::WorldControllerView::worldCamera();
    if (!mainCamera) return 6;
    // Menu parity with msoc, which skipped for a threadpool reason we no longer
    // have. Kept for D4 so the A/B compares like with like; menus fall back to
    // the frustum walk exactly as they do today. Lifting it is a separate,
    // measurable change.
    if (MGE::WorldControllerView::menuMode()) return 7;
    auto* camera = static_cast<NI::Camera*>(cameraIn);
    if (camera && camera != mainCamera) return 8;

    void* dh = MGE::SceneGraph::getDataHandler();
    if (!dh || !MGE::DataHandlerView::worldObjectRoot(dh)) return 9;

    // Staleness: a worldspace change frees the captured root. Null it so the
    // engine's own pass re-captures this frame; we never dereference it.
    if (!isRenderRootOf(g_topLevelRoot, dh)) {
        g_topLevelRoot = nullptr;
        return 11;
    }

    g_pending.clear();
    {
        DeferGuard defer(mainCamera);
        cullShowBody(g_topLevelRoot, nullptr, mainCamera);
    }

    fireVisibleGeomFeed();
    g_displayPending = true;
    return 0;
}

void setVisibleGeomCallback(FnVisibleGeom cb) {
    g_visibleGeomCb = cb;
}

const Stats& lastFrameStats() {
    return g_stats;
}

bool isInstalled() {
    return g_installed;
}

bool install() {
    if (g_installed) return true;
    if (!Configuration.ForgeEngineCullTakeover) return false;

    // Two detours on one 5-byte prologue is unrecoverable: whoever patches second
    // overwrites the first's jump and the first's target is orphaned. msoc arms
    // its detour from luaopen_msoc, so the module being loaded is not by itself
    // proof — check the prologue bytes, which is the fact that actually matters.
    const auto* prologue = reinterpret_cast<const unsigned char*>(kCullShowAddr);
    if (prologue[0] == 0xE9 || prologue[0] == 0xE8) {
        LOG::logline("!! [enginecull] CullShow at 0x%X is already detoured (lead byte 0x%02X) - "
                     "refusing to install. Unload the msoc mod to use MGE's own traversal.",
                     kCullShowAddr, prologue[0]);
        return false;
    }
    // Belt for the other ordering: msoc.dll present but not yet armed (it arms
    // from luaopen_msoc, which may run after us). Patching now would let msoc
    // overwrite our jump later and silently orphan this traversal.
    if (GetModuleHandleA("msoc.dll") != nullptr) {
        LOG::logline("!! [enginecull] msoc.dll is loaded - refusing to install, its detour may arm later. "
                     "Disable the msoc lua mod to hand the traversal to MGE.");
        return false;
    }

    // 5-byte prologue overwrite, NO trampoline: the original CullShow is gone from
    // here on and cullShowBody is the whole implementation for all seven callers.
    // There is deliberately no uninstall — restoring the prologue mid-frame, with
    // a traversal possibly on the stack, is far more dangerous than staying
    // patched for the process lifetime.
    se::memory::genJumpUnprotected(kCullShowAddr, reinterpret_cast<DWORD>(&cullShowBody));
    g_installed = true;
    LOG::logline(">> [enginecull] CullShow detour installed at 0x%X - MGE owns the traversal "
                 "and the discovery feed. First frame runs vanilla (no root captured yet), "
                 "then the early classify takes over.", kCullShowAddr);
    return true;
}

}  // namespace MGE::EngineCull
