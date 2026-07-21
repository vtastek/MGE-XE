#include "mge_se_prelude.h"

#include "NIAmbientLight.h"
#include "NIAVObject.h"
#include "NIGeometry.h"
#include "NIGeometryData.h"
#include "NINode.h"
#include "NISwitchNode.h"
#include "NIBillboardNode.h"
#include "NICamera.h"
#include "NIProperty.h"
#include "NIRTTIDefines.h"
#include "NISourceTexture.h"
#include "NIDX8TextureData.h"
#include "NITriBasedGeometry.h"
#include "NIParticles.h"
#include "NIParticleSystemController.h"
#include "NIColor.h"
#include "NITriBasedGeometryData.h"
#include "NISkinInstance.h"
#include "NIUVController.h"

#include "configuration.h"
#include "datahandler_view.h"
#include "mge_tracy.h"
#include "mwbridge.h"
#include "proxydx/d3d8texture.h"
#include "proxydx/devicelock.h"
#include "scenegraph.h"
#include "scenegraph_geometry_cache.h"
#include "renderprocess.h"
#include "distantland.h"   // DistantLand::mwWorldSuppress (MW-ONLY-UI suppression level)
#include "ipc/geomwire.h"
#include "support/log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace MGE::GeometryCache {

    namespace {

        IDirect3DDevice9* g_device          = nullptr;
        IDirect3DVertexDeclaration9* g_skinnedDecl = nullptr;
        uint64_t          g_frame          = 0;
        // Set while walking the landscape (terrain) root so visitGeometry can tag
        // entries (CachedGeometry::isLandscape). Terrain is excluded from the
        // cache color pass — it stays on the distant-land path.
        bool              g_walkingLandscape = false;
        // Set while walking worldPickObjectRoot so visitGeometry can tag entries
        // (CachedGeometry::isPickRoot) — that root is outside the engine's world-camera
        // occlusion classify; the Stage 2 engine-set cull keeps these via frustum.
        bool              g_walkingPick = false;
        // Set while walking skyRoot (SK1 sky takeover) so visitGeometry tags entries
        // (CachedGeometry::isSky) and forces a per-frame re-upload — the dome's vertex-colour
        // gradient changes every frame (sun angle / weather) without bumping revisionID.
        bool              g_walkingSky = false;
        // Set while walking the WorldController armCamera root (FP1a first-person
        // takeover) so visitGeometry tags entries (CachedGeometry::isFP). The FP walk
        // is exempt from the active-cell gate (the arm subtree rides the camera) and
        // bypasses the ROOT's own app-cull flag (FP1b forces it culled to suppress
        // MW's own arm draws while the capture must keep running).
        bool              g_walkingFP = false;
        // FP particle systems (torch flame, enchant sparks) met during the FP walk. NiParticles
        // reach visitGeometry (they ARE NiTriBasedGeom in MW RTTI) but capture only as degenerate
        // particle-center meshes; collected here instead so buildFPParticleQuads can billboard
        // them against the arm camera. Cleared at each FP walk start; pointers are live only for
        // the remainder of the frame's build. See tasks/forge-fp-particles.md.
        std::vector<NI::Particles*> g_fpParticleSystems;
        // Billboarded FP-particle quads built by buildFPParticleQuads, shipped by the client's
        // buildFPFrame as FP captured-alpha geometry. Reused frame-to-frame. g_fpPartRecs is one
        // record per particle system (its vertex/index range + texture + blend).
        std::vector<IPC::GeomVertexWire>  g_fpPartVerts;
        std::vector<std::uint16_t>        g_fpPartIndices;
        std::vector<FPParticleRec>        g_fpPartRecs;
        // SK2: monotonic counter assigned to each isSky entry's skyOrder during the skyRoot
        // walk (reset to 0 at the start of each sky walk). Encodes back-to-front subtree order
        // so the Forge sky pass can sort its alpha-blended draws (dome → stars → sun → moons).
        uint16_t          g_skyVisitCounter = 0;

        // NiAlphaProperty blend-function index (Gamebryo order) -> D3DBLEND. Defined with the
        // moon support below; forward-declared so extractMaterial can translate sky blend modes.
        D3DBLEND niBlendToD3D(unsigned int ni);
        uint32_t          g_uploadedThisFrame  = 0;
        uint64_t          g_uploadedInterval   = 0; // cumulative over log interval
        uint32_t          g_walkSerial         = 0; // ++ per onFrameReady (reship-streak probe)
        // Phase 0 diagnostic: null-bone influence accounting (the suspected NPC
        // "explosion" source — a null SkinInstance::bones[b] currently falls back
        // to identity, parking influenced verts at the model/cell origin).
        uint32_t          g_nullBoneHitsInterval  = 0; // total null bones seen / interval
        uint32_t          g_nullBonePartsInterval = 0; // distinct parts with >=1 null bone
        const char*       g_nullBoneSampleTex     = nullptr; // a sample part's texture

        std::unordered_map<uint32_t, CachedGeometry>      g_cache;
        // Offscreen shadow-caster candidate keys — the subset of g_cache whose entry passes the
        // pre-distance mover filter (skinned / multimap head / rigid LIVE). Maintained incrementally
        // (updateMoverMembership at capture/reclassify; erased at every g_cache erase/clear) so the
        // Forge feed's offscreen re-emit loop iterates this instead of scanning the whole cache each
        // frame. Exposed via moverCandidates(). Invariant: g_moverCandidates ⊆ keys(g_cache).
        std::unordered_set<uint32_t>                      g_moverCandidates;
        // STRONG reference to every cached NiTriShape, keyed exactly like g_cache.
        //
        // The cache KEY IS THE RAW ADDRESS of the shape, and ensureLive() dereferences that address
        // (geom->getModelData()) BEFORE it can validate anything — so if the engine freed the shape
        // while we still hold the key, that deref is a use-after-free. It is not defendable at the
        // deref site: you cannot ask a dangling pointer whether it is dangling. (Crash 2026-07-13:
        // loading a save tore down the scene; the offscreen near-actor re-emit path ensureLive()d the
        // now-freed keys still sitting in the cache. The old guard reasoned "near ⇒ alive", which is a
        // proximity argument, not a liveness one — a teardown frees near and far alike.)
        //
        // So make the pointer genuinely valid instead: hold an actual engine reference for exactly as
        // long as we cache the key. NI::Pointer is RAII (claim on assign, DecRef in the dtor), so the
        // shape cannot be destroyed under us, and every later deref of a cached key is sound BY
        // CONSTRUCTION. The ref is taken in visitGeometry — the one capture point, reached only from
        // the walk or ensureLive's lazy capture, where the shape is provably alive because we are
        // reading it right then.
        //
        // It also retires the recycled-address hazard: an address cannot be handed to a NEW shape
        // while we still own a reference to the old one. The dataPtr identity guards stay as
        // belt-and-braces (a live setModelData swap still needs them).
        std::unordered_map<uint32_t, NI::Pointer<NI::TriBasedGeometry>> g_geomRefs;
        // Keys evicted by the sweep since the last drain (object left the world within a cell).
        // The Forge feed drains these each frame (takeEvictedKeys) to release the matching host
        // mesh slot so its shadow-caster record stops ghosting. Only the "genuinely gone" sweep
        // eviction feeds this — NOT the identity-mismatch rebuild (that reuses the same key/slot).
        std::vector<uint32_t>                             g_evictedKeys;
        // Character-subtree verdict per NiNode* (walk()'s "does any direct child carry
        // a skin" look-ahead). The scan is O(children) RTTI checks per node PER FRAME
        // and character assemblies essentially never change, so the verdict is computed
        // once on first sight and cached. Cleared at every eviction sweep, which bounds
        // both staleness (a node GAINING a skinned child later) and pointer recycling
        // (freed NiNode address reused) to kEvictSweepInterval frames — and the only
        // consumer of the flag is dynamicHint, a distance-fade/VB heuristic that
        // self-heals via the per-frame transform compare (movement forces hint=4).
        std::unordered_map<uint32_t, bool>                g_charNodeVerdict;
        // Deferred eviction: entries not visited by the walk used to be evicted EVERY
        // frame — a full traversal of the (scattered, ~10k-entry) map per frame just to
        // find stale entries. Stale entries are harmless between sweeps: nothing draws
        // them (all draw paths consume current-frame visible sets or filter on
        // lastFrame == g_frame), they only hold memory/VBs a little longer. The sweep
        // now runs every kEvictSweepInterval frames.
        constexpr uint64_t kEvictSweepInterval = 30;
        // Parent-chain eviction. NOTE: this replaced a REFCOUNT test that did not work — recorded
        // here because the failure is the interesting part. The refcount premise was "a parented
        // shape is held by its parent's child list, so unlinking leaves us the sole holder at
        // refCount == 1". In practice MW keeps a SECOND reference to a removed object's shape
        // (rc stayed 2 on entries the walk proved gone), so the signal never once fired: 20 sweeps
        // logged `evicted=0 byRef=0` while `walk=GONE ref=ALIVE` ran ~344/sweep. Eviction silently
        // degraded to the kFarKeepFrames age rule alone, which is the "picked-up clutter keeps its
        // shadow for ~3.6s, but re-appears instantly" asymmetry, and left ~14k entries (each
        // holding a strong NI::Pointer) resident far longer than the walk rule did.
        //
        // Reachability is the property we actually care about, and it is directly observable:
        // climb NIAVObject::parentNode (0x18) and see whether the chain still terminates at one of
        // MW's live roots. A lingering reference cannot mask that. O(depth) — a handful of derefs
        // per entry per sweep, nothing like a full walk, so the spike stays dead.
        //
        // CAVEATS, both real:
        //  - parentNode is a RAW back-pointer. Sound only under the NI convention that a dying
        //    parent detaches its children (nulling the field); our strong ref protects the shape
        //    itself, NOT its ancestors.
        //  - Sky entries hang off skyRoot, which is found on demand (findSkyRoot) and not tracked
        //    here, so their chain would end at an "unknown" root and read as GONE. isSky entries
        //    are therefore excluded and fall back to the age rule.
        // The kEvictValidateSweeps window below cross-checks every verdict against the walk oracle;
        // `walk=KEEP parent=GONE` is the dangerous direction and must stay 0.
        //
        // The walk-based "visitable but unvisited = gone" rule is precise but costs a FULL graph
        // walk on every sweep frame (~6.6ms over 12.5k entries) — amortised in the [gc] average to
        // a harmless-looking 0.22ms, but on the wire a ~7ms main-thread hitch 3x/second (Tracy:
        // walksappear.png). The parent-chain test gets the same answer without traversing anything.
        //
        // g_evictByParentChain = A/B escape hatch back to the walk rule. While
        // g_evictSweepNo < kEvictValidateSweeps the sweep still forces the walk and CROSS-CHECKS the
        // two verdicts into [evict-cmp] lines; "walk=KEEP parent=GONE" is the dangerous direction
        // (would evict a live object) and must be zero before the validation window is retired.
        // Non-recovering freeze reported 2026-07-20: ROOT CAUSE FOUND, and it is NOT eviction. The
        // bisect (g_evictByParentChain off = original forced-walk path, no eviction bursts) STILL
        // froze, which exonerated the verdict change. The real culprit is the produce finish-gate:
        // waitFinishGate() was the one unbounded wait in the pipeline (every IPC wait caps at
        // MaxWait=60s), and drainPendingIfFull parks the WORKER on it mid-build — any missed
        // openFinishGate() wedged forever with zero log. Fixed in renderprocess.cpp by bounding that
        // wait (500ms) + logging "[gate] finish-gate WEDGED ... in '<who>'". The earlier "wrong
        // waiter drains the completion" theory was wrong: finishAndCopy already skips a second
        // renderSceneFinish on the early-finished path.
        // 2026-07-20: perf win RE-ENABLED (parentChain=true, forceWalk=false) with the ROOT-CAUSE
        // FIX finally in place. The two earlier freezes on this flip were BOTH downstream of the
        // eviction verdict, not the verdict itself:
        //   1. Raw flip froze on a DANGLING parentNode deref (we hold a ref on each cached leaf but
        //      not its parent; a freed detached subtree left stale heap under p->parentNode). FIXED
        //      by the isLiveNode() vtable guard in the sweep block (engine-informed, MWSE vtable
        //      addresses, no disasm) — the instrumented probe measured ~87% of the cache dangling
        //      (vtBad) yet countMs~1ms, proving the climb itself is cheap.
        //   2. Instrumented flip then froze on the RELEASE FLOOD: with no forced re-stamp, orphaned
        //      far cells age out en masse (512/sweep), and each host-slot release rode the BLOCKING
        //      present-time geometry RPC (measured 517 parts = 5.5ms, backlog blew the 32MB staging
        //      cap). FIXED in renderprocess.cpp: releases now queue and drain a bounded
        //      kMaxReleasesPerFlush per present (safe because host slots are monotonic/never reused),
        //      so mass eviction can never flood the RPC again. Watch >> [release] draining.
        // The instrumentation (isLiveNode guard, 250ms climb watchdog, >> [evict-climb] line) stays.
        // Escape hatch back to the known-good walk rule: false / true.
        bool               g_evictByParentChain = true;
        bool               g_evictForceWalk     = false;  // bisect knob; false = the perf win
        constexpr unsigned kEvictValidateSweeps = 20;
        // Ancestor-climb bound. MW object subtrees are shallow (single digits); exceeding this means
        // an unexpected/cyclic graph, so the verdict is "unknown" and the age rule decides.
        constexpr int      kMaxParentDepth      = 24;
        unsigned           g_evictSweepNo       = 0;
        unsigned           g_evictDisagreeWalkGone   = 0;  // walk=GONE parent=ALIVE (retention only)
        unsigned           g_evictDisagreeGraphGone  = 0;  // walk=KEEP parent=GONE (DANGEROUS)
        // W1.5 active-cell gate (see onFrameReady's header comment). g_gateThisFrame
        // arms the walk's subtree skip for the current frame only; g_gateEye/
        // g_gateRadius persist across ungated frames (menus, the renderDepth
        // fallback) so the eviction hysteresis below keeps working there — the
        // player can't move while the gate is down, so the last gated eye is valid.
        bool     g_gateThisFrame = false;
        float    g_gateEye[3]    = {};
        float    g_gateRadius    = 0.0f;
        // Eviction hysteresis: a stale entry BEYOND the gate radius was most likely
        // just skipped by the gate (not removed from the scene), so it is kept — no
        // re-capture/re-upload churn when the player returns. Bounded by age: far
        // entries untouched this long are evicted anyway (frees VBs of genuinely
        // unloaded far cells; they re-capture only if the area is ever revisited).
        constexpr uint64_t kFarKeepFrames = 600;
        // W3 live-read at build: on Forge-owned frames the per-frame refresh walk is
        // SKIPPED entirely — buildFrustumVisibleSet freshens exactly the classify-
        // visible keys via ensureLive() (live NiTriShape reads + lazy capture on first
        // sight) and calls ensureFullWalk() on frames with no classify. State below
        // tracks whether the walk ran this frame (eviction rule + deferred walk) and
        // the per-frame root pointers (capture context + deferred walk).
        uint64_t  g_walkRanFrame = 0;      // frame stamp of the last full refresh walk
        NI::Node* g_objRoot  = nullptr;
        NI::Node* g_pickRoot = nullptr;

        // MW-ONLY-UI world suppression: the level currently APPLIED to the engine's roots, so the
        // flags can be restored exactly (and only when they were ours to set). 0 = nothing culled.
        int g_worldSuppressApplied = 0;
        NI::Node* g_landRoot = nullptr;
        uint32_t  g_liveRefreshThisFrame = 0;
        uint32_t  g_liveCaptureThisFrame = 0;
        // Per-phase walk timing (QPC), logged every kGcHeartbeatFrames frames as
        // ">> [gc] ..." — the walk is on the dense-city serial chain, so its cost is
        // tracked with the same always-on heartbeat discipline as [hb]/[forge-hb].
        struct GcAccum { double obj, pick, land, sky, evict, visited, gateSkip, live, cap; uint64_t n; };
        GcAccum           g_gcAccum = {};
        constexpr uint64_t kGcHeartbeatFrames = 300;
        // Per-frame gate observability, accumulated into the [gc] heartbeat:
        // entries actually visited vs whole subtrees the gate skipped.
        uint32_t          g_visitedThisFrame  = 0;
        uint32_t          g_gateSkipsThisFrame = 0;
        // Reverse map GPU texture -> SourceTexture::fileName, for resolveTextureName().
        // AT3 captured-alpha consumer: populated INCREMENTALLY (never per-frame) from
        // extractMaterial (base/dark/detail/glow/overlay maps of every cached shape) AND from
        // walk()'s NiGeometry branch (NiParticles, which never reach extractMaterial). The
        // client's captureAlphaDraw resolves rs.texture (a proxy realTexture pointer, identical
        // to getDX9Texture) -> the source name -> a bindless slot. insert_or_assign so a
        // recycled NiTriShape*/GPU-texture pointer self-corrects to the current name.
        std::unordered_map<IDirect3DTexture9*, const char*> g_textureNameMap;

        void releaseEntry(CachedGeometry& e) {
            if (e.vb[0]) { e.vb[0]->Release(); e.vb[0] = nullptr; }
            if (e.vb[1]) { e.vb[1]->Release(); e.vb[1] = nullptr; }
            if (e.ib)    { e.ib->Release();    e.ib    = nullptr; }
            e.hostUploaded = false;   // buffers/uploads invalidated -> must re-process + re-ship
        }

        // The DX9 mirror VB/IB (e.vb/e.ib) is consumed by the legacy DX9 cache draws that are
        // STILL live A/B paths:
        //   - renderDepthFromCache  — runs when !forgeOwnsDepth (F7-off MGE-water A/B, or Forge off
        //                             entirely); also via ForgeNearDepthReplay + the UseRenderThread
        //                             depth job (both their own flags)
        //   - renderShadowFromCache — gated by the same forgeOwnsDepth expression as depth
        // forgeOwnsDepth == ownsOpaqueWorld() && wantsWaterCapture() (renderdepth.cpp:338,
        // distantland.cpp:436). NUMPAD7 CACHE (rendercachedcolor / cacheOpaqueMode) is the
        // obsolete pre-DX12 opaque-cache attempt — no longer maintained, so it is deliberately NOT
        // a mirror consumer here (it renders empty; superseded by the Forge takeover). In default
        // Forge play (F11 + F7 on) forgeOwnsDepth is true and both explicit flags are off, so the
        // mirror is dead weight — skip it so the produce/capture path is D3D9-free
        // (worker-relocatable). The host IPC capture (captureGeometry/…) is independent and always
        // runs. This predicate mirrors the live consumers' gates exactly, so a mirror is built iff
        // something will draw it; the content-identity gate rebuilds a missing mirror on demand
        // when a mode toggles on (no purge / no host re-ship needed).
        static bool needMirror() {
            const bool forgeOwnsDepth =
                RenderProcess::ownsOpaqueWorld() && RenderProcess::wantsWaterCapture();
            return Configuration.ForgeNearDepthReplay
                || Configuration.UseRenderThread
                || !forgeOwnsDepth;
        }

        IDirect3DTexture9* getDX9Texture(NI::Texture* tex) {
            if (!tex || !tex->rendererData) return nullptr;
            auto* srd = static_cast<NI::DX8SourceTextureData*>(
                static_cast<void*>(tex->rendererData));
            if (!srd->d3dTexture) return nullptr;
            return static_cast<ProxyTexture*>(srd->d3dTexture)->realTexture;
        }

        // AT3: register a confirmed-NiSourceTexture's GPU texture -> source fileName in the
        // reverse map, so the Forge alpha-capture path can resolve rs.texture (a proxy
        // realTexture pointer) to a bindless slot by name. No-op for non-SourceTextures /
        // unloaded (no rendererData) textures. Cheap: one hash insert per material extract.
        void registerTextureName(NI::Texture* tex) {
            if (!tex) return;
            if (!tex->isInstanceOfType(NI::RTTIStaticPtr::NiSourceTexture)) return;
            IDirect3DTexture9* d3d = getDX9Texture(tex);
            if (!d3d) return;
            const char* name = static_cast<NI::SourceTexture*>(tex)->fileName;
            if (!name) return;
            g_textureNameMap.insert_or_assign(d3d, name);
        }

        // Registration-only subtree sweep: map every NiGeometry's base-map GPU texture to its
        // source name, no capture, no cache writes. For the worldRoot siblings the walks never
        // visit (Precipitation Rain/Snow Root, Storm Root, WorldProjectileRoot, WorldSpellRoot,
        // WorldVFXRoot): their alpha-blended particle DIPs reach captureAlphaDraw with GPU
        // textures absent from g_textureNameMap, fell back to slot 0, and drew as opaque WHITE
        // quads (ashstorm/blizzard whiteout). Culled subtrees are swept too, so inactive
        // precipitation pre-registers before its storm starts.
        void registerSubtreeTextureNames(NI::AVObject* av) {
            if (!av) return;
            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiGeometry)) {
                auto* geom = static_cast<NI::Geometry*>(av);
                auto* ps = reinterpret_cast<NI::PropertyState*>(geom->propertyState);
                if (ps && ps->texture) {
                    const auto* baseMap = ps->texture->getBaseMap();
                    if (baseMap && baseMap->texture) {
                        registerTextureName(baseMap->texture.get());
                    }
                }
                return;
            }
            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) {
                auto* node = static_cast<NI::Node*>(av);
                const auto count = node->children.getEndIndex();
                for (size_t i = 0; i < count; ++i) {
                    registerSubtreeTextureNames(node->children.at(i).get());
                }
            }
        }

        // SK4: FNV-1a over the live vertex-colour array (PackedColor = 4 bytes/vert).
        // Sky shapes hash a few hundred bytes/frame — cheaper than any IPC round-trip.
        uint32_t hashVertexColors(const void* vcol, uint32_t vertexCount) {
            const auto* p = static_cast<const uint8_t*>(vcol);
            uint32_t h = 2166136261u;
            for (uint32_t i = 0; i < vertexCount * 4u; ++i) {
                h = (h ^ p[i]) * 16777619u;
            }
            return h;
        }

        // Per-walk copy of the scene-graph point-light snapshot, for computeEmissiveGain.
        // Copied (not read under the lock) because the geometry walk is ~1.5ms and would
        // stall the async scene-graph worker's swap, and because ensureLive's lazy capture
        // calls extractMaterial after onFrameReady has returned. One frame stale on the
        // async path — fixture light colours are static, so that never shows.
        std::vector<MGE::SceneGraph::PointLight> g_lightSnapshot;

        // Emissive fixtures carry HDR the 8-bit texture never held. A light's own emissive
        // surface is far brighter than the illumination it casts — a candle core is ~1000 nits —
        // so `emissive 1,1,1` ("show the texture unlit at full") lands ~two orders of magnitude
        // short of the authored intent. The engine is not at fault: all three independent FFE
        // implementations agree on lit = matDiffuse*d + matAmbient*a + matEmissive with no
        // multiplier, and the clipping happened at texture-export time, in ~1999. Restoring it
        // is our deliberate feature.
        //
        // The restored value is the emitter's RADIANCE, and radiance is flux over area:
        //
        //     emissive = authored(1.0) * kEmissiveFlux * lightColour / emissiveArea
        //
        // The light record supplies the flux; the MESH supplies the area. That split is the
        // whole point — it is why one constant covers every fixture class. A candle flame
        // concentrates a modest flux into a tiny billboard (huge radiance); a paper lantern
        // spreads the same kind of flux over a big diffuse sphere (small radiance). Nothing is
        // fitted per asset: feed it the mesh and the light and the answer falls out. Radius is
        // deliberately NOT an input — the same lantern NIF ships at radii 64..512, so radius and
        // emitter size are independent authored values.
        //
        // The texture then shapes the result for free: the frag already computes
        // c = albedo * lit, so a candle-flame texture's own falloff scales the radiance down
        // exactly where the artist painted it dark. (For fixtures whose texture is itself an
        // unwitting record of clip(k*L) — the lantern papers — this applies the texture's colour
        // a second time and reads slightly more saturated than the raw light. Deliberate: the
        // texture is a modulation here, not the colour source.)
        //
        // Applied only where the emission demonstrably CAME FROM a light: the shape must be
        // emissive AND own a light. Meshless-emissive assets (glowing mushrooms, the ampoule
        // pod) have no flux to divide and are deliberately left untouched.
        //
        // CALIBRATION: kEmissiveFlux = k_ref * area_ref.
        //
        // k_ref is deliberately BELOW the measured 4.5 (Light_paper_lantern_01's paper fits
        // clip(4.52*L), rms 0.063; the blue _02 fits 5.03). Reason: the colour chain is
        // B8G8R8A8_UNORM and tonemap()'s polynomial reaches 1.0 at c=2.2, so anything past that
        // pins to white. The reference paper's brightest texel clips once albedo_R*k*L_R > 2.2
        // => k > 2.29; at the authored 4.5 both R and G pin, R:G snaps to 1.00, and the lantern
        // reads hot CREAM instead of its light's orange. 2.2 is the largest boost that keeps the
        // reference fixture entirely under the knee, so its hue survives and can be matched
        // against. Restore k_ref to the measured 4.5 when the HDR/linear post lands — that is
        // what makes the authored value expressible.
        //
        // area_ref is the paper's emissive area, which the [emissive] logline prints; until a
        // lantern has actually been walked past in-game it is an ESTIMATE (a ~20cm paper sphere
        // ~= r7 units => 4*pi*r^2 ~= 600), so every fixture's absolute brightness scales with it.
        // The RATIOS between fixture classes are already correct — they come from the meshes, not
        // from this number. (Small-area emitters — candle flames — still blow past 2.2 and pin to
        // white by construction; only HDR fixes those.)
        constexpr float kEmissiveRefK    = 2.2f;    // LDR-limited; measured value is 4.5 (see above)
        constexpr float kEmissiveRefArea = 600.0f;  // ESTIMATE: paper-lantern emissive area (units^2)
        constexpr float kEmissiveFlux    = kEmissiveRefK * kEmissiveRefArea;

        // World-space emissive surface area: sum of triangle areas, scaled by the world
        // transform. Correct for both a lantern's sphere and a flame's flat billboard, which a
        // bound-radius sphere proxy would get wrong by ~4x in opposite directions. Runs once per
        // cache entry (alongside the VB upload's own full vertex walk), never per frame.
        float computeEmissiveArea(const NI::TriBasedGeometry* geom) {
            const auto* data = geom->getModelData().get();
            if (!data) return 0.0f;
            const auto* verts = data->vertex;
            const auto* tris  = data->getTriList();
            const uint32_t tc = static_cast<uint32_t>(data->getActiveTriangleCount());
            if (!verts || !tris || !tc) return 0.0f;

            double area = 0.0;
            for (uint32_t i = 0; i < tc; ++i) {
                const auto& a = verts[tris[i].vertices[0]];
                const auto& b = verts[tris[i].vertices[1]];
                const auto& c = verts[tris[i].vertices[2]];
                const float ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
                const float vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
                const float cx = uy * vz - uz * vy;
                const float cy = uz * vx - ux * vz;
                const float cz = ux * vy - uy * vx;
                area += 0.5 * std::sqrt(double(cx * cx + cy * cy + cz * cz));
            }
            const float s = geom->worldTransform.scale;   // model -> world: area scales by s^2
            return static_cast<float>(area) * s * s;
        }

        // The shape's OWN light = a point light whose world position lies inside the shape's own
        // world bound — a lantern's NiPointLight sits at the flame, inside its paper. Nearest
        // wins if several qualify. The snapshot is already filtered to lights the engine would
        // actually render (radius > 0 and affectedNodes non-empty — see scenegraph.cpp's walk),
        // so a logically-off lantern can't donate a gain.
        void computeEmissiveGain(CachedGeometry& e, const NI::TriBasedGeometry* geom) {
            if (e.matEmissive[0] <= 0.0f && e.matEmissive[1] <= 0.0f && e.matEmissive[2] <= 0.0f) {
                return;
            }
            const float r = geom->worldBoundRadius;
            if (!(r > 0.0f)) return;

            const MGE::SceneGraph::PointLight* own = nullptr;
            float bestD2 = r * r;   // doubles as the inside-the-bound threshold
            for (const auto& pl : g_lightSnapshot) {
                const float dx = pl.worldPos[0] - geom->worldBoundOrigin.x;
                const float dy = pl.worldPos[1] - geom->worldBoundOrigin.y;
                const float dz = pl.worldPos[2] - geom->worldBoundOrigin.z;
                const float d2 = dx * dx + dy * dy + dz * dz;
                if (d2 <= bestD2) { bestD2 = d2; own = &pl; }
            }
            if (!own) return;

            const float area = computeEmissiveArea(geom);
            if (!(area > 0.0f)) return;

            for (int i = 0; i < 3; ++i) {
                e.emissiveGain[i] = kEmissiveFlux * own->diffuse[i] / area;
            }

            if (Configuration.LogDistantPipeline) {
                static uint32_t s_logged = 0;
                if (s_logged < 16) {
                    ++s_logged;
                    LOG::logline(">> [emissive] fixture tex=%s emissive=(%.2f,%.2f,%.2f) light=(%.3f,%.3f,%.3f) "
                                 "d=%.1f r=%.1f area=%.1f gain=(%.2f,%.2f,%.2f)",
                                 e.textureName ? e.textureName : "(none)",
                                 e.matEmissive[0], e.matEmissive[1], e.matEmissive[2],
                                 own->diffuse[0], own->diffuse[1], own->diffuse[2],
                                 std::sqrt(bestD2), r, area,
                                 e.emissiveGain[0], e.emissiveGain[1], e.emissiveGain[2]);
                }
            }
        }

        void extractMaterial(CachedGeometry& e, NI::TriBasedGeometry* geom) {
            e.d3dTexture  = nullptr;
            e.d3dOverlay  = nullptr;
            e.d3dDark     = nullptr;
            e.d3dDetail   = nullptr;
            e.d3dGlow     = nullptr;
            e.baseUV = e.darkUV = e.detailUV = e.glowUV = 0;
            e.textureName = nullptr;
            e.overlayTextureName = nullptr;
            e.darkTextureName = e.detailTextureName = e.glowTextureName = nullptr;
            e.alphaRef    = 0.0f;
            e.alphaTest   = false;
            e.blendEnable = false;
            e.twoSided    = false;   // single-sided (CULL_BACK) unless NiStencilProperty DRAW_BOTH
            // SK1 sky: default to the standard transparency blend; overwritten below from the
            // NiAlphaProperty flags when present. Only consumed for isSky entries (the Forge
            // sky pass); opaque/alpha-test draws ignore these.
            e.srcBlend    = static_cast<unsigned char>(D3DBLEND_SRCALPHA);
            e.destBlend   = static_cast<unsigned char>(D3DBLEND_INVSRCALPHA);
            // Default material: white diffuse/ambient, no emissive (texture-only
            // opaque). Overwritten below when the shape carries a MaterialProperty.
            e.matDiffuse[0]  = e.matDiffuse[1]  = e.matDiffuse[2]  = e.matDiffuse[3]  = 1.0f;
            e.matAmbient[0]  = e.matAmbient[1]  = e.matAmbient[2]  = e.matAmbient[3]  = 1.0f;
            e.matEmissive[0] = e.matEmissive[1] = e.matEmissive[2] = e.matEmissive[3] = 0.0f;
            e.emissiveGain[0] = e.emissiveGain[1] = e.emissiveGain[2] = 1.0f;   // no boost
            e.vColSource = 0;   // SOURCE_IGNORE until a VertexColorProperty says otherwise

            auto* ps = reinterpret_cast<NI::PropertyState*>(geom->propertyState);
            if (!ps) return;

            if (ps->vertexColor) {
                e.vColSource = static_cast<uint8_t>(ps->vertexColor->source);  // 0 ignore, 1 emissive, 2 amb+diff
            }

            if (ps->material) {
                const auto* mp = ps->material;
                e.matDiffuse[0]  = mp->diffuse.r;  e.matDiffuse[1]  = mp->diffuse.g;
                e.matDiffuse[2]  = mp->diffuse.b;  e.matDiffuse[3]  = mp->alpha;
                e.matAmbient[0]  = mp->ambient.r;  e.matAmbient[1]  = mp->ambient.g;
                e.matAmbient[2]  = mp->ambient.b;  e.matAmbient[3]  = 1.0f;
                e.matEmissive[0] = mp->emissive.r; e.matEmissive[1] = mp->emissive.g;
                e.matEmissive[2] = mp->emissive.b; e.matEmissive[3] = 0.0f;
            }

            if (ps->alpha) {
                const auto* ap = ps->alpha;
                e.alphaTest   = (ap->flags & NI::AlphaProperty::TEST_ENABLE_MASK) != 0;
                e.blendEnable = (ap->flags & NI::AlphaProperty::ALPHA_MASK) != 0;
                e.alphaRef    = ap->alphaTestRef / 255.0f;
                // Sky blend factors (same translation the moon path uses).
                e.srcBlend  = static_cast<unsigned char>(niBlendToD3D(
                    (ap->flags & NI::AlphaProperty::SRC_BLEND_MASK)  >> NI::AlphaProperty::SRC_BLEND_POS));
                e.destBlend = static_cast<unsigned char>(niBlendToD3D(
                    (ap->flags & NI::AlphaProperty::DEST_BLEND_MASK) >> NI::AlphaProperty::DEST_BLEND_POS));
            }

            // NiStencilProperty draw mode → two-sided flag. DRAW_BOTH means MW disables
            // culling (thin double-sided geometry). Absent stencil or any other mode
            // (DRAW_CCW_OR_BOTH / DRAW_CCW / DRAW_CW) is single-sided → CULL_BACK in the
            // alpha pass. (DRAW_CW is reversed single-sided; rare — treated as CULL_BACK
            // for now, revisit if a shape reads inside-out.)
            if (ps->stencil) {
                e.twoSided = (ps->stencil->drawMode == NI::StencilProperty::DRAW_BOTH);
            }

            if (ps->texture) {
                const auto* baseMap = ps->texture->getBaseMap();
                if (baseMap && baseMap->texture) {
                    auto* tex = baseMap->texture.get();
                    if (tex->isInstanceOfType(NI::RTTIStaticPtr::NiSourceTexture)) {
                        auto* st = static_cast<NI::SourceTexture*>(tex);
                        e.textureName = st->fileName;
                        e.d3dTexture  = getDX9Texture(tex);
                        e.baseUV = baseMap->texCoordSet >= 3u ? 3u : static_cast<uint8_t>(baseMap->texCoordSet);
                        // The map's address mode (CLAMP_S_CLAMP_T .. WRAP_S_WRAP_T), shipped raw to
                        // the host — without it every draw sampled REPEAT and clamped meshes tiled.
                        e.baseClamp = static_cast<uint8_t>(baseMap->clampMode) & 3u;
                        registerTextureName(tex);   // AT3 reverse-map populate (base map)
                    }
                }
                // Multi-map siblings (dark/detail/glow) on the same property — the
                // PPL fixed-function blend the cache color pass reconstructs. Store
                // the D3D9 texture and the UV set it samples (its true texCoordSet,
                // clamped to 0..3). uploadEntry sizes the VB so every used set is
                // carried (e.g. the glow-mod detail map on set 2).
                auto captureMap = [&](NI::TexturingProperty::Map* map,
                                      IDirect3DTexture9*& outTex, uint8_t& outUV,
                                      const char*& outName, uint8_t& outClamp) {
                    if (!map || !map->texture) return;
                    auto* mtex = map->texture.get();
                    if (!mtex->isInstanceOfType(NI::RTTIStaticPtr::NiSourceTexture)) return;
                    IDirect3DTexture9* d3d = getDX9Texture(mtex);
                    if (!d3d) return;
                    outTex = d3d;
                    // The map's texture is a confirmed NiSourceTexture — also record its
                    // source filename so the Forge path can resolve it to a bindless slot.
                    outName = static_cast<NI::SourceTexture*>(mtex)->fileName;
                    registerTextureName(mtex);   // AT3 reverse-map populate (dark/detail/glow)
                    // Store the map's TRUE UV set (clamped to 3 — FFE texcoordIndex is
                    // 2-bit / FVF carries <=4 sets). uploadEntry sizes the VB to cover it.
                    outUV  = map->texCoordSet >= 3u ? 3u : static_cast<uint8_t>(map->texCoordSet);
                    // Per-map address mode — each stage carries its OWN (a glow map can clamp over a
                    // wrapping base), so this cannot be hoisted to one value per shape.
                    outClamp = static_cast<uint8_t>(map->clampMode) & 3u;
                };
                captureMap(ps->texture->getDarkMap(),   e.d3dDark,   e.darkUV,   e.darkTextureName,   e.darkClamp);
                captureMap(ps->texture->getDetailMap(), e.d3dDetail, e.detailUV, e.detailTextureName, e.detailClamp);
                captureMap(ps->texture->getGlowMap(),   e.d3dGlow,   e.glowUV,   e.glowTextureName,   e.glowClamp);

                // Terrain decal overlay: maps[6] = DECAL_1 (the second land texture
                // for splat blending). Present on multi-texture terrain patches.
                if (ps->texture->maps.getEndIndex() > 6u) {
                    const auto* decalMap = ps->texture->maps.at(6);
                    if (decalMap && decalMap->texture) {
                        auto* dtex = decalMap->texture.get();
                        if (dtex->isInstanceOfType(NI::RTTIStaticPtr::NiSourceTexture)) {
                            e.d3dOverlay = getDX9Texture(dtex);
                            // Record the overlay source filename so the Forge path can
                            // resolve it to a bindless slot (same cast captureMap uses).
                            e.overlayTextureName = static_cast<NI::SourceTexture*>(dtex)->fileName;
                            registerTextureName(dtex);   // AT3 reverse-map populate (decal overlay)
                        }
                    }
                }
            }

            // Last: needs matEmissive (read above) and, for the diagnostic logline, the
            // captured base texture name. Derived unconditionally rather than gated on the
            // INI knob so ForgeEmissiveBoost can be flipped live — materials are captured
            // once per cache entry, so the gain must already be sitting there when the
            // toggle turns on.
            computeEmissiveGain(e, geom);
        }

        // Vertex layout matching MorrowindVertIn (depth/shadow VS input).
        // D3DFVF_XYZ | D3DFVF_NORMAL | D3DFVF_DIFFUSE | D3DFVF_TEX1, stride 36.
        struct DepthVertex {
            float x, y, z;    // POSITION  (12)
            float nx, ny, nz; // NORMAL    (12, model-space; depth/shadow ignore, color pass lights)
            DWORD color;       // DIFFUSE   ( 4, 0xFFFFFFFF — hasVCol=false, unused)
            float u, v;        // TEXCOORD0 ( 8, from UV set 0)
        };
        static_assert(sizeof(DepthVertex) == 36, "DepthVertex size mismatch");
        // Multi-map shapes append extra UV sets after the TEXCOORD0 of DepthVertex
        // (stride = kVBStridePos + 8*uvSetCount). uploadEntry writes them with a
        // generic byte-offset writer rather than a fixed struct, so 1..4 UV sets
        // share one code path; the depth/shadow VS read only TEXCOORD0.

        // Skinned vertex layout matching SkinnedVertIn (VS palette skinning input).
        // Drawn with g_skinnedDecl; stride 56.
        struct SkinnedVertex {
            float x, y, z;          // POSITION     (12) bind-pose
            float nx, ny, nz;       // NORMAL       (12) bind-pose model-space
            float w0, w1, w2, w3;   // BLENDWEIGHT  (16) top-4 influences, normalized
            DWORD indices;          // BLENDINDICES ( 4) UBYTE4 bone palette indices
            float u, v;             // TEXCOORD0    ( 8)
            DWORD color;            // COLOR        ( 4) Phase 2: per-vertex colour
        };
        static_assert(sizeof(SkinnedVertex) == 56, "SkinnedVertex size mismatch");

        // NiUVController takeover: find an ACTIVE NiUVController that animates the UV OFFSETS
        // of this shape. The engine applies such a controller by rewriting the mesh's vertex UV
        // array every update tick (revisionID bump -> D3D9 VB rebuild + blocking geom-RPC
        // reship — the GitD night-collapse class). Instead the key track ships ONCE and the
        // Forge host scrolls the UVs in-shader from sim time. Returns nullptr when absent or
        // unsupported (animated TILING — rare; those shapes keep the engine reship path).
        NI::UVController* findUVAnimController(NI::TriBasedGeometry* geom) {
            for (NI::TimeController* c = geom->controllers; c; c = c->nextController) {
                if ((c->flags & NI::TimeControllerFlags::Active) == 0) { continue; }
                if (!c->isOfType(NI::RTTIStaticPtr::NiUVController)) { continue; }
                auto* uvc = static_cast<NI::UVController*>(c);
                NI::UVData* d = uvc->uvData.get();
                if (!d) { return nullptr; }
                if (d->UTilingData.numKeys > 1 || d->VTilingData.numKeys > 1) {
                    static bool warnedTiling = false;
                    if (!warnedTiling) {
                        warnedTiling = true;
                        const char* nm = geom->getName();
                        LOG::logline("!! [uvanim] animated UV TILING unsupported (shape '%s') — engine reship path kept",
                                     (nm && *nm) ? nm : "?");
                    }
                    return nullptr;
                }
                if (d->UOffsetData.numKeys < 2 && d->VOffsetData.numKeys < 2) {
                    return nullptr;   // constant offsets — nothing animates
                }
                return uvc;
            }
            return nullptr;
        }

        // Extract one NiUVData float-key track as linear (time, value) pairs. Linear keys
        // (type 1, stride 8) ship as-is; bezier keys (type 2, stride 16: t/v/forward/backward)
        // are Hermite-RESAMPLED to linear at 15 Hz over the track's own time span; TBC keys
        // (type 3, stride 20) fall back to a linear walk of their (t, v) lanes (log once —
        // base-game UV anims are linear/bezier). Unknown types clear the output.
        void extractUVTrack(const NI::UVData::KeyData& kd,
                            std::vector<std::pair<float, float>>& out) {
            out.clear();
            if (!kd.keys || kd.numKeys == 0) { return; }
            const auto* bytes = static_cast<const uint8_t*>(kd.keys);
            const uint32_t n = kd.numKeys;
            if (kd.type == 1) {                       // linear {t, v}
                out.reserve(n);
                for (uint32_t i = 0; i < n; ++i) {
                    const float* k = reinterpret_cast<const float*>(bytes + (size_t)i * 8);
                    out.emplace_back(k[0], k[1]);
                }
            } else if (kd.type == 2) {                // bezier {t, v, forward, backward}
                // TANGENT CONVENTION (2026-07-19): MW's runtime treats a key's BACKWARD tangent as
                // the OUTGOING (segment-start) slope and its FORWARD tangent as the INCOMING
                // (segment-end) slope — the OPPOSITE of niflib. Proven on ex_vivec_waterfall_01:
                // the VOffset ramp stores its real slope (-3) in key0.backward + key1.forward and
                // 0 in key0.forward + key1.backward. Feeding (a.fwd, b.bwd) into the Hermite basis
                // gave both end tangents = 0 → an ease-in/ease-out ramp (visible as a waterfall
                // that slows then speeds each loop); (a.bwd, b.fwd) reproduces MW's exact linear
                // constant-speed scroll. So the h10 (start) term takes a.bwd, the h11 (end) b.fwd.
                struct BK { float t, v, fwd, bwd; };
                auto keyAt = [&](uint32_t i) {
                    BK k; std::memcpy(&k, bytes + (size_t)i * 16, sizeof(k)); return k;
                };
                const float t0 = keyAt(0).t, t1 = keyAt(n - 1).t;
                const float span = t1 - t0;
                if (span <= 0.0f || n < 2) { out.emplace_back(t0, keyAt(0).v); return; }
                uint32_t samples = (uint32_t)(span * 15.0f) + 2u;   // ~15 Hz, ends inclusive
                if (samples > 512u) { samples = 512u; }
                out.reserve(samples);
                uint32_t seg = 0;
                for (uint32_t s = 0; s < samples; ++s) {
                    const float t = t0 + span * ((float)s / (float)(samples - 1));
                    while (seg + 2 < n && keyAt(seg + 1).t <= t) { ++seg; }
                    const BK a = keyAt(seg), b = keyAt(seg + 1);
                    const float dt = b.t - a.t;
                    const float u = dt > 0.0f ? (t - a.t) / dt : 0.0f;
                    const float u2 = u * u, u3 = u2 * u;
                    const float v = (2*u3 - 3*u2 + 1) * a.v + (-2*u3 + 3*u2) * b.v
                                  + (u3 - 2*u2 + u) * a.bwd + (u3 - u2) * b.fwd;
                    out.emplace_back(t, v);
                }
            } else if (kd.type == 3) {                // TBC {t, v, tension, bias, continuity}
                static bool warnedTBC = false;
                if (!warnedTBC) {
                    warnedTBC = true;
                    LOG::logline("-- [uvanim] TBC UV keys approximated as linear");
                }
                out.reserve(n);
                for (uint32_t i = 0; i < n; ++i) {
                    const float* k = reinterpret_cast<const float*>(bytes + (size_t)i * 20);
                    out.emplace_back(k[0], k[1]);
                }
            }
        }

        // Build the GeomUVAnimWire blob (header + U keys + V keys) shipped once with the mesh.
        // Returns false (empty out) when nothing usable — the caller then keeps the engine
        // reship path (and must NOT exclude the UV component from the content gate).
        bool buildUVAnimPayload(NI::UVController* uvc, uint8_t uvSetCount,
                                std::vector<uint8_t>& out) {
            out.clear();
            NI::UVData* d = uvc->uvData.get();
            if (!d) { return false; }
            static std::vector<std::pair<float, float>> uKeys, vKeys;  // single-threaded walk
            extractUVTrack(d->UOffsetData, uKeys);
            extractUVTrack(d->VOffsetData, vKeys);
            if (uKeys.size() < 2 && vKeys.size() < 2) { return false; }
            const size_t bytes = sizeof(IPC::GeomUVAnimWire) + 8u * (uKeys.size() + vKeys.size());
            if (bytes > 0xFFFFu) { return false; }   // uvAnimBytes is uint16 (never hit in practice)

            IPC::GeomUVAnimWire w = {};
            const uint8_t maxSet = uvSetCount ? (uint8_t)(uvSetCount - 1) : 0u;
            w.setIndex  = uvc->textureSet < maxSet ? (uint8_t)uvc->textureSet : maxSet;
            // TimeController cycleType bits 1-2: Loop=0 / Reverse=2 / Clamp=4 -> 0/1/2.
            w.cycleType = (uint8_t)((uvc->flags & NI::TimeControllerFlags::CycleTypeMask) >> 1);
            w.keyCountU = (uint16_t)uKeys.size();
            w.keyCountV = (uint16_t)vKeys.size();
            w.frequency = uvc->frequency;
            w.phase     = uvc->phase;
            w.keyMin    = uvc->lowKeyFrame;
            w.keyMax    = uvc->highKeyFrame;
            // The verts captured THIS run already embed the controller's current offsets —
            // the host applies eval(t) - base so capture time drops out.
            w.baseU     = uvc->currentUOffset;
            w.baseV     = uvc->currentVOffset;

            out.resize(bytes);
            uint8_t* dst = out.data();
            std::memcpy(dst, &w, sizeof(w));                       dst += sizeof(w);
            if (!uKeys.empty()) {
                std::memcpy(dst, uKeys.data(), uKeys.size() * 8);  dst += uKeys.size() * 8;
            }
            if (!vKeys.empty()) {
                std::memcpy(dst, vKeys.data(), vKeys.size() * 8);
            }
            return true;
        }

        void uploadEntry(CachedGeometry& e, NI::TriBasedGeometry* geom,
                         NI::TriBasedGeometryData* data, uint32_t key) {
            const auto vertexCount = static_cast<uint32_t>(data->getActiveVertexCount());
            const auto triCount    = static_cast<uint32_t>(data->getActiveTriangleCount());

            const auto* mv = data->vertex;          // model-space, always present

            // Invalid geometry — drop any stale buffers so the entry is skipped.
            if (!vertexCount || !triCount || !mv) {
                releaseEntry(e);
                return;
            }

            // Reship probe: uploadEntry only runs on first upload or a revision bump, so an
            // entry passing through here EVERY walk means an engine controller bumps
            // data->revisionID per frame — and each rebuild re-ships the part to the Forge
            // host over the blocking geom RPC (the [hb] geom= floor; seen live as one slot
            // rev++ every frame in the host log). Name the offender once so the source can
            // be fixed or rerouted to the dynamic ring.
            {
                struct ReshipStreak { uint32_t lastWalk; uint32_t streak; bool logged; };
                static std::unordered_map<uint32_t, ReshipStreak> s_reship;
                auto& rs = s_reship[key];
                rs.streak = (rs.lastWalk + 1 == g_walkSerial) ? rs.streak + 1 : 1;
                rs.lastWalk = g_walkSerial;
                if (rs.streak == 120 && !rs.logged) {
                    rs.logged = true;
                    const char* nm = geom->getName();
                    NI::Node* p1 = geom->parentNode;
                    NI::Node* p2 = p1 ? p1->parentNode : nullptr;
                    NI::Node* p3 = p2 ? p2->parentNode : nullptr;
                    LOG::logline("!! [reship] every-walk rebuild: key=%08X v=%u tri=%u rev=%u shape='%s' parents='%s' <- '%s' <- '%s'",
                                 key, vertexCount, triCount, (unsigned)data->revisionID,
                                 (nm && *nm) ? nm : "?",
                                 (p1 && p1->getName()) ? p1->getName() : "?",
                                 (p2 && p2->getName()) ? p2->getName() : "?",
                                 (p3 && p3->getName()) ? p3->getName() : "?");
                }
            }

            // UV-set count: carry as many sets as the maps actually use (the glow-mod
            // windows put base/dark on sets 0/1 and the detail map on set 2). The maps'
            // texCoordSets were captured in extractMaterial (which runs first). Bound by
            // the mesh's own set count and 4 (FFE texcoordIndex is 2-bit). Per-set blocks
            // are contiguous in textureCoords — set s starts at +s*storedVerts, sized by
            // the data's stored vertexCount. Single-UV geometry resolves to count 1.
            // Landscape excluded: terrain splats via d3dOverlay and its passes bind the
            // single-UV stride unconditionally.
            const uint16_t storedVerts = data->vertexCount;
            uint8_t maxMapUV = e.baseUV;
            if (e.d3dDark)   maxMapUV = std::max(maxMapUV, e.darkUV);
            if (e.d3dDetail) maxMapUV = std::max(maxMapUV, e.detailUV);
            if (e.d3dGlow)   maxMapUV = std::max(maxMapUV, e.glowUV);
            const uint8_t availSets = data->textureCoords
                ? static_cast<uint8_t>(std::min<unsigned>(data->textureSets, 4u)) : 1u;
            uint8_t uvSetCount = std::min<uint8_t>(static_cast<uint8_t>(maxMapUV + 1), availSets);
            if (uvSetCount < 1 || g_walkingLandscape) uvSetCount = 1;
            const unsigned int stride = MGE::GeometryCache::kVBStridePos + 8u * uvSetCount;
            const DWORD vbFVF = MGE::GeometryCache::kVBFVFBase
                              | (static_cast<DWORD>(uvSetCount) << D3DFVF_TEXCOUNT_SHIFT);

            // NiUVController takeover: build the key-track payload BEFORE the content gate so
            // the gate can drop the UV component from its hash — the engine's per-tick UV
            // rewrite then short-circuits the ENTIRE rebuild+reship above instead of only
            // absorbing identical frames. Payload-build failure (or an unsupported controller)
            // leaves the vector empty -> no exclusion, engine reship path unchanged. Sky keeps
            // its own SK3 scroll-diff mechanism; FP and landscape have no host id stamping.
            static std::vector<uint8_t> uvAnimPayload;   // single-threaded cache walk
            uvAnimPayload.clear();
            if (!g_walkingSky && !g_walkingFP && !g_walkingLandscape && data->textureCoords) {
                if (NI::UVController* uvc = findUVAnimController(geom)) {
                    buildUVAnimPayload(uvc, uvSetCount, uvAnimPayload);
                }
            }
            const bool hasUVAnim = !uvAnimPayload.empty();

            // Part A upload accounting: tag WHY this part reships (new vs which array changed) so
            // the [uploads] heartbeat + host panel can break the per-frame host-geom cost down by
            // cause. Set from the content gate below; kUpTopo overrides on a size/stride change.
            int uploadCat = RenderProcess::kUpNew;

            // Content-identity gate (NightDaySwitch finding, 2026-07-17): a mod can bump
            // data->revisionID EVERY frame on hundreds of parts without changing a byte —
            // the glow-windows day/night switch does exactly that at night (235 4-vert
            // window quads, named by the [reship] probe), and each spurious bump costs a
            // D3D9 VB rebuild here AND a client-blocking geom-RPC reship. Hash the source
            // arrays; identical content ⇒ consume the revision stamp and skip everything.
            // Real animation (morph heads, sky vcol fades) hashes differently and passes
            // through unchanged, so this can never drop a genuine update.
            {
                auto fnv = [](const void* p, size_t n, uint64_t h) {
                    const uint8_t* b = static_cast<const uint8_t*>(p);
                    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
                    return h;
                };
                constexpr uint64_t kFnvBasis = 1469598103934665603ull;
                // Per-component hashes: the combined value drives the skip; the parts tell
                // the [reship-diff] log WHICH array a per-frame animator actually rewrites
                // (pos/nrm/vcol/uv/tri) — the artist-facing answer.
                uint64_t hc[5] = { kFnvBasis, kFnvBasis, kFnvBasis, kFnvBasis, kFnvBasis };
                hc[0] = fnv(mv, vertexCount * sizeof(mv[0]), hc[0]);
                if (data->normal) hc[1] = fnv(data->normal, vertexCount * sizeof(data->normal[0]), hc[1]);
                if (data->color)  hc[2] = fnv(data->color,  vertexCount * sizeof(data->color[0]), hc[2]);
                if (data->textureCoords) {
                    for (uint8_t s = 0; s < uvSetCount; ++s) {
                        hc[3] = fnv(data->textureCoords + (uint32_t)s * storedVerts,
                                    vertexCount * sizeof(data->textureCoords[0]), hc[3]);
                    }
                }
                if (const auto* tl = data->getTriList()) hc[4] = fnv(tl, (size_t)triCount * 6u, hc[4]);
                const uint32_t counts[3] = { vertexCount, triCount, uvSetCount };
                uint64_t h = fnv(counts, sizeof(counts), kFnvBasis);
                // NiUVController takeover: the host animates this shape's UVs from the shipped
                // key track, so the engine's per-tick UV rewrites must NOT read as "content
                // changed" — exclude the UV component from the skip hash. hc[3] is still
                // computed for the [reship-diff] component log.
                for (int c = 0; c < 5; ++c) {
                    if (c == 3 && hasUVAnim) { continue; }
                    h = fnv(&hc[c], sizeof(hc[c]), h);
                }

                struct ContentSig { uint64_t h; uint64_t hc[5]; uint32_t lastChangeWalk; bool loggedDiff; };
                static std::unordered_map<uint32_t, ContentSig> s_contentSig;
                auto it = s_contentSig.find(key);
                // Skip only if the content is unchanged AND every output the CURRENT consumers
                // want is already produced: the host ship (if wantsGeometryCapture) and the DX9
                // mirror VB (if needMirror). When a mode toggles on (F7-off, NUMPAD7 CACHE) the
                // mirror term goes false for entries captured mirror-free, so this stops skipping
                // and the body below rebuilds the mirror on the next walk — no purge required.
                const bool haveHost   = e.hostUploaded || !RenderProcess::wantsGeometryCapture();
                const bool haveMirror = e.vb[e.writeSlot] || !needMirror();
                if (it != s_contentSig.end() && it->second.h == h && haveHost && haveMirror) {
                    e.revisionID = data->revisionID;   // consume the spurious bump
                    return;
                }
                if (it != s_contentSig.end()) {
                    // Part A: tag the dominant changed component for the upload breakdown. uv first
                    // (a uv change here after the takeover is the unmigrated remainder / leak), then
                    // pos (morph + particle regen), vcol, tri, normals.
                    if      (it->second.hc[3] != hc[3]) uploadCat = RenderProcess::kUpUvLeak;
                    else if (it->second.hc[0] != hc[0]) uploadCat = RenderProcess::kUpMorph;
                    else if (it->second.hc[2] != hc[2]) uploadCat = RenderProcess::kUpVcol;
                    else if (it->second.hc[4] != hc[4]) uploadCat = RenderProcess::kUpTopo;
                    else                                uploadCat = RenderProcess::kUpOther;
                    // Content really changed under a revision bump. Name the changed
                    // component(s) + the tick interval once per mesh — this is the
                    // "which animation method" answer for content authors.
                    if (!it->second.loggedDiff) {
                        it->second.loggedDiff = true;
                        char comps[24]; int off = 0;
                        static const char* kCompName[5] = { "pos", "nrm", "vcol", "uv", "tri" };
                        for (int c = 0; c < 5; ++c) {
                            if (it->second.hc[c] != hc[c]) {
                                off += std::snprintf(comps + off, sizeof(comps) - off, "%s%s",
                                                     off ? "+" : "", kCompName[c]);
                            }
                        }
                        const char* nm = geom->getName();
                        LOG::logline("!! [reship-diff] key=%08X changed=%s interval=%u walks v=%u shape='%s'",
                                     key, off ? comps : "counts", g_walkSerial - it->second.lastChangeWalk,
                                     vertexCount, (nm && *nm) ? nm : "?");
                    }
                    it->second.h = h;
                    std::memcpy(it->second.hc, hc, sizeof(hc));
                    it->second.lastChangeWalk = g_walkSerial;
                } else {
                    ContentSig sig = {};
                    sig.h = h;
                    std::memcpy(sig.hc, hc, sizeof(hc));
                    sig.lastChangeWalk = g_walkSerial;
                    s_contentSig.emplace(key, sig);
                }
            }

            // On a size OR UV-layout change, drop both slots + IB so they repopulate
            // at the new size/stride.
            const bool sizeChanged = (e.vertexCount != vertexCount)
                || (e.triangleCount != triCount) || (e.uvSetCount != uvSetCount);
            if (sizeChanged) {
                releaseEntry(e);
                if (uploadCat != RenderProcess::kUpNew) uploadCat = RenderProcess::kUpTopo;  // realloc dominates
            }

            const uint8_t slot = 1u - e.writeSlot;

            // DX9 mirror VB/IB — legacy cache consumers only (needMirror). Skipped in default
            // Forge play so the capture path stays D3D9-free (the host IPC ship below is
            // independent and always runs). The AABB + sky baselines are CPU-side cache outputs
            // (world-AABB light selection, SK3/SK4) and are ALWAYS computed; only the per-vertex
            // VB write is folded in when the mirror buffer is live. Model-space VB
            // (XYZ|NORMAL|DIFFUSE|TEXn); per-draw world*view palette.
            uint8_t* vbBase = nullptr;
            if (needMirror()) {
                if (!e.vb[slot]) {
                    HRESULT hr = g_device->CreateVertexBuffer(
                        vertexCount * stride, D3DUSAGE_WRITEONLY, vbFVF,
                        g_spikeForceDefaultPool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &e.vb[slot], nullptr);
                    if (FAILED(hr)) e.vb[slot] = nullptr;
                }
                if (e.ib == nullptr) {
                    g_device->CreateIndexBuffer(
                        triCount * 6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16,
                        g_spikeForceDefaultPool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &e.ib, nullptr);
                }
                void* vbData = nullptr;
                if (e.vb[slot] && SUCCEEDED(e.vb[slot]->Lock(0, 0, &vbData, 0)))
                    vbBase = static_cast<uint8_t*>(vbData);
            }

            {
                const auto* uvs = data->textureCoords;  // NI::Point2*, set-major, nullptr if no UVs
                const auto* nrm = data->normal;         // NI::Point3*, nullptr if no normals
                const auto* vcol = data->color;         // NI::PackedColor*(b,g,r,a)=D3DCOLOR, null if none
                // Tight model-space AABB over the verts (world-AABB light selection matching the
                // reactive computeBoundingBox). The per-vertex VB write (DepthVertex prefix
                // pos+normal+color+UV0, then 8 bytes per additional UV set — set s for vertex i
                // at uvs[s*storedVerts+i], set-major) is folded in only when vbBase is live.
                float mn[3] = { mv[0].x, mv[0].y, mv[0].z };
                float mx[3] = { mv[0].x, mv[0].y, mv[0].z };
                for (uint32_t i = 0; i < vertexCount; ++i) {
                    if (mv[i].x < mn[0]) mn[0] = mv[i].x; if (mv[i].x > mx[0]) mx[0] = mv[i].x;
                    if (mv[i].y < mn[1]) mn[1] = mv[i].y; if (mv[i].y > mx[1]) mx[1] = mv[i].y;
                    if (mv[i].z < mn[2]) mn[2] = mv[i].z; if (mv[i].z > mx[2]) mx[2] = mv[i].z;
                    if (vbBase) {
                        auto* v = reinterpret_cast<DepthVertex*>(vbBase + i * stride);
                        v->x = mv[i].x; v->y = mv[i].y; v->z = mv[i].z;
                        if (nrm) { v->nx = nrm[i].x; v->ny = nrm[i].y; v->nz = nrm[i].z; }
                        else     { v->nx = 0.0f; v->ny = 0.0f; v->nz = 1.0f; }
                        // PackedColor byte order (b,g,r,a) is exactly D3DCOLOR, copy straight.
                        v->color = vcol ? *reinterpret_cast<const DWORD*>(&vcol[i]) : 0xFFFFFFFF;
                        v->u = uvs ? uvs[i].x : 0.0f;          // UV set 0
                        v->v = uvs ? uvs[i].y : 0.0f;
                        auto* extra = reinterpret_cast<float*>(vbBase + i * stride + 36);
                        for (uint8_t s = 1; s < uvSetCount; ++s) {
                            const auto& p = uvs[s * storedVerts + i];
                            *extra++ = p.x;
                            *extra++ = p.y;
                        }
                    }
                }
                e.aabbMin[0] = mn[0]; e.aabbMin[1] = mn[1]; e.aabbMin[2] = mn[2];
                e.aabbMax[0] = mx[0]; e.aabbMax[1] = mx[1]; e.aabbMax[2] = mx[2];
                // SK3 cloud scroll baseline: the UVs baked into THIS upload. The per-frame sky
                // walk diffs the live UVs against these to derive the scroll offset (sky VBs
                // never re-upload, so the baseline stays valid for the entry's lifetime).
                e.skyBaseUV[0]     = uvs ? uvs[0].x : 0.0f;
                e.skyBaseUV[1]     = uvs ? uvs[0].y : 0.0f;
                e.skyBaseUVLast[0] = uvs ? uvs[vertexCount - 1].x : 0.0f;
                e.skyBaseUVLast[1] = uvs ? uvs[vertexCount - 1].y : 0.0f;
                e.skyUVOffset[0] = 0.0f;
                e.skyUVOffset[1] = 0.0f;
                // SK4 vcol baseline: hash of the colours baked into THIS upload; the
                // per-frame sky walk re-uploads when the live colours diverge.
                e.skyVcolHash = (g_walkingSky && vcol)
                    ? hashVertexColors(vcol, vertexCount) : 0u;
                if (vbBase) e.vb[slot]->Unlock();
            }

            // Non-skinned topology may change on a revision bump, so rewrite the IB whenever
            // uploadEntry runs (rare — first upload or revision change). needMirror only (e.ib
            // is null in default Forge play — the host gets its indices via captureGeometry).
            if (e.ib) {
                const auto* triList = data->getTriList();
                if (triList) {
                    void* ibData = nullptr;
                    if (SUCCEEDED(e.ib->Lock(0, 0, &ibData, 0))) {
                        memcpy(ibData, triList, triCount * 6);
                        e.ib->Unlock();
                    }
                }
            }

            e.writeSlot          = slot;
            e.vertexCount        = vertexCount;
            e.triangleCount      = triCount;
            e.revisionID         = data->revisionID;
            e.isSkinned          = false;
            e.numBones           = 0;
            e.skinnedUnsupported = false;
            e.uvSetCount         = uvSetCount;
            e.vbStride           = static_cast<uint16_t>(stride);
            e.vbFVF              = vbFVF;
            e.hasVertexColor     = (data->color != nullptr);

            const auto& b = data->bounds;
            e.boundsCenter[0] = b.center.x;
            e.boundsCenter[1] = b.center.y;
            e.boundsCenter[2] = b.center.z;
            e.boundsRadius    = b.radius;

            // M1: ship model-space pos+normal+indices to the Forge host (non-skinned
            // opaques AND near terrain — worldLandscapeRoot patches carry model-space
            // vertex/normal/triList just like objects, and buildD3DTransform already set
            // worldTransformD3D for them above). Flat-shaded for now (texturing is the
            // next milestone, shared by objects+terrain). Re-uploads only on revision change.
            if (RenderProcess::wantsGeometryCapture()) {
                const auto* nrm = data->normal;
                const auto* capUvs = data->textureCoords;
                // Tier 2a lighting: ship the real per-vertex colour ONLY when the mesh uses
                // VertexColorProperty source 2 (ambient+diffuse / DiffAmb) — the case where MW
                // actually folds vcol into lighting. Otherwise ship white (0xFFFFFFFF) so the
                // host's universal col*(d+a) path reduces to the white-material (d+a) case.
                // (Emissive routing / non-white material constants are Tier 2b.)
                const auto* vcol = (e.hasVertexColor && e.vColSource == 2) ? data->color : nullptr;
                const auto* triList = data->getTriList();

                // Tier 4 multi-map: a part with dark/detail/glow siblings rides the SEPARATE
                // wide vertex format (GeomVertexWireMM, 4 UV sets) + its own host pipeline.
                // Single-map parts (the 99% case) keep the lean GeomVertexWire path. Landscape
                // is excluded (uvSetCount forced to 1 above; terrain splats via d3dOverlay).
                const bool isMultiMap = !g_walkingLandscape
                    && (e.d3dDark || e.d3dDetail || e.d3dGlow);

                if (isMultiMap && triList) {
                    static std::vector<IPC::GeomVertexWireMM> mmScratch;  // single-threaded cache walk
                    mmScratch.resize(vertexCount);
                    for (uint32_t i = 0; i < vertexCount; ++i) {
                        auto& w = mmScratch[i];
                        w.px = mv[i].x; w.py = mv[i].y; w.pz = mv[i].z;
                        if (nrm) { w.nx = nrm[i].x; w.ny = nrm[i].y; w.nz = nrm[i].z; }
                        else     { w.nx = 0.0f;    w.ny = 0.0f;    w.nz = 1.0f; }
                        w.color = vcol ? *reinterpret_cast<const DWORD*>(&vcol[i]) : 0xFFFFFFFFu;
                        // UV sets 0..3, read set-major (uvs[set*storedVerts + i]). Sets the VB
                        // doesn't carry (>= uvSetCount) duplicate set 0 — those stages are
                        // dropped client-side (cacheMapActive: uv < uvSetCount) so never read.
                        for (uint8_t s = 0; s < 4; ++s) {
                            const uint8_t src = (capUvs && s < uvSetCount) ? s : 0u;
                            if (capUvs) {
                                const auto& p = capUvs[(uint32_t)src * storedVerts + i];
                                w.uv[s][0] = p.x; w.uv[s][1] = p.y;
                            } else {
                                w.uv[s][0] = 0.0f; w.uv[s][1] = 0.0f;
                            }
                        }
                    }
                    RenderProcess::captureMultiMapGeometry(key, data->revisionID,
                        reinterpret_cast<uint32_t>(data),   // object identity (recycled-key guard)
                        mmScratch.data(), vertexCount,
                        reinterpret_cast<const uint16_t*>(triList), triCount * 3u,
                        hasUVAnim ? uvAnimPayload.data() : nullptr,
                        hasUVAnim ? (uint16_t)uvAnimPayload.size() : 0u);
                    RenderProcess::noteUpload((uint8_t)uploadCat,
                        vertexCount * (uint32_t)sizeof(IPC::GeomVertexWireMM)
                        + triCount * 6u + (uint32_t)uvAnimPayload.size());
                } else {
                    static std::vector<IPC::GeomVertexWire> scratch;  // single-threaded cache walk
                    scratch.resize(vertexCount);
                    // Base-map UV: set e.baseUV (set-major, uvs[set*storedVerts + i]). Most static
                    // meshes use set 0; honour the captured base map's true set for correctness.
                    const uint32_t uvBase = (uint32_t)e.baseUV * storedVerts;
                    for (uint32_t i = 0; i < vertexCount; ++i) {
                        auto& w = scratch[i];
                        w.px = mv[i].x; w.py = mv[i].y; w.pz = mv[i].z;
                        if (nrm) { w.nx = nrm[i].x; w.ny = nrm[i].y; w.nz = nrm[i].z; }
                        else     { w.nx = 0.0f;    w.ny = 0.0f;    w.nz = 1.0f; }
                        if (capUvs) { w.u = capUvs[uvBase + i].x; w.v = capUvs[uvBase + i].y; }
                        else        { w.u = 0.0f;                 w.v = 0.0f; }
                        w.color = vcol ? *reinterpret_cast<const DWORD*>(&vcol[i]) : 0xFFFFFFFFu;
                    }
                    if (triList) {
                        // NI::Triangle is 3 packed uint16 indices (== the IB byte layout used above
                        // via memcpy(.., triCount*6)). SK1 dome: NO forced re-upload anymore. The host
                        // now colours the atmosphere dome geometrically (sky.frag vertical gradient
                        // fog->zenith from the thin per-frame skyZenith param), so its per-frame baked
                        // vertex-colour gradient is irrelevant — ship the mesh ONCE like any static
                        // (modelId,vc,rev), leaving the geometry channel idle in a static scene.
                        RenderProcess::captureGeometry(key, data->revisionID,
                            reinterpret_cast<uint32_t>(data),   // object identity (recycled-key guard)
                            scratch.data(), vertexCount,
                            reinterpret_cast<const uint16_t*>(triList), triCount * 3u,
                            false,
                            hasUVAnim ? uvAnimPayload.data() : nullptr,
                            hasUVAnim ? (uint16_t)uvAnimPayload.size() : 0u);
                        RenderProcess::noteUpload((uint8_t)uploadCat,
                            vertexCount * (uint32_t)sizeof(IPC::GeomVertexWire)
                            + triCount * 6u + (uint32_t)uvAnimPayload.size());
                    }
                }
            }

            // Content processed + (if a consumer wanted it) shipped this revision. Marks the
            // content-identity gate's "already produced" state now that the dead DX9 mirror VB
            // no longer serves as that proxy in Forge play. Cleared by releaseEntry on invalidation.
            e.hostUploaded = true;

            ++g_uploadedThisFrame;
        }

        void buildD3DFromTransform(float out[16], const NI::Transform& t);

        // Static skinned VB: bind-pose positions + per-vertex top-4 bone influences
        // (weights + palette indices). Built once (or on revision change); the bone
        // matrices update per frame via buildBonePalette. Sets skinnedUnsupported
        // when numBones exceeds the shader palette (kMaxBones) — no CPU fallback.
        void buildSkinnedVB(CachedGeometry& e, NI::TriBasedGeometry* geom,
                            NI::TriBasedGeometryData* data,
                            NI::SkinInstance* si, NI::SkinData* sd) {
            const auto vertexCount = static_cast<uint32_t>(data->getActiveVertexCount());
            const auto triCount    = static_cast<uint32_t>(data->getActiveTriangleCount());
            const auto* mv = data->vertex;
            if (!vertexCount || !triCount || !mv) { releaseEntry(e); return; }

            const uint32_t key      = reinterpret_cast<uint32_t>(geom);
            const uint32_t numBones = sd->numBones;

            releaseEntry(e);
            e.writeSlot          = 0;
            e.vertexCount        = vertexCount;
            e.triangleCount      = triCount;
            e.revisionID         = data->revisionID;
            e.isSkinned          = true;
            e.numBones           = numBones;
            e.uvSetCount         = 1;       // skinnedDecl carries one UV set; multi-map is non-skinned only
            e.vbStride           = MGE::GeometryCache::kSkinnedVBStride;
            e.vbFVF              = 0;        // skinned draws use skinnedDecl, not an FVF
            e.hasVertexColor     = (data->color != nullptr);   // Phase 2: skinned VB now carries colour

            if (numBones > MGE::GeometryCache::kMaxBones) {
                // Too many bones for the VS palette — skip this caster, report once.
                e.skinnedUnsupported = true;
                static bool warnOnce = true;
                if (warnOnce) {
                    LOG::logline("!! [GEOM CACHE] skinned mesh has %u bones (> kMaxBones %u); skipped",
                                 numBones, MGE::GeometryCache::kMaxBones);
                    warnOnce = false;
                }
                return;
            }
            e.skinnedUnsupported = false;

            // Invert per-bone weight lists into per-vertex influences.
            struct Inf { float w; uint8_t b; };
            std::vector<std::vector<Inf>> perVert(vertexCount);
            for (uint32_t b = 0; b < numBones; ++b) {
                const auto& bd = sd->boneData[b];
                if (!bd.weights) continue;
                for (uint32_t k = 0; k < bd.weightCount; ++k) {
                    const uint32_t vi = bd.weights[k].index;
                    if (vi >= vertexCount) continue;
                    perVert[vi].push_back({ bd.weights[k].weight, static_cast<uint8_t>(b) });
                }
            }

            // M-Skinning: compute the per-vertex skinned layout ONCE into a CPU staging array.
            // The Forge host wire stream (wantCapture) and the DX9 mirror VB (needMirror) both
            // copy from it, so in default Forge play neither the mirror nor the D3D9 device is
            // touched here. Wire stream shipped once per (key,revision), re-uploaded on change;
            // carries base-map UV (per-vertex colour DiffAmb is a later fidelity tier).
            const bool wantCapture = RenderProcess::wantsGeometryCapture();
            const bool wantMirror  = needMirror();
            static std::vector<IPC::SkinnedVertexWire> skScratch;   // single-threaded cache walk
            static std::vector<SkinnedVertex>          stg;         // CPU staging (mirror source)
            if (wantCapture) skScratch.resize(vertexCount);
            stg.resize(vertexCount);

            {
                const auto* uvs = data->textureCoords;
                const auto* nrm = data->normal;         // bind-pose model-space normals
                const auto* vcol = data->color;         // NI::PackedColor*(b,g,r,a)=D3DCOLOR, null if none
                for (uint32_t i = 0; i < vertexCount; ++i) {
                    auto& infs = perVert[i];
                    std::sort(infs.begin(), infs.end(),
                              [](const Inf& a, const Inf& b) { return a.w > b.w; });
                    float w[4] = {0,0,0,0};
                    uint8_t idx[4] = {0,0,0,0};
                    float sum = 0.0f;
                    const size_t n = infs.size() < 4 ? infs.size() : 4;
                    for (size_t j = 0; j < n; ++j) { w[j] = infs[j].w; idx[j] = infs[j].b; sum += infs[j].w; }
                    if (sum > 1e-6f) { for (int j = 0; j < 4; ++j) w[j] /= sum; }
                    else             { w[0] = 1.0f; }

                    auto& v = stg[i];
                    v.x = mv[i].x; v.y = mv[i].y; v.z = mv[i].z;
                    // Bind-pose normals; the VS skins them by the bone palette (same as
                    // position) for the cache color pass. Up if the mesh has none.
                    if (nrm) { v.nx = nrm[i].x; v.ny = nrm[i].y; v.nz = nrm[i].z; }
                    else     { v.nx = 0.0f; v.ny = 0.0f; v.nz = 1.0f; }
                    v.w0 = w[0]; v.w1 = w[1]; v.w2 = w[2]; v.w3 = w[3];
                    v.indices = static_cast<DWORD>(idx[0])
                              | (static_cast<DWORD>(idx[1]) << 8)
                              | (static_cast<DWORD>(idx[2]) << 16)
                              | (static_cast<DWORD>(idx[3]) << 24);
                    v.u = uvs ? uvs[i].x : 0.0f;
                    v.v = uvs ? uvs[i].y : 0.0f;
                    // PackedColor byte order (b,g,r,a) is exactly D3DCOLOR, copy straight.
                    v.color = vcol ? *reinterpret_cast<const DWORD*>(&vcol[i]) : 0xFFFFFFFF;
                    if (wantCapture) {
                        auto& sw = skScratch[i];
                        sw.px = v.x;  sw.py = v.y;  sw.pz = v.z;
                        sw.nx = v.nx; sw.ny = v.ny; sw.nz = v.nz;
                        sw.w0 = v.w0; sw.w1 = v.w1;
                        sw.w2 = v.w2; sw.w3 = v.w3;
                        sw.indices = v.indices;
                        sw.u = v.u;   sw.v = v.v;   // base-map UV for the host
                    }
                }
            }

            // DX9 mirror VB/IB — legacy cache consumers only (needMirror). Copies from the CPU
            // staging array; skipped (D3D9-free) in default Forge play. A mirror-alloc failure
            // no longer aborts the host ship below — the wire stream is already staged.
            if (wantMirror) {
                HRESULT hr = g_device->CreateVertexBuffer(
                    vertexCount * MGE::GeometryCache::kSkinnedVBStride, D3DUSAGE_WRITEONLY,
                    0, g_spikeForceDefaultPool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &e.vb[0], nullptr);
                if (SUCCEEDED(hr)) {
                    void* vbData = nullptr;
                    if (SUCCEEDED(e.vb[0]->Lock(0, 0, &vbData, 0))) {
                        memcpy(vbData, stg.data(), vertexCount * (uint32_t)sizeof(SkinnedVertex));
                        e.vb[0]->Unlock();
                    }
                    hr = g_device->CreateIndexBuffer(
                        triCount * 6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16,
                        g_spikeForceDefaultPool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &e.ib, nullptr);
                    if (SUCCEEDED(hr)) {
                        const auto* triList = data->getTriList();
                        if (triList) {
                            void* ibData = nullptr;
                            if (SUCCEEDED(e.ib->Lock(0, 0, &ibData, 0))) {
                                memcpy(ibData, triList, triCount * 6);
                                e.ib->Unlock();
                            }
                        }
                    }
                } else {
                    e.vb[0] = nullptr;
                }
            }

            const auto& b = data->bounds;
            e.boundsCenter[0] = b.center.x;
            e.boundsCenter[1] = b.center.y;
            e.boundsCenter[2] = b.center.z;
            e.boundsRadius    = b.radius;

            // Ship the captured skinned VB to the Forge host (one part, SKINNED flag +
            // numBones). The per-frame bone palette ships separately from buildDrawList.
            if (wantCapture) {
                const auto* triList = data->getTriList();
                if (triList) {
                    RenderProcess::captureSkinnedGeometry(key, data->revisionID,
                        reinterpret_cast<uint32_t>(data),   // object identity (recycled-key guard)
                        skScratch.data(), vertexCount,
                        reinterpret_cast<const uint16_t*>(triList), triCount * 3u, numBones);
                    // Part A: skinned reships are the known NPC/creature cost.
                    RenderProcess::noteUpload(RenderProcess::kUpSkin,
                        vertexCount * (uint32_t)sizeof(IPC::SkinnedVertexWire) + triCount * 6u);
                }
            }

            e.hostUploaded = true;   // consistent with uploadEntry (cleared by releaseEntry)

            ++g_uploadedThisFrame;
        }

        // Per-frame: fill bonePalette with each bone's model->world matrix
        // (D3D row-vector form, pos*M = world). Cheap: numBones matrices, no
        // per-vertex work. Assumes numBones <= kMaxBones (guarded at VB build).
        void buildD3DTransform(float out[16], const NI::TriBasedGeometry* geom);

        void buildBonePalette(CachedGeometry& e, const NI::TriBasedGeometry* geom,
                              NI::SkinInstance* si, NI::SkinData* sd) {
            const uint32_t numBones = sd->numBones;
            e.bonePalette.resize(numBones * 16);
            bool partHadNullBone = false;
            for (uint32_t b = 0; b < numBones; ++b) {
                float* m = &e.bonePalette[b * 16];
                NI::AVObject* boneNode = si->bones[b];
                if (!boneNode) {
                    // Phase 0: a null bone influence. Fall back to the geometry's own
                    // world transform (keeps influenced verts attached to the object)
                    // instead of identity, which parked them at the model/cell origin
                    // and stretched origin->NPC triangles into the depth buffer. The
                    // diagnostic counters stay on permanently to catch any recurrence.
                    ++g_nullBoneHitsInterval;
                    if (!partHadNullBone) {
                        partHadNullBone = true;
                        ++g_nullBonePartsInterval;
                        if (e.textureName) g_nullBoneSampleTex = e.textureName;
                    }
                    buildD3DTransform(m, geom);
                    continue;
                }
                // Compose: apply bone offset, then bone world (matches CPU-skin math).
                const NI::Transform composed = boneNode->worldTransform * sd->boneData[b].transform;
                buildD3DFromTransform(m, composed);
            }
            e.numBones = numBones;
        }

        // Build a D3D9 row-major matrix (pos*M form) from an NI::Transform.
        void buildD3DFromTransform(float out[16], const NI::Transform& t) {
            const float s = t.scale;
            const auto& R = t.rotation;
            const auto& T = t.translation;
            out[0]  = s * R.m0.x; out[1]  = s * R.m1.x; out[2]  = s * R.m2.x; out[3]  = 0;
            out[4]  = s * R.m0.y; out[5]  = s * R.m1.y; out[6]  = s * R.m2.y; out[7]  = 0;
            out[8]  = s * R.m0.z; out[9]  = s * R.m1.z; out[10] = s * R.m2.z; out[11] = 0;
            out[12] = T.x;        out[13] = T.y;         out[14] = T.z;         out[15] = 1;
        }

        void buildD3DTransform(float out[16], const NI::TriBasedGeometry* geom) {
            buildD3DFromTransform(out, geom->worldTransform);
        }

        // FP particle billboarding (see tasks/forge-fp-particles.md). MW's particle renderer
        // expands each live particle into a camera-facing quad at draw time; under FP
        // suppression that never runs and visitGeometry captured only particle centers. Rebuild
        // the quads here against the ARM camera. P0 = build into g_fpPart* + log (confirm vertex
        // space / sizes); P1 ships them as FP captured-alpha geometry.
        void buildFPParticleQuads() {
            g_fpPartVerts.clear();
            g_fpPartIndices.clear();
            g_fpPartRecs.clear();
            if (g_fpParticleSystems.empty()) return;

            // Arm camera basis (which=1): right/up billboard the quads to face the FP view.
            float pos[3], dir[3], up[3], right[3], cd[5];
            if (!MWBridge::get()->getRenderCameraState(1, pos, dir, up, right, cd)) return;

            // Under FP suppression the arm subtree's Update traversal is skipped, so the particle
            // SIMULATION (NiParticleSystemController) is frozen — positions AND per-particle sizes
            // stall until an F11 seam cycle lets a real arm render tick it. Drive it ourselves with
            // MW's app clock (same clock the engine ticks controllers with). Gated to suppression so
            // we never double-tick with MW (when the seam is off, MW ticks it and we don't capture).
            const bool tickSim = RenderProcess::wantsFPSuppression();
            const float simTime = tickSim ? MWBridge::get()->simulationTime() : 0.0f;

            for (NI::Particles* p : g_fpParticleSystems) {
                if (!p) continue;

                // Locate the NiParticleSystemController (on the NiParticles or its parent): tick it
                // (un-freeze the sim) and read initialSize — the base particle size in world units
                // that the per-particle sizes[] fraction ([0,1] grow/fade) scales.
                NI::ParticleSystemController* psc = nullptr;
                auto findPSC = [&](NI::ObjectNET* o) {
                    if (!o) return;
                    for (NI::TimeController* c = o->controllers.get(); c && !psc; c = c->nextController.get()) {
                        if (c->isInstanceOfType(NI::RTTIStaticPtr::NiParticleSystemController))
                            psc = static_cast<NI::ParticleSystemController*>(c);
                    }
                };
                findPSC(p);
                if (!psc) findPSC(p->parentNode);
                if (psc && tickSim) psc->update(simTime);
                const float initSize = (psc && psc->initialSize > 0.0f) ? psc->initialSize : 1.0f;

                auto* pd = p->getModelData().get();
                if (!pd || !pd->vertex) continue;
                const unsigned active = pd->activeCount;
                if (active == 0) continue;

                // Per-system material: base-map GPU texture + NiAlphaProperty blend. Default to
                // additive (SRC_ALPHA/ONE) — the torch-flame case — when no alpha prop is present.
                IDirect3DTexture9* tex = nullptr;
                std::uint32_t srcB = (std::uint32_t)D3DBLEND_SRCALPHA, dstB = (std::uint32_t)D3DBLEND_ONE;
                float matEmis[3] = { 0.0f, 0.0f, 0.0f };   // flame NIF glow; smoke NIF = 0
                if (auto* ps = reinterpret_cast<NI::PropertyState*>(p->propertyState)) {
                    if (ps->texture) {
                        const auto* bm = ps->texture->getBaseMap();
                        if (bm && bm->texture) tex = getDX9Texture(bm->texture.get());
                    }
                    if (ps->alpha) {
                        const unsigned f = ps->alpha->flags;
                        srcB = (std::uint32_t)niBlendToD3D((f & NI::AlphaProperty::SRC_BLEND_MASK)  >> NI::AlphaProperty::SRC_BLEND_POS);
                        dstB = (std::uint32_t)niBlendToD3D((f & NI::AlphaProperty::DEST_BLEND_MASK) >> NI::AlphaProperty::DEST_BLEND_POS);
                    }
                    // Material emissive: the torch-flame NIF sets emissive=(1,1,1) so the world
                    // captured path lights it to solid white (matE forwarded from the NiMaterial);
                    // the smoke NIF has none. Forward it so the FP flame matches (vColSource=2:
                    // lit = Color*(d+a) + MatEmissive). See renderprocess [cap-diag] proof.
                    if (ps->material) {
                        matEmis[0] = ps->material->emissive.r;
                        matEmis[1] = ps->material->emissive.g;
                        matEmis[2] = ps->material->emissive.b;
                    }
                }

                const NI::Point3*      verts = pd->vertex;
                const NI::PackedColor*  cols = pd->color;   // may be null → opaque white
                const float*           sizes = pd->sizes;   // may be null → radius fallback
                const NI::Transform&      wt = p->worldTransform;
                const float                s = wt.scale;
                const NI::Matrix33&        R = wt.rotation;
                const NI::Point3&          T = wt.translation;

                FPParticleRec rec{};
                rec.vertexBase = (std::uint32_t)g_fpPartVerts.size();
                rec.indexBase  = (std::uint32_t)g_fpPartIndices.size();
                rec.texture    = tex;
                rec.srcBlend   = srcB;
                rec.destBlend  = dstB;
                rec.matEmissive[0] = matEmis[0]; rec.matEmissive[1] = matEmis[1]; rec.matEmissive[2] = matEmis[2];
                // Self-illuminated flame (emissive ~ 1): MW's FFP clamp pins the lit vertex colour to
                // white, so the flame renders as pure albedo regardless of the per-particle colour.
                // Force the vertex RGB to white here (keeping per-particle ALPHA for the fade); the
                // client routes these through vColSource=1 → lit = Color.rgb = white → c = albedo,
                // immune to world point lights (exactly as MW renders a torch flame under any light).
                const bool emissiveSat = (matEmis[0] > 0.5f || matEmis[1] > 0.5f || matEmis[2] > 0.5f);

                for (unsigned i = 0; i < active; ++i) {
                    const NI::Point3& v = verts[i];
                    // world = scale * (R · v) + T  (row-major rows m0/m1/m2 — matches buildD3DFromTransform)
                    const float wx = s * (v.x * R.m0.x + v.y * R.m0.y + v.z * R.m0.z) + T.x;
                    const float wy = s * (v.x * R.m1.x + v.y * R.m1.y + v.z * R.m1.z) + T.y;
                    const float wz = s * (v.x * R.m2.x + v.y * R.m2.y + v.z * R.m2.z) + T.z;
                    // MW's particle quad half-extent = per-particle grow/fade fraction × the
                    // controller's base initialSize (world units), scaled by the node transform.
                    const float half = (sizes ? sizes[i] : 1.0f) * initSize * s;
                    std::uint32_t col = 0xFFFFFFFFu;
                    if (cols) std::memcpy(&col, &cols[i], sizeof(std::uint32_t));  // B,G,R,A == D3DCOLOR
                    if (emissiveSat) col |= 0x00FFFFFFu;   // force RGB white, keep A (flame fade)

                    // Camera-facing quad: center ± half*right ± half*up. Indices are rebased to
                    // 0 at rec.vertexBase (the host adds vertexBase back), matching the captured path.
                    const std::uint16_t base = (std::uint16_t)(g_fpPartVerts.size() - rec.vertexBase);
                    const float uvx[4] = { 0.f, 1.f, 1.f, 0.f };
                    const float uvy[4] = { 0.f, 0.f, 1.f, 1.f };
                    const float sx[4]  = { -1.f, 1.f, 1.f, -1.f };
                    const float sy[4]  = {  1.f, 1.f,-1.f, -1.f };
                    for (int k = 0; k < 4; ++k) {
                        IPC::GeomVertexWire gv{};
                        gv.px = wx + half * (sx[k] * right[0] + sy[k] * up[0]);
                        gv.py = wy + half * (sx[k] * right[1] + sy[k] * up[1]);
                        gv.pz = wz + half * (sx[k] * right[2] + sy[k] * up[2]);
                        // MW's particle renderer writes this exact CONSTANT normal to every
                        // billboard vertex (proven by renderprocess [nrm-diag] on world torches);
                        // it makes ndl constant per frame so the lit flame (vColSource=2) matches
                        // the world path and never swings colour with the camera.
                        gv.nx = 0.3f; gv.ny = 0.3f; gv.nz = 0.3f;
                        gv.u = uvx[k]; gv.v = uvy[k];
                        gv.color = col;
                        g_fpPartVerts.push_back(gv);
                    }
                    const std::uint16_t idx[6] = { base, (std::uint16_t)(base+1), (std::uint16_t)(base+2),
                                                   base, (std::uint16_t)(base+2), (std::uint16_t)(base+3) };
                    g_fpPartIndices.insert(g_fpPartIndices.end(), idx, idx + 6);
                }

                rec.vertexCount = (std::uint32_t)g_fpPartVerts.size() - rec.vertexBase;
                rec.indexCount  = (std::uint32_t)g_fpPartIndices.size() - rec.indexBase;
                if (rec.indexCount) g_fpPartRecs.push_back(rec);
            }

            static unsigned s_hb = 0;
            if ((s_hb++ % 600) == 0 && !g_fpPartRecs.empty()) {
                LOG::logline(">> [fp part] systems=%u quads=%u",
                             (unsigned)g_fpPartRecs.size(), (unsigned)(g_fpPartIndices.size() / 6));
            }
        }

        // Sign of the upper-left 3x3 determinant of a row-major affine matrix.
        // Negative => the transform mirrors (reflects) the mesh, flipping clip-space
        // triangle winding. Left-side body parts / armor reuse the right mesh via a
        // negative-scale node, so they hit this. The depth/shadow cache must cull the
        // OPPOSITE face for these, else it records the inner surface (SSAO shows
        // "inside-out" left limbs). Sign is layout-invariant (det A == det Aᵀ).
        bool isMirroredMatrix(const float m[16]) {
            const float det = m[0] * (m[5] * m[10] - m[6] * m[9])
                            - m[1] * (m[4] * m[10] - m[6] * m[8])
                            + m[2] * (m[4] * m[9]  - m[5] * m[8]);
            return det < 0.0f;
        }

        // Pick the transform that maps this part's mesh into world space: the bone
        // palette root for skinned parts (the limb's bones share the reflection sign),
        // the geometry's world transform otherwise.
        bool computeMirrored(const CachedGeometry& e) {
            const float* m = (e.isSkinned && e.numBones > 0 && !e.skinnedUnsupported)
                             ? e.bonePalette.data() : e.worldTransformD3D;
            return isMirroredMatrix(m);
        }

        // C4d shadow-caster category: does the owning TES3 reference say this part MOVES?
        // Resolved through the node's TES3 extra data (SharedSE getTes3Reference walks the
        // parent chain — a held weapon has no reference of its own, so the search lands on
        // the wielding NPC → live; the same weapon placed on a table IS its own Misc/Weapon
        // reference → not live). MGE consumes only SharedSE, where TES3::Reference is
        // opaque, so the two fields are read at the MWSE-documented offsets
        // (MWSE/TES3Reference.h: baseObject @ 0x28; MWSE/TES3Object.h: objectType @ 0x4).
        // Live types: Activator (silt strider idles, steam machinery), Door, NPC/Creature
        // (+ clones). Everything else — statics, clutter, containers, light fixtures —
        // stays on the cached static shadow path. Called ONCE per entry at capture.
        // C4d/Deliverable-A: LIVE record categories split by whether they are a DEFINITE mover or
        // only an AMBIGUOUS "live" record that may in fact be static. NPC/Creature (+clones) animate
        // their skeletons every frame — always dynamic. Activators AND Doors are ambiguous: a door's
        // swing is an ENGINE-applied 90° transform rotation (no controller — the game hardcodes it
        // for any non-teleport door; scripted rotate/playgroup likewise mutate the transform, and no
        // vanilla door actually scripts one). So a door has no active transform controller →
        // hasTransformAnim=false → it takes the STATIC path: shut/teleport doors stay cached, while a
        // swinging door's panel center arcs far past the host move-eps and re-renders via the
        // caster-moved epoch bump that same frame. Same story as a still hammock / fixed lantern.
        enum class LiveKind { None, Mover, Ambiguous };
        LiveKind referenceLiveKind(const NI::ObjectNET* obj) {
            const void* ref = obj->getTes3Reference(/*searchParents=*/true);
            if (!ref) return LiveKind::None;
            const void* base = *reinterpret_cast<void* const*>(
                static_cast<const char*>(ref) + 0x28);
            if (!base) return LiveKind::None;
            const uint32_t t = *reinterpret_cast<const uint32_t*>(
                static_cast<const char*>(base) + 0x4);
            if (t == '_CPN' /*NPC*/ || t == 'CCPN' /*NPCClone*/
             || t == 'AERC' /*Creature*/ || t == 'CERC' /*CreatureClone*/) { return LiveKind::Mover; }
            if (t == 'ITCA' /*Activator*/ || t == 'ROOD' /*Door*/) { return LiveKind::Ambiguous; }
            return LiveKind::None;
        }

        // Deliverable A source-mover: does this node — or an ancestor within its OWN object
        // hierarchy — carry an ACTIVE transform-animating controller? referenceLiveKind is a
        // coarse TES3 record-TYPE flag (every Activator/Door is "live"), but the shadow atlas
        // debug view (F12 12) showed most of those never move: a fixed hammock, a still lantern
        // whose only controller flips its flame texture. NI's controller taxonomy is closed —
        // exactly four controllers mutate a node's TRANSFORM (Keyframe / Path / LookAt / Roll);
        // UV/Flip/Vis/Alpha/Material/Color/Morpher/Particle controllers never touch it. So a node
        // driven by an ACTIVE transform controller genuinely moves (→ dynamic shadow tile); a
        // fixture with only a flame-texture controller reads as static (→ cached tile). Walks up
        // to the node owning the TES3 reference (object root); parents above it are shared
        // cell/world scene nodes. Called ONCE per entry at capture, like referenceLiveKind.
        bool hasTransformAnim(const NI::ObjectNET* obj) {
            for (const NI::ObjectNET* node = obj; node; ) {
                for (const NI::TimeController* c = node->controllers; c; c = c->nextController) {
                    if ((c->flags & NI::TimeControllerFlags::Active) == 0) continue;
                    if (c->isInstanceOfType(NI::RTTIStaticPtr::NiKeyframeController)
                     || c->isInstanceOfType(NI::RTTIStaticPtr::NiPathController)
                     || c->isInstanceOfType(NI::RTTIStaticPtr::NiLookAtController)
                     || c->isInstanceOfType(NI::RTTIStaticPtr::NiRollController)) {
                        return true;
                    }
                }
                // Stop once the node owning the TES3 reference (object root) is checked — parents
                // above it are shared cell/world nodes not specific to this reference.
                bool atRoot = false;
                for (const NI::ExtraData* ed = node->extraData; ed; ed = ed->next) {
                    if (ed->isOfType(NI::RTTIStaticPtr::TES3ObjectExtraData)) { atRoot = true; break; }
                }
                if (atRoot) break;
                if (!node->isInstanceOfType(NI::RTTIStaticPtr::NiAVObject)) break;
                node = static_cast<const NI::AVObject*>(node)->parentNode;
            }
            return false;
        }

        // Offscreen shadow-caster PRE-DISTANCE predicate — a byte-for-byte mirror of the filter in
        // renderprocess.cpp buildGeometryDrawLists' offscreen re-emit loop, MINUS the per-frame parts
        // (distance, visible-set, suppressedFrame, want-flags), which the consumer keeps. It reads only
        // capture-time classification fields, so membership changes only when those are (re)computed.
        // Keep in exact sync with that loop; a mesh that is a candidate there MUST be a candidate here.
        bool isMoverCandidate(const CachedGeometry& e) {
            if (e.isSky || e.isFP) return false;
            const bool isMM = !e.isLandscape && !e.blendEnable && e.d3dTexture
                              && (e.d3dDark || e.d3dDetail || e.d3dGlow);
            if (e.isSkinned) {
                return !(e.skinnedUnsupported || e.numBones == 0);
            }
            if (isMM) {
                return true;
            }
            if (e.isLive && !e.blendEnable && !e.isLandscape) {
                if (!e.d3dTexture && e.isPickRoot) return false;   // collision proxies only
                return true;
            }
            return false;
        }

        // Add/drop the key from the candidate set to match the entry's current classification.
        // Called wherever an entry's classification fields are (re)computed.
        void updateMoverMembership(uint32_t key, const CachedGeometry& e) {
            if (isMoverCandidate(e)) g_moverCandidates.insert(key);
            else                     g_moverCandidates.erase(key);
        }

        void visitGeometry(NI::TriBasedGeometry* geom, bool inCharacter) {
            auto* data = geom->getModelData().get();
            if (!data) return;

            const uint32_t key = reinterpret_cast<uint32_t>(geom);

            // Skin state: a valid SkinInstance with SkinData + bone array.
            NI::SkinInstance* si = geom->skinInstance.get();
            NI::SkinData*     sd = si ? si->skinData.get() : nullptr;
            const bool sk = (si && sd && si->bones);

            ++g_visitedThisFrame;

            auto it = g_cache.find(key);
            if (it != g_cache.end() && it->second.dataPtr != data) {
                // Recycled NiTriShape address (or a live setModelData swap): the entry
                // describes a different mesh — drop it and rebuild through the fresh path.
                releaseEntry(it->second);
                g_cache.erase(it);
                g_geomRefs.erase(key);   // key leaving the cache → drop the engine ref
                it = g_cache.end();
            }
            if (it == g_cache.end()) {
                auto& e = g_cache[key];
                // Take the engine reference the moment the key enters the cache. `geom` is provably
                // alive here (we are dereferencing it), and holding this ref is what makes every LATER
                // deref of this key safe — including ensureLive()'s, on an offscreen key, frames after
                // the engine dropped the object. Released only where the key leaves the cache.
                g_geomRefs[key] = geom;
                e.vb[0] = e.vb[1] = nullptr; e.ib = nullptr; e.writeSlot = 0;
                e.numBones = 0; e.skinnedUnsupported = false;
                e.dataPtr = data;
                // Material first: uploadEntry reads the captured map UV sets (baseUV/
                // darkUV/detailUV/glowUV, set here) to size the VB's UV-set count.
                extractMaterial(e, geom);
                if (sk) {
                    buildSkinnedVB(e, geom, data, si, sd);          // static
                    if (!e.skinnedUnsupported) buildBonePalette(e, geom, si, sd);
                } else {
                    uploadEntry(e, geom, data, key);
                }
                buildD3DTransform(e.worldTransformD3D, geom);       // bounds center
                e.dynamicHint = (sk || inCharacter) ? 4 : 0;
                e.lastFrame = g_frame;
                e.isLandscape = g_walkingLandscape;
                e.isPickRoot = g_walkingPick;
                // C4d: inCharacter is the cheap verdict (full-walk path); the reference
                // walk is authoritative and also covers the ensureLive lazy-capture path
                // (which passes inCharacter=false) — an NPC's equipment resolves to the
                // NPC reference either way. Sky/landscape never have a TES3 reference.
                // Deliverable A: classify the LIVE category, then decide `animated` = does this part
                // actually MOVE. Character parts (inCharacter — incl. MW's RIGID hair/neck/limbs
                // attached to animated bones) and definite-mover records (NPC/Creature/Door) are
                // movers by construction → animated. Only an Activator is ambiguous, so ONLY it pays
                // the transform-controller walk (silt strider animates → dyn; still hammock → static).
                const LiveKind kind = (g_walkingSky || g_walkingLandscape)
                                      ? LiveKind::None
                                      : (inCharacter ? LiveKind::Mover : referenceLiveKind(geom));
                e.isLive   = (kind != LiveKind::None);
                e.animated = (kind == LiveKind::Mover)
                          || (kind == LiveKind::Ambiguous && hasTransformAnim(geom));
                e.isSky = g_walkingSky;
                e.isFP = g_walkingFP;
                if (g_walkingSky) e.skyOrder = g_skyVisitCounter++;  // SK2 back-to-front key
                e.mirrored = computeMirrored(e);                    // winding flip for depth/shadow
            } else {
                auto& e = it->second;
                e.lastFrame = g_frame;
                e.isLandscape = g_walkingLandscape;
                e.isPickRoot = g_walkingPick;
                e.isSky = g_walkingSky;
                e.isFP = g_walkingFP;
                if (g_walkingSky) e.skyOrder = g_skyVisitCounter++;  // SK2 back-to-front key
                bool transformChanged = true;   // skinned/inCharacter paths always re-derive
                if (sk) {
                    // Static skinned VB: rebuild only on revision / skin-state change.
                    if (data->revisionID != e.revisionID || !e.isSkinned) {
                        extractMaterial(e, geom);
                        buildSkinnedVB(e, geom, data, si, sd);
                    }
                    if (!e.skinnedUnsupported) buildBonePalette(e, geom, si, sd);  // per frame
                    buildD3DTransform(e.worldTransformD3D, geom);            // bounds center
                    e.dynamicHint = 4;
                } else {
                    // Non-skinned upload decision:
                    //  - Sky (dome + SK2): the host consumes only STATIC vertex data — it colours the
                    //    dome geometrically (skyZenith gradient; baked vcol ignored) and the SK2 shapes'
                    //    orbit + weather/night fade ride SkyDrawWire per-frame (world transform +
                    //    matColor/matAlpha), NOT the VB. MW bumps the sky's revisionID EVERY frame
                    //    (atmosphere gradient / star fade), but the VB never meaningfully changes, so
                    //    re-uploading it is a pure per-frame IPC tax (~0.85ms blocking round-trip for a
                    //    ~5KB blob). Refresh only the cheap CPU-side MATERIAL that buildSkyDrawList reads
                    //    (e.matColor/e.matAlpha) and skip the VB re-upload — the mesh shipped once on
                    //    first capture (new-entry branch).
                    //  - Everything else: re-extract + re-upload on revision/skin change.
                    if (g_walkingSky) {
                        extractMaterial(e, geom);
                        // SK4 live vertex colour: MW rebakes sky vcols in place (cloud weather/
                        // time-of-day tint, star fade) but the VB + wire capture shipped once, so
                        // vcol-routed shapes froze at capture-time colours (the "F11 off/on
                        // catches up" bug — the toggle evicts + recaptures). Hash the live vcols
                        // of vColSource==2 shapes (the only ones whose real vcol ships — see the
                        // Tier 2a rule in uploadEntry) and re-run the full upload on change,
                        // which re-bakes the VB, re-ships the wire capture, and re-bases the
                        // SK3 UV baselines (offset correctly returns 0 for the fresh bake).
                        if (e.hasVertexColor && e.vColSource == 2 && data->color) {
                            const uint32_t liveHash = hashVertexColors(
                                data->color, (uint32_t)data->getActiveVertexCount());
                            if (liveHash != e.skyVcolHash) {
                                static uint32_t s_sk4Logged = 0;
                                if (s_sk4Logged < 8) {
                                    ++s_sk4Logged;
                                    LOG::logline(">> [sk4] sky vcol changed: key=%08X tex=%s vc=%u — re-upload",
                                                 key, e.textureName ? e.textureName : "(none)",
                                                 (unsigned)data->getActiveVertexCount());
                                }
                                uploadEntry(e, geom, data, key);
                            }
                        }
                        // SK3 cloud scroll: MW rebakes the cloud shape's UVs every frame (the
                        // per-frame sky revisionID bump), but the VB shipped once — derive the
                        // uniform scroll offset from vertex 0 instead; buildSkyDrawList ships it
                        // and sky.vert adds it back. One-shot uniformity check: the last vertex
                        // must have moved by the same delta (MW shifts the whole set together).
                        if (const auto* uvs = data->textureCoords) {
                            e.skyUVOffset[0] = uvs[0].x - e.skyBaseUV[0];
                            e.skyUVOffset[1] = uvs[0].y - e.skyBaseUV[1];
                            static bool s_sk3Checked = false;
                            if (!s_sk3Checked
                                && (std::fabs(e.skyUVOffset[0]) > 0.01f || std::fabs(e.skyUVOffset[1]) > 0.01f)) {
                                s_sk3Checked = true;
                                const uint32_t last = data->getActiveVertexCount()
                                    ? (uint32_t)data->getActiveVertexCount() - 1u : 0u;
                                const float dxL = uvs[last].x - e.skyBaseUVLast[0];
                                const float dyL = uvs[last].y - e.skyBaseUVLast[1];
                                LOG::logline(">> [sk3] cloud UV scroll live: v0 offset=(%.4f,%.4f) vLast delta=(%.4f,%.4f)%s",
                                             e.skyUVOffset[0], e.skyUVOffset[1], dxL, dyL,
                                             (std::fabs(dxL - e.skyUVOffset[0]) > 0.001f
                                              || std::fabs(dyL - e.skyUVOffset[1]) > 0.001f)
                                                 ? "  !! NON-UNIFORM — offset transport is wrong for this shape" : "");
                            }
                        }
                    } else if ((data->revisionID != e.revisionID) || e.isSkinned) {
                        extractMaterial(e, geom);
                        uploadEntry(e, geom, data, key);
                    } else if (e.d3dTexture == nullptr && e.textureName != nullptr) {
                        // UNRESOLVED-TEXTURE RETRY. getDX9Texture returns null while MW has not yet
                        // uploaded the NiSourceTexture (rendererData null) — the texture streams in
                        // lazily, on MW's schedule. A STATIC shape (a roof, a wall) never bumps
                        // revisionID, so without this the entry captured mid-stream keeps
                        // d3dTexture=null for the rest of the session and the host draws it
                        // untextured/white. Hoisting the walk to BeginScene(0) widened exactly this
                        // race, which is why it shows on this branch.
                        //
                        // textureName != nullptr is what separates "should have a texture, didn't
                        // resolve" from "legitimately textureless" (both null) — the latter is a real
                        // class we must NOT churn on (see the textureless-visual-drop regression), and
                        // it never enters this branch. Self-limiting: an entry retries only until its
                        // texture lands, so the steady-state cost is zero. Material only — the VB is
                        // unaffected by texture residency, so no re-upload and no wire traffic.
                        extractMaterial(e, geom);
                    }
                    // Transform (orbit for sky) + dynamic hint, shared by sky and opaque.
                    if (inCharacter) {
                        buildD3DTransform(e.worldTransformD3D, geom);
                        e.dynamicHint = 4;
                    } else {
                        float newTransform[16];
                        buildD3DTransform(newTransform, geom);
                        if (memcmp(newTransform, e.worldTransformD3D, sizeof(newTransform)) != 0) {
                            e.dynamicHint = 4;
                            memcpy(e.worldTransformD3D, newTransform, sizeof(newTransform));
                        } else {
                            transformChanged = false;   // mirrored can't have flipped
                            if (e.dynamicHint > 0) {
                                --e.dynamicHint;
                            }
                        }
                    }
                }
                // Winding flip for depth/shadow. The sign is a pure function of the
                // world/bone transform, so recompute only when that changed (skinned
                // and inCharacter paths re-derive every frame; the static path's
                // memcmp above proves it identical).
                if (transformChanged) {
                    e.mirrored = computeMirrored(e);
                }
            }
            // Mover-candidate membership: the entry (new or refreshed) is now fully classified.
            // Covers full-walk capture/refresh AND ensureLive's first-sight lazy capture (which
            // routes through here). The ensureLive REFRESH path re-checks separately on reclassify.
            auto mit = g_cache.find(key);
            if (mit != g_cache.end()) updateMoverMembership(key, mit->second);
        }

        // bypassCull skips the entry's own app-cull check. Used for a NiSwitchNode's
        // active child: switchIndex already selected it, and a menu-frame cull pass may
        // have transiently app-culled it; the cache frustum-culls later anyway.
        void walk(NI::AVObject* av, bool inCharacter = false, bool bypassCull = false) {
            if (!av) return;
            if (!bypassCull && av->getAppCulled()) return;

            // W1.5 active-cell gate: skip anything whose world bound lies entirely
            // beyond the gate sphere — a NiNode prunes its whole subtree (per-cell
            // containers at the roots' direct children), a leaf prunes itself (cell
            // bounds are ~half-a-diagonal fat, so a kept edge cell still has a far
            // half worth trimming). Distance-only (not frustum) so panning never
            // churns capture. The sky walk is exempt — sky shapes ride orbit
            // transforms unrelated to eye distance.
            if (g_gateThisFrame && !g_walkingSky && !g_walkingFP) {
                const float dx = av->worldBoundOrigin.x - g_gateEye[0];
                const float dy = av->worldBoundOrigin.y - g_gateEye[1];
                const float dz = av->worldBoundOrigin.z - g_gateEye[2];
                const float reach = g_gateRadius + av->worldBoundRadius;
                if (dx * dx + dy * dy + dz * dz > reach * reach) {
                    ++g_gateSkipsThisFrame;
                    return;
                }
            }

            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiTriBasedGeom)) {
                // NiParticles ARE NiTriBasedGeom in MW's RTTI, so they land here — but
                // visitGeometry would capture only the particle CENTERS as a degenerate mesh
                // (the billboarded quads are generated by MW's particle renderer at draw time).
                // In the FP walk, collect them for client-side billboarding instead of shipping
                // the useless point-mesh. World particles keep the captureAlphaDraw path.
                if (g_walkingFP && av->isInstanceOfType(NI::RTTIStaticPtr::NiParticles)) {
                    auto* pgeom = static_cast<NI::Particles*>(av);
                    // Register the base-map texture HERE: the FP walk intercepts particles before
                    // the NiGeometry branch (and they aren't cached → no extractMaterial), so
                    // without this the reverse map never learns the flame texture and the host
                    // draws it white/pale until a 3rd-person world render registers it for us.
                    if (auto* ps = reinterpret_cast<NI::PropertyState*>(pgeom->propertyState)) {
                        if (ps->texture) {
                            const auto* bm = ps->texture->getBaseMap();
                            if (bm && bm->texture) registerTextureName(bm->texture.get());
                        }
                    }
                    g_fpParticleSystems.push_back(pgeom);
                    return;
                }
                visitGeometry(static_cast<NI::TriBasedGeometry*>(av), inCharacter);
                return;
            }

            // AT3 captured-alpha: NiParticles (chimney smoke, candle/camp flames) derive from
            // NiGeometry but NOT NiTriBasedGeom, so visitGeometry never sees them and their GPU
            // texture never lands in g_textureNameMap. Register the base-map name here (leaf,
            // registration-only — no capture; MW still simulates + billboards the particles) so
            // the client's captureAlphaDraw can resolve rs.texture -> a bindless slot.
            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiGeometry)) {
                auto* geom = static_cast<NI::Geometry*>(av);
                auto* ps = reinterpret_cast<NI::PropertyState*>(geom->propertyState);
                if (ps && ps->texture) {
                    const auto* baseMap = ps->texture->getBaseMap();
                    if (baseMap && baseMap->texture) {
                        registerTextureName(baseMap->texture.get());
                    }
                }
                return;
            }

            // NiSwitchNode (e.g. "Glow in the Dark"'s NightDaySwitch): only the child
            // at switchIndex is displayed. Walk that child EXPLICITLY rather than
            // iterating all children and trusting per-child app-cull — in menu frames
            // the engine's cull pass can leave the inactive (day) variant un-culled and
            // the active (night) one culled, so the generic NiNode path below would
            // capture the wrong variant (day window showing while a menu is open).
            // switchIndex < 0 means no active child. Bypass the active child's own
            // app-cull so a transient menu cull can't drop it.
            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiSwitchNode)) {
                auto* sw = static_cast<NI::SwitchNode*>(av);
                const int idx = sw->switchIndex;
                if (idx >= 0 && (size_t)idx < sw->children.getEndIndex()) {
                    walk(sw->children.at(idx).get(), inCharacter, true);
                }
                return;
            }

            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) {
                auto* node = static_cast<NI::Node*>(av);
                const auto count = node->children.getEndIndex();

                // FP1c "gravity" tick: the erect mod hangs a NiLookAtController on the candle-
                // flame parent (NiBSAnimationNode) that aims it at an up-∞ target → the flame
                // stays world-vertical. That controller is a scene-graph node controller, only
                // ticked by the arm-scene Update traversal — which FP suppression (armRoot
                // appCulled) skips, so it stays frozen (flame tilted) until an F11 seam cycle
                // lets one real render re-seed it. Drive it ourselves here: tick the look-at
                // (writes node LOCAL rotation), then refresh the node world so the billboard
                // child below reads a vertical parent up. FP-walk-only; idempotent (static
                // target). Runs at the controller-owning node, one recursion level above the
                // NiBillboardNode handled below.
                if (g_walkingFP) {
                    for (NI::TimeController* c = node->controllers.get(); c;
                         c = c->nextController.get()) {
                        if (c->isInstanceOfType(NI::RTTIStaticPtr::NiLookAtController)) {
                            c->update(0.0f);
                            node->update(0.0f, false, false);
                            break;
                        }
                    }
                }

                // FP1c billboard re-face: MW re-orients NiBillboardNode quads (held candle
                // flame, enchant glow) toward the camera during the arm cull/render pass —
                // which FP suppression skips, so the quad's WORLD rotation goes stale
                // (edge-on → invisible even though the alpha part still captures & draws).
                // During the FP walk only, re-face the billboard against the arm camera and
                // re-derive its children's world transforms so visitGeometry (below) reads
                // the corrected orientation.
                //
                // WORLD billboards need the SAME re-face, and NOT because of suppression — it is
                // a walk-ORDER problem that predates it. This walk runs at BeginScene(0); MW's
                // world cull pass, where NiBillboardNode re-faces itself, runs later in scene 0.
                // The host draws from what we capture here, so we were always shipping the
                // PREVIOUS frame's facing — invisible while still, plainly wrong as soon as the
                // camera moves (flames edge-on / not tracking). Proven at suppression level 1,
                // where MW still traverses objRoot and re-faces normally, and they were stale
                // anyway. First person never showed it because FP1c above already re-faces.
                //
                // So: gate on the host owning the world, not on the suppression level. Doing it
                // early is harmless when MW re-faces again later — the computation is the same
                // one, we just need our capture to see the result rather than precede it.
                const bool refaceWorldBillboards =
                    !g_walkingFP && !g_walkingSky && RenderProcess::ownsOpaqueWorld();
                if ((g_walkingFP || refaceWorldBillboards) &&
                    node->isInstanceOfType(NI::RTTIStaticPtr::NiBillboardNode)) {
                    NI::Camera* faceCam = g_walkingFP ? MWBridge::get()->getArmCamera()
                                                      : MWBridge::get()->getWorldCamera();
                    if (NI::Camera* armCam = faceCam) {
                        auto* bb = static_cast<NI::BillboardNode*>(node);
                        const NI::Point3& wt = bb->worldTransform.translation;
                        // Guard the erect-mod up-∞ LookUpTarget (z≈3.4e38): never re-face or
                        // recompute a node whose world position is non-finite / astronomical.
                        if (std::isfinite(wt.x) && std::isfinite(wt.y) &&
                            std::isfinite(wt.z) && std::fabs(wt.z) < 1.0e30f) {
                            // Match the engine's order: recompute the billboard's own world
                            // from its parent FIRST (the parent carries the erect-mod
                            // NiLookAtController "gravity" → a vertical world up), THEN
                            // rotateToCamera. RotateAboutUp mode preserves the world up, so
                            // feeding it the fresh parent-derived up is what keeps the flame
                            // vertical. Without this it yawed around a stale up (flame tilted
                            // until an F11 seam cycle let a real engine render re-seed it).
                            // bUpdateChildren=false — children are re-derived below.
                            bb->update(0.0f, false, false);
                            bb->rotateToCamera(armCam);
                            // Re-derive children off the freshly-rotated billboard world
                            // (child.world = billboard.world * child.local). update() on the
                            // CHILDREN only — calling it on the billboard itself would rebuild
                            // its world from parent*local and clobber the facing just set.
                            for (size_t i = 0; i < count; ++i) {
                                if (NI::AVObject* child = node->children.at(i).get())
                                    child->update(0.0f, false, true);
                            }
                        }
                    }
                }

                // If not already flagged, check whether any direct child is a skinned
                // mesh — if so, the whole subtree is a character (NPC/creature) and all
                // non-skinned geometry within it (bone-attached equipment, head, etc.)
                // must be treated as dynamic regardless of per-frame transform delta.
                // The O(children) RTTI scan runs once per node; the verdict is cached
                // (g_charNodeVerdict, cleared at each eviction sweep — see declaration).
                bool isCharNode = inCharacter;
                if (!isCharNode) {
                    const uint32_t nodeKey = reinterpret_cast<uint32_t>(node);
                    auto vit = g_charNodeVerdict.find(nodeKey);
                    if (vit != g_charNodeVerdict.end()) {
                        isCharNode = vit->second;
                    } else {
                        for (size_t i = 0; i < count; ++i) {
                            NI::AVObject* child = node->children.at(i).get();
                            if (!child) continue;
                            if (child->isInstanceOfType(NI::RTTIStaticPtr::NiTriBasedGeom)) {
                                if (static_cast<NI::TriBasedGeometry*>(child)->skinInstance.get()) {
                                    isCharNode = true;
                                    break;
                                }
                            }
                        }
                        g_charNodeVerdict.emplace(nodeKey, isCharNode);
                    }
                }
                for (size_t i = 0; i < count; ++i)
                    walk(node->children.at(i), isCharNode);
            }
        }

    // ---- Reflection moon support (scene-graph-sourced) -----------------------------
    // The engine records a moon in recordSky only when it draws it for the MAIN camera,
    // so a moon up but outside the main frustum (looking away/down) never reaches the
    // water reflection that way. These helpers materialize the moon billboards straight
    // from the live scene graph, gated by the engine's own appCulled, so the reflection
    // can draw them frustum-independently.

    // Per-moon-shape D3D9 geometry keyed on NiGeometry*. Created once; vertex data is
    // refreshed each call (4 verts — negligible) so phase/fade/orientation updates track.
    // D3DPOOL_MANAGED so the buffers survive device reset without an explicit release.
    struct MoonGeom {
        IDirect3DVertexBuffer9* vb;
        IDirect3DIndexBuffer9*  ib;
        uint32_t vertCount, triCount;
    };
    std::unordered_map<uint32_t, MoonGeom> g_moonGeom;

    // NiAlphaProperty blend-function index (Gamebryo order) -> D3DBLEND.
    D3DBLEND niBlendToD3D(unsigned int ni) {
        switch (ni) {
            case 0:  return D3DBLEND_ONE;
            case 1:  return D3DBLEND_ZERO;
            case 2:  return D3DBLEND_SRCCOLOR;
            case 3:  return D3DBLEND_INVSRCCOLOR;
            case 4:  return D3DBLEND_DESTCOLOR;
            case 5:  return D3DBLEND_INVDESTCOLOR;
            case 6:  return D3DBLEND_SRCALPHA;
            case 7:  return D3DBLEND_INVSRCALPHA;
            case 8:  return D3DBLEND_DESTALPHA;
            case 9:  return D3DBLEND_INVDESTALPHA;
            case 10: return D3DBLEND_SRCALPHASAT;
            default: return D3DBLEND_ONE;
        }
    }

    // (Re)create + refresh the DepthVertex-layout VB/IB for one moon shape. Returns
    // false (shape skipped) on degenerate geometry or allocation failure.
    bool materializeMoonShape(NI::TriBasedGeometry* geom, MGE::GeometryCache::MoonShapeDraw& out) {
        auto* data = static_cast<NI::TriBasedGeometryData*>(geom->getModelData().get());
        if (!data) return false;
        const uint32_t vc  = static_cast<uint32_t>(data->getActiveVertexCount());
        const uint32_t tc  = static_cast<uint32_t>(data->getActiveTriangleCount());
        const auto*    mv  = data->vertex;
        const auto*    tri = data->getTriList();
        if (!vc || !tc || !mv || !tri) return false;

        const uint32_t key = reinterpret_cast<uint32_t>(geom);
        auto& g = g_moonGeom[key];

        if (!g.vb || g.vertCount != vc) {
            if (g.vb) { g.vb->Release(); g.vb = nullptr; }
            if (FAILED(g_device->CreateVertexBuffer(vc * MGE::GeometryCache::kVBStride, 0,
                    MGE::GeometryCache::kVBFVF, D3DPOOL_MANAGED, &g.vb, nullptr)))
                return false;
        }
        if (!g.ib || g.triCount != tc) {
            if (g.ib) { g.ib->Release(); g.ib = nullptr; }
            if (FAILED(g_device->CreateIndexBuffer(tc * 6, 0, D3DFMT_INDEX16,
                    D3DPOOL_MANAGED, &g.ib, nullptr)))
                return false;
        }
        g.vertCount = vc;
        g.triCount  = tc;

        const auto* nrm = data->normal;
        const auto* col = data->color;
        const auto* uv  = data->textureCoords;   // set 0 (moons are single-UV)
        void* vbData = nullptr;
        if (FAILED(g.vb->Lock(0, 0, &vbData, 0))) return false;
        auto* dst = static_cast<DepthVertex*>(vbData);
        for (uint32_t i = 0; i < vc; ++i) {
            dst[i].x = mv[i].x; dst[i].y = mv[i].y; dst[i].z = mv[i].z;
            if (nrm) { dst[i].nx = nrm[i].x; dst[i].ny = nrm[i].y; dst[i].nz = nrm[i].z; }
            else     { dst[i].nx = 0.0f;     dst[i].ny = 0.0f;     dst[i].nz = 1.0f; }
            dst[i].color = col ? *reinterpret_cast<const DWORD*>(&col[i]) : 0xFFFFFFFFu;
            if (uv) { dst[i].u = uv[i].x; dst[i].v = uv[i].y; }
            else    { dst[i].u = 0.0f;    dst[i].v = 0.0f; }
        }
        g.vb->Unlock();

        void* ibData = nullptr;
        if (SUCCEEDED(g.ib->Lock(0, 0, &ibData, 0))) {
            memcpy(ibData, tri, tc * 6);
            g.ib->Unlock();
        }

        out.vb        = g.vb;
        out.ib        = g.ib;
        out.vertCount = vc;
        out.triCount  = tc;
        buildD3DTransform(out.worldTransform, geom);
        return true;
    }

    // Recursively collect up to 2 drawable moon shapes under av, respecting per-node
    // appCulled (a full moon has no dark-side cutout, so its Shadow Node is culled).
    void collectMoonShapes(NI::AVObject* av, MGE::GeometryCache::MoonShapeDraw* out, int& count) {
        if (!av || count >= 2) return;
        if (av->getAppCulled()) return;

        if (av->isInstanceOfType(NI::RTTIStaticPtr::NiTriBasedGeom)) {
            auto* geom = static_cast<NI::TriBasedGeometry*>(av);
            MGE::GeometryCache::MoonShapeDraw d = {};
            if (!materializeMoonShape(geom, d)) return;

            d.texture   = nullptr;
            d.srcBlend  = D3DBLEND_SRCALPHA;
            d.destBlend = D3DBLEND_INVSRCALPHA;
            auto* ps = reinterpret_cast<NI::PropertyState*>(geom->propertyState);
            if (ps) {
                if (ps->texture) {
                    auto* baseMap = ps->texture->getBaseMap();
                    if (baseMap && baseMap->texture) {
                        auto* tex = baseMap->texture.get();
                        if (tex->isInstanceOfType(NI::RTTIStaticPtr::NiSourceTexture))
                            d.texture = getDX9Texture(tex);
                    }
                }
                if (ps->alpha) {
                    const unsigned short f = ps->alpha->flags;
                    d.srcBlend  = static_cast<unsigned char>(niBlendToD3D(
                        (f & NI::AlphaProperty::SRC_BLEND_MASK)  >> NI::AlphaProperty::SRC_BLEND_POS));
                    d.destBlend = static_cast<unsigned char>(niBlendToD3D(
                        (f & NI::AlphaProperty::DEST_BLEND_MASK) >> NI::AlphaProperty::DEST_BLEND_POS));
                }
            }
            if (!d.texture) return;   // no base map -> nothing to draw
            // The dark-side cutout lives under the moon root's 'Shadow Node'; the lit disc
            // under 'Moon Node'. Discriminate by that parent name — robust, since the disc
            // can share the cutout's alpha-blend mode (blend alone is ambiguous).
            d.isMoonShadow = false;
            if (av->parentNode) {
                const char* pn = av->parentNode->getName();
                if (pn && std::strcmp(pn, "Shadow Node") == 0) d.isMoonShadow = true;
            }
            out[count++] = d;
            return;
        }

        if (av->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) {
            auto* node = static_cast<NI::Node*>(av);
            const auto n = node->children.getEndIndex();
            for (size_t i = 0; i < n && count < 2; ++i)
                collectMoonShapes(node->children.at(i).get(), out, count);
        }
    }

    int buildMoonDrawListImpl(NI::Node* root, MGE::GeometryCache::MoonShapeDraw out[2]) {
        if (!g_device || !root) return 0;
        if (root->getAppCulled()) return 0;   // moon is down / hidden by phase
        int count = 0;
        collectMoonShapes(root, out, count);
        // Draw the dark-side cutout before the disc.
        if (count == 2 && !out[0].isMoonShadow && out[1].isMoonShadow) {
            MGE::GeometryCache::MoonShapeDraw t = out[0]; out[0] = out[1]; out[1] = t;
        }
        return count;
    }

    }

    void init(IDirect3DDevice9* device) {
        g_device = device;

        // Vertex declaration for skinned VBs (SkinnedVertex, stride 56). The COLOR
        // element (offset 52) is read only by the FFE cache-skin color path when the
        // part uses vertex colour; the depth/shadow skinned VS don't declare it and
        // D3D9 ignores unread elements, so those passes are unaffected.
        if (!g_skinnedDecl && g_device) {
            static const D3DVERTEXELEMENT9 elems[] = {
                {0, 0,  D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION,     0},
                {0, 12, D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL,       0},
                {0, 24, D3DDECLTYPE_FLOAT4,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDWEIGHT,  0},
                {0, 40, D3DDECLTYPE_UBYTE4,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDINDICES, 0},
                {0, 44, D3DDECLTYPE_FLOAT2,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD,     0},
                {0, 52, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR,        0},
                D3DDECL_END()
            };
            g_device->CreateVertexDeclaration(elems, &g_skinnedDecl);
        }
    }

    IDirect3DVertexDeclaration9* skinnedDecl() {
        return g_skinnedDecl;
    }

    // ---- SK0: sky-takeover diagnostic (scene-graph inspect, NO capture) ----------
    // The sky is a proper NiNode subtree "skyRoot" — a sibling of worldRoot under the
    // World Scene Graph Root (the node that parents the cell roots + the camera root).
    // Reach it by climbing worldObjectRoot to the topmost ancestor and finding the
    // "skyRoot" child by name. SK0 only LOGS the subtree (name / cull / verts / blend /
    // alpha-test / texture) so we can confirm what the real walk would capture before
    // any host draw. No g_cache writes, no side effects. Gated + throttled by the caller.
    static NI::Node* findSkyRoot(void* dataHandler) {
        NI::AVObject* n = MGE::DataHandlerView::worldObjectRoot(dataHandler);
        if (!n) return nullptr;
        while (n->parentNode) n = n->parentNode;          // climb to World Scene Graph Root
        if (!n->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) return nullptr;
        auto* root = static_cast<NI::Node*>(n);
        const auto count = root->children.getEndIndex();
        for (size_t i = 0; i < count; ++i) {
            NI::AVObject* c = root->children.at(i).get();
            if (!c) continue;
            const char* nm = c->getName();
            if (nm && std::strcmp(nm, "skyRoot") == 0
                   && c->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) {
                return static_cast<NI::Node*>(c);
            }
        }
        return nullptr;
    }

    static void dumpSkyNode(NI::AVObject* av, int depth) {
        if (!av) return;
        const char* nm     = av->getName();
        const bool  culled = av->getAppCulled();
        if (av->isInstanceOfType(NI::RTTIStaticPtr::NiTriBasedGeom)) {
            auto* geom = static_cast<NI::TriBasedGeometry*>(av);
            auto* data = geom->getModelData().get();
            const uint32_t vc = data ? static_cast<uint32_t>(data->getActiveVertexCount()) : 0u;
            bool blend = false, atest = false; const char* texName = nullptr;
            int vcolSrc = -1;   // -1 = no VertexColorProperty
            auto* ps = reinterpret_cast<NI::PropertyState*>(geom->propertyState);
            if (ps) {
                if (ps->alpha) {
                    blend = (ps->alpha->flags & NI::AlphaProperty::ALPHA_MASK) != 0;
                    atest = (ps->alpha->flags & NI::AlphaProperty::TEST_ENABLE_MASK) != 0;
                }
                if (ps->texture) {
                    auto* baseMap = ps->texture->getBaseMap();
                    if (baseMap && baseMap->texture) {
                        auto* tex = baseMap->texture.get();
                        if (tex->isInstanceOfType(NI::RTTIStaticPtr::NiSourceTexture))
                            texName = static_cast<NI::SourceTexture*>(tex)->fileName;
                    }
                }
                if (ps->vertexColor) {
                    vcolSrc = static_cast<int>(ps->vertexColor->source);
                }
            }
            // SK4 diag: live vertex-0 colour (D3DCOLOR) — tracks MW's in-place sky vcol
            // rebake (cloud tint / star fade) across successive dumps.
            const uint32_t vcol0 = (data && data->color)
                ? *reinterpret_cast<const uint32_t*>(&data->color[0]) : 0u;
            LOG::logline("[SKY] %*sGEOM '%s' culled=%d verts=%u blend=%d atest=%d vcolsrc=%d vcol0=%08X tex=%s",
                depth * 2, "", nm ? nm : "(null)", culled ? 1 : 0, vc,
                blend ? 1 : 0, atest ? 1 : 0, vcolSrc, vcol0, texName ? texName : "(none)");
            return;
        }
        // [sky-lit] diag: the sky subtree's own NiAmbientLight — the FFP modulation
        // vanilla applies to lit amb+diff-vcol sky shapes (out = vcol · ambient).
        if (av->isInstanceOfType(NI::RTTIStaticPtr::NiAmbientLight)) {
            auto* li = static_cast<NI::Light*>(av);
            LOG::logline("[SKY] %*sLIGHT '%s' culled=%d dimmer=%.3f amb=(%.3f,%.3f,%.3f) diff=(%.3f,%.3f,%.3f)",
                depth * 2, "", nm ? nm : "(null)", culled ? 1 : 0, li->dimmer,
                li->ambient.r, li->ambient.g, li->ambient.b,
                li->diffuse.r, li->diffuse.g, li->diffuse.b);
            return;
        }
        LOG::logline("[SKY] %*sNODE '%s' culled=%d", depth * 2, "", nm ? nm : "(null)", culled ? 1 : 0);
        if (av->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) {
            auto* node = static_cast<NI::Node*>(av);
            const auto count = node->children.getEndIndex();
            for (size_t i = 0; i < count; ++i)
                dumpSkyNode(node->children.at(i).get(), depth + 1);
        }
    }

    // QPC millisecond clock for the [gc] heartbeat (same pattern as renderprocess's nowMs).
    static double gcNowMs() {
        static LARGE_INTEGER freq = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
        LARGE_INTEGER c; QueryPerformanceCounter(&c);
        return 1000.0 * (double)c.QuadPart / (double)freq.QuadPart;
    }

    // Raw-offset walk over the active mobile actors: fn(animData, headGeometry) for
    // every actor that has a head. MGE only pulls SharedSE, so the TES3 structs are
    // read by raw x86 offset, all validated against the MWSE headers:
    //   WorldController*        @ 0x7C67DC          (TES3WorldController.cpp:565)
    //   ->mobManager            +0x5C               (TES3WorldController.h:326)
    //   ->processManager        +0x24               (TES3MobManager.h:82)
    //   plannerCount / planners +0x58 / +0x5C[500]  (TES3MobManager.h:19-20)
    //   AIPlanner->mobileActor  +0x4                (TES3AIData.h:9)
    //   MobileActor vtbl getAnimationAttachment @ byte 0xCC (TES3MobileObject.h:167)
    //   AnimationData: headGeometry 0x2E8, headMorphTiming 0x2F4
    //   (TES3AnimationData.h:43-50)
    template <typename Fn>
    static void forEachActorHead(Fn&& fn) {
        const char* wc = *(const char* const*)0x7C67DC;
        if (!wc) return;
        const char* mobMgr = *(const char* const*)(wc + 0x5C);
        if (!mobMgr) return;
        const char* procMgr = *(const char* const*)(mobMgr + 0x24);
        if (!procMgr) return;
        const uint32_t plannerCount = *(const uint32_t*)(procMgr + 0x58);
        const char* const* planners = (const char* const*)(procMgr + 0x5C);
        const uint32_t n = plannerCount > 500u ? 500u : plannerCount;
        for (uint32_t i = 0; i < n; ++i) {
            const char* planner = planners[i];
            if (!planner) continue;
            const char* actor = *(const char* const*)(planner + 0x4);
            if (!actor) continue;
            using GetAnim = const char* (__thiscall*)(const void*);
            const void* const* vtbl = *(const void* const* const*)actor;
            const char* anim = ((GetAnim)vtbl[0xCC / 4])(actor);
            if (!anim) continue;
            auto* head = *(NI::Geometry* const*)(anim + 0x2E8);
            if (!head) continue;
            fn(anim, head);
        }
    }

    // Forge lip/blink fix: MW applies the head morph (talk mouth-flap + blink) to
    // data->vertex only from ITS OWN display of the head — which the msoc owned-display
    // skip suppresses under Forge takeover, freezing every face at rest pose while the
    // per-actor clocks keep advancing. AnimationData::headMorphTiming is already mapped
    // into the morpher's talk/blink key windows, so drive the apply ourselves each
    // frame: update(timing) evaluates the morph weights at that time, onPreDisplay()
    // writes the blended verts. Both calls are required — update alone never touches
    // verts. The clock MUST be headMorphTiming, not the global sim timestamp (~3.7M),
    // which evaluates outside the key range and zeroes the weights.
    // Runs before the walks/ensureLive so the same frame's capture sees moved verts.
    static void driveHeadMorphs() {
        forEachActorHead([](const char* anim, NI::Geometry* head) {
            const float timing = *(const float*)(anim + 0x2F4);
            for (NI::TimeController* c = head->controllers; c; c = c->nextController) {
                if (!c->isOfType(NI::RTTIStaticPtr::NiGeomMorpherController)) continue;
                c->vTable.asController->update(c, timing);
                c->vTable.asController->onPreDisplay(c);
            }
        });
    }

    // The full per-frame refresh walk (objects + pick + landscape) over this frame's
    // stored roots. On live-draw-build frames this is skipped in onFrameReady and only
    // runs on demand (ensureFullWalk) when a frame has no classify result to drive
    // ensureLive. Accumulates its own [gc] phase timings; stamps g_walkRanFrame.
    static void runRefreshWalks() {
        const double t0 = gcNowMs();
        {
            MGE_ZoneScopedN("GeomCache:walkObjects");
            walk(g_objRoot);
        }
        const double tObj = gcNowMs();
        {
            MGE_ZoneScopedN("GeomCache:walkPickObjects");
            g_walkingPick = true;
            walk(g_pickRoot);
            g_walkingPick = false;
        }
        const double tPick = gcNowMs();
        {
            MGE_ZoneScopedN("GeomCache:walkLandscape");
            g_walkingLandscape = true;
            walk(g_landRoot);
            g_walkingLandscape = false;
        }
        g_gcAccum.obj  += tObj - t0;
        g_gcAccum.pick += tPick - tObj;
        g_gcAccum.land += gcNowMs() - tPick;
        g_walkRanFrame = g_frame;
    }

    void onFrameReady(void* dataHandler, const float* gateEye, float gateRadius, bool liveDrawBuild) {
        if (!g_device || !dataHandler) return;
        if (!Configuration.UseSceneGraphSnapshot) return;
        MGE_ZoneScopedN("GeometryCache::onFrameReady");

        ++g_frame;
        // Accumulate the PREVIOUS frame's per-frame counters and uploads before
        // resetting: ensureLive runs AFTER onFrameReady returns (inside
        // buildFrustumVisibleSet), so its counts/uploads finalize between calls.
        // One frame of skew is irrelevant to a 300-frame average.
        g_gcAccum.visited  += g_visitedThisFrame;
        g_gcAccum.gateSkip += g_gateSkipsThisFrame;
        g_gcAccum.live     += g_liveRefreshThisFrame;
        g_gcAccum.cap      += g_liveCaptureThisFrame;
        g_uploadedInterval += g_uploadedThisFrame;
        g_uploadedThisFrame = 0;
        ++g_walkSerial;
        g_visitedThisFrame = 0;
        g_gateSkipsThisFrame = 0;
        g_liveRefreshThisFrame = 0;
        g_liveCaptureThisFrame = 0;
        // Arm the active-cell gate for this walk; the eye/radius persist for the
        // eviction hysteresis even on later ungated frames (see declarations).
        g_gateThisFrame = (gateEye != nullptr && gateRadius > 0.0f);
        if (g_gateThisFrame) {
            g_gateEye[0] = gateEye[0];
            g_gateEye[1] = gateEye[1];
            g_gateEye[2] = gateEye[2];
            g_gateRadius = gateRadius;
        }
        // Emissive-boost light data. SceneGraph::onFrameReady() ran earlier this frame
        // (distantland.cpp), so this is the current walk's lights on the sync path and last
        // frame's on the async path. Copy is a few KB — noise against the walk.
        {
            MGE::SceneGraph::SnapshotReadLock lk;
            g_lightSnapshot = MGE::SceneGraph::pointLights();
        }

        // Roots for this frame: the walks below, ensureFullWalk's deferred walk, and
        // ensureLive's capture-context climb all key off these.
        g_objRoot  = MGE::DataHandlerView::worldObjectRoot(dataHandler);
        g_pickRoot = MGE::DataHandlerView::worldPickObjectRoot(dataHandler);
        g_landRoot = MGE::DataHandlerView::worldLandscapeRoot(dataHandler);

        // Weather/VFX texture-name registration (alpha-rigor): sweep worldRoot's UNWALKED
        // children — Precipitation Rain/Snow Root, Storm Root, WorldProjectileRoot,
        // WorldSpellRoot, WorldVFXRoot, ... — so their particle textures resolve by name in
        // captureAlphaDraw instead of falling back to opaque white (ashstorm whiteout). The
        // walked roots are skipped (their walks already register); subtrees are a handful of
        // nodes each, so a per-frame sweep is noise.
        if (RenderProcess::wantsGeometryCapture() && g_objRoot && g_objRoot->parentNode) {
            NI::Node* worldRoot = g_objRoot->parentNode;
            const auto count = worldRoot->children.getEndIndex();
            for (size_t i = 0; i < count; ++i) {
                NI::AVObject* c = worldRoot->children.at(i).get();
                if (!c || c == g_objRoot || c == g_pickRoot || c == g_landRoot) continue;
                registerSubtreeTextureNames(c);
            }
        }

        // Forge lip/blink: the msoc owned-display skip starves MW's own head-morph
        // apply, so drive it here — BEFORE the walks/ensureLive capture this frame's
        // verts. DX9 frames (F11 off) keep the engine's own display-time apply.
        if (RenderProcess::ownsOpaqueWorld()) {
            driveHeadMorphs();
        }

        // FP0: in 1st person the engine appCulls the player's 3rd-person body, but the
        // offscreen shadow-caster re-emit loop (renderprocess) sweeps the whole cache
        // with no appCulled knowledge and the body sits at the camera — well inside its
        // radius — so it kept rendering until the eviction sweep (~30 frames) after a
        // 3rd→1st switch. Stamp the body subtree suppressed EVERY 1st-person frame (the
        // stamp is per-frame); the re-emit loop skips stamped entries. Derefs only the
        // live node fetched from the engine this frame (never cached keys).
        if (RenderProcess::ownsOpaqueWorld()) {
            auto* mwBridge = MWBridge::get();
            static bool s_was3rd = true;
            const bool is3rd = mwBridge->is3rdPerson();
            if (!is3rd) {
                const uint32_t stamped =
                    markSubtreeSuppressed(mwBridge->getPlayer3rdPersonNode());
                if (s_was3rd) {
                    LOG::logline("[fp0] POV switch -> 1st person (frame %llu): stamped %u body entries suppressed",
                                 (unsigned long long)g_frame, stamped);
                }
            } else if (!s_was3rd) {
                LOG::logline("[fp0] POV switch -> 3rd person (frame %llu)", (unsigned long long)g_frame);
            }
            s_was3rd = is3rd;
        }

        // W3 live-read: on Forge-owned frames the refresh walk is dead work — the
        // classify-visible keys are freshened one-by-one off their live NiTriShapes
        // in buildFrustumVisibleSet (ensureLive), and a frame with no classify pulls
        // the full walk in via ensureFullWalk. Sky/eviction/heartbeat still run here.
        if (!liveDrawBuild) {
            runRefreshWalks();
        }
        const double tLand = gcNowMs();
        // SK1 sky takeover: walk skyRoot only when the Forge sky pass is live (F7 toggle ON +
        // seam compositing). walk()'s getAppCulled() early-return gives free day/night/phase/
        // weather selection; captured shapes ride the same capture/IPC seam as opaques but are
        // tagged isSky → the Forge host's dedicated alpha-blend sky pass draws them.
        if (RenderProcess::wantsSkyCapture()) {
            MGE_ZoneScopedN("GeomCache:walkSky");
            g_walkingSky = true;
            g_skyVisitCounter = 0;   // SK2: restart back-to-front ordering each sky walk
            walk(findSkyRoot(dataHandler));
            g_walkingSky = false;
        }
        const double tSky = gcNowMs();

        // FP1a first-person takeover: walk the WorldController armCamera root (the arms/
        // weapon subtree MW renders in its own post-z-clear scene). inCharacter=true — the
        // whole subtree is the player's skinned body; bypassCull=true on the ROOT only,
        // because FP1b force-culls that root to suppress MW's own arm draws while this
        // capture must keep running (children keep their engine cull state, which selects
        // the sheathed/drawn weapon variants etc.). Exempt from the active-cell gate
        // (see walk()) — the arm subtree rides the camera, not the world grid.
        if (RenderProcess::wantsFPCapture()) {
            MGE_ZoneScopedN("GeomCache:walkFP");
            g_fpParticleSystems.clear();   // populated by walk() when it meets NiParticles leaves
            g_walkingFP = true;
            walk(MWBridge::get()->getArmCameraRoot(), /*inCharacter*/true, /*bypassCull*/true);
            g_walkingFP = false;

            // FP particle billboarding (P0, log-only): MW's particle renderer expands each
            // particle into a camera-facing quad at draw time; under FP suppression that draw
            // never runs, and visitGeometry captures only particle CENTERS. Rebuild the quads
            // ourselves against the arm camera. P0 just logs counts + sample positions so we
            // can confirm the vertex space (local-vs-world) and sizes before wiring to the host.
            buildFPParticleQuads();
        }

        // FP1b: while the host FP pass owns the arms, force MW's arm-scene ROOT appCulled
        // so the engine's own first-person draws no-op (no double image). Applied every
        // frame — the engine re-asserts its own cull state on POV/weapon changes — and
        // restored ONCE on any gate release (F11 off / 3rd person / ini off / host dead).
        // The FP walk above bypasses this root flag, so capture keeps running.
        {
            static bool s_fpForcedCull = false;
            const bool want = RenderProcess::wantsFPSuppression();
            if (want || s_fpForcedCull) {
                if (NI::Node* armRoot = MWBridge::get()->getArmCameraRoot()) {
                    armRoot->setAppCulled(want);
                }
                s_fpForcedCull = want;
            }
        }

        // MW-ONLY-UI: force MW's world roots appCulled so the ENGINE never traverses them.
        // Phase 2 stopped MGE drawing, and the proxy reject gate drops MW's draws one by one —
        // but MW still walks its whole scene graph and issues every DrawIndexedPrimitive first.
        // That traversal + per-draw rejection is the "mwsky"/"mwdraws" time in Tracy: work whose
        // only product is calls we throw away.
        //
        // Applied HERE, after the capture walks above, so our walk still sees the full graph —
        // and on the ROOTS only, which our walks start AT (walk() checks the node's own flag, and
        // these roots' children are untouched), exactly like the FP1b arm-root trick.
        //
        // Per-root because MW's draws are NOT all waste: captureAlphaDraw (distantland.cpp:1671)
        // consumes MW's blended DIPs in scene >= 1 for particles/VFX. MW culls ONCE per frame and
        // draws both scenes off that one list, so suppressing a root kills its scene-1 alpha too.
        // Hence the ladder — each step is a bet about what the host already draws:
        //   1 land  : landscape only. Host owns terrain; terrain has no blended DIPs. Safe.
        //   2 +pick : ground items. Host owns them; loses any captured blend they emit.
        //   3 +obj  : the big one — statics/NPCs, the bulk of the traversal. Loses captured
        //             blends from object subtrees (banners/curtains/decals the host cache pass
        //             does not own). Watch for vanishing alpha bits before trusting it.
        // Weather/VFX/projectile/spell roots are NEVER suppressed — they are worldRoot siblings
        // and their particle DIPs are exactly what AT3 capture exists for.
        {
            const int level = RenderProcess::ownsOpaqueWorld() ? DistantLand::mwWorldSuppress : 0;
            applyWorldSuppression(level);
        }

        // SK0 (sky takeover, diagnostic): periodically dump the skyRoot subtree so we can
        // confirm the shapes/materials the real walk will capture. No capture, no draw.
        if (Configuration.LogDistantPipeline) {
            static uint64_t s_lastSkyLog = 0;
            if (g_frame - s_lastSkyLog >= 300) {
                s_lastSkyLog = g_frame;
                if (NI::Node* skyRoot = findSkyRoot(dataHandler)) {
                    LOG::logline("== [SKY DUMP] frame %llu ==", (unsigned long long)g_frame);
                    dumpSkyNode(skyRoot, 0);
                } else {
                    LOG::logline("== [SKY DUMP] skyRoot NOT found (frame %llu) ==", (unsigned long long)g_frame);
                }
            }
        }

        // Precise gone-detection for the sweep. On W3 live-read frames runRefreshWalks is
        // skipped, so the sweep below can only AGE entries out (kFarKeepFrames ~3.6s at 165fps) —
        // a picked-up / despawned object's shadow-caster record then lingers that long before its
        // slot-release ships (the drop/pickup ghost delay). Force ONE full walk on sweep frames so
        // the walk-based "visitable but unvisited = gone" rule fires instead: a removed near object
        // is evicted (and released) within one sweep interval (~0.18s at 165fps), while culled-but-
        // resident objects are re-stamped by the walk and correctly kept. Cost: a full graph walk
        // 1-in-kEvictSweepInterval frames (idempotent if the walk already ran this frame).
        // Menu-mode fast sweep: in menu mode the game advances ONE frame per mouse click, and the
        // player can drop or pick at most ONE near (hand-reach) object per tick — so the per-frame
        // full walk that the 30-frame throttle exists to avoid (165fps) is free here, while the
        // normal cadence would leave a picked-up item's ghost shadow lingering a whole sweep
        // interval (~30 dead clicks, not 0.18s). Force the sweep EVERY menu frame so the removed
        // near object is detected + released on the very next click. Hand-reach ⇒ always within the
        // gate radius, so the plain walk-based "visitable but unvisited = gone" rule applies cleanly.
        // Under g_evictByParentChain the forced walk is unnecessary — reachability answers "gone?"
        // without traversing anything — so it fires only during the validation window, where the
        // walk verdict is the oracle the refcount verdict is checked against. Menu-mode sweeps keep
        // running every frame and get MORE precise this way (an unparented item is detected on the
        // very next click) while no longer pulling a walk per click.
        const bool sweepNow = (g_frame % kEvictSweepInterval == 0) || MWBridge::get()->IsMenu();
        const bool validating = g_evictByParentChain && (g_evictSweepNo < kEvictValidateSweeps);
        if (sweepNow && (!g_evictByParentChain || validating || g_evictForceWalk)) {
            ensureFullWalk();
        }

        // Deferred eviction sweep (see kEvictSweepInterval declaration): drop entries
        // the walk hasn't touched since the last sweep. Consumers that scan the whole
        // cache filter on lastFrame == currentFrame() (buildSkyDrawList, the visible-set
        // frustum fallback), so a stale entry between sweeps is memory, not pixels.
        // The character-verdict cache is wiped on the same cadence, bounding its
        // staleness/pointer-recycle window to one sweep interval (or every frame in menu mode).
        if (sweepNow) {
            MGE_ZoneScopedN("GeomCache:evict");
            // Two eviction rules, selected by whether the full walk ran THIS frame:
            //  - walk ran: stale == visitable but unvisited == genuinely gone. Evict,
            //    except the active-cell hysteresis — a stale entry beyond the gate
            //    radius was likely SKIPPED, not removed; keep it up to kFarKeepFrames
            //    so a returning player doesn't pay re-capture/re-upload.
            //  - live frame (walk skipped): only classify-VISIBLE entries got stamped,
            //    so off-screen != gone — pure age rule (untouched > kFarKeepFrames).
            //    Rotation churn is impossible: anything re-seen within that window is
            //    still cached; beyond it, one lazy re-capture (no IPC geometry — the
            //    capture-side dedup still knows the mesh).
            const bool walkedThisFrame = (g_walkRanFrame == g_frame);
            const float evictR2 = g_gateRadius * g_gateRadius;
            ++g_evictSweepNo;
            // Parent-chain verdict: is this shape still REACHABLE from a live MW root?
            //   0 = alive   (chain terminates at g_objRoot / g_pickRoot / g_landRoot)
            //   1 = gone    (detached: no parent, or the chain dead-ends at an unknown top)
            //   2 = unknown (no stored ref, sky entry, or depth cap hit → age rule decides)
            // Read THROUGH the stored pointer; never copy the NI::Pointer (a copy would claim a ref).
            // "Chain dead-ends at a top we don't recognise ⇒ gone" is only sound if we know EVERY
            // live root, and we do not. Matching {obj,pick,land} evicted 12 live FIRST-PERSON hand
            // parts per sweep (Dark Elf Hfinger/Hpalm/Hthumb/_M_LA/shirt sleeve, every one at
            // `lastFrame=-0` — walked that very frame): FP geometry hangs off the FP root, absent
            // from that set, so its chain dead-ended and read as GONE. skyRoot is the same trap, and
            // any root added later would silently re-introduce it.
            //
            // Climbing to a shared "live top" was tried and does NOT work: MW's roots are
            // INDEPENDENT tops (g_objRoot has no parent), so the climb returns g_objRoot itself and
            // the FP chain still fails to match — the re-run reproduced 12/sweep exactly.
            //
            // The fix is to NAME the FP root rather than guess: MWBridge exposes it as
            // getArmCameraRoot() (already used by FP suppression below), so it joins the known set
            // and FP chains resolve as ALIVE for the right reason.
            //
            // A blanket "stamped within the last sweep interval ⇒ alive" guard was tried first and
            // is WRONG, because a picked-up object is exactly the ambiguous case: MW detaches the
            // object's ROOT node, so its shapes keep a non-null parentNode into that detached root
            // and land in the same dead-end branch as FP. Holding everything recently stamped meant
            // parts of one object evicted on different sweeps — the reported "object left some of
            // its parts". What IS sound is the walk's own verdict: if the walk ran THIS frame and
            // stamped this entry, the entry is reachable by definition. That is authoritative, not
            // a heuristic, and it costs no latency for objects that stopped being stamped.
            // Roots are re-read from dataHandler EVERY frame (see onFrameReady). Two states make
            // the graph verdict meaningless, and both would otherwise condemn the whole cache:
            //  - roots null (teardown / mid-load): nothing matches, everything reads GONE.
            //  - roots REPLACED (cell change): surviving entries are parented under the OLD root,
            //    so their chains dead-end and the entire previous world reads GONE in one sweep.
            // Evicting ~14k entries in a single sweep pushes a release record per key into
            // g_pendingBlob and flushes them as blocking RPCs at present time — a multi-second
            // stall that presents as a game freeze. Cross-cell removal is ALREADY handled by the
            // cell-epoch eviction (see drainReleasedSlots), so this sweep doing it too is redundant,
            // not load-bearing: refusing is strictly safer than participating.
            const bool rootsValid = (g_objRoot != nullptr);
            const bool walkAuthoritative = walkedThisFrame;
            NI::Node* const armRoot = MWBridge::get()->getArmCameraRoot();
            // --- parent-climb INSTRUMENTATION + engine-informed liveness guard (2026-07-20) ---
            // The un-instrumented flip to this path froze the CLIENT. The climb is bounded in DEPTH
            // (kMaxParentDepth) so it cannot infinite-loop, and an AV would crash (with a dump), not
            // hang — so a probe must establish two things: (a) is the climb burning real wall time,
            // (b) is it walking DANGLING parents. A cached leaf keeps a strong ref on ITSELF but NOT
            // on its parent, so once MW frees a detached subtree's root our stored parentNode dangles
            // into stale heap; the next `p->parentNode` reads dead memory. Fix + probe: validate each
            // candidate parent against the real NI node-subtype vtables (MWSE-provided addresses in
            // NI::VirtualTableAddress — no disassembly). A freed slot no longer matches any node
            // vtable, so we treat it as GONE instead of chasing it further. This makes EVERY deref in
            // the loop safe (first hop is off a leaf we hold; every later hop is off a vtable-validated
            // live node) and turns the unbounded hazard into counted diagnostics (climbVtBad).
            unsigned climbMaxDepth = 0, climbVtBad = 0, climbDepthCap = 0;
            auto isLiveNode = [](const NI::Node* n) -> bool {
                if (!n) return false;
                namespace VA = NI::VirtualTableAddress;
                switch (reinterpret_cast<std::uintptr_t>(n->vTable.asNode)) {
                    case VA::NiNode:            case VA::NiBillboardNode:  case VA::BSMirroredNode:
                    case VA::NiSwitchNode:      case VA::NiBSAnimationNode: case VA::NiBSParticleNode:
                    case VA::NiBSPNode:         case VA::NiFltAnimationNode: case VA::NiLODNode:
                    case VA::NiSortAdjustNode:  case VA::AvoidNode:        case VA::RootCollisionNode:
                        return true;
                    default:
                        return false;   // dangling/freed parent, or a node type outside the family
                }
            };
            auto parentVerdict = [&](uint32_t key, const CachedGeometry& e) -> int {
                if (!rootsValid) return 2;              // no usable reference point → age rule only
                // The walk reached it this frame ⇒ reachable. Belt-and-braces against any root we
                // still don't know about (this is what the FP hands tripped).
                if (walkAuthoritative && e.lastFrame == g_frame) return 0;
                if (e.isSky) return 2;                  // skyRoot untracked; sky is few entries
                auto rit = g_geomRefs.find(key);
                if (rit == g_geomRefs.end() || !rit->second.get()) return 2;
                NI::Node* p = rit->second.get()->parentNode;
                // Detached outright — the common drop/pick-up case, and unambiguous.
                if (!p) return 1;
                for (int depth = 0; depth < kMaxParentDepth; ++depth) {
                    if (p == g_objRoot || p == g_pickRoot || p == g_landRoot || (armRoot && p == armRoot)) {
                        if ((unsigned)depth > climbMaxDepth) climbMaxDepth = (unsigned)depth;
                        return 0;
                    }
                    // Engine-informed guard: a real parent is a live node subtype. A dangling/freed
                    // parent fails the vtable check ⇒ the subtree was unlinked and released ⇒ gone.
                    if (!isLiveNode(p)) { ++climbVtBad; return 1; }
                    NI::Node* next = p->parentNode;
                    // Chain ended without reaching a known root: the whole subtree was unlinked
                    // (MW removes an object's ROOT node, so the shape keeps a non-null parent that
                    // is itself detached). This is the case a plain !parent test would miss.
                    if (!next) return 1;
                    p = next;
                }
                ++climbDepthCap;
                return 2;   // depth cap: refuse to guess
            };
            unsigned nEvicted = 0, nByGraph = 0, nByAge = 0, nUnknown = 0, nDeferred = 0;
            unsigned nLogWalkGone = 0, nLogGraphGone = 0;
            constexpr unsigned kEvictCmpLogCap = 12;
            // CIRCUIT BREAKER. A correct sweep retires a handful of entries (an object left the
            // world). A verdict condemning a large FRACTION of the cache is not a world event, it
            // is a root-identity artifact — and acting on it is what stalls the frame. Count first,
            // then refuse the graph verdict wholesale for this sweep if it looks like a mass kill;
            // the age rule still retires anything genuinely stale within kFarKeepFrames. Loud,
            // ALWAYS-ON log: the [evict] summary below only prints while validating, so without
            // this a mass eviction after sweep 20 would be completely invisible.
            // PER-SWEEP EVICTION BUDGET — the actual freeze guard, and it covers EVERY cause.
            // Observed with the always-on log above: one sweep evicted 10186 entries, all byAge
            // (entries 12572 -> 2386). Removing the forced walk left far more entries unstamped, so
            // instead of trickling out they all cross kFarKeepFrames on the SAME sweep. Each
            // eviction pushes a slot-release record into g_pendingBlob, and 10k of them flush as
            // blocking RPCs at present time — a multi-second stall, i.e. the reported game freeze.
            // Evicting them is correct; doing it in one frame is not. Whatever is left over is
            // simply re-evaluated next sweep, so nothing is retained permanently — the backlog
            // drains at kMaxEvictPerSweep per sweep with no visible hitch.
            constexpr unsigned kMaxEvictPerSweep = 512;
            bool graphTrusted = true;
            // WATCHDOG: the un-instrumented flip froze in this climb. If it ever burns real wall
            // time, abort the graph verdict, fall back to the age rule, and log the exact ms +
            // progress. Combined with isLiveNode() above (no dangling deref), the probe can NEVER
            // hard-freeze the game the way the raw flip did.
            constexpr double kClimbWatchdogMs = 250.0;
            const double     tClimb0 = gcNowMs();
            double           climbCountMs = 0.0;
            bool             climbRunaway = false;
            if (g_evictByParentChain && rootsValid) {
                unsigned wouldEvict = 0, scanned = 0;
                for (const auto& kv : g_cache) {
                    if (parentVerdict(kv.first, kv.second) == 1) ++wouldEvict;
                    if ((++scanned & 2047) == 0 && gcNowMs() - tClimb0 > kClimbWatchdogMs) {
                        climbRunaway = true;
                        graphTrusted = false;
                        LOG::logline("!! [evict-climb] RUNAWAY in count pass: %.0fms at %u/%zu entries"
                                     " (maxDepth=%u vtBad=%u depthCap=%u) — aborting graph verdict,"
                                     " age rule only this sweep",
                                     gcNowMs() - tClimb0, scanned, g_cache.size(),
                                     climbMaxDepth, climbVtBad, climbDepthCap);
                        break;
                    }
                }
                const size_t cap = g_cache.size() / 4;   // >25% of the cache in one sweep
                if (!climbRunaway && g_cache.size() > 64 && wouldEvict > cap) {
                    graphTrusted = false;
                    LOG::logline("!! [evict] MASS-EVICT REFUSED: graph verdict condemned %u/%zu entries"
                                 " (>25%%) — treating as a root-identity artifact (cell change/teardown),"
                                 " falling back to the age rule this sweep",
                                 wouldEvict, g_cache.size());
                }
            }
            climbCountMs = gcNowMs() - tClimb0;
            for (auto it = g_cache.begin(); it != g_cache.end(); ) {
                auto& e = it->second;
                bool evict;
                if (walkedThisFrame) {
                    evict = (e.lastFrame != g_frame);
                    // Far-keep hysteresis: a stale entry BEYOND the gate radius was gate-SKIPPED by
                    // this frame's walk, not removed — keep it (returning player pays no re-upload).
                    // Gate this on g_gateThisFrame (armed THIS frame, so g_gateEye is current), NOT
                    // the persisted g_gateRadius: after exterior->interior the radius lingers >0 while
                    // g_gateEye still points at the old exterior camera, so a dropped-at-your-feet
                    // interior item measured against that stale eye looks "far" and was wrongly held
                    // the full kFarKeepFrames (~3.6s) before aging out — the delayed drop/pickup ghost.
                    // When the gate isn't armed this frame the walk covered everything → unvisited = gone.
                    if (evict && g_gateThisFrame && g_frame - e.lastFrame <= kFarKeepFrames) {
                        const float dx = e.worldTransformD3D[12] - g_gateEye[0];
                        const float dy = e.worldTransformD3D[13] - g_gateEye[1];
                        const float dz = e.worldTransformD3D[14] - g_gateEye[2];
                        if (dx * dx + dy * dy + dz * dz > evictR2) {
                            evict = false;
                        }
                    }
                } else {
                    evict = (g_frame - e.lastFrame > kFarKeepFrames);
                }
                if (g_evictByParentChain) {
                    const int  graph = graphTrusted ? parentVerdict(it->first, e) : 2;
                    const bool aged  = (g_frame - e.lastFrame > kFarKeepFrames);
                    // Age is the backstop, not the hysteresis: it covers `unknown` verdicts (sky
                    // entries, missing refs, depth-cap bailouts). The active-cell far-keep
                    // hysteresis is not needed here — a gate-SKIPPED object is still PARENTED, so
                    // reachability already says "alive"; it survives only in the walk verdict above.
                    const bool refEvict = (graph == 1) || aged;
                    if (validating && evict != refEvict) {
                        const char* tex = e.textureName ? e.textureName : resolveTextureName(e.d3dTexture);
                        // Real engine refCount, not the verdict — how far above 1 an "alive"
                        // entry sits is the diagnostic (2 = parent only; higher = other holders).
                        auto rit = g_geomRefs.find(it->first);
                        const int rc = (rit != g_geomRefs.end() && rit->second.get())
                                           ? rit->second.get()->refCount : -1;
                        if (evict && !refEvict) {
                            ++g_evictDisagreeWalkGone;
                            if (nLogWalkGone < kEvictCmpLogCap) {
                                ++nLogWalkGone;
                                LOG::logline("!! [evict-cmp] sweep=%u key=%08x tex=%s walk=GONE parent=ALIVE rc=%d sky=%d lastFrame=-%llu",
                                             g_evictSweepNo, it->first, tex ? tex : "(none)", rc,
                                             (int)e.isSky,
                                             (unsigned long long)(g_frame - e.lastFrame));
                            }
                        } else {
                            ++g_evictDisagreeGraphGone;
                            if (nLogGraphGone < kEvictCmpLogCap) {
                                ++nLogGraphGone;
                                LOG::logline("!! [evict-cmp] sweep=%u key=%08x tex=%s walk=KEEP parent=GONE rc=%d sky=%d lastFrame=-%llu",
                                             g_evictSweepNo, it->first, tex ? tex : "(none)", rc,
                                             (int)e.isSky,
                                             (unsigned long long)(g_frame - e.lastFrame));
                            }
                        }
                    }
                    if (refEvict) {
                        if (graph == 1)      ++nByGraph;
                        else if (graph == 2) ++nUnknown;
                        else                 ++nByAge;
                    }
                    evict = refEvict;
                }
                // Budget spent: leave the rest for the next sweep (see kMaxEvictPerSweep).
                if (evict && nEvicted >= kMaxEvictPerSweep) {
                    evict = false;
                    ++nDeferred;
                }
                if (evict) {
                    ++nEvicted;
                    g_evictedKeys.push_back(it->first);   // tell the Forge feed to release the host slot
                    releaseEntry(e);
                    g_geomRefs.erase(it->first);          // key leaving the cache → drop the engine ref
                    g_moverCandidates.erase(it->first);   // key leaving the cache → drop the candidate
                    it = g_cache.erase(it);
                } else {
                    ++it;
                }
            }
            g_charNodeVerdict.clear();
            // Print while validating, and ALWAYS whenever a sweep actually evicted something — the
            // eviction path is where the freeze/ghosting bugs live, so a non-zero sweep must never
            // be silent (it was, between sweep 20 and the mass-evict investigation).
            if (validating || nEvicted > 0) {
                LOG::logline(">> [evict] sweeps=%u walk-forced=%u entries=%zu evicted=%u byGraph=%u byAge=%u unknown=%u deferred=%u disagree(walkGone/parentGone)=%u/%u",
                             g_evictSweepNo, walkedThisFrame ? 1u : 0u, g_cache.size(),
                             nEvicted, nByGraph, nByAge, nUnknown, nDeferred,
                             g_evictDisagreeWalkGone, g_evictDisagreeGraphGone);
            }
            // Parent-climb probe (every sweep while the graph verdict is active). countMs = wall time
            // of the full-cache climb; if it ever spikes this IS the freeze. vtBad = dangling/freed
            // parents the vtable guard rejected (the health of the whole premise — a large number
            // means the raw pointer climb was chasing dead memory). depthCap = chains that never
            // reached a known root within kMaxParentDepth (garbage-chain smell). runaway = watchdog
            // tripped this sweep.
            if (g_evictByParentChain) {
                LOG::logline(">> [evict-climb] sweep=%u entries=%zu countMs=%.2f maxDepth=%u vtBad=%u depthCap=%u runaway=%d trusted=%d walked=%d",
                             g_evictSweepNo, g_cache.size(), climbCountMs,
                             climbMaxDepth, climbVtBad, climbDepthCap,
                             (int)climbRunaway, (int)graphTrusted, walkedThisFrame ? 1 : 0);
            }
        }
        const double tEvict = gcNowMs();

        // [gc] heartbeat: per-phase walk cost, averaged over the window. Always on —
        // this walk sits on the dense-city serial frame chain.
        g_gcAccum.sky   += tSky - tLand;
        g_gcAccum.evict += tEvict - tSky;
        if (++g_gcAccum.n >= kGcHeartbeatFrames) {
            const double n = (double)g_gcAccum.n;
            LOG::logline(">> [gc] %llu frames avg: walk=%.2f (obj=%.2f pick=%.2f land=%.2f sky=%.2f evict=%.2f) entries=%zu visited=%.0f gateSkip=%.0f live=%.0f cap=%.1f%s gateR=%.0f",
                         (unsigned long long)g_gcAccum.n,
                         (g_gcAccum.obj + g_gcAccum.pick + g_gcAccum.land + g_gcAccum.sky + g_gcAccum.evict) / n,
                         g_gcAccum.obj / n, g_gcAccum.pick / n, g_gcAccum.land / n,
                         g_gcAccum.sky / n, g_gcAccum.evict / n, g_cache.size(),
                         g_gcAccum.visited / n, g_gcAccum.gateSkip / n,
                         g_gcAccum.live / n, g_gcAccum.cap / n,
                         g_gateThisFrame ? " gate=ON" : "", g_gateRadius);
            g_gcAccum = GcAccum{};
        }

        if (Configuration.LogDistantPipeline) {
            static uint64_t s_lastLog = 0;
            if (g_frame - s_lastLog >= 1800) {
                uint32_t skinnedCount = 0, namedTexCount = 0, nullTexCount = 0, nullNameCount = 0;
                uint32_t mirroredCount = 0;
                // Terrain (isLandscape) characterization: how the splat passes land
                // in the cache. landOpaque = base layers (no blend), landAlphaTest,
                // landBlend = alpha-splat layers (separate trishapes if >0 here).
                // landVCol = carry vertex colours. Samples a few texture names.
                uint32_t landCount = 0, landOpaque = 0, landAlphaTest = 0, landBlend = 0, landVCol = 0;
                const char* landTexA = nullptr; const char* landTexB = nullptr;
                for (const auto& kv : g_cache) {
                    // Hysteresis-kept far entries may outlive their cell (NI string
                    // pointers dangle after unload) — characterize recent entries only.
                    // (<=1: on live frames this frame's stamps happen after this dump.)
                    if (g_frame - kv.second.lastFrame > 1) continue;
                    if (kv.second.isSkinned) ++skinnedCount;
                    if (kv.second.mirrored) ++mirroredCount;
                    if (kv.second.d3dTexture && kv.second.textureName) ++namedTexCount;
                    else if (!kv.second.d3dTexture) ++nullTexCount;
                    else ++nullNameCount;
                    if (kv.second.isLandscape) {
                        ++landCount;
                        if (kv.second.blendEnable) ++landBlend;
                        else if (kv.second.alphaTest) ++landAlphaTest;
                        else ++landOpaque;
                        if (kv.second.hasVertexColor) ++landVCol;
                        if (kv.second.textureName) {
                            if (!landTexA) landTexA = kv.second.textureName;
                            else if (!landTexB && kv.second.textureName != landTexA) landTexB = kv.second.textureName;
                        }
                    }
                }
                LOG::logline("-- [GEOM CACHE] frame=%llu cached=%zu skinned=%u named=%u nulltex=%u nullname=%u mapSize=%zu uploads/interval=%llu",
                    g_frame, g_cache.size(), skinnedCount, namedTexCount, nullTexCount, nullNameCount, g_textureNameMap.size(), g_uploadedInterval);
                LOG::logline("-- [GEOM CACHE] nullbone hits/interval=%u parts/interval=%u sampleTex=%s mirrored=%u",
                    g_nullBoneHitsInterval, g_nullBonePartsInterval,
                    g_nullBoneSampleTex ? g_nullBoneSampleTex : "(none)", mirroredCount);
                LOG::logline("-- [GEOM CACHE] landscape=%u opaque=%u alphatest=%u blend=%u vcol=%u texA=%s texB=%s",
                    landCount, landOpaque, landAlphaTest, landBlend, landVCol,
                    landTexA ? landTexA : "(none)", landTexB ? landTexB : "(none)");
                g_uploadedInterval = 0;
                g_nullBoneHitsInterval = 0;
                g_nullBonePartsInterval = 0;
                g_nullBoneSampleTex = nullptr;
                s_lastLog = g_frame;
            }
        }
    }

    const std::unordered_map<uint32_t, CachedGeometry>& cache() {
        return g_cache;
    }

    const std::unordered_set<uint32_t>& moverCandidates() {
        return g_moverCandidates;
    }

    const void* fpParticleVerts(uint32_t& countOut) {
        countOut = (uint32_t)g_fpPartVerts.size();
        return g_fpPartVerts.data();
    }
    const void* fpParticleIndices(uint32_t& countOut) {
        countOut = (uint32_t)g_fpPartIndices.size();
        return g_fpPartIndices.data();
    }
    const std::vector<FPParticleRec>& fpParticleRecs() {
        return g_fpPartRecs;
    }

    void emissiveForDraw(const CachedGeometry& e, float* out) {
        if (Configuration.ForgeEmissiveBoost) {
            out[0] = e.matEmissive[0] * e.emissiveGain[0];
            out[1] = e.matEmissive[1] * e.emissiveGain[1];
            out[2] = e.matEmissive[2] * e.emissiveGain[2];
        } else {
            out[0] = e.matEmissive[0];
            out[1] = e.matEmissive[1];
            out[2] = e.matEmissive[2];
        }
    }

    uint64_t currentFrame() {
        return g_frame;
    }

    void takeEvictedKeys(std::vector<uint32_t>& out) {
        out.clear();
        out.swap(g_evictedKeys);   // move-out + leave g_evictedKeys empty for the next sweep
    }

    void purgeAll() {
        // A cell teardown (load door, teleport, save load) destroys the whole scene graph, but the
        // cache is keyed on shape ADDRESSES and knows nothing about it — so every entry survives into
        // the new cell as a corpse. They were then still being emitted for a frame before the age
        // sweep caught them: the one-frame flash of the OLD cell's objects and NPCs on a transition.
        //
        // Drop the lot. Every key goes through the SAME eviction channel the age sweep uses, so the
        // Forge feed releases the matching host mesh slots (and their shadow-caster records) instead
        // of leaving them ghosting host-side. Everything still present in the new cell is re-captured
        // lazily on first sight — cell changes already re-upload, so this costs nothing that the
        // transition wasn't paying anyway.
        //
        // Note this is a CORRECTNESS fix, not a safety one: the g_geomRefs strong refs are what make
        // a stale key safe to touch. That ordering matters — it means a MISSED purge (the cell-change
        // signal is a heuristic) degrades to a harmless ghost rather than to a use-after-free.
        for (auto& kv : g_cache) {
            g_evictedKeys.push_back(kv.first);
            releaseEntry(kv.second);
        }
        g_cache.clear();
        g_moverCandidates.clear();   // whole cache dropped → no candidates survive
        g_geomRefs.clear();   // NI::Pointer dtors → DecRef every shape we were pinning
    }

    const CachedGeometry* ensureLive(uint32_t key) {
        if (!g_device) return nullptr;
        auto* geom = reinterpret_cast<NI::TriBasedGeometry*>(key);

        auto it = g_cache.find(key);
        if (it != g_cache.end() && it->second.lastFrame == g_frame) {
            return &it->second;     // already fresh (full walk ran, or a duplicate key)
        }

        // The deref below is only sound for two kinds of key, and both are covered:
        //   - ALREADY CACHED  → g_geomRefs holds an engine reference, so the shape cannot have been
        //                       freed, however long ago it was last seen and wherever it is now.
        //   - FIRST SIGHT     → the caller passed a key from THIS frame's classify set (live by
        //                       construction; foldVisibleKeys() is consume-once for exactly that
        //                       reason), so it is alive right now and the lazy capture below takes
        //                       the ref before anyone can deref it again.
        // Never hand this function a raw key from any other source.
        auto* data = geom->getModelData().get();
        if (!data) return nullptr;

        if (it != g_cache.end() && it->second.dataPtr != data) {
            // A live setModelData swap (the address itself can no longer be recycled onto a new
            // shape — g_geomRefs pins it — but the DATA behind it can still be replaced).
            releaseEntry(it->second);
            g_cache.erase(it);
            g_geomRefs.erase(key);   // key leaving the cache → drop the engine ref
            g_moverCandidates.erase(key);   // stale entry gone; re-capture below re-adds if applicable
            it = g_cache.end();
        }

        if (it == g_cache.end()) {
            // Lazy capture on first sight. Classify the leaf by climbing to its root so
            // isPickRoot/isLandscape (and landscape's forced single-UV upload) come out
            // exactly as the walk would have set them. The classify set covers the whole
            // world-camera scene — leaves OUTSIDE the walk's three roots (engine water
            // plane, shadow receivers, ...) were never cached by the walk and must not
            // be captured here either.
            bool isLand = false, isPick = false, inDomain = false;
            for (NI::AVObject* a = geom->parentNode; a; a = a->parentNode) {
                if (a == g_landRoot) { isLand = true; inDomain = true; break; }
                if (a == g_pickRoot) { isPick = true; inDomain = true; break; }
                if (a == g_objRoot)  { inDomain = true; break; }
            }
            if (!inDomain) return nullptr;
            g_walkingLandscape = isLand;
            g_walkingPick = isPick;
            visitGeometry(geom, false);
            g_walkingLandscape = false;
            g_walkingPick = false;
            ++g_liveCaptureThisFrame;
            it = g_cache.find(key);
            return (it != g_cache.end()) ? &it->second : nullptr;
        }

        // Refresh the per-frame-varying fields the draw paths read — transform, bone
        // palette, mirrored — plus the revision-gated re-extract/re-upload. This is the
        // walk's existing-entry path, run only for keys the engine actually drew.
        auto& e = it->second;
        e.lastFrame = g_frame;
        ++g_liveRefreshThisFrame;

        NI::SkinInstance* si = geom->skinInstance.get();
        NI::SkinData*     sd = si ? si->skinData.get() : nullptr;
        const bool sk = (si && sd && si->bones);
        // Only a re-extract/re-upload can change the mover classification (skin state, blend,
        // multimap maps) — a plain pose refresh cannot. Track it so the membership update stays
        // off the hot per-visible-key path unless something actually reclassified.
        bool reclassified = false;
        if (sk) {
            if (data->revisionID != e.revisionID || !e.isSkinned) {
                extractMaterial(e, geom);
                buildSkinnedVB(e, geom, data, si, sd);
                reclassified = true;
            }
            if (!e.skinnedUnsupported) buildBonePalette(e, geom, si, sd);
            buildD3DTransform(e.worldTransformD3D, geom);
            e.dynamicHint = 4;
            e.mirrored = computeMirrored(e);
        } else {
            if (data->revisionID != e.revisionID || e.isSkinned) {
                g_walkingLandscape = e.isLandscape;   // landscape re-upload keeps single-UV
                extractMaterial(e, geom);
                uploadEntry(e, geom, data, key);
                g_walkingLandscape = false;
                reclassified = true;
            }
            float newTransform[16];
            buildD3DTransform(newTransform, geom);
            if (memcmp(newTransform, e.worldTransformD3D, sizeof(newTransform)) != 0) {
                memcpy(e.worldTransformD3D, newTransform, sizeof(newTransform));
                e.dynamicHint = 4;
                e.mirrored = computeMirrored(e);
            } else if (e.dynamicHint > 0) {
                --e.dynamicHint;
            }
        }
        if (reclassified) updateMoverMembership(key, e);
        return &e;
    }

    void ensureFullWalk() {
        if (g_walkRanFrame == g_frame) return;
        if (!g_device || !g_objRoot) return;
        runRefreshWalks();
    }

    // FP0: recursive stamp over a live subtree, deliberately IGNORING appCulled (the
    // caller marks a subtree the engine just culled). Stamps only entries that already
    // exist in the cache — no capture, no NI data derefs beyond the child arrays.
    static uint32_t stampSuppressed(NI::AVObject* av) {
        if (!av) return 0;
        if (av->isInstanceOfType(NI::RTTIStaticPtr::NiTriBasedGeom)) {
            auto it = g_cache.find(reinterpret_cast<uint32_t>(av));
            if (it != g_cache.end()) {
                it->second.suppressedFrame = g_frame;
                return 1;
            }
            return 0;
        }
        uint32_t n = 0;
        if (av->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) {
            auto* node = static_cast<NI::Node*>(av);
            const auto count = node->children.getEndIndex();
            for (size_t i = 0; i < count; ++i) {
                n += stampSuppressed(node->children.at(i).get());
            }
        }
        return n;
    }

    // MW-ONLY-UI: drive the engine's world-root appCulled flags to `level` (0..3, see the ladder
    // at the call site). Idempotent — re-asserted every frame because the engine rewrites its own
    // cull state on cell/POV changes — and it only ever clears flags it set itself, so a level
    // drop or a gate release restores MW exactly.
    //
    // MUST be restorable: these roots are shared with MW's OTHER render targets (local map, and
    // anything else drawn off the back buffer). Leaving them culled past the main view would draw
    // an empty local map, which is why restoreWorldSuppression() runs at the UI transition and
    // again at Present rather than trusting one call site.
    void applyWorldSuppression(int mask) {
        if (mask == g_worldSuppressApplied) return;
        // Roots are re-read per frame in onFrameReady; nulls just mean "nothing to do".
        if (g_landRoot) g_landRoot->setAppCulled((mask & kSuppressLand) != 0);
        if (g_pickRoot) g_pickRoot->setAppCulled((mask & kSuppressPick) != 0);
        if (g_objRoot)  g_objRoot->setAppCulled((mask & kSuppressObjects) != 0);
        g_worldSuppressApplied = mask;
    }

    void restoreWorldSuppression() {
        applyWorldSuppression(0);
    }

    int worldSuppressionApplied() {
        return g_worldSuppressApplied;
    }

    uint32_t markSubtreeSuppressed(void* avObject) {
        return stampSuppressed(static_cast<NI::AVObject*>(avObject));
    }

    int buildMoonDrawList(void* moonRoot, MoonShapeDraw out[2]) {
        return buildMoonDrawListImpl(static_cast<NI::Node*>(moonRoot), out);
    }

    const char* resolveTextureName(IDirect3DTexture9* tex) {
        if (!tex) return nullptr;
        auto it = g_textureNameMap.find(tex);
        return (it != g_textureNameMap.end()) ? it->second : nullptr;
    }

}
