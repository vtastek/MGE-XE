
#include "configuration.h"
#include "distantland.h"
#include "mwbridge.h"
#include "renderprocess.h"
#include "phasetimers.h"
#include "proxydx/d3d8header.h"
#include "scenegraph.h"
#include "scenegraph_geometry_cache.h"
#include "cachebounds.h"
#include "enginecull.h"
#include "msocclient.h"
#include "support/log.h"
#include "mge_tracy.h"

#include <unordered_set>
#include <vector>

// MSOC-culled visible set received from the VisibleGeomCallback, current frame.
// Keys = NiTriBasedGeometry* cast to uint32_t — matches GeometryCache keys.
// updateVisibleSet (the only writer) runs on the MAIN thread — msoc.dll fires the
// callback from inside the engine's drainPendingDisplays — so a main-thread
// snapshot at kick cannot race it.
//
// A plain VECTOR, not a set: every consumer only ITERATES it (buildFrustumVisibleSet
// MSOC branch, the Cut 2B fold loop) — nothing does membership lookups — and the
// callback fires INSIDE the synchronous classify call, so the old per-key
// unordered_set insert (~3.5k hashes) billed straight to the [classify] serial
// block. assign() is a flat copy. The plugin defers each leaf exactly once per
// scene, so the set's de-dup was doing nothing (wire counts verify).
// (The previous-frame copy, s_prevVisibleKeys/visibleCacheKeys, had NO consumers
// left — removed with the vector change.)
static std::vector<uint32_t> s_visibleKeys;

// Deterministic, current-frame frustum-culled visible set over the full cacheMap,
// built in the early stage (frameSetupEarly, after the cache walk) by
// buildFrustumVisibleSet. This is the set MGE owns: the cache depth pre-pass and
// the cache opaque color pass both drive off it, so leading-edge tiles a pan
// reveals THIS frame get depth (no sky holes) and an empty engine-MSOC set no
// longer collapses the depth pass to an unculled full-cache draw. Frustum-only
// (no occlusion yet — Phase 3 adds the early MGE-driven MSOC mask).
static std::vector<uint32_t> s_frustumVisibleKeys;

// Count of cache entries the last buildFrustumVisibleSet dropped (in MSOC-culled mode:
// entries the engine did not draw — occluded / LOD-deselected / out of the engine
// frustum). Surfaced as the cacheOccCull diagnostic (LogDistantPipeline).
static unsigned     s_refineCulledCount = 0;

// Stage 2 engine-set consumption. s_visibleCallbackFired is a tripwire flipped by
// updateVisibleSet; earlyClassifyMainScene() resets it, calls the plugin's early
// classify, and latches s_earlyClassifyRan = "the callback fired DURING the call" -
// i.e. the world classify ran THIS frame, before the cull, so s_visibleKeys is the
// engine's current-frame drawn set (no lag). buildFrustumVisibleSet reads
// s_earlyClassifyRan to drive the cache passes off s_visibleKeys directly (absence
// culls), then clears it. A late Mode-A callback (engine CullShow, after the cull)
// also trips s_visibleCallbackFired, but it lands after this frame's consume and is
// reset before the next early classify, so it never masquerades as a current set.
static bool         s_visibleCallbackFired = false;
static bool         s_earlyClassifyRan     = false;

// Cut 2B fold latch: this frame's buildFrustumVisibleSet took the MSOC branch but
// DEFERRED its ensureLive loop into the kickoff draw-list build (see the fold gate
// in buildFrustumVisibleSet). Set there; consumed by foldVisibleKeys() (once) and
// reset at the top of the next buildFrustumVisibleSet call. A kickoff early-out
// wastes it harmlessly (in fold mode nothing else consumes the visible set).
static bool         s_foldDeferred         = false;

void DistantLand::updateVisibleSet(void* const* shapes, int count) {
    // x86: void* is 4 bytes, so the plugin's pointer array reinterprets directly as
    // the key array — one flat assign, no per-key hashing (see s_visibleKeys above).
    static_assert(sizeof(void*) == sizeof(uint32_t), "key = pointer bits");
    const uint32_t* keys = reinterpret_cast<const uint32_t*>(shapes);
    s_visibleKeys.assign(keys, keys + count);
    s_visibleCallbackFired = true;
}

// Stage 2 early classify (main thread, called from frameSetupEarly at BeginScene(0),
// before buildFrustumVisibleSet). Run the world-camera scene walk NOW; the producer
// fires the visible-geom callback synchronously on this thread (updateVisibleSet) so
// the current-frame set is ready for the cull below. Latches whether it actually
// classified (callback fired during the call) - both producers self-decline (root
// unverified, scene disabled, menu, absent export) without signalling, so the
// tripwire is the authoritative "did it run".
//
// TWO PRODUCERS, one sink (MSOC retirement D3). MGE's own absorbed CullShow
// traversal (enginecull.cpp) when it owns the prologue patch, msoc.dll otherwise.
// They are mutually exclusive by construction: EngineCull::install() refuses
// whenever msoc could own that prologue. Both call the same onVisibleGeom, return
// the same status codes and honour the same owned flags, so everything downstream
// of here is producer-blind - which is what makes D5 a deletion rather than a port.
void DistantLand::earlyClassifyMainScene(void* worldCamera) {
    s_earlyClassifyRan = false;

    // UseOcclusionCulling is msoc's master switch, and msoc's alone: it gates an
    // occlusion verdict we no longer consume. Our traversal does no occlusion, so
    // the knob that gates it is the takeover knob itself, already proven by
    // isInstalled().
    const bool ownTraversal = MGE::EngineCull::isInstalled();
    if (!ownTraversal && (!Configuration.UseOcclusionCulling || !MSOCClient::hasEarlyClassify())) {
        return;
    }

    // Owned display skips (Cut 1 opaque + AT2 alpha): tell the plugin which
    // parts of the frame the Forge side covers THIS frame, before either
    // classify path latches its per-frame state. Opaque bit: the plugin skips
    // engine display() of covered-opaque leaves — the DIPs our proxy rejects
    // per-draw anyway (inspectIndexedPrimitive). Alpha bit: it also skips
    // single-map blended leaves our sorted-alpha host pass redraws (AT1).
    // F11 (forgeOwnsFrame false) restores full display next frame. Logged
    // plugin-side on change; absent/old export = opaque-only or no-op = the
    // per-DIP reject fallback keeps correctness.
    int ownedFlags = 0;
    if (RenderProcess::forgeOwnsFrame()) {
        ownedFlags = MSOCClient::kOwnedOpaque | MSOCClient::kOwnedAlpha;
    }
    s_visibleCallbackFired = false;
    // The classify is SYNCHRONOUS: the engine's world-camera scene walk (~10k nodes)
    // bills here, and under msoc so does the owned-mode deferred-display drain — post-W3
    // it was the largest un-zoned block in the frameready→stage1 window (MSOC.log drainUs
    // alone ~1.8ms), so it gets its own timer + heartbeat. The absorbed traversal splits
    // that: the walk bills here, the display bills to the engine's own CullShow.
    LARGE_INTEGER ecFreq, ecT0, ecT1;
    QueryPerformanceFrequency(&ecFreq);
    QueryPerformanceCounter(&ecT0);
    int rc;
    {
        MGE_ZoneScopedN("earlyClassifyMainScene");
        MGE_SCOPED_TIMER("earlyClassifyMainScene");
        if (ownTraversal) {
            MGE::EngineCull::beginFrame(ownedFlags);
            rc = MGE::EngineCull::classifyNow(worldCamera);
        } else {
            MSOCClient::setOwnedFlags(ownedFlags);
            rc = MSOCClient::classifyMainSceneNow(worldCamera);
        }
    }
    QueryPerformanceCounter(&ecT1);
    static double s_ecMsAccum = 0.0; static unsigned s_ecN = 0;
    s_ecMsAccum += 1000.0 * (double)(ecT1.QuadPart - ecT0.QuadPart) / (double)ecFreq.QuadPart;
    if (++s_ecN >= 300) {
        LOG::logline(">> [classify] 300 frames avg: %s=%.2f ms",
                     ownTraversal ? "EngineCull::classifyNow" : "classifyMainSceneNow",
                     s_ecMsAccum / (double)s_ecN);
        s_ecMsAccum = 0.0; s_ecN = 0;
    }
    s_earlyClassifyRan = s_visibleCallbackFired;

    // Diagnostic: surface the producer's status code so we can see WHETHER the early
    // classify engaged and, if not, which guard declined. Both producers share the
    // numbering (see EngineCull::classifyNow / msoc OcclusionPass.cpp); codes 2 and 10
    // are msoc-only, 12 is ours. Logged on every change + periodically.
    static int s_lastRc = -2;
    static unsigned s_n = 0;
    if (rc != s_lastRc || (++s_n % 600 == 0)) {
        LOG::logline("-- [EARLY CLASSIFY] src=%s rc=%d ran=%d (0=classified; 1=reentrant "
                     "2=!inRenderMainScene 3=noRootCaptured 4=alreadyClassified 5=sceneHandled "
                     "6=noCamera 7=menuMode 8=wrongCamera 9=noDataHandler 10=gateDeclined "
                     "11=staleRoot 12=notInstalled -1=exportAbsent)",
                     ownTraversal ? "mge" : "msoc", rc, s_earlyClassifyRan ? 1 : 0);
        s_lastRc = rc;
    }
}

const std::vector<uint32_t>& DistantLand::frustumVisibleKeys() {
    return s_frustumVisibleKeys;
}

const std::vector<uint32_t>* DistantLand::foldVisibleKeys() {
    // Consume-once: the latch pairs ONE buildFrustumVisibleSet (which latched it,
    // classify keys valid THIS frame) with ONE kickoff draw-list build. Without the
    // consume, a frame where buildFrustumVisibleSet is skipped but the kickoff still
    // runs (menus) would re-serve last frame's classify set — dangling NiTriShape
    // pointers under ensureLive.
    if (!s_foldDeferred) return nullptr;
    s_foldDeferred = false;
    return &s_visibleKeys;
}

unsigned DistantLand::lastRefineCulled() {
    return s_refineCulledCount;
}

// buildFrustumVisibleSet - produces the current-frame visible cache set
// (s_frustumVisibleKeys) the Forge kickoff's draw-list build consumes. (Before S5a it also
// fed MGE's own DX9 depth pre-pass and cache colour passes, which is why depth and colour
// were required to replay the SAME set; the host is the only consumer now.)
//
// Two modes, picked per frame:
//  - MSOC-culled (preferred): the early classify ran this frame
//    (mwse_classifyMainSceneNow), so s_visibleKeys is the engine's EXACT current-frame
//    drawn set - occlusion-culled AND with the engine's own LOD switch already applied
//    (it holds the engine-selected LOD leaf, not every cached level). Keep a cache
//    entry iff the engine drew it. Absence culls, which is safe because the verdict is
//    current-frame (no lag, no leading-edge holes). NO second frustum cull - the engine
//    already frustum-culled and its frustum may differ from ours. worldPickObjectRoot
//    entries (isPickRoot) are outside the world-camera classify, so keep those via
//    frustum. First-person (arm root) is a separate camera and not cached - nothing to do.
//  - Frustum only (fallback): no MSOC this frame (occlusion culling off, plugin absent,
//    or the early classify declined). Deterministic current-frame frustum cull over the
//    full cache, from the game view*proj.
//
// Skinned entries with no usable palette are excluded in both modes - they're never
// drawn and would trip the bound helper (div-by-zero / pts[] overrun); see cachebounds.h.
void DistantLand::buildFrustumVisibleSet(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    MGE_ZoneScopedN("buildFrustumVisibleSet");
    s_foldDeferred = false;   // one-frame latch (Cut 2B fold)
    s_frustumVisibleKeys.clear();
    s_refineCulledCount = 0;

    // NOTE: no empty-cache early-out — on live-draw-build frames the cache fills HERE
    // (ensureLive lazy capture below), and the fallback runs ensureFullWalk first.
    const auto& cacheMap = MGE::GeometryCache::cache();

    D3DXMATRIX viewproj;
    D3DXMatrixMultiply(&viewproj, view, proj);
    ViewFrustum frustum(&viewproj);

    // Engine-classified mode: the engine's authoritative current-frame drawn set,
    // from whichever producer owns the traversal (MGE's absorbed CullShow or msoc).
    // Phase 1 host-cull-only (VK_SCROLL): bypass this branch so the produce feeds off the
    // self-contained frustum-only fallback below (no s_visibleKeys / engine-classify
    // dependency); the Forge host's Hi-Z GPU cull owns occlusion.
    // D3: the old `Configuration.UseOcclusionCulling &&` term is gone - it was already
    // implied (earlyClassifyMainScene returns before latching s_earlyClassifyRan when the
    // knob is off) and it is msoc's switch, which must not gate MGE's own feed.
    if (s_earlyClassifyRan && !hostCullOnly) {
        s_earlyClassifyRan = false;   // one frame only

        // Cut 2B fold: under the Forge seam NOTHING else consumes s_frustumVisibleKeys —
        // the Forge kickoff's draw-list build is the set's only reader.
        // So don't run the ensureLive loop here and copy survivors,
        // only for buildGeometryDrawLists to re-hash the same keys at kickoff: DEFER —
        // hand the raw classify set across (foldVisibleKeys) and let the kickoff build
        // do ensureLive + emit in ONE pass. s_frustumVisibleKeys stays EMPTY on fold
        // frames (cleared above; nothing may consume it stale). The VISKEYS /
        // refineCulled diagnostics simply don't update on fold frames.
        // S5a: the gate also required !UseRenderThread && !ForgeNearDepthReplay — the two
        // flags that could still put a DX9 depth consumer on s_frustumVisibleKeys. Both the
        // render thread and the near-depth replay are gone, so nothing outside the kickoff
        // reads the set at all, and the live-draw-build flag that also rode this gate is now
        // unconditional — the seam owning the frame is the whole condition.
        if (RenderProcess::forgeOwnsFrame()) {
            s_foldDeferred = true;
            return;
        }

        s_frustumVisibleKeys.reserve(s_visibleKeys.size() + 16);
        // Iterate the DRAWN set and probe the cache — not the whole cache probing the
        // drawn set. The drawn set (~3-4k) is a fraction of the cache (~10k+ across
        // 16 cells), so this cuts the per-frame work ~3x; a drawn key with no cache
        // entry simply doesn't match (identical to the old absence-from-cache case).
        // Kept-set is IDENTICAL to the old full-cache intersection; only the emission
        // ORDER changes (set order vs map order — both unordered; the opaque consumers
        // are order-independent).
        // Pick-root entries are NOT frustum-rescued: the visible-geom callback reports
        // occlusion SURVIVORS only (the exact set the engine draws) — keep iff drawn.
        // Diagnostic: split into landscape/object/pick; refineCulled = cache entries
        // NOT drawn this frame (now computed as size difference, incl. stale entries
        // awaiting the deferred eviction sweep — diagnostic only).
        unsigned visLand = 0, visObj = 0, visPick = 0;
        // W3 diagnostic: wall cost of the ensureLive loop (the walk's replacement),
        // averaged into the periodic VISKEYS line below.
        static double s_liveMsAccum = 0.0; static unsigned s_liveN = 0;
        LARGE_INTEGER liveFreq, liveT0, liveT1;
        QueryPerformanceFrequency(&liveFreq);
        QueryPerformanceCounter(&liveT0);
        for (const uint32_t key : s_visibleKeys) {
            // W3 live-read: freshen (or lazily capture) the entry straight off the live
            // NiTriShape — a classify key is engine-drawn THIS frame, so the pointer is
            // valid by construction. On full-walk frames this is a no-op lookup, so the
            // kept-set is identical to the old cacheMap.find probe.
            const auto* e = MGE::GeometryCache::ensureLive(key);
            if (!e) continue;                     // no model data / capture failed
            if (e->isSkinned && (e->skinnedUnsupported || e->numBones == 0)) continue;
            s_frustumVisibleKeys.push_back(key);      // engine drew it this frame
            if (e->isLandscape)    ++visLand;
            else if (e->isPickRoot) ++visPick;
            else                   ++visObj;
        }
        QueryPerformanceCounter(&liveT1);
        s_liveMsAccum += 1000.0 * (double)(liveT1.QuadPart - liveT0.QuadPart) / (double)liveFreq.QuadPart;
        ++s_liveN;
        s_refineCulledCount = (unsigned)(cacheMap.size() - s_frustumVisibleKeys.size());
        if (Configuration.LogDistantPipeline) {
            static unsigned s_n = 0;
            if (++s_n % 300 == 0) {
                LOG::logline("-- [VISKEYS] MSOC: land=%u obj=%u pickVis=%u total=%zu (refineCulled=%u) liveMs=%.2f",
                             visLand, visObj, visPick, s_frustumVisibleKeys.size(), s_refineCulledCount,
                             s_liveMsAccum / (double)s_liveN);
                s_liveMsAccum = 0.0; s_liveN = 0;
            }
        }
        return;
    }

    // Frustum-only fallback: no MSOC this frame. On live-draw-build frames the refresh
    // walk was skipped and this path iterates the WHOLE cache with a freshness filter —
    // pull the full walk in now (idempotent; no-op when onFrameReady already walked).
    MGE::GeometryCache::ensureFullWalk();
    s_earlyClassifyRan = false;
    s_frustumVisibleKeys.reserve(cacheMap.size());
    const auto cacheFrame = MGE::GeometryCache::currentFrame();
    for (const auto& kv : cacheMap) {
        const auto& e = kv.second;
        // Eviction is a periodic sweep now — skip entries the walk no longer visits
        // (despawned/appCulled) or a stale entry could re-enter the visible set here.
        if (e.lastFrame != cacheFrame) continue;
        // FP1a: arm-scene entries are walked fresh every 1st-person frame but belong to
        // the host FP pass only — near the camera they'd otherwise always pass this
        // frustum test and leak into the main visible set.
        if (e.isFP) continue;
        BoundingSphere bs;
        if (e.isSkinned) {
            if (e.skinnedUnsupported || e.numBones == 0) continue;
            cacheSkinnedWorldBounds(e, bs.center, bs.radius);
        } else {
            cacheWorldBounds(e, bs.center, bs.radius);
        }
        if (frustum.ContainsSphere(bs) == ViewFrustum::OUTSIDE) continue;
        s_frustumVisibleKeys.push_back(kv.first);
    }
    if (Configuration.LogDistantPipeline) {
        static unsigned s_n = 0;
        if (++s_n % 300 == 0)
            LOG::logline("-- [VISKEYS] FRUSTUM-FALLBACK (no early classify): total=%zu",
                         s_frustumVisibleKeys.size());
    }
}


// S5a: everything below this point was MGE's DX9 depth pre-pass and its render-thread
// twin. renderDepth() cleared texDepthFrame and replayed the geometry cache into it;
// renderDepthFromCache did the per-entry non-skinned + skinned draws; renderDepthAdditional
// / renderDepthRecorded replayed the recorded scene draws for scenes 1+;
// renderCacheDepthToMainZ re-drew the same set into MW's MAIN depthstencil (the
// ForgeNearDepthReplay seam, off by default); and renderThreadDepthCacheJob /
// snapshotVisibleKeysForThread ran the first two on the MGE render thread during the sky
// window. The only thing that ever sampled the result was the DX9 post chain's depth
// effects (SSAO/DOF), which postProcess() already skipped under the Forge seam — the host
// applies its own GTAO. So the whole stack was producing a texture nobody read.
//
// What survives in this file is the part that was never about drawing: the classify feed
// (updateVisibleSet / earlyClassifyMainScene), the current-frame visible set
// (buildFrustumVisibleSet) and the fold handoff (foldVisibleKeys) — the Forge host's
// draw-list inputs. renderStage0 carries the non-early cache-walk fallback that used to
// sit at the top of renderDepth().
