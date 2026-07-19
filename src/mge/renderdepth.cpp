
#include "configuration.h"
#include "distantland.h"
#include "drawstats.h"
#include "distantshader.h"
#include "mwbridge.h"
#include "renderprocess.h"
#include "phasetimers.h"
#include "proxydx/d3d8header.h"
#include "proxydx/devicelock.h"
#include "scenegraph.h"
#include "scenegraph_geometry_cache.h"
#include "cachebounds.h"
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

// Render-thread snapshot of s_frustumVisibleKeys, populated on the main thread at
// kick (snapshotVisibleKeysForThread) and read by the worker job. Decouples the
// job from any later main-thread mutation of the set.
static std::vector<uint32_t> s_threadVisibleKeys;

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
// before buildFrustumVisibleSet). Ask the plugin to run the world-camera occlusion
// classify NOW; the plugin fires the visible-geom callback synchronously on this
// thread (updateVisibleSet) so the current-frame set is ready for the cull below.
// Latches whether it actually classified (callback fired during the
// call) - the plugin self-declines (root unverified, scene disabled, menu, absent
// export) without signalling, so the tripwire is the authoritative "did it run".
void DistantLand::earlyClassifyMainScene(void* worldCamera) {
    s_earlyClassifyRan = false;
    if (!Configuration.UseOcclusionCulling || !MSOCClient::hasEarlyClassify()) return;

    // Owned display skips (Cut 1 opaque + AT2 alpha): tell the plugin which
    // parts of the frame the Forge side covers THIS frame, before either
    // classify path latches its per-frame state. Opaque bit: the plugin skips
    // engine display() of covered-opaque leaves — the DIPs our proxy rejects
    // per-draw anyway (inspectIndexedPrimitive). Alpha bit: it also skips
    // single-map blended leaves our sorted-alpha host pass redraws (AT1) —
    // gated on SuppressS1 so the F-panel A/B (suppress off = MW's full alpha
    // path) stays intact. F11 (ownsOpaqueWorld false) restores full display
    // next frame. Logged plugin-side on change; absent/old export = opaque-
    // only or no-op = the per-DIP reject fallback keeps correctness.
    int ownedFlags = 0;
    if (RenderProcess::ownsOpaqueWorld()) {
        if (Configuration.ForgeOpaqueDisplaySkip) {
            ownedFlags |= MSOCClient::kOwnedOpaque;
        }
        if (Configuration.ForgeAlphaPass && Configuration.ForgeAlphaSuppressS1) {
            ownedFlags |= MSOCClient::kOwnedAlpha;
        }
    }
    MSOCClient::setOwnedFlags(ownedFlags);

    s_visibleCallbackFired = false;
    // classifyMainSceneNow is a SYNCHRONOUS plugin call: the engine's world-camera
    // classify walk (~10k nodes) AND the owned-mode deferred-display drain both bill
    // here — post-W3 it is the largest un-zoned block in the frameready→stage1
    // window (MSOC.log drainUs alone ~1.8ms), so it gets its own timer + heartbeat.
    LARGE_INTEGER ecFreq, ecT0, ecT1;
    QueryPerformanceFrequency(&ecFreq);
    QueryPerformanceCounter(&ecT0);
    int rc;
    {
        MGE_ZoneScopedN("earlyClassifyMainScene");
        MGE_SCOPED_TIMER("earlyClassifyMainScene");
        rc = MSOCClient::classifyMainSceneNow(worldCamera);
    }
    QueryPerformanceCounter(&ecT1);
    static double s_ecMsAccum = 0.0; static unsigned s_ecN = 0;
    s_ecMsAccum += 1000.0 * (double)(ecT1.QuadPart - ecT0.QuadPart) / (double)ecFreq.QuadPart;
    if (++s_ecN >= 300) {
        LOG::logline(">> [classify] 300 frames avg: classifyMainSceneNow=%.2f ms", s_ecMsAccum / (double)s_ecN);
        s_ecMsAccum = 0.0; s_ecN = 0;
    }
    s_earlyClassifyRan = s_visibleCallbackFired;

    // Diagnostic: surface the plugin's status code so we can see WHETHER the early
    // classify engaged and, if not, which guard declined. rc 0 = classified; non-zero
    // codes (see msoc OcclusionPass.cpp): 2 = engine not in renderMainScene at this
    // point (timing assumption broken), 3 = render root not yet confirmed, 5 = scene
    // already handled, 10 = scene gate declined. Logged on every change + periodically.
    static int s_lastRc = -2;
    static unsigned s_n = 0;
    if (rc != s_lastRc || (++s_n % 600 == 0)) {
        LOG::logline("-- [EARLY CLASSIFY] rc=%d ran=%d (0=classified; 2=!inRenderMainScene "
                     "3=noRootCaptured 5=sceneHandled 10=gateDeclined 11=staleRoot -1=exportAbsent)",
                     rc, s_earlyClassifyRan ? 1 : 0);
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
// (s_frustumVisibleKeys) that the cache depth pre-pass AND the cache opaque/terrain
// colour passes all consume, so depth and colour always replay the SAME set.
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

    // MSOC-culled mode: the engine's authoritative current-frame drawn set.
    // Phase 1 host-cull-only (VK_SCROLL): bypass this branch so the produce feeds off the
    // self-contained frustum-only fallback below (no s_visibleKeys / engine-classify
    // dependency); the Forge host's Hi-Z GPU cull owns occlusion.
    if (Configuration.UseOcclusionCulling && s_earlyClassifyRan && !hostCullOnly) {
        s_earlyClassifyRan = false;   // one frame only

        // Cut 2B fold: in the Forge baseline (F11 composite + F7 water, no render
        // thread, near-depth replay off) NOTHING else consumes s_frustumVisibleKeys —
        // the near depth draws are skipped (forgeOwnsDepth), the render-thread
        // snapshot is off, and the legacy cache color/shadow paths only run without
        // Forge ownership. So don't run the ensureLive loop here and copy survivors,
        // only for buildGeometryDrawLists to re-hash the same keys at kickoff: DEFER —
        // hand the raw classify set across (foldVisibleKeys) and let the kickoff build
        // do ensureLive + emit in ONE pass. s_frustumVisibleKeys stays EMPTY on fold
        // frames (cleared above; nothing may consume it stale). The VISKEYS /
        // refineCulled diagnostics simply don't update on fold frames.
        const bool fold = Configuration.ForgeLiveDrawBuild
            && RenderProcess::ownsOpaqueWorld() && RenderProcess::wantsWaterCapture()  // == forgeOwnsDepth
            && !Configuration.UseRenderThread
            && !Configuration.ForgeNearDepthReplay;
        if (fold) {
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



void DistantLand::renderDepth() {
    MGE_ZoneScopedN("renderDepth");
    MGE_SCOPED_TIMER("renderDepth");
    DrawStats::ScopedStage _ds(DrawStats::Depth);
    auto mwBridge = MWBridge::get();

    // Switch to render target
    RenderTargetSwitcher rtsw(texDepthFrame, surfDepthDepth);

    // When the render thread produced the cleared depth + MW cache depth during
    // the sky window, it already wrote them into texDepthFrame/surfDepthDepth
    // (fenced in renderStage0 before this runs). Skip the Clear, the float-depth
    // clear pass, and the cache pass here; land/statics/grass below render on top
    // of the worker's buffer, byte-identical to the serial path.
    const bool depthCacheOnThread = renderThreadJobKicked;

    // Forge owns the composited frame AND the water (F11 + Forge water always-on): nothing samples
    // texDepthFrame this frame — post-process (SSAO/DOF), the MW↔MGE blend, caustics, and MGE water
    // are all suppressed. So skip PRODUCING the depth texture: the float-depth clear + the cache /
    // land / statics (renderdepth.cpp:310) / grass DEPTH draws. The cache WALK, the frustum-visible
    // set, the IPC channel drain (waitCullChannelFree), and the statics + grass CULLS still run —
    // Forge's draw lists + MGE grass color depend on them, and the statics RPC MUST be drained to
    // keep the one-at-a-time IPC channel paired. F7-off (MGE water A/B) restores the full depth pass.
    const bool forgeOwnsDepth =
        RenderProcess::ownsOpaqueWorld() && RenderProcess::wantsWaterCapture();

    if (!depthCacheOnThread && !forgeOwnsDepth) {
        device->Clear(0, 0, D3DCLEAR_ZBUFFER, 0, 1.0, 0);
    }

    // Unbind depth sampler
    effect->SetTexture(ehTex3, NULL);

    // Projection should cover whole scene (also used by the land/statics depth
    // passes below, so set it on both paths)
    D3DXMATRIX distProj = mwProj;
    editProjectionZ(&distProj, 4.0f, Configuration.DL.DrawDist * kCellSize);
    effect->SetMatrix(ehProj, &distProj);

    if (!depthCacheOnThread) {
        // Clear floating point buffer to far depth (skipped when Forge owns depth — no consumer).
        if (!forgeOwnsDepth) {
            effectDepth->BeginPass(PASS_CLEARDEPTH);
            device->SetVertexDeclaration(WaterDecl);
            device->SetStreamSource(0, vbFullFrame, 0, 12);
            DrawStats::count(2);
            device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
            effectDepth->EndPass();
        }

        // Rebuild the geometry cache (walk + per-frame bone palettes) BEFORE the
        // depth draw so depth uses this frame's skinned poses — otherwise depth
        // lags a frame behind the shadow pass and SSAO detaches from moving NPCs.
        // Cheap now that VS palette skinning replaced per-frame CPU skinning, and
        // the VBs are static so there's no depth/shadow aliasing to overlap around.
        //
        // Skipped when frameSetupEarly() already walked it at BeginScene(0) (IPC
        // path) so the ~2ms walk overlaps the sky pass. This call is the fallback
        // for the non-IPC / menu / not-ready paths where earlyWalkedCache is false.
        // (depthCacheOnThread implies earlyWalkedCache, so this whole block is
        // skipped on the threaded path — the worker did the cache pass.)
        if (!earlyWalkedCache) {
            MGE::GeometryCache::onFrameReady(MGE::SceneGraph::getDataHandler());
            // Non-IPC / menu / not-ready fallback: frameSetupEarly didn't run the
            // early walk, so it didn't build the frustum-visible set either. Build
            // it here (after the walk, before the consume) so this path drives off
            // the same deterministic current-frame set as the IPC/threaded paths.
            // (The early classify itself must run at BeginScene(0) in frameSetupEarly,
            // before the engine's CullShow — running it here is too late: the engine's
            // MSOC is already active and declines with rc=1. frameSetupEarly runs it
            // for interiors too, so s_earlyClassifyRan may already be latched here.)
            buildFrustumVisibleSet(&mwView, &mwProj);
        }
        // Cache near-depth draw (skipped when Forge owns depth — no consumer). The WALK above still
        // ran (feeds Forge's draw lists); only the texDepthFrame draw is elided.
        if (!forgeOwnsDepth) {
            MGE_SCOPED_TIMER("renderDepth:cache");
            renderDepthFromCache(&mwView);   // owns its non-skinned + skinned passes
        }
    }

    // Channel-free gate: when frameSetupEarly dispatched the statics verdict to
    // the cull worker, the worker holds the single-channel ipcClient until it
    // has drained the statics RPC. Block here — after the cache-only depth pass
    // above (which touches no ipcClient and overlaps the drain), and before any
    // main-thread ipcClient touch downstream (distant statics consume / grass /
    // shadow / land / water RPCs) — so none of them race the worker's drain.
    // No-op when the worker path is inactive. Now that the kickoff fires before
    // the GeometryCache walk, the drain typically completes during sky + walk
    // and this reads ~0.
    waitCullChannelFree();

    if (isDistantCell()) {
        if (!mwBridge->IsUnderwater(eyePos.z)) {
            // Distant land depth (skipped when Forge owns depth — no consumer).
            if (mwBridge->IsExterior() && !forgeOwnsDepth) {
                MGE_ZoneScopedN("renderDepth:land");
                MGE_SCOPED_TIMER("renderDepth:land");
                effectDepth->BeginPass(PASS_RENDERLANDDEPTH);
                renderDistantLandZ();
                effectDepth->EndPass();
            }

            // Finish the async statics cull here so depth and color both
            // consume the same msocOccluded mask. Moved from Stage0's
            // color pass so it overlaps with land depth on the GPU.
            // SURVIVES forgeOwnsDepth: the statics RPC must be drained to keep the IPC channel paired.
            // Skipped on early-Forge-kickoff frames: the cull kickoff was gated off in
            // frameSetupEarly (no RPC to drain), keeping the kickoff/finish pairing symmetric.
            if ((Configuration.MGEFlags & USE_DISTANT_STATICS) && !earlyForgeKickoff) {
                cullDistantStatics_finish();
            }

            // Distant statics depth (renderdepth.cpp:310 — skipped when Forge owns depth, no consumer).
            if (!forgeOwnsDepth) {
                MGE_ZoneScopedN("renderDepth:statics");
                MGE_SCOPED_TIMER("renderDepth:statics");
                DrawStats::ScopedStage _ds(DrawStats::DepthStatics);
                effectDepth->BeginPass(PASS_RENDERSTATICSDEPTH);
                device->SetVertexDeclaration(StaticDecl);
                // Cull-then-sort: iterate the compacted survivor set (occluded
                // instances already removed by applyMSOCToDistantStatics).
                visDistantSurvivors.Render(device, effectDepth, effect, &ehTex0, &ehHasAlpha, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT, false);
                effectDepth->EndPass();
            }
        }

        if (Configuration.MGEFlags & USE_GRASS) {
            // Cull grass here (moved out of Stage0): culling before the depth
            // pre-pass drained the distant-statics RPC on the one-at-a-time IPC
            // channel right after its Stage0 kickoff, before the GeometryCache
            // walk above could overlap that ~2.3ms server-cull. Culling here —
            // after the walk and cullDistantStatics_finish — lets the statics cull
            // overlap the walk, and grass's own RPC is cheap with the channel now
            // free. Grass still renders in this depth pre-pass, so early-Z holds.
            // SURVIVES forgeOwnsDepth: MGE grass COLOR still draws (Forge has no grass) and needs this.
            // On early-Forge-kickoff frames frameSetupEarly already culled it pre-kickoff
            // (this point is inside the IPC-free async window); consume that result.
            if (mwBridge->IsExterior() && !earlyCulledGrass) {
                cullGrass(&mwView, &mwProj);
            }

            // Grass depth (skipped when Forge owns depth — no consumer; the grass COLOR pass in
            // renderStage1 uses the backbuffer depth, not texDepthFrame).
            if (!forgeOwnsDepth) {
                MGE_ZoneScopedN("renderDepth:grass");
                MGE_SCOPED_TIMER("renderDepth:grass");
                effectDepth->BeginPass(PASS_RENDERGRASSDEPTHINST);
                renderGrassInstZ();
                effectDepth->EndPass();
            }
        }
    }

    // Reset projection matrix
    effect->SetMatrix(ehProj, &mwProj);
}

void DistantLand::renderDepthAdditional() {
    MGE_ZoneScopedN("renderDepthAdditional");
    DrawStats::ScopedStage _ds(DrawStats::Depth);
    // Switch to render target
    RenderTargetSwitcher rtsw(texDepthFrame, surfDepthDepth);

    // Unbind depth sampler
    effect->SetTexture(ehTex3, NULL);

    // Projection should cover whole scene
    D3DXMATRIX distProj = mwProj;
    editProjectionZ(&distProj, 4.0f, Configuration.DL.DrawDist * kCellSize);
    effect->SetMatrix(ehProj, &distProj);

    // Recorded draw calls
    effectDepth->BeginPass(PASS_RENDERMWDEPTH);
    renderDepthRecorded();
    effectDepth->EndPass();

    // Reset projection matrix
    effect->SetMatrix(ehProj, &mwProj);
}

void DistantLand::renderDepthRecorded() {
    MGE_ZoneScopedN("renderDepthRecorded");
    // Use an alpha threshold for solidity that isn't precisely equal to a commonly used value (such as 0.5).
    // Vertex interpolators can be slightly inaccurate and cause a value that should be constant across a triangle
    // to have interpolated fragment values that vary either side of the threshold and cause noise.
    const float solidThreshold = 0.499f;

    // Recorded renders
    const auto& recordMW_const = recordMW;
    for (const auto& i : recordMW_const) {
        // Set variables in main effect; variables are shared via effect pool

        // Fragment colour routing
        bool alphaDependent = i.alphaTest || i.blendEnable;
        effect->SetBool(ehHasVCol, alphaDependent && (i.fvf & D3DFVF_DIFFUSE) != 0);
        effect->SetFloat(ehMaterialAlpha, alphaDependent ? i.diffuseMaterial.a : 1.0f);

        // Only bind texture for alphas
        if (alphaDependent && i.texture) {
            effect->SetTexture(ehTex0, i.texture);
            effect->SetBool(ehHasAlpha, true);
            effect->SetFloat(ehAlphaRef, i.alphaTest ? (i.alphaRef / 255.0f) : solidThreshold);
        } else {
            effect->SetTexture(ehTex0, 0);
            effect->SetBool(ehHasAlpha, false);
            effect->SetFloat(ehAlphaRef, -1.0f);
        }

        // Skin using worldview matrices for numerical accuracy
        effect->SetBool(ehHasBones, i.vertexBlendState != 0);
        effect->SetInt(ehVertexBlendState, i.vertexBlendState);
        effect->SetMatrixArray(ehVertexBlendPalette, i.worldViewTransforms, 4);
        effectDepth->CommitChanges();

        device->SetRenderState(D3DRS_CULLMODE, i.cullMode);
        device->SetStreamSource(0, i.vb, i.vbOffset, i.vbStride);
        device->SetIndices(i.ib);
        device->SetFVF(i.fvf);
        DrawStats::count(i.primCount);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);
    }
}

void DistantLand::renderDepthFromCache(const D3DXMATRIX* gameView,
                                       const std::vector<uint32_t>* visibleOverride) {
    MGE_ZoneScopedN("renderDepthFromCache");
    // Cache-geometry depth draws bucket separately from the distant-land depth
    // replays so cache-depth (frustum-only post-Phase-1) can be compared directly
    // against the MSOC-culled scene0. Counter is thread_local-stage safe (render
    // thread vs main both push their own g_stage; the shared array tolerates it).
    DrawStats::ScopedStage _dsCache(DrawStats::DepthCache);

    const float solidThreshold = 0.499f;
    const auto& cacheMap = MGE::GeometryCache::cache();

    auto bindMaterial = [&](const MGE::GeometryCache::CachedGeometry& e) {
        bool alphaDependent = e.alphaTest || e.blendEnable;
        if (alphaDependent && e.d3dTexture) {
            effect->SetTexture(ehTex0, e.d3dTexture);
            effect->SetBool(ehHasAlpha, true);
            effect->SetFloat(ehAlphaRef, e.alphaTest ? e.alphaRef : solidThreshold);
        } else {
            effect->SetTexture(ehTex0, nullptr);
            effect->SetBool(ehHasAlpha, false);
            effect->SetFloat(ehAlphaRef, -1.0f);
        }
        // Mirrored (negative-determinant) parts flip clip-space winding, so cull the
        // opposite face — else depth records the inner surface and SSAO shows
        // "inside-out" left limbs. Matches the engine's per-draw mirror swap.
        const DWORD cull = e.blendEnable ? D3DCULL_NONE
                                         : (e.mirrored ? D3DCULL_CCW : D3DCULL_CW);
        device->SetRenderState(D3DRS_CULLMODE, cull);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
        device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
    };

    // Iterate the deterministic current-frame frustum-visible set (built early by
    // buildFrustumVisibleSet): s_frustumVisibleKeys for the serial path, or the
    // render-thread snapshot of it (visibleOverride) for the threaded job. No
    // full-cache fallback — an empty set means nothing is in frustum, which is the
    // correct verdict (replaces the old s_prevVisibleKeys / unculled-cache branch).
    const std::vector<uint32_t>& keys = visibleOverride ? *visibleOverride : s_frustumVisibleKeys;
    auto forEach = [&](auto&& fn) {
        for (uint32_t key : keys) {
            auto it = cacheMap.find(key);
            if (it != cacheMap.end()) fn(it->second);
        }
    };

    // ---- Non-skinned: model-space VB, per-draw palette[0] = worldTransform*view ----
    effect->SetBool(ehHasVCol, false);
    effect->SetFloat(ehMaterialAlpha, 1.0f);
    effect->SetBool(ehHasBones, false);
    effect->SetInt(ehVertexBlendState, 0);

    effectDepth->BeginPass(PASS_RENDERMWDEPTH);
    forEach([&](const MGE::GeometryCache::CachedGeometry& e) {
        if (e.isSkinned) return;
        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib) return;

        D3DXMATRIX wvMat;
        D3DXMatrixMultiply(&wvMat, reinterpret_cast<const D3DXMATRIX*>(e.worldTransformD3D), gameView);
        D3DXMATRIX wvPalette[4] = { wvMat, wvMat, wvMat, wvMat };
        effect->SetMatrixArray(ehVertexBlendPalette, wvPalette, 4);

        bindMaterial(e);
        effectDepth->CommitChanges();
        // Per-entry stride/FVF: multi-map objects carry extra UV sets. The depth VS
        // reads only TEXCOORD0, but the stream stride MUST match the VB layout or every
        // vertex past the first is misaligned.
        device->SetStreamSource(0, vb, 0, e.vbStride);
        device->SetIndices(e.ib);
        device->SetFVF(e.vbFVF);
        DrawStats::count(e.triangleCount);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, e.vertexCount, 0, e.triangleCount);
    });
    effectDepth->EndPass();

    // ---- Skinned: static bind-pose VB + per-frame bone palette (VS skinning) ----
    // skinIndexed -> view -> proj; view must be the game view, proj is the depth proj.
    effect->SetMatrix(ehView, gameView);

    effectDepth->BeginPass(PASS_RENDERMWDEPTH_SKINNED);
    device->SetVertexDeclaration(MGE::GeometryCache::skinnedDecl());
    forEach([&](const MGE::GeometryCache::CachedGeometry& e) {
        if (!e.isSkinned || e.skinnedUnsupported) return;
        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib || e.numBones == 0) return;

        effect->SetMatrixArray(ehBoneMatrices,
            reinterpret_cast<const D3DXMATRIX*>(e.bonePalette.data()), e.numBones);
        bindMaterial(e);
        effectDepth->CommitChanges();
        device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kSkinnedVBStride);
        device->SetIndices(e.ib);
        DrawStats::count(e.triangleCount);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, e.vertexCount, 0, e.triangleCount);
    });
    effectDepth->EndPass();

    effect->SetTexture(ehTex0, nullptr);
    effect->SetBool(ehHasAlpha, false);
    effect->SetFloat(ehAlphaRef, -1.0f);
}

// renderCacheDepthToMainZ - Forge composite depth seam. In Forge mode the engine's scene-0
// opaque draw is suppressed (inspectIndexedPrimitive), so MW's MAIN depthstencil holds only
// sky + distant land — the near-opaque depth is missing, and s1's sorted-alpha / first-person
// would draw over the Forge walls. Re-draw the SAME cache set Forge renders, DEPTH-ONLY, into
// the main depthstencil using the GAME projection (mwProj — NOT the extended depth proj the
// depth-texture pre-pass uses) so the values match the engine's own s1 depth test. Colour is
// masked off (Forge owns colour via the composite). Reuses renderDepthFromCache's complete
// objects+terrain+skinned iteration and the lean depth effect (no shading). Mirrors the
// interior depth pattern (state block + effectDepth DONOTSAVESTATE). Relies on the main
// backbuffer/depthstencil being the bound RT (true between EndScene s0 and BeginScene s1).
void DistantLand::renderCacheDepthToMainZ() {
    MGE_ZoneScopedN("renderCacheDepthToMainZ");
    UINT passes = 0;
    IDirect3DStateBlock9* sb = nullptr;
    device->CreateStateBlock(D3DSBT_ALL, &sb);

    device->SetRenderState(D3DRS_COLORWRITEENABLE, 0);   // depth only — Forge provides colour
    device->SetRenderState(D3DRS_ZENABLE, TRUE);
    device->SetRenderState(D3DRS_ZWRITEENABLE, TRUE);
    device->SetRenderState(D3DRS_ZFUNC, D3DCMP_LESSEQUAL);

    // GAME projection (not the editProjectionZ-extended distProj) so the hardware Z matches
    // what the engine's s1 pass tests against. renderDepthFromCache reads ehProj for the
    // final projection (ehVertexBlendPalette / ehView carry only world*view).
    effect->SetMatrix(ehView, &mwView);
    effect->SetMatrix(ehProj, &mwProj);
    effect->SetTexture(ehTex3, NULL);

    effectDepth->Begin(&passes, D3DXFX_DONOTSAVESTATE);
    renderDepthFromCache(&mwView);
    effectDepth->End();

    if (sb) { sb->Apply(); sb->Release(); }
}

// snapshotVisibleKeysForThread - copy the early frustum-visible set into the
// render-thread snapshot. MAIN THREAD ONLY, called at kick (frameSetupEarly, right
// after buildFrustumVisibleSet) before the worker reads it. s_frustumVisibleKeys is
// only rewritten by the next frame's buildFrustumVisibleSet, which runs after the
// job is fenced (renderStage0), so the snapshot also decouples the job from that.
void DistantLand::snapshotVisibleKeysForThread() {
    s_threadVisibleKeys.assign(s_frustumVisibleKeys.begin(), s_frustumVisibleKeys.end());
}

// renderThreadDepthCacheJob - Phase 1 render-thread payload.
//
// Runs on the MGE render thread, kicked from frameSetupEarly() during the
// engine's sky window, fenced at the top of renderStage0() before any main-thread
// device/effect work. Holds the device-submission lock for its entire body so it
// is atomic against the engine's proxy forwarders. Produces exactly what the
// serial renderDepth cache path produces — cleared depth + the MW geometry-cache
// depth — into texDepthFrame/surfDepthDepth, just earlier (overlapping sky).
//
// Reads only frame-stable, kick-time-fixed data: mwView/mwProj (set by
// frameSetupEarly before the kick, not rewritten until renderStage0 after the
// fence), the GeometryCache (built by the walk before the kick), and the
// s_threadVisibleKeys snapshot. Uses its OWN effectDepth Begin/End bracket — the
// fence guarantees the main thread is not inside an effect bracket concurrently.
void DistantLand::renderThreadDepthCacheJob() {
    MGE_ZoneScopedN("RenderThread:job");
    MGE_DEVLOCK();   // hold the device lock for the whole pass
    DrawStats::ScopedStage _ds(DrawStats::Depth);  // render-thread stage (thread_local)

    if (!device) {
        return;
    }

    // Save the full device state the engine left mid-sky; restore it before
    // releasing the lock so the engine resumes intact. The render target is not
    // captured by state blocks — RenderTargetSwitcher restores it separately.
    IDirect3DStateBlock9* sb = nullptr;
    if (device->CreateStateBlock(D3DSBT_ALL, &sb) != D3D_OK) {
        sb = nullptr;
    }

    UINT passes;
    {
        RenderTargetSwitcher rtsw(texDepthFrame, surfDepthDepth);
        device->Clear(0, 0, D3DCLEAR_ZBUFFER, 0, 1.0f, 0);

        // Unbind depth sampler
        effect->SetTexture(ehTex3, NULL);

        // Projection should cover whole scene
        D3DXMATRIX distProj = mwProj;
        editProjectionZ(&distProj, 4.0f, Configuration.DL.DrawDist * kCellSize);
        effect->SetMatrix(ehProj, &distProj);

        // Own effect bracket — main is not in one yet (fenced before renderStage0).
        effectDepth->Begin(&passes, D3DXFX_DONOTSAVESTATE);

        // Clear floating point buffer to far depth
        effectDepth->BeginPass(PASS_CLEARDEPTH);
        device->SetVertexDeclaration(WaterDecl);
        device->SetStreamSource(0, vbFullFrame, 0, 12);
        DrawStats::count(2);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
        effectDepth->EndPass();

        renderDepthFromCache(&mwView, &s_threadVisibleKeys);

        effectDepth->End();
    }

    if (sb) {
        sb->Apply();
        sb->Release();
    }
}
