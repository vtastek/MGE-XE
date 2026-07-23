
#include "proxydx/d3d8header.h"
#include "mgedinput.h"
#include "configuration.h"
#include "distantland.h"
#include "distantshader.h"
#include "postshaders.h"
#include "mwbridge.h"
#include "scenegraph.h"
#include "scenegraph_geometry_cache.h"
#include "renderprocess.h"
#include "renderthread.h"
#include "mge_tracy.h"
#include "statusoverlay.h"



using std::string;
using std::unordered_map;

#ifdef TRACY_ENABLE
static constexpr tracy::SourceLocationData s_mwSkyLoc   { "MW sky",       "BeginScene",  __FILE__, 0, 0 };
static constexpr tracy::SourceLocationData s_mwDrawsLoc { "MW draw loop", "renderStage0", __FILE__, 0, 0 };
static tracy::ScopedZone* s_mwSkyZone   = nullptr;
static tracy::ScopedZone* s_mwDrawsZone = nullptr;
#endif

// --- [fse] frameSetupEarly main-thread budget probe --------------------------------
// The frame is client-main-thread bound, not GPU bound: dt ~= mwstart (engine, ~4.2ms)
// + ~6.8ms of our own main-thread work, while render= (the wait on the host) sits at
// ~0.2-1.0ms. frameSetupEarly is where most of that work lives and it has never been
// zoned, so the 6.8ms is unattributed. Time every step with QPC and average over the
// same 300-frame window as [hb]. Steps of interest: `cell` issues a BLOCKING setWorldSpace
// RPC on main, and `walk` is the geometry-cache walk — the relocation target. Only frames
// that pass the ready/UseSharedMemory gate are counted (menu/load frames are exactly the
// load-burst contamination that made the first [s0cost] reading useless).
namespace {

inline double fseNowMs() {
    static LARGE_INTEGER freq = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)freq.QuadPart;
}

constexpr unsigned kFseWindow = 300;
struct FseAccum {
    double lights, cell, view, statics, walk, classify, vis, kick, rt, total;
    double maxTotal;
    unsigned n, extN;
};
FseAccum s_fse = {};

// [mwdraw] probe state: opened in beginDrawsZone (first world DIP of scene 0), closed at
// renderStage1 (EndScene(0)). 0 = the window never opened this frame (MW culled everything,
// so EndScene took the !stage0Complete path and no draw ever triggered it).
double s_mwDrawsT0 = 0.0;

void mwDrawsAccum(double ms) {
    static double s_sum = 0.0, s_max = 0.0;
    static unsigned s_n = 0, s_skipped = 0;
    constexpr double kSteadyMs = 25.0;   // same burst gate as [s0cost]
    if (ms > kSteadyMs) { ++s_skipped; return; }
    s_sum += ms; if (ms > s_max) s_max = ms;
    if (++s_n < kFseWindow) return;
    LOG::logline(">> [mwdraw] %u frames avg: window=%.3f ms (max=%.3f, %u burst dropped) "
                 "-- first world DIP -> EndScene(0)",
                 s_n, s_sum / s_n, s_max, s_skipped);
    s_sum = 0.0; s_max = 0.0; s_n = 0; s_skipped = 0;
}

// One call at the end of frameSetupEarly; logs the averaged breakdown every window.
void fseAccum(double lights, double cell, double view, double statics, double walk,
              double classify, double vis, double kick, double rt,
              double total, bool exterior) {
    s_fse.lights += lights; s_fse.cell += cell; s_fse.view += view;
    s_fse.statics += statics; s_fse.walk += walk; s_fse.classify += classify;
    s_fse.vis += vis; s_fse.kick += kick; s_fse.rt += rt;
    s_fse.total += total;
    if (total > s_fse.maxTotal) s_fse.maxTotal = total;
    if (exterior) ++s_fse.extN;
    if (++s_fse.n < kFseWindow) return;

    const double inv = 1.0 / s_fse.n;
    // accounted = the sum of the named steps; total - accounted is frameSetupEarly's own
    // residue (branch/gate work). A large residue means the split is missing a step.
    const double accounted = (s_fse.lights + s_fse.cell + s_fse.view + s_fse.statics
                            + s_fse.walk + s_fse.classify + s_fse.vis
                            + s_fse.kick + s_fse.rt) * inv;
    LOG::logline(">> [fse] %u frames avg: total=%.2f | lights=%.2f cell=%.2f view=%.2f "
                 "statics=%.2f walk=%.2f classify=%.2f vis=%.2f kick=%.2f rt=%.2f "
                 "=> accounted=%.2f residue=%.2f | max total=%.2f ext=%u",
                 s_fse.n, s_fse.total * inv, s_fse.lights * inv, s_fse.cell * inv,
                 s_fse.view * inv, s_fse.statics * inv, s_fse.walk * inv,
                 s_fse.classify * inv, s_fse.vis * inv,
                 s_fse.kick * inv, s_fse.rt * inv,
                 accounted, s_fse.total * inv - accounted,
                 s_fse.maxTotal, s_fse.extN);
    s_fse = {};
}

} // namespace

// Set by frameSetupEarly() when the per-frame statics-cull setup ran at
// BeginScene(scene 0); read by renderStage0 to skip the redundant work.
static bool s_frameSetupEarly   = false;  // selectDistantCell + camera/fog setup done early
static bool s_earlyKickedStatics = false; // cullDistantStatics_kickoff already issued early

// Run at BeginScene(scene 0), before the engine renders sky. The camera is
// already this-frame-valid here (verified: BeginScene-vs-Stage0 view/proj
// delta = 0), so we can run the distant-statics cull prerequisites and kick
// the cull off now — its ~4ms server-side compute then overlaps the ~1.5ms
// sky window plus the GeometryCache walk, instead of stalling cullDistantStatics_finish.
void DistantLand::frameSetupEarly() {
    MGE_ZoneScopedN("frameSetupEarly");
    // [fse] step stamps (see fseAccum above). Deltas stay 0 for steps this frame skipped.
    const double tFse0 = fseNowMs();
    double dStatics = 0, dWalk = 0, dClassify = 0, dVis = 0, dKick = 0, dRt = 0;

    s_frameSetupEarly = false;
    s_earlyKickedStatics = false;
    earlyWalkedCache = false;
    renderThreadJobKicked = false;
    earlyForgeKickoff = false;

    // Drive the scene-graph lights snapshot here (moved from renderStage0, which
    // runs *after* the engine's ~1.6ms sky pass). On the async path onFrameReady
    // only signals the worker, so rebuildAsync now runs concurrent with sky
    // instead of starting after it. Fires on every path (incl. non-IPC), so it
    // sits before the shared-memory gate below. Pose-safe: the scene graph is
    // fully posed for the frame at BeginScene(0) — same transforms renderStage0
    // would see (verified camera delta = 0).
    MGE::SceneGraph::onFrameReady();

    // Bin point lights into screen tiles for the tiled FFE path (USE_TILED_LIGHTS),
    // main view only. Runs right after the snapshot drive above and BEFORE the
    // UseSharedMemory gate below — tiled isn't IPC-specific. No-op unless tiled
    // lighting is active (config flag + VK_DECIMAL toggle) and the snapshot has
    // point lights. The camera is this-frame-valid here (BeginScene scene 0), which
    // matches the view/proj the main draws use, so the grid lines up with VPOS.
    FixedFunctionShader::buildTileGrid();
    const double tLights = fseNowMs();

    // Only the IPC (shared-memory) path benefits from the early statics/geometry
    // work: there the cull is async and overlaps. The non-IPC path does
    // synchronous quadtree work in the kickoff, which has no overlap to gain —
    // leave selectDistantCell + the GeometryCache walk in renderStage0/renderDepth.
    // Not counted in [fse]: menu/load/non-IPC frames are the load-burst contamination
    // that has to stay out of the steady-state budget.
    if (!ready || !device || !Configuration.UseSharedMemory) return;

    auto mwBridge = MWBridge::get();

    // selectDistantCell issues the blocking setWorldSpace RPC + cell-change
    // scan; running it here overlaps it with sky too. setView/adjustFog are
    // pure computation into members (no device/effect side-effects — those live
    // in setupCommonEffect/updateLighting, which stay in renderStage0).
    selectDistantCell();
    const double tCell = fseNowMs();
    device->GetTransform(D3DTS_VIEW, &mwView);
    device->GetTransform(D3DTS_PROJECTION, &mwProj);
    setView(&mwView);
    adjustFog();
    s_frameSetupEarly = true;
    const double tView = fseNowMs();

    // Async full-frame overlap (Phase 2): decide whether THIS frame runs the early
    // Forge kickoff at BeginScene(0), making the whole scene 0 the IPC-free window.
    // Requires the Forge baseline — seam compositing (F11) —
    // because exactly then every MGE consumer of the statics cull, shadow, reflection
    // and depth-production RPCs is suppressed, so the main channel can be vacated for
    // the async RenderFrame. Cell shape (2a + 2b):
    //   - exterior distant cell    → eligible; the statics cull is gated off below.
    //   - plain interior / any non-distant cell → eligible; NO scene-0 RPC exists there
    //     at all (no statics/grass/land cull; no weather → no shadow map; the interior
    //     reflection RPC is forgeOwnsFrame-suppressed). The else-branch below hoists
    //     the cache walk + visible set so the kickoff has fresh data (2b).
    //   - interior DISTANT cell (worldspace interior — DL gen bakes LOD for it; the
    //     TR-showcase category) → eligible TOO: on kickoff frames the statics cull and
    //     the PASS_RENDERSTATICSINTERIOR colour draw both self-gate off the SAME
    //     (USE_DISTANT_STATICS && !earlyForgeKickoff) predicate (kickedOffDistantStatics
    //     = false → visDistant.RemoveAll(); renderDepth skips the finish — pairing stays
    //     symmetric). Interior-worldspace LOD is deliberately NOT rendered in Forge mode
    //     (host DL is exterior-only; interior LOD is future host work), so the cull was a
    //     dead RPC squatting the channel — and it silently forced these monster interiors
    //     to the fused serial render (client blocked for the whole host GPU frame).
    // MENUS ARE ORDINARY FORGE FRAMES (S2). The old gate excluded them, so a menu frame fell to
    // the LATE kickoff (mged3d8device.cpp) — the fused serial render, client blocked for the whole
    // host GPU frame — while ALSO still running the legacy DX9 work the composite then overwrote.
    // That was the "menus are slow" report. The host owns the frame in menus exactly as it does in
    // play (forgeOwnsFrame is not menu-aware and the composite is on screen there), so there is no
    // reason for the frame SHAPE to differ. The one thing menus really do change — renderStage0 may
    // take the render-cached path and never reach cullDistantStatics_finish — is a property of the
    // statics-cull RPC pairing, so that guard now lives on the RPC itself (below) instead of
    // disqualifying the whole early stage.
    //
    // The one-frame warm-up latch STAYS, but its trigger is now explicit. It used to reset only as
    // a SIDE EFFECT of the menu term above: loading screens are menus, so a load made the frame
    // ineligible and the first frame back was a warm-up. Dropping !IsMenu() therefore deleted the
    // latch's only reset trigger along with the menu exclusion — and the first frame after a load
    // then kicked the host before the freshly-walked cache had shipped its geometry, rendering an
    // empty fogged world (in-game, 2026-07-22). isLoadingBar() is the signal the menu term was
    // accidentally standing in for, so name it directly: menus stay eligible (the FPS win), loads
    // still get their warm-up frame.
    //
    // What the warm-up frame buys, on top of the geometry above: the host would otherwise render a
    // transition frame with sun/ambient captured from the PREVIOUS environment (those arrive
    // mid-scene-0 via SetLight(6)/D3DRS_AMBIENT, i.e. after the early kickoff has already fired).
    // buildFrameLighting reads sun dir/diffuse/ambient LIVE from the scene graph now
    // (getSceneSunlight, interiors and exteriors both), so only the exterior D3DRS_AMBIENT global
    // (ambCol) is still captured state across a transition.
    static bool s_forgePrevEligible = false;
    const bool loadingBar = mwBridge->isLoadingBar();
    const bool forgeEligibleNow = Configuration.UseAsyncHostFrame
        && RenderProcess::forgeOwnsFrame()
        && !loadingBar;
    earlyForgeKickoff = forgeEligibleNow && s_forgePrevEligible;
    s_forgePrevEligible = forgeEligibleNow;

    // S2 verification counter (temporary): the change's premise is that every Forge frame is now an
    // early-kickoff frame, so anything below 100% is a frame shape we did not account for. Also
    // quantifies the residual "menus are slower" report by splitting the FRAME PERIOD (successive
    // frameSetupEarly calls, i.e. wall clock per MW frame) into menu vs play. If menu frames are
    // still slower with both at 100% early, the cost is OUTSIDE the stage machinery this stage
    // touches — MW's own UI DIP storm through the proxy, post-process, and the composite — not the
    // frame shape. Drop this whole block once S2 is signed off.
    {
        static unsigned s_n = 0, s_early = 0, s_menuN = 0, s_menuEarly = 0;
        static double s_prevMs = 0.0, s_playSum = 0.0, s_menuSum = 0.0, s_menuMax = 0.0;
        const bool inMenu = mwBridge->IsMenu();
        const double nowMs = fseNowMs();
        const double period = (s_prevMs > 0.0) ? (nowMs - s_prevMs) : 0.0;
        s_prevMs = nowMs;

        ++s_n;
        if (earlyForgeKickoff) { ++s_early; if (inMenu) ++s_menuEarly; }
        // Drop load bursts (>100ms) so a cell load can't masquerade as menu cost.
        if (period > 0.0 && period < 100.0) {
            if (inMenu) { ++s_menuN; s_menuSum += period; if (period > s_menuMax) s_menuMax = period; }
            else        { s_playSum += period; }
        }
        if (s_n >= 600) {
            const unsigned playN = (s_n > s_menuN) ? (s_n - s_menuN) : 0;
            LOG::logline(">> [s2] earlyForgeKickoff %u/%u (%.1f%%) | play %u frames avg %.2f ms | "
                         "menu %u frames avg %.2f ms (max %.2f, early %u)",
                         s_early, s_n, 100.0 * s_early / (double)s_n,
                         playN, playN ? s_playSum / playN : 0.0,
                         s_menuN, s_menuN ? s_menuSum / s_menuN : 0.0, s_menuMax, s_menuEarly);
            s_n = s_early = s_menuN = s_menuEarly = 0;
            s_playSum = s_menuSum = s_menuMax = 0.0;
        }
    }

    // MENU FREEZE ("pause world during menus", USE_MENU_CACHING). Menu mode pauses MW's
    // simulation, so re-rendering the world every frame is pure waste — re-show the last host
    // frame instead. Everything below this point is world work (grass cull, park fire, statics
    // cull, cache walk, classify, visible set, produce kick) and the caller then skips the host
    // kickoff, so the whole client+host world pipeline drops out; only MW's UI still draws, over
    // the composite blit of g_mainTex. See the header for the full contract.
    //
    // NOT a hard freeze — a LOW-RATE refresh. The world still renders every kMenuRefreshEvery-th
    // frame, plus a short burst after each click. The first cut froze on everything except the
    // click frame itself, which was wrong twice over (in-game, 2026-07-23):
    //   - "item appears/disappears after the SECOND click": the refresh frame is synchronous with
    //     the click, but MW applies the drop/pickup AFTER it, so the fresh frame still showed the
    //     old world and the change only surfaced on the NEXT click's refresh. A click is a poor
    //     proxy for "the world changed" — it fires strictly too early.
    //   - anything that evolves on its own (a caster settling, an effect finishing) can never
    //     appear at all, because a genuinely frozen image cannot change and the host's own
    //     per-frame settle/expiry logic stops advancing.
    // A refresh cadence fixes both without needing to predict what changed: any world change shows
    // up within kMenuRefreshEvery frames, the host keeps ticking, and the click burst keeps direct
    // interaction feeling immediate. At 8 we still skip ~7/8 of all world rendering.
    //
    // Requires a valid composited frame to re-show, so a host death falls straight back to normal
    // rendering. Not gated on isLoadingBar: load screens ARE menus, but earlyForgeKickoff is
    // already false there (forgeEligibleNow), and menuFreeze implies it.
    constexpr int kMenuRefreshEvery  = 8;   // world refresh cadence while a menu is open
    constexpr int kMenuClickRefresh  = 4;   // frames of normal rendering after a click (MW applies
                                            // the drop/pickup a frame or two after the release edge)
    static int s_menuFrame = 0, s_menuClickFrames = 0;

    const bool freezeEligible = earlyForgeKickoff
                             && (Configuration.MGEFlags & USE_MENU_CACHING)
                             && mwBridge->IsMenu()
                             && RenderProcess::hasCompositeFrame();
    if (!freezeEligible) {
        // Leaving the menu (or losing the composite) resets the cadence so the next menu always
        // opens on a freshly rendered frame rather than mid-cycle.
        s_menuFrame = s_menuClickFrames = 0;
        menuFreeze = false;
    } else {
        if (MGEProxyDirectInput::mouseClick) {
            s_menuClickFrames = kMenuClickRefresh;
        }
        bool refresh = ((s_menuFrame++ % kMenuRefreshEvery) == 0);   // frame 0 of the menu renders
        if (s_menuClickFrames > 0) {
            --s_menuClickFrames;
            refresh = true;
        }
        menuFreeze = !refresh;
    }
    if (menuFreeze) {
        // The IPC channel is left exactly as onFrameAheadCollect (which ran at BeginScene(0),
        // before this) left it: produce drained, any deferred/parked finish stashed and closed.
        // Nothing is issued while frozen, so there is no pairing to maintain — the next unfrozen
        // frame kicks from a clean slate.
        //
        // But captureAlphaDraw keeps appending for the blended world DIPs MW still issues behind
        // the menu, and its only clear point (swapCaptureBuffers) rides the produce we just
        // skipped. Drop the pending captures or they grow for the life of the menu.
        RenderProcess::discardPendingCaptures();
        return;
    }

    // S4: the early grass cull sat here. Grass was the one scene-0 RPC with a live
    // consumer in Forge mode (MGE drew grass colour; the host has no grass yet), so it
    // had to run ahead of the park fire rather than be gated off. MGE's grass renderer
    // is gone now; the host reimplements it from distant-land data.

    // PRODUCE MODE 3 FIRE POINT: ship the payload the worker parked last frame, restamped
    // with this frame's camera. Everything is in place exactly here — camera fresh (read
    // above), latch decided, IPC window closed (the BeginScene(0) collect), channel free
    // (grass done above), and the classify/walk/build/render-thread work all still ahead —
    // so the host gets essentially the whole client frame to render. Before the
    // isDistantCell split so exteriors and eligible interiors share it. No-op outside
    // mode 3 / frame-ahead / early-kickoff frames.
    const double tFire0 = fseNowMs();
    RenderProcess::fireParked(device);
    dKick = fseNowMs() - tFire0;   // folded with the produce kick below into [fse] kick=

    // W1.5 active-cell gate for the GeomCache walk: when Forge owns the opaque world,
    // every cache consumer (near draw lists, classify-driven visible set) is bounded
    // by MW's own view distance — subtrees beyond it can never be drawn, so the walk
    // skips them wholesale. Radius must reach the frustum's far CORNERS: the engine
    // culls on view-z, so a corner object sits at viewDist * sqrt(1 + tanX² + tanY²)
    // Euclidean from the eye (tan factors from the live projection — fov is moddable).
    // Exterior only (interiors are one root, nothing to cut) and gated by
    // forgeOwnsFrame so legacy consumers that reach past the view distance (MGE
    // shadows/reflections) are never starved. 0 disables the gate (full walk).
    float cacheGateRadius = 0.0f;
    if (Configuration.ForgeActiveCellWalk
            && RenderProcess::forgeOwnsFrame() && mwBridge->IsExterior()) {
        const float tanX = (mwProj._11 != 0.0f) ? 1.0f / mwProj._11 : 1.0f;
        const float tanY = (mwProj._22 != 0.0f) ? 1.0f / mwProj._22 : 1.0f;
        cacheGateRadius = mwBridge->GetViewDistance()
                        * sqrtf(1.0f + tanX * tanX + tanY * tanY) + 1024.0f;
    }
    // W3 live-read at build: skip the refresh walk on Forge-owned frames — the
    // classify-visible keys are freshened off their live NiTriShapes inside
    // buildFrustumVisibleSet (ensureLive), with a full-walk fallback when no
    // classify ran. Exteriors AND interiors (classify runs in both).
    // Phase 1 host-cull-only: live-draw-build leans on the engine classify to discover
    // newly-visible objects (lazy capture from s_visibleKeys). With the classify decoupled,
    // force it off so onFrameReady's full refresh walk does the discovery instead.
    const bool liveDrawBuild =
        Configuration.ForgeLiveDrawBuild && RenderProcess::forgeOwnsFrame() && !hostCullOnly;

    if (isDistantCell()) {
        // Kick the distant-statics cull FIRST — before the GeometryCache walk —
        // so the cull worker's IPC drain (the server-side quadtree cull, ~2.7ms
        // in heavy scenes) overlaps BOTH the ~1.46ms cache walk below (main
        // thread, no ipcClient) AND the engine's ~2ms sky pass. Previously the
        // kickoff ran after the walk, so the drain only overlapped sky and its
        // tail stalled the main thread at the renderStage0 channel-free gate.
        // The kickoff needs just the camera (selectDistantCell + setView, done
        // above), not the geometry cache — they share no state.
        //
        // GATED OFF under the early Forge kickoff: the host renders those distant
        // statics itself (GPU cull over the resident set) and every MGE draw that
        // consumed this cull is suppressed in this mode, so the RPC is dead weight —
        // and it would squat the main channel exactly where the async RenderFrame
        // needs it. First cut of the MGE gutting, not a workaround; F11-off or
        // fused mode restores it (clean A/B).
        //
        // ALSO skipped in menus (S2 moved this guard down from the branch condition): renderStage0
        // may take the render-cached path in a menu and never reach cullDistantStatics_finish, so
        // the RPC would go unpaired. That is safe in itself — the next frame's WAIT_FOR_PREVIOUS
        // drains it — but it is work nobody consumes. This is the ONLY part of the early stage that
        // menus genuinely change; everything else below runs in menus exactly as in play.
        // S4b: the distant-statics cull RPC + MSOC verdict dispatch were here. The whole
        // chain is gone — the Forge host loads, culls (two-phase Hi-Z on the GPU) and
        // draws distant statics itself. s_earlyKickedStatics stays false forever now.
        const double tStatics0 = fseNowMs();
        dStatics = fseNowMs() - tStatics0;

        // Build the geometry cache (walk + VB uploads). The ~1.46ms main-thread
        // walk now overlaps both the sky pass and the worker drain kicked above;
        // renderDepth skips its own call when earlyWalkedCache is set. Cache
        // CONSUME (renderDepthFromCache) stays in renderDepth where the render
        // target/effect are bound. Touches no ipcClient, so it can run while the
        // worker drains the statics RPC on the single channel.
        const double tWalk0 = fseNowMs();
        MGE::GeometryCache::onFrameReady(MGE::SceneGraph::getDataHandler(),
                                         &eyePos.x, cacheGateRadius, liveDrawBuild);
        earlyWalkedCache = true;
        dWalk = fseNowMs() - tWalk0;

        // Stage 2 early classify. Ask the plugin to run the engine's world-camera
        // occlusion classify NOW (before the engine's own renderMainScene CullShow), so
        // the visible/occluded callbacks fire with the CURRENT-frame set. The plugin's
        // natural scene-0 CullShow then only displays the survivors. Must run before
        // buildFrustumVisibleSet (which consumes the result) and before the render-thread
        // kick (so the snapshot below captures the engine-set). Null camera = plugin
        // resolves the world camera. No-op (frustum cull stands) if the plugin predates
        // the export or self-declines (root unverified, scene disabled).
        //
        // SKIPPED under host-cull-only: renderdepth.cpp's MSOC branch is gated on
        // (UseOcclusionCulling && s_earlyClassifyRan && !hostCullOnly), so with the host's Hi-Z
        // GPU cull owning occlusion the classify's answer is DISCARDED — we were paying ~1.4ms of
        // main-thread time per frame for a result nobody reads, and paying it on the critical path
        // between MW's physics and the produce kick, delaying the host RPC by its full duration.
        // buildFrustumVisibleSet falls back to the self-contained frustum cull (the same path the
        // plugin-absent case already takes), and the host culls what the frustum over-includes.
        if (!hostCullOnly) {
            const double tCls0 = fseNowMs();
            earlyClassifyMainScene(nullptr);
            dClassify = fseNowMs() - tCls0;
        }

        // Build the current-frame visible set over the fresh cache, using the camera
        // read above (mwView/mwProj). This is the set MGE owns: the depth pre-pass
        // (renderDepthFromCache) and the cache opaque color pass both consume it. When
        // the early classify ran (engine-set mode), it is the engine's exact drawn set
        // (occlusion-culled); otherwise it is the frustum cull (optionally MSOC-refined)
        // — either way current-frame, so leading-edge tiles a pan reveals get depth (no
        // sky holes). Must run before snapshotVisibleKeysForThread (the render-thread
        // job reads a snapshot of it). ~0.4ms, overlapping the sky window.
        const double tVis0 = fseNowMs();
        buildFrustumVisibleSet(&mwView, &mwProj);
        dVis = fseNowMs() - tVis0;

        // FRAME-START KICK. Everything the produce consumes now exists (camera, cache walk,
        // classify, visible set) and the last scene-0 RPC (the grass cull, hoisted above the
        // fire point pre-latch) has vacated the channel — so dispatch the worker HERE rather
        // than after frameSetupEarly returns. The render-thread kick below and the rest of the
        // frame then overlap the ~3.5ms build instead of sitting in front of it. No-op outside
        // async OVERLAP/PARK modes / early-kickoff frames.
        const double tKick0 = fseNowMs();
        RenderProcess::kickProduceEarly(device);
        dKick += fseNowMs() - tKick0;

        // Kick the render-thread depth-cache job now — after the geometry-cache
        // walk (so the cache + VBs it consumes are stable) and before the engine's
        // sky pass — so the worker's ~1ms submission CPU overlaps the engine's
        // non-device sky-prep CPU. Fenced at the top of renderStage0
        // (RenderThread::wait) before any main-thread device/effect work; renderDepth
        // then skips its own Clear/clear-depth/cache and renders land/statics/grass
        // onto the worker's depth buffer. Snapshot the visible-key set here (main
        // thread) so the worker never races updateVisibleSet.
        if (Configuration.UseRenderThread) {
            const double tRt0 = fseNowMs();
            snapshotVisibleKeysForThread();
            MGE::RenderThread::kick(&DistantLand::renderThreadDepthCacheJob);
            renderThreadJobKicked = true;
            dRt = fseNowMs() - tRt0;
        }
    } else {
        // Interior / non-distant cell: the exterior early block above is skipped (it
        // exists for the distant-statics overlap, which interiors don't have), so its
        // cache walk + buildFrustumVisibleSet fall through to the serial renderDepth
        // path. But the early classify MUST run HERE, at BeginScene(0) before the
        // engine's scene-0 CullShow — running it later in renderDepth is too late (the
        // engine's MSOC is already active and declines with rc=1, so interiors fell to
        // a pure-frustum visible set and drew clutter the engine occludes). This fills
        // s_visibleKeys + latches s_earlyClassifyRan; renderDepth's buildFrustumVisibleSet
        // then consumes it (MSOC branch) once it has walked the cache.
        // Same host-cull-only skip as the exterior branch above — the result is discarded when
        // the host's Hi-Z GPU cull owns occlusion, so don't pay for it on the critical path.
        if (!hostCullOnly) {
            const double tCls0 = fseNowMs();
            earlyClassifyMainScene(nullptr);
            dClassify = fseNowMs() - tCls0;
        }

        // 2b: early-Forge-kickoff frames need the kickoff's inputs ready NOW — hoist the
        // cache walk + current-frame visible set here (the exact pair renderDepth's
        // !earlyWalkedCache fallback runs, in the same order: classify above, walk, build).
        // renderDepth then skips its own copies. Non-latched frames keep the serial
        // renderDepth path untouched.
        if (earlyForgeKickoff) {
            const double tWalk0 = fseNowMs();
            MGE::GeometryCache::onFrameReady(MGE::SceneGraph::getDataHandler(),
                                             &eyePos.x, cacheGateRadius, liveDrawBuild);
            earlyWalkedCache = true;
            const double tVis0 = fseNowMs();
            dWalk = tVis0 - tWalk0;
            buildFrustumVisibleSet(&mwView, &mwProj);
            // Frame-start kick (see the exterior branch): interiors have no scene-0 RPC at all, so
            // the channel is free the moment the visible set is built.
            const double tKick0 = fseNowMs();
            dVis = tKick0 - tVis0;
            RenderProcess::kickProduceEarly(device);
            dKick += fseNowMs() - tKick0;
        }
    }

    const double tEnd = fseNowMs();
    fseAccum(tLights - tFse0, tCell - tLights, tView - tCell, dStatics, dWalk,
             dClassify, dVis, dKick, dRt, tEnd - tFse0,
             isDistantCell() && !mwBridge->IsMenu());
}

void DistantLand::beginSkyZone() {
#ifdef TRACY_ENABLE
    if (g_tracyActive) s_mwSkyZone = new tracy::ScopedZone(&s_mwSkyLoc, 0, true);
#endif
}

void DistantLand::beginDrawsZone() {
    // [mwdraw] Always-on QPC twin of the Tracy "MW draw loop" zone. That zone reads ~2.1ms
    // and stayed ~2.1ms even with every world root appCulled (2.0 DIPs/frame), which is not
    // physically sensible — and [s0cost] since showed only 0.04ms/frame is spent inside our
    // DIP handler across 246 draws. Every DX9 stage inside this window (renderStage1/2/Blend/
    // Water) already early-returns under Forge, so there is no MGE work left in it either.
    // Either the window really does hold 2.1ms of engine/driver time, or the Tracy zone is an
    // artifact like the "MW sky gap" was. Tracy can't answer that about itself; QPC in the
    // shipping build can.
    s_mwDrawsT0 = fseNowMs();
#ifdef TRACY_ENABLE
    if (g_tracyActive) s_mwDrawsZone = new tracy::ScopedZone(&s_mwDrawsLoc, 0, true);
#endif
}

// Handover band clip plane. Builds a view-space-z slab in CLIP space (vs_3_0 user
// clip planes are evaluated in clip space) so it must be constructed from the SAME
// projection the bracketed geometry is drawn with. The slab is independent of view
// orientation — it depends only on the projection's _33/_43 (the view-z -> clip-z/w
// mapping). keepNear=true keeps view-z > d (cull-near: bound LOD to the band START);
// keepNear=false keeps view-z < d (cull-far: bound near content to the band END).
// Mirrors the proven forms at renderexterior.cpp (interior statics) and the water
// fillrate clip. Caller enables D3DRS_CLIPPLANEENABLE and disables it afterwards.
static D3DXPLANE makeBandClipPlane(const D3DXMATRIX& proj, float d, bool keepNear) {
    const float a = proj._33 * d + proj._43;
    return keepNear ? D3DXPLANE(0, 0, d, -a)    // keep view-z > d
                    : D3DXPLANE(0, 0, -d, a);   // keep view-z < d
}

// renderStage0 - Render distant land at beginning of scene 0, after sky
void DistantLand::renderStage0() {
#ifdef TRACY_ENABLE
    delete s_mwSkyZone;
    s_mwSkyZone = nullptr;
#endif
    MGE_ZoneScopedN("Stage0");
    auto mwBridge = MWBridge::get();
    IDirect3DStateBlock9* stateSaved;
    UINT passes;

    // Render-thread fence. Join the worker BEFORE any main-thread device or
    // ID3DXEffect work below (setupCommonEffect, the depth/shadow/distant passes
    // all share the single effect/effectDepth objects, which are not safe to use
    // concurrently with the worker). Kicked at BeginScene(0)/frameSetupEarly, the
    // job has had the whole sky window to finish, so this typically reads ~0.
    if (renderThreadJobKicked) {
        MGE::RenderThread::wait();
    }

    // Menu-cache latch maintenance (USE_MENU_CACHING). MUST run on EVERY frame, so it sits ABOVE
    // the early return below — this is the latch's only RELEASE. postProcess() sets isRenderCached
    // on the first menu frame and thereafter blits the cached frame instead of rendering; nothing
    // else lowers it except the mouse-click expiry in that same blit path. Leaving it below the
    // early return meant the world stayed frozen after the menu closed until you clicked (in-game,
    // 2026-07-22) — the UI/minimap kept updating over the stale blit, which reads exactly like a
    // lost-focus freeze. That was invisible before S2 only because menu frames were excluded from
    // earlyForgeKickoff and so always reached the full path.
    isRenderCached &= (Configuration.MGEFlags & USE_MENU_CACHING) && mwBridge->IsMenu();

    // Phase 2 (MW-only pipeline): Forge renders ALL 3D. On steady-state Forge frames
    // (earlyForgeKickoff — the geometry-cache walk + frustum-visible set were already
    // built in frameSetupEarly, the distant-statics cull RPC is gated off there, and
    // every DX9 pass below would be overwritten by the present composite), skip the
    // entire DX9 scene layer. Preserve only the per-frame state this stage OWNS: the
    // record-list drain (recordMW is still captured in inspectIndexedPrimitive but has
    // no Forge consumer, so they grow across frames if not cleared) and the menu-cache latch above.
    // The render-thread fence above still ran and the Tracy sky zone was closed at the top.
    // Warm-up / F11-off frames (!earlyForgeKickoff) fall through to the full path
    // — that's where the cache-walk fallback and the statics-RPC drain still live.
    if (earlyForgeKickoff) {
        recordMW.clear();
        return;
    }

    // (Channel-free gate moved into renderDepth, right after renderDepthFromCache:
    // that cache-only depth work touches no ipcClient and can overlap the worker's
    // statics-RPC drain, so we let it run before blocking on the channel. Nothing
    // in renderStage0 before renderDepth touches ipcClient on the worker path —
    // selectDistantCell and the fallback kickoff are both skipped there.)

    // (Scene-graph lights snapshot — MGE::SceneGraph::onFrameReady() — moved to
    // frameSetupEarly() at BeginScene(0) so the async worker overlaps the sky
    // pass. frameSetupEarly runs once per frame before renderStage0, so the
    // once-per-frame guarantee is preserved.)

    // Update current cell and select distant static set. Skipped if
    // frameSetupEarly() already ran it this frame at BeginScene — re-running
    // would re-issue the blocking setWorldSpace RPC for nothing.
    if (!s_frameSetupEarly) {
        selectDistantCell();
    }

    // Get Morrowind camera matrices
    device->GetTransform(D3DTS_VIEW, &mwView);
    device->GetTransform(D3DTS_PROJECTION, &mwProj);

    // Set variables derived from current game state and camera configuration.
    // setView/adjustFog re-run harmlessly even when frameSetupEarly already did
    // (pure idempotent computation; camera/state are unchanged within the frame).
    setView(&mwView);
    adjustFog();
    setupCommonEffect(&mwView, &mwProj);
    FixedFunctionShader::updateLighting(lightSunMult, lightAmbMult);

    // (isRenderCached maintenance hoisted above the earlyForgeKickoff early return — see there.)
    isPPLActive = (Configuration.MGEFlags & USE_FFESHADER) && !(Configuration.PerPixelLightFlags == 1 && !mwBridge->IntCurCellAddr());

    // S4: the NUMPAD7 opaque-source cycle (ENGINE / CACHE / CACHE-ONLY) went with
    // rendercachedcolor.cpp. CACHE mode was the pre-DX12 opaque-cache attempt; its DX9
    // mirror VB stopped being built long ago (needMirror() excluded it), so it had been
    // drawing nothing for some time. The Forge takeover is what CACHE mode was reaching for.

    // VK_DECIMAL: A/B toggle tiled vs per-mesh point lighting (numpad digits are all
    // taken — Numpad4=heatmap, Numpad7=cache, Numpad1=water proxy). Master config flag
    // (UseTiledLights) still gates: when off, this is inert. Polled once per frame.
    if ((GetAsyncKeyState(VK_DECIMAL) & 0x0001) && Configuration.UseTiledLights) {
        FixedFunctionShader::tiledLightsActive = !FixedFunctionShader::tiledLightsActive;
        StatusOverlay::setStatus(FixedFunctionShader::tiledLightsActive
            ? "Point lights: TILED (screen grid)" : "Point lights: PER-MESH (selection)");
    }

    if (!isRenderCached) {
        // Save state block manually since we can change FVF/decl, and because the depth
        // pass runs with D3DXFX_DONOTSAVESTATE.
        //
        // S4b: this used to fork on isDistantCell() — the exterior branch ran the statics
        // cull, the distant land/statics colour passes and the MSOC overlays around the
        // depth call, the interior branch just ran depth. With all of that deleted the two
        // branches were character-for-character identical, so there is one path now.
        // renderDepth still self-gates its own distant parts on isDistantCell().
        device->CreateStateBlock(D3DSBT_ALL, &stateSaved);
        effect->BeginPass(PASS_SETUP);
        effect->EndPass();

        effectDepth->Begin(&passes, D3DXFX_DONOTSAVESTATE);
        renderDepth();
        effectDepth->End();

        // Restore render state
        stateSaved->Apply();
        stateSaved->Release();
    }

    // Clear stray recordings
    recordMW.clear();
}

// renderStage1 - EndScene(scene 0). Closes the MW-draw timing window and drains the
// per-scene record list.
//
// S4: the grass colour pass (the last thing this stage drew) is gone with rendergrass.cpp.
// What remains is bookkeeping the stage owns and nothing else does: the MW-draw accumulator,
// the Tracy zone close, and the recordMW drain — without which scene 0's records would leak
// into renderStage2's depth replay for scenes 1+.
void DistantLand::renderStage1() {
    if (s_mwDrawsT0 != 0.0) {
        mwDrawsAccum(fseNowMs() - s_mwDrawsT0);
        s_mwDrawsT0 = 0.0;
    }
#ifdef TRACY_ENABLE
    delete s_mwDrawsZone;
    s_mwDrawsZone = nullptr;
#endif
    MGE_ZoneScopedN("Stage1");
    MGE_TracyPlot("MW draw calls", (int64_t)recordMW.size());

    recordMW.clear();
}

// renderStage2 - Render shadows and depth texture for scenes 1+ (post-stencil redraw/alpha/1st person)
void DistantLand::renderStage2() {
    MGE_ZoneScopedN("Stage2");
    auto mwBridge = MWBridge::get();
    IDirect3DStateBlock9* stateSaved;
    UINT passes;

    ///LOG::logline("Stage 2 prims: %d", recordMW.size());

    // Early out if nothing is happening
    if (recordMW.empty()) {
        return;
    }

    // Skip in Forge mode too: this entire block is the recorded-render shadow receiver +
    // recorded depth replay (renderDepthAdditional), both superseded — depth now comes from
    // the cache (renderCacheDepthToMainZ + the depth-texture cache pass) and shadows are
    // deferred host-side. The recordMW capture and the clear below are untouched.
    if (!isRenderCached && !RenderProcess::forgeOwnsFrame()) {
        // Save state block manually since we can change FVF/decl
        device->CreateStateBlock(D3DSBT_ALL, &stateSaved);

        // S3: shadowing onto recorded renders was here (same receiver as renderStage1).

        // Depth texture from recorded renders
        effectDepth->Begin(&passes, D3DXFX_DONOTSAVESTATE);
        renderDepthAdditional();
        effectDepth->End();

        // Restore state
        stateSaved->Apply();
        stateSaved->Release();
    }

    recordMW.clear();
}


// S4: renderStageBlend is gone. It fed the MW<->MGE handover feather (PASS_BLENDMGE over a
// distant-only capture) and, before S3, the water caustics. Both of its inputs died with the
// DX9 distant land: there is no distant-only frame to feather any more, and the Forge
// composite owns the handover band outright.

// setupCommonEffect - Set shared shader variables for this frame
void DistantLand::setupCommonEffect(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    auto mwBridge = MWBridge::get();

    // View position
    effect->SetMatrix(ehView, view);
    effect->SetMatrix(ehProj, proj);
    effect->SetFloatArray(ehEyePos, eyePos, 3);

    // Sunlight
    D3DXVECTOR3 sunVecView;
    RGBVECTOR totalAmb = sunAmb + ambCol;
    D3DXVec3TransformNormal(&sunVecView, (const D3DXVECTOR3*)&sunVec, view);

    effect->SetFloatArray(ehSunVec, sunVec, 3);
    effect->SetFloatArray(ehSunVecView, sunVecView, 3);
    effect->SetFloatArray(ehSunCol, sunCol, 3);
    effect->SetFloatArray(ehSunAmb, totalAmb, 3);
    effect->SetFloatArray(ehSunPos, sunPos, 3);
    effect->SetFloat(ehSunVis, sunVis);

    if (isPPLActive) {
        // Apply light multiplier settings to distant land
        RGBVECTOR s = lightSunMult * sunCol, a = lightAmbMult * totalAmb;
        effect->SetFloatArray(ehSunCol, s, 3);
        effect->SetFloatArray(ehSunAmb, a, 3);
    }

    // Sky/fog
    bool isExpFog = (Configuration.MGEFlags & EXP_FOG) != 0;
    const RGBVECTOR* skyCol = mwBridge->CellHasWeather() ?  mwBridge->getCurrentWeatherSkyCol() : &horizonCol;
    effect->SetFloat(ehFogStart, isExpFog ? fogExpStart : fogStart);
    effect->SetFloat(ehFogRange, isExpFog ? fogExpDivisor : fogEnd);
    effect->SetFloat(ehFogNearStart, fogNearStart);
    effect->SetFloat(ehFogNearRange, fogNearEnd);
    effect->SetFloatArray(ehSkyCol, *skyCol, 3);
    effect->SetFloatArray(ehFogColNear, nearFogCol, 3);
    effect->SetFloatArray(ehFogColFar, horizonCol, 3);
    effect->SetFloat(ehNearViewRange, nearViewRange);
    effect->SetFloat(ehNiceWeather, niceWeather);

    if (ehOutscatter) {
        effect->SetFloatArray(ehOutscatter, atmOutscatter, 3);
        effect->SetFloatArray(ehInscatter, atmInscatter, 3);
        effect->SetFloatArray(ehSkyScatterFar, atmSkylightScatter, 4);
    }

    // Wind, requires smoothing as it is very noisy
    static float smoothWind[2];
    if (!mwBridge->IsMenu()) {
        const float f = 0.02;
        const float* wind = mwBridge->GetWindVector();
        smoothWind[0] += f * (windScaling * wind[0] - smoothWind[0]);
        smoothWind[1] += f * (windScaling * wind[1] - smoothWind[1]);
        effect->SetFloatArray(ehWindVec, smoothWind, 2);
    }

    // Other
    effect->SetFloatArray(ehFootPos, (float*)mwBridge->PlayerPositionPointer(), 3);
    effect->SetFloat(ehTime, mwBridge->simulationTime());

}

// setScattering - Set scattering coefficients for atmospheric scattering shader
void DistantLand::setScattering(const RGBVECTOR& out, const RGBVECTOR& in) {
    atmOutscatter = out;
    atmInscatter = in;
}

static double lerp(double x0, double x1, double t) {
    return (1.0 - t) * x0 + t * x1;
}

static double saturate(double x) {
    return std::min(std::max(0.0, x), 1.0);
}

// adjustFog - Set fog distance, wind speed adjust and fog colour for this frame
void DistantLand::adjustFog() {
    auto mwBridge = MWBridge::get();

    nearViewRange = mwBridge->GetViewDistance();

    // Morrowind does not update weather during menu mode, except when time is changed
    // Therefore always run adjustment during menu mode, except if background caching is used
    if (isRenderCached) {
        return;
    }

    // Forge-owned horizon: the host renders DL.DrawDist cells of world into the composite
    // regardless of MGE's own DL flag, so with USE_DISTANT_LAND off MW's vanilla short fog
    // would wall off a world that IS there. Take the DL fog path (weather-scaled, cell
    // units) whenever the seam owns the frame in a weather cell — this is the cut that
    // lets MGE's DL machinery turn off entirely while the horizon stays at host distance.
    // Weather-gated so interiors keep vanilla fog exactly like the DL-on non-distant case;
    // F11-off restores vanilla fog via the else-branch's nearViewRange re-sync (clean A/B).
    const bool forgeFog = RenderProcess::forgeOwnsFrame() && mwBridge->CellHasWeather();

    // Get fog cell ranges based on environment and weather
    if (mwBridge->IsUnderwater(eyePos.z)) {
        fogStart = Configuration.DL.BelowWaterFogStart;
        fogEnd = Configuration.DL.BelowWaterFogEnd;
    } else if (mwBridge->CellHasWeather()) {
        int wthr1 = mwBridge->GetCurrentWeather(), wthr2 = mwBridge->GetNextWeather();
        float ratio = mwBridge->GetWeatherRatio(), ff = 1.0, fo = 0.0, ws = 0.0;

        if (ratio != 0 && wthr2 >= 0 && wthr2 <= 9) {
            ff = float(lerp(Configuration.DL.FogD[wthr1], Configuration.DL.FogD[wthr2], ratio));
            fo = float(0.01 * lerp(Configuration.DL.FgOD[wthr1], Configuration.DL.FgOD[wthr2], ratio));
            ws = float(lerp(Configuration.DL.Wind[wthr1], Configuration.DL.Wind[wthr2], ratio));
            niceWeather = float(lerp((wthr1 <= 1) ? 1.0 : 0.0, (wthr2 <= 1) ? 1.0 : 0.0, ratio));
            niceWeather *= niceWeather;
            lightSunMult = float(lerp(Configuration.Lighting.SunMult[wthr1], Configuration.Lighting.SunMult[wthr2], ratio));
            lightAmbMult = float(lerp(Configuration.Lighting.AmbMult[wthr1], Configuration.Lighting.AmbMult[wthr2], ratio));
        } else if (wthr1 >= 0 && wthr1 <= 9) {
            ff = Configuration.DL.FogD[wthr1];
            fo = Configuration.DL.FgOD[wthr1] / 100.0f;
            ws = Configuration.DL.Wind[wthr1];
            niceWeather = (wthr1 <= 1) ? 1.0f : 0.0f;
            lightSunMult = Configuration.Lighting.SunMult[wthr1];
            lightAmbMult = Configuration.Lighting.AmbMult[wthr1];
        }

        // Fog distance scale calculation, ensure fogEnd does not scale closer than vanilla Morrowind
        fogEnd = std::max(0.875f, ff * Configuration.DL.AboveWaterFogEnd);
        fogStart = ff * Configuration.DL.AboveWaterFogStart - fo * fogEnd;
        windScaling = ws;

        // For exp fog, adjust start distance so that starting fog approximately equals fo, to retain near visibility comparable to vanilla
        if (((Configuration.MGEFlags & USE_DISTANT_LAND) || forgeFog) && (Configuration.MGEFlags & EXP_FOG)) {
            float lg = log(1.0f - 0.25f * fo);
            float expCorrection = lg / (1 + lg);
            fogStart = ff * Configuration.DL.AboveWaterFogStart + expCorrection * fogEnd;
        }
    } else {
        // Avoid density == 0, as when fogStart and fogEnd are equal, the fog equation denominator goes to infinity
        float density = std::max(0.01f, mwBridge->getInteriorFogDens());
        fogStart = float(lerp(Configuration.DL.InteriorFogEnd, Configuration.DL.InteriorFogStart, density));
        fogEnd = Configuration.DL.InteriorFogEnd;
        niceWeather = 0;
        windScaling = 0;
        lightSunMult = 1.0;
        lightAmbMult = 1.0;
    }

    // Convert from cells to in-game units
    fogStart *= kCellSize;
    fogEnd *= kCellSize;

    if (((Configuration.MGEFlags & USE_DISTANT_LAND) && isDistantCell()) || forgeFog) {
        // Set hardware fog for Morrowind's use
        if (Configuration.MGEFlags & EXP_FOG) {
            // Exponential fog mode
            // Adjust exp curve so that at the fog end boundary, the same fog value is reached for all values of fogStart
            constexpr float expFogDistScale = 4.4f;
            fogExpStart = fogStart / expFogDistScale;
            fogExpDivisor = (fogEnd - fogExpStart) / expFogDistScale;

            if (mwBridge->IsUnderwater(eyePos.z) || !mwBridge->CellHasWeather()) {
                // Leave fog ranges as set, shaders use all linear fogging in this case
                fogNearStart = fogStart;
                fogNearEnd = fogEnd;
            } else {
                // Adjust near region linear Morrowind fogging to approximation of exp fog curve
                // Linear density matched to exp fog at dist = 1280 and dist = viewrange (or fog end if closer)
                // Note to self: Don't use saturate here or the denominators can become zero.
                float farIntercept = std::min(fogEnd, nearViewRange);
                float expFogNear = exp(-(1280.0f - fogExpStart) / fogExpDivisor);
                float expFogFar = exp(-(farIntercept - fogExpStart) / fogExpDivisor);
                fogNearStart = 1280.0f + (farIntercept - 1280.0f) * (1.0f - expFogNear) / (expFogFar - expFogNear);
                fogNearEnd = 1280.0f + (farIntercept - 1280.0f) * -expFogNear / (expFogFar - expFogNear);
            }
        } else {
            // Linear mode
            fogNearStart = fogStart;
            fogNearEnd = fogEnd;
        }

        device->SetRenderState(D3DRS_FOGSTART, *(DWORD*)&fogNearStart);
        device->SetRenderState(D3DRS_FOGEND, *(DWORD*)&fogNearEnd);
    } else {
        // Update fog when near render distance changes, and on startup when fogNearEnd == 0
        bool doFogUpdate = fogNearEnd != nearViewRange;

        // Read Morrowind-set fog range
        fogNearEnd = nearViewRange;
        fogNearStart = fogNearEnd * std::min(1.0f - mwBridge->getScenegraphFogDensity(), 0.99f);
        fogStart = fogNearStart;
        fogEnd = fogNearEnd;

        if (doFogUpdate) {
            device->SetRenderState(D3DRS_FOGSTART, *(DWORD*)&fogNearStart);
            device->SetRenderState(D3DRS_FOGEND, *(DWORD*)&fogNearEnd);
        }
    }

    // Adjust Morrowind fog colour towards scatter colour if necessary
    if ((Configuration.MGEFlags & USE_DISTANT_LAND) && (Configuration.MGEFlags & USE_ATM_SCATTER) && mwBridge->CellHasWeather() && !mwBridge->IsUnderwater(eyePos.z)) {
        // Read unadjusted colour, as the scenegraph fog colour may not be updated during menu transitions
        RGBVECTOR c0 = *mwBridge->getCurrentWeatherFogCol();
        RGBVECTOR c1 = c0;

        // Simplified version of scattering from the shader
        const RGBVECTOR* skyCol = mwBridge->getCurrentWeatherSkyCol();
        const D3DXVECTOR3 newSkyCol = {
            float(lerp(skyCol->r, atmSkylightScatter.x, atmSkylightScatter.w)),
            float(lerp(skyCol->g, atmSkylightScatter.y, atmSkylightScatter.w)),
            float(lerp(skyCol->b, atmSkylightScatter.z, atmSkylightScatter.w))
        };
        const float sunaltitude = powf(1 + sunPos.z, 10);
        const float sunaltitude_a = 2.8 + 4.3 / sunaltitude;
        const float sunaltitude_b = saturate(1.0 - exp2(-1.9 * sunaltitude));
        const float sunaltitude_c = saturate(exp(-4.0 * sunPos.z)) * saturate(sunaltitude);

        // Calculate scatter colour at Morrowind draw distance boundary
        float fogdist = (nearViewRange - fogExpStart) / fogExpDivisor;
        float fog = saturate(exp(-fogdist));
        fogdist = saturate(0.224 * fogdist);

        D3DXVECTOR2 horizonDir(eyeVec.x, eyeVec.y);
        D3DXVec2Normalize(&horizonDir, &horizonDir);
        float suncos =  horizonDir.x * sunPos.x + horizonDir.y * sunPos.y;
        float mie = (1.58 / (1.24 - suncos)) * sunaltitude_c;
        float rayl = 1.0 - 0.09 * mie;
        float atmdep = 1.33;

        D3DXVECTOR3 scatter;
        float scatterT = 0.5 * (1 + suncos);
        scatter.x = float(lerp(atmInscatter.r, atmOutscatter.r, scatterT));
        scatter.y = float(lerp(atmInscatter.g, atmOutscatter.g, scatterT));
        scatter.z = float(lerp(atmInscatter.b, atmOutscatter.b, scatterT));

        D3DXVECTOR3 att = atmdep * scatter * (sunaltitude_a + 0.7 * mie);
        att.x = (1 - exp(-fogdist * att.x)) / att.x;
        att.y = (1 - exp(-fogdist * att.y)) / att.y;
        att.z = (1 - exp(-fogdist * att.z)) / att.z;

        D3DXVECTOR3 k0 = mie * D3DXVECTOR3(0.125, 0.125, 0.125) + rayl * newSkyCol;
        D3DXVECTOR3 k1 = att * (1.17 * atmdep + 0.89) * sunaltitude_b;
        c1.r = k0.x * k1.x;
        c1.g = k0.y * k1.y;
        c1.b = k0.z * k1.z;

        // Convert from additive inscatter to Direct3D fog model
        // The correction factor is clamped to avoid creating infinities
        c1 /= std::max(0.02f, 1.0f - fog);

        // Scattering fog only occurs in nice weather
        c0 = (1.0f - niceWeather) * c0 + niceWeather * c1;

        // Save colour for matching near fog in shaders
        nearFogCol = c0;

        // Alter Morrowind's fog colour through its scenegraph
        // This way it automatically restores the correct colour if it has to switch fog modes mid-frame
        DWORD fc = (DWORD)nearFogCol;
        mwBridge->setScenegraphFogCol(fc);

        // Set device fog colour to propagate change immediately
        device->SetRenderState(D3DRS_FOGCOLOR, fc);
    } else {
        // Save current fog colour for matching near fog in shaders
        nearFogCol = RGBVECTOR(mwBridge->getScenegraphFogCol());
    }
}

// postProcess - Calls post process module, or captures and applies frame cache to avoid rendering
void DistantLand::postProcess() {
    if (!isRenderCached) {
        auto mwBridge = MWBridge::get();

        // Save state block
        IDirect3DStateBlock9* stateSaved;
        device->CreateStateBlock(D3DSBT_ALL, &stateSaved);

        // Forge owns the composited frame (F11): the host already shades + applies its own GTAO,
        // and the MGE post chain's depth effects (SSAO/DOF) read texDepthFrame — the last consumer
        // of the MGE depth pre-pass. Skip the whole HW post chain when Forge is on so depth can be
        // retired. NOTE: this also drops any depth-free MGE post (bloom/colour/underwater); F11-off
        // restores it (clean A/B). Screenshot + menu-cache + foam-debug below still run.
        if ((Configuration.MGEFlags & USE_HW_SHADER) && !RenderProcess::forgeOwnsFrame()) {
            // Set flags to reflect cell environment
            int envFlags = 0;

            if (!mwBridge->CellHasWeather()) {
                envFlags |= 1;
            }
            if (mwBridge->IsExterior()) {
                envFlags |= 2;
            }
            if (mwBridge->IntLikeExterior()) {
                envFlags |= 4;
            }
            if (mwBridge->IsUnderwater(eyePos.z)) {
                envFlags |= 8;
            } else {
                envFlags |= 16;
            }
            if (sunVis >= 0.001) {
                envFlags |= 32;
            } else {
                envFlags |= 64;
            }

            // Run all shaders (with callback to set changed vars)
            PostShaders::shaderTime(&updatePostShader, envFlags, mwBridge->frameTime());
        }

        // Capture pre-UI screenshots here
        checkCaptureScreenshot(false);

        // Cache render for first frame of menu mode.
        //
        // DX9 path only. Under Forge the same flag drives DistantLand::menuFreeze instead, which
        // pauses the world at the source (no host frame is rendered at all) rather than hiding a
        // fully-rendered one behind a stale blit — this capture saved nothing there and cost a
        // borrowBuffer + a full-screen StretchRect over the composite every menu frame. Not
        // latching under Forge also removes the menu consumer of PostShaders::borrowBuffer, which
        // S6 has to retire regardless.
        if ((Configuration.MGEFlags & USE_MENU_CACHING) && mwBridge->IsMenu()
                && !RenderProcess::forgeOwnsFrame()) {
            texMenuCache = PostShaders::borrowBuffer(0);
            isRenderCached = true;
        }

        // Shadow map inset
        ///if(!mwBridge->IsMenu()) { renderShadowDebug(); }

        // Restore state
        stateSaved->Apply();
        stateSaved->Release();
    } else {
        // Blit cached frame to screen
        IDirect3DSurface9* backbuffer, *surfDistant;
        device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backbuffer);
        texMenuCache->GetSurfaceLevel(0, &surfDistant);
        device->StretchRect(surfDistant, 0, backbuffer, 0, D3DTEXF_NONE);
        surfDistant->Release();
        backbuffer->Release();

        // Cache expires for frame after mouse click, so as not to affect click response time
        isRenderCached &= !MGEProxyDirectInput::mouseClick;
    }
}

// updatePostShader - callback for setting post shader variables based on environment
void DistantLand::updatePostShader(MGEShader* shader) {
    auto mwBridge = MWBridge::get();

    // Internal textures
    // TODO: Should be set once at init time
    shader->SetTexture(EV_depthframe, texDepthFrame);
    // S3: EV_watertexture fed MGE's animated water volume map, deleted with the water
    // renderer. Post shaders that declare it now see whatever they default to; the
    // input is retired for real when post-processing moves to the host (S7).

    // View position
    float zoom = (Configuration.MGEFlags & ZOOM_ASPECT) ? Configuration.CameraEffects.zoom : 1.0f;
    shader->SetMatrix(EV_mview, &mwView);
    shader->SetMatrix(EV_mproj, &mwProj);
    shader->SetFloatArray(EV_eyevec, eyeVec, 3);
    shader->SetFloatArray(EV_eyepos, eyePos, 3);
    shader->SetFloat(EV_fov, Configuration.ScreenFOV / zoom);

    // Lighting
    RGBVECTOR totalAmb = sunAmb + ambCol;
    shader->SetFloatArray(EV_sunvec, sunVec, 3);
    shader->SetFloatArray(EV_suncol, sunCol, 3);
    shader->SetFloatArray(EV_sunamb, totalAmb, 3);
    shader->SetFloatArray(EV_sunpos, sunPos, 3);
    shader->SetFloat(EV_sunvis, float(lerp(sunVis, 1.0, 0.333 * niceWeather)));

    // Sky/fog
    bool isExpFog = (Configuration.MGEFlags & EXP_FOG) != 0;
    shader->SetFloatArray(EV_fogcol, horizonCol, 3);
    shader->SetFloatArray(EV_fognearcol, nearFogCol, 3);
    shader->SetFloat(EV_fogstart, isExpFog ? fogExpStart : fogStart);
    shader->SetFloat(EV_fogrange, isExpFog ? fogExpDivisor : fogEnd);
    shader->SetFloat(EV_fognearstart, fogNearStart);
    shader->SetFloat(EV_fognearrange, fogNearEnd);

    // Other
    // In cells without water, set very low waterlevel for shaders that clip against water
    float water = mwBridge->CellHasWater() ? mwBridge->WaterLevel() : -1e9f;
    shader->SetFloat(EV_time, mwBridge->simulationTime());
    shader->SetFloat(EV_waterlevel, water);
    shader->SetBool(EV_isinterior, !mwBridge->CellHasWeather());
    shader->SetBool(EV_isunderwater, mwBridge->IsUnderwater(eyePos.z));
}

//------------------------------------------------------------

// selectDistantCell - Select the correct set of distant land meshes for the current cell
bool DistantLand::selectDistantCell() {
    auto mwBridge = MWBridge::get();

    if (Configuration.MGEFlags & USE_DISTANT_LAND) {
        // Scan dynamic vis on cell change
        void* playerCell = mwBridge->getPlayerCell();
        if (playerCell != lastDistantVisCell) {
            scanDynamicVisGroups();
            lastDistantVisCell = playerCell;
        }

        // Get worldspace key
        string cellname;
        if (mwBridge->IsExterior()) {
            cellname = string();
        }
        else {
            cellname = mwBridge->getInteriorName();
        }

        if (Configuration.UseSharedMemory) {
            DistantLandShare::hasCurrentWorldSpace = ipcClient.setWorldSpaceBlocking(cellname);
            if (DistantLandShare::hasCurrentWorldSpace) {
                return true;
            }
        } else {
            const auto iWS = DistantLandShare::mapWorldSpaces.find(cellname);
            if (iWS != DistantLandShare::mapWorldSpaces.end()) {
                DistantLandShare::currentWorldSpace = &iWS->second;
                DistantLandShare::hasCurrentWorldSpace = true;
                return true;
            }
        }
    }

    DistantLandShare::currentWorldSpace = nullptr;
    DistantLandShare::hasCurrentWorldSpace = false;
    return false;
}

// isDistantCell - Check if there is distant land selected for this cell
bool DistantLand::isDistantCell() {
    return DistantLandShare::hasCurrentWorldSpace;
}

// resolveDynamicVisGroups - Resolve pointers to game objects on load/reload
void DistantLand::resolveDynamicVisGroups() {
    auto mwBridge = MWBridge::get();
    const DynamicVisGroup *lastDVG = nullptr;

    for (auto& vis : dynamicVisGroups) {
        // Clear previous pointer
        vis.gameObject = nullptr;

        // Re-use previous result if the id matches
        if (lastDVG && vis.id == lastDVG->id) {
            vis.gameObject = lastDVG->gameObject;
            continue;
        }
        else {
            lastDVG = &vis;
        }

        // Resolve IDs to pointers
        switch (vis.source) {
        case DynamicVisGroup::DataSource::Journal:
            vis.gameObject = mwBridge->getDialogue(vis.id.c_str());
            break;
        case DynamicVisGroup::DataSource::Global:
            vis.gameObject = mwBridge->getGlobalVar(vis.id.c_str());
            break;
        case DynamicVisGroup::DataSource::UniqueObject:
            vis.gameObject = mwBridge->findFirstReferenceById(vis.id.c_str());
            break;
        }
    }

    // Ensure reloading into the same cell still triggers updates
    lastDistantVisCell = nullptr;
}

// scanDynamicVisGroups - Scan through game data for visibility changes
void DistantLand::scanDynamicVisGroups() {
    auto mwBridge = MWBridge::get();
    if (Configuration.UseSharedMemory) {
        dynVisFlagsShared.clear();
    }

    std::uint16_t i = 0;
    for (auto& vis : dynamicVisGroups) {
        int value;
        auto groupIndex = i++;

        // Ignore unresolved objects
        if (!vis.gameObject) {
            continue;
        }

        switch (vis.source) {
        case DynamicVisGroup::DataSource::Journal:
            value = mwBridge->getJournalIndex(vis.gameObject);
            break;
        case DynamicVisGroup::DataSource::Global:
            value = int(mwBridge->getGlobalVarValue(vis.gameObject));
            break;
        case DynamicVisGroup::DataSource::UniqueObject:
            const int disabledRecordFlag = 0x800;
            value = (mwBridge->getRecordFlags(vis.gameObject) & disabledRecordFlag) == 0;
            break;
        }

        // Enable if value is inside any range
        bool enable = false;
        for (const auto& r : vis.ranges) {
            if (r.begin <= value && value < r.end) {
                enable = true;
                break;
            }
        }

        // If enable state has changed, propagate to distant land mesh instances
        if (enable ^ vis.enabled) {
            vis.enabled = enable;
            if (Configuration.UseSharedMemory) {
                dynVisFlagsShared.push_back({ groupIndex, enable });
            } else {
                for (auto& m : vis.references) {
                    m->enabled = enable;
                }
            }
        }
    }

    if (Configuration.UseSharedMemory && !dynVisFlagsShared.empty()) {
        ipcClient.updateDynVis(dynVisFlagsSharedId);
    }
}

// setView - Called once per frame to setup view dependent data
void DistantLand::setView(const D3DMATRIX* m) {
    auto mwBridge = MWBridge::get();

    // Calculate eyePos, eyeVec for shaders
    D3DXVECTOR4 origin(0.0, 0.0, 0.0, 1.0);
    D3DXMATRIX invView, view = *m;

    D3DXMatrixInverse(&invView, 0, &view);
    D3DXVec4Transform(&eyePos, &origin, &invView);
    eyeVec.x = m->_13;
    eyeVec.y = m->_23;
    eyeVec.z = m->_33;

    // Set sun disc position
    if (mwBridge->IsLoaded() && mwBridge->CellHasWeather()) {
        mwBridge->GetSunDir(sunPos.x, sunPos.y, sunPos.z);
        sunPos.w = 1;
        sunPos /= sqrt(sunPos.x * sunPos.x + sunPos.y * sunPos.y + sunPos.z * sunPos.z);

        // Sun position "bounces" at the horizon to follow night lighting instead of setting
        // Sun visibility goes to zero at night, so use this to correct the sun position so it sets
        sunVis = mwBridge->GetSunVis() / 255.0f;
        if (sunVis == 0) {
            sunPos.z = -sunPos.z;
        }
    } else {
        sunPos = D3DXVECTOR4(0, 0, -1, 1);
        sunVis = 0;
    }
}

// setProjection - Called when a D3D projection matrix is set, and edits it
void DistantLand::setProjection(D3DMATRIX* proj) {
    // Move near plane from 1.0 to 4.0 for more z accuracy
    // Move far plane back to edge of draw distance
    if (Configuration.MGEFlags & USE_DISTANT_LAND) {
        editProjectionZ(proj, kDistantNearPlane, Configuration.DL.DrawDist * kCellSize);
    }
}

// editProjectionZ - Alter the near and far clip planes of a projection matrix
void DistantLand::editProjectionZ(D3DMATRIX* m, float zn, float zf) {
    // Override near and far clip planes
    m->_33 = zf / (zf - zn);
    m->_43 = -zn * zf / (zf - zn);
}

void DistantLand::setHorizonColour(const RGBVECTOR& c) {
    horizonCol = c;
}

void DistantLand::setAmbientColour(const RGBVECTOR& c) {
    ambCol = c;
}

void DistantLand::setSunLight(const D3DLIGHT8* s) {
    // Sun is used for both interiors and exteriors; the sun in interiors is a fixed light
    sunVec.x = s->Direction.x;
    sunVec.y = s->Direction.y;
    sunVec.z = s->Direction.z;
    D3DXVec3Normalize((D3DXVECTOR3*)&sunVec, (D3DXVECTOR3*)&sunVec);

    sunCol = s->Diffuse;
    sunAmb = s->Ambient;
}

// isCoveredOpaque - Phase 1 Milestone 1 predicate for the engine draws that
// renderCachedOpaque takes over: any opaque (zWrite, non-blend) draw that samples a
// base texture. CACHE mode owns ALL such geometry (drawn base-texture-only from the
// cache), so the engine draw is suppressed here.
//
// Earlier this was narrower (single-stage, UV0 only), leaving multi-stage parts to
// the engine. But renderCachedOpaque draws every non-blend textured cache part
// base-only regardless of the original stage count, so a multi-stage part got drawn
// TWICE (cache base-only + engine full). For statics both land at the same depth
// (harmless), but ANIMATED parts differ by a sub-frame between the cache snapshot
// transform and the engine's live transform — the two draws z-fight. Suppressing
// the whole opaque-textured set makes every covered part single-draw (cache only),
// killing the z-fight. The tradeoff is that multi-texture detail (dark/detail/glow)
// renders base-only in CACHE mode — the documented Phase-2 coverage gap, now uniform
// instead of double-drawn. Untextured / alpha-blended / terrain-splat stay engine.
static bool isCoveredOpaque(const RenderedState* rs, const FragmentState* frs) {
    if (!rs->zWrite || rs->blendEnable) return false;   // opaque only
    const auto& s0 = frs->stage[0];
    return s0.colorArg1 == D3DTA_TEXTURE || s0.colorArg2 == D3DTA_TEXTURE;  // has a base texture
}

// inspectIndexedPrimitive
// Filters and records DIP calls for later use; returning false should cause the draw call to be skipped
// Can also replace selected fixed function calls with an augmented shader
bool DistantLand::inspectIndexedPrimitive(int sceneCount, const RenderedState* rs, const FragmentState* frs, LightState* lightrs) {
    auto mwBridge = MWBridge::get();

    // Avoid recording landscape alpha blend drawcalls, a form of multi-pass splatting
    static IDirect3DVertexBuffer9* lastVB = nullptr;
    bool isLandSplat = sceneCount == 0 && rs->vb == lastVB && rs->blendEnable && (rs->fvf & D3DFVF_DIFFUSE) && mwBridge->IsExterior();
    lastVB = rs->vb;

    // Avoid recording decal passes from UV sets >0, shadow rendering only samples alpha from texture 0 with UV 0
    const auto& stage0 = frs->stage[0];
    bool isDecal = stage0.texcoordIndex != 0 && (stage0.colorArg1 == D3DTA_TEXTURE || stage0.colorArg2 == D3DTA_TEXTURE);

    // Capture all writes to z-buffer, except detectable second passes of multi-pass rendering
    if (rs->zWrite && !isLandSplat && !isDecal) {
        recordMW.emplace_back(*rs);

        // Unify alpha test operator/reference to be equivalent to GREATEREQUAL
        if (rs->alphaFunc == D3DCMP_GREATER) {
            recordMW.back().alphaRef++;
        }
    }

    // AT1 sorted-alpha A/B + AT3 inventory: while the Forge alpha pass is live, MW's sorted-alpha
    // draw is redundant — the host already drew the blended set (depth-correct, behind Forge
    // walls). Reject BLENDED DIPs in any scene >= 1. NOT a bare sceneCount==1 gate: MW's scene
    // indices are CONDITIONAL (see EndScene) — with no sorted alpha visible, scene 1 IS the
    // 1st-person scene, and with stencil shadows the indices shift; a scene-index gate would eat
    // the player's hands. Blended-only matches exactly what the sorter draws (No-Sorter blends
    // went to scene 0; water never reaches here — the isWaterMaterial branch bypasses
    // inspection). Placed AFTER the recordMW capture above so depth-replay records are
    // untouched. Whatever still renders with this ON is the AT3 leftover set (particles/VFX —
    // not NiTriShapes, not captured by the cache walk).
    // AT2: the msoc plugin's kOwnedAlpha display skip (renderdepth.cpp setOwnedFlags) is now
    // the PRIMARY mechanism — most covered blended leaves never display, so their DIPs never
    // reach here. This gate stays as the belt: old msoc.dll (opaque-only), and blended leaves
    // the plugin conservatively keeps displaying (decal/multi-map/untextured) that our host
    // pass doesn't draw either.
    if (Configuration.ForgeAlphaPass && Configuration.ForgeAlphaSuppressS1
        && sceneCount >= 1 && rs->blendEnable && RenderProcess::forgeOwnsFrame()) {
        // AT3: before rejecting, capture MW's already-billboarded blended DIP (NiParticles smoke/
        // flames + multimap/decal/untextured blends the host cache pass doesn't own) so the Forge
        // host can draw it in the post-water sorted-alpha pass. Silent no-op when disabled/unsuited;
        // the reject below is unchanged whether or not the capture takes.
        RenderProcess::captureAlphaDraw(rs, frs);
        return false;
    }

    // Special case, detect sky. MW's sky is the first blended geometry of scene 0 in a
    // weather cell — nothing has written z yet, so recordMW is still empty.
    //
    // S4: MGE's atmosphere-scattering sky (renderSky + the recordSky replay list it drew
    // from) is gone; the Forge host owns the sky (SK3/SK4). So the suppression that used to
    // hang off USE_ATM_SCATTER — "reject MW's sky, MGE will draw it later in stage 0" — now
    // hangs off the host actually drawing one. With the seam off, MW draws its own vanilla
    // sky, which is the fallback this retirement is meant to leave intact.
    if (recordMW.empty() && rs->blendEnable && sceneCount == 0 && mwBridge->CellHasWeather()) {
        if (RenderProcess::forgeOwnsFrame()) {
            return false;
        }
    } else {
        // Suppress the engine's scene-0 opaque draw when the Forge seam owns the frame
        // (F11 composite live): the host's full-screen composite overwrites MW's frame, so
        // the engine's draw is redundant double work.
        //
        // sceneCount == 0 is REQUIRED. isCoveredOpaque matches textured opaque in ANY scene,
        // so without this gate the first-person arm gets suppressed but never host-drawn ->
        // hands vanish. isLandSplat covers the terrain splat overlay passes and already
        // self-gates to scene 0. The recordMW capture above is untouched — the depth replay
        // still sees the engine geometry.
        //
        // (S4: this gate used to read `cacheOpaqueMode || forgeOwnsFrame()`. The cache half
        // was the CACHE-mode opaque takeover, which the Forge host superseded.)
        if (RenderProcess::forgeOwnsFrame() && sceneCount == 0
            && (isCoveredOpaque(rs, frs) || isLandSplat)) {
            return false;
        }
        // PPL reactive colour path: render non-covered opaque (and, in plain PPL mode,
        // all opaque) with the replacement FFE shader. Only when the PPL renderer is
        // active; in fixed-function mode the engine draws its own colour.
        if (isPPLActive) {
            // Reactive main view → tiled point lighting (no-op fallback to per-mesh
            // when tiled is inactive; the LightMode param is a static call-site signal).
            FixedFunctionShader::renderMorrowind(rs, frs, lightrs, 1.0f,
                nullptr, 0, nullptr, nullptr, nullptr,
                FixedFunctionShader::LightMode::Tiled);
            return false;
        }
    }

    return true;
}

// requestCapture - Set a function to be called with a screen capture
// Either before the UI is drawn, or after UI and before MGE messages
void DistantLand::requestCapture(std::function<void(IDirect3DSurface9*)> handler, bool captureWithUI) {
    captureScreenHandler = handler;
    captureScreenWithUI = captureWithUI;
}

void DistantLand::checkCaptureScreenshot(bool isUIDrawn) {
    if (bool(captureScreenHandler) && captureScreenWithUI == isUIDrawn) {
        IDirect3DSurface9* surface = captureScreenshot();
        captureScreenHandler(surface);
        if (surface) {
            surface->Release();
        }
        captureScreenHandler = nullptr;
    }
}

// captureScreen - Capture a screenshot
IDirect3DSurface9* DistantLand::captureScreenshot() {
    IDirect3DTexture9* t;
    IDirect3DSurface9* s;

    // Resolve multisampled back buffer
    t = PostShaders::borrowBuffer(0);
    t->GetSurfaceLevel(0, &s);

    // Cancel render cache, borrowBuffer just overwrote it
    isRenderCached = false;

    // Copy buffer to system memory surface
    IDirect3DSurface9* surfSS;
    D3DSURFACE_DESC desc;

    s->GetDesc(&desc);
    DWORD hr = device->CreateOffscreenPlainSurface(desc.Width, desc.Height, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, &surfSS, NULL);
    if (hr != D3D_OK) {
        s->Release();
        return nullptr;
    }

    hr = device->GetRenderTargetData(s, surfSS);
    s->Release();
    if (hr != D3D_OK) {
        surfSS->Release();
        return nullptr;
    }

    return surfSS;
}


// ------------------------------------
// DistantLand::RecordedState

DistantLand::RecordedState::RecordedState(const RenderedState& state)
    : RenderedState(state) {
    vb->AddRef();
    ib->AddRef();
    if (texture) {
        texture->AddRef();
    }
}

DistantLand::RecordedState::~RecordedState() {
    if (vb) {
        vb->Release();
    }
    if (ib) {
        ib->Release();
    }
    if (texture) {
        texture->Release();
    }
}

DistantLand::RecordedState::RecordedState(RecordedState&& source) noexcept
    : RenderedState(source) {
    source.vb = nullptr;
    source.ib = nullptr;
    source.texture = nullptr;
}


// ------------------------------------
// RenderTargetSwitcher

// RenderTargetSwitcher - Switch to a render target, restoring state at end of scope
RenderTargetSwitcher::RenderTargetSwitcher(IDirect3DSurface9* target, IDirect3DSurface9* targetDepthStencil) {
    init(target, targetDepthStencil);
}

// RenderTargetSwitcher - Switch to a render surface belonging to a texture, restoring state at end of scope
RenderTargetSwitcher::RenderTargetSwitcher(IDirect3DTexture9* targetTex, IDirect3DSurface9* targetDepthStencil) {
    // Note the device still holds a reference to the target while it's active
    IDirect3DSurface9* target;
    targetTex->GetSurfaceLevel(0, &target);
    init(target, targetDepthStencil);
    target->Release();
}

void RenderTargetSwitcher::init(IDirect3DSurface9* target, IDirect3DSurface9* targetDepthStencil) {
    DistantLand::device->GetRenderTarget(0, &savedTarget);
    DistantLand::device->GetDepthStencilSurface(&savedDepthStencil);

    DistantLand::device->SetRenderTarget(0, target);
    DistantLand::device->SetDepthStencilSurface(targetDepthStencil);
}

RenderTargetSwitcher::~RenderTargetSwitcher() {
    DistantLand::device->SetRenderTarget(0, savedTarget);
    DistantLand::device->SetDepthStencilSurface(savedDepthStencil);

    if (savedTarget) {
        savedTarget->Release();
    }
    if (savedDepthStencil) {
        savedDepthStencil->Release();
    }
}
