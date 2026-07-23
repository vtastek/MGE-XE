#include "renderprocess.h"
#include "configuration.h"
#include "ipc/client.h"
#include "ipc/geomwire.h"
#include "support/log.h"
#include "dxvk_interop.h"
#include "distantland.h"
#include "mwbridge.h"
#include "scenegraph_geometry_cache.h"
#include "cachebounds.h"
#include "scenegraph.h"
#include "datahandler_view.h"
#include "morrowindbsa.h"
#include "mge_tracy.h"
#include "imgui.h"

#include <windows.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <thread>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
    // Seam ALLOCATION size. Set at bring-up to ceiling render-scale x backbuffer (the host's
    // Forge render target + the imported VkImage + g_mainTex are all created at THIS size).
    // Falls back to 640x360 if the backbuffer can't be queried.
    UINT g_w = 640;
    UINT g_h = 360;

    // Live render-scale (supersampling). The host renders into a g_rw x g_rh sub-rect of the
    // g_w x g_h allocation; the composite samples that sub-rect and stretches it to the g_bbW x
    // g_bbH backbuffer. Changing g_renderScale (panel slider) just re-derives g_rw/g_rh and
    // restamps the host render size — no reallocation, no host re-init.
    constexpr float kMaxRenderScale = 2.0f;   // allocation ceiling (SSAA up to 2x)
    constexpr float kMinRenderScale = 1.0f;   // supersampling-only (no downscale)
    float g_renderScale = 1.0f;               // live, panel-driven; [kMin..kMax]
    UINT  g_bbW = 640, g_bbH = 360;           // backbuffer (composite destination)
    UINT  g_rw = 640, g_rh = 360;             // current internal render size (<= g_w/g_h)

    IPC::Client* g_client = nullptr;
    bool   g_initOk  = false;
    bool   g_enabled = true;           // composite ON by default; F11 toggles it OFF/ON
    int    g_debugMode = 0;            // F12 diagnostic cycle: 0=normal, 1=depth (world-distance), 2=scatter, 3=AO, 4=bent normal
    unsigned g_frame = 0;

    // Dev overlay (Stage 2): F9 toggles the in-host Forge panel; mouse is polled each frame and
    // forwarded over the renderScene RPC. The host injects it into Forge UI (uiSetExternalInput).
    bool   g_devUiVisible = false;     // F9; default off so it never blocks normal play
    HWND   g_devHwnd = nullptr;        // MW focus window (cached from device creation params)
    bool   g_reloadShadersPending = false; // F8 latched at composite finish, consumed by the next kickoff
    bool   g_distLightsTogglePending = false; // numpad- latched at composite finish; one-shot host dist-light A/B
    bool   g_fpSuppressLive = false;   // FP1b: MW arm suppression; seeded from ForgeFPSuppress at init, numpad-/ flips live

    // Frame-ahead pipelining (ForgeFrameAhead): on early-kickoff frames the paired
    // renderSceneFinish + RT copy defer to the NEXT frame's BeginScene(0) collect, and the
    // EndScene(0) composite point blits the PREVIOUS host frame from g_mainTex (which the
    // copy left as a stable snapshot — the blit never reads the shared RT directly). The
    // host frame thus overlaps the WHOLE MW frame, not just scene 0. World image lags
    // input by one frame (UI stays current).
    bool   g_frameAheadLive = false;   // seeded from ForgeFrameAhead at init; numpad-* flips live (A/B)
    // Client produce (buildGeometryDrawLists + flush + RPC-start, D3D9-free after Tier 1a) on a
    // fresh dedicated worker. NUMPAD8 cycles 3 modes; gated on !UseRenderThread (that path owns the
    // legacy device-lock worker + a MULTITHREADED device — don't mix the two):
    //   0 OFF     — run inline on the MW main thread (pre-worker behaviour).
    //   1 FENCED  — Tier 1b: run on the worker, wait() immediately (serial, behaviour-identical;
    //               a correctness checkpoint that the produce runs correctly off-main).
    //   2 OVERLAP — Tier 2: on early-kickoff frames kick at BeginScene(0) and DON'T wait; wait at
    //               EndScene(0) before the finish. The produce overlaps MW's scene-0 draw window.
    //               Bounded to scene 0 on purpose: the worker reads the LIVE NI scene graph, which
    //               is quiescent during the scene-0 draw but mutated by mwstart(N+1) — deferring the
    //               wait past EndScene would race that. Non-early frames (interior/menu/warm-up)
    //               fence like mode 1 (the finish follows immediately, no window to overlap).
    int    g_produceMode = 3;          // 0 OFF / 1 FENCED / 2 OVERLAP / 3 PARK; NUMPAD8/panel cycles. DEFAULT PARK
                                       // (user 2026-07-18: keep overlap always on) — degrades to a
                                       // fence on non-early frames, so it's safe as the boot default.
    bool   g_produceInFlight = false;  // an async (mode 2) produce is running / not yet waited
    // Tracy host-frame fiber lane (see mge_tracy.h): the ctx spans the host's inflight window
    // (kickoff RPC issued on the produce worker -> completion drained on the main thread).
    TracyCZoneCtx g_hostZoneCtx{};
    bool          g_hostZoneOpen = false;
    double g_produceKickMs = 0.0;      // nowMs() at the async kick (overlap measurement)
    double g_produceWorkerMs = 0.0;    // last worker run duration (set by the worker)
    double g_produceBlockedMs = 0.0;   // last main-thread block at waitProduce (0 == fully hidden)
    bool   g_mainTexValid = false;     // g_mainTex holds a successfully copied host frame (deferred-blit gate)
    double g_lastBlitMs = 0.0;         // deferred EndScene blit cost, folded into the next collect's [hb]
    unsigned g_frameSerial = 0;        // Present tick (onFramePresented) — dedups the dev-key poll
    unsigned g_lastPollSerial = ~0u;   // frame serial of the last dev-key poll

    // A0 Cut-4 probe: MW frame-start span = engine Present return → BeginScene(0).
    // Everything MW does there (input, sim, AI, animation) is engine-side and has never
    // been zoned; its magnitude decides whether a frame-start no-op (Cut 4) exists at
    // all. Stamped in the proxy Present, consumed at the BeginScene(0) collect, averaged
    // into the [hb] heartbeat as mwstart=.
    double g_presentReturnMs = 0.0;    // engine Present return stamp (0 = none pending)
    double g_pendingMwStart = 0.0;     // this frame's Present→BeginScene(0) gap (ms)

    // Frame-ahead observability: last-frame values shipped to the host Stats panel via
    // DevInput each kickoff (bridge.h DevInput frame-ahead fields). Set at the collect
    // (wait) and the mwstart close; read at the next kickoff — 1-frame skew, panel only.
    double g_lastWaitMsStat = 0.0;     // last residual collect wait (pipeline success metric)
    // Last host phase split received over the wire (Tier 1, tasks/forge-host-gpu-lane.md). Feeds
    // the Tracy plots at the finish site and the [hb] echo below — the echo is the CHECK that the
    // x86/x64 struct actually arrived intact: its numbers must match mgeHost64.log's own
    // `host split` / `gpu split` for the same frames. Garbage or zeros here = layout mismatch.
    IPC::HostFrameTimings g_lastHostTimings{};
    double g_lastMwStartStat = 0.0;    // last Present-return → BeginScene(0) gap

    // --- Feeding-side spike logging --------------------------------------------------
    // Periodic FPS dips are suspected to come from the client feed (geometry/texture
    // uploads + the blocking host RPC), not the host GPU. Time each phase of onPresent
    // with QPC and, when the whole feed exceeds kSpikeMs, emit ONE breakdown line to
    // mgeXE.log so the periodic culprit (almost certainly texture streaming) is visible.
    constexpr double kSpikeMs = 10.0;  // spike-log threshold (raised from 5.0 — 5ms was log spam)
    double g_lastPresentMs = 0.0;      // for the inter-present delta (dip magnitude)

    // Baseline heartbeat: the spike log only fires on >=kSpikeMs frames, so it's BLIND to the
    // normal per-frame cost ("always slow" lives in the baseline, not the spikes). Accumulate
    // every composited frame and log avg/max over a window so the true standing-still cost and
    // its breakdown are visible without spamming.
    //
    // `overlap` = wall time between kickoff-return and finish-entry: MW frame work the host
    // render ran UNDER (recovered time). It is NOT part of `feed` — feed stays the client's
    // own cost (kickoff prep + residual wait + copy + blit), so feed is comparable across
    // fused/async modes and IS the perf baseline once the wait bucket collapses.
    constexpr unsigned kHeartbeatFrames = 300;
    struct Accum { double feed, geom, build, render, host, overlap, copy, blit, dt, mwstart; double maxFeed, maxDt; double captured; unsigned n, earlyN, pipeN;
                   double bEnsure, bEmit, bTail, bTailEnsure, bTailScan, bTailAlpha; double bKeys;
                   double maxBuild, bCaptures; };   // Phase 0: build-split sub-probe (+ Stage 0 spike observability)
    Accum g_hb = {};

    // Phase 0 build sub-probe: decompose buildGeometryDrawLists' cost (the main-thread `build=`
    // bucket) into ensureLive (live NiTriShape read — immovable, must stay main) vs emit/dispatch
    // (our own scratch memory — a worker-offload candidate) vs the tail (offscreen casters + pose
    // refresh + alpha sort/pack). Decision gate: emit ≥ ~2ms ⇒ the emit→worker offload is worth it.
    // buildGeometryDrawLists writes these; the kickoff copies them into g_kick so accumFrameStats
    // reads them skew-free alongside build= (same 1-frame pipelined discipline as tBuild).
    double g_lastBuildEnsureLiveMs = 0.0;
    double g_lastBuildEmitMs       = 0.0;
    double g_lastBuildTailMs       = 0.0;
    double g_lastBuildTailEnsureMs = 0.0;   // live pose-refresh subset of the tail (immovable)
    double g_lastBuildTailScanMs   = 0.0;   // offscreen-caster full-cacheMap scan subset of the tail
    double g_lastBuildTailAlphaMs  = 0.0;   // alpha merge/sort/pack subset of the tail
    std::uint32_t g_lastBuildKeys  = 0;
    std::uint32_t g_lastBuildCaptures = 0;  // first-sight lazy captures during the build

    // Build-spike observability (Forge Dev panel): throttled [bspike] one-liner naming the
    // component that carried a slow build. DEFAULT OFF — hot-path logging has collapsed the
    // frame twice before (LOGF/printf lessons), so even when on it is hard-throttled >=1s
    // apart + a session cap. All three build-shrink toggles are panel-only (no keys — the
    // free numpad keys collide with the water-flow handlers' edge polls).
    bool g_buildSpikeLog = false;
    // Sky/FP membership-set drift validation (panel): run the old full-cache scans alongside
    // the set-driven consumers and log !! [memb-cmp] on any kept-key mismatch. DEFAULT OFF.
    bool g_membershipValidate = false;
    // First-sight capture budget A/B (panel): capped (default, the fix) vs unlimited (the
    // old behavior — a reveal burst captures whole in one build). See buildGeometryDrawLists.
    bool g_captureBudgetOn = true;
    constexpr int      kCaptureBudgetPerFrame   = 32;
    // Frames after a cell-epoch change during which the budget stays unlimited. Every load-door
    // transition runs purgeAll() (the one-frame-flash guard), so the WHOLE cell re-captures
    // through the budget — and the first look-around takes seconds, not 30 frames (0.2s at
    // 150fps was the "buildings appear late after interior->exterior" trickle). ~4s covers a
    // full post-transition 360; the capture spikes it allows hide inside the load hitch, and
    // steady state stays capped (the eviction parent-chain rescue prevents re-capture churn).
    constexpr unsigned kCaptureEpochGraceFrames = 600;

    // Upload cost accounting (Part A). Two accumulators: g_upFrame is THIS frame's per-category
    // reship cost (parts + bytes), read into lighting[33..34] when the kickoff builds the frame
    // and then zeroed; g_upHb sums the same over the kHeartbeatFrames window for the [uploads]
    // breakdown line logged alongside [hb]. noteUpload() (called from the geometry cache) is the
    // only writer; both are single-threaded with the cache walk + kickoff.
    struct UploadAccum { std::uint32_t parts[RenderProcess::kUpCount]; std::uint64_t bytes[RenderProcess::kUpCount]; };
    UploadAccum g_upFrame = {};
    UploadAccum g_upHb    = {};

    // Async-frame split: everything the finish half needs from the kickoff half. Reset at
    // every kickoff entry; rpcPending=true only when renderSceneKickoff actually started the
    // host (finish no-ops otherwise, so every kickoff early-out stays a whole-frame no-op).
    struct KickState {
        bool rpcPending;
        bool early;     // fired from the BeginScene(0) site (DistantLand::earlyForgeKickoff latch)
        // ForgeFrameAhead: this kickoff's finish is deferred to the NEXT frame's
        // BeginScene(0) collect; the EndScene(0) composite point blits the previous
        // frame instead of finishing. Set only on early-kickoff frames — exactly the
        // frames whose whole MW frame is already validated IPC-free on both channels.
        bool deferFinish;
        // Mid-walk geometry drain (drainPendingIfFull) fired inside the async window and
        // closed it early: renderSceneFinish already ran; the composite finish must consume
        // these stored results instead of finishing again. Load-burst frames only.
        bool rpcEarlyFinished;
        // Mode 3 (PARK): this kickoff was a frame-start fire of a parked payload
        // (fireParked). The collect keys its window-close on THIS state, not on the
        // current g_produceMode, so a NUMPAD8 switch mid-flight can never orphan it.
        bool parkFired;
        bool earlyOk;
        double earlyHostMs;
        // Host phase split captured by that early finish — the composite path must plot THESE,
        // not a fresh read, or early-finish frames would plot the previous frame's numbers.
        IPC::HostFrameTimings earlyHostTimings;
        double tEarlyFinish;
        unsigned frame;
        std::uint32_t drawCount, skinnedCount, multiMapCount, lightCount, skyCount, alphaCount;
        std::uint32_t capturedCount;   // AT3: captured blended DIPs merged into the alpha list this frame
        std::uint32_t geomParts, texCount;
        std::size_t geomBytes, texBytes;
        double dtPresent;
        double tStart, tGeomFlush, tBuild, tTexFlush, tAssign, tKick;
        double bEnsure, bEmit, bTail, bTailEnsure, bTailScan, bTailAlpha; std::uint32_t bKeys;   // Phase 0 build-split sub-probe
        std::uint32_t bCaptures;   // first-sight captures during that build (Stage 0)
    };
    KickState g_kick = {};

    // Frame-ahead finish holder. The produce worker OWNS g_kick from the moment it is kicked
    // (kickoffBody resets it), so N-1's deferred-finish state must be moved out of g_kick BEFORE
    // the kick — that hand-off is what lets the kick run ahead of the finish instead of behind it.
    // stashDeferredFinish() moves it here; doDeferredFinish() consumes it. Main thread only.
    KickState g_pendingFinish = {};

    // Host-RPC serialisation gate. There is ONE shared host RT (g_importImg), so the host must not
    // begin frame N until the client has finished frame N-1 and copied that RT out. With the kick
    // moved ahead of the finish, the worker can reach renderSceneKickoff before main has done
    // either — so the worker blocks here until main signals. In practice main's finish+copy runs
    // under the worker's ~3.5ms build and the gate is already open when the worker arrives (wait
    // ≈ 0); the block is correctness insurance, not the expected path. Armed only on the async
    // (mode 2, early-kickoff) path — every other path is main-thread-serial and leaves it open.
    inline double nowMs();   // fwd (defined below)

    std::mutex              g_finishGateMx;
    std::condition_variable g_finishGateCv;
    bool                    g_finishGateOpen = true;

    // Set by kickProduceEarly, cleared at the frame-start collect. Guards the dispatcher against
    // re-kicking (and re-waiting on) a produce that frameSetupEarly already dispatched.
    bool g_earlyKicked = false;

    void armFinishGate()  { std::lock_guard<std::mutex> l(g_finishGateMx); g_finishGateOpen = false; }
    void openFinishGate() {
        { std::lock_guard<std::mutex> l(g_finishGateMx); g_finishGateOpen = true; }
        g_finishGateCv.notify_all();
    }
    // Worker-side: block until the previous frame's finish+copy has released the shared state.
    // MAIN thread returns immediately — main is the thread that OPENS the gate, so waiting on it
    // there would self-deadlock (flushGeometry has main-thread callers on the non-async paths).
    std::atomic<std::thread::id> g_produceThreadId{};

    double waitFinishGate(const char* who) {
        if (std::this_thread::get_id() != g_produceThreadId.load(std::memory_order_relaxed)) {
            return 0.0;
        }
        std::unique_lock<std::mutex> l(g_finishGateMx);
        if (g_finishGateOpen) return 0.0;
        MGE_ZoneScopedN("Forge kickoff gate (prev finish)");
        const double t0 = nowMs();
        // BOUNDED wait — this is the ONLY unbounded wait left in the pipeline (every IPC wait caps
        // at MaxWait=60s and fails loud). The gate is normally opened by main's doDeferredFinish
        // within a few ms of the kick; an unbounded wait here turned any missed open — a circular
        // main<->worker wait, or a frame whose EndScene(0) gate-open never runs — into a PERMANENT,
        // silent freeze. Every prior "doesn't recover, no log" freeze was this. Cap it, break the
        // deadlock by proceeding, and log LOUD with the caller. Proceeding early can at worst tear
        // ONE host frame (host begins N before main copied N-1's shared RT), which self-corrects the
        // next frame — infinitely better than a hang, and now observable.
        constexpr auto kGateDeadline = std::chrono::milliseconds(500);
        if (!g_finishGateCv.wait_for(l, kGateDeadline, [] { return g_finishGateOpen; })) {
            const double waited = nowMs() - t0;
            LOG::logline("!! [gate] finish-gate WEDGED %.0fms in '%s' — breaking deadlock, proceeding "
                         "(one frame may tear)", waited, who ? who : "?");
            return waited;
        }
        return nowMs() - t0;
    }

    // Gate + probe. Called at every point where the produce crosses from client-private memory
    // into shared memory the host may still be reading for the previous frame.
    void gateOnPrevFinish(const char* who) {
        const double gateMs = waitFinishGate(who);
        if (gateMs > 0.05) {
            MGE_TracyPlot("Forge kickoff gate ms", gateMs);   // compiles out in Release
            (void)gateMs;
        }
    }

    inline double nowMs() {
        static LARGE_INTEGER freq = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
        LARGE_INTEGER c; QueryPerformanceCounter(&c);
        return 1000.0 * (double)c.QuadPart / (double)freq.QuadPart;
    }

    // --- Main-thread wedge watchdog (freeze diagnostic) ------------------------------
    // Every prior freeze this session left MW's main thread wedged with NO log naming WHERE.
    // All internal waits are now bounded (finish-gate 500ms, IPC 60s) yet a freeze still
    // reproduces (user suspects alt-tab / focus loss), so a bounded-wait audit is blind. This
    // breadcrumbs the whole main-thread present pipeline into an atomic phase id + a monotonically
    // bumped tick; a detached watchdog thread notices when the tick stops advancing and logs the
    // last phase plus the focus state — the wedge names itself on the next freeze. Observe-only,
    // zero behaviour change. markMainPhase MUST be called from the MAIN thread only (never the
    // produce worker) or the breadcrumb would report the wrong thread's location.
    enum MainPhase : std::uint32_t {
        MP_MW_FRAME = 0,        // between our frames — MW input/sim/AI/anim (or paused on focus loss)
        MP_FRAME_COLLECT,       // onFrameAheadCollect entry
        MP_WAIT_PRODUCE,        // blocked on the produce worker
        MP_DEFERRED_FINISH,     // doDeferredFinish (host wait + RT copy of N-1)
        MP_FINISH_COPY,         // finishAndCopy — the host renderSceneFinish IPC wait
        MP_KICKOFF,             // onStage0CompositeKickoff (main-thread dispatch)
        MP_COMPOSITE_FINISH,    // onStage0CompositeFinish
        MP_BLIT,                // onFrameAheadBlit
        MP_COUNT
    };
    const char* const kMainPhaseName[MP_COUNT] = {
        "MW-frame(input/sim)", "frame-collect", "wait-produce", "deferred-finish",
        "finish-copy(IPC)", "kickoff", "composite-finish", "blit"
    };
    std::atomic<std::uint32_t> g_mainPhase{MP_MW_FRAME};
    std::atomic<std::uint64_t> g_mainPhaseTick{0};
    inline void markMainPhase(std::uint32_t p) {
        g_mainPhase.store(p, std::memory_order_relaxed);
        g_mainPhaseTick.fetch_add(1, std::memory_order_relaxed);
    }

    // Produce-WORKER phase breadcrumb. When main stalls in 'wait-produce' the wedge is inside the
    // worker's kickoffBody, invisible to the main breadcrumb — this names WHICH stage the worker is
    // stuck in (build-drawlists reading g_cache, a flush RPC, the kickoff RPC, …). Set from the
    // worker thread only; read by the watchdog. Observe-only.
    enum WorkerPhase : std::uint32_t {
        WK_IDLE = 0,        // between jobs / kickoffBody returned
        WK_BUILD_DRAW,      // buildGeometryDrawLists — ensureLive reads g_cache + LIVE scene graph (prime hang suspect)
        WK_BUILD_LIGHT,     // buildLightList
        WK_BUILD_SKY,       // buildSkyDrawList
        WK_BUILD_FP,        // buildFPFrame
        WK_GATE,            // gateOnPrevFinish("kickoff")
        WK_GEOM,            // flushGeometry (blocking geom RPC)
        WK_TEX,             // flushTextures (blocking tex RPC)
        WK_ASSIGN,          // assign shared draw/skinned/multimap/light/sky vectors
        WK_KICKOFF,         // renderSceneKickoff (start the host frame)
        WK_COUNT
    };
    const char* const kWorkerPhaseName[WK_COUNT] = {
        "idle/done", "build-drawlists", "build-lights", "build-sky", "build-fp",
        "gate-kickoff", "geom-flush", "tex-flush", "assign-vecs", "kickoff-rpc"
    };
    std::atomic<std::uint32_t> g_workerPhase{WK_IDLE};
    inline void markWorkerPhase(std::uint32_t p) {
        g_workerPhase.store(p, std::memory_order_relaxed);
    }

    std::atomic<bool> g_watchdogStarted{false};
    void startWedgeWatchdog() {
        bool expected = false;
        if (!g_watchdogStarted.compare_exchange_strong(expected, true)) return;
        std::thread([] {
            using namespace std::chrono;
            // A real freeze is indefinite; a legit cell-load / interior transition can pause the
            // main thread ~1-2s. 4s clears the noise and still names a true wedge fast.
            constexpr double kStallMs = 4000.0;
            std::uint64_t lastTick = g_mainPhaseTick.load(std::memory_order_relaxed);
            double lastMoveMs = nowMs();
            bool reported = false;
            for (;;) {
                std::this_thread::sleep_for(milliseconds(500));
                const std::uint64_t t = g_mainPhaseTick.load(std::memory_order_relaxed);
                const double now = nowMs();
                if (t != lastTick) {
                    if (reported) {
                        LOG::logline(">> [watchdog] main thread RECOVERED after %.1fs stall",
                                     (now - lastMoveMs) / 1000.0);
                    }
                    lastTick = t;
                    lastMoveMs = now;
                    reported = false;
                    continue;
                }
                if (!reported && now - lastMoveMs > kStallMs) {
                    const std::uint32_t ph =
                        std::min<std::uint32_t>(g_mainPhase.load(std::memory_order_relaxed), MP_COUNT - 1);
                    const HWND fg = GetForegroundWindow();
                    const std::uint32_t wph =
                        std::min<std::uint32_t>(g_workerPhase.load(std::memory_order_relaxed), WK_COUNT - 1);
                    LOG::logline("!! [watchdog] MAIN STALLED %.1fs at phase '%s' | worker at '%s' — "
                                 "foreground=%s (MW hwnd=%p fg=%p) frame=%u",
                                 (now - lastMoveMs) / 1000.0, kMainPhaseName[ph], kWorkerPhaseName[wph],
                                 (fg == g_devHwnd) ? "MW" : "OTHER", (void*)g_devHwnd, (void*)fg, g_frame);
                    reported = true;
                }
            }
        }).detach();
    }

    // Focus-transition probe: log when MW gains/loses foreground so a freeze can be correlated
    // with an alt-tab. Main thread only (called from onFrameAheadCollect).
    void probeFocusTransition() {
        static int s_lastFg = -1;
        const int fg = (GetForegroundWindow() == g_devHwnd) ? 1 : 0;
        if (fg != s_lastFg) {
            if (s_lastFg != -1) {
                LOG::logline("-- [focus] MW %s (frame=%u)",
                             fg ? "GAINED foreground" : "LOST foreground", g_frame);
            }
            s_lastFg = fg;
        }
    }

    // --- M1b: opaque-geometry capture + upload -----------------------------------
    // The cache hands us model-space parts (pos+normal+indices). We assign each a
    // dense host slot, accumulate them into a pending byte blob, and flush them to
    // the Forge host in window-sized whole-part chunks at present time (a safe point
    // with no other RPC in flight). Slots are stable per cache key; re-upload only on
    // revision change. The host stores meshes slot-indexed for M1c's per-frame draw.
    // Chunked shared vec (see ipc/geomwire.h): an 8MB window as 8x1MB chunks, so the
    // Vec reservation (maxSize*windowBytes) stays small. Chunk cap = the full window.
    constexpr std::uint32_t kGeomChunkCap = IPC::kGeomWindowBytes;  // <= window (assign_bytes)
    // Mid-walk drain cap on the staging blob. A showcase-scale interior can capture ~1GB of
    // geometry in ONE cell-load walk; accumulating it all until the present-time flush killed
    // the 32-bit process (bad_alloc growing g_pendingBlob past ~700MB). Draining mid-walk is
    // safe: geometry rides its own IPC channel, and the host is never mid-frame while the
    // client walks (renderScene fence-waits before returning; only the arena-blind Hi-Z
    // prologue overlaps). Steady-state frames never come near this, so it only fires on loads.
    constexpr std::size_t kPendingFlushBytes = 32u << 20;           // 32 MB

    std::optional<IPC::VecView<IPC::GeomChunk>> g_geomVec;          // persistent geometry upload vec
    std::optional<IPC::VecView<IPC::GeomChunk>> g_drawVec;          // persistent per-frame draw-list vec
    std::optional<IPC::VecView<IPC::GeomChunk>> g_skinnedVec;       // persistent per-frame skinned draw-list vec
    std::optional<IPC::VecView<IPC::GeomChunk>> g_multiMapVec;      // persistent per-frame multi-map draw-list vec (Tier 4)
    std::optional<IPC::VecView<IPC::GeomChunk>> g_lightVec;         // persistent per-frame point-light vec (Tier 3a)
    std::optional<IPC::VecView<IPC::GeomChunk>> g_skyVec;           // persistent per-frame sky draw-list vec (SK1)
    std::optional<IPC::VecView<IPC::GeomChunk>> g_alphaVec;         // persistent per-frame sorted-alpha draw-list vec (AT1)
    std::optional<IPC::VecView<IPC::GeomChunk>> g_fpDrawVec;        // persistent per-frame FP rigid draw-list vec (FP1a)
    std::optional<IPC::VecView<IPC::GeomChunk>> g_fpSkinnedVec;     // persistent per-frame FP skinned draw-list vec (FP1a)
    std::optional<IPC::VecView<IPC::GeomChunk>> g_fpAlphaVec;       // persistent per-frame FP alpha draw-list vec (FP1c)
    std::vector<std::uint8_t>                 g_pendingBlob;        // packed parts awaiting flush
    std::uint32_t                             g_pendingParts = 0;
    // Cache key -> host slot, plus per-key cached bindless texture slots so the
    // per-frame draw-list build doesn't re-run resolveTextureSlot's normalize
    // (heap string) + string-hash find for every item every frame. A cached slot
    // is valid iff BOTH hold:
    //   - the entry's texture-name POINTER is unchanged (NiSourceTexture fileName
    //     storage is stable; NiFlipController animation swaps to a different
    //     SourceTexture = different pointer, so animated textures still re-resolve)
    //   - the epoch matches g_texEpoch (bumped when the LRU recycles a bindless
    //     slot to a new texture, which invalidates every cached slot value).
    // The cached fast path still refreshes g_slotLastUsed so the LRU stays exact.
    struct SlotInfo {
        std::uint32_t slot = 0;              // host geometry slot (stable per key)
        const char*   baseNamePtr = nullptr; // texture-name identity for baseSlot
        std::uint32_t baseSlot = 0;
        std::uint32_t baseEpoch = 0;
        const char*   ovNamePtr = nullptr;   // texture-name identity for ovSlot
        std::uint32_t ovSlot = 0;
        std::uint32_t ovEpoch = 0;           // separate epochs: one field re-validating
                                             // the other's stale slot after a recycle
                                             // would alias textures
    };
    std::unordered_map<std::uint32_t, SlotInfo> g_keySlot;
    std::uint32_t g_texEpoch = 0;            // bumped on bindless-slot LRU recycle
    // Last-shipped identity per cache key. Dedup is on (modelId, vc, rev), NOT rev alone:
    // the key is a recycled NiTriShape*, so a new object can inherit a freed key+slot; the
    // GeometryData ptr (modelId) + vertexCount disambiguate it (see captureGeometry).
    struct UploadSig { std::uint32_t id; std::uint32_t vc; std::uint16_t rev; };
    std::unordered_map<std::uint32_t, UploadSig> g_uploadedRev;    // cache key -> last sent identity
    std::uint32_t                             g_nextSlot = 0;
    // Deferred host-slot release queue. Evicted keys resolve to their (old, monotonic, never-reused)
    // host slot immediately in drainReleasedSlots, but the release SENTINEL is shipped a bounded
    // number per flush (appendBoundedReleaseRecords) so a mass eviction can't flood the blocking
    // geometry RPC at present time. Same produce-thread context as g_keySlot/g_pendingBlob → no lock.
    std::vector<std::uint32_t>                g_pendingReleaseSlots;
    // Per-flush release cap. Freeze evidence: 517 release parts in one flush cost 5.49ms host-side
    // (~10us each) and the backlog blew the 32MB staging cap. 64/flush ~ 0.7ms, backlog drains over
    // frames. Slots are monotonic so deferral is always safe (a late release only frees a dead mesh).
    constexpr std::uint32_t                   kMaxReleasesPerFlush = 64;
    std::vector<std::uint8_t>                 g_drawScratch;        // packed DrawItemWire[] this frame
    std::vector<std::uint8_t>                 g_skinnedScratch;     // packed [SkinnedDrawWire][palette]* this frame
    std::vector<std::uint8_t>                 g_fpDrawScratch;      // packed FP rigid DrawItemWire[] this frame (FP1a)
    std::vector<std::uint8_t>                 g_fpSkinnedScratch;   // packed FP [SkinnedDrawWire][palette]* this frame (FP1a)
    std::vector<std::uint8_t>                 g_fpAlphaScratch;     // packed FP AlphaDrawWire[] this frame (FP1c, back-to-front)
    std::vector<std::uint8_t>                 g_multiMapScratch;    // packed MultiMapDrawWire[] this frame (Tier 4)
    std::vector<std::uint8_t>                 g_lightScratch;       // packed PointLightWire[] this frame

    // Offscreen shadow casters: skinned + multimap NPC parts this far (world units) from the eye
    // are re-emitted for shadowing even when frustum-culled, so their shadows don't freeze/lose
    // parts as they leave view. ~2r of the largest shadow lights; bounded to a few NPCs indoors.
    constexpr float kShadowCasterRadius = 2048.0f;

    // Live near-actor pose refresh (fixes the offscreen dyn-shadow drift). Offscreen movers
    // were re-emitted from CACHED palettes, which the geometry cache only refreshes on the
    // eviction sweep (~every 30 frames) — so a still-animating offscreen NPC's shadow baked a
    // 30-frame-stale pose and SNAPPED every 30 frames (measured: skinnedΔ=0 for 29f, ~1.9u
    // spike on the 30th). ON = fresh pose every frame; OFF = the old 30-frame-stale re-emit (A/B).
    //
    // This ensureLive()s OFFSCREEN keys, which is only safe because GeometryCache now holds a real
    // engine reference to every cached shape (g_geomRefs), pinning it alive for as long as we hold
    // the key. It shipped originally on the argument that these movers are radius-gated (2048u) →
    // "near ⇒ alive" — which is a PROXIMITY argument, not a liveness one. A scene teardown (save
    // load) frees near and far alike, and this path then dereferenced the freed shapes: crash
    // 2026-07-13, AV in ensureLive → NI::Pointer::claim. Distance never implied liveness; only
    // owning a reference does.
    bool g_shadowLiveNearActors = true;

    // --- P2 light identity tracking -------------------------------------------------------
    // Persistent per-light id across frames, keyed by the scene-graph snapshot's stable
    // NI::PointLight* (PointLight::source). Feeds the host shadow manager a real identity
    // (P2 = log-only; P3 holds a shadow slot on the id instead of the position-tolerance
    // hack). A live light keeps its id; the id is RECYCLED (fresh + NEW flag) when the same
    // pointer resurfaces after a long gap (a freed NiLight's address reused) OR teleports
    // farther than max(2r, floor) in one step (cell-door swap onto the same address).
    struct LightTrack {
        std::uint32_t id;
        std::uint32_t lastFrame;    // g_frame it was last seen
        float         lastPos[3];   // world position when last seen (for moved / teleport test)
    };
    std::unordered_map<const void*, LightTrack> g_lightTracks;
    std::uint32_t                               g_nextLightId = 1;   // 0 = "no identity"
    constexpr std::uint32_t kLightIdGapFrames = 10;      // gap > this frames -> recycle id
    constexpr float         kLightIdJumpFloor = 512.0f;  // teleport floor (min of the 2r/floor test)
    constexpr float         kLightMovedEps    = 0.5f;    // moved flag threshold (world units)
    constexpr std::uint32_t kLightTrackEvict  = 300u;    // drop map entries unseen this long

    // Cell-change shadow eviction. A load-door transition (interior<->interior/exterior, fast
    // travel, coc) discontinuously swaps the resident geometry AND recycles NiPointLight
    // addresses, so a lantern in the NEW cell can inherit an OLD cell's light id (address reuse
    // inside the recycle window at a similar local position) and, host-side, keep sampling the
    // previous cell's cached shadow tile — the "shadows from another interior" bug. This monotonic
    // epoch is bumped on any such transition (shipped in lighting[19]); the host evicts every
    // shadow slot + stale caster record when it changes, and the client drops all identity tracks
    // so every new-cell light takes a fresh id. Continuous exterior walking never trips it.
    std::uint32_t                               g_cellEpoch = 0;
    constexpr float         kCellTeleportDist = 8192.0f;   // one MW cell moved in ONE frame = teleport

    // --- Produce mode 3 "PARK-AND-FIRE" ---------------------------------------------------
    // The worker builds frame N's payload into the client-private scratch vectors and PARKS it
    // (no flush/assign/RPC); at the START of frame N+1 (frameSetupEarly, camera fresh, window
    // closed) the main thread fires it with a restamped camera (fireParked). The payload parks
    // IN PLACE: the scratch vectors are cleared only at the start of the next build, and the
    // fire (main, frame start) always precedes the next worker kick in the same frameSetupEarly
    // — zero copy. Only counts + the FP bundle + the build-time eye need holding here.
    // Invalidated (a) at every kickoffBody entry (a serial/inline kickoff supersedes it),
    // (b) at fire on a cell-epoch mismatch (teleport between build and fire → drop, one
    // repeated composite frame), (c) on fire (consumed).
    struct ParkedPayload {
        bool valid = false;
        std::uint32_t epoch = 0;              // g_cellEpoch at build → fire-time invalidation
        float bakeEye[3] = {};                // DistantLand::eyePos at build (payload's relative space)
        std::uint32_t drawCount = 0, skinnedCount = 0, multiMapCount = 0,
                      lightCount = 0, skyCount = 0, alphaCount = 0;
        IPC::FPFrame fpFrame;                 // shipped VERBATIM at fire — self-consistent
        std::uint32_t fpDraws = 0, fpSkinnedDraws = 0, fpAlphaDraws = 0;   // (pose N, arm-cam N) bundle
        bool fpHave = false;
        std::uint32_t capturedEmitted = 0;    // AT3 captured count at build (g_capturedEmitted snapshot)
        double tBuildEnd = 0.0, buildMs = 0.0;   // telemetry (parkAge at fire)
    };
    ParkedPayload g_park;

    std::vector<std::uint8_t>                 g_skyScratch;         // packed SkyDrawWire[] this frame (SK1)
    std::vector<std::uint8_t>                 g_alphaScratch;       // packed AlphaDrawWire[] this frame (AT1, back-to-front)

    // --- AT3 captured-alpha (particles/smoke/flames restored under Forge) ------------
    // MW already did the CPU billboarding + sort for these blended DIPs before the reject gate
    // (distantland.cpp inspectIndexedPrimitive). captureAlphaDraw Locks the live VB/IB, copies
    // the final verts + rebased indices into pending scratch DURING frame N, then the NEXT
    // buildGeometryDrawLists (kickoff N+1) merges them into the sorted-alpha list (ONE sort with
    // the cached blends) and ships the geometry to the host. Single-buffered: one pending scratch,
    // consumed + cleared at the next consume. 1-frame-late particle positions are accepted (they
    // were invisible before this feature); camera-relative subtraction uses the CURRENT eyePos at
    // emit so there is no swim.
    struct CapturedAlphaRec {
        std::uint32_t vertexBase, vertexCount;   // into g_capVertScratch
        std::uint32_t indexBase, indexCount;     // into g_capIdxScratch
        float world[16];                          // ABSOLUTE model->world (emit subtracts eyePos)
        float centroid[3];                        // ABSOLUTE world-space centroid (depth sort)
        std::uint32_t texIndex;
        std::uint32_t srcBlend, destBlend;
        float alphaRef;
        std::uint32_t vColSource;
        float matDiffuse[3], matAmbient[3], matEmissive[3], matAlpha;
    };
    // CONSUME/SHIP side (produce WORKER only): buildGeometryDrawLists reads g_capRecs, appends FP
    // particles to g_capVertScratch/g_capIdxScratch, and the kickoff ships them. Filled by the swap
    // below from the main-thread INCOMING buffers — never written by captureAlphaDraw directly.
    std::vector<IPC::GeomVertexWire> g_capVertScratch;   // consumed/shipped verts (frame N-1)
    std::vector<std::uint16_t>       g_capIdxScratch;    // consumed/shipped indices (frame N-1)
    std::vector<CapturedAlphaRec>    g_capRecs;          // consumed records (frame N-1)
    std::optional<IPC::VecView<IPC::GeomChunk>> g_capturedVec;   // shipped at kickoff ([verts][indices])
    // Old-msoc double-draw guard: (d3dTexture<<32 | vertCount) of every cached blended shape the
    // host already draws; a captured DIP that matches one is a duplicate and is skipped.
    // DOUBLE-BUFFERED (2026-07-21): the WORKER clears+builds g_alphaDedup in buildGeometryDrawLists
    // while MAIN's captureAlphaDraw reads it — the same worker-build/main-read race as the capture
    // scratch. Reading it directly made the glow-window blended-MM draw flicker white/orange (main
    // read the set before the worker inserted the window's key → dedup miss → MW's AT3 white drew
    // too; worse at distance = later insert order). swapCaptureBuffers() swaps the worker's freshly
    // built set into g_alphaDedupActive at the drained dispatch, and MAIN reads ONLY g_alphaDedupActive
    // (the PREVIOUS frame's completed set — 1 frame late, fine since the visible set changes slowly).
    std::unordered_set<std::uint64_t> g_alphaDedup;         // WORKER builds (clear + insert)
    std::unordered_set<std::uint64_t> g_alphaDedupActive;   // MAIN reads (stable prev-frame snapshot)
    // Per-frame texture-slot memo for the CONSUME side (FP-particle append): rs.texture ptr -> slot.
    std::unordered_map<IDirect3DTexture9*, std::uint32_t> g_capTexMemo;

    // CAPTURE/INCOMING side (MAIN thread only): captureAlphaDraw appends here during MW's scene
    // render. swapCaptureBuffers() (main, at the per-frame produce dispatch, AFTER waitProduce drains
    // the previous worker) std::swaps these into the consume-side buffers above and clears these, so
    // MAIN's fill and the WORKER's consume/ship always touch DIFFERENT vectors under produce OVERLAP.
    // Single-buffered before (2026-07-21): main's push_back realloc raced the worker's iterate/clear,
    // handing torn CapturedAlphaRec.world[16] (90°/stretch) and torn vertex ranges (vanish) — the
    // "wrong geometry frames" on flames/smoke + the glow-window alpha quad. The swap point is drained
    // by waitProduce and re-kicked, both full barriers, so no atomics are needed (as with g_drawScratch).
    std::vector<IPC::GeomVertexWire> g_capInVerts;       // captured verts, filling for frame N
    std::vector<std::uint16_t>       g_capInIdx;         // captured indices, filling for frame N
    std::vector<CapturedAlphaRec>    g_capInRecs;        // captured records, filling for frame N
    std::unordered_map<IDirect3DTexture9*, std::uint32_t> g_capInTexMemo;   // main's per-frame slot memo
    // Swap the just-filled INCOMING captures onto the consume/ship side and reset incoming for the new
    // frame. Call on the MAIN thread at the produce dispatch, AFTER waitProduce (previous worker done)
    // and BEFORE the kick / inline kickoffBody + this frame's captureAlphaDraw appends.
    void swapCaptureBuffers() {
        std::swap(g_capVertScratch, g_capInVerts);
        std::swap(g_capIdxScratch,  g_capInIdx);
        std::swap(g_capRecs,        g_capInRecs);
        std::swap(g_capTexMemo,     g_capInTexMemo);
        g_capInVerts.clear();
        g_capInIdx.clear();
        g_capInRecs.clear();
        g_capInTexMemo.clear();
        // Hand the worker's just-built alpha-dedup set to MAIN (captureAlphaDraw reads Active only).
        // The worker refills g_alphaDedup next frame (it clears at buildGeometryDrawLists start).
        std::swap(g_alphaDedup, g_alphaDedupActive);
    }
    // Drop counters (one-shot logged): lock fail / cap overflow / INDEX32 rebase / no-name-white.
    std::uint32_t g_capDropLock = 0, g_capDropCap = 0, g_capDropIdx32 = 0, g_capNoName = 0;
    // [alpha-dedup] Is the (texture, vertexCount) dedup key eating LIVE particle draws?
    // World particles are never cached shapes ("not NiTriShapes, not captured by the cache walk"),
    // so the dedup must never drop one — but its key is a coarse alias, and a particle system whose
    // count momentarily makes vertCount match a cached blend on the SAME texture hashes identically
    // (a 1-particle flame and a cached flame billboard are both a 4-vert 2-tri quad). Particle counts
    // vary per frame, so such a draw would drop and undrop = flicker. Count drops and the distinct
    // keys they hit; a nonzero, FLUCTUATING count is the flicker's signature.
    std::uint32_t g_capDedupDrops = 0;        // this frame
    std::uint32_t g_capDedupDropsMin = ~0u, g_capDedupDropsMax = 0;   // over the log window
    std::uint32_t g_capDedupWindow = 0;
    // Captured blended DIPs merged into the alpha list this frame (heartbeat cap=N). Set in
    // buildGeometryDrawLists before the records are cleared; read into KickState at kickoff.
    std::uint32_t g_capturedEmitted = 0;

    // --- Phase 2 bindless texture residency (client) ---
    // Each unique texture (by normalized name) gets a dense bindless slot; its raw DDS bytes
    // are loaded once via BSA::loadFileBytes and shipped to the host (which decodes into
    // gTextures[slot]). Misses/oversize map to slot 0 (host default white) and are cached so
    // we don't retry. Rides the geometry channel's chunked vec (g_texVec).
    std::optional<IPC::VecView<IPC::GeomChunk>> g_texVec;           // persistent texture upload vec
    // Texture residency (g_texSlot + g_slotName/g_slotLastUsed + g_nextTexSlot/g_texEpoch +
    // g_texPendingBlob/Count) is mutated from TWO threads: the produce worker's build
    // (buildGeometryDrawLists -> resolveTextureSlot) AND the MAIN thread's captured-alpha proxy
    // interception (captureAlphaDraw -> resolveTextureSlot). produce-off-main OVERLAPS them by
    // design, so a concurrent emplace/rehash on g_texSlot corrupted the heap (crash walking a torn
    // std::string key in _Forced_rehash, 2026-07-20). This mutex serialises the whole subsystem.
    // NEVER held across a blocking IPC RPC — flushTextures swaps the blob out under the lock and
    // RPCs on the local copy (holding it across texUploadBlocking would be a new 60s-freeze class).
    std::mutex                                g_texResidencyMx;
    std::unordered_map<std::string, std::uint32_t> g_texSlot;       // normalized name -> bindless slot
    std::uint32_t                             g_nextTexSlot = 1;    // 0 = host default white
    std::vector<std::uint8_t>                 g_texPendingBlob;     // [TexUploadWire][dds]* awaiting flush
    std::uint32_t                             g_texPendingCount = 0;
    // LRU eviction over the client's bindless range [1, kMaxTextures-kDlReserve). Residency is
    // cumulative all session (no per-cell reset), so without eviction a long traversal exhausts the
    // slots and every NEW near texture goes white ("near white far from spawn"). Recycle the
    // least-recently-used slot instead. g_frame is the LRU clock.
    std::vector<std::string>                  g_slotName;           // slot -> name (for eviction; size kMaxTextures)
    std::vector<std::uint32_t>                g_slotLastUsed;       // slot -> last g_frame it was referenced

    // Per-frame draw list rides a 4-chunk (4MB) vec: ~61K DrawItemWire, well over the
    // host's kMaxDraws cap. Geometry vec stays 8 chunks (8MB).
    constexpr unsigned kDrawChunks = 4;

    void initSceneVecs();   // defined below; called from lazyInit
    void flushGeometry();
    void flushTextures();
    std::uint32_t resolveTextureSlot(const char* textureName);

    // --- DXVK Vulkan-interop seam ---
    // MW's MAIN device is DXVK (Vulkan-backed). DXVK exposes ID3D9VkInteropDevice, which
    // hands us its own VkInstance/VkPhysicalDevice/VkDevice/VkQueue. We:
    //   - import the Forge host's shared D3D12 render target (an NT handle) as external
    //     memory bound to a VkImage on DXVK's device, and
    //   - create a DXVK-owned D3D9 render-target texture (via ID3D9VkInteropDevice::
    //     CreateImage, with TRANSFER_DST usage) whose VkImage we copy INTO each frame
    //     with a raw vkCmdCopyImage on DXVK's queue.
    // The MAIN device then StretchRects that D3D9 texture to its backbuffer (unchanged).
    // One Vulkan device throughout: no native d3d9, no D3D9On12, no cross-backend share.
    // The D3D12->Vulkan import handle type (D3D12_RESOURCE / D3D11_TEXTURE) is chosen at
    // bring-up by querying the physical device, so the same binary adapts per vendor.

    ID3D9VkInteropDevice* g_vki = nullptr;       // QI of MW's DXVK device (owned ref)

    VkInstance       g_inst   = VK_NULL_HANDLE;  // DXVK's handles (borrowed; do not destroy)
    VkPhysicalDevice g_phys   = VK_NULL_HANDLE;
    VkDevice         g_dev    = VK_NULL_HANDLE;
    VkQueue          g_queue  = VK_NULL_HANDLE;
    uint32_t         g_qFamily = 0;

    HANDLE           g_hostHandle = nullptr;     // Forge RT shared NT handle (we own it)
    VkExternalMemoryHandleTypeFlagBits g_htype = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    VkImage          g_importImg = VK_NULL_HANDLE;  // imported host RT (owned)
    VkDeviceMemory   g_importMem = VK_NULL_HANDLE;  // imported external memory (owned)

    IDirect3DTexture9* g_mainTex = nullptr;      // DXVK D3D9 RT texture; copy dst + blit src
    VkImage            g_dstImg = VK_NULL_HANDLE;   // its backing VkImage (borrowed)
    VkImageLayout      g_dstLayout = VK_IMAGE_LAYOUT_GENERAL;  // DXVK's resting layout for g_mainTex

    VkCommandPool   g_cmdPool = VK_NULL_HANDLE;  // owned
    VkCommandBuffer g_cmd     = VK_NULL_HANDLE;
    VkFence         g_fence   = VK_NULL_HANDLE;  // owned

    HMODULE g_vulkanDll = nullptr;

    // Dynamically-resolved Vulkan entry points (DXVK's loader, via vulkan-1.dll).
    struct VkApi {
        PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
        PFN_vkGetDeviceProcAddr   GetDeviceProcAddr;
        PFN_vkGetPhysicalDeviceImageFormatProperties2 GetPhysicalDeviceImageFormatProperties2;
        PFN_vkGetPhysicalDeviceMemoryProperties       GetPhysicalDeviceMemoryProperties;
        PFN_vkCreateImage                CreateImage;
        PFN_vkDestroyImage               DestroyImage;
        PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
        PFN_vkAllocateMemory             AllocateMemory;
        PFN_vkFreeMemory                 FreeMemory;
        PFN_vkBindImageMemory            BindImageMemory;
        PFN_vkGetMemoryWin32HandlePropertiesKHR GetMemoryWin32HandlePropertiesKHR;
        PFN_vkCreateCommandPool          CreateCommandPool;
        PFN_vkDestroyCommandPool         DestroyCommandPool;
        PFN_vkAllocateCommandBuffers     AllocateCommandBuffers;
        PFN_vkBeginCommandBuffer         BeginCommandBuffer;
        PFN_vkEndCommandBuffer           EndCommandBuffer;
        PFN_vkResetCommandBuffer         ResetCommandBuffer;
        PFN_vkCmdPipelineBarrier         CmdPipelineBarrier;
        PFN_vkCmdCopyImage               CmdCopyImage;
        PFN_vkQueueSubmit                QueueSubmit;
        PFN_vkCreateFence                CreateFence;
        PFN_vkDestroyFence               DestroyFence;
        PFN_vkWaitForFences              WaitForFences;
        PFN_vkResetFences                ResetFences;
    } vk = {};

    bool loadVulkan() {
        if (!g_vulkanDll) {
            g_vulkanDll = LoadLibraryA("vulkan-1.dll");
        }
        if (!g_vulkanDll) {
            LOG::logline("!! [seam] LoadLibrary(vulkan-1.dll) failed — seam disabled");
            return false;
        }
        vk.GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)GetProcAddress(g_vulkanDll, "vkGetInstanceProcAddr");
        if (!vk.GetInstanceProcAddr) {
            LOG::logline("!! [seam] vkGetInstanceProcAddr missing — seam disabled");
            return false;
        }
        vk.GetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vk.GetInstanceProcAddr(g_inst, "vkGetDeviceProcAddr");

        bool ok = vk.GetDeviceProcAddr != nullptr;
        #define INST(name) do { vk.name = (PFN_vk##name)vk.GetInstanceProcAddr(g_inst, "vk" #name); ok = ok && vk.name; } while (0)
        #define DEV(name)  do { vk.name = (PFN_vk##name)vk.GetDeviceProcAddr(g_dev, "vk" #name);   ok = ok && vk.name; } while (0)
        INST(GetPhysicalDeviceImageFormatProperties2);
        INST(GetPhysicalDeviceMemoryProperties);
        DEV(CreateImage);
        DEV(DestroyImage);
        DEV(GetImageMemoryRequirements);
        DEV(AllocateMemory);
        DEV(FreeMemory);
        DEV(BindImageMemory);
        DEV(GetMemoryWin32HandlePropertiesKHR);
        DEV(CreateCommandPool);
        DEV(DestroyCommandPool);
        DEV(AllocateCommandBuffers);
        DEV(BeginCommandBuffer);
        DEV(EndCommandBuffer);
        DEV(ResetCommandBuffer);
        DEV(CmdPipelineBarrier);
        DEV(CmdCopyImage);
        DEV(QueueSubmit);
        DEV(CreateFence);
        DEV(DestroyFence);
        DEV(WaitForFences);
        DEV(ResetFences);
        #undef INST
        #undef DEV
        if (!ok) {
            LOG::logline("!! [seam] failed to resolve required Vulkan entry points — seam disabled");
        }
        return ok;
    }

    uint32_t pickMemoryType(uint32_t typeBits, VkMemoryPropertyFlags want) {
        VkPhysicalDeviceMemoryProperties mp = {};
        vk.GetPhysicalDeviceMemoryProperties(g_phys, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
                return i;
            }
        }
        return UINT32_MAX;
    }

    // Is a given external handle type importable for our RT format on this physical device?
    bool handleTypeImportable(VkExternalMemoryHandleTypeFlagBits htype) {
        VkPhysicalDeviceExternalImageFormatInfo ext = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO };
        ext.handleType = htype;
        VkPhysicalDeviceImageFormatInfo2 fi = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2 };
        fi.pNext  = &ext;
        fi.format = VK_FORMAT_B8G8R8A8_UNORM;
        fi.type   = VK_IMAGE_TYPE_2D;
        fi.tiling = VK_IMAGE_TILING_OPTIMAL;
        fi.usage  = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VkExternalImageFormatProperties efp = { VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES };
        VkImageFormatProperties2 ifp = { VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2 };
        ifp.pNext = &efp;
        if (vk.GetPhysicalDeviceImageFormatProperties2(g_phys, &fi, &ifp) != VK_SUCCESS) {
            return false;
        }
        return (efp.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
    }

    // Import the host's shared NT handle as a VkImage on DXVK's device.
    bool importHostImage() {
        VkExternalMemoryImageCreateInfo extImg = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
        extImg.handleTypes = g_htype;
        VkImageCreateInfo ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ici.pNext         = &extImg;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = VK_FORMAT_B8G8R8A8_UNORM;
        ici.extent        = { g_w, g_h, 1 };
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vk.CreateImage(g_dev, &ici, nullptr, &g_importImg) != VK_SUCCESS) {
            LOG::logline("!! [seam] vkCreateImage (imported host RT) failed");
            return false;
        }

        VkMemoryRequirements mr = {};
        vk.GetImageMemoryRequirements(g_dev, g_importImg, &mr);

        VkMemoryWin32HandlePropertiesKHR whp = { VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR };
        uint32_t handleTypeBits = mr.memoryTypeBits;
        if (vk.GetMemoryWin32HandlePropertiesKHR(g_dev, g_htype, g_hostHandle, &whp) == VK_SUCCESS) {
            handleTypeBits &= whp.memoryTypeBits;
        }
        uint32_t typeIdx = pickMemoryType(handleTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (typeIdx == UINT32_MAX) {
            typeIdx = pickMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        if (typeIdx == UINT32_MAX) {
            LOG::logline("!! [seam] no device-local memory type for imported handle");
            return false;
        }

        VkMemoryDedicatedAllocateInfo ded = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
        ded.image = g_importImg;
        VkImportMemoryWin32HandleInfoKHR imp = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR };
        imp.pNext      = &ded;
        imp.handleType = g_htype;
        imp.handle     = g_hostHandle;
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.pNext           = &imp;
        mai.allocationSize  = mr.size;
        mai.memoryTypeIndex = typeIdx;
        VkResult r = vk.AllocateMemory(g_dev, &mai, nullptr, &g_importMem);
        if (r != VK_SUCCESS) {
            LOG::logline("!! [seam] *** vkAllocateMemory (import) FAILED VkResult=%d *** (handle type %d not importable on this GPU)", (int)r, (int)g_htype);
            return false;
        }
        if (vk.BindImageMemory(g_dev, g_importImg, g_importMem, 0) != VK_SUCCESS) {
            LOG::logline("!! [seam] vkBindImageMemory (imported host RT) failed");
            return false;
        }
        return true;
    }

    // Create the DXVK D3D9 RT texture (copy destination + StretchRect source) and capture
    // its backing VkImage + resting layout. Priming with ColorFill forces DXVK to settle
    // the image into a defined, content-preserving layout (so our copy isn't discarded).
    bool createDstTexture() {
        D3D9VkExtImageDesc d = {};
        d.Type               = D3DRTYPE_TEXTURE;
        d.Width              = g_w;
        d.Height             = g_h;
        d.Depth              = 1;
        d.MipLevels          = 1;
        d.Usage              = D3DUSAGE_RENDERTARGET;
        d.Format             = D3DFMT_A8R8G8B8;
        d.Pool               = D3DPOOL_DEFAULT;
        d.MultiSample        = D3DMULTISAMPLE_NONE;
        d.MultiSampleQuality = 0;
        d.Discard            = false;
        d.IsAttachmentOnly   = false;
        d.IsLockable         = false;
        d.ImageUsage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT;   // we vkCmdCopyImage INTO it

        IDirect3DResource9* res = nullptr;
        if (FAILED(g_vki->CreateImage(&d, &res)) || !res) {
            LOG::logline("!! [seam] ID3D9VkInteropDevice::CreateImage (dst RT) failed");
            return false;
        }
        HRESULT hr = res->QueryInterface(__uuidof(IDirect3DTexture9), (void**)&g_mainTex);
        res->Release();
        if (FAILED(hr) || !g_mainTex) {
            LOG::logline("!! [seam] dst resource is not a Texture9");
            return false;
        }

        // Prime the layout: ColorFill the surface so DXVK transitions+tracks a real layout.
        IDirect3DSurface9* surf = nullptr;
        if (SUCCEEDED(g_mainTex->GetSurfaceLevel(0, &surf)) && surf) {
            // g_vki's device is MW's device; reach it through the surface's device.
            IDirect3DDevice9* dev = nullptr;
            if (SUCCEEDED(surf->GetDevice(&dev)) && dev) {
                dev->ColorFill(surf, nullptr, D3DCOLOR_ARGB(255, 0, 0, 0));
                dev->Release();
            }
            surf->Release();
        }
        g_vki->FlushRenderingCommands();

        ID3D9VkInteropTexture* vkt = nullptr;
        if (FAILED(g_mainTex->QueryInterface(__uuidof(ID3D9VkInteropTexture), (void**)&vkt)) || !vkt) {
            LOG::logline("!! [seam] dst texture has no ID3D9VkInteropTexture");
            return false;
        }
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        hr = vkt->GetVulkanImageInfo(&g_dstImg, &layout, nullptr);
        vkt->Release();
        if (FAILED(hr) || g_dstImg == VK_NULL_HANDLE) {
            LOG::logline("!! [seam] GetVulkanImageInfo (dst) failed");
            return false;
        }
        // Restore target must preserve contents; never transition back to UNDEFINED.
        g_dstLayout = (layout == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_IMAGE_LAYOUT_GENERAL : layout;
        LOG::logline(">> [seam] dst DXVK texture VkImage=%p resting layout=%d", (void*)g_dstImg, (int)g_dstLayout);
        return true;
    }

    bool createCopyInfra() {
        VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = g_qFamily;
        if (vk.CreateCommandPool(g_dev, &pci, nullptr, &g_cmdPool) != VK_SUCCESS) {
            return false;
        }
        VkCommandBufferAllocateInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cbi.commandPool        = g_cmdPool;
        cbi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        if (vk.AllocateCommandBuffers(g_dev, &cbi, &g_cmd) != VK_SUCCESS) {
            return false;
        }
        VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        return vk.CreateFence(g_dev, &fci, nullptr, &g_fence) == VK_SUCCESS;
    }

    void releaseAll() {
        if (g_dev) {
            if (g_fence)     { vk.DestroyFence(g_dev, g_fence, nullptr); g_fence = VK_NULL_HANDLE; }
            if (g_cmdPool)   { vk.DestroyCommandPool(g_dev, g_cmdPool, nullptr); g_cmdPool = VK_NULL_HANDLE; g_cmd = VK_NULL_HANDLE; }
            if (g_importImg) { vk.DestroyImage(g_dev, g_importImg, nullptr); g_importImg = VK_NULL_HANDLE; }
            if (g_importMem) { vk.FreeMemory(g_dev, g_importMem, nullptr); g_importMem = VK_NULL_HANDLE; }
        }
        if (g_mainTex)     { g_mainTex->Release(); g_mainTex = nullptr; }
        g_mainTexValid = false;
        g_dstImg = VK_NULL_HANDLE;
        if (g_hostHandle)  { CloseHandle(g_hostHandle); g_hostHandle = nullptr; }
        if (g_vki)         { g_vki->Release(); g_vki = nullptr; }
        g_inst = VK_NULL_HANDLE; g_phys = VK_NULL_HANDLE; g_dev = VK_NULL_HANDLE; g_queue = VK_NULL_HANDLE;
    }

    // Re-derive the current internal render size from g_renderScale + the backbuffer, clamp it to
    // the allocation (g_w/g_h), and restamp it into every subsequent host render RPC. Cheap — call
    // on any scale change (panel slider). Takes effect on the next kickoff with no reallocation.
    void recomputeRenderSize() {
        float s = g_renderScale;
        if (s < kMinRenderScale) s = kMinRenderScale;
        if (s > kMaxRenderScale) s = kMaxRenderScale;
        UINT rw = (UINT)(g_bbW * s + 0.5f);
        UINT rh = (UINT)(g_bbH * s + 0.5f);
        if (rw > g_w) rw = g_w;   if (rw < 1) rw = 1;
        if (rh > g_h) rh = g_h;   if (rh < 1) rh = 1;
        g_rw = rw;
        g_rh = rh;
        if (g_client) g_client->setNextRenderSize(g_rw, g_rh);
    }

    // Reset ALL client-side host-residency + dedup state so a freshly re-initialised host is fully
    // re-fed. Called on a seam RE-init (device re-creation, e.g. an in-session resolution change:
    // MW releases the device and calls CreateDevice again, re-running DistantLand::init → lazyInit).
    // The host's init() tears down its prior instance (fresh geometry arena + EMPTY bindless texture
    // table), so every stale client map would otherwise make the client believe old slots are still
    // resident and never re-ship. Pure client-side state (maps/ints/pending blobs); no D3D/VK release.
    //
    // NOTE (2026-07-22): an in-session RESOLUTION change (MW re-creating its device) is NOT supported
    // while the D3D9Ex takeover is live — see tasks/forge-device-recreation.md. g_spikeForceDefaultPool
    // translates MW's MANAGED allocations to DEFAULT, and MW's engine assumes MANAGED survives a device
    // change, so its own resources (UI textures included) are left dangling and it crashes in unrelated
    // subsystems. This reset is kept because it is correct and needed for the paths that DO re-init,
    // but it cannot make a full device re-creation safe on its own.
    void resetResidencyForReinit() {
        {
            std::lock_guard<std::mutex> lk(g_texResidencyMx);
            g_texSlot.clear();
            g_slotName.clear();
            g_slotLastUsed.clear();
            g_nextTexSlot = 1;                 // 0 = host default white
            ++g_texEpoch;                      // invalidate every cached ResolvedTex fast-path (name+epoch)
            g_texPendingBlob.clear();          // drop tex uploads staged for the OLD host
            g_texPendingCount = 0;
        }
        g_keySlot.clear();
        g_uploadedRev.clear();
        g_nextSlot = 0;
        g_pendingBlob.clear();                 // drop geom staged for the OLD host
        g_pendingParts = 0;
        // Bump the cell epoch so the fresh host evicts its stale shadow/caster records.
        ++g_cellEpoch;
        // The geometry cache is purged by GeometryCache::init, which sees the device change and runs
        // earlier in DistantLand::init — the cache must not outlive its device.
        LOG::logline(">> [seam] device re-creation: reset client residency (epoch=%u)", g_cellEpoch);
    }

    // Seam bring-up (called from RenderProcess::init, under the loading bar / live by menu).
    void lazyInit(IDirect3DDevice9* device) {
        // Seam already up ⇒ this is a RE-init on a re-created device (resolution/mode change). Drop all
        // stale residency BEFORE the fresh renderInit so the new host gets a complete re-feed.
        if (g_initOk) {
            g_initOk = false;                  // a failed re-init must report the seam down, not stale-up
            resetResidencyForReinit();
        }
        if (FAILED(device->QueryInterface(__uuidof(ID3D9VkInteropDevice), (void**)&g_vki)) || !g_vki) {
            LOG::logline("!! [seam] main device is not DXVK (no ID3D9VkInteropDevice) — seam disabled");
            return;
        }
        g_vki->GetVulkanHandles(&g_inst, &g_phys, &g_dev);
        uint32_t queueIndex = 0;
        g_vki->GetSubmissionQueue(&g_queue, &queueIndex, &g_qFamily);
        if (!g_inst || !g_phys || !g_dev || !g_queue) {
            LOG::logline("!! [seam] DXVK returned null Vulkan handles — seam disabled");
            releaseAll();
            return;
        }
        if (!loadVulkan()) {
            releaseAll();
            return;
        }

        // Size the ALLOCATION to ceiling render-scale x backbuffer (the shared RT + imported
        // VkImage + g_mainTex are all created at this size). The scene renders into a g_rw x g_rh
        // sub-rect for live supersampling; the composite stretches it to the g_bbW x g_bbH
        // backbuffer. At the default scale 1.0 the render size equals the backbuffer.
        {
            IDirect3DSurface9* bb = nullptr;
            if (SUCCEEDED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &bb)) && bb) {
                D3DSURFACE_DESC sd = {};
                if (SUCCEEDED(bb->GetDesc(&sd)) && sd.Width && sd.Height) {
                    g_bbW = sd.Width;
                    g_bbH = sd.Height;
                }
                bb->Release();
            }
            g_w = (UINT)(g_bbW * kMaxRenderScale + 0.5f);
            g_h = (UINT)(g_bbH * kMaxRenderScale + 0.5f);
            recomputeRenderSize();   // sets g_rw/g_rh + stamps the host render size (g_client is live here)
            LOG::logline(">> [seam] backbuffer %ux%u — alloc %ux%u (ceiling %.2fx), render %ux%u (scale %.2fx)",
                         g_bbW, g_bbH, g_w, g_h, kMaxRenderScale, g_rw, g_rh, g_renderScale);
        }

        // Host brings up Forge + creates the shared RT; returns the NT handle already
        // duplicated into THIS process.
        HANDLE hostHandle = nullptr;
        // MSAA: Configuration.AALevel is the D3DMULTISAMPLE value (0/2/4/8); map 0 -> 1 sample.
        const std::uint32_t sampleCount = Configuration.AALevel > 0 ? (std::uint32_t)Configuration.AALevel : 1u;
        // AF: Configuration.AnisoLevel (0 = off, else max anisotropy) — host sampler (Phase 2).
        const std::uint32_t anisoLevel = (std::uint32_t)Configuration.AnisoLevel;
        if (!g_client->renderInitBlocking(g_w, g_h, sampleCount, anisoLevel, nullptr, nullptr, &hostHandle) || hostHandle == nullptr) {
            LOG::logline("!! [seam] renderInit RPC failed or no shared handle; seam disabled");
            releaseAll();
            return;
        }
        g_hostHandle = hostHandle;
        LOG::logline(">> [seam] host shared-RT NT handle (this process) = %p", hostHandle);

        // Cross-vendor: pick the first external handle type the GPU can import.
        const VkExternalMemoryHandleTypeFlagBits candidates[] = {
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT,
        };
        bool picked = false;
        for (VkExternalMemoryHandleTypeFlagBits c : candidates) {
            if (handleTypeImportable(c)) { g_htype = c; picked = true; break; }
        }
        if (!picked) {
            LOG::logline("!! [seam] GPU reports no importable D3D12/D3D11 external image type — seam disabled");
            releaseAll();
            return;
        }
        LOG::logline(">> [seam] importing host RT as external handle type %d", (int)g_htype);

        if (!importHostImage()) {
            releaseAll();
            return;
        }
        if (!createDstTexture()) {
            releaseAll();
            return;
        }
        if (!createCopyInfra()) {
            LOG::logline("!! [seam] failed to create Vulkan copy infrastructure — seam disabled");
            releaseAll();
            return;
        }

        g_initOk = true;
        LOG::logline(">> [seam] DXVK Vulkan-interop seam ready (%ux%u). F11 toggles the composite.", g_w, g_h);

        // M1b/M1c: bring up the geometry upload + per-frame draw-list channels.
        initSceneVecs();

        // NOTE: do NOT walk the geometry cache here. lazyInit runs inside DistantLand::init during MW's
        // device RE-CREATION, while the engine is mid-teardown — dereferencing cached NiTriShape
        // pointers at that point walks memory MW is in the middle of freeing (observed: MW's own
        // "Menu Error: Memory pointer corrupted" dialog on an exterior resolution change). The world
        // is re-captured later, by the warm-up's full walks, once MW is back in a stable state.
    }

    // Copy the imported host RT -> the DXVK D3D9 texture's VkImage, on DXVK's own queue.
    // Blocking: CPU-waits the copy so the subsequent StretchRect reads finished pixels.
    bool copyHostRtToDst() {
        // Spike attribution (rare 12ms "Forge RT copy" with host already finished): the outer
        // zone can't say WHICH of the three main-thread blockers stalled — the DXVK flush (drains
        // MW's whole pending D3D9 batch), the submit-queue lock (contends DXVK's submit thread),
        // or the fence wait (our copy drains BEHIND MW's already-queued GPU work). Wall-time each
        // phase; on a spike, log the split so the next occurrence names its own cause.
        const double tc0 = nowMs();
        {
            // Flush any DXVK rendering that touches the dst image before we use its queue.
            MGE_ZoneScopedN("RTcopy: DXVK flush");
            g_vki->FlushRenderingCommands();
        }
        const double tcFlush = nowMs();

        vk.ResetCommandBuffer(g_cmd, 0);
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk.BeginCommandBuffer(g_cmd, &bi);

        const VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        // dst: DXVK resting layout -> TRANSFER_DST
        VkImageMemoryBarrier toDst = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        toDst.srcAccessMask       = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        toDst.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        toDst.oldLayout           = g_dstLayout;
        toDst.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.image               = g_dstImg;
        toDst.subresourceRange    = range;
        vk.CmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &toDst);

        VkImageCopy region = {};
        region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        // Copy only the current render sub-rect (top-left g_rw x g_rh of the g_w x g_h allocation);
        // the composite samples exactly this region. Outside it the host RT is the (transparent)
        // clear and is never sampled.
        region.extent         = { g_rw, g_rh, 1 };
        vk.CmdCopyImage(g_cmd, g_importImg, VK_IMAGE_LAYOUT_GENERAL,
                        g_dstImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        // dst: TRANSFER_DST -> DXVK resting layout (so DXVK's tracking stays valid)
        VkImageMemoryBarrier toRest = toDst;
        toRest.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toRest.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        toRest.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toRest.newLayout     = g_dstLayout;
        vk.CmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &toRest);

        vk.EndCommandBuffer(g_cmd);

        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &g_cmd;

        const double tcRecord = nowMs();
        double tcLocked = tcRecord, tcSubmit = tcRecord;
        // [experimental, uncommitted] Hold DXVK's submission-queue lock ONLY for the submit —
        // NOT across the fence wait. The fence is ours; waiting on it touches no DXVK queue state.
        // Hygiene win (the old code blocked DXVK's submit thread for our whole GPU copy wait) but
        // did NOT move the lock= spike ([rtcopy] still lock-dominated post-change), so it stays
        // out of the commit pending a real-play A/B.
        g_vki->LockSubmissionQueue();
        tcLocked = nowMs();                       // time spent waiting on DXVK's submit lock
        VkResult r = vk.QueueSubmit(g_queue, 1, &si, g_fence);
        g_vki->ReleaseSubmissionQueue();
        tcSubmit = nowMs();
        if (r == VK_SUCCESS) {
            MGE_ZoneScopedN("RTcopy: fence wait");   // unlocked — our copy's GPU completion only
            vk.WaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX);
            vk.ResetFences(g_dev, 1, &g_fence);
        }
        const double tcEnd = nowMs();

        const double total = tcEnd - tc0;
        MGE_TracyPlot("Forge RTcopy flush ms", tcFlush - tc0);
        MGE_TracyPlot("Forge RTcopy wait ms",  tcEnd - tcSubmit);
        // Only rare spikes log — the steady RT copy is ~0.2ms, so >3ms means a real stall to
        // attribute (flush / lock / submit / fence-wait). Rare by construction ⇒ no hot-path spam.
        if (total > 3.0) {
            LOG::logline(">> [rtcopy] spike total=%.2fms | flush=%.2f record=%.2f lock=%.2f submit=%.2f wait=%.2f",
                         total, tcFlush - tc0, tcRecord - tcFlush, tcLocked - tcRecord,
                         tcSubmit - tcLocked, tcEnd - tcSubmit);
        }
        return r == VK_SUCCESS;
    }

    // Allocate the persistent shared vecs: geometry upload (8 chunks) + per-frame draw
    // list (4 chunks). Both chunk-element typed to dodge the Vec uint32 reservation trap.
    void initSceneVecs() {
        auto gv = g_client->allocVecBlocking<IPC::GeomChunk>(
            IPC::kGeomChunks, IPC::kGeomChunks, IPC::kGeomChunks);
        if (!gv) {
            LOG::logline("!! [seam] geometry upload vec alloc failed — capture disabled");
            return;
        }
        g_geomVec.emplace(std::move(*gv));

        auto dv = g_client->allocVecBlocking<IPC::GeomChunk>(kDrawChunks, kDrawChunks, kDrawChunks);
        if (!dv) {
            LOG::logline("!! [seam] draw-list vec alloc failed — scene path disabled (triangle only)");
        } else {
            g_drawVec.emplace(std::move(*dv));
        }

        // Skinned draw list rides its own 4-chunk vec. The blob is [SkinnedDrawWire]
        // [palette]* — bounded at the host's 256-part cap (256 * (12 + 32*64) ≈ 527KB),
        // well under 4MB.
        auto sv = g_client->allocVecBlocking<IPC::GeomChunk>(kDrawChunks, kDrawChunks, kDrawChunks);
        if (!sv) {
            LOG::logline("!! [seam] skinned draw-list vec alloc failed — skinned path disabled");
        } else {
            g_skinnedVec.emplace(std::move(*sv));
        }

        // Multi-map draw list (Tier 4) rides its own 1-chunk vec. MultiMapDrawWire[] bounded at
        // the host's 256-part cap (256 * 132B ≈ 34KB), well under 1MB.
        auto mm = g_client->allocVecBlocking<IPC::GeomChunk>(1, 1, 1);
        if (!mm) {
            LOG::logline("!! [seam] multi-map draw-list vec alloc failed — multi-map path disabled");
        } else {
            g_multiMapVec.emplace(std::move(*mm));
        }

        // Point-light list (Tier 3a) rides its own 1-chunk vec — kMaxPointLights * 48B ≈ 6KB,
        // far under 1MB. PointLightWire[] world-space, rebuilt each frame from the SceneGraph snapshot.
        auto lv = g_client->allocVecBlocking<IPC::GeomChunk>(1, 1, 1);
        if (!lv) {
            LOG::logline("!! [seam] light vec alloc failed — point lights disabled");
        } else {
            g_lightVec.emplace(std::move(*lv));
        }

        // Sky draw list (SK1) rides its own 1-chunk vec — kMaxSkyDraws * 88B ≈ 5.6KB, far under 1MB.
        // SkyDrawWire[] alpha-blended sky shapes, rebuilt each frame from the cache's isSky entries.
        auto kv = g_client->allocVecBlocking<IPC::GeomChunk>(1, 1, 1);
        if (!kv) {
            LOG::logline("!! [seam] sky draw-list vec alloc failed — Forge sky (SK1) disabled");
        } else {
            g_skyVec.emplace(std::move(*kv));
        }

        // Sorted-alpha draw list (AT1) rides its own 1-chunk vec — kMaxAlphaDraws * 128B = 128KB,
        // well under 1MB. AlphaDrawWire[] back-to-front sorted, rebuilt each frame from the
        // blendEnable cache entries the opaque lists skip.
        auto av = g_client->allocVecBlocking<IPC::GeomChunk>(1, 1, 1);
        if (!av) {
            LOG::logline("!! [seam] alpha draw-list vec alloc failed — Forge alpha (AT1) disabled");
        } else {
            g_alphaVec.emplace(std::move(*av));
        }

        // FP draw lists (FP1a) ride two 1-chunk vecs — the arm scene is ~10-30 parts:
        // rigid DrawItemWire[] a few KB; skinned [SkinnedDrawWire][palette]* bounded far
        // under the main skinned list (same wire formats, drawn by the host FP pass).
        auto fdv = g_client->allocVecBlocking<IPC::GeomChunk>(1, 1, 1);
        if (!fdv) {
            LOG::logline("!! [seam] FP draw-list vec alloc failed — Forge FP pass (FP1a) disabled");
        } else {
            g_fpDrawVec.emplace(std::move(*fdv));
        }
        auto fsv = g_client->allocVecBlocking<IPC::GeomChunk>(1, 1, 1);
        if (!fsv) {
            LOG::logline("!! [seam] FP skinned draw-list vec alloc failed — Forge FP pass (FP1a) disabled");
        } else {
            g_fpSkinnedVec.emplace(std::move(*fsv));
        }
        // FP1c: blended FP parts (torch flame, enchant glow) ride their own tiny
        // AlphaDrawWire[] vec — a handful of shapes, same wire format as the world list.
        auto fav = g_client->allocVecBlocking<IPC::GeomChunk>(1, 1, 1);
        if (!fav) {
            LOG::logline("!! [seam] FP alpha draw-list vec alloc failed — FP alpha (FP1c) disabled");
        } else {
            g_fpAlphaVec.emplace(std::move(*fav));
        }

        // AT3 captured-alpha geometry vec: one 1-chunk (1MB) vec carrying [captured verts][captured
        // indices] — 20000 verts (720KB) + 60000 uint16 (120KB) = 840KB fits one chunk (see
        // kMaxCapturedAlpha* in geomwire.h). A 1-chunk vec sidesteps the IPC uint32-reservation hazard.
        auto cv = g_client->allocVecBlocking<IPC::GeomChunk>(1, 1, 1);
        if (!cv) {
            LOG::logline("!! [seam] captured-alpha vec alloc failed — Forge captured alpha (AT3) disabled");
        } else {
            g_capturedVec.emplace(std::move(*cv));
        }

        // Texture upload vec: kTexChunks (32MB window) — larger than geometry because a single DDS
        // must fit one window (oversize textures are dropped to white in resolveTextureSlot).
        auto tv = g_client->allocVecBlocking<IPC::GeomChunk>(
            IPC::kTexChunks, IPC::kTexChunks, IPC::kTexChunks);
        if (!tv) {
            LOG::logline("!! [seam] texture upload vec alloc failed — textures disabled (white)");
        } else {
            g_texVec.emplace(std::move(*tv));
        }
        LOG::logline(">> [seam] scene vecs ready (geom %u, draw %u, skinned %u, multimap 1, light 1, sky 1, alpha 1, tex %u chunks)",
                     IPC::kGeomChunks, kDrawChunks, kDrawChunks, IPC::kTexChunks);
    }

    // Normalize an NI SourceTexture::fileName to the bare name BSA::loadFileBytes expects
    // (it re-adds "textures\"). Lower-cased (BSA hashing + case-insensitive loose lookup),
    // backslash-separated, with any leading "data files\" then "textures\" stripped.
    static std::string normalizeTextureName(const char* fileName) {
        if (!fileName) {
            return std::string();
        }
        std::string s(fileName);
        for (auto& c : s) {
            c = (char)std::tolower((unsigned char)c);
            if (c == '/') { c = '\\'; }
        }
        if (s.size() >= 11 && s.compare(0, 11, "data files\\") == 0) { s.erase(0, 11); }
        if (s.size() >= 9  && s.compare(0, 9,  "textures\\")  == 0) { s.erase(0, 9); }
        return s;
    }

    // Map a texture name to its bindless slot, loading + queueing its DDS on first sight.
    // Misses / oversize / residency-full → slot 0 (host default white), cached so we don't retry.
    std::uint32_t resolveTextureSlot(const char* textureName) {
        if (!textureName || !*textureName || !g_texVec) {
            return 0;
        }
        std::string name = normalizeTextureName(textureName);
        if (name.empty()) {
            return 0;
        }
        // Serialise the whole residency mutation against the other thread (see g_texResidencyMx).
        // The BSA disk load below runs under the lock — bounded (first-sight only), no IPC involved.
        std::lock_guard<std::mutex> lk(g_texResidencyMx);
        const std::uint32_t cap = IPC::kMaxTextures - IPC::kDlReserve;   // client range [1, cap)
        if (g_slotName.size() != IPC::kMaxTextures) {
            g_slotName.assign(IPC::kMaxTextures, std::string());
            g_slotLastUsed.assign(IPC::kMaxTextures, 0u);
        }
        auto it = g_texSlot.find(name);
        if (it != g_texSlot.end()) {
            if (it->second != 0) { g_slotLastUsed[it->second] = g_frame; }   // refresh LRU age
            return it->second;   // already resolved (slot or cached-miss 0)
        }

        void* data = nullptr;
        unsigned size = 0;
        // skipDistantStatics=true: the Forge near path must NOT pick the distantland\statics
        // downscaled-LOD copies (they blur near geometry) — resolve loose Data Files -> BSA.
        if (!BSA::loadFileBytes(name.c_str(), &data, &size, true) || !data || size == 0) {
            static int misses = 0;
            if (misses < 20) { LOG::logline("!! [tex] not found: %s (white)", name.c_str()); ++misses; }
            if (data) { std::free(data); }
            g_texSlot.emplace(name, 0);
            return 0;
        }

        const std::size_t windowBytes = (std::size_t)IPC::kTexWindowBytes;
        if (sizeof(IPC::TexUploadWire) + size > windowBytes) {
            LOG::logline("!! [tex] %s too large (%u bytes) for %zu window — white", name.c_str(), size, windowBytes);
            std::free(data);
            g_texSlot.emplace(name, 0);
            return 0;
        }

        // Assign a slot: grow while the range has room, else recycle the least-recently-used slot.
        std::uint32_t slot;
        if (g_nextTexSlot < cap) {
            slot = g_nextTexSlot++;
        } else {
            std::uint32_t lru = 1, best = 0xFFFFFFFFu;
            for (std::uint32_t s = 1; s < cap; ++s) {
                if (g_slotLastUsed[s] < best) { best = g_slotLastUsed[s]; lru = s; }
            }
            if (best == g_frame) {
                static bool warned = false;   // working set > capacity this frame: unavoidable thrash
                if (!warned) { LOG::logline("!! [tex] working set exceeds %u client slots — thrashing (white)", cap); warned = true; }
            }
            g_texSlot.erase(g_slotName[lru]);   // evict the recycled name
            slot = lru;
            // The recycled slot now means a different texture: every SlotInfo-cached
            // slot value is suspect. Epoch bump forces per-key re-resolve (one-off).
            ++g_texEpoch;
        }
        g_texSlot[name] = slot;
        g_slotName[slot] = name;
        g_slotLastUsed[slot] = g_frame;
        IPC::TexUploadWire hdr{ slot, size };
        const std::size_t at = g_texPendingBlob.size();
        g_texPendingBlob.resize(at + sizeof(hdr) + size);
        std::memcpy(g_texPendingBlob.data() + at, &hdr, sizeof(hdr));
        std::memcpy(g_texPendingBlob.data() + at + sizeof(hdr), data, size);
        std::free(data);
        ++g_texPendingCount;
        return slot;
    }

    // Ship queued texture uploads to the host in window-sized batches on whole-entry
    // boundaries (each entry is guaranteed <= window by resolveTextureSlot). Blocking RPCs.
    void flushTextures() {
        // Take ownership of the pending blob under the lock, then batch + RPC on the LOCAL copy
        // UNLOCKED — main's captureAlphaDraw keeps appending to a fresh g_texPendingBlob, and we
        // never hold g_texResidencyMx across a blocking RPC (that would be a new 60s-freeze class).
        std::vector<std::uint8_t> blob;
        std::uint32_t pendingCount = 0;
        {
            std::lock_guard<std::mutex> lk(g_texResidencyMx);
            if (!g_texVec || g_texPendingBlob.empty() || g_texPendingCount == 0) {
                return;
            }
            blob.swap(g_texPendingBlob);      // main now appends to an empty vector, race-free
            pendingCount = g_texPendingCount;
            g_texPendingCount = 0;            // fresh count for whatever main appends from here
        }
        // A0 sub-buckets (see flushGeometry): assign vs RPC wait, one line per >=1ms flush.
        const double tFlush0 = nowMs();
        double assignMs = 0.0, rpcMs = 0.0;
        const std::size_t windowBytes = (std::size_t)IPC::kTexWindowBytes;
        const std::uint8_t* base = blob.data();
        const std::size_t total = blob.size();
        std::size_t off = 0;
        std::uint32_t entriesFlushed = 0;
        while (off < total) {
            std::size_t batchEnd = off;
            std::uint32_t batchCount = 0;
            while (batchEnd < total) {
                IPC::TexUploadWire h;
                std::memcpy(&h, base + batchEnd, sizeof(h));
                const std::size_t entryBytes = sizeof(h) + h.byteLen;
                if ((batchEnd - off) + entryBytes > windowBytes) { break; }   // window full
                batchEnd += entryBytes;
                ++batchCount;
            }
            if (batchCount == 0) { break; }   // safety (each entry <= window)
            const std::uint32_t bytes = (std::uint32_t)(batchEnd - off);
            std::uint32_t uploaded = 0;
            const double tAssign0 = nowMs();
            const bool assigned = g_texVec->assign_bytes(base + off, bytes);
            assignMs += nowMs() - tAssign0;
            if (assigned) {
                const double tRpc0 = nowMs();
                MGE_ZoneScopedN("Forge texUpload RPC");
                g_client->texUploadBlocking(g_texVec->id(), batchCount, bytes, &uploaded);
                rpcMs += nowMs() - tRpc0;
            }
            if (uploaded == 0xFFFFFFFFu) {
                // Host opaque path not built yet (first scene frame) — keep the UNFLUSHED remainder
                // for next frame. Put it BACK at the FRONT of g_texPendingBlob (main may have
                // appended after the swap) and restore its share of the count. Slots are already
                // assigned; draws show white until then.
                std::lock_guard<std::mutex> lk(g_texResidencyMx);
                g_texPendingBlob.insert(g_texPendingBlob.begin(), blob.begin() + off, blob.end());
                g_texPendingCount += (pendingCount - entriesFlushed);
                return;
            }
            entriesFlushed += batchCount;
            off = batchEnd;
        }
        // Fully consumed the local blob. Do NOT touch g_texPendingBlob/Count here — they now hold
        // only what main appended during the unlocked RPC, which must be kept for the next flush.
        const double flushMs = nowMs() - tFlush0;
        if (flushMs >= 1.0) {
            LOG::logline("-- [texflush] %.2fms tex=%u bytes=%uKB assign=%.2f rpc=%.2f",
                         flushMs, entriesFlushed, (unsigned)(total >> 10), assignMs, rpcMs);
        }
    }

    // Turn this frame's cache evictions into host slot-release records. An object that LEFT the
    // world within a cell (picked up into inventory, despawned, MWSE-disabled) is dropped by the
    // geometry cache's eviction sweep; without a signal the host keeps its last transform as a
    // persistent SHADOW CASTER and its shadow ghosts in place. Emit a header-only GeomPartWire with
    // kGeomFlagRelease for each evicted key that owns a host slot, then prune the slot/upload maps
    // (a re-drop is a new NiTriShape → new key → fresh slot; a recycled key must re-upload, not
    // dedup-skip). Cross-cell removals are handled separately by the cell-epoch eviction (C1).
    // Resolve evicted keys to their (old) host slots and QUEUE them for release. Key->slot MUST be
    // resolved HERE, not at ship time: on a cell purge the caller runs this BEFORE
    // buildGeometryDrawLists re-captures the same recycled addresses into NEW slots, so a late
    // resolve would read the new slot and free the freshly-uploaded object (see the 3257 call site).
    // The queue is drained a bounded number per flush (appendBoundedReleaseRecords) so a mass
    // eviction — orphaned far cells under the walk-free rule, or a whole-cache purgeAll — can't flood
    // the blocking geometry RPC at present time (measured: 517 release parts = 5.5ms, backlog blew
    // the 32MB staging cap → the client freeze). Erasing g_keySlot here is what lets a returning
    // object re-capture into a fresh slot immediately; the old slot's release drains independently
    // and safely (monotonic slots, never reused).
    void drainReleasedSlots() {
        static std::vector<std::uint32_t> evicted;
        MGE::GeometryCache::takeEvictedKeys(evicted);
        for (std::uint32_t key : evicted) {
            auto ks = g_keySlot.find(key);
            if (ks == g_keySlot.end()) {
                continue;   // never had a host slot (culled-only / never shipped)
            }
            g_pendingReleaseSlots.push_back(ks->second.slot);
            g_keySlot.erase(ks);
            g_uploadedRev.erase(key);
        }
    }

    // Append up to kMaxReleasesPerFlush queued slot-releases to g_pendingBlob as header-only
    // sentinels. Called by flushGeometry each present; bounding the per-flush count keeps the release
    // contribution to any one blocking RPC small while the backlog drains over subsequent frames.
    // Queue order is irrelevant (monotonic slots, no reuse), so a plain front-drain is fine.
    void appendBoundedReleaseRecords() {
        if (g_pendingReleaseSlots.empty()) return;
        const std::size_t n = std::min<std::size_t>(g_pendingReleaseSlots.size(), kMaxReleasesPerFlush);
        for (std::size_t i = 0; i < n; ++i) {
            IPC::GeomPartWire hdr = {};
            hdr.slot  = g_pendingReleaseSlots[i];
            hdr.flags = IPC::kGeomFlagRelease;   // vertexCount = indexCount = 0 → header-only sentinel
            const std::size_t at = g_pendingBlob.size();
            g_pendingBlob.resize(at + sizeof(hdr));
            memcpy(g_pendingBlob.data() + at, &hdr, sizeof(hdr));
            ++g_pendingParts;
        }
        g_pendingReleaseSlots.erase(g_pendingReleaseSlots.begin(),
                                    g_pendingReleaseSlots.begin() + n);
        // Backlog observability, throttled: watch a mass eviction drain instead of freeze.
        static std::uint32_t s_relLogThrottle = 0;
        if (!g_pendingReleaseSlots.empty() && (++s_relLogThrottle % 30) == 0) {
            LOG::logline("-- [release] draining: shipped %zu this flush, %zu still queued",
                         n, g_pendingReleaseSlots.size());
        }
    }

    // Drain g_pendingBlob to the host in window-sized whole-part chunks. Parts are
    // self-describing (GeomPartWire carries vert/index counts), so we walk part
    // boundaries to never split a part across a chunk. Blocking RPCs — called at
    // present time. Static cost: each part ships once per (key,revision).
    void flushGeometry() {
        // A0 geom-flush sub-buckets: the [hb]/[spike] geom= bucket is this whole
        // function; split it into drain (eviction records) / assign (shared-vec memcpy)
        // / rpc (geomUploadBlocking wait — the host-side turnaround) so an anomalous
        // flush names its component. One [geomflush] line per >=1ms flush.
        const double tFlush0 = nowMs();
        double drainMs = 0.0, assignMs = 0.0, rpcMs = 0.0;
        std::uint32_t chunkCount = 0;
        {
            const double t0 = nowMs();
            drainReleasedSlots();          // resolve evictions → release queue (cheap, unbounded)
            appendBoundedReleaseRecords(); // ship <=kMaxReleasesPerFlush sentinels before the empty check
            drainMs = nowMs() - t0;
        }
        if (!g_geomVec || g_pendingBlob.empty() || g_pendingParts == 0) {
            return;
        }
        // Geometry now ships on the client's DEDICATED geometry IPC channel
        // (geomUploadBlocking → its own Parameters + events), so it no longer contends
        // with the one-at-a-time cull/scene RPCs on the main channel. That contention used
        // to starve this flush at present time and leave exteriors black (the draw list
        // referenced slots whose geometry never shipped). The old isRpcPending() bail —
        // and its clobber hazard — is gone because there is no shared Parameters union to
        // interleave. The host services the geometry channel on its single thread (WFMO),
        // so this upload just waits its turn behind any in-flight cull instead of being
        // skipped, and can never race renderScene.
        const std::uint8_t* const data = g_pendingBlob.data();
        const std::uint32_t total = static_cast<std::uint32_t>(g_pendingBlob.size());

        std::uint32_t off = 0;
        std::uint32_t shippedParts = 0;
        bool corrupt = false;
        // Consecutive failures on the LEADING chunk, surviving across flush calls. Bounded
        // retry: a transient failure retries next flush, but a chunk the host persistently
        // won't take is dropped after kFlushFailLimit attempts — an unbounded retry loop
        // here re-sends 8MB per capture and grows the blob to the load-time bad_alloc
        // (seen live: host miscounted release sentinels → 3376 re-sends of one chunk).
        constexpr std::uint32_t kFlushFailLimit = 8;
        static std::uint32_t s_flushFailStreak = 0;
        while (off < total) {
            std::uint32_t chunkBytes = 0;
            std::uint32_t chunkParts = 0;
            std::uint32_t cursor = off;
            while (cursor < total) {
                IPC::GeomPartWire hdr;
                memcpy(&hdr, data + cursor, sizeof(hdr));
                const std::size_t vStride = (hdr.flags & IPC::kGeomFlagSkinned)
                    ? sizeof(IPC::SkinnedVertexWire)
                    : (hdr.flags & IPC::kGeomFlagMultiMap)
                        ? sizeof(IPC::GeomVertexWireMM) : sizeof(IPC::GeomVertexWire);
                const std::uint32_t partSize = static_cast<std::uint32_t>(
                    sizeof(IPC::GeomPartWire)
                    + (std::uint64_t)hdr.vertexCount * vStride
                    + (std::uint64_t)hdr.indexCount * sizeof(std::uint16_t)
                    + hdr.uvAnimBytes);   // NiUVController key track (0 for most parts)
                if (chunkBytes != 0 && chunkBytes + partSize > kGeomChunkCap) {
                    break;  // close this chunk on a part boundary
                }
                chunkBytes += partSize;
                ++chunkParts;
                cursor += partSize;
            }
            if (chunkParts == 0) {
                // Single part exceeds the window — impossible for real content (uint16 indices
                // cap a part at ~4.3MB < 8MB window), so this means a corrupt header. The blob
                // is unparseable from here — drop it all, loudly.
                LOG::logline("!! [seam] geometry part exceeds chunk cap at offset %u/%u — dropping %u remaining bytes",
                             off, total, total - off);
                corrupt = true;
                break;
            }

            const double tAssign0 = nowMs();
            const bool assigned = g_geomVec->assign_bytes(data + off, chunkBytes);
            assignMs += nowMs() - tAssign0;
            if (!assigned) {
                if (++s_flushFailStreak >= kFlushFailLimit) {
                    LOG::logline("!! [seam] geometry chunk assign_bytes failed %u times — DROPPING chunk (%u bytes, %u parts)",
                                 s_flushFailStreak, chunkBytes, chunkParts);
                    s_flushFailStreak = 0;
                    shippedParts += chunkParts;   // gone either way — keep the parts count in sync
                    off = cursor;
                    continue;
                }
                LOG::logline("!! [seam] geometry chunk assign_bytes failed (%u bytes) — keeping %u bytes for retry",
                             chunkBytes, total - off);
                break;
            }
            // Blocking upload. With the dynamic-VB ring the host side is now a cheap memcpy
            // (animated re-uploads) or a one-time static build, so blocking no longer carries
            // the old recreate+fence cost — and it keeps the present pipeline simple (async
            // kickoff was reverted: it cost the framerate cap without helping the steady state).
            std::uint32_t uploaded = 0;
            const double tRpc0 = nowMs();
            bool rpcOk;
            {
                MGE_ZoneScopedN("Forge geomUpload RPC");
                rpcOk = g_client->geomUploadBlocking(g_geomVec->id(), chunkParts, chunkBytes, &uploaded);
            }
            rpcMs += nowMs() - tRpc0;
            ++chunkCount;
            if (!rpcOk) {
                // Transient (async-window refusal / lost RPC): keep this chunk onward and let
                // the next flush retry — slots are idempotent, a re-send just rebuilds. The
                // old clear-anyway lost the geometry FOREVER (revs were already stamped).
                // Bounded (kFlushFailLimit): a chunk the host persistently rejects gets
                // dropped, not re-sent every capture until the blob re-creates the bad_alloc.
                if (++s_flushFailStreak >= kFlushFailLimit) {
                    LOG::logline("!! [seam] geomUpload built %u/%u parts %u times running — DROPPING chunk (%u bytes)",
                                 uploaded, chunkParts, s_flushFailStreak, chunkBytes);
                    s_flushFailStreak = 0;
                    shippedParts += chunkParts;   // gone either way — keep the parts count in sync
                    off = cursor;
                    continue;
                }
                LOG::logline("!! [seam] geomUpload RPC built %u/%u parts — keeping %u bytes for retry",
                             uploaded, chunkParts, total - off);
                break;
            }
            s_flushFailStreak = 0;
            shippedParts += chunkParts;
            off = cursor;
        }

        if (corrupt || off >= total) {
            g_pendingBlob.clear();
            g_pendingParts = 0;
        } else if (off > 0) {
            g_pendingBlob.erase(g_pendingBlob.begin(), g_pendingBlob.begin() + off);
            g_pendingParts = (shippedParts < g_pendingParts) ? (g_pendingParts - shippedParts) : 0;
        }

        const double flushMs = nowMs() - tFlush0;
        if (flushMs >= 1.0) {
            LOG::logline("-- [geomflush] %.2fms parts=%u bytes=%uKB chunks=%u drain=%.2f assign=%.2f rpc=%.2f",
                         flushMs, shippedParts, total >> 10, chunkCount, drainMs, assignMs, rpcMs);
        }
    }

    // Mid-walk drain (see kPendingFlushBytes): called by the capture functions before they
    // append, so the staging blob's peak stays bounded on cell-load bursts instead of
    // accumulating the whole cell until present time. The walk runs inside the async
    // RenderFrame window (host frame overlaps MW scene 0), where geomUploadBlocking REFUSES
    // — so first close the window early (renderSceneFinish now, composite consumes the
    // stored result). Costs that frame's overlap; load-burst frames only.
    void drainPendingIfFull() {
        if (g_pendingBlob.size() < kPendingFlushBytes) {
            return;
        }
        // Load-burst escape hatch out of the client-private half: this drains to shared memory
        // mid-build, so it needs the same release as the flush below (see gateOnPrevFinish).
        gateOnPrevFinish("drain");
        if (g_kick.rpcPending) {
            g_kick.rpcPending       = false;
            g_kick.rpcEarlyFinished = true;
            g_kick.earlyHostMs      = 0.0;
            g_kick.earlyHostTimings = {};
            g_kick.earlyOk          = g_client->renderSceneFinish(&g_kick.earlyHostMs,
                                                                  &g_kick.earlyHostTimings);
            g_kick.tEarlyFinish     = nowMs();
        }
        LOG::logline("-- [seam] geometry staging at %u KB mid-walk — draining to host",
                     static_cast<unsigned>(g_pendingBlob.size() >> 10));
        flushGeometry();
        // Failsafe: if flushes keep failing (host gone), the retry-kept blob would grow
        // right back to the ~1GB bad_alloc this drain exists to prevent. Cap it loudly.
        constexpr std::size_t kPendingAbandonBytes = 8 * kPendingFlushBytes;   // 256 MB
        if (g_pendingBlob.size() >= kPendingAbandonBytes) {
            LOG::logline("!! [seam] geometry staging still %u KB after drain — host unreachable, ABANDONING blob",
                         static_cast<unsigned>(g_pendingBlob.size() >> 10));
            g_pendingBlob.clear();
            g_pendingBlob.shrink_to_fit();
            g_pendingParts = 0;
        }
    }

    // Gather this frame's visible opaque parts into g_drawScratch as DrawItemWire[]:
    // each is the part's host slot + its current world transform. Source is the cache's
    // current-frame frustum-visible key set (built in renderStage0); only keys we've
    // assigned a host slot (uploaded, non-skinned, non-landscape) are included. Returns
    // the packed item count (0 if nothing to draw / scene path unavailable).
    // Resolve a bindless texture slot through a SlotInfo cache field (see SlotInfo).
    // Fast path = pointer-identity + epoch check, no string normalize/hash. The
    // LRU age refresh matches what resolveTextureSlot's memoized path would do.
    std::uint32_t resolveCachedSlot(const char* name, const char*& namePtr,
                                    std::uint32_t& slotVal, std::uint32_t& slotEpoch) {
        if (name == namePtr && slotEpoch == g_texEpoch) {
            if (slotVal != 0) { g_slotLastUsed[slotVal] = g_frame; }
            return slotVal;
        }
        slotVal = resolveTextureSlot(name);
        namePtr = name;
        slotEpoch = g_texEpoch;
        return slotVal;
    }

    // Emit one STATIC opaque draw (pre-filtered by buildGeometryDrawLists — the
    // per-entry filter rationale lives there). Mirrors the PROVEN D3D9 cache color
    // pass's per-entry packing (drawEntry in rendercachedcolor.cpp) so the Forge
    // draw list draws the same set. Terrain draws too (near worldLandscapeRoot
    // patches; flat-shaded geometry).
    // TEMP normal-flip diagnostic (remove after triage): one-shot per (name,tag) dump of the world
    // 3x3 determinant sign. Negative det = MIRRORED placement — mul((float3x3)world, N) then flips
    // the normal relative to the surface (green<->purple in F12 mode 8), which mis-lights the shape.
    // Called from BOTH the opaque (STATIC) and sorted-alpha (ALPHA) emit so a wall's sign can be
    // compared directly to a tapestry's.
    void diagWorldDet(const char* tag, const char* name, const float* w) {
        static std::unordered_set<std::string> s_seen;
        std::string key = std::string(tag) + "|" + (name ? name : "(null)");
        if (!s_seen.insert(key).second || s_seen.size() > 128) return;
        const float det = w[0] * (w[5]*w[10] - w[6]*w[9])
                        - w[1] * (w[4]*w[10] - w[6]*w[8])
                        + w[2] * (w[4]*w[9]  - w[5]*w[8]);
        LOG::logline(">> [norm-diag] %s %s det=%.3f %s", tag, name ? name : "(null)",
                     det, det < 0.0f ? "MIRRORED" : "normal");
    }

    // FP1a: `dst` defaults to the main-pass scratch; buildFPDrawLists redirects the
    // identical packing into the FP scratch (same wire format, different host pass).
    void emitStaticDraw(SlotInfo& si, const MGE::GeometryCache::CachedGeometry& e,
                        std::uint32_t& count, std::vector<std::uint8_t>& dst = g_drawScratch) {
            IPC::DrawItemWire item;
            item.slot = si.slot;
            diagWorldDet("STATIC", e.textureName, e.worldTransformD3D);
            // Textureless visual (e.textureName null) → slot 0 = host default white; resolveCachedSlot
            // would std::string(nullptr) on the name lookup, so short-circuit it.
            item.texIndex = e.textureName
                ? resolveCachedSlot(e.textureName, si.baseNamePtr, si.baseSlot, si.baseEpoch) : 0u;
            // Terrain DECAL_1 overlay (second land texture). resolveTextureSlot ships its DDS
            // bytes the same way as the base map. Non-landscape / single-texture draws get 0,
            // which gates the frag's splat off → byte-for-byte unchanged.
            item.overlayTexIndex = (e.isLandscape && e.d3dOverlay && e.overlayTextureName)
                ? resolveCachedSlot(e.overlayTextureName, si.ovNamePtr, si.ovSlot, si.ovEpoch) : 0u;
            item.alphaRef = e.alphaTest ? e.alphaRef : 0.0f;     // alpha-test cutout (0 = no test)
            item.clampMode = e.baseClamp;                        // MW's per-map texture address mode
            // Tier 2b material: ship the captured MaterialProperty colours + the vertex-colour
            // routing, replicating buildCacheReflectionState/buildCacheMainState EXACTLY. useVCol
            // = mesh has colours AND its VertexColorProperty says to use them (else real material
            // colours get white-washed); when off we send vColSource 0 so the frag uses the
            // constant material (vertexMaterialNone), ignoring the VB's colour slot.
            item.matDiffuse[0]  = e.matDiffuse[0];  item.matDiffuse[1]  = e.matDiffuse[1];  item.matDiffuse[2]  = e.matDiffuse[2];
            item.matAmbient[0]  = e.matAmbient[0];  item.matAmbient[1]  = e.matAmbient[1];  item.matAmbient[2]  = e.matAmbient[2];
            MGE::GeometryCache::emissiveForDraw(e, item.matEmissive);
            item.vColSource = (e.hasVertexColor && e.vColSource != 0) ? e.vColSource : 0u;
            // C4d shadow-caster category: LIVE (NPC/creature parts + held equipment,
            // activators, doors) → the host's dynamic shadow tile, not the cached statics.
            item.casterFlags = (e.isLive ? IPC::kDrawCasterLive : 0u)
                             | (e.animated ? IPC::kDrawCasterAnimated : 0u);
            memcpy(item.world, e.worldTransformD3D, 16 * sizeof(float));
            // CAMERA-RELATIVE rendering: subtract the camera world position from the world
            // translation so vertices reach the shader near the origin. At MW's exterior
            // coordinates (|eye| ~150k) absolute world positions quantise to ~0.01-0.1 units
            // in float32, and the combined viewProj's per-vertex cancellation then produces
            // orientation-dependent stretching. Shifting world + viewProj + lights + eyePos by
            // -eye keeps all vertex math small/precise. (viewProj is built translation-free below.)
            item.world[12] -= DistantLand::eyePos.x;
            item.world[13] -= DistantLand::eyePos.y;
            item.world[14] -= DistantLand::eyePos.z;
            // F12 diagnostic: displace each object by a deterministic per-slot vector. The
            // world matrix is row-major D3DX (translation in m[12..14]); a fixed offset per
            // slot means duplicates of one object (same or different slot) appear as two
            // separated copies. Hash the slot to a pseudo-random ±range.
            if (g_debugMode == 2) {
                const std::uint32_t s = si.slot;
                std::uint32_t h = s * 2654435761u;        // Knuth multiplicative hash
                auto axis = [&](std::uint32_t shift) {
                    std::uint32_t v = (h >> shift) & 0x3FF;   // 10 bits
                    return (static_cast<float>(v) / 1023.0f - 0.5f) * 100.0f;  // ±50 units
                };
                item.world[12] += axis(0);
                item.world[13] += axis(10);
                item.world[14] += axis(20);
            }
            const std::size_t at = dst.size();
            dst.resize(at + sizeof(item));
            memcpy(dst.data() + at, &item, sizeof(item));
            ++count;
    }

    // M-Skinning: emit one visible SKINNED part into g_skinnedScratch as
    // [SkinnedDrawWire][palette] (pre-filtered by buildGeometryDrawLists; skinned
    // keys are EXCLUDED from the static list — stride-44 VB + GPU palette pipeline).
    // There is no per-draw world transform — the bone palette (read fresh from the
    // cache entry each frame, that IS the animation) is world-space.
    void emitSkinnedDraw(SlotInfo& si, const MGE::GeometryCache::CachedGeometry& e,
                         std::uint32_t& count, std::vector<std::uint8_t>& dst = g_skinnedScratch) {
            // Only GPU-skinnable parts: a built skinned VB, bones within the palette cap,
            // and a current bone palette of the expected size.
            if (e.skinnedUnsupported || e.numBones == 0) {
                return;
            }
            if (e.bonePalette.size() < (std::size_t)e.numBones * 16) {
                return;   // palette not yet built this frame
            }

            IPC::SkinnedDrawWire item;
            item.slot     = si.slot;
            item.numBones = e.numBones;
            item.mirror   = e.mirrored ? 1u : 0u;
            item.texIndex = resolveCachedSlot(e.textureName, si.baseNamePtr, si.baseSlot, si.baseEpoch);
            item.alphaRef = e.alphaTest ? e.alphaRef : 0.0f;     // alpha-test cutout (0 = no test)
            item.clampMode = e.baseClamp;                        // MW's per-map texture address mode

            const std::size_t paletteBytes = (std::size_t)e.numBones * 64;  // numBones * 16 floats
            const std::size_t at = dst.size();
            dst.resize(at + sizeof(item) + paletteBytes);
            std::uint8_t* out = dst.data() + at;
            memcpy(out, &item, sizeof(item));                  out += sizeof(item);
            memcpy(out, e.bonePalette.data(), paletteBytes);
            // CAMERA-RELATIVE: the bone palette is world-space; shift each bone matrix's
            // translation by -eye so the skinned vertices land near the origin, consistent
            // with the translation-free viewProj + shifted lights (see buildDrawList).
            {
                float* pal = reinterpret_cast<float*>(out);
                for (std::uint32_t b = 0; b < e.numBones; ++b) {
                    pal[b * 16 + 12] -= DistantLand::eyePos.x;
                    pal[b * 16 + 13] -= DistantLand::eyePos.y;
                    pal[b * 16 + 14] -= DistantLand::eyePos.z;
                }
            }
            ++count;
    }

    // Tier 4 multi-map: emit one visible STATIC multi-map part (dark/detail/glow
    // siblings) into g_multiMapScratch as a MultiMapDrawWire (pre-filtered by
    // buildGeometryDrawLists; multi-map keys are EXCLUDED from the static list —
    // wide stride-60 VB + multimap pipeline). The ORDERED stage list is built here
    // on the CLIENT, replicating rendercachedcolor.cpp::buildCacheStages EXACTLY
    // (present maps pushed with their op + UV set, stable-sorted by texCoordSet
    // ascending — MW assigns the D3D stage index = texCoordSet, not the map slot).
    // Stage textures resolve through resolveTextureSlot directly (multi-map items
    // are rare — a handful per frame — not worth SlotInfo fields for 4 maps).
    void emitMultiMapDraw(SlotInfo& si, const MGE::GeometryCache::CachedGeometry& e,
                          std::uint32_t& count, bool blended = false) {
            // Build the ordered stage list, replicating buildCacheStages. A present map is usable
            // only if the wide VB carries the UV set it samples (uv < uvSetCount) — cacheMapActive.
            // Base is pushed unconditionally (e.d3dTexture); the others gated by cacheMapActive.
            // clamp rides PER STAGE: each NiTexturingProperty::Map has its own address mode, so a
            // glow map can clamp over a base map that wraps. It follows the map through the sort.
            struct Stage { const char* name; std::uint8_t uv; std::uint32_t op; std::uint8_t clamp; };
            Stage st[4];
            int ns = 0;
            st[ns++] = { e.textureName, e.baseUV, IPC::kMMOpBase, e.baseClamp };
            if (e.d3dDark   && e.darkUV   < e.uvSetCount) st[ns++] = { e.darkTextureName,   e.darkUV,   IPC::kMMOpMod,   e.darkClamp };
            if (e.d3dDetail && e.detailUV < e.uvSetCount) st[ns++] = { e.detailTextureName, e.detailUV, IPC::kMMOpMod2X, e.detailClamp };
            if (e.d3dGlow   && e.glowUV   < e.uvSetCount) st[ns++] = { e.glowTextureName,   e.glowUV,   IPC::kMMOpAdd,   e.glowClamp };
            // Stable insertion sort by UV ascending (<=4 stages; keeps slot order on ties).
            for (int a = 1; a < ns; ++a) {
                Stage tmp = st[a];
                int b = a - 1;
                while (b >= 0 && st[b].uv > tmp.uv) { st[b + 1] = st[b]; --b; }
                st[b + 1] = tmp;
            }

            IPC::MultiMapDrawWire item = {};
            item.slot = si.slot;
            memcpy(item.world, e.worldTransformD3D, 16 * sizeof(float));
            // CAMERA-RELATIVE: shift translation by -eye (see buildDrawList).
            item.world[12] -= DistantLand::eyePos.x;
            item.world[13] -= DistantLand::eyePos.y;
            item.world[14] -= DistantLand::eyePos.z;
            item.matDiffuse[0]  = e.matDiffuse[0];  item.matDiffuse[1]  = e.matDiffuse[1];  item.matDiffuse[2]  = e.matDiffuse[2];
            item.matAmbient[0]  = e.matAmbient[0];  item.matAmbient[1]  = e.matAmbient[1];  item.matAmbient[2]  = e.matAmbient[2];
            MGE::GeometryCache::emissiveForDraw(e, item.matEmissive);
            item.vColSource = (e.hasVertexColor && e.vColSource != 0) ? e.vColSource : 0u;
            item.alphaRef   = e.alphaTest ? e.alphaRef : 0.0f;   // base-stage alpha test
            item.stageCount = (std::uint32_t)ns;
            item.drawFlags  = blended ? IPC::kMMDrawFlagBlended : 0u;   // Route C: alpha-stage blend draw
            for (int s = 0; s < ns; ++s) {
                const std::uint32_t tex = resolveTextureSlot(st[s].name);   // bindless slot (0 = white)
                item.stages[s] = IPC::packMMStage(tex, st[s].uv, st[s].op, st[s].clamp);
            }
            const std::size_t at = g_multiMapScratch.size();
            g_multiMapScratch.resize(at + sizeof(item));
            memcpy(g_multiMapScratch.data() + at, &item, sizeof(item));
            ++count;
    }

    // AT1 sorted-alpha: emit one blended STATIC part into g_alphaScratch as an AlphaDrawWire.
    // Called AFTER the visible-set loop in back-to-front order (the collect/sort lives in
    // buildGeometryDrawLists), so the wire order IS the draw order — the host never re-sorts.
    // Field packing mirrors emitStaticDraw (texture SlotInfo cache, material, vColSource,
    // camera-relative world) plus the SkyDrawWire blend fields and the material alpha.
    // FP1c: `dst` defaults to the main-pass scratch; buildFPFrame redirects the identical
    // packing into the FP alpha scratch (same wire format, drawn by the host FP pass).
    void emitAlphaDraw(SlotInfo& si, const MGE::GeometryCache::CachedGeometry& e,
                       std::uint32_t& count, std::vector<std::uint8_t>& dst = g_alphaScratch) {
            IPC::AlphaDrawWire item;
            item.slot      = si.slot;
            diagWorldDet("ALPHA", e.textureName, e.worldTransformD3D);
            item.texIndex  = resolveCachedSlot(e.textureName, si.baseNamePtr, si.baseSlot, si.baseEpoch);
            item.srcBlend  = e.srcBlend;
            item.destBlend = e.destBlend;
            item.alphaRef  = e.alphaTest ? e.alphaRef : 0.0f;
            item.clampMode = e.baseClamp;            // MW's per-map texture address mode
            item.matDiffuse[0]  = e.matDiffuse[0];  item.matDiffuse[1]  = e.matDiffuse[1];  item.matDiffuse[2]  = e.matDiffuse[2];
            item.matAlpha       = e.matDiffuse[3];   // MaterialProperty::alpha (the FFE per-draw fade)
            item.matAmbient[0]  = e.matAmbient[0];  item.matAmbient[1]  = e.matAmbient[1];  item.matAmbient[2]  = e.matAmbient[2];
            MGE::GeometryCache::emissiveForDraw(e, item.matEmissive);
            item.vColSource = (e.hasVertexColor && e.vColSource != 0) ? e.vColSource : 0u;
            memcpy(item.world, e.worldTransformD3D, 16 * sizeof(float));
            // CAMERA-RELATIVE: shift translation by -eye (see emitStaticDraw).
            item.world[12] -= DistantLand::eyePos.x;
            item.world[13] -= DistantLand::eyePos.y;
            item.world[14] -= DistantLand::eyePos.z;
            // AT3 captured-geometry locators unused for a cached-mesh (slot) item — the host
            // reads them only when slot == kAlphaSlotCaptured. Zero so they never alias garbage.
            item.vertexBase = item.indexBase = item.indexCount = 0;
            // Cull mode from the shape's live NiStencilProperty (DRAW_BOTH → CULL_NONE) + winding
            // from the mirror flag, so the host draws it exactly as MW does (single-sided alpha
            // like the draped altar cloth gets CULL_BACK, hiding its back/interior faces).
            item.cullFlags = (e.twoSided ? IPC::kAlphaCullTwoSided : 0u)
                           | (e.mirrored ? IPC::kAlphaCullMirrored : 0u);
            const std::size_t at = dst.size();
            dst.resize(at + sizeof(item));
            memcpy(dst.data() + at, &item, sizeof(item));
            ++count;
    }

    // AT3: emit one captured blended DIP as a sentinel-slot AlphaDrawWire. Unlike emitAlphaDraw
    // there is no uploaded mesh slot — slot = kAlphaSlotCaptured and vertexBase/indexBase/indexCount
    // locate the geometry in the shared captured VB/IB the kickoff ships. Camera-relative world
    // subtraction uses the CURRENT eyePos (rec.world is absolute), so 1-frame-old records don't swim.
    void emitCapturedAlphaDraw(const CapturedAlphaRec& rec, std::uint32_t& count) {
            IPC::AlphaDrawWire item;
            item.slot      = IPC::kAlphaSlotCaptured;
            item.texIndex  = rec.texIndex;
            item.srcBlend  = rec.srcBlend;
            item.destBlend = rec.destBlend;
            item.alphaRef  = rec.alphaRef;
            item.matDiffuse[0]  = rec.matDiffuse[0];  item.matDiffuse[1]  = rec.matDiffuse[1];  item.matDiffuse[2]  = rec.matDiffuse[2];
            item.matAlpha       = rec.matAlpha;
            item.matAmbient[0]  = rec.matAmbient[0];  item.matAmbient[1]  = rec.matAmbient[1];  item.matAmbient[2]  = rec.matAmbient[2];
            item.matEmissive[0] = rec.matEmissive[0]; item.matEmissive[1] = rec.matEmissive[1]; item.matEmissive[2] = rec.matEmissive[2];
            item.vColSource = rec.vColSource;
            memcpy(item.world, rec.world, 16 * sizeof(float));
            item.world[12] -= DistantLand::eyePos.x;
            item.world[13] -= DistantLand::eyePos.y;
            item.world[14] -= DistantLand::eyePos.z;
            item.vertexBase = rec.vertexBase;
            item.indexBase  = rec.indexBase;
            item.indexCount = rec.indexCount;
            // Captured DIPs are billboarded particles (smoke/flames) — keep CULL_NONE (two-sided)
            // so they render exactly as today; culling a camera-facing quad by winding is fragile.
            item.cullFlags  = IPC::kAlphaCullTwoSided;
            // SCOPE GAP (disclosed, not a silent default): captured DIPs are reconstructed from D3D8
            // render state, not from the scene graph, and RenderedState does not carry the sampler
            // address mode (the proxy translates D3DTSS_ADDRESSU->D3DSAMP_ADDRESSU but never records
            // it). So we ship MW's default here. Harmless for what actually lands on this path —
            // billboarded particle quads whose UVs live inside [0,1], where wrap and clamp are the
            // same sample. If a captured DECAL ever shows the tiling artifact, the fix is to snapshot
            // D3DSAMP_ADDRESSU/V into RenderedState at the reject gate and ship it here.
            item.clampMode  = IPC::kTexWrapSWrapT;
            const std::size_t at = g_alphaScratch.size();
            g_alphaScratch.resize(at + sizeof(item));
            memcpy(g_alphaScratch.data() + at, &item, sizeof(item));
            ++count;
    }

    // Cut 2 (dense-city frame attack): ONE pass over the visible set builds all three
    // geometry draw lists. Previously three functions each re-iterated
    // frustumVisibleKeys with their own g_keySlot + cache-map finds — 3x the hash
    // work on the frame's hottest client loop — and re-resolved every texture name
    // through resolveTextureSlot's normalize (heap string) + string-hash find every
    // frame (now SlotInfo-cached, see resolveCachedSlot). Emission order within each
    // list is unchanged (same key iteration order), so the wire bytes are identical.
    //
    // Source = MGE's OWN current-frame frustum-visible set (frustumVisibleKeys /
    // buildFrustumVisibleSet, built in renderStage0) — the SAME source the proven D3D9
    // cache color path (renderCachedOpaque) consumes. This is adaptive and decouples the
    // Forge near draw from the MWSE occlusion plugin entirely:
    //   - plugin ON  → buildFrustumVisibleSet fills it from the engine's drawn set
    //     (s_visibleKeys: de-duped, one LOD per object, occlusion-correct, CURRENT frame).
    //   - plugin OFF → frustum-only fallback (whole-cache frustum walk). Near still renders.
    //
    // Dispatch replicates the three old loops' per-entry filters EXACTLY:
    //   - skinned first (stride-44 VB + GPU palette pipeline), regardless of texture.
    //   - !d3dTexture: drops UNTEXTURED entries — the worldPickObjectRoot collision
    //     proxies that mirror each visual object; drawing them z-fights movers. This is
    //     THE de-dup. (Do NOT also filter isPickRoot — legit movers like dropped
    //     items/projectiles LIVE in the pick root and ARE textured.)
    //   - blendEnable (non-landscape): alpha-blended OBJECTS stay on the engine/alpha
    //     path. Terrain is the exception — the D9 oracle draws ALL isLandscape
    //     regardless of blendEnable (alpha-splat trishapes included).
    //   - non-landscape parts with dark/detail/glow siblings → multi-map pipeline
    //     (stride-60 wide VB); everything else (terrain included) → static pipeline.
    void buildGeometryDrawLists(std::uint32_t& drawCount, std::uint32_t& skinnedCount,
                                std::uint32_t& multiMapCount, std::uint32_t& alphaCount) {
        MGE_ZoneScopedN("build:geometry");
        const double tFn0 = nowMs();   // whole-function wall (the [bspike] "build" number)
        const std::uint32_t cap0 = MGE::GeometryCache::liveCaptureCount();
        drawCount = 0;
        skinnedCount = 0;
        multiMapCount = 0;
        alphaCount = 0;
        // Phase 0 sub-probe accumulators (published to g_lastBuild* below; ~20-30ns/call QPC tax).
        // tailEnsureMs = the tail's LIVE near-actor pose-refresh ensureLive (immovable, subset of
        // tailMs); tailMs - tailEnsureMs is the worker-safe part of the tail (cached caster re-emit
        // + alpha sort/pack). movable = emit + (tail - tailEnsure); immovable = ensureLive + tailEnsure.
        double ensureLiveMs = 0.0, emitMs = 0.0, tailMs = 0.0, tailEnsureMs = 0.0;
        // Tail sub-probe: is the fixed ~1.1ms tail the offscreen-caster full-cacheMap scan
        // (tailScanMs — the @1694 for-loop over EVERY cached entry) or the alpha merge/sort/pack
        // (tailAlphaMs)? The scan is the reduce-directly candidate (near-caster spatial list).
        double tailScanMs = 0.0, tailAlphaMs = 0.0;
        std::uint32_t visitedKeys = 0;
        // Cut 2B fold: on fold frames buildFrustumVisibleSet deferred its ensureLive
        // loop here (foldVisibleKeys = the raw classify set) instead of running it AND
        // having this function re-hash the same keys — ONE pass does live-freshen +
        // emit. Non-fold frames iterate frustumVisibleKeys exactly as before (entries
        // already freshened by buildFrustumVisibleSet or the full walk).
        const auto* foldKeys = DistantLand::foldVisibleKeys();
        const auto& keys = DistantLand::frustumVisibleKeys();
        const auto& cacheMap = MGE::GeometryCache::cache();

        g_drawScratch.clear();
        g_drawScratch.reserve((foldKeys ? foldKeys->size() : keys.size()) * sizeof(IPC::DrawItemWire));
        g_skinnedScratch.clear();
        g_multiMapScratch.clear();
        g_alphaScratch.clear();

        const bool wantStatic  = (bool)g_drawVec;
        const bool wantSkinned = (bool)g_skinnedVec;
        const bool wantMM      = (bool)g_multiMapVec;
        // AT1 sorted-alpha: gated by the bring-up ini flag on top of the vec (capture + emit +
        // host draw all ride this one gate; off = the blended set stays engine-drawn as before).
        const bool wantAlpha   = (bool)g_alphaVec && Configuration.ForgeAlphaPass;
        if (!wantStatic && !wantSkinned && !wantMM && !wantAlpha) {
            // AT3 defensive clear: drop any captured records/geometry so an idle frame (menu/
            // loading) can't leave last frame's captures to accumulate or ship stale.
            g_capturedEmitted = 0;
            g_capRecs.clear();
            g_capVertScratch.clear();
            g_capIdxScratch.clear();
            g_capTexMemo.clear();
            g_alphaDedup.clear();
            g_lastBuildEnsureLiveMs = 0.0;   // Phase 0: idle frame did no build work
            g_lastBuildEmitMs       = 0.0;
            g_lastBuildTailMs       = 0.0;
            g_lastBuildTailEnsureMs = 0.0;
            g_lastBuildTailScanMs   = 0.0;
            g_lastBuildTailAlphaMs  = 0.0;
            g_lastBuildKeys         = 0;
            g_lastBuildCaptures     = 0;
            return;
        }

        // AT1 collect: blended shapes are gathered (not emitted) during the loop, then sorted
        // back-to-front and packed AFTER it — MW's sorter criterion is bound-center view depth.
        // Pointers into the two unordered_maps are element-stable across ensureLive inserts.
        // AT1 cached-mesh (si+e) OR AT3 captured-geometry (cap) candidate — ONE sort covers both
        // so cross-set blend order is correct. Exactly one of {e, cap} is non-null per entry.
        struct AlphaCand {
            float depth;
            SlotInfo* si;                                       // cached: valid; captured: nullptr
            const MGE::GeometryCache::CachedGeometry* e;        // cached: valid; captured: nullptr
            const CapturedAlphaRec* cap;                        // captured: valid; cached: nullptr
        };
        static std::vector<AlphaCand> alphaCands;   // single-threaded; reused frame-to-frame
        alphaCands.clear();
        // AT3 old-msoc double-draw guard set — rebuilt THIS frame from the cached blends the host
        // draws, then read by captureAlphaDraw during this frame's scene>=1 (see g_alphaDedup).
        g_alphaDedup.clear();
        // World-space view forward (mwView's 3rd column) for the depth key; eye-relative so the
        // key is invariant to the camera-relative world shift emitAlphaDraw applies later.
        const float fwdX = DistantLand::mwView._13;
        const float fwdY = DistantLand::mwView._23;
        const float fwdZ = DistantLand::mwView._33;

        // Per-entry dispatch, identical on both paths (see the filter contract above).
        // slot is mutable — the emit helpers update its cached texture SlotInfo.
        auto dispatch = [&](auto& slot, const auto& e) {
            if (e.isSky) return;   // sky rides the Forge alpha-blend sky pass (buildSkyDrawList)
            if (e.isFP) return;    // FP arms ride the dedicated host FP pass (buildFPDrawLists)
            if (e.isSkinned) {
                if (wantSkinned) emitSkinnedDraw(slot, e, skinnedCount);
                return;
            }
            if (!e.d3dTexture) {
                // Textureless. Pick-root entries are worldPickObjectRoot COLLISION PROXIES that
                // mirror each visual object and z-fight movers → still dropped (THE de-dup). But
                // genuine textureless VISUAL geometry in the object root (e.g. ex_de_shack_02's
                // solid day plane behind the netting) is drawn OPAQUE by MW — emit it on the static
                // path with slot 0 (host default white) so material/vcol give the solid colour.
                // Skip the blend path (a textureless alpha shape has no base to composite).
                if (e.isPickRoot || e.blendEnable) return;
                if (wantStatic) emitStaticDraw(slot, e, drawCount);
                return;
            }
            if (!e.isLandscape && e.blendEnable) {
                // AT1: non-landscape blended shapes ride the host alpha pass. Single-map tri shapes
                // go to the sorted-alpha list; MULTI-map blends (Glow-in-the-Dahrk night windows =
                // base orange x dark shape) ride the multimap vec tagged BLENDED (Route C) — the
                // host draws them in the alpha stage with multimap_alpha.frag so the dark modulate
                // shows (else base-map-only white via AT3). Skinned blends went to the skinned path.
                const bool multiMap = (e.d3dDark || e.d3dDetail || e.d3dGlow);
                if (wantAlpha && !multiMap) {
                    const float* w = e.worldTransformD3D;
                    const float cx = e.boundsCenter[0], cy = e.boundsCenter[1], cz = e.boundsCenter[2];
                    const float wx = cx * w[0] + cy * w[4] + cz * w[8]  + w[12] - DistantLand::eyePos.x;
                    const float wy = cx * w[1] + cy * w[5] + cz * w[9]  + w[13] - DistantLand::eyePos.y;
                    const float wz = cx * w[2] + cy * w[6] + cz * w[10] + w[14] - DistantLand::eyePos.z;
                    alphaCands.push_back({ wx * fwdX + wy * fwdY + wz * fwdZ, &slot, &e, nullptr });
                } else if (wantMM && multiMap) {
                    emitMultiMapDraw(slot, e, multiMapCount, /*blended=*/true);
                } else {
                    return;   // feature off — leave MW's DIP (AT3 white) as the fallback, no dedup
                }
                // AT3: record (GPU texture, vertexCount) so captureAlphaDraw skips this exact blend
                // if MW's own DIP for it still reaches the reject gate (old-msoc dll: opaque-only
                // skip → cached blends both DIP AND ride the host). Both host paths above suppress it.
                if (e.d3dTexture) {
                    g_alphaDedup.insert(((std::uint64_t)(std::uintptr_t)e.d3dTexture << 32)
                                        | (std::uint64_t)e.vertexCount);
                }
                return;
            }
            if (!e.isLandscape && (e.d3dDark || e.d3dDetail || e.d3dGlow)) {
                if (wantMM) emitMultiMapDraw(slot, e, multiMapCount);
            } else if (wantStatic) {
                emitStaticDraw(slot, e, drawCount);
            }
        };

        // Stage 2a: arm the first-sight capture budget for the visible-set loop only.
        // Unlimited when uncapped (panel A/B), when the cache is empty (boot), or within the
        // cell-epoch grace window (a transition's mass recapture must land at once, not
        // trickle). Reset to unlimited right after the loop so no OTHER ensureLive caller
        // (tail pose refresh here, main-thread classify in buildFrustumVisibleSet) is gated.
        {
            static std::uint32_t s_seenEpoch  = ~0u;
            static unsigned      s_epochFrame = 0;
            if (g_cellEpoch != s_seenEpoch) { s_seenEpoch = g_cellEpoch; s_epochFrame = g_frame; }
            const bool unlimited = !g_captureBudgetOn || cacheMap.empty()
                                   || (g_frame - s_epochFrame < kCaptureEpochGraceFrames);
            MGE::GeometryCache::setCaptureBudget(unlimited ? -1 : kCaptureBudgetPerFrame);
        }
        {
            MGE_ZoneScopedN("geom:visLoop");
            if (foldKeys) {
                for (std::uint32_t key : *foldKeys) {
                    ++visitedKeys;
                    // Freshen (or lazily capture) straight off the live NiTriShape — a
                    // classify key is engine-drawn THIS frame, so the pointer is valid by
                    // construction. MUST run before the g_keySlot probe: a first-sight key
                    // has no slot until ensureLive's capture registers it (same frame).
                    const double te0 = nowMs();
                    const auto* e = MGE::GeometryCache::ensureLive(key);
                    ensureLiveMs += nowMs() - te0;
                    if (!e) continue;                     // no model data / capture failed / deferred
                    // The unusable-skinned filter buildFrustumVisibleSet applies on
                    // non-fold frames (never drawn; trips the bound helper).
                    if (e->isSkinned && (e->skinnedUnsupported || e->numBones == 0)) continue;
                    auto ks = g_keySlot.find(key);
                    if (ks == g_keySlot.end()) {
                        continue;   // not an uploaded part (or not yet shipped)
                    }
                    const double td0 = nowMs();
                    dispatch(ks->second, *e);
                    emitMs += nowMs() - td0;
                }
            } else {
                for (std::uint32_t key : keys) {
                    ++visitedKeys;
                    auto ks = g_keySlot.find(key);
                    if (ks == g_keySlot.end()) {
                        continue;   // not an uploaded part (or not yet shipped)
                    }
                    auto ce = cacheMap.find(key);
                    if (ce == cacheMap.end()) {
                        continue;   // evicted since the visible-set build
                    }
                    const double td0 = nowMs();
                    dispatch(ks->second, ce->second);
                    emitMs += nowMs() - td0;
                }
            }
        }
        // Snapshot BEFORE the reset — setCaptureBudget clears the deferred counter.
        const std::uint32_t capDeferred = MGE::GeometryCache::captureDeferredLastBuild();
        MGE::GeometryCache::setCaptureBudget(-1);

        // Offscreen shadow casters: skinned + multimap NPC parts near the camera but OUTSIDE the
        // frustum were dropped by the visible-set loops above, so their shadows freeze and shed
        // parts as they leave view (the host can only cast what it receives). Re-emit them from
        // CACHED data — NO live NiTriShape read, because an offscreen key may be stale/freed and
        // ensureLive's recycled-address guard runs after the deref. The emit helpers subtract the
        // CURRENT eye from the absolute cached pose, so a stationary/idle NPC's frozen pose is
        // exactly right; a genuinely-moving offscreen NPC lags at its last-seen spot until it
        // re-enters view (a live near-actor walk is the follow-up). De-dup against the ACTUAL
        // emitted set (this frame's visible keys) — NOT lastFrame. The eviction sweep runs
        // ensureFullWalk() every kEvictSweepInterval (~30) frames, which visits offscreen near
        // entries and stamps lastFrame = currentFrame to keep them resident, WITHOUT adding them
        // to the visible set. A lastFrame de-dup would then skip the offscreen NPC on exactly
        // those frames while the main loop also skips it → the caster blinks every ~30 frames (the
        // tradehouse pulsing). Visible-set membership is immune to that stamping.
        const double tail0 = nowMs();   // Phase 0: offscreen casters + pose refresh + alpha sort/pack
        if (wantSkinned || wantMM || wantStatic) {
            MGE_ZoneScopedN("geom:offscreenCasters");
            static std::unordered_set<std::uint32_t> s_visLookup;
            s_visLookup.clear();
            if (foldKeys) { for (std::uint32_t k : *foldKeys) s_visLookup.insert(k); }
            else          { for (std::uint32_t k : keys)      s_visLookup.insert(k); }
            // [drift-fix] near offscreen movers to pose-refresh after this loop (see below).
            struct ReEmitCand { std::uint32_t key; std::uint8_t kind; };
            static std::vector<ReEmitCand> s_reEmitRefresh;   s_reEmitRefresh.clear();
            const float r2 = kShadowCasterRadius * kShadowCasterRadius;
            const std::uint64_t cacheFrame = MGE::GeometryCache::currentFrame();
            std::uint32_t suppressSkips = 0;
            // Iterate the cache's maintained mover-candidate set (skinned / multimap head / rigid
            // LIVE) instead of the WHOLE cache — the offscreen re-emit only ever cared about that
            // subset, and the full-map scan was the fixed ~1ms tail the build sub-probe pinned down.
            // The set is a want-flag-independent SUPERSET, so the per-frame filter below runs
            // unchanged (belt-and-braces: any key still fails exactly as it did in the full scan).
            const auto& moverKeys = MGE::GeometryCache::moverCandidates();
            for (std::uint32_t mkey : moverKeys) {
                auto cit = cacheMap.find(mkey);
                if (cit == cacheMap.end()) continue;        // set ⊆ cache invariant; guard anyway
                const auto& e = cit->second;
                if (s_visLookup.count(mkey)) continue;      // in the visible set → already emitted above
                if (e.isSky) continue;
                if (e.isFP) continue;   // FP arms: dedicated host FP pass, never a world caster (FP1a)
                // FP0: entries the engine appCulled this frame (the 1st-person player's
                // 3rd-person body — at the camera, so always inside the radius below).
                // Without this the freshly-culled body kept re-emitting into the colour
                // lists until the eviction sweep (the 3rd→1st POV-switch linger).
                if (e.suppressedFrame == cacheFrame) { ++suppressSkips; continue; }
                const bool isMM = !e.isLandscape && !e.blendEnable && e.d3dTexture
                                  && (e.d3dDark || e.d3dDetail || e.d3dGlow);
                if (e.isSkinned) {
                    if (!wantSkinned || e.skinnedUnsupported || e.numBones == 0) continue;
                } else if (isMM) {
                    if (!wantMM) continue;
                } else if (e.isLive && !e.blendEnable && !e.isLandscape) {
                    // Rigid LIVE parts (C4d): on NPCs that is MOST of the body — MW attaches
                    // hair/neck/arms/legs rigidly to bones; only a few parts are skin-deformed.
                    // Since C4d these render ONLY from the dynamic tile, whose gather requires a
                    // record refreshed within kMoverFresh(3) frames (forgerender.cpp:5637) — and
                    // LIVE records are barred from the static gathers (5806). So an offscreen
                    // actor's rigid parts vanished from the shadow in 3 frames (shack repro
                    // 2026-07-10; the expire-movers toggle can't help — it only gates record
                    // DELETION, not gather membership). Re-emitting here refreshes the host
                    // record every frame; emitStaticDraw already ships casterFlags=LIVE + the
                    // cached world (same freshness as the skinned palettes above).
                    // isPickRoot does NOT disqualify: NPCs live under worldPickObjectRoot, so
                    // every body part carries the flag — it only marks a COLLISION PROXY when
                    // the shape is also textureless (dispatch()'s exact rule, replicated here;
                    // first cut required !isPickRoot outright and skipped the whole body).
                    if (!e.d3dTexture && e.isPickRoot) continue;   // collision proxies only
                    if (!wantStatic) continue;
                } else {
                    continue;   // shadow-relevant movers only (skinned / MM heads / rigid LIVE)
                }
                // World-space centre from cached bounds (no NiTriShape deref).
                D3DXVECTOR3 c; float rad;
                if (e.isSkinned) cacheSkinnedWorldBounds(e, c, rad);
                else             cacheWorldBounds(e, c, rad);
                const float dx = c.x - DistantLand::eyePos.x;
                const float dy = c.y - DistantLand::eyePos.y;
                const float dz = c.z - DistantLand::eyePos.z;
                if (dx * dx + dy * dy + dz * dz > r2) continue;
                auto ks = g_keySlot.find(mkey);
                if (ks == g_keySlot.end()) continue;       // never uploaded a host slot
                if (g_shadowLiveNearActors) {
                    // Defer: these near movers get a live pose refresh AFTER the loop (ensureLive
                    // must not mutate cacheMap / the candidate set while we iterate it). kind: 0
                    // skinned, 1 MM, 2 rigid.
                    s_reEmitRefresh.push_back({ mkey, (std::uint8_t)(e.isSkinned ? 0 : isMM ? 1 : 2) });
                } else {
                    if (e.isSkinned)    emitSkinnedDraw(ks->second, e, skinnedCount);
                    else if (isMM)      emitMultiMapDraw(ks->second, e, multiMapCount);
                    else                emitStaticDraw(ks->second, e, drawCount);
                }
            }
            // [drift-fix] Live near-actor pose refresh. Now OUTSIDE the cacheMap iteration, so
            // ensureLive() may freely refresh/re-capture each entry. It rebuilds the bone palette
            // from the CURRENT skeleton pose (skips VB/material unless the mesh revision changed),
            // so the offscreen shadow tracks every frame instead of snapping on the 30-frame sweep.
            for (const auto& rc : s_reEmitRefresh) {
                const double tpr0 = nowMs();   // Phase 0: live pose-refresh ensureLive (immovable)
                const auto* fr = MGE::GeometryCache::ensureLive(rc.key);
                tailEnsureMs += nowMs() - tpr0;
                if (!fr) continue;                          // key gone/out-of-domain → skip (no stale fallback)
                auto ks = g_keySlot.find(rc.key);
                if (ks == g_keySlot.end()) continue;
                if      (rc.kind == 0) emitSkinnedDraw(ks->second, *fr, skinnedCount);
                else if (rc.kind == 1) emitMultiMapDraw(ks->second, *fr, multiMapCount);
                else                   emitStaticDraw(ks->second, *fr, drawCount);
            }
            // FP0 instrumentation: each skipped frame is a frame the body WOULD have
            // lingered pre-fix. A run ends when the re-emit condition stops matching
            // (eviction / visible again) — its length is the exact pre-fix linger.
            static std::uint32_t s_supRunFrames = 0, s_supRunMax = 0;
            if (suppressSkips > 0) {
                ++s_supRunFrames;
                if (suppressSkips > s_supRunMax) s_supRunMax = suppressSkips;
            } else if (s_supRunFrames > 0) {
                LOG::logline("[fp0] re-emit suppression run ended: %u frames (pre-fix linger), max %u draws/frame",
                             s_supRunFrames, s_supRunMax);
                s_supRunFrames = 0;
                s_supRunMax = 0;
            }
        }
        const double tScanEnd = nowMs();   // Phase 0: offscreen-caster scan done; alpha merge/sort next
        tailScanMs = tScanEnd - tail0;

        // AT3: merge the captured blended DIPs (particles/smoke/flames + multimap/decal/untextured
        // blends the host cache pass doesn't own) captured LAST frame (g_capRecs) into the SAME
        // candidate list so ONE sort orders the whole blended set back-to-front. Depth uses the
        // CURRENT eye + forward against the record's absolute world-space centroid (no swim from the
        // 1-frame latency). The records are consumed (cleared) after emit; the geometry bytes stay
        // in g_capVertScratch/g_capIdxScratch for the kickoff to ship.
        {
        MGE_ZoneScopedN("geom:alphaMergeSort");   // AT3 merge + AT1 sort/pack (the alpha tail)
        if (wantAlpha) {
            for (const auto& rec : g_capRecs) {
                const float dx = rec.centroid[0] - DistantLand::eyePos.x;
                const float dy = rec.centroid[1] - DistantLand::eyePos.y;
                const float dz = rec.centroid[2] - DistantLand::eyePos.z;
                alphaCands.push_back({ dx * fwdX + dy * fwdY + dz * fwdZ, nullptr, nullptr, &rec });
            }
        }

        // AT1/AT3: back-to-front (descending view depth — farthest drawn first, MW's sort) then pack.
        if (!alphaCands.empty()) {
            std::sort(alphaCands.begin(), alphaCands.end(),
                      [](const AlphaCand& a, const AlphaCand& b) { return a.depth > b.depth; });
            std::size_t first = 0;
            if (alphaCands.size() > IPC::kMaxAlphaDraws) {
                // Over cap: drop the FARTHEST (sorted front) — the near draws matter most.
                first = alphaCands.size() - IPC::kMaxAlphaDraws;
                static bool logged = false;
                if (!logged) {
                    LOG::logline("!! [alpha] %zu blended shapes this frame > cap %u — farthest dropped",
                                 alphaCands.size(), IPC::kMaxAlphaDraws);
                    logged = true;
                }
            }
            g_alphaScratch.reserve((alphaCands.size() - first) * sizeof(IPC::AlphaDrawWire));
            for (std::size_t i = first; i < alphaCands.size(); ++i) {
                if (alphaCands[i].cap) {
                    emitCapturedAlphaDraw(*alphaCands[i].cap, alphaCount);
                } else {
                    emitAlphaDraw(*alphaCands[i].si, *alphaCands[i].e, alphaCount);
                }
            }
            // Bring-up diagnostic: name the captured set once (what the alpha list actually IS —
            // if the in-game "sorted" the eye notices isn't in here, it's an AT3 leftover, not a
            // draw bug). Remove after verify.
            static bool s_alphaNamed = false;
            if (!s_alphaNamed) {
                s_alphaNamed = true;
                LOG::logline(">> [alpha-cap] %zu blended shapes this frame (%zu captured AT3):",
                             alphaCands.size(), g_capRecs.size());
                const std::size_t nDump = (alphaCands.size() < 16u) ? alphaCands.size() : 16u;
                for (std::size_t i = 0; i < nDump; ++i) {
                    const auto& c = alphaCands[i];
                    if (c.cap) {
                        LOG::logline(">> [alpha-cap] #%zu depth=%.0f CAPTURED tex=%u vc=%u tri=%u blend=%u/%u matA=%.2f vcs=%u",
                                     i, c.depth, c.cap->texIndex, c.cap->vertexCount, c.cap->indexCount / 3,
                                     c.cap->srcBlend, c.cap->destBlend, c.cap->matAlpha, c.cap->vColSource);
                    } else {
                        const auto& e = *c.e;
                        LOG::logline(">> [alpha-cap] #%zu depth=%.0f tex=%s vc=%u tri=%u blend=%u/%u matA=%.2f",
                                     i, c.depth, e.textureName ? e.textureName : "(null)",
                                     e.vertexCount, e.triangleCount, e.srcBlend, e.destBlend, e.matDiffuse[3]);
                    }
                }
            }
        }
        }   // geom:alphaMergeSort

        // AT3: the captured RECORDS are now consumed (emitted as wire items). Clear them + the
        // per-frame texture memo so this frame's scene>=1 captures start fresh; the geometry BYTES
        // (g_capVertScratch/g_capIdxScratch) stay for the kickoff assign to ship, then are cleared
        // there. On !wantAlpha frames the records were never merged — drop them so they can't
        // accumulate (and drop the bytes too, since nothing will ship them).
        g_capturedEmitted = wantAlpha ? (std::uint32_t)g_capRecs.size() : 0u;
        g_capRecs.clear();
        g_capTexMemo.clear();
        if (!wantAlpha) {
            g_capVertScratch.clear();
            g_capIdxScratch.clear();
        }

        // Phase 0: publish the build-split. tailMs covers the offscreen-caster re-emit + live
        // near-actor pose refresh + alpha merge/sort/pack (everything after the two main loops).
        const double tBuildEnd = nowMs();
        tailMs = tBuildEnd - tail0;
        tailAlphaMs = tBuildEnd - tScanEnd;   // alpha merge + sort + pack (worker-safe, but not the scan)
        g_lastBuildEnsureLiveMs = ensureLiveMs;
        g_lastBuildEmitMs       = emitMs;
        g_lastBuildTailMs       = tailMs;
        g_lastBuildTailEnsureMs = tailEnsureMs;
        g_lastBuildTailScanMs   = tailScanMs;
        g_lastBuildTailAlphaMs  = tailAlphaMs;
        g_lastBuildKeys         = visitedKeys;
        const std::uint32_t captures = MGE::GeometryCache::liveCaptureCount() - cap0;
        g_lastBuildCaptures     = captures;

        // Stage 0 sub-probe plots (no-ops without Tracy): the same split as [hb], but per-frame
        // so a spike frame's carrier is directly readable off the timeline.
        MGE_TracyPlot("build ensureLive ms", ensureLiveMs);
        MGE_TracyPlot("build emit ms", emitMs);
        MGE_TracyPlot("build tailScan ms", tailScanMs);
        MGE_TracyPlot("build tailAlpha ms", tailAlphaMs);
        MGE_TracyPlot("build captures", (std::int64_t)captures);
        MGE_TracyPlot("build alphaCands", (std::int64_t)alphaCands.size());

        // [bspike] (panel toggle, default OFF): one line naming WHICH component carried a slow build
        // and whether it aligns with an eviction sweep or a capture burst. Hard-throttled >=1s
        // apart + session cap — spike logging on this hot path has collapsed the frame before.
        const double buildTotal = tBuildEnd - tFn0;
        static double s_buildEma = 0.0;
        s_buildEma = (s_buildEma == 0.0) ? buildTotal : s_buildEma * 0.95 + buildTotal * 0.05;
        if (g_buildSpikeLog && buildTotal > std::max(1.5 * s_buildEma, 1.2)) {
            static double   s_lastSpikeLogMs = 0.0;
            static unsigned s_spikeLogged    = 0;
            if (tBuildEnd - s_lastSpikeLogMs >= 1000.0 && s_spikeLogged < 64) {
                s_lastSpikeLogMs = tBuildEnd;
                ++s_spikeLogged;
                LOG::logline("!! [bspike] build=%.2f ema=%.2f ensure=%.2f emit=%.2f scan=%.2f alpha=%.2f tailLive=%.2f "
                             "keys=%u captures=%u deferred=%u alphaCands=%zu movers=%zu cache=%zu sweepAge=%u",
                             buildTotal, s_buildEma, ensureLiveMs, emitMs, tailScanMs, tailAlphaMs,
                             tailEnsureMs, visitedKeys, captures, capDeferred, alphaCands.size(),
                             MGE::GeometryCache::moverCandidates().size(), cacheMap.size(),
                             MGE::GeometryCache::framesSinceEvictSweep());
            }
        }
    }

    // Tier 3a: gather this frame's point lights into g_lightScratch as PointLightWire[]. Source is
    // the MGE scene-graph snapshot (MGE::SceneGraph::pointLights()) — the same NI::PointLight POD
    // the FFE many-lights path consumes — read under the SnapshotReadLock (the async walk swaps the
    // vector atomically). World-space, no culling (Tier 3a is correctness-first; the frag loops the
    // whole set). Diffuse is already dimmer-scaled; pointLightMult is baked here (1.0 on the main
    // cache path → identity). Clamped to kMaxPointLights (logged once if exceeded). Returns the count.
    std::uint32_t buildLightList() {
        MGE_ZoneScopedN("build:lights");
        if (!g_lightVec) {
            return 0;
        }
        // The main cache color path (rendercachedcolor.cpp) renders the SAME opaque set Forge does
        // and passes pointLightMult = 1.0f, so bake 1.0 (kept explicit for future per-frame scaling).
        constexpr float pointLightMult = 1.0f;

        // Stage 4b feed: how long this builder stalls acquiring the walk-thread snapshot lock.
        // Plotted (not zoned — the lock must outlive any acquisition scope); >0.1ms stalls
        // coinciding with build spikes are the go-signal for the RCU snapshot swap.
        const double tLock0 = nowMs();
        MGE::SceneGraph::SnapshotReadLock lk;
        MGE_TracyPlot("lights snapshotLock ms", nowMs() - tLock0);
        const auto& lights = MGE::SceneGraph::pointLights();

        // Frustum cull (host GPU shrink): only the NEAR scene consumes point lights
        // (opaque.frag/multimap.frag loop them; statics/distantland don't), and a light's
        // contribution is EXACTLY zero beyond 2·radius (the shader's smoothstep cutoff).
        // So a light whose 2r sphere misses the game frustum can't touch any lit pixel —
        // drop it before it costs every pixel a loop iteration. This also makes the
        // kMaxPointLights cap meaningful: cull first, THEN cap, so a dense cell keeps the
        // lights that can actually show (was: arbitrary first-128 of the snapshot order).
        D3DXMATRIX lightVP;
        D3DXMatrixMultiply(&lightVP, &DistantLand::mwView, &DistantLand::mwProj);
        const ViewFrustum lightFrustum(&lightVP);

        g_lightScratch.clear();
        std::uint32_t count = 0;
        std::uint32_t culled = 0;
        for (const auto& pl : lights) {
            // P2: resolve persistent identity BEFORE the frustum cull so a light that steps
            // off-screen keeps its id + lastPos updated (it can't recycle just for being culled).
            std::uint32_t id = 0, flags = 0;
            {
                const float* p = pl.worldPos;
                auto it = g_lightTracks.find(pl.source);
                if (it == g_lightTracks.end()) {
                    LightTrack t{ g_nextLightId++, g_frame, { p[0], p[1], p[2] } };
                    g_lightTracks.emplace(pl.source, t);
                    id = t.id;
                    flags = IPC::kLightFlagNew;
                } else {
                    LightTrack& t = it->second;
                    const float dx = p[0] - t.lastPos[0];
                    const float dy = p[1] - t.lastPos[1];
                    const float dz = p[2] - t.lastPos[2];
                    const float d2 = dx*dx + dy*dy + dz*dz;
                    const float jump = std::max(2.0f * pl.radius, kLightIdJumpFloor);
                    if ((g_frame - t.lastFrame) > kLightIdGapFrames || d2 > jump * jump) {
                        t.id = g_nextLightId++;       // recycled address -> new light
                        flags = IPC::kLightFlagNew;
                    } else if (d2 > kLightMovedEps * kLightMovedEps) {
                        flags = IPC::kLightFlagMoved;
                    }
                    id = t.id;
                    t.lastFrame  = g_frame;
                    t.lastPos[0] = p[0]; t.lastPos[1] = p[1]; t.lastPos[2] = p[2];
                }
            }

            BoundingSphere ls;
            ls.center = D3DXVECTOR3(pl.worldPos[0], pl.worldPos[1], pl.worldPos[2]);
            ls.radius = 2.0f * pl.radius;
            if (lightFrustum.ContainsSphere(ls) == ViewFrustum::OUTSIDE) {
                ++culled;
                continue;
            }
            if (count >= IPC::kMaxPointLights) {
                static bool logged = false;
                if (!logged) {
                    LOG::logline("!! [light] %zu in-frustum point lights this frame > cap %u — extra dropped (Tier 3b clustering lifts this)",
                                 lights.size() - culled, IPC::kMaxPointLights);
                    logged = true;
                }
                break;
            }
            // Magic-light falloff classification — mirror of the engine-emit PPL path
            // (ffeshader.cpp:1132-1167). The snapshot ships raw NiPointLight attenuation and
            // the host frag evaluates c + l·d + q·d² verbatim, so MW's magic-light encodings
            // need the same rewrites (FFE's post-memset default is c=0.33, l=0, q=0):
            //   x > 0              — standard placed light: ship as-is.
            //   z > 0 (x == 0)     — MCP-patched magic light: quadratic-only, over-bright diffuse.
            //   y == 0.10000001f   — projectile light (MW hardcodes {0, 3/30, 0}): FFE's brighter
            //                        pure-quadratic replacement, colour/position unchanged.
            //   y > 0              — Light magic effect {0, 3/(22·mag), 0}: brightness override +
            //                        quadratic falloff + Z lift. FFE's half-lambert ambient weight
            //                        has no wire lane / frag term — known fidelity gap (backface
            //                        glow-through is dimmer than PPL).
            constexpr float kFFEDefaultFalloffConstant = 0.33f;
            float diffuse[3] = { pl.diffuse[0], pl.diffuse[1], pl.diffuse[2] };
            float falloff[3] = { pl.falloff[0], pl.falloff[1], pl.falloff[2] };
            float zLift = 0.0f;
            if (falloff[0] <= 0.0f) {
                if (falloff[2] > 0.0f) {
                    diffuse[0] *= kFFEDefaultFalloffConstant;
                    diffuse[1] *= kFFEDefaultFalloffConstant;
                    diffuse[2] *= kFFEDefaultFalloffConstant;
                    falloff[0] = kFFEDefaultFalloffConstant;
                    falloff[1] = 0.0f;
                    falloff[2] = kFFEDefaultFalloffConstant * falloff[2];
                } else if (falloff[1] == 0.10000001f) {
                    falloff[0] = kFFEDefaultFalloffConstant;
                    falloff[1] = 0.0f;
                    falloff[2] = 5e-5f;
                } else if (falloff[1] > 0.0f) {
                    const float brightness = 0.25f + 1e-4f / falloff[1];
                    diffuse[0] = brightness;
                    diffuse[1] = brightness;
                    diffuse[2] = brightness;
                    falloff[0] = kFFEDefaultFalloffConstant;
                    falloff[2] = 0.5555f * falloff[1] * falloff[1];
                    falloff[1] = 0.0f;
                    zLift = 25.0f;
                }
            }

            IPC::PointLightWire w;
            // CAMERA-RELATIVE: light positions are compared against the (now camera-relative)
            // WorldPos in the frag, so shift them by -eye too (see buildDrawList).
            w.posRadius[0] = pl.worldPos[0] - DistantLand::eyePos.x;
            w.posRadius[1] = pl.worldPos[1] - DistantLand::eyePos.y;
            w.posRadius[2] = pl.worldPos[2] - DistantLand::eyePos.z + zLift;
            w.posRadius[3] = pl.radius;
            w.color[0] = diffuse[0] * pointLightMult;
            w.color[1] = diffuse[1] * pointLightMult;
            w.color[2] = diffuse[2] * pointLightMult;
            if (pl.fixture) { flags |= IPC::kLightFlagFixture; }   // ESM fixture → host shadow-priority boost
            w.color[3] = IPC::packLightIdFlags(id, flags);   // P2 identity lane (shader ignores .w)
            w.falloff[0] = falloff[0];
            w.falloff[1] = falloff[1];
            w.falloff[2] = falloff[2];
            w.falloff[3] = 0.0f;
            const std::size_t at = g_lightScratch.size();
            g_lightScratch.resize(at + sizeof(w));
            memcpy(g_lightScratch.data() + at, &w, sizeof(w));
            ++count;
        }

        // Evict identity entries whose light hasn't been seen in a long time (cell changes drop
        // whole lantern sets — their pointers age out here so the map stays bounded and a future
        // NiLight reusing an old address is guaranteed a fresh id via the not-found path).
        for (auto it = g_lightTracks.begin(); it != g_lightTracks.end(); ) {
            if (g_frame - it->second.lastFrame > kLightTrackEvict) {
                it = g_lightTracks.erase(it);
            } else {
                ++it;
            }
        }
        return count;
    }

    // SK1 sky takeover: gather this frame's sky parts into g_skyScratch as SkyDrawWire[]. Source is
    // the WHOLE geometry cache (sky shapes are NOT in DistantLand::visibleCacheKeys — that's the MSOC
    // world-object drawn set), filtered to isSky entries that have a host slot. SK1 emits ONLY the
    // dome: the untextured vertex-colour shape (isSky && !d3dTexture); SK2 adds the textured shapes
    // (sun/moons/clouds/stars). Camera-relative shift matches the host's translation-free viewProj
    // (the sky is camera-attached). Only runs while Forge owns the frame (F11). Returns the packed
    // item count.
    std::uint32_t buildSkyDrawList() {
        MGE_ZoneScopedN("build:sky");
        if (!g_skyVec) {
            return 0;
        }
        const auto& cacheMap = MGE::GeometryCache::cache();

        g_skyScratch.clear();
        // SK2: gather every isSky entry that has a host slot (dome + textured sun/moons/stars),
        // then sort by skyOrder (MW's back-to-front subtree order) so the alpha-blended shapes
        // layer correctly — the cache is an unordered_map, so we can't rely on iteration order.
        struct SkyCand { std::uint16_t order; const MGE::GeometryCache::CachedGeometry* e; std::uint32_t slot; };
        static std::vector<SkyCand> cands;   // single-threaded; reused frame-to-frame
        cands.clear();
        // Stage 1: iterate the cache's maintained sky membership set instead of scanning the
        // WHOLE cache (this was one of the two remaining full-map scans, cache-size-
        // proportional). Every filter is kept byte-identical — isSky stays as belt-and-braces
        // (membership mirrors it), and eviction being a periodic sweep still means stale
        // entries must be skipped here: a sky shape the walk stopped visiting (moon set,
        // weather change → appCulled) would otherwise keep drawing until the next sweep.
        const auto cacheFrame = MGE::GeometryCache::currentFrame();
        const auto& skySet = MGE::GeometryCache::skyKeys();
        for (std::uint32_t skey : skySet) {
            auto cit = cacheMap.find(skey);
            if (cit == cacheMap.end()) continue;   // set ⊆ cache invariant; guard anyway
            const auto& e = cit->second;
            if (!e.isSky) continue;
            if (e.lastFrame != cacheFrame) continue;   // stale (awaiting eviction sweep)
            auto ks = g_keySlot.find(skey);
            if (ks == g_keySlot.end()) {
                continue;   // not yet uploaded to the host
            }
            cands.push_back({ e.skyOrder, &e, ks->second.slot });
        }
        // Stage 1 drift safety (panel toggle, default off): the old full-cache scan must keep
        // exactly the keys the membership set delivered. Any line here = set-maintenance bug.
        if (g_membershipValidate) {
            for (const auto& kv : cacheMap) {
                if (kv.second.isSky && !skySet.count(kv.first)) {
                    LOG::logline("!! [memb-cmp] sky key=%08X tex=%s isSky in cache but NOT in skyKeys",
                                 kv.first, kv.second.textureName ? kv.second.textureName : "(none)");
                }
            }
            for (std::uint32_t skey : skySet) {
                auto cit = cacheMap.find(skey);
                if (cit == cacheMap.end()) {
                    LOG::logline("!! [memb-cmp] sky key=%08X in skyKeys but NOT in cache", skey);
                } else if (!cit->second.isSky) {
                    LOG::logline("!! [memb-cmp] sky key=%08X in skyKeys but entry !isSky", skey);
                }
            }
        }
        std::sort(cands.begin(), cands.end(),
                  [](const SkyCand& a, const SkyCand& b) { return a.order < b.order; });

        // [sk-diag] night-sky decay: whenever the packed count CHANGES, dump every isSky entry
        // with its freshness (lastFrame vs cacheFrame) and host-slot state — the missing stars/
        // moons must show up here as either absent (walk/eviction), stale, or slot-less (upload
        // chain). Remove after the decay root-cause is fixed.
        {
            // Stage 1: the count loop rides the sky membership set too (it was the SECOND
            // unconditional full-cache scan in this builder).
            static std::size_t s_lastCandCount = (std::size_t)-1;
            std::size_t staleN = 0, noSlotN = 0, totalSky = 0;
            for (std::uint32_t skey : skySet) {
                auto cit = cacheMap.find(skey);
                if (cit == cacheMap.end()) continue;
                const auto& e = cit->second;
                if (!e.isSky) continue;
                ++totalSky;
                if (e.lastFrame != cacheFrame) { ++staleN; continue; }
                if (g_keySlot.find(skey) == g_keySlot.end()) { ++noSlotN; }
            }
            if (cands.size() != s_lastCandCount) {
                s_lastCandCount = cands.size();
                LOG::logline(">> [sk-diag] sky list CHANGED: packed=%zu (cache isSky=%zu stale=%zu noSlot=%zu) cacheFrame=%llu",
                             cands.size(), totalSky, staleN, noSlotN,
                             (unsigned long long)cacheFrame);
                for (std::uint32_t skey : skySet) {
                    auto cit = cacheMap.find(skey);
                    if (cit == cacheMap.end()) continue;
                    const auto& e = cit->second;
                    if (!e.isSky) continue;
                    const bool stale  = (e.lastFrame != cacheFrame);
                    const bool noSlot = (g_keySlot.find(skey) == g_keySlot.end());
                    LOG::logline(">> [sk-diag]   key=%08X tex=%s order=%u vc=%u %s%s",
                                 skey, e.textureName ? e.textureName : "(none)",
                                 (unsigned)e.skyOrder, e.vertexCount,
                                 stale ? "STALE " : "fresh ", noSlot ? "NOSLOT" : "slot-ok");
                }
            }
        }

        IPC::SkyDrawWire item;
        std::uint32_t count = 0;
        for (const auto& c : cands) {
            const auto& e = *c.e;
            item.slot      = c.slot;
            // Dome stays 0 (vertex-colour only → host default white); textured shapes resolve
            // their DDS to a bindless slot via the existing residency path.
            item.texIndex  = e.d3dTexture ? resolveTextureSlot(e.textureName) : 0u;
            item.srcBlend  = e.srcBlend;
            item.destBlend = e.destBlend;
            item.alphaRef  = e.alphaTest ? e.alphaRef : 0.0f;
            // SK2 FFP modulation: material diffuse rgb + per-element alpha fade + vcol routing,
            // exactly as buildDrawList ships for opaques.
            item.matColor[0] = e.matDiffuse[0];
            item.matColor[1] = e.matDiffuse[1];
            item.matColor[2] = e.matDiffuse[2];
            item.matAlpha    = e.matDiffuse[3];
            // SK3 cloud scroll: live-vs-uploaded UV diff from the sky walk (zero for every
            // non-UV-animated shape); sky.vert adds it back to the baked UV.
            item.uvOffset[0] = e.skyUVOffset[0];
            item.uvOffset[1] = e.skyUVOffset[1];
            {
                // SK4 retired the C3 host-gradient dome (vColSource 3): sky vcols now re-ship
                // whenever MW rebakes them, so the dome's REAL per-frame gradient (vcolsrc=2,
                // alpha FF — SKY DUMP-verified) renders through the standard In.Color path,
                // pixel-exact with vanilla instead of the fogColNear->skyZenith approximation
                // (which read slightly bright at the zenith). sky.frag keeps its 3-branch;
                // it's just never selected now.
                item.vColSource = (e.hasVertexColor && e.vColSource != 0) ? e.vColSource : 0u;
            }
            memcpy(item.world, e.worldTransformD3D, 16 * sizeof(float));
            // SK2 billboard fix (SUN ONLY): the sun disc hangs under a NiBillboardNode that MW
            // re-faces to the camera each frame via rotateToCamera — but that runs AFTER our
            // onFrameReady scene walk, so e.worldTransformD3D still holds the billboard's BASE
            // celestial-sphere (tangent) orientation. Replayed verbatim the quad is a flat plate
            // tangent to the sky sphere: round looking straight at it (zenith), squashed at grazing
            // angles (near the horizon). The two-part MOONS are also 4-vert/2-tri textured quads but
            // arrive ALREADY camera-faced in their captured transform (re-billboarding them broke
            // their facing in-game), so leave those alone — gate on the sun's base-texture name
            // ("tx_sun_05" — the moons are tx_masser/tx_secunda/tx_mooncircle, no "sun"). Rebuild a
            // camera-facing basis: preserve position (translation) + per-axis size; orient model
            // +X -> camera right, +Y -> camera up (spherical / full-facing → always round = vanilla).
            bool isSunDisc = false;
            if (e.d3dTexture && e.textureName && e.vertexCount == 4 && e.triangleCount == 2) {
                // case-insensitive substring "sun" (|32 lowercases ASCII letters; loop guard keeps
                // the p[1]/p[2] look-ahead inside the null-terminated string).
                for (const char* p = e.textureName; p[0] && p[1] && p[2]; ++p) {
                    if ((p[0] | 32) == 's' && (p[1] | 32) == 'u' && (p[2] | 32) == 'n') { isSunDisc = true; break; }
                }
            }
            if (isSunDisc) {
                const D3DXMATRIX& V = DistantLand::mwView;  // row-vector view: columns = world camera axes
                const float* m = item.world;
                const float sx = sqrtf(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);     // model +X length (width)
                const float sy = sqrtf(m[4]*m[4] + m[5]*m[5] + m[6]*m[6]);     // model +Y length (height)
                const float sz = sqrtf(m[8]*m[8] + m[9]*m[9] + m[10]*m[10]);   // model +Z length (normal)
                const float Rx = V._11, Ry = V._21, Rz = V._31;   // camera right  (world)
                const float Ux = V._12, Uy = V._22, Uz = V._32;   // camera up     (world)
                const float Fx = V._13, Fy = V._23, Fz = V._33;   // camera forward(world, into scene)
                item.world[0] =  sx*Rx; item.world[1] =  sx*Ry; item.world[2]  =  sx*Rz;   // +X -> right
                item.world[4] =  sy*Ux; item.world[5] =  sy*Uy; item.world[6]  =  sy*Uz;   // +Y -> up
                item.world[8] = -sz*Fx; item.world[9] = -sz*Fy; item.world[10] = -sz*Fz;   // +Z -> toward camera
            }
            // WT2: tell the Forge reflection pass which shape is the sun, so it can re-face the disc
            // for the mirrored view (the world above faces the MAIN camera → squashed once mirrored).
            item.isSunDisc = isSunDisc ? 1u : 0u;
            // CAMERA-RELATIVE: the sky is camera-attached; shift by -eye to match the host's
            // translation-free viewProj (see buildDrawList) and keep vertex math near the origin.
            item.world[12] -= DistantLand::eyePos.x;
            item.world[13] -= DistantLand::eyePos.y;
            item.world[14] -= DistantLand::eyePos.z;
            const std::size_t at = g_skyScratch.size();
            g_skyScratch.resize(at + sizeof(item));
            memcpy(g_skyScratch.data() + at, &item, sizeof(item));
            if (++count >= IPC::kMaxSkyDraws) {
                break;
            }
        }
        return count;
    }

    // ---- FP1a first-person takeover --------------------------------------------------

    // Build a D3D row-major VIEW matrix the way MW's DX8 renderer derives it from a
    // NiCamera basis (D3D camera space: X=right, Y=up, Z=forward — the LookAtLH form),
    // with the camera position already made RELATIVE by the caller (camera-relative
    // pipeline: the emit helpers shift all world translations by -DistantLand::eyePos).
    void buildNiCameraView(const float dir[3], const float up[3], const float right[3],
                           const float eye[3], D3DXMATRIX* m) {
        m->_11 = right[0]; m->_12 = up[0]; m->_13 = dir[0]; m->_14 = 0.0f;
        m->_21 = right[1]; m->_22 = up[1]; m->_23 = dir[1]; m->_24 = 0.0f;
        m->_31 = right[2]; m->_32 = up[2]; m->_33 = dir[2]; m->_34 = 0.0f;
        m->_41 = -(right[0] * eye[0] + right[1] * eye[1] + right[2] * eye[2]);
        m->_42 = -(up[0]    * eye[0] + up[1]    * eye[1] + up[2]    * eye[2]);
        m->_43 = -(dir[0]   * eye[0] + dir[1]   * eye[1] + dir[2]   * eye[2]);
        m->_44 = 1.0f;
    }

    // Build the D3D projection MW's scenes ACTUALLY rasterize with. fov/aspect come
    // from WorldControllerRenderCamera::CameraData {fovDegrees, near, far, viewportW,
    // viewportH}: fovDegrees is the HORIZONTAL field of view (proj._11 = cot(fov/2),
    // proj._22 = _11 * w/h), symmetric frustum — in-game validated element-exact
    // against mwProj. (The NiCamera's Gamebryo viewFrustum is NOT the projection
    // authority — it carries different (MGE-patched cull) values: FOV≈88°/near=1 vs
    // the real proj's 75°.) The z terms are NOT CameraData's near/far either: the
    // proxy re-edits every main-view projection (DistantLand::setProjection under
    // USE_DISTANT_LAND: near 4, far = DrawDist·cell), and the arm scene rides the
    // SAME edit (isMainView stays true there) — so copy _33/_43 from the LIVE
    // device-captured mwProj, which holds exactly the post-edit mapping.
    void buildNiCameraProj(const float cd[5], D3DXMATRIX* m) {
        const float fovRad = cd[0] * (3.14159265358979323846f / 180.0f);
        const float aspect = (cd[4] > 0.0f) ? (cd[3] / cd[4]) : 1.0f;
        memset(m, 0, sizeof(*m));
        m->_11 = 1.0f / tanf(fovRad * 0.5f);
        m->_22 = m->_11 * aspect;
        m->_33 = DistantLand::mwProj._33;
        m->_34 = 1.0f;
        m->_43 = DistantLand::mwProj._43;
    }

    // Build the ARM scene's projection the way the engine does for THAT scene: from the
    // live NiCamera viewFrustum (l/r/t/b = plane slopes at unit distance, + n/f), then
    // the same proxy z-edit every main-view projection submission receives
    // (DistantLand::setProjection — a no-op with DL off). In-game latch diag 2026-07-11:
    // MW's submitted arm proj is EXACTLY frustum-form (_11 = 1/r = 1.03553 → 88°
    // horizontal, _22 = 1/t, z = the frustum's n=1/f=7168 mapping) while its view
    // matched ours to the last digit — the arm scene genuinely draws WIDER than the
    // world scene's CameraData 75°, which is why CameraData-built arms rendered ~1.26x
    // too large (the "mirrored offset from center"). Off-center terms kept general.
    bool buildArmCameraProj(D3DXMATRIX* m) {
        float fr[6], port[4];
        if (!MWBridge::get()->getRenderCameraFrustum(1, fr, port)) {
            return false;
        }
        const float l = fr[0], r = fr[1], t = fr[2], b = fr[3], n = fr[4], f = fr[5];
        if (r - l <= 0.0f || t - b <= 0.0f || f - n <= 0.0f) {
            return false;
        }
        memset(m, 0, sizeof(*m));
        m->_11 = 2.0f / (r - l);
        m->_22 = 2.0f / (t - b);
        m->_31 = -(r + l) / (r - l);
        m->_32 = -(t + b) / (t - b);
        m->_33 = f / (f - n);
        m->_43 = -n * f / (f - n);
        m->_34 = 1.0f;
        DistantLand::setProjection((D3DMATRIX*)m);
        return true;
    }

    // FP camera ground-truth diagnostic (offset-from-center triage). The proxy arms this
    // at the z-only clear before MW's first-person scene (noteFPZClear) and the FIRST
    // view + proj MW submits afterwards latch here — the exact matrices the native arms
    // rasterize with. buildFPFrame logs them against the built fpView/fpProj so the log
    // answers "fov, scale, or offset?" directly. Diagnostic only; nothing consumes the
    // latched values. Requires FP suppression OFF (MW must actually render its arm scene).
    bool       g_fpDiagArmed = false;
    bool       g_fpDiagHaveView = false;
    bool       g_fpDiagHaveProj = false;
    D3DXMATRIX g_fpDiagView;
    D3DXMATRIX g_fpDiagProj;
    unsigned   g_fpDiagAge = 0;          // frames since the last complete latch

    // FP validation gate: rebuild the MAIN camera from engine state with the same code
    // and diff against the proxy-captured DistantLand::mwView/mwProj. Retires the
    // axis-convention risk before any FP frame ships — on mismatch the FP pass refuses
    // to enable (throttled log says why). Latches after the first clean pass.
    bool g_fpCamValid = false;
    bool validateFPCameraMath() {
        if (g_fpCamValid) {
            return true;
        }
        float pos[3], dir[3], up[3], right[3], cd[5];
        if (!MWBridge::get()->getRenderCameraState(0, pos, dir, up, right, cd)) {
            return false;
        }
        D3DXMATRIX view, proj;
        buildNiCameraView(dir, up, right, pos, &view);   // absolute eye == mwView's own form
        buildNiCameraProj(cd, &proj);
        const float* a  = (const float*)&view;
        const float* av = (const float*)&DistantLand::mwView;
        const float* b  = (const float*)&proj;
        const float* bv = (const float*)&DistantLand::mwProj;
        float maxV = 0.0f, maxP = 0.0f;
        for (int i = 0; i < 16; ++i) {
            maxV = std::max(maxV, fabsf(a[i] - av[i]) / std::max(1.0f, fabsf(av[i])));
            maxP = std::max(maxP, fabsf(b[i] - bv[i]) / std::max(1.0f, fabsf(bv[i])));
        }
        if (maxV < 1e-3f && maxP < 1e-3f) {
            g_fpCamValid = true;
            LOG::logline(">> [fp] camera math VALIDATED (viewErr=%.2e projErr=%.2e) — FP pass may ship", maxV, maxP);
            return true;
        }
        static unsigned s_n = 0;
        if (s_n++ % 300 == 0) {
            LOG::logline("!! [fp] camera validation MISMATCH (viewErr=%.4f projErr=%.4f) — FP pass held off", maxV, maxP);
            LOG::logline("   built view r0 % .5f % .5f % .5f | mw % .5f % .5f % .5f", a[0], a[1], a[2], av[0], av[1], av[2]);
            LOG::logline("   built view r3 % .2f % .2f % .2f | mw % .2f % .2f % .2f", a[12], a[13], a[14], av[12], av[13], av[14]);
            LOG::logline("   built proj    % .5f % .5f % .5f % .2f | mw % .5f % .5f % .5f % .2f",
                         b[0], b[5], b[10], b[14], bv[0], bv[5], bv[10], bv[14]);
        }
        return false;
    }

    // Build the FP draw lists (rigid + skinned) from the isFP entries the FP walk
    // stamped THIS frame, plus the FP viewProj from the arm camera (camera-relative
    // against the SAME eye the emit helpers subtract, then the arm scene's own
    // projection — its FOV/near/far differ from the world camera's). Returns true
    // when the FP pass has something to ship this frame.
    bool buildFPFrame(IPC::FPFrame& fp, std::uint32_t& fpDraws, std::uint32_t& fpSkinned,
                      std::uint32_t& fpAlpha) {
        MGE_ZoneScopedN("build:fp");
        fpDraws = 0;
        fpSkinned = 0;
        fpAlpha = 0;
        g_fpDrawScratch.clear();
        g_fpSkinnedScratch.clear();
        g_fpAlphaScratch.clear();
        if (!RenderProcess::wantsFPCapture() || !g_fpDrawVec || !g_fpSkinnedVec) {
            return false;
        }
        if (!validateFPCameraMath()) {
            return false;
        }

        float pos[3], dir[3], up[3], right[3], cd[5];
        if (!MWBridge::get()->getRenderCameraState(1, pos, dir, up, right, cd)) {
            return false;
        }
        const float rel[3] = { pos[0] - DistantLand::eyePos.x,
                               pos[1] - DistantLand::eyePos.y,
                               pos[2] - DistantLand::eyePos.z };
        D3DXMATRIX view, proj, viewProj;
        buildNiCameraView(dir, up, right, rel, &view);
        if (!buildArmCameraProj(&proj)) {
            return false;
        }
        D3DXMatrixMultiply(&viewProj, &view, &proj);
        memcpy(fp.viewProj, &viewProj, 16 * sizeof(float));

        // FP1c fp-alpha candidates: blended FP parts (torch flame, enchant glow) are
        // collected during the loop, sorted back-to-front against the ARM camera, then
        // packed — same collect/sort/emit shape as the world alpha list (AT1). Same
        // representability rules too: single-map, non-skinned, textured (a textureless
        // blend has no base to composite; skinned/multimap blends aren't expressible in
        // AlphaDrawWire and are skipped exactly as before).
        struct FPAlphaCand {
            float depth;
            SlotInfo* si;
            const MGE::GeometryCache::CachedGeometry* e;
        };
        static std::vector<FPAlphaCand> s_fpAlphaCands;   // single-threaded; reused frame-to-frame
        s_fpAlphaCands.clear();

        const auto& cacheMap = MGE::GeometryCache::cache();
        const auto cacheFrame = MGE::GeometryCache::currentFrame();
        // Stage 1: iterate the cache's maintained FP membership set instead of scanning the
        // whole cache (the last remaining full-map scan on the build path). isFP stays as
        // belt-and-braces; every other filter unchanged. FP opaque emit order is
        // order-insensitive and fp-alpha is depth-sorted below, so set iteration order is fine.
        const auto& fpSet = MGE::GeometryCache::fpKeys();
        if (g_membershipValidate) {
            // Stage 1 drift safety (panel toggle, default off) — mirror of the sky check.
            for (const auto& kv : cacheMap) {
                if (kv.second.isFP && !fpSet.count(kv.first)) {
                    LOG::logline("!! [memb-cmp] fp key=%08X tex=%s isFP in cache but NOT in fpKeys",
                                 kv.first, kv.second.textureName ? kv.second.textureName : "(none)");
                }
            }
            for (std::uint32_t fkey : fpSet) {
                auto cit = cacheMap.find(fkey);
                if (cit == cacheMap.end()) {
                    LOG::logline("!! [memb-cmp] fp key=%08X in fpKeys but NOT in cache", fkey);
                } else if (!cit->second.isFP) {
                    LOG::logline("!! [memb-cmp] fp key=%08X in fpKeys but entry !isFP", fkey);
                }
            }
        }
        for (std::uint32_t fkey : fpSet) {
            auto cit = cacheMap.find(fkey);
            if (cit == cacheMap.end()) continue;   // set ⊆ cache invariant; guard anyway
            const auto& e = cit->second;
            if (!e.isFP || e.lastFrame != cacheFrame) continue;
            // Multi-map FP parts would need the wide-VB multimap pipeline in the FP pass;
            // none expected on arms — skip rather than bind the wrong vertex layout.
            if (e.d3dDark || e.d3dDetail || e.d3dGlow) continue;
            auto ks = g_keySlot.find(fkey);
            if (ks == g_keySlot.end()) continue;   // not uploaded yet (first-sight frame)
            if (e.blendEnable && !e.isSkinned) {
                if (!e.d3dTexture) continue;   // textureless blend: nothing to composite
                // Depth key: bound-center view depth against the ARM camera (pos/dir from
                // getRenderCameraState above) — MW's sorter criterion, FP camera's frame.
                const float* w = e.worldTransformD3D;
                const float cx = e.boundsCenter[0], cy = e.boundsCenter[1], cz = e.boundsCenter[2];
                const float wx = cx * w[0] + cy * w[4] + cz * w[8]  + w[12] - pos[0];
                const float wy = cx * w[1] + cy * w[5] + cz * w[9]  + w[13] - pos[1];
                const float wz = cx * w[2] + cy * w[6] + cz * w[10] + w[14] - pos[2];
                s_fpAlphaCands.push_back({ wx * dir[0] + wy * dir[1] + wz * dir[2],
                                           &ks->second, &e });
                continue;
            }
            if (e.isSkinned) {
                // Skinned blends stay dropped (as in FP1a) — the skinned pipeline has no
                // blend state, so drawing them opaquely would be wrong, not better.
                if (e.blendEnable) continue;
                emitSkinnedDraw(ks->second, e, fpSkinned, g_fpSkinnedScratch);
            } else {
                // Same textureless rule as dispatch: collision proxies drop; genuine
                // textureless visuals draw opaque on slot 0 (host default white).
                if (!e.d3dTexture && e.isPickRoot) continue;
                emitStaticDraw(ks->second, e, fpDraws, g_fpDrawScratch);
            }
        }

        // FP1c: back-to-front (descending view depth), then pack into the FP alpha scratch.
        if (!s_fpAlphaCands.empty()) {
            std::sort(s_fpAlphaCands.begin(), s_fpAlphaCands.end(),
                      [](const FPAlphaCand& a, const FPAlphaCand& b) { return a.depth > b.depth; });
            for (const auto& c : s_fpAlphaCands) {
                emitAlphaDraw(*c.si, *c.e, fpAlpha, g_fpAlphaScratch);
            }
        }

        // FP1c particles (torch flame, enchant sparks): the cache walk billboarded each FP
        // particle system's live particles against the arm camera. Ship the quads through the
        // shared captured-alpha VB/IB (same buffers the world captured path uses; the host binds
        // them for kAlphaSlotCaptured items) and emit one FP alpha item per system. Appended to
        // g_capVertScratch AFTER buildGeometryDrawLists emitted the world captured items (their
        // bases already fixed), BEFORE the captured blob ships — so the FP quads ride this frame's
        // blob and the FP items' bases point past the world region. See tasks/forge-fp-particles.md.
        {
            const auto& precs = MGE::GeometryCache::fpParticleRecs();
            if (!precs.empty()) {
                std::uint32_t pvCount = 0, piCount = 0;
                const IPC::GeomVertexWire* pv =
                    (const IPC::GeomVertexWire*)MGE::GeometryCache::fpParticleVerts(pvCount);
                const std::uint16_t* pi =
                    (const std::uint16_t*)MGE::GeometryCache::fpParticleIndices(piCount);
                for (const auto& rec : precs) {
                    if (!rec.indexCount || rec.vertexBase + rec.vertexCount > pvCount
                        || rec.indexBase + rec.indexCount > piCount) continue;
                    if (g_capVertScratch.size() + rec.vertexCount > IPC::kMaxCapturedAlphaVerts
                        || g_capIdxScratch.size() + rec.indexCount > IPC::kMaxCapturedAlphaIndices)
                        break;
                    const std::uint32_t vBase = (std::uint32_t)g_capVertScratch.size();
                    const std::uint32_t iBase = (std::uint32_t)g_capIdxScratch.size();
                    g_capVertScratch.insert(g_capVertScratch.end(),
                                            pv + rec.vertexBase, pv + rec.vertexBase + rec.vertexCount);
                    for (std::uint32_t j = 0; j < rec.indexCount; ++j)
                        g_capIdxScratch.push_back(pi[rec.indexBase + j]);   // 0-based; host adds vertexBase

                    std::uint32_t texIndex = 0;
                    auto mit = g_capTexMemo.find(rec.texture);
                    if (mit != g_capTexMemo.end()) texIndex = mit->second;
                    else {
                        const char* name = rec.texture ? MGE::GeometryCache::resolveTextureName(rec.texture) : nullptr;
                        texIndex = name ? resolveTextureSlot(name) : 0u;
                        g_capTexMemo.emplace(rec.texture, texIndex);
                    }

                    IPC::AlphaDrawWire item{};
                    item.slot      = IPC::kAlphaSlotCaptured;
                    item.texIndex  = texIndex;
                    item.srcBlend  = rec.srcBlend;
                    item.destBlend = rec.destBlend;
                    item.alphaRef  = 0.0f;
                    // Two particle regimes, matching MW's FFP (which clamps the lit vertex colour to
                    // [0,1] BEFORE modulating the texture):
                    //   * Self-illuminated flame — NIF material emissive ~ (1,1,1). MW's clamp pins
                    //     the vertex colour to white, so the flame renders as PURE albedo, immune to
                    //     world lights (a blue point light can't tint it). We reproduce that with
                    //     vColSource=1 (lit = MatDiffuse*d + MatAmbient*a + Color.rgb) + zeroed
                    //     MatDiffuse/MatAmbient → d (incl. point lights) and a drop out → lit =
                    //     Color.rgb, and buildFPParticleQuads forces the flame vertex RGB to white →
                    //     lit = (1,1,1) → c = albedo. Per-particle ALPHA still drives the fade.
                    //   * Smoke — emissive 0. vColSource=2 (Color*(d+a)) so it picks up the scene's
                    //     sun+ambient like MW (values stay < 1, so the missing clamp is invisible).
                    // NOTE (accepted deviation, 2026-07-19): real MW DOES let a flame go faintly pale-
                    // blue when a blue point light is very close (its clamp saturates only most of the
                    // way, not fully). We render the flame fully light-immune instead — the user
                    // prefers this look. If exact MW parity is ever wanted, replace this with a real
                    // FFP-clamp path (saturate(Color*(d+a)+MatEmissive)) in alpha.frag. See also the
                    // emissive-lantern regression follow-up in tasks/forge-fp-particles.md.
                    const bool emissiveSat = (rec.matEmissive[0] > 0.5f || rec.matEmissive[1] > 0.5f
                                              || rec.matEmissive[2] > 0.5f);
                    item.matDiffuse[0] = item.matDiffuse[1] = item.matDiffuse[2] = 0.0f;
                    item.matAlpha      = 1.0f;
                    item.matAmbient[0] = item.matAmbient[1] = item.matAmbient[2] = 0.0f;
                    item.matEmissive[0] = item.matEmissive[1] = item.matEmissive[2] = 0.0f;
                    item.vColSource = emissiveSat ? 1u : 2u;
                    // Quads are already in absolute world space → identity world minus the
                    // camera-relative eye (matches the world captured path + FP emit helpers).
                    memset(item.world, 0, sizeof(item.world));
                    item.world[0] = item.world[5] = item.world[10] = item.world[15] = 1.0f;
                    item.world[12] = -DistantLand::eyePos.x;
                    item.world[13] = -DistantLand::eyePos.y;
                    item.world[14] = -DistantLand::eyePos.z;
                    item.vertexBase = vBase;
                    item.indexBase  = iBase;
                    item.indexCount = rec.indexCount;
                    item.cullFlags  = IPC::kAlphaCullTwoSided;
                    item.clampMode  = IPC::kTexWrapSWrapT;
                    const std::size_t at = g_fpAlphaScratch.size();
                    g_fpAlphaScratch.resize(at + sizeof(item));
                    memcpy(g_fpAlphaScratch.data() + at, &item, sizeof(item));
                    ++fpAlpha;
                }
            }
        }

        static unsigned s_hb = 0;
        if (s_hb++ % 300 == 0) {
            LOG::logline(">> [fp] draws=%u skinned=%u alpha=%u | cam fov=%.1f near=%.1f far=%.1f vp=%.0fx%.0f",
                         fpDraws, fpSkinned, fpAlpha, cd[0], cd[1], cd[2], cd[3], cd[4]);
        }

        // Continuous arm-camera validation: diff the latched native-arm-scene matrices
        // (see noteFPZClear/noteFPSceneTransform) against what we built. The latch is
        // from LAST frame's FP scene (MW renders it after this kickoff), so only compare
        // a FRESH latch (age<=2 — a stale one means suppression is on / the camera moved
        // since). One PASS line on first agreement; full dump (throttled) on mismatch.
        // built view uses the ABSOLUTE eye to match MW's own form.
        ++g_fpDiagAge;
        if (g_fpDiagHaveView && g_fpDiagHaveProj && g_fpDiagAge <= 2) {
            D3DXMATRIX absView;
            buildNiCameraView(dir, up, right, pos, &absView);
            const float* mp = (const float*)&g_fpDiagProj;
            const float* bp = (const float*)&proj;
            const float* mv = (const float*)&g_fpDiagView;
            const float* bv = (const float*)&absView;
            float errP = 0.0f, errV = 0.0f;
            for (int i = 0; i < 16; ++i) {
                errP = std::max(errP, fabsf(bp[i] - mp[i]) / std::max(1.0f, fabsf(mp[i])));
                errV = std::max(errV, fabsf(bv[i] - mv[i]) / std::max(1.0f, fabsf(mv[i])));
            }
            static bool s_passLogged = false;
            if (errP < 1e-3f && errV < 2e-2f) {          // view tol loose: latch is 1 frame old
                if (!s_passLogged) {
                    s_passLogged = true;
                    LOG::logline(">> [fp cam-diag] arm camera MATCHES native arm scene (projErr=%.2e viewErr=%.2e)", errP, errV);
                }
            } else {
                static unsigned s_diag = 0;
                if (s_diag++ % 300 == 0) {
                    LOG::logline("!! [fp cam-diag] arm-scene MISMATCH (projErr=%.4f viewErr=%.4f, age=%u)", errP, errV, g_fpDiagAge);
                    LOG::logline("   proj mw   _11=%.5f _22=%.5f _31=%.5f _32=%.5f _33=%.6f _43=%.3f",
                                 mp[0], mp[5], mp[8], mp[9], mp[10], mp[14]);
                    LOG::logline("   proj built _11=%.5f _22=%.5f _31=%.5f _32=%.5f _33=%.6f _43=%.3f",
                                 bp[0], bp[5], bp[8], bp[9], bp[10], bp[14]);
                    LOG::logline("   view mw    r0=(% .5f % .5f % .5f) t=(% .2f % .2f % .2f)",
                                 mv[0], mv[1], mv[2], mv[12], mv[13], mv[14]);
                    LOG::logline("   view built r0=(% .5f % .5f % .5f) t=(% .2f % .2f % .2f)",
                                 bv[0], bv[1], bv[2], bv[12], bv[13], bv[14]);
                }
            }
        }
        return (fpDraws + fpSkinned) > 0;
    }
}

namespace RenderProcess {
    void init(IPC::Client* client, IDirect3DDevice9* device) {
        if (!Configuration.UseRenderProcess) {
            return;
        }
        g_client = client;
        g_fpSuppressLive = Configuration.ForgeFPSuppress;   // FP1b seed; numpad-/ flips live
        g_frameAheadLive = Configuration.ForgeFrameAhead;   // frame-ahead seed; numpad-* flips live
        if (!device) {
            LOG::logline("!! [seam] no device at init; seam disabled");
            return;
        }
        // Bring up the seam NOW, under the "...MGE XE..." loading bar (this runs inside
        // DistantLand::init()), so the shared texture is live by the main menu.
        lazyInit(device);
    }

    // --- Device-reset seam hooks (see header) -----------------------------------------
    void preDeviceReset(IDirect3DDevice9* device) {
        if (!g_initOk) {
            return;
        }
        // Quiesce the seam BEFORE the real device resets. collectDeferredFinish drains any
        // in-flight produce and finishes+copies a pending (frame-ahead) host frame; the copy
        // path signals+waits g_fence inline, so on return no host GPU work is in flight. The
        // extra waitProduce is belt-and-braces (idempotent no-op once collect has drained).
        collectDeferredFinish(device);
        waitProduce();
        LOG::logline(">> [seam] preDeviceReset: host frame drained, produce worker idle");
    }

    void postDeviceReset(IDirect3DDevice9* device) {
        if (!g_initOk || !device) {
            return;
        }
        // Re-read the new backbuffer exactly as lazyInit does, then re-derive the render size.
        // g_w/g_h (the fixed 2x allocation) and the imported host image persist across ResetEx,
        // so there is NO host RPC and NO seam-surface rebuild — just follow the new backbuffer.
        const UINT oldW = g_bbW, oldH = g_bbH;
        IDirect3DSurface9* bb = nullptr;
        if (SUCCEEDED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &bb)) && bb) {
            D3DSURFACE_DESC sd = {};
            if (SUCCEEDED(bb->GetDesc(&sd)) && sd.Width && sd.Height) {
                g_bbW = sd.Width;
                g_bbH = sd.Height;
            }
            bb->Release();
        }
        if (g_bbW > g_w || g_bbH > g_h) {
            // New backbuffer exceeds the 2x launch-time allocation (launched low, jumped high).
            // recomputeRenderSize clamps the render size to the alloc — mild supersample-down.
            LOG::logline("!! [seam] new backbuffer %ux%u exceeds alloc %ux%u — render clamped to alloc (supersample-down)",
                         g_bbW, g_bbH, g_w, g_h);
        }
        recomputeRenderSize();   // clamps g_rw/g_rh to the alloc, restamps the host render size
        LOG::logline(">> [seam] device reset %ux%u -> %ux%u (alloc %ux%u, render %ux%u, scale %.2fx)",
                     oldW, oldH, g_bbW, g_bbH, g_w, g_h, g_rw, g_rh, g_renderScale);
    }

    void onDeviceResetFailed() {
        // The real device is in the lost/hung state: drop the last host frame so neither the
        // deferred blit nor the same-frame composite lays a stale image while MW retries. g_initOk
        // stays true (Ex persists the seam's resources); a later successful reset recovers via
        // postDeviceReset. The F11 composite toggle (g_enabled) is left untouched.
        g_mainTexValid = false;
        LOG::logline("!! [seam] device reset failed — composite suspended until next successful reset");
    }


    // --- Finish-half pieces -----------------------------------------------------------
    // The composite finish is split into once-per-frame pieces so frame-ahead pipelining
    // (ForgeFrameAhead) can run them at different points: dev-key poll + finish/copy at
    // the NEXT frame's BeginScene(0) collect, the composite blit alone at EndScene(0).
    // The non-deferred path (onStage0CompositeFinish) runs all of them back to back —
    // behaviour identical to before the split.

    // Poll the seam's edge-triggered dev keys, once per display frame (Present-serial
    // dedup — collect and finish can both run in one frame during mode transitions, and
    // a double edge-poll would eat or double-flip a keypress). Deferred mode polls at
    // the BeginScene(0) collect — BEFORE the earlyForgeKickoff latch and every
    // suppression gate, so all per-frame consumers see one consistent value; on
    // non-deferred frames the finish still polls (whichever runs first this frame wins).
    void pollDevKeys() {
        if (!g_initOk) {
            return;
        }
        if (g_lastPollSerial == g_frameSerial) {
            return;     // already polled this display frame
        }
        g_lastPollSerial = g_frameSerial;
        // F11: live composite toggle.
        if (GetAsyncKeyState(VK_F11) & 0x0001) {
            g_enabled = !g_enabled;
            LOG::logline(">> [seam] composite %s", g_enabled ? "ON" : "OFF");
        }
        // F12 diagnostic: scatter each object by a fixed per-slot offset so any object that
        // is drawn more than once appears as TWO separated copies of the same mesh (a single
        // draw just looks displaced). Reveals duplicate draws regardless of source.
        if (GetAsyncKeyState(VK_F12) & 0x0001) {
            // Tier 2 GTAO added 3 (AO) + 4 (bent normal); dev panel added 5 (albedo) 6 (lit)
            // 7 (ambient) shading-isolation views; 8 (world normal) 9 (point-light count);
            // P1 shadows added 10 (shadow mask — the host panel's face-id/atlas checkboxes
            // pick what it displays); shadow observability added 11 (shadow-atlas static) +
            // 12 (shadow-atlas dynamic) fullscreen atlas blits — cycle is now %13.
            g_debugMode = (g_debugMode + 1) % 13;
            const char* name = (g_debugMode == 1) ? "DEPTH" : (g_debugMode == 2) ? "SCATTER"
                             : (g_debugMode == 3) ? "AO" : (g_debugMode == 4) ? "BENT NORMAL"
                             : (g_debugMode == 5) ? "ALBEDO" : (g_debugMode == 6) ? "LIT"
                             : (g_debugMode == 7) ? "AMBIENT" : (g_debugMode == 8) ? "WORLD NORMAL"
                             : (g_debugMode == 9) ? "LIGHT COUNT"
                             : (g_debugMode == 10) ? "SHADOW MASK"
                             : (g_debugMode == 11) ? "SHADOW ATLAS (STATIC)"
                             : (g_debugMode == 12) ? "SHADOW ATLAS (DYN)" : "NORMAL";
            LOG::logline(">> [seam] debug mode %d (%s)", g_debugMode, name);
        }
        // F9 toggles the in-host dev overlay.
        if (GetAsyncKeyState(VK_F9) & 0x0001) {
            g_devUiVisible = !g_devUiVisible;
            LOG::logline(">> [seam] dev overlay %s", g_devUiVisible ? "ON" : "OFF");
        }
        // F8: one-shot host compute-shader hot-reload — rebuild gtao/linearize from the
        // dxil on disk (recompile + redeploy first). Latched into the NEXT kickoff's
        // DevInput. Independent of panel visibility.
        if (GetAsyncKeyState(VK_F8) & 0x0001) {
            g_reloadShadersPending = true;
            LOG::logline(">> [seam] compute shader hot-reload requested (F8)");
        }
        // Numpad -: perf A/B for the baked distant point-light loop. Latched here (edge), flipped
        // host-side in the next kickoff's DevInput so the gpu-split `dl=` bracket can be diffed
        // on/off in a heavy night scene (the clustered-vs-forward cost measurement).
        if (GetAsyncKeyState(VK_SUBTRACT) & 0x0001) {
            g_distLightsTogglePending = true;
            LOG::logline(">> [seam] distant-light A/B toggle requested (numpad -)");
        }
        // FP1b: numpad-/ toggles MW first-person arm suppression live (A/B of MW arms
        // over the host FP pass vs host arms alone). Only takes effect while the FP
        // pass ships (wantsFPSuppression gates on capture + camera validation).
        if (GetAsyncKeyState(VK_DIVIDE) & 0x0001) {
            g_fpSuppressLive = !g_fpSuppressLive;
            LOG::logline(">> [seam] FP suppression (FP1b) %s", g_fpSuppressLive ? "ON" : "OFF");
        }
        // Numpad *: frame-ahead pipelining live A/B (ForgeFrameAhead). Takes effect at the
        // next kickoff; a pending deferred frame still collects normally (the collect keys
        // on g_kick.deferFinish, not this flag), so the toggle can never wedge the window.
        if (GetAsyncKeyState(VK_MULTIPLY) & 0x0001) {
            g_frameAheadLive = !g_frameAheadLive;
            LOG::logline(">> [seam] frame-ahead pipelining %s", g_frameAheadLive ? "ON" : "OFF");
        }
        // NUMPAD8: produce-worker mode cycle OFF -> FENCED (Tier 1b) -> OVERLAP (Tier 2) ->
        // PARK (mode 3, fire-at-frame-start) -> OFF. FENCED runs the produce on the worker but
        // waits immediately (serial A/B); OVERLAP defers the wait to EndScene(0) so it overlaps
        // scene-0 draw; PARK builds frame N on the worker without firing and the main thread
        // fires it at the START of frame N+1 with a restamped camera (frame = max, not sum).
        // Inert on the legacy UseRenderThread path (dispatcher gates it out). Takes effect at
        // the next kickoff.
        if (GetAsyncKeyState(VK_NUMPAD8) & 0x0001) {
            g_produceMode = (g_produceMode + 1) % 4;
            const char* name = (g_produceMode == 1) ? "FENCED (Tier 1b)"
                             : (g_produceMode == 2) ? "OVERLAP (Tier 2)"
                             : (g_produceMode == 3) ? "PARK (fire-at-frame-start)" : "OFF (inline)";
            LOG::logline(">> [seam] produce worker: %s%s", name,
                         (g_produceMode != 0 && Configuration.UseRenderThread) ? " (inert — UseRenderThread owns the worker)" : "");
        }
        // VK_SCROLL (Scroll Lock): Phase 1 host-cull-only A/B. ON routes the Forge produce
        // off the engine MSOC classify onto the self-contained frustum-only visible set
        // (full refresh walk + whole-cache frustum cull; host Hi-Z GPU cull owns occlusion).
        // OFF = the known-good MSOC/live-draw-build path. Takes effect next frame
        // (frameSetupEarly reads liveDrawBuild; buildFrustumVisibleSet reads the branch gate).
        if (GetAsyncKeyState(VK_SCROLL) & 0x0001) {
            DistantLand::hostCullOnly = !DistantLand::hostCullOnly;
            LOG::logline(">> [seam] host-cull-only (Phase 1, frustum set) %s",
                         DistantLand::hostCullOnly ? "ON" : "OFF");
        }

        // (MW-ONLY-UI world suppression has NO key: the keyspace is full — numpad - is already
        // the distant-light A/B a few lines above, and binding it there made one press fire both.
        // It lives in the Forge Dev imgui panel instead, where its level is also readable.
        // The build-shrink toggles — [bspike] spike log, sky/FP membership validation, the
        // first-sight capture budget — live there too: the free numpad keys collide with the
        // water-flow handlers' GetAsyncKeyState edge bits when UseWaterFlowMap is on, and a
        // panel checkbox is READABLE state besides.)
    }

    // Consume the pending host RenderFrame: drain the completion (or the stored mid-walk
    // early-finish result) and copy the shared RT into g_mainTex. Shared by the finish
    // (same frame) and the frame-ahead collect (next frame). Updates g_mainTexValid —
    // the deferred-blit gate; on failure the blit is skipped, which is exactly today's
    // "no composite this frame" failure behaviour.
    struct FinishResult {
        bool consumed;      // a pending / early-finished frame existed
        bool ok;            // finish AND copy both succeeded
        double hostMs, overlap, tWait0, tRender, tCopy;
    };
    // ks = the kickoff state being finished: g_kick for a same-frame (fused/async) finish,
    // g_pendingFinish for a frame-ahead deferred finish — by then the worker owns g_kick.
    FinishResult finishAndCopy(KickState& ks) {
        FinishResult r = {};
        const bool earlyFinished = ks.rpcEarlyFinished;   // mid-walk drain closed the window
        if (!ks.rpcPending && !earlyFinished) {
            return r;
        }
        ks.rpcPending       = false;
        ks.rpcEarlyFinished = false;
        r.consumed = true;

        // overlap = the MW frame work the host render ran under. In fused mode (Finish
        // called straight after Kickoff) this is ~0 and every bucket reduces to the old
        // serial breakdown — the exact A/B. Async ≈ scene 0; deferred collect ≈ the whole
        // previous MW frame. An early finish ends the overlap at the drain.
        r.tWait0 = nowMs();
        r.overlap = (earlyFinished ? ks.tEarlyFinish : r.tWait0) - ks.tKick;

        IPC::HostFrameTimings hostT{};
        if (earlyFinished) {
            r.ok     = ks.earlyOk;
            r.hostMs = ks.earlyHostMs;
            hostT    = ks.earlyHostTimings;   // captured at the mid-walk drain, not re-read here
        } else {
            MGE_ZoneScopedN("Forge renderSceneFinish (host wait)");
            markMainPhase(MP_FINISH_COPY);   // main is now blocked on the host IPC finish
            r.ok = g_client->renderSceneFinish(&r.hostMs, &hostT);
        }
        r.tRender = nowMs();
        // Split the wait: hostMs = host self-timed cost; (residual wait + kickoff cost - hostMs)
        // = IPC/sync/host-present overhead. With the finish deferred to the next frame's
        // collect, the residual wait collapsing toward 0 is the whole point of the pipeline.
        MGE_TracyPlot("Forge client wait ms", r.tRender - r.tWait0);
        MGE_TracyPlot("Forge overlap ms", r.overlap);
        MGE_TracyPlot("Forge host ms", r.hostMs);
        // Host-inflight lane: the paired 1.0 is set when the kickoff RPC is issued — the
        // step plot spans the host's whole render window across the frame boundary, the
        // thing per-thread zones can't show (the host is another process). The fiber zone
        // below turns that same window into a box on the "Forge Host Frame (inflight)" lane.
        if (g_hostZoneOpen) { MGE_TracyHostFrameEnd(g_hostZoneCtx); g_hostZoneOpen = false; }
        MGE_TracyPlot("Forge host inflight", 0.0);
        // Tier 1 host CPU/GPU split (tasks/forge-host-gpu-lane.md). The inflight box above is the
        // whole RPC round trip, NOT GPU time — these plots are what tell you which half of it you
        // are actually looking at. "host GPU frame ms" is whole-command-buffer GPU execution: if
        // it sits far below the box width, the frame is host-CPU/serial-bound and shrinking shader
        // work buys little. Names carry (N-1) because under ForgeFrameAhead the drained frame is
        // one behind MW's current frame — the plot point lands on the frame that CONSUMED it.
        g_lastHostTimings = hostT;
        MGE_TracyPlot("host GPU frame ms (N-1)",  (double)hostT.gpuFrameMs);
        MGE_TracyPlot("host CPU setup ms (N-1)",  (double)hostT.cpuSetupMs);
        MGE_TracyPlot("host CPU cull ms (N-1)",   (double)hostT.cpuCullMs);
        MGE_TracyPlot("host CPU record ms (N-1)", (double)hostT.cpuRecordMs);
        MGE_TracyPlot("host CPU post ms (N-1)",   (double)hostT.cpuPostMs);
        MGE_TracyPlot("host GPU wait ms (N-1)",   (double)hostT.gpuWaitMs);
        MGE_TracyPlot("host total ms (N-1)",      (double)hostT.totalMs);
        // Per-pass GPU, for locating cost once the frame total says the GPU is worth attacking.
        MGE_TracyPlot("host GPU: cull ms",      (double)hostT.gpuCullMs);
        MGE_TracyPlot("host GPU: prepass ms",   (double)hostT.gpuPrepassMs);
        MGE_TracyPlot("host GPU: shadow ms",    (double)hostT.gpuShadowMs);
        MGE_TracyPlot("host GPU: postdepth ms", (double)hostT.gpuPostDepthMs);
        MGE_TracyPlot("host GPU: reflect ms",   (double)hostT.gpuReflectMs);
        MGE_TracyPlot("host GPU: color ms",     (double)hostT.gpuColorMs);
        MGE_TracyPlot("host GPU: water ms",     (double)hostT.gpuWaterMs);
        MGE_TracyPlot("host GPU: resolve ms",   (double)hostT.gpuResolveMs);
        g_lastWaitMsStat = r.tRender - r.tWait0;   // → host Stats panel next kickoff
        if (!r.ok) {
            g_mainTexValid = false;
            return r;
        }

        bool copyOk;
        {
            MGE_ZoneScopedN("Forge RT copy");
            copyOk = copyHostRtToDst();
        }
        if (!copyOk) {
            static bool logged = false;
            if (!logged) { LOG::logline("!! [seam] copyHostRtToDst failed"); logged = true; }
            g_mainTexValid = false;
            r.ok = false;
            return r;
        }
        r.tCopy = nowMs();
        g_mainTexValid = true;
        return r;
    }

    // Composite the Forge layer (g_mainTex) OVER MW's frame as a full-screen textured quad
    // with PREMULTIPLIED alpha blend. The Forge RT clears to alpha=0 and geometry writes
    // alpha=1, so alpha is a coverage mask: sky/distant land (already on the backbuffer from
    // renderStage0) show through alpha=0 regions and alpha-test holes. Premultiplied
    // (SRCBLEND=ONE, DESTBLEND=INVSRCALPHA) — NOT SRCALPHA — because MSAA resolve leaves
    // edge pixels premultiplied (rgb already scaled by partial coverage); SRCALPHA would
    // darken edges. State-blocked so nothing leaks into MW's scene 1 (every DistantLand
    // stage does this). -0.5 px offset + POINT filter = the 1:1 texel mapping StretchRect
    // gave (the geometry half-pixel was already corrected host-side). Returns its cost (ms).
    double compositeBlitMainTex(IDirect3DDevice9* device) {
        MGE_ZoneScopedN("Forge composite blit");
        const double t0 = nowMs();
        IDirect3DStateBlock9* sb = nullptr;
        device->CreateStateBlock(D3DSBT_ALL, &sb);

        IDirect3DSurface9* backbuffer = nullptr;
        if (SUCCEEDED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backbuffer)) && backbuffer) {
            device->SetRenderTarget(0, backbuffer);
            backbuffer->Release();
        }

        device->SetPixelShader(nullptr);
        device->SetVertexShader(nullptr);
        device->SetFVF(D3DFVF_XYZRHW | D3DFVF_TEX1);
        device->SetTexture(0, g_mainTex);

        device->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
        device->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_ONE);
        device->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA);
        device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
        device->SetRenderState(D3DRS_ZENABLE, FALSE);
        device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
        device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
        device->SetRenderState(D3DRS_LIGHTING, FALSE);
        device->SetRenderState(D3DRS_FOGENABLE, FALSE);
        device->SetRenderState(D3DRS_STENCILENABLE, FALSE);
        device->SetRenderState(D3DRS_COLORWRITEENABLE, 0x0F);

        device->SetTextureStageState(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1);
        device->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
        device->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1);
        device->SetTextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
        device->SetTextureStageState(1, D3DTSS_COLOROP, D3DTOP_DISABLE);
        device->SetTextureStageState(1, D3DTSS_ALPHAOP, D3DTOP_DISABLE);
        // CRITICAL: disable texture-coordinate transformation. MW leaves a VIEW-DEPENDENT
        // texture matrix active for environment/sphere-map reflections; without this the FF
        // pipeline would transform our blit UVs by that matrix, shearing the composited image
        // as the camera rotates (host output g_mainTex is correct; only the sampled blit skews).
        device->SetTextureStageState(0, D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_DISABLE);
        device->SetTextureStageState(1, D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_DISABLE);
        // CRITICAL: MW leaves environment/sphere-map TEXGEN active (D3DTSS_TEXCOORDINDEX carries
        // a TCI_CAMERASPACE* flag). With texgen the FF pipeline SYNTHESISES texcoords from
        // camera-space position/normal and IGNORES the vertex UVs — disabling the texture matrix
        // alone isn't enough (the generated coords are the zoom/skew gradient we saw). Force the
        // stage to read vertex texcoord set 0 with no texgen.
        device->SetTextureStageState(0, D3DTSS_TEXCOORDINDEX, 0);
        // Belt-and-suspenders: also neutralise the texture matrix itself. Identity == passthrough.
        {
            D3DMATRIX ident = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
            device->SetTransform(D3DTS_TEXTURE0, &ident);
        }
        // Render-scale: the host draws a g_rw x g_rh sub-rect of the g_w x g_h allocation, which we
        // stretch to the g_bbW x g_bbH backbuffer. At scale 1.0 the sub-rect is a 1:1 aligned map so
        // LINEAR == POINT; at >1.0 LINEAR is the supersample downfilter. Sample the sub-rect only.
        const bool superSample = (g_rw != g_bbW) || (g_rh != g_bbH);
        const D3DTEXTUREFILTERTYPE filt = superSample ? D3DTEXF_LINEAR : D3DTEXF_POINT;
        device->SetSamplerState(0, D3DSAMP_MINFILTER, filt);
        device->SetSamplerState(0, D3DSAMP_MAGFILTER, filt);
        device->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
        device->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);

        const float fw = (float)g_bbW, fh = (float)g_bbH;         // destination: the backbuffer
        const float umax = (float)g_rw / (float)g_w;              // source: current render sub-rect
        const float vmax = (float)g_rh / (float)g_h;
        struct CV { float x, y, z, rhw, u, v; };
        const CV quad[4] = {
            { -0.5f,      -0.5f,      0.0f, 1.0f, 0.0f, 0.0f },
            { fw - 0.5f,  -0.5f,      0.0f, 1.0f, umax, 0.0f },
            { -0.5f,      fh - 0.5f,  0.0f, 1.0f, 0.0f, vmax },
            { fw - 0.5f,  fh - 0.5f,  0.0f, 1.0f, umax, vmax },
        };
        device->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, quad, sizeof(CV));

        device->SetTexture(0, nullptr);
        if (sb) { sb->Apply(); sb->Release(); }
        return nowMs() - t0;
    }

    // [hb] heartbeat + spike accounting, shared by the finish (same-frame) and the
    // frame-ahead collect (next frame). On deferred frames tEnd is the collect end
    // (post-copy) and the blit bucket is the PREVIOUS EndScene's deferred blit — carried
    // in via pipeBlitMs (1-frame skew, irrelevant across the 300-frame window) — folded
    // into feed so it stays "the client's own cost" and comparable across all modes.
    void accumFrameStats(const KickState& ks, const FinishResult& fr, double tEnd,
                         bool pipelined, double pipeBlitMs) {
        const double blitMs = pipelined ? pipeBlitMs : (tEnd - fr.tCopy);
        // feed EXCLUDES the overlap window — it's the client's own cost, comparable
        // across fused/async/pipelined: (kickoff prep) + (residual wait + copy + blit).
        const double feed = (ks.tKick - ks.tStart) + (tEnd - fr.tWait0)
                          + (pipelined ? pipeBlitMs : 0.0);
        const double renderBucket = (ks.tKick - ks.tAssign) + (fr.tRender - fr.tWait0);

        // Baseline heartbeat over every composited frame (sees the <kSpikeMs majority).
        // Cut 2B: build first, then geom flush (fold frames capture during build).
        g_hb.feed += feed; g_hb.geom += (ks.tGeomFlush - ks.tBuild);
        g_hb.build += (ks.tBuild - ks.tStart);
        g_hb.render += renderBucket; g_hb.host += fr.hostMs; g_hb.overlap += fr.overlap;
        g_hb.copy += (fr.tCopy - fr.tRender); g_hb.blit += blitMs; g_hb.dt += ks.dtPresent;
        g_hb.mwstart += g_pendingMwStart; g_pendingMwStart = 0.0;   // A0 Cut-4 probe (1-frame skew on deferred frames)
        g_hb.captured += ks.capturedCount;
        // [alpha-dedup] window min/max. A steady count is a real duplicate being suppressed every
        // frame (working as designed); a count that swings frame to frame is the dedup aliasing
        // live particle draws — exactly the smoke/flame flicker signature.
        if (g_capDedupDrops < g_capDedupDropsMin) g_capDedupDropsMin = g_capDedupDrops;
        if (g_capDedupDrops > g_capDedupDropsMax) g_capDedupDropsMax = g_capDedupDrops;
        g_capDedupWindow += g_capDedupDrops;
        g_capDedupDrops = 0;
        g_hb.bEnsure += ks.bEnsure; g_hb.bEmit += ks.bEmit; g_hb.bTail += ks.bTail;   // Phase 0 build-split
        g_hb.bTailEnsure += ks.bTailEnsure; g_hb.bKeys += ks.bKeys;
        g_hb.bTailScan += ks.bTailScan; g_hb.bTailAlpha += ks.bTailAlpha;
        g_hb.bCaptures += ks.bCaptures;   // Stage 0: capture-burst visibility in the window
        const double buildMs = ks.tBuild - ks.tStart;
        if (buildMs > g_hb.maxBuild) g_hb.maxBuild = buildMs;
        if (feed > g_hb.maxFeed) g_hb.maxFeed = feed;
        if (ks.dtPresent > g_hb.maxDt) g_hb.maxDt = ks.dtPresent;
        if (ks.early) ++g_hb.earlyN;
        if (pipelined) ++g_hb.pipeN;
        if (++g_hb.n >= kHeartbeatFrames) {
            // pipe = frames whose finish deferred to the collect; refuse = CUMULATIVE
            // window-guard refusals (must stay 0 — nonzero means an unaudited RPC site
            // fired inside the now frame-long async window).
            LOG::logline(">> [hb] %u frames avg: feed=%.2f geom=%.2f build=%.2f render=%.2f[host=%.2f] "
                         "overlap=%.2f copy=%.2f blit=%.2f dt=%.2f mwstart=%.2f early=%u pipe=%u refuse=%u cap=%.1f | max feed=%.2f dt=%.2f (~%.0f fps)",
                         g_hb.n, g_hb.feed / g_hb.n, g_hb.geom / g_hb.n, g_hb.build / g_hb.n,
                         g_hb.render / g_hb.n, g_hb.host / g_hb.n, g_hb.overlap / g_hb.n,
                         g_hb.copy / g_hb.n, g_hb.blit / g_hb.n, g_hb.dt / g_hb.n,
                         g_hb.mwstart / g_hb.n, g_hb.earlyN,
                         g_hb.pipeN, g_client ? g_client->windowRefusals() : 0u,
                         g_hb.captured / g_hb.n,
                         g_hb.maxFeed, g_hb.maxDt,
                         g_hb.dt > 0.0 ? 1000.0 * g_hb.n / g_hb.dt : 0.0);
            LOG::logline(">> [alpha-dedup] %u frames: drops/frame avg=%.2f min=%u max=%u "
                         "(captured/frame=%.1f) -- fluctuating min!=max => dedup is eating live particle draws",
                         g_hb.n, (double)g_capDedupWindow / g_hb.n,
                         g_capDedupDropsMin == ~0u ? 0u : g_capDedupDropsMin, g_capDedupDropsMax,
                         g_hb.captured / g_hb.n);
            g_capDedupWindow = 0; g_capDedupDropsMin = ~0u; g_capDedupDropsMax = 0;
            // Phase 0 build sub-probe: how much of build= is ensureLive (immovable live read)
            // vs emit (worker-offload candidate) vs tail. Gate: emit ≥ ~2ms ⇒ emit→worker worth it.
            // movable = emit + (tail - tailEnsure); immovable = ensureLive + tailEnsure (live reads).
            LOG::logline(">> [hb] build split: ensureLive=%.2f emit=%.2f tail=%.2f (tailLive=%.2f) ms "
                         "=> movable=%.2f immovable=%.2f (keys=%u) | maxBuild=%.2f captures/f=%.2f",
                         g_hb.bEnsure / g_hb.n, g_hb.bEmit / g_hb.n, g_hb.bTail / g_hb.n,
                         g_hb.bTailEnsure / g_hb.n,
                         (g_hb.bEmit + g_hb.bTail - g_hb.bTailEnsure) / g_hb.n,
                         (g_hb.bEnsure + g_hb.bTailEnsure) / g_hb.n,
                         (unsigned)(g_hb.bKeys / g_hb.n),
                         g_hb.maxBuild, g_hb.bCaptures / g_hb.n);
            // Tail sub-probe: scan = offscreen-caster full-cacheMap loop (@1694, direct-reduce
            // candidate); alpha = merge/sort/pack. tailLive is the scan's live pose-refresh subset.
            LOG::logline(">> [hb] tail split: scan=%.2f alpha=%.2f ms (scan-live=%.2f => scan-scanwork=%.2f)",
                         g_hb.bTailScan / g_hb.n, g_hb.bTailAlpha / g_hb.n,
                         g_hb.bTailEnsure / g_hb.n,
                         (g_hb.bTailScan - g_hb.bTailEnsure) / g_hb.n);
            // Host phase split as the CLIENT received it (Tier 1 wire echo). Cross-check against
            // mgeHost64.log's `host split`/`gpu split`: matching numbers prove the x86/x64
            // HostFrameTimings layout agrees. gpuFrame is whole-command-buffer GPU EXECUTION —
            // compare it to the "Forge Host Frame (inflight)" Tracy box, which is the whole RPC
            // round trip and typically ~2x this. Instantaneous (last frame), not averaged.
            {
                const auto& h = g_lastHostTimings;
                LOG::logline(">> [hb] host recv: gpuFrame=%.2f | cpu setup=%.2f cull=%.2f record=%.2f post=%.2f"
                             " | gpuWait=%.2f total=%.2f ms",
                             h.gpuFrameMs, h.cpuSetupMs, h.cpuCullMs, h.cpuRecordMs, h.cpuPostMs,
                             h.gpuWaitMs, h.totalMs);
            }
            // Part A: per-frame host-geom upload cost, broken down by cause over the same window.
            // parts/frame + KB/frame per category; skin=NPC/creature, morph=pos rewrite (morphing
            // statics + particle regen), uvleak should stay ~0 after the UVController takeover.
            const double invN = 1.0 / g_hb.n;
            auto pf = [&](RenderProcess::UploadCat c) { return g_upHb.parts[c] * invN; };
            auto kf = [&](RenderProcess::UploadCat c) { return g_upHb.bytes[c] * invN / 1024.0; };
            std::uint64_t totB = 0; std::uint32_t totP = 0;
            for (int c = 0; c < RenderProcess::kUpCount; ++c) { totB += g_upHb.bytes[c]; totP += g_upHb.parts[c]; }
            LOG::logline(">> [uploads] %u frames avg/frame: total=%.1f parts/%.1fKB | "
                         "new=%.1f/%.1f skin=%.1f/%.1f morph=%.1f/%.1f vcol=%.1f/%.1f "
                         "topo=%.1f/%.1f uvleak=%.1f/%.1f other=%.1f/%.1f",
                         g_hb.n, totP * invN, totB * invN / 1024.0,
                         pf(RenderProcess::kUpNew),   kf(RenderProcess::kUpNew),
                         pf(RenderProcess::kUpSkin),  kf(RenderProcess::kUpSkin),
                         pf(RenderProcess::kUpMorph), kf(RenderProcess::kUpMorph),
                         pf(RenderProcess::kUpVcol),  kf(RenderProcess::kUpVcol),
                         pf(RenderProcess::kUpTopo),  kf(RenderProcess::kUpTopo),
                         pf(RenderProcess::kUpUvLeak),kf(RenderProcess::kUpUvLeak),
                         pf(RenderProcess::kUpOther), kf(RenderProcess::kUpOther));
            g_upHb = UploadAccum{};
            g_hb = Accum{};
        }

        if (feed >= kSpikeMs) {
            // Overlap diagnosis: early = kickoff fired at BeginScene(0) (vs late EndScene); defer =
            // frame-ahead deferred its finish to the next collect; pipe = this stat came from the
            // deferred collect path. A slow frame with early=0/defer=0 ran FUSED (host fully exposed
            // → the overlap was never armed); early=1/defer=1 but render≈host means the host GPU
            // simply outran the MW overlap window (host-bound — needs host shrink or earlier kick).
            LOG::logline("!! [spike] frame %u feed=%.2fms early=%u defer=%u pipe=%u (geomflush=%.2f build=%.2f texflush=%.2f "
                         "assign=%.2f render=%.2f[host=%.2f] overlap=%.2f copy=%.2f blit=%.2f) "
                         "draws=%u skin=%u mm=%u light=%u alpha=%u cap=%u geom+=%u/%uKB tex+=%u/%uKB dt=%.2fms",
                         ks.frame, feed,
                         (unsigned)ks.early, (unsigned)ks.deferFinish, (unsigned)pipelined,
                         ks.tGeomFlush - ks.tBuild, ks.tBuild - ks.tStart,
                         ks.tTexFlush - ks.tGeomFlush, ks.tAssign - ks.tTexFlush,
                         renderBucket, fr.hostMs, fr.overlap, fr.tCopy - fr.tRender, blitMs,
                         ks.drawCount, ks.skinnedCount, ks.multiMapCount, ks.lightCount,
                         ks.alphaCount, ks.capturedCount,
                         ks.geomParts, (unsigned)(ks.geomBytes >> 10),
                         ks.texCount, (unsigned)(ks.texBytes >> 10),
                         ks.dtPresent);
        }
    }

    // Cell-change shadow eviction (see g_cellEpoch): bump the epoch on any load-door
    // transition BEFORE buildLightList so the cleared identity map hands every new-cell light a
    // fresh id. Signal = the engine's authoritative interior-cell pointer (covers
    // interior<->interior and interior<->exterior) OR a single-frame eye teleport (covers
    // exterior<->exterior load doors / fast travel, which keep the interior pointer null).
    // Runs once per produced frame: from kickoffBody on the serial/inline paths, from
    // fireParked (main, frame start) in produce mode 3 — never both in one frame.
    void checkCellEpochAndPurge(unsigned frame) {
        void* dh = MGE::SceneGraph::getDataHandler();
        const void* interiorCell = dh ? MGE::DataHandlerView::currentInteriorCell(dh) : nullptr;
        const float eye[3] = { DistantLand::eyePos.x, DistantLand::eyePos.y, DistantLand::eyePos.z };
        static const void* s_lastInteriorCell = nullptr;
        static float       s_lastEye[3] = { 1e30f, 1e30f, 1e30f };
        static bool        s_haveEye = false;
        const float ex = eye[0] - s_lastEye[0], ey = eye[1] - s_lastEye[1], ez = eye[2] - s_lastEye[2];
        const bool teleport = s_haveEye && (ex*ex + ey*ey + ez*ez > kCellTeleportDist * kCellTeleportDist);
        if (interiorCell != s_lastInteriorCell || teleport) {
            ++g_cellEpoch;
            g_lightTracks.clear();   // new-cell lights all take fresh ids (no address-reuse inheritance)
            // The scene graph was torn down with the old cell, but the geometry cache keys on
            // shape ADDRESSES and cannot see that — so the old cell's entries survive and get
            // emitted below for a frame (the one-frame flash of the previous cell's objects/NPCs
            // on a transition). Purge them here, BEFORE buildGeometryDrawLists runs.
            //
            // NOT on the very first evaluation: s_lastInteriorCell starts null, so frame 0 always
            // looks like a "transition" — and purging there would throw away the 1524 entries the
            // startup walk just captured, for nothing. There is no old cell to leave yet. (s_haveEye
            // is false exactly once, which is the same condition the teleport test already uses.)
            const bool firstEval = !s_haveEye;
            LOG::logline(">> [cell-purge] epoch=%u frame=%u interiorChanged=%d teleport=%d first=%d cached=%u",
                         g_cellEpoch, frame, (int)(interiorCell != s_lastInteriorCell), (int)teleport,
                         (int)firstEval, (unsigned)MGE::GeometryCache::cache().size());
            if (!firstEval) {
                MGE::GeometryCache::purgeAll();
                // Then resolve those keys to host slots IMMEDIATELY — do not leave them for the
                // deferred drain in flushGeometry(). The purge just released our engine refs, so
                // the old shapes are freed and the allocator will hand the SAME ADDRESSES to the
                // new cell's shapes; buildGeometryDrawLists (below) then re-captures them under
                // the very same keys and assigns them fresh host slots. A drain running after
                // that would look each evicted key up in g_keySlot, find the NEW slot, and
                // release the object we just uploaded — losing every part of the new cell that
                // landed on a recycled address (measured: ~half a cell, every save load).
                // Draining here resolves the keys to the OLD slots, while they still mean what
                // they meant at purge time.
                drainReleasedSlots();
            }
        }
        s_lastInteriorCell = interiorCell;
        s_lastEye[0] = eye[0]; s_lastEye[1] = eye[1]; s_lastEye[2] = eye[2];
        s_haveEye = true;
    }

    void flushAssignAndKick(IDirect3DDevice9* device, unsigned frame,
                            std::uint32_t drawCount, std::uint32_t skinnedCount,
                            std::uint32_t multiMapCount, std::uint32_t lightCount,
                            std::uint32_t skyCount, std::uint32_t alphaCount,
                            IPC::FPFrame& fpFrame, std::uint32_t fpDraws,
                            std::uint32_t fpSkinnedDraws, std::uint32_t fpAlphaDraws, bool fpHave,
                            const float bakeEye[3], bool parkFired,
                            double dtPresent, double tStart, double tBuild);   // fwd (defined below)

    // The produce+RPC-start body (Tier 1a made it D3D9-free). Runs either inline on the MW
    // main thread or, when g_produceOffMain, on the fresh ProduceWorker below. onStage0Composite-
    // Kickoff is the dispatcher that chooses.
    void kickoffBody(IDirect3DDevice9* device) {
        // Phase 0: the previous deferred frame is finished by onStage0CompositeKickoff (the
        // dispatcher, on the MAIN thread) BEFORE this body runs — so g_kick.deferFinish is always
        // cleared on entry. The old in-body backstop (finishAndCopy here) was removed: this body
        // can run on the produce WORKER (OVERLAP mode), and finishAndCopy does a D3D9 RT copy that
        // must never run off the main thread. If a deferred frame ever reached here uncollected it
        // would be a dispatcher bug, not something to paper over with a worker-side D3D9 finish.
        // Whole-frame no-op guarantee: rpcPending stays false on every early-out below, so
        // the paired Finish returns immediately.
        g_park.valid = false;   // mode 3: a serial/inline kickoff supersedes any parked payload
        g_kick = KickState{};
        if (!device || !g_initOk) {
            return;
        }
        // Client-side Forge driving phase, kickoff half: build draw lists → geom flush → tex
        // flush → start the host RenderFrame WITHOUT waiting. The host renders frame N while
        // MW's own frame-N work continues; onStage0CompositeFinish drains the completion and
        // composites. Zoned so the Tracy frame has NO unaccounted gap here.
        MGE_ZoneScopedN("Forge composite kickoff");

        const double tStart = nowMs();
        const double dtPresent = (g_lastPresentMs > 0.0) ? (tStart - g_lastPresentMs) : 0.0;
        g_lastPresentMs = tStart;

        // (Edge-triggered dev-key polls — F11/F12/F9/F8/… — live in pollDevKeys,
        // run once per frame from the frame-ahead collect at BeginScene(0) or from
        // onStage0CompositeFinish, so every per-frame ownership gate, including the
        // frameSetupEarly early-kickoff latch that runs BEFORE this function, sees
        // one consistent value per frame. See the comment there.)
        if (!g_enabled) {
            // Still ship any geometry the cache captured this frame (walk-driven while
            // the composite is off), so the host's mesh store is ready when F11 turns
            // it on.
            MGE_ZoneScopedN("Forge geom flush");
            flushGeometry();
            // AT3 defensive clear: nothing will consume/ship captured alpha while off — drop it so
            // a stale record can't leak in on the next F11-on frame (captures require g_enabled).
            g_capRecs.clear();
            g_capVertScratch.clear();
            g_capIdxScratch.clear();
            g_capTexMemo.clear();
            return;
        }

        const unsigned frame = g_frame++;

        checkCellEpochAndPurge(frame);

        // Re-walk the geometry cache to build the host's draw lists (the MGE→Forge feeding cost —
        // Phase 2 makes this GPU-resident so it goes to 0).
        std::uint32_t drawCount, skinnedCount, multiMapCount, lightCount, skyCount, alphaCount;
        IPC::FPFrame fpFrame;
        std::uint32_t fpDraws = 0, fpSkinnedDraws = 0, fpAlphaDraws = 0;
        bool fpHave = false;
        {
            MGE_ZoneScopedN("Forge build draw lists");
            markWorkerPhase(WK_BUILD_DRAW);   // ensureLive reads g_cache + LIVE scene graph
            buildGeometryDrawLists(drawCount, skinnedCount, multiMapCount, alphaCount);
            markWorkerPhase(WK_BUILD_LIGHT);
            lightCount = buildLightList();
            markWorkerPhase(WK_BUILD_SKY);
            skyCount = buildSkyDrawList();
            // FP1a: before the geom flush below so first-sight arm meshes (captured by
            // this frame's FP walk) ship in the same flush the pass draws from.
            markWorkerPhase(WK_BUILD_FP);
            fpHave = buildFPFrame(fpFrame, fpDraws, fpSkinnedDraws, fpAlphaDraws);
        }
        const double tBuild = nowMs();

        // Everything above wrote only client-private scratch; the shared-memory half
        // (gate → flushes → assigns → constants → renderSceneKickoff) is the extracted
        // flushAssignAndKick, shared verbatim with the mode-3 park fire. bakeEye == eyePos
        // here ⇒ the camera restamp inside is the exact identity (byte-identical constants).
        const float bakeEye[3] = { DistantLand::eyePos.x, DistantLand::eyePos.y, DistantLand::eyePos.z };
        flushAssignAndKick(device, frame, drawCount, skinnedCount, multiMapCount, lightCount,
                           skyCount, alphaCount, fpFrame, fpDraws, fpSkinnedDraws, fpAlphaDraws,
                           fpHave, bakeEye, /*parkFired=*/false, dtPresent, tStart, tBuild);
    }

    // The shared-memory half of the produce: gate → flushGeometry → flushTextures → vec
    // assigns (incl. captured-alpha) → frame constants → renderSceneKickoff → populate
    // g_kick. Extracted from kickoffBody so the mode-3 park fire (fireParked, main thread,
    // frame start) can ship a payload the worker built LAST frame. `bakeEye` = the eye the
    // payload was emitted relative to (build-time DistantLand::eyePos); the view restamp
    // below re-aims it with the CURRENT camera. On the serial paths bakeEye == eyePos and
    // every byte matches the pre-extraction code. `parkFired` marks g_kick so the collect
    // keys the finish on state, not the current produce mode.
    void flushAssignAndKick(IDirect3DDevice9* device, unsigned frame,
                            std::uint32_t drawCount, std::uint32_t skinnedCount,
                            std::uint32_t multiMapCount, std::uint32_t lightCount,
                            std::uint32_t skyCount, std::uint32_t alphaCount,
                            IPC::FPFrame& fpFrame, std::uint32_t fpDraws,
                            std::uint32_t fpSkinnedDraws, std::uint32_t fpAlphaDraws, bool fpHave,
                            const float bakeEye[3], bool parkFired,
                            double dtPresent, double tStart, double tBuild) {
        // Drive the host renderer into the shared RT. Async — the host fence-waits before
        // signalling completion, so at renderSceneFinish the draw is GPU-complete and the
        // resource quiescent before our copy.
        bool ok = false;

        // Ship this frame's captured geometry AFTER the build (Cut 2B): on fold frames
        // first-sight parts are captured DURING buildGeometryDrawLists (ensureLive lazy
        // capture), and the host must have the mesh bytes before the RenderFrame RPC
        // draws them — same reason textures flush after the build. Cheap on steady
        // frames (geom+=0KB), spikes on cell loads; goes to ~0 with Phase 2 (resident
        // GPU geometry — nothing to ship).
        const std::uint32_t geomParts = g_pendingParts;            // snapshot (flush clears it)
        const std::size_t   geomBytes = g_pendingBlob.size();

        // ---- END OF THE CLIENT-PRIVATE HALF. Everything above wrote only *Scratch/g_pendingBlob
        // (client memory); everything below writes SHARED memory the host is still reading for
        // frame N-1 — the geometry blob, then the draw/skinned/multiMap/light/sky/alpha vectors,
        // then the RenderFrame params, and finally the one imported host RT.
        //
        // Being a frame ahead means two frames are alive at once, and every resource they share is
        // single-buffered. The kick now runs AHEAD of the finish (that is what recovered the host
        // overlap), so without this gate the worker overwrites frame N's geometry while the host is
        // mid-render on N-1 — visible as alternating good/bad frames, worst on geometry that is
        // re-uploaded every frame (particles: smoke/flames flicker).
        //
        // So the gate sits HERE, not at renderSceneKickoff: the whole ~3-4ms build overlaps the
        // finish (the win), and not one shared byte is touched until main's renderSceneFinish +
        // RT copy have released frame N-1. Expected wait ≈ 0 — the finish is ~1.7ms under a ~3.5ms
        // build. If "Forge kickoff gate ms" starts showing real time, the build got cheaper than
        // the finish and double-buffering the shared vectors is the next move.
        markWorkerPhase(WK_GATE);
        gateOnPrevFinish("kickoff");

        {
            markWorkerPhase(WK_GEOM);
            MGE_ZoneScopedN("Forge geom flush");
            flushGeometry();
        }
        const double tGeomFlush = nowMs();

        // Ship any textures newly referenced this frame BEFORE the scene draw that uses them
        // (buildDrawList queued their DDS via resolveTextureSlot). Lazy: only first-seen textures.
        const std::uint32_t texCount = g_texPendingCount;          // snapshot (flush clears it)
        const std::size_t   texBytes = g_texPendingBlob.size();
        {
            markWorkerPhase(WK_TEX);
            MGE_ZoneScopedN("Forge tex flush");
            flushTextures();
        }
        const double tTexFlush = nowMs();
        markWorkerPhase(WK_ASSIGN);

        const bool haveDraw = g_drawVec && drawCount > 0
            && g_drawVec->assign_bytes(g_drawScratch.data(), (std::uint32_t)g_drawScratch.size());

        IPC::VecId   skinnedId = IPC::InvalidVector;
        std::uint32_t skinnedBytes = 0;
        if (g_skinnedVec && skinnedCount > 0
            && g_skinnedVec->assign_bytes(g_skinnedScratch.data(), (std::uint32_t)g_skinnedScratch.size())) {
            skinnedId    = g_skinnedVec->id();
            skinnedBytes = (std::uint32_t)g_skinnedScratch.size();
        }

        IPC::VecId   multiMapId = IPC::InvalidVector;
        std::uint32_t multiMapBytes = 0;
        if (g_multiMapVec && multiMapCount > 0
            && g_multiMapVec->assign_bytes(g_multiMapScratch.data(), (std::uint32_t)g_multiMapScratch.size())) {
            multiMapId    = g_multiMapVec->id();
            multiMapBytes = (std::uint32_t)g_multiMapScratch.size();
        }

        IPC::VecId   lightId = IPC::InvalidVector;
        std::uint32_t lightBytes = 0;
        if (g_lightVec && lightCount > 0
            && g_lightVec->assign_bytes(g_lightScratch.data(), (std::uint32_t)g_lightScratch.size())) {
            lightId    = g_lightVec->id();
            lightBytes = (std::uint32_t)g_lightScratch.size();
        }

        IPC::VecId   skyId = IPC::InvalidVector;
        std::uint32_t skyBytes = 0;
        if (g_skyVec && skyCount > 0
            && g_skyVec->assign_bytes(g_skyScratch.data(), (std::uint32_t)g_skyScratch.size())) {
            skyId    = g_skyVec->id();
            skyBytes = (std::uint32_t)g_skyScratch.size();
        }

        IPC::VecId   alphaId = IPC::InvalidVector;
        std::uint32_t alphaBytes = 0;
        if (g_alphaVec && alphaCount > 0
            && g_alphaVec->assign_bytes(g_alphaScratch.data(), (std::uint32_t)g_alphaScratch.size())) {
            alphaId    = g_alphaVec->id();
            alphaBytes = (std::uint32_t)g_alphaScratch.size();
        }

        // FP1a: assign the FP scratches into their vecs; a failed assign just drops that
        // list this frame (fpEnabled derives from the counts inside renderSceneKickoff).
        if (fpHave) {
            if (fpDraws > 0
                && g_fpDrawVec->assign_bytes(g_fpDrawScratch.data(), (std::uint32_t)g_fpDrawScratch.size())) {
                fpFrame.drawList  = g_fpDrawVec->id();
                fpFrame.drawCount = fpDraws;
                fpFrame.drawBytes = (std::uint32_t)g_fpDrawScratch.size();
            }
            if (fpSkinnedDraws > 0
                && g_fpSkinnedVec->assign_bytes(g_fpSkinnedScratch.data(), (std::uint32_t)g_fpSkinnedScratch.size())) {
                fpFrame.skinnedList  = g_fpSkinnedVec->id();
                fpFrame.skinnedCount = fpSkinnedDraws;
                fpFrame.skinnedBytes = (std::uint32_t)g_fpSkinnedScratch.size();
            }
            // FP1c: the blended FP parts (torch flame, enchant glow), client-sorted
            // back-to-front vs the ARM camera in buildFPFrame.
            if (fpAlphaDraws > 0 && g_fpAlphaVec
                && g_fpAlphaVec->assign_bytes(g_fpAlphaScratch.data(), (std::uint32_t)g_fpAlphaScratch.size())) {
                fpFrame.alphaList  = g_fpAlphaVec->id();
                fpFrame.alphaCount = fpAlphaDraws;
                fpFrame.alphaBytes = (std::uint32_t)g_fpAlphaScratch.size();
            }
        }

        // AT3 captured-alpha: ship the geometry buildGeometryDrawLists left in the scratch
        // (referenced by the sentinel-slot AlphaDrawWire items just packed). One 1-chunk vec
        // holds [verts (GeomVertexWire)][indices (uint16)] contiguously — indices begin at
        // capturedVertBytes. Then clear the scratch so this frame's fresh captures start at base 0
        // (the wire bases match the shipped layout because we ship the whole scratch verbatim).
        IPC::VecId   capturedId = IPC::InvalidVector;
        std::uint32_t capVertBytes = 0, capIdxBytes = 0;
        if (g_capturedVec && !g_capVertScratch.empty()) {
            capVertBytes = (std::uint32_t)(g_capVertScratch.size() * sizeof(IPC::GeomVertexWire));
            capIdxBytes  = (std::uint32_t)(g_capIdxScratch.size() * sizeof(std::uint16_t));
            static std::vector<std::uint8_t> capBlob;   // reused staging blob (contiguous verts+indices)
            capBlob.resize((std::size_t)capVertBytes + capIdxBytes);
            memcpy(capBlob.data(), g_capVertScratch.data(), capVertBytes);
            if (capIdxBytes) { memcpy(capBlob.data() + capVertBytes, g_capIdxScratch.data(), capIdxBytes); }
            if (g_capturedVec->assign_bytes(capBlob.data(), (std::uint32_t)capBlob.size())) {
                capturedId = g_capturedVec->id();
            } else {
                capVertBytes = capIdxBytes = 0;   // over one window (shouldn't happen at these caps)
            }
        }
        g_capVertScratch.clear();
        g_capIdxScratch.clear();
        const double tAssign = nowMs();

        // The MGE→Forge FEEDING cost (the bulk of the "unaccounted" client gap): every frame MGE
        // walks its scene-graph cache to rebuild the host's draw lists + uploads geom/tex over IPC.
        // This is exactly the dependence to drive toward 0 — a resident flat GPU scene (Phase 2) lets
        // the host keep geometry across frames and the GPU cull build draw lists, removing both.
        // Cut 2B bucket order: build runs FIRST now (geom flush moved after it), and on
        // fold frames the build bucket includes the ensureLive cost that used to bill
        // to buildFrustumVisibleSet (cross-run comparison caveat).
        MGE_TracyPlot("Forge prep: buildLists ms", tBuild - tStart);
        MGE_TracyPlot("Forge prep: geomFlush ms", tGeomFlush - tBuild);
        MGE_TracyPlot("Forge prep: texFlush ms", tTexFlush - tGeomFlush);
        MGE_TracyPlot("Forge prep: assign ms", tAssign - tTexFlush);

        // Only drive + composite the host when there's actual scene data this frame. With no
        // draw list (loading doors, menus, empty cells) we must NOT fall back to the bring-up
        // triangle and composite it — that flashes the debug triangle over MW's loading/menu
        // frame. Skip the seam entirely and let MW present its own (fixed-function) frame.
        if (!haveDraw && skinnedId == IPC::InvalidVector && multiMapId == IPC::InvalidVector
            && skyId == IPC::InvalidVector && alphaId == IPC::InvalidVector) {
            return;
        }

        // CAMERA-RELATIVE viewProj: the payload's world translations are pre-shifted by -bakeEye
        // (the eye at BUILD time), so the view translation row must be (bakeEye - eyeNow)·R_now:
        // v_view = (v_rel + bakeEye - eyeNow)·R_now. On the serial paths bakeEye == eyePos, the
        // delta is 0 and this reduces to the original "zero the translation row" (eyePos ==
        // inverse(mwView)·origin, hence mwView's translation row == -eye·R — the cancellation is
        // exact). Mode-3 park fires re-aim last frame's payload with THIS frame's rotation +
        // position — one adjusted matrix covers the whole payload, which is emitted relative to
        // bakeEye throughout (statics/multimap/alpha/lights/captures). Keeps the vertex pipeline
        // near the origin (no float32 large-world stretching) in every mode.
        D3DXMATRIX viewRel = DistantLand::mwView;
        {
            const float dx = bakeEye[0] - DistantLand::eyePos.x;
            const float dy = bakeEye[1] - DistantLand::eyePos.y;
            const float dz = bakeEye[2] - DistantLand::eyePos.z;
            viewRel._41 = dx * viewRel._11 + dy * viewRel._21 + dz * viewRel._31;
            viewRel._42 = dx * viewRel._12 + dy * viewRel._22 + dz * viewRel._32;
            viewRel._43 = dx * viewRel._13 + dy * viewRel._23 + dz * viewRel._33;
        }

        // UNIFIED FAR PROJECTION (Phase 1a/1b): push the far plane out to the distant-land draw
        // distance so near opaque + host-owned DL share ONE reverse-Z depth mapping (statics get
        // occluded behind distant terrain in the shared depth). Near geometry is unaffected — only
        // the far plane moves; reverse-Z keeps near precision. mwProj is the NEAR projection; recover
        // its near plane (zn = -_43/_33 for a standard D3D perspective) and re-edit only z.
        //
        // EXTERIORS ONLY — the far extension exists solely for host-owned DL, which is itself
        // gated on isExterior (lighting[27]). In ANY interior (weather or not: showcase mod
        // interiors fly a full sky) MW hardware-clips at its own far plane (~view distance):
        // giant meshes spanning the cell get most of their triangles clipped for free, and
        // everything past the fog wall is simply absent. Extending the far plane there made the
        // host rasterize + shade ALL of it — huge fill cost, and the beyond-fog geometry shades
        // to solid fogColNear (black in dark interiors), walling the view where MW shows
        // nothing. Keeping mwProj untouched clips exactly where MW clips; interior sky geometry
        // is walked from MW's own skyRoot, which MW renders inside mwProj by construction.
        const bool isExterior = MWBridge::get()->IsExterior();
        D3DXMATRIX farProj = DistantLand::mwProj;
        if (isExterior) {
            const float zn = (farProj._33 != 0.0f) ? (-farProj._43 / farProj._33) : 4.0f;
            DistantLand::editProjectionZ(&farProj, zn, Configuration.DL.DrawDist * DistantLand::kCellSize);
        }
        D3DXMATRIX viewProj;
        D3DXMatrixMultiply(&viewProj, &viewRel, &farProj);

        // Tier 1 lighting (6 × float4): MW sun/ambient/fog for this frame, uploaded into the
        // host gFrameData after viewProj. sunVec is the world-space sun TRAVEL direction (the
        // shader does dot(N, -sunDir)); fogParams = (fogNearStart, fogNearEnd); dist for fog is
        // |worldPos - eyePos| in-shader.
        //
        // Sun/ambient use the CANONICAL MGE PPL/DL formula (distantland.cpp:780-792, :1141):
        //   sun     = lightSunMult * sunCol
        //   ambient = lightAmbMult * (sunAmb + ambCol)   // ambCol alone == globalAmbient, often
        //                                                 // ~0 in exteriors; the ambient fill is
        //                                                 // mostly the sun light's sunAmb term.
        RGBVECTOR sunColEff = DistantLand::lightSunMult * DistantLand::sunCol;
        RGBVECTOR ambColEff = DistantLand::lightAmbMult * (DistantLand::sunAmb + DistantLand::ambCol);
        D3DXVECTOR4 sunVecEff = DistantLand::sunVec;
        // INTERIORS: don't trust the proxy-captured sun/ambient — the capture only refreshes
        // when MW re-programs light state (SetLight(6) / D3DRS_AMBIENT), which happens on
        // light-set churn, not per frame. Exteriors churn constantly (the sun MOVES, geometry
        // streams), but a static interior under full Forge ownership can go untouched for
        // MINUTES: on first load the crossing kept the previous cell's sun/ambient until
        // movement or a weapon raise dirtied the light state ("stuck ambient/sun"). Read the
        // authoritative live values instead: the scene-graph sunlight (TES3DataHandler
        // sgSunlight — the NiDirectionalLight MW programs light 6 FROM; the weather
        // controller drives it in weather-flying interiors too) and, in weatherless
        // interiors, the cell record's ambient (what MW feeds D3DRS_AMBIENT; the sun light's
        // own ambient rides along live and is ~0 there — same split OpenMW uses).
        // EXTERIORS TOO (2026-07-14): the capture starves in exteriors as well, whenever MW draws no
        // lit geometry. Start the game high in the sky, outside MW's own draw range, and it programs
        // no light state at all — so sun/ambient stay FROZEN at whatever was last captured, then snap
        // the moment you drop back into range and MW re-sends light 6 ("doesn't update in the sky,
        // updates instantly when I move close"). The premise above — that exteriors churn constantly —
        // holds only while MW is actually drawing something. sgSunlight is what MW programs light 6
        // FROM and the weather controller animates it every frame regardless of what's on screen, so
        // read it live in BOTH cases. Only the STALE sunVec/sunCol/sunAmb are replaced; the exterior
        // ambient composition (sun ambient + the D3DRS_AMBIENT global) is preserved as-is.
        {
            float sdir[3], sdif[3], samb[3], sdim = 1.0f;
            if (MWBridge::get()->getSceneSunlight(sdir, sdif, samb, &sdim)) {
                D3DXVECTOR3 sd(sdir[0], sdir[1], sdir[2]);
                D3DXVec3Normalize(&sd, &sd);
                sunVecEff = D3DXVECTOR4(sd.x, sd.y, sd.z, 1.0f);
                const RGBVECTOR liveSun(sdif[0] * sdim, sdif[1] * sdim, sdif[2] * sdim);
                const RGBVECTOR liveSunAmb(samb[0] * sdim, samb[1] * sdim, samb[2] * sdim);
                sunColEff = DistantLand::lightSunMult * liveSun;
                // sgSunlight.ambient ALREADY carries the cell ambient in interiors (in-game
                // verified: it equals the cell record's ambientColor byte-for-byte), so there it IS
                // the whole ambient term and adding the cell record on top would double it. Exteriors
                // keep their existing composition (sun ambient + the D3DRS_AMBIENT global ambCol) —
                // same formula as the captured path, just with a live sun ambient.
                ambColEff = isExterior ? (DistantLand::lightAmbMult * (liveSunAmb + DistantLand::ambCol))
                                       : (DistantLand::lightAmbMult * liveSunAmb);
                // Periodic live-vs-captured compare (mapping oracle): after movement/weapon
                // churn refreshes the captures, fresh captured ambCol tells whether interior
                // D3DRS_AMBIENT really is ~0 (assumed above). cellAmb logged for reference.
                static std::uint32_t s_lightLogEpoch = ~0u;
                static unsigned s_lightLogN = 0;
                const bool epochEdge = (g_cellEpoch != s_lightLogEpoch);
                if (epochEdge || (s_lightLogN++ % 900 == 0)) {
                    s_lightLogEpoch = g_cellEpoch;
                    const BYTE* ca = MWBridge::get()->CellHasWeather() ? nullptr : MWBridge::get()->getInteriorAmb();
                    LOG::logline(">> [light] %s live: sun=(%.3f %.3f %.3f) sunAmb=(%.3f %.3f %.3f) cellAmb=(%.3f %.3f %.3f) dir=(%.2f %.2f %.2f) dim=%.2f",
                                 isExterior ? "exterior" : "interior",
                                 liveSun.r, liveSun.g, liveSun.b, liveSunAmb.r, liveSunAmb.g, liveSunAmb.b,
                                 ca ? ca[0] / 255.0f : -1.0f, ca ? ca[1] / 255.0f : -1.0f, ca ? ca[2] / 255.0f : -1.0f,
                                 sd.x, sd.y, sd.z, sdim);
                    LOG::logline(">> [light] captured (stale if MW drew nothing lit): sun=(%.3f %.3f %.3f) sunAmb=(%.3f %.3f %.3f) ambCol=(%.3f %.3f %.3f) dir=(%.2f %.2f %.2f)",
                                 DistantLand::sunCol.r, DistantLand::sunCol.g, DistantLand::sunCol.b,
                                 DistantLand::sunAmb.r, DistantLand::sunAmb.g, DistantLand::sunAmb.b,
                                 DistantLand::ambCol.r, DistantLand::ambCol.g, DistantLand::ambCol.b,
                                 DistantLand::sunVec.x, DistantLand::sunVec.y, DistantLand::sunVec.z);
                }
            }
        }
        // Host-computed sky dome (C2): the current interpolated ZENITH sky colour MW already blended
        // this frame from the Weather_*_Sky_*_Color ini keys (getCurrentWeatherSkyCol; MW does the
        // time-of-day blend internally). The host colours the dome with a vertical gradient
        // fogColNear(horizon) -> skyZenith(zenith), so we ship ONLY this colour — the dome geometry
        // itself no longer re-uploads (see scenegraph_geometry_cache SK1). Null-guarded to black.
        // Guard on CellHasWeather like DistantLand::update (the weather-struct pointer is only valid
        // then); fall back to the horizon (nearFogCol) so a no-weather cell yields a flat dome. The
        // dome only draws in weather exteriors anyway, so skyZenith is otherwise unconsumed.
        MWBridge* mwb = MWBridge::get();
        const RGBVECTOR* skyColPtr = mwb->CellHasWeather() ? mwb->getCurrentWeatherSkyCol() : nullptr;
        const float skyZenithR = skyColPtr ? skyColPtr->r : DistantLand::nearFogCol.r;
        const float skyZenithG = skyColPtr ? skyColPtr->g : DistantLand::nearFogCol.g;
        const float skyZenithB = skyColPtr ? skyColPtr->b : DistantLand::nearFogCol.b;
        // Wind magnitude (lighting[18]) — drives the host's flame-flicker rate: a windy exterior makes
        // torch/candle shadows dance harder. MW's wind vector is very noisy, so smooth it the same way
        // DistantLand::update does (EWMA f=0.02) and ship the magnitude only (the flicker is isotropic).
        // Exterior + weather-cell gated CLIENT-side (IsExterior is authoritative here), so an interior
        // always ships 0 and the host needs no exterior gate of its own.
        static float s_smoothWind[2] = {};
        float windMag = 0.0f;
        if (isExterior && mwb->CellHasWeather() && !mwb->IsMenu()) {
            const float* wind = mwb->GetWindVector();
            s_smoothWind[0] += 0.02f * (wind[0] - s_smoothWind[0]);
            s_smoothWind[1] += 0.02f * (wind[1] - s_smoothWind[1]);
            windMag = std::sqrt(s_smoothWind[0] * s_smoothWind[0] + s_smoothWind[1] * s_smoothWind[1]);
        }
        const float lighting[36] = {
            sunVecEff.x,               sunVecEff.y,               sunVecEff.z,               0.0f,
            sunColEff.r,               sunColEff.g,               sunColEff.b,               0.0f,
            ambColEff.r,               ambColEff.g,               ambColEff.b,               0.0f,
            DistantLand::nearFogCol.r, DistantLand::nearFogCol.g, DistantLand::nearFogCol.b, 0.0f,
            // [18] = smoothed wind magnitude (0 in interiors); [19] = cell epoch. Both land in
            // fogParams.zw, which no shader reads — the host consumes them CPU-side (wind → flicker
            // rate; epoch change → evict all shadow slots + caster records).
            DistantLand::fogNearStart, DistantLand::fogNearEnd,   windMag,                   float(g_cellEpoch),
            // CAMERA-RELATIVE: WorldPos reaches the shader already relative to the eye, so the
            // eyePos used for the per-vertex fog distance |worldPos - eyePos| is the origin (0).
            0.0f,                      0.0f,                      0.0f,                      0.0f,
            // Phase 1a/1b: the absolute camera eye the payload is RELATIVE to (host shifts
            // resident DL by -this eye — it must match the payload's relative space, so on a
            // mode-3 park fire this is the BUILD-time eye, not the current one; serial paths
            // pass bakeEye == eyePos) + isExterior gate (1 = feed host-owned DL).
            bakeEye[0],                bakeEye[1],                bakeEye[2],                isExterior ? 1.0f : 0.0f,
            // C2 skyZenith (float4 28..31): zenith sky colour for the host dome gradient. Host reads
            // it into FrameData.skyZenith; only sky.frag (dome branch) consumes it.
            skyZenithR,                skyZenithG,                skyZenithB,                0.0f,
            // [32] MW SIMULATION time (seconds this session, frozen in menus) — drives the UV scroll
            // of UV-animated distant statics (ghostfence) in statics.vert. This is the SAME clock MGE's
            // DX9 path feeds its `time` uniform (distantland.cpp:987), so the Forge and MGE fences
            // scroll in step under an F11 A/B. NOT a host wall clock: that keeps running in menus and,
            // being steady_clock-since-BOOT, quantizes to 0.02-0.13s steps once cast to float32 (the
            // bug that made the host's water normals judder — see forgerender.cpp:9150).
            // [33]/[34] Part A upload cost: THIS frame's total host-geom reship KB + part count,
            // surfaced on the host perf panel. Filled from g_upFrame just below (the array is const,
            // so a non-const alias writes the two slots after the aggregate is computed).
            mwb->simulationTime(),     0.0f,                      0.0f,                      0.0f,
        };

        // Part A: aggregate this frame's per-category upload cost, publish the total to the host
        // (lighting[33]=KB, [34]=parts), fold the window sum for the [uploads] heartbeat, then zero
        // the frame accumulator for the next walk.
        {
            float* lightingW = const_cast<float*>(lighting);
            std::uint64_t frameBytes = 0; std::uint32_t frameParts = 0;
            for (int c = 0; c < RenderProcess::kUpCount; ++c) {
                frameBytes += g_upFrame.bytes[c]; frameParts += g_upFrame.parts[c];
                g_upHb.bytes[c] += g_upFrame.bytes[c]; g_upHb.parts[c] += g_upFrame.parts[c];
            }
            lightingW[33] = float(frameBytes) * (1.0f / 1024.0f);   // KB this frame
            lightingW[34] = float(frameParts);
            g_upFrame = UploadAccum{};
        }

        // Dev overlay input (Stage 2): poll the mouse in MW client-space pixels (1:1 with the host
        // render target) + L/R/M buttons, and forward with the F9 visibility flag. Only meaningful
        // when the panel is up; otherwise uiVisible=0 leaves the host overlay hidden/inert.
        IPC::DevInput devInput;
        devInput.uiVisible = g_devUiVisible ? 1u : 0u;
        // F8 one-shot host compute-shader hot-reload: the edge poll happens at composite
        // finish (see there); consume the latched request into THIS frame's DevInput.
        if (g_reloadShadersPending) {
            devInput.reloadShaders = 1u;
            g_reloadShadersPending = false;
        }
        if (g_distLightsTogglePending) {
            devInput.distLightsToggle = 1u;
            g_distLightsTogglePending = false;
        }
        // Frame-ahead observability → host Stats panel: last frame's collect wait and
        // mwstart (1-frame skew, panel only) + this frame's dt and the live toggle.
        devInput.frameAhead     = g_frameAheadLive ? 1u : 0u;
        devInput.clientWaitMs   = (float)g_lastWaitMsStat;
        devInput.clientDtMs     = (float)dtPresent;
        devInput.clientMwStartMs = (float)g_lastMwStartStat;
        if (g_devUiVisible) {
            if (!g_devHwnd) {
                D3DDEVICE_CREATION_PARAMETERS cp = {};
                if (SUCCEEDED(device->GetCreationParameters(&cp))) {
                    g_devHwnd = cp.hFocusWindow;
                }
            }
            POINT p;
            if (GetCursorPos(&p) && g_devHwnd && ScreenToClient(g_devHwnd, &p)) {
                devInput.x = p.x;
                devInput.y = p.y;
            }
            if (GetAsyncKeyState(VK_LBUTTON) & 0x8000) devInput.buttons |= 0x1u;
            if (GetAsyncKeyState(VK_RBUTTON) & 0x8000) devInput.buttons |= 0x2u;
            if (GetAsyncKeyState(VK_MBUTTON) & 0x8000) devInput.buttons |= 0x4u;
        }

        // WT1 Forge water: per-frame surface params (no geometry — the host generates the
        // geo-clipmap mesh). waterOn gates the host water pass. depthBaseColor mirrors XE Mod
        // Water.fx:24 using available DistantLand colours as the skyCol/fogColFar proxies; windFactor
        // is a calm constant (windVec isn't exposed to MGE — tune later). camFwd = mwView's 3rd column
        // (world-space view forward) for the slant→perpendicular shoreline depth correction.
        float waterParams[12] = {};
        // CellHasWater: waterless interiors (most of them) must not draw the host water —
        // MGE's own water path always gated on this and the Forge crossing lost it, so the
        // host drew the geo-clipmap at a stale WaterLevel() in dry cells. Per-frame cell
        // state gates only THIS wire flag (host skips water + its reflection pass);
        // forgeOwnsFrame() stays the mode gate (forgeOwnsDepth / fold / WT3 suppression
        // must not flip per cell). Exteriors always have water (CellHasWater true there).
        const std::uint32_t waterOn =
            (forgeOwnsFrame() && MWBridge::get()->CellHasWater()) ? 1u : 0u;
        if (waterOn) {
            MWBridge* mw = MWBridge::get();
            const float sunlightFactor = 1.0f - (1.0f - DistantLand::sunVis) * (1.0f - DistantLand::sunVis);
            const RGBVECTOR sunAdj = sunlightFactor * DistantLand::sunCol;
            const RGBVECTOR skyC = DistantLand::horizonCol;     // skyCol proxy
            const RGBVECTOR fogF = DistantLand::nearFogCol;     // fogColFar proxy
            waterParams[0]  = mw->WaterLevel();
            waterParams[1]  = 0.013f;                            // windFactor (calm; tune later)
            waterParams[2]  = 24.0f;                             // shoreDepthBias (XE Mod Water.fx:27)
            waterParams[3]  = sunAdj.r * 0.03f + (2.0f * skyC.r + fogF.r) * 0.075f;
            waterParams[4]  = sunAdj.g * 0.04f + (2.0f * skyC.g + fogF.g) * 0.080f;
            waterParams[5]  = sunAdj.b * 0.05f + (2.0f * skyC.b + fogF.b) * 0.085f;
            waterParams[6]  = DistantLand::nearViewRange;
            waterParams[7]  = mw->IsUnderwater(DistantLand::eyePos.z) ? 1.0f : 0.0f;
            waterParams[8]  = DistantLand::mwView._13;
            waterParams[9]  = DistantLand::mwView._23;
            waterParams[10] = DistantLand::mwView._33;
            waterParams[11] = 0.0f;
        }

        // Async kickoff: copy the frame params into shared memory and start the host, then
        // RETURN — the host renders while MW's frame-N work continues. All the pointer args
        // (viewProj/lighting/devInput/waterParams) are memcpy'd into the IPC block before
        // renderSceneKickoff returns, so these stack locals can die here.
        {
            markWorkerPhase(WK_KICKOFF);
            MGE_ZoneScopedN("Forge renderSceneKickoff");
            ok = g_client->renderSceneKickoff(frame, (const float*)&viewProj, lighting,
                     haveDraw ? g_drawVec->id() : IPC::InvalidVector,
                     haveDraw ? drawCount : 0,
                     haveDraw ? (std::uint32_t)g_drawScratch.size() : 0,
                     skinnedId, skinnedCount, skinnedBytes,
                     multiMapId, (multiMapId != IPC::InvalidVector) ? multiMapCount : 0, multiMapBytes,
                     lightId, (lightId != IPC::InvalidVector) ? lightCount : 0, lightBytes,
                     skyId, (skyId != IPC::InvalidVector) ? skyCount : 0, skyBytes,
                     alphaId, (alphaId != IPC::InvalidVector) ? alphaCount : 0, alphaBytes,
                     capturedId, capVertBytes, capIdxBytes,
                     (std::uint32_t)g_debugMode, &devInput, waterParams, waterOn,
                     fpHave ? &fpFrame : nullptr);
        }
        if (!ok) {
            return;     // rpcPending stays false → Finish no-ops
        }

        // Hand everything the finish half needs across the overlap window.
        // Host-inflight lane start (paired 0.0 in finishAndCopy): from here until the
        // collect/finish drains the completion, the host owns the frame.
        MGE_TracyPlot("Forge host inflight", 1.0);
        // Open the host-frame fiber zone (box on the "Forge Host Frame (inflight)" lane) — spans until the
        // finish drains the completion. Begins here on whichever thread issued the kickoff (the
        // produce worker in OVERLAP/FENCED mode, else the main thread).
        MGE_TracyHostFrameBegin(g_hostZoneCtx);
        g_hostZoneOpen = true;
        g_kick.rpcPending    = true;
        g_kick.parkFired     = parkFired;   // mode 3: collect closes this window keyed on state
        g_kick.early         = DistantLand::earlyForgeKickoff || parkFired;  // BeginScene(0)/frame-start site vs late (EndScene)
        // Frame-ahead pipelining: defer the finish to the NEXT frame's BeginScene(0)
        // collect — but ONLY on early-kickoff frames. The earlyForgeKickoff latch is
        // exactly the "whole MW frame is IPC-free on both channels" predicate (statics
        // cull gated off, grass hoisted, reflection/shadow RPCs suppressed under
        // forgeOwnsFrame), so widening the window from scene 0 to the whole frame is
        // safe precisely there. Late frames (F11-off / seam down) keep the same-frame finish.
        // Since S2 retired the warm-up latch there is no longer a guaranteed serial frame ahead
        // of the first deferred blit, so g_mainTex may still be unprimed then — onFrameAheadBlit
        // self-gates on g_mainTexValid, costing at most one unblitted frame at startup.
        g_kick.deferFinish   = (g_frameAheadLive && DistantLand::earlyForgeKickoff) || parkFired;
        g_kick.frame         = frame;
        g_kick.drawCount     = drawCount;
        g_kick.skinnedCount  = skinnedCount;
        g_kick.multiMapCount = multiMapCount;
        g_kick.lightCount    = lightCount;
        g_kick.skyCount      = skyCount;
        g_kick.alphaCount    = alphaCount;
        g_kick.capturedCount = g_capturedEmitted;
        g_kick.geomParts     = geomParts;
        g_kick.texCount      = texCount;
        g_kick.geomBytes     = geomBytes;
        g_kick.texBytes      = texBytes;
        g_kick.dtPresent     = dtPresent;
        g_kick.tStart        = tStart;
        g_kick.tGeomFlush    = tGeomFlush;
        g_kick.tBuild        = tBuild;
        g_kick.tTexFlush     = tTexFlush;
        g_kick.tAssign       = tAssign;
        g_kick.tKick         = nowMs();
        g_kick.bEnsure       = g_lastBuildEnsureLiveMs;   // Phase 0 build-split (skew-free w/ tBuild)
        g_kick.bEmit         = g_lastBuildEmitMs;
        g_kick.bTail         = g_lastBuildTailMs;
        g_kick.bTailEnsure   = g_lastBuildTailEnsureMs;
        g_kick.bTailScan     = g_lastBuildTailScanMs;
        g_kick.bTailAlpha    = g_lastBuildTailAlphaMs;
        g_kick.bKeys         = g_lastBuildKeys;
        g_kick.bCaptures     = g_lastBuildCaptures;
    }

    // Mode 3 worker body: build frame N's payload into the client-private scratch and PARK it —
    // no gate, no flush, no assign, no RPC, and NEVER a g_kick touch (g_kick belongs to the
    // main thread's in-flight fire in this mode). The main thread fires the park at the START
    // of frame N+1 (fireParked) with a restamped camera. g_frame is NOT minted here — the fire
    // mints it; the build's LRU/light-track uses of g_frame tolerate the ±1. The worker never
    // enters WK_GATE/WK_GEOM/... in this mode, so no watchdog variant is needed.
    void buildOnlyBody(IDirect3DDevice9* device) {
        g_park.valid = false;   // stale park never survives a new build attempt
        if (!device || !g_initOk || !g_enabled) {
            return;             // park stays invalid; the fire drops cleanly
        }
        MGE_ZoneScopedN("Forge produce build-only (park)");
        const double t0 = nowMs();
        std::uint32_t drawCount, skinnedCount, multiMapCount, lightCount, skyCount, alphaCount;
        IPC::FPFrame fpFrame;
        std::uint32_t fpDraws = 0, fpSkinnedDraws = 0, fpAlphaDraws = 0;
        bool fpHave = false;
        {
            MGE_ZoneScopedN("Forge build draw lists");
            markWorkerPhase(WK_BUILD_DRAW);   // ensureLive reads g_cache + LIVE scene graph
            buildGeometryDrawLists(drawCount, skinnedCount, multiMapCount, alphaCount);
            markWorkerPhase(WK_BUILD_LIGHT);
            lightCount = buildLightList();
            markWorkerPhase(WK_BUILD_SKY);
            skyCount = buildSkyDrawList();
            markWorkerPhase(WK_BUILD_FP);
            fpHave = buildFPFrame(fpFrame, fpDraws, fpSkinnedDraws, fpAlphaDraws);
        }
        g_park.epoch           = g_cellEpoch;
        g_park.bakeEye[0]      = DistantLand::eyePos.x;
        g_park.bakeEye[1]      = DistantLand::eyePos.y;
        g_park.bakeEye[2]      = DistantLand::eyePos.z;
        g_park.drawCount       = drawCount;
        g_park.skinnedCount    = skinnedCount;
        g_park.multiMapCount   = multiMapCount;
        g_park.lightCount      = lightCount;
        g_park.skyCount        = skyCount;
        g_park.alphaCount      = alphaCount;
        g_park.fpFrame         = fpFrame;      // shipped verbatim at fire (pose N + arm-cam N bundle)
        g_park.fpDraws         = fpDraws;
        g_park.fpSkinnedDraws  = fpSkinnedDraws;
        g_park.fpAlphaDraws    = fpAlphaDraws;
        g_park.fpHave          = fpHave;
        g_park.capturedEmitted = g_capturedEmitted;
        g_park.tBuildEnd       = nowMs();
        g_park.buildMs         = g_park.tBuildEnd - t0;
        g_park.valid           = true;
    }

    // Mode-3 park telemetry, logged every ~300 fires alongside [produce]: the fire's own
    // main-thread cost split + park age + drop causes. Fire ≤ ~1ms and parkAge ≈ one frame
    // are the design targets; drops should appear only at cell transitions.
    struct ParkStats {
        unsigned fires = 0, dropEpoch = 0, dropInvalid = 0;
        double fireMs = 0.0, geomF = 0.0, texF = 0.0, assign = 0.0, kick = 0.0;
        double parkAge = 0.0, buildMs = 0.0;
    };
    ParkStats g_parkStats;

    // ---- Tier 1b: fresh produce worker ------------------------------------------------------
    // A single dedicated std::thread that runs kickoffBody() off the MW main thread. Deliberately
    // NOT MGE::RenderThread — that worker owns a D3D9 device lock + a D3DCREATE_MULTITHREADED
    // device and proved crashy; this one touches NO D3D9 (the produce path is D3D9-free after
    // Tier 1a), so it needs no device lock. For Tier 1b the caller kicks then wait()s immediately
    // (fully serial, zero race); Tier 2 will drop that fence to overlap MW's frame. The mutex the
    // kick/wait handshake takes also publishes every write the worker made to g_kick + the cache
    // globals back to the main thread (happens-before) before the finish half reads them.
    class ProduceWorker {
    public:
        void ensure() {
            if (m_thread.joinable()) return;
            m_stop = false;
            m_thread = std::thread([this] { run(); });
        }
        // buildOnly (mode 3): run buildOnlyBody (park, no RPC) instead of kickoffBody.
        void kick(IDirect3DDevice9* device, bool buildOnly = false) {
            {
                std::lock_guard<std::mutex> lk(m_mx);
                m_device = device;
                m_buildOnly = buildOnly;
                m_hasJob = true;
            }
            m_cvJob.notify_one();
        }
        void wait() {
            std::unique_lock<std::mutex> lk(m_mx);
            m_cvDone.wait(lk, [this] { return !m_hasJob; });
        }
        void stop() {
            if (!m_thread.joinable()) return;
            {
                std::lock_guard<std::mutex> lk(m_mx);
                m_stop = true;
            }
            m_cvJob.notify_one();
            m_thread.join();
        }
    private:
        void run() {
            MGE_TracyNameThread("Forge Produce Worker");   // labels the worker's lane in Tracy
            // Identify this thread to waitFinishGate — only the produce worker blocks on the gate.
            g_produceThreadId.store(std::this_thread::get_id(), std::memory_order_relaxed);
            for (;;) {
                IDirect3DDevice9* device;
                bool buildOnly;
                {
                    std::unique_lock<std::mutex> lk(m_mx);
                    m_cvJob.wait(lk, [this] { return m_hasJob || m_stop; });
                    if (m_stop) return;
                    device = m_device;
                    buildOnly = m_buildOnly;
                }
                const double t0 = nowMs();
                if (buildOnly) {
                    buildOnlyBody(device);           // mode 3: park only, no shared memory, no g_kick
                } else {
                    kickoffBody(device);
                }
                markWorkerPhase(WK_IDLE);            // body returned (covers every body exit path)
                g_produceWorkerMs = nowMs() - t0;   // published under the lock below (happens-before)
                {
                    std::lock_guard<std::mutex> lk(m_mx);
                    m_hasJob = false;
                }
                m_cvDone.notify_one();
            }
        }
        std::thread              m_thread;
        std::mutex               m_mx;
        std::condition_variable  m_cvJob, m_cvDone;
        IDirect3DDevice9*        m_device = nullptr;
        bool                     m_hasJob = false;
        bool                     m_buildOnly = false;   // mode 3: this job parks, doesn't fire
        bool                     m_stop   = false;
    };
    ProduceWorker g_produceWorker;

    void doDeferredFinish();      // fwd (defined below); Phase 0 deferred wait
    void stashDeferredFinish();   // fwd (defined below); move g_kick's finish state to the holder

    // Drain an in-flight async (mode 2) produce. Idempotent + cheap when none is pending, so it can
    // be called unconditionally at the finish boundary. Records how long the main thread actually
    // blocked (0 == the produce was fully hidden under scene 0). MUST be called before any code
    // reads g_kick or kickoffPending()/finishDeferred() for control flow.
    void waitProduce() {
        if (!g_produceInFlight) return;
        // DEADLOCK INVARIANT: main must never block on the worker while the finish gate is shut.
        // The gate is armed at the kick (kickProduceEarly, inside frameSetupEarly) and opened by
        // doDeferredFinish in the dispatcher moments later — but any main-thread waitProduce()
        // landing in that window would park main on a worker that is itself parked on a gate only
        // main can open. That is the load freeze: initOnLoad -> collectDeferredFinish ->
        // waitProduce, reached without ever passing the dispatcher.
        //
        // Doing the pending finish here is not just the unwedge, it is the correct action: the
        // finish is precisely what the worker is waiting for, and it has to happen before the
        // frame can advance anyway. No-op once the dispatcher has already run it.
        doDeferredFinish();
        markMainPhase(MP_WAIT_PRODUCE);
        const double t0 = nowMs();
        g_produceWorker.wait();
        g_produceBlockedMs = nowMs() - t0;
        g_produceInFlight = false;
        // Periodic overlap read: workerRun = the produce's wall time on the worker; blocked = the
        // slice scene 0 could NOT hide (main stalled for it). blocked≈0 ⇒ produce fully overlapped.
        static unsigned s_n = 0;
        if (++s_n % 300 == 0) {
            LOG::logline(">> [produce] overlap: workerRun=%.2f blocked=%.2f ms (hidden=%.2f)",
                         g_produceWorkerMs, g_produceBlockedMs,
                         g_produceWorkerMs - g_produceBlockedMs);
        }
    }


    // Dispatcher: route the produce onto the fresh worker per g_produceMode (unless the legacy
    // UseRenderThread path owns threading). Mode 2 (OVERLAP) only defers the wait on early-kickoff
    // frames — there the kick fires at BeginScene(0) and the paired waitProduce() runs at
    // EndScene(0); every other frame fences here so g_kick is complete on return exactly as inline.
    void onStage0CompositeKickoff(IDirect3DDevice9* device) {
        markMainPhase(MP_KICKOFF);
        // Frame-start kick already dispatched this frame's produce: the ONLY thing left here is the
        // N-1 finish, which runs under the worker's build. Returning early is essential, not an
        // optimisation — the waitProduce() below would drain the produce we just kicked (blocking
        // main for the whole build) and clear g_produceInFlight, after which the kick guard would
        // read "nothing in flight" and dispatch a SECOND kickoffBody. That was two composite
        // kickoffs and two blocking waits in one frame (~24ms).
        if (g_earlyKicked) {
            doDeferredFinish();
            return;
        }

        // Never kick while a previous async produce is still in flight (would race m_device); the
        // normal EndScene waitProduce already drained it, this is a belt-and-braces no-op there.
        waitProduce();
        // Non-early dispatch (mode 0 inline / mode 1 fenced / mode 2 non-early): hand main's captured
        // alpha to the consume side now, worker drained. (The g_earlyKicked path returned above already
        // swapped in kickProduceEarly — never swap twice in one frame.)
        swapCaptureBuffers();

        // Phase 0 deferred wait: finish the PREVIOUS deferred frame HERE (main thread, late) — on
        // early-kickoff frames the BeginScene collect skipped it, so the host had the whole
        // BeginScene window to finish and this wait collapses to ~0. MUST precede kickoffBody's
        // g_kick reset (it consumes N-1's g_kick state) AND the worker dispatch (finishAndCopy is
        // D3D9, main-only). On non-early frames this is a no-op — collectDeferredFinish already ran.
        if (g_produceMode == 0 || Configuration.UseRenderThread) {
            doDeferredFinish();
            kickoffBody(device);
            return;
        }
        g_produceWorker.ensure();
        if (g_produceMode == 2 && DistantLand::earlyForgeKickoff) {
            // KICK FIRST, FINISH SECOND. The old order (finish N-1, then kick N) put the whole
            // deferred finish — host wait + Forge RT copy — in front of the worker, which fed a
            // loop: a late kick issues the host RPC late, so the host gets less overlap, so the
            // next finish waits longer, so the next kick is later still ([hb] render 1.4 -> 5.3,
            // overlap 8.1 -> 2.6). Kicking first breaks it: the ~3.5ms build now runs UNDER the
            // finish+copy instead of after it, and the host RPC goes out ~5ms earlier.
            //
            // Hand-off: the worker resets g_kick on entry, so N-1's finish state must be moved to
            // g_pendingFinish BEFORE the kick. The shared-RT ordering (host must not start frame N
            // until the copy released the RT) is held by the finish gate, which the worker waits on
            // immediately before renderSceneKickoff.
            // frameSetupEarly may already have kicked at the earliest possible point (see
            // kickProduceEarly) — then only the finish is left to do here.
            if (!g_produceInFlight) {
                stashDeferredFinish();
                armFinishGate();
                g_produceKickMs   = nowMs();
                g_produceInFlight = true;
                g_produceWorker.kick(device);    // async — drained at the NEXT frame's collect
            }
            doDeferredFinish();                  // runs under the worker's build; opens the gate
            return;
        }
        doDeferredFinish();
        // Mode 1 (FENCED) or mode 2/3 on a non-early frame: kick + wait immediately (serial).
        // For mode 3 this IS the priming/fallback path (menus, transitions, warm-up latch):
        // kickoffBody drops any parked payload at entry and finishes same-frame, exactly like
        // mode 2 non-early — the latch needs 2 eligible frames, so a serial frame always
        // precedes the first park fire and primes g_mainTex for the deferred blit.
        g_produceWorker.kick(device);
        g_produceWorker.wait();
    }

    // Kick the produce at FRAME START — the earliest point in the frame where its inputs exist.
    // Called from frameSetupEarly the instant the geometry-cache walk + frustum-visible set are
    // built (and after the grass cull, the one remaining scene-0 RPC), rather than waiting for
    // frameSetupEarly to return and the dispatcher to run. Everything the produce needs is ready
    // there; the rest of frameSetupEarly (render-thread kick, return path) is pure main-thread work
    // that now overlaps the build instead of delaying it.
    //
    // The N-1 finish deliberately does NOT run first: the host frame is a full frame ahead and ends
    // well before the worker's ~3.5ms build reaches shared memory. onStage0CompositeKickoff does
    // the finish moments later, under the build. gateOnPrevFinish is the proof — if the previous
    // frame ever does NOT end in time, it shows up as "Forge kickoff gate ms" in Tracy instead of
    // as corruption.
    void kickProduceEarly(IDirect3DDevice9* device) {
        // Mode 3 (PARK) requires frame-ahead: with it off nothing would ever fire the park
        // (fireParked guards on it), so decline here and let the dispatcher run the serial
        // fenced path — the same degradation mode 2 has on non-early frames.
        const bool mode3 = (g_produceMode == 3 && g_frameAheadLive);
        if ((g_produceMode != 2 && !mode3)
            || Configuration.UseRenderThread || !DistantLand::earlyForgeKickoff) {
            return;   // every other path is main-thread-serial and kicks from the dispatcher
        }
        waitProduce();          // belt-and-braces: never kick over an in-flight produce
        // Capture swap point is the same in both modes: in mode 3 the fire (frame start, before
        // this) already shipped + cleared the previous consume-side captures, so the swap hands
        // the worker exactly one frame of main-thread captures — invariants unchanged.
        swapCaptureBuffers();   // hand main's captured alpha to the consume side; fresh incoming for this frame
        g_earlyKicked = true;   // tells the dispatcher this frame is already dispatched
        g_produceWorker.ensure();
        if (mode3) {
            // PARK build: the worker never touches shared memory or g_kick, so there is no
            // finish state to stash (the frame-start fire owns g_kick; the collect already
            // consumed N-1's) and no gate to arm (it stays trivially open).
            g_produceKickMs   = nowMs();
            g_produceInFlight = true;
            g_produceWorker.kick(device, /*buildOnly=*/true);
            return;
        }
        stashDeferredFinish();  // hand N-1's finish state over before the worker resets g_kick
        armFinishGate();
        g_produceKickMs   = nowMs();
        g_produceInFlight = true;
        g_produceWorker.kick(device);
    }

    // Mode 3 PARK-AND-FIRE: fire the payload the worker parked LAST frame, at the START of
    // this frame (frameSetupEarly — camera fresh, latch decided, window closed by the collect,
    // channel free after the grass cull, classify/walk/build all still ahead). The host then
    // owns essentially the whole client frame: frame time = max(client CPU, host wall), not
    // the sum. The restamp inside flushAssignAndKick pairs frame-N geometry with the frame-N+1
    // camera, so there is no added camera/input latency — only 1-frame pose/frustum staleness.
    void fireParked(IDirect3DDevice9* device) {
        if (g_produceMode != 3 || !g_frameAheadLive || Configuration.UseRenderThread
            || !g_initOk || !g_enabled || !DistantLand::earlyForgeKickoff || !device) {
            return;
        }
        MGE_ZoneScopedN("Forge park fire");
        const double tStart = nowMs();
        const double dtPresent = (g_lastPresentMs > 0.0) ? (tStart - g_lastPresentMs) : 0.0;
        g_lastPresentMs = tStart;
        // The collect already consumed the previous fire (window closed, state stashed), so
        // g_kick is main-owned and stale here — reset it exactly as kickoffBody would, so a
        // dropped park below leaves a clean whole-frame no-op (rpcPending false).
        g_kick = KickState{};
        const unsigned frame = g_frame++;
        checkCellEpochAndPurge(frame);
        if (!g_park.valid) {
            ++g_parkStats.dropInvalid;   // build declined / already consumed / serial superseded
            return;
        }
        if (g_park.epoch != g_cellEpoch) {
            // Teleport / load door between build and fire: the parked payload is the OLD cell.
            // Drop it — one repeated composite frame, masked by the transition itself.
            g_park.valid = false;
            ++g_parkStats.dropEpoch;
            LOG::logline(">> [park] drop: epoch %u -> %u at frame %u (transition between build and fire)",
                         g_park.epoch, g_cellEpoch, frame);
            return;
        }
        const double parkAge = tStart - g_park.tBuildEnd;
        g_capturedEmitted = g_park.capturedEmitted;   // restore the build-time AT3 count for g_kick
        // tBuild == tStart: the build cost lives in the park ([park] build=), not this frame's
        // [hb] build bucket — the fire's own cost is exactly the geomF/texF/assign/kick split.
        flushAssignAndKick(device, frame, g_park.drawCount, g_park.skinnedCount,
                           g_park.multiMapCount, g_park.lightCount, g_park.skyCount,
                           g_park.alphaCount, g_park.fpFrame, g_park.fpDraws,
                           g_park.fpSkinnedDraws, g_park.fpAlphaDraws, g_park.fpHave,
                           g_park.bakeEye, /*parkFired=*/true, dtPresent, tStart, tStart);
        g_park.valid = false;   // consumed (fired, or skipped as empty — rebuilt this frame either way)
        // [park] telemetry from the stamps flushAssignAndKick just wrote (0 on an empty-payload
        // skip — those frames ship nothing and cost ~nothing, so folding them in is honest).
        g_parkStats.fireMs  += nowMs() - tStart;
        g_parkStats.parkAge += parkAge;
        g_parkStats.buildMs += g_park.buildMs;
        if (g_kick.rpcPending) {
            g_parkStats.geomF  += g_kick.tGeomFlush - g_kick.tBuild;
            g_parkStats.texF   += g_kick.tTexFlush - g_kick.tGeomFlush;
            g_parkStats.assign += g_kick.tAssign - g_kick.tTexFlush;
            g_parkStats.kick   += g_kick.tKick - g_kick.tAssign;
        }
        if (++g_parkStats.fires >= 300) {
            const double inv = 1.0 / g_parkStats.fires;
            LOG::logline(">> [park] %u fires avg: fire=%.2f (geomF=%.2f texF=%.2f assign=%.2f kick=%.2f) "
                         "parkAge=%.2f build=%.2f ms | drops epoch=%u invalid=%u",
                         g_parkStats.fires, g_parkStats.fireMs * inv, g_parkStats.geomF * inv,
                         g_parkStats.texF * inv, g_parkStats.assign * inv, g_parkStats.kick * inv,
                         g_parkStats.parkAge * inv, g_parkStats.buildMs * inv,
                         g_parkStats.dropEpoch, g_parkStats.dropInvalid);
            g_parkStats = ParkStats{};
        }
    }

    void onStage0CompositeFinish(IDirect3DDevice9* device) {
        markMainPhase(MP_COMPOSITE_FINISH);
        // Dev-key poll: once per display frame, at a stable point BEFORE the finish
        // consumers — normally the BeginScene(0) collect already polled (it runs first
        // in frame order and pollDevKeys dedups on the Present serial); this call only
        // wins on frames without a scene-0 collect. One consistent value per frame for
        // every ownership gate, including the frameSetupEarly early-kickoff latch.
        pollDevKeys();

        if (!g_kick.rpcPending && !g_kick.rpcEarlyFinished) {
            return;
        }
        MGE_ZoneScopedN("Forge composite finish");

        const FinishResult fr = finishAndCopy(g_kick);
        if (!fr.ok) {
            return;
        }
        compositeBlitMainTex(device);
        accumFrameStats(g_kick, fr, nowMs(), false, 0.0);
    }

    void onStage0Composite(IDirect3DDevice9* device) {
        // Fused kickoff+finish — the exact pre-split serial behaviour (A/B reference,
        // UseAsyncHostFrame=0). overlap ≈ 0 by construction.
        onStage0CompositeKickoff(device);
        onStage0CompositeFinish(device);
    }

    bool kickoffPending() {
        // g_produceInFlight covers the mode-2 window between the BeginScene(0) async kick and the
        // EndScene(0) waitProduce, so the late-kick guard never double-kicks before g_kick is set.
        return g_kick.rpcPending || g_produceInFlight;
    }

    // A deferred finish is pending in the HOLDER (stashDeferredFinish moved it there). Reads no
    // worker-owned state, so it is safe while a produce is in flight.
    bool finishDeferred() {
        return g_pendingFinish.deferFinish
            && (g_pendingFinish.rpcPending || g_pendingFinish.rpcEarlyFinished);
    }

    // Move this frame's deferred finish out of g_kick into the holder. MUST run on the main thread
    // before the produce worker is kicked (kickoffBody resets g_kick). No-op on non-deferred frames.
    void stashDeferredFinish() {
        if (!g_kick.deferFinish || (!g_kick.rpcPending && !g_kick.rpcEarlyFinished)) {
            return;
        }
        g_pendingFinish = g_kick;
        g_kick.deferFinish = g_kick.rpcPending = g_kick.rpcEarlyFinished = false;
    }

    void onFramePresented() {
        ++g_frameSerial;
    }

    void noteEnginePresentReturn() {
        g_presentReturnMs = nowMs();
        // Main now leaves our code and runs MW's un-zoned between-frame work (input/sim/AI/anim)
        // until the next BeginScene(0). A stall breadcrumbed here == wedged in MW itself (or MW
        // paused on focus loss), NOT in our pipeline — the key alt-tab discriminator.
        markMainPhase(MP_MW_FRAME);
    }

    void onFrameAheadCollect(IDirect3DDevice9* /*device*/) {
        // BeginScene(0), BEFORE frameSetupEarly: poll the dev keys first so the
        // earlyForgeKickoff latch and every suppression gate this frame see the fresh
        // values, then consume the PREVIOUS frame's deferred host render (finish + RT
        // copy). This closes the IPC window before any of the new frame's RPCs
        // (setWorldSpace, grass cull, geom/tex flushes) need the channel. The residual
        // wait here is ~0 whenever the MW frame outlasts the host frame — the success
        // metric of the whole pipeline (the [hb] render= bucket).
        //
        // A0 Cut-4 probe: close the Present→BeginScene(0) span (MW's un-zoned frame
        // start — input/sim/AI/animation) before anything else this frame runs.
        startWedgeWatchdog();       // idempotent; arms the main-thread freeze watchdog once
        probeFocusTransition();     // log alt-tab focus changes for freeze correlation
        markMainPhase(MP_FRAME_COLLECT);
        if (g_presentReturnMs > 0.0) {
            g_pendingMwStart = nowMs() - g_presentReturnMs;
            g_presentReturnMs = 0.0;
            g_lastMwStartStat = g_pendingMwStart;   // → host Stats panel next kickoff
            MGE_TracyPlot("MW frame start ms", g_pendingMwStart);
        }
        pollDevKeys();
        g_earlyKicked = false;   // new frame: no frame-start kick has fired yet
        // THE produce drain point (mode 2). The previous frame kicked the worker at its
        // BeginScene(0) and nothing in that frame waited for it, so it has had all of frame
        // N-1's tail + Present + MW's un-zoned frame start (mwstart, ~5.5ms of engine
        // input/sim/AI/animation) to run — the only overlap window left once Phase 2 emptied
        // MW's scenes. Waiting here also fixes the ordering: it precedes frameSetupEarly (which
        // rewalks the geometry cache + rebuilds the visible set the worker reads) and every
        // downstream reader of g_kick (doDeferredFinish consumes N-1's state; kickoffBody resets
        // it). No-op in modes 0/1 and on non-early frames, which fence at the kickoff.
        //
        // NI-read note: the worker reads the live scene graph, so it now spans mwstart(N), where
        // the engine mutates it. Accepted deliberately (we are a frame ahead and the host owns
        // the draw); g_produceMode 1 (FENCED) is the instant A/B back to an in-frame fence.
        waitProduce();
        // Phase 0: the previous frame's deferred finish NO LONGER runs here. It moves to
        // collectDeferredFinish (non-early / not-ready frames, after frameSetupEarly latches the
        // gate) or to onStage0CompositeKickoff (early frames, run late). See collectDeferredFinish.
        //
        // EXCEPT mode-3 park fires: their window MUST close HERE, before frameSetupEarly's RPCs
        // (setWorldSpace, grass) need the channel — the next fire happens right after those, at
        // the frame start. The host had fire → collect ≈ a whole frame + present + mwstart to
        // render, so the residual wait here is ~0 (the [hb] render= bucket). stash moves the
        // fire's g_kick state to the holder first (waitProduce above re-published g_kick if a
        // drain-side early finish touched it on the worker).
        //
        // Guard on `mode == 3` OR the parkFired state — NOT parkFired alone. The mode==3 arm is
        // the FREEZE FIX for a live 2->3 switch (panel/NUMPAD8): the just-drained frame carries a
        // mode-2 kickoff's deferred finish in g_kick with parkFired=FALSE. Keyed on parkFired only,
        // this block skipped it, then fireParked's `g_kick = KickState{}` DISCARDED that finish —
        // orphaning the host's open IPC window forever (host frame vanishes, client keeps running).
        // Draining any stranded deferred finish here closes the window before fireParked resets
        // g_kick. The parkFired arm still covers the reverse 3->2 switch (a mode-3 fire's finish
        // stranded in g_kick after the mode flips away). stash/finish are no-ops when g_kick holds
        // nothing deferred (priming frames), so the mode==3 arm is harmless in steady state.
        if (g_produceMode == 3 || g_kick.parkFired || g_pendingFinish.parkFired) {
            stashDeferredFinish();
            doDeferredFinish();
            g_kick.parkFired = g_pendingFinish.parkFired = false;   // consumed — reset the keyed state
        }
    }

    // Finish + copy the previous deferred host frame (frame-ahead). Main thread ONLY — finishAndCopy
    // does a D3D9 RT copy. Shared by collectDeferredFinish and onStage0CompositeKickoff. No-op unless
    // a deferred finish is pending; consumes g_kick's N-1 state, so it MUST run before kickoffBody
    // resets g_kick for the new frame.
    void doDeferredFinish() {
        markMainPhase(MP_DEFERRED_FINISH);
        if (!finishDeferred()) {
            // Nothing to release, but the worker may be parked on the gate (armed unconditionally
            // at the kick) — open it or the host RPC never goes out.
            openFinishGate();
            return;
        }
        MGE_ZoneScopedN("Forge deferred finish");
        g_pendingFinish.deferFinish = false;
        const FinishResult fr = finishAndCopy(g_pendingFinish);
        // The shared host RT is released the moment finishAndCopy returns (success or not) — the
        // copy is the last thing that reads it. Let the worker issue frame N's kickoff now.
        openFinishGate();
        if (!fr.ok) {
            // Host death / copy failure: g_mainTexValid is already down, so the next
            // deferred blit skips — the same empty-world frame as today's finish-failure
            // path; ownership gates release via g_initOk/ServerLost as before.
            LOG::logline("!! [pipe] deferred finish failed (host dead?) — composite skipped until recovery");
            g_lastBlitMs = 0.0;
            return;
        }
        accumFrameStats(g_pendingFinish, fr, nowMs(), true, g_lastBlitMs);
        g_lastBlitMs = 0.0;
    }

    void collectDeferredFinish(IDirect3DDevice9* /*device*/) {
        // Non-early / not-ready frames: finish the previous deferred frame NOW, at BeginScene,
        // before the MGE pipeline (selectDistantCell/culls/depth) reuses the IPC channel. Early
        // frames skip this and defer to the kickoff. See doDeferredFinish.
        //
        // A produce can still be in flight here now that the wait spans the frame boundary — the
        // load/reset backstop (initOnLoad) reaches this without passing onFrameAheadCollect. Drain
        // it first, then stash: these paths never went through the kick-first dispatcher, so a
        // deferred finish may still be sitting in g_kick rather than the holder.
        waitProduce();
        stashDeferredFinish();
        doDeferredFinish();
    }

    void onFrameAheadBlit(IDirect3DDevice9* device) {
        markMainPhase(MP_BLIT);
        // EndScene(0) composite point on a deferred frame: no finish, no IPC, no wait —
        // just lay the PREVIOUS host frame (still valid in g_mainTex; the composite
        // never reads the shared RT directly) over MW's backbuffer. Skipped while
        // g_mainTexValid is down (host death / copy failure) — the same empty-world
        // frame as today's finish-failure path.
        if (!g_mainTexValid) {
            g_lastBlitMs = 0.0;
            return;
        }
        g_lastBlitMs = compositeBlitMainTex(device);
    }

    bool wantsGeometryCapture() {
        return g_initOk && g_geomVec.has_value();
    }

    bool forgeOwnsFrame() {
        // The one mode predicate — see the header. Seam live + composite ON (F11) means the host
        // owns opaque world, distant land, sky, water and depth; MW's own draws for all of it are
        // overwritten by the composite, so MGE suppresses them. F11 off (or a dead host / failed
        // seam → g_initOk false) releases every suppression and MW renders vanilla.
        return g_initOk && g_enabled;
    }

    bool hasCompositeFrame() {
        return g_mainTexValid;
    }

    void discardPendingCaptures() {
        // Menu freeze: no produce runs, so swapCaptureBuffers() — the ONLY place the incoming
        // capture buffers are cleared — never fires, while captureAlphaDraw keeps appending for
        // every blended world DIP MW still issues behind the menu. Left alone these grow for as
        // long as the menu is open. Nothing will consume them (the frame they belong to is never
        // built), so drop them. Called once per freeze frame, before that frame's appends, which
        // bounds the capture side to a single frame's worth.
        //
        // Deliberately NOT swapCaptureBuffers(): that would also rotate the alpha-dedup sets, and
        // the worker is not running to rebuild them.
        g_capInVerts.clear();
        g_capInIdx.clear();
        g_capInRecs.clear();
        g_capInTexMemo.clear();
    }

    bool wantsFPCapture() {
        // FP1a: seam live + geometry capture up + composite ON (F11) + ini flag + FIRST person.
        // Gates the cache's armCamera-root walk, the FP draw-list build and the fp wire crossing.
        return g_initOk && g_geomVec.has_value() && g_enabled
            && Configuration.ForgeFPPass && !MWBridge::get()->is3rdPerson();
    }

    bool wantsFPSuppression() {
        // FP1b: suppress MW's own arm draws ONLY while the host FP pass is actually able
        // to ship them (capture live + camera math validated) — the arms must never
        // vanish without a replacement. g_fpSuppressLive seeds from ForgeFPSuppress and
        // flips live on numpad-/ for the A/B.
        return wantsFPCapture() && g_fpSuppressLive && g_fpCamValid;
    }

    // FP camera diag latch (see the g_fpDiag* block above): the proxy calls these from
    // Clear (z-only clear in scenes >= 1 → arm) and SetTransform (first view+proj submitted
    // while armed → latch, disarm when both landed). Self-gates on wantsFPCapture so it's
    // inert unless the FP pass is live.
    void noteFPZClear() {
        if (!wantsFPCapture()) {
            return;
        }
        g_fpDiagArmed = true;
        g_fpDiagHaveView = false;
        g_fpDiagHaveProj = false;
    }

    void noteFPSceneTransform(bool isProj, const D3DMATRIX* m) {
        if (!g_fpDiagArmed || !m) {
            return;
        }
        if (isProj && !g_fpDiagHaveProj) {
            g_fpDiagProj = *m;
            g_fpDiagHaveProj = true;
        } else if (!isProj && !g_fpDiagHaveView) {
            g_fpDiagView = *m;
            g_fpDiagHaveView = true;
        }
        if (g_fpDiagHaveView && g_fpDiagHaveProj) {
            g_fpDiagArmed = false;
            g_fpDiagAge = 0;
        }
    }

    void captureAlphaDraw(const RenderedState* rs, const FragmentState* frs) {
        if (!Configuration.ForgeAlphaCapture) return;
        if (!g_initOk || !g_enabled || !g_capturedVec) return;
        if (!rs || !frs) return;
        // HW-skinned blends (ghosts) excluded — the bind-pose VB here is the wrong pose (a
        // separate follow-up). Need an indexed TRIANGLELIST with a real stride (TRISTRIP/FAN and
        // non-indexed particle DIPs are dropped; a counter would reveal if MW emits any).
        if (rs->vertexBlendState != 0) return;
        if (rs->primType != D3DPT_TRIANGLELIST) return;
        if (!rs->vb || !rs->ib || rs->vbStride == 0) return;
        if ((rs->fvf & D3DFVF_POSITION_MASK) != D3DFVF_XYZ) return;   // untransformed XYZ only
        if (rs->vertCount == 0 || rs->primCount == 0) return;

        // Old-msoc double-draw guard: this DIP is a duplicate iff the host already draws a cached
        // blended shape with the same (GPU texture, vertexCount) — the cache draws from its own VB
        // copies so rs->vb never matches, but (tex, vertCount) does (the vertCount term kills the
        // shared-texture false positive — particle counts vary). Read the ACTIVE (prev-frame) set,
        // NOT g_alphaDedup — the worker is concurrently rebuilding g_alphaDedup and reading it here
        // raced (glow-window white/orange flicker). swapCaptureBuffers hands over a stable snapshot.
        const std::uint64_t dedupKey =
            ((std::uint64_t)(std::uintptr_t)rs->texture << 32) | (std::uint64_t)rs->vertCount;
        if (g_alphaDedupActive.find(dedupKey) != g_alphaDedupActive.end()) {
            // [alpha-dedup] probe: log the first few victims with enough identity to tell a real
            // cached-blend duplicate from an aliased particle draw (a particle's vertCount moves
            // frame to frame; a cached blend's does not).
            static std::uint32_t s_logged = 0;
            if (s_logged < 12) {
                ++s_logged;
                const char* nm = MGE::GeometryCache::resolveTextureName(rs->texture);
                LOG::logline("!! [alpha-dedup] dropped DIP tex=%s vc=%u tri=%u (key collision)",
                             nm ? nm : "(unnamed)", rs->vertCount, rs->primCount);
            }
            ++g_capDedupDrops;
            return;
        }

        // Caps: keep the shared captured buffers within their single-chunk budget. Drop WHOLE
        // (never partial) on overflow so an index range can't dangle past the shipped vertex window.
        const std::uint32_t idxCount = rs->primCount * 3;
        if (g_capInRecs.size() >= IPC::kMaxCapturedAlphaDraws
            || g_capInVerts.size() + rs->vertCount > IPC::kMaxCapturedAlphaVerts
            || g_capInIdx.size() + idxCount > IPC::kMaxCapturedAlphaIndices) {
            if (g_capDropCap++ == 0) {
                LOG::logline("!! [alpha-cap] captured-alpha cap hit (draws/verts/indices) — dropping rest this session-window");
            }
            return;
        }

        // Texture slot: rs->texture is the proxy realTexture pointer, identical to the GPU texture
        // the cache walk registered in g_textureNameMap. Resolve name -> bindless slot (per-frame
        // memo). No name -> slot 0 (host default white): a visible white particle beats an invisible
        // one, and NiFlipController textures self-correct after the first walk registers them.
        std::uint32_t texIndex;
        auto mit = g_capInTexMemo.find(rs->texture);
        if (mit != g_capInTexMemo.end()) {
            texIndex = mit->second;
        } else {
            const char* name = MGE::GeometryCache::resolveTextureName(rs->texture);
            texIndex = name ? resolveTextureSlot(name) : 0u;
            if (!name && g_capNoName++ < 20) {
                LOG::logline("!! [alpha-cap] no source name for tex=%p (white)", (void*)rs->texture);
            }
            g_capInTexMemo.emplace(rs->texture, texIndex);
        }

        // FVF offsets within the source stride (XYZ at 0; then NORMAL, PSIZE, DIFFUSE, SPECULAR,
        // TEX in D3D order). We read pos/normal/diffuse/UV0.
        const UINT stride = rs->vbStride;
        UINT off = 12;   // past XYZ
        const bool hasNorm = (rs->fvf & D3DFVF_NORMAL) != 0; const UINT normOff = off; if (hasNorm) off += 12;
        if (rs->fvf & D3DFVF_PSIZE) off += 4;
        const bool hasCol = (rs->fvf & D3DFVF_DIFFUSE) != 0;  const UINT colOff = off; if (hasCol) off += 4;
        if (rs->fvf & D3DFVF_SPECULAR) off += 4;
        const UINT texCount = (rs->fvf & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT;
        const bool hasUV = texCount >= 1; const UINT uvOff = off;

        // vColSource forward map (inverse proven at rendercachedcolor.cpp:141-142). The captured
        // frag reads real vertex color for source 1 and vertex alpha for any nonzero source, so we
        // write the REAL captured D3DCOLOR whenever vColSource != 0 (particles fade via vertex alpha).
        std::uint32_t vColSource = 0;
        if (hasCol) {
            if (rs->matSrcDiffuse == D3DMCS_COLOR1) vColSource = 2;
            else if (rs->matSrcEmissive == D3DMCS_COLOR1) vColSource = 1;
        }

        // TEMP AT3 lighting-mismatch diagnostic (remove after triage): dump each first-seen captured
        // source-name's full state — FVF layout (rule out a wrong normal offset), lighting/vColSource/
        // material (the frag's lit path), and whether stage 1+ carries a texture (multi-map: the frag
        // only samples the BASE map, so a dark/detail/glow blend renders wrong). Keyed on the name ptr.
        {
            static std::unordered_set<const void*> s_capDiagSeen;
            const char* dname = MGE::GeometryCache::resolveTextureName(rs->texture);
            const void* dkey = dname ? (const void*)dname : (const void*)rs->texture;
            if (s_capDiagSeen.insert(dkey).second && s_capDiagSeen.size() <= 64) {
                const int st1 = (int)frs->stage[1].colorOp, st2 = (int)frs->stage[2].colorOp, st3 = (int)frs->stage[3].colorOp;
                LOG::logline(">> [cap-diag] %s fvf=0x%X stride=%u norm=%d@%u col=%d@%u uv=%d@%u "
                             "vcs=%u lit=%u matD=(%.2f,%.2f,%.2f) matA=%.2f stageOps=[%d,%d,%d,%d]",
                             dname ? dname : "(null)", (unsigned)rs->fvf, (unsigned)stride,
                             (int)hasNorm, (unsigned)normOff, (int)hasCol, (unsigned)colOff, (int)hasUV, (unsigned)uvOff,
                             vColSource, (unsigned)rs->useLighting,
                             frs->material.diffuse.r, frs->material.diffuse.g, frs->material.diffuse.b,
                             frs->material.diffuse.a, (int)frs->stage[0].colorOp, st1, st2, st3);
            }
        }

        // Lock VB whole-remainder from vbOffset + IB (computeBoundingBox's exact flags); graceful drop.
        void* pVerts = nullptr;
        if (FAILED(rs->vb->Lock(rs->vbOffset, 0, &pVerts, D3DLOCK_READONLY | D3DLOCK_NOSYSLOCK))) {
            if (g_capDropLock++ == 0) LOG::logline("!! [alpha-cap] VB lock failed — dropping");
            return;
        }
        void* pIdx = nullptr;
        if (FAILED(rs->ib->Lock(0, 0, &pIdx, D3DLOCK_READONLY | D3DLOCK_NOSYSLOCK))) {
            rs->vb->Unlock();
            if (g_capDropLock++ == 0) LOG::logline("!! [alpha-cap] IB lock failed — dropping");
            return;
        }
        D3DINDEXBUFFER_DESC ibd; rs->ib->GetDesc(&ibd);
        const bool is16 = (ibd.Format == D3DFMT_INDEX16);

        // Copy the vertex window [baseIndex+minIndex, +vertCount) → captured VB, building the
        // model-space bbox for the centroid. (The host adds vertexBase to each index, so indices
        // are rebased to 0 within this window below.)
        const std::uint32_t vBase = (std::uint32_t)g_capInVerts.size();
        const UINT srcVBase = rs->baseIndex + rs->minIndex;
        float mn[3] = { 1e30f, 1e30f, 1e30f }, mx[3] = { -1e30f, -1e30f, -1e30f };
        for (UINT j = 0; j < rs->vertCount; ++j) {
            const BYTE* v = (const BYTE*)pVerts + (size_t)(srcVBase + j) * stride;
            const float* p = (const float*)v;
            IPC::GeomVertexWire out;
            out.px = p[0]; out.py = p[1]; out.pz = p[2];
            if (hasNorm) { const float* n = (const float*)(v + normOff); out.nx = n[0]; out.ny = n[1]; out.nz = n[2]; }
            else { out.nx = 0.0f; out.ny = 0.0f; out.nz = 1.0f; }
            out.color = (vColSource != 0) ? *(const std::uint32_t*)(v + colOff) : 0xFFFFFFFFu;
            if (hasUV) { const float* t = (const float*)(v + uvOff); out.u = t[0]; out.v = t[1]; }
            else { out.u = 0.0f; out.v = 0.0f; }
            g_capInVerts.push_back(out);
            for (int c = 0; c < 3; ++c) { mn[c] = std::min(mn[c], p[c]); mx[c] = std::max(mx[c], p[c]); }
        }

        // Copy primCount*3 indices from startIndex, rebased by -minIndex → [0, vertCount). Any
        // rebased value out of uint16 → drop the whole record (roll back the pushed verts+indices).
        const std::uint32_t iBase = (std::uint32_t)g_capInIdx.size();
        bool idxDrop = false;
        for (UINT i = 0; i < idxCount; ++i) {
            const UINT raw = is16 ? ((const WORD*)pIdx)[rs->startIndex + i]
                                  : ((const DWORD*)pIdx)[rs->startIndex + i];
            const long rebased = (long)raw - (long)rs->minIndex;
            if (rebased < 0 || rebased > 0xFFFF) { idxDrop = true; break; }
            g_capInIdx.push_back((std::uint16_t)rebased);
        }
        rs->ib->Unlock();
        rs->vb->Unlock();

        if (idxDrop) {
            g_capInVerts.resize(vBase);
            g_capInIdx.resize(iBase);
            if (g_capDropIdx32++ == 0) LOG::logline("!! [alpha-cap] index rebase out of uint16 — dropping");
            return;
        }
        if (mn[0] > mx[0]) { g_capInVerts.resize(vBase); g_capInIdx.resize(iBase); return; }

        // World-space centroid via worldTransforms[0] (model-space bbox center → world). The
        // camera-relative subtraction is applied at emit with the CURRENT eyePos (no swim).
        const float cx = 0.5f * (mn[0] + mx[0]), cy = 0.5f * (mn[1] + mx[1]), cz = 0.5f * (mn[2] + mx[2]);
        const D3DXMATRIX& w = rs->worldTransforms[0];

        CapturedAlphaRec rec;
        rec.vertexBase = vBase; rec.vertexCount = rs->vertCount;
        rec.indexBase = iBase;  rec.indexCount = idxCount;
        memcpy(rec.world, &w, 16 * sizeof(float));   // D3DXMATRIX is row-major (host convention)
        rec.centroid[0] = cx * w._11 + cy * w._21 + cz * w._31 + w._41;
        rec.centroid[1] = cx * w._12 + cy * w._22 + cz * w._32 + w._42;
        rec.centroid[2] = cx * w._13 + cy * w._23 + cz * w._33 + w._43;
        rec.texIndex = texIndex;
        rec.srcBlend = rs->srcBlend; rec.destBlend = rs->destBlend;
        rec.alphaRef = rs->alphaTest
            ? (float)(rs->alphaRef + (rs->alphaFunc == D3DCMP_GREATER ? 1 : 0)) / 255.0f : 0.0f;
        rec.vColSource = vColSource;
        // FFP-unlit approximation (verify in-game): unlit draws (particles) get zero diffuse/ambient
        // + white emissive so the frag emits texture * vcol; lit draws pass the material through.
        if (rs->useLighting == 0) {
            rec.matDiffuse[0] = rec.matDiffuse[1] = rec.matDiffuse[2] = 0.0f;
            rec.matAmbient[0] = rec.matAmbient[1] = rec.matAmbient[2] = 0.0f;
            rec.matEmissive[0] = rec.matEmissive[1] = rec.matEmissive[2] = 1.0f;
        } else {
            rec.matDiffuse[0]  = frs->material.diffuse.r;  rec.matDiffuse[1]  = frs->material.diffuse.g;  rec.matDiffuse[2]  = frs->material.diffuse.b;
            rec.matAmbient[0]  = frs->material.ambient.r;  rec.matAmbient[1]  = frs->material.ambient.g;  rec.matAmbient[2]  = frs->material.ambient.b;
            rec.matEmissive[0] = frs->material.emissive.r; rec.matEmissive[1] = frs->material.emissive.g; rec.matEmissive[2] = frs->material.emissive.b;
        }
        rec.matAlpha = frs->material.diffuse.a;
        g_capInRecs.push_back(rec);
    }

    // Part A upload accounting: the geometry cache tags each host geom reship by cause.
    void noteUpload(std::uint8_t cat, std::uint32_t bytes) {
        if (cat >= kUpCount) cat = kUpOther;
        g_upFrame.parts[cat] += 1; g_upFrame.bytes[cat] += bytes;
    }

    void captureGeometry(std::uint32_t key, std::uint16_t revision, std::uint32_t modelId,
                         const IPC::GeomVertexWire* verts, std::uint32_t vertexCount,
                         const std::uint16_t* indices, std::uint32_t indexCount,
                         bool forceReupload,
                         const std::uint8_t* uvAnim, std::uint16_t uvAnimBytes) {
        if (!wantsGeometryCapture() || !verts || !indices || !vertexCount || !indexCount) {
            return;
        }
        // Skip only if the SAME object (modelId), same shape (vertexCount) and same revision
        // was already shipped. Keying on revision alone aliased recycled NiTriShape* keys (a
        // freed object's key+slot inherited by a new mesh with a colliding revisionID).
        // forceReupload (SK1 sky dome) bypasses this — its vertex colours change every frame
        // without a revisionID bump, so the dedup would otherwise freeze the gradient.
        auto rev = g_uploadedRev.find(key);
        if (!forceReupload && rev != g_uploadedRev.end() && rev->second.id == modelId &&
            rev->second.vc == vertexCount && rev->second.rev == revision) {
            return;
        }
        // Stable host slot per cache key (reused on re-upload so the host frees
        // and rebuilds in place).
        std::uint32_t slot;
        auto ks = g_keySlot.find(key);
        if (ks != g_keySlot.end()) {
            slot = ks->second.slot;
        } else {
            slot = g_nextSlot++;
            g_keySlot.emplace(key, SlotInfo{ slot });
        }

        IPC::GeomPartWire hdr = {};
        hdr.slot        = slot;
        hdr.revisionID  = revision;
        hdr.vertexCount = vertexCount;
        hdr.indexCount  = indexCount;
        if (uvAnim && uvAnimBytes) {
            hdr.flags      |= IPC::kGeomFlagUVAnim;
            hdr.uvAnimBytes = uvAnimBytes;
        }

        drainPendingIfFull();
        const std::size_t vbBytes = (std::size_t)vertexCount * sizeof(IPC::GeomVertexWire);
        const std::size_t ibBytes = (std::size_t)indexCount * sizeof(std::uint16_t);
        const std::size_t at = g_pendingBlob.size();
        g_pendingBlob.resize(at + sizeof(hdr) + vbBytes + ibBytes + hdr.uvAnimBytes);
        std::uint8_t* dst = g_pendingBlob.data() + at;
        memcpy(dst, &hdr, sizeof(hdr));            dst += sizeof(hdr);
        memcpy(dst, verts, vbBytes);               dst += vbBytes;
        memcpy(dst, indices, ibBytes);             dst += ibBytes;
        if (hdr.uvAnimBytes) { memcpy(dst, uvAnim, hdr.uvAnimBytes); }

        ++g_pendingParts;
        g_uploadedRev[key] = { modelId, vertexCount, revision };
    }

    void captureSkinnedGeometry(std::uint32_t key, std::uint16_t revision, std::uint32_t modelId,
                                const IPC::SkinnedVertexWire* verts, std::uint32_t vertexCount,
                                const std::uint16_t* indices, std::uint32_t indexCount,
                                std::uint32_t numBones) {
        if (!wantsGeometryCapture() || !verts || !indices || !vertexCount || !indexCount || numBones == 0) {
            return;
        }
        // Skip only if the same object+shape+revision was already shipped (shared map with
        // captureGeometry); identity (modelId,vc) guards against recycled NiTriShape* keys.
        auto rev = g_uploadedRev.find(key);
        if (rev != g_uploadedRev.end() && rev->second.id == modelId &&
            rev->second.vc == vertexCount && rev->second.rev == revision) {
            return;
        }
        // Stable host slot per cache key (shared slot map / nextSlot with the static path).
        std::uint32_t slot;
        auto ks = g_keySlot.find(key);
        if (ks != g_keySlot.end()) {
            slot = ks->second.slot;
        } else {
            slot = g_nextSlot++;
            g_keySlot.emplace(key, SlotInfo{ slot });
        }

        IPC::GeomPartWire hdr = {};
        hdr.slot        = slot;
        hdr.revisionID  = revision;
        hdr.flags       = IPC::kGeomFlagSkinned;
        hdr.vertexCount = vertexCount;
        hdr.indexCount  = indexCount;
        hdr.numBones    = static_cast<std::uint16_t>(numBones);

        drainPendingIfFull();
        const std::size_t vbBytes = (std::size_t)vertexCount * sizeof(IPC::SkinnedVertexWire);
        const std::size_t ibBytes = (std::size_t)indexCount * sizeof(std::uint16_t);
        const std::size_t at = g_pendingBlob.size();
        g_pendingBlob.resize(at + sizeof(hdr) + vbBytes + ibBytes);
        std::uint8_t* dst = g_pendingBlob.data() + at;
        memcpy(dst, &hdr, sizeof(hdr));            dst += sizeof(hdr);
        memcpy(dst, verts, vbBytes);               dst += vbBytes;
        memcpy(dst, indices, ibBytes);

        ++g_pendingParts;
        g_uploadedRev[key] = { modelId, vertexCount, revision };
    }

    void captureMultiMapGeometry(std::uint32_t key, std::uint16_t revision, std::uint32_t modelId,
                                 const IPC::GeomVertexWireMM* verts, std::uint32_t vertexCount,
                                 const std::uint16_t* indices, std::uint32_t indexCount,
                                 const std::uint8_t* uvAnim, std::uint16_t uvAnimBytes) {
        if (!wantsGeometryCapture() || !verts || !indices || !vertexCount || !indexCount) {
            return;
        }
        // Dedup on (modelId, vertexCount, revision) — shared map with captureGeometry; identity
        // guards against recycled NiTriShape* keys (see captureGeometry).
        auto rev = g_uploadedRev.find(key);
        if (rev != g_uploadedRev.end() && rev->second.id == modelId &&
            rev->second.vc == vertexCount && rev->second.rev == revision) {
            return;
        }
        // Stable host slot per cache key (shared slot map / nextSlot with the static path).
        std::uint32_t slot;
        auto ks = g_keySlot.find(key);
        if (ks != g_keySlot.end()) {
            slot = ks->second.slot;
        } else {
            slot = g_nextSlot++;
            g_keySlot.emplace(key, SlotInfo{ slot });
        }

        IPC::GeomPartWire hdr = {};
        hdr.slot        = slot;
        hdr.revisionID  = revision;
        hdr.flags       = IPC::kGeomFlagMultiMap;
        hdr.vertexCount = vertexCount;
        hdr.indexCount  = indexCount;
        if (uvAnim && uvAnimBytes) {
            hdr.flags      |= IPC::kGeomFlagUVAnim;
            hdr.uvAnimBytes = uvAnimBytes;
        }

        drainPendingIfFull();
        const std::size_t vbBytes = (std::size_t)vertexCount * sizeof(IPC::GeomVertexWireMM);
        const std::size_t ibBytes = (std::size_t)indexCount * sizeof(std::uint16_t);
        const std::size_t at = g_pendingBlob.size();
        g_pendingBlob.resize(at + sizeof(hdr) + vbBytes + ibBytes + hdr.uvAnimBytes);
        std::uint8_t* dst = g_pendingBlob.data() + at;
        memcpy(dst, &hdr, sizeof(hdr));            dst += sizeof(hdr);
        memcpy(dst, verts, vbBytes);               dst += vbBytes;
        memcpy(dst, indices, ibBytes);             dst += ibBytes;
        if (hdr.uvAnimBytes) { memcpy(dst, uvAnim, hdr.uvAnimBytes); }

        ++g_pendingParts;
        g_uploadedRev[key] = { modelId, vertexCount, revision };
    }

    void shutdown() {
        // Tier 1b: stop the produce worker BEFORE draining/releasing anything it may touch —
        // no-op if it was never started (produce-off-main stayed OFF the whole session).
        g_produceWorker.stop();
        // Never tear down with a host RenderFrame still pending (deferred or not):
        // drain it so the host isn't mid-frame and the window guard isn't latched
        // if the seam comes back up.
        if (g_client && g_kick.rpcPending) {
            g_client->renderSceneFinish(nullptr);
        }
        if (g_hostZoneOpen) { MGE_TracyHostFrameEnd(g_hostZoneCtx); g_hostZoneOpen = false; }
        g_kick = KickState{};
        releaseAll();
        g_geomVec.reset();
        g_drawVec.reset();
        g_skinnedVec.reset();
        g_multiMapVec.reset();
        g_lightVec.reset();
        g_skyVec.reset();
        g_alphaVec.reset();
        g_fpDrawVec.reset();
        g_fpSkinnedVec.reset();
        g_fpAlphaVec.reset();
        g_texVec.reset();
        g_pendingBlob.clear();
        g_pendingBlob.shrink_to_fit();
        g_drawScratch.clear();
        g_drawScratch.shrink_to_fit();
        g_skinnedScratch.clear();
        g_skinnedScratch.shrink_to_fit();
        g_multiMapScratch.clear();
        g_multiMapScratch.shrink_to_fit();
        g_lightScratch.clear();
        g_lightScratch.shrink_to_fit();
        g_skyScratch.clear();
        g_skyScratch.shrink_to_fit();
        g_alphaScratch.clear();
        g_alphaScratch.shrink_to_fit();
        g_fpDrawScratch.clear();
        g_fpDrawScratch.shrink_to_fit();
        g_fpSkinnedScratch.clear();
        g_fpSkinnedScratch.shrink_to_fit();
        g_fpAlphaScratch.clear();
        g_fpAlphaScratch.shrink_to_fit();
        g_texPendingBlob.clear();
        g_texPendingBlob.shrink_to_fit();
        g_pendingParts = 0;
        g_texPendingCount = 0;
        g_keySlot.clear();
        g_uploadedRev.clear();
        g_texSlot.clear();
        g_slotName.clear();
        g_slotLastUsed.clear();
        g_nextSlot = 0;
        g_nextTexSlot = 1;
        g_initOk = false;
        g_enabled = false;
    }
}

// ---------------------------------------------------------------------------
// Client-side Dear ImGui dev panel — drawn by ImGuiWater::onPresent under the F10
// overlay, alongside the Water/Foam panel. Replaces the hardware-key seam A/B
// toggles with clickable controls (some keys, e.g. Scroll Lock for host-cull, are
// missing on compact keyboards). Defined here (not imgui_water.cpp) so it can reach
// the file-static seam state + DistantLand::hostCullOnly directly; declared extern
// in imgui_water.cpp. Each control mirrors the equivalent [seam] key in pollDevKeys,
// logging on change so the log reads the same whether toggled by key or click.
// ---------------------------------------------------------------------------
void DrawForgeDevPanel() {
    if (!ImGui::Begin("Forge Dev")) { ImGui::End(); return; }

    // Flip a bool AND log on change (matches the key handlers' >> [seam] lines).
    auto logCheck = [](const char* label, bool& v, const char* msg) {
        if (ImGui::Checkbox(label, &v))
            LOG::logline(">> [seam] %s %s", msg, v ? "ON" : "OFF");
    };

    ImGui::Separator();
    ImGui::Text("Phase 1 - MW-only pipeline");
    logCheck("Host-cull only (frustum set)", DistantLand::hostCullOnly,
             "host-cull-only (Phase 1, frustum set)");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "ON: the Forge produce feeds off the self-contained frustum set (full refresh\n"
            "walk + whole-cache frustum cull); the host Hi-Z GPU cull owns occlusion. The\n"
            "engine MSOC classify is bypassed (was Scroll Lock).\n"
            "OFF: known-good MSOC / live-draw-build path.");

    // MW-ONLY-UI: how much of MW's world the ENGINE is forbidden to traverse. Phase 2 stopped
    // MGE drawing and the proxy rejects MW's draws one by one, but MW still walks its whole
    // scene graph and issues every call first — that traversal is the mwsky/mwdraws time.
    // A combo, not a key: the level must be READABLE. It was invisible before, a level-1
    // session got read as level 3, and the conclusion drawn from it was wrong.
    // INDEPENDENT checkboxes, not a ladder: a cumulative level cannot attribute a symptom to a
    // root. "Smoke vanished at level 2" only ever implied pick, because level 2 culled landscape
    // AND pick together — the conclusion was an inference, not an observation. One box per root
    // makes each answer direct.
    ImGui::Separator();
    ImGui::Text("MW world traversal (appCulled roots)");
    {
        int m = DistantLand::mwWorldSuppress;
        auto bit = [&](const char* label, int flag) {
            bool on = (m & flag) != 0;
            if (ImGui::Checkbox(label, &on)) {
                m = on ? (m | flag) : (m & ~flag);
                DistantLand::mwWorldSuppress = m;
                LOG::logline(">> [seam] MW world suppression mask -> %d (land=%d pick=%d obj=%d)",
                             m, (m & MGE::GeometryCache::kSuppressLand)    ? 1 : 0,
                                (m & MGE::GeometryCache::kSuppressPick)    ? 1 : 0,
                                (m & MGE::GeometryCache::kSuppressObjects) ? 1 : 0);
            }
        };
        bit("suppress landscape", MGE::GeometryCache::kSuppressLand);
        bit("suppress pick objects", MGE::GeometryCache::kSuppressPick);
        bit("suppress world objects", MGE::GeometryCache::kSuppressObjects);
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Forbids the ENGINE from traversing a world root, so it stops walking that\n"
            "subtree and issuing draws the proxy would only reject.\n"
            "Content reaching the host via captureAlphaDraw (world particles, blended fire)\n"
            "exists ONLY because MW really issues the draw - suppressing its root removes it.\n"
            "Toggle ONE at a time to attribute a missing effect to a specific root.\n"
            "Weather/VFX/projectile/spell roots are worldRoot siblings, never suppressed.");

    ImGui::Separator();
    ImGui::Text("Produce worker (NUMPAD8)");
    const char* modes[] = { "OFF (inline)", "FENCED (Tier 1b)", "OVERLAP (Tier 2)",
                            "PARK (fire-at-frame-start)" };
    int mode = g_produceMode;
    if (ImGui::Combo("produce mode", &mode, modes, 4)) {
        g_produceMode = mode;
        LOG::logline(">> [seam] produce worker: %s", modes[mode]);
    }

    ImGui::Separator();
    ImGui::Text("Seam A/B");
    logCheck("Forge composite (F11)", g_enabled, "composite");
    logCheck("Frame-ahead pipelining (numpad *)", g_frameAheadLive, "frame-ahead pipelining");
    logCheck("FP arm suppression (numpad /)", g_fpSuppressLive, "FP suppression (FP1b)");

    // Live render-scale (supersampling). The host renders into a g_rw x g_rh sub-rect of the fixed
    // g_w x g_h allocation (ceiling x backbuffer); the slider just restamps the render size — no
    // reallocation, no host re-init. 1.0 = native (identical to the pre-feature path).
    {
        float s = g_renderScale;
        if (ImGui::SliderFloat("Render scale (SSAA)", &s, kMinRenderScale, kMaxRenderScale, "%.2fx")) {
            g_renderScale = s;
            recomputeRenderSize();
            LOG::logline(">> [seam] render scale %.2fx -> render %ux%u (alloc %ux%u, bb %ux%u)",
                         g_renderScale, g_rw, g_rh, g_w, g_h, g_bbW, g_bbH);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Supersampling: host renders the world at scale x backbuffer, downfiltered on\n"
                "composite. VRAM is fixed at the ceiling (%.2fx); this only moves the per-frame\n"
                "viewport. UI stays at native res. 1.0x is byte-identical to the native path.",
                kMaxRenderScale);
        ImGui::SameLine();
        ImGui::Text("(%ux%u)", g_rw, g_rh);
    }

    // Build-shrink toggles (Stages 0-2a). Panel-only, no keys: the free numpad keys collide
    // with the water-flow handlers' edge-triggered polls when UseWaterFlowMap is on.
    ImGui::Separator();
    ImGui::Text("Build draw lists (shrink)");
    logCheck("[bspike] build-spike log", g_buildSpikeLog, "build-spike log [bspike]");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Logs one !! [bspike] line when a buildGeometryDrawLists frame runs\n"
            "> max(1.5x EMA, 1.2ms), naming which component carried it (ensure/emit/\n"
            "scan/alpha) + captures/movers/cacheSize/sweepAge. Hard-throttled >=1s\n"
            "apart, 64 lines/session — hot-path logging has collapsed the frame before.");
    logCheck("sky/FP membership validation", g_membershipValidate, "sky/FP membership validation");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Runs the old full-cache sky/FP scans alongside the g_skyKeys/g_fpKeys\n"
            "set consumers every frame. Any !! [memb-cmp] line = set-maintenance bug;\n"
            "must stay silent across city / interior / weather / moon rise / POV\n"
            "switch / cell transitions. Costs a full-map scan per frame while on.");
    {
        bool capped = g_captureBudgetOn;
        if (ImGui::Checkbox("first-sight capture budget", &capped)) {
            g_captureBudgetOn = capped;
            LOG::logline(">> [seam] first-sight capture budget %s (cap=%d/frame)",
                         g_captureBudgetOn ? "CAPPED" : "UNLIMITED", kCaptureBudgetPerFrame);
        }
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "ON (default): ensureLive defers first-sight lazy captures past %d per build,\n"
            "spreading reveal bursts over frames (deferred keys re-arrive via the next\n"
            "classify — 1-2 frame late pop-in at reveal edges is the trade).\n"
            "OFF: old behavior — a burst captures whole in one frame's build (the spike).\n"
            "Unlimited either way on cell transitions and an empty cache.",
            kCaptureBudgetPerFrame);

    ImGui::End();
}
