#include "renderprocess.h"
#include "configuration.h"
#include "ipc/client.h"
#include "ipc/geomwire.h"
#include "ipc/frametrace.h"   // MGE_FRAME_TRACE=1 timeline spans (frametrace-html.py)
#include "support/log.h"
#include "dxvk_interop.h"
#include "distantland.h"
#include "mwbridge.h"
#include "scenegraph_geometry_cache.h"
#include "cachebounds.h"
#include "scenegraph.h"
#include "datahandler_view.h"
#include "worldcontroller_view.h"
#include "exactpos.h"
#include "morrowindbsa.h"
#include "statusoverlay.h"
#include "mge_tracy.h"
#include "imgui.h"
#include "proxydx/texledger.h"   // Morrowind's own texture bytes (tasks/forge-memory-shape.md)

#include <windows.h>
#include <dxgi1_4.h>   // QueryVideoMemoryInfo: Morrowind.exe's OWN VRAM
#include <algorithm>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <deque>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace RenderProcess {
    void dropPendingCopy();   // 1.5-ahead parked copy (defined with g_pendingCopy); used by lazyInit
}

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
    // ⚠ THE CEILING IS A VRAM DECISION, NOT A QUALITY ONE, AND IT IS PAID WHETHER OR NOT SSAA RUNS.
    // Every scene target is allocated at ceiling x backbuffer, so a 2.0 ceiling is 2x LINEAR = 4x
    // AREA: at 2560x1600 that reserves 5120x3200 across 17 render targets and leaves ~75% of each
    // one allocated and never rendered into. Measured 2026-09-03: the host sits at 3660 MB with only
    // 178 textures resident — i.e. the fixed allocation, not the scene, is the footprint — and a
    // cell-churn run then crossed the DXGI budget (102% OVER), started the driver paging, and took
    // the frame from 105 to 64 fps. See [[project_forge_vram_overbudget_degradation]].
    //
    // So the ceiling now DEFAULTS TO OFF (1.0 = allocate exactly the backbuffer). SSAA is opt-in via
    // MGE_RENDER_SCALE at startup, which raises the ceiling to what was asked for. It has to be a
    // STARTUP decision because the shared RT, the imported VkImage and g_mainTex are all created at
    // the allocation size; the panel slider moves the render sub-rect within it and cannot grow it.
    constexpr float kRenderScaleHardCap = 2.0f;   // most the ceiling may ever be raised to
    constexpr float kMinRenderScale     = 1.0f;   // supersampling-only (no downscale)
    float g_maxRenderScale = 1.0f;                // live ceiling; 1.0 unless MGE_RENDER_SCALE asks
    float g_renderScale = 1.0f;               // live, panel-driven; [kMin..kMax]
    UINT  g_bbW = 640, g_bbH = 360;           // backbuffer (composite destination)
    UINT  g_rw = 640, g_rh = 360;             // current internal render size (<= g_w/g_h)

    IPC::Client* g_client = nullptr;
    bool   g_initOk  = false;
    bool   g_enabled = true;           // composite ON by default; F11 toggles it OFF/ON
    IDirect3DTexture9* g_sunDX9Texture = nullptr;  // real tex MW binds for the sun disc; set by the
                                                   // sky-list build, read by inspectIndexedPrimitive
                                                   // to suppress MW's double-drawn sun (see header)
    int    g_debugMode = 0;            // F12 diagnostic cycle: 0=normal, 1=depth (world-distance), 2=scatter, 3=AO, 4=bent normal
    unsigned g_frame = 0;

    // Dev overlay (Stage 2): F9 toggles the in-host Forge panel; mouse is polled each frame and
    // forwarded over the renderScene RPC. The host injects it into Forge UI (uiSetExternalInput).
    bool   g_devUiVisible = false;     // F9; default off so it never blocks normal play
    HWND   g_devHwnd = nullptr;        // MW focus window (cached from device creation params)
    bool   g_reloadShadersPending = false; // F8 latched at composite finish, consumed by the next kickoff
    bool   g_distLightsTogglePending = false; // numpad- latched at composite finish; one-shot host dist-light A/B
    unsigned g_gpuCapturePending = 0;  // numpad0 latched at composite finish; N host frames of RenderDoc capture
    bool g_hdrDumpPending = false;     // numpad1 latched at composite finish; one linear-scene EXR + TGA dump
    bool   g_fpSuppressLive = true;    // FP1b: MW arm suppression; on by default, numpad-/ flips live for the A/B

    // T3 (tasks/forge-terrain.md): stop emitting MW's OWN near terrain once the host is drawing the
    // real LAND heightfield for near AND far on one LOD ladder. Two producers of one surface is the
    // seam this whole plan exists to delete — and the old handover (z-sink, nearViewRange-1152, the
    // band) has nothing left to reconcile.
    //
    // Gated on the HOST's report, not on a client setting: g_hostOwnsTerrain mirrors
    // HostFrameTimings::terrainOwned, so if the host is not actually drawing terrain (still loading,
    // load failed, panel toggle off) MW keeps drawing its own. Suppressing without that check turns
    // a double-draw into a HOLE, which is strictly worse. Cleared whenever the seam drops so an
    // F11-off frame is always vanilla.
    // No client-side override and no new key: the host's own "Draw: terrain" panel checkbox IS the
    // A/B. Unchecking it drops terrainOwned to 0, which hands the near field straight back to MW on
    // the very next frame. One switch, both halves, and the keyspace stays untouched (the free
    // numpad keys collide with the water-flow handlers' edge polls — see the note at the key block).
    bool   g_hostOwnsTerrain = false;

    // Frame-ahead pipelining: on early-kickoff frames the paired
    // renderSceneFinish + RT copy defer to the NEXT frame's BeginScene(0) collect, and the
    // EndScene(0) composite point blits the PREVIOUS host frame from g_mainTex (which the
    // copy left as a stable snapshot — the blit never reads the shared RT directly). The
    // host frame thus overlaps the WHOLE MW frame, not just scene 0. World image lags
    // input by one frame (UI stays current).
    bool   g_frameAheadLive = true;    // on by default; numpad-* flips live (A/B)
    // 1.5-ahead (tasks/forge-pipeline-depth.md P4). On a park-fired frame the collect does the
    // finish RPC only — it waits host CPU, not GPU — and parks {fence, rtSlot} in g_pendingCopy;
    // the RT copy (the host-GPU wait) moves to the blit point. The next park fire therefore goes
    // out while the host GPU is still drawing the previous frame, so the host records N+1 under
    // GPU N instead of leaving the GPU idle for its whole setup+cull+record. What is displayed, and
    // when, is unchanged: frame N still reaches g_mainTex before this MW frame's blit.
    // numpad-* cycles off -> 1-ahead -> 1.5-ahead; MGE_COPY_AT_BLIT=0|1 sets it at seam init.
    bool   g_copyAtBlit = true;

    // The pipelining mode as ONE value (numpad-* and the panel's radio set it; both flags derive).
    enum PipeMode { kPipeOff = 0, kPipeAhead1 = 1, kPipeAhead15 = 2 };
    int pipeMode() { return !g_frameAheadLive ? kPipeOff : g_copyAtBlit ? kPipeAhead15 : kPipeAhead1; }
    const char* pipeModeName(int m) {
        return m == kPipeOff ? "OFF (serial)" : m == kPipeAhead1 ? "1-ahead" : "1.5-ahead (copy at blit)";
    }
    // Set, log, and SAY it on screen: a key press with no visible answer reads as a key that did
    // nothing, and this switch has no instant visual tell of its own.
    void setPipeMode(int m) {
        g_frameAheadLive = (m != kPipeOff);
        g_copyAtBlit     = (m == kPipeAhead15);
        char msg[96];
        std::snprintf(msg, sizeof(msg), "Forge pipelining: %s", pipeModeName(m));
        LOG::logline(">> [seam] %s", msg);
        StatusOverlay::setStatus(msg);
    }
    // Client produce (buildGeometryDrawLists + flush + RPC-start, D3D9-free after Tier 1a) on a
    // fresh dedicated worker. NUMPAD8 cycles 3 modes. (S5a: every mode was additionally gated on
    // !UseRenderThread — the legacy MGE render thread owned the device lock and forced a
    // MULTITHREADED device, and the two workers could not be mixed. It is gone.)
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
    // Tier 1 event handoff: CPU ms the last RT copy spent waiting the host's frame event (part of
    // the [hb] copy= bucket; echoed alone as evw= so a host GPU tail landing on MW's main thread is
    // visible). Always 0 on the semaphore and blocking paths.
    double g_lastEventWaitMs = 0.0;
    // Frame trace: the shared-fence value of the frame the last RT copy delivered — the key that
    // ties the client's spans to the host's in the timeline.
    std::int64_t g_traceCopyFence = -1;
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
    struct Accum { double feed, geom, build, render, host, overlap, copy, evwait, blit, dt, mwstart; double maxFeed, maxDt; double captured; unsigned n, earlyN, pipeN;
                   unsigned atBlitN;   // pipe frames whose RT copy ran at the blit (1.5-ahead)
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
    // ...and a short grace opened by a change in the references MW holds (a grid shift), for the same
    // reason — see openCaptureGrace.
    unsigned g_captureGraceUntil = 0;

    // NEAR-MISS diagnostic. A classify key is a shape MW DREW this frame; the engine-cull skip drops
    // MW's own display of it on category alone, and the DL slab clips the far copy at MW's reach — so
    // every key the build fails to emit is drawn by NOBODY. Counted per reason and averaged into
    // [nearmiss]; a key missed for kNearMissStreak consecutive frames is named once (the standing-
    // still hole), and a transient burst shows up as the per-frame averages.
    enum class NearMiss : std::uint8_t { Deferred, NoEntry, NoSlot, Count };
    struct NearMissTrack { unsigned lastFrame; unsigned streak; bool named; };
    std::unordered_map<std::uint32_t, NearMissTrack> g_nearMissTrack;
    std::uint64_t g_nearMissCount[(int)NearMiss::Count] = {};
    std::uint32_t g_nearMissFrameMax = 0, g_nearMissFrameN = 0;
    unsigned      g_nearMissNamed = 0;
    constexpr unsigned kNearMissStreak = 30;
    constexpr unsigned kNearMissNameCap = 200;

    // This build's misses by reason, and the first few keys by name — a transient (one grid shift,
    // one capture burst) is only attributable from the frame it happened in.
    std::uint32_t g_nearMissBuild[(int)NearMiss::Count] = {};
    char          g_nearMissSample[3][320] = {};
    unsigned      g_nearMissBuildLines = 0;
    std::uint32_t g_nearMissFresh = 0;
    constexpr unsigned kNearMissBuildLineCap = 300;

    void noteNearMiss(std::uint32_t key, NearMiss why) {
        ++g_nearMissCount[(int)why];
        ++g_nearMissBuild[(int)why];
        ++g_nearMissFrameN;
        auto& t = g_nearMissTrack[key];
        t.streak = (t.streak != 0 && t.lastFrame + 1 == g_frame) ? t.streak + 1 : 1;
        t.lastFrame = g_frame;
        if (t.streak == 1) {   // FRESH: not missed last frame — the transient class
            if (g_nearMissFresh < 3 && g_nearMissBuildLines < kNearMissBuildLineCap) {
                MGE::GeometryCache::describeKey(key, g_nearMissSample[g_nearMissFresh], sizeof(g_nearMissSample[0]));
            }
            ++g_nearMissFresh;
        }
        if (t.streak >= kNearMissStreak && !t.named && g_nearMissNamed < kNearMissNameCap) {
            t.named = true;
            ++g_nearMissNamed;
            char desc[512];
            MGE::GeometryCache::describeKey(key, desc, sizeof(desc));
            static const char* kWhy[] = { "DEFERRED", "NO-ENTRY", "NO-SLOT" };
            LOG::logline("!! [nearmiss] %s for %u frames: %s", kWhy[(int)why], t.streak, desc);
        }
    }

    // Once per build, after the visible loop: fold this build's misses into the window, report it.
    void nearMissEndBuild() {
        if (g_nearMissFrameN > g_nearMissFrameMax) g_nearMissFrameMax = g_nearMissFrameN;
        if (g_nearMissFresh > 0 && g_nearMissBuildLines < kNearMissBuildLineCap) {
            ++g_nearMissBuildLines;
            LOG::logline("!! [nearmiss] frame=%u t=%llu fresh=%u of missed=%u (deferred=%u noEntry=%u noSlot=%u) e.g. %s | %s | %s",
                         g_frame, (unsigned long long)GetTickCount64(), g_nearMissFresh, g_nearMissFrameN,
                         g_nearMissBuild[0], g_nearMissBuild[1], g_nearMissBuild[2],
                         g_nearMissSample[0], g_nearMissFresh > 1 ? g_nearMissSample[1] : "-",
                         g_nearMissFresh > 2 ? g_nearMissSample[2] : "-");
        }
        for (auto& c : g_nearMissBuild) c = 0;
        g_nearMissFrameN = 0;
        g_nearMissFresh = 0;
        static unsigned s_builds = 0;
        if (++s_builds >= 300) {
            LOG::logline(">> [nearmiss] %u builds: deferred=%.2f noEntry=%.2f noSlot=%.2f /build, worst build=%u, named=%u",
                         s_builds, g_nearMissCount[0] / (double)s_builds, g_nearMissCount[1] / (double)s_builds,
                         g_nearMissCount[2] / (double)s_builds, g_nearMissFrameMax, g_nearMissNamed);
            s_builds = 0;
            g_nearMissFrameMax = 0;
            for (auto& c : g_nearMissCount) c = 0;
            // Drop keys not missed in the window, so the table tracks only live holes.
            for (auto it = g_nearMissTrack.begin(); it != g_nearMissTrack.end();) {
                if (g_frame - it->second.lastFrame > 2) it = g_nearMissTrack.erase(it); else ++it;
            }
        }
    }

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
        // Frame-ahead: this kickoff's finish is deferred to the NEXT frame's
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
        // Tier 1: the shared frame-fence value that early finish read. Same reasoning as the
        // timings — the composite path must wait on THIS frame's value, and re-reading the shared
        // Parameters later would hand it whatever the next frame has since written.
        std::uint64_t earlyFenceValue;
        std::uint32_t earlyRtSlot;   // P3: ...and which of the host's two RTs it rendered into
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

    // Host-RPC serialisation gate. The host must not begin frame N until the client has finished
    // frame N-1 and copied its RT out. (Since P3 the host alternates between TWO shared RTs, so N
    // itself only overwrites N-2's. This gate is mode 2's and still copies N-1 first; the 1.5-ahead
    // copy-at-blit is park-only — see g_copyAtBlit — and park frames never arm it.) With the kick
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
    std::optional<IPC::VecView<IPC::GeomChunk>> g_fpMultiMapVec;    // persistent per-frame FP multi-map draw-list vec (FP1e)
    std::vector<std::uint8_t>                 g_pendingBlob;        // packed parts awaiting flush
    std::uint32_t                             g_pendingParts = 0;
    // [geomflush] batch shape (setWindowCapture): parts/bytes appended while the post-load window
    // walk ran, since the last flush. Measurement only.
    bool                                      g_windowCapture = false;
    std::uint32_t                             g_windowParts   = 0;
    std::uint64_t                             g_windowBytes   = 0;
    inline void noteWindowPart(std::size_t bytes) {
        if (g_windowCapture) { ++g_windowParts; g_windowBytes += bytes; }
    }
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
        // PBR param map (tasks/forge-pbr-materials.md): the slot of the BASE texture's
        // <base>_paramh.dds, 0 = the base has none. Keyed on the BASE name's pointer, because the
        // param map is a property of the base texture — the same identity rule as baseSlot, with
        // its own epoch for the same aliasing reason as ovEpoch.
        const char*   paramNamePtr = nullptr;
        std::uint32_t paramSlot = 0;
        std::uint32_t paramEpoch = 0;
        // Exterior cell the part was last emitted in (stampSlotCell). evictStaleTextures lets a
        // part pin its textures only while that cell is in Morrowind's ACTIVE grid.
        std::int32_t  cellX = 0, cellY = 0;
        bool          cellKnown = false;
    };
    std::unordered_map<std::uint32_t, SlotInfo> g_keySlot;
    std::uint32_t g_texEpoch = 0;            // bumped on bindless-slot LRU recycle
    // Last-shipped identity per cache key. Dedup is on (modelId, vc, rev), NOT rev alone:
    // the key is a recycled NiTriShape*, so a new object can inherit a freed key+slot; the
    // GeometryData ptr (modelId) + vertexCount disambiguate it (see captureGeometry).
    struct UploadSig { std::uint32_t id; std::uint32_t vc; std::uint16_t rev; };
    std::unordered_map<std::uint32_t, UploadSig> g_uploadedRev;    // cache key -> last sent identity
    std::uint32_t                             g_nextSlot = 0;
    // GEOMETRY DEDUP (tasks/forge-geometry-dedup.md). Vertices ship model-space, so every placed copy
    // of a mesh is byte-identical; the host keeps one arena range (a BLOCK) per unique content and
    // further instances send a header-only kGeomFlagAlias record. Keyed on {hash, vc, ic} so a 64-bit
    // collision must also match both sizes. `refs` counts the slots holding the block (owner + aliases)
    // and is dropped when a slot is QUEUED for release (drainReleasedSlots) or re-uploads — always no
    // later than the host drops it, so an alias is only ever sent for a block the host still holds.
    struct ContentKey {
        std::uint64_t hash;
        std::uint32_t vc, ic;
        bool operator==(const ContentKey& o) const { return hash == o.hash && vc == o.vc && ic == o.ic; }
    };
    struct ContentKeyHash {
        std::size_t operator()(const ContentKey& k) const {
            return (std::size_t)(k.hash ^ (k.hash >> 32) ^ ((std::uint64_t)k.vc * 0x9E3779B1u) ^ k.ic);
        }
    };
    struct BlockRef { std::uint32_t block; std::uint32_t refs; };
    std::unordered_map<ContentKey, BlockRef, ContentKeyHash> g_blockByContent;
    std::unordered_map<std::uint32_t, ContentKey>            g_slotBlock;     // host slot -> the block it holds a ref on
    std::uint32_t                                            g_nextBlock = 0; // never reused (0 = none)

    // MGE_GEOM_ALIAS=0 is the kill switch (the A/B oracle arm). Default ON.
    bool geomAliasOn() {
        static const bool on = [] {
            const char* e = std::getenv("MGE_GEOM_ALIAS");
            return !(e && e[0] == '0');
        }();
        return on;
    }

    // Per load window (logged at the load that ends it): what dedup did, and what the hash cost.
    struct AliasStats {
        std::uint32_t blocks = 0;        // first sights shipped as a new block
        std::uint32_t aliases = 0;       // instances shipped as a header-only alias
        std::uint64_t aliasedBytes = 0;  // vertex+index bytes those aliases did not ship
        double        hashMs = 0.0;      // content hashing, all first uploads
    };
    AliasStats g_aliasStats;

    void aliasWindowEnd(unsigned frame) {
        if (!geomAliasOn()) {
            return;
        }
        LOG::logline(">> [geom-alias] window ending at frame %u: %u new blocks, %u aliases (%.1f MB not shipped), hash %.2f ms, live blocks %zu / slots holding %zu",
                     frame, g_aliasStats.blocks, g_aliasStats.aliases, g_aliasStats.aliasedBytes / 1048576.0,
                     g_aliasStats.hashMs, g_blockByContent.size(), g_slotBlock.size());
        g_aliasStats = AliasStats{};
    }

    // Drop `slot`'s block ref, if it holds one; the block entry dies with its last ref.
    void dropBlockRef(std::uint32_t slot) {
        auto sb = g_slotBlock.find(slot);
        if (sb == g_slotBlock.end()) {
            return;
        }
        auto it = g_blockByContent.find(sb->second);
        if (it != g_blockByContent.end() && --it->second.refs == 0) {
            g_blockByContent.erase(it);
        }
        g_slotBlock.erase(sb);
    }
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
    std::vector<std::uint8_t>                 g_fpMultiMapScratch;  // packed FP MultiMapDrawWire[] this frame (FP1e)
    std::vector<std::uint8_t>                 g_multiMapScratch;    // packed MultiMapDrawWire[] this frame (Tier 4)
    std::vector<std::uint8_t>                 g_lightScratch;       // packed PointLightWire[] this frame
    std::vector<IPC::LightGoboWire>           g_lightGoboScratch;   // G3: the parallel gobo side array

    // PARK-LAG CORRECTION for the player's own geometry.
    //
    // Park mode builds frame N's payload on the worker and fires it at the start of N+1 with a
    // restamped camera: v_view = (v_rel + bakeEye - eyeNow)·R_now. For anything world-anchored
    // that resolves to (world - eyeNow)·R_now — exactly right, and the whole point of the restamp
    // (no camera latency). For the PLAYER it is exactly wrong: the body ships as (pcBake -
    // bakeEye), so it renders at (pcBake - eyeNow) when it wants (pcNow - eyeNow). The error is
    // one frame of the player's own motion, and because the camera tracks the player it is fully
    // correlated with the camera — which is why it reads as the body sliding off centre and
    // catching up when you stop, while nothing else in the world looks late.
    //
    // Same disease and same cure as the camera-anchored sky (see the pre-cancel in
    // flushAssignAndKick): record where the player's transforms landed in the scratch, then add
    // the player's own bake→fire delta back before the bytes are assigned. Deltas come from the
    // player NODE, not the eye — in 3rd person the camera also ORBITS a standing player, and an
    // eye delta would shove the body sideways on a pure mouse-look.
    struct PlayerPatch {
        std::uint8_t  scratch;   // 0 = static, 1 = skinned palette, 2 = multimap
        std::uint32_t at;        // byte offset of the FIRST translation
        std::uint32_t count;     // translations at 64-byte stride (bone count; 1 for rigid)
    };
    std::vector<PlayerPatch>                  g_playerPatch;
    double                                    g_playerBake[3] = {};
    bool                                      g_playerBakeValid = false;
    std::uint64_t                             g_buildCacheFrame = 0;   // cache frame the build read
    unsigned                                  g_seamSkipRun = 0;       // consecutive no-payload seam skips

    // markSubtreePlayer stamps the body subtree every 3rd-person frame, so a stale stamp from an
    // earlier frame never counts — which is what keeps a piece of equipment you just dropped from
    // being dragged along by the correction.
    inline bool isPlayerOwned(const MGE::GeometryCache::CachedGeometry& e) {
        return g_playerBakeValid && e.playerFrame == g_buildCacheFrame;
    }

    // CAMERA-RELATIVE translation of a rigid entry (world rows 12..14): float(T - eye), with T the
    // entry's exact double translation under ExactPos kRigid, else its stored float one — mode 0
    // is float(stored - eyePos), bit for bit the old `world[12..14] -= eyePos`. Also feeds the
    // [exactpos] shipped-vs-exact diagnostic. See exactpos.h.
    inline void shipEntryT(float out[3], const MGE::GeometryCache::CachedGeometry& e) {
        MGE::ExactPos::rel(out, &e.worldTransformD3D[12], e.worldT,
                           MGE::ExactPos::on(MGE::ExactPos::kRigid));
        MGE::ExactPos::noteShipped(MGE::ExactPos::kShipRigid, e.isFP, out, e.worldT);
    }


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
    // MW's loading bar was up since the last cell-epoch check ⇒ the scene graph was rebuilt under
    // the cache. Set by noteLoadingBar from the per-frame path, consumed by checkCellEpochAndPurge.
    // STICKY on purpose: the two run on different cadences (every frame vs produced frames only).
    bool                                        g_sawLoadingBar = false;

    // --- Produce mode 3 "PARK-AND-FIRE" ---------------------------------------------------
    // The worker builds frame N's payload into the client-private scratch vectors and PARKS it
    // (no flush/assign/RPC); at the START of frame N+1 (frameSetupEarly, camera fresh, window
    // closed) the main thread fires it with a restamped camera (fireParked). The payload parks
    // IN PLACE: the scratch vectors are cleared only at the start of the next build, and the
    // fire (main, frame start) always precedes the next worker kick in the same frameSetupEarly
    // — zero copy. Only counts + the FP bundle + the build-time eye need holding here.
    // Invalidated (a) at every kickoffBody entry (a serial/inline kickoff supersedes it),
    // (b) at fire on a cell-epoch mismatch (teleport between build and fire → drop, one
    // repeated composite frame), (c) at fire on a POV flip (see bake3rd), (d) on fire (consumed).
    // Statics near/far handover: MW's ACTIVE exterior cell set plus how far MW's own cull reaches,
    // so the host can clip its distant-statics LOD proxies at the same plane the NEAR path stops at
    // (both drawing = the handover z-fight; neither = a hole). MW culls per NiTriShape against its
    // view distance, so that plane is what `reach` means — the host slices the proxies there rather
    // than dropping whole objects. Read straight off DataHandler::exteriorCellData[9] — the engine's
    // own residency table, no detour. Only a cell confirmed LOADED counts; everything else
    // (background-loading, unloading, world edge, no DataHandler) leaves its bit clear so the host
    // keeps drawing DL there. The CENTRE bit is the valid flag: without the player's own cell there
    // is nothing to trust, and the host falls back to its fixed near-cut distance.
    //
    // ⚠ SNAPPED WHEN THE DRAW LIST IS BUILT, not when it is fired. The mask and the held-reference
    // version describe which references the near path covers, and in park mode the payload is fired
    // a frame after it was built: reading them at fire time handed the host the NEXT frame's cells
    // with THIS frame's draw list — on a grid shift, DL cut for a whole cell row the fired near set
    // did not carry yet.
    struct NearCellsSnap {
        std::int32_t  x = 0, y = 0;
        std::uint32_t mask = 0;
        float         reach = 0.0f;
        std::uint32_t refsVersion = 0;
        float         fwd[3] = { 0.0f, 1.0f, 0.0f };   // the classify camera's forward (view-Z axis)
    };
    NearCellsSnap snapNearCells() {
        NearCellsSnap n;
        n.refsVersion = DistantLand::nearRefsVersion;
        n.fwd[0] = DistantLand::eyeVec.x; n.fwd[1] = DistantLand::eyeVec.y; n.fwd[2] = DistantLand::eyeVec.z;
        if (MWBridge::get()->IsExterior()) {
            void* dh = MGE::SceneGraph::getDataHandler();
            if (dh) {
                n.x = MGE::DataHandlerView::centralGridX(dh);
                n.y = MGE::DataHandlerView::centralGridY(dh);
                for (std::size_t i = 0; i < MGE::DataHandlerView::EXT_CELL_DATA_COUNT; ++i) {
                    void* ecd = MGE::DataHandlerView::exteriorCellData(dh, i);
                    if (!MGE::DataHandlerView::exteriorCellLoaded(ecd)) { continue; }
                    void* cell = MGE::DataHandlerView::exteriorCellRecord(ecd);
                    // Grid coords come from the cell record, not the slot index — the CellGrid
                    // slot order never has to be assumed correct.
                    const int dx = MGE::DataHandlerView::cellExteriorGridX(cell) - n.x;
                    const int dy = MGE::DataHandlerView::cellExteriorGridY(cell) - n.y;
                    if (dx < -1 || dx > 1 || dy < -1 || dy > 1) { continue; }
                    n.mask |= 1u << ((dy + 1) * 3 + (dx + 1));
                }
                // MW's cull reach = its view distance (the engine culls subtrees on view-z against
                // it — the same bound distantland.cpp's cache gate uses). nearViewRange is that
                // value, re-read every frame in adjustFog.
                n.reach = DistantLand::nearViewRange;
            }
        }
        if (n.reach <= 0.0f) { n.mask = 0; }   // no reach ⇒ nothing to hand over
        return n;
    }

    struct ParkedPayload {
        bool valid = false;
        NearCellsSnap nearCells;              // at BUILD (see snapNearCells)
        std::uint32_t epoch = 0;              // g_cellEpoch at build → fire-time invalidation
        // POV at build. The camera restamp re-aims the payload with the CURRENT camera, and a
        // 3rd→1st switch moves the camera INSIDE the head — so frame N's body geometry, which was
        // perfectly correct behind a 3rd-person camera, gets re-aimed from inside itself and you
        // see the inside of your own head for exactly one frame. No transform patch can fix that:
        // the body must not be DRAWN, and whether to draw it was decided a frame before MW
        // changed its mind. So treat a POV flip the way a teleport is treated — the payload
        // describes a world that no longer exists. Dropping repeats one composite frame, which
        // during a POV cut is invisible; drawing it is not.
        bool bake3rd = true;
        double bakeEye[3] = {};               // ExactPos::eye() at build (payload's relative space)
        std::uint32_t drawCount = 0, skinnedCount = 0, multiMapCount = 0,
                      lightCount = 0, skyCount = 0, alphaCount = 0;
        IPC::FPFrame fpFrame;                 // shipped VERBATIM at fire — self-consistent
        std::uint32_t fpDraws = 0, fpSkinnedDraws = 0, fpAlphaDraws = 0,
                      fpMMDraws = 0;                    // (pose N, arm-cam N) bundle
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
        // FFE stages beyond the base map (dark/detail/glow folded into the same DIP) — see
        // IPC::AlphaDrawWire::stages. packMMStage-encoded, uvSet always 0.
        std::uint32_t stageCount;
        std::uint32_t stages[3];
    };
    // CONSUME/SHIP side (produce WORKER only): buildGeometryDrawLists reads g_capRecs, appends FP
    // particles to g_capVertScratch/g_capIdxScratch, and the kickoff ships them. Filled by the swap
    // below from the main-thread INCOMING buffers — never written by captureAlphaDraw directly.
    std::vector<IPC::GeomVertexWire> g_capVertScratch;   // consumed/shipped verts (frame N-1)
    std::vector<std::uint16_t>       g_capIdxScratch;    // consumed/shipped indices (frame N-1)
    std::vector<CapturedAlphaRec>    g_capRecs;          // consumed records (frame N-1)
    std::optional<IPC::VecView<IPC::GeomChunk>> g_capturedVec;   // shipped at kickoff ([verts][indices])
    // Old-msoc double-draw guard: (bindless slot << 32 | vertCount) of every cached blended shape
    // the host already draws; a captured DIP that matches one is a duplicate and is skipped.
    //
    // The key used to be the GPU TEXTURE POINTER, which silently stopped working for flip books.
    // A NiFlipController at secondsPerFrame=0.0067 (150 fps) rebinds a different
    // IDirect3DTexture9* nearly every displayed frame, so the pointer the worker recorded never
    // matched the one MAIN saw — Enhanced Light's magelight quad was drawn TWICE every frame
    // (Route C base x dark, plus an AT3 base-only copy blended on top = the "too bright" orb).
    // Keying on the RESOLVED BINDLESS SLOT fixes it: since the flip-array work every frame of a
    // book encodes as bit15|bucket<<11|layer, so masking the LAYER off yields one value for the
    // whole book, and both sides already resolve that slot for their own draw.
    // Slot 0 (unnamed / host default white) is NEVER inserted or matched — a pointer key kept
    // unnamed draws distinct, a slot key would alias every unnamed draw of the same vertCount.
    inline std::uint64_t alphaDedupKey(std::uint32_t texSlot, std::uint32_t vertCount) {
        const std::uint32_t book = IPC::isFlipSlot(texSlot)
                                 ? (texSlot & ~IPC::kFlipLayerMask) : texSlot;
        return ((std::uint64_t)book << 32) | (std::uint64_t)vertCount;
    }
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
    // AT3 extra-stage drops: a dark/detail/glow stage sampling UV set 1+ that the single-UV
    // captured VB cannot carry (see captureAlphaDraw). Non-zero means the documented limit is
    // being hit by real content and the captured path needs a wide-vertex variant.
    std::uint32_t g_capStageUVDrop = 0;
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
    // g_texPendingEntries/Bytes/Count) is mutated from TWO threads: the produce worker's build
    // (buildGeometryDrawLists -> resolveTextureSlot) AND the MAIN thread's captured-alpha proxy
    // interception (captureAlphaDraw -> resolveTextureSlot). produce-off-main OVERLAPS them by
    // design, so a concurrent emplace/rehash on g_texSlot corrupted the heap (crash walking a torn
    // std::string key in _Forced_rehash, 2026-07-20). This mutex serialises the whole subsystem.
    // NEVER held across a blocking IPC RPC — flushTextures swaps the blob out under the lock and
    // RPCs on the local copy (holding it across texUploadBlocking would be a new 60s-freeze class).
    std::mutex                                g_texResidencyMx;
    std::unordered_map<std::string, std::uint32_t> g_texSlot;       // normalized name -> bindless slot
    std::uint32_t                             g_nextTexSlot = 1;    // 0 = host default white
    // Staged texture uploads awaiting flush: ONE ENTRY PER ELEMENT ([TexUploadWire][dds]), never a
    // single contiguous blob. This was a `std::vector<std::uint8_t>` that every staged entry was
    // appended to, and in a 32-bit process that is a loaded gun. kickoffBody resolves the WHOLE
    // frame's first-seen texture set before flushTextures runs, so a cell load stages everything at
    // once: measured 568 MB on one load (273 MB of base textures + 213 MB of 4096 _paramh + 82 MB of
    // 4096 _paramd), and a geometric regrow at that size needs old+new live — ~1.4 GB of CONTIGUOUS
    // address space. Morrowind is 4GB-patched but heavily fragmented, so the regrow threw
    // std::bad_alloc on the produce worker, where nothing catches it: uncaught -> fail-fast
    // c0000409, the "crash at load" of 2026-09-17. It died appending a 1.4 MB file; its own size was
    // never the point. Per-entry elements cap the largest single allocation at one window
    // (kTexWindowBytes, 32 MB) and, as a bonus, make the requeue below a splice instead of an
    // insert-at-front that memmoved half a gigabyte.
    std::deque<std::vector<std::uint8_t>>     g_texPendingEntries;  // one [TexUploadWire][dds] each
    std::size_t                               g_texPendingBytes = 0;// sum of the above (stats + budget)
    std::uint32_t                             g_texPendingCount = 0;
    // NOTE: no staging budget. Chunking (above) is what makes the peak survivable; rationing it by
    // refusing to resolve textures is not, because "deferred" and "white" are the same return value
    // and the callers memoise. The peak itself is capped at the SOURCE — see the mip-slice plan in
    // tasks/forge-pbr-materials.md.
    // LRU eviction over the client's bindless range [1, kMaxTextures-kDlReserve). Residency is
    // cumulative all session (no per-cell reset), so without eviction a long traversal exhausts the
    // slots and every NEW near texture goes white ("near white far from spawn"). Recycle the
    // least-recently-used slot instead. g_frame is the LRU clock.
    std::vector<std::string>                  g_slotName;           // slot -> name (for eviction; size kMaxTextures)
    std::vector<std::uint32_t>                g_slotLastUsed;       // slot -> last g_frame it was referenced
    // H2 occupancy (external audit PR #2 items 1/2). The audit read this pool as "no eviction,
    // hard-bounded per session" — stale: the LRU above recycles. What was genuinely silent is the
    // PRESSURE. Recycling is not free (each one bumps g_texEpoch, forcing a re-resolve of every
    // cached slot), and thrash — recycling a slot that was used THIS frame — means the working set
    // does not fit and some texture is drawing white. That warning used to be a `static bool`:
    // one line per session, then unbounded silence. Now counted and surfaced in the heartbeat.
    std::uint64_t                             g_texRecycles = 0;    // LRU evictions, session total
    std::uint64_t                             g_texThrashes = 0;    // recycled a slot used this frame

    // STALE-TEXTURE EVICTION (tasks/forge-memory-shape.md). The LRU above only ever recycles once the
    // range is FULL, and the host frees a texture only when its slot is re-uploaded, so residency
    // ratcheted to 872 slots (~3.2 GB of mostly-unsampled textures) and never came back down.
    // evictStaleTextures releases a slot when BOTH hold:
    //   - no live g_keySlot part INSIDE MORROWIND'S ACTIVE GRID names it (base / overlay / param).
    //     The active 3x3 is the gone-signal: a texture only cells behind the grid use stops being
    //     pinned the moment the grid moves on (and indoors every live part pins). Every cross-frame
    //     copy of a slot lives in a SlotInfo in g_keySlot, so after a release the SlotInfos still
    //     naming it (outer-ring parts) are cleared in the same pass and re-resolve by name — no epoch
    //     bump, no re-resolve storm. The host's persistent shadow-caster records carry the base slot
    //     of a live key; an outer-ring key is a cell or more away, beyond the sun cascades.
    //   - it has not been referenced for g_texEvictAgeFrames. Every per-frame consumer that resolves
    //     by name (multimap stages, captured alpha, sky, FP, glow) stamps g_slotLastUsed, so the age
    //     covers them — and doubles as the hysteresis that stops a door hop re-uploading a cell.
    // Released slots go on g_texFreeSlots and are reused before the range grows; the host is told
    // with a zero-length TexUploadWire flagged kTexUploadRelease, and retires the texture.
    std::vector<std::uint32_t>                g_texFreeSlots;
    std::vector<std::uint32_t>                g_slotBytes;          // slot -> DDS bytes shipped (eviction log)
    std::uint32_t                             g_texEvictAgeFrames = 600;   // MGE_TEX_EVICT_FRAMES; 0 = off
    std::uint64_t                             g_texEvictions = 0;   // session totals
    std::uint64_t                             g_texEvictedBytes = 0;

    // ---- Texture I/O thread (tasks/forge-crossing-frame.md P1) -----------------------------------
    // The frame after a crossing spent 13-17 ms of the BUILD (which main's waitProduce blocks on in
    // park mode) reading texture files: the stream lane's full files (~4.5 ms read + ~5 ms copying
    // each into a fresh vector) and the prefetch drain's first sights (a 2 ms read cap, 4.5-6 ms
    // measured). None of those bytes is needed THIS frame. So one thread reads them ahead and the
    // build only takes reads that have FINISHED: stream entries gather straight from the read buffer
    // into the stream window, prefetch resolves hand their bytes to resolveTextureSlotEx.
    //
    // The thread touches no residency state and takes no residency lock; BSA reads are positional
    // (BSA::readAt), so they need none. A request is a refcounted TexIoRead: whoever drops the last
    // ref (a consumer, or the thread finding the request abandoned) frees the bytes. Bytes read but
    // not yet consumed are capped (kTexIoMaxHeldBytes): this is a 32-bit process, and an unbounded
    // read-ahead of 21 MB 4K maps would be the content-sized buffer again
    // ([[feedback_content_sized_staging_buffer]]). MGE_TEX_IO_THREAD=0 = read inline in the build.
    struct FirstSightBytes {
        void*    data = nullptr;   // malloc'd: the full file, or the LOD placeholder when `deferred`
        unsigned size = 0;
        bool     deferred = false; // the full file streams later (data = placeholder, or null)
        bool     found = false;    // false = a miss (the slot goes white)
    };
    // resolveTextureSlotEx's first-sight read, factored out so it can run on any thread: the
    // placeholder decision (deferAllowed) is made by the caller under the residency lock and carried.
    FirstSightBytes readFirstSight(const std::string& name, bool dataTexture, bool deferAllowed) {
        FirstSightBytes r;
        if (deferAllowed) {
            // The LOD library carries the PBR companions too since 2026-09-20, so a _paramh can have
            // a stand-in like any base map. Bakes made before that have none, and a param map then
            // defers with nothing to show.
            r.deferred = BSA::loadDistantLodBytes(name.c_str(), &r.data, &r.size) && r.data && r.size > 0;
            if (!r.deferred && r.data) { std::free(r.data); r.data = nullptr; r.size = 0; }
            if (!r.deferred && dataTexture) {
                r.deferred = BSA::fileExists(name.c_str(), true);
            }
        }
        if (r.deferred) {
            r.found = true;
            return r;
        }
        // skipDistantStatics=true: the Forge near path must NOT pick the distantland\statics
        // downscaled-LOD copies (they blur near geometry) — resolve loose Data Files -> BSA.
        if (BSA::loadFileBytes(name.c_str(), &r.data, &r.size, true) && r.data && r.size > 0) {
            r.found = true;
        } else {
            if (r.data) { std::free(r.data); }
            r.data = nullptr;
            r.size = 0;
        }
        return r;
    }

    constexpr std::uint64_t kTexIoMaxHeldBytes = 48ull << 20;
    struct TexIoRead {
        enum Kind : std::uint8_t { kFull, kFirstSight };
        std::string       name;
        Kind              kind = kFull;
        bool              dataTexture = false;   // kFirstSight only
        bool              deferAllowed = false;  // kFirstSight only: the decision the read was made under
        FirstSightBytes   r;                     // kFull: data/size/found only
        std::atomic<bool> done{ false };
        ~TexIoRead();
    };
    using TexIoHandle = std::shared_ptr<TexIoRead>;

    class TexIo {
    public:
        bool on() const { return m_thread.joinable(); }
        void start() {
            if (on()) { return; }
            m_stop = false;
            m_thread = std::thread([this] { run(); });
        }
        void stop() {
            {
                std::lock_guard<std::mutex> lk(m_mx);
                m_stop = true;
                // Complete what never ran as a failed read, so no consumer still holding one waits on it
                // forever (the stream stage stops at an unfinished head).
                for (auto& q : m_queue) { q->done.store(true, std::memory_order_release); }
                m_queue.clear();
            }
            m_cv.notify_all();
            if (m_thread.joinable()) { m_thread.join(); }
        }
        ~TexIo() {
            // Normal teardown stops the thread in RenderProcess::shutdown. If that never ran, a joinable
            // std::thread here would std::terminate, and joining under the loader lock can deadlock.
            if (m_thread.joinable()) { m_thread.detach(); }
        }
        // Queue a read. With the thread off, read it now (MGE_TEX_IO_THREAD=0: today's inline reads).
        TexIoHandle submit(std::string name, TexIoRead::Kind kind, bool dataTexture = false, bool deferAllowed = false) {
            auto h = std::make_shared<TexIoRead>();
            h->name = std::move(name);
            h->kind = kind;
            h->dataTexture = dataTexture;
            h->deferAllowed = deferAllowed;
            if (!on()) {
                read(*h);
                return h;
            }
            {
                std::lock_guard<std::mutex> lk(m_mx);
                m_queue.push_back(h);
            }
            m_cv.notify_one();
            return h;
        }
        void release(std::uint64_t bytes) {
            m_held.fetch_sub(bytes);
            m_cv.notify_one();
        }
        // Session totals for the [tex-stream] line.
        std::atomic<std::uint64_t> reads{ 0 }, readBytes{ 0 }, abandoned{ 0 }, heldPeak{ 0 };
        std::atomic<double>        readMs{ 0.0 };

    private:
        void read(TexIoRead& q) {
            MGE_ZoneScopedN("tex:io read");
            const auto t0 = std::chrono::steady_clock::now();
            if (q.kind == TexIoRead::kFull) {
                q.r.found = BSA::loadFileBytes(q.name.c_str(), &q.r.data, &q.r.size, true) && q.r.data && q.r.size > 0;
                if (!q.r.found && q.r.data) { std::free(q.r.data); q.r.data = nullptr; q.r.size = 0; }
            } else {
                q.r = readFirstSight(q.name, q.dataTexture, q.deferAllowed);
            }
            if (q.r.data) {
                const std::uint64_t held = m_held.fetch_add(q.r.size) + q.r.size;
                std::uint64_t peak = heldPeak.load();
                while (held > peak && !heldPeak.compare_exchange_weak(peak, held)) {}
            }
            reads.fetch_add(1);
            readBytes.fetch_add(q.r.size);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            double cur = readMs.load();
            while (!readMs.compare_exchange_weak(cur, cur + ms)) {}
            q.done.store(true, std::memory_order_release);
        }
        void run() {
            for (;;) {
                TexIoHandle q;
                {
                    std::unique_lock<std::mutex> lk(m_mx);
                    // Back-pressure: hold off the next read while the unconsumed bytes are over the cap.
                    m_cv.wait(lk, [this] { return m_stop || (!m_queue.empty() && m_held.load() < kTexIoMaxHeldBytes); });
                    if (m_stop) { return; }
                    q = std::move(m_queue.front());
                    m_queue.pop_front();
                }
                if (q.use_count() == 1) {
                    abandoned.fetch_add(1);   // its job was dropped while it waited: nobody wants the bytes
                    q->done.store(true, std::memory_order_release);
                    continue;
                }
                read(*q);
            }
        }
        std::thread                 m_thread;
        std::mutex                  m_mx;
        std::condition_variable     m_cv;
        std::deque<TexIoHandle>     m_queue;
        std::atomic<std::uint64_t>  m_held{ 0 };
        bool                        m_stop = false;
    };
    TexIo g_texIo;   // defined BEFORE every queue holding a TexIoHandle: destroyed after them
    TexIoRead::~TexIoRead() {
        if (r.data) {
            std::free(r.data);
            g_texIo.release(r.size);
        }
    }
    // Take a finished read's bytes out of its handle (the caller now owns and frees them).
    inline FirstSightBytes takeBytes(TexIoRead& q) {
        FirstSightBytes r = q.r;
        if (q.r.data) { g_texIo.release(q.r.size); }
        q.r.data = nullptr;
        q.r.size = 0;
        return r;
    }

    // FIRST-SIGHT STREAMING (user, 2026-09-20: "same texture is already in DL and it can just be
    // reused until real texture is loaded, without a stall or hitch"). A cell load or a door used to
    // resolve 100-370 MB of textures in ONE frame and ship them in one blocking flush (texflush 50-105
    // ms). Now a first-sight texture that the distant-land bake made a LOD copy of gets that copy
    // (distantland\statics\textures, KB-sized) as its slot's first upload — drawable this frame — and
    // the full file joins g_texStreamQueue; streamPendingTextures ships at most g_texStreamBudgetMB
    // of full files per frame into the SAME slots (the host replaces a slot's texture in place, as it
    // always has for re-uploads). _paramh companions have no LOD copy: they are queued with the slot
    // reset to white, which the shader already reads as "no param map", so PBR pops in instead of the
    // frame stalling. Everything else (no LOD copy, not data) stays synchronous — a white card would
    // be worse than the stall. MGE_TEX_STREAM_MB overrides; 0 = off (everything synchronous).
    struct TexStreamJob {
        std::uint32_t slot;
        std::string   name;
        bool          data;   // a _paramh: upload with kTexUploadData
        TexIoHandle   io;     // its full-file read, once the stream lane has asked for it (P1)
    };
    std::deque<TexStreamJob>                  g_texStreamQueue;
    // A streamed texture's full file goes into a FRESH slot, not over its placeholder: the host is
    // 2-deep, so frames that draw with the placeholder are usually still executing when the full file
    // lands, and rewriting their slot's descriptor under them is a race the host could only avoid by
    // waiting (tasks/forge-pipeline-depth.md P4 refinement). The name moves to the new slot at once;
    // the placeholder's slot retires here and returns to the pool IPC::kTexColdFrames later, when no
    // draw list still in flight can name it.
    struct TexSlotRetire {
        std::uint32_t slot;
        std::uint32_t frame;   // g_frame when the name moved off it
    };
    std::deque<TexSlotRetire>                 g_texRetireSlots;
    std::uint64_t                             g_texFreshStreams = 0;   // session totals
    std::uint64_t                             g_texInPlaceStreams = 0; // no fresh slot free: old path
    std::uint32_t                             g_texStreamBudgetMB = 16;
    // The g_frame value of a load's first produced build: first sights then load their FULL file, no
    // placeholder (see texBookkeepingRelease's neighbour comment, "FIRST FRAME AFTER A LOAD").
    std::uint32_t                             g_texSyncFrame = 0xFFFFFFFFu;
    std::uint64_t                             g_texPlaceholders = 0;  // session totals
    std::uint64_t                             g_texDeferredData = 0;
    std::uint64_t                             g_texStreamed = 0;
    std::uint64_t                             g_texStreamedBytes = 0;

    // ASYNC STREAM LANE (tasks/forge-async-texture-stream.md). The full files above used to ride the
    // blocking sync flush: 5-7 ms of host ingest per 4K replacer inside the client's wait, and the
    // host's next frame queued behind it (crossings: 45-69 ms frames). Now they go on the host's third
    // IPC channel, where a host worker ingests them, and the client only POLLS for the receipt.
    //   - The name moves on CONFIRMATION, not at stage time: the full file goes into a fresh COLD slot
    //     that is RESERVED meanwhile (empty name, out of the free list, age pinned), the placeholder
    //     keeps drawing, and the move happens in the first build after the host reports the slot bound.
    //   - One batch in flight (<= IPC::kMaxStreamBatch entries, its own 32 MB window g_texStreamVec).
    //   - A fresh slot must have NO unshipped sync traffic: a release staged for a free-list slot but
    //     not yet flushed would land on the host AFTER the stream installed into it and blank the full
    //     texture. g_slotStagedBatch[s] = the sync flush that carries s's last staged record;
    //     g_texShippedSerial = the last flush fully consumed. Usable iff staged <= shipped.
    //   - In-place streams (no cold slot to spare) stay on the SYNC path: an in-place upload landing
    //     out of band could overwrite a slot the sync lane has since given to another texture.
    // MGE_TEX_STREAM_ASYNC=0 = the previous path (full files staged into the sync flush).
    struct TexStreamFlight {
        std::uint32_t oldSlot;   // the placeholder's slot (the name stays on it until confirmed)
        std::uint32_t newSlot;   // the reserved fresh slot the full file goes into
        std::string   name;
        std::uint32_t size;
        bool          data;
    };
    bool                                      g_texStreamAsync = true;   // MGE_TEX_STREAM_ASYNC
    std::optional<IPC::VecView<IPC::GeomChunk>> g_texStreamVec;         // the stream lane's window
    std::vector<TexStreamFlight>              g_texStreamFlight;          // under g_texResidencyMx
    std::uint32_t                             g_texSwapSerial = 0;        // sync flushes started
    std::uint32_t                             g_texShippedSerial = 0;     // ...and fully shipped
    std::vector<std::uint32_t>                g_slotStagedBatch;          // slot -> carrying flush
    std::uint64_t                             g_texStreamBatches = 0;     // session totals
    std::uint64_t                             g_texStreamConfirmed = 0;
    std::uint64_t                             g_texStreamFailed = 0;
    std::uint64_t                             g_texStreamStale = 0;

    // GRID PREFETCH (tasks/forge-pipeline-depth.md "Fast-turn fps drop"). Textures reach the host on
    // FIRST SIGHT, so the first look behind you after a load resolved ~400 of them in 4 frames
    // (50-80 ms each on Dragonstar East): ~290 with no DL LOD copy shipped full-size, synchronously.
    // But the post-load residency walk has already captured the whole active grid into the geometry
    // cache — the names are known long before the turn. prefetchGridTextures resolves them in the
    // background, a few MB per frame, through the same resolveTextureSlotEx the draws use (so
    // placeholders, streaming and cold slots all behave exactly as on first sight). The first turn
    // then finds every slot already resident.
    //
    // The set it built also PINS: evictStaleTextures only held textures named by an emitted part in
    // the grid, and a prefetched texture behind the camera has none — it would be released 600
    // frames later and the first look would hitch again. g_gridTexPins holds every normalized name
    // (base, overlay, multimap, param companions) of the active grid's cached entries; it is rebuilt
    // when the grid moves. Guarded by g_texResidencyMx (eviction reads it on the fire thread).
    struct GridTexJob {
        std::string name;   // normalized; for a param job, the base STEM
        bool        param;  // try <stem>_paramh.dds, then <stem>_paramh_np.dds (resolveParamSlot's order)
        TexIoHandle io;     // its first-sight read (P1); for a param job, the _paramh one
        TexIoHandle io2;    // param job only: the _paramh_np read
    };
    std::deque<GridTexJob>                    g_gridTexQueue;       // produce-owned
    std::unordered_set<std::string>           g_gridTexPins;        // under g_texResidencyMx
    std::uint32_t                             g_gridPrefetchMB = 4; // MGE_TEX_PREFETCH_MB; 0 = off
    std::uint64_t                             g_gridPrefetched = 0; // session totals
    std::uint64_t                             g_gridPrefetchedBytes = 0;

    // ---- Flip-book texture arrays (NiFlipController) ---------------------------------------
    // A book used to claim ONE SLOT PER FRAME out of the ~872 client slots; Enhanced Light's
    // magelight is 300 frames. Books are uniform by construction, so each becomes a run of LAYERS
    // in a gFlipArrays bucket (see IPC::makeFlipSlot) and the whole book costs ONE descriptor.
    //
    // Buckets are keyed by (format, width, height) and sized in blocks so small books SHARE one:
    // a 4-frame candle flame should not burn a descriptor. The declared size is fixed at creation
    // and every slice of every book in the bucket ships it (TexUploadWire::arraySize), which is
    // what lets the host build the array from whichever slice arrives first.
    struct FlipBucket {
        std::uint64_t key = 0;        // (fourcc/format, w, h) identity; 0 = unused
        std::uint32_t declared = 0;   // layer count declared to the host at creation
        std::uint32_t used = 0;       // layers handed out so far
    };
    FlipBucket                                g_flipBuckets[IPC::kMaxFlipBuckets];
    // Books already registered, keyed by their FIRST frame's normalized name. Books are shared by
    // every instance that plays them (six orbs = one book), which is why the key is the content and
    // not the controller: controllers are per-clone.
    std::unordered_set<std::string>           g_flipBooksDone;
    std::uint32_t                             g_flipBooksBuilt = 0;
    std::uint32_t                             g_flipLayersBuilt = 0;
    std::uint32_t                             g_flipRefused = 0;    // non-uniform / no bucket / unreadable
    // Layer granularity. Rounding up lets several small books share a bucket; a big book still gets
    // an almost-exact fit (300 -> 304). Too large wastes VRAM on tiny books, too small fragments
    // the 16 buckets.
    constexpr std::uint32_t kFlipLayerBlock = 16;

    // Per-frame draw list rides a 4-chunk (4MB) vec: ~61K DrawItemWire, well over the
    // host's kMaxDraws cap. Geometry vec stays 8 chunks (8MB).
    constexpr unsigned kDrawChunks = 4;

    void initSceneVecs();   // defined below; called from lazyInit
    void flushGeometry();
    void flushTextures();
    std::uint32_t resolveTextureSlot(const char* textureName);
    // supplied: a FINISHED first-sight read of this name (the prefetch's, off the I/O thread); its bytes
    // are used instead of reading, when it was made under the same placeholder decision.
    std::uint32_t resolveTextureSlotEx(const char* textureName, bool dataTexture, bool quietMiss,
                                       TexIoRead* supplied = nullptr);

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

    // P3: the host renders into a PAIR of shared RTs, one per host frame slot, so it can draw frame
    // N+1 while we still copy N. Each frame's finish reply names its slot (rtSlot); we copy from it.
    HANDLE           g_hostHandle[2] = {};       // Forge RT shared NT handles (we own them)
    VkExternalMemoryHandleTypeFlagBits g_htype = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    VkImage          g_importImg[2] = {};        // imported host RTs (owned)
    VkDeviceMemory   g_importMem[2] = {};        // imported external memory (owned)

    IDirect3DTexture9* g_mainTex = nullptr;      // DXVK D3D9 RT texture; copy dst + blit src
    VkImage            g_dstImg = VK_NULL_HANDLE;   // its backing VkImage (borrowed)
    VkImageLayout      g_dstLayout = VK_IMAGE_LAYOUT_GENERAL;  // DXVK's resting layout for g_mainTex

    VkCommandPool   g_cmdPool = VK_NULL_HANDLE;  // owned
    VkCommandBuffer g_cmd     = VK_NULL_HANDLE;
    VkFence         g_fence   = VK_NULL_HANDLE;  // owned

    // Tier 1 (tasks/forge-host-gpu-lane.md): the host's SHARED monotonic D3D12 frame fence,
    // imported here as a Vulkan TIMELINE semaphore. When the host stops CPU-blocking on its own
    // frame fence, the RPC reply no longer means "GPU-complete" — this is the sync object that
    // makes the RT copy wait for the host's draw instead. g_frameSemOk is the ONLY authority on
    // whether the import actually worked: a resolved vkImportSemaphoreWin32HandleKHR pointer is
    // not proof the extension was enabled at vkCreateDevice (loaders hand those out regardless).
    HANDLE      g_fenceHandle = nullptr;          // duplicated host fence NT handle (we own it)
    VkSemaphore g_frameSem    = VK_NULL_HANDLE;   // owned
    bool        g_frameSemOk  = false;
    // Tier 1 EVENT handoff — the fallback when the semaphore import is unavailable (Wine/Proton).
    // A manual-reset event the HOST resets and re-arms (SetEventOnCompletion) with every frame's
    // fence signal; we only ever WAIT it, on the CPU, before the RT copy. g_frameEventOk ⇒ we told
    // the host clientSyncsOnFence = 2. Never used while g_frameSemOk (the GPU wait is strictly better).
    // One per host RT slot (P3): the host re-arms only the slot it is about to draw into, so the
    // event we wait for frame N is never reset by the arming of N+1.
    HANDLE      g_frameEvent[2] = {};             // duplicated host event handles (we own them)
    bool        g_frameEventOk = false;

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
        // Tier 1 shared-fence import. OPTIONAL — resolved outside the required-entry-point block
        // below, because a device that lacks VK_KHR_external_semaphore_win32 must still get the
        // (working) seam, just without the semaphore handoff.
        PFN_vkGetPhysicalDeviceExternalSemaphoreProperties GetPhysicalDeviceExternalSemaphoreProperties;
        PFN_vkImportSemaphoreWin32HandleKHR ImportSemaphoreWin32HandleKHR;
        PFN_vkCreateSemaphore            CreateSemaphore;
        PFN_vkDestroySemaphore           DestroySemaphore;
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
        // Tier 1 optional set — deliberately NOT folded into `ok`. Missing entry points here cost
        // us the semaphore handoff, not the seam.
        vk.GetPhysicalDeviceExternalSemaphoreProperties =
            (PFN_vkGetPhysicalDeviceExternalSemaphoreProperties)vk.GetInstanceProcAddr(g_inst, "vkGetPhysicalDeviceExternalSemaphoreProperties");
        vk.ImportSemaphoreWin32HandleKHR =
            (PFN_vkImportSemaphoreWin32HandleKHR)vk.GetDeviceProcAddr(g_dev, "vkImportSemaphoreWin32HandleKHR");
        vk.CreateSemaphore  = (PFN_vkCreateSemaphore)vk.GetDeviceProcAddr(g_dev, "vkCreateSemaphore");
        vk.DestroySemaphore = (PFN_vkDestroySemaphore)vk.GetDeviceProcAddr(g_dev, "vkDestroySemaphore");
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

    // Tier 1 step 0a: can DXVK's device import the host's shared D3D12 fence as a Vulkan semaphore?
    //
    // This is NOT answerable by reading our code: we do not create the Vulkan device, DXVK does, and
    // device extensions must be enabled at vkCreateDevice. So the question is entirely about the
    // shipped DXVK build, and the ONLY honest test is a real vkImportSemaphoreWin32HandleKHR on the
    // real handle — a non-null function pointer proves nothing (loaders return pointers for
    // extensions the device never enabled).
    //
    // A D3D12 fence is a monotonic 64-bit counter, so the natural Vulkan mirror is a TIMELINE
    // semaphore; that is what the RT copy wants to wait on (a specific frame's value). We try
    // timeline first and fall back to BINARY (the pre-timeline VK_KHR_external_semaphore_win32
    // shape, driven by VkD3D12FenceSubmitInfoKHR) only to distinguish "no external-semaphore
    // support at all" from "no timeline import" in the log — the caller only uses the timeline.
    // Sets g_frameSem/g_frameSemOk on success. Never fatal.
    void probeSharedFenceSemaphore() {
        if (!g_fenceHandle) {
            LOG::logline(">> [seam][tier1] host provided no shared frame fence — semaphore handoff unavailable");
            return;
        }
        LOG::logline(">> [seam][tier1] shared frame-fence handle (this process) = %p", g_fenceHandle);

        // Advisory: does the physical device even claim D3D12_FENCE is importable? Logged, not gated
        // on — the import call below is the authority.
        if (vk.GetPhysicalDeviceExternalSemaphoreProperties) {
            VkPhysicalDeviceExternalSemaphoreInfo si = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO };
            si.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
            VkExternalSemaphoreProperties sp = { VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES };
            vk.GetPhysicalDeviceExternalSemaphoreProperties(g_phys, &si, &sp);
            LOG::logline(">> [seam][tier1] D3D12_FENCE external-semaphore features=0x%X (importable=%d) compatible=0x%X",
                         (unsigned)sp.externalSemaphoreFeatures,
                         (int)((sp.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) != 0),
                         (unsigned)sp.compatibleHandleTypes);
        } else {
            LOG::logline(">> [seam][tier1] vkGetPhysicalDeviceExternalSemaphoreProperties unresolved");
        }
        LOG::logline(">> [seam][tier1] entry points: ImportSemaphoreWin32HandleKHR=%p CreateSemaphore=%p",
                     (void*)vk.ImportSemaphoreWin32HandleKHR, (void*)vk.CreateSemaphore);
        if (!vk.ImportSemaphoreWin32HandleKHR || !vk.CreateSemaphore || !vk.DestroySemaphore) {
            LOG::logline("!! [seam][tier1] RESULT: semaphore import UNAVAILABLE (entry points missing)");
            return;
        }

        // MGE_TIER1_SEM=0: skip the import, so a Windows run takes the EVENT handoff — the Wine/Proton
        // path — and it can be exercised (and A/B'd) without the Linux box.
        {
            char v[4] = {};
            if (GetEnvironmentVariableA("MGE_TIER1_SEM", v, sizeof(v)) > 0 && v[0] == '0') {
                LOG::logline(">> [seam][tier1] MGE_TIER1_SEM=0 — semaphore import skipped (event handoff test)");
                return;
            }
        }

        // Wine/Proton (through GE-Proton 11-7 at least) cannot import a D3D12 fence owned by another
        // process: win32u logs "fixme: d3d12 fence from other process" and then faults inside its own
        // Unix side, taking Morrowind down with it. The call never returns an error we could handle,
        // so skip it and let the host keep its own fence wait. MGE_WINE_FENCE_IMPORT=1 retries it.
        if (GetProcAddress(GetModuleHandleA("ntdll.dll"), "wine_get_version")) {
            char force[8] = {};
            if (GetEnvironmentVariableA("MGE_WINE_FENCE_IMPORT", force, sizeof(force)) == 0 || force[0] != '1') {
                LOG::logline("!! [seam][tier1] RESULT: running under Wine — cross-process D3D12 fence import "
                             "skipped (crashes win32u); set MGE_WINE_FENCE_IMPORT=1 to try it anyway");
                return;
            }
            LOG::logline(">> [seam][tier1] running under Wine, MGE_WINE_FENCE_IMPORT=1 — attempting the import");
        }

        // The import does NOT consume the handle (NT handles: we keep ownership and CloseHandle it),
        // so both attempts can use g_fenceHandle directly.
        for (int pass = 0; pass < 2; ++pass) {
            const bool timeline = (pass == 0);
            VkSemaphoreTypeCreateInfo stci = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
            stci.semaphoreType = timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY;
            stci.initialValue  = 0;
            VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            sci.pNext = &stci;
            VkSemaphore sem = VK_NULL_HANDLE;
            VkResult r = vk.CreateSemaphore(g_dev, &sci, nullptr, &sem);
            if (r != VK_SUCCESS || sem == VK_NULL_HANDLE) {
                LOG::logline("!! [seam][tier1] vkCreateSemaphore (%s) failed VkResult=%d",
                             timeline ? "timeline" : "binary", (int)r);
                continue;
            }
            VkImportSemaphoreWin32HandleInfoKHR imp = { VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR };
            imp.semaphore  = sem;
            imp.flags      = 0;   // permanent import: the payload outlives any single wait
            imp.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
            imp.handle     = g_fenceHandle;
            r = vk.ImportSemaphoreWin32HandleKHR(g_dev, &imp);
            LOG::logline(">> [seam][tier1] vkImportSemaphoreWin32HandleKHR(%s, D3D12_FENCE) VkResult=%d",
                         timeline ? "timeline" : "binary", (int)r);
            if (r == VK_SUCCESS) {
                if (timeline) {
                    g_frameSem   = sem;
                    g_frameSemOk = true;
                    LOG::logline(">> [seam][tier1] RESULT: shared-fence semaphore import OK (timeline) — "
                                 "Tier 1 can use the semaphore handoff");
                    return;
                }
                // Binary imported but timeline did not: usable in principle, but the RT copy needs a
                // per-frame VALUE, so we do not adopt it. Report the distinction and stop.
                vk.DestroySemaphore(g_dev, sem, nullptr);
                LOG::logline("!! [seam][tier1] RESULT: only BINARY import works — timeline unavailable; "
                             "Tier 1 must take the double-RT branch");
                return;
            }
            vk.DestroySemaphore(g_dev, sem, nullptr);
        }
        LOG::logline("!! [seam][tier1] RESULT: shared-fence semaphore import FAILED — "
                     "Tier 1 must take the double-RT branch");
    }

    // Import the host's shared NT handle of RT slot i as a VkImage on DXVK's device.
    bool importHostImage(unsigned i) {
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
        if (vk.CreateImage(g_dev, &ici, nullptr, &g_importImg[i]) != VK_SUCCESS) {
            LOG::logline("!! [seam] vkCreateImage (imported host RT %u) failed", i);
            return false;
        }

        VkMemoryRequirements mr = {};
        vk.GetImageMemoryRequirements(g_dev, g_importImg[i], &mr);

        VkMemoryWin32HandlePropertiesKHR whp = { VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR };
        uint32_t handleTypeBits = mr.memoryTypeBits;
        if (vk.GetMemoryWin32HandlePropertiesKHR(g_dev, g_htype, g_hostHandle[i], &whp) == VK_SUCCESS) {
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
        ded.image = g_importImg[i];
        VkImportMemoryWin32HandleInfoKHR imp = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR };
        imp.pNext      = &ded;
        imp.handleType = g_htype;
        imp.handle     = g_hostHandle[i];
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.pNext           = &imp;
        mai.allocationSize  = mr.size;
        mai.memoryTypeIndex = typeIdx;
        VkResult r = vk.AllocateMemory(g_dev, &mai, nullptr, &g_importMem[i]);
        if (r != VK_SUCCESS) {
            LOG::logline("!! [seam] *** vkAllocateMemory (import %u) FAILED VkResult=%d *** (handle type %d not importable on this GPU)", i, (int)r, (int)g_htype);
            return false;
        }
        if (vk.BindImageMemory(g_dev, g_importImg[i], g_importMem[i], 0) != VK_SUCCESS) {
            LOG::logline("!! [seam] vkBindImageMemory (imported host RT %u) failed", i);
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
            if (g_frameSem && vk.DestroySemaphore) { vk.DestroySemaphore(g_dev, g_frameSem, nullptr); }
            g_frameSem = VK_NULL_HANDLE;
            g_frameSemOk = false;
            if (g_fence)     { vk.DestroyFence(g_dev, g_fence, nullptr); g_fence = VK_NULL_HANDLE; }
            if (g_cmdPool)   { vk.DestroyCommandPool(g_dev, g_cmdPool, nullptr); g_cmdPool = VK_NULL_HANDLE; g_cmd = VK_NULL_HANDLE; }
            for (unsigned i = 0; i < 2; ++i) {
                if (g_importImg[i]) { vk.DestroyImage(g_dev, g_importImg[i], nullptr); g_importImg[i] = VK_NULL_HANDLE; }
                if (g_importMem[i]) { vk.FreeMemory(g_dev, g_importMem[i], nullptr); g_importMem[i] = VK_NULL_HANDLE; }
            }
        }
        if (g_mainTex)     { g_mainTex->Release(); g_mainTex = nullptr; }
        g_mainTexValid = false;
        g_dstImg = VK_NULL_HANDLE;
        for (unsigned i = 0; i < 2; ++i) {
            if (g_hostHandle[i]) { CloseHandle(g_hostHandle[i]); g_hostHandle[i] = nullptr; }
            if (g_frameEvent[i]) { CloseHandle(g_frameEvent[i]); g_frameEvent[i] = nullptr; }
        }
        if (g_fenceHandle) { CloseHandle(g_fenceHandle); g_fenceHandle = nullptr; }
        g_frameEventOk = false;
        if (g_vki)         { g_vki->Release(); g_vki = nullptr; }
        g_inst = VK_NULL_HANDLE; g_phys = VK_NULL_HANDLE; g_dev = VK_NULL_HANDLE; g_queue = VK_NULL_HANDLE;
    }

    // Re-derive the current internal render size from g_renderScale + the backbuffer, clamp it to
    // the allocation (g_w/g_h), and restamp it into every subsequent host render RPC. Cheap — call
    // on any scale change (panel slider). Takes effect on the next kickoff with no reallocation.
    void recomputeRenderSize() {
        float s = g_renderScale;
        if (s < kMinRenderScale)  s = kMinRenderScale;
        if (s > g_maxRenderScale) s = g_maxRenderScale;   // never exceed what was ALLOCATED
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
        // The stream lane's batch belongs to the OLD host frame state: let it finish (the host's
        // worker answers it, or answers it failed if its renderer is torn down) before forgetting it.
        if (g_client) { g_client->streamUploadDrain(); }
        {
            std::lock_guard<std::mutex> lk(g_texResidencyMx);
            g_texStreamFlight.clear();
            g_slotStagedBatch.clear();
            g_texSwapSerial = 0;
            g_texShippedSerial = 0;
            g_texSlot.clear();
            g_slotName.clear();
            g_slotLastUsed.clear();
            g_slotBytes.clear();
            g_texFreeSlots.clear();
            g_texStreamQueue.clear();          // its slots belonged to the OLD host
            g_texRetireSlots.clear();          // ...and so did these
            g_gridTexQueue.clear();            // the epoch bump below makes the prefetch rescan
            g_nextTexSlot = 1;                 // 0 = host default white
            ++g_texEpoch;                      // invalidate every cached ResolvedTex fast-path (name+epoch)
            g_texPendingEntries.clear();       // drop tex uploads staged for the OLD host
            g_texPendingBytes = 0;
            g_texPendingCount = 0;
        }
        g_keySlot.clear();
        g_uploadedRev.clear();
        g_blockByContent.clear();              // the fresh host has an empty block table
        g_slotBlock.clear();
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
            RenderProcess::dropPendingCopy();  // its fence/slot name the OLD host's frame
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
            // ⚠ ORDER: the env override is read BEFORE g_w/g_h are derived. It used to be read
            // after, which was harmless only because the ceiling was a hardcoded 2.0 that always
            // covered it. Now that the ceiling defaults to 1.0, reading it afterwards would size
            // the allocation to 1.0x and then set a render scale it has no room for.
            {
                char rs[32] = {};
                if (GetEnvironmentVariableA("MGE_RENDER_SCALE", rs, sizeof(rs)) > 0) {
                    const float s = (float)std::atof(rs);
                    if (s >= kMinRenderScale && s <= kRenderScaleHardCap) {
                        g_maxRenderScale = s;   // raise the ceiling to exactly what was asked for
                        g_renderScale    = s;
                        LOG::logline(">> [seam] MGE_RENDER_SCALE=%.2f applied at init "
                                     "(SSAA opt-in; allocation ceiling raised to %.2fx)", s, s);
                    } else {
                        LOG::logline("!! [seam] MGE_RENDER_SCALE='%s' out of [%.2f..%.2f] — ignored,"
                                     " ceiling stays %.2fx",
                                     rs, kMinRenderScale, kRenderScaleHardCap, g_maxRenderScale);
                    }
                }
            }
            g_w = (UINT)(g_bbW * g_maxRenderScale + 0.5f);
            g_h = (UINT)(g_bbH * g_maxRenderScale + 0.5f);
            // MGE_RENDER_SCALE is read above, before the sizing. It exists so a resolution sweep can
            // be SCRIPTED: the scale is otherwise reachable only through the panel slider, and the
            // perf harness runs minimized with no one at the keyboard — without it, "measure at 3
            // resolutions" means 3 rebuilds that are then not provably identical in anything else.
            // GetEnvironmentVariableA rather than getenv: getenv reads a CRT snapshot and trips
            // C4996 here, and we already have windows.h.
            recomputeRenderSize();   // sets g_rw/g_rh + stamps the host render size (g_client is live here)
            LOG::logline(">> [seam] backbuffer %ux%u — alloc %ux%u (ceiling %.2fx%s), render %ux%u (scale %.2fx)",
                         g_bbW, g_bbH, g_w, g_h, g_maxRenderScale,
                         g_maxRenderScale > 1.0f ? "" : ", SSAA off", g_rw, g_rh, g_renderScale);
        }

        // Host brings up Forge + creates the shared RT; returns the NT handle already
        // duplicated into THIS process.
        HANDLE hostHandle[2] = {};
        HANDLE hostFence = nullptr;
        HANDLE hostEvent[2] = {};
        // MSAA: Configuration.AALevel is the D3DMULTISAMPLE value (0/2/4/8); map 0 -> 1 sample.
        const std::uint32_t sampleCount = Configuration.AALevel > 0 ? (std::uint32_t)Configuration.AALevel : 1u;
        // AF: Configuration.AnisoLevel (0 = off, else max anisotropy) — host sampler (Phase 2).
        const std::uint32_t anisoLevel = (std::uint32_t)Configuration.AnisoLevel;
        const bool initOk = g_client->renderInitBlocking(g_w, g_h, sampleCount, anisoLevel, nullptr, nullptr,
                                                         &hostHandle[0], &hostFence, &hostEvent[0],
                                                         &hostHandle[1], &hostEvent[1]);
        // Adopt everything we were handed first, so releaseAll() closes it on any failure below.
        for (unsigned i = 0; i < 2; ++i) {
            g_hostHandle[i] = hostHandle[i];
            g_frameEvent[i] = hostEvent[i];   // may be null; only used if the semaphore import fails
        }
        g_fenceHandle = hostFence;   // may be null; probeSharedFenceSemaphore reports either way
        if (!initOk || hostHandle[0] == nullptr || hostHandle[1] == nullptr) {
            LOG::logline("!! [seam] renderInit RPC failed or no shared RT pair (%p %p); seam disabled",
                         hostHandle[0], hostHandle[1]);
            releaseAll();
            return;
        }
        LOG::logline(">> [seam] host shared-RT NT handles (this process) = %p %p", hostHandle[0], hostHandle[1]);

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

        if (!importHostImage(0) || !importHostImage(1)) {
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
        // Tier 1: import the host's shared frame fence as a timeline semaphore, and tell the host
        // whether it worked. THIS CALL IS THE SAFETY INTERLOCK — the host only stops fence-waiting
        // its own frame once we have said we can wait it ourselves, so an import failure silently
        // degrades to the old blocking behaviour instead of to a torn composite.
        probeSharedFenceSemaphore();
        // No semaphore ⇒ fall back to the host's frame EVENT: a CPU wait before the copy instead of
        // a GPU one, but it still frees the host from settling its own frame. Needs the fence too —
        // the event is only ever set by that fence's SetEventOnCompletion.
        g_frameEventOk = !g_frameSemOk && g_frameEvent[0] != nullptr && g_frameEvent[1] != nullptr
                      && g_fenceHandle != nullptr;
        {   // MGE_TIER1_EVENT=0: force the old blocking contract, for A/B against the event handoff.
            char v[4] = {};
            if (g_frameEventOk && GetEnvironmentVariableA("MGE_TIER1_EVENT", v, sizeof(v)) > 0 && v[0] == '0') {
                g_frameEventOk = false;
                LOG::logline(">> [seam][tier1] MGE_TIER1_EVENT=0 — event handoff declined");
            }
        }
        g_client->setClientSyncsOnFence(g_frameSemOk ? 1u : g_frameEventOk ? 2u : 0u);
        LOG::logline(">> [seam][tier1] host frame overlap %s",
                     g_frameSemOk   ? "ENABLED (client GPU-waits the shared fence before each RT copy)"
                     : g_frameEventOk ? "ENABLED via EVENT (client CPU-waits the host's frame event before each RT copy)"
                                      : "DISABLED (no semaphore, no event — host keeps its own fence wait)");
        {   // MGE_COPY_AT_BLIT=0|1: the harness's switch for the 1-ahead vs 1.5-ahead A/B (numpad-*
            // cycles it live). Read here so every seam bring-up starts from the environment's choice.
            char v[4] = {};
            if (GetEnvironmentVariableA("MGE_COPY_AT_BLIT", v, sizeof(v)) > 0) {
                g_copyAtBlit = (v[0] != '0');
                LOG::logline(">> [seam] MGE_COPY_AT_BLIT=%c — RT copy %s", v[0],
                             g_copyAtBlit ? "at the blit (1.5-ahead)" : "at the collect (1-ahead)");
            }
            // MGE_FRAME_AHEAD=0: boot with frame-ahead OFF — the third numpad-* state, which the
            // harness cannot reach by key. Exists so the OFF path gets tested at all (it once froze).
            if (GetEnvironmentVariableA("MGE_FRAME_AHEAD", v, sizeof(v)) > 0) {
                g_frameAheadLive = (v[0] != '0');
                LOG::logline(">> [seam] MGE_FRAME_AHEAD=%c — frame-ahead pipelining %s", v[0],
                             g_frameAheadLive ? "ON" : "OFF");
            }
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
    //
    // hostFenceValue (Tier 1): the value the host signalled on the SHARED D3D12 frame fence for the
    // frame we are copying. The host no longer blocks on its own fence, so the RPC reply arrives
    // while the GPU may still be drawing into this very image — without an explicit wait, every
    // frame would copy a half-drawn RT. We add the imported timeline semaphore as a WAIT on the
    // copy's submit, which is a GPU-side dependency: it costs nothing on the CPU, and under
    // frame-ahead (deferFinish) the copy already happens at the NEXT frame's BeginScene, by which
    // point the semaphore is long signalled and the wait is free in the steady state.
    //
    // 0 (or no imported semaphore) ⇒ no sync object. We cannot make the copy safe from this side, so
    // the client tells the host so every kickoff (renderFrameParams.clientSyncsOnFence) and the host
    // keeps its old end-of-frame fence wait — the reply then means "GPU-complete" exactly as before
    // and this copy is safe with no wait at all. That is the fallback for a DXVK build where step
    // 0a's import fails; it costs the Tier 1 win, not correctness.
    //
    // rtSlot (P3): which of the host's two shared RTs this frame rendered into (the finish reply's
    // rtSlot) — the image copied, and in event mode the event waited.
    bool copyHostRtToDst(std::uint64_t hostFenceValue, std::uint32_t rtSlot) {
        const unsigned slot = rtSlot & 1u;
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
        vk.CmdCopyImage(g_cmd, g_importImg[slot], VK_IMAGE_LAYOUT_GENERAL,
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

        // Tier 1: order this copy behind the host's draw with a GPU-side wait on the imported
        // timeline semaphore. TOP_OF_PIPE is the correct stage mask for a timeline wait that must
        // gate everything in the command buffer, including the first layout transition — the copy
        // has no earlier work to overlap with, so nothing is lost by blocking the whole submit.
        const uint64_t waitValue = hostFenceValue;
        VkTimelineSemaphoreSubmitInfo tsi = { VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        if (g_frameSemOk && waitValue != 0) {
            tsi.waitSemaphoreValueCount = 1;
            tsi.pWaitSemaphoreValues    = &waitValue;
            si.pNext                = &tsi;
            si.waitSemaphoreCount   = 1;
            si.pWaitSemaphores      = &g_frameSem;
            si.pWaitDstStageMask    = &waitStage;
        }

        // Tier 1 EVENT handoff: no semaphore to put on the submit, so hold the submit itself until
        // the host's frame event says its fence reached this frame's value. Placed after recording
        // so the record overlaps the host's GPU tail. The event is manual-reset and only the host
        // resets it — slot `slot`'s event only when it arms the NEXT frame in this slot (N+2), which
        // cannot happen until we have copied this one and kicked N+2 off — so a set event here
        // always vouches for THIS frame. A timeout copies anyway: a torn frame beats a hung game;
        // it is logged.
        const double tcEv0 = nowMs();
        if (g_frameEventOk && waitValue != 0) {
            MGE_ZoneScopedN("RTcopy: frame event wait");
            const DWORD w = WaitForSingleObject(g_frameEvent[slot], 2000);
            if (w != WAIT_OBJECT_0) {
                static unsigned s_evTimeouts = 0;
                if (s_evTimeouts++ < 8) {
                    LOG::logline("!! [seam][tier1] frame event wait for fence %llu timed out (wait=0x%lX) — copying anyway",
                                 (unsigned long long)waitValue, (unsigned long)w);
                }
            }
        }
        const double tcRecord = nowMs();
        g_lastEventWaitMs = tcRecord - tcEv0;
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
        // The submit above waits on the host's timeline semaphore, so the fence clearing is the
        // first moment the client KNOWS the host GPU finished this frame. Pair for the 1.0 set at
        // kickoff. Set unconditionally (even on a failed submit) so a bad frame cannot leave the
        // step latched high and paint every later frame as GPU-inflight.
        MGE_TracyPlot("Forge host GPU inflight", 0.0);
        const double tcEnd = nowMs();

        const double total = tcEnd - tc0;
        MGE_TracyPlot("Forge RTcopy flush ms", tcFlush - tc0);
        MGE_TracyPlot("Forge RTcopy wait ms",  tcEnd - tcSubmit);
        MGE_TracyPlot("Forge RTcopy event wait ms", tcRecord - tcEv0);
        if (FrameTrace::enabled()) {
            const std::int64_t fv = (std::int64_t)waitValue;
            g_traceCopyFence = fv;
            FrameTrace::span("client main", "RT flush",      tc0,      tcFlush,  fv);
            FrameTrace::span("client main", "RT record",     tcFlush,  tcEv0,    fv);
            FrameTrace::span("client main", "event wait",    tcEv0,    tcRecord, fv);
            FrameTrace::span("client main", "RT submit",     tcRecord, tcSubmit, fv);
            FrameTrace::span("client main", "RT copy GPU",   tcSubmit, tcEnd,    fv);
        }
        // Only rare spikes log. The trigger deliberately EXCLUDES the fence-wait phase: since Tier 1
        // that phase also contains the semaphore wait for the host's draw, which is multi-ms by
        // design on every frame — testing `total` here made this fire once per frame (1848 lines in
        // one session), and a LOG::logline on the per-frame path is the exact shape that collapsed
        // the multimap night scene. What is still worth a line is a stall in the parts that are
        // supposed to be ~0: the DXVK flush, the submit lock, the record, the submit. The waiting
        // itself stays visible in the Tracy plot below and in the [hb] copy= average.
        // The frame-event wait (Wine path) is a host-GPU wait by design too, so it is excluded the
        // same way and reported as its own column.
        const double evWait = tcRecord - tcEv0;
        const double nonWait = total - (tcEnd - tcSubmit) - evWait;
        if (nonWait > 3.0) {
            LOG::logline(">> [rtcopy] spike nonWait=%.2fms total=%.2f | flush=%.2f record=%.2f evwait=%.2f lock=%.2f submit=%.2f wait=%.2f",
                         nonWait, total, tcFlush - tc0, tcEv0 - tcFlush, evWait, tcLocked - tcRecord,
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

        // Point-light list (Tier 3a) rides its own 1-chunk vec — kMaxPointLights * 56B ≈ 7KB (48B wire + the G3 gobo side entry),
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
        // FP1e: multi-map FP parts (a glass weapon's glow map, an enchanted gauntlet's detail map)
        // ride their own tiny MultiMapDrawWire[] vec — the same wire format as the world multi-map
        // list, drawn by the host FP pass with its own GEQUAL+write PSO pair.
        auto fmv = g_client->allocVecBlocking<IPC::GeomChunk>(1, 1, 1);
        if (!fmv) {
            LOG::logline("!! [seam] FP multi-map draw-list vec alloc failed — FP multi-map (FP1e) disabled");
        } else {
            g_fpMultiMapVec.emplace(std::move(*fmv));
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

        // First-sight streaming budget (g_texStreamQueue), read before the first texture resolves.
        {
            char e[32] = {};
            if (GetEnvironmentVariableA("MGE_TEX_STREAM_MB", e, sizeof(e)) > 0) {
                g_texStreamBudgetMB = (std::uint32_t)std::strtoul(e, nullptr, 10);
            }
            LOG::logline(">> [tex-stream] first-sight streaming %s (%u MB of full files per frame, DL LOD"
                         " copies as placeholders; MGE_TEX_STREAM_MB overrides, 0 = off)",
                         g_texStreamBudgetMB ? "ON" : "OFF", g_texStreamBudgetMB);
            if (GetEnvironmentVariableA("MGE_TEX_PREFETCH_MB", e, sizeof(e)) > 0) {
                g_gridPrefetchMB = (std::uint32_t)std::strtoul(e, nullptr, 10);
            }
            LOG::logline(">> [tex-prefetch] active-grid texture prefetch %s (%u MB per frame;"
                         " MGE_TEX_PREFETCH_MB overrides, 0 = off)",
                         g_gridPrefetchMB ? "ON" : "OFF", g_gridPrefetchMB);
            if (GetEnvironmentVariableA("MGE_TEX_STREAM_ASYNC", e, sizeof(e)) > 0) {
                g_texStreamAsync = std::strtoul(e, nullptr, 10) != 0;
            }
        }
        // The async stream lane's own window (see g_texStreamFlight). Only when the lane is on: it is
        // 32 MB of address space in a 32-bit process.
        if (g_texStreamAsync && g_texStreamBudgetMB != 0) {
            auto stv = g_client->allocVecBlocking<IPC::GeomChunk>(
                IPC::kTexChunks, IPC::kTexChunks, IPC::kTexChunks);
            if (!stv) {
                LOG::logline("!! [tex-stream] stream vec alloc failed — full files ride the sync flush");
            } else {
                g_texStreamVec.emplace(std::move(*stv));
            }
        }
        LOG::logline(">> [tex-stream] async lane %s (MGE_TEX_STREAM_ASYNC=0 = full files in the sync flush)",
                     g_texStreamVec ? "ON" : "OFF");
        // The texture I/O thread (see TexIo). It reads ahead for the stream lane and the prefetch, so
        // with neither on it has nothing to do.
        {
            bool ioOn = true;
            char e[16] = {};
            if (GetEnvironmentVariableA("MGE_TEX_IO_THREAD", e, sizeof(e)) > 0) {
                ioOn = std::strtoul(e, nullptr, 10) != 0;
            }
            if (ioOn && (g_texStreamBudgetMB != 0 || g_gridPrefetchMB != 0)) {
                g_texIo.start();
            }
            LOG::logline(">> [tex-io] texture I/O thread %s (MGE_TEX_IO_THREAD=0 = reads inline in the build)",
                         g_texIo.on() ? "ON" : "OFF");
        }
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

    // No draw list still in flight can name slot s: its last reference is at least IPC::kTexColdFrames
    // old. Read BEFORE the caller stamps the slot for its new use. Callers hold g_texResidencyMx.
    // Released slots rest at g_slotLastUsed 0, so they are cold once the session is 8 frames old.
    inline bool slotIsCold(std::uint32_t s) {
        return s != 0 && s < g_slotLastUsed.size() && g_frame - g_slotLastUsed[s] >= IPC::kTexColdFrames;
    }
    inline std::uint32_t coldBit(std::uint32_t s) { return slotIsCold(s) ? IPC::kTexUploadCold : 0u; }

    // Stage one [TexUploadWire][dds] entry for the next flush. Callers hold g_texResidencyMx.
    // Returns false if the allocation failed — a texture we cannot stage must go WHITE, never take
    // the process down. The produce worker runs this with no handler above it (std::thread ->
    // ProduceWorker::run -> kickoffBody), so an escaping bad_alloc is a fail-fast, not an error.
    // Degrade, never dereference — the same rule the host-side index bounds follow.
    bool stageTexUpload(const IPC::TexUploadWire& hdr, const void* data, unsigned size) {
        try {
            std::vector<std::uint8_t> entry(sizeof(hdr) + size);
            std::memcpy(entry.data(), &hdr, sizeof(hdr));
            if (size) { std::memcpy(entry.data() + sizeof(hdr), data, size); }   // 0: a release
            g_texPendingBytes += entry.size();
            g_texPendingEntries.push_back(std::move(entry));
            ++g_texPendingCount;
            // The stream lane may not take this slot until the flush carrying this record has
            // shipped (see g_slotStagedBatch). The NEXT flush to start carries it.
            const std::uint32_t s = hdr.slot & ~(IPC::kTexUploadData | IPC::kTexUploadRelease | IPC::kTexUploadCold);
            if (!IPC::isFlipSlot(hdr.slot) && s < g_slotStagedBatch.size()) {
                g_slotStagedBatch[s] = g_texSwapSerial + 1;
            }
            return true;
        } catch (const std::bad_alloc&) {
            static std::uint32_t s_staged = 0;
            if (s_staged++ < 8) {
                LOG::logline("!! [tex] out of memory staging %u bytes (%zu MB already staged, %u "
                             "entries) — texture goes white",
                             size, g_texPendingBytes >> 20, g_texPendingCount);
            }
            return false;
        }
    }

    // Map a texture name to its bindless slot, loading + queueing its DDS on first sight.
    // Misses / oversize / residency-full → slot 0 (host default white), cached so we don't retry.
    std::uint32_t resolveTextureSlot(const char* textureName) {
        return resolveTextureSlotEx(textureName, false, false);
    }

    // dataTexture: ship it with IPC::kTexUploadData, so the host keeps its stored UNORM format
    //   instead of the scene's sRGB view (a _paramh's RGB is metal/rough/IOR, not colour).
    // quietMiss: a miss is the EXPECTED answer (probing for a companion file most textures do not
    //   have), so it must not spend the 20-line "not found (white)" budget — which exists to
    //   surface textures that SHOULD resolve — on files that were never there.
    std::uint32_t resolveTextureSlotEx(const char* textureName, bool dataTexture, bool quietMiss,
                                       TexIoRead* supplied) {
        if (!textureName || !*textureName || !g_texVec) {
            return 0;
        }
        std::string name = normalizeTextureName(textureName);
        if (name.empty()) {
            return 0;
        }
        // Serialise the whole residency mutation against the other thread (see g_texResidencyMx).
        // ⚠ NOT ACROSS THE DISK READ (P1, tasks/forge-crossing-frame.md). It used to be: the build's
        // first sights and the prefetch drain held the lock through every read, so main's
        // captureAlphaDraw waited on the worker's disk I/O. BSA reads are positional now and need no
        // lock, so the lookup takes the lock, the read runs without it, and the assignment takes it
        // again and RE-CHECKS: the other thread may have resolved the same name meanwhile, and then
        // its slot wins and these bytes are dropped.
        const std::uint32_t cap = IPC::kMaxTextures - IPC::kDlReserve;   // client range [1, cap)
        auto lookup = [&](std::uint32_t* out) {
            if (g_slotName.size() != IPC::kMaxTextures) {
                g_slotName.assign(IPC::kMaxTextures, std::string());
                g_slotLastUsed.assign(IPC::kMaxTextures, 0u);
                g_slotBytes.assign(IPC::kMaxTextures, 0u);
                g_slotStagedBatch.assign(IPC::kMaxTextures, 0u);
            }
            auto it = g_texSlot.find(name);
            if (it == g_texSlot.end()) { return false; }
            // A flip-book frame resolves to an ENCODED array slot, not an index into the LRU range —
            // subscripting g_slotLastUsed with it would run ~0x8000 past the end. Array-backed
            // textures are resident for the session and never recycled, so they have no LRU age.
            if (it->second != 0 && !IPC::isFlipSlot(it->second)) {
                g_slotLastUsed[it->second] = g_frame;   // refresh LRU age
            }
            *out = it->second;   // already resolved (slot, encoded flip slot, or cached-miss 0)
            return true;
        };
        // First-sight streaming (see g_texStreamQueue): a texture the DL bake has a LOD copy of goes
        // in as that copy now and streams its full file later; a _paramh that exists is deferred with
        // nothing to show. Both skip the full read here, which is the whole point. A load's first
        // build (g_texSyncFrame) reads full files.
        bool deferAllowed;
        {
            std::lock_guard<std::mutex> lk(g_texResidencyMx);
            std::uint32_t known = 0;
            if (lookup(&known)) { return known; }
            deferAllowed = g_texStreamBudgetMB != 0 && g_frame != g_texSyncFrame;
        }
        FirstSightBytes r;
        if (supplied && supplied->done.load(std::memory_order_acquire) && supplied->deferAllowed == deferAllowed
            && supplied->dataTexture == dataTexture && supplied->name == name) {
            r = takeBytes(*supplied);
        } else {
            r = readFirstSight(name, dataTexture, deferAllowed);
        }
        void* data = r.data;
        unsigned size = r.size;
        const bool deferred = r.deferred;

        std::lock_guard<std::mutex> lk(g_texResidencyMx);
        {
            std::uint32_t known = 0;
            if (lookup(&known)) {
                if (data) { std::free(data); }
                return known;
            }
        }
        if (!r.found) {
            static int misses = 0;
            if (!quietMiss && misses < 20) { LOG::logline("!! [tex] not found: %s (white)", name.c_str()); ++misses; }
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

        // NO STAGING BUDGET HERE. There was one (defer past 96 MB, re-resolve next frame) and it was
        // the wrong mechanism twice over. Practically: a deferral returns 0, which is
        // indistinguishable from "white" to every caller that memoises, and it cost 130 permanently
        // white textures on the Seyda Neen load — measured 0 before the change. Threading a
        // `deferred` flag out to the three SlotInfo memos recovered 107 of them and left 23 still
        // stuck, i.e. the mechanism kept finding new ways to be mistaken for a miss.
        //
        // Structurally: the client is a D3D8 PROXY and controls what Morrowind loads. A 21 MB blob
        // should never reach this path — the size is capped at the source by shipping a mip slice,
        // not by rationing the staging buffer downstream of a decision already made wrong. Bounding
        // the SUM was the right instinct about a contiguous vector (the deque above is that fix);
        // bounding it by REFUSING WORK was not. See tasks/forge-pbr-materials.md for the slice plan.
        // Assign a slot: reuse one eviction released, else grow while the range has room, else
        // recycle the least-recently-used slot.
        std::uint32_t slot;
        if (!g_texFreeSlots.empty()) {
            slot = g_texFreeSlots.back();
            g_texFreeSlots.pop_back();
        } else if (g_nextTexSlot < cap) {
            slot = g_nextTexSlot++;
        } else {
            std::uint32_t lru = 1, best = 0xFFFFFFFFu;
            for (std::uint32_t s = 1; s < cap; ++s) {
                if (g_slotLastUsed[s] < best) { best = g_slotLastUsed[s]; lru = s; }
            }
            ++g_texRecycles;
            if (best == g_frame) {
                // Working set > capacity THIS frame: unavoidable thrash, textures go white. Rate
                // limited rather than once-per-session, so a scene that starts thrashing an hour
                // in still says so — but carrying the running count, so the log can't flood.
                ++g_texThrashes;
                static std::uint32_t lastWarnFrame = 0;
                if (g_frame - lastWarnFrame >= 600) {
                    lastWarnFrame = g_frame;
                    LOG::logline("!! [tex] working set exceeds %u client slots — thrashing (white)"
                                 " [%llu thrashes / %llu recycles this session]",
                                 cap, (unsigned long long)g_texThrashes,
                                 (unsigned long long)g_texRecycles);
                }
            }
            g_texSlot.erase(g_slotName[lru]);   // evict the recycled name
            slot = lru;
            // The recycled slot now means a different texture: every SlotInfo-cached
            // slot value is suspect. Epoch bump forces per-key re-resolve (one-off).
            ++g_texEpoch;
        }
        // Cold (no in-flight frame names it) is decided from the slot's PREVIOUS use: fresh and
        // released slots are, a just-recycled LRU slot may not be — then the host waits, as before.
        const std::uint32_t cold = coldBit(slot);
        g_texSlot[name] = slot;
        g_slotName[slot] = name;
        g_slotLastUsed[slot] = g_frame;
        g_slotBytes[slot] = size;
        const bool havePlaceholder = (data != nullptr);
        bool staged;
        if (deferred && !havePlaceholder) {
            // Nothing to show until the file lands (a param map whose bake predates the companion
            // copies). Reset the slot anyway: an LRU-recycled slot still holds its previous occupant
            // on the host, and a stale param map would shade this draw with someone else's material.
            // White + not-data is exactly "no param map" to the shader.
            const IPC::TexUploadWire rel{ slot | IPC::kTexUploadRelease | cold, 0u, 0u };
            staged = stageTexUpload(rel, nullptr, 0u);
        } else {
            IPC::TexUploadWire hdr{ slot | (dataTexture ? IPC::kTexUploadData : 0u) | cold, size, 0u };
            staged = stageTexUpload(hdr, data, size);
        }
        std::free(data);
        if (staged && deferred) {
            g_texStreamQueue.push_back(TexStreamJob{ slot, name, dataTexture });
            if (havePlaceholder) { ++g_texPlaceholders; } else { ++g_texDeferredData; }
        }
        if (!staged) {
            // Out of memory, not a bad file. Hand the slot straight back (free list, empty name)
            // and cache the miss so a scene full of textures we cannot stage doesn't re-read every
            // one of them from the BSA every frame.
            g_slotName[slot].clear();
            g_slotLastUsed[slot] = 0;
            g_slotBytes[slot] = 0;
            g_texFreeSlots.push_back(slot);
            g_texSlot[name] = 0;
            return 0;
        }
        return slot;
    }

    // Minimal DDS header identity for bucketing: (pixel-format, width, height). We never decode —
    // the host does that — so this only has to be a faithful EQUALITY key, and the raw format words
    // are exactly that. Returns 0 for anything that isn't a DDS we can key on.
    static std::uint64_t ddsBucketKey(const void* data, unsigned size,
                                      std::uint32_t* outW, std::uint32_t* outH) {
        if (!data || size < 128) { return 0; }
        const auto* d = static_cast<const std::uint8_t*>(data);
        auto rd = [&](unsigned off) {
            std::uint32_t v; std::memcpy(&v, d + off, 4); return v;
        };
        if (rd(0) != 0x20534444u) { return 0; }               // 'DDS '
        const std::uint32_t h = rd(12), w = rd(16);
        const std::uint32_t pfFlags = rd(80), fourCC = rd(84), rgbBits = rd(88);
        if (w == 0 || h == 0) { return 0; }
        // FourCC when compressed, else bit depth — enough to separate DXT1/3/5 from 32-bit BGRA,
        // which is all the host's own bucketing distinguishes.
        const std::uint32_t fmt = (pfFlags & 0x4u) ? fourCC : (0x80000000u | rgbBits);
        if (outW) { *outW = w; }
        if (outH) { *outH = h; }
        return ((std::uint64_t)fmt << 32) ^ ((std::uint64_t)w << 16) ^ (std::uint64_t)h;
    }

    // Register a NiFlipController's whole frame list as a run of layers in a gFlipArrays bucket,
    // and point every frame's NAME at its encoded array slot. That last part is what makes this
    // invisible to the rest of the client: every consumer already resolves a texture by name
    // (buildGeometryDrawLists via the entry's textureName, captureAlphaDraw via the reverse map),
    // so once g_texSlot holds encoded slots, both paths ship the right slice with no other change.
    //
    // Refuses — leaving the book on the per-slot path, which still works — when the frames are not
    // uniform, when no bucket can hold them, or when a frame won't load. Called once per distinct
    // book (keyed by its first frame), so N casts of the same spell share one array.
    bool registerFlipBookImpl(const char* const* names, std::uint32_t count) {
        if (!names || count == 0 || count > IPC::kMaxFlipLayers || !g_texVec) { return false; }

        std::vector<std::string> norm;
        norm.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            std::string n = normalizeTextureName(names[i]);
            if (n.empty()) { return false; }
            norm.push_back(std::move(n));
        }

        std::lock_guard<std::mutex> lk(g_texResidencyMx);
        if (!g_flipBooksDone.insert(norm[0]).second) { return true; }   // already built (or refused)

        // Pass 1: load every frame and check uniformity BEFORE claiming any layers — a half-built
        // book that then refuses would strand its bucket range and leave the rest per-slot, i.e.
        // the same texture reachable two ways.
        std::vector<void*>         blobs(count, nullptr);
        std::vector<unsigned>      sizes(count, 0u);
        std::uint64_t key = 0;
        std::uint32_t w = 0, h = 0;
        bool ok = true;
        for (std::uint32_t i = 0; i < count && ok; ++i) {
            if (!BSA::loadFileBytes(norm[i].c_str(), &blobs[i], &sizes[i], true)
                || !blobs[i] || sizes[i] == 0) { ok = false; break; }
            std::uint32_t fw = 0, fh = 0;
            const std::uint64_t k = ddsBucketKey(blobs[i], sizes[i], &fw, &fh);
            if (k == 0) { ok = false; break; }
            if (i == 0) { key = k; w = fw; h = fh; }
            else if (k != key) { ok = false; }                  // non-uniform book
            if (sizeof(IPC::TexUploadWire) + sizes[i] > (std::size_t)IPC::kTexWindowBytes) { ok = false; }
        }

        std::uint32_t bucket = IPC::kMaxFlipBuckets, baseLayer = 0;
        if (ok) {
            // Reuse a bucket of the same identity with room, else open a new one. Sizing in blocks
            // is what lets several small books share (a 4-frame flame must not burn a descriptor).
            for (std::uint32_t b = 0; b < IPC::kMaxFlipBuckets && bucket == IPC::kMaxFlipBuckets; ++b) {
                if (g_flipBuckets[b].key == key && g_flipBuckets[b].used + count <= g_flipBuckets[b].declared) {
                    bucket = b;
                }
            }
            if (bucket == IPC::kMaxFlipBuckets) {
                for (std::uint32_t b = 0; b < IPC::kMaxFlipBuckets; ++b) {
                    if (g_flipBuckets[b].key == 0) {
                        std::uint32_t decl = ((count + kFlipLayerBlock - 1) / kFlipLayerBlock) * kFlipLayerBlock;
                        if (decl > IPC::kMaxFlipLayers) { decl = IPC::kMaxFlipLayers; }
                        g_flipBuckets[b].key = key;
                        g_flipBuckets[b].declared = decl;
                        g_flipBuckets[b].used = 0;
                        bucket = b;
                        break;
                    }
                }
            }
            if (bucket == IPC::kMaxFlipBuckets) { ok = false; }   // all 16 buckets spoken for
            else { baseLayer = g_flipBuckets[bucket].used; }
        }

        if (!ok) {
            for (std::uint32_t i = 0; i < count; ++i) { if (blobs[i]) { std::free(blobs[i]); } }
            ++g_flipRefused;
            LOG::logline("!! [flipbook] '%s' (%u frames) refused — non-uniform, unreadable, or no free "
                         "bucket; stays on the per-slot path", norm[0].c_str(), count);
            return false;
        }

        // Pass 2: claim the layers and queue every slice. arraySize is the BUCKET's declared size,
        // not this book's frame count — the host creates one array per bucket and later books fill
        // the remaining layers of the same array.
        const std::uint32_t declared = g_flipBuckets[bucket].declared;
        g_flipBuckets[bucket].used += count;
        for (std::uint32_t i = 0; i < count; ++i) {
            const std::uint32_t slot = IPC::makeFlipSlot(bucket, baseLayer + i);
            g_texSlot[norm[i]] = slot;
            IPC::TexUploadWire hdr{ slot, sizes[i], declared };
            // A slice that will not stage leaves its layer claimed but never uploaded — the host
            // shows that frame white. The book has already claimed its bucket range by here (pass 2
            // is past the point of refusal), so this is the honest degradation: keep going and say
            // so, rather than strand the whole book or take the process down.
            stageTexUpload(hdr, blobs[i], sizes[i]);
            std::free(blobs[i]);
        }
        ++g_flipBooksBuilt;
        g_flipLayersBuilt += count;
        LOG::logline(">> [flipbook] '%s' %u frames -> bucket %u layers %u..%u (%ux%u, bucket holds %u) "
                     "— %u bindless slots saved",
                     norm[0].c_str(), count, bucket, baseLayer, baseLayer + count - 1,
                     w, h, declared, count > 1 ? count - 1 : 0);
        return true;
    }

    // Put an unflushed remainder back at the FRONT of the pending queue, ahead of anything main
    // staged while the lock was released, and restore its share of the totals. Caller holds
    // g_texResidencyMx. Moves entry by entry — copying would defeat the whole point of chunking.
    void requeueTexEntries(std::deque<std::vector<std::uint8_t>>& entries) {
        while (!entries.empty()) {
            g_texPendingBytes += entries.back().size();
            ++g_texPendingCount;
            g_texPendingEntries.push_front(std::move(entries.back()));   // back-to-front: order kept
            entries.pop_back();
        }
    }

    // Ship queued texture uploads to the host in window-sized batches on whole-entry
    // boundaries (each entry is guaranteed <= window by resolveTextureSlot). Blocking RPCs.
    void flushTextures() {
        // Take ownership of the pending entries under the lock, then batch + RPC on the LOCAL deque
        // UNLOCKED — main's captureAlphaDraw keeps staging into a fresh g_texPendingEntries, and we
        // never hold g_texResidencyMx across a blocking RPC (that would be a new 60s-freeze class).
        std::deque<std::vector<std::uint8_t>> entries;
        std::uint32_t swapSerial = 0;
        {
            std::lock_guard<std::mutex> lk(g_texResidencyMx);
            if (!g_texVec || g_texPendingEntries.empty()) {
                return;
            }
            entries.swap(g_texPendingEntries);   // main now stages into an empty deque, race-free
            g_texPendingBytes = 0;               // fresh totals for whatever main stages from here
            g_texPendingCount = 0;
            swapSerial = ++g_texSwapSerial;      // this flush carries every record staged before now
        }
        // A0 sub-buckets (see flushGeometry): assign vs RPC wait, one line per >=1ms flush.
        const double tFlush0 = nowMs();
        double assignMs = 0.0, rpcMs = 0.0;
        const std::size_t windowBytes = (std::size_t)IPC::kTexWindowBytes;
        std::size_t total = 0;
        std::uint32_t entriesFlushed = 0;
        std::vector<const void*>   parts;      // window-sized batch, gathered by POINTER
        std::vector<std::uint32_t> partSizes;  // (assign_gather copies straight into shared memory)
        while (!entries.empty()) {
            // Select whole entries until the window is full. Nothing is copied or consumed here —
            // a host that isn't ready must be able to hand the whole remainder back, this batch
            // included, and the single copy that does happen goes direct to the IPC window.
            std::size_t bytes = 0;
            parts.clear();
            partSizes.clear();
            for (const std::vector<std::uint8_t>& e : entries) {
                if (bytes + e.size() > windowBytes) { break; }
                parts.push_back(e.data());
                partSizes.push_back((std::uint32_t)e.size());
                bytes += e.size();
            }
            const std::uint32_t batchCount = (std::uint32_t)parts.size();
            if (batchCount == 0) { break; }   // safety (each entry <= window by construction)
            total += bytes;
            std::uint32_t uploaded = 0;
            const double tAssign0 = nowMs();
            const bool assigned = g_texVec->assign_gather(parts.data(), partSizes.data(), batchCount);
            assignMs += nowMs() - tAssign0;
            if (assigned) {
                const double tRpc0 = nowMs();
                MGE_ZoneScopedN("Forge texUpload RPC");
                g_client->texUploadBlocking(g_texVec->id(), batchCount, (std::uint32_t)bytes, &uploaded);
                rpcMs += nowMs() - tRpc0;
            }
            if (uploaded == 0xFFFFFFFFu) {
                // Host opaque path not built yet (first scene frame) — keep the UNFLUSHED remainder
                // for next frame, IN ORDER, ahead of whatever main staged during the unlocked RPC.
                // Nothing was popped, so `entries` IS the whole remainder, this batch included.
                // Slots are already assigned; draws show white until then.
                std::lock_guard<std::mutex> lk(g_texResidencyMx);
                requeueTexEntries(entries);
                return;
            }
            // The host returns how many it actually BUILT. Anything less is a silent drop —
            // unsupported DDS, an out-of-range slot, a flip slice whose bucket refused it — and the
            // only symptom downstream is geometry drawing white with nothing in either log saying
            // why. Cheap to check, and it is the receipt that a flip book's slices really landed.
            if (uploaded != batchCount) {
                static std::uint32_t s_mismatchLogged = 0;
                if (s_mismatchLogged++ < 8) {
                    LOG::logline("!! [texflush] host built %u of %u sent (%u dropped — unsupported "
                                 "format / bad slot / rejected flip slice)",
                                 uploaded, batchCount, batchCount - uploaded);
                }
            }
            entriesFlushed += batchCount;
            entries.erase(entries.begin(), entries.begin() + batchCount);   // consumed
        }
        // Fully consumed the local deque. Do NOT touch g_texPendingEntries/Bytes/Count here — they
        // now hold only what main staged during the unlocked RPC, kept for the next flush.
        {
            // Every record this flush carried is on the host now: the stream lane may use its slots.
            // (The requeue path above returns before this — its records ship with a later flush.)
            std::lock_guard<std::mutex> lk(g_texResidencyMx);
            if (swapSerial > g_texShippedSerial) { g_texShippedSerial = swapSerial; }
        }
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
            dropBlockRef(ks->second.slot);   // the host drops its ref when the record arrives (later)
            g_keySlot.erase(ks);
            g_uploadedRev.erase(key);
        }
    }

    // Append up to `limit` queued slot-releases to g_pendingBlob as header-only sentinels. Called by
    // flushGeometry each present with kMaxReleasesPerFlush: bounding the per-flush count keeps the
    // release contribution to any one blocking RPC small while the backlog drains over subsequent
    // frames. A LOAD ships the whole queue instead (checkCellEpochAndPurge), ahead of the new cell's
    // parts. Queue order is irrelevant (monotonic slots, no reuse), so a plain front-drain is fine.
    void appendBoundedReleaseRecords(std::size_t limit = kMaxReleasesPerFlush) {
        if (g_pendingReleaseSlots.empty()) return;
        const std::size_t n = std::min<std::size_t>(g_pendingReleaseSlots.size(), limit);
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

    // Ship queued full textures (see g_texStreamQueue) into their slots, at most g_texStreamBudgetMB
    // per frame and always at least one, so a single 21 MB 4K map cannot starve behind the budget.
    // Called at the end of the BUILD (worker in the async modes, never the park fire on main), so the
    // disk reads land where the draw-list build already pays for first-sight reads. (This sync path —
    // MGE_TEX_STREAM_ASYNC=0 — still reads under g_texResidencyMx. That is no longer REQUIRED: BSA reads
    // are positional since P1, so concurrent readers do not share a seek position.)
    // Placeholder slots whose name moved off kTexColdFrames ago: no draw list in flight names them
    // now. Release on the host (cold: no wait) and hand them back to the pool. Caller holds
    // g_texResidencyMx.
    void retireStreamedPlaceholders() {
        while (!g_texRetireSlots.empty() && g_frame - g_texRetireSlots.front().frame >= IPC::kTexColdFrames) {
            const std::uint32_t s = g_texRetireSlots.front().slot;
            g_texRetireSlots.pop_front();
            if (!g_slotName[s].empty()) {
                // Reallocated while it waited (only possible if something re-stamped its age and the
                // LRU took it): it belongs to another texture now — releasing it would blank that one.
                continue;
            }
            g_slotLastUsed[s] = 0;
            g_slotBytes[s] = 0;
            const IPC::TexUploadWire rel{ s | IPC::kTexUploadRelease | IPC::kTexUploadCold, 0u, 0u };
            stageTexUpload(rel, nullptr, 0u);   // a failed stage only delays the host's free
            g_texFreeSlots.push_back(s);
        }
    }

    // Every cross-frame copy of a slot lives in a SlotInfo (see g_texFreeSlots). Point the ones naming
    // a moved slot at its replacement directly — no epoch bump, so nothing else re-resolves.
    // g_keySlot is produce-owned, and both callers run in the produce.
    void retargetMovedSlots(const std::vector<std::pair<std::uint32_t, std::uint32_t>>& moved) {
        if (moved.empty()) { return; }
        for (auto& kv : g_keySlot) {
            SlotInfo& si = kv.second;
            for (const auto& mv : moved) {
                if (si.baseSlot  == mv.first) { si.baseSlot  = mv.second; }
                if (si.ovSlot    == mv.first) { si.ovSlot    = mv.second; }
                if (si.paramSlot == mv.first) { si.paramSlot = mv.second; }
            }
        }
    }

    void streamPendingTexturesAsync();

    void streamPendingTextures() {
        if (g_texStreamBudgetMB == 0 || !g_texVec) {
            return;
        }
        if (g_texStreamAsync && g_texStreamVec) {
            streamPendingTexturesAsync();
            return;
        }
        const std::size_t budget = (std::size_t)g_texStreamBudgetMB << 20;
        const std::size_t windowBytes = (std::size_t)IPC::kTexWindowBytes;
        std::size_t staged = 0;
        std::uint32_t shipped = 0, dropped = 0;
        // (old slot -> new slot) moves made below, applied to the per-key caches after the lock.
        static std::vector<std::pair<std::uint32_t, std::uint32_t>> s_moved;
        s_moved.clear();
        std::unique_lock<std::mutex> lk(g_texResidencyMx);
        retireStreamedPlaceholders();
        const std::uint32_t cap = IPC::kMaxTextures - IPC::kDlReserve;
        while (!g_texStreamQueue.empty() && (shipped == 0 || staged < budget)) {
            TexStreamJob job = std::move(g_texStreamQueue.front());
            g_texStreamQueue.pop_front();
            // Evicted or recycled while it waited: the slot means something else now.
            if (job.slot >= g_slotName.size() || g_slotName[job.slot] != job.name) {
                ++dropped;
                continue;
            }
            void* data = nullptr;
            unsigned size = 0;
            if (!BSA::loadFileBytes(job.name.c_str(), &data, &size, true) || !data || size == 0
                || sizeof(IPC::TexUploadWire) + size > windowBytes) {
                // The placeholder stays (or, for a param map, no PBR) — degraded, never broken.
                static std::uint32_t s_failLogged = 0;
                if (s_failLogged++ < 8) {
                    LOG::logline("!! [tex-stream] full file for %s did not load (%u bytes) — keeping %s",
                                 job.name.c_str(), size, job.data ? "no param map" : "the LOD placeholder");
                }
                if (data) { std::free(data); }
                ++dropped;
                continue;
            }
            // A fresh slot (free list, else grow) — never the LRU: recycling would evict a live
            // texture to make room for one that already has a drawable stand-in.
            std::uint32_t slot = 0;
            if (!g_texFreeSlots.empty()) {
                slot = g_texFreeSlots.back();
                g_texFreeSlots.pop_back();
            } else if (g_nextTexSlot < cap) {
                slot = g_nextTexSlot++;
            }
            const std::uint32_t cold = slot ? coldBit(slot) : 0u;
            if (!cold) {
                // No cold slot to spare (range full, or the session is younger than kTexColdFrames):
                // replace the placeholder in place, as before — the host then waits the frames in
                // flight (descWriteGuard), which is correct, only slower.
                if (slot) { g_texFreeSlots.push_back(slot); }
                slot = job.slot;
            }
            const IPC::TexUploadWire hdr{ slot | (job.data ? IPC::kTexUploadData : 0u) | cold, size, 0u };
            if (stageTexUpload(hdr, data, size)) {
                if (slot != job.slot) {
                    // The name moves now; draws built from here on use the new slot. The old one is
                    // pinned out of the LRU (age "in the future") until it retires.
                    g_texSlot[job.name] = slot;
                    g_slotName[slot] = job.name;
                    g_slotLastUsed[slot] = g_frame;
                    g_slotName[job.slot].clear();
                    g_slotLastUsed[job.slot] = 0xFFFFFFFFu;
                    g_texRetireSlots.push_back(TexSlotRetire{ job.slot, g_frame });
                    s_moved.emplace_back(job.slot, slot);
                    ++g_texFreshStreams;
                } else {
                    ++g_texInPlaceStreams;
                }
                g_slotBytes[slot] = size;
                staged += size;
                ++shipped;
                ++g_texStreamed;
                g_texStreamedBytes += size;
            } else if (slot != job.slot) {
                g_texFreeSlots.push_back(slot);   // not staged: the fresh slot goes back unused
            }
            std::free(data);
        }
        const std::size_t queued = g_texStreamQueue.size();
        lk.unlock();
        retargetMovedSlots(s_moved);
        if (shipped || dropped) {
            static std::uint32_t s_logThrottle = 0;
            if ((s_logThrottle++ % 30) == 0 || queued == 0) {
                LOG::logline("-- [tex-stream] shipped %u full textures (%.1f MB) this frame, %u dropped, %zu queued"
                             " | session: %llu into fresh slots, %llu in place",
                             shipped, (double)staged / (1024.0 * 1024.0), dropped, queued,
                             (unsigned long long)g_texFreshStreams, (unsigned long long)g_texInPlaceStreams);
            }
        }
    }

    // A reserved fresh slot that will not receive its full file goes back to the pool. Its host slot
    // is as the reservation found it (white or never used), unless `landed`: then the host holds the
    // full texture and it must be released first. Caller holds g_texResidencyMx.
    void unreserveStreamSlot(std::uint32_t s, bool landed) {
        g_slotLastUsed[s] = 0;   // was pinned out of the LRU; released slots rest at 0
        g_slotBytes[s] = 0;
        if (landed) {
            // Cold: a reserved slot was never named by any draw list.
            const IPC::TexUploadWire rel{ s | IPC::kTexUploadRelease | IPC::kTexUploadCold, 0u, 0u };
            stageTexUpload(rel, nullptr, 0u);
        }
        g_texFreeSlots.push_back(s);
    }

    // The async stream lane (see g_texStreamFlight). Produce context, like the sync path it replaces.
    //   1. POLL the batch in flight. Per entry: landed AND the name still on its placeholder -> move the
    //      name to the fresh slot (the old slot retires kTexColdFrames later, as before). Landed but
    //      the placeholder was evicted/recycled meanwhile -> release the fresh slot. Not landed -> the
    //      fresh slot goes back untouched and the placeholder stays (degraded, never broken).
    //   2. READ AHEAD: the queue's head reads on the I/O thread (TexIo), every call, in flight or not.
    //   3. STAGE the next batch only when none is in flight, from FINISHED reads only, in queue order
    //      (an unfinished head ends the batch): fresh slots RESERVED (never named until confirmed), then
    //      — outside the lock — one gather of header + read buffer into the stream window and a
    //      non-blocking kick. With MGE_TEX_IO_THREAD=0 the stage reads inline, as before.
    void streamPendingTexturesAsync() {
        MGE_ZoneScopedN("tex:stream");
        const std::size_t budget = (std::size_t)g_texStreamBudgetMB << 20;
        const std::size_t windowBytes = (std::size_t)IPC::kTexWindowBytes;
        static std::vector<std::pair<std::uint32_t, std::uint32_t>> s_moved;
        // The next batch: each entry is its header plus the read that holds its bytes, gathered as two
        // parts straight into the stream window — no intermediate [TexUploadWire][dds] vector (that copy,
        // with its zero-fill, was ~5 ms per 21 MB map). Index i matches g_texStreamFlight[i].
        struct BatchEntry { IPC::TexUploadWire hdr; TexIoHandle io; };
        static std::vector<BatchEntry> s_entries;
        s_moved.clear();
        s_entries.clear();
        std::uint32_t confirmed = 0, failed = 0, stale = 0, shipped = 0, dropped = 0, inPlace = 0;
        std::size_t staged = 0;
        bool notReady = false;

        std::uint32_t built = 0, failedMask = 0;
        const bool completed = g_client->streamUploadPoll(&built, &failedMask);

        std::unique_lock<std::mutex> lk(g_texResidencyMx);
        if (g_slotName.size() != IPC::kMaxTextures) {
            return;   // nothing has resolved yet, so nothing can be queued or in flight
        }
        retireStreamedPlaceholders();

        // --- 1. The receipt ---
        if (completed) {
            if (built == 0xFFFFFFFFu) {
                // Host not ready (first scene not built): nothing was touched. Unreserve and put the
                // jobs back at the FRONT, in order, for the next attempt.
                notReady = true;
                for (auto it = g_texStreamFlight.rbegin(); it != g_texStreamFlight.rend(); ++it) {
                    unreserveStreamSlot(it->newSlot, false);
                    g_texStreamQueue.push_front(TexStreamJob{ it->oldSlot, std::move(it->name), it->data });
                }
            } else {
                for (std::size_t i = 0; i < g_texStreamFlight.size(); ++i) {
                    TexStreamFlight& f = g_texStreamFlight[i];
                    const bool landed = (failedMask & (1u << i)) == 0u;
                    const bool live = f.oldSlot < g_slotName.size() && g_slotName[f.oldSlot] == f.name;
                    if (landed && live) {
                        g_texSlot[f.name] = f.newSlot;
                        g_slotName[f.newSlot] = f.name;
                        g_slotLastUsed[f.newSlot] = g_frame;
                        g_slotBytes[f.newSlot] = f.size;
                        g_slotName[f.oldSlot].clear();
                        g_slotLastUsed[f.oldSlot] = 0xFFFFFFFFu;   // pinned out of the LRU until it retires
                        g_texRetireSlots.push_back(TexSlotRetire{ f.oldSlot, g_frame });
                        s_moved.emplace_back(f.oldSlot, f.newSlot);
                        ++confirmed;
                        ++g_texFreshStreams;
                        ++g_texStreamed;
                        g_texStreamedBytes += f.size;
                    } else if (landed) {
                        unreserveStreamSlot(f.newSlot, true);
                        ++stale;
                    } else {
                        unreserveStreamSlot(f.newSlot, false);
                        ++failed;
                        static std::uint32_t s_failLogged = 0;
                        if (s_failLogged++ < 8) {
                            LOG::logline("!! [tex-stream] host did not build %s (slot %u) — keeping %s",
                                         f.name.c_str(), f.newSlot, f.data ? "no param map" : "the LOD placeholder");
                        }
                    }
                }
                g_texStreamConfirmed += confirmed;
                g_texStreamFailed += failed;
                g_texStreamStale += stale;
            }
            g_texStreamFlight.clear();
        }

        // --- Read-ahead (P1): the head of the queue reads on the I/O thread while a batch is in flight
        // or the window is busy, so the stage below finds finished reads instead of doing them. ---
        if (g_texIo.on()) {
            constexpr std::size_t kStreamReadAhead = 8;
            std::size_t n = 0;
            for (TexStreamJob& job : g_texStreamQueue) {
                if (n++ >= kStreamReadAhead) { break; }
                if (!job.io) { job.io = g_texIo.submit(job.name, TexIoRead::kFull); }
            }
        }

        // --- 2. The next batch ---
        if (!notReady && !g_client->streamUploadPending() && g_texStreamFlight.empty()) {
            MGE_ZoneScopedN("tex:stream stage");
            const std::uint32_t cap = IPC::kMaxTextures - IPC::kDlReserve;
            std::size_t windowUsed = 0;
            while (!g_texStreamQueue.empty() && g_texStreamFlight.size() < IPC::kMaxStreamBatch
                   && (shipped + inPlace == 0 || staged < budget)) {
                TexStreamJob& head = g_texStreamQueue.front();
                if (head.slot >= g_slotName.size() || g_slotName[head.slot] != head.name) {
                    g_texStreamQueue.pop_front();   // drops its read too (abandoned if not started)
                    ++dropped;   // evicted or recycled while it waited
                    continue;
                }
                if (head.io && !head.io->done.load(std::memory_order_acquire)) {
                    break;   // its read is still running: the batch goes with what is ready, in order
                }
                TexStreamJob job = std::move(head);
                g_texStreamQueue.pop_front();
                if (!job.io) {
                    // I/O thread off (MGE_TEX_IO_THREAD=0): read inline, as before P1.
                    MGE_ZoneScopedN("tex:stream read");
                    job.io = g_texIo.submit(job.name, TexIoRead::kFull);
                }
                const unsigned size = job.io->r.size;
                if (!job.io->r.found || sizeof(IPC::TexUploadWire) + size > windowBytes) {
                    static std::uint32_t s_loadLogged = 0;
                    if (s_loadLogged++ < 8) {
                        LOG::logline("!! [tex-stream] full file for %s did not load (%u bytes) — keeping %s",
                                     job.name.c_str(), size, job.data ? "no param map" : "the LOD placeholder");
                    }
                    ++dropped;
                    continue;
                }
                const void* data = job.io->r.data;
                const std::size_t entryBytes = sizeof(IPC::TexUploadWire) + size;
                if (windowUsed + entryBytes > windowBytes) {
                    // The window is full: this one (and its finished read) leads the next batch.
                    g_texStreamQueue.push_front(std::move(job));
                    break;
                }
                // A fresh COLD slot with no unshipped sync record (see g_slotStagedBatch), else grow.
                std::uint32_t slot = 0;
                for (std::size_t k = g_texFreeSlots.size(); k-- > 0;) {
                    const std::uint32_t s = g_texFreeSlots[k];
                    if (slotIsCold(s) && g_slotStagedBatch[s] <= g_texShippedSerial) {
                        slot = s;
                        g_texFreeSlots.erase(g_texFreeSlots.begin() + (std::ptrdiff_t)k);
                        break;
                    }
                }
                if (slot == 0 && g_nextTexSlot < cap && slotIsCold(g_nextTexSlot)) {
                    slot = g_nextTexSlot++;
                }
                if (slot == 0) {
                    // No fresh slot to spare: replace the placeholder in place on the SYNC path, as
                    // before — never out of band (see g_texStreamFlight).
                    const IPC::TexUploadWire hdr{ job.slot | (job.data ? IPC::kTexUploadData : 0u)
                                                  | coldBit(job.slot), size, 0u };
                    if (stageTexUpload(hdr, data, size)) {
                        g_slotBytes[job.slot] = size;
                        staged += size;   // the same per-frame budget as the async entries
                        ++inPlace;
                        ++g_texInPlaceStreams;
                        ++g_texStreamed;
                        g_texStreamedBytes += size;
                    }
                    continue;   // the read's bytes go with its handle
                }
                const IPC::TexUploadWire hdr{ slot | (job.data ? IPC::kTexUploadData : 0u)
                                              | IPC::kTexUploadCold, size, 0u };
                s_entries.push_back(BatchEntry{ hdr, job.io });
                g_slotLastUsed[slot] = 0xFFFFFFFFu;   // RESERVED: no name, out of the free list and the LRU
                g_texStreamFlight.push_back(TexStreamFlight{ job.slot, slot, std::move(job.name), size, job.data });
                windowUsed += entryBytes;
                staged += size;
                ++shipped;
            }
        }
        const std::size_t queued = g_texStreamQueue.size();
        lk.unlock();
        retargetMovedSlots(s_moved);

        // --- Kick, outside the lock (the gather is the one copy; the RPC does not block) ---
        bool kicked = false;
        if (!s_entries.empty()) {
            static std::vector<const void*>   parts;
            static std::vector<std::uint32_t> partSizes;
            parts.clear();
            partSizes.clear();
            std::size_t bytes = 0;
            for (const auto& e : s_entries) {
                parts.push_back(&e.hdr);
                partSizes.push_back((std::uint32_t)sizeof(e.hdr));
                parts.push_back(e.io->r.data);
                partSizes.push_back(e.io->r.size);
                bytes += sizeof(e.hdr) + e.io->r.size;
            }
            MGE_ZoneScopedN("Forge texStream kick");
            // The entry count is what the host walks: one per texture, not per gathered part.
            kicked = g_texStreamVec->assign_gather(parts.data(), partSizes.data(), (std::uint32_t)parts.size())
                  && g_client->streamUploadKick(g_texStreamVec->id(), (std::uint32_t)s_entries.size(), (std::uint32_t)bytes);
            if (kicked) {
                ++g_texStreamBatches;
            } else {
                // Could not hand it over: undo the reservations and put the jobs back, in order, with
                // their finished reads (s_entries[i] is g_texStreamFlight[i]).
                std::lock_guard<std::mutex> lk2(g_texResidencyMx);
                for (std::size_t i = g_texStreamFlight.size(); i-- > 0;) {
                    TexStreamFlight& f = g_texStreamFlight[i];
                    unreserveStreamSlot(f.newSlot, false);
                    g_texStreamQueue.push_front(TexStreamJob{ f.oldSlot, std::move(f.name), f.data,
                                                              i < s_entries.size() ? s_entries[i].io : TexIoHandle{} });
                }
                g_texStreamFlight.clear();
                static std::uint32_t s_kickLogged = 0;
                if (s_kickLogged++ < 8) {
                    LOG::logline("!! [tex-stream] stream kick failed (%zu entries, %zu bytes) — requeued",
                                 s_entries.size(), bytes);
                }
            }
            s_entries.clear();   // the window holds the bytes now (or the requeued jobs do): free the reads
        }
        if (confirmed || failed || stale || shipped || dropped || inPlace) {
            static std::uint32_t s_logThrottle = 0;
            if ((s_logThrottle++ % 30) == 0 || (queued == 0 && !kicked) || failed || stale) {
                LOG::logline("-- [tex-stream] async: confirmed %u, failed %u, stale %u | kicked %u (%.1f MB), %u in place,"
                             " %u dropped, %zu queued | session: %llu batches, %llu confirmed, %llu failed, %llu stale"
                             " | io %s: %llu reads %.0f MB %.0f ms, %llu abandoned, held peak %.0f MB",
                             confirmed, failed, stale, kicked ? shipped : 0u, (double)staged / (1024.0 * 1024.0),
                             inPlace, dropped, queued,
                             (unsigned long long)g_texStreamBatches, (unsigned long long)g_texStreamConfirmed,
                             (unsigned long long)g_texStreamFailed, (unsigned long long)g_texStreamStale,
                             g_texIo.on() ? "thread" : "inline", (unsigned long long)g_texIo.reads.load(),
                             (double)g_texIo.readBytes.load() / (1024.0 * 1024.0), g_texIo.readMs.load(),
                             (unsigned long long)g_texIo.abandoned.load(),
                             (double)g_texIo.heldPeak.load() / (1024.0 * 1024.0));
            }
        }
    }

    // Resolve the active grid's textures ahead of first sight (see g_gridTexQueue). Called in the
    // BUILD, beside streamPendingTextures and before it, so its budget is measured against only this
    // frame's own first sights: a cell-load frame that already staged a lot prefetches nothing.
    constexpr std::uint32_t kGridPrefetchRescanFrames = 30;   // re-read the cache while it still grows
    constexpr std::uint64_t kGridPrefetchFreshFrames  = 600;  // entries visited this recently only
    constexpr std::uint32_t kGridPrefetchSlotHeadroom = 256;  // never prefetch into an LRU recycle
    constexpr double        kGridPrefetchMaxMs        = 2.0;  // disk reads, per frame
    void prefetchGridTextures() {
        if (g_gridPrefetchMB == 0 || !g_texVec) {
            return;
        }
        // --- Scan: when the grid moved, or while the post-load walk is still filling the cache. ---
        void* dh = MGE::SceneGraph::getDataHandler();
        const void* interiorCell = dh ? MGE::DataHandlerView::currentInteriorCell(dh) : nullptr;
        const std::int32_t gx = dh ? MGE::DataHandlerView::centralGridX(dh) : 0;
        const std::int32_t gy = dh ? MGE::DataHandlerView::centralGridY(dh) : 0;
        static const void*   s_cell = nullptr;
        static std::int32_t  s_gx = 0, s_gy = 0;
        static std::uint32_t s_epoch = 0xFFFFFFFFu;
        static std::size_t   s_cacheSize = 0;
        static std::uint32_t s_scanFrame = 0;
        const auto& cacheMap = MGE::GeometryCache::cache();
        const bool gridMoved = interiorCell != s_cell || gx != s_gx || gy != s_gy || g_cellEpoch != s_epoch;
        const bool cacheGrew = cacheMap.size() != s_cacheSize && g_frame - s_scanFrame >= kGridPrefetchRescanFrames;
        if (dh && (gridMoved || cacheGrew)) {
            MGE_ZoneScopedN("tex:prefetch scan");
            // A CROSSING (user, 2026-09-30: "we can predict and load textures early in a relaxed way
            // during cell crossing"). An exterior grid slide loads a new row of cells, but under the
            // seam the cache fills only from what the camera sees, so the new row's names would reach
            // this prefetch one glance at a time. Arm the post-load residency walk — budgeted (256
            // captures a frame) and self-closing, exactly as a load does — so the whole new row is
            // captured over the next frames and the rescans below (cacheGrew) queue its textures at
            // the prefetch budget. A purge (same test with the epoch moved) arms it on its own.
            const bool crossing = gridMoved && s_epoch == g_cellEpoch && !interiorCell && !s_cell
                               && s_epoch != 0xFFFFFFFFu;
            if (crossing) {
                MGE::GeometryCache::armPostLoadWalk();
            }
            s_cell = interiorCell; s_gx = gx; s_gy = gy; s_epoch = g_cellEpoch;
            s_cacheSize = cacheMap.size();
            s_scanFrame = g_frame;
            // Every distinct name pointer in the grid, once. Names are engine pointers, and a string is
            // only READ through one whose entry was visited recently, so the NiSourceTexture behind it
            // is alive (the post-load walk stamps the whole grid; the draw build stamps what it sees).
            // Older entries — a cell behind you that you have not looked at since — reuse the name
            // their pointer normalised to when it was fresh (s_ptrName), so they stay pinned across a
            // grid move without the pointer being dereferenced. A recycled pointer in that map can at
            // worst pin or prefetch one wrong texture; the map is dropped with every cell epoch.
            const std::uint64_t cacheFrame = MGE::GeometryCache::currentFrame();
            static std::unordered_map<const char*, std::string> s_ptrName;
            static std::uint32_t s_ptrEpoch = 0xFFFFFFFFu;
            if (s_ptrEpoch != g_cellEpoch) { s_ptrName.clear(); s_ptrEpoch = g_cellEpoch; }
            constexpr std::uint8_t kFresh = 1, kBase = 2;   // kBase: gets a param-companion probe
            static std::unordered_map<const char*, std::uint8_t> s_ptrs;
            s_ptrs.clear();
            for (const auto& kv : cacheMap) {
                const MGE::GeometryCache::CachedGeometry& e = kv.second;
                if (e.isSky || e.texAnimated) { continue; }
                if (!interiorCell) {
                    // Same cell derivation as stampSlotCell / the eviction's active-grid hold.
                    const std::int32_t cx = (std::int32_t)std::floor(e.worldTransformD3D[12] / 8192.0f);
                    const std::int32_t cy = (std::int32_t)std::floor(e.worldTransformD3D[13] / 8192.0f);
                    if (std::abs(cx - gx) > 1 || std::abs(cy - gy) > 1) { continue; }
                }
                const std::uint8_t fresh = cacheFrame - e.lastFrame <= kGridPrefetchFreshFrames ? kFresh : 0;
                if (e.textureName) {
                    s_ptrs[e.textureName] |= fresh | (!e.isSkinned && !e.isFP ? kBase : 0);
                }
                if (e.isLandscape && e.d3dOverlay && e.overlayTextureName) { s_ptrs[e.overlayTextureName] |= fresh; }
                if (e.d3dDark   && e.darkTextureName)   { s_ptrs[e.darkTextureName]   |= fresh; }
                if (e.d3dDetail && e.detailTextureName) { s_ptrs[e.detailTextureName] |= fresh; }
                if (e.d3dGlow   && e.glowTextureName)   { s_ptrs[e.glowTextureName]   |= fresh; }
            }
            std::vector<GridTexJob> jobs;
            std::unordered_set<std::string> names;
            for (const auto& pf : s_ptrs) {
                std::string n;
                if (pf.second & kFresh) {
                    n = normalizeTextureName(pf.first);
                    s_ptrName[pf.first] = n;
                } else {
                    const auto known = s_ptrName.find(pf.first);
                    if (known == s_ptrName.end()) { continue; }   // never seen fresh: not read
                    n = known->second;
                }
                if (n.empty()) { continue; }
                if (pf.second & kBase) {
                    std::string stem = n;
                    const std::size_t dot = stem.find_last_of('.');
                    const std::size_t sep = stem.find_last_of('\\');
                    if (dot != std::string::npos && (sep == std::string::npos || dot > sep)) { stem.erase(dot); }
                    if (!stem.empty() && names.insert(stem + "_paramh.dds").second) {
                        names.insert(stem + "_paramh_np.dds");
                        jobs.push_back(GridTexJob{ stem, true });
                    }
                }
                if (names.insert(n).second) { jobs.push_back(GridTexJob{ std::move(n), false }); }
            }
            std::size_t queued = 0, resident = 0;
            {
                std::lock_guard<std::mutex> lk(g_texResidencyMx);
                if (gridMoved) {
                    g_gridTexPins.swap(names);   // the old grid's textures age out normally
                } else {
                    g_gridTexPins.insert(names.begin(), names.end());
                }
                // `jobs` is the whole current set, so it REPLACES the queue: appending re-queued every
                // name the last scan had queued and not yet reached (444 duplicates on Dragonstar).
                g_gridTexQueue.clear();
                for (GridTexJob& j : jobs) {
                    bool known;
                    if (j.param) {
                        // Settled once _paramh resolved, or missed AND _paramh_np has been tried.
                        const auto ph = g_texSlot.find(j.name + "_paramh.dds");
                        known = ph != g_texSlot.end() && (ph->second != 0 || g_texSlot.count(j.name + "_paramh_np.dds"));
                    } else {
                        known = g_texSlot.count(j.name) != 0;
                    }
                    if (known) { ++resident; continue; }
                    g_gridTexQueue.push_back(std::move(j));
                    ++queued;
                }
            }
            if (queued) {
                LOG::logline("-- [tex-prefetch] grid (%d,%d)%s: %zu textures, %zu resident, %zu queued"
                             " (%zu entries)", gx, gy, interiorCell ? " interior" : "",
                             s_ptrs.size(), resident, queued, cacheMap.size());
            }
        }
        if (g_gridTexQueue.empty()) {
            return;
        }
        // --- Drain: resolve until this frame has staged the budget, or spent its time cap. ---
        MGE_ZoneScopedN("tex:prefetch drain");
        const std::size_t budget = (std::size_t)g_gridPrefetchMB << 20;
        const std::uint32_t cap = IPC::kMaxTextures - IPC::kDlReserve;
        // P1: with the I/O thread on, the reads run AHEAD of the drain — the head of the queue is
        // submitted every frame, and the drain only resolves jobs whose reads have finished, in queue
        // order. The time cap then bounds staging copies, not disk reads. The placeholder decision a
        // read is made under is this frame's; a read made under the other one (a load's sync frame in
        // between) is simply redone inline by the resolve.
        bool deferAllowed;
        {
            std::lock_guard<std::mutex> lk(g_texResidencyMx);
            deferAllowed = g_texStreamBudgetMB != 0 && g_frame != g_texSyncFrame;
        }
        if (g_texIo.on()) {
            constexpr std::size_t kPrefetchReadAhead = 16;
            std::size_t n = 0;
            for (GridTexJob& j : g_gridTexQueue) {
                if (n++ >= kPrefetchReadAhead) { break; }
                if (j.io) { continue; }
                if (j.param) {
                    j.io  = g_texIo.submit(j.name + "_paramh.dds",    TexIoRead::kFirstSight, true, deferAllowed);
                    j.io2 = g_texIo.submit(j.name + "_paramh_np.dds", TexIoRead::kFirstSight, true, deferAllowed);
                } else {
                    j.io  = g_texIo.submit(j.name, TexIoRead::kFirstSight, false, deferAllowed);
                }
            }
        }
        auto ready = [](const GridTexJob& j) {
            return (!j.io  || j.io->done.load(std::memory_order_acquire))
                && (!j.io2 || j.io2->done.load(std::memory_order_acquire));
        };
        const double t0 = nowMs();
        std::uint32_t done = 0;
        std::size_t bytes = 0;
        while (!g_gridTexQueue.empty() && nowMs() - t0 < kGridPrefetchMaxMs
               && (!g_texIo.on() || ready(g_gridTexQueue.front()))) {
            std::size_t before;
            {
                std::lock_guard<std::mutex> lk(g_texResidencyMx);
                if (g_texPendingBytes >= budget) { break; }
                // A prefetch must never push a slot out: with the range full the resolve recycles the
                // LRU — a live texture, and an epoch bump that re-resolves every cached slot.
                const std::uint32_t freeSlots = (std::uint32_t)g_texFreeSlots.size() + (cap - std::min(g_nextTexSlot, cap));
                if (freeSlots <= kGridPrefetchSlotHeadroom) {
                    LOG::logline("!! [tex-prefetch] only %u free slots — dropping %zu queued prefetches",
                                 freeSlots, g_gridTexQueue.size());
                    g_gridTexQueue.clear();
                    break;
                }
                before = g_texPendingBytes;
            }
            const GridTexJob j = std::move(g_gridTexQueue.front());
            g_gridTexQueue.pop_front();
            if (j.param) {
                if (resolveTextureSlotEx((j.name + "_paramh.dds").c_str(), true, true, j.io.get()) == 0) {
                    resolveTextureSlotEx((j.name + "_paramh_np.dds").c_str(), true, true, j.io2.get());
                }
            } else {
                resolveTextureSlotEx(j.name.c_str(), false, false, j.io.get());   // a miss logs, as the draw's would
            }
            {
                std::lock_guard<std::mutex> lk(g_texResidencyMx);
                if (g_texPendingBytes > before) { bytes += g_texPendingBytes - before; }
            }
            ++done;
        }
        g_gridPrefetched += done;
        g_gridPrefetchedBytes += bytes;
        if (done && g_gridTexQueue.empty()) {
            LOG::logline("-- [tex-prefetch] grid done | session: %llu prefetched, %.1f MB staged",
                         (unsigned long long)g_gridPrefetched, (double)g_gridPrefetchedBytes / (1024.0 * 1024.0));
        }
    }

    // Release the texture slots no live part names and nothing has sampled for g_texEvictAgeFrames
    // (the rule, and why it is safe, is at g_texFreeSlots). Runs in the produce context between
    // flushGeometry and flushTextures: g_keySlot is produce-owned, and the releases staged here ride
    // THIS frame's texture flush, which the host takes after this frame's geometry releases.
    constexpr std::uint32_t kTexEvictPeriodFrames = 30;
    constexpr std::uint32_t kMaxTexEvictPerPass   = 128;

    // A part may still NAME a slot just released, in a SlotInfo whose fast path would hand it back
    // unchecked — and the slot is about to be reused for another texture. Forget the cached value, so
    // that part re-resolves by name if it ever draws again. (No epoch bump: only the holders of a
    // released slot re-resolve, not every cached slot in the scene.) g_keySlot is produce-owned.
    void forgetReleasedSlots(const std::vector<std::uint8_t>& gone) {
        for (auto& kv : g_keySlot) {
            SlotInfo& si = kv.second;
            if (si.baseSlot  < IPC::kMaxTextures && gone[si.baseSlot])  { si.baseNamePtr  = nullptr; si.baseSlot  = 0; }
            if (si.ovSlot    < IPC::kMaxTextures && gone[si.ovSlot])    { si.ovNamePtr    = nullptr; si.ovSlot    = 0; }
            if (si.paramSlot < IPC::kMaxTextures && gone[si.paramSlot]) { si.paramNamePtr = nullptr; si.paramSlot = 0; }
        }
    }

    // THE HOST MIRRORS MORROWIND (see g_texNameLive in scenegraph_geometry_cache.cpp). Release every
    // texture Morrowind itself let go of since the last produce — and its _paramh companion, which
    // exists only because its base does — so the host holds what Morrowind holds, at full resolution,
    // instead of everything seen in the last 600 frames. It overrides the grid pin: a name Morrowind
    // has freed is not in any grid it will draw. evictStaleTextures' age rule stays as the backstop
    // for names the reverse map never saw. Same produce context and same release protocol (cold bit,
    // free list, SlotInfo holders forgotten) as evictStaleTextures.
    std::vector<const char*> g_mwDroppedNames;   // produce-owned: drained, not yet released
    std::uint64_t            g_texMirrorReleases = 0;
    std::uint64_t            g_texMirrorBytes = 0;
    void releaseMorrowindDroppedTextures() {
        static std::vector<const char*> s_new;
        MGE::GeometryCache::takeDroppedTextureNames(s_new);
        g_mwDroppedNames.insert(g_mwDroppedNames.end(), s_new.begin(), s_new.end());
        if (g_mwDroppedNames.empty()) {
            return;
        }
        // NO wait on the geometry release backlog, unlike evictStaleTextures. That wait keeps a departed
        // key's host shadow-caster record from sampling a slot after its texture left — but the host
        // puts DEFAULT WHITE back in a released slot (and retires the old texture only once no frame in
        // flight can reach it), so the worst case is a gone object's ghost shadow drawing white-cut for
        // the frames until its own release ships, a ghost that exists either way. What the wait cost
        // was measured by AutoZip: fast cell changes kept the backlog non-empty (15403 releases queued
        // at 64 per flush), so NOTHING was released and the host held 4-5 GB of textures.
        // Normalised candidates, decided BEFORE taking the residency lock (textureNameLive takes the
        // name-map lock; the two are never nested). A name Morrowind re-created before we got here is
        // live again and stays.
        static std::vector<std::string> s_cands;
        s_cands.clear();
        for (const char* p : g_mwDroppedNames) {
            if (MGE::GeometryCache::textureNameLive(p)) { continue; }
            std::string n = normalizeTextureName(p);
            if (n.empty()) { continue; }
            std::string stem = n;
            const std::size_t dot = stem.find_last_of('.');
            const std::size_t sep = stem.find_last_of('\\');
            if (dot != std::string::npos && (sep == std::string::npos || dot > sep)) { stem.erase(dot); }
            s_cands.push_back(std::move(n));
            if (!stem.empty()) {
                s_cands.push_back(stem + "_paramh.dds");
                s_cands.push_back(stem + "_paramh_np.dds");
            }
        }
        g_mwDroppedNames.clear();
        static std::vector<std::uint8_t> s_gone;
        s_gone.assign(IPC::kMaxTextures, 0u);
        std::uint32_t released = 0;
        std::uint64_t bytes = 0;
        {
            std::lock_guard<std::mutex> lk(g_texResidencyMx);
            if (g_slotName.size() != IPC::kMaxTextures) {
                return;
            }
            for (const std::string& n : s_cands) {
                g_gridTexPins.erase(n);
                const auto it = g_texSlot.find(n);
                if (it == g_texSlot.end()) { continue; }
                const std::uint32_t s = it->second;
                if (s == 0) { g_texSlot.erase(it); continue; }   // a cached miss: forget it too
                if (IPC::isFlipSlot(s) || s >= IPC::kMaxTextures || g_slotName[s] != n) { continue; }
                const std::uint32_t cold = coldBit(s);
                g_texSlot.erase(it);
                g_slotName[s].clear();
                // g_slotLastUsed KEPT (unlike evictStaleTextures, whose slots are 600+ frames old):
                // Morrowind may have dropped a texture drawn last frame, and the slot's reuse must
                // still read hot so the host waits the frames in flight before rewriting it.
                bytes += g_slotBytes[s];
                g_slotBytes[s] = 0;
                g_texFreeSlots.push_back(s);
                const IPC::TexUploadWire rel{ s | IPC::kTexUploadRelease | cold, 0u, 0u };
                stageTexUpload(rel, nullptr, 0u);   // a failed stage only delays the host's free
                s_gone[s] = 1u;
                ++released;
            }
            g_texMirrorReleases += released;
            g_texMirrorBytes += bytes;
        }
        if (released) {
            forgetReleasedSlots(s_gone);
            static std::uint32_t s_logThrottle = 0;
            if ((s_logThrottle++ % 16) == 0) {
                LOG::logline("-- [tex-mirror] Morrowind dropped them: released %u slots (%.1f MB) | session"
                             " %llu slots, %.0f MB", released, (double)bytes / (1024.0 * 1024.0),
                             (unsigned long long)g_texMirrorReleases, (double)g_texMirrorBytes / (1024.0 * 1024.0));
            }
        }
    }

    // BOOKKEEPING AT A QUIET MOMENT (user, 2026-09-30: "player is expected to enter an interior or
    // sleep or pause the game etc. best moment for bookkeeping"). Releasing textures one by one frees
    // their bytes but not the VRAM: D3D12MA hands out 64 MB heap blocks and returns one to the driver
    // only when it is EMPTY, and streamed textures share blocks with each other and with long-lived
    // resources. So the host's local VRAM ratcheted to its high-water mark — AutoZip, 500 cell
    // changes: allocator slack up to 1.77 GB, local VRAM ending at its max while textures had fallen.
    // A load is the moment nobody is looking and everything is being re-resolved anyway (the geometry
    // cache was just purged): release EVERY streamed slot, so whole blocks empty and go back, and let
    // the post-load window + grid prefetch refill what the new cell uses into fresh blocks.
    // Flip-book arrays stay (session-resident, one descriptor each). Produce context, like the purge.
    // MGE_TEX_BOOKKEEP=0 turns it off (A/B).
    //
    // KEEP THE LAST LOCATION (user: "I like fast load of last cell, interior<->exterior"). Morrowind
    // itself keeps the exterior loaded across an interior visit (the mirror releases ~nothing on the
    // way in), so a release-all here was the ONLY thing throwing it away — and the way back out then
    // re-streamed it through placeholders, visibly low-res for several frames. Kept: every slot used
    // within g_texEvictAgeFrames (the place being LEFT) and every name kept at the previous load (the
    // place before it — for an interior hop, the exterior). Two locations, bounded; the next load
    // drops whichever of them is no longer one of the last two.
    std::unordered_set<std::string> g_texKeptLastLoad;   // produce-owned
    void texBookkeepingRelease(const char* why) {
        static int s_on = -1;
        if (s_on < 0) {
            char e[16] = {};
            s_on = (GetEnvironmentVariableA("MGE_TEX_BOOKKEEP", e, sizeof(e)) > 0 && e[0] == '0') ? 0 : 1;
            LOG::logline(">> [tex-bookkeep] release-all at loads %s (MGE_TEX_BOOKKEEP=0 turns it off)",
                         s_on ? "ON" : "OFF");
        }
        if (!s_on || !g_texVec) {
            return;
        }
        std::uint32_t released = 0, kept = 0;
        std::uint64_t bytes = 0, keptBytes = 0;
        std::unordered_set<std::string> keepNow;   // the place being left -> next load's "previous"
        {
            std::lock_guard<std::mutex> lk(g_texResidencyMx);
            if (g_slotName.size() != IPC::kMaxTextures) {
                return;
            }
            const std::uint32_t cap = IPC::kMaxTextures - IPC::kDlReserve;
            const std::uint32_t hotAge = g_texEvictAgeFrames ? g_texEvictAgeFrames : 600u;
            for (std::uint32_t s = 1; s < g_nextTexSlot && s < cap; ++s) {
                if (g_slotName[s].empty()) { continue; }   // free, or a placeholder awaiting retire
                const bool hot = g_frame - g_slotLastUsed[s] < hotAge;
                if (hot || g_texKeptLastLoad.count(g_slotName[s])) {
                    if (hot) { keepNow.insert(g_slotName[s]); }
                    ++kept;
                    keptBytes += g_slotBytes[s];
                    continue;
                }
                const std::uint32_t cold = coldBit(s);
                g_texSlot.erase(g_slotName[s]);
                g_slotName[s].clear();
                // g_slotLastUsed KEPT: this slot may have been drawn last frame, and its reuse must
                // still read hot (host waits the frames in flight) — zeroing it would make it look cold.
                bytes += g_slotBytes[s];
                g_slotBytes[s] = 0;
                g_texFreeSlots.push_back(s);
                const IPC::TexUploadWire rel{ s | IPC::kTexUploadRelease | cold, 0u, 0u };
                stageTexUpload(rel, nullptr, 0u);
                ++released;
            }
            // Cached misses go too (cheap to re-probe; the new cell may have files the old one
            // lacked). Released names were erased above; kept and flip-book names stay.
            for (auto it = g_texSlot.begin(); it != g_texSlot.end();) {
                if (it->second == 0) { it = g_texSlot.erase(it); } else { ++it; }
            }
            ++g_texEpoch;   // every cached slot re-resolves by name (kept ones find themselves again)
        }
        g_texKeptLastLoad.swap(keepNow);
        g_capTexMemo.clear();
        g_gridTexQueue.clear();
        LOG::logline("-- [tex-bookkeep] %s: released %u streamed slots (%.0f MB), kept %u (%.0f MB) — the"
                     " place left and the one before it", why, released, (double)bytes / (1024.0 * 1024.0),
                     kept, (double)keptBytes / (1024.0 * 1024.0));
    }

    // FIRST FRAME AFTER A LOAD: SYNCHRONOUS (user: "first frame being late is acceptable and it hides
    // the issues"). First-sight streaming shows a DL placeholder and ships the full file over the next
    // frames — right mid-play, wrong on the first frame of a new place, where it reads as several
    // frames of low-res on arrival. On the load's first produced build, first sights take the full
    // file at once: that one frame is late (hidden by the load), every later frame is sharp.
    // (g_texSyncFrame is declared with the streaming globals; set in checkCellEpochAndPurge.)

    void evictStaleTextures() {
        static bool s_envRead = false;
        if (!s_envRead) {
            s_envRead = true;
            char e[32] = {};
            if (GetEnvironmentVariableA("MGE_TEX_EVICT_FRAMES", e, sizeof(e)) > 0) {
                g_texEvictAgeFrames = (std::uint32_t)std::strtoul(e, nullptr, 10);
            }
            LOG::logline(">> [tex-evict] stale-texture eviction %s (age %u frames; MGE_TEX_EVICT_FRAMES"
                         " overrides, 0 = off)", g_texEvictAgeFrames ? "ON" : "OFF", g_texEvictAgeFrames);
        }
        if (g_texEvictAgeFrames == 0) {
            return;
        }
        static std::uint32_t s_lastPass = 0;
        if (g_frame - s_lastPass < kTexEvictPeriodFrames) {
            return;
        }
        // A geometry release still queued (or unshipped) means a departed key's host shadow-caster
        // record may still carry its base slot. Its SlotInfo is already gone, so the reference scan
        // below cannot see it: wait until the host has dropped the record.
        if (!g_pendingReleaseSlots.empty() || !g_pendingBlob.empty()) {
            return;
        }
        s_lastPass = g_frame;

        // WHO PINS A TEXTURE: a live part inside Morrowind's ACTIVE grid (the 3x3 around the central
        // cell). The geometry cache keeps a 5x5 (cellGridGone, radius 2) as hysteresis for its own
        // meshes; letting that outer ring pin textures held 50-66 high-res resident against 5-22
        // actually sampled (gridwalk, 2026-09-19). Indoors, or with no grid to read, every live part
        // pins. An outer-ring part is at least a cell from the player, beyond the sun cascades, so its
        // host shadow-caster record cannot be sampling the texture it loses.
        void* dh = MGE::SceneGraph::getDataHandler();
        const bool interior = !dh || MGE::DataHandlerView::currentInteriorCell(dh) != nullptr;
        const std::int32_t gx = dh ? MGE::DataHandlerView::centralGridX(dh) : 0;
        const std::int32_t gy = dh ? MGE::DataHandlerView::centralGridY(dh) : 0;
        constexpr std::int32_t kActiveGridRadius = 1;
        static std::vector<std::uint8_t> s_ref;
        s_ref.assign(IPC::kMaxTextures, 0u);
        for (const auto& kv : g_keySlot) {
            const SlotInfo& si = kv.second;
            const bool pins = interior || !si.cellKnown
                || (std::abs(si.cellX - gx) <= kActiveGridRadius && std::abs(si.cellY - gy) <= kActiveGridRadius);
            if (!pins) { continue; }
            // Flip slots are >= 0x8000 and never evicted, so the bound drops them too.
            const std::uint32_t held[3] = { si.baseSlot, si.ovSlot, si.paramSlot };
            for (std::uint32_t s : held) {
                if (s != 0 && s < IPC::kMaxTextures) { s_ref[s] = 1u; }
            }
        }

        static std::vector<std::uint8_t> s_gone;
        s_gone.assign(IPC::kMaxTextures, 0u);
        std::uint32_t released = 0, heldStale = 0;
        std::uint64_t bytes = 0;
        {
            std::lock_guard<std::mutex> lk(g_texResidencyMx);
            if (g_slotName.size() != IPC::kMaxTextures) {
                return;
            }
            const std::uint32_t cap = IPC::kMaxTextures - IPC::kDlReserve;
            for (std::uint32_t s = 1; s < g_nextTexSlot && s < cap; ++s) {
                if (g_slotName[s].empty()) { continue; }   // already free
                if (g_frame - g_slotLastUsed[s] < g_texEvictAgeFrames) { continue; }
                if (s_ref[s]) { ++heldStale; continue; }     // a live part still names it
                if (g_gridTexPins.count(g_slotName[s])) { ++heldStale; continue; }   // grid prefetch set
                if (released >= kMaxTexEvictPerPass) { continue; }   // next pass
                const std::uint32_t cold = coldBit(s);   // before the age resets below
                g_texSlot.erase(g_slotName[s]);
                g_slotName[s].clear();
                g_slotLastUsed[s] = 0;
                bytes += g_slotBytes[s];
                g_slotBytes[s] = 0;
                g_texFreeSlots.push_back(s);
                const IPC::TexUploadWire rel{ s | IPC::kTexUploadRelease | cold, 0u, 0u };
                stageTexUpload(rel, nullptr, 0u);   // a failed stage only delays the host's free
                s_gone[s] = 1u;
                ++released;
            }
            g_texEvictions += released;
            g_texEvictedBytes += bytes;
        }
        if (released) { forgetReleasedSlots(s_gone); }
        if (released) {
            LOG::logline("-- [tex-evict] released %u slots (%.1f MB of DDS); %u stale slots held by"
                         " live parts", released, (double)bytes / (1024.0 * 1024.0), heldStale);
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
            // window= the share appended by the post-load window walk (off-screen captures) since
            // the last flush; the rest is this frame's draws + releases (tasks/forge-crossing-frame.md P0).
            LOG::logline("-- [geomflush] %.2fms parts=%u bytes=%uKB chunks=%u drain=%.2f assign=%.2f rpc=%.2f window=%u/%lluKB",
                         flushMs, shippedParts, total >> 10, chunkCount, drainMs, assignMs, rpcMs,
                         g_windowParts, (unsigned long long)(g_windowBytes >> 10));
        }
        g_windowParts = 0;
        g_windowBytes = 0;
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
            g_kick.earlyFenceValue  = 0;
            g_kick.earlyRtSlot      = 0;
            g_kick.earlyOk          = g_client->renderSceneFinish(&g_kick.earlyHostMs,
                                                                  &g_kick.earlyHostTimings,
                                                                  &g_kick.earlyFenceValue,
                                                                  &g_kick.earlyRtSlot);
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
    // Where a part was last emitted, for evictStaleTextures' active-grid hold. Same derivation as the
    // geometry cache's cellGridGone (world translation / 8192), so both agree on a part's cell.
    inline void stampSlotCell(SlotInfo& si, const MGE::GeometryCache::CachedGeometry& e) {
        si.cellX = (std::int32_t)std::floor(e.worldTransformD3D[12] / 8192.0f);
        si.cellY = (std::int32_t)std::floor(e.worldTransformD3D[13] / 8192.0f);
        si.cellKnown = true;
    }

    std::uint32_t resolveCachedSlot(const char* name, const char*& namePtr,
                                    std::uint32_t& slotVal, std::uint32_t& slotEpoch) {
        if (name == namePtr && slotEpoch == g_texEpoch) {
            // isFlipSlot is NOT optional here. A flip-book frame resolves to an ENCODED array slot
            // (bucket<<16|layer), not an index into the LRU range, so subscripting g_slotLastUsed
            // with it runs ~0x8000 elements (~128 KB) past the end. resolveTextureSlot's memoized
            // return guards exactly this (:1452); this fast path — which is the one that actually
            // runs, every frame, once a name is cached — did not. The stray store lands in whatever
            // the allocator put after the vector, so the crash surfaces LATER and ELSEWHERE: the
            // reported one was an access violation inside g_texSlot's own node insert, on the
            // produce worker, with a garbage list pointer. Array-backed textures are resident for
            // the session and never recycled, so they have no LRU age to refresh.
            if (slotVal != 0 && !IPC::isFlipSlot(slotVal)) { g_slotLastUsed[slotVal] = g_frame; }
            return slotVal;
        }
        slotVal = resolveTextureSlot(name);
        namePtr = name;
        slotEpoch = g_texEpoch;
        return slotVal;
    }

    // PBR param map for a draw's base texture: <base>_paramh.dds beside it (texturematcher's
    // Morrowind export, tasks/forge-pbr-materials.md), else <base>_paramh_np.dds — the variant
    // authored for "no parallax", whose height is still valid for the gradient, which is all this
    // path reads. 0 = the base has no param map, which is the COMMON case and must stay cheap: the
    // miss is cached twice, in g_texSlot (name -> 0, never recycled) and here per key, so a
    // texture without one costs one pointer compare per frame after its first sight.
    //
    // Deliberately client-side: the client already reads every texture's bytes and ships them
    // (TexUploadWire carries the DDS inline), so the host needs no filesystem knowledge at all.
    std::uint32_t resolveParamSlot(const char* baseName, std::uint32_t baseSlot, SlotInfo& si) {
        // Nothing to pair: textureless, a cached miss on the base, or a flip-book frame (an
        // animated base has no single companion file).
        if (!baseName || baseSlot == 0 || IPC::isFlipSlot(baseSlot)) { return 0; }
        if (baseName == si.paramNamePtr && si.paramEpoch == g_texEpoch) {
            // Same LRU refresh as resolveCachedSlot's fast path: the param map is referenced
            // every frame its draw is, so it ages exactly as its base does and is never recycled
            // out from under a draw that still names it.
            if (si.paramSlot != 0) { g_slotLastUsed[si.paramSlot] = g_frame; }
            return si.paramSlot;
        }
        std::string stem = normalizeTextureName(baseName);
        const std::size_t dot = stem.find_last_of('.');
        const std::size_t sep = stem.find_last_of('\\');
        if (dot != std::string::npos && (sep == std::string::npos || dot > sep)) { stem.erase(dot); }
        std::uint32_t slot = 0;
        if (!stem.empty()) {
            slot = resolveTextureSlotEx((stem + "_paramh.dds").c_str(), true, true);
            if (slot == 0) { slot = resolveTextureSlotEx((stem + "_paramh_np.dds").c_str(), true, true); }
        }
        si.paramNamePtr = baseName;
        si.paramSlot    = slot;
        si.paramEpoch   = g_texEpoch;   // AFTER resolving: a recycle inside it bumps the epoch
        return slot;
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
    // THREAD SAFETY FOR BOTH PROBES BELOW. The emit helpers run on the produce worker AND on the
    // main thread (that is why resolveTextureSlot holds g_texResidencyMx at all), so a `static`
    // container inside one of them is shared mutable state. Unsynchronized, it corrupts the heap
    // under cell-change churn — when the set of names turns over fastest — and the access violation
    // then lands somewhere else entirely, on whichever thread next touches the damaged block.
    // Each probe is ARMED until it has said everything it has to say; the lock is taken only while
    // armed, so once disarmed a call is one relaxed atomic load and a predicted branch.
    std::mutex            g_diagProbeMx;
    std::atomic<bool>     g_normDiagArmed{true};
    std::atomic<bool>     g_texCensusArmed{true};

    void diagWorldDet(const char* tag, const char* name, const float* w) {
        if (!g_normDiagArmed.load(std::memory_order_relaxed)) { return; }
        static std::unordered_set<std::string> s_seen;
        std::string key = std::string(tag) + "|" + (name ? name : "(null)");
        std::lock_guard<std::mutex> lk(g_diagProbeMx);
        if (!s_seen.insert(key).second) { return; }
        if (s_seen.size() >= 128) { g_normDiagArmed.store(false, std::memory_order_relaxed); }
        const float det = w[0] * (w[5]*w[10] - w[6]*w[9])
                        - w[1] * (w[4]*w[10] - w[6]*w[8])
                        + w[2] * (w[4]*w[9]  - w[5]*w[8]);
        LOG::logline(">> [norm-diag] %s %s det=%.3f %s", tag, name ? name : "(null)",
                     det, det < 0.0f ? "MIRRORED" : "normal");
    }

    // WHITE-TEXTURE TRIAGE (TEMP): a census of what each draw path actually SHIPS as its bindless
    // slot. resolveTextureSlot only speaks up on its loud failures (not found / too large /
    // thrashing); a draw can still reach the host with slot 0 = host default white through routes
    // that say nothing at all — a null or empty texture name, a name-gated call site that never
    // asks the resolver, or a cached SlotInfo holding a stale 0. This prints one line per unique
    // (path, name, slot) triple and then switches itself off, so the steady-state cost on the
    // produce path is one predicted branch. A name that reappears under a DIFFERENT slot prints
    // again, which is exactly the LRU-recycle signal. Pairs with the host's "referenced slot(s)
    // still DEFAULT WHITE" line: this side maps name -> slot, that side maps slot -> white.
    void diagTexSlot(const char* tag, const char* name, std::uint32_t slot, bool hasD3D, bool glow) {
        if (!g_texCensusArmed.load(std::memory_order_relaxed)) { return; }
        static std::unordered_set<std::string> s_seen;
        std::string key = std::string(tag) + "|" + (name ? name : "(null)") + "|" + std::to_string(slot);
        std::lock_guard<std::mutex> lk(g_diagProbeMx);
        if (!s_seen.insert(key).second) { return; }
        if (s_seen.size() >= 600) { g_texCensusArmed.store(false, std::memory_order_relaxed); }
        LOG::logline(">> [tex-census] %-5s slot=%-4u d3d=%d glow=%d %s", tag, slot,
                     hasD3D ? 1 : 0, glow ? 1 : 0, name ? name : "(null)");
    }

    // FP1a: `dst` defaults to the main-pass scratch; buildFPDrawLists redirects the
    // identical packing into the FP scratch (same wire format, different host pass).
    void emitStaticDraw(SlotInfo& si, const MGE::GeometryCache::CachedGeometry& e,
                        std::uint32_t& count, std::vector<std::uint8_t>& dst = g_drawScratch,
                        std::uint32_t portalFlags = 0) {
            IPC::DrawItemWire item;
            item.slot = si.slot;
            stampSlotCell(si, e);
            diagWorldDet("STATIC", e.textureName, e.worldTransformD3D);
            // Textureless visual (e.textureName null) → slot 0 = host default white; resolveCachedSlot
            // would std::string(nullptr) on the name lookup, so short-circuit it.
            item.texIndex = e.textureName
                ? resolveCachedSlot(e.textureName, si.baseNamePtr, si.baseSlot, si.baseEpoch) : 0u;
            diagTexSlot("STAT", e.textureName, item.texIndex, e.d3dTexture != nullptr, e.enchantGlow);
            // Terrain DECAL_1 overlay (second land texture). resolveTextureSlot ships its DDS
            // bytes the same way as the base map. Non-landscape / single-texture draws get 0,
            // which gates the frag's splat off → byte-for-byte unchanged.
            item.overlayTexIndex = (e.isLandscape && e.d3dOverlay && e.overlayTextureName)
                ? resolveCachedSlot(e.overlayTextureName, si.ovNamePtr, si.ovSlot, si.ovEpoch) : 0u;
            // PBR param map of the base texture (0 = none → the host shades it exactly as before).
            item.paramTexIndex = resolveParamSlot(e.textureName, item.texIndex, si);
            item.alphaRef = e.alphaTest ? e.alphaRef : 0.0f;     // alpha-test cutout (0 = no test)
            // MW's per-map texture address mode, plus the enchanted-item glow bit riding this
            // lane's spare bits (see IPC::kTexFlagEnchantGlow — one decode, in packTexAlpha).
            item.clampMode = e.baseClamp | (e.enchantGlow ? IPC::kTexFlagEnchantGlow : 0u);
            // Tier 2b material: ship the captured MaterialProperty colours + the vertex-colour
            // routing, replicating buildCacheReflectionState/buildCacheMainState EXACTLY. useVCol
            // = mesh has colours AND its VertexColorProperty says to use them (else real material
            // colours get white-washed); when off we send vColSource 0 so the frag uses the
            // constant material (vertexMaterialNone), ignoring the VB's colour slot.
            item.matDiffuse[0]  = e.matDiffuse[0];  item.matDiffuse[1]  = e.matDiffuse[1];  item.matDiffuse[2]  = e.matDiffuse[2];
            item.matAmbient[0]  = e.matAmbient[0];  item.matAmbient[1]  = e.matAmbient[1];  item.matAmbient[2]  = e.matAmbient[2];
            // The AUTHORED emissive rides its own lane and gets decodeAuthored() on the host;
            // the flux/area GAIN rides its own and deliberately does not. See emissiveForDraw.
            item.matEmissive[0] = e.matEmissive[0];  item.matEmissive[1] = e.matEmissive[1];  item.matEmissive[2] = e.matEmissive[2];
            MGE::GeometryCache::emissiveForDraw(e, item.emissiveGain);
            item.vColSource = (e.hasVertexColor && e.vColSource != 0) ? e.vColSource : 0u;
            // C4d shadow-caster category: LIVE (NPC/creature parts + held equipment,
            // activators, doors) → the host's dynamic shadow tile, not the cached statics.
            // …plus the stencil-portal role (IPC::kDrawPortalMask): 0 for every ordinary draw, so
            // the host's classify sees no change until a portal is actually on screen.
            item.casterFlags = (e.isLive ? IPC::kDrawCasterLive : 0u)
                             | (e.animated ? IPC::kDrawCasterAnimated : 0u)
                             | portalFlags;
            memcpy(item.world, e.worldTransformD3D, 16 * sizeof(float));
            // CAMERA-RELATIVE rendering: subtract the camera world position from the world
            // translation so vertices reach the shader near the origin. At MW's exterior
            // coordinates (|eye| ~150k) absolute world positions quantise to ~0.01-0.1 units
            // in float32, and the combined viewProj's per-vertex cancellation then produces
            // orientation-dependent stretching. Shifting world + viewProj + lights + eyePos by
            // -eye keeps all vertex math small/precise. (viewProj is built translation-free below.)
            // Far from the origin the eye and the translation are each exact in double (ExactPos).
            shipEntryT(&item.world[12], e);
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
            if (isPlayerOwned(e) && &dst == &g_drawScratch) {
                g_playerPatch.push_back({ 0, (std::uint32_t)(at + offsetof(IPC::DrawItemWire, world)
                                                                + 12 * sizeof(float)), 1 });
            }
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
            stampSlotCell(si, e);
            if (e.bonePalette.size() < (std::size_t)e.numBones * 16) {
                return;   // palette not yet built this frame
            }

            IPC::SkinnedDrawWire item;
            item.slot     = si.slot;
            item.numBones = e.numBones;
            item.mirror   = e.mirrored ? 1u : 0u;
            item.texIndex = resolveCachedSlot(e.textureName, si.baseNamePtr, si.baseSlot, si.baseEpoch);
            diagTexSlot("SKIN", e.textureName, item.texIndex, e.d3dTexture != nullptr, e.enchantGlow);
            item.alphaRef = e.alphaTest ? e.alphaRef : 0.0f;     // alpha-test cutout (0 = no test)
            // Address mode + the enchanted-item glow bit (IPC::kTexFlagEnchantGlow). Skinned parts
            // are how ENCHANTED ARMOUR AND CLOTHING glow — they are body parts, not rigid props.
            item.clampMode = e.baseClamp | (e.enchantGlow ? IPC::kTexFlagEnchantGlow : 0u);
            // Alpha-BLEND state. dispatch() still routes every skinned entry here — a blended
            // skinned part needs the bone palette that only this list carries — so the blend is a
            // TAG, not a re-route: the host packs it in every walk exactly as before and moves only
            // its draw into the alpha stage. Ghosts and hair/mane cards live entirely in this flag.
            // alphaTest rides its OWN bit rather than being inferred from alphaRef != 0: MW lets a
            // cutout test at ref 0 (GREATER 0 = drop only fully transparent texels), and the velk's
            // mane does exactly that, so alphaRef alone reports it as untested. The host uses these
            // two to tell a solid-body-that-fades from a card whose texture alpha carves it.
            item.blendFlags = (e.blendEnable  ? IPC::kSkinFlagBlended   : 0u)
                            | (e.twoSided     ? IPC::kSkinFlagTwoSided  : 0u)
                            | (e.alphaTest    ? IPC::kSkinFlagAlphaTest : 0u)
                            | (e.alphaAnimated? IPC::kSkinFlagAlphaAnim : 0u);
            item.matAlpha   = e.matDiffuse[3];   // FFE per-draw fade (same source as emitAlphaDraw)
            item.srcBlend   = e.srcBlend;
            item.destBlend  = e.destBlend;
            // Sort key: BONE 0's world translation, eye-relative, along the view forward. Read
            // BEFORE the camera-relative shift below (the shift subtracts the same eyePos, so the
            // key would be identical either way — taking it here keeps it independent of that).
            {
                const float bx = e.bonePalette[12] - DistantLand::eyePos.x;
                const float by = e.bonePalette[13] - DistantLand::eyePos.y;
                const float bz = e.bonePalette[14] - DistantLand::eyePos.z;
                item.viewDepth = bx * DistantLand::mwView._13
                               + by * DistantLand::mwView._23
                               + bz * DistantLand::mwView._33;
            }

            const std::size_t paletteBytes = (std::size_t)e.numBones * 64;  // numBones * 16 floats
            const std::size_t at = dst.size();
            dst.resize(at + sizeof(item) + paletteBytes);
            std::uint8_t* out = dst.data() + at;
            memcpy(out, &item, sizeof(item));                  out += sizeof(item);
            memcpy(out, e.bonePalette.data(), paletteBytes);
            // CAMERA-RELATIVE: the bone palette is world-space; shift each bone matrix's
            // translation by -eye so the skinned vertices land near the origin, consistent
            // with the translation-free viewProj + shifted lights (see buildDrawList).
            // ExactPos kSkinned ships each bone's exact double translation instead (bonePaletteT,
            // filled beside the palette by buildBonePalette); mode 0 is the old float subtraction.
            {
                float* pal = reinterpret_cast<float*>(out);
                const bool haveExact = e.bonePaletteT.size() >= (std::size_t)e.numBones * 3;
                const bool useExact = haveExact && MGE::ExactPos::on(MGE::ExactPos::kSkinned);
                for (std::uint32_t b = 0; b < e.numBones; ++b) {
                    const double* exactT = haveExact ? &e.bonePaletteT[b * 3] : nullptr;
                    MGE::ExactPos::rel(&pal[b * 16 + 12], &e.bonePalette[b * 16 + 12], exactT, useExact);
                    if (haveExact) {
                        MGE::ExactPos::noteShipped(MGE::ExactPos::kShipSkinned, e.isFP,
                                                   &pal[b * 16 + 12], exactT);
                    }
                }
            }
            // The palette IS this part's transform, so the whole palette carries the player's
            // motion — every bone, not just the root.
            if (isPlayerOwned(e) && &dst == &g_skinnedScratch) {
                g_playerPatch.push_back({ 1, (std::uint32_t)(at + sizeof(item) + 12 * sizeof(float)),
                                          e.numBones });
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
    // FP1e: `dst` defaults to the main-pass scratch; buildFPFrame redirects the identical
    // packing into the FP multi-map scratch (same wire format, drawn by the host FP pass) —
    // the same redirection emitStaticDraw/emitSkinnedDraw/emitAlphaDraw already take.
    void emitMultiMapDraw(SlotInfo& si, const MGE::GeometryCache::CachedGeometry& e,
                          std::uint32_t& count, std::vector<std::uint8_t>& dst = g_multiMapScratch,
                          bool blended = false) {
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
            // CAMERA-RELATIVE: shift translation by -eye (see emitStaticDraw).
            shipEntryT(&item.world[12], e);
            item.matDiffuse[0]  = e.matDiffuse[0];  item.matDiffuse[1]  = e.matDiffuse[1];  item.matDiffuse[2]  = e.matDiffuse[2];
            item.matAmbient[0]  = e.matAmbient[0];  item.matAmbient[1]  = e.matAmbient[1];  item.matAmbient[2]  = e.matAmbient[2];
            // The AUTHORED emissive rides its own lane and gets decodeAuthored() on the host;
            // the flux/area GAIN rides its own and deliberately does not. See emissiveForDraw.
            item.matEmissive[0] = e.matEmissive[0];  item.matEmissive[1] = e.matEmissive[1];  item.matEmissive[2] = e.matEmissive[2];
            MGE::GeometryCache::emissiveForDraw(e, item.emissiveGain);
            item.vColSource = (e.hasVertexColor && e.vColSource != 0) ? e.vColSource : 0u;
            item.alphaRef   = e.alphaTest ? e.alphaRef : 0.0f;   // base-stage alpha test
            item.stageCount = (std::uint32_t)ns;
            item.drawFlags  = (blended ? IPC::kMMDrawFlagBlended : 0u)   // Route C: alpha-stage blend draw
                            | (e.enchantGlow ? IPC::kMMDrawFlagEnchantGlow : 0u);
            item.matAlpha   = e.matDiffuse[3];   // FFE per-draw fade (same source as emitAlphaDraw)
            for (int s = 0; s < ns; ++s) {
                const std::uint32_t tex = resolveTextureSlot(st[s].name);   // bindless slot (0 = white)
                diagTexSlot(s == 0 ? "MM0" : "MMn", st[s].name, tex, true, e.enchantGlow);
                item.stages[s] = IPC::packMMStage(tex, st[s].uv, st[s].op, st[s].clamp);
            }
            const std::size_t at = dst.size();
            dst.resize(at + sizeof(item));
            memcpy(dst.data() + at, &item, sizeof(item));
            // Scratch id 2 IS g_multiMapScratch (see the park-lag patch loop), so an offset
            // recorded from another scratch would patch a stranger's matrix — guard the push the
            // way emitStaticDraw/emitAlphaDraw do. The arms are camera-welded anyway: the FP
            // payload ships with the ARM camera and needs no player park-lag correction.
            if (isPlayerOwned(e) && &dst == &g_multiMapScratch) {
                g_playerPatch.push_back({ 2, (std::uint32_t)(at + offsetof(IPC::MultiMapDrawWire, world)
                                                                + 12 * sizeof(float)), 1 });
            }
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
            stampSlotCell(si, e);
            diagWorldDet("ALPHA", e.textureName, e.worldTransformD3D);
            item.texIndex  = resolveCachedSlot(e.textureName, si.baseNamePtr, si.baseSlot, si.baseEpoch);
            diagTexSlot("ALPH", e.textureName, item.texIndex, e.d3dTexture != nullptr, e.enchantGlow);
            item.srcBlend  = e.srcBlend;
            item.destBlend = e.destBlend;
            item.alphaRef  = e.alphaTest ? e.alphaRef : 0.0f;
            // Address mode + the enchanted-item glow bit (IPC::kTexFlagEnchantGlow).
            item.clampMode = e.baseClamp | (e.enchantGlow ? IPC::kTexFlagEnchantGlow : 0u);
            item.matDiffuse[0]  = e.matDiffuse[0];  item.matDiffuse[1]  = e.matDiffuse[1];  item.matDiffuse[2]  = e.matDiffuse[2];
            item.matAlpha       = e.matDiffuse[3];   // MaterialProperty::alpha (the FFE per-draw fade)
            item.matAmbient[0]  = e.matAmbient[0];  item.matAmbient[1]  = e.matAmbient[1];  item.matAmbient[2]  = e.matAmbient[2];
            // The AUTHORED emissive rides its own lane and gets decodeAuthored() on the host;
            // the flux/area GAIN rides its own and deliberately does not. See emissiveForDraw.
            item.matEmissive[0] = e.matEmissive[0];  item.matEmissive[1] = e.matEmissive[1];  item.matEmissive[2] = e.matEmissive[2];
            MGE::GeometryCache::emissiveForDraw(e, item.emissiveGain);
            item.vColSource = (e.hasVertexColor && e.vColSource != 0) ? e.vColSource : 0u;
            memcpy(item.world, e.worldTransformD3D, 16 * sizeof(float));
            // CAMERA-RELATIVE: shift translation by -eye (see emitStaticDraw).
            shipEntryT(&item.world[12], e);
            // AT3 captured-geometry locators unused for a cached-mesh (slot) item — the host
            // reads them only when slot == kAlphaSlotCaptured. Zero so they never alias garbage.
            item.vertexBase = item.indexBase = item.indexCount = 0;
            // Cull mode from the shape's live NiStencilProperty (DRAW_BOTH → CULL_NONE) + winding
            // from the mirror flag, so the host draws it exactly as MW does (single-sided alpha
            // like the draped altar cloth gets CULL_BACK, hiding its back/interior faces).
            item.cullFlags = (e.twoSided ? IPC::kAlphaCullTwoSided : 0u)
                           | (e.mirrored ? IPC::kAlphaCullMirrored : 0u);
            // A CACHED blend on this list is single-map by construction — dispatch() routes any
            // entry carrying dark/detail/glow to Route C (MultiMapDrawWire) instead, which has its
            // own full stage loop. The AT3 extra stages are for captured DIPs only; zero them so
            // the host's per-draw stage table never reads stale bytes for a cached item.
            item.stageCount = 0;
            item.stages[0] = item.stages[1] = item.stages[2] = 0;
            const std::size_t at = dst.size();
            dst.resize(at + sizeof(item));
            memcpy(dst.data() + at, &item, sizeof(item));
            if (isPlayerOwned(e) && &dst == &g_alphaScratch) {
                g_playerPatch.push_back({ 3, (std::uint32_t)(at + offsetof(IPC::AlphaDrawWire, world)
                                                                + 12 * sizeof(float)), 1 });
            }
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
            // Captured DIPs are reconstructed from D3D8 render state, not from the cache, so there is
            // no fixture/light/area to derive a flux-per-area gain from — 1.0 = no boost. It MUST be
            // written: `item` is not zero-initialised, and the gain's identity is 1, not 0 (the host's
            // vert MULTIPLIES by it, so a zero would black this DIP's emissive out entirely).
            item.emissiveGain[0] = item.emissiveGain[1] = item.emissiveGain[2] = 1.0f;
            item.vColSource = rec.vColSource;
            memcpy(item.world, rec.world, 16 * sizeof(float));
            // Same eye as every other subtraction (ExactPos::eye); the record itself carries no
            // node, so its translation stays the float one — a recorded residual.
            MGE::ExactPos::rel(&item.world[12], &rec.world[12], nullptr, false);
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
            // Dark/detail/glow stages MW folded into this same DIP (captureAlphaDraw resolved them).
            item.stageCount = rec.stageCount;
            item.stages[0] = rec.stages[0]; item.stages[1] = rec.stages[1]; item.stages[2] = rec.stages[2];
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

        // Park-lag correction: where the player was when this payload was built, and a clean patch
        // list for the emit helpers to fill. Captured HERE, not at the stamp, so the origin and the
        // transforms that ride it come from the same instant.
        g_playerPatch.clear();
        g_buildCacheFrame = MGE::GeometryCache::currentFrame();
        g_playerBakeValid = MGE::GeometryCache::playerRootOrigin(g_playerBake);

        g_drawScratch.clear();
        g_drawScratch.reserve((foldKeys ? foldKeys->size() : keys.size()) * sizeof(IPC::DrawItemWire));
        g_skinnedScratch.clear();
        g_multiMapScratch.clear();
        g_alphaScratch.clear();

        const bool wantStatic  = (bool)g_drawVec;
        const bool wantSkinned = (bool)g_skinnedVec;
        const bool wantMM      = (bool)g_multiMapVec;
        const bool wantAlpha   = (bool)g_alphaVec;
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

        // T3: MW's near terrain is the host's job now — one heightfield covers near and far on a
        // single LOD ladder, so emitting MW's copy is a second producer of one surface. OFF unless
        // the host reports it is actually drawing terrain this frame (see g_hostOwnsTerrain):
        // suppressing when the host is not drawing turns a double-draw into a HOLE.
        // forgeOwnsFrame() is ANDed in for correctness-by-construction, not belt-and-braces: when
        // the seam drops there is no RPC, so g_hostOwnsTerrain keeps its last value and would read
        // stale-true. Every other suppression keys on this same predicate for the same reason.
        const bool dropMWLand = g_hostOwnsTerrain && RenderProcess::forgeOwnsFrame();

        // STENCIL "FAKE HOLE" PORTALS. Which PLACED OBJECTS are portals? An object qualifies only if
        // it owns BOTH a mask and a hull — a mask alone has nothing to open, and a HULL alone is the
        // dangerous half: it is drawn depth-test-off, so with no gate to confine it, it erases the
        // depth of whatever it rasterizes over. Pair or don't play.
        //
        // Built from the cache's tiny roled-key set (~50 shapes install-wide), not the visible set,
        // so membership is a property of the OBJECT and does not flicker with what is on screen this
        // frame. Owners are TES3 reference pointers, compared only for identity.
        static std::unordered_map<const void*, std::uint32_t> s_portalRoles;  // owner -> mask|hull bits
        static std::unordered_set<const void*> s_portalOwners;
        s_portalRoles.clear();
        s_portalOwners.clear();
        for (std::uint32_t rk : MGE::GeometryCache::portalRoleKeys()) {
            auto rit = cacheMap.find(rk);
            if (rit == cacheMap.end()) continue;
            const auto& re = rit->second;
            if (!re.portalOwner) continue;                  // no TES3 reference → never a portal
            s_portalRoles[re.portalOwner] |= (re.stencilRole == 1) ? 1u : 2u;
        }
        for (const auto& kv : s_portalRoles) {
            if (kv.second == 3u) s_portalOwners.insert(kv.first);   // has a mask AND a hull
        }
        const bool anyPortals = !s_portalOwners.empty();
        // Is this entry part of a portal object? Its whole object leaves the GPU-driven indirect
        // groups together — the trick needs author-role ordering, which cmdExecuteIndirect cannot
        // express, and needs to escape the Hi-Z occlusion test, which culls anything living behind
        // the surface it is opening.
        auto portalOwned = [&](const MGE::GeometryCache::CachedGeometry& e) {
            return anyPortals && e.portalOwner && s_portalOwners.count(e.portalOwner) != 0;
        };
        // Deferred portal draws, emitted after the visible-set loop grouped by owner in role order.
        struct PortalCand {
            const void* owner;
            SlotInfo* si;
            const MGE::GeometryCache::CachedGeometry* e;
            std::uint8_t role;    // 0 member, 1 mask, 2 hull
        };
        static std::vector<PortalCand> portalCands;
        portalCands.clear();

        // Per-entry dispatch, identical on both paths (see the filter contract above).
        // slot is mutable — the emit helpers update its cached texture SlotInfo.
        auto dispatch = [&](auto& slot, const auto& e) {
            if (e.isSky) return;   // sky rides the Forge alpha-blend sky pass (buildSkyDrawList)
            if (e.isFP) return;    // FP arms ride the dedicated host FP pass (buildFPDrawLists)
            // Terrain: host-owned. Covers the alpha-SPLAT trishapes too (MW multi-passes terrain
            // texture blending), which is why this sits above the blendEnable branches — the host
            // heightfield already carries the blend, per-pixel, from the same VTEX data.
            if (e.isLandscape && dropMWLand) return;
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
                // Portal helpers are textureless BY DESIGN — comGravePit's mask and hull are both
                // bare geometry — which puts them in front of the collision-proxy filter below,
                // and a dropped mask or hull is the whole feature gone. They are not proxies and
                // the mesh says so: a roled shape carries a NiStencilProperty driving the stencil
                // buffer, which no worldPickObjectRoot proxy ever does. So the ROLE, not the
                // texture, decides here. Untagged portal MEMBERS keep the ordinary filter.
                if (wantStatic && e.stencilRole && portalOwned(e)) {
                    portalCands.push_back({ e.portalOwner, &slot, &e, e.stencilRole });
                    return;
                }
                if (e.isPickRoot || e.blendEnable) return;
                if (wantStatic && portalOwned(e)) {
                    portalCands.push_back({ e.portalOwner, &slot, &e, e.stencilRole });
                    return;
                }
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
                    emitMultiMapDraw(slot, e, multiMapCount, g_multiMapScratch, /*blended=*/true);
                } else {
                    return;   // feature off — leave MW's DIP (AT3 white) as the fallback, no dedup
                }
                // AT3: record (bindless slot, vertexCount) so captureAlphaDraw skips this exact blend
                // if MW's own DIP for it still reaches the reject gate (old-msoc dll: opaque-only
                // skip → cached blends both DIP AND ride the host). Both host paths above suppress it.
                // The slot — not e.d3dTexture — is the key: a NiFlipController rebinds a new GPU
                // texture nearly every frame, so a pointer key never matched for flip books and the
                // magelight quad was drawn twice (see alphaDedupKey). resolveCachedSlot is memoized
                // into this SlotInfo, so the emit above/below re-reads it for free.
                if (e.d3dTexture && e.textureName) {
                    const std::uint32_t texSlot = resolveCachedSlot(e.textureName, slot.baseNamePtr,
                                                                    slot.baseSlot, slot.baseEpoch);
                    if (texSlot != 0u) {
                        g_alphaDedup.insert(alphaDedupKey(texSlot, e.vertexCount));
                    }
                }
                return;
            }
            if (!e.isLandscape && (e.d3dDark || e.d3dDetail || e.d3dGlow)) {
                if (wantMM) emitMultiMapDraw(slot, e, multiMapCount);
            } else if (wantStatic && portalOwned(e)) {
                // Textured portal members and roled shapes (dwrvgratepipe's mask wears
                // stencilerror.dds; its hull IS the visible room behind the grate).
                portalCands.push_back({ e.portalOwner, &slot, &e, e.stencilRole });
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
                                   || (g_frame - s_epochFrame < kCaptureEpochGraceFrames)
                                   || (int)(g_captureGraceUntil - g_frame) > 0;
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
                    const std::uint32_t defBefore = MGE::GeometryCache::captureDeferredLastBuild();
                    const auto* e = MGE::GeometryCache::ensureLive(key);
                    ensureLiveMs += nowMs() - te0;
                    if (!e) {                             // no model data / capture failed / deferred
                        noteNearMiss(key, MGE::GeometryCache::captureDeferredLastBuild() != defBefore
                                              ? NearMiss::Deferred : NearMiss::NoEntry);
                        continue;
                    }
                    // The unusable-skinned filter buildFrustumVisibleSet applies on
                    // non-fold frames (never drawn; trips the bound helper).
                    if (e->isSkinned && (e->skinnedUnsupported || e->numBones == 0)) continue;
                    auto ks = g_keySlot.find(key);
                    if (ks == g_keySlot.end()) {
                        noteNearMiss(key, NearMiss::NoSlot);
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
        if (foldKeys) nearMissEndBuild();

        // ---- STENCIL "FAKE HOLE" PORTALS: emit the deferred objects, in ROLE order ---------------
        // masks -> hulls -> the rest, per object. That is the order the trick is built on: the mask
        // stamps where the opening is VISIBLE (its own depth test is the gate the stencil used to
        // provide), the hull then overwrites the occluder's depth but only inside that stamp, and
        // the real content draws over the hull. Order WITHIN "the rest" does not matter — those are
        // ordinary depth-tested draws and depth resolves them — so this needs no author index.
        //
        // The pairing is re-checked HERE, on what was actually DEFERRED, not on what the cache
        // holds: an object whose mask or hull went down some other path (blended, skinned, multimap)
        // would otherwise ship a hull with no gate, which erases depth wherever it rasterizes.
        // Unpaired objects fall back to plain static draws — exactly today's behaviour.
        std::uint32_t portalObjects = 0, portalMasks = 0, portalHulls = 0, portalMembers = 0;
        std::uint32_t portalUnpaired = 0;
        {
            MGE_ZoneScopedN("geom:portals");
            static std::unordered_map<const void*, std::uint32_t> s_emitRoles;
            s_emitRoles.clear();
            for (const PortalCand& c : portalCands) {
                if (c.role) s_emitRoles[c.owner] |= (c.role == 1) ? 1u : 2u;
            }
            // Stable owner order (first appearance) so the emitted list does not reshuffle
            // frame-to-frame on unordered_map iteration order.
            static std::vector<const void*> s_owners;
            static std::unordered_set<const void*> s_seenOwner;
            s_owners.clear();
            s_seenOwner.clear();
            for (const PortalCand& c : portalCands) {
                if (s_seenOwner.insert(c.owner).second) s_owners.push_back(c.owner);
            }
            for (const void* owner : s_owners) {
                const bool paired = (s_emitRoles[owner] == 3u);
                if (paired) ++portalObjects;
                // Three sweeps per object: role 1, then 2, then 0. The candidate list is small
                // (one portal object is a handful of shapes), so this stays trivially cheap.
                for (std::uint8_t want = 1; want <= 3; ++want) {
                    const std::uint8_t role = (want == 3) ? 0 : want;
                    for (const PortalCand& c : portalCands) {
                        if (c.owner != owner || c.role != role) continue;
                        std::uint32_t flags = 0;
                        if (paired) {
                            flags = (role == 1) ? IPC::kDrawPortalMask
                                  : (role == 2) ? IPC::kDrawPortalHull
                                                : IPC::kDrawPortalMember;
                            if (role == 1)      ++portalMasks;
                            else if (role == 2) ++portalHulls;
                            else                ++portalMembers;
                        }
                        emitStaticDraw(*c.si, *c.e, drawCount, g_drawScratch, flags);
                    }
                }
            }
            portalUnpaired = (std::uint32_t)(s_owners.size() - portalObjects);
        }
        // One line per COMPOSITION change, in the style of [fp-set] / [alpha-cap] — and logged even
        // when the counts are all ZERO, which is the whole point. Counts alone cannot tell "no
        // portals in this cell" from "the walk dropped them", and that ambiguity is exactly the
        // failure mode this change exists to stop being invisible ([[project_shadowbox_name_prune]]
        // item 4). `deferred` is what separates them: a cell with no portal meshes defers NOTHING,
        // while a portal whose mask went missing still defers its hull and shows up as
        // deferred>0 objects=0 unpaired=1.
        {
            static std::uint64_t s_lastPortalSig = ~0ull;
            // Five 12-bit fields — non-overlapping, so a change in any one of them is always a
            // change in the signature (a portal object never approaches 4096 shapes).
            auto f12 = [](std::size_t v) { return (std::uint64_t)(v & 0xFFFu); };
            const std::uint64_t sig = (f12(portalObjects) << 48) | (f12(portalMasks) << 36)
                                    | (f12(portalHulls)   << 24) | (f12(portalMembers) << 12)
                                    | f12(portalCands.size());
            if (sig != s_lastPortalSig) {
                s_lastPortalSig = sig;
                LOG::logline(">> [portal] objects=%u masks=%u hulls=%u members=%u "
                             "(deferred=%u, unpaired=%u)",
                             portalObjects, portalMasks, portalHulls, portalMembers,
                             (unsigned)portalCands.size(), portalUnpaired);
            }
        }

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
            std::uint32_t suppressSkips = 0, ghostSkips = 0;
            // OFF-SCREEN is a frustum statement, and this loop exists only for off-screen casters.
            // "The engine didn't draw it" has three causes and only that one wants a re-emit; hidden
            // and DESPAWNED look identical from here and both ship a FROZEN cached transform. That is
            // the sheathed-weapon ghost: MW unparents the drawn weapon, so it leaves the classify set
            // and nothing updates its world transform again — while this loop keeps feeding the host
            // the last pose it had, so the sword hangs in the air where your hand was and you walk out
            // from under it. Only the eviction sweep retired it, up to 30 frames later.
            // So let the frustum answer its own question: a bound sphere FULLY inside the view that
            // the engine still declined to draw is not an off-screen caster, and only those pay the
            // parent climb below. Steady state costs 6 dot products — near movers that miss the
            // visible set genuinely are behind you, and their sphere fails INSIDE immediately.
            D3DXMATRIX viewProjNow;
            D3DXMatrixMultiply(&viewProjNow, &DistantLand::mwView, &DistantLand::mwProj);
            const ViewFrustum viewNow(&viewProjNow);
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
                // Portal objects ship ONLY through the inline portal block above. Re-emitting a
                // member here would put the same part in the host's indirect groups AND inline —
                // a double draw — and a re-emitted MASK or HULL would be a depth-eraser loose in
                // the GPU-driven path with no gate in front of it.
                if (portalOwned(e)) continue;
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
                // In full view and still not drawn ⇒ ask the graph whether it is there at all.
                BoundingSphere bs;
                bs.center = c;
                bs.radius = rad;
                if (viewNow.ContainsSphere(bs) == ViewFrustum::INSIDE
                    && !MGE::GeometryCache::attachedNow(mkey)) {
                    ++ghostSkips;
                    continue;
                }
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
            // Offscreen PLAIN-STATIC shadow casters (lanterns, wall fixtures) — the mover loop above
            // handles skinned / MM-head / rigid-LIVE, but a plain static is not a mover candidate, so
            // on first cell load a fixture behind the camera has no host caster record and casts no
            // shadow until it enters the frustum once (then "completes" as you turn — the reported
            // bug). The cache maintains a near-eye static-caster snapshot on the eviction sweep; re-
            // emit any not already in this frame's visible set so the host seeds the record regardless
            // of view direction. Statics don't move, so one sighting's record stays correct for the
            // cell; re-emitting every frame just keeps it refreshed cheaply (the host overwrites the
            // same absolute world). CACHED data only — emitStaticDraw does no NiTriShape deref — and
            // the host GPU-culls these offscreen draws from the COLOUR pass while keeping their caster
            // record, exactly the movers' contract.
            // Once-per-cell seed gate. A plain static's host caster record persists for the WHOLE
            // cell once it is drawn once: everMoved stays false so the host's compaction sweep never
            // expires it (forgerender.cpp:7349 gates expiry on everMoved), and the record is only
            // wiped on a cell change — which bumps g_cellEpoch and clears g_casterSlots host-side
            // (forgerender.cpp:7069). So re-emitting the same near static every frame is pure waste.
            // Seed each key once, keyed to its host SLOT: an eviction+re-upload lands on a fresh slot
            // (host record gone) so the identity mismatch re-seeds it, while a resident static keeps
            // its slot and stays skipped. The map is cleared on epoch change, matching the host wipe.
            static std::unordered_map<std::uint32_t, std::uint32_t> s_offscreenSeeded;  // cacheKey -> host slot
            static std::uint32_t s_offscreenSeededEpoch = 0xFFFFFFFFu;
            // Moving plain statics collected by the two loops below (kind 0 opaque, 1 alpha),
            // pose-refreshed and emitted after both, like the movers' s_reEmitRefresh.
            static std::vector<ReEmitCand> s_animStaticRefresh;   s_animStaticRefresh.clear();
            // "Moving" = moved within kSettleResend cache frames. MUST outlast the host's settle window
            // (kCasterSettleFrames 10, forgerender.cpp): the host keeps a mover's record past its
            // expiry only if its LAST sighting was >10 frames after its last move, so the re-send
            // has to keep reporting the part at rest for that long. dynamicHint (4) alone stopped
            // first, and a pause menu — the script stops turning the lantern — dropped its shadow.
            constexpr std::uint64_t kSettleResend = 30;
            auto movingStatic = [cacheFrame](const MGE::GeometryCache::CachedGeometry& ge) {
                return ge.dynamicHint > 0
                    || (ge.lastMoveFrame != 0 && cacheFrame - ge.lastMoveFrame <= kSettleResend);
            };
            if (s_offscreenSeededEpoch != g_cellEpoch) {
                s_offscreenSeededEpoch = g_cellEpoch;
                s_offscreenSeeded.clear();
            }
            if (wantStatic) {
                const auto& staticCasters = MGE::GeometryCache::nearStaticCasters();
                for (std::uint32_t skey : staticCasters) {
                    auto cit = cacheMap.find(skey);
                    if (cit == cacheMap.end()) continue;                      // stale key (evicted since the sweep)
                    const auto& e = cit->second;
                    if (s_visLookup.count(skey)) continue;                    // already emitted in the visible set
                    if (portalOwned(e)) continue;                            // inline portal block only (see above)
                    if (e.suppressedFrame == cacheFrame) continue;
                    // Precise per-frame classification (the sweep collected a padded superset).
                    if (e.isSky || e.isFP || e.isSkinned || e.isLandscape || e.blendEnable || e.isLive) continue;
                    if (e.d3dTexture && (e.d3dDark || e.d3dDetail || e.d3dGlow)) continue;   // MM → mover loop
                    if (!e.d3dTexture && e.isPickRoot) continue;                             // collision proxy
                    D3DXVECTOR3 c; float rad;
                    cacheWorldBounds(e, c, rad);
                    const float dx = c.x - DistantLand::eyePos.x;
                    const float dy = c.y - DistantLand::eyePos.y;
                    const float dz = c.z - DistantLand::eyePos.z;
                    if (dx * dx + dy * dy + dz * dz > r2) continue;           // precise near-eye cull
                    auto ks = g_keySlot.find(skey);
                    if (ks == g_keySlot.end()) continue;                      // never uploaded a host slot
                    // A MOVING plain static is the exception to the seed rule: the host flags it
                    // everMoved and its mover expiry forgets the record kMoverFresh frames after the
                    // last sighting, so one seed per cell lost its shadow the moment it left view.
                    // Tribunal's Light_MH_Rope_Lantern swings by SCRIPT — its NIF holds no controller
                    // at all — so "moving" is read from MOTION (movingStatic: the transform changed
                    // within kSettleResend frames), not from controllers. Frozen at its last value
                    // off screen; the re-send's ensureLive keeps it current, so a part that comes to
                    // rest drops back to the seed rule once the host has had time to call it settled
                    // (C4b keeps a settled record). Deferred: ensureLive must not run while this loop
                    // reads cacheMap.
                    if (movingStatic(e)) { s_animStaticRefresh.push_back({ skey, 0 }); continue; }
                    auto seedIt = s_offscreenSeeded.find(skey);
                    if (seedIt != s_offscreenSeeded.end() && seedIt->second == ks->second.slot) continue;
                    s_offscreenSeeded[skey] = ks->second.slot;   // seed (or re-seed on a new slot)
                    emitStaticDraw(ks->second, e, drawCount);
                }
            }
            // Offscreen BLENDED (alpha-over) casters — the lantern case. Blended fixtures cast via
            // the host ALPHA shadow path (refreshCasterRecord alphaCaster=true), fed by the VISIBLE
            // alpha draw list, so an off-screen blended lantern casts nothing until looked at. Push
            // the near ones into alphaCands so they sort + emit with the visible alpha and the host
            // registers their caster records. They draw in the alpha pass but, being off-screen, are
            // GPU frustum-culled from the visible image — only the shadow persists. Same CACHED-data,
            // near-eye, deduped-against-visible discipline as the opaque re-emit above.
            if (wantAlpha) {
                const auto& alphaCasters = MGE::GeometryCache::nearAlphaCasters();
                for (std::uint32_t akey : alphaCasters) {
                    auto cit = cacheMap.find(akey);
                    if (cit == cacheMap.end()) continue;               // stale key (evicted since sweep)
                    const auto& e = cit->second;
                    if (s_visLookup.count(akey)) continue;             // already in the visible alpha set
                    if (e.suppressedFrame == cacheFrame) continue;
                    if (!e.blendEnable || e.isSky || e.isFP || e.isSkinned || e.isLandscape || e.isLive) continue;
                    if (!e.d3dTexture) continue;                       // need a base map to mask/composite
                    const float* w = e.worldTransformD3D;
                    const float cx = e.boundsCenter[0], cy = e.boundsCenter[1], cz = e.boundsCenter[2];
                    const float wx = cx*w[0] + cy*w[4] + cz*w[8]  + w[12] - DistantLand::eyePos.x;
                    const float wy = cx*w[1] + cy*w[5] + cz*w[9]  + w[13] - DistantLand::eyePos.y;
                    const float wz = cx*w[2] + cy*w[6] + cz*w[10] + w[14] - DistantLand::eyePos.z;
                    if (wx*wx + wy*wy + wz*wz > r2) continue;          // precise near-eye cull
                    auto ks = g_keySlot.find(akey);
                    if (ks == g_keySlot.end()) continue;               // never uploaded a host slot
                    // Moving: every frame with a fresh pose, as in the opaque loop above.
                    if (movingStatic(e)) { s_animStaticRefresh.push_back({ akey, 1 }); continue; }
                    // Once-per-cell seed gate (shares s_offscreenSeeded; alpha/static keys are disjoint).
                    // A blended static lantern's alpha caster record is likewise persistent (everMoved
                    // false), so it need only enter the alpha pass once per cell to register.
                    auto aSeedIt = s_offscreenSeeded.find(akey);
                    if (aSeedIt != s_offscreenSeeded.end() && aSeedIt->second == ks->second.slot) continue;
                    s_offscreenSeeded[akey] = ks->second.slot;
                    alphaCands.push_back({ wx*fwdX + wy*fwdY + wz*fwdZ, &ks->second, &e, nullptr });
                }
            }
            // The moving statics both loops deferred: refresh the pose from the live node (a script
            // keeps turning an off-screen reference; only its draw is skipped), then emit. The cache pointer
            // ensureLive returns is node-stable, so alphaCands may hold it like the loop's own.
            // [anim-static] DIAG: is the moving-static re-emit reaching its parts? Kept on purpose
            // (animated-light investigation) — one line per 120 frames.
            static std::uint32_t s_asFrames = 0, s_asQueued = 0, s_asEmitted = 0, s_asNull = 0;
            ++s_asFrames;
            s_asQueued += (std::uint32_t)s_animStaticRefresh.size();
            if (s_asFrames >= 120) {
                std::uint32_t nAnimAll = 0;
                for (std::uint32_t k : MGE::GeometryCache::nearStaticCasters()) {
                    auto c2 = cacheMap.find(k);
                    if (c2 != cacheMap.end() && movingStatic(c2->second)) { ++nAnimAll; }
                }
                for (std::uint32_t k : MGE::GeometryCache::nearAlphaCasters()) {
                    auto c2 = cacheMap.find(k);
                    if (c2 != cacheMap.end() && movingStatic(c2->second)) { ++nAnimAll; }
                }
                LOG::logline("[anim-static] 120f: queued=%u emitted=%u ensureNull=%u movingInNearSets=%u",
                             s_asQueued, s_asEmitted, s_asNull, nAnimAll);
                s_asFrames = s_asQueued = s_asEmitted = s_asNull = 0;
            }
            for (const auto& rc : s_animStaticRefresh) {
                const double tpr0 = nowMs();
                const auto* fr = MGE::GeometryCache::ensureLive(rc.key);
                tailEnsureMs += nowMs() - tpr0;
                if (!fr) { ++s_asNull; continue; }                     // key gone → skip, no stale pose
                auto ks = g_keySlot.find(rc.key);
                if (ks == g_keySlot.end()) continue;
                ++s_asEmitted;
                if (rc.kind == 0) {
                    emitStaticDraw(ks->second, *fr, drawCount);
                } else {
                    const float* w = fr->worldTransformD3D;
                    const float cx = fr->boundsCenter[0], cy = fr->boundsCenter[1], cz = fr->boundsCenter[2];
                    const float wx = cx*w[0] + cy*w[4] + cz*w[8]  + w[12] - DistantLand::eyePos.x;
                    const float wy = cx*w[1] + cy*w[5] + cz*w[9]  + w[13] - DistantLand::eyePos.y;
                    const float wz = cx*w[2] + cy*w[6] + cz*w[10] + w[14] - DistantLand::eyePos.z;
                    alphaCands.push_back({ wx*fwdX + wy*fwdY + wz*fwdZ, &ks->second, fr, nullptr });
                }
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
            // Same run-length shape, same reason: each frame counted here is a frame a despawned
            // part WOULD have hung in full view, so the run length IS the pre-fix ghost duration —
            // and it should read as a fraction of the 30-frame sweep interval that used to own it.
            // A run that never ends means something in view is permanently condemned: look there
            // before believing the fix.
            static std::uint32_t s_ghostRunFrames = 0, s_ghostRunMax = 0;
            if (ghostSkips > 0) {
                ++s_ghostRunFrames;
                if (ghostSkips > s_ghostRunMax) s_ghostRunMax = ghostSkips;
            } else if (s_ghostRunFrames > 0) {
                LOG::logline("[ghost] in-view despawn run ended: %u frames (pre-fix linger), max %u draws/frame",
                             s_ghostRunFrames, s_ghostRunMax);
                s_ghostRunFrames = 0;
                s_ghostRunMax = 0;
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
            // Bring-up diagnostic: name the set (what the alpha list actually IS — if the
            // in-game "sorted" the eye notices isn't in here, it's an AT3 leftover, not a draw
            // bug). Re-arms whenever the COUNT changes, not once per session: the list at the
            // first frame that has any alpha is the loading-screen scene, and the part being
            // asked about is routinely an actor that streams in later (the NPC belt did exactly
            // that — absent from the frame-0 dump, present in every frame after). Session-capped
            // so a scene whose particle count flickers cannot flood the log.
            static std::size_t s_lastAlphaCount = (std::size_t)-1;
            static unsigned    s_alphaDumps = 0;
            if (alphaCands.size() != s_lastAlphaCount && s_alphaDumps < 20u) {
                s_lastAlphaCount = alphaCands.size();
                ++s_alphaDumps;
                // Printed in DRAW order, which is the sorted order — back to front. An item that
                // paints over something it should sit behind is one whose row is BELOW the other's
                // while its depth says it is further away.
                LOG::logline(">> [alpha-cap] %zu blended shapes this frame (%zu captured AT3), draw order:",
                             alphaCands.size(), g_capRecs.size());
                const std::size_t nDump = (alphaCands.size() < 24u) ? alphaCands.size() : 24u;
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
        g_lightGoboScratch.clear();
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
            // Exact under ExactPos kRigid, like the rigid entries the light shades.
            MGE::ExactPos::rel(w.posRadius, pl.worldPos, pl.worldPosExact,
                               MGE::ExactPos::on(MGE::ExactPos::kRigid));
            w.posRadius[2] += zLift;
            w.posRadius[3] = pl.radius;
            w.color[0] = diffuse[0] * pointLightMult;
            w.color[1] = diffuse[1] * pointLightMult;
            w.color[2] = diffuse[2] * pointLightMult;
            if (pl.fixture) { flags |= IPC::kLightFlagFixture; }   // ESM fixture → host shadow-priority boost
            if (pl.carried) { flags |= IPC::kLightFlagCarried; }   // player's held light → host renders it dynamic
            w.color[3] = IPC::packLightIdFlags(id, flags);   // P2 identity lane (shader ignores .w)
            w.falloff[0] = falloff[0];
            w.falloff[1] = falloff[1];
            w.falloff[2] = falloff[2];
            w.falloff[3] = 0.0f;
            const std::size_t at = g_lightScratch.size();
            g_lightScratch.resize(at + sizeof(w));
            memcpy(g_lightScratch.data() + at, &w, sizeof(w));
            g_lightGoboScratch.push_back({ pl.goboIdHash, pl.goboRot });
            ++count;
        }
        // G3: the gobo side array rides the SAME blob, after the PointLightWire[] (geomwire.h
        // LightGoboWire): lightBytes = count * 56 tells the host it is there.
        if (count > 0) {
            const std::size_t at = g_lightScratch.size();
            const std::size_t sz = g_lightGoboScratch.size() * sizeof(IPC::LightGoboWire);
            g_lightScratch.resize(at + sz);
            memcpy(g_lightScratch.data() + at, g_lightGoboScratch.data(), sz);
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

    // P2b — WHICH OF THE FIVE SKY ELEMENTS IS THIS.
    //
    // The base-map NAME is the only thing MW's sky subtree carries that identifies an element, and
    // it is a reliable one: the shapes are engine-built from a fixed texture set. Verified against
    // the in-game `[sk-diag]` census (14 shapes, exterior, clear):
    //
    //   (none)                  vc=32  order=0      -> DOME    the untextured atmosphere gradient
    //   Tx_Stars*.tga           vc=6..85 order=1..7 -> STARS   incl. Tx_Stars_Nebula*_02.tga
    //   tx_sun_05.dds           vc=4   order=8      -> SUN
    //   tx_mooncircle_full_M|S  vc=4   order=9,11   -> MOON    the shadow layer under each disc
    //   tx_masser_*/tx_secunda_*vc=4   order=10,12  -> MOON    the lit disc
    //   Tx_Sky_Clear.dds        vc=65  order=13     -> CLOUD   Tx_Sky_<weather> per preset
    //
    // ⚠ THE NAMES ARRIVE IN MIXED FORM — bare (`Tx_Stars.tga`) and path-prefixed
    // (`Data Files\Textures\tx_sun_05.dds`) in the SAME census — so every test is a
    // case-insensitive SUBSTRING over the whole string, never a prefix or an equality.
    //
    // ⚠ ORDER MATTERS ONCE, and only once: nothing else here can match a moon, but "secunda"
    // contains "cun" and not "sun", so the SUN test is safe wherever it sits. The moons are still
    // tested first because that is the ordering the reader will assume is load-bearing.
    //
    // ⚠ "sky_" IS ONLY SAFE BECAUSE THIS FUNCTION IS SKY-ONLY. Bloodmoon's whole architecture and
    // terrain set is named `TX_Sky_*` (Tx_Sky_FA_Pine_01, Tx_Sky_crops_01, ...). Those are world
    // meshes and never reach this function — it is called from the sky-list build, over entries the
    // scene walk already flagged isSky. Do not lift this test anywhere else.
    IPC::SkyClass classifySky(const char* texName, bool hasTexture,
                              std::uint32_t vertexCount, std::uint32_t triangleCount) {
        // The dome is the ONE shape with no base map, which is also exactly what makes it the one
        // the Hosek-Wilkie field can replace: it carries no MW art, only a baked gradient.
        if (!hasTexture || !texName) { return IPC::kSkyClassDome; }

        // case-insensitive substring (|32 lowercases ASCII letters; both strings are ASCII).
        auto has = [](const char* hay, const char* needle) -> bool {
            for (const char* h = hay; *h; ++h) {
                const char* a = h;
                const char* b = needle;
                while (*b && *a && ((*a | 32) == (*b | 32))) { ++a; ++b; }
                if (!*b) { return true; }
            }
            return false;
        };

        if (has(texName, "stars") || has(texName, "nebula")) { return IPC::kSkyClassStars; }
        // ⚠ MOONCIRCLE IS TESTED BEFORE THE MOONS AND IS A DIFFERENT CLASS. The moon arrives as a
        // STACK — `tx_mooncircle_full_M` alpha-over at order 9, then `tx_masser_*` ADDITIVELY at
        // order 10 (host [sk-mat], 2026-08-21: blend 5/6 then 5/2). The circle is painted with MW's
        // SKY colour and exists to occlude the stars behind the dark limb; the disc is the moon. One
        // class for both would pin the circle to MW's authored night sky, which is a dark blue over
        // an H-W night sky that is black — reported in play as a dark side that is "a tad brighter".
        if (has(texName, "mooncircle")) { return IPC::kSkyClassMoonShadow; }
        if (has(texName, "masser") || has(texName, "secunda")) {
            return IPC::kSkyClassMoon;
        }
        // The sun keeps the vc==4 && tri==2 quad guard the billboard fix has always applied — it is
        // the shape that gets RE-FACED to the camera, and re-facing anything else would deform it.
        if (vertexCount == 4 && triangleCount == 2 && has(texName, "sun")) { return IPC::kSkyClassSun; }
        if (has(texName, "sky_")) { return IPC::kSkyClassCloud; }
        return IPC::kSkyClassOther;
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
                    // P2b: the CLASS beside the name, so this table is the one place the
                    // classifier can be read against the texture that produced it (verification
                    // step 2 — read once, never again). Same call the packer makes.
                    static const char* kClsName[] = { "OTHER", "DOME", "CLOUD", "SUN", "MOON",
                                                      "STARS", "MOONSHADOW" };
                    const IPC::SkyClass dcls = classifySky(e.textureName, e.d3dTexture != nullptr,
                                                           e.vertexCount, e.triangleCount);
                    LOG::logline(">> [sk-diag]   key=%08X tex=%s order=%u vc=%u cls=%s %s%s",
                                 skey, e.textureName ? e.textureName : "(none)",
                                 (unsigned)e.skyOrder, e.vertexCount,
                                 kClsName[(unsigned)dcls <= 6u ? (unsigned)dcls : 0u],
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
            // P2b: WHAT IS THIS SHAPE. One classification, two consumers — the host's per-class
            // treatment and the sun-disc billboard fix immediately below, which used to run its own
            // copy of the "sun" substring test.
            const IPC::SkyClass cls = classifySky(e.textureName, e.d3dTexture != nullptr,
                                                  e.vertexCount, e.triangleCount);
            item.skyClass = (std::uint32_t)cls;
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
            // ⚠ isSunDisc STAYS A SINGLE FLAG and is NOT widened into skyClass. It has two live
            // consumers that mean "the sun and nothing else" — the reflection re-face
            // (forgerender.cpp) and g_sunDX9Texture, which the proxy rejects MW's own sun draw by —
            // and `if (it.isSunDisc)` would fire for every moon the moment it became an enum. The
            // predicate is unchanged: classifySky applies the same vc==4 && tri==2 && has-texture
            // guard the inline test did.
            const bool isSunDisc = (cls == IPC::kSkyClassSun);
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
            // Publish the sun's real D3D9 texture so inspectIndexedPrimitive can reject MW's own
            // sun draw (the double-sun seam). e.d3dTexture is non-null here — isSunDisc requires it.
            if (isSunDisc) { g_sunDX9Texture = e.d3dTexture; }
            // CAMERA-RELATIVE: the sky is camera-attached; shift by -eye to match the host's
            // translation-free viewProj (see buildDrawList) and keep vertex math near the origin.
            // Same eye as the park pre-cancel in flushAssignAndKick (ExactPos::eye), so the two
            // cancel exactly.
            MGE::ExactPos::rel(&item.world[12], &item.world[12], nullptr, false);
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
                      std::uint32_t& fpAlpha, std::uint32_t& fpMM) {
        MGE_ZoneScopedN("build:fp");
        fpDraws = 0;
        fpSkinned = 0;
        fpAlpha = 0;
        fpMM = 0;
        g_fpDrawScratch.clear();
        g_fpSkinnedScratch.clear();
        g_fpAlphaScratch.clear();
        g_fpMultiMapScratch.clear();
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
        // ExactPos kFP: the arm camera's exact position, if it is still where the FP walk read it
        // (the arm parts were composed at that instant, so camera and parts share one pose).
        double camExact[3] = {};
        const bool haveCamExact = MGE::ExactPos::armCameraExact(pos, camExact);
        float rel[3];
        MGE::ExactPos::rel(rel, pos, camExact, haveCamExact && MGE::ExactPos::on(MGE::ExactPos::kFP));
        MGE::ExactPos::setFPCamera(rel, haveCamExact ? camExact : nullptr);
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
        // Every `continue` below is a part the arms LOSE, and until now they were all silent and
        // indistinguishable in the heartbeat — which is why "some angles drop arm pieces" could not
        // be pinned to the client or the host from a log. Count them by cause; the heartbeat prints
        // the breakdown whenever the composition changes, so a part vanishing names its own reason
        // in the same instant. A drop with EVERY counter zero is the host's, not ours.
        struct FPSkips { std::uint32_t stale, multimap, noslot, blendless, blendSkin, proxy, palette; };
        FPSkips skip = {};
        // How far the furthest shipped FP part sits from the arm camera. A measurement, not a
        // verdict: a threshold I choose can hide the bug (and did — the first one was 256 units,
        // i.e. across the room), whereas a number in the heartbeat cannot. Arms-on-the-arm reads
        // as a few tens of units; anything larger is the artifact, in units, every frame.
        float fpMaxDist2 = 0.0f;
        // Of the multi-map parts shipped, how many are Route C (blended). Its own number because
        // "the blade draws" and "the hilt draws" are different questions — see the walk below.
        std::uint32_t fpMMBlend = 0;
        for (std::uint32_t fkey : fpSet) {
            auto cit = cacheMap.find(fkey);
            if (cit == cacheMap.end()) continue;   // set ⊆ cache invariant; guard anyway
            const auto& e = cit->second;
            if (!e.isFP) continue;
            if (e.lastFrame != cacheFrame) { ++skip.stale; continue; }
            auto ks = g_keySlot.find(fkey);
            if (ks == g_keySlot.end()) { ++skip.noslot; continue; }   // not uploaded yet (first sight)
            // FP1e: multi-map FP parts (dark/detail/glow siblings) ride their OWN list and the
            // host's FP multi-map PSO pair. Classified exactly as buildGeometryDrawLists' dispatch
            // classifies them, and the ORDER of the three tests below IS that dispatch's order: a
            // SKINNED part takes the skinned path whatever sibling maps it carries (the cache walk
            // uploads it as a skinned mesh, never as a wide-VB multi-map one, so the host would
            // reject it), and a TEXTURELESS part has no base stage to composite.
            //
            // ⚠ THIS GATE USED TO DROP EVERY ONE OF THEM — "none expected on arms". A glass weapon
            // carries a glow map with no enchantment involved, so the assumption was simply wrong,
            // and the cost was not a missing effect but MISSING PIXELS: wantsFPSuppression culls
            // MW's whole arm root, so a part the FP pass declines is a part NOBODY draws. Same
            // shape as MB-1c, one population along. [[feedback_confirm_population_reaches_the_filter]]
            const bool isMM = !e.isSkinned && e.d3dTexture
                            && (e.d3dDark || e.d3dDetail || e.d3dGlow);
            // ⚠ AND THE **BLENDED** ONES RIDE THE SAME LIST, TAGGED. The first FP1e build dropped
            // them ("nothing on the arms is expected here") and the very next look in game came back
            // as *"weapon is partial, only hilt is rendering"* — because a glass dagger's BLADE is a
            // blended multi-map part (`[fp-set] key=37E7C8E0 rigid blend tex=tx_w_crystal_blade_.tga`)
            // while its hilt and fittings are opaque ones. Two drops in a row from the same habit of
            // predicting what an arm carries; the counter is what named it in one line both times.
            //
            // Route C's own list, exactly as the world does it: same scratch, same wire, the
            // kMMDrawFlagBlended bit in drawFlags, and the HOST splits on that flag into an opaque
            // record list and an alpha-stage one. So this costs no second vec and no wire field.
            // Emitted INLINE and unsorted, which is also what the world dispatch does — with more
            // than one blended MM part on a weapon their relative order would be arbitrary, and
            // that is a limitation shared with the world path rather than a new one.
            if (isMM) {
                emitMultiMapDraw(ks->second, e, fpMM, g_fpMultiMapScratch, e.blendEnable);
                if (e.blendEnable) { ++fpMMBlend; }
            } else if (e.blendEnable && !e.isSkinned) {
                if (!e.d3dTexture) { ++skip.blendless; continue; }   // no base to composite
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
            } else if (e.isSkinned) {
                // Skinned blends stay dropped (as in FP1a) — the skinned pipeline has no
                // blend state, so drawing them opaquely would be wrong, not better.
                if (e.blendEnable) { ++skip.blendSkin; continue; }
                // emitSkinnedDraw returns SILENTLY on an unbuilt/short bone palette, which on a
                // per-frame-rebuilt skeleton is a real and invisible way to lose a limb. Ask the
                // same question here so the loss is attributable instead of just absent.
                if (e.skinnedUnsupported || e.numBones == 0
                    || e.bonePalette.size() < (std::size_t)e.numBones * 16) {
                    ++skip.palette;
                    continue;
                }
                emitSkinnedDraw(ks->second, e, fpSkinned, g_fpSkinnedScratch);
            } else {
                // Same textureless rule as dispatch: collision proxies drop; genuine
                // textureless visuals draw opaque on slot 0 (host default white).
                if (!e.d3dTexture && e.isPickRoot) { ++skip.proxy; continue; }
                emitStaticDraw(ks->second, e, fpDraws, g_fpDrawScratch);
            }
            // Distance of this part from the ARM camera; the heartbeat reports the max.
            {
                const float* bt = e.isSkinned && e.bonePalette.size() >= 16
                                  ? e.bonePalette.data() : e.worldTransformD3D;
                const float ox = bt[12] - pos[0], oy = bt[13] - pos[1], oz = bt[14] - pos[2];
                const float d2 = ox * ox + oy * oy + oz * oz;
                if (d2 > fpMaxDist2) { fpMaxDist2 = d2; }
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
                    // ⚠ 1.0, and the `= {}` above is exactly why this line has to exist. The emissive
                    // gain's identity is 1, not 0, because the host's vert multiplies the selected
                    // emissive by it — and on the flame branch (vColSource 1) the selected emissive is
                    // the vertex COLOUR. A zero-initialised gain would therefore render every
                    // first-person flame particle black, not merely un-boosted.
                    item.emissiveGain[0] = item.emissiveGain[1] = item.emissiveGain[2] = 1.0f;
                    item.vColSource = emissiveSat ? 1u : 2u;
                    // Quads are already in absolute world space → identity world minus the
                    // camera-relative eye (matches the world captured path + FP emit helpers).
                    memset(item.world, 0, sizeof(item.world));
                    item.world[0] = item.world[5] = item.world[10] = item.world[15] = 1.0f;
                    const float origin[3] = { 0.0f, 0.0f, 0.0f };
                    MGE::ExactPos::rel(&item.world[12], origin, nullptr, false);
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

        // Heartbeat on CHANGE, not on a 300-frame timer. A part that disappears for the two
        // seconds between timed lines is invisible to the log at exactly the moment it matters,
        // and "rotate until the arm drops, then read the last line" is the whole diagnostic.
        // Timed lines still come through so a steady state is confirmable.
        static unsigned s_hb = 0;
        static std::uint32_t s_lastD = 0xFFFFFFFFu, s_lastS = 0, s_lastA = 0, s_lastM = 0;
        // maxDist joins the change test: the counts were constant all along while parts moved, so
        // composition alone could never have caught this. A 32-unit bucket keeps it from chattering.
        static std::uint32_t s_lastBucket = 0xFFFFFFFFu;
        const std::uint32_t distBucket = (std::uint32_t)(sqrtf(fpMaxDist2) / 32.0f);
        const bool composeChanged = (fpDraws != s_lastD || fpSkinned != s_lastS || fpAlpha != s_lastA
                                     || fpMM != s_lastM || distBucket != s_lastBucket);
        s_lastBucket = distBucket;
        if (composeChanged || s_hb % 300 == 0) {
            // mmDrawn is the FP1e lane and `blend=` its Route C half; mm= stays the SKIP counter it
            // always was, so the fix reads as `mm=0 mmDrawn=5(blend=1)` rather than as a number that
            // merely moved. mm= should now be 0 in every scene — a multi-map FP part has a lane
            // whatever it is, so a nonzero value there means a NEW cause, not this one.
            LOG::logline(">> [fp] draws=%u skinned=%u alpha=%u mmDrawn=%u(blend=%u) | skips stale=%u mm=%u noslot=%u "
                         "blendless=%u blendskin=%u proxy=%u palette=%u | set=%u maxDist=%.0f | "
                         "cam fov=%.1f near=%.1f far=%.1f vp=%.0fx%.0f",
                         fpDraws, fpSkinned, fpAlpha, fpMM, fpMMBlend,
                         skip.stale, skip.multimap, skip.noslot, skip.blendless,
                         skip.blendSkin, skip.proxy, skip.palette, (unsigned)fpSet.size(),
                         sqrtf(fpMaxDist2),
                         cd[0], cd[1], cd[2], cd[3], cd[4]);
            // ...and NAME the set. Counts alone cannot answer either of the two questions this
            // pass actually gets asked. "Why is part X missing" needs the roster to show X is
            // absent (a shape pruned in walk() never becomes an entry, so every skip counter
            // stays 0 while the part is gone — that is exactly how the candle's silver body
            // hid). "Why is X drawn over the world" needs it to show X is PRESENT, because the
            // FP pass clears depth and draws last, so anything wrongly in this set paints over
            // the finished frame no matter where it sits. Roster only on a composition change,
            // never on the 300-frame tick — it is one line per part.
            if (composeChanged) {
                for (std::uint32_t fkey : fpSet) {
                    auto cit = cacheMap.find(fkey);
                    if (cit == cacheMap.end()) continue;
                    const auto& fe = cit->second;
                    LOG::logline(">> [fp-set] key=%08X %-7s%s%s tex=%s",
                                 fkey,
                                 fe.isSkinned ? "skinned" : "rigid",
                                 fe.blendEnable ? " blend" : "",
                                 fe.lastFrame != cacheFrame ? " STALE" : "",
                                 fe.textureName ? fe.textureName : "(none)");
                }
            }
        }
        ++s_hb;
        s_lastD = fpDraws; s_lastS = fpSkinned; s_lastA = fpAlpha; s_lastM = fpMM;

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
    // can run them at different points: dev-key poll + finish/copy at
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
        // F11: live composite toggle. Also fired by a one-shot `mge_seam_toggle` file in the game
        // folder (deleted on use), so an unattended MWSE fixture can take the MW-vs-host A/B —
        // GetAsyncKeyState cannot be reached from tes3.tapKey. Checked twice a second.
        bool seamFile = false;
        if ((g_frameSerial & 31u) == 0u && GetFileAttributesA("mge_seam_toggle") != INVALID_FILE_ATTRIBUTES) {
            seamFile = DeleteFileA("mge_seam_toggle") != 0;
        }
        if ((GetAsyncKeyState(VK_F11) & 0x0001) || seamFile) {
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
            // 12 (shadow-atlas dynamic) fullscreen atlas blits; sun shadows added 13 (moments
            // cascade atlas); SH2 sky AO added 14 (the top-down world height map); S2a added the two
            // atmosphere LUT overlays 15/16 and did NOT move this modulus, which made them
            // unreachable until M1 found it; M1 added 17 (motion vectors) and 18 (reactive mask);
            // PBR materials added 19 (gradient source: baked vs 8-bit) and 20 (terrain blend
            // count: how many land textures meet at a pixel) — cycle is %21.
            // THIS MODULUS AND THE HOST'S kDebugModeNames MUST MOVE IN THE SAME COMMIT: the host
            // indexes that array with the value we send here, and a mode with no name is a garbage
            // char* straight into ImGui's dev panel (an instant AV, recorded in forgerender.cpp).
            // ...and 21 (terrain height AO) had ALREADY drifted the same way 15/16 did — its view
            // was in terrain.frag with no name and no modulus, so it was unreachable. 22 (terrain
            // filter width) and 23 (terrain texture size) land with both, which is what the
            // paragraph above asks for. Cycle is %28 — 27 (PBR specular only) lands with a name
            // here AND an entry in kDebugModeNames in the same commit, which is the whole of what
            // the paragraph above asks for and what nobody did the three times it drifted.
            // 28 (near/far producer: DL magenta, near green) lands with its host name — cycle %29.
            g_debugMode = (g_debugMode + 1) % 29;
            const char* name = (g_debugMode == 1) ? "DEPTH" : (g_debugMode == 2) ? "SCATTER"
                             : (g_debugMode == 3) ? "AO" : (g_debugMode == 4) ? "BENT NORMAL"
                             : (g_debugMode == 5) ? "ALBEDO" : (g_debugMode == 6) ? "LIT"
                             : (g_debugMode == 7) ? "AMBIENT" : (g_debugMode == 8) ? "WORLD NORMAL"
                             : (g_debugMode == 9) ? "LIGHT COUNT"
                             : (g_debugMode == 10) ? "SHADOW MASK"
                             : (g_debugMode == 11) ? "SHADOW ATLAS (STATIC)"
                             : (g_debugMode == 12) ? "SHADOW ATLAS (DYN)"
                             : (g_debugMode == 13) ? "SUN MOMENTS"
                             : (g_debugMode == 14) ? "SKY HEIGHT MAP"
                             // 15/16 existed in the host from S2a and were UNREACHABLE: the modulus
                             // above stayed at 15 and the host's kDebugModeNames stayed at 15 entries,
                             // so setDebugMode() clamped both to 14. Corrected alongside mode 17.
                             : (g_debugMode == 15) ? "ATMOS SKY-VIEW LUT"
                             : (g_debugMode == 16) ? "ATMOS TRANSMITTANCE LUT"
                             : (g_debugMode == 17) ? "MOTION VECTORS"
                             : (g_debugMode == 18) ? "REACTIVE MASK"
                             : (g_debugMode == 19) ? "PBR GRADIENT SOURCE"
                             : (g_debugMode == 20) ? "TERRAIN BLEND COUNT"
                             : (g_debugMode == 21) ? "TERRAIN HEIGHT AO"
                             : (g_debugMode == 22) ? "TERRAIN FILTER WIDTH"
                             : (g_debugMode == 23) ? "TERRAIN TEXTURE SIZE"
                             : (g_debugMode == 24) ? "TERRAIN ANISOTROPY"
                             : (g_debugMode == 25) ? "PARALLAX UV DELTA"
                             : (g_debugMode == 26) ? "TERRAIN HEIGHT BLEND"
                             : (g_debugMode == 27) ? "PBR SPECULAR"
                             : (g_debugMode == 28) ? "NEAR/FAR PRODUCER" : "NORMAL";
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
        // Numpad 0: capture the NEXT host frame with RenderDoc, bracketed by the host itself.
        // Not RenderDoc's own hotkey: the host has no Present of its own (it hands a shared
        // texture back and WE present it, a frame or two later), so a UI-triggered capture lands
        // on whatever frame RenderDoc guessed at — never the one on screen. Needs renderdoc.dll in
        // the HOST process: launch via the RenderDoc UI with child-process capture, or set
        // MGE_RDOC=1. The host logs loudly to mgeHost64.log if it is not attached.
        if (GetAsyncKeyState(VK_NUMPAD0) & 0x0001) {
            g_gpuCapturePending = 1u;
            LOG::logline(">> [seam] GPU frame capture requested (numpad 0) — see mgeHost64.log");
        }
        // Numpad 1: dump the host's LINEAR scene target (fp16, pre-exposure, pre-curve) as an EXR,
        // plus the composited frame as a TGA, into hdrdump/ in the install dir. Latched on the same
        // edge and consumed into the same DevInput as numpad 0, for the same reason: the arm has to
        // reach the host BEFORE the frame it is meant to read. Needs a scene-referred (fp16 + MSAA)
        // build — at 1x or LDR there is no linear target to dump and the host says so and declines.
        if (GetAsyncKeyState(VK_NUMPAD1) & 0x0001) {
            g_hdrDumpPending = true;
            LOG::logline(">> [seam] linear HDR frame dump requested (numpad 1) — see mgeHost64.log");
        }
        // FP1b: numpad-/ toggles MW first-person arm suppression live (A/B of MW arms
        // over the host FP pass vs host arms alone). Only takes effect while the FP
        // pass ships (wantsFPSuppression gates on capture + camera validation).
        if (GetAsyncKeyState(VK_DIVIDE) & 0x0001) {
            g_fpSuppressLive = !g_fpSuppressLive;
            LOG::logline(">> [seam] FP suppression (FP1b) %s", g_fpSuppressLive ? "ON" : "OFF");
        }
        // Numpad *: frame-ahead pipelining live A/B. Takes effect at the
        // next kickoff; a pending deferred frame still collects normally (the collect keys
        // on g_kick.deferFinish, not this flag), so the toggle can never wedge the window.
        // Cycles OFF -> 1-ahead -> 1.5-ahead (copy at blit) -> OFF. A copy already parked when
        // g_copyAtBlit drops still drains at the blit or a backstop (drainPendingCopy keys on
        // g_pendingCopy.valid, not this flag).
        if (GetAsyncKeyState(VK_MULTIPLY) & 0x0001) {
            setPipeMode((pipeMode() + 1) % 3);   // OFF -> 1-ahead -> 1.5-ahead -> OFF
        }
        // (The NUMPAD8 produce-worker mode cycle lived here — a bring-up knob from before PARK
        // became the shipping default. Deleted 2026-08-02: one stray keypress cycled the mode to
        // OFF (inline), which DEADLOCKS against the async frame-split. Inline produce runs inside
        // the async RenderFrame window, so every geom upload it needs is refused by the
        // clobber guard and re-queued into the same forbidden window forever — observed 6988
        // consecutive "keeping N bytes for retry" with zero progress, image frozen, game live and
        // unrecoverable without a restart. The mode still exists behind the imgui combo; nothing
        // reaches it by accident any more.)
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
        bool ok;            // finish AND copy both succeeded (finish only, when copyDeferred)
        bool copyDeferred;  // 1.5-ahead: finished, copy parked in g_pendingCopy (fence/rtSlot below)
        double hostMs, overlap, tWait0, tRender, tCopy;
        std::uint64_t frameFence;
        std::uint32_t rtSlot;
    };
    bool drainPendingCopy();   // fwd (defined after accumFrameStats)
    // ks = the kickoff state being finished: g_kick for a same-frame (fused/async) finish,
    // g_pendingFinish for a frame-ahead deferred finish — by then the worker owns g_kick.
    // deferCopy (1.5-ahead): stop after the finish RPC and hand back {frameFence, rtSlot} for the
    // caller to park; g_mainTex is left holding the previous frame until drainPendingCopy.
    FinishResult finishAndCopy(KickState& ks, bool deferCopy = false) {
        FinishResult r = {};
        const bool earlyFinished = ks.rpcEarlyFinished;   // mid-walk drain closed the window
        if (!ks.rpcPending && !earlyFinished) {
            return r;
        }
        // A parked copy is always of an OLDER frame than this one: land it first, so g_mainTex is
        // never overwritten backwards and its RT is read before the host's next use of that slot.
        drainPendingCopy();
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
        // Tier 1: the shared frame-fence value for the frame we are about to copy. The RPC reply no
        // longer implies the host's GPU is done, so this is what copyHostRtToDst waits on.
        std::uint64_t frameFence = 0;
        std::uint32_t rtSlot = 0;   // P3: the host RT (and frame event) this frame used
        if (earlyFinished) {
            r.ok     = ks.earlyOk;
            r.hostMs = ks.earlyHostMs;
            hostT    = ks.earlyHostTimings;   // captured at the mid-walk drain, not re-read here
            frameFence = ks.earlyFenceValue;  // ...and neither is the fence value (same reason)
            rtSlot     = ks.earlyRtSlot;      // ...nor the slot
        } else {
            MGE_ZoneScopedN("Forge renderSceneFinish (host wait)");
            markMainPhase(MP_FINISH_COPY);   // main is now blocked on the host IPC finish
            r.ok = g_client->renderSceneFinish(&r.hostMs, &hostT, &frameFence, &rtSlot);
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
        // work buys little. Names carry (N-1) because under frame-ahead the drained frame is
        // one behind MW's current frame — the plot point lands on the frame that CONSUMED it.
        g_lastHostTimings = hostT;
        // A slow HOST frame, split. [spike] fires on the CLIENT's feed, so a host-bound frame with a
        // small feed (the frame after a crossing: feed 12 ms, RT copy waiting 43 ms) never said where
        // the host's time went. One line per frame whose host total or GPU frame reached 20 ms.
        if (r.ok && (hostT.totalMs >= 20.0f || hostT.gpuFrameMs >= 20.0f)) {
            LOG::logline("!! [host-spike] client frame %u: host total=%.2f gpuWait=%.2f | cpu setup=%.2f cull=%.2f record=%.2f post=%.2f"
                         " | gpu frame=%.2f cull=%.2f prepass=%.2f shadow=%.2f postdepth=%.2f reflect=%.2f color=%.2f water=%.2f resolve=%.2f",
                         g_frame, hostT.totalMs, hostT.gpuWaitMs,
                         hostT.cpuSetupMs, hostT.cpuCullMs, hostT.cpuRecordMs, hostT.cpuPostMs,
                         hostT.gpuFrameMs, hostT.gpuCullMs, hostT.gpuPrepassMs, hostT.gpuShadowMs,
                         hostT.gpuPostDepthMs, hostT.gpuReflectMs, hostT.gpuColorMs, hostT.gpuWaterMs,
                         hostT.gpuResolveMs);
        }
        // T3: does the host own the world's terrain this frame? Only then may MW stop drawing its
        // own (buildGeometryDrawLists' dropMWLand). r.ok gates it because a FAILED finish leaves
        // hostT default-constructed — reading a zeroed block as "not owned" is the safe direction
        // anyway (MW keeps drawing), but keying on r.ok makes that explicit rather than incidental.
        {
            const bool owns = r.ok && hostT.terrainOwned > 0.5f;
            if (owns != g_hostOwnsTerrain) {
                g_hostOwnsTerrain = owns;
                LOG::logline(">> [seam] host terrain ownership %s — MW near land %s",
                             owns ? "ON" : "OFF", owns ? "SUPPRESSED" : "restored");
            }
        }
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
        if (deferCopy) {
            r.copyDeferred = true;
            r.frameFence   = frameFence;
            r.rtSlot       = rtSlot;
            r.tCopy        = r.tRender;   // restamped with the copy's own duration when it drains
            return r;
        }

        bool copyOk;
        {
            MGE_ZoneScopedN("Forge RT copy");
            copyOk = copyHostRtToDst(frameFence, rtSlot);
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
        const double t1 = nowMs();
        FrameTrace::span("client main", "blit", t0, t1, g_traceCopyFence);
        return t1 - t0;
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
        if (FrameTrace::enabled()) {
            // ks = the kick whose host frame fr collected, so all of these carry that frame's fence.
            const std::int64_t fv = g_traceCopyFence;
            if (ks.dtPresent > 0.0) {
                FrameTrace::span("client frames", "MW frame", ks.tStart - ks.dtPresent, ks.tStart, fv);
            }
            FrameTrace::span("client main", "build",       ks.tStart,     ks.tBuild,     fv);
            FrameTrace::span("client main", "geom flush",  ks.tBuild,     ks.tGeomFlush, fv);
            FrameTrace::span("client main", "kick",        ks.tGeomFlush, ks.tKick,      fv);
            FrameTrace::span("client RPC",  "host frame in flight", ks.tKick, fr.tRender, fv);
            FrameTrace::span("client main", "finish wait", fr.tWait0,     fr.tRender,    fv);
        }
        g_hb.copy += (fr.tCopy - fr.tRender); g_hb.evwait += g_lastEventWaitMs; g_hb.blit += blitMs; g_hb.dt += ks.dtPresent;
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
                         "overlap=%.2f copy=%.2f[evw=%.2f] blit=%.2f dt=%.2f mwstart=%.2f early=%u pipe=%u(atblit=%u) refuse=%u cap=%.1f | max feed=%.2f dt=%.2f (~%.0f fps)",
                         g_hb.n, g_hb.feed / g_hb.n, g_hb.geom / g_hb.n, g_hb.build / g_hb.n,
                         g_hb.render / g_hb.n, g_hb.host / g_hb.n, g_hb.overlap / g_hb.n,
                         g_hb.copy / g_hb.n, g_hb.evwait / g_hb.n, g_hb.blit / g_hb.n, g_hb.dt / g_hb.n,
                         g_hb.mwstart / g_hb.n, g_hb.earlyN,
                         g_hb.pipeN, g_hb.atBlitN, g_client ? g_client->windowRefusals() : 0u,
                         g_hb.captured / g_hb.n,
                         g_hb.maxFeed, g_hb.maxDt,
                         g_hb.dt > 0.0 ? 1000.0 * g_hb.n / g_hb.dt : 0.0);
            FrameTrace::dump("mgeXE-trace.csv", "client");
            // white= is the count of captured-alpha DIPs in THIS window whose GPU texture had no
            // registered source name and drew as bindless slot 0 (host default white — an opaque
            // white quad). The [alpha-cap] log for it is capped at 20 per session, so only this
            // delta can distinguish a one-off warm-up from a per-frame strobe (a flip-book VFX
            // whose textures are not being registered fast enough shows up here and nowhere else).
            static std::uint32_t s_capNoNamePrev = 0;
            const std::uint32_t noNameWindow = g_capNoName - s_capNoNamePrev;
            s_capNoNamePrev = g_capNoName;
            LOG::logline(">> [alpha-dedup] %u frames: drops/frame avg=%.2f min=%u max=%u "
                         "(captured/frame=%.1f) white=%u (%llu session) "
                         "-- fluctuating min!=max => dedup is eating live particle draws",
                         g_hb.n, (double)g_capDedupWindow / g_hb.n,
                         g_capDedupDropsMin == ~0u ? 0u : g_capDedupDropsMin, g_capDedupDropsMax,
                         g_hb.captured / g_hb.n,
                         noNameWindow, (unsigned long long)g_capNoName);
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
            // H2 texture-residency occupancy (external audit PR #2 items 1/2). slots= is how much
            // of the client's bindless range is claimed; once it saturates every new texture costs
            // an LRU recycle + a g_texEpoch bump (re-resolve of every cached slot), so a climbing
            // recycles= is a real per-frame cost and not just bookkeeping. thrash= MUST stay 0 —
            // non-zero means the frame's working set does not fit and textures are drawing white.
            // Read the residency state under g_texResidencyMx — the worker mutates g_texSlot
            // concurrently with main, and an unlocked .size() on a rehashing map is exactly the
            // race that corrupted the heap once already (see the g_texResidencyMx comment).
            {
                std::size_t resident = 0, freeSlots = 0, streamQueue = 0;
                std::uint32_t nextSlot = 0, epoch = 0;
                std::uint64_t recycles = 0, thrashes = 0, evictions = 0, evictedBytes = 0;
                std::uint64_t placeholders = 0, deferredData = 0, streamed = 0, streamedBytes = 0;
                // Stale-slot histogram. slots= only ever CLIMBS (residency is cumulative all
                // session), so on its own it cannot distinguish "the working set really is this
                // big" from "we are hoarding textures nothing has sampled in minutes". The LRU
                // already stamps g_slotLastUsed[slot] = g_frame on every reference, so the age
                // distribution is free to read and answers exactly that.
                //
                // ⚠ Ages are only meaningful against the SAME clock: g_frame is the LRU clock and
                // the stamp is written under this mutex, so both are read here, together, rather
                // than sampling g_frame outside the lock and racing a concurrent stamp.
                //
                // Buckets are the timescales that mean different things at ~60 fps:
                //   >1800f  (~30 s)  a texture the current area stopped using
                //   >18000f (~5 min) a texture from an area we have LEFT, still holding a slot
                // A large hot count with a small stale count means the set is genuinely big and
                // eviction will not help; the reverse means it will.
                std::uint32_t stale30s = 0, stale5m = 0, hot = 0;
                std::uint32_t frameNow = 0;
                {
                    std::lock_guard<std::mutex> lk(g_texResidencyMx);
                    resident = g_texSlot.size();
                    nextSlot = g_nextTexSlot;
                    epoch    = g_texEpoch;
                    recycles = g_texRecycles;
                    thrashes = g_texThrashes;
                    evictions = g_texEvictions;
                    evictedBytes = g_texEvictedBytes;
                    freeSlots = g_texFreeSlots.size();
                    streamQueue = g_texStreamQueue.size();
                    placeholders = g_texPlaceholders;
                    deferredData = g_texDeferredData;
                    streamed = g_texStreamed;
                    streamedBytes = g_texStreamedBytes;
                    frameNow = g_frame;
                    // Slot 0 is the host default white and is never LRU-tracked, so start at 1. A
                    // free (evicted) slot holds nothing, so it is no age class at all.
                    for (std::uint32_t s = 1; s < nextSlot && s < g_slotLastUsed.size(); ++s) {
                        if (g_slotName[s].empty()) { continue; }
                        const std::uint32_t age = frameNow - g_slotLastUsed[s];
                        if (age > 18000u)     { ++stale5m; }
                        else if (age > 1800u) { ++stale30s; }
                        else                  { ++hot; }
                    }
                }
                LOG::logline(">> [hb] tex residency: slots=%u/%u (free %zu) resident=%zu | recycles=%llu"
                             " thrash=%llu epoch=%u | age hot=%u stale30s=%u stale5m=%u"
                             " | evicted %llu = %.0f MB this session",
                             nextSlot, IPC::kMaxTextures - IPC::kDlReserve, freeSlots, resident,
                             (unsigned long long)recycles, (unsigned long long)thrashes, epoch,
                             hot, stale30s, stale5m, (unsigned long long)evictions,
                             (double)evictedBytes / (1024.0 * 1024.0));
                LOG::logline(">> [hb] tex stream: placeholders %llu, param maps deferred %llu | streamed %llu"
                             " = %.0f MB | queued now %zu",
                             (unsigned long long)placeholders, (unsigned long long)deferredData,
                             (unsigned long long)streamed, (double)streamedBytes / (1024.0 * 1024.0),
                             streamQueue);
            }
            // MORROWIND.EXE'S OWN MEMORY (tasks/forge-memory-shape.md, alarms 3 + 4). The Windows
            // "GPU Process Memory" counter showed this process holding 2355 MB beside the host's
            // 4595 MB, and nothing in either log could say of what. Two meters:
            //   process      WDDM's per-process accounting through the SYSTEM dxgi.dll (loaded by
            //                full System32 path: DXVK is loaded explicitly as d3d9_dxvk.dll and must
            //                never answer this), so it counts every API this process allocates
            //                through, Vulkan included. The adapter is the one where THIS process holds
            //                the most — a hybrid laptop also enumerates its iGPU — and it is only
            //                chosen once that is nonzero, so an early heartbeat cannot pin the wrong one.
            //   MW textures  the proxy's own ledger: what Morrowind created, in GPU bytes.
            // What neither covers (MGE's DX9 distant land and targets, DXVK's own) is the difference.
            {
                static IDXGIAdapter3* s_mwAdapter = nullptr;
                if (!s_mwAdapter) {
                    typedef HRESULT (WINAPI *PFN_CreateDXGIFactory1)(REFIID, void**);
                    static PFN_CreateDXGIFactory1 s_create = nullptr;
                    static bool s_looked = false;
                    if (!s_looked) {
                        s_looked = true;
                        wchar_t sys[MAX_PATH] = {};
                        const UINT n = GetSystemDirectoryW(sys, MAX_PATH);
                        if (n > 0 && n < MAX_PATH - 12) {
                            wcscat_s(sys, L"\\dxgi.dll");
                            if (HMODULE dx = LoadLibraryW(sys)) {
                                s_create = (PFN_CreateDXGIFactory1)GetProcAddress(dx, "CreateDXGIFactory1");
                            }
                        }
                    }
                    IDXGIFactory1* fac = nullptr;
                    if (s_create && SUCCEEDED(s_create(__uuidof(IDXGIFactory1), (void**)&fac)) && fac) {
                        UINT64 best = 0;
                        IDXGIAdapter1* a1 = nullptr;
                        for (UINT i = 0; fac->EnumAdapters1(i, &a1) != DXGI_ERROR_NOT_FOUND; ++i) {
                            IDXGIAdapter3* a3 = nullptr;
                            if (SUCCEEDED(a1->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&a3)) && a3) {
                                DXGI_QUERY_VIDEO_MEMORY_INFO q = {};
                                if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &q))
                                    && q.CurrentUsage > best) {
                                    best = q.CurrentUsage;
                                    if (s_mwAdapter) { s_mwAdapter->Release(); }
                                    s_mwAdapter = a3;
                                    a3 = nullptr;
                                }
                                if (a3) { a3->Release(); }
                            }
                            a1->Release();
                        }
                        fac->Release();
                    }
                }
                // The envelope: Morrowind's world draw is a no-op under Forge, so what it needs is
                // the local map's textures and its UI. With the map cap and without the DX9 distant
                // land it measured 408 MB fresh (2026-09-20; it was 1608 before either); play adds a
                // few hundred MB of its own caches. MGE_MEM_MW_MB overrides.
                static uint32_t s_mwEnvMB = 0;
                if (s_mwEnvMB == 0) {
                    s_mwEnvMB = 640u;
                    char e[32] = {};
                    if (GetEnvironmentVariableA("MGE_MEM_MW_MB", e, sizeof(e)) > 0) {
                        const unsigned long v = std::strtoul(e, nullptr, 10);
                        if (v > 0) { s_mwEnvMB = (uint32_t)v; }
                    }
                }
                DXGI_QUERY_VIDEO_MEMORY_INFO q = {};
                const bool okQ = s_mwAdapter != nullptr &&
                    SUCCEEDED(s_mwAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &q));
                const ProxyTexLedger& L = proxyTexLedger();
                const uint64_t liveB = L.liveBytes.load(), bigB = L.bigBytes.load(), midB = L.midBytes.load();
                const uint32_t liveN = L.liveCount.load(), bigN = L.bigCount.load(), midN = L.midCount.load();
                const double kMB = 1024.0 * 1024.0;
                char proc[64];
                if (okQ) {
                    std::snprintf(proc, sizeof(proc), "%llu/%llu MB", (unsigned long long)(q.CurrentUsage >> 20),
                                  (unsigned long long)(q.Budget >> 20));
                } else {
                    std::snprintf(proc, sizeof(proc), "unavailable");
                }
                LOG::logline(">> [hb] mw mem: process local=%s | MW textures live %u = %.0f MB"
                             " (>=4MB %u = %.0f MB, 1-4MB %u = %.0f MB, <1MB %u = %.0f MB)"
                             " | created this session %u = %.0f MB",
                             proc, liveN, (double)liveB / kMB, bigN, (double)bigB / kMB,
                             midN, (double)midB / kMB, liveN - bigN - midN,
                             (double)(liveB - bigB - midB) / kMB,
                             L.createdCount.load(), (double)L.createdBytes.load() / kMB);
                // The map cap's receipt. filled < capped (after a load settles) means Morrowind fills
                // some textures by a route the cap does not redirect, and those are blank on its map.
                if (g_proxyTexCapDim != 0) {
                    const uint32_t capN = L.capCount.load(), capF = L.capFilled.load();
                    LOG::logline(">> [hb] mw cap %u: capped %u (filled %u) saved %.0f MB this session"
                                 " | dropped-level writes %u locks, %u surfaces",
                                 g_proxyTexCapDim, capN, capF, (double)L.capSavedBytes.load() / kMB,
                                 L.capScratchLocks.load(), L.capScratchSurfaces.load());
                    static uint32_t s_unfilledWarned = 0;
                    if (capN > capF + 16 && s_unfilledWarned < 4) {
                        ++s_unfilledWarned;
                        LOG::logline("!! [mwcap] %u of %u capped textures never had a kept level written"
                                     " — Morrowind fills them some other way; they are BLANK on its map",
                                     capN - capF, capN);
                    }
                }
                if (okQ && (q.CurrentUsage >> 20) > s_mwEnvMB) {
                    LOG::logline("!! [mem] Morrowind.exe holds %llu MB of VRAM (envelope %u MB) — its own"
                                 " textures are %.0f MB of it (%u at 2048^2+), and under Forge its world draw"
                                 " is a no-op: they serve the local map",
                                 (unsigned long long)(q.CurrentUsage >> 20), s_mwEnvMB,
                                 (double)liveB / kMB, bigN);
                }
            }
            // EcoQoS state of THIS process (Morrowind.exe). The seam's `copy=` bucket is almost
            // entirely FlushRenderingCommands, which is CPU work on this thread, so an execution-
            // speed throttle here would inflate it while leaving every GPU timestamp in the host's
            // `gpu split` untouched. That is precisely the signature of the sticky ~5.8x flush
            // regime change, and nothing currently logged can rule it in or out.
            //
            // ⚠ Tri-state on purpose: "unmanaged" and "managed, off" both mean full speed, but a
            // FAILED query must NOT read as "not throttled" — that is the one case this exists for.
            // Logged every heartbeat rather than once, because the regime FLIPS mid-session (it was
            // observed recovering on its own), so a startup-only reading would be worse than none.
            {
                // GetProcessInformation is gated behind a newer _WIN32_WINNT than this project
                // compiles with; resolved from kernel32 at runtime so a logging probe does not
                // move the SDK floor for the whole client.
                typedef BOOL (WINAPI *PFN_GetProcessInformation)(HANDLE, PROCESS_INFORMATION_CLASS, LPVOID, DWORD);
                static PFN_GetProcessInformation s_getProcInfo = nullptr;
                static bool s_looked = false;
                if (!s_looked) {
                    s_looked = true;
                    if (HMODULE k32 = GetModuleHandleW(L"kernel32.dll")) {
                        s_getProcInfo = (PFN_GetProcessInformation)GetProcAddress(k32, "GetProcessInformation");
                    }
                }
                // -1 = no such export, -(GetLastError()) = the call refused. Distinguishing them
                // matters: the first says "wrong Windows", the second says "wrong call", and they
                // want opposite fixes. One undifferentiated "?" is undiagnosable, which is exactly
                // what the first build of this probe produced.
                PROCESS_POWER_THROTTLING_STATE pt = {};
                pt.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
                int eco = -1;
                if (s_getProcInfo) {
                    SetLastError(0);
                    if (s_getProcInfo(GetCurrentProcess(), ProcessPowerThrottling, &pt, sizeof(pt))) {
                        eco = (pt.ControlMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED)
                            ? ((pt.StateMask & PROCESS_POWER_THROTTLING_EXECUTION_SPEED) ? 1 : 0)
                            : 0;
                    } else {
                        const DWORD e = GetLastError();
                        eco = e ? -(int)e : -1;
                    }
                }
                // fg=1 when the foreground window belongs to THIS process. Compared by owning
                // PID rather than against a stored HWND so it needs no handle from the proxy and
                // stays correct if MW recreates its window. This is the input to the throttle
                // question: EcoQoS normally spares a foreground app, so "fg=1 ecoqos=THROTTLED"
                // and "fg=0" are very different stories about the same slow frame.
                DWORD fgPid = 0;
                GetWindowThreadProcessId(GetForegroundWindow(), &fgPid);
                char ecobuf[32];
                if (eco == 1)       { _snprintf_s(ecobuf, sizeof(ecobuf), _TRUNCATE, "THROTTLED"); }
                else if (eco == 0)  { _snprintf_s(ecobuf, sizeof(ecobuf), _TRUNCATE, "off"); }
                else if (eco == -1) { _snprintf_s(ecobuf, sizeof(ecobuf), _TRUNCATE, "?(noexport)"); }
                else                { _snprintf_s(ecobuf, sizeof(ecobuf), _TRUNCATE, "?(err=%d)", -eco); }
                LOG::logline(">> [hb] proc: ecoqos=%s prio=0x%lx fg=%d",
                             ecobuf,
                             (unsigned long)GetPriorityClass(GetCurrentProcess()),
                             (fgPid && fgPid == GetCurrentProcessId()) ? 1 : 0);
            }
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

    // 1.5-ahead: a finished host frame whose RT copy waits for the blit point (see g_copyAtBlit).
    // At most one, and always the newest finished frame: set only by the park collect, drained
    // before any later finish consumes a newer frame (finishAndCopy). Its host RT slot is safe until
    // the host records the frame after next, which needs a finish first — so the drain always wins.
    // Main thread only (the copy is D3D9/DXVK work).
    struct PendingCopy {
        bool          valid = false;
        std::uint64_t frameFence = 0;
        std::uint32_t rtSlot = 0;
        KickState     ks = {};      // the kick it finished, for the [hb] accounting at the copy
        FinishResult  fr = {};
    };
    PendingCopy g_pendingCopy;
    void dropPendingCopy() { g_pendingCopy = PendingCopy{}; }

    // Copy the parked frame into g_mainTex. Called at the blit (its home) and by every backstop:
    // the next finish, collectDeferredFinish (load / reset / non-early / menu-freeze frames), and
    // Present when no UI scene blitted. No-op when nothing is parked. Returns false on a failed copy.
    bool drainPendingCopy() {
        if (!g_pendingCopy.valid) {
            return true;
        }
        g_pendingCopy.valid = false;
        if (!g_initOk) {
            return false;   // seam torn down under it: nothing to copy into
        }
        MGE_ZoneScopedN("Forge RT copy (at blit)");
        const double t0 = nowMs();
        const bool ok = copyHostRtToDst(g_pendingCopy.frameFence, g_pendingCopy.rtSlot);
        const double copyMs = nowMs() - t0;
        if (!ok) {
            static bool logged = false;
            if (!logged) { LOG::logline("!! [seam] copyHostRtToDst failed (deferred copy)"); logged = true; }
            g_mainTexValid = false;
            return false;
        }
        g_mainTexValid = true;
        // The [hb] copy bucket is tCopy - tRender: restamp it to the copy's own wall time, so feed and
        // copy mean what they mean in 1-ahead (the MW work between finish and copy is not the copy).
        FinishResult fr = g_pendingCopy.fr;
        fr.tCopy = fr.tRender + copyMs;
        ++g_hb.atBlitN;
        accumFrameStats(g_pendingCopy.ks, fr, fr.tCopy, true, g_lastBlitMs);
        g_lastBlitMs = 0.0;
        return true;
    }

    // Cell-change shadow eviction (see g_cellEpoch): bump the epoch on any load-door
    // transition BEFORE buildLightList so the cleared identity map hands every new-cell light a
    // fresh id. Signal = the engine's authoritative interior-cell pointer (covers
    // interior<->interior and interior<->exterior) OR a single-frame eye teleport (covers
    // exterior<->exterior load doors / fast travel, which keep the interior pointer null).
    // Runs once per produced frame: from kickoffBody on the serial/inline paths, from
    // fireParked (main, frame start) in produce mode 3 — never both in one frame.
    void contentProbeWindowEnd(unsigned frame);   // geometry content probe (defined with captureGeometry)

    void checkCellEpochAndPurge(unsigned frame) {
        void* dh = MGE::SceneGraph::getDataHandler();
        const void* interiorCell = dh ? MGE::DataHandlerView::currentInteriorCell(dh) : nullptr;
        const float eye[3] = { DistantLand::eyePos.x, DistantLand::eyePos.y, DistantLand::eyePos.z };
        static const void* s_lastInteriorCell = nullptr;
        static float       s_lastEye[3] = { 1e30f, 1e30f, 1e30f };
        static bool        s_haveEye = false;
        const float ex = eye[0] - s_lastEye[0], ey = eye[1] - s_lastEye[1], ez = eye[2] - s_lastEye[2];
        const bool teleport = s_haveEye && (ex*ex + ey*ey + ez*ez > kCellTeleportDist * kCellTeleportDist);
        // SAVE-GAME RELOAD. Neither signal above fires when you reload a save of where you already
        // are: the interior-cell pointer is unchanged and the eye lands back on the same spot — yet
        // MW tore the scene graph down and rebuilt it, so every cached entry is keyed on a dead
        // shape address while the new scene captures alongside it. That is the reported "reload →
        // frozen duplicate NPCs", and the stale skinned entries (dead bone palettes) are the
        // "broken skinning" in the same report — one root cause, two symptoms.
        //
        // The signal is MW's loading bar, latched for us by frameSetupEarly (see noteLoadingBar):
        // this function runs only on PRODUCED frames and a load produces none, so reading the flag
        // here would never catch it up. g_sawLoadingBar is sticky — consumed, not sampled — so the
        // purge fires on the first produced frame after the load however long that takes.
        const bool reloaded = g_sawLoadingBar;
        g_sawLoadingBar = false;
        if (interiorCell != s_lastInteriorCell || teleport || reloaded) {
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
            LOG::logline(">> [cell-purge] epoch=%u frame=%u interiorChanged=%d teleport=%d reloaded=%d first=%d cached=%u",
                         g_cellEpoch, frame, (int)(interiorCell != s_lastInteriorCell), (int)teleport,
                         (int)reloaded, (int)firstEval, (unsigned)MGE::GeometryCache::cache().size());
            g_texSyncFrame = g_frame;   // this build's first sights load full files (see g_texSyncFrame)
            contentProbeWindowEnd(frame);   // MGE_GEOM_CONTENT_PROBE: the window up to this load
            aliasWindowEnd(frame);          // geometry dedup: the same window
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
                // BULK release at a load: the whole old cell, in this build's blob AHEAD of the new
                // cell's parts (the build below appends those). Trickled 64 per flush, the old cell's
                // arena ranges were still held when the new cell uploaded, so the host doubled the
                // arena instead of reusing them (stress: 320 -> 640 MB, 15403 releases queued). The
                // host processes a chunk in order and reclaims parked ranges before it grows
                // (forgerender reclaimArenaRetired), so the new cell lands in the old one's space.
                // A load frame is allowed to be late; the per-release host cost is paid here once.
                {
                    const std::size_t queued = g_pendingReleaseSlots.size();
                    appendBoundedReleaseRecords(queued);
                    LOG::logline("-- [release] load: %zu releases shipped ahead of the new cell", queued);
                }
                texBookkeepingRelease("load");
            } else {
                // First load: no purge, so no post-load residency window either — yet this needs one
                // as much as a transition does. Whatever the pre-seam loading-screen walks left in
                // the cache (see cached= above), from HERE on the cache fills purely from the
                // engine's frustum-limited classify set, so everything behind the camera that those
                // walks missed arrives as first-sight keys on the first 180 turn. Arm the window
                // without purging: there is no old cell to flush, only a new one to make resident.
                MGE::GeometryCache::armPostLoadWalk();
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
                            std::uint32_t fpSkinnedDraws, std::uint32_t fpAlphaDraws,
                            std::uint32_t fpMMDraws, bool fpHave,
                            const double bakeEye[3], bool parkFired,
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
        std::uint32_t fpDraws = 0, fpSkinnedDraws = 0, fpAlphaDraws = 0, fpMMDraws = 0;
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
            fpHave = buildFPFrame(fpFrame, fpDraws, fpSkinnedDraws, fpAlphaDraws, fpMMDraws);
            prefetchGridTextures();    // budgeted against this frame's first sights, so before streaming
            streamPendingTextures();   // after every resolver of the frame has queued its first sights
        }
        const double tBuild = nowMs();

        // Everything above wrote only client-private scratch; the shared-memory half
        // (gate → flushes → assigns → constants → renderSceneKickoff) is the extracted
        // flushAssignAndKick, shared verbatim with the mode-3 park fire. bakeEye == eyePos
        // here ⇒ the camera restamp inside is the exact identity (byte-identical constants).
        const double* eyeNow = MGE::ExactPos::eye();
        const double bakeEye[3] = { eyeNow[0], eyeNow[1], eyeNow[2] };
        flushAssignAndKick(device, frame, drawCount, skinnedCount, multiMapCount, lightCount,
                           skyCount, alphaCount, fpFrame, fpDraws, fpSkinnedDraws, fpAlphaDraws,
                           fpMMDraws, fpHave, bakeEye, /*parkFired=*/false, dtPresent, tStart, tBuild);
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
                            std::uint32_t fpSkinnedDraws, std::uint32_t fpAlphaDraws,
                            std::uint32_t fpMMDraws, bool fpHave,
                            const double bakeEye[3], bool parkFired,
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
        releaseMorrowindDroppedTextures();   // mirror first: what Morrowind freed goes now
        evictStaleTextures();   // after the geometry flush (see its comment), before the tex flush
        const std::uint32_t texCount = g_texPendingCount;          // snapshot (flush clears it)
        const std::size_t   texBytes = g_texPendingBytes;
        {
            markWorkerPhase(WK_TEX);
            MGE_ZoneScopedN("Forge tex flush");
            flushTextures();
        }
        const double tTexFlush = nowMs();
        markWorkerPhase(WK_ASSIGN);

        // PARK-LAG CORRECTION — same shape as the sky pre-cancel below, for the same reason, and
        // it must run BEFORE the assigns: everything under here is shared memory the host reads.
        //
        // The camera restamp re-aims the whole payload with THIS frame's camera, which plants every
        // shipped transform at the world position it held at BUILD time. Right for the world, wrong
        // for the player: the camera is welded to them, so their one-frame staleness is the only
        // lag in the frame that is fully correlated with where you are looking — the body slides
        // off centre while you move and slots back when you stop.
        //
        // The delta comes from the player NODE, not the eye. In 3rd person the camera also ORBITS
        // a standing player, and an eye delta would shove the body sideways on a pure mouse-look —
        // a new artifact in place of the old one. Pose is deliberately NOT corrected: the limbs are
        // one frame stale and stay that way, because that error is small and uncorrelated.
        //
        // On the serial paths build and fire are the same instant, so the delta is exactly zero and
        // every byte matches the pre-correction code.
        if (!g_playerPatch.empty() && g_playerBakeValid) {
            double now[3];
            if (MGE::GeometryCache::playerRootOrigin(now)) {
                const float dx = static_cast<float>(now[0] - g_playerBake[0]);
                const float dy = static_cast<float>(now[1] - g_playerBake[1]);
                const float dz = static_cast<float>(now[2] - g_playerBake[2]);
                if (dx != 0.0f || dy != 0.0f || dz != 0.0f) {
                    for (const PlayerPatch& p : g_playerPatch) {
                        std::vector<std::uint8_t>* s =
                            (p.scratch == 0) ? &g_drawScratch     :
                            (p.scratch == 1) ? &g_skinnedScratch  :
                            (p.scratch == 2) ? &g_multiMapScratch : &g_alphaScratch;
                        // A scratch is only rebuilt wholesale, never truncated between build and
                        // fire, so an out-of-range offset would mean the two disagree about the
                        // payload — skip rather than write past it.
                        const std::size_t need = (std::size_t)p.at
                                               + (std::size_t)(p.count - 1) * 64 + 3 * sizeof(float);
                        if (p.count == 0 || need > s->size()) continue;
                        auto* t = reinterpret_cast<float*>(s->data() + p.at);
                        for (std::uint32_t i = 0; i < p.count; ++i) {
                            t[i * 16 + 0] += dx;
                            t[i * 16 + 1] += dy;
                            t[i * 16 + 2] += dz;
                        }
                    }
                }
            }
        }

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

        // The sky is CAMERA-ANCHORED, and that makes it the one payload the camera restamp
        // below gets wrong. buildSkyDrawList shipped each shape as (skyWorld - bakeEye), and
        // the restamp folds (bakeEye - eyeNow) into the view translation — which resolves to
        // (skyWorld - eyeNow), i.e. the sky planted at the world position it occupied at BUILD
        // time. That is exactly right for statics (they really are world-anchored) and exactly
        // wrong here: MW re-centres the sky on the camera every frame, so a mode-3 park fire
        // renders it around last frame's camera and it slides by the frame's camera delta —
        // the sky visibly lagging translation while the world tracks it (reported in-game,
        // 2026-07-23). Re-anchor to the fire-time eye by pre-cancelling the restamp: the
        // shipped (skyWorld - bakeEye) then survives verbatim into view space, which is what
        // "stationary, fixed to the camera" means. Rotation was never affected — the restamp
        // uses R_now for everything, so the sky turns with the current camera either way.
        //
        // On the serial paths bakeEye == eyePos, so the delta is 0 and this is a no-op.
        {
            const double* eyeNow = MGE::ExactPos::eye();
            const float ex = static_cast<float>(eyeNow[0] - bakeEye[0]);
            const float ey = static_cast<float>(eyeNow[1] - bakeEye[1]);
            const float ez = static_cast<float>(eyeNow[2] - bakeEye[2]);
            // ...and tell the host the same delta. The pre-cancel below makes the sky camera-attached
            // in the MAIN view, but the reflect pass mirrors it about "z = 0 camera-relative" — a
            // plane that, like everything in the payload, is BAKE-relative. The mirror therefore
            // re-introduces this delta DOUBLED (a reflection doubles any plane offset), and the
            // reflected sky steps by 2*ez whenever the camera's height changes between bake and fire:
            // walking a slope, stairs, a jump. Set unconditionally and from the SAME three
            // subtractions the pre-cancel uses, so the host's correction and the client's can never
            // describe different deltas. See bridge.h's skyParkEyeDelta.
            if (g_client) { g_client->setSkyParkEyeDelta(ex, ey, ez); }
        }
        if (skyCount > 0 && !g_skyScratch.empty()) {
            const double* eyeNow = MGE::ExactPos::eye();
            const float ex = static_cast<float>(eyeNow[0] - bakeEye[0]);
            const float ey = static_cast<float>(eyeNow[1] - bakeEye[1]);
            const float ez = static_cast<float>(eyeNow[2] - bakeEye[2]);
            if (ex != 0.0f || ey != 0.0f || ez != 0.0f) {
                const std::size_t n = g_skyScratch.size() / sizeof(IPC::SkyDrawWire);
                auto* sky = reinterpret_cast<IPC::SkyDrawWire*>(g_skyScratch.data());
                for (std::size_t i = 0; i < n; ++i) {
                    sky[i].world[12] += ex;
                    sky[i].world[13] += ey;
                    sky[i].world[14] += ez;
                }
            }
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
            // FP1e: the multi-map FP parts (glow/dark/detail siblings on a held weapon).
            if (fpMMDraws > 0 && g_fpMultiMapVec
                && g_fpMultiMapVec->assign_bytes(g_fpMultiMapScratch.data(), (std::uint32_t)g_fpMultiMapScratch.size())) {
                fpFrame.mmList  = g_fpMultiMapVec->id();
                fpFrame.mmCount = fpMMDraws;
                fpFrame.mmBytes = (std::uint32_t)g_fpMultiMapScratch.size();
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
        //
        // THE ARMS ARE SCENE DATA. This listed only the WORLD payloads, so standing where no world
        // geometry is in view — out past the last object, looking into flat fog — emptied all five
        // and skipped the seam. "Let MW present its own frame" is not a safe fallback there: MW's
        // world draws are suppressed at the reject gate AND FP1b force-culls the arm root, so what
        // MW presents is a frame with no arms in it, while the composite stops advancing. That is
        // the reported freeze — the world stops, the arms vanish, and MW goes on animating a
        // first-person skeleton nobody draws. A frame holding nothing but arms is a real frame.
        const bool haveFP = fpHave && (fpDraws > 0 || fpSkinnedDraws > 0 || fpAlphaDraws > 0
                                       || fpMMDraws > 0);
        if (!haveDraw && skinnedId == IPC::InvalidVector && multiMapId == IPC::InvalidVector
            && skyId == IPC::InvalidVector && alphaId == IPC::InvalidVector && !haveFP) {
            // Silent until now, which is why an empty-payload skip and a genuinely stalled produce
            // looked identical from the log. Run-length: a skip lasting one frame is a load door,
            // a skip lasting hundreds is a player standing in the void.
            ++g_seamSkipRun;
            if (g_seamSkipRun == 1 || (g_seamSkipRun % 60) == 0) {
                LOG::logline(">> [seam] skip %u: no payload (draw=%u skin=%u mm=%u sky=%u alpha=%u fp=%u)",
                             g_seamSkipRun, drawCount, skinnedCount, multiMapCount, skyCount,
                             alphaCount, fpDraws + fpSkinnedDraws + fpAlphaDraws + fpMMDraws);
            }
            return;
        }
        if (g_seamSkipRun > 0) {
            LOG::logline(">> [seam] resumed after %u skipped frame(s)", g_seamSkipRun);
            g_seamSkipRun = 0;
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
            const double* eyeNow = MGE::ExactPos::eye();
            const float dx = static_cast<float>(bakeEye[0] - eyeNow[0]);
            const float dy = static_cast<float>(bakeEye[1] - eyeNow[1]);
            const float dz = static_cast<float>(bakeEye[2] - eyeNow[2]);
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
        // ─── ...AND THE SAME PAIR WITH MGE'S OWN PER-WEATHER LOOK MULTIPLIERS LEFT OUT ──────────
        //
        // ⚠ Configuration.Lighting.SunMult/AmbMult ("Cloudy Sun Brightness" and its nine siblings,
        // MGEgui's Per Pixel Lighting page) are a LOOK, not a measurement — a legacy DX9-era dial
        // for shaping how bright each weather reads. They belong on the LIGHT, which is what the
        // DX9 path draws with and what the host still lights the NIGHT from. They must never reach
        // the host's EXPOSURE SETPOINT, which asks a different question: "how much light did MW put
        // in this frame?" A look dial answering a measurement question is a camera that re-exposes
        // itself every time the weather changes.
        //
        // ⚠⚠ AND IT DID EXACTLY THAT, FOR AS LONG AS THE MW-REFERRED SETPOINT HAS EXISTED. This
        // install ships Cloudy at sun 1.60 / amb 1.35 and every other non-clear weather at sun 0.00.
        // The host's reference (forgerender.cpp, mwRefLevel) latched the PRODUCT off lighting[4..10],
        // so the servo aimed 1.435x day in Cloudy and 0.398x in Overcast against 0.996x in Clear —
        // 1.5 stops of weather-driven exposure swing, off an ini table that no longer draws a single
        // pixel in the Forge path (the physical sky overwrites sunCol/ambCol wholesale at
        // blend=1/ramp=1). "clear weather is acceptable, but cloudy is so washed out" and "overcast
        // is so dark" are three rows of that table, and clear was acceptable because its two entries
        // are 1.00.
        //
        // ⚠ THE UNSCALED PAIR MUST BE CARRIED, NOT DIVIDED BACK OUT ON THE HOST. SunMult is 0.00 for
        // eight of this install's ten weathers, and a product with a zero in it does not remember
        // its other factor. So the reference rides its own two lanes ([38..39]).
        RGBVECTOR sunColRef = DistantLand::sunCol;
        RGBVECTOR ambColRef = DistantLand::sunAmb + DistantLand::ambCol;
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
                sunColRef = liveSun;                       // ...unscaled; see the decl
                // sgSunlight.ambient ALREADY carries the cell ambient in interiors (in-game
                // verified: it equals the cell record's ambientColor byte-for-byte), so there it IS
                // the whole ambient term and adding the cell record on top would double it. Exteriors
                // keep their existing composition (sun ambient + the D3DRS_AMBIENT global ambCol) —
                // same formula as the captured path, just with a live sun ambient.
                ambColEff = isExterior ? (DistantLand::lightAmbMult * (liveSunAmb + DistantLand::ambCol))
                                       : (DistantLand::lightAmbMult * liveSunAmb);
                ambColRef = isExterior ? (liveSunAmb + DistantLand::ambCol) : liveSunAmb;
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
            } else {
                // The SILENT branch, and the one that matters here: with no live sgSunlight the sun
                // and ambient stay on the CAPTURED values, which refresh only on SetLight(6) /
                // D3DRS_AMBIENT — i.e. only while MW is drawing something lit. A static interior
                // reached from another interior can starve that capture completely, leaving the
                // whole frame lit by whatever was last captured, or by nothing at all. Geometry lit
                // by nothing is geometry you cannot see, which is why dropping a light into the
                // room "brings the missing pieces back": they were never missing, they were black.
                static std::uint32_t s_noSunEpoch = ~0u;
                static unsigned s_noSunN = 0;
                if (g_cellEpoch != s_noSunEpoch || (s_noSunN++ % 900 == 0)) {
                    s_noSunEpoch = g_cellEpoch;
                    LOG::logline("!! [light] getSceneSunlight FAILED (%s) — falling back to CAPTURED: "
                                 "sun=(%.3f %.3f %.3f) sunAmb=(%.3f %.3f %.3f) ambCol=(%.3f %.3f %.3f) "
                                 "-> ambient=(%.3f %.3f %.3f)",
                                 isExterior ? "exterior" : "interior",
                                 DistantLand::sunCol.r, DistantLand::sunCol.g, DistantLand::sunCol.b,
                                 DistantLand::sunAmb.r, DistantLand::sunAmb.g, DistantLand::sunAmb.b,
                                 DistantLand::ambCol.r, DistantLand::ambCol.g, DistantLand::ambCol.b,
                                 ambColEff.r, ambColEff.g, ambColEff.b);
                }
            }
        }
        // ─── THE SUN THAT SETS (tasks/forge-physical-sky.md P2) ──────────────────────────────────
        // ⚠ MW HAS TWO SUNS AND THEY ARE NOT THE SAME SUN. They share an AZIMUTH exactly and run
        // DIFFERENT ELEVATION ARCS, so the split is signed, varies through the day, and passes
        // through zero — one frame can therefore "disprove" it by accident. A full session traced
        // (2026-08-20, the user's own play log; raw disc elevation, i.e. before the set-correction):
        //
        //     LIGHT elev   41.09  19.92  13.88  14.24  17.25  21.42  44.72  15.34  17.86
        //     DISC  elev   69.89  25.96   0.32   1.91  15.09  31.52  73.10   6.78  17.66
        //     split       +28.80  +6.04 -13.56 -12.33  -2.16 +10.10 +28.38  -8.56  -0.20
        //
        // The LIGHT never leaves 13.8..53.1 degrees, which is exactly the range of MW's fixed sun
        // transit (-400*orbit, 75, 100): its horizontal term can never fall below 75, so the light
        // is CLAMPED to a shallow arc by construction and physically cannot rise higher or set. The
        // DISC sweeps 73 degrees on both sides of the horizon. They are two different curves that
        // happen to CROSS twice a day, and a measurement taken at a crossing (the -0.20 above, from
        // an hour-4.76 harness run) says nothing about the rest of it.
        //
        // For a physical sky the DISC is authoritative: Hosek-Wilkie is parameterised by solar
        // elevation and its circumsolar brightening has to be centred on the sun you can actually
        // SEE, or the bright region of the sky sits tens of degrees away from the sun drawn in it.
        // The `split=` field on the line below keeps this a measured quantity rather than a
        // remembered one.
        //
        // ✅ AND THE SAME TRACE SETTLES THE DUSK QUESTION THIS PORT COULD NOT ANSWER FROM SOURCE.
        // The raw disc elevation reaches 0.32 degrees and then turns back up (0.32 -> 1.91) — MW
        // bounces the DISC at the horizon too, and the correction below un-bounces it. Because the
        // turning point IS the horizon, the correction acts on an elevation of ~0 and the sky does
        // NOT snap: measured 0.32deg at the last frame before the flip, 1.91deg at the first frame
        // after. Continuous to within a couple of degrees, which the night ramp then covers.
        //
        // ⚠⚠ AND MW'S SUN LIGHT DOES NOT SET — IT BOUNCES AT THE HORIZON. Once the sun goes down MW
        // mirrors the light direction back above the horizon, deliberately, so night lighting comes
        // from a plausible direction instead of from underground. Its reported elevation therefore
        // stays POSITIVE all night, which for a model parameterised by elevation means a DAYTIME sky
        // at midnight. Reported from play as "MW's sun doesn't set, so it is daytime all the time".
        //
        // MGE has always carried the SHAPE of the fix, in two lines of DistantLand::setView — keep
        // the LIGHT source bouncing exactly as MW intends (that is the look P2's lighting blend
        // interpolates against, and it must not move) and carry a SECOND direction that actually
        // sets, by mirroring the disc's z once the sun is down:
        //     sunVis = GetSunVis() / 255;   if (sunVis == 0) sunPos.z = -sunPos.z;
        //
        // ⚠⚠⚠ BUT `sunVis` IS NOT A NIGHT TEST. It is the sun disc's own MATERIAL ALPHA
        // (MWBridge::eSunVis walks shTriSunBase -> property -> material colours, +3 for the alpha
        // byte; mwbridge.h calls it "sun(glare) alpha value" and means it literally). MW fades that
        // alpha to zero whenever it stops drawing the disc — and OVERCAST, RAIN, THUNDER, ASH,
        // BLIGHT and BLIZZARD are all exactly that. So the test fires at NOON IN THE RAIN, the z
        // mirrors while the sun is high, and a physical sky parameterised by solar elevation goes to
        // NIGHT in daylight. Reported from play as "overcast/rainy weathers are also black".
        //
        // It stayed invisible in the DX9 path for twenty years because every DX9 consumer of the
        // corrected sunPos is itself gated on sunVis (distantland.cpp:966 `if (sunVis >= 0.001)`,
        // :1039 multiplies EV_sunvis in), so the frames where the correction is wrong are precisely
        // the frames nothing reads it. This is the first consumer that does not gate on sunVis, and
        // it is therefore the first thing that could ever see the fault.
        //
        // THE NIGHT TEST IS THE CLOCK, not anything weather touches:
        //     nightStart = sunsetHour + sunsetDuration     nightEnd = sunriseHour
        // which is the very window MW itself uses to pick which branch of the sun's fixed transit to
        // place the disc on, so the sign we impose and the position MW drew agree by construction.
        // Read through MGE::WorldControllerView::sunAboveHorizon(), which owns the arithmetic; see
        // the ⚠ block there for the offsets and the OpenMW cross-reference. An elevation threshold
        // still cannot substitute, for the original reason: the bounce makes "+5 degrees" mean
        // either dawn or dusk-plus-five with nothing to separate them.
        //
        // The RAW GetSunDir() is re-read here rather than DistantLand::sunPos, so that MGE's own
        // (wrong) correction is never in the chain — and DistantLand::sunPos is deliberately LEFT
        // ALONE, because MW's atmospheric-scattering fog adjustment reads its z unguarded
        // (distantland.cpp:873-885) and writes the result back into MW's scenegraph fog colour,
        // which this client then ships as fogColNear. Fixing it there would move the F11 DX9
        // baseline AND the host's fog in one step, on a frame that is supposed to be a control.
        //
        // ⚠ SHIPPED AS ELEVATION + AZIMUTH, NOT AS A VECTOR, because the wire has exactly two free
        // padding floats (sunDir.w and ambCol.w, neither read by any FSL) and a direction needs
        // three. Two ANGLES are exact and reconstruct without ambiguity, where two of three
        // components would leave the third's SIGN underdetermined — and the sign of z is precisely
        // the day/night bit this whole block exists to carry. The host rebuilds
        // (cos(az)*r, sin(az)*r, z), r = sqrt(1 - z*z).
        MWBridge* mwb = MWBridge::get();
        D3DXVECTOR4 sp = DistantLand::sunPos;   // fallback: weatherless cells, pre-load, no schedule
        bool sunUp = false;
        float schedHour = 0.0f, schedNightEnd = 0.0f, schedNightStart = 0.0f;
        const bool haveSched = MGE::WorldControllerView::sunAboveHorizon(
            sunUp, &schedHour, &schedNightEnd, &schedNightStart);
        float sunDiscRawZ = std::max(-1.0f, std::min(1.0f, sp.z));
        if (haveSched && mwb->IsLoaded() && mwb->CellHasWeather()) {
            D3DXVECTOR3 raw;
            mwb->GetSunDir(raw.x, raw.y, raw.z);
            D3DXVec3Normalize(&raw, &raw);
            sunDiscRawZ = std::max(-1.0f, std::min(1.0f, raw.z));
            // Make the SIGN agree with the clock, rather than forcing it — `raw.z = sunUp ? |z| :
            // -|z|` would erase a genuine below-horizon dip in the minutes the schedule still calls
            // day, which is exactly the region where a step would show.
            if (sunUp != (raw.z >= 0.0f)) { raw.z = -raw.z; }
            sp = D3DXVECTOR4(raw.x, raw.y, raw.z, 1.0f);
        }
        const float sunDiscZ  = std::max(-1.0f, std::min(1.0f, sp.z));
        const float sunDiscAz = std::atan2(sp.y, sp.x);
        {
            // Both suns and the window that separates day from night, on one line every 900 frames:
            // the 29-degree elevation split above is a property of MW and this is what keeps it a
            // MEASURED one. `vis` stays on the line as a WITNESS, not as an input — watching it fall
            // to 0 while `up=1` is what a rainstorm looks like from here, and it is the reason this
            // block no longer believes it.
            //
            // ⚠ WHAT THIS STILL HAS TO CONFIRM: that the CLOCK boundary lands on the disc's turning
            // point. MW's own `vis` test empirically did (0.32deg before the flip, 1.91deg after),
            // and that is why the transition was continuous; the clock is a different boundary and
            // only coincides if `sunsetHour + sunsetDuration` is where MW bottoms the arc out. If it
            // is early or late, `disc elev` steps by twice whatever it had left and the sky snaps —
            // so the flip line wants |elev| within a degree or two of 0 on both sides.
            // A once-per-900-frames line cannot photograph a transition that lasts one frame, so the
            // throttle FOLLOWS the question: dense inside a quarter game-hour of either boundary,
            // and edge-triggered on the sign change itself so the flip frame is in the log whatever
            // the counter is doing. Everywhere else it stays a background heartbeat.
            static unsigned s_sunSetLogN = 0;
            static int s_sunUpPrev = -1;
            const int upNow = sunUp ? 1 : 0;
            const bool flipped = (s_sunUpPrev >= 0 && upNow != s_sunUpPrev);
            s_sunUpPrev = upNow;
            // The dense window is measured in ELEVATION, not in game hours, for two reasons: it does
            // not assume the boundary is where the schedule says (which is half of what is being
            // checked), and it is independent of the TIMESCALE — this install runs at ~1, where a
            // quarter game-hour is a quarter of a REAL hour and an hour-based window would emit tens
            // of thousands of lines.
            const bool nearEdge = (std::abs(sunDiscRawZ) < 0.052f);   // |elevation| < 3 degrees
            const unsigned every = nearEdge ? 120u : 900u;
            if (flipped || (s_sunSetLogN++ % every) == 0) {
                const float elevLight = std::asin(std::max(-1.0f, std::min(1.0f, -sunVecEff.z))) * 57.29578f;
                const float elevDisc = std::asin(sunDiscZ) * 57.29578f;
                // `raw` is MW's disc BEFORE the set-correction and `split` is measured against it,
                // because the two-suns question is about the arcs MW itself runs — folding our own
                // sign flip into the difference would make the split jump 2x at every dusk and
                // report the correction back to us as if it were engine behaviour. `raw` reaching 0
                // at the FLIP line is also exactly the continuity check.
                const float elevRaw = std::asin(sunDiscRawZ) * 57.29578f;
                LOG::logline(">> [sun-set]%s hour=%.3f window=[%.2f..%.2f] up=%d sched=%d vis=%.3f"
                             " | LIGHT -sunVec=(%.3f %.3f %.3f) elev=%.2fdeg"
                             " | DISC sunPos=(%.3f %.3f %.3f) elev=%.2fdeg raw=%.2fdeg"
                             " (split=%.2fdeg) -> shipped z=%.4f az=%.2fdeg",
                             flipped ? " FLIP" : "",
                             schedHour, schedNightEnd, schedNightStart, upNow,
                             haveSched ? 1 : 0,
                             mwb->IsLoaded() ? (mwb->GetSunVis() / 255.0f) : 0.0f,
                             -sunVecEff.x, -sunVecEff.y, -sunVecEff.z, elevLight,
                             sp.x, sp.y, sp.z, elevDisc, elevRaw, elevRaw - elevLight,
                             sunDiscZ, sunDiscAz * 57.29578f);
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
        const RGBVECTOR* skyColPtr = mwb->CellHasWeather() ? mwb->getCurrentWeatherSkyCol() : nullptr;
        const float skyZenithR = skyColPtr ? skyColPtr->r : DistantLand::nearFogCol.r;
        const float skyZenithG = skyColPtr ? skyColPtr->g : DistantLand::nearFogCol.g;
        const float skyZenithB = skyColPtr ? skyColPtr->b : DistantLand::nearFogCol.b;
        // Wind VECTOR (lighting[36..37]). Two consumers now, and that is why the whole vector ships
        // rather than the magnitude it used to: the host's flame-flicker rate wants |wind| only (the
        // flicker is isotropic), but G1 grass sways ALONG the wind, so it needs the direction. The
        // host derives the magnitude back off this — one wind on the wire, so the flicker and the
        // grass cannot end up describing different weather. MW's raw vector is very noisy, so it is
        // smoothed the same way DistantLand::update does (EWMA f=0.02).
        // Exterior + weather-cell gated CLIENT-side (IsExterior is authoritative here), so an
        // interior always ships (0,0) and the host needs no exterior gate of its own.
        //
        // ⚠ The EWMA state must NOT be reset when the gate closes: walking into an inn for ten
        // seconds and back out would otherwise restart the filter from zero, and the field would
        // spend the next few seconds accelerating from dead calm into whatever the weather is.
        // Freezing it (the `if` guards the update, not the state) means the wind is simply where it
        // was, which is also what MW's own weather does across a load door.
        static float s_smoothWind[2] = {};
        if (isExterior && mwb->CellHasWeather() && !mwb->IsMenu()) {
            const float* wind = mwb->GetWindVector();
            s_smoothWind[0] += 0.02f * (wind[0] - s_smoothWind[0]);
            s_smoothWind[1] += 0.02f * (wind[1] - s_smoothWind[1]);
        }
        // ⚠⚠ AND THE SHIPPED VALUE DROPS `!IsMenu()`, WHICH THE COMMENT ABOVE ALREADY ARGUED FOR
        // AND THE LINE BELOW IT THREW AWAY. Freezing the EWMA *state* is worth nothing if the value
        // on the wire is zeroed anyway: reported from play as *"grass ambient animation seems to
        // reset position when menu paused"*, which is exactly what a wind of (0,0) looks like — the
        // sway amplitude collapses and every blade springs back to its rest pose, mid-gust.
        //
        // The two halves of the old gate are DIFFERENT QUESTIONS and only look alike:
        //   isExterior && CellHasWeather — "is there any wind here at all". An interior genuinely
        //                                  has none, so (0,0) is the right answer and stays.
        //   !IsMenu()                    — "is the world running". The wind has not gone away
        //                                  because a menu is open; it has stopped CHANGING, which
        //                                  is what freezing the EWMA update above already says.
        // A guard whose fallback is the wrong answer reads exactly like a working guard
        // ([[feedback_guard_fallback_is_the_bug]]). The host holds up its end: the grass harmonics
        // run on MW's sim clock, so a frozen wind and a frozen clock hold the bent pose still.
        const bool  windLive = isExterior && mwb->CellHasWeather();
        const float windVecX = windLive ? s_smoothWind[0] : 0.0f;
        const float windVecY = windLive ? s_smoothWind[1] : 0.0f;
        // Glow in the Dahrk distant windows (lighting[35]): hours into the period where GitD shows a
        // window mesh's lit "on" child (>0 lit, <0 dark). The host adds a per-instance stagger and
        // uses the sign to pick the night or day variant of a distant window subset. Before a world
        // exists there is nothing to light and no distant land either, so fall back to a value that
        // is unambiguously "day" rather than to 0, which sits exactly on the switch boundary.
        float glowMargin = -24.0f;
        MGE::WorldControllerView::glowLitMargin(glowMargin);
        // Enchanted-item glow (lighting[31]): the bindless slot of the caustic frame MW's shared
        // enchant NiTextureEffect is showing RIGHT NOW. The engine cycles magicitem\caust00..31.dds
        // itself, so this is re-resolved every frame; the 32 frames cost 32 of ~870 residency slots
        // once and never churn after that. 0 = nothing enchanted on screen, which is also what the
        // shader reads as "no glow" — so a scene with no enchanted items pays nothing and a failed
        // texture load degrades to the current (glow-less) image rather than to garbage.
        // Warm EVERY frame of the caustic book the first time it appears (and again if MW ever
        // reallocates it — the generation says so). Resolving only the frame currently on screen
        // works, but first-sights all 32 one at a time as the animation plays, and a slot whose DDS
        // upload has not landed yet samples empty — the few-second flicker at load. 32 textures at
        // 32x32 DXT1 is ~21 KB of residency, paid once.
        // And keep the whole book warm WHILE a glow is on screen: stale-texture eviction
        // (evictStaleTextures) releases frames nothing has resolved for a while, and resolving only
        // the frame on screen would first-sight them one at a time again after an absence. 32
        // lookups a frame, only while something enchanted is visible.
        const char* const glowTex = MGE::GeometryCache::enchantGlowTexture();
        {
            const char* const* book = nullptr;
            uint32_t gen = 0;
            const int n = MGE::GeometryCache::enchantGlowBook(book, gen);
            static uint32_t s_warmedGen = 0;
            if (n > 0 && (gen != s_warmedGen || glowTex)) {
                const bool firstSight = gen != s_warmedGen;
                s_warmedGen = gen;
                for (int i = 0; i < n; ++i) resolveTextureSlot(book[i]);
                if (firstSight) {
                    LOG::logline(">> [enchant] warmed %d caustic frames into bindless residency", n);
                }
            }
        }
        const float enchantSlot = float(resolveTextureSlot(glowTex));
        // [7] = "MW's WeatherController tinted these lights underwater" (sunCol.w, a padding slot no
        // FSL reads). The host undoes MW's underwater blend on sun/ambient, and that inverse is only
        // defined where the forward blend was applied — which is a weather operation. A weatherless
        // interior takes sun/ambient straight from sgSunlight (see the live-light block above), so
        // MW never tinted them, and un-blending an untinted value lifts red 5x, or clamps all three
        // to zero in a dark room and freezes the in-scatter at black. Same weather predicate the
        // interior-ambient and skyZenith paths already gate on. Fog is NOT covered by this flag: MW
        // overrides fog underwater weather or not, so the host keeps un-blending that lane ungated.
        const float mwTintsUnderwater = mwb->CellHasWeather() ? 1.0f : 0.0f;
        // [38..39]: the reference pair as ONE LUMA each, because only its ratio to a fixed anchor is
        // ever used (forgerender.cpp, g_mwSunCode: "Latched as ONE luma each"). Rec.709 — these
        // three constants and the host's SceneCal::luma709 are one number in two processes, and a
        // disagreement between them moves the exposure setpoint.
        float sunCodeRef = 0.2126f * sunColRef.r + 0.7152f * sunColRef.g + 0.0722f * sunColRef.b;
        float ambCodeRef = 0.2126f * ambColRef.r + 0.7152f * ambColRef.g + 0.0722f * ambColRef.b;

        // ─── ...NORMALISED BY THIS WEATHER'S OWN DAY ROW ─────────────────────────────────────────
        //
        // *"overcast must be perfect exposure. Day reads dark otherwise."* — user, 2026-09-09, after
        // the look-multiplier fix above landed Overcast at 0.674x day and Cloudy at 0.990x.
        //
        // ⚠ MW AUTHORS A PER-WEATHER LEVEL **BECAUSE MW HAS NO EXPOSURE**. Its whole delivered image
        // is `texCode * (ambCode + sunCode * N.L)` in gamma space with no camera anywhere, so the
        // only way MW can say "it is duller under cloud" is to author the light dimmer. We have a
        // camera AND a physical sky that already delivers the dimming (Overcast measures
        // sunNormal=83 lx against Clear's 109,917) — so tracking MW's weather level on top of that
        // counts it TWICE, and the second count is the one the player sees as "day reads dark".
        //
        // ⚠ THE HOUR IS A DIFFERENT QUESTION AND IT KEEPS TRACKING MW. Dusk genuinely should read
        // dimmer, the eye does not fully adapt across it, and play signed that off explicitly
        // (*"still too bright sunset and sunrise"* -> fixed by exactly this tracking). Weather and
        // hour both come out of the same four authored colours — the weather picks WHICH ROW, the
        // hour picks WHERE IN IT — so they are separable, and this separates them: divide the
        // reference by the CURRENT WEATHER'S OWN DAY ROW and every weather's noon becomes the unit
        // while every dawn and dusk curve inside that weather is untouched.
        //
        // ⚠ SCALED TO THE CLEAR DAY ROW, NOT TO 1, so the host needs no change and CLEAR STAYS
        // BIT-IDENTICAL: for Clear the two ratios are 1.0000 exactly, the expression is the
        // identity, and `calMwDayRef()` on the far side still divides by the same anchor it always
        // has. These two constants ARE `kCalMwAmbDay` / `kCalMwSunDay` in forgerender.cpp — MW's
        // authored [Weather Clear] Ambient/Sun Day Color — and the two copies must agree or noon
        // stops landing on 1.00x. They are written here rather than derived so that the
        // cancellation is visible at both ends.
        //
        // ⚠ GATED ON A LIVE EXTERIOR WEATHER. Interiors have no weather row to normalise by and
        // must not be touched (their multipliers are 1.0 and their setpoint is a different rule
        // entirely); a failed read leaves the pair exactly as computed above.
        {
            constexpr float kMwAmbDayClear = 0.552182f;   // luma of 137,140,160
            constexpr float kMwSunDayClear = 0.986772f;   // luma of 255,252,238
            MWBridge* const mwbRef = MWBridge::get();
            MWBridge::WeatherState wsRef;
            if (isExterior && mwbRef->CellHasWeather() && mwbRef->getWeatherState(wsRef)) {
                const float aDay = 0.2126f * wsRef.ambDayCol.r + 0.7152f * wsRef.ambDayCol.g
                                 + 0.0722f * wsRef.ambDayCol.b;
                const float sDay = 0.2126f * wsRef.sunDayCol.r + 0.7152f * wsRef.sunDayCol.g
                                 + 0.0722f * wsRef.sunDayCol.b;
                // A floor, not a branch-per-channel: a weather whose authored day row is black
                // would otherwise divide the setpoint to infinity. Nothing in vanilla is near it.
                if (aDay > 1.0e-3f) { ambCodeRef *= kMwAmbDayClear / aDay; }
                if (sDay > 1.0e-3f) { sunCodeRef *= kMwSunDayClear / sDay; }
            }
        }

        const float lighting[40] = {
            // [3] = the SUN DISC's elevation sine, MGE's bounce-corrected DistantLand::sunPos.z —
            // a sun that actually SETS, unlike sunVec.xyz beside it, which keeps bouncing because
            // that is the lighting MW intends. Its azimuth rides [11]. Both are padding slots no FSL
            // reads; the host consumes them CPU-side when cooking the physical sky. See above.
            sunVecEff.x,               sunVecEff.y,               sunVecEff.z,               sunDiscZ,
            sunColEff.r,               sunColEff.g,               sunColEff.b,               mwTintsUnderwater,
            // [11] = the sun DISC's azimuth (radians, atan2(y,x)); pairs with [3] to reconstruct the
            // whole direction exactly. Two ANGLES rather than two COMPONENTS because the sign of z
            // is the day/night bit and components would leave it underdetermined.
            ambColEff.r,               ambColEff.g,               ambColEff.b,               sunDiscAz,
            DistantLand::nearFogCol.r, DistantLand::nearFogCol.g, DistantLand::nearFogCol.b, 0.0f,
            // [18] RETIRED (was the smoothed wind MAGNITUDE — G1 moved the wind to [36..37] as a
            // vector and the host derives |wind| from it); [19] = cell epoch. Both land in
            // fogParams.zw, which no shader reads — the host consumes them CPU-side (epoch change →
            // evict all shadow slots + caster records).
            DistantLand::fogNearStart, DistantLand::fogNearEnd,   0.0f,                      float(g_cellEpoch),
            // CAMERA-RELATIVE: WorldPos reaches the shader already relative to the eye, so the
            // eyePos used for the per-vertex fog distance |worldPos - eyePos| is the origin (0).
            0.0f,                      0.0f,                      0.0f,                      0.0f,
            // Phase 1a/1b: the absolute camera eye the payload is RELATIVE to (host shifts
            // resident DL by -this eye — it must match the payload's relative space, so on a
            // mode-3 park fire this is the BUILD-time eye, not the current one; serial paths
            // pass bakeEye == eyePos) + isExterior gate (1 = feed host-owned DL).
            (float)bakeEye[0],         (float)bakeEye[1],         (float)bakeEye[2],         isExterior ? 1.0f : 0.0f,
            // C2 skyZenith (float4 28..31): zenith sky colour for the host dome gradient. Host reads
            // it into FrameData.skyZenith; only sky.frag (dome branch) consumes it.
            // [31] = skyZenith.w, which the dome never used: the enchanted-item glow's caustic
            // bindless slot. It rides HERE because the host already copies floats 28..31 straight
            // into FrameData.skyZenith AND carries that block into the reflection and first-person
            // frame cbuffers — so an enchanted sword glows in its own reflection and in the player's
            // hands off one write, with no wire growth and no new cbuffer field.
            skyZenithR,                skyZenithG,                skyZenithB,                enchantSlot,
            // [32] MW SIMULATION time (seconds this session, frozen in menus) — drives the UV scroll
            // of UV-animated distant statics (ghostfence) in statics.vert. This is the SAME clock MGE's
            // DX9 path feeds its `time` uniform (distantland.cpp:987), so the Forge and MGE fences
            // scroll in step under an F11 A/B. NOT a host wall clock: that keeps running in menus and,
            // being steady_clock-since-BOOT, quantizes to 0.02-0.13s steps once cast to float32 (the
            // bug that made the host's water normals judder — see forgerender.cpp:9150).
            // [33]/[34] Part A upload cost: THIS frame's total host-geom reship KB + part count,
            // surfaced on the host perf panel. Filled from g_upFrame just below (the array is const,
            // so a non-const alias writes the two slots after the aggregate is computed).
            // [35] GitD night signal -> FrameData.timeParams.z (statics.vert day/night window clip).
            mwb->simulationTime(),     0.0f,                      0.0f,               glowMargin,
            // [36..37] G1: MW's WIND VECTOR, EWMA-smoothed above. Drives the host grass lane's four
            // wind harmonics (gShadowParams.grassParams.xy) AND — via the magnitude the host derives
            // from it — the flame-flicker rate that [18] used to carry. One wind, one wire.
            // [38..39] THE EXPOSURE REFERENCE'S OWN LIGHT: sun and ambient as MW authored them,
            // WITHOUT MGE's per-weather look multipliers. Two lanes rather than a divisor because
            // SunMult is 0.00 in eight weathers here — see sunColRef at the top of this function.
            windVecX,                  windVecY,                  sunCodeRef,                ambCodeRef,
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
        // Numpad 0: one-shot host-bracketed RenderDoc capture. Latched at the edge and consumed
        // here so the arm reaches the host BEFORE the renderScene it is meant to capture.
        if (g_gpuCapturePending) {
            devInput.gpuCapture = g_gpuCapturePending;
            g_gpuCapturePending = 0;
        }
        // Numpad 1: one-shot linear scene-target dump (EXR + TGA). Same latch-then-consume as above.
        if (g_hdrDumpPending) {
            devInput.dumpHdr = 1u;
            g_hdrDumpPending = false;
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
        // ⚠ MUST match bridge.h's waterParams[] exactly — client.cpp memcpy's sizeof(destination).
        float waterParams[14] = {};
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
            // [11] MW's GameHour — the host's unified water fog picks its extinction density from
            // MW's own [Water] Underwater*Fog time-of-day values, and the host has no clock that
            // can stand in for this one: FrameData.timeParams.x is sim time wrapped to [0, 12.5)
            // for a UV scroll and .z is the Glow-in-the-Dahrk margin. This lane was the water
            // block's reserved slot, so it costs no wire growth — and it belongs here rather than
            // on a lighting lane because the block is already gated on the cell having water,
            // which is exactly when a water-fog density matters.
            waterParams[11] = mw->getGameHour();
            // [12]/[13] R1 impulse ripples: MW's own live precipitation counters. The engine ramps
            // these across a weather TRANSITION, which is the whole reason to read them rather than
            // the weather TYPE — a type is a step and rain arriving as a step looks like a bug.
            // Sent RAW; the host maps count -> density, because the reference count can only be
            // found by watching a real storm with the dev panel open.
            // Zeroed in interiors: the counters are exterior-only state and a stale one would rain
            // indoors, which is exactly the class of bug [[project_forge_interior_stale_terrain]]
            // was — an ungated lane still reporting the last exterior frame.
            MWBridge::WeatherState ws;
            if (mw->IsExterior() && mw->getWeatherState(ws)) {
                waterParams[12] = (float)ws.rainParticles;
                waterParams[13] = (float)ws.snowParticles;
            }
        }

        // R2: actor ripples straight out of MW's pool. The engine already switches a decal on for
        // every actor moving in water — player, NPC, creature — so this needs no actor detection;
        // it is a read of a list MW maintains anyway. ~1-2us for the whole 75-slot walk.
        //
        // Nearest-to-eye selection rather than a distance CUTOFF: a hard radius makes a wake pop
        // out of existence at the boundary, while taking the closest N degrades by dropping the
        // ripple that was already the smallest on screen. Insertion into a sorted array — N is 32
        // and the input is 75, so the obvious O(n*N) is cheaper than setting up anything cleverer.
        std::uint32_t rippleCount = 0;
        float ripplePack[IPC::kMaxActorRipples * 4] = {};
        {
            MWBridge::RippleState rs;
            MWBridge::RippleSource rsrc[128];
            if (MWBridge::get()->getRippleState(rs, rsrc, 128) && rs.count > 0) {
                float bestD2[IPC::kMaxActorRipples];
                int   bestIx[IPC::kMaxActorRipples];
                for (int i = 0; i < rs.count; ++i) {
                    const float dx = rsrc[i].x - DistantLand::eyePos.x;
                    const float dy = rsrc[i].y - DistantLand::eyePos.y;
                    const float d2 = dx * dx + dy * dy;
                    if (rippleCount == IPC::kMaxActorRipples && d2 >= bestD2[rippleCount - 1]) {
                        continue;
                    }
                    std::uint32_t at = rippleCount;
                    if (at == IPC::kMaxActorRipples) { at = rippleCount - 1; }
                    else { ++rippleCount; }
                    while (at > 0 && bestD2[at - 1] > d2) {
                        bestD2[at] = bestD2[at - 1];
                        bestIx[at] = bestIx[at - 1];
                        --at;
                    }
                    bestD2[at] = d2;
                    bestIx[at] = i;
                }
                // BIRTH DETECTION, riding the `w` lane that used to carry MW's decal scale.
                //
                // The wave sim needs an IMPULSE — a displacement applied once, at the instant a
                // ripple appears. Feeding it every live ripple every frame would drive the surface
                // continuously instead, and the field would saturate into a permanent mound
                // following the actor rather than waves leaving it behind.
                //
                // Detected here and not host-side because identity is a POOL SLOT and only this
                // side has it: the host receives the list already reordered by distance every
                // frame, so it cannot tell "slot 7 was recycled into a new ripple" from "the list
                // shifted". A birth is a slot going inactive->active, or its age running BACKWARDS
                // (MW recycles a live slot straight into a new ripple, which reads as the same
                // ripple getting younger). Same test the [ripl+] probe uses; 0.02 of normalised age
                // is well under one frame of a 3 s life, so it cannot fire on ordinary ageing.
                //
                // scale is not lost by this: it is lerp(0.15, 6.5, age), verified live, so age
                // already carries it.
                constexpr int kRipSlots = 256;
                static float s_age[kRipSlots] = {};
                static bool  s_act[kRipSlots] = {};
                static bool  s_seen[kRipSlots];
                for (int i = 0; i < kRipSlots; ++i) { s_seen[i] = false; }
                for (std::uint32_t k = 0; k < rippleCount; ++k) {
                    const MWBridge::RippleSource& s = rsrc[bestIx[k]];
                    bool born = true;
                    if (s.slot >= 0 && s.slot < kRipSlots) {
                        s_seen[s.slot] = true;
                        born = !s_act[s.slot] || (s.age + 0.02f < s_age[s.slot]);
                        s_act[s.slot] = true;
                        s_age[s.slot] = s.age;
                    }
                    ripplePack[k * 4 + 0] = s.x;
                    ripplePack[k * 4 + 1] = s.y;
                    ripplePack[k * 4 + 2] = s.age;
                    ripplePack[k * 4 + 3] = born ? 1.0f : 0.0f;
                }
                // ⚠ Only slots we actually SHIPPED were refreshed above, and the nearest-N cut
                // drops the far ones — so clear liveness for everything in the POOL that is gone,
                // not just for what made the wire. Otherwise a ripple that fell out of the nearest
                // set and came back would not register as a birth, and one that genuinely died in
                // a slot we stopped shipping would keep its stale age forever.
                for (int i = 0; i < rs.count; ++i) {
                    if (rsrc[i].slot >= 0 && rsrc[i].slot < kRipSlots) { s_seen[rsrc[i].slot] = true; }
                }
                for (int i = 0; i < kRipSlots; ++i) { if (!s_seen[i]) { s_act[i] = false; } }
            }
        }

        // G7 GRASS PLASTICITY: the player's crush disc. The host builds a world-locked clearance
        // field that every skinned bone presses into — a corpse flattens its own silhouette, an NPC
        // presses at the ankles — but the PLAYER has to ride the wire, because IN FIRST PERSON MW
        // submits no skinned draws for the body at all. That is exactly the view where a missing
        // footprint is most obvious, and it is the one case the host's own harvest cannot see.
        //
        // MW's reference point is the actor's FEET, which is the surface we want: the field stores
        // "how low is the lowest occupant surface", and for a walking player that is the ground it
        // is standing on. Radius 0 (the default) means "no disc" — interiors and menus, where there
        // is no grass to press. Harmless double cover in third person: the field is built by `min`,
        // so a second disc over the same ground changes nothing.
        // ⚠ THE CLIENT SAYS WHERE, THE HOST SAYS HOW WIDE. The disc radius is a look and it is
        // already a host knob beside every other crush dial; a second copy here would be the same
        // tuning number on both sides of an IPC boundary.
        if (g_client) {
            // ⚠ SAME SPLIT, SAME REASON, AND IT IS THE SAME BUG ONE SYSTEM OVER. The old gate read
            // `isExterior && !IsMenu()` and the comment above justified it as "interiors and menus,
            // where there is no grass to press" — but a menu is not an interior. The player is
            // still standing on the same grass; they have merely stopped moving. Dropping the disc
            // let the blades under their feet spring upright the moment a menu opened, and the
            // crush field's healing is frozen now, so nothing put them back down.
            const bool crushHere = isExterior;
            g_client->setPlayerCrush(crushHere ? mwb->PlayerPositionX() : 0.0f,
                                     crushHere ? mwb->PlayerPositionY() : 0.0f,
                                     crushHere ? mwb->PlayerPositionZ() : 0.0f,
                                     crushHere);
        }

        // S1 ATMOSPHERE — MW'S WEATHER REACHES THE HOST (tasks/forge-atmosphere.md).
        //
        // ⚠ THIS IS THE MISSING WIRE, not a new feature. MGE has read the weather controller for
        // years and thrown all of it away but two particle counts (see the waterParams block
        // above); the physical sky therefore generates a CLEAR sky from a fixed turbidity slider no
        // matter what the game thinks the weather is, which is the single defect behind "overcast
        // sky is gray from the clouds texture, horizon has hosek's bluer sky". Nothing here changes
        // a pixel — the host reports the row and does not yet render from it.
        //
        // ⚠ THE AUTHORED SCALARS ARE INTERPOLATED HERE, ON PURPOSE. MW blends its weather COLOURS
        // across a transition but reads cloud cover, fog depth and wind off `currentWeather` alone,
        // so those three step at the instant of the swap. The lerp needs BOTH Weather objects and
        // only this side has them (the wire carries one row), so it happens here. The weather INDEX
        // pair rides untouched — the host's per-weather physics table is lerped by the same
        // `transition` on the far side, which is what makes a weather change a continuous walk
        // through parameter space rather than a cross-fade between two pictures.
        //
        // Gated on a real exterior weather cell and not on a menu: `valid = 0` is the interior
        // answer and zeroes the row, the same idiom the sun, the sky-AO and the sky ambient lanes
        // already use, so the host needs no exterior gate of its own and a stale exterior row can
        // never rain indoors.
        if (g_client) {
            IPC::WeatherWire ww = {};
            MWBridge::WeatherState ws;
            if (isExterior && mwb->CellHasWeather() && mwb->getWeatherState(ws)) {
                const float t = std::max(0.0f, std::min(1.0f, ws.transition));
                const auto mix = [t](float a, float b) { return a + t * (b - a); };
                ww.valid            = 1;
                ww.cur              = ws.curWeather;
                ww.next             = ws.nextWeather;
                ww.transition       = t;
                ww.cloudsMaxPercent = mix(ws.cloudsMaxPercent, ws.nextCloudsMaxPercent);
                ww.cloudsSpeed      = mix(ws.cloudsSpeed,      ws.nextCloudsSpeed);
                ww.windSpeed        = mix(ws.windSpeed,        ws.nextWindSpeed);
                ww.landFogDay       = mix(ws.landFogDay,       ws.nextLandFogDay);
                ww.landFogNight     = mix(ws.landFogNight,     ws.nextLandFogNight);
                ww.thunderFlash     = ws.thunderFlash;
                ww.sunglareVis      = ws.sunglareVis;
                ww.sunOccluded      = ws.sunOccluded ? 1 : 0;
                // ⚠ REFERENCES, NEVER DRIVERS — see bridge.h. MW's authored display codes, already
                // blended for the hour and the transition, ride so the host can REPORT how far the
                // generated atmosphere has drifted from MW's intent. A shader reading these is the
                // bug [[project_forge_sky_is_a_display_code]] describes.
                ww.skyColRef[0] = ws.skyCol.r; ww.skyColRef[1] = ws.skyCol.g; ww.skyColRef[2] = ws.skyCol.b;
                ww.fogColRef[0] = ws.fogCol.r; ww.fogColRef[1] = ws.fogCol.g; ww.fogColRef[2] = ws.fogCol.b;
            } else {
                ww.cur = ww.next = -1;
            }
            g_client->setWeather(ww);
        }

        // Statics near/far handover (snapNearCells): the park's BUILD-time snapshot when this is a
        // parked fire, otherwise a fresh one — built and fired in the same frame, the two agree.
        const NearCellsSnap nc = parkFired ? g_park.nearCells : snapNearCells();
        g_client->setNextNearCells(nc.x, nc.y, nc.mask, nc.reach, nc.refsVersion, nc.fwd);

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
                     fpHave ? &fpFrame : nullptr,
                     ripplePack, rippleCount);
        }
        if (!ok) {
            return;     // rpcPending stays false → Finish no-ops
        }

        // Hand everything the finish half needs across the overlap window.
        // Host-inflight lane start (paired 0.0 in finishAndCopy): from here until the
        // collect/finish drains the completion, the host owns the frame.
        MGE_TracyPlot("Forge host inflight", 1.0);
        // ...and the GPU half of it. Tier 1 took the host GPU OUT of the inflight box (the host
        // replies before its fence signals), so the box width answers "how long was the RPC" and
        // this step answers "how long was the GPU still drawing" — dropped to 0.0 the moment
        // copyHostRtToDst's semaphore wait clears. Read the two together: where this step extends
        // past the box is exactly the GPU time the client cannot overlap with.
        MGE_TracyPlot("Forge host GPU inflight", 1.0);
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
        std::uint32_t fpDraws = 0, fpSkinnedDraws = 0, fpAlphaDraws = 0, fpMMDraws = 0;
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
            fpHave = buildFPFrame(fpFrame, fpDraws, fpSkinnedDraws, fpAlphaDraws, fpMMDraws);
            prefetchGridTextures();    // same place as the serial kickoff (see its comment)
            streamPendingTextures();   // worker-side, like the resolvers above (see its comment)
        }
        g_park.epoch           = g_cellEpoch;
        g_park.nearCells       = snapNearCells();
        g_park.bake3rd         = MWBridge::get()->is3rdPerson();
        g_park.bakeEye[0]      = MGE::ExactPos::eye()[0];
        g_park.bakeEye[1]      = MGE::ExactPos::eye()[1];
        g_park.bakeEye[2]      = MGE::ExactPos::eye()[2];
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
        g_park.fpMMDraws       = fpMMDraws;
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
        unsigned fires = 0, dropEpoch = 0, dropInvalid = 0, dropPov = 0;
        double fireMs = 0.0, geomF = 0.0, texF = 0.0, assign = 0.0, kick = 0.0;
        double parkAge = 0.0, buildMs = 0.0;
    };
    ParkStats g_parkStats;

    // ---- Tier 1b: fresh produce worker ------------------------------------------------------
    // A single dedicated std::thread that runs kickoffBody() off the MW main thread. Deliberately
    // NOT the old MGE::RenderThread (deleted in S5a) — that worker owned a D3D9 device lock +
    // a D3DCREATE_MULTITHREADED device and proved crashy; this one touches NO D3D9 (the produce
    // path is D3D9-free after Tier 1a), so it needs no device lock. For Tier 1b the caller kicks then wait()s immediately
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
            MGE::GeometryCache::adoptVisiblePins();   // this job's classify set stays alive until wait()
            {
                std::lock_guard<std::mutex> lk(m_mx);
                m_device = device;
                m_buildOnly = buildOnly;
                m_hasJob = true;
            }
            m_cvJob.notify_one();
        }
        void wait() {
            {
                std::unique_lock<std::mutex> lk(m_mx);
                m_cvDone.wait(lk, [this] { return !m_hasJob; });
            }
            MGE::GeometryCache::releaseAdoptedPins();   // main thread, job done: nothing reads them now
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

    void doDeferredFinish(bool allowCopyDefer = false);   // fwd (defined below); Phase 0 deferred wait
    void finishUndeferred(const char* where, bool atCollect);   // fwd (defined below); frame-ahead OFF
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


    // Dispatcher: route the produce onto the fresh worker per g_produceMode.
    // Mode 2 (OVERLAP) only defers the wait on early-kickoff
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
        if (g_produceMode == 0) {
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
        if ((g_produceMode != 2 && !mode3) || !DistantLand::earlyForgeKickoff) {
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
        if (g_produceMode != 3 || !g_frameAheadLive
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
        if (g_park.bake3rd != MWBridge::get()->is3rdPerson()) {
            // POV flipped between build and fire (see ParkedPayload::bake3rd). Same treatment as a
            // teleport, for the same reason: the payload's visibility set was decided against a
            // camera that no longer exists.
            g_park.valid = false;
            ++g_parkStats.dropPov;
            LOG::logline(">> [park] drop: POV %s at frame %u (switch between build and fire)",
                         g_park.bake3rd ? "3rd -> 1st" : "1st -> 3rd", frame);
            return;
        }
        const double parkAge = tStart - g_park.tBuildEnd;
        g_capturedEmitted = g_park.capturedEmitted;   // restore the build-time AT3 count for g_kick
        // tBuild == tStart: the build cost lives in the park ([park] build=), not this frame's
        // [hb] build bucket — the fire's own cost is exactly the geomF/texF/assign/kick split.
        flushAssignAndKick(device, frame, g_park.drawCount, g_park.skinnedCount,
                           g_park.multiMapCount, g_park.lightCount, g_park.skyCount,
                           g_park.alphaCount, g_park.fpFrame, g_park.fpDraws,
                           g_park.fpSkinnedDraws, g_park.fpAlphaDraws, g_park.fpMMDraws,
                           g_park.fpHave,
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
                         "parkAge=%.2f build=%.2f ms | drops epoch=%u invalid=%u pov=%u",
                         g_parkStats.fires, g_parkStats.fireMs * inv, g_parkStats.geomF * inv,
                         g_parkStats.texF * inv, g_parkStats.assign * inv, g_parkStats.kick * inv,
                         g_parkStats.parkAge * inv, g_parkStats.buildMs * inv,
                         g_parkStats.dropEpoch, g_parkStats.dropInvalid, g_parkStats.dropPov);
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
        // Present backstop for a copy the collect parked when no UI scene came to blit it (the
        // race menu's extra scene, a frame with no main view). Never leave it across the boundary.
        drainPendingCopy();
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
        // Backstop: an early frame with frame-ahead OFF that never reached its blit left its host
        // render unfinished (see finishUndeferred). Close it before anything reuses the channel.
        finishUndeferred("next collect (blit missed)", /*atCollect=*/true);
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
            doDeferredFinish(/*allowCopyDefer=*/true);   // 1.5-ahead: the copy waits for the blit
            g_kick.parkFired = g_pendingFinish.parkFired = false;   // consumed — reset the keyed state
        }
    }

    // Finish + copy the previous deferred host frame (frame-ahead). Main thread ONLY — finishAndCopy
    // does a D3D9 RT copy. Shared by collectDeferredFinish and onStage0CompositeKickoff. No-op unless
    // a deferred finish is pending; consumes g_kick's N-1 state, so it MUST run before kickoffBody
    // resets g_kick for the new frame.
    //
    // allowCopyDefer: only the frame-start collect passes true. A park-fired frame then finishes
    // here but copies at the blit (1.5-ahead, g_copyAtBlit). Every other caller is a backstop or a
    // non-park path, which must leave g_mainTex current before it returns.
    void doDeferredFinish(bool allowCopyDefer) {
        markMainPhase(MP_DEFERRED_FINISH);
        if (!finishDeferred()) {
            // Nothing to release, but the worker may be parked on the gate (armed unconditionally
            // at the kick) — open it or the host RPC never goes out.
            openFinishGate();
            return;
        }
        MGE_ZoneScopedN("Forge deferred finish");
        g_pendingFinish.deferFinish = false;
        // Park frames only: they arm no finish gate (the gate is mode 2's, and it still means "the
        // copy is done"), and their next kick is the frame-start fire, which the RT pair makes safe.
        const bool deferCopy = allowCopyDefer && g_copyAtBlit && g_pendingFinish.parkFired;
        const FinishResult fr = finishAndCopy(g_pendingFinish, deferCopy);
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
        if (fr.copyDeferred) {
            // [hb] accounting happens at the copy, where the frame's figures are complete.
            g_pendingCopy.valid      = true;
            g_pendingCopy.frameFence = fr.frameFence;
            g_pendingCopy.rtSlot     = fr.rtSlot;
            g_pendingCopy.ks         = g_pendingFinish;
            g_pendingCopy.fr         = fr;
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
        // A copy the collect parked (1.5-ahead) lands here on every path that will not reach the
        // blit this frame as a park frame: load, reset, non-early and menu-freeze frames.
        drainPendingCopy();
    }

    // An early-kickoff frame whose kick did NOT defer its finish: frame-ahead OFF (numpad-*) with the
    // early latch still up. The dispatcher then kicks serially at BeginScene(0) and nothing else
    // finishes it — EndScene(0)'s finish only runs on non-early frames, and the collect only takes
    // deferred ones. Left alone the host's RPC window stays open for good: every later RPC is refused
    // ("attempted inside the async RenderFrame window", geomUpload built 0/N) and the world freezes.
    // Finish + copy it here instead. Main thread only.
    //
    // atCollect: the next frame's collect, a backstop for a frame that never reached the blit. There
    // the produce is already drained and an undeferred pending kick is orphaned by definition (non-early
    // frames finish at their own EndScene(0)), so the state alone decides — the flag may have just
    // flipped back ON in this collect's key poll. At the blit the flag gates first, so frame-ahead ON
    // never pays a waitProduce there (mode 2 keeps its worker in flight until the next collect).
    void finishUndeferred(const char* where, bool atCollect) {
        if ((!atCollect && g_frameAheadLive) || g_kick.deferFinish) {
            return;   // deferred frames belong to the next collect
        }
        waitProduce();   // mode 2 may still own g_kick on the worker
        if (!g_kick.rpcPending && !g_kick.rpcEarlyFinished) {
            return;
        }
        MGE_ZoneScopedN("Forge undeferred finish");
        const FinishResult fr = finishAndCopy(g_kick);
        if (fr.ok) {
            accumFrameStats(g_kick, fr, nowMs(), false, 0.0);
        }
        static unsigned s_logged = 0;
        if (s_logged < 4) {
            ++s_logged;
            LOG::logline(">> [pipe] frame-ahead OFF: finished the early frame's host render at the %s", where);
        }
    }

    void onFrameAheadBlit(IDirect3DDevice9* device) {
        markMainPhase(MP_BLIT);
        // Frame-ahead OFF on an early frame: this frame's own host render, finished at the latest point
        // that is still before the blit (the whole MW frame is IPC-free by the early latch).
        finishUndeferred("blit", /*atCollect=*/false);
        // 1.5-ahead: the collect finished the previous host frame but left its RT copy for here, so
        // the host could start the next frame while its GPU was still drawing this one. Copy it now
        // (the wait for the host GPU lands here, under MW's frame instead of ahead of the fire).
        drainPendingCopy();
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

    void setWindowCapture(bool on) {
        g_windowCapture = on;
    }

    bool registerFlipBook(const char* const* names, std::uint32_t count) {
        return registerFlipBookImpl(names, count);
    }

    void openCaptureGrace(unsigned frames) {
        const unsigned until = g_frame + frames;
        if ((int)(until - g_captureGraceUntil) > 0) g_captureGraceUntil = until;
    }

    bool forgeOwnsFrame() {
        // The one mode predicate — see the header. Seam live + composite ON (F11) means the host
        // owns opaque world, distant land, sky, water and depth; MW's own draws for all of it are
        // overwritten by the composite, so MGE suppresses them. F11 off (or a dead host / failed
        // seam → g_initOk false) releases every suppression and MW renders vanilla.
        return g_initOk && g_enabled;
    }

    bool parkFiresLater() {
        return forgeOwnsFrame() && g_produceMode == 3;
    }

    IDirect3DTexture9* sunDX9Texture() {
        return g_sunDX9Texture;
    }

    bool hasCompositeFrame() {
        return g_mainTexValid;
    }

    void noteLoadingBar(bool loading) {
        // Set only — cleared by checkCellEpochAndPurge when it acts on it. A load spans many
        // frames and the purge may not get a produced frame until well after the bar drops, so
        // sampling the edge here and hoping the consumer is listening would lose it.
        // Rising-edge log: this arming step is invisible in [cell-purge] when it fails (the purge
        // simply reports reloaded=0, which is also what a working latch prints when there was no
        // load), and two fixes have already been shipped blind on exactly that ambiguity. One line
        // per load says whether the flag was ever seen at all.
        static bool s_prev = false;
        if (loading && !s_prev) {
            LOG::logline(">> [loadbar] up — reload purge ARMED");
        }
        s_prev = loading;
        if (loading) { g_sawLoadingBar = true; }
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
        // FP1a: seam live + geometry capture up + composite ON (F11) + FIRST person.
        // Gates the cache's armCamera-root walk, the FP draw-list build and the fp wire crossing.
        return g_initOk && g_geomVec.has_value() && g_enabled
            && !MWBridge::get()->is3rdPerson();
    }

    bool wantsFPSuppression() {
        // FP1b: suppress MW's own arm draws ONLY while the host FP pass is actually able
        // to ship them (capture live + camera math validated) — the arms must never
        // vanish without a replacement. g_fpSuppressLive is on by default and flips live
        // on numpad-/ for the A/B.
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

    void noteScene0Blend(const RenderedState* rs) {
        static std::unordered_set<const void*> s_seen;
        if (!rs || s_seen.size() >= 256 || !s_seen.insert(rs->texture).second) return;
        const char* nm = MGE::GeometryCache::resolveTextureName(rs->texture);
        LOG::logline("!! [scene0-blend] tex=%s prim=%d fvf=0x%X vc=%u tri=%u blend=%u/%u zw=%d",
                     nm ? nm : "(unnamed)", (int)rs->primType, (unsigned)rs->fvf,
                     rs->vertCount, rs->primCount, (unsigned)rs->srcBlend, (unsigned)rs->destBlend,
                     (int)rs->zWrite);
    }

    void captureAlphaDraw(const RenderedState* rs, const FragmentState* frs) {
        if (!g_initOk || !g_enabled || !g_capturedVec) return;
        if (!rs || !frs) return;
        // [cap-reject] flame/particle triage: one line per (texture, reason) the guards below
        // refuse, so a blended DIP that silently never reaches the host has a name in the log.
        auto rejectOnce = [&](const char* why) {
            static std::unordered_set<std::uint64_t> s_seen;
            const std::uint64_t key = ((std::uint64_t)(std::uintptr_t)rs->texture << 8) ^ (std::uint64_t)(std::uintptr_t)why;
            if (s_seen.size() < 256 && s_seen.insert(key).second) {
                const char* nm = MGE::GeometryCache::resolveTextureName(rs->texture);
                LOG::logline("!! [cap-reject] %s tex=%s prim=%d fvf=0x%X vbs=%u vc=%u tri=%u blend=%u/%u",
                             why, nm ? nm : "(unnamed)", (int)rs->primType, (unsigned)rs->fvf,
                             (unsigned)rs->vertexBlendState, rs->vertCount, rs->primCount,
                             (unsigned)rs->srcBlend, (unsigned)rs->destBlend);
            }
        };
        // HW-skinned blends (ghosts) excluded — the bind-pose VB here is the wrong pose (a
        // separate follow-up). Need an indexed TRIANGLELIST with a real stride (TRISTRIP/FAN and
        // non-indexed particle DIPs are dropped; a counter would reveal if MW emits any).
        if (rs->vertexBlendState != 0) { rejectOnce("skinned"); return; }
        if (rs->primType != D3DPT_TRIANGLELIST) { rejectOnce("primtype"); return; }
        if (!rs->vb || !rs->ib || rs->vbStride == 0) { rejectOnce("no-vb-ib"); return; }
        if ((rs->fvf & D3DFVF_POSITION_MASK) != D3DFVF_XYZ) { rejectOnce("fvf-pos"); return; }   // untransformed XYZ only
        if (rs->vertCount == 0 || rs->primCount == 0) return;

        // Texture slot: rs->texture is the proxy realTexture pointer, identical to the GPU texture
        // the cache walk registered in g_textureNameMap. Resolve name -> bindless slot (per-frame
        // memo). No name -> slot 0 (host default white): a visible white particle beats an invisible
        // one, and NiFlipController textures self-correct after the first walk registers them.
        // Resolved BEFORE the dedup below because the dedup keys on this slot, not the pointer.
        auto slotOfStageTexture = [&](IDirect3DTexture9* tex, bool warnUnnamed) -> std::uint32_t {
            auto mit = g_capInTexMemo.find(tex);
            if (mit != g_capInTexMemo.end()) {
                return mit->second;
            }
            const char* name = MGE::GeometryCache::resolveTextureName(tex);
            const std::uint32_t slot = name ? resolveTextureSlot(name) : 0u;
            if (!name && warnUnnamed && g_capNoName++ < 20) {
                LOG::logline("!! [alpha-cap] no source name for tex=%p (white)", (void*)tex);
            }
            g_capInTexMemo.emplace(tex, slot);
            return slot;
        };
        const std::uint32_t texIndex = slotOfStageTexture(rs->texture, /*warnUnnamed=*/true);

        // Old-msoc double-draw guard: this DIP is a duplicate iff the host already draws a cached
        // blended shape with the same (bindless slot, vertexCount) — the cache draws from its own VB
        // copies so rs->vb never matches, but (slot, vertCount) does (the vertCount term kills the
        // shared-texture false positive — particle counts vary). Read the ACTIVE (prev-frame) set,
        // NOT g_alphaDedup — the worker is concurrently rebuilding g_alphaDedup and reading it here
        // raced (glow-window white/orange flicker). swapCaptureBuffers hands over a stable snapshot.
        // See alphaDedupKey for why the key is the SLOT and not the GPU texture pointer.
        if (texIndex != 0u) {
            const std::uint64_t dedupKey = alphaDedupKey(texIndex, rs->vertCount);
            if (g_alphaDedupActive.find(dedupKey) != g_alphaDedupActive.end()) {
                // [alpha-dedup] probe: log the first few victims with enough identity to tell a real
                // cached-blend duplicate from an aliased particle draw (a particle's vertCount moves
                // frame to frame; a cached blend's does not).
                static std::uint32_t s_logged = 0;
                if (s_logged < 48) {
                    ++s_logged;
                    const char* nm = MGE::GeometryCache::resolveTextureName(rs->texture);
                    LOG::logline("!! [alpha-dedup] dropped DIP tex=%s vc=%u tri=%u (key collision)",
                                 nm ? nm : "(unnamed)", rs->vertCount, rs->primCount);
                }
                ++g_capDedupDrops;
                return;
            }
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

        // FFE stages BEYOND the base map. MW folds a shape's dark/detail/glow siblings into extra
        // stages of this same DIP, so reading stage 0 alone draws it at full base brightness —
        // exactly the "too bright" kurp VFX (every one pairs its base with a blackmip*/darkmap*
        // MODULATE layer). Walk forward and stop at the first DISABLE, which is precisely how D3D
        // terminates the FFE chain: MW leaves stale COLOROPs in later stages (stageOps=[4,1,4,1]
        // is ONE effective stage, not two), so scanning all four would resurrect dead state.
        std::uint32_t capStageCount = 0;
        std::uint32_t capStages[3] = { 0, 0, 0 };
        for (std::uint32_t s = 1; s < 4 && capStageCount < 3; ++s) {
            const auto& st = frs->stage[s];
            if (st.colorOp == D3DTOP_DISABLE) break;
            std::uint32_t op;
            if (st.colorOp == D3DTOP_MODULATE)        op = IPC::kMMOpMod;
            else if (st.colorOp == D3DTOP_MODULATE2X) op = IPC::kMMOpMod2X;
            else if (st.colorOp == D3DTOP_ADD)        op = IPC::kMMOpAdd;
            else break;   // BLENDTEXTUREALPHA/DOTPRODUCT3/... — unmodelled, stop rather than guess
            IDirect3DTexture9* stex = rs->stageTexture[s];
            if (!stex) break;                       // op with no texture bound: nothing to fold
            // The captured VB carries ONE UV set (GeomVertexWire, shared with every cached mesh and
            // with the alpha PSO input layout), so a stage sampling set 1+ cannot be honoured. DROP
            // it — sampling the wrong coordinates would be worse than the pre-existing base-only
            // look. Those shapes are cached/Route-C-owned in every case measured (see
            // tasks/forge-at3-darkmap.md); the counter says if that ever stops being true.
            if (st.texcoordIndex != 0) {
                if (g_capStageUVDrop++ == 0) {
                    const char* nm = MGE::GeometryCache::resolveTextureName(stex);
                    LOG::logline("!! [cap-stage] stage %u on UV set %u dropped (captured VB is single-UV) tex=%s",
                                 s, (unsigned)st.texcoordIndex, nm ? nm : "(unnamed)");
                }
                break;
            }
            const std::uint32_t sslot = slotOfStageTexture(stex, /*warnUnnamed=*/false);
            if (sslot == 0u) break;                 // unnamed → white → MODULATE by white = no-op
            // clampMode: RenderedState carries no sampler address state at all, per stage or
            // otherwise — the captured base map ships MW's default for the same reason
            // (emitCapturedAlphaDraw's disclosed scope gap). Match it.
            capStages[capStageCount++] = IPC::packMMStage(sslot, 0u, op, IPC::kTexWrapSWrapT);
        }
        if (capStageCount) {
            static std::uint32_t s_stageLogged = 0;
            if (s_stageLogged < 8) {
                ++s_stageLogged;
                const char* nm = MGE::GeometryCache::resolveTextureName(rs->texture);
                LOG::logline(">> [cap-stage] %s + %u stage(s) ops=[%d,%d,%d,%d]",
                             nm ? nm : "(unnamed)", capStageCount,
                             (int)frs->stage[0].colorOp, (int)frs->stage[1].colorOp,
                             (int)frs->stage[2].colorOp, (int)frs->stage[3].colorOp);
            }
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
            if (s_capDiagSeen.insert(dkey).second && s_capDiagSeen.size() <= 256) {
                const int st1 = (int)frs->stage[1].colorOp, st2 = (int)frs->stage[2].colorOp, st3 = (int)frs->stage[3].colorOp;
                LOG::logline(">> [cap-diag] %s fvf=0x%X stride=%u norm=%d@%u col=%d@%u uv=%d@%u "
                             "vcs=%u lit=%u matD=(%.2f,%.2f,%.2f) matA=%.2f matE=(%.2f,%.2f,%.2f) "
                             "srcD=%u srcE=%u blend=%u/%u slot=%u stageOps=[%d,%d,%d,%d]",
                             dname ? dname : "(null)", (unsigned)rs->fvf, (unsigned)stride,
                             (int)hasNorm, (unsigned)normOff, (int)hasCol, (unsigned)colOff, (int)hasUV, (unsigned)uvOff,
                             vColSource, (unsigned)rs->useLighting,
                             frs->material.diffuse.r, frs->material.diffuse.g, frs->material.diffuse.b,
                             frs->material.diffuse.a,
                             frs->material.emissive.r, frs->material.emissive.g, frs->material.emissive.b,
                             (unsigned)rs->matSrcDiffuse, (unsigned)rs->matSrcEmissive,
                             (unsigned)rs->srcBlend, (unsigned)rs->destBlend, texIndex,
                             (int)frs->stage[0].colorOp, st1, st2, st3);
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
        rec.stageCount = capStageCount;
        rec.stages[0] = capStages[0]; rec.stages[1] = capStages[1]; rec.stages[2] = capStages[2];
        g_capInRecs.push_back(rec);
    }

    // Part A upload accounting: the geometry cache tags each host geom reship by cause.
    void noteUpload(std::uint8_t cat, std::uint32_t bytes) {
        if (cat >= kUpCount) cat = kUpOther;
        g_upFrame.parts[cat] += 1; g_upFrame.bytes[cat] += bytes;
    }

    // ---- Geometry CONTENT probe (MGE_GEOM_CONTENT_PROBE=1) -----------------------------------------
    // MEASUREMENT for a content-keyed host mesh cache ("DL is same meshes, maybe that cache can be
    // reused for cell loads"). Uploads are keyed per INSTANCE (cache key = shape) and the host drops
    // them at every load, so each copy of a rock uploads its own geometry, and a revisit or a cell
    // built from the same kit re-uploads all of it. This hashes every shipped part's vertex+index
    // bytes and, per window between two loads, sorts the bytes into:
    //   new      content never shipped before this session
    //   earlier  shipped in an EARLIER window (the host had it, the load threw it away)
    //   repeat   shipped earlier in THIS window by ANOTHER instance (another copy of the same mesh)
    //   resend   shipped earlier in this window by the SAME instance (a re-upload, not instancing)
    // earlier + repeat is what a content-keyed cache would not have to upload; resend is a separate
    // problem (the per-instance dedup missing its own earlier copy). Off by default: the hash reads
    // every uploaded byte once (~32 MB on a load).
    // A fifth class, aliased, counts what GEOMETRY DEDUP did NOT ship: instances sent as a header-only
    // kGeomFlagAlias record (their bytes are what would otherwise have landed in repeat).
    struct ContentWindow {
        std::uint64_t bytes[3][5] = {};   // [kind: static, skinned, multimap][new, earlier, repeat, resend, aliased]
        std::uint32_t parts[3][5] = {};
    };
    struct ContentSeen {
        std::uint32_t window;
        std::uint32_t key;      // the instance that shipped it last
    };
    ContentWindow                                  g_contentWin;
    std::unordered_map<std::uint64_t, ContentSeen> g_contentSeen;   // content hash -> last window / key
    std::uint32_t                                  g_contentWindowId = 0;

    bool contentProbeOn() {
        static const bool on = [] {
            const char* e = std::getenv("MGE_GEOM_CONTENT_PROBE");
            return e && e[0] == '1';
        }();
        return on;
    }

    std::uint64_t hashBytes(const void* p, std::size_t n, std::uint64_t h) {
        const std::uint8_t* b = static_cast<const std::uint8_t*>(p);
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            std::uint64_t w;
            memcpy(&w, b + i, 8);
            h ^= w;
            h *= 0x9E3779B97F4A7C15ull;
            h ^= h >> 29;
        }
        for (; i < n; ++i) {
            h ^= b[i];
            h *= 0x100000001B3ull;
        }
        return h ^ (std::uint64_t)n;
    }

    // GEOMETRY DEDUP content hash, built for THIS client: it is 32-bit x86, where hashBytes' 64-bit
    // multiply is emulated (three imuls) and every step waits on the last — 3.2 GB/s, ~22 ms for one
    // exterior's first sights. xxHash32's round instead: four INDEPENDENT 32-bit lanes over 16-byte
    // stripes, so the multiplies overlap. The 128-bit state folds to 64 bits through two different
    // projections; the dedup key also requires equal vertex and index counts, and each buffer's tail
    // is mixed with its length known from those counts, so the two-buffer stream is unambiguous.
    struct ContentHasher {
        static constexpr std::uint32_t P1 = 2654435761u, P2 = 2246822519u, P3 = 3266489917u,
                                       P4 = 668265263u,  P5 = 374761393u;
        static std::uint32_t rotl(std::uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }
        static std::uint32_t avalanche(std::uint32_t h) {
            h ^= h >> 15; h *= P2; h ^= h >> 13; h *= P3; h ^= h >> 16;
            return h;
        }
        std::uint32_t v[4] = { P1 + P2, P2, 0u, 0u - P1 };

        void update(const void* p, std::size_t n) {
            const std::uint8_t* b = static_cast<const std::uint8_t*>(p);
            std::uint32_t a0 = v[0], a1 = v[1], a2 = v[2], a3 = v[3];
            std::size_t i = 0;
            for (; i + 16 <= n; i += 16) {
                std::uint32_t w[4];
                memcpy(w, b + i, 16);
                a0 = rotl(a0 + w[0] * P2, 13) * P1;
                a1 = rotl(a1 + w[1] * P2, 13) * P1;
                a2 = rotl(a2 + w[2] * P2, 13) * P1;
                a3 = rotl(a3 + w[3] * P2, 13) * P1;
            }
            for (; i + 4 <= n; i += 4) {
                std::uint32_t w;
                memcpy(&w, b + i, 4);
                a0 = rotl(a0 + w * P3, 17) * P4;
            }
            for (; i < n; ++i) {
                a1 = rotl(a1 + b[i] * P5, 11) * P1;
            }
            v[0] = a0; v[1] = a1; v[2] = a2; v[3] = a3;
        }

        std::uint64_t finish() const {
            const std::uint32_t lo = avalanche(rotl(v[0], 1) + rotl(v[1], 7) + rotl(v[2], 12) + rotl(v[3], 18));
            const std::uint32_t hi = avalanche((v[0] * P3) ^ rotl(v[1], 9)) + avalanche((v[2] * P4) ^ rotl(v[3], 23));
            return ((std::uint64_t)hi << 32) | lo;
        }
    };

    void noteContent(int kind, std::uint32_t key, const void* verts, std::size_t vbBytes, const void* indices, std::size_t ibBytes) {
        if (!contentProbeOn()) {
            return;
        }
        const std::uint64_t h = hashBytes(indices, ibBytes, hashBytes(verts, vbBytes, 0xCBF29CE484222325ull + (std::uint64_t)kind));
        const std::uint64_t bytes = vbBytes + ibBytes;
        auto it = g_contentSeen.find(h);
        int cls;
        if (it == g_contentSeen.end()) {
            cls = 0;
            g_contentSeen.emplace(h, ContentSeen{ g_contentWindowId, key });
        } else {
            if (it->second.window != g_contentWindowId) {
                cls = 1;
            } else {
                cls = (it->second.key == key) ? 3 : 2;
            }
            it->second = ContentSeen{ g_contentWindowId, key };
        }
        g_contentWin.bytes[kind][cls] += bytes;
        ++g_contentWin.parts[kind][cls];
    }

    void noteContentAliased(std::size_t bytes) {
        if (!contentProbeOn()) {
            return;
        }
        g_contentWin.bytes[0][4] += bytes;
        ++g_contentWin.parts[0][4];
    }

    // One line per window, at the load that ends it.
    void contentProbeWindowEnd(unsigned frame) {
        if (!contentProbeOn()) {
            return;
        }
        static const char* kKind[3] = { "static", "skinned", "multimap" };
        std::uint64_t tot[5] = {};
        for (int k = 0; k < 3; ++k) {
            for (int c = 0; c < 5; ++c) {
                tot[c] += g_contentWin.bytes[k][c];
            }
        }
        const std::uint64_t all = tot[0] + tot[1] + tot[2] + tot[3];
        LOG::logline(">> [geom-content] window %u ending at frame %u: %.1f MB shipped = new %.1f | earlier %.1f | repeat %.1f | resend %.1f MB  (content cache saves %.0f%%, unique hashes this session %zu) | aliased (not shipped) %.1f MB",
                     g_contentWindowId, frame, all / 1048576.0, tot[0] / 1048576.0, tot[1] / 1048576.0,
                     tot[2] / 1048576.0, tot[3] / 1048576.0, all ? 100.0 * (tot[1] + tot[2]) / all : 0.0,
                     g_contentSeen.size(), tot[4] / 1048576.0);
        for (int k = 0; k < 3; ++k) {
            const std::uint64_t kb = g_contentWin.bytes[k][0] + g_contentWin.bytes[k][1] + g_contentWin.bytes[k][2] + g_contentWin.bytes[k][3];
            if (kb == 0 && g_contentWin.bytes[k][4] == 0) {
                continue;
            }
            LOG::logline("   [geom-content]   %-8s %.1f MB in %u parts: new %.1f MB/%u | earlier %.1f MB/%u | repeat %.1f MB/%u | resend %.1f MB/%u | aliased %.1f MB/%u",
                         kKind[k], kb / 1048576.0,
                         g_contentWin.parts[k][0] + g_contentWin.parts[k][1] + g_contentWin.parts[k][2] + g_contentWin.parts[k][3],
                         g_contentWin.bytes[k][0] / 1048576.0, g_contentWin.parts[k][0],
                         g_contentWin.bytes[k][1] / 1048576.0, g_contentWin.parts[k][1],
                         g_contentWin.bytes[k][2] / 1048576.0, g_contentWin.parts[k][2],
                         g_contentWin.bytes[k][3] / 1048576.0, g_contentWin.parts[k][3],
                         g_contentWin.bytes[k][4] / 1048576.0, g_contentWin.parts[k][4]);
        }
        g_contentWin = ContentWindow{};
        ++g_contentWindowId;
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
        // Latched BEFORE drainPendingIfFull: that can run a whole flushGeometry, whose eviction drain
        // erases g_uploadedRev entries (and with them, possibly, `rev`).
        const bool firstUpload = (rev == g_uploadedRev.end());

        drainPendingIfFull();
        const std::size_t vbBytes = (std::size_t)vertexCount * sizeof(IPC::GeomVertexWire);
        const std::size_t ibBytes = (std::size_t)indexCount * sizeof(std::uint16_t);

        // GEOMETRY DEDUP. A re-upload of a slot that holds a block ref drops it first — the host's
        // rebuild path releases the slot's range the same way — and ships plain (block 0), so a
        // morphing mesh keeps the sameShape/streak path it always had. Only a key's FIRST upload
        // (no prior g_uploadedRev entry) may share: a block is plain, unanimated static content.
        dropBlockRef(slot);
        if (firstUpload && !forceReupload && !hdr.uvAnimBytes && geomAliasOn()) {
            const double tHash0 = nowMs();
            ContentHasher hasher;
            hasher.update(verts, vbBytes);
            hasher.update(indices, ibBytes);
            const ContentKey ck{ hasher.finish(), vertexCount, indexCount };
            g_aliasStats.hashMs += nowMs() - tHash0;
            auto bc = g_blockByContent.find(ck);
            if (bc != g_blockByContent.end()) {
                IPC::GeomPartWire ah = {};
                ah.slot       = slot;
                ah.revisionID = revision;
                ah.flags      = IPC::kGeomFlagAlias;   // vertexCount = indexCount = 0 -> header-only
                ah.block      = bc->second.block;
                const std::size_t at = g_pendingBlob.size();
                g_pendingBlob.resize(at + sizeof(ah));
                memcpy(g_pendingBlob.data() + at, &ah, sizeof(ah));
                ++g_pendingParts;
                noteWindowPart(sizeof(ah));
                ++bc->second.refs;
                g_slotBlock.emplace(slot, ck);
                g_uploadedRev[key] = { modelId, vertexCount, revision };
                ++g_aliasStats.aliases;
                g_aliasStats.aliasedBytes += vbBytes + ibBytes;
                noteContentAliased(vbBytes + ibBytes);
                return;
            }
            hdr.block = ++g_nextBlock;
            g_blockByContent.emplace(ck, BlockRef{ hdr.block, 1u });
            g_slotBlock.emplace(slot, ck);
            ++g_aliasStats.blocks;
        }

        const std::size_t at = g_pendingBlob.size();
        g_pendingBlob.resize(at + sizeof(hdr) + vbBytes + ibBytes + hdr.uvAnimBytes);
        std::uint8_t* dst = g_pendingBlob.data() + at;
        memcpy(dst, &hdr, sizeof(hdr));            dst += sizeof(hdr);
        memcpy(dst, verts, vbBytes);               dst += vbBytes;
        memcpy(dst, indices, ibBytes);             dst += ibBytes;
        if (hdr.uvAnimBytes) { memcpy(dst, uvAnim, hdr.uvAnimBytes); }

        ++g_pendingParts;
        noteWindowPart(sizeof(hdr) + vbBytes + ibBytes + hdr.uvAnimBytes);
        g_uploadedRev[key] = { modelId, vertexCount, revision };
        noteContent(0, key, verts, vbBytes, indices, ibBytes);
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

        dropBlockRef(slot);   // a recycled key's slot may hold a block; the host releases it on rebuild
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
        noteWindowPart(sizeof(hdr) + vbBytes + ibBytes);
        g_uploadedRev[key] = { modelId, vertexCount, revision };
        noteContent(1, key, verts, vbBytes, indices, ibBytes);
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

        dropBlockRef(slot);   // a recycled key's slot may hold a block; the host releases it on rebuild
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
        noteWindowPart(sizeof(hdr) + vbBytes + ibBytes + hdr.uvAnimBytes);
        g_uploadedRev[key] = { modelId, vertexCount, revision };
        noteContent(2, key, verts, vbBytes, indices, ibBytes);
    }

    void shutdown() {
        // Tier 1b: stop the produce worker BEFORE draining/releasing anything it may touch —
        // no-op if it was never started (produce-off-main stayed OFF the whole session).
        g_produceWorker.stop();
        g_texIo.stop();   // after the produce worker (its only submitter besides main): reads in flight finish
        // Never tear down with a host RenderFrame still pending (deferred or not):
        // drain it so the host isn't mid-frame and the window guard isn't latched
        // if the seam comes back up.
        if (g_client && g_kick.rpcPending) {
            g_client->renderSceneFinish(nullptr);
        }
        if (g_hostZoneOpen) { MGE_TracyHostFrameEnd(g_hostZoneCtx); g_hostZoneOpen = false; }
        g_kick = KickState{};
        dropPendingCopy();   // a parked copy has nothing to copy into once this releases
        // The stream lane's window is the host worker's until its batch completes.
        if (g_client) { g_client->streamUploadDrain(); }
        {
            std::lock_guard<std::mutex> lk(g_texResidencyMx);
            g_texStreamFlight.clear();
        }
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
        g_fpMultiMapVec.reset();
        g_texVec.reset();
        g_texStreamVec.reset();
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
        g_lightGoboScratch.clear();
        g_lightGoboScratch.shrink_to_fit();
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
        g_fpMultiMapScratch.clear();
        g_fpMultiMapScratch.shrink_to_fit();
        g_texPendingEntries.clear();
        g_texPendingEntries.shrink_to_fit();
        g_texPendingBytes = 0;
        g_pendingParts = 0;
        g_texPendingCount = 0;
        g_keySlot.clear();
        g_uploadedRev.clear();
        g_blockByContent.clear();
        g_slotBlock.clear();
        g_texSlot.clear();
        g_slotName.clear();
        g_slotLastUsed.clear();
        g_slotBytes.clear();
        g_texFreeSlots.clear();
        g_texStreamQueue.clear();
        g_texRetireSlots.clear();
        g_gridTexQueue.clear();
        g_gridTexPins.clear();
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
// missing on compact keyboards). Defined here (not imgui_panels.cpp) so it can reach
// the file-static seam state + DistantLand::hostCullOnly directly; declared extern
// in imgui_panels.cpp. Each control mirrors the equivalent [seam] key in pollDevKeys,
// logging on change so the log reads the same whether toggled by key or click.
// ---------------------------------------------------------------------------
void DrawForgeDevPanel() {
    if (!ImGui::Begin("Forge Dev")) { ImGui::End(); return; }

    // Flip a bool AND log on change (matches the key handlers' >> [seam] lines).
    auto logCheck = [](const char* label, bool& v, const char* msg) {
        if (ImGui::Checkbox(label, &v))
            LOG::logline(">> [seam] %s %s", msg, v ? "ON" : "OFF");
    };

    // GPU capture. This panel exists because some seam keys are missing on compact keyboards,
    // and NUMPAD0 is exactly that kind of key — it was the one A/B with no clickable twin, so a
    // capture simply could not be armed without a numpad. Same latch the key sets, so the host
    // path (armGpuCapture -> StartFrameCapture at the next renderScene) is untouched.
    ImGui::Separator();
    ImGui::Text("GPU capture (RenderDoc)");
    if (ImGui::Button("Capture next host frame")) {
        g_gpuCapturePending = 1u;
        LOG::logline(">> [seam] GPU frame capture requested (panel) — see mgeHost64.log");
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Arms ONE host frame, bracketed by the host itself (mgeHost64 has no Present of\n"
            "its own, so RenderDoc's hotkey would capture a Morrowind frame instead).\n"
            "Needs renderdoc.dll in the HOST process: launch under the RenderDoc UI with\n"
            "'capture child processes', or set MGE_RDOC=1. Writes forge_frame_frameNNN.rdc\n"
            "next to mgeHost64.exe; mgeHost64.log says ARMED / STARTED / WRITTEN.\n"
            "Equivalent to numpad 0.");

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
    {   // One control for the one mode numpad-* cycles (it used to be two checkboxes, both "numpad *").
        ImGui::Text("Pipelining (numpad *):");
        const int cur = pipeMode();
        for (int m = kPipeOff; m <= kPipeAhead15; ++m) {
            ImGui::SameLine();
            if (ImGui::RadioButton(m == kPipeOff ? "OFF" : m == kPipeAhead1 ? "1-ahead" : "1.5-ahead",
                                   cur == m) && cur != m) {
                setPipeMode(m);
            }
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "OFF: the host frame is finished and copied inside the same MW frame (serial).\n"
                "1-ahead: finish + RT copy at the next frame's start; the host overlaps MW's frame.\n"
                "1.5-ahead: finish at the next frame's start, RT copy at the blit; the host also\n"
                "records its next frame while its GPU draws this one (tasks/forge-pipeline-depth.md).");
    }
    logCheck("FP arm suppression (numpad /)", g_fpSuppressLive, "FP suppression (FP1b)");

    // Live render-scale (supersampling). The host renders into a g_rw x g_rh sub-rect of the fixed
    // g_w x g_h allocation (ceiling x backbuffer); the slider just restamps the render size — no
    // reallocation, no host re-init. 1.0 = native (identical to the pre-feature path).
    {
        float s = g_renderScale;
        // ⚠ The slider's MAX is the live ceiling, not the hard cap. With SSAA off the allocation IS
        // the backbuffer, so there is no sub-rect to grow into and offering 2.0x here would be a
        // control that silently does nothing (ImGui would clamp on the way back in, and the user
        // would be left reading a slider that disagrees with the image).
        const bool ssaaOff = (g_maxRenderScale <= kMinRenderScale);
        ImGui::BeginDisabled(ssaaOff);
        if (ImGui::SliderFloat("Render scale (SSAA)", &s, kMinRenderScale, g_maxRenderScale, "%.2fx")) {
            g_renderScale = s;
            recomputeRenderSize();
            LOG::logline(">> [seam] render scale %.2fx -> render %ux%u (alloc %ux%u, bb %ux%u)",
                         g_renderScale, g_rw, g_rh, g_w, g_h, g_bbW, g_bbH);
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered()) {
            if (ssaaOff) {
                ImGui::SetTooltip(
                    "SSAA is OFF (default). The allocation is exactly the backbuffer, which is what\n"
                    "keeps the host's fixed VRAM footprint down — a 2.0x ceiling is 4x the AREA on\n"
                    "17 render targets, reserved whether or not supersampling ever runs, and it is\n"
                    "what pushed this machine over the DXGI budget into driver paging.\n"
                    "To enable: set MGE_RENDER_SCALE=1.5 (up to %.2f) before launch. It has to be a\n"
                    "startup setting because the shared RT is created at the allocation size.",
                    kRenderScaleHardCap);
            } else {
                ImGui::SetTooltip(
                    "Supersampling: host renders the world at scale x backbuffer, downfiltered on\n"
                    "composite. VRAM is fixed at the ceiling (%.2fx) that MGE_RENDER_SCALE asked for;\n"
                    "this only moves the per-frame viewport. UI stays at native res.\n"
                    "1.0x is byte-identical to the native path.",
                    g_maxRenderScale);
            }
        }
        ImGui::SameLine();
        ImGui::Text("(%ux%u)", g_rw, g_rh);
    }

    // EMISSIVE LEVEL (S3 calibration). Multiplies the flux/area gain at draw-list build, so it is
    // live — the per-entry gain itself is computed once at cache-fill and a constant change there
    // needs a rebuild AND a full cache re-walk. See emissiveForDraw() for the model.
    //
    // 1.0 puts the reference paper lantern at k_eff = 2.2 (what the LDR-era constant always aimed at
    // and never reached, because area_ref was a 3.1x underestimate — fixed 2026-08-18 from the
    // measured area=1883.5 logline). 2.05 reaches the MEASURED k_ref of 4.5, which HDR has now made
    // expressible; that is the number to try first.
    {
        float es = MGE::GeometryCache::g_emissiveScale;
        if (ImGui::SliderFloat("Emissive level (x flux/area gain)", &es, 0.0f, 6.0f, "%.2fx")) {
            MGE::GeometryCache::g_emissiveScale = es;
            LOG::logline(">> [emissive] level %.2fx (paper-lantern k_eff %.2f; 2.05x = the measured 4.5)",
                         es, 2.2f * es);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Scales every fixture whose emission came from its own light\n"
                "(emissive = authored x k x lightColour / meshArea). Class RATIOS come from\n"
                "the meshes and do not move; this is the absolute level only.\n\n"
                "1.00x = reference paper lantern at k_eff 2.2\n"
                "2.05x = the MEASURED k_ref of 4.5 (try this first)\n\n"
                "WARNING: pair with Exposure 'Meter statistic' = p90. A single lantern can be\n"
                "69%% of a frame's light energy in 0.23%% of its area (measured), so a frame-MEAN\n"
                "meter chases it and the ROOM goes dark while the lantern looks unchanged.\n"
                "Vanilla candles are unaffected: their meshes author no emissive material at all,\n"
                "so they have no gain to scale.");

        // CEILING on k_eff. flux/area double-counts whenever a fixture has several emissive shapes
        // sharing one light: each claims the FULL flux and divides by its own area, so the smallest
        // wins hardest. light_de_lantern_02 has 8 emissive shapes — glass k_eff 26, flame 402.
        float mk = MGE::GeometryCache::g_emissiveMaxK;
        if (ImGui::SliderFloat("Emissive ceiling (max k_eff)", &mk, 0.0f, 256.0f, "%.0f")) {
            MGE::GeometryCache::g_emissiveMaxK = mk;
            LOG::logline(">> [emissive] ceiling k_eff <= %.0f (%s)", mk,
                         (mk <= 0.0f) ? "OFF - raw flux/area" : "hue-preserving scalar clamp");
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Bounds flux/area's small-area limit, where it stops being trustworthy:\n"
                " - a fixture with N emissive shapes gives EACH the full flux (light_de_lantern_02\n"
                "   has 8), so the smallest shape wins hardest;\n"
                " - a flame BILLBOARD is a sprite standing in for a volume, so its triangle area\n"
                "   was never an emitting area.\n\n"
                "DEFAULT 0 = OFF, and it should stay off: this was a band-aid for flux\n"
                "double-counting, and the fixture GROUPING fixed that properly (glass shade\n"
                "k_eff 26.05 -> 3.26 on its own). At 32 it was suppressing the flame 12.6x -\n"
                "k_eff 402 clamped to 32 - i.e. throttling the brightest thing in the room.\n\n"
                "Measured reference points: paper lantern k_eff 2.2, glass shade 3.3, flame 403.\n"
                "Check grpArea/pieces in the [emissive] log before reaching for this: pieces=1\n"
                "on a fixture that plainly has many means the GROUPING is broken instead.\n\n"
                "Hue-preserving: ONE scalar over all three channels. Per-channel clipping would\n"
                "snap R:G:B toward 1:1:1 and read as hot cream - the LDR-era bug.");
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
