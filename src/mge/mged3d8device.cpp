
#include "mged3d8device.h"
#include "proxydx/d3d8texture.h"
#include "proxydx/d3d8surface.h"
#include "proxydx/devicelock.h"

#include <algorithm>
#include <cstdio>
#include <tlhelp32.h>
#include "mgeversion.h"
#include "configuration.h"
#include "renderthread.h"
#include "distantland.h"
#include "mwbridge.h"
#include "statusoverlay.h"
#include "userhud.h"
#include "videobackground.h"
#include "mge_tracy.h"
#include "drawstats.h"
#include "support/timing.h"
#include "support/log.h"

bool g_tracyActive = false;

// Per-stage draw-call breakdown. Emits a Tracy plot per stage every frame and a
// 60-frame summary line to mgexe.log (gated on LogDistantPipeline) so the totals
// can be eyeballed against the single draw-call number DXVK reports. Counters are
// reset here for the next frame. Called from Present, after all draws complete.
void DrawStats::logFrame() {
    using namespace DrawStats;

    std::uint32_t totalCalls = 0, totalPrims = 0;
    for (int i = 0; i < COUNT; ++i) { totalCalls += g_calls[i]; totalPrims += g_prims[i]; }

#ifdef TRACY_ENABLE
    if (g_tracyActive) {
        for (int i = 0; i < COUNT; ++i) {
            TracyPlot(plotName((Stage)i), (int64_t)g_calls[i]);
        }
        TracyPlot("draws:TOTAL", (int64_t)totalCalls);
    }
#endif

    static int s_frame = 0;
    if (Configuration.LogDistantPipeline && (s_frame++ % 60) == 0) {
        // scene = scene0+1+2+UI, pre = depth(all)+shadow+refl(all), DL = land+statics+grass+water+sky
        const std::uint32_t scene = g_calls[Scene0] + g_calls[Scene1] + g_calls[Scene2] + g_calls[UI];
        const std::uint32_t depth = g_calls[Depth] + g_calls[DepthCache] + g_calls[DepthLand]
                                  + g_calls[DepthStatics] + g_calls[DepthGrass];
        const std::uint32_t shadow = g_calls[Shadow] + g_calls[ShadowCache] + g_calls[ShadowDL];
        const std::uint32_t reflCache = g_calls[ReflCacheColor] + g_calls[ReflCacheShadow] + g_calls[ReflCacheTerrain];
        const std::uint32_t refl  = g_calls[Reflection] + g_calls[ReflStatics] + reflCache;
        const std::uint32_t pre   = depth + shadow + refl;
        const std::uint32_t dl    = g_calls[Land] + g_calls[Statics] + g_calls[Grass] + g_calls[Water] + g_calls[Sky];
        LOG::logline(
            "-- draws: TOTAL=%u  scene=%u(s0=%u s1=%u s2=%u ui=%u)  pre=%u "
            "depth=%u(cache=%u land=%u stat=%u grass=%u misc=%u) "
            "shadow=%u(cache=%u dl=%u misc=%u) "
            "refl=%u(dl.stat=%u cache=%u[color=%u shadow=%u terr=%u] misc=%u)  "
            "DL=%u(land=%u statics=%u grass=%u water=%u sky=%u)  post=%u dbg=%u other=%u",
            totalCalls,
            scene, g_calls[Scene0], g_calls[Scene1], g_calls[Scene2], g_calls[UI], pre,
            depth, g_calls[DepthCache], g_calls[DepthLand], g_calls[DepthStatics], g_calls[DepthGrass], g_calls[Depth],
            shadow, g_calls[ShadowCache], g_calls[ShadowDL], g_calls[Shadow],
            refl, g_calls[ReflStatics], reflCache,
            g_calls[ReflCacheColor], g_calls[ReflCacheShadow], g_calls[ReflCacheTerrain], g_calls[Reflection],
            dl, g_calls[Land], g_calls[Statics], g_calls[Grass], g_calls[Water], g_calls[Sky],
            g_calls[Post], g_calls[Debug], g_calls[Other]);
    }

    reset();
}

// timeBeginPeriod for Sleep granularity in the frame limiter.
#pragma comment(lib, "winmm.lib")

// MGE-owned frame limiter. Replaces Morrowind's built-in Max-FPS pacer (which
// anchors to a pre-sleep timestamp, so it releases ~twice per intended period
// with an alternating short/long beat). This paces in Present() with a QPC
// accumulator: the next-present deadline advances by exactly one period
// (deadline += period) rather than re-anchoring to "now", which is what kills
// the beat. Hybrid wait — Sleep() to ~1ms short, then spin to the mark — for
// accuracy without burning a core. Target comes from Configuration.FPSLimit
// (MGE.ini "FPS Limit"); 0 = off. The engine's own limiter is neutralized by
// MGEgui forcing Morrowind.ini "Max FPS" high so it never triggers.
static void framePaceLimit() {
    const int target = Configuration.FPSLimit;

    static LARGE_INTEGER freq = {};
    static double nextTick = 0.0;     // QPC ticks: deadline for the current present
    static bool periodSet = false;

    if (target <= 0) { nextTick = 0.0; return; }   // off; reset so re-enabling resyncs

    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    if (!periodSet) { timeBeginPeriod(1); periodSet = true; }

    const double period = double(freq.QuadPart) / double(target);

    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    if (nextTick == 0.0) {            // first paced frame: seed the deadline
        nextTick = double(now.QuadPart) + period;
        return;
    }

    for (;;) {
        QueryPerformanceCounter(&now);
        const double remain = nextTick - double(now.QuadPart);
        if (remain <= 0.0) break;
        const double remMs = remain * 1000.0 / double(freq.QuadPart);
        if (remMs > 2.0) {
            Sleep((DWORD)(remMs - 1.0));   // coarse sleep to ~1ms short
        } else {
            YieldProcessor();              // fine spin to the exact mark
        }
    }

    nextTick += period;                    // accumulator anchor (kills the beat)

    // Catch-up clamp: after a hitch/load-screen/alt-tab we may be many periods
    // behind. Resync instead of sprinting through catch-up frames.
    QueryPerformanceCounter(&now);
    if (double(now.QuadPart) - nextTick > period) {
        nextTick = double(now.QuadPart) + period;
    }
}

static bool isTracyProfilerRunning() {
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return false;
    PROCESSENTRY32 pe;
    pe.dwSize = sizeof(pe);
    bool found = false;
    if (Process32First(snap, &pe)) {
        do {
            if (_strnicmp(pe.szExeFile, "tracy", 5) == 0) {
                found = true;
                break;
            }
        } while (Process32Next(snap, &pe));
    }
    CloseHandle(snap);
    return found;
}

static int sceneCount;
static bool rendertargetNormal, isHUDready;
static bool isMainView, isStencilScene, isAmbientWhite;

#ifdef TRACY_ENABLE
static constexpr tracy::SourceLocationData s_sceneZoneLocs[] = {
    { "scene0", "BeginScene", __FILE__, 0, 0 },
    { "scene1", "BeginScene", __FILE__, 0, 0 },
    { "scene2+", "BeginScene", __FILE__, 0, 0 },
    { "UI",     "BeginScene", __FILE__, 0, 0 },
};
static tracy::ScopedZone* s_currentSceneZone = nullptr;

// ---------------------------------------------------------------------------
// GPU frame timer (diagnostic). The vendored Tracy has no D3D9 backend, so
// instead of a full GPU zone context this measures total GPU time per frame
// with D3D9 timestamp queries and emits it as the Tracy plot "GPU frame (ms)".
// Purpose: decide whether the frame is GPU-bound — if GPU ms ~= frame ms, the
// frame is GPU-bound. Read-back is deferred across a small ring so the CPU never
// blocks on GetData. The span for frame F is bracketed beginFrame() (at the
// first scene's BeginScene — start of frame F's GPU work) .. endFrame() (at
// Present F). This measures GPU BUSY time: inter-frame idle (the frame
// limiter's pace wait + present/back-buffer wait) falls between Present F's
// tsEnd and frame F+1's tsStart, OUTSIDE the span — so the plot stays honest
// when the limiter is active.
namespace {
struct GpuFrameTimer {
    static constexpr int N = 4;
    struct Slot {
        IDirect3DQuery9* tsStart  = nullptr;
        IDirect3DQuery9* tsEnd    = nullptr;
        IDirect3DQuery9* tsSky0   = nullptr;  // MW sky window start (within frame disjoint)
        IDirect3DQuery9* tsSky1   = nullptr;  // MW sky window end (= renderStage0)
        IDirect3DQuery9* disjoint = nullptr;
        IDirect3DQuery9* freq     = nullptr;
        bool inFlight = false;
        bool skyValid = false;   // both sky timestamps issued this frame → emit "GPU sky (ms)"
    };
    Slot slots[N];
    int  cur = -1;
    bool initialized = false;
    bool disabled = false;
    bool spanOpen = false;   // beginFrame opened a span this frame (endFrame closes it)
    bool skyOpen  = false;   // markSkyStart issued tsSky0, awaiting markSkyEnd

    void releaseAll() {
        for (auto& s : slots) {
            if (s.tsStart)  { s.tsStart->Release();  s.tsStart  = nullptr; }
            if (s.tsEnd)    { s.tsEnd->Release();    s.tsEnd    = nullptr; }
            if (s.tsSky0)   { s.tsSky0->Release();   s.tsSky0   = nullptr; }
            if (s.tsSky1)   { s.tsSky1->Release();   s.tsSky1   = nullptr; }
            if (s.disjoint) { s.disjoint->Release(); s.disjoint = nullptr; }
            if (s.freq)     { s.freq->Release();     s.freq     = nullptr; }
            s.inFlight = false;
            s.skyValid = false;
        }
        initialized = false;
        spanOpen = false;
        skyOpen  = false;
        cur = -1;
    }

    bool ensureInit(IDirect3DDevice9* dev) {
        if (initialized) return true;
        if (disabled || !dev) return false;
        for (auto& s : slots) {
            if (FAILED(dev->CreateQuery(D3DQUERYTYPE_TIMESTAMP,         &s.tsStart))  ||
                FAILED(dev->CreateQuery(D3DQUERYTYPE_TIMESTAMP,         &s.tsEnd))    ||
                FAILED(dev->CreateQuery(D3DQUERYTYPE_TIMESTAMP,         &s.tsSky0))   ||
                FAILED(dev->CreateQuery(D3DQUERYTYPE_TIMESTAMP,         &s.tsSky1))   ||
                FAILED(dev->CreateQuery(D3DQUERYTYPE_TIMESTAMPDISJOINT, &s.disjoint)) ||
                FAILED(dev->CreateQuery(D3DQUERYTYPE_TIMESTAMPFREQ,     &s.freq))) {
                disabled = true;     // GPU lacks timestamp queries — give up silently
                releaseAll();
                return false;
            }
        }
        initialized = true;
        return true;
    }

    // Mark the GPU-side start of the MW sky window (called right after the sky
    // Tracy zone opens). Records a timestamp inside the live frame disjoint block.
    void markSkyStart() {
        if (!initialized || !spanOpen || skyOpen) return;
        slots[cur].tsSky0->Issue(D3DISSUE_END);
        skyOpen = true;
    }

    // Mark the GPU-side end of the MW sky window (= first world draw / renderStage0).
    // Idempotent: only the first call per frame records the end.
    void markSkyEnd() {
        if (!initialized || !skyOpen) return;
        slots[cur].tsSky1->Issue(D3DISSUE_END);
        slots[cur].skyValid = true;
        skyOpen = false;
    }

    // Close the current frame's span (before the real Present). No-op unless
    // beginFrame opened a span this frame (e.g. a present with no main scene).
    void endFrame() {
        if (!initialized || !spanOpen) return;
        Slot& s = slots[cur];
        s.tsEnd->Issue(D3DISSUE_END);
        s.freq->Issue(D3DISSUE_END);
        s.disjoint->Issue(D3DISSUE_END);
        s.inFlight = true;
        spanOpen = false;
    }

    // Non-blocking read-back of any completed slots → emit the plot.
    void readCompleted() {
        if (!initialized) return;
        for (auto& s : slots) {
            if (!s.inFlight) continue;
            BOOL   dis = FALSE;
            UINT64 f = 0, t0 = 0, t1 = 0;
            if (s.disjoint->GetData(&dis, sizeof(dis), 0) != S_OK) continue;
            if (s.freq->GetData(&f, sizeof(f), 0)         != S_OK) continue;
            if (s.tsStart->GetData(&t0, sizeof(t0), 0)    != S_OK) continue;
            if (s.tsEnd->GetData(&t1, sizeof(t1), 0)      != S_OK) continue;
            s.inFlight = false;
            if (!dis && f != 0 && t1 >= t0) {
                MGE_TracyPlot("GPU frame (ms)", double(t1 - t0) / double(f) * 1000.0);
                if (s.skyValid) {
                    UINT64 sk0 = 0, sk1 = 0;
                    if (s.tsSky0->GetData(&sk0, sizeof(sk0), 0) == S_OK &&
                        s.tsSky1->GetData(&sk1, sizeof(sk1), 0) == S_OK &&
                        sk1 >= sk0) {
                        MGE_TracyPlot("GPU sky (ms)", double(sk1 - sk0) / double(f) * 1000.0);
                    }
                }
            }
            s.skyValid = false;
        }
    }

    // Open this frame's span at the start of GPU work (first scene's BeginScene).
    void beginFrame() {
        if (!initialized || spanOpen) return;
        cur = (cur + 1) % N;
        Slot& s = slots[cur];
        s.inFlight = false;                  // drop any unread slot we're reusing
        s.skyValid = false;
        skyOpen = false;
        s.disjoint->Issue(D3DISSUE_BEGIN);
        s.tsStart->Issue(D3DISSUE_END);
        spanOpen = true;
    }
};
GpuFrameTimer g_gpuTimer;
} // namespace
#endif
static DWORD stencilRef;
static bool stage0Complete, isFrameComplete, isHUDComplete;
static bool isWaterMaterial, waterDrawn, distantWater;

static bool zoomSensSaved;
static float zoomSensX, zoomSensY;
static D3DXMATRIX camEffectsMatrix;
static float crosshairTimeout;

static RenderedState rs;
static FragmentState frs;
static LightState lightrs;

static void initOnLoad();
static bool detectMenu(const D3DMATRIX* m);
static void captureRenderState(D3DRENDERSTATETYPE a, DWORD b);
static void captureFragmentRenderState(DWORD a, D3DTEXTURESTAGESTATETYPE b, DWORD c);
static void captureTransform(D3DTRANSFORMSTATETYPE a, const D3DMATRIX* b);
static void captureLight(DWORD a, const D3DLIGHT8* b);
static void captureMaterial(const D3DMATERIAL8* a);
static float calcFPS();



MGEProxyDevice::MGEProxyDevice(IDirect3DDevice9* real, ProxyD3D* d3d) : ProxyDevice(real, d3d) {
    // Initialize state here, as the device is released and recreated on fullscreen Alt-Tab
    sceneCount = -1;
    rendertargetNormal = true;
    isHUDready = false;
    isMainView = isStencilScene = isAmbientWhite = stage0Complete = isFrameComplete = isHUDComplete = false;
    stencilRef = 0;
    isWaterMaterial = waterDrawn = false;
    D3DXMatrixIdentity(&camEffectsMatrix);

    Configuration.CameraEffects.zoom = 1.0;
    Configuration.CameraEffects.zoomRate = 0;
    Configuration.CameraEffects.zoomRateTarget = 0;

    // Initialize state recorder to D3D defaults
    memset(&rs, 0, sizeof(rs));
    rs.zWrite = true;
    rs.diffuseMaterial.r = 1.0f;
    rs.diffuseMaterial.g = 1.0f;
    rs.diffuseMaterial.b = 1.0f;
    rs.diffuseMaterial.a = 1.0f;
    rs.cullMode = D3DCULL_CCW;
    rs.useLighting = true;

    rs.matSrcDiffuse = D3DMCS_COLOR1;
    rs.matSrcEmissive = D3DMCS_MATERIAL;

    memset(&frs, 0, sizeof(frs));
    for (FragmentState::Stage* s = &frs.stage[0]; s != &frs.stage[8]; ++s) {
        s->colorOp = D3DTOP_DISABLE;
        s->alphaOp = D3DTOP_DISABLE;
        s->colorArg1 = s->alphaArg1 = D3DTA_TEXTURE;
        s->colorArg2 = s->alphaArg2 = D3DTA_CURRENT;
        s->colorArg0 = s->alphaArg0 = s->resultArg = D3DTA_CURRENT;
    }
    frs.stage[0].colorOp = D3DTOP_MODULATE;
    frs.stage[0].alphaOp = D3DTOP_SELECTARG1;

    lightrs.lights.clear();
    lightrs.active.clear();

    // Detect Tracy profiler for conditional zone activation
    g_tracyActive = isTracyProfilerRunning();
    LOG::logline("Tracy: profiler %s, zones %s", g_tracyActive ? "detected" : "not detected", g_tracyActive ? "enabled" : "disabled");

    // Store active device in distant land, occurs on startup and after fullscreen alt-tab
    DistantLand::device = realDevice;

    // Patch splash screen minor issues
    D3DVIEWPORT9 vp;
    realDevice->GetViewport(&vp);
    MWBridge::get()->patchSplashScreen(vp.Width, vp.Height);
}

// Present - End of MW frame
// MGE end of frame processing
HRESULT _stdcall MGEProxyDevice::Present(const RECT* a, const RECT* b, HWND c, const RGNDATA* d) {
    auto mwBridge = MWBridge::get();

    // Load Morrowind's dynamic memory pointers
    if (!mwBridge->IsLoaded() && mwBridge->CanLoad()) {
        mwBridge->Load();

        // Apply patch to load distant land before the main menu, and on renderer restart
        mwBridge->patchGameLoading(&initOnLoad);
        // Patch world rendering (on a branch without the water) to split alphas to their own scene
        mwBridge->patchWorldRenderingAccumulation();
        // Disable MW screenshot function to allow MGE to use the same key
        mwBridge->disableScreenshotFunc();
        // Mark water material to allow MGEProxyDevice to detect it
        mwBridge->markWaterNode(99999.0f);
    }

    if (mwBridge->IsLoaded()) {
        if (Configuration.Force3rdPerson && DistantLand::ready) {
            // Set 3rd person camera
            D3DXVECTOR3* camera = mwBridge->PCam3Offset();
            if (camera) {
                camera->x = Configuration.Offset3rdPerson.x;
                camera->y = Configuration.Offset3rdPerson.y;
                camera->z = Configuration.Offset3rdPerson.z;
            }
        }

        if ((Configuration.MGEFlags & CROSSHAIR_AUTOHIDE) && !mwBridge->IsLoadScreen()) {
            // Update crosshair visibility
            float t = mwBridge->simulationTime();

            // Turn on if Morrowind ray cast picks up a target
            if (mwBridge->getPlayerTarget()) {
                crosshairTimeout = t + 1.5f;
            }

            // Turn on short duration if the player requires aim
            if (mwBridge->isPlayerCasting() || mwBridge->isPlayerAimingWeapon()) {
                crosshairTimeout = t + 0.5f;
            }

            // Turn off in menu mode
            if (mwBridge->IsMenu()) {
                crosshairTimeout = t;
            }

            // Allow manual toggle of crosshair to work again from 0.5 seconds after timeout
            if (t < crosshairTimeout + 0.5) {
                mwBridge->SetCrosshairEnabled(t < crosshairTimeout);
            }
        }

        if (Configuration.CameraEffects.zoomRateTarget != 0 && !mwBridge->IsMenu()) {
            // Update zoom controller
            Configuration.CameraEffects.zoomRate += 0.25f * Configuration.CameraEffects.zoomRateTarget * mwBridge->frameTime();
            if (Configuration.CameraEffects.zoomRate / Configuration.CameraEffects.zoomRateTarget > 1.0) {
                Configuration.CameraEffects.zoomRate = Configuration.CameraEffects.zoomRateTarget;
            }

            Configuration.CameraEffects.zoom += Configuration.CameraEffects.zoomRate * mwBridge->frameTime();
            Configuration.CameraEffects.zoom = std::max(1.0f, Configuration.CameraEffects.zoom);
            Configuration.CameraEffects.zoom = std::min(Configuration.CameraEffects.zoom, 8.0f);
        }

        float* mwSens = mwBridge->getMouseSensitivityYX();
        if ((Configuration.MGEFlags & ZOOM_ASPECT) && !mwBridge->IsMenu()) {
            // Adjust sensitivity to accommodate zoom level
            if (!zoomSensSaved) {
                zoomSensY = mwSens[0];
                zoomSensX = mwSens[1];
                zoomSensSaved = true;
            }
            mwSens[0] = zoomSensY / Configuration.CameraEffects.zoom;
            mwSens[1] = zoomSensX / Configuration.CameraEffects.zoom;
        } else if (zoomSensSaved) {
            // Restore unzoomed sensitivity
            mwSens[0] = zoomSensY;
            mwSens[1] = zoomSensX;
            zoomSensSaved = false;
        }

        if (Configuration.CameraEffects.rotateUpdate) {
            Configuration.CameraEffects.rotation += Configuration.CameraEffects.rotationRate * mwBridge->frameTime();
            D3DXMatrixRotationZ(&camEffectsMatrix, Configuration.CameraEffects.rotation);
            if (Configuration.CameraEffects.rotationRate == 0) {
                Configuration.CameraEffects.rotateUpdate = false;
            }
        }
        if (Configuration.CameraEffects.shake) {
            // Update screen shake controller
            Configuration.CameraEffects.shakeMagnitude += Configuration.CameraEffects.shakeAccel * mwBridge->frameTime();
            Configuration.CameraEffects.shakeMagnitude = std::max(0.0f, std::min(100.0f, Configuration.CameraEffects.shakeMagnitude));
            camEffectsMatrix._41 = Configuration.CameraEffects.shakeMagnitude * sin(0.001f*GetTickCount());
        }

        // Main menu background video
        VideoPatch::monitor(realDevice);
    }

    // Reset scene identifiers
    sceneCount = -1;
    stage0Complete = false;
    waterDrawn = false;
    isFrameComplete = false;
    isHUDComplete = false;

#ifdef TRACY_ENABLE
    if (g_tracyActive) {
        // CPU frame period (Present -> Present) — shows the limiter beat directly.
        static LARGE_INTEGER s_periodFreq = {};
        static LARGE_INTEGER s_periodLast = {};
        if (s_periodFreq.QuadPart == 0) QueryPerformanceFrequency(&s_periodFreq);
        LARGE_INTEGER nowQpc; QueryPerformanceCounter(&nowQpc);
        if (s_periodLast.QuadPart != 0) {
            MGE_TracyPlot("CPU frame (ms)",
                1000.0 * double(nowQpc.QuadPart - s_periodLast.QuadPart) / double(s_periodFreq.QuadPart));
        }
        s_periodLast = nowQpc;

        // Engine frame-limiter probe: timer reads/frame (spin = thousands, sleep
        // = a handful) plus which engine call site does them (logged every 60
        // frames so we can localize the limiter loop for the fix).
        unsigned timerCalls = TimerProbe::takeFrameCallCount();
        MGE_TracyPlot("engine time() reads", (double)timerCalls);
        static int s_probeFrame = 0;
        if ((s_probeFrame++ % 60) == 0) {
            TimerProbe::CallerStat callers[8];
            int nc = TimerProbe::takeCallerStats(callers, 8);
            char buf[256];
            int off = _snprintf_s(buf, sizeof(buf), _TRUNCATE,
                                  "[timerprobe] reads(last frame)=%u  callers(60f):", timerCalls);
            for (int i = 0; i < nc && off > 0 && off < (int)sizeof(buf) - 24; ++i) {
                int w = _snprintf_s(buf + off, sizeof(buf) - off, _TRUNCATE,
                                    " %p=%u", callers[i].addr, callers[i].count);
                if (w <= 0) break;
                off += w;
            }
            LOG::logline("%s", buf);
        }

        g_gpuTimer.ensureInit(realDevice);
        g_gpuTimer.endFrame();       // close frame F's GPU span (before pace + present)
        g_gpuTimer.readCompleted();  // emit any finished frame's GPU ms
        // beginFrame() is issued at the first scene's BeginScene, not here, so
        // the limiter's inter-frame idle stays outside the measured span.
    }
#endif

    // MGE frame limiter: pace to the target before presenting. Off when
    // Configuration.FPSLimit == 0.
    {
        MGE_ZoneScopedN("framePace");
        framePaceLimit();
    }

    HRESULT hr;
    {
        MGE_ZoneScopedN("engPresent");
        hr = ProxyDevice::Present(a, b, c, d);
    }

#ifdef TRACY_ENABLE
    if (g_tracyActive) g_gpuTimer.beginFrame();  // open frame F+1's GPU span
#endif
    DrawStats::logFrame();
    MGE_FrameMark;
    return hr;
}

// SetRenderTarget
// Remember if MW is rendering to back buffer
HRESULT _stdcall MGEProxyDevice::SetRenderTarget(IDirect3DSurface8* a, IDirect3DSurface8* b) {
    if (a) {
        IDirect3DSurface9* back;
        realDevice->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &back);
        rendertargetNormal = (static_cast<ProxySurface*>(a)->realSurface == back);
        back->Release();
    }

    return ProxyDevice::SetRenderTarget(a, b);
}

// BeginScene - Multiple scenes per frame, non-alpha / 2x stencil / post-stencil redraw / alpha / 1st person / UI
// Fogging needs to be set for Morrowind rendering at start of scene
HRESULT _stdcall MGEProxyDevice::BeginScene() {
    auto mwBridge = MWBridge::get();

    HRESULT hr;
    {
        MGE_ZoneScopedN("engBeginScene");
        hr = ProxyDevice::BeginScene();
    }
    if (hr != D3D_OK) {
        return hr;
    }

    if (mwBridge->IsLoaded() && rendertargetNormal) {
        if (!isHUDready) {
            // Initialize HUD
            StatusOverlay::init(realDevice);
            StatusOverlay::setStatus(XE_VERSION_STRING);
            MGEhud::init(realDevice);

            // Set scaling on Morrowind's UI system
            if (Configuration.UIScale != 1.0f) {
                mwBridge->setUIScale(Configuration.UIScale);
            }

            isHUDready = true;
        }

        if (isMainView) {
            // Track scene count here in BeginScene
            // isMainView is not always valid at EndScene if Morrowind draws sunglare
            ++sceneCount;

            // Tag the current draw-stats bucket for this Morrowind scene. The
            // distant-land passes triggered mid-scene0 push their own ScopedStage
            // and restore back to this on exit.
            DrawStats::g_stage = (DrawStats::Stage)(DrawStats::Scene0 + (sceneCount <= 2 ? sceneCount : 2));

#ifdef TRACY_ENABLE
            if (g_tracyActive) {
                int idx = sceneCount <= 1 ? sceneCount : 2;
                s_currentSceneZone = new tracy::ScopedZone(&s_sceneZoneLocs[idx], 0, true);
            }
#endif
            if (sceneCount == 0) {
#ifdef TRACY_ENABLE
                // Open the GPU-busy-time span at the start of the frame's GPU
                // work (paired with endFrame() at Present). ensureInit ran in a
                // prior Present, so this is live from frame 1.
                if (g_tracyActive) g_gpuTimer.beginFrame();
#endif
            }

            // Set any custom FOV and check distant water state
            if (sceneCount == 0) {
                if (Configuration.ScreenFOV > 0) {
                    mwBridge->SetFOV(Configuration.ScreenFOV);
                }
                distantWater = (Configuration.MGEFlags & USE_DISTANT_LAND) || (Configuration.MGEFlags & USE_DISTANT_WATER);

                // Kick off the distant-statics cull now — the camera is already
                // this-frame-valid (after the FOV override above), so its ~4ms
                // server-side compute overlaps the engine's sky pass instead of
                // stalling cullDistantStatics_finish later. renderStage0 detects
                // this and skips the redundant setup + kickoff.
                if (DistantLand::ready) {
                    DistantLand::frameSetupEarly();
                }

                // Open the MW sky zone *after* frameSetupEarly so it brackets only
                // the engine's true sky pass — not our main-thread geometry-cache
                // walk / cull kickoff, which frameSetupEarly parks here to overlap
                // the cull worker. Closed at the top of renderStage0().
                DistantLand::beginSkyZone();
#ifdef TRACY_ENABLE
                if (g_tracyActive) g_gpuTimer.markSkyStart();
#endif
            }
        } else {
            DrawStats::g_stage = DrawStats::UI;
#ifdef TRACY_ENABLE
            if (g_tracyActive)
                s_currentSceneZone = new tracy::ScopedZone(&s_sceneZoneLocs[3], 0, true);
#endif
            // UI scene, apply post-process if there was anything drawn before it
            // Race menu will render an extra scene past this point
            if (DistantLand::ready && sceneCount > 0 && !isFrameComplete) {
                DistantLand::postProcess();
            }

            // Render user HUD before Morrowind HUD
            if (isHUDready && !isHUDComplete) {
                MGEhud::draw();
            }

            isFrameComplete = true;
        }
    }

    return D3D_OK;
}

// EndScene - Multiple scenes per frame, non-alpha / 2x stencil / post-stencil redraw / alpha / 1st person / UI
// MGE intercepts first scene to draw distant land before it finishes, others it applies shadows to
HRESULT _stdcall MGEProxyDevice::EndScene() {
    if (DistantLand::ready && rendertargetNormal) {
        // The following Morrowind scenes get past the filters:
        // ~ Opaque meshes, plus alpha meshes with 'No Sorter' property (which should use alpha test)
        // ~ If stencil shadows are active, then shadow casters are deferred to be drawn in a scene after
        //    shadows are fully applied to avoid self-shadowing problems with simplified shadow meshes
        // ~ If any alpha meshes are visible, they are sorted and drawn in another scene (except those with 'No Sorter' property)
        // ~ If 1st person or sunglare is visible, they are drawn in another scene after a Z clear
        if (sceneCount == 0) {
            // Edge case, render distant land even if Morrowind has culled everything
            if (!stage0Complete) {
#ifdef TRACY_ENABLE
                if (g_tracyActive) g_gpuTimer.markSkyEnd();
#endif
                DistantLand::renderStage0();
                stage0Complete = true;
            }

            // Opaque features
            DistantLand::renderStage1();

            // Blend close objects over distant land
            DistantLand::renderStageBlend();
        } else if (!isFrameComplete) {
            // Everything else except UI
            DistantLand::renderStage2();

            // Draw water if the Morrowind water plane doesn't appear in view
            // it may be too distant or stencil scene order is non-normative
            if (distantWater && !waterDrawn && !isStencilScene) {
                DistantLand::renderStageWater();
                waterDrawn = true;
            }
        }
    }

    if (isFrameComplete && isHUDready && !isHUDComplete) {
        // Capture post-UI screenshots here
        DistantLand::checkCaptureScreenshot(true);

        // Render status overlay
        StatusOverlay::setFPS(calcFPS());
        StatusOverlay::show(realDevice);

        isHUDComplete = true;
    }

#ifdef TRACY_ENABLE
    delete s_currentSceneZone;
    s_currentSceneZone = nullptr;
#endif

    return ProxyDevice::EndScene();
}

// Clear - Occurs at start of frame, and also a z-clear before rendering 1st person and sunglare
// Skybox mesh doesn't extend over whole background; cleared background colour is visible at horizon
HRESULT _stdcall MGEProxyDevice::Clear(DWORD a, const D3DRECT* b, DWORD c, D3DCOLOR d, float e, DWORD f) {
    DistantLand::setHorizonColour(d);
    // Zone the clear during the scene-0 sky window — a Clear is a classic
    // point at which the CPU blocks on GPU backpressure, and that wait would
    // otherwise hide inside the "MW sky" zone.
    if (isMainView && sceneCount == 0 && !stage0Complete) {
        MGE_ZoneScopedN("MWsky:Clear");
        return ProxyDevice::Clear(a, b, c, d, e, f);
    }
    return ProxyDevice::Clear(a, b, c, d, e, f);
}

// SetTransform
// Projection needs modifying to allow room for distant land
HRESULT _stdcall MGEProxyDevice::SetTransform(D3DTRANSFORMSTATETYPE a, const D3DMATRIX* b) {
    captureTransform(a, b);

    if (rendertargetNormal) {
        if (a == D3DTS_VIEW) {
            // Check for UI view
            isMainView = !detectMenu(b);

            if (isMainView) {
                D3DXMATRIX view = *b;
                view *= camEffectsMatrix;
                return ProxyDevice::SetTransform(a, &view);
            }
        } else if (a == D3DTS_PROJECTION) {
            // Only screw with main scene projection
            if (isMainView) {
                D3DXMATRIX proj = *b;
                DistantLand::setProjection(&proj);

                if (Configuration.MGEFlags & ZOOM_ASPECT) {
                    proj._11 *= Configuration.CameraEffects.zoom;
                    proj._22 *= Configuration.CameraEffects.zoom;
                }

                return ProxyDevice::SetTransform(a, &proj);
            }
        }
    }

    return ProxyDevice::SetTransform(a, b);
}

// SetMaterial
// Check for materials marked for hiding
HRESULT _stdcall MGEProxyDevice::SetMaterial(const D3DMATERIAL8* a) {
    captureMaterial(a);
    isWaterMaterial = (a->Power == 99999.0f);

    return ProxyDevice::SetMaterial(a);
}

// SetLight
// Capture what the sun is doing
HRESULT _stdcall MGEProxyDevice::SetLight(DWORD a, const D3DLIGHT8* b) {
    captureLight(a, b);

    // Exterior sunlight/interior "sun" appears to always be light 6
    if (a == 6 && DistantLand::ready) {
        DistantLand::setSunLight(b);
    }

    return ProxyDevice::SetLight(a, b);
}

// SetRenderState
// Ignore Morrowind fog settings, and run stage 0 rendering after lighting setup
HRESULT _stdcall MGEProxyDevice::SetRenderState(D3DRENDERSTATETYPE a, DWORD b) {
    captureRenderState(a, b);

    if (a == D3DRS_FOGVERTEXMODE || a == D3DRS_FOGTABLEMODE) {
        return D3D_OK;
    }
    if ((Configuration.MGEFlags & USE_DISTANT_LAND) && (a == D3DRS_FOGSTART || a == D3DRS_FOGEND)) {
        return D3D_OK;
    }
    if (a == D3DRS_STENCILENABLE) {
        isStencilScene = b;
    }
    else if (a == D3DRS_STENCILREF) {
        stencilRef = b;
    }

    // Ambient is used for scene detection
    if (a == D3DRS_AMBIENT) {
        // Pure white ambient occurs with skydome and menu mode rendering
        // Ambient is also never set properly when high enough outside that Morrowind renders nothing
        isAmbientWhite = (b == 0xffffffff);

        if (!isAmbientWhite) {
            // Save real ambient, can be used in future frames if no draw calls are provoked
            RGBVECTOR amb = D3DCOLOR(b);
            DistantLand::setAmbientColour(amb);
            lightrs.globalAmbient.r = amb.r;
            lightrs.globalAmbient.g = amb.g;
            lightrs.globalAmbient.b = amb.b;
        }
    }

    return ProxyDevice::SetRenderState(a, b);
}

// SetTextureStageState
// Override some sampler options
HRESULT _stdcall MGEProxyDevice::SetTextureStageState(DWORD a, D3DTEXTURESTAGESTATETYPE b, DWORD c) {
    // Whole-method lock: this override submits directly via realDevice->
    // SetSamplerState in two branches (bypassing the locked ProxyDevice base),
    // so the bare-forward lock in the base would miss those. Base
    // SetTextureStageState is intentionally left unlocked to avoid a re-lock.
    MGE_DEVLOCK();
    captureFragmentRenderState(a, b, c);

    // Sampler overrides to ensure trilinear/anisotropic filtering works
    // Note that DX8 had sampling state bound to texture stages instead of samplers
    if (b == D3DTSS_MINFILTER) {
        DWORD filter = (c != D3DTEXF_NONE) ? Configuration.ScaleFilter : D3DTEXF_NONE;
        return realDevice->SetSamplerState(a, D3DSAMP_MINFILTER, filter);
    } else if (b == D3DTSS_MIPFILTER) {
        DWORD filter = (c != D3DTEXF_NONE) ? D3DTEXF_LINEAR : D3DTEXF_NONE;
        return realDevice->SetSamplerState(a, D3DSAMP_MIPFILTER, filter);
    }

    return ProxyDevice::SetTextureStageState(a, b, c);
}

// DrawIndexedPrimitive - Where all the drawing happens
// Inspect draw calls for re-use later
HRESULT _stdcall MGEProxyDevice::DrawIndexedPrimitive(D3DPRIMITIVETYPE a, UINT b, UINT c, UINT d, UINT e) {
    // Allow distant land to inspect draw calls
    bool isShadowStencil = isStencilScene && stencilRef <= 1;
    if (DistantLand::ready && rendertargetNormal && isMainView && !isShadowStencil) {
        rs.primType = a;
        rs.baseIndex = baseVertexIndex;
        rs.minIndex = b;
        rs.vertCount = c;
        rs.startIndex = d;
        rs.primCount = e;

        if (!stage0Complete && !isAmbientWhite) {
            // At this point, only the sky is rendered in exteriors, or nothing in interiors
#ifdef TRACY_ENABLE
            if (g_tracyActive) g_gpuTimer.markSkyEnd();
#endif
            DistantLand::renderStage0();
            stage0Complete = true;
            DistantLand::beginDrawsZone();
        }

        if (isWaterMaterial) {
            if (distantWater) {
                // Call distant land instead of drawing water grid
                if (!waterDrawn) {
                    DistantLand::renderStageWater();
                    waterDrawn = true;
                }
                return D3D_OK;
            }
        } else {
            // Let distant land record call and skip if signalled
            if (!DistantLand::inspectIndexedPrimitive(sceneCount, &rs, &frs, &lightrs)) {
                return D3D_OK;
            }
        }
    }

    // Zone the engine's scene-0 sky draws (everything before the first world
    // draw triggers renderStage0). Localizes whether the long "MW sky" window
    // is spent in these draws vs. before the first draw (BeginScene/Clear).
    if (isMainView && sceneCount == 0 && !stage0Complete) {
        MGE_ZoneScopedN("MWsky:DIP");
        return ProxyDevice::DrawIndexedPrimitive(a, b, c, d, e);
    }
    return ProxyDevice::DrawIndexedPrimitive(a, b, c, d, e);
}

// Release - Free all resources when refcount hits 0
ULONG _stdcall MGEProxyDevice::Release() {
    // Join the render thread BEFORE ProxyDevice::Release frees the real device
    // (it releases realDevice at refcount 0). refcount==1 here means this call
    // drops it to 0. The worker may still be mid-pass touching the device /
    // depth RT, so drain + join it while everything is still alive.
    if (refcount == 1) {
        MGE::RenderThread::stop();
    }

    ULONG r = ProxyDevice::Release();

    if (r == 0) {
#ifdef TRACY_ENABLE
        g_gpuTimer.releaseAll();
#endif
        DistantLand::release();
        MGEhud::release();
        StatusOverlay::release();
    }

    return r;
}

// --------------------------------------------------------

// initOnLoad
// Initializes distant land
// Called after new game or load game is selected from the main menu
void initOnLoad() {
    auto mwBridge = MWBridge::get();

    // Compose loading message from translated string
    char buffer[64];
    const char* loadingMessage = *(const char**)mwBridge->getGMSTPointer(602);
    int firstWordLength = 0;

    for (const char *c = loadingMessage; *c; ++c) {
        if (*c == ' ') { break; }
        ++firstWordLength;
    }

    std::snprintf(buffer, sizeof(buffer), "%.*s MGE XE...", firstWordLength, loadingMessage);
    mwBridge->showLoadingBar(buffer, 95.0);

    // Initialize distant land
    if (DistantLand::init()) {
        // Initially force view distance to max, required for full extent shadows and grass
        if (Configuration.MGEFlags & USE_DISTANT_LAND) {
            mwBridge->SetViewDistance(7168.0);
        }
    } else {
        Configuration.MGEFlags &= ~USE_DISTANT_LAND;
        StatusOverlay::setStatus("MGE XE serious error condition. Exit Morrowind and check mgeXE.log for details.", StatusOverlay::PriorityError);
    }

    // Clean up loading bar menu, otherwise it persists in the background
    mwBridge->destroyLoadingBar();

    VideoPatch::start(DistantLand::device);
}

// detectMenu
// detects if view matrix is for UI / load bars
// the projection matrix is never set to ortho, making it unusable for detection
bool detectMenu(const D3DMATRIX* m) {
    if (m->_41 != 0.0f || !(m->_42 == 0.0f || m->_42 == -600.0f) || m->_43 != 0.0f) {
        return false;
    }

    if ((m->_11 == 0.0f || m->_11 == 1.0f) && m->_12 == 0.0f && (m->_13 == 0.0f || m->_13 == 1.0f) &&
            m->_21 == 0.0f && (m->_22 == 0.0f || m->_22 == 1.0f) && (m->_23 == 0.0f || m->_23 == 1.0f) &&
            (m->_31 == 0.0f || m->_31 == 1.0f) && (m->_32 == 0.0f || m->_32 == 1.0f) && m->_33 == 0.0f) {
        return true;
    }

    return false;
}

// --------------------------------------------------------
// State recording

HRESULT _stdcall MGEProxyDevice::SetTexture(DWORD a, IDirect3DBaseTexture8* b) {
    if (a == 0) {
        rs.texture = b ? static_cast<ProxyTexture*>(b)->realTexture : NULL;
    }
    return ProxyDevice::SetTexture(a, b);
}

HRESULT _stdcall MGEProxyDevice::SetVertexShader(DWORD a) {
    rs.fvf = a;
    return ProxyDevice::SetVertexShader(a);
}

HRESULT _stdcall MGEProxyDevice::SetStreamSource(UINT a, IDirect3DVertexBuffer8* b, UINT c) {
    if (a == 0) {
        rs.vb = (IDirect3DVertexBuffer9*)b;
        rs.vbOffset = 0;
        rs.vbStride = c;
    }
    return ProxyDevice::SetStreamSource(a, b, c);
}

HRESULT _stdcall MGEProxyDevice::SetIndices(IDirect3DIndexBuffer8* a, UINT b) {
    rs.ib = (IDirect3DIndexBuffer9*)a;
    return ProxyDevice::SetIndices(a, b);
}

HRESULT _stdcall MGEProxyDevice::LightEnable(DWORD a, BOOL b) {
    if (b) {
        if (std::find(lightrs.active.begin(), lightrs.active.end(), a) == lightrs.active.end()) {
            lightrs.active.push_back(a);
        }
    } else {
        if (std::remove(lightrs.active.begin(), lightrs.active.end(), a) != lightrs.active.end()) {
            lightrs.active.pop_back();
        }
    }
    return ProxyDevice::LightEnable(a, b);
}

void captureRenderState(D3DRENDERSTATETYPE a, DWORD b) {
    switch (a) {
    case D3DRS_VERTEXBLEND:
        rs.vertexBlendState = b;
        break;
    case D3DRS_ZWRITEENABLE:
        rs.zWrite = b;
        break;
    case D3DRS_CULLMODE:
        rs.cullMode = b;
        break;
    case D3DRS_ALPHABLENDENABLE:
        rs.blendEnable = (BYTE)b;
        break;
    case D3DRS_SRCBLEND:
        rs.srcBlend = (BYTE)b;
        break;
    case D3DRS_DESTBLEND:
        rs.destBlend = (BYTE)b;
        break;
    case D3DRS_ALPHATESTENABLE:
        rs.alphaTest = (BYTE)b;
        break;
    case D3DRS_ALPHAFUNC:
        rs.alphaFunc = (BYTE)b;
        break;
    case D3DRS_ALPHAREF:
        rs.alphaRef = (BYTE)b;
        break;
    case D3DRS_LIGHTING:
        rs.useLighting = (BYTE)b;
        break;
    case D3DRS_FOGENABLE:
        rs.useFog = (BYTE)b;
        break;
    case D3DRS_DIFFUSEMATERIALSOURCE:
        rs.matSrcDiffuse = (BYTE)b;
        break;
    case D3DRS_EMISSIVEMATERIALSOURCE:
        rs.matSrcEmissive = (BYTE)b;
        break;
    }
}

void captureFragmentRenderState(DWORD a, D3DTEXTURESTAGESTATETYPE b, DWORD c) {
    FragmentState::Stage* s = &frs.stage[a];

    switch (b) {
    case D3DTSS_COLOROP:
        s->colorOp = (BYTE)c;
        break;
    case D3DTSS_COLORARG1:
        s->colorArg1 = (BYTE)c;
        break;
    case D3DTSS_COLORARG2:
        s->colorArg2 = (BYTE)c;
        break;
    case D3DTSS_ALPHAOP:
        s->alphaOp = (BYTE)c;
        break;
    case D3DTSS_ALPHAARG1:
        s->alphaArg1 = (BYTE)c;
        break;
    case D3DTSS_ALPHAARG2:
        s->alphaArg2 = (BYTE)c;
        break;
    case D3DTSS_BUMPENVMAT00:
        s->bumpEnvMat[0][0] = reinterpret_cast<float&>(c);
        break;
    case D3DTSS_BUMPENVMAT01:
        s->bumpEnvMat[0][1] = reinterpret_cast<float&>(c);
        break;
    case D3DTSS_BUMPENVMAT10:
        s->bumpEnvMat[1][0] = reinterpret_cast<float&>(c);
        break;
    case D3DTSS_BUMPENVMAT11:
        s->bumpEnvMat[1][1] = reinterpret_cast<float&>(c);
        break;
    case D3DTSS_TEXCOORDINDEX:
        s->texcoordIndex = c;
        break;
    case D3DTSS_BUMPENVLSCALE:
        s->bumpLumiScale = reinterpret_cast<float&>(c);
        break;
    case D3DTSS_BUMPENVLOFFSET:
        s->bumpLumiBias = reinterpret_cast<float&>(c);
        break;
    case D3DTSS_TEXTURETRANSFORMFLAGS:
        s->texTransformFlags = c;
        break;
    case D3DTSS_COLORARG0:
        s->colorArg0 = (BYTE)c;
        break;
    case D3DTSS_ALPHAARG0:
        s->alphaArg0 = (BYTE)c;
        break;
    case D3DTSS_RESULTARG:
        s->resultArg = (BYTE)c;
        break;
    }
}

void captureTransform(D3DTRANSFORMSTATETYPE a, const D3DMATRIX* b) {
    switch (a) {
    case D3DTS_WORLDMATRIX(0):
        rs.worldTransforms[0] = *b;
        D3DXMatrixMultiply(&rs.worldViewTransforms[0], static_cast<const D3DXMATRIX*>(b), &rs.viewTransform);
        break;
    case D3DTS_WORLDMATRIX(1):
        rs.worldTransforms[1] = *b;
        D3DXMatrixMultiply(&rs.worldViewTransforms[1], static_cast<const D3DXMATRIX*>(b), &rs.viewTransform);
        break;
    case D3DTS_WORLDMATRIX(2):
        rs.worldTransforms[2] = *b;
        D3DXMatrixMultiply(&rs.worldViewTransforms[2], static_cast<const D3DXMATRIX*>(b), &rs.viewTransform);
        break;
    case D3DTS_WORLDMATRIX(3):
        rs.worldTransforms[3] = *b;
        D3DXMatrixMultiply(&rs.worldViewTransforms[3], static_cast<const D3DXMATRIX*>(b), &rs.viewTransform);
        break;
    case D3DTS_VIEW:
        rs.viewTransform = *b;
        lightrs.lightsTransformed.clear();
        break;
    }
}

void captureLight(DWORD a, const D3DLIGHT8* b) {
    // Morrowind uses non-contigous light IDs up to a large number (>512)
    LightState::Light* light = &lightrs.lights[a];

    // Copy values relevant to Morrowind
    // i.e. Morrowind has no spotlights and always sets range to FLT_MAX
    // The only light source with ambient is sunlight
    light->type = b->Type;
    light->diffuse = b->Diffuse;

    if (b->Type == D3DLIGHT_POINT) {
        light->position = b->Position;
        light->falloff.x = b->Attenuation0;
        light->falloff.y = b->Attenuation1;
        light->falloff.z = b->Attenuation2;
    } else {
        D3DXVec3Normalize((D3DXVECTOR3*)&light->position, (D3DXVECTOR3*)&b->Direction);
        light->ambient.x = b->Ambient.r;
        light->ambient.y = b->Ambient.g;
        light->ambient.z = b->Ambient.b;
    }
}

void captureMaterial(const D3DMATERIAL8* a) {
    // Morrowind does not use specular lighting
    rs.diffuseMaterial = a->Diffuse;
    frs.material.diffuse = a->Diffuse;
    frs.material.ambient = a->Ambient;
    frs.material.emissive = a->Emissive;
    frs.material.emissive.a = a->Power;
}

// --------------------------------------------------------
// FPS meter - Updates every 500ms. Morrowind's internal meter changes too fast and falsely clamps the fps.

float calcFPS() {
    static int lastMillis, framesSinceUpdate;
    static float fps = 0.0f;

    ++framesSinceUpdate;
    int millis = MWBridge::get()->getFrameBeginMillis();
    int diff = millis - lastMillis;

    if (diff >= 500) {
        fps = 1000.0f * framesSinceUpdate / diff;
        lastMillis = millis;
        framesSinceUpdate = 0;
    } else if (diff < 0) {
        lastMillis = millis;
    }

    return fps;
}
