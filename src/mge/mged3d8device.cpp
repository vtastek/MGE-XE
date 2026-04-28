
#include "mged3d8device.h"
#include "mgedevicehelpers.h"
#include "mge_tracy.h"
#include "proxydx/d3d8texture.h"
#include "proxydx/d3d8surface.h"
#include "support/log.h"

#include <algorithm>
#include <chrono>
#include <tlhelp32.h>
#include "mgeversion.h"
#include "configuration.h"
#include "distantland.h"
#include "mwbridge.h"
#include "statusoverlay.h"
#include "userhud.h"
#include "videobackground.h"
#include "d3dcommandbuffer.h"
#include "cpuprepthread.h"
#include "renderthread.h"

bool g_tracyActive = false;
PipelineDiag g_pipelineDiag = {};

// Global frame counter for debug logging (defined in ffeshader.cpp)
extern int g_diagFrameCounter;

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

// Grouped state structs (see mgedevicehelpers.h)
SceneState g_scene;  // Non-static for external access (declared in mged3d8device.h)
static DeferredSceneState g_deferred;
static DeferralCounters g_deferralCounters;
static OffscreenState g_offscreen;
static DLContext frameCtx;  // Per-frame rendering context

// Camera effects state
static bool zoomSensSaved;
static float zoomSensX, zoomSensY;
static D3DXMATRIX camEffectsMatrix;
static float crosshairTimeout;

// MW state recorders
static RenderedState rs;
static FragmentState frs;
static LightState lightrs;
uint32_t g_lightStateGen = 1;  // Generation counter - incremented only when light data actually changes
DeviceStateSnapshot g_deviceState;

// ImGui state
static HWND gameWindow = nullptr;
static bool imguiInitialized = false;

// Command buffer
D3DCommandBufferSet g_cmdBufferSet;
static int g_lastCmdBufferSize = 0;

// DIP counters
static ImGuiManager::DIPBinStats g_dipBinStats = {};
static int g_dipScene0 = 0, g_dipScene1 = 0, g_dipScene2 = 0;

// Async device ownership: render thread sets this when it owns the device.
// Main thread checks this to detect illegal device access during async overlap.
std::atomic<bool> g_renderThreadOwnsDevice{false};
static std::atomic<int> g_deviceRaceCount{0};  // Count races detected this frame

// Log a device access race (main thread touching device while render thread owns it)
static void logDeviceRace(const char* caller) {
    int count = g_deviceRaceCount.fetch_add(1);
    if (count < 20) {  // Cap to avoid log spam
        LOG::logline("[RACE] Main thread device call during RT: %s (scene=%d)", caller, g_scene.sceneCount);
    }
}

// Safe zone tracking for async GPU thread planning
// Safe zone = time window where render thread can work without touching main thread's device
// Normal frames: Present() → UI BeginScene
// Offscreen frames: Offscreen End → UI BeginScene (stall during offscreen)
static bool g_renderThreadSafeZone = false;
static bool g_safeZoneEndedForOffscreen = false;  // Track if we need to restart after offscreen
static std::chrono::high_resolution_clock::time_point g_safeZoneStart;
static float g_lastSafeZoneMs = 0.0f;

// Per-scene draw characteristic tracking (to understand what each scene contains)
struct SceneDrawStats {
    int total = 0;
    int skinned = 0;      // vertexBlendState != 0
    int blended = 0;      // blendEnable
    int zWriting = 0;     // zWrite
    int alphaTested = 0;  // alphaTest
    void reset() { total = skinned = blended = zWriting = alphaTested = 0; }
};
static SceneDrawStats g_sceneStats[4];  // Scene 0, 1, 2, 3+
static bool g_sceneStatsLogged = false;

// Frame counter for display and correlation with logs
static int g_frameNumber = 0;

int getFrameNumber() { return g_frameNumber; }

static void initOnLoad();
static bool detectMenu(const D3DMATRIX* m);
static ScenePhase detectScenePhase();
static void captureRenderState(D3DRENDERSTATETYPE a, DWORD b);
static void captureFragmentRenderState(DWORD a, D3DTEXTURESTAGESTATETYPE b, DWORD c);
static void captureTransform(D3DTRANSFORMSTATETYPE a, const D3DMATRIX* b);
static void captureLight(DWORD a, const D3DLIGHT8* b);
static void captureMaterial(const D3DMATERIAL8* a);
static float calcFPS();
static void capturePostProcessData(FixedFunctionShader::FrameBuffer& fb, const DLContext& ctx, const char* source);

static void capturePostProcessData(FixedFunctionShader::FrameBuffer& fb, const DLContext& ctx, const char* source) {
    auto mwb = MWBridge::get();
    auto& ppd = fb.postProcessData;
    ppd.frameTime = mwb->frameTime();
    ppd.simulationTime = mwb->simulationTime();
    ppd.waterLevel = mwb->CellHasWater() ? mwb->WaterLevel() : -1e9f;
    ppd.isMenu = mwb->IsMenu();
    ppd.isInterior = !mwb->CellHasWeather();
    ppd.isUnderwater = mwb->IsUnderwater(ctx.eyePos.z);

    int envFlags = 0;
    if (!mwb->CellHasWeather()) envFlags |= 1;
    if (mwb->IsExterior()) envFlags |= 2;
    if (mwb->IntLikeExterior()) envFlags |= 4;
    if (ppd.isUnderwater) envFlags |= 8; else envFlags |= 16;
    if (ctx.sunVis >= 0.001f) envFlags |= 32; else envFlags |= 64;
    ppd.envFlags = envFlags;

    if (LOG::catEnabled(LOG::Cat_DistantLand)) {
        static int logCount = 0;
        if (logCount++ < 12) {
            LOG::logline(
                "[PPDCAP][%s] cell=0x%08X water=%.2f env=0x%02X interior=%d underwater=%d",
                source ? source : "?",
                mwb->IntCurCellAddr(),
                ppd.waterLevel,
                ppd.envFlags,
                ppd.isInterior ? 1 : 0,
                ppd.isUnderwater ? 1 : 0);
        }
    }
}

// Helper: process debug hotkeys when ImGui is initialized
static void processImGuiHotkeys() {
    static bool f11Pressed = false, gPressed = false, uPressed = false, ePressed = false, lPressed = false;

    // F11: Toggle PCF interface (gated)
    if (ImGuiManager::GetDebugKeysEnabled()) {
        bool f11State = (GetAsyncKeyState(VK_F11) & 0x8000) != 0;
        if (f11State && !f11Pressed) {
            LOG::logline(">> F11 key pressed, toggling PCF interface");
            ImGuiManager::TogglePCFInterface();
        }
        f11Pressed = f11State;
    }

    // G key: Toggle debug interface (gated by hidden mge.ini option, default off)
    if (Configuration.EnableDebugInterfaceKey) {
        bool gState = (GetAsyncKeyState('G') & 0x8000) != 0;
        if (gState && !gPressed) {
            ImGuiManager::ToggleDebugInterface();
        }
        gPressed = gState;
    } else {
        gPressed = false;
    }

    // U: Toggle Hi-Z interface (gated)
    if (ImGuiManager::GetDebugKeysEnabled()) {
        bool uState = (GetAsyncKeyState('U') & 0x8000) != 0;
        if (uState && !uPressed) {
            ImGuiManager::ToggleHiZInterface();
        }
        uPressed = uState;
    }

    // E: Toggle Frame Event Log (gated)
    if (ImGuiManager::GetDebugKeysEnabled()) {
        bool eState = (GetAsyncKeyState('E') & 0x8000) != 0;
        if (eState && !ePressed) {
            ImGuiManager::ToggleFrameEventLog();
        }
        ePressed = eState;
    }

    // L: Dump current HLSL light snapshot (gated)
    if (ImGuiManager::GetDebugKeysEnabled()) {
        bool lState = (GetAsyncKeyState('L') & 0x8000) != 0;
        if (lState && !lPressed) {
            LOG::logline(">> L key pressed, requesting light snapshot dump");
            ImGuiManager::RequestLightSnapshotDump();
        }
        lPressed = lState;
    }
}

// When true, MW state/draw calls record to command buffer only — no device forwarding.
// Keeps main thread off the device for threading safety.
static inline bool shouldSuppressMWState() {
    if (!ImGuiManager::GetStateSuppressionEnabled()) {
        return false;
    }
    // TRUE ASYNC: Suppress ALL device calls when GPU thread owns device
    // This covers scene -1 setup, scene 0/1/2 recording, everything
    if (ImGuiManager::GetAsyncGpuThread() && g_renderThreadOwnsDevice.load(std::memory_order_acquire)) {
        return true;
    }
    // SYNC: Suppress during Scene 0, 1, 2 HLSL recording only
    return isHLSLActive()
        && FixedFunctionShader::getIsRecording()
        && g_scene.sceneCount <= 2;
}

// Check for device access race: main thread touching device while render thread owns it
#define CHECK_DEVICE_RACE(name) \
    do { if (g_renderThreadOwnsDevice.load(std::memory_order_acquire)) logDeviceRace(name); } while(0)



// Snapshot key device render states and record them as preamble to UI command buffer.
// After state block restore + z-clear, MW expects certain inherited states for UI rendering.
// When the UI buffer is replayed later, these states make it self-contained.
static void recordUIStatePreamble(IDirect3DDevice9* dev, D3DCommandBuffer& buf) {
    DWORD val;

    // Blend state
    dev->GetRenderState(D3DRS_ALPHABLENDENABLE, &val); buf.recordSetRenderState(D3DRS_ALPHABLENDENABLE, val);
    dev->GetRenderState(D3DRS_SRCBLEND, &val);         buf.recordSetRenderState(D3DRS_SRCBLEND, val);
    dev->GetRenderState(D3DRS_DESTBLEND, &val);        buf.recordSetRenderState(D3DRS_DESTBLEND, val);

    // Depth state
    dev->GetRenderState(D3DRS_ZENABLE, &val);          buf.recordSetRenderState(D3DRS_ZENABLE, val);
    dev->GetRenderState(D3DRS_ZWRITEENABLE, &val);     buf.recordSetRenderState(D3DRS_ZWRITEENABLE, val);
    dev->GetRenderState(D3DRS_ZFUNC, &val);            buf.recordSetRenderState(D3DRS_ZFUNC, val);

    // Lighting / fog / cull
    dev->GetRenderState(D3DRS_LIGHTING, &val);         buf.recordSetRenderState(D3DRS_LIGHTING, val);
    dev->GetRenderState(D3DRS_FOGENABLE, &val);        buf.recordSetRenderState(D3DRS_FOGENABLE, val);
    dev->GetRenderState(D3DRS_CULLMODE, &val);         buf.recordSetRenderState(D3DRS_CULLMODE, val);

    // Material source
    dev->GetRenderState(D3DRS_COLORVERTEX, &val);              buf.recordSetRenderState(D3DRS_COLORVERTEX, val);
    dev->GetRenderState(D3DRS_AMBIENTMATERIALSOURCE, &val);    buf.recordSetRenderState(D3DRS_AMBIENTMATERIALSOURCE, val);
    dev->GetRenderState(D3DRS_DIFFUSEMATERIALSOURCE, &val);    buf.recordSetRenderState(D3DRS_DIFFUSEMATERIALSOURCE, val);

    // Alpha test
    dev->GetRenderState(D3DRS_ALPHATESTENABLE, &val);  buf.recordSetRenderState(D3DRS_ALPHATESTENABLE, val);
    dev->GetRenderState(D3DRS_ALPHAREF, &val);         buf.recordSetRenderState(D3DRS_ALPHAREF, val);
    dev->GetRenderState(D3DRS_ALPHAFUNC, &val);        buf.recordSetRenderState(D3DRS_ALPHAFUNC, val);

    // Stencil
    dev->GetRenderState(D3DRS_STENCILENABLE, &val);    buf.recordSetRenderState(D3DRS_STENCILENABLE, val);

    // Ambient / texture factor
    dev->GetRenderState(D3DRS_AMBIENT, &val);          buf.recordSetRenderState(D3DRS_AMBIENT, val);
    dev->GetRenderState(D3DRS_TEXTUREFACTOR, &val);    buf.recordSetRenderState(D3DRS_TEXTUREFACTOR, val);

    // Texture stage state for stages 0-1
    for (DWORD stage = 0; stage < 2; stage++) {
        dev->GetTextureStageState(stage, D3DTSS_COLOROP, &val);   buf.recordSetTextureStageState(stage, D3DTSS_COLOROP, val);
        dev->GetTextureStageState(stage, D3DTSS_COLORARG1, &val); buf.recordSetTextureStageState(stage, D3DTSS_COLORARG1, val);
        dev->GetTextureStageState(stage, D3DTSS_COLORARG2, &val); buf.recordSetTextureStageState(stage, D3DTSS_COLORARG2, val);
        dev->GetTextureStageState(stage, D3DTSS_ALPHAOP, &val);   buf.recordSetTextureStageState(stage, D3DTSS_ALPHAOP, val);
        dev->GetTextureStageState(stage, D3DTSS_ALPHAARG1, &val); buf.recordSetTextureStageState(stage, D3DTSS_ALPHAARG1, val);
        dev->GetTextureStageState(stage, D3DTSS_ALPHAARG2, &val); buf.recordSetTextureStageState(stage, D3DTSS_ALPHAARG2, val);
    }

    // Fixed-function pipeline (UI doesn't use shaders)
    buf.recordSetVertexShader(nullptr);
    buf.recordSetPixelShader(nullptr);

    // FVF — MW doesn't set FVF before first UI DIP, inherits from scene rendering
    dev->GetFVF(&val);                          buf.recordSetFVF(val);

    // Viewport — MW sets it before BeginScene (goes to Scene 1/2 buffer, not UI)
    D3DVIEWPORT9 vp;
    dev->GetViewport(&vp);                      buf.recordSetViewport(&vp);

    // Transforms — MW sets VIEW/PROJ before BeginScene (goes to Scene 1/2 buffer, not UI)
    D3DMATRIX mat;
    dev->GetTransform(D3DTS_VIEW, &mat);        buf.recordSetTransform(D3DTS_VIEW, &mat);
    dev->GetTransform(D3DTS_PROJECTION, &mat);   buf.recordSetTransform(D3DTS_PROJECTION, &mat);

    // Record z-clear into the UI buffer so it's self-contained
    buf.recordClear(0, D3DCLEAR_ZBUFFER, 0, 1.0f, 0);
}

MGEProxyDevice::MGEProxyDevice(IDirect3DDevice9* real, ProxyD3D* d3d) : ProxyDevice(real, d3d) {
    // Initialize state here, as the device is released and recreated on fullscreen Alt-Tab
    g_scene = SceneState();
    g_deferred = DeferredSceneState();
    g_deferralCounters = DeferralCounters();
    g_offscreen = OffscreenState();
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

    // Start shader precaching immediately when device is available (before save game load)
    if (Configuration.MGEFlags & USE_FFESHADER) {
        LOG::logline("-- Starting immediate shader precaching (device constructor)");
        FixedFunctionShader::startEarlyPrecache(realDevice);
    }

    // Patch splash screen minor issues
    D3DVIEWPORT9 vp;
    realDevice->GetViewport(&vp);
    MWBridge::get()->patchSplashScreen(vp.Width, vp.Height);
}

// Present - End of MW frame
// MGE end of frame processing
// bisect test comment
HRESULT _stdcall MGEProxyDevice::Present(const RECT* a, const RECT* b, HWND c, const RGNDATA* d) {
    MGE_ZoneScopedN("MGE_Present");
    ++g_frameNumber;
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
        // Detect when Morrowind starts its data loading phase (first loading bar before MGE init)
        if (!DistantLand::ready && mwBridge->isLoadingBar()) {
            static bool dataLoadStartLogged = false;
            if (!dataLoadStartLogged) {
                LOG::logline("== Initializing data: START ==");
                dataLoadStartLogged = true;
            }
        }

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

    // Initialize ImGui on first Present call
    if (!imguiInitialized) {
        // Get window handle from device creation parameters since Present HWND is null
        D3DDEVICE_CREATION_PARAMETERS creationParams;
        if (SUCCEEDED(realDevice->GetCreationParameters(&creationParams))) {
            gameWindow = creationParams.hFocusWindow;
            LOG::logline(">> Got HWND from device creation params: %p", gameWindow);
            if (gameWindow != nullptr) {
                LOG::logline(">> Attempting to initialize ImGui with HWND: %p, device: %p", gameWindow, realDevice);
                if (ImGuiManager::Initialize(gameWindow, realDevice)) {
                    imguiInitialized = true;
                    LOG::logline(">> ImGui integration initialized successfully");
                } else {
                    LOG::logline(">> ImGui initialization failed");
                }
            } else {
                LOG::logline(">> Device creation HWND is null");
            }
        } else {
            LOG::logline(">> Failed to get device creation parameters");
        }
    }

    // Per-stage command buffer replay is now handled in executeGpuPhase (ffeshader.cpp)
    // and UI replay is handled in BeginScene (UI handler).
    // Dump per-stage logs for trace analysis.
    if (ImGuiManager::GetCmdBufferReplay()) {
        g_cmdBufferSet.dumpToFrameLog();
    }

    {
        MGE_ZoneScopedN("Present_ImGui");
        static int debugCounter = 0;
        if (imguiInitialized) {
            processImGuiHotkeys();
            ImGuiManager::NewFrame();
            ImGuiManager::Render();
        } else if (++debugCounter % 60 == 0) {
            LOG::logline(">> ImGui not initialized (frame %d)", debugCounter);
        }
    }

    // Increment global debug frame counter (works for both HLSL and PPL modes)
    ++g_diagFrameCounter;

    // Log render pass break instrumentation
    {
        static int frameCounter = 0;
        if (LOG::catEnabled(LOG::Cat_FrameStats) && (++frameCounter <= 60 || frameCounter % 300 == 0)) {
            LOG::logline("DXVK passes: raw[RT=%d DS=%d Clear=%d Stretch=%d] cat[MW_Clear=%d depth=%d shadow=%d water=%d post=%d stretch=%d other=%d =%d]",
                g_passBreaks.raw_setRT, g_passBreaks.raw_setDS,
                g_passBreaks.raw_clear, g_passBreaks.raw_stretchRect,
                g_passBreaks.mw_clear,
                g_passBreaks.mge_depthRT, g_passBreaks.mge_shadowRT,
                g_passBreaks.mge_waterRT, g_passBreaks.mge_postRT,
                g_passBreaks.mge_stretchRect, g_passBreaks.mge_otherRT,
                g_passBreaks.categorized());
        }
        g_passBreaks.reset();
    }

    // Log DIP category counters (only when there were actual DIPs)
    {
        int dipTotal = g_dipBinStats.total();
        if (dipTotal > 0) {
            static int dipLogCounter = 0;
            bool isSpike = dipTotal >= ImGuiManager::GetDIPSpikeThreshold();
            // Both spike alerts and periodic stats gated by FrameStats — local-map
            // frames trip the spike threshold every time and aren't actionable.
            bool wantLog = LOG::catEnabled(LOG::Cat_FrameStats)
                           && (isSpike || ++dipLogCounter <= 60 || dipLogCounter % 300 == 0);
            if (wantLog) {
                LOG::logline("%sDIPs: Sky=%d Ter=%d Opq=%d Skin=%d Grass=%d AT=%d Blend=%d 1PS=%d 1PA=%d 1PO=%d Wat=%d Off=%d UI=%d Sten=%d Pre=%d T=%d",
                    isSpike ? "DIP SPIKE! " : "",
                    g_dipBinStats.sky, g_dipBinStats.terrain, g_dipBinStats.opaque,
                    g_dipBinStats.skinning, g_dipBinStats.grass, g_dipBinStats.alphaTested,
                    g_dipBinStats.blending, g_dipBinStats.firstPersonSkinning,
                    g_dipBinStats.firstPersonAlpha, g_dipBinStats.firstPersonOther,
                    g_dipBinStats.water, g_dipBinStats.offscreen, g_dipBinStats.ui,
                    g_dipBinStats.stencilShadow, g_dipBinStats.preScene, dipTotal);
            }
            // Auto-freeze DIP stats on spike
            if (isSpike && ImGuiManager::GetDIPAutoFreeze() && !ImGuiManager::GetDIPFrozen()) {
                ImGuiManager::FreezeDIPStats(g_dipBinStats);
            }
        }
        ImGuiManager::UpdateDIPStats(g_dipBinStats);
        g_dipBinStats.reset();
        g_dipScene0 = 0;
        g_dipScene1 = 0;
        g_dipScene2 = 0;
    }

    // Snapshot frame events for ImGui display
    ImGuiManager::SnapshotFrameEvents();

    // Snapshot command buffer stats for ImGui, then clear for next frame
    if (ImGuiManager::GetCmdBufferRecording()) {
        g_lastCmdBufferSize = g_cmdBufferSet.totalSize();
        ImGuiManager::UpdateCmdBufferStats(g_lastCmdBufferSize, (int)(g_cmdBufferSet.totalSizeBytes() / 1024));
        ImGuiManager::UpdateCmdBufferPerStageStats(
            g_cmdBufferSet[CmdStage::PreScene].size(),
            g_cmdBufferSet[CmdStage::Scene0].size(),
            g_cmdBufferSet[CmdStage::InterScene].size(),
            g_cmdBufferSet[CmdStage::Scene1].size() + g_cmdBufferSet[CmdStage::Scene2].size(),
            g_cmdBufferSet[CmdStage::UI].size());
        g_cmdBufferSet.clearAll();
    }

    // Fill pipeline diagnostic snapshot (end-of-frame state, before reset)
    {
        auto mwb = MWBridge::get();
        g_pipelineDiag.dipScene0 = g_dipScene0;
        g_pipelineDiag.dipScene1 = g_dipScene1 + g_dipScene2;
        g_pipelineDiag.dipOffscreen = g_dipBinStats.offscreen;
        g_pipelineDiag.dipUI = g_dipBinStats.ui;
        g_pipelineDiag.dipStencilShadow = g_dipBinStats.stencilShadow;
        g_pipelineDiag.dipUnknown = g_dipBinStats.preScene;
        g_pipelineDiag.sceneCount = g_scene.sceneCount;
        g_pipelineDiag.isMainView = g_scene.isMainView;
        g_pipelineDiag.rendertargetNormal = g_scene.rendertargetNormal;
        g_pipelineDiag.stage0Complete = g_scene.stage0Complete;
        g_pipelineDiag.isFrameComplete = g_scene.isFrameComplete;
        g_pipelineDiag.isHUDComplete = g_scene.isHUDComplete;
        g_pipelineDiag.isHUDready = g_scene.isHUDready;
        g_pipelineDiag.isStencilScene = g_scene.isStencilScene;
        g_pipelineDiag.isAmbientWhite = g_scene.isAmbientWhite;
        g_pipelineDiag.distantLandReady = DistantLand::ready;
        g_pipelineDiag.isPPLActive = DistantLand::s_staging.isPPLActive;
        g_pipelineDiag.view_11 = rs.viewTransform._11;
        g_pipelineDiag.view_12 = rs.viewTransform._12;
        g_pipelineDiag.view_13 = rs.viewTransform._13;
        g_pipelineDiag.view_41 = rs.viewTransform._41;
        g_pipelineDiag.view_42 = rs.viewTransform._42;
        g_pipelineDiag.view_43 = rs.viewTransform._43;
        g_pipelineDiag.mwLoaded = mwb->IsLoaded();
        g_pipelineDiag.mwCellAddr = mwb->IsLoaded() ? mwb->IntCurCellAddr() : 0;
    }

    // F5/F6 pipeline snapshot hotkeys — fires every frame from Present()
    FixedFunctionShader::checkSnapshotHotkeys();

    {
        MGE_ZoneScopedN("Present_ResetState");
        // Reset scene identifiers
        g_scene.resetForFrame();
        g_deferred.reset();

        // Log deferral stats on frames with offscreen scenes (FrameStats category).
        if ((g_offscreen.scenesThisFrame > 0 || g_deferralCounters.copyRectsTotal > 0)
            && LOG::catEnabled(LOG::Cat_FrameStats)) {
            LOG::logline("Deferral: scenes %d/%d/%d RT %d/%d/%d offscreen %d (mega-scene) CopyRects %d/%d",
                g_deferralCounters.scenesRequested, g_deferralCounters.scenesForwarded, g_deferralCounters.scenesSuppressed,
                g_deferralCounters.rtRequested, g_deferralCounters.rtForwarded, g_deferralCounters.rtSuppressed,
                g_offscreen.scenesThisFrame,
                g_deferralCounters.copyRectsTotal, g_deferralCounters.copyRectsSuppressed);
        }
        g_deferralCounters.reset();
        g_offscreen.resetForFrame();
        g_deviceRaceCount.store(0);
        // Safety: close mega-scene if still open at Present
        if (g_offscreen.megaSceneOpen) {
            if (ImGuiManager::GetCmdBufferRecording() && !ImGuiManager::GetCmdBufferReplay()) {
                g_cmdBufferSet.active().recordEndScene();
            }
            ProxyDevice::EndScene();
            g_offscreen.megaSceneOpen = false;
        }

        // Reset stage for next frame — Offscreen because MW always starts with offscreen
        // rendering (local map tiles) before the main scene. Events before the first
        // BeginScene (SetRenderTarget, Viewport, Clear, Transforms) must go to the
        // Offscreen buffer, not PreScene.
        g_cmdBufferSet.activeStage = CmdStage::Offscreen;

        // Snapshot tracker state into recording buffer BEFORE swap.
        // After swap, this buffer becomes the rendering buffer — render thread reads it.
        // Must happen before swap clears the tracker for the new frame.
        if (isHLSLActive()) {
            FixedFunctionShader::getRecordingBuffer().trackerSnapshot = g_cmdBufferSet.stateTracker();
        }

        // N-1: Swap buffer indices and clear new recording buffer
        // Recording buffer (frame N) becomes rendering buffer (will be rendered as N-1 next frame)
        // Rendering buffer (just rendered) becomes recording buffer (cleared for frame N+1)
        FixedFunctionShader::swapBuffers();

        // Clear tracker for new frame's recording
        g_cmdBufferSet.stateTracker().clear();

        // Start safe zone for async GPU thread
        // Normal frames: Present() → UI BeginScene
        // Offscreen frames: stall during offscreen, then Offscreen End → UI BeginScene
        // During safe zone, main thread only records to FrameBuffer (CPU work)
        // Render thread can safely submit GPU calls from N-1 data
        g_renderThreadSafeZone = true;
        g_safeZoneEndedForOffscreen = false;  // Reset for new frame
        g_safeZoneStart = std::chrono::high_resolution_clock::now();
        MGE_TracyMessage("RT_SafeZone_START", 17);

        // Phase 3: True Async Overlap - submit both CPU prep and GPU work at Present().
        // Main thread records frame N while:
        //   - CPU prep thread processes frame N-1
        //   - GPU thread renders frame N-2
        if (ImGuiManager::GetAsyncGpuThread() && isHLSLActive() && g_cpuPrepThread && g_cpuPrepThread->isRunning()) {
            // Wait for previous CPU prep to complete before starting GPU work
            // This ensures render buffer (N-2) is in ReadyToRender state
            {
                MGE_ZoneScopedN("WaitPrevCpuPrep");
                g_cpuPrepThread->waitForCompletion();
            }
            // Submit GPU work async - N-2 buffer now ready, runs in parallel with recording
            if (g_renderThread && g_renderThread->isRunning() && FixedFunctionShader::isN2Ready()) {
                RenderThread::SceneWork gpuWork;
                gpuWork.type = RenderThread::WorkType::RenderFullFrame;
                g_renderThread->submitWork(std::move(gpuWork), false);  // waitNow=false — async!
                MGE_TracyMessage("RT_GpuSubmit", 12);
            }
            // Submit CPU prep for current frame (N-1) async
            if (FixedFunctionShader::isN1Ready()) {
                auto& prepBuf = FixedFunctionShader::getPrepBuffer();
                CpuPrepThread::PrepWork work;
                work.type = CpuPrepThread::WorkType::PrepareFrame;
                work.viewMatrix = prepBuf.view;
                work.projMatrix = prepBuf.proj;
                g_cpuPrepThread->submitWork(std::move(work), false);  // waitNow=false — async!
                MGE_TracyMessage("CPT_PrepSubmit", 14);
            }
        }

        // Reset per-frame flags
        FixedFunctionShader::resetHiZBuiltFlag();
        FixedFunctionShader::resetRecordingCompletedFlag();

        // Reset HLSL texture caches at frame boundary to prevent stale texture pointers
        // Skip when async GPU is active - GPU thread handles its own cache reset and may still be using resources
        if (!ImGuiManager::GetAsyncGpuThread()) {
            FixedFunctionShader::resetHLSLCaches();
        }
    }

    MGE_FrameMark;  // Mark frame boundary at the very end of Present()
    MGE_TracyPlot("Frame", (int64_t)g_frameNumber);
    {
        MGE_ZoneScopedN("Present_ProxyDevicePresent");
        return ProxyDevice::Present(a, b, c, d);
    }
}

// Forward pending render target change to real device
static HRESULT flushPendingRT() {
    if (g_deferred.rtPending) {
        g_deferred.rtPending = false;
        g_deferralCounters.rtForwarded++;
        auto device = DistantLand::device;
        HRESULT hr1 = D3D_OK, hr2 = D3D_OK;
        IDirect3DSurface9* rtSurface = g_deferred.pendingRT_color ? static_cast<ProxySurface*>(g_deferred.pendingRT_color)->realSurface : nullptr;
        IDirect3DSurface9* dsSurface = g_deferred.pendingRT_depth ? static_cast<ProxySurface*>(g_deferred.pendingRT_depth)->realSurface : nullptr;

        // Record RT/DS changes to command buffer
        if (ImGuiManager::GetCmdBufferRecording()) {
            if (rtSurface) g_cmdBufferSet.active().recordSetRenderTarget(rtSurface);
            g_cmdBufferSet.active().recordSetDepthStencilSurface(dsSurface);
        }

        // Forward to device — except during Offscreen replay (device calls deferred to buffer)
        if (!shouldSuppressMWState()) {
            CHECK_DEVICE_RACE("flushPendingRT");
            if (rtSurface) {
                hr1 = device->SetRenderTarget(0, rtSurface);
            }
            hr2 = device->SetDepthStencilSurface(dsSurface);
        }

        return (hr1 != D3D_OK) ? hr1 : hr2;
    }
    return D3D_OK;
}

// SetRenderTarget
// Remember if MW is rendering to back buffer, defer forwarding
HRESULT _stdcall MGEProxyDevice::SetRenderTarget(IDirect3DSurface8* a, IDirect3DSurface8* b) {
    if (a) {
        IDirect3DSurface9* back;
        realDevice->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &back);
        g_scene.rendertargetNormal = (static_cast<ProxySurface*>(a)->realSurface == back);
        back->Release();
    }

    g_passBreaks.raw_setRT++;
    g_passBreaks.raw_setDS++;
    ImGuiManager::LogFrameEvent(FrameEvent::SetRenderTarget, g_scene.sceneCount);
    ImGuiManager::TraceRT(g_scene.sceneCount, a, b);

    // Defer RT change — forward only when a Clear or Draw needs it
    // If previous pending RT was never forwarded, count it as suppressed
    if (g_deferred.rtPending) {
        g_deferralCounters.rtSuppressed++;
    }
    g_deferralCounters.rtRequested++;
    g_deferred.pendingRT_color = a;
    g_deferred.pendingRT_depth = b;
    g_deferred.rtPending = true;
    return D3D_OK;
}

// Forward deferred RT + BeginScene to real device (call before any draw)
static HRESULT ensureSceneActive() {
    if (shouldSuppressMWState()) return D3D_OK;
    CHECK_DEVICE_RACE("ensureSceneActive");
    flushPendingRT();
    if (g_deferred.scenePending && !g_deferred.sceneForwarded) {
        HRESULT hr = DistantLand::device->BeginScene();
        if (hr != D3D_OK) {
            return hr;
        }
        if (ImGuiManager::GetCmdBufferRecording()) {
            // Don't record BeginScene to UI buffer — UI replay happens within the existing scene
            if (g_cmdBufferSet.activeStage != CmdStage::UI) {
                g_cmdBufferSet.active().recordBeginScene();
            }
        }
        g_deferred.sceneForwarded = true;
        g_deferred.scenePending = false;
    }
    return D3D_OK;
}

// BeginScene - Multiple scenes per frame, non-alpha / 2x stencil / post-stencil redraw / alpha / 1st person / UI
// Fogging needs to be set for Morrowind rendering at start of scene
HRESULT _stdcall MGEProxyDevice::BeginScene() {
    auto mwBridge = MWBridge::get();

    // Defer real device BeginScene until a draw call needs it
    g_deferred.scenePending = true;
    g_deferred.sceneForwarded = false;
    g_deferralCounters.scenesRequested++;

    // Offscreen scene collapsing: keep one device-level scene open for all offscreen work
    // This eliminates per-scene DXVK command buffer submissions
    if (!g_scene.rendertargetNormal) {
        // End safe zone: offscreen rendering needs direct device access
        // Will restart safe zone after offscreen completes (at mega scene close)
        if (g_renderThreadSafeZone) {
            auto elapsed = std::chrono::high_resolution_clock::now() - g_safeZoneStart;
            g_lastSafeZoneMs = std::chrono::duration<float, std::milli>(elapsed).count();
            MGE_TracyPlot("RT_SafeZone_ms", g_lastSafeZoneMs);
            MGE_TracyMessage("RT_SafeZone_STALL_Offscreen", 26);
            g_renderThreadSafeZone = false;
            g_safeZoneEndedForOffscreen = true;
        }

        MGE_ZoneScopedN("OffscreenRender");
        // SYNC: Wait for async threads before off-screen rendering touches device.
        // Off-screen work (local map, inventory doll) needs direct device access.
        if (isHLSLActive()) {
            if (g_cpuPrepThread && g_cpuPrepThread->isRunning())
                g_cpuPrepThread->waitForCompletion();
            if (g_renderThread && g_renderThread->isRunning())
                g_renderThread->waitForCompletion();
        }

        g_cmdBufferSet.activeStage = CmdStage::Offscreen;
        g_offscreen.suppressingCurrentScene = false;
        g_offscreen.startTiming();
        g_offscreen.sceneCount++;
        if (!g_offscreen.megaSceneOpen) {
            // First offscreen scene: open device scene, keep it open for all subsequent offscreen work
            FixedFunctionShader::transitionTo(PhaseTransition::OffscreenEntry);
            // Save blend state before offscreen rendering (restore at OffscreenExit)
            FixedFunctionShader::saveOffscreenState();
            flushPendingRT();
            if (!shouldSuppressMWState()) {
                ProxyDevice::BeginScene();
            }
            if (ImGuiManager::GetCmdBufferRecording() && !ImGuiManager::GetCmdBufferReplay()) {
                // Only record BeginScene when not replaying — during replay, the Offscreen
                // buffer is replayed inside an existing scene (from ensureSceneActive).
                // Recording BeginScene/EndScene would close that scene and break subsequent draws.
                g_cmdBufferSet.active().recordBeginScene();
            }
            g_offscreen.megaSceneOpen = true;
        }
        // Device is already in a scene — mark as forwarded so draws go through
        g_deferred.sceneForwarded = true;
        g_deferred.scenePending = false;
    } else {
        g_offscreen.suppressingCurrentScene = false;
        // Transitioning back to normal rendering — close the offscreen mega-scene
        if (g_offscreen.megaSceneOpen) {
            FixedFunctionShader::transitionTo(PhaseTransition::OffscreenExit);
            if (ImGuiManager::GetCmdBufferRecording() && !ImGuiManager::GetCmdBufferReplay()) {
                g_cmdBufferSet.active().recordEndScene();
            }
            if (!shouldSuppressMWState()) {
                ProxyDevice::EndScene();
            }
            // Restore blend state after offscreen rendering (saved at OffscreenEntry)
            FixedFunctionShader::restoreOffscreenState();
            g_offscreen.megaSceneOpen = false;
            g_cmdBufferSet.activeStage = CmdStage::PreScene;
            // Safe zone restart now happens in EndScene (captures gap before Scene 0)
        }
    }

    ImGuiManager::LogFrameEvent(FrameEvent::BeginScene, g_scene.sceneCount);

    if (mwBridge->IsLoaded() && g_scene.rendertargetNormal) {
        if (!g_scene.isHUDready) {
            // Initialize HUD
            StatusOverlay::init(realDevice);
            StatusOverlay::setStatus(XE_VERSION_STRING);
            MGEhud::init(realDevice);

            // Set scaling on Morrowind's UI system
            if (Configuration.UIScale != 1.0f) {
                mwBridge->setUIScale(Configuration.UIScale);
            }

            g_scene.isHUDready = true;
        }

        // Detect scene phase by characteristics before processing
        g_scene.phase = detectScenePhase();

        // Scene 1/2 (particles, hands) may use view matrices that trigger detectMenu() false positive.
        // Force main view path for Scene 1/2 after we've had a valid Scene 0.
        // sceneCount starts at -1. After Scene 0 increment it's 0, after Scene 1 it's 1.
        // Main menu has sceneCount=-1 and isMainView=false - must NOT override that.
        bool isScene12Override = g_scene.sceneCount >= 0 && g_scene.sceneCount < 2;
        if (g_scene.isMainView || isScene12Override) {
            // Track scene count here in BeginScene
            // g_scene.isMainView is not always valid at EndScene if Morrowind draws sunglare
            ++g_scene.sceneCount;

            // Log phase transition for debugging
            static const char* phaseNames[] = { "Unknown", "Offscreen", "World", "Particles", "Hands", "UI" };
            if (ImGuiManager::GetHandoverLogging()) {
                LOG::logline("BeginScene: sceneCount=%d, phase=%s",
                    g_scene.sceneCount, phaseNames[static_cast<int>(g_scene.phase)]);
            }

            // Stage transitions for per-stage command buffers
            // sceneCount starts at -1; after increment: 0=Scene0, 1=Scene1, 2=Scene2
            if (g_scene.sceneCount == 0) {
                // Scene 0 (world) - first main scene
                g_cmdBufferSet.activeStage = CmdStage::Scene0;
            } else if (g_scene.sceneCount == 1) {
                // Scene 1 (particles)
                g_cmdBufferSet.activeStage = CmdStage::Scene1;
                // Step 3b: Sync point for async threads
                // Wait for CPU prep and GPU phase to complete before Scene 1 draws
                if (isHLSLActive() && g_scene.stage0Complete) {
                    if (g_cpuPrepThread && g_cpuPrepThread->isPending())
                        g_cpuPrepThread->waitForCompletion();
                    if (g_renderThread && g_renderThread->isPending())
                        g_renderThread->waitForCompletion();
                }
                // Phase transition: Scene1Entry - validates state matches RecordingExit
                if (isHLSLActive()) {
                    FixedFunctionShader::transitionTo(PhaseTransition::Scene1Entry);

                    // Capture Scene 1 view/proj (camera may have moved during Scene 0)
                    FixedFunctionShader::captureScene1Matrices();

                    // Start recording Scene 1 (particles) to separate vector
                    FixedFunctionShader::setCurrentRecordingScene(1);
                    FixedFunctionShader::setRecordingState(true);
                }
            } else if (g_scene.sceneCount == 2) {
                // Scene 2 (hands) - detected by Z-clear after world
                g_cmdBufferSet.activeStage = CmdStage::Scene2;
                g_scene.handsStarted = true;
                g_scene.hadZClearSinceWorld = false;  // Consumed
                if (isHLSLActive()) {
                    // Capture Scene 2 view/proj (hands use different view matrix than world)
                    FixedFunctionShader::captureScene2Matrices();
                    // Continue recording Scene 2 (hands)
                    FixedFunctionShader::setCurrentRecordingScene(2);
                    FixedFunctionShader::setRecordingState(true);
                }
            } else if (g_scene.sceneCount > 2 && isHLSLActive()) {
                // Scene 3+: Continue recording (unlikely, but handle gracefully)
                FixedFunctionShader::setCurrentRecordingScene(g_scene.sceneCount);
                FixedFunctionShader::setRecordingState(true);
            }

            // Set any custom FOV and check distant water state (Scene 0 = post-increment 0)
            if (g_scene.sceneCount == 0) {
                // Log offscreen local map timing if any offscreen work happened
                if (g_offscreen.timingActive) {
                    float totalMs = g_offscreen.stopTimingMs();
                    if (LOG::catEnabled(LOG::Cat_FrameStats)) {
                        LOG::logline("Local map: %.1fms, %d DIPs, %d scenes", totalMs, g_offscreen.dipCount, g_offscreen.sceneCount);
                    }
                }
                if (Configuration.ScreenFOV > 0) {
                    mwBridge->SetFOV(Configuration.ScreenFOV);
                }
                g_scene.distantWater = (Configuration.MGEFlags & USE_DISTANT_LAND) || (Configuration.MGEFlags & USE_DISTANT_WATER);
            }
        } else {
            // UI scene — frame finalize point (detected by !isMainView after world scenes)
            g_scene.phase = ScenePhase::UI;

            // End safe zone: UI needs device for menu/HUD rendering
            if (g_renderThreadSafeZone) {
                auto elapsed = std::chrono::high_resolution_clock::now() - g_safeZoneStart;
                g_lastSafeZoneMs = std::chrono::duration<float, std::milli>(elapsed).count();
                MGE_TracyPlot("RT_SafeZone_ms", g_lastSafeZoneMs);
                MGE_TracyMessage("RT_SafeZone_END_UI", 18);
                g_renderThreadSafeZone = false;
            }

            if (ImGuiManager::GetHandoverLogging()) {
                LOG::logline("BeginScene: phase=UI (GPU phase trigger)");
            }

            if (isHLSLActive()) {
                // Stop Scene 1/2 recording before UI
                if (FixedFunctionShader::getIsRecording()) {
                    FixedFunctionShader::setRecordingState(false);
                }
                FixedFunctionShader::transitionTo(PhaseTransition::UIEntry);
            }
            g_cmdBufferSet.activeStage = CmdStage::UI;
            if (DistantLand::ready && g_scene.sceneCount >= 0 && !g_scene.isFrameComplete) {
                ensureSceneActive();

                // Track scene emptiness for debugging
                if (isHLSLActive()) {
                    auto& recBuf = FixedFunctionShader::getRecordingBuffer();
                    g_scene.scene0Empty = recBuf.recordedCalls.empty();
                    g_scene.scene1Empty = recBuf.recordedCallsScene1.empty();
                    g_scene.scene2Empty = recBuf.recordedCallsScene2.empty();

                    // Log empty scenes for debugging
                    if (g_scene.scene0Empty) {
                        ImGuiManager::LogFrameEvent(FrameEvent::Scene0_Empty, g_scene.sceneCount);
                    }
                    if (g_scene.scene1Empty) {
                        ImGuiManager::LogFrameEvent(FrameEvent::Scene1_Empty, g_scene.sceneCount);
                    }
                    if (g_scene.scene2Empty) {
                        ImGuiManager::LogFrameEvent(FrameEvent::Scene2_Empty, g_scene.sceneCount);
                    }
                }

                // HLSL GPU phase: Phase 2 waits for prep, then submits GPU sync
                if (isHLSLActive() && g_scene.stage0Complete) {
                    // N-1: Store current frame's waterSeen into recording buffer
                    FixedFunctionShader::getRecordingBuffer().waterSeen = g_scene.waterDrawn;

                    if (ImGuiManager::GetAsyncGpuThread() && g_cpuPrepThread && g_cpuPrepThread->isRunning()
                        && g_renderThread && g_renderThread->isRunning()
                        && FixedFunctionShader::isN2Ready()) {  // Wait for 2-frame warm-up
                        // Phase 3: Wait for async work submitted at Present()
                        // GPU has been running in parallel with main thread recording
                        {
                            MGE_ZoneScopedN("WaitForCpuPrepThread");
                            g_cpuPrepThread->waitForCompletion();
                            MGE_TracyMessage("CPT_PrepComplete", 15);
                        }
                        {
                            MGE_ZoneScopedN("WaitForGpuWork");
                            g_renderThread->waitForCompletion();
                            MGE_TracyMessage("RT_GpuComplete", 14);
                        }
                        // State restoration happens after GpuExit (below) for both paths
                    } else {
                        // Async toggle OFF: run GPU phase on main thread
                        FixedFunctionShader::renderFullFrameAsync();
                        // Clear depth so UI isn't depth-tested against 3D geometry
                        realDevice->Clear(0, NULL, D3DCLEAR_ZBUFFER, 0, 1.0f, 0);
                    }
                    FixedFunctionShader::transitionTo(PhaseTransition::GpuExit);
                    // Apply tracked state so UI sees what MW expects.
                    // MUST be after GpuExit: GpuExit.txt sets alphaBlendEnable=0 which would overwrite.
                    // MW's pre-BeginScene(UI) state calls are suppressed during recording.
                    {
                        auto& tracker = g_cmdBufferSet.stateTracker();
                        DWORD val;
                        if (tracker.getRenderState(D3DRS_ALPHABLENDENABLE, &val))
                            realDevice->SetRenderState(D3DRS_ALPHABLENDENABLE, val);
                        if (tracker.getRenderState(D3DRS_SRCBLEND, &val))
                            realDevice->SetRenderState(D3DRS_SRCBLEND, val);
                        if (tracker.getRenderState(D3DRS_DESTBLEND, &val))
                            realDevice->SetRenderState(D3DRS_DESTBLEND, val);
                        if (tracker.getRenderState(D3DRS_ZENABLE, &val))
                            realDevice->SetRenderState(D3DRS_ZENABLE, val);
                        if (tracker.getRenderState(D3DRS_ZWRITEENABLE, &val))
                            realDevice->SetRenderState(D3DRS_ZWRITEENABLE, val);
                        if (tracker.getRenderState(D3DRS_ZFUNC, &val))
                            realDevice->SetRenderState(D3DRS_ZFUNC, val);
                        if (tracker.getRenderState(D3DRS_ALPHAREF, &val))
                            realDevice->SetRenderState(D3DRS_ALPHAREF, val);
                        if (tracker.getRenderState(D3DRS_ALPHATESTENABLE, &val))
                            realDevice->SetRenderState(D3DRS_ALPHATESTENABLE, val);
                        if (tracker.getRenderState(D3DRS_ALPHAFUNC, &val))
                            realDevice->SetRenderState(D3DRS_ALPHAFUNC, val);
                        if (tracker.getRenderState(D3DRS_CULLMODE, &val))
                            realDevice->SetRenderState(D3DRS_CULLMODE, val);
                        if (tracker.getRenderState(D3DRS_FOGENABLE, &val))
                            realDevice->SetRenderState(D3DRS_FOGENABLE, val);
                        if (tracker.getRenderState(D3DRS_LIGHTING, &val))
                            realDevice->SetRenderState(D3DRS_LIGHTING, val);
                        if (tracker.getRenderState(D3DRS_AMBIENTMATERIALSOURCE, &val))
                            realDevice->SetRenderState(D3DRS_AMBIENTMATERIALSOURCE, val);
                        if (tracker.getRenderState(D3DRS_DIFFUSEMATERIALSOURCE, &val))
                            realDevice->SetRenderState(D3DRS_DIFFUSEMATERIALSOURCE, val);
                        // Texture stage states for UI (stage 0)
                        if (tracker.getTextureStageState(0, D3DTSS_COLOROP, &val))
                            realDevice->SetTextureStageState(0, D3DTSS_COLOROP, val);
                        if (tracker.getTextureStageState(0, D3DTSS_COLORARG1, &val))
                            realDevice->SetTextureStageState(0, D3DTSS_COLORARG1, val);
                        if (tracker.getTextureStageState(0, D3DTSS_COLORARG2, &val))
                            realDevice->SetTextureStageState(0, D3DTSS_COLORARG2, val);
                        if (tracker.getTextureStageState(0, D3DTSS_ALPHAOP, &val))
                            realDevice->SetTextureStageState(0, D3DTSS_ALPHAOP, val);
                        if (tracker.getTextureStageState(0, D3DTSS_ALPHAARG1, &val))
                            realDevice->SetTextureStageState(0, D3DTSS_ALPHAARG1, val);
                        if (tracker.getTextureStageState(0, D3DTSS_ALPHAARG2, &val))
                            realDevice->SetTextureStageState(0, D3DTSS_ALPHAARG2, val);
                        if (tracker.getFVF(&val))
                            realDevice->SetFVF(val);
                        for (auto& [index, enable] : tracker.lightEnables)
                            realDevice->LightEnable(index, enable);
                        D3DMATRIX mat;
                        if (tracker.getTransform(D3DTS_VIEW, &mat))
                            realDevice->SetTransform(D3DTS_VIEW, &mat);
                        if (tracker.getTransform(D3DTS_PROJECTION, &mat))
                            realDevice->SetTransform(D3DTS_PROJECTION, &mat);
                        if (tracker.getTransform(D3DTS_WORLD, &mat))
                            realDevice->SetTransform(D3DTS_WORLD, &mat);
                        realDevice->SetVertexShader(NULL);
                        realDevice->SetPixelShader(NULL);

                        // STRESS TEST: Verify state restoration after GpuExit
                        if (ImGuiManager::GetStressVerifyRestore()) {
                            DWORD expected, actual;
                            bool mismatch = false;
                            #define VERIFY_RS(state) \
                                if (tracker.getRenderState(state, &expected)) { \
                                    realDevice->GetRenderState(state, &actual); \
                                    if (expected != actual) { \
                                        LOG::logline("!! RESTORE FAILED: " #state " expected=%d actual=%d", expected, actual); \
                                        mismatch = true; \
                                    } \
                                }
                            VERIFY_RS(D3DRS_ALPHABLENDENABLE);
                            VERIFY_RS(D3DRS_SRCBLEND);
                            VERIFY_RS(D3DRS_DESTBLEND);
                            VERIFY_RS(D3DRS_ZENABLE);
                            VERIFY_RS(D3DRS_ZWRITEENABLE);
                            VERIFY_RS(D3DRS_ZFUNC);
                            VERIFY_RS(D3DRS_ALPHAREF);
                            VERIFY_RS(D3DRS_ALPHATESTENABLE);
                            VERIFY_RS(D3DRS_ALPHAFUNC);
                            VERIFY_RS(D3DRS_CULLMODE);
                            VERIFY_RS(D3DRS_FOGENABLE);
                            VERIFY_RS(D3DRS_LIGHTING);
                            VERIFY_RS(D3DRS_AMBIENTMATERIALSOURCE);
                            VERIFY_RS(D3DRS_DIFFUSEMATERIALSOURCE);
                            #undef VERIFY_RS

                            // Verify texture stage states
                            #define VERIFY_TSS(stage, state) \
                                if (tracker.getTextureStageState(stage, state, &expected)) { \
                                    realDevice->GetTextureStageState(stage, state, &actual); \
                                    if (expected != actual) { \
                                        LOG::logline("!! RESTORE FAILED: TSS[%d]." #state " expected=%d actual=%d", stage, expected, actual); \
                                        mismatch = true; \
                                    } \
                                }
                            VERIFY_TSS(0, D3DTSS_COLOROP);
                            VERIFY_TSS(0, D3DTSS_COLORARG1);
                            VERIFY_TSS(0, D3DTSS_COLORARG2);
                            VERIFY_TSS(0, D3DTSS_ALPHAOP);
                            VERIFY_TSS(0, D3DTSS_ALPHAARG1);
                            VERIFY_TSS(0, D3DTSS_ALPHAARG2);
                            #undef VERIFY_TSS

                            if (mismatch) {
                                LOG::logline("!! State restoration mismatch detected - check mgexe.log");
                            }
                        }
                    }
                } else if (isHLSLActive()) {
                    FixedFunctionShader::transitionTo(PhaseTransition::GpuExit);
                } else {
                    // Non-HLSL: synchronous postProcess
                    DistantLand::postProcess(&frameCtx);
                }
            }

            // Render user HUD before Morrowind HUD
            if (g_scene.isHUDready && !g_scene.isHUDComplete) {
                ensureSceneActive();
                MGEhud::draw();
            }

            g_scene.isFrameComplete = true;
        }
    }

    return D3D_OK;
}

// EndScene - Multiple scenes per frame, non-alpha / 2x stencil / post-stencil redraw / alpha / 1st person / UI
// MGE intercepts first scene to draw distant land before it finishes, others it applies shadows to
HRESULT _stdcall MGEProxyDevice::EndScene() {
    ImGuiManager::LogFrameEvent(FrameEvent::EndScene, g_scene.sceneCount);

    // Restart safe zone after offscreen EndScene - captures gap before Scene 0
    // Main thread done with offscreen API calls, GPU may still be processing
    if (!g_scene.rendertargetNormal && g_safeZoneEndedForOffscreen) {
        g_renderThreadSafeZone = true;
        g_safeZoneStart = std::chrono::high_resolution_clock::now();
        g_safeZoneEndedForOffscreen = false;
        MGE_TracyMessage("RT_SafeZone_START_PostOffscreen", 31);
    }

    if (DistantLand::ready && g_scene.rendertargetNormal) {
        // Ensure real device scene is active before MGE renders anything
        ensureSceneActive();

        // The following Morrowind scenes get past the filters:
        // ~ Opaque meshes, plus alpha meshes with 'No Sorter' property (which should use alpha test)
        // ~ If stencil shadows are active, then shadow casters are deferred to be drawn in a scene after
        //    shadows are fully applied to avoid self-shadowing problems with simplified shadow meshes
        // ~ If any alpha meshes are visible, they are sorted and drawn in another scene (except those with 'No Sorter' property)
        // ~ If 1st person or sunglare is visible, they are drawn in another scene after a Z clear
        if (g_scene.sceneCount == 0) {
            // Transition to InterScene after Scene0 ends (sceneCount is 0 post-increment from -1)
            g_cmdBufferSet.activeStage = CmdStage::InterScene;

            if (isHLSLActive()) {
                // HLSL path: do GPU work at EndScene 0 (sync, same timing as legacy)
                // Scene 1/2 draws happen after this, so Scene 0 must be complete first.
                if (!g_scene.stage0Complete) {
                    // Async: wait for async threads before captureStage0Context.
                    // captureStage0Context writes s_staging which DL render functions read.
                    if (ImGuiManager::GetAsyncGpuThread()) {
                        if (g_cpuPrepThread && !g_cpuPrepThread->isComplete()) {
                            MGE_ZoneScopedN("WaitForCPT_BeforeCapture");
                            g_cpuPrepThread->waitForCompletion();
                        }
                        if (g_renderThread && !g_renderThread->isComplete()) {
                            MGE_ZoneScopedN("WaitForRT_BeforeCapture");
                            g_renderThread->waitForCompletion();
                            MGE_TracyMessage("RT_AsyncComplete_ES0", 20);
                        }
                    }
                    frameCtx = DistantLand::captureStage0Context();
                    g_scene.stage0Complete = true;
                    // N-1: Store context into recording buffer for rendering next frame
                    auto& recBuf = FixedFunctionShader::getRecordingBuffer();
                    recBuf.dlContext = frameCtx;
                    // N-1: Stamp currentView/currentProj to match dlContext camera
                    recBuf.currentView = frameCtx.mwView;
                    recBuf.currentProj = frameCtx.mwProj;
                    capturePostProcessData(recBuf, frameCtx, "ENDSCENE");

                    // Diagnostic: log nearViewRange when stored
                    if (LOG::catEnabled(LOG::Cat_DistantLand)) {
                        static int storeLogCount = 0;
                        if (storeLogCount++ < 10) {
                            LOG::logline("[N1-STORE] nearViewRange=%.1f stored to recording buffer", frameCtx.nearViewRange);
                        }
                    }
                }

                // Phase transition: RecordingExit - MW state at end of Scene 0
                FixedFunctionShader::transitionTo(PhaseTransition::RecordingExit);

                // Capture MW device state before replay (for Scene 1/2 restoration)
                FixedFunctionShader::capturePostRecordingState();

                // DEFERRED: GPU work happens at UI BeginScene after all scenes recorded
                // finalizeAndRender moved to UI BeginScene
            } else {
                // Legacy path: interleaved GPU work as before
                if (!g_scene.stage0Complete) {
                    frameCtx = DistantLand::renderStage0();
                    g_scene.stage0Complete = true;
                }
                FixedFunctionShader::finalizeBatchAndSubmitCull();
                DistantLand::renderStage1(&frameCtx);
                DistantLand::renderStageBlend(&frameCtx);
                FixedFunctionShader::waitCullAndReplay();
            }

            // Mark world scene complete — enables particles/hands detection
            g_scene.worldComplete = true;
            g_scene.phase = ScenePhase::Particles;  // Default next phase (may become Hands if Z-clear)
            if (ImGuiManager::GetHandoverLogging()) {
                LOG::logline("EndScene(0): worldComplete=true, awaiting next scene");
            }
        } else if (!g_scene.isFrameComplete) {
            // Draw water if the Morrowind water plane doesn't appear in view
            // it may be too distant or stencil scene order is non-normative
            LOG_CAT(LOG::Cat_FrameStats, "DW:%d, WD:%d, SS:%d", g_scene.distantWater, g_scene.waterDrawn, g_scene.isStencilScene);

            if (g_scene.distantWater && !g_scene.waterDrawn && !g_scene.isStencilScene) {
                if (isHLSLActive()) {
                    // HLSL: just flag — finalizeAndRender will render water in GPU phase
                    g_scene.waterDrawn = true;
                } else {
                    DistantLand::renderStageWater(&frameCtx);
                    LOG::logline("Water Rendered for non HLSL for weird scenes.");
                    g_scene.waterDrawn = true;
                }
            }
        }
    }

    if (g_scene.isFrameComplete && g_scene.isHUDready && !g_scene.isHUDComplete) {
        ensureSceneActive();
        // Capture post-UI screenshots here
        DistantLand::checkCaptureScreenshot(true);

        // Render status overlay
        StatusOverlay::setFPS(calcFPS());
        if (ImGuiManager::GetShowFrameNumber()) {
            StatusOverlay::setFrameNumber(g_frameNumber);
        } else {
            StatusOverlay::clearFrameNumber();
        }
        StatusOverlay::show(realDevice);

        g_scene.isHUDComplete = true;
        if (isHLSLActive()) {
            FixedFunctionShader::logSceneHandoverState("HUDComplete");
        }
    }

    // Finalize any HLSL batch immediately after scene draw calls complete
    // This ensures HLSL replay happens within the same scene, not deferred to next stage
    FixedFunctionShader::finalizeBatchAndReplay(g_scene.sceneCount);

    // Render depth for Scene 1/2 - HLSL defers to UI BeginScene, legacy runs here
    if (!g_scene.isFrameComplete && g_scene.sceneCount > 0) {
        if (!isHLSLActive()) {
            // Legacy path only - HLSL defers depth to UI BeginScene
            DistantLand::renderStage2(&frameCtx);
        }
    }

    // Log per-scene draw characteristics once (to understand what each scene contains)
    if (!g_sceneStatsLogged && g_scene.sceneCount >= 1 && g_sceneStats[1].total > 0) {
        LOG::logline("=== SCENE DRAW CHARACTERISTICS (first frame with data) ===");
        for (int i = 0; i < 4; i++) {
            auto& s = g_sceneStats[i];
            if (s.total > 0) {
                LOG::logline("Scene %d: total=%d, skinned=%d, blended=%d, zWrite=%d, alphaTest=%d",
                    i, s.total, s.skinned, s.blended, s.zWriting, s.alphaTested);
            }
        }
        LOG::logline("=== END SCENE CHARACTERISTICS ===");
        g_sceneStatsLogged = true;
    }

    // Track offscreen scenes
    if (!g_scene.rendertargetNormal) {
        g_offscreen.scenesThisFrame++;
        // Don't forward EndScene — mega-scene stays open until normal rendering resumes
        g_deferralCounters.scenesForwarded++;
        g_deferred.sceneForwarded = false;
        g_deferred.scenePending = false;
        return D3D_OK;
    }

    // Only forward EndScene if BeginScene was actually forwarded to real device
    if (g_deferred.sceneForwarded) {
        g_deferralCounters.scenesForwarded++;
        g_deferred.sceneForwarded = false;
        g_deferred.scenePending = false;
        if (ImGuiManager::GetCmdBufferRecording()) {
            // Don't record EndScene to UI buffer — UI replay happens within the existing scene
            if (g_cmdBufferSet.activeStage != CmdStage::UI) {
                g_cmdBufferSet.active().recordEndScene();
            }
        }

        // UI replay disabled — UI draws go direct to device for correct hit-testing
        // and inherited state. Will be re-enabled when Scene0 suppression + render thread
        // overlap is implemented (UI stays on main thread either way).
        // UI command buffer is still recorded for diagnostics and future use.

        HRESULT hr = ProxyDevice::EndScene();
        return hr;
    }

    // Empty scene — swallow both BeginScene and EndScene
    if (g_deferred.rtPending) {
        g_deferralCounters.rtSuppressed++;
        g_deferred.rtPending = false;
    }
    g_deferralCounters.scenesSuppressed++;
    g_deferred.scenePending = false;
    g_deferred.sceneForwarded = false;
    return D3D_OK;
}

// CopyRects — intercept GPU→CPU readback for offscreen amortization
HRESULT _stdcall MGEProxyDevice::CopyRects(IDirect3DSurface8* a, const RECT* b, UINT c, IDirect3DSurface8* d, const POINT* e) {
    g_deferralCounters.copyRectsTotal++;

    // Check if this is an offscreen RT→sysmem readback we should suppress
    // Only suppress when the current scene was actually suppressed (over budget)
    if (g_offscreen.suppressingCurrentScene) {
        IDirect3DSurface9* a_real = static_cast<ProxySurface*>(a)->realSurface;
        IDirect3DSurface9* d_real = static_cast<ProxySurface*>(d)->realSurface;
        D3DSURFACE_DESC9 source, dest;
        if (a_real->GetDesc(&source) == D3D_OK && d_real->GetDesc(&dest) == D3D_OK) {
            if (source.Usage == 1 && dest.Usage == 0) {
                g_deferralCounters.copyRectsSuppressed++;
                return D3D_OK;
            }
        }
    }

    // DIAGNOSTIC: CopyRects deferral disabled — always execute immediately.
    // if (isHLSLActive() && ImGuiManager::GetCmdBufferReplay() && b == NULL && e == NULL) {
    //     ...defer to Offscreen buffer replay...
    // }

    return ProxyDevice::CopyRects(a, b, c, d, e);
}

// Clear - Occurs at start of frame, and also a z-clear before rendering 1st person and sunglare
// Skybox mesh doesn't extend over whole background; cleared background colour is visible at horizon
HRESULT _stdcall MGEProxyDevice::Clear(DWORD a, const D3DRECT* b, DWORD c, D3DCOLOR d, float e, DWORD f) {
    g_passBreaks.mw_clear++;
    g_passBreaks.raw_clear++;
    ImGuiManager::LogFrameEvent(FrameEvent::Clear, g_scene.sceneCount);
    ImGuiManager::TraceClear(g_scene.sceneCount, c, d, e);

    // Track Z-clear for hands scene detection
    // Z-clear after world EndScene signals next main scene is hands
    if ((c & D3DCLEAR_ZBUFFER) && g_scene.rendertargetNormal) {
        if (g_scene.worldComplete && !g_scene.handsStarted) {
            g_scene.hadZClearSinceWorld = true;
            if (ImGuiManager::GetHandoverLogging()) {
                LOG::logline("Z-clear after world: next scene is Hands");
            }
        }
    }

    // Handover logging: Clear calls
    if (ImGuiManager::GetHandoverLogging()) {
        char flagStr[32] = "";
        if (c & D3DCLEAR_TARGET) strcat(flagStr, "COLOR ");
        if (c & D3DCLEAR_ZBUFFER) strcat(flagStr, "Z ");
        if (c & D3DCLEAR_STENCIL) strcat(flagStr, "STENCIL ");
        LOG::logline("HANDOVER Clear S%d: flags=[%s] color=0x%08X z=%.3f",
            g_scene.sceneCount, flagStr, d, e);
    }
    DistantLand::setHorizonColour(d);
    if (ImGuiManager::GetCmdBufferRecording()) {
        // Flush pending RT to command buffer BEFORE recording Clear — Clear needs correct target
        flushPendingRT();
        g_cmdBufferSet.active().recordClear(a, c, d, e, f);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    // Suppress Clear for offscreen scenes over amortization budget
    if (!g_scene.rendertargetNormal && g_offscreen.suppressingCurrentScene) return D3D_OK;
    // Flush pending RT before Clear — Clear needs correct target
    flushPendingRT();
    CHECK_DEVICE_RACE("Clear");
    return ProxyDevice::Clear(a, b, c, d, e, f);
}

// SetTransform
// Projection needs modifying to allow room for distant land
// NOTE: Transforms are NOT suppressed — device needs correct values for capture/restore
HRESULT _stdcall MGEProxyDevice::SetTransform(D3DTRANSFORMSTATETYPE a, const D3DMATRIX* b) {
    ImGuiManager::TraceTransform(g_scene.sceneCount, (DWORD)a, b ? (const float*)b : nullptr);

    if (g_scene.rendertargetNormal) {
        if (a == D3DTS_VIEW) {
            // Check for UI view
            g_scene.isMainView = !detectMenu(b);

            if (g_scene.isMainView) {
                D3DXMATRIX view = *b;
                view *= camEffectsMatrix;
                // Store modified view for CPU-side reads (replaces device->GetTransform)
                DistantLand::s_staging.mwView = view;
                // N-1: Capture MODIFIED view (with camEffectsMatrix) to match s_staging.mwView
                captureTransform(a, &view);
                // Always track transforms for HLSL async (MWStateTracker used instead of GetTransform)
                if (ImGuiManager::GetCmdBufferRecording() || isHLSLActive()) {
                    g_cmdBufferSet.stateTracker().trackTransform((DWORD)a, view);
                    if (ImGuiManager::GetCmdBufferRecording()) {
                        g_cmdBufferSet.active().recordSetTransform((DWORD)a, &view);
                    }
                }
                // Suppress transform during async — tracker has the value, device not needed
                if (shouldSuppressMWState()) return D3D_OK;
                CHECK_DEVICE_RACE("SetTransform_View");
                return ProxyDevice::SetTransform(a, &view);
            }
            // Non-main view (Scene 1/2): capture original and track for async
            captureTransform(a, b);
            if (isHLSLActive()) {
                g_cmdBufferSet.stateTracker().trackTransform((DWORD)a, *b);
            }
        } else if (a == D3DTS_PROJECTION) {
            // Only screw with main scene projection
            if (g_scene.isMainView) {
                D3DXMATRIX proj = *b;
                DistantLand::setProjection(&proj);

                if (Configuration.MGEFlags & ZOOM_ASPECT) {
                    proj._11 *= Configuration.CameraEffects.zoom;
                    proj._22 *= Configuration.CameraEffects.zoom;
                }

                // Store modified projection for CPU-side reads (replaces device->GetTransform)
                DistantLand::s_staging.mwProj = proj;
                // Always track transforms for HLSL async
                if (ImGuiManager::GetCmdBufferRecording() || isHLSLActive()) {
                    g_cmdBufferSet.stateTracker().trackTransform((DWORD)a, proj);
                    if (ImGuiManager::GetCmdBufferRecording()) {
                        g_cmdBufferSet.active().recordSetTransform((DWORD)a, &proj);
                    }
                }
                // Suppress transform during async — tracker has the value, device not needed
                if (shouldSuppressMWState()) return D3D_OK;
                CHECK_DEVICE_RACE("SetTransform_Proj");
                return ProxyDevice::SetTransform(a, &proj);
            }
            // Non-main view (Scene 1/2): track for async
            if (isHLSLActive()) {
                g_cmdBufferSet.stateTracker().trackTransform((DWORD)a, *b);
            }
        }
    }

    // Capture non-view/proj transforms (WORLD matrices) and fall-through cases
    captureTransform(a, b);

    // Always track transforms for HLSL async
    if (ImGuiManager::GetCmdBufferRecording() || isHLSLActive()) {
        g_cmdBufferSet.stateTracker().trackTransform((DWORD)a, *b);
        if (ImGuiManager::GetCmdBufferRecording()) {
            g_cmdBufferSet.active().recordSetTransform((DWORD)a, b);
        }
    }
    // Suppress transform during async — tracker has the value, device not needed
    if (shouldSuppressMWState()) return D3D_OK;
    CHECK_DEVICE_RACE("SetTransform");
    return ProxyDevice::SetTransform(a, b);
}

// SetMaterial
// Check for materials marked for hiding
HRESULT _stdcall MGEProxyDevice::SetMaterial(const D3DMATERIAL8* a) {
    captureMaterial(a);
    g_scene.isWaterMaterial = (a->Power == 99999.0f);
    if (g_scene.isWaterMaterial && !g_scene.waterDrawn) {
        LOG_CAT(LOG::Cat_FrameStats, "Water material detected: Power=%.1f", a->Power);
    }

    ImGuiManager::TraceMaterial(g_scene.sceneCount, a->Diffuse.r, a->Diffuse.g, a->Diffuse.b, a->Diffuse.a);

    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordSetMaterial(a);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    CHECK_DEVICE_RACE("SetMaterial");
    return ProxyDevice::SetMaterial(a);
}

// SetLight
// Capture what the sun is doing
HRESULT _stdcall MGEProxyDevice::SetLight(DWORD a, const D3DLIGHT8* b) {
    captureLight(a, b);
    ImGuiManager::TraceLight(g_scene.sceneCount, a, true);

    // Exterior sunlight/interior "sun" appears to always be light 6
    if (a == 6 && DistantLand::ready) {
        DistantLand::setSunLight(b);
    }

    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordSetLight(a, b);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    CHECK_DEVICE_RACE("SetLight");
    return ProxyDevice::SetLight(a, b);
}

// SetRenderState
// Ignore Morrowind fog settings, and run stage 0 rendering after lighting setup
HRESULT _stdcall MGEProxyDevice::SetRenderState(D3DRENDERSTATETYPE a, DWORD b) {
    captureRenderState(a, b);
    ImGuiManager::TraceRS(g_scene.sceneCount, (DWORD)a, b);

    if (a == D3DRS_FOGVERTEXMODE || a == D3DRS_FOGTABLEMODE) {
        return D3D_OK;
    }
    if ((Configuration.MGEFlags & USE_DISTANT_LAND) && (a == D3DRS_FOGSTART || a == D3DRS_FOGEND)) {
        return D3D_OK;
    }
    if (a == D3DRS_STENCILENABLE) {
        g_scene.isStencilScene = b;
    }
    else if (a == D3DRS_STENCILREF) {
        g_scene.stencilRef = b;
    }

    // Ambient is used for scene detection
    if (a == D3DRS_AMBIENT) {
        // Pure white ambient occurs with skydome and menu mode rendering
        // Ambient is also never set properly when high enough outside that Morrowind renders nothing
        g_scene.isAmbientWhite = (b == 0xffffffff);

        if (!g_scene.isAmbientWhite) {
            // Save real ambient, can be used in future frames if no draw calls are provoked
            RGBVECTOR amb = D3DCOLOR(b);
            DistantLand::setAmbientColour(amb);
            if (lightrs.globalAmbient.r != amb.r ||
                lightrs.globalAmbient.g != amb.g ||
                lightrs.globalAmbient.b != amb.b) {
                lightrs.globalAmbient.r = amb.r;
                lightrs.globalAmbient.g = amb.g;
                lightrs.globalAmbient.b = amb.b;
                ++g_lightStateGen;
            }
        }
    }

    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.recordAndTrackRenderState((DWORD)a, b);
    } else if (isHLSLActive()) {
        // Track render states even outside recording (UI setup calls between EndScene/BeginScene)
        g_cmdBufferSet.stateTracker().trackRenderState((DWORD)a, b);
    }

    // Don't suppress point sprite states — particles need these for correct sizing
    bool isPointSpriteState = (a == D3DRS_POINTSIZE || a == D3DRS_POINTSIZE_MIN ||
                               a == D3DRS_POINTSIZE_MAX || a == D3DRS_POINTSCALEENABLE ||
                               a == D3DRS_POINTSCALE_A || a == D3DRS_POINTSCALE_B ||
                               a == D3DRS_POINTSCALE_C || a == D3DRS_POINTSPRITEENABLE);
    if (shouldSuppressMWState() && (g_renderThreadOwnsDevice.load(std::memory_order_acquire) || !isPointSpriteState)) return D3D_OK;
    CHECK_DEVICE_RACE("SetRenderState");
    return ProxyDevice::SetRenderState(a, b);
}

// SetTextureStageState
// Override some sampler options
HRESULT _stdcall MGEProxyDevice::SetTextureStageState(DWORD a, D3DTEXTURESTAGESTATETYPE b, DWORD c) {
    captureFragmentRenderState(a, b, c);
    ImGuiManager::TraceTSS(g_scene.sceneCount, a, (DWORD)b, c);

    // Check if this is a sampler-related state (filter, address modes)
    // These must NOT be suppressed so we can capture MW's intended values for UI restoration
    bool isSamplerState = false;
    switch (b) {
    case D3DTSS_ADDRESSU: case D3DTSS_ADDRESSV: case D3DTSS_ADDRESSW:
    case D3DTSS_BORDERCOLOR: case D3DTSS_MAGFILTER: case D3DTSS_MINFILTER:
    case D3DTSS_MIPFILTER: case D3DTSS_MIPMAPLODBIAS: case D3DTSS_MAXMIPLEVEL:
    case D3DTSS_MAXANISOTROPY:
        isSamplerState = true;
        break;
    default:
        break;
    }

    // Sampler overrides to ensure trilinear/anisotropic filtering works
    // Note that DX8 had sampling state bound to texture stages instead of samplers
    if (b == D3DTSS_MINFILTER) {
        DWORD filter = (c != D3DTEXF_NONE) ? Configuration.ScaleFilter : D3DTEXF_NONE;
        if (ImGuiManager::GetCmdBufferRecording()) {
            g_cmdBufferSet.recordAndTrackSamplerState(a, D3DSAMP_MINFILTER, filter);
        }
        // Don't suppress sampler states — needed for UI state capture
        return realDevice->SetSamplerState(a, D3DSAMP_MINFILTER, filter);
    } else if (b == D3DTSS_MIPFILTER) {
        DWORD filter = (c != D3DTEXF_NONE) ? D3DTEXF_LINEAR : D3DTEXF_NONE;
        if (ImGuiManager::GetCmdBufferRecording()) {
            g_cmdBufferSet.recordAndTrackSamplerState(a, D3DSAMP_MIPFILTER, filter);
        }
        // Don't suppress sampler states — needed for UI state capture
        return realDevice->SetSamplerState(a, D3DSAMP_MIPFILTER, filter);
    }

    // ProxyDevice::SetTextureStageState splits sampler states from TSS
    // Record at DX9 level — proxy does the split, so we check what it would do
    if (ImGuiManager::GetCmdBufferRecording()) {
        // Mirror proxy's DX8→DX9 sampler state split logic
        D3DSAMPLERSTATETYPE sampler = (D3DSAMPLERSTATETYPE)-1;
        switch (b) {
        case D3DTSS_ADDRESSU: sampler = D3DSAMP_ADDRESSU; break;
        case D3DTSS_ADDRESSV: sampler = D3DSAMP_ADDRESSV; break;
        case D3DTSS_BORDERCOLOR: sampler = D3DSAMP_BORDERCOLOR; break;
        case D3DTSS_MAGFILTER: sampler = D3DSAMP_MAGFILTER; break;
        case D3DTSS_MINFILTER: sampler = D3DSAMP_MINFILTER; break;
        case D3DTSS_MIPFILTER: sampler = D3DSAMP_MIPFILTER; break;
        case D3DTSS_MIPMAPLODBIAS: sampler = D3DSAMP_MIPMAPLODBIAS; break;
        case D3DTSS_MAXMIPLEVEL: sampler = D3DSAMP_MAXMIPLEVEL; break;
        case D3DTSS_MAXANISOTROPY: sampler = D3DSAMP_MAXANISOTROPY; break;
        case D3DTSS_ADDRESSW: sampler = D3DSAMP_ADDRESSW; break;
        default: break;
        }
        if (sampler != (D3DSAMPLERSTATETYPE)-1) {
            g_cmdBufferSet.recordAndTrackSamplerState(a, sampler, c);
        } else {
            g_cmdBufferSet.active().recordSetTextureStageState(a, (DWORD)b, c);
            g_cmdBufferSet.stateTracker().trackTextureStageState(a, (DWORD)b, c);
        }
    } else if (isHLSLActive() && !isSamplerState) {
        // Track non-sampler TSS outside recording (UI setup between EndScene/BeginScene)
        g_cmdBufferSet.stateTracker().trackTextureStageState(a, (DWORD)b, c);
    }
    // During async overlap, suppress ALL TSS (including sampler states) to avoid device race
    // Otherwise, only suppress non-sampler TSS during recording (sampler states needed for UI capture)
    if (shouldSuppressMWState() && (g_renderThreadOwnsDevice.load(std::memory_order_acquire) || !isSamplerState)) return D3D_OK;
    CHECK_DEVICE_RACE("SetTextureStageState");
    return ProxyDevice::SetTextureStageState(a, b, c);
}

// DrawIndexedPrimitive - Where all the drawing happens
// Inspect draw calls for re-use later
HRESULT _stdcall MGEProxyDevice::DrawIndexedPrimitive(D3DPRIMITIVETYPE a, UINT b, UINT c, UINT d, UINT e) {
    // Allow distant land to inspect draw calls
    bool isShadowStencil = g_scene.isStencilScene && g_scene.stencilRef <= 1;

    // Categorize this DIP for Tracy profiling and per-bin counting
    // Scene 0 mainview non-stencil calls are NOT logged here — they get classified
    // more precisely downstream (sky/water/HLSL bins via inspectIndexedPrimitive)
    const char* dipCategory;
    static thread_local char dipBuf[32];
    FrameEvent::Type dipEventType = FrameEvent::Count;  // Count = "don't log yet"
    bool deferEventLog = false;
    if (!g_scene.rendertargetNormal) {
        snprintf(dipBuf, sizeof(dipBuf), "DIP_Offscreen_S%d", g_scene.sceneCount);
        dipCategory = dipBuf;
        dipEventType = FrameEvent::DIP_Offscreen;
        g_offscreen.dipCount++;
        g_dipBinStats.offscreen++;
    } else if (!g_scene.isMainView) {
        dipCategory = "DIP_UI";
        dipEventType = FrameEvent::DIP_UI;
        g_dipBinStats.ui++;
    } else if (isShadowStencil) {
        dipCategory = "DIP_StencilShadow";
        dipEventType = FrameEvent::DIP_StencilShadow;
        g_dipBinStats.stencilShadow++;
    } else if (g_scene.sceneCount == 0) {
        // Scene 0 (world) - sceneCount is 0 post-increment from -1
        dipCategory = "DIP_Scene0";
        g_dipScene0++;
        deferEventLog = true;  // Classified downstream by inspect/water/HLSL replay
        // Track Scene 0 characteristics
        auto& s = g_sceneStats[0];
        s.total++;
        if (rs.vertexBlendState != 0) s.skinned++;
        if (rs.blendEnable) s.blended++;
        if (rs.zWrite) s.zWriting++;
        if (rs.alphaTest) s.alphaTested++;
    } else if (g_scene.sceneCount == 1) {
        // Scene 1 (particles) - sceneCount is 1 post-increment
        dipCategory = "DIP_Scene1";
        g_dipScene1++;
        // Track Scene 1 characteristics
        auto& s = g_sceneStats[1];
        s.total++;
        if (rs.vertexBlendState != 0) s.skinned++;
        if (rs.blendEnable) s.blended++;
        if (rs.zWrite) s.zWriting++;
        if (rs.alphaTest) s.alphaTested++;
        // Classify for DIP stats
        if (rs.vertexBlendState != 0) {
            dipEventType = FrameEvent::DIP_1P_Skinning;
            g_dipBinStats.firstPersonSkinning++;
        } else if (rs.blendEnable) {
            dipEventType = FrameEvent::DIP_1P_Alpha;
            g_dipBinStats.firstPersonAlpha++;
        } else {
            dipEventType = FrameEvent::DIP_1P_Other;
            g_dipBinStats.firstPersonOther++;
        }
    } else if (g_scene.sceneCount >= 2) {
        // Scene 2+ (hands, etc.) - sceneCount is 2+ post-increment
        snprintf(dipBuf, sizeof(dipBuf), "DIP_Scene%d", g_scene.sceneCount);
        dipCategory = dipBuf;
        g_dipScene2++;
        // Track Scene 2+ characteristics
        int idx = (g_scene.sceneCount < 4) ? g_scene.sceneCount : 3;
        auto& s = g_sceneStats[idx];
        s.total++;
        if (rs.vertexBlendState != 0) s.skinned++;
        if (rs.blendEnable) s.blended++;
        if (rs.zWrite) s.zWriting++;
        if (rs.alphaTest) s.alphaTested++;
        // Classify for DIP stats
        if (rs.vertexBlendState != 0) {
            dipEventType = FrameEvent::DIP_1P_Skinning;
            g_dipBinStats.firstPersonSkinning++;
        } else if (rs.blendEnable) {
            dipEventType = FrameEvent::DIP_1P_Alpha;
            g_dipBinStats.firstPersonAlpha++;
        } else {
            dipEventType = FrameEvent::DIP_1P_Other;
            g_dipBinStats.firstPersonOther++;
        }
    } else {
        dipCategory = "DIP_PreScene";
        dipEventType = FrameEvent::DIP_PreScene;
        g_dipBinStats.preScene++;
    }

    if (!deferEventLog) {
        ImGuiManager::LogFrameEvent(dipEventType, g_scene.sceneCount, e);
    }

    // Detailed trace for DIP — log regardless of deferral (trace wants ALL calls)
    if (ImGuiManager::GetTraceEnabled()) {
        FrameEvent::Type traceType = deferEventLog ? FrameEvent::DIP_Opaque : dipEventType;  // placeholder bin for deferred
        ImGuiManager::TraceDIP(g_scene.sceneCount, traceType, rs.fvf, rs.vb, rs.ib, rs.texture,
            e, c, rs.zWrite, rs.cullMode, rs.blendEnable, rs.alphaTest,
            rs.srcBlend, rs.destBlend, rs.vertexBlendState);
    }

    // Command buffer: record ALL draws MW sends, before any suppression or HLSL interception
    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordDrawIndexedPrimitive(a, (INT)baseVertexIndex, b, c, d, e);
    }

#ifdef TRACY_ENABLE
    static constexpr tracy::SourceLocationData dipSrcLoc { nullptr, TracyFunction, TracyFile, (uint32_t)__LINE__, 0 };
    tracy::ScopedZone dipZone(&dipSrcLoc, TRACY_CALLSTACK, g_tracyActive);
    dipZone.Name(dipCategory, strlen(dipCategory));
#endif

    // Suppress offscreen scenes over amortization budget (or via ImGui toggle)
    // Return BEFORE ensureSceneActive so deferred scene stays unforwarded
    if (!g_scene.rendertargetNormal && (g_offscreen.suppressingCurrentScene || ImGuiManager::GetSuppressOffscreen())) return D3D_OK;

    // Debug: suppress other DIP categories via ImGui toggles
    if (!g_scene.isMainView && g_scene.rendertargetNormal && ImGuiManager::GetSuppressUI()) return D3D_OK;
    if (isShadowStencil && ImGuiManager::GetSuppressStencilShadow()) return D3D_OK;
    if (g_scene.sceneCount < 0 && g_scene.rendertargetNormal && ImGuiManager::GetSuppressPreScene()) return D3D_OK;
    // Scene 1/2 per-subcategory suppress (particles/hands)
    if (g_scene.sceneCount > 0 && g_scene.rendertargetNormal && g_scene.isMainView && !isShadowStencil) {
        if (rs.vertexBlendState != 0 && ImGuiManager::GetSuppress1PSkinning()) return D3D_OK;
        if (rs.vertexBlendState == 0 && rs.blendEnable && ImGuiManager::GetSuppress1PAlpha()) return D3D_OK;
        if (rs.vertexBlendState == 0 && !rs.blendEnable && ImGuiManager::GetSuppress1POther()) return D3D_OK;
    }

    // Skip stencil shadow rendering entirely in HLSL mode (HLSL has its own shadows)
    if (isShadowStencil && isHLSLActive()) {
        return D3D_OK;
    }

    // Forward deferred scene to real device — AFTER all suppression checks
    ensureSceneActive();

    // Water suppression — check BEFORE isParticlesOrHands (water may arrive in any scene phase)
    // Must suppress MW water regardless of scene phase to avoid double-rendering with shader water
    if (g_scene.isWaterMaterial && g_scene.rendertargetNormal && DistantLand::ready && g_scene.distantWater) {
        g_dipBinStats.water++;
        ImGuiManager::LogFrameEvent(FrameEvent::DIP_Water, g_scene.sceneCount, e);
        if (ImGuiManager::GetSuppressWater()) {
            // Render wireframe outline to show water mesh location
            realDevice->SetRenderState(D3DRS_FILLMODE, D3DFILL_WIREFRAME);
            HRESULT hr = ProxyDevice::DrawIndexedPrimitive(a, b, c, d, e);
            realDevice->SetRenderState(D3DRS_FILLMODE, D3DFILL_SOLID);
            return hr;
        }
        if (!g_scene.waterDrawn) {
            bool isWorld = (g_scene.phase == ScenePhase::World) || (g_scene.sceneCount == 0);
            if (ImGuiManager::GetHandoverLogging()) {
                LOG::logline("Water DIP: phase=%d, sceneCount=%d, isWorld=%d, isHLSL=%d",
                    static_cast<int>(g_scene.phase), g_scene.sceneCount, isWorld, isHLSLActive());
            }
            // HLSL mode: always flag for deferred rendering (never render immediately)
            if (isHLSLActive()) {
                g_scene.waterDrawn = true;
            } else {
                // Legacy mode: render water immediately
                DistantLand::renderStageWater(&frameCtx);
                LOG::logline("Water is rendered for non HLSL, empty world perhaps?");
                g_scene.waterDrawn = true;
            }
        }
        return D3D_OK;  // Suppress MW water
    }

    // Particles/Hands HLSL suppression — recorded, not drawn immediately
    // Hands use a different view matrix, so isMainView=false, but we still record them
    // Use phase detection with sceneCount fallback (phase may not be set on first frame)
    bool isParticlesOrHands = (g_scene.phase == ScenePhase::Particles || g_scene.phase == ScenePhase::Hands)
                              || (g_scene.sceneCount >= 1 && g_scene.phase != ScenePhase::UI);
    if (isParticlesOrHands && g_scene.rendertargetNormal && !isShadowStencil
        && isHLSLActive() && FixedFunctionShader::getIsRecording()) {
        // Populate primitive fields (normally done in isMainView block)
        rs.primType = a;
        rs.baseIndex = baseVertexIndex;
        rs.minIndex = b;
        rs.vertCount = c;
        rs.startIndex = d;
        rs.primCount = e;
        // Also record to recordMW for depth texture rendering (hands need depth for SSAO/DOF)
        // inspectIndexedPrimitive adds to recordMW if is1PDepthCandidate (skinned or opaque)
        DistantLand::inspectIndexedPrimitive(g_scene.sceneCount, &rs, &frs, &lightrs);
        // Route through renderMorrowind for HLSL color recording
        FixedFunctionShader::renderMorrowind(&rs, &frs, &lightrs, -1);
        return D3D_OK;  // Suppress MW draw
    }
    // Note: UI draws arrive with sceneCount >= 1 but phase=UI - they fall through to normal path below

    if (DistantLand::ready && g_scene.rendertargetNormal && g_scene.isMainView && !isShadowStencil) {
        rs.primType = a;
        rs.baseIndex = baseVertexIndex;
        rs.minIndex = b;
        rs.vertCount = c;
        rs.startIndex = d;
        rs.primCount = e;

        if (!g_scene.stage0Complete && !g_scene.isAmbientWhite) {
            // At this point, only the sky is rendered in exteriors, or nothing in interiors
            if (isHLSLActive()) {
                // HLSL: CPU-only context capture, GPU work deferred to finalizeAndRender
                frameCtx = DistantLand::captureStage0Context();
                // N-1: Store context into recording buffer for rendering next frame
                auto& recBuf = FixedFunctionShader::getRecordingBuffer();
                recBuf.dlContext = frameCtx;
                // N-1: Stamp currentView/currentProj to match dlContext camera
                recBuf.currentView = frameCtx.mwView;
                recBuf.currentProj = frameCtx.mwProj;
                capturePostProcessData(recBuf, frameCtx, "DIP");
                // Diagnostic: log nearViewRange when stored (DIP path)
                if (LOG::catEnabled(LOG::Cat_DistantLand)) {
                    static int storeLogCountDIP = 0;
                    if (storeLogCountDIP++ < 10) {
                        LOG::logline("[N1-STORE-DIP] nearViewRange=%.1f stored to recording buffer", frameCtx.nearViewRange);
                    }
                }
            } else {
                // Legacy: interleaved GPU work (distant land renders now)
                frameCtx = DistantLand::renderStage0();
            }
            g_scene.stage0Complete = true;
        }

        // Water material with distantWater enabled was already handled before this block.
        // Water material with distantWater disabled falls through to render MW water normally.
        if (!g_scene.isWaterMaterial) {
            // Let distant land record call and skip if signalled
            if (!DistantLand::inspectIndexedPrimitive(g_scene.sceneCount, &rs, &frs, &lightrs)) {
                return D3D_OK;
            }
        }
    }

    // Log deferred Scene 0 calls that fell through without specific classification
    // (e.g. fixed-function path, sky without atmosphere scatter)
    if (deferEventLog) {
        ImGuiManager::LogFrameEvent(FrameEvent::DIP_Opaque, g_scene.sceneCount, e);
    }

    // In HLSL mode, inspectIndexedPrimitive already returns false (suppressing) for all
    // draws it records (HLSL scene objects, sky). Draws that reach here returned true
    // from inspect — they are NOT part of the HLSL replay pipeline and must go through
    // to the device immediately.
    // EXCEPT: with suppression ON, strays must be recorded or they're lost (e.g. fade effects)
    if (isHLSLActive() && FixedFunctionShader::getIsRecording() && g_scene.sceneCount <= 2) {
        if (shouldSuppressMWState()) {
            // Record stray draws (fade effects, overlays) that would otherwise be lost
            rs.primType = a;
            rs.baseIndex = baseVertexIndex;
            rs.minIndex = b;
            rs.vertCount = c;
            rs.startIndex = d;
            rs.primCount = e;
            FixedFunctionShader::renderMorrowind(&rs, &frs, &lightrs, -1);
            return D3D_OK;  // Suppressed but now recorded for replay
        }
    }
    if (shouldSuppressMWState()) return D3D_OK;
    CHECK_DEVICE_RACE("DrawIndexedPrimitive");
    return ProxyDevice::DrawIndexedPrimitive(a, b, c, d, e);
}

// Release - Free all resources when refcount hits 0
ULONG _stdcall MGEProxyDevice::Release() {
    ULONG r = ProxyDevice::Release();

    if (r == 0) {
        if (imguiInitialized) {
            ImGuiManager::Shutdown();
            imguiInitialized = false;
        }
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
    LOG::logline("== Initializing data: END ==");
    LOG::logline("== Loading MGE XE: START ==");

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
    LOG::logline("== Loading MGE XE: END ==");

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

// detectScenePhase - Characteristic-based scene detection
// Detects scene type by signals, not position in frame
ScenePhase detectScenePhase() {
    // Offscreen: local map, inventory doll
    if (!g_scene.rendertargetNormal) {
        return ScenePhase::Offscreen;
    }

    // UI: orthographic projection detected
    if (!g_scene.isMainView) {
        return ScenePhase::UI;
    }

    // After world, Z-clear signals hands scene
    if (g_scene.hadZClearSinceWorld) {
        return ScenePhase::Hands;
    }

    // After world EndScene, before Z-clear = particles
    if (g_scene.worldComplete) {
        return ScenePhase::Particles;
    }

    // Main view, world not complete = world scene
    if (g_scene.isMainView) {
        return ScenePhase::World;
    }

    return ScenePhase::Unknown;
}

// --------------------------------------------------------
// State recording

HRESULT _stdcall MGEProxyDevice::SetTexture(DWORD a, IDirect3DBaseTexture8* b) {
    if (a == 0) {
        rs.texture = b ? static_cast<ProxyTexture*>(b)->realTexture : NULL;
    }
    ImGuiManager::TraceTexture(g_scene.sceneCount, a, rs.texture);
    if (ImGuiManager::GetCmdBufferRecording()) {
        IDirect3DBaseTexture9* realTex = b ? static_cast<ProxyTexture*>(b)->realTexture : nullptr;
        g_cmdBufferSet.active().recordSetTexture(a, realTex);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    CHECK_DEVICE_RACE("SetTexture");
    return ProxyDevice::SetTexture(a, b);
}

// Draw method overrides — ensure deferred BeginScene is forwarded before any draw
// Suppress offscreen draws (same logic as DrawIndexedPrimitive)
HRESULT _stdcall MGEProxyDevice::DrawPrimitive(D3DPRIMITIVETYPE a, UINT b, UINT c) {
    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordDrawPrimitive(a, b, c);
    }
    if (!g_scene.rendertargetNormal && (g_offscreen.suppressingCurrentScene || ImGuiManager::GetSuppressOffscreen())) return D3D_OK;
    if (shouldSuppressMWState()) return D3D_OK;
    CHECK_DEVICE_RACE("DrawPrimitive");
    ensureSceneActive();
    return ProxyDevice::DrawPrimitive(a, b, c);
}

HRESULT _stdcall MGEProxyDevice::DrawPrimitiveUP(D3DPRIMITIVETYPE a, UINT b, const void* c, UINT d) {
    if (!g_scene.rendertargetNormal && (g_offscreen.suppressingCurrentScene || ImGuiManager::GetSuppressOffscreen())) return D3D_OK;
    if (shouldSuppressMWState()) return D3D_OK;
    CHECK_DEVICE_RACE("DrawPrimitiveUP");
    ensureSceneActive();
    return ProxyDevice::DrawPrimitiveUP(a, b, c, d);
}

HRESULT _stdcall MGEProxyDevice::DrawIndexedPrimitiveUP(D3DPRIMITIVETYPE a, UINT b, UINT c, UINT d, const void* e, D3DFORMAT f, const void* g, UINT h) {
    if (!g_scene.rendertargetNormal && (g_offscreen.suppressingCurrentScene || ImGuiManager::GetSuppressOffscreen())) return D3D_OK;
    if (shouldSuppressMWState()) return D3D_OK;
    ensureSceneActive();
    return ProxyDevice::DrawIndexedPrimitiveUP(a, b, c, d, e, f, g, h);
}

HRESULT _stdcall MGEProxyDevice::SetVertexShader(DWORD a) {
    rs.fvf = a;
    ImGuiManager::TraceVertexShader(g_scene.sceneCount, a);
    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordSetFVF(a);
        g_cmdBufferSet.stateTracker().trackFVF(a);
    } else if (isHLSLActive()) {
        g_cmdBufferSet.stateTracker().trackFVF(a);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    CHECK_DEVICE_RACE("SetVertexShader");
    return ProxyDevice::SetVertexShader(a);
}

HRESULT _stdcall MGEProxyDevice::SetStreamSource(UINT a, IDirect3DVertexBuffer8* b, UINT c) {
    if (a == 0) {
        rs.vb = (IDirect3DVertexBuffer9*)b;
        rs.vbOffset = 0;
        rs.vbStride = c;
    }
    ImGuiManager::TraceStreamSource(g_scene.sceneCount, a, b, c);
    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordSetStreamSource(a, (IDirect3DVertexBuffer9*)b, 0, c);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    CHECK_DEVICE_RACE("SetStreamSource");
    return ProxyDevice::SetStreamSource(a, b, c);
}

HRESULT _stdcall MGEProxyDevice::SetIndices(IDirect3DIndexBuffer8* a, UINT b) {
    rs.ib = (IDirect3DIndexBuffer9*)a;
    ImGuiManager::TraceIndexBuffer(g_scene.sceneCount, a);
    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordSetIndices((IDirect3DIndexBuffer9*)a);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    CHECK_DEVICE_RACE("SetIndices");
    return ProxyDevice::SetIndices(a, b);
}

HRESULT _stdcall MGEProxyDevice::LightEnable(DWORD a, BOOL b) {
    ImGuiManager::TraceLight(g_scene.sceneCount, a, b != 0);
    if (b) {
        if (std::find(lightrs.active.begin(), lightrs.active.end(), a) == lightrs.active.end()) {
            lightrs.active.push_back(a);
            ++g_lightStateGen;  // Actually adding a light
        }
    } else {
        if (std::remove(lightrs.active.begin(), lightrs.active.end(), a) != lightrs.active.end()) {
            lightrs.active.pop_back();
            ++g_lightStateGen;  // Actually removing a light
        }
    }
    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordLightEnable(a, b);
        g_cmdBufferSet.stateTracker().trackLightEnable(a, b);
    } else if (isHLSLActive()) {
        g_cmdBufferSet.stateTracker().trackLightEnable(a, b);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    CHECK_DEVICE_RACE("LightEnable");
    return ProxyDevice::LightEnable(a, b);
}

void captureRenderState(D3DRENDERSTATETYPE a, DWORD b) {
    // Update legacy RenderedState for existing code paths
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

    // Update complete DeviceStateSnapshot for async replay (no assumptions about prior state)
    switch (a) {
    // Depth states
    case D3DRS_ZENABLE:
        g_deviceState.zEnable = b;
        break;
    case D3DRS_ZWRITEENABLE:
        g_deviceState.zWriteEnable = b;
        break;
    case D3DRS_ZFUNC:
        g_deviceState.zFunc = b;
        break;
    case D3DRS_DEPTHBIAS:
        g_deviceState.depthBias = *(float*)&b;
        break;
    case D3DRS_SLOPESCALEDEPTHBIAS:
        g_deviceState.slopeScaleDepthBias = *(float*)&b;
        break;

    // Culling
    case D3DRS_CULLMODE:
        g_deviceState.cullMode = b;
        break;

    // Blending
    case D3DRS_ALPHABLENDENABLE:
        g_deviceState.alphaBlendEnable = b;
        break;
    case D3DRS_SRCBLEND:
        g_deviceState.srcBlend = b;
        break;
    case D3DRS_DESTBLEND:
        g_deviceState.destBlend = b;
        break;

    // Alpha test
    case D3DRS_ALPHATESTENABLE:
        g_deviceState.alphaTestEnable = b;
        break;
    case D3DRS_ALPHAFUNC:
        g_deviceState.alphaFunc = b;
        break;
    case D3DRS_ALPHAREF:
        g_deviceState.alphaRef = b;
        break;

    // Lighting/Material
    case D3DRS_LIGHTING:
        g_deviceState.lighting = b;
        break;
    case D3DRS_SPECULARENABLE:
        g_deviceState.specularEnable = b;
        break;
    case D3DRS_LOCALVIEWER:
        g_deviceState.localViewer = b;
        break;
    case D3DRS_NORMALIZENORMALS:
        g_deviceState.normalizeNormals = b;
        break;
    case D3DRS_DIFFUSEMATERIALSOURCE:
        g_deviceState.diffuseMatSrc = b;
        break;
    case D3DRS_EMISSIVEMATERIALSOURCE:
        g_deviceState.emissiveMatSrc = b;
        break;
    case D3DRS_AMBIENTMATERIALSOURCE:
        g_deviceState.ambientMatSrc = b;
        break;
    case D3DRS_COLORVERTEX:
        g_deviceState.colorVertex = b;
        break;
    case D3DRS_VERTEXBLEND:
        g_deviceState.vertexBlend = b;
        break;

    // Fog
    case D3DRS_FOGENABLE:
        g_deviceState.fogEnable = b;
        break;

    // Output
    case D3DRS_COLORWRITEENABLE:
        g_deviceState.colorWriteEnable = b;
        break;

    // Stencil
    case D3DRS_STENCILENABLE:
        g_deviceState.stencilEnable = b;
        break;

    // UI-specific
    case D3DRS_AMBIENT:
        g_deviceState.ambient = b;
        break;
    case D3DRS_TEXTUREFACTOR:
        g_deviceState.textureFactor = b;
        break;

    // Clip planes
    case D3DRS_CLIPPLANEENABLE:
        g_deviceState.clipPlaneEnable = b;
        break;

    // Debug
    case D3DRS_FILLMODE:
        g_deviceState.fillMode = b;
        break;

    // Point sprites (particles)
    case D3DRS_POINTSIZE:
        g_deviceState.pointSize = *(float*)&b;
        break;
    case D3DRS_POINTSPRITEENABLE:
        g_deviceState.pointSpriteEnable = b;
        break;
    case D3DRS_POINTSCALEENABLE:
        g_deviceState.pointScaleEnable = b;
        break;
    case D3DRS_POINTSCALE_A:
        g_deviceState.pointScaleA = *(float*)&b;
        break;
    case D3DRS_POINTSCALE_B:
        g_deviceState.pointScaleB = *(float*)&b;
        break;
    case D3DRS_POINTSCALE_C:
        g_deviceState.pointScaleC = *(float*)&b;
        break;
    }
}

void captureFragmentRenderState(DWORD a, D3DTEXTURESTAGESTATETYPE b, DWORD c) {
    FragmentState::Stage* s = &frs.stage[a];

    if (a < 8) {
        if (b == D3DTSS_ADDRESSU) {
            g_deviceState.samplerAddressU[a] = c;
        } else if (b == D3DTSS_ADDRESSV) {
            g_deviceState.samplerAddressV[a] = c;
        }
    }

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
    // Capture data changes as well as enable changes; the per-frame light snapshot
    // cache depends on this generation to know when an in-place LightState mutated.
    LightState::Light* light = &lightrs.lights[a];
    light->type = b->Type;
    light->diffuse = b->Diffuse;
    if (b->Type == D3DLIGHT_POINT) {
        light->position = b->Position;
        light->falloff.x = b->Attenuation0;
        light->falloff.y = b->Attenuation1;
        light->falloff.z = b->Attenuation2;
    } else {
        D3DXVec3Normalize((D3DXVECTOR3*)&light->position, (D3DXVECTOR3*)&b->Direction);
        light->falloff.x = b->Ambient.r;  // Union with ambient
        light->falloff.y = b->Ambient.g;
        light->falloff.z = b->Ambient.b;
    }
    ++g_lightStateGen;
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
// Trace-only overrides — forward to ProxyDevice, log when trace enabled

HRESULT _stdcall MGEProxyDevice::SetViewport(const D3DVIEWPORT8* a) {
    if (a) {
        ImGuiManager::TraceViewport(g_scene.sceneCount, a->X, a->Y, a->Width, a->Height, a->MinZ, a->MaxZ);
        if (ImGuiManager::GetCmdBufferRecording()) {
            g_cmdBufferSet.active().recordSetViewport(a);
        }
    }
    if (shouldSuppressMWState()) return D3D_OK;
    CHECK_DEVICE_RACE("SetViewport");
    return ProxyDevice::SetViewport(a);
}

HRESULT _stdcall MGEProxyDevice::SetClipPlane(DWORD a, const float* b) {
    ImGuiManager::TraceClipPlane(g_scene.sceneCount, a);
    return ProxyDevice::SetClipPlane(a, b);
}

HRESULT _stdcall MGEProxyDevice::MultiplyTransform(D3DTRANSFORMSTATETYPE a, const D3DMATRIX* b) {
    ImGuiManager::TraceMultiplyTransform(g_scene.sceneCount, (DWORD)a, b ? (const float*)b : nullptr);
    return ProxyDevice::MultiplyTransform(a, b);
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
