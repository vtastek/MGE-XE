
#include "mged3d8device.h"
#include "mge_tracy.h"
#include "proxydx/d3d8texture.h"
#include "proxydx/d3d8surface.h"

#include <algorithm>
#include <tlhelp32.h>
#include "mgeversion.h"
#include "configuration.h"
#include "distantland.h"
#include "mwbridge.h"
#include "statusoverlay.h"
#include "userhud.h"
#include "videobackground.h"
#include "imgui_manager.h"
#include "d3dcommandbuffer.h"

bool g_tracyActive = false;

// Pipeline state diagnostic snapshot — filled at Present() before reset
PipelineDiag g_pipelineDiag = {};

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
static DWORD stencilRef;
static bool stage0Complete, isFrameComplete, isHUDComplete;
static bool isWaterMaterial, waterDrawn, distantWater;
static DLContext frameCtx;  // Per-frame rendering context, created by renderStage0


// Deferred scene forwarding — suppress empty BeginScene/EndScene pairs
static bool scenePending = false;      // BeginScene called but not forwarded to real device
static bool sceneForwarded = false;    // BeginScene has been forwarded to real device

// Deferred render target — suppress RT churn from empty scenes
static bool rtPending = false;
static IDirect3DSurface8* pendingRT_color = nullptr;
static IDirect3DSurface8* pendingRT_depth = nullptr;

// Deferral instrumentation — per-frame counters logged at Present()
static int g_scenesRequested = 0, g_scenesForwarded = 0, g_scenesSuppressed = 0;
static int g_rtRequested = 0, g_rtForwarded = 0, g_rtSuppressed = 0;
static int g_copyRectsTotal = 0, g_copyRectsSuppressed = 0;

// Offscreen amortization — limit offscreen scenes per frame
static int g_offscreenScenesThisFrame = 0;
static bool g_suppressingCurrentScene = false;  // true when current offscreen scene is over budget

// Offscreen scene collapsing — keep one device-level scene open for all offscreen work
static bool g_offscreenMegaScene = false;  // true when device has an open scene for offscreen rendering

// Offscreen local map timing
static LARGE_INTEGER g_offscreenStartQPC = {};
static bool g_offscreenTimingActive = false;
static int g_offscreenDIPs = 0;
static int g_offscreenScenes = 0;

static bool zoomSensSaved;
static float zoomSensX, zoomSensY;
static D3DXMATRIX camEffectsMatrix;
static float crosshairTimeout;

static RenderedState rs;
static FragmentState frs;
static LightState lightrs;

static HWND gameWindow = nullptr;
static bool imguiInitialized = false;

// D3D command buffer set — records all MW calls per-stage for future replay
D3DCommandBufferSet g_cmdBufferSet;
static int g_lastCmdBufferSize = 0;  // Snapshot for ImGui display

// Per-frame DIP bin counters (replaces old coarse DIPCounters)
static ImGuiManager::DIPBinStats g_dipBinStats = {};
// Keep coarse counters for Tracy log formatting
static int g_dipScene0 = 0, g_dipScene1plus = 0;

static void initOnLoad();
static bool detectMenu(const D3DMATRIX* m);
static void captureRenderState(D3DRENDERSTATETYPE a, DWORD b);
static void captureFragmentRenderState(DWORD a, D3DTEXTURESTAGESTATETYPE b, DWORD c);
static void captureTransform(D3DTRANSFORMSTATETYPE a, const D3DMATRIX* b);
static void captureLight(DWORD a, const D3DLIGHT8* b);
static void captureMaterial(const D3DMATERIAL8* a);
static float calcFPS();

// When true, MW state/draw calls record to command buffer only — no device forwarding.
// Keeps main thread off the device for threading safety.
// Batch 1: Material, Light, LightEnable, FVF, StreamSource, Indices (low-risk)
// Batch 2: TSS, Texture, RenderState (medium-risk)
// Batch 3: Transform, Viewport, Clear (high-risk — affects coordinate system and framebuffer)
// When true, MW state/draw calls record to command buffer only — no device forwarding.
// Keeps main thread off the device for threading safety.
static inline bool shouldSuppressMWState() {
    // Currently disabled — UI draws go direct to device.
    // Will be re-enabled for Scene0 suppression when render thread overlap is implemented.
    return false;
}



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

    // Viewport — MW sets it before BeginScene (goes to Scene1Plus buffer, not UI)
    D3DVIEWPORT9 vp;
    dev->GetViewport(&vp);                      buf.recordSetViewport(&vp);

    // Transforms — MW sets VIEW/PROJ before BeginScene (goes to Scene1Plus buffer, not UI)
    D3DMATRIX mat;
    dev->GetTransform(D3DTS_VIEW, &mat);        buf.recordSetTransform(D3DTS_VIEW, &mat);
    dev->GetTransform(D3DTS_PROJECTION, &mat);   buf.recordSetTransform(D3DTS_PROJECTION, &mat);

    // Record z-clear into the UI buffer so it's self-contained
    buf.recordClear(0, D3DCLEAR_ZBUFFER, 0, 1.0f, 0);
}

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
        // Handle F11 key to toggle ImGui interface
        static bool f11Pressed = false;
        static int debugCounter = 0;
        if (imguiInitialized) {
            // F11: Toggle PCF interface (gated)
            if (ImGuiManager::GetDebugKeysEnabled()) {
                bool f11State = (GetAsyncKeyState(VK_F11) & 0x8000) != 0;
                if (f11State && !f11Pressed) {
                    LOG::logline(">> F11 key pressed, toggling PCF interface");
                    ImGuiManager::TogglePCFInterface();
                    f11Pressed = true;
                } else if (!f11State) {
                    f11Pressed = false;
                }
            }

            // G key: Toggle debug interface (always active)
            static bool gPressed = false;
            bool gState = (GetAsyncKeyState('G') & 0x8000) != 0;
            if (gState && !gPressed) {
                ImGuiManager::ToggleDebugInterface();
                gPressed = true;
            } else if (!gState) {
                gPressed = false;
            }

            // U: Toggle Hi-Z interface (gated)
            if (ImGuiManager::GetDebugKeysEnabled()) {
                static bool uPressed = false;
                bool uState = (GetAsyncKeyState('U') & 0x8000) != 0;
                if (uState && !uPressed) {
                    ImGuiManager::ToggleHiZInterface();
                    uPressed = true;
                } else if (!uState) {
                    uPressed = false;
                }
            }

            // E: Toggle Frame Event Log (gated)
            if (ImGuiManager::GetDebugKeysEnabled()) {
                static bool ePressed = false;
                bool eState = (GetAsyncKeyState('E') & 0x8000) != 0;
                if (eState && !ePressed) {
                    ImGuiManager::ToggleFrameEventLog();
                    ePressed = true;
                } else if (!eState) {
                    ePressed = false;
                }
            }

            {
                MGE_ZoneScopedN("Present_ImGuiNewFrame");
                ImGuiManager::NewFrame();
            }
            {
                MGE_ZoneScopedN("Present_ImGuiRender");
                ImGuiManager::Render();
            }
        } else {
            // Log every 60 frames that ImGui is not initialized
            if (++debugCounter % 60 == 0) {
                LOG::logline(">> ImGui not initialized (frame %d)", debugCounter);
            }
        }
    }

    // Log render pass break instrumentation
    {
        static int frameCounter = 0;
        if (++frameCounter <= 60 || frameCounter % 300 == 0) {
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
            if (++dipLogCounter <= 60 || dipLogCounter % 300 == 0 || isSpike) {
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
        g_dipScene1plus = 0;
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
            g_cmdBufferSet[CmdStage::Scene1Plus].size(),
            g_cmdBufferSet[CmdStage::UI].size());
        g_cmdBufferSet.clearAll();
    }

    // Fill pipeline diagnostic snapshot (end-of-frame state, before reset)
    {
        auto mwb = MWBridge::get();
        g_pipelineDiag.dipScene0 = g_dipScene0;
        g_pipelineDiag.dipScene1plus = g_dipScene1plus;
        g_pipelineDiag.dipOffscreen = g_dipBinStats.offscreen;
        g_pipelineDiag.dipUI = g_dipBinStats.ui;
        g_pipelineDiag.dipStencilShadow = g_dipBinStats.stencilShadow;
        g_pipelineDiag.dipUnknown = g_dipBinStats.preScene;
        g_pipelineDiag.sceneCount = sceneCount;
        g_pipelineDiag.isMainView = isMainView;
        g_pipelineDiag.rendertargetNormal = rendertargetNormal;
        g_pipelineDiag.stage0Complete = stage0Complete;
        g_pipelineDiag.isFrameComplete = isFrameComplete;
        g_pipelineDiag.isHUDComplete = isHUDComplete;
        g_pipelineDiag.isHUDready = isHUDready;
        g_pipelineDiag.isStencilScene = isStencilScene;
        g_pipelineDiag.isAmbientWhite = isAmbientWhite;
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
        sceneCount = -1;
        stage0Complete = false;
        waterDrawn = false;
        isFrameComplete = false;
        isHUDComplete = false;
        scenePending = false;
        sceneForwarded = false;
        rtPending = false;

        // Log deferral stats on frames with offscreen scenes
        if (g_offscreenScenesThisFrame > 0 || g_copyRectsTotal > 0) {
            LOG::logline("Deferral: scenes %d/%d/%d RT %d/%d/%d offscreen %d (mega-scene) CopyRects %d/%d",
                g_scenesRequested, g_scenesForwarded, g_scenesSuppressed,
                g_rtRequested, g_rtForwarded, g_rtSuppressed,
                g_offscreenScenesThisFrame,
                g_copyRectsTotal, g_copyRectsSuppressed);
        }
        g_scenesRequested = g_scenesForwarded = g_scenesSuppressed = 0;
        g_rtRequested = g_rtForwarded = g_rtSuppressed = 0;
        g_copyRectsTotal = g_copyRectsSuppressed = 0;
        g_offscreenScenesThisFrame = 0;
        g_suppressingCurrentScene = false;
        // Safety: close mega-scene if still open at Present
        if (g_offscreenMegaScene) {
            if (ImGuiManager::GetCmdBufferRecording() && !ImGuiManager::GetCmdBufferReplay()) {
                g_cmdBufferSet.active().recordEndScene();
            }
            ProxyDevice::EndScene();
            g_offscreenMegaScene = false;
        }

        // Reset stage for next frame — Offscreen because MW always starts with offscreen
        // rendering (local map tiles) before the main scene. Events before the first
        // BeginScene (SetRenderTarget, Viewport, Clear, Transforms) must go to the
        // Offscreen buffer, not PreScene.
        g_cmdBufferSet.activeStage = CmdStage::Offscreen;

        // Stamp current camera into the FrameBuffer that just finished recording
        // (for future render pass to use fresh matrices instead of stale recording-time ones)
        {
            auto& fb = FixedFunctionShader::getFrameBuffer(FixedFunctionShader::getRecordingBufferIndex());
            fb.currentView = DistantLand::s_staging.mwView;
            fb.currentProj = DistantLand::s_staging.mwProj;
            fb.currentShadowViewproj[0] = DistantLand::s_staging.smViewproj[0];
            fb.currentShadowViewproj[1] = DistantLand::s_staging.smViewproj[1];
        }

        // Rotate to next FrameBuffer for the next frame's recording
        FixedFunctionShader::rotateRecordingBuffer();

        // Reset per-frame flags
        FixedFunctionShader::resetHiZBuiltFlag();
        FixedFunctionShader::resetRecordingCompletedFlag();

        // Reset HLSL texture caches at frame boundary to prevent stale texture pointers
        FixedFunctionShader::resetHLSLCaches();
    }

    MGE_FrameMark;  // Mark frame boundary at the very end of Present()
    {
        MGE_ZoneScopedN("Present_ProxyDevicePresent");
        return ProxyDevice::Present(a, b, c, d);
    }
}

// Forward pending render target change to real device
static HRESULT flushPendingRT() {
    if (rtPending) {
        rtPending = false;
        g_rtForwarded++;
        auto device = DistantLand::device;
        HRESULT hr1 = D3D_OK, hr2 = D3D_OK;
        IDirect3DSurface9* rtSurface = pendingRT_color ? static_cast<ProxySurface*>(pendingRT_color)->realSurface : nullptr;
        IDirect3DSurface9* dsSurface = pendingRT_depth ? static_cast<ProxySurface*>(pendingRT_depth)->realSurface : nullptr;

        // Record RT/DS changes to command buffer
        if (ImGuiManager::GetCmdBufferRecording()) {
            if (rtSurface) g_cmdBufferSet.active().recordSetRenderTarget(rtSurface);
            g_cmdBufferSet.active().recordSetDepthStencilSurface(dsSurface);
        }

        // Forward to device — except during Offscreen replay (device calls deferred to buffer)
        if (!shouldSuppressMWState()) {
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
        rendertargetNormal = (static_cast<ProxySurface*>(a)->realSurface == back);
        back->Release();
    }

    g_passBreaks.raw_setRT++;
    g_passBreaks.raw_setDS++;
    ImGuiManager::LogFrameEvent(FrameEvent::SetRenderTarget, sceneCount);
    ImGuiManager::TraceRT(sceneCount, a, b);

    // Defer RT change — forward only when a Clear or Draw needs it
    // If previous pending RT was never forwarded, count it as suppressed
    if (rtPending) {
        g_rtSuppressed++;
    }
    g_rtRequested++;
    pendingRT_color = a;
    pendingRT_depth = b;
    rtPending = true;
    return D3D_OK;
}

// Forward deferred RT + BeginScene to real device (call before any draw)
static HRESULT ensureSceneActive() {
    flushPendingRT();
    if (scenePending && !sceneForwarded) {
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
        sceneForwarded = true;
        scenePending = false;
    }
    return D3D_OK;
}

// BeginScene - Multiple scenes per frame, non-alpha / 2x stencil / post-stencil redraw / alpha / 1st person / UI
// Fogging needs to be set for Morrowind rendering at start of scene
HRESULT _stdcall MGEProxyDevice::BeginScene() {
    auto mwBridge = MWBridge::get();

    // Defer real device BeginScene until a draw call needs it
    scenePending = true;
    sceneForwarded = false;
    g_scenesRequested++;

    // Offscreen scene collapsing: keep one device-level scene open for all offscreen work
    // This eliminates per-scene DXVK command buffer submissions
    if (!rendertargetNormal) {
        g_cmdBufferSet.activeStage = CmdStage::Offscreen;
        g_suppressingCurrentScene = false;
        if (!g_offscreenTimingActive) {
            QueryPerformanceCounter(&g_offscreenStartQPC);
            g_offscreenTimingActive = true;
            g_offscreenDIPs = 0;
            g_offscreenScenes = 0;
        }
        g_offscreenScenes++;
        if (!g_offscreenMegaScene) {
            // First offscreen scene: open device scene, keep it open for all subsequent offscreen work
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
            g_offscreenMegaScene = true;
        }
        // Device is already in a scene — mark as forwarded so draws go through
        sceneForwarded = true;
        scenePending = false;
    } else {
        g_suppressingCurrentScene = false;
        // Transitioning back to normal rendering — close the offscreen mega-scene
        if (g_offscreenMegaScene) {
            if (ImGuiManager::GetCmdBufferRecording() && !ImGuiManager::GetCmdBufferReplay()) {
                g_cmdBufferSet.active().recordEndScene();
            }
            if (!shouldSuppressMWState()) {
                ProxyDevice::EndScene();
            }
            g_offscreenMegaScene = false;
            g_cmdBufferSet.activeStage = CmdStage::PreScene;
        }
    }

    ImGuiManager::LogFrameEvent(FrameEvent::BeginScene, sceneCount);

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

            // Stage transitions for per-stage command buffers
            if (sceneCount == 0) {
                g_cmdBufferSet.activeStage = CmdStage::Scene0;
            } else if (sceneCount == 1) {
                g_cmdBufferSet.activeStage = CmdStage::Scene1Plus;
            }

            // Set any custom FOV and check distant water state
            if (sceneCount == 0) {
                // Log offscreen local map timing if any offscreen work happened
                if (g_offscreenTimingActive) {
                    LARGE_INTEGER now, freq;
                    QueryPerformanceFrequency(&freq);
                    QueryPerformanceCounter(&now);
                    float totalMs = (float)((now.QuadPart - g_offscreenStartQPC.QuadPart) * 1000.0 / freq.QuadPart);
                    LOG::logline("Local map: %.1fms, %d DIPs, %d scenes", totalMs, g_offscreenDIPs, g_offscreenScenes);
                    g_offscreenTimingActive = false;
                }
                if (Configuration.ScreenFOV > 0) {
                    mwBridge->SetFOV(Configuration.ScreenFOV);
                }
                distantWater = (Configuration.MGEFlags & USE_DISTANT_LAND) || (Configuration.MGEFlags & USE_DISTANT_WATER);
            }
        } else {
            // UI scene — frame finalize point
            g_cmdBufferSet.activeStage = CmdStage::UI;
            if (DistantLand::ready && sceneCount > 0 && !isFrameComplete) {
                ensureSceneActive();

                if (stage0Complete && isHLSLActive()) {
                    // Save UI scene state — finalizeAndRender changes RT, DS, shaders, blend state
                    IDirect3DSurface9* savedRT = nullptr;
                    IDirect3DSurface9* savedDS = nullptr;
                    realDevice->GetRenderTarget(0, &savedRT);
                    realDevice->GetDepthStencilSurface(&savedDS);

                    IDirect3DStateBlock9* uiStateSaved;
                    realDevice->CreateStateBlock(D3DSBT_ALL, &uiStateSaved);

                    FixedFunctionShader::finalizeAndRender(&frameCtx, waterDrawn);

                    // Restore UI scene state
                    uiStateSaved->Apply();
                    uiStateSaved->Release();

                    realDevice->SetRenderTarget(0, savedRT);
                    realDevice->SetDepthStencilSurface(savedDS);
                    if (savedRT) savedRT->Release();
                    if (savedDS) savedDS->Release();

                    realDevice->Clear(0, NULL, D3DCLEAR_ZBUFFER, 0, 1.0f, 0);
                }

                DistantLand::postProcess(&frameCtx);

                // Record state preamble AFTER postProcess — captures actual device state MW sees
                // postProcess has its own state block but may leak some state
                if (stage0Complete && isHLSLActive() && ImGuiManager::GetCmdBufferRecording()) {
                    recordUIStatePreamble(realDevice, g_cmdBufferSet[CmdStage::UI]);
                }

                // UI command buffer is replayed at EndScene (after all UI draws are recorded)
            }

            // Render user HUD before Morrowind HUD
            if (isHUDready && !isHUDComplete) {
                ensureSceneActive();
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
    ImGuiManager::LogFrameEvent(FrameEvent::EndScene, sceneCount);

    if (DistantLand::ready && rendertargetNormal) {
        // Ensure real device scene is active before MGE renders anything
        ensureSceneActive();

        // The following Morrowind scenes get past the filters:
        // ~ Opaque meshes, plus alpha meshes with 'No Sorter' property (which should use alpha test)
        // ~ If stencil shadows are active, then shadow casters are deferred to be drawn in a scene after
        //    shadows are fully applied to avoid self-shadowing problems with simplified shadow meshes
        // ~ If any alpha meshes are visible, they are sorted and drawn in another scene (except those with 'No Sorter' property)
        // ~ If 1st person or sunglare is visible, they are drawn in another scene after a Z clear
        if (sceneCount == 0) {
            // Transition to InterScene after Scene0 ends
            g_cmdBufferSet.activeStage = CmdStage::InterScene;

            if (isHLSLActive()) {
                // HLSL path: capture context only, defer all GPU work to frame finalize
                if (!stage0Complete) {
                    frameCtx = DistantLand::captureStage0Context();
                    stage0Complete = true;
                }
                FixedFunctionShader::capturePostRecordingState();
            } else {
                // Legacy path: interleaved GPU work as before
                if (!stage0Complete) {
                    frameCtx = DistantLand::renderStage0();
                    stage0Complete = true;
                }
                FixedFunctionShader::finalizeBatchAndSubmitCull();
                DistantLand::renderStage1(&frameCtx);
                DistantLand::renderStageBlend(&frameCtx);
                FixedFunctionShader::waitCullAndReplay();
            }
        } else if (!isFrameComplete) {
            // Draw water if the Morrowind water plane doesn't appear in view
            // it may be too distant or stencil scene order is non-normative
            if (distantWater && !waterDrawn && !isStencilScene) {
                if (isHLSLActive()) {
                    // HLSL: just flag — finalizeAndRender will render water in GPU phase
                    waterDrawn = true;
                } else {
                    DistantLand::renderStageWater(&frameCtx);
                    waterDrawn = true;
                }
            }
        }
    }

    if (isFrameComplete && isHUDready && !isHUDComplete) {
        ensureSceneActive();
        // Capture post-UI screenshots here
        DistantLand::checkCaptureScreenshot(true);

        // Render status overlay
        StatusOverlay::setFPS(calcFPS());
        StatusOverlay::show(realDevice);

        isHUDComplete = true;
    }

    // Finalize any HLSL batch immediately after scene draw calls complete
    // This ensures HLSL replay happens within the same scene, not deferred to next stage
    FixedFunctionShader::finalizeBatchAndReplay(sceneCount);

    // Render depth for Scene 1+ AFTER all geometry has been captured
    // HLSL defers this to finalizeAndRender (renderStage2 clears recordMW, which renderStage1 needs)
    if (!isFrameComplete && sceneCount > 0 && !isHLSLActive()) {
        DistantLand::renderStage2(&frameCtx);
    }

    // Track offscreen scenes
    if (!rendertargetNormal) {
        g_offscreenScenesThisFrame++;
        // Don't forward EndScene — mega-scene stays open until normal rendering resumes
        g_scenesForwarded++;
        sceneForwarded = false;
        scenePending = false;
        return D3D_OK;
    }

    // Only forward EndScene if BeginScene was actually forwarded to real device
    if (sceneForwarded) {
        g_scenesForwarded++;
        sceneForwarded = false;
        scenePending = false;
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
    if (rtPending) {
        g_rtSuppressed++;
        rtPending = false;
    }
    g_scenesSuppressed++;
    scenePending = false;
    sceneForwarded = false;
    return D3D_OK;
}

// CopyRects — intercept GPU→CPU readback for offscreen amortization
HRESULT _stdcall MGEProxyDevice::CopyRects(IDirect3DSurface8* a, const RECT* b, UINT c, IDirect3DSurface8* d, const POINT* e) {
    g_copyRectsTotal++;

    // Check if this is an offscreen RT→sysmem readback we should suppress
    // Only suppress when the current scene was actually suppressed (over budget)
    if (g_suppressingCurrentScene) {
        IDirect3DSurface9* a_real = static_cast<ProxySurface*>(a)->realSurface;
        IDirect3DSurface9* d_real = static_cast<ProxySurface*>(d)->realSurface;
        D3DSURFACE_DESC9 source, dest;
        if (a_real->GetDesc(&source) == D3D_OK && d_real->GetDesc(&dest) == D3D_OK) {
            if (source.Usage == 1 && dest.Usage == 0) {
                g_copyRectsSuppressed++;
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
    ImGuiManager::LogFrameEvent(FrameEvent::Clear, sceneCount);
    ImGuiManager::TraceClear(sceneCount, c, d, e);
    DistantLand::setHorizonColour(d);
    if (ImGuiManager::GetCmdBufferRecording()) {
        // Flush pending RT to command buffer BEFORE recording Clear — Clear needs correct target
        flushPendingRT();
        g_cmdBufferSet.active().recordClear(a, c, d, e, f);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    // Suppress Clear for offscreen scenes over amortization budget
    if (!rendertargetNormal && g_suppressingCurrentScene) return D3D_OK;
    // Flush pending RT before Clear — Clear needs correct target
    flushPendingRT();
    return ProxyDevice::Clear(a, b, c, d, e, f);
}

// SetTransform
// Projection needs modifying to allow room for distant land
HRESULT _stdcall MGEProxyDevice::SetTransform(D3DTRANSFORMSTATETYPE a, const D3DMATRIX* b) {
    captureTransform(a, b);
    ImGuiManager::TraceTransform(sceneCount, (DWORD)a, b ? (const float*)b : nullptr);

    if (rendertargetNormal) {
        if (a == D3DTS_VIEW) {
            // Check for UI view
            isMainView = !detectMenu(b);

            if (isMainView) {
                D3DXMATRIX view = *b;
                view *= camEffectsMatrix;
                // Store modified view for CPU-side reads (replaces device->GetTransform)
                DistantLand::s_staging.mwView = view;
                if (ImGuiManager::GetCmdBufferRecording()) {
                    g_cmdBufferSet.active().recordSetTransform((DWORD)a, &view);
                }
                if (shouldSuppressMWState()) return D3D_OK;
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

                // Store modified projection for CPU-side reads (replaces device->GetTransform)
                DistantLand::s_staging.mwProj = proj;
                if (ImGuiManager::GetCmdBufferRecording()) {
                    g_cmdBufferSet.active().recordSetTransform((DWORD)a, &proj);
                }
                if (shouldSuppressMWState()) return D3D_OK;
                return ProxyDevice::SetTransform(a, &proj);
            }
        }
    }

    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordSetTransform((DWORD)a, b);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    return ProxyDevice::SetTransform(a, b);
}

// SetMaterial
// Check for materials marked for hiding
HRESULT _stdcall MGEProxyDevice::SetMaterial(const D3DMATERIAL8* a) {
    captureMaterial(a);
    isWaterMaterial = (a->Power == 99999.0f);
    ImGuiManager::TraceMaterial(sceneCount, a->Diffuse.r, a->Diffuse.g, a->Diffuse.b, a->Diffuse.a);

    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordSetMaterial(a);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    return ProxyDevice::SetMaterial(a);
}

// SetLight
// Capture what the sun is doing
HRESULT _stdcall MGEProxyDevice::SetLight(DWORD a, const D3DLIGHT8* b) {
    captureLight(a, b);
    ImGuiManager::TraceLight(sceneCount, a, true);

    // Exterior sunlight/interior "sun" appears to always be light 6
    if (a == 6 && DistantLand::ready) {
        DistantLand::setSunLight(b);
    }

    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordSetLight(a, b);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    return ProxyDevice::SetLight(a, b);
}

// SetRenderState
// Ignore Morrowind fog settings, and run stage 0 rendering after lighting setup
HRESULT _stdcall MGEProxyDevice::SetRenderState(D3DRENDERSTATETYPE a, DWORD b) {
    captureRenderState(a, b);
    ImGuiManager::TraceRS(sceneCount, (DWORD)a, b);

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

    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordSetRenderState((DWORD)a, b);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    return ProxyDevice::SetRenderState(a, b);
}

// SetTextureStageState
// Override some sampler options
HRESULT _stdcall MGEProxyDevice::SetTextureStageState(DWORD a, D3DTEXTURESTAGESTATETYPE b, DWORD c) {
    captureFragmentRenderState(a, b, c);
    ImGuiManager::TraceTSS(sceneCount, a, (DWORD)b, c);

    // Sampler overrides to ensure trilinear/anisotropic filtering works
    // Note that DX8 had sampling state bound to texture stages instead of samplers
    if (b == D3DTSS_MINFILTER) {
        DWORD filter = (c != D3DTEXF_NONE) ? Configuration.ScaleFilter : D3DTEXF_NONE;
        if (ImGuiManager::GetCmdBufferRecording()) {
            g_cmdBufferSet.active().recordSetSamplerState(a, D3DSAMP_MINFILTER, filter);
        }
        if (shouldSuppressMWState()) return D3D_OK;
        return realDevice->SetSamplerState(a, D3DSAMP_MINFILTER, filter);
    } else if (b == D3DTSS_MIPFILTER) {
        DWORD filter = (c != D3DTEXF_NONE) ? D3DTEXF_LINEAR : D3DTEXF_NONE;
        if (ImGuiManager::GetCmdBufferRecording()) {
            g_cmdBufferSet.active().recordSetSamplerState(a, D3DSAMP_MIPFILTER, filter);
        }
        if (shouldSuppressMWState()) return D3D_OK;
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
            g_cmdBufferSet.active().recordSetSamplerState(a, sampler, c);
        } else {
            g_cmdBufferSet.active().recordSetTextureStageState(a, (DWORD)b, c);
        }
    }
    if (shouldSuppressMWState()) return D3D_OK;
    return ProxyDevice::SetTextureStageState(a, b, c);
}

// DrawIndexedPrimitive - Where all the drawing happens
// Inspect draw calls for re-use later
HRESULT _stdcall MGEProxyDevice::DrawIndexedPrimitive(D3DPRIMITIVETYPE a, UINT b, UINT c, UINT d, UINT e) {
    // Allow distant land to inspect draw calls
    bool isShadowStencil = isStencilScene && stencilRef <= 1;

    // Categorize this DIP for Tracy profiling and per-bin counting
    // Scene 0 mainview non-stencil calls are NOT logged here — they get classified
    // more precisely downstream (sky/water/HLSL bins via inspectIndexedPrimitive)
    const char* dipCategory;
    static thread_local char dipBuf[32];
    FrameEvent::Type dipEventType = FrameEvent::Count;  // Count = "don't log yet"
    bool deferEventLog = false;
    if (!rendertargetNormal) {
        snprintf(dipBuf, sizeof(dipBuf), "DIP_Offscreen_S%d", sceneCount);
        dipCategory = dipBuf;
        dipEventType = FrameEvent::DIP_Offscreen;
        g_offscreenDIPs++;
        g_dipBinStats.offscreen++;
    } else if (!isMainView) {
        dipCategory = "DIP_UI";
        dipEventType = FrameEvent::DIP_UI;
        g_dipBinStats.ui++;
    } else if (isShadowStencil) {
        dipCategory = "DIP_StencilShadow";
        dipEventType = FrameEvent::DIP_StencilShadow;
        g_dipBinStats.stencilShadow++;
    } else if (sceneCount == 0) {
        dipCategory = "DIP_Scene0";
        g_dipScene0++;
        deferEventLog = true;  // Classified downstream by inspect/water/HLSL replay
    } else if (sceneCount > 0) {
        snprintf(dipBuf, sizeof(dipBuf), "DIP_Scene%d", sceneCount);
        dipCategory = dipBuf;
        g_dipScene1plus++;
        // Sub-classify Scene 1+ by render state
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
        ImGuiManager::LogFrameEvent(dipEventType, sceneCount, e);
    }

    // Detailed trace for DIP — log regardless of deferral (trace wants ALL calls)
    if (ImGuiManager::GetTraceEnabled()) {
        FrameEvent::Type traceType = deferEventLog ? FrameEvent::DIP_Opaque : dipEventType;  // placeholder bin for deferred
        ImGuiManager::TraceDIP(sceneCount, traceType, rs.fvf, rs.vb, rs.ib, rs.texture,
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
    if (!rendertargetNormal && (g_suppressingCurrentScene || ImGuiManager::GetSuppressOffscreen())) return D3D_OK;

    // Debug: suppress other DIP categories via ImGui toggles
    if (!isMainView && rendertargetNormal && ImGuiManager::GetSuppressUI()) return D3D_OK;
    if (isShadowStencil && ImGuiManager::GetSuppressStencilShadow()) return D3D_OK;
    if (sceneCount < 0 && rendertargetNormal && ImGuiManager::GetSuppressPreScene()) return D3D_OK;
    // Scene 1+ per-subcategory suppress
    if (sceneCount > 0 && rendertargetNormal && isMainView && !isShadowStencil) {
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

    if (DistantLand::ready && rendertargetNormal && isMainView && !isShadowStencil) {
        rs.primType = a;
        rs.baseIndex = baseVertexIndex;
        rs.minIndex = b;
        rs.vertCount = c;
        rs.startIndex = d;
        rs.primCount = e;

        if (!stage0Complete && !isAmbientWhite) {
            // At this point, only the sky is rendered in exteriors, or nothing in interiors
            if (isHLSLActive()) {
                // HLSL: CPU-only context capture, GPU work deferred to finalizeAndRender
                frameCtx = DistantLand::captureStage0Context();
            } else {
                // Legacy: interleaved GPU work (distant land renders now)
                frameCtx = DistantLand::renderStage0();
            }
            stage0Complete = true;
        }

        if (isWaterMaterial) {
            g_dipBinStats.water++;
            ImGuiManager::LogFrameEvent(FrameEvent::DIP_Water, sceneCount, e);
            if (ImGuiManager::GetSuppressWater()) {
                // Render wireframe outline to show water mesh location
                realDevice->SetRenderState(D3DRS_FILLMODE, D3DFILL_WIREFRAME);
                HRESULT hr = ProxyDevice::DrawIndexedPrimitive(a, b, c, d, e);
                realDevice->SetRenderState(D3DRS_FILLMODE, D3DFILL_SOLID);
                return hr;
            }
            if (distantWater) {
                if (!waterDrawn) {
                    if (isHLSLActive()) {
                        // HLSL: just flag — finalizeAndRender will render water in GPU phase
                        waterDrawn = true;
                    } else {
                        // Legacy: render water immediately
                        DistantLand::renderStageWater(&frameCtx);
                        waterDrawn = true;
                    }
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

    // Log deferred Scene 0 calls that fell through without specific classification
    // (e.g. fixed-function path, sky without atmosphere scatter)
    if (deferEventLog) {
        ImGuiManager::LogFrameEvent(FrameEvent::DIP_Opaque, sceneCount, e);
    }

    // Suppress scene draws in replay mode — replayed via executeGpuPhase HLSL pipeline.
    // UI draws (!isMainView) go direct to device for correct hit-testing and inherited state.
    if (isMainView && isHLSLActive() && ImGuiManager::GetCmdBufferReplay()) return D3D_OK;
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

// --------------------------------------------------------
// State recording

HRESULT _stdcall MGEProxyDevice::SetTexture(DWORD a, IDirect3DBaseTexture8* b) {
    if (a == 0) {
        rs.texture = b ? static_cast<ProxyTexture*>(b)->realTexture : NULL;
    }
    ImGuiManager::TraceTexture(sceneCount, a, rs.texture);
    if (ImGuiManager::GetCmdBufferRecording()) {
        IDirect3DBaseTexture9* realTex = b ? static_cast<ProxyTexture*>(b)->realTexture : nullptr;
        g_cmdBufferSet.active().recordSetTexture(a, realTex);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    return ProxyDevice::SetTexture(a, b);
}

// Draw method overrides — ensure deferred BeginScene is forwarded before any draw
// Suppress offscreen draws (same logic as DrawIndexedPrimitive)
HRESULT _stdcall MGEProxyDevice::DrawPrimitive(D3DPRIMITIVETYPE a, UINT b, UINT c) {
    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordDrawPrimitive(a, b, c);
    }
    if (!rendertargetNormal && (g_suppressingCurrentScene || ImGuiManager::GetSuppressOffscreen())) return D3D_OK;
    ensureSceneActive();
    if (isMainView && isHLSLActive() && ImGuiManager::GetCmdBufferReplay()) return D3D_OK;
    return ProxyDevice::DrawPrimitive(a, b, c);
}

HRESULT _stdcall MGEProxyDevice::DrawPrimitiveUP(D3DPRIMITIVETYPE a, UINT b, const void* c, UINT d) {
    if (!rendertargetNormal && (g_suppressingCurrentScene || ImGuiManager::GetSuppressOffscreen())) return D3D_OK;
    ensureSceneActive();
    return ProxyDevice::DrawPrimitiveUP(a, b, c, d);
}

HRESULT _stdcall MGEProxyDevice::DrawIndexedPrimitiveUP(D3DPRIMITIVETYPE a, UINT b, UINT c, UINT d, const void* e, D3DFORMAT f, const void* g, UINT h) {
    if (!rendertargetNormal && (g_suppressingCurrentScene || ImGuiManager::GetSuppressOffscreen())) return D3D_OK;
    ensureSceneActive();
    return ProxyDevice::DrawIndexedPrimitiveUP(a, b, c, d, e, f, g, h);
}

HRESULT _stdcall MGEProxyDevice::SetVertexShader(DWORD a) {
    rs.fvf = a;
    ImGuiManager::TraceVertexShader(sceneCount, a);
    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordSetFVF(a);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    return ProxyDevice::SetVertexShader(a);
}

HRESULT _stdcall MGEProxyDevice::SetStreamSource(UINT a, IDirect3DVertexBuffer8* b, UINT c) {
    if (a == 0) {
        rs.vb = (IDirect3DVertexBuffer9*)b;
        rs.vbOffset = 0;
        rs.vbStride = c;
    }
    ImGuiManager::TraceStreamSource(sceneCount, a, b, c);
    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordSetStreamSource(a, (IDirect3DVertexBuffer9*)b, 0, c);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    return ProxyDevice::SetStreamSource(a, b, c);
}

HRESULT _stdcall MGEProxyDevice::SetIndices(IDirect3DIndexBuffer8* a, UINT b) {
    rs.ib = (IDirect3DIndexBuffer9*)a;
    ImGuiManager::TraceIndexBuffer(sceneCount, a);
    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordSetIndices((IDirect3DIndexBuffer9*)a);
    }
    if (shouldSuppressMWState()) return D3D_OK;
    return ProxyDevice::SetIndices(a, b);
}

HRESULT _stdcall MGEProxyDevice::LightEnable(DWORD a, BOOL b) {
    ImGuiManager::TraceLight(sceneCount, a, b != 0);
    if (b) {
        if (std::find(lightrs.active.begin(), lightrs.active.end(), a) == lightrs.active.end()) {
            lightrs.active.push_back(a);
        }
    } else {
        if (std::remove(lightrs.active.begin(), lightrs.active.end(), a) != lightrs.active.end()) {
            lightrs.active.pop_back();
        }
    }
    if (ImGuiManager::GetCmdBufferRecording()) {
        g_cmdBufferSet.active().recordLightEnable(a, b);
    }
    if (shouldSuppressMWState()) return D3D_OK;
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
// Trace-only overrides — forward to ProxyDevice, log when trace enabled

HRESULT _stdcall MGEProxyDevice::SetViewport(const D3DVIEWPORT8* a) {
    if (a) {
        ImGuiManager::TraceViewport(sceneCount, a->X, a->Y, a->Width, a->Height, a->MinZ, a->MaxZ);
        if (ImGuiManager::GetCmdBufferRecording()) {
            g_cmdBufferSet.active().recordSetViewport(a);
        }
    }
    if (shouldSuppressMWState()) return D3D_OK;
    return ProxyDevice::SetViewport(a);
}

HRESULT _stdcall MGEProxyDevice::SetClipPlane(DWORD a, const float* b) {
    ImGuiManager::TraceClipPlane(sceneCount, a);
    return ProxyDevice::SetClipPlane(a, b);
}

HRESULT _stdcall MGEProxyDevice::MultiplyTransform(D3DTRANSFORMSTATETYPE a, const D3DMATRIX* b) {
    ImGuiManager::TraceMultiplyTransform(sceneCount, (DWORD)a, b ? (const float*)b : nullptr);
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
