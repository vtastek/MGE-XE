#pragma once

#include "proxydx/d3d9header.h"
#include "imgui_manager.h"

// Scene phase — detected by characteristics, not position
enum class ScenePhase {
    Unknown,
    Offscreen,   // !rendertargetNormal (local map, inventory doll)
    World,       // isMainView=true, first main scene
    Particles,   // After world EndScene, before Z-clear
    Hands,       // After Z-clear, skinned draws
    UI           // detectMenu() true
};

// Draw type classification — for per-draw analysis
enum class DrawType {
    Unknown,
    Sky,         // First blended in outdoor, before opaques
    Water,       // Power == 99999.0f magic marker
    Terrain,     // Land splat pattern
    Opaque,      // zWrite, no blend
    AlphaTested, // zWrite, alpha test
    Blended,     // blendEnable, no zWrite (particles)
    Skinned,     // vertexBlendState != 0 (hands)
    Decal        // texcoordIndex != 0
};

// Scene state — groups scattered statics from mged3d8device.cpp
struct SceneState {
    int sceneCount = -1;              // Legacy counter (kept for compatibility)
    ScenePhase phase = ScenePhase::Unknown;  // Characteristic-based phase
    bool rendertargetNormal = true;
    bool isHUDready = false;
    bool isMainView = false;
    bool isStencilScene = false;
    bool isAmbientWhite = false;
    bool stage0Complete = false;
    bool isFrameComplete = false;
    bool isHUDComplete = false;
    bool isWaterMaterial = false;
    bool waterDrawn = false;
    bool distantWater = false;
    DWORD stencilRef = 0;

    // Characteristic-based detection state
    bool worldComplete = false;        // Set at EndScene of World
    bool hadZClearSinceWorld = false;  // Set in Clear() after world
    bool skyDrawn = false;             // First sky draw detected
    bool handsStarted = false;         // Hands scene started
    DrawType lastDrawType = DrawType::Unknown;

    // Scene emptiness tracking (set at finalize time)
    bool scene0Empty = true;
    bool scene1Empty = true;
    bool scene2Empty = true;
    bool uiStateCaptured = false;

    void resetForFrame() {
        sceneCount = -1;
        phase = ScenePhase::Unknown;
        stage0Complete = false;
        waterDrawn = false;
        isFrameComplete = false;
        isHUDComplete = false;
        // Reset characteristic detection state
        worldComplete = false;
        hadZClearSinceWorld = false;
        skyDrawn = false;
        handsStarted = false;
        lastDrawType = DrawType::Unknown;
        // Reset scene emptiness tracking
        scene0Empty = true;
        scene1Empty = true;
        scene2Empty = true;
        uiStateCaptured = false;
    }
};

// Deferred scene state — suppresses empty BeginScene/EndScene pairs
struct DeferredSceneState {
    bool scenePending = false;
    bool sceneForwarded = false;
    bool rtPending = false;
    IDirect3DSurface8* pendingRT_color = nullptr;
    IDirect3DSurface8* pendingRT_depth = nullptr;

    void reset() {
        scenePending = false;
        sceneForwarded = false;
        rtPending = false;
    }
};

// Deferral instrumentation — per-frame counters
struct DeferralCounters {
    int scenesRequested = 0, scenesForwarded = 0, scenesSuppressed = 0;
    int rtRequested = 0, rtForwarded = 0, rtSuppressed = 0;
    int copyRectsTotal = 0, copyRectsSuppressed = 0;

    void reset() {
        scenesRequested = scenesForwarded = scenesSuppressed = 0;
        rtRequested = rtForwarded = rtSuppressed = 0;
        copyRectsTotal = copyRectsSuppressed = 0;
    }
};

// Offscreen scene state
struct OffscreenState {
    int scenesThisFrame = 0;
    bool suppressingCurrentScene = false;
    bool megaSceneOpen = false;
    LARGE_INTEGER startQPC = {};
    bool timingActive = false;
    int dipCount = 0;
    int sceneCount = 0;

    void resetForFrame() {
        scenesThisFrame = 0;
        suppressingCurrentScene = false;
        // megaSceneOpen is closed explicitly, not reset
    }

    void startTiming() {
        if (!timingActive) {
            QueryPerformanceCounter(&startQPC);
            timingActive = true;
            dipCount = 0;
            sceneCount = 0;
        }
    }

    float stopTimingMs() {
        if (!timingActive) return 0.0f;
        LARGE_INTEGER now, freq;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&now);
        timingActive = false;
        return (float)((now.QuadPart - startQPC.QuadPart) * 1000.0 / freq.QuadPart);
    }
};

// DIP category result
struct DIPCategory {
    FrameEvent::Type eventType;
    const char* tracyName;
    bool deferEventLog;  // true = classified downstream (Scene0 main view)
};

// Categorize a DrawIndexedPrimitive call
// Scene 0 = world, Scene 1 = particles (alpha sorted), Scene 2 = hands (skinned)
inline DIPCategory categorizeDIP(
    bool rendertargetNormal, bool isMainView, bool isShadowStencil,
    int sceneCount, DWORD vertexBlendState, bool blendEnable,
    ImGuiManager::DIPBinStats& stats, int& dipScene0, int& dipScene1, int& dipScene2)
{
    DIPCategory result = { FrameEvent::Count, nullptr, false };
    static thread_local char dipBuf[32];

    if (!rendertargetNormal) {
        snprintf(dipBuf, sizeof(dipBuf), "DIP_Offscreen_S%d", sceneCount);
        result.tracyName = dipBuf;
        result.eventType = FrameEvent::DIP_Offscreen;
        stats.offscreen++;
    } else if (!isMainView) {
        result.tracyName = "DIP_UI";
        result.eventType = FrameEvent::DIP_UI;
        stats.ui++;
    } else if (isShadowStencil) {
        result.tracyName = "DIP_StencilShadow";
        result.eventType = FrameEvent::DIP_StencilShadow;
        stats.stencilShadow++;
    } else if (sceneCount == 0) {
        result.tracyName = "DIP_Scene0";
        dipScene0++;
        result.deferEventLog = true;
    } else if (sceneCount == 1) {
        result.tracyName = "DIP_Scene1";
        dipScene1++;
        // Scene 1 = particles (alpha sorted, no depth write)
        if (blendEnable) {
            result.eventType = FrameEvent::DIP_1P_Alpha;
            stats.firstPersonAlpha++;
        } else {
            result.eventType = FrameEvent::DIP_1P_Other;
            stats.firstPersonOther++;
        }
    } else if (sceneCount >= 2) {
        snprintf(dipBuf, sizeof(dipBuf), "DIP_Scene%d", sceneCount);
        result.tracyName = dipBuf;
        dipScene2++;
        // Scene 2 = hands (skinned, solid depth after Z-clear)
        if (vertexBlendState != 0) {
            result.eventType = FrameEvent::DIP_1P_Skinning;
            stats.firstPersonSkinning++;
        } else if (blendEnable) {
            result.eventType = FrameEvent::DIP_1P_Alpha;
            stats.firstPersonAlpha++;
        } else {
            result.eventType = FrameEvent::DIP_1P_Other;
            stats.firstPersonOther++;
        }
    } else {
        result.tracyName = "DIP_PreScene";
        result.eventType = FrameEvent::DIP_PreScene;
        stats.preScene++;
    }

    return result;
}

// Complete device state snapshot for async replay (no assumptions about prior state)
// Captured per-draw-call to enable correct replay from any starting device state.
struct DeviceStateSnapshot {
    // Depth
    DWORD zEnable = D3DZB_TRUE;
    DWORD zWriteEnable = TRUE;
    DWORD zFunc = D3DCMP_LESSEQUAL;
    float depthBias = 0.0f;
    float slopeScaleDepthBias = 0.0f;

    // Culling
    DWORD cullMode = D3DCULL_CW;

    // Blending
    DWORD alphaBlendEnable = FALSE;
    DWORD srcBlend = D3DBLEND_ONE;
    DWORD destBlend = D3DBLEND_ZERO;

    // Alpha test
    DWORD alphaTestEnable = FALSE;
    DWORD alphaFunc = D3DCMP_ALWAYS;
    DWORD alphaRef = 0;

    // Lighting/Material
    DWORD lighting = TRUE;
    DWORD specularEnable = FALSE;
    DWORD localViewer = FALSE;
    DWORD normalizeNormals = FALSE;
    DWORD diffuseMatSrc = D3DMCS_COLOR1;
    DWORD emissiveMatSrc = D3DMCS_MATERIAL;
    DWORD ambientMatSrc = D3DMCS_MATERIAL;
    DWORD colorVertex = TRUE;
    DWORD vertexBlend = D3DVBF_DISABLE;

    // Fog
    DWORD fogEnable = FALSE;

    // Output
    DWORD colorWriteEnable = 0xF;

    // Stencil
    DWORD stencilEnable = FALSE;

    // UI-specific
    DWORD ambient = 0;
    DWORD textureFactor = 0xFFFFFFFF;

    // Sampler address modes used by MW texture stages. DX8 exposes these through
    // SetTextureStageState, but HLSL replay needs the corresponding D3D9 values.
    DWORD samplerAddressU[8] = {
        D3DTADDRESS_WRAP, D3DTADDRESS_WRAP, D3DTADDRESS_WRAP, D3DTADDRESS_WRAP,
        D3DTADDRESS_WRAP, D3DTADDRESS_WRAP, D3DTADDRESS_WRAP, D3DTADDRESS_WRAP
    };
    DWORD samplerAddressV[8] = {
        D3DTADDRESS_WRAP, D3DTADDRESS_WRAP, D3DTADDRESS_WRAP, D3DTADDRESS_WRAP,
        D3DTADDRESS_WRAP, D3DTADDRESS_WRAP, D3DTADDRESS_WRAP, D3DTADDRESS_WRAP
    };

    // Clip planes
    DWORD clipPlaneEnable = 0;

    // Debug
    DWORD fillMode = D3DFILL_SOLID;

    // Point sprites (particles)
    float pointSize = 1.0f;
    DWORD pointSpriteEnable = FALSE;
    DWORD pointScaleEnable = FALSE;
    float pointScaleA = 1.0f;
    float pointScaleB = 0.0f;
    float pointScaleC = 0.0f;
};

extern DeviceStateSnapshot g_deviceState;

// PipelineDiag is defined in ffeshader.h
