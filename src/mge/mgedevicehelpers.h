#pragma once

#include "proxydx/d3d9header.h"
#include "imgui_manager.h"

// Scene state — groups scattered statics from mged3d8device.cpp
struct SceneState {
    int sceneCount = -1;
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

    void resetForFrame() {
        sceneCount = -1;
        stage0Complete = false;
        waterDrawn = false;
        isFrameComplete = false;
        isHUDComplete = false;
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
inline DIPCategory categorizeDIP(
    bool rendertargetNormal, bool isMainView, bool isShadowStencil,
    int sceneCount, DWORD vertexBlendState, bool blendEnable,
    ImGuiManager::DIPBinStats& stats, int& dipScene0, int& dipScene1plus)
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
    } else if (sceneCount > 0) {
        snprintf(dipBuf, sizeof(dipBuf), "DIP_Scene%d", sceneCount);
        result.tracyName = dipBuf;
        dipScene1plus++;
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

// PipelineDiag is defined in ffeshader.h
