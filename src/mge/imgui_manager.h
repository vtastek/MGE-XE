#pragma once

#include "imgui.h"
#include "imgui_impl_dx9.h"
#include "imgui_impl_win32.h"
#include "proxydx/d3d9header.h"

class ImGuiManager {
public:
    static bool Initialize(HWND hwnd, IDirect3DDevice9* device);
    static void Shutdown();
    static void NewFrame();
    static void Render();
    static bool WantCaptureMouse();
    static bool WantCaptureKeyboard();
    static void RenderPCFFilteringInterface();
    static void RenderDebugInterface();
    static LRESULT HandleWindowMessage(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    static void OnLostDevice();
    static void OnResetDevice();

    // Getters for PCF parameters
    static float GetPCFFilterSize();
    static float GetPCFPenumbraScale();
    static float GetPCFMinPenumbra();
    static float GetPCFMaxPenumbra();
    static float GetPCFBias();
    static float GetPCFBias2();
    static float GetPCFSlopeBias();
    static bool GetShowPCFInterface();
    static void TogglePCFInterface();

    // Debug interface controls
    static void ToggleDebugInterface();
    static bool GetEnableRecording();
    static bool GetEnableReplay();
    static bool GetEnableImmediateRendering();
    static bool GetEnableDepthPass();
    static bool GetEnableLightProcessing();
    static int GetBBoxVisualizationMode();
    static void UpdateDebugStats(int recordedCalls, int renderedCalls, int culledCalls,
                                  int sceneLights, int visibleLights, int recordMWSize, int immediateCount);

    // Hi-Z visualization interface
    static void RenderHiZInterface();
    static void ToggleHiZInterface();
    static bool GetShowHiZInterface();
    static int GetHiZDisplayMip() { return hiZDisplayMip; }
    static float GetHiZBrightness() { return hiZBrightness; }
    static float GetHiZGamma() { return hiZGamma; }
    static bool GetHiZInvert() { return hiZInvert; }
    static bool GetHiZShowRaycastGrid() { return hiZShowRaycastGrid; }
    static int GetHiZRaycastStep() { return hiZRaycastStep; }

    // Hi-Z occluder selection parameters
    static int GetOccluderMaxCount() { return occluderMaxCount; }
    static int GetOccluderP0ExtraBudget() { return occluderP0ExtraBudget; }
    static int GetOccluderP1ExtraBudget() { return occluderP1ExtraBudget; }
    static int GetOccluderP2ExtraBudget() { return occluderP2ExtraBudget; }
    static int GetOccluderMinTriangles() { return occluderMinTriangles; }
    static int GetOccluderMaxTriangles() { return occluderMaxTriangles; }
    static float GetOccluderCloseDistance() { return occluderCloseDistance; }

    // Wall detection heuristics (shape-based occluder selection)
    static bool GetWallDetectionEnabled() { return wallDetectionEnabled; }
    static float GetWallFlatnessThreshold() { return wallFlatnessThreshold; }
    static float GetWallMinLargeDim() { return wallMinLargeDim; }
    static float GetWallMaxThinDim() { return wallMaxThinDim; }
    static int GetOccluderWallExtraBudget() { return occluderWallExtraBudget; }

    // Occluder highlighting (debug visualization)
    static bool GetHighlightOccluders() { return highlightOccluders; }

    // Rasterize All mode - bypass all heuristics, rasterize everything
    static bool GetRasterizeAll() { return rasterizeAll; }

    // Debug/Performance mode toggles
    static bool GetStateLeakDetection() { return stateLeakDetection; }
    static bool GetPerformanceMode() { return performanceMode; }

    // Hi-Z single object visualization mode (public for ffeshader access)
    static bool hiZSingleObjectMode;
    static int hiZSingleObjectIndex;
    static int hiZTotalObjectCount;

private:
    static bool initialized;
    static bool showDemo;
    static bool showPCFInterface;
    static bool showDebugInterface;
    static bool showHiZInterface;
    static HWND windowHandle;

    // PCF filtering variables for tweaking
    static float pcfFilterSize;        // Base filter size in texels (maps to PCF_filterSize)
    static float pcfPenumbraScale;     // Scale factor for distance-based penumbra
    static float pcfMinPenumbra;       // Minimum penumbra size
    static float pcfMaxPenumbra;       // Maximum penumbra size
    static float pcfBias;              // Depth bias to prevent acne
    static float pcfBias2;             // Second depth bias for lerp
    static float pcfSlopeBias;         // Slope-based bias to prevent acne on angled surfaces

    // Debug interface controls
    static bool enableRecording;
    static bool enableReplay;
    static bool enableImmediateRendering;
    static bool enableDepthPass;
    static bool enableLightProcessing;
    static int bboxVisualizationMode;  // 0=off, 1=objects, 2=lights

    // Debug stats (updated each frame)
    static int debugRecordedCalls;
    static int debugRenderedCalls;
    static int debugCulledCalls;
    static int debugSceneLights;
    static int debugVisibleLights;
    static int debugRecordMWSize;
    static int debugImmediateCount;

    // Hi-Z visualization variables
    static int hiZDisplayMip;
    static float hiZBrightness;
    static float hiZGamma;
    static bool hiZInvert;
    static bool hiZShowRaycastGrid;
    static int hiZRaycastStep;

    // Hi-Z occluder selection parameters
    static int occluderMaxCount;          // Base budget (default 100)
    static int occluderP0ExtraBudget;     // Extra budget for P0 (camera inside bbox) - default 80
    static int occluderP1ExtraBudget;     // Extra budget for P1 (off-screen corners) - default 50
    static int occluderP2ExtraBudget;     // Extra budget for P2 (very close) - default 30
    static int occluderMinTriangles;      // Minimum triangle count to be considered - default 0
    static int occluderMaxTriangles;      // Maximum triangle count (filter out too complex) - default 9999999
    static float occluderCloseDistance;   // Distance threshold for "very close" (P2) - default 2048

    // Wall detection heuristics (shape-based occluder selection)
    static bool wallDetectionEnabled;     // Enable wall shape detection - default true
    static float wallFlatnessThreshold;   // Ratio of thin dim to mid dim (e.g., 0.15 = thin < 15% of mid) - default 0.15
    static float wallMinLargeDim;         // Min size of the largest dimension to qualify as wall - default 200
    static float wallMaxThinDim;          // Max size of thin dimension to qualify as wall - default 50
    static int occluderWallExtraBudget;   // Extra budget for wall-shaped occluders - default 100

    // Occluder highlighting (debug visualization)
    static bool highlightOccluders;       // Tint occluders green for debugging - default false

    // Rasterize All mode - bypass all heuristics
    static bool rasterizeAll;             // Rasterize ALL objects to Hi-Z - default false

    // Debug/Performance mode toggles
    static bool stateLeakDetection;       // State leak detection debug mode - default false
    static bool performanceMode;          // Dirty tracking performance mode - default true
};