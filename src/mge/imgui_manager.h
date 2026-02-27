#pragma once

#include "imgui.h"
#include "imgui_impl_dx9.h"
#include "imgui_impl_win32.h"
#include "proxydx/d3d9header.h"

#include <vector>

// Frame event log entry — lightweight record of what happened this frame and in what order
struct FrameEvent {
    enum Type : uint8_t {
        Clear, BeginScene, EndScene, SetRenderTarget,
        DIP_Sky, DIP_Terrain, DIP_Opaque, DIP_Skinning, DIP_Grass,
        DIP_AlphaTested, DIP_Blending,
        DIP_1P_Skinning, DIP_1P_Alpha, DIP_1P_Other,
        DIP_Offscreen, DIP_UI, DIP_StencilShadow, DIP_PreScene, DIP_Water,
        MGE_Stage0, MGE_Stage1, MGE_StageBlend, MGE_Stage2,
        MGE_PostProcess, MGE_HLSLReplay,
        // MGE internal rendering (summarized, primCount = draw call count)
        MGE_ShadowMap, MGE_DistantLand, MGE_DistantStatics,
        MGE_Grass, MGE_GrassZ,
        MGE_ShadowOverlay, MGE_Depth, MGE_DepthDistant,
        MGE_HiZGen, MGE_WaterReflection, MGE_WaterPlane,
        MGE_SkyRender,
        Count
    };
    Type type;
    int sceneNum;       // which scene this occurred in
    int primCount;      // for DIP events: prim count. For MGE internal: draw call count

    static const char* typeName(Type t) {
        static const char* names[] = {
            "Clear", "BeginScene", "EndScene", "SetRenderTarget",
            "DIP_Sky", "DIP_Terrain", "DIP_Opaque", "DIP_Skinning", "DIP_Grass",
            "DIP_AlphaTested", "DIP_Blending",
            "DIP_1P_Skinning", "DIP_1P_Alpha", "DIP_1P_Other",
            "DIP_Offscreen", "DIP_UI", "DIP_StencilShadow", "DIP_PreScene", "DIP_Water",
            "MGE_Stage0", "MGE_Stage1", "MGE_StageBlend", "MGE_Stage2",
            "MGE_PostProcess", "MGE_HLSLReplay",
            "MGE_ShadowMap", "MGE_DistantLand", "MGE_DistantStatics",
            "MGE_Grass", "MGE_GrassZ",
            "MGE_ShadowOverlay", "MGE_Depth", "MGE_DepthDistant",
            "MGE_HiZGen", "MGE_WaterReflection", "MGE_WaterPlane",
            "MGE_SkyRender"
        };
        return (t < Count) ? names[t] : "Unknown";
    }
};

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
    static int GetBBoxVisualizationMode();
    static void UpdateDebugStats(int recordedCalls, int renderedCalls, int culledCalls,
                                  int sceneLights, int recordMWSize, int immediateCount);

    // Hi-Z culling bypass for terrain hole diagnosis
    static bool GetDisableHiZCulling();

    // Hi-Z visualization interface
    static void RenderHiZInterface();
    static void ToggleHiZInterface();
    static bool GetShowHiZInterface();
    static int GetHiZDisplayMip() { return hiZDisplayMip; }
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

    // Render bin highlighting (debug visualization)
    static bool GetHighlightBins() { return highlightBins; }

    // Rasterize All mode - bypass all heuristics, rasterize everything
    static bool GetRasterizeAll() { return rasterizeAll; }

    // Debug/Performance mode toggles
    static bool GetStateLeakDetection() { return stateLeakDetection; }
    static bool GetPerformanceMode() { return performanceMode; }

    // Per-bin DIP suppression toggles
    static bool GetSuppressSky() { return suppressSky; }
    static bool GetSuppressTerrain() { return suppressTerrain; }
    static bool GetSuppressOpaque() { return suppressOpaque; }
    static bool GetSuppressSkinning() { return suppressSkinning; }
    static bool GetSuppressGrass() { return suppressGrass; }
    static bool GetSuppressAlphaTested() { return suppressAlphaTested; }
    static bool GetSuppressBlending() { return suppressBlending; }
    static bool GetSuppress1PSkinning() { return suppress1PSkinning; }
    static bool GetSuppress1PAlpha() { return suppress1PAlpha; }
    static bool GetSuppress1POther() { return suppress1POther; }
    static bool GetSuppressOffscreen() { return suppressOffscreen; }
    static int GetOffscreenBudget() { return offscreenBudget; }
    static bool GetSuppressUI() { return suppressUI; }
    static bool GetSuppressStencilShadow() { return suppressStencilShadow; }
    static bool GetSuppressPreScene() { return suppressPreScene; }
    static bool GetSuppressWater() { return suppressWater; }

    // Per-bin DIP counter stats
    struct DIPBinStats {
        int sky, terrain, opaque, skinning, grass, alphaTested, blending;
        int firstPersonSkinning, firstPersonAlpha, firstPersonOther;
        int offscreen, ui, stencilShadow, preScene, water;
        int total() const { return sky + terrain + opaque + skinning + grass + alphaTested + blending + firstPersonSkinning + firstPersonAlpha + firstPersonOther + offscreen + ui + stencilShadow + preScene + water; }
        void reset() { memset(this, 0, sizeof(*this)); }
    };
    static void UpdateDIPStats(const DIPBinStats& stats);

    // DIP spike freeze
    static bool GetDIPFrozen() { return dipFrozen; }
    static bool GetDIPAutoFreeze() { return dipAutoFreeze; }
    static int GetDIPSpikeThreshold() { return dipSpikeThreshold; }
    static void FreezeDIPStats(const DIPBinStats& stats);

    // Per-bin counts from HLSL replay (called from ffeshader.cpp after replay loop)
    static void UpdateReplayBinCounts(int terrain, int opaque, int skinning, int grass, int alphaTested, int blending);
    static void IncrementSkyStat() { dipStats.sky++; }

    // Frame event log
    static void LogFrameEvent(FrameEvent::Type type, int sceneNum, int primCount = 0);
    static void SnapshotFrameEvents();  // Call at Present() to snapshot for display
    static void RenderFrameEventLog();
    static void ToggleFrameEventLog();
    static bool GetShowFrameEventLog() { return showFrameEventLog; }
    static void PrintFrameEventsToFile();  // Dump current display events to file

    // Slow frame detection
    static float GetSlowFrameThreshold() { return slowFrameThreshold; }
    static float GetSlowCallThreshold() { return slowCallThreshold; }
    static bool GetSlowFrameAutoFreeze() { return slowFrameAutoFreeze; }
    static bool GetSlowFrameFrozen() { return slowFrameFrozen; }
    static void FreezeSlowFrame(float prepareMs, float replayMs, int worstCallIndex, float worstCallMs, int worstCallPrims, int worstCallBin);

    // Debug hotkey gating
    static bool GetDebugKeysEnabled();

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
    static int bboxVisualizationMode;  // 0=off, 1=objects, 2=lights
    static bool disableHiZCulling;     // Bypass Hi-Z culling for terrain hole diagnosis

    // Debug stats (updated each frame)
    static int debugRecordedCalls;
    static int debugRenderedCalls;
    static int debugCulledCalls;
    static int debugSceneLights;
    static int debugRecordMWSize;
    static int debugImmediateCount;

    // Hi-Z visualization variables
    static int hiZDisplayMip;
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

    // Render bin highlighting (debug visualization)
    static bool highlightBins;            // Tint draw calls by render bin for debugging - default false

    // Rasterize All mode - bypass all heuristics
    static bool rasterizeAll;             // Rasterize ALL objects to Hi-Z - default false

    // Debug/Performance mode toggles
    static bool stateLeakDetection;       // State leak detection debug mode - default false
    static bool performanceMode;          // Dirty tracking performance mode - default true

    // Per-bin DIP suppression toggles
    static bool suppressSky;
    static bool suppressTerrain;
    static bool suppressOpaque;
    static bool suppressSkinning;
    static bool suppressGrass;
    static bool suppressAlphaTested;
    static bool suppressBlending;
    static bool suppress1PSkinning;
    static bool suppress1PAlpha;
    static bool suppress1POther;
    static bool suppressOffscreen;
    static int offscreenBudget;
    static bool suppressUI;
    static bool suppressStencilShadow;
    static bool suppressPreScene;
    static bool suppressWater;

    // Per-bin DIP counter stats
    static DIPBinStats dipStats;

    // DIP spike freeze state
    static bool dipFrozen;
    static bool dipAutoFreeze;
    static int dipSpikeThreshold;
    static DIPBinStats frozenDipStats;
    static int frozenTotal;

    // Frame event log
    static bool showFrameEventLog;
    static std::vector<FrameEvent> frameEvents;          // Current frame accumulator
    static std::vector<FrameEvent> displayFrameEvents;   // Snapshot for ImGui display (live or frozen)
    static std::vector<FrameEvent> frozenFrameEvents;    // Frozen snapshot
    static bool eventLogFrozen;
    static bool eventLogAutoFreeze;
    static int eventLogFreezeOffscreenThreshold;  // Auto-freeze when offscreen DIPs >= this (0=disabled)

    // Slow frame detection state
    static bool slowFrameFrozen;
    static bool slowFrameAutoFreeze;
    static float slowFrameThreshold;      // Frame-level threshold (ms) - default 5.0
    static float slowCallThreshold;       // Per-call threshold (ms) - default 5.0
    static float frozenPrepareMs;
    static float frozenReplayMs;
    static int frozenSlowCallIndex;
    static float frozenSlowCallMs;
    static int frozenSlowCallPrims;
    static int frozenSlowCallBin;

    // Debug hotkey gating
    static bool debugKeysEnabled;         // Gate debug hotkeys (F11/U/Y/L/F5/F6) - default false
};