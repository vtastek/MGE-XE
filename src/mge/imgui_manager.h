#pragma once

#include "imgui.h"
#include "imgui_impl_dx9.h"
#include "imgui_impl_win32.h"
#include "proxydx/d3d9header.h"

#include <vector>

// Detailed state attached to trace-level frame events
struct StateDetail {
    enum Kind : uint8_t {
        None, RenderState, TextureStageState, Transform, Texture,
        DrawCall, ClearCall, RenderTarget, Light, Material,
        Viewport, ClipPlane, StreamSource, VertexShader, IndexBuffer
    };
    Kind kind = None;

    union {
        struct { DWORD state; DWORD value; DWORD prev; bool changed; } rs;
        struct { DWORD stage; DWORD state; DWORD value; DWORD prev; bool changed; } tss;
        struct { DWORD type; float m[16]; } xform;
        struct { DWORD stage; uintptr_t ptr; } tex;
        struct { DWORD fvf; uintptr_t vb; uintptr_t ib; uintptr_t tex0;
                 DWORD primCount; DWORD vertCount;
                 DWORD zWrite; DWORD cull; DWORD alphaBlend; DWORD alphaTest;
                 DWORD srcBlend; DWORD destBlend; DWORD vertBlend; } dip;
        struct { DWORD flags; DWORD color; float z; } clear;
        struct { uintptr_t color; uintptr_t depth; } rt;
        struct { DWORD index; bool enable; } light;
        struct { float dr; float dg; float db; float da; } mat;
        struct { DWORD x; DWORD y; DWORD w; DWORD h; float minZ; float maxZ; } vp;
        struct { DWORD index; } clip;
        struct { DWORD stream; uintptr_t vb; DWORD stride; } ss;
        struct { DWORD fvf; } vs;
        struct { uintptr_t ib; } ib;
    };
};

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
        // Command buffer replay events (logged during dumpToFrameLog)
        Replay_Clear, Replay_BeginScene, Replay_EndScene,
        Replay_SetRT, Replay_SetDS, Replay_SetViewport,
        Replay_RS, Replay_TSS, Replay_SS,
        Replay_Transform, Replay_Texture, Replay_Material,
        Replay_Light, Replay_LightEnable, Replay_FVF,
        Replay_StreamSource, Replay_Indices,
        Replay_DIP, Replay_DP,
        // Detailed trace events (only logged when trace enabled)
        State_RS, State_TSS, State_Transform, State_Texture,
        State_Light, State_Material, State_Viewport, State_ClipPlane,
        State_StreamSource, State_VertexShader, State_IndexBuffer,
        State_MultiplyTransform,
        // Scene emptiness tracking
        Scene0_Empty, Scene1_Empty, Scene2_Empty,
        UIState_Captured, UIState_Skipped,
        Count
    };
    Type type;
    int sceneNum;       // which scene this occurred in
    int primCount;      // for DIP events: prim count. For MGE internal: draw call count
    StateDetail detail; // detailed state (only filled when trace enabled)

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
            "MGE_SkyRender",
            "R_Clear", "R_BeginScene", "R_EndScene",
            "R_SetRT", "R_SetDS", "R_Viewport",
            "R_RS", "R_TSS", "R_SS",
            "R_Transform", "R_Texture", "R_Material",
            "R_Light", "R_LightEn", "R_FVF",
            "R_Stream", "R_Indices",
            "R_DIP", "R_DP",
            "RS", "TSS", "Transform", "Texture",
            "Light", "Material", "Viewport", "ClipPlane",
            "StreamSource", "VertexShader", "IndexBuffer",
            "MultiplyTransform",
            "Scene0_Empty", "Scene1_Empty", "Scene2_Empty",
            "UIState_Captured", "UIState_Skipped"
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
    // True if any ImGui item was active this frame or the previous one.
    // Consumed by the menu render cache to bypass it for one frame so debug
    // changes apply while paused. Resets after read.
    static bool ConsumeRenderInvalidation();
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
    static float GetPCFTerrainBias();
    static float GetIntensityScalar();
    static float GetAttenuationMultiplier();
    static float GetAttenuationCutoffDist();
    static float GetParallaxScale();
    static float GetParallaxBias();
    static bool GetShowPCFInterface();
    static void TogglePCFInterface();

    // Debug interface controls
    static void ToggleDebugInterface();
    static bool GetEnableRecording();       // Always returns true (feature always enabled)
    static bool GetEnableReplay();          // Always returns true (feature always enabled)
    static bool GetEnableImmediateRendering(); // Always returns true (feature always enabled)
    static bool GetEnableDepthPass();       // Always returns true (feature always enabled)
    static int GetBBoxVisualizationMode();
    static void UpdateDebugStats(int recordedCalls, int renderedCalls, int culledCalls,
                                  int sceneLights, int culledLights, int recordMWSize, int immediateCount);

    // Frame number overlay toggle
    static bool GetShowFrameNumber();

    // Hi-Z culling bypass for terrain hole diagnosis
    static bool GetDisableHiZCulling();

    // Stateless batching toggle (experimental)
    static bool GetEnableStatelessBatch();
    static bool GetAndClearDumpStatelessBatch(); // Returns true once, then clears
    static bool GetAndClearDumpLightSnapshot();  // Returns true once, then clears

    // Force lightMode 3 for all lit Opaque/Terrain (debug)
    static bool GetForceLightMode3();
    static void RequestLightSnapshotDump();

    // Shader debug visualization mode (0=off, 1-15=debug views)
    static int GetShaderDebugMode();

    // Hi-Z visualization interface
    static void RenderHiZInterface();
    static void ToggleHiZInterface();
    static bool GetShowHiZInterface();
    static int GetHiZDisplayMip() { return hiZDisplayMip; }
    static bool GetHiZInvert() { return hiZInvert; }
    static float GetBboxExpansion() { return bboxExpansion; }

    // Forward SSAO visualization interface
    static void RenderSSAOInterface();

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
    static bool GetMaterialSortEnabled() { return materialSortEnabled; }
    static bool GetInstancingEnabled() { return instancingEnabled; }

    // Phase 7/8: Near-camera landscape displacement LOD accessors.
    // CPU baker only — VTF path was dropped (too slow for the quality it bought).
    static bool GetEnableNearDisplacement() { return enableNearDisplacement; }
    static float GetDisplacementScale() { return displacementScale; }
    static float GetDisplacementGamma() { return displacementGamma; }
    static float GetDisplacementPivot() { return displacementPivot; }
    static float GetHeightBlendStrength() { return heightBlendStrength; }
    static float GetHeightBlendContrast() { return heightBlendContrast; }
    static bool GetDebugHighlightNearPatches() { return debugHighlightNearPatches; }

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
    static void LogFrameEventDetailed(FrameEvent::Type type, int sceneNum, int primCount, const StateDetail& detail);
    static void SnapshotFrameEvents();  // Call at Present() to snapshot for display
    static void RenderFrameEventLog();
    static void ToggleFrameEventLog();
    static bool GetShowFrameEventLog() { return showFrameEventLog; }
    static void PrintFrameEventsToFile();  // Dump current display events to file

    // Detailed trace
    static bool GetTraceEnabled() { return traceEnabled; }
    static void ResetStateShadow();
    static void AutoNameScenario();  // Auto-fill scenario label from game state
    // Convenience trace loggers (handle state shadow internally)
    static void TraceRS(int sceneNum, DWORD state, DWORD value);
    static void TraceTSS(int sceneNum, DWORD stage, DWORD state, DWORD value);
    static void TraceTransform(int sceneNum, DWORD type, const float* matrix);
    static void TraceMultiplyTransform(int sceneNum, DWORD type, const float* matrix);
    static void TraceTexture(int sceneNum, DWORD stage, void* ptr);
    static void TraceDIP(int sceneNum, FrameEvent::Type dipType, DWORD fvf, void* vb, void* ib, void* tex0,
                         DWORD primCount, DWORD vertCount, DWORD zWrite, DWORD cull,
                         DWORD alphaBlend, DWORD alphaTest, DWORD srcBlend, DWORD destBlend, DWORD vertBlend);
    static void TraceClear(int sceneNum, DWORD flags, DWORD color, float z);
    static void TraceRT(int sceneNum, void* color, void* depth);
    static void TraceLight(int sceneNum, DWORD index, bool enable);
    static void TraceMaterial(int sceneNum, float dr, float dg, float db, float da);
    static void TraceViewport(int sceneNum, DWORD x, DWORD y, DWORD w, DWORD h, float minZ, float maxZ);
    static void TraceClipPlane(int sceneNum, DWORD index);
    static void TraceStreamSource(int sceneNum, DWORD stream, void* vb, DWORD stride);
    static void TraceVertexShader(int sceneNum, DWORD fvf);
    static void TraceIndexBuffer(int sceneNum, void* ib);

    // Slow frame detection
    static float GetSlowFrameThreshold() { return slowFrameThreshold; }
    static float GetSlowCallThreshold() { return slowCallThreshold; }
    static bool GetSlowFrameAutoFreeze() { return slowFrameAutoFreeze; }
    static bool GetSlowFrameFrozen() { return slowFrameFrozen; }
    static void FreezeSlowFrame(float prepareMs, float replayMs, int worstCallIndex, float worstCallMs, int worstCallPrims, int worstCallBin);

    // D3D Command Buffer
    static bool GetCmdBufferRecording() { return cmdBufferRecording || cmdBufferReplay; }
    static bool GetCmdBufferReplay() { return cmdBufferReplay; }
    static bool GetStateSuppressionEnabled() { return stateSuppression || syncGpuThread || asyncGpuThread; }
    static bool GetSyncGpuThread() { return syncGpuThread; }
    static bool GetAsyncGpuThread() { return asyncGpuThread; }
    static void UpdateCmdBufferStats(int cmdCount, int sizeKB);
    static void UpdateCmdBufferPerStageStats(int preScene, int scene0, int interScene, int scene1Plus, int ui);

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
    static bool showSSAOInterface;
    static bool showVelocityInterface;
    static HWND windowHandle;

    // PCF filtering variables for tweaking
    static float pcfFilterSize;        // Base filter size in texels (maps to PCF_filterSize)
    static float pcfPenumbraScale;     // Scale factor for distance-based penumbra
    static float pcfMinPenumbra;       // Minimum penumbra size
    static float pcfMaxPenumbra;       // Maximum penumbra size
    static float pcfBias;              // Depth bias to prevent acne
    static float pcfBias2;             // Second depth bias for lerp
    static float pcfSlopeBias;         // Slope-based bias to prevent acne on angled surfaces
    static float pcfTerrainBias;       // Extra bias added only to terrain receivers (near cascade, all modes)

    // HLSL-pipeline unified-look intensity multiplier (mirrors Configuration.IntensityScalar).
    static float intensityScalar;

    // Point light attenuation parameters (debug tunable)
    static float attenuationMultiplier;   // range 1-100000, default 40000
    static float attenuationCutoffDist;   // range 1-5000, default 1000

    // Parallax mapping parameters (debug tunable)
    static float parallaxScale;           // default 0.026
    static float parallaxBias;            // default 0.3

    // Debug interface controls
    static int bboxVisualizationMode;  // 0=off, 1=objects, 2=lights
    static bool disableHiZCulling;     // Bypass Hi-Z culling for terrain hole diagnosis
    static bool enableStatelessBatch;  // Enable stateless batching (experimental)
    static bool dumpStatelessBatchDetail; // One-shot flag for detailed batch dump
    static bool dumpLightSnapshot;       // One-shot flag for per-frame light diagnostics
    static bool forceLightMode3;       // Force lightMode 3 for all lit Opaque/Terrain
    static int shaderDebugMode;        // Shader debug visualization (0=off, 1-15=debug views)

    // Debug stats (updated each frame)
    static int debugRecordedCalls;
    static int debugRenderedCalls;
    static int debugCulledCalls;
    static int debugSceneLights;
    static int debugCulledLights;
    static int debugRecordMWSize;
    static int debugImmediateCount;

    // Hi-Z visualization variables
    static int hiZDisplayMip;
    static bool hiZInvert;
    static float bboxExpansion;

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
    static bool materialSortEnabled;      // Material sorting for state change reduction - default false
    static bool instancingEnabled;        // GPU instancing for draw call reduction - default false

    // Phase 7/8: Near-camera landscape displacement LOD
    static bool enableNearDisplacement;   // Master toggle for 2x2 near-camera patch displacement
    static float displacementScale;       // World-unit crevice depth from _paramh alpha.
    // CPU baker remap: alpha 1 stays at original height; alpha 0 reaches full negative scale.
    // gamma shapes full-range depth; pivot saturates magnitude before scale.
    static float displacementGamma;
    static float displacementPivot;
    static float heightBlendStrength;     // Lerp between AlphaGrid and height-biased mask in PS (0..1)
    static float heightBlendContrast;     // Sharpness of the height pick (0..1)
    static bool debugHighlightNearPatches; // Tint the 4 selected near patches yellow for selection QA

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
    static char frameConfigLabel[128];         // Current frame's mode/config string
    static char frozenConfigLabel[128];        // Frozen frame's mode/config string

    // Detailed trace
    static bool traceEnabled;
    static bool traceShowStateChanges;
    static bool traceOnlyDeltas;
    static bool traceShowDIPDetails;
    static bool traceShowViewportClip;
    static char traceScenarioLabel[128];

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

    // D3D Command Buffer
    static bool cmdBufferRecording;       // Enable command buffer recording - default false
    static bool cmdBufferReplay;          // Enable command buffer replay mode - default false
    static bool stateSuppression;         // Enable MW state suppression for async (Phase B) - default false
    static bool syncGpuThread;            // Enable sync GPU thread (Phase B2) - default false
    static bool asyncGpuThread;           // Enable async GPU thread (Phase C) - default false
    static int cmdBufferCmdCount;         // Last frame's command count
    static int cmdBufferSizeKB;           // Last frame's buffer size in KB
    static int cmdStagePreScene;          // Per-stage command counts
    static int cmdStageScene0;
    static int cmdStageInterScene;
    static int cmdStageScene1;            // Scene 1 (particles) + Scene 2 (hands) combined
    static int cmdStageUI;

    // Debug hotkey gating
    static bool debugKeysEnabled;         // Gate debug hotkeys (F11/U/Y/L/F5/F6) - default false

    // On-screen frame number overlay - default false
    static bool showFrameNumber;

    // Scene handover logging (particle bug diagnostics)
    static bool handoverLogging;          // Log state at Scene 0/1/2 boundaries - default false

    // Stress testing toggles (sync path validation)
    static bool stressCorruptState;       // Inject bad states before HLSL rendering - default false
    static bool stressValidateTracker;    // Validate tracker matches device state - default false
    static bool stressVerifyRestore;      // Verify state restoration after GpuExit - default false
    static bool stressPoisonBuffers;      // Poison old buffers to catch N-1/N-2 bugs - default false
    static bool stressAsyncDelay;         // Simulate async delays for race detection - default false

public:
    static bool GetHandoverLogging() { return handoverLogging; }
    static void SetHandoverLogging(bool v) { handoverLogging = v; }

    // Stress test getters
    static bool GetStressCorruptState() { return stressCorruptState; }
    static bool GetStressValidateTracker() { return stressValidateTracker; }
    static bool GetStressVerifyRestore() { return stressVerifyRestore; }
    static bool GetStressPoisonBuffers() { return stressPoisonBuffers; }
    static bool GetStressAsyncDelay() { return stressAsyncDelay; }
};
