#include "imgui_manager.h"
#include "support/log.h"
#include "configuration.h"
#include "ffeshader.h"
#include "mwbridge.h"
#include <cstdio>

bool ImGuiManager::initialized = false;
bool ImGuiManager::showDemo = false;
bool ImGuiManager::showPCFInterface = false;
bool ImGuiManager::showDebugInterface = false;
bool ImGuiManager::showHiZInterface = false;
HWND ImGuiManager::windowHandle = nullptr;

// PCF filtering variables
float ImGuiManager::pcfFilterSize = 3.0f;        // Base filter size in texels
float ImGuiManager::pcfPenumbraScale = 1.0f;     // Scale factor for distance-based penumbra
float ImGuiManager::pcfMinPenumbra = 2.0f;       // Minimum penumbra size
float ImGuiManager::pcfMaxPenumbra = 5.0f;       // Maximum penumbra size
float ImGuiManager::pcfBias = 0.0015f;           // Depth bias to prevent acne
float ImGuiManager::pcfBias2 = 0.0045f;          // Second depth bias for lerp
float ImGuiManager::pcfSlopeBias = 0.001f;       // Slope-based bias to prevent acne on angled surfaces

// Debug interface controls
int ImGuiManager::bboxVisualizationMode = 0;
bool ImGuiManager::disableHiZCulling = false;
bool ImGuiManager::enableStatelessBatch = false;
bool ImGuiManager::forceLightMode3 = false;

// Debug stats
int ImGuiManager::debugRecordedCalls = 0;
int ImGuiManager::debugRenderedCalls = 0;
int ImGuiManager::debugCulledCalls = 0;
int ImGuiManager::debugSceneLights = 0;
int ImGuiManager::debugRecordMWSize = 0;
int ImGuiManager::debugImmediateCount = 0;

// Hi-Z visualization variables
int ImGuiManager::hiZDisplayMip = 0;
bool ImGuiManager::hiZInvert = false;

// Hi-Z occluder selection parameters
int ImGuiManager::occluderMaxCount = 100;
int ImGuiManager::occluderP0ExtraBudget = 80;
int ImGuiManager::occluderP1ExtraBudget = 50;
int ImGuiManager::occluderP2ExtraBudget = 30;
int ImGuiManager::occluderMinTriangles = 0;
int ImGuiManager::occluderMaxTriangles = 9999999;
float ImGuiManager::occluderCloseDistance = 2048.0f;

// Wall detection heuristics (shape-based occluder selection)
bool ImGuiManager::wallDetectionEnabled = true;
float ImGuiManager::wallFlatnessThreshold = 0.15f;   // Thin dim < 15% of mid dim = wall-like
float ImGuiManager::wallMinLargeDim = 200.0f;        // Must be at least 200 units in largest dim
float ImGuiManager::wallMaxThinDim = 50.0f;          // Thin dimension must be < 50 units
int ImGuiManager::occluderWallExtraBudget = 100;     // Extra budget for wall-shaped occluders

// Occluder highlighting (debug visualization)
bool ImGuiManager::highlightOccluders = false;

// Render bin highlighting (debug visualization)
bool ImGuiManager::highlightBins = false;

// Rasterize All mode - bypass all heuristics
bool ImGuiManager::rasterizeAll = false;

// Debug/Performance mode toggles
bool ImGuiManager::stateLeakDetection = false;
bool ImGuiManager::performanceMode = true;
bool ImGuiManager::materialSortEnabled = false;
bool ImGuiManager::instancingEnabled = false;

// Per-bin DIP suppression toggles
bool ImGuiManager::suppressSky = false;
bool ImGuiManager::suppressTerrain = false;
bool ImGuiManager::suppressOpaque = false;
bool ImGuiManager::suppressSkinning = false;
bool ImGuiManager::suppressGrass = false;
bool ImGuiManager::suppressAlphaTested = false;
bool ImGuiManager::suppressBlending = false;
bool ImGuiManager::suppress1PSkinning = false;
bool ImGuiManager::suppress1PAlpha = false;
bool ImGuiManager::suppress1POther = false;
bool ImGuiManager::suppressOffscreen = false;
int ImGuiManager::offscreenBudget = 9999;  // unlimited by default
bool ImGuiManager::suppressUI = false;
bool ImGuiManager::suppressStencilShadow = false;
bool ImGuiManager::suppressPreScene = false;
bool ImGuiManager::suppressWater = false;

// Per-bin DIP counter stats
ImGuiManager::DIPBinStats ImGuiManager::dipStats = {};

// DIP spike freeze state
bool ImGuiManager::dipFrozen = false;
bool ImGuiManager::dipAutoFreeze = true;
int ImGuiManager::dipSpikeThreshold = 2000;
ImGuiManager::DIPBinStats ImGuiManager::frozenDipStats = {};
int ImGuiManager::frozenTotal = 0;

// Frame event log
bool ImGuiManager::showFrameEventLog = false;
std::vector<FrameEvent> ImGuiManager::frameEvents;
std::vector<FrameEvent> ImGuiManager::displayFrameEvents;
std::vector<FrameEvent> ImGuiManager::frozenFrameEvents;
bool ImGuiManager::eventLogFrozen = false;
bool ImGuiManager::eventLogAutoFreeze = true;
int ImGuiManager::eventLogFreezeOffscreenThreshold = 1;  // Freeze when offscreen >= 1
char ImGuiManager::frameConfigLabel[128] = {};
char ImGuiManager::frozenConfigLabel[128] = {};

// Detailed trace
bool ImGuiManager::traceEnabled = false;
bool ImGuiManager::traceShowStateChanges = true;
bool ImGuiManager::traceOnlyDeltas = false;
bool ImGuiManager::traceShowDIPDetails = true;
bool ImGuiManager::traceShowViewportClip = true;
char ImGuiManager::traceScenarioLabel[128] = "";

// State shadow for delta tracking (file-scope)
static DWORD s_rsState[256] = {};
static DWORD s_tssState[8][32] = {};

// Slow frame detection state
bool ImGuiManager::slowFrameFrozen = false;
bool ImGuiManager::slowFrameAutoFreeze = true;
float ImGuiManager::slowFrameThreshold = 5.0f;
float ImGuiManager::slowCallThreshold = 5.0f;
float ImGuiManager::frozenPrepareMs = 0.0f;
float ImGuiManager::frozenReplayMs = 0.0f;
int ImGuiManager::frozenSlowCallIndex = -1;
float ImGuiManager::frozenSlowCallMs = 0.0f;
int ImGuiManager::frozenSlowCallPrims = 0;
int ImGuiManager::frozenSlowCallBin = 0;

// Debug hotkey gating
// D3D Command Buffer
bool ImGuiManager::cmdBufferRecording = false;
bool ImGuiManager::cmdBufferReplay = false;
bool ImGuiManager::stateSuppression = false;
bool ImGuiManager::asyncGpuThread = false;
int ImGuiManager::cmdBufferCmdCount = 0;
int ImGuiManager::cmdBufferSizeKB = 0;
int ImGuiManager::cmdStagePreScene = 0;
int ImGuiManager::cmdStageScene0 = 0;
int ImGuiManager::cmdStageInterScene = 0;
int ImGuiManager::cmdStageScene1 = 0;
int ImGuiManager::cmdStageUI = 0;

bool ImGuiManager::debugKeysEnabled = false;
bool ImGuiManager::handoverLogging = false;

// Stress testing toggles
bool ImGuiManager::stressCorruptState = false;
bool ImGuiManager::stressValidateTracker = false;
bool ImGuiManager::stressVerifyRestore = false;
bool ImGuiManager::stressPoisonBuffers = false;
bool ImGuiManager::stressAsyncDelay = false;

// Hi-Z single object visualization mode
bool ImGuiManager::hiZSingleObjectMode = false;
int ImGuiManager::hiZSingleObjectIndex = 0;
int ImGuiManager::hiZTotalObjectCount = 0;

bool ImGuiManager::Initialize(HWND hwnd, IDirect3DDevice9* device) {
    if (initialized) {
        return true;
    }

    // Store window handle for mouse input polling
    windowHandle = hwnd;

    // Load PCF values from configuration
    pcfFilterSize = Configuration.PCF.FilterSize;
    pcfPenumbraScale = Configuration.PCF.PenumbraScale;
    pcfMinPenumbra = Configuration.PCF.MinPenumbra;
    pcfMaxPenumbra = Configuration.PCF.MaxPenumbra;
    pcfBias = Configuration.PCF.Bias;
    pcfBias2 = Configuration.PCF.Bias2;
    pcfSlopeBias = Configuration.PCF.SlopeBias;
    

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    
    // Enable keyboard and gamepad controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    
    // Enable ImGui to draw its own cursor when interface is showing
    io.MouseDrawCursor = showPCFInterface;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    if (!ImGui_ImplWin32_Init(hwnd)) {
        LOG::logline(">> Failed to initialize ImGui Win32 implementation");
        return false;
    }

    if (!ImGui_ImplDX9_Init(device)) {
        LOG::logline(">> Failed to initialize ImGui DX9 implementation");
        ImGui_ImplWin32_Shutdown();
        return false;
    }

    initialized = true;
    LOG::logline(">> ImGui initialized successfully");
    return true;
}

void ImGuiManager::Shutdown() {
    if (!initialized) {
        return;
    }


    ImGui_ImplDX9_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    
    initialized = false;
    LOG::logline(">> ImGui shutdown");
}

void ImGuiManager::NewFrame() {
    if (!initialized) {
        return;
    }

    // Update mouse position and button states manually since we don't intercept Windows messages
    ImGuiIO& io = ImGui::GetIO();
    
    // Get mouse position relative to the window
    POINT mousePos;
    if (GetCursorPos(&mousePos) && ScreenToClient(windowHandle, &mousePos)) {
        io.MousePos = ImVec2((float)mousePos.x, (float)mousePos.y);
    }
    
    // Update mouse button states
    io.MouseDown[0] = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
    io.MouseDown[1] = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
    io.MouseDown[2] = (GetAsyncKeyState(VK_MBUTTON) & 0x8000) != 0;
    
    // Update mouse wheel (if needed, would require additional polling)
    
    ImGui_ImplDX9_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
}

void ImGuiManager::Render() {
    if (!initialized) {
        return;
    }

    // Show demo window if enabled (for testing)
    if (showDemo) {
        ImGui::ShowDemoWindow(&showDemo);
    }

    // Show PCF filtering interface
    if (showPCFInterface) {
        RenderPCFFilteringInterface();
    }

    // Show debug interface
    if (showDebugInterface) {
        RenderDebugInterface();
    }

    // Show Hi-Z visualization interface
    if (showHiZInterface) {
        RenderHiZInterface();
    }

    // Show frame event log
    if (showFrameEventLog) {
        RenderFrameEventLog();
    }

    ImGui::Render();
    ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());
}

bool ImGuiManager::WantCaptureMouse() {
    if (!initialized) {
        return false;
    }
    return ImGui::GetIO().WantCaptureMouse;
}

bool ImGuiManager::WantCaptureKeyboard() {
    if (!initialized) {
        return false;
    }
    return ImGui::GetIO().WantCaptureKeyboard;
}

void ImGuiManager::RenderPCFFilteringInterface() {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(350, 250), ImGuiCond_FirstUseEver);
    
    bool windowWasOpen = showPCFInterface;
    if (ImGui::Begin("PCF Shadow Filtering", &showPCFInterface, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("PCF Shadow Parameters");
        
        // New PCF parameters that map to shader constants
        ImGui::SliderFloat("Filter Size", &pcfFilterSize, 1.0f, 8.0f, "%.1f");
        
        ImGui::SliderFloat("Penumbra Scale", &pcfPenumbraScale, 0.1f, 3.0f, "%.2f");
        ImGui::SliderFloat("Min Penumbra", &pcfMinPenumbra, 0.5f, 5.0f, "%.1f");
        ImGui::SliderFloat("Max Penumbra", &pcfMaxPenumbra, 2.0f, 15.0f, "%.1f");
        ImGui::SliderFloat("Depth Bias", &pcfBias, 0.0f, 0.01f, "%.4f");
        ImGui::SliderFloat("Depth Bias 2", &pcfBias2, 0.0f, 0.01f, "%.4f");
        ImGui::SliderFloat("Slope Bias", &pcfSlopeBias, 0.0f, 0.01f, "%.4f");

        ImGui::Separator();
        ImGui::Checkbox("Show Demo Window", &showDemo);
        
        ImGui::Text("Press F11 to toggle this interface");
    }
    ImGui::End();
    
    // Check if window was closed by clicking X button
    if (windowWasOpen && !showPCFInterface) {
        Configuration.PCF.FilterSize = pcfFilterSize;
        Configuration.PCF.PenumbraScale = pcfPenumbraScale;
        Configuration.PCF.MinPenumbra = pcfMinPenumbra;
        Configuration.PCF.MaxPenumbra = pcfMaxPenumbra;
        Configuration.PCF.Bias = pcfBias;
        Configuration.PCF.Bias2 = pcfBias2;
        Configuration.PCF.SlopeBias = pcfSlopeBias;
        Configuration.SaveSettings();
    }
    
    // Update mouse cursor visibility based on interface state
    ImGuiIO& io = ImGui::GetIO();
    io.MouseDrawCursor = showPCFInterface;
}

// Getter functions for shader constants
float ImGuiManager::GetPCFFilterSize() { return pcfFilterSize; }
float ImGuiManager::GetPCFPenumbraScale() { return pcfPenumbraScale; }
float ImGuiManager::GetPCFMinPenumbra() { return pcfMinPenumbra; }
float ImGuiManager::GetPCFMaxPenumbra() { return pcfMaxPenumbra; }
float ImGuiManager::GetPCFBias() { return pcfBias; }
float ImGuiManager::GetPCFBias2() { return pcfBias2; }
float ImGuiManager::GetPCFSlopeBias() { return pcfSlopeBias; }
bool ImGuiManager::GetShowPCFInterface() { return showPCFInterface; }

void ImGuiManager::TogglePCFInterface() { 
    bool wasShowing = showPCFInterface;
    showPCFInterface = !showPCFInterface;
    
    // Update mouse cursor visibility
    ImGuiIO& io = ImGui::GetIO();
    io.MouseDrawCursor = showPCFInterface;
    
    // Save PCF settings when interface is being closed
    if (wasShowing && !showPCFInterface) {
        Configuration.PCF.FilterSize = pcfFilterSize;
        Configuration.PCF.PenumbraScale = pcfPenumbraScale;
        Configuration.PCF.MinPenumbra = pcfMinPenumbra;
        Configuration.PCF.MaxPenumbra = pcfMaxPenumbra;
        Configuration.PCF.Bias = pcfBias;
        Configuration.PCF.Bias2 = pcfBias2;
        Configuration.PCF.SlopeBias = pcfSlopeBias;
        Configuration.SaveSettings();
    }
}

LRESULT ImGuiManager::HandleWindowMessage(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (!initialized) {
        return 0;
    }
    
    extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    return ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam);
}

void ImGuiManager::OnLostDevice() {
    if (initialized) {
        ImGui_ImplDX9_InvalidateDeviceObjects();
        LOG::logline(">> ImGui device objects invalidated");
    }
}

void ImGuiManager::OnResetDevice() {
    if (initialized) {
        ImGui_ImplDX9_CreateDeviceObjects();
        LOG::logline(">> ImGui device objects recreated");
    }
}

bool ImGuiManager::GetDebugKeysEnabled() { return debugKeysEnabled; }

void ImGuiManager::RenderDebugInterface() {
    ImGui::SetNextWindowPos(ImVec2(400, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(450, 400), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("HLSL Pipeline Debug", &showDebugInterface, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Checkbox("Enable Debug Hotkeys (F11/U/F5/F6)", &debugKeysEnabled);
        ImGui::Checkbox("Handover Logging (Scene 0/1/2 state)", &handoverLogging);
        ImGui::SameLine();
        if (ImGui::Button("Save Baselines")) {
            FixedFunctionShader::saveCurrentAsBaseline();
        }
        ImGui::SameLine();
        if (ImGui::Button("Validate")) {
            FixedFunctionShader::validateAgainstBaselines();
        }
        ImGui::Separator();

        ImGui::Text("Recording/Replay System (Scene 0)");
        ImGui::Separator();

        ImGui::Checkbox("Disable Hi-Z Culling (terrain hole diagnosis)", &disableHiZCulling);
        ImGui::Checkbox("Force LightMode 3 (texture lights)", &forceLightMode3);
        ImGui::Checkbox("Enable Stateless Batching (experimental)", &enableStatelessBatch);

        ImGui::Separator();
        ImGui::Text("Optimization Modes");
        if (ImGui::Checkbox("Performance Mode (Dirty Tracking)", &performanceMode)) {
            if (performanceMode) stateLeakDetection = false;  // Mutually exclusive
        }
        ImGui::SetItemTooltip("Skip redundant GPU state updates for unchanged meshes between frames");
        if (ImGui::Checkbox("State Leak Detection (Heavy)", &stateLeakDetection)) {
            if (stateLeakDetection) performanceMode = false;  // Mutually exclusive
        }
        ImGui::SetItemTooltip("Query device state before each draw to detect state leaks. Very slow!");
        ImGui::Checkbox("Material Sort (Opaque/Grass)", &materialSortEnabled);
        ImGui::SetItemTooltip("Sort opaque/grass draws by material to minimize state changes. A/B comparison.");
        ImGui::Checkbox("GPU Instancing (Experimental)", &instancingEnabled);
        ImGui::SetItemTooltip("Batch identical geometry+material draws using GPU instancing. Requires material sort.");

        ImGui::Separator();
        ImGui::Text("Statistics");

        // Recording stats
        ImGui::Text("Scene 0 (Recorded): %d calls", debugRecordedCalls);
        ImGui::Text("  Rendered: %d (%.1f%%)", debugRenderedCalls,
                    debugRecordedCalls > 0 ? (debugRenderedCalls * 100.0f) / debugRecordedCalls : 0.0f);
        ImGui::Text("  Culled: %d (%.1f%%)", debugCulledCalls,
                    debugRecordedCalls > 0 ? (debugCulledCalls * 100.0f) / debugRecordedCalls : 0.0f);

        ImGui::Separator();

        // Lighting stats
        ImGui::Text("Lights:");
        ImGui::Text("  Scene Lights: %d", debugSceneLights);

        ImGui::Separator();

        // Material sorting optimization metrics
        ImGui::Text("Material Sorting Analysis:");
        ImGui::Text("  Unique Textures: %d", g_replayMetrics.uniqueTextures);
        ImGui::Text("  Unique Materials: %d", g_replayMetrics.uniqueMaterialKeys);
        ImGui::Text("  Material Transitions: %d", g_replayMetrics.materialTransitions);
        ImGui::Text("  Draw Calls: %d", g_replayMetrics.totalDrawCalls);
        if (g_replayMetrics.totalDrawCalls > 0 && g_replayMetrics.uniqueMaterialKeys > 0) {
            int optimalTransitions = g_replayMetrics.uniqueMaterialKeys - 1;
            int excessTransitions = g_replayMetrics.materialTransitions - optimalTransitions;
            float sortingEfficiency = optimalTransitions > 0 ?
                (float)optimalTransitions / g_replayMetrics.materialTransitions * 100.0f : 100.0f;
            ImGui::Text("  Optimal Transitions: %d", optimalTransitions);
            ImGui::Text("  Excess Transitions: %d", excessTransitions > 0 ? excessTransitions : 0);
            ImGui::Text("  Sorting Efficiency: %.1f%%", sortingEfficiency);
        }

        ImGui::Separator();

        // Instancing opportunity analysis (Opaque/Terrain/Grass bins only)
        ImGui::Text("Instancing Analysis (Opaque+Terrain+Grass):");
        ImGui::Text("  Unique InstanceKeys: %d", g_replayMetrics.uniqueInstanceKeys);
        ImGui::Text("  Potential Batches: %d", g_replayMetrics.potentialBatches);
        ImGui::Text("  Batchable Draws: %d", g_replayMetrics.totalBatchableDraws);
        if (g_replayMetrics.uniqueInstanceKeys > 0) {
            float instanceRatio = (float)g_replayMetrics.totalBatchableDraws / g_replayMetrics.uniqueInstanceKeys;
            float batchPotential = g_replayMetrics.potentialBatches > 0 ?
                (1.0f - (float)g_replayMetrics.potentialBatches / g_replayMetrics.totalBatchableDraws) * 100.0f : 0.0f;
            ImGui::Text("  Instance Ratio: %.2f draws/key", instanceRatio);
            ImGui::Text("  Draw Call Reduction: %.1f%%", batchPotential);
        }

        ImGui::Separator();

        // Depth buffer stats
        ImGui::Text("Depth Buffer (recordMW): %d geometries", debugRecordMWSize);
        ImGui::Text("Scene 1/2 Immediate Renders: %d", debugImmediateCount);

        ImGui::Separator();
        ImGui::Text("Bounding Box Visualization");

        const char* bboxModes[] = { "OFF", "Objects (Green=Rendered, Red=Culled)", "Lights (Green=Visible, Red=Culled)" };
        ImGui::Combo("BBox Mode", &bboxVisualizationMode, bboxModes, 3);

        ImGui::Separator();
        ImGui::Text("DIP Categories (uncheck to suppress)");

        // Choose frozen or live values for display
        const DIPBinStats& disp = dipFrozen ? frozenDipStats : dipStats;
        int dispTotal = disp.total();

        if (dipFrozen) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
            ImGui::Text("FROZEN (spike: %d total)", frozenTotal);
            ImGui::PopStyleColor();
            if (ImGui::Button("Unfreeze")) {
                dipFrozen = false;
            }
        } else {
            ImGui::Text("Total: %d DIPs/frame", dispTotal);
        }

        ImGui::Checkbox("Auto-freeze on spike", &dipAutoFreeze);
        ImGui::SliderInt("Spike threshold", &dipSpikeThreshold, 500, 10000);

        ImGui::Text("Scene 0 (HLSL Recorded):");
        ImGui::Checkbox("Sky##s0", &suppressSky);
        ImGui::SameLine(); ImGui::Text("= %d", disp.sky);
        if (suppressSky) { ImGui::SameLine(); ImGui::TextColored(ImVec4(0.5f,0.5f,1.0f,1.0f), "[wireframe]"); }

        ImGui::Checkbox("Terrain##s0", &suppressTerrain);
        ImGui::SameLine(); ImGui::Text("= %d", disp.terrain);

        ImGui::Checkbox("Opaque##s0", &suppressOpaque);
        ImGui::SameLine(); ImGui::Text("= %d", disp.opaque);

        ImGui::Checkbox("Skinning##s0", &suppressSkinning);
        ImGui::SameLine(); ImGui::Text("= %d", disp.skinning);

        ImGui::Checkbox("Grass##s0", &suppressGrass);
        ImGui::SameLine(); ImGui::Text("= %d", disp.grass);

        ImGui::Checkbox("Alpha Tested##s0", &suppressAlphaTested);
        ImGui::SameLine(); ImGui::Text("= %d", disp.alphaTested);

        ImGui::Checkbox("Blending##s0", &suppressBlending);
        ImGui::SameLine(); ImGui::Text("= %d", disp.blending);

        ImGui::Checkbox("Water##s0", &suppressWater);
        ImGui::SameLine(); ImGui::Text("= %d", disp.water);
        if (suppressWater) { ImGui::SameLine(); ImGui::TextColored(ImVec4(0.5f,0.5f,1.0f,1.0f), "[wireframe]"); }

        ImGui::Text("Scene 1 (Particles) / Scene 2 (Hands):");
        ImGui::Checkbox("Scene 2 Skinning (Hands)##s1", &suppress1PSkinning);
        ImGui::SameLine(); ImGui::Text("= %d", disp.firstPersonSkinning);

        ImGui::Checkbox("Scene 1 Alpha (Particles)##s1", &suppress1PAlpha);
        ImGui::SameLine(); ImGui::Text("= %d", disp.firstPersonAlpha);

        ImGui::Checkbox("Scene 1/2 Other##s1", &suppress1POther);
        ImGui::SameLine(); ImGui::Text("= %d", disp.firstPersonOther);

        ImGui::Text("Pass-through:");
        ImGui::Checkbox("Offscreen (Map/Inv)##pt", &suppressOffscreen);
        ImGui::SameLine(); ImGui::Text("= %d", disp.offscreen);
        ImGui::SliderInt("Offscreen Budget##pt", &offscreenBudget, 0, 50, "%d scenes/frame");
        ImGui::SameLine();
        if (ImGui::SmallButton("Unlimited##ob")) offscreenBudget = 9999;

        ImGui::Checkbox("UI / Menu##pt", &suppressUI);
        ImGui::SameLine(); ImGui::Text("= %d", disp.ui);

        ImGui::Checkbox("Stencil Shadow##pt", &suppressStencilShadow);
        ImGui::SameLine(); ImGui::Text("= %d", disp.stencilShadow);

        ImGui::Checkbox("Pre-Scene##pt", &suppressPreScene);
        ImGui::SameLine(); ImGui::Text("= %d", disp.preScene);

        ImGui::Separator();
        ImGui::Text("Slow Frame Detection");

        static const char* binNamesUI[] = { "Terrain", "Opaque", "Skinning", "Grass", "AlphaTested", "Blending" };

        if (slowFrameFrozen) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
            ImGui::Text("FROZEN (prepare: %.1fms, replay: %.1fms)", frozenPrepareMs, frozenReplayMs);
            if (frozenSlowCallIndex >= 0) {
                const char* binName = (frozenSlowCallBin >= 0 && frozenSlowCallBin < 6) ? binNamesUI[frozenSlowCallBin] : "?";
                ImGui::Text("Worst call[%d]: %.1fms, bin=%s, prims=%d", frozenSlowCallIndex, frozenSlowCallMs, binName, frozenSlowCallPrims);
            }
            ImGui::PopStyleColor();
            if (ImGui::Button("Unfreeze Slow Frame")) {
                slowFrameFrozen = false;
            }
        } else {
            ImGui::Text("(no slow frame detected)");
        }

        ImGui::Checkbox("Auto-freeze on slow frame", &slowFrameAutoFreeze);
        ImGui::SliderFloat("Frame threshold (ms)", &slowFrameThreshold, 1.0f, 50.0f, "%.1f");
        ImGui::SliderFloat("Call threshold (ms)", &slowCallThreshold, 1.0f, 50.0f, "%.1f");

        ImGui::Separator();
        ImGui::Text("D3D Command Buffer");
        ImGui::Checkbox("Record All MW Calls", &cmdBufferRecording);
        ImGui::SetItemTooltip("Record every MW D3D call into a command buffer (dual-write: still forwards to device)");
        ImGui::Checkbox("Command Buffer Replay", &cmdBufferReplay);
        ImGui::SetItemTooltip("Skip MW forwards, replay entire buffer at Present(). Forces recording on.");
        ImGui::Checkbox("State Suppression (Phase B)", &stateSuppression);
        ImGui::SetItemTooltip("Suppress Scene 0 MW state calls to device. Uses tracked state for restore. Requires recording.");
        ImGui::Checkbox("Async GPU Thread (Phase C)", &asyncGpuThread);
        ImGui::SetItemTooltip("Submit GPU work to render thread at Present(). Auto-enables suppression. Wait at UI BeginScene.");
        if (cmdBufferRecording || cmdBufferReplay) {
            ImGui::Text("  Commands: %d  Size: %d KB", cmdBufferCmdCount, cmdBufferSizeKB);
            ImGui::Text("  Pre:%d S0:%d Inter:%d S1/2:%d UI:%d",
                cmdStagePreScene, cmdStageScene0, cmdStageInterScene, cmdStageScene1, cmdStageUI);
        }

        ImGui::Separator();
        ImGui::Text("Stress Tests (Sync Path Validation)");
        ImGui::Checkbox("Corrupt Device State", &stressCorruptState);
        ImGui::SetItemTooltip("Inject wrong states before HLSL rendering - visual should be identical (proves state is properly set)");
        ImGui::Checkbox("Validate Tracker", &stressValidateTracker);
        ImGui::SetItemTooltip("Compare MWStateTracker vs device state at key points - logs mismatches");
        ImGui::Checkbox("Verify Restore", &stressVerifyRestore);
        ImGui::SetItemTooltip("Verify state restoration after GpuExit - logs any mismatches");
        ImGui::Checkbox("Poison Buffers", &stressPoisonBuffers);
        ImGui::SetItemTooltip("Zero old buffer after swap to catch N-1/N-2 confusion bugs");
        ImGui::Checkbox("Async Delay Sim", &stressAsyncDelay);
        ImGui::SetItemTooltip("Insert artificial delays to simulate async race conditions");

        ImGui::Separator();
        ImGui::Text("Press G to toggle this interface");
    }
    ImGui::End();

    // Update mouse cursor visibility based on interface state
    ImGuiIO& io = ImGui::GetIO();
    io.MouseDrawCursor = showDebugInterface || showPCFInterface || showHiZInterface || showFrameEventLog;
}

void ImGuiManager::ToggleDebugInterface() {
    showDebugInterface = !showDebugInterface;

    // Update mouse cursor visibility
    ImGuiIO& io = ImGui::GetIO();
    io.MouseDrawCursor = showDebugInterface || showPCFInterface || showHiZInterface || showFrameEventLog;
}

// Debug control getters (Recording/Replay/Immediate/Depth always enabled)
bool ImGuiManager::GetEnableRecording() { return true; }
bool ImGuiManager::GetEnableReplay() { return true; }
bool ImGuiManager::GetEnableImmediateRendering() { return true; }
bool ImGuiManager::GetEnableDepthPass() { return true; }
int ImGuiManager::GetBBoxVisualizationMode() { return bboxVisualizationMode; }
bool ImGuiManager::GetDisableHiZCulling() { return disableHiZCulling; }
bool ImGuiManager::GetEnableStatelessBatch() { return enableStatelessBatch; }
bool ImGuiManager::GetForceLightMode3() { return forceLightMode3; }

void ImGuiManager::UpdateDebugStats(int recordedCalls, int renderedCalls, int culledCalls,
                                     int sceneLights, int recordMWSize, int immediateCount) {
    debugRecordedCalls = recordedCalls;
    debugRenderedCalls = renderedCalls;
    debugCulledCalls = culledCalls;
    debugSceneLights = sceneLights;
    debugRecordMWSize = recordMWSize;
    debugImmediateCount = immediateCount;
}

void ImGuiManager::UpdateDIPStats(const DIPBinStats& stats) {
    dipStats = stats;
}

void ImGuiManager::UpdateCmdBufferStats(int cmdCount, int sizeKB) {
    cmdBufferCmdCount = cmdCount;
    cmdBufferSizeKB = sizeKB;
}

void ImGuiManager::UpdateCmdBufferPerStageStats(int preScene, int scene0, int interScene, int scene1, int ui) {
    cmdStagePreScene = preScene;
    cmdStageScene0 = scene0;
    cmdStageInterScene = interScene;
    cmdStageScene1 = scene1;
    cmdStageUI = ui;
}

void ImGuiManager::FreezeDIPStats(const DIPBinStats& stats) {
    dipFrozen = true;
    frozenDipStats = stats;
    frozenTotal = stats.total();
}

void ImGuiManager::UpdateReplayBinCounts(int terrain, int opaque, int skinning, int grass, int alphaTested, int blending) {
    dipStats.terrain += terrain;
    dipStats.opaque += opaque;
    dipStats.skinning += skinning;
    dipStats.grass += grass;
    dipStats.alphaTested += alphaTested;
    dipStats.blending += blending;
}

// D3D enum name lookups
static const char* D3DRSName(DWORD rs) {
    switch (rs) {
    case 7:  return "ZENABLE";
    case 8:  return "FILLMODE";
    case 9:  return "SHADEMODE";
    case 14: return "ZWRITEENABLE";
    case 15: return "ALPHATESTENABLE";
    case 16: return "LASTPIXEL";
    case 19: return "SRCBLEND";
    case 20: return "DESTBLEND";
    case 22: return "CULLMODE";
    case 23: return "ZFUNC";
    case 24: return "ALPHAREF";
    case 25: return "ALPHAFUNC";
    case 26: return "DITHERENABLE";
    case 27: return "ALPHABLENDENABLE";
    case 28: return "FOGENABLE";
    case 29: return "SPECULARENABLE";
    case 34: return "FOGCOLOR";
    case 35: return "FOGTABLEMODE";
    case 36: return "FOGSTART";
    case 37: return "FOGEND";
    case 38: return "FOGDENSITY";
    case 48: return "RANGEFOGENABLE";
    case 52: return "STENCILENABLE";
    case 53: return "STENCILFAIL";
    case 54: return "STENCILZFAIL";
    case 55: return "STENCILPASS";
    case 56: return "STENCILFUNC";
    case 57: return "STENCILREF";
    case 58: return "STENCILMASK";
    case 59: return "STENCILWRITEMASK";
    case 60: return "TEXTUREFACTOR";
    case 136: return "CLIPPING";
    case 137: return "LIGHTING";
    case 139: return "AMBIENT";
    case 140: return "FOGVERTEXMODE";
    case 141: return "COLORVERTEX";
    case 142: return "LOCALVIEWER";
    case 143: return "NORMALIZENORMALS";
    case 145: return "DIFFUSEMATERIALSOURCE";
    case 146: return "SPECULARMATERIALSOURCE";
    case 147: return "AMBIENTMATERIALSOURCE";
    case 148: return "EMISSIVEMATERIALSOURCE";
    case 151: return "VERTEXBLEND";
    case 152: return "CLIPPLANEENABLE";
    case 161: return "MULTISAMPLEANTIALIAS";
    case 168: return "COLORWRITEENABLE";
    case 171: return "BLENDOP";
    default: {
        static thread_local char buf[16];
        snprintf(buf, sizeof(buf), "RS_%d", (int)rs);
        return buf;
    }
    }
}

static const char* D3DTSSName(DWORD tss) {
    switch (tss) {
    case 1:  return "COLOROP";
    case 2:  return "COLORARG1";
    case 3:  return "COLORARG2";
    case 4:  return "ALPHAOP";
    case 5:  return "ALPHAARG1";
    case 6:  return "ALPHAARG2";
    case 7:  return "BUMPENVMAT00";
    case 8:  return "BUMPENVMAT01";
    case 9:  return "BUMPENVMAT10";
    case 10: return "BUMPENVMAT11";
    case 11: return "TEXCOORDINDEX";
    case 22: return "BUMPENVLSCALE";
    case 23: return "BUMPENVLOFFSET";
    case 24: return "TEXTURETRANSFORMFLAGS";
    case 26: return "COLORARG0";
    case 27: return "ALPHAARG0";
    case 28: return "RESULTARG";
    default: {
        static thread_local char buf[16];
        snprintf(buf, sizeof(buf), "TSS_%d", (int)tss);
        return buf;
    }
    }
}

static const char* D3DTSName(DWORD ts) {
    switch (ts) {
    case 2:   return "VIEW";
    case 3:   return "PROJECTION";
    case 256: return "WORLD";
    case 257: return "WORLD1";
    case 258: return "WORLD2";
    case 259: return "WORLD3";
    default: {
        static thread_local char buf[16];
        snprintf(buf, sizeof(buf), "TS_%d", (int)ts);
        return buf;
    }
    }
}

void ImGuiManager::ResetStateShadow() {
    memset(s_rsState, 0, sizeof(s_rsState));
    memset(s_tssState, 0, sizeof(s_tssState));
}

void ImGuiManager::AutoNameScenario() {
    auto mw = MWBridge::get();
    if (!mw || !mw->IsLoaded()) return;

    const char* location = mw->IsExterior() ? "exterior" : "interior";
    const char* weather = mw->CellHasWeather() ? "" : "_noweather";

    const char* lightMode;
    if (isHLSLActive()) {
        lightMode = "hlsl";
    } else if (Configuration.PerPixelLightFlags == 1) {
        lightMode = "ppl";
    } else {
        lightMode = "fixedfunc";
    }

    // Check for water presence from last frame's events
    bool hasWater = false;
    const auto& events = eventLogFrozen ? frozenFrameEvents : displayFrameEvents;
    for (const auto& e : events) {
        if (e.type == FrameEvent::DIP_Water || e.type == FrameEvent::MGE_WaterPlane || e.type == FrameEvent::MGE_WaterReflection) {
            hasWater = true;
            break;
        }
    }

    snprintf(traceScenarioLabel, sizeof(traceScenarioLabel), "%s_%s%s%s",
        location, lightMode, weather, hasWater ? "_water" : "");
}

// Frame event log
void ImGuiManager::LogFrameEvent(FrameEvent::Type type, int sceneNum, int primCount) {
    FrameEvent e;
    e.type = type;
    e.sceneNum = sceneNum;
    e.primCount = primCount;
    e.detail.kind = StateDetail::None;
    frameEvents.push_back(e);
}

void ImGuiManager::LogFrameEventDetailed(FrameEvent::Type type, int sceneNum, int primCount, const StateDetail& detail) {
    FrameEvent e;
    e.type = type;
    e.sceneNum = sceneNum;
    e.primCount = primCount;
    e.detail = detail;
    frameEvents.push_back(e);
}

void ImGuiManager::TraceRS(int sceneNum, DWORD state, DWORD value) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::RenderState;
    d.rs.state = state;
    d.rs.value = value;
    d.rs.prev = (state < 256) ? s_rsState[state] : 0;
    d.rs.changed = (d.rs.value != d.rs.prev);
    if (state < 256) s_rsState[state] = value;
    LogFrameEventDetailed(FrameEvent::State_RS, sceneNum, 0, d);
}

void ImGuiManager::TraceTSS(int sceneNum, DWORD stage, DWORD state, DWORD value) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::TextureStageState;
    d.tss.stage = stage;
    d.tss.state = state;
    d.tss.value = value;
    d.tss.prev = (stage < 8 && state < 32) ? s_tssState[stage][state] : 0;
    d.tss.changed = (d.tss.value != d.tss.prev);
    if (stage < 8 && state < 32) s_tssState[stage][state] = value;
    LogFrameEventDetailed(FrameEvent::State_TSS, sceneNum, 0, d);
}

void ImGuiManager::TraceTransform(int sceneNum, DWORD type, const float* matrix) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::Transform;
    d.xform.type = type;
    if (matrix) {
        memcpy(d.xform.m, matrix, 16 * sizeof(float));
    } else {
        memset(d.xform.m, 0, 16 * sizeof(float));
    }
    LogFrameEventDetailed(FrameEvent::State_Transform, sceneNum, 0, d);
}

void ImGuiManager::TraceMultiplyTransform(int sceneNum, DWORD type, const float* matrix) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::Transform;
    d.xform.type = type;
    if (matrix) {
        memcpy(d.xform.m, matrix, 16 * sizeof(float));
    } else {
        memset(d.xform.m, 0, 16 * sizeof(float));
    }
    LogFrameEventDetailed(FrameEvent::State_MultiplyTransform, sceneNum, 0, d);
}

void ImGuiManager::TraceTexture(int sceneNum, DWORD stage, void* ptr) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::Texture;
    d.tex.stage = stage;
    d.tex.ptr = (uintptr_t)ptr;
    LogFrameEventDetailed(FrameEvent::State_Texture, sceneNum, 0, d);
}

void ImGuiManager::TraceDIP(int sceneNum, FrameEvent::Type dipType, DWORD fvf, void* vb, void* ib, void* tex0,
                            DWORD primCount, DWORD vertCount, DWORD zWrite, DWORD cull,
                            DWORD alphaBlend, DWORD alphaTest, DWORD srcBlend, DWORD destBlend, DWORD vertBlend) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::DrawCall;
    d.dip.fvf = fvf;
    d.dip.vb = (uintptr_t)vb;
    d.dip.ib = (uintptr_t)ib;
    d.dip.tex0 = (uintptr_t)tex0;
    d.dip.primCount = primCount;
    d.dip.vertCount = vertCount;
    d.dip.zWrite = zWrite;
    d.dip.cull = cull;
    d.dip.alphaBlend = alphaBlend;
    d.dip.alphaTest = alphaTest;
    d.dip.srcBlend = srcBlend;
    d.dip.destBlend = destBlend;
    d.dip.vertBlend = vertBlend;
    LogFrameEventDetailed(dipType, sceneNum, primCount, d);
}

void ImGuiManager::TraceClear(int sceneNum, DWORD flags, DWORD color, float z) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::ClearCall;
    d.clear.flags = flags;
    d.clear.color = color;
    d.clear.z = z;
    LogFrameEventDetailed(FrameEvent::Clear, sceneNum, 0, d);
}

void ImGuiManager::TraceRT(int sceneNum, void* color, void* depth) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::RenderTarget;
    d.rt.color = (uintptr_t)color;
    d.rt.depth = (uintptr_t)depth;
    LogFrameEventDetailed(FrameEvent::SetRenderTarget, sceneNum, 0, d);
}

void ImGuiManager::TraceLight(int sceneNum, DWORD index, bool enable) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::Light;
    d.light.index = index;
    d.light.enable = enable;
    LogFrameEventDetailed(FrameEvent::State_Light, sceneNum, 0, d);
}

void ImGuiManager::TraceMaterial(int sceneNum, float dr, float dg, float db, float da) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::Material;
    d.mat.dr = dr;
    d.mat.dg = dg;
    d.mat.db = db;
    d.mat.da = da;
    LogFrameEventDetailed(FrameEvent::State_Material, sceneNum, 0, d);
}

void ImGuiManager::TraceViewport(int sceneNum, DWORD x, DWORD y, DWORD w, DWORD h, float minZ, float maxZ) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::Viewport;
    d.vp.x = x;
    d.vp.y = y;
    d.vp.w = w;
    d.vp.h = h;
    d.vp.minZ = minZ;
    d.vp.maxZ = maxZ;
    LogFrameEventDetailed(FrameEvent::State_Viewport, sceneNum, 0, d);
}

void ImGuiManager::TraceClipPlane(int sceneNum, DWORD index) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::ClipPlane;
    d.clip.index = index;
    LogFrameEventDetailed(FrameEvent::State_ClipPlane, sceneNum, 0, d);
}

void ImGuiManager::TraceStreamSource(int sceneNum, DWORD stream, void* vb, DWORD stride) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::StreamSource;
    d.ss.stream = stream;
    d.ss.vb = (uintptr_t)vb;
    d.ss.stride = stride;
    LogFrameEventDetailed(FrameEvent::State_StreamSource, sceneNum, 0, d);
}

void ImGuiManager::TraceVertexShader(int sceneNum, DWORD fvf) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::VertexShader;
    d.vs.fvf = fvf;
    LogFrameEventDetailed(FrameEvent::State_VertexShader, sceneNum, 0, d);
}

void ImGuiManager::TraceIndexBuffer(int sceneNum, void* ib) {
    if (!traceEnabled) return;
    StateDetail d;
    d.kind = StateDetail::IndexBuffer;
    d.ib.ib = (uintptr_t)ib;
    LogFrameEventDetailed(FrameEvent::State_IndexBuffer, sceneNum, 0, d);
}

void ImGuiManager::SnapshotFrameEvents() {
    // Build config label for this frame
    {
        const char* lightMode = isHLSLActive() ? "HLSL" :
            (Configuration.PerPixelLightFlags == 1) ? "PPL" : "Standard";
        bool distLand = (Configuration.MGEFlags & USE_DISTANT_LAND) != 0;
        const char* cmdBuf = cmdBufferReplay ? "Replay" :
            cmdBufferRecording ? "Record" : "Off";
        snprintf(frameConfigLabel, sizeof(frameConfigLabel),
            "Mode: %s | DistLand: %s | CmdBuf: %s",
            lightMode, distLand ? "ON" : "OFF", cmdBuf);
    }

    // Check auto-freeze trigger before swapping
    if (eventLogAutoFreeze && !eventLogFrozen && eventLogFreezeOffscreenThreshold > 0) {
        int offscreenCount = 0;
        for (const auto& e : frameEvents) {
            if (e.type == FrameEvent::DIP_Offscreen) offscreenCount++;
        }
        if (offscreenCount >= eventLogFreezeOffscreenThreshold) {
            frozenFrameEvents = frameEvents;  // Copy before swap
            memcpy(frozenConfigLabel, frameConfigLabel, sizeof(frozenConfigLabel));
            eventLogFrozen = true;
        }
    }

    if (!eventLogFrozen) {
        displayFrameEvents.swap(frameEvents);
    }
    // Always clear the accumulator for next frame
    frameEvents.clear();
    frameEvents.reserve(256);
    // Reset state shadow for next frame's delta tracking
    if (traceEnabled) {
        ResetStateShadow();
        AutoNameScenario();
    }
}

void ImGuiManager::ToggleFrameEventLog() {
    showFrameEventLog = !showFrameEventLog;

    ImGuiIO& io = ImGui::GetIO();
    io.MouseDrawCursor = showDebugInterface || showPCFInterface || showHiZInterface || showFrameEventLog;
}

void ImGuiManager::RenderFrameEventLog() {
    ImGui::SetNextWindowPos(ImVec2(10, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(600, 650), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Frame Event Log", &showFrameEventLog)) {
        // Use frozen or live events for display
        const auto& events = eventLogFrozen ? frozenFrameEvents : displayFrameEvents;

        // Freeze controls
        if (eventLogFrozen) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
            ImGui::Text("FROZEN (%d events)", (int)events.size());
            ImGui::PopStyleColor();
            ImGui::SameLine();
            if (ImGui::Button("Unfreeze")) {
                eventLogFrozen = false;
            }
        } else {
            ImGui::Text("%d events (live)", (int)events.size());
            ImGui::SameLine();
            if (ImGui::Button("Freeze")) {
                frozenFrameEvents = displayFrameEvents;
                memcpy(frozenConfigLabel, frameConfigLabel, sizeof(frozenConfigLabel));
                eventLogFrozen = true;
            }
        }

        // Show current mode/config
        const char* configLabel = eventLogFrozen ? frozenConfigLabel : frameConfigLabel;
        ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "%s", configLabel);

        ImGui::Checkbox("Auto-freeze on offscreen", &eventLogAutoFreeze);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::InputInt("##offThresh", &eventLogFreezeOffscreenThreshold);
        if (eventLogFreezeOffscreenThreshold < 0) eventLogFreezeOffscreenThreshold = 0;

        // Detailed trace controls
        ImGui::Separator();
        ImGui::Checkbox("Detailed Trace", &traceEnabled);
        if (traceEnabled) {
            ImGui::SameLine();
            ImGui::Checkbox("State changes", &traceShowStateChanges);
            ImGui::SameLine();
            ImGui::Checkbox("Only deltas", &traceOnlyDeltas);
            ImGui::Checkbox("DIP details", &traceShowDIPDetails);
            ImGui::SameLine();
            ImGui::Checkbox("Viewport/Clip", &traceShowViewportClip);
        }

        // Scenario label + file dump
        ImGui::SetNextItemWidth(200);
        ImGui::InputText("Scenario", traceScenarioLabel, sizeof(traceScenarioLabel));
        ImGui::SameLine();
        if (ImGui::Button("Print to File")) {
            PrintFrameEventsToFile();
        }

        ImGui::Separator();

        // Scrollable list
        if (ImGui::BeginChild("EventList", ImVec2(0, 0), false)) {
            // Collapsible summary: count each event type
            if (ImGui::TreeNode("Summary")) {
                int counts[(int)FrameEvent::Count] = {};
                int totalPrims = 0;
                for (const auto& e : events) {
                    counts[(int)e.type]++;
                    totalPrims += e.primCount;
                }
                for (int i = 0; i < (int)FrameEvent::Count; i++) {
                    if (counts[i] > 0) {
                        ImGui::Text("%-20s %d", FrameEvent::typeName((FrameEvent::Type)i), counts[i]);
                    }
                }
                ImGui::Separator();
                ImGui::Text("Total primitives: %d", totalPrims);
                ImGui::TreePop();
            }

            ImGui::Separator();

            for (int i = 0; i < (int)events.size(); i++) {
                const auto& e = events[i];
                const auto& d = e.detail;
                ImVec4 color(1.0f, 1.0f, 1.0f, 1.0f);

                bool isMGEInternal = (e.type >= FrameEvent::MGE_ShadowMap && e.type <= FrameEvent::MGE_SkyRender);
                bool isTraceEvent = (e.type >= FrameEvent::State_RS && e.type <= FrameEvent::State_MultiplyTransform);

                // Filter trace events based on checkboxes
                if (isTraceEvent) {
                    if (!traceShowStateChanges && (e.type == FrameEvent::State_RS || e.type == FrameEvent::State_TSS))
                        continue;
                    if (!traceShowViewportClip && (e.type == FrameEvent::State_Viewport || e.type == FrameEvent::State_ClipPlane))
                        continue;
                    // Only-deltas filter for RS/TSS
                    if (traceOnlyDeltas) {
                        if (e.type == FrameEvent::State_RS && d.kind == StateDetail::RenderState && !d.rs.changed)
                            continue;
                        if (e.type == FrameEvent::State_TSS && d.kind == StateDetail::TextureStageState && !d.tss.changed)
                            continue;
                    }
                }

                // Color coding
                if (e.type >= FrameEvent::DIP_Sky && e.type <= FrameEvent::DIP_Water) {
                    color = ImVec4(0.4f, 1.0f, 0.4f, 1.0f);  // green for DIP
                } else if (isMGEInternal) {
                    color = ImVec4(1.0f, 0.7f, 0.5f, 1.0f);  // orange for MGE internal
                } else if (e.type >= FrameEvent::MGE_Stage0 && e.type <= FrameEvent::MGE_HLSLReplay) {
                    color = ImVec4(0.8f, 0.8f, 1.0f, 1.0f);  // blue for MGE stages
                } else if (e.type == FrameEvent::BeginScene || e.type == FrameEvent::EndScene) {
                    color = ImVec4(1.0f, 1.0f, 0.6f, 1.0f);  // yellow for scene bounds
                } else if (isTraceEvent) {
                    // Trace state events
                    if (e.type == FrameEvent::State_RS && d.kind == StateDetail::RenderState) {
                        color = d.rs.changed ? ImVec4(0.5f, 1.0f, 1.0f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                    } else if (e.type == FrameEvent::State_TSS && d.kind == StateDetail::TextureStageState) {
                        color = d.tss.changed ? ImVec4(0.5f, 1.0f, 1.0f, 1.0f) : ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
                    } else {
                        color = ImVec4(0.5f, 1.0f, 1.0f, 1.0f);  // cyan for other state
                    }
                }

                ImGui::PushStyleColor(ImGuiCol_Text, color);

                // Format based on detail kind
                if (isTraceEvent && d.kind == StateDetail::RenderState) {
                    if (d.rs.changed) {
                        ImGui::Text("[%3d] S%d RS %s: %d->%d", i, e.sceneNum, D3DRSName(d.rs.state), d.rs.prev, d.rs.value);
                    } else {
                        ImGui::Text("[%3d] S%d RS %s: %d (=)", i, e.sceneNum, D3DRSName(d.rs.state), d.rs.value);
                    }
                } else if (isTraceEvent && d.kind == StateDetail::TextureStageState) {
                    if (d.tss.changed) {
                        ImGui::Text("[%3d] S%d TSS[%d] %s: %d->%d", i, e.sceneNum, d.tss.stage, D3DTSSName(d.tss.state), d.tss.prev, d.tss.value);
                    } else {
                        ImGui::Text("[%3d] S%d TSS[%d] %s: %d (=)", i, e.sceneNum, d.tss.stage, D3DTSSName(d.tss.state), d.tss.value);
                    }
                } else if (isTraceEvent && d.kind == StateDetail::Transform) {
                    const float* m = d.xform.m;
                    bool isIdentity = (m[0]==1 && m[5]==1 && m[10]==1 && m[15]==1 &&
                        m[1]==0 && m[2]==0 && m[3]==0 && m[4]==0 && m[6]==0 && m[7]==0 &&
                        m[8]==0 && m[9]==0 && m[11]==0 && m[12]==0 && m[13]==0 && m[14]==0);
                    if (isIdentity) {
                        ImGui::Text("[%3d] S%d %s %s = IDENTITY", i, e.sceneNum, FrameEvent::typeName(e.type), D3DTSName(d.xform.type));
                    } else {
                        // Show first row inline, full matrix in tooltip
                        ImGui::Text("[%3d] S%d %s %s [%.2f %.2f %.2f %.2f | ...]", i, e.sceneNum,
                            FrameEvent::typeName(e.type), D3DTSName(d.xform.type), m[0], m[1], m[2], m[3]);
                        if (ImGui::IsItemHovered()) {
                            ImGui::SetTooltip("%.4f %.4f %.4f %.4f\n%.4f %.4f %.4f %.4f\n%.4f %.4f %.4f %.4f\n%.4f %.4f %.4f %.4f",
                                m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                                m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
                        }
                    }
                } else if (isTraceEvent && d.kind == StateDetail::Texture) {
                    ImGui::Text("[%3d] S%d Texture[%d] = %p", i, e.sceneNum, d.tex.stage, (void*)d.tex.ptr);
                } else if (isTraceEvent && d.kind == StateDetail::DrawCall && traceShowDIPDetails) {
                    ImGui::Text("[%3d] S%d %s p=%d v=%d fvf=%X zW=%d cull=%d blend=%d vb=%X",
                        i, e.sceneNum, FrameEvent::typeName(e.type), d.dip.primCount, d.dip.vertCount,
                        d.dip.fvf, d.dip.zWrite, d.dip.cull, d.dip.alphaBlend, (unsigned)(d.dip.vb & 0xFFFF));
                } else if (isTraceEvent && d.kind == StateDetail::ClearCall) {
                    ImGui::Text("[%3d] S%d Clear flags=%X color=%08X z=%.2f", i, e.sceneNum, d.clear.flags, d.clear.color, d.clear.z);
                } else if (isTraceEvent && d.kind == StateDetail::RenderTarget) {
                    ImGui::Text("[%3d] S%d SetRT color=%p depth=%p", i, e.sceneNum, (void*)d.rt.color, (void*)d.rt.depth);
                } else if (isTraceEvent && d.kind == StateDetail::Light) {
                    ImGui::Text("[%3d] S%d Light[%d] %s", i, e.sceneNum, d.light.index, d.light.enable ? "ON" : "OFF");
                } else if (isTraceEvent && d.kind == StateDetail::Material) {
                    ImGui::Text("[%3d] S%d Material d=(%.2f,%.2f,%.2f,%.2f)", i, e.sceneNum, d.mat.dr, d.mat.dg, d.mat.db, d.mat.da);
                } else if (isTraceEvent && d.kind == StateDetail::Viewport) {
                    ImGui::Text("[%3d] S%d Viewport %dx%d+%d+%d z=[%.2f,%.2f]", i, e.sceneNum, d.vp.w, d.vp.h, d.vp.x, d.vp.y, d.vp.minZ, d.vp.maxZ);
                } else if (isTraceEvent && d.kind == StateDetail::ClipPlane) {
                    ImGui::Text("[%3d] S%d ClipPlane[%d]", i, e.sceneNum, d.clip.index);
                } else if (isTraceEvent && d.kind == StateDetail::StreamSource) {
                    ImGui::Text("[%3d] S%d Stream[%d] vb=%X stride=%d", i, e.sceneNum, d.ss.stream, (unsigned)(d.ss.vb & 0xFFFF), d.ss.stride);
                } else if (isTraceEvent && d.kind == StateDetail::VertexShader) {
                    ImGui::Text("[%3d] S%d FVF=%08X", i, e.sceneNum, d.vs.fvf);
                } else if (isTraceEvent && d.kind == StateDetail::IndexBuffer) {
                    ImGui::Text("[%3d] S%d IB=%X", i, e.sceneNum, (unsigned)(d.ib.ib & 0xFFFF));
                } else if (isMGEInternal && e.primCount > 0) {
                    ImGui::Text("[%3d] S%d %-20s draws=%d", i, e.sceneNum, FrameEvent::typeName(e.type), e.primCount);
                } else if (e.primCount > 0) {
                    ImGui::Text("[%3d] S%d %-20s prims=%d", i, e.sceneNum, FrameEvent::typeName(e.type), e.primCount);
                } else {
                    ImGui::Text("[%3d] S%d %s", i, e.sceneNum, FrameEvent::typeName(e.type));
                }
                ImGui::PopStyleColor();
            }
        }
        ImGui::EndChild();
    }
    ImGui::End();
}

void ImGuiManager::PrintFrameEventsToFile() {
    const auto& events = eventLogFrozen ? frozenFrameEvents : displayFrameEvents;
    if (events.empty()) return;

    // Build filename from scenario label
    char filename[256];
    if (traceScenarioLabel[0]) {
        // Sanitize label for filename
        char sanitized[128];
        int j = 0;
        for (int i = 0; traceScenarioLabel[i] && j < 120; i++) {
            char c = traceScenarioLabel[i];
            sanitized[j++] = (c == ' ') ? '_' : ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-') ? c : '_';
        }
        sanitized[j] = 0;
        snprintf(filename, sizeof(filename), "frame_trace_%s.txt", sanitized);
    } else {
        snprintf(filename, sizeof(filename), "frame_trace.txt");
    }

    FILE* f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "Frame Trace Log (%d events)\n", (int)events.size());
    const char* configLabel = eventLogFrozen ? frozenConfigLabel : frameConfigLabel;
    if (configLabel[0]) {
        fprintf(f, "Config: %s\n", configLabel);
    }
    if (traceScenarioLabel[0]) {
        fprintf(f, "Scenario: %s\n", traceScenarioLabel);
    }
    fprintf(f, "========================================\n\n");

    // Summary
    int counts[(int)FrameEvent::Count] = {};
    int totalPrims = 0;
    for (const auto& e : events) {
        counts[(int)e.type]++;
        totalPrims += e.primCount;
    }
    fprintf(f, "Summary:\n");
    for (int i = 0; i < (int)FrameEvent::Count; i++) {
        if (counts[i] > 0) {
            fprintf(f, "  %-24s %d\n", FrameEvent::typeName((FrameEvent::Type)i), counts[i]);
        }
    }
    fprintf(f, "  Total primitives: %d\n\n", totalPrims);

    // Ordered event list with full detail
    fprintf(f, "Events (ordered):\n");
    fprintf(f, "----------------------------------------\n");
    for (int i = 0; i < (int)events.size(); i++) {
        const auto& e = events[i];
        const auto& d = e.detail;
        bool isMGEInternal = (e.type >= FrameEvent::MGE_ShadowMap && e.type <= FrameEvent::MGE_SkyRender);
        bool isTraceEvent = (e.type >= FrameEvent::State_RS && e.type <= FrameEvent::State_MultiplyTransform);
        bool isReplayEvent = (e.type >= FrameEvent::Replay_Clear && e.type <= FrameEvent::Replay_DP);

        if (isReplayEvent && d.kind == StateDetail::ClearCall) {
            fprintf(f, "[%4d] S%d R_Clear flags=0x%X color=0x%08X z=%.4f\n", i, e.sceneNum,
                d.clear.flags, d.clear.color, d.clear.z);
        } else if (isReplayEvent && d.kind == StateDetail::RenderTarget) {
            fprintf(f, "[%4d] S%d %s ptr=0x%X\n", i, e.sceneNum, FrameEvent::typeName(e.type),
                (unsigned)((d.rt.color ? d.rt.color : d.rt.depth) & 0xFFFF));
        } else if (isReplayEvent && d.kind == StateDetail::Viewport) {
            fprintf(f, "[%4d] S%d R_Viewport %dx%d+%d+%d z=[%.4f,%.4f]\n", i, e.sceneNum,
                d.vp.w, d.vp.h, d.vp.x, d.vp.y, d.vp.minZ, d.vp.maxZ);
        } else if (isReplayEvent && d.kind == StateDetail::RenderState) {
            fprintf(f, "[%4d] S%d R_RS %-24s = %d\n", i, e.sceneNum, D3DRSName(d.rs.state), d.rs.value);
        } else if (isReplayEvent && d.kind == StateDetail::TextureStageState) {
            fprintf(f, "[%4d] S%d R_TSS[%d] %-20s = %d\n", i, e.sceneNum, d.tss.stage,
                D3DTSSName(d.tss.state), d.tss.value);
        } else if (isReplayEvent && d.kind == StateDetail::Transform) {
            const float* m = d.xform.m;
            fprintf(f, "[%4d] S%d R_Transform %s\n", i, e.sceneNum, D3DTSName(d.xform.type));
            fprintf(f, "       [%10.4f %10.4f %10.4f %10.4f]\n", m[0], m[1], m[2], m[3]);
            fprintf(f, "       [%10.4f %10.4f %10.4f %10.4f]\n", m[4], m[5], m[6], m[7]);
            fprintf(f, "       [%10.4f %10.4f %10.4f %10.4f]\n", m[8], m[9], m[10], m[11]);
            fprintf(f, "       [%10.4f %10.4f %10.4f %10.4f]\n", m[12], m[13], m[14], m[15]);
        } else if (isReplayEvent && d.kind == StateDetail::Texture) {
            fprintf(f, "[%4d] S%d R_Texture[%d] = 0x%p\n", i, e.sceneNum, d.tex.stage, (void*)d.tex.ptr);
        } else if (isReplayEvent && d.kind == StateDetail::Material) {
            fprintf(f, "[%4d] S%d R_Material diffuse=(%.3f,%.3f,%.3f,%.3f)\n", i, e.sceneNum, d.mat.dr, d.mat.dg, d.mat.db, d.mat.da);
        } else if (isReplayEvent && d.kind == StateDetail::Light) {
            fprintf(f, "[%4d] S%d R_LightEn[%d] %s\n", i, e.sceneNum, d.light.index, d.light.enable ? "ON" : "OFF");
        } else if (isReplayEvent && d.kind == StateDetail::DrawCall) {
            fprintf(f, "[%4d] S%d R_DIP prims=%d verts=%d\n", i, e.sceneNum, d.dip.primCount, d.dip.vertCount);
        } else if (isReplayEvent && d.kind == StateDetail::VertexShader) {
            fprintf(f, "[%4d] S%d R_FVF=0x%08X\n", i, e.sceneNum, d.vs.fvf);
        } else if (isReplayEvent && d.kind == StateDetail::StreamSource) {
            fprintf(f, "[%4d] S%d R_Stream[%d] vb=0x%X stride=%d\n", i, e.sceneNum, d.ss.stream, (unsigned)(d.ss.vb & 0xFFFF), d.ss.stride);
        } else if (isReplayEvent && d.kind == StateDetail::IndexBuffer) {
            fprintf(f, "[%4d] S%d R_IB=0x%X\n", i, e.sceneNum, (unsigned)(d.ib.ib & 0xFFFF));
        } else if (isReplayEvent) {
            fprintf(f, "[%4d] S%d %s prims=%d\n", i, e.sceneNum, FrameEvent::typeName(e.type), e.primCount);
        } else if (isTraceEvent && d.kind == StateDetail::RenderState) {
            fprintf(f, "[%4d] S%d RS %-24s %d -> %d %s\n", i, e.sceneNum, D3DRSName(d.rs.state),
                d.rs.prev, d.rs.value, d.rs.changed ? "CHANGED" : "(same)");
        } else if (isTraceEvent && d.kind == StateDetail::TextureStageState) {
            fprintf(f, "[%4d] S%d TSS[%d] %-20s %d -> %d %s\n", i, e.sceneNum, d.tss.stage,
                D3DTSSName(d.tss.state), d.tss.prev, d.tss.value, d.tss.changed ? "CHANGED" : "(same)");
        } else if (isTraceEvent && d.kind == StateDetail::Transform) {
            const float* m = d.xform.m;
            fprintf(f, "[%4d] S%d %-12s %s\n", i, e.sceneNum, FrameEvent::typeName(e.type), D3DTSName(d.xform.type));
            fprintf(f, "       [%10.4f %10.4f %10.4f %10.4f]\n", m[0], m[1], m[2], m[3]);
            fprintf(f, "       [%10.4f %10.4f %10.4f %10.4f]\n", m[4], m[5], m[6], m[7]);
            fprintf(f, "       [%10.4f %10.4f %10.4f %10.4f]\n", m[8], m[9], m[10], m[11]);
            fprintf(f, "       [%10.4f %10.4f %10.4f %10.4f]\n", m[12], m[13], m[14], m[15]);
        } else if (isTraceEvent && d.kind == StateDetail::Texture) {
            fprintf(f, "[%4d] S%d Texture[%d] = 0x%p\n", i, e.sceneNum, d.tex.stage, (void*)d.tex.ptr);
        } else if (isTraceEvent && d.kind == StateDetail::DrawCall) {
            fprintf(f, "[%4d] S%d %s prims=%d verts=%d fvf=0x%08X vb=0x%X ib=0x%X tex0=0x%X\n",
                i, e.sceneNum, FrameEvent::typeName(e.type), d.dip.primCount, d.dip.vertCount,
                d.dip.fvf, (unsigned)(d.dip.vb & 0xFFFF), (unsigned)(d.dip.ib & 0xFFFF), (unsigned)(d.dip.tex0 & 0xFFFF));
            fprintf(f, "       zWrite=%d cull=%d alphaBlend=%d alphaTest=%d src=%d dst=%d vertBlend=%d\n",
                d.dip.zWrite, d.dip.cull, d.dip.alphaBlend, d.dip.alphaTest,
                d.dip.srcBlend, d.dip.destBlend, d.dip.vertBlend);
        } else if (isTraceEvent && d.kind == StateDetail::ClearCall) {
            fprintf(f, "[%4d] S%d Clear flags=0x%X color=0x%08X z=%.4f\n", i, e.sceneNum,
                d.clear.flags, d.clear.color, d.clear.z);
        } else if (isTraceEvent && d.kind == StateDetail::RenderTarget) {
            fprintf(f, "[%4d] S%d SetRT color=%p depth=%p\n", i, e.sceneNum, (void*)d.rt.color, (void*)d.rt.depth);
        } else if (isTraceEvent && d.kind == StateDetail::Light) {
            fprintf(f, "[%4d] S%d Light[%d] %s\n", i, e.sceneNum, d.light.index, d.light.enable ? "ON" : "OFF");
        } else if (isTraceEvent && d.kind == StateDetail::Material) {
            fprintf(f, "[%4d] S%d Material diffuse=(%.3f,%.3f,%.3f,%.3f)\n", i, e.sceneNum, d.mat.dr, d.mat.dg, d.mat.db, d.mat.da);
        } else if (isTraceEvent && d.kind == StateDetail::Viewport) {
            fprintf(f, "[%4d] S%d Viewport %dx%d+%d+%d z=[%.4f,%.4f]\n", i, e.sceneNum, d.vp.w, d.vp.h, d.vp.x, d.vp.y, d.vp.minZ, d.vp.maxZ);
        } else if (isTraceEvent && d.kind == StateDetail::ClipPlane) {
            fprintf(f, "[%4d] S%d ClipPlane[%d]\n", i, e.sceneNum, d.clip.index);
        } else if (isTraceEvent && d.kind == StateDetail::StreamSource) {
            fprintf(f, "[%4d] S%d Stream[%d] vb=0x%X stride=%d\n", i, e.sceneNum, d.ss.stream, (unsigned)(d.ss.vb & 0xFFFF), d.ss.stride);
        } else if (isTraceEvent && d.kind == StateDetail::VertexShader) {
            fprintf(f, "[%4d] S%d FVF=0x%08X\n", i, e.sceneNum, d.vs.fvf);
        } else if (isTraceEvent && d.kind == StateDetail::IndexBuffer) {
            fprintf(f, "[%4d] S%d IB=0x%X\n", i, e.sceneNum, (unsigned)(d.ib.ib & 0xFFFF));
        } else if (isMGEInternal && e.primCount > 0) {
            fprintf(f, "[%4d] S%d %-20s draws=%d\n", i, e.sceneNum, FrameEvent::typeName(e.type), e.primCount);
        } else if (e.primCount > 0) {
            fprintf(f, "[%4d] S%d %-20s prims=%d\n", i, e.sceneNum, FrameEvent::typeName(e.type), e.primCount);
        } else {
            fprintf(f, "[%4d] S%d %s\n", i, e.sceneNum, FrameEvent::typeName(e.type));
        }
    }

    fclose(f);
    LOG::logline(">> Frame trace written to %s (%d events)", filename, (int)events.size());
}

void ImGuiManager::FreezeSlowFrame(float prepareMs, float replayMs, int worstCallIndex, float worstCallMs, int worstCallPrims, int worstCallBin) {
    slowFrameFrozen = true;
    frozenPrepareMs = prepareMs;
    frozenReplayMs = replayMs;
    frozenSlowCallIndex = worstCallIndex;
    frozenSlowCallMs = worstCallMs;
    frozenSlowCallPrims = worstCallPrims;
    frozenSlowCallBin = worstCallBin;
}

void ImGuiManager::RenderHiZInterface() {
    ImGui::SetNextWindowPos(ImVec2(800, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(650, 800), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Hi-Z Occlusion Buffer", &showHiZInterface, ImGuiWindowFlags_AlwaysAutoResize)) {
        // Hi-Z visualization disabled in async mode (buffer access would race with GPU thread)
        if (asyncGpuThread) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Hi-Z visualization disabled in async mode");
            ImGui::Text("Disable Async GPU Thread to view Hi-Z buffer.");
            ImGui::End();
            return;
        }

        auto& culler = FixedFunctionShader::softwareOcclusionCuller;

        if (culler.getHiZTexture()) {
            int maxMipLevel = culler.getHiZMipLevels() - 1;

            // Display controls
            ImGui::Text("Visualization Settings");
            ImGui::SliderInt("Mip Level", &hiZDisplayMip, 0, maxMipLevel);
            ImGui::Checkbox("Invert Depth", &hiZInvert);

            UINT width = culler.getHiZWidth(hiZDisplayMip);
            UINT height = culler.getHiZHeight(hiZDisplayMip);

            ImGui::Separator();
            ImGui::Text("Single Object Visualization");
            ImGui::Checkbox("Enable Single Object Mode", &hiZSingleObjectMode);
            if (hiZSingleObjectMode) {
                ImGui::Text("Object %d of %d", hiZSingleObjectIndex + 1, hiZTotalObjectCount);
                if (ImGui::Button("< Previous")) {
                    hiZSingleObjectIndex = (hiZSingleObjectIndex - 1 + hiZTotalObjectCount) % std::max(1, hiZTotalObjectCount);
                }
                ImGui::SameLine();
                if (ImGui::Button("Next >")) {
                    hiZSingleObjectIndex = (hiZSingleObjectIndex + 1) % std::max(1, hiZTotalObjectCount);
                }
            }

            ImGui::Separator();
            ImGui::Text("Occluder Selection Settings");
            ImGui::SliderInt("Max Occluders (Base)", &occluderMaxCount, 10, 1000);
            ImGui::Text("Priority Budgets (extra above base):");
            ImGui::SliderInt("P0: Camera Inside BBox", &occluderP0ExtraBudget, 0, 200);
            ImGui::SliderInt("P1: Off-Screen Corners", &occluderP1ExtraBudget, 0, 150);
            ImGui::SliderInt("P2: Very Close", &occluderP2ExtraBudget, 0, 100);
            ImGui::Separator();
            ImGui::Text("Triangle Count Filters:");
            ImGui::SliderInt("Min Triangles", &occluderMinTriangles, 0, 1000);
            ImGui::SliderInt("Max Triangles", &occluderMaxTriangles, 10, 50000);
            ImGui::Separator();
            ImGui::Text("Distance Thresholds:");
            ImGui::SliderFloat("Close Distance (P2)", &occluderCloseDistance, 256.0f, 8192.0f, "%.0f units");

            ImGui::Separator();
            ImGui::Text("Wall Detection (Shape-Based):");
            ImGui::Checkbox("Enable Wall Detection", &wallDetectionEnabled);
            if (wallDetectionEnabled) {
                ImGui::SliderFloat("Flatness Threshold", &wallFlatnessThreshold, 0.05f, 0.5f, "%.2f");
                ImGui::SetItemTooltip("Thin dim / mid dim ratio. Lower = stricter (0.15 = thin < 15%% of mid)");
                ImGui::SliderFloat("Min Large Dim", &wallMinLargeDim, 50.0f, 500.0f, "%.0f units");
                ImGui::SetItemTooltip("Minimum size of largest dimension to qualify as wall");
                ImGui::SliderFloat("Max Thin Dim", &wallMaxThinDim, 10.0f, 150.0f, "%.0f units");
                ImGui::SetItemTooltip("Maximum size of thin dimension to qualify as wall");
                ImGui::SliderInt("Wall Extra Budget", &occluderWallExtraBudget, 0, 200);
                ImGui::SetItemTooltip("Extra occluder budget for wall-shaped objects");
            }

            ImGui::Separator();
            ImGui::Text("Debug Visualization:");
            ImGui::Checkbox("Highlight Occluders (Green Tint)", &highlightOccluders);
            ImGui::SetItemTooltip("Tint objects selected as occluders with green to visualize selection");

            ImGui::Checkbox("Highlight Render Bins (Color Tint)", &highlightBins);
            ImGui::SetItemTooltip("Tint draw calls by render bin: Blue=Skinning, Green=Grass, Yellow=AlphaTested, Magenta=Blending");

            ImGui::Checkbox("Rasterize ALL (Bypass Heuristics)", &rasterizeAll);
            ImGui::SetItemTooltip("Rasterize ALL objects to Hi-Z buffer - bypasses all selection heuristics for debugging");

            ImGui::Separator();
            ImGui::Text("Mip %d: %dx%d", hiZDisplayMip, width, height);
            ImGui::Separator();

            // Display the Hi-Z texture at 2x scale for visibility
            float scale = 2.0f;
            ImGui::Image((void*)culler.getHiZTexture(),
                        ImVec2(width * scale, height * scale));
        } else {
            ImGui::Text("No Hi-Z buffer available");
            ImGui::Text("(Hi-Z is built during scene rendering)");
        }

        ImGui::Separator();
        ImGui::Text("Press U to toggle this interface");
    }
    ImGui::End();

    // Update mouse cursor visibility based on interface state
    ImGuiIO& io = ImGui::GetIO();
    io.MouseDrawCursor = showDebugInterface || showPCFInterface || showHiZInterface || showFrameEventLog;
}

void ImGuiManager::ToggleHiZInterface() {
    showHiZInterface = !showHiZInterface;

    // Update mouse cursor visibility
    ImGuiIO& io = ImGui::GetIO();
    io.MouseDrawCursor = showDebugInterface || showPCFInterface || showHiZInterface || showFrameEventLog;
}

bool ImGuiManager::GetShowHiZInterface() {
    return showHiZInterface;
}

