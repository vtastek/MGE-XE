#include "imgui_manager.h"
#include "support/log.h"
#include "configuration.h"
#include "ffeshader.h"
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
bool ImGuiManager::enableRecording = true;
bool ImGuiManager::enableReplay = true;
bool ImGuiManager::enableImmediateRendering = true;
bool ImGuiManager::enableDepthPass = true;
int ImGuiManager::bboxVisualizationMode = 0;
bool ImGuiManager::disableHiZCulling = false;

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
bool ImGuiManager::hiZShowRaycastGrid = false;
int ImGuiManager::hiZRaycastStep = 2;

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
bool ImGuiManager::debugKeysEnabled = false;

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
        ImGui::Checkbox("Enable Debug Hotkeys (F11/U/Y/L/F5/F6/K/O)", &debugKeysEnabled);
        ImGui::Separator();

        ImGui::Text("Recording/Replay System (Scene 0)");
        ImGui::Separator();

        // Pass enable/disable controls
        ImGui::Checkbox("Enable Recording", &enableRecording);
        ImGui::SameLine();
        ImGui::Checkbox("Enable Replay", &enableReplay);
        ImGui::Checkbox("Enable Immediate Rendering (Scene 1+)", &enableImmediateRendering);
        ImGui::Checkbox("Enable Depth Pass (recordMW)", &enableDepthPass);
        ImGui::Checkbox("Disable Hi-Z Culling (terrain hole diagnosis)", &disableHiZCulling);

        ImGui::Separator();
        ImGui::Text("Optimization Modes");
        ImGui::Checkbox("Performance Mode (Dirty Tracking)", &performanceMode);
        ImGui::SetItemTooltip("Skip redundant GPU state updates for unchanged meshes between frames");
        ImGui::Checkbox("State Leak Detection (Heavy)", &stateLeakDetection);
        ImGui::SetItemTooltip("Query device state before each draw to detect state leaks. Very slow!");

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

        // Depth buffer stats
        ImGui::Text("Depth Buffer (recordMW): %d geometries", debugRecordMWSize);
        ImGui::Text("Scene 1+ Immediate Renders: %d", debugImmediateCount);

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

        ImGui::Text("Scene 1+ (First Person / Alpha):");
        ImGui::Checkbox("1P Skinning (Hands)##s1", &suppress1PSkinning);
        ImGui::SameLine(); ImGui::Text("= %d", disp.firstPersonSkinning);

        ImGui::Checkbox("1P Alpha (Sorted/Weather)##s1", &suppress1PAlpha);
        ImGui::SameLine(); ImGui::Text("= %d", disp.firstPersonAlpha);

        ImGui::Checkbox("1P Other##s1", &suppress1POther);
        ImGui::SameLine(); ImGui::Text("= %d", disp.firstPersonOther);

        ImGui::Text("Pass-through:");
        ImGui::Checkbox("Offscreen (Map/Inv)##pt", &suppressOffscreen);
        ImGui::SameLine(); ImGui::Text("= %d", disp.offscreen);

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

// Debug control getters
bool ImGuiManager::GetEnableRecording() { return enableRecording; }
bool ImGuiManager::GetEnableReplay() { return enableReplay; }
bool ImGuiManager::GetEnableImmediateRendering() { return enableImmediateRendering; }
bool ImGuiManager::GetEnableDepthPass() { return enableDepthPass; }
int ImGuiManager::GetBBoxVisualizationMode() { return bboxVisualizationMode; }
bool ImGuiManager::GetDisableHiZCulling() { return disableHiZCulling; }

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

// Frame event log
void ImGuiManager::LogFrameEvent(FrameEvent::Type type, int sceneNum, int primCount) {
    frameEvents.push_back({type, sceneNum, primCount});
}

void ImGuiManager::SnapshotFrameEvents() {
    // Check auto-freeze trigger before swapping
    if (eventLogAutoFreeze && !eventLogFrozen && eventLogFreezeOffscreenThreshold > 0) {
        int offscreenCount = 0;
        for (const auto& e : frameEvents) {
            if (e.type == FrameEvent::DIP_Offscreen) offscreenCount++;
        }
        if (offscreenCount >= eventLogFreezeOffscreenThreshold) {
            frozenFrameEvents = frameEvents;  // Copy before swap
            eventLogFrozen = true;
        }
    }

    if (!eventLogFrozen) {
        displayFrameEvents.swap(frameEvents);
    }
    // Always clear the accumulator for next frame
    frameEvents.clear();
    frameEvents.reserve(256);
}

void ImGuiManager::ToggleFrameEventLog() {
    showFrameEventLog = !showFrameEventLog;

    ImGuiIO& io = ImGui::GetIO();
    io.MouseDrawCursor = showDebugInterface || showPCFInterface || showHiZInterface || showFrameEventLog;
}

void ImGuiManager::RenderFrameEventLog() {
    ImGui::SetNextWindowPos(ImVec2(10, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(430, 550), ImGuiCond_FirstUseEver);

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
                eventLogFrozen = true;
            }
        }

        ImGui::Checkbox("Auto-freeze on offscreen", &eventLogAutoFreeze);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::InputInt("##offThresh", &eventLogFreezeOffscreenThreshold);
        if (eventLogFreezeOffscreenThreshold < 0) eventLogFreezeOffscreenThreshold = 0;

        // Print to file button
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

            // Color lookup for event types
            for (int i = 0; i < (int)events.size(); i++) {
                const auto& e = events[i];
                ImVec4 color(1.0f, 1.0f, 1.0f, 1.0f);

                // Color-code by category
                bool isMGEInternal = (e.type >= FrameEvent::MGE_ShadowMap && e.type <= FrameEvent::MGE_SkyRender);
                if (e.type >= FrameEvent::DIP_Sky && e.type <= FrameEvent::DIP_Water) {
                    color = ImVec4(0.8f, 1.0f, 0.8f, 1.0f);  // green for DIP
                } else if (isMGEInternal) {
                    color = ImVec4(1.0f, 0.7f, 0.5f, 1.0f);  // orange for MGE internal rendering
                } else if (e.type >= FrameEvent::MGE_Stage0 && e.type <= FrameEvent::MGE_HLSLReplay) {
                    color = ImVec4(0.8f, 0.8f, 1.0f, 1.0f);  // blue for MGE stage markers
                } else if (e.type == FrameEvent::BeginScene || e.type == FrameEvent::EndScene) {
                    color = ImVec4(1.0f, 1.0f, 0.6f, 1.0f);  // yellow for scene boundaries
                }

                ImGui::PushStyleColor(ImGuiCol_Text, color);
                if (isMGEInternal && e.primCount > 0) {
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

    // Write to MGE XE log directory
    FILE* f = fopen("frame_events.txt", "w");
    if (!f) return;

    fprintf(f, "Frame Event Log (%d events)\n", (int)events.size());
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
            fprintf(f, "  %-20s %d\n", FrameEvent::typeName((FrameEvent::Type)i), counts[i]);
        }
    }
    fprintf(f, "  Total primitives: %d\n\n", totalPrims);

    // Ordered event list
    fprintf(f, "Events (ordered):\n");
    fprintf(f, "----------------------------------------\n");
    for (int i = 0; i < (int)events.size(); i++) {
        const auto& e = events[i];
        bool isMGEInternal = (e.type >= FrameEvent::MGE_ShadowMap && e.type <= FrameEvent::MGE_SkyRender);
        if (isMGEInternal && e.primCount > 0) {
            fprintf(f, "[%3d] S%d %-20s draws=%d\n", i, e.sceneNum, FrameEvent::typeName(e.type), e.primCount);
        } else if (e.primCount > 0) {
            fprintf(f, "[%3d] S%d %-20s prims=%d\n", i, e.sceneNum, FrameEvent::typeName(e.type), e.primCount);
        } else {
            fprintf(f, "[%3d] S%d %s\n", i, e.sceneNum, FrameEvent::typeName(e.type));
        }
    }

    fclose(f);
    LOG::logline(">> Frame events written to frame_events.txt (%d events)", (int)events.size());
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
        auto& culler = FixedFunctionShader::softwareOcclusionCuller;

        if (culler.getHiZTexture()) {
            int maxMipLevel = culler.getHiZMipLevels() - 1;

            // Display controls
            ImGui::Text("Visualization Settings");
            ImGui::SliderInt("Mip Level", &hiZDisplayMip, 0, maxMipLevel);
            ImGui::Checkbox("Invert Depth", &hiZInvert);

            ImGui::Separator();
            ImGui::Text("Raycast Grid Overlay");
            ImGui::Checkbox("Show Raycast Grid", &hiZShowRaycastGrid);
            if (hiZShowRaycastGrid) {
                ImGui::SliderInt("Grid Step Size", &hiZRaycastStep, 1, 8);
                ImGui::Text("(Red dots show occlusion test sample points)");
            }

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

