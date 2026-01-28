#include "imgui_manager.h"
#include "support/log.h"
#include "configuration.h"
#include "ffeshader.h"

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
bool ImGuiManager::enableLightProcessing = true;
int ImGuiManager::bboxVisualizationMode = 0;

// Debug stats
int ImGuiManager::debugRecordedCalls = 0;
int ImGuiManager::debugRenderedCalls = 0;
int ImGuiManager::debugCulledCalls = 0;
int ImGuiManager::debugSceneLights = 0;
int ImGuiManager::debugVisibleLights = 0;
int ImGuiManager::debugRecordMWSize = 0;
int ImGuiManager::debugImmediateCount = 0;

// Hi-Z visualization variables
int ImGuiManager::hiZDisplayMip = 0;
float ImGuiManager::hiZBrightness = 1.0f;
float ImGuiManager::hiZGamma = 2.2f;
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

// Rasterize All mode - bypass all heuristics
bool ImGuiManager::rasterizeAll = false;

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

void ImGuiManager::RenderDebugInterface() {
    ImGui::SetNextWindowPos(ImVec2(400, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(450, 400), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("HLSL Pipeline Debug", &showDebugInterface, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Recording/Replay System (Scene 0)");
        ImGui::Separator();

        // Pass enable/disable controls
        ImGui::Checkbox("Enable Recording", &enableRecording);
        ImGui::SameLine();
        ImGui::Checkbox("Enable Replay", &enableReplay);
        ImGui::Checkbox("Enable Immediate Rendering (Scene 1+)", &enableImmediateRendering);
        ImGui::Checkbox("Enable Depth Pass (recordMW)", &enableDepthPass);
        ImGui::Checkbox("Enable Light Processing", &enableLightProcessing);

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
        ImGui::Text("  Visible: %d", debugVisibleLights);
        ImGui::Text("  Culled: %d", debugSceneLights - debugVisibleLights);

        ImGui::Separator();

        // Depth buffer stats
        ImGui::Text("Depth Buffer (recordMW): %d geometries", debugRecordMWSize);
        ImGui::Text("Scene 1+ Immediate Renders: %d", debugImmediateCount);

        ImGui::Separator();
        ImGui::Text("Bounding Box Visualization");

        const char* bboxModes[] = { "OFF", "Objects (Green=Rendered, Red=Culled)", "Lights (Green=Visible, Red=Culled)" };
        ImGui::Combo("BBox Mode", &bboxVisualizationMode, bboxModes, 3);

        ImGui::Separator();
        ImGui::Text("Press G to toggle this interface");
    }
    ImGui::End();

    // Update mouse cursor visibility based on interface state
    ImGuiIO& io = ImGui::GetIO();
    io.MouseDrawCursor = showDebugInterface || showPCFInterface || showHiZInterface;
}

void ImGuiManager::ToggleDebugInterface() {
    showDebugInterface = !showDebugInterface;

    // Update mouse cursor visibility
    ImGuiIO& io = ImGui::GetIO();
    io.MouseDrawCursor = showDebugInterface || showPCFInterface || showHiZInterface;
}

// Debug control getters
bool ImGuiManager::GetEnableRecording() { return enableRecording; }
bool ImGuiManager::GetEnableReplay() { return enableReplay; }
bool ImGuiManager::GetEnableImmediateRendering() { return enableImmediateRendering; }
bool ImGuiManager::GetEnableDepthPass() { return enableDepthPass; }
bool ImGuiManager::GetEnableLightProcessing() { return enableLightProcessing; }
int ImGuiManager::GetBBoxVisualizationMode() { return bboxVisualizationMode; }

void ImGuiManager::UpdateDebugStats(int recordedCalls, int renderedCalls, int culledCalls,
                                     int sceneLights, int visibleLights, int recordMWSize, int immediateCount) {
    debugRecordedCalls = recordedCalls;
    debugRenderedCalls = renderedCalls;
    debugCulledCalls = culledCalls;
    debugSceneLights = sceneLights;
    debugVisibleLights = visibleLights;
    debugRecordMWSize = recordMWSize;
    debugImmediateCount = immediateCount;
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
            ImGui::SliderFloat("Brightness", &hiZBrightness, 0.1f, 10.0f, "%.2f");
            ImGui::SliderFloat("Gamma", &hiZGamma, 0.5f, 4.0f, "%.2f");
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
    io.MouseDrawCursor = showDebugInterface || showPCFInterface || showHiZInterface;
}

void ImGuiManager::ToggleHiZInterface() {
    showHiZInterface = !showHiZInterface;

    // Update mouse cursor visibility
    ImGuiIO& io = ImGui::GetIO();
    io.MouseDrawCursor = showDebugInterface || showPCFInterface || showHiZInterface;
}

bool ImGuiManager::GetShowHiZInterface() {
    return showHiZInterface;
}

