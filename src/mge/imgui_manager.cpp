#include "imgui_manager.h"
#include "support/log.h"
#include "configuration.h"

bool ImGuiManager::initialized = false;
bool ImGuiManager::showDemo = false;
bool ImGuiManager::showPCFInterface = true;
HWND ImGuiManager::windowHandle = nullptr;

// PCF filtering variables
float ImGuiManager::pcfFilterSize = 3.0f;        // Base filter size in texels
float ImGuiManager::pcfPenumbraScale = 1.0f;     // Scale factor for distance-based penumbra
float ImGuiManager::pcfMinPenumbra = 2.0f;       // Minimum penumbra size 
float ImGuiManager::pcfMaxPenumbra = 5.0f;       // Maximum penumbra size
float ImGuiManager::pcfBias = 0.0015f;           // Depth bias to prevent acne
float ImGuiManager::pcfBias2 = 0.0045f;          // Second depth bias for lerp

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
    

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    
    // Enable keyboard and gamepad controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    
    // Enable ImGui to draw its own cursor since we do manual mouse polling
    io.MouseDrawCursor = true;

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
        Configuration.SaveSettings();
    }
}

// Getter functions for shader constants
float ImGuiManager::GetPCFFilterSize() { return pcfFilterSize; }
float ImGuiManager::GetPCFPenumbraScale() { return pcfPenumbraScale; }
float ImGuiManager::GetPCFMinPenumbra() { return pcfMinPenumbra; }
float ImGuiManager::GetPCFMaxPenumbra() { return pcfMaxPenumbra; }
float ImGuiManager::GetPCFBias() { return pcfBias; }
float ImGuiManager::GetPCFBias2() { return pcfBias2; }
bool ImGuiManager::GetShowPCFInterface() { return showPCFInterface; }

void ImGuiManager::TogglePCFInterface() { 
    bool wasShowing = showPCFInterface;
    showPCFInterface = !showPCFInterface;
    
    // Save PCF settings when interface is being closed
    if (wasShowing && !showPCFInterface) {
        Configuration.PCF.FilterSize = pcfFilterSize;
        Configuration.PCF.PenumbraScale = pcfPenumbraScale;
        Configuration.PCF.MinPenumbra = pcfMinPenumbra;
        Configuration.PCF.MaxPenumbra = pcfMaxPenumbra;
        Configuration.PCF.Bias = pcfBias;
        Configuration.PCF.Bias2 = pcfBias2;
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

