#include "imgui_manager.h"
#include "support/log.h"
#include "configuration.h"

bool ImGuiManager::initialized = false;
bool ImGuiManager::showDemo = false;
bool ImGuiManager::showPCFInterface = true;
HWND ImGuiManager::windowHandle = nullptr;

// PCF filtering variables
float ImGuiManager::pcfFilterRadius = 2.0f;
int ImGuiManager::pcfSampleCount = 16;
float ImGuiManager::pcfBiasConstant = 0.001f;
float ImGuiManager::pcfBiasSlope = 1.0f;

bool ImGuiManager::Initialize(HWND hwnd, IDirect3DDevice9* device) {
    if (initialized) {
        return true;
    }

    // Store window handle for mouse input polling
    windowHandle = hwnd;

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
    
    if (ImGui::Begin("PCF Shadow Filtering", &showPCFInterface, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Shadow PCF Filtering Controls");
        ImGui::Separator();

        // Filter radius control
        if (ImGui::SliderFloat("Filter Radius", &pcfFilterRadius, 0.1f, 10.0f, "%.2f")) {
            LOG::logline(">> PCF Filter Radius: %.2f", pcfFilterRadius);
        }

        // Sample count control
        if (ImGui::SliderInt("Sample Count", &pcfSampleCount, 4, 64)) {
            LOG::logline(">> PCF Sample Count: %d", pcfSampleCount);
        }

        // Bias controls
        if (ImGui::SliderFloat("Bias Constant", &pcfBiasConstant, 0.0f, 0.01f, "%.4f")) {
            LOG::logline(">> PCF Bias Constant: %.4f", pcfBiasConstant);
        }

        if (ImGui::SliderFloat("Bias Slope", &pcfBiasSlope, 0.1f, 5.0f, "%.2f")) {
            LOG::logline(">> PCF Bias Slope: %.2f", pcfBiasSlope);
        }

        ImGui::Separator();
        
        // Preset buttons
        if (ImGui::Button("Soft Shadows")) {
            pcfFilterRadius = 3.0f;
            pcfSampleCount = 24;
            pcfBiasConstant = 0.002f;
            pcfBiasSlope = 1.5f;
            LOG::logline(">> Applied Soft Shadows preset");
        }
        ImGui::SameLine();
        
        if (ImGui::Button("Sharp Shadows")) {
            pcfFilterRadius = 1.0f;
            pcfSampleCount = 8;
            pcfBiasConstant = 0.0005f;
            pcfBiasSlope = 0.8f;
            LOG::logline(">> Applied Sharp Shadows preset");
        }

        if (ImGui::Button("High Quality")) {
            pcfFilterRadius = 2.5f;
            pcfSampleCount = 36;
            pcfBiasConstant = 0.0015f;
            pcfBiasSlope = 1.2f;
            LOG::logline(">> Applied High Quality preset");
        }
        ImGui::SameLine();
        
        if (ImGui::Button("Performance")) {
            pcfFilterRadius = 1.5f;
            pcfSampleCount = 12;
            pcfBiasConstant = 0.001f;
            pcfBiasSlope = 1.0f;
            LOG::logline(">> Applied Performance preset");
        }

        ImGui::Separator();
        ImGui::Checkbox("Show Demo Window", &showDemo);
        
        ImGui::Text("Press F11 to toggle this interface");
    }
    ImGui::End();
}

// Getter functions for shader constants
float ImGuiManager::GetPCFFilterRadius() { return pcfFilterRadius; }
int ImGuiManager::GetPCFSampleCount() { return pcfSampleCount; }
float ImGuiManager::GetPCFBiasConstant() { return pcfBiasConstant; }
float ImGuiManager::GetPCFBiasSlope() { return pcfBiasSlope; }
bool ImGuiManager::GetShowPCFInterface() { return showPCFInterface; }

void ImGuiManager::TogglePCFInterface() { showPCFInterface = !showPCFInterface; }

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