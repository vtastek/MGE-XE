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
    static LRESULT HandleWindowMessage(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    static void OnLostDevice();
    static void OnResetDevice();

    // Getters for PCF parameters
    static float GetPCFFilterRadius();
    static int GetPCFSampleCount();
    static float GetPCFBiasConstant();
    static float GetPCFBiasSlope();
    static bool GetShowPCFInterface();
    static void TogglePCFInterface();

private:
    static bool initialized;
    static bool showDemo;
    static bool showPCFInterface;
    static HWND windowHandle;
    
    // PCF filtering variables for tweaking
    static float pcfFilterRadius;
    static int pcfSampleCount;
    static float pcfBiasConstant;
    static float pcfBiasSlope;
};