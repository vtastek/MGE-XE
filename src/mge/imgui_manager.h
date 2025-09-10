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
    static float GetPCFFilterSize();
    static float GetPCFPenumbraScale(); 
    static float GetPCFMinPenumbra();
    static float GetPCFMaxPenumbra();
    static float GetPCFBias();
    static float GetPCFBias2();
    static float GetPCFSlopeBias();
    static bool GetShowPCFInterface();
    static void TogglePCFInterface();

private:
    static bool initialized;
    static bool showDemo;
    static bool showPCFInterface;
    static HWND windowHandle;
    
    // PCF filtering variables for tweaking
    static float pcfFilterSize;        // Base filter size in texels (maps to PCF_filterSize)
    static float pcfPenumbraScale;     // Scale factor for distance-based penumbra  
    static float pcfMinPenumbra;       // Minimum penumbra size
    static float pcfMaxPenumbra;       // Maximum penumbra size
    static float pcfBias;              // Depth bias to prevent acne
    static float pcfBias2;             // Second depth bias for lerp
    static float pcfSlopeBias;         // Slope-based bias to prevent acne on angled surfaces
};