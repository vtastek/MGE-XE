#pragma once

// Minimal Dear ImGui bring-up for the in-game dev panels. Self-contained: owns the
// ImGui context + DX9/Win32 backends and draws the F10 overlay. Panel bodies live
// where their state does (DrawForgeDevPanel in renderprocess.cpp).
//
// Was imgui_water.h: the only panel used to be Water Flow / Foam, which S3 deleted
// along with MGE's water renderer.

struct IDirect3DDevice9;

namespace ImGuiPanels {
    // Call once per Present (after the scene is drawn, before the real Present). Lazily
    // initializes on the first call, polls the toggle hotkey + mouse, and renders the panels.
    void onPresent(IDirect3DDevice9* device);
    // True while the overlay is open (so the caller can suppress game mouse-look if it wants).
    bool wantMouse();
}
