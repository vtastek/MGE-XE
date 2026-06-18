#pragma once

// Minimal Dear ImGui bring-up for the Water Flow / Foam tuning panel (scene-walk-v2).
// Self-contained: owns the ImGui context + DX9/Win32 backends and draws one window.
// The panel body lives in renderexterior.cpp (DrawFlowFoamPanel) where every flow/foam
// setting — DistantLand:: statics and the file-static g_flow* knobs — is reachable.

struct IDirect3DDevice9;

namespace ImGuiWater {
    // Call once per Present (after the scene is drawn, before the real Present). Lazily
    // initializes on the first call, polls the toggle hotkey + mouse, and renders the panel.
    void onPresent(IDirect3DDevice9* device);
    // True while the panel is open (so the caller can suppress game mouse-look if it wants).
    bool wantMouse();
}
