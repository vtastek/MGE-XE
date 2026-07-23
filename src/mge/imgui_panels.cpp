#include "imgui_panels.h"

#include "imgui.h"
#include "imgui_impl_dx9.h"
#include "imgui_impl_win32.h"
#include "proxydx/d3d9header.h"
#include "support/log.h"

#include <windows.h>

// Forge seam / dev toggles — defined in renderprocess.cpp (reaches the seam state).
// Clickable replacement for the hardware-key A/B toggles. This was imgui_water.cpp and
// carried DrawFlowFoamPanel too, until S3 deleted MGE's water renderer under it; the
// ImGui bring-up itself was never water-specific, so the module was renamed rather
// than deleted.
extern void DrawForgeDevPanel();

namespace {
    bool g_init    = false;
    bool g_visible = false;
    HWND g_hwnd    = nullptr;
}

namespace ImGuiPanels {

void onPresent(IDirect3DDevice9* device) {
    if (!device)
        return;

    if (!g_init) {
        // Present's HWND arg is null in this engine; pull the focus window from the device.
        D3DDEVICE_CREATION_PARAMETERS cp;
        if (FAILED(device->GetCreationParameters(&cp)) || !cp.hFocusWindow)
            return;
        g_hwnd = cp.hFocusWindow;

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.MouseDrawCursor = true;                 // engine hides the OS cursor; ImGui draws its own
        ImGui::StyleColorsDark();
        if (!ImGui_ImplWin32_Init(g_hwnd) || !ImGui_ImplDX9_Init(device)) {
            LOG::logline("!! [imgui] backend init failed");
            return;
        }
        g_init = true;
        LOG::logline(">> [imgui] dev panels ready (F10 toggles)");
    }

    // F10 toggles the panel.
    if (GetAsyncKeyState(VK_F10) & 0x0001)
        g_visible = !g_visible;

    // Manual mouse poll (we do not subclass the engine's WndProc; mouse drag is enough for sliders).
    ImGuiIO& io = ImGui::GetIO();
    POINT p;
    if (GetCursorPos(&p) && ScreenToClient(g_hwnd, &p))
        io.MousePos = ImVec2((float)p.x, (float)p.y);
    io.MouseDown[0] = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
    io.MouseDown[1] = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;

    ImGui_ImplDX9_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    if (g_visible) {
        DrawForgeDevPanel();
    }

    ImGui::EndFrame();
    ImGui::Render();
    if (g_visible)
        ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());
}

bool wantMouse() {
    return g_init && g_visible && ImGui::GetIO().WantCaptureMouse;
}

} // namespace ImGuiPanels
