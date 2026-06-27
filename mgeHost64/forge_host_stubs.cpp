// Headless-host stubs for Forge app-layer symbols referenced by the vendored UI.cpp /
// FontSystem.cpp but NOT linked by the host. The host owns no window (renders into a shared
// texture DXVK composites into MW's D3D9 window) and links neither the OS window layer nor the
// GPU profiler. So:
//   - getActiveMonitorIdx / getMonitorDpiScale  (OS window layer, extern "C") -> identity DPI.
//   - updateProfilerUI                          (profiler, declared inline in UI.cpp)    -> no-op.
// Input is bridged via FORGE_HOST_EXTERNAL_INPUT (UI.cpp), so InputFillImguiKeyMap is compiled out
// and gainput/WindowsInput need not be linked. When the host eventually owns the window, drop these
// stubs and link the real OS/profiler/input layer instead.
#include <cstdint>

extern "C"
{
    uint32_t getActiveMonitorIdx(void) { return 0; }
    void     getMonitorDpiScale(uint32_t /*monitorIndex*/, float dpiScale[2])
    {
        dpiScale[0] = 1.0f;
        dpiScale[1] = 1.0f;
    }
}

void updateProfilerUI() {}
