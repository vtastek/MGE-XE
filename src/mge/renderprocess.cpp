#include "renderprocess.h"
#include "configuration.h"
#include "ipc/client.h"
#include "support/log.h"

#include <windows.h>

namespace {
    // Fixed offscreen target size for the spike (Milestone A). Small enough to keep
    // the CPU staging copy cheap; large enough to read the triangle clearly.
    constexpr UINT kW = 640;
    constexpr UINT kH = 360;

    IPC::Client* g_client = nullptr;
    bool   g_initOk  = false;
    bool   g_enabled = false;          // F11 live toggle for the per-frame blit
    HANDLE g_fbHandle = nullptr;       // file-mapping handle (this process) of the host framebuffer
    const void* g_fbPtr = nullptr;     // mapped W*H*4 BGRA pixels
    IDirect3DSurface9* g_surf = nullptr;
    IDirect3DDevice9*  g_surfDevice = nullptr;
    unsigned g_frame = 0;

    // Rolling round-trip stats, logged every 60 presented spike frames.
    double g_sumRoundtripMs = 0.0;
    double g_sumHostMs = 0.0;
    int    g_statSamples = 0;

    inline double msPerTick() {
        static const double v = [] {
            LARGE_INTEGER f; QueryPerformanceFrequency(&f);
            return 1000.0 / double(f.QuadPart);
        }();
        return v;
    }

    void releaseSurface() {
        if (g_surf) { g_surf->Release(); g_surf = nullptr; }
        g_surfDevice = nullptr;
    }

    bool ensureSurface(IDirect3DDevice9* device) {
        if (g_surf && g_surfDevice == device) {
            return true;
        }
        releaseSurface();
        // Offscreen plain surface in DEFAULT pool: lockable for the CPU upload and a
        // valid StretchRect source onto the backbuffer.
        HRESULT hr = device->CreateOffscreenPlainSurface(
            kW, kH, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT, &g_surf, nullptr);
        if (FAILED(hr)) {
            static bool logged = false;
            if (!logged) { LOG::logline("!! [spike] CreateOffscreenPlainSurface failed 0x%x", hr); logged = true; }
            g_surf = nullptr;
            return false;
        }
        g_surfDevice = device;
        return true;
    }
}

namespace RenderProcess {
    void init(IPC::Client* client) {
        if (!Configuration.UseRenderProcess) {
            return;
        }
        g_client = client;

        HANDLE fb = nullptr;
        if (!g_client->renderInitBlocking(kW, kH, &fb) || fb == nullptr) {
            LOG::logline("!! [spike] renderInit RPC failed; spike disabled");
            return;
        }
        g_fbHandle = fb;
        g_fbPtr = MapViewOfFile(g_fbHandle, FILE_MAP_READ, 0, 0, kW * kH * 4);
        if (g_fbPtr == nullptr) {
            LOG::winerror("[spike] failed to map host framebuffer; spike disabled");
            CloseHandle(g_fbHandle);
            g_fbHandle = nullptr;
            return;
        }

        g_initOk = true;
        LOG::logline(">> [spike] render process ready (%ux%u). F11 toggles the corner-quad composite.", kW, kH);
    }

    void onPresent(IDirect3DDevice9* device) {
        if (!g_initOk || !device) {
            return;
        }

        // Live toggle (debug key). Edge-triggered.
        if (GetAsyncKeyState(VK_F11) & 0x0001) {
            g_enabled = !g_enabled;
            LOG::logline(">> [spike] composite %s", g_enabled ? "ON" : "OFF");
        }
        if (!g_enabled) {
            return;
        }

        LARGE_INTEGER t0; QueryPerformanceCounter(&t0);

        // Drive the host renderer. Blocking: render, readback, into our mapped blob.
        double hostMs = 0.0;
        if (!g_client->renderFrameBlocking(g_frame++, &hostMs)) {
            return;
        }

        if (!ensureSurface(device)) {
            return;
        }

        // Upload the host pixels into the offscreen surface (row-by-row for pitch).
        D3DLOCKED_RECT lr;
        if (FAILED(g_surf->LockRect(&lr, nullptr, 0))) {
            return;
        }
        const char* src = static_cast<const char*>(g_fbPtr);
        char* dst = static_cast<char*>(lr.pBits);
        const UINT rowBytes = kW * 4;
        for (UINT y = 0; y < kH; ++y) {
            memcpy(dst + size_t(y) * lr.Pitch, src + size_t(y) * rowBytes, rowBytes);
        }
        g_surf->UnlockRect();

        // Composite as a corner quad (top-left, 1:1 — no stretch/filter so the copy
        // stays format-compatible across DXVK paths). Non-destructive to the frame.
        IDirect3DSurface9* backbuffer = nullptr;
        if (SUCCEEDED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backbuffer)) && backbuffer) {
            RECT dstRect = { 20, 20, 20 + (LONG)kW, 20 + (LONG)kH };
            HRESULT hr = device->StretchRect(g_surf, nullptr, backbuffer, &dstRect, D3DTEXF_NONE);
            if (FAILED(hr)) {
                static bool logged = false;
                if (!logged) { LOG::logline("!! [spike] StretchRect failed 0x%x", hr); logged = true; }
            }
            backbuffer->Release();
        }

        LARGE_INTEGER t1; QueryPerformanceCounter(&t1);
        g_sumRoundtripMs += double(t1.QuadPart - t0.QuadPart) * msPerTick();
        g_sumHostMs += hostMs;
        if (++g_statSamples >= 60) {
            LOG::logline("-- [spike] avg/60: client roundtrip=%.3f ms (host render+readback=%.3f ms)",
                g_sumRoundtripMs / g_statSamples, g_sumHostMs / g_statSamples);
            g_sumRoundtripMs = g_sumHostMs = 0.0;
            g_statSamples = 0;
        }
    }

    void shutdown() {
        releaseSurface();
        if (g_fbPtr)    { UnmapViewOfFile(g_fbPtr); g_fbPtr = nullptr; }
        if (g_fbHandle) { CloseHandle(g_fbHandle); g_fbHandle = nullptr; }
        g_initOk = false;
        g_enabled = false;
    }
}
