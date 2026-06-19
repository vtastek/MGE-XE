#include "renderprocess.h"
#include "configuration.h"
#include "ipc/client.h"
#include "support/log.h"

#include <windows.h>

namespace {
    // Fixed offscreen target size for the spike. Small enough to keep the (A-path) CPU
    // staging copy cheap; large enough to read the triangle clearly.
    constexpr UINT kW = 640;
    constexpr UINT kH = 360;

    IPC::Client* g_client = nullptr;
    bool   g_pendingInit = false;      // do the RPC + resource setup on the first onPresent (needs the device)
    bool   g_initOk  = false;
    bool   g_enabled = false;          // F11 live toggle for the per-frame composite
    bool   g_useShared = false;        // B path: host renders into a shared GPU texture (zero-copy)
    unsigned g_frame = 0;

    // B/C path (zero-copy): double-buffered D3D9Ex shared render targets. The host renders
    // into buf[N&1] via Vulkan while we composite buf[(N-1)&1] (rendered & GPU-completed last
    // frame) — removes the write/read race on a single texture.
    IDirect3DTexture9* g_sharedTex[2] = { nullptr, nullptr };
    HANDLE g_sharedHandle[2] = { nullptr, nullptr };

    // A path (CPU staging): host writes pixels into a shared mapping; we upload to an
    // offscreen surface and blit that.
    HANDLE g_fbHandle = nullptr;
    const void* g_fbPtr = nullptr;
    IDirect3DSurface9* g_surf = nullptr;
    IDirect3DDevice9*  g_surfDevice = nullptr;

    // Rolling round-trip stats, logged every 60 presented spike frames.
    double g_sumRoundtripMs = 0.0;
    double g_sumHostMs = 0.0;
    int    g_statSamples = 0;

    inline double msPerTick() {
        static const double v = [] {
            LARGE_INTEGER f; QueryPerformanceCounter(&f);
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

    // First-present setup: pick the transport (B if the device is D3D9Ex, else A) and
    // run the RenderInit RPC. Needs the live device, hence lazy.
    void lazyInit(IDirect3DDevice9* device) {
        // B/C path: on a D3D9Ex device, create TWO shared render-target textures for the host
        // to render into (double-buffered). CreateTexture's pSharedHandle out-param yields a
        // KMT/global handle the 64-bit Vulkan side imports as external memory.
        IDirect3DDevice9Ex* exDev = nullptr;
        if (SUCCEEDED(device->QueryInterface(__uuidof(IDirect3DDevice9Ex), reinterpret_cast<void**>(&exDev))) && exDev) {
            bool ok = true;
            for (int i = 0; i < 2 && ok; ++i) {
                HANDLE sh = nullptr;
                HRESULT hr = exDev->CreateTexture(kW, kH, 1, D3DUSAGE_RENDERTARGET,
                                                  D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &g_sharedTex[i], &sh);
                if (SUCCEEDED(hr) && g_sharedTex[i] && sh) {
                    g_sharedHandle[i] = sh;
                } else {
                    LOG::logline("!! [spike] shared RT CreateTexture %d failed 0x%08X; using CPU staging path", i, hr);
                    ok = false;
                }
            }
            exDev->Release();
            if (ok) {
                g_useShared = true;
                LOG::logline(">> [spike] created 2 shared RT textures (%p / %p) for zero-copy double-buffered GPU share",
                    g_sharedHandle[0], g_sharedHandle[1]);
            } else {
                for (int i = 0; i < 2; ++i) {
                    if (g_sharedTex[i]) { g_sharedTex[i]->Release(); g_sharedTex[i] = nullptr; }
                    g_sharedHandle[i] = nullptr;
                }
            }
        }

        HANDLE fb = nullptr;
        if (!g_client->renderInitBlocking(kW, kH, g_sharedHandle[0], g_sharedHandle[1], &fb)) {
            LOG::logline("!! [spike] renderInit RPC failed; spike disabled");
            return;
        }

        if (!g_useShared) {
            // A path: map the host's CPU framebuffer for the upload-and-blit composite.
            if (fb == nullptr) {
                LOG::logline("!! [spike] no framebuffer handle returned; spike disabled");
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
        }

        g_initOk = true;
        LOG::logline(">> [spike] render process ready (%ux%u, %s). F11 toggles the composite.",
            kW, kH, g_useShared ? "GPU shared texture / zero-copy" : "CPU staging");
    }
}

namespace RenderProcess {
    void init(IPC::Client* client) {
        if (!Configuration.UseRenderProcess) {
            return;
        }
        g_client = client;
        g_pendingInit = true;   // finish setup on the first onPresent, when the device is available
    }

    void onPresent(IDirect3DDevice9* device) {
        if (!device || (!g_pendingInit && !g_initOk)) {
            return;
        }

        if (g_pendingInit) {
            g_pendingInit = false;
            lazyInit(device);
        }
        if (!g_initOk) {
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

        const unsigned frame = g_frame++;
        const unsigned writeIdx = frame & 1u;        // host renders into this buffer
        const unsigned readIdx  = writeIdx ^ 1u;     // we composite the one rendered last frame

        // Drive the host renderer. B/C: render into buf[writeIdx]; A: single buffer (0).
        double hostMs = 0.0;
        if (!g_client->renderFrameBlocking(frame, g_useShared ? writeIdx : 0, &hostMs)) {
            return;
        }

        IDirect3DSurface9* src = nullptr;   // the surface we blit onto the backbuffer
        bool releaseSrc = false;

        if (g_useShared) {
            // Composite buf[readIdx] — rendered & GPU-completed during the previous frame, so
            // it never collides with this frame's render into buf[writeIdx]. The first frame
            // has no prior buffer yet, so just prime the pipeline (render only).
            if (frame == 0) {
                return;
            }
            if (FAILED(g_sharedTex[readIdx]->GetSurfaceLevel(0, &src)) || !src) {
                return;
            }
            releaseSrc = true;
        } else {
            // CPU staging: upload host pixels into the offscreen surface (row-by-row for pitch).
            if (!ensureSurface(device)) {
                return;
            }
            D3DLOCKED_RECT lr;
            if (FAILED(g_surf->LockRect(&lr, nullptr, 0))) {
                return;
            }
            const char* sp = static_cast<const char*>(g_fbPtr);
            char* dp = static_cast<char*>(lr.pBits);
            const UINT rowBytes = kW * 4;
            for (UINT y = 0; y < kH; ++y) {
                memcpy(dp + size_t(y) * lr.Pitch, sp + size_t(y) * rowBytes, rowBytes);
            }
            g_surf->UnlockRect();
            src = g_surf;
        }

        // Composite as a 1:1 corner quad (top-left), non-destructive to the frame.
        IDirect3DSurface9* backbuffer = nullptr;
        if (SUCCEEDED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backbuffer)) && backbuffer) {
            RECT dstRect = { 20, 20, 20 + (LONG)kW, 20 + (LONG)kH };
            HRESULT hr = device->StretchRect(src, nullptr, backbuffer, &dstRect, D3DTEXF_NONE);
            if (FAILED(hr)) {
                static bool logged = false;
                if (!logged) { LOG::logline("!! [spike] StretchRect failed 0x%x", hr); logged = true; }
            }
            backbuffer->Release();
        }
        if (releaseSrc && src) {
            src->Release();
        }

        LARGE_INTEGER t1; QueryPerformanceCounter(&t1);
        g_sumRoundtripMs += double(t1.QuadPart - t0.QuadPart) * msPerTick();
        g_sumHostMs += hostMs;
        if (++g_statSamples >= 60) {
            LOG::logline("-- [spike] avg/60: client roundtrip=%.3f ms (host render%s=%.3f ms) [%s]",
                g_sumRoundtripMs / g_statSamples,
                g_useShared ? "" : "+readback",
                g_sumHostMs / g_statSamples,
                g_useShared ? "zero-copy" : "CPU staging");
            g_sumRoundtripMs = g_sumHostMs = 0.0;
            g_statSamples = 0;
        }
    }

    void shutdown() {
        releaseSurface();
        for (int i = 0; i < 2; ++i) {
            if (g_sharedTex[i]) { g_sharedTex[i]->Release(); g_sharedTex[i] = nullptr; }
            g_sharedHandle[i] = nullptr;
        }
        if (g_fbPtr)    { UnmapViewOfFile(g_fbPtr); g_fbPtr = nullptr; }
        if (g_fbHandle) { CloseHandle(g_fbHandle); g_fbHandle = nullptr; }
        g_initOk = false;
        g_enabled = false;
    }
}
