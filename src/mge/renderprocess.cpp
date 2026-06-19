#include "renderprocess.h"
#include "configuration.h"
#include "ipc/client.h"
#include "support/log.h"

#include <windows.h>
#include <d3d12.h>
#include <d3d9on12.h>

namespace {
    // Fixed offscreen target size for the seam. Matches the host's Forge render
    // target. Small corner quad; non-destructive to the real frame.
    constexpr UINT kW = 640;
    constexpr UINT kH = 360;

    typedef HRESULT (_stdcall* D3DProc9On12Ex)(UINT, D3D9ON12_ARGS*, UINT, IDirect3D9Ex**);

    IPC::Client* g_client = nullptr;
    bool   g_pendingInit = false;      // do the RPC + resource setup on the first onPresent (needs the device)
    bool   g_initOk  = false;
    bool   g_enabled = false;          // F11 live toggle for the per-frame composite
    unsigned g_frame = 0;

    // --- Decoupled D3D9On12 seam ---
    // The GAME renders/presents on MW's native D3D9Ex MAIN device (unchanged, proven).
    // The seam uses a SEPARATE, dedicated D3D9On12 side-device so MGE's heavy D3D9
    // pipeline never goes through the 9On12 translation layer (which black-screens the
    // whole game). The side-device:
    //   - is D3D12-backed, so it can OpenSharedHandle the Forge host's shared D3D12 RT,
    //   - owns a D3D9 RT texture created as a D3D9<->D3D9 KMT SHARED texture,
    //   - each frame copies the host RT into that texture (Unwrap -> D3D12 copy -> Return).
    // The MAIN device opens the same KMT-shared texture and StretchRects it (D3D9<->D3D9
    // KMT sharing is a supported path — both ends are D3D9 on the same adapter).

    IDirect3DDevice9Ex*   g_mainDevice = nullptr;   // borrowed (not owned): MW's main device, for the blit

    IDirect3D9Ex*         g_seamFactory = nullptr;  // 9On12 factory (owned)
    IDirect3DDevice9Ex*   g_seamDevice  = nullptr;  // 9On12 side-device (owned)
    IDirect3DDevice9On12* g_dev9on12    = nullptr;  // QI of g_seamDevice
    ID3D12Device*         g_d3d12dev    = nullptr;  // side-device's underlying D3D12 device
    ID3D12Resource*       g_hostRT12    = nullptr;  // host's shared RT, opened in this process

    IDirect3DTexture9*    g_seamTex = nullptr;      // on g_seamDevice; KMT-shared; copy destination
    HANDLE                g_kmtHandle = nullptr;    // D3D9<->D3D9 KMT shared handle for g_seamTex
    IDirect3DTexture9*    g_mainTex = nullptr;      // on g_mainDevice; opened from g_kmtHandle; blit source

    // Side-device's D3D12 copy infrastructure (we own these).
    ID3D12CommandQueue*        g_copyQueue = nullptr;
    ID3D12CommandAllocator*    g_copyAlloc = nullptr;
    ID3D12GraphicsCommandList* g_copyList  = nullptr;
    ID3D12Fence*               g_copyFence = nullptr;
    UINT64                     g_fenceVal  = 0;
    HANDLE                     g_fenceEvent = nullptr;

    // Rolling round-trip stats, logged every 60 presented frames.
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

    void releaseAll() {
        if (g_mainTex)    { g_mainTex->Release();    g_mainTex = nullptr; }
        if (g_seamTex)    { g_seamTex->Release();    g_seamTex = nullptr; }
        g_kmtHandle = nullptr;   // owned by the textures
        if (g_copyList)   { g_copyList->Release();   g_copyList = nullptr; }
        if (g_copyAlloc)  { g_copyAlloc->Release();  g_copyAlloc = nullptr; }
        if (g_copyFence)  { g_copyFence->Release();  g_copyFence = nullptr; }
        if (g_copyQueue)  { g_copyQueue->Release();  g_copyQueue = nullptr; }
        if (g_fenceEvent) { CloseHandle(g_fenceEvent); g_fenceEvent = nullptr; }
        if (g_hostRT12)   { g_hostRT12->Release();   g_hostRT12 = nullptr; }
        if (g_d3d12dev)   { g_d3d12dev->Release();   g_d3d12dev = nullptr; }
        if (g_dev9on12)   { g_dev9on12->Release();   g_dev9on12 = nullptr; }
        if (g_seamDevice) { g_seamDevice->Release(); g_seamDevice = nullptr; }
        if (g_seamFactory){ g_seamFactory->Release();g_seamFactory = nullptr; }
        g_mainDevice = nullptr;  // borrowed
    }

    HWND deviceWindow(IDirect3DDevice9* device) {
        IDirect3DSwapChain9* sc = nullptr;
        if (SUCCEEDED(device->GetSwapChain(0, &sc)) && sc) {
            D3DPRESENT_PARAMETERS pp = {};
            HWND hwnd = (SUCCEEDED(sc->GetPresentParameters(&pp))) ? pp.hDeviceWindow : nullptr;
            sc->Release();
            if (hwnd) return hwnd;
        }
        return GetForegroundWindow();
    }

    // Create the dedicated D3D9On12 side-device (no rendering, never presented).
    bool createSeamDevice(HWND hwnd) {
        HMODULE d3d9 = GetModuleHandleA("d3d9.dll");
        D3DProc9On12Ex create9On12 = d3d9 ? (D3DProc9On12Ex)GetProcAddress(d3d9, "Direct3DCreate9On12Ex") : nullptr;
        if (!create9On12) {
            LOG::logline("!! [seam] Direct3DCreate9On12Ex not exported (DXVK or old runtime?) — seam disabled");
            return false;
        }
        D3D9ON12_ARGS args = {};
        args.Enable9On12 = TRUE;
        args.pD3D12Device = nullptr;   // 9On12 makes its own internal D3D12 device on the default adapter
        args.NumQueues = 0;
        if (FAILED(create9On12(D3D_SDK_VERSION, &args, 1, &g_seamFactory)) || !g_seamFactory) {
            LOG::logline("!! [seam] Direct3DCreate9On12Ex failed — seam disabled");
            return false;
        }

        D3DPRESENT_PARAMETERS pp = {};
        pp.Windowed = TRUE;
        pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
        pp.BackBufferWidth = 1;
        pp.BackBufferHeight = 1;
        pp.BackBufferFormat = D3DFMT_X8R8G8B8;
        pp.BackBufferCount = 1;
        pp.hDeviceWindow = hwnd;
        pp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
        HRESULT hr = g_seamFactory->CreateDeviceEx(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hwnd,
            D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE | D3DCREATE_MULTITHREADED,
            &pp, nullptr, &g_seamDevice);
        if (FAILED(hr) || !g_seamDevice) {
            LOG::logline("!! [seam] 9On12 CreateDeviceEx failed 0x%08X — seam disabled", hr);
            return false;
        }
        return true;
    }

    bool createCopyInfra() {
        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(g_d3d12dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&g_copyQueue))) return false;
        if (FAILED(g_d3d12dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                __uuidof(ID3D12CommandAllocator), (void**)&g_copyAlloc))) return false;
        if (FAILED(g_d3d12dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g_copyAlloc,
                nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&g_copyList))) return false;
        g_copyList->Close();
        if (FAILED(g_d3d12dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&g_copyFence))) return false;
        g_fenceEvent = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        return g_fenceEvent != nullptr;
    }

    // First-present setup: stand up the 9On12 side-device, run RenderInit, open the
    // host RT, build the KMT-shared blit texture (opened on both devices) + copy infra.
    void lazyInit(IDirect3DDevice9* device) {
        if (FAILED(device->QueryInterface(__uuidof(IDirect3DDevice9Ex), (void**)&g_mainDevice)) || !g_mainDevice) {
            LOG::logline("!! [seam] main device is not D3D9Ex; cannot open KMT shared texture — seam disabled");
            return;
        }
        g_mainDevice->Release();   // borrowed: QI AddRef'd; we don't own the main device

        HWND hwnd = deviceWindow(device);
        if (!createSeamDevice(hwnd)) {
            releaseAll();
            return;
        }
        if (FAILED(g_seamDevice->QueryInterface(__uuidof(IDirect3DDevice9On12), (void**)&g_dev9on12)) || !g_dev9on12) {
            LOG::logline("!! [seam] seam device is not D3D9On12 — seam disabled");
            releaseAll();
            return;
        }
        if (FAILED(g_dev9on12->GetD3D12Device(__uuidof(ID3D12Device), (void**)&g_d3d12dev)) || !g_d3d12dev) {
            LOG::logline("!! [seam] GetD3D12Device failed — seam disabled");
            releaseAll();
            return;
        }

        // Host brings up Forge + creates the shared RT; returns the NT handle already
        // duplicated into THIS process.
        HANDLE hostHandle = nullptr;
        if (!g_client->renderInitBlocking(kW, kH, nullptr, nullptr, &hostHandle) || hostHandle == nullptr) {
            LOG::logline("!! [seam] renderInit RPC failed or no shared handle; seam disabled");
            releaseAll();
            return;
        }
        LOG::logline(">> [seam] host shared-RT NT handle (this process) = %p", hostHandle);

        HRESULT hr = g_d3d12dev->OpenSharedHandle(hostHandle, __uuidof(ID3D12Resource), (void**)&g_hostRT12);
        if (FAILED(hr) || !g_hostRT12) {
            LOG::logline("!! [seam] *** OpenSharedHandle FAILED 0x%08X *** (side D3D12 could not open the host's RT)", hr);
            releaseAll();
            return;
        }
        LOG::logline(">> [seam] *** OpenSharedHandle OK *** (host D3D12 RT opened on the 9On12 side-device)");

        // KMT-shared D3D9 RT texture on the side-device. pSharedHandle OUT yields the
        // KMT handle (D3DFMT_A8R8G8B8 == DXGI B8G8R8A8_UNORM, the host RT format).
        hr = g_seamDevice->CreateTexture(kW, kH, 1, D3DUSAGE_RENDERTARGET,
                                         D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &g_seamTex, &g_kmtHandle);
        if (FAILED(hr) || !g_seamTex || !g_kmtHandle) {
            LOG::logline("!! [seam] side CreateTexture (KMT shared) failed 0x%08X — seam disabled", hr);
            releaseAll();
            return;
        }

        // Open the SAME KMT-shared texture on the MAIN device for the StretchRect.
        hr = g_mainDevice->CreateTexture(kW, kH, 1, D3DUSAGE_RENDERTARGET,
                                         D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &g_mainTex, &g_kmtHandle);
        if (FAILED(hr) || !g_mainTex) {
            LOG::logline("!! [seam] main CreateTexture (open KMT shared) failed 0x%08X — seam disabled", hr);
            releaseAll();
            return;
        }

        if (!createCopyInfra()) {
            LOG::logline("!! [seam] failed to create side D3D12 copy infrastructure — seam disabled");
            releaseAll();
            return;
        }

        g_initOk = true;
        LOG::logline(">> [seam] decoupled D3D9On12 seam ready (%ux%u). F11 toggles the composite.", kW, kH);
    }

    // Copy host RT -> g_seamTex's D3D12 backing on the side-device, then CPU-wait the
    // copy so the MAIN device's StretchRect reads finished pixels (cross-device).
    bool copyHostRtToSeamTex() {
        ID3D12Resource* dst12 = nullptr;
        if (FAILED(g_dev9on12->UnwrapUnderlyingResource(g_seamTex, g_copyQueue,
                __uuidof(ID3D12Resource), (void**)&dst12)) || !dst12) {
            return false;
        }

        // Both resources are COMMON at hand-off; D3D12 implicit promotion covers the copy.
        g_copyAlloc->Reset();
        g_copyList->Reset(g_copyAlloc, nullptr);
        g_copyList->CopyResource(dst12, g_hostRT12);
        g_copyList->Close();

        ID3D12CommandList* lists[] = { g_copyList };
        g_copyQueue->ExecuteCommandLists(1, lists);
        const UINT64 signalVal = ++g_fenceVal;
        g_copyQueue->Signal(g_copyFence, signalVal);

        ID3D12Fence* fences[] = { g_copyFence };
        UINT64       vals[]   = { signalVal };
        HRESULT hr = g_dev9on12->ReturnUnderlyingResource(g_seamTex, 1, vals, fences);
        dst12->Release();
        if (FAILED(hr)) {
            return false;
        }

        // CPU-wait the copy: the main device is a separate timeline, so ensure the
        // shared surface is fully written before it reads (single-buffer, blocking).
        if (g_copyFence->GetCompletedValue() < signalVal) {
            g_copyFence->SetEventOnCompletion(signalVal, g_fenceEvent);
            WaitForSingleObject(g_fenceEvent, INFINITE);
        }
        return true;
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
            LOG::logline(">> [seam] composite %s", g_enabled ? "ON" : "OFF");
        }
        if (!g_enabled) {
            return;
        }

        LARGE_INTEGER t0; QueryPerformanceCounter(&t0);

        const unsigned frame = g_frame++;

        // Drive the host renderer: render this frame into the shared RT. Blocking —
        // the host fence-waits before replying, so the draw is GPU-complete and the
        // resource is quiescent before our copy.
        double hostMs = 0.0;
        if (!g_client->renderFrameBlocking(frame, 0, &hostMs)) {
            return;
        }

        if (!copyHostRtToSeamTex()) {
            static bool logged = false;
            if (!logged) { LOG::logline("!! [seam] copyHostRtToSeamTex failed"); logged = true; }
            return;
        }

        IDirect3DSurface9* src = nullptr;
        if (FAILED(g_mainTex->GetSurfaceLevel(0, &src)) || !src) {
            return;
        }

        // Composite as a 1:1 corner quad (top-left), non-destructive to the frame.
        IDirect3DSurface9* backbuffer = nullptr;
        if (SUCCEEDED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backbuffer)) && backbuffer) {
            RECT dstRect = { 20, 20, 20 + (LONG)kW, 20 + (LONG)kH };
            HRESULT hr = device->StretchRect(src, nullptr, backbuffer, &dstRect, D3DTEXF_NONE);
            if (FAILED(hr)) {
                static bool logged = false;
                if (!logged) { LOG::logline("!! [seam] StretchRect failed 0x%x", hr); logged = true; }
            }
            backbuffer->Release();
        }
        src->Release();

        LARGE_INTEGER t1; QueryPerformanceCounter(&t1);
        g_sumRoundtripMs += double(t1.QuadPart - t0.QuadPart) * msPerTick();
        g_sumHostMs += hostMs;
        if (++g_statSamples >= 60) {
            LOG::logline("-- [seam] avg/60: client roundtrip=%.3f ms (host render=%.3f ms)",
                g_sumRoundtripMs / g_statSamples, g_sumHostMs / g_statSamples);
            g_sumRoundtripMs = g_sumHostMs = 0.0;
            g_statSamples = 0;
        }
    }

    void shutdown() {
        releaseAll();
        g_initOk = false;
        g_enabled = false;
    }
}
