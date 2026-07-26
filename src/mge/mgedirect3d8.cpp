
#include "mgedirect3d8.h"
#include "mged3d8device.h"
#include "configuration.h"
#include "proxydx/devicelock.h"
#include "support/log.h"

#include <algorithm>



static const UINT MorrowindRequiredD3DVersion = 120;

MGEProxyD3D::MGEProxyD3D(IDirect3D9* real) : ProxyD3D(real, MorrowindRequiredD3DVersion) {
    // Force pixel shaders off, to simplify water override
    d3d8Caps.VertexShaderVersion = 0;
    d3d8Caps.PixelShaderVersion = 0;

    // Log adapter details
    D3DADAPTER_IDENTIFIER9 adapter;
    realD3D->GetAdapterIdentifier(D3DADAPTER_DEFAULT, 0, &adapter);
    LOG::logline("GPU: %s (%d.%d.%d.%d)", adapter.Description,
                 HIWORD(adapter.DriverVersion.HighPart), LOWORD(adapter.DriverVersion.HighPart),
                 HIWORD(adapter.DriverVersion.LowPart), LOWORD(adapter.DriverVersion.LowPart));
}

void translatePresentParams8to9(D3DPRESENT_PARAMETERS8* e, bool isEx,
                                D3DPRESENT_PARAMETERS9& pp,
                                D3DDISPLAYMODEEX& dm, D3DDISPLAYMODEEX** pdm) {
    // MSAA parameters
    D3DMULTISAMPLE_TYPE msaaSamples = (D3DMULTISAMPLE_TYPE)Configuration.AALevel;
    DWORD msaaQuality = 0;

    // Override device parameters
    // Note that Morrowind will look at the modified parameters
    if (e->Flags & D3DPRESENTFLAG_LOCKABLE_BACKBUFFER) {
        e->Flags ^= D3DPRESENTFLAG_LOCKABLE_BACKBUFFER;
    }

    e->MultiSampleType = msaaSamples;
    e->AutoDepthStencilFormat = (D3DFORMAT)Configuration.ZBufFormat;
    e->FullScreen_RefreshRateInHz = (!e->Windowed) ? Configuration.RefreshRate : 0;
    e->FullScreen_PresentationInterval = (Configuration.VWait == 255) ? D3DPRESENT_INTERVAL_IMMEDIATE : Configuration.VWait;

    // Convert presentation parameters to DX9
    pp.BackBufferWidth = e->BackBufferWidth;
    pp.BackBufferHeight = e->BackBufferHeight;
    pp.BackBufferFormat = e->BackBufferFormat;
    pp.BackBufferCount = e->BackBufferCount;
    pp.MultiSampleType = e->MultiSampleType;
    pp.MultiSampleQuality = msaaQuality;
    pp.SwapEffect = e->SwapEffect;
    pp.hDeviceWindow = e->hDeviceWindow;
    pp.Windowed = e->Windowed;
    pp.Flags = e->Flags;
    pp.EnableAutoDepthStencil = e->EnableAutoDepthStencil;
    pp.AutoDepthStencilFormat = e->AutoDepthStencilFormat;
    pp.FullScreen_RefreshRateInHz = e->FullScreen_RefreshRateInHz;
    pp.PresentationInterval = e->FullScreen_PresentationInterval;

    // Present-seam spike: on a D3D9Ex device the present rules are stricter — SwapEffect must be
    // DISCARD/FLIP (not COPY) and BackBufferCount >= 1; fullscreen needs a D3DDISPLAYMODEEX,
    // windowed needs NULL. The normal game path (spike off / plain D3D9) leaves these untouched.
    *pdm = nullptr;
    if (isEx) {
        if (pp.SwapEffect == D3DSWAPEFFECT_COPY) {
            pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
        }
        if (pp.BackBufferCount == 0) {
            pp.BackBufferCount = 1;
        }
        if (!pp.Windowed) {
            dm.Size = sizeof(D3DDISPLAYMODEEX);
            dm.Width = pp.BackBufferWidth;
            dm.Height = pp.BackBufferHeight;
            dm.RefreshRate = pp.FullScreen_RefreshRateInHz;
            dm.Format = pp.BackBufferFormat;
            dm.ScanLineOrdering = D3DSCANLINEORDERING_PROGRESSIVE;
            *pdm = &dm;
        }
    }
}

HRESULT _stdcall MGEProxyD3D::CreateDevice(UINT a, D3DDEVTYPE b, HWND c, DWORD d, D3DPRESENT_PARAMETERS8* e, IDirect3DDevice8** f) {
    // Window positioning
    if (e->Windowed) {
        HWND hMainWnd = GetParent(c);
        int wx = std::max(0, Configuration.WindowAlignX * (GetSystemMetrics(SM_CXSCREEN) - (int)e->BackBufferWidth) / 2);
        int wy = std::max(0, Configuration.WindowAlignY * (GetSystemMetrics(SM_CYSCREEN) - (int)e->BackBufferHeight) / 2);

        if (Configuration.Borderless) {
            // Remove non-client window parts and move window flush to screen edge / centre if smaller than display
            SetWindowLong(hMainWnd, GWL_STYLE, WS_VISIBLE);
            SetWindowPos(hMainWnd, NULL, wx, wy, e->BackBufferWidth, e->BackBufferHeight, SWP_NOACTIVATE|SWP_NOCOPYBITS|SWP_NOZORDER);
        } else {
            // Move window to top, with client area centred on one axis
            RECT rect = { wx, wy, int(e->BackBufferWidth), int(e->BackBufferHeight) };
            AdjustWindowRect(&rect, GetWindowLong(hMainWnd, GWL_STYLE), FALSE);
            SetWindowPos(hMainWnd, NULL, rect.left, 0, 0, 0, SWP_NOSIZE|SWP_NOACTIVATE|SWP_NOCOPYBITS|SWP_NOZORDER);
        }

        // Ensure that the render window appears on the taskbar, as it is a child window that may become hidden
        LONG style = GetWindowLong(hMainWnd, GWL_EXSTYLE);
        SetWindowLong(hMainWnd, GWL_EXSTYLE, style | WS_EX_APPWINDOW);

        // Windowed mode does not allow multiple frame vsync
        if (Configuration.VWait >= D3DPRESENT_INTERVAL_TWO && Configuration.VWait <= D3DPRESENT_INTERVAL_FOUR) {
            Configuration.VWait = D3DPRESENT_INTERVAL_ONE;
            LOG::logline("VWait greater than one is not supported in windowed mode.");
        }
    }

    // Present seam: when the factory is a D3D9Ex factory (g_useD3D9Ex), create a D3D9Ex
    // device via CreateDeviceEx. An Ex device is what lets us create a shared render-target
    // texture for zero-copy hand-off to Vulkan; it also carries stricter present rules,
    // applied by the shared translation helper below. If Ex is unavailable the plain-D3D9
    // path below runs instead, untouched.
    IDirect3D9Ex* d3dEx = nullptr;
    const bool useEx = g_useD3D9Ex &&
                       SUCCEEDED(realD3D->QueryInterface(__uuidof(IDirect3D9Ex), reinterpret_cast<void**>(&d3dEx))) && d3dEx;

    // Translate DX8 -> DX9 present params (MGE overrides + DX9 conversion + Ex fixups when useEx).
    // Shared with the proxy device Reset so a reset backbuffer matches the created one. Morrowind
    // inspects the (mutated) DX8 params after this returns.
    D3DPRESENT_PARAMETERS9 pp;
    D3DDISPLAYMODEEX dm = {};
    D3DDISPLAYMODEEX* pdm = nullptr;
    translatePresentParams8to9(e, useEx, pp, dm, &pdm);

    // S5a: UseRenderThread armed D3DCREATE_MULTITHREADED + the device-submission lock here
    // so the MGE render thread could submit concurrently with the engine. The worker is gone
    // (its one job was the DX9 depth pre-pass), and the flag defaulted off anyway, so this is
    // the path that always ran. g_deviceLockEnabled stays defined-and-false: MGE_DEVLOCK is
    // a no-op scope guard in the proxy forwarders, kept for whoever next needs to serialise
    // a worker against them.
    LOG::logline("-- device created SINGLE-THREADED (flags=0x%08X)", d);

    // Create device in the same manner as the proxy.
    IDirect3DDevice9* realDevice = NULL;
    HRESULT hr = D3DERR_INVALIDCALL;

    if (useEx) {
        IDirect3DDevice9Ex* exDevice = nullptr;
        hr = d3dEx->CreateDeviceEx(a, b, c, d, &pp, pdm, &exDevice);
        if (SUCCEEDED(hr)) {
            realDevice = exDevice;   // IDirect3DDevice9Ex derives from IDirect3DDevice9
            // Ex rejects D3DPOOL_MANAGED: arm the proxy + MGE MANAGED->DEFAULT translation.
            g_spikeForceDefaultPool = true;
            LOG::logline(">> [spike] CreateDeviceEx OK (D3D9Ex device, windowed=%d, swap=%d); MANAGED->DEFAULT pool translation armed", pp.Windowed, pp.SwapEffect);
        } else {
            LOG::logline("!! [seam] CreateDeviceEx failed 0x%08X; disabling D3D9Ex path, using plain CreateDevice", hr);
            g_useD3D9Ex = false;
        }
    }
    if (d3dEx) {
        d3dEx->Release();
    }

    if (!realDevice) {
        hr = realD3D->CreateDevice(a, b, c, d, &pp, &realDevice);
    }

    if (hr != D3D_OK) {
        LOG::logline("!! D3D Proxy CreateDevice failure");
        LOG::flush();
        return hr;
    }

    *f = factoryProxyDevice(realDevice);

    // Set up default render states
    Configuration.ScaleFilter = (Configuration.AnisoLevel > 0) ? D3DTEXF_ANISOTROPIC : D3DTEXF_LINEAR;

    for (int i = 0; i != 8; ++i) {
        realDevice->SetSamplerState(i, D3DSAMP_MINFILTER, Configuration.ScaleFilter);
        realDevice->SetSamplerState(i, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
        realDevice->SetSamplerState(i, D3DSAMP_MAXANISOTROPY, Configuration.AnisoLevel);
    }

    // Set variables dependent on configuration
    DWORD FogPixelMode, FogVertexMode, RangedFog;
    if (Configuration.FogMode == 2) {
        FogVertexMode = D3DFOG_LINEAR;
        FogPixelMode = D3DFOG_NONE;
        RangedFog = 1;
    } else if (Configuration.FogMode == 1) {
        FogVertexMode = D3DFOG_LINEAR;
        FogPixelMode = D3DFOG_NONE;
        RangedFog = 0;
    } else {
        FogVertexMode = D3DFOG_NONE;
        FogPixelMode = D3DFOG_LINEAR;
        RangedFog = 0;
    }

    realDevice->SetRenderState(D3DRS_FOGVERTEXMODE, FogVertexMode);
    realDevice->SetRenderState(D3DRS_FOGTABLEMODE, FogPixelMode);
    realDevice->SetRenderState(D3DRS_RANGEFOGENABLE, RangedFog);
    realDevice->SetRenderState(D3DRS_MULTISAMPLEANTIALIAS, (Configuration.AALevel > 0));

    LOG::logline("-- D3D Proxy Device OK");
    return D3D_OK;
}

IDirect3DDevice8* MGEProxyD3D::factoryProxyDevice(IDirect3DDevice9* d) {
    return new MGEProxyDevice(d, this);
}
