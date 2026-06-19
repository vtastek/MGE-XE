
#include "mgedirect3d8.h"
#include "mge/configuration.h"
#include "support/log.h"


typedef IDirect3D9* (_stdcall* D3DProc9)(UINT);
typedef HRESULT (_stdcall* D3DProc9Ex)(UINT, IDirect3D9Ex**);



static const UINT MorrowindRequiredD3DVersion = 120;

void* CreateD3DWrapper(UINT version) {
    HMODULE d3ddll = LoadLibrary("d3d9.dll");

    IDirect3D9* d3d = nullptr;

    // Present-seam spike (Milestone B): a D3D9Ex factory is required to create the
    // D3D9Ex device that can produce a shared render-target HANDLE for zero-copy
    // hand-off to the 64-bit Vulkan renderer. Gated by UseRenderProcess; the normal
    // game keeps the plain D3D9 path untouched. If the Ex factory can't be created,
    // the spike is disabled for this run (logged) and we proceed on plain D3D9 so the
    // game still launches normally.
    if (Configuration.UseRenderProcessEx) {
        D3DProc9Ex func9Ex = (D3DProc9Ex)GetProcAddress(d3ddll, "Direct3DCreate9Ex");
        IDirect3D9Ex* d3dEx = nullptr;
        if (func9Ex && SUCCEEDED(func9Ex(D3D_SDK_VERSION, &d3dEx)) && d3dEx) {
            d3d = d3dEx;   // IDirect3D9Ex derives from IDirect3D9
            LOG::logline(">> [spike] Direct3DCreate9Ex OK (D3D9Ex factory active)");
        } else {
            LOG::logline("!! [spike] Direct3DCreate9Ex unavailable; disabling D3D9Ex spike path for this run");
            Configuration.UseRenderProcessEx = false;
        }
    }

    if (!d3d) {
        D3DProc9 func9 = (D3DProc9)GetProcAddress(d3ddll, "Direct3DCreate9");
        d3d = func9(D3D_SDK_VERSION);
    }

    if (!Configuration.OnlyProxyD3D8To9) {
        return new MGEProxyD3D(d3d);
    }
    else {
        LOG::logline(">> Using D3D8To9 proxy mode");
        return new ProxyD3D(d3d, MorrowindRequiredD3DVersion);
    }
}
