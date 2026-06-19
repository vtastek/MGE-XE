
#include "mgedirect3d8.h"
#include "mge/configuration.h"
#include "support/log.h"


typedef IDirect3D9* (_stdcall* D3DProc9)(UINT);
typedef HRESULT (_stdcall* D3DProc9Ex)(UINT, IDirect3D9Ex**);



static const UINT MorrowindRequiredD3DVersion = 120;

void* CreateD3DWrapper(UINT version) {
    HMODULE d3ddll = LoadLibrary("d3d9.dll");

    IDirect3D9* d3d = nullptr;

    // Present seam: MW's MAIN device is a native D3D9Ex device (D3D9Ex is required to
    // open the KMT-shared blit texture the seam re-shares from its dedicated D3D9On12
    // side-device — see renderprocess.cpp). The game renders/presents on this proven
    // native path; the 9On12 device is created separately and used ONLY for the seam,
    // so MGE's heavy D3D9 pipeline never goes through the 9On12 translation layer
    // (which black-screens the whole game). Gated by UseRenderProcessEx. If the Ex
    // factory is unavailable, the seam is disabled and we proceed on plain D3D9.
    if (Configuration.UseRenderProcessEx) {
        D3DProc9Ex func9Ex = (D3DProc9Ex)GetProcAddress(d3ddll, "Direct3DCreate9Ex");
        IDirect3D9Ex* d3dEx = nullptr;
        if (func9Ex && SUCCEEDED(func9Ex(D3D_SDK_VERSION, &d3dEx)) && d3dEx) {
            d3d = d3dEx;   // IDirect3D9Ex derives from IDirect3D9
            LOG::logline(">> [seam] Direct3DCreate9Ex OK (native D3D9Ex main device)");
        } else {
            LOG::logline("!! [seam] Direct3DCreate9Ex unavailable; disabling D3D9Ex seam path for this run");
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
