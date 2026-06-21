
#include "mgedirect3d8.h"
#include "mge/configuration.h"
#include "support/log.h"


typedef IDirect3D9* (_stdcall* D3DProc9)(UINT);
typedef HRESULT (_stdcall* D3DProc9Ex)(UINT, IDirect3D9Ex**);



static const UINT MorrowindRequiredD3DVersion = 120;

void* CreateD3DWrapper(UINT version) {
    // Routing: MW's MAIN device runs on DXVK, renamed d3d9_dxvk.dll in the game dir. DXVK
    // is Vulkan-backed, so it pays no native-D3D9 cold-init stall, and — crucially — it
    // exposes ID3D9VkInteropDevice, which the present seam uses to import the Forge host's
    // shared D3D12 render target straight into DXVK's own VkDevice (see renderprocess.cpp).
    HMODULE d3ddll = LoadLibrary("d3d9_dxvk.dll");
    if (!d3ddll) {
        LOG::logline("!! [routing] d3d9_dxvk.dll not found; using system d3d9.dll (native, no DXVK interop seam)");
        d3ddll = LoadLibrary("d3d9.dll");
    }

    IDirect3D9* d3d = nullptr;

    // Present seam: create a D3D9Ex factory so MW's main device is a D3D9Ex device. Gated
    // by UseRenderProcessEx; if the Ex factory is unavailable we fall through to plain D3D9.
    if (Configuration.UseRenderProcessEx) {
        D3DProc9Ex func9Ex = (D3DProc9Ex)GetProcAddress(d3ddll, "Direct3DCreate9Ex");
        IDirect3D9Ex* d3dEx = nullptr;
        if (func9Ex && SUCCEEDED(func9Ex(D3D_SDK_VERSION, &d3dEx)) && d3dEx) {
            d3d = d3dEx;   // IDirect3D9Ex derives from IDirect3D9
            LOG::logline(">> [seam] Direct3DCreate9Ex OK (DXVK D3D9Ex main device)");
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
