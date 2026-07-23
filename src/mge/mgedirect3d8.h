#pragma once

#include "proxydx/direct3d8.h"



class MGEProxyD3D : public ProxyD3D {
public:
    MGEProxyD3D(IDirect3D9* real);

    HRESULT _stdcall CreateDevice(UINT a, D3DDEVTYPE b, HWND c, DWORD d, D3DPRESENT_PARAMETERS8* e, IDirect3DDevice8** f);

    IDirect3DDevice8* factoryProxyDevice(IDirect3DDevice9* d);
};

// Single source of truth for the D3D8 -> D3D9 present-params translation, shared by
// MGEProxyD3D::CreateDevice and the proxy device Reset (device-reset seam). Applies the MGE
// overrides Morrowind then inspects (MSAA, ZBuf format, refresh, VWait, strip LOCKABLE_BACKBUFFER
// -- e is mutated in place), fills the DX9 pp, and, when isEx (the D3D9Ex takeover device), the
// Ex present fixups: COPY->DISCARD, BackBufferCount 0->1, and a D3DDISPLAYMODEEX for fullscreen
// (*pdm = &dm) or NULL for windowed (*pdm = nullptr).
void translatePresentParams8to9(D3DPRESENT_PARAMETERS8* e, bool isEx,
                                D3DPRESENT_PARAMETERS9& pp,
                                D3DDISPLAYMODEEX& dm, D3DDISPLAYMODEEX** pdm);
