#pragma once

#include "d3d8interface.h"

#include "texledger.h"

#include <cstdint>

// GPU bytes of a width x height texture with `levels` mips in `format` (block-compressed aware).
uint32_t proxyTextureBytes(UINT width, UINT height, UINT levels, D3DFORMAT format);

// Top mip levels to drop so a w x h texture fits g_proxyTexCapDim; 0 = leave it whole. `levels` is
// what Morrowind asked for (0 = full chain): a level can only be dropped if one survives below it.
UINT proxyTextureCapSkip(UINT w, UINT h, UINT levels, D3DFORMAT format);

// THE MAP CAP. Under Forge, Morrowind's own draw of the world is a no-op; what still samples its
// textures is the local map (a top-down render where a boulder is a few pixels) and the inventory
// doll. So a large mip-mapped texture is created with its top levels DROPPED, and Morrowind, which
// believes it has the full chain, is shown VIRTUAL levels: level L >= capSkip is the real level
// L - capSkip (exactly the same size), and a lock of L < capSkip lands in a scratch buffer that is
// discarded on unlock. Morrowind actually fills its GPU textures by UpdateTexture from a full-chain
// SYSTEMMEM twin (ProxyDevice::UpdateTexture), which lands the twin's bottom levels -- exactly the
// kept ones -- so they arrive intact, from its own data, with nothing resampled.
class ProxyTexture : public IDirect3DTexture8 {
public:
    IDirect3DTexture9* realTexture;
    ProxyDevice* proxDevice;
    uint32_t ledgerBytes = 0;   // counted in proxyTexLedger() while > 0; see ledgerTrack
    UINT capSkip = 0;           // top levels dropped at creation (0 = uncapped)
    UINT capWidth = 0, capHeight = 0;   // the level-0 size Morrowind asked for
    D3DFORMAT capFormat = D3DFMT_UNKNOWN;
    bool capFilled = false;     // Morrowind has written a kept level (ledger capFilled)
    static const UINT kMaxCapSkip = 8;
    void* capScratch[kMaxCapSkip] = {};  // per dropped level, alive from lock to unlock

    ProxyTexture(IDirect3DTexture9* real, ProxyDevice* device);

    //-----------------------------------------------------------------------------
    /*** IUnknown methods ***/
    //-----------------------------------------------------------------------------
    HRESULT _stdcall QueryInterface(REFIID riid, void** ppvObj);
    ULONG _stdcall AddRef();
    ULONG _stdcall Release();

    //-----------------------------------------------------------------------------
    /*** IDirect3DBaseTexture8 methods ***/
    //-----------------------------------------------------------------------------
    HRESULT _stdcall GetDevice(IDirect3DDevice8** ppDevice);
    HRESULT _stdcall SetPrivateData(REFGUID refguid, CONST void* pData, DWORD SizeOfData, DWORD Flags);
    HRESULT _stdcall GetPrivateData(REFGUID refguid, void* pData, DWORD* pSizeOfData);
    HRESULT _stdcall FreePrivateData(REFGUID refguid);
    DWORD _stdcall SetPriority(DWORD PriorityNew);
    DWORD _stdcall GetPriority();
    void _stdcall PreLoad();
    D3DRESOURCETYPE _stdcall GetType();
    DWORD _stdcall SetLOD(DWORD LODNew);
    DWORD _stdcall GetLOD();
    DWORD _stdcall GetLevelCount();
    HRESULT _stdcall GetLevelDesc(UINT Level, D3DSURFACE_DESC8* pDesc);
    HRESULT _stdcall GetSurfaceLevel(UINT Level, IDirect3DSurface8** ppSurfaceLevel);
    HRESULT _stdcall LockRect(UINT Level, D3DLOCKED_RECT* pLockedRect, CONST RECT* pRect, DWORD Flags);
    HRESULT _stdcall UnlockRect(UINT Level);
    HRESULT _stdcall AddDirtyRect(CONST RECT* pDirtyRect);

    // Proxy methods
    static ProxyTexture* getProxyFromDX(IDirect3DBaseTexture9* real);
    void ledgerTrack(uint32_t bytes);   // once, at creation, for GPU-pool textures only
    void capArm(UINT skip, UINT w, UINT h, D3DFORMAT format);   // once, at creation
    void capNoteFilled();
};
