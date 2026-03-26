#pragma once

#include "d3d8interface.h"

// Upload hash: computed at UnlockRect time, avoids GPU readback for texture identification
struct UploadHash {
    unsigned int crc32;
    unsigned int size;  // width * height
};

// Global lookup: real texture → upload hash (populated during UnlockRect, evicted on Release)
#include <unordered_map>
extern std::unordered_map<IDirect3DTexture9*, UploadHash> g_uploadHashMap;

class ProxyTexture : public IDirect3DTexture8 {
public:
    DWORD refcount;
    IDirect3DTexture9* realTexture;
    ProxyDevice* proxDevice;

    // Upload hash state: captured during LockRect(0), hashed during UnlockRect(0)
    D3DLOCKED_RECT pendingLockRect;
    bool hasActiveLock0;

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
};
