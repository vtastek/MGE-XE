#pragma once

#include "d3d8interface.h"

// Proxy class for IDirect3DVertexBuffer8 to intercept Lock/Unlock
class ProxyVertexBuffer : public IDirect3DVertexBuffer8 {
public:
    DWORD refcount;
    IDirect3DVertexBuffer9* realBuffer;

    ProxyVertexBuffer(IDirect3DVertexBuffer9* real);

    /*** IUnknown methods ***/
    HRESULT _stdcall QueryInterface(REFIID a, LPVOID* b);
    ULONG _stdcall AddRef(void);
    ULONG _stdcall Release(void);

    /*** IDirect3DResource8 methods ***/
    HRESULT _stdcall GetDevice(IDirect3DDevice8** ppDevice);
    HRESULT _stdcall SetPrivateData(REFGUID refguid, const void* pData, DWORD SizeOfData, DWORD Flags);
    HRESULT _stdcall GetPrivateData(REFGUID refguid, void* pData, DWORD* pSizeOfData);
    HRESULT _stdcall FreePrivateData(REFGUID refguid);
    DWORD _stdcall SetPriority(DWORD PriorityNew);
    DWORD _stdcall GetPriority();
    void _stdcall PreLoad();
    D3DRESOURCETYPE _stdcall GetType();

    /*** IDirect3DVertexBuffer8 methods ***/
    HRESULT _stdcall Lock(UINT OffsetToLock, UINT SizeToLock, BYTE** ppbData, DWORD Flags);
    HRESULT _stdcall Unlock();
    HRESULT _stdcall GetDesc(D3DVERTEXBUFFER_DESC* pDesc);
};

// Proxy class for IDirect3DIndexBuffer8 to intercept Lock/Unlock
class ProxyIndexBuffer : public IDirect3DIndexBuffer8 {
public:
    DWORD refcount;
    IDirect3DIndexBuffer9* realBuffer;

    ProxyIndexBuffer(IDirect3DIndexBuffer9* real);

    /*** IUnknown methods ***/
    HRESULT _stdcall QueryInterface(REFIID a, LPVOID* b);
    ULONG _stdcall AddRef(void);
    ULONG _stdcall Release(void);

    /*** IDirect3DResource8 methods ***/
    HRESULT _stdcall GetDevice(IDirect3DDevice8** ppDevice);
    HRESULT _stdcall SetPrivateData(REFGUID refguid, const void* pData, DWORD SizeOfData, DWORD Flags);
    HRESULT _stdcall GetPrivateData(REFGUID refguid, void* pData, DWORD* pSizeOfData);
    HRESULT _stdcall FreePrivateData(REFGUID refguid);
    DWORD _stdcall SetPriority(DWORD PriorityNew);
    DWORD _stdcall GetPriority();
    void _stdcall PreLoad();
    D3DRESOURCETYPE _stdcall GetType();

    /*** IDirect3DIndexBuffer8 methods ***/
    HRESULT _stdcall Lock(UINT OffsetToLock, UINT SizeToLock, BYTE** ppbData, DWORD Flags);
    HRESULT _stdcall Unlock();
    HRESULT _stdcall GetDesc(D3DINDEXBUFFER_DESC* pDesc);
};
