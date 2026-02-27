#include "d3d8buffer.h"
#include "mge/resource_tracker.h"

//-----------------------------------------------------------------------------
// ProxyVertexBuffer
//-----------------------------------------------------------------------------

ProxyVertexBuffer::ProxyVertexBuffer(IDirect3DVertexBuffer9* real) : refcount(1), realBuffer(real) {
}

HRESULT _stdcall ProxyVertexBuffer::QueryInterface(REFIID a, LPVOID* b) {
    return realBuffer->QueryInterface(a, b);
}

ULONG _stdcall ProxyVertexBuffer::AddRef(void) {
    return ++refcount;
}

ULONG _stdcall ProxyVertexBuffer::Release(void) {
    if (--refcount == 0) {
        // Untrack from resource tracker before releasing
        ResourceTracker::getInstance().untrackVertexBuffer(realBuffer);
        realBuffer->Release();
        delete this;
        return 0;
    }
    return refcount;
}

HRESULT _stdcall ProxyVertexBuffer::GetDevice(IDirect3DDevice8** ppDevice) {
    return realBuffer->GetDevice((IDirect3DDevice9**)ppDevice);
}

HRESULT _stdcall ProxyVertexBuffer::SetPrivateData(REFGUID refguid, const void* pData, DWORD SizeOfData, DWORD Flags) {
    return realBuffer->SetPrivateData(refguid, pData, SizeOfData, Flags);
}

HRESULT _stdcall ProxyVertexBuffer::GetPrivateData(REFGUID refguid, void* pData, DWORD* pSizeOfData) {
    return realBuffer->GetPrivateData(refguid, pData, pSizeOfData);
}

HRESULT _stdcall ProxyVertexBuffer::FreePrivateData(REFGUID refguid) {
    return realBuffer->FreePrivateData(refguid);
}

DWORD _stdcall ProxyVertexBuffer::SetPriority(DWORD PriorityNew) {
    return realBuffer->SetPriority(PriorityNew);
}

DWORD _stdcall ProxyVertexBuffer::GetPriority() {
    return realBuffer->GetPriority();
}

void _stdcall ProxyVertexBuffer::PreLoad() {
    realBuffer->PreLoad();
}

D3DRESOURCETYPE _stdcall ProxyVertexBuffer::GetType() {
    return D3DRTYPE_VERTEXBUFFER;
}

HRESULT _stdcall ProxyVertexBuffer::Lock(UINT OffsetToLock, UINT SizeToLock, BYTE** ppbData, DWORD Flags) {
    HRESULT hr = realBuffer->Lock(OffsetToLock, SizeToLock, (void**)ppbData, Flags);

    if (SUCCEEDED(hr) && ppbData && *ppbData) {
        ResourceTracker::getInstance().onVertexBufferLock(realBuffer, *ppbData, OffsetToLock, SizeToLock, Flags);
    }

    return hr;
}

HRESULT _stdcall ProxyVertexBuffer::Unlock() {
    ResourceTracker::getInstance().onVertexBufferUnlock(realBuffer);

    return realBuffer->Unlock();
}

HRESULT _stdcall ProxyVertexBuffer::GetDesc(D3DVERTEXBUFFER_DESC* pDesc) {
    return realBuffer->GetDesc(pDesc);
}

//-----------------------------------------------------------------------------
// ProxyIndexBuffer
//-----------------------------------------------------------------------------

ProxyIndexBuffer::ProxyIndexBuffer(IDirect3DIndexBuffer9* real) : refcount(1), realBuffer(real) {
}

HRESULT _stdcall ProxyIndexBuffer::QueryInterface(REFIID a, LPVOID* b) {
    return realBuffer->QueryInterface(a, b);
}

ULONG _stdcall ProxyIndexBuffer::AddRef(void) {
    return ++refcount;
}

ULONG _stdcall ProxyIndexBuffer::Release(void) {
    if (--refcount == 0) {
        // Untrack from resource tracker before releasing
        ResourceTracker::getInstance().untrackIndexBuffer(realBuffer);
        realBuffer->Release();
        delete this;
        return 0;
    }
    return refcount;
}

HRESULT _stdcall ProxyIndexBuffer::GetDevice(IDirect3DDevice8** ppDevice) {
    return realBuffer->GetDevice((IDirect3DDevice9**)ppDevice);
}

HRESULT _stdcall ProxyIndexBuffer::SetPrivateData(REFGUID refguid, const void* pData, DWORD SizeOfData, DWORD Flags) {
    return realBuffer->SetPrivateData(refguid, pData, SizeOfData, Flags);
}

HRESULT _stdcall ProxyIndexBuffer::GetPrivateData(REFGUID refguid, void* pData, DWORD* pSizeOfData) {
    return realBuffer->GetPrivateData(refguid, pData, pSizeOfData);
}

HRESULT _stdcall ProxyIndexBuffer::FreePrivateData(REFGUID refguid) {
    return realBuffer->FreePrivateData(refguid);
}

DWORD _stdcall ProxyIndexBuffer::SetPriority(DWORD PriorityNew) {
    return realBuffer->SetPriority(PriorityNew);
}

DWORD _stdcall ProxyIndexBuffer::GetPriority() {
    return realBuffer->GetPriority();
}

void _stdcall ProxyIndexBuffer::PreLoad() {
    realBuffer->PreLoad();
}

D3DRESOURCETYPE _stdcall ProxyIndexBuffer::GetType() {
    return D3DRTYPE_INDEXBUFFER;
}

HRESULT _stdcall ProxyIndexBuffer::Lock(UINT OffsetToLock, UINT SizeToLock, BYTE** ppbData, DWORD Flags) {
    HRESULT hr = realBuffer->Lock(OffsetToLock, SizeToLock, (void**)ppbData, Flags);

    if (SUCCEEDED(hr) && ppbData && *ppbData) {
        ResourceTracker::getInstance().onIndexBufferLock(realBuffer, *ppbData, OffsetToLock, SizeToLock, Flags);
    }

    return hr;
}

HRESULT _stdcall ProxyIndexBuffer::Unlock() {
    ResourceTracker::getInstance().onIndexBufferUnlock(realBuffer);

    return realBuffer->Unlock();
}

HRESULT _stdcall ProxyIndexBuffer::GetDesc(D3DINDEXBUFFER_DESC* pDesc) {
    return realBuffer->GetDesc(pDesc);
}
