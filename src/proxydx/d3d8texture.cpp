
#include "d3d8device.h"
#include "d3d8surface.h"
#include "d3d8texture.h"

// Declared in ffeshader.h — evicts texture from suffix resolution cache on release
extern void (*g_onTextureReleased)(IDirect3DTexture9* realTexture);

// Upload hash map: populated during UnlockRect(0), queried by calculateTextureHash
std::unordered_map<IDirect3DTexture9*, UploadHash> g_uploadHashMap;

// CRC32 for upload hashing (same polynomial as morrowindbsa.cpp)
static unsigned int crc32_table_upload[256];
static bool crc32_table_upload_init = false;

static void init_crc32_upload() {
    if (crc32_table_upload_init) return;
    for (int i = 0; i < 256; i++) {
        unsigned int c = (unsigned int)i;
        for (int j = 0; j < 8; j++) {
            c = (c & 1) ? (0xedb88320L ^ (c >> 1)) : (c >> 1);
        }
        crc32_table_upload[i] = c;
    }
    crc32_table_upload_init = true;
}

static unsigned int crc32_upload(const unsigned char* buf, size_t len) {
    if (!buf || len == 0) return 0;
    init_crc32_upload();
    unsigned int c = 0xffffffffL;
    for (size_t i = 0; i < len; i++) {
        c = crc32_table_upload[(c ^ buf[i]) & 0xff] ^ (c >> 8);
    }
    return c ^ 0xffffffffL;
}

ProxyTexture::ProxyTexture(IDirect3DTexture9* real, ProxyDevice* device) : realTexture(real), proxDevice(device), hasActiveLock0(false) {
    pendingLockRect = {};
    ProxyTexture* proxy = this;
    real->SetPrivateData(guid_proxydx, (void*)&proxy, sizeof(proxy), 0);
}

//-----------------------------------------------------------------------------
/*** IUnknown methods ***/
//-----------------------------------------------------------------------------

HRESULT _stdcall ProxyTexture::QueryInterface(REFIID riid, void** ppvObj) {
    return realTexture->QueryInterface(riid, ppvObj);
}
ULONG _stdcall ProxyTexture::AddRef() {
    return realTexture->AddRef();
}
ULONG _stdcall ProxyTexture::Release() {
    ULONG refcount = realTexture->Release();
    if (!refcount) {
        if (g_onTextureReleased) {
            g_onTextureReleased(realTexture);
        }
        g_uploadHashMap.erase(realTexture);
        delete this;
        return 0;
    }
    return refcount;
}

//-----------------------------------------------------------------------------
/*** IDirect3DBaseTexture8 methods ***/
//-----------------------------------------------------------------------------

HRESULT _stdcall ProxyTexture::GetDevice(IDirect3DDevice8** ppDevice) {
    *ppDevice = proxDevice;
    return D3D_OK;
}

HRESULT _stdcall ProxyTexture::SetPrivateData(REFGUID refguid, CONST void* pData, DWORD SizeOfData, DWORD Flags) {
    return realTexture->SetPrivateData(refguid, pData, SizeOfData, Flags);
}

HRESULT _stdcall ProxyTexture::GetPrivateData(REFGUID refguid, void* pData, DWORD* pSizeOfData) {
    return realTexture->GetPrivateData(refguid, pData, pSizeOfData);
}

HRESULT _stdcall ProxyTexture::FreePrivateData(REFGUID refguid) {
    return realTexture->FreePrivateData(refguid);
}

//-----------------------------------------------------------------------------

DWORD _stdcall ProxyTexture::SetPriority(DWORD PriorityNew) {
    return realTexture->SetPriority(PriorityNew);
}
DWORD _stdcall ProxyTexture::GetPriority() {
    return realTexture->GetPriority();
}
void _stdcall ProxyTexture::PreLoad() {
    return realTexture->PreLoad();
}
D3DRESOURCETYPE _stdcall ProxyTexture::GetType() {
    return realTexture->GetType();
}

//-----------------------------------------------------------------------------

DWORD _stdcall ProxyTexture::SetLOD(DWORD LODNew) {
    return realTexture->SetLOD(LODNew);
}
DWORD _stdcall ProxyTexture::GetLOD() {
    return realTexture->GetLOD();
}

//-----------------------------------------------------------------------------

DWORD _stdcall ProxyTexture::GetLevelCount() {
    return realTexture->GetLevelCount();
}

HRESULT _stdcall ProxyTexture::GetLevelDesc(UINT Level, D3DSURFACE_DESC8* pDesc) {
    D3DSURFACE_DESC b2;
    HRESULT hr = realTexture->GetLevelDesc(Level, &b2);
    if (hr == D3D_OK) {
        pDesc->Format = b2.Format;
        pDesc->Height = b2.Height;
        pDesc->MultiSampleType = b2.MultiSampleType;
        pDesc->Pool = b2.Pool;
        pDesc->Size = 0; // TODO: Fix;
        pDesc->Type = b2.Type;
        pDesc->Usage = b2.Usage;
        pDesc->Width = b2.Width;
    }
    return hr;
}

HRESULT _stdcall ProxyTexture::GetSurfaceLevel(UINT Level, IDirect3DSurface8** ppSurfaceLevel) {
    IDirect3DSurface9* surface_real = NULL;
    *ppSurfaceLevel = NULL;

    HRESULT hr = realTexture->GetSurfaceLevel(Level, &surface_real);
    if (hr != D3D_OK || surface_real == NULL) {
        return hr;
    }

    ProxySurface* surface = ProxySurface::getProxyFromDX(surface_real);
    if (surface) {
        *ppSurfaceLevel = surface;
    } else {
        *ppSurfaceLevel = proxDevice->factoryProxySurface(surface_real);
    }

    return D3D_OK;
}

//-----------------------------------------------------------------------------

HRESULT _stdcall ProxyTexture::LockRect(UINT Level, D3DLOCKED_RECT* pLockedRect, CONST RECT* pRect, DWORD Flags) {
    HRESULT hr = realTexture->LockRect(Level, pLockedRect, pRect, Flags);
    if (SUCCEEDED(hr) && Level == 0 && pLockedRect && !(Flags & D3DLOCK_READONLY)) {
        pendingLockRect = *pLockedRect;
        hasActiveLock0 = true;
    }
    return hr;
}
HRESULT _stdcall ProxyTexture::UnlockRect(UINT Level) {
    // Hash texture data at upload time (level 0 only, write locks only)
    if (Level == 0 && hasActiveLock0 && pendingLockRect.pBits) {
        hasActiveLock0 = false;

        // Only hash non-RT textures with valid data
        D3DSURFACE_DESC desc;
        if (SUCCEEDED(realTexture->GetLevelDesc(0, &desc)) &&
            !(desc.Usage & D3DUSAGE_RENDERTARGET) &&
            !(desc.Usage & D3DUSAGE_DEPTHSTENCIL) &&
            desc.Width > 0 && desc.Height > 0 && pendingLockRect.Pitch > 0) {

            // Calculate data size (same logic as morrowindbsa.cpp)
            size_t dataSize = 0;
            switch (desc.Format) {
                case D3DFMT_DXT1:
                    dataSize = ((desc.Width + 3) / 4) * ((desc.Height + 3) / 4) * 8;
                    break;
                case D3DFMT_DXT3:
                case D3DFMT_DXT5:
                    dataSize = ((desc.Width + 3) / 4) * ((desc.Height + 3) / 4) * 16;
                    break;
                default:
                    dataSize = (size_t)pendingLockRect.Pitch * desc.Height;
                    break;
            }

            if (dataSize > 0 && dataSize < 100 * 1024 * 1024) {
                const unsigned char* dataPtr = reinterpret_cast<const unsigned char*>(pendingLockRect.pBits);

                // Replicate exact hash algorithm from morrowindbsa.cpp:
                // metadata (5 DWORDs) + multi-point samples (3 × 64 bytes)
                DWORD mipLevels = realTexture->GetLevelCount();
                size_t pointSize = 64;
                size_t maxSampleSize = (dataSize < 2048) ? dataSize : 2048;
                size_t metadataSize = 20;
                size_t totalHashSize = metadataSize + maxSampleSize;
                unsigned char hashBuffer[2068]; // 20 + 2048

                // Pack metadata
                DWORD* metadata = reinterpret_cast<DWORD*>(hashBuffer);
                metadata[0] = desc.Width;
                metadata[1] = desc.Height;
                metadata[2] = (DWORD)desc.Format;
                metadata[3] = mipLevels;
                metadata[4] = (DWORD)dataSize;

                // Multi-point samples
                unsigned char* sampleBuf = hashBuffer + metadataSize;
                memset(sampleBuf, 0, maxSampleSize);

                // Sample 1: beginning
                size_t s1 = (pointSize < dataSize) ? pointSize : dataSize;
                memcpy(sampleBuf, dataPtr, s1);

                // Sample 2: middle
                if (dataSize > pointSize * 2) {
                    size_t mid = dataSize / 2;
                    size_t s2 = (pointSize < (dataSize - mid)) ? pointSize : (dataSize - mid);
                    memcpy(sampleBuf + pointSize, dataPtr + mid, s2);
                }

                // Sample 3: end
                if (dataSize > pointSize * 3) {
                    size_t end = dataSize - pointSize;
                    size_t s3 = (pointSize < (dataSize - end)) ? pointSize : (dataSize - end);
                    memcpy(sampleBuf + (2 * pointSize), dataPtr + end, s3);
                }

                // Note: skip 8x8 mip data (not available at upload time for level 0)
                // Use metadata + 3-point sample only — totalHashSize stays at metadataSize + 3*pointSize
                totalHashSize = metadataSize + (3 * pointSize);
                if (totalHashSize > sizeof(hashBuffer)) totalHashSize = sizeof(hashBuffer);

                unsigned int crc = crc32_upload(hashBuffer, totalHashSize);
                if (crc != 0) {
                    UploadHash uh;
                    uh.crc32 = crc;
                    uh.size = desc.Width * desc.Height;
                    g_uploadHashMap[realTexture] = uh;
                }
            }
        }
        pendingLockRect = {};
    }
    return realTexture->UnlockRect(Level);
}
HRESULT _stdcall ProxyTexture::AddDirtyRect(CONST RECT* pDirtyRect ) {
    return realTexture->AddDirtyRect(pDirtyRect);
}

//-----------------------------------------------------------------------------

// Proxy methods
ProxyTexture* ProxyTexture::getProxyFromDX(IDirect3DBaseTexture9* real) {
    ProxyTexture* tex;
    DWORD data_sz = sizeof(tex);

    HRESULT hr = real->GetPrivateData(guid_proxydx, (void*)&tex, &data_sz);
    if (hr == D3D_OK) {
        return tex;
    }

    return 0;
}
