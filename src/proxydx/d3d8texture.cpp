
#include "d3d8device.h"
#include "d3d8surface.h"
#include "d3d8texture.h"

#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace {
    std::mutex g_texWriteMx;
    std::unordered_map<void*, uint32_t> g_texWrites;
}
void proxyTexNoteWrite(void* realTexture) {
    if (!realTexture) { return; }
    std::lock_guard<std::mutex> lk(g_texWriteMx);
    // Bounded by a reset: every serial reads 0 again, which a reader holding a non-zero serial sees as
    // one more change (a redundant re-read), never as a missed one.
    if (g_texWrites.size() > 65536) { g_texWrites.clear(); }
    ++g_texWrites[realTexture];
}
uint32_t proxyTexWriteSerial(void* realTexture) {
    std::lock_guard<std::mutex> lk(g_texWriteMx);
    auto it = g_texWrites.find(realTexture);
    return it == g_texWrites.end() ? 0u : it->second;
}



ProxyTexLedger& proxyTexLedger() {
    static ProxyTexLedger s_ledger;
    return s_ledger;
}

uint32_t g_proxyTexCapDim = 0;
void (*g_onProxyTextureDestroyed)(void* realTexture) = nullptr;

static UINT atLeast1(UINT v) { return v ? v : 1u; }   // a mip edge never reaches 0

// Block-compressed formats cost per 4x4 block; everything else per texel. Unknown formats are
// priced at 4 bytes/texel, which over-reports rather than hides (and, for a scratch level, over-
// allocates rather than under-).
static void formatCost(D3DFORMAT format, uint32_t* blockBytes, uint32_t* texelBytes) {
    *blockBytes = 0;
    *texelBytes = 4;
    switch ((DWORD)format) {
        case MAKEFOURCC('D', 'X', 'T', '1'): *blockBytes = 8; break;
        case MAKEFOURCC('D', 'X', 'T', '2'): case MAKEFOURCC('D', 'X', 'T', '3'):
        case MAKEFOURCC('D', 'X', 'T', '4'): case MAKEFOURCC('D', 'X', 'T', '5'): *blockBytes = 16; break;
        case D3DFMT_R5G6B5: case D3DFMT_X1R5G5B5: case D3DFMT_A1R5G5B5: case D3DFMT_A4R4G4B4:
        case D3DFMT_X4R4G4B4: case D3DFMT_A8L8: case D3DFMT_A8R3G3B2: *texelBytes = 2; break;
        case D3DFMT_A8: case D3DFMT_L8: case D3DFMT_P8: case D3DFMT_R3G3B2: case D3DFMT_A4L4: *texelBytes = 1; break;
        default: break;
    }
}

uint32_t proxyTextureBytes(UINT width, UINT height, UINT levels, D3DFORMAT format) {
    uint32_t blockBytes, texelBytes;
    formatCost(format, &blockBytes, &texelBytes);
    uint64_t total = 0;
    UINT w = width, h = height;
    for (UINT l = 0; l < (levels ? levels : 1u); ++l) {
        total += blockBytes ? (uint64_t)((w + 3) / 4) * ((h + 3) / 4) * blockBytes
                            : (uint64_t)w * h * texelBytes;
        w = w > 1 ? w >> 1 : 1; h = h > 1 ? h >> 1 : 1;
    }
    return total > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)total;
}

UINT proxyTextureCapSkip(UINT w, UINT h, UINT levels, D3DFORMAT format) {
    const UINT cap = g_proxyTexCapDim;
    if (cap == 0 || levels == 1) { return 0; }   // no chain to drop from (UI, splash, targets)
    UINT skip = 0;
    while (((w >> skip) > cap || (h >> skip) > cap) && skip < ProxyTexture::kMaxCapSkip) { ++skip; }
    if (skip == 0) { return 0; }
    if (levels != 0 && levels <= skip) { return 0; }   // nothing would survive below the drop
    if ((w >> skip) == 0 || (h >> skip) == 0) { return 0; }
    uint32_t blockBytes, texelBytes;
    formatCost(format, &blockBytes, &texelBytes);
    if (blockBytes && (((w >> skip) & 3u) || ((h >> skip) & 3u))) { return 0; }   // whole blocks only
    return skip;
}

void ProxyTexture::capArm(UINT skip, UINT w, UINT h, D3DFORMAT format) {
    capSkip = skip;
    capWidth = w;
    capHeight = h;
    capFormat = format;
}

void ProxyTexture::capNoteFilled() {
    if (capSkip && !capFilled) {
        capFilled = true;
        ++proxyTexLedger().capFilled;
    }
}

static void ledgerAdjust(uint32_t bytes, int sign) {
    ProxyTexLedger& L = proxyTexLedger();
    const uint64_t b = bytes;
    if (sign > 0) { L.liveBytes += b; ++L.liveCount; L.createdBytes += b; ++L.createdCount; }
    else          { L.liveBytes -= b; --L.liveCount; }
    if (bytes >= (4u << 20))      { if (sign > 0) { L.bigBytes += b; ++L.bigCount; } else { L.bigBytes -= b; --L.bigCount; } }
    else if (bytes >= (1u << 20)) { if (sign > 0) { L.midBytes += b; ++L.midCount; } else { L.midBytes -= b; --L.midCount; } }
}

void ProxyTexture::ledgerTrack(uint32_t bytes) {
    if (ledgerBytes || !bytes) { return; }
    ledgerBytes = bytes;
    ledgerAdjust(bytes, +1);
}

ProxyTexture::ProxyTexture(IDirect3DTexture9* real, ProxyDevice* device) : realTexture(real), proxDevice(device) {
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
        if (ledgerBytes) { ledgerAdjust(ledgerBytes, -1); }
        if (g_onProxyTextureDestroyed) { g_onProxyTextureDestroyed(realTexture); }
        for (UINT l = 0; l < kMaxCapSkip; ++l) { std::free(capScratch[l]); }   // locked, never unlocked
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

// Levels are VIRTUAL throughout (see the class comment): Morrowind's level L is the real level
// L - capSkip, and every translation below is the identity when capSkip == 0.
DWORD _stdcall ProxyTexture::SetLOD(DWORD LODNew) {
    return realTexture->SetLOD(LODNew > capSkip ? LODNew - capSkip : 0) + capSkip;
}
DWORD _stdcall ProxyTexture::GetLOD() {
    return realTexture->GetLOD() + capSkip;
}

//-----------------------------------------------------------------------------

DWORD _stdcall ProxyTexture::GetLevelCount() {
    return realTexture->GetLevelCount() + capSkip;
}

HRESULT _stdcall ProxyTexture::GetLevelDesc(UINT Level, D3DSURFACE_DESC8* pDesc) {
    // A dropped level describes itself from the real top level, at the size Morrowind expects.
    const bool dropped = Level < capSkip;
    D3DSURFACE_DESC b2;
    HRESULT hr = realTexture->GetLevelDesc(dropped ? 0 : Level - capSkip, &b2);
    if (hr == D3D_OK) {
        pDesc->Format = b2.Format;
        pDesc->Height = dropped ? atLeast1(capHeight >> Level) : b2.Height;
        pDesc->MultiSampleType = b2.MultiSampleType;
        pDesc->Pool = b2.Pool;
        pDesc->Size = 0; // TODO: Fix;
        pDesc->Type = b2.Type;
        pDesc->Usage = b2.Usage;
        pDesc->Width = dropped ? atLeast1(capWidth >> Level) : b2.Width;
    }
    return hr;
}

HRESULT _stdcall ProxyTexture::GetSurfaceLevel(UINT Level, IDirect3DSurface8** ppSurfaceLevel) {
    IDirect3DSurface9* surface_real = NULL;
    *ppSurfaceLevel = NULL;

    if (Level < capSkip) {
        // A dropped level has no surface. Hand out a system-memory one of the size Morrowind
        // expects: whatever it writes there is discarded with the surface.
        HRESULT hr = proxDevice->realDevice->CreateOffscreenPlainSurface(
            atLeast1(capWidth >> Level), atLeast1(capHeight >> Level), capFormat,
            D3DPOOL_SYSTEMMEM, &surface_real, NULL);
        if (hr != D3D_OK || surface_real == NULL) {
            return hr;
        }
        ++proxyTexLedger().capScratchSurfaces;
        *ppSurfaceLevel = proxDevice->factoryProxySurface(surface_real);
        return D3D_OK;
    }

    HRESULT hr = realTexture->GetSurfaceLevel(Level - capSkip, &surface_real);
    if (hr != D3D_OK || surface_real == NULL) {
        return hr;
    }
    capNoteFilled();

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
    if (Level < capSkip) {
        // A dropped level: Morrowind writes it into scratch, freed at UnlockRect.
        if (!pLockedRect) {
            return D3DERR_INVALIDCALL;
        }
        uint32_t blockBytes, texelBytes;
        formatCost(capFormat, &blockBytes, &texelBytes);
        const UINT w = atLeast1(capWidth >> Level), h = atLeast1(capHeight >> Level);
        const UINT pitch = blockBytes ? ((w + 3) / 4) * blockBytes : w * texelBytes;
        const UINT rows  = blockBytes ? (h + 3) / 4 : h;
        if (!capScratch[Level]) {
            capScratch[Level] = std::malloc((size_t)pitch * rows);
            if (!capScratch[Level]) {
                return E_OUTOFMEMORY;
            }
        }
        uint8_t* bits = static_cast<uint8_t*>(capScratch[Level]);
        if (pRect) {
            bits += blockBytes ? (size_t)(pRect->top / 4) * pitch + (size_t)(pRect->left / 4) * blockBytes
                               : (size_t)pRect->top * pitch + (size_t)pRect->left * texelBytes;
        }
        pLockedRect->pBits = bits;
        pLockedRect->Pitch = (INT)pitch;
        ++proxyTexLedger().capScratchLocks;
        return D3D_OK;
    }
    HRESULT hr = realTexture->LockRect(Level - capSkip, pLockedRect, pRect, Flags);
    if (hr == D3D_OK) {
        capNoteFilled();
    }
    return hr;
}
HRESULT _stdcall ProxyTexture::UnlockRect(UINT Level) {
    if (Level < capSkip) {
        std::free(capScratch[Level]);
        capScratch[Level] = nullptr;
        return D3D_OK;
    }
    proxyTexNoteWrite(realTexture);
    return realTexture->UnlockRect(Level - capSkip);
}
HRESULT _stdcall ProxyTexture::AddDirtyRect(CONST RECT* pDirtyRect ) {
    // The rect is in level-0 texels; on a capped texture that is the wrong scale, so dirty it all.
    return realTexture->AddDirtyRect(capSkip ? NULL : pDirtyRect);
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
