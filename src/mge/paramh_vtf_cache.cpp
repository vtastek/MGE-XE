// Phase 8C: R16F _paramh VTF cache — implementation.
#include "paramh_vtf_cache.h"
#include "support/log.h"
#include "mge_tracy.h"

#include <d3dx9.h>
#include <unordered_map>

namespace ParamHVTF {

namespace {

struct Entry {
    IDirect3DTexture9* tex = nullptr;  // R16F, D3DPOOL_DEFAULT, top-mip only
};

std::unordered_map<IDirect3DBaseTexture9*, Entry> s_cache;
SRWLOCK s_lock = SRWLOCK_INIT;

// One-time VTF support query. -1 = untested, 0 = unsupported, 1 = supported.
int s_vtfSupportState = -1;
SRWLOCK s_supportLock = SRWLOCK_INIT;

double s_totalBuildMs = 0.0;
unsigned int s_buildCount = 0;

bool queryVTFSupport(IDirect3DDevice9* device) {
    if (!device) return false;
    IDirect3D9* d3d = nullptr;
    if (FAILED(device->GetDirect3D(&d3d)) || !d3d) return false;
    D3DDEVICE_CREATION_PARAMETERS cp;
    if (FAILED(device->GetCreationParameters(&cp))) {
        d3d->Release();
        return false;
    }
    D3DDISPLAYMODE dm;
    if (FAILED(d3d->GetAdapterDisplayMode(cp.AdapterOrdinal, &dm))) {
        d3d->Release();
        return false;
    }
    HRESULT hr = d3d->CheckDeviceFormat(cp.AdapterOrdinal, cp.DeviceType, dm.Format,
                                        D3DUSAGE_QUERY_VERTEXTEXTURE, D3DRTYPE_TEXTURE,
                                        D3DFMT_R16F);
    d3d->Release();
    return SUCCEEDED(hr);
}

} // anonymous namespace

bool isSupported(IDirect3DDevice9* device) {
    AcquireSRWLockShared(&s_supportLock);
    int cached = s_vtfSupportState;
    ReleaseSRWLockShared(&s_supportLock);
    if (cached >= 0) return cached == 1;

    bool supported = queryVTFSupport(device);
    AcquireSRWLockExclusive(&s_supportLock);
    if (s_vtfSupportState < 0) {
        s_vtfSupportState = supported ? 1 : 0;
        LOG::logline("ParamHVTF: D3DFMT_R16F vertex-texture support = %s",
                     supported ? "yes" : "no (VTF displacement mode will be disabled)");
    } else {
        supported = (s_vtfSupportState == 1);
    }
    ReleaseSRWLockExclusive(&s_supportLock);
    return supported;
}

IDirect3DTexture9* getOrBuild(IDirect3DDevice9* device, IDirect3DBaseTexture9* paramhSource) {
    MGE_ZoneScopedN("ParamHVTF_getOrBuild");
    if (!device || !paramhSource) return nullptr;
    if (!isSupported(device)) return nullptr;

    // Fast path: cached.
    AcquireSRWLockShared(&s_lock);
    auto it = s_cache.find(paramhSource);
    if (it != s_cache.end()) {
        IDirect3DTexture9* t = it->second.tex;
        ReleaseSRWLockShared(&s_lock);
        return t;
    }
    ReleaseSRWLockShared(&s_lock);

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    // Query source dimensions from the top mip.
    IDirect3DTexture9* src2d = nullptr;
    if (FAILED(paramhSource->QueryInterface(IID_IDirect3DTexture9, (void**)&src2d)) || !src2d) {
        return nullptr;
    }
    D3DSURFACE_DESC desc;
    if (FAILED(src2d->GetLevelDesc(0, &desc))) {
        src2d->Release();
        return nullptr;
    }
    IDirect3DSurface9* srcSurf = nullptr;
    if (FAILED(src2d->GetSurfaceLevel(0, &srcSurf))) {
        src2d->Release();
        return nullptr;
    }

    // Create R16F target in DEFAULT pool so VTF can sample from it.
    IDirect3DTexture9* dstTex = nullptr;
    HRESULT hr = device->CreateTexture(desc.Width, desc.Height, 1, 0, D3DFMT_R16F,
                                       D3DPOOL_DEFAULT, &dstTex, nullptr);
    if (FAILED(hr) || !dstTex) {
        srcSurf->Release(); src2d->Release();
        LOG::logline("ParamHVTF: CreateTexture(R16F DEFAULT %ux%u) failed hr=0x%08x",
                     desc.Width, desc.Height, (unsigned)hr);
        return nullptr;
    }
    IDirect3DSurface9* dstTop = nullptr;
    if (FAILED(dstTex->GetSurfaceLevel(0, &dstTop))) {
        dstTex->Release(); srcSurf->Release(); src2d->Release();
        return nullptr;
    }

    // Staging R16F in SYSTEMMEM for format conversion from DXT; then UpdateSurface
    // copies into the VTF-sampleable DEFAULT pool texture. D3DXLoadSurfaceFromSurface
    // handles DXT→R16F decode; filtering copies green channel as-is via .g source.
    IDirect3DSurface9* stageSurf = nullptr;
    hr = device->CreateOffscreenPlainSurface(desc.Width, desc.Height, D3DFMT_R16F,
                                             D3DPOOL_SYSTEMMEM, &stageSurf, nullptr);
    if (FAILED(hr) || !stageSurf) {
        dstTop->Release(); dstTex->Release(); srcSurf->Release(); src2d->Release();
        LOG::logline("ParamHVTF: CreateOffscreenPlainSurface(R16F SYSTEMMEM) failed hr=0x%08x",
                     (unsigned)hr);
        return nullptr;
    }

    // D3DXLoadSurfaceFromSurface maps source RGBA → dest R for R16F destinations:
    // the red channel of the converted pixel is written. _paramh's *green* holds
    // the height, so we first decode DXT to A8R8G8B8 in systemmem, swizzle G→R,
    // then push to R16F. Two-step, but correctness > elegance for a one-time build.
    IDirect3DSurface9* rgbaStage = nullptr;
    hr = device->CreateOffscreenPlainSurface(desc.Width, desc.Height, D3DFMT_A8R8G8B8,
                                             D3DPOOL_SYSTEMMEM, &rgbaStage, nullptr);
    if (FAILED(hr) || !rgbaStage) {
        stageSurf->Release(); dstTop->Release(); dstTex->Release();
        srcSurf->Release(); src2d->Release();
        LOG::logline("ParamHVTF: CreateOffscreenPlainSurface(A8R8G8B8 SYSTEMMEM) failed hr=0x%08x",
                     (unsigned)hr);
        return nullptr;
    }
    hr = D3DXLoadSurfaceFromSurface(rgbaStage, nullptr, nullptr, srcSurf, nullptr, nullptr,
                                    D3DX_FILTER_NONE, 0);
    if (FAILED(hr)) {
        rgbaStage->Release(); stageSurf->Release();
        dstTop->Release(); dstTex->Release(); srcSurf->Release(); src2d->Release();
        LOG::logline("ParamHVTF: D3DXLoadSurfaceFromSurface(DXT→A8R8G8B8) failed hr=0x%08x",
                     (unsigned)hr);
        return nullptr;
    }
    // Swizzle G→R into the R16F stage. Lock both.
    D3DLOCKED_RECT rgbaLock = {}, r16Lock = {};
    if (FAILED(rgbaStage->LockRect(&rgbaLock, nullptr, D3DLOCK_READONLY))) {
        rgbaStage->Release(); stageSurf->Release();
        dstTop->Release(); dstTex->Release(); srcSurf->Release(); src2d->Release();
        return nullptr;
    }
    if (FAILED(stageSurf->LockRect(&r16Lock, nullptr, 0))) {
        rgbaStage->UnlockRect(); rgbaStage->Release(); stageSurf->Release();
        dstTop->Release(); dstTex->Release(); srcSurf->Release(); src2d->Release();
        return nullptr;
    }
    for (UINT y = 0; y < desc.Height; ++y) {
        const uint8_t* srcRow = (const uint8_t*)rgbaLock.pBits + y * rgbaLock.Pitch;
        uint16_t*      dstRow = (uint16_t*)     ((uint8_t*)r16Lock.pBits + y * r16Lock.Pitch);
        for (UINT x = 0; x < desc.Width; ++x) {
            // A8R8G8B8 little-endian -> bytes are B, G, R, A. Green at offset 1.
            float g = srcRow[x * 4 + 1] * (1.0f / 255.0f);
            // Pack to half-float (D3DFMT_R16F). Use D3DXFloat32To16Array for safety.
            D3DXFLOAT16 h;
            D3DXFloat32To16Array(&h, &g, 1);
            dstRow[x] = *reinterpret_cast<uint16_t*>(&h);
        }
    }
    stageSurf->UnlockRect();
    rgbaStage->UnlockRect();

    hr = device->UpdateSurface(stageSurf, nullptr, dstTop, nullptr);
    stageSurf->Release();
    rgbaStage->Release();
    dstTop->Release();
    srcSurf->Release();
    src2d->Release();
    if (FAILED(hr)) {
        dstTex->Release();
        LOG::logline("ParamHVTF: UpdateSurface (SYSTEMMEM R16F -> DEFAULT R16F) failed hr=0x%08x",
                     (unsigned)hr);
        return nullptr;
    }

    QueryPerformanceCounter(&t1);
    double ms = (t1.QuadPart - t0.QuadPart) * 1000.0 / freq.QuadPart;

    // Insert. If two threads race on the same source, keep the first winner and
    // release the duplicate — the cache lookup below would otherwise leak.
    AcquireSRWLockExclusive(&s_lock);
    auto [insIt, inserted] = s_cache.emplace(paramhSource, Entry{dstTex});
    IDirect3DTexture9* winner = insIt->second.tex;
    if (!inserted && winner != dstTex) {
        ReleaseSRWLockExclusive(&s_lock);
        dstTex->Release();
        return winner;
    }
    s_totalBuildMs += ms;
    ++s_buildCount;
    ReleaseSRWLockExclusive(&s_lock);

    LOG::logline("ParamHVTF: built R16F VTF for %p  %ux%u  %.2f ms",
                 paramhSource, desc.Width, desc.Height, ms);
    return dstTex;
}

void clearAll() {
    AcquireSRWLockExclusive(&s_lock);
    for (auto& kv : s_cache) {
        if (kv.second.tex) kv.second.tex->Release();
    }
    s_cache.clear();
    ReleaseSRWLockExclusive(&s_lock);
}

void onTextureReleased(IDirect3DBaseTexture9* tex) {
    if (!tex) return;
    AcquireSRWLockExclusive(&s_lock);
    auto it = s_cache.find(tex);
    if (it != s_cache.end()) {
        if (it->second.tex) it->second.tex->Release();
        s_cache.erase(it);
    }
    ReleaseSRWLockExclusive(&s_lock);
}

double getTotalBuildMs() {
    AcquireSRWLockShared(&s_lock);
    double v = s_totalBuildMs;
    ReleaseSRWLockShared(&s_lock);
    return v;
}

unsigned int getBuildCount() {
    AcquireSRWLockShared(&s_lock);
    unsigned int v = s_buildCount;
    ReleaseSRWLockShared(&s_lock);
    return v;
}

} // namespace ParamHVTF
