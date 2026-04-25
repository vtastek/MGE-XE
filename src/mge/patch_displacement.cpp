// Phase 7: near-camera subdivided landscape patch builder + cache.
// See header for design intent. This file is all CPU work: lock source VB/IB,
// expand the 5x5 grid, sample _paramh alpha channel for per-vertex signed
// displacement, zero the perimeter, and upload to managed VB/IB.
#include "patch_displacement.h"

#include "support/log.h"
#include "texture_suffix.h"
#include "morrowindbsa.h"
#include "mge_tracy.h"

#include <d3dx9.h>
#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace PatchDisplacement {

namespace {

constexpr UINT kSrcGrid = 5;                    // Source patch is 5x5 verts (stride-4 spec)
constexpr UINT kSrcVerts = kSrcGrid * kSrcGrid; // 25
constexpr UINT kSrcTris  = 32;                  // 4x4 quads * 2 tris

// Phase 8.4: destination grid density is tier-dependent. Tier 0 = inner 65x65
// (4225 verts, 8192 tris), tier 1 = outer 33x33 (1089 verts, 2048 tris). Both
// fit in a 16-bit IB. Dst edges always land on src edges because (kDstGrid-1)
// is a multiple of (kSrcGrid-1) for both tiers (64=16*4, 32=8*4), but the
// subdivision is correct at any density — perimeter verts linearly interpolate
// between adjacent src corners, staying collinear with non-subdivided neighbor
// edge polylines.

struct VertexLayout {
    int posOffset = -1;
    int normalOffset = -1;
    int colorOffset = -1;
    int uvOffset = -1;
    UINT stride = 0;

    bool parse(DWORD fvf, UINT srcStride) {
        int off = 0;
        DWORD posMask = fvf & D3DFVF_POSITION_MASK;
        if (posMask != D3DFVF_XYZ) return false;
        posOffset = off; off += 12;
        if (fvf & D3DFVF_NORMAL) { normalOffset = off; off += 12; }
        if (fvf & D3DFVF_PSIZE)  off += 4;
        if (fvf & D3DFVF_DIFFUSE) { colorOffset = off; off += 4; }
        if (fvf & D3DFVF_SPECULAR) off += 4;
        DWORD texCount = (fvf & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT;
        if (texCount >= 1) { uvOffset = off; off += 8; }
        for (DWORD i = 1; i < texCount; ++i) off += 8;
        stride = (UINT)off;
        return stride <= srcStride;
    }
};

struct HeightMap {
    std::vector<uint8_t> alpha;
    int width = 0;
    int height = 0;
};

// Build a world-frame cache so multiple patches sharing the same _paramh
// don't re-decode. Keyed on the base texture pointer (Morrowind reuses these).
struct HeightCacheKey {
    IDirect3DBaseTexture9* tex;
    bool operator==(const HeightCacheKey& o) const { return tex == o.tex; }
};
struct HeightCacheKeyHash {
    size_t operator()(const HeightCacheKey& k) const { return (size_t)(uintptr_t)k.tex; }
};

struct PatchCacheEntry {
    std::unique_ptr<SubdivPatch> patch;
    // Height scale at build time — a scale change via ImGui forces rebuild.
    float heightScale = 0.0f;
    // Overlay texture at build time — a flip forces rebuild (Phase 6 absorption flip).
    IDirect3DBaseTexture9* overlayTexture = nullptr;
    // _paramh base/overlay texture pointers captured at build time; if either
    // underlying heightmap pointer changes we rebuild.
    IDirect3DBaseTexture9* baseParamH = nullptr;
    IDirect3DBaseTexture9* overlayParamH = nullptr;
    // Phase 8.2: subdivided-neighbor direction mask at build time. When the
    // near-set composition changes (tile enters/leaves), the mask flips and
    // we must rebuild so edge-locked verts switch state accordingly — stale
    // caches were producing asymmetric seams between adjacent tiles.
    uint8_t subdivNeighborDirMask = 0;
    // Phase 8.4: tier that produced this cached mesh (0 = inner 65x65, 1 =
    // outer 33x33). The same VB+IB patch legitimately switches tier as the
    // player moves through the inner/outer radii, so we must rebuild when it
    // changes.
    uint8_t subdivTier = 0;
    // Phase 8.7: neighbor-context hash at build time. Captures all near-set
    // neighbors' paramh identities. When any neighbor's texture changes, the
    // averaged edge values in the coalesced map change, and this cache entry
    // must rebuild so the baked edge heights stay consistent with both sides.
    uint32_t edgeContextHash = 0;
    // Displacement curve (gamma/pivot) at build time; a slider change forces rebuild.
    float dispGamma = 1.0f;
    float dispPivot = 1.0f;
};

// Patch cache — key is VB+IB identity, value is the latest built patch.
std::unordered_map<FixedFunctionShader::TerrainPatchKey, PatchCacheEntry,
                   FixedFunctionShader::TerrainPatchKeyHash> s_patchCache;

// Heightmap cache — keyed on the TextureSuffix resolution "variants" pointer
// (stable for a given base texture) via the texture pointer.
std::unordered_map<HeightCacheKey, HeightMap, HeightCacheKeyHash> s_heightCache;

SRWLOCK s_lock = SRWLOCK_INIT;

// Decode a heightmap to a cached 8-bit alpha-channel buffer. Top-mip is
// converted to A8R8G8B8 via D3DXLoadSurfaceFromSurface which handles any
// source format (uncompressed, DXT1/3/5, etc.) — much simpler than writing
// format-specific decoders. We clamp the readback to 256x256; we only sample
// it 81 times per patch, so higher resolution buys nothing.
const HeightMap* getOrDecodeHeight(IDirect3DDevice9* device, IDirect3DBaseTexture9* tex) {
    if (!tex) return nullptr;
    AcquireSRWLockShared(&s_lock);
    auto it = s_heightCache.find({tex});
    if (it != s_heightCache.end()) {
        const HeightMap* r = &it->second;
        ReleaseSRWLockShared(&s_lock);
        return r;
    }
    ReleaseSRWLockShared(&s_lock);

    IDirect3DTexture9* tex2d = nullptr;
    if (FAILED(tex->QueryInterface(IID_IDirect3DTexture9, (void**)&tex2d)) || !tex2d) {
        return nullptr;
    }
    D3DSURFACE_DESC desc;
    if (FAILED(tex2d->GetLevelDesc(0, &desc))) {
        tex2d->Release();
        return nullptr;
    }
    IDirect3DSurface9* srcSurf = nullptr;
    if (FAILED(tex2d->GetSurfaceLevel(0, &srcSurf))) {
        tex2d->Release();
        return nullptr;
    }
    UINT w = std::min<UINT>(desc.Width,  256u);
    UINT h = std::min<UINT>(desc.Height, 256u);
    IDirect3DSurface9* dstSurf = nullptr;
    HRESULT hr = device->CreateOffscreenPlainSurface(w, h, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, &dstSurf, nullptr);
    if (FAILED(hr)) {
        srcSurf->Release(); tex2d->Release();
        return nullptr;
    }
    hr = D3DXLoadSurfaceFromSurface(dstSurf, nullptr, nullptr, srcSurf, nullptr, nullptr, D3DX_FILTER_LINEAR, 0);
    if (FAILED(hr)) {
        dstSurf->Release(); srcSurf->Release(); tex2d->Release();
        return nullptr;
    }
    D3DLOCKED_RECT lr;
    if (FAILED(dstSurf->LockRect(&lr, nullptr, D3DLOCK_READONLY))) {
        dstSurf->Release(); srcSurf->Release(); tex2d->Release();
        return nullptr;
    }
    HeightMap hm;
    hm.width = (int)w;
    hm.height = (int)h;
    hm.alpha.resize((size_t)w * h);
    for (UINT y = 0; y < h; ++y) {
        const uint8_t* row = (const uint8_t*)lr.pBits + y * lr.Pitch;
        for (UINT x = 0; x < w; ++x) {
            // A8R8G8B8 in D3D is 0xAARRGGBB stored little-endian as B,G,R,A.
            // _paramh height is alpha, which sits at byte offset 3.
            hm.alpha[(size_t)y * w + x] = row[x * 4 + 3];
        }
    }
    dstSurf->UnlockRect();
    dstSurf->Release();
    srcSurf->Release();
    tex2d->Release();

    AcquireSRWLockExclusive(&s_lock);
    auto [insIt, inserted] = s_heightCache.emplace(HeightCacheKey{tex}, std::move(hm));
    const HeightMap* r = &insIt->second;
    ReleaseSRWLockExclusive(&s_lock);
    return r;
}

// Sample a heightmap with bilinear interpolation, wrapping UVs, and apply a
// one-sided crevice displacement. Alpha 1 keeps original terrain height; alpha
// 0 pushes down by the full scale.
//
//   h01 = bilinear(_paramh.a) / 255
//   depth01 = 1 - h01
//   shaped = pow(depth01, 1/gamma) / pivot
//   out = -saturate(shaped) * scale
float sampleHeight(const HeightMap* hm, float u, float v, float scale,
                   float gamma, float pivot) {
    if (!hm) return 0.0f;
    // Wrap UV
    u -= floorf(u);
    v -= floorf(v);
    float fx = u * hm->width  - 0.5f;
    float fy = v * hm->height - 0.5f;
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    float tx = fx - x0;
    float ty = fy - y0;
    auto wrap = [](int x, int n) {
        x %= n;
        if (x < 0) x += n;
        return x;
    };
    int xa = wrap(x0,     hm->width);
    int xb = wrap(x0 + 1, hm->width);
    int ya = wrap(y0,     hm->height);
    int yb = wrap(y0 + 1, hm->height);
    auto heightAt = [&](int xi, int yi) { return (float)hm->alpha[(size_t)yi * hm->width + xi] / 255.0f; };
    float a = heightAt(xa, ya) * (1 - tx) + heightAt(xb, ya) * tx;
    float b = heightAt(xa, yb) * (1 - tx) + heightAt(xb, yb) * tx;
    float h01 = a * (1 - ty) + b * ty;
    float depth01 = std::min(1.0f, std::max(0.0f, 1.0f - h01));
    float mag = powf(depth01, 1.0f / std::max(gamma, 1e-4f));
    mag = std::min(1.0f, mag / std::max(pivot, 1e-4f));
    return -mag * scale;
}

// Locate _paramh for the base color texture via TextureSuffix's existing
// resolution cache — we must not introduce a new suffix plumbing path.
IDirect3DBaseTexture9* resolveParamH(IDirect3DDevice9* device, IDirect3DBaseTexture9* baseTex) {
    if (!baseTex) return nullptr;
    IDirect3DTexture9* base2d = nullptr;
    if (FAILED(baseTex->QueryInterface(IID_IDirect3DTexture9, (void**)&base2d)) || !base2d) {
        return nullptr;
    }
    const TextureSuffix::ResolutionCache* res =
        TextureSuffix::getOrCreateResolution(device, base2d, /*allowDeviceCalls*/ true);
    base2d->Release();
    if (!res || !res->variants || !res->variants->hasParamH()) return nullptr;
    IDirect3DTexture9* paramH = BSA::loadSuffixTexture(device, *res->variants, "paramh");
    return paramH;
}

void bilerpBytes(const uint8_t* src, int sstride,
                 const VertexLayout& srcLayout,
                 int i00, int i10, int i01, int i11,
                 float fr, float fc,
                 uint8_t* dst, const VertexLayout& dstLayout,
                 UINT dstStride)
{
    // Zero-init so non-parsed trailing bytes stay clean.
    std::memset(dst, 0, dstStride);

    auto readF3 = [&](int i, int off) {
        const float* p = (const float*)(src + i * sstride + off);
        return D3DXVECTOR3{p[0], p[1], p[2]};
    };
    auto writeF3 = [&](int off, const D3DXVECTOR3& v) {
        float* p = (float*)(dst + off);
        p[0] = v.x; p[1] = v.y; p[2] = v.z;
    };
    auto bilerpV3 = [&](int off) {
        D3DXVECTOR3 v00 = readF3(i00, off);
        D3DXVECTOR3 v10 = readF3(i10, off);
        D3DXVECTOR3 v01 = readF3(i01, off);
        D3DXVECTOR3 v11 = readF3(i11, off);
        D3DXVECTOR3 a = v00 * (1 - fc) + v10 * fc;
        D3DXVECTOR3 b = v01 * (1 - fc) + v11 * fc;
        return a * (1 - fr) + b * fr;
    };

    if (srcLayout.posOffset >= 0) {
        writeF3(dstLayout.posOffset, bilerpV3(srcLayout.posOffset));
    }
    if (srcLayout.normalOffset >= 0) {
        D3DXVECTOR3 n = bilerpV3(srcLayout.normalOffset);
        D3DXVec3Normalize(&n, &n);
        writeF3(dstLayout.normalOffset, n);
    }
    if (srcLayout.colorOffset >= 0) {
        // D3DCOLOR is 0xAARRGGBB packed; lerp each byte.
        DWORD c00 = *(const DWORD*)(src + i00 * sstride + srcLayout.colorOffset);
        DWORD c10 = *(const DWORD*)(src + i10 * sstride + srcLayout.colorOffset);
        DWORD c01 = *(const DWORD*)(src + i01 * sstride + srcLayout.colorOffset);
        DWORD c11 = *(const DWORD*)(src + i11 * sstride + srcLayout.colorOffset);
        DWORD out = 0;
        for (int byte = 0; byte < 4; ++byte) {
            float b00 = (float)((c00 >> (byte * 8)) & 0xff);
            float b10 = (float)((c10 >> (byte * 8)) & 0xff);
            float b01 = (float)((c01 >> (byte * 8)) & 0xff);
            float b11 = (float)((c11 >> (byte * 8)) & 0xff);
            float a = b00 * (1 - fc) + b10 * fc;
            float bb = b01 * (1 - fc) + b11 * fc;
            float v = a * (1 - fr) + bb * fr;
            out |= ((DWORD)std::clamp(v + 0.5f, 0.0f, 255.0f)) << (byte * 8);
        }
        *(DWORD*)(dst + dstLayout.colorOffset) = out;
    }
    if (srcLayout.uvOffset >= 0) {
        const float* u00 = (const float*)(src + i00 * sstride + srcLayout.uvOffset);
        const float* u10 = (const float*)(src + i10 * sstride + srcLayout.uvOffset);
        const float* u01 = (const float*)(src + i01 * sstride + srcLayout.uvOffset);
        const float* u11 = (const float*)(src + i11 * sstride + srcLayout.uvOffset);
        float ux_a = u00[0] * (1 - fc) + u10[0] * fc;
        float ux_b = u01[0] * (1 - fc) + u11[0] * fc;
        float uy_a = u00[1] * (1 - fc) + u10[1] * fc;
        float uy_b = u01[1] * (1 - fc) + u11[1] * fc;
        float* du = (float*)(dst + dstLayout.uvOffset);
        du[0] = ux_a * (1 - fr) + ux_b * fr;
        du[1] = uy_a * (1 - fr) + uy_b * fr;
    }
}

} // anonymous namespace

SubdivPatch::~SubdivPatch() {
    if (vb) { vb->Release(); vb = nullptr; }
    if (ib) { ib->Release(); ib = nullptr; }
}

SubdivPatch* findCached(const FixedFunctionShader::TerrainPatchKey& key,
                        const FixedFunctionShader::HLSLRecordedCall& call,
                        IDirect3DBaseTexture9* overlayTex,
                        float heightScale,
                        uint8_t subdivTier,
                        float dispGamma,
                        float dispPivot)
{
    AcquireSRWLockShared(&s_lock);
    auto cit = s_patchCache.find(key);
    SubdivPatch* r = nullptr;
    if (cit != s_patchCache.end()
        && cit->second.overlayTexture == overlayTex
        && cit->second.heightScale == heightScale
        && cit->second.subdivNeighborDirMask == call.subdivNeighborDirMask
        && cit->second.subdivTier == subdivTier
        && cit->second.edgeContextHash == call.edgeContextHash
        && cit->second.dispGamma == dispGamma
        && cit->second.dispPivot == dispPivot)
    {
        r = cit->second.patch.get();
    }
    ReleaseSRWLockShared(&s_lock);
    return r;
}

SubdivPatch* getOrBuild(IDirect3DDevice9* device,
                        const FixedFunctionShader::TerrainPatchKey& key,
                        const FixedFunctionShader::HLSLRecordedCall& call,
                        IDirect3DBaseTexture9* overlayTex,
                        float heightScale,
                        uint8_t subdivTier,
                        float dispGamma,
                        float dispPivot,
                        const NearPatchEdgeHeights* edgeHeights)
{
    MGE_ZoneScopedN("PatchDisplacement_getOrBuild");
    if (!device || !key.vb || !key.ib) return nullptr;

    // Phase 8.4: per-tier density. Tier 0 = inner 65x65, tier 1 = outer 33x33.
    const UINT kDstGrid  = (subdivTier == 0) ? 65u : 33u;
    const UINT kDstVerts = kDstGrid * kDstGrid;
    const UINT kDstTris  = (kDstGrid - 1) * (kDstGrid - 1) * 2;
    const UINT kDstIdx   = kDstTris * 3;
    const float kStep    = float(kSrcGrid - 1) / float(kDstGrid - 1);

    // Fast path: cached entry that still matches.
    AcquireSRWLockShared(&s_lock);
    auto cit = s_patchCache.find(key);
    if (cit != s_patchCache.end()
        && cit->second.overlayTexture == overlayTex
        && cit->second.heightScale == heightScale
        && cit->second.subdivNeighborDirMask == call.subdivNeighborDirMask
        && cit->second.subdivTier == subdivTier
        && cit->second.edgeContextHash == call.edgeContextHash
        && cit->second.dispGamma == dispGamma
        && cit->second.dispPivot == dispPivot)
    {
        SubdivPatch* r = cit->second.patch.get();
        ReleaseSRWLockShared(&s_lock);
        return r;
    }
    ReleaseSRWLockShared(&s_lock);

    // Validate source shape. Morrowind landscape patches are 25 verts / 32 tris
    // per spec §stride-4. Bail quietly on anything else (grass, custom meshes).
    const auto& rs = call.rs;
    if (rs.vertCount != kSrcVerts || rs.primCount != kSrcTris) return nullptr;
    if (rs.primType != D3DPT_TRIANGLELIST) return nullptr;

    VertexLayout srcLayout;
    if (!srcLayout.parse(rs.fvf, rs.vbStride)) return nullptr;
    // Must have POSITION + COLOR at minimum (blend mask lives in color.a) and a UV
    // (needed for _paramh sampling). NORMAL is needed for VS displacement axis.
    if (srcLayout.posOffset < 0 || srcLayout.colorOffset < 0
        || srcLayout.uvOffset < 0 || srcLayout.normalOffset < 0)
    {
        return nullptr;
    }

    // Lock source VB read-only. Mirror the meshlodcache pattern — DONOTWAIT so
    // we never stall if Morrowind happens to own the buffer this frame.
    void* srcVbPtr = nullptr;
    HRESULT hr = key.vb->Lock(rs.vbOffset, 0, &srcVbPtr, D3DLOCK_READONLY | D3DLOCK_DONOTWAIT);
    if (FAILED(hr) || !srcVbPtr) {
        LOG::logline("[DISPCACHE][BUILD-LOCK-FAIL] phase=VB vb=%p ib=%p hr=0x%08x ptr=%p tier=%u neigh=%02x",
                     key.vb, key.ib, (unsigned)hr, srcVbPtr,
                     (unsigned)call.subdivTier, (unsigned)call.subdivNeighborDirMask);
        return nullptr;
    }

    void* srcIbPtr = nullptr;
    hr = key.ib->Lock(0, 0, &srcIbPtr, D3DLOCK_READONLY | D3DLOCK_DONOTWAIT);
    if (FAILED(hr) || !srcIbPtr) {
        LOG::logline("[DISPCACHE][BUILD-LOCK-FAIL] phase=IB vb=%p ib=%p hr=0x%08x ptr=%p tier=%u neigh=%02x",
                     key.vb, key.ib, (unsigned)hr, srcIbPtr,
                     (unsigned)call.subdivTier, (unsigned)call.subdivNeighborDirMask);
        key.vb->Unlock();
        return nullptr;
    }

    // Source is assumed row-major 5x5 per spec §stride-4. We don't re-derive the
    // grid from the IB — the plan settled this as an invariant.
    const uint8_t* srcV = (const uint8_t*)srcVbPtr + rs.baseIndex * rs.vbStride;

    // Phase 8.2: map each of the 4 grid-edge sides (row=0 / row=max / col=0 / col=max)
    // to a world-direction bit (+X=0, -X=1, +Y=2, -Y=3) so we can consult the
    // subdivNeighborDirMask the prepare thread stamped. Landscape patches are
    // axis-aligned in practice; we still detect sign from the source corner vertices
    // so an unexpected mirror/rotation does not silently flip edge decisions.
    auto getLocalPos = [&](int vi) {
        return *(const D3DXVECTOR3*)(srcV + vi * rs.vbStride + srcLayout.posOffset);
    };
    D3DXVECTOR3 vCol0Row0 = getLocalPos(0);
    D3DXVECTOR3 vColMaxRow0 = getLocalPos(kSrcGrid - 1);
    D3DXVECTOR3 vCol0RowMax = getLocalPos((kSrcGrid - 1) * kSrcGrid);
    D3DXVECTOR3 dCol = vColMaxRow0 - vCol0Row0;
    D3DXVECTOR3 dRow = vCol0RowMax - vCol0Row0;
    auto dirBitForDelta = [](const D3DXVECTOR3& d) -> int {
        if (fabsf(d.x) >= fabsf(d.y)) return d.x >= 0 ? 0 : 1;
        else                          return d.y >= 0 ? 2 : 3;
    };
    int colMaxBit = dirBitForDelta(dCol);
    int colMinBit = dirBitForDelta(-dCol);
    int rowMaxBit = dirBitForDelta(dRow);
    int rowMinBit = dirBitForDelta(-dRow);
    const uint8_t neighborMask = call.subdivNeighborDirMask;
    const bool colMaxIsSub = (neighborMask & (1u << colMaxBit)) != 0;
    const bool colMinIsSub = (neighborMask & (1u << colMinBit)) != 0;
    const bool rowMaxIsSub = (neighborMask & (1u << rowMaxBit)) != 0;
    const bool rowMinIsSub = (neighborMask & (1u << rowMinBit)) != 0;

    // Output layout: same fields plus one extra TEXCOORD.
    //   TEXCOORD1 (float2) — pre-baked base/overlay signed displacements;
    //   perimeter edge-locked verts carry 0 so shared edges keep original terrain height.
    VertexLayout dstLayout = srcLayout;
    UINT heightOff = srcLayout.stride;
    UINT dstStride = srcLayout.stride + 8;
    dstLayout.stride = dstStride;
    DWORD dstFvf = (rs.fvf & ~D3DFVF_TEXCOUNT_MASK) | D3DFVF_TEX2;
    // TEXCOORD0 and TEXCOORD1 both use the default size 2.

    // Resolve _paramh heights. Base always; overlay only if present.
    // Phase 8.1: base pointer is stamped on the call at prepare time, so the replay
    // thread never has to run TextureSuffix resolution here. Overlay still falls
    // through resolveParamH because the arg is an arbitrary texture pointer.
    const HeightMap* baseHM = nullptr;
    const HeightMap* overlayHM = nullptr;
    IDirect3DBaseTexture9* baseParamH = call.baseParamHTexture;
    if (!call.baseParamHTexture) {
        static int miss = 0;
        if (miss++ < 5) {
            LOG::logline("PatchDisplacement: no baseParamHTexture on terrain call rs.texture=%p",
                         rs.texture);
        }
    }
    IDirect3DBaseTexture9* overlayParamH = overlayTex ? resolveParamH(device, overlayTex) : nullptr;
    if (baseParamH) baseHM = getOrDecodeHeight(device, baseParamH);
    if (overlayParamH) overlayHM = getOrDecodeHeight(device, overlayParamH);

    // Build subdivided vertex buffer on the heap, then upload in one lock.
    std::vector<uint8_t> dstVerts((size_t)kDstVerts * dstStride, 0);

    // Pass 1: bilerp P/N/C/UV into every dst vert, and for interior verts
    // sample signed displacements locally from the paramhs. Edge verts get 0 placeholders
    // and are resolved in a post-pass against the coalesced edge map.
    for (UINT r = 0; r < kDstGrid; ++r) {
        for (UINT c = 0; c < kDstGrid; ++c) {
            UINT vi = r * kDstGrid + c;
            float srcPosV = r * kStep;
            float srcPosU = c * kStep;
            UINT srcR = std::min<UINT>((UINT)srcPosV, kSrcGrid - 1);
            UINT srcC = std::min<UINT>((UINT)srcPosU, kSrcGrid - 1);
            float fr = srcPosV - (float)srcR;
            float fc = srcPosU - (float)srcC;
            UINT srcR1 = std::min<UINT>(srcR + 1, kSrcGrid - 1);
            UINT srcC1 = std::min<UINT>(srcC + 1, kSrcGrid - 1);
            int i00 = (int)(srcR  * kSrcGrid + srcC);
            int i10 = (int)(srcR  * kSrcGrid + srcC1);
            int i01 = (int)(srcR1 * kSrcGrid + srcC);
            int i11 = (int)(srcR1 * kSrcGrid + srcC1);

            uint8_t* dstV = dstVerts.data() + (size_t)vi * dstStride;
            bilerpBytes(srcV, rs.vbStride, srcLayout,
                        i00, i10, i01, i11, fr, fc,
                        dstV, dstLayout, dstStride);

            bool onRow0   = (r == 0);
            bool onRowMax = (r == kDstGrid - 1);
            bool onCol0   = (c == 0);
            bool onColMax = (c == kDstGrid - 1);
            bool isEdge = onRow0 || onRowMax || onCol0 || onColMax;

            float* h = (float*)(dstV + heightOff);
            if (!isEdge) {
                const float* uv = (const float*)(dstV + dstLayout.uvOffset);
                h[0] = sampleHeight(baseHM,    uv[0], uv[1], heightScale, dispGamma, dispPivot);
                h[1] = overlayHM ? sampleHeight(overlayHM, uv[0], uv[1], heightScale, dispGamma, dispPivot) : 0.0f;
            } else {
                h[0] = 0.0f;
                h[1] = 0.0f;
            }
        }
    }

    // Pass 2: resolve edge vert heights against the coalesced map. For each of
    // the 4 edges, build a length-kDstGrid array of (base, overlay, matched).
    // "matched" means the map entry at this edge vert has contributions from
    // >= 2 distinct tiles (i.e., a neighbor also sampled here). Intermediate
    // verts with no neighbor contribution (count < 2) are linearly interpolated
    // between the two nearest matched indices on the same edge, so the denser
    // inner-tier edge lands exactly on the sparser outer-tier's linear segment.
    auto worldPosOf = [&](UINT vi) {
        const uint8_t* dstV = dstVerts.data() + (size_t)vi * dstStride;
        const float* p = (const float*)(dstV + dstLayout.posOffset);
        D3DXVECTOR3 local{p[0], p[1], p[2]};
        D3DXVECTOR3 world;
        D3DXVec3TransformCoord(&world, &local, &call.rs.worldTransforms[0]);
        return world;
    };
    auto localUvOf = [&](UINT vi) -> const float* {
        const uint8_t* dstV = dstVerts.data() + (size_t)vi * dstStride;
        return (const float*)(dstV + dstLayout.uvOffset);
    };

    struct EdgeArr {
        std::vector<float> base;
        std::vector<float> overlay;
        std::vector<uint8_t> matched;
        EdgeArr(UINT n) : base(n, 0.0f), overlay(n, 0.0f), matched(n, 0) {}
    };
    EdgeArr edgeRow0(kDstGrid), edgeRowMax(kDstGrid);
    EdgeArr edgeCol0(kDstGrid), edgeColMax(kDstGrid);

    auto fillEdge = [&](EdgeArr& arr, auto indexOf) {
        for (UINT k = 0; k < kDstGrid; ++k) {
            UINT vi = indexOf(k);
            D3DXVECTOR3 w = worldPosOf(vi);
            FixedFunctionShader::EdgeHeightKey key{
                (int32_t)lroundf(w.x), (int32_t)lroundf(w.y)
            };
            bool filled = false;
            if (edgeHeights) {
                auto it = edgeHeights->find(key);
                if (it != edgeHeights->end()) {
                    arr.base[k]    = it->second.avgBase;
                    arr.overlay[k] = it->second.avgOverlay;
                    arr.matched[k] = (it->second.count >= 2) ? 1 : 0;
                    filled = true;
                }
            }
            if (!filled) {
                // Map miss (shouldn't happen if this tile contributed). Fall
                // back to a local sample so the edge at least matches self.
                const float* uv = localUvOf(vi);
                arr.base[k]    = sampleHeight(baseHM, uv[0], uv[1], heightScale, dispGamma, dispPivot);
                arr.overlay[k] = overlayHM ? sampleHeight(overlayHM, uv[0], uv[1], heightScale, dispGamma, dispPivot) : 0.0f;
                arr.matched[k] = 0;
            }
        }
    };

    fillEdge(edgeRow0,   [kDstGrid](UINT k){ return (UINT)0 * kDstGrid + k; });
    fillEdge(edgeRowMax, [kDstGrid](UINT k){ return (UINT)(kDstGrid - 1) * kDstGrid + k; });
    fillEdge(edgeCol0,   [kDstGrid](UINT k){ return k * kDstGrid + 0; });
    fillEdge(edgeColMax, [kDstGrid](UINT k){ return k * kDstGrid + (kDstGrid - 1); });

    auto interpolateEdge = [kDstGrid](EdgeArr& arr) {
        int firstMatched = -1;
        for (UINT k = 0; k < kDstGrid; ++k) {
            if (arr.matched[k]) { firstMatched = (int)k; break; }
        }
        if (firstMatched < 0) return;  // no matched verts — whole edge is lone-tile
        int lastMatched = -1;
        for (UINT k = kDstGrid; k-- > 0; ) {
            if (arr.matched[k]) { lastMatched = (int)k; break; }
        }
        int prev = -1;
        for (UINT k = 0; k < kDstGrid; ++k) {
            if (arr.matched[k]) { prev = (int)k; continue; }
            if ((int)k < firstMatched || (int)k > lastMatched) continue;
            int next = -1;
            for (UINT j = k + 1; j < kDstGrid; ++j) {
                if (arr.matched[j]) { next = (int)j; break; }
            }
            if (prev >= 0 && next >= 0) {
                float t = (float)((int)k - prev) / (float)(next - prev);
                arr.base[k]    = arr.base[prev]    * (1.0f - t) + arr.base[next]    * t;
                arr.overlay[k] = arr.overlay[prev] * (1.0f - t) + arr.overlay[next] * t;
            }
        }
    };
    interpolateEdge(edgeRow0);
    interpolateEdge(edgeRowMax);
    interpolateEdge(edgeCol0);
    interpolateEdge(edgeColMax);

    // Pass 3: overwrite edge vert displacements. An edge vert on a non-subdivided
    // neighbor edge bakes 0 (keeps original terrain height crack-free).
    // An edge vert on a subdivided neighbor edge reads from the appropriate
    // edge array. Corner verts inherit h=0 if ANY adjacent edge is non-sub.
    auto writeH = [&](UINT vi, float b, float o) {
        uint8_t* dstV = dstVerts.data() + (size_t)vi * dstStride;
        float* h = (float*)(dstV + heightOff);
        h[0] = b;
        h[1] = o;
    };
    for (UINT r = 0; r < kDstGrid; ++r) {
        for (UINT c = 0; c < kDstGrid; ++c) {
            bool onRow0   = (r == 0);
            bool onRowMax = (r == kDstGrid - 1);
            bool onCol0   = (c == 0);
            bool onColMax = (c == kDstGrid - 1);
            if (!(onRow0 || onRowMax || onCol0 || onColMax)) continue;

            bool nonSub = (onRow0   && !rowMinIsSub)
                       || (onRowMax && !rowMaxIsSub)
                       || (onCol0   && !colMinIsSub)
                       || (onColMax && !colMaxIsSub);
            if (nonSub) {
                writeH(r * kDstGrid + c, 0.0f, 0.0f);
                continue;
            }
            // All edges through this vert face subdivided neighbors. Pick one
            // edge array; where multiple apply (corners) they refer to the same
            // world-position map entry, so the values agree.
            //
            // Phase 8.7c: at shared edges the VS lerp must be inert (base only)
            // so neighbors with different overlay signatures still displace to
            // the same displacement. One side might be "texture A pure base" and
            // the other "texture A with overlay B"; their base displacement at
            // the shared edge agrees (same paramh_A), their overlay does not. Writing
            // h[1] = h[0] makes lerp(base, overlay, alpha) = base regardless
            // of each side's alpha grid, so the edge matches across the seam.
            float b = 0.0f;
            if      (onRow0)   { b = edgeRow0.base[c];   }
            else if (onRowMax) { b = edgeRowMax.base[c]; }
            else if (onCol0)   { b = edgeCol0.base[r];   }
            else               { b = edgeColMax.base[r]; }
            writeH(r * kDstGrid + c, b, b);
        }
    }

    bool haveBakedDisplacement = false;
    float minBaseDisp = 0.0f, maxBaseDisp = 0.0f;
    float minOverlayDisp = 0.0f, maxOverlayDisp = 0.0f;
    for (UINT vi = 0; vi < kDstVerts; ++vi) {
        const uint8_t* dstV = dstVerts.data() + (size_t)vi * dstStride;
        const float* h = (const float*)(dstV + heightOff);
        if (!haveBakedDisplacement) {
            minBaseDisp = maxBaseDisp = h[0];
            minOverlayDisp = maxOverlayDisp = h[1];
            haveBakedDisplacement = true;
        } else {
            minBaseDisp = std::min(minBaseDisp, h[0]);
            maxBaseDisp = std::max(maxBaseDisp, h[0]);
            minOverlayDisp = std::min(minOverlayDisp, h[1]);
            maxOverlayDisp = std::max(maxOverlayDisp, h[1]);
        }
    }

    key.ib->Unlock();
    key.vb->Unlock();

    // Build the 16-bit IB for (kDstGrid-1)^2 quads. Row-major quads.
    // 33x33 dst = 32*32*2 = 2048 tris -> 12 KB, too big for the stack.
    std::vector<uint16_t> ibBuf(kDstIdx);
    UINT w = 0;
    for (UINT qr = 0; qr < kDstGrid - 1; ++qr) {
        for (UINT qc = 0; qc < kDstGrid - 1; ++qc) {
            uint16_t i00 = (uint16_t)(qr * kDstGrid + qc);
            uint16_t i10 = (uint16_t)(qr * kDstGrid + qc + 1);
            uint16_t i01 = (uint16_t)((qr + 1) * kDstGrid + qc);
            uint16_t i11 = (uint16_t)((qr + 1) * kDstGrid + qc + 1);
            ibBuf[w++] = i00; ibBuf[w++] = i10; ibBuf[w++] = i11;
            ibBuf[w++] = i00; ibBuf[w++] = i11; ibBuf[w++] = i01;
        }
    }

    auto patch = std::make_unique<SubdivPatch>();
    patch->stride = dstStride;
    patch->fvf = dstFvf;
    patch->vertCount = kDstVerts;
    patch->primCount = kDstTris;
    patch->overlayTexture = overlayTex;

    UINT vbBytes = (UINT)kDstVerts * dstStride;
    hr = device->CreateVertexBuffer(vbBytes, D3DUSAGE_WRITEONLY, 0, D3DPOOL_MANAGED, &patch->vb, nullptr);
    if (FAILED(hr) || !patch->vb) return nullptr;
    void* vbData = nullptr;
    hr = patch->vb->Lock(0, 0, &vbData, 0);
    if (FAILED(hr)) { return nullptr; }
    std::memcpy(vbData, dstVerts.data(), vbBytes);
    patch->vb->Unlock();

    UINT ibBytes = (UINT)(ibBuf.size() * sizeof(uint16_t));
    hr = device->CreateIndexBuffer(ibBytes, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_MANAGED, &patch->ib, nullptr);
    if (FAILED(hr) || !patch->ib) return nullptr;
    void* ibData = nullptr;
    hr = patch->ib->Lock(0, 0, &ibData, 0);
    if (FAILED(hr)) return nullptr;
    std::memcpy(ibData, ibBuf.data(), ibBytes);
    patch->ib->Unlock();

    // Insert into cache (steal ownership). If a prior entry exists (e.g. overlay
    // changed) its unique_ptr goes out of scope here and releases the old VB/IB.
    PatchCacheEntry entry;
    entry.patch = std::move(patch);
    entry.heightScale = heightScale;
    entry.overlayTexture = overlayTex;
    entry.baseParamH = baseParamH;
    entry.overlayParamH = overlayParamH;
    entry.subdivNeighborDirMask = call.subdivNeighborDirMask;
    entry.subdivTier = subdivTier;
    entry.edgeContextHash = call.edgeContextHash;
    entry.dispGamma = dispGamma;
    entry.dispPivot = dispPivot;
    SubdivPatch* ret = entry.patch.get();
    AcquireSRWLockExclusive(&s_lock);
    s_patchCache[key] = std::move(entry);
    ReleaseSRWLockExclusive(&s_lock);

    LOG::logline("PatchDisplacement: built subdivided patch vb=%p ib=%p overlay=%p "
                 "baseH=%s overlayH=%s scale=%.2f gamma=%.2f pivot=%.2f "
                 "baseRange=[%.2f,%.2f] overlayRange=[%.2f,%.2f] tier=%u grid=%ux%u",
                 key.vb, key.ib, overlayTex,
                 baseHM ? "y" : "n", overlayHM ? "y" : "n", heightScale, dispGamma, dispPivot,
                 minBaseDisp, maxBaseDisp, minOverlayDisp, maxOverlayDisp,
                 (unsigned)subdivTier, (unsigned)kDstGrid, (unsigned)kDstGrid);
    return ret;
}

void coalesceEdgeHeights(IDirect3DDevice9* device,
                         NearPatchEdgeHeights& outMap,
                         const std::vector<FixedFunctionShader::HLSLRecordedCall>& calls,
                         float heightScale,
                         float dispGamma,
                         float dispPivot)
{
    MGE_ZoneScopedN("PatchDisplacement_coalesceEdgeHeights");
    outMap.clear();
    if (!device) return;

    // Per-tile accumulator to dedup corner verts within a single tile before
    // merging into the shared map. Without dedup, a tile's own 4 corners would
    // each be counted twice (once per edge they sit on), inflating count and
    // breaking the "count >= 2 ⇒ another tile also sampled here" invariant.
    struct Accum { float base = 0.0f, overlay = 0.0f; };
    std::unordered_map<FixedFunctionShader::EdgeHeightKey, Accum,
                       FixedFunctionShader::EdgeHeightKeyHash> tileAccum;

    for (const auto& call : calls) {
        if (!call.sk.hasDisplacement) continue;
        if (call.bin != RenderBin::Terrain) continue;

        const auto& rs = call.rs;
        if (rs.vertCount != kSrcVerts || rs.primCount != kSrcTris) continue;
        if (rs.primType != D3DPT_TRIANGLELIST) continue;

        VertexLayout srcLayout;
        if (!srcLayout.parse(rs.fvf, rs.vbStride)) continue;
        if (srcLayout.posOffset < 0 || srcLayout.colorOffset < 0
            || srcLayout.uvOffset < 0 || srcLayout.normalOffset < 0)
        {
            continue;
        }

        const UINT kDstGrid = (call.subdivTier == 0) ? 65u : 33u;
        const float kStep   = float(kSrcGrid - 1) / float(kDstGrid - 1);

        const HeightMap* baseHM = nullptr;
        const HeightMap* overlayHM = nullptr;
        if (call.baseParamHTexture) baseHM = getOrDecodeHeight(device, call.baseParamHTexture);
        IDirect3DBaseTexture9* overlayParamH =
            call.overlayTexture ? resolveParamH(device, call.overlayTexture) : nullptr;
        if (overlayParamH) overlayHM = getOrDecodeHeight(device, overlayParamH);

        void* srcVbPtr = nullptr;
        HRESULT hr = rs.vb->Lock(rs.vbOffset, 0, &srcVbPtr,
                                 D3DLOCK_READONLY | D3DLOCK_DONOTWAIT);
        if (FAILED(hr) || !srcVbPtr) {
            // Silent skip means this tile contributes NO edge samples to the
            // coalesced map; adjacent near-set tiles will see count=1 at the
            // shared boundary and fall through to local-sample fallback,
            // producing crack-shaped Z mismatches at the seam.
            LOG::logline("[DISPCACHE][COALESCE-LOCK-FAIL] vb=%p ib=%p hr=0x%08x ptr=%p tier=%u neigh=%02x ctxHash=%08x",
                         rs.vb, rs.ib, (unsigned)hr, srcVbPtr,
                         (unsigned)call.subdivTier,
                         (unsigned)call.subdivNeighborDirMask,
                         (unsigned)call.edgeContextHash);
            continue;
        }
        const uint8_t* srcV = (const uint8_t*)srcVbPtr + rs.baseIndex * rs.vbStride;

        // Linearly interpolate the source edge between two 5x5 corner verts.
        // k in [0, kDstGrid-1] maps back to a source coord in [0, kSrcGrid-1].
        // We use this to compute world XY (for map key) and UV (for sampling).
        auto edgeVert = [&](int srcR_base, int srcC_base, int srcR_inc, int srcC_inc, UINT k,
                            D3DXVECTOR3& outWorld, float& outU, float& outV) {
            float t = k * kStep;
            int cell = std::min((int)t, (int)(kSrcGrid - 1));
            float frac = t - (float)cell;
            int srcR0 = srcR_base + cell * srcR_inc;
            int srcC0 = srcC_base + cell * srcC_inc;
            int srcR1 = std::min(srcR0 + srcR_inc, (int)(kSrcGrid - 1));
            int srcC1 = std::min(srcC0 + srcC_inc, (int)(kSrcGrid - 1));
            if (srcR_inc == 0) { srcR1 = srcR0; }
            if (srcC_inc == 0) { srcC1 = srcC0; }
            int i0 = srcR0 * kSrcGrid + srcC0;
            int i1 = srcR1 * kSrcGrid + srcC1;
            const float* p0 = (const float*)(srcV + i0 * rs.vbStride + srcLayout.posOffset);
            const float* p1 = (const float*)(srcV + i1 * rs.vbStride + srcLayout.posOffset);
            D3DXVECTOR3 local{
                p0[0] * (1.0f - frac) + p1[0] * frac,
                p0[1] * (1.0f - frac) + p1[1] * frac,
                p0[2] * (1.0f - frac) + p1[2] * frac
            };
            D3DXVec3TransformCoord(&outWorld, &local, &rs.worldTransforms[0]);
            const float* u0 = (const float*)(srcV + i0 * rs.vbStride + srcLayout.uvOffset);
            const float* u1 = (const float*)(srcV + i1 * rs.vbStride + srcLayout.uvOffset);
            outU = u0[0] * (1.0f - frac) + u1[0] * frac;
            outV = u0[1] * (1.0f - frac) + u1[1] * frac;
        };

        tileAccum.clear();
        auto addEdge = [&](int srcR_base, int srcC_base, int srcR_inc, int srcC_inc) {
            for (UINT k = 0; k < kDstGrid; ++k) {
                D3DXVECTOR3 w;
                float u, v;
                edgeVert(srcR_base, srcC_base, srcR_inc, srcC_inc, k, w, u, v);
                FixedFunctionShader::EdgeHeightKey key{
                    (int32_t)lroundf(w.x), (int32_t)lroundf(w.y)
                };
                float bh = sampleHeight(baseHM,    u, v, heightScale, dispGamma, dispPivot);
                float oh = overlayHM ? sampleHeight(overlayHM, u, v, heightScale, dispGamma, dispPivot) : 0.0f;
                // Last-write-wins within the same tile (corner verts reached by
                // two edges land on identical local pos / UV → identical h).
                tileAccum[key] = {bh, oh};
            }
        };
        addEdge(0,               0,               0, 1);  // row 0 (vary c)
        addEdge(kSrcGrid - 1,    0,               0, 1);  // row max
        addEdge(0,               0,               1, 0);  // col 0 (vary r)
        addEdge(0,               kSrcGrid - 1,    1, 0);  // col max

        rs.vb->Unlock();

        // Merge this tile's unique edge contributions into the shared map.
        // Phase 8.7b: first-write-wins. The first tile to touch a world-keyed
        // position stores its locally-sampled height; subsequent tiles only
        // bump count (for matched-detection) and do NOT overwrite. Both sides
        // of any shared edge therefore read the identical stored value — no
        // averaging artifact, no half-step between A and B's local samples.
        for (const auto& kv : tileAccum) {
            auto it = outMap.find(kv.first);
            if (it == outMap.end()) {
                auto& slot = outMap[kv.first];
                slot.avgBase    = kv.second.base;
                slot.avgOverlay = kv.second.overlay;
                slot.count      = 1;
            } else {
                it->second.count += 1;
            }
        }
    }
}

void prebuildNearPatches(IDirect3DDevice9* device,
                         FixedFunctionShader::FrameBuffer& fb,
                         float heightScale,
                         float dispGamma,
                         float dispPivot)
{
    MGE_ZoneScopedN("PatchDisplacement_prebuildNearPatches");
    if (!device) return;
    if (fb.nearPatchCount == 0) return;

    coalesceEdgeHeights(device, fb.nearPatchEdgeHeights, fb.recordedCalls,
                        heightScale, dispGamma, dispPivot);

    for (const auto& call : fb.recordedCalls) {
        if (!call.sk.hasDisplacement) continue;
        if (call.bin != RenderBin::Terrain) continue;

        FixedFunctionShader::TerrainPatchKey pk{call.rs.vb, call.rs.ib};
        bool inSet = false;
        for (uint32_t s = 0; s < fb.nearPatchCount; ++s) {
            if (fb.nearPatches[s] == pk) { inSet = true; break; }
        }
        if (!inSet) continue;

        SubdivPatch* sp = getOrBuild(device, pk, call, call.overlayTexture,
                                     heightScale, call.subdivTier,
                                     dispGamma, dispPivot,
                                     &fb.nearPatchEdgeHeights);
        if (!sp) {
            LOG::logline("[DISPCACHE][PREBUILD-FAIL] vb=%p ib=%p tier=%u neigh=%02x ctxHash=%08x scale=%.2f gamma=%.2f pivot=%.2f overlay=%p",
                         call.rs.vb, call.rs.ib, (unsigned)call.subdivTier,
                         (unsigned)call.subdivNeighborDirMask,
                         (unsigned)call.edgeContextHash,
                         heightScale, dispGamma, dispPivot, call.overlayTexture);
        }
    }
}

void clearAll() {
    AcquireSRWLockExclusive(&s_lock);
    s_patchCache.clear();
    // Heightmaps are content-addressed by texture pointer; only dropping them
    // when the underlying texture is released keeps us from decoding the same
    // BC3 twice when the player ping-pongs between cells. They are small.
    ReleaseSRWLockExclusive(&s_lock);
}

void onVertexBufferReleased(IDirect3DVertexBuffer9* vb) {
    if (!vb) return;
    AcquireSRWLockExclusive(&s_lock);
    for (auto it = s_patchCache.begin(); it != s_patchCache.end(); ) {
        if (it->first.vb == vb) it = s_patchCache.erase(it);
        else ++it;
    }
    ReleaseSRWLockExclusive(&s_lock);
}

void onIndexBufferReleased(IDirect3DIndexBuffer9* ib) {
    if (!ib) return;
    AcquireSRWLockExclusive(&s_lock);
    for (auto it = s_patchCache.begin(); it != s_patchCache.end(); ) {
        if (it->first.ib == ib) it = s_patchCache.erase(it);
        else ++it;
    }
    ReleaseSRWLockExclusive(&s_lock);
}

void onTextureReleased(IDirect3DBaseTexture9* tex) {
    if (!tex) return;
    AcquireSRWLockExclusive(&s_lock);
    s_heightCache.erase({tex});
    // Also rebuild any patches whose recorded _paramh matches, by dropping them.
    for (auto it = s_patchCache.begin(); it != s_patchCache.end(); ) {
        if (it->second.baseParamH == tex || it->second.overlayParamH == tex
            || it->second.overlayTexture == tex)
        {
            it = s_patchCache.erase(it);
        } else {
            ++it;
        }
    }
    ReleaseSRWLockExclusive(&s_lock);
}

} // namespace PatchDisplacement
