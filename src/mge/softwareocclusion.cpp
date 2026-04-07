#include "softwareocclusion.h"
#include <algorithm>
#include <cstring>
#include <d3dx9.h>
#include "support/log.h"
#include "mge_tracy.h"

SoftwareOcclusionCuller::SoftwareOcclusionCuller()
    : mWidth(0), mHeight(0), mNumMipLevels(0), mHiZVisualizationTexture(nullptr), mVisTexWidth(0), mVisTexHeight(0)
{
    for (int i = 0; i < MAX_MIP_LEVELS; i++) {
        mHiZBuffer[i] = nullptr;
        mHiZWidth[i] = 0;
        mHiZHeight[i] = 0;
    }
}

SoftwareOcclusionCuller::~SoftwareOcclusionCuller()
{
    shutdown();
}

void SoftwareOcclusionCuller::init(UINT width, UINT height)
{
    mWidth = width;
    mHeight = height;

    // Build Hi-Z pyramid: mip 0 = half-res, down to 8x8 minimum
    UINT mipW = width / 2;
    UINT mipH = height / 2;
    mNumMipLevels = 0;

    while (mipW >= 8 && mipH >= 8 && mNumMipLevels < MAX_MIP_LEVELS) {
        mHiZWidth[mNumMipLevels] = mipW;
        mHiZHeight[mNumMipLevels] = mipH;
        mHiZBuffer[mNumMipLevels] = new float[mipW * mipH];
        mNumMipLevels++;

        mipW /= 2;
        mipH /= 2;
    }

    clear();
}

void SoftwareOcclusionCuller::shutdown()
{
    for (int i = 0; i < mNumMipLevels; i++) {
        if (mHiZBuffer[i]) {
            delete[] mHiZBuffer[i];
            mHiZBuffer[i] = nullptr;
        }
    }
    mNumMipLevels = 0;

    mMeshCache.clear();
    mBlacklistedMeshes.clear();

    if (mHiZVisualizationTexture) {
        mHiZVisualizationTexture->Release();
        mHiZVisualizationTexture = nullptr;
    }
}

void SoftwareOcclusionCuller::clear()
{
    // Clear all mip levels to far plane
    for (int i = 0; i < mNumMipLevels; i++) {
        std::fill_n(mHiZBuffer[i], mHiZWidth[i] * mHiZHeight[i], 1.0f);
    }
}

void SoftwareOcclusionCuller::rasterizeOccluders(
    const std::vector<D3DXVECTOR3>& vertices,
    const std::vector<UINT>& indices,
    const D3DXMATRIX& view,
    const D3DXMATRIX& proj
)
{
    // UNUSED: We use rasterizeMesh() instead which works directly with D3D buffers
    mView = view;
    mProj = proj;
}

int SoftwareOcclusionCuller::rasterizeMesh(
    IDirect3DVertexBuffer9* vb,
    UINT vbOffset,
    UINT vbStride,
    IDirect3DIndexBuffer9* ib,
    UINT ibBase,
    UINT startIndex,
    UINT primCount,
    D3DPRIMITIVETYPE primType,
    DWORD fvf,
    const D3DXMATRIX& world,
    const D3DXMATRIX& view,
    const D3DXMATRIX& proj
)
{
    MGE_ZoneScoped;
    if (!vb || !ib) return 0;

    // Only support triangle lists for cached path (most common)
    if (primType != D3DPT_TRIANGLELIST) return 0;

    // Check if FVF has position data
    bool hasPosition = (fvf & D3DFVF_POSITION_MASK) != 0;
    if (!hasPosition) return 0;

    // Cache key for this mesh
    MeshCacheKey cacheKey = { vb, ib, vbOffset, vbStride, startIndex, primCount };

    // Check cache first - avoids expensive VB/IB lock on subsequent frames
    CachedMeshGeometry* cached = nullptr;
    auto it = mMeshCache.find(cacheKey);
    if (it != mMeshCache.end()) {
        cached = &it->second;
    } else {
        // Cache miss - need to lock buffers and read geometry
        MGE_ZoneScopedN("MeshCache Miss");

        void* pVertices = nullptr;
        HRESULT hr = vb->Lock(vbOffset, 0, &pVertices, D3DLOCK_READONLY);
        if (FAILED(hr)) {
            static int vbLockFailCount = 0;
            if (vbLockFailCount++ < 5) {
                LOG::logline("!! SoftwareOcclusion: VB lock failed (hr=0x%X)", hr);
            }
            return 0;
        }

        void* pIndices = nullptr;
        hr = ib->Lock(0, 0, &pIndices, D3DLOCK_READONLY);
        if (FAILED(hr)) {
            static int ibLockFailCount = 0;
            if (ibLockFailCount++ < 5) {
                LOG::logline("!! SoftwareOcclusion: IB lock failed (hr=0x%X)", hr);
            }
            vb->Unlock();
            return 0;
        }

        // Determine index format
        D3DINDEXBUFFER_DESC ibDesc;
        ib->GetDesc(&ibDesc);
        bool is16Bit = (ibDesc.Format == D3DFMT_INDEX16);

        // Read indices and find unique vertex indices
        UINT indexCount = primCount * 3;
        CachedMeshGeometry newCache;
        newCache.is16BitIndices = is16Bit;
        newCache.indices.reserve(indexCount);

        // Find min/max vertex indices to know range
        UINT minVertIdx = UINT_MAX, maxVertIdx = 0;
        for (UINT i = 0; i < indexCount; i++) {
            UINT vertIdx;
            if (is16Bit) {
                vertIdx = ((WORD*)pIndices)[startIndex + i] + ibBase;
            } else {
                vertIdx = ((DWORD*)pIndices)[startIndex + i] + ibBase;
            }
            newCache.indices.push_back(vertIdx);
            minVertIdx = std::min(minVertIdx, vertIdx);
            maxVertIdx = std::max(maxVertIdx, vertIdx);
        }

        // Read vertex positions for the range we need
        UINT vertexRange = maxVertIdx - minVertIdx + 1;
        newCache.positions.resize(vertexRange);
        for (UINT i = 0; i < vertexRange; i++) {
            UINT vertIdx = minVertIdx + i;
            D3DXVECTOR3* pos = (D3DXVECTOR3*)((BYTE*)pVertices + vertIdx * vbStride);
            newCache.positions[i] = *pos;
        }

        // Adjust indices to be relative to minVertIdx
        for (auto& idx : newCache.indices) {
            idx -= minVertIdx;
        }

        ib->Unlock();
        vb->Unlock();

        // Store in cache
        mMeshCache[cacheKey] = std::move(newCache);
        cached = &mMeshCache[cacheKey];
    }

    // Track pixels written
    int pixelsWritten = 0;

    // Compute transforms
    D3DXMATRIX worldView = world * view;
    D3DXMATRIX worldViewProj = worldView * proj;

    const float NEAR_DIST = 0.01f;
    const float MAX_DIST = 10000.0f;

    UINT mip0Width = mHiZWidth[0];
    UINT mip0Height = mHiZHeight[0];
    float* mip0Buffer = mHiZBuffer[0];

    // Rasterize triangles from cached data
    const auto& positions = cached->positions;
    const auto& indices = cached->indices;
    UINT triCount = (UINT)indices.size() / 3;

    for (UINT t = 0; t < triCount; t++) {
        UINT idx0 = indices[t * 3 + 0];
        UINT idx1 = indices[t * 3 + 1];
        UINT idx2 = indices[t * 3 + 2];

        const D3DXVECTOR3& v0 = positions[idx0];
        const D3DXVECTOR3& v1 = positions[idx1];
        const D3DXVECTOR3& v2 = positions[idx2];

        // Transform to view space for distance check
        D3DXVECTOR4 vs0, vs1, vs2;
        D3DXVec3Transform(&vs0, &v0, &worldView);
        D3DXVec3Transform(&vs1, &v1, &worldView);
        D3DXVec3Transform(&vs2, &v2, &worldView);

        if (vs0.z < NEAR_DIST || vs1.z < NEAR_DIST || vs2.z < NEAR_DIST) continue;
        if (vs0.z > MAX_DIST  || vs1.z > MAX_DIST  || vs2.z > MAX_DIST)  continue;

        // Project to clip space
        D3DXVECTOR4 c0, c1, c2;
        D3DXVec3Transform(&c0, &v0, &worldViewProj);
        D3DXVec3Transform(&c1, &v1, &worldViewProj);
        D3DXVec3Transform(&c2, &v2, &worldViewProj);

        // Perspective divide
        D3DXVECTOR4 p0 = c0 / c0.w;
        D3DXVECTOR4 p1 = c1 / c1.w;
        D3DXVECTOR4 p2 = c2 / c2.w;

        // Convert to fixed-point screen coordinates
        int fx0 = (int)((p0.x * 0.5f + 0.5f) * mip0Width * 16.0f);
        int fy0 = (int)((1.0f - (p0.y * 0.5f + 0.5f)) * mip0Height * 16.0f);
        float z0 = p0.z;

        int fx1 = (int)((p1.x * 0.5f + 0.5f) * mip0Width * 16.0f);
        int fy1 = (int)((1.0f - (p1.y * 0.5f + 0.5f)) * mip0Height * 16.0f);
        float z1 = p1.z;

        int fx2 = (int)((p2.x * 0.5f + 0.5f) * mip0Width * 16.0f);
        int fy2 = (int)((1.0f - (p2.y * 0.5f + 0.5f)) * mip0Height * 16.0f);
        float z2 = p2.z;

        float conservativeDepth = std::max({z0, z1, z2});

        // Edge function setup
        int A0 = fy1 - fy2;
        int B0 = fx2 - fx1;
        int C0 = fx1 * fy2 - fx2 * fy1;

        int A1 = fy2 - fy0;
        int B1 = fx0 - fx2;
        int C1 = fx2 * fy0 - fx0 * fy2;

        int A2 = fy0 - fy1;
        int B2 = fx1 - fx0;
        int C2 = fx0 * fy1 - fx1 * fy0;

        int triArea = B2 * A1 - B1 * A2;
        if (triArea == 0) continue;

        // Bounding box
        int minX = std::max(0, std::min({fx0, fx1, fx2}) >> 4);
        int maxX = std::min((int)mip0Width - 1, std::max({fx0, fx1, fx2}) >> 4);
        int minY = std::max(0, std::min({fy0, fy1, fy2}) >> 4);
        int maxY = std::min((int)mip0Height - 1, std::max({fy0, fy1, fy2}) >> 4);

        // Rasterize
        for (int y = minY; y <= maxY; y++) {
            int fy = (y << 4) + 8;
            for (int x = minX; x <= maxX; x++) {
                int fx = (x << 4) + 8;

                int e0 = A0 * fx + B0 * fy + C0;
                int e1 = A1 * fx + B1 * fy + C1;
                int e2 = A2 * fx + B2 * fy + C2;

                bool frontFace = (e0 >= 0 && e1 >= 0 && e2 >= 0);
                bool backFace = (e0 <= 0 && e1 <= 0 && e2 <= 0);
                if (frontFace || backFace) {
                    int idx = y * mip0Width + x;
                    if (conservativeDepth < mip0Buffer[idx]) {
                        mip0Buffer[idx] = conservativeDepth;
                        pixelsWritten++;
                    }
                }
            }
        }
    }

    return pixelsWritten;
}

void SoftwareOcclusionCuller::buildHiZPyramid()
{
    MGE_ZoneScopedN("Build Hi-Z Pyramid");

    // Build mipmap chain from mip 0 down to 8x8
    // Each mip stores MAX depth of 2x2 region from previous mip
    for (int mip = 1; mip < mNumMipLevels; mip++) {
        float* srcBuffer = mHiZBuffer[mip - 1];
        float* dstBuffer = mHiZBuffer[mip];
        UINT srcW = mHiZWidth[mip - 1];
        UINT srcH = mHiZHeight[mip - 1];
        UINT dstW = mHiZWidth[mip];
        UINT dstH = mHiZHeight[mip];

        for (UINT y = 0; y < dstH; y++) {
            for (UINT x = 0; x < dstW; x++) {
                // Sample 2x2 region from previous mip
                UINT sx = x * 2;
                UINT sy = y * 2;

                float d00 = srcBuffer[sy * srcW + sx];
                float d10 = (sx + 1 < srcW) ? srcBuffer[sy * srcW + sx + 1] : 1.0f;
                float d01 = (sy + 1 < srcH) ? srcBuffer[(sy + 1) * srcW + sx] : 1.0f;
                float d11 = (sx + 1 < srcW && sy + 1 < srcH) ? srcBuffer[(sy + 1) * srcW + sx + 1] : 1.0f;

                // Store MAX depth (furthest) - conservative for occlusion testing
                float maxDepth = std::max(std::max(d00, d10), std::max(d01, d11));
                dstBuffer[y * dstW + x] = maxDepth;
            }
        }
    }
}

void SoftwareOcclusionCuller::saveToDisk(IDirect3DDevice9* device, const char* directory)
{
    if (!device) {
        LOG::logline("!! SoftwareOcclusionCuller::saveToDisk: device is null");
        return;
    }

    LOG::logline(">> Saving CPU Hi-Z buffers to %s...", directory);

    for (int mip = 0; mip < mNumMipLevels; mip++) {
        UINT width = mHiZWidth[mip];
        UINT height = mHiZHeight[mip];
        float* cpuData = mHiZBuffer[mip];

        if (!cpuData) continue;

        // Create texture and fill with CPU data
        IDirect3DTexture9* tex = nullptr;
        HRESULT hr = device->CreateTexture(width, height, 1, 0, D3DFMT_R32F, D3DPOOL_MANAGED, &tex, NULL);
        if (FAILED(hr)) {
            LOG::logline("   !! Failed to create texture for mip %d (hr=0x%x)", mip, hr);
            continue;
        }

        // Lock and copy CPU data to texture
        D3DLOCKED_RECT lr;
        hr = tex->LockRect(0, &lr, NULL, 0);
        if (FAILED(hr)) {
            LOG::logline("   !! Failed to lock texture for mip %d (hr=0x%x)", mip, hr);
            tex->Release();
            continue;
        }

        float* texData = (float*)lr.pBits;
        int texStride = lr.Pitch / sizeof(float);
        for (UINT y = 0; y < height; y++) {
            for (UINT x = 0; x < width; x++) {
                texData[y * texStride + x] = cpuData[y * width + x];
            }
        }
        tex->UnlockRect(0);

        // Save DDS
        char ddsFilename[512];
        snprintf(ddsFilename, sizeof(ddsFilename), "%s/cpu_hiz_mip%d.dds", directory, mip);
        hr = D3DXSaveTextureToFile(ddsFilename, D3DXIFF_DDS, tex, NULL);
        if (FAILED(hr)) {
            LOG::logline("   !! Failed to save DDS for mip %d (hr=0x%x)", mip, hr);
        } else {
            LOG::logline("   -> Saved %s (%dx%d)", ddsFilename, width, height);
        }

        // Analyze depth values
        float minDepth = 1e10f, maxDepth = -1e10f;
        int zeroCount = 0, oneCount = 0;
        for (UINT i = 0; i < width * height; i++) {
            float d = cpuData[i];
            if (d < minDepth) minDepth = d;
            if (d > maxDepth) maxDepth = d;
            if (d == 0.0f) zeroCount++;
            if (d == 1.0f) oneCount++;
        }

        LOG::logline("      Depth range: [%.6f, %.6f], zeros=%d, ones=%d",
                     minDepth, maxDepth, zeroCount, oneCount);

        // Create normalized PNG for visualization
        if (maxDepth > 0.0f && maxDepth != 1.0f) {
            IDirect3DSurface9* pngSurf = nullptr;
            hr = device->CreateOffscreenPlainSurface(width, height, D3DFMT_A8R8G8B8, D3DPOOL_SCRATCH, &pngSurf, NULL);
            if (SUCCEEDED(hr)) {
                D3DLOCKED_RECT pngRect;
                hr = pngSurf->LockRect(&pngRect, NULL, 0);
                if (SUCCEEDED(hr)) {
                    DWORD* pngData = (DWORD*)pngRect.pBits;
                    int pngStride = pngRect.Pitch / sizeof(DWORD);

                    // Normalize depth to 0-255 grayscale
                    for (UINT y = 0; y < height; y++) {
                        for (UINT x = 0; x < width; x++) {
                            float depth = cpuData[y * width + x];
                            float normalized = depth / maxDepth;
                            BYTE gray = (BYTE)(normalized * 255.0f);
                            pngData[y * pngStride + x] = 0xFF000000 | (gray << 16) | (gray << 8) | gray;
                        }
                    }

                    pngSurf->UnlockRect();

                    char pngFilename[512];
                    snprintf(pngFilename, sizeof(pngFilename), "%s/cpu_hiz_mip%d.png", directory, mip);
                    hr = D3DXSaveSurfaceToFile(pngFilename, D3DXIFF_PNG, pngSurf, NULL, NULL);
                    if (SUCCEEDED(hr)) {
                        LOG::logline("      -> Saved %s (normalized)", pngFilename);
                    }
                }
                pngSurf->Release();
            }
        }

        tex->Release();
    }

    LOG::logline(">> CPU Hi-Z save complete");
}

bool SoftwareOcclusionCuller::isMeshBlacklisted(IDirect3DVertexBuffer9* vb, IDirect3DIndexBuffer9* ib) const
{
    MeshID id = { vb, ib };
    return mBlacklistedMeshes.find(id) != mBlacklistedMeshes.end();
}

void SoftwareOcclusionCuller::blacklistMesh(IDirect3DVertexBuffer9* vb, IDirect3DIndexBuffer9* ib)
{
    MeshID id = { vb, ib };
    mBlacklistedMeshes.insert(id);
}

void SoftwareOcclusionCuller::clearBlacklist()
{
    mBlacklistedMeshes.clear();
}

void SoftwareOcclusionCuller::clearMeshCache()
{
    mMeshCache.clear();
}

bool SoftwareOcclusionCuller::testBoundingBox(
    const D3DXVECTOR3& bboxMin,
    const D3DXVECTOR3& bboxMax,
    const D3DXMATRIX& view,
    const D3DXMATRIX& proj
)
{
    MGE_ZoneScoped;
    D3DXMATRIX viewProj = view * proj;

    // Test all 8 corners of bounding box
    D3DXVECTOR3 corners[8] = {
        D3DXVECTOR3(bboxMin.x, bboxMin.y, bboxMin.z),
        D3DXVECTOR3(bboxMax.x, bboxMin.y, bboxMin.z),
        D3DXVECTOR3(bboxMin.x, bboxMax.y, bboxMin.z),
        D3DXVECTOR3(bboxMax.x, bboxMax.y, bboxMin.z),
        D3DXVECTOR3(bboxMin.x, bboxMin.y, bboxMax.z),
        D3DXVECTOR3(bboxMax.x, bboxMin.y, bboxMax.z),
        D3DXVECTOR3(bboxMin.x, bboxMax.y, bboxMax.z),
        D3DXVECTOR3(bboxMax.x, bboxMax.y, bboxMax.z)
    };

    // Find screen-space AABB and closest depth
    float screenMinX = FLT_MAX, screenMaxX = -FLT_MAX;
    float screenMinY = FLT_MAX, screenMaxY = -FLT_MAX;
    float closestZ = FLT_MAX;
    bool anyInFront = false;
    bool anyBehindCamera = false;

    for (int i = 0; i < 8; i++) {
        D3DXVECTOR4 clipPos;
        D3DXVec3Transform(&clipPos, &corners[i], &viewProj);

        if (clipPos.w <= 0.0f) {
            anyBehindCamera = true;
            continue; // Behind camera
        }

        anyInFront = true;
        clipPos /= clipPos.w;

        // NDC to screen (use mip 0 = half-res coordinates)
        UINT mip0W = mHiZWidth[0];
        UINT mip0H = mHiZHeight[0];
        float sx = (clipPos.x * 0.5f + 0.5f) * mip0W;
        float sy = (1.0f - (clipPos.y * 0.5f + 0.5f)) * mip0H;

        screenMinX = std::min(screenMinX, sx);
        screenMaxX = std::max(screenMaxX, sx);
        screenMinY = std::min(screenMinY, sy);
        screenMaxY = std::max(screenMaxY, sy);
        closestZ = std::min(closestZ, clipPos.z);
    }

    if (!anyInFront) return true; // Conservative - assume visible

    // If any corner is behind camera OR off-screen, this is likely a large nearby occluder (wall/floor/ceiling)
    // DO NOT cull these - they're important for occlusion and shouldn't be tested
    bool anyOffScreen = anyBehindCamera ||
                        (screenMinX < 0.0f || screenMaxX > (float)mHiZWidth[0] ||
                         screenMinY < 0.0f || screenMaxY > (float)mHiZHeight[0]);
    if (anyOffScreen) return true; // Visible - don't cull large nearby geometry

    // Clamp to mip 0 bounds
    int minX = std::max(0, (int)screenMinX);
    int maxX = std::min((int)mHiZWidth[0] - 1, (int)screenMaxX);
    int minY = std::max(0, (int)screenMinY);
    int maxY = std::min((int)mHiZHeight[0] - 1, (int)screenMaxY);

    // Simple single-mip test: sample mip 0 sparsely
    // TODO: Use hierarchical testing for better performance
    float* mip0Buffer = mHiZBuffer[0];
    for (int y = minY; y <= maxY; y += 2) {
        for (int x = minX; x <= maxX; x += 2) {
            int idx = y * mHiZWidth[0] + x;
            if (mip0Buffer[idx] >= closestZ) {
                return true; // Visible - depth buffer has something behind bbox
            }
        }
    }

    return false; // Occluded - all depth samples are in front of bbox
}

void SoftwareOcclusionCuller::uploadHiZToTexture(IDirect3DDevice9* device, int mipLevel, const D3DXMATRIX& proj, bool invert)
{
    if (mNumMipLevels == 0 || !mHiZBuffer[0]) return;

    // Clamp mip level
    if (mipLevel < 0 || mipLevel >= mNumMipLevels) mipLevel = 0;

    UINT width = mHiZWidth[mipLevel];
    UINT height = mHiZHeight[mipLevel];

    // Recreate texture when dimensions change (switching mip levels)
    if (!mHiZVisualizationTexture || width != mVisTexWidth || height != mVisTexHeight) {
        if (mHiZVisualizationTexture) mHiZVisualizationTexture->Release();
        device->CreateTexture(width, height, 1, 0,
                             D3DFMT_A8R8G8B8, D3DPOOL_MANAGED,
                             &mHiZVisualizationTexture, nullptr);
        mVisTexWidth = width;
        mVisTexHeight = height;
    }

    // Extract near/far from projection matrix for depth linearization
    // D3D projection: proj._33 = far/(far-near), proj._43 = -near*far/(far-near)
    // near = -proj._43 / proj._33, far = -proj._43 / (proj._33 - 1)
    float nearPlane = -proj._43 / proj._33;
    float farPlane = -proj._43 / (proj._33 - 1.0f);

    // Lock and fill texture with depth visualization
    D3DLOCKED_RECT rect;
    if (FAILED(mHiZVisualizationTexture->LockRect(0, &rect, nullptr, 0))) {
        return;
    }

    DWORD* pixels = (DWORD*)rect.pBits;
    float* buffer = mHiZBuffer[mipLevel];

    for (UINT y = 0; y < height; y++) {
        for (UINT x = 0; x < width; x++) {
            float zNdc = buffer[y * width + x];

            // Linearize: convert NDC depth to view-space distance, normalize by far plane
            // z_view = near * far / (far - zNdc * (far - near))
            // linear01 = z_view / far = near / (far - zNdc * (far - near))
            float linear01;
            if (zNdc >= 1.0f) {
                linear01 = 1.0f; // At or beyond far plane (empty pixel)
            } else {
                float zView = nearPlane * farPlane / (farPlane - zNdc * (farPlane - nearPlane));
                linear01 = zView / farPlane;
                linear01 = std::min(1.0f, std::max(0.0f, linear01));
            }

            if (invert) {
                linear01 = 1.0f - linear01;
            }

            BYTE gray = (BYTE)(linear01 * 255.0f);
            pixels[y * (rect.Pitch / 4) + x] = D3DCOLOR_ARGB(255, gray, gray, gray);
        }
    }

    mHiZVisualizationTexture->UnlockRect(0);
}
