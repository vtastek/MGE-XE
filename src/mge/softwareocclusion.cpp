#include "softwareocclusion.h"
#include <algorithm>
#include <cstring>
#include <d3dx9.h>
#include "support/log.h"
#include "tracy/Tracy.hpp"

SoftwareOcclusionCuller::SoftwareOcclusionCuller()
    : mWidth(0), mHeight(0), mNumMipLevels(0), mHiZVisualizationTexture(nullptr)
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
    ZoneScoped;
    if (!vb || !ib) return 0;

    // Check if FVF has position data
    bool hasPosition = (fvf & D3DFVF_POSITION_MASK) != 0;
    if (!hasPosition) return 0;

    // Track pixels written to measure raster efficiency
    int pixelsWritten = 0;

    // Lock vertex buffer (use D3DLOCK_READONLY only - DONOTWAIT causes failures during GPU use)
    void* pVertices = nullptr;
    HRESULT hr = vb->Lock(vbOffset, 0, &pVertices, D3DLOCK_READONLY);
    if (FAILED(hr)) {
        static int vbLockFailCount = 0;
        if (vbLockFailCount++ < 5) {
            LOG::logline("!! SoftwareOcclusion: VB lock failed (hr=0x%X)", hr);
        }
        return 0;
    }

    // Lock index buffer
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

    // Compute transforms
    D3DXMATRIX worldView = world * view;           // For view-space distance check
    D3DXMATRIX worldViewProj = worldView * proj;   // For final projection

    // Simple distance limits - avoid near-plane clipping complexity
    const float NEAR_DIST = 0.01f;     // Minimum distance from camera (very small - just reject behind-camera)
    const float MAX_DIST = 10000.0f;   // Maximum distance (very large - essentially no limit for now)

    // Process triangles based on primitive type
    UINT indexCount = 0;
    switch (primType) {
        case D3DPT_TRIANGLELIST: indexCount = primCount * 3; break;
        case D3DPT_TRIANGLESTRIP: indexCount = primCount + 2; break;
        case D3DPT_TRIANGLEFAN: indexCount = primCount + 2; break;
        default:
            ib->Unlock();
            vb->Unlock();
            return 0;
    }

    // Rasterize triangles
    for (UINT i = 0; i < indexCount; i += 3) {
        // Get vertex indices
        UINT idx0, idx1, idx2;
        if (primType == D3DPT_TRIANGLELIST) {
            if (is16Bit) {
                idx0 = ((WORD*)pIndices)[startIndex + i + 0] + ibBase;
                idx1 = ((WORD*)pIndices)[startIndex + i + 1] + ibBase;
                idx2 = ((WORD*)pIndices)[startIndex + i + 2] + ibBase;
            } else {
                idx0 = ((DWORD*)pIndices)[startIndex + i + 0] + ibBase;
                idx1 = ((DWORD*)pIndices)[startIndex + i + 1] + ibBase;
                idx2 = ((DWORD*)pIndices)[startIndex + i + 2] + ibBase;
            }
        } else if (primType == D3DPT_TRIANGLESTRIP) {
            if (is16Bit) {
                idx0 = ((WORD*)pIndices)[startIndex + i + 0] + ibBase;
                idx1 = ((WORD*)pIndices)[startIndex + i + 1] + ibBase;
                idx2 = ((WORD*)pIndices)[startIndex + i + 2] + ibBase;
            } else {
                idx0 = ((DWORD*)pIndices)[startIndex + i + 0] + ibBase;
                idx1 = ((DWORD*)pIndices)[startIndex + i + 1] + ibBase;
                idx2 = ((DWORD*)pIndices)[startIndex + i + 2] + ibBase;
            }
            // Triangle strip winding order alternates
            if (i & 1) std::swap(idx1, idx2);
        } else { // D3DPT_TRIANGLEFAN
            if (is16Bit) {
                idx0 = ((WORD*)pIndices)[startIndex] + ibBase;
                idx1 = ((WORD*)pIndices)[startIndex + i + 1] + ibBase;
                idx2 = ((WORD*)pIndices)[startIndex + i + 2] + ibBase;
            } else {
                idx0 = ((DWORD*)pIndices)[startIndex] + ibBase;
                idx1 = ((DWORD*)pIndices)[startIndex + i + 1] + ibBase;
                idx2 = ((DWORD*)pIndices)[startIndex + i + 2] + ibBase;
            }
        }

        // Get vertex positions (position is always first in FVF)
        D3DXVECTOR3 v0 = *(D3DXVECTOR3*)((BYTE*)pVertices + idx0 * vbStride);
        D3DXVECTOR3 v1 = *(D3DXVECTOR3*)((BYTE*)pVertices + idx1 * vbStride);
        D3DXVECTOR3 v2 = *(D3DXVECTOR3*)((BYTE*)pVertices + idx2 * vbStride);

        // Transform to VIEW SPACE first (for distance check)
        D3DXVECTOR4 vs0, vs1, vs2;
        D3DXVec3Transform(&vs0, &v0, &worldView);
        D3DXVec3Transform(&vs1, &v1, &worldView);
        D3DXVec3Transform(&vs2, &v2, &worldView);

        // Simple rejection: ALL vertices must be within [NEAR_DIST, MAX_DIST]
        // View space Z is positive looking into screen (D3D convention)
        // Skip triangles that cross near plane - avoids all clipping complexity
        if (vs0.z < NEAR_DIST || vs1.z < NEAR_DIST || vs2.z < NEAR_DIST) continue;
        if (vs0.z > MAX_DIST  || vs1.z > MAX_DIST  || vs2.z > MAX_DIST)  continue;

        // Now safe to project - no clipping needed, all vertices are in front of camera
        D3DXVECTOR4 c0, c1, c2;
        D3DXVec3Transform(&c0, &v0, &worldViewProj);
        D3DXVec3Transform(&c1, &v1, &worldViewProj);
        D3DXVec3Transform(&c2, &v2, &worldViewProj);

        // Perspective divide (all w values guaranteed positive now)
        D3DXVECTOR4 p0 = c0 / c0.w;
        D3DXVECTOR4 p1 = c1 / c1.w;
        D3DXVECTOR4 p2 = c2 / c2.w;

        // Convert to fixed-point screen coordinates (rasterize to mip 0 = half-res)
        UINT mip0Width = mHiZWidth[0];
        UINT mip0Height = mHiZHeight[0];

        int fx0 = (int)((p0.x * 0.5f + 0.5f) * mip0Width * 16.0f);
        int fy0 = (int)((1.0f - (p0.y * 0.5f + 0.5f)) * mip0Height * 16.0f);
        float z0 = p0.z;

        int fx1 = (int)((p1.x * 0.5f + 0.5f) * mip0Width * 16.0f);
        int fy1 = (int)((1.0f - (p1.y * 0.5f + 0.5f)) * mip0Height * 16.0f);
        float z1 = p1.z;

        int fx2 = (int)((p2.x * 0.5f + 0.5f) * mip0Width * 16.0f);
        int fy2 = (int)((1.0f - (p2.y * 0.5f + 0.5f)) * mip0Height * 16.0f);
        float z2 = p2.z;

        // CONSERVATIVE DEPTH: Use max (furthest) depth of triangle vertices
        // This prevents false occlusions from depth interpolation errors with off-screen vertices
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

            // Triangle area (use absolute value - no backface culling for occluders!)
            int triArea = B2 * A1 - B1 * A2;
            if (triArea == 0) continue; // Skip degenerate triangles only
            triArea = abs(triArea); // Accept both front and back faces

            float oneOverTriArea = 1.0f / (float)triArea;

            // Bounding box (clamp to mip 0 resolution)
            int minX = std::max(0, std::min({fx0, fx1, fx2}) >> 4);
            int maxX = std::min((int)mip0Width - 1, std::max({fx0, fx1, fx2}) >> 4);
            int minY = std::max(0, std::min({fy0, fy1, fy2}) >> 4);
            int maxY = std::min((int)mip0Height - 1, std::max({fy0, fy1, fy2}) >> 4);

            // Rasterize to mip 0 (half-res Hi-Z buffer)
            float* mip0Buffer = mHiZBuffer[0];
            for (int y = minY; y <= maxY; y++) {
                int fy = (y << 4) + 8;
                for (int x = minX; x <= maxX; x++) {
                    int fx = (x << 4) + 8;

                    int e0 = A0 * fx + B0 * fy + C0;
                    int e1 = A1 * fx + B1 * fy + C1;
                    int e2 = A2 * fx + B2 * fy + C2;

                    // Accept BOTH windings - front faces have all positive, back faces have all negative
                    bool frontFace = (e0 >= 0 && e1 >= 0 && e2 >= 0);
                    bool backFace = (e0 <= 0 && e1 <= 0 && e2 <= 0);
                    if (frontFace || backFace) {
                        // Use CONSERVATIVE depth (max of vertices) to avoid false occlusions
                        // Interpolated depth can be wrong when vertices are off-screen
                        int idx = y * mip0Width + x;
                        if (conservativeDepth < mip0Buffer[idx]) {
                            mip0Buffer[idx] = conservativeDepth;
                            pixelsWritten++;
                        }
                    }
                }
            }
    }  // End of triangle loop

    ib->Unlock();
    vb->Unlock();
    return pixelsWritten;
}

void SoftwareOcclusionCuller::buildHiZPyramid()
{
    ZoneScopedN("Build Hi-Z Pyramid");

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

bool SoftwareOcclusionCuller::testBoundingBox(
    const D3DXVECTOR3& bboxMin,
    const D3DXVECTOR3& bboxMax,
    const D3DXMATRIX& view,
    const D3DXMATRIX& proj
)
{
    ZoneScoped;
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

void SoftwareOcclusionCuller::uploadHiZToTexture(IDirect3DDevice9* device, int mipLevel, float brightness, float gamma, bool invert, bool showRaycastGrid, int raycastStep)
{
    if (mNumMipLevels == 0 || !mHiZBuffer[0]) return;

    // Clamp mip level
    if (mipLevel < 0 || mipLevel >= mNumMipLevels) mipLevel = 0;

    UINT width = mHiZWidth[mipLevel];
    UINT height = mHiZHeight[mipLevel];

    // Create/recreate texture if needed (always use mip 0 size for consistency)
    if (!mHiZVisualizationTexture || mipLevel == 0) {
        if (mHiZVisualizationTexture) mHiZVisualizationTexture->Release();
        device->CreateTexture(width, height, 1, 0,
                             D3DFMT_A8R8G8B8, D3DPOOL_MANAGED,
                             &mHiZVisualizationTexture, nullptr);
    }

    // Lock and fill texture with depth visualization
    D3DLOCKED_RECT rect;
    if (FAILED(mHiZVisualizationTexture->LockRect(0, &rect, nullptr, 0))) {
        return;
    }

    DWORD* pixels = (DWORD*)rect.pBits;
    float* buffer = mHiZBuffer[mipLevel];

    for (UINT y = 0; y < height; y++) {
        for (UINT x = 0; x < width; x++) {
            float depth = buffer[y * width + x];

            // Remap depth values from compressed range to full [0,1] for better visibility
            // Most depth values cluster near 1.0 (0.9-1.0 range), so use smoothstep for better contrast
            float remappedDepth = depth;
            if (depth > 0.85f) {
                // Apply smoothstep to expand the 0.85-1.0 range to 0.0-1.0
                float t = (depth - 0.85f) / 0.15f; // Map [0.85, 1.0] to [0, 1]
                remappedDepth = t * t * (3.0f - 2.0f * t); // Smoothstep for better visibility
            }

            // Apply gamma correction
            if (gamma != 1.0f) {
                remappedDepth = powf(remappedDepth, 1.0f / gamma);
            }

            // Apply brightness
            remappedDepth *= brightness;
            remappedDepth = std::min(1.0f, std::max(0.0f, remappedDepth)); // Clamp to [0,1]

            // Invert if requested
            if (invert) {
                remappedDepth = 1.0f - remappedDepth;
            }

            BYTE gray = (BYTE)(remappedDepth * 255.0f);

            // Overlay 5x4 raycast grid visualization (matches ffeshader.cpp raycast pattern)
            // Grid covers central 80% of screen (10% margin on each edge)
            bool isRaycastPoint = false;
            if (showRaycastGrid) {
                const int gridWidth = 5;
                const int gridHeight = 4;
                const float edgeMargin = 0.1f;

                // Calculate exact raycast positions (matching ffeshader.cpp:3635)
                // u = edgeMargin + (x + 0.5f) / gridWidth * (1.0f - 2.0f * edgeMargin)
                for (int gy = 0; gy < gridHeight && !isRaycastPoint; gy++) {
                    for (int gx = 0; gx < gridWidth && !isRaycastPoint; gx++) {
                        float rayU = edgeMargin + (gx + 0.5f) / gridWidth * (1.0f - 2.0f * edgeMargin);
                        float rayV = edgeMargin + (gy + 0.5f) / gridHeight * (1.0f - 2.0f * edgeMargin);

                        // Convert to pixel coordinates in Hi-Z texture
                        float rayPixelX = rayU * width;
                        float rayPixelY = rayV * height;

                        // Check if current pixel is near this raycast position (larger radius for visibility)
                        float dx = (float)x - rayPixelX;
                        float dy = (float)y - rayPixelY;
                        float distSq = dx * dx + dy * dy;

                        // Use larger radius (5 pixels) to make dots visible
                        if (distSq < 25.0f) {
                            isRaycastPoint = true;
                        }
                    }
                }
            }

            if (isRaycastPoint) {
                // Red dot for raycast sample points
                pixels[y * (rect.Pitch / 4) + x] = D3DCOLOR_ARGB(255, 255, 0, 0);
            } else {
                pixels[y * (rect.Pitch / 4) + x] = D3DCOLOR_ARGB(255, gray, gray, gray);
            }
        }
    }

    mHiZVisualizationTexture->UnlockRect(0);
}
