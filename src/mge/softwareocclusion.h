#pragma once

#include <d3d9.h>
#include <d3dx9math.h>
#include <vector>
#include <unordered_set>
#include "meshlodcache.h"

// Minimal software occlusion culler for MGE-XE
// Based on Intel Software Occlusion Culling (blog branch)
// Eliminates GPU→CPU transfer overhead by rasterizing depth on CPU

// Cached mesh geometry for avoiding repeated VB/IB locks
struct CachedMeshGeometry {
    std::vector<D3DXVECTOR3> positions;  // Object-space vertex positions
    std::vector<UINT> indices;           // Triangle indices (for triangle list)
    bool is16BitIndices;
};

// Key for mesh cache lookup
struct MeshCacheKey {
    void* vb;
    void* ib;
    UINT vbOffset;
    UINT vbStride;
    UINT startIndex;
    UINT primCount;

    bool operator==(const MeshCacheKey& other) const {
        return vb == other.vb && ib == other.ib &&
               vbOffset == other.vbOffset && vbStride == other.vbStride &&
               startIndex == other.startIndex && primCount == other.primCount;
    }
};

struct MeshCacheKeyHash {
    std::size_t operator()(const MeshCacheKey& k) const {
        size_t h = std::hash<void*>{}(k.vb);
        h ^= std::hash<void*>{}(k.ib) << 1;
        h ^= std::hash<UINT>{}(k.vbOffset) << 2;
        h ^= std::hash<UINT>{}(k.startIndex) << 3;
        h ^= std::hash<UINT>{}(k.primCount) << 4;
        return h;
    }
};

class SoftwareOcclusionCuller {
public:
    // Simple mesh identifier for blacklist (vb + ib pointers uniquely identify a mesh)
    struct MeshID {
        void* vb;
        void* ib;

        bool operator==(const MeshID& other) const {
            return vb == other.vb && ib == other.ib;
        }
    };

    struct MeshIDHash {
        std::size_t operator()(const MeshID& m) const {
            return std::hash<void*>{}(m.vb) ^ (std::hash<void*>{}(m.ib) << 1);
        }
    };

    SoftwareOcclusionCuller();
    ~SoftwareOcclusionCuller();

    // Initialize with depth buffer resolution
    void init(UINT width, UINT height);
    void shutdown();

    // Rasterize occluder geometry to CPU depth buffer
    // Call this after recordMW rendering, before testing occludees
    void rasterizeOccluders(
        const std::vector<D3DXVECTOR3>& vertices,
        const std::vector<UINT>& indices,
        const D3DXMATRIX& view,
        const D3DXMATRIX& proj
    );

    // Extract and rasterize geometry from a single mesh (used during recording)
    // Returns number of pixels written to Hi-Z buffer (for efficiency tracking)
    int rasterizeMesh(
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
    );

    // Test bounding boxes against CPU depth buffer
    // Returns true if visible, false if occluded
    bool testBoundingBox(
        const D3DXVECTOR3& bboxMin,
        const D3DXVECTOR3& bboxMax,
        const D3DXMATRIX& view,
        const D3DXMATRIX& proj
    );

    // Clear depth buffer for next frame
    void clear();

    // Build Hi-Z mipmap pyramid from rasterized depth (call after all rasterization done)
    void buildHiZPyramid();

    // Save CPU Hi-Z buffers to disk for debugging (DDS + PNG)
    void saveToDisk(IDirect3DDevice9* device, const char* directory);

    // Mesh blacklist: track inefficient occluders (thin objects with poor pixel coverage)
    bool isMeshBlacklisted(IDirect3DVertexBuffer9* vb, IDirect3DIndexBuffer9* ib) const;
    void blacklistMesh(IDirect3DVertexBuffer9* vb, IDirect3DIndexBuffer9* ib);
    void clearBlacklist();

    // Mesh geometry cache: avoids VB/IB locks after first access
    void clearMeshCache();

    // Mesh LOD cache for simplified geometry (public for pre-caching during recording)
    MeshLODCache mLODCache;

    // ImGui Hi-Z visualization
    void uploadHiZToTexture(IDirect3DDevice9* device, int mipLevel, const D3DXMATRIX& proj, bool invert = false);
    IDirect3DTexture9* getHiZTexture() const { return mHiZVisualizationTexture; }
    int getHiZMipLevels() const { return mNumMipLevels; }
    UINT getHiZWidth(int mip) const { return (mip < mNumMipLevels) ? mHiZWidth[mip] : 0; }
    UINT getHiZHeight(int mip) const { return (mip < mNumMipLevels) ? mHiZHeight[mip] : 0; }
    float* getMip0Buffer() const { return mHiZBuffer[0]; }
    UINT getMip0Width() const { return mHiZWidth[0]; }
    UINT getMip0Height() const { return mHiZHeight[0]; }

private:
    UINT mWidth;
    UINT mHeight;

    // Hi-Z pyramid: mip 0 = half-res, mip N = 8x8 minimum
    static const int MAX_MIP_LEVELS = 16;
    float* mHiZBuffer[MAX_MIP_LEVELS];  // Hierarchical depth mipmaps
    UINT mHiZWidth[MAX_MIP_LEVELS];
    UINT mHiZHeight[MAX_MIP_LEVELS];
    int mNumMipLevels;

    D3DXMATRIX mView;
    D3DXMATRIX mProj;

    // Blacklist of inefficient meshes (thin objects with poor pixel coverage ratio)
    std::unordered_set<MeshID, MeshIDHash> mBlacklistedMeshes;

    // Cache of mesh geometry to avoid VB/IB locks after first access
    std::unordered_map<MeshCacheKey, CachedMeshGeometry, MeshCacheKeyHash> mMeshCache;

    // ImGui visualization texture
    IDirect3DTexture9* mHiZVisualizationTexture;
    UINT mVisTexWidth, mVisTexHeight;
};
