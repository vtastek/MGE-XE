#pragma once

#include <d3d9.h>
#include <d3dx9math.h>
#include <unordered_map>
#include <vector>

// Mesh LOD cache for software occlusion culling
// Generates simplified versions of meshes for faster CPU rasterization
// Based on meshoptimizer library

struct MeshSignature {
    UINT vertexCount;
    UINT indexCount;
    DWORD fvf;
    UINT vbStride;

    bool operator==(const MeshSignature& other) const {
        return vertexCount == other.vertexCount &&
               indexCount == other.indexCount &&
               fvf == other.fvf &&
               vbStride == other.vbStride;
    }
};

// Hash function for MeshSignature
struct MeshSignatureHash {
    size_t operator()(const MeshSignature& sig) const {
        size_t h1 = std::hash<UINT>()(sig.vertexCount);
        size_t h2 = std::hash<UINT>()(sig.indexCount);
        size_t h3 = std::hash<DWORD>()(sig.fvf);
        size_t h4 = std::hash<UINT>()(sig.vbStride);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
    }
};

// Simplified mesh data for CPU rasterization
struct SimplifiedMesh {
    std::vector<D3DXVECTOR3> positions;  // Simplified vertex positions
    std::vector<UINT> indices;           // Simplified indices
    float targetError;                   // Simplification error threshold
    UINT originalTriCount;               // Original triangle count
    UINT simplifiedTriCount;             // Simplified triangle count
};

class MeshLODCache {
public:
    MeshLODCache();
    ~MeshLODCache();

    // Get or create simplified mesh for given signature
    // Returns nullptr if mesh is too small to simplify
    const SimplifiedMesh* getOrCreateLOD(
        const MeshSignature& signature,
        IDirect3DVertexBuffer9* vb,
        UINT vbOffset,
        UINT vbStride,
        IDirect3DIndexBuffer9* ib,
        UINT ibBase,
        UINT startIndex,
        UINT indexCount,
        DWORD fvf,
        float targetError = 0.01f  // 1% error threshold
    );

    // Clear the cache (call when loading new cells)
    void clear();

    // Get cache statistics
    void getStats(UINT& totalEntries, UINT& totalSimplifiedTris, UINT& totalOriginalTris) const;

private:
    // Cache: signature -> simplified mesh
    std::unordered_map<MeshSignature, SimplifiedMesh, MeshSignatureHash> mCache;

    // Minimum triangle count to bother simplifying
    static const UINT MIN_TRI_COUNT = 100;

    // Extract mesh data and simplify
    bool simplifyMesh(
        SimplifiedMesh& outMesh,
        IDirect3DVertexBuffer9* vb,
        UINT vbOffset,
        UINT vbStride,
        IDirect3DIndexBuffer9* ib,
        UINT ibBase,
        UINT startIndex,
        UINT indexCount,
        DWORD fvf,
        float targetError
    );
};
