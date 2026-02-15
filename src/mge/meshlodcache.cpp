#include "meshlodcache.h"
#include "meshoptimizer.h"
#include "support/log.h"
#include "mge_tracy.h"
#include <algorithm>

MeshLODCache::MeshLODCache()
{
}

MeshLODCache::~MeshLODCache()
{
    clear();
}

void MeshLODCache::clear()
{
    mCache.clear();
}

void MeshLODCache::getStats(UINT& totalEntries, UINT& totalSimplifiedTris, UINT& totalOriginalTris) const
{
    totalEntries = (UINT)mCache.size();
    totalSimplifiedTris = 0;
    totalOriginalTris = 0;

    for (const auto& pair : mCache) {
        totalOriginalTris += pair.second.originalTriCount;
        totalSimplifiedTris += pair.second.simplifiedTriCount;
    }
}

const SimplifiedMesh* MeshLODCache::getOrCreateLOD(
    const MeshSignature& signature,
    IDirect3DVertexBuffer9* vb,
    UINT vbOffset,
    UINT vbStride,
    IDirect3DIndexBuffer9* ib,
    UINT ibBase,
    UINT startIndex,
    UINT indexCount,
    DWORD fvf,
    float targetError
)
{
    MGE_ZoneScoped;

    // Check if mesh is large enough to bother simplifying
    UINT triCount = indexCount / 3;
    if (triCount < MIN_TRI_COUNT) {
        return nullptr;  // Too small, use original mesh
    }

    // Check cache
    auto it = mCache.find(signature);
    if (it != mCache.end()) {
        return &it->second;
    }

    // Create new simplified mesh
    SimplifiedMesh simplified;
    if (simplifyMesh(simplified, vb, vbOffset, vbStride, ib, ibBase, startIndex, indexCount, fvf, targetError)) {
        // Add to cache
        auto result = mCache.emplace(signature, std::move(simplified));
        return &result.first->second;
    }

    return nullptr;  // Simplification failed
}

bool MeshLODCache::simplifyMesh(
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
)
{
    MGE_ZoneScoped;

    if (!vb || !ib) return false;

    // Check if FVF has position data
    bool hasPosition = (fvf & D3DFVF_POSITION_MASK) != 0;
    if (!hasPosition) return false;

    // Lock vertex buffer
    void* pVertices = nullptr;
    HRESULT hr = vb->Lock(vbOffset, 0, &pVertices, D3DLOCK_READONLY | D3DLOCK_DONOTWAIT);
    if (FAILED(hr)) return false;

    // Lock index buffer
    void* pIndices = nullptr;
    hr = ib->Lock(0, 0, &pIndices, D3DLOCK_READONLY | D3DLOCK_DONOTWAIT);
    if (FAILED(hr)) {
        vb->Unlock();
        return false;
    }

    // Determine index format
    D3DINDEXBUFFER_DESC ibDesc;
    ib->GetDesc(&ibDesc);
    bool is16Bit = (ibDesc.Format == D3DFMT_INDEX16);

    // Extract positions and indices
    std::vector<D3DXVECTOR3> positions;
    std::vector<UINT> indices;

    positions.reserve(indexCount);  // Over-allocate
    indices.reserve(indexCount);

    for (UINT i = 0; i < indexCount; i++) {
        UINT idx;
        if (is16Bit) {
            idx = ((WORD*)pIndices)[startIndex + i] + ibBase;
        } else {
            idx = ((DWORD*)pIndices)[startIndex + i] + ibBase;
        }

        D3DXVECTOR3 pos = *(D3DXVECTOR3*)((BYTE*)pVertices + idx * vbStride);
        positions.push_back(pos);
        indices.push_back(i);  // Use local indices for simplification
    }

    ib->Unlock();
    vb->Unlock();

    // Simplify using meshoptimizer
    size_t targetIndexCount = indexCount / 2;  // Target 50% reduction
    std::vector<UINT> simplified(indexCount);

    float resultError = 0.0f;
    size_t simplifiedCount = meshopt_simplify(
        simplified.data(),
        indices.data(),
        indexCount,
        (const float*)positions.data(),
        positions.size(),
        sizeof(D3DXVECTOR3),
        targetIndexCount,
        targetError,
        0,  // flags
        &resultError
    );

    // Check if simplification was successful
    if (simplifiedCount == 0 || simplifiedCount >= indexCount) {
        return false;  // No reduction or failed
    }

    simplified.resize(simplifiedCount);

    // Build unique vertex set
    std::vector<D3DXVECTOR3> uniquePositions;
    std::vector<UINT> remappedIndices;
    uniquePositions.reserve(simplifiedCount);
    remappedIndices.reserve(simplifiedCount);

    std::unordered_map<UINT, UINT> vertexRemap;
    for (UINT idx : simplified) {
        auto it = vertexRemap.find(idx);
        if (it == vertexRemap.end()) {
            UINT newIdx = (UINT)uniquePositions.size();
            uniquePositions.push_back(positions[idx]);
            vertexRemap[idx] = newIdx;
            remappedIndices.push_back(newIdx);
        } else {
            remappedIndices.push_back(it->second);
        }
    }

    // Store results
    outMesh.positions = std::move(uniquePositions);
    outMesh.indices = std::move(remappedIndices);
    outMesh.targetError = resultError;
    outMesh.originalTriCount = indexCount / 3;
    outMesh.simplifiedTriCount = (UINT)simplifiedCount / 3;

    LOG::logline(">> Simplified mesh: %u -> %u tris (%.1f%% reduction, error: %.3f)",
        outMesh.originalTriCount,
        outMesh.simplifiedTriCount,
        100.0f * (1.0f - (float)outMesh.simplifiedTriCount / (float)outMesh.originalTriCount),
        resultError
    );

    return true;
}
