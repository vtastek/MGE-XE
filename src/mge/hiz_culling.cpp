// Hi-Z occlusion culling implementation
// Split from ffeshader.cpp for organization - these are still FixedFunctionShader members
#include "ffeshader.h"
#include "configuration.h"
#include "support/log.h"
#include "distantland.h"
#include "imgui_manager.h"
#include "mge_tracy.h"
#include "texture_suffix.h"
#include "mwbridge.h"

#include <algorithm>
#include <vector>
#include <set>
#include <map>

// External reference to file-scope variable in ffeshader.cpp
extern float lastPrepareMs;
extern bool deviceCallsSafeInPrepare;

// Geometry hash cache: VB+offset+stride+count -> content hash
// Allows identifying identical geometry across different VB allocations
struct VBGeometryKey {
    IDirect3DVertexBuffer9* vb;
    UINT offset;
    UINT stride;
    UINT vertCount;
    DWORD fvf;

    bool operator==(const VBGeometryKey& o) const {
        return vb == o.vb && offset == o.offset && stride == o.stride &&
               vertCount == o.vertCount && fvf == o.fvf;
    }
};

struct VBGeometryKeyHash {
    size_t operator()(const VBGeometryKey& k) const {
        size_t h = reinterpret_cast<size_t>(k.vb);
        h ^= k.offset + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= k.stride + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= k.vertCount + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= k.fvf + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static std::unordered_map<VBGeometryKey, size_t, VBGeometryKeyHash> s_geometryHashCache;
static SRWLOCK s_geometryHashLock = SRWLOCK_INIT;

// Local mesh cache: geometry hash -> centered mesh data
// Allows recognizing identical geometry at different world positions
static std::unordered_map<size_t, LocalMeshData> s_localMeshCache;
static SRWLOCK s_localMeshLock = SRWLOCK_INIT;

static void hashCombineValue(size_t& h, size_t value) {
    h ^= value + 0x9e3779b9 + (h << 6) + (h >> 2);
}

// Cell batch cache: caches merged VB/IB per cell+layout to avoid regenerating geometry every frame.
struct CellBatchCacheKey {
    void* cellPtr = nullptr;
    size_t layoutHash = 0;

    bool operator==(const CellBatchCacheKey& other) const {
        return cellPtr == other.cellPtr && layoutHash == other.layoutHash;
    }
};

struct CellBatchCacheKeyHash {
    size_t operator()(const CellBatchCacheKey& key) const {
        size_t h = reinterpret_cast<size_t>(key.cellPtr);
        hashCombineValue(h, key.layoutHash);
        return h;
    }
};

static std::unordered_map<CellBatchCacheKey, CellBatchCache, CellBatchCacheKeyHash> s_cellBatchCaches;
static SRWLOCK s_cellBatchCacheLock = SRWLOCK_INIT;
static constexpr size_t MAX_CACHED_CELL_BATCH_LAYOUTS = 16;
static uint64_t s_cellBatchCacheUseSerial = 0;

static MeshKey makeMeshKey(const FixedFunctionShader::HLSLRecordedCall& call);

static CachedBatchTemplate makeCachedBatchTemplate(const MergedBatch& batch) {
    CachedBatchTemplate templ = {};
    templ.key = batch.key;
    templ.totalVertices = batch.totalVertices;
    templ.totalIndices = batch.totalIndices;
    templ.drawDataOffset = batch.drawDataOffset;
    templ.drawCount = (UINT)batch.callIndices.size();
    return templ;
}

static CachedMergedCallLayout makeCachedMergedCallLayout(
    const FixedFunctionShader::HLSLRecordedCall& call,
    const MergedBatchKey& batchKey)
{
    CachedMergedCallLayout layout = {};
    layout.mesh = makeMeshKey(call);
    layout.vbOffset = call.rs.vbOffset;
    layout.vbStride = call.rs.vbStride;
    layout.recordMWIndex = call.recordMWIndex;
    layout.batchKey = batchKey;
    layout.worldTransform = call.rs.worldTransforms[0];
    layout.diffuseMaterial = call.frs.material.diffuse;
    layout.ambientMaterial = call.frs.material.ambient;
    layout.emissiveMaterial = call.frs.material.emissive;
    layout.alphaRef = call.rs.alphaRef;
    layout.vertexMaterial = (uint8_t)call.sk.vertexMaterial;
    return layout;
}

static void hashFloatValue(size_t& h, float value) {
    DWORD bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    hashCombineValue(h, bits);
}

static void hashMeshKeyValue(size_t& h, const MeshKey& key) {
    hashCombineValue(h, reinterpret_cast<size_t>(key.vb));
    hashCombineValue(h, reinterpret_cast<size_t>(key.ib));
    hashCombineValue(h, key.fvf);
    hashCombineValue(h, key.baseIndex);
    hashCombineValue(h, key.vertCount);
    hashCombineValue(h, key.startIndex);
    hashCombineValue(h, key.primCount);
}

static void hashMergedBatchKeyValue(size_t& h, const MergedBatchKey& key) {
    hashCombineValue(h, reinterpret_cast<size_t>(key.texture));
    hashCombineValue(h, key.blendState);
    hashCombineValue(h, key.zState);
    hashCombineValue(h, key.cullMode);
    hashCombineValue(h, key.useLighting);
    hashCombineValue(h, key.bin);
    hashCombineValue(h, key.fvf);
    hashCombineValue(h, key.stride);
}

static void hashColorValue(size_t& h, const D3DCOLORVALUE& color) {
    hashFloatValue(h, color.r);
    hashFloatValue(h, color.g);
    hashFloatValue(h, color.b);
    hashFloatValue(h, color.a);
}

static void hashMatrixValue(size_t& h, const D3DXMATRIX& matrix) {
    const float* values = &matrix._11;
    for (int i = 0; i < 16; ++i) {
        hashFloatValue(h, values[i]);
    }
}

static void hashCachedBatchTemplateValue(size_t& h, const CachedBatchTemplate& templ) {
    hashMergedBatchKeyValue(h, templ.key);
    hashCombineValue(h, templ.totalVertices);
    hashCombineValue(h, templ.totalIndices);
    hashCombineValue(h, templ.drawDataOffset);
    hashCombineValue(h, templ.drawCount);
}

static void hashCachedMergedCallLayoutValue(size_t& h, const CachedMergedCallLayout& layout) {
    hashCombineValue(h, static_cast<size_t>(layout.recordMWIndex));
    if (layout.recordMWIndex >= 0) {
        hashCombineValue(h, layout.mesh.fvf);
        hashCombineValue(h, layout.mesh.vertCount);
        hashCombineValue(h, layout.mesh.primCount);
        hashCombineValue(h, layout.vbStride);
    } else {
        hashMeshKeyValue(h, layout.mesh);
        hashCombineValue(h, layout.vbOffset);
        hashCombineValue(h, layout.vbStride);
    }
    hashMergedBatchKeyValue(h, layout.batchKey);
}

static size_t computeCellBatchLayoutHash(
    const std::vector<CachedBatchTemplate>& batchTemplates,
    const std::vector<CachedMergedCallLayout>& mergedLayout)
{
    size_t h = sizeof(size_t) == 8 ? static_cast<size_t>(1469598103934665603ull) : static_cast<size_t>(2166136261u);
    hashCombineValue(h, batchTemplates.size());
    for (const auto& templ : batchTemplates) {
        hashCachedBatchTemplateValue(h, templ);
    }

    hashCombineValue(h, mergedLayout.size());
    for (const auto& layout : mergedLayout) {
        hashCachedMergedCallLayoutValue(h, layout);
    }
    return h;
}

static bool lessMeshKey(const MeshKey& lhs, const MeshKey& rhs) {
    if (lhs.vb != rhs.vb) return lhs.vb < rhs.vb;
    if (lhs.ib != rhs.ib) return lhs.ib < rhs.ib;
    if (lhs.fvf != rhs.fvf) return lhs.fvf < rhs.fvf;
    if (lhs.baseIndex != rhs.baseIndex) return lhs.baseIndex < rhs.baseIndex;
    if (lhs.vertCount != rhs.vertCount) return lhs.vertCount < rhs.vertCount;
    if (lhs.startIndex != rhs.startIndex) return lhs.startIndex < rhs.startIndex;
    return lhs.primCount < rhs.primCount;
}

static bool lessColorValue(const D3DCOLORVALUE& lhs, const D3DCOLORVALUE& rhs) {
    if (lhs.r != rhs.r) return lhs.r < rhs.r;
    if (lhs.g != rhs.g) return lhs.g < rhs.g;
    if (lhs.b != rhs.b) return lhs.b < rhs.b;
    return lhs.a < rhs.a;
}

static bool lessMatrix(const D3DXMATRIX& lhs, const D3DXMATRIX& rhs) {
    const float* l = &lhs._11;
    const float* r = &rhs._11;
    for (int i = 0; i < 16; ++i) {
        if (l[i] != r[i]) return l[i] < r[i];
    }
    return false;
}

static bool lessMergedBatchKey(const MergedBatchKey& lhs, const MergedBatchKey& rhs) {
    if (lhs.texture != rhs.texture) return lhs.texture < rhs.texture;
    if (lhs.blendState != rhs.blendState) return lhs.blendState < rhs.blendState;
    if (lhs.zState != rhs.zState) return lhs.zState < rhs.zState;
    if (lhs.cullMode != rhs.cullMode) return lhs.cullMode < rhs.cullMode;
    if (lhs.useLighting != rhs.useLighting) return lhs.useLighting < rhs.useLighting;
    if (lhs.bin != rhs.bin) return lhs.bin < rhs.bin;
    if (lhs.fvf != rhs.fvf) return lhs.fvf < rhs.fvf;
    return lhs.stride < rhs.stride;
}

static bool lessCachedMergedCallLayout(const CachedMergedCallLayout& lhs, const CachedMergedCallLayout& rhs) {
    if (lhs.recordMWIndex != rhs.recordMWIndex)
        return lhs.recordMWIndex < rhs.recordMWIndex;

    if (lessMeshKey(lhs.mesh, rhs.mesh)) return true;
    if (lessMeshKey(rhs.mesh, lhs.mesh)) return false;
    if (lhs.vbOffset != rhs.vbOffset) return lhs.vbOffset < rhs.vbOffset;
    if (lhs.vbStride != rhs.vbStride) return lhs.vbStride < rhs.vbStride;
    if (lessMergedBatchKey(lhs.batchKey, rhs.batchKey)) return true;
    return false;
}

// Invalidate cell batch cache for a specific cell
void FixedFunctionShader::invalidateCellBatchCache(void* cellPtr) {
    AcquireSRWLockExclusive(&s_cellBatchCacheLock);
    for (auto it = s_cellBatchCaches.begin(); it != s_cellBatchCaches.end(); ) {
        if (it->first.cellPtr != cellPtr) {
            ++it;
            continue;
        }
        it->second.release();
        it = s_cellBatchCaches.erase(it);
    }
    ReleaseSRWLockExclusive(&s_cellBatchCacheLock);
}

// Clear all cell batch caches (called on interior/exterior transition)
void FixedFunctionShader::clearAllCellBatchCaches() {
    AcquireSRWLockExclusive(&s_cellBatchCacheLock);
    for (auto& kv : s_cellBatchCaches) {
        kv.second.release();
    }
    s_cellBatchCaches.clear();
    ReleaseSRWLockExclusive(&s_cellBatchCacheLock);
}

// Evict least recently used cache if over limit.
// Caller must hold s_cellBatchCacheLock exclusively.
static void evictLRUCellBatchCacheLocked(const CellBatchCacheKey& protectedKey) {
    while (s_cellBatchCaches.size() > MAX_CACHED_CELL_BATCH_LAYOUTS) {
        auto victim = s_cellBatchCaches.end();
        uint64_t oldestUse = ~0ull;

        for (auto it = s_cellBatchCaches.begin(); it != s_cellBatchCaches.end(); ++it) {
            if (it->first == protectedKey)
                continue;
            if (it->second.lastUsedSerial < oldestUse) {
                oldestUse = it->second.lastUsedSerial;
                victim = it;
            }
        }

        if (victim != s_cellBatchCaches.end()) {
            victim->second.release();
            s_cellBatchCaches.erase(victim);
        } else {
            break;
        }
    }
}

// Store VB/IB in cell batch cache (called by hlsl_replay after building VB/IB)
void FixedFunctionShader::storeCellBatchCacheVB(void* cellPtr, size_t layoutHash, IDirect3DVertexBuffer9* vb,
                                                  IDirect3DIndexBuffer9* ib,
                                                  const std::vector<CachedDrawInfo>& drawInfos) {
    if (!cellPtr || layoutHash == 0 || !vb || !ib || drawInfos.empty())
        return;

    AcquireSRWLockExclusive(&s_cellBatchCacheLock);

    CellBatchCacheKey cacheKey = { cellPtr, layoutHash };
    auto it = s_cellBatchCaches.find(cacheKey);
    if (it == s_cellBatchCaches.end() || !it->second.valid || it->second.layoutHash != layoutHash) {
        LOG::logline("CellBatchCache: Rejected VB/IB store - missing layout cell=%p hash=%Ix",
                     cellPtr, layoutHash);
        ReleaseSRWLockExclusive(&s_cellBatchCacheLock);
        return;
    }

    CellBatchCache& cache = it->second;
    if (cache.mergedVB) {
        cache.mergedVB->Release();
        cache.mergedVB = nullptr;
    }
    if (cache.mergedIB) {
        cache.mergedIB->Release();
        cache.mergedIB = nullptr;
    }

    vb->AddRef();
    ib->AddRef();
    cache.mergedVB = vb;
    cache.mergedIB = ib;
    cache.drawInfos = drawInfos;
    cache.valid = true;
    cache.lastUsedSerial = ++s_cellBatchCacheUseSerial;

    UINT vbSizeBytes = 0;
    UINT ibSizeIndices = 0;
    for (const auto& info : drawInfos) {
        vbSizeBytes = std::max(vbSizeBytes, info.vbByteOffset + (info.vertCount * info.expandedStride));
        ibSizeIndices = std::max(ibSizeIndices, info.ibStartIndex + (info.primCount * 3));
    }
    cache.vbSizeBytes = vbSizeBytes;
    cache.ibSizeIndices = ibSizeIndices;

    LOG::logline("CellBatchCache: Stored VB/IB for cell=%p hash=%Ix, %d drawInfos",
                 cellPtr, layoutHash, (int)drawInfos.size());

    ReleaseSRWLockExclusive(&s_cellBatchCacheLock);
}

// Compute geometry hash from VB content (samples first N vertices for speed)
static size_t computeGeometryHash(IDirect3DVertexBuffer9* vb, UINT offset, UINT stride,
                                   UINT vertCount, DWORD fvf) {
    // Check cache first
    VBGeometryKey key = { vb, offset, stride, vertCount, fvf };

    AcquireSRWLockShared(&s_geometryHashLock);
    auto it = s_geometryHashCache.find(key);
    if (it != s_geometryHashCache.end()) {
        size_t cached = it->second;
        ReleaseSRWLockShared(&s_geometryHashLock);
        return cached;
    }
    ReleaseSRWLockShared(&s_geometryHashLock);

    // Compute hash from vertex data
    size_t hash = fvf;
    hash ^= vertCount + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= stride + 0x9e3779b9 + (hash << 6) + (hash >> 2);

    // Sample first few vertices for content hash
    const int SAMPLE_VERTS = 4;  // Sample first 4 vertices
    void* data = nullptr;
    UINT lockSize = std::min(vertCount, (UINT)SAMPLE_VERTS) * stride;

    if (vb && lockSize > 0 && SUCCEEDED(vb->Lock(offset, lockSize, &data, D3DLOCK_READONLY | D3DLOCK_NOOVERWRITE))) {
        const BYTE* bytes = static_cast<const BYTE*>(data);
        for (UINT i = 0; i < lockSize; i += 4) {
            DWORD val = 0;
            memcpy(&val, bytes + i, std::min(4U, lockSize - i));
            hash ^= val + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        vb->Unlock();
    }

    // Cache the result
    AcquireSRWLockExclusive(&s_geometryHashLock);
    s_geometryHashCache[key] = hash;
    ReleaseSRWLockExclusive(&s_geometryHashLock);

    return hash;
}

// Compute local mesh data: center geometry and compute position-independent hash
// Returns pointer to cached LocalMeshData, or nullptr if caching failed
static const LocalMeshData* getOrCreateLocalMesh(
    size_t geometryHash,  // Original geometry hash (world-space)
    IDirect3DVertexBuffer9* vb, UINT vbOffset, UINT vbStride, UINT vertCount, DWORD fvf,
    IDirect3DIndexBuffer9* ib, UINT startIndex, UINT primCount)
{
    // Check cache first
    AcquireSRWLockShared(&s_localMeshLock);
    auto it = s_localMeshCache.find(geometryHash);
    if (it != s_localMeshCache.end()) {
        const LocalMeshData* cached = &it->second;
        ReleaseSRWLockShared(&s_localMeshLock);
        return cached;
    }
    ReleaseSRWLockShared(&s_localMeshLock);

    // Need to compute local mesh - lock VB and IB
    LocalMeshData data;
    data.fvf = fvf;
    data.stride = vbStride;
    data.vertCount = vertCount;
    data.indexCount = primCount * 3;  // Triangle list

    // Lock VB to read vertices
    void* vbData = nullptr;
    UINT vbSize = vertCount * vbStride;
    if (!vb || FAILED(vb->Lock(vbOffset, vbSize, &vbData, D3DLOCK_READONLY))) {
        return nullptr;
    }

    // Calculate bounding box center from positions
    float minPos[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float maxPos[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    const BYTE* srcVerts = static_cast<const BYTE*>(vbData);

    for (UINT i = 0; i < vertCount; i++) {
        const float* pos = reinterpret_cast<const float*>(srcVerts + i * vbStride);
        for (int j = 0; j < 3; j++) {
            minPos[j] = std::min(minPos[j], pos[j]);
            maxPos[j] = std::max(maxPos[j], pos[j]);
        }
    }

    data.center[0] = (minPos[0] + maxPos[0]) * 0.5f;
    data.center[1] = (minPos[1] + maxPos[1]) * 0.5f;
    data.center[2] = (minPos[2] + maxPos[2]) * 0.5f;

    // Copy vertices with centered positions
    data.vertices.resize(vbSize);
    memcpy(data.vertices.data(), vbData, vbSize);

    // Center the positions in our copy
    BYTE* dstVerts = data.vertices.data();
    for (UINT i = 0; i < vertCount; i++) {
        float* pos = reinterpret_cast<float*>(dstVerts + i * vbStride);
        pos[0] -= data.center[0];
        pos[1] -= data.center[1];
        pos[2] -= data.center[2];
    }

    vb->Unlock();

    // Lock IB to read indices
    if (ib && data.indexCount > 0) {
        void* ibData = nullptr;
        // Note: startIndex is in indices, not bytes
        if (SUCCEEDED(ib->Lock(startIndex * sizeof(WORD), data.indexCount * sizeof(WORD), &ibData, D3DLOCK_READONLY))) {
            data.indices.resize(data.indexCount);
            memcpy(data.indices.data(), ibData, data.indexCount * sizeof(WORD));
            ib->Unlock();
        }
    }

    // Compute local hash from centered vertex positions only
    size_t localHash = fvf;
    localHash ^= vertCount + 0x9e3779b9 + (localHash << 6) + (localHash >> 2);

    // Hash centered positions (first N vertices for speed)
    const int SAMPLE_VERTS = std::min(8U, vertCount);
    for (int i = 0; i < SAMPLE_VERTS; i++) {
        const float* pos = reinterpret_cast<const float*>(data.vertices.data() + i * vbStride);
        // Quantize to reduce floating point noise
        int qx = (int)(pos[0] * 100.0f);
        int qy = (int)(pos[1] * 100.0f);
        int qz = (int)(pos[2] * 100.0f);
        localHash ^= qx + 0x9e3779b9 + (localHash << 6) + (localHash >> 2);
        localHash ^= qy + 0x9e3779b9 + (localHash << 6) + (localHash >> 2);
        localHash ^= qz + 0x9e3779b9 + (localHash << 6) + (localHash >> 2);
    }

    data.localHash = localHash;

    // Cache the result
    AcquireSRWLockExclusive(&s_localMeshLock);
    s_localMeshCache[geometryHash] = std::move(data);
    const LocalMeshData* result = &s_localMeshCache[geometryHash];
    ReleaseSRWLockExclusive(&s_localMeshLock);

    return result;
}

// Helper to access prep buffer's recorded calls (N-1 data being prepared)
static auto& currentRecordedCalls() {
    return FixedFunctionShader::getPrepBuffer().recordedCalls;
}

// Compute material hash for sorting (texture + blend/alpha/z state)
static size_t computeMaterialHash(const FixedFunctionShader::HLSLRecordedCall& call) {
    size_t h = reinterpret_cast<size_t>(call.rs.texture);
    // Pack blend state: alphaBlendEnable | (srcBlend << 4) | (destBlend << 8)
    DWORD blendPacked = (call.expectedState.captured ? call.expectedState.alphaBlendEnable : 0) |
                        ((call.expectedState.captured ? call.expectedState.srcBlend : 0) << 4) |
                        ((call.expectedState.captured ? call.expectedState.destBlend : 0) << 8);
    h ^= blendPacked + 0x9e3779b9 + (h << 6) + (h >> 2);
    // Pack z state: zEnable | (zWriteEnable << 2)
    DWORD zPacked = (call.expectedState.captured ? call.expectedState.zEnable : 1) |
                    ((call.expectedState.captured ? call.expectedState.zWriteEnable : 1) << 2);
    h ^= zPacked + 0x9e3779b9 + (h << 6) + (h >> 2);
    // Add cull mode
    h ^= (call.expectedState.captured ? call.expectedState.cullMode : D3DCULL_CW) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

// Sort recorded calls by material to minimize state changes
// Rules:
// - Sortable bins: Opaque, Grass (no depth dependencies within bin)
// - Depth-ordered bins: AlphaTested, Blending (preserve relative order)
// - Excluded: Skinning (bone palette per-object)
static void sortCallsByMaterial(std::vector<FixedFunctionShader::HLSLRecordedCall>& calls) {
    MGE_ZoneScopedN("sortCallsByMaterial");

    if (calls.empty()) return;

    // Create sort keys with original index for stability
    struct SortKey {
        size_t originalIndex;
        RenderBin bin;
        size_t materialHash;
        bool sortable;  // false for bins that must preserve order
    };

    std::vector<SortKey> sortKeys;
    sortKeys.reserve(calls.size());

    for (size_t i = 0; i < calls.size(); i++) {
        const auto& call = calls[i];
        SortKey key;
        key.originalIndex = i;
        key.bin = call.bin;
        key.materialHash = computeMaterialHash(call);
        // Sortable: Opaque, Grass, Terrain
        // Not sortable: Skinning (bone state), AlphaTested (depth), Blending (depth)
        key.sortable = (call.bin == RenderBin::Opaque || call.bin == RenderBin::Grass || call.bin == RenderBin::Terrain);
        sortKeys.push_back(key);
    }

    // Stable sort: first by bin, then by material hash (for sortable bins), then by original index
    std::stable_sort(sortKeys.begin(), sortKeys.end(), [](const SortKey& a, const SortKey& b) {
        // Primary: bin order
        if (a.bin != b.bin) return static_cast<int>(a.bin) < static_cast<int>(b.bin);
        // Secondary: for sortable bins, sort by material; otherwise preserve order
        if (a.sortable && b.sortable) {
            if (a.materialHash != b.materialHash) return a.materialHash < b.materialHash;
        }
        // Tertiary: preserve original order (stable sort guarantees this for equal keys)
        return a.originalIndex < b.originalIndex;
    });

    // Reorder calls according to sort keys
    std::vector<FixedFunctionShader::HLSLRecordedCall> sortedCalls;
    sortedCalls.reserve(calls.size());
    for (const auto& key : sortKeys) {
        sortedCalls.push_back(std::move(calls[key.originalIndex]));
    }
    calls = std::move(sortedCalls);
}

// executeHiZCulling - Pure CPU work: bbox computation, occluder rasterization, Hi-Z pyramid build,
// visibility testing on recordMW and recordedCalls. Takes view/proj as parameters (no D3D device access).
// This is a draw-thread candidate: touches no GPU state.
void FixedFunctionShader::executeHiZCulling(const D3DXMATRIX& currentView, const D3DXMATRIX& currentProj) {
    MGE_ZoneScopedN("Execute Hi-Z Culling");

    // Hi-Z visibility is only consumed by the HLSL replacement pipeline. Standard/PPL
    // lighting still replays recordMW for the depth texture, so building a pyramid here
    // just adds CPU work without filtering any draws.
    if (Configuration.PerPixelLightFlags != 2) {
        return;
    }

    // Resize visibility results for this frame (indexed by draw order)
    // N-1: HLSL mode uses prep buffer (previous frame's data being prepared), legacy mode uses global static
    auto& renderBuf = getPrepBuffer();
    const auto& activeRecordMW = renderBuf.recordMW;
    visibilityResults.assign(activeRecordMW.size(), -1);  // -1 = not yet tested

    // Phase 2a: Compute deferred bboxes and rasterize occluders from recorded HLSL calls
    // This must happen before Hi-Z build so the depth pass benefits from culling
    auto& recCalls = renderBuf.recordedCalls;
    if (!recCalls.empty() && !hiZBuiltThisFrame) {
        // Compute bounding boxes for calls that missed the cache during recording
        {
            MGE_ZoneScopedN("Prepare: BBox Cache Misses");
            for (auto& call : recCalls) {
                if (!call.hasBoundingBox) {
                    call.hasBoundingBox = computeBoundingBox(&call.rs, call.bboxMin, call.bboxMax);
                    if (call.hasBoundingBox) {
                        VBIBKey key{call.rs.vb, call.rs.ib};
                        // Use per-buffer bboxLookup to avoid race with main thread
                        renderBuf.bboxLookup[key] = {call.bboxMin, call.bboxMax};
                    }
                }
            }
        }

        // Occluder selection and rasterization
        {
            MGE_ZoneScopedN("Prepare: Occluder Rasterization");
            int occluderCount = 0;
            const int MAX_OCCLUDERS = ImGuiManager::GetOccluderMaxCount();
            const float MIN_SCREEN_COVERAGE = 0.01f;
            rasterizedOccluderMeshes.clear();

            // N-1: Use rendering buffer (previous frame's matrices)
            auto& activeFb = renderBuf;
            D3DXMATRIX viewProj = activeFb.view * activeFb.proj;

            for (auto& call : recCalls) {
                if (occluderCount >= MAX_OCCLUDERS) break;

                const RenderedState* rs = &call.rs;
                bool isTransparent = (rs->alphaTest || rs->blendEnable);
                if (!rs->vb || !rs->ib || isTransparent || !call.hasBoundingBox) continue;

                float bboxSizeX = call.bboxMax.x - call.bboxMin.x;
                float bboxSizeY = call.bboxMax.y - call.bboxMin.y;
                float bboxSizeZ = call.bboxMax.z - call.bboxMin.z;
                float bboxSize = std::max({bboxSizeX, bboxSizeY, bboxSizeZ});

                const UINT BASE_TRIANGLES = 2;
                const UINT MAX_TRIANGLES = 400;
                const float MAX_SIZE = 3000.0f;
                float sizeRatio = std::min(bboxSize / MAX_SIZE, 1.0f);
                UINT allowedTriangles = BASE_TRIANGLES + (UINT)(sizeRatio * (MAX_TRIANGLES - BASE_TRIANGLES));

                if (rs->primCount > allowedTriangles) continue;

                // Screen coverage test
                D3DXVECTOR3 corners[8] = {
                    D3DXVECTOR3(call.bboxMin.x, call.bboxMin.y, call.bboxMin.z),
                    D3DXVECTOR3(call.bboxMax.x, call.bboxMin.y, call.bboxMin.z),
                    D3DXVECTOR3(call.bboxMin.x, call.bboxMax.y, call.bboxMin.z),
                    D3DXVECTOR3(call.bboxMax.x, call.bboxMax.y, call.bboxMin.z),
                    D3DXVECTOR3(call.bboxMin.x, call.bboxMin.y, call.bboxMax.z),
                    D3DXVECTOR3(call.bboxMax.x, call.bboxMin.y, call.bboxMax.z),
                    D3DXVECTOR3(call.bboxMin.x, call.bboxMax.y, call.bboxMax.z),
                    D3DXVECTOR3(call.bboxMax.x, call.bboxMax.y, call.bboxMax.z)
                };

                float minX = FLT_MAX, maxX = -FLT_MAX;
                float minY = FLT_MAX, maxY = -FLT_MAX;
                bool allBehindCamera = true;

                for (int i = 0; i < 8; i++) {
                    D3DXVECTOR4 clipPos;
                    D3DXVec3Transform(&clipPos, &corners[i], &viewProj);
                    if (clipPos.w > 0.0f) {
                        allBehindCamera = false;
                        float ndcX = std::max(-1.0f, std::min(1.0f, clipPos.x / clipPos.w));
                        float ndcY = std::max(-1.0f, std::min(1.0f, clipPos.y / clipPos.w));
                        minX = std::min(minX, ndcX);
                        maxX = std::max(maxX, ndcX);
                        minY = std::min(minY, ndcY);
                        maxY = std::max(maxY, ndcY);
                    }
                }

                if (allBehindCamera) continue;
                float screenCoverage = ((maxX - minX) * (maxY - minY)) / 4.0f;
                if (screenCoverage < MIN_SCREEN_COVERAGE) continue;

                MeshKey occluderKey;
                occluderKey.vb = rs->vb;
                occluderKey.ib = rs->ib;
                occluderKey.fvf = rs->fvf;
                occluderKey.baseIndex = rs->baseIndex;
                occluderKey.vertCount = rs->vertCount;
                occluderKey.startIndex = rs->startIndex;
                occluderKey.primCount = rs->primCount;
                rasterizedOccluderMeshes.insert(occluderKey);

                softwareOcclusionCuller.rasterizeMesh(
                    rs->vb, rs->vbOffset, rs->vbStride,
                    rs->ib, rs->baseIndex, rs->startIndex, rs->primCount, rs->primType,
                    rs->fvf, rs->worldTransforms[0],
                    activeFb.view, activeFb.proj
                );
                occluderCount++;
            }
        }
    }

    // Build Hi-Z mipmap pyramid from rasterized occluders (only once per frame)
    if (!hiZBuiltThisFrame) {
        MGE_ZoneScopedN("Build Hi-Z Pyramid");
        softwareOcclusionCuller.buildHiZPyramid();
        if (ImGuiManager::GetShowHiZInterface()) {
            // N-1: Use rendering buffer's projection matrix
            const D3DXMATRIX& uploadProj = renderBuf.proj;
            softwareOcclusionCuller.uploadHiZToTexture(reinterpret_cast<IDirect3DDevice9*>(device), ImGuiManager::GetHiZDisplayMip(), uploadProj, ImGuiManager::GetHiZInvert());
        }
        hiZBuiltThisFrame = true;
    }

    // Hi-Z bypass toggle for terrain hole diagnosis
    if (ImGuiManager::GetDisableHiZCulling()) {
        for (size_t i = 0; i < visibilityResults.size(); i++)
            visibilityResults[i] = 1;
        // Mark all recordedCalls visible when culling disabled
        for (auto& call : recCalls) {
            call.shouldRender = true;
        }
        LOG::logline(">> Depth: %d objects (Hi-Z culling DISABLED by toggle)",
                     (int)DistantLand::recordMW.size());
        return;
    }

    // Unified Hi-Z visibility pass over recordedCalls (pure CPU, no device access)
    // recordedCalls already have world-space bboxes from recording/prepare phase.
    // One loop: test each call, set shouldRender, AND populate visibilityResults[recordMWIndex]
    // for the subsequent applyVisibilityAndFilterRecordMW() pass.
    {
        MGE_ZoneScopedN("Hi-Z Test recordedCalls");

        int visibleCount = 0, culledCount = 0, noBboxCount = 0;

        for (auto& call : recCalls) {
            // Scene 1 (particles) / Scene 2 (hands): always visible, skip Hi-Z test
            if (call.sceneNum > 0) {
                call.shouldRender = true;
                if (call.recordMWIndex >= 0 && call.recordMWIndex < (int)visibilityResults.size()) {
                    visibilityResults[call.recordMWIndex] = 1;
                }
                visibleCount++;
                continue;
            }

            if (!call.hasBoundingBox) {
                call.shouldRender = true;  // No bbox = conservative visible
                if (call.recordMWIndex >= 0 && call.recordMWIndex < (int)visibilityResults.size()) {
                    visibilityResults[call.recordMWIndex] = 1;
                }
                noBboxCount++;
                continue;
            }

            // Check if occluder (never cull occluders)
            MeshKey key{call.rs.vb, call.rs.ib, call.rs.fvf, call.rs.baseIndex,
                        call.rs.vertCount, call.rs.startIndex, call.rs.primCount};
            bool isOccluder = (rasterizedOccluderMeshes.count(key) > 0);

            if (isOccluder) {
                call.shouldRender = true;
                if (call.recordMWIndex >= 0 && call.recordMWIndex < (int)visibilityResults.size()) {
                    visibilityResults[call.recordMWIndex] = 1;
                }
                visibleCount++;
                continue;
            }

            // Hi-Z test using current matrices (all CPU, no device access)
            bool isVisible = softwareOcclusionCuller.testBoundingBox(
                call.bboxMin, call.bboxMax, currentView, currentProj);

            call.shouldRender = isVisible;

            // Propagate to visibilityResults for recordMW filtering
            if (call.recordMWIndex >= 0 && call.recordMWIndex < (int)visibilityResults.size()) {
                visibilityResults[call.recordMWIndex] = isVisible ? 1 : 0;
            }

            if (isVisible) visibleCount++; else culledCount++;
        }

        LOG::logline(">> Hi-Z: %d visible, %d culled, %d no bbox (of %d calls)",
                     visibleCount, culledCount, noBboxCount,
                     visibleCount + culledCount + noBboxCount);
    }
}

// applyVisibilityAndFilterRecordMW - Lightweight filter pass over recordMW using visibilityResults
// Runs after executeHiZCulling, before renderDepth
void FixedFunctionShader::applyVisibilityAndFilterRecordMW() {
    MGE_ZoneScopedN("Apply Visibility Filter to recordMW");

    // Non-HLSL mode or culling disabled: nothing to filter
    if (Configuration.PerPixelLightFlags != 2 || ImGuiManager::GetDisableHiZCulling()) {
        return;
    }

    // N-1: Use prep buffer's recordMW in HLSL mode, global static otherwise
    auto& renderBuf = getPrepBuffer();
    auto& activeRecordMW = (Configuration.PerPixelLightFlags == 2) ? renderBuf.recordMW : DistantLand::recordMW;

    int inputCount = (int)activeRecordMW.size();

    std::vector<RecordedMWState> filteredRecordMW;
    filteredRecordMW.reserve(activeRecordMW.size());

    for (size_t i = 0; i < activeRecordMW.size(); i++) {
        if (visibilityResults[i] == 1 || visibilityResults[i] == -1) {
            filteredRecordMW.push_back(std::move(activeRecordMW[i]));
        }
    }

    activeRecordMW = std::move(filteredRecordMW);

    int outputCount = (int)activeRecordMW.size();
    LOG::logline(">> Filter recordMW: %d in, %d out, %d culled",
                 inputCount, outputCount, inputCount - outputCount);
}

// Dirty tracking: simplified for single buffer (no previous frame comparison)
// Just marks all calls as dirty - can be optimized later if needed
// N-1: marks prep buffer's calls (previous frame being prepared)
void FixedFunctionShader::markAllCallsDirty() {
    MGE_ZoneScopedN("markAllCallsDirty");

    auto& fb = getPrepBuffer();
    // Scene 0 (world)
    for (auto& call : fb.recordedCalls) {
        call.dirtyFlags = DIRTY_ALL;
    }
    // Scene 1 (particles)
    for (auto& call : fb.recordedCallsScene1) {
        call.dirtyFlags = DIRTY_ALL;
    }
    // Scene 2 (hands)
    for (auto& call : fb.recordedCallsScene2) {
        call.dirtyFlags = DIRTY_ALL;
    }
}

static void extractLightStateLights(
    const LightState* lightrs,
    const D3DXVECTOR4& eyePos,
    std::vector<HLSLSceneLight>& sceneLights,
    std::unordered_map<int, size_t>& sceneLightIndexMap)
{
    if (!lightrs) return;

    // Only process ACTIVE lights — lightrs->lights accumulates all SetLight() calls
    // across the session and never removes entries, so inactive/stale lights persist.
    for (DWORD id : lightrs->active) {
        auto lightIt = lightrs->lights.find(id);
        if (lightIt == lightrs->lights.end()) continue;
        const auto& light = lightIt->second;

        // Skip directional lights (sun) — only point lights for scene lighting
        if (light.type != D3DLIGHT_POINT) continue;

        // Distance filter: reject lights far from camera (stale interior lights after cell change)
        float dx = light.position.x - eyePos.x;
        float dy = light.position.y - eyePos.y;
        float dz = light.position.z - eyePos.z;
        float distSq = dx*dx + dy*dy + dz*dz;
        const float MAX_LIGHT_DIST_SQ = 8192.0f * 8192.0f;
        if (distSq > MAX_LIGHT_DIST_SQ) continue;

        // Check if light already exists using O(1) hash map lookup
        auto mapIt = sceneLightIndexMap.find(id);

        if (mapIt == sceneLightIndexMap.end()) {
            // New light - add to scene
            size_t newIndex = sceneLights.size();
            HLSLSceneLight sl;
            sl.id = id;
            sl.position = light.position;
            sl.diffuse = light.diffuse;
            sl.falloff = light.falloff;  // (constant, linear, quadratic)
            sl.radius = DistantLand::computeLightRadius(sl.falloff.x, sl.falloff.y, sl.falloff.z);
            sl.isVisible = false;
            sl.lastSeenFrame = 0;
            sceneLights.push_back(sl);
            sceneLightIndexMap[id] = newIndex;  // Add to index map
        } else {
            // Update dynamic properties (color may pulse)
            size_t index = mapIt->second;
            sceneLights[index].position = light.position;
            sceneLights[index].diffuse = light.diffuse;
            sceneLights[index].falloff = light.falloff;
            sceneLights[index].radius = DistantLand::computeLightRadius(
                light.falloff.x, light.falloff.y, light.falloff.z);
        }
    }
}

// Extract lights once per frame instead of during every DrawIndexedPrimitive.
// Union all recorded draw-call light states; Morrowind changes active light slots
// per object, so the final frame state alone can miss lights used earlier.
static void extractFrameLights(
    const std::vector<FixedFunctionShader::HLSLRecordedCall>& calls,
    const LightState* fallbackLightState,
    const D3DXVECTOR4& eyePos,
    std::vector<HLSLSceneLight>& sceneLights,
    std::unordered_map<int, size_t>& sceneLightIndexMap)
{
    MGE_ZoneScopedN("extractFrameLights");
    sceneLights.clear();
    sceneLightIndexMap.clear();

    std::unordered_set<const LightState*> seenLightStates;
    for (const auto& call : calls) {
        const LightState* lightrs = call.lightrs.get();
        if (!lightrs || !seenLightStates.insert(lightrs).second) continue;
        extractLightStateLights(lightrs, eyePos, sceneLights, sceneLightIndexMap);
    }

    if (sceneLights.empty()) {
        extractLightStateLights(fallbackLightState, eyePos, sceneLights, sceneLightIndexMap);
    }
}

// Phase 2b: Prepare shader keys — runs after recording completes, before replay.
// BBox computation and occluder rasterization already done in executeHiZCulling (Phase 2a).
// Phase 2: prepares prepBuffer (N-1 frame's data)
void FixedFunctionShader::prepareRecordedCalls() {
    MGE_ZoneScopedN("prepareRecordedCalls");

    auto& fb = getPrepBuffer();
    auto& recCalls = fb.recordedCalls;

    // Extract lights once per frame by unioning the recorded per-call light states.
    // Sampling only fb.lastLightState is not enough: Morrowind changes active light
    // slots per object, so the final state can miss lights used earlier in the frame.
    // Derive eye pos from fb.view (frame N-1 view) instead of DistantLand::s_staging.eyePos,
    // which has already been advanced to the next frame by the time we run here.
    D3DXVECTOR4 fbEyePos;
    {
        D3DXMATRIX invView;
        D3DXMatrixInverse(&invView, nullptr, &fb.view);
        D3DXVECTOR4 origin(0.0f, 0.0f, 0.0f, 1.0f);
        D3DXVec4Transform(&fbEyePos, &origin, &invView);
    }
    extractFrameLights(recCalls, fb.lastLightState.get(), fbEyePos, fb.sceneLights, fb.sceneLightIndexMap);

    // Compute shader keys and assign render bins (with slow-frame timing)
    LARGE_INTEGER freqQPC, prepStartQPC, prepEndQPC;
    QueryPerformanceFrequency(&freqQPC);
    QueryPerformanceCounter(&prepStartQPC);
    {
        MGE_ZoneScopedN("Prepare: Shader Keys + Bins");

        // Scene 0: World geometry
        for (auto& call : recCalls) {
            call.sk = computeShaderKeyWithSuffixes(&call.rs, &call.frs, call.lightrs.get());
            call.prepared = true;

            // Classify into render bin
            // Terrain: large vertex count (typically 500+ verts), not skinned/grass/alpha
            const UINT TERRAIN_VERT_THRESHOLD = 500;
            if (call.sk.hasGrass)              call.bin = RenderBin::Grass;
            else if (call.sk.usesSkinning)     call.bin = RenderBin::Skinning;
            else if (call.rs.blendEnable)      call.bin = RenderBin::Blending;
            else if (call.rs.alphaTest)        call.bin = RenderBin::AlphaTested;
            else if (call.rs.vertCount >= TERRAIN_VERT_THRESHOLD)
                                               call.bin = RenderBin::Terrain;
            else                               call.bin = RenderBin::Opaque;
        }

        // Scene 1: Particles (always visible, no Hi-Z culling)
        for (auto& call : fb.recordedCallsScene1) {
            call.sk = computeShaderKeyWithSuffixes(&call.rs, &call.frs, call.lightrs.get());
            call.prepared = true;
            call.shouldRender = true;

            if (call.sk.hasGrass)              call.bin = RenderBin::Grass;
            else if (call.sk.usesSkinning)     call.bin = RenderBin::Skinning;
            else if (call.rs.blendEnable)      call.bin = RenderBin::Blending;
            else if (call.rs.alphaTest)        call.bin = RenderBin::AlphaTested;
            else                               call.bin = RenderBin::Opaque;
        }

        // Scene 2: Hands (always visible, no Hi-Z culling)
        for (auto& call : fb.recordedCallsScene2) {
            call.sk = computeShaderKeyWithSuffixes(&call.rs, &call.frs, call.lightrs.get());
            call.prepared = true;
            call.shouldRender = true;

            if (call.sk.hasGrass)              call.bin = RenderBin::Grass;
            else if (call.sk.usesSkinning)     call.bin = RenderBin::Skinning;
            else if (call.rs.blendEnable)      call.bin = RenderBin::Blending;
            else if (call.rs.alphaTest)        call.bin = RenderBin::AlphaTested;
            else                               call.bin = RenderBin::Opaque;
        }

        // Optional material sorting for state change reduction (Scene 0 only)
        if (ImGuiManager::GetMaterialSortEnabled()) {
            sortCallsByMaterial(recCalls);
        }
        // Note: buildInstanceBatches is called AFTER executeHiZCulling (in cpuprepthread.cpp)
        // so that shouldRender flags are already set
    }
    QueryPerformanceCounter(&prepEndQPC);
    lastPrepareMs = (prepEndQPC.QuadPart - prepStartQPC.QuadPart) * 1000.0f / freqQPC.QuadPart;
    if (lastPrepareMs > ImGuiManager::GetSlowFrameThreshold()) {
        LOG::logline("SLOW PREPARE: %.1fms for %d calls (Shader Keys + Bins)", lastPrepareMs, (int)recCalls.size());
    }

    // Mark all calls dirty (single buffer - no previous frame comparison)
    // Can be optimized later with same-frame dirty tracking
    markAllCallsDirty();
}

bool FixedFunctionShader::computeBoundingBox(const RenderedState* rs, D3DXVECTOR3& bboxMin, D3DXVECTOR3& bboxMax) {
    static bool debugBBox = false;
    static int debugCount = 0;
    static bool keyCheckedThisRecording = false;

    // Check for U key only once per recording session (avoid GetAsyncKeyState in per-mesh hotpath)
    if (isRecording && !keyCheckedThisRecording) {
        if (GetAsyncKeyState('U') & 0x8000) {
            static bool wasPressed = false;
            if (!wasPressed) {
                debugBBox = true;
                debugCount = 10;
                LOG::logline(">> BBox Debug: Enabled for next 10 calls");
                wasPressed = true;
            }
        } else {
            static bool wasPressed = false;
            wasPressed = false;
        }
        keyCheckedThisRecording = true;
    }

    // Reset flag when not recording
    if (!isRecording) {
        keyCheckedThisRecording = false;
    }

    if (!rs->vb) {
        if (debugBBox && debugCount > 0) {
            LOG::logline("!! computeBBox: no VB");
            debugCount--;
        }
        return false;
    }

    // Determine position offset in vertex structure (FVF formats always have position first if XYZ is present)
    bool hasPosition = (rs->fvf & D3DFVF_POSITION_MASK) != 0;
    if (!hasPosition) {
        if (debugBBox && debugCount > 0) {
            LOG::logline("!! computeBBox: no position in FVF (fvf=0x%X, mask=0x%X)", rs->fvf, D3DFVF_POSITION_MASK);
            debugCount--;
        }
        return false;
    }

    // Create mesh key for cache lookup
    MeshKey key;
    key.vb = rs->vb;
    key.ib = rs->ib;
    key.fvf = rs->fvf;
    key.baseIndex = rs->baseIndex;
    key.vertCount = rs->vertCount;
    key.startIndex = rs->startIndex;
    key.primCount = rs->primCount;

    // Check cache first (thread-safe access)
    ObjectSpaceBBox objBBox;
    bool cacheHit = false;
    {
        std::lock_guard<std::mutex> lock(bboxCacheMutex);
        auto it = bboxCache.find(key);
        if (it != bboxCache.end()) {
            objBBox = it->second;
            cacheHit = true;
        }
    }
    if (cacheHit) {
        // Cache hit! Transform cached object-space bbox to world-space

        // Transform 8 corners of object-space bbox by world matrix
        D3DXVECTOR3 corners[8] = {
            D3DXVECTOR3(objBBox.bboxMin.x, objBBox.bboxMin.y, objBBox.bboxMin.z),
            D3DXVECTOR3(objBBox.bboxMax.x, objBBox.bboxMin.y, objBBox.bboxMin.z),
            D3DXVECTOR3(objBBox.bboxMin.x, objBBox.bboxMax.y, objBBox.bboxMin.z),
            D3DXVECTOR3(objBBox.bboxMax.x, objBBox.bboxMax.y, objBBox.bboxMin.z),
            D3DXVECTOR3(objBBox.bboxMin.x, objBBox.bboxMin.y, objBBox.bboxMax.z),
            D3DXVECTOR3(objBBox.bboxMax.x, objBBox.bboxMin.y, objBBox.bboxMax.z),
            D3DXVECTOR3(objBBox.bboxMin.x, objBBox.bboxMax.y, objBBox.bboxMax.z),
            D3DXVECTOR3(objBBox.bboxMax.x, objBBox.bboxMax.y, objBBox.bboxMax.z),
        };

        bboxMin = D3DXVECTOR3(1e10f, 1e10f, 1e10f);
        bboxMax = D3DXVECTOR3(-1e10f, -1e10f, -1e10f);

        for (int i = 0; i < 8; i++) {
            D3DXVECTOR3 worldCorner;
            D3DXVec3TransformCoord(&worldCorner, &corners[i], &rs->worldTransforms[0]);

            bboxMin.x = std::min(bboxMin.x, worldCorner.x);
            bboxMin.y = std::min(bboxMin.y, worldCorner.y);
            bboxMin.z = std::min(bboxMin.z, worldCorner.z);
            bboxMax.x = std::max(bboxMax.x, worldCorner.x);
            bboxMax.y = std::max(bboxMax.y, worldCorner.y);
            bboxMax.z = std::max(bboxMax.z, worldCorner.z);
        }

        if (debugBBox && debugCount > 0) {
            LOG::logline(">> computeBBox: CACHE HIT! world bbox=[%.2f,%.2f,%.2f] to [%.2f,%.2f,%.2f]",
                bboxMin.x, bboxMin.y, bboxMin.z, bboxMax.x, bboxMax.y, bboxMax.z);
            debugCount--;
        }

        return true;
    }

    // Cache miss - compute object-space bbox and cache it
    if (debugBBox && debugCount > 0) {
        LOG::logline(">> computeBBox: CACHE MISS - computing from vertices, VB=%p, fvf=0x%X, stride=%d", rs->vb, rs->fvf, rs->vbStride);
    }

    D3DXVECTOR3 objBBoxMin(1e10f, 1e10f, 1e10f);
    D3DXVECTOR3 objBBoxMax(-1e10f, -1e10f, -1e10f);

    // Lock vertex buffer to read position data (non-blocking to avoid GPU stalls)
    void* pVertices = nullptr;
    HRESULT hr = rs->vb->Lock(rs->vbOffset, 0, &pVertices, D3DLOCK_READONLY | D3DLOCK_DONOTWAIT);
    if (FAILED(hr)) {
        if (debugBBox && debugCount > 0) {
            LOG::logline("!! computeBBox: VB Lock failed hr=0x%X", hr);
            debugCount--;
        }
        return false;
    }

    // Get vertex stride from FVF
    UINT stride = rs->vbStride;
    if (stride == 0) {
        if (debugBBox && debugCount > 0) {
            LOG::logline("!! computeBBox: stride == 0");
            debugCount--;
        }
        rs->vb->Unlock();
        return false;
    }

    // For indexed primitives, we need to check all referenced vertices
    // Lock index buffer to determine which vertices to check (non-blocking)
    void* pIndices = nullptr;
    if (rs->ib) {
        hr = rs->ib->Lock(0, 0, &pIndices, D3DLOCK_READONLY | D3DLOCK_DONOTWAIT);
        if (FAILED(hr)) {
            if (debugBBox && debugCount > 0) {
                LOG::logline("!! computeBBox: IB lock failed, hr=0x%X", hr);
                debugCount--;
            }
            rs->vb->Unlock();
            return false;
        }

        // Determine index format (16-bit or 32-bit)
        D3DINDEXBUFFER_DESC ibDesc;
        rs->ib->GetDesc(&ibDesc);
        bool is16Bit = (ibDesc.Format == D3DFMT_INDEX16);

        // Process indexed vertices
        UINT indexCount = 0;
        switch (rs->primType) {
            case D3DPT_TRIANGLELIST: indexCount = rs->primCount * 3; break;
            case D3DPT_TRIANGLESTRIP: indexCount = rs->primCount + 2; break;
            case D3DPT_TRIANGLEFAN: indexCount = rs->primCount + 2; break;
            default:
                if (debugBBox && debugCount > 0) {
                    LOG::logline("!! computeBBox: unsupported primType=%d", rs->primType);
                    debugCount--;
                }
                rs->ib->Unlock();
                rs->vb->Unlock();
                return false;
        }

        for (UINT i = 0; i < indexCount; i++) {
            UINT vertexIndex;
            if (is16Bit) {
                vertexIndex = ((WORD*)pIndices)[rs->startIndex + i] + rs->baseIndex;
            } else {
                vertexIndex = ((DWORD*)pIndices)[rs->startIndex + i] + rs->baseIndex;
            }

            // Get vertex position (positions are always at offset 0 in FVF) - object space
            BYTE* vertexData = ((BYTE*)pVertices) + vertexIndex * stride;
            D3DXVECTOR3* pos = (D3DXVECTOR3*)vertexData;

            objBBoxMin.x = std::min(objBBoxMin.x, pos->x);
            objBBoxMin.y = std::min(objBBoxMin.y, pos->y);
            objBBoxMin.z = std::min(objBBoxMin.z, pos->z);
            objBBoxMax.x = std::max(objBBoxMax.x, pos->x);
            objBBoxMax.y = std::max(objBBoxMax.y, pos->y);
            objBBoxMax.z = std::max(objBBoxMax.z, pos->z);
        }

        rs->ib->Unlock();
    } else {
        // Non-indexed primitives - check sequential vertices
        UINT vertexCount = rs->vertCount;
        for (UINT i = 0; i < vertexCount; i++) {
            BYTE* vertexData = ((BYTE*)pVertices) + (rs->baseIndex + i) * stride;
            D3DXVECTOR3* pos = (D3DXVECTOR3*)vertexData;

            objBBoxMin.x = std::min(objBBoxMin.x, pos->x);
            objBBoxMin.y = std::min(objBBoxMin.y, pos->y);
            objBBoxMin.z = std::min(objBBoxMin.z, pos->z);
            objBBoxMax.x = std::max(objBBoxMax.x, pos->x);
            objBBoxMax.y = std::max(objBBoxMax.y, pos->y);
            objBBoxMax.z = std::max(objBBoxMax.z, pos->z);
        }
    }

    rs->vb->Unlock();

    // Validate object-space bounding box
    bool valid = (objBBoxMin.x <= objBBoxMax.x && objBBoxMin.y <= objBBoxMax.y && objBBoxMin.z <= objBBoxMax.z);
    if (!valid) {
        if (debugBBox && debugCount > 0) {
            LOG::logline("!! computeBBox: INVALID object bbox! min=[%.2f,%.2f,%.2f] max=[%.2f,%.2f,%.2f]",
                objBBoxMin.x, objBBoxMin.y, objBBoxMin.z, objBBoxMax.x, objBBoxMax.y, objBBoxMax.z);
            debugCount--;
        }
        return false;
    }

    // Cache the object-space bbox (thread-safe write)
    ObjectSpaceBBox cachedBBox;
    cachedBBox.bboxMin = objBBoxMin;
    cachedBBox.bboxMax = objBBoxMax;
    {
        std::lock_guard<std::mutex> lock(bboxCacheMutex);
        bboxCache[key] = cachedBBox;
    }

    // Transform 8 corners to world space
    D3DXVECTOR3 corners[8] = {
        D3DXVECTOR3(objBBoxMin.x, objBBoxMin.y, objBBoxMin.z),
        D3DXVECTOR3(objBBoxMax.x, objBBoxMin.y, objBBoxMin.z),
        D3DXVECTOR3(objBBoxMin.x, objBBoxMax.y, objBBoxMin.z),
        D3DXVECTOR3(objBBoxMax.x, objBBoxMax.y, objBBoxMin.z),
        D3DXVECTOR3(objBBoxMin.x, objBBoxMin.y, objBBoxMax.z),
        D3DXVECTOR3(objBBoxMax.x, objBBoxMin.y, objBBoxMax.z),
        D3DXVECTOR3(objBBoxMin.x, objBBoxMax.y, objBBoxMax.z),
        D3DXVECTOR3(objBBoxMax.x, objBBoxMax.y, objBBoxMax.z),
    };

    bboxMin = D3DXVECTOR3(1e10f, 1e10f, 1e10f);
    bboxMax = D3DXVECTOR3(-1e10f, -1e10f, -1e10f);

    for (int i = 0; i < 8; i++) {
        D3DXVECTOR3 worldCorner;
        D3DXVec3TransformCoord(&worldCorner, &corners[i], &rs->worldTransforms[0]);

        bboxMin.x = std::min(bboxMin.x, worldCorner.x);
        bboxMin.y = std::min(bboxMin.y, worldCorner.y);
        bboxMin.z = std::min(bboxMin.z, worldCorner.z);
        bboxMax.x = std::max(bboxMax.x, worldCorner.x);
        bboxMax.y = std::max(bboxMax.y, worldCorner.y);
        bboxMax.z = std::max(bboxMax.z, worldCorner.z);
    }

    if (debugBBox && debugCount > 0) {
        LOG::logline(">> computeBBox: CACHED! obj=[%.2f,%.2f,%.2f] to [%.2f,%.2f,%.2f], world=[%.2f,%.2f,%.2f] to [%.2f,%.2f,%.2f]",
            objBBoxMin.x, objBBoxMin.y, objBBoxMin.z, objBBoxMax.x, objBBoxMax.y, objBBoxMax.z,
            bboxMin.x, bboxMin.y, bboxMin.z, bboxMax.x, bboxMax.y, bboxMax.z);
        debugCount--;
    }

    return true;
}

// Helper to create MeshKey from recorded call
static MeshKey makeMeshKey(const FixedFunctionShader::HLSLRecordedCall& call) {
    MeshKey key;
    key.vb = call.rs.vb;
    key.ib = call.rs.ib;
    key.fvf = call.rs.fvf;
    key.baseIndex = call.rs.baseIndex;
    key.vertCount = call.rs.vertCount;
    key.startIndex = call.rs.startIndex;
    key.primCount = call.rs.primCount;
    return key;
}

// Helper to fill StatelessDrawData from a recorded call
static void fillDrawData(StatelessDrawData& data, const FixedFunctionShader::HLSLRecordedCall& call,
                         bool highlightBatches, bool isVisible) {
    const D3DXMATRIX& wv = call.rs.worldViewTransforms[0];
    data.world0[0] = wv._11; data.world0[1] = wv._21; data.world0[2] = wv._31; data.world0[3] = wv._41;
    data.world1[0] = wv._12; data.world1[1] = wv._22; data.world1[2] = wv._32; data.world1[3] = wv._42;
    data.world2[0] = wv._13; data.world2[1] = wv._23; data.world2[2] = wv._33; data.world2[3] = wv._43;

    // Material (tint green if highlighting)
    float tintR = highlightBatches ? 0.3f : 1.0f;
    float tintG = highlightBatches ? 1.0f : 1.0f;
    float tintB = highlightBatches ? 0.3f : 1.0f;
    data.diffuse[0] = call.frs.material.diffuse.r * tintR;
    data.diffuse[1] = call.frs.material.diffuse.g * tintG;
    data.diffuse[2] = call.frs.material.diffuse.b * tintB;
    data.diffuse[3] = call.frs.material.diffuse.a;

    data.ambient[0] = call.frs.material.ambient.r;
    data.ambient[1] = call.frs.material.ambient.g;
    data.ambient[2] = call.frs.material.ambient.b;
    data.ambient[3] = call.frs.material.ambient.a;

    // Store normres (parameter texture resolution) and visibility flag
    float normresX = 512.0f, normresY = 512.0f;  // Default fallback
    if (call.sk.hasParamH && call.rs.texture) {
        D3DSURFACE_DESC desc;
        if (SUCCEEDED(call.rs.texture->GetLevelDesc(0, &desc))) {
            normresX = (float)desc.Width;
            normresY = (float)desc.Height;
        }
    }
    data.normres[0] = normresX;
    data.normres[1] = normresY;
    data.normres[2] = 0.0f;
    data.normres[3] = isVisible ? 1.0f : 0.0f;  // Visibility flag

    // Zero out reserved texels
    memset(data.reserved1, 0, sizeof(data.reserved1));
    memset(data.reserved2, 0, sizeof(data.reserved2));
    memset(data.reserved3, 0, sizeof(data.reserved3));
    memset(data.reserved4, 0, sizeof(data.reserved4));
    memset(data.reserved5, 0, sizeof(data.reserved5));
    memset(data.reserved6, 0, sizeof(data.reserved6));
    memset(data.reserved7, 0, sizeof(data.reserved7));

    data.emissive[0] = call.frs.material.emissive.r;
    data.emissive[1] = call.frs.material.emissive.g;
    data.emissive[2] = call.frs.material.emissive.b;
    data.emissive[3] = (float)call.rs.alphaRef / 255.0f;

    data.lightParams[0] = 0.0f;  // Will be filled with lightCount later
    data.lightParams[1] = 0.0f;  // Will be filled with texelSize later
    data.lightParams[2] = 0.0f;  // Will be filled with texelOffset later
    data.lightParams[3] = (float)call.sk.vertexMaterial;  // Material mode (1/2/3)

    // Full 4th column of worldView for robust w computation
    data.flags[0] = wv._14;
    data.flags[1] = wv._24;
    data.flags[2] = wv._34;
    data.flags[3] = wv._44;
}

// Check if a call is batchable (Opaque/Terrain, no skinning/grass/vertexBlending)
static bool isBatchableCall(const FixedFunctionShader::HLSLRecordedCall& call) {
    // Merged batches rebuild indexed triangle-list geometry into a new VB/IB.
    if (call.rs.primType != D3DPT_TRIANGLELIST || !call.rs.vb || !call.rs.ib)
        return false;

    // Only batch Opaque/Terrain bins (no Skinning, no Grass, no Blending)
    if (call.bin != RenderBin::Opaque && call.bin != RenderBin::Terrain)
        return false;

    // Skip geometry with vertex blending
    if (call.rs.vertexBlendState != 0)
        return false;

    // Skip geometry that can't be merged
    if (call.sk.usesSkinning || call.sk.hasGrass)
        return false;

    return true;
}

// Create MergedBatchKey from a recorded call
static MergedBatchKey makeMergedBatchKey(const FixedFunctionShader::HLSLRecordedCall& call) {
    MergedBatchKey mkey = {};
    mkey.texture = call.rs.texture;
    mkey.blendState = (uint16_t)((call.expectedState.captured ? call.expectedState.alphaBlendEnable : 0) |
                     ((call.expectedState.captured ? call.expectedState.srcBlend : 0) << 4) |
                     ((call.expectedState.captured ? call.expectedState.destBlend : 0) << 8));
    mkey.zState = (uint8_t)((call.expectedState.captured ? call.expectedState.zEnable : 1) |
                 ((call.expectedState.captured ? call.expectedState.zWriteEnable : 1) << 2));
    mkey.cullMode = call.expectedState.captured ? (uint8_t)call.expectedState.cullMode : D3DCULL_CW;
    mkey.useLighting = call.rs.useLighting ? 1 : 0;
    mkey.bin = (uint8_t)call.bin;
    mkey.fvf = call.rs.fvf;
    mkey.stride = call.rs.vbStride;
    return mkey;
}

// Build merged batches: groups different geometries sharing same texture into mega-draws
void FixedFunctionShader::buildStatelessBatches(FrameBuffer& fb) {
    MGE_ZoneScopedN("buildMergedBatches");

    if (fb.cachedMergedVB) {
        fb.cachedMergedVB->Release();
        fb.cachedMergedVB = nullptr;
    }
    if (fb.cachedMergedIB) {
        fb.cachedMergedIB->Release();
        fb.cachedMergedIB = nullptr;
    }

    fb.drawDataStaging.clear();
    fb.singletonCallIndices.clear();
    fb.mergedBatches.clear();
    fb.cellBatchCacheKey = nullptr;
    fb.cellBatchLayoutHash = 0;
    fb.useCachedMergedVB = false;
    fb.cachedDrawInfos.clear();

    if (!ImGuiManager::GetEnableStatelessBatch()) return;

    auto& calls = fb.recordedCalls;
    if (calls.empty()) return;

    struct PendingMergedGroup {
        MergedBatchKey key = {};
        std::vector<size_t> callIndices;
        UINT totalVertices = 0;
        UINT totalIndices = 0;
    };

    // Group all batchable Scene 0 calls deterministically by first-seen MergedBatchKey.
    std::unordered_map<MergedBatchKey, size_t, MergedBatchKey::Hasher> mergeGroupLookup;
    std::vector<PendingMergedGroup> mergeGroups;
    bool highlightBatches = ImGuiManager::GetHighlightStatelessBatch();

    // Mirror/capture diagnostic counters (per frame)
    int diagBatchable = 0;
    int diagUncaptured = 0;
    int diagMirrored = 0;
    int diagMirroredCW = 0;
    int diagMirroredCCW = 0;
    int diagMirroredNone = 0;
    int diagMirroredUncaptured = 0;

    for (size_t i = 0; i < calls.size(); i++) {
        const auto& call = calls[i];

        if (call.sceneNum != 0 || !isBatchableCall(call))
            continue;

        MergedBatchKey mkey = makeMergedBatchKey(call);
        auto insertResult = mergeGroupLookup.emplace(mkey, mergeGroups.size());
        if (insertResult.second) {
            mergeGroups.push_back(PendingMergedGroup());
            mergeGroups.back().key = mkey;
        }

        PendingMergedGroup& group = mergeGroups[insertResult.first->second];
        group.callIndices.push_back(i);
        group.totalVertices += call.rs.vertCount;
        group.totalIndices += call.rs.primCount * 3;

        // Diagnostic: detect mirrored draws and how they're getting keyed
        diagBatchable++;
        if (!call.expectedState.captured) diagUncaptured++;
        const D3DXMATRIX& w = call.rs.worldTransforms[0];
        float det3 = w._11 * (w._22 * w._33 - w._23 * w._32)
                   - w._12 * (w._21 * w._33 - w._23 * w._31)
                   + w._13 * (w._21 * w._32 - w._22 * w._31);
        if (det3 < 0.0f) {
            diagMirrored++;
            if (!call.expectedState.captured) {
                diagMirroredUncaptured++;
            } else {
                switch (mkey.cullMode) {
                    case D3DCULL_CW:   diagMirroredCW++; break;
                    case D3DCULL_CCW:  diagMirroredCCW++; break;
                    case D3DCULL_NONE: diagMirroredNone++; break;
                    default: break;
                }
            }
        }
    }

    // Log diagnostic once per second to avoid spam
    if (diagMirrored > 0 || diagUncaptured > 0) {
        static DWORD lastDiagTick = 0;
        DWORD nowTick = GetTickCount();
        if (nowTick - lastDiagTick >= 1000) {
            lastDiagTick = nowTick;
            LOG::logline("StatelessBatch diag: batchable=%d uncap=%d mirrored=%d (mirCW=%d mirCCW=%d mirNone=%d mirUncap=%d)",
                         diagBatchable, diagUncaptured, diagMirrored,
                         diagMirroredCW, diagMirroredCCW, diagMirroredNone, diagMirroredUncaptured);
        }
    }

    std::sort(mergeGroups.begin(), mergeGroups.end(),
        [](const PendingMergedGroup& lhs, const PendingMergedGroup& rhs) {
            return lessMergedBatchKey(lhs.key, rhs.key);
        });

    std::vector<CachedBatchTemplate> batchTemplates;
    std::vector<CachedMergedCallLayout> mergedLayout;

    // Build MergedBatch entries for groups with 2+ calls. Hidden calls remain in the batch
    // with visibility encoded in drawDataStaging so geometry can be reused across frames.
    UINT drawDataOffset = 0;
    for (auto& group : mergeGroups) {
        std::vector<std::pair<CachedMergedCallLayout, size_t>> sortedCalls;
        sortedCalls.reserve(group.callIndices.size());
        for (size_t callIdx : group.callIndices) {
            sortedCalls.push_back(std::make_pair(makeCachedMergedCallLayout(calls[callIdx], group.key), callIdx));
        }
        std::sort(sortedCalls.begin(), sortedCalls.end(),
            [](const std::pair<CachedMergedCallLayout, size_t>& lhs,
               const std::pair<CachedMergedCallLayout, size_t>& rhs) {
                return lessCachedMergedCallLayout(lhs.first, rhs.first);
            });

        group.callIndices.clear();
        group.callIndices.reserve(sortedCalls.size());
        for (const auto& sortedCall : sortedCalls) {
            group.callIndices.push_back(sortedCall.second);
        }

        if (group.callIndices.size() >= 2) {
            MergedBatch mbatch;
            mbatch.key = group.key;
            mbatch.callIndices = group.callIndices;
            mbatch.drawDataOffset = drawDataOffset;
            mbatch.totalVertices = group.totalVertices;
            mbatch.totalIndices = group.totalIndices;

            for (size_t i = 0; i < mbatch.callIndices.size(); ++i) {
                size_t callIdx = mbatch.callIndices[i];
                const auto& call = calls[callIdx];
                StatelessDrawData data;
                fillDrawData(data, call, highlightBatches, call.shouldRender);
                fb.drawDataStaging.push_back(data);
                mergedLayout.push_back(sortedCalls[i].first);
                drawDataOffset++;
            }

            fb.mergedBatches.push_back(std::move(mbatch));
            batchTemplates.push_back(makeCachedBatchTemplate(fb.mergedBatches.back()));
        } else if (group.callIndices.size() == 1 && highlightBatches) {
            // Track singletons for highlighting
            fb.singletonCallIndices.insert(group.callIndices[0]);
        }
    }

    void* currentCell = MWBridge::get() ? MWBridge::get()->getPlayerCell() : nullptr;
    size_t layoutHash = computeCellBatchLayoutHash(batchTemplates, mergedLayout);
    fb.cellBatchCacheKey = currentCell;
    fb.cellBatchLayoutHash = layoutHash;
    if (currentCell && !fb.mergedBatches.empty()) {
        AcquireSRWLockExclusive(&s_cellBatchCacheLock);

        CellBatchCacheKey cacheKey = { currentCell, layoutHash };
        auto it = s_cellBatchCaches.find(cacheKey);
        bool layoutMatch = false;
        if (it != s_cellBatchCaches.end()) {
            layoutMatch = it->second.valid &&
                          it->second.layoutHash == layoutHash &&
                          it->second.batches == batchTemplates &&
                          it->second.mergedLayout == mergedLayout;
        }

        if (layoutMatch) {
            CellBatchCache& cache = it->second;
            cache.lastUsedSerial = ++s_cellBatchCacheUseSerial;

            if (cache.hasGeometry()) {
                cache.mergedVB->AddRef();
                cache.mergedIB->AddRef();
                fb.cachedMergedVB = cache.mergedVB;
                fb.cachedMergedIB = cache.mergedIB;
                fb.cachedDrawInfos = cache.drawInfos;
                fb.useCachedMergedVB = (fb.cachedDrawInfos.size() == fb.mergedBatches.size());

                if (!fb.useCachedMergedVB) {
                    fb.cachedMergedVB->Release();
                    fb.cachedMergedIB->Release();
                    fb.cachedMergedVB = nullptr;
                    fb.cachedMergedIB = nullptr;
                    fb.cachedDrawInfos.clear();
                }
            }
        } else {
            CellBatchCache& cache = s_cellBatchCaches[cacheKey];
            cache.release();
            cache.cellPtr = currentCell;
            cache.layoutHash = layoutHash;
            cache.valid = true;
            cache.lastUsedSerial = ++s_cellBatchCacheUseSerial;
            cache.batches = batchTemplates;
            cache.mergedLayout = mergedLayout;
            evictLRUCellBatchCacheLocked(cacheKey);
        }

        ReleaseSRWLockExclusive(&s_cellBatchCacheLock);
    }

    // Log batching stats
    if (!fb.mergedBatches.empty()) {
        int totalMerged = 0;
        for (const auto& mb : fb.mergedBatches) {
            totalMerged += (int)mb.callIndices.size();
        }
        LOG::logline("MergedBatch: %d batches, %d draws merged -> %d GPU calls",
                     (int)fb.mergedBatches.size(), totalMerged, (int)fb.mergedBatches.size());
    }

    // Detailed logging on button press
    bool dumpDetail = ImGuiManager::GetAndClearDumpStatelessBatch();
    if (dumpDetail) {
        LOG::logline("=== MergedBatch Breakdown ===");
        LOG::logline("Total recorded calls: %d", (int)calls.size());

        // Count calls by category
        int batchedCount = 0;
        for (const auto& mb : fb.mergedBatches) {
            batchedCount += (int)mb.callIndices.size();
        }

        int batchIdx = 0;
        for (const auto& mb : fb.mergedBatches) {
            LOG::logline("Batch %d: %d draws, tex=%p, verts=%d, fvf=0x%X, stride=%d",
                batchIdx++, (int)mb.callIndices.size(), mb.key.texture,
                mb.totalVertices, mb.key.fvf, mb.key.stride);
        }

        // Analyze ALL calls to show why they weren't batched
        int notOpaqueTerrain = 0, culled = 0, skinning = 0, grass = 0, vertBlend = 0;
        std::unordered_map<IDirect3DTexture9*, std::vector<size_t>> unbatchedByTexture;

        // Track which calls are in batches
        std::unordered_set<size_t> batchedIndices;
        for (const auto& mb : fb.mergedBatches) {
            for (size_t idx : mb.callIndices) {
                batchedIndices.insert(idx);
            }
        }

        for (size_t i = 0; i < calls.size(); i++) {
            if (batchedIndices.count(i)) continue;  // Already batched

            const auto& call = calls[i];

            // Categorize why not batched
            if (call.bin != RenderBin::Opaque && call.bin != RenderBin::Terrain) {
                notOpaqueTerrain++;
                continue;
            }
            if (!call.shouldRender) { culled++; continue; }
            if (call.rs.vertexBlendState != 0) { vertBlend++; continue; }
            if (call.sk.usesSkinning) { skinning++; continue; }
            if (call.sk.hasGrass) { grass++; continue; }

            // This is an unbatched Opaque/Terrain call - track by texture
            unbatchedByTexture[call.rs.texture].push_back(i);
        }

        LOG::logline("Batched: %d, Singletons: %d", batchedCount, (int)fb.singletonCallIndices.size());
        LOG::logline("Excluded: notOpaque/Terrain=%d culled=%d vertBlend=%d skinning=%d grass=%d",
            notOpaqueTerrain, culled, vertBlend, skinning, grass);

        // Show unbatched opaque/terrain draws grouped by texture
        int unbatchedOpaqueCount = 0;
        for (const auto& ut : unbatchedByTexture) {
            unbatchedOpaqueCount += (int)ut.second.size();
        }
        LOG::logline("Unbatched Opaque/Terrain: %d draws across %d textures",
            unbatchedOpaqueCount, (int)unbatchedByTexture.size());

        // Log textures with multiple unbatched draws (near-misses)
        int logged = 0;
        for (const auto& ut : unbatchedByTexture) {
            if (ut.second.size() >= 2 && logged < 10) {
                LOG::logline("Tex %p: %d unbatched draws (could batch if same VB/state):",
                    ut.first, (int)ut.second.size());
                for (size_t j = 0; j < std::min(ut.second.size(), (size_t)5); j++) {
                    size_t idx = ut.second[j];
                    const auto& call = calls[idx];
                    LOG::logline("  [%d] vb=%p fvf=0x%X stride=%d lit=%d vmat=%d",
                        (int)idx, call.rs.vb, call.rs.fvf, call.rs.vbStride,
                        call.rs.useLighting ? 1 : 0, call.sk.vertexMaterial);
                }
                logged++;
            }
        }
    }
}
