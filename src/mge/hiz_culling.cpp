// Hi-Z occlusion culling implementation
// Split from ffeshader.cpp for organization - these are still FixedFunctionShader members
#include "ffeshader.h"
#include "configuration.h"
#include "support/log.h"
#include "distantland.h"
#include "imgui_manager.h"
#include "mge_tracy.h"
#include "texture_suffix.h"

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

    // Resize visibility results for this frame (indexed by draw order)
    // N-1: HLSL mode uses prep buffer (previous frame's data being prepared), legacy mode uses global static
    auto& renderBuf = getPrepBuffer();
    const auto& activeRecordMW = (Configuration.PerPixelLightFlags == 2) ? renderBuf.recordMW : DistantLand::recordMW;
    visibilityResults.assign(activeRecordMW.size(), -1);  // -1 = not yet tested

    // Phase 2a: Compute deferred bboxes and rasterize occluders from recorded HLSL calls
    // This must happen before Hi-Z build so the depth pass benefits from culling
    auto& recCalls = (Configuration.PerPixelLightFlags == 2) ? renderBuf.recordedCalls : currentRecordedCalls();
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

    // Only do Hi-Z culling in HLSL mode (PerPixelLightFlags == 2)
    // Non-HLSL modes don't have bboxCache populated, so skip culling
    if (Configuration.PerPixelLightFlags != 2) {
        LOG::logline(">> Depth: %d objects (no culling - non-HLSL mode)",
                     (int)DistantLand::recordMW.size());
        return;
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

// Phase 2b: Prepare shader keys — runs after recording completes, before replay.
// BBox computation and occluder rasterization already done in executeHiZCulling (Phase 2a).
// Phase 2: prepares prepBuffer (N-1 frame's data)
void FixedFunctionShader::prepareRecordedCalls() {
    MGE_ZoneScopedN("prepareRecordedCalls");

    auto& fb = getPrepBuffer();
    auto& recCalls = fb.recordedCalls;

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

//------------------------------------------------------------
// GPU Instancing Implementation

bool FixedFunctionShader::initInstancing() {
    if (vbFFEInstances) return true;  // Already initialized

    auto* device = DistantLand::device;
    if (!device) return false;

    HRESULT hr = device->CreateVertexBuffer(
        MaxFFEInstances * FFEInstStride,
        D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
        0,  // No FVF, using vertex declaration
        D3DPOOL_DEFAULT,
        &vbFFEInstances,
        nullptr
    );

    if (FAILED(hr)) {
        LOG::logline("!! Failed to create FFE instance buffer");
        return false;
    }

    LOG::logline("-- FFE instancing initialized: %d max instances, %d bytes",
                 MaxFFEInstances, MaxFFEInstances * FFEInstStride);
    return true;
}

void FixedFunctionShader::releaseInstancing() {
    if (vbFFEInstances) {
        vbFFEInstances->Release();
        vbFFEInstances = nullptr;
    }

    // Release cached vertex declarations
    for (auto& kv : fvfDeclCache) {
        if (kv.second) {
            kv.second->Release();
        }
    }
    fvfDeclCache.clear();

    for (auto& kv : statelessDeclCache) {
        if (kv.second) {
            kv.second->Release();
        }
    }
    statelessDeclCache.clear();
}

void FixedFunctionShader::buildInstanceBatches(FrameBuffer& fb) {
    MGE_ZoneScopedN("buildInstanceBatches");

    fb.instanceBatches.clear();

    if (!ImGuiManager::GetInstancingEnabled()) return;

    auto& calls = fb.recordedCalls;
    if (calls.empty()) return;

    // Group calls by InstanceKey (geometry + material)
    std::unordered_map<InstanceKey, std::vector<size_t>, InstanceKey::Hasher> instanceGroups;

    for (size_t i = 0; i < calls.size(); i++) {
        const auto& call = calls[i];

        // Only batch Opaque/Terrain bins (no Skinning - different bone palettes, no Grass - not HLSL)
        if (call.bin != RenderBin::Opaque && call.bin != RenderBin::Terrain)
            continue;

        // Skip if already culled
        if (!call.shouldRender) continue;

        InstanceKey key;
        // Geometry identity
        key.vb = call.rs.vb;
        key.ib = call.rs.ib;
        key.fvf = call.rs.fvf;
        key.baseIndex = call.rs.baseIndex;
        key.vertCount = call.rs.vertCount;
        key.startIndex = call.rs.startIndex;
        key.primCount = call.rs.primCount;
        // Material identity
        key.texture = call.rs.texture;
        key.blendState = (uint16_t)((call.expectedState.captured ? call.expectedState.alphaBlendEnable : 0) |
                         ((call.expectedState.captured ? call.expectedState.srcBlend : 0) << 4) |
                         ((call.expectedState.captured ? call.expectedState.destBlend : 0) << 8));
        key.alphaState = (uint16_t)(call.expectedState.captured ? call.expectedState.alphaTestEnable : 0);
        key.zState = (uint8_t)((call.expectedState.captured ? call.expectedState.zEnable : 1) |
                     ((call.expectedState.captured ? call.expectedState.zWriteEnable : 1) << 2));
        key.cullMode = call.expectedState.captured ? (uint8_t)call.expectedState.cullMode : D3DCULL_CW;

        instanceGroups[key].push_back(i);
    }

    // Build batches from groups with 2+ instances
    for (auto& kv : instanceGroups) {
        if (kv.second.size() >= 2) {
            InstanceBatch batch;
            batch.key = kv.first;
            batch.callIndices = std::move(kv.second);
            batch.instanceBufferOffset = 0;  // Will be set when filling instance buffer
            fb.instanceBatches.push_back(std::move(batch));
        }
    }

    // Log batching stats
    int totalBatched = 0;
    for (const auto& batch : fb.instanceBatches) {
        totalBatched += (int)batch.callIndices.size();
    }

    if (!fb.instanceBatches.empty()) {
        LOG::logline("Instancing: %d batches, %d batched draws (of %d total)",
                     (int)fb.instanceBatches.size(), totalBatched, (int)calls.size());
    }
}

// Build stateless batches: groups by geometry + render state (material data in texture)
void FixedFunctionShader::buildStatelessBatches(FrameBuffer& fb) {
    MGE_ZoneScopedN("buildStatelessBatches");

    fb.statelessBatches.clear();
    fb.drawDataStaging.clear();
    fb.singletonCallIndices.clear();
    fb.mergedBatches.clear();

    if (!ImGuiManager::GetEnableStatelessBatch()) return;

    auto& calls = fb.recordedCalls;
    if (calls.empty()) return;

    // Group calls by StatelessBatchKey (geometry + render state, NOT material)
    std::unordered_map<StatelessBatchKey, std::vector<size_t>, StatelessBatchKey::Hasher> batchGroups;

    for (size_t i = 0; i < calls.size(); i++) {
        const auto& call = calls[i];

        // Only batch Opaque/Terrain bins (no Skinning - bone palettes, no Grass)
        if (call.bin != RenderBin::Opaque && call.bin != RenderBin::Terrain)
            continue;

        // Also skip any geometry with vertex blending (even if classified as Opaque)
        // This catches animated parts that may not be flagged as skinning
        if (call.rs.vertexBlendState != 0)
            continue;

        // Skip if already culled
        if (!call.shouldRender) continue;

        StatelessBatchKey key = {};  // Zero-initialize all fields

        // Compute geometry hash from VB content (identifies identical meshes across different VB allocations)
        size_t geoHash = computeGeometryHash(call.rs.vb, call.rs.vbOffset, call.rs.vbStride,
                                              call.rs.vertCount, call.rs.fvf);

        // Geometry identity - use content hash for batching, keep VB for actual draw
        key.geometryHash = geoHash;
        key.vb = call.rs.vb;  // Stored for draw, but hash used for matching
        key.ib = call.rs.ib;
        key.baseIndex = call.rs.baseIndex;
        key.vertCount = call.rs.vertCount;
        key.startIndex = call.rs.startIndex;
        key.primCount = call.rs.primCount;

        // Render state (NOT material - that goes in texture)
        key.texture = call.rs.texture;
        key.blendState = (uint16_t)((call.expectedState.captured ? call.expectedState.alphaBlendEnable : 0) |
                         ((call.expectedState.captured ? call.expectedState.srcBlend : 0) << 4) |
                         ((call.expectedState.captured ? call.expectedState.destBlend : 0) << 8));
        key.zState = (uint8_t)((call.expectedState.captured ? call.expectedState.zEnable : 1) |
                     ((call.expectedState.captured ? call.expectedState.zWriteEnable : 1) << 2));
        key.cullMode = call.expectedState.captured ? (uint8_t)call.expectedState.cullMode : D3DCULL_CW;
        key.useLighting = call.rs.useLighting ? 1 : 0;
        key.vertexMaterial = (uint8_t)call.sk.vertexMaterial;
        key.vertexColour = (uint8_t)call.sk.vertexColour;

        batchGroups[key].push_back(i);
    }

    // Build batches from groups with 2+ instances
    UINT drawDataOffset = 0;
    bool highlightBatches = ImGuiManager::GetHighlightStatelessBatch();

    for (auto& kv : batchGroups) {
        if (kv.second.size() >= 2) {
            StatelessBatch batch;
            batch.key = kv.first;
            batch.callIndices = std::move(kv.second);
            batch.drawDataOffset = drawDataOffset;

            // Fill draw data for each instance in this batch
            for (size_t callIdx : batch.callIndices) {
                const auto& call = calls[callIdx];
                StatelessDrawData data;

                // WorldView matrix columns (pre-combined for Z precision matching regular path)
                const D3DXMATRIX& wv = call.rs.worldViewTransforms[0];
                data.world0[0] = wv._11; data.world0[1] = wv._21; data.world0[2] = wv._31; data.world0[3] = wv._41;
                data.world1[0] = wv._12; data.world1[1] = wv._22; data.world1[2] = wv._32; data.world1[3] = wv._42;
                data.world2[0] = wv._13; data.world2[1] = wv._23; data.world2[2] = wv._33; data.world2[3] = wv._43;

                // Material diffuse (tint green if highlight mode)
                float tintR = highlightBatches ? 0.3f : 1.0f;
                float tintG = highlightBatches ? 1.0f : 1.0f;
                float tintB = highlightBatches ? 0.3f : 1.0f;
                data.diffuse[0] = call.frs.material.diffuse.r * tintR;
                data.diffuse[1] = call.frs.material.diffuse.g * tintG;
                data.diffuse[2] = call.frs.material.diffuse.b * tintB;
                data.diffuse[3] = call.frs.material.diffuse.a;

                // Material ambient
                data.ambient[0] = call.frs.material.ambient.r;
                data.ambient[1] = call.frs.material.ambient.g;
                data.ambient[2] = call.frs.material.ambient.b;
                data.ambient[3] = call.frs.material.ambient.a;

                // Material emissive + alphaRef
                data.emissive[0] = call.frs.material.emissive.r;
                data.emissive[1] = call.frs.material.emissive.g;
                data.emissive[2] = call.frs.material.emissive.b;
                data.emissive[3] = (float)call.rs.alphaRef / 255.0f;

                // Light params (for future texture-based lighting)
                data.lightParams[0] = 0.0f;  // pointLightCount - TODO
                data.lightParams[1] = 0.0f;  // lightTexelOffset - TODO
                data.lightParams[2] = 0.0f;  // texelSize - TODO
                data.lightParams[3] = 0.0f;

                // Full 4th column of worldView for robust w computation via dot product
                // Note: vertexMaterial and hasVCol are passed via shader uniforms, not texture
                data.flags[0] = wv._14;
                data.flags[1] = wv._24;
                data.flags[2] = wv._34;
                data.flags[3] = wv._44;

                fb.drawDataStaging.push_back(data);
                drawDataOffset++;
            }

            fb.statelessBatches.push_back(std::move(batch));
        } else if (kv.second.size() == 1) {
            // Track singletons for highlighting and merging
            if (highlightBatches) {
                fb.singletonCallIndices.insert(kv.second[0]);
            }
        }
    }

    // Phase 2: Build merged batches from singletons (bucket by texture + renderState)
    // This collapses many singletons into one mega-draw per texture
    std::unordered_map<MergedBatchKey, std::vector<size_t>, MergedBatchKey::Hasher> mergeGroups;

    for (const auto& kv : batchGroups) {
        if (kv.second.size() == 1) {
            size_t callIdx = kv.second[0];
            const auto& call = calls[callIdx];

            // Skip geometry that can't be merged: skinned, grass, blending, vertex blending
            // These require per-draw state that can't be batched
            if (call.sk.usesSkinning || call.sk.hasGrass || call.bin == RenderBin::Blending ||
                call.rs.vertexBlendState != 0) {
                continue;
            }

            MergedBatchKey mkey;
            mkey.texture = call.rs.texture;
            mkey.blendState = (uint16_t)((call.expectedState.captured ? call.expectedState.alphaBlendEnable : 0) |
                             ((call.expectedState.captured ? call.expectedState.srcBlend : 0) << 4) |
                             ((call.expectedState.captured ? call.expectedState.destBlend : 0) << 8));
            mkey.zState = (uint8_t)((call.expectedState.captured ? call.expectedState.zEnable : 1) |
                         ((call.expectedState.captured ? call.expectedState.zWriteEnable : 1) << 2));
            mkey.cullMode = call.expectedState.captured ? (uint8_t)call.expectedState.cullMode : D3DCULL_CW;
            mkey.useLighting = call.rs.useLighting ? 1 : 0;
            mkey.fvf = call.rs.fvf;

            mergeGroups[mkey].push_back(callIdx);
        }
    }

    // Build MergedBatch entries for groups with 2+ singletons
    for (auto& kv : mergeGroups) {
        if (kv.second.size() >= 2) {
            MergedBatch mbatch;
            mbatch.key = kv.first;
            mbatch.callIndices = std::move(kv.second);
            mbatch.drawDataOffset = drawDataOffset;
            mbatch.totalVertices = 0;
            mbatch.totalIndices = 0;

            // Calculate totals and fill draw data for each call
            for (size_t callIdx : mbatch.callIndices) {
                const auto& call = calls[callIdx];
                mbatch.totalVertices += call.rs.vertCount;
                mbatch.totalIndices += call.rs.primCount * 3;

                // Fill draw data (same as stateless batch)
                StatelessDrawData data;
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

                data.emissive[0] = call.frs.material.emissive.r;
                data.emissive[1] = call.frs.material.emissive.g;
                data.emissive[2] = call.frs.material.emissive.b;
                data.emissive[3] = (float)call.rs.alphaRef / 255.0f;

                data.lightParams[0] = 0.0f;
                data.lightParams[1] = 0.0f;
                data.lightParams[2] = 0.0f;
                data.lightParams[3] = 0.0f;

                // Full 4th column of worldView for robust w computation via dot product
                data.flags[0] = wv._14;
                data.flags[1] = wv._24;
                data.flags[2] = wv._34;
                data.flags[3] = wv._44;

                fb.drawDataStaging.push_back(data);
                drawDataOffset++;
            }

            fb.mergedBatches.push_back(std::move(mbatch));
        }
    }

    // Log batching stats with detailed breakdown
    int totalBatched = 0;
    for (const auto& batch : fb.statelessBatches) {
        totalBatched += (int)batch.callIndices.size();
    }

    // Count visible calls for accurate reporting
    int visibleCalls = 0;
    for (const auto& call : calls) {
        if (call.shouldRender && (call.bin == RenderBin::Opaque || call.bin == RenderBin::Terrain)) {
            visibleCalls++;
        }
    }

    // Detailed batch breakdown logging (on ImGui button press)
    bool dumpDetail = ImGuiManager::GetAndClearDumpStatelessBatch();
    if (dumpDetail && !fb.statelessBatches.empty()) {

        // Count unique textures
        std::set<IDirect3DTexture9*> uniqueTextures;
        for (const auto& kv : batchGroups) {
            uniqueTextures.insert(kv.first.texture);
        }

        // Analyze batch breakers
        int singletonCount = 0;  // Groups with only 1 call (not batched)
        int batchedCount = 0;
        std::map<std::string, int> breakerCounts;

        for (const auto& kv : batchGroups) {
            if (kv.second.size() == 1) {
                singletonCount++;
            } else {
                batchedCount++;
            }
        }

        // Log unique textures vs batch groups
        LOG::logline("=== StatelessBatch Breakdown ===");
        LOG::logline("Unique textures: %d", (int)uniqueTextures.size());
        LOG::logline("Total batch groups: %d (batched: %d, singletons: %d)",
                     (int)batchGroups.size(), batchedCount, singletonCount);

        // Analyze what's fragmenting batches by comparing calls with same texture
        std::map<IDirect3DTexture9*, std::vector<size_t>> callsByTexture;
        for (size_t i = 0; i < calls.size(); i++) {
            const auto& call = calls[i];
            if (call.shouldRender && (call.bin == RenderBin::Opaque || call.bin == RenderBin::Terrain)) {
                callsByTexture[call.rs.texture].push_back(i);
            }
        }

        // Analyze VB vs geometry hash - how many unique VBs vs unique geometry content
        std::set<IDirect3DVertexBuffer9*> allVBs;
        std::set<size_t> allGeoHashes;
        std::map<size_t, std::set<IDirect3DVertexBuffer9*>> vbsByGeoHash;  // Which VBs share same geometry

        // Track what's making hashes unique
        std::map<DWORD, int> fvfCounts;
        std::map<UINT, int> vertCountCounts;
        std::map<UINT, int> offsetCounts;
        std::map<UINT, int> strideCounts;

        for (size_t i = 0; i < calls.size(); i++) {
            const auto& call = calls[i];
            if (call.shouldRender && (call.bin == RenderBin::Opaque || call.bin == RenderBin::Terrain)) {
                allVBs.insert(call.rs.vb);
                fvfCounts[call.rs.fvf]++;
                vertCountCounts[call.rs.vertCount]++;
                offsetCounts[call.rs.vbOffset]++;
                strideCounts[call.rs.vbStride]++;

                size_t geoHash = computeGeometryHash(call.rs.vb, call.rs.vbOffset, call.rs.vbStride,
                                                      call.rs.vertCount, call.rs.fvf);
                allGeoHashes.insert(geoHash);
                vbsByGeoHash[geoHash].insert(call.rs.vb);
            }
        }

        // Log hash component distributions
        LOG::logline("Hash components - FVF types: %d, VertCount values: %d, Offsets: %d, Strides: %d",
                     (int)fvfCounts.size(), (int)vertCountCounts.size(), (int)offsetCounts.size(), (int)strideCounts.size());

        // If few unique FVFs/strides, content must be differentiating
        if (fvfCounts.size() <= 5) {
            LOG::logline("  FVF distribution:");
            for (auto& kv : fvfCounts) {
                LOG::logline("    FVF 0x%X: %d calls", kv.first, kv.second);
            }
        }
        if (offsetCounts.size() <= 3) {
            LOG::logline("  All offsets: %s", offsetCounts.count(0) ? "0 (good)" : "varying (problem!)");
        } else {
            LOG::logline("  Offset values: %d unique (batching by offset broken!)", (int)offsetCounts.size());
        }

        // Count duplicate VBs (same geometry hash, different VB pointer)
        int duplicateVBs = 0;
        int geoHashesWithDupes = 0;
        for (const auto& kv : vbsByGeoHash) {
            if (kv.second.size() > 1) {
                geoHashesWithDupes++;
                duplicateVBs += (int)kv.second.size() - 1;
            }
        }

        LOG::logline("Unique VBs: %d, Unique geometry hashes: %d", (int)allVBs.size(), (int)allGeoHashes.size());
        LOG::logline("Duplicate VBs (same geometry): %d across %d meshes", duplicateVBs, geoHashesWithDupes);

        // Find textures where calls aren't batching together
        int fragmentedTextures = 0;
        int totalFragmentation = 0;
        for (const auto& texCalls : callsByTexture) {
            if (texCalls.second.size() >= 2) {
                // Count unique geometry hashes for this texture
                std::set<size_t> geoHashes;
                for (size_t idx : texCalls.second) {
                    size_t geoHash = computeGeometryHash(calls[idx].rs.vb, calls[idx].rs.vbOffset,
                                                          calls[idx].rs.vbStride, calls[idx].rs.vertCount,
                                                          calls[idx].rs.fvf);
                    geoHashes.insert(geoHash);
                }
                if (geoHashes.size() > 1) {
                    fragmentedTextures++;
                    totalFragmentation += (int)geoHashes.size() - 1;
                }
            }
        }

        LOG::logline("Textures fragmented by geometry: %d (extra batches: %d)",
                     fragmentedTextures, totalFragmentation);

        // Full batch dump
        LOG::logline("");
        LOG::logline("=== BATCHED GROUPS (2+ draws) ===");
        int batchIdx = 0;
        for (const auto& batch : fb.statelessBatches) {
            const auto& key = batch.key;
            LOG::logline("Batch %d: %d draws, tex=%p, vc=%d, pc=%d, blend=0x%X, z=0x%X, cull=%d, lit=%d",
                batchIdx++, (int)batch.callIndices.size(), key.texture,
                key.vertCount, key.primCount, key.blendState, key.zState, key.cullMode, key.useLighting);
        }

        // Singleton analysis - why didn't they batch?
        LOG::logline("");
        LOG::logline("=== SINGLETONS (unique geometry per texture) ===");

        // Group singletons by texture to show potential
        std::map<IDirect3DTexture9*, std::vector<const StatelessBatchKey*>> singletonsByTex;
        for (const auto& kv : batchGroups) {
            if (kv.second.size() == 1) {
                singletonsByTex[kv.first.texture].push_back(&kv.first);
            }
        }

        // Show textures with multiple singletons (potential batching if geometry merged)
        for (const auto& texSingles : singletonsByTex) {
            if (texSingles.second.size() >= 2) {
                LOG::logline("Tex %p: %d singletons (could batch if geometry merged)",
                    texSingles.first, (int)texSingles.second.size());
                // Show first few
                int shown = 0;
                for (const auto* key : texSingles.second) {
                    if (shown++ >= 5) {
                        LOG::logline("  ... and %d more", (int)texSingles.second.size() - 5);
                        break;
                    }
                    LOG::logline("  vc=%d pc=%d blend=0x%X z=0x%X cull=%d",
                        key->vertCount, key->primCount, key->blendState, key->zState, key->cullMode);
                }
            }
        }

        // VertCount distribution (shows mesh variety)
        LOG::logline("");
        LOG::logline("=== VERTEX COUNT DISTRIBUTION ===");
        std::map<UINT, int> vcDist;
        for (const auto& kv : batchGroups) {
            vcDist[kv.first.vertCount] += (int)kv.second.size();
        }
        // Sort by frequency
        std::vector<std::pair<int, UINT>> vcSorted;
        for (const auto& kv : vcDist) {
            vcSorted.push_back({kv.second, kv.first});
        }
        std::sort(vcSorted.rbegin(), vcSorted.rend());
        LOG::logline("Top 10 vertex counts by draw frequency:");
        for (int i = 0; i < std::min(10, (int)vcSorted.size()); i++) {
            LOG::logline("  vc=%d: %d draws", vcSorted[i].second, vcSorted[i].first);
        }

        // Index Buffer analysis - can we batch by shared IB?
        LOG::logline("");
        LOG::logline("=== INDEX BUFFER ANALYSIS ===");

        // Collect all singleton call indices for IB analysis
        std::vector<size_t> singletonCallIdxs;
        for (const auto& kv : batchGroups) {
            if (kv.second.size() == 1) {
                singletonCallIdxs.push_back(kv.second[0]);
            }
        }

        // Group singletons by IB
        std::map<IDirect3DIndexBuffer9*, std::vector<size_t>> singletonsByIB;
        std::set<IDirect3DIndexBuffer9*> uniqueIBs;
        for (size_t idx : singletonCallIdxs) {
            const auto& call = calls[idx];
            singletonsByIB[call.rs.ib].push_back(idx);
            uniqueIBs.insert(call.rs.ib);
        }

        LOG::logline("Singletons: %d calls, %d unique IBs", (int)singletonCallIdxs.size(), (int)uniqueIBs.size());

        // Find IBs shared by multiple singletons (batching potential without IB merge)
        int sharedIBCount = 0;
        int callsWithSharedIB = 0;
        for (const auto& ibCalls : singletonsByIB) {
            if (ibCalls.second.size() >= 2) {
                sharedIBCount++;
                callsWithSharedIB += (int)ibCalls.second.size();
            }
        }
        LOG::logline("Shared IBs: %d IBs used by %d calls (no merge needed!)", sharedIBCount, callsWithSharedIB);

        // Show top shared IBs with their draw ranges
        LOG::logline("Top shared IBs:");
        std::vector<std::pair<int, IDirect3DIndexBuffer9*>> ibSorted;
        for (const auto& kv : singletonsByIB) {
            if (kv.second.size() >= 2) {
                ibSorted.push_back({(int)kv.second.size(), kv.first});
            }
        }
        std::sort(ibSorted.rbegin(), ibSorted.rend());

        for (int i = 0; i < std::min(5, (int)ibSorted.size()); i++) {
            auto ib = ibSorted[i].second;
            const auto& callIdxs = singletonsByIB[ib];
            LOG::logline("  IB %p: %d singletons", ib, (int)callIdxs.size());

            // Show startIndex ranges to verify they use different offsets
            std::set<IDirect3DTexture9*> textures;
            UINT minStart = UINT_MAX, maxStart = 0;
            UINT minPrims = UINT_MAX, maxPrims = 0;
            for (size_t idx : callIdxs) {
                const auto& call = calls[idx];
                textures.insert(call.rs.texture);
                minStart = std::min(minStart, call.rs.startIndex);
                maxStart = std::max(maxStart, call.rs.startIndex);
                minPrims = std::min(minPrims, call.rs.primCount);
                maxPrims = std::max(maxPrims, call.rs.primCount);
            }
            LOG::logline("    textures=%d, startIndex=[%d-%d], primCount=[%d-%d]",
                (int)textures.size(), minStart, maxStart, minPrims, maxPrims);
        }

        // Unique IB singletons (would need IB merge)
        int uniqueIBSingletons = (int)singletonCallIdxs.size() - callsWithSharedIB;
        LOG::logline("Unique IB singletons: %d (would need IB merge)", uniqueIBSingletons);

        // Local Hash analysis - can centering collapse duplicates?
        LOG::logline("");
        LOG::logline("=== LOCAL HASH ANALYSIS (Centered Geometry) ===");

        std::map<size_t, int> localHashCounts;  // localHash -> count
        std::map<std::pair<size_t, IDirect3DTexture9*>, int> mergedKeyCount;  // (localHash, tex) -> count
        int localMeshCacheHits = 0;
        int localMeshCacheMisses = 0;

        for (size_t idx : singletonCallIdxs) {
            const auto& call = calls[idx];

            // Compute geometry hash first (needed as cache key)
            size_t geoHash = computeGeometryHash(call.rs.vb, call.rs.vbOffset, call.rs.vbStride,
                                                  call.rs.vertCount, call.rs.fvf);

            // Get or create local mesh (centered)
            const LocalMeshData* localMesh = getOrCreateLocalMesh(
                geoHash, call.rs.vb, call.rs.vbOffset, call.rs.vbStride, call.rs.vertCount, call.rs.fvf,
                call.rs.ib, call.rs.startIndex, call.rs.primCount);

            if (localMesh) {
                localHashCounts[localMesh->localHash]++;
                mergedKeyCount[{localMesh->localHash, call.rs.texture}]++;

                // Check if this was a cache hit (already existed before this frame)
                // We can't easily tell, so just count total
            }
        }

        int uniqueLocalHashes = (int)localHashCounts.size();
        int potentialMergedBatches = (int)mergedKeyCount.size();

        LOG::logline("Singletons: %d -> %d unique local hashes (%.1fx reduction)",
            (int)singletonCallIdxs.size(), uniqueLocalHashes,
            uniqueLocalHashes > 0 ? (float)singletonCallIdxs.size() / uniqueLocalHashes : 0.0f);
        LOG::logline("Potential merged batches (localHash + texture): %d", potentialMergedBatches);

        // Show top repeated local hashes
        std::vector<std::pair<int, size_t>> localHashSorted;
        for (const auto& kv : localHashCounts) {
            if (kv.second >= 2) {
                localHashSorted.push_back({kv.second, kv.first});
            }
        }
        std::sort(localHashSorted.rbegin(), localHashSorted.rend());

        if (!localHashSorted.empty()) {
            LOG::logline("Top repeated local hashes (identical geometry at different positions):");
            for (int i = 0; i < std::min(5, (int)localHashSorted.size()); i++) {
                LOG::logline("  localHash %zX: %d instances", localHashSorted[i].second, localHashSorted[i].first);
            }
        }

        LOG::logline("================================");
    }

    // Always log summary
    if (!fb.statelessBatches.empty() || !fb.mergedBatches.empty()) {
        int totalMerged = 0;
        for (const auto& mb : fb.mergedBatches) {
            totalMerged += (int)mb.callIndices.size();
        }
        LOG::logline("Batching: %d stateless (%d draws) + %d merged (%d draws) = %d total draws -> %d GPU calls",
                     (int)fb.statelessBatches.size(), totalBatched,
                     (int)fb.mergedBatches.size(), totalMerged,
                     totalBatched + totalMerged,
                     (int)fb.statelessBatches.size() + (int)fb.mergedBatches.size());
    }
}

// Helper: Convert FVF to vertex declaration elements for stream 0
static bool fvfToElements(DWORD fvf, std::vector<D3DVERTEXELEMENT9>& elements) {
    WORD offset = 0;

    // Position (required)
    if (fvf & D3DFVF_XYZRHW) {
        elements.push_back({0, offset, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITIONT, 0});
        offset += 16;
    } else if (fvf & D3DFVF_XYZ) {
        elements.push_back({0, offset, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0});
        offset += 12;
    }

    // Blend weights (XYZB1-4)
    // FVF position encoding: XYZ=1, XYZRHW=2, XYZB1=3, XYZB2=4, XYZB3=5, XYZB4=6
    // Blend weight count = position_type - 2 (for types >= 3)
    int posType = (fvf >> 1) & 0x7;
    int blendWeights = (posType >= 3) ? (posType - 2) : 0;
    if (blendWeights > 0 && blendWeights <= 4) {
        BYTE types[] = {D3DDECLTYPE_FLOAT1, D3DDECLTYPE_FLOAT2, D3DDECLTYPE_FLOAT3, D3DDECLTYPE_FLOAT4};
        elements.push_back({0, offset, types[blendWeights - 1], D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDWEIGHT, 0});
        offset += blendWeights * 4;
    }

    // Normal
    if (fvf & D3DFVF_NORMAL) {
        elements.push_back({0, offset, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0});
        offset += 12;
    }

    // Diffuse color
    if (fvf & D3DFVF_DIFFUSE) {
        elements.push_back({0, offset, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0});
        offset += 4;
    }

    // Specular color
    if (fvf & D3DFVF_SPECULAR) {
        elements.push_back({0, offset, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 1});
        offset += 4;
    }

    // Texture coordinates
    int numTexCoords = (fvf & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT;
    for (int i = 0; i < numTexCoords; i++) {
        // Get texcoord format (default is 2D float)
        int fmt = (fvf >> (16 + i * 2)) & 0x3;
        BYTE type = D3DDECLTYPE_FLOAT2;
        int size = 8;
        switch (fmt) {
            case 0: type = D3DDECLTYPE_FLOAT2; size = 8; break;  // D3DFVF_TEXTUREFORMAT2
            case 1: type = D3DDECLTYPE_FLOAT3; size = 12; break; // D3DFVF_TEXTUREFORMAT3
            case 2: type = D3DDECLTYPE_FLOAT4; size = 16; break; // D3DFVF_TEXTUREFORMAT4
            case 3: type = D3DDECLTYPE_FLOAT1; size = 4; break;  // D3DFVF_TEXTUREFORMAT1
        }
        elements.push_back({0, offset, type, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, (BYTE)i});
        offset += size;
    }

    return !elements.empty();
}

IDirect3DVertexDeclaration9* FixedFunctionShader::getInstancedDecl(DWORD fvf) {
    // Check cache first
    auto it = fvfDeclCache.find(fvf);
    if (it != fvfDeclCache.end()) {
        return it->second;
    }

    // Build declaration elements from FVF
    std::vector<D3DVERTEXELEMENT9> elements;
    if (!fvfToElements(fvf, elements)) {
        LOG::logline("!! Failed to convert FVF 0x%08X to vertex elements", fvf);
        fvfDeclCache[fvf] = nullptr;
        return nullptr;
    }

    // Add instance stream elements (TEXCOORD 8-10 for world matrix rows)
    elements.push_back({1, 0,  D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 8});
    elements.push_back({1, 16, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 9});
    elements.push_back({1, 32, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 10});

    // End marker
    elements.push_back(D3DDECL_END());

    // Create vertex declaration
    auto* device = DistantLand::device;
    IDirect3DVertexDeclaration9* decl = nullptr;
    HRESULT hr = device->CreateVertexDeclaration(elements.data(), &decl);

    if (FAILED(hr)) {
        LOG::logline("!! Failed to create instanced vertex decl for FVF 0x%08X", fvf);
        fvfDeclCache[fvf] = nullptr;
        return nullptr;
    }

    fvfDeclCache[fvf] = decl;
    LOG::logline("-- Created instanced vertex decl for FVF 0x%08X", fvf);
    return decl;
}

// Get or create stateless batch vertex declaration for FVF
// Similar to getInstancedDecl but stream 1 only has a single float (drawIndex)
IDirect3DVertexDeclaration9* FixedFunctionShader::getStatelessBatchDecl(DWORD fvf) {
    // Check cache first
    auto it = statelessDeclCache.find(fvf);
    if (it != statelessDeclCache.end()) {
        return it->second;
    }

    // Build declaration elements from FVF
    std::vector<D3DVERTEXELEMENT9> elements;
    if (!fvfToElements(fvf, elements)) {
        LOG::logline("!! Failed to convert FVF 0x%08X to vertex elements for stateless batch", fvf);
        statelessDeclCache[fvf] = nullptr;
        return nullptr;
    }

    // Add stateless batch stream element (TEXCOORD8 for drawIndex - single float)
    elements.push_back({1, 0, D3DDECLTYPE_FLOAT1, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 8});

    // End marker
    elements.push_back(D3DDECL_END());

    // Create vertex declaration
    auto* device = DistantLand::device;
    IDirect3DVertexDeclaration9* decl = nullptr;
    HRESULT hr = device->CreateVertexDeclaration(elements.data(), &decl);

    if (FAILED(hr)) {
        LOG::logline("!! Failed to create stateless batch vertex decl for FVF 0x%08X", fvf);
        statelessDeclCache[fvf] = nullptr;
        return nullptr;
    }

    statelessDeclCache[fvf] = decl;
    LOG::logline("-- Created stateless batch vertex decl for FVF 0x%08X", fvf);
    return decl;
}
