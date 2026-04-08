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

// External reference to file-scope variable in ffeshader.cpp
extern float lastPrepareMs;
extern bool deviceCallsSafeInPrepare;

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
            if (call.sk.hasGrass)              call.bin = RenderBin::Grass;
            else if (call.sk.usesSkinning)     call.bin = RenderBin::Skinning;
            else if (call.rs.blendEnable)      call.bin = RenderBin::Blending;
            else if (call.rs.alphaTest)        call.bin = RenderBin::AlphaTested;
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
