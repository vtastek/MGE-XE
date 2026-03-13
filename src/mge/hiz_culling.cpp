// Hi-Z occlusion culling implementation
// Split from ffeshader.cpp for organization - these are still FixedFunctionShader members
#include "ffeshader.h"
#include "configuration.h"
#include "support/log.h"
#include "distantland.h"
#include "imgui_manager.h"
#include "mge_tracy.h"
#include "texture_suffix.h"

// External reference to file-scope variable in ffeshader.cpp
extern float lastPrepareMs;
extern bool deviceCallsSafeInPrepare;

// Helper to access current recording buffer's recorded calls
static auto& currentRecordedCalls() {
    return FixedFunctionShader::frameBuffers[FixedFunctionShader::recordingBuffer].recordedCalls;
}

// executeHiZCulling - Pure CPU work: bbox computation, occluder rasterization, Hi-Z pyramid build,
// visibility testing on recordMW and recordedCalls. Takes view/proj as parameters (no D3D device access).
// This is a draw-thread candidate: touches no GPU state.
void FixedFunctionShader::executeHiZCulling(const D3DXMATRIX& currentView, const D3DXMATRIX& currentProj) {
    MGE_ZoneScopedN("Execute Hi-Z Culling");

    // Resize visibility results for this frame (indexed by draw order)
    // HLSL mode: recordMW is per-buffer; legacy mode: global static
    const auto& activeRecordMW = isHLSLActive()
        ? frameBuffers[recordingBuffer].recordMW
        : DistantLand::recordMW;
    visibilityResults.assign(activeRecordMW.size(), -1);  // -1 = not yet tested

    // Phase 2a: Compute deferred bboxes and rasterize occluders from recorded HLSL calls
    // This must happen before Hi-Z build so the depth pass benefits from culling
    auto& recCalls = currentRecordedCalls();
    if (!recCalls.empty() && !hiZBuiltThisFrame) {
        // Compute bounding boxes for calls that missed the cache during recording
        {
            MGE_ZoneScopedN("Prepare: BBox Cache Misses");
            for (auto& call : recCalls) {
                if (!call.hasBoundingBox) {
                    call.hasBoundingBox = computeBoundingBox(&call.rs, call.bboxMin, call.bboxMax);
                    if (call.hasBoundingBox) {
                        VBIBKey key{call.rs.vb, call.rs.ib};
                        bboxLookup[key] = {call.bboxMin, call.bboxMax};
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

            auto& fb = frameBuffers[recordingBuffer];
            D3DXMATRIX viewProj = fb.view * fb.proj;

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
                    fb.view, fb.proj
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
            softwareOcclusionCuller.uploadHiZToTexture(reinterpret_cast<IDirect3DDevice9*>(device), ImGuiManager::GetHiZDisplayMip(), frameBuffers[recordingBuffer].proj, ImGuiManager::GetHiZInvert(), ImGuiManager::GetHiZShowRaycastGrid(), ImGuiManager::GetHiZRaycastStep());
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
            // Scene 1+ (hands/alpha after Z-clear): always visible, skip Hi-Z test
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
void FixedFunctionShader::applyVisibilityAndFilterRecordMW(FrameBuffer* fb) {
    MGE_ZoneScopedN("Apply Visibility Filter to recordMW");

    // Non-HLSL mode or culling disabled: nothing to filter
    if (Configuration.PerPixelLightFlags != 2 || ImGuiManager::GetDisableHiZCulling()) {
        return;
    }

    // Select recordMW source: per-buffer in HLSL mode, global static otherwise
    auto& activeRecordMW = fb ? fb->recordMW : DistantLand::recordMW;

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

// Dirty tracking: match current frame calls to previous frame by MeshKey
void FixedFunctionShader::matchPreviousFrameCalls(int bufferIndex) {
    MGE_ZoneScopedN("matchPreviousFrameCalls");

    auto& curCalls = frameBuffers[bufferIndex].recordedCalls;

    // Find previous frame's FrameBuffer (the one before this buffer)
    int prevBuf = (bufferIndex + 2) % 3;
    auto& prevCalls = frameBuffers[prevBuf].recordedCalls;

    if (prevCalls.empty()) {
        // First frame or no previous data — all dirty
        for (auto& call : curCalls) {
            call.dirtyFlags = DIRTY_ALL;
        }
        return;
    }

    // Build lookup from previous frame
    std::unordered_map<MeshKey, int, MeshKeyHash> prevLookup;
    prevLookup.reserve(prevCalls.size());
    for (int i = 0; i < (int)prevCalls.size(); ++i) {
        auto& prev = prevCalls[i];
        MeshKey key;
        key.vb = prev.rs.vb;
        key.ib = prev.rs.ib;
        key.fvf = prev.rs.fvf;
        key.baseIndex = prev.rs.baseIndex;
        key.vertCount = prev.rs.vertCount;
        key.startIndex = prev.rs.startIndex;
        key.primCount = prev.rs.primCount;
        prevLookup[key] = i;  // Last wins for duplicates
    }

    for (auto& call : curCalls) {
        MeshKey key;
        key.vb = call.rs.vb;
        key.ib = call.rs.ib;
        key.fvf = call.rs.fvf;
        key.baseIndex = call.rs.baseIndex;
        key.vertCount = call.rs.vertCount;
        key.startIndex = call.rs.startIndex;
        key.primCount = call.rs.primCount;

        auto it = prevLookup.find(key);
        if (it == prevLookup.end()) {
            call.dirtyFlags = DIRTY_ALL;
            continue;
        }

        auto& prev = prevCalls[it->second];
        DWORD flags = DIRTY_NONE;

        // Compare world transform (object moved?)
        if (memcmp(&call.rs.worldTransforms[0], &prev.rs.worldTransforms[0], sizeof(D3DXMATRIX)) != 0) {
            flags |= DIRTY_TRANSFORM;
        }

        // Compare light state (pointer comparison — shared_ptr reuse)
        if (call.lightrs.get() != prev.lightrs.get()) {
            flags |= DIRTY_LIGHT;
        }

        // Compare material
        if (memcmp(&call.frs.material, &prev.frs.material, sizeof(FragmentState::Material)) != 0) {
            flags |= DIRTY_MATERIAL;
        }

        // Compare shader key
        if (!(call.sk == prev.sk)) {
            flags |= DIRTY_SHADER;
        }

        // Compare blend state
        if (call.rs.blendEnable != prev.rs.blendEnable ||
            call.rs.srcBlend != prev.rs.srcBlend ||
            call.rs.destBlend != prev.rs.destBlend) {
            flags |= DIRTY_BLEND;
        }

        // Compare base texture
        if (call.rs.texture != prev.rs.texture) {
            flags |= DIRTY_TEXTURE;
        }

        call.dirtyFlags = flags;
    }
}

// Phase 2b: Prepare shader keys — runs after recording completes, before replay.
// BBox computation and occluder rasterization already done in executeHiZCulling (Phase 2a).
void FixedFunctionShader::prepareRecordedCalls(int bufferIndex) {
    MGE_ZoneScopedN("prepareRecordedCalls");

    auto& recCalls = frameBuffers[bufferIndex].recordedCalls;
    if (recCalls.empty()) {
        return;
    }

    // Compute shader keys and assign render bins (with slow-frame timing)
    LARGE_INTEGER freqQPC, prepStartQPC, prepEndQPC;
    QueryPerformanceFrequency(&freqQPC);
    QueryPerformanceCounter(&prepStartQPC);
    {
        MGE_ZoneScopedN("Prepare: Shader Keys + Bins");
        for (auto& call : recCalls) {
            call.sk = computeShaderKeyWithSuffixes(&call.rs, &call.frs, call.lightrs.get());
            call.prepared = true;

            // Classify into render bin
            if (call.sk.hasGrass)              call.bin = RenderBin::Grass;
            else if (call.sk.usesSkinning)     call.bin = RenderBin::Skinning;
            else if (call.rs.blendEnable)      call.bin = RenderBin::Blending;
            else if (call.rs.alphaTest)         call.bin = RenderBin::AlphaTested;
            else                                call.bin = RenderBin::Opaque;
        }
    }
    QueryPerformanceCounter(&prepEndQPC);
    lastPrepareMs = (prepEndQPC.QuadPart - prepStartQPC.QuadPart) * 1000.0f / freqQPC.QuadPart;
    if (lastPrepareMs > ImGuiManager::GetSlowFrameThreshold()) {
        LOG::logline("SLOW PREPARE: %.1fms for %d calls (Shader Keys + Bins)", lastPrepareMs, (int)recCalls.size());
    }

    // Performance mode: match against previous frame for dirty tracking
    if (ImGuiManager::GetPerformanceMode()) {
        matchPreviousFrameCalls(bufferIndex);
    }
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

    // Check cache first
    auto it = bboxCache.find(key);
    if (it != bboxCache.end()) {
        // Cache hit! Transform cached object-space bbox to world-space
        const ObjectSpaceBBox& objBBox = it->second;

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

    // Cache the object-space bbox
    ObjectSpaceBBox cachedBBox;
    cachedBBox.bboxMin = objBBoxMin;
    cachedBBox.bboxMax = objBBoxMax;
    bboxCache[key] = cachedBBox;

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
