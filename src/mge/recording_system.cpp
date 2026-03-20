// Recording system for HLSL render dispatch pipeline
// Extracted from ffeshader.cpp - Phase 4 of refactor

#include "ffeshader.h"
#include "texture_suffix.h"
#include "renderthread.h"
#include "d3dcommandbuffer.h"
#include "mge_tracy.h"
#include "configuration.h"
#include "support/log.h"
#include "mwbridge.h"
#include "morrowindbsa.h"
#include "statusoverlay.h"
#include "distantland.h"
#include "imgui_manager.h"

#include <unordered_set>

// Per-stage command buffer set (defined in mged3d8device.cpp)
extern D3DCommandBufferSet g_cmdBufferSet;

// File-scope helpers for recording system
static std::unordered_map<IDirect3DBaseTexture9*, std::pair<DWORD, DWORD>> samplerCache;

// Route to correct vector based on current recording scene
// Scene 0 = world, Scene 1 = particles (alpha sorted), Scene 2 = hands (skinned)
static auto& currentRecordedCalls() {
    auto& fb = FixedFunctionShader::frameBuffers[FixedFunctionShader::recordingBuffer];
    int scene = FixedFunctionShader::getCurrentRecordingScene();
    if (scene == 1) {
        return fb.recordedCallsScene1;
    } else if (scene >= 2) {
        return fb.recordedCallsScene2;
    }
    return fb.recordedCalls;
}

// Diagnostic: cache hit/miss logging for first N frames (temporary)
static int hlslDiagFrameCounter = 0;

// === Debug Validation ===
// Compare tracked state vs device state (for verifying tracker accuracy)
#ifdef _DEBUG
static void validateTrackedVsDevice(const MWStateTracker& tracker, const StateContract& device, const char* context) {
    StateContract tracked;
    tracker.exportToStateContract(&tracked);

    // Compare key fields and log mismatches
    auto checkRS = [&](const char* name, DWORD t, DWORD d) {
        if (t != d) {
            LOG::logline("[%s] Tracker mismatch %s: tracked=%lu device=%lu", context, name, t, d);
        }
    };

    checkRS("zEnable", tracked.zEnable, device.zEnable);
    checkRS("zWriteEnable", tracked.zWriteEnable, device.zWriteEnable);
    checkRS("zFunc", tracked.zFunc, device.zFunc);
    checkRS("alphaBlendEnable", tracked.alphaBlendEnable, device.alphaBlendEnable);
    checkRS("srcBlend", tracked.srcBlend, device.srcBlend);
    checkRS("destBlend", tracked.destBlend, device.destBlend);
    checkRS("alphaTestEnable", tracked.alphaTestEnable, device.alphaTestEnable);
    checkRS("alphaFunc", tracked.alphaFunc, device.alphaFunc);
    checkRS("alphaRef", tracked.alphaRef, device.alphaRef);
    checkRS("cullMode", tracked.cullMode, device.cullMode);
    checkRS("fogEnable", tracked.fogEnable, device.fogEnable);

    // Compare sampler states
    for (int s = 0; s < 2; ++s) {
        char buf[32];
        snprintf(buf, sizeof(buf), "sampler%d.minFilter", s);
        checkRS(buf, tracked.samplers[s].minFilter, device.samplers[s].minFilter);
        snprintf(buf, sizeof(buf), "sampler%d.magFilter", s);
        checkRS(buf, tracked.samplers[s].magFilter, device.samplers[s].magFilter);
        snprintf(buf, sizeof(buf), "sampler%d.addressU", s);
        checkRS(buf, tracked.samplers[s].addressU, device.samplers[s].addressU);
        snprintf(buf, sizeof(buf), "sampler%d.addressV", s);
        checkRS(buf, tracked.samplers[s].addressV, device.samplers[s].addressV);
    }
}
#endif

// === MWStateTracker Implementation ===
// Export tracked shadow state to StateContract for restoration

void MWStateTracker::exportToStateContract(StateContract* out) const {
    // Helper to get tracked value or keep default
    auto getRS = [this](DWORD state, DWORD* target) {
        DWORD val;
        if (getRenderState(state, &val)) *target = val;
    };
    auto getSS = [this](DWORD sampler, DWORD state, DWORD* target) {
        DWORD val;
        if (getSamplerState(sampler, state, &val)) *target = val;
    };

    // Depth state
    getRS(D3DRS_ZENABLE, &out->zEnable);
    getRS(D3DRS_ZWRITEENABLE, &out->zWriteEnable);
    getRS(D3DRS_ZFUNC, &out->zFunc);

    // Blending state
    getRS(D3DRS_ALPHABLENDENABLE, &out->alphaBlendEnable);
    getRS(D3DRS_SRCBLEND, &out->srcBlend);
    getRS(D3DRS_DESTBLEND, &out->destBlend);

    // Alpha test state
    getRS(D3DRS_ALPHATESTENABLE, &out->alphaTestEnable);
    getRS(D3DRS_ALPHAFUNC, &out->alphaFunc);
    getRS(D3DRS_ALPHAREF, &out->alphaRef);

    // Culling and fog
    getRS(D3DRS_CULLMODE, &out->cullMode);
    getRS(D3DRS_FOGENABLE, &out->fogEnable);

    // Specular and lighting
    getRS(D3DRS_SPECULARENABLE, &out->specularEnable);
    getRS(D3DRS_LOCALVIEWER, &out->localViewer);
    getRS(D3DRS_NORMALIZENORMALS, &out->normalizeNormals);

    // Sampler states for stages 0-1
    for (DWORD s = 0; s < 2; ++s) {
        getSS(s, D3DSAMP_MINFILTER, &out->samplers[s].minFilter);
        getSS(s, D3DSAMP_MAGFILTER, &out->samplers[s].magFilter);
        getSS(s, D3DSAMP_MIPFILTER, &out->samplers[s].mipFilter);
        getSS(s, D3DSAMP_ADDRESSU, &out->samplers[s].addressU);
        getSS(s, D3DSAMP_ADDRESSV, &out->samplers[s].addressV);
    }

    // Transforms
    getTransform(D3DTS_WORLD, &out->world);
    getTransform(D3DTS_VIEW, &out->view);
    getTransform(D3DTS_PROJECTION, &out->projection);
}

// === StateContract Implementation ===
// Centralized device state capture/restore for explicit phase handoffs

void StateContract::captureFromTracker(const MWStateTracker& tracker) {
    // Delegate to MWStateTracker's export method
    tracker.exportToStateContract(this);
}

void StateContract::captureFrom(IDirect3DDevice9* dev) {
    // Depth state
    dev->GetRenderState(D3DRS_ZENABLE, &zEnable);
    dev->GetRenderState(D3DRS_ZWRITEENABLE, &zWriteEnable);
    dev->GetRenderState(D3DRS_ZFUNC, &zFunc);

    // Blending state
    dev->GetRenderState(D3DRS_ALPHABLENDENABLE, &alphaBlendEnable);
    dev->GetRenderState(D3DRS_SRCBLEND, &srcBlend);
    dev->GetRenderState(D3DRS_DESTBLEND, &destBlend);

    // Alpha test state
    dev->GetRenderState(D3DRS_ALPHATESTENABLE, &alphaTestEnable);
    dev->GetRenderState(D3DRS_ALPHAFUNC, &alphaFunc);
    dev->GetRenderState(D3DRS_ALPHAREF, &alphaRef);

    // Culling and fog
    dev->GetRenderState(D3DRS_CULLMODE, &cullMode);
    dev->GetRenderState(D3DRS_FOGENABLE, &fogEnable);

    // Specular and lighting (legacy FFE state)
    dev->GetRenderState(D3DRS_SPECULARENABLE, &specularEnable);
    dev->GetRenderState(D3DRS_LOCALVIEWER, &localViewer);
    dev->GetRenderState(D3DRS_NORMALIZENORMALS, &normalizeNormals);

    // Sampler states for stages 0-1
    dev->GetSamplerState(0, D3DSAMP_MINFILTER, &samplers[0].minFilter);
    dev->GetSamplerState(0, D3DSAMP_MAGFILTER, &samplers[0].magFilter);
    dev->GetSamplerState(0, D3DSAMP_MIPFILTER, &samplers[0].mipFilter);
    dev->GetSamplerState(0, D3DSAMP_ADDRESSU, &samplers[0].addressU);
    dev->GetSamplerState(0, D3DSAMP_ADDRESSV, &samplers[0].addressV);
    dev->GetSamplerState(1, D3DSAMP_MINFILTER, &samplers[1].minFilter);
    dev->GetSamplerState(1, D3DSAMP_MAGFILTER, &samplers[1].magFilter);
    dev->GetSamplerState(1, D3DSAMP_MIPFILTER, &samplers[1].mipFilter);
    dev->GetSamplerState(1, D3DSAMP_ADDRESSU, &samplers[1].addressU);
    dev->GetSamplerState(1, D3DSAMP_ADDRESSV, &samplers[1].addressV);

    // Transforms
    dev->GetTransform(D3DTS_WORLD, (D3DMATRIX*)&world);
    dev->GetTransform(D3DTS_VIEW, (D3DMATRIX*)&view);
    dev->GetTransform(D3DTS_PROJECTION, (D3DMATRIX*)&projection);
}

void StateContract::applyTo(IDirect3DDevice9* dev) const {
    applyRenderStatesTo(dev);
    applySamplersTo(dev);
    applyTransformsTo(dev);
}

void StateContract::applyRenderStatesTo(IDirect3DDevice9* dev) const {
    // Depth state
    dev->SetRenderState(D3DRS_ZENABLE, zEnable);
    dev->SetRenderState(D3DRS_ZWRITEENABLE, zWriteEnable);
    dev->SetRenderState(D3DRS_ZFUNC, zFunc);

    // Blending state
    dev->SetRenderState(D3DRS_ALPHABLENDENABLE, alphaBlendEnable);
    dev->SetRenderState(D3DRS_SRCBLEND, srcBlend);
    dev->SetRenderState(D3DRS_DESTBLEND, destBlend);

    // Alpha test state
    dev->SetRenderState(D3DRS_ALPHATESTENABLE, alphaTestEnable);
    dev->SetRenderState(D3DRS_ALPHAFUNC, alphaFunc);
    dev->SetRenderState(D3DRS_ALPHAREF, alphaRef);

    // Culling and fog
    dev->SetRenderState(D3DRS_CULLMODE, cullMode);
    dev->SetRenderState(D3DRS_FOGENABLE, fogEnable);

    // Specular and lighting
    dev->SetRenderState(D3DRS_SPECULARENABLE, specularEnable);
    dev->SetRenderState(D3DRS_LOCALVIEWER, localViewer);
    dev->SetRenderState(D3DRS_NORMALIZENORMALS, normalizeNormals);
}

void StateContract::applySamplersTo(IDirect3DDevice9* dev) const {
    // Sampler states for stages 0-1
    dev->SetSamplerState(0, D3DSAMP_MINFILTER, samplers[0].minFilter);
    dev->SetSamplerState(0, D3DSAMP_MAGFILTER, samplers[0].magFilter);
    dev->SetSamplerState(0, D3DSAMP_MIPFILTER, samplers[0].mipFilter);
    dev->SetSamplerState(0, D3DSAMP_ADDRESSU, samplers[0].addressU);
    dev->SetSamplerState(0, D3DSAMP_ADDRESSV, samplers[0].addressV);
    dev->SetSamplerState(1, D3DSAMP_MINFILTER, samplers[1].minFilter);
    dev->SetSamplerState(1, D3DSAMP_MAGFILTER, samplers[1].magFilter);
    dev->SetSamplerState(1, D3DSAMP_MIPFILTER, samplers[1].mipFilter);
    dev->SetSamplerState(1, D3DSAMP_ADDRESSU, samplers[1].addressU);
    dev->SetSamplerState(1, D3DSAMP_ADDRESSV, samplers[1].addressV);
}

void StateContract::applyTransformsTo(IDirect3DDevice9* dev) const {
    dev->SetTransform(D3DTS_WORLD, (const D3DMATRIX*)&world);
    dev->SetTransform(D3DTS_VIEW, (const D3DMATRIX*)&view);
    dev->SetTransform(D3DTS_PROJECTION, (const D3DMATRIX*)&projection);
}

#ifdef _DEBUG
bool StateContract::validate(IDirect3DDevice9* dev, const char* context) const {
    StateContract actual;
    actual.captureFrom(dev);

    bool valid = true;
    char buf[256];

    // Check depth state
    if (actual.zEnable != zEnable) {
        snprintf(buf, sizeof(buf), "[%s] StateContract mismatch: zEnable expected %lu, got %lu", context, zEnable, actual.zEnable);
        LOG::logline(buf);
        valid = false;
    }
    if (actual.zWriteEnable != zWriteEnable) {
        snprintf(buf, sizeof(buf), "[%s] StateContract mismatch: zWriteEnable expected %lu, got %lu", context, zWriteEnable, actual.zWriteEnable);
        LOG::logline(buf);
        valid = false;
    }
    if (actual.zFunc != zFunc) {
        snprintf(buf, sizeof(buf), "[%s] StateContract mismatch: zFunc expected %lu, got %lu", context, zFunc, actual.zFunc);
        LOG::logline(buf);
        valid = false;
    }

    // Check blending state
    if (actual.alphaBlendEnable != alphaBlendEnable) {
        snprintf(buf, sizeof(buf), "[%s] StateContract mismatch: alphaBlendEnable expected %lu, got %lu", context, alphaBlendEnable, actual.alphaBlendEnable);
        LOG::logline(buf);
        valid = false;
    }
    if (actual.srcBlend != srcBlend) {
        snprintf(buf, sizeof(buf), "[%s] StateContract mismatch: srcBlend expected %lu, got %lu", context, srcBlend, actual.srcBlend);
        LOG::logline(buf);
        valid = false;
    }
    if (actual.destBlend != destBlend) {
        snprintf(buf, sizeof(buf), "[%s] StateContract mismatch: destBlend expected %lu, got %lu", context, destBlend, actual.destBlend);
        LOG::logline(buf);
        valid = false;
    }

    // Check alpha test state
    if (actual.alphaTestEnable != alphaTestEnable) {
        snprintf(buf, sizeof(buf), "[%s] StateContract mismatch: alphaTestEnable expected %lu, got %lu", context, alphaTestEnable, actual.alphaTestEnable);
        LOG::logline(buf);
        valid = false;
    }
    if (actual.alphaFunc != alphaFunc) {
        snprintf(buf, sizeof(buf), "[%s] StateContract mismatch: alphaFunc expected %lu, got %lu", context, alphaFunc, actual.alphaFunc);
        LOG::logline(buf);
        valid = false;
    }
    if (actual.alphaRef != alphaRef) {
        snprintf(buf, sizeof(buf), "[%s] StateContract mismatch: alphaRef expected %lu, got %lu", context, alphaRef, actual.alphaRef);
        LOG::logline(buf);
        valid = false;
    }

    // Check culling and fog
    if (actual.cullMode != cullMode) {
        snprintf(buf, sizeof(buf), "[%s] StateContract mismatch: cullMode expected %lu, got %lu", context, cullMode, actual.cullMode);
        LOG::logline(buf);
        valid = false;
    }
    if (actual.fogEnable != fogEnable) {
        snprintf(buf, sizeof(buf), "[%s] StateContract mismatch: fogEnable expected %lu, got %lu", context, fogEnable, actual.fogEnable);
        LOG::logline(buf);
        valid = false;
    }

    return valid;
}

#endif

void FixedFunctionShader::startRecording() {
    // Capture device state at recording start using StateContract
    trackDeviceRead("StateContract::captureFrom(preRecording)");
    preRecordingContract.captureFrom((IDirect3DDevice9*)device);

    // Seed the state tracker with current device state BEFORE suppression takes effect.
    // This ensures the tracker has complete state (baseline + MW changes during recording).
    // Without this, tracker only has values MW explicitly set during recording.
    if (ImGuiManager::GetStateSuppressionEnabled()) {
        g_cmdBufferSet.stateTracker().seedFromDevice((IDirect3DDevice9*)device);
    }

    // Reset HLSL caches for new recording session
    resetHLSLCaches();

    currentRecordedCalls().reserve(4000);  // Pre-allocate (already cleared by rotateRecordingBuffer)
    samplerCache.clear();  // Clear sampler cache for new frame
    bboxLookup.clear();  // Clear for new frame (populated during recording)
    // NOTE: Do NOT clear recordMW here - it's populated by inspectIndexedPrimitive()
    // BEFORE startRecording() is called. recordMW is cleared at end of renderStage1/2.
    lastLightState.reset();  // Clear LightState cache for new recording

    // Clear pointer-keyed caches on cell transitions.
    // Morrowind reuses freed VB/IB/texture pointers for different data after cell changes,
    // so pointer-keyed caches (bboxCache, textureSuffixResolutionCache, blacklist) return stale data.
    {
        static bool lastWasExterior = false;
        static void* lastPlayerCell = nullptr;

        bool isExterior = MWBridge::get()->IsExterior();
        void* currentCell = MWBridge::get()->getPlayerCell();

        if (isExterior != lastWasExterior) {
            // Interior ↔ exterior transition: clear geometry caches
            // textureSuffixResolutionCache is now evict-on-release (no bulk clear needed)
            bboxCache.clear();
            softwareOcclusionCuller.clearBlacklist();
            lastWasExterior = isExterior;
        } else if (currentCell != lastPlayerCell) {
            // Any cell change (exterior-to-exterior, interior-to-interior): clear geometry caches
            bboxCache.clear();
            softwareOcclusionCuller.clearBlacklist();
        }

        lastPlayerCell = currentCell;
    }

    // Clear lights from previous frame (simple approach, no persistence)
    DistantLand::sceneLights.clear();
    DistantLand::sceneLightIndexMap.clear();
    isRecording = true;
    isReplaying = false;
    recordingCompletedThisFrame = false;  // Allow recording to proceed

    // Clear CPU depth buffer for new frame (replaces GPU Hi-Z readback)
    softwareOcclusionCuller.clear();

    // Capture view/projection matrices once at start of recording directly into FrameBuffer
    // Note: World transforms are captured per-call in each RenderedState
    auto& fb = frameBuffers[recordingBuffer];
    trackDeviceRead("GetTransform(VIEW,PROJ)");
    device->GetTransform(D3DTS_VIEW, &fb.view);
    device->GetTransform(D3DTS_PROJECTION, &fb.proj);
    fb.shadowViewproj[0] = DistantLand::s_staging.smViewproj[0];
    fb.shadowViewproj[1] = DistantLand::s_staging.smViewproj[1];
    fb.state = BufferState::Recording;
    fb.valid = true;

    currentPhase = PipelinePhase::Recording;
    LOG::logline("HLSL Recording: Started recording render dispatches");
}

void FixedFunctionShader::stopRecordingAndReplay() {
    if (!isRecording) {
        return;
    }

    isRecording = false;

    // Capture device state NOW — this is Morrowind's last mesh state (correct end-of-Scene-0 state).
    // We restore this after replay instead of preRecordingContract (which was the FIRST mesh's state
    // and could have different alpha test/blend settings that corrupt the sky).
    StateContract endState;
    endState.captureFrom((IDirect3DDevice9*)device);


    // Batch-warm suffix cache — resolve all unique textures and pre-load suffix files
    // before prepare/replay, so neither stalls on hash computation or disk I/O.
    {
        MGE_ZoneScopedN("BatchWarmSuffixCache");
        auto& recCalls = frameBuffers[recordingBuffer].recordedCalls;
        std::unordered_set<IDirect3DTexture9*> seen;
        for (const auto& call : recCalls) {
            if (call.rs.texture && seen.insert(call.rs.texture).second) {
                TextureSuffix::warmCache((IDirect3DDevice9*)device, call.rs.texture);
            }
        }
    }

    // Phase 2b: Prepare shader keys (bbox + occluders already done in executeHiZCulling)
    prepareRecordedCalls(recordingBuffer);

    // Phase 3: Replay all prepared calls (Hi-Z already built by executeHiZCulling)
    replayRecordedCalls(0);

    // Data is already in frameBuffers[recordingBuffer].recordedCalls (recorded directly there)
    // No move or clear needed — buffer ownership transfers via rotateRecordingBuffer()

    // Clean up HLSL-only texture slots to prevent DXVK descriptor bloat
    // Use raw SetTexture to truly unbind (setCachedTexture substitutes default textures)
    // Slots 0-1 are used by Morrowind normally, don't touch them
    // Slots 2-5 are HLSL-specific (paramH, paramX, shadow, lightData)
    for (int i = 2; i < 6; i++) {
        device->SetTexture(i, NULL);
        textureCache.updateCache(i, nullptr);
        textureCache.textureValid[i] = false;
    }
    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);

    // Restore Morrowind's end-of-Scene-0 state (last mesh state, not first mesh state).
    // This undoes any state changes from replay/HLSL rendering, and also cleans up
    // leaked state from previous frame's Scene 1/2 immediate rendering.
    endState.applyRenderStatesTo((IDirect3DDevice9*)device);


    // Reset state to allow new recording sessions
    // Note: isRecording stays false until next startRecording() call
    isReplaying = false;
}

// Step 2: Submit recording to cull thread for async prepare (called before renderStage1)
void FixedFunctionShader::finalizeBatchAndSubmitCull() {
    // Handle dump request (same logic as finalizeBatchAndReplay)
    if (dumpRequested) {
        if (recordingEnabled) {
            auto& recCalls = currentRecordedCalls();
            LOG::logline("Frame dump: Dumping %d recorded calls (batch)", recCalls.size());
            StatusOverlay::setStatus("Frame dump: Batch complete");

            char logline[512];
            for (size_t i = 0; i < recCalls.size(); ++i) {
                const auto& call = recCalls[i];
                snprintf(logline, sizeof(logline), "Call %zu: texture=0x%p, vb=0x%p, ib=0x%p, hasShadows=%d, hasParamH=%d (batch)",
                         i, call.rs.texture, call.rs.vb, call.rs.ib,
                         call.sk.hasShadows, call.sk.hasParamH);
                LOG::logline(logline);

                if (call.rs.texture) {
                    D3DSURFACE_DESC desc;
                    if (SUCCEEDED(call.rs.texture->GetLevelDesc(0, &desc))) {
                        snprintf(logline, sizeof(logline), "  Texture: %dx%d, format=%d",
                                 desc.Width, desc.Height, desc.Format);
                        LOG::logline(logline);
                    }
                }
            }
        } else {
            LOG::logline("Frame dump: Immediate dump complete");
            StatusOverlay::setStatus("Frame dump: Immediate complete");
        }
        dumpRequested = false;
    }

    if (!recordingEnabled || !isRecording || currentRecordedCalls().empty()) {
        // Nothing to cull — just mark Scene 0 as done
        isRecording = false;
        recordingCompletedThisFrame = true;
        return;
    }

    isRecording = false;

    // Capture Morrowind's end-of-Scene-0 device state for restoration after replay
    if (ImGuiManager::GetStateSuppressionEnabled()) {
        // Suppression ON: use tracked state (device has HLSL values, not MW values)
        postRecordingContract.captureFromTracker(g_cmdBufferSet.stateTracker());
    } else {
        // Suppression OFF: use device directly (MW calls forwarded, device has correct values)
        postRecordingContract.captureFrom((IDirect3DDevice9*)device);
    }


    // Diagnostic: increment frame counter for cache miss/hit logging
    ++hlslDiagFrameCounter;

    // After frame 5, dump unused precached variants
    if (hlslDiagFrameCounter == 6) {
        LOG::logline("-- PRECACHE HIT REPORT: %d keys hit out of %d cached", (int)diagHitKeys.size(), (int)cacheHLSLShaders.size());
        AcquireSRWLockShared(&hlslCacheLock);
        for (const auto& entry : cacheHLSLShaders) {
            const auto& k = entry.first;
            bool hit = diagHitKeys.count(k) > 0;
            char buf[256];
            snprintf(buf, sizeof(buf),
                "%s lm=%d lit=%d vc=%d vm=%d hl=%d skin=%d fog=%d uv=%d stg=%d shd=%d det=%d dp=%d ph=%d px=%d gr=%d bm=%d tg=%d",
                hit ? "HIT " : "UNUSED",
                (int)k.lightMode, (int)k.useLighting, (int)k.vertexColour,
                (int)k.vertexMaterial, (int)k.heavyLighting,
                (int)k.usesSkinning, (int)k.fogMode, (int)k.uvSets,
                (int)k.activeStages,
                (int)k.hasShadows, (int)k.hasDetail, (int)k.hasDiffParam,
                (int)k.hasParamH, (int)k.hasParamX, (int)k.hasGrass,
                (int)k.usesBumpmap, (int)k.usesTexgen);
            LOG::logline("%s", buf);
        }
        ReleaseSRWLockShared(&hlslCacheLock);
        diagHitKeys.clear();
    }

    // Batch-warm suffix cache on main thread before cull thread gets the buffer.
    // Resolves all unique textures and pre-loads suffix files so that
    // computeShaderKeyWithSuffixes on the cull thread never hits expensive fallbacks,
    // and replay never stalls on hash computation or disk I/O.
    {
        MGE_ZoneScopedN("BatchWarmSuffixCache");
        auto& recCalls = frameBuffers[recordingBuffer].recordedCalls;
        std::unordered_set<IDirect3DTexture9*> seen;
        for (const auto& call : recCalls) {
            if (call.rs.texture && seen.insert(call.rs.texture).second) {
                TextureSuffix::warmCache((IDirect3DDevice9*)device, call.rs.texture);
            }
        }
    }

    // Prepare recorded calls inline on main thread
    int buf = recordingBuffer;
    prepareRecordedCalls(buf);
    frameBuffers[buf].state = BufferState::ReadyToRender;

    recordingCompletedThisFrame = true;
}

// Step 2: Wait for cull completion and replay (called after renderStageBlend)
void FixedFunctionShader::waitCullAndReplay() {
    if (!recordingEnabled) return;

    auto& fb = frameBuffers[recordingBuffer];
    if (fb.state != BufferState::ReadyToRender) {
        return;  // Nothing prepared
    }
    fb.state = BufferState::ReadyToRender;

    // Replay all prepared calls
    replayRecordedCalls(0);

    // Clean up HLSL-only texture slots to prevent DXVK descriptor bloat
    for (int i = 2; i < 6; i++) {
        device->SetTexture(i, NULL);
        textureCache.updateCache(i, nullptr);
        textureCache.textureValid[i] = false;
    }
    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);

    // Restore Morrowind's end-of-Scene-0 state
    postRecordingContract.applyRenderStatesTo((IDirect3DDevice9*)device);


    isReplaying = false;

    // Reset HLSL caches after rendering session completes
    resetHLSLCaches();
}

// Call this when HLSL rendering session is complete to trigger replay
void FixedFunctionShader::finalizeBatchAndReplay(int sceneCount) {
    if (recordingEnabled) {
        if (sceneCount == 0) {
            // Scene 0: recording continues through Scene 1/2
            // capturePostRecordingState() already saved device state
            // finalizeAndRender() will do the actual prepare+render later
        } else if (sceneCount == 1) {
            // Scene 1 (particles): log recording count, continue recording
            auto& fb = frameBuffers[recordingBuffer];
            LOG::logline("Scene 1 (particles) recording: %d calls", fb.recordedCallsScene1.size());
        } else if (sceneCount >= 2) {
            // Scene 2+ (hands): log recording count, continue recording
            auto& fb = frameBuffers[recordingBuffer];
            LOG::logline("Scene %d recording: scene2 buffer=%d calls", sceneCount, fb.recordedCallsScene2.size());
        }
        // sceneCount < 0: pre-scene, ignore
    } else {
        // When recording disabled, still clear calls after potential dump
        auto& fb = frameBuffers[recordingBuffer];
        fb.recordedCalls.clear();
        fb.recordedCallsScene1.clear();
        fb.recordedCallsScene2.clear();
        resetHLSLCaches();
    }
}

// capturePostRecordingState - Capture MW device state at end of Scene 0
// Recording continues through Scene 1/2; this just saves what we need to restore later.
void FixedFunctionShader::capturePostRecordingState() {
    if (!isRecording) return;

    auto& fb = frameBuffers[recordingBuffer];

    if (ImGuiManager::GetStateSuppressionEnabled()) {
        // Suppression ON: use tracked state (device has HLSL values, not MW values)
        fb.stateContract.captureFromTracker(g_cmdBufferSet.stateTracker());
    } else {
        // Suppression OFF: use device directly (MW calls forwarded, device has correct values)
        trackDeviceRead("StateContract::captureFrom(postRecording)");
        fb.stateContract.captureFrom((IDirect3DDevice9*)device);
    }

#ifdef _DEBUG
    // Validation: compare tracked vs device (only useful when suppression is OFF)
    // When suppression is ON, device has HLSL values so comparison is meaningless.
    if (!ImGuiManager::GetStateSuppressionEnabled()) {
        StateContract deviceState;
        deviceState.captureFrom((IDirect3DDevice9*)device);
        // Log any mismatches (tracked should equal device when suppression is off)
        validateTrackedVsDevice(g_cmdBufferSet.stateTracker(), deviceState, "capturePostRecordingState");
    }
#endif

}

// restorePostRecordingState - Clean up device state for Scene 1/2 after HLSL recording
// Called at EndScene 0 in deferred HLSL mode so particles/hands render correctly.
// NOTE: State restoration for suppression mode now happens in executeGpuPhase().
void FixedFunctionShader::restorePostRecordingState() {
    if (!isRecording) return;

    // Stop recording - Scene 0 calls are captured, Scene 1/2 will use immediate path
    // Recording will be finalized at finalizeAndRender
    isRecording = false;

    // Clean up HLSL-only texture slots
    for (int i = 2; i < 6; i++) {
        device->SetTexture(i, NULL);
        textureCache.updateCache(i, nullptr);
        textureCache.textureValid[i] = false;
    }

    // Clear shaders so Scene 1/2 uses fixed-function or gets fresh shader setup
    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);

    // Reset all HLSL caches - shadow matrices, material cache, shader LRU
    // This ensures Scene 1/2 gets fresh state, not stale cached values from Scene 0
    resetHLSLCaches();
}

// finalizeAndRender - Complete prepare+render phase at frame finalize point (BeginScene UI)
// Stops recording, runs cull/prepare, then all GPU stages, then replay.
void FixedFunctionShader::finalizeAndRender(DLContext* frameCtx, bool waterSeen) {
    MGE_ZoneScopedN("Frame_PrepareAndRender");

    // Handle dump request
    if (dumpRequested) {
        auto& recCalls = currentRecordedCalls();
        LOG::logline("Frame dump: Dumping %d recorded calls (finalize)", recCalls.size());
        StatusOverlay::setStatus("Frame dump: Finalize complete");
        char logline[512];
        for (size_t i = 0; i < recCalls.size(); ++i) {
            const auto& call = recCalls[i];
            snprintf(logline, sizeof(logline), "Call %zu: texture=0x%p, vb=0x%p, ib=0x%p (finalize)",
                     i, call.rs.texture, call.rs.vb, call.rs.ib);
            LOG::logline(logline);
        }
        dumpRequested = false;
    }

    // === STOP RECORDING ===
    if (!recordingEnabled || !isRecording) {
        // Nothing recorded — still need to run GPU stages
        isRecording = false;
        recordingCompletedThisFrame = true;

        // GPU stages even without HLSL recording
        DistantLand::renderStage0GPU(frameCtx);
        DistantLand::renderStage1(frameCtx);
        DistantLand::renderStageBlend(frameCtx);
        if (waterSeen) {
            DistantLand::renderStageWater(frameCtx);
        }
        DistantLand::renderStage2(frameCtx);
        return;
    }

    // Log phase summary: FrameCapture + Recording device call counts
    {
        auto& fc = frameCaptureGpuCalls;
        auto& rc = recordingGpuCalls;
        bool hasViolation = (fc.deviceSubmits > 0 || rc.deviceSubmits > 0 || rc.deviceWrites > 0);
        if (hasViolation) {
            LOG::logline("!! PHASE: FrameCapture={reads:%d writes:%d submits:%d} Recording={reads:%d writes:%d submits:%d}",
                fc.deviceReads, fc.deviceWrites, fc.deviceSubmits,
                rc.deviceReads, rc.deviceWrites, rc.deviceSubmits);
        }
    }
    currentPhase = PipelinePhase::Idle;

    isRecording = false;
    recordingCompletedThisFrame = true;

    // Diagnostic: increment frame counter for cache miss/hit logging
    ++hlslDiagFrameCounter;

    // After frame 5, dump unused precached variants
    if (hlslDiagFrameCounter == 6) {
        LOG::logline("-- PRECACHE HIT REPORT: %d keys hit out of %d cached", (int)diagHitKeys.size(), (int)cacheHLSLShaders.size());
        AcquireSRWLockShared(&hlslCacheLock);
        for (const auto& entry : cacheHLSLShaders) {
            const auto& k = entry.first;
            bool hit = diagHitKeys.count(k) > 0;
            char buf[256];
            snprintf(buf, sizeof(buf),
                "%s lm=%d lit=%d vc=%d vm=%d hl=%d skin=%d fog=%d uv=%d stg=%d shd=%d det=%d dp=%d ph=%d px=%d gr=%d bm=%d tg=%d",
                hit ? "HIT " : "UNUSED",
                (int)k.lightMode, (int)k.useLighting, (int)k.vertexColour,
                (int)k.vertexMaterial, (int)k.heavyLighting,
                (int)k.usesSkinning, (int)k.fogMode, (int)k.uvSets,
                (int)k.activeStages,
                (int)k.hasShadows, (int)k.hasDetail, (int)k.hasDiffParam,
                (int)k.hasParamH, (int)k.hasParamX, (int)k.hasGrass,
                (int)k.usesBumpmap, (int)k.usesTexgen);
            LOG::logline("%s", buf);
        }
        ReleaseSRWLockShared(&hlslCacheLock);
        diagHitKeys.clear();
    }

    // Batch-warm suffix cache before prepare
    {
        MGE_ZoneScopedN("BatchWarmSuffixCache");
        auto& recCalls = frameBuffers[recordingBuffer].recordedCalls;
        std::unordered_set<IDirect3DTexture9*> seen;
        for (const auto& call : recCalls) {
            if (call.rs.texture && seen.insert(call.rs.texture).second) {
                TextureSuffix::warmCache((IDirect3DDevice9*)device, call.rs.texture);
            }
        }
    }

    // Diagnostic: log scene breakdown once
    {
        static bool loggedOnce = false;
        if (!loggedOnce) {
            auto& fb = frameBuffers[recordingBuffer];
            int s0 = (int)fb.recordedCalls.size();
            int s1 = (int)fb.recordedCallsScene1.size();
            int s2 = (int)fb.recordedCallsScene2.size();
            int m0 = 0, m1 = 0, m2 = 0;
            for (const auto& m : fb.recordMW) {
                if (m.sceneNum == 0) m0++;
                else if (m.sceneNum == 1) m1++;
                else m2++;
            }
            LOG::logline(">> finalizeAndRender: HLSL scene0=%d, scene1=%d, scene2=%d; recordMW scene0=%d, scene1=%d, scene2=%d",
                         s0, s1, s2, m0, m1, m2);
            loggedOnce = true;
        }
    }

    // === PREPARE (CPU) ===
    currentPhase = PipelinePhase::CpuPrepare;
    {
        MGE_ZoneScopedN("Frame_Prepare");

        // Prepare recorded calls inline on main thread
        int buf = recordingBuffer;
        prepareRecordedCalls(buf);
        frameBuffers[buf].state = BufferState::ReadyToRender;
    }

    // === RENDER (GPU) ===
    currentPhase = PipelinePhase::GpuRender;
    {
        auto& fb = frameBuffers[recordingBuffer];

        // Store DLContext and waterSeen in FrameBuffer for per-frame isolation
        fb.dlContext = *frameCtx;
        fb.waterSeen = waterSeen;

        // Capture MWBridge state for postProcess (render thread safe)
        {
            auto mwBridge = MWBridge::get();
            auto& ppd = fb.postProcessData;
            ppd.frameTime = mwBridge->frameTime();
            ppd.simulationTime = mwBridge->simulationTime();
            ppd.waterLevel = mwBridge->CellHasWater() ? mwBridge->WaterLevel() : -1e9f;
            ppd.isMenu = mwBridge->IsMenu();
            ppd.isInterior = !mwBridge->CellHasWeather();
            ppd.isUnderwater = mwBridge->IsUnderwater(frameCtx->eyePos.z);

            int envFlags = 0;
            if (!mwBridge->CellHasWeather()) envFlags |= 1;
            if (mwBridge->IsExterior()) envFlags |= 2;
            if (mwBridge->IntLikeExterior()) envFlags |= 4;
            if (ppd.isUnderwater) envFlags |= 8; else envFlags |= 16;
            if (frameCtx->sunVis >= 0.001) envFlags |= 32; else envFlags |= 64;
            ppd.envFlags = envFlags;
        }

        if (g_renderThread && g_renderThread->isRunning()) {
            RenderThread::SceneWork work;
            work.type = RenderThread::WorkType::RenderFullFrame;
            work.bufferIndex = recordingBuffer;
            g_renderThread->submitWork(std::move(work), false);  // wait=false — async for 3b
        } else {
            executeGpuPhase(recordingBuffer);
        }
    }

    currentPhase = PipelinePhase::Idle;
}

void FixedFunctionShader::executeGpuPhase(int bufferIndex) {
    MGE_ZoneScopedN("Frame_Render");

    auto& fb = frameBuffers[bufferIndex];
    DLContext* frameCtx = &fb.dlContext;
    bool waterSeen = fb.waterSeen;

    // Offscreen/PreScene buffer replay is only needed when those draws are suppressed
    // from the device (future: Scene0 suppression for render thread overlap).
    // Currently, offscreen and prescene draws go direct to device (not suppressed),
    // so replaying them here would double-render and leave wrong RT/state.
    // if (ImGuiManager::GetCmdBufferReplay() && g_cmdBufferSet[CmdStage::Offscreen].size() > 0) {
    //     g_cmdBufferSet[CmdStage::Offscreen].replay(device);
    // }
    // if (ImGuiManager::GetCmdBufferReplay()) {
    //     g_cmdBufferSet[CmdStage::PreScene].replay(device);
    // }

    // Stage 0 GPU: shadow map, distant land, sky, water reflection, wave sim
    DistantLand::renderStage0GPU(frameCtx, &fb);

    // Update shadow VP in frame buffer — renderShadowMap just computed current frame's
    // shadow matrices and wrote them back to s_staging. The stale previous-frame values
    // captured at startRecording() would cause shadow shaking on camera movement.
    {
        fb.shadowViewproj[0] = DistantLand::s_staging.smViewproj[0];
        fb.shadowViewproj[1] = DistantLand::s_staging.smViewproj[1];
    }

    // Stage 1: grass, shadow overlay, depth
    DistantLand::renderStage1(frameCtx, &fb);

    // Blend close objects over distant land
    DistantLand::renderStageBlend(frameCtx, &fb);

    // Build HLSL replay into command buffer, then replay it
    // Scene 0/1/2 all go through HLSL replay
    transitionTo(PhaseTransition::ReplayEntry);
    fb.hlslCmds.clear();

    // Scene 0: World geometry
    replayRecordedCalls(0, &fb.hlslCmds);

    // Scene 1: Particles (alpha sorted, blended)
    replayRecordedCalls(1, &fb.hlslCmds);

    // Scene 2: Hands (skinned, after Z-clear in MW but we replay without clear)
    replayRecordedCalls(2, &fb.hlslCmds);

    fb.hlslCmds.replay(device);
    transitionTo(PhaseTransition::ReplayExit);

    // Water surface AFTER replay — refraction samples backbuffer which needs scene content
    if (waterSeen) {
        DistantLand::renderStageWater(frameCtx);
    }

    // NOTE: renderStage2 (depth for Scene 2 hands) runs at EndScene(1/2) in mged3d8device.cpp
    // It cannot run here because Scene 1/2 haven't happened yet at EndScene(0)

    // NOTE: postProcess moved to UI BeginScene to avoid RT/DS state corruption for Scene 1/2

    // Clean up HLSL-only texture slots to prevent DXVK descriptor bloat
    for (int i = 2; i < 6; i++) {
        device->SetTexture(i, NULL);
        textureCache.updateCache(i, nullptr);
        textureCache.textureValid[i] = false;
    }
    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);

    // Restore Morrowind's end-of-Scene-0 state for Scene 1/2
    if (ImGuiManager::GetStateSuppressionEnabled()) {
        // Suppression ON: replay full Scene0 buffer (materials, lights, textures, etc.)
        // StateContract only has a subset; Scene0 buffer has complete recorded state.
        g_cmdBufferSet[CmdStage::Scene0].replayStateOnly((IDirect3DDevice9*)device);
    } else {
        // Suppression OFF: StateContract captured from device is accurate
        fb.stateContract.applyTo((IDirect3DDevice9*)device);
    }


    isReplaying = false;
    resetHLSLCaches();
}

// Full deferred GPU phase at UI BeginScene - all scenes recorded, now render everything
// Flow: Depth(Scene 0) → Depth(Scene 2) → Replay(0) → Replay(1) → Replay(2)
void FixedFunctionShader::finalizeAndRenderAllScenes(DLContext* frameCtx, bool waterSeen) {
    MGE_ZoneScopedN("finalizeAndRenderAllScenes");

    if (!recordingEnabled) return;

    auto& fb = frameBuffers[recordingBuffer];

    LOG::logline("finalizeAndRenderAllScenes: scene0=%d, scene1=%d, scene2=%d",
        (int)fb.recordedCalls.size(), (int)fb.recordedCallsScene1.size(), (int)fb.recordedCallsScene2.size());

    // === SAVE UI TRANSFORMS ===
    // MW set up UI view/projection before calling BeginScene. We need to restore these
    // after 3D rendering so HUD draws correctly.
    D3DXMATRIX savedUIView, savedUIProj;
    device->GetTransform(D3DTS_VIEW, &savedUIView);
    device->GetTransform(D3DTS_PROJECTION, &savedUIProj);

    // === RESTORE ENDSCENE(0) STATE ===
    // State was captured at EndScene(0), but MW ran Scene 1/2 since then.
    // Restore to match baseline expectations for renderStage0GPU.
    fb.stateContract.applyTo((IDirect3DDevice9*)device);

    // === PREPARE ALL SCENES ===
    {
        MGE_ZoneScopedN("Prepare All Scenes");
        prepareRecordedCalls(recordingBuffer);  // This now prepares Scene 0, 1, and 2
        fb.state = BufferState::ReadyToRender;
    }

    // Store context in FrameBuffer
    fb.dlContext = *frameCtx;
    fb.waterSeen = waterSeen;

    // === DEPTH PASSES ===
    // Render depth for Scene 0 (world) and Scene 2 (hands)
    // Scene 1 (particles) skipped - they're alpha blended, no depth write
    {
        MGE_ZoneScopedN("Depth Passes");

        // Use renderStage1 for Scene 0 depth (includes shadows, distant land setup)
        DistantLand::renderStage1(frameCtx, &fb);

        // Render Scene 2 (hands) depth into the depth texture
        // Filter for sceneNum >= 2 only
        DistantLand::renderStage2(frameCtx, &fb);
    }

    // === REPLAY ALL SCENES ===
    isReplaying = true;
    {
        MGE_ZoneScopedN("Replay All Scenes");

        // Blend close objects over distant land (before replay)
        DistantLand::renderStageBlend(frameCtx, &fb);

        transitionTo(PhaseTransition::ReplayEntry);

        // Replay Scene 0 (world)
        replayRecordedCalls(0, nullptr);

        // Replay Scene 1 (particles) - depth tested against Scene 0
        replayRecordedCalls(1, nullptr);

        // Replay Scene 2 (hands) - rendered on top
        replayRecordedCalls(2, nullptr);

        transitionTo(PhaseTransition::ReplayExit);
    }

    // Water surface AFTER replay
    if (waterSeen) {
        DistantLand::renderStageWater(frameCtx);
    }

    // Clean up HLSL state (textures, shaders)
    for (int i = 2; i < 6; i++) {
        device->SetTexture(i, NULL);
        textureCache.updateCache(i, nullptr);
        textureCache.textureValid[i] = false;
    }
    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);

    // === RESTORE UI STATE ===
    // MW set up UI view/projection before calling BeginScene. Restore them so HUD draws correctly.
    device->SetTransform(D3DTS_VIEW, &savedUIView);
    device->SetTransform(D3DTS_PROJECTION, &savedUIProj);

    // MW cleared depth buffer before BeginScene, but we filled it with 3D scene.
    // Clear it again so UI draws aren't depth-tested against 3D geometry.
    device->Clear(0, NULL, D3DCLEAR_ZBUFFER, 0, 1.0f, 0);

    isReplaying = false;
    resetHLSLCaches();
}

// Replay Scene 1 and Scene 2 at UI BeginScene (after they've been recorded)
// Called separately from executeGpuPhase because Scene 1/2 are recorded AFTER EndScene(0)
void FixedFunctionShader::replayScene1And2(FrameBuffer* fb) {
    if (!fb) return;

    // Prepare Scene 1/2 if not already done (shader keys, bins)
    // Scene 0 was prepared in executeGpuPhase, but we need to prepare 1/2 now
    {
        MGE_ZoneScopedN("Prepare Scene 1/2");
        for (auto& call : fb->recordedCallsScene1) {
            if (!call.prepared) {
                call.sk = computeShaderKeyWithSuffixes(&call.rs, &call.frs, call.lightrs.get());
                call.prepared = true;
                call.shouldRender = true;

                if (call.sk.hasGrass)              call.bin = RenderBin::Grass;
                else if (call.sk.usesSkinning)     call.bin = RenderBin::Skinning;
                else if (call.rs.blendEnable)      call.bin = RenderBin::Blending;
                else if (call.rs.alphaTest)        call.bin = RenderBin::AlphaTested;
                else                               call.bin = RenderBin::Opaque;
            }
        }
        for (auto& call : fb->recordedCallsScene2) {
            if (!call.prepared) {
                call.sk = computeShaderKeyWithSuffixes(&call.rs, &call.frs, call.lightrs.get());
                call.prepared = true;
                call.shouldRender = true;

                if (call.sk.hasGrass)              call.bin = RenderBin::Grass;
                else if (call.sk.usesSkinning)     call.bin = RenderBin::Skinning;
                else if (call.rs.blendEnable)      call.bin = RenderBin::Blending;
                else if (call.rs.alphaTest)        call.bin = RenderBin::AlphaTested;
                else                               call.bin = RenderBin::Opaque;
            }
        }
    }

    // Skip if nothing to replay
    if (fb->recordedCallsScene1.empty() && fb->recordedCallsScene2.empty()) {
        return;
    }

    LOG::logline("replayScene1And2: scene1=%d, scene2=%d calls",
        (int)fb->recordedCallsScene1.size(), (int)fb->recordedCallsScene2.size());

    isReplaying = true;

    // Replay Scene 1 (particles)
    replayRecordedCalls(1, nullptr);

    // Replay Scene 2 (hands)
    replayRecordedCalls(2, nullptr);

    isReplaying = false;
}

// Scene lifecycle stubs for triple-buffered pipeline (Step 1: no-op, infrastructure only)
void FixedFunctionShader::markSceneStart(int sceneNum, bool isUI) {
    // Will be used in Step 2+ to track scene boundaries within a FrameBuffer
    // For now, processAsyncCompletions is called from renderMorrowindHLSL_Internal
    if (sceneNum == 0 && !isUI) {
        processAsyncCompletions();
    }
}

void FixedFunctionShader::markSceneEnd() {
    // Will be used in Step 2+ to finalize scene boundaries
}

// Scene handover debugging - logs device state for particle bug investigation
void FixedFunctionShader::logSceneHandoverState(const char* label) {
    if (!ImGuiManager::GetHandoverLogging()) return;

    // Capture full StateContract
    StateContract state;
    state.captureFrom((IDirect3DDevice9*)device);

    // Point sprite render states (not in StateContract - particle-specific)
    float pointSize;
    DWORD pointSpriteEnable, pointScaleEnable;
    float pointScaleA, pointScaleB, pointScaleC;
    device->GetRenderState(D3DRS_POINTSIZE, (DWORD*)&pointSize);
    device->GetRenderState(D3DRS_POINTSPRITEENABLE, &pointSpriteEnable);
    device->GetRenderState(D3DRS_POINTSCALEENABLE, &pointScaleEnable);
    device->GetRenderState(D3DRS_POINTSCALE_A, (DWORD*)&pointScaleA);
    device->GetRenderState(D3DRS_POINTSCALE_B, (DWORD*)&pointScaleB);
    device->GetRenderState(D3DRS_POINTSCALE_C, (DWORD*)&pointScaleC);

    // FVF
    DWORD fvf;
    device->GetFVF(&fvf);

    // Texture on stage 0
    IDirect3DBaseTexture9* tex0 = nullptr;
    device->GetTexture(0, &tex0);
    if (tex0) tex0->Release();

    // Vertex/pixel shader
    IDirect3DVertexShader9* vs = nullptr;
    IDirect3DPixelShader9* ps = nullptr;
    device->GetVertexShader(&vs);
    device->GetPixelShader(&ps);
    if (vs) vs->Release();
    if (ps) ps->Release();

    LOG::logline("== HANDOVER [%s] ==", label);
    LOG::logline("  Z: enable=%lu write=%lu func=%lu | Blend: %lu src=%lu dst=%lu",
        state.zEnable, state.zWriteEnable, state.zFunc,
        state.alphaBlendEnable, state.srcBlend, state.destBlend);
    LOG::logline("  AlphaTest: %lu func=%lu ref=%lu | Cull=%lu Fog=%lu",
        state.alphaTestEnable, state.alphaFunc, state.alphaRef,
        state.cullMode, state.fogEnable);
    LOG::logline("  World: [%.2f,%.2f,%.2f,%.2f] ...",
        state.world._11, state.world._12, state.world._13, state.world._14);
    LOG::logline("  View: [%.2f,%.2f,%.2f,%.2f] ...",
        state.view._11, state.view._12, state.view._13, state.view._14);
    LOG::logline("  PointSprite: size=%.4f enable=%d scaleEnable=%d A=%.4f B=%.4f C=%.4f",
        pointSize, pointSpriteEnable, pointScaleEnable, pointScaleA, pointScaleB, pointScaleC);
    LOG::logline("  FVF=0x%08X tex0=%p VS=%p PS=%p", fvf, tex0, vs, ps);
}

// === Phase Transition Tracking ===
// Captures state at each boundary and validates against expected state

bool FixedFunctionShader::transitionTo(PhaseTransition trans) {
    int idx = static_cast<int>(trans);
    StateContract& captured = transitionState[idx];

    // Capture current device state
    captured.captureFrom((IDirect3DDevice9*)device);

    const char* name = getPhaseTransitionName(trans);

    // Log if handover logging is enabled
    if (ImGuiManager::GetHandoverLogging()) {
        LOG::logline("== TRANSITION [%s] ==", name);
        LOG::logline("  Z: enable=%lu write=%lu func=%lu | Blend: %lu src=%lu dst=%lu",
            captured.zEnable, captured.zWriteEnable, captured.zFunc,
            captured.alphaBlendEnable, captured.srcBlend, captured.destBlend);
        LOG::logline("  AlphaTest: %lu func=%lu ref=%lu | Cull=%lu Fog=%lu",
            captured.alphaTestEnable, captured.alphaFunc, captured.alphaRef,
            captured.cullMode, captured.fogEnable);
    }

    // Validate against expected state based on transition
    bool valid = true;

    switch (trans) {
        case PhaseTransition::GpuExit:
        case PhaseTransition::Scene1Entry: {
            // These must match RecordingExit - that's the invariant
            const StateContract& expected = transitionState[static_cast<int>(PhaseTransition::RecordingExit)];

            // Check critical render states (not transforms - those are set by replay)
            if (captured.zEnable != expected.zEnable ||
                captured.zWriteEnable != expected.zWriteEnable ||
                captured.alphaBlendEnable != expected.alphaBlendEnable ||
                captured.alphaTestEnable != expected.alphaTestEnable ||
                captured.cullMode != expected.cullMode ||
                captured.fogEnable != expected.fogEnable) {

                valid = false;
                LOG::logline("!! STATE MISMATCH at %s (expected RecordingExit state) !!", name);
                LOG::logline("  Z: expected enable=%lu write=%lu, got enable=%lu write=%lu",
                    expected.zEnable, expected.zWriteEnable,
                    captured.zEnable, captured.zWriteEnable);
                LOG::logline("  Blend: expected %lu, got %lu",
                    expected.alphaBlendEnable, captured.alphaBlendEnable);
                LOG::logline("  AlphaTest: expected %lu, got %lu",
                    expected.alphaTestEnable, captured.alphaTestEnable);
                LOG::logline("  Cull: expected %lu, got %lu",
                    expected.cullMode, captured.cullMode);
                LOG::logline("  Fog: expected %lu, got %lu",
                    expected.fogEnable, captured.fogEnable);
            }
            break;
        }
        default:
            // Other transitions just capture, no validation target yet
            break;
    }

    return valid;
}

const StateContract& FixedFunctionShader::getTransitionState(PhaseTransition trans) {
    return transitionState[static_cast<int>(trans)];
}

// Save all captured transition states as baselines
void FixedFunctionShader::saveCurrentAsBaseline() {
    LOG::logline("=== SAVING STATE BASELINES ===");
    for (int i = 0; i < static_cast<int>(PhaseTransition::Count); ++i) {
        PhaseTransition trans = static_cast<PhaseTransition>(i);
        saveStateBaseline(trans, transitionState[i]);
    }
    LOG::logline("=== BASELINES SAVED ===");
}

// Compare all captured states to saved baselines
void FixedFunctionShader::validateAgainstBaselines() {
    LOG::logline("=== VALIDATING AGAINST BASELINES ===");
    int mismatches = 0;
    for (int i = 0; i < static_cast<int>(PhaseTransition::Count); ++i) {
        PhaseTransition trans = static_cast<PhaseTransition>(i);
        if (!compareToBaseline(trans, transitionState[i])) {
            mismatches++;
        }
    }
    if (mismatches == 0) {
        LOG::logline("=== ALL STATES MATCH BASELINES ===");
    } else {
        LOG::logline("=== %d STATES DIFFER FROM BASELINES ===", mismatches);
    }
}

// Triple-buffer pipeline: cull pass (called by CullThread or inline on main thread)
void FixedFunctionShader::executeCullPass(int bufferIndex) {
    // Cull thread is idle - prepareRecordedCalls moved to main thread
    (void)bufferIndex;
}

void FixedFunctionShader::executeRenderPass(int bufferIndex) {
    executeGpuPhase(bufferIndex);
}

// Compare two LightStates for equality (to detect if we can reuse cached state)
bool FixedFunctionShader::compareLightStates(const LightState* a, const LightState* b) {
    // Fast path: check sizes first (cheapest comparison)
    if (a->lights.size() != b->lights.size() || a->active.size() != b->active.size()) {
        return false;
    }

    // Compare active array with memcmp (faster than loop for larger arrays)
    if (!a->active.empty() && memcmp(a->active.data(), b->active.data(), a->active.size() * sizeof(DWORD)) != 0) {
        return false;
    }

    // Compare globalAmbient as 4 DWORDs instead of 4 floats (avoids FP comparison)
    const DWORD* aAmb = reinterpret_cast<const DWORD*>(&a->globalAmbient);
    const DWORD* bAmb = reinterpret_cast<const DWORD*>(&b->globalAmbient);
    if (aAmb[0] != bAmb[0] || aAmb[1] != bAmb[1] || aAmb[2] != bAmb[2] || aAmb[3] != bAmb[3]) {
        return false;
    }

    return true;  // Same lighting state
}

void FixedFunctionShader::recordRenderCall(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, const ShaderKey& sk, int recordMWIdx) {
    {
        if (isReplaying) {
            return;  // Don't record during replay to avoid recursion
        }
    }

    // When recording is OFF and dump is requested, dump each call immediately
    if (!recordingEnabled && dumpRequested) {
        static int callIndex = 0;
        char logline[512];
        snprintf(logline, sizeof(logline), "Call %d: texture=0x%p, vb=0x%p, ib=0x%p (immediate)",
                 callIndex++, rs->texture, rs->vb, rs->ib);
        LOG::logline(logline);

        if (rs->texture) {
            D3DSURFACE_DESC desc;
            if (SUCCEEDED(rs->texture->GetLevelDesc(0, &desc))) {
                snprintf(logline, sizeof(logline), "  Texture: %dx%d, format=%d",
                         desc.Width, desc.Height, desc.Format);
                LOG::logline(logline);
            }
        }
    }

    // Extract lights for texture-based lighting system (all objects, even if culled)
    // Lights from culled objects can still illuminate visible geometry
    // Only process ACTIVE lights — lightrs->lights accumulates all SetLight() calls
    // across the session and never removes entries, so inactive/stale lights persist.
    for (DWORD id : lightrs->active) {
        auto lightIt = lightrs->lights.find(id);
        if (lightIt == lightrs->lights.end()) continue;
        const auto& light = lightIt->second;

        // Skip directional lights (sun) — only point lights for scene lighting
        if (light.type != D3DLIGHT_POINT) continue;

        // Distance filter: reject lights far from camera (stale interior lights after cell change)
        float dx = light.position.x - DistantLand::s_staging.eyePos.x;
        float dy = light.position.y - DistantLand::s_staging.eyePos.y;
        float dz = light.position.z - DistantLand::s_staging.eyePos.z;
        float distSq = dx*dx + dy*dy + dz*dz;
        const float MAX_LIGHT_DIST_SQ = 8192.0f * 8192.0f;
        if (distSq > MAX_LIGHT_DIST_SQ) continue;

        // Check if light already exists using O(1) hash map lookup
        auto mapIt = DistantLand::sceneLightIndexMap.find(id);

        if (mapIt == DistantLand::sceneLightIndexMap.end()) {
            // New light - add to scene
            size_t newIndex = DistantLand::sceneLights.size();
            DistantLand::SceneLight sl;
            sl.id = id;
            sl.position = light.position;
            sl.diffuse = light.diffuse;
            sl.falloff = light.falloff;  // (constant, linear, quadratic)
            sl.radius = DistantLand::computeLightRadius(sl.falloff.x, sl.falloff.y, sl.falloff.z);
            sl.isVisible = false;
            sl.lastSeenFrame = 0;
            DistantLand::sceneLights.push_back(sl);
            DistantLand::sceneLightIndexMap[id] = newIndex;  // Add to index map
        } else {
            // Update dynamic properties (color may pulse)
            size_t index = mapIt->second;
            DistantLand::sceneLights[index].diffuse = light.diffuse;
        }
    }

    // Reuse last LightState if identical to avoid allocation overhead
    // NOTE: Cannot use raw pointer comparison here! lightrs is a file-scope static in
    // mged3d8device.cpp — its address never changes, but Morrowind mutates it in-place
    // between draw calls via LightEnable(). Pointer equality always returns true,
    // causing ALL calls to share the first call's light config.
    std::shared_ptr<LightState> sharedLightState;
    {
        MGE_ZoneScopedN("record_LightStateCache");
        // Content comparison: reuse if lights haven't changed since last call
        if (lastLightState && compareLightStates(lastLightState.get(), lightrs)) {
            // Reuse existing shared_ptr (no allocation)
            sharedLightState = lastLightState;
        }
        // Different lights: create new copy
        else {
            sharedLightState = std::make_shared<LightState>(*lightrs);
            lastLightState = sharedLightState;
        }
    }

    // Warm suffix cache on main thread (device calls not safe off main thread)
    TextureSuffix::warmCache((IDirect3DDevice9*)device, rs->texture);

    // NOTE: DIP events are already logged in mged3d8device.cpp DrawIndexedPrimitive
    // before reaching this recording path. Do NOT log again here to avoid double-counting.

    {
        currentRecordedCalls().emplace_back(rs, frs, sharedLightState, sk, recordMWIdx);
        currentRecordedCalls().back().sceneNum = currentRecordingScene;

        // Immediately populate bboxLookup so depth pass can use current-frame bboxes
        // (executeHiZCulling runs BEFORE finalizeBatchAndReplay)
        const auto& call = currentRecordedCalls().back();
        if (call.hasBoundingBox) {
            VBIBKey key{call.rs.vb, call.rs.ib};
            bboxLookup[key] = {call.bboxMin, call.bboxMax};
        }
    }

    // Occluder selection and rasterization deferred to prepareRecordedCalls()
    // This removes the heaviest per-draw work from the recording hot path
}
