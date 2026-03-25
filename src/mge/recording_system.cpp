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
// Uses recording buffer for N-1 frame buffering
static auto& currentRecordedCalls() {
    auto& fb = FixedFunctionShader::getRecordingBuffer();
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

    currentRecordedCalls().reserve(4000);  // Pre-allocate (cleared at Present())
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
    // N-1: recording into recording buffer
    // Use MWStateTracker instead of device->GetTransform for async safety
    auto& fb = getRecordingBuffer();
    auto& tracker = g_cmdBufferSet.stateTracker();
    if (!tracker.getTransform(D3DTS_VIEW, &fb.view) || !tracker.getTransform(D3DTS_PROJECTION, &fb.proj)) {
        // Fallback to device query if tracker doesn't have values (shouldn't happen)
        LOG::logline("!! startRecording: MWStateTracker missing VIEW/PROJ, falling back to device query");
        trackDeviceRead("GetTransform(VIEW,PROJ) - fallback");
        device->GetTransform(D3DTS_VIEW, &fb.view);
        device->GetTransform(D3DTS_PROJECTION, &fb.proj);
    }
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
    // N-1: work on rendering buffer (previous frame being rendered)
    {
        MGE_ZoneScopedN("BatchWarmSuffixCache");
        auto& recCalls = getRenderingBuffer().recordedCalls;
        std::unordered_set<IDirect3DTexture9*> seen;
        for (const auto& call : recCalls) {
            if (call.rs.texture && seen.insert(call.rs.texture).second) {
                TextureSuffix::warmCache((IDirect3DDevice9*)device, call.rs.texture);
            }
        }
    }

    // Phase 2b: Prepare shader keys (bbox + occluders already done in executeHiZCulling)
    prepareRecordedCalls();

    // Phase 3: Replay all prepared calls (Hi-Z already built by executeHiZCulling)
    replayRecordedCalls(0);

    // N-1: Data is in rendering buffer — cleared when it becomes recording buffer at swap

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
    // N-1: work on rendering buffer (previous frame being rendered)
    {
        MGE_ZoneScopedN("BatchWarmSuffixCache");
        auto& recCalls = getRenderingBuffer().recordedCalls;
        std::unordered_set<IDirect3DTexture9*> seen;
        for (const auto& call : recCalls) {
            if (call.rs.texture && seen.insert(call.rs.texture).second) {
                TextureSuffix::warmCache((IDirect3DDevice9*)device, call.rs.texture);
            }
        }
    }

    // Prepare recorded calls inline on main thread
    prepareRecordedCalls();
    getRenderingBuffer().state = BufferState::ReadyToRender;

    recordingCompletedThisFrame = true;
}

// Step 2: Wait for cull completion and replay (called after renderStageBlend)
// N-1: rendering from rendering buffer (previous frame's data)
void FixedFunctionShader::waitCullAndReplay() {
    if (!recordingEnabled) return;

    auto& fb = getRenderingBuffer();
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
            // N-1: recording buffer
            auto& fb = getRecordingBuffer();
        } else if (sceneCount >= 2) {
            // Scene 2+ (hands): continue recording
        }
        // sceneCount < 0: pre-scene, ignore
    } else {
        // When recording disabled, still clear calls after potential dump
        // N-1: recording buffer
        auto& fb = getRecordingBuffer();
        fb.recordedCalls.clear();
        fb.recordedCallsScene1.clear();
        fb.recordedCallsScene2.clear();
        resetHLSLCaches();
    }
}

// captureScene1Matrices - Capture view/proj for Scene 1 (particles)
// Camera may move during Scene 0 - particles should use Scene 1's matrices, not Scene 0's
// N-1: recording buffer
// Use MWStateTracker instead of device->GetTransform for async safety
void FixedFunctionShader::captureScene1Matrices() {
    auto& fb = getRecordingBuffer();
    auto& tracker = g_cmdBufferSet.stateTracker();
    if (!tracker.getTransform(D3DTS_VIEW, &fb.viewScene1) || !tracker.getTransform(D3DTS_PROJECTION, &fb.projScene1)) {
        LOG::logline("!! captureScene1Matrices: MWStateTracker missing VIEW/PROJ, falling back");
        trackDeviceRead("GetTransform(Scene1) - fallback");
        device->GetTransform(D3DTS_VIEW, &fb.viewScene1);
        device->GetTransform(D3DTS_PROJECTION, &fb.projScene1);
    }
}

// captureScene2Matrices - Capture view/proj for Scene 2 (hands)
// Hands use a different view matrix than the world scene - must be captured separately
// N-1: recording buffer
// Use MWStateTracker instead of device->GetTransform for async safety
void FixedFunctionShader::captureScene2Matrices() {
    auto& fb = getRecordingBuffer();
    auto& tracker = g_cmdBufferSet.stateTracker();
    if (!tracker.getTransform(D3DTS_VIEW, &fb.viewScene2) || !tracker.getTransform(D3DTS_PROJECTION, &fb.projScene2)) {
        LOG::logline("!! captureScene2Matrices: MWStateTracker missing VIEW/PROJ, falling back");
        trackDeviceRead("GetTransform(Scene2) - fallback");
        device->GetTransform(D3DTS_VIEW, &fb.viewScene2);
        device->GetTransform(D3DTS_PROJECTION, &fb.projScene2);
    }
}

// Offscreen state save/restore - prevents blend state leak to UI when world rendering is off
static struct {
    DWORD alphaBlendEnable;
    DWORD srcBlend;
    DWORD destBlend;
    DWORD alphaTestEnable;
    DWORD alphaRef;
    DWORD alphaFunc;
    bool saved;
} s_offscreenState = { 0, 0, 0, 0, 0, 0, false };

void FixedFunctionShader::saveOffscreenState() {
    device->GetRenderState(D3DRS_ALPHABLENDENABLE, &s_offscreenState.alphaBlendEnable);
    device->GetRenderState(D3DRS_SRCBLEND, &s_offscreenState.srcBlend);
    device->GetRenderState(D3DRS_DESTBLEND, &s_offscreenState.destBlend);
    device->GetRenderState(D3DRS_ALPHATESTENABLE, &s_offscreenState.alphaTestEnable);
    device->GetRenderState(D3DRS_ALPHAREF, &s_offscreenState.alphaRef);
    device->GetRenderState(D3DRS_ALPHAFUNC, &s_offscreenState.alphaFunc);
    s_offscreenState.saved = true;
}

void FixedFunctionShader::restoreOffscreenState() {
    if (!s_offscreenState.saved) return;
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, s_offscreenState.alphaBlendEnable);
    device->SetRenderState(D3DRS_SRCBLEND, s_offscreenState.srcBlend);
    device->SetRenderState(D3DRS_DESTBLEND, s_offscreenState.destBlend);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, s_offscreenState.alphaTestEnable);
    device->SetRenderState(D3DRS_ALPHAREF, s_offscreenState.alphaRef);
    device->SetRenderState(D3DRS_ALPHAFUNC, s_offscreenState.alphaFunc);
    s_offscreenState.saved = false;
}

// capturePostRecordingState - Capture MW device state at end of Scene 0
// Recording continues through Scene 1/2; this just saves what we need to restore later.
// N-1: recording buffer
void FixedFunctionShader::capturePostRecordingState() {
    if (!isRecording) return;

    auto& fb = getRecordingBuffer();

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

    // NOTE: finalizeAndRender is a legacy path, not used in main HLSL flow
    // N-1: Use rendering buffer for prepare/render operations
    auto& renderFb = getRenderingBuffer();

    // Batch-warm suffix cache before prepare
    {
        MGE_ZoneScopedN("BatchWarmSuffixCache");
        auto& recCalls = renderFb.recordedCalls;
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
            auto& fb = renderFb;
            int s0 = (int)fb.recordedCalls.size();
            int s1 = (int)fb.recordedCallsScene1.size();
            int s2 = (int)fb.recordedCallsScene2.size();
            int m0 = 0, m1 = 0, m2 = 0;
            for (const auto& m : fb.recordMW) {
                if (m.sceneNum == 0) m0++;
                else if (m.sceneNum == 1) m1++;
                else m2++;
            }
            loggedOnce = true;
        }
    }

    // === PREPARE (CPU) ===
    currentPhase = PipelinePhase::CpuPrepare;
    {
        MGE_ZoneScopedN("Frame_Prepare");

        // Prepare recorded calls inline on main thread
        prepareRecordedCalls();
        renderFb.state = BufferState::ReadyToRender;
    }

    // === RENDER (GPU) ===
    currentPhase = PipelinePhase::GpuRender;
    {
        auto& fb = renderFb;

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
            work.bufferIndex = 0;  // Single buffer
            g_renderThread->submitWork(std::move(work), false);  // wait=false — async for 3b
        } else {
            executeGpuPhase();
        }
    }

    currentPhase = PipelinePhase::Idle;
}

void FixedFunctionShader::executeGpuPhase() {
    MGE_ZoneScopedN("Frame_Render");

    // N-1: First frame check - skip if no previous frame data
    if (!n1Ready) {
        LOG::logline("executeGpuPhase: N-1 not ready (first frame), skipping");
        return;
    }

    // N-1: rendering from rendering buffer (previous frame's data)
    auto& fb = getRenderingBuffer();
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

    // Build HLSL replay into command buffer, then replay it
    // IMPORTANT: Only replay Scene 0 here! Scene 1/2 are recorded AFTER EndScene(0),
    // so replaying them here would use STALE data from previous frame (race condition).
    // Scene 1/2 replay is deferred to finalizeAndRenderAllScenes.
    transitionTo(PhaseTransition::ReplayEntry);
    fb.hlslCmds.clear();

    // Scene 0: World geometry (only scene recorded at this point)
    LOG::logline("[ORDER] executeGpuPhase: Replay Scene 0 (%d calls)", (int)fb.recordedCalls.size());
    replayRecordedCalls(0, &fb.hlslCmds);

    // Scene 1/2 NOT replayed here — they haven't been recorded yet at EndScene(0)
    // They will be replayed in finalizeAndRenderAllScenes after recording completes

    fb.hlslCmds.replay(device);
    transitionTo(PhaseTransition::ReplayExit);

    // Blend distant land over near objects AFTER replay
    // (MW/MGE blend uses depth to composite distant land behind near objects)
    DistantLand::renderStageBlend(frameCtx, &fb);

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
// N-1: renders from rendering buffer (previous frame's data)
// First frame: n1Ready is false until first swap completes, so we skip
void FixedFunctionShader::finalizeAndRenderAllScenes(DLContext* frameCtx, bool waterSeen) {
    MGE_ZoneScopedN("finalizeAndRenderAllScenes");

    if (!recordingEnabled) return;

    // First frame check: N-1 not ready until first buffer swap at Present()
    if (!n1Ready) {
        LOG::logline("finalizeAndRenderAllScenes: N-1 not ready (first frame), skipping render");
        // On first frame, no previous frame data exists - skip rendering
        // MW's direct draws will still appear, distant land/water won't render this frame
        return;
    }

    // N-1: use rendering buffer (previous frame's recorded data)
    auto& fb = getRenderingBuffer();


    // === SAVE UI STATE ===
    // MW set up UI state before calling BeginScene. We need to restore these
    // after 3D rendering so HUD draws correctly.
    D3DXMATRIX savedUIView, savedUIProj;
    device->GetTransform(D3DTS_VIEW, &savedUIView);
    device->GetTransform(D3DTS_PROJECTION, &savedUIProj);
    // Save blend/alpha state - GPU phases will corrupt this
    DWORD savedAlphaBlend, savedSrcBlend, savedDestBlend;
    DWORD savedAlphaTest, savedAlphaRef, savedAlphaFunc;
    DWORD savedZEnable, savedZWrite, savedZFunc;
    device->GetRenderState(D3DRS_ALPHABLENDENABLE, &savedAlphaBlend);
    device->GetRenderState(D3DRS_SRCBLEND, &savedSrcBlend);
    device->GetRenderState(D3DRS_DESTBLEND, &savedDestBlend);
    device->GetRenderState(D3DRS_ALPHATESTENABLE, &savedAlphaTest);
    device->GetRenderState(D3DRS_ALPHAREF, &savedAlphaRef);
    device->GetRenderState(D3DRS_ALPHAFUNC, &savedAlphaFunc);
    device->GetRenderState(D3DRS_ZENABLE, &savedZEnable);
    device->GetRenderState(D3DRS_ZWRITEENABLE, &savedZWrite);
    device->GetRenderState(D3DRS_ZFUNC, &savedZFunc);
    // Save FVF - GPU phases use vertex declarations that corrupt UI's expected format
    DWORD savedFVF;
    device->GetFVF(&savedFVF);

    // === RESTORE ENDSCENE(0) STATE ===
    // State was captured at EndScene(0), but MW ran Scene 1/2 since then.
    // Restore to match baseline expectations for renderStage0GPU.
    fb.stateContract.applyTo((IDirect3DDevice9*)device);

    // === PREPARE ALL SCENES ===
    {
        MGE_ZoneScopedN("Prepare All Scenes");
        prepareRecordedCalls();  // This now prepares Scene 0, 1, and 2
        fb.state = BufferState::ReadyToRender;
    }

    // N-1: Use stored context from rendering buffer (captured when this was recording buffer)
    // dlContext and waterSeen are already stored in fb from previous frame's recording phase
    DLContext* renderCtx = &fb.dlContext;
    bool renderWaterSeen = fb.waterSeen;

    // Diagnostic: log nearViewRange when retrieved for rendering
    static int retrieveLogCount = 0;
    if (retrieveLogCount++ < 10) {
        LOG::logline("[N1-RETRIEVE] nearViewRange=%.1f from rendering buffer", renderCtx->nearViewRange);
    }

    // === STAGE 0 GPU: Shadow map, distant land, sky, water reflection, wave sim ===
    {
        MGE_ZoneScopedN("Stage0 GPU");
        DistantLand::renderStage0GPU(renderCtx, &fb);
    }

    // Update shadow VP in frame buffer — renderShadowMap just computed current frame's
    // shadow matrices and wrote them back to s_staging. The stale previous-frame values
    // captured at startRecording() would cause shadow shaking on camera movement.
    {
        fb.shadowViewproj[0] = DistantLand::s_staging.smViewproj[0];
        fb.shadowViewproj[1] = DistantLand::s_staging.smViewproj[1];
    }

    // === DEPTH PASSES ===
    // Render depth for Scene 0 (world) and Scene 2 (hands)
    // Scene 1 (particles) skipped - they're alpha blended, no depth write
    {
        MGE_ZoneScopedN("Depth Passes");

        // Use renderStage1 for Scene 0 depth (includes shadows, distant land setup)
        DistantLand::renderStage1(renderCtx, &fb);

        // Render Scene 2 (hands) depth into the depth texture
        // Filter for sceneNum >= 2 only
        DistantLand::renderStage2(renderCtx, &fb);
    }

    // === REPLAY ALL SCENES ===
    isReplaying = true;
    {
        MGE_ZoneScopedN("Replay All Scenes");

        transitionTo(PhaseTransition::ReplayEntry);

        // Replay Scene 0 (world) FIRST - Morrowind near objects
        LOG::logline("[ORDER] finalizeAndRenderAllScenes: Replay Scene 0 (%d calls)", (int)fb.recordedCalls.size());
        replayRecordedCalls(0, nullptr);

        transitionTo(PhaseTransition::ReplayExit);

        // Blend distant land over near objects AFTER replay
        // (MW/MGE blend uses depth to composite distant land behind near objects)
        DistantLand::renderStageBlend(renderCtx, &fb);
    }

    // Water surface AFTER Scene 0 but BEFORE Scene 1 particles
    // Water writes depth, then particles blend over it
    if (renderWaterSeen) {
        DistantLand::renderStageWater(renderCtx);
    }

    // Upload snapshot vertex/index data to staging buffers before Scene 1/2 replay
    {
        MGE_ZoneScopedN("Upload Particle Staging Buffers");

        // Calculate total bytes needed for Scene 1 + Scene 2
        UINT totalVBBytes = 0, totalIBBytes = 0;
        for (auto& call : fb.recordedCallsScene1) {
            if (call.usesSnapshot) {
                call.stagingVBOffset = totalVBBytes;
                call.stagingIBOffset = totalIBBytes;
                totalVBBytes += (UINT)call.vertexSnapshot.size();
                totalIBBytes += (UINT)call.indexSnapshot.size();
            }
        }
        for (auto& call : fb.recordedCallsScene2) {
            if (call.usesSnapshot) {
                call.stagingVBOffset = totalVBBytes;
                call.stagingIBOffset = totalIBBytes;
                totalVBBytes += (UINT)call.vertexSnapshot.size();
                totalIBBytes += (UINT)call.indexSnapshot.size();
            }
        }

        // Resize staging VB if needed
        if (totalVBBytes > 0 && totalVBBytes > fb.stagingVBSize) {
            if (fb.particleStagingVB) {
                fb.particleStagingVB->Release();
                fb.particleStagingVB = nullptr;
            }
            HRESULT hr = device->CreateVertexBuffer(totalVBBytes, D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                0, D3DPOOL_DEFAULT, &fb.particleStagingVB, nullptr);
            if (SUCCEEDED(hr)) {
                fb.stagingVBSize = totalVBBytes;
                LOG::logline("[STAGING] Created VB %u bytes", totalVBBytes);
            } else {
                LOG::logline("!! Failed to create particle staging VB (%u bytes)", totalVBBytes);
            }
        }

        // Resize staging IB if needed
        if (totalIBBytes > 0 && totalIBBytes > fb.stagingIBSize) {
            if (fb.particleStagingIB) {
                fb.particleStagingIB->Release();
                fb.particleStagingIB = nullptr;
            }
            HRESULT hr = device->CreateIndexBuffer(totalIBBytes, D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                D3DFMT_INDEX16, D3DPOOL_DEFAULT, &fb.particleStagingIB, nullptr);
            if (SUCCEEDED(hr)) {
                fb.stagingIBSize = totalIBBytes;
                LOG::logline("[STAGING] Created IB %u bytes", totalIBBytes);
            } else {
                LOG::logline("!! Failed to create particle staging IB (%u bytes)", totalIBBytes);
            }
        }

        // Batch upload all snapshot data
        if (totalVBBytes > 0 && fb.particleStagingVB && totalIBBytes > 0 && fb.particleStagingIB) {
            void* pVB = nullptr;
            void* pIB = nullptr;
            if (SUCCEEDED(fb.particleStagingVB->Lock(0, totalVBBytes, &pVB, D3DLOCK_DISCARD)) &&
                SUCCEEDED(fb.particleStagingIB->Lock(0, totalIBBytes, &pIB, D3DLOCK_DISCARD))) {

                for (auto& call : fb.recordedCallsScene1) {
                    if (call.usesSnapshot && !call.vertexSnapshot.empty()) {
                        memcpy((BYTE*)pVB + call.stagingVBOffset, call.vertexSnapshot.data(), call.vertexSnapshot.size());
                        memcpy((BYTE*)pIB + call.stagingIBOffset, call.indexSnapshot.data(), call.indexSnapshot.size());
                    }
                }
                for (auto& call : fb.recordedCallsScene2) {
                    if (call.usesSnapshot && !call.vertexSnapshot.empty()) {
                        memcpy((BYTE*)pVB + call.stagingVBOffset, call.vertexSnapshot.data(), call.vertexSnapshot.size());
                        memcpy((BYTE*)pIB + call.stagingIBOffset, call.indexSnapshot.data(), call.indexSnapshot.size());
                    }
                }

                fb.particleStagingVB->Unlock();
                fb.particleStagingIB->Unlock();
                LOG::logline("[STAGING] Uploaded %u VB bytes, %u IB bytes", totalVBBytes, totalIBBytes);
            }
        }
    }

    // Replay Scene 1 (particles) - use Scene 1's view/proj, blend over water
    {
        isReplaying = true;
        D3DXMATRIX savedView = fb.view;
        D3DXMATRIX savedProj = fb.proj;
        fb.view = fb.viewScene1;
        fb.proj = fb.projScene1;

        LOG::logline("[ORDER] finalizeAndRenderAllScenes: Replay Scene 1 (%d calls)", (int)fb.recordedCallsScene1.size());
        replayRecordedCalls(1, nullptr);

        fb.view = savedView;
        fb.proj = savedProj;
        isReplaying = false;
    }

    // Z-clear before Scene 2: MW clears depth before hands so they render in front
    device->Clear(0, NULL, D3DCLEAR_ZBUFFER, 0, 1.0f, 0);

    // Replay Scene 2 (hands) - use hands view/proj matrices
    // Hands use a different view matrix to stay fixed on screen
    {
        D3DXMATRIX savedView = fb.view;
        D3DXMATRIX savedProj = fb.proj;
        fb.view = fb.viewScene2;
        fb.proj = fb.projScene2;

        isReplaying = true;
        replayRecordedCalls(2, nullptr);
        isReplaying = false;

        fb.view = savedView;
        fb.proj = savedProj;
    }

    // Clean up HLSL state (textures, shaders)
    for (int i = 2; i < 6; i++) {
        device->SetTexture(i, NULL);
        textureCache.updateCache(i, nullptr);
        textureCache.textureValid[i] = false;
    }
    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);
    device->SetFVF(savedFVF);  // Restore vertex format for fixed-function UI

    // === RESTORE UI STATE ===
    // MW set up UI state before calling BeginScene. Restore it so HUD draws correctly.
    device->SetTransform(D3DTS_VIEW, &savedUIView);
    device->SetTransform(D3DTS_PROJECTION, &savedUIProj);
    // Restore blend/alpha/depth state (GPU phases corrupted these)
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, savedAlphaBlend);
    device->SetRenderState(D3DRS_SRCBLEND, savedSrcBlend);
    device->SetRenderState(D3DRS_DESTBLEND, savedDestBlend);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, savedAlphaTest);
    device->SetRenderState(D3DRS_ALPHAREF, savedAlphaRef);
    device->SetRenderState(D3DRS_ALPHAFUNC, savedAlphaFunc);
    device->SetRenderState(D3DRS_ZENABLE, savedZEnable);
    device->SetRenderState(D3DRS_ZWRITEENABLE, savedZWrite);
    device->SetRenderState(D3DRS_ZFUNC, savedZFunc);

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

    // Upload snapshot vertex/index data to staging buffers before replay
    {
        MGE_ZoneScopedN("Upload Particle Staging Buffers");

        // Calculate total bytes needed for Scene 1 + Scene 2
        UINT totalVBBytes = 0, totalIBBytes = 0;
        for (auto& call : fb->recordedCallsScene1) {
            if (call.usesSnapshot) {
                call.stagingVBOffset = totalVBBytes;
                call.stagingIBOffset = totalIBBytes;
                totalVBBytes += (UINT)call.vertexSnapshot.size();
                totalIBBytes += (UINT)call.indexSnapshot.size();
            }
        }
        for (auto& call : fb->recordedCallsScene2) {
            if (call.usesSnapshot) {
                call.stagingVBOffset = totalVBBytes;
                call.stagingIBOffset = totalIBBytes;
                totalVBBytes += (UINT)call.vertexSnapshot.size();
                totalIBBytes += (UINT)call.indexSnapshot.size();
            }
        }

        // Resize staging VB if needed
        if (totalVBBytes > 0 && totalVBBytes > fb->stagingVBSize) {
            if (fb->particleStagingVB) {
                fb->particleStagingVB->Release();
                fb->particleStagingVB = nullptr;
            }
            HRESULT hr = device->CreateVertexBuffer(totalVBBytes, D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                0, D3DPOOL_DEFAULT, &fb->particleStagingVB, nullptr);
            if (SUCCEEDED(hr)) {
                fb->stagingVBSize = totalVBBytes;
            } else {
                LOG::logline("!! Failed to create particle staging VB (%u bytes)", totalVBBytes);
            }
        }

        // Resize staging IB if needed
        if (totalIBBytes > 0 && totalIBBytes > fb->stagingIBSize) {
            if (fb->particleStagingIB) {
                fb->particleStagingIB->Release();
                fb->particleStagingIB = nullptr;
            }
            HRESULT hr = device->CreateIndexBuffer(totalIBBytes, D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                D3DFMT_INDEX16, D3DPOOL_DEFAULT, &fb->particleStagingIB, nullptr);
            if (SUCCEEDED(hr)) {
                fb->stagingIBSize = totalIBBytes;
            } else {
                LOG::logline("!! Failed to create particle staging IB (%u bytes)", totalIBBytes);
            }
        }

        // Batch upload all snapshot data
        if (totalVBBytes > 0 && fb->particleStagingVB && totalIBBytes > 0 && fb->particleStagingIB) {
            void* pVB = nullptr;
            void* pIB = nullptr;
            if (SUCCEEDED(fb->particleStagingVB->Lock(0, totalVBBytes, &pVB, D3DLOCK_DISCARD)) &&
                SUCCEEDED(fb->particleStagingIB->Lock(0, totalIBBytes, &pIB, D3DLOCK_DISCARD))) {

                for (auto& call : fb->recordedCallsScene1) {
                    if (call.usesSnapshot && !call.vertexSnapshot.empty()) {
                        memcpy((BYTE*)pVB + call.stagingVBOffset, call.vertexSnapshot.data(), call.vertexSnapshot.size());
                        memcpy((BYTE*)pIB + call.stagingIBOffset, call.indexSnapshot.data(), call.indexSnapshot.size());
                    }
                }
                for (auto& call : fb->recordedCallsScene2) {
                    if (call.usesSnapshot && !call.vertexSnapshot.empty()) {
                        memcpy((BYTE*)pVB + call.stagingVBOffset, call.vertexSnapshot.data(), call.vertexSnapshot.size());
                        memcpy((BYTE*)pIB + call.stagingIBOffset, call.indexSnapshot.data(), call.indexSnapshot.size());
                    }
                }

                fb->particleStagingVB->Unlock();
                fb->particleStagingIB->Unlock();
            }
        }
    }

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
        case PhaseTransition::Scene1Entry: {
            // Scene1Entry must match RecordingExit - Scene 1 should see same state as end of recording
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
        case PhaseTransition::GpuExit:
            // GpuExit happens after GPU work at UI BeginScene - state is restored for UI, not Scene 0
            // No validation needed here - state is intentionally different (UI state vs recording state)
            break;
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

// Single-buffer pipeline: cull pass (called by CullThread or inline on main thread)
void FixedFunctionShader::executeCullPass() {
    // Cull thread is idle - prepareRecordedCalls moved to main thread
}

void FixedFunctionShader::executeRenderPass() {
    executeGpuPhase();
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

    // Record to current scene's buffer - each scene uses its own captured VIEW matrix
    // Do NOT reroute blending draws to Scene 1 - that causes VIEW matrix mismatch
    // (MW computed billboard vertices using current scene's VIEW, not Scene 1's VIEW)
    {
        // Debug: Log particle draws (blendEnable && !zWrite)
        static int hlslRecParticleIdx = 0;
        static int hlslRecLastFrame = -1;
        if (hlslDiagFrameCounter != hlslRecLastFrame) {
            hlslRecParticleIdx = 0;
            hlslRecLastFrame = hlslDiagFrameCounter;
        }
        if (rs->blendEnable && !rs->zWrite && hlslRecParticleIdx < 20) {
            // N-1: recording buffer for debug logging during recording
            auto& recFb = getRecordingBuffer();
            size_t callIdx = currentRecordedCalls().size();
            LOG::logline("[REC-P] #%d vbs=%d world41=%.2f wvt41=%.2f view41=%.2f proj11=%.3f prims=%d",
                hlslRecParticleIdx, rs->vertexBlendState,
                rs->worldTransforms[0]._41, rs->worldViewTransforms[0]._41,
                rs->viewTransform._41, recFb.proj._11, rs->primCount);
            hlslRecParticleIdx++;
        }

        currentRecordedCalls().emplace_back(rs, frs, sharedLightState, sk, recordMWIdx);
        auto& newCall = currentRecordedCalls().back();
        newCall.sceneNum = currentRecordingScene;

        // Snapshot VB/IB data for Scene 1/2 (particles/hands) with dynamic VBs
        // The shared particle VB may be overwritten by other particles before replay
        if (currentRecordingScene >= 1 && rs->vb && rs->ib) {
            newCall.usesSnapshot = true;

            // Snapshot vertex data
            UINT vbStartByte = rs->baseIndex * rs->vbStride;
            UINT vbByteCount = rs->vertCount * rs->vbStride;
            newCall.vertexSnapshot.resize(vbByteCount);
            void* pVertices = nullptr;
            HRESULT hr = rs->vb->Lock(vbStartByte, vbByteCount, &pVertices, D3DLOCK_READONLY | D3DLOCK_NOOVERWRITE);
            if (SUCCEEDED(hr) && pVertices) {
                memcpy(newCall.vertexSnapshot.data(), pVertices, vbByteCount);
                rs->vb->Unlock();
            } else {
                newCall.usesSnapshot = false;  // Fallback to original VB if lock fails
                newCall.vertexSnapshot.clear();
                LOG::logline("[SNAP] VB lock failed scene=%d hr=0x%X", currentRecordingScene, hr);
            }

            // Snapshot index data (only if vertex snapshot succeeded)
            if (newCall.usesSnapshot) {
                // Determine index size from IB format
                D3DINDEXBUFFER_DESC ibDesc;
                rs->ib->GetDesc(&ibDesc);
                UINT indexSize = (ibDesc.Format == D3DFMT_INDEX32) ? 4 : 2;
                UINT ibStartByte = rs->startIndex * indexSize;
                UINT ibByteCount = rs->primCount * 3 * indexSize;  // Triangle list
                newCall.indexSnapshot.resize(ibByteCount);
                void* pIndices = nullptr;
                hr = rs->ib->Lock(ibStartByte, ibByteCount, &pIndices, D3DLOCK_READONLY | D3DLOCK_NOOVERWRITE);
                if (SUCCEEDED(hr) && pIndices) {
                    memcpy(newCall.indexSnapshot.data(), pIndices, ibByteCount);
                    rs->ib->Unlock();
                    // Log first few snapshots per frame
                    static int snapLogCount = 0;
                    static int snapLastFrame = -1;
                    if (hlslDiagFrameCounter != snapLastFrame) {
                        snapLogCount = 0;
                        snapLastFrame = hlslDiagFrameCounter;
                    }
                    if (snapLogCount < 3) {
                        LOG::logline("[SNAP] OK scene=%d VB=%u IB=%u stride=%d verts=%d prims=%d",
                            currentRecordingScene, vbByteCount, ibByteCount, rs->vbStride, rs->vertCount, rs->primCount);
                        snapLogCount++;
                    }
                } else {
                    newCall.usesSnapshot = false;  // Fallback if lock fails
                    newCall.vertexSnapshot.clear();
                    newCall.indexSnapshot.clear();
                    LOG::logline("[SNAP] IB lock failed scene=%d hr=0x%X", currentRecordingScene, hr);
                }
            }
        } else if (currentRecordingScene >= 1) {
            // Log why we didn't snapshot
            static int noSnapLogCount = 0;
            static int noSnapLastFrame = -1;
            if (hlslDiagFrameCounter != noSnapLastFrame) {
                noSnapLogCount = 0;
                noSnapLastFrame = hlslDiagFrameCounter;
            }
            if (noSnapLogCount < 3) {
                LOG::logline("[SNAP] SKIP scene=%d vb=%p ib=%p", currentRecordingScene, rs->vb, rs->ib);
                noSnapLogCount++;
            }
        }

        // Log recording order for Scene 1/2 (particles/hands)
        if (currentRecordingScene > 0) {
            static int orderLogCount = 0;
            if (orderLogCount < 10) {
                LOG::logline("[ORDER] Record Scene %d call #%d (world41=%.1f)",
                    currentRecordingScene, (int)currentRecordedCalls().size(), rs->worldTransforms[0]._41);
                orderLogCount++;
            }
        }

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
