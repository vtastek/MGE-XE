// Recording system for HLSL render dispatch pipeline
// Extracted from ffeshader.cpp - Phase 4 of refactor

#include "ffeshader.h"
#include "texture_suffix.h"
#include "cullthread.h"
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

static auto& currentRecordedCalls() {
    return FixedFunctionShader::frameBuffers[FixedFunctionShader::recordingBuffer].recordedCalls;
}

// Diagnostic: cache hit/miss logging for first N frames (temporary)
static int hlslDiagFrameCounter = 0;

void FixedFunctionShader::startRecording() {
    // Save render states before recording so we can restore after replay
    trackDeviceRead("GetRenderState(x14 preRecording)");
    device->GetRenderState(D3DRS_ALPHABLENDENABLE, &preRecordingState.alphaBlendEnable);
    device->GetRenderState(D3DRS_ALPHATESTENABLE, &preRecordingState.alphaTestEnable);
    device->GetRenderState(D3DRS_ZENABLE, &preRecordingState.zEnable);
    device->GetRenderState(D3DRS_ZWRITEENABLE, &preRecordingState.zWriteEnable);
    device->GetRenderState(D3DRS_CULLMODE, &preRecordingState.cullMode);
    device->GetRenderState(D3DRS_SRCBLEND, &preRecordingState.srcBlend);
    device->GetRenderState(D3DRS_DESTBLEND, &preRecordingState.destBlend);
    device->GetRenderState(D3DRS_FOGENABLE, &preRecordingState.fogEnable);
    device->GetRenderState(D3DRS_SPECULARENABLE, &preRecordingState.specularEnable);
    device->GetRenderState(D3DRS_LOCALVIEWER, &preRecordingState.localViewer);
    device->GetRenderState(D3DRS_NORMALIZENORMALS, &preRecordingState.normalizeNormals);
    device->GetRenderState(D3DRS_ZFUNC, &preRecordingState.zFunc);
    device->GetRenderState(D3DRS_ALPHAFUNC, &preRecordingState.alphaFunc);
    device->GetRenderState(D3DRS_ALPHAREF, &preRecordingState.alphaRef);

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
    // We restore this after replay instead of preRecordingState (which was the FIRST mesh's state
    // and could have different alpha test/blend settings that corrupt the sky).
    SavedRenderStates postRecordingState;
    device->GetRenderState(D3DRS_ALPHABLENDENABLE, &postRecordingState.alphaBlendEnable);
    device->GetRenderState(D3DRS_ALPHATESTENABLE, &postRecordingState.alphaTestEnable);
    device->GetRenderState(D3DRS_ZENABLE, &postRecordingState.zEnable);
    device->GetRenderState(D3DRS_ZWRITEENABLE, &postRecordingState.zWriteEnable);
    device->GetRenderState(D3DRS_CULLMODE, &postRecordingState.cullMode);
    device->GetRenderState(D3DRS_SRCBLEND, &postRecordingState.srcBlend);
    device->GetRenderState(D3DRS_DESTBLEND, &postRecordingState.destBlend);
    device->GetRenderState(D3DRS_FOGENABLE, &postRecordingState.fogEnable);
    device->GetRenderState(D3DRS_SPECULARENABLE, &postRecordingState.specularEnable);
    device->GetRenderState(D3DRS_LOCALVIEWER, &postRecordingState.localViewer);
    device->GetRenderState(D3DRS_NORMALIZENORMALS, &postRecordingState.normalizeNormals);
    device->GetRenderState(D3DRS_ZFUNC, &postRecordingState.zFunc);
    device->GetRenderState(D3DRS_ALPHAFUNC, &postRecordingState.alphaFunc);
    device->GetRenderState(D3DRS_ALPHAREF, &postRecordingState.alphaRef);

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
    // leaked state from previous frame's Scene 1+ immediate rendering.
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, postRecordingState.alphaBlendEnable);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, postRecordingState.alphaTestEnable);
    device->SetRenderState(D3DRS_ZENABLE, postRecordingState.zEnable);
    device->SetRenderState(D3DRS_ZWRITEENABLE, postRecordingState.zWriteEnable);
    device->SetRenderState(D3DRS_CULLMODE, postRecordingState.cullMode);
    device->SetRenderState(D3DRS_SRCBLEND, postRecordingState.srcBlend);
    device->SetRenderState(D3DRS_DESTBLEND, postRecordingState.destBlend);
    device->SetRenderState(D3DRS_FOGENABLE, postRecordingState.fogEnable);
    device->SetRenderState(D3DRS_SPECULARENABLE, postRecordingState.specularEnable);
    device->SetRenderState(D3DRS_LOCALVIEWER, postRecordingState.localViewer);
    device->SetRenderState(D3DRS_NORMALIZENORMALS, postRecordingState.normalizeNormals);
    device->SetRenderState(D3DRS_ZFUNC, postRecordingState.zFunc);
    device->SetRenderState(D3DRS_ALPHAFUNC, postRecordingState.alphaFunc);
    device->SetRenderState(D3DRS_ALPHAREF, postRecordingState.alphaRef);

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
    device->GetRenderState(D3DRS_ALPHABLENDENABLE, &postRecordingState.alphaBlendEnable);
    device->GetRenderState(D3DRS_ALPHATESTENABLE, &postRecordingState.alphaTestEnable);
    device->GetRenderState(D3DRS_ZENABLE, &postRecordingState.zEnable);
    device->GetRenderState(D3DRS_ZWRITEENABLE, &postRecordingState.zWriteEnable);
    device->GetRenderState(D3DRS_CULLMODE, &postRecordingState.cullMode);
    device->GetRenderState(D3DRS_SRCBLEND, &postRecordingState.srcBlend);
    device->GetRenderState(D3DRS_DESTBLEND, &postRecordingState.destBlend);
    device->GetRenderState(D3DRS_FOGENABLE, &postRecordingState.fogEnable);
    device->GetRenderState(D3DRS_SPECULARENABLE, &postRecordingState.specularEnable);
    device->GetRenderState(D3DRS_LOCALVIEWER, &postRecordingState.localViewer);
    device->GetRenderState(D3DRS_NORMALIZENORMALS, &postRecordingState.normalizeNormals);
    device->GetRenderState(D3DRS_ZFUNC, &postRecordingState.zFunc);
    device->GetRenderState(D3DRS_ALPHAFUNC, &postRecordingState.alphaFunc);
    device->GetRenderState(D3DRS_ALPHAREF, &postRecordingState.alphaRef);

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

    // Submit to cull thread for async prepare
    int buf = recordingBuffer;
    frameBuffers[buf].state = BufferState::ReadyToCull;

    if (g_cullThread && g_cullThread->isRunning()) {
        g_cullThread->submitWork(buf, false);  // false = don't wait
    } else {
        // Fallback: no cull thread, run prepare inline
        executeCullPass(buf);
        frameBuffers[buf].state = BufferState::ReadyToRender;
    }

    recordingCompletedThisFrame = true;
}

// Step 2: Wait for cull completion and replay (called after renderStageBlend)
void FixedFunctionShader::waitCullAndReplay() {
    if (!recordingEnabled) return;

    auto& fb = frameBuffers[recordingBuffer];
    if (fb.state != BufferState::ReadyToCull && fb.state != BufferState::Culling
        && fb.state != BufferState::ReadyToRender) {
        return;  // Nothing was submitted
    }

    // Wait for cull thread to finish (should be done by now — renderStage1+Blend gave it time)
    if (g_cullThread && g_cullThread->isRunning() && fb.state != BufferState::ReadyToRender) {
        g_cullThread->waitForCompletion();
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
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, postRecordingState.alphaBlendEnable);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, postRecordingState.alphaTestEnable);
    device->SetRenderState(D3DRS_ZENABLE, postRecordingState.zEnable);
    device->SetRenderState(D3DRS_ZWRITEENABLE, postRecordingState.zWriteEnable);
    device->SetRenderState(D3DRS_CULLMODE, postRecordingState.cullMode);
    device->SetRenderState(D3DRS_SRCBLEND, postRecordingState.srcBlend);
    device->SetRenderState(D3DRS_DESTBLEND, postRecordingState.destBlend);
    device->SetRenderState(D3DRS_FOGENABLE, postRecordingState.fogEnable);
    device->SetRenderState(D3DRS_SPECULARENABLE, postRecordingState.specularEnable);
    device->SetRenderState(D3DRS_LOCALVIEWER, postRecordingState.localViewer);
    device->SetRenderState(D3DRS_NORMALIZENORMALS, postRecordingState.normalizeNormals);
    device->SetRenderState(D3DRS_ZFUNC, postRecordingState.zFunc);
    device->SetRenderState(D3DRS_ALPHAFUNC, postRecordingState.alphaFunc);
    device->SetRenderState(D3DRS_ALPHAREF, postRecordingState.alphaRef);

    isReplaying = false;

    // Reset HLSL caches after rendering session completes
    resetHLSLCaches();
}

// Call this when HLSL rendering session is complete to trigger replay
void FixedFunctionShader::finalizeBatchAndReplay(int sceneCount) {
    if (recordingEnabled) {
        if (sceneCount == 0) {
            // Scene 0: recording continues through Scene 1+
            // capturePostRecordingState() already saved device state
            // finalizeAndRender() will do the actual prepare+render later
        } else {
            // Scene 1+: recording continues, no-op
            // Calls accumulate into the same buffer for unified replay
        }
    } else {
        // When recording disabled, still clear calls after potential dump
        currentRecordedCalls().clear();
        resetHLSLCaches();
    }
}

// capturePostRecordingState - Capture MW device state at end of Scene 0
// Recording continues through Scene 1+; this just saves what we need to restore later.
void FixedFunctionShader::capturePostRecordingState() {
    if (!isRecording) return;

    // Write to per-buffer postRecordingState for HLSL isolation
    trackDeviceRead("GetRenderState(x14 postRecording)");
    auto& prs = frameBuffers[recordingBuffer].postRecordingState;
    device->GetRenderState(D3DRS_ALPHABLENDENABLE, &prs.alphaBlendEnable);
    device->GetRenderState(D3DRS_ALPHATESTENABLE, &prs.alphaTestEnable);
    device->GetRenderState(D3DRS_ZENABLE, &prs.zEnable);
    device->GetRenderState(D3DRS_ZWRITEENABLE, &prs.zWriteEnable);
    device->GetRenderState(D3DRS_CULLMODE, &prs.cullMode);
    device->GetRenderState(D3DRS_SRCBLEND, &prs.srcBlend);
    device->GetRenderState(D3DRS_DESTBLEND, &prs.destBlend);
    device->GetRenderState(D3DRS_FOGENABLE, &prs.fogEnable);
    device->GetRenderState(D3DRS_SPECULARENABLE, &prs.specularEnable);
    device->GetRenderState(D3DRS_LOCALVIEWER, &prs.localViewer);
    device->GetRenderState(D3DRS_NORMALIZENORMALS, &prs.normalizeNormals);
    device->GetRenderState(D3DRS_ZFUNC, &prs.zFunc);
    device->GetRenderState(D3DRS_ALPHAFUNC, &prs.alphaFunc);
    device->GetRenderState(D3DRS_ALPHAREF, &prs.alphaRef);
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
            auto& recCalls = currentRecordedCalls();
            int s0 = 0, s1 = 0;
            for (const auto& c : recCalls) {
                if (c.sceneNum == 0) s0++; else s1++;
            }
            int m0 = 0, m1 = 0;
            auto& activeRecordMW = frameBuffers[recordingBuffer].recordMW;
            for (const auto& m : activeRecordMW) {
                if (m.sceneNum == 0) m0++; else m1++;
            }
            LOG::logline(">> finalizeAndRender: HLSL calls=%d (scene0=%d, scene1+=%d), recordMW=%d (scene0=%d, scene1+=%d)",
                         (int)recCalls.size(), s0, s1, (int)activeRecordMW.size(), m0, m1);
            loggedOnce = true;
        }
    }

    // === PREPARE (CPU) ===
    currentPhase = PipelinePhase::CpuPrepare;
    {
        MGE_ZoneScopedN("Frame_Prepare");

        // Submit to cull thread or run inline
        int buf = recordingBuffer;
        frameBuffers[buf].state = BufferState::ReadyToCull;

        if (g_cullThread && g_cullThread->isRunning()) {
            g_cullThread->submitWork(buf, false);
        } else {
            executeCullPass(buf);
            frameBuffers[buf].state = BufferState::ReadyToRender;
        }
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
            g_renderThread->submitWork(std::move(work), true);  // wait=true — blocking for 3a
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

    // Stage 1: grass, shadow overlay, depth (cull thread runs in parallel with this)
    DistantLand::renderStage1(frameCtx, &fb);

    // Blend close objects over distant land
    DistantLand::renderStageBlend(frameCtx, &fb);

    // Wait for cull completion
    if (g_cullThread && g_cullThread->isRunning() && fb.state != BufferState::ReadyToRender) {
        g_cullThread->waitForCompletion();
    }
    fb.state = BufferState::ReadyToRender;

    // Build HLSL replay into command buffer, then replay it
    // Scene 0 + Scene 1+ always go through HLSL replay (recordMW populated by inspectIndexedPrimitive)
    // Build HLSL replay into command buffer, then replay it
    fb.hlslCmds.clear();
    replayRecordedCalls(0, &fb.hlslCmds);
    fb.hlslCmds.replay(device);

    // Water surface AFTER replay — refraction samples backbuffer which needs scene content
    if (waterSeen) {
        DistantLand::renderStageWater(frameCtx);
    }

    // Stage 2: additional depth for all recorded geometry
    DistantLand::renderStage2(frameCtx, &fb);

    // Clean up HLSL-only texture slots to prevent DXVK descriptor bloat
    for (int i = 2; i < 6; i++) {
        device->SetTexture(i, NULL);
        textureCache.updateCache(i, nullptr);
        textureCache.textureValid[i] = false;
    }
    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);

    // Restore Morrowind's end-of-Scene-0 state (from per-buffer snapshot)
    const auto& prs = fb.postRecordingState;
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, prs.alphaBlendEnable);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, prs.alphaTestEnable);
    device->SetRenderState(D3DRS_ZENABLE, prs.zEnable);
    device->SetRenderState(D3DRS_ZWRITEENABLE, prs.zWriteEnable);
    device->SetRenderState(D3DRS_CULLMODE, prs.cullMode);
    device->SetRenderState(D3DRS_SRCBLEND, prs.srcBlend);
    device->SetRenderState(D3DRS_DESTBLEND, prs.destBlend);
    device->SetRenderState(D3DRS_FOGENABLE, prs.fogEnable);
    device->SetRenderState(D3DRS_SPECULARENABLE, prs.specularEnable);
    device->SetRenderState(D3DRS_LOCALVIEWER, prs.localViewer);
    device->SetRenderState(D3DRS_NORMALIZENORMALS, prs.normalizeNormals);
    device->SetRenderState(D3DRS_ZFUNC, prs.zFunc);
    device->SetRenderState(D3DRS_ALPHAFUNC, prs.alphaFunc);
    device->SetRenderState(D3DRS_ALPHAREF, prs.alphaRef);

    isReplaying = false;
    resetHLSLCaches();
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

// Triple-buffer pipeline: cull pass (called by CullThread or inline on main thread)
void FixedFunctionShader::executeCullPass(int bufferIndex) {
    // When called from CullThread, device calls are not safe (D3D9 is single-threaded).
    // When called inline from main thread (no cull thread fallback), device calls are OK.
    // CullThread::executeCull sets this to false before calling us.
    prepareRecordedCalls(bufferIndex);
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

    // Log per-bin frame event (we know the bin from sk and rs at record time)
    {
        FrameEvent::Type evType;
        if (sk.hasGrass)              evType = FrameEvent::DIP_Grass;
        else if (sk.usesSkinning)     evType = FrameEvent::DIP_Skinning;
        else if (rs->blendEnable)     evType = FrameEvent::DIP_Blending;
        else if (rs->alphaTest)       evType = FrameEvent::DIP_AlphaTested;
        else                          evType = FrameEvent::DIP_Opaque;
        ImGuiManager::LogFrameEvent(evType, 0, rs->primCount);
    }

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
