
#include "proxydx/d3d8header.h"
#include "support/log.h"
#include "mgedinput.h"
#include "configuration.h"
#include "distantland.h"
#include "mged3d8device.h"
#include "mge_tracy.h"
#include "distantshader.h"
#include "postshaders.h"
#include "mwbridge.h"
#include "ffeshader.h"
#include "imgui_manager.h"
#include "patch_displacement.h"

#include <algorithm>
#include <cstdint>
#include <cstring>



using std::string;
using std::unordered_map;

PassBreakCounters g_passBreaks;

// File-scope ctx pointer for updatePostShader callback (set/cleared in postProcess)
static DLContext* s_postShaderCtx = nullptr;

// ============================================================================
// Frame State Logging - Auto-diff system to track state changes across frames
// ============================================================================
struct FrameStateSnapshot {
    int frameNum;
    // Menu/cache state
    bool isMenu;
    bool isRenderCached;
    // Environment state (captured)
    bool cellHasWater;
    bool cellHasWeather;
    bool isExterior;
    bool isUnderwater;
    float waterLevel;
    int postEnvFlags;
    bool hasWorldSpace;
    // Camera position
    float eyeZ;
    // Live mwBridge state (for comparison)
    bool live_isMenu;
    bool live_cellHasWater;
    bool live_cellHasWeather;
    bool live_isExterior;
    bool live_isUnderwater;
    float live_waterLevel;

    void clear() { memset(this, 0, sizeof(*this)); frameNum = -1; }
};

static FrameStateSnapshot s_prevState = {};
static FrameStateSnapshot s_currState = {};
static int s_frameStateLogCount = 0;

static void logFrameStateDiff(const char* tag, const DLContext* ctx) {
    auto mwBridge = MWBridge::get();
    if (!mwBridge || !mwBridge->IsLoaded()) {
        return;
    }

    int frameNum = getFrameNumber();

    // Capture current state
    s_currState.frameNum = frameNum;
    s_currState.isMenu = ctx ? ctx->isMenu : false;
    s_currState.isRenderCached = DistantLand::s_staging.isRenderCached;
    s_currState.cellHasWater = ctx ? ctx->cellHasWater : false;
    s_currState.cellHasWeather = ctx ? ctx->cellHasWeather : false;
    s_currState.isExterior = ctx ? ctx->isExterior : false;
    s_currState.isUnderwater = ctx ? ctx->isUnderwater : false;
    s_currState.waterLevel = ctx ? ctx->waterLevel : -1e9f;
    s_currState.postEnvFlags = ctx ? ctx->postEnvFlags : 0;
    s_currState.hasWorldSpace = ctx ? ctx->hasWorldSpace : false;
    s_currState.eyeZ = ctx ? ctx->eyePos.z : 0;

    // Capture live state
    s_currState.live_isMenu = mwBridge->IsMenu();
    s_currState.live_cellHasWater = mwBridge->CellHasWater();
    s_currState.live_cellHasWeather = mwBridge->CellHasWeather();
    s_currState.live_isExterior = mwBridge->IsExterior();
    s_currState.live_isUnderwater = mwBridge->IsUnderwater(ctx ? ctx->eyePos.z : 0);
    s_currState.live_waterLevel = mwBridge->CellHasWater() ? mwBridge->WaterLevel() : -1e9f;

    // Check for any differences from previous frame
    bool frameDiff = (s_prevState.frameNum != -1 && s_prevState.frameNum != s_currState.frameNum - 1);
    bool menuDiff = (s_prevState.frameNum != -1 && s_prevState.isMenu != s_currState.isMenu);
    bool cacheDiff = (s_prevState.frameNum != -1 && s_prevState.isRenderCached != s_currState.isRenderCached);
    bool waterDiff = (s_prevState.frameNum != -1 && s_prevState.cellHasWater != s_currState.cellHasWater);
    bool weatherDiff = (s_prevState.frameNum != -1 && s_prevState.cellHasWeather != s_currState.cellHasWeather);
    bool extDiff = (s_prevState.frameNum != -1 && s_prevState.isExterior != s_currState.isExterior);
    bool underDiff = (s_prevState.frameNum != -1 && s_prevState.isUnderwater != s_currState.isUnderwater);
    bool envFlagDiff = (s_prevState.frameNum != -1 && s_prevState.postEnvFlags != s_currState.postEnvFlags);
    bool worldDiff = (s_prevState.frameNum != -1 && s_prevState.hasWorldSpace != s_currState.hasWorldSpace);

    // Check for ctx vs live mismatches (the actual bug we're looking for)
    bool menuMismatch = (s_currState.isMenu != s_currState.live_isMenu);
    bool waterMismatch = (s_currState.cellHasWater != s_currState.live_cellHasWater);
    bool weatherMismatch = (s_currState.cellHasWeather != s_currState.live_cellHasWeather);
    bool extMismatch = (s_currState.isExterior != s_currState.live_isExterior);
    bool underMismatch = (s_currState.isUnderwater != s_currState.live_isUnderwater);

    bool anyDiff = frameDiff || menuDiff || cacheDiff || waterDiff || weatherDiff || extDiff || underDiff || envFlagDiff || worldDiff;
    bool anyMismatch = menuMismatch || waterMismatch || weatherMismatch || extMismatch || underMismatch;

    // Log on state change or mismatch (gated by SyncThread category)
    if (anyDiff || anyMismatch) {
        LOG_CAT(LOG::Cat_SyncThread, "[FSTATE][%s] frame=%d ctx: menu=%d cache=%d water=%d weather=%d ext=%d under=%d envFlags=0x%02X world=%d waterLvl=%.1f eyeZ=%.1f",
            tag, frameNum, s_currState.isMenu, s_currState.isRenderCached,
            s_currState.cellHasWater, s_currState.cellHasWeather, s_currState.isExterior, s_currState.isUnderwater,
            s_currState.postEnvFlags, s_currState.hasWorldSpace, s_currState.waterLevel, s_currState.eyeZ);

        LOG_CAT(LOG::Cat_SyncThread, "[FSTATE][%s] frame=%d LIVE: menu=%d water=%d weather=%d ext=%d under=%d waterLvl=%.1f",
            tag, frameNum, s_currState.live_isMenu, s_currState.live_cellHasWater,
            s_currState.live_cellHasWeather, s_currState.live_isExterior, s_currState.live_isUnderwater,
            s_currState.live_waterLevel);

        if (anyDiff && s_prevState.frameNum != -1) {
            LOG_CAT(LOG::Cat_SyncThread, "[FSTATE][%s] DIFF from frame %d: %s%s%s%s%s%s%s%s%s",
                tag, s_prevState.frameNum,
                frameDiff ? "FRAME_SKIP " : "",
                menuDiff ? "menu " : "",
                cacheDiff ? "cache " : "",
                waterDiff ? "water " : "",
                weatherDiff ? "weather " : "",
                extDiff ? "exterior " : "",
                underDiff ? "underwater " : "",
                envFlagDiff ? "envFlags " : "",
                worldDiff ? "world " : "");
        }

        if (anyMismatch) {
            LOG_CAT(LOG::Cat_SyncThread, "[FSTATE][%s] *** MISMATCH ctx!=live: %s%s%s%s%s ***",
                tag,
                menuMismatch ? "MENU " : "",
                waterMismatch ? "WATER " : "",
                weatherMismatch ? "WEATHER " : "",
                extMismatch ? "EXTERIOR " : "",
                underMismatch ? "UNDERWATER " : "");
        }
    }

    // Copy current to previous
    s_prevState = s_currState;
}
// File-scope pointer for updatePostShader to access captured MWBridge state
static const PostProcessData* s_postProcessData = nullptr;
// Skip velocity buffer writes on first frame after menu exit to preserve motion blur
static bool s_skipVelocityBufferThisFrame = false;
// Frame-consistent render decision - snapshotted at S0GPU entry, used by all stages
// Prevents mid-frame inconsistency when s_staging.isRenderCached changes during capture
static bool s_frameRenderCached = false;

bool DistantLand::shouldSkipVelocityBuffer() {
    return s_skipVelocityBufferThisFrame;
}

void DistantLand::logWaterDiagnostics(const char* tag, const DLContext* ctx, const PostProcessData* ppd) {
    if (!LOG::catEnabled(LOG::Cat_DistantLand)) {
        return;
    }
    auto mwBridge = MWBridge::get();
    if (!mwBridge || !mwBridge->IsLoaded() || mwBridge->IsExterior()) {
        return;
    }

    static int postLiveCounter = 0;
    static int postPpdCounter = 0;
    static int prepassCounter = 0;
    int* counter = &postPpdCounter;

    if (tag && std::strcmp(tag, "PREPASS") == 0) {
        counter = &prepassCounter;
    } else if (tag && std::strcmp(tag, "POST-LIVE") == 0) {
        counter = &postLiveCounter;
    }

    if (++(*counter) < 60) {
        return;
    }
    *counter = 0;

    const DWORD envCell = mwBridge->IntCurCellAddr();
    const BYTE envFlag = mwBridge->GetCellWaterFlag();
    const BYTE envMask73 = envFlag & 0x73;
    const BYTE envMaskF3 = envFlag & 0xF3;

    const DWORD playerCell = static_cast<DWORD>(reinterpret_cast<uintptr_t>(mwBridge->getPlayerCell()));
    BYTE playerFlag = 0xFF;
    BYTE playerMask73 = 0xFF;
    BYTE playerMaskF3 = 0xFF;
    int playerHasWater = -1;
    int playerHasWeather = -1;
    int playerLikeExterior = -1;
    float playerWaterLevel = -1e9f;

    if (playerCell != 0) {
        playerFlag = *reinterpret_cast<const BYTE*>(playerCell + 0x18);
        playerMask73 = playerFlag & 0x73;
        playerMaskF3 = playerFlag & 0xF3;
        playerHasWater = (playerMask73 == 0x13) ? 1 : 0;
        playerHasWeather = (playerMaskF3 == 0x93) ? 1 : 0;
        playerLikeExterior = (playerMaskF3 == 0x93) ? 1 : 0;
        if (playerHasWater) {
            playerWaterLevel = *reinterpret_cast<const float*>(playerCell + 0x90);
        }
    }

    const float playerZ = mwBridge->PlayerPositionZ();
    const float playerEyeEstimate = playerZ + 125.0f * mwBridge->PlayerHeight();
    const float eyeZ = ctx ? ctx->eyePos.z : playerEyeEstimate;
    const float ctxWater = ctx ? ctx->waterLevel : -1e9f;
    const int ctxEnvFlags = ctx ? ctx->postEnvFlags : 0;
    const float ppdWater = ppd ? ppd->waterLevel : -1e9f;
    const int ppdEnvFlags = ppd ? ppd->envFlags : 0;
    const int ppdInterior = ppd ? (ppd->isInterior ? 1 : 0) : -1;
    const int ppdUnderwater = ppd ? (ppd->isUnderwater ? 1 : 0) : -1;
    const char* interiorName = mwBridge->getInteriorName();

    LOG::logline(
        "[WATERDIAG][%s] cell='%s' isExt=%d envCell=0x%08X envFlag=0x%02X mask73=0x%02X maskF3=0x%02X cellHasWater=%d cellHasWeather=%d intLikeExt=%d intHasWater=%d water=%.2f",
        tag ? tag : "?", interiorName ? interiorName : "<ext>", mwBridge->IsExterior(), envCell, envFlag, envMask73, envMaskF3,
        mwBridge->CellHasWater(), mwBridge->CellHasWeather(), mwBridge->IntLikeExterior(), mwBridge->IntHasWater(), mwBridge->WaterLevel());
    LOG::logline(
        "[WATERDIAG][%s] playerCell=0x%08X playerFlag=0x%02X mask73=0x%02X maskF3=0x%02X playerHasWater=%d playerHasWeather=%d playerLikeExt=%d playerWater=%.2f eyeZ=%.2f playerZ=%.2f playerEye=%.2f underEye=%d underPlayerEye=%d ctxWater=%.2f ctxEnv=0x%02X ppdWater=%.2f ppdEnv=0x%02X ppdInt=%d ppdUnder=%d",
        tag ? tag : "?", playerCell, playerFlag, playerMask73, playerMaskF3, playerHasWater, playerHasWeather,
        playerLikeExterior, playerWaterLevel, eyeZ, playerZ, playerEyeEstimate, mwBridge->IsUnderwater(eyeZ),
        mwBridge->IsUnderwater(playerEyeEstimate), ctxWater, ctxEnvFlags, ppdWater, ppdEnvFlags, ppdInterior, ppdUnderwater);
}

// captureContext - Snapshot staging into a DLContext for this frame
DLContext DistantLand::captureContext() {
    return s_staging;
}

// captureStage0Context - CPU-only capture of per-frame context at start of Scene 0
// Called from DIP trigger (first non-sky draw). No GPU work.
DLContext DistantLand::captureStage0Context() {
    MGE_ZoneScopedN("DL_CaptureStage0");
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_Stage0, 0);
    // Frame header is only useful as a delimiter for spammy per-frame diagnostics.
    // Skip it when all log categories are off so an idle log isn't filled with empty headers.
    if (LOG::g_categoryMask != 0) {
        LOG::logline("======== FRAME %d START (Scene 0) ========", getFrameNumber());
    }
    FixedFunctionShader::transitionTo(PhaseTransition::RecordingEntry);

    // Phase tracking: mark frame capture start (effect uniforms, camera reads)
    FixedFunctionShader::setPhase(FixedFunctionShader::PipelinePhase::FrameCapture);
    FixedFunctionShader::frameCaptureGpuCalls.reset();
    FixedFunctionShader::recordingGpuCalls.reset();

    auto mwBridge = MWBridge::get();

    // Reset recording flag for new frame (Scene 0 starts a new frame)
    FixedFunctionShader::resetRecordingCompletedFlag();

    // Reset Hi-Z build flag for new frame
    FixedFunctionShader::resetHiZBuiltFlag();

    // Update current cell and select distant static set
    selectDistantCell();

    // Capture worldSpace into staging context for async Stage0Early (avoids global race)
    s_staging.worldSpace = DistantLandShare::currentWorldSpace;
    s_staging.hasWorldSpace = DistantLandShare::hasCurrentWorldSpace;

    // Morrowind camera matrices — already captured CPU-side by SetTransform handler
    // (no device->GetTransform needed, works with state forwarding suppression)

    // Set variables derived from current game state and camera configuration
    setView(&s_staging.mwView);
    adjustFog();
    forwardSSAOActive = false;

    bool wasRenderCached = s_staging.isRenderCached;
    s_staging.isRenderCached &= (Configuration.MGEFlags & USE_MENU_CACHING) && mwBridge->IsMenu();
    s_staging.isPPLActive = (Configuration.MGEFlags & USE_FFESHADER) && !(Configuration.PerPixelLightFlags == 1 && !mwBridge->IntCurCellAddr());

    // Menu caching with motion blur preservation:
    // On first menu frame, use the cached frame from N-1 (which has correct blur) instead of
    // rendering a new frame with mismatched depth/velocity buffers.
    bool isFirstMenuFrame = !wasRenderCached && mwBridge->IsMenu() && (Configuration.MGEFlags & USE_MENU_CACHING);
    if (isFirstMenuFrame && menuCacheValid) {
        s_staging.isRenderCached = true;  // Use existing cache, skip rendering
    }

    // Skip velocity buffer writes on menu exit frame to preserve blur for click-to-advance
    bool isMenuExitFrame = wasRenderCached && !s_staging.isRenderCached;
    s_skipVelocityBufferThisFrame = isMenuExitFrame;

    // Diagnostic: log when isRenderCached changes (no limit)
    if (LOG::catEnabled(LOG::Cat_DistantLand)) {
        static int captureLogCount = 0;
        if (captureLogCount < 200 && (wasRenderCached || s_staging.isRenderCached)) {
            LOG::logline("[CAPTURE] isRenderCached: %d -> %d (IsMenu=%d, USE_MENU_CACHING=%d)",
                wasRenderCached, s_staging.isRenderCached, mwBridge->IsMenu(),
                (Configuration.MGEFlags & USE_MENU_CACHING) ? 1 : 0);
            captureLogCount++;
        }
    }

    // Snapshot all per-frame state into context (foundation for threading)
    s_staging.postEnvFlags = 0;
    if (!mwBridge->CellHasWeather()) {
        s_staging.postEnvFlags |= 1;
    }
    if (mwBridge->IsExterior()) {
        s_staging.postEnvFlags |= 2;
    }
    if (mwBridge->IntLikeExterior()) {
        s_staging.postEnvFlags |= 4;
    }
    if (mwBridge->IsUnderwater(s_staging.eyePos.z)) {
        s_staging.postEnvFlags |= 8;
    } else {
        s_staging.postEnvFlags |= 16;
    }
    if (s_staging.sunVis >= 0.001f) {
        s_staging.postEnvFlags |= 32;
    } else {
        s_staging.postEnvFlags |= 64;
    }
    s_staging.waterLevel = mwBridge->CellHasWater() ? mwBridge->WaterLevel() : -1e9f;

    // Snapshot environment state for N-1 rendering (avoids mwBridge race conditions)
    s_staging.cellHasWater = mwBridge->CellHasWater();
    s_staging.cellHasWeather = mwBridge->CellHasWeather();
    s_staging.isExterior = mwBridge->IsExterior();
    s_staging.isUnderwater = mwBridge->IsUnderwater(s_staging.eyePos.z);
    s_staging.isMenu = mwBridge->IsMenu();

    // Log captured state at capture time
    if (LOG::catEnabled(LOG::Cat_DistantLand)) {
        int frameNum = getFrameNumber();
        LOG::logline("[FSTATE][CAPTURE] frame=%d staging: menu=%d cache=%d water=%d weather=%d ext=%d under=%d envFlags=0x%02X waterLvl=%.1f eyeZ=%.1f",
            frameNum, s_staging.isMenu, s_staging.isRenderCached,
            s_staging.cellHasWater, s_staging.cellHasWeather, s_staging.isExterior, s_staging.isUnderwater,
            s_staging.postEnvFlags, s_staging.waterLevel, s_staging.eyePos.z);
    }

    // Snapshot all per-frame state into context (foundation for threading)
    DLContext ctx = captureContext();

    // Note: setupCommonEffect moved to renderStage0GPU for N-1 rendering
    // It needs to use the rendering buffer's context, not the current capture
    FixedFunctionShader::updateLighting(ctx.lightSunMult, ctx.lightAmbMult);

    return ctx;
}

// renderStage0GPU - All GPU work from stage 0 (shadow map, distant land, sky, water reflection, wave sim)
// Called in render phase after recording is complete.
void DistantLand::renderStage0GPU(DLContext* ctx, FixedFunctionShader::FrameBuffer* fb) {
    MGE_ZoneScopedN("DL_RenderStage0GPU");
    logFrameStateDiff("S0GPU", ctx);

    // N-1 camera debug: Check if first recorded call's worldViewTransform matches ctx->mwView
    if (LOG::catEnabled(LOG::Cat_DistantLand)) {
        static int logCount = 0;
        if (fb && !fb->recordedCalls.empty() && logCount < 5) {
            // Extract the view translation embedded in worldViewTransforms
            // worldViewTransforms = world * view, so view._41 component is embedded
            auto& firstCall = fb->recordedCalls[0];
            float wvt_41 = firstCall.rs.worldViewTransforms[0]._41;
            float w_41 = firstCall.rs.worldTransforms[0]._41;
            // Embedded view contribution = worldView - world (approximate)
            LOG::logline("[WVT] wvt._41=%.1f world._41=%.1f ctx.view._41=%.1f fb.currentView._41=%.1f",
                wvt_41, w_41, ctx->mwView._41, fb->currentView._41);
            logCount++;
        }
    }

    FixedFunctionShader::transitionTo(PhaseTransition::Stage0Entry);

    // N-1: Setup effect uniforms with rendering buffer's context (not current frame's capture)
    // This ensures distant land renders with the same camera as the recorded main scene
    setupCommonEffect(ctx, &ctx->mwView, &ctx->mwProj);

    auto mwBridge = MWBridge::get();
    IDirect3DStateBlock9* stateSaved;
    UINT passes;

    // Snapshot render decision at S0GPU entry - use this for ALL stages in this frame
    // This prevents mid-frame inconsistency when s_staging.isRenderCached changes during capture
    s_frameRenderCached = s_staging.isRenderCached;

    // Menu caching uses CURRENT frame's state - must respond immediately to menu exit
    // WorldSpace uses N-1 buffer's snapshot (ctx) - thread-safe vs selectDistantCell race
    LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] s_frameRenderCached=%d ctx->hasWorldSpace=%d", s_frameRenderCached ? 1 : 0, ctx->hasWorldSpace ? 1 : 0);
    if (!s_frameRenderCached) {
        if (ctx->hasWorldSpace) {
            // Save state block manually since we can change FVF/decl
            LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] CreateStateBlock");
            {
                MGE_ZoneScopedN("DL_CreateStateBlock");
                device->CreateStateBlock(D3DSBT_ALL, &stateSaved);
            }
            LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] BeginPass SETUP");
            effect->BeginPass(PASS_SETUP);
            effect->EndPass();

            // Shadow map early render
            // Use s_staging.isMenu (current) not ctx->isMenu (N-1) for menu skip decision
            // This ensures shadows render on menu exit frame (responsive UI)
            LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] Shadow check flags=%d weather=%d menu=%d (staging=%d)",
                (Configuration.MGEFlags & USE_SHADOWS) ? 1 : 0,
                ctx->cellHasWeather ? 1 : 0,
                ctx->isMenu ? 1 : 0,
                s_staging.isMenu ? 1 : 0);
            if (Configuration.MGEFlags & USE_SHADOWS) {
                if (ctx->cellHasWeather && !s_staging.isMenu) {
                    MGE_ZoneScopedN("DL_ShadowMap");
                    LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] Shadow transitionTo");
                    FixedFunctionShader::transitionTo(PhaseTransition::ShadowEntry);
                    LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] Shadow effectBegin");
                    effectShadow->Begin(&passes, D3DXFX_DONOTSAVESTATE);
                    LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] Shadow renderShadowMap");
                    renderShadowMap(ctx);
                    LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] Shadow effectEnd");
                    g_passBreaks.mge_shadowRT += 2;
                    effectShadow->End();
                    FixedFunctionShader::transitionTo(PhaseTransition::ShadowExit);
                    LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] Shadow DONE");

                    // Write shadow viewproj back to s_staging for ffeshader/mged3d8device reads
                    memcpy(s_staging.smViewproj, ctx->smViewproj, sizeof(s_staging.smViewproj));
                }
            }

            // Distant everything; bias the projection matrix such that
            // distant land gets drawn behind anything Morrowind would draw
            D3DXMATRIX distProj = ctx->mwProj;
            editProjectionZ(&distProj, kDistantNearPlane - 1e-2, Configuration.DL.DrawDist * kCellSize);
            effect->SetMatrix(ehProj, &distProj);

            LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] effect->Begin");
            effect->Begin(&passes, D3DXFX_DONOTSAVESTATE);

            LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] underwater=%d exterior=%d", ctx->isUnderwater ? 1 : 0, ctx->isExterior ? 1 : 0);
            if (!ctx->isUnderwater) {
                // Draw distant landscape
                if (ctx->isExterior) {
                    MGE_ZoneScopedN("DL_Land");
                    LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] renderDistantLand START");
                    effect->BeginPass(PASS_RENDERLAND);
                    renderDistantLand(ctx, effect, &ctx->mwView, &distProj);
                    effect->EndPass();
                    LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] renderDistantLand DONE");
                }

                // Draw distant statics, with alpha dissolve as they pass the near view boundary
                if (Configuration.MGEFlags & USE_DISTANT_STATICS) {
                    MGE_ZoneScopedN("DL_Statics");
                    DWORD p = ctx->cellHasWeather ? PASS_RENDERSTATICSEXTERIOR : PASS_RENDERSTATICSINTERIOR;
                    effect->BeginPass(p);
                    vsr.beginAlphaToCoverage(device);

                    LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] cullDistantStatics START");
                    cullDistantStatics(ctx, &ctx->mwView, &distProj);
                    renderDistantStatics(ctx);

                    vsr.endAlphaToCoverage(device);
                    effect->EndPass();
                }
                else {
                    visDistant.RemoveAll();
                }
            }

            // Sky scattering and sky objects (should be drawn late as possible)
            // In HLSL mode, sky is always deferred here and rendered via shader
            if (ctx->cellHasWeather) {
                MGE_ZoneScopedN("DL_Sky");
                FixedFunctionShader::transitionTo(PhaseTransition::SkyEntry);
                const auto& sky = fb ? fb->recordSky : recordSky;
                if (!sky.empty()) {
                    // Always use shader path - renderSky handles ATM_SCATTER on/off
                    bool useAtmScatter = (Configuration.MGEFlags & USE_ATM_SCATTER) && !ImGuiManager::GetSuppressSky();
                    renderSky(sky, useAtmScatter);
                }
                FixedFunctionShader::transitionTo(PhaseTransition::SkyExit);
            }

            // Update reflection (use ctx flag for N-1 thread safety)
            if (ctx->cellHasWater) {
                MGE_ZoneScopedN("DL_WaterRefl");
                FixedFunctionShader::transitionTo(PhaseTransition::WaterReflEntry);
                const auto* sky = fb ? &fb->recordSky : nullptr;
                renderWaterReflection(ctx, &ctx->mwView, &distProj, sky);
                g_passBreaks.mge_waterRT += 2;
                FixedFunctionShader::transitionTo(PhaseTransition::WaterReflExit);
            }

            // Update water simulation
            if (Configuration.MGEFlags & DYNAMIC_RIPPLES) {
                MGE_ZoneScopedN("DL_WaterSim");
                simulateDynamicWaves();
                g_passBreaks.mge_waterRT += 4;
            }

            effect->End();

            // Reset matrices
            effect->SetMatrix(ehView, &ctx->mwView);
            effect->SetMatrix(ehProj, &ctx->mwProj);

            // Save distant land only frame to texture
            if (~Configuration.MGEFlags & NO_MW_MGE_BLEND) {
                IDirect3DTexture9* blendTex = PostShaders::borrowBuffer(1);
                LOG_CAT(LOG::Cat_SyncThread, "[S0GPU] texDistantBlend captured: tex=%p fb=%p", (void*)blendTex, (void*)fb);
                if (fb) {
                    fb->texDistantBlend = blendTex;
                } else {
                    texDistantBlend = blendTex;
                }
                g_passBreaks.mge_stretchRect++;
            }

            // Restore render state
            {
                MGE_ZoneScopedN("DL_StateRestore");
                stateSaved->Apply();
            }
            stateSaved->Release();
        } else {
            // Non-distant cell path (DL off or interior without distant statics)

            // Sky rendering - still needed even without DL (use ctx flag for N-1 thread safety)
            if (ctx->cellHasWeather) {
                // Save state block for sky rendering
                device->CreateStateBlock(D3DSBT_ALL, &stateSaved);
                effect->Begin(&passes, D3DXFX_DONOTSAVESTATE);

                FixedFunctionShader::transitionTo(PhaseTransition::SkyEntry);
                const auto& sky = fb ? fb->recordSky : recordSky;
                if (!sky.empty()) {
                    bool useAtmScatter = (Configuration.MGEFlags & USE_ATM_SCATTER) && !ImGuiManager::GetSuppressSky();
                    renderSky(sky, useAtmScatter);
                }
                FixedFunctionShader::transitionTo(PhaseTransition::SkyExit);

                effect->End();
                stateSaved->Apply();
                stateSaved->Release();
            }

            // Clear water reflection to avoid seeing previous cell environment reflected
            // Must be done every frame to react to lighting changes
            // Skip for cells without water to avoid unnecessary GPU work (use ctx flag for N-1 thread safety)
            if (ctx->cellHasWater) {
                clearReflection(ctx);
            }

            // Update water simulation
            if (Configuration.MGEFlags & DYNAMIC_RIPPLES) {
                // Save state block manually since we can change FVF/decl
                device->CreateStateBlock(D3DSBT_ALL, &stateSaved);

                effect->Begin(&passes, D3DXFX_DONOTSAVESTATE);
                simulateDynamicWaves();
                effect->End();

                // Restore render state
                stateSaved->Apply();

                stateSaved->Release();
            }
        }
    }

    // Clear stray sky recordings (but NOT recordMW - it's needed for depth rendering)
    // Per-buffer recordSky is cleared by FrameBuffer::clear(); only clear global static
    if (!fb) {
        recordSky.clear();
    }

    FixedFunctionShader::transitionTo(PhaseTransition::Stage0Exit);
}

// renderStage0 - Legacy combined path (calls capture + GPU)
DLContext DistantLand::renderStage0() {
    DLContext ctx = captureStage0Context();
    renderStage0GPU(&ctx);
    return ctx;
}

// renderEarlyZDepth - Write Morrowind scene depth to backbuffer depth-stencil AND surfDepthFrameMSAA
// This enables early-Z rejection for Stage0GPU (distant land, sky, etc.)
// Triple-duty: backbuffer depth (early-Z), depth texture (SSAO/DOF), velocity (motion blur)
void DistantLand::renderEarlyZDepth(DLContext* ctx, FixedFunctionShader::FrameBuffer* fb) {
    MGE_ZoneScopedN("DL_EarlyZDepth");

    // Toggle check - skip if early-Z disabled
    if (!ImGuiManager::GetEnableEarlyZ()) {
        return;
    }

    ImGuiManager::LogFrameEvent(FrameEvent::MGE_Depth, 0);

    // Skip if no render needed (same logic as renderStage1)
    if (s_frameRenderCached) {
        return;
    }

    auto mwBridge = MWBridge::get();
    IDirect3DStateBlock9* stateSaved;
    UINT passes;

    // Save state block
    device->CreateStateBlock(D3DSBT_ALL, &stateSaved);

    // Get backbuffer depth-stencil surface (Stage0GPU will use this)
    IDirect3DSurface9* backbufferDS = nullptr;
    device->GetDepthStencilSurface(&backbufferDS);

    // Save current render target
    IDirect3DSurface9* savedRT0 = nullptr;
    device->GetRenderTarget(0, &savedRT0);

    // Set up triple-duty MRT:
    // RT0 = surfDepthFrameMSAA (depth as color for SSAO/DOF)
    // RT1 = surfVelocityMSAA (velocity for motion blur)
    // Depth-stencil = backbuffer's depth (for early-Z benefit in Stage0GPU)
    device->SetRenderTarget(0, surfDepthFrameMSAA);
    device->SetDepthStencilSurface(backbufferDS);  // Key difference: use backbuffer depth!

    if (velocityBufferEnabled && surfVelocityMSAA && !shouldSkipVelocityBuffer()) {
        device->SetRenderTarget(1, surfVelocityMSAA);
    }

    // Clear both color (depth texture) and Z-buffer (backbuffer depth)
    device->Clear(0, 0, D3DCLEAR_ZBUFFER, 0, 1.0f, 0);
    g_passBreaks.raw_clear++;

    // Select recordMW source: per-buffer in HLSL mode, global static otherwise
    auto& activeRecordMW = fb ? fb->recordMW : recordMW;

    // Unbind depth sampler
    effect->SetTexture(ehTex3, NULL);

    // Projection should match main rendering
    D3DXMATRIX mwDepthProj = ctx->mwProj;
    if (Configuration.MGEFlags & USE_DISTANT_LAND) {
        editProjectionZ(&mwDepthProj, 1.0f, Configuration.DL.DrawDist * kCellSize);
    }
    effect->SetMatrix(ehProj, &mwDepthProj);

    // Push displacement falloff
    {
        D3DXMATRIX invView;
        D3DXMatrixInverse(&invView, nullptr, &ctx->mwView);
        D3DXVECTOR4 origin(0.0f, 0.0f, 0.0f, 1.0f);
        D3DXVECTOR4 eyeW;
        D3DXVec4Transform(&eyeW, &origin, &invView);
        D3DXVECTOR4 falloff(2560.0f, 1280.0f, eyeW.x, eyeW.y);
        effect->SetVector(ehDisplacementFalloff, &falloff);
    }

    effectDepth->Begin(&passes, D3DXFX_DONOTSAVESTATE);

    // Clear floating point depth buffer to far depth
    effectDepth->BeginPass(PASS_CLEARDEPTH);
    device->SetVertexDeclaration(WaterDecl);
    device->SetStreamSource(0, vbFullFrame, 0, 12);
    device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
    effectDepth->EndPass();

    // Render MW depth (Scene 0 only for early-Z)
    effectDepth->BeginPass(PASS_RENDERMWDEPTH);
    renderDepthRecorded(activeRecordMW, 0, &ctx->mwView, fb);  // Scene 0 only
    effectDepth->EndPass();

    // Displaced terrain depth
    if (fb && fb->nearPatchCount > 0) {
        effectDepth->BeginPass(PASS_RENDERMWDEPTHDISPLACED);
        renderDepthRecordedDisplaced(activeRecordMW, fb);
        effectDepth->EndPass();
    }

    effectDepth->End();

    // Clear RT1 after depth pass
    if (velocityBufferEnabled && surfVelocityMSAA && !shouldSkipVelocityBuffer()) {
        device->SetRenderTarget(1, nullptr);
    }

    // Restore render target (but KEEP backbuffer depth populated for Stage0GPU!)
    device->SetRenderTarget(0, savedRT0);
    // Restore original depth-stencil (should be same as backbufferDS, but be safe)
    device->SetDepthStencilSurface(backbufferDS);

    if (savedRT0) savedRT0->Release();
    if (backbufferDS) backbufferDS->Release();

    // MSAA resolve to non-MSAA textures (needed for SSAO/DOF/motion blur)
    if (Configuration.AALevel > 0) {
        MGE_ZoneScopedN("DL_EarlyZMSAAResolve");
        IDirect3DSurface9* texDepthFrameSurface;
        texDepthFrame->GetSurfaceLevel(0, &texDepthFrameSurface);
        device->StretchRect(surfDepthFrameMSAA, NULL, texDepthFrameSurface, NULL, D3DTEXF_NONE);
        g_passBreaks.mge_stretchRect++;
        g_passBreaks.raw_stretchRect++;
        texDepthFrameSurface->Release();

        // Resolve MSAA velocity buffer
        if (velocityBufferEnabled && texVelocity && surfVelocityMSAA && !shouldSkipVelocityBuffer()) {
            IDirect3DSurface9* texVelocitySurface;
            texVelocity->GetSurfaceLevel(0, &texVelocitySurface);
            device->StretchRect(surfVelocityMSAA, NULL, texVelocitySurface, NULL, D3DTEXF_NONE);
            g_passBreaks.mge_stretchRect++;
            g_passBreaks.raw_stretchRect++;
            texVelocitySurface->Release();
        }
    }

    // Reset projection matrix
    effect->SetMatrix(ehProj, &ctx->mwProj);

    // Restore render state
    stateSaved->Apply();
    stateSaved->Release();
}

// renderStage1 - Render grass and shadows over near features, and write depth texture for scene 0
void DistantLand::renderStage1(DLContext* ctx, FixedFunctionShader::FrameBuffer* fb) {
    MGE_ZoneScopedN("DL_RenderStage1");
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_Stage1, 0);
    FixedFunctionShader::transitionTo(PhaseTransition::Stage1Entry);
    auto mwBridge = MWBridge::get();
    IDirect3DStateBlock9* stateSaved;
    UINT passes;

    ///LOG::logline("Stage 1 prims: %d", recordMW.size());

    // Use frame-consistent render decision (snapshotted at S0GPU entry)
    if (!s_frameRenderCached) {
        // Save state block manually since we can change FVF/decl
        device->CreateStateBlock(D3DSBT_ALL, &stateSaved);

        // TODO: Locate this properly
        if (ctx->hasWorldSpace) {
            cullGrass(ctx, &ctx->mwView, &ctx->mwProj);
        }

        if (ctx->hasWorldSpace) {
            // Render over Morrowind domain
            effect->Begin(&passes, D3DXFX_DONOTSAVESTATE);

            // Draw grass with shadows
            if (Configuration.MGEFlags & USE_GRASS) {
                effect->BeginPass(PASS_RENDERGRASSINST);
                vsr.beginAlphaToCoverage(device);
                renderGrassInst(ctx);
                vsr.endAlphaToCoverage(device);
                effect->EndPass();
            }

            // Overlay shadow onto Morrowind objects (skip if HLSL shadows are handling it)
            // Use ctx flag for N-1 thread safety
            if ((Configuration.MGEFlags & USE_SHADOWS) && ctx->cellHasWeather && !isHLSLActive()) {
                effect->BeginPass(ctx->isPPLActive ? PASS_RENDERSHADOWFFE : PASS_RENDERSHADOW);
                renderShadow(ctx);
                effect->EndPass();
            }

            effect->End();
        }

        // Select recordMW source: per-buffer in HLSL mode, global static otherwise
        auto& activeRecordMW = fb ? fb->recordMW : recordMW;

        // Hi-Z culling: split into CPU-only visibility testing and lightweight recordMW filter
        // executeHiZCulling: bbox, occluder rasterization, Hi-Z pyramid, visibility test (no D3D device)
        // applyVisibilityAndFilterRecordMW: filters recordMW using visibility results
        // Skip if async or sync threading enabled - CpuPrepThread already did this work
        if (!ImGuiManager::GetAsyncGpuThread() && !ImGuiManager::GetSyncGpuThread() && isHLSLActive()) {
            FixedFunctionShader::executeHiZCulling(ctx->mwView, ctx->mwProj);
            FixedFunctionShader::applyVisibilityAndFilterRecordMW();
            // Build merged batches after culling (shouldRender flags now set)
            if (ImGuiManager::GetEnableStatelessBatch()) {
                FixedFunctionShader::buildStatelessBatches(FixedFunctionShader::getPrepBuffer());
            }
        }

        // Phase 8.4: build the near-patch SubdivPatch cache once per frame, here,
        // after Hi-Z culling has finalized fb->recordedCalls and fb->nearPatches
        // and BEFORE both the depth prepass (renderDepth below) and the color
        // replay (renderStage2). Both consumers then use findCached only — no
        // build races, one consistent edge map, no menu-mode flicker.
        if (fb && isHLSLActive()) {
            PatchDisplacement::prebuildNearPatches(
                device, *fb,
                ImGuiManager::GetDisplacementScale(),
                ImGuiManager::GetDisplacementGamma(),
                ImGuiManager::GetDisplacementPivot());
        }

        // Single RT switch for all depth rendering (renderDepth + StretchRect + renderDepthDistantLand + MSAA resolve)
        // When early-Z is enabled and sync mode active, renderEarlyZDepth already handled MW depth + velocity
        bool earlyZActive = ImGuiManager::GetSyncGpuThread() && ImGuiManager::GetEnableEarlyZ();
        {
            MGE_ZoneScopedN("DL_DepthSection");
            RenderTargetSwitcher rtsw(surfDepthFrameMSAA, surfDepthDepth);
            g_passBreaks.mge_depthRT += 2; // Single RenderTargetSwitcher in+out for entire depth section

            // Skip MW depth if early-Z already did it
            if (!earlyZActive) {
                // Set RT1 for velocity buffer (MRT with depth)
                // Skip on first frame after menu exit to preserve motion blur from before menu
                if (velocityBufferEnabled && surfVelocityMSAA && !s_skipVelocityBufferThisFrame) {
                    device->SetRenderTarget(1, surfVelocityMSAA);
                }

                // Depth texture from recorded renders
                // HLSL path: only render Scene 0 depth (Scene 2 hands handled by renderStage2)
                int sceneFilter = (isHLSLActive()) ? 0 : -1;
                FixedFunctionShader::transitionTo(PhaseTransition::DepthEntry);
                effectDepth->Begin(&passes, D3DXFX_DONOTSAVESTATE);
                if (ImGuiManager::GetEnableDepthPass()) {
                    renderDepth(ctx, activeRecordMW, sceneFilter, fb);
                }
                effectDepth->End();
                FixedFunctionShader::transitionTo(PhaseTransition::DepthExit);

                // Clear RT1 after depth pass (distant land doesn't need velocity)
                if (velocityBufferEnabled && surfVelocityMSAA && !s_skipVelocityBufferThisFrame) {
                    device->SetRenderTarget(1, nullptr);
                }
            }

            // Copy recordMW to Hi-Z for culling (before distant land adds to depth)
            // Disabled: Hi-Z generation is disabled, so this StretchRect is wasted
            // Re-enable when generateHiZMipsGPU() is active
            /*if (texCullDepth) {
                IDirect3DSurface9* cullDepthSurface;
                texCullDepth->GetSurfaceLevel(0, &cullDepthSurface);
                HRESULT hr = device->StretchRect(surfDepthFrameMSAA, NULL, cullDepthSurface, NULL, D3DTEXF_NONE);
                g_passBreaks.mge_stretchRect++;
                g_passBreaks.raw_stretchRect++;
                if (FAILED(hr)) {
                    static bool logged = false;
                    if (!logged) {
                        LOG::logline("!! Failed to copy depth to texCullDepth (hr=0x%x)", hr);
                        logged = true;
                    }
                }
                cullDepthSurface->Release();
            }*/

            // Continue depth with distant land (skip for interiors - no distant geometry)
            if (ctx->hasWorldSpace) {
                effectDepth->Begin(&passes, D3DXFX_DONOTSAVESTATE);
                if (ImGuiManager::GetEnableDepthPass()) {
                    renderDepthDistantLand(ctx);
                }
                effectDepth->End();
            }

            // Phase A: Resolve MSAA depth frame to non-MSAA texture for post-processing
            // When early-Z is active, MW depth was already resolved, but distant land depth
            // was just added, so we still need to re-resolve the combined depth
            if (Configuration.AALevel > 0) {
                MGE_ZoneScopedN("DL_DepthMSAAResolve");
                IDirect3DSurface9* texDepthFrameSurface;
                texDepthFrame->GetSurfaceLevel(0, &texDepthFrameSurface);
                device->StretchRect(surfDepthFrameMSAA, NULL, texDepthFrameSurface, NULL, D3DTEXF_NONE);
                g_passBreaks.mge_stretchRect++;
                g_passBreaks.raw_stretchRect++;
                texDepthFrameSurface->Release();

                // Resolve MSAA velocity buffer (only needed when early-Z didn't do it)
                if (!earlyZActive && velocityBufferEnabled && texVelocity && surfVelocityMSAA) {
                    IDirect3DSurface9* texVelocitySurface;
                    texVelocity->GetSurfaceLevel(0, &texVelocitySurface);
                    device->StretchRect(surfVelocityMSAA, NULL, texVelocitySurface, NULL, D3DTEXF_NONE);
                    g_passBreaks.mge_stretchRect++;
                    g_passBreaks.raw_stretchRect++;
                    texVelocitySurface->Release();
                }
            }
        }

        // Hi-Z generation DISABLED (culling disabled for baseline testing)
        // generateHiZMipsGPU();

        // Restore render state
        stateSaved->Apply();

        stateSaved->Release();
    }

    // HLSL path: keep recordMW for renderStage2 (Scene 2 hands depth still needed)
    // Legacy path: clear now (Scene 1/2 will re-populate during their own recording)
    // Per-buffer recordMW is cleared by FrameBuffer::clear(); only clear global static
    if (!fb && !isHLSLActive()) {
        recordMW.clear();
    }

    FixedFunctionShader::transitionTo(PhaseTransition::Stage1Exit);
}

// renderStage2 - Render depth texture for Scene 2 (hands after Z-clear)
void DistantLand::renderStage2(DLContext* ctx, FixedFunctionShader::FrameBuffer* fb) {
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_Stage2, 1);
    auto mwBridge = MWBridge::get();
    IDirect3DStateBlock9* stateSaved;
    UINT passes;

    // Select recordMW source: per-buffer in HLSL mode, global static otherwise
    auto& activeRecordMW = fb ? fb->recordMW : recordMW;

    // Legacy path reuses global recordMW for Scene 1/2 additions.
    // HLSL path only wants Scene 2+ here (hands and any later 1P passes).
    bool hasStage2Depth = !activeRecordMW.empty();
    if (hasStage2Depth && isHLSLActive()) {
        hasStage2Depth = std::any_of(activeRecordMW.begin(), activeRecordMW.end(),
            [](const RecordedMWState& call) { return call.sceneNum >= 2; });
    }
    if (!hasStage2Depth) {
        return;
    }

    // Use frame-consistent render decision (snapshotted at S0GPU entry)
    if (!s_frameRenderCached) {
        // Save state block manually since we can change FVF/decl
        device->CreateStateBlock(D3DSBT_ALL, &stateSaved);

        if (ctx->hasWorldSpace) {
            // Shadowing onto recorded renders (skip if HLSL shadows are handling it)
            // Use ctx flag for N-1 thread safety
            if ((Configuration.MGEFlags & USE_SHADOWS) && ctx->cellHasWeather && !isHLSLActive()) {
                effect->Begin(&passes, D3DXFX_DONOTSAVESTATE);
                effect->BeginPass(ctx->isPPLActive ? PASS_RENDERSHADOWFFE : PASS_RENDERSHADOW);
                renderShadow(ctx);
                effect->EndPass();
                effect->End();
            }
        }

        // Depth texture from recorded renders.
        // HLSL path: render Scene 2+ (hands and any later 1P passes); Scene 0 was already done in renderStage1.
        int sceneFilter = (isHLSLActive()) ? 2 : -1;
        effectDepth->Begin(&passes, D3DXFX_DONOTSAVESTATE);
        if (ImGuiManager::GetEnableDepthPass()) {
            // Pass hands view matrix for skinned depth rendering (Scene 2 uses different view)
            // Non-skinned objects use pre-recorded worldViewTransforms, so this mainly affects hands.
            const D3DXMATRIX* viewForDepth = (fb && sceneFilter >= 2) ? &fb->viewScene2 : nullptr;
            if (isHLSLActive() && Configuration.AALevel > 0) {
                // Stage 1 already resolved world depth into texDepthFrame.
                // Append Scene 2+ directly there with a fresh non-MSAA Z buffer so we don't rely on
                // the MSAA RT preserving its contents across the Stage 1 -> Stage 2 RT switch.
                IDirect3DSurface9* texDepthFrameSurface = nullptr;
                texDepthFrame->GetSurfaceLevel(0, &texDepthFrameSurface);
                renderDepthAdditional(ctx, activeRecordMW, sceneFilter, viewForDepth,
                    texDepthFrameSurface, surfDepthDepthResolved, true);
                texDepthFrameSurface->Release();
            } else {
                renderDepthAdditional(ctx, activeRecordMW, sceneFilter, viewForDepth);
            }
            g_passBreaks.mge_depthRT += 2; // RenderTargetSwitcher in+out
        }
        effectDepth->End();

        // Restore state
        stateSaved->Apply();

        stateSaved->Release();

        // Clear recordMW: per-buffer cleared by FrameBuffer::clear(), only clear global static
        if (!fb) {
            recordMW.clear();
        }
    }
}


// renderStageBlend - Blend between MGE distant land and Morrowind, rendering caustics first so it blends out
void DistantLand::renderStageBlend(DLContext* ctx, FixedFunctionShader::FrameBuffer* fb) {
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_StageBlend, 0);
    logFrameStateDiff("BLEND", ctx);
    FixedFunctionShader::transitionTo(PhaseTransition::StageBlendEntry);
    auto mwBridge = MWBridge::get();
    IDirect3DStateBlock9* stateSaved;
    UINT passes;

    // Use frame-consistent render decision (snapshotted at S0GPU entry)
    if (s_frameRenderCached) {
        return;
    }

    // Early out: skip state block overhead when no blend work will be done
    // Use ctx flag for N-1 thread safety
    bool hasCaustics = ctx->isExterior && Configuration.DL.WaterCaustics > 0;
    bool hasBlend = ctx->hasWorldSpace && (~Configuration.MGEFlags & NO_MW_MGE_BLEND);
    if (!hasCaustics && !hasBlend) {
        return;
    }

    // Save state block manually since we can change FVF/decl
    device->CreateStateBlock(D3DSBT_ALL, &stateSaved);
    effect->Begin(&passes, D3DXFX_DONOTSAVESTATE);

    // Render caustics (use ctx->waterLevel for N-1 thread safety)
    if (hasCaustics) {
        D3DXMATRIX m;
        IDirect3DTexture9* tex = PostShaders::borrowBuffer(0);
        D3DXMatrixTranslation(&m, ctx->eyePos.x, ctx->eyePos.y, ctx->waterLevel);

        effect->SetTexture(ehTex0, tex);
        effect->SetTexture(ehTex1, texWater);
        effect->SetTexture(ehTex3, texDepthFrame);
        effect->SetMatrix(ehWorld, &m);
        effect->SetFloat(ehAlphaRef, Configuration.DL.WaterCaustics);
        effect->CommitChanges();

        effect->BeginPass(PASS_RENDERCAUSTICS);
        PostShaders::applyBlend();
        effect->EndPass();
    }

    // Blend MW/MGE
    if (hasBlend) {
        IDirect3DTexture9* blendTex = fb ? fb->texDistantBlend : texDistantBlend;
        // Skip blend if texture is NULL (buffer was captured during menu mode when DL was skipped)
        if (!blendTex) {
            LOG_CAT(LOG::Cat_SyncThread, "[BLEND] SKIP: blendTex is NULL (fb=%p)", (void*)fb);
        } else {
            LOG_CAT(LOG::Cat_SyncThread, "[BLEND] using blendTex=%p fb=%p", (void*)blendTex, (void*)fb);
            effect->SetTexture(ehTex0, blendTex);
            effect->SetTexture(ehTex3, texDepthFrame);
            effect->CommitChanges();

            effect->BeginPass(PASS_BLENDMGE);
            PostShaders::applyBlend();
            effect->EndPass();
        }
    }

    effect->End();
    stateSaved->Apply();
    stateSaved->Release();

    FixedFunctionShader::transitionTo(PhaseTransition::StageBlendExit);
}

// renderStageWater - Render replacement water plane
void DistantLand::renderStageWater(DLContext* ctx) {
    MGE_ZoneScopedN("Water_Stage_DL");
    logFrameStateDiff("WATER", ctx);
    auto mwBridge = MWBridge::get();
    IDirect3DStateBlock9* stateSaved;
    UINT passes;

    // Diagnostic: log water state every 60 frames (1 per second at 60fps)
    if (LOG::catEnabled(LOG::Cat_DistantLand)) {
        static int waterLogFrame = 0;
        if (++waterLogFrame >= 60) {
            waterLogFrame = 0;
            DWORD cellAddr = mwBridge->IntCurCellAddr();
            BYTE waterFlag = mwBridge->GetCellWaterFlag();
            LOG::logline("[WATER] CellHasWater=%d flag=0x%02X (masked=0x%02X) addr=0x%08X IsExterior=%d",
                mwBridge->CellHasWater(), waterFlag, (waterFlag & 0x73), cellAddr, mwBridge->IsExterior());
        }
    }

    // Use ctx flags for N-1 thread safety (avoid mwBridge race conditions)
    if (ctx->cellHasWater) {
        // Save state block manually since we can change FVF/decl
        device->CreateStateBlock(D3DSBT_ALL, &stateSaved);
        effect->Begin(&passes, D3DXFX_DONOTSAVESTATE);

        // Draw water plane (use ctx flags, not live mwBridge state)
        bool u = ctx->isUnderwater;
        bool i = !ctx->isExterior;

        if (u || i) {
            // Set up clip plane at fog end for certain environments to save fillrate
            float clipAt = Configuration.DL.InteriorFogEnd * kCellSize;
            D3DXPLANE clipPlane(0, 0, -clipAt, ctx->mwProj._33 * clipAt + ctx->mwProj._43);
            device->SetClipPlane(0, clipPlane);
            device->SetRenderState(D3DRS_CLIPPLANEENABLE, 1);
        }

        // Switch to appropriate shader and render
        FixedFunctionShader::transitionTo(PhaseTransition::WaterPlaneEntry);
        effect->BeginPass(u ? PASS_RENDERUNDERWATER : PASS_RENDERWATER);
        renderWaterPlane(ctx);
        effect->EndPass();
        FixedFunctionShader::transitionTo(PhaseTransition::WaterPlaneExit);

        effect->End();
        stateSaved->Apply();

        stateSaved->Release();
    }
}

// setupCommonEffect - Set shared shader variables for this frame
void DistantLand::setupCommonEffect(DLContext* ctx, const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    auto mwBridge = MWBridge::get();

    // View position
    effect->SetMatrix(ehView, view);
    effect->SetMatrix(ehProj, proj);
    effect->SetFloatArray(ehEyePos, ctx->eyePos, 3);

    // Sunlight
    D3DXVECTOR3 sunVecView;
    RGBVECTOR totalAmb = ctx->sunAmb + ctx->ambCol;
    D3DXVec3TransformNormal(&sunVecView, (const D3DXVECTOR3*)&ctx->sunVec, view);

    effect->SetFloatArray(ehSunVec, ctx->sunVec, 3);
    effect->SetFloatArray(ehSunVecView, sunVecView, 3);
    effect->SetFloatArray(ehSunCol, ctx->sunCol, 3);
    effect->SetFloatArray(ehSunAmb, totalAmb, 3);
    effect->SetFloatArray(ehSunPos, ctx->sunPos, 3);
    effect->SetFloat(ehSunVis, ctx->sunVis);

    if (ctx->isPPLActive) {
        // Apply light multiplier settings to distant land
        RGBVECTOR s = ctx->lightSunMult * ctx->sunCol, a = ctx->lightAmbMult * totalAmb;
        effect->SetFloatArray(ehSunCol, s, 3);
        effect->SetFloatArray(ehSunAmb, a, 3);
    }

    // Sky/fog (use ctx flags for N-1 thread safety)
    bool isExpFog = (Configuration.MGEFlags & EXP_FOG) != 0;
    const RGBVECTOR* skyCol = ctx->cellHasWeather ? mwBridge->getCurrentWeatherSkyCol() : &ctx->horizonCol;
    effect->SetFloat(ehFogStart, isExpFog ? ctx->fogExpStart : ctx->fogStart);
    effect->SetFloat(ehFogRange, isExpFog ? ctx->fogExpDivisor : ctx->fogEnd);
    effect->SetFloat(ehFogNearStart, ctx->fogNearStart);
    effect->SetFloat(ehFogNearRange, ctx->fogNearEnd);
    effect->SetFloatArray(ehSkyCol, *skyCol, 3);
    effect->SetFloatArray(ehFogColNear, ctx->nearFogCol, 3);
    effect->SetFloatArray(ehFogColFar, ctx->horizonCol, 3);
    effect->SetFloat(ehNearViewRange, ctx->nearViewRange);
    if (ehIntensityScalar) {
        // Mirror the HLSL FFE intensityScalar so DL and FFE move together. The
        // parameter is only declared inside USE_HLSL_PIPELINE in the FX, so the
        // handle is null on legacy effect compiles and this is a no-op.
        effect->SetFloat(ehIntensityScalar, ImGuiManager::GetIntensityScalar());
        // Debug mode for shader visualization (shared between HLSL and PPL paths)
        if (ehDebugMode) {
            effect->SetInt(ehDebugMode, ImGuiManager::GetShaderDebugMode());
        }
    }
    if (LOG::catEnabled(LOG::Cat_DistantLand)) {
        static int nvLogCount = 0;
        if (nvLogCount++ < 10) {
            LOG::logline("[NVR] setupCommonEffect: nearViewRange=%.1f", ctx->nearViewRange);
        }
    }
    effect->SetFloat(ehNiceWeather, ctx->niceWeather);

    if (ehOutscatter) {
        effect->SetFloatArray(ehOutscatter, ctx->atmOutscatter, 3);
        effect->SetFloatArray(ehInscatter, ctx->atmInscatter, 3);
        effect->SetFloatArray(ehSkyScatterFar, ctx->atmSkylightScatter, 4);
    }

    // Wind, requires smoothing as it is very noisy
    // Use s_staging.isMenu (current) for responsive menu exit
    static float smoothWind[2];
    if (!s_staging.isMenu) {
        const float f = 0.02;
        const float* wind = mwBridge->GetWindVector();
        smoothWind[0] += f * (ctx->windScaling * wind[0] - smoothWind[0]);
        smoothWind[1] += f * (ctx->windScaling * wind[1] - smoothWind[1]);
        effect->SetFloatArray(ehWindVec, smoothWind, 2);
    }

    // Other
    effect->SetFloatArray(ehFootPos, (float*)mwBridge->PlayerPositionPointer(), 3);
    effect->SetFloat(ehTime, mwBridge->simulationTime());
}

// setScattering - Set scattering coefficients for atmospheric scattering shader
void DistantLand::setScattering(const RGBVECTOR& out, const RGBVECTOR& in) {
    s_staging.atmOutscatter = out;
    s_staging.atmInscatter = in;
}

static double lerp(double x0, double x1, double t) {
    return (1.0 - t) * x0 + t * x1;
}

static double saturate(double x) {
    return std::min(std::max(0.0, x), 1.0);
}

// adjustFog - Set fog distance, wind speed adjust and fog colour for this frame
void DistantLand::adjustFog() {
    auto mwBridge = MWBridge::get();

    s_staging.nearViewRange = mwBridge->GetViewDistance();

    // Morrowind does not update weather during menu mode, except when time is changed
    // Therefore always run adjustment during menu mode, except if background caching is used
    if (s_staging.isRenderCached) {
        return;
    }

    // Get fog cell ranges based on environment and weather
    if (mwBridge->IsUnderwater(s_staging.eyePos.z)) {
        s_staging.fogStart = Configuration.DL.BelowWaterFogStart;
        s_staging.fogEnd = Configuration.DL.BelowWaterFogEnd;
    } else if (mwBridge->CellHasWeather()) {
        int wthr1 = mwBridge->GetCurrentWeather(), wthr2 = mwBridge->GetNextWeather();
        float ratio = mwBridge->GetWeatherRatio(), ff = 1.0, fo = 0.0, ws = 0.0;

        if (ratio != 0 && wthr2 >= 0 && wthr2 <= 9) {
            ff = float(lerp(Configuration.DL.FogD[wthr1], Configuration.DL.FogD[wthr2], ratio));
            fo = float(0.01 * lerp(Configuration.DL.FgOD[wthr1], Configuration.DL.FgOD[wthr2], ratio));
            ws = float(lerp(Configuration.DL.Wind[wthr1], Configuration.DL.Wind[wthr2], ratio));
            s_staging.niceWeather = float(lerp((wthr1 <= 1) ? 1.0 : 0.0, (wthr2 <= 1) ? 1.0 : 0.0, ratio));
            s_staging.niceWeather *= s_staging.niceWeather;
            s_staging.lightSunMult = float(lerp(Configuration.Lighting.SunMult[wthr1], Configuration.Lighting.SunMult[wthr2], ratio));
            s_staging.lightAmbMult = float(lerp(Configuration.Lighting.AmbMult[wthr1], Configuration.Lighting.AmbMult[wthr2], ratio));
        } else if (wthr1 >= 0 && wthr1 <= 9) {
            ff = Configuration.DL.FogD[wthr1];
            fo = Configuration.DL.FgOD[wthr1] / 100.0f;
            ws = Configuration.DL.Wind[wthr1];
            s_staging.niceWeather = (wthr1 <= 1) ? 1.0f : 0.0f;
            s_staging.lightSunMult = Configuration.Lighting.SunMult[wthr1];
            s_staging.lightAmbMult = Configuration.Lighting.AmbMult[wthr1];
        }

        // Fog distance scale calculation, ensure fogEnd does not scale closer than vanilla Morrowind
        s_staging.fogEnd = std::max(0.875f, ff * Configuration.DL.AboveWaterFogEnd);
        s_staging.fogStart = ff * Configuration.DL.AboveWaterFogStart - fo * s_staging.fogEnd;
        s_staging.windScaling = ws;

        // For exp fog, adjust start distance so that starting fog approximately equals fo, to retain near visibility comparable to vanilla
        if ((Configuration.MGEFlags & USE_DISTANT_LAND) && (Configuration.MGEFlags & EXP_FOG)) {
            float lg = log(1.0f - 0.25f * fo);
            float expCorrection = lg / (1 + lg);
            s_staging.fogStart = ff * Configuration.DL.AboveWaterFogStart + expCorrection * s_staging.fogEnd;
        }
    } else {
        // Avoid density == 0, as when fogStart and fogEnd are equal, the fog equation denominator goes to infinity
        float density = std::max(0.01f, mwBridge->getInteriorFogDens());
        s_staging.fogStart = float(lerp(Configuration.DL.InteriorFogEnd, Configuration.DL.InteriorFogStart, density));
        s_staging.fogEnd = Configuration.DL.InteriorFogEnd;
        s_staging.niceWeather = 0;
        s_staging.windScaling = 0;
        s_staging.lightSunMult = 1.0;
        s_staging.lightAmbMult = 1.0;
    }

    // Convert from cells to in-game units
    s_staging.fogStart *= kCellSize;
    s_staging.fogEnd *= kCellSize;

    if ((Configuration.MGEFlags & USE_DISTANT_LAND) && isDistantCell()) {
        // Set hardware fog for Morrowind's use
        if (Configuration.MGEFlags & EXP_FOG) {
            // Exponential fog mode
            // Adjust exp curve so that at the fog end boundary, the same fog value is reached for all values of fogStart
            constexpr float expFogDistScale = 4.4f;
            s_staging.fogExpStart = s_staging.fogStart / expFogDistScale;
            s_staging.fogExpDivisor = (s_staging.fogEnd - s_staging.fogExpStart) / expFogDistScale;

            if (mwBridge->IsUnderwater(s_staging.eyePos.z) || !mwBridge->CellHasWeather()) {
                // Leave fog ranges as set, shaders use all linear fogging in this case
                s_staging.fogNearStart = s_staging.fogStart;
                s_staging.fogNearEnd = s_staging.fogEnd;
            } else {
                // Adjust near region linear Morrowind fogging to approximation of exp fog curve
                // Linear density matched to exp fog at dist = 1280 and dist = viewrange (or fog end if closer)
                // Note to self: Don't use saturate here or the denominators can become zero.
                float farIntercept = std::min(s_staging.fogEnd, s_staging.nearViewRange);
                float expFogNear = exp(-(1280.0f - s_staging.fogExpStart) / s_staging.fogExpDivisor);
                float expFogFar = exp(-(farIntercept - s_staging.fogExpStart) / s_staging.fogExpDivisor);
                s_staging.fogNearStart = 1280.0f + (farIntercept - 1280.0f) * (1.0f - expFogNear) / (expFogFar - expFogNear);
                s_staging.fogNearEnd = 1280.0f + (farIntercept - 1280.0f) * -expFogNear / (expFogFar - expFogNear);
            }
        } else {
            // Linear mode
            s_staging.fogNearStart = s_staging.fogStart;
            s_staging.fogNearEnd = s_staging.fogEnd;
        }

        device->SetRenderState(D3DRS_FOGSTART, *(DWORD*)&s_staging.fogNearStart);
        device->SetRenderState(D3DRS_FOGEND, *(DWORD*)&s_staging.fogNearEnd);
    } else {
        // Update fog when near render distance changes, and on startup when fogNearEnd == 0
        bool doFogUpdate = s_staging.fogNearEnd != s_staging.nearViewRange;

        // Read Morrowind-set fog range
        s_staging.fogNearEnd = s_staging.nearViewRange;
        s_staging.fogNearStart = s_staging.fogNearEnd * std::min(1.0f - mwBridge->getScenegraphFogDensity(), 0.99f);
        s_staging.fogStart = s_staging.fogNearStart;
        s_staging.fogEnd = s_staging.fogNearEnd;

        if (doFogUpdate) {
            device->SetRenderState(D3DRS_FOGSTART, *(DWORD*)&s_staging.fogNearStart);
            device->SetRenderState(D3DRS_FOGEND, *(DWORD*)&s_staging.fogNearEnd);
        }
    }

    // Adjust Morrowind fog colour towards scatter colour if necessary
    if ((Configuration.MGEFlags & USE_DISTANT_LAND) && (Configuration.MGEFlags & USE_ATM_SCATTER) && mwBridge->CellHasWeather() && !mwBridge->IsUnderwater(s_staging.eyePos.z)) {
        // Read unadjusted colour, as the scenegraph fog colour may not be updated during menu transitions
        RGBVECTOR c0 = *mwBridge->getCurrentWeatherFogCol();
        RGBVECTOR c1 = c0;

        // Simplified version of scattering from the shader
        const RGBVECTOR* skyCol = mwBridge->getCurrentWeatherSkyCol();
        const D3DXVECTOR3 newSkyCol = {
            float(lerp(skyCol->r, s_staging.atmSkylightScatter.x, s_staging.atmSkylightScatter.w)),
            float(lerp(skyCol->g, s_staging.atmSkylightScatter.y, s_staging.atmSkylightScatter.w)),
            float(lerp(skyCol->b, s_staging.atmSkylightScatter.z, s_staging.atmSkylightScatter.w))
        };
        const float sunaltitude = powf(1 + s_staging.sunPos.z, 10);
        const float sunaltitude_a = 2.8 + 4.3 / sunaltitude;
        const float sunaltitude_b = saturate(1.0 - exp2(-1.9 * sunaltitude));
        const float sunaltitude_c = saturate(exp(-4.0 * s_staging.sunPos.z)) * saturate(sunaltitude);

        // Calculate scatter colour at Morrowind draw distance boundary
        float fogdist = (s_staging.nearViewRange - s_staging.fogExpStart) / s_staging.fogExpDivisor;
        float fog = saturate(exp(-fogdist));
        fogdist = saturate(0.224 * fogdist);

        D3DXVECTOR2 horizonDir(s_staging.eyeVec.x, s_staging.eyeVec.y);
        D3DXVec2Normalize(&horizonDir, &horizonDir);
        float suncos =  horizonDir.x * s_staging.sunPos.x + horizonDir.y * s_staging.sunPos.y;
        float mie = (1.58 / (1.24 - suncos)) * sunaltitude_c;
        float rayl = 1.0 - 0.09 * mie;
        float atmdep = 1.33;

        D3DXVECTOR3 scatter;
        float scatterT = 0.5 * (1 + suncos);
        scatter.x = float(lerp(s_staging.atmInscatter.r, s_staging.atmOutscatter.r, scatterT));
        scatter.y = float(lerp(s_staging.atmInscatter.g, s_staging.atmOutscatter.g, scatterT));
        scatter.z = float(lerp(s_staging.atmInscatter.b, s_staging.atmOutscatter.b, scatterT));

        D3DXVECTOR3 att = atmdep * scatter * (sunaltitude_a + 0.7 * mie);
        att.x = (1 - exp(-fogdist * att.x)) / att.x;
        att.y = (1 - exp(-fogdist * att.y)) / att.y;
        att.z = (1 - exp(-fogdist * att.z)) / att.z;

        D3DXVECTOR3 k0 = mie * D3DXVECTOR3(0.125, 0.125, 0.125) + rayl * newSkyCol;
        D3DXVECTOR3 k1 = att * (1.17 * atmdep + 0.89) * sunaltitude_b;
        c1.r = k0.x * k1.x;
        c1.g = k0.y * k1.y;
        c1.b = k0.z * k1.z;

        // Convert from additive inscatter to Direct3D fog model
        // The correction factor is clamped to avoid creating infinities
        c1 /= std::max(0.02f, 1.0f - fog);

        // Scattering fog only occurs in nice weather
        c0 = (1.0f - s_staging.niceWeather) * c0 + s_staging.niceWeather * c1;

        // Save colour for matching near fog in shaders
        s_staging.nearFogCol = c0;

        // Alter Morrowind's fog colour through its scenegraph
        // This way it automatically restores the correct colour if it has to switch fog modes mid-frame
        DWORD fc = (DWORD)s_staging.nearFogCol;
        mwBridge->setScenegraphFogCol(fc);

        // Set device fog colour to propagate change immediately
        device->SetRenderState(D3DRS_FOGCOLOR, fc);
    } else {
        // Save current fog colour for matching near fog in shaders
        s_staging.nearFogCol = RGBVECTOR(mwBridge->getScenegraphFogCol());
    }
}

// postProcess - Calls post process module, or captures and applies frame cache to avoid rendering
void DistantLand::postProcess(DLContext* ctx) {
    MGE_ZoneScopedN("postProcess");
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_PostProcess, -1);
    logFrameStateDiff("POST", ctx);
    // Use frame-consistent render decision (snapshotted at S0GPU entry)
    if (!s_frameRenderCached) {
        auto mwBridge = MWBridge::get();

        // Save state block
        IDirect3DStateBlock9* stateSaved;
        device->CreateStateBlock(D3DSBT_ALL, &stateSaved);

        if (Configuration.MGEFlags & USE_HW_SHADER) {
            // Use captured envFlags from ctx for N-1 thread safety
            // ctx->postEnvFlags already has weather/exterior/underwater state from capture time

            // Run all shaders (with callback to set changed vars)
            s_postShaderCtx = ctx;
            logWaterDiagnostics("POST-LIVE", ctx, nullptr);
            PostShaders::shaderTime(&updatePostShader, ctx->postEnvFlags, mwBridge->frameTime());
            s_postShaderCtx = nullptr;
        }

        // Capture pre-UI screenshots here
        checkCaptureScreenshot(false);

        // Cache render for first frame of menu mode
        if ((Configuration.MGEFlags & USE_MENU_CACHING) && mwBridge->IsMenu()) {
            LOG_CAT(LOG::Cat_DistantLand, "[RENDERCACHE] Setting isRenderCached=true in postProcess (IsMenu=true)");
            texDistantBlend = PostShaders::borrowBuffer(0);
            s_staging.isRenderCached = true;
        }

        // Shadow map inset
        ///if(!mwBridge->IsMenu()) { renderShadowDebug(); }

        // Restore state
        stateSaved->Apply();

        stateSaved->Release();
    } else {
        // Blit cached frame to screen
        IDirect3DSurface9* backbuffer, *surfDistant;
        device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backbuffer);
        texDistantBlend->GetSurfaceLevel(0, &surfDistant);
        device->StretchRect(surfDistant, 0, backbuffer, 0, D3DTEXF_NONE);
        surfDistant->Release();
        backbuffer->Release();

        // Cache expires for frame after mouse click, so as not to affect click response time.
        // Also expires for one frame after any ImGui debug widget interaction, so changes
        // made in the debug window apply while the game is paused.
        s_staging.isRenderCached &= !MGEProxyDirectInput::mouseClick
                                  && !ImGuiManager::ConsumeRenderInvalidation();
    }
}

// postProcess overload using captured PostProcessData (render thread safe)
void DistantLand::postProcess(DLContext* ctx, const PostProcessData& ppd) {
    MGE_ZoneScopedN("postProcess");
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_PostProcess, -1);
    logFrameStateDiff("POST2", ctx);
    // Use frame-consistent render decision (snapshotted at S0GPU entry)
    if (!s_frameRenderCached) {
        // Save state block
        IDirect3DStateBlock9* stateSaved;
        device->CreateStateBlock(D3DSBT_ALL, &stateSaved);

        if (Configuration.MGEFlags & USE_HW_SHADER) {
            // Run all shaders (with callback to set changed vars)
            s_postShaderCtx = ctx;
            s_postProcessData = &ppd;
            logWaterDiagnostics("POST-PPD", ctx, &ppd);
            PostShaders::shaderTime(&updatePostShader, ppd.envFlags, ppd.frameTime);
            s_postProcessData = nullptr;
            s_postShaderCtx = nullptr;
        }

        // Capture pre-UI screenshots here
        checkCaptureScreenshot(false);

        // Deferred menu caching: this block runs only after a live render (isRenderCached
        // was false this frame), so the backbuffer holds fresh post-process content.
        // Always copy it into surfMenuCache — this also refreshes the cache after a
        // mid-menu invalidation (mouse click, ImGui debug change), so re-renders persist.
        if ((Configuration.MGEFlags & USE_MENU_CACHING) && surfMenuCache) {
            IDirect3DSurface9* backbuffer = nullptr;
            device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backbuffer);
            if (backbuffer) {
                HRESULT hr = device->StretchRect(backbuffer, nullptr, surfMenuCache, nullptr, D3DTEXF_NONE);
                if (SUCCEEDED(hr)) {
                    menuCacheValid = true;
                    if (ppd.isMenu) {
                        s_staging.isRenderCached = true;
                    }
                }
                backbuffer->Release();
            }
        }

        // Restore state
        stateSaved->Apply();
        stateSaved->Release();
    } else {
        // Blit cached frame to screen
        IDirect3DSurface9* backbuffer = nullptr;
        device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backbuffer);
        if (backbuffer && surfMenuCache && menuCacheValid) {
            device->StretchRect(surfMenuCache, nullptr, backbuffer, nullptr, D3DTEXF_NONE);
        }
        if (backbuffer) {
            backbuffer->Release();
        }

        // Cache expires for frame after mouse click, so as not to affect click response time.
        // Also expires for one frame after any ImGui debug widget interaction, so changes
        // made in the debug window apply while the game is paused.
        s_staging.isRenderCached &= !MGEProxyDirectInput::mouseClick
                                  && !ImGuiManager::ConsumeRenderInvalidation();
    }
}

// updatePostShader - callback for setting post shader variables based on environment
void DistantLand::updatePostShader(MGEShader* shader) {
    const DLContext* ctx = s_postShaderCtx;

    // Internal textures
    // TODO: Should be set once at init time
    shader->SetTexture(EV_depthframe, texDepthFrame);
    shader->SetTexture(EV_watertexture, texWater);
    if (velocityBufferEnabled && texVelocity) {
        shader->SetTexture(EV_velocityframe, texVelocity);
    }

    // View position
    float zoom = (Configuration.MGEFlags & ZOOM_ASPECT) ? Configuration.CameraEffects.zoom : 1.0f;
    shader->SetMatrix(EV_mview, &ctx->mwView);
    shader->SetMatrix(EV_mproj, &ctx->mwProj);
    shader->SetFloatArray(EV_eyevec, ctx->eyeVec, 3);
    shader->SetFloatArray(EV_eyepos, ctx->eyePos, 3);
    shader->SetFloat(EV_fov, Configuration.ScreenFOV / zoom);

    // Lighting
    RGBVECTOR totalAmb = ctx->sunAmb + ctx->ambCol;
    shader->SetFloatArray(EV_sunvec, ctx->sunVec, 3);
    shader->SetFloatArray(EV_suncol, ctx->sunCol, 3);
    shader->SetFloatArray(EV_sunamb, totalAmb, 3);
    shader->SetFloatArray(EV_sunpos, ctx->sunPos, 3);
    shader->SetFloat(EV_sunvis, float(lerp(ctx->sunVis, 1.0, 0.333 * ctx->niceWeather)));

    // Sky/fog
    bool isExpFog = (Configuration.MGEFlags & EXP_FOG) != 0;
    shader->SetFloatArray(EV_fogcol, ctx->horizonCol, 3);
    shader->SetFloatArray(EV_fognearcol, ctx->nearFogCol, 3);
    shader->SetFloat(EV_fogstart, isExpFog ? ctx->fogExpStart : ctx->fogStart);
    shader->SetFloat(EV_fogrange, isExpFog ? ctx->fogExpDivisor : ctx->fogEnd);
    shader->SetFloat(EV_fognearstart, ctx->fogNearStart);
    shader->SetFloat(EV_fognearrange, ctx->fogNearEnd);

    // Other — use captured PostProcessData when available, else use ctx flags for N-1 safety
    if (s_postProcessData) {
        shader->SetFloat(EV_time, s_postProcessData->simulationTime);
        shader->SetFloat(EV_waterlevel, s_postProcessData->waterLevel);
        shader->SetBool(EV_isinterior, s_postProcessData->isInterior);
        shader->SetBool(EV_isunderwater, s_postProcessData->isUnderwater);
    } else {
        // Use ctx flags for N-1 thread safety (only simulationTime needs live access)
        auto mwBridge = MWBridge::get();
        shader->SetFloat(EV_time, mwBridge->simulationTime());
        shader->SetFloat(EV_waterlevel, ctx->waterLevel);
        shader->SetBool(EV_isinterior, !ctx->cellHasWeather);
        shader->SetBool(EV_isunderwater, ctx->isUnderwater);
    }
}

//------------------------------------------------------------

// selectDistantCell - Select the correct set of distant land meshes for the current cell
bool DistantLand::selectDistantCell() {
    auto mwBridge = MWBridge::get();

    if (Configuration.MGEFlags & USE_DISTANT_LAND) {
        // Scan dynamic vis on cell change
        void* playerCell = mwBridge->getPlayerCell();
        if (playerCell != lastDistantVisCell) {
            scanDynamicVisGroups();
            lastDistantVisCell = playerCell;
        }

        // Get worldspace key
        string cellname;
        if (mwBridge->IsExterior()) {
            cellname = string();
        }
        else {
            cellname = mwBridge->getInteriorName();
        }

        if (Configuration.UseSharedMemory) {
            DistantLandShare::hasCurrentWorldSpace = ipcClient.setWorldSpaceBlocking(cellname);
            if (DistantLandShare::hasCurrentWorldSpace) {
                return true;
            }
        } else {
            const auto iWS = DistantLandShare::mapWorldSpaces.find(cellname);
            if (iWS != DistantLandShare::mapWorldSpaces.end()) {
                DistantLandShare::currentWorldSpace = &iWS->second;
                DistantLandShare::hasCurrentWorldSpace = true;
                return true;
            }
        }
    }

    DistantLandShare::currentWorldSpace = nullptr;
    DistantLandShare::hasCurrentWorldSpace = false;
    return false;
}

// isDistantCell - Check if there is distant land selected for this cell
bool DistantLand::isDistantCell() {
    return DistantLandShare::hasCurrentWorldSpace;
}

// resolveDynamicVisGroups - Resolve pointers to game objects on load/reload
void DistantLand::resolveDynamicVisGroups() {
    auto mwBridge = MWBridge::get();
    const DynamicVisGroup *lastDVG = nullptr;

    for (auto& vis : dynamicVisGroups) {
        // Clear previous pointer
        vis.gameObject = nullptr;

        // Re-use previous result if the id matches
        if (lastDVG && vis.id == lastDVG->id) {
            vis.gameObject = lastDVG->gameObject;
            continue;
        }
        else {
            lastDVG = &vis;
        }

        // Resolve IDs to pointers
        switch (vis.source) {
        case DynamicVisGroup::DataSource::Journal:
            vis.gameObject = mwBridge->getDialogue(vis.id.c_str());
            break;
        case DynamicVisGroup::DataSource::Global:
            vis.gameObject = mwBridge->getGlobalVar(vis.id.c_str());
            break;
        case DynamicVisGroup::DataSource::UniqueObject:
            vis.gameObject = mwBridge->findFirstReferenceById(vis.id.c_str());
            break;
        }
    }

    // Ensure reloading into the same cell still triggers updates
    lastDistantVisCell = nullptr;
}

// scanDynamicVisGroups - Scan through game data for visibility changes
void DistantLand::scanDynamicVisGroups() {
    auto mwBridge = MWBridge::get();
    if (Configuration.UseSharedMemory) {
        dynVisFlagsShared.clear();
    }

    std::uint16_t i = 0;
    for (auto& vis : dynamicVisGroups) {
        int value;
        auto groupIndex = i++;

        // Ignore unresolved objects
        if (!vis.gameObject) {
            continue;
        }

        switch (vis.source) {
        case DynamicVisGroup::DataSource::Journal:
            value = mwBridge->getJournalIndex(vis.gameObject);
            break;
        case DynamicVisGroup::DataSource::Global:
            value = int(mwBridge->getGlobalVarValue(vis.gameObject));
            break;
        case DynamicVisGroup::DataSource::UniqueObject:
            const int disabledRecordFlag = 0x800;
            value = (mwBridge->getRecordFlags(vis.gameObject) & disabledRecordFlag) == 0;
            break;
        }

        // Enable if value is inside any range
        bool enable = false;
        for (const auto& r : vis.ranges) {
            if (r.begin <= value && value < r.end) {
                enable = true;
                break;
            }
        }

        // If enable state has changed, propagate to distant land mesh instances
        if (enable ^ vis.enabled) {
            vis.enabled = enable;
            if (Configuration.UseSharedMemory) {
                dynVisFlagsShared.push_back({ groupIndex, enable });
            } else {
                for (auto& m : vis.references) {
                    m->enabled = enable;
                }
            }
        }
    }

    if (Configuration.UseSharedMemory && !dynVisFlagsShared.empty()) {
        ipcClient.updateDynVis(dynVisFlagsSharedId);
    }
}

// setView - Called once per frame to setup view dependent data
void DistantLand::setView(const D3DMATRIX* m) {
    auto mwBridge = MWBridge::get();

    // Calculate eyePos, eyeVec for shaders
    D3DXVECTOR4 origin(0.0, 0.0, 0.0, 1.0);
    D3DXMATRIX invView, view = *m;

    D3DXMatrixInverse(&invView, 0, &view);
    D3DXVec4Transform(&s_staging.eyePos, &origin, &invView);
    s_staging.eyeVec.x = m->_13;
    s_staging.eyeVec.y = m->_23;
    s_staging.eyeVec.z = m->_33;

    // Set sun disc position
    if (mwBridge->IsLoaded() && mwBridge->CellHasWeather()) {
        mwBridge->GetSunDir(s_staging.sunPos.x, s_staging.sunPos.y, s_staging.sunPos.z);
        s_staging.sunPos.w = 1;
        s_staging.sunPos /= sqrt(s_staging.sunPos.x * s_staging.sunPos.x + s_staging.sunPos.y * s_staging.sunPos.y + s_staging.sunPos.z * s_staging.sunPos.z);

        // Sun position "bounces" at the horizon to follow night lighting instead of setting
        // Sun visibility goes to zero at night, so use this to correct the sun position so it sets
        s_staging.sunVis = mwBridge->GetSunVis() / 255.0f;
        if (s_staging.sunVis == 0) {
            s_staging.sunPos.z = -s_staging.sunPos.z;
        }
    } else {
        s_staging.sunPos = D3DXVECTOR4(0, 0, -1, 1);
        s_staging.sunVis = 0;
    }
}

// setProjection - Called when a D3D projection matrix is set, and edits it
void DistantLand::setProjection(D3DMATRIX* proj) {
    // Move near plane from 1.0 to 4.0 for more z accuracy
    // Move far plane back to edge of draw distance
    if (Configuration.MGEFlags & USE_DISTANT_LAND) {
        editProjectionZ(proj, kDistantNearPlane, Configuration.DL.DrawDist * kCellSize);
    }
}

// editProjectionZ - Alter the near and far clip planes of a projection matrix
void DistantLand::editProjectionZ(D3DMATRIX* m, float zn, float zf) {
    // Override near and far clip planes
    m->_33 = zf / (zf - zn);
    m->_43 = -zn * zf / (zf - zn);
}

void DistantLand::setHorizonColour(const RGBVECTOR& c) {
    s_staging.horizonCol = c;
}

void DistantLand::setAmbientColour(const RGBVECTOR& c) {
    s_staging.ambCol = c;
}

void DistantLand::setSunLight(const D3DLIGHT8* s) {
    // Sun is used for both interiors and exteriors; the sun in interiors is a fixed light
    s_staging.sunVec.x = s->Direction.x;
    s_staging.sunVec.y = s->Direction.y;
    s_staging.sunVec.z = s->Direction.z;
    D3DXVec3Normalize((D3DXVECTOR3*)&s_staging.sunVec, (D3DXVECTOR3*)&s_staging.sunVec);

    s_staging.sunCol = s->Diffuse;
    s_staging.sunAmb = s->Ambient;
}

// inspectIndexedPrimitive
// Filters and records DIP calls for later use; returning false should cause the draw call to be skipped
// Can also replace selected fixed function calls with an augmented shader
bool DistantLand::inspectIndexedPrimitive(int sceneCount, const RenderedState* rs, const FragmentState* frs, LightState* lightrs) {
    auto mwBridge = MWBridge::get();

    // Phase-based detection with sceneCount fallback for robustness
    // (phase may not be set correctly on first frame or mode switch)
    bool isWorldPhase = (g_scene.phase == ScenePhase::World) || (sceneCount == 0);
    bool isParticlesPhase = (g_scene.phase == ScenePhase::Particles) || (sceneCount == 1 && g_scene.phase != ScenePhase::World);
    bool isHandsPhase = (g_scene.phase == ScenePhase::Hands) || (sceneCount >= 2 && g_scene.phase != ScenePhase::UI);
    bool isParticlesOrHands = isParticlesPhase || isHandsPhase;

    // Log ALL Particles/Hands draws to debug (one-time dump)
    static int scene12DrawCount = 0;
    static bool scene12LogDone = false;
    if (isParticlesOrHands && !scene12LogDone) {
        scene12DrawCount++;
        LOG_CAT(LOG::Cat_DistantLand, ">> %s draw #%d: zWrite=%d, blendEnable=%d, alphaTest=%d, vertBlend=%d, prims=%d",
                     isParticlesPhase ? "Particles" : "Hands", scene12DrawCount, rs->zWrite, rs->blendEnable, rs->alphaTest,
                     rs->vertexBlendState, rs->primCount);
    }
    // After first frame finalize, stop logging
    if (isWorldPhase && scene12DrawCount > 0) {
        scene12LogDone = true;
    }

    // Avoid recording landscape alpha blend drawcalls, a form of multi-pass splatting
    static IDirect3DVertexBuffer9* lastVB = nullptr;
    bool isLandSplat = isWorldPhase && rs->vb == lastVB && rs->blendEnable && (rs->fvf & D3DFVF_DIFFUSE) && mwBridge->IsExterior();
    lastVB = rs->vb;

    // Avoid recording decal passes from UV sets >0, shadow rendering only samples alpha from texture 0 with UV 0
    const auto& stage0 = frs->stage[0];
    bool isDecal = stage0.texcoordIndex != 0 && (stage0.colorArg1 == D3DTA_TEXTURE || stage0.colorArg2 == D3DTA_TEXTURE);

    // Track index of entry added to recordMW (local variable, not global state)
    int recordMWIdx = -1;

    // Select target recordMW/recordSky: per-buffer in HLSL mode, global static otherwise
    // N-1: recording goes into recording buffer
    bool hlsl = isHLSLActive();
    auto& targetRecordMW = hlsl ? FixedFunctionShader::getRecordingBuffer().recordMW : recordMW;
    auto& targetRecordSky = hlsl ? FixedFunctionShader::getRecordingBuffer().recordSky : recordSky;

    // Capture z-writing draws, plus Hands (for depth texture even if zWrite=0)
    // World: only zWrite draws (skip multi-pass splatting and decals)
    // Particles: vertexBlendState == 0 && blendEnable → skip depth (alpha sorted)
    // Hands: vertexBlendState != 0 (skinned) or !blendEnable (opaque) → depth for SSAO/DOF
    bool is1PDepthCandidate = isParticlesOrHands && (rs->vertexBlendState != 0 || !rs->blendEnable);
    if ((rs->zWrite && !isLandSplat && !isDecal) || is1PDepthCandidate) {
        targetRecordMW.emplace_back(*rs);

        // Unify alpha test operator/reference to be equivalent to GREATEREQUAL
        if (rs->alphaFunc == D3DCMP_GREATER) {
            targetRecordMW.back().alphaRef++;
        }

        // Don't compute bboxes here - too expensive (buffer locks cause stalls)
        // Bboxes will be computed on-the-fly during culling if needed
        targetRecordMW.back().hasBoundingBox = false;
        targetRecordMW.back().sceneNum = sceneCount;

        // Store index of just-added entry (for HLSL recording to reuse visibility)
        recordMWIdx = static_cast<int>(targetRecordMW.size() - 1);
    }

    // Special case, capture sky (World phase only, before any opaques recorded)
    if (targetRecordMW.empty() && rs->blendEnable && isWorldPhase && mwBridge->CellHasWeather()) {
        ImGuiManager::LogFrameEvent(FrameEvent::DIP_Sky, sceneCount, rs->primCount);
        ImGuiManager::IncrementSkyStat();
        targetRecordSky.emplace_back(*rs);

        // Check for moon geometry, and mark those records by setting lighting off
        if (frs->material.emissive.a == kMoonTag) {
            targetRecordSky.back().useLighting = false;
        }

        // Mark sky for wireframe debug (rendered in GPU phase, not here)
        if (ImGuiManager::GetSuppressSky()) {
            targetRecordSky.back().debugWireframe = true;
        }

        // HLSL: Always defer sky for async safety (both ATM_SCATTER and FFP paths)
        // Sky will be rendered in renderStage0GPU or renderSkyFFP
        if (isHLSLActive()) {
            return false;  // Defer to GPU phase
        }

        // Legacy: defer sky only if using atmosphere scattering
        if ((Configuration.MGEFlags & USE_DISTANT_LAND) && (Configuration.MGEFlags & USE_ATM_SCATTER)) {
            return false;
        }

        // Legacy FFP sky without ATM_SCATTER - fall through to immediate render
        // (This is expected behavior in legacy mode, not a stray)
    } else if (s_staging.isPPLActive) {
        // Event logging deferred to recordRenderCall where exact RenderBin is known
        // Render Morrowind with replacement shaders
        // Pass recordMWIdx so HLSL recording can reuse visibility results from depth pass
        FixedFunctionShader::renderMorrowind(rs, frs, lightrs, recordMWIdx);
        return false;
    } else if (isParticlesOrHands && isHLSLActive() && FixedFunctionShader::getIsRecording()) {
        // Particles/Hands: record to HLSL for deferred replay
        // This path is separate from isPPLActive to allow recording even in non-PPL modes
        FixedFunctionShader::renderMorrowind(rs, frs, lightrs, recordMWIdx);
        return false;
    }

    return true;
}

// requestCapture - Set a function to be called with a screen capture
// Either before the UI is drawn, or after UI and before MGE messages
void DistantLand::requestCapture(std::function<void(IDirect3DSurface9*)> handler, bool captureWithUI) {
    captureScreenHandler = handler;
    captureScreenWithUI = captureWithUI;
}

void DistantLand::checkCaptureScreenshot(bool isUIDrawn) {
    if (bool(captureScreenHandler) && captureScreenWithUI == isUIDrawn) {
        IDirect3DSurface9* surface = captureScreenshot();
        captureScreenHandler(surface);
        if (surface) {
            surface->Release();
        }
        captureScreenHandler = nullptr;
    }
}

// captureScreen - Capture a screenshot
IDirect3DSurface9* DistantLand::captureScreenshot() {
    IDirect3DTexture9* t;
    IDirect3DSurface9* s;

    // Resolve multisampled back buffer
    t = PostShaders::borrowBuffer(0);
    t->GetSurfaceLevel(0, &s);

    // Cancel render cache, borrowBuffer just overwrote it
    s_staging.isRenderCached = false;

    // Copy buffer to system memory surface
    IDirect3DSurface9* surfSS;
    D3DSURFACE_DESC desc;

    s->GetDesc(&desc);
    DWORD hr = device->CreateOffscreenPlainSurface(desc.Width, desc.Height, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, &surfSS, NULL);
    if (hr != D3D_OK) {
        s->Release();
        return nullptr;
    }

    hr = device->GetRenderTargetData(s, surfSS);
    s->Release();
    if (hr != D3D_OK) {
        surfSS->Release();
        return nullptr;
    }

    return surfSS;
}


// ------------------------------------
// DistantLand::RecordedState

RecordedMWState::RecordedMWState(const RenderedState& state)
    : RenderedState(state) {
    vb->AddRef();
    ib->AddRef();
    if (texture) {
        texture->AddRef();
    }
}

RecordedMWState::~RecordedMWState() {
    if (vb) {
        vb->Release();
    }
    if (ib) {
        ib->Release();
    }
    if (texture) {
        texture->Release();
    }
}

RecordedMWState::RecordedMWState(RecordedMWState&& source) noexcept
    : RenderedState(source) {
    source.vb = nullptr;
    source.ib = nullptr;
    source.texture = nullptr;
}


// ------------------------------------
// RenderTargetSwitcher

// RenderTargetSwitcher - Switch to a render target, restoring state at end of scope
RenderTargetSwitcher::RenderTargetSwitcher(IDirect3DSurface9* target, IDirect3DSurface9* targetDepthStencil) {
    init(target, targetDepthStencil);
}

// RenderTargetSwitcher - Switch to a render surface belonging to a texture, restoring state at end of scope
RenderTargetSwitcher::RenderTargetSwitcher(IDirect3DTexture9* targetTex, IDirect3DSurface9* targetDepthStencil) {
    // Note the device still holds a reference to the target while it's active
    IDirect3DSurface9* target;
    targetTex->GetSurfaceLevel(0, &target);
    init(target, targetDepthStencil);
    target->Release();
}

void RenderTargetSwitcher::init(IDirect3DSurface9* target, IDirect3DSurface9* targetDepthStencil) {
    DistantLand::device->GetRenderTarget(0, &savedTarget);
    DistantLand::device->GetDepthStencilSurface(&savedDepthStencil);

    DistantLand::device->SetRenderTarget(0, target);
    DistantLand::device->SetDepthStencilSurface(targetDepthStencil);
    g_passBreaks.raw_setRT++;
    g_passBreaks.raw_setDS++;
}

RenderTargetSwitcher::~RenderTargetSwitcher() {
    DistantLand::device->SetRenderTarget(0, savedTarget);
    DistantLand::device->SetDepthStencilSurface(savedDepthStencil);
    g_passBreaks.raw_setRT++;
    g_passBreaks.raw_setDS++;

    if (savedTarget) {
        savedTarget->Release();
    }
    if (savedDepthStencil) {
        savedDepthStencil->Release();
    }
}
