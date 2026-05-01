
#include "configuration.h"
#include "distantland.h"
#include "distantshader.h"
#include "ffeshader.h"
#include "mwbridge.h"
#include "proxydx/d3d8header.h"
#include "imgui_manager.h"
#include "patch_displacement.h"
#include "postshaders.h"
#include "support/log.h"
#include "mge_tracy.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

// Cache for previous frame's world matrices (for velocity buffer)
// Key uses geometry hash + draw params + position + mirror flag to identify unique instances
struct VelocityGeometryKey {
    size_t geometryHash;  // Content-based hash from vertex data
    UINT primCount;       // Distinguishes sub-meshes within same VB (e.g., palm vs fingers)
    UINT startIndex;      // Index buffer offset for this draw
    int posX, posY, posZ;  // Coarse position bucket (64 units)
    int rotBucket;        // Rotation bucket (8 directions, 45° each) to distinguish rotated instances
    bool mirrored;        // True if world matrix has negative determinant (left/right distinction)

    bool operator==(const VelocityGeometryKey& o) const {
        return geometryHash == o.geometryHash &&
               primCount == o.primCount && startIndex == o.startIndex &&
               posX == o.posX && posY == o.posY && posZ == o.posZ &&
               rotBucket == o.rotBucket && mirrored == o.mirrored;
    }
};

struct VelocityGeometryKeyHash {
    size_t operator()(const VelocityGeometryKey& k) const {
        size_t h = k.geometryHash;
        h ^= k.primCount + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= k.startIndex + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= (size_t)k.posX + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= (size_t)k.posY + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= (size_t)k.posZ + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= (size_t)k.rotBucket + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= (size_t)k.mirrored + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static std::unordered_map<VelocityGeometryKey, D3DXMATRIX, VelocityGeometryKeyHash> s_prevWorldCache;
static std::unordered_map<VelocityGeometryKey, D3DXMATRIX, VelocityGeometryKeyHash> s_curWorldCache;

void DistantLand::renderDepth(DLContext* ctx, const std::vector<RecordedMWState>& recMW, int sceneFilter, FixedFunctionShader::FrameBuffer* fb) {
    MGE_ZoneScopedN("renderDepth");
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_Depth, 0, (int)recMW.size());
    auto mwBridge = MWBridge::get();

    // DEBUG: Log that renderDepth is being called
    static bool loggedOnce = false;
    if (!loggedOnce) {
        LOG::logline(">> renderDepth() is being called - depth rendering active");
        loggedOnce = true;
    }

    // RT is set by caller (renderStage1) to avoid redundant switches
    device->Clear(0, 0, D3DCLEAR_ZBUFFER, 0, 1.0, 0);
    g_passBreaks.raw_clear++;

    // Unbind depth sampler
    effect->SetTexture(ehTex3, NULL);

    // Projection should match main rendering
    // For Morrowind geometry, use Morrowind's native near plane to match recording
    D3DXMATRIX mwDepthProj = ctx->mwProj;
    // Morrowind uses near=1.0, but we extend far plane for better depth precision
    if (Configuration.MGEFlags & USE_DISTANT_LAND) {
        editProjectionZ(&mwDepthProj, 1.0f, Configuration.DL.DrawDist * kCellSize);
    }
    effect->SetMatrix(ehProj, &mwDepthProj);

    // Clear floating point buffer to far depth
    effectDepth->BeginPass(PASS_CLEARDEPTH);
    device->SetVertexDeclaration(WaterDecl);
    device->SetStreamSource(0, vbFullFrame, 0, 12);
    device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
    effectDepth->EndPass();

    // Push displacement falloff (matches the c73 constant set in hlsl_replay.cpp
    // for the color path). xy = (R_outer, R_inner); zw = camera world XY derived
    // from the inverse of the current view matrix.
    {
        D3DXMATRIX invView;
        D3DXMatrixInverse(&invView, nullptr, &ctx->mwView);
        D3DXVECTOR4 origin(0.0f, 0.0f, 0.0f, 1.0f);
        D3DXVECTOR4 eyeW;
        D3DXVec4Transform(&eyeW, &origin, &invView);
        D3DXVECTOR4 falloff(2560.0f, 1280.0f, eyeW.x, eyeW.y);
        effect->SetVector(ehDisplacementFalloff, &falloff);
    }

    // Recorded draw calls with Morrowind near plane
    // Pass game view for skinned object transforms (device may have UI view in deferred pipeline)
    effectDepth->BeginPass(PASS_RENDERMWDEPTH);
    renderDepthRecorded(recMW, sceneFilter, &ctx->mwView, fb);
    effectDepth->EndPass();

    // Phase 8.4: emit displaced near-patch terrain depth in a separate pass that
    // mirrors the color VS displacement+falloff so SSAO/DOF see the same Z.
    if (fb && fb->nearPatchCount > 0 && sceneFilter == 0) {
        effectDepth->BeginPass(PASS_RENDERMWDEPTHDISPLACED);
        renderDepthRecordedDisplaced(recMW, fb);
        effectDepth->EndPass();
    }

    // Reset projection matrix
    effect->SetMatrix(ehProj, &ctx->mwProj);
}

void DistantLand::renderDepthDistantLand(DLContext* ctx) {
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_DepthDistant, 0);
    auto mwBridge = MWBridge::get();

    // RT is set by caller (renderStage1) to avoid redundant switches

    // Projection for distant land
    D3DXMATRIX mwDepthProj = ctx->mwProj;
    if (Configuration.MGEFlags & USE_DISTANT_LAND) {
        editProjectionZ(&mwDepthProj, 1.0f, Configuration.DL.DrawDist * kCellSize);
    }
    effect->SetMatrix(ehProj, &mwDepthProj);

    if (isDistantCell()) {
        if (!mwBridge->IsUnderwater(ctx->eyePos.z)) {
            // Distant land
            if (mwBridge->IsExterior()) {
                effectDepth->BeginPass(PASS_RENDERLANDDEPTH);
                renderDistantLandZ();
                effectDepth->EndPass();
            }

            // Distant statics
            effectDepth->BeginPass(PASS_RENDERSTATICSDEPTH);
            device->SetVertexDeclaration(StaticDecl);
            if (Configuration.UseSharedMemory) {
                visDistantShared.Render(device, effectDepth, effect, &ehTex0, &ehHasAlpha, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT);
            } else {
                visDistant.Render(device, effectDepth, effect, &ehTex0, &ehHasAlpha, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT);
            }
            effectDepth->EndPass();
        }

        if (Configuration.MGEFlags & USE_GRASS) {
            // Grass
            effectDepth->BeginPass(PASS_RENDERGRASSDEPTHINST);
            renderGrassInstZ();
            effectDepth->EndPass();
        }
    }

    // Reset projection matrix
    effect->SetMatrix(ehProj, &ctx->mwProj);
}

void DistantLand::renderDepthAdditional(DLContext* ctx, const std::vector<RecordedMWState>& recMW, int sceneFilter, const D3DXMATRIX* viewOverride,
    IDirect3DSurface9* targetOverride, IDirect3DSurface9* depthStencilOverride, bool clearZ) {
    // Switch to render target
    RenderTargetSwitcher rtsw(targetOverride ? targetOverride : surfDepthFrameMSAA,
        depthStencilOverride ? depthStencilOverride : surfDepthDepth);

    // Set RT1 for velocity buffer (MRT with depth) - only if not using custom target
    // Skip on first frame after menu exit to preserve motion blur from before menu
    bool useVelocityMRT = velocityBufferEnabled && surfVelocityMSAA && !targetOverride && !shouldSkipVelocityBuffer();
    if (useVelocityMRT) {
        device->SetRenderTarget(1, surfVelocityMSAA);
    }

    if (clearZ) {
        device->Clear(0, 0, D3DCLEAR_ZBUFFER, 0, 1.0f, 0);
        g_passBreaks.raw_clear++;
    }

    // Unbind depth sampler
    effect->SetTexture(ehTex3, NULL);

    // Projection should match main rendering
    // For Morrowind geometry, use Morrowind's native near plane to match recording
    D3DXMATRIX mwDepthProj = ctx->mwProj;
    if (Configuration.MGEFlags & USE_DISTANT_LAND) {
        editProjectionZ(&mwDepthProj, 1.0f, Configuration.DL.DrawDist * kCellSize);
    }
    effect->SetMatrix(ehProj, &mwDepthProj);

    // Recorded draw calls with Morrowind near plane
    // Pass game view for skinned object transforms (device may have UI view in deferred pipeline)
    // Use viewOverride if provided (for hands which use a different view matrix)
    const D3DXMATRIX* viewToUse = viewOverride ? viewOverride : &ctx->mwView;
    effectDepth->BeginPass(PASS_RENDERMWDEPTH);
    renderDepthRecorded(recMW, sceneFilter, viewToUse);
    effectDepth->EndPass();

    // Clear RT1 after rendering
    if (useVelocityMRT) {
        device->SetRenderTarget(1, nullptr);
    }

    // Reset projection matrix
    effect->SetMatrix(ehProj, &ctx->mwProj);
}

void DistantLand::renderForwardPrepassChain(DLContext* ctx, const PostProcessData* ppd) {
    forwardSSAOActive = false;
    static const DWORD fvfForwardPrepass = D3DFVF_XYZRHW | D3DFVF_TEX1;

    const int prepassEnvFlags = ppd ? ppd->envFlags : (ctx ? ctx->postEnvFlags : 0);

    if (!ctx || !isHLSLActive() || !PostShaders::usesForwardSSAO(prepassEnvFlags)) {
        return;
    }
    if (!texDepthFrame || !texForwardSSAORaw || !texForwardSSAO ||
        !surfForwardSSAORaw || !surfForwardSSAO ||
        !texForwardSSAONoise || !psForwardSSAO || !psForwardSSAOBlur || !vbForwardPrepass) {
        return;
    }

    DistantLand::logWaterDiagnostics("PREPASS", ctx, ppd);

    MGE_ZoneScopedN("renderForwardPrepassChain");

    IDirect3DStateBlock9* stateSaved = nullptr;
    if (FAILED(device->CreateStateBlock(D3DSBT_ALL, &stateSaved)) || !stateSaved) {
        LOG::logline("!! Forward prepass: failed to capture state block");
        return;
    }

    IDirect3DSurface9* savedRT0 = nullptr;
    IDirect3DSurface9* savedDepthStencil = nullptr;
    D3DVIEWPORT9 savedViewport{};
    device->GetRenderTarget(0, &savedRT0);
    device->GetDepthStencilSurface(&savedDepthStencil);
    device->GetViewport(&savedViewport);

    const float zoom = (Configuration.MGEFlags & ZOOM_ASPECT) ? Configuration.CameraEffects.zoom : 1.0f;
    const float fov = Configuration.ScreenFOV / zoom;
    const float viewportWidth = std::max(static_cast<float>(savedViewport.Width), 1.0f);
    const float viewportHeight = std::max(static_cast<float>(savedViewport.Height), 1.0f);
    float ssaoParams[4] = {
        1.0f / viewportWidth,
        1.0f / viewportHeight,
        fov,
        0.0f
    };

    auto setupFullscreenPass = [&]() {
        device->SetViewport(&savedViewport);
        device->SetVertexShader(nullptr);
        device->SetFVF(fvfForwardPrepass);
        device->SetStreamSource(0, vbForwardPrepass, 0, 24);
        device->SetRenderState(D3DRS_ZENABLE, FALSE);
        device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
        device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
        device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
    };

    {
        RenderTargetSwitcher rt(surfForwardSSAO, nullptr);
        setupFullscreenPass();
        device->SetPixelShader(psForwardSSAO);
        device->SetPixelShaderConstantF(0, ssaoParams, 1);
        device->SetTexture(0, texDepthFrame);
        device->SetTexture(1, texForwardSSAONoise);
        device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
        device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
        device->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
        device->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
        device->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
        device->SetSamplerState(1, D3DSAMP_MINFILTER, D3DTEXF_POINT);
        device->SetSamplerState(1, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
        device->SetSamplerState(1, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
        device->SetSamplerState(1, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
        device->SetSamplerState(1, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
        device->SetTexture(0, nullptr);
        device->SetTexture(1, nullptr);
    }

    {
        RenderTargetSwitcher rt(surfForwardSSAORaw, nullptr);
        setupFullscreenPass();
        device->SetPixelShader(psForwardSSAOBlur);
        device->SetPixelShaderConstantF(0, ssaoParams, 1);
        device->SetTexture(0, texForwardSSAO);
        device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
        device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
        device->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
        device->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_MIRROR);
        device->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_MIRROR);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
        device->SetTexture(0, nullptr);
    }

    {
        RenderTargetSwitcher rt(surfForwardSSAO, nullptr);
        setupFullscreenPass();
        device->SetPixelShader(psForwardSSAOBlur);
        device->SetPixelShaderConstantF(0, ssaoParams, 1);
        device->SetTexture(0, texForwardSSAORaw);
        device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
        device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
        device->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
        device->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_MIRROR);
        device->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_MIRROR);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
        device->SetTexture(0, nullptr);
    }

    if (stateSaved) {
        stateSaved->Apply();
        stateSaved->Release();
    }

    device->SetRenderTarget(0, savedRT0);
    device->SetDepthStencilSurface(savedDepthStencil);
    device->SetViewport(&savedViewport);

    if (savedRT0) {
        savedRT0->Release();
    }
    if (savedDepthStencil) {
        savedDepthStencil->Release();
    }

    FixedFunctionShader::resetHLSLCaches();
    forwardSSAOActive = true;
}

void DistantLand::renderDepthRecorded(const std::vector<RecordedMWState>& recMW, int sceneFilter, const D3DXMATRIX* gameView, FixedFunctionShader::FrameBuffer* fb) {
    // Use an alpha threshold for solidity that isn't precisely equal to a commonly used value (such as 0.5).
    // Vertex interpolators can be slightly inaccurate and cause a value that should be constant across a triangle
    // to have interpolated fragment values that vary either side of the threshold and cause noise.
    const float solidThreshold = 0.499f;

    // Swap world matrix caches for velocity buffer (once per frame on Scene 0)
    // Skip entirely when velocity buffer writes are disabled (menu exit frame)
    static int lastFrameSwapped = -1;
    if (sceneFilter == 0 && velocityBufferEnabled && !shouldSkipVelocityBuffer()) {
        int curFrame = FixedFunctionShader::getRenderingBuffer().frameNumber;
        if (curFrame != lastFrameSwapped) {
            // If more than 1 frame gap, prev cache is stale - clear both to avoid velocity spikes
            bool frameGap = (lastFrameSwapped >= 0 && curFrame - lastFrameSwapped > 1);
            if (frameGap) {
                s_prevWorldCache.clear();
                s_curWorldCache.clear();
            } else {
                s_prevWorldCache = std::move(s_curWorldCache);
                s_curWorldCache.clear();
            }
            lastFrameSwapped = curFrame;
        }
    }

    // DEBUG: Log renderDepthRecorded call and scene filter breakdown
    static int logCount = 0;
    if (logCount < 4) {
        int s0 = 0, s1 = 0, s2 = 0;
        for (const auto& m : recMW) {
            if (m.sceneNum == 0) s0++;
            else if (m.sceneNum == 1) s1++;
            else s2++;
        }
        logCount++;
    }

    // Note: Culling is already done by executeHiZCulling/applyVisibilityAndFilterRecordMW() using Hi-Z.
    // recordMW is pre-filtered - only visible objects remain.

    // Recorded renders (pre-filtered by executeHiZCulling/applyVisibilityAndFilterRecordMW)
    int scene2Count = 0;
    const bool allowDisplacedDepth = fb && sceneFilter == 0 && fb->nearPatchCount > 0;
    bool edgeHeightsReady = false;

    // Phase 1 skip set: near-set displaced terrain VB/IBs that have a cached
    // SubdivPatch and will be rendered by Phase 2. We index by VB/IB pointer
    // because applyVisibilityAndFilterRecordMW renumbers recMW after filtering,
    // so call.recordMWIndex no longer matches our recIdx — matching by buffer
    // pointer is the only reliable correlation.
    std::vector<FixedFunctionShader::TerrainPatchKey> skipPhase1;
    if (allowDisplacedDepth) {
        skipPhase1.reserve(fb->nearPatchCount);
        for (const auto& call : fb->recordedCalls) {
            if (!call.sk.hasDisplacement || call.bin != RenderBin::Terrain) continue;
            FixedFunctionShader::TerrainPatchKey pk{call.rs.vb, call.rs.ib};
            bool inSet = false;
            for (uint32_t s = 0; s < fb->nearPatchCount; ++s) {
                if (fb->nearPatches[s] == pk) { inSet = true; break; }
            }
            if (!inSet) continue;
            PatchDisplacement::SubdivPatch* sp = PatchDisplacement::findCached(
                pk, call, call.overlayTexture,
                ImGuiManager::GetDisplacementScale(),
                call.subdivTier,
                ImGuiManager::GetDisplacementGamma(),
                ImGuiManager::GetDisplacementPivot());
            if (sp) skipPhase1.push_back(pk);
        }
    }
    for (size_t recIdx = 0; recIdx < recMW.size(); ++recIdx) {
        const auto& i = recMW[recIdx];
        // Scene filter: -1 = all, 0 = Scene 0 only, N>0 = Scene N and later.
        if (sceneFilter == 0 && i.sceneNum != 0) continue;
        if (sceneFilter > 0 && i.sceneNum < sceneFilter) continue;

        // Log Scene 2 depth draws
        if (i.sceneNum == 2 && scene2Count < 5 && LOG::catEnabled(LOG::Cat_Recording)) {
            LOG::logline("[DEPTH-S2] scene=%d vbs=%d prims=%d vb=%p ib=%p gameView41=%.2f wvt41=%.2f world41=%.2f",
                i.sceneNum, i.vertexBlendState, i.primCount, i.vb, i.ib,
                gameView ? gameView->_41 : -999.0f, i.worldViewTransforms[0]._41, i.worldTransforms[0]._41);
            scene2Count++;
        }

        MGE_ZoneScopedN("renderDepth_Draw");
        // Set variables in main effect; variables are shared via effect pool

        // Fragment colour routing
        bool alphaDependent = i.alphaTest || i.blendEnable;
        effect->SetBool(ehHasVCol, alphaDependent && (i.fvf & D3DFVF_DIFFUSE) != 0);
        effect->SetFloat(ehMaterialAlpha, alphaDependent ? i.diffuseMaterial.a : 1.0f);

        // Only bind texture for alphas
        if (alphaDependent && i.texture) {
            effect->SetTexture(ehTex0, i.texture);
            effect->SetBool(ehHasAlpha, true);
            effect->SetFloat(ehAlphaRef, i.alphaTest ? (i.alphaRef / 255.0f) : solidThreshold);
        } else {
            effect->SetTexture(ehTex0, 0);
            effect->SetBool(ehHasAlpha, false);
            effect->SetFloat(ehAlphaRef, -1.0f);
        }

        // Skin using worldview matrices for numerical accuracy
        effect->SetBool(ehHasBones, i.vertexBlendState != 0);
        effect->SetInt(ehVertexBlendState, i.vertexBlendState);

        // Recalculate skinned transforms with game view matrix (not device view, which may be UI)
        D3DXMATRIX currentWorldViewForCache = i.worldViewTransforms[0];
        if (i.vertexBlendState > 0 && gameView) {
            D3DXMATRIX currentWorldViewTransforms[4];
            for (int j = 0; j < 4; j++) {
                currentWorldViewTransforms[j] = i.worldTransforms[j] * (*gameView);
            }
            effect->SetMatrixArray(ehVertexBlendPalette, currentWorldViewTransforms, 4);
            currentWorldViewForCache = currentWorldViewTransforms[0];
        } else if (i.vertexBlendState > 0) {
            // Fallback: use staging view (device may not have current transforms with state suppression)
            D3DXMATRIX currentView, currentWorldViewTransforms[4];
            currentView = DistantLand::s_staging.mwView;
            for (int j = 0; j < 4; j++) {
                currentWorldViewTransforms[j] = i.worldTransforms[j] * currentView;
            }
            effect->SetMatrixArray(ehVertexBlendPalette, currentWorldViewTransforms, 4);
            currentWorldViewForCache = currentWorldViewTransforms[0];
        } else {
            effect->SetMatrixArray(ehVertexBlendPalette, i.worldViewTransforms, 4);
        }

        // Velocity buffer: cache WORLD matrices (not worldview) so camera motion cancels out.
        // prevWorldView = prev_world * current_view, so only object motion produces velocity.
        // Key uses geometry hash + coarse position + mirror flag (for left/right body parts).
        if (velocityBufferEnabled && gameView) {
            const int kPosBucketSize = 8;  // ~6cm buckets to separate nearby instances
            size_t geoHash = computeGeometryHash(i.vb, i.vbOffset, i.vbStride, i.vertCount, i.fvf);

            // Compute 3x3 determinant to detect mirrored meshes (negative scale)
            const D3DXMATRIX& w = i.worldTransforms[0];
            float det3 = w._11 * (w._22 * w._33 - w._23 * w._32)
                       - w._12 * (w._21 * w._33 - w._23 * w._31)
                       + w._13 * (w._21 * w._32 - w._22 * w._31);
            bool mirrored = det3 < 0;

            // Quantize rotation to 8 buckets (45° each) based on local X axis direction in world XY
            // This distinguishes objects at same position but rotated around Z (vertical axis)
            // Row 1 of world matrix (_11, _12) = where local X axis points in world XY
            int rotBucket = 0;
            float rightX = w._11, rightY = w._12;  // Local X axis in world XY (how mesh "faces")
            if (rightX != 0 || rightY != 0) {
                float angle = atan2f(rightY, rightX);  // -PI to PI
                rotBucket = (int)((angle + 3.14159265f) / (6.28318530f / 8.0f));  // 0-8
                rotBucket = rotBucket & 7;  // Wrap to 0-7
            }

            VelocityGeometryKey vkey{
                geoHash,
                i.primCount,
                i.startIndex,
                (int)(w._41 / kPosBucketSize),
                (int)(w._42 / kPosBucketSize),
                (int)(w._43 / kPosBucketSize),
                rotBucket,
                mirrored
            };
            D3DXMATRIX prevWorldView[4];

            auto it = s_prevWorldCache.find(vkey);
            if (it != s_prevWorldCache.end()) {
                // Sanity check: reject if prev world position is wildly different (collision)
                float dx = it->second._41 - i.worldTransforms[0]._41;
                float dy = it->second._42 - i.worldTransforms[0]._42;
                float dz = it->second._43 - i.worldTransforms[0]._43;
                float distSq = dx*dx + dy*dy + dz*dz;

                // Max 300 units movement per frame (generous for fast objects)
                if (distSq < 300.0f * 300.0f) {
                    // Found valid previous world matrix - multiply by CURRENT view
                    D3DXMatrixMultiply(&prevWorldView[0], &it->second, gameView);
                } else {
                    // Cache collision - use current (no velocity)
                    prevWorldView[0] = currentWorldViewForCache;
                }
            } else {
                // No prev, use current worldview (no velocity on first frame)
                prevWorldView[0] = currentWorldViewForCache;
            }
            for (int j = 1; j < 4; j++) {
                prevWorldView[j] = prevWorldView[0];  // Copy for skinned (simplified)
            }
            effect->SetMatrixArray(ehPrevVertexBlendPalette, prevWorldView, 4);

            // Store WORLD matrix (not worldview) for next frame
            s_curWorldCache[vkey] = i.worldTransforms[0];

            // Debug: log first few velocity lookups
            static int velLogCount = 0;
            if (velLogCount < 5 && it != s_prevWorldCache.end()) {
                float curW41 = currentWorldViewForCache._41;
                float prevW41 = prevWorldView[0]._41;
                float diff = curW41 - prevW41;
                if (std::abs(diff) > 0.01f) {
                    LOG::logline(">> Velocity: vb=%p diff=%.3f (cur=%.2f prev=%.2f) - object moving",
                        i.vb, diff, curW41, prevW41);
                    velLogCount++;
                }
            }
        } else if (velocityBufferEnabled) {
            // No gameView available - use identity for prev (will show some velocity)
            LOG::logline("!! Velocity: gameView is null for vb=%p", i.vb);
            effect->SetMatrixArray(ehPrevVertexBlendPalette, i.worldViewTransforms, 4);
        }

        // Always set prevVertexBlendPalette to avoid undefined shader behavior
        // (shader always reads it, even if velocity buffer is disabled)
        if (!velocityBufferEnabled) {
            effect->SetMatrixArray(ehPrevVertexBlendPalette, i.worldViewTransforms, 4);
        }

        effectDepth->CommitChanges();

        // Scene 2 objects (hands): force opaque depth write
        // They have zWrite=0 and blendEnable=1 in Morrowind, but we need solid depth for SSAO/DOF
        // Scene 1 (particles) typically skipped via sceneFilter, but if present also force depth
        bool forceOpaqueDepth = (i.sceneNum > 0);

        // Set render states for depth pass
        if (forceOpaqueDepth) {
            // Scene 1/2: write solid depth, no blending, standard culling
            device->SetRenderState(D3DRS_CULLMODE, i.cullMode);
            device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
            device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
        } else {
            // Scene 0: match HLSL rendering states
            if (i.blendEnable) {
                device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
            } else {
                device->SetRenderState(D3DRS_CULLMODE, i.cullMode);
            }
            device->SetRenderState(D3DRS_ALPHABLENDENABLE, i.blendEnable);
            if (i.blendEnable) {
                device->SetRenderState(D3DRS_SRCBLEND, i.srcBlend);
                device->SetRenderState(D3DRS_DESTBLEND, i.destBlend);
            }

            // Alpha test for depth-only rendering to match color rendering
            device->SetRenderState(D3DRS_ALPHATESTENABLE, i.alphaTest);
            if (i.alphaTest) {
                device->SetRenderState(D3DRS_ALPHAREF, i.alphaRef);
                device->SetRenderState(D3DRS_ALPHAFUNC, i.alphaFunc);
            }
        }

        // Skip if this draw's VB/IB is in the Phase-2 displaced set (cached and
        // ready). If it's a near-set displaced tile but the cache miss left it
        // out of skipPhase1, we fall through and render the original flat VB
        // here so depth stays continuous; Phase 2 will pick it up next frame.
        if (allowDisplacedDepth && i.sceneNum == 0 && !skipPhase1.empty()) {
            bool deferToDisplacedPhase = false;
            for (const auto& sk : skipPhase1) {
                if (sk.vb == i.vb && sk.ib == i.ib) { deferToDisplacedPhase = true; break; }
            }
            if (deferToDisplacedPhase) continue;
        }

        device->SetStreamSource(0, i.vb, i.vbOffset, i.vbStride);
        device->SetIndices(i.ib);
        device->SetFVF(i.fvf);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);
    }
    // Note: Culling stats are now logged in executeHiZCulling/applyVisibilityAndFilterRecordMW()
}

void DistantLand::renderDepthRecordedDisplaced(const std::vector<RecordedMWState>& recMW, FixedFunctionShader::FrameBuffer* fb) {
    if (!fb || fb->nearPatchCount == 0) return;

    // Lookup-only: never call coalesceEdgeHeights or getOrBuild from depth.
    // The color phase is the sole owner of cache builds and edge-map updates.
    // This eliminates the race where depth-phase build would bake a partial
    // edge map into a SubdivPatch that color-phase getOrBuild then cache-hit
    // on, propagating stale edges into the color render (visible as cracks).
    //
    // Iterate fb->recordedCalls directly (not the filtered recMW). After
    // applyVisibilityAndFilterRecordMW renumbers recMW, call.recordMWIndex no
    // longer maps to recIdx, so cross-referencing is impossible. Each call
    // already carries everything needed (rs.worldTransforms, cullMode, etc.).

    device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);

    for (const auto& call : fb->recordedCalls) {
        if (!call.sk.hasDisplacement) continue;
        if (call.bin != RenderBin::Terrain) continue;

        FixedFunctionShader::TerrainPatchKey pk{call.rs.vb, call.rs.ib};
        bool inSet = false;
        for (uint32_t s = 0; s < fb->nearPatchCount; ++s) {
            if (fb->nearPatches[s] == pk) { inSet = true; break; }
        }
        if (!inSet) continue;

        PatchDisplacement::SubdivPatch* sp = PatchDisplacement::findCached(
            pk, call, call.overlayTexture,
            ImGuiManager::GetDisplacementScale(),
            call.subdivTier,
            ImGuiManager::GetDisplacementGamma(),
            ImGuiManager::GetDisplacementPivot());
        if (!sp || !sp->vb) continue;

        // Recompute worldview from the current frame's view (call.rs.worldView
        // was captured at recording time, which can be the previous frame in
        // N-1 mode). Mirrors the color path's recompute in hlsl_replay.cpp.
        D3DXMATRIX worldView;
        D3DXMatrixMultiply(&worldView, &call.rs.worldTransforms[0], &fb->currentView);
        D3DXMATRIX wvPalette[4] = { worldView, worldView, worldView, worldView };

        effect->SetMatrix(ehWorld, &call.rs.worldTransforms[0]);
        effect->SetMatrixArray(ehVertexBlendPalette, wvPalette, 4);
        effect->SetBool(ehHasAlpha, false);
        effect->SetFloat(ehAlphaRef, -1.0f);
        effectDepth->CommitChanges();

        device->SetRenderState(D3DRS_CULLMODE, call.rs.cullMode);

        device->SetStreamSource(0, sp->vb, 0, sp->stride);
        device->SetIndices(sp->ib);
        device->SetFVF(sp->fvf);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, sp->vertCount, 0, sp->primCount);
    }
}

// GPU-only Hi-Z mip generation (non-blocking, ~0.5ms)
// Called at end of Present() - generates mips on GPU, returns immediately
void DistantLand::generateHiZMipsGPU() {
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_HiZGen, 0, hiZValidMips);
    MGE_ZoneScopedN("HiZ_GenerateMipsGPU");

    static int callCount = 0;
    if (callCount < 5) {
        LOG::logline(">> Hi-Z: generateHiZMipsGPU() called (frame %d)", callCount);
        callCount++;
    }

    if (!texHiZ || !texHiZPrev || !effectHiZ) {
        LOG::logline("Hi-Z: Skipping GPU generation - texHiZ=%p texHiZPrev=%p effectHiZ=%p", texHiZ, texHiZPrev, effectHiZ);
        return;
    }

    // Save current render targets and states that we'll modify
    IDirect3DSurface9* savedRT0;
    IDirect3DSurface9* savedDepthStencil;
    D3DVIEWPORT9 savedViewport;
    IDirect3DVertexShader9* savedVS = nullptr;
    IDirect3DPixelShader9* savedPS = nullptr;
    IDirect3DVertexDeclaration9* savedDecl = nullptr;
    IDirect3DVertexBuffer9* savedVB = nullptr;
    UINT savedVBStride, savedVBOffset;

    // Save render states we'll modify
    DWORD savedZEnable, savedZWriteEnable, savedCullMode, savedAlphaBlendEnable;

    {
        device->GetRenderTarget(0, &savedRT0);
        device->GetDepthStencilSurface(&savedDepthStencil);
        device->GetViewport(&savedViewport);
        device->GetVertexShader(&savedVS);
        device->GetPixelShader(&savedPS);
        device->GetVertexDeclaration(&savedDecl);
        device->GetStreamSource(0, &savedVB, &savedVBOffset, &savedVBStride);

        device->GetRenderState(D3DRS_ZENABLE, &savedZEnable);
        device->GetRenderState(D3DRS_ZWRITEENABLE, &savedZWriteEnable);
        device->GetRenderState(D3DRS_CULLMODE, &savedCullMode);
        device->GetRenderState(D3DRS_ALPHABLENDENABLE, &savedAlphaBlendEnable);
    }

    // Use cached shaders (compiled once during initialization, not per-frame!)
    IDirect3DSurface9* dstSurf;

    if (!vsHiZ || !psHiZ) {
        LOG::logline("!! Hi-Z: Shaders not initialized (vsHiZ=%p psHiZ=%p)", vsHiZ, psHiZ);
        device->SetRenderTarget(0, savedRT0);
        device->SetDepthStencilSurface(savedDepthStencil);
        savedRT0->Release();
        if (savedDepthStencil) savedDepthStencil->Release();
        return;
    }

    // Set cached shaders
    device->SetVertexShader(vsHiZ);
    device->SetPixelShader(psHiZ);

    // Generate all mip levels (0-N) using MAX downsampling shader
    // Mip 0: Downsample from texCullDepth (320x240) to texHiZ mip 0 (160x120) using MAX
    // Mips 1-N: Ping-pong downsample between texHiZ and texHiZPrev
    {
        int actualMipsGenerated = 0;
        for (int mipLevel = 0; mipLevel < hiZValidMips; mipLevel++) {
            // Determine which texture to render to and which to sample from (ping-pong)
            // Even mips (0, 2, 4...): render to texHiZ, sample from texHiZPrev
            // Odd mips  (1, 3, 5...): render to texHiZPrev, sample from texHiZ
            IDirect3DTexture9* dstTexture = (mipLevel % 2 == 0) ? texHiZ : texHiZPrev;
            IDirect3DTexture9* srcTexture;

            // Get dimensions of destination mip level
            D3DSURFACE_DESC dstDesc;
            dstTexture->GetLevelDesc(mipLevel, &dstDesc);

            // Double-check: Stop at 8x8 minimum - don't generate smaller mips
            if (dstDesc.Width < 8 || dstDesc.Height < 8) {
                break;
            }
            actualMipsGenerated++;

            // Set render target to current mip level of destination texture
            dstTexture->GetSurfaceLevel(mipLevel, &dstSurf);
            device->SetRenderTarget(0, dstSurf);
            device->SetDepthStencilSurface(NULL);

            // Set viewport to match the current mip size
            D3DVIEWPORT9 vp;
            vp.X = 0;
            vp.Y = 0;
            vp.Width = dstDesc.Width;
            vp.Height = dstDesc.Height;
            vp.MinZ = 0.0f;
            vp.MaxZ = 1.0f;
            device->SetViewport(&vp);

            // Get SOURCE dimensions and mip level (texture we're sampling from)
            D3DSURFACE_DESC srcDesc;
            int sourceMipLevel;

            if (mipLevel == 0) {
                // Mip 0: Sample from texCullDepth (Morrowind geometry only, copied from surfDepthFrameMSAA)
                // texCullDepth is filled by StretchRect in distantland.cpp:206 after renderDepth()
                texCullDepth->GetLevelDesc(0, &srcDesc);
                sourceMipLevel = 0;
                srcTexture = texCullDepth;
            } else {
                // Mip N: Sample from OPPOSITE texture's mip N-1 (ping-pong to avoid read/write conflict!)
                srcTexture = (mipLevel % 2 == 0) ? texHiZPrev : texHiZ;
                srcTexture->GetLevelDesc(mipLevel - 1, &srcDesc);
                sourceMipLevel = mipLevel - 1;
            }

            // Bind source texture (will be different from destination!)
            device->SetTexture(0, srcTexture);

            // Configure sampler with POINT filtering (matches HizMipmap9.fx)
            // CRITICAL: MIPFILTER must be POINT (not NONE) for tex2Dlod to access mip levels!
            device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
            device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
            device->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT);  // Was NONE - broke tex2Dlod!
            device->SetSamplerState(0, D3DSAMP_MAXMIPLEVEL, 0);  // Allow all mip levels
            device->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
            device->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);

            // Set SOURCE texture info as pixel shader constant (c0)
            // xy = source dimensions, z = source mip level, w = destination mip level
            // CRITICAL: tex2Dlod needs the SOURCE mip level, not destination!
            float sourceInfo[4] = { (float)srcDesc.Width, (float)srcDesc.Height, (float)sourceMipLevel, (float)mipLevel };
            device->SetPixelShaderConstantF(0, sourceInfo, 1);

            // Set render states
            device->SetRenderState(D3DRS_ZENABLE, FALSE);
            device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
            device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
            device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);

            // Render full-screen quad with MAX downsample shader
            device->SetVertexDeclaration(WaterDecl);
            device->SetStreamSource(0, vbFullFrame, 0, 12);
            device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);

            // CRITICAL: Unbind source texture BEFORE next iteration (matches HZBManagerD3D9.cpp:213)
            // D3D9 requires this so the texture can be sampled in the next ping-pong step
            // Do NOT unbind render target - let next iteration SetRenderTarget handle it
            device->SetTexture(0, nullptr);

            dstSurf->Release();
        }

        // Even mips (0,2,4...) are in texHiZ, odd mips (1,3,5...) are in texHiZPrev
    }

    // Reset sampler states we modified
    device->SetSamplerState(0, D3DSAMP_MAXMIPLEVEL, 0);
    device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
    device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
    device->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
    device->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
    device->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);

    // Restore all modified state
    {
        device->SetRenderTarget(0, savedRT0);
        device->SetDepthStencilSurface(savedDepthStencil);
        device->SetViewport(&savedViewport);
        device->SetVertexShader(savedVS);
        device->SetPixelShader(savedPS);
        device->SetVertexDeclaration(savedDecl);
        device->SetStreamSource(0, savedVB, savedVBOffset, savedVBStride);
        device->SetTexture(0, nullptr);  // Clear texture binding

        // Restore render states
        device->SetRenderState(D3DRS_ZENABLE, savedZEnable);
        device->SetRenderState(D3DRS_ZWRITEENABLE, savedZWriteEnable);
        device->SetRenderState(D3DRS_CULLMODE, savedCullMode);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, savedAlphaBlendEnable);

        // Release saved references
        savedRT0->Release();
        if (savedDepthStencil) savedDepthStencil->Release();
        if (savedVS) savedVS->Release();
        if (savedPS) savedPS->Release();
        if (savedDecl) savedDecl->Release();
        if (savedVB) savedVB->Release();
    }
}

// CPU-GPU sync copy from render target to staging texture (blocking, ~1.5ms)
// Called at beginning of Clear/BeginScene() - copies GPU-generated mips to CPU-readable staging
void DistantLand::copyHiZToStaging() {
    {
        // Unlock staging texture from 2 frames ago (all locked mips)
        if (hiZLockedMips > 0 && texHiZStagingPrev) {
            for (int mip = 0; mip < hiZLockedMips; mip++) {
                texHiZStagingPrev->UnlockRect(mip);
            }
            hiZLockedMips = 0;
        }
    }

    if (!texHiZ || !texHiZPrev) {
        return;
    }

    // Step 3: Copy valid Hi-Z mips to staging texture for CPU readback (only mips down to 8x8)
    // IMPORTANT: Check if previous GetRenderTargetData is complete BEFORE issuing a new one
    // This allows us to skip the copy if GPU is still busy, avoiding 6ms stalls
    bool canCopyThisFrame = false;

    {
        if (queryHiZCopy) {
            // Non-blocking query check: Is the previous GetRenderTargetData complete?
            HRESULT hr = queryHiZCopy->GetData(NULL, 0, D3DGETDATA_FLUSH);
            if (hr == S_OK) {
                // Previous copy finished - safe to issue new one
                canCopyThisFrame = true;
            } else {
                // GPU still busy with previous copy - skip this frame, use old Hi-Z data
                // This is the KEY to non-blocking behavior!
                canCopyThisFrame = false;
            }
        } else {
            // No query available - first frame, just try the copy
            canCopyThisFrame = true;
        }
    }

    if (!canCopyThisFrame) {
        // Skip GetRenderTargetData this frame - GPU busy, use previous Hi-Z data
        // This prevents the 6ms stall and maintains frame rate!
        // Don't rotate buffers or issue query - keep using current pyramid
        return;
    }

    // Only copy if GPU is ready (non-blocking async behavior)
    {
        for (int mipLevel = 0; mipLevel < hiZValidMips; mipLevel++) {
            IDirect3DSurface9* srcSurf = nullptr;
            IDirect3DSurface9* dstSurf = nullptr;

            texHiZ->GetSurfaceLevel(mipLevel, &srcSurf);
            texHiZStaging->GetSurfaceLevel(mipLevel, &dstSurf);
            HRESULT hr = device->GetRenderTargetData(srcSurf, dstSurf);
            srcSurf->Release();
            dstSurf->Release();

            if (FAILED(hr)) {
                static bool loggedOnce = false;
                if (!loggedOnce) {
                    LOG::logline("!! Hi-Z: GetRenderTargetData failed for mip %d", mipLevel);
                    loggedOnce = true;
                }
                return;
            }
        }
    }

    // Debug: Sample a few depth values from mip 0 to verify Hi-Z content
    static bool loggedOnce = false;
    if (!loggedOnce && texHiZStaging) {
        D3DLOCKED_RECT lr;
        if (SUCCEEDED(texHiZStaging->LockRect(0, &lr, NULL, D3DLOCK_READONLY))) {
            D3DSURFACE_DESC desc;
            texHiZStaging->GetLevelDesc(0, &desc);
            float* data = (float*)lr.pBits;
            int stride = lr.Pitch / sizeof(float);

            // Sample center and corners
            float center = data[(desc.Height/2) * stride + (desc.Width/2)];
            float tl = data[0];
            float tr = data[desc.Width-1];
            float bl = data[(desc.Height-1) * stride];
            float br = data[(desc.Height-1) * stride + (desc.Width-1)];

            // Removed debug mip0 sampling logging
            texHiZStaging->UnlockRect(0);
            loggedOnce = true;
        }
    }

    {
        // Swap Hi-Z buffers: current becomes previous for next frame's culling
        std::swap(texHiZ, texHiZPrev);

        // Triple-buffer rotation: N → N-1 → N-2 → N (gives 2 frames = ~40ms for GetRenderTargetData to complete)
        IDirect3DTexture9* temp = texHiZStagingPrev;  // Save N-2 (oldest, about to be freed for reuse)
        texHiZStagingPrev = texHiZStaging2;           // N-1 becomes N-2 (safe to lock now)
        texHiZStaging2 = texHiZStaging;               // N becomes N-1 (still copying)
        texHiZStaging = temp;                         // N-2 becomes N (ready for fresh GetRenderTargetData)

        // Rotate queries along with buffers
        IDirect3DQuery9* tempQuery = queryHiZCopyPrev;
        queryHiZCopyPrev = queryHiZCopy2;
        queryHiZCopy2 = queryHiZCopy;
        queryHiZCopy = tempQuery;

        // Issue D3D9 Event Query AFTER rotation, so queryHiZCopy tracks the buffer we just copied to
        if (queryHiZCopy) {
            queryHiZCopy->Issue(D3DISSUE_END);
        }
    }

    {
        // Pre-lock FIRST HALF of mip levels from 2-frame-old staging (query-synced, minimal stall)
        // Wait for GPU copy to complete before locking (check query for N-2 frame)
        static int checkCount = 0;

        // Skip first 2 checks - triple buffer needs 2 frames to prime
        if (checkCount < 2) {
            checkCount++;
            return;
        }

        if (queryHiZCopyPrev) {
            // Poll query until GPU signals completion (should complete immediately after 2 frames)
            // Use minimal attempts since 2 frames is plenty of time for GetRenderTargetData
            HRESULT hr = queryHiZCopyPrev->GetData(NULL, 0, 0);

            if (hr != S_OK) {
                // Query not ready - very rare, skip locking this frame
                static int timeoutCount = 0;
                if (timeoutCount < 3) {
                    LOG::logline("!! Hi-Z: Query not ready after 2 frames (unusual) - skipping pre-lock");
                    timeoutCount++;
                }
                return;
            }
        }

        int mipsToLockNow = (hiZValidMips + 1) / 2; // Lock lower mips (0 to mid-1), most commonly used
        for (int mip = 0; mip < mipsToLockNow; mip++) {
            HRESULT hr = texHiZStagingPrev->LockRect(mip, &hiZLockedRects[mip], NULL, D3DLOCK_READONLY);
            if (FAILED(hr)) {
                LOG::logline("!! Hi-Z: Failed to pre-lock mip %d (2 frames old)", mip);
                // Unlock any that succeeded
                for (int j = 0; j < mip; j++) {
                    texHiZStagingPrev->UnlockRect(j);
                }
                hiZLockedMips = 0;
                return;
            }
        }
        hiZLockedMips = mipsToLockNow;
    }
}

void DistantLand::lockRemainingHiZMips() {

    if (!texHiZStagingPrev || hiZLockedMips >= hiZValidMips) {
        return; // Already all valid mips locked or no texture
    }

    // Lock SECOND HALF of mip levels (remaining valid mips from where we left off)
    int startMip = hiZLockedMips;
    for (int mip = startMip; mip < hiZValidMips; mip++) {
        HRESULT hr = texHiZStagingPrev->LockRect(mip, &hiZLockedRects[mip], NULL, D3DLOCK_READONLY);
        if (FAILED(hr)) {
            LOG::logline("!! Hi-Z: Failed to lock remaining mip %d", mip);
            // Unlock the ones we just tried to lock (keep the first half locked)
            for (int j = startMip; j < mip; j++) {
                texHiZStagingPrev->UnlockRect(j);
            }
            return;
        }
    }
    hiZLockedMips = hiZValidMips; // Now all valid mips are locked
}

// Save CPU Hi-Z pyramid snapshot to temp folder (all mip levels as DDS + PNG files)
// Called when L key is pressed for debugging/analysis
void DistantLand::saveHiZSnapshot() {
    // Create temp directory if it doesn't exist
    const char* tempDir = "temp";
    CreateDirectoryA(tempDir, NULL);

    // Generate timestamped folder name for this snapshot
    SYSTEMTIME st;
    GetLocalTime(&st);
    char snapshotDir[256];
    snprintf(snapshotDir, sizeof(snapshotDir), "%s/cpu_hiz_%04d%02d%02d_%02d%02d%02d",
             tempDir, st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    CreateDirectoryA(snapshotDir, NULL);

    // Save CPU Hi-Z buffers from software occlusion culler
    FixedFunctionShader::softwareOcclusionCuller.saveToDisk(device, snapshotDir);
}

bool DistantLand::cullAgainstHiZ(const D3DXVECTOR3& bboxMin, const D3DXVECTOR3& bboxMax, const D3DXMATRIX& worldViewProj, bool debugLog) {

    // TEMPORARY: Disable Hi-Z culling while debugging pyramid data issue
    return false;

    // Use previous frame's Hi-Z for async culling (no GPU stall waiting for current frame)
    if (!texHiZStagingPrev || hiZLockedMips == 0) return false;

    float minX, maxX, minY, maxY, minLinearDepth, maxLinearDepth;
    int mipLevel;
    D3DSURFACE_DESC desc;
    float cornerDepths[8];
    float boxWidth, boxHeight;
    int pixelMinX, pixelMaxX, pixelMinY, pixelMaxY;

    // Transform bounding box corners to clip space
    D3DXVECTOR3 corners[8] = {
        D3DXVECTOR3(bboxMin.x, bboxMin.y, bboxMin.z),
        D3DXVECTOR3(bboxMax.x, bboxMin.y, bboxMin.z),
        D3DXVECTOR3(bboxMin.x, bboxMax.y, bboxMin.z),
        D3DXVECTOR3(bboxMax.x, bboxMax.y, bboxMin.z),
        D3DXVECTOR3(bboxMin.x, bboxMin.y, bboxMax.z),
        D3DXVECTOR3(bboxMax.x, bboxMin.y, bboxMax.z),
        D3DXVECTOR3(bboxMin.x, bboxMax.y, bboxMax.z),
        D3DXVECTOR3(bboxMax.x, bboxMax.y, bboxMax.z)
    };

    // Transform corners and find screen space bounds
    minX = 1e10f, maxX = -1e10f;
    minY = 1e10f, maxY = -1e10f;
    minLinearDepth = 1e10f;
    maxLinearDepth = -1e10f;
    bool anyInFront = false;
    bool anyBehindCamera = false;

    for (int i = 0; i < 8; i++) {
        D3DXVECTOR4 clipPos;
        D3DXVec3Transform(&clipPos, &corners[i], &worldViewProj);

        if (clipPos.w > 0.0f) {
            anyInFront = true;
            float invW = 1.0f / clipPos.w;
            float ndcX = clipPos.x * invW;
            float ndcY = clipPos.y * invW;
            float linearDepth = clipPos.w;
            cornerDepths[i] = linearDepth;

            float screenX = ndcX * 0.5f + 0.5f;
            float screenY = -ndcY * 0.5f + 0.5f;

            minX = std::min(minX, screenX);
            maxX = std::max(maxX, screenX);
            minY = std::min(minY, screenY);
            maxY = std::max(maxY, screenY);
            minLinearDepth = std::min(minLinearDepth, linearDepth);
            maxLinearDepth = std::max(maxLinearDepth, linearDepth);
        } else {
            anyBehindCamera = true;
            cornerDepths[i] = -1.0f;
        }
    }

    if (!anyInFront) return false;

    // If any corners behind camera, object intersects near plane - never cull
    if (anyBehindCamera) {
        if (debugLog) {
            LOG::logline("Hi-Z Debug: Object intersects near plane - forced visible");
        }
        return false;
    }

    // Clamp to screen bounds
    minX = std::max(0.0f, std::min(1.0f, minX));
    maxX = std::max(0.0f, std::min(1.0f, maxX));
    minY = std::max(0.0f, std::min(1.0f, minY));
    maxY = std::max(0.0f, std::min(1.0f, maxY));

    if (maxX <= 0.0f || minX >= 1.0f || maxY <= 0.0f || minY >= 1.0f) {
        return false;
    }

    // Calculate bbox screen size and select appropriate mip level
    texHiZStagingPrev->GetLevelDesc(0, &desc);
    boxWidth = (maxX - minX) * desc.Width;
    boxHeight = (maxY - minY) * desc.Height;
    float boxSize = std::max(boxWidth, boxHeight);

    // Select mip level: target ~6 pixels at selected mip for good coverage
    // Add -1 mip bias to use higher resolution (one mip level lower/more detailed)
    mipLevel = 0;
    if (boxSize > 8.0f) {
        mipLevel = (int)std::floor(std::log2(boxSize / 6.0f)) - 1;
        mipLevel = std::max(0, std::min(hiZLevels - 1, mipLevel));
    }

    // Check if the required mip is locked yet (split locking strategy)
    if (mipLevel >= hiZLockedMips) {
        // This mip hasn't been locked yet - don't cull (conservative approach)
        return false;
    }

    // Use pre-locked staging texture (locked in Present, no stall here!)
    float maxHiZDepth, minHiZDepth;
    int pixelCount;

    // Use pre-locked rect data
    D3DLOCKED_RECT& lr = hiZLockedRects[mipLevel];

    // Get mip level dimensions
    texHiZStagingPrev->GetLevelDesc(mipLevel, &desc);

    // Calculate pixel range covered by bbox at this mip level
    pixelMinX = (int)(minX * desc.Width);
    pixelMaxX = (int)(maxX * desc.Width);
    pixelMinY = (int)(minY * desc.Height);
    pixelMaxY = (int)(maxY * desc.Height);

    // Clamp to texture bounds
    pixelMinX = std::max(0, std::min((int)desc.Width - 1, pixelMinX));
    pixelMaxX = std::max(0, std::min((int)desc.Width - 1, pixelMaxX));
    pixelMinY = std::max(0, std::min((int)desc.Height - 1, pixelMinY));
    pixelMaxY = std::max(0, std::min((int)desc.Height - 1, pixelMaxY));

    // Sample ALL Hi-Z pixels in bbox region and find maximum depth
    float* depthData = (float*)lr.pBits;
    int pitch = lr.Pitch / sizeof(float);
    maxHiZDepth = 0.0f;
    minHiZDepth = 1e10f;
    pixelCount = 0;

    for (int y = pixelMinY; y <= pixelMaxY; y++) {
        for (int x = pixelMinX; x <= pixelMaxX; x++) {
            float depth = depthData[y * pitch + x];
            maxHiZDepth = std::max(maxHiZDepth, depth);
            minHiZDepth = std::min(minHiZDepth, depth);
            pixelCount++;
        }
    }

    // No unlock needed - already unlocked at start of next frame's generateHiZPyramid

    // Detect depth discontinuities (gaps between buildings, fences, etc.)
    // If there's a large depth variation in the screen-space region, don't cull
    // because the object might be visible through the gap
    float depthRange = maxHiZDepth - minHiZDepth;

    // For large objects, always check at mip 0 for gap detection to avoid false culling
    // when coarse mips fill in thin gaps through dilation
    bool hasGap = false;
    if (mipLevel > 0 && depthRange > 50.0f) {
        // Re-check at mip 0 for more accurate gap detection using pre-locked data
        D3DLOCKED_RECT& lr0 = hiZLockedRects[0];
        D3DSURFACE_DESC desc0;
        texHiZStagingPrev->GetLevelDesc(0, &desc0);

        int pix0MinX = (int)(minX * desc0.Width);
        int pix0MaxX = (int)(maxX * desc0.Width);
        int pix0MinY = (int)(minY * desc0.Height);
        int pix0MaxY = (int)(maxY * desc0.Height);

        pix0MinX = std::max(0, std::min((int)desc0.Width - 1, pix0MinX));
        pix0MaxX = std::max(0, std::min((int)desc0.Width - 1, pix0MaxX));
        pix0MinY = std::max(0, std::min((int)desc0.Height - 1, pix0MinY));
        pix0MaxY = std::max(0, std::min((int)desc0.Height - 1, pix0MaxY));

        float* depth0 = (float*)lr0.pBits;
        int pitch0 = lr0.Pitch / sizeof(float);
        float max0 = 0.0f;
        float min0 = 1e10f;

        for (int y = pix0MinY; y <= pix0MaxY; y++) {
            for (int x = pix0MinX; x <= pix0MaxX; x++) {
                float d = depth0[y * pitch0 + x];
                max0 = std::max(max0, d);
                min0 = std::min(min0, d);
            }
        }

        float range0 = max0 - min0;
        if (range0 > 100.0f) {
            hasGap = true;
            if (debugLog) {
                LOG::logline("Hi-Z Debug: Gap detected at mip 0 (range=%.1f) - forced visible", range0);
            }
        }
    } else if (depthRange > 100.0f) {
        hasGap = true;
        if (debugLog) {
            LOG::logline("Hi-Z Debug: Depth discontinuity detected (range=%.1f) - forced visible", depthRange);
        }
    }

    if (hasGap) {
        return false; // Don't cull - might be visible through gap
    }

    // Occlusion test: if closest bbox corner is behind furthest visible depth, cull
    // Increased bias to handle depth precision errors and bbox expansion padding
    float bias = 20.0f;
    bool culled = minLinearDepth > maxHiZDepth + bias;

    if (debugLog && culled) {
        LOG::logline("Hi-Z Debug CULLED: minDepth=%.4f, maxDepth=%.4f, maxHiZ=%.4f, mip=%d, region=[%d,%d]->[%d,%d]",
                     minLinearDepth, maxLinearDepth, maxHiZDepth, mipLevel, pixelMinX, pixelMinY, pixelMaxX, pixelMaxY);
        LOG::logline("  Corner depths: [0]=%.2f [1]=%.2f [2]=%.2f [3]=%.2f [4]=%.2f [5]=%.2f [6]=%.2f [7]=%.2f",
                     cornerDepths[0], cornerDepths[1], cornerDepths[2], cornerDepths[3],
                     cornerDepths[4], cornerDepths[5], cornerDepths[6], cornerDepths[7]);
        LOG::logline("  Hi-Z depth range: min=%.4f max=%.4f (%d pixels sampled)", minHiZDepth, maxHiZDepth, pixelCount);
        LOG::logline("  Screen bounds: X=[%.3f,%.3f] Y=[%.3f,%.3f] size=%.1fx%.1f pixels",
                     minX, maxX, minY, maxY, boxWidth, boxHeight);
        LOG::logline("  BBox: min=(%.1f,%.1f,%.1f) max=(%.1f,%.1f,%.1f)",
                     bboxMin.x, bboxMin.y, bboxMin.z, bboxMax.x, bboxMax.y, bboxMax.z);
    }

    return culled;
}

// --------------------------------------------------------
// Texture-Based Lighting System
// --------------------------------------------------------

float DistantLand::computeLightRadius(float constant, float linear, float quadratic) {
    // Match shader attenuation: (1 / (40*q*d² + c)) * (1 - (d/350)^4)
    // Use fixed 350 unit cutoff from shader, but extend slightly for culling safety margin
    // The shader's quartic cutoff at 350 units means lights are effectively invisible beyond that

    // Use 400 units as safe maximum (350 + margin for Hi-Z culling tolerance)
    return 400.0f;
}



