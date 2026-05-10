
#include "distantland.h"
#include "distantshader.h"
#include "configuration.h"
#include "mwbridge.h"
#include "proxydx/d3d8header.h"
#include "imgui_manager.h"
#include "support/log.h"

#include <cmath>
#include <algorithm>

// Frustum-fitted Cascaded Shadow Maps (Lengyel-style)
// Each cascade tightly fits a slice of the view frustum projected into light space

// Compute 8 corners of a frustum slice between splitNear and splitFar (world distances)
// Uses linear interpolation in world space since NDC z is non-linear for perspective
static void computeFrustumSliceCorners(
    const D3DXMATRIX* invViewProj,
    float splitNear, float splitFar,
    D3DXVECTOR3 corners[8])
{
    // First get the full frustum corners at NDC z=0 (near) and z=1 (far)
    const float ndcX[4] = {-1, 1, 1, -1};
    const float ndcY[4] = {-1, -1, 1, 1};

    D3DXVECTOR3 nearCorners[4], farCorners[4];

    for (int i = 0; i < 4; i++) {
        D3DXVECTOR4 nearNDC(ndcX[i], ndcY[i], 0.0f, 1.0f);
        D3DXVECTOR4 farNDC(ndcX[i], ndcY[i], 1.0f, 1.0f);
        D3DXVECTOR4 worldNear, worldFar;

        D3DXVec4Transform(&worldNear, &nearNDC, invViewProj);
        D3DXVec4Transform(&worldFar, &farNDC, invViewProj);

        worldNear /= worldNear.w;
        worldFar /= worldFar.w;

        nearCorners[i] = D3DXVECTOR3(worldNear.x, worldNear.y, worldNear.z);
        farCorners[i] = D3DXVECTOR3(worldFar.x, worldFar.y, worldFar.z);
    }

    // Compute actual frustum depth from corners (camera's real far plane)
    D3DXVECTOR3 nearCenter = (nearCorners[0] + nearCorners[1] + nearCorners[2] + nearCorners[3]) * 0.25f;
    D3DXVECTOR3 farCenter = (farCorners[0] + farCorners[1] + farCorners[2] + farCorners[3]) * 0.25f;
    float frustumDepth = D3DXVec3Length(&(farCenter - nearCenter));

    // Lerp factors for the slice (relative to actual camera frustum)
    float nearLerp = splitNear / frustumDepth;
    float farLerp = splitFar / frustumDepth;

    // Clamp to valid range (in case shadowDistance > camera far)
    nearLerp = std::min(nearLerp, 1.0f);
    farLerp = std::min(farLerp, 1.0f);

    // Interpolate to get slice corners
    for (int i = 0; i < 4; i++) {
        D3DXVECTOR3 dir = farCorners[i] - nearCorners[i];
        corners[i] = nearCorners[i] + dir * nearLerp;
        corners[i + 4] = nearCorners[i] + dir * farLerp;
    }
}

// Compute light-space AABB for frustum slice corners
struct LightSpaceBounds {
    float minX, maxX, minY, maxY, minZ, maxZ;
};

static LightSpaceBounds computeLightSpaceBounds(
    const D3DXVECTOR3 corners[8],
    const D3DXMATRIX* lightView)
{
    LightSpaceBounds bounds = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };

    for (int i = 0; i < 8; i++) {
        D3DXVECTOR4 corner(corners[i].x, corners[i].y, corners[i].z, 1.0f);
        D3DXVECTOR4 lightSpace;
        D3DXVec4Transform(&lightSpace, &corner, lightView);

        bounds.minX = std::min(bounds.minX, lightSpace.x);
        bounds.maxX = std::max(bounds.maxX, lightSpace.x);
        bounds.minY = std::min(bounds.minY, lightSpace.y);
        bounds.maxY = std::max(bounds.maxY, lightSpace.y);
        bounds.minZ = std::min(bounds.minZ, lightSpace.z);
        bounds.maxZ = std::max(bounds.maxZ, lightSpace.z);
    }

    return bounds;
}

// Compute split distances using practical split scheme (blend of log and linear)
static void computeCascadeSplits(float nearClip, float farClip, float lambda, float splits[kShadowCascadeCount + 1])
{
    splits[0] = 0.0f;

    for (int i = 1; i < kShadowCascadeCount; i++) {
        float p = (float)i / (float)kShadowCascadeCount;
        float logSplit = nearClip * std::pow(farClip / nearClip, p);
        float linearSplit = nearClip + (farClip - nearClip) * p;
        splits[i] = lambda * logSplit + (1.0f - lambda) * linearSplit;
    }
    splits[kShadowCascadeCount] = farClip;
}



// renderShadowMap
// Renders multiple shadow map layers to channels in one texture
// Applies filtering to soften shadow edges
// This *must* restore render state on return
void DistantLand::renderShadowMap(DLContext* ctx, FixedFunctionShader::FrameBuffer* fb) {
    LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] renderShadowMap ENTER");
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_ShadowMap, 0);
    IDirect3DSurface9* target, *targetSoft;
    texShadow->GetSurfaceLevel(0, &target);
    texSoftShadow->GetSurfaceLevel(0, &targetSoft);
    LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] GetSurfaceLevel done");

    // Switch to render target
    RenderTargetSwitcher rtsw(targetSoft, surfShadowZ);
    D3DVIEWPORT9 vp;
    device->GetViewport(&vp);
    LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] RT switch done");

    // Unbind shadow samplers
    effect->SetTexture(ehTex0, 0);
    effect->SetTexture(ehTex2, 0);

    // Clear floating point buffer to far depth
    device->Clear(0, 0, D3DCLEAR_ZBUFFER|D3DCLEAR_STENCIL, 0, 1.0, 0);
    effectShadow->BeginPass(PASS_CLEARSHADOWMAP);
    effect->SetBool(ehHasAlpha, false);
    effectShadow->CommitChanges();
    device->SetVertexDeclaration(WaterDecl);
    device->SetStreamSource(0, vbFullFrame, 0, 12);
    device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
    effectShadow->EndPass();
    LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] clear pass done");

    // Calculate transform to map view frustum into world space
    D3DXMATRIX inverseCameraProj, cameraViewProj;
    D3DXMatrixMultiply(&cameraViewProj, &ctx->mwView, &ctx->mwProj);
    D3DXMatrixInverse(&inverseCameraProj, NULL, &cameraViewProj);

    // Compute cascade split points using practical split scheme
    float nearClip = 1.0f;  // Positive reference distance for logarithmic splitting.
    float shadowDistance = ImGuiManager::GetShadowDistance();
    float splitLambda = ImGuiManager::GetSplitLambda();
    float splits[kShadowCascadeCount + 1];
    computeCascadeSplits(nearClip, shadowDistance, splitLambda, splits);

    for (int layer = 0; layer < kShadowCascadeCount; ++layer) {
        LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] layer%d start", layer);
        renderShadowLayer(ctx, fb, layer, splits[layer], splits[layer + 1], &inverseCameraProj);
        LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] layer%d done", layer);
    }

    // Reset viewport
    device->SetViewport(&vp);

    // Soften shadow map
    device->SetRenderTarget(0, target);
    effectShadow->BeginPass(PASS_SOFTENSHADOWMAP);
    effect->SetTexture(ehTex3, texSoftShadow);
    effect->SetBool(ehHasAlpha, false);     // flag as horizontal filter pass
    effectShadow->CommitChanges();

    device->SetVertexDeclaration(WaterDecl);
    device->SetStreamSource(0, vbFullFrame, 0, 12);
    device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);

    device->SetRenderTarget(0, targetSoft);
    effect->SetTexture(ehTex3, texShadow);
    effect->SetBool(ehHasAlpha, true);      // flag as vertical filter pass
    effectShadow->CommitChanges();

    device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
    effectShadow->EndPass();

    // Clean up surface pointers
    target->Release();
    targetSoft->Release();
}

void DistantLand::renderShadowRecorded(const std::vector<RecordedMWState>& recMW, int layer, const D3DXMATRIX* viewproj) {
    if (layer > 1 || recMW.empty()) {
        return;
    }

    effectShadow->BeginPass(PASS_RENDERSHADOWMAP_MW);

    for (const auto& i : recMW) {
        if (i.sceneNum != 0) {
            continue;
        }
        if (i.blendEnable && i.destBlend == D3DBLEND_ONE) {
            continue;
        }

        bool alphaDependent = i.alphaTest || i.blendEnable;
        effect->SetFloat(ehMaterialAlpha, alphaDependent ? i.diffuseMaterial.a : 1.0f);
        if (alphaDependent && i.texture) {
            effect->SetTexture(ehTex0, i.texture);
            effect->SetBool(ehHasAlpha, true);
            effect->SetFloat(ehAlphaRef, i.alphaTest ? (i.alphaRef / 255.0f) : 180.0f / 255.0f);
        } else {
            effect->SetTexture(ehTex0, 0);
            effect->SetBool(ehHasAlpha, false);
            effect->SetFloat(ehAlphaRef, -1.0f);
        }

        D3DXMATRIX shadowPalette[4];
        if (i.vertexBlendState != 0) {
            for (int j = 0; j < 4; ++j) {
                shadowPalette[j] = i.worldTransforms[j] * (*viewproj);
            }
        } else {
            shadowPalette[0] = i.worldTransforms[0] * (*viewproj);
            for (int j = 1; j < 4; ++j) {
                shadowPalette[j] = shadowPalette[0];
            }
        }

        effect->SetBool(ehHasBones, i.vertexBlendState != 0);
        effect->SetInt(ehVertexBlendState, i.vertexBlendState);
        effect->SetMatrixArray(ehVertexBlendPalette, shadowPalette, 4);
        effectShadow->CommitChanges();

        DWORD cull = (i.cullMode != D3DCULL_NONE) ? i.cullMode : (DWORD)D3DCULL_CW;
        device->SetRenderState(D3DRS_CULLMODE, cull);
        device->SetStreamSource(0, i.vb, i.vbOffset, i.vbStride);
        device->SetIndices(i.ib);
        device->SetFVF(i.fvf);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);
    }

    effectShadow->EndPass();
}

template<class T>
void DistantLand::renderShadowLayerGeneric(DLContext* ctx, MWBridge* mwBridge, int layer, const D3DXMATRIX* inverseCameraProj, D3DXMATRIX* view, D3DXMATRIX* proj, VisibleSet<T>& visible_set) {
    // Clip to atlas region with viewport
    const DWORD res = Configuration.DL.ShadowResolution;
    D3DVIEWPORT9 vp = { layer * res, 0, res, res, 0.0f, 1.0f };
    device->SetViewport(&vp);

    // Render view frustum to stencil, which limits rendering to visible texels
    effect->SetMatrix(ehWorld, inverseCameraProj);
    effectShadow->BeginPass(PASS_SHADOWSTENCIL);
    device->SetVertexDeclaration(WaterDecl);
    device->SetStreamSource(0, vbClipCube, 0, 12);
    device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 12);
    effectShadow->EndPass();

    // Render land and statics (use ctx flag for N-1 thread safety)
    effectShadow->BeginPass(PASS_RENDERSHADOWMAP);

    if (ctx->isExterior) {
        renderDistantLand(ctx, effectShadow, view, proj);
    }

    device->SetVertexDeclaration(StaticDecl);
    visible_set.Render(device, effectShadow, effect, &ehTex0, &ehHasAlpha, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT, true);

    effectShadow->EndPass();
}

// renderShadowLayer - Calculates frustum-fitted projection for one shadow cascade
void DistantLand::renderShadowLayer(DLContext* ctx, FixedFunctionShader::FrameBuffer* fb, int layer, float splitNear, float splitFar, const D3DXMATRIX* inverseCameraProj) {
    LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] renderShadowLayer %d ENTER (split %.1f-%.1f)", layer, splitNear, splitFar);
    auto mwBridge = MWBridge::get();
    D3DXVECTOR3 up(0, 0, 1);
    D3DXMATRIX* view = &ctx->smView[layer], *proj = &ctx->smProj[layer], *viewproj = &ctx->smViewproj[layer];

    // Select light vector, sunPos during daytime, sunVec during night
    D3DXVECTOR4 lightVec = (ctx->sunPos.z > 0) ? -ctx->sunPos : ctx->sunVec;
    D3DXVECTOR3 lightDir(lightVec.x, lightVec.y, lightVec.z);
    D3DXVec3Normalize(&lightDir, &lightDir);

    // Compute frustum slice corners for this cascade
    D3DXVECTOR3 frustumCorners[8];
    computeFrustumSliceCorners(inverseCameraProj, splitNear, splitFar, frustumCorners);

    // Compute frustum slice center for light view matrix positioning
    D3DXVECTOR3 frustumCenter(0, 0, 0);
    for (int i = 0; i < 8; i++) {
        frustumCenter += frustumCorners[i];
    }
    frustumCenter *= 0.125f;

    // Build light view matrix looking along light direction from above the frustum
    const float zrange = kCellSize;
    D3DXVECTOR3 shadowCameraPos = frustumCenter - lightDir * zrange;

    // Quantize camera position to reduce temporal jitter during movement
    shadowCameraPos.x = float(16.0 * std::floor(0.0625 * shadowCameraPos.x));
    shadowCameraPos.y = float(16.0 * std::floor(0.0625 * shadowCameraPos.y));
    shadowCameraPos.z = float(16.0 * std::floor(0.0625 * shadowCameraPos.z));

    D3DXVECTOR3 lookAtPoint = shadowCameraPos + lightDir;
    D3DXMatrixLookAtRH(view, &shadowCameraPos, &lookAtPoint, &up);

    // Transform frustum corners to light space and compute AABB
    LightSpaceBounds bounds = computeLightSpaceBounds(frustumCorners, view);

    // Extend Z range to include potential casters above the frustum
    bounds.minZ -= zrange;

    // Create ortho projection fitting the light-space AABB
    D3DXMatrixOrthoOffCenterRH(proj, bounds.minX, bounds.maxX, bounds.minY, bounds.maxY,
                               -bounds.maxZ, -bounds.minZ);

    *viewproj = (*view) * (*proj);

    // Texel snapping: quantize viewproj translation to shadow map texel grid
    // This prevents shimmer during camera rotation while keeping ortho size stable
    // Clip space range [-1, +1] maps to ShadowResolution texels
    const float quantizer = 2.0f / Configuration.DL.ShadowResolution;
    viewproj->_41 = quantizer * std::floor(viewproj->_41 / quantizer);
    viewproj->_42 = quantizer * std::floor(viewproj->_42 / quantizer);

    LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] layer %d matrix done", layer);
    effect->SetMatrixArray(ehShadowViewproj, viewproj, 1);
    effectShadow->CommitChanges();

    // Cull
    ViewFrustum range_frustum(viewproj);

    // Use snapshotted worldSpace from ctx (thread-safe vs global race with selectDistantCell)
    auto worldSpace = static_cast<const DistantLandShare::WorldSpace*>(ctx->worldSpace);

    LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] layer %d sharedMem=%d worldSpace=%p", layer, Configuration.UseSharedMemory ? 1 : 0, (void*)worldSpace);
    if (Configuration.UseSharedMemory) {
        LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] layer %d IPC RemoveAll", layer);
        visExtraShared.RemoveAll();
        // because shadow meshes don't need to be sorted, we can read and write in parallel
        LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] layer %d IPC getVisibleMeshesCoarse", layer);
        ipcClient.getVisibleMeshesCoarse(visExtraSharedId, range_frustum, VIS_STATIC);
        LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] layer %d IPC done, rendering", layer);

        renderShadowLayerGeneric(ctx, mwBridge, layer, inverseCameraProj, view, proj, visExtraShared);
        renderShadowRecorded(fb ? fb->recordMW : recordMW, layer, viewproj);
        LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] layer %d render done", layer);
    } else {
        // Skip if worldSpace not loaded (e.g., during cell transition)
        if (!worldSpace) {
            LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] layer %d SKIP: ctx->worldSpace is NULL", layer);
            return;
        }

        VisibleSet<StlVector> visible_set((StlVector()));

        worldSpace->NearStatics->GetVisibleMeshesCoarse(range_frustum, visible_set);
        worldSpace->FarStatics->GetVisibleMeshesCoarse(range_frustum, visible_set);
        worldSpace->VeryFarStatics->GetVisibleMeshesCoarse(range_frustum, visible_set);

        renderShadowLayerGeneric(ctx, mwBridge, layer, inverseCameraProj, view, proj, visible_set);
        renderShadowRecorded(fb ? fb->recordMW : recordMW, layer, viewproj);
    }
}

// renderShadow - Renders shadows (using blending) over Morrowind shadow receivers
void DistantLand::renderShadow(DLContext* ctx) {
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_ShadowOverlay, 0, (int)recordMW.size());
    // Supply view space -> shadow clip space matrix
    D3DXMATRIX inverseView, viewToShadow[kShadowCascadeCount];
    D3DXMatrixInverse(&inverseView, NULL, &ctx->mwView);
    for (int i = 0; i < kShadowCascadeCount; ++i) {
        viewToShadow[i] = inverseView * ctx->smViewproj[i];
    }
    effect->SetMatrixArray(ehShadowViewproj, viewToShadow, kShadowCascadeCount);

    // Bind filtered ESM
    effect->SetTexture(ehTex3, texSoftShadow);

    // Use an alpha threshold for solidity that isn't precisely equal to a commonly used value (such as 0.5).
    // Vertex interpolators can be slightly inaccurate and cause a value that should be constant across a triangle
    // to have interpolated fragment values that vary either side of the threshold and cause noise.
    const float alphaThreshold = 0.0101f;

    // Draw shadows over recorded renders
    const auto& recordMW_const = recordMW;
    for (const auto& i : recordMW_const) {
        // Additive alphas do not receive shadows
        if (i.blendEnable && i.destBlend == D3DBLEND_ONE) {
            continue;
        }

        // Fragment colour routing
        bool alphaDependent = i.alphaTest || i.blendEnable;
        effect->SetBool(ehHasVCol, alphaDependent && (i.fvf & D3DFVF_DIFFUSE) != 0);
        effect->SetFloat(ehMaterialAlpha, alphaDependent ? i.diffuseMaterial.a : 1.0f);

        // Only bind texture for alphas
        if (alphaDependent && i.texture) {
            effect->SetTexture(ehTex0, i.texture);
            effect->SetBool(ehHasAlpha, true);
            effect->SetFloat(ehAlphaRef, i.alphaTest ? (i.alphaRef / 255.0f) : alphaThreshold);
        } else {
            effect->SetTexture(ehTex0, 0);
            effect->SetBool(ehHasAlpha, false);
            effect->SetFloat(ehAlphaRef, -1.0f);
        }

        // Skin using worldview matrices for numerical accuracy
        effect->SetBool(ehHasBones, i.vertexBlendState != 0);
        effect->SetInt(ehVertexBlendState, i.vertexBlendState);
        effect->SetMatrixArray(ehVertexBlendPalette, i.worldViewTransforms, 4);
        effect->CommitChanges();

        // Ignore two-sided poly (cull none) mode, shadow casters are drawn with CW culling only,
        // which causes false shadows when cast on the reverse side (wrt normals) of a two-sided poly
        DWORD cull = (i.cullMode != D3DCULL_NONE) ? i.cullMode : (DWORD)D3DCULL_CW;
        device->SetRenderState(D3DRS_CULLMODE, cull);
        device->SetStreamSource(0, i.vb, i.vbOffset, i.vbStride);
        device->SetIndices(i.ib);
        device->SetFVF(i.fvf);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);
    }
}

// renderShadowDebug - display shadow layers
void DistantLand::renderShadowDebug(DLContext* ctx) {
    UINT passes;

    // Create shadow clip space -> camera clip space matrices
    D3DXMATRIX inverseShadowViewProj, cameraViewProj, shadowToCameraProj[kShadowCascadeCount];

    D3DXMatrixMultiply(&cameraViewProj, &ctx->mwView, &ctx->mwProj);
    for (int i = 0; i < kShadowCascadeCount; ++i) {
        D3DXMatrixInverse(&inverseShadowViewProj, NULL, &ctx->smViewproj[i]);
        D3DXMatrixMultiply(&shadowToCameraProj[i], &inverseShadowViewProj, &cameraViewProj);
    }

    // Display shadow layers in top right corner
    effect->Begin(&passes, D3DXFX_DONOTSAVESTATE);
    effect->BeginPass(PASS_DEBUGSHADOW);
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
    effect->SetTexture(ehTex3, texSoftShadow);
    effect->SetMatrixArray(ehVertexBlendPalette, shadowToCameraProj, kShadowCascadeCount);
    effect->CommitChanges();
    device->SetVertexDeclaration(WaterDecl);
    device->SetStreamSource(0, vbFullFrame, 0, 12);
    device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
    effect->EndPass();
    effect->End();
}
