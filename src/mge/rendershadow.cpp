
#include "distantland.h"
#include "distantshader.h"
#include "configuration.h"
#include "mwbridge.h"
#include "proxydx/d3d8header.h"
#include "imgui_manager.h"
#include "support/log.h"

#include <cmath>



// Shadow cascade radii now controlled via ImGui (imgui_manager.cpp)
// Default values: near=1000, far=4000



// renderShadowMap
// Renders multiple shadow map layers to channels in one texture
// Applies filtering to soften shadow edges
// This *must* restore render state on return
void DistantLand::renderShadowMap(DLContext* ctx) {
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

    // Render near layer (changes viewport)
    LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] layer0 start");
    renderShadowLayer(ctx, 0, ImGuiManager::GetShadowNearRadius(), &inverseCameraProj);
    LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] layer0 done");

    // Render far layer (changes viewport)
    LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] layer1 start");
    renderShadowLayer(ctx, 1, ImGuiManager::GetShadowFarRadius(), &inverseCameraProj);
    LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] layer1 done");

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

// renderShadowLayer - Calculates projection for, and renders, one shadow layer
void DistantLand::renderShadowLayer(DLContext* ctx, int layer, float radius, const D3DXMATRIX* inverseCameraProj) {
    LOG_CAT(LOG::Cat_SyncThread, "[SHADOW] renderShadowLayer %d ENTER", layer);
    auto mwBridge = MWBridge::get();
    D3DXVECTOR3 lookAt, lookAtEye, shadowCameraPos, up(0, 0, 1);
    D3DXMATRIX* view = &ctx->smView[layer], *proj = &ctx->smProj[layer], *viewproj = &ctx->smViewproj[layer];

    // Select light vector, sunPos during daytime, sunVec during night
    D3DXVECTOR4 lightVec = (ctx->sunPos.z > 0) ? -ctx->sunPos : ctx->sunVec;

    // Centre of projection is one radius ahead of the player
    // Not as far in z direction as player is likely looking at the ground plane rather than below
    // This will be split into a non-texel-quantized but temporally stable view position part,
    // and a texel-quantized view rotation part with small magnitude
    lookAt.x = ctx->eyePos.x + radius * ctx->eyeVec.x;
    lookAt.y = ctx->eyePos.y + radius * ctx->eyeVec.y;
    lookAt.z = ctx->eyePos.z + 0.5f * radius * ctx->eyeVec.z;

    // Quantize eye position to partially reduce texture swimming during camera movement
    lookAtEye.x = float(16.0 * std::floor(0.0625 * ctx->eyePos.x));
    lookAtEye.y = float(16.0 * std::floor(0.0625 * ctx->eyePos.y));
    lookAtEye.z = float(16.0 * std::floor(0.0625 * ctx->eyePos.z));

    // Create shadow frustum centred on lookAtEye, looking along lightVec
    const float zrange = kCellSize;
    shadowCameraPos.x = lookAtEye.x - zrange * lightVec.x;
    shadowCameraPos.y = lookAtEye.y - zrange * lightVec.y;
    shadowCameraPos.z = lookAtEye.z - zrange * lightVec.z;

    D3DXMatrixLookAtRH(view, &shadowCameraPos, &lookAtEye, &up);
    D3DXMatrixOrthoRH(proj, 2 * radius, (1 + std::fabs(lightVec.z)) * radius, 0, 2.0 * zrange);
    *viewproj = (*view) * (*proj);

    // Transform remainder into shadow clip space and quantize
    // Prevents all shimmer during camera rotation
    D3DXVECTOR3 dv, deltaLookAt = lookAtEye - lookAt;
    D3DXVec3TransformNormal(&dv, &deltaLookAt, viewproj);

    // Quantize clip space range [-1, +1] over ShadowResolution texels
    const float quantizer = 2.0f / Configuration.DL.ShadowResolution;
    viewproj->_41 += quantizer * floor(dv.x / quantizer);
    viewproj->_42 += quantizer * floor(dv.y / quantizer);
    viewproj->_43 += dv.z;

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
    }
}

// renderShadow - Renders shadows (using blending) over Morrowind shadow receivers
void DistantLand::renderShadow(DLContext* ctx) {
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_ShadowOverlay, 0, (int)recordMW.size());
    // Supply view space -> shadow clip space matrix
    D3DXMATRIX inverseView, viewToShadow[2];
    D3DXMatrixInverse(&inverseView, NULL, &ctx->mwView);
    viewToShadow[0] = inverseView * ctx->smViewproj[0];
    viewToShadow[1] = inverseView * ctx->smViewproj[1];
    effect->SetMatrixArray(ehShadowViewproj, viewToShadow, 2);

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
    D3DXMATRIX inverseShadowViewProj, cameraViewProj, shadowToCameraProj[2];

    D3DXMatrixMultiply(&cameraViewProj, &ctx->mwView, &ctx->mwProj);
    D3DXMatrixInverse(&inverseShadowViewProj, NULL, &ctx->smViewproj[0]);
    D3DXMatrixMultiply(&shadowToCameraProj[0], &inverseShadowViewProj, &cameraViewProj);
    D3DXMatrixInverse(&inverseShadowViewProj, NULL, &ctx->smViewproj[1]);
    D3DXMatrixMultiply(&shadowToCameraProj[1], &inverseShadowViewProj, &cameraViewProj);

    // Display shadow layers in top right corner
    effect->Begin(&passes, D3DXFX_DONOTSAVESTATE);
    effect->BeginPass(PASS_DEBUGSHADOW);
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
    effect->SetTexture(ehTex3, texSoftShadow);
    effect->SetMatrixArray(ehVertexBlendPalette, shadowToCameraProj, 2);
    effect->CommitChanges();
    device->SetVertexDeclaration(WaterDecl);
    device->SetStreamSource(0, vbFullFrame, 0, 12);
    device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
    effect->EndPass();
    effect->End();
}
