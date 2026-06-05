
#include "distantland.h"
#include "distantshader.h"
#include "drawstats.h"
#include "configuration.h"
#include "mwbridge.h"
#include "phasetimers.h"
#include "proxydx/d3d8header.h"
#include "scenegraph_geometry_cache.h"
#include "support/log.h"
#include "mge_tracy.h"

#include <cmath>



static const float shadowNearRadius = 1000.0;
static const float shadowFarRadius = 4000.0;



// clearShadowCascade — clears one cascade's region of the shadow atlas.
// device->Clear is viewport-bounded when no rect array is provided, and
// the fullscreen quad in PASS_CLEARSHADOWMAP is also viewport-clipped,
// so this leaves the OTHER cascade's region untouched. The adaptive
// scheduler in renderShadowMap relies on this for the cascade-1-skip
// case (cascade 1's region must persist across the skip frame).
void DistantLand::clearShadowCascade(int layer) {
    const DWORD res = Configuration.DL.ShadowResolution;
    D3DVIEWPORT9 vp = { (DWORD)(layer * (int)res), 0, res, res, 0.0f, 1.0f };
    device->SetViewport(&vp);

    device->Clear(0, 0, D3DCLEAR_ZBUFFER|D3DCLEAR_STENCIL, 0, 1.0, 0);
    effectShadow->BeginPass(PASS_CLEARSHADOWMAP);
    effect->SetBool(ehHasAlpha, false);
    effectShadow->CommitChanges();
    device->SetVertexDeclaration(WaterDecl);
    device->SetStreamSource(0, vbFullFrame, 0, 12);
    DrawStats::count(2);
    device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
    effectShadow->EndPass();
}

// renderShadowMap
// Renders cascaded shadow map atlas with adaptive scheduling:
//   - Still frame (camera + sun + cell unchanged) → skip everything;
//     last frame's atlas is byte-identical valid.
//   - Moving frame, cascade 1 skip parity → render only cascade 0;
//     cascade 1's region of the atlas (and its smViewproj matrix)
//     persist from the previous render.
//   - Moving frame, cascade 1 update parity → render both cascades.
//   - First frame / cell change → force-render both cascades.
//
// Soften pass is viewport-clipped to match the cascades that were
// re-rendered, so the cascade 1 region (when skipped) doesn't get
// re-blurred on top of last frame's already-blurred result.
//
// This *must* restore render state on return.
void DistantLand::renderShadowMap() {
    MGE_ZoneScopedN("renderShadowMap");
    MGE_SCOPED_TIMER("renderShadowMap");
    DrawStats::ScopedStage _ds(DrawStats::Shadow);

    // ---- adaptive scheduler state ----
    static D3DXMATRIX  s_lastView = {};
    static D3DXVECTOR4 s_lastSunVec = {};
    static D3DXVECTOR4 s_lastSunPos = {};
    static const void* s_lastWorld = nullptr;
    static bool        s_haveValidAtlas = false;
    static unsigned    s_movingFrameCounter = 0;

    // Skip-rate diagnostics — gated by Configuration.LogDistantPipeline,
    // dumped every 1800 frames (~30s at 60fps). Tells us how often each
    // skip path activates so we can tune the adaptive thresholds.
    static unsigned s_diagFull = 0;       // full-skip frame count
    static unsigned s_diagC1Skip = 0;     // cascade-1-skip frame count
    static unsigned s_diagFull2 = 0;      // both-cascade render count
    static unsigned s_diagTotal = 0;

    const void* curWorld = (const void*)DistantLandShare::currentWorldSpace;
    const bool cellChanged = (s_lastWorld != curWorld);
    const bool viewChanged = memcmp(&s_lastView, &mwView, sizeof(D3DXMATRIX)) != 0;
    const bool sunChanged  = memcmp(&s_lastSunVec, &sunVec, sizeof(D3DXVECTOR4)) != 0
                          || memcmp(&s_lastSunPos, &sunPos, sizeof(D3DXVECTOR4)) != 0;
    const bool stillFrame = s_haveValidAtlas && !cellChanged && !viewChanged && !sunChanged;

    ++s_diagTotal;
    if (stillFrame) {
        ++s_diagFull;
        if (Configuration.LogDistantPipeline && (s_diagTotal % 1800 == 0)) {
            LOG::logline("-- [SHADOW-ADAPT] frames=%u full_skip=%u(%.0f%%) c1_skip=%u(%.0f%%) full_render=%u(%.0f%%)",
                s_diagTotal,
                s_diagFull,   100.0 * s_diagFull   / s_diagTotal,
                s_diagC1Skip, 100.0 * s_diagC1Skip / s_diagTotal,
                s_diagFull2,  100.0 * s_diagFull2  / s_diagTotal);
        }
        return;
    }

    // Cache state for next frame's still detection.
    s_lastView = mwView;
    s_lastSunVec = sunVec;
    s_lastSunPos = sunPos;
    s_lastWorld  = curWorld;

    // Decide cascade 1: render every other moving frame, but force-render
    // when the cell changed or there's no valid prior atlas.
    const bool forceFullUpdate = cellChanged || !s_haveValidAtlas;
    const bool renderCascade1  = forceFullUpdate || ((++s_movingFrameCounter & 1u) == 1u);

    if (renderCascade1) ++s_diagFull2; else ++s_diagC1Skip;

    IDirect3DSurface9* target, *targetSoft;
    texShadow->GetSurfaceLevel(0, &target);
    texSoftShadow->GetSurfaceLevel(0, &targetSoft);

    // Switch to render target (caster pass writes to texSoftShadow)
    RenderTargetSwitcher rtsw(targetSoft, surfShadowZ);
    D3DVIEWPORT9 vp;
    device->GetViewport(&vp);

    // Unbind shadow samplers
    effect->SetTexture(ehTex0, 0);
    effect->SetTexture(ehTex2, 0);

    // Per-cascade clear (viewport-bounded). On cascade-1-skip frames we
    // clear only cascade 0's region; cascade 1's region keeps last
    // frame's content.
    clearShadowCascade(0);
    if (renderCascade1) {
        clearShadowCascade(1);
    }

    // Calculate transform to map view frustum into world space
    D3DXMATRIX inverseCameraProj, cameraViewProj;
    D3DXMatrixMultiply(&cameraViewProj, &mwView, &mwProj);
    D3DXMatrixInverse(&inverseCameraProj, NULL, &cameraViewProj);

    // Render near layer (always; cheaper of the two)
    renderShadowLayer(0, shadowNearRadius, &inverseCameraProj);

    // Render far layer conditionally. When skipped, smViewproj[1] keeps
    // last frame's value — receiver pass sees a matrix that matches the
    // atlas content we left in cascade 1's region.
    if (renderCascade1) {
        renderShadowLayer(1, shadowFarRadius, &inverseCameraProj);
    }

    // Soften shadow map. Viewport restricts the soften writes so when
    // cascade 1 is skipped, its already-blurred region from last frame
    // is preserved (re-blurring an already-blurred result would compound
    // the gaussian and make far shadows progressively mushier each skip).
    if (renderCascade1) {
        device->SetViewport(&vp);  // full atlas
    } else {
        const DWORD res = Configuration.DL.ShadowResolution;
        D3DVIEWPORT9 vp0 = { 0, 0, res, res, 0.0f, 1.0f };
        device->SetViewport(&vp0);  // cascade 0 only
    }

    device->SetRenderTarget(0, target);
    effectShadow->BeginPass(PASS_SOFTENSHADOWMAP);
    effect->SetTexture(ehTex3, texSoftShadow);
    effect->SetBool(ehHasAlpha, false);     // flag as horizontal filter pass
    effectShadow->CommitChanges();

    device->SetVertexDeclaration(WaterDecl);
    device->SetStreamSource(0, vbFullFrame, 0, 12);
    DrawStats::count(2);
    device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);

    device->SetRenderTarget(0, targetSoft);
    effect->SetTexture(ehTex3, texShadow);
    effect->SetBool(ehHasAlpha, true);      // flag as vertical filter pass
    effectShadow->CommitChanges();

    DrawStats::count(2);
    device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
    effectShadow->EndPass();

    // Restore full viewport for callers
    device->SetViewport(&vp);

    // Clean up surface pointers
    target->Release();
    targetSoft->Release();

    s_haveValidAtlas = true;

    if (Configuration.LogDistantPipeline && (s_diagTotal % 1800 == 0)) {
        LOG::logline("-- [SHADOW-ADAPT] frames=%u full_skip=%u(%.0f%%) c1_skip=%u(%.0f%%) full_render=%u(%.0f%%)",
            s_diagTotal,
            s_diagFull,   100.0 * s_diagFull   / s_diagTotal,
            s_diagC1Skip, 100.0 * s_diagC1Skip / s_diagTotal,
            s_diagFull2,  100.0 * s_diagFull2  / s_diagTotal);
    }
}

void DistantLand::renderShadowFromCache(int layer, const D3DXMATRIX* viewproj) {
    if (layer != 0) return;  // near cascade only — before zone to avoid ghost entry
    MGE_ZoneScopedN("renderShadowFromCache");

    static constexpr float kShadowMinSize = 50.0f;

    ViewFrustum frustum(viewproj);
    const auto& cacheMap = MGE::GeometryCache::cache();

    auto bindAlpha = [&](const MGE::GeometryCache::CachedGeometry& e) {
        if (e.alphaTest && e.d3dTexture) {
            effect->SetTexture(ehTex0, e.d3dTexture);
            effect->SetBool(ehHasAlpha, true);
            effect->SetFloat(ehAlphaRef, e.alphaRef);
        } else {
            effect->SetTexture(ehTex0, nullptr);
            effect->SetBool(ehHasAlpha, false);
            effect->SetFloat(ehAlphaRef, -1.0f);
        }
    };
    auto culled = [&](const MGE::GeometryCache::CachedGeometry& e) -> bool {
        BoundingSphere bs;
        bs.center = D3DXVECTOR3(e.worldTransformD3D[12], e.worldTransformD3D[13], e.worldTransformD3D[14]);
        bs.radius = e.boundsRadius;
        return frustum.ContainsSphere(bs) == ViewFrustum::OUTSIDE;
    };

    effect->SetFloat(ehMaterialAlpha, 1.0f);
    effect->SetBool(ehHasBones, false);

    // ---- Non-skinned casters: model-space VB, palette[0] = worldTransform*shadowVP ----
    effectShadow->BeginPass(PASS_RENDERSHADOWMAP_MW);
    for (const auto& kv : cacheMap) {
        const auto& e = kv.second;
        if (e.isSkinned) continue;
        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib) continue;
        if (e.blendEnable) continue;
        if (e.dynamicHint == 0 && e.boundsRadius < kShadowMinSize) continue;
        if (culled(e)) continue;

        D3DXMATRIX shadowPalette[4];
        D3DXMatrixMultiply(&shadowPalette[0], reinterpret_cast<const D3DXMATRIX*>(e.worldTransformD3D), viewproj);
        shadowPalette[1] = shadowPalette[2] = shadowPalette[3] = shadowPalette[0];
        effect->SetInt(ehVertexBlendState, 0);
        effect->SetMatrixArray(ehVertexBlendPalette, shadowPalette, 4);

        bindAlpha(e);
        effectShadow->CommitChanges();
        device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
        device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kVBStride);
        device->SetIndices(e.ib);
        device->SetFVF(MGE::GeometryCache::kVBFVF);
        DrawStats::count(e.triangleCount);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, e.vertexCount, 0, e.triangleCount);
    }
    effectShadow->EndPass();

    // ---- Skinned casters: static bind-pose VB + per-frame bone palette (VS skinning).
    // The skinned VS uses shadowViewProj[0], already set to this cascade's viewproj. ----
    effectShadow->BeginPass(PASS_RENDERSHADOWMAP_MW_SKINNED);
    device->SetVertexDeclaration(MGE::GeometryCache::skinnedDecl());
    for (const auto& kv : cacheMap) {
        const auto& e = kv.second;
        if (!e.isSkinned || e.skinnedUnsupported) continue;
        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib || e.numBones == 0) continue;
        if (e.blendEnable) continue;
        if (culled(e)) continue;   // skinned are dynamic — no size cull

        effect->SetMatrixArray(ehBoneMatrices,
            reinterpret_cast<const D3DXMATRIX*>(e.bonePalette.data()), e.numBones);
        bindAlpha(e);
        effectShadow->CommitChanges();
        device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
        device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kSkinnedVBStride);
        device->SetIndices(e.ib);
        DrawStats::count(e.triangleCount);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, e.vertexCount, 0, e.triangleCount);
    }
    effectShadow->EndPass();
}

template<class T>
void DistantLand::renderShadowLayerGeneric(MWBridge* mwBridge, int layer, const D3DXMATRIX* inverseCameraProj, D3DXMATRIX* view, D3DXMATRIX* proj, VisibleSet<T>& visible_set) {
    // Clip to atlas region with viewport
    const DWORD res = Configuration.DL.ShadowResolution;
    D3DVIEWPORT9 vp = { layer * res, 0, res, res, 0.0f, 1.0f };
    device->SetViewport(&vp);

    // Render view frustum to stencil, which limits rendering to visible
    // texels. Pre-scale the stencil cube outward in clip XY by 10% so
    // pixels at the camera-frustum boundary always have caster data
    // when the camera rotates a sub-stencil-pixel amount. Without this
    // margin, screen-edge pixels (especially the bottom band per user
    // observation) periodically fall outside the camera-frustum
    // projection in cascade space as the camera rotates -> the stencil
    // marks no caster contribution there -> fragment thinks it's lit
    // when it should be shadowed -> visible flicker. Z scale stays 1
    // to keep the depth bounds tight; only lateral margin is needed
    // for camera-rotation jitter.
    //
    // Cost: ~21% more stencil pixels covered (1.1 * 1.1 - 1.0). Stencil
    // writes are cheap (no color, no depth-write cost), and the caster
    // pass only does extra work for pixels that would otherwise have
    // had MISSING data — useful work, not waste.
    D3DXMATRIX stencilMargin, expandedInverseCameraProj;
    D3DXMatrixScaling(&stencilMargin, 1.1f, 1.1f, 1.0f);
    D3DXMatrixMultiply(&expandedInverseCameraProj, &stencilMargin, inverseCameraProj);
    effect->SetMatrix(ehWorld, &expandedInverseCameraProj);
    effectShadow->BeginPass(PASS_SHADOWSTENCIL);
    device->SetVertexDeclaration(WaterDecl);
    device->SetStreamSource(0, vbClipCube, 0, 12);
    DrawStats::count(12);
    device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 12);
    effectShadow->EndPass();

    // Render land and statics
    effectShadow->BeginPass(PASS_RENDERSHADOWMAP);

    if (mwBridge->IsExterior()) {
        renderDistantLand(effectShadow, view, proj);
    }

    device->SetVertexDeclaration(StaticDecl);
    visible_set.Render(device, effectShadow, effect, &ehTex0, &ehHasAlpha, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT, true);

    effectShadow->EndPass();
}

// renderShadowLayer - Calculates projection for, and renders, one shadow layer
void DistantLand::renderShadowLayer(int layer, float radius, const D3DXMATRIX* inverseCameraProj) {
    // Per-cascade total: matrix construction + visibility query +
    // stencil pass + caster render. In IPC mode the cull cost overlaps
    // with rendering via parallelRead, so splitting cull from render
    // wouldn't be meaningful — one timer per cascade is the right grain.
    MGE_SCOPED_TIMER(layer == 0 ? "renderShadowLayer:c0" : "renderShadowLayer:c1");
    ZoneNamedN(___tracy_zone_slc0, "renderShadowLayer:c0", g_tracyActive && layer == 0);
    ZoneNamedN(___tracy_zone_slc1, "renderShadowLayer:c1", g_tracyActive && layer != 0);
    auto mwBridge = MWBridge::get();
    D3DXVECTOR3 lookAt, lookAtEye, shadowCameraPos, up(0, 0, 1);
    D3DXMATRIX* view = &smView[layer], *proj = &smProj[layer], *viewproj = &smViewproj[layer];

    // Select light vector, sunPos during daytime, sunVec during night
    D3DXVECTOR4 lightVec = (sunPos.z > 0) ? -sunPos : sunVec;

    // Centre of projection is one radius ahead of the player
    // Not as far in z direction as player is likely looking at the ground plane rather than below
    // This will be split into a non-texel-quantized but temporally stable view position part,
    // and a texel-quantized view rotation part with small magnitude
    lookAt.x = eyePos.x + radius * eyeVec.x;
    lookAt.y = eyePos.y + radius * eyeVec.y;
    lookAt.z = eyePos.z + 0.5f * radius * eyeVec.z;

    // Quantize eye position to partially reduce texture swimming during camera movement
    lookAtEye.x = float(16.0 * std::floor(0.0625 * eyePos.x));
    lookAtEye.y = float(16.0 * std::floor(0.0625 * eyePos.y));
    lookAtEye.z = float(16.0 * std::floor(0.0625 * eyePos.z));

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

    effect->SetMatrixArray(ehShadowViewproj, viewproj, 1);
    effectShadow->CommitChanges();

    // Cull
    ViewFrustum range_frustum(viewproj);

    if (Configuration.UseSharedMemory) {
        visExtraShared.RemoveAll();
        // because shadow meshes don't need to be sorted, we can read and write in parallel
        ipcClient.getVisibleMeshesCoarse(visExtraSharedId, range_frustum, VIS_STATIC);

        renderShadowLayerGeneric(mwBridge, layer, inverseCameraProj, view, proj, visExtraShared);
    } else {
        VisibleSet<StlVector> visible_set((StlVector()));

        DistantLandShare::currentWorldSpace->NearStatics->GetVisibleMeshesCoarse(range_frustum, visible_set);
        DistantLandShare::currentWorldSpace->FarStatics->GetVisibleMeshesCoarse(range_frustum, visible_set);
        DistantLandShare::currentWorldSpace->VeryFarStatics->GetVisibleMeshesCoarse(range_frustum, visible_set);

        renderShadowLayerGeneric(mwBridge, layer, inverseCameraProj, view, proj, visible_set);
    }

    // Render Morrowind world geometry as shadow casters from the geometry cache.
    renderShadowFromCache(layer, viewproj);
}

// renderShadow - Renders shadows (using blending) over Morrowind shadow receivers
void DistantLand::renderShadow() {
    MGE_ZoneScopedN("applyShadows");
    DrawStats::ScopedStage _ds(DrawStats::Shadow);
    // Supply view space -> shadow clip space matrix
    D3DXMATRIX inverseView, viewToShadow[2];
    D3DXMatrixInverse(&inverseView, NULL, &mwView);
    viewToShadow[0] = inverseView * smViewproj[0];
    viewToShadow[1] = inverseView * smViewproj[1];
    effect->SetMatrixArray(ehShadowViewproj, viewToShadow, 2);

    // Bind filtered ESM
    effect->SetTexture(ehTex3, texSoftShadow);

    // Full-strength receiver in the main view (the cache reflection pass fades this
    // per object; the receiver multiplies OUT.light by it, so it must be 1 here).
    effect->SetFloat(ehShadowReflMult, 1.0f);

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
        DrawStats::count(i.primCount);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);
    }
}

// renderShadowDebug - display shadow layers
void DistantLand::renderShadowDebug() {
    DrawStats::ScopedStage _ds(DrawStats::Debug);
    UINT passes;

    // Create shadow clip space -> camera clip space matrices
    D3DXMATRIX inverseShadowViewProj, cameraViewProj, shadowToCameraProj[2];

    D3DXMatrixMultiply(&cameraViewProj, &mwView, &mwProj);
    D3DXMatrixInverse(&inverseShadowViewProj, NULL, &smViewproj[0]);
    D3DXMatrixMultiply(&shadowToCameraProj[0], &inverseShadowViewProj, &cameraViewProj);
    D3DXMatrixInverse(&inverseShadowViewProj, NULL, &smViewproj[1]);
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
    DrawStats::count(2);
    device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
    effect->EndPass();
    effect->End();
}
