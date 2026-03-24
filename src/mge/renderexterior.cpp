
#include "distantland.h"
#include "distantshader.h"
#include "configuration.h"
#include "mwbridge.h"
#include "proxydx/d3d8header.h"
#include "imgui_manager.h"

#include <algorithm>



// renderSky - Render sky with atmosphere scattering (or simple vertex color when disabled)
// useAtmScatter: true = full atmosphere scattering, false = use vertex colors directly
void DistantLand::renderSky(const std::vector<RecordedMWState>& sky, bool useAtmScatter) {
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_SkyRender, 0, (int)sky.size());
    const int standardCloudVerts = 65, standardCloudTris = 112;
    const int standardMoonVerts = 4, standardMoonTris = 2;

    // Render sky without clouds first
    effect->BeginPass(PASS_RENDERSKY);
    for (const auto& i : sky) {
        // Skip clouds (rendered separately)
        if (i.texture && i.vertCount == standardCloudVerts && i.primCount == standardCloudTris) {
            continue;
        }

        // Debug wireframe mode
        if (i.debugWireframe) {
            device->SetRenderState(D3DRS_FILLMODE, D3DFILL_WIREFRAME);
        }

        effect->SetTexture(ehTex0, i.texture);
        if (i.texture) {
            // Textured object (sun/moon/stars)
            bool isBillboard = (i.vertCount == standardMoonVerts && i.primCount == standardMoonTris);
            bool isMoonShadow = i.destBlend == D3DBLEND_INVSRCALPHA && !i.useLighting;

            effect->SetBool(ehHasAlpha, true);
            effect->SetBool(ehHasBones, isBillboard);
            // Moon shadow uses vertex color for atmosphere tinting (only when ATM_SCATTER on)
            effect->SetBool(ehHasVCol, isMoonShadow && useAtmScatter);
            device->SetRenderState(D3DRS_ALPHABLENDENABLE, 1);
            device->SetRenderState(D3DRS_SRCBLEND, i.srcBlend);
            device->SetRenderState(D3DRS_DESTBLEND, i.destBlend);
            device->SetRenderState(D3DRS_ALPHATESTENABLE, 1);
        } else {
            // Sky dome - always use vertex colors (atmosphere scattering is in the shader)
            effect->SetBool(ehHasAlpha, false);
            effect->SetBool(ehHasVCol, true);  // Shader does atmosphere scattering when true
            device->SetRenderState(D3DRS_ALPHABLENDENABLE, 0);
            device->SetRenderState(D3DRS_ALPHATESTENABLE, 0);
        }

        effect->SetMatrix(ehWorld, &i.worldTransforms[0]);
        effect->CommitChanges();

        device->SetStreamSource(0, i.vb, i.vbOffset, i.vbStride);
        device->SetIndices(i.ib);
        device->SetFVF(i.fvf);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);

        if (i.debugWireframe) {
            device->SetRenderState(D3DRS_FILLMODE, D3DFILL_SOLID);
        }
    }
    effect->EndPass();

    // Render clouds with a separate shader
    effect->BeginPass(PASS_RENDERCLOUDS);
    for (const auto& i : sky) {
        // Clouds only
        if (!(i.texture && i.vertCount == standardCloudVerts && i.primCount == standardCloudTris)) {
            continue;
        }

        // Debug wireframe mode
        if (i.debugWireframe) {
            device->SetRenderState(D3DRS_FILLMODE, D3DFILL_WIREFRAME);
        }

        effect->SetTexture(ehTex0, i.texture);
        effect->SetBool(ehHasAlpha, true);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, 1);
        device->SetRenderState(D3DRS_SRCBLEND, i.srcBlend);
        device->SetRenderState(D3DRS_DESTBLEND, i.destBlend);
        device->SetRenderState(D3DRS_ALPHATESTENABLE, 1);
        effect->SetMatrix(ehWorld, &i.worldTransforms[0]);
        effect->CommitChanges();

        device->SetStreamSource(0, i.vb, i.vbOffset, i.vbStride);
        device->SetIndices(i.ib);
        device->SetFVF(i.fvf);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);

        if (i.debugWireframe) {
            device->SetRenderState(D3DRS_FILLMODE, D3DFILL_SOLID);
        }
    }
    effect->EndPass();
}

void DistantLand::renderDistantLand(DLContext* ctx, ID3DXEffect* e, const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_DistantLand, 0);
    D3DXMATRIX world, viewproj = (*view) * (*proj);
    D3DXVECTOR4 viewsphere(ctx->eyePos.x, ctx->eyePos.y, ctx->eyePos.z, Configuration.DL.DrawDist * kCellSize);

    // Cull and draw
    ViewFrustum frustum(&viewproj);

    if (Configuration.UseSharedMemory) {
        // kick the operation off early so we can do some additional work while it runs
        visLandShared.RemoveAll();
        ipcClient.getVisibleMeshes(visLandSharedId, frustum, viewsphere, VIS_LAND);
    }

    D3DXMatrixIdentity(&world);
    effect->SetMatrix(ehWorld, &world);

    effect->SetTexture(ehTex0, texWorldColour);
    effect->SetTexture(ehTex1, texWorldNormals);
    effect->SetTexture(ehTex2, texWorldDetail);
    e->CommitChanges();

    if (!Configuration.UseSharedMemory) {
        visLand.RemoveAll();
        DistantLandShare::LandQuadTree.GetVisibleMeshes(frustum, viewsphere, visLand);
    }

    device->SetVertexDeclaration(LandDecl);
    
    if (Configuration.UseSharedMemory) {
        visLandShared.Render(device, SIZEOFLANDVERT, true);
    } else {
        visLand.Render(device, SIZEOFLANDVERT);
    }
}

void DistantLand::renderDistantLandZ() {
    D3DXMATRIX world;

    D3DXMatrixIdentity(&world);
    effect->SetMatrix(DistantLand::ehWorld, &world);
    effectDepth->CommitChanges();

    // Draw with cached vis set
    device->SetVertexDeclaration(LandDecl);
    if (Configuration.UseSharedMemory) {
        visLandShared.Render(device, SIZEOFLANDVERT);
    } else {
        visLand.Render(device, SIZEOFLANDVERT);
    }
}

void DistantLand::cullDistantStatics(DLContext* ctx, const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    D3DXMATRIX ds_proj = *proj, ds_viewproj;
    D3DXVECTOR4 viewsphere(ctx->eyePos.x, ctx->eyePos.y, ctx->eyePos.z, 0);
    float zn = ctx->nearViewRange - 768.0f, zf = zn;
    float cullDist = ctx->fogEnd;

    // Diagnostic: log culling distances
    static int cullLogCount = 0;
    if (cullLogCount++ < 10) {
        LOG::logline("[CULL] cullDistantStatics: nearViewRange=%.1f zn=%.1f fogEnd=%.1f",
            ctx->nearViewRange, zn, cullDist);
    }

    if (Configuration.UseSharedMemory) {
        visDistantShared.RemoveAll();
    } else {
        visDistant.RemoveAll();
    }

    zf = std::min(Configuration.DL.NearStaticEnd * kCellSize, cullDist);
    if (zn < zf) {
        // For ultra-wide FOV, use original projection matrix to avoid overly restrictive frustum planes
        if (Configuration.ScreenFOV > 90.0f) {
            ds_viewproj = (*view) * (*proj);
        } else {
            editProjectionZ(&ds_proj, zn, zf);
            ds_viewproj = (*view) * ds_proj;
        }
        ViewFrustum range_frustum(&ds_viewproj);
        viewsphere.w = zf;
        if (Configuration.UseSharedMemory) {
            ipcClient.getVisibleMeshes(visDistantSharedId, range_frustum, viewsphere, VIS_NEAR);
        } else {
            DistantLandShare::currentWorldSpace->NearStatics->GetVisibleMeshes(range_frustum, viewsphere, visDistant);
        }
    }

    zf = std::min(Configuration.DL.FarStaticEnd * kCellSize, cullDist);
    if (zn < zf) {
        // For ultra-wide FOV, use original projection matrix to avoid overly restrictive frustum planes
        if (Configuration.ScreenFOV > 90.0f) {
            ds_viewproj = (*view) * (*proj);
        } else {
            editProjectionZ(&ds_proj, zn, zf);
            ds_viewproj = (*view) * ds_proj;
        }
        ViewFrustum range_frustum(&ds_viewproj);
        viewsphere.w = zf;
        if (Configuration.UseSharedMemory) {
            ipcClient.getVisibleMeshes(visDistantSharedId, range_frustum, viewsphere, VIS_FAR);
        } else {
            DistantLandShare::currentWorldSpace->FarStatics->GetVisibleMeshes(range_frustum, viewsphere, visDistant);
        }
    }

    zf = std::min(Configuration.DL.VeryFarStaticEnd * kCellSize, cullDist);
    if (zn < zf) {
        // For ultra-wide FOV, use original projection matrix to avoid overly restrictive frustum planes
        if (Configuration.ScreenFOV > 90.0f) {
            ds_viewproj = (*view) * (*proj);
        } else {
            editProjectionZ(&ds_proj, zn, zf);
            ds_viewproj = (*view) * ds_proj;
        }
        ViewFrustum range_frustum(&ds_viewproj);
        viewsphere.w = zf;
        if (Configuration.UseSharedMemory) {
            ipcClient.getVisibleMeshes(visDistantSharedId, range_frustum, viewsphere, VIS_VERY_FAR);
        } else {
            DistantLandShare::currentWorldSpace->VeryFarStatics->GetVisibleMeshes(range_frustum, viewsphere, visDistant);
        }
    }

    if (Configuration.UseSharedMemory) {
        ipcClient.sortVisibleSet(visDistantSharedId, VisibleSetSort::ByState);
        ipcClient.waitForCompletion();
    } else {
        visDistant.SortByState();
    }
}

void DistantLand::renderDistantStatics(DLContext* ctx) {
    ImGuiManager::LogFrameEvent(FrameEvent::MGE_DistantStatics, 0);
    if (!MWBridge::get()->IsExterior()) {
        // Set clipping to stop large architectural meshes (that don't match exactly)
        // from visible overdrawing and causing z-buffer occlusion
        float clipAt = ctx->nearViewRange - 768.0f;
        D3DXPLANE clipPlane(0, 0, clipAt, -(ctx->mwProj._33 * clipAt + ctx->mwProj._43));
        device->SetClipPlane(0, clipPlane);
        device->SetRenderState(D3DRS_CLIPPLANEENABLE, 1);
    }

    device->SetVertexDeclaration(StaticDecl);

    if (Configuration.UseSharedMemory) {
        visDistantShared.Render(device, effect, effect, &ehTex0, nullptr, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT);
    } else {
        visDistant.Render(device, effect, effect, &ehTex0, nullptr, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT);
    }

    device->SetRenderState(D3DRS_CLIPPLANEENABLE, 0);
}
