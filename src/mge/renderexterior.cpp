
#include "distantland.h"
#include "distantshader.h"
#include "configuration.h"
#include "mwbridge.h"
#include "proxydx/d3d8header.h"
#include "support/log.h"

#include <algorithm>



// renderSky - Render atmosphere scattering sky layer and other recorded draw calls on top
void DistantLand::renderSky() {
    // Recorded renders
    const auto& recordSky_const = recordSky;
    const int standardCloudVerts = 65, standardCloudTris = 112;
    const int standardMoonVerts = 4, standardMoonTris = 2;

    // Render sky without clouds first
    effect->BeginPass(PASS_RENDERSKY);
    for (const auto& i : recordSky_const) {
        // Skip clouds
        if (i.texture && i.vertCount == standardCloudVerts && i.primCount == standardCloudTris) {
            continue;
        }

        // Set variables in main effect; variables are shared via effect pool
        effect->SetTexture(ehTex0, i.texture);
        if (i.texture) {
            // Textured object; draw as normal in shader, with exceptions:
            // - Sun/moon billboards do not use mipmaps
            // - Moon shadow cutout (prevents stars shining through moons)
            //   which requires colour to be replaced with atmosphere scattering colour
            bool isBillboard = (i.vertCount == standardMoonVerts && i.primCount == standardMoonTris);
            bool isMoonShadow = i.destBlend == D3DBLEND_INVSRCALPHA && !i.useLighting;

            effect->SetBool(ehHasAlpha, true);
            effect->SetBool(ehHasBones, isBillboard);
            effect->SetBool(ehHasVCol, isMoonShadow);
            device->SetRenderState(D3DRS_ALPHABLENDENABLE, 1);
            device->SetRenderState(D3DRS_SRCBLEND, i.srcBlend);
            device->SetRenderState(D3DRS_DESTBLEND, i.destBlend);
            device->SetRenderState(D3DRS_ALPHATESTENABLE, 1);
        } else {
            // Sky; perform atmosphere scattering in shader
            effect->SetBool(ehHasAlpha, false);
            effect->SetBool(ehHasVCol, true);
            device->SetRenderState(D3DRS_ALPHABLENDENABLE, 0);
            device->SetRenderState(D3DRS_ALPHATESTENABLE, 0);
        }

        effect->SetMatrix(ehWorld, &i.worldTransforms[0]);
        effect->CommitChanges();

        device->SetStreamSource(0, i.vb, i.vbOffset, i.vbStride);
        device->SetIndices(i.ib);
        device->SetFVF(i.fvf);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);
    }
    effect->EndPass();

    // Render clouds with a separate shader
    effect->BeginPass(PASS_RENDERCLOUDS);
    for (const auto& i : recordSky_const) {
        // Clouds only
        if (!(i.texture && i.vertCount == standardCloudVerts && i.primCount == standardCloudTris)) {
            continue;
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
    }
    effect->EndPass();
}

void DistantLand::renderDistantLand(ID3DXEffect* e, const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    D3DXMATRIX world, viewproj = (*view) * (*proj);
    D3DXVECTOR4 viewsphere(eyePos.x, eyePos.y, eyePos.z, Configuration.DL.DrawDist * kCellSize);

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

void DistantLand::cullDistantStatics(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    D3DXMATRIX ds_proj = *proj, ds_viewproj;
    D3DXVECTOR4 viewsphere(eyePos.x, eyePos.y, eyePos.z, 0);
    float zn = nearViewRange - 768.0f, zf = zn;
    float cullDist = fogEnd;

    if (Configuration.UseSharedMemory) {
        visDistantShared.RemoveAll();
    } else {
        visDistant.RemoveAll();
    }

    zf = std::min(Configuration.DL.NearStaticEnd * kCellSize, cullDist);
    if (zn < zf) {
        editProjectionZ(&ds_proj, zn, zf);
        ds_viewproj = (*view) * ds_proj;
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
        editProjectionZ(&ds_proj, zn, zf);
        ds_viewproj = (*view) * ds_proj;
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
        editProjectionZ(&ds_proj, zn, zf);
        ds_viewproj = (*view) * ds_proj;
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

    // MOREFPS phase 4: build the per-frame instance VB when instancing is enabled.
    // Must run after the sort so groups form over consecutive same-state meshes.
    if (Configuration.UseStaticInstancing) {
        if (Configuration.UseSharedMemory) {
            buildStaticInstanceVB(visDistantShared);
        } else {
            buildStaticInstanceVB(visDistant);
        }
    }
}

void DistantLand::renderDistantStatics() {
    if (!MWBridge::get()->IsExterior()) {
        // Set clipping to stop large architectural meshes (that don't match exactly)
        // from visible overdrawing and causing z-buffer occlusion
        float clipAt = nearViewRange - 768.0f;
        D3DXPLANE clipPlane(0, 0, clipAt, -(mwProj._33 * clipAt + mwProj._43));
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

// MOREFPS phase 2: distant-statics instancing helpers.
//
// buildStaticInstanceVB walks the sorted visible set, groups consecutive
// meshes sharing (vBuffer, tex, hasAlpha, animateUV), and packs each
// group's transforms as transposed 4x3 matrices into vbStaticInstances.
// batchedStatics is populated with (firstMesh, instanceCount) pairs.
//
// Mirrors buildGrassInstanceVB (rendergrass.cpp). Difference: grass
// groups only on vBuffer since grass shares texture/pipeline state;
// statics require a wider group key because each static can have its
// own texture/alpha/uv-anim state.
//
// NOTE: the sort key in QuadTreeMesh::CompareByState is currently
// (tex, vBuffer). For optimal grouping, phase 4 will extend it to
// (tex, vBuffer, hasAlpha, animateUV). Until then, this function
// will still produce correct output, just with smaller batches when
// hasAlpha/animateUV vary within a (tex, vBuffer) cluster.
template<class T>
void DistantLand::buildStaticInstanceVB(VisibleSet<T>& staticSet) {
    batchedStatics.clear();

    if (staticSet.Empty()) {
        return;
    }

    if (staticSet.Size() > MaxStaticInstances) {
        static bool warnOnce = true;
        if (warnOnce) {
            LOG::logline("!! Too many distant-static instances. (%d elements, limit %d)", staticSet.Size(), MaxStaticInstances);
            LOG::logline("!! Some distant statics will not render. Reduce draw distance or lower static density.");
            warnOnce = false;
        }
        staticSet.Truncate(MaxStaticInstances);
    }

    // Snapshot the group head BY VALUE. We can't hold a pointer into the
    // source visible-set: with shared-memory IPC, visDistantShared iterates
    // through a remapping window (see IpcClientVector::next()), so any
    // reference into the source is invalidated on the next advance.
    RenderMesh groupHead{};
    bool haveGroupHead = false;
    float* vbwrite = nullptr;
    int instanceCount = 0;

    HRESULT hr = vbStaticInstances->Lock(0, StaticInstStride * staticSet.Size(), (void**)&vbwrite, D3DLOCK_DISCARD);
    if (hr != D3D_OK || vbwrite == 0) {
        return;
    }

    staticSet.Reset();
    while (!staticSet.AtEnd()) {
        const auto& m = staticSet.Next();

        if (!haveGroupHead) {
            groupHead = m;
            haveGroupHead = true;
        } else if (groupHead.vBuffer != m.vBuffer
                   || groupHead.tex != m.tex
                   || groupHead.hasAlpha != m.hasAlpha
                   || groupHead.animateUV != m.animateUV) {
            batchedStatics.push_back(std::make_pair(groupHead, instanceCount));
            groupHead = m;
            instanceCount = 0;
        }

        // Pack per-instance transform as a transposed 4x3 matrix (12 floats).
        // Layout matches vbGrassInstances, so GrassDecl's stream 1 elements
        // (TEXCOORD1/2/3, three FLOAT4 rows) can consume it directly.
        const D3DMATRIX* world = &m.transform;
        vbwrite[0]  = world->_11; vbwrite[1]  = world->_21; vbwrite[2]  = world->_31; vbwrite[3]  = world->_41;
        vbwrite[4]  = world->_12; vbwrite[5]  = world->_22; vbwrite[6]  = world->_32; vbwrite[7]  = world->_42;
        vbwrite[8]  = world->_13; vbwrite[9]  = world->_23; vbwrite[10] = world->_33; vbwrite[11] = world->_43;
        vbwrite += 12;
        instanceCount++;
    }
    if (haveGroupHead) {
        batchedStatics.push_back(std::make_pair(groupHead, instanceCount));
    }

    vbStaticInstances->Unlock();
}

// Explicit instantiations for the two VisibleSet container variants (StlVector / IpcClientVector).
// Mirrors buildGrassInstanceVB's implicit instantiation via direct callers in rendergrass.cpp.
// Not strictly required yet (no callers), but keeps phase 4 wire-up from producing surprise
// linker errors and lets the linker strip this code as dead until that wire-up lands.
template void DistantLand::buildStaticInstanceVB(VisibleSet<StlVector>& staticSet);
template void DistantLand::buildStaticInstanceVB(VisibleSet<IpcClientVector>& staticSet);

// Common core of the two instanced-statics entry points.
// Iterates batchedStatics, rebinds per-group state (texture, alpha test,
// animate-uv), and issues one instanced DrawIndexedPrimitive per group.
//
// Effect-variable writes go through `effect` (the main effect that owns
// the shared effect-pool parameters ehTex0 / ehHasVCol). CommitChanges
// is called on `e` so the currently-running pass (which may live in
// `effect` OR `effectDepth`) picks them up. Mirrors the pattern used in
// renderGrassCommon.
void DistantLand::renderDistantStaticsInstancedCommon(ID3DXEffect* e) {
    device->SetVertexDeclaration(GrassDecl);

    // Reset animate_uv so the first group picks up a fresh state.
    effect->SetBool(ehHasVCol, false);
    e->CommitChanges();

    IDirect3DTexture9* lastTexture = nullptr;
    bool lastAnimateUV = false;
    bool lastHasAlpha = false;
    int firstInstance = 0;

    for (const auto& batch : batchedStatics) {
        const RenderMesh& mesh = batch.first;
        int count = batch.second;

        // Per-group state setup (rebind-filtered by the sort + group boundary rules).
        if (lastTexture != mesh.tex) {
            effect->SetTexture(ehTex0, mesh.tex);
            device->SetRenderState(D3DRS_ALPHATESTENABLE, mesh.hasAlpha);
            lastTexture = mesh.tex;
            lastHasAlpha = mesh.hasAlpha;
        } else if (lastHasAlpha != mesh.hasAlpha) {
            device->SetRenderState(D3DRS_ALPHATESTENABLE, mesh.hasAlpha);
            lastHasAlpha = mesh.hasAlpha;
        }

        if (lastAnimateUV != mesh.animateUV) {
            effect->SetBool(ehHasVCol, mesh.animateUV);
            lastAnimateUV = mesh.animateUV;
        }

        e->CommitChanges();

        // Bind per-group static VB + shared instance VB and issue one instanced draw.
        device->SetIndices(mesh.iBuffer);
        device->SetStreamSourceFreq(0, D3DSTREAMSOURCE_INDEXEDDATA | count);
        device->SetStreamSource(0, mesh.vBuffer, 0, SIZEOFSTATICVERT);
        device->SetStreamSourceFreq(1, D3DSTREAMSOURCE_INSTANCEDATA | 1);
        device->SetStreamSource(1, vbStaticInstances, StaticInstStride * firstInstance, StaticInstStride);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, mesh.verts, 0, mesh.faces);

        firstInstance += count;
    }

    // Restore non-instanced state so later unrelated draws don't inherit the instance freq.
    device->SetStreamSourceFreq(0, 1);
    device->SetStreamSourceFreq(1, 1);
    device->SetStreamSource(1, NULL, 0, 0);
}

// Render the sets built by buildStaticInstanceVB as instanced draws in the
// main color pass. Caller is expected to have invoked BeginPass with either
// PASS_RENDERSTATICSEXTERIOR_INST or PASS_RENDERSTATICSINTERIOR_INST.
// Uses GrassDecl (the layout static-stream + per-instance-transform-stream
// is identical to grass's).
void DistantLand::renderDistantStaticsInstanced() {
    if (batchedStatics.empty()) {
        return;
    }

    if (!MWBridge::get()->IsExterior()) {
        // Same architectural clip-plane workaround as renderDistantStatics.
        float clipAt = nearViewRange - 768.0f;
        D3DXPLANE clipPlane(0, 0, clipAt, -(mwProj._33 * clipAt + mwProj._43));
        device->SetClipPlane(0, clipPlane);
        device->SetRenderState(D3DRS_CLIPPLANEENABLE, 1);
    }

    renderDistantStaticsInstancedCommon(effect);

    device->SetRenderState(D3DRS_CLIPPLANEENABLE, 0);
}

// Render the sets built by buildStaticInstanceVB into the depth pre-pass.
// Caller is expected to have invoked effectDepth->BeginPass with
// PASS_RENDERSTATICSDEPTH_INST. No clip-plane handling here — the
// existing non-instanced depth path doesn't do that either.
void DistantLand::renderDistantStaticsInstancedZ() {
    if (batchedStatics.empty()) {
        return;
    }

    renderDistantStaticsInstancedCommon(effectDepth);
}
