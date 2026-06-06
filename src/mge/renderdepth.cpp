
#include "configuration.h"
#include "distantland.h"
#include "drawstats.h"
#include "distantshader.h"
#include "mwbridge.h"
#include "phasetimers.h"
#include "proxydx/d3d8header.h"
#include "proxydx/devicelock.h"
#include "scenegraph.h"
#include "scenegraph_geometry_cache.h"
#include "support/log.h"
#include "mge_tracy.h"

#include <unordered_set>
#include <vector>

// MSOC-culled visible set received from the VisibleGeomCallback.
// s_visibleKeys: current frame (being built by callback).
// s_prevVisibleKeys: previous frame (ready at renderDepth time).
// Both keyed on NiTriBasedGeometry* cast to uint32_t — matches GeometryCache keys.
// updateVisibleSet (the only writer) runs on the MAIN thread — msoc.dll fires the
// callback from inside the engine's drainPendingDisplays — so a main-thread
// snapshot at kick cannot race it.
static std::unordered_set<uint32_t> s_visibleKeys;
static std::unordered_set<uint32_t> s_prevVisibleKeys;

// Render-thread snapshot of s_prevVisibleKeys, populated on the main thread at
// kick (snapshotVisibleKeysForThread) and read by the worker job. Decouples the
// job from any later main-thread mutation of the set.
static std::vector<uint32_t> s_threadVisibleKeys;

void DistantLand::updateVisibleSet(void* const* shapes, int count) {
    s_prevVisibleKeys = std::move(s_visibleKeys);
    s_visibleKeys.clear();
    s_visibleKeys.reserve(count);
    for (int i = 0; i < count; ++i)
        s_visibleKeys.insert(reinterpret_cast<uint32_t>(shapes[i]));
}

const std::unordered_set<uint32_t>& DistantLand::visibleCacheKeys() {
    return s_prevVisibleKeys;
}



void DistantLand::renderDepth() {
    MGE_ZoneScopedN("renderDepth");
    MGE_SCOPED_TIMER("renderDepth");
    DrawStats::ScopedStage _ds(DrawStats::Depth);
    auto mwBridge = MWBridge::get();

    // Switch to render target
    RenderTargetSwitcher rtsw(texDepthFrame, surfDepthDepth);

    // When the render thread produced the cleared depth + MW cache depth during
    // the sky window, it already wrote them into texDepthFrame/surfDepthDepth
    // (fenced in renderStage0 before this runs). Skip the Clear, the float-depth
    // clear pass, and the cache pass here; land/statics/grass below render on top
    // of the worker's buffer, byte-identical to the serial path.
    const bool depthCacheOnThread = renderThreadJobKicked;

    if (!depthCacheOnThread) {
        device->Clear(0, 0, D3DCLEAR_ZBUFFER, 0, 1.0, 0);
    }

    // Unbind depth sampler
    effect->SetTexture(ehTex3, NULL);

    // Projection should cover whole scene (also used by the land/statics depth
    // passes below, so set it on both paths)
    D3DXMATRIX distProj = mwProj;
    editProjectionZ(&distProj, 4.0f, Configuration.DL.DrawDist * kCellSize);
    effect->SetMatrix(ehProj, &distProj);

    if (!depthCacheOnThread) {
        // Clear floating point buffer to far depth
        effectDepth->BeginPass(PASS_CLEARDEPTH);
        device->SetVertexDeclaration(WaterDecl);
        device->SetStreamSource(0, vbFullFrame, 0, 12);
        DrawStats::count(2);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
        effectDepth->EndPass();

        // Rebuild the geometry cache (walk + per-frame bone palettes) BEFORE the
        // depth draw so depth uses this frame's skinned poses — otherwise depth
        // lags a frame behind the shadow pass and SSAO detaches from moving NPCs.
        // Cheap now that VS palette skinning replaced per-frame CPU skinning, and
        // the VBs are static so there's no depth/shadow aliasing to overlap around.
        //
        // Skipped when frameSetupEarly() already walked it at BeginScene(0) (IPC
        // path) so the ~2ms walk overlaps the sky pass. This call is the fallback
        // for the non-IPC / menu / not-ready paths where earlyWalkedCache is false.
        // (depthCacheOnThread implies earlyWalkedCache, so this whole block is
        // skipped on the threaded path — the worker did the cache pass.)
        if (!earlyWalkedCache) {
            MGE::GeometryCache::onFrameReady(MGE::SceneGraph::getDataHandler());
        }
        {
            MGE_SCOPED_TIMER("renderDepth:cache");
            renderDepthFromCache(&mwView);   // owns its non-skinned + skinned passes
        }
    }

    // Channel-free gate: when frameSetupEarly dispatched the statics verdict to
    // the cull worker, the worker holds the single-channel ipcClient until it
    // has drained the statics RPC. Block here — after the cache-only depth pass
    // above (which touches no ipcClient and overlaps the drain), and before any
    // main-thread ipcClient touch downstream (distant statics consume / grass /
    // shadow / land / water RPCs) — so none of them race the worker's drain.
    // No-op when the worker path is inactive. Now that the kickoff fires before
    // the GeometryCache walk, the drain typically completes during sky + walk
    // and this reads ~0.
    waitCullChannelFree();

    if (isDistantCell()) {
        if (!mwBridge->IsUnderwater(eyePos.z)) {
            // Distant land
            if (mwBridge->IsExterior()) {
                MGE_ZoneScopedN("renderDepth:land");
                MGE_SCOPED_TIMER("renderDepth:land");
                effectDepth->BeginPass(PASS_RENDERLANDDEPTH);
                renderDistantLandZ();
                effectDepth->EndPass();
            }

            // Finish the async statics cull here so depth and color both
            // consume the same msocOccluded mask. Moved from Stage0's
            // color pass so it overlaps with land depth on the GPU.
            if (Configuration.MGEFlags & USE_DISTANT_STATICS) {
                cullDistantStatics_finish();
            }

            {
                MGE_ZoneScopedN("renderDepth:statics");
                MGE_SCOPED_TIMER("renderDepth:statics");
                effectDepth->BeginPass(PASS_RENDERSTATICSDEPTH);
                device->SetVertexDeclaration(StaticDecl);
                // Cull-then-sort: iterate the compacted survivor set (occluded
                // instances already removed by applyMSOCToDistantStatics).
                visDistantSurvivors.Render(device, effectDepth, effect, &ehTex0, &ehHasAlpha, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT, false);
                effectDepth->EndPass();
            }
        }

        if (Configuration.MGEFlags & USE_GRASS) {
            // Cull grass here (moved out of Stage0): culling before the depth
            // pre-pass drained the distant-statics RPC on the one-at-a-time IPC
            // channel right after its Stage0 kickoff, before the GeometryCache
            // walk above could overlap that ~2.3ms server-cull. Culling here —
            // after the walk and cullDistantStatics_finish — lets the statics cull
            // overlap the walk, and grass's own RPC is cheap with the channel now
            // free. Grass still renders in this depth pre-pass, so early-Z holds.
            if (mwBridge->IsExterior()) {
                cullGrass(&mwView, &mwProj);
            }

            // Grass
            MGE_ZoneScopedN("renderDepth:grass");
            MGE_SCOPED_TIMER("renderDepth:grass");
            effectDepth->BeginPass(PASS_RENDERGRASSDEPTHINST);
            renderGrassInstZ();
            effectDepth->EndPass();
        }
    }

    // Reset projection matrix
    effect->SetMatrix(ehProj, &mwProj);
}

void DistantLand::renderDepthAdditional() {
    MGE_ZoneScopedN("renderDepthAdditional");
    DrawStats::ScopedStage _ds(DrawStats::Depth);
    // Switch to render target
    RenderTargetSwitcher rtsw(texDepthFrame, surfDepthDepth);

    // Unbind depth sampler
    effect->SetTexture(ehTex3, NULL);

    // Projection should cover whole scene
    D3DXMATRIX distProj = mwProj;
    editProjectionZ(&distProj, 4.0f, Configuration.DL.DrawDist * kCellSize);
    effect->SetMatrix(ehProj, &distProj);

    // Recorded draw calls
    effectDepth->BeginPass(PASS_RENDERMWDEPTH);
    renderDepthRecorded();
    effectDepth->EndPass();

    // Reset projection matrix
    effect->SetMatrix(ehProj, &mwProj);
}

void DistantLand::renderDepthRecorded() {
    MGE_ZoneScopedN("renderDepthRecorded");
    // Use an alpha threshold for solidity that isn't precisely equal to a commonly used value (such as 0.5).
    // Vertex interpolators can be slightly inaccurate and cause a value that should be constant across a triangle
    // to have interpolated fragment values that vary either side of the threshold and cause noise.
    const float solidThreshold = 0.499f;

    // Recorded renders
    const auto& recordMW_const = recordMW;
    for (const auto& i : recordMW_const) {
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
        effect->SetMatrixArray(ehVertexBlendPalette, i.worldViewTransforms, 4);
        effectDepth->CommitChanges();

        device->SetRenderState(D3DRS_CULLMODE, i.cullMode);
        device->SetStreamSource(0, i.vb, i.vbOffset, i.vbStride);
        device->SetIndices(i.ib);
        device->SetFVF(i.fvf);
        DrawStats::count(i.primCount);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);
    }
}

void DistantLand::renderDepthFromCache(const D3DXMATRIX* gameView,
                                       const std::vector<uint32_t>* visibleOverride) {
    MGE_ZoneScopedN("renderDepthFromCache");

    const float solidThreshold = 0.499f;
    const auto& cacheMap = MGE::GeometryCache::cache();
    const bool useVisibleSet = visibleOverride ? !visibleOverride->empty()
                                               : !s_prevVisibleKeys.empty();

    auto bindMaterial = [&](const MGE::GeometryCache::CachedGeometry& e) {
        bool alphaDependent = e.alphaTest || e.blendEnable;
        if (alphaDependent && e.d3dTexture) {
            effect->SetTexture(ehTex0, e.d3dTexture);
            effect->SetBool(ehHasAlpha, true);
            effect->SetFloat(ehAlphaRef, e.alphaTest ? e.alphaRef : solidThreshold);
        } else {
            effect->SetTexture(ehTex0, nullptr);
            effect->SetBool(ehHasAlpha, false);
            effect->SetFloat(ehAlphaRef, -1.0f);
        }
        // Mirrored (negative-determinant) parts flip clip-space winding, so cull the
        // opposite face — else depth records the inner surface and SSAO shows
        // "inside-out" left limbs. Matches the engine's per-draw mirror swap.
        const DWORD cull = e.blendEnable ? D3DCULL_NONE
                                         : (e.mirrored ? D3DCULL_CCW : D3DCULL_CW);
        device->SetRenderState(D3DRS_CULLMODE, cull);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
        device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
    };

    auto forEach = [&](auto&& fn) {
        if (useVisibleSet) {
            if (visibleOverride) {
                for (uint32_t key : *visibleOverride) {
                    auto it = cacheMap.find(key);
                    if (it != cacheMap.end()) fn(it->second);
                }
            } else {
                for (uint32_t key : s_prevVisibleKeys) {
                    auto it = cacheMap.find(key);
                    if (it != cacheMap.end()) fn(it->second);
                }
            }
        } else {
            for (const auto& kv : cacheMap) fn(kv.second);
        }
    };

    // ---- Non-skinned: model-space VB, per-draw palette[0] = worldTransform*view ----
    effect->SetBool(ehHasVCol, false);
    effect->SetFloat(ehMaterialAlpha, 1.0f);
    effect->SetBool(ehHasBones, false);
    effect->SetInt(ehVertexBlendState, 0);

    effectDepth->BeginPass(PASS_RENDERMWDEPTH);
    forEach([&](const MGE::GeometryCache::CachedGeometry& e) {
        if (e.isSkinned) return;
        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib) return;

        D3DXMATRIX wvMat;
        D3DXMatrixMultiply(&wvMat, reinterpret_cast<const D3DXMATRIX*>(e.worldTransformD3D), gameView);
        D3DXMATRIX wvPalette[4] = { wvMat, wvMat, wvMat, wvMat };
        effect->SetMatrixArray(ehVertexBlendPalette, wvPalette, 4);

        bindMaterial(e);
        effectDepth->CommitChanges();
        device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kVBStride);
        device->SetIndices(e.ib);
        device->SetFVF(MGE::GeometryCache::kVBFVF);
        DrawStats::count(e.triangleCount);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, e.vertexCount, 0, e.triangleCount);
    });
    effectDepth->EndPass();

    // ---- Skinned: static bind-pose VB + per-frame bone palette (VS skinning) ----
    // skinIndexed -> view -> proj; view must be the game view, proj is the depth proj.
    effect->SetMatrix(ehView, gameView);

    effectDepth->BeginPass(PASS_RENDERMWDEPTH_SKINNED);
    device->SetVertexDeclaration(MGE::GeometryCache::skinnedDecl());
    forEach([&](const MGE::GeometryCache::CachedGeometry& e) {
        if (!e.isSkinned || e.skinnedUnsupported) return;
        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib || e.numBones == 0) return;

        effect->SetMatrixArray(ehBoneMatrices,
            reinterpret_cast<const D3DXMATRIX*>(e.bonePalette.data()), e.numBones);
        bindMaterial(e);
        effectDepth->CommitChanges();
        device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kSkinnedVBStride);
        device->SetIndices(e.ib);
        DrawStats::count(e.triangleCount);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, e.vertexCount, 0, e.triangleCount);
    });
    effectDepth->EndPass();

    effect->SetTexture(ehTex0, nullptr);
    effect->SetBool(ehHasAlpha, false);
    effect->SetFloat(ehAlphaRef, -1.0f);
}

// snapshotVisibleKeysForThread - copy the live visible-key set into the
// render-thread snapshot. MAIN THREAD ONLY (updateVisibleSet, the only writer of
// s_prevVisibleKeys, also runs on the main thread), called at kick before the
// worker reads it.
void DistantLand::snapshotVisibleKeysForThread() {
    s_threadVisibleKeys.assign(s_prevVisibleKeys.begin(), s_prevVisibleKeys.end());
}

// renderThreadDepthCacheJob - Phase 1 render-thread payload.
//
// Runs on the MGE render thread, kicked from frameSetupEarly() during the
// engine's sky window, fenced at the top of renderStage0() before any main-thread
// device/effect work. Holds the device-submission lock for its entire body so it
// is atomic against the engine's proxy forwarders. Produces exactly what the
// serial renderDepth cache path produces — cleared depth + the MW geometry-cache
// depth — into texDepthFrame/surfDepthDepth, just earlier (overlapping sky).
//
// Reads only frame-stable, kick-time-fixed data: mwView/mwProj (set by
// frameSetupEarly before the kick, not rewritten until renderStage0 after the
// fence), the GeometryCache (built by the walk before the kick), and the
// s_threadVisibleKeys snapshot. Uses its OWN effectDepth Begin/End bracket — the
// fence guarantees the main thread is not inside an effect bracket concurrently.
void DistantLand::renderThreadDepthCacheJob() {
    MGE_ZoneScopedN("RenderThread:job");
    MGE_DEVLOCK();   // hold the device lock for the whole pass
    DrawStats::ScopedStage _ds(DrawStats::Depth);  // render-thread stage (thread_local)

    if (!device) {
        return;
    }

    // Save the full device state the engine left mid-sky; restore it before
    // releasing the lock so the engine resumes intact. The render target is not
    // captured by state blocks — RenderTargetSwitcher restores it separately.
    IDirect3DStateBlock9* sb = nullptr;
    if (device->CreateStateBlock(D3DSBT_ALL, &sb) != D3D_OK) {
        sb = nullptr;
    }

    UINT passes;
    {
        RenderTargetSwitcher rtsw(texDepthFrame, surfDepthDepth);
        device->Clear(0, 0, D3DCLEAR_ZBUFFER, 0, 1.0f, 0);

        // Unbind depth sampler
        effect->SetTexture(ehTex3, NULL);

        // Projection should cover whole scene
        D3DXMATRIX distProj = mwProj;
        editProjectionZ(&distProj, 4.0f, Configuration.DL.DrawDist * kCellSize);
        effect->SetMatrix(ehProj, &distProj);

        // Own effect bracket — main is not in one yet (fenced before renderStage0).
        effectDepth->Begin(&passes, D3DXFX_DONOTSAVESTATE);

        // Clear floating point buffer to far depth
        effectDepth->BeginPass(PASS_CLEARDEPTH);
        device->SetVertexDeclaration(WaterDecl);
        device->SetStreamSource(0, vbFullFrame, 0, 12);
        DrawStats::count(2);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
        effectDepth->EndPass();

        renderDepthFromCache(&mwView, &s_threadVisibleKeys);

        effectDepth->End();
    }

    if (sb) {
        sb->Apply();
        sb->Release();
    }
}
