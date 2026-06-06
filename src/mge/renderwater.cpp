
#include "distantland.h"
#include "distantshader.h"
#include "drawstats.h"
#include "configuration.h"
#include "doublesurface.h"
#include "ffeshader.h"
#include "mwbridge.h"
#include "postshaders.h"
#include "scenegraph_geometry_cache.h"
#include "mge_tracy.h"

#include <algorithm>
#include <cstdint>
#include <vector>



// Tweakable: the cache-driven reflection covers near objects within this fraction
// of Morrowind's view distance (nearViewRange). Beyond it, distant-land statics
// take over (dissolved in via ehNearViewRange). 1.0 = full MW view distance; lower
// (down to ~0.5) shrinks the lit near-field. TODO: promote to a config/Numpad knob.
static float s_reflectionCacheNearFactor = 1.0f;

// Downward bias (world units) for the cache reflection below-water clip. The clip at
// the exact waterline cuts a thin gap at terrain/water intersections (background
// shows through) and slices straddling characters open at the surface (hollow
// interior visible). Lowering the clip a few units keeps a sliver of near-waterline
// geometry: it fills the terrain gap and sinks the character cut below the surface,
// where the water shading (fresnel/refraction/fog) masks it. Larger = less
// aggressive (more below-water reflection); 0 = clip exactly at the true level.
// TODO: promote to a config/Numpad knob.
static float s_reflWaterClipBias = 3.0f;

// Water-surface snap offsets (world units) applied to the rendered water mesh height,
// chosen by the engine underwater state. A single fixed height can't satisfy both
// sides: above water the wave crests poke up and intersect the near-surface camera,
// and a low surface renders below the camera once submerged. So push the mesh DOWN
// when above water (crests stay below the eye) and keep it at/above the level when
// underwater (surface stays overhead). The snap happens exactly at the IsUnderwater
// threshold, where the camera is at the surface and the jump is hidden. Fine-tune to
// the wave height. TODO: promote to config/Numpad knobs.
static float s_waterMeshSnapAbove      = -5.0f;   // above water: push down
static float s_waterMeshSnapUnderwater =  5.0f;   // underwater: at WaterLevel (default)

// Per-object cache->distant-land handover fade: 1 up close, ramps to 0 over the
// last quarter before nearDist (the handover), using the object's Euclidean origin
// distance — the SAME metric the geometry cull uses. Statics only; dynamics (NPCs)
// keep full intensity. Shared by the point-light fade and the shadow fade so they
// ramp identically and both land exactly on the geometry handover.
static float cacheHandoverFade(const D3DXVECTOR3& center, const D3DXVECTOR3& eye,
                               unsigned dynamicHint, float nearDist) {
    if (dynamicHint != 0 || nearDist <= 0.0f) return 1.0f;
    const D3DXVECTOR3 d = center - eye;
    const float t = (nearDist - D3DXVec3Length(&d)) / (0.25f * nearDist);
    return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
}

// World-space bounding sphere for a cache entry. The cache stores the geometry's
// bound in MODEL space (boundsCenter/boundsRadius); the cull frustum is in world
// space. Using the object ORIGIN (worldTransformD3D translation) as the sphere
// center is wrong whenever the geometry is offset from its node origin — most
// dramatically for terrain patches, whose origin is the cell corner ~4096u from
// the real patch center, causing false culls when the camera is close/over the
// cell. Transform the model center by the world matrix and scale the radius by the
// largest axis scale so the sphere actually encloses the drawn geometry.
static void cacheWorldBounds(const MGE::GeometryCache::CachedGeometry& e, D3DXVECTOR3& outCenter, float& outRadius) {
    const D3DXMATRIX& w = *reinterpret_cast<const D3DXMATRIX*>(e.worldTransformD3D);
    const D3DXVECTOR3 modelC(e.boundsCenter[0], e.boundsCenter[1], e.boundsCenter[2]);
    D3DXVec3TransformCoord(&outCenter, &modelC, &w);
    const float sx = D3DXVec3Length(reinterpret_cast<const D3DXVECTOR3*>(&w._11));
    const float sy = D3DXVec3Length(reinterpret_cast<const D3DXVECTOR3*>(&w._21));
    const float sz = D3DXVec3Length(reinterpret_cast<const D3DXVECTOR3*>(&w._31));
    outRadius = e.boundsRadius * std::max(sx, std::max(sy, sz));
}

// World-space bounding sphere for a SKINNED cache entry, derived from the per-frame
// bone palette. The skinned VB is bind-pose; the posed geometry is placed entirely
// by the bones, so the geom's node origin (worldTransformD3D) is NOT where the posed
// part is. Using the origin as the cull center with the small bind-pose radius
// falsely culls close skinned parts whose skeleton root sits away from the part
// (e.g. a character's own torso at 1-2m: it reflects fine at 5-7m, then vanishes as
// the reflected frustum tightens around the misplaced origin sphere). Transform the
// bind-pose bounds center by each bone (model->world) and bound the resulting point
// set: every posed vertex is a weighted (convex) combination of bone-transformed
// bind positions, so a sphere over those centers plus the scaled bind radius
// conservatively encloses the posed part.
static void cacheSkinnedWorldBounds(const MGE::GeometryCache::CachedGeometry& e,
                                    D3DXVECTOR3& outCenter, float& outRadius) {
    const int n = (int)e.numBones;
    const D3DXMATRIX* pal = reinterpret_cast<const D3DXMATRIX*>(e.bonePalette.data());
    const D3DXVECTOR3 modelC(e.boundsCenter[0], e.boundsCenter[1], e.boundsCenter[2]);

    D3DXVECTOR3 pts[MGE::GeometryCache::kMaxBones];
    D3DXVECTOR3 c(0, 0, 0);
    float maxScale = 0.0f;
    for (int i = 0; i < n; ++i) {
        D3DXVec3TransformCoord(&pts[i], &modelC, &pal[i]);
        c += pts[i];
        const float sx = D3DXVec3Length(reinterpret_cast<const D3DXVECTOR3*>(&pal[i]._11));
        const float sy = D3DXVec3Length(reinterpret_cast<const D3DXVECTOR3*>(&pal[i]._21));
        const float sz = D3DXVec3Length(reinterpret_cast<const D3DXVECTOR3*>(&pal[i]._31));
        maxScale = std::max(maxScale, std::max(sx, std::max(sy, sz)));
    }
    c /= (float)n;
    float r = 0.0f;
    for (int i = 0; i < n; ++i) {
        const D3DXVECTOR3 d = pts[i] - c;
        r = std::max(r, D3DXVec3Length(&d));
    }
    outCenter = c;
    outRadius = r + e.boundsRadius * maxScale;
}

void DistantLand::renderWaterReflection(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    MGE_ZoneScopedN("renderWaterReflection");
    DrawStats::ScopedStage _ds(DrawStats::Reflection);
    auto mwBridge = MWBridge::get();

    // Switch to render target
    RenderTargetSwitcher rtsw(texReflection, surfReflectionZ);
    device->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, horizonCol, 1.0, 0);

    // Calculate reflected view matrix, mirror plane at water mesh level
    D3DXMATRIX reflView;
    D3DXPLANE plane(0, 0, 1.0f, -(mwBridge->WaterLevel() - 1.0f));
    D3DXMatrixReflect(&reflView, &plane);
    D3DXMatrixMultiply(&reflView, &reflView, view);
    effect->SetMatrix(ehView, &reflView);

    // Calculate new projection
    D3DXMATRIX reflProj = *proj;
    editProjectionZ(&reflProj, 4.0, Configuration.DL.DrawDist * kCellSize);
    effect->SetMatrix(ehProj, &reflProj);

    // Clipping setup
    D3DXMATRIX clipMat;

    // Clip geometry on opposite side of water plane
    plane *= mwBridge->IsUnderwater(eyePos.z) ? -1.0f : 1.0f;

    // If using dynamic ripples, the water level can be lowered by up to 0.5 * waveheight
    // so move clip plane downwards at the cost of some reflection errors
    if (Configuration.MGEFlags & DYNAMIC_RIPPLES) {
        plane.d += 0.5f * Configuration.DL.WaterWaveHeight;
    }

    // Doing inverses separately is a lot more numerically stable
    D3DXMatrixInverse(&clipMat, 0, &reflView);
    D3DXMatrixTranspose(&clipMat, &clipMat);
    D3DXPlaneTransform(&plane, &plane, &clipMat);

    D3DXMatrixInverse(&clipMat, 0, &reflProj);
    D3DXMatrixTranspose(&clipMat, &clipMat);
    D3DXPlaneTransform(&plane, &plane, &clipMat);

    if (visDistant.Empty()) {
        // Workaround for a Direct3D bug with clipping planes, where SetClipPlane
        // has no effect on the shader pipeline if the last rendered draw call was using
        // the fixed function pipeline. This is usually covered by distant statics, but
        // not in compact interiors where all distant statics may be culled.
        // Provoking a DrawPrimitive with shader here makes the following SetClipPlane work.
        effect->BeginPass(PASS_WORKAROUND);
        device->SetVertexDeclaration(WaterDecl);
        device->SetStreamSource(0, vbFullFrame, 0, 12);
        DrawStats::count(2);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
        effect->EndPass();
    }

    device->SetClipPlane(0, plane);
    device->SetRenderState(D3DRS_CLIPPLANEENABLE, 1);

    // Per-pixel below-water clip plane for the cache reflection passes, at the TRUE
    // water level. The device clip plane above is lowered by 0.5*waveHeight for
    // dynamic ripples, which lets a band of below-water geometry bleed into the now
    // detailed cache reflection. Same world plane, transformed only into reflected-
    // view space (the first half of the device-plane transform above, minus the
    // projection step) so the cache pixel shaders clip per-fragment against viewpos.
    {
        D3DXPLANE wpln(0, 0, 1.0f, -(mwBridge->WaterLevel() - 1.0f));
        wpln *= mwBridge->IsUnderwater(eyePos.z) ? -1.0f : 1.0f;
        // Lower the clip by s_reflWaterClipBias on the kept side (extends the kept
        // half-space). Increasing d keeps geometry farther past the boundary; the
        // sign-flip above already orients the normal toward the kept side, so this
        // works for both the above-water and underwater reflection cases.
        wpln.d += s_reflWaterClipBias;
        D3DXMATRIX itRefl;
        D3DXMatrixInverse(&itRefl, 0, &reflView);
        D3DXMatrixTranspose(&itRefl, &itRefl);
        D3DXPlaneTransform(&wpln, &wpln, &itRefl);
        effect->SetVector(ehReflWaterClip, reinterpret_cast<const D3DXVECTOR4*>(&wpln));
    }

    // Near-field distance the cache owns in the reflection (Morrowind's view
    // distance by default). Statics, terrain, shadows and lights all hand off to
    // distant land at this distance.
    const float cacheNearDist = Configuration.UseSceneGraphSnapshot
                                ? nearViewRange * s_reflectionCacheNearFactor : 0.0f;

    // Rendering
    if (mwBridge->IsExterior() && (Configuration.MGEFlags & REFLECTIVE_WATER)) {
        // Draw land reflection, with opposite culling. Near-clip the LOD land so the
        // cache draws the real near terrain at full resolution; DL owns beyond.
        effect->SetFloat(ehLandNearCull, cacheNearDist);
        effect->BeginPass(PASS_RENDERLANDREFL);
        device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);
        renderDistantLand(effect, &reflView, &reflProj);
        effect->EndPass();
        effect->SetFloat(ehLandNearCull, 0);   // off for the main-view land later this frame

        // Real near terrain from the cache (two-texture AlphaGrid splat).
        renderReflectionTerrainFromCache(&reflView, &reflProj, cacheNearDist);
    }

    if (isDistantCell() && (Configuration.MGEFlags & REFLECT_NEAR)) {
        // Draw statics reflection, with opposite culling and no dissolve.
        // Per-object handover: distant statics whose object origin is within
        // cacheNearDist are culled whole (staticNearCull) — the cache draws those,
        // lit. DL owns everything beyond. Complementary to the cache's near-distance
        // test (same metric, same threshold) => no duplicates, no z-fight, no slice.
        DWORD p = (mwBridge->CellHasWeather() && !mwBridge->IsUnderwater(eyePos.z)) ? PASS_RENDERSTATICSEXTERIOR : PASS_RENDERSTATICSINTERIOR;
        effect->SetFloat(ehNearViewRange, 0);
        effect->SetFloat(ehStaticNearCull, cacheNearDist);
        effect->BeginPass(p);
        device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);
        renderReflectedStatics(&reflView, &reflProj);
        effect->EndPass();
        effect->SetFloat(ehStaticNearCull, 0);   // off for the main-view statics later this frame
        effect->SetFloat(ehNearViewRange, nearViewRange);
    }

    // Phase 0.5-B/C: inject the GeometryCache near-field set (NPCs + dynamic +
    // static objects within cacheNearDist, excluding terrain) into the reflection
    // with full FFE color, driven from the cache walk. Additive — runs inside the
    // clip-plane setup, after distant statics. renderMorrowind manages its own
    // effectFFE bracket (the distant-land effect is mid-Begin but not in a pass
    // here). No-op when the snapshot cache is empty/disabled.
    renderReflectionsFromCache(&reflView, &reflProj, cacheNearDist);

    // Apply sun shadows to the cache reflection objects + terrain (world-space
    // shadow map, reflected lookup). Runs after the cache color so it darkens the
    // drawn pixels; distance-faded by the receiver's built-in fog attenuation.
    //
    // The receiver gates/scales by surface lit-ness dot(normal, -sunVecView). The
    // cache geometry is rasterized in REFLECTED-view space (normals * reflView), so
    // sunVecView must be the sun in reflected-view space too — otherwise the dot is
    // inconsistent and flat up-facing surfaces (terrain) fail the lit-ness clip and
    // receive no shadow. setupCommonEffect set the main-view sun; swap to the
    // reflected sun for this pass, then restore for the rest of the frame.
    D3DXVECTOR3 sunVecViewRefl, sunVecViewMain;
    D3DXVec3TransformNormal(&sunVecViewRefl, reinterpret_cast<const D3DXVECTOR3*>(&sunVec), &reflView);
    D3DXVec3TransformNormal(&sunVecViewMain, reinterpret_cast<const D3DXVECTOR3*>(&sunVec), view);
    effect->SetFloatArray(ehSunVecView, sunVecViewRefl, 3);
    renderReflectionShadowsFromCache(&reflView, &reflProj, cacheNearDist);
    effect->SetFloatArray(ehSunVecView, sunVecViewMain, 3);   // restore for later passes

    // Restore pass-all so the cache color shader (shared with the main reactive
    // scene) and any later passes don't clip against the water plane.
    const D3DXVECTOR4 reflClipPassAll(0, 0, 0, 1);
    effect->SetVector(ehReflWaterClip, &reflClipPassAll);

    if ((Configuration.MGEFlags & REFLECT_SKY) && !recordSky.empty() && !mwBridge->IsUnderwater(eyePos.z)) {
        // Draw sky reflection, with opposite culling
        renderReflectedSky();
    }

    // Restore view state
    device->SetRenderState(D3DRS_CLIPPLANEENABLE, 0);
    effect->SetMatrix(ehView, view);
    effect->SetMatrix(ehProj, proj);
}

void DistantLand::renderReflectedSky() {
    // Sky objects are not correctly positioned at infinity, so correction is required
    const float adjustZ = -2.0f * eyePos.z;
    D3DXMATRIX skyScale, worldTransform;
    D3DXMatrixScaling(&skyScale, 1e6, 1e6, 1e6);

    // Recorded renders
    const auto& recordSky_const = recordSky;
    const int standardCloudVerts = 65, standardCloudTris = 112;
    const int standardMoonVerts = 4, standardMoonTris = 2;

    // Render sky without clouds first
    effect->BeginPass(PASS_RENDERSKY);
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);

    for (const auto& i : recordSky_const) {
        // Skip clouds
        if (i.texture && i.vertCount == standardCloudVerts && i.primCount == standardCloudTris) {
            continue;
        }

        // Adjust world transform, as skydome objects are positioned relative to the viewer
        worldTransform = i.worldTransforms[0];
        worldTransform._43 += adjustZ;
        if (i.texture == nullptr) {
            // Inflate sky mesh towards infinity, makes skypos in shader calculate correctly
            D3DXMatrixMultiply(&worldTransform, &skyScale, &worldTransform);
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

        effect->SetMatrix(ehWorld, &worldTransform);
        effect->CommitChanges();

        device->SetStreamSource(0, i.vb, i.vbOffset, i.vbStride);
        device->SetIndices(i.ib);
        device->SetFVF(i.fvf);
        DrawStats::count(i.primCount);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);
    }
    effect->EndPass();

    // Render clouds with a separate shader
    effect->BeginPass(PASS_RENDERCLOUDS);
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);

    for (const auto& i : recordSky_const) {
        // Clouds only
        if (!(i.texture && i.vertCount == standardCloudVerts && i.primCount == standardCloudTris)) {
            continue;
        }

        // Adjust world transform, as skydome objects are positioned relative to the viewer
        worldTransform = i.worldTransforms[0];
        worldTransform._43 += adjustZ;

        effect->SetTexture(ehTex0, i.texture);
        effect->SetBool(ehHasAlpha, true);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, 1);
        device->SetRenderState(D3DRS_SRCBLEND, i.srcBlend);
        device->SetRenderState(D3DRS_DESTBLEND, i.destBlend);
        device->SetRenderState(D3DRS_ALPHATESTENABLE, 1);
        effect->SetMatrix(ehWorld, &worldTransform);
        effect->CommitChanges();

        device->SetStreamSource(0, i.vb, i.vbOffset, i.vbStride);
        device->SetIndices(i.ib);
        device->SetFVF(i.fvf);
        DrawStats::count(i.primCount);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);
    }
    effect->EndPass();
}

void DistantLand::renderReflectedStatics(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    // Select appropriate static clipping distance
    D3DXMATRIX ds_proj = *proj, ds_viewproj;
    float zn = 4.0f, zf = Configuration.DL.NearStaticEnd * kCellSize;

    // Don't draw beyond fully fogged distance; early out if frustum is empty
    zf = std::min(fogEnd, zf);
    if (zf <= zn) {
        return;
    }

    // Create a clipping frustum for visibility determination
    editProjectionZ(&ds_proj, zn, zf);
    ds_viewproj = (*view) * ds_proj;

    // Cull sort and draw
    ViewFrustum range_frustum(&ds_viewproj);
    D3DXVECTOR4 viewsphere(eyePos.x, eyePos.y, eyePos.z, zf);

    if (Configuration.UseSharedMemory) {
        // Worker path: the cull worker already issued the reflection RPC,
        // materialized it, and culled it to reflectionSurvivors during the sky
        // window. Just draw the stable survivor set — no RPC, no wait, no IPC
        // window traversal on the main thread.
        if (reflStaticsWanted) {
            device->SetVertexDeclaration(StaticDecl);
            reflectionSurvivors.Render(device, effect, effect, &ehTex0, nullptr, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT);
            return;
        }

        // Inline fallback (worker not dispatched): issue + wait + materialize +
        // cull + draw, all on main. Single-threaded, so the live IPC window is
        // stable across the steps.
        visExtraShared.RemoveAll();
        ipcClient.getVisibleMeshes(visExtraSharedId, range_frustum, viewsphere, VIS_STATIC, VisibleSetSort::ByState);
        ipcClient.waitForCompletion();

        {
            MGE_ZoneScopedN("reflStatics:cull");
            DistantLand::materializeReflectionMeshes();
            // reflectionWaterRects was filled by the inline gate in renderStage0.
            DistantLand::cullReflectionSurvivors(ds_viewproj, ds_proj);
        }

        device->SetVertexDeclaration(StaticDecl);
        reflectionSurvivors.Render(device, effect, effect, &ehTex0, nullptr, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT);
    } else {
        VisibleSet<StlVector> visReflected((StlVector()));

        DistantLandShare::currentWorldSpace->NearStatics->GetVisibleMeshes(range_frustum, viewsphere, visReflected);
        DistantLandShare::currentWorldSpace->FarStatics->GetVisibleMeshes(range_frustum, viewsphere, visReflected);
        DistantLandShare::currentWorldSpace->VeryFarStatics->GetVisibleMeshes(range_frustum, viewsphere, visReflected);
        visReflected.SortByState();

        device->SetVertexDeclaration(StaticDecl);
        visReflected.Render(device, effect, effect, &ehTex0, nullptr, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT);
    }
}

// buildCacheReflectionState - synthesize the RenderedState / FragmentState /
// LightState that FixedFunctionShader::renderMorrowind expects, from a cache
// entry's captured material + the frame-global sun/ambient. Mirrors the
// "standard diffuse texturing" precached variant (single-stage MODULATE,
// uvSet 0, constant material, no vcol/texgen/bump) so the draw hits a
// precompiled shader. Point lights are added inside renderMorrowind from
// MGE::SceneGraph::pointLights() — not synthesized here.
static void buildCacheReflectionState(const MGE::GeometryCache::CachedGeometry& e,
                                      const D3DXMATRIX& view,
                                      const D3DXVECTOR4& sunVec, const RGBVECTOR& sunCol,
                                      const RGBVECTOR& sunAmb, const RGBVECTOR& ambCol,
                                      RenderedState& rs, FragmentState& frs,
                                      LightState& lightrs) {
    // ---- RenderedState ----
    memset(&rs, 0, sizeof(rs));
    // Use vertex colours only when the mesh has them AND its VertexColorProperty
    // says to (else real material colours would be white-washed). vColSource: 1
    // emissive, 2 ambient+diffuse. fvf drives ShaderKey only (the draw uses the
    // device-bound cache FVF); DIFFUSE present -> vertexColour=1.
    const bool useVCol = e.hasVertexColor && e.vColSource != 0;
    rs.fvf            = D3DFVF_XYZ | D3DFVF_NORMAL | D3DFVF_TEX1 | (useVCol ? D3DFVF_DIFFUSE : 0);
    rs.zWrite         = true;
    rs.cullMode       = D3DCULL_CCW;
    rs.useLighting    = true;
    rs.useFog         = true;            // fogMode=1, shared fog constants from the main effect
    rs.blendEnable    = false;
    // vColSource 2 (ambient+diffuse) -> vcol drives diffuse (vertexMaterial 2);
    // 1 (emissive) -> vcol drives emissive (vertexMaterial 3); else constant material.
    rs.matSrcDiffuse  = (useVCol && e.vColSource == 2) ? D3DMCS_COLOR1 : D3DMCS_MATERIAL;
    rs.matSrcEmissive = (useVCol && e.vColSource == 1) ? D3DMCS_COLOR1 : D3DMCS_MATERIAL;
    rs.vertexBlendState = 0;             // non-skinned (0.5-B)

    rs.diffuseMaterial.r = e.matDiffuse[0]; rs.diffuseMaterial.g = e.matDiffuse[1];
    rs.diffuseMaterial.b = e.matDiffuse[2]; rs.diffuseMaterial.a = e.matDiffuse[3];

    rs.viewTransform = view;
    rs.worldTransforms[0] = *reinterpret_cast<const D3DXMATRIX*>(e.worldTransformD3D);
    D3DXMatrixMultiply(&rs.worldViewTransforms[0],
        reinterpret_cast<const D3DXMATRIX*>(e.worldTransformD3D), &view);

    rs.primType = D3DPT_TRIANGLELIST;
    rs.baseIndex = 0; rs.minIndex = 0;
    rs.vertCount = e.vertexCount; rs.startIndex = 0; rs.primCount = e.triangleCount;

    // ---- FragmentState ----
    memset(&frs, 0, sizeof(frs));
    // Single-stage MODULATE(texture, diffuse); stage 1 DISABLE bounds activeStages.
    FragmentState::Stage& s0 = frs.stage[0];
    s0.colorOp   = D3DTOP_MODULATE; s0.colorArg1 = D3DTA_TEXTURE; s0.colorArg2 = D3DTA_DIFFUSE;
    s0.alphaOp   = D3DTOP_MODULATE; s0.alphaArg1 = D3DTA_TEXTURE; s0.alphaArg2 = D3DTA_DIFFUSE;
    s0.colorArg0 = D3DTA_CURRENT;   s0.alphaArg0 = D3DTA_CURRENT; s0.resultArg = D3DTA_CURRENT;
    s0.texcoordIndex = 0;
    frs.stage[1].colorOp = D3DTOP_DISABLE;   // memset 0 == colorOp 0 != DISABLE; set explicitly

    frs.material.diffuse.r  = e.matDiffuse[0];  frs.material.diffuse.g  = e.matDiffuse[1];
    frs.material.diffuse.b  = e.matDiffuse[2];  frs.material.diffuse.a  = e.matDiffuse[3];
    frs.material.ambient.r  = e.matAmbient[0];  frs.material.ambient.g  = e.matAmbient[1];
    frs.material.ambient.b  = e.matAmbient[2];  frs.material.ambient.a  = e.matAmbient[3];
    frs.material.emissive.r = e.matEmissive[0]; frs.material.emissive.g = e.matEmissive[1];
    frs.material.emissive.b = e.matEmissive[2]; frs.material.emissive.a = e.matEmissive[3];

    // ---- LightState: one directional sun; point lights added in renderMorrowind ----
    lightrs.lights.clear();
    lightrs.active.clear();
    lightrs.lightsTransformed.clear();
    lightrs.globalAmbient.r = ambCol.r;
    lightrs.globalAmbient.g = ambCol.g;
    lightrs.globalAmbient.b = ambCol.b;
    lightrs.globalAmbient.a = 1.0f;

    LightState::Light& sun = lightrs.lights[0];
    sun.type = D3DLIGHT_DIRECTIONAL;
    sun.diffuse.r = sunCol.r; sun.diffuse.g = sunCol.g; sun.diffuse.b = sunCol.b; sun.diffuse.a = 1.0f;
    sun.position = D3DVECTOR{ sunVec.x, sunVec.y, sunVec.z };
    sun.ambient  = D3DVECTOR{ sunAmb.r, sunAmb.g, sunAmb.b };
    lightrs.active.push_back(0);
}

void DistantLand::renderReflectionsFromCache(const D3DXMATRIX* view, const D3DXMATRIX* proj, float nearDist) {
    MGE_ZoneScopedN("renderReflectionsFromCache");

    const auto& cacheMap = MGE::GeometryCache::cache();
    if (cacheMap.empty()) return;

    // Reflection frustum cull (reflected camera sees a different set than the
    // main MSOC visible set, so iterate the whole cache and cull here).
    D3DXMATRIX viewproj;
    D3DXMatrixMultiply(&viewproj, view, proj);
    ViewFrustum frustum(&viewproj);

    // Cache covers near objects within nearDist of the camera (Morrowind's view
    // distance by default); distant-land statics dissolve in beyond it. nearDist<=0
    // means "no near limit" (cull by frustum only). Compare on center distance.
    const D3DXVECTOR3 eye(eyePos.x, eyePos.y, eyePos.z);
    const bool haveNearLimit = nearDist > 0.0f;

    // renderMorrowind reads a few live device states; satisfy them once up front.
    // - D3DRS_AMBIENT must not be 0xffffffff (full-bright special case).
    // - D3DTS_PROJECTION feeds the point-light frustum precull.
    device->SetRenderState(D3DRS_AMBIENT, 0);
    device->SetTransform(D3DTS_PROJECTION, proj);
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
    device->SetRenderState(D3DRS_ZENABLE, TRUE);
    device->SetRenderState(D3DRS_ZWRITEENABLE, TRUE);

    RenderedState rs;
    FragmentState frs;
    static LightState lightrs;   // holds maps; reused to avoid per-draw realloc

    for (const auto& kv : cacheMap) {
        const auto& e = kv.second;
        if (e.blendEnable) continue;         // alpha-blended: engine path
        if (!e.d3dTexture) continue;         // untextured: different shader key
        if (e.isLandscape) continue;         // terrain: cache-terrain / distant-land path
        if (e.isSkinned && (e.skinnedUnsupported || e.numBones == 0)) continue;  // unskinnable

        const D3DXVECTOR3 center(e.worldTransformD3D[12], e.worldTransformD3D[13], e.worldTransformD3D[14]);

        // Near-field ownership (handover): static objects within nearDist are owned
        // by the cache (drawn lit here); the distant-land statics pass culls those
        // same objects (staticNearCull = nearDist) and owns everything beyond. Same
        // metric (object-origin distance) + threshold on both sides => exact
        // partition: no duplicates, no z-fight, no slice. Dynamic objects (NPCs)
        // aren't distant statics, so they reflect out to the full MW view distance
        // regardless of the (possibly smaller) static handover factor.
        if (haveNearLimit) {
            const float limit = (e.dynamicHint == 0) ? nearDist : nearViewRange;
            const D3DXVECTOR3 d = center - eye;
            if (D3DXVec3Length(&d) > limit) continue;
        }

        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib) continue;

        // Frustum cull. Non-skinned: true world-space bound (not the object origin)
        // so geometry offset from its node origin isn't falsely rejected at screen
        // edges. Skinned: bone-palette-derived bound at the actual posed location
        // (the bind-pose VB is placed by bones, so the node origin is wrong — see
        // cacheSkinnedWorldBounds; fixes close skinned parts vanishing).
        BoundingSphere bs;
        if (e.isSkinned) { cacheSkinnedWorldBounds(e, bs.center, bs.radius); }
        else             { cacheWorldBounds(e, bs.center, bs.radius); }
        if (frustum.ContainsSphere(bs) == ViewFrustum::OUTSIDE) continue;

        buildCacheReflectionState(e, *view, sunVec, sunCol, sunAmb, ambCol, rs, frs, lightrs);

        // Alpha test (cutout foliage/armor): the FFE shader outputs texture.a *
        // material.a; the FF alpha test then discards. renderMorrowind doesn't set
        // these states, so drive them per part. alphaRef is stored normalized.
        if (e.alphaTest) {
            device->SetRenderState(D3DRS_ALPHATESTENABLE, TRUE);
            device->SetRenderState(D3DRS_ALPHAREF, (DWORD)(e.alphaRef * 255.0f));
            device->SetRenderState(D3DRS_ALPHAFUNC, D3DCMP_GREATEREQUAL);
        } else {
            device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
        }

        // Point-light fade toward the static handover (matches the shadow fade).
        const float plMult = cacheHandoverFade(center, eye, e.dynamicHint, nearDist);

        // Reflection base winding is inverted (CCW); a mirrored part flips again.
        device->SetRenderState(D3DRS_CULLMODE, e.mirrored ? D3DCULL_CW : D3DCULL_CCW);
        device->SetTexture(0, e.d3dTexture);
        device->SetIndices(e.ib);

        if (e.isSkinned) {
            // VS palette skinning: bind-pose VB + per-frame bone palette (model->world);
            // renderMorrowind selects the skinIndexed path and applies reflView.
            device->SetVertexDeclaration(MGE::GeometryCache::skinnedDecl());
            device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kSkinnedVBStride);
            const D3DXMATRIX* pal = reinterpret_cast<const D3DXMATRIX*>(e.bonePalette.data());
            FixedFunctionShader::renderMorrowind(&rs, &frs, &lightrs, plMult, pal, (int)e.numBones, view);
        } else {
            device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kVBStride);
            device->SetFVF(MGE::GeometryCache::kVBFVF);
            FixedFunctionShader::renderMorrowind(&rs, &frs, &lightrs, plMult);
        }
    }
}

void DistantLand::renderReflectionShadowsFromCache(const D3DXMATRIX* view, const D3DXMATRIX* proj, float nearDist) {
    MGE_ZoneScopedN("renderReflectionShadowsFromCache");

    const auto& cacheMap = MGE::GeometryCache::cache();
    if (cacheMap.empty()) return;

    // Sun shadow map is world-space. The cache objects are rasterized in reflected
    // view space, so map reflected-view -> world (inverse reflView) -> shadow clip.
    D3DXMATRIX invView, viewToShadow[2];
    D3DXMatrixInverse(&invView, nullptr, view);
    viewToShadow[0] = invView * smViewproj[0];
    viewToShadow[1] = invView * smViewproj[1];
    effect->SetMatrixArray(ehShadowViewproj, viewToShadow, 2);
    effect->SetTexture(ehTex3, texSoftShadow);

    D3DXMATRIX viewproj;
    D3DXMatrixMultiply(&viewproj, view, proj);
    ViewFrustum frustum(&viewproj);
    const D3DXVECTOR3 eye(eyePos.x, eyePos.y, eyePos.z);

    effect->SetBool(ehHasBones, false);
    effect->SetInt(ehVertexBlendState, 0);
    effect->SetFloat(ehMaterialAlpha, 1.0f);
    // Receiver alpha = materialAlpha (1), NOT vertex colour. Critical for terrain:
    // its vcol.a is the AlphaGrid splat factor, which must not modulate shadow
    // strength (would erase shadows wherever the base texture shows through).
    effect->SetBool(ehHasVCol, false);

    // P2ffe — shadows over FFE output (our cache color was drawn via the FFE path).
    // Same object set/cull as the color pass so the receiver lands on the drawn
    // pixels (ZFunc LessEqual against the reflection depth the color pass wrote).
    effect->BeginPass(PASS_RENDERSHADOWFFE);
    for (const auto& kv : cacheMap) {
        const auto& e = kv.second;
        if (e.isSkinned) continue;          // skinned receiver = skinIndexed (0.5-C)
        if (e.blendEnable) continue;
        // Terrain IS included here: the cache near terrain receives sun shadows the
        // same way the cache objects do (additive receiver over the FFE/terrain color
        // that already wrote depth). The distant-land LOD beyond the handover has no
        // shadows, so the fade ramps cache-terrain shadows to 0 at the boundary.
        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib) continue;

        // True world-space bound for the frustum cull (origin is wrong for offset
        // geometry / terrain patches — see cacheWorldBounds).
        BoundingSphere bs;
        cacheWorldBounds(e, bs.center, bs.radius);
        if (frustum.ContainsSphere(bs) == ViewFrustum::OUTSIDE) continue;

        // Handover metric: statics match the distant-land staticNearCull (object
        // origin) for an exact partition; terrain uses its world-space patch center
        // (origin = cell corner) and counts a straddling patch as in-range.
        const D3DXVECTOR3 origin(e.worldTransformD3D[12], e.worldTransformD3D[13], e.worldTransformD3D[14]);
        const D3DXVECTOR3 metric = e.isLandscape ? bs.center : origin;
        if (nearDist > 0.0f) {
            const float limit = (e.dynamicHint == 0) ? nearDist : nearViewRange;
            const float reach = e.isLandscape ? bs.radius : 0.0f;
            const D3DXVECTOR3 d = metric - eye;
            if (D3DXVec3Length(&d) - reach > limit) continue;
        }

        D3DXMATRIX wv;
        D3DXMatrixMultiply(&wv, reinterpret_cast<const D3DXMATRIX*>(e.worldTransformD3D), view);
        D3DXMATRIX pal[4] = { wv, wv, wv, wv };
        effect->SetMatrixArray(ehVertexBlendPalette, pal, 4);

        // Handover fade, identical to the point-light fade so shadows + lights ramp
        // together and land exactly on the geometry handover.
        effect->SetFloat(ehShadowReflMult, cacheHandoverFade(metric, eye, e.dynamicHint, nearDist));

        // Alpha-tested receivers need the base texture for the cutout clip.
        if (e.alphaTest && e.d3dTexture) {
            effect->SetTexture(ehTex0, e.d3dTexture);
            effect->SetBool(ehHasAlpha, true);
            effect->SetFloat(ehAlphaRef, e.alphaRef);
        } else {
            effect->SetTexture(ehTex0, nullptr);
            effect->SetBool(ehHasAlpha, false);
            effect->SetFloat(ehAlphaRef, -1.0f);
        }
        effect->CommitChanges();

        device->SetRenderState(D3DRS_CULLMODE, e.mirrored ? D3DCULL_CW : D3DCULL_CCW);
        device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kVBStride);
        device->SetIndices(e.ib);
        device->SetFVF(MGE::GeometryCache::kVBFVF);
        DrawStats::count(e.triangleCount);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, e.vertexCount, 0, e.triangleCount);
    }
    effect->EndPass();

    // Cache-skinned receivers (NPCs). Same world-space shadow lookup, but the
    // receiver depth is rebuilt in the VS via 32-bone skinIndexed -> mul(view),
    // bit-identical to the cache color pass (cacheSkinnedVertex), so it lands on the
    // drawn pixels without acne. Separate pass = dedicated skinned VS + skinnedDecl
    // stream + bone palette. view (reflView) and hasVCol=false are already bound.
    effect->BeginPass(PASS_RENDERSHADOWFFE_SKINNED);
    for (const auto& kv : cacheMap) {
        const auto& e = kv.second;
        if (!e.isSkinned) continue;
        if (e.skinnedUnsupported || e.numBones == 0) continue;
        if (e.blendEnable) continue;
        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib) continue;

        // Skinned cull: bone-palette-derived bound at the actual posed location
        // (matches the color pass; the node origin is wrong for bind-pose skinned VBs).
        BoundingSphere bs;
        cacheSkinnedWorldBounds(e, bs.center, bs.radius);
        if (frustum.ContainsSphere(bs) == ViewFrustum::OUTSIDE) continue;

        // Handover fade (NPCs are dynamic => full strength, but compute generally).
        effect->SetFloat(ehShadowReflMult, cacheHandoverFade(bs.center, eye, e.dynamicHint, nearDist));

        // Bone palette (model->world); view (reflView) already bound on the effect.
        const D3DXMATRIX* pal = reinterpret_cast<const D3DXMATRIX*>(e.bonePalette.data());
        effect->SetMatrixArray(ehBoneMatrices, pal, (int)e.numBones);

        if (e.alphaTest && e.d3dTexture) {
            effect->SetTexture(ehTex0, e.d3dTexture);
            effect->SetBool(ehHasAlpha, true);
            effect->SetFloat(ehAlphaRef, e.alphaRef);
        } else {
            effect->SetTexture(ehTex0, nullptr);
            effect->SetBool(ehHasAlpha, false);
            effect->SetFloat(ehAlphaRef, -1.0f);
        }
        effect->CommitChanges();

        device->SetRenderState(D3DRS_CULLMODE, e.mirrored ? D3DCULL_CW : D3DCULL_CCW);
        device->SetVertexDeclaration(MGE::GeometryCache::skinnedDecl());
        device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kSkinnedVBStride);
        device->SetIndices(e.ib);
        DrawStats::count(e.triangleCount);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, e.vertexCount, 0, e.triangleCount);
    }
    effect->EndPass();

    effect->SetFloat(ehShadowReflMult, 1.0f);   // restore full strength for later passes
}

void DistantLand::renderReflectionTerrainFromCache(const D3DXMATRIX* view, const D3DXMATRIX* proj, float nearDist) {
    MGE_ZoneScopedN("renderReflectionTerrainFromCache");
    if (nearDist <= 0.0f) return;

    const auto& cacheMap = MGE::GeometryCache::cache();
    if (cacheMap.empty()) return;

    D3DXMATRIX viewproj;
    D3DXMatrixMultiply(&viewproj, view, proj);
    ViewFrustum frustum(&viewproj);
    const D3DXVECTOR3 eye(eyePos.x, eyePos.y, eyePos.z);

    effect->BeginPass(PASS_RENDERCACHETERRAIN);
    for (const auto& kv : cacheMap) {
        const auto& e = kv.second;
        if (!e.isLandscape || e.isSkinned) continue;
        if (!e.d3dTexture) continue;          // need at least a base texture
        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib) continue;

        // Terrain patches are large; cull by the true world-space bound sphere
        // (the patch origin is the cell corner, ~4096u from the real center — using
        // it falsely culls the patch the camera is standing on). Skip patches whose
        // bound is wholly beyond the handover (the DL LOD owns those).
        BoundingSphere bs;
        cacheWorldBounds(e, bs.center, bs.radius);
        if (frustum.ContainsSphere(bs) == ViewFrustum::OUTSIDE) continue;
        const D3DXVECTOR3 d = bs.center - eye;
        if (D3DXVec3Length(&d) - bs.radius > nearDist) continue;

        effect->SetMatrix(ehWorld, reinterpret_cast<const D3DXMATRIX*>(e.worldTransformD3D));
        effect->SetTexture(ehTex0, e.d3dTexture);
        // No overlay -> bind base to tex2 so lerp(base, base, a) == base.
        effect->SetTexture(ehTex2, e.d3dOverlay ? e.d3dOverlay : e.d3dTexture);
        // Position is computed from vertexBlendPalette[0] (= world*view), the exact
        // same premultiplied matrix the shadow receiver uses, so the LessEqual shadow
        // pass lands on the same depth (no acne). Matches the object color/shadow path.
        D3DXMATRIX wv;
        D3DXMatrixMultiply(&wv, reinterpret_cast<const D3DXMATRIX*>(e.worldTransformD3D), view);
        effect->SetMatrixArray(ehVertexBlendPalette, &wv, 1);
        effect->CommitChanges();

        device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kVBStride);
        device->SetIndices(e.ib);
        device->SetFVF(MGE::GeometryCache::kVBFVF);
        DrawStats::count(e.triangleCount);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, e.vertexCount, 0, e.triangleCount);
    }
    effect->EndPass();
}

void DistantLand::clearReflection() {
    auto mwBridge = MWBridge::get();
    IDirect3DSurface9* target;
    DWORD baseColour;

    texReflection->GetSurfaceLevel(0, &target);
    if (mwBridge->CellHasWeather() || mwBridge->IsUnderwater(eyePos.z)) {
        // Use fog colour as reflection
        baseColour = (DWORD)horizonCol;
    } else {
        // Interior fog colour is typically too bright
        // Guess a reflection colour based on cell lighting parameters
        const BYTE* sun = mwBridge->getInteriorSun();
        RGBVECTOR c(sun[0] / 255.0f, sun[1] / 255.0f, sun[2] / 255.0f);
        c += ambCol;
        baseColour = (DWORD)c;
    }
    device->ColorFill(target, 0, baseColour);
    target->Release();
}

void DistantLand::simulateDynamicWaves() {
    MGE_ZoneScopedN("simulateDynamicWaves");
    DrawStats::ScopedStage _ds(DrawStats::Water);
    auto mwBridge = MWBridge::get();

    static bool resetRippleSurface = true;
    static float remainingWaveTime = 0;
    static const float waveStep = 0.0125f;  // time per wave simulation step (1/80 sec)

    // Simulation paused in menu mode
    if (mwBridge->IsMenu()) {
        return;
    }

    device->SetFVF(fvfWave);
    device->SetStreamSource(0, vbWaveSim, 0, 32);

    // Calc number of wave iterations to run this frame
    float frameTime = std::min(mwBridge->frameTime(), 0.5f);
    remainingWaveTime += frameTime;
    int numWaveSteps = (int)(remainingWaveTime / waveStep);
    remainingWaveTime -= numWaveSteps * waveStep;

    // Preciptation (rain/snow) ripples
    if (mwBridge->CellHasWeather()) {
        static float remainingRipples = 0;

        // Reset surface when not needed next time
        resetRippleSurface = true;

        // Weather types: rain = 4; thunderstorm = 5; snow = 8; blizzard = 9
        // Thunderstorm causes 50% more ripples
        int w0 = mwBridge->GetCurrentWeather(), w1 = mwBridge->GetNextWeather();
        float precipitation0 = (w0 == 4 || w0 == 5 || w0 == 8 || w0 == 9) ? 1.0f : -1.5f;
        float precipitation1 = (w1 == 4 || w1 == 5 || w1 == 8 || w1 == 9) ? 1.0f : -1.5f;
        precipitation0 += (w0 == 5) ? 0.5f : 0;
        precipitation1 += (w1 == 5) ? 0.5f : 0;

        // 150 drops per second for normal precipitation
        float precipitation = (1.0f - mwBridge->GetWeatherRatio()) * precipitation0 + mwBridge->GetWeatherRatio() * precipitation1;
        float rippleFrequency = 150.0f * precipitation;

        if (rippleFrequency > 0) {
            static double randomizer = 0.546372819;
            int ripplePos[2];
            RECT drop;

            remainingRipples += rippleFrequency * frameTime;
            int n = int(std::floor(remainingRipples));
            remainingRipples -= n;

            while (n-- > 0) {
                // Place rain ripple at random location
                for (int i = 0; i != 2; ++i) {
                    randomizer = randomizer * (1337.134511337451 + 0.0001 * rand()) + 0.12351523;
                    randomizer -= floor(randomizer);
                    ripplePos[i] = (int)(randomizer * waveTexResolution);
                }

                drop.left = ripplePos[0] - 2;
                drop.right = ripplePos[0] + 2;
                drop.top = ripplePos[1] - 1;
                drop.bottom = ripplePos[1] + 1;
                device->ColorFill(surfRain, &drop, 0x6060);

                drop.left = ripplePos[0] - 1;
                drop.right = ripplePos[0] + 1;
                drop.top = ripplePos[1] - 2;
                drop.bottom = ripplePos[1] + 2;
                device->ColorFill(surfRain, &drop, 0x6060);

                drop.left = ripplePos[0] - 1;
                drop.right = ripplePos[0] + 1;
                drop.top = ripplePos[1] - 1;
                drop.bottom = ripplePos[1] + 1;
                device->ColorFill(surfRain, &drop, 0x4040);
            }
        }

        // Apply wave equation numWaveSteps times
        // Uses double buffering to avoid reads and writes on the same target
        RenderTargetSwitcher rtsw(surfRippleBuffer, NULL);
        SurfaceDoubleBuffer doublebuffer;
        doublebuffer.init(texRain, surfRain, texRippleBuffer, surfRippleBuffer);

        effect->BeginPass(PASS_WAVESTEP);
        for (int i = 0; i != numWaveSteps; ++i) {
            device->SetRenderTarget(0, doublebuffer.sinkSurface());
            effect->SetTexture(ehTex4, doublebuffer.sourceTexture());
            effect->CommitChanges();

            DrawStats::count(1);
            device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 1);
            doublebuffer.cycle();
        }
        effect->EndPass();

        if (doublebuffer.sourceSurface() != surfRain) {
            device->StretchRect(surfRippleBuffer, 0, surfRain, 0, D3DTEXF_NONE);
        }
    } else if (resetRippleSurface) {
        // No weather - clear rain ripples
        device->ColorFill(surfRain, NULL, 0);
        resetRippleSurface = false;
    }

    // Player local ripples
    // Move ripple texture with player; lock to texel alignment to prevent visible jitter
    const D3DXVECTOR3* playerPos = (const D3DXVECTOR3*)mwBridge->PlayerPositionPointer();
    static int lastXpos = (int)floor(playerPos->x / waveTexWorldRes);
    static int lastYpos = (int)floor(playerPos->y / waveTexWorldRes);

    int newXpos = (int)floor(playerPos->x / waveTexWorldRes);
    int newYpos = (int)floor(playerPos->y / waveTexWorldRes);

    int shiftX = newXpos - lastXpos;
    int shiftY = newYpos - lastYpos;

    lastXpos = newXpos;
    lastYpos = newYpos;

    int shiftXp = (shiftX > 0) ? +shiftX : 0;
    int shiftXn = (shiftX < 0) ? -shiftX : 0;
    int shiftYp = (shiftY > 0) ? +shiftY : 0;
    int shiftYn = (shiftY < 0) ? -shiftY : 0;

    // Shift texture by (shiftX, shiftY) pixels
    RECT source;
    source.left = 1 + shiftXp;
    source.right = waveTexResolution - shiftXn;
    source.top = 1 + shiftYp;
    source.bottom = waveTexResolution - shiftYn;

    RECT target;
    target.left = 1 + shiftXn;
    target.right = waveTexResolution - shiftXp;
    target.top = 1 + shiftYn;
    target.bottom = waveTexResolution - shiftYp;

    device->ColorFill(surfRippleBuffer, 0, 0);
    device->StretchRect(surfRipples, &source, surfRippleBuffer, &target, D3DTEXF_NONE);

    // Water simulation; realigned water starts in surfRippleBuffer
    // Uses double buffering to avoid reads and writes on the same target
    RenderTargetSwitcher rtsw(surfRipples, NULL);
    SurfaceDoubleBuffer doublebuffer;
    doublebuffer.init(texRippleBuffer, surfRippleBuffer, texRipples, surfRipples);

    float rippleOrigin[2];
    float dz = playerPos->z - mwBridge->WaterLevel();
    if (dz < 0 && dz > -128.0f * mwBridge->PlayerHeight()) {
        // Create waves around the player
        effect->BeginPass(PASS_PLAYERWAVE);
        for (int i = 0; i != numWaveSteps; ++i) {
            // Interpolate between starting and ending point, so that low framerates do not cause less waves
            float w = -(float)i / (float)numWaveSteps / (float)waveTexResolution;
            rippleOrigin[0] = w * shiftX + 0.5f;
            rippleOrigin[1] = w * shiftY + 0.5f;

            device->SetRenderTarget(0, doublebuffer.sinkSurface());
            effect->SetTexture(ehTex4, doublebuffer.sourceTexture());
            effect->SetFloatArray(ehRippleOrigin, rippleOrigin, 2);
            effect->CommitChanges();

            DrawStats::count(1);
            device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 1);
            doublebuffer.cycle();
        }
        effect->EndPass();
    }

    // Apply wave equation numWaveSteps times
    effect->BeginPass(PASS_WAVESTEP);
    for (int i = 0; i != numWaveSteps; ++i) {
        device->SetRenderTarget(0, doublebuffer.sinkSurface());
        effect->SetTexture(ehTex4, doublebuffer.sourceTexture());
        effect->CommitChanges();

        DrawStats::count(1);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 1);
        doublebuffer.cycle();
    }
    effect->EndPass();

    if (doublebuffer.sourceSurface() != surfRipples) {
        device->StretchRect(surfRippleBuffer, 0, surfRipples, 0, D3DTEXF_NONE);
    }

    // Set wave texture world origin
    static float halfWaveTexWorldSize = 0.5f * waveTexWorldRes * waveTexResolution;
    rippleOrigin[0] = lastXpos*waveTexWorldRes - halfWaveTexWorldSize;
    rippleOrigin[1] = lastYpos*waveTexWorldRes - halfWaveTexWorldSize;
    effect->SetFloatArray(ehRippleOrigin, rippleOrigin, 2);

    // Set weather-dependent wave height
    effect->SetFloat(ehWaveHeight, (float)Configuration.DL.WaterWaveHeight);
}

void DistantLand::renderWaterPlane() {
    DrawStats::ScopedStage _ds(DrawStats::Water);
    D3DXMATRIX m;
    IDirect3DTexture9* texRefract = PostShaders::borrowBuffer(0);

    // Snap the surface height by underwater state so it never intersects the camera
    // at the transition (see s_waterMeshSnap*). The snap lands at the IsUnderwater
    // threshold where the camera is at the surface, hiding the jump.
    const bool underwater = MWBridge::get()->IsUnderwater(eyePos.z);
    const float waterZ = MWBridge::get()->WaterLevel()
                       + (underwater ? s_waterMeshSnapUnderwater : s_waterMeshSnapAbove);
    D3DXMatrixTranslation(&m, eyePos.x, eyePos.y, waterZ);
    effect->SetMatrix(ehWorld, &m);
    effect->SetTexture(ehTex0, texReflection);
    effect->SetTexture(ehTex1, texWater);
    effect->SetTexture(ehTex2, texRefract);
    effect->SetTexture(ehTex3, texDepthFrame);
    if (Configuration.MGEFlags & DYNAMIC_RIPPLES) {
        effect->SetTexture(ehTex4, texRain);
        effect->SetTexture(ehTex5, texRipples);
    }
    effect->CommitChanges();

    device->SetVertexDeclaration(WaterDecl);
    device->SetStreamSource(0, vbWater, 0, 12);
    device->SetIndices(ibWater);
    DrawStats::count(numWaterTris);
    device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, numWaterVerts, 0, numWaterTris);
}
