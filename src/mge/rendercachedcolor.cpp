// rendercachedcolor.cpp — cache-driven color (the renderer-takeover path).
//
// Owns the machinery that renders the MGE GeometryCache snapshot with full FFE
// color, shared by two consumers:
//   - the water reflection injection (renderReflectionsFromCache / shadows /
//     terrain), called from renderWaterReflection in renderwater.cpp;
//   - the main-scene opaque pass (renderCachedOpaque), called from renderStage0.
//
// The synthetic-state builders (buildCacheReflectionState / buildCacheMainState)
// turn a cache entry's captured material + the frame-global sun/ambient into the
// RenderedState / FragmentState / LightState that FixedFunctionShader::
// renderMorrowind expects, and the bounds helpers (cacheWorldBounds /
// cacheSkinnedWorldBounds) provide world-space cull spheres. Kept together here so
// renderwater.cpp stays water-only and renderStage0 stays orchestration-only.

#include "distantland.h"
#include "distantshader.h"
#include "drawstats.h"
#include "configuration.h"
#include "ffeshader.h"
#include "scenegraph_geometry_cache.h"
#include "support/log.h"
#include "mge_tracy.h"

#include <algorithm>
#include <cstdint>
#include <vector>



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

        // World-space AABB for texture-light selection. The cache VB is WRITEONLY so
        // renderMorrowind's computeBoundingBox can't read it and would fall back to the
        // object origin (selecting lights as if the whole part were a point at its
        // node origin -> wrong lights). Hand it the real bound, same as renderCachedOpaque.
        const D3DXVECTOR3 lbMin(bs.center.x - bs.radius, bs.center.y - bs.radius, bs.center.z - bs.radius);
        const D3DXVECTOR3 lbMax(bs.center.x + bs.radius, bs.center.y + bs.radius, bs.center.z + bs.radius);

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
            FixedFunctionShader::renderMorrowind(&rs, &frs, &lightrs, plMult, pal, (int)e.numBones, view, &lbMin, &lbMax);
        } else {
            device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kVBStride);
            device->SetFVF(MGE::GeometryCache::kVBFVF);
            FixedFunctionShader::renderMorrowind(&rs, &frs, &lightrs, plMult, nullptr, 0, nullptr, &lbMin, &lbMax);
        }
    }
}

// buildCacheMainState - main-view sibling of buildCacheReflectionState. The
// reflection synthetic state is already view-agnostic (it takes view/sun as
// params and never touches the clip plane or below-water bias — those live in
// the renderWaterReflection wrapper), so we delegate to it and override only the
// documented base winding: the main view culls CW where the reflection culls CCW
// (the reflection inverts winding once). cullMode is cosmetic (renderMorrowind
// drives the device cull per-part), but set it to keep rs self-describing.
static void buildCacheMainState(const MGE::GeometryCache::CachedGeometry& e,
                                const D3DXMATRIX& view,
                                const D3DXVECTOR4& sunVec, const RGBVECTOR& sunCol,
                                const RGBVECTOR& sunAmb, const RGBVECTOR& ambCol,
                                RenderedState& rs, FragmentState& frs,
                                LightState& lightrs) {
    buildCacheReflectionState(e, view, sunVec, sunCol, sunAmb, ambCol, rs, frs, lightrs);
    rs.cullMode = D3DCULL_CW;
}

void DistantLand::renderCachedOpaque(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    MGE_ZoneScopedN("renderCachedOpaque");

    const auto& cacheMap = MGE::GeometryCache::cache();
    if (cacheMap.empty()) return;

    // Main-view projection feeds the FFE shader (shared pool param `proj`) and the
    // point-light precull (D3DTS_PROJECTION). At the call site ehProj holds the
    // far-extended distant projection; the caller restores it after this returns.
    effect->SetMatrix(ehView, view);
    effect->SetMatrix(ehProj, proj);

    // No water clip in the main view: force the shared per-pixel clip plane to
    // pass-all (the reflection leaves it pass-all at its end, but be self-contained
    // for the first frame / water-less cells). dot(viewpos.xyzw, (0,0,0,1)) = 1 >= 0.
    const D3DXVECTOR4 clipPassAll(0, 0, 0, 1);
    effect->SetVector(ehReflWaterClip, &clipPassAll);

    // renderMorrowind reads a few live device states; satisfy them once up front.
    // - D3DRS_AMBIENT must not be 0xffffffff (the full-bright special case).
    // - D3DTS_PROJECTION feeds the point-light frustum precull.
    // - Own the depth: the MGE depth pre-pass writes a separate RT, so the real
    //   z-buffer here holds only sky + distant land (pushed to the far range by the
    //   distant projection bias). ZWRITE on + ZFUNC LESSEQUAL lets the near scene
    //   draw in front, exactly as the engine's reactive near pass does.
    device->SetRenderState(D3DRS_AMBIENT, 0);
    device->SetTransform(D3DTS_PROJECTION, proj);
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
    device->SetRenderState(D3DRS_ZENABLE, TRUE);
    device->SetRenderState(D3DRS_ZWRITEENABLE, TRUE);
    device->SetRenderState(D3DRS_ZFUNC, D3DCMP_LESSEQUAL);

    // Iterate the engine-submitted on-screen visible set (s_prevVisibleKeys) — the
    // SAME set the depth pass walks — so cache color stays in lockstep with depth
    // and the reactive engine. An independent per-part frustum cull (as in the
    // reflection, which needs it for the different reflected camera) rejected close
    // parts the engine still draws (e.g. a hand reaching toward the camera), making
    // them vanish in CACHE mode while depth kept them. The visible set IS the main-
    // camera cull, so no extra frustum test is needed; only the empty-set fallback
    // (no MSOC verdict yet) frustum-culls the full cache to avoid drawing offscreen.
    const auto& visKeys = visibleCacheKeys();
    const bool useVisibleSet = !visKeys.empty();

    D3DXMATRIX viewproj;
    D3DXMatrixMultiply(&viewproj, view, proj);
    ViewFrustum frustum(&viewproj);

    // Diagnostics (gated on LogDistantPipeline): per-interval skip accounting so we
    // can see which covered-but-skipped parts the engine suppression would drop.
    // skinnedUnsupported (>kMaxBones armor like full-skeleton belts) is the prime
    // suspect for "missing on multiple NPCs"; log a sample texture name.
    const bool logPerf = Configuration.LogDistantPipeline;
    static unsigned s_drawn = 0, s_skBlend = 0, s_skNoTex = 0, s_skLand = 0, s_skUnskin = 0, s_skFrustum = 0;
    static const char* s_lastUnskinTex = nullptr;
    static unsigned s_calls = 0;

    RenderedState rs;
    FragmentState frs;
    static LightState lightrs;   // holds maps; reused to avoid per-draw realloc

    auto drawEntry = [&](const MGE::GeometryCache::CachedGeometry& e) {
        if (e.blendEnable)  { if (logPerf) ++s_skBlend; return; }   // alpha-blended: engine path
        if (!e.d3dTexture)  { if (logPerf) ++s_skNoTex; return; }   // untextured: different shader key
        if (e.isLandscape)  { if (logPerf) ++s_skLand;  return; }   // terrain: stays engine-drawn this phase
        if (e.isSkinned && (e.skinnedUnsupported || e.numBones == 0)) {
            if (logPerf) { ++s_skUnskin; if (e.skinnedUnsupported) s_lastUnskinTex = e.textureName; }
            return;                                                  // >kMaxBones / zero-bone: no VS palette
        }

        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib) return;

        // World-space bound: skinned = bone-palette-derived posed bound; non-skinned
        // = true world bound (not the object origin). Used for the fallback frustum
        // cull AND the texture-light selection AABB below (renderMorrowind can't read
        // our WRITEONLY VB via computeBoundingBox, so we hand it these bounds — the
        // object light-seam fix, keeping cache light selection identical to reactive).
        BoundingSphere bs;
        if (e.isSkinned) { cacheSkinnedWorldBounds(e, bs.center, bs.radius); }
        else             { cacheWorldBounds(e, bs.center, bs.radius); }

        // Only the full-cache fallback frustum-culls (the visible set is already
        // the main-camera cull).
        if (!useVisibleSet) {
            if (frustum.ContainsSphere(bs) == ViewFrustum::OUTSIDE) { if (logPerf) ++s_skFrustum; return; }
        }

        const D3DXVECTOR3 lbMin(bs.center.x - bs.radius, bs.center.y - bs.radius, bs.center.z - bs.radius);
        const D3DXVECTOR3 lbMax(bs.center.x + bs.radius, bs.center.y + bs.radius, bs.center.z + bs.radius);

        buildCacheMainState(e, *view, sunVec, sunCol, sunAmb, ambCol, rs, frs, lightrs);

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

        // Winding ground truth (renderdepth.cpp:256): this snapshot rasterizes in
        // game-view space with `mirrored ? CCW : CW`. The main opaque pass matches:
        // CW normal, CCW for mirrored (negative-determinant) left-side parts.
        device->SetRenderState(D3DRS_CULLMODE, e.mirrored ? D3DCULL_CCW : D3DCULL_CW);
        device->SetTexture(0, e.d3dTexture);
        device->SetIndices(e.ib);

        if (e.isSkinned) {
            // VS palette skinning: bind-pose VB + per-frame bone palette (model->world);
            // renderMorrowind selects the skinIndexed path and applies the main view.
            device->SetVertexDeclaration(MGE::GeometryCache::skinnedDecl());
            device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kSkinnedVBStride);
            const D3DXMATRIX* pal = reinterpret_cast<const D3DXMATRIX*>(e.bonePalette.data());
            FixedFunctionShader::renderMorrowind(&rs, &frs, &lightrs, 1.0f, pal, (int)e.numBones, view, &lbMin, &lbMax);
        } else {
            device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kVBStride);
            device->SetFVF(MGE::GeometryCache::kVBFVF);
            FixedFunctionShader::renderMorrowind(&rs, &frs, &lightrs, 1.0f, nullptr, 0, nullptr, &lbMin, &lbMax);
        }
        if (logPerf) ++s_drawn;
    };

    if (useVisibleSet) {
        for (uint32_t key : visKeys) {
            auto it = cacheMap.find(key);
            if (it != cacheMap.end()) drawEntry(it->second);
        }
    } else {
        for (const auto& kv : cacheMap) drawEntry(kv.second);
    }

    if (logPerf && (++s_calls % 300 == 0)) {
        LOG::logline("-- [CACHE OPAQUE] drawn=%u skip{blend=%u notex=%u land=%u unskin=%u frustum=%u} visSet=%d sampleUnskinTex=%s",
                     s_drawn, s_skBlend, s_skNoTex, s_skLand, s_skUnskin, s_skFrustum,
                     useVisibleSet ? 1 : 0, s_lastUnskinTex ? s_lastUnskinTex : "(none)");
        s_drawn = s_skBlend = s_skNoTex = s_skLand = s_skUnskin = s_skFrustum = 0;
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

    // Texture-light setup, mirroring renderCachedTerrain so reflected terrain gets the
    // same dynamic point lights as reflected objects (previously it used the non-lit
    // CacheTerrainPS and stayed sun+vcol only). texLightView = view (= reflView) so the
    // shader transforms world-space light positions into the reflected-view space its
    // pixels live in. D3DTS_PROJECTION drives selectTextureLights' frustum precull.
    const bool logPerf = Configuration.LogDistantPipeline;
    const float texelSize = FixedFunctionShader::texLightTexelSize();
    effect->SetTexture(ehLightData, FixedFunctionShader::textureLightData());
    effect->SetMatrix(ehTexLightView, view);
    device->SetTransform(D3DTS_PROJECTION, proj);
    float idxFloats[8 * 4] = { 0 };

    MGE::SceneGraph::SnapshotReadLock snapshotLock;
    const auto& snapshotLights = MGE::SceneGraph::pointLights();
    const unsigned int snapshotCount =
        std::min<unsigned int>((unsigned int)snapshotLights.size(), FixedFunctionShader::maxTexLights());

    effect->BeginPass(PASS_RENDERCACHETERRAINREFLLIT);
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

        // Per-patch point-light selection (sphere-AABB nearest-32), byte-identical to
        // the object/main-terrain path. AABB from the patch's world bound sphere.
        const D3DXVECTOR3 lbMin(bs.center.x - bs.radius, bs.center.y - bs.radius, bs.center.z - bs.radius);
        const D3DXVECTOR3 lbMax(bs.center.x + bs.radius, bs.center.y + bs.radius, bs.center.z + bs.radius);
        const int lightCount = FixedFunctionShader::selectTextureLights(
            snapshotLights, snapshotCount, *view, lbMin, lbMax, idxFloats, logPerf);
        const D3DXVECTOR4 lightDataParams_v((float)lightCount, texelSize, 0.0f, 0.0f);
        effect->SetVector(ehLightDataParams, &lightDataParams_v);
        if (lightCount > 0) effect->SetVectorArray(ehLightIndices, (D3DXVECTOR4*)idxFloats, 8);

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

// renderCachedTerrain - main-view sibling of renderReflectionTerrainFromCache.
// Draws the real near terrain (the engine's submitted landscape patches, two-texture
// AlphaGrid splat) from the cache into the MAIN view so MGE owns ALL opaque
// (objects + terrain) in CACHE mode. Lit with sun + vcol AND dynamic point lights
// via the SHARED FixedFunctionShader::selectTextureLights (per-patch, byte-identical
// to the object path), so cache terrain matches the reactive PPL terrain lighting
// (the A/B light-seam fix). Uses PASS_RENDERCACHETERRAINLIT (CacheTerrainLitVS/PS);
// the reflection keeps PASS_RENDERCACHETERRAIN untouched. Differences from the
// reflection version: no near-dist handover (cache landscape IS the engine's near
// terrain; DL LOD owns beyond via the distant projection), main-view CW winding, no
// water clip.
void DistantLand::renderCachedTerrain(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    MGE_ZoneScopedN("renderCachedTerrain");

    const auto& cacheMap = MGE::GeometryCache::cache();
    if (cacheMap.empty()) return;

    // Main-view projection for the shared `proj` pool param; caller restores distProj.
    effect->SetMatrix(ehProj, proj);
    // D3DTS_PROJECTION drives selectTextureLights' frustum precull (renderCachedOpaque
    // already set it to mwProj, but be self-contained for the object-less case).
    device->SetTransform(D3DTS_PROJECTION, proj);

    D3DXMATRIX viewproj;
    D3DXMatrixMultiply(&viewproj, view, proj);
    ViewFrustum frustum(&viewproj);

    // Texture-light setup: bind the shared light-data texture once; per patch we run
    // the same selection the object path uses and push lightIndices/lightDataParams/
    // texLightView. texLightView = view (world->view) so the shader transforms the
    // texture's world-space light positions into the terrain's view space.
    const bool logPerf = Configuration.LogDistantPipeline;
    const float texelSize = FixedFunctionShader::texLightTexelSize();
    effect->SetTexture(ehLightData, FixedFunctionShader::textureLightData());
    effect->SetMatrix(ehTexLightView, view);
    float idxFloats[8 * 4] = { 0 };

    // Hold the snapshot lock across the patch loop: selectTextureLights reads the
    // point-light snapshot (and, on a revision bump, uploads texLightData) under it.
    MGE::SceneGraph::SnapshotReadLock snapshotLock;
    const auto& snapshotLights = MGE::SceneGraph::pointLights();
    const unsigned int snapshotCount =
        std::min<unsigned int>((unsigned int)snapshotLights.size(), FixedFunctionShader::maxTexLights());

    effect->BeginPass(PASS_RENDERCACHETERRAINLIT);
    for (const auto& kv : cacheMap) {
        const auto& e = kv.second;
        if (!e.isLandscape || e.isSkinned) continue;
        if (!e.d3dTexture) continue;          // need at least a base texture
        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib) continue;

        // Cull by the true world-space bound sphere (the patch origin is the cell
        // corner, ~4096u from the real center — using it falsely culls the patch the
        // camera stands on). No near-dist limit: all cache landscape is near terrain.
        BoundingSphere bs;
        cacheWorldBounds(e, bs.center, bs.radius);
        if (frustum.ContainsSphere(bs) == ViewFrustum::OUTSIDE) continue;

        // Per-patch point-light selection (sphere-AABB nearest-32), byte-identical to
        // the object path. AABB from the patch's world bound sphere.
        const D3DXVECTOR3 lbMin(bs.center.x - bs.radius, bs.center.y - bs.radius, bs.center.z - bs.radius);
        const D3DXVECTOR3 lbMax(bs.center.x + bs.radius, bs.center.y + bs.radius, bs.center.z + bs.radius);
        const int lightCount = FixedFunctionShader::selectTextureLights(
            snapshotLights, snapshotCount, *view, lbMin, lbMax, idxFloats, logPerf);
        const D3DXVECTOR4 lightDataParams_v((float)lightCount, texelSize, 0.0f, 0.0f);
        effect->SetVector(ehLightDataParams, &lightDataParams_v);
        if (lightCount > 0) effect->SetVectorArray(ehLightIndices, (D3DXVECTOR4*)idxFloats, 8);

        effect->SetMatrix(ehWorld, reinterpret_cast<const D3DXMATRIX*>(e.worldTransformD3D));
        effect->SetTexture(ehTex0, e.d3dTexture);
        // No overlay -> bind base to tex2 so lerp(base, base, a) == base.
        effect->SetTexture(ehTex2, e.d3dOverlay ? e.d3dOverlay : e.d3dTexture);
        // Position via vertexBlendPalette[0] (= world*view), the same premultiplied
        // matrix the reflection/shadow paths use (bit-identical depth, no acne).
        D3DXMATRIX wv;
        D3DXMatrixMultiply(&wv, reinterpret_cast<const D3DXMATRIX*>(e.worldTransformD3D), view);
        effect->SetMatrixArray(ehVertexBlendPalette, &wv, 1);
        effect->CommitChanges();

        // Main-view winding: CW (the pass declares reflection CCW). Set after
        // CommitChanges, matching the shadow path's per-part cull override.
        device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CW);
        device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kVBStride);
        device->SetIndices(e.ib);
        device->SetFVF(MGE::GeometryCache::kVBFVF);
        DrawStats::count(e.triangleCount);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, e.vertexCount, 0, e.triangleCount);
    }
    effect->EndPass();
}

// renderShadowReceiverFromCache - main-view sibling of renderReflectionShadowsFromCache.
// Re-draws the cache opaque set (textured objects + terrain + skinned NPCs) as sun-
// shadow receivers AT THE SNAPSHOT POSE, so the receiver depth is bit-consistent with
// the cache color/depth that renderCachedOpaque/renderCachedTerrain wrote. The default
// receiver (renderShadow over recordMW) replays the engine's LIVE pose; with the async
// scene-graph walk the snapshot is one frame stale, so a live-pose receiver mismatches
// the cache depth and flickers on animated geometry (waving banners). renderShadow()
// skips the cache-covered set (skipCacheCovered) and this owns it instead. Full strength
// (no distant-land handover fade — the whole near scene is cache-owned in the main view).
// The main-view sun (ehSunVecView) is already bound by setupCommonEffect (no reflected
// swap), and the shadow lookup is view->world->shadow exactly as renderShadow() builds it.
void DistantLand::renderShadowReceiverFromCache(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    MGE_ZoneScopedN("renderShadowReceiverFromCache");

    const auto& cacheMap = MGE::GeometryCache::cache();
    if (cacheMap.empty()) return;

    // View space -> world (inverse view) -> shadow clip, same mapping renderShadow uses.
    D3DXMATRIX invView, viewToShadow[2];
    D3DXMatrixInverse(&invView, nullptr, view);
    viewToShadow[0] = invView * smViewproj[0];
    viewToShadow[1] = invView * smViewproj[1];
    effect->SetMatrixArray(ehShadowViewproj, viewToShadow, 2);
    effect->SetTexture(ehTex3, texSoftShadow);

    D3DXMATRIX viewproj;
    D3DXMatrixMultiply(&viewproj, view, proj);
    ViewFrustum frustum(&viewproj);

    effect->SetBool(ehHasBones, false);
    effect->SetInt(ehVertexBlendState, 0);
    effect->SetFloat(ehMaterialAlpha, 1.0f);
    // Receiver alpha = materialAlpha (1), not vertex colour (terrain vcol.a is the
    // AlphaGrid splat factor and must not modulate shadow strength).
    effect->SetBool(ehHasVCol, false);
    effect->SetFloat(ehShadowReflMult, 1.0f);   // full strength in the main view

    // Non-skinned receivers (objects + terrain). Same covered set the cache color owns
    // (textured, non-blend); untextured/alpha opaque stay on the recordMW receiver.
    effect->BeginPass(PASS_RENDERSHADOWFFE);
    for (const auto& kv : cacheMap) {
        const auto& e = kv.second;
        if (e.isSkinned) continue;          // skinned receiver = skinIndexed pass below
        if (e.blendEnable) continue;
        if (!e.d3dTexture) continue;        // untextured opaque is engine-drawn (recordMW receiver)
        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib) continue;

        BoundingSphere bs;
        cacheWorldBounds(e, bs.center, bs.radius);
        if (frustum.ContainsSphere(bs) == ViewFrustum::OUTSIDE) continue;

        D3DXMATRIX wv;
        D3DXMatrixMultiply(&wv, reinterpret_cast<const D3DXMATRIX*>(e.worldTransformD3D), view);
        D3DXMATRIX pal[4] = { wv, wv, wv, wv };
        effect->SetMatrixArray(ehVertexBlendPalette, pal, 4);

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

        // Main-view winding matches the cache color pass (CW normal, CCW mirrored).
        device->SetRenderState(D3DRS_CULLMODE, e.mirrored ? D3DCULL_CCW : D3DCULL_CW);
        device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kVBStride);
        device->SetIndices(e.ib);
        device->SetFVF(MGE::GeometryCache::kVBFVF);
        DrawStats::count(e.triangleCount);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, e.vertexCount, 0, e.triangleCount);
    }
    effect->EndPass();

    // Cache-skinned receivers (NPCs). Receiver depth rebuilt in the VS via 32-bone
    // skinIndexed -> mul(view), bit-identical to the cache skinned color pass.
    effect->BeginPass(PASS_RENDERSHADOWFFE_SKINNED);
    for (const auto& kv : cacheMap) {
        const auto& e = kv.second;
        if (!e.isSkinned) continue;
        if (e.skinnedUnsupported || e.numBones == 0) continue;
        if (e.blendEnable) continue;
        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib) continue;

        BoundingSphere bs;
        cacheSkinnedWorldBounds(e, bs.center, bs.radius);
        if (frustum.ContainsSphere(bs) == ViewFrustum::OUTSIDE) continue;

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

        device->SetRenderState(D3DRS_CULLMODE, e.mirrored ? D3DCULL_CCW : D3DCULL_CW);
        device->SetVertexDeclaration(MGE::GeometryCache::skinnedDecl());
        device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kSkinnedVBStride);
        device->SetIndices(e.ib);
        DrawStats::count(e.triangleCount);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, e.vertexCount, 0, e.triangleCount);
    }
    effect->EndPass();
}
