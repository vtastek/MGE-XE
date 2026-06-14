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
// renderMorrowind expects. World-space cull spheres come from cachebounds.h
// (cacheWorldBounds / cacheSkinnedWorldBounds), shared with the early visible-set
// build. Kept together here so renderwater.cpp stays water-only and renderStage0
// stays orchestration-only.

#include "distantland.h"
#include "distantshader.h"
#include "drawstats.h"
#include "configuration.h"
#include "ffeshader.h"
#include "scenegraph_geometry_cache.h"
#include "cachebounds.h"
#include "mwbridge.h"
#include "support/log.h"
#include "mge_tracy.h"

#include <algorithm>
#include <cstdint>
#include <tuple>
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

// cacheWorldBounds / cacheSkinnedWorldBounds now live in cachebounds.h so the
// early deterministic visible-set build (renderdepth.cpp) and the cache color
// passes here share one definition of the cull sphere.

// A present map is usable only if the cache VB actually carries the UV set it
// samples — i.e. its texCoordSet is within the entry's uvSetCount.
static bool cacheMapActive(IDirect3DTexture9* tex, uint8_t uv, uint8_t uvSetCount) {
    return tex != nullptr && uv < uvSetCount;
}

// One reconstructed fixed-function texture stage.
struct CacheStage {
    IDirect3DTexture9* tex;
    uint8_t uv;          // texcoordIndex (the map's texCoordSet)
    BYTE colorOp;
    BYTE colorArg2;      // DIFFUSE for base, CURRENT for the rest
    BYTE alphaOp;        // MODULATE (matched) or SELECTARG2 (keep prev alpha)
};

// Build the ordered cache texture-stage list, matching MW's LIVE fixed-function
// setup (verified against [PPLFRS] A/B). Two rules, both load-bearing:
//   1. Per-slot ops (NOT OpenMW's table — MW is the match target):
//        BASE   : MODULATE(tex, DIFFUSE),   alpha MODULATE
//        DARK   : MODULATE(tex, CURRENT),   alpha MODULATE
//        DETAIL : MODULATE2X(tex, CURRENT), alpha keep-prev
//        GLOW   : ADD(tex, CURRENT),        alpha keep-prev
//      "keep-prev" alpha = MW's ALPHAOP DISABLE; encoded as SELECTARG2 so ShaderKey
//      sees neither alphaOpMatched nor alphaOpSelect1 and the JIT leaves c.a alone.
//   2. ORDER stages by texCoordSet ascending (MW assigns D3D stage index =
//      texCoordSet, NOT map slot). The "Glow in the Dark" night mesh re-authors
//      base/dark/detail across scrambled UV sets, so slot order != stage order;
//      because each stage's [0,1] saturation clamps the running result, applying
//      MODULATE2X at the wrong point made some windows over-bright in cache.
//      stable_sort keeps slot order on ties (two maps sharing a UV set).
// bindCacheTextures and buildCacheReflectionState both consume this, so the sampler
// binding and the stage ops can never drift. Returns the stage count.
static int buildCacheStages(const MGE::GeometryCache::CachedGeometry& e, CacheStage out[8]) {
    int n = 0;
    if (e.d3dTexture)
        out[n++] = { e.d3dTexture, e.baseUV,   D3DTOP_MODULATE,   D3DTA_DIFFUSE, D3DTOP_MODULATE };
    if (cacheMapActive(e.d3dDark,   e.darkUV,   e.uvSetCount))
        out[n++] = { e.d3dDark,     e.darkUV,   D3DTOP_MODULATE,   D3DTA_CURRENT, D3DTOP_MODULATE };
    if (cacheMapActive(e.d3dDetail, e.detailUV, e.uvSetCount))
        out[n++] = { e.d3dDetail,   e.detailUV, D3DTOP_MODULATE2X, D3DTA_CURRENT, D3DTOP_SELECTARG2 };
    if (cacheMapActive(e.d3dGlow,   e.glowUV,   e.uvSetCount))
        out[n++] = { e.d3dGlow,     e.glowUV,   D3DTOP_ADD,        D3DTA_CURRENT, D3DTOP_SELECTARG2 };
    std::stable_sort(out, out + n, [](const CacheStage& a, const CacheStage& b) { return a.uv < b.uv; });
    return n;
}

// Bind the entry's textures to sampler slots in stage order (renderMorrowind reads
// device texture[i] for stage i) — same buildCacheStages order the FragmentState uses.
static void bindCacheTextures(IDirect3DDevice9* device,
                              const MGE::GeometryCache::CachedGeometry& e) {
    CacheStage stages[8];
    const int n = buildCacheStages(e, stages);
    for (int i = 0; i < n; ++i) device->SetTexture(i, stages[i].tex);
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
    // TEXn matching the entry's uvSetCount: lifts the ShaderKey's uvSets so the JIT
    // declares texcoord0..n-1 and the dark/detail/glow stages can sample their own
    // sets (e.g. detail on set 2). Single-UV entries stay TEX1. DIFFUSE drives
    // vertexColour. (ShaderKey clamps uvSets down to the max texcoordIndex actually
    // used, so an over-stated count costs nothing.)
    const DWORD texFvf = static_cast<DWORD>(e.uvSetCount) << D3DFVF_TEXCOUNT_SHIFT;
    rs.fvf            = D3DFVF_XYZ | D3DFVF_NORMAL | texFvf | (useVCol ? D3DFVF_DIFFUSE : 0);
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
    // Reconstruct MW's live fixed-function multi-map blend so the FFE JIT replays
    // exactly what the reactive (PPL) path does on the same NiTexturingProperty. The
    // per-slot ops AND the stage ORDER (by texCoordSet) come from buildCacheStages,
    // shared with bindCacheTextures so the sampler binding matches. Single-map entries
    // collapse to one stage, identical to before. Terminated by a DISABLE stage.
    memset(&frs, 0, sizeof(frs));
    CacheStage stages[8];
    int ns = buildCacheStages(e, stages);
    if (ns > 7) ns = 7;   // leave room for the DISABLE terminator at stage[7]
    for (int i = 0; i < ns; ++i) {
        FragmentState::Stage& s = frs.stage[i];
        s.colorOp   = stages[i].colorOp; s.colorArg1 = D3DTA_TEXTURE; s.colorArg2 = stages[i].colorArg2;
        s.alphaOp   = stages[i].alphaOp; s.alphaArg1 = D3DTA_TEXTURE; s.alphaArg2 = stages[i].colorArg2;
        s.colorArg0 = D3DTA_CURRENT;     s.alphaArg0 = D3DTA_CURRENT; s.resultArg = D3DTA_CURRENT;
        s.texcoordIndex = stages[i].uv;
    }
    frs.stage[ns].colorOp = D3DTOP_DISABLE;   // memset 0 == colorOp 0 != DISABLE; set explicitly

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

// Reflection point-light cap. Reflections see a mirrored, often water-distorted
// surface, so the seam from selecting fewer lights is imperceptible — 8 instead of
// the main view's 32 keeps the heavy per-mesh FFE variant's per-pixel loop short.
static const unsigned int kReflMaxLights = 8;

void DistantLand::renderReflectionsFromCache(const D3DXMATRIX* view, const D3DXMATRIX* proj, float nearDist) {
    MGE_ZoneScopedN("renderReflectionsFromCache");
    DrawStats::ScopedStage _ds(DrawStats::ReflCacheColor);   // cache objects (lit color) injected into the reflection

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

    // Sun shadow fold: the FFE color shader samples the shadow atlas inline
    // (applyCacheShadow branch in PerPixelPS), replacing the standalone
    // renderReflectionShadowsFromCache receiver re-draw of this same object set.
    // The sun shadow map is world-space and the cache objects rasterize in
    // reflected view space, so bind reflected-view -> world (inverse reflView) ->
    // shadow clip, exactly as the standalone pass built it. shadowViewProj and
    // shadowReflMult are `shared` pool params: set on the distant-land effect,
    // read by the FFE variants. The caller has already bound the reflected-view
    // sunVecView (the fold's lit-ness gate needs the sun in the same space as
    // the rasterized normals).
    D3DXMATRIX invView, viewToShadow[2];
    D3DXMatrixInverse(&invView, nullptr, view);
    viewToShadow[0] = invView * smViewproj[0];
    viewToShadow[1] = invView * smViewproj[1];
    effect->SetMatrixArray(ehShadowViewproj, viewToShadow, 2);
    FixedFunctionShader::setCacheShadow(texSoftShadow, true);

    RenderedState rs;
    FragmentState frs;
    static LightState lightrs;   // holds maps; reused to avoid per-draw realloc

    // Diagnostics (gated on LogDistantPipeline): bucket the lit draws so the cache's
    // weight in the reflection is attributable vs the size-gated distant-statics LOD.
    // Statics carry no size gate here (shadow uses boundsRadius<50, the DL LOD ~150),
    // so the small-static buckets show exactly how many sub-LOD-size objects the cache
    // lights that the distant-statics pass would never generate — the candidate trim.
    const bool logPerf = Configuration.LogDistantPipeline;
    static unsigned s_dyn = 0, s_stLt50 = 0, s_st50_150 = 0, s_stGe150 = 0;
    static unsigned s_calls = 0;

    // Gather→sort→batched-draw. Pass 1 runs the existing filters/cull/size-gate and
    // pushes survivors (with their already-computed bounds) into this reused vector;
    // pass 2 sorts by a cheap effect-proxy key so consecutive draws share a ShaderKey,
    // then draws under one held-open effect pass (beginBatch/endBatch). Reordering is
    // visually order-independent here: only opaque + alpha-tested entries reach pass 2
    // (blended entries are filtered out below), all z-tested and depth-writing.
    struct ReflDraw {
        const MGE::GeometryCache::CachedGeometry* e;
        D3DXVECTOR3 center;
        BoundingSphere bs;
        D3DXVECTOR3 lbMin, lbMax;
        bool keep;   // exterior: mirrored sphere lands on a visible water rect (else culled)
    };
    static std::vector<ReflDraw> survivors;
    survivors.clear();

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

        // Histogram the frustum-visible set by type/size (world-space radius, matching
        // the LOD generation metric) BEFORE the size gate below, so the buckets show
        // the full distribution and quantify what the gate drops. dynamicHint!=0 => NPC.
        if (logPerf) {
            if (e.dynamicHint != 0)        ++s_dyn;
            else if (bs.radius < 50.0f)    ++s_stLt50;
            else if (bs.radius < 150.0f)   ++s_st50_150;
            else                           ++s_stGe150;
        }

        // Size gate (statics only): skip sub-150-radius statics. They're below the
        // distant-statics LOD generation threshold, so they never appear in the
        // dl.stat pass either — drawing them lit here is pure cache overdraw with no
        // distant counterpart, and they're too small to read in a reflection. NPCs /
        // dynamics (dynamicHint != 0) are always drawn. Matches the LOD ~150 gate and
        // the intent of the shadow caster gate (kShadowMinSize = 50). The shadow
        // re-draw (renderReflectionShadowsFromCache) carries the SAME gate so every
        // lit object is also shadowed (no lit-but-unshadowed mismatch).
        if (e.dynamicHint == 0 && bs.radius < 150.0f) continue;

        // World-space AABB for texture-light selection. The cache VB is WRITEONLY so
        // renderMorrowind's computeBoundingBox can't read it. Non-skinned: the TIGHT
        // world AABB (matches reactive computeBoundingBox -> same light set; a fat
        // sphere-cube over-selects). Skinned: the bone-derived sphere box.
        D3DXVECTOR3 lbMin, lbMax;
        if (e.isSkinned) {
            lbMin = D3DXVECTOR3(bs.center.x - bs.radius, bs.center.y - bs.radius, bs.center.z - bs.radius);
            lbMax = D3DXVECTOR3(bs.center.x + bs.radius, bs.center.y + bs.radius, bs.center.z + bs.radius);
        } else {
            cacheWorldAABB(e, lbMin, lbMax);
        }

        survivors.push_back({ &e, center, bs, lbMin, lbMax, true });
    }

    // Water-rect cull: a cache object reflects into VISIBLE water only if its mirrored
    // sphere (projected via the reflected viewproj — x,y NDC identical to reflCullViewProj)
    // lands on a surviving water-tile rect (reflectionWaterRects, worker-built: terrain-wet
    // + MSOC-visible + hysteresis). Objects whose reflection falls on no visible water are
    // pure overdraw — culled. The cache range (nearDist ~7168) sits inside the 2-cell
    // grazing-free zone and within staticReflRange, so neither the grazing nor range filter
    // drops a cache tile; only MSOC + hysteresis gate. reflWaterCullActive folds in the
    // interior / no-data / SWIMMING fallbacks (set by isReflectionWaterVisible): when the
    // screen-space water projection is unreliable it is false → keep all (no per-tile cull),
    // since the rects AND mask would otherwise collapse edge-on and over-cull.
    const bool cullByWater = DistantLand::reflWaterCullActive;
    unsigned reflStage2Drop = 0;   // rect survivors the fine silhouette mask additionally culled
    {
        const auto& rects = reflectionWaterRects;
        for (ReflDraw& s : survivors) {
            if (!cullByWater) { s.keep = true; continue; }
            const D3DXVECTOR3& c = s.bs.center;
            const float rad = s.bs.radius;
            const float cx = c.x*viewproj._11 + c.y*viewproj._21 + c.z*viewproj._31 + viewproj._41;
            const float cy = c.x*viewproj._12 + c.y*viewproj._22 + c.z*viewproj._32 + viewproj._42;
            const float cw = c.x*viewproj._14 + c.y*viewproj._24 + c.z*viewproj._34 + viewproj._44;
            bool keep = false;
            if (cw < 1e-3f) {
                keep = true;   // behind near plane → conservatively keep (matches the statics cull)
            } else {
                const float inv = 1.0f / cw;
                const float nx = cx * inv, ny = cy * inv;
                const float rx = rad * proj->_11 * inv;   // sphere radius in NDC
                const float ry = rad * proj->_22 * inv;
                bool rectKeep = false;
                for (const D3DXVECTOR4& wr : rects) {   // stage 1: coarse rect
                    if (nx + rx >= wr.x && nx - rx <= wr.z &&
                        ny + ry >= wr.y && ny - ry <= wr.w) { rectKeep = true; break; }
                }
                // Stage 2: fine water-silhouette mask refines rect survivors, dropping
                // objects flanking a thin/diagonal river (near a wet tile, off the water).
                const bool maskKeep = rectKeep && DistantLand::reflWaterMaskTestNDC(nx, ny, rx, ry);
                if (rectKeep && !maskKeep) ++reflStage2Drop;
                keep = maskKeep;
            }
            s.keep = keep;
        }
    }

    // Debug overlay (cycle state 5): stash each candidate as a box, GREEN = kept
    // (drawn), RED = culled. Same keep the draw consumes, so the overlay is an exact
    // before/after of the cull. The reflected viewproj is stashed for the frustum
    // wireframe drawn in the main view.
    if (debugReflFrustum) {
        reflDbgViewProj = viewproj;
        reflDbgValid = true;
        reflCacheDbg.clear();
        reflCacheDbg.reserve(survivors.size());
        unsigned green = 0;
        for (const ReflDraw& s : survivors) {
            reflCacheDbg.push_back({ s.bs.center, s.bs.radius, s.keep });
            if (s.keep) ++green;
        }
        LOG::logline("-- [REFL DBG] cache=%u waterRects=%u maskBits=%d kept=%u culled=%u "
                     "stage2drop=%u (cullByWater=%d)",
                     (unsigned)survivors.size(), (unsigned)reflectionWaterRects.size(),
                     DistantLand::reflWaterMaskSetBits(), green,
                     (unsigned)survivors.size() - green, reflStage2Drop, cullByWater ? 1 : 0);
    }

    // Sort by a cheap effect-proxy key so same-ShaderKey draws sit adjacent
    // without running the per-mesh selection. The light-variant bit can still
    // split a group in two, but the list is mostly grouped — enough to collapse
    // the per-draw effect switches inside the batched tail. (texture, skin, FVF,
    // mirror are the ShaderKey-relevant fields the loop drives per object.)
    std::sort(survivors.begin(), survivors.end(),
              [](const ReflDraw& a, const ReflDraw& b) {
                  return std::tie(a.e->d3dTexture, a.e->isSkinned, a.e->vbFVF, a.e->mirrored)
                       < std::tie(b.e->d3dTexture, b.e->isSkinned, b.e->vbFVF, b.e->mirrored);
              });

    FixedFunctionShader::beginBatch();
    unsigned drawnCount = 0;
    for (const ReflDraw& d : survivors) {
        if (!d.keep) continue;   // culled: reflection lands on no visible water
        ++drawnCount;
        const auto& e = *d.e;
        IDirect3DVertexBuffer9* vb = e.readVB();

        {
            MGE_ZoneScopedN("refl:state");
            buildCacheReflectionState(e, *view, sunVec, sunCol, sunAmb, ambCol, rs, frs, lightrs);
        }

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
        const float plMult = cacheHandoverFade(d.center, eye, e.dynamicHint, nearDist);

        // Shadow handover fade rides the SAME per-object fade (shared
        // shadowReflMult, read by the FFE shadow fold), so shadows and point
        // lights ramp together and land exactly on the geometry handover.
        effect->SetFloat(ehShadowReflMult, plMult);

        // Reflection base winding is inverted (CCW); a mirrored part flips again.
        device->SetRenderState(D3DRS_CULLMODE, e.mirrored ? D3DCULL_CW : D3DCULL_CCW);
        // Base + any dark/detail/glow maps to consecutive sampler slots (packed to
        // match the FragmentState stages buildCacheReflectionState emitted). Build the
        // stage list once: bind to the device (kept for state parity) AND capture it
        // to hand straight to renderMorrowind, so the FFE shader's tex0..N are set
        // without the per-stage device->GetTexture readback (a proxy round-trip).
        IDirect3DBaseTexture9* texArr[8];
        int texCount;
        {
            MGE_ZoneScopedN("refl:bind");
            CacheStage stages[8];
            texCount = buildCacheStages(e, stages);
            for (int i = 0; i < texCount; ++i) {
                device->SetTexture(i, stages[i].tex);
                texArr[i] = stages[i].tex;
            }
            device->SetIndices(e.ib);
        }

        MGE_ZoneScopedN("refl:draw");
        if (e.isSkinned) {
            // VS palette skinning: bind-pose VB + per-frame bone palette (model->world);
            // renderMorrowind selects the skinIndexed path and applies reflView.
            device->SetVertexDeclaration(MGE::GeometryCache::skinnedDecl());
            device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kSkinnedVBStride);
            const D3DXMATRIX* pal = reinterpret_cast<const D3DXMATRIX*>(e.bonePalette.data());
            // Reflections cap point lights at 8 (kReflMaxLights): the mirrored
            // surface hides the seam from the tighter selection, and the heavy
            // per-mesh variant loops far fewer lights.
            FixedFunctionShader::renderMorrowind(&rs, &frs, &lightrs, plMult, pal, (int)e.numBones, view, &d.lbMin, &d.lbMax,
                                                 FixedFunctionShader::LightMode::PerMesh, kReflMaxLights, texArr, (unsigned)texCount);
        } else {
            // Per-entry stride/FVF: dual-UV (44 / TEX2) for multi-map shapes, else 36 / TEX1.
            device->SetStreamSource(0, vb, 0, e.vbStride);
            device->SetFVF(e.vbFVF);
            FixedFunctionShader::renderMorrowind(&rs, &frs, &lightrs, plMult, nullptr, 0, nullptr, &d.lbMin, &d.lbMax,
                                                 FixedFunctionShader::LightMode::PerMesh, kReflMaxLights, texArr, (unsigned)texCount);
        }
    }
    FixedFunctionShader::endBatch();

    // Shadow fold off + full receiver strength for everything after this pass
    // (the main scene shares the FFE shader; gate must never leak).
    effect->SetFloat(ehShadowReflMult, 1.0f);
    FixedFunctionShader::setCacheShadow(nullptr, false);

    // Grouping quality: drawn ≫ effectSwitches confirms the sort collapsed the
    // draw list into a handful of held-open passes. drawn is the post-water-cull
    // count (<= survivors); candidates shows what the cull removed.
    if (logPerf) {
        LOG::logline("-- [REFL BATCH] drawn=%u candidates=%u effectSwitches=%u",
                     drawnCount, (unsigned)survivors.size(), FixedFunctionShader::batchEffectSwitches());
    }

    constexpr unsigned kIv = 300;
    if (logPerf && (++s_calls % kIv == 0)) {
        const unsigned st = s_stLt50 + s_st50_150 + s_stGe150;
        // Per-frame averages. visible = frustum survivors (pre-gate); drawn =
        // dynamics + statics>=150 (post-gate); dropped = sub-150 statics the gate removes.
        LOG::logline("-- [REFL CACHE COLOR] per-frame: visible=%u static=%u{<50=%u 50-150=%u >=150=%u} dynamic=%u "
                     "-> drawn=%u dropped<150=%u (nearDist=%.0f)",
                     (s_dyn + st) / kIv, st / kIv, s_stLt50 / kIv, s_st50_150 / kIv, s_stGe150 / kIv, s_dyn / kIv,
                     (s_dyn + s_stGe150) / kIv, (s_stLt50 + s_st50_150) / kIv, nearDist);
        s_dyn = s_stLt50 = s_st50_150 = s_stGe150 = 0;
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
    DrawStats::ScopedStage _ds(DrawStats::CacheOpaque);   // cache near objects (split from scene0)

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

    // Sun shadow fold (main-view sibling of renderReflectionsFromCache): the FFE color
    // shader samples the shadow atlas inline (applyCacheShadow branch in PerPixelPS),
    // replacing the standalone renderShadowReceiverFromCache receiver re-draw of this
    // same object set. Because the fold rides the color draws it tracks visKeys exactly
    // (occlusion-culled objects are absent from both), and shadows cost zero extra draws.
    // Main view: view -> world (inverse view) -> shadow clip; sunVecView is already the
    // main-view sun (setupCommonEffect; renderCachedOpaque runs before any reflected
    // swap). Gate matches the engine's own receiver pass (renderShadow in renderStage1:
    // isDistantCell + shadows enabled + weather) so the cache scene shadows wherever the
    // engine scene would. NOT gated on !IsMenu: menu frames keep the (stale) atlas the
    // last non-menu frame rendered and the receiver still applies it — dropping the fold
    // during menus would make cache shadows vanish whenever a menu is open. A non-distant
    // cell never has a valid atlas, so isDistantCell() guards that. shadowReflMult = 1
    // (no handover fade in the main view, all near is cache-owned).
    auto mwBridge = MWBridge::get();
    const bool foldShadows = (Configuration.MGEFlags & USE_SHADOWS) && isDistantCell()
        && mwBridge->CellHasWeather();
    if (foldShadows) {
        D3DXMATRIX invView, viewToShadow[2];
        D3DXMatrixInverse(&invView, nullptr, view);
        viewToShadow[0] = invView * smViewproj[0];
        viewToShadow[1] = invView * smViewproj[1];
        effect->SetMatrixArray(ehShadowViewproj, viewToShadow, 2);
        effect->SetFloat(ehShadowReflMult, 1.0f);
        FixedFunctionShader::setCacheShadow(texSoftShadow, true);
    }

    // Consume the SAME deterministic set the depth pre-pass drove off this frame
    // (frustumVisibleKeys / buildFrustumVisibleSet) rather than re-culling per part.
    // This keeps the color pass byte-for-byte in lockstep with depth — including any
    // MSOC occlusion refinement buildFrustumVisibleSet applied — so a refined-away
    // static is absent from BOTH depth and color (no lit-but-no-depth or vice versa).
    // The set is built from the game view*proj (the camera the engine culls against);
    // the lateral planes match this pass's projection, so close parts/hands/belts that
    // pass the engine's cull are in the set. drawEntry still applies the per-pass
    // material filters (blend / untextured / terrain / unsupported-skin) below.
    const std::vector<uint32_t>& visKeys = frustumVisibleKeys();

    // Diagnostics (gated on LogDistantPipeline): per-interval skip accounting so we
    // can see which covered-but-skipped parts the engine suppression would drop.
    // skinnedUnsupported (>kMaxBones armor like full-skeleton belts) is the prime
    // suspect for "missing on multiple NPCs"; log a sample texture name.
    const bool logPerf = Configuration.LogDistantPipeline;
    static unsigned s_drawn = 0, s_skBlend = 0, s_skNoTex = 0, s_skLand = 0, s_skUnskin = 0;
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

        // Texture-light selection AABB (renderMorrowind can't read our WRITEONLY VB
        // via computeBoundingBox, so we hand it these bounds — the object light-seam
        // fix). Non-skinned: the TIGHT world AABB (8 model-AABB corners transformed by
        // the world matrix), byte-matching the reactive computeBoundingBox so the cache
        // selects the SAME lights — a fat sphere-cube over-selects nearby lights and
        // made windows brighter in cache. Skinned: the bone-palette-derived sphere box.
        D3DXVECTOR3 lbMin, lbMax;
        if (e.isSkinned) {
            BoundingSphere bs;
            cacheSkinnedWorldBounds(e, bs.center, bs.radius);
            lbMin = D3DXVECTOR3(bs.center.x - bs.radius, bs.center.y - bs.radius, bs.center.z - bs.radius);
            lbMax = D3DXVECTOR3(bs.center.x + bs.radius, bs.center.y + bs.radius, bs.center.z + bs.radius);
        } else {
            cacheWorldAABB(e, lbMin, lbMax);
        }

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
        // Base + any dark/detail/glow maps to consecutive sampler slots (packed to
        // match the FragmentState stages buildCacheMainState emitted).
        bindCacheTextures(device, e);
        device->SetIndices(e.ib);

        if (e.isSkinned) {
            // VS palette skinning: bind-pose VB + per-frame bone palette (model->world);
            // renderMorrowind selects the skinIndexed path and applies the main view.
            device->SetVertexDeclaration(MGE::GeometryCache::skinnedDecl());
            device->SetStreamSource(0, vb, 0, MGE::GeometryCache::kSkinnedVBStride);
            const D3DXMATRIX* pal = reinterpret_cast<const D3DXMATRIX*>(e.bonePalette.data());
            FixedFunctionShader::renderMorrowind(&rs, &frs, &lightrs, 1.0f, pal, (int)e.numBones, view, &lbMin, &lbMax);
        } else {
            // Per-entry stride/FVF: dual-UV (44 / TEX2) for multi-map shapes, else 36 / TEX1.
            device->SetStreamSource(0, vb, 0, e.vbStride);
            device->SetFVF(e.vbFVF);
            FixedFunctionShader::renderMorrowind(&rs, &frs, &lightrs, 1.0f, nullptr, 0, nullptr, &lbMin, &lbMax);
        }
        if (logPerf) ++s_drawn;
    };

    // Iterate the shared frustum/occlusion-refined set (key -> cacheMap), the same
    // forEach pattern renderDepthFromCache uses, so depth and color draw the same set.
    for (uint32_t key : visKeys) {
        auto it = cacheMap.find(key);
        if (it != cacheMap.end()) drawEntry(it->second);
    }

    // Shadow fold off + full strength for everything after this pass (the main scene
    // shares the FFE shader; the gate must never leak past the cache opaque draws).
    if (foldShadows) {
        FixedFunctionShader::setCacheShadow(nullptr, false);
        effect->SetFloat(ehShadowReflMult, 1.0f);
    }

    if (logPerf && (++s_calls % 300 == 0)) {
        LOG::logline("-- [CACHE OPAQUE] drawn=%u skip{blend=%u notex=%u land=%u unskin=%u} visKeys=%zu sampleUnskinTex=%s",
                     s_drawn, s_skBlend, s_skNoTex, s_skLand, s_skUnskin, visKeys.size(),
                     s_lastUnskinTex ? s_lastUnskinTex : "(none)");
        s_drawn = s_skBlend = s_skNoTex = s_skLand = s_skUnskin = 0;
    }
}

// NOTE: no longer invoked — the reflection sun shadow is folded into the FFE color
// pass (applyCacheShadow in XE FixedFuncEmu.fx; see renderReflectionsFromCache).
// Kept because PASS_RENDERSHADOWFFE/_SKINNED are still used by the main-view
// receiver (renderShadowReceiverFromCache) and as the reference implementation of
// the standalone receiver math the fold mirrors.
void DistantLand::renderReflectionShadowsFromCache(const D3DXMATRIX* view, const D3DXMATRIX* proj, float nearDist) {
    MGE_ZoneScopedN("renderReflectionShadowsFromCache");
    DrawStats::ScopedStage _ds(DrawStats::ReflCacheShadow);   // cache objects' shadow re-draw

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

        // Size gate, identical to the color pass (statics only, terrain exempt): only
        // objects the color pass actually drew get a shadow re-draw, so there's no
        // lit-but-unshadowed mismatch and the receiver set tracks the lit set exactly.
        if (!e.isLandscape && e.dynamicHint == 0 && bs.radius < 150.0f) continue;

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
        // Per-entry stride/FVF: multi-map objects carry extra UV sets.
        device->SetStreamSource(0, vb, 0, e.vbStride);
        device->SetIndices(e.ib);
        device->SetFVF(e.vbFVF);
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
    DrawStats::ScopedStage _ds(DrawStats::ReflCacheTerrain);   // cache terrain injected into the reflection
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
    // pixels live in. selectTextureLights reads D3DTS_PROJECTION for its frustum
    // precull — but unlike the main view, the reflection has an ACTIVE hardware clip
    // plane (renderWaterReflection: SetClipPlane + CLIPPLANEENABLE) whose device-space
    // interpretation depends on D3DTS_PROJECTION. Leaving reflProj set across the patch
    // DRAWS corrupts that clip plane and clips ALL reflected terrain (the regression).
    // So capture the clip-consistent projection here and bracket the reflProj swap
    // tightly around selectTextureLights, restoring it before each draw.
    const bool logPerf = Configuration.LogDistantPipeline;
    const float texelSize = FixedFunctionShader::texLightTexelSize();
    effect->SetTexture(ehLightData, FixedFunctionShader::textureLightData());
    effect->SetMatrix(ehTexLightView, view);
    device->SetTransform(D3DTS_PROJECTION, proj);
    float idxFloats[8 * 4] = { 0 };

    // Set the shared `proj` param explicitly, exactly as the WORKING main-view lit path
    // does (renderCachedTerrain). P13 (CacheTerrainVS) renders without this because the
    // reflection's prior passes leave `proj` = reflProj. But the lit pass (P16) carries
    // extra globals (lightIndices[8], texLightView, the s7 sampler) that perturb the
    // ID3DXEffect register layout — the same class of collision that corrupted the depth
    // effect (commit 35fcd0f). If P16 resolves the shared `proj` to a different register
    // than the prior pass wrote, the VS multiplies by a stale projection -> degenerate
    // clip space -> nothing rasterizes (which is exactly the P16 reflection symptom:
    // VS-stage rejection, no fragments even with a solid-color PS). Writing `proj` here
    // under the P16-bound layout pins reflProj into whatever register P16 actually reads.
    effect->SetMatrix(ehProj, proj);

    // Sun shadow fold inputs (CacheTerrainReflLitPS darkens shadowed fragments
    // inline, mirroring the FFE object fold — the standalone terrain receiver
    // re-draw went away with renderReflectionShadowsFromCache). Same P16 lesson
    // as ehProj above: this pass now reads shadowViewProj / sunVecView /
    // shadowReflMult / tex3, so set every one of them under the P16 layout.
    // World-space shadow map, reflected-view pixels: reflected-view -> world
    // (inverse reflView) -> shadow clip. The atlas rides sampDepth (tex3) here —
    // no conflict, terrain uses tex0/tex2 only. sunVecView must be the sun in
    // reflected-view space (the lit-ness dot uses rasterized reflView normals);
    // the cache object pass re-binds it and renderWaterReflection restores the
    // main-view sun afterwards.
    D3DXMATRIX invView, viewToShadow[2];
    D3DXMatrixInverse(&invView, nullptr, view);
    viewToShadow[0] = invView * smViewproj[0];
    viewToShadow[1] = invView * smViewproj[1];
    effect->SetMatrixArray(ehShadowViewproj, viewToShadow, 2);
    effect->SetTexture(ehTex3, texSoftShadow);
    D3DXVECTOR3 sunVecViewRefl;
    D3DXVec3TransformNormal(&sunVecViewRefl, reinterpret_cast<const D3DXVECTOR3*>(&sunVec), view);
    effect->SetFloatArray(ehSunVecView, sunVecViewRefl, 3);

    MGE::SceneGraph::SnapshotReadLock snapshotLock;
    const auto& snapshotLights = MGE::SceneGraph::pointLights();
    const unsigned int snapshotCount =
        std::min<unsigned int>((unsigned int)snapshotLights.size(), FixedFunctionShader::maxTexLights());

    // Lit reflection terrain (P16: CacheTerrainLitVS/CacheTerrainReflLitPS) so reflected
    // near terrain receives the same dynamic point lights (candles/torches) as reflected
    // objects, plus the below-water shader clip. P16 previously failed to rasterize; the
    // suspected cause is the stale shared `proj` (see the explicit SetMatrix(ehProj) above).
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

        // Shadow handover fade: same per-patch metric the standalone terrain
        // receiver used (world-space patch center; DL LOD beyond has no shadows).
        effect->SetFloat(ehShadowReflMult, cacheHandoverFade(bs.center, eye, e.dynamicHint, nearDist));

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

    // Full receiver strength for everything after this pass (shared param).
    effect->SetFloat(ehShadowReflMult, 1.0f);
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
    DrawStats::ScopedStage _ds(DrawStats::CacheTerrain);   // cache near terrain (split from scene0)

    const auto& cacheMap = MGE::GeometryCache::cache();
    if (cacheMap.empty()) return;

    // Main-view projection for the shared `proj` pool param; caller restores distProj.
    effect->SetMatrix(ehProj, proj);
    // D3DTS_PROJECTION drives selectTextureLights' frustum precull (renderCachedOpaque
    // already set it to mwProj, but be self-contained for the object-less case).
    device->SetTransform(D3DTS_PROJECTION, proj);

    // Consume the SAME engine-visible set the depth pre-pass and the opaque pass drive
    // off (frustumVisibleKeys / buildFrustumVisibleSet). Terrain tiles ARE in that set:
    // the engine's MSOC classify covers worldLandscapeRoot, so the set already carries
    // the engine's occlusion-culled terrain (the [VISKEYS] land count). Iterating it here
    // instead of re-frustum-culling the full ~2304-tile cache drops cache terrain to the
    // engine's drawn count (no occluded tiles behind hills), and keeps depth/opaque/terrain
    // replaying one identical set. The handover band clip plane (set by the caller) still
    // far-cuts to nearViewRange so the DL LOD owns everything beyond.
    const std::vector<uint32_t>& visKeys = frustumVisibleKeys();

    // Texture-light setup: bind the shared light-data texture once; per patch we run
    // the same selection the object path uses and push lightIndices/lightDataParams/
    // texLightView. texLightView = view (world->view) so the shader transforms the
    // texture's world-space light positions into the terrain's view space.
    const bool logPerf = Configuration.LogDistantPipeline;
    const float texelSize = FixedFunctionShader::texLightTexelSize();
    effect->SetTexture(ehLightData, FixedFunctionShader::textureLightData());
    effect->SetMatrix(ehTexLightView, view);
    float idxFloats[8 * 4] = { 0 };

    // Sun shadow fold (CacheTerrainLitPS samples inline, mirroring the FFE object fold).
    // Main-view view -> world (inverse view) -> shadow clip; sunVecView is already the
    // main-view sun. The fold is gated by shadowReflMult: 1 = shadowed (shadows enabled +
    // weather + not a menu, the condition that filled texSoftShadow this frame), 0 = no
    // fold. Replaces the standalone main-view terrain receiver (renderShadowReceiverFromCache).
    // Gate matches the engine receiver (isDistantCell + shadows + weather); NOT gated on
    // !IsMenu so cache terrain keeps shadows while a menu is open (the atlas stays valid,
    // just not regenerated). isDistantCell() guards the no-atlas case.
    auto mwBridge = MWBridge::get();
    const bool foldShadows = (Configuration.MGEFlags & USE_SHADOWS) && isDistantCell()
        && mwBridge->CellHasWeather();
    if (foldShadows) {
        D3DXMATRIX invView, viewToShadow[2];
        D3DXMatrixInverse(&invView, nullptr, view);
        viewToShadow[0] = invView * smViewproj[0];
        viewToShadow[1] = invView * smViewproj[1];
        effect->SetMatrixArray(ehShadowViewproj, viewToShadow, 2);
        effect->SetTexture(ehTex3, texSoftShadow);
    }
    effect->SetFloat(ehShadowReflMult, foldShadows ? 1.0f : 0.0f);

    // Hold the snapshot lock across the patch loop: selectTextureLights reads the
    // point-light snapshot (and, on a revision bump, uploads texLightData) under it.
    MGE::SceneGraph::SnapshotReadLock snapshotLock;
    const auto& snapshotLights = MGE::SceneGraph::pointLights();
    const unsigned int snapshotCount =
        std::min<unsigned int>((unsigned int)snapshotLights.size(), FixedFunctionShader::maxTexLights());

    effect->BeginPass(PASS_RENDERCACHETERRAINLIT);
    for (uint32_t key : visKeys) {
        auto it = cacheMap.find(key);
        if (it == cacheMap.end()) continue;
        const auto& e = it->second;
        if (!e.isLandscape || e.isSkinned) continue;
        if (!e.d3dTexture) continue;          // need at least a base texture
        IDirect3DVertexBuffer9* vb = e.readVB();
        if (!vb || !e.ib) continue;

        // World-space bound for the light-selection AABB. No frustum/occlusion cull here:
        // visKeys is already the engine's culled drawn set. (Patch origin is the cell
        // corner ~4096u off the real center, so use the true world bound, not the origin.)
        BoundingSphere bs;
        cacheWorldBounds(e, bs.center, bs.radius);

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

    // Full receiver strength for everything after this pass (shared param must not leak).
    effect->SetFloat(ehShadowReflMult, 1.0f);
}

// renderShadowReceiverFromCache removed: the main-view sun shadow is now folded into
// the cache color passes (applyCacheShadow in renderCachedOpaque's FFE draws,
// cacheTerrainSunShadow in renderCachedTerrain), so receivers ride the color draws and
// track the occlusion-culled visKeys exactly — no separate full-cache receiver re-draw.
