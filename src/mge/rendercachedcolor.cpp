// rendercachedcolor.cpp — cache-driven color (the renderer-takeover path).
//
// Owns the machinery that renders the MGE GeometryCache snapshot with full FFE
// color. Two consumers until S3 -- the water-reflection injection and the main
// scene -- now just the main-scene opaque + terrain passes (renderCachedOpaque /
// renderCachedTerrain), called from renderStage0.
//
// buildCacheMainState turns a cache entry's captured material + the frame-global
// sun/ambient into the RenderedState / FragmentState / LightState that
// FixedFunctionShader::renderMorrowind expects. World-space cull spheres come from
// cachebounds.h (cacheWorldBounds / cacheSkinnedWorldBounds), shared with the early
// visible-set build.

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
// bindCacheTextures and buildCacheMainState both consume this, so the sampler
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

// S3: the water-reflection cache passes lived here -- renderReflectionsFromCache
// (mirrored cache objects), renderReflectionShadowsFromCache (their sun-shadow
// re-draw) and renderReflectionTerrainFromCache (mirrored near terrain), plus the
// reflection point-light budget they shared. All three drew into texReflection,
// which went with MGE's water renderer.

// Build the RenderedState / FragmentState / LightState triple for one cached part.
// Was buildCacheReflectionState + a two-line buildCacheMainState wrapper that only
// flipped the cull winding; with the reflection caller gone (S3) the two collapsed,
// and CCW->CW moved in here as the single main-view winding.
static void buildCacheMainState(const MGE::GeometryCache::CachedGeometry& e,
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
    rs.cullMode       = D3DCULL_CW;    // main view (the reflection's mirrored CCW is gone)
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

    // S3: the sun-shadow fold (FFE applyCacheShadow, fed by MGE's cascaded atlas) was
    // set up here. The atlas went with rendershadow.cpp; the host owns sun shadows.

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

    // S3: the cache-terrain sun-shadow fold (cacheTerrainSunShadow) was set up here.

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

}

// renderShadowReceiverFromCache removed: the main-view sun shadow is now folded into
// the cache color passes (applyCacheShadow in renderCachedOpaque's FFE draws,
// cacheTerrainSunShadow in renderCachedTerrain), so receivers ride the color draws and
// track the occlusion-culled visKeys exactly — no separate full-cache receiver re-draw.
