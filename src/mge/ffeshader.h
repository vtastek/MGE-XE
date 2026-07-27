#pragma once

#include "proxydx/d3d8header.h"
#include "scenegraph.h"   // MGE::SceneGraph::PointLight for selectTextureLights

#include <unordered_map>
#include <vector>



struct RenderedState {
    IDirect3DTexture9* texture;
    // The textures bound to FFE stages 0..3 (texture == stageTexture[0]). MW folds a shape's
    // dark/detail/glow siblings into extra stages of the SAME DIP, so a consumer that reads only
    // stage 0 reproduces the base map at full brightness. FragmentState::stage[] already carried
    // the ops; this carries what those ops sample. (Stages 4-7 exist in D3D but MW never binds
    // past 3 — NiTexturingProperty has base/dark/detail/gloss/glow/bump + decals, and the FFE
    // chain terminates at the first COLOROP = DISABLE well before then.)
    IDirect3DTexture9* stageTexture[4];
    IDirect3DVertexBuffer9* vb;
    UINT vbOffset, vbStride;
    IDirect3DIndexBuffer9* ib;
    DWORD ibBase;
    DWORD fvf;
    DWORD zWrite, cullMode;
    DWORD vertexBlendState;
    D3DXMATRIX worldTransforms[4];
    D3DXMATRIX viewTransform;
    D3DXMATRIX worldViewTransforms[4];
    D3DCOLORVALUE diffuseMaterial;
    BYTE blendEnable, srcBlend, destBlend;
    BYTE alphaTest, alphaFunc, alphaRef;
    BYTE useLighting, useFog, matSrcDiffuse, matSrcEmissive;

    D3DPRIMITIVETYPE primType;
    UINT baseIndex, minIndex, vertCount, startIndex, primCount;
};

struct FragmentState {
    struct Stage {
        BYTE colorOp, colorArg1, colorArg2;
        BYTE alphaOp, alphaArg1, alphaArg2;
        BYTE colorArg0, alphaArg0, resultArg;
        DWORD texcoordIndex;
        DWORD texTransformFlags;
        float bumpEnvMat[2][2];
        float bumpLumiScale, bumpLumiBias;
    } stage[8];

    struct Material {
        D3DCOLORVALUE diffuse, ambient, emissive;
    } material;
};

struct LightState {
    struct Light {
        D3DLIGHTTYPE type;
        D3DCOLORVALUE diffuse;
        D3DVECTOR position;     // position / normalized direction
        D3DVECTOR viewspacePos;
        union {
            D3DVECTOR falloff;  // constant, linear, quadratic
            D3DVECTOR ambient;  // for directional lights
        };
    };

    D3DCOLORVALUE globalAmbient;
    std::unordered_map<DWORD, Light> lights;
    std::unordered_map<DWORD, bool> lightsTransformed;
    std::vector<DWORD> active;
};

class FixedFunctionShader {
    struct ShaderKey {
        DWORD uvSets : 4;
        DWORD usesSkinning : 1;
        // Cache-driven 32-bone indexed skinning (boneMatrices + skinIndexed), as
        // opposed to the reactive 4-matrix vertexBlendPalette path (usesSkinning).
        // Mutually exclusive with usesSkinning; set only by the cache reflection feed.
        DWORD usesCacheSkin : 1;
        DWORD vertexColour : 1;
        DWORD heavyLighting : 1;
        // When 1, generate the USE_TEXTURE_LIGHTS variant — shader reads
        // point lights from a 1D dynamic texture (3 texels per light)
        // populated from MGE::SceneGraph::pointLights() with a runtime loop
        // count. Overrides the heavyLighting 4/8 choice.
        DWORD useTextureLightVariant : 1;
        // When 1, generate the USE_TILED_LIGHTS variant — shader reads
        // point lights from the per-frame screen-tile grid (texLightGrid +
        // texLightIndexList) instead of the per-mesh lightIndices array.
        // Mutually exclusive with useTextureLightVariant; set only by the
        // main-view tiled callers (reactive + cache main). Reflection keeps
        // useTextureLightVariant.
        DWORD usesTiledLightVariant : 1;
        DWORD vertexMaterial : 2;
        DWORD fogMode : 2;
        DWORD activeStages : 3;
        DWORD usesBumpmap : 1;
        DWORD bumpmapStage : 3;
        DWORD usesTexgen : 1;
        DWORD projectiveTexgen : 1;
        DWORD texgenStage : 3;

        struct Stage {
            DWORD colorOp : 6;
            DWORD colorArg1 : 6;
            DWORD colorArg2 : 6;
            DWORD colorArg0 : 6;
            DWORD alphaOpMatched : 1;
            DWORD alphaOpSelect1 : 1;
            DWORD texcoordIndex : 2;
            DWORD texcoordGen : 4;
        } stage[8];

        ShaderKey() {}
        ShaderKey(const RenderedState* rs, const FragmentState* frs, const LightState* lightrs);
        bool operator<(const ShaderKey& other) const;
        bool operator==(const ShaderKey& other) const;
        void log() const;

        struct hasher {
            std::size_t operator()(const ShaderKey& k) const;
        };
    };

    struct ShaderLRU {
        ID3DXEffect* effect;
        FixedFunctionShader::ShaderKey last_sk;
    };

    static IDirect3DDevice* device;
    static ID3DXEffectPool* constantPool;
    static std::unordered_map<ShaderKey, ID3DXEffect*, ShaderKey::hasher> cacheEffects;
    static ShaderLRU shaderLRU;
    static ID3DXEffect* effectDefaultPurple;

    // --- Batched FFE submit (opt-in; see beginBatch/endBatch) ---
    // When s_inBatch, renderMorrowind holds one effect pass open across
    // consecutive same-effect draws (CommitChanges per object) instead of a
    // full Begin/BeginPass/EndPass/End per draw. s_batchEffect is the effect
    // whose pass is currently open (nullptr = none). s_batchSwitches counts how
    // many times the batched tail had to close one pass and open another
    // (effect switches) since beginBatch — a grouping-quality metric the caller
    // logs. Non-batched callers never touch these (default path unchanged).
    static ID3DXEffect* s_batchEffect;
    static bool s_inBatch;
    static unsigned int s_batchSwitches;

    // Batched-mode frame-invariant param cache. The FFE light/view params below
    // are `shared` (constant pool) so a value set on any variant propagates to
    // all of them; within a single-view batch they never change. While s_inBatch,
    // renderMorrowind pushes each only when its value differs from the last draw
    // (and caches it here), so a param fires once (draw 1) and the rest skip both
    // the Set* and the CommitChanges upload it would force. Each has a validity
    // flag because a demoted draw (candidateCount==0) takes the engine path and
    // never touches the texture-light params — beginBatch() clears all flags so
    // draw 1 always re-sets. Non-batched callers ignore this (always Set).
    static bool s_biSunDir, s_biSceneAmbient, s_biSunDiffuse;
    static bool s_biTexLightView, s_biTexLightData, s_biDebugHeat, s_biCheckAmbient;
    static D3DXVECTOR3 s_lastSunDir, s_lastSceneAmbient, s_lastSunDiffuse;
    static D3DXMATRIX s_lastTexLightView;
    static IDirect3DTexture9* s_lastTexLightData;
    static float s_lastDebugHeat;
    static DWORD s_lastCheckAmbient;   // GetRenderState(D3DRS_AMBIENT), read once per batch

    // Dynamic 1D texture holding per-frame light data for the
    // USE_TEXTURE_LIGHTS shader path. Layout: 3 texels per light
    // (pos+ambient, diffuse, falloff+radius) at R32G32B32A32F. Width =
    // 3 * kMaxTexLights; height = 1. Updated once per SceneGraph
    // frameRevision via LockRect with DISCARD; bound to sampler slot 7
    // per draw when the variant is selected. See
    // evaluatePointLightsTextured in XE FixedFuncEmu.fx.
    //
    // Pool size = upper bound on per-frame snapshot lights we can store
    // on the GPU. 256 is a deliberately generous cap: dense modded
    // interiors top out around 150-200 lights. Memory: 256 * 3 texels
    // * 16 bytes/texel = 12 KB texture + ~20 KB per-call stack alloc
    // in renderMorrowind. Texture-upload, per-mesh selection, and
    // pixel-shader cost are all bounded by the actual snapshot size and
    // kMaxIndicesPerMesh — pool size doesn't enter the hot path.
    static const unsigned int kMaxTexLights = 256;
    static const unsigned int kTexelsPerLight = 3;
    // Per-mesh selected-light cap. 32 indices = 8 float4s in the
    // shader's `lightIndices` array; ps_3_0 source instruction budget
    // covers this comfortably and DXVK lifts the limit anyway. Visual
    // benefit kicks in only on dense interiors / Vivec cantons where a
    // single wall sits in range of >16 nearby lights — sparse exteriors
    // pay nothing extra (the runtime loop is bounded by the actual
    // selected count, not the cap).
    static const unsigned int kMaxIndicesPerMesh = 32;
    static IDirect3DTexture9* texLightData;

    // --- Tiled point lighting (USE_TILED_LIGHTS, main view only) ---
    // Screen-tile light grid built once per frame by buildTileGrid. Two
    // textures (Doom-2016 scheme): texLightGrid holds one texel per cluster
    // (.x=offset .y=count into the index list); texLightIndexList is a flat
    // array of light-row indices (4 per texel) into texLightData. Both are
    // DYNAMIC A32B32G32R32F, uploaded via LockRect-DISCARD when the grid
    // changes (revision + view hash). The shader pixel loop is bounded by the
    // per-tile count, capped at kPerTileCap.
    //
    // Z-extensible addressing: cluster id = (sliceZ*tilesY + tileY)*tilesX +
    // tileX, with kMaxSlicesZ==1 now. The grid texture rows are
    // kMaxTilesY*kMaxSlicesZ tall so a depth-slice dimension drops in without
    // a realloc — clustered later = set numSlicesZ>1 + add a Z loop here and
    // a sliceZ=f(viewpos.z) term in the shader.
    static const unsigned int kTileSizePx     = 32;     // 16 = one-line A/B
    static const unsigned int kMaxTilesX      = 128;    // covers 4096px wide
    static const unsigned int kMaxTilesY      = 72;     // covers 2304px tall (4K UHD = 68)
    static const unsigned int kMaxSlicesZ     = 1;      // tiled only (no Z yet)
    static const unsigned int kPerTileCap     = 64;     // per-pixel loop bound
    // Flat index-list budget = the grid's TRUE max demand (every tile filled to the
    // per-tile cap). Sizing it this way means the GLOBAL budget can NEVER starve
    // before the per-tile cap at ANY resolution the grid covers (up to 4096x2304,
    // i.e. 4K UHD): 128*72*64 = 589824. The per-tile cap alone then bounds quality,
    // not the screen resolution — so 2K/4K behave like 1080p (no "fill ends
    // mid-screen" regression on bigger displays). The index texture (~2.4MB) is
    // allocated once; per-frame upload cost still scales with the ACTUAL filled
    // count (usedRows), so low-density scenes stay cheap. Clustered Z (deferred)
    // is the real fix for the per-pixel cost of long per-tile lists.
    static const unsigned int kMaxTotalIndices = kMaxTilesX * kMaxTilesY * kMaxSlicesZ * kPerTileCap;
    static const unsigned int kIndexTexW      = 1024;   // index-list texel width
    // kIndexTexH = ceil(kMaxTotalIndices / 4 / kIndexTexW)
    static const unsigned int kIndexTexH      = (kMaxTotalIndices / 4 + kIndexTexW - 1) / kIndexTexW;
    static IDirect3DTexture9* texLightGrid;
    static IDirect3DTexture9* texLightIndexList;
    // Active grid dimensions for the current frame (<= kMaxTiles*). Set by
    // buildTileGrid, pushed to the shader's tileGridParams.
    static unsigned int gridTilesX, gridTilesY;
    // Rebuild key: the grid is light-dependent (revision) AND view-dependent
    // (camera view+proj), unlike texLightData which is revision-only.
    static uint64_t lastGridRevision;
    static uint32_t lastGridViewHash;

    // Light-selection scratch, shared by selectTextureLights (per-mesh) and
    // buildTileGrid (tiled) via ensureLightUpload()/precullLights(). Lifted
    // from selectTextureLights' function-local statics so both paths reuse the
    // single revision-keyed upload + view-keyed frustum precull.
    static D3DXVECTOR3 s_lightWorldPos[kMaxTexLights];
    static bool        s_lightAlive[kMaxTexLights];
    static D3DXMATRIX  s_lastPrecullView;

    // Revision-keyed texLightData upload (pos/diffuse/falloff per light) +
    // s_lightWorldPos fill. No-op when the snapshot revision is unchanged.
    static void ensureLightUpload(const std::vector<MGE::SceneGraph::PointLight>& snapshotLights,
                                  unsigned int snapshotCount, bool logPerf);
    // View-keyed frustum precull: mark s_lightAlive[i] for lights whose
    // 2*radius sphere intersects the camera frustum. device's D3DTS_PROJECTION
    // must match `view`. Recomputed only when the view changes.
    static void precullLights(const std::vector<MGE::SceneGraph::PointLight>& snapshotLights,
                              const D3DXMATRIX& view, bool logPerf);
    // Cached SceneGraph::frameRevision() that's currently in the
    // texture. (uint64_t)-1 sentinel means "never uploaded."
    static uint64_t lastUploadedRevision;
    // Cached count of POINT lights actually packed into the texture
    // (≤ snapshot size; non-points are skipped). Used to bound the
    // per-mesh selection scan and to detect "no point lights this
    // frame" so we can skip the whole texture-light path.
    static unsigned int lastUploadedPointCount;

    // Per-mesh bbox cache (vtastek pattern). Object-space AABB computed
    // once per unique mesh by walking the vertex buffer; cache hit
    // transforms 8 corners by worldTransforms[0] to get world-space
    // bbox. MeshKey deduplicates by VB+IB+FVF+range so identical meshes
    // share a cache entry.
    struct MeshKey {
        IDirect3DVertexBuffer9* vb;
        IDirect3DIndexBuffer9*  ib;
        DWORD fvf;
        UINT  baseIndex, vertCount, startIndex, primCount;
        bool operator==(const MeshKey& o) const {
            return vb == o.vb && ib == o.ib && fvf == o.fvf
                && baseIndex == o.baseIndex && vertCount == o.vertCount
                && startIndex == o.startIndex && primCount == o.primCount;
        }
    };
    struct MeshKeyHash {
        std::size_t operator()(const MeshKey& k) const {
            std::size_t h = (std::size_t)k.vb;
            h = h * 31 + (std::size_t)k.ib;
            h = h * 31 + (std::size_t)k.fvf;
            h = h * 31 + (std::size_t)k.baseIndex;
            h = h * 31 + (std::size_t)k.vertCount;
            h = h * 31 + (std::size_t)k.startIndex;
            h = h * 31 + (std::size_t)k.primCount;
            return h;
        }
    };
    struct ObjectSpaceBBox {
        float minX, minY, minZ;
        float maxX, maxY, maxZ;
    };
    static std::unordered_map<MeshKey, ObjectSpaceBBox, MeshKeyHash> bboxCache;

    // Returns true and fills [outMin, outMax] with the world-space AABB
    // of the mesh referenced by `rs`. Returns false if the bbox can't
    // be derived (no VB, no position FVF, lock fails, unsupported prim).
    // Caches object-space bboxes by MeshKey so the VB walk runs once
    // per unique mesh per session.
    static bool computeBoundingBox(const RenderedState* rs,
        D3DXVECTOR3& outMin, D3DXVECTOR3& outMax);

    static D3DXHANDLE ehWorld, ehWorldView;
    static D3DXHANDLE ehView, ehBoneMatrices;   // cache 32-bone indexed skinning
    static D3DXHANDLE ehVertexBlendState, ehVertexBlendPalette;
    static D3DXHANDLE ehTex0, ehTex1, ehTex2, ehTex3, ehTex4, ehTex5;
    static D3DXHANDLE ehMaterialDiffuse, ehMaterialAmbient, ehMaterialEmissive;
    static D3DXHANDLE ehLightSceneAmbient, ehLightSunDiffuse, ehLightSunDirection;
    static D3DXHANDLE ehLightDiffuse, ehLightAmbient, ehLightPosition;
    static D3DXHANDLE ehLightFalloffQuadratic, ehLightFalloffLinear, ehLightFalloffConstant;
    // Texture-light path handles
    static D3DXHANDLE ehTexLightData, ehLightDataParams, ehLightIndices, ehTexLightView;
    // Tiled-light path handles
    static D3DXHANDLE ehTexLightGrid, ehTexLightIndexList, ehTileGridParams, ehTileGridParams2;
    static D3DXHANDLE ehTexgenTransform, ehBumpMatrix, ehBumpLumiScaleBias;
    static D3DXHANDLE ehPointLightMult;
    static D3DXHANDLE ehDebugLightCount;   // per-object light-count heatmap (debug)
    // Reflection-cache shadow fold (shadow atlas at s6 + applyCacheShadow gate)
    static D3DXHANDLE ehShadowAtlas, ehApplyCacheShadow;

    static float sunMultiplier, ambMultiplier;

    static ID3DXEffect* generateMWShader(const ShaderKey& sk);

public:
    // Debug visualizer: when true, renderMorrowind pushes the per-mesh selected light
    // count into the FFE shader's debugLightCount uniform, which overrides the pixel
    // output with a 0..kMaxIndicesPerMesh heatmap. Toggled per-frame (Numpad4) in
    // BeginScene. Shows the per-object light density both PPL and cache draws pay.
    static bool debugLightHeatmap;

    // Point-light routing for renderMorrowind. PerMesh = the per-mesh
    // selectTextureLights path (USE_TEXTURE_LIGHTS variant; default, used by
    // reflection + fallback). Tiled = the per-frame screen-tile grid
    // (USE_TILED_LIGHTS variant; main view only). The "which view" signal is
    // static at the call site, so callers pass it explicitly.
    enum class LightMode { PerMesh, Tiled };

    // Runtime A/B toggle for tiled lighting (VK_DECIMAL). Initialised to
    // Configuration.UseTiledLights at init; the master config flag still gates
    // (off => never tiled regardless of this). See tiledLightingActive().
    static bool tiledLightsActive;
    // Effective tiled-lighting state: master config flag AND the runtime
    // toggle. buildTileGrid + the main callers all key on this.
    static bool tiledLightingActive();

    static bool init(IDirect3DDevice* d, ID3DXEffectPool* pool);
    // Per-frame screen-tile light binning for the tiled path (main view only).
    // Hooked from DistantLand::frameSetupEarly after the scene-graph snapshot.
    // No-op when tiled lighting is inactive or there are no point lights.
    static void buildTileGrid();
    static void precacheAsync();
    // Block until the async precache thread (precacheAsync) has finished. Called before a device
    // Reset so ResetEx never overlaps D3DXCreateEffectFromFile on the precache thread (undefined:
    // concurrent device reset + effect creation corrupts the shared effect pool). No-op if idle.
    static void waitPrecache();
    static void updateLighting(float sunMult, float ambMult);
    // pointLightMult scales the point-light contribution (1 = normal). The cache
    // reflection pass fades it toward 0 at the cache->distant-land handover.
    // cacheBonePalette/cacheNumBones/cacheView drive the cache 32-bone indexed
    // skinning path (used by the cache reflection feed). When cacheBonePalette is
    // null, the rigid/reactive-skinning path is unchanged.
    // cacheWorldBoundsMin/Max override the per-mesh light-selection bounds for cache
    // draws (their VBs are D3DUSAGE_WRITEONLY, so computeBoundingBox can't read them
    // and would fall back to the object origin — the A/B light seam). Pass the cache
    // entry's true world AABB; null = reactive path (computeBoundingBox).
    // maxIndices caps the per-mesh selected-light count (default kMaxIndicesPerMesh
    // = 32). The cache reflection pass passes 8 — far fewer lights, so its heavy
    // per-mesh variant loops less; reflection seams from the tighter cap are
    // imperceptible. Only used on the PerMesh path (Tiled ignores it).
    static void renderMorrowind(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, float pointLightMult = 1.0f,
                                const D3DXMATRIX* cacheBonePalette = nullptr, int cacheNumBones = 0, const D3DXMATRIX* cacheView = nullptr,
                                const D3DXVECTOR3* cacheWorldBoundsMin = nullptr, const D3DXVECTOR3* cacheWorldBoundsMax = nullptr,
                                LightMode lightMode = LightMode::PerMesh,
                                unsigned int maxIndices = kMaxIndicesPerMesh,
                                // Cache path: the per-stage textures, passed directly so the
                                // effect's tex0..N are set without reading them back from the
                                // device (GetTexture per stage is a proxy round-trip the cache
                                // caller can avoid — it already knows the textures). When null,
                                // the reactive path reads device textures as before. Count is
                                // the stage count (== ShaderKey activeStages for cache draws).
                                IDirect3DBaseTexture9* const* cacheTextures = nullptr,
                                unsigned int cacheTextureCount = 0);
    // Shared texture-light selection (revision-keyed upload + view-keyed frustum
    // precull + per-mesh sphere-AABB nearest-kMaxIndicesPerMesh). Used by
    // renderMorrowind (objects) and the cache terrain pass so both pick lights with
    // byte-identical logic. Caller holds MGE::SceneGraph::SnapshotReadLock; device's
    // D3DTS_PROJECTION must match `view`. Returns the selected count; fills idxFloats
    // (>= kMaxIndicesPerMesh floats, packed as the lightIndices c-register layout).
    // maxIndices caps the selected count (clamped to kMaxIndicesPerMesh — the
    // idxFloats buffer + shader lightIndices[8] hold 32). 8 for reflections.
    static int selectTextureLights(const std::vector<MGE::SceneGraph::PointLight>& snapshotLights,
                                   unsigned int snapshotCount, const D3DXMATRIX& view,
                                   const D3DXVECTOR3& bMin, const D3DXVECTOR3& bMax,
                                   float* idxFloats, bool logPerf,
                                   unsigned int maxIndices = kMaxIndicesPerMesh);
    // Reflection-cache shadow fold: bind the sun-shadow atlas (sampler s6) and the
    // applyCacheShadow gate, shared-pool-propagated to every FFE variant. The cache
    // reflection color pass enables this around its draw loop so PerPixelPS darkens
    // shadowed fragments inline (replacing the separate receiver re-draw); disabled
    // (default) the branch is never taken and the main scene is byte-identical.
    // Caller binds shadowViewProj (reflected-view -> shadow clip), the reflected
    // sunVecView and per-draw shadowReflMult on the distant-land effect (shared).
    static void setCacheShadow(IDirect3DTexture9* atlas, bool enable);
    // Batched submit bracket. Between beginBatch()/endBatch(), renderMorrowind
    // holds an effect pass open across consecutive same-effect draws and only
    // CommitChanges per object, collapsing the per-draw Begin/End cost. Callers
    // must order their draw list by effect (texture/skin/fvf/mirror) so a pass
    // spans many objects. endBatch flushes the open pass and nulls the device
    // shaders; it is idempotent (safe on any exit path). batchEffectSwitches()
    // returns the effect-switch count accumulated since beginBatch (drawn ≫
    // switches confirms grouping worked).
    static void beginBatch();
    static void endBatch();
    static unsigned int batchEffectSwitches() { return s_batchSwitches; }
    static IDirect3DTexture9* textureLightData() { return texLightData; }
    static unsigned int maxTexLights()      { return kMaxTexLights; }
    static float        texLightTexelSize() { return 1.0f / (float)(kTexelsPerLight * kMaxTexLights); }
    static void release();
};
