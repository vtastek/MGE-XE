#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

struct IDirect3DDevice9;
struct IDirect3DVertexBuffer9;
struct IDirect3DIndexBuffer9;
struct IDirect3DTexture9;
struct IDirect3DVertexDeclaration9;

namespace MGE::GeometryCache {

    // Per-geometry entry keyed on NiTriShape* (cast to uint32_t on x86).
    // Non-skinned objects: VB holds model-space positions (kVBFVF); worldTransformD3D
    //   updated each frame; per-draw world*view palette applied in the renderers.
    // Skinned objects: STATIC bind-pose VB with per-vertex weights + bone indices
    //   (kSkinnedDecl); the bone matrices update per frame into bonePalette and the
    //   vertex shader skins (no per-frame CPU skinning, no per-frame VB rewrite).
    // Entries are created on first visit and evicted when the object is no longer
    // in the scene (skipped by a complete onFrameReady walk).
    struct CachedGeometry {
        // Double-buffered VB (kept for non-skinned model-space VBs; skinned VBs are
        // static so they only use slot 0). Use readVB() to fetch the draw slot.
        IDirect3DVertexBuffer9* vb[2];
        uint8_t  writeSlot;             // slot holding the most recently written VB
        IDirect3DIndexBuffer9*  ib;     // D3DFMT_INDEX16 triangle list (slot-shared)
        IDirect3DVertexBuffer9* readVB() const { return vb[writeSlot]; }
        uint32_t vertexCount;
        uint32_t triangleCount;
        float    boundsCenter[3];       // model-space bound center (for culling)
        float    boundsRadius;          // model-space bound radius
        // Tight model-space AABB (min/max over the mesh verts), captured at upload.
        // Transformed (8 corners) to a world AABB for point-light selection, matching
        // FixedFunctionShader::computeBoundingBox on the reactive path — the cache VB
        // is WRITEONLY so renderMorrowind can't walk it. Non-skinned only; skinned
        // entries keep the bone-derived sphere bound for light selection.
        float    aabbMin[3];
        float    aabbMax[3];
        uint16_t revisionID;            // GeometryData::revisionID at last upload
        bool     isSkinned;
        // Skinned: per-frame bone palette (model->world, 16 floats per bone) and
        // bone count. skinnedUnsupported set when numBones exceeds the shader palette
        // (kMaxBones) — such entries are skipped (reported), no CPU fallback.
        std::vector<float> bonePalette;
        uint32_t numBones;
        bool     skinnedUnsupported;
        uint8_t  dynamicHint;           // counts down from N when transform moves; 0 = static
        // True when the part's world/bone transform has negative determinant (a
        // mirrored left-side part). Clip-space winding is flipped, so the cache
        // depth/shadow draws must cull the opposite face for these.
        bool     mirrored;
        // True for entries walked from the world landscape (terrain) root. The
        // cache-driven color pass excludes these — terrain needs vertex colours +
        // texture splatting we don't synthesize yet (Phase 2); it stays on the
        // distant-land reflection path. Depth/shadow ignore this flag.
        bool     isLandscape;
        // True for entries walked from worldPickObjectRoot (dropped items, projectiles
        // etc.). That root is NOT traversed by the engine's world-camera occlusion
        // classify, so the Stage 2 engine-set cache cull keeps these via frustum
        // instead of dropping them for absence from the world classify.
        bool     isPickRoot;
        // Material (pointers into NI memory — valid for the session)
        IDirect3DTexture9* d3dTexture;  // null if no base texture
        // Terrain decal overlay (TexturingProperty maps[6] = DECAL_1): the second
        // land texture, blended over the base by the AlphaGrid in vertex-colour
        // alpha. Null for non-terrain / single-texture tiles. Drives the cache
        // terrain reflection's two-texture splat.
        IDirect3DTexture9* d3dOverlay;
        // Multi-map texturing (e.g. "Glow in the Dark" night windows): the base
        // map's DARK/DETAIL/GLOW siblings on the same NiTexturingProperty. Each is
        // null when absent; *UV is the cached UV set the map samples (clamped to
        // {0,1} — the VB carries at most a second UV set). The cache color pass
        // reconstructs the PPL fixed-function multi-stage blend from these
        // (MODULATE dark / MODULATE2X detail / ADD glow), matching the FFE JIT.
        // BUMP/GLOSS are env-map effects MW disables in fixed function — out of
        // scope; terrain DECAL_1 stays on d3dOverlay.
        IDirect3DTexture9* d3dDark;
        IDirect3DTexture9* d3dDetail;
        IDirect3DTexture9* d3dGlow;
        // Each map's true UV set (NI texCoordSet, clamped 0..3 — the FFE shader's
        // texcoordIndex is 2-bit and an FVF carries at most 4 sets). The cache color
        // pass sets each stage's texcoordIndex to these so a map samples its OWN set
        // (e.g. the "Glow in the Dark" detail map on set 2), matching PPL. Glow-mod
        // windows carry 3 UV sets: base(0), dark(1), detail(2).
        uint8_t baseUV, darkUV, detailUV, glowUV;
        // Non-skinned VB UV-set count + the derived stride/FVF. uvSetCount = the
        // highest UV set any present map uses + 1, bounded by the mesh's set count and
        // 4 (1 = ordinary single-UV geometry, the 99% case → stride 36). Every cache
        // draw path binds vbStride/vbFVF per entry so depth/shadow/color agree on the
        // layout. Skinned entries keep uvSetCount=1 (skinnedDecl has one UV set).
        uint8_t  uvSetCount;
        uint16_t vbStride;
        uint32_t vbFVF;
        const char*        textureName; // SourceTexture::fileName, null if none
        float alphaRef;
        bool  alphaTest;
        bool  blendEnable;
        // Material colours (RGBA) captured from the NI MaterialProperty on the
        // create/material-change path, for the cache-driven color pass
        // (Phase 0.5). Default to white diffuse/ambient, zero emissive when the
        // shape has no material. Depth/shadow ignore these.
        float matDiffuse[4];
        float matAmbient[4];
        float matEmissive[4];
        // Vertex colour usage. hasVertexColor: the mesh carries per-vertex colours
        // (filled into the non-skinned VB's DIFFUSE slot). vColSource: NI
        // VertexColorProperty::source — 0 ignore (vcol unused, constant material),
        // 1 emissive, 2 ambient+diffuse. The color pass uses vcol only when both say
        // so, else real material colours win (don't white-wash them).
        bool    hasVertexColor;
        uint8_t vColSource;
        // Lifecycle
        uint64_t lastFrame;             // frame counter from most recent visit
        // D3D row-major world transform for non-skinned objects (model-space VBs).
        // Cast to D3DXMATRIX* for use with D3DXMatrixMultiply.
        // Skinned objects are CPU-skinned to world-space; worldTransformD3D is unused for them.
        float worldTransformD3D[16];
    };

    // Must be called once before onFrameReady, with the D3D9 device.
    void init(IDirect3DDevice9* device);

    // Called once per frame from renderStage0. Walks the scenegraph
    // from the two world roots, creates/updates VBs for new/changed geometry,
    // and evicts entries not seen this frame. dataHandler is the
    // TES3::DataHandler* (typed void* to avoid TES3 header deps).
    void onFrameReady(void* dataHandler);

    const std::unordered_map<uint32_t, CachedGeometry>& cache();

    // Vertex buffer format used by each CachedGeometry::vb.
    // D3DFVF_XYZ | D3DFVF_NORMAL | D3DFVF_DIFFUSE | D3DFVF_TEX1
    // Layout: float3 pos, float3 normal, DWORD color(0xFFFFFFFF), float2 uv
    static constexpr unsigned int kVBStride = 36;
    static constexpr unsigned int kVBFVF    = 0x152; // XYZ|NORMAL|DIFFUSE|TEX1

    // Multi-map shapes carry extra UV sets. Per-entry stride = kVBStridePos (28:
    // pos+normal+color) + 8 bytes per UV set; FVF = kVBFVFBase | (uvSetCount <<
    // D3DFVF_TEXCOUNT_SHIFT). Computed into CachedGeometry::vbStride/vbFVF
    // (uvSetCount 1..4). Single-UV geometry resolves to kVBStride/kVBFVF.
    static constexpr unsigned int kVBStridePos = 28;
    static constexpr unsigned int kVBFVFBase   = 0x052; // XYZ|NORMAL|DIFFUSE (no TEX bits)

    // Skinned vertex layout (SkinnedVertIn in the shaders): float3 pos,
    // float3 normal, float4 blendweights, UBYTE4 blendindices, float2 uv, DWORD color.
    // Drawn with skinnedDecl(). kMaxBones must match MAX_BONES in "XE Common.fx".
    // Phase 2: color appended at offset 52 (depth/shadow skinned VS ignore it; the
    // FFE cache-skin color path reads it when the part uses vertex colour).
    static constexpr unsigned int kSkinnedVBStride = 56;
    static constexpr unsigned int kMaxBones        = 32;

    // Vertex declaration for skinned VBs (created in init). Null until init runs.
    IDirect3DVertexDeclaration9* skinnedDecl();

    // Reverse map: IDirect3DTexture9* → SourceTexture::fileName.
    // DORMANT — the reverse map is currently unpopulated (the per-frame rebuild was
    // removed as unused; see scenegraph_geometry_cache.cpp). Always returns null until
    // a consumer repopulates it incrementally from extractMaterial.
    const char* resolveTextureName(IDirect3DTexture9* tex);

}
