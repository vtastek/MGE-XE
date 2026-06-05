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
        // Material (pointers into NI memory — valid for the session)
        IDirect3DTexture9* d3dTexture;  // null if no base texture
        // Terrain decal overlay (TexturingProperty maps[6] = DECAL_1): the second
        // land texture, blended over the base by the AlphaGrid in vertex-colour
        // alpha. Null for non-terrain / single-texture tiles. Drives the cache
        // terrain reflection's two-texture splat.
        IDirect3DTexture9* d3dOverlay;
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

    // Skinned vertex layout (SkinnedVertIn in the shaders): float3 pos,
    // float3 normal, float4 blendweights, UBYTE4 blendindices, float2 uv. Drawn with
    // skinnedDecl(). kMaxBones must match MAX_BONES in "XE Common.fx".
    static constexpr unsigned int kSkinnedVBStride = 52;
    static constexpr unsigned int kMaxBones        = 32;

    // Vertex declaration for skinned VBs (created in init). Null until init runs.
    IDirect3DVertexDeclaration9* skinnedDecl();

    // Reverse map: IDirect3DTexture9* → SourceTexture::fileName.
    // DORMANT — the reverse map is currently unpopulated (the per-frame rebuild was
    // removed as unused; see scenegraph_geometry_cache.cpp). Always returns null until
    // a consumer repopulates it incrementally from extractMaterial.
    const char* resolveTextureName(IDirect3DTexture9* tex);

}
