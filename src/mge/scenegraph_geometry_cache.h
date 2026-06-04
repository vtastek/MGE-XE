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
        // Material (pointers into NI memory — valid for the session)
        IDirect3DTexture9* d3dTexture;  // null if no base texture
        const char*        textureName; // SourceTexture::fileName, null if none
        float alphaRef;
        bool  alphaTest;
        bool  blendEnable;
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
    // Layout: float3 pos, float3 normal(zeros), DWORD color(0xFFFFFFFF), float2 uv
    static constexpr unsigned int kVBStride = 36;
    static constexpr unsigned int kVBFVF    = 0x152; // XYZ|NORMAL|DIFFUSE|TEX1

    // Skinned vertex layout (SkinnedVertIn in the shaders): float3 pos,
    // float4 blendweights, UBYTE4 blendindices, float2 uv. Drawn with skinnedDecl().
    // kMaxBones must match MAX_BONES in "XE Common.fx".
    static constexpr unsigned int kSkinnedVBStride = 40;
    static constexpr unsigned int kMaxBones        = 32;

    // Vertex declaration for skinned VBs (created in init). Null until init runs.
    IDirect3DVertexDeclaration9* skinnedDecl();

    // Reverse map: IDirect3DTexture9* → SourceTexture::fileName.
    // DORMANT — the reverse map is currently unpopulated (the per-frame rebuild was
    // removed as unused; see scenegraph_geometry_cache.cpp). Always returns null until
    // a consumer repopulates it incrementally from extractMaterial.
    const char* resolveTextureName(IDirect3DTexture9* tex);

}
