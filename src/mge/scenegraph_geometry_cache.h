#pragma once

#include <cstdint>
#include <unordered_map>

struct IDirect3DDevice9;
struct IDirect3DVertexBuffer9;
struct IDirect3DIndexBuffer9;
struct IDirect3DTexture9;

namespace MGE::GeometryCache {

    // Per-geometry entry keyed on NiTriShape* (cast to uint32_t on x86).
    // Non-skinned objects: VB holds model-space positions; worldTransformD3D updated
    //   each frame; per-draw world*view palette applied in the depth/shadow renderers.
    // Skinned objects: CPU-skinned to world-space each frame via NiSkinInstance bone
    //   matrices; depth/shadow uses palette[0]=gameView (no world transform needed).
    // Entries are created on first visit and evicted when the object is no longer
    // in the scene (skipped by a complete onFrameReady walk).
    struct CachedGeometry {
        // Double-buffered VB. onFrameReady writes the slot NOT being read by
        // the in-flight depth pre-pass (vb[writeSlot] from last frame), then
        // flips writeSlot so the shadow pass reads the fresh slot. This keeps
        // the depth/shadow CPU-GPU overlap without reallocating per frame.
        // Static (non-skinned) entries are written rarely and typically use a
        // single slot. Use readVB() to fetch the slot a consumer should draw.
        IDirect3DVertexBuffer9* vb[2];  // D3DFVF_XYZ, world-space positions
        uint8_t  writeSlot;             // slot holding the most recently written VB
        IDirect3DIndexBuffer9*  ib;     // D3DFMT_INDEX16 triangle list (slot-shared)
        IDirect3DVertexBuffer9* readVB() const { return vb[writeSlot]; }
        uint32_t vertexCount;
        uint32_t triangleCount;
        float    boundsCenter[3];       // model-space bound center (for culling)
        float    boundsRadius;          // model-space bound radius
        uint16_t revisionID;            // GeometryData::revisionID at last upload
        bool     isSkinned;
        uint8_t  dynamicHint;           // counts down from N when transform moves; 0 = static
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

    // Reverse map: IDirect3DTexture9* → SourceTexture::fileName.
    // Rebuilt each frame from surviving cache entries. Returns null if not found.
    const char* resolveTextureName(IDirect3DTexture9* tex);

}
