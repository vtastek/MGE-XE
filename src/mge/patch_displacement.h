// Phase 7: near-camera landscape patch subdivision with pre-baked height offsets.
// For each selected Terrain patch (5x5 / 32 tris) within the 2x2 camera ring we
// produce a dense 33x33 / 2048-tri subdivided copy.
// Each vertex carries two extra floats (baseH, overlayH) decoded once from the
// _paramh textures of the base and the absorbed overlay (may be null for
// non-overlay tiles). The VS lerps between them using input.color.a (the same
// AlphaGrid factor that drives the PS albedo blend), so the silhouette matches
// the material actually visible on the tile.
//
// Perimeter vertices of the 33x33 grid carry 0 on both heights so the rendered
// boundary sits exactly on the original patch's Z — no cracks with the
// non-subdivided neighbors that share those edge verts (spec §stride-4).
#pragma once

#include "proxydx/d3d9header.h"
#include "ffeshader.h"

namespace PatchDisplacement {

struct SubdivPatch {
    IDirect3DVertexBuffer9* vb = nullptr;   // vertCount verts, stride = origStride + 8
    IDirect3DIndexBuffer9*  ib = nullptr;   // primCount tris, 16-bit
    UINT stride = 0;                        // origStride + 8
    DWORD fvf = 0;                          // origFvf | D3DFVF_TEX2 (heights on TEXCOORD1)
    UINT vertCount = 0;                     // dst grid verts (1089 for 33x33)
    UINT primCount = 0;                     // dst grid tris  (2048 for 33x33)
    // For invalidation — if the overlay texture pointer flips we rebuild.
    IDirect3DBaseTexture9* overlayTexture = nullptr;

    ~SubdivPatch();
    SubdivPatch() = default;
    SubdivPatch(const SubdivPatch&) = delete;
    SubdivPatch& operator=(const SubdivPatch&) = delete;
};

// Build or return a cached subdivided near patch for `call`. Returns nullptr if
// the source VB/IB can't be locked this frame (retry next frame) or the patch
// geometry isn't a stride-4 5x5 grid.
//
// overlayTex may be null (non-overlay patch). Passing a different overlayTex
// than the cached entry invalidates and rebuilds.
//
// Thread: render/replay thread only (locks the device VB/IB).
SubdivPatch* getOrBuild(IDirect3DDevice9* device,
                        const FixedFunctionShader::TerrainPatchKey& key,
                        const FixedFunctionShader::HLSLRecordedCall& call,
                        IDirect3DBaseTexture9* overlayTex,
                        float heightScale,
                        uint8_t subdivTier,
                        float dispGamma,
                        float dispPivot);

// Evict all cached patches. Called from FixedFunctionShader::clearAllCellBatchCaches
// on cell / interior-exterior transitions, since landscape VBs are reallocated there.
void clearAll();

// Evict a single entry when its source VB/IB is being released.
void onVertexBufferReleased(IDirect3DVertexBuffer9* vb);
void onIndexBufferReleased(IDirect3DIndexBuffer9* ib);

// Evict _paramh height caches when a texture is released.
void onTextureReleased(IDirect3DBaseTexture9* tex);

} // namespace PatchDisplacement
