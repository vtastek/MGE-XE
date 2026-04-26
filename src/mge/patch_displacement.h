// Phase 7: near-camera landscape patch subdivision with pre-baked height offsets.
// For each selected Terrain patch (5x5 / 32 tris) within the near-set we produce
// a dense 65x65 (inner tier) or 33x33 (outer tier) subdivided copy.
// Each vertex carries two extra floats (baseH, overlayH) decoded once from the
// _paramh textures of the base and the absorbed overlay (may be null for
// non-overlay tiles). The VS lerps between them using input.color.a (the same
// AlphaGrid factor that drives the PS albedo blend), so the silhouette matches
// the material actually visible on the tile.
//
// Phase 8.2: perimeter verts facing a non-subdivided neighbor bake 0 on both
// heights so they meet the dropped -scale baseline and stay crack-free.
//
// Phase 8.7: perimeter verts facing a subdivided neighbor read from a coalesced
// per-frame edge-height map built by coalesceEdgeHeights(). Every near-set tile
// contributes samples along all 4 of its edges; adjacent tiles contribute at
// the same quantized world position, and the map averages them. Both sides then
// read the same value at their shared edge. Intermediate inner-tier verts with
// no matching outer-tier contributor are linearly interpolated between the two
// nearest matched positions — this makes inner's edge sit exactly on outer's
// piecewise-linear segment, eliminating LOD T-junction cracks.
#pragma once

#include "proxydx/d3d9header.h"
#include "ffeshader.h"

namespace PatchDisplacement {

using NearPatchEdgeHeights = FixedFunctionShader::NearPatchEdgeHeights;

struct SubdivPatch {
    IDirect3DVertexBuffer9* vb = nullptr;
    IDirect3DIndexBuffer9*  ib = nullptr;
    UINT stride = 0;
    DWORD fvf = 0;
    UINT vertCount = 0;
    UINT primCount = 0;
    IDirect3DBaseTexture9* overlayTexture = nullptr;

    ~SubdivPatch();
    SubdivPatch() = default;
    SubdivPatch(const SubdivPatch&) = delete;
    SubdivPatch& operator=(const SubdivPatch&) = delete;
};

// Phase 8.7: per-frame edge-height coalescing pass. Iterates every near-set
// Terrain call in `calls`, locks its source VB, samples per-edge heights at the
// tile's own density (inner 65 / outer 33), and accumulates into `outMap` keyed
// by quantized world (X, Y). After accumulation the map is normalized so each
// entry's avgBase/avgOverlay is the average across distinct contributing tiles.
//
// Must be called once per frame on the render thread BEFORE any getOrBuild,
// because getOrBuild reads this map to resolve shared-edge heights. Safe to
// re-enter across scenes since the map is cleared at the start.
void coalesceEdgeHeights(IDirect3DDevice9* device,
                         NearPatchEdgeHeights& outMap,
                         const std::vector<FixedFunctionShader::HLSLRecordedCall>& calls,
                         float heightScale,
                         float dispGamma,
                         float dispPivot);

// Cheap predicate: does this Terrain draw match the stride-4 5x5 contract that
// getOrBuild expects (25 verts / 32 tris / TRIANGLELIST / FVF carries
// POS+NORMAL+COLOR+UV)? Cell selectors should gate near-set inclusion on this so
// they don't enqueue meshes the validator will silently reject every frame.
bool canSubdivide(const RenderedState& rs);

// Build or return a cached subdivided near patch for `call`. Returns nullptr if
// the source VB/IB can't be locked this frame (retry next frame) or the patch
// geometry isn't a stride-4 5x5 grid.
//
// `edgeHeights` is the coalesced edge map produced by coalesceEdgeHeights for
// this frame; may be nullptr for diagnostic paths but normally required.
SubdivPatch* getOrBuild(IDirect3DDevice9* device,
                        const FixedFunctionShader::TerrainPatchKey& key,
                        const FixedFunctionShader::HLSLRecordedCall& call,
                        IDirect3DBaseTexture9* overlayTex,
                        float heightScale,
                        uint8_t subdivTier,
                        float dispGamma,
                        float dispPivot,
                        const NearPatchEdgeHeights* edgeHeights);

// Lookup-only sibling of getOrBuild. Returns the cached patch if all key params
// match (vb/ib/overlay/scale/tier/neighborMask/edgeContextHash/gamma/pivot),
// nullptr otherwise. Never builds. Used by both the depth prepass and the color
// replay so they can never race the single per-frame cache builder.
SubdivPatch* findCached(const FixedFunctionShader::TerrainPatchKey& key,
                        const FixedFunctionShader::HLSLRecordedCall& call,
                        IDirect3DBaseTexture9* overlayTex,
                        float heightScale,
                        uint8_t subdivTier,
                        float dispGamma,
                        float dispPivot);

// One-shot per-frame cache prep. Coalesces the edge-height map across all
// near-set displaced terrain calls, then calls getOrBuild for every near-set
// tile so the cache is fully populated before any consumer uses it. Both the
// depth prepass and the color replay then use findCached only — no builds,
// no edge-map mutations, so they cannot race each other or produce stale
// edge data. Call this from renderStage1 immediately after Hi-Z culling
// finalizes fb->recordedCalls and fb->nearPatches.
void prebuildNearPatches(IDirect3DDevice9* device,
                         FixedFunctionShader::FrameBuffer& fb,
                         float heightScale,
                         float dispGamma,
                         float dispPivot);

void clearAll();
void onVertexBufferReleased(IDirect3DVertexBuffer9* vb);
void onIndexBufferReleased(IDirect3DIndexBuffer9* ib);
void onTextureReleased(IDirect3DBaseTexture9* tex);

} // namespace PatchDisplacement
