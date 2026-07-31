// mgeHost64 — SH2 top-down world HEIGHT MAP: the terrain-fill compute SRT (tasks/lighting.md).
//
// SH1 gave ambient a direction (which way is the sky brightest); it still assumes the WHOLE
// hemisphere is visible. Screen-space GTAO covers the fine end — contact shadows, a few hundred
// world units — and only reaches what is ON SCREEN. Nothing covered the band between the two:
// the building next to you, the canyon wall, Red Mountain. This map answers that band.
//
// It holds ONE world height per texel: max(terrain, statics), in a window that follows the camera.
// The receiver (skyamb.h.fsl) marches a few taps outward and takes the horizon elevation, so the
// whole feature costs one texture sample per tap and no scene traversal at all.
//
// TERRAIN ENTERS BY COMPUTE, NOT BY DRAWING. The heightfield is already resident on the GPU
// (gTerrainHeights + gTerrainCellGrid, uploaded once at first exterior), so one thread per texel
// reads it straight out — no cull, no instance build, no draw call, and "no LAND record here" is a
// natural sentinel rather than a hole to reason about. Drawing terrain geometry into the map would
// have needed a second terrainCullAndBuild against the map box for strictly less accuracy.
//
// The buffers are re-declared HERE rather than reached through the graphics PerFrame set: this is
// a COMPUTE root signature, and binding the same two host buffers into our own set is one
// DescriptorData pair. The alternative — pushing gTerrainHeights/gTerrainCellGrid into four more
// PerFrame sets — is exactly the plumbing that reads ZERO silently when one is missed
// ([[project_forge_perframe_set_instances]]: the same pair unbound in the mirror flattened the
// world to a black sheet at height 0).
#pragma once

// "No occluder here." NOT 0 — 0 is Morrowind's sea level and a perfectly ordinary terrain height,
// so a zero-cleared map would have every unwritten texel claim a wall at the waterline. Large and
// negative instead, which the receiver needs no branch to handle: the horizon slope it produces is
// hugely negative, so max(s, 0) discards it for free. Representable in R16_FLOAT (half tops out at
// 65504; precision there is ~32 units, which is meaningless for a value whose only job is to lose
// every max()).
#define SKY_HEIGHT_NONE (-30000.0f)

STRUCT(SkyHeightParams)
{
    // xy = ABSOLUTE world XY of the map's (0,0) texel CORNER, z = world units per texel,
    // w = map resolution in texels (square). The host publishes the same origin into
    // gShadowParams.skyAOMap, so map space is defined in exactly one place and switching this
    // pass to toroidal addressing later changes the host and one uv line, nothing else.
    DATA(float4, mapOrigin, None);
    // The LAND cell grid this host uploaded: x = gridMinX, y = gridMinY (cell coords, may be
    // negative), z = spanX, w = spanY. Cell indices are small integers and exact in float, so this
    // rides the cbuffer as float4 rather than dragging an int4 into the merged compute root sig.
    DATA(float4, gridInfo,  None);
};

BEGIN_SRT(SkyHeightSrtData)
    // ONE set holding CBV + SRVs + UAV, the shape gtao.srt.h had to collapse to before its UAV
    // write stopped vanishing. PerBatch is the frequency every other data-driven compute pass here
    // uses (cull, froxel, shadowmask, linearize) — they are neighbours in the same command list and
    // rebind correctly, because the host's cache compares the SET's gpu handle, not just the root
    // index.
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER  (PerBatch, CBUFFER(SkyHeightParams), gSkyHeightParams)
        DECL_BUFFER   (PerBatch, Buffer(uint),             gSkyTerrainHeights)   // 2 int16 per uint, 2113 uints/cell
        DECL_BUFFER   (PerBatch, Buffer(uint),             gSkyTerrainCellGrid)  // local (x,y) -> slot+1, 0 = no LAND
        DECL_RWTEXTURE(PerBatch, RWTex2D(float),           gSkyHeightOut)        // READ-modify-write: max() over statics
    END_SRT_SET(PerBatch)
END_SRT(SkyHeightSrtData)
