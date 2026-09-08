// mgeHost64 — H1: the MIN-PYRAMID over the sky-height field (tasks/forge-heightfield-occlusion.md).
//
// `pSkyHeight` is a MAX-height raster: a texel that is half building and half street stores the
// ROOF. That is exactly right for its own consumers (sky AO wants the horizon, sunocc wants the
// blocker) and exactly wrong to march as an OCCLUDER, because it would block a ray that in fact
// passes down the street. A conservative occluder field has to be the other reduction: per coarse
// texel, the LOWEST surface anywhere inside its footprint. A ray above that is definitely
// unblocked; a ray below it is a candidate. `min` over a neighbourhood of max-texels IS what H0's
// floor arm gathered by brute force, and this pyramid is that same quantity, precomputed.
//
// ⚠ A SEPARATE TEXTURE, NOT MIPS ON pSkyHeight. Three reasons, all load-bearing:
//   1. pSkyHeight's existing consumers sample it and must keep getting the MAX. Mips on it would
//      hand a min to anything that ever took a non-zero LOD.
//   2. pSkyHeight is a RenderTarget — its clear/raster/RMW path (LOAD_ACTION_CLEAR + BM_MAX
//      blending + the terrain compute's read-modify-write) is defined over its shape.
//   3. This one is created through addResource with TEXTURE|RW_TEXTURE, so Forge gives it one UAV
//      per mip (mUAVMipSlice selects the level) and one SRV over the whole chain — the pHiz
//      arrangement, which is what the reduce needs.
//
// ⚠ LEVEL 0 IS THE MAX FIELD, COPIED VERBATIM, and that is the correct semantics rather than an
// inconsistency. "min over a footprint of max-texels" is what a conservative test wants at every
// level; at level 0 the footprint is one texel and the min of one sample IS the sample. It is also
// the built-in self-check the H0 probe already used from the other side (occProbeMinR=0 collapses
// the two arms).
//
// ⚠ THE -30000 SENTINEL NEEDS NO SPECIAL CASE, AND THAT IS LOAD-BEARING. SKY_HEIGHT_NONE ("no
// occluder here") is hugely negative, so it WINS a min: a footprint with a gap in it cannot claim
// to block anything. That is precisely the rule occprobe.comp.fsl states for its gather. Do not
// "fix" it into a skip — skipping the gap is asserting occlusion from missing data, the one
// direction a conservative field may never fail in.
//
// Update frequency = PerBatch, matching every other data-driven compute pass here (cull, froxel,
// shadowmask, linearize, skyheight) — they are neighbours in the same command list and the host's
// rebind cache compares the SET's gpu handle, not just the root index. Resource names are unique
// across all merged compute SRTs (house rule since the d3d.py aliasing bug).
#pragma once

BEGIN_SRT(SkyHeightMinSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_TEXTURE  (PerBatch, Tex2D(float),  gSkyMinSrc)     // pSkyHeight SRV (first pass only)
        DECL_RWTEXTURE(PerBatch, RTex2D(float), gSkyMinSrcMip)  // pyramid mip i-1 (reduce only)
        DECL_RWTEXTURE(PerBatch, WTex2D(float), gSkyMinDstMip)  // pyramid mip i (both passes)
    END_SRT_SET(PerBatch)
END_SRT(SkyHeightMinSrtData)
