// mgeHost64 — SKY-AO's SMOOTHED copy of the height map (tasks/forge-skyao-oracle.md, option 1).
//
// pSkyHeight is a MAX raster at 32 u/texel, so a diagonal wall is stored as a staircase, and a
// receiver marching it sees the staircase: that is the "square artifacts" the sky AO showed. The
// map itself cannot be smoothed in place — H1's min-pyramid (occlusion culling) and the long-range
// sun map need the raw max, which is CONSERVATIVE where a blur is not. So sky AO gets its own copy,
// Gaussian-smoothed once per rebuild (rebuilds are snap-cell crossings, not frames), and every
// PerFrame gSkyHeight bind points at it. Measured against the ground-truth oracle: the blur costs
// +0.006-0.009 floor MAE and removes the staircase.
//
// Unique resource names: fsl.py unions every compute shader into one root signature.
#pragma once

BEGIN_SRT(SkyHeightBlurSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_TEXTURE  (PerBatch, Tex2D(float),  gSkyBlurSrc)   // pSkyHeight (the raw max field)
        DECL_RWTEXTURE(PerBatch, WTex2D(float), gSkyBlurDst)   // pSkyHeightAO
    END_SRT_SET(PerBatch)
END_SRT(SkyHeightBlurSrtData)
