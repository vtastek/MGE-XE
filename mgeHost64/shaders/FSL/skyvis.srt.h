// mgeHost64 — SKY-VISIBILITY, SCREEN PASS (tasks/forge-skyao-oracle.md "TWISTER" S3).
//
// The K sky-visibility maps (one D16 reverse-Z depth map of the static world per fixed sky
// direction) are read ONCE PER PIXEL here, at half resolution, right after the depth prepass — not
// per fragment in every lit shader. Per fragment was +6 ms: every overdrawn layer and every
// statics/grass VERTEX paid 32 maps, and pixels past the maps' reach paid the loop AND the march.
// The lit paths then read this texture with four loads (skyamb.h.fsl skyAOTermPx).
//
// gSkyVisParams.tiles holds, per map, the three clip rows (xyz . r + w) and the map's eye relative to
// THIS frame's camera — uniform-indexed cbuffer reads, the cheap kind.
#pragma once

#define SKYVIS_MAX_DIRS 32

STRUCT(SkyVisParams)
{
    DATA(float4x4, invViewProj, None);   // device depth -> camera-relative world (the GTAO matrix)
    DATA(float4,   dims,        None);   // xy = full render w,h ; zw = half (output) w,h
    DATA(float4,   params,      None);   // x = K, y = map res, z = depth bias (map units), w = PCF radius (texels)
    DATA(float4,   params2,     None);   // x = reach: no map covers a point farther than this; yzw spare
    DATA(float4,   tiles[SKYVIS_MAX_DIRS * 4], None);
};

BEGIN_SRT(SkyVisSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER  (PerBatch, CBUFFER(SkyVisParams), gSkyVisParams)
        DECL_TEXTURE  (PerBatch, Tex2D(float),          gSvDepth)   // = pLinearDepth (device depth)
        DECL_TEXTURE  (PerBatch, Tex2DArray(float),     gSvMaps)    // = pSvmDepth
        DECL_RWTEXTURE(PerBatch, WTex2D(float4),        gSvOut)     // (vis, distance, coverage, 0)
    END_SRT_SET(PerBatch)
END_SRT(SkyVisSrtData)
