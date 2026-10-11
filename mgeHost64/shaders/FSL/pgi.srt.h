// mgeHost64 — PROBE GI (tasks/forge-gi.md "Runtime probe GI v1"): the deferred probe the oracle
// passed with (probe-relight.py V1), at runtime.
//
// A camera-following grid of probe COLUMNS, G x G at `spacing`, two probes per column (layer 0 just
// above the lowest surface in the cell, layer 1 above the highest), stored TOROIDALLY: column (ix, iy)
// lives at storage (ix mod G, iy mod G), so the grid scrolls without moving data.
//
//   PLACE    one thread per column: min/max of the raw height map over the cell -> gPgiPlaceOut,
//            read back by the host (it needs probe positions to aim the capture).
//   GATHER   after the water-mirror pass was re-aimed from a probe face (debug view 5 = ALBEDO) and
//            resolved: the 8x8 face tile of the probe's G-BUFFER (albedo, sky fraction, normal from
//            depth, hit distance) -> gPgiGBuf.
//   RELIGHT  one group per probe: each texel's radiance = sky (MW's sky SH shape x ambCol) or
//            albedo x (sun x N.S x HEIGHT-FIELD sun visibility + the renderer's own ambient at the
//            hit, skyamb.h.fsl's skyAmbFactorAO), projected to SH9, CONVOLVED (E/pi, the units a lit
//            frag's ambient term has) -> gPgiSHOut slices. Receivers: skyamb.h.fsl pgiIrradiance.
//
// gPgiSHOut layout (G x G per slice, RGBA16F): slice l*7 + k = SH plane k of layer l (27 floats =
// the 9 RGB coefficients flattened c0.rgb c1.rgb ..., 4 per plane), slice 14 + l = (z * valid,
// valid, 0, 0). Every value is PREMULTIPLIED by validity, so hardware bilinear over a half-built
// neighbourhood stays exact after dividing by the bilinear validity.
//
// Resource names are unique across all merged compute SRTs (house rule). skyamb.h.fsl is reached
// through #define aliases in pgi.comp.fsl, onto the resources below.
#pragma once
#include "shadowparams.h.fsl"

STRUCT(PgiParams)
{
    DATA(float4, ambCol,  None);   // skyamb.h.fsl's gFrameData.ambCol (same name on purpose)
    DATA(float4, lodEye,  None);   // skyamb.h.fsl's gFrameData.lodEye: the ABSOLUTE eye
    DATA(float4, sunDir,  None);   // xyz = sun TRAVEL direction (to-sun = -sunDir)
    DATA(float4, sunCol,  None);   // xyz = sun colour (litTerm units)
    DATA(float4, grid,    None);   // x = spacing, y = G, z = relight batch phase, w = batch count
    DATA(float4, capPos,  None);   // GATHER: probe ABSOLUTE position xyz, w = face (0..5)
    DATA(float4, capTile, None);   // GATHER: atlas tile origin xy (texels), z = capture res (px), w = near
    DATA(float4, capMisc, None);   // GATHER: x = far
    DATA(float4, place,   None);   // PLACE: xy = window's first column index (ix0, iy0)
};

BEGIN_SRT(PgiSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER  (PerBatch, CBUFFER(PgiParams),        gPgiParams)
        DECL_CBUFFER  (PerBatch, CBUFFER(ShadowMaskParams), gPgiShadow)     // = pShadowMaskParamsCbv
        DECL_TEXTURE  (PerBatch, Tex2D(float),  gPgiSkyAO)                  // = PerFrame gSkyHeight
        DECL_TEXTURE  (PerBatch, Tex2D(float),  gPgiSkyTop)                 // raw pSkyHeight (max)
        DECL_TEXTURE  (PerBatch, Tex2D(float),  gPgiSkyLow)                 // pSkyHeightLow
        DECL_TEXTURE  (PerBatch, Tex2D(float4), gPgiDummy4)                 // gSkyVisScreen / gGiField
        DECL_TEXTURE  (PerBatch, Tex2D(float4), gPgiReflColor)              // pReflectMips (mip 0)
        DECL_TEXTURE  (PerBatch, Tex2D(float),  gPgiReflDepth)              // pReflectDepthResolved
        DECL_BUFFER   (PerBatch, Buffer(float4), gPgiProbes)                // abs pos xyz, w = state
        DECL_RWTEXTURE(PerBatch, RWTex2D(uint4),        gPgiGBuf)
        DECL_RWTEXTURE(PerBatch, RWTex2DArray(float4),  gPgiSHOut)
        DECL_RWBUFFER (PerBatch, RWBuffer(float4),      gPgiPlaceOut)
    END_SRT_SET(PerBatch)
END_SRT(PgiSrtData)
