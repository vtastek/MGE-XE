// mgeHost64 — PROBE GI (tasks/forge-gi.md "Runtime probe GI v1"): the deferred probe the oracle
// passed with (probe-relight.py V1), at runtime.
//
// A camera-following grid of probe COLUMNS, G x G at `spacing`, two probes per column (layer 0 just
// above the lowest surface in the cell, layer 1 above the highest), stored TOROIDALLY: column (ix, iy)
// lives at storage (ix mod G, iy mod G), so the grid scrolls without moving data.
//
//   PLACE    one thread per column: min/max of the raw height map over the cell -> gPgiPlaceOut,
//            read back by the host (it needs probe positions to aim the capture). Where a static
//            floats over the TERRAIN with room beneath (a giant mushroom's cap, a deck), the floor
//            there is the terrain, not the static's top: layer 0 then stands UNDER the cover.
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
#include "atmosparams.h.fsl"   // AtmosphereParams: the relight samples the LIVE sky (RELIGHT skyLive)

STRUCT(PgiParams)
{
    DATA(float4, ambCol,  None);   // skyamb.h.fsl's gFrameData.ambCol (same name on purpose)
    DATA(float4, lodEye,  None);   // skyamb.h.fsl's gFrameData.lodEye: the ABSOLUTE eye
    DATA(float4, sunDir,  None);   // xyz = sun TRAVEL direction (to-sun = -sunDir)
    DATA(float4, sunCol,  None);   // xyz = sun colour (litTerm units)
    DATA(float4, grid,    None);   // x = spacing, y = G, z = relight batch phase, w = batch count
    DATA(float4, capPos,  None);   // GATHER: probe ABSOLUTE position xyz, w = face (0..5)
    DATA(float4, capTile, None);   // GATHER: atlas tile origin xy (texels), z = capture res (px), w = near
    DATA(float4, capMisc, None);   // GATHER: x = far, y = 1: also copy the raw capture to gPgiPeek at zw
    DATA(float4, place,   None);   // PLACE: xy = window's first column index (ix0, iy0); GATHER (H2-A): x = 1 keep the static face at row y, 2 merge over it (no G-buffer write)
    DATA(float4, terrain, None);   // PLACE: the LAND grid (skyheight.srt.h gridInfo): minX, minY, spanX, spanY
    // BALLS: x = camera balls, y = sun balls, z = frame seed, w = lift (u off the hit surface)
    DATA(float4, balls,   None);
    // BALLS: x = 1 the sun map is valid this frame (else the sky-top map stands in: overcast / no sun),
    // y = camera depth valid, z = sun-ball spread (u, the sky-top fallback), w = camera-ball spread (uv)
    DATA(float4, ballsMisc, None);
    DATA(float4x4, sunInv0, None);  // BALLS: inverse sun cascade 0 / 1 view*proj (camera-relative)
    DATA(float4x4, sunInv1, None);
    // SPLAT: vol xyz = the window's first voxel index (absolute), w = voxel size (u);
    // vol2 x = kernel R0, y = kernel Rk (radius = R0 + Rk x distance to the camera), z = update period
    // (frames), w = this update's temporal blend; vol3 x = phase, y = VX (= VY), z = VZ, w = full weight.
    DATA(float4, vol,  None);
    DATA(float4, vol2, None);
    DATA(float4, vol3, None);
    // RELIGHT: the sky from the atmosphere's sky-view LUT (the field skyhw.frag draws; its MIXED arm,
    // the clouds' mean cover included). x = 1 on (knob pgiSkyLive), y = native -> scene scale
    // (gSkyView.radiance.w), z = the camera's radius in the medium (gSkyView.sunDirW.w).
    DATA(float4, skyLive, None);
    // EMISSION (knob pgiEmis, P2): x = 1 the capture is the checkerboard (PGI_CAPTURE_EMIS: even pixels
    // albedo, odd the emitted radiance) and the gather splits it; RELIGHT: x = 1 add gPgiGBufE, y = gain.
    DATA(float4, emis, None);
    // RELIGHT FOG, energy only (knob pgiFog): x = MW fog start, y = fog end (u, the probe's distance to the
    // hit), z = 1 on; fogCol.rgb = the frame's fog colour in scene units (the main view's fogColNear).
    // fog.w (GATHER + RELIGHT) = knob pgiSplit, the depth ratio that splits a block in two (0 = off; gPgiGBuf2).
    DATA(float4, fog, None);
    DATA(float4, fogCol, None);
};

// POINT LIGHTS for the relight (knob pgiPoint): the renderer's NEAR live list (pLightCbv), read through
// the leading part of opaque.srt.h's LightData — the same bytes: lightParams (x count, y reach in radii),
// then per light pos.xyz + radius / colour / k0 k1 k2 + falloff.w = shadow slot + 1.
#ifndef MAX_POINT_LIGHTS
#define MAX_POINT_LIGHTS 128   // MUST match opaque.srt.h / IPC::kMaxPointLights
#endif
#ifndef POINT_LIGHT_TAIL
#define POINT_LIGHT_TAIL 0.75f // opaque.srt.h's: the 1 -> 0 ramp to the reach starts here
#endif
STRUCT(PgiLightData)
{
    DATA(float4, lightParams, None);
    DATA(float4, lights[MAX_POINT_LIGHTS * 3], None);
    // The rest of LightData's layout up to the G3 gobo tail, so a SLOTLESS light takes its fixture gobo
    // here as the frame's near frags do (gobosample.h.fsl fixtureGoboNear; pgipoint.h.fsl).
    DATA(float4, froxelDimsNear, None);
    DATA(float4, froxelZNear,    None);
    DATA(float4, goboNear[MAX_POINT_LIGHTS / 2], None);
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
        DECL_BUFFER   (PerBatch, Buffer(uint),  gPgiTerrainHeights)         // = gSkyTerrainHeights
        DECL_BUFFER   (PerBatch, Buffer(uint),  gPgiTerrainCellGrid)        // = gSkyTerrainCellGrid
        DECL_RWTEXTURE(PerBatch, RWTex2D(float4),       gPgiPeek)           // debug: raw captures (pgiDebug 8)
        DECL_TEXTURE  (PerBatch, Tex2D(float),  gPgiSunDepth)               // = gSunDepth (cascades side by side)
        DECL_TEXTURE  (PerBatch, Tex2D(float),  gPgiCamDepth)               // = pLinearDepth (device reverse-Z)
        DECL_RWBUFFER (PerBatch, RWBuffer(float4),      gPgiBallOut)        // BALLS: 2 per ball (pos+type, normal)
        DECL_RWTEXTURE(PerBatch, RWTex2DArray(float4),  gPgiVolOut)         // SPLAT: the volume (opaque.srt.h gPgiVol)
        DECL_RWTEXTURE(PerBatch, RWTex2DArray(uint),    gPgiVolTag)         // SPLAT: each cell's world voxel (0 = none)
        DECL_RWTEXTURE(PerBatch, RWTex2D(float4),       gPgiLit)            // debug: relit radiance per G-buffer texel
        DECL_RWBUFFER (PerBatch, RWBuffer(float4),      gPgiBlameOut)       // debug BLAME: 2 per probe slot (pgiBlame)
        DECL_RWBUFFER (PerBatch, RWBuffer(float4),      gPgiMoveOut)        // NUDGE: 2 per probe slot (relight -> host)
        DECL_CBUFFER  (PerBatch, CBUFFER(AtmosphereParams), gPgiAtmos)       // = pAtmosParamsCbv
        DECL_TEXTURE  (PerBatch, Tex2D(float4),  gPgiAtmosSky)              // = pAtmosSkyView (mixed field)
        DECL_TEXTURE  (PerBatch, Tex2D(float),   gPgiSunOcc)                // = gSunOcc: the long-range sun occlusion
        DECL_CBUFFER  (PerBatch, CBUFFER(PgiLightData), gPgiLights)         // = pLightCbv (the near live lights)
        DECL_TEXTURE  (PerBatch, Tex2D(float),   gPgiShadowAtlas)           // = pShadowAtlas (point shadows)
        DECL_TEXTURE  (PerBatch, Tex2D(float),   gPgiShadowAtlasDyn)        // = pShadowAtlasDyn
        DECL_RWTEXTURE(PerBatch, RWTex2D(uint),  gPgiGBufE)                 // G-buffer EMISSION lane (R11G11B10 packed), gPgiGBuf's layout
        // HI-RES WORLD PROBE (knob pgiHiRes): the cube's mip this dispatch writes, and the one it reads
        // (the mip above; mip 0's build reads the peek atlas's world-lit row instead). Per set instance.
        DECL_RWTEXTURE(PerBatch, RWTex2DArray(float4),  gPgiCubeOut)
        DECL_RWTEXTURE(PerBatch, RWTex2DArray(float4),  gPgiCubeIn)
        // FIXTURE GOBOS (G3) for the relight's slotless lights = g_gobo.pArray, bound once the bake lands
        // (pgiBindGobo). Unbound reads 0 = occlusion 0 = open: the pre-gobo relight, never a black light.
        DECL_TEXTURE  (PerBatch, Tex2DArray(float), gPgiGoboArray)
        // SPLIT BLOCKS (knob pgiSplit, fog.w): a G-buffer texel whose depths spread past the ratio is kept as
        // two surfaces — slice 0 the near one, 1 the far — each in gPgiGBuf's layout (x albedo rgb + w the
        // FAR share of the block's geometry, y normal, w count | plane distance f16 << 16). Slice 1's x.w = 0:
        // one surface, relit from gPgiGBuf as before.
        DECL_RWTEXTURE(PerBatch, RWTex2DArray(uint4), gPgiGBuf2)
    END_SRT_SET(PerBatch)
END_SRT(PgiSrtData)
