// mgeHost64 — the AERIAL-PERSPECTIVE volume's SRT (tasks/forge-atmosphere.md S3.3).
//
// Its own header, not a fifth instance of atmosphere.srt.h: this pass needs a second cbuffer (the
// camera's froxel rays) and writes a different-shaped output, and one SRT per header is the rule —
// two in one file alias each other's registers and DXC dead-strips the evidence.
// [[project_forge_srt_one_per_header]]
//
//   atmos_aerial.comp   writes gAtmosAerialOut (1024x64 atlas)   reads both LUTs + both cbuffers
//
// THE ATLAS. Slice k (of dims.z = 32) occupies columns [32k, 32k + 32); rows 0..31 hold the IN-SCATTERED
// radiance (native units, the LUTs' own) accumulated from the eye to the slice's far distance, rows
// 32..63 the CHROMATIC transmittance over the same path. A 2D atlas rather than a Texture3D because FSL's
// 3D/array sampling has gaps this repo has hit three times ([[project_forge_fsl_lvl_array_sampler_typed_3d]]):
// an atlas is a plain RWTex2D to write, a plain Tex2D to read, and its debug view is the texture itself.
#pragma once

#include "atmosparams.h.fsl"   // AtmosphereParams — the struct only

STRUCT(AtmosAerialParams)
{
    DATA(float4, ray00, None);  // xyz = camera-relative world direction through the render rect's
                                //       TOP-LEFT corner (unnormalised; interpolates exactly under a
                                //       perspective projection), w = D, the volume's far distance (MW units)
    DATA(float4, rayDx, None);  // xyz = top-right minus top-left, w = MEDIUM metres per MW unit
                                //       (kMwUnitToMetre x the S3.2 world scale)
    DATA(float4, rayDy, None);  // xyz = bottom-left minus top-left, w = the camera's radius in the
                                //       scaled medium (m from the planet centre)
    DATA(float4, dims,  None);  // x = froxels across, y = froxels down, z = slices, w = sub-steps per slice
};

BEGIN_SRT(AtmosAerialSrtData)
    BEGIN_SRT_SET(PerBatch)
        // The SAME params cbuffer the LUT chain cooks from — bound, not copied, so the fog is
        // integrated from exactly the medium the sky was.
        DECL_CBUFFER  (PerBatch, CBUFFER(AtmosphereParams),  gAtmosParams)
        DECL_CBUFFER  (PerBatch, CBUFFER(AtmosAerialParams), gApParams)
        // Read by atmossource.h.fsl BY NAME — these two names are that header's contract.
        DECL_TEXTURE  (PerBatch, Tex2D(float4),              gAtmosTransmittance)
        DECL_TEXTURE  (PerBatch, Tex2D(float4),              gAtmosMultiScatter)
        DECL_RWTEXTURE(PerBatch, WTex2D(float4),             gAtmosAerialOut)
    END_SRT_SET(PerBatch)
END_SRT(AtmosAerialSrtData)
