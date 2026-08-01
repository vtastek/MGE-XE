// mgeHost64 — Tier 2 depth-takeover: GTAO compute SRT.
//
// Second consumer of the ComputeRootSignature (shared, merged with linearizedepth.srt.h by fsl.py:
// each compute shader includes its own .srt.h, all declare ROOT_SIGNATURE(ComputeRootSignature),
// and the toolchain unions them into ONE compute.rootsig — mirrors 09_LightShadowPlayground).
//
//   PerFrame  gAOParams      : per-frame AO constants (invViewProj for depth->world, eye, knobs).
//   PerBatch  gLinearDepthIn : the single-sample reverse-Z DEVICE depth from the linearize pass.
//             gAOOut         : RGBA16F output — rgb = view-of-world bent normal, a = AO scalar (option A).
//
// PerBatch layout (SRV@0, UAV@1) matches linearizedepth.srt.h's PerBatch exactly so the merged
// root parameter is identical for both passes. GTAO reconstructs world position per tap from
// gLinearDepthIn + invViewProj (only the combined viewProj is available host-side; no separate
// view/proj, so world-space horizon search keyed off gAOParams.eyePos is the robust choice).
#pragma once

// SHARED BYTE LAYOUT with aoblur.srt.h's BlurParams — ONE host buffer (pAOParamsCbv) backs both
// gAOParams and gBlurParams. Growing this struct WITHOUT mirroring the change there silently
// corrupts the blur's knobs instead of failing to compile, so the two must be edited together.
// Float indices are host-side (forgerender.cpp writes ap[0..31]).
STRUCT(AOParams)
{
    DATA(float4x4, invViewProj,  None);   //  0..15 inverse of the reverse-Z world->clip (row-major bytes)
    DATA(float4,   screenParams, None);   // 16..19 xy = screen w,h ; zw = 1/w, 1/h
    DATA(float4,   aoParams,     None);   // 20..23 x = radius (world u), y = falloff, z = intensity, w = horizon bias
    DATA(float4,   eyePos,       None);   // 24..27 xyz = world camera position, w = SLICE COUNT
    DATA(float4,   sliceParams,  None);   // 28..31 xy = the BLUR's knobs (aoblur owns them), z = STEP COUNT,
                                          //        w = bitmask occluder thickness (world u; vbao/ssaofast)
};

// All three resources live in ONE PerBatch set (CBV + SRV + UAV), mirroring 09's
// ScreenSpaceShadows (CBV gSSSUniform + UAV gOutputTexture in one PerBatch). A SEPARATE PerFrame
// set for the cbuffer made the UAV write silently vanish on this host (linearize's single-set
// PerBatch UAV worked; gtao's two-set variant did not) — collapsing to one set fixes it.
// PerDraw (root param 0) — DELIBERATELY a different frequency than linearize's PerBatch (root 1).
// The gtao dispatch is the SECOND compute dispatch in the cmd; reusing linearize's PerBatch root
// slot made the host's descriptor-table rebind not take (cache keyed on [pipelineType][rootIndex]),
// so gtao's UAV write vanished while linearize's identical PerBatch write worked. PerDraw sidesteps
// the collision — mirrors 09's GaussianBlur, which writes its RWTextures from a PerDraw set.
BEGIN_SRT(AOSrtData)
    BEGIN_SRT_SET(PerDraw)
        DECL_CBUFFER(PerDraw, CBUFFER(AOParams), gAOParams)
        DECL_TEXTURE(PerDraw, Tex2D(float), gLinearDepthIn)
        DECL_RWTEXTURE(PerDraw, WTex2D(float4), gAOOut)
    END_SRT_SET(PerDraw)
END_SRT(AOSrtData)
