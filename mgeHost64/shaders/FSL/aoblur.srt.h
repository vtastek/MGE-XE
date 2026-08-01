// mgeHost64 — Tier 2 AO bilateral blur compute SRT.
//
// Third consumer of the shared ComputeRootSignature (after linearizedepth + gtao). Reads the raw
// GTAO output (gAOSrc = pAO) + the single-sample reverse-Z DEVICE depth (gBlurDepthIn =
// pLinearDepth) and writes a depth-aware (bilateral) blurred copy to gAODst (pAOBlur), which the
// colour frags then sample as gAO.
//
// Two deliberate distinctnesses (both learned the hard way — see gtao.srt.h):
//   * Update frequency = PerFrame, different from linearize's PerBatch and gtao's PerDraw, so the
//     host's descriptor-table rebind cache (keyed [pipelineType][rootIndex]) doesn't collide and
//     drop this dispatch's binds.
//   * Resource NAMES are unique across all three compute SRTs so the unioned compute.rootsig has
//     no aliasing.
//
// gBlurParams shares the SAME host buffer as gtao's gAOParams (pAOParamsCbv); the struct mirrors
// AOParams's byte layout (invViewProj + screenParams + two unused float4 standing in for aoParams
// and eyePos) then appends blurParams, so the host writes the blur knobs at float index 28..31 of
// the one cbuffer.
#pragma once

// MIRROR of gtao.srt.h's AOParams, byte for byte — same host buffer. Edit the two together.
STRUCT(BlurParams)
{
    DATA(float4x4, invViewProj,  None);   // 0..15  depth -> world (bilateral range weight)
    DATA(float4,   screenParams, None);   // 16..19 xy = w,h ; zw = 1/w,1/h
    DATA(float4,   _padAO,       None);   // 20..23 (aoParams slot — unused here)
    DATA(float4,   _padEye,      None);   // 24..27 (eyePos slot — unused here)
    DATA(float4,   blurParams,   None);   // 28..31 x = spatial sigma (px), y = range sigma (world u);
                                          //        zw belong to the AO pass (slice/step + thickness)
};

BEGIN_SRT(AOBlurSrtData)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(BlurParams), gBlurParams)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gAOSrc)
        DECL_TEXTURE(PerFrame, Tex2D(float),  gBlurDepthIn)
        DECL_RWTEXTURE(PerFrame, WTex2D(float4), gAODst)
    END_SRT_SET(PerFrame)
END_SRT(AOBlurSrtData)
