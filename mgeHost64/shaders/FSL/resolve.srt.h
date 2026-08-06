// mgeHost64 — custom MSAA resolve SRT.  tasks/forge-postprocess.md step 4.
//
// WHY THIS PASS EXISTS AT ALL. A hardware ResolveSubresource cannot format-convert, so the moment
// pMSAAColor goes fp16 (step 6, HDR) the fixed-function resolve is *gone* — RGBA16F MSAA ->
// BGRA8_UNORM is not a legal resolve. The shader resolve is therefore not a quality bolt-on, it is
// HDR's delivery mechanism, and it is built FIRST, in LDR, with today's colours, so its cost can be
// measured against the 0.14-0.18 ms fixed-function baseline before anything about the look changes.
// It is also where fireflies get fixed, SPATIALLY — no velocity buffer, no jitter, no TAA history
// (MJP applies the same inverse-luminance weighting inside the spatial loop regardless of TAA).
//
// Ported from MJP's MSAAFilter 2.0 Resolve.hlsl; the SRT/vert shape follows The-Forge's own
// Examples_3/Visibility_Buffer/src/Shaders/FSL/Resolve.srt.h, which is the same pass upstream.
//
// gResolveSource is Tex2DMS, so this header is compiled once per SAMPLE_COUNT — the linearizedepth
// /apl idiom. There is no SAMPLE_COUNT == 1 variant on purpose: at 1x the scene renders straight
// into pRT and there is no resolve to do.
//
// ⚠ ALPHA IS LOAD-BEARING and the reference throws it away (`return float4(output, 1.0f)`).
// pRT's alpha is the PRESENT-SEAM COVERAGE MASK feeding an ONE/INVSRCALPHA composite over MW's
// frame. So resolve.frag carries a SECOND accumulator for alpha with the plain filter weights and
// NOT the 1/(1+luma) modulation — luminance-weighting a coverage value is meaningless — and
// saturates the result, because Catmull-Rom has negative lobes and coverage outside [0,1] breaks
// the composite.
#pragma once

#ifndef SAMPLE_COUNT
#define SAMPLE_COUNT 4
#endif

STRUCT(ResolveParams)
{
    // x, y = RENDER size in pixels, NOT the allocation size. The colour target is ALLOC-sized while
    //        the scene renders into a width x height sub-rect ([[project_forge_alloc_vs_render_uv]]);
    //        this is the clamp bound, so the filter footprint can never reach into texels the scene
    //        never wrote. Same trap the APL instrument documents.
    // z = filter diameter in pixels (MJP's ResolveFilterDiameter; he ships 2.0, exposes up to 6.0).
    // w = integer sample radius as a float — round(z/2), MJP's MSAAFilter.cpp:290. Handed in rather
    //     than recomputed so the loop bound is one cbuffer read. LIVE (dev panel), because ~200 MSAA
    //     loads/pixel at diameter 6 is the one number in the plan with no estimate behind it and the
    //     cost curve has to be walked in a running game, not guessed across rebuilds — graphics
    //     shaders do NOT hot-reload.
    DATA(float4, dims, None);
    // x = inverse-luminance (Karis) firefly weighting on/off. The A/B for "did it actually catch
    //     anything", and in LDR it is weak by construction: the samples are already tonemapped into
    //     [0,1], so 1/(1+luma) spans only 1.0..0.5. It earns its keep at step 6 when the source is
    //     genuinely HDR — wired now so that switch is a format change and nothing else.
    // yzw reserved.
    DATA(float4, opts, None);
};

BEGIN_SRT(ResolveSrtData)
    BEGIN_SRT_SET(PerDraw)
        DECL_CBUFFER(PerDraw, CBUFFER(ResolveParams), gResolveParams)
        DECL_TEXTURE(PerDraw, Tex2DMS(float4, SAMPLE_COUNT), gResolveSource)
    END_SRT_SET(PerDraw)
END_SRT(ResolveSrtData)
