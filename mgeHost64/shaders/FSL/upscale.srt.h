// mgeHost64 — spatial upscale compute SRT.  tasks/forge-upscale.md M1 step 4b.
//
// The PASSTHROUGH backend of the IUpscaler seam (mgeHost64/upscale.h): pSceneColor's INPUT sub-rect
// resampled into a separate alloc-sized target's OUTPUT sub-rect. One dispatch, one set, one
// cbuffer.
//
// ⚠ Tex2D, NOT Tex2DMS, AND THAT IS ENFORCED HOST-SIDE. An upscaler REPLACES MSAA — they are
// mutually exclusive by construction, and `resolve_sc4.frag` binds a Tex2DMS where the
// single-sample upscaled output would go, which is a TYPE mismatch and not a quality question. The
// host refuses to create this backend at `sampleCount > 1` and says so in the log, so this header
// needs no SAMPLE_COUNT split and must never grow one.
//
// COMPUTE, NOT GRAPHICS, AND DELIBERATELY — the same call bloom.srt.h and reflectmip.srt.h made for
// the same reason. Compute hot-reloads with F8; graphics load at host startup. The FILTER is exactly
// the thing that wants walking in a running game, and this one is the permanent reconstruction for
// every GPU that will never run NGX, so it will be walked.
//
// Frequency = Persistent, like every other compute SRT in this tree. Several Persistent-frequency
// sets in one cmd are safe: cmdBindDescriptorSet's rebind cache is keyed on the GPU DESCRIPTOR
// HANDLE, not on the set object (Direct3D12.c:4685). Resource names are unique across all merged
// compute SRTs (house rule since the d3d.py aliasing bug) — note `gUpscale*` here versus
// aoupscale.srt.h's `gAOUp*`, which are a DIFFERENT pass entirely — and there is exactly ONE SRT
// per header ([[project_forge_srt_one_per_header]]).
#pragma once

STRUCT(UpscaleParams)
{
    // xy = the INPUT rect in texels — where the scene actually rasterised. It is the inclusive clamp
    //      bound minus one for every tap, i.e. `clamp(q, 0, in.xy - 1)`.
    //
    //      ⚠ THE CLAMP IS THE **RENDER** RECT AND NEVER THE ALLOCATION
    //      ([[project_forge_alloc_vs_render_uv]]). pSceneColor is alloc-sized with the scene drawn
    //      into a sub-rect; at render scale 1.00x that sub-rect is a QUARTER of the surface and
    //      everything outside it holds LAST FRAME'S texels. A footprint that reaches one texel past
    //      the edge picks up a stale frame, and at the frame's border — where a 4x4 cubic footprint
    //      always reaches — that is a one-pixel ghost rim, which reads as "the upscaler is soft"
    //      rather than as an out-of-bounds read. GetDimensions cannot answer this: it returns the
    //      ALLOCATION dims, which is why the rect rides a cbuffer.
    // zw = 1 / xy.
    DATA(float4, inRect, None);
    // xy = the OUTPUT rect in texels — the extent this pass WRITES, and therefore the extent
    //      `gResolveParams.dims.xy` becomes on an upscaled frame (the resolve's source clamp is a
    //      statement about the source's written extent, not about the surface).
    //      The dispatch is sized from it AND every thread re-checks it, because the groups are 8x8
    //      and a rect of 1050 does not divide: the overhang threads would otherwise write into the
    //      alloc region outside the render rect — the stale-texel border again, one pass earlier.
    // zw = 1 / xy.
    DATA(float4, outRect, None);
    // x = CUBIC SHARPNESS — Mitchell's C at B = 0, i.e. the whole filter family on one knob, and
    //     deliberately the SAME parameterisation resolve.frag.fsl's reconstruction uses so the two
    //     filters in the frame can be reasoned about in one vocabulary:
    //        0.5 = Catmull-Rom (the shipped default),
    //        0.4 = SMAA's filmic-reprojection value (Jimenez, SIGGRAPH 2016 course, p92),
    //        0.0 = pure cubic Hermite — NO negative lobes, i.e. no ringing and no sharpening.
    //     Every value is a partition of unity, so this trades ringing against sharpness and cannot
    //     change the image's energy.
    // y = ANTI-RINGING CLAMP on/off.
    //     ⚠⚠ IT IS REQUIRED, NOT OPTIONAL, AND 0 IS A DIAGNOSTIC ARM RATHER THAN A SETTING.
    //     Catmull-Rom UNDERSHOOTS, and this source is scene-referred and unbounded — a lantern at
    //     luma 400 in a negative lobe rings hundreds of units BELOW ZERO, where the surrounding
    //     content is at 0.1. resolve.srt.h warns about exactly this one pass downstream ("negative
    //     lobes undershoot proportionally to the sample values, and the source is scene-referred, so
    //     a bright HDR sample rings much further below zero than a [0,1] one ever could"), and the
    //     rgb floor then clips that undershoot ASYMMETRICALLY — energy the overshoot side keeps.
    //     The knob exists so the artefact can be SEEN once, not so it can be left off.
    // zw = reserved.
    DATA(float4, opts, None);
};

BEGIN_SRT(UpscaleSrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_CBUFFER  (Persistent, CBUFFER(UpscaleParams), gUpscaleParams)
        DECL_TEXTURE  (Persistent, Tex2D(float4),   gUpscaleSrc)   // pSceneColor (INPUT rect inside it)
        DECL_RWTEXTURE(Persistent, WTex2D(float4),  gUpscaleDst)   // the backend's own OUTPUT target
    END_SRT_SET(Persistent)
END_SRT(UpscaleSrtData)
