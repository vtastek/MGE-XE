// mgeHost64 — bloom pyramid compute SRT.  tasks/forge-postprocess.md step 4.
//
// WHY IT CAN EXIST NOW AND COULD NOT BEFORE. Three separate files record the same blocker:
//   scenecolor.h.fsl:5  — until step 6a every colour frag ended `c = tonemap(c)`, so pSceneColor held
//                         DISPLAY-referred values and "bloom had nothing to bloom from".
//   linearize.h.fsl:6   — bloom is a CONVOLUTION and blur(x^(1/g)) != blur(x)^(1/g), so a true 16:1
//                         highlight ratio reached the blur compressed to ~3.5:1 and read as flat
//                         haze rather than light spilling off something bright.
//   glow.frag.fsl:8     — the distant-light glow billboards already write un-clipped radiance and
//                         were explicitly waiting on "the HDR post path".
// All three have landed: pSceneColor is R16G16B16A16_SFLOAT, scene-referred, linear, premultiplied.
//
// ⚠ THERE IS NO RESOLVE/COMPOSITE SPLIT, and forge-postprocess.md said in three places that bloom
// would need one. It does not. The split exists to hand bloom a resolved, single-sample, linear
// image — but bloom's FIRST pass is a DOWNSAMPLE, and a downsample of an MSAA target already IS a
// resolve (reflectmipfirst was extended to exactly this at W5). So bloomprefilter reads pSceneColor
// directly and emits half-res mip 0 in one dispatch, resolve.frag keeps its whole existing tail and
// gains one texture read. What the split would have cost at 4096x3072 is one extra full-res fp16
// WRITE (~96 MB) plus one extra full-res READ for an image nothing else needs; what it buys is
// nothing a 2x downsample can tell apart from a box average.
//
// THREE PASSES, ONE SET, reflectmip's variant shape:
//   bloomprefilter_sc{1,4}.comp — pSceneColor (Tex2DMS) -> mip 0. 2x2 pixel footprint x SAMPLE_COUNT
//                                 samples. THIS PASS IS THE RESOLVE.
//   bloomdown.comp              — mip i-1 -> mip i, [1,3,3,1] separable tent.
//   bloomup.comp                — mip i+1 -> ACCUMULATED INTO mip i, symmetric tent, RWTex2D
//                                 read-modify-write (skyheight.srt.h:57's idiom).
//
// COMPUTE, NOT GRAPHICS, AND DELIBERATELY. Compute hot-reloads with F8; graphics load at host
// startup. Strength / radius / level count are exactly the knobs that want walking in a running
// game, which is the same call reflectmip.comp made for the same reason.
//
// pBloomMips is R16G16B16A16_SFLOAT regardless of sceneColorFormat, for reflectmip.srt.h's two
// reasons verbatim: B8G8R8A8_UNORM is not in D3D12's TypedUAVLoadAdditionalFormats set (so a UAV
// load of the source mip would not be legal at LDR), and it decouples the pyramid from any later
// format flip.
//
// ── THE TWO TRAPS ────────────────────────────────────────────────────────────────────────────────
//
// 1. ALLOC vs RENDER RECT ([[project_forge_alloc_vs_render_uv]]). pSceneColor is ALLOC-sized while the
//    scene draws into a width x height sub-rect; at render scale 1.00x that is a QUARTER of the
//    allocation and the rest holds last frame's texels. Every dispatch covers, and every tap clamps
//    to, the RENDER sub-rect at its own level — ceil(width / 2^(n+1)). Repeated ceil-halving equals
//    ceil(w / 2^n), so the chain is exact.
//
//    GetDimensions (hizreduce's trick, and reflectmip's) CANNOT supply this: it returns the
//    ALLOCATION dims. So every per-level descriptor set carries its OWN copy of this cbuffer with
//    the src and dst render rects written into it — 13 x 256 B, persistently mapped, rewritten per
//    frame. Explicit beats deriving the level from a dims ratio.
//
// 2. THE PREMULTIPLIED PAIR ([[project_forge_premultiplied_pair]] — this pass is the FOURTH place
//    the rule has had to be stated). pSceneColor's rgb is already scaled by its coverage.
//
//    ⚠⚠ THE RULE SPLITS IN TWO HERE, and getting the split wrong is silent:
//
//    FILTER WEIGHTS hit ALL FOUR CHANNELS IDENTICALLY. Every tent, every average, every Karis
//    firefly weight. reflectmip.srt.h's reason applies unchanged — an alpha-weighted downsample
//    rebuilds the dark-fringe bug one mip down, and the horizon fog band is the only witness.
//
//    RADIOMETRIC SCALES (the exposure multiply, the soft-knee threshold factor) hit **rgb ONLY**,
//    and alpha stays pure coverage. This is not an exception to the rule, it is what the rule
//    requires, and the algebra says so unambiguously. Write per sample (w*s*rgb, w*a):
//        numerator/denominator = sum(w*s*rad*a) / sum(w*a) = coverage-weighted mean of s*rad   ✓
//    Scale all four instead — (w*s*rgb, w*s*a) — and s CANCELS in the division:
//        sum(w*s*rad*a) / sum(w*s*a) = s-weighted mean of rad                                  ✗
//    i.e. a bloom that silently ignores both the exposure and the threshold. The distinction is
//    that w is a FILTER weight (it must not change what the pair means) while s is a change to the
//    RADIANCE, and in a premultiplied representation radiance lives in rgb alone.
//
//    resolve.frag then un-premultiplies with the FILTERED alpha, which is the correct inverse and
//    far more stable than the per-pixel alpha would be: a wide blur's alpha is a neighbourhood mean
//    coverage, ~1 wherever any neighbour is covered, and where the host covers nothing the numerator
//    is ~0 too — so the 1e-3 floor UNDER-adds rather than exploding. Fail-safe in the only direction
//    that matters.
//
// Frequency = Persistent, same as hizreduce / reflectmip. Several Persistent-frequency sets in one
// cmd are safe: cmdBindDescriptorSet's rebind cache is keyed on the GPU DESCRIPTOR HANDLE, not on
// the set object (Direct3D12.c:4685). Resource names are unique across all merged compute SRTs
// (house rule since the d3d.py aliasing bug), and there is exactly ONE SRT per header
// ([[project_forge_srt_one_per_header]]).
#pragma once

#ifndef SAMPLE_COUNT
#define SAMPLE_COUNT 1
#endif

STRUCT(BloomParams)
{
    // xy = the SRC extent for THIS level, in texels of the source surface, as a RENDER rect — not
    //      the allocation. Prefilter: the RASTER rect (pSceneColor's own written extent). Down/up:
    //      the source mip's own ceil-halved rect. It is the inclusive clamp bound minus one, i.e.
    //      every tap does clamp(q, 0, src.xy - 1), so no filter footprint can reach a texel this
    //      frame never wrote. GetDimensions cannot answer this — see the header note.
    //
    //      ⚠ FOR THE PREFILTER, src AND dst NO LONGER DIFFER BY EXACTLY 2 (M1 4c follow-up,
    //      tasks/forge-upscale.md). src is the RASTER rect while dst is sized from the DELIVERED
    //      rect, so `src/dst` is `2 x inputScale` and the pass is an AREA-AVERAGE RESAMPLE rather
    //      than a fixed 2x2 box. That split is deliberate and it is two different questions:
    //
    //        the pyramid's EXTENT follows the DELIVERED image, so its angular REACH does not change
    //        when a resolution slider moves (the defect 4c fixed: L7 -> L6 at inputScale 0.5);
    //
    //        the pyramid's DATA comes from the RAW SCENE, because bloom is an ENERGY operation and
    //        an upscaler is a DISPLAY RECONSTRUCTION carrying a non-energy-conserving anti-ringing
    //        clamp. Feeding bloom the upscaled frame cost the glow its punch on candle flames — see
    //        bloomprefilter.comp.fsl, and the same argument the HDR EXR dump's box resolve and
    //        `invLuma` shipping OFF are both made of.
    //
    //      At inputScale 1.0 the ratio is exactly 2 and the resample is bit-for-bit the old box.
    // zw = reserved.
    DATA(float4, src, None);
    // xy = the DST extent for THIS level, same convention. The dispatch is sized from it AND every
    //      thread re-checks it, because the groups are 8x8 and a rect of 12 does not divide: the
    //      overhang threads would otherwise write into the alloc region outside the render rect,
    //      which is exactly the stale-texel border verification step 6 looks for.
    // zw = reserved.
    DATA(float4, dst, None);
    // x = THRESHOLD in post-exposure scene-referred units. **SHIPS AT 0**, i.e. thresholdless, and
    //     the whole design rests on that: with a lerp composite and no threshold, energy in equals
    //     energy out, the result is exposure-invariant, and a uniformly bright sky blooms to itself
    //     and adds no haze because blur(const) == const. The knob is built so the question can be
    //     asked, not because the answer is expected to be non-zero.
    // y = soft-knee width. Inert while x is 0.
    // z = UPSAMPLE RADIUS in src texels — the "how wide is the glow" knob. 0 = a pure bilinear
    //     upsample (no smoothing); 1 = the reference tent. See bloomup.comp.fsl for the kernel.
    // w = THE PSF EXPONENT — the lerp factor b in `dst = lerp(dst, up(src), b)`. 0.5 ships.
    //     ⚠ IT MUST BE A LERP AND NOT AN ADD, and this is the load-bearing line of the whole
    //     design. `dst += up(src)` makes blur(const) = 7c over a 7-level chain, so a flat overcast
    //     sky would bloom to 7x itself and the composite would haze the frame by 1.36x — precisely
    //     the failure the thresholdless form exists to avoid. A lerp keeps every level's kernel a
    //     partition of unity, so the total pyramid kernel integrates to 1 and the composite adds no
    //     energy.
    //     ⚠⚠ AND IT IS THE PHYSICAL PARAMETER, which is why it is a knob rather than a constant.
    //     Octave n carries weight b^n(1-b) over radius 2^n, i.e. over solid angle ~4^n, so intensity
    //     per solid angle goes as b^n / 4^n; substituting theta = 2^n,
    //             I(theta) ~ theta^(log2(b) - 2).
    //     So b = 0.5 is **I ~ theta^-3**, which is the falloff Spencer et al. (1995) measured for
    //     human ocular glare — their PSF is a sum of theta^-2 and theta^-3 terms. The pyramid is not
    //     an approximation OF a physical glare kernel, it IS one, and this lane is its exponent
    //     (0.25 -> theta^-4 tight/lens-like, 0.71 -> theta^-2.5 wide veil). At 0.5 mip 0 ends up 1/2
    //     its own content + 1/4 mip1 + 1/8 mip2 + ... + 1/64 mip6 — a tight core with a wide skirt,
    //     which is what theta^-3 looks like assembled out of octaves.
    DATA(float4, filt, None);
    // x = INVERSE-LUMINANCE (Karis) prefilter weighting, 1/(1 + luma), on/off. **DEFAULT OFF.**
    //     ⚠ IT IS A DISPLAY RECONSTRUCTION AND THIS IS AN ENERGY OPERATION, which is the whole
    //     reason it is off. forge-postprocess.md:2016 had already drawn exactly this line for a
    //     different pass: the HDR EXR dump uses a hardware BOX resolve because "resolve.frag's
    //     Catmull-Rom + inverse-luminance firefly weighting is a display reconstruction and is
    //     deliberately not energy-conserving". A bloom that is meant to be PHYSICAL takes the
    //     energy-honest resolve for the same reason a measurement does.
    //     The magnitude is not marginal: at scene-referred luma 50 the weight is 1/51, so one lantern
    //     sample in a 16-sample footprint loses ~40x of its contribution — and a candle flame IS only
    //     a few pixels. Karis cannot separate "sampling noise" from "a small, genuinely bright light",
    //     and at HDR the second case is the entire point of the feature: a partially-covered sun-disc
    //     pixel really does emit that much light into that solid angle. In LDR the weight spanned only
    //     1.0..0.5, which is why reflectmip.comp.fsl:34 could flag the gap without resolving it.
    //     Kept as a switch because it answers the opposite question — if a lone specular sparkle on
    //     water reads as a hard dot rather than a glow, ticking it on says whether it is one
    //     sub-sample. Inert in the down/up passes, which are pure linear filters.
    // y = EXPOSURE E, the servo's current value (step 2). Applied in the PREFILTER, on rgb only, so
    //     the pyramid is stored in the same post-exposure domain resolve.frag's composite site sits
    //     in (immediately after `straight *= tone.x`). Two reasons it lands here rather than in the
    //     composite: the threshold above is then naturally in post-exposure units, and one multiply
    //     per quarter-res texel is cheaper than one per full-res pixel.
    // zw = reserved.
    DATA(float4, opts, None);
};

BEGIN_SRT(BloomSrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_CBUFFER (Persistent, CBUFFER(BloomParams), gBloomParams)
#if SAMPLE_COUNT > 1
        DECL_TEXTURE (Persistent, Tex2DMS(float4, SAMPLE_COUNT), gBloomSceneTex)  // pSceneColor (prefilter only)
#else
        DECL_TEXTURE (Persistent, Tex2D(float4), gBloomSceneTex)                  // pSceneColor (prefilter only)
#endif
        // The pyramid, twice. gBloomSrc is the level being READ (mip i-1 for the down pass, mip i+1
        // for the up pass); gBloomDst is the level being written. Declared RWTex2D rather than
        // WTex2D because the UP pass reads its destination back — it accumulates into a level the
        // down chain already filled. Both are DESCRIPTOR_TYPE_RW_TEXTURE either way, so one
        // declaration serves all three passes and DXC strips whichever half a variant never touches.
        DECL_RWTEXTURE(Persistent, RTex2D(float4),  gBloomSrc)
        DECL_RWTEXTURE(Persistent, RWTex2D(float4), gBloomDst)
    END_SRT_SET(Persistent)
END_SRT(BloomSrtData)
