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
// frame, so the resolve has to carry real coverage through. It is saturated on the way out, because
// Catmull-Rom has negative lobes and coverage outside [0,1] breaks the composite.
//
// ⚠⚠ AND THE BUFFER IS PREMULTIPLIED, WHICH DECIDES HOW ALPHA IS FILTERED. `ONE/INVSRCALPHA` means
// `dst = src.rgb + dst*(1 - src.a)`: src.rgb is ALREADY scaled by src.a. RGB and coverage are two
// components of one premultiplied 4-vector, not two independent signals — so a linear filter must
// apply the SAME weight to all four, or the RGB:A ratio drifts and the composite reads the result
// at the wrong brightness.
//
// This was got wrong first time round, and the symptom is worth recording because it points
// straight at the cause: a **darker seam along the horizon fog band, and along anything melting
// into it**. That is where volfog and the sky write PARTIAL coverage — everywhere else alpha is a
// flat 0 or 1 and the error is identically zero, which is why it hid. Filtering RGB with the
// inverse-luminance weights while filtering alpha with the plain ones down-weights the bright sky
// samples in RGB only; alpha keeps the full average, so the pair lands darker than it should.
//
// The plausible-sounding argument for splitting them — "luminance-weighting a coverage value is
// meaningless" — is true of STRAIGHT alpha and false of premultiplied alpha. Do not re-split them.
//
// **THE RULE, stated once so it does not have to be rediscovered a third time: EVERY operation in
// this shader treats (rgb, a) as ONE premultiplied vector.** Both bugs found so far were the same
// mistake wearing different clothes, and both showed up at the horizon band because that is the
// only place in the frame where coverage is neither 0 nor 1:
//   1. WEIGHTS  — filtering rgb and a with different weight sets  -> a DARKER seam.
//   2. CLAMPING — saturate()ing a while letting rgb keep its ring -> a BRIGHTER seam.
//   3. TONEMAP  — the curve is NON-LINEAR, so tonemap(c·a) != tonemap(c)·a. Applying it to the
//                 premultiplied buffer directly re-tints every partially-covered pixel. Step 6a
//                 un-premultiplies, tonemaps, and re-premultiplies by the alpha that is ACTUALLY
//                 returned. Third instance, same band, same lesson.
// Anything that touches one component without the other will produce a fourth version of this.
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
    //     genuinely HDR — wired now so that switch is a format change and nothing else. It is also
    //     the A/B for a residual horizon-band tint that is NOT ringing: a luminance-weighted mean is
    //     pulled toward the darker samples, so a strong luminance gradient biases the result even
    //     with the premultiplied pair kept intact. Different mechanism, different fix.
    // y = the source is SCENE-REFERRED, so THIS PASS owns the tonemap (step 6a). The host's folded
    //     `g_hdrSceneColor && sampleCount > 1`, the same bit it publishes to gShadowParams.toneParams.x
    //     for the colour frags — one expression, two receivers, because this pass has a private SRT
    //     and cannot see gShadowParams. Scene-referred and fp16 are ONE switch: an un-tonemapped
    //     value exceeds 1.0 and a UNORM target would clamp it, which is worse than tonemapping in
    //     the pass. 0 = the frags already tonemapped and this pass must not touch the curve.
    // z = cubic SHARPNESS — Mitchell's C at B=0, i.e. the whole filter family on one knob:
    //     0.5 = Catmull-Rom (the shipped default, and what this pass has always done),
    //     0.4 = SMAA's filmic-reprojection value (Jimenez, SIGGRAPH 2016 course, p92),
    //     0.0 = pure cubic Hermite / smoothstep — NO negative lobes at all, i.e. no ringing and
    //           no apparent sharpening.
    //     Every value is a partition of unity (verified: |sum(w) - 1| < 1e-15 across c and f), so
    //     this trades ringing against sharpness and cannot change the image's energy.
    //     Negative lobe by value: 0.3 -> -0.044, 0.4 -> -0.059, 0.5 -> -0.074, 1.0 -> -0.148.
    //
    //     LIVE (dev panel) rather than a constant, for the reason the diameter is: a prior-art
    //     number was derived in someone else's pass at someone else's resolution, so port the
    //     mechanism and re-derive the value here ([[feedback_prior_art_constants_dont_transfer]]).
    //     ⚠ It is partly REDUNDANT with the diameter — both read as "sharper" — so move ONE at a
    //     time or neither can be attributed.
    //
    //     It matters more now than it did in LDR: negative lobes undershoot proportionally to the
    //     sample values, and the source is scene-referred (step 6a), so a bright HDR sample rings
    //     much further below zero than a [0,1] one ever could. The RGB floor at 0 then clips that
    //     undershoot asymmetrically — energy the overshoot side keeps. Lowering C is the direct
    //     lever on that, and the horizon fog band is where to look.
    // w = SCENE IS LINEAR (step 5). 1 = the source holds values proportional to radiance and this
    //     pass owns the compensating sRGB ENCODE, applied between the un-premultiply and the curve.
    //     It is the exact inverse of every decode the frame went through — hardware _SRGB views on
    //     the textures, decodeAuthored() on the cbuffer and vertex colours — which is what makes the
    //     migration checkable: any chain that is a pure product must come back unchanged.
    //     0 = the scene is still in MW's gamma domain and this pass must not encode.
    //
    //     ⚠ IT ALSO CHANGES THE FILTER, and deliberately. Both the Catmull-Rom reconstruction above
    //     and the inverse-luminance firefly weight now run on linear samples, which is where they
    //     were always supposed to run: averaging gamma values darkens an edge between two
    //     brightnesses, and `1/(1+luma)` was weighting an encoded number. Neither is a knob change,
    //     so do not re-tune the diameter or C at S1 — that is what makes them attributable at S2.
    DATA(float4, opts, None);
    // x = OUTPUT DITHER amplitude, in LSB of the 8-bit destination. 0 = off (bit-identical to before
    //     this lane existed). 1.0 is the textbook TPDF span; 0.5 is the "never touch a value that is
    //     already exactly on a code" variant — see rDitherTPDF() in resolve.frag for what that buys
    //     and costs. LIVE, because "is the grain visible" is a question about a screen.
    // y, z, w = reserved. y is the intended home for a frame counter if the dither is ever animated;
    //     it is deliberately static today (fixed-pattern noise does not shimmer on a still camera).
    DATA(float4, dither, None);
    // x = EXPOSURE E (tasks/forge-postprocess.md step 2 — the exposure servo). A scene-referred
    //     scale, so it lands inside the un-premultiplied window immediately BEFORE the encode and
    //     the curve: it multiplies linear radiance, which is the only domain a "stops of exposure"
    //     number means anything in. One multiply, one site, and every pass in the frame — sky,
    //     water, reflections, first person — inherits it by being CONTENT in the scene target
    //     rather than by carrying a lane of its own.
    //
    //     1.0 = identity. It is also identically 1.0 whenever opts.y is clear, because the LDR
    //     partner's byte-for-byte equality with the pre-6a build is load-bearing and adaptation is
    //     defined on scene-referred radiance anyway.
    //
    //     ⚠ IT BREAKS THE CLASS-2 ROUND TRIP, and that was decided rather than overlooked. Sky /
    //     statics_add / statics_blend write inverseTonemap(authored) so that tonemap() hands MW's
    //     authored value back unchanged; tonemap(E·inverseTonemap(a)) != a for any E != 1. The
    //     decision (2026-08-15) is that the sky JOINS the exposed world — a bright sky must be able
    //     to drive the meter it is most of — and the "old sky is the reference" requirement is
    //     re-read as a fixed-exposure reference SHOT rather than a runtime constraint.
    //
    //     ✅ AND THE LIFT IS GONE (step 3, below). It predicted its own death here and this is it.
    // y = THE CURVE SELECT (step 3 / S2). 0 = the legacy semi-HDR cubic in tonemap.h.fsl, 1 = AgX
    //     (agx.h.fsl). AgX is the default; the legacy curve is kept as the A/B and is BYTE-IDENTICAL
    //     when this lane is 0, which is the regression test that the swap touched nothing else.
    //
    //     ⚠ IT ALSO MOVES THE ENCODE, and that is not an option this lane leaves open. The legacy
    //     curve was FITTED against display-referred input, so opts.w's linearToSrgb runs BEFORE it;
    //     AgX eats linear scene radiance and returns linear display, so the OETF runs AFTER it. One
    //     branch owns the whole tail of PS_MAIN for that reason — a cbuffer bit is wave-uniform but
    //     not compile-time ([[project_uniform_branch_is_not_free]]), so it guards a block and never
    //     an instruction.
    //
    //     ⚠⚠ SECOND RECEIVER: gShadowParams.toneParams.w carries the SAME folded host expression to
    //     scenecolor.h.fsl's liftInPass(), which is disarmed by it. The class-2 lift is stated in
    //     terms of the LEGACY curve's inverse (inverseTonemap), so a frame running AgX with the lift
    //     still in it is not "the sky as a reference", it is broken. The two cannot be allowed to
    //     disagree, so they are one host expression — the same house rule opts.y follows, for the
    //     same reason (this pass has a private SRT and cannot see gShadowParams).
    // z, w = reserved (the servo's own state stays on the host — the shader only ever sees E).
    DATA(float4, tone, None);
    // AgX's LOOK TRANSFORM (agxLook) — an ASC CDL applied between the sigmoid and the outset matrix.
    // x = slope, y = power, z = saturation, w = offset.
    //
    // SHIPPED AT **BASE** — 1 / 1 / 1 / 0, which is an exact identity: slope 1 + power 1 + offset 0
    // make the ASC CDL an identity, and saturation 1 makes the SAT stage one. So the frame's shape
    // comes entirely from the inset/sigmoid/outset chain and the look is provably not carrying it.
    //
    // ⚠ AgX base IS flat and desaturated, by design, and that is the point of shipping it. The
    // blocker S2 exists to fix is a HUE failure, and contrast + saturation stacked on top of the
    // curve are exactly the two knobs that make a hue judgement unattributable. Blender's *Punchy*
    // (power 1.20, saturation 1.40) is two slider drags away when a look is actually being chosen.
    //
    // Fifth float4 = 80 B against a 256 B cbuffer, so still no allocation change.
    DATA(float4, look, None);
};

BEGIN_SRT(ResolveSrtData)
    BEGIN_SRT_SET(PerDraw)
        DECL_CBUFFER(PerDraw, CBUFFER(ResolveParams), gResolveParams)
        DECL_TEXTURE(PerDraw, Tex2DMS(float4, SAMPLE_COUNT), gResolveSource)
    END_SRT_SET(PerDraw)
END_SRT(ResolveSrtData)
