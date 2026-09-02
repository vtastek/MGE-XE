// mgeHost64 — CAMERA-ONLY MOTION VECTORS.  tasks/forge-upscale.md M1.
//
// WHY THIS IS ONE COMPUTE DISPATCH AND NOT TWELVE SHADER EDITS. A motion vector says where the
// surface under this pixel was on the previous frame's screen. For anything that did not MOVE IN
// THE WORLD, that is a pure function of the depth buffer and two camera matrices — so it can be
// reconstructed screen-space, after the fact, with no per-draw velocity output, no MRT on the
// colour pass, and no change to opaque / statics / terrain / grass / skin / alpha / water frags.
// That is the single biggest simplification available in this milestone and it is what makes M1
// tractable at all. What it CANNOT describe is anything that moved independently of the camera;
// those lanes are covered by the reactive mask instead (M1 step 3), and by per-object vectors later
// (M2).
//
// ⚠ IT READS gMvDepth, WHICH IS pLinearDepth, AND THAT NAME LIES IN OUR FAVOUR. linearizedepth's
// own header says it: "the 'linearize' naming follows the spec; the actual job is the MSAA resolve
// + SRV exposure", and the value it stores is the RAW REVERSE-Z DEVICE DEPTH at full R32F. So this
// pass gets exactly what it needs — device depth, single-sample, already resolved — and needs NO
// SAMPLE_COUNT variants of its own. Reading pDepth directly would have forced the sc1/sc4 split
// linearizedepth exists to absorb.
//
// ⚠ AND IT MUST BE THE SEAM REFRESH, NOT THE PRE-COLOUR SNAPSHOT. pLinearDepth is written twice a
// frame: once before the colour pass (the Z-prepass only) and once at the colour->water seam (the
// full scene, including DL statics, which are drawn straight into the colour pass and never reach
// the prepass). Reading the earlier one would give every distant building the sky's motion. The
// host adds this pass to the seam-refresh gate for that reason.
//
// ─── THE SPACE, AND THE HAZARD THAT LIVES IN IT ──────────────────────────────────────────────────
//
// invViewProj reconstructs into the payload's CAMERA-RELATIVE space: world minus `bakeEye`, the eye
// at the moment the client BUILT the payload. Every frame has its own bakeEye and they are not the
// same number — produce mode 3 (PARK, the default) builds frame N's payload against bakeEye_N and
// fires it at the start of N+1 with a restamped view.
//
// ⚠⚠ SO A `prevViewProj` THAT IS SIMPLY LAST FRAME'S MATRIX IS WRONG BY A WHOLE FRAME OF CAMERA
// MOTION. Last frame's matrix expects positions relative to bakeEye_{N-1}; this frame's
// reconstruction produces them relative to bakeEye_N. That is the same class of defect as
// [[project_park_restamp_eye_origin]], which cost five wrong theories before it was found, and it
// has the same signature: error proportional to Δeye/distance, exactly zero standing still, worst
// close in and at speed. **A motion-vector bug that is invisible when you stop moving is the worst
// possible shape for one**, because the natural way to check a still image is to hold still.
//
// The fix is host-side and this shader never sees it: `prevViewProjRel` is NOT last frame's viewProj,
// it is `Translate(bakeEye_N - bakeEye_{N-1}) * viewProj_{N-1}` — a matrix that takes THIS frame's
// camera-relative positions straight to LAST frame's clip space, with the origin change folded in.
// One matrix, one multiply here, and no way for a reader of this shader to reintroduce the bug.
//
// ─── JITTER CONVENTION ───────────────────────────────────────────────────────────────────────────
//
// Both matrices are the ones actually RENDERED WITH, so both carry their own frame's sub-pixel
// jitter and the vectors this pass writes are therefore JITTERED motion vectors: "where on the
// previous frame's screen was this surface", literally. That is the directly checkable statement,
// which is why it was chosen over the more common jitter-excluded convention.
// ⚠⚠ WHEN THE DLSS BACKEND LANDS IT MUST BE TOLD SO — `NVSDK_NGX_DLSS_Feature_Flags_MVJittered`.
// DLSS's DEFAULT is jitter-EXCLUDED vectors, so getting this wrong costs a sub-pixel error that
// looks like a slightly soft image rather than like a convention mismatch. Same note in upscale.cpp
// when it exists.
// ─── THE REACTIVE MASK (M1 step 3), AND WHY IT IS NOT DRAWN BY THE DYNAMIC LANES ─────────────────
//
// A reactive mask says "this pixel's motion vector is not to be trusted; weight the current frame".
// The plan proposed marking a STENCIL BIT during the dynamic passes (skin, alpha, water, glow,
// grass, first-person) and resolving it here. ⚠ THAT PREMISE IS FALSE IN THIS RENDERER: pDepth has
// NO STENCIL PLANE — the portal feature says so where it explains why IT needs a separate R8 target
// ("pDepth has no stencil plane, so the mask's stamp lands in this 1-bit R8 target"). And the
// fallback, a second render target on those passes, is not the cheap change it sounds like: their
// frags are hundreds of lines with `RETURN(float4(...))` at a dozen sites each, and replaying the
// draws instead means duplicating the skinned loop's 130-line window/palette/mirror state machine.
// Either way it is exactly the "twelve shader changes" this pass was designed to avoid.
//
// ⚠⚠ SO ASK THE QUESTION THE MASK IS ACTUALLY FOR. "Which lane drew this pixel" is a PROXY. The real
// question is "is this pixel's reprojection self-consistent" — and that is answerable here, from
// data this dispatch already has, for one extra texture and a handful of taps:
//
//     reproject the pixel into the previous frame, then ask the PREVIOUS FRAME'S DEPTH BUFFER
//     whether anything was actually at that distance there.
//
// If the surface was static and visible, it was: expected and sampled depth agree. If it moved
// independently of the camera, this pixel's vector points at where the BACKGROUND was, and the
// depths disagree. If it was hidden last frame (disocclusion), they disagree too — and disocclusion
// is a thing a mask must flag anyway, which the stencil scheme would have missed entirely.
//
// So one screen-space test covers every dynamic lane, disocclusion included, with NO change to any
// colour pass, no extra render target on one, and no draw replayed. It is also what FSR2 and friends
// actually do; the stencil route was the special case, not the general one.
//
// ⚠ WHAT IT CANNOT SEE: an object that moves so that its depth at the reprojected pixel matches what
// was there before — sliding sideways at constant depth across a wall at the same distance. Rare,
// and it degrades to "unmarked", i.e. the pre-mask behaviour. Per-object vectors (M2) are the real
// answer for those lanes; this is the cheap 90%.
#pragma once

STRUCT(MvParams)
{
    // clip -> THIS frame's camera-relative space. COPIED from the host's gShadowParams lane rather
    // than inverted again here, for apl.srt.h's reason: a second inversion proves its own arithmetic
    // and says nothing about whether the matrix reaching this dispatch is the one the DEPTH BUFFER
    // was rasterised with. One source, several receivers.
    DATA(float4x4, invViewProj, None);
    // THIS frame's camera-relative space -> LAST frame's clip. Origin change already folded in; see
    // the hazard note above. Identity-ish garbage on the first frame — `opts.x` is the gate.
    DATA(float4x4, prevViewProjRel, None);
    // x, y = RENDER rect in pixels; z, w = its reciprocal. ⚠ The RENDER rect, never the allocation
    // ([[project_forge_alloc_vs_render_uv]]) — the pixel<->NDC mapping has to be the one the raster
    // used, and every screen target here is alloc-sized with the frame drawn into a sub-rect of it.
    DATA(float4, screenParams, None);
    // ─── M1 STEP 3: THE REACTIVE MASK, DERIVED RATHER THAN DRAWN ────────────────────────────
    // x = disocclusion threshold (relative depth error at which a pixel starts to be reactive)
    // y = saturation threshold (relative error at which it is fully reactive)
    // z = reactive output gain (0 disables the mask entirely; the bit-identity arm)
    // w = reserved
    DATA(float4, reactive, None);
    // xy = 1/ALLOCATION size. ⚠ NOT the reciprocal of screenParams.xy above, and the two are not
    // interchangeable: this pass RECONSTRUCTS in render-rect pixels (screenParams) but SAMPLES the
    // previous depth, which like every screen target is alloc-sized with the frame drawn into a
    // sub-rect of it. Sampling with the render reciprocal reads never-written border.
    // [[project_forge_alloc_vs_render_uv]]. zw reserved.
    DATA(float4, allocParams, None);
    // x = PREVIOUS FRAME IS VALID. 0 on the very first frame and after any discontinuity (a device
    // reset, a resolution change, a teleport). Zero vectors are the honest answer there: an
    // accumulator told "nothing moved" reuses history it should not, but an accumulator handed a
    // garbage vector fetches from an arbitrary place, and only one of those degrades gracefully.
    //
    // y = THE CAMERA IS PARKED, tested BIT-IDENTICALLY. **NOT reserved** — this comment said it was
    // long after the lane was in use, which is drift that costs a diagnosis: the shader reads it
    // (motionvectors.comp.fsl carries the argument for why exact equality is the only test that
    // justifies the conclusion) and the host writes it from a 16-float compare against last frame's
    // matrix. ⚠ For three weeks the host then CLEARED it one statement later as part of a
    // reserved-lane sweep, so this lane was 0 in every shipped frame while the heartbeat reported
    // it set. Fixed in MB-2 step 0; the counter that proves it is `mv final: nonzero%` going to
    // 0.000% on a frame the log calls parked.
    //
    // z, w reserved.
    DATA(float4, opts, None);
};

BEGIN_SRT(MotionVectorSrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_CBUFFER  (Persistent, CBUFFER(MvParams), gMvParams)
        // pLinearDepth: single-sample RAW REVERSE-Z DEVICE depth, R32F. See the header note.
        DECL_TEXTURE  (Persistent, Tex2D(float),  gMvDepth)
        DECL_RWTEXTURE(Persistent, WTex2D(float2), gMvOut)
        // ─── THE FIELD'S OWN STATISTICS, so "is it grey when still" is a NUMBER ─────────────────
        // 5 uints: [0] max |mv|, [1] min |mv|, both as asuint bit patterns; [2] count of pixels
        // under 0.01 px; [3] pixels counted; [4] count marked reactive (> 0.5). Host clears, reads
        // back, divides.
        //
        // ⚠ WHY THIS EXISTS. The plan's acceptance test for this pass is "static world must read
        // ZERO motion while the camera is still" — and that test is BY EYE, which is precisely what
        // the bakeEye origin error defeats: a uniform DC offset over a moving field looks like a
        // moving field. A picture cannot distinguish "every pixel moved a little because the camera
        // moved" from "every pixel moved a little because the origin is wrong". Three numbers can.
        //
        // ⚠ MIN IS THE DIAGNOSTIC ONE, and it is not the obvious choice. A DC offset lifts the
        // WHOLE field, so the quietest pixel in the frame stops being quiet — min |mv| is exactly
        // the offset's magnitude. Max only says how fast you were going. (Under pure rotation there
        // is legitimately no still pixel, so min is read against a still or purely-translating
        // camera, not against every frame.)
        //
        // asuint on a NON-NEGATIVE float is monotonic in its bit pattern, so InterlockedMax/Min on
        // the raw bits is a correct float max/min — the standard trick, and the reason this needs no
        // SM6.6 float atomics.
        DECL_RWBUFFER (Persistent, RWBuffer(uint), gMvStats)
        // LAST frame's device depth, in LAST frame's camera-relative space — one R32 copy of
        // pLinearDepth taken at the end of this dispatch. See the reactive-mask note in the header.
        DECL_TEXTURE  (Persistent, Tex2D(float),  gMvPrevDepth)
        DECL_RWTEXTURE(Persistent, WTex2D(float), gMvReactive)
    END_SRT_SET(Persistent)
END_SRT(MotionVectorSrtData)
