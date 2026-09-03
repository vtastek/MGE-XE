// mgeHost64 — MB-2: THE MOTION BLUR FILTER (the consumer).  tasks/forge-postprocess.md.
//
// MB-1 is the producer and it is finished: the camera pass fills pMotionVectors, the object-velocity
// pass overwrites the movers, and the field is verified (`mv final:` / `mv parked latch:`). This is
// the pass that turns that field into an image.
//
// ── THREE PASSES, ALL COMPUTE ────────────────────────────────────────────────────────────────────
//
//   mbtilemax.comp      one GROUP per KxK tile  -> pMbTile      max |v| in the tile
//   mbneighbormax.comp  one THREAD per tile     -> pMbNeighbor  max over the 3x3 tile neighbourhood
//   mbgather.comp       one THREAD per pixel    -> pMotionBlur  a 1D line integral along v
//
// ⚠⚠ THE GATHER MUST NOT BE SEPARATED INTO X-THEN-Y. Motion blur is a **line integral along each
// pixel's own velocity**, not a 2D convolution: X-then-Y turns a diagonal streak into an
// axis-aligned box. `max` IS separable, which is exactly why the DILATION splits into two cheap
// passes and the gather does not. Getting that backwards is the one structural mistake available
// here and it is invisible in a screenshot taken while walking forward — take it while STRAFING.
//
// ⚠ COMPUTE, NOT GRAPHICS, AND FOR TWO REASONS. Shutter angle, tap count and K are precisely the
// knobs that want walking in a running game, and compute hot-reloads with F8 (bloom and reflectmip
// made the same call for the same reason). The second reason is specific to this milestone: a
// UAV-writing PIXEL shader has side effects, so D3D12 moves the depth test AFTER it — the late-Z
// trap that cost MB-1a three rounds of diagnosis ([[feedback_uav_pixel_shader_late_z]]). A compute
// dispatch cannot reach that failure at all.
//
// ── THE COST ARGUMENT ────────────────────────────────────────────────────────────────────────────
// The DX9 filter this replaces spends ~24 taps on EVERY pixel in `getDilatedVelocity` — a 12-tap
// cross at radius 24 px plus 12 depth taps — just to ask "is something fast moving near me". A tile
// max reads each source pixel exactly ONCE (K*K taps for K*K pixels = 1.0 taps/px, whatever K is),
// and the 3x3 neighbour pass adds 9/K^2 = 0.02 at K = 20. **~24 -> ~1.02, and it is strictly
// better**: a cross finds nothing on the diagonals, a tile max sees the whole neighbourhood.
//
// ── THE RECTS, WHICH ARE THE ONE PIECE OF ARITHMETIC THAT IS WRONG BY DEFAULT ────────────────────
// This pass runs AFTER the upscale, on the DELIVERED image, and the motion vectors are at the INPUT
// rect in INPUT-rect pixels. So every velocity read has to be
//   (a) FETCHED at the same normalised position — delivered pixel p maps to input texel
//       floor((p + 0.5) / delivered * input), and
//   (b) SCALED by delivered/input to become a displacement in DELIVERED pixels.
// At scale 1.0 that factor is exactly 1.0 and **proves nothing** — the identical trap
// [[project_forge_upscale_seam_4b]] records for the resolve, where `resolve.frag needed NO rescale`
// looked like a result and was an untested path. The test is `upscaleMode=2` (Quality), where a
// wrong out/in shows as a blur length wrong by exactly that ratio.
//
// ⚠ EVERY FETCH CLAMPS TO THE RENDER RECT, NEVER TO THE ALLOCATION. gSamplerBilinearClamp clamps to
// the RESOURCE, and every screen target here is alloc-sized with the frame drawn into a sub-rect of
// it, so the hardware clamp reads never-written border ([[project_forge_alloc_vs_render_uv]]). The
// clamp is done in pixel space, before the uv divide, in all three passes.
//
// ── WHAT THE DX9 FILTER GOT WRONG THAT THIS FIXES ────────────────────────────────────────────────
//  * `blur_scale = 0.1` is a raw multiplier on a per-frame displacement, i.e. **framerate-dependent**
//    — a blur tuned at 60 fps is half as long at 30. Replaced by a SHUTTER ANGLE: velocity is already
//    per-frame, so `v * (shutter/360)` is the displacement during the exposure and the result is
//    framerate-independent by construction. 180 degrees = half a frame = the film convention.
//  * The `/50`-then-signed-cube-root encode is gone; the field is dense RG16F.
//  * `nSamples` was computed from the velocity and then never used — the loop ran `MAX_SAMPLES`, so
//    it was a fixed 9-tap. Here the adaptive count is real: one tap per pixel of streak, clamped.
//  * `softDepthCompare` assumes a monotone-INCREASING linear depth. **pLinearDepth holds RAW
//    REVERSE-Z DEVICE depth despite its name** (stated at upscale.h:135 and forgerender.cpp:1934),
//    so ported straight across the sign is INVERTED and the filter would blur background over
//    foreground. See mbDepthWeight for the form that is correct AND needs no projection constants.
//
// ── THE DILATION SETS THE SEARCH; THE WEIGHTS SET THE ANSWER ─────────────────────────────────────
// ⚠⚠ THIS WAS SCOPED OUT AS "MB-3" AND THAT WAS WRONG, WHICH THE FIRST IN-GAME LOOK SETTLED IN ONE
// SENTENCE: *"motion blur in rectangular tiles instead of on object's own pixels."*
//
// The first build picked its direction with `|vTile| > |vSelf| ? vTile : vSelf`, transcribed from
// the DX9 filter. There the test means something, because that filter's "neighbour max" is a 5-TAP
// CROSS which can miss the true maximum. Here the tile max is a genuine max over a 3x3 tile
// neighbourhood **containing this pixel's own tile**, so `|vTile| >= |vSelf|` identically, the
// ternary can never select `vSelf`, and every pixel in a tile was blurred along that tile's dominant
// velocity. Not a halo around fast things — K-quantised RECTANGLES over the whole frame.
//
// The scoping note that shipped with it predicted "a halo around something fast", which understated
// it by the width of a tile. The lesson is narrower than "don't defer work": **a dilation is a SEARCH
// REGION, and a search region without an acceptance test is just a bigger answer.** The two halves
// are one mechanism and cannot be landed separately.
//
// So the gather now weights every tap by whether it could actually have reached the centre, from the
// tap's OWN velocity and the CENTRE's OWN velocity (McGuire 2012, `mbCone` / `mbCylinder` in
// mbcommon.h.fsl). It costs one extra RG16F load per tap. A static pixel sharing a tile with a fast
// mover still SEARCHES along the mover's direction — that is where the mover's colour legitimately
// comes from — and every tap scores ~0, so it keeps its own colour bit-exactly.
//
// The centre sample is in the sum with weight 1 always, which makes the denominator >= 1 by
// construction: no zero-weight branch, and a fully-disagreeing neighbourhood returns the source
// pixel exactly rather than approximately.
#pragma once

STRUCT(MotionBlurParams)
{
    // xy = THE DELIVERED RECT in pixels — the rect this pass reads, writes, tiles and clamps to.
    //      `upscaled ? out : in`, read from the host's ONE `delivered` derivation rather than
    //      re-deduced here (M1 4c's rule: two derivations of one rect are two things that can
    //      disagree).
    // zw = THE MOTION-VECTOR RECT in pixels — the INPUT rect, where the field was actually written.
    //      Equal to xy whenever no upscaler ran, which is why a 1x session cannot test the mapping.
    DATA(float4, rects, None);
    // x = TILE SIZE K, in DELIVERED pixels.
    //     ⚠ K IS ALSO THE MAXIMUM BLUR LENGTH, and that is a property of the algorithm rather than a
    //     second meaning bolted onto one knob. NeighborMax searches the 3x3 tile neighbourhood, so a
    //     tile can only be told about motion within +-K of itself; a streak longer than that would
    //     reach pixels whose tile never heard about it and would end in a hard edge. Clamping the
    //     shutter displacement to K is exactly the bound the 3x3 search buys, with a factor of two
    //     in hand (the streak spans +-len/2, so the radius is K/2).
    // y = tile-grid extent x = ceil(delivered.x / K)
    // z = tile-grid extent y = ceil(delivered.y / K)
    // w = OUT/IN SCALE = delivered.x / mvRect.x. Converts a vector written in INPUT-rect pixels into
    //     DELIVERED-rect pixels. Exactly 1.0 at 1x — see the header note on why that proves nothing.
    DATA(float4, tile, None);
    // x = SHUTTER FRACTION, shutter_degrees / 360. 0.5 = a 180-degree shutter = half a frame.
    // y = MAX TAPS. The adaptive count is min(ceil(len_px), this) with a floor of 3 — so a 4 px
    //     streak costs 4 taps and only genuinely fast motion pays the maximum.
    // z = VELOCITY FLOOR in DELIVERED pixels, below which the pass writes the source colour
    //     unchanged.
    //     ⚠⚠ MANDATORY, NOT AN OPTIMISATION. Only a BIT-IDENTICAL camera frame reaches exact zero
    //     (the parked short circuit in motionvectors.comp), and MW's camera matrix is bit-stable on
    //     a MINORITY of parked frames — measured worst deltas of 4.3e-4, 2.9e-11 and 1.5e-3 across
    //     three consecutive still frames, i.e. on most of them the camera really did move a hair and
    //     a small vector is the CORRECT answer. Without a floor every still frame gets a sub-pixel
    //     smear, and a still image that softens is the single most visible failure this pass has.
    // w = SOFT DEPTH EXTENT, as a FRACTION of the centre pixel's view distance (0.1 = "10% nearer").
    //     ⚠ FRACTIONAL AND NOT ABSOLUTE, because the comparison happens in reverse-Z device depth
    //     where an absolute epsilon means a different distance at every range. See mbDepthWeight.
    DATA(float4, blur, None);
    // Reserved. ⚠ CLEARED BY THE HOST **BEFORE** THE LANES IT WRITES, not after — see the MB-2 step 0
    // commit, where a trailing reserved-lane clear silently ate a flag the shader depended on and
    // the heartbeat went on reporting the flag it had computed.
    DATA(float4, opts, None);
};

BEGIN_SRT(MotionBlurSrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_CBUFFER (Persistent, CBUFFER(MotionBlurParams), gMbParams)
        // The finished field, at the INPUT rect in INPUT-rect pixels, previous-minus-current.
        DECL_TEXTURE (Persistent, Tex2D(float2), gMbVelocity)
        // ⚠ RAW REVERSE-Z DEVICE DEPTH. The name pLinearDepth lies in our favour: what the linearize
        // pass does is the MSAA resolve plus the SRV exposure, not a linearisation. Bigger = CLOSER.
        DECL_TEXTURE (Persistent, Tex2D(float),  gMbDepth)
        // The DELIVERED image: pSceneColor when nothing upscaled, the backend's output when it did.
        // Two set instances, one pointer different — the resolve's own trick at M1 4b, because a
        // frame that picks between two textures with a descriptor INDEX cannot get the two out of
        // step the way a mid-frame updateDescriptorSet can.
        //
        // ⚠ ITS rgb IS PREMULTIPLIED BY ITS ALPHA, so every tap weight here hits ALL FOUR channels
        // identically. This blur is a FILTER, not a radiometric scale — the rule stated in
        // resolve.srt.h, bloom.srt.h, reflectmip.srt.h and upscale.h, and got wrong at least twice.
        // Split them and the witness is a darker seam along the horizon fog band, the only place in
        // the frame where coverage is neither 0 nor 1 ([[project_forge_premultiplied_pair]]).
        DECL_TEXTURE (Persistent, Tex2D(float4), gMbColor)
        // The tile pyramid, twice — bloom.srt.h's gBloomSrc/gBloomDst arrangement exactly, and for
        // its reason: both are DESCRIPTOR_TYPE_RW_TEXTURE, both surfaces REST in UNORDERED_ACCESS,
        // and the passes are sequenced with plain UAV barriers. No SRV/UAV state ping-pong on a
        // surface that is read one dispatch after it is written, and therefore no window in which a
        // descriptor points at a resource in the wrong state.
        //
        // ⚠ R16G16_SFLOAT IS A **TYPED UAV LOAD**, which needs D3D12's TypedUAVLoadAdditionalFormats
        // tier. The bloom pyramid already UAV-loads R16G16B16A16_SFLOAT from the same set, and the
        // two formats are in the same all-or-nothing group, so this adds no new hardware requirement.
        DECL_RWTEXTURE(Persistent, RTex2D(float2),  gMbTileIn)
        DECL_RWTEXTURE(Persistent, WTex2D(float2),  gMbTileOut)
        // pMotionBlur. Its OWN target rather than an in-place rewrite of the source: a compute pass
        // that reads and writes one texture is a hazard with no barrier that can express it, and
        // pSceneColor may be MSAA, which cannot be UAV-written at all.
        DECL_RWTEXTURE(Persistent, WTex2D(float4), gMbOut)
        // ─── THE GATHER'S OWN STATISTICS, so "did a still frame get touched" is a NUMBER ─────────
        // 4 uints: [0] pixels that RAN the gather (above the velocity floor), [1] the sum of their
        // tap counts, [3] pixels the gather actually CHANGED — the two are different questions since
        // the agreement weights landed, and their DIVERGENCE is what says the tiling is gone (a
        // static pixel beside a mover runs and comes out identical). [2] the longest streak in
        // delivered px, as an asuint bit pattern (monotonic on
        // a non-negative float, so InterlockedMax on the raw bits is a correct float max — the trick
        // motionvectors.comp and mvfieldstats.comp both use, and the reason neither needs SM6.6).
        //
        // ⚠ WHY THIS EXISTS AT ALL. The acceptance test for the velocity floor is "a parked camera
        // must come out UNCHANGED", and this session has just spent a whole step establishing what
        // that costs when it is checked by looking: MB-2 step 0 was a three-week-old dead store that
        // survived because the claim was read rather than measured, and the counter that would have
        // caught it had to be built three times before it could fire. `blurred%` reads 0.000% on a
        // still frame or the floor is not doing its job — and the same number says what fraction of
        // the screen the gather is actually paying for when something IS moving, which is the other
        // question the heartbeat cannot answer from a millisecond alone.
        //
        // ⚠ ONE ATOMIC PER GROUP, NOT PER PIXEL. A 4.1 Mpixel dispatch hammering three addresses
        // would cost more than the filter it is measuring, and an instrument that changes the number
        // beside it is worse than none. The counts are tree-reduced in groupshared first, which is
        // why the gather has a single exit rather than the early RETURN()s it would otherwise want:
        // EVERY thread must reach EVERY GroupMemoryBarrier.
        DECL_RWBUFFER(Persistent, RWBuffer(uint), gMbStats)
    END_SRT_SET(Persistent)
END_SRT(MotionBlurSrtData)
