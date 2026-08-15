// mgeHost64 — APL (Average Picture Level) instrument SRT.  tasks/forge-postprocess.md step 2.
//
// This is a MEASURING STICK, not a feature. The linear-space migration (step 5) has to prove it is
// neutral — "match APL, colours and hues" — and that is not a judgement anyone can make by eye
// across two builds a week apart. This pass reduces the finished scene colour to four numbers per
// frame so the perf harness records them automatically:
//
//     mean R, mean G, mean B, mean log(luma)
//
// mean RGB gives the literal APL (its luma) AND the average hue (the channel ratios), which is the
// "colours and hues" half. mean log(luma) is the geometric-mean luminance — the standard exposure
// metric, and later the input to the ported Eye Adaptation (step 7), so the instrument is not
// throwaway.
//
// IT IS ON BY DEFAULT, DELIBERATELY. An instrument that ships switched off produces no BEFORE
// numbers, and the before numbers are the entire point — they have to come from runs made now,
// while the renderer is still gamma-space. Cost is one 256-thread group (see apl.comp.fsl).
//
// gAplColor's type switches on SAMPLE_COUNT (Tex2DMS vs Tex2D), the linearizedepth.srt.h idiom;
// both are one SRV slot, so the merged compute.rootsig is identical across the two variants.
#pragma once

#ifndef SAMPLE_COUNT
#define SAMPLE_COUNT 1
#endif

STRUCT(AplParams)
{
    // x, y = image size in pixels. z = sample-lattice side (samples per axis).
    // w = 1 / (z*z), the LATTICE's reciprocal count. ⚠ NO LONGER THE MEAN'S DIVISOR — see opts.x:
    //     with sky rejection on, the number of samples that COUNT is decided per frame, so the mean
    //     divides by the counted total that comes back in gAplOut[7]. Kept because it still names
    //     the lattice, and because a frame with rejection off must divide by exactly this.
    // float4 and NOT uint4, deliberately: skyheight.srt.h records that int4/uint4 was kept OUT of
    // the merged compute root signature. These are small integers, exact in float.
    DATA(float4, dims, None);
    // x = REJECT SKY (0/1). y,z,w spare.
    //
    // ⚠ THIS FLAG REDEFINES EVERY NUMBER THE INSTRUMENT PRODUCES, which is why it is a published
    // flag and not a compile-time choice: with it on, `apl`, the percentiles and therefore the
    // exposure servo all describe the SCENE, and with it off they describe the FRAME. Two readings
    // taken under different settings are not comparable and the heartbeat says which is which.
    // The reason it exists (user, 2026-08-16): MW's sky is reproduced authored data, not a physical
    // radiance, and it covers a camera-pitch-dependent fraction of the frame — so metering on it
    // makes exposure a function of where the player is LOOKING. Measured: two exterior frames
    // seconds apart read mean=104 (p90=160, sky in shot) and mean=41 (p90=60, sky out), swinging E
    // by 28% with the scene unchanged.
    DATA(float4, opts, None);
};

BEGIN_SRT(AplSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER(PerBatch, CBUFFER(AplParams), gAplParams)
#if SAMPLE_COUNT > 1
        DECL_TEXTURE(PerBatch, Tex2DMS(float4, SAMPLE_COUNT), gAplColor)
#else
        DECL_TEXTURE(PerBatch, Tex2D(float4), gAplColor)
#endif
        // pLinearDepth — the RAW reverse-Z device depth of sample 0, and the sky discriminator.
        //
        // ⚠ NOT SAMPLE_COUNT-SWITCHED, unlike gAplColor: the linearize pass already resolved sample
        // 0 into a single-sample R32F, so this is Tex2D in every variant. It is also written
        // UNCONDITIONALLY every frame — "linearize always runs; only the GTAO dispatches are gated"
        // (forgerender.cpp) — so this instrument does not acquire a hidden dependency on AO being
        // enabled, which would have made the metering change under a knob that has nothing to do
        // with it. Same valid extent as pRT (both are read at g_live.width/height), so one integer
        // pixel coordinate addresses both and no second size has to be published.
        //
        // SKY IS EXACTLY 0.0 HERE, not approximately: pSkyPipeline is built with depth test AND
        // write OFF, and the reverse-Z clear is 0.0, so a pixel showing only sky was never written.
        DECL_TEXTURE(PerBatch, Tex2D(float), gAplDepth)
        // 8 uints: [0..3] = asuint(mean R, mean G, mean B, mean logLuma), [4..6] = the p10 / p50 /
        // p90 luma DISPLAY LEVELS (0..255, plain integers, no asuint), [7] = the COUNTED sample
        // total, i.e. how many of the z*z lattice samples survived sky rejection. uint element type
        // + asuint() for the float half, matching gInstOut in cull.srt.h — the merged compute
        // rootsig has no float-typed RWBuffer and this is not the place to introduce one.
        //
        // The percentiles are S3a's doing: the calibration targets are readings off a REGION ("128
        // for lit parts"), and a frame mean is not the same statistic — comparing them overstated an
        // interior's gap as 3.51x where the real one was nearer 2.5x. See apl.comp.fsl.
        DECL_RWBUFFER(PerBatch, RWBuffer(uint), gAplOut)
    END_SRT_SET(PerBatch)
END_SRT(AplSrtData)
