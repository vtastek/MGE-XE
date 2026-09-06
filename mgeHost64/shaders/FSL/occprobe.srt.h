// mgeHost64 — H0: the HEIGHT-FIELD OCCLUSION PROBE (tasks/forge-heightfield-occlusion.md).
//
// A MEASUREMENT, not a cull. It answers one question, for two views that have no occlusion culling
// at all — the SUN caster (1.41 ms) and the WATER MIRROR (1.64 ms of `refl geo`) — and the question
// is: how many of the instances those views are drawing right now could a height-field test throw
// away? Nothing here changes what is drawn. If the answer is small the whole height-field occlusion
// plan dies here and no min-pyramid is ever built; if it is large, H1 is what makes it safe to act
// on. Default OFF (`occProbe`), so it costs nothing until it is asked for.
//
// ⚠ IT REPORTS A BRACKET, NOT A NUMBER, and the reason is structural rather than a caveat about
// precision. gSkyHeight (SH2) and gSunOcc are MAX fields — a texel that is half building and half
// street stores the roof — so a test against them as they stand rejects instances a real occlusion
// test would keep. That arm is the CEILING. The second arm reduces the same field with `min` over a
// texel neighbourhood, which is exactly what H1's min-pyramid would hold, so it previews the answer
// H2 would actually get. Between them:
//
//     rejected(FLOOR)  <=  what H2 would reject  <=  rejected(CEILING)
//
// A ceiling alone could not decide anything — 87% and 30% look identical from above — so the WIDTH
// of this bracket is the deliverable, not either end of it.
//
// ⚠ THE DENOMINATOR IS THE VIEW'S OWN DRAWN SET, not a re-derivation of it. gCullParams here is a
// VERBATIM COPY of the params that view's real cull ran with (the sun cull's cbuffer; the reflect
// cull's published record), and this shader re-runs that same rule before it tests occlusion. A
// probe whose denominator was reconstructed from first principles would be measuring its own
// reconstruction — see [[feedback_verify_the_right_artifact]].
#pragma once

#include "cullparams.h.fsl"   // CullInstance + CullParams. THE STRUCTS ONLY (see that header for
                              // why two SRTs may not meet in one file).

// Mode selector for the one pipeline. Two views, two completely different costs: the sun half is a
// single texture compare (gSunOcc already stores the answer), the mirror half is a march.
#define OCC_PROBE_MODE_SUN    0
#define OCC_PROBE_MODE_MIRROR 1

STRUCT(OccProbeParams)
{
    // xy = ABSOLUTE world XY of gProbeHeight's (0,0) texel CORNER (= g_skyHeightOrigin, the origin
    //      gSunOcc shares), z = world units per texel of the HEIGHT map, w = its resolution.
    DATA(float4, map,     None);
    // x = gSunOcc resolution, y = its world units per texel, z = 1 when the sun map is VALID (0
    //     disarms the sun half — night, or not built yet), w = spare.
    DATA(float4, sunMap,  None);
    // x = mode (OCC_PROBE_MODE_*), y = ABSOLUTE world Z of the water plane the mirror reflects
    //     about, z = march step count, w = height BIAS in world units (added to the field before the
    //     compare, so a coarse texel that clips its own occludee does not report a rejection).
    DATA(float4, probe,   None);
    // x = the march's END MARGIN in world units — the march stops this far short of the instance,
    //     because the instance IS IN THE FIELD and its own roof would otherwise block every ray to
    //     it.
    // y = the CONSERVATIVE arm's neighbourhood radius in texels (see the header note below). 0
    //     collapses it onto the raw arm exactly, which is the self-check.
    // z/w spare.
    DATA(float4, margin,  None);
};

BEGIN_SRT(OccProbeSrtData)
    // ONE set holding both CBVs, the instance SRV, both height textures and the counter UAV — the
    // shape every data-driven compute pass on this host uses (see sunocc.srt.h for why the split-set
    // arrangement was abandoned). Resource counts are a strict subset of the cull set's, so the
    // merged ComputeRootSignature already fits it.
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER (PerBatch, CBUFFER(CullParams),     gCullParams)    // the VIEW's own cull params, copied
        DECL_CBUFFER (PerBatch, CBUFFER(OccProbeParams), gProbeParams)
        DECL_BUFFER  (PerBatch, Buffer(CullInstance),    gCullInst)      // shared input (= the cull's)
        DECL_TEXTURE (PerBatch, Tex2D(float),            gProbeHeight)   // = gSkyHeight  (SH2 max terrain+statics)
        DECL_TEXTURE (PerBatch, Tex2D(float),            gProbeSunOcc)   // = gSunOcc     (sun-BLOCKED world Z)
        DECL_RWBUFFER(PerBatch, RWBuffer(uint),          gProbeCount)    // [0] tested, [1] rejected CEILING,
                                                                         // [2] rejected FLOOR
    END_SRT_SET(PerBatch)
END_SRT(OccProbeSrtData)
