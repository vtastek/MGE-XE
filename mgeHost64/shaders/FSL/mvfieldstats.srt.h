// mgeHost64 — STATISTICS OF THE FINISHED MOTION-VECTOR FIELD.  tasks/forge-postprocess.md (MB-1).
//
// motionvectors.comp measures its own output with gMvStats, and that was enough while it was the
// only writer. It is not any more: the object-velocity pass OVERWRITES mover pixels afterwards, so
// those in-shader statistics describe an INTERMEDIATE field that no longer exists by the time
// anything reads it. A number that names the frame and silently excludes the last writer is the
// same defect kGpuPhaseObjVel exists to avoid one level down.
//
// This runs LAST, over the finished texture, and therefore describes exactly what F12 mode 17 draws.
// Together the two lines attribute the difference: `mv field:` is the camera pass alone, `mv final:`
// is the camera pass plus the overwrite, and toggling objVelEnable moves only the second.
//
// ⚠⚠ THE HEADLINE IS `nonzero`, NOT `still`, AND IT IS NOW STRICTER THAN THE PICTURE.
// gMvStats already counts pixels under 0.01 px — a TOLERANCE — which reported 100.0% for a frame the
// eye called dirty. When this pass was written mvview.frag's grey branch was `m == 0.0f` EXACTLY, so
// `nonzero` and that branch were literally one predicate and this counter was "what the eye counts".
// The view has since moved to `m < 0.01f`, because the exact branch turned out to be unreachable in
// this engine (MW's camera matrix is not bit-stable when the player stands still — measured deltas
// 4.3e-4, 2.9e-11, 1.5e-3 across three consecutive parked frames), and the grey disc around the
// focus of expansion is the field's own geometry rather than an artifact.
//
// `nonzero` did NOT follow it, deliberately. A counter that tracks the view can only ever confirm
// the view; this one is kept exact so it can answer questions no picture and no tolerance can:
//   * did the host's parked-camera lane actually reach the GPU? On a frame `mv camera:` reports
//     `parked(bit-identical)=1`, `nonzero` must be **0.000%** — the acceptance test for MB-2 step 0,
//     where opts.y was computed, logged, and then cleared before upload for three weeks.
//   * is a residue that is invisible to the debug view still large enough to matter to a CONSUMER?
//     MB-2's gather steps along `mv`, so ~1e-3 px on a parked frame is a sub-pixel smear over a
//     still image; `still` cannot see that and this can.
#pragma once

STRUCT(MvFieldStatsParams)
{
    // xy = render rect px.
    // z  = THE PARKED FLAG FOR THE FRAME THESE STATISTICS DESCRIBE, echoed straight back out in
    //      gMvFsOut[6]. ⚠ IT IS CARRIED WITH THE MEASUREMENT RATHER THAN READ BESIDE IT, and that
    //      is what makes the acceptance test attributable at all: the readback is consumed a frame
    //      or more later (no fence — pAplReadback's arrangement), so pairing a percentage with
    //      whatever `g_mvCamParked` happens to hold at PRINT time compares two different frames.
    //      "nonzero% must be 0.000% when parked" is only checkable if the two halves come from one
    //      frame, so the flag rides in the buffer.
    // w  = unused.
    DATA(float4, screenParams, None);
};

BEGIN_SRT(MvFieldStatsSrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_CBUFFER (Persistent, CBUFFER(MvFieldStatsParams), gMvFsParams)
        // The FINISHED field — read as an SRV, never written here. This pass is a measurement and
        // must not be able to change what it measures.
        DECL_TEXTURE (Persistent, Tex2D(float2), gMvFsField)
        DECL_TEXTURE (Persistent, Tex2D(float),  gMvFsReactive)
        // 7 uints: [0] max |mv| asuint, [1] min |mv| asuint, [2] count < 0.01 px, [3] pixels counted,
        // [4] count reactive > 0.5, [5] count |mv| != 0 EXACTLY, [6] the parked flag for THIS
        // frame, echoed from screenParams.z so the percentage and the condition it must be read
        // against travel together. asuint on a non-negative float is
        // monotonic in its bit pattern, so InterlockedMax/Min on the raw bits is a correct float
        // max/min — the same trick motionvectors.comp uses, and the reason neither needs SM6.6
        // float atomics.
        DECL_RWBUFFER(Persistent, RWBuffer(uint), gMvFsOut)
    END_SRT_SET(Persistent)
END_SRT(MvFieldStatsSrtData)
