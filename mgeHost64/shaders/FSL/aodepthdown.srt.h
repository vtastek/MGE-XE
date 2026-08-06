// mgeHost64 — half-res AO chain, step 1: depth downsample SRT. Chain overview + the shared
// AOUpParams cbuffer: aohalfres.srt.h.
//
// Full pLinearDepth -> half pLinearDepthHalf, taking the MAX of each 2x2. MAX, not min and not
// average. Reverse-Z: near = 1, far = 0 (aocommon.h.fsl treats <= 0 as sky), so the CLOSEST surface
// is the LARGEST value. Backwards, this silently biases every AO tap toward the background, which
// reads as "AO got weaker" rather than as a bug. Averaging is worse still — a depth averaged across
// a silhouette exists nowhere, and aoReconstructWorld would place the surface in mid-air.
//
// Update frequency Persistent — distinct from the AO chain's linearize (PerBatch), AO (PerDraw) and
// blur (PerFrame), so the host's descriptor-table rebind cache (keyed [pipelineType][rootIndex])
// cannot collide with theirs and drop a bind. gtao.srt.h:31-39 records that this failure mode is a
// SILENTLY VANISHED UAV write, not a validation error. Resource names are unique across every
// compute SRT for the same reason.
#pragma once

#include "aohalfres.srt.h"

BEGIN_SRT(AODownSrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_CBUFFER(Persistent, CBUFFER(AOUpParams), gAODownParams)
        DECL_TEXTURE(Persistent, Tex2D(float), gAODownDepthIn)
        DECL_RWTEXTURE(Persistent, WTex2D(float), gAODownDepthOut)
    END_SRT_SET(Persistent)
END_SRT(AODownSrtData)
