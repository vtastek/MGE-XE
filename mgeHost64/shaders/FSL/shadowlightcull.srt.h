// mgeHost64 — Follow-on 3: occlusion-cull shadow lights vs the Hi-Z pyramid.
//
// One compute thread per shadow slot (kMaxShadowLights = 32). Each slot's INFLUENCE SPHERE
// (center = current camera-relative world, radius = the mask TEST reach = g_shadowRangeK·r) is
// tested against the PREVIOUS frame's depth pyramid, reprojected via hizEyeDelta — the exact
// same one-pass test the statics cull runs (hizocclusion.h.fsl). A fully-occluded slot ORs its
// bit into gLightOccBits[0]; the host reads that back with 1-frame latency and skips the slot's
// shadow-atlas BAKE (never its cached tile or the screen-space mask — bakes-only veto is pop-free).
//
// The per-slot spheres ride IN the cbuffer (float4 spheres[32]) rather than a structured SRV: a
// structured-buffer SRV over a per-frame CPU_TO_GPU (upload) heap is illegal in D3D12 (silent
// device-remove — see the cull's GPU_ONLY note), and 32 float4 fit trivially in one 256B-aligned
// cbuffer. Reuses hizocclusion.h.fsl VERBATIM, so the cbuffer/texture MUST be named gCullParams +
// gCullHiz (the two globals that header reads); the struct TYPE is free (LightCullParams). Merged
// into ComputeRootSignature (PerBatch set): 1 CBV + 1 SRV-tex + 1 UAV — a strict subset of the
// cull set's counts, so the existing union already fits it.
#pragma once

STRUCT(LightCullParams)
{
    DATA(float4x4, hizVP,       None);  // 0..15  prev-frame relative world -> clip (raw rzViewProj)
    DATA(float4,   hizParams,   None);  // 16..19 x=mip0 W, y=mip0 H, z=mipCount-1, w=valid (0 = pass-through)
    DATA(float4,   hizEyeDelta, None);  // 20..23 xyz = eyeNow - hizEye (rebase cam-rel center into hiz space)
    DATA(float4,   spheres[32], None);  // 24..151 per slot: xyz = center (absPos - lodEye), w = radius (<=0 inactive)
};

BEGIN_SRT(ShadowLightCullSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER (PerBatch, CBUFFER(LightCullParams), gCullParams)     // named for hizocclusion.h.fsl
        DECL_RWBUFFER(PerBatch, RWBuffer(uint),           gLightOccBits)   // [0] = occluded-slot bitmask
        DECL_TEXTURE (PerBatch, Tex2D(float),             gCullHiz)        // named for hizocclusion.h.fsl
    END_SRT_SET(PerBatch)
END_SRT(ShadowLightCullSrtData)
