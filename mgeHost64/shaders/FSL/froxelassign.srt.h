// mgeHost64 — clustered forward lighting: froxel light-assignment SRT.
//
// The distant baked point lights are binned into a froxel grid (screen tiles x view-Z slices) so
// each distant fragment loops ONLY the lights whose reach-sphere touches its froxel, not the whole
// streamed <=128 set (the brute loop measured ~3.6ms on a hilltop over a lit town). Assignment is a
// LIGHT-SCATTER compute pass: one thread per uploaded light projects the light's world AABB to a
// screen-tile box + a radial-distance slice range, and AtomicOrs the light's bit into every covered
// froxel's 128-bit mask (4 x uint). froxelclear.comp zeros the mask first (own dispatch + UAV barrier).
//
// Mirrors shadowlightcull.srt.h: spheres ride IN the cbuffer (a structured SRV over a per-frame
// CPU_TO_GPU heap is illegal in D3D12), 128 float4 fit trivially. Merged into ComputeRootSignature
// (PerBatch set): 1 CBV + 1 UAV — within the cull set's union, so no root-sig growth.
#pragma once

#define FROXEL_MAX_LIGHTS 128   // MUST match MAX_POINT_LIGHTS (opaque.srt.h) / IPC::kMaxPointLights

STRUCT(FroxelParams)
{
    DATA(float4x4, viewProj, None);   // 0..15   cam-relative world -> clip (raw rzViewProj)
    DATA(float4,   dims,     None);   // 16..19  x=tilesX, y=tilesY, z=NZslices, w=lightCount
    DATA(float4,   screen,   None);   // 20..23  x=W, y=H, z=tileSize(px), w=numWords (froxels*4)
    DATA(float4,   zparams,  None);   // 24..27  x=d0, y=d1, z=log(d0), w=invLogRange (1/log(d1/d0))
    DATA(float4,   spheres[FROXEL_MAX_LIGHTS], None);  // 28.. per light: xyz=center (absPos-lodEye), w=reach (<=0 skip)
};

BEGIN_SRT(FroxelSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER (PerBatch, CBUFFER(FroxelParams), gFroxelParams)
        DECL_RWBUFFER(PerBatch, RWBuffer(uint),        gFroxelMaskRW)   // tilesX*tilesY*NZ*4 uints
    END_SRT_SET(PerBatch)
END_SRT(FroxelSrtData)
