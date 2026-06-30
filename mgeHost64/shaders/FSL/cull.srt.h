// mgeHost64 — Stage B (M1) GPU frustum/tier cull, COUNT-only validation SRT.
//
// One compute thread per resident exterior static instance. Reads the canonical GpuCullInstance
// (uploaded ONCE from g_cullInst — the SAME 96 B struct the CPU cull reads) plus the per-frame
// CullParams (6 frustum planes + eye + tier ranges², all extracted host-side from rzViewProj so the
// GPU test is bit-for-bit the CPU's dlLiveCullAndBuild rule). Survivors atomic-add their numSubsets
// into gCullCount[0]; the host reads it back and compares to the CPU survivor total (g_liveLastInst).
//
// B2 = VALIDATION ONLY: no scatter, no draw cutover. (B3 adds the per-subset prefix-sum + scatter +
// execute-indirect; the matrix + firstSubset fields, unused here, are already resident for it.)
//
// One CBV (b0) + one structured SRV (t0) + one structured UAV (u0) — three DIFFERENT register types,
// so no same-type slot aliasing (the d3d.py `is`-bug only bites a 2nd resource of the SAME type).
#pragma once

// NB: named CullInstance (NOT GpuCullInstance) — this header is #included in the host C++ TU too,
// where STRUCT(T) expands to `struct T`; reusing the host's GpuCullInstance name would redefine it.
// Byte layout (not the name) is what must match the host upload stride (96 B).
STRUCT(CullInstance)
{
    DATA(float4x4, world,       None);  // 64B absolute world (B3 scatter reads it; unused in the count)
    DATA(float,    posX,        None);
    DATA(float,    posY,        None);
    DATA(float,    posZ,        None);
    DATA(float,    effR,        None);  // -> 80B  frustum sphere radius
    DATA(uint,     rangeEndIdx, None);  // 0=near 1=far 2=vfar ; 0xFFFFFFFF = skip (grass/invalid)
    DATA(uint,     firstSubset, None);
    DATA(uint,     numSubsets,  None);
    DATA(uint,     pad,         None);  // -> 96B (matches the host C++ GpuCullInstance stride)
};

STRUCT(CullParams)
{
    DATA(float4, planes[6], None);  // 96B  Gribb-Hartmann planes (a,b,c,d); inside == a·x+b·y+c·z+d >= 0
    DATA(float4, eye,       None);  // xyz = camera eye (absolute world)
    DATA(float4, ranges,    None);  // x=nearEnd² y=farEnd² z=vfarEnd² w=nearCut²
    DATA(float4, misc,      None);  // x = instance count as float (dispatch-tail bounds guard; uint4 not C++-safe)
};

BEGIN_SRT(CullSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER (PerBatch, CBUFFER(CullParams),  gCullParams)
        DECL_BUFFER  (PerBatch, Buffer(CullInstance), gCullInst)
        DECL_RWBUFFER(PerBatch, RWBuffer(uint),       gCullCount)
    END_SRT_SET(PerBatch)
END_SRT(CullSrtData)
