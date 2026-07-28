// mgeHost64 — Stage B (B3) GPU statics cull: count -> prefix-sum -> scatter -> execute-indirect.
//
// One compute thread per resident exterior static instance. Reads the canonical GpuCullInstance
// (uploaded ONCE from g_cullInst — the SAME 96 B struct the CPU cull reads) plus the per-frame
// CullParams (6 frustum planes + eye + tier ranges², all extracted host-side from rzViewProj so the
// GPU test is bit-for-bit the CPU's dlLiveCullAndBuild rule). Three pipelines share this ONE SRT set:
//   cull.comp        (COUNT)   — survivors AtomicAdd numSubsets into gCullCount[0] (kept for the CPU
//                                parity check) AND AtomicAdd 1 into gSubsetCount[sid] per subset.
//   cullscan.comp    (PREFIX)  — single thread: prefix-sum gSubsetCount -> gSubsetOffset, reset
//                                gSubsetCursor, and fill one IndirectDrawIndexArguments per subset.
//   cullscatter.comp (SCATTER) — re-test survivors; slot=AtomicAdd(gSubsetCursor[sid],1); write the
//                                camera-relative instance row (world −eye + texSlot/flags) at
//                                gInstOut[(gSubsetOffset[sid]+slot)*20 ..].
// The GPU-filled gArgs drive cmdExecuteIndirect; gInstOut is bound as the per-instance vertex stream.
//
// Register model: within a SET, fsl.py assigns a FLAT slot per resource regardless of type (CBV/SRV/
// UAV), so the 9 resources below get distinct descriptor-table slots — no same-type aliasing (the
// d3d.py `is`-bug that aliased 2nd+ same-type resources is fixed to `==`).
#pragma once

// Dynamic-visibility mask geometry, shared by the host fill and BOTH cull shaders so the three can
// never disagree on a bound. 64 words of 32 bits = 2048 groups (a heavy TR+Bloodmoon bake uses ~100).
#define DL_VIS_MASK_WORDS 64u
#define DL_VIS_MASK_BITS  2048u

// NB: named CullInstance (NOT GpuCullInstance) — this header is #included in the host C++ TU too,
// where STRUCT(T) expands to `struct T`; reusing the host's GpuCullInstance name would redefine it.
// Byte layout (not the name) is what must match the host upload stride (96 B). world stored as 4
// explicit ROWS (row-major; wr3.xyz = translation) so the scatter reads/writes raw bytes without any
// float4x4 matrix-majorness ambiguity (DXC defaults matrices to column-major).
STRUCT(CullInstance)
{
    DATA(float4, wr0, None);            // world row 0 (64B total = the CPU GpuCullInstance.world[0..15])
    DATA(float4, wr1, None);
    DATA(float4, wr2, None);
    DATA(float4, wr3, None);            // wr3.xyz = absolute translation
    DATA(float,  posX,        None);
    DATA(float,  posY,        None);
    DATA(float,  posZ,        None);
    DATA(float,  effR,        None);    // -> 80B  frustum sphere radius
    DATA(uint,   rangeEndIdx, None);    // 0=near 1=far 2=vfar ; 0xFFFFFFFF = skip (grass/invalid)
    DATA(uint,   firstSubset, None);
    DATA(uint,   numSubsets,  None);
    DATA(uint,   visIndex,    None);    // -> 96B. usage.data dynamic-vis group (0 = ungated); gated
                                        // instances draw only while CullParams.visMask has the bit.
};

// Mirrors the host StaticsSubsetGPU (5 uints, 20B): the mega-VB/IB spans + bindless texSlot + flags.
STRUCT(StaticsSubset)
{
    DATA(uint, vbBase,     None);
    DATA(uint, ibBase,     None);
    DATA(uint, indexCount, None);
    DATA(uint, texSlot,    None);
    DATA(uint, flags,      None);
};

STRUCT(CullParams)
{
    DATA(float4, planes[6], None);  // 96B  Gribb-Hartmann planes (a,b,c,d); inside == a·x+b·y+c·z+d >= 0
    DATA(float4, eye,       None);  // xyz = camera eye (absolute world)
    DATA(float4, ranges,    None);  // x=nearEnd² y=farEnd² z=vfarEnd² w=nearCut²
    DATA(float4, misc,      None);  // x = instance count, y = subset count (as floats; uint4 not C++-safe)
    // -- Occlusion M2: the previous frame's Hi-Z pyramid camera (snapshotted at prologue submit).
    // hizVP = the RAW rzViewProj bytes of the frame that filled the pyramid (camera-relative,
    // reverse-Z, extended-far — the exact matrix statics.vert projected with). float4x4 in a
    // C++-included SRT is proven (AOParams.invViewProj). Total 240B <= the 256B cbuffer.
    DATA(float4x4, hizVP,       None);  // floats 36..51  prev-frame relative world -> clip
    DATA(float4,   hizParams,   None);  // 52..55: x=mip0 W, y=mip0 H, z=mipCount-1, w=valid (0 = pass-through)
    DATA(float4,   hizEyeDelta, None);  // 56..59: xyz = eyeNow - hizEye (rebase this frame's c_rel into hiz space)
    // Dynamic visibility mask: 2048 bits (one per usage.data vis group), bit set = group VISIBLE.
    // The client already ships per-group enable deltas on every cell change (scanDynamicVisGroups ->
    // Server::updateDynVis); this is that state, mirrored into the cull. Rides the cbuffer so no new
    // SRT resource / descriptor-set change is needed, and BOTH pCullSet and pSunCullSet inherit it
    // (the sun cull copies the whole 496B, which is what stops LOD shadows from ghost buildings).
    // float4 not uint4: this header is #included in the host C++ TU, where uint4 is not available —
    // the shaders read it back with asuint, a pure bitcast. floats 60..123 -> 496B total.
    DATA(float4,   visMask[16], None);
};

BEGIN_SRT(CullSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER (PerBatch, CBUFFER(CullParams),   gCullParams)
        DECL_BUFFER  (PerBatch, Buffer(CullInstance),  gCullInst)
        DECL_BUFFER  (PerBatch, Buffer(StaticsSubset), gStaticsSubsets)
        DECL_RWBUFFER(PerBatch, RWBuffer(uint),        gCullCount)     // [0] = Σ numSubsets (CPU parity)
        DECL_RWBUFFER(PerBatch, RWBuffer(uint),        gSubsetCount)   // [sid] survivor count
        DECL_RWBUFFER(PerBatch, RWBuffer(uint),        gSubsetOffset)  // [sid] prefix-sum start (mStartInstance)
        DECL_RWBUFFER(PerBatch, RWBuffer(uint),        gSubsetCursor)  // [sid] scatter cursor (reset in scan)
        DECL_RWBUFFER(PerBatch, RWBuffer(uint),        gArgs)          // IndirectDrawIndexArguments[sid] (5 uint each)
        DECL_RWBUFFER(PerBatch, RWBuffer(uint),        gInstOut)       // survivor rows (20 uint/inst = 80B), asuint(float)
        DECL_TEXTURE (PerBatch, Tex2D(float),          gCullHiz)       // prev-frame Hi-Z pyramid (M2; appended
                                                                       // LAST so the 9 existing slots are stable)
    END_SRT_SET(PerBatch)
END_SRT(CullSrtData)
