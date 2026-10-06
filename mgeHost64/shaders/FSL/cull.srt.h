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

// The DATA TYPES (CullInstance / StaticsSubset / CullParams + the vis-mask geometry) live in their
// own struct-only header so occprobe.srt.h can share them: fsl.py emits every BEGIN_SRT it finds in
// an included header, so two SRTs may not meet in one file — but a struct carries no registers.
// This header stays the C++-safe one; forgerender.cpp includes IT and gets the types through here.
#include "cullparams.h.fsl"

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
        // H2b: the WATER MIRROR's height-field occlusion test (heightocclusion.h.fsl). Both APPENDED
        // LAST, for gCullHiz's reason — the 10 existing slots stay where every lane's DescriptorData
        // fill already puts them.
        //
        // ⚠ A CBV OF ITS OWN RATHER THAN MORE CullParams. CullParams is exactly 512 B and FULL
        // (cellOwn ends at float 127), so there is no room; and a second cbuffer is the honest shape
        // anyway, because these fields describe the height FIELD (a world-space resource shared by
        // every lane) rather than a view's frustum.
        //
        // ⚠ ALL SIX LANES BIND BOTH. Only the reflect lane ARMS the test, through a flag in the new
        // cbuffer — exactly as hizParams.w gates Hi-Z today. A lane that bound neither would read
        // whatever was last in those heap slots the moment someone armed it by accident.
        DECL_CBUFFER (PerBatch, CBUFFER(HeightOccParams), gHeightOccParams)
        DECL_TEXTURE (PerBatch, Tex2D(float),          gSkyHeightMin)  // H1's MIN-pyramid over pSkyHeight
        // Release 1: the COUNT pass appends every survivor's instance index here (counter =
        // gCullCount[2], reset with the other two), so cullscatter_list.comp writes rows for the
        // survivors alone instead of re-testing every instance in the world (~400k on the camera
        // lane for ~9k survivors). Sized to the lane's instance count: it cannot overflow. APPENDED
        // LAST for gCullHiz's reason.
        DECL_RWBUFFER(PerBatch, RWBuffer(uint),        gSurvivors)
    END_SRT_SET(PerBatch)
END_SRT(CullSrtData)
