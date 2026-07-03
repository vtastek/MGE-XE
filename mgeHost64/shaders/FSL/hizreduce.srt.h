// mgeHost64 — Hi-Z mip-pyramid build compute SRT (occlusion M2: pyramid = FULL-scene depth).
//
// Fifth consumer of the shared ComputeRootSignature (after linearizedepth + gtao + aoblur + the
// cull trio). Three pipelines share this ONE set:
//   hizfirst_sc1/sc4.comp — copy sample 0 of pDepth (the real scene DSV, reverse-Z device depth,
//                           AFTER the colour pass so DL land + statics depth is in) -> pyramid
//                           mip 0. Runs in the MAIN cmd at the colour->water seam — water depth
//                           must NOT enter the pyramid (it would falsely occlude refraction-
//                           visible geometry).
//   hizreduce.comp        — 2x2 MIN-reduce mip i-1 -> mip i (reverse-Z: min = FARTHEST, the
//                           conservative value the occlusion test needs). Prologue cmd.
// gHizSceneDepth switches on SAMPLE_COUNT (Depth2DMS vs Depth2D, linearizedepth.srt.h's pattern);
// both are one SRV slot, so the merged compute.rootsig is identical across variants. The reduce
// variant compiles with SAMPLE_COUNT=1 and never reads the slot (DXC strips it).
//
// The pyramid texture is SHADER_RESOURCE between cmds (cull.comp samples it at the top of the
// next frame's cmd); every cmd that touches it brackets SR -> UAV ... UAV -> SR. This Forge
// creates one UAV per mip (DescriptorData::mUAVMipSlice selects it) but NO per-mip SRVs, so the
// reduce reads the source mip through a UAV too (R32F typed UAV load is D3D12-guaranteed) — no
// mixed subresource states, no per-mip barriers beyond plain UAV barriers.
//
// Update frequency = Persistent — the only compute frequency still free (PerBatch = linearize +
// cull, PerDraw = gtao, PerFrame = aoblur; the descriptor-table rebind cache is keyed per
// frequency/rootIndex — see gtao.srt.h for the collision this avoids). Resource names are unique
// across all merged compute SRTs (house rule since the d3d.py aliasing bug).
#pragma once

#ifndef SAMPLE_COUNT
#define SAMPLE_COUNT 1
#endif

BEGIN_SRT(HizSrtData)
    BEGIN_SRT_SET(Persistent)
#if SAMPLE_COUNT > 1
        DECL_TEXTURE  (Persistent, Depth2DMS(float, SAMPLE_COUNT), gHizSceneDepth)  // pDepth (first pass only)
#else
        DECL_TEXTURE  (Persistent, Depth2D(float), gHizSceneDepth)  // pDepth (first pass only)
#endif
        DECL_RWTEXTURE(Persistent, RTex2D(float),  gHizSrcMip)     // pyramid mip i-1 (reduce only)
        DECL_RWTEXTURE(Persistent, WTex2D(float),  gHizDstMip)     // pyramid mip i (both passes)
    END_SRT_SET(Persistent)
END_SRT(HizSrtData)
