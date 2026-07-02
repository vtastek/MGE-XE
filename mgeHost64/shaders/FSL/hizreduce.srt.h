// mgeHost64 — Phase 3 prologue: Hi-Z mip-pyramid build compute SRT.
//
// Fifth consumer of the shared ComputeRootSignature (after linearizedepth + gtao + aoblur + the
// cull trio). Two pipelines share this ONE set:
//   hizreduce_first.comp — copy pLinearDepth (SRV, reverse-Z device depth) -> pyramid mip 0.
//   hizreduce.comp       — 2x2 MIN-reduce mip i-1 -> mip i (reverse-Z: min = FARTHEST, the
//                          conservative value an occlusion test needs).
// The pyramid texture stays UNORDERED_ACCESS for its whole life: this Forge creates one UAV per
// mip (DescriptorData::mUAVMipSlice selects it) but NO per-mip SRVs, so the reduce reads the
// source mip through a UAV too (R32F typed UAV load is D3D12-guaranteed) — no mixed subresource
// states, no per-mip barriers beyond plain UAV barriers.
//
// Update frequency = Persistent — the only compute frequency still free (PerBatch = linearize +
// cull, PerDraw = gtao, PerFrame = aoblur; the descriptor-table rebind cache is keyed per
// frequency/rootIndex — see gtao.srt.h for the collision this avoids). Resource names are unique
// across all merged compute SRTs (house rule since the d3d.py aliasing bug).
#pragma once

BEGIN_SRT(HizSrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_TEXTURE  (Persistent, Tex2D(float),   gHizLinDepth)   // pLinearDepth (first pass only)
        DECL_RWTEXTURE(Persistent, RTex2D(float),  gHizSrcMip)     // pyramid mip i-1 (reduce only)
        DECL_RWTEXTURE(Persistent, WTex2D(float),  gHizDstMip)     // pyramid mip i (both passes)
    END_SRT_SET(Persistent)
END_SRT(HizSrtData)
