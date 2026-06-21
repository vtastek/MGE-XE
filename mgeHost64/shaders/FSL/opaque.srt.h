// mgeHost64 — M1c opaque scene SRT (batched cbuffer, scales past the 64KB cap).
//
// A single 64KB cbuffer holds at most 1024 float4x4 (the D3D12 cbuffer limit). To draw
// more than that per frame we BATCH: the host keeps a big world buffer of N 64KB windows
// and a PerBatch descriptor set whose instance b points at window b (via pRanges). Draws
// are issued in batches of 1024; within a batch the per-INSTANCE DrawIndex attribute
// (0..1023) selects the matrix from the bound window.
//
//   PerFrame  gFrameData : camera view*proj (bound once).
//   PerBatch  gBatch     : worlds[1024] for the current 1024-draw window (rebound per batch).
//
// Two cbuffers in SEPARATE sets get different register spaces (no overlap — two cbuffers
// in ONE set both land on b0 and collide). Matrix convention (PROVEN, column-major
// cbuffer): host uploads D3DXMATRIX bytes as-is, mul(M, v) reproduces D3DX's row-vector v*M.
#pragma once

#define OPAQUE_BATCH 1024   // matrices per 64KB cbuffer window; must match host kBatchSize

STRUCT(FrameData)
{
    DATA(float4x4, viewProj, None);
};

STRUCT(BatchData)
{
    DATA(float4x4, worlds[OPAQUE_BATCH], None);
};

BEGIN_SRT_NO_AB(SrtData)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(FrameData), gFrameData)
    END_SRT_SET(PerFrame)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER(PerBatch, CBUFFER(BatchData), gBatch)
    END_SRT_SET(PerBatch)
END_SRT(SrtData)
