// mgeHost64 — Tier 2 depth-takeover: linearize/resolve compute SRT.
//
// First COMPUTE root signature in the host (ComputeRootSignature, separate from the graphics
// DefaultRootSignature in opaque.srt.h). This pass makes the scene depth a single-sample,
// shader-readable resource: it reads the (possibly MSAA) DEPTH_WRITE depth target as an SRV and
// writes sample 0's device depth (reverse-Z) into a single-sample R32F UAV (gLinearDepthOut),
// which GTAO then samples AA-agnostically. The "linearize" naming follows the spec; the actual
// job is the MSAA resolve + SRV exposure (GTAO reconstructs world position from this device
// depth via gAOParams.invViewProj — see gtao.srt.h, so no projection constants live here).
//
// gSceneDepth's type switches on SAMPLE_COUNT (Depth2DMS vs Depth2D); both are one SRV slot, so
// the merged compute.rootsig is identical across the two compiled variants. Bounds checks are
// omitted — out-of-bounds UAV writes from ceil()-rounded dispatch tails are safely dropped on D3D12.
#pragma once

#ifndef SAMPLE_COUNT
#define SAMPLE_COUNT 1
#endif

BEGIN_SRT(LinDepthSrtData)
    BEGIN_SRT_SET(PerBatch)
#if SAMPLE_COUNT > 1
        DECL_TEXTURE(PerBatch, Depth2DMS(float, SAMPLE_COUNT), gSceneDepth)
#else
        DECL_TEXTURE(PerBatch, Depth2D(float), gSceneDepth)
#endif
        DECL_RWTEXTURE(PerBatch, WTex2D(float), gLinearDepthOut)
    END_SRT_SET(PerBatch)
END_SRT(LinDepthSrtData)
