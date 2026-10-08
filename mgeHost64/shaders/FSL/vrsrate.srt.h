// mgeHost64 — the VARIABLE-RATE-SHADING rate image (D3D12 VRS Tier 2), built per frame from the
// prepass depth. One texel per shading-rate tile (the device's ShadingRateImageTileSize, 16 on
// NVIDIA/AMD/Intel), holding a D3D12_SHADING_RATE code. The host binds it only around draws that
// ask for it (the terrain colour pass): far tiles shade once per 2x2 pixels, near tiles per pixel.
// MSAA coverage stays per sample at any rate, so silhouettes keep their edges.
#pragma once

// Reads the PREPASS DEPTH BUFFER directly (sample 0 under MSAA), not the pLinearDepth copy, so the
// rate image does not keep the full-screen pre-colour copy alive on its own (preLinSkip).
#ifndef SAMPLE_COUNT
#define SAMPLE_COUNT 1
#endif

STRUCT(VrsParams)
{
    // x = device-depth threshold: a tile is COARSE when its NEAREST sample (largest reverse-Z) is
    //     below this, i.e. the whole tile lies farther than the host's vrsDist. 0 = never coarse.
    // y = the coarse rate code (D3D12_SHADING_RATE_2X2 = 0x5), z = tile size in pixels,
    // w = taps per tile side (the tile is subsampled on a w x w grid — conservative enough for a
    //     distance test, and 16 loads instead of 256).
    DATA(float4, p, None);
    // xy = the extent the taps clamp to (the RENDER rect, in pixels); zw unused.
    DATA(float4, dims, None);
};

BEGIN_SRT(VrsSrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_CBUFFER  (Persistent, CBUFFER(VrsParams), gVrsParams)
#if SAMPLE_COUNT > 1
        DECL_TEXTURE  (Persistent, Depth2DMS(float, SAMPLE_COUNT), gVrsDepth)   // pDepth itself, sample 0
#else
        DECL_TEXTURE  (Persistent, Depth2D(float),     gVrsDepth)   // pDepth itself
#endif
        DECL_RWTEXTURE(Persistent, WTex2D(uint),       gVrsRate)    // R8_UINT, one texel per tile
    END_SRT_SET(Persistent)
END_SRT(VrsSrtData)
