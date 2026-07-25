// mgeHost64 — SUN shadow moments-map blur compute SRT (forge-sun-shadows.md, Phase B2).
//
// Consumer of the shared ComputeRootSignature. Runs as TWO dispatches PER CASCADE over 2*N
// INSTANCES of this one set (instance 2c+0 = horizontal, 2c+1 = vertical — the same per-instance
// pattern the shadow face passes use), ping-ponging one atlas TILE -> scratch -> the same tile.
//
// Tile-at-a-time rather than whole-atlas is deliberate: the scratch then only has to be ONE tile
// (not the whole N-wide atlas), each cascade gets its own sigma from its own instance CBV with no
// per-pixel tile lookup, and the tap clamp cannot reach a neighbouring cascade's texels.
//
// Frequency = PerFrame (aoblur's precedent for a blur); resource NAMES are unique across every
// compute SRT so the unioned compute.rootsig has no aliasing.
#pragma once

STRUCT(SunBlurParams)
{
    // x = Gaussian sigma in TEXELS (<= 0 => passthrough copy),
    // y = TILE resolution (square), zw = tap direction in texels: (1,0) horizontal, (0,1) vertical.
    DATA(float4, params, None);
    // The atlas is N tiles wide and the scratch is one tile wide, so exactly one side of each pass
    // needs an x offset: H reads the atlas at srcOriginX and writes the scratch at 0, V reads the
    // scratch at 0 and writes the atlas at dstOriginX.
    //   x = srcOriginX (texels), y = dstOriginX (texels),
    //   zw = inclusive clamp range for the SOURCE coordinate along the tap axis — the tile's own
    //        bounds, so an edge tap saturates inside the tile instead of sampling the next cascade.
    DATA(float4, params2, None);
};

BEGIN_SRT(SunBlurSrtData)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(SunBlurParams), gSunBlurParams)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSunBlurSrc)
        DECL_RWTEXTURE(PerFrame, WTex2D(float4), gSunBlurDst)
    END_SRT_SET(PerFrame)
END_SRT(SunBlurSrtData)
