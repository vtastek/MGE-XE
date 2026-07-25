// mgeHost64 — SUN shadow moments-map blur compute SRT (forge-sun-shadows.md, Phase B2).
//
// Consumer of the shared ComputeRootSignature. Runs as TWO dispatches over two INSTANCES of this
// one set (index 0 = horizontal, index 1 = vertical — the same per-instance pattern the shadow
// face passes use), ping-ponging moments map -> scratch -> moments map.
//
// Frequency = PerFrame (aoblur's precedent for a blur); resource NAMES are unique across every
// compute SRT so the unioned compute.rootsig has no aliasing.
#pragma once

STRUCT(SunBlurParams)
{
    // x = Gaussian sigma in TEXELS (<= 0 => passthrough copy),
    // y = map resolution (square), zw = tap direction in texels: (1,0) horizontal, (0,1) vertical.
    DATA(float4, params, None);
};

BEGIN_SRT(SunBlurSrtData)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(SunBlurParams), gSunBlurParams)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSunBlurSrc)
        DECL_RWTEXTURE(PerFrame, WTex2D(float4), gSunBlurDst)
    END_SRT_SET(PerFrame)
END_SRT(SunBlurSrtData)
