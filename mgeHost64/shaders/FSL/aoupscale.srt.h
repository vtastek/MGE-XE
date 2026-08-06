// mgeHost64 — half-res AO chain, step 2: adaptive Lanczos upscale SRT. Chain overview + the shared
// AOUpParams cbuffer: aohalfres.srt.h.
//
// Half-res blurred AO + both linear depths -> the FULL pAOBlur, so everything downstream (colour
// frags included) sees exactly the resource it always did.
//
// Update frequency Persistent, same reasoning as aodepthdown.srt.h — and its own file, because two
// SRTs sharing a frequency alias each other's registers if one shader ends up declaring both.
#pragma once

#include "aohalfres.srt.h"

BEGIN_SRT(AOUpSrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_CBUFFER(Persistent, CBUFFER(AOUpParams), gAOUpParams)
        DECL_TEXTURE(Persistent, Tex2D(float4), gAOUpSrc)        // pAOBlurHalf
        DECL_TEXTURE(Persistent, Tex2D(float),  gAOUpDepthHalf)  // pLinearDepthHalf
        DECL_TEXTURE(Persistent, Tex2D(float),  gAOUpDepthFull)  // pLinearDepth
        DECL_RWTEXTURE(Persistent, WTex2D(float4), gAOUpDst)     // pAOBlur (full)
    END_SRT_SET(Persistent)
END_SRT(AOUpSrtData)
