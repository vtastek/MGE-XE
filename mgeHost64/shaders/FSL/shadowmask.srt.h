// mgeHost64 — P1 point-light shadows: screen-space shadow-mask compute SRT.
//
// Third consumer of the shared ComputeRootSignature (fsl.py unions it with linearizedepth/
// gtao/aoblur/cull/hiz — see [[project_forge_fsl_shader_compile]]). One dispatch per frame:
// reconstruct the camera-relative world position from the resolved prepass depth
// (gShadowLinDepth = pLinearDepth, raw reverse-Z device depth, single-sample so NO MSAA
// variants needed), run the analytic cube-face atlas test per ACTIVE shadow slot, and pack
// 4-bit visibilities into the R32G32_UINT mask (x = slots 0-7, y = 8-15; 15 = fully lit).
//
// PerBatch frequency (proven multi-user root slot: LinDepthSrtData + CullSrtData both ride
// PerBatch with different layouts). Everything the pass needs lives in gShadowMaskParams —
// the forward gLights cbuffer is NOT read here; the host mirrors the slotted lights' pos/
// radius into slotPosRad so the two stay decoupled.
#pragma once

#define MAX_SHADOW_SLOTS 16   // MUST match host kMaxShadowLights (forgerender.cpp)

STRUCT(ShadowMaskParams)
{
    DATA(float4x4, invViewProj,  None);   // inverse of the reverse-Z camera-relative viewProj
    DATA(float4,   screenParams, None);   // xy = screen w,h ; zw = 1/w, 1/h
    // maskParams: x = active-slot BITMASK (bit s = slot s live), y = face near plane (world u),
    //             z = relative reverse-Z compare slack (acne knob, live-tunable),
    //             w = debug mode (0 off, 1 face-id nibble, 2 atlas-depth view)
    DATA(float4,   maskParams,   None);
    DATA(float4,   slotPosRad[MAX_SHADOW_SLOTS], None);   // xyz = camera-relative light pos, w = radius (far = 2r)
    DATA(float4,   slotTile[MAX_SHADOW_SLOTS],   None);   // xy = 3x2 face-block origin (atlas px), z = face size (px)
};

BEGIN_SRT(ShadowMaskSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER(PerBatch, CBUFFER(ShadowMaskParams), gShadowMaskParams)
        DECL_TEXTURE(PerBatch, Tex2D(float), gShadowLinDepth)
        DECL_TEXTURE(PerBatch, Tex2D(float), gShadowAtlas)
        DECL_RWTEXTURE(PerBatch, WTex2D(uint2), gShadowMaskOut)
    END_SRT_SET(PerBatch)
END_SRT(ShadowMaskSrtData)
