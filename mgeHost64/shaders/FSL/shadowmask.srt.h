// mgeHost64 — P1 point-light shadows: screen-space shadow-mask compute SRT.
//
// Third consumer of the shared ComputeRootSignature (fsl.py unions it with linearizedepth/
// gtao/aoblur/cull/hiz — see [[project_forge_fsl_shader_compile]]). One dispatch per frame:
// reconstruct the camera-relative world position from the resolved prepass depth
// (gShadowLinDepth = pLinearDepth, raw reverse-Z device depth, single-sample so NO MSAA
// variants needed), run the analytic cube-face atlas test per ACTIVE shadow slot, and pack
// 4-bit visibilities into the R32G32B32A32_UINT mask (uint4 lanes of 8 slots each:
// x = 0-7, y = 8-15, z = 16-23, w = 24-31; 15 = fully lit).
//
// PerBatch frequency (proven multi-user root slot: LinDepthSrtData + CullSrtData both ride
// PerBatch with different layouts). Everything the pass needs lives in gShadowMaskParams —
// the forward gLights cbuffer is NOT read here; the host mirrors the slotted lights' pos/
// radius into slotPosRad so the two stay decoupled.
#pragma once

// ShadowMaskParams (per-slot atlas tile / pos / bias / flicker) is shared with the first-person
// direct-atlas path in opaque.frag — see shadowparams.h.fsl for the full field docs.
#include "shadowparams.h.fsl"

BEGIN_SRT(ShadowMaskSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER(PerBatch, CBUFFER(ShadowMaskParams), gShadowMaskParams)
        DECL_TEXTURE(PerBatch, Tex2D(float), gShadowLinDepth)
        DECL_TEXTURE(PerBatch, Tex2D(float), gShadowAtlas)
        DECL_RWTEXTURE(PerBatch, WTex2D(uint4), gShadowMaskOut)
        // C4b composite: the parallel DYNAMIC atlas (same block layout as gShadowAtlas —
        // movers only, re-rendered per frame). Appended LAST so existing indices stay put.
        DECL_TEXTURE(PerBatch, Tex2D(float), gShadowAtlasDyn)
    END_SRT_SET(PerBatch)
END_SRT(ShadowMaskSrtData)
