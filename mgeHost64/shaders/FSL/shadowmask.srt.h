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

#define MAX_SHADOW_SLOTS 32   // MUST match host kMaxShadowLights (forgerender.cpp)

STRUCT(ShadowMaskParams)
{
    DATA(float4x4, invViewProj,  None);   // inverse of the reverse-Z camera-relative viewProj
    DATA(float4,   screenParams, None);   // xy = screen w,h ; zw = 1/w, 1/h
    // maskParams: x = shadow TEST RANGE in light radii (live knob, host g_shadowRangeK; 2.0 =
    //             the atlas far plane = old behaviour). Range cull + slot test use it; the
    //             refZ depth mapping always uses farZ = 2r (the atlas render far plane),
    //             y = face near plane (world u),
    //             z = relative reverse-Z compare slack (acne knob, live-tunable),
    //             w = debug mode (0 off, 1 face-id nibble, 2 atlas-depth view)
    DATA(float4,   maskParams,   None);
    DATA(float4,   slotPosRad[MAX_SHADOW_SLOTS], None);   // xyz = camera-relative light pos, w = radius (far = 2r)
    DATA(float4,   slotTile[MAX_SHADOW_SLOTS],   None);   // xy = 3x2 face-block origin (atlas px), z = face size (px)
    // biasParams: x = ABSOLUTE reverse-Z compare bias (contact/interpenetration knob, live). Added
    //             to the compare threshold in place of the old PSO constant depth bias (higher =
    //             less acne, more contact gap).
    //             y = NORMAL-OFFSET bias in atlas texels (live). Pushes the receiver sample off
    //             its surface along the depth-reconstructed normal, scaled by grazing angle — the
    //             sole grazing-acne mechanism now that the PSO slope-scaled term is zeroed. Face-on
    //             contact stays tight (offset → 0 there); grazing surfaces get the most.
    //             z = UNUSED (was the dynamic-slot bitmask; moved to slotBits.y).
    DATA(float4,   biasParams,   None);
    // slotBits: the 32-bit slot masks as REAL uints (float lanes drop bits >= 24).
    //             x = ACTIVE-slot bitmask (bit s = slot s live).
    //             y = DYNAMIC-slot bitmask (C4b composite): bit s set = slot s's dynamic tile
    //             (movers: skinned + multimap, re-rendered every frame) is valid THIS frame —
    //             each PCF texel takes max(static, dynamic) = the nearer reverse-Z occluder.
    //             Unset = the dynamic tile is stale (mover left reach); sample static only.
    //             z = FLICKER-class bitmask (fClass==2 slots → shadow direction wobble).
    //             w = LANTERN bitmask (light enclosed by its own cage → soft attenuation-fade).
    // Appended at the struct tail so every offset above stays fixed.
    DATA(uint4,    slotBits,     None);
};

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
