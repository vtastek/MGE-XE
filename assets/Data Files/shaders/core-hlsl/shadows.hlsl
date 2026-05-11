//============================================================================
// MGE XE - Shadow System HLSL Include
// Provides shadow mapping, PCF filtering, and cascaded shadow functionality
//============================================================================

#ifndef SHADOWS_HLSL_INCLUDED
#define SHADOWS_HLSL_INCLUDED

#include "common.hlsl"

// Receiver plane depth bias calculation from MJP's Shadows sample
float2 ComputeReceiverPlaneDepthBias(float3 texCoordDX, float3 texCoordDY)
{
    float2 biasUV;
    biasUV.x = texCoordDY.y * texCoordDX.z - texCoordDX.y * texCoordDY.z;
    biasUV.y = texCoordDX.x * texCoordDY.z - texCoordDY.x * texCoordDX.z;
    biasUV *= 1.0f / ((texCoordDX.x * texCoordDY.y) - (texCoordDX.y * texCoordDY.x));
    return biasUV;
}

// Shadow configuration
#ifdef HAS_SHADOWS
float PCF_bias : register(c11);
float PCF_bias2 : register(c12);
float PCF_filterSize : register(c13);
float PCF_penumbraScale : register(c14);
float PCF_minPenumbra : register(c15);
float PCF_maxPenumbra : register(c16);
float PCF_slopeBias : register(c17);
matrix shadowViewProjPS[3] : register(c31);
float4 shadowCascadeDepths : register(c43); // x = exclusive close split, y = old near/far split, z = far distance
float4 closePCFBiasParams : register(c44); // x = bias, y = bias2, z = slope bias, w = terrain bias
float4 closePCFFilterParams : register(c45); // x = filter size
// Terrain receiver hint + extra near-cascade bias.
//   .x = isTerrain (0 or 1, set per-draw / per-merged-batch on C++ side)
//   .y = terrain bias amount (global, from imgui PCF window)
// Terrain self-occludes more visibly than other geometry in the near cascade
// across all rendering modes; this lets us bump bias for terrain receivers
// without inflating the global PCF_bias and softening contact shadows on
// non-terrain meshes.
float4 terrainShadowParams : register(c25);

// Shadow constants
static const int shadowCascades = 3;
static const float shadowCascadeSize = 1.0 / shadowCascades;
static const float ESM_scale = 32768.0;

// Shadow UV to shadow atlas UV conversion
float4 mapShadowToAtlas(float2 t, int layer) {
    return float4(t.x * shadowCascadeSize + layer * shadowCascadeSize, t.y, 0, 0);
}

// Blue noise texture sampler (34x1: texels 0-24 = 25-sample, texels 25-33 = 9-sample)
sampler sampBlueNoise : register(s7) = sampler_state {
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = NONE;
    AddressU = CLAMP;
    AddressV = CLAMP;
};

// Sample blue noise from texture (replaces static arrays for faster compile)
float2 sampleBlueNoise25(int i) {
    return tex2Dlod(sampBlueNoise, float4((i + 0.5) / 34.0, 0.5, 0, 0)).xy;
}

float2 sampleBlueNoise9(int i) {
    return tex2Dlod(sampBlueNoise, float4((25 + i + 0.5) / 34.0, 0.5, 0, 0)).xy;
}

// Fixed-radius PCF shadow filtering for the exclusive close cascade.
float shadowSamplePCF(float4 shadowPos, float2 shadowUV, int cascade, float receiverDepth, float ndotlgeo, float4 biasParams, float filterSize) {
    // Calculate slope bias based on surface angle to light
    // ndotlgeo ranges from 0 (perpendicular) to 1 (parallel)
    // For steep angles (low ndotlgeo), we need more bias
    float slopeFactor = saturate(ndotlgeo * 5); // 0 for parallel surfaces, 1 for perpendicular
    float dynamicSlopeBias = biasParams.z * slopeFactor;

    // Calculate partial derivatives for receiver plane depth bias
    float2 shadowMapSize = float2(2048, 2048); // Adjust based on your shadow map size
    float2 texelSize = 1.0f / shadowMapSize;

    // Approximate derivatives using neighboring samples
    float3 texCoordDX = float3(texelSize.x, 0, 0);
    float3 texCoordDY = float3(0, texelSize.y, 0);

    // Compute receiver plane depth bias
    float2 receiverPlaneDepthBias = ComputeReceiverPlaneDepthBias(texCoordDX, texCoordDY);

    // Static depth biasing to make up for incorrect fractional sampling on the shadow map grid
    float fractionalSamplingError = 2.0 * dot(float2(1.0f, 1.0f) * texelSize, abs(receiverPlaneDepthBias));
    float compareDepth = receiverDepth - min(fractionalSamplingError, 0.01f);

    float finalBias = lerp(biasParams.x, biasParams.y, step(0.7, ndotlgeo)) + dynamicSlopeBias;
    finalBias += terrainShadowParams.x * biasParams.w;
    compareDepth -= finalBias;

    // PCF filtering pass with blue noise sampling - invert values so shadows=1, lit=0
    float shadow = 1.0;
    float sampleCount = 0.0;
    float filterRadius = (shadowRcpRes + shadowRcpRes / 2) * max(0.25, filterSize * 0.1);

    // Use 25-sample blue noise pattern for main PCF filtering
    for (int i = 0; i < 25; i++) {
        // Blue noise provides random distribution within the filter radius
        float2 offset = sampleBlueNoise25(i) * filterRadius;

        // Use tex2D with hardware bilinear filtering
        float sampledDepth = tex2D(sampShadow, mapShadowToAtlas(shadowUV + offset, cascade).xy).r / (ESM_scale);
        // Invert: shadow=1.0, lit=0.0 (so shadows dominate when averaged)
        float sampleShadow = sampledDepth >= compareDepth ? 0.0 : 1.0;
        shadow += sampleShadow;
        sampleCount += 1.0;
    }

    // Return inverted result: 1.0=lit, 0.0=shadow (normal convention)
    return 1 - saturate(shadow / sampleCount);
}

// Simple ESM shadow sampling with blur for far cascade
float shadowSampleESM(float4 shadowPos, float2 shadowUV, int cascade, float receiverDepth, float ndotlgeo) {
    float4 biasParams = cascade == 0 ? closePCFBiasParams : float4(PCF_bias, PCF_bias2, PCF_slopeBias, terrainShadowParams.y);

    // Calculate slope bias based on surface angle to light
    float slopeFactor = 1.0 - pow(ndotlgeo, 11); // 0 for parallel surfaces, 1 for perpendicular
    float dynamicSlopeBias = biasParams.z * slopeFactor;

    float biasLerp = lerp(biasParams.x, biasParams.y, step(0.4, ndotlgeo)) + dynamicSlopeBias;
    biasLerp += terrainShadowParams.x * (cascade == 2 ? 0.0 : biasParams.w);
    float shadow = 0.0;
    float sampleCount = 0.0;

    // Use blue noise sampling pattern for ESM
    for (int i = 0; i < 9; i++) {
        float2 offset = sampleBlueNoise9(i) * shadowRcpRes * 2.0; // Blue noise within small radius
        // Use tex2D with hardware bilinear filtering instead of tex2Dlod
        float sampledDepth = tex2D(sampShadow, mapShadowToAtlas(shadowUV + offset, cascade).xy).r / ESM_scale;
        shadow += (sampledDepth >= receiverDepth - biasLerp) ? 1.0 : 0.0;
        sampleCount += 1.0;
    }

    return shadow / sampleCount;
}

// Shadow texel density checkerboard visualization
// Returns: RGB color showing cascade (hue) and texel density (checker size)
// checkerScale: texels per checker square (8 = 8x8 texel blocks)
float4 shadowPosFromView(float3 viewPos, int cascade) {
    float4 shadowPos = mul(float4(viewPos, 1.0), shadowViewProjPS[cascade]);
    shadowPos.z = shadowPos.z / shadowPos.w;
    return shadowPos;
}

float2 shadowSplitDepths() {
    float closeSplit = max(shadowCascadeDepths.x, 0.0);
    float oldNearFarSplit = shadowCascadeDepths.y > closeSplit
        ? shadowCascadeDepths.y
        : max(closeSplit + 1.0, shadowCascadeDepths.z * 0.5);
    return float2(closeSplit, oldNearFarSplit);
}

float3 shadowTexelCheckerboard(float3 viewPos, float checkerScale) {
    float4 shadow0pos = shadowPosFromView(viewPos, 0);
    float4 shadow1pos = shadowPosFromView(viewPos, 1);
    float4 shadow2pos = shadowPosFromView(viewPos, 2);

    // Shadow map size (matches shadowSamplePCF hardcoded value)
    float shadowMapSize = 2048.0;

    float2 shadowUV0 = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow0pos.xy;
    float2 shadowUV1 = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow1pos.xy;
    float2 shadowUV2 = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow2pos.xy;

    // Checkerboard at scaled texel resolution (checkerScale texels per square)
    float2 texelPos0 = shadowUV0 * shadowMapSize / checkerScale;
    float2 texelPos1 = shadowUV1 * shadowMapSize / checkerScale;
    float2 texelPos2 = shadowUV2 * shadowMapSize / checkerScale;

    float checker0 = fmod(floor(texelPos0.x) + floor(texelPos0.y), 2.0);
    float checker1 = fmod(floor(texelPos1.x) + floor(texelPos1.y), 2.0);
    float checker2 = fmod(floor(texelPos2.x) + floor(texelPos2.y), 2.0);
    float viewDepth = max(viewPos.z, 0.0);
    float2 splitDepths = shadowSplitDepths();

    // Cascade 0 (close): cyan/magenta
    float3 color0A = float3(0.0, 1.0, 1.0);  // cyan
    float3 color0B = float3(1.0, 0.0, 1.0);  // magenta

    // Cascade 1 (old near): green/red
    float3 color1A = float3(0.0, 1.0, 0.0);  // green
    float3 color1B = float3(1.0, 0.0, 0.0);  // red

    // Cascade 2 (far): yellow/blue
    float3 color2A = float3(1.0, 1.0, 0.0);  // yellow
    float3 color2B = float3(0.0, 0.0, 1.0);  // blue

    if (viewDepth <= splitDepths.x) {
        return lerp(color0A, color0B, checker0);
    }
    else if (viewDepth <= splitDepths.y) {
        return lerp(color1A, color1B, checker1);
    }
    return lerp(color2A, color2B, checker2);
}

// Main shadow sampling function for cascaded shadow maps
float shadowSample(float3 viewPos, float ndotlgeo, float alphaFlag) {
    float3 receiverLimit = float3(1.0 + 2.0 * 16.0 * shadowRcpRes, 1.0 + 2.0 * 16.0 * shadowRcpRes, 1.0);
    float4 shadow0pos = shadowPosFromView(viewPos, 0);
    float4 shadow1pos = shadowPosFromView(viewPos, 1);
    float4 shadow2pos = shadowPosFromView(viewPos, 2);
    bool inNear = all(saturate(receiverLimit - abs(shadow0pos.xyz)));
    bool inMid = all(saturate(receiverLimit - abs(shadow1pos.xyz)));
    bool inFar = all(saturate(receiverLimit - abs(shadow2pos.xyz)));
    float2 uvMargin = 4.0 * shadowRcpRes;

    float2 shadowUV0 = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow0pos.xy;
    float2 shadowUV1 = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow1pos.xy;
    float2 shadowUV2 = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow2pos.xy;
    shadowUV0 = clamp(shadowUV0, uvMargin, 1.0 - uvMargin);
    shadowUV1 = clamp(shadowUV1, uvMargin, 1.0 - uvMargin);
    shadowUV2 = clamp(shadowUV2, uvMargin, 1.0 - uvMargin);

    float shadow0 = alphaFlag > 0.5
        ? shadowSampleESM(shadow0pos, shadowUV0, 0, shadow0pos.z, ndotlgeo)
        : shadowSamplePCF(shadow0pos, shadowUV0, 0, shadow0pos.z, ndotlgeo, closePCFBiasParams, closePCFFilterParams.x);
    float shadow1 = shadowSampleESM(shadow1pos, shadowUV1, 1, shadow1pos.z, ndotlgeo);
    float shadow2 = shadowSampleESM(shadow2pos, shadowUV2, 2, shadow2pos.z, ndotlgeo);

    float viewDepth = max(viewPos.z, 0.0);
    float2 splitDepths = shadowSplitDepths();

    if (viewDepth <= splitDepths.x) {
        if (inNear) return shadow0;
        if (inMid) return shadow1;
        if (inFar) return shadow2;
    }
    else if (viewDepth <= splitDepths.y) {
        if (inMid) return shadow1;
        if (inFar) return shadow2;
        if (inNear) return shadow0;
    }
    else {
        if (inFar) return shadow2;
        if (inMid) return shadow1;
        if (inNear) return shadow0;
    }

    return 1.0;
}

#endif // HAS_SHADOWS

#endif // SHADOWS_HLSL_INCLUDED
