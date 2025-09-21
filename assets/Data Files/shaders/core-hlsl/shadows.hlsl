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

// Shadow constants
static const int shadowCascades = 2;
static const float shadowCascadeSize = 1.0 / shadowCascades;
static const float ESM_scale = 32768.0;

// Shadow UV to shadow atlas UV conversion
float4 mapShadowToAtlas(float2 t, int layer) {
    return float4(t.x * shadowCascadeSize + layer * shadowCascadeSize, t.y, 0, 0);
}

// Blue noise pattern for PCF sampling (25 samples, optimized distribution)
static const float2 BLUE_NOISE_25[25] = {
    float2(-0.4706,  0.2941), float2(0.0588, -0.4706), float2(0.4118,  0.1765),
    float2(-0.1765, -0.2353), float2(0.2353,  0.4118), float2(-0.3529, -0.4118),
    float2(0.4706, -0.1176), float2(-0.0588,  0.3529), float2(0.1176, -0.3529),
    float2(-0.4118,  0.0588), float2(0.3529, -0.2941), float2(-0.2353,  0.4706),
    float2(0.0000,  0.1176), float2(0.2941, -0.4706), float2(-0.4706, -0.0588),
    float2(0.4118,  0.3529), float2(-0.1176, -0.4118), float2(0.1765,  0.2353),
    float2(-0.3529,  0.1176), float2(0.4706, -0.3529), float2(-0.0588, -0.1765),
    float2(0.2353,  0.0588), float2(-0.2941,  0.4118), float2(0.3529, -0.0588),
    float2(-0.1765,  0.3529)
};

// Smaller blue noise pattern for lighter sampling (9 samples)
static const float2 BLUE_NOISE_9[9] = {
    float2(-0.3333,  0.3333), float2(0.1111, -0.4444), float2(0.4444,  0.1111),
    float2(-0.1111, -0.3333), float2(0.3333,  0.4444), float2(-0.4444, -0.1111),
    float2(0.0000,  0.2222), float2(0.2222, -0.2222), float2(-0.2222,  0.0000)
};

// PCF shadow filtering with distance-based penumbra
float shadowSamplePCF(float4 shadowPos, float2 shadowUV, int cascade, float receiverDepth, float ndotlgeo) {
    // Blocker search pass using blue noise - find average blocker depth
    float blockerSum = 0.0;
    float blockerCount = 0.0;

    // Use 9-sample blue noise pattern for blocker search
    for (int i = 0; i < 9; i++) {
        float2 offset = BLUE_NOISE_9[i] * shadowRcpRes; // Blue noise within single texel radius
        // Use tex2D with hardware bilinear filtering instead of tex2Dlod
        float sampledDepth = tex2D(sampShadow, mapShadowToAtlas(shadowUV + offset, cascade).xy).r / (ESM_scale);

        if (sampledDepth < receiverDepth - PCF_bias) {
            blockerSum += sampledDepth;
            blockerCount += 1.0;
        }
    }

    float avgBlockerDepth = blockerSum / blockerCount;
    float penumbraSize = (receiverDepth - avgBlockerDepth) * 100.0; // Scale factor for visibility
    penumbraSize = penumbraSize * PCF_penumbraScale;
    penumbraSize = clamp(penumbraSize, PCF_minPenumbra, PCF_maxPenumbra);

    // Calculate slope bias based on surface angle to light
    // ndotlgeo ranges from 0 (perpendicular) to 1 (parallel)
    // For steep angles (low ndotlgeo), we need more bias
    float slopeFactor = saturate(ndotlgeo * 5); // 0 for parallel surfaces, 1 for perpendicular
    float dynamicSlopeBias = PCF_slopeBias * slopeFactor;

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
    float biasedDepth = receiverDepth - min(fractionalSamplingError, 0.01f);

    float finalBias = lerp(PCF_bias, PCF_bias2, step(0.7, ndotlgeo)) + dynamicSlopeBias;
    biasedDepth -= finalBias;



    // PCF filtering pass with blue noise sampling - invert values so shadows=1, lit=0
    float shadow = 1.0;
    float sampleCount = 0.0;
    float filterRadius = (shadowRcpRes + shadowRcpRes / 2) * penumbraSize * max(0.25, PCF_filterSize * 0.1);

    // Use 25-sample blue noise pattern for main PCF filtering
    for (int i = 0; i < 25; i++) {
        // Blue noise provides random distribution within the filter radius
        float2 offset = BLUE_NOISE_25[i] * filterRadius;

        // Use tex2D with hardware bilinear filtering
        float sampledDepth = tex2D(sampShadow, mapShadowToAtlas(shadowUV + offset, cascade).xy).r / (ESM_scale);
        // Invert: shadow=1.0, lit=0.0 (so shadows dominate when averaged)
        float sampleShadow = ((sampledDepth) >= receiverDepth - biasedDepth * 0.0032) ? 0.0 : 1.0;
        shadow += sampleShadow;
        sampleCount += 1.0;
    }

    // Return inverted result: 1.0=lit, 0.0=shadow (normal convention)
    return 1 - saturate(shadow / sampleCount);
}

// Optimized PCF with receiver plane depth bias (from MJP's method)
float shadowSampleOptimizedPCF(float4 shadowPos, float2 shadowUV, int cascade, float receiverDepth, float ndotlgeo) {
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
    float biasedDepth = receiverDepth - min(fractionalSamplingError, 0.01f);

    // Apply additional slope-based bias
    float slopeFactor = saturate(ndotlgeo * 5);
    float dynamicSlopeBias = PCF_slopeBias * slopeFactor;
    float finalBias = lerp(PCF_bias, PCF_bias2, step(0.7, ndotlgeo)) + dynamicSlopeBias;
    biasedDepth -= finalBias;

    // Sample with blue noise, bilinear filtering and planar depth bias
    float shadow = 0.0;
    float sampleCount = 0.0;

    // Use 9-sample blue noise for optimized PCF
    for (int i = 0; i < 9; i++) {
        float2 blueNoiseOffset = BLUE_NOISE_9[i] * shadowRcpRes * PCF_filterSize;
        float2 sampleUV = shadowUV + blueNoiseOffset;

        // Apply planar depth bias based on offset
        float sampleDepth = biasedDepth + dot(blueNoiseOffset, receiverPlaneDepthBias);

        // Sample with hardware bilinear filtering
        float sampledDepth = tex2D(sampShadow, mapShadowToAtlas(sampleUV, cascade).xy).r / ESM_scale;

        shadow += (sampledDepth >= sampleDepth) ? 1.0 : 0.0;
        sampleCount += 1.0;
    }

    return shadow / sampleCount;
}

// Simple ESM shadow sampling with blur for far cascade
float shadowSampleESM(float4 shadowPos, float2 shadowUV, int cascade, float receiverDepth, float ndotlgeo) {
    // Calculate slope bias based on surface angle to light
    float slopeFactor = 1.0 - pow(ndotlgeo, 11); // 0 for parallel surfaces, 1 for perpendicular
    float dynamicSlopeBias = PCF_slopeBias * slopeFactor;

    float biasLerp = lerp(PCF_bias * 2, PCF_bias2 * 2, step(0.4, ndotlgeo)) + dynamicSlopeBias;
    float shadow = 0.0;
    float sampleCount = 0.0;

    // Use blue noise sampling pattern for ESM
    for (int i = 0; i < 9; i++) {
        float2 offset = BLUE_NOISE_9[i] * shadowRcpRes * 0.5; // Blue noise within small radius
        // Use tex2D with hardware bilinear filtering instead of tex2Dlod
        float sampledDepth = tex2D(sampShadow, mapShadowToAtlas(shadowUV + offset * 0, cascade).xy).r / ESM_scale;
        shadow += (sampledDepth >= receiverDepth - biasLerp) ? 1.0 : 0.0;
        sampleCount += 1.0;
    }

    return shadow / sampleCount;
}

// Main shadow sampling function for cascaded shadow maps
float shadowSample(float4 shadow0pos, float4 shadow1pos, float ndotlgeo, float alphaFlag) {
    float3 atlasMargin = float3(1.0 - 2.0 * 4.0 * shadowRcpRes, 1.0 - 2.0 * 4.0 * shadowRcpRes, 1.0);
    float3 blendMargin = float3(1.0 - 2.0 * 264.0 * shadowRcpRes, 1.0 - 2.0 * 264.0 * shadowRcpRes, 1.0); // 50% wider blend zone

    bool inNear = all(saturate(atlasMargin - abs(shadow0pos.xyz)));
    bool inFar = all(saturate(atlasMargin - abs(shadow1pos.xyz)));

    float2 shadowUV0 = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow0pos.xy;
    float2 shadowUV1 = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow1pos.xy;

    // Use optimized PCF for opaque geometry, original PCF for alpha
    float shadow0 = inNear ? (alphaFlag > 0.5 ? shadowSampleESM(shadow0pos, shadowUV0, 0, shadow0pos.z, ndotlgeo) : shadowSamplePCF(shadow0pos, shadowUV0, 0, shadow0pos.z, ndotlgeo)) : 1.0;
    float shadow1 = inFar ? shadowSampleESM(shadow1pos, shadowUV1, 1, shadow1pos.z, ndotlgeo) : 1.0;

    // Ensure we have valid shadow data before blending
    if (inNear) {
        return shadow0;
    }
    else if (inFar) {
        return shadow1;
    }

    // Outside both cascades - no shadow
    return 1.0;
}

#endif // HAS_SHADOWS

#endif // SHADOWS_HLSL_INCLUDED