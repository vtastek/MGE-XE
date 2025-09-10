//============================================================================
// MGE XE - Shadow System HLSL Include
// Provides shadow mapping, PCF filtering, and cascaded shadow functionality
//============================================================================

#ifndef SHADOWS_HLSL_INCLUDED
#define SHADOWS_HLSL_INCLUDED

#include "common.hlsl"

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

// PCF shadow filtering with distance-based penumbra
float shadowSamplePCF(float4 shadowPos, float2 shadowUV, int cascade, float receiverDepth, float ndotlgeo) {
    // Blocker search pass - find average blocker depth
    float blockerSum = 0.0;
    float blockerCount = 0.0;
    
    for (int sx = -1; sx <= 1; sx++) {
        for (int sy = -1; sy <= 1; sy++) {
            float2 offset = float2(sx, sy) * shadowRcpRes;
            float sampledDepth = tex2Dlod(sampShadow, mapShadowToAtlas(shadowUV + offset, cascade)).r/ESM_scale;
            
            if (sampledDepth < receiverDepth - PCF_bias) {
                blockerSum += sampledDepth;
                blockerCount += 1.0;
            }
        }
    }
    
    if (blockerCount == 0.0) return 1.0; // No blockers = no shadow
    
    float avgBlockerDepth = blockerSum / blockerCount;
    float penumbraSize = (receiverDepth - avgBlockerDepth) * 100.0; // Scale factor for visibility
    penumbraSize = penumbraSize * PCF_penumbraScale;
    penumbraSize = clamp(penumbraSize, PCF_minPenumbra, PCF_maxPenumbra);
    
    // Calculate slope bias based on surface angle to light
    // ndotlgeo ranges from 0 (perpendicular) to 1 (parallel)
    // For steep angles (low ndotlgeo), we need more bias
    float slopeFactor = 1.0 - ndotlgeo; // 0 for parallel surfaces, 1 for perpendicular
    float dynamicSlopeBias = PCF_slopeBias * slopeFactor;
    
    // Lerp between two bias values based on surface angle to light, then add slope bias
    float biasLerp = lerp(PCF_bias, PCF_bias2, step(0.4, ndotlgeo)) + dynamicSlopeBias;
    
    // PCF filtering pass with variable penumbra
    float shadow = 0.0;
    float sampleCount = 0.0;
    
    for (int px = -2; px <= 2; px++) {
        for (int py = -2; py <= 2; py++) {
            float2 offset = float2(px, py) * shadowRcpRes * penumbraSize * PCF_filterSize;
            float sampledDepth = tex2Dlod(sampShadow, mapShadowToAtlas(shadowUV + offset, cascade)).r/ESM_scale;
            shadow += (sampledDepth >= receiverDepth - biasLerp) ? 1.0 : 0.0;
            sampleCount += 1.0;
        }
    }
    
    return shadow / sampleCount;
}

// Simple ESM shadow sampling with blur for far cascade
float shadowSampleESM(float4 shadowPos, float2 shadowUV, int cascade, float receiverDepth, float ndotlgeo) {
    // Calculate slope bias based on surface angle to light
    float slopeFactor = 1.0 - ndotlgeo; // 0 for parallel surfaces, 1 for perpendicular
    float dynamicSlopeBias = PCF_slopeBias * slopeFactor;
    
    float biasLerp = lerp(PCF_bias, PCF_bias2, step(0.4, ndotlgeo)) + dynamicSlopeBias;
    float shadow = 0.0;
    float sampleCount = 0.0;
    
    // Simple 3x3 blur pattern for ESM
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float2 offset = float2(x, y) * shadowRcpRes * 0.5; // Smaller blur than PCF
            float sampledDepth = tex2Dlod(sampShadow, mapShadowToAtlas(shadowUV + offset, cascade)).r/ESM_scale;
            shadow += (sampledDepth >= receiverDepth - biasLerp) ? 1.0 : 0.0;
            sampleCount += 1.0;
        }
    }
    
    return shadow / sampleCount;
}

// Main shadow sampling function for cascaded shadow maps
float shadowSample(float4 shadow0pos, float4 shadow1pos, float ndotlgeo, float alphaFlag) {
    float3 atlasMargin = float3(1.0 - 2.0 * 4.0 * shadowRcpRes, 1.0 - 2.0 * 4.0 * shadowRcpRes, 1.0);
    float3 blendMargin = float3(1.0 - 2.0 * 264.0 * shadowRcpRes, 1.0 - 2.0 * 264.0 * shadowRcpRes, 1.0); // 50% wider blend zone
    
    bool inNear = all(saturate(atlasMargin - abs(shadow0pos.xyz)));
    bool inFar = all(saturate(atlasMargin - abs(shadow1pos.xyz)));
    bool nearBlend = all(saturate(blendMargin - abs(shadow0pos.xyz)));
    
    if (inNear && nearBlend) {
        // Near cascade - use ESM for alpha objects, PCF for opaque
        float2 shadowUV = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow0pos.xy;
        float receiverDepth = shadow0pos.z;
        if (alphaFlag > 0.5) {
            return shadowSampleESM(shadow0pos, shadowUV, 0, receiverDepth, ndotlgeo);
        } else {
            return shadowSamplePCF(shadow0pos, shadowUV, 0, receiverDepth, ndotlgeo);
        }
    }
    else if (inNear || inFar) {
        // Always blend when in overlap zone or transition
        float2 shadowUV0 = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow0pos.xy;
        float2 shadowUV1 = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow1pos.xy;
        
        float shadow0 = inNear ? (alphaFlag > 0.5 ? shadowSampleESM(shadow0pos, shadowUV0, 0, shadow0pos.z, ndotlgeo) : shadowSamplePCF(shadow0pos, shadowUV0, 0, shadow0pos.z, ndotlgeo)) : 1.0;
        float shadow1 = inFar ? shadowSampleESM(shadow1pos, shadowUV1, 1, shadow1pos.z, ndotlgeo) : 1.0;
        
        // Calculate blend factor based on distance from near cascade center
        float3 nearDist = abs(shadow0pos.xyz);
        float nearBlendFactor = saturate(max(max(nearDist.x, nearDist.y), 0.0) - 0.7); // Start blending at 70% of cascade
        
        // Ensure we have valid shadow data before blending
        if (inNear && inFar) {
            return shadow1;
        } else if (inNear) {
            return shadow0;
        } else if (inFar) {
            return shadow1;
        }
    }
    
    // Outside both cascades - no shadow
    return 1.0;
}
	
#endif // HAS_SHADOWS

#endif // SHADOWS_HLSL_INCLUDED