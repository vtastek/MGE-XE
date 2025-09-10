//============================================================================
// MGE XE - Shadow System HLSL Include
// Provides shadow mapping, PCF filtering, and cascaded shadow functionality
//============================================================================

#ifndef SHADOWS_HLSL_INCLUDED
#define SHADOWS_HLSL_INCLUDED

#include "common.hlsl"

// Shadow configuration
#ifdef HAS_SHADOWS

// Shadow samplers and constants 
texture tex5 : register(t5);
sampler sampShadow : register(s5) = sampler_state{ texture = <tex5>; };
float shadowRcpRes : register(c10);

// Shadow constants
static const int shadowCascades = 2;
static const float shadowCascadeSize = 1.0 / shadowCascades;
static const float ESM_scale = 10.0;
static const float depth_bias = 0.0005;

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
            
            if (sampledDepth < receiverDepth - depth_bias) {
                blockerSum += sampledDepth;
                blockerCount += 1.0;
            }
        }
    }
    
    if (blockerCount == 0.0) return 1.0; // No blockers = no shadow
    
    float avgBlockerDepth = blockerSum / blockerCount;
    float penumbraSize = (receiverDepth - avgBlockerDepth) / avgBlockerDepth;
    penumbraSize = saturate(penumbraSize * 2.0);
    
    // PCF filtering pass with variable penumbra
    float shadow = 0.0;
    float sampleCount = 0.0;
    
    for (int px = -2; px <= 2; px++) {
        for (int py = -2; py <= 2; py++) {
            float2 offset = float2(px, py) * shadowRcpRes * penumbraSize;
            float sampledDepth = tex2Dlod(sampShadow, mapShadowToAtlas(shadowUV + offset, cascade)).r/ESM_scale;
            shadow += (sampledDepth >= receiverDepth - depth_bias * 3) ? 1.0 : 0.0;
            sampleCount += 1.0;
        }
    }
    
    return shadow / sampleCount;
}

// Main shadow sampling function for cascaded shadow maps
float shadowSample(float4 shadow0pos, float4 shadow1pos, float ndotlgeo) {
    float3 atlasMargin = float3(1.0 - 2.0 * 4.0 * shadowRcpRes, 1.0 - 2.0 * 4.0 * shadowRcpRes, 1.0);
    
    if (all(saturate(atlasMargin - abs(shadow0pos.xyz)))) {
        // Near cascade
        float2 shadowUV = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow0pos.xy;
        float receiverDepth = shadow0pos.z;
        return shadowSamplePCF(shadow0pos, shadowUV, 0, receiverDepth, ndotlgeo);
    }
    else if (all(saturate(atlasMargin - abs(shadow1pos.xyz)))) {
        // Far cascade
        float2 shadowUV = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow1pos.xy;
        float receiverDepth = shadow1pos.z;
        return shadowSamplePCF(shadow1pos, shadowUV, 1, receiverDepth, ndotlgeo);
    }
    else {
        // Outside both cascades - no shadow
        return 1.0;
    }
}

// Soft parallax shadowing for height-mapped surfaces
float ParallaxSoftShadow(sampler2D heightSampler, float2 texCoord, float2 lightDirTangent, 
                        float soften, float scale) {
    float h0 = 1.0 - tex2D(heightSampler, texCoord).a;
    float h = h0;
    float2 lDir = -lightDirTangent * scale;
    h = min(1.0, 1.0 - tex2D(heightSampler, texCoord + 0.20 * lDir).a);
    h = min(h, 1.0 - tex2D(heightSampler, texCoord + 0.35 * lDir).a);
    h = min(h, 1.0 - tex2D(heightSampler, texCoord + 0.45 * lDir).a);
    h = min(h, 1.0 - tex2D(heightSampler, texCoord + 0.55 * lDir).a);
    float shadowpara = min(1.0, 1.0 - saturate((h0 - h) * soften));
    return shadowpara;
}

#endif // HAS_SHADOWS

#endif // SHADOWS_HLSL_INCLUDED