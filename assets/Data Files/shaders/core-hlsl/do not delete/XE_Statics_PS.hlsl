// XE_Statics_PS.hlsl  
// MGE XE Phase 8 - Distant Land HLSL Pixel Shader for Statics
// Simplified permutation-based version of XE Mod Statics.fx

// Textures
sampler2D sampBaseTex : register(s0);

// Input structure matching StatVertOut from XE Common.fx
struct PSInput {
    float4 color : COLOR0;
    float4 fog : TEXCOORD0;
    float3 texcoords_range : TEXCOORD1;
};

// Alpha to coverage calculation (from XE Common.fx)
float calc_coverage(float alpha, float alphaRef, float scale) {
    return saturate((alpha - alphaRef) * scale + alphaRef);
}

// Fog application
float3 fogApply(float3 color, float4 fog) {
    return lerp(color, fog.rgb, fog.a);
}

float4 main(PSInput input) : COLOR0 {
    float2 texcoords = input.texcoords_range.xy;
    float range = input.texcoords_range.z;
    
    // Sample base texture
    float4 result = tex2D(sampBaseTex, texcoords);
    
    // Apply vertex lighting
    result.rgb *= input.color.rgb;
    
    // Apply fog
    result.rgb = fogApply(result.rgb, input.fog);
    
    #ifdef HAS_ALPHA
    // Alpha to coverage conversion for distant statics
    result.a = calc_coverage(result.a, 133.0/255.0, 2.0);
    #endif
    
    return result;
}