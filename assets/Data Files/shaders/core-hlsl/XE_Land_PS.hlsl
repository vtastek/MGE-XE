// XE_Land_PS.hlsl
// MGE XE Phase 8 - Distant Land HLSL Pixel Shader for Landscape  
// Simplified permutation-based version of XE Mod Landscape.fx

// Textures
sampler2D sampBaseTex : register(s0);
sampler2D sampDetail : register(s1);

// Input structure
struct PSInput {
    float4 color : COLOR0;
    float4 fog : TEXCOORD0;
    float2 texcoords : TEXCOORD1;
};

// Fog application
float3 fogApply(float3 color, float4 fog) {
    return lerp(color, fog.rgb, fog.a);
}

float4 main(PSInput input) : COLOR0 {
    // Sample base texture
    float4 result = tex2D(sampBaseTex, input.texcoords);
    
    // Apply vertex lighting
    result.rgb *= input.color.rgb;
    
    // Apply fog
    result.rgb = fogApply(result.rgb, input.fog);
    
    return result;
}