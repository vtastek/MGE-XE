// XE_Statics_VS.hlsl
// MGE XE Phase 8 - Distant Land HLSL Vertex Shader for Statics
// Simplified permutation-based version of XE Mod Statics.fx

// Matrices and transforms
float4x4 world : WORLD;
float4x4 view : VIEW;
float4x4 proj : PROJECTION;

// Lighting
float3 eyePos;
float3 sunVec;
float3 sunCol;
float3 sunAmb;

// Fogging
float3 fogColNear;
float3 fogColFar;
float fogStart;
float fogRange;
float nearViewRange;

// Environment
float time;
float2 windVec;
float niceWeather;

// Input structure matching StatVertIn from XE Common.fx
struct VSInput {
    float4 pos : POSITION;
    float4 normal : NORMAL;  // normal.w contains emissive value
    float4 color : COLOR0;
    float2 texcoords : TEXCOORD0;
};

// Output structure matching StatVertOut from XE Common.fx
struct VSOutput {
    float4 pos : POSITION;
    float4 color : COLOR0;
    float4 fog : TEXCOORD0;
    float3 texcoords_range : TEXCOORD1;
};

// Transform vertex with implicit depth bias (from transformStaticVert)
float4 transformStaticVert(float4 inPos) {
    float4 worldpos = mul(inPos, world);
    float4 viewpos = mul(worldpos, view);
    float4 pos = mul(viewpos, proj);
    return pos;
}

// Light calculation (from lightStaticVert)
float4 lightStaticVert(VSInput input) {
    // Decompress normal
    float4 normal = float4(normalize(2.0 * input.normal.xyz - 1.0), 0.0);
    normal = mul(normal, world);
    
    // Lighting (worldspace)
    // Emissive is stored in the 4th value of the normal vector
    float emissive = input.normal.w;
    float3 light = sunCol * saturate(dot(normal.xyz, -sunVec)) + sunAmb + emissive;
    
    return float4(input.color.rgb * light, input.color.a);
}

// Texture coordinate modification (from texcoordsModifier)
float2 texcoordsModifier(VSInput input) {
    float2 tc = input.texcoords;
    
    #ifdef HAS_VCOL
    // Linked to animateUV static flag
    // Render with fixed scrolling that approximates ghostfence
    tc.y += fmod(0.08 * time, 1.0);
    #endif
    
    return tc;
}

// Fog color calculation for exterior
float4 fogColour(float3 eyeVecNorm, float dist) {
    #ifdef USE_ATM_SCATTER
    // Atmospheric scattering would go here
    // For now, simple linear interpolation
    float fogFactor = saturate((dist - fogStart) / fogRange);
    return lerp(float4(fogColNear, 1.0), float4(fogColFar, 1.0), fogFactor);
    #else
    // Simple fog
    float fogFactor = saturate((dist - fogStart) / fogRange);
    return float4(fogColNear, fogFactor);
    #endif
}

// Fog color for interior (simpler MW-style)
float4 fogMWColour(float dist) {
    float fogFactor = saturate((dist - fogStart) / fogRange);
    return float4(fogColNear, fogFactor);
}

VSOutput main(VSInput input) {
    VSOutput output;
    
    // Transform vertex
    float4 worldpos = mul(input.pos, world);
    float4 viewpos = mul(worldpos, view);
    output.pos = mul(viewpos, proj);
    
    // Lighting
    output.color = lightStaticVert(input);
    
    // Distance for fogging and range calculation
    float3 eyevec = worldpos.xyz - eyePos.xyz;
    float dist = length(eyevec);
    
    // Fogging - different for exterior vs interior
    #ifdef STATICS_EXTERIOR
    output.fog = fogColour(eyevec / dist, dist);
    #else  // STATICS_INTERIOR
    output.fog = fogMWColour(length(viewpos.xyz));
    #endif
    
    // Texture coordinates and range
    output.texcoords_range = float3(texcoordsModifier(input), dist);
    
    return output;
}