// XE_Land_VS.hlsl
// MGE XE Phase 8 - Distant Land HLSL Vertex Shader for Landscape
// Simplified permutation-based version of XE Mod Landscape.fx

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

// Input structure for landscape vertices
struct VSInput {
    float4 pos : POSITION;
    float3 normal : NORMAL;
    float2 texcoords : TEXCOORD0;
};

// Output structure
struct VSOutput {
    float4 pos : POSITION;
    float4 color : COLOR0;
    float4 fog : TEXCOORD0;
    float2 texcoords : TEXCOORD1;
};

// Fog color calculation 
float4 fogColour(float3 eyeVecNorm, float dist) {
    #ifdef USE_ATM_SCATTER
    // Atmospheric scattering would go here
    float fogFactor = saturate((dist - fogStart) / fogRange);
    return lerp(float4(fogColNear, 1.0), float4(fogColFar, 1.0), fogFactor);
    #else
    // Simple fog
    float fogFactor = saturate((dist - fogStart) / fogRange);
    return float4(fogColNear, fogFactor);
    #endif
}

VSOutput main(VSInput input) {
    VSOutput output;
    
    // Transform vertex
    float4 worldpos = mul(input.pos, world);
    float4 viewpos = mul(worldpos, view);
    output.pos = mul(viewpos, proj);
    
    // Simple diffuse lighting for landscape
    float3 worldNormal = normalize(mul(float4(input.normal, 0.0), world).xyz);
    float diffuse = saturate(dot(worldNormal, -sunVec));
    output.color = float4(sunCol * diffuse + sunAmb, 1.0);
    
    // Fogging
    float3 eyevec = worldpos.xyz - eyePos.xyz;
    float dist = length(eyevec);
    output.fog = fogColour(eyevec / dist, dist);
    
    // Pass through texture coordinates
    output.texcoords = input.texcoords;
    
    return output;
}