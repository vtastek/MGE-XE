//============================================================================
// MGE XE - HLSL Fixed Function Emulation - Vertex Shader
//============================================================================

// Matrices  
matrix proj;
matrix worldview;
matrix world;
matrix view;
matrix vertexBlendPalette[4];
float4 vertexBlendState;

#ifdef HAS_SHADOWS
// Shadow matrices - using shader constants c20-c27
matrix shadowViewProj[2] : register(c20);
#endif

// Fog parameters
float nearFogStart, nearFogRange;

//------------------------------------------------------------
// Vertex Input/Output
struct VS_INPUT {
    float4 pos : POSITION;
    float3 normal : NORMAL;
    float4 color : COLOR0;
    float2 texcoord : TEXCOORD0;
    float4 blendweights : BLENDWEIGHT;
};

struct VS_OUTPUT {
    float4 position : POSITION;
    float3 normal : TEXCOORD0;
    float4 color : COLOR0;
    float2 texcoord : TEXCOORD1;
    float3 viewPos : TEXCOORD2;  // View space position
    float fog : TEXCOORD3;       // Fog factor
#ifdef HAS_SHADOWS
    float4 shadow0pos : TEXCOORD4;  // Shadow map 0 position
    float4 shadow1pos : TEXCOORD5;  // Shadow map 1 position
#endif
};

//------------------------------------------------------------
// Utility Functions

float saturate(float x) { return max(0.0, min(x, 1.0)); }

// Skinning function
float4 skin(float4 pos, float4 blend) {
    float blendState = vertexBlendState.x;

    // Calculate missing blend weights
    if (blendState == 1)
        blend.y = 1 - blend.x;
    else if (blendState == 2)
        blend.z = 1 - (blend.x + blend.y);
    else if (blendState == 3)
        blend.w = 1 - (blend.x + blend.y + blend.z);

    // Weighted blend of matrices - ROW MAJOR (pos * matrix)
    float4 viewpos = mul(pos, vertexBlendPalette[0]) * blend.x;

    if (blendState >= 1)
        viewpos += mul(pos, vertexBlendPalette[1]) * blend.y;
    if (blendState >= 2)
        viewpos += mul(pos, vertexBlendPalette[2]) * blend.z;
    if (blendState >= 3)
        viewpos += mul(pos, vertexBlendPalette[3]) * blend.w;

    return viewpos;
}

// Fog calculation
float fogMWScalar(float dist) {
    return saturate((nearFogRange - dist) / (nearFogRange - nearFogStart));
}

//------------------------------------------------------------
// Vertex Shader Main
VS_OUTPUT vs_main(VS_INPUT input) {
    VS_OUTPUT output;

    // Transform vertex
    float4 viewpos;
    float3 normal;

    if (vertexBlendState.x > 0.5) {
        // Skinned vertex
        viewpos = skin(input.pos, input.blendweights);
        normal = skin(float4(input.normal, 0), input.blendweights).xyz;
    }
    else {
        // Rigid vertex
        viewpos = mul(input.pos, worldview);
        normal = mul(float4(input.normal, 0), worldview).xyz;
    }

    // Project to screen
    output.position = mul(viewpos, proj);

    // Pass through data
    output.normal = normalize(normal);
    output.color = input.color;
    output.texcoord = input.texcoord;
    output.viewPos = viewpos.xyz;

    // Calculate fog
    float dist = length(viewpos);
    output.fog = fogMWScalar(dist);

#ifdef HAS_SHADOWS
    // Transform view position to shadow coordinates (like original system)
    // Both skinned and rigid vertices end up in view space, so use that consistently
    output.shadow0pos = mul(viewpos, shadowViewProj[0]);
    output.shadow1pos = mul(viewpos, shadowViewProj[1]);
#endif

    return output;
}