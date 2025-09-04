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

// Animation parameters (available for all geometry)
float2 windVec;
float time;
bool hasAlpha;  // For detecting alpha-tested geometry

#ifdef HAS_GRASS
// Additional grass-specific parameters
float3 eyePos;
float2 footPos;
#endif

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

#ifdef HAS_GRASS
// Basic noise function for grass animation
float bnoise(float x) {
    float i = floor(x);
    float f = frac(x);
    float s = sign(frac(x / 2.0) - 0.5);
    float k = frac(i * 0.1731);
    return s * f * (f - 1.0) * ((16.0 * k - 4.0) * f * (f - 1.0) - 1.0);
}
#endif

// Grass displacement function based on wind and player proximity
float3 grassDisplacement(float3 worldpos, float h, float speed) {
    float v = length(windVec);
    float2 displace = 2 * v * 0.001 + 0.5;
    float2 harmonics = 0;

    float gtime = time * 0.1 * speed;

    float fi = 5.5;
    float cg = 1.0;
    float bi = 0.05;


    harmonics.x += abs(((fi * 1.0 + 0.03 * v) * sin(-2 * cg * (worldpos.x + worldpos.y + worldpos.z + gtime))) + bi * 1);

    harmonics.y += abs(((fi * 2.0 + 0.044 * v) * sin(-3 * cg * (worldpos.x + worldpos.y + worldpos.z + gtime))) + bi * 0.5);

    float3 stomp = 0;
#ifdef HAS_GRASS
    float d = length(worldpos.xy - footPos.xy);
    //d += pow(0.06 * max(0, footPos.- worldpos.z - 60), 2);


    if (d < 150) {
        stomp.xy = (60 / d - 0.4) * (worldpos.xy - footPos.xy);
    }
    stomp.z = 0;
#endif

    return float3(saturate(0.01 * (abs(h))) * speed * 5 * (harmonics.xy * displace.xy + stomp.xy), 0);
}

//------------------------------------------------------------
// Vertex Shader Main
VS_OUTPUT vs_main(VS_INPUT input) {
    VS_OUTPUT output;

    // Transform vertex
    float4 worldpos;
    float4 viewpos;
    float3 normal;


    // Standard transformation for non-grass
    if (vertexBlendState.x > 0.5) {
        // Skinned vertex
        viewpos = skin(input.pos, input.blendweights);
        normal = skin(float4(input.normal, 0), input.blendweights).xyz;
    }
    else {
        // Rigid vertex

#ifdef HAS_GRASS
        // Use grass displacement for grass geometry
        float3 displacement = grassDisplacement(input.pos.xyz, input.pos.z, 0.5);
        input.pos.xy += (1 - input.color.z) * displacement.xy;
#else
        // Apply simple wind animation to alpha-tested geometry (trees, bushes, etc.)
        if (hasAlpha) {
            float3 displacement = grassDisplacement(input.pos.xyz, input.pos.z, 1.0);
            input.pos.xyz += displacement;
        }
#endif

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