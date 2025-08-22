// XE FixedFuncEmu_Simple.hlsl  
// MGE XE 0.16.0
// Proper HLSL shader for Morrowind object rendering

//------------------------------------------------------------
// Shared Variables

// Matrices
matrix proj;
matrix worldview;
matrix vertexBlendPalette[4];
float4 vertexBlendState;

// Materials
float4 materialDiffuse, materialAmbient, materialEmissive;

// Lighting - Basic
float3 lightSceneAmbient;
float3 lightSunDiffuse;
float3 lightSunDirection;

// Lighting - Point lights (arrays)
float3 lightDiffuse[8];
float4 lightPosition[6];

// Individual light constants for C++ compatibility
float3 lightDiffuse0, lightDiffuse1;
float3 lightPosition0, lightPosition1;

// Fog
float3 fogColNear;
float nearFogStart, nearFogRange;

// Textures
texture tex0, tex1;
sampler sampTex0 = sampler_state { texture = <tex0>; };
sampler sampTex1 = sampler_state { texture = <tex1>; };

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
    float fog : FOG;
};

//------------------------------------------------------------
// Helper Functions

// Skinning function
float4 skin(float4 pos, float4 blend) {
    float blendState = vertexBlendState.x;
    
    // Calculate missing blend weights
    if(blendState == 1)
        blend.y = 1 - blend.x;
    else if(blendState == 2)
        blend.z = 1 - (blend.x + blend.y);
    else if(blendState == 3)
        blend.w = 1 - (blend.x + blend.y + blend.z);
    
    // Weighted blend of matrices
    float4 viewpos = mul(pos, vertexBlendPalette[0]) * blend.x;
    
    if(blendState >= 1)
        viewpos += mul(pos, vertexBlendPalette[1]) * blend.y;
    if(blendState >= 2)
        viewpos += mul(pos, vertexBlendPalette[2]) * blend.z;
    if(blendState >= 3)
        viewpos += mul(pos, vertexBlendPalette[3]) * blend.w;
    
    return viewpos;
}

// Fog function
float fogMWScalar(float dist) {
    return saturate((nearFogRange - dist) / (nearFogRange - nearFogStart));
}

//------------------------------------------------------------
// Vertex Shader

VS_OUTPUT vs_main(VS_INPUT input) {
    VS_OUTPUT output;
    
    // Transform vertex
    float4 viewpos;
    float3 normal;
    
    if (vertexBlendState.x > 0.5) {
        // Skinned vertex
        viewpos = skin(input.pos, input.blendweights);
        normal = skin(float4(input.normal, 0), input.blendweights).xyz;
    } else {
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
    
    // Simple fog
    float dist = length(viewpos);
    output.fog = fogMWScalar(dist);
    
    return output;
}

//------------------------------------------------------------
// Pixel Shader

float4 ps_main(VS_OUTPUT input) : COLOR {
    float3 normal = normalize(input.normal);
    
    // Standard morrowind lighting: sun and ambient (matching Effect shader)
    float3 d = lightSunDiffuse * saturate(dot(normal, -lightSunDirection));
    float3 a = lightSceneAmbient;
    
    // Material - match Effect shader vertexMaterialDiffAmb logic
    float4 diffuse = float4(input.color.rgb * (d + a) + materialEmissive.rgb, input.color.a);
    
    // Texturing - multiply diffuse by texture (standard approach)
    float4 texColor = tex2D(sampTex0, input.texcoord);
    float4 c = diffuse * texColor;
    
    // Fog - match Effect shader
    c.rgb = lerp(fogColNear, c.rgb, input.fog);
    
    return c;
}