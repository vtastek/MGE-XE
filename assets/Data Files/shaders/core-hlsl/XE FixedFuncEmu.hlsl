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
float4 shadingMode; // .z = materialMode (1=none, 2=diffamb, 3=emissive)

// Lighting - Basic
float3 lightSceneAmbient;
float3 lightSunDiffuse;
float3 lightSunDirection;

// Lighting - Point lights
float4 lightDiffuse[8];  // Changed to float4 to match D3DXVECTOR4 from C++
float3 lightPosition[8]; // Proper float3 positions
float lightAmbient[8];
float lightFalloffQuadratic[8];
float lightFalloffConstant;
int pointLightCount; // Number of real point lights

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
    float3 viewpos : TEXCOORD2;  // View space position
    float fog : FOG;
};

//------------------------------------------------------------
// Helper Functions

// Skinning function from XE Common.hlsl
float4 skin(float4 pos, float4 blend) {
    float blendState = vertexBlendState.x;
    
    // Calculate missing blend weights
    if(blendState == 1)
        blend.y = 1 - blend.x;
    else if(blendState == 2)
        blend.z = 1 - (blend.x + blend.y);
    else if(blendState == 3)
        blend.w = 1 - (blend.x + blend.y + blend.z);
    
    // Weighted blend of matrices - ROW MAJOR (pos * matrix)
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
    output.viewpos = viewpos.xyz;
    
    // Simple fog
    float dist = length(viewpos);
    output.fog = fogMWScalar(dist);
    
    return output;
}

//------------------------------------------------------------
// Pixel Shader

float4 ps_main(VS_OUTPUT input) : COLOR {
    float3 normal = normalize(input.normal);
    
    // Basic lighting calculation
    float3 lighting = lightSceneAmbient;
    
    // Sun light
    float sunDot = saturate(dot(normal, -lightSunDirection));
    lighting += lightSunDiffuse * sunDot;
    
    // Point lights 
    float3 pointLightContribution = float3(0, 0, 0);
    for (int i = 0; i < pointLightCount; i++) {
        // Use proper float3 positions 
        float3 L = lightPosition[i] - input.viewpos;
        float dist = length(L);
        L = L / dist;
        
        float NdotL = saturate(dot(normal, L));
        
        // Use actual Effect shader falloff formula: 1/(quadratic*dist² + constant) - no linear term!
        float falloff = lightFalloffQuadratic[i] * dist * dist + lightFalloffConstant;
        float attenuation = (falloff > 0.0) ? (1.0 / falloff) : 0.0;
        
        // Add diffuse and ambient components like Effect shader
        pointLightContribution += (lightDiffuse[i].rgb * NdotL + lightAmbient[i]) * attenuation;
    }
    
    lighting += pointLightContribution;
    
    // Sample texture
    float4 texColor = tex2D(sampTex0, input.texcoord);
    
    // Material calculation - straightforward approach
    float3 effectiveDiffuse;
    float3 effectiveEmissive;
    float effectiveAlpha;
    
    int materialMode = (int)shadingMode.z;
    if (materialMode == 2) {
        // Mode 2: Use vertex color for diffuse/ambient
        effectiveDiffuse = input.color.rgb;
        effectiveEmissive = materialEmissive.rgb;
        effectiveAlpha = input.color.a;
    } else if (materialMode == 3) {
        // Mode 3: Use vertex color for emissive
        effectiveDiffuse = materialDiffuse.rgb;
        effectiveEmissive = input.color.rgb;
        effectiveAlpha = materialDiffuse.a;
    } else {
        // Mode 1: Use material constants
        effectiveDiffuse = materialDiffuse.rgb;
        effectiveEmissive = materialEmissive.rgb;
        effectiveAlpha = materialDiffuse.a;
    }
    
    float3 litColor = effectiveDiffuse * lighting;
    litColor += effectiveEmissive;
    
    float4 diffuse = float4(litColor, effectiveAlpha);
    
    // Apply texture
    float4 c = diffuse * texColor;
    
    // Apply fog
    // c.rgb = lerp(fogColNear, c.rgb, input.fog);
    
    return c;
}