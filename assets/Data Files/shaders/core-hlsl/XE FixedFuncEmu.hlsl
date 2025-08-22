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

// Lighting - Point lights (scattered arrays, same as C++ buffer format)
float3 lightDiffuse[8];
float lightPosition[24]; // 3*8 lights: [x0..x7, y0..y7, z0..z7] 
float lightAmbient[8];
float4 lightFalloffQuadratic[2];
float lightFalloffConstant;

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
    float3 worldpos : TEXCOORD2;
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
    output.worldpos = viewpos.xyz;
    
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
    
    // Point lights (using scattered array format like Effect shader)
    float3 pointLightContribution = float3(0, 0, 0);
    [loop]
    for (int i = 0; i < 8; i++) {
        // Access scattered position arrays: [x0..x7, y0..y7, z0..z7]
        float3 lightPos = float3(lightPosition[i], lightPosition[i + 8], lightPosition[i + 16]);
        float3 L = lightPos - input.worldpos;
        
        float distSq = dot(L, L);
        float dist = sqrt(distSq);
        L = L / dist;
        
        float NdotL = saturate(dot(normal, L));
        
        // Attenuation using Effect shader approach (quadratic + constant only)
        int group = i / 4;
        int index = i % 4;
        float attenuation = 1.0 / (lightFalloffQuadratic[group][index] * distSq + lightFalloffConstant);
        
        // Add diffuse and ambient components like Effect shader
        pointLightContribution += (lightDiffuse[i] * NdotL + lightAmbient[i]) * attenuation;
    }
    
    lighting += pointLightContribution;
    
    // Sample texture
    float4 texColor = tex2D(sampTex0, input.texcoord);
    
    // Material calculation (exact copy from debug shader)
    float isMode2 = step(1.5, shadingMode.z) * (1 - step(2.5, shadingMode.z));
    float isMode3 = step(2.5, shadingMode.z);
    
    float3 effectiveDiffuse = lerp(materialDiffuse.rgb, input.color.rgb, isMode2);
    float3 effectiveEmissive = lerp(materialEmissive.rgb, input.color.rgb, isMode3);
    float effectiveAlpha = lerp(materialDiffuse.a, input.color.a, isMode2);
    
    float3 litColor = effectiveDiffuse * lighting;
    litColor += effectiveEmissive;
    
    float4 diffuse = float4(litColor, effectiveAlpha);
    
    // Apply texture
    float4 c = diffuse * texColor;
    
    // Apply fog
    // c.rgb = lerp(fogColNear, c.rgb, input.fog);
    
    return c;
}