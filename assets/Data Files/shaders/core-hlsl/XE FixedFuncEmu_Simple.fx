// XE FixedFuncEmu_Simple.fx
// Simplified HLSL shader based on working approach

// Vertex shader constants (matches Morrowind's skinning system)
float4x4 vertexBlendPalette[4] : register(c0);  // c0-c15: 4 bone matrices
float4x4 proj : register(c16);                  // c16-c19: projection matrix
float4 vertexBlendState : register(c20);        // c20.x = blend state (0=rigid, 1-3=skinned)
float4 materialAlpha : register(c21);           // c21.x = material alpha

// Vertex input
struct VS_INPUT {
    float4 pos : POSITION;
    float4 blendweights : BLENDWEIGHT;
    float3 normal : NORMAL;
    float4 color : COLOR0;
    float2 texcoord : TEXCOORD0;
};

struct VS_OUTPUT {
    float4 position : POSITION;
    float3 normal : TEXCOORD0;
    float3 viewpos : TEXCOORD1;
    float4 color : COLOR0;
    float2 texcoord : TEXCOORD2;
    float fog : FOG;
};

// Skinning function from MGE (matches your working shader)
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

VS_OUTPUT vs_main(VS_INPUT IN) {
    VS_OUTPUT OUT;
    
    float4 viewPos;
    float3 viewNormal;
    
    // Use Combined shader's working approach
    if (vertexBlendState.x > 0.5) {
        // Skinned vertex - use skin function like Combined shader
        viewPos = skin(IN.pos, IN.blendweights);
        viewNormal = skin(float4(IN.normal, 0), IN.blendweights).xyz;
    } else {
        // Non-skinned - use worldview matrix like Combined shader
        float4x4 worldview = vertexBlendPalette[0];  // Assuming first matrix is worldview
        viewPos = mul(IN.pos, worldview);
        viewNormal = mul(float4(IN.normal, 0), worldview).xyz;
    }
    
    // Project to screen space - ROW MAJOR again
    OUT.position = mul(viewPos, proj);
    OUT.viewpos = viewPos.xyz;
    OUT.normal = normalize(viewNormal);
    
    // Vertex color
    OUT.color = IN.color;
    
    // Texture coordinate
    OUT.texcoord = IN.texcoord;
    
    // Simple fog calculation
    float fogStart = 2000.0;
    float fogEnd = 4500.0;
    OUT.fog = saturate((fogEnd - viewPos.z) / (fogEnd - fogStart));
    
    return OUT;
}

// Pixel shader constants
float4 materialDiffuse : register(c0);
float4 materialAmbient : register(c1);
float4 materialEmissive : register(c2);
float3 lightSceneAmbient : register(c3);
float3 lightSunDiffuse : register(c4);
float3 lightSunDirection : register(c5);
float4 shadingMode : register(c6); // .x = hasTex, .y = hasVtxColor, .z = materialMode

// Texture samplers
sampler2D tex0 : register(s0);

// Pixel shader
float4 ps_main(VS_OUTPUT IN) : COLOR {
    float3 normal = normalize(IN.normal);
    float NdotL_sun = saturate(dot(normal, -lightSunDirection));
    float3 sun_light = lightSunDiffuse * NdotL_sun;
    float3 ambient_light = max(lightSceneAmbient, float3(0.3, 0.3, 0.3));
    
    // Combine Sun and Ambient lights
    float3 total_light = sun_light + ambient_light;

    float4 surfaceAlbedo = tex2D(tex0, IN.texcoord);
    
    // More sophisticated material handling (like your working shader)
    // Check shading mode to determine how to combine material and vertex colors
    float3 effectiveDiffuse;
    float3 effectiveEmissive;
    float effectiveAlpha;
    
    if (shadingMode.z > 1.5 && shadingMode.z < 2.5) {
        // Mode 2: Use vertex color for diffuse
        effectiveDiffuse = IN.color.rgb;
        effectiveEmissive = materialEmissive.rgb;
        effectiveAlpha = IN.color.a;
    } else if (shadingMode.z > 2.5) {
        // Mode 3: Use vertex color for emissive
        effectiveDiffuse = materialDiffuse.rgb;
        effectiveEmissive = IN.color.rgb;
        effectiveAlpha = materialDiffuse.a;
    } else {
        // Mode 0/1: Use material colors
        effectiveDiffuse = materialDiffuse.rgb;
        effectiveEmissive = materialEmissive.rgb;
        effectiveAlpha = materialDiffuse.a;
    }
    
    // Ensure we have some minimum lighting to avoid completely dark objects
    total_light = max(total_light, float3(0.1, 0.1, 0.1));
    
    float3 litColor = effectiveDiffuse * surfaceAlbedo.rgb * total_light;
    litColor += surfaceAlbedo.rgb * effectiveEmissive;
    
    float finalAlpha = surfaceAlbedo.a * effectiveAlpha;

    float3 fogColor = float3(0.6, 0.65, 0.7);
    litColor = lerp(fogColor, litColor, IN.fog);
        
    return float4(litColor, finalAlpha);
}