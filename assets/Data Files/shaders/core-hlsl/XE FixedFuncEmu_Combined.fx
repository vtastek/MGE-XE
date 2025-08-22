// XE FixedFuncEmu_Combined.fx
// MGE XE 0.16.0
// Combined HLSL shader for Morrowind's object rendering
// Includes all necessary common functions inline

//------------------------------------------------------------
// Uniform variables - using constant registers for D3D9

// Vertex shader constants
matrix worldViewProj : register(c0);
matrix view : register(c4);
matrix proj : register(c8);
matrix world : register(c12);
matrix vertexBlendPalette[4] : register(c16);  // c16-c31
float4 vertexBlendState : register(c32);

// Pixel shader constants
float4 materialDiffuse : register(c0);
float4 materialAmbient : register(c1);
float4 materialEmissive : register(c2);
float4 lightSunDirection : register(c3);
float4 lightSunDiffuse : register(c4);
float4 lightSceneAmbient : register(c5);
float4 fogColNear : register(c6);

// Legacy shared variables for compatibility
float2 rcpRes;
float shadowRcpRes;
matrix shadowViewProj[2];
bool hasAlpha, hasBones, hasVCol;
float alphaRef, materialAlpha;
float3 eyePos, footPos;
float3 sunVec, sunVecView, sunCol, sunAmb;
float3 skyCol, fogColFar;
float4 skyScatterColFar;
float fogStart, fogRange;
float nearFogStart, nearFogRange;
float nearViewRange;
float3 sunPos;
float sunVis;
float2 windVec;
float niceWeather;
float time;

//------------------------------------------------------------
// Textures

texture tex0, tex1, tex2, tex3, tex4, tex5;

sampler sampBaseTex = sampler_state { 
    texture = <tex0>; 
    minfilter = anisotropic; 
    magfilter = linear; 
    mipfilter = linear; 
    addressu = wrap; 
    addressv = wrap; 
};

sampler sampNormals = sampler_state { 
    texture = <tex1>; 
    minfilter = anisotropic; 
    magfilter = linear; 
    mipfilter = linear; 
    addressu = wrap; 
    addressv = wrap; 
};

sampler sampDetail = sampler_state { 
    texture = <tex2>; 
    minfilter = anisotropic; 
    magfilter = linear; 
    mipfilter = linear; 
    addressu = wrap; 
    addressv = wrap; 
};

sampler sampDepth = sampler_state { 
    texture = <tex3>; 
    minfilter = linear; 
    magfilter = linear; 
    mipfilter = none; 
    addressu = clamp; 
    addressv = clamp; 
};

sampler sampFFE0 = sampler_state { texture = <tex0>; };
sampler sampFFE1 = sampler_state { texture = <tex1>; };
sampler sampFFE2 = sampler_state { texture = <tex2>; };
sampler sampFFE3 = sampler_state { texture = <tex3>; };
sampler sampFFE4 = sampler_state { texture = <tex4>; };
sampler sampFFE5 = sampler_state { texture = <tex5>; };

//------------------------------------------------------------
// Additional material constants

matrix worldview;
float3 lightDiffuse[8];
float4 lightAmbient[2];
float4 lightPosition[6];
float4 lightFalloffQuadratic[2], lightFalloffLinear[2];
float lightFalloffConstant;
matrix texgenTransform;
float4 bumpMatrix;
float2 bumpLumiScaleBias;

//------------------------------------------------------------
// Configuration flags

#define FFE_LIGHTS_ACTIVE 8

//------------------------------------------------------------
// Fogging functions

float fogMWScalar(float dist) {
    return saturate((nearFogRange - dist) / (nearFogRange - nearFogStart));
}

//------------------------------------------------------------
// Skinning function - HLSL version with float blend state

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

//------------------------------------------------------------
// Vertex material routing

float4 vertexMaterial(float4 vertexColour) {
    if (hasVCol) {
        return vertexColour;
    }
    else {
        return float4(1, 1, 1, materialAlpha);
    }
}

//------------------------------------------------------------
// Transform library functions

float4 rigidVertex(float4 pos) { return mul(pos, worldview); }
float3 rigidNormal(float3 normal) { return mul(float4(normal, 0), worldview).xyz; }

float4 skinnedVertex(float4 pos, float4 weights) { return skin(pos, weights); }
float3 skinnedNormal(float3 normal, float4 weights) { return skin(float4(normal, 0), weights).xyz; }

float3 texgenNormal(float3 normal) { return normalize(normal); }
float3 texgenPosition(float4 pos) { return pos.xyz; }
float3 texgenReflection(float4 pos, float3 normal) { return reflect(normalize(pos.xyz), normalize(normal)); }
float3 texgenSphere(float2 tex) { return float3(0.5 * tex + 0.5, 0); }

//------------------------------------------------------------
// Lighting library functions

static const int LGs = max(1, ceil(FFE_LIGHTS_ACTIVE / 4.0));

float4 calcLighting4(float4 lightvec[3*LGs], int group, float3 normal) {
    float4 dist2 = 0, lambert = 0;

    for(int i = 0; i != 3; ++i)
        dist2 += pow(lightvec[3*group + i], 2);

    for(int i = 0; i != 3; ++i)
        lambert += normal[i] * lightvec[3*group + i];

    float4 dist = sqrt(dist2);
    lambert = saturate(lambert / dist);

    float4 att = 1.0 / (lightFalloffQuadratic[group] * dist2 + lightFalloffConstant);
    return (lambert + lightAmbient[group]) * att;
}

float3 calcPointLighting(uniform int lights, float4 lightvec[3*LGs], float3 normal) {
    float4 lambert[LGs];
    float3 l = 0;

    for(int i = 0; i != LGs; ++i)
        lambert[i] = calcLighting4(lightvec, i, normal);

    for(int i = 0; i != lights; ++i)
        l += lambert[i/4][i%4] * lightDiffuse[i];

    return l;
}

float3 tonemap(float3 c) {
    c = clamp(c, 0, 2.2);
    c = (((0.0548303 * c - 0.189786) * c - 0.154732) * c + 1.12969) * c;
    return c;
}

// Vertex material routing
float4 vertexMaterialNone(float3 d, float3 a) {
    return float4(materialDiffuse.rgb * d + materialAmbient.rgb * a + materialEmissive.rgb, materialDiffuse.a);
}

float4 vertexMaterialDiffAmb(float3 d, float3 a, float4 col) {
    return float4(col.rgb * (d + a) + materialEmissive.rgb, col.a);
}

float4 vertexMaterialEmissive(float3 d, float3 a, float4 col) {
    return float4(materialDiffuse.rgb * d + materialAmbient.rgb * a + col.rgb, materialDiffuse.a);
}

// Bumpmap stages
float4 bumpmapStage(sampler s, float2 tc, float4 dUdV) {
    float2 offset = mul(dUdV.rg, float2x2(bumpMatrix.xy, bumpMatrix.zw));
    return float4(tex2D(s, tc + offset).rgb, dUdV.a);
}

float4 bumpmapLumiStage(sampler s, float2 tc, float4 dUdVL) {
    float4 c = bumpmapStage(s, tc, dUdVL);
    c.rgb *= saturate(dUdVL.b * bumpLumiScaleBias.x + bumpLumiScaleBias.y);
    return c;
}

//------------------------------------------------------------
// Shader input/output structures

struct FFEVertIn {
    float4 pos : POSITION;
    float3 nrm : NORMAL;
    float4 col : COLOR0;
    float2 texcoord0 : TEXCOORD0;
    float4 blendweights : BLENDWEIGHT;
};

struct FFEPixelInput {
    float4 pos : POSITION;
    centroid float4 nrm_fog : NORMAL;
    centroid float4 texcoord01 : TEXCOORD0;
    centroid float4 col : COLOR0;
    float4 lightvec[3*LGs] : TEXCOORD2;
};

//------------------------------------------------------------
// Vertex Shader Entry Points

FFEPixelInput PerPixelVS_Rigid(FFEVertIn input) {
    FFEPixelInput output;

    // Use the rigidVertex function like the original shader does
    float4 viewPos = rigidVertex(input.pos);
    float3 normal = rigidNormal(input.nrm);
    
    float dist = length(viewPos);
    output.pos = mul(viewPos, proj);
    output.nrm_fog = float4(normal, fogMWScalar(dist));

    // Texcoord routing and texgen
    float3 texgen = texgenReflection(viewPos, normal);
    texgen = mul(float4(texgen, 1), texgenTransform).xyz;
    output.texcoord01 = float4(input.texcoord0, texgen.xy);

    // Vertex colour
    output.col = input.col;

    // Point lighting setup, vectorized (like original)
    for(int i = 0; i != LGs; ++i) {
        output.lightvec[3*i + 0] = lightPosition[i + 0] - viewPos.x;
        output.lightvec[3*i + 1] = lightPosition[i + 2] - viewPos.y;
        output.lightvec[3*i + 2] = lightPosition[i + 4] - viewPos.z;
    }

    return output;
}

FFEPixelInput PerPixelVS_Skinned(FFEVertIn input) {
    FFEPixelInput output;

    // Use the skinnedVertex function like the original shader does
    float4 viewPos = skinnedVertex(input.pos, input.blendweights);
    float3 normal = skinnedNormal(input.nrm, input.blendweights);
    
    float dist = length(viewPos);
    output.pos = mul(viewPos, proj);
    output.nrm_fog = float4(normal, fogMWScalar(dist));

    // Texcoord routing and texgen
    float3 texgen = texgenReflection(viewPos, normal);
    texgen = mul(float4(texgen, 1), texgenTransform).xyz;
    output.texcoord01 = float4(input.texcoord0, texgen.xy);

    // Vertex colour
    output.col = input.col;

    // Point lighting setup, vectorized (like original)
    for(int i = 0; i != LGs; ++i) {
        output.lightvec[3*i + 0] = lightPosition[i + 0] - viewPos.x;
        output.lightvec[3*i + 1] = lightPosition[i + 2] - viewPos.y;
        output.lightvec[3*i + 2] = lightPosition[i + 4] - viewPos.z;
    }

    return output;
}

//------------------------------------------------------------
// Pixel Shader Entry Points

float4 PerPixelPS_Standard(FFEPixelInput input) : COLOR0 {
    float3 normal = normalize(input.nrm_fog.xyz);
    float fog = input.nrm_fog.w;

    // Standard morrowind lighting: sun, ambient, and point lights (like original)
    float3 d = lightSunDiffuse * saturate(dot(normal, -lightSunDirection));
    float3 a = lightSceneAmbient;
    d += calcPointLighting(FFE_LIGHTS_ACTIVE, input.lightvec, normal);

    // Material - vertex colour version (like original)
    float4 diffuse = vertexMaterialDiffAmb(d, a, input.col);

    // Texturing and combinators (like original)
    float4 c = diffuse + bumpmapLumiStage(sampFFE1, input.texcoord01.zw, tex2D(sampFFE0, input.texcoord01.xy));

    // Static tonemap and final fogging (like original)
    c.rgb = tonemap(c.rgb);
    c.rgb = lerp(fogColNear, c.rgb, fog);

    return c;
}

float4 PerPixelPS_Error(FFEPixelInput input) : COLOR0 {
    // Error material - red shader for debugging
    return float4(1, 0, 0.5, 1);
}