
// XE Mod Grass.fx
// MGE XE 0.16.0
// Grass rendering. Can be used as a core mod.

//------------------------------------------------------------
// Common functions

TransformedVert transformGrassVert(StatVertInstIn IN) {
    TransformedVert v;

    v.worldpos = instancedMul(IN.pos, IN.world0, IN.world1, IN.world2);

    // Transforms with wind displacement
    v.worldpos.xy += grassDisplacement(v.worldpos, IN.pos.z, 1.0);
    v.viewpos = mul(v.worldpos, view);
    v.pos = mul(v.viewpos, proj);

    // Decompress normal
    float4 normal = float4(normalize(2 * IN.normal.xyz - 1), 0);
    v.normal = instancedMul(normal, IN.world0, IN.world1, IN.world2);
    return v;
}

//------------------------------------------------------------
// Grass

struct GrassVertOut {
    float4 pos : POSITION;
    half2 texcoords : TEXCOORD0;
    centroid half4 color : COLOR0;
    centroid half4 fog : COLOR1;

    float4 shadow0pos : TEXCOORD1;
    float4 shadow1pos : TEXCOORD2;
#ifdef USE_HLSL_PIPELINE
    float4 normalFacing : TEXCOORD3;  // .xyz = world normal, .w = facing sign for two-sided
#endif
};

GrassVertOut GrassInstVS(StatVertInstIn IN) {
    GrassVertOut OUT;
    TransformedVert v = transformGrassVert(IN);
    float3 eyevec = v.worldpos.xyz - eyePos.xyz;

    OUT.pos = v.pos;
    OUT.fog = fogMWColour(length(eyevec));

    // Two-sided facing sign for grass (flip normal based on view direction)
    float facingSign = -sign(dot(eyevec, v.normal.xyz));

#ifdef USE_HLSL_PIPELINE
    // Pass normal and facing sign to PS for per-pixel lighting
    OUT.normalFacing = float4(v.normal.xyz, facingSign);
    // color.rgb unused in HLSL PS (lighting computed there), but set for shadow estimate
    float lambert = dot(v.normal.xyz, -sunVec) * facingSign;
    if(lambert < 0) lambert *= -0.3;
    OUT.color.rgb = float3(1, 1, 1);  // placeholder, lighting done in PS
#else
    // Lighting for two-sided rendering, no emissive
    float lambert = dot(v.normal.xyz, -sunVec) * facingSign;
    if(lambert < 0) lambert *= -0.3;
    OUT.color.rgb = sunCol * lambert + sunAmb;
#endif

    // Non-standard shadow luminance, to create sufficient contrast when ambient is high
    OUT.color.a = shadowSunEstimate(lambert);

    // Find position in light space, output light depth
    OUT.shadow0pos = mul(v.worldpos, shadowViewProj[0]);
    OUT.shadow1pos = mul(v.worldpos, shadowViewProj[1]);
    OUT.shadow0pos.z = OUT.shadow0pos.z / OUT.shadow0pos.w;
    OUT.shadow1pos.z = OUT.shadow1pos.z / OUT.shadow1pos.w;

    OUT.texcoords = IN.texcoords;
    return OUT;
}

float4 GrassPS(GrassVertOut IN): COLOR0 {
    float4 result = tex2D(sampBaseTex, IN.texcoords);

    // Alpha test early
    if(result.a < 64.0/255.0)
        discard;

    // Soft shadowing
    float dz = shadowDeltaZ(IN.shadow0pos, IN.shadow1pos);
    float v = shadowESM(dz);
    v *= IN.color.a;

#ifdef USE_HLSL_PIPELINE
    // Per-pixel lighting in linear space matching FFE/landscape.
    float3 normal = normalize(IN.normalFacing.xyz);
    float facingSign = IN.normalFacing.w;

    // Two-sided lambert with backscatter
    float lambert = dot(normal, -sunVec) * facingSign;
    if(lambert < 0) lambert *= -0.3;

    float3 albedoLin = toLinearSrgb(result.rgb);
    float3 sunColLin = toLinearSrgb(sunCol);
    float3 sunAmbLin = toLinearSrgb(sunAmb) / PI;

    float3 lit = albedoLin * (sunColLin * lambert + sunAmbLin);
    lit *= intensityScalar;

    // Apply shadow darkening (towards blue like legacy)
    lit *= 1 - v * shadecolor;

    result.rgb = fogApplyLinearAgX(lit, toLinearSrgb(fogColFar), IN.fog.a);
#else
    result.rgb *= IN.color.rgb;

    // Darken shadow area according to existing lighting (slightly towards blue)
    result.rgb *= 1 - v * shadecolor;

    // Fogging
    result.rgb = fogApply(result.rgb, IN.fog);
#endif

    // Alpha to coverage conversion
    result.a = calc_coverage(result.a, 128.0/255.0, 4.0);

    return result;
}

//------------------------------------------------------------
// Depth buffer output


DepthVertOut DepthGrassInstVS(StatVertInstIn IN) {
    DepthVertOut OUT;
    TransformedVert v = transformGrassVert(IN);

    OUT.pos = v.pos;
    OUT.depth = v.pos.w;
    OUT.alpha = 1;
    OUT.texcoords = IN.texcoords;

    return OUT;
}