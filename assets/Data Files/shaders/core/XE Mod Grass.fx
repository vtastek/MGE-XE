
// XE Mod Grass.fx
// MGE XE 0.16.0
// Grass rendering. Can be used as a core mod.

//------------------------------------------------------------
// Common functions

// Calculate mip level from texture coordinates (in texel space)
float CalcMipLevel(float2 texcoordTexels) {
    float2 dx = ddx(texcoordTexels);
    float2 dy = ddy(texcoordTexels);
    float delta = max(dot(dx, dx), dot(dy, dy));
    return max(0, 0.5 * log2(delta));
}

float grassCoverageAlpha(float alpha, float2 texcoords, float viewDepth) {
    float alphaRef = 64.0/255.0;
    float derivative = fwidth(alpha);
    float distanceFade = pow(saturate(viewDepth / 4000.0), 2.0);

#ifdef USE_HLSL_PIPELINE
    float mipLevel = CalcMipLevel(texcoords * a2cTexSize);
    alpha *= 1.0 + mipLevel * a2cMipScale;
    derivative = fwidth(alpha);
    float distanceSharpness = lerp(a2cSharpnessClose, a2cSharpnessFar, distanceFade);
#else
    float distanceSharpness = lerp(1.0, 0.25, distanceFade);
#endif

    float edgeWidth = max(derivative / max(distanceSharpness, 0.001), 1.0/255.0);
    return saturate(0.5 + 0.5 * (alpha - alphaRef) / edgeWidth);
}

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
    float4 worldDepth : TEXCOORD4;    // .xyz = world position, .w = view depth
};

GrassVertOut GrassInstVS(StatVertInstIn IN) {
    GrassVertOut OUT;
    TransformedVert v = transformGrassVert(IN);
    float3 eyevec = v.worldpos.xyz - eyePos.xyz;

    OUT.pos = v.pos;
    OUT.worldDepth = float4(v.worldpos.xyz, v.pos.w);
    OUT.fog = fogMWColour(length(eyevec));

    // Two-sided facing sign for grass. Keep it binary so lighting never crosses through zero.
    float facingSign = dot(eyevec, v.normal.xyz) < 0 ? 1.0 : -1.0;

#ifdef USE_HLSL_PIPELINE
    // Pass normal to PS; facing is recomputed per-pixel from world position.
    OUT.normalFacing = float4(v.normal.xyz, 1.0);
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
    float3 eyevec = IN.worldDepth.xyz - eyePos.xyz;
    float facingSign = dot(eyevec, normal) < 0 ? 1.0 : -1.0;
	float3 albedoLin = toLinearSrgb(result.rgb);
    // Two-sided lambert with backscatter
    float lambert = max(0, dot(normal, -sunVec));

	float3 vt = -sunVec  + normal * 0.5;
	float vd = pow(saturate(dot(normalize(eyevec), vt)), 2.0) * 0.5 * albedoLin;
	float vl = 1.0 * (vd + toLinearSrgb(sunAmb)/PI);


	//lambert = dot(normal, -sunVec);
    //if(lambert < 0.0) lambert *= -0.5 * PI;
	
	//lambert = dot(normal, -normalize(sunVec));
	float shadows = 1 - saturate(v * shadecolor * 50.);   
    
    float3 sunColLin = toLinearSrgb(sunCol);
    float3 sunAmbLin = toLinearSrgb(sunAmb)/PI;

    float3 lit = albedoLin * (sunColLin * lambert * shadows  +  sunColLin  * vl * shadows + sunAmbLin);
	
    lit *= intensityScalar;

    // Apply shadow darkening (towards blue like legacy)
    
	//result.rgb = lambert;
    result.rgb = fogApplyLinearAgX(lit, toLinearSrgb(fogColFar), IN.fog.a);
#else
    result.rgb *= IN.color.rgb;

    // Darken shadow area according to existing lighting (slightly towards blue)
    result.rgb *= 1 - v * shadecolor;

    // Fogging
    result.rgb = fogApply(result.rgb, IN.fog);
#endif

    // Alpha to coverage with mipmap compensation and fwidth sharpening
    result.a = grassCoverageAlpha(result.a, IN.texcoords, IN.worldDepth.w);

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

    // Grass is static (sway is cosmetic), no velocity
    OUT.curClip = OUT.pos;
    OUT.prevClip = OUT.pos;

    return OUT;
}

struct GrassDepthVelocityOut {
    float4 depth : COLOR0;
    float4 velocity : COLOR1;
};

GrassDepthVelocityOut DepthGrassInstPS(DepthVertOut IN) {
    GrassDepthVelocityOut OUT;

    clip(nearViewRange + 64.0 - IN.depth);

    float alpha = tex2D(sampBaseTex, IN.texcoords).a;
    if(alpha < 64.0/255.0)
        discard;

    float coverage = grassCoverageAlpha(alpha, IN.texcoords, IN.depth);

    OUT.depth = float4(IN.depth, IN.depth, IN.depth, coverage);
    OUT.velocity = float4(0, 0, 0, 1);

    return OUT;
}
