
// XE Mod Statics.fx
// MGE XE 0.16.0
// Distant statics rendering. Can be used as a core mod.


//------------------------------------------------------------
// Common functions

static const float windAnimationMaxDistance = 16384.0;

float4 animateStaticPos(StatVertIn IN) {
    float4 pos = IN.pos;
    float4 worldpos = mul(pos, world);

    // Match the HLSL fixed-function path for rigid alpha-tested geometry.
    if (hasAlpha && length(worldpos.xy - eyePos.xy) < windAnimationMaxDistance) {
        float3 displacement = grassDisplacement(IN.pos.xyz, IN.pos.z, 1.0);
        pos.xyz += displacement;
    }

    return pos;
}

TransformedVert transformStaticVert(StatVertIn IN) {
    // Transforms with implicit depth bias
    TransformedVert v;
    float4 pos = animateStaticPos(IN);

    v.worldpos = mul(pos, world);
    v.viewpos = mul(v.worldpos, view);
    v.pos = mul(v.viewpos, proj);

    // Cull vertices closer than nearViewRange threshold (preserves early-Z, no clip/discard)
    if (nearViewRange - 1500 > length(v.viewpos))
        v.pos = float4(0, 0, -10000, 0);

    return v;
}

TransformedVert transformStaticVertReflection(StatVertIn IN) {
    TransformedVert v;
    float4 pos = animateStaticPos(IN);

    v.worldpos = mul(pos, world);
    v.viewpos = mul(v.worldpos, reflectionView);
    v.pos = pancakeReflectionClip(mul(v.viewpos, reflectionProj));

    return v;
}

float4 lightStaticVert(StatVertIn IN) {
    // Decompress normal
    float4 normal = float4(normalize(2 * IN.normal.xyz - 1), 0);
    normal = mul(normal, world);

    // Lighting (worldspace)
    // Emissive is stored in the 4th value of the normal vector
    float emissive = IN.normal.w;
    float3 light = sunCol * saturate(dot(normal.xyz, -sunVec)) + sunAmb + emissive;

    return float4(IN.color.rgb * light, IN.color.a);
}

#ifdef USE_HLSL_PIPELINE
// HLSL path: return vertex color only, lighting done in PS
float4 lightStaticVertHLSL(StatVertIn IN, out float4 normalEmissive) {
    float4 normal = float4(normalize(2 * IN.normal.xyz - 1), 0);
    normal = mul(normal, world);
    normalEmissive = float4(normal.xyz, IN.normal.w);
    return IN.color;
}
#endif

float2 texcoordsModifier(StatVertIn IN) {
    float2 tc = IN.texcoords;

    if (hasVCol) {
        // Linked to animateUV static flag
        // Render with fixed scrolling that approximates ghostfence
        tc.y += fmod(0.08 * time, 1);
    }
    return tc;
}

//------------------------------------------------------------
// Statics rendering

StatVertOut StaticExteriorVS(StatVertIn IN) {
    StatVertOut OUT;
    TransformedVert v = transformStaticVert(IN);
    OUT.pos = v.pos;
#ifdef USE_HLSL_PIPELINE
    OUT.color = lightStaticVertHLSL(IN, OUT.normalEmissive);
#else
    OUT.color = lightStaticVert(IN);
#endif

    // Fogging (exterior)
    float3 eyevec = v.worldpos.xyz - eyePos.xyz;
    float dist = length(eyevec);
    OUT.fog = fogColour(eyevec / dist, dist);

    OUT.texcoords_range = float3(texcoordsModifier(IN), dist);
    return OUT;
}

StatVertOut StaticInteriorVS (StatVertIn IN) {
    StatVertOut OUT;
    TransformedVert v = transformStaticVert(IN);
    OUT.pos = v.pos;
#ifdef USE_HLSL_PIPELINE
    OUT.color = lightStaticVertHLSL(IN, OUT.normalEmissive);
#else
    OUT.color = lightStaticVert(IN);
#endif

    // Fogging (interior)
    float dist = length(v.viewpos.xyz);
    OUT.fog = fogMWColour(dist);

    OUT.texcoords_range = float3(texcoordsModifier(IN), dist);
    return OUT;
}

StatVertOut StaticExteriorReflVS(StatVertIn IN) {
    StatVertOut OUT;
    TransformedVert v = transformStaticVertReflection(IN);
    OUT.pos = v.pos;
#ifdef USE_HLSL_PIPELINE
    OUT.color = lightStaticVertHLSL(IN, OUT.normalEmissive);
#else
    OUT.color = lightStaticVert(IN);
#endif

    float3 eyevec = v.worldpos.xyz - eyePos.xyz;
    float dist = length(eyevec);
    OUT.fog = fogColour(eyevec / dist, dist);

    OUT.texcoords_range = float3(texcoordsModifier(IN), dist);
    return OUT;
}

StatVertOut StaticInteriorReflVS(StatVertIn IN) {
    StatVertOut OUT;
    TransformedVert v = transformStaticVertReflection(IN);
    OUT.pos = v.pos;
#ifdef USE_HLSL_PIPELINE
    OUT.color = lightStaticVertHLSL(IN, OUT.normalEmissive);
#else
    OUT.color = lightStaticVert(IN);
#endif

    float dist = length(v.viewpos.xyz);
    OUT.fog = fogMWColour(dist);

    OUT.texcoords_range = float3(texcoordsModifier(IN), dist);
    return OUT;
}

float4 StaticPS (StatVertOut IN): COLOR0 {
    float2 texcoords = IN.texcoords_range.xy;
    float range = IN.texcoords_range.z;

    float4 result = tex2D(sampBaseTex, texcoords);

#ifdef USE_HLSL_PIPELINE
    // Per-pixel lighting in linear space matching FFE/landscape.
    // IN.color.rgb = vertex color, IN.normalEmissive = world normal + emissive.
    float3 normal = normalize(IN.normalEmissive.xyz);
    float emissive = IN.normalEmissive.w;
    float NdotL = saturate(dot(normal, -sunVec));

    float3 albedoLin = toLinearSrgb(result.rgb);
    float3 vertColLin = toLinearSrgb(IN.color.rgb);
    float3 sunColLin = toLinearSrgb(sunCol);
    float3 sunAmbLin = toLinearSrgb(sunAmb) / PI;

    float3 lit = albedoLin * vertColLin * (sunColLin * NdotL + sunAmbLin + emissive);
    lit *= intensityScalar;
    result.rgb = fogApplyLinearAgX(lit, toLinearSrgb(fogColFar), IN.fog.a);
#else
    result.rgb *= IN.color.rgb;
    result.rgb = fogApply(result.rgb, IN.fog);
#endif

    // Alpha to coverage conversion
    result.a = calc_coverage(result.a, 133.0/255.0, 2.0);

    return result;
}

//------------------------------------------------------------
// Depth buffer output

DepthVertOut DepthStaticVS (StatVertIn IN) {
    DepthVertOut OUT;

    TransformedVert v = transformStaticVert(IN);
    OUT.pos = v.pos;

    OUT.depth = OUT.pos.w;
    OUT.alpha = 1;
    OUT.texcoords = texcoordsModifier(IN);

    // Static distant statics have no velocity
    OUT.curClip = OUT.pos;
    OUT.prevClip = OUT.pos;

    return OUT;
}

float4 DepthStaticPS (DepthVertOut IN) : COLOR0 {
    clip(IN.depth - nearViewRange);

    if(hasAlpha) {
        float alpha = tex2D(sampBaseTex, IN.texcoords).a;
        clip(alpha - 133.0/255.0);
    }
    return IN.depth;
}
