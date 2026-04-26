
// XE Mod Statics.fx
// MGE XE 0.16.0
// Distant statics rendering. Can be used as a core mod.


//------------------------------------------------------------
// Common functions

TransformedVert transformStaticVert(StatVertIn IN) {
    // Transforms with implicit depth bias
    TransformedVert v;

    v.worldpos = mul(IN.pos, world);
    v.viewpos = mul(v.worldpos, view);
    v.pos = mul(v.viewpos, proj);
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
    OUT.color = lightStaticVert(IN);

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
    OUT.color = lightStaticVert(IN);

    // Fogging (interior)
    float dist = length(v.viewpos.xyz);
    OUT.fog = fogMWColour(dist);

    OUT.texcoords_range = float3(texcoordsModifier(IN), dist);
    return OUT;
}

float4 StaticPS (StatVertOut IN): COLOR0 {
    float2 texcoords = IN.texcoords_range.xy;
    float range = IN.texcoords_range.z;

    float4 result = tex2D(sampBaseTex, texcoords);
    result.rgb *= IN.color.rgb;
    result.rgb = fogApply(result.rgb, IN.fog);

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

//------------------------------------------------------------
// Instanced variants of the static vertex shaders.
//
// Transforms read the world matrix from per-instance input (world0/1/2
// in stream 1 via StatVertInstIn) instead of the `world` uniform. The
// rest of the pipeline (lighting, texcoord modifier, fogging) matches
// the non-instanced variants byte-for-byte so visual output is identical.
//
// Used only when the C++ caller issues DrawIndexedPrimitive with
// D3DSTREAMSOURCE_INSTANCEDATA bound on stream 1 (see
// StaticInstancing::renderColor / renderDepth). The non-instanced
// variants above remain the default path; whether instancing is used
// is gated on Configuration.UseStaticInstancing.

TransformedVert transformStaticVertInst(StatVertInstIn IN) {
    TransformedVert v;

    v.worldpos = instancedMul(IN.pos, IN.world0, IN.world1, IN.world2);
    v.viewpos = mul(v.worldpos, view);
    v.pos = mul(v.viewpos, proj);
    return v;
}

float4 lightStaticVertInst(StatVertInstIn IN) {
    // Decompress normal; transform through the per-instance matrix.
    float4 normal = float4(normalize(2 * IN.normal.xyz - 1), 0);
    normal = instancedMul(normal, IN.world0, IN.world1, IN.world2);

    // Emissive is stored in the 4th value of the normal vector
    float emissive = IN.normal.w;
    float3 light = sunCol * saturate(dot(normal.xyz, -sunVec)) + sunAmb + emissive;

    return float4(IN.color.rgb * light, IN.color.a);
}

float2 texcoordsModifierInst(StatVertInstIn IN) {
    float2 tc = IN.texcoords;

    if (hasVCol) {
        // Linked to animateUV static flag; same scrolling as non-instanced path.
        tc.y += fmod(0.08 * time, 1);
    }
    return tc;
}

StatVertOut StaticExteriorInstVS(StatVertInstIn IN) {
    StatVertOut OUT;
    TransformedVert v = transformStaticVertInst(IN);
    OUT.pos = v.pos;
    OUT.color = lightStaticVertInst(IN);

    float3 eyevec = v.worldpos.xyz - eyePos.xyz;
    float dist = length(eyevec);
    OUT.fog = fogColour(eyevec / dist, dist);

    OUT.texcoords_range = float3(texcoordsModifierInst(IN), dist);
    return OUT;
}

StatVertOut StaticInteriorInstVS(StatVertInstIn IN) {
    StatVertOut OUT;
    TransformedVert v = transformStaticVertInst(IN);
    OUT.pos = v.pos;
    OUT.color = lightStaticVertInst(IN);

    float dist = length(v.viewpos.xyz);
    OUT.fog = fogMWColour(dist);

    OUT.texcoords_range = float3(texcoordsModifierInst(IN), dist);
    return OUT;
}

DepthVertOut DepthStaticInstVS(StatVertInstIn IN) {
    DepthVertOut OUT;

    TransformedVert v = transformStaticVertInst(IN);
    OUT.pos = v.pos;

    OUT.depth = OUT.pos.w;
    OUT.alpha = 1;
    OUT.texcoords = texcoordsModifierInst(IN);

    return OUT;
}