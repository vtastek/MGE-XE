
// XE Mod Landscape.fx
// MGE XE 0.16.0
// Distant landscape rendering. Can be used as a core mod.


//------------------------------------------------------------
// Common functions

TransformedVert transformLandVert(float4 pos) {
    TransformedVert v;

    v.viewpos = mul(mul(pos, world), view);
    v.pos = mul(v.viewpos, proj);
    return v;
}

//------------------------------------------------------------
// Distant land height bias to prevent low lod meshes from clipping

float landBias(float dist) {
    float maxDist = nearViewRange - 1152;
    return -30 + -2 * max(0, maxDist - dist);
}

//------------------------------------------------------------
// Distant landscape rendering

struct LandVertOut {
    float4 pos: POSITION;
    float2 texcoord: TEXCOORD0;
    centroid float4 fog : TEXCOORD1;
    float range : TEXCOORD2;       // camera distance, for the cache handover near-clip
};

LandVertOut LandscapeVS(float4 pos : POSITION, float2 texcoord : TEXCOORD0) {
    LandVertOut OUT;

    // Fogging
    float3 eyevec = mul(pos, world).xyz - eyePos.xyz;
    float dist = length(eyevec);
    OUT.fog = fogColour(eyevec / dist, dist);

    // Move land down to avoid it appearing where reduced mesh doesn't match
    pos.z += landBias(dist);

    // Transforms
    TransformedVert v = transformLandVert(pos);
    OUT.pos = v.pos;
    OUT.texcoord = texcoord;
    OUT.range = dist;
    return OUT;
}

LandVertOut LandscapeReflVS(float4 pos : POSITION, float2 texcoord : TEXCOORD0) {
    LandVertOut OUT;

    // Fogging
    float3 eyevec = mul(pos, world).xyz - eyePos.xyz;
    float dist = length(eyevec);
    if(isAboveSeaLevel(eyePos))
        OUT.fog = fogColour(eyevec / dist, dist);
    else
        OUT.fog = fogMWColour(dist);

    // Sink land near waterline a small amount
    pos.z += -16 * saturate(1 - pos.z/16);

    // Transforms
    TransformedVert v = transformLandVert(pos);
    OUT.pos = v.pos;
    OUT.texcoord = texcoord;
    OUT.range = dist;
    return OUT;
}

float4 LandscapePS(LandVertOut IN) : COLOR0 {
    // Cache handover: clip near land so the MGE cache draws the near field at full
    // resolution. Inert in the main view (landNearCull == 0).
    clip(IN.range - landNearCull);

    // Expand and normalize normal map
    float3 normal = normalize(2 * tex2D(sampNormals, IN.texcoord).rgb - 1);

    // World texture
    float3 result = tex2D(sampBaseTex, IN.texcoord).rgb;

    // Detail texture
    float detail = tex2D(sampDetail, IN.texcoord * 333).g + 0.5;
    detail *= 0.5 * tex2D(sampDetail, IN.texcoord * 90).g + 0.75;

    // Lighting
    result *= sunCol * saturate(dot(-sunVec, normal)) + sunAmb;
    result *= detail;

    // Fogging
    result = fogApply(result, IN.fog);
    return float4(result, 1);
}

DepthVertOut DepthLandVS(float4 pos: POSITION, float2 texcoord: TEXCOORD0) {
    DepthVertOut OUT;

    // Move land down to avoid it appearing where reduced mesh doesn't match
    pos.z += landBias(length(pos.xyz - eyePos.xyz));

    TransformedVert v = transformLandVert(pos);
    OUT.pos = v.pos;
    OUT.depth = v.pos.w;
    OUT.alpha = 1;
    OUT.texcoords = texcoord;

    return OUT;
}

float4 DepthLandPS(DepthVertOut IN) : COLOR0 {
    clip(IN.depth - nearViewRange);
    return IN.depth;
}

//------------------------------------------------------------
// Cache near terrain (reflections): the real Morrowind terrain mesh from the
// scene-graph cache, replacing the coarse distant-land LOD in the near field.
// Two-texture splat: base (tex0) + decal overlay (tex2) blended by the AlphaGrid
// carried in vertex-colour alpha (vanilla terrain texturing). Lit like the LOD
// land (sun N.L + ambient, using the cache normal) so the handover matches. The
// tile's world transform is in `world`; vertex format is the cache VB (kVBFVF).

sampler sampTerrainOverlay = sampler_state { texture = <tex2>; minfilter = anisotropic; magfilter = linear; mipfilter = linear; addressu = wrap; addressv = wrap; };

struct CacheTerrainVertOut {
    float4 pos : POSITION;
    float2 texcoord : TEXCOORD0;
    centroid float4 fog : TEXCOORD1;
    float4 color : COLOR0;      // .a = AlphaGrid splat factor
    float3 normal : TEXCOORD2;  // world-space, for sun N.L
    float3 viewpos : TEXCOORD3; // reflected-view space, for the below-water clip
};

CacheTerrainVertOut CacheTerrainVS(float4 pos : POSITION, float3 normal : NORMAL,
                                   float4 color : COLOR0, float2 texcoord : TEXCOORD0) {
    CacheTerrainVertOut OUT;
    float3 worldpos = mul(pos, world).xyz;

    float3 eyevec = worldpos - eyePos.xyz;
    float dist = length(eyevec);
    if(isAboveSeaLevel(eyePos))
        OUT.fog = fogColour(eyevec / dist, dist);
    else
        OUT.fog = fogMWColour(dist);

    // Position via vertexBlendPalette[0] (= world*view), the SAME premultiplied
    // matrix the shadow receiver (transformShadowVert) uses. Computing it any other
    // way (e.g. pos*world then *view) is mathematically equal but not bit-identical,
    // so the LessEqual shadow receiver would z-fight (acne) on this surface.
    OUT.pos = mul(mul(pos, vertexBlendPalette[0]), proj);
    OUT.texcoord = texcoord;
    OUT.color = color;
    OUT.normal = mul(float4(normal, 0), world).xyz;
    OUT.viewpos = mul(pos, vertexBlendPalette[0]).xyz;   // = world*reflView, for the clip
    return OUT;
}

float4 CacheTerrainPS(CacheTerrainVertOut IN) : COLOR0 {
    // Below-water clip (true water level) so near terrain doesn't bleed past the
    // lowered device clip plane into the reflection. Pass-all in the main view.
    clip(dot(float4(IN.viewpos, 1), reflWaterClipPlane));

    float3 normal = normalize(IN.normal);
    float3 base    = tex2D(sampBaseTex, IN.texcoord).rgb;
    float3 overlay = tex2D(sampTerrainOverlay, IN.texcoord).rgb;

    // AlphaGrid splat (vcol.a). When there is no overlay the caller binds the base
    // texture to tex2, so this resolves to base.
    float3 albedo = lerp(base, overlay, IN.color.a);

    // Vertex colour (vcol.rgb) is Morrowind's per-vertex terrain lighting/tint
    // (VertexColorProperty AMBIENT_DIFFUSE), modulating the albedo before sun
    // lighting — so the near cache terrain matches the above-water terrain (which is
    // vcol-lit) rather than the flat LOD land.
    float3 result = albedo * IN.color.rgb * (sunCol * saturate(dot(-sunVec, normal)) + sunAmb);
    result = fogApply(result, IN.fog);
    return float4(result, 1);
}