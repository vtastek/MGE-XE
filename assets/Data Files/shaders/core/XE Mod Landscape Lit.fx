
// XE Mod Landscape Lit.fx
//
// Main-view cache near-terrain WITH dynamic point lights. Split out of
// "XE Mod Landscape.fx" so it is compiled ONLY into the distant-land effect
// ("XE Main.fx") and NOT into "XE Depth.fx", which also includes
// "XE Mod Landscape.fx". The texture-light path here declares a hard
// register(s7) sampler plus extra uniform globals; when those were compiled
// into the depth effect they perturbed its register layout and corrupted the
// cache depth pass (exploded geometry / empty interiors). Depth never uses this
// lit-terrain technique, so it lives here, included by Main.fx only.
//
// Requires (include AFTER "XE Common.fx" and "XE Mod Landscape.fx"):
//   sampBaseTex (Common), sampTerrainOverlay (Mod Landscape), world/proj/
//   vertexBlendPalette/eyePos/sunCol/sunVec/sunAmb (Common), fog helpers (Common).

//------------------------------------------------------------
// Cache near terrain WITH dynamic point lights (main view). Same two-texture
// AlphaGrid splat + sun + vcol as CacheTerrainPS, plus the texture-backed point
// light path so cache terrain matches the reactive PPL terrain lighting (the A/B
// light-seam fix). Selection is byte-identical to the object path: the CPU side
// (renderCachedTerrain) fills lightIndices/lightDataParams/texLightView via the
// SHARED FixedFunctionShader::selectTextureLights and binds the same texLightData
// texture. A separate VS/PS so the reflection CacheTerrainVS/PS stay untouched.
// Mirrors evaluatePointLightsTextured in "XE FixedFuncEmu.fx".

shared texture texLightData;
sampler CacheLightDataSampler : register(s7) = sampler_state {
    texture = <texLightData>;
    MinFilter = POINT; MagFilter = POINT; MipFilter = NONE;
    AddressU = CLAMP; AddressV = CLAMP;
};
float4 lightDataParams;     // .x = runtime light count, .y = 1/textureWidth (texel U stride)
float4 lightIndices[8];     // 32 packed per-patch light indices
matrix texLightView;        // world->view, to transform world-space light positions

// Identical compression curve to FFE's tonemap (XE FixedFuncEmu.fx): maps
// 0 -> 0, 1.0 -> 0.84, up to 2.2 -> 1.0. Duplicated here because the distant-land
// effect (XE Main.fx) and the FFE object effect are separate compilation units.
// The cache object path (renderMorrowind -> PerPixelPS) tonemaps every pixel, so
// cache terrain must apply the SAME curve to match brightness/contrast (A/B parity).
float3 cacheTerrainTonemap(float3 c) {
    c = clamp(c, 0, 2.2);
    c = (((0.0548303 * c - 0.189786) * c - 0.154732) * c + 1.12969) * c;
    return c;
}

float3 cacheTerrainPointLights(float3 viewPos, float3 normal) {
    float3 acc = 0;
    int numLights = (int)lightDataParams.x;
    float stride  = lightDataParams.y;
    for (int i = 0; i < numLights; ++i) {
        float idxF = lightIndices[i / 4][i % 4];
        float u0 = (idxF * 3.0 + 0.5) * stride;
        float u1 = u0 + stride;
        float u2 = u0 + stride * 2.0;
        float4 lpos    = tex2Dlod(CacheLightDataSampler, float4(u0, 0.5, 0, 0));
        float4 lcolor  = tex2Dlod(CacheLightDataSampler, float4(u1, 0.5, 0, 0));
        float4 falloff = tex2Dlod(CacheLightDataSampler, float4(u2, 0.5, 0, 0));
        float radius   = falloff.w;
        float3 lightViewPos = mul(float4(lpos.xyz, 1.0), texLightView).xyz;
        float3 toLight = lightViewPos - viewPos;
        float dist2    = dot(toLight, toLight);
        float invDist  = rsqrt(dist2);
        float dist     = dist2 * invDist;
        // Full attenuation incl. linear k1 (falloff.y) — Morrowind candle/torch
        // lights are pure-linear (k0=k2=0); dropping k1 blows them to white. Matches
        // evaluatePointLightsTextured in "XE FixedFuncEmu.fx".
        float att = 1.0 / max(falloff.z * dist2 + falloff.y * dist + falloff.x, 1e-4);
        att *= 1.0 - smoothstep(radius, 2.0 * radius, dist);
        float lambert = saturate(dot(normal, toLight) * invDist);
        acc += lambert * att * lcolor.rgb;
    }
    return acc;
}

struct CacheTerrainLitVertOut {
    float4 pos : POSITION;
    float2 texcoord : TEXCOORD0;
    centroid float4 fog : TEXCOORD1;
    float4 color : COLOR0;          // .a = AlphaGrid splat factor, .rgb = vcol tint
    float3 normal : TEXCOORD2;      // world-space, for sun N.L
    float3 viewpos : TEXCOORD3;     // view-space (world*view), for point lights
    float3 viewnormal : TEXCOORD4;  // view-space, for point lights
};

CacheTerrainLitVertOut CacheTerrainLitVS(float4 pos : POSITION, float3 normal : NORMAL,
                                         float4 color : COLOR0, float2 texcoord : TEXCOORD0) {
    CacheTerrainLitVertOut OUT;
    float3 worldpos = mul(pos, world).xyz;

    float3 eyevec = worldpos - eyePos.xyz;
    float dist = length(eyevec);
    if(isAboveSeaLevel(eyePos))
        OUT.fog = fogColour(eyevec / dist, dist);
    else
        OUT.fog = fogMWColour(dist);

    // Position via vertexBlendPalette[0] (= world*view), same as CacheTerrainVS.
    OUT.pos = mul(mul(pos, vertexBlendPalette[0]), proj);
    OUT.texcoord = texcoord;
    OUT.color = color;
    OUT.normal = mul(float4(normal, 0), world).xyz;
    OUT.viewpos = mul(pos, vertexBlendPalette[0]).xyz;                  // world*view = view space
    OUT.viewnormal = mul(float4(normal, 0), vertexBlendPalette[0]).xyz; // view-space normal
    return OUT;
}

// Fold-free lit color (albedo + sun + dynamic point lights + tonemap + fog). Both
// the main-view (CacheTerrainLitPS) and reflection (CacheTerrainReflLitPS) shaders
// build on this; the sun shadow fold below is layered on top so the two paths share
// one base and one fold, differing only in the matrices/sun the CPU binds per pass.
float4 cacheTerrainLitColor(CacheTerrainLitVertOut IN) {
    float3 normal  = normalize(IN.normal);
    float3 base    = tex2D(sampBaseTex, IN.texcoord).rgb;
    float3 overlay = tex2D(sampTerrainOverlay, IN.texcoord).rgb;
    float3 albedo  = lerp(base, overlay, IN.color.a);

    // Sun (as CacheTerrainPS) + dynamic point lights (matching the reactive PPL terrain).
    float3 sun = sunCol * saturate(dot(-sunVec, normal)) + sunAmb;
    float3 pts = cacheTerrainPointLights(IN.viewpos, normalize(IN.viewnormal));
    float3 result = albedo * IN.color.rgb * (sun + pts);
    result = cacheTerrainTonemap(result);   // match the FFE object path before fog
    result = fogApply(result, IN.fog);
    return float4(result, 1);
}

// Sun shadow fold, mirroring the FFE object fold (applyCacheShadow in
// "XE FixedFuncEmu.fx") so near terrain darkens like the cache objects do. Replaces
// the standalone terrain receiver re-draw (renderShadowReceiverFromCache in the main
// view, renderReflectionShadowsFromCache in the reflection, both removed). Uses the
// receiver machinery from "XE Mod Shadow.fx" directly (sampDepth/tex3 holds the shadow
// atlas during these passes; unlike FFE there is no tex3 conflict, terrain uses
// tex0/tex2 only). shadowViewProj = view -> shadow clip and sunVecView = the
// same-space sun, bound by the CPU caller (main: renderCachedTerrain; reflection:
// renderReflectionTerrainFromCache). Applied after tonemap + fog: the standalone
// receiver darkened the FINAL color (SrcBlend=Zero / DestBlend=InvSrcColor ==
// c.rgb * (1 - v*shadecolor)). No vcol.a: the splat factor must not modulate shadow
// strength. shadowReflMult gates AND fades: 0 disables the fold entirely (shadows-off,
// or the DL LOD handover beyond the band), 1 = full strength.
float4 cacheTerrainSunShadow(float4 c, float3 viewpos, float3 viewnormal) {
    float4 shadow0pos = mul(float4(viewpos, 1), shadowViewProj[0]);
    float4 shadow1pos = mul(float4(viewpos, 1), shadowViewProj[1]);
    shadow0pos.z /= shadow0pos.w;
    shadow1pos.z /= shadow1pos.w;

    float lightT = shadowSunEstimate(saturate(dot(normalize(viewnormal), -sunVecView)));
    float fogatt = pow(fogMWScalar(length(viewpos)), 2);
    lightT *= isAboveSeaLevel(eyePos) ? fogatt : saturate(4 * fogatt);
    lightT *= shadowReflMult;   // gate/fade (0 = no fold)

    // Shadowed fragments have NEGATIVE dz; shadowESM is exactly 0 for dz >= 0
    // (the no-caster case), so no guard is needed.
    float dz = shadowDeltaZ(shadow0pos, shadow1pos);
    float v = shadowESM(dz) * lightT;

    // Fade out shadows at map edges
    float2 fade = saturate(25 * (1 - abs(shadow1pos.xy)));
    v *= fade.x * fade.y;

    c.rgb *= 1 - v * shadecolor;
    return c;
}

// Main-view cache near terrain: lit color + the sun shadow fold. The fold reads the
// main-view shadowViewProj / sunVecView and shadowReflMult bound by renderCachedTerrain
// (1 when shadows are enabled, 0 to disable). Replaces the separate main-view receiver
// re-draw (renderShadowReceiverFromCache).
float4 CacheTerrainLitPS(CacheTerrainLitVertOut IN) : COLOR0 {
    return cacheTerrainSunShadow(cacheTerrainLitColor(IN), IN.viewpos, IN.viewnormal);
}

//------------------------------------------------------------
// Reflection variant: the main-view lit shading (sun + vcol + dynamic point lights)
// PLUS the reflection's below-water clip, so reflected near terrain gets the same
// point lights as reflected objects (the reflection terrain previously used the
// non-lit CacheTerrainPS, leaving it unlit by candles/torches). reflWaterClipPlane
// comes from "XE Mod Landscape.fx" (included before this file by XE Main.fx); it is
// the true water level in the reflection (pass-all in the main view). CacheTerrainLitVS
// already outputs viewpos (= world*reflView) for the clip dot product. Same fold as
// the main view, but renderReflectionTerrainFromCache binds the reflected-view
// matrices and the per-patch handover fade in shadowReflMult.
float4 CacheTerrainReflLitPS(CacheTerrainLitVertOut IN) : COLOR0 {
    clip(dot(float4(IN.viewpos, 1), reflWaterClipPlane));
    return cacheTerrainSunShadow(cacheTerrainLitColor(IN), IN.viewpos, IN.viewnormal);
}
