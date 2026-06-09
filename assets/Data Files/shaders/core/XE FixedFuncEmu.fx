
// XE FixedFuncEmu.fx
// MGE XE 0.16.0
// Replacement shaders for Morrowind's object rendering

#include "XE Common.fx"
// Shadow constants (shade/shadecolor/ESM_*) for the reflection-cache shadow fold
// below. Constants only — the receiver VS/structs/samplers stay in XE Mod Shadow.fx.
#include "XE Mod Shadow Data.fx"

shared texture tex4, tex5;
shared matrix worldview;
shared float4 materialDiffuse, materialAmbient, materialEmissive;
shared float3 lightSceneAmbient, lightSunDiffuse, lightDiffuse[8];
shared float4 lightAmbient[2];
shared float3 lightSunDirection;
// Per-draw point-light intensity scale. 1 = normal. The cache reflection pass
// fades it toward 0 as a static approaches the cache->distant-land handover, so
// point lights don't pop against the sun-only far field. renderMorrowind sets it
// every draw (1 in the main reactive path), so it never leaks between draws.
shared float pointLightMult;
shared float4 lightPosition[6];
shared float4 lightFalloffQuadratic[2], lightFalloffLinear[2];
shared float4 lightFalloffConstant[2];
shared matrix texgenTransform;
shared float4 bumpMatrix;
shared float2 bumpLumiScaleBias;

sampler sampFFE0 = sampler_state { texture = <tex0>; };
sampler sampFFE1 = sampler_state { texture = <tex1>; };
sampler sampFFE2 = sampler_state { texture = <tex2>; };
sampler sampFFE3 = sampler_state { texture = <tex3>; };
sampler sampFFE4 = sampler_state { texture = <tex4>; };
sampler sampFFE5 = sampler_state { texture = <tex5>; };

//------------------------------------------------------------
// Reflection-cache sun shadow fold. The cache reflection color pass computes the
// shadow receiver term inline (gated by applyCacheShadow), replacing the separate
// darkening re-draw (PASS_RENDERSHADOWFFE) that submitted the same geometry twice.
//
// The shadow atlas CANNOT ride the receiver's sampDepth (tex3): FFE's sampFFE3
// also reads tex3 for object texture stage 3. Dedicated shared texture + sampler
// pinned at s6 instead — provably free (FFE uses s0-s5 for stages, s7 for
// texLightData), same pattern as LightDataSampler below.
//
// The ESM atlas lookup is duplicated from XE Mod Shadow.fx (ffe- prefix, bound to
// s6) rather than #including it (that would pull in sampDepth<-tex3 and the
// receiver VS/structs). KEEP IN SYNC with shadowDeltaZ / shadowESM /
// mapShadowToAtlas / shadowSunEstimate there.
shared texture texShadowAtlas;
sampler sampFFEShadow : register(s6) = sampler_state {
    texture = <texShadowAtlas>;
    minfilter = linear; magfilter = linear; mipfilter = none;
    addressu = clamp; addressv = clamp;
};
// Default false => the main reactive scene never takes the branch and is
// byte-identical. Set true only around the cache reflection color loop.
shared bool applyCacheShadow;

// Clip space margin of 4 texels (see XE Mod Shadow.fx atlasMargin).
static float3 ffeAtlasMargin = float3(1 - 2*4*shadowRcpRes, 1 - 2*4*shadowRcpRes, 1);

// Shadow UV to shadow atlas UV, for tex2Dlod.
float4 ffeMapShadowToAtlas(float2 t, int layer) {
    return float4(t.x * shadowCascadeSize + layer * shadowCascadeSize, t.y, 0, 0);
}

// Incoming vertex sunlight estimation (non-standard shadow luminance, for
// contrast when ambient is high).
float ffeShadowSunEstimate(float lambert) {
    float x = lambert * dot(sunCol, float3(0.36, 0.53, 0.11));
    x *= 0.25 + 0.75 * sunVis;
    return x / (shade + x);
}

// 2 layer cascade ortho ESM lookup with the cascade-boundary blend band
// (edge-flicker fix) — same structure as XE Mod Shadow.fx::shadowDeltaZ.
float ffeShadowDeltaZ(float4 shadow0pos, float4 shadow1pos) {
    float dz = 1e-6;

    bool inC0 = all(saturate(ffeAtlasMargin - abs(shadow0pos.xyz)));
    bool inC1 = all(saturate(ffeAtlasMargin - abs(shadow1pos.xyz)));

    [branch] if(inC0) {
        float2 c0UV = (0.5 + 0.5*shadowRcpRes) + float2(0.5, -0.5) * shadow0pos.xy;
        float  dz0  = tex2Dlod(sampFFEShadow, ffeMapShadowToAtlas(c0UV, 0)).r / ESM_scale - shadow0pos.z;

        float2      c0Out      = abs(shadow0pos.xy) / ffeAtlasMargin.xy;
        float       c0OutMax   = max(c0Out.x, c0Out.y);
        const float blendStart = 0.8;

        [branch] if(c0OutMax > blendStart && inC1) {
            float2 c1UV = (0.5 + 0.5*shadowRcpRes) + float2(0.5, -0.5) * shadow1pos.xy;
            float  dz1  = tex2Dlod(sampFFEShadow, ffeMapShadowToAtlas(c1UV, 1)).r / ESM_scale - shadow1pos.z;
            float  t    = smoothstep(blendStart, 1.0, c0OutMax);
            dz = lerp(dz0, dz1, t);
        }
        else {
            dz = dz0;
        }
    }
    else if(inC1) {
        float2 c1UV = (0.5 + 0.5*shadowRcpRes) + float2(0.5, -0.5) * shadow1pos.xy;
        dz = tex2Dlod(sampFFEShadow, ffeMapShadowToAtlas(c1UV, 1)).r / ESM_scale - shadow1pos.z;
    }

    return dz;
}

float ffeShadowESM(float dz) {
    return 1 - saturate(exp(ESM_c * dz + ESM_bias));
}

//------------------------------------------------------------

#ifdef VERIFY
#define FFE_VB_COUPLING float2 texcoord0 : TEXCOORD0; float4 col : COLOR;
#define FFE_SHADER_COUPLING float4 texcoord01 : TEXCOORD0; float4 col : COLOR;
#define FFE_TRANSFORM_SKIN viewpos = rigidVertex(IN.pos); normal = rigidNormal(IN.nrm);
#define FFE_TEXCOORDS_TEXGEN float3 texgen = texgenReflection(viewpos, normal); texgen = mul(float4(texgen, 1), texgenTransform).xyz; OUT.texcoord01 = float4(IN.texcoord0, texgen.xy);
#define FFE_VERTEX_COLOUR OUT.col = IN.col;
#define FFE_LIGHTS_ACTIVE 8
#define FFE_VERTEX_MATERIAL diffuse = vertexMaterialDiffAmb(d, a, IN.col);
#define FFE_TEXTURING c = diffuse + bumpmapLumiStage(sampFFE1, IN.texcoord01.zw, tex2D(sampFFE0, IN.texcoord01.xy));
#define FFE_FOG_APPLICATION c.rgb = lerp(fogColNear, c.rgb, fog);
#endif

#ifdef FFE_ERROR_MATERIAL
#define FFE_VB_COUPLING
#define FFE_SHADER_COUPLING
#define FFE_TRANSFORM_SKIN viewpos = rigidVertex(IN.pos); normal = float3(0, 0, 1);
#define FFE_TEXCOORDS_TEXGEN
#define FFE_VERTEX_COLOUR
#define FFE_LIGHTS_ACTIVE 0
#define FFE_VERTEX_MATERIAL diffuse = float4(1, 0, 0.5, 1);
#define FFE_TEXTURING
#define FFE_FOG_APPLICATION c.rgb = lerp(fogColNear, c.rgb, fog);
#endif


//------------------------------------------------------------
// Transform library functions

// Vertex transform to view space, note that scaled normals are normalized in the pixel shader
float4 rigidVertex(float4 pos) { return mul(pos, worldview); }
float3 rigidNormal(float3 normal) { return mul(float4(normal, 0), worldview).xyz; }

float4 skinnedVertex(float4 pos, float4 weights) { return skin(pos, weights); }
float3 skinnedNormal(float3 normal, float4 weights) { return skin(float4(normal, 0), weights).xyz; }

// Cache (32-bone indexed) skinning, used by the cache-driven color pass. boneMatrices
// are model->world (the rigid path bakes world*view into 'worldview', which has no
// per-bone equivalent here), so skin to world then apply the view matrix.
float4 cacheSkinnedVertex(float4 pos, float4 weights, float4 indices) {
    return mul(skinIndexed(pos, weights, indices), view);
}
float3 cacheSkinnedNormal(float3 normal, float4 weights, float4 indices) {
    float3 worldn = skinIndexed(float4(normal, 0), weights, indices).xyz;
    return mul(float4(worldn, 0), view).xyz;
}

// Texgens with view space inputs, normals must be normalized due to non-uniform scaling matrices
float3 texgenNormal(float3 normal) { return normalize(normal); }
float3 texgenPosition(float4 pos) { return pos.xyz; }
float3 texgenReflection(float4 pos, float3 normal) { return reflect(normalize(pos.xyz), normalize(normal)); }
float3 texgenSphere(float2 tex) { return float3(0.5 * tex + 0.5, 0); }

//------------------------------------------------------------
// Lighting library functions

// Number of light groups; lights are vectorized into groups of 4
static const int LGs = max(1, ceil(FFE_LIGHTS_ACTIVE / 4.0));

//------------------------------------------------------------
// Texture-based point light path. Lights packed into a dynamic 1D
// texture (3 texels per light), iterated by runtime count in
// lightDataParams.x. Format mirrors vtastek's core-hlsl/lighting.hlsl
// evaluatePointLights — same general layout, parameters register, and
// sampler slot — so we can later converge with that path if it lands
// upstream.
//
// Wire format per light (3 texels, R32G32B32A32F):
//   texel 0: (posX, posY, posZ, _)        — WORLD-space position (.w reserved)
//   texel 1: (diffR, diffG, diffB, _)     — raw NI diffuse × dimmer
//   texel 2: (k0, k1, k2, radius)         — falloff const/lin/quad + radius
//
// Texel 0 is in WORLD space — the shader transforms it to view-space
// per pixel using the `texLightView` uniform (one mat-vec per light).
// Storing world-space (rather than view-space) means the texture only
// needs re-upload when the light snapshot changes; camera rotation no
// longer triggers an upload. Trade is a small per-light per-pixel
// matrix-vector multiply (~12 ALU ops/light) on GPU.
//
// Texel-2.w carries Bethesda's modder-set radius (NI::Light::specular.r).
// The shader uses it as the inner edge of a smoothstep window that
// drives attenuation to exactly 0 at 2×radius, masking the seam from
// the CPU's per-mesh sphere-AABB cull (also at 2×radius).
//
// lightDataParams (constant register c50):
//   .x = number of lights to iterate (runtime)
//   .y = 1.0 / textureWidth        (texel U coordinate stride)
//   .z = first-light texel offset  (always 0 in our scene-wide setup)
//   .w = unused
// Sampler slot diverges from vtastek's s5: the FFE shader already
// declares sampFFE5 at s5 (mapped to engine's tex5), so we can't reuse
// that slot. s7 is unused by FFE and safely above MGE's other shader
// paths.
shared texture texLightData;
sampler LightDataSampler : register(s7) = sampler_state {
    texture   = <texLightData>;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = NONE;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};
shared float4 lightDataParams;
// Per-mesh light indices (selected via sphere-AABB on CPU, ranked by
// dist²/radius²). Each float4 packs 4 indices. 8 float4s give us
// kMaxIndicesPerMesh=32 indices per mesh. lightDataParams.x is the
// runtime-bounded loop count over these indices.
shared float4 lightIndices[8];
// View matrix used to transform world-space light positions (from
// texLightData texel 0) into view-space for the lighting math. Pushed
// per-draw from ffeshader.cpp::renderMorrowind alongside the other
// texture-light uniforms.
shared matrix texLightView;
#ifdef USE_TEXTURE_LIGHTS

float3 evaluatePointLightsTextured(float3 viewPos, float3 normal) {
    float3 acc = 0;
    int numLights = (int)lightDataParams.x;
    float stride  = lightDataParams.y;

    for (int i = 0; i < numLights; ++i) {
        // Resolve i -> texture index via the per-mesh selection list.
        // lightIndices is a float4[8] holding 32 packed indices; lane
        // selection via i/4 row, i%4 column.
        float idxF = lightIndices[i / 4][i % 4];

        // Strength-reduced texel U coordinates: compute u0 once, derive
        // u1/u2 by adding stride. Saves two multiplies per light vs.
        // recomputing (idx*3 + offset + 0.5) * stride for each texel.
        float u0 = (idxF * 3.0 + 0.5) * stride;
        float u1 = u0 + stride;
        float u2 = u0 + stride * 2.0;

        float4 pos     = tex2Dlod(LightDataSampler, float4(u0, 0.5, 0, 0));
        float4 color   = tex2Dlod(LightDataSampler, float4(u1, 0.5, 0, 0));
        float4 falloff = tex2Dlod(LightDataSampler, float4(u2, 0.5, 0, 0));
        float radius   = falloff.w;

        // Transform world-space light position into view-space using the
        // per-draw `texLightView` matrix. This is the "world-space-in-
        // texture" half of the camera-rotation upload optimisation — the
        // texture stays revision-keyed; camera motion only triggers this
        // ~12-ALU mat-vec per light per pixel.
        float3 lightViewPos = mul(float4(pos.xyz, 1.0), texLightView).xyz;
        float3 toLight = lightViewPos - viewPos;
        float dist2    = dot(toLight, toLight);
        // rsqrt + dist2 * invDist beats sqrt + 1/sqrt on modern HW.
        float invDist  = rsqrt(dist2);
        float dist     = dist2 * invDist;

        // Full attenuation 1/(k0 + k1*d + k2*d2). The linear term k1 (falloff.y)
        // is essential: Morrowind candle/torch lights are pure-linear (k0=k2=0),
        // so dropping k1 collapses the denominator to the 1e-4 guard and blows the
        // light to white inside its radius. `dist` is already computed above.
        // max() guards the singular d->0 flame-center texel (not a fallback path).
        float att = 1.0 / max(falloff.z * dist2 + falloff.y * dist + falloff.x, 1e-4);

        // Soft-cutoff window. The engine's 1/(C + L*d + Q*d²) never
        // reaches 0; without bounding it, the per-mesh CPU cull (which
        // IS binary) creates visible hard seams wherever a mesh sits
        // on the cull boundary. Multiply attenuation by a smoothstep
        // that ramps from 1 at d == radius down to 0 at d == 2*radius —
        // same 2×radius point the CPU uses for cull, so the two
        // boundaries align and the seam disappears. Pattern from
        // OpenMW's PerObjectUniform mode (lighting_util.glsl).
        att *= 1.0 - smoothstep(radius, 2.0 * radius, dist);

        // Lambert (saturate(N · L) — matches FFE convention).
        // dot(N, L) = dot(N, toLight) / dist; we already have invDist,
        // so skip forming the explicit L vector — saves a vec3 divide.
        float lambert = saturate(dot(normal, toLight) * invDist);

        // Standard 1/(k0 + k2·d²) attenuation × Lambert × diffuse.
        // No per-light ambient term: NI fields are raw (no engine
        // markers to interpret), so the constant-array path's
        // bufferAmbient correction does not apply here.
        acc += lambert * att * color.rgb;
    }
    return acc;
}
#endif

// Point lights
float4 calcLighting4(float4 lightvec[3*LGs], int group, float3 normal) {
    float4 dist2 = 0, lambert = 0;

    // Do four dot products as three mads
    for(int i = 0; i != 3; ++i)
        dist2 += pow(lightvec[3*group + i], 2);

    // Same for N.L
    for(int i = 0; i != 3; ++i)
        lambert += normal[i] * lightvec[3*group + i];

    // Normalize L after the fact
    float4 dist = sqrt(dist2);
    lambert = saturate(lambert / dist);

    // Attenuation
    float4 att = 1.0 / (lightFalloffQuadratic[group] * dist2 + lightFalloffConstant[group]);
    // (slower) float4 att = 1.0 / (lightFalloffQuadratic[group] * dist2 + lightFalloffLinear[group] * dist + lightFalloffConstant[group]);
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

// Static tonemap
float3 tonemap(float3 c) {
    // Curve maps 0 -> 0, 1.0 -> 0.84, up to 2.2 -> 1.0
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

// Bumpmap stages return dUdV alpha channel due to select1 alpha op
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
// Data coupling framework

struct FFEVertIn {
    float4 pos : POSITION;
    float3 nrm : NORMAL;

    /* template */ FFE_VB_COUPLING
};

struct FFEPixel {
    float4 pos : POSITION;
    centroid float4 nrm_fog : NORMAL;

    /* template */ FFE_SHADER_COUPLING

    float3 viewpos : TEXCOORD2;
};

//------------------------------------------------------------
// Shader framework

// Passes view-space position; per-light vectors are reconstructed in PS
FFEPixel PerPixelVS(FFEVertIn IN) {
    FFEPixel OUT;

    // Transforms
    float4 viewpos;
    float3 normal;
    /* template */ FFE_TRANSFORM_SKIN

    float dist = length(viewpos);
    OUT.pos = mul(viewpos, proj);
    OUT.nrm_fog = float4(normal, fogMWScalar(dist));

    // Texcoord routing and texgen
    /* template */ FFE_TEXCOORDS_TEXGEN

    // Vertex colour
    /* template */ FFE_VERTEX_COLOUR

    // Pass view-space position for per-pixel light-vector reconstruction
    OUT.viewpos = viewpos.xyz;

    return OUT;
}

// Per-pixel lighting augmented with semi-HDR tonemap instead of light clamping
float4 PerPixelPS(FFEPixel IN) : COLOR0 {
    // Below-water clip for the cache reflection passes (true water level). Pass-all
    // (0,0,0,1) in the main reactive scene, so this is a no-op there.
    clip(dot(float4(IN.viewpos, 1), reflWaterClipPlane));

    float3 normal = normalize(IN.nrm_fog.xyz);
    float fog = IN.nrm_fog.w;

    // Standard morrowind lighting: sun, ambient, and point lights
    float3 d = lightSunDiffuse * saturate(dot(normal, -lightSunDirection));
    float3 a = lightSceneAmbient;

#ifdef USE_TEXTURE_LIGHTS
    // _Claude_ Phase 2 texture-light path. Lights packed into a 1D
    // texture (3 texels per light); shader iterates lightDataParams.x
    // lights (runtime count, no compile-time array). See
    // evaluatePointLightsTextured below.
    d += pointLightMult * evaluatePointLightsTextured(IN.viewpos, normal);
#else
    // Reconstruct per-light L vectors from view-space position. Was per-vertex
    // via interpolators; moved here to lift the interpolator-budget cap on light count.
    float4 lightvec[3*LGs];
    for (int i = 0; i != LGs; ++i) {
        lightvec[3*i + 0] = lightPosition[i + 0] - IN.viewpos.x;
        lightvec[3*i + 1] = lightPosition[i + 2] - IN.viewpos.y;
        lightvec[3*i + 2] = lightPosition[i + 4] - IN.viewpos.z;
    }
    d += pointLightMult * calcPointLighting(FFE_LIGHTS_ACTIVE, lightvec, normal);
#endif

    // Material
    float4 diffuse;
    /* template */ FFE_VERTEX_MATERIAL

    // Texturing and combinators
    float4 c = diffuse;
    /* template */ FFE_TEXTURING

    // Static tonemap and final fogging
    c.rgb = tonemap(c.rgb);
    /* template */ FFE_FOG_APPLICATION

    // Reflection-cache sun shadow fold. Applied after tonemap + fog because the
    // standalone receiver pass it replaces darkened the FINAL framebuffer color
    // (SrcBlend=Zero / DestBlend=InvSrcColor == c.rgb * (1 - v*shadecolor)).
    // No clip(): v = 0 makes the multiply a no-op on non-shadowed fragments.
    // shadowViewProj is bound as reflected-view -> shadow clip, and sunVecView as
    // the reflected-view sun, by the cache reflection pass before its draw loop.
    [branch] if (applyCacheShadow) {
        float4 shadow0pos = mul(float4(IN.viewpos, 1), shadowViewProj[0]);
        float4 shadow1pos = mul(float4(IN.viewpos, 1), shadowViewProj[1]);
        shadow0pos.z /= shadow0pos.w;
        shadow1pos.z /= shadow1pos.w;

        // Surface lit-ness estimate + fog attenuation (shadow darkness and
        // distance fade), as shadowReceiverBody computes per vertex.
        float lightT = ffeShadowSunEstimate(saturate(dot(normal, -sunVecView)));
        float fogatt = pow(fogMWScalar(length(IN.viewpos)), 2);
        lightT *= isAboveSeaLevel(eyePos) ? fogatt : saturate(4 * fogatt);
        // Per-object cache->distant-land handover fade (1 in the main view).
        lightT *= shadowReflMult;

        // Shadowed fragments have NEGATIVE dz (dz = casterDepth - fragmentDepth;
        // the standalone receiver keeps them via clip(-dz)). No guard needed:
        // ffeShadowESM saturates to exactly 0 for dz >= 0 (the no-caster case).
        float dz = ffeShadowDeltaZ(shadow0pos, shadow1pos);
        float v = ffeShadowESM(dz) * lightT;

        // Fade out shadows at map edges
        float2 fade = saturate(25 * (1 - abs(shadow1pos.xy)));
        v *= fade.x * fade.y;

        c.rgb *= 1 - v * shadecolor;
    }

    return c;
}

//-----------------------------------------------------------------------------

technique FFE {
    pass {
        VertexShader = compile vs_3_0 PerPixelVS();
        PixelShader = compile ps_3_0 PerPixelPS();
    }
}
