
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

//------------------------------------------------------------
// Tiled point-light path (USE_TILED_LIGHTS). Lights are binned into
// screen-space tiles ONCE per frame on the CPU (FixedFunctionShader::
// buildTileGrid); each pixel loops only over its own tile's light list,
// and every main-view draw shares the same grid (so neighbouring opaque
// draws are batchable — unlike the per-mesh USE_TEXTURE_LIGHTS path).
//
// Two textures (Doom-2016 two-texture scheme), both A32B32G32R32F:
//   LightGridSampler  (s8): one texel per cluster. .x = flat offset into
//                            the index list, .y = light count. .z/.w
//                            reserved for the clustered (depth-slice) add.
//   LightIndexSampler (s9): flat index list, 4 light-indices per texel
//                            (each index is a row into texLightData, the
//                            SAME wire format the per-mesh path reads).
//
// Cluster id = (sliceZ*tilesY + tileY)*tilesX + tileX, with numSlicesZ==1
// now. The grid texture is addressed directly by 2D (tileX, sliceZ*tilesY
// + tileY) coords; clustered later = set numSlicesZ>1 and add a Z term.
//
// KNOWN 2D-tiled weakness: a tile spanning near+far depth loops the union
// of every light in that screen column. The deferred clustered Z-slice
// fixes exactly this — the numSlicesZ plumbing keeps it an incremental add.
shared texture texLightGrid;
sampler LightGridSampler : register(s8) = sampler_state {
    texture   = <texLightGrid>;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = NONE;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};
shared texture texLightIndexList;
sampler LightIndexSampler : register(s9) = sampler_state {
    texture   = <texLightIndexList>;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = NONE;
    AddressU  = CLAMP;
    AddressV  = CLAMP;
};
// tileGridParams  = (tileSizePx, tilesX, tilesY, numSlicesZ)
// tileGridParams2 = (invGridTexW, invGridTexH, idxTexW, invIdxTexH)
shared float4 tileGridParams;
shared float4 tileGridParams2;

// Debug: per-object light-count heatmap. Pushed per draw from
// renderMorrowind = the selected per-mesh light count when the visualizer is on,
// -1 when off. The pixel shader overrides its output with lightCountHeatmap() so
// both reactive PPL and cache draws show identical density.
shared float debugLightCount;
float3 lightCountHeatmap(float n) {
    // Cold(0) -> hot heatmap. Scaled to 0..16, not the 32 hard cap: real per-object
    // selected counts top out ~8-12 even in dense interiors, so normalizing by the
    // cap wasted two thirds of the ramp (everything blue/green). 0 = dim blue (sun
    // only), green ~5, yellow ~8, red >=16. Tune the divisor if a denser scene clips.
    float t = saturate(n / 16.0);
    float3 c = lerp(float3(0.04, 0.05, 0.30), float3(0.0, 0.9, 0.1), saturate(t / 0.33));
    c = lerp(c, float3(1.0, 0.9, 0.0), saturate((t - 0.33) / 0.33));
    c = lerp(c, float3(1.0, 0.05, 0.0), saturate((t - 0.66) / 0.34));
    return c;
}

#if defined(USE_TEXTURE_LIGHTS) || defined(USE_TILED_LIGHTS)

// Shade one point light, given its row index `idxF` into texLightData (3
// texels: pos / diffuse / falloff+radius). Shared by both the per-mesh
// (USE_TEXTURE_LIGHTS) and the tiled (USE_TILED_LIGHTS) paths so the
// attenuation / soft-cutoff math stays identical between variants — the
// only difference between the two is how idxF is resolved (per-mesh index
// array vs the per-tile flat index list).
float3 evalOnePointLight(float idxF, float3 viewPos, float3 normal) {
    float stride = lightDataParams.y;

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
    // reaches 0; without bounding it, the CPU cull (which IS binary)
    // creates visible hard seams wherever the cull boundary lands.
    // Multiply attenuation by a smoothstep that ramps from 1 at
    // d == radius down to 0 at d == 2*radius — same 2×radius point the
    // CPU uses for cull (per-mesh sphere-AABB, or per-tile sphere-AABB
    // in buildTileGrid), so the two boundaries align and the seam
    // disappears. Pattern from OpenMW's PerObjectUniform mode.
    att *= 1.0 - smoothstep(radius, 2.0 * radius, dist);

    // Lambert (saturate(N · L) — matches FFE convention).
    // dot(N, L) = dot(N, toLight) / dist; we already have invDist,
    // so skip forming the explicit L vector — saves a vec3 divide.
    float lambert = saturate(dot(normal, toLight) * invDist);

    // Standard attenuation × Lambert × diffuse. No per-light ambient
    // term: NI fields are raw (no engine markers to interpret).
    return lambert * att * color.rgb;
}
#endif

#ifdef USE_TEXTURE_LIGHTS
float3 evaluatePointLightsTextured(float3 viewPos, float3 normal) {
    float3 acc = 0;
    int numLights = (int)lightDataParams.x;

    for (int i = 0; i < numLights; ++i) {
        // Resolve i -> texture index via the per-mesh selection list.
        // lightIndices is a float4[8] holding 32 packed indices; lane
        // selection via i/4 row, i%4 column.
        float idxF = lightIndices[i / 4][i % 4];
        acc += evalOnePointLight(idxF, viewPos, normal);
    }
    return acc;
}
#endif

#ifdef USE_TILED_LIGHTS
// Tiled per-pixel light evaluation. The pixel's screen tile resolves to a
// grid cell (offset,count) into the flat index list; loop only that tile's
// lights. `outCount` returns the looped count for the Numpad4 heatmap.
float3 evaluatePointLightsTiled(float3 viewPos, float3 normal, float2 vpos, out float outCount) {
    float3 acc = 0;

    float tileSizePx = tileGridParams.x;
    float tilesX     = tileGridParams.y;
    float tilesY     = tileGridParams.z;
    // tileGridParams.w = numSlicesZ (==1 now; sliceZ pinned to 0).

    float2 tile = floor(vpos / tileSizePx);
    tile.x = clamp(tile.x, 0.0, tilesX - 1.0);
    tile.y = clamp(tile.y, 0.0, tilesY - 1.0);

    float sliceZ   = 0.0;
    float gridRow  = sliceZ * tilesY + tile.y;
    float gridU    = (tile.x + 0.5) * tileGridParams2.x;   // invGridTexW
    float gridV    = (gridRow + 0.5) * tileGridParams2.y;  // invGridTexH
    float4 cell    = tex2Dlod(LightGridSampler, float4(gridU, gridV, 0, 0));
    float offset   = cell.x;
    float count    = cell.y;
    outCount       = count;

    float idxTexW    = tileGridParams2.z;
    float invIdxTexW = 1.0 / idxTexW;
    float invIdxTexH = tileGridParams2.w;

    int n = (int)count;
    for (int j = 0; j < n; ++j) {
        // Flat index into the index list -> texel (4 indices/texel) + lane.
        float k     = offset + (float)j;
        float texel = floor(k * 0.25);
        float lane  = k - texel * 4.0;
        float ty    = floor(texel * invIdxTexW);
        float tx    = texel - ty * idxTexW;
        float4 four = tex2Dlod(LightIndexSampler,
                               float4((tx + 0.5) * invIdxTexW, (ty + 0.5) * invIdxTexH, 0, 0));
        float idxF = (lane < 0.5) ? four.x
                   : (lane < 1.5) ? four.y
                   : (lane < 2.5) ? four.z : four.w;
        acc += evalOnePointLight(idxF, viewPos, normal);
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

// Per-pixel lighting augmented with semi-HDR tonemap instead of light clamping.
// vpos : VPOS is the screen-space pixel coordinate, used by the tiled light path
// to resolve the pixel's screen tile (same VPOS pattern as XE Mod Sky.fx).
float4 PerPixelPS(FFEPixel IN, float2 vpos : VPOS) : COLOR0 {
    // Below-water clip for the cache reflection passes (true water level). Pass-all
    // (0,0,0,1) in the main reactive scene, so this is a no-op there.
    clip(dot(float4(IN.viewpos, 1), reflWaterClipPlane));

    float3 normal = normalize(IN.nrm_fog.xyz);
    float fog = IN.nrm_fog.w;

    // Standard morrowind lighting: sun, ambient, and point lights
    float3 d = lightSunDiffuse * saturate(dot(normal, -lightSunDirection));
    float3 a = lightSceneAmbient;

#if defined(USE_TILED_LIGHTS)
    // Tiled point-light path. Lights binned into screen tiles once per
    // frame (FixedFunctionShader::buildTileGrid); this pixel loops only
    // its own tile's list. tileLightCount is the looped count, fed to the
    // Numpad4 heatmap below.
    float tileLightCount = 0;
    d += pointLightMult * evaluatePointLightsTiled(IN.viewpos, normal, vpos, tileLightCount);
#elif defined(USE_TEXTURE_LIGHTS)
    // Per-mesh texture-light path. Lights packed into a 1D texture (3
    // texels per light); shader iterates lightDataParams.x lights
    // (runtime count, no compile-time array). See
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

    // Debug: override with the light-count heatmap (sun-only fog kept so shape
    // reads). debugLightCount < 0 = visualizer off (the common case); >= 0 = on.
    // Under tiled, the count is the in-shader per-TILE loop count (not the CPU
    // per-mesh candidateCount, which the tiled path never computes). Tile counts
    // run higher than per-object counts, so feed the ramp a retuned divisor (48
    // via the 16/48 scale) vs the per-mesh /16 in lightCountHeatmap.
    [branch] if (debugLightCount >= 0) {
#ifdef USE_TILED_LIGHTS
        c.rgb = lightCountHeatmap(tileLightCount * (16.0 / 48.0));
#else
        c.rgb = lightCountHeatmap(debugLightCount);
#endif
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
