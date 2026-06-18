
// XE Mod Water.fx
// MGE XE 0.16.0
// Water plane rendering. Can be used as a core mod.

// DEBUG (vtastek): when 1, WaterPS early-returns the raw reflection RT sampled at
// the flat screen position — undistorted (no ripple reffactor) and without the
// water fog/fresnel/specular mix — so the reflection contents (e.g. cache terrain)
// can be inspected directly. Set to 0 to restore normal water shading.
#define DEBUG_REFLECTION_RAW 0


//------------------------------------------------------------
// Samplers, clamping mode

sampler sampReflect = sampler_state { texture = <tex0>; minfilter = linear; magfilter = linear; mipfilter = none; addressu = clamp; addressv = clamp; };
sampler sampRefract = sampler_state { texture = <tex2>; minfilter = linear; magfilter = linear; mipfilter = none; addressu = clamp; addressv = clamp; };

//------------------------------------------------------------
// Water constants

static const float sunlightFactor = 1 - pow(1 - sunVis, 2);
static const float3 sunColAdjusted = sunCol * sunlightFactor;
static const float3 depthBaseColor = sunColAdjusted * float3(0.03, 0.04, 0.05) + (2 * skyCol + fogColFar) * float3(0.075, 0.08, 0.085);
static const float windFactor = (length(windVec) + 1.5) / 140;
static const float waterLevel = world[3][2];
static const float shoreDepthBias = 24.0;

shared texture tex4, tex5;
shared float3 rippleOrigin;
shared float waveHeight;

sampler sampRain = sampler_state { texture = <tex4>; minfilter = linear; magfilter = linear; mipfilter = linear; addressu = wrap; addressv = wrap; };
sampler sampWave = sampler_state { texture = <tex5>; minfilter = linear; magfilter = linear; mipfilter = linear; bordercolor = 0; addressu = border; addressv = border; };

#ifdef WATER_FLOW_MAP
// Per-water-body flow field, baked CPU-side (see buildWaterFlowMap). R,G = downstream
// flow direction (encoded *0.5+0.5), B = wave intensity, A = packed routing
// (0.5 neutral, <0.5 near/river+beach strength, >0.5 far/sea-refraction strength).
shared texture texFlow;
shared float4 flowMapTransform;     // origin.xy, invSize.xy  (world XY → [0,1] map UV)
shared float  flowMapWeight;        // debug A/B: 1 = flow map on, 0 = neutral
sampler sampFlow = sampler_state { texture = <texFlow>; minfilter = linear; magfilter = linear; mipfilter = none; addressu = clamp; addressv = clamp; };
shared float flowScrollSpeed;        // river directional advection rate (live NUMPAD8/6/3 tuning)
shared float flowSeaSpeed;           // base wave animation rate scale (1 = stock; live tuning)
shared float flowCycleUV;            // bounded per-cycle UV displacement (Valve flow map; live tuning)
shared float flowSeaRefract;         // sea far-wave (refraction) strength multiplier (live tuning)
shared float flowDebugView;          // CLASSIFY view: 1 = paint water flat by category colour
shared float flowMapWarp;            // domain-warp amount (world u) to break the 512u texel grid
// Domain warp: jitter world XY before mapping to flow UV, so the axis-aligned 512u texel grid
// reads as organic wiggles instead of hard squares. Self-contained 2-octave sin (the foam fbm
// lives under WATER_FOAM and isn't always available in this block).
float2 flowWarpUV(float2 worldXY) {
    float2 p = worldXY * (1.0 / 512.0);   // ~one wiggle per flow cell
    float2 j = float2(sin(p.x * 6.28 + p.y * 3.1) + 0.5 * sin(p.x * 13.1 - p.y * 7.7),
                      sin(p.y * 6.28 - p.x * 2.7) + 0.5 * sin(p.y * 12.3 + p.x * 5.9));
    worldXY += j * flowMapWarp;
    return (worldXY - flowMapTransform.xy) * flowMapTransform.zw;
}
#endif

#ifdef WATER_LOD_MESH
// Flow-steered anisotropic crest displacement (world-snapped LOD mesh). Live knobs.
shared float waveAmp;       // crest amplitude (world units)
shared float waveLen;       // base wavelength along the flow (world units)
shared float waveSpeed;     // crest travel speed scale
shared float crestSpread;   // directional fan: small = longer, straighter crest lines
#endif

#ifdef WATER_FOAM
// World-anchored hybrid Voronoi particle foam (see XE Mod Foam.fx for the sim). Single
// low-res CARRIER: the sim writes foam intensity into texFoam0 over a player-tracking
// window whose world min-corner is foamOrigin0; the water shader samples it by world XY
// and the procedural detail below supplies the up-close crispness. foamOrigin is the sim's
// TRANSIENT origin (declared here, included before XE Mod Foam.fx, set for the sim passes —
// NOT used by the consume).
shared float2  foamOrigin;          // transient window origin used by the sim passes
shared float   foamWeight;          // runtime A/B: 1 = foam on, 0 = off (legacy)
shared texture texFoam0;            // the carrier buffer
shared float2  foamOrigin0;         // carrier window world min-corner (world XY)
static const float foamTexWorldSize0 = 512.0 * 25.0;    // carrier window (= 12800u; foamTexResolution * foamCascadeWorldRes[0])
sampler sampFoam0 = sampler_state { texture = <texFoam0>; minfilter = linear; magfilter = linear; mipfilter = none; bordercolor = 0; addressu = border; addressv = border; };
// Advected detail-UV offset field (River Editor; same window as texFoam0). xy = world-unit offset.
shared texture texFoamUV;
sampler sampFoamUVc = sampler_state { texture = <texFoamUV>; minfilter = linear; magfilter = linear; mipfilter = none; addressu = clamp; addressv = clamp; };

// Two-layer foam (Phase 2). Both layers share one erosion + ridged-fbm so far and near read
// as the same streaky material. foamDetail = (worldUnitsPerCell, advectRate, erodeThreshold, –).
// FAR  : flow-map-driven (river/beach routing × flow-advected ridged fbm), all distance.
// NEAR : the sim carrier (texFoam0, warped by texFoamUV) ADDS choke-point detail in its window.
shared float4 foamDetail;
shared float  foamFarAmount;        // far flow-map foam layer strength (0 = near-only)
shared float  foamVortGain;         // static vorticity (curl) concentration for the far foam (VortGain)
shared float  foamFineScale;        // multi-scale erosion: how much finer the perforating octave is (FineScale)
shared float  foamFineAmt;          // multi-scale erosion: fine-perforation strength (FineAmt; 0 = single-scale)
shared float  foamCoarseScale;      // multi-scale erosion: how much COARSER the clumping octave is (CoarseScale)
shared float  foamCoarseAmt;        // multi-scale erosion: coarse-clumping strength (CoarseAmt; 0 = off)
float foamHash(float2 p)   { p = frac(p * float2(127.1, 311.7)); p += dot(p, p + 34.23); return frac(p.x * p.y); }
float foamVnoise(float2 p) {
    float2 i = floor(p), f = frac(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = foamHash(i), b = foamHash(i + float2(1, 0)), c = foamHash(i + float2(0, 1)), d = foamHash(i + float2(1, 1));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}
float foamFbm(float2 p) { return 0.6 * foamVnoise(p) + 0.4 * foamVnoise(p * 2.13 + 19.7); }
// Ridged 4-octave fbm: v = 1 - |2*vnoise-1| folds the rounded value noise into sharp filament
// ridges; non-integer lacunarity (2.07) breaks axis-aligned tiling, gain 0.5. Sum of octave
// weights ~0.94, so the result spans ~[0,0.94] (used as 0.5 + foamFbm2 in the erosion below).
float foamFbm2(float2 p) {
    float v = 0.0, a = 0.5, f = 1.0;
    for (int i = 0; i < 4; ++i) {
        float r = 1.0 - abs(2.0 * foamVnoise(p * f) - 1.0);
        v += a * r;
        f *= 2.07; a *= 0.5;
    }
    return v;
}
// Shared erosion sharpness (consume-side const, not knobbed — foamDetail slots stay for live tuning).
static const float foamErodeSharp = 3.0;
#endif

static const float waveTexResolution = 512;
static const float waveTexWorldSize = waveTexResolution * 2.5;
static const float waveTexRcpRes = 1.0 / waveTexResolution;
static const float playerWaveSize = 12.0 / waveTexWorldSize; // 12 world units radius

//------------------------------------------------------------
// Static functions

float3 getFinalWaterNormal(float2 texcoord1, float2 texcoord2, float dist, float2 vertXY) : NORMAL
{
    float2 far_normal, close_normal;

#ifdef WATER_FLOW_MAP
    // Sample the per-body flow field at this world position.
    float4 flow = tex2Dlod(sampFlow, float4(flowWarpUV(vertXY), 0, 0));
    float2 flowDir = flow.rg * 2 - 1;
    // B is the wave-amplitude mask (decoded *3). Saturate it for normal strength so the
    // 3x open-sea amplitude doesn't over-steepen normals: pond 0 (calm), river ~0.5,
    // beach/sea full. weight 0 → sea-equivalent (1.0).
    float  intensity = lerp(1.0, saturate(flow.b * 3.0), flowMapWeight);
    // Alpha is a packed routing channel: 0.5 = neutral, <0.5 = NEAR group (river/beach), >0.5 =
    // FAR group (sea). Rivers/beaches advect the CLOSE normal (detail rolls downstream/onshore);
    // sea advects the FAR normal along the blurred coastline normal (large swells refract shoreward).
    float  nearStr = saturate((0.5 - flow.a) * 2) * flowMapWeight;
    float  farStr  = saturate((flow.a - 0.5) * 2) * flowMapWeight * flowSeaRefract;

    float t = 0.4 * flowSeaSpeed * time;   // base (sea) animation rate, separately tunable

    // Valve flow-map ping-pong. A plain "texcoord -= flowDir*speed*time" advection grows the UV
    // offset without bound; because flowDir varies in space, neighbouring pixels then diverge
    // ever further across the noise texture → severe minification = the pixelated specular and
    // apparent wave-size/frequency changes. Instead bound the offset to one cycle (frac phase,
    // max displacement flowCycleUV) and crossfade two half-phase-shifted samples so the reset is
    // never visible. Apparent scroll velocity stays = flowScrollSpeed (the RiverSpeed knob).
    float  rate  = flowScrollSpeed / flowCycleUV;                // cycles/time → velocity = flowScrollSpeed
    float  ph0   = frac(time * rate);
    float  ph1   = frac(time * rate + 0.5);
    float  blend = abs(1.0 - 2.0 * ph0);                         // triangle wave; weight→0 as a phase resets
    float2 dispF = flowDir * farStr  * flowCycleUV;              // sea → far normal (refraction); 0 elsewhere
    float2 dispN = flowDir * nearStr * flowCycleUV;              // river/beach → close normal; 0 elsewhere
    far_normal   = lerp(tex3D(sampWater3d, float3(texcoord1 - dispF * ph0, t)).rg,
                        tex3D(sampWater3d, float3(texcoord1 - dispF * ph1, t)).rg, blend);
    close_normal = lerp(tex3D(sampWater3d, float3(texcoord2 - dispN * ph0, t)).rg,
                        tex3D(sampWater3d, float3(texcoord2 - dispN * ph1, t)).rg, blend);
#else
    // Calculate the W texture coordinate based on the time that has passed
    float t = 0.4 * time;
    float3 w1 = float3(texcoord1, t);
    float3 w2 = float3(texcoord2, t);

    // Blend together the normals from different sized areas of the same texture
    far_normal = tex3D(sampWater3d, w1).rg;
    close_normal = tex3D(sampWater3d, w2).rg;
#endif

#ifdef DYNAMIC_RIPPLES
    // Blend normals from rain and player ripples (static taps — the hybrid foam sim is
    // the vehicle for flow-driven player/rain disturbance; the ripple wake stays master).
    close_normal.rg += tex2Dlod(sampRain, float4(texcoord2, 0, 0)).ba;
    close_normal.rg += tex2Dlod(sampWave, float4((vertXY - rippleOrigin) / waveTexWorldSize, 0, 0)).ba;
#endif

    float2 normal_R = 2 * lerp(close_normal, far_normal, saturate(dist / 8000)) - 1;
#ifdef WATER_FLOW_MAP
    // Calmer ponds/shallows (low intensity); sea (intensity 1) unchanged.
    normal_R *= intensity;
#endif
    return normalize(float3(normal_R, 1));
}

#ifndef FILTER_WATER_REFLECTION

float3 getProjectedReflection(float4 tex)
{
    return tex2Dproj(sampReflect, tex).rgb;
}

#else

float3 getProjectedReflection(float4 tex)
{
    float4 radius = 0.006 * saturate(0.11 + tex.w/6000) * tex.w * float4(1, rcpRes.y/rcpRes.x, 0, 0);

    float3 reflected = tex2Dproj(sampReflect, tex);
    reflected += tex2Dproj(sampReflect, tex + radius*float4(0.60, 0.10, 0, 0));
    reflected += tex2Dproj(sampReflect, tex + radius*float4(0.30, -0.21, 0, 0));
    reflected += tex2Dproj(sampReflect, tex + radius*float4(0.96, -0.03, 0, 0));
    reflected += tex2Dproj(sampReflect, tex + radius*float4(-0.40, 0.06, 0, 0));
    reflected += tex2Dproj(sampReflect, tex + radius*float4(-0.70, 0.18, 0, 0));
    reflected /= 6.0;

    return reflected.rgb;
}

#endif

float reflectionOcclusionAt(float4 tex, float actualWaterDepth)
{
    float sceneDepth = tex2Dproj(sampDepth, tex).r;
    float validScene = step(sceneDepth, nearViewRange + 64.0);

    return validScene * smoothstep(-128.0, 384.0, actualWaterDepth - sceneDepth);
}

//------------------------------------------------------------
// Water shader

struct WaterVertOut
{
    float4 position : POSITION;
    float4 pos : TEXCOORD0;
    float4 texcoords : TEXCOORD1;
    float4 screenpos : TEXCOORD2;
#if defined(DYNAMIC_RIPPLES) || defined(WATER_LOD_MESH)
    float4 screenposclamp : TEXCOORD3;
#endif
};

#if defined(WATER_LOD_MESH)

WaterVertOut WaterVS (in float4 pos : POSITION)
{
    WaterVertOut OUT;

    // World-snapped lattice: the per-level world matrix supplies cell size + snap, so
    // mul(pos, world) lands every vertex on a stable world position (no swimming).
    OUT.pos = mul(pos, world);

    // Calculate various texture coordinates
    OUT.texcoords.xy = OUT.pos.xy / 3900;
    OUT.texcoords.zw = OUT.pos.xy / 527;

    float2 worldXY = OUT.pos.xy;
    float  dist = length(eyePos.xyz - OUT.pos.xyz);

    // Sample the per-body flow field: direction (R,G), wave-amplitude mask (B), routing (A).
    float4 flow = tex2Dlod(sampFlow, float4((worldXY - flowMapTransform.xy) * flowMapTransform.zw, 0, 0));
    float2 flowDir = flow.rg * 2 - 1;
    float  flowMag = length(flowDir);
    // Per-body wave-amplitude mask baked into B (decoded *3): pond 0, river ~0.5x,
    // beach/default 1x, open sea 3x. Weight 0 (flow off) → uniform 1x (master ambient).
    float  waveMul = lerp(1.0, flow.b * 3.0, flowMapWeight);

    // ISOTROPIC ambient waves — master's water3d height field (two scales, distance-blended,
    // direction-free, tiny). The base chop present everywhere; OFF reproduces it exactly.
    // Same displacement as the standard / DYNAMIC_RIPPLES WaterVS.
    float  ta = 0.4 * time;
    float  height  = tex3Dlod(sampWater3d, float4(worldXY / 1104, ta, 0)).a;
    float  height2 = tex3Dlod(sampWater3d, float4(worldXY / 3900, ta, 0)).a;
    float  ambient = waveHeight * (lerp(height, height2, saturate(dist / 8000)) - 0.5);

    // DIRECTIONAL crests — Gerstner waves fanned about the flow heading, crests running
    // ACROSS the flow (beaches roll onshore). Added only where the flow map gives a heading.
    float2 dir0     = (flowMag > 1e-3) ? flowDir / flowMag : float2(1, 0);
    float2 crestDir = float2(-dir0.y, dir0.x);
    float  h_dir = 0.0;
    [unroll] for (int i = 0; i < 4; ++i)
    {
        float  fan    = (i - 1.5) * crestSpread;            // small directional spread
        float2 dir    = normalize(dir0 + crestDir * fan);
        float  lambda = waveLen * (1.0 + 0.35 * i);
        float  k      = 6.28318530718 / lambda;
        float  w      = waveSpeed * 30.0 * sqrt(k);         // deep-water-like dispersion
        h_dir += sin(dot(dir, worldXY) * k - w * time) / (1.0 + 0.6 * i);
    }
    float  dirW = saturate(flowMag * 4.0) * flowMapWeight;  // 0 open sea/pond, 1 beach/river

    // Combine: per-body-scaled (ambient everywhere + directional crests on flowing bodies),
    // faded near the eye and toward the fog horizon like the DYNAMIC_RIPPLES displacement.
    float addheight = waveMul * (ambient + dirW * waveAmp * h_dir)
                    * saturate(1 - dist / 6400) * saturate(dist / 200);
    OUT.pos.z += addheight;

    // Silhouette uses the displaced height.
    OUT.position = mul(OUT.pos, view);
    OUT.position = mul(OUT.position, proj);
    OUT.screenpos = float4(0.5 * (1 + rcpRes) * OUT.position.w + float2(0.5, -0.5) * OUT.position.xy, OUT.position.zw);

    // Planar reflection samples the FLAT water plane (height removed), as in the
    // DYNAMIC_RIPPLES path — a planar mirror RT has no height information.
    float4 flatPos = OUT.pos - float4(0, 0, addheight, 0);
    flatPos = mul(flatPos, view);
    flatPos = mul(flatPos, proj);
    OUT.screenposclamp = float4(0.5 * (1 + rcpRes) * flatPos.w + float2(0.5, -0.5) * flatPos.xy, flatPos.zw);

    return OUT;
}

#elif !defined(DYNAMIC_RIPPLES)

WaterVertOut WaterVS (in float4 pos : POSITION)
{
    WaterVertOut OUT;

    // Add z bias to avoid fighting with MW ripples quads
    OUT.pos = mul(pos, world);
    OUT.pos.z -= 0.1;

    // Calculate various texture coordinates
    OUT.texcoords.xy = OUT.pos.xy / 3900;
    OUT.texcoords.zw = OUT.pos.xy / 527;

    // Calculate screen position for refraction
    OUT.position = mul(OUT.pos, view);
    OUT.position = mul(OUT.position, proj);
    OUT.screenpos = float4(0.5 * (1 + rcpRes) * OUT.position.w + float2(0.5, -0.5) * OUT.position.xy, OUT.position.zw);

    return OUT;
}

#else

WaterVertOut WaterVS (in float4 pos : POSITION)
{
    WaterVertOut OUT;

    // Move to world space
    OUT.pos = mul(pos, world);

    // Calculate various texture coordinates
    OUT.texcoords.xy = OUT.pos.xy / 3900;
    OUT.texcoords.zw = OUT.pos.xy / 527;

    // Apply vertex displacement
    float t = 0.4 * time;
    float height = tex3Dlod(sampWater3d, float4(OUT.pos.xy / 1104, t, 0)).a;
    float height2 = tex3Dlod(sampWater3d, float4(OUT.pos.xy / 3900, t, 0)).a;
    float dist = length(eyePos.xyz - OUT.pos.xyz);

    float addheight = waveHeight * (lerp(height, height2, saturate(dist/8000)) - 0.5) * saturate(1 - dist/6400) * saturate(dist/200);
    OUT.pos.z += addheight;
    // Calculate screen position for refraction
    OUT.position = mul(OUT.pos, view);
    OUT.position = mul(OUT.position, proj);
    OUT.screenpos = float4(0.5 * (1 + rcpRes) * OUT.position.w + float2(0.5, -0.5) * OUT.position.xy, OUT.position.zw);

    // Reflection sample point: reconstruct the FLAT water plane by removing the wave
    // displacement (signed), so planar reflection depends only on the surface normal
    // (ripple distortion via reffactor), not on vertex height. A planar mirror RT has
    // no height information, so coupling them just makes displaced water near shore
    // sample above the flat waterline and read sky. Height drives the silhouette
    // (OUT.position) alone.
    float4 flatPos = OUT.pos - float4(0, 0, addheight, 0);
    flatPos = mul(flatPos, view);
    flatPos = mul(flatPos, proj);
    OUT.screenposclamp = float4(0.5 * (1 + rcpRes) * flatPos.w + float2(0.5, -0.5) * flatPos.xy, flatPos.zw);

    return OUT;
}
#endif

float4 WaterPS(in WaterVertOut IN): COLOR0
{
    // Calculate eye vector
    float3 EyeVec = IN.pos.xyz - eyePos.xyz;
    float dist = length(EyeVec);
    EyeVec /= dist;

    // Define fog
    float4 fog = fogColourWater(EyeVec, dist);
    float3 depthColor = fogApply(depthBaseColor, fog);

    // Calculate water normal
    float3 normal = getFinalWaterNormal(IN.texcoords.xy, IN.texcoords.zw, dist, IN.pos.xy);

    // Reflection/refraction pixel distortion factor, wind strength increases distortion
    float2 reffactor = (windFactor * dist + 0.1) * normal.xy;

    // Distort refraction dependent on depth
    float4 newscrpos = IN.screenpos + float4(reffactor.yx, 0, 0);
    float sceneDepth = tex2Dproj(sampDepth, newscrpos).r;
    float aboveWaterDepth = step(sceneDepth + shoreDepthBias, IN.screenpos.w);
    float depth = max(shoreDepthBias, sceneDepth - IN.screenpos.w);

    // Refraction
    float3 refracted = depthColor;
    float shorefactor = 0;

    // Avoid sampling deep water
    if(depth < 4000 && aboveWaterDepth < 0.5)
    {
        // Sample refraction texture
        newscrpos = IN.screenpos + saturate(depth / 100) * float4(reffactor.yx, 0, 0);
        refracted = tex2Dproj(sampRefract, newscrpos).rgb;

        // Get distorted depth
        sceneDepth = tex2Dproj(sampDepth, newscrpos).r;
        aboveWaterDepth = step(sceneDepth + shoreDepthBias, IN.screenpos.w);
        depth = max(shoreDepthBias, sceneDepth - IN.screenpos.w);
        depth /= dot(EyeVec, float3(view[0][2], view[1][2], view[2][2]));

        // Small scale shoreline animation
        depth += 300 * (0.95 - normal.z);

        float depthscale = saturate(exp(-depth / 500) * 1.0);
        shorefactor = pow(depthscale, 25);
		

        // Make transition between actual refraction image and depth color depending on water depth
        refracted = lerp(depthColor, refracted, 0.8 * depthscale + 0.2 * shorefactor);
    }

    // Sample reflection texture
#if defined(DYNAMIC_RIPPLES) || defined(WATER_LOD_MESH)
    float4 screenpos = IN.screenposclamp;
#else
    float4 screenpos = IN.screenpos;
#endif

#if DEBUG_REFLECTION_RAW
    // Raw reflection inspection: flat-position sample, no distortion/fog/fresnel.
    return float4(getProjectedReflection(screenpos), 1);
#endif

    float4 reflectedPos = screenpos - float4(2.1 * reffactor.x, -abs(reffactor.y), 0, 0);
    reflectedPos.xy = lerp(reflectedPos.xy, screenpos.xy, reflectionOcclusionAt(reflectedPos, IN.screenpos.w));
    float3 reflected = getProjectedReflection(reflectedPos);

    // Fade reflection into an inscatter dominated horizon
    reflected = lerp(reflected * 0.96, reflected, fog.a);

    // Smooth out high frequencies at a distance
    float3 adjustnormal = lerp(float3(0, 0, 0.1), normal, pow(saturate(1.05 * fog.a), 2));
    adjustnormal = lerp(adjustnormal, float3(0, 0, 1.0), (1 + EyeVec.z) * (1 - saturate(1 / (dist / 1000 + 1))));

    // Fresnel equation determines reflection/refraction
    float fresnel = dot(-EyeVec, adjustnormal);
    fresnel = 0.02 + pow(saturate(0.9988 - 0.28 * fresnel), 16);
    float3 result = lerp(refracted, reflected, fresnel);
	


    // Specular lighting
    // This should use Blinn-Phong, but it doesn't work so well for area lights like the sun
    // Instead multiply and saturate to widen a Phong specular lobe which better simulates an area light
    float vdotr = dot(-EyeVec, reflect(-sunPos, normal));
    vdotr = saturate(1.0025 * vdotr);
    float3 spec = sunColAdjusted * (pow(vdotr, 170) + 0.07 * pow(vdotr, 4));
    result += spec * fog.a;
	
	// Water cut feature, vtastek
	float wdist = dist/lerp(1200, 0, saturate((eyePos.z - waterLevel)/7.0));
	float wcut = smoothstep(0.09,0.1, wdist);
	float wcutdark = smoothstep(0.0889, 0.101, wdist);
	wcutdark = wcutdark *  (1 - wcutdark);
	wcutdark = saturate(wcutdark*3);
	
	// Include water cut feature
	result = lerp(refracted, result, wcut);
	result = lerp(result, result * 0.1, wcutdark);

    // Smooth transition at shore line
    result = lerp(result, refracted, shorefactor * fog.a);

    // Note that both refraction and reflection textures have fog applied already

    float3 dbgShader = 0;   // debug: in-shader foam quantity captured for the SH_* overlay views

#ifdef WATER_FOAM
    // World-anchored hybrid particle foam (sim in XE Mod Foam.fx). Sample the low-res carrier
    // by world XY over its player-tracking window; the border (0) outside the window yields no
    // foam, and foamWeight 0 (A/B off) → legacy water untouched.
    {
        float2 fuv0 = (IN.pos.xy - foamOrigin0) / foamTexWorldSize0;
        // Near-layer window edge fade: fade to 0 over the outer ~12% so the hard carrier
        // window boundary is not visible; the far layer carries on past it seamlessly.
        float2 ef0v = saturate(min(fuv0, 1.0 - fuv0) / 0.12);

        // --- FAR: flow-map-driven, all distance (extends as far as the flow map has data) ---
        // Routing mask uses flow.a (river/beach NEAR group), NOT flow.b — B is 1/3 in dry water
        // and would foam open sea. Advect a ridged fbm along the flow direction, then erode to
        // crisp streaks. No sim dependency, so this reaches the full flow-map extent.
        float4 fl    = tex2Dlod(sampFlow, float4(flowWarpUV(IN.pos.xy), 0, 0));
        float2 fdir  = fl.rg * 2 - 1;
        float  rmask = saturate((0.5 - fl.a) * 2) * flowMapWeight;     // river/beach routing
        // Static vorticity = curl of the baked flow direction (d(dirY)/dx - d(dirX)/dy). High at
        // bends / confluences / shear lines — where real foam gathers. 4 unwarped neighbour taps
        // (~one 512u cell apart); |curl| × VortGain concentrates the scrolling foam there. The
        // *0.5+0.5 encoding scale folds into the gain. VortGain 0 = today's uniform-river far foam.
        float2 ftO   = flowMapTransform.xy;
        float2 ftS   = flowMapTransform.zw;
        float  gR    = tex2Dlod(sampFlow, float4((IN.pos.xy + float2(512, 0) - ftO) * ftS, 0, 0)).g;
        float  gL    = tex2Dlod(sampFlow, float4((IN.pos.xy - float2(512, 0) - ftO) * ftS, 0, 0)).g;
        float  rU    = tex2Dlod(sampFlow, float4((IN.pos.xy + float2(0, 512) - ftO) * ftS, 0, 0)).r;
        float  rD    = tex2Dlod(sampFlow, float4((IN.pos.xy - float2(0, 512) - ftO) * ftS, 0, 0)).r;
        float  vort  = abs((gR - gL) - (rU - rD)) * foamVortGain;
        float  farCover = rmask * (1.0 + vort);                        // concentrate at turbulent spots
        // Valve flow-map ping-pong (mirrors getFinalWaterNormal L134+): bound the advection to one
        // cycle and crossfade two half-phase samples so it never shears into vortexes. DECOUPLED from
        // the normals' CycleUV (which the user keeps large for the normals → huge slow morph that hid
        // FoamSpeed). Fixed modest 512-world-u cycle → short period, smooth crossfade, and FoamSpeed
        // is now a clean, responsive scroll-speed control. World velocity = 512 * foamDetail.y.
        float2 fP    = IN.pos.xy / foamDetail.x;
        float  fcyc  = 512.0 / foamDetail.x;                     // fixed displacement per cycle (fbm-tile units)
        float  frate = foamDetail.y;                             // cycles/time → world velocity = 512 * FoamSpeed
        float  fph0  = frac(time * frate);
        float  fph1  = frac(time * frate + 0.5);
        float  fbl   = abs(1.0 - 2.0 * fph0);
        float2 fdisp = fdir * fcyc;
        float  nf    = lerp(foamFbm2(fP - fdisp * fph0), foamFbm2(fP - fdisp * fph1), fbl);
        // Multi-scale erosion, COARSE octave: a lower-frequency advected octave clumps the coverage
        // into patches so big uniform stretches break into foam clusters (real foam gathers, it isn't
        // a flat sheet). foamCoarseScale = how much coarser than the base; foamCoarseAmt = clump strength.
        float2 fPc   = fP / foamCoarseScale;                          // coarser (bigger clumps)
        float2 fdc   = fdisp / foamCoarseScale;                       // same world scroll velocity
        float  ncrs  = lerp(foamFbm2(fPc - fdc * fph0), foamFbm2(fPc - fdc * fph1), fbl);
        farCover    *= lerp(1.0, saturate(0.4 + ncrs), foamCoarseAmt);  // clump into patches
        // Multi-scale erosion, FINE octave: a finer advected octave perforates the foam edge into lacy /
        // bubbly detail (real foam has structure at several scales). foamFineScale = how much finer than
        // the base; foamFineAmt = how hard it carves (0 = single-scale, today's streaky look).
        float2 fPf   = fP * foamFineScale;
        float2 fdf   = fdisp * foamFineScale;                          // same world scroll velocity
        float  nfine = lerp(foamFbm2(fPf - fdf * fph0), foamFbm2(fPf - fdf * fph1), fbl);
        float  farFoam = saturate((farCover * (0.5 + nf) - foamDetail.z) * foamErodeSharp);
        farFoam *= lerp(1.0, saturate(0.5 + nfine), foamFineAmt);      // fine perforation
        farFoam *= foamFarAmount;

        // --- NEAR: sim carrier adds dynamic, FLOWING choke-point detail within its window. It uses
        // the SAME ping-pong advection as the far layer (so it flows/matches instead of just being
        // stretched by the static sim warp), PLUS the sim's UV warp (off) for sim-driven structure. ---
        float  simD   = tex2Dlod(sampFoam0, float4(fuv0, 0, 0)).r;
        float  window = foamWeight * ef0v.x * ef0v.y;                  // carrier window fade (0 when foam A/B off)
        float  cover  = simD * window;
        float2 off    = tex2Dlod(sampFoamUVc, float4(fuv0, 0, 0)).xy;  // sim flow perturbation (warp)
        float2 nP     = (IN.pos.xy + off) / foamDetail.x;
        float  nn     = lerp(foamFbm2(nP - fdisp * fph0), foamFbm2(nP - fdisp * fph1), fbl);  // SAME ping-pong as far
        float  nearFoam = saturate((cover * (0.5 + nn) - foamDetail.z) * foamErodeSharp);

        // Mixing: the sim MODULATES the far foam up and down (more where foam piles, less where it
        // clears) instead of a plain max(). foamDetail.w (FoamMix) scales how strongly it overrides
        // the far baseline; faded by the window so the far layer carries on seamlessly past the carrier.
        float  foam  = lerp(farFoam, nearFoam, saturate(foamDetail.w * window));

        // Debug: capture an in-pipeline quantity for the SH_* overlay views (8..11). The foam is
        // still composited below, so the effect stays live (only the final overlay swaps it in).
        if (flowDebugView > 7.5)
        {
            if (flowDebugView < 8.5)       dbgShader = rmask.xxx;                 // 8  SH RMASK
            else if (flowDebugView < 9.5)  dbgShader = float3(fdir * 0.5 + 0.5, 0.5); // 9  SH FDIR
            else if (flowDebugView < 10.5) dbgShader = saturate(nf).xxx;          // 10 SH FBM
            else                           dbgShader = saturate(farFoam).xxx;     // 11 SH FARFOAM
        }

        float  shoreMask = saturate(1.0 - depth / 800.0);   // gather toward shallows/shore
        float3 foamCol = sunColAdjusted + 0.25;
        result = lerp(result, foamCol, saturate(foam * (0.4 + 0.6 * shoreMask)) * fog.a);
    }
#endif

#ifdef WATER_FLOW_MAP
    // Water-flow debug overlays (id in flowDebugView):
    //   2..7  BAKED source views — show the re-baked texture RGB (CLASSIFY / DIRECTION / STRENGTH /
    //         INTENSITY / DIST RAW / DIST SMOOTH). The bake encodes the value; effect is off.
    //   8..11 SHADER views — dbgShader captured above from the live foam pipeline; foam stays on.
    if (flowDebugView > 0.5)
    {
        if (flowDebugView < 7.5)
        {
            float3 t = tex2Dlod(sampFlow, float4(flowWarpUV(IN.pos.xy), 0, 0)).rgb;
            float m = step(0.01, dot(t, 1));   // only tint where a value was baked
            result = lerp(result, t, 0.8 * m);
        }
        else
        {
            result = lerp(result, dbgShader, 0.85);
        }
    }
#endif

    return float4(result, 1);
}

float4 UnderwaterPS(in WaterVertOut IN): COLOR0
{
    // Calculate eye vector
    float3 EyeVec = IN.pos.xyz - eyePos.xyz;
    float dist = length(EyeVec);
    EyeVec /= dist;

    // Special case fog, avoid fog offset
    float fog = saturate(exp(-dist / 4096));

    // Calculate water normal
    float3 normal = -getFinalWaterNormal(IN.texcoords.xy, IN.texcoords.zw, dist, IN.pos.xy);

    // Reflection / refraction pixel distortion factor, wind strength increases distortion
    float2 reffactor = 2 * (windFactor * dist + 0.1) * normal.xy;

    // Distort refraction
    float4 newscrpos = IN.screenpos + float4(2 * -reffactor.xy, 0, 0);
    float3 refracted = tex2Dproj(sampRefract, newscrpos).rgb;
    refracted = lerp(fogColFar, refracted, exp(-dist / 500));

    // Sample reflection texture
    float4 reflectedPos = IN.screenpos - float4(2.1 * reffactor.x, -abs(reffactor.y), 0, 0);
    reflectedPos.xy = lerp(reflectedPos.xy, IN.screenpos.xy, reflectionOcclusionAt(reflectedPos, IN.screenpos.w));
    float3 reflected = getProjectedReflection(reflectedPos);

    // Fresnel equation, including total internal reflection
    float fresnel = pow(saturate(1.12 - 0.65 * dot(-EyeVec, normal)), 8);
    float3 result = lerp(refracted, reflected, fresnel);

    // Sun refraction
    float refractsun = dot(-EyeVec, normalize(-sunPos + normal));
    float3 spec = sunColAdjusted * pow(refractsun, 6) * fog;

    return float4(result + spec, 1);
}
