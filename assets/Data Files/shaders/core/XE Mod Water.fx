
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
#endif

#ifdef WATER_LOD_MESH
// Flow-steered anisotropic crest displacement (world-snapped LOD mesh). Live knobs.
shared float waveAmp;       // crest amplitude (world units)
shared float waveLen;       // base wavelength along the flow (world units)
shared float waveSpeed;     // crest travel speed scale
shared float crestSpread;   // directional fan: small = longer, straighter crest lines
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
    float4 flow = tex2Dlod(sampFlow, float4((vertXY - flowMapTransform.xy) * flowMapTransform.zw, 0, 0));
    float2 flowDir = flow.rg * 2 - 1;
    float  intensity = lerp(1.0, flow.b, flowMapWeight);          // weight 0 → sea-equivalent
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
    // Blend normals from rain and player ripples
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

    // Sample the per-body flow field: direction, intensity (B), near/far routing (A).
    float4 flow = tex2Dlod(sampFlow, float4((worldXY - flowMapTransform.xy) * flowMapTransform.zw, 0, 0));
    float2 flowDir = flow.rg * 2 - 1;
    float  flowMag = length(flowDir);
    flowDir = (flowMag > 1e-3) ? flowDir / flowMag : float2(1, 0);
    float  intensity = lerp(1.0, flow.b, flowMapWeight);
    float  farStr    = saturate((flow.a - 0.5) * 2);   // sea group → longer swell

    // Crest lines run ACROSS the flow; waves are short ALONG it. The sea class widens
    // the wavelength (long swells); rivers/beaches stay tight.
    float2 crestDir = float2(-flowDir.y, flowDir.x);
    float  baseLen  = waveLen * lerp(1.0, 3.0, farStr);

    // Vertical Gerstner sum: a few waves fanned slightly about flowDir → short
    // wavelength in the travel direction, long coherent crest lines perpendicular.
    // h depends on stable world XY + time, so crests travel with the flow through the
    // world and stay put on the lattice.
    float h = 0.0;
    [unroll] for (int i = 0; i < 4; ++i)
    {
        float  fan    = (i - 1.5) * crestSpread;            // small directional spread
        float2 dir    = normalize(flowDir + crestDir * fan);
        float  lambda = baseLen * (1.0 + 0.35 * i);
        float  k      = 6.28318530718 / lambda;
        float  w      = waveSpeed * 30.0 * sqrt(k);         // deep-water-like dispersion
        h += sin(dot(dir, worldXY) * k - w * time) / (1.0 + 0.6 * i);
    }

    // Amplitude: knob * body intensity, faded near the eye and toward the fog horizon
    // exactly like the radial DYNAMIC_RIPPLES displacement.
    float amp = waveAmp * intensity * lerp(1.0, 0.7, farStr);
    float addheight = amp * h * saturate(1 - dist / 6400) * saturate(dist / 200);
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

#ifdef WATER_FLOW_MAP
    // CLASSIFY debug view: overlay the baked category colour (green=river, red=pond,
    // yellow=beach, blue=sea) so the flood-fill classification can be validated directly.
    if (flowDebugView > 0.5)
    {
        float3 cat = tex2Dlod(sampFlow, float4((IN.pos.xy - flowMapTransform.xy) * flowMapTransform.zw, 0, 0)).rgb;
        float m = step(0.01, dot(cat, 1));   // only tint where a category was baked
        result = lerp(result, cat, 0.8 * m);
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
