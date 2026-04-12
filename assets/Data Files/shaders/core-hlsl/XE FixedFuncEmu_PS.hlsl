//============================================================================
// MGE XE - HLSL Fixed Function Emulation - Pixel Shader  
//============================================================================

//------------------------------------------------------------
// Shared Variables

// Materials
float4 materialDiffuse, materialAmbient, materialEmissive;
float4 shadingMode; // .z = materialMode (1=none, 2=diffamb, 3=emissive)

float3 lightSceneAmbient;
#ifndef NOLIT
// Lighting - Basic

float3 lightSunDiffuse;
float3 lightSunDirection;

// Point light uniforms (only compiled for LIGHT_MODE 1 or 2)
#if defined(LIGHT_MODE) && LIGHT_MODE >= 1 && LIGHT_MODE <= 2
float4 lightDiffuse[8];
float3 lightPosition[8];
float lightAmbient[8];
float lightFalloffQuadratic[8];
float lightFalloffConstant;
int pointLightCount;
#endif
#endif

// Fog
float3 fogColNear;

// Textures with explicit register bindings for DX9 HLSL (original slot order)
texture tex0 : register(t0);  // Base texture (or _diffparam when diffparam replaces base)
#if defined(HAS_DETAIL)
texture tex1 : register(t1);  // Detail texture (conditional only)
#endif
#if defined(HAS_PARAMH)
texture tex2 : register(t2);  // Parameter map (_paramh: metallic/roughness|height/IOR)
#endif
#if defined(HAS_PARAMX)
texture tex3 : register(t3);  // Anisotropic map (_paramx: aniso rotation/strength/metallic)
#endif
#if defined(HAS_SHADOWS)
texture tex4 : register(t4);  // Shadow map (original slot)
#endif
sampler sampTex0 : register(s0) = sampler_state{ texture = <tex0>; minfilter = anisotropic; magfilter = linear; mipfilter = linear; maxanisotropy = 16; };  // Base or diffparam texture
#if defined(HAS_DETAIL)
sampler sampDetail : register(s1) = sampler_state{ texture = <tex1>; minfilter = anisotropic; magfilter = linear; mipfilter = linear; maxanisotropy = 16; }; // Detail texture (conditional)
#endif
#if defined(HAS_PARAMH)
sampler sampTex2 : register(s2) = sampler_state{ texture = <tex2>; minfilter = anisotropic; magfilter = linear; mipfilter = linear; maxanisotropy = 16; };  // Parameter map (_paramh)
#endif
#if defined(HAS_PARAMX)
sampler sampTex3 : register(s3) = sampler_state{ texture = <tex3>; minfilter = anisotropic; magfilter = linear; mipfilter = linear; maxanisotropy = 16; };  // Anisotropic map (_paramx)
#endif
#if defined(HAS_SHADOWS)
sampler sampShadow : register(s4) = sampler_state{ texture = <tex4>; addressu = border; addressv = border; bordercolor = 0xffffffff; minfilter = linear; magfilter = linear; }; // Shadow map (original slot)
#endif

// View inverse matrix for converting view-space to world space (used by texture lights and shadows)
matrix viewInverse : register(c18);

#if defined(HAS_SHADOWS)
// Shadow resolution parameter (will be set via shader constants)
float shadowRcpRes : register(c10);
#endif

// Alpha testing/blending flag
bool hasAlpha;

// Texture suffix support - using preprocessor defines
//#define HAS_DIFFPARAM  // _diffparam/_diffparam_t replaces base texture (tex0)
//#define HAS_PARAMH     // _paramh metallic/roughness|height/IOR (tex2)
//#define HAS_PARAMX     // _paramx aniso rotation/strength/metallic (tex3)

#define USE_PARALLAX // Enable this for simple offset parallax mapping
#define USE_PARALLAX_SHADOWS // Enable this for simple offset parallax mapping
#ifdef HAS_PARAMH
float2 normres;  // Parameter map texture resolution (width, height) for height mapping
#endif

#ifdef USE_STATELESS_BATCH
// Draw data texture for per-draw material lookup
texture texDrawData : register(t6);
sampler sampDrawData : register(s6) = sampler_state {
    texture = <texDrawData>;
    minfilter = point;
    magfilter = point;
    mipfilter = none;
    addressu = clamp;
    addressv = clamp;
};
float4 drawDataParams : register(c20);  // {1/width=0.125, 1/height, 0, 0}
#endif

//------------------------------------------------------------
// Vertex Output (matches VS_OUTPUT from vertex shader)
struct VS_OUTPUT {
	float4 position : POSITION;
	float3 normal : TEXCOORD0;
	float4 color : COLOR0;
	float2 texcoord : TEXCOORD1;
	float3 viewPos : TEXCOORD2;  // View space position
	float fog : TEXCOORD3;       // Fog factor
#ifdef HAS_SHADOWS
	float4 shadow0pos : TEXCOORD4;  // Shadow map 0 position
	float4 shadow1pos : TEXCOORD5;  // Shadow map 1 position
#endif
#ifdef USE_STATELESS_BATCH
	float drawIndex : TEXCOORD6;  // Draw index for material lookup
#endif
};


#include "common.hlsl"
#include "lighting.hlsl"
#include "shadows.hlsl"

//------------------------------------------------------------
// Pixel Shader Main
float4 ps_main(VS_OUTPUT input) : COLOR{
	// Local material variables - either sampled from draw data texture or copied from uniforms
	float4 useDiffuse;
	float4 useAmbient;
	float4 useEmissive;
	float4 useLightParams;

#ifdef USE_STATELESS_BATCH
	// Sample material and light params from draw data texture (overrides uniform values)
	float drawV = (input.drawIndex + 0.5) * drawDataParams.y;
	float texelW = drawDataParams.x;  // 0.125 for 8-wide texture

	// Texels 3,4,5 are diffuse, ambient, emissive
	// Use tex2Dlod with mip 0 to force point sampling and ignore PS derivatives
	useDiffuse = tex2Dlod(sampDrawData, float4(3.5 * texelW, drawV, 0, 0));
	useAmbient = tex2Dlod(sampDrawData, float4(4.5 * texelW, drawV, 0, 0));
	useEmissive = tex2Dlod(sampDrawData, float4(5.5 * texelW, drawV, 0, 0));
	// Note: useEmissive.w contains alphaRef (not used in PS, handled by device state)

	// Texel 6 is light params: {lightCount, texelSize, texelOffset, 0}
	useLightParams = tex2Dlod(sampDrawData, float4(6.5 * texelW, drawV, 0, 0));
#else
	// Use uniform values for non-batched rendering
	useDiffuse = materialDiffuse;
	useAmbient = materialAmbient;
	useEmissive = materialEmissive;
	#ifdef USE_TEXTURE_LIGHTS
	useLightParams = lightDataParams;
	#else
	useLightParams = float4(0, 0, 0, 0);  // Not used for non-texture light modes
	#endif
#endif

	// Enhanced texture sampling with suffix support
	float3 diffuseParam = float3(1.0, 1.0, 1.0);  // Default white
	float3 deb = 0;
	float2 parallaxUV = input.texcoord;
	float3 normalVS = normalize(input.normal);
	
	float3 ambient = INTENSITY * pow((lightSceneAmbient) + EPS, 2.2) / PI;
	float shadows = 1.0;
	#ifdef HAS_SHADOWS
		float ndotlgeo = 1;
		#ifndef NOLIT
		ndotlgeo = dot(normalVS, -normalize(lightSunDirection));
		//deb = ndotlgeo;
		
		shadows = shadowSample(input.shadow0pos, input.shadow1pos, ndotlgeo, hasAlpha ? 1.0 : 0.0);
		deb = shadows; // returns 0 black
		#endif
	#endif

	#if defined(HAS_NORMAL) || defined(HAS_PARAMH)
	float3 inputVS = GetSafeNormal(input.viewPos, input.normal);
	#else
	float3 inputVS = input.normal;
	#endif

	float bumpIntensity = 5.5;
	float3 T, B, N;
	BuildPerPixelTBN(inputVS, input.viewPos, parallaxUV, T, B, N);

	float3 Vvs = normalize(input.viewPos);
	float handedness = (dot(cross(T, B), N) < 0) ? -1.0 : 1.0;
	//B *= handedness;
	float3x3 TBN = float3x3(T, B, N);
	float3x3 TBN_T = transpose(TBN);
	float3 Vts = mul(Vvs, TBN_T);

	// Height mapping and normal calculation using _paramh green channel
	#if defined(HAS_PARAMH)
		#if defined(USE_PARALLAX)
		// Use height from green channel of _paramh texture
		parallaxUV = Parallax(sampTex2, parallaxUV, Vts, 0.000015 * 1.33, 1); // heightScale, green channel
		#endif

		// Calculate normal from height gradient in green channel
		
		float deriv = 1.5;
		// if(parallaxUV.x > 0.5)
			// deriv = 2.0;
		
		float2 texel = 1.0 / normres;
		float hL = tex2D(sampTex2, parallaxUV + float2(-texel.x * deriv, 0)).g; // Green = height
		float hR = tex2D(sampTex2, parallaxUV + float2(texel.x * deriv, 0)).g;
		float hD = tex2D(sampTex2, parallaxUV + float2(0, -texel.y * deriv)).g;
		float hU = tex2D(sampTex2, parallaxUV + float2(0,  texel.y * deriv)).g;
		float dhdu = (hR - hL);
		float dhdv = (hU - hD);

		float3 nTS = normalize(float3(-dhdu * heightScale, dhdv * heightScale, 1.0));

		normalVS = normalize(mul(nTS, TBN));
		// Soft parallax shadowing (only if lighting is enabled)
			#ifndef NOLIT
			float3 lightDirVS = -lightSunDirection;
			float3 lightDirTS = mul(lightDirVS, TBN_T);
				#ifdef USE_PARALLAX_SHADOWS
					float shadowpara = ParallaxSoftShadow(sampTex2, input.texcoord, lightDirTS.xy, 5.0, 0.04 * 0.75, 1); // green channel
					shadows *= shadowpara;
				#endif
			#endif
	#else
		// No height mapping - use interpolated normal
		normalVS = inputVS;
	#endif


	float4 texColor = tex2D(sampTex0, parallaxUV);
	texColor.rgb = max(0.0, toLinear(texColor.rgb));
	//texColor.rgb = 0.18;
	// Note: When HAS_DIFFPARAM is defined, sampTex0 contains the _diffparam/_diffparam_t texture
	#ifdef HAS_DIFFPARAM
	texColor.a = 1.0; // Ignore alpha from _diffparam texture
	#endif

	// PBR lighting calculation
	float3 V = normalize(-input.viewPos); // view direction in view space
	float3 Norm = normalVS;
	float3 albedo = texColor.rgb;
	float roughness = 0.9;  // Default roughness
	float metalness = 0.0;  // Default metalness
	float ao = 1.0;         // Default ambient occlusion
	float radius = 1.6;
	float3 F0 = 0.08 * float3(0.5, 0.5, 0.5); // Default specular reflectance

	// PBR parameter system with Disney parametrization
#ifdef HAS_PARAMH
	float4 paramh = tex2D(sampTex2, parallaxUV);
	metalness = paramh.r;  // Red = metalness
	// Green = height (already used for parallax above)
	float ior_param = paramh.b;  // Blue = IOR parameter (Disney parametrization)

	// Check if base texture has alpha for roughness
	#ifndef HAS_DIFFPARAM
		// Base texture has alpha, use _paramh green for roughness
		if (texColor.a > 0.0) {
			roughness = paramh.g;  // Green = roughness when base has alpha
		}
	#endif

	// Disney F0 parametrization: 0.5 maps to standard dielectric values
	float ior_factor = ior_param * ior_param;  // Square for better control
	F0 = 0.04 * ior_factor;  // Dielectric base reflectance scaled by IOR

	#ifdef HAS_PARAMX
		// Anisotropic parameters - _paramx overrides metalness from _paramh
		float4 paramx = tex2D(sampTex3, parallaxUV);
		// paramx.r = aniso rotation (unused for now)
		// paramx.g = aniso strength (unused for now)
		metalness = paramx.b;  // Blue = metalness override when _paramx exists
	#endif

	// Apply metallic workflow: F0 = lerp(dielectric_F0, albedo, metalness)
	F0 = lerp(F0, albedo, metalness);
	ao = 1.0;  // Full AO for _paramh textures
#endif

#ifdef HAS_DIFFPARAM
	// Basic PBR mode: _diffparam texture with fixed material properties
	// RGB = albedo, A = roughness. Fixed: metalness=0, specular=0.5, AO=1
	// Sample diffparam alpha directly for roughness (before texColor.a was overwritten with original alpha)
	float diffparamAlpha = tex2D(sampTex0, parallaxUV).a;
	roughness = diffparamAlpha;
	metalness = 0.0;         // Non-metallic materials
	F0 = 0.08 * 0.5;         // Fixed specular reflectance = 0.5
	ao = 1.0;                // Full ambient occlusion
#endif

#ifndef NOLIT


	// Sun light (Oren-Nayar) - using LightResult structure
	float3 Lsun = normalize(-lightSunDirection);
	float sunAtten = shadows;
	LightResult sunLR = BRDF(Norm, V, Lsun, texColor.rgb, metalness, roughness, roughness, radius, F0, 1, shadows);
	//deb = sunAtten;
	float dotsun = dot(Norm, Lsun);
	float NdotL_sun = max(dotsun, 0.0);

	#ifdef HAS_GRASS
	// Apply grass-specific wrap lighting for two-sided grass rendering
	// if (input.color.r > 0.5) {
		  float w = GRASS_WRAP_LIGHTING_COEFF_W;
		  float n = GRASS_WRAP_LIGHTING_COEFF_N;
		  float lambert = dotsun * -sign(dot(V, Norm));
		  lambert = pow(saturate((lambert + w) / (1.0f + w)), n) * (n + 1) / (2 * (1 + w)) + max(0.0, -1.0 * lambert) * GRASS_BACKLIGHTING_COEFF;
		  lambert = max(0.0, lambert);
		  NdotL_sun = lambert;

		  // }
	#endif

	// Separate diffuse and specular lighting
	float3 sunIntensity = INTENSITY * pow(lightSunDiffuse + EPS, 2.2) * sunAtten * NdotL_sun;
	//deb = sunAtten;
	float3 diffuseLight = sunLR.diffuse * sunIntensity;
	#if defined(HAS_PARAMH) || defined(HAS_NORMAL)
	float3 specularLight = sunLR.specular * sunIntensity;
	#else
	float3 specularLight = 0.0;
	#endif
	//deb = shadowpara;

	float neglight = 0.0;

	// Point light evaluation — 4-way switch on LIGHT_MODE
	#if defined(LIGHT_MODE) && LIGHT_MODE == 1
	// Mode 1: Single point light — unrolled, no loop
	{
		float3 L = lightPosition[0] - input.viewPos;
		float dist = length(L);
		L = L / dist;

		float falloff = 40 * lightFalloffQuadratic[0] * dist * dist + lightFalloffConstant;
		float t = saturate(dist / 350.0);
		float cutoff = 1.0 - t * t * t * t;
		float attenuation = (falloff > 0.0) ? (1.0 / falloff) * cutoff : 0.0;
		LightResult pointLR = BRDF(Norm, V, L, texColor.rgb, metalness, roughness, roughness, radius, F0, 0, 1.0);
		float dotpoint = dot(Norm, L);
		float NdotL_point = max(dotpoint, 0.0);

		#ifdef HAS_GRASS
			lambert = dotpoint * -sign(dot(V, Norm));
			lambert = pow(saturate((lambert + w) / (1.0f + w)), n) * (n + 1) / (2 * (1 + w)) + max(0.0, -1.0 * lambert) * GRASS_BACKLIGHTING_COEFF;
			lambert = max(0.0, lambert);
			NdotL_point = 3.14 * lambert;
		#endif

		float3 pointIntensity = 10 * INTENSITY * pow(max(0.0, lightDiffuse[0].rgb) + EPS, 2.2) * NdotL_point * attenuation;
		diffuseLight += pointLR.diffuse * pointIntensity;

		#if defined(HAS_PARAMH) || defined(HAS_NORMAL)
		specularLight += pointLR.specular * pointIntensity;
		#endif
		neglight -= max(0.0, -lightDiffuse[0].r) * 1 / pow(falloff, 1 / 3.2);
	}
	#elif defined(LIGHT_MODE) && LIGHT_MODE == 2
	// Mode 2: Few point lights — loop up to pointLightCount (max 8)
	for (int i = 0; i < pointLightCount; i++)
	{
		float3 L = lightPosition[i] - input.viewPos;
		float dist = length(L);
		L = L / dist;

		float falloff = 40 * lightFalloffQuadratic[i] * dist * dist + lightFalloffConstant;
		float t = saturate(dist / 350.0);
		float cutoff = 1.0 - t * t * t * t;
		float attenuation = (falloff > 0.0) ? (1.0 / falloff) * cutoff : 0.0;
		LightResult pointLR = BRDF(Norm, V, L, texColor.rgb, metalness, roughness, roughness, radius, F0, 0, 1.0);
		float dotpoint = dot(Norm, L);
		float NdotL_point = max(dotpoint, 0.0);

		#ifdef HAS_GRASS
			lambert = dotpoint * -sign(dot(V, Norm));
			lambert = pow(saturate((lambert + w) / (1.0f + w)), n) * (n + 1) / (2 * (1 + w)) + max(0.0, -1.0 * lambert) * GRASS_BACKLIGHTING_COEFF;
			lambert = max(0.0, lambert);
			NdotL_point = 3.14 * lambert;
		#endif

		float3 pointIntensity = 10 * INTENSITY * pow(max(0.0, lightDiffuse[i].rgb) + EPS, 2.2) * NdotL_point * attenuation;
		diffuseLight += pointLR.diffuse * pointIntensity;

		#if defined(HAS_PARAMH) || defined(HAS_NORMAL)
		specularLight += pointLR.specular * pointIntensity;
		#endif
		neglight -= max(0.0, -lightDiffuse[i].r) * 1 / pow(falloff, 1 / 3.2);
	}
	#elif defined(LIGHT_MODE) && LIGHT_MODE == 3
	// Mode 3: Texture-based point light system (>8 lights, rare)
	{
		PointLightResult pointLightResult = evaluatePointLightsPBR(useLightParams, input.viewPos, Norm, V, texColor.rgb, metalness, roughness, radius, F0);
		diffuseLight += pointLightResult.diffuse;

		#if defined(HAS_PARAMH) || defined(HAS_NORMAL)
		specularLight += pointLightResult.specular;
		#endif

		neglight = pointLightResult.neglight;
	}
	#endif
	// LIGHT_MODE==0 or undefined: no point light code

	neglight = max(0.0, 1 - 0.95 * saturate(-neglight));
	diffuseLight *= neglight;
	specularLight *= neglight;

#else
	// Unlit shader - no lighting calculations
	float3 diffuseLight = float3(1.0, 1.0, 1.0);
	float3 specularLight = float3(0.0, 0.0, 0.0);
#endif


	// PBR material calculation following pbr.hlsl pattern
	float4 diffuse;
	int materialMode = (int)shadingMode.z;
	if (materialMode == 2) {
		// Mode 2: Use vertex color for diffuse/ambient
		#ifdef HAS_GRASS
		float3 col = float3(1.0, 1.0, 1.0);
		#else
		float3 col = sqrt(input.color.rgb);
		#endif
		diffuse = float4(col * (diffuseLight + ambient) + useEmissive.rgb * useEmissive.rgb, input.color.a);
	}
	else if (materialMode == 3) {
		// Mode 3: Use vertex color for emissive
		diffuse = float4(useDiffuse.rgb * (diffuseLight + ambient) + input.color.rgb * input.color.rgb, useDiffuse.a);
	}
	else {
		// Mode 1: Use material constants
		diffuse = float4(useDiffuse.rgb * (diffuseLight + ambient) + useEmissive.rgb * useEmissive.rgb, useDiffuse.a);
	}

	// Apply correct PBR formula: (diffuse + ambient) * texture + specular
	float4 c = float4(diffuse.rgb * texColor.rgb + specularLight, diffuse.a * texColor.a);
	
	//c.rgb = diffuseLight.rgb;
	//c = diffuse * float4(1,1,1,texColor.a);
	c.rgb = ToneMap_AgX(c.rgb, 0);
	//c.rgb = 0.16;
	//c.rgb = encode3(c.rgb);

#ifdef HAS_GRASS
	// Alpha test early to improve performance
	c.a = (c.a - 64.0 / 255.0) / max(fwidth(c.a), 0.0001) + 0.5;
#endif

	// shadows DEBUG
	// #ifdef HAS_NORMAL
	// c.rgb = 1.;
	// #endif
	// c.rgb = deb;  // DEBUG: Commented out to restore normal lighting 
	
	// Apply fog --will enable when all rendering goes through HLSL with unified fogging.
	//c.rgb = lerp(fogColNear, c.rgb , saturate(exp(-0.0002 * length(input.viewPos))));
	return c;
}