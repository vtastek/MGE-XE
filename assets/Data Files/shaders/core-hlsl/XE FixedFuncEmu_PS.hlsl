//============================================================================
// MGE XE - HLSL Fixed Function Emulation - Pixel Shader  
//============================================================================

//------------------------------------------------------------
// Shared Variables

// Materials
float4 materialDiffuse, materialAmbient, materialEmissive;
float4 shadingMode; // .z = materialMode (1=none, 2=diffamb, 3=emissive)

#ifndef NOLIT
// Lighting - Basic
float3 lightSceneAmbient;
float3 lightSunDiffuse;
float3 lightSunDirection;

#ifndef NO_POINT_LIGHTS
// Lighting - Point lights
float4 lightDiffuse[8];  // Changed to float4 to match D3DXVECTOR4 from C++
float3 lightPosition[8]; // Proper float3 positions
float lightAmbient[8];
float lightFalloffQuadratic[8];
float lightFalloffConstant;
int pointLightCount; // Number of real point lights
#endif
#endif

// Fog
float3 fogColNear;

// Textures with explicit register bindings for DX9 HLSL
texture tex0 : register(t0);
texture tex1 : register(t1);
#if defined(HAS_DIFFPARAM)
texture tex2 : register(t2);
#endif
#if defined(HAS_NORMAL)
texture tex3 : register(t3);
#endif
#if defined(HAS_PARAM)
texture tex4 : register(t4);
#endif
#if defined(HAS_SHADOWS)
texture tex5 : register(t5);
#endif
sampler sampTex0 : register(s0) = sampler_state{ texture = <tex0>; };
sampler sampTex1 : register(s1) = sampler_state{ texture = <tex1>; };
#if defined(HAS_DIFFPARAM)
sampler sampTex2 : register(s2) = sampler_state{ texture = <tex2>; }; // Diffuse parameter (_diffparam)  
#endif
#if defined(HAS_NORMAL)
sampler sampTex3 : register(s3) = sampler_state{ texture = <tex3>; }; // Normal map (_nh)
#endif
#if defined(HAS_PARAM)
sampler sampTex4 : register(s4) = sampler_state{ texture = <tex4>; }; // Param map (_param)
#endif
#if defined(HAS_SHADOWS)
sampler sampShadow : register(s5) = sampler_state{ texture = <tex5>; addressu = border; addressv = border; bordercolor = 0xffffffff; minfilter = linear; magfilter = linear; }; // Shadow map



// Shadow resolution parameter (will be set via shader constants)
float shadowRcpRes : register(c10);
#endif

// Alpha testing/blending flag
bool hasAlpha;

// Texture suffix support - using preprocessor defines
//#define HAS_DIFFPARAM
//#define HAS_NORMAL
//#define HAS_PARAM

#define USE_SIMPLE_PARALLAX // Enable this for simple offset parallax mapping
#ifdef HAS_NORMAL
float2 normres;  // Normal map texture resolution (width, height)
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
};


#include "common.hlsl"
#include "lighting.hlsl"
#include "shadows.hlsl"

//------------------------------------------------------------
// Pixel Shader Main
float4 ps_main(VS_OUTPUT input) : COLOR{
	// Enhanced texture sampling with suffix support
	float3 diffuseParam = float3(1.0, 1.0, 1.0);  // Default white
	float3 deb = 0;
	float2 parallaxUV = input.texcoord;
	float3 normalVS = normalize(input.normal);

	#ifdef HAS_SHADOWS
	float ndotlgeo = 1;
	#ifndef NOLIT
	ndotlgeo = dot(normalVS, -normalize(lightSunDirection));
	#endif
	
	float shadows = shadowSample(input.shadow0pos, input.shadow1pos, ndotlgeo, hasAlpha ? 1.0 : 0.0);
	float3 atlasMargin = float3(1.0 - 2.0 * 4.0 * shadowRcpRes, 1.0 - 2.0 * 4.0 * shadowRcpRes, 1.0);
    float3 blendMargin = float3(1.0 - 2.0 * 264.0 * shadowRcpRes, 1.0 - 2.0 * 264.0 * shadowRcpRes, 1.0); // 50% wider blend zone
    
    bool inNear = all(saturate(atlasMargin - abs(input.shadow0pos.xyz)));
    bool inFar = all(saturate(atlasMargin - abs(input.shadow1pos.xyz)));
	deb.y = float(inNear);
	deb.xz = float(inFar);
	
	float shadowpara = 1.0;
	//deb = shadows * ndotlgeo;
#else
	float shadows = 1.0; // No shadows - fully lit
	float shadowpara = 1.0;
#endif

	#ifdef HAS_NORMAL
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

	#ifdef HAS_NORMAL
		#if defined(USE_SIMPLE_PARALLAX)
		parallaxUV = ParallaxSimple(sampTex3, parallaxUV, Vts, 0.000015 * 1.33); // heightScale
		#endif

		// After parallax, sample normal as usual
		float2 texel = 1.0 / normres;
		float hL = tex2D(sampTex3, parallaxUV + float2(-texel.x, 0)).a;
		float hR = tex2D(sampTex3, parallaxUV + float2(texel.x, 0)).a;
		float hD = tex2D(sampTex3, parallaxUV + float2(0, -texel.y)).a;
		float hU = tex2D(sampTex3, parallaxUV + float2(0,  texel.y)).a;
		float dhdu = (hR - hL);
		float dhdv = (hU - hD);
		float3 nTS = normalize(float3(-dhdu * heightScale, dhdv * heightScale, 1.0));

		normalVS = normalize(mul(nTS, TBN));
		// Soft parallax shadowing (only if lighting is enabled)
			#ifndef NOLIT
				#ifdef HAS_NORMAL
				float3 lightDirVS = -lightSunDirection;
				float3 lightDirTS = mul(lightDirVS, TBN_T);
				shadowpara = ParallaxSoftShadow(sampTex3, input.texcoord, lightDirTS.xy, 5.0, 0.04 * 0.75);
				shadows *= shadowpara;
				#endif
			#endif
		#else
	//float3 nTS = normalize(float3(0,0,1));
	normalVS = inputVS;
	#endif


	float4 texColor = tex2D(sampTex0, parallaxUV);
	texColor.rgb = max(0.008, pow(texColor.rgb + EPS, 2.2));
	// Note: When HAS_DIFFPARAM is defined, sampTex0 contains the _diffparam/_diffparam_t texture
	#ifdef HAS_DIFFPARAM
	texColor.a = 1.0; // Ignore alpha from _diffparam texture
	#endif

	// PBR lighting calculation
	float3 V = normalize(-input.viewPos); // view direction in view space
	float3 Norm = normalVS;
	float3 albedo = texColor.rgb;
	float roughness = 0.9;
	float metalness = 0.0;
	float ao = 1.0;
	float radius = 1.6;
	float3 F0 = 0.08 * float3(0.5, 0.5, 0.5); // specular reflectance

#ifdef HAS_PARAM
	#ifdef HAS_NORMAL
	float4 param = tex2D(sampTex4, parallaxUV);
	metalness = param.x;
	roughness = param.y * param.y;
	F0 = 0.08 * param.z * param.z;
	ao = param.w;
	#endif
#endif
#ifdef HAS_DIFFPARAM
	// Basic PBR mode: _diffparam texture with fixed material properties
	// RGB = albedo, A = roughness. Fixed: metalness=0, specular=0.5, AO=1
	// Sample diffparam alpha directly for roughness (before texColor.a was overwritten with original alpha)
	float diffparamAlpha = tex2D(sampTex0, parallaxUV).a;
	roughness = diffparamAlpha * diffparamAlpha;
	metalness = 0.0;         // Non-metallic materials
	F0 = 0.08 * 0.5;         // Fixed specular reflectance = 0.5
	ao = 1.0;                // Full ambient occlusion
#endif

#ifndef NOLIT
	float3 ambient = INTENSITY * pow(lightSceneAmbient + EPS, 2.2) / PI;

	// Sun light (Oren-Nayar)
	float3 Lsun = normalize(-lightSunDirection);
	float sunAtten = shadows;
	float3 sunBRDF = BRDF(Norm, V, Lsun, texColor.rgb, metalness, roughness, roughness, radius, F0, 1, shadows);
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

	float3 lighting = shadowpara * INTENSITY * pow(lightSunDiffuse + EPS, 2.2) * sunBRDF * sunAtten * NdotL_sun;
	//deb = shadowpara;
	float neglight = 0.0;
	#ifndef NO_POINT_LIGHTS
	// Point lights (Lambert)
	for (int i = 0; i < pointLightCount; i++) 
	{
		float3 L = lightPosition[i] - input.viewPos;
		float dist = length(L);
		L = L / dist;

		float falloff = lightFalloffQuadratic[i] * dist * dist + lightFalloffConstant;
		float t = saturate(dist / 350.0);
		float cutoff = 1.0 - t * t * t * t;
		float attenuation = (falloff > 0.0) ? (1.0 / falloff) * cutoff : 0.0;
		float3 pointBRDF = BRDF(Norm, V, L, texColor.rgb, metalness, roughness, roughness, radius, F0, 0, 1.0);
		float dotpoint = dot(Norm, L);
		float NdotL_point = max(dotpoint, 0.0);
		
		#ifdef HAS_GRASS
			// Apply grass-specific wrap lighting for two-sided grass rendering
		    // if (input.color.r > 0.5) {

			lambert = dotpoint * -sign(dot(V, Norm));
			lambert = pow(saturate((lambert + w) / (1.0f + w)), n) * (n + 1) / (2 * (1 + w)) + max(0.0, -1.0 * lambert) * GRASS_BACKLIGHTING_COEFF;
			lambert = max(0.0, lambert);
			NdotL_point = 3.14 * lambert;
			//deb = NdotL_point;

			// }
		#endif

		lighting += INTENSITY * (pow(max(0.0, lightDiffuse[i].rgb) + EPS, 2.2) * pointBRDF * NdotL_point) * attenuation;
		neglight -= max(0.0, -lightDiffuse[i].r) * 1 / pow(falloff, 1 / 3.2);

		//deb += attenuation;
	}
	#endif

	lighting += ambient;
	neglight = max(0.0, 1 - 0.95 * saturate(-neglight));
	lighting *= neglight;
#else
	// Unlit shader - no lighting calculations
	float3 lighting = float3(1.0, 1.0, 1.0);
#endif


	// Material calculation with diffuse parameter modulation
	float3 effectiveDiffuse;
	float3 effectiveEmissive;
	float effectiveAlpha;

	int materialMode = (int)shadingMode.z;
	if (materialMode == 2) {
		// Mode 2: Use vertex color for diffuse/ambient
		#ifdef HAS_GRASS
		effectiveDiffuse = 1.0;
		#else
		effectiveDiffuse = sqrt(input.color.rgb);
		#endif
		effectiveEmissive = materialEmissive.rgb * materialEmissive.rgb * 5;
		effectiveAlpha = input.color.a;
	}
	 else if (materialMode == 3)
	{
		// Mode 3: Use vertex color for emissive
		effectiveDiffuse = materialDiffuse.rgb;
		effectiveEmissive = input.color.rgb * input.color.rgb;
		effectiveAlpha = materialDiffuse.a;
	}
	else
	{
		// Mode 1: Use material constants
		effectiveDiffuse = materialDiffuse.rgb;
		effectiveEmissive = materialEmissive.rgb * materialEmissive.rgb;
		effectiveAlpha = materialDiffuse.a;
	}

	float3 litColor = effectiveDiffuse * lighting;
	litColor += effectiveEmissive;

	float4 diffuse = float4(litColor, effectiveAlpha);

	// Apply base texture with enhanced material properties
	float4 c = diffuse * texColor;
	//c = diffuse * float4(1,1,1,texColor.a);
	//c.rgb = ToneMap_AgX(c.rgb, 0);
	//c.rgb = 0.16;
	c.rgb = encode3(c.rgb);

#ifdef HAS_GRASS
	// Alpha test early to improve performance
	c.a = (c.a - 64.0 / 255.0) / max(fwidth(c.a), 0.0001) + 0.5;
#endif

	// shadows DEBUG
	// #ifdef HAS_NORMAL
	// c.rgb = 1.;
	// #endif
	//c.rgb = deb; 
	
	// Apply fog --will enable when all rendering goes through HLSL with unified fogging.
	//c.rgb = lerp(fogColNear, c.rgb , saturate(exp(-0.0002 * length(input.viewPos))));
	return c;
}