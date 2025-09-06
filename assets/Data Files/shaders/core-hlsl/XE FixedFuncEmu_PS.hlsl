//============================================================================
// MGE XE - HLSL Fixed Function Emulation - Pixel Shader  
//============================================================================

//---------------------------- PBR ----------------------------
#define PI 3.14159
#define PI_DIV2 1.57079632679

#ifdef HAS_GRASS
// Grass lighting constants
#define GRASS_WRAP_LIGHTING_COEFF_W 0.6
#define GRASS_WRAP_LIGHTING_COEFF_N 1.5
#define GRASS_BACKLIGHTING_COEFF 0.4

// Alpha to coverage conversion for grass transparency
float calc_coverage(float alpha, float alphaThreshold, float scale) {
	return saturate((alpha - alphaThreshold) * scale + 0.5);
}
#endif

float saturate(float x) { return max(0.0, min(x, 1.0)); }
float3 saturate(float3 x) { return float3(saturate(x.x), saturate(x.y), saturate(x.z)); }

float minDot = 1e-5;
float dot_c(float3 a, float3 b) { return max(dot(a, b), minDot); }

float2 EnvBRDFApprox(float NoV, float roughness)
{
	float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
	float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
	float4 r = roughness * c0 + c1;
	float2 a004 = min(r.xx * r.xx, exp2(-9.28 * float2(max(0.5, NoV), NoV))) * r.xx + r.yy;
	float2 AB = float2(-1.04, 1.04) * a004 + r.zw;
	return float2(AB.y, AB.x);
}

float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
	float3 ret = float3(0.0, 0.0, 0.0);
	float powTheta = pow(1.0 - cosTheta, 5.0);
	float invRough = 1.0 - roughness;
	ret.x = F0.x + (max(invRough, F0.x) - F0.x) * powTheta;
	ret.y = F0.y + (max(invRough, F0.y) - F0.y) * powTheta;
	ret.z = F0.z + (max(invRough, F0.z) - F0.z) * powTheta;
	return ret;
}

float3 FresnelSchlick(float cosTheta, float3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float DistributionGGX(float NdotH2, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;
	return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;
	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;
	return nom / denom;
}

float GeometrySmith(float3 N, float3 V, float3 L, float roughness)
{
	float NdotV = max(dot(N, V), 0.0);
	float NdotL = max(dot(N, L), 0.0);
	float ggx2 = GeometrySchlickGGX(NdotV, roughness);
	float ggx1 = GeometrySchlickGGX(NdotL, roughness);
	return ggx1 * ggx2;
}

float3 LambertDiffuse(float3 albedo, float3 kD)
{
	return kD * albedo / PI;
}

float3 OrenNayarDiffuse(float3 L, float3 V, float3 N, float roughness, float3 albedo, float3 kD)
{
	float NdotL = max(dot(N, L), 0.0f);
	float NdotV = max(dot(N, V), 0.0f);
	float3 Vproj = normalize(V - N * NdotV);
	float3 Lproj = normalize(L - N * NdotL);
	float gamma = max(0.0f, dot(Vproj, Lproj));
	float alpha = max(acos(NdotV), acos(NdotL));
	float beta = min(acos(NdotV), acos(NdotL));
	float sigma2 = roughness * roughness;
	float A = 1.0f - 0.5f * sigma2 / (sigma2 + 0.33f);
	float B = 0.45f * sigma2 / (sigma2 + 0.09f);
	if (gamma >= 0)
		B *= sin(alpha) * clamp(tan(beta), -PI_DIV2, PI_DIV2);
	else
		B = 0.0f;
	return (A + B) * albedo * kD / PI;
}

float3 BRDF(float3 N, float3 V, float3 L, float3 albedo, float metalness, float roughness, float roughnessPrime, float radius, float3 F0, int isOrenNayar)
{
	float3 H = normalize(V + L);
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;
	float radiusTan = roughnessPrime;
	// roughnessPrime = EvaluateNormalizationFactor(roughness, dot(L, H), radius); // skip for now
	roughnessPrime = clamp(roughnessPrime, 0.05, 0.999);
	// NdotH2 = GetNoHSquared(radiusTan, dot(N, L), dot(N, V), dot(V, L)); // skip for now
	float NDF = DistributionGGX(NdotH2, roughnessPrime);
	float G = GeometrySmith(N, V, L, roughness);
	float3 F = FresnelSchlick(max(dot(V, H), 0.0), F0);
	float3 kS = F;
	float3 kD = (float3(1.0, 1.0, 1.0) - kS) * (1.0 - metalness);
	float amask = step(0.05, dot(albedo, 0.33));
	float3 Is = NDF * G * F * amask;
	float3 Id = isOrenNayar != 0 ? OrenNayarDiffuse(L, V, N, roughness, albedo, kD) : LambertDiffuse(albedo, kD);
	return Id + Is;
}

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
sampler sampShadow : register(s5) = sampler_state{ texture = <tex5>; }; // Shadow map

// Shadow constants
static const int shadowCascades = 2;
static const float shadowCascadeSize = 1.0 / shadowCascades;
static const float ESM_c = 60.0;
static const float ESM_bias = 2e-3 * ESM_c;
static const float ESM_scale = 32768.0;

// Shadow resolution parameter (will be set via shader constants)
float shadowRcpRes : register(c10);
#endif

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

//------------------------------------------------------------
// Tone Mapping Functions
// Primary crosstalk (dye-layer ?seen? mixing). Keep diagonal-dominant.

//   https://github.com/sobotka/AgX

float3 ToneMap_AgX(float3 linCol, int lookMode)
{
	// Minimal AgX, see by https://iolite-engine.com/blog_posts/minimal_agx_implementation

	float3x3 agx_mat = float3x3(
		0.842479062253094, 0.0423282422610123, 0.0423756549057051,
		0.0784335999999992, 0.878468636469772, 0.0784336,
		0.0792237451477643, 0.0791661274605434, 0.879142973793104
	);

	float3x3 agx_mat_inv = float3x3(
		1.19687900512017, -0.0528968517574562, -0.0529716355144438,
		-0.0980208811401368, 1.15190312990417, -0.0980434501171241,
		-0.0990297440797205, -0.0989611768448433, 1.15107367264116
	);

	float min_ev = -12.47393;
	float max_ev = 4.026069;
	float bias = 1.0;

	// Input transform
	float3 val = mul((linCol * bias), agx_mat);

	// Log2 space encoding
	val = clamp(log2(val), min_ev, max_ev);
	val = (val - min_ev) / (max_ev - min_ev);

	// Apply sigmoid function approximation
	val = ((((((((((15.5 * val) - 40.14) * val) + 31.96) * val) - 6.868) * val) + 0.4298) * val) + 0.1191) * val - 0.00232;

	// Apply Look Transform
	float luma = dot(val, float3(0.2126, 0.7152, 0.0722));
	float3 offset = (0.0);
	float3 slope = (1.0);
	float3 power = (1.0);
	float sat = 1.0;

	if (lookMode == 1) // "Golden"
	{
		slope = float3(1.0, 0.9, 0.5);
		power = (0.8);
		sat = 0.8;
	}
	else if (lookMode == 2) // "Punchy"
	{
		slope = (1.0);
		power = (1.35);
		sat = 1.4;
	}

	val = pow(val * slope + offset, power);
	val = luma + sat * (val - luma);

	// Inverse Input transform
	return mul(val, agx_mat_inv);
}

static const float3x3 CROSSTALK_MATRIX = float3x3(
	1.0, 0.05, 0.05,
	0.05, 1.0, 0.57,
	0.05, 0.05, 1.0
);

float3x3 Balanced(float3x3 M)
{
	float3 resp = mul(M, float3(1, 1, 1));
	float3 fix = 1.0 / max(resp, 1e-6.xxx);
	float3x3 WB = float3x3(
		fix.x, 0, 0,
		0, fix.y, 0,
		0, 0, fix.z
	);
	return mul(WB, M);
}

float3 PBRNeutralToneMapping(float3 color) {
	float startCompression = 0.8 - 0.04;
	float desaturation = 0.15;

	color = max(0.0, color);

	float x = min(color.r, min(color.g, color.b));
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;

	float peak = max(color.r, max(color.g, color.b));
	if (peak < startCompression) return color;

	float d = 1. - startCompression;
	float newPeak = 1. - d * d / (peak + d - startCompression);
	color *= newPeak / peak;

	float g = 1. - 1. / (desaturation * (peak - newPeak) + 1.0);
	float3 toneMapped = lerp(color, newPeak, g);

	// Apply balanced crosstalk to maintain neutral gray
	float3x3 balancedCrosstalk = Balanced(CROSSTALK_MATRIX);
	return mul(balancedCrosstalk, toneMapped);
}


#define EPS 1.17549435e-38f

#define GM 1.6
#define LINEAR_END 0.2
#define INPUT_MAX 16.0
#define OUTPUT_MAX 1.0
#define A (20.0 / 21.0)  // Rational function parameter for smooth transition


float encode(float x) {


	if (x <= LINEAR_END) {
		// Linear part: f(x) = x
		return x;
	}
	else if (x >= INPUT_MAX) {
		// Clamp to maximum
		return OUTPUT_MAX;
	}
	else {
		// Smooth compression using rational function
		float t = (x - LINEAR_END) / (INPUT_MAX - LINEAR_END);
		float h = t / (A * t + (1.0 - A));
		return LINEAR_END + (OUTPUT_MAX - LINEAR_END) * h;
	}
}

float decode(float y) {


	if (y <= LINEAR_END) {
		// Linear part: inverse of f(x) = x
		return y;
	}
	else if (y >= OUTPUT_MAX) {
		// Return maximum input value
		return INPUT_MAX;
	}
	else {
		// Inverse of rational function
		float y_norm = (y - LINEAR_END) / (OUTPUT_MAX - LINEAR_END);
		float t = y_norm * (1.0 - A) / (1.0 - y_norm * A);
		return LINEAR_END + t * (INPUT_MAX - LINEAR_END);
	}
}

// Vector versions for RGB
float3 encode3(float3 rgb) {
	return pow(saturate(float3(encode(rgb.r), encode(rgb.g), encode(rgb.b))), 1.0 / GM);
}

float3 decode3(float3 encoded) {
	float3 enc = pow(encoded, GM);
	return float3(decode(enc.r), decode(enc.g), decode(enc.b));
}



// --------------------------------------------------------------------------
// -----------------FILMIC PROCESS ------------------------------------------



//------------------------------------------------------------
// Normal Mapping and Parallax Functions

// Input: view-space position of the pixel
// Output: approximate normal in view space
float3 GetSafeNormal(float3 viewPos, float3 inputNormal)
{
	float3 dpdx = ddx(viewPos);
	float3 dpdy = ddy(viewPos);
	float3 pseudoNormal = normalize(cross(dpdx, dpdy));

	// Blend factor based on original normal length
	float factor = saturate((0.92 - length(inputNormal)) / 0.04);
	// 0.0 = normal ok, 1.0 = normal broken

	return normalize(lerp(inputNormal, pseudoNormal, factor));
}



// Per-pixel TBN from screen-space derivatives (view space)
void BuildPerPixelTBN(
	float3 normalVS,
	float3 viewPos,
	float2 uv,
	out float3 T,
	out float3 B,
	out float3 N)
{
	N = normalize(normalVS);

	float3 dpdx = ddx(-viewPos);
	float3 dpdy = ddy(-viewPos);
	float2 dtdx = ddx(uv);
	float2 dtdy = ddy(uv);

	float det = dtdx.x * dtdy.y - dtdx.y * dtdy.x;
	float invDet = (abs(det) > 1e-8) ? (1.0 / det) : 0.0;

	T = normalize((dpdx * dtdy.y - dpdy * dtdx.y) * invDet);
	B = normalize((-dpdx * dtdy.x + dpdy * dtdx.x) * invDet);
}

#ifdef HAS_NORMAL
// Parallax height and normal parameters
static const float parallaxScale = 10.2;
static const float parallaxBias = 0.00005;
static const float heightScale = -4;

#ifdef USE_SIMPLE_PARALLAX
// Simple offset parallax mapping (no raymarch, just single offset)
float2 ParallaxSimple(
	sampler2D hmap,
	float2 uv,
	float3 Vts,
	float heightScale)
{
	float h = 2 * (1 - tex2D(hmap, uv).a) - 1;
	float2 offset = (h * heightScale) * (-Vts.xy / max(Vts.z, 1e-3));
	return uv + offset;
}
#endif

// Soft parallax shadowing for parallax-mapped surfaces
float ParallaxSoftShadow(
	sampler2D hmap,
	float2 uv,
	float2 lightDirTS,
	float soften,
	float scale)
{
	float h0 = 1.0 - tex2D(hmap, uv).a;
	float h = h0;
	float2 lDir = -lightDirTS * scale;
	h = min(1.0, 1.0 - tex2D(hmap, uv + 0.20 * lDir).a);
	h = min(h, 1.0 - tex2D(hmap, uv + 0.35 * lDir).a);
	h = min(h, 1.0 - tex2D(hmap, uv + 0.45 * lDir).a);
	h = min(h, 1.0 - tex2D(hmap, uv + 0.55 * lDir).a);
	float shadowpara = min(1.0, 1.0 - saturate((h0 - h) * soften));
	return shadowpara;
}


#endif

#if defined(HAS_SHADOWS)
//------------------------------------------------------------
// Shadow Sampling Functions

// Shadow UV to shadow atlas UV
float4 mapShadowToAtlas(float2 t, int layer) {
	return float4(t.x * shadowCascadeSize + layer * shadowCascadeSize, t.y, 0, 0);
}

// 2 layer cascade ortho ESM lookup
float shadowSample(float4 shadow0pos, float4 shadow1pos) {
	// Clip space margin of 4 texels, to prevent bleeding from the filter kernel + adjacent textures  
	float3 atlasMargin = float3(1.0 - 2.0 * 4.0 * shadowRcpRes, 1.0 - 2.0 * 4.0 * shadowRcpRes, 1.0);

	float dz = 1e-6;

	if (all(saturate(atlasMargin - abs(shadow0pos.xyz)))) {
		// Layer 0, inner (near shadows)
		float2 shadowUV = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow0pos.xy;
		dz = tex2Dlod(sampShadow, mapShadowToAtlas(shadowUV, 0)).r / ESM_scale - shadow0pos.z;
	}
	else if (all(saturate(atlasMargin - abs(shadow1pos.xyz)))) {
		// Layer 1 (far shadows)
		float2 shadowUV = (0.5 + 0.5 * shadowRcpRes) + float2(0.5, -0.5) * shadow1pos.xy;
		dz = tex2Dlod(sampShadow, mapShadowToAtlas(shadowUV, 1)).r / ESM_scale - shadow1pos.z;
	}

	// ESM shadow filtering  
	return 1.0 - saturate(exp(ESM_c * dz + ESM_bias));
}
#endif

//------------------------------------------------------------
// Pixel Shader Main
float4 ps_main(VS_OUTPUT input) : COLOR{
	// Enhanced texture sampling with suffix support
	float3 diffuseParam = float3(1.0, 1.0, 1.0);  // Default white
	float3 deb = 0;
	float2 parallaxUV = input.texcoord;
	float3 normalVS = normalize(input.normal);
	#ifdef HAS_SHADOWS
	// Sample shadow map using cascaded ESM
	float shadowpara = 1- shadowSample(input.shadow0pos, input.shadow1pos);
	deb = shadowpara;
#else
	float shadowpara = 1.0; // No shadows
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
				//shadowpara *= ParallaxSoftShadow(sampTex3, parallaxUV, lightDirTS.xy, 5.0, 0.04 * 0.75);
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
	float3 ambient = 0 * pow(lightSceneAmbient + EPS, 2.2) / PI;

	// Sun light (Oren-Nayar)
	float3 Lsun = normalize(-lightSunDirection);
	float sunAtten = shadowpara;
	float3 sunBRDF = BRDF(Norm, V, Lsun, texColor.rgb, metalness, roughness, roughness, radius, F0, 1);
	//deb = sunBRDF;
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

	  float3 lighting = 18 * pow(lightSunDiffuse + EPS, 2.2) * sunBRDF * sunAtten * NdotL_sun;

		  float neglight = 0.0;
		  #ifndef NO_POINT_LIGHTS
		  // Point lights (Lambert)
		  for (int i = 0; i < pointLightCount; i++) {
			  float3 L = lightPosition[i] - input.viewPos;
			  float dist = length(L);
			  L = L / dist;

			  float falloff = lightFalloffQuadratic[i] * dist * dist + lightFalloffConstant;
			  float t = saturate(dist / 350.0);
			  float cutoff = 1.0 - t * t * t * t;
			  float attenuation = (falloff > 0.0) ? (1.0 / falloff) * cutoff : 0.0;
			  float3 pointBRDF = BRDF(Norm, V, L, texColor.rgb, metalness, roughness, roughness, radius, F0, 0);
			  float dotpoint = dot(Norm, L);
			  float NdotL_point = max(dotpoint, 0.0);
			  #ifdef HAS_GRASS
			  // Apply grass-specific wrap lighting for two-sided grass rendering
			  // if (input.color.r > 0.5) {

					lambert = dotpoint * -sign(dot(V, Norm));
					lambert = pow(saturate((lambert + w) / (1.0f + w)), n) * (n + 1) / (2 * (1 + w)) + max(0.0, -1.0 * lambert) * GRASS_BACKLIGHTING_COEFF;
					lambert = max(0.0, lambert);
					NdotL_point = 3.14 * lambert;
					deb = NdotL_point;

					// }
						#endif

						lighting += 18 * (pow(max(0.0, lightDiffuse[i].rgb) + EPS, 2.2) * pointBRDF * NdotL_point) * attenuation;
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
		effectiveEmissive = materialEmissive.rgb * materialEmissive.rgb;
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
c.rgb = ToneMap_AgX(c.rgb, 0);
//c.rgb = encode3(c.rgb);

#ifdef HAS_GRASS
	// Alpha test early to improve performance
	c.a = (c.a - 64.0 / 255.0) / max(fwidth(c.a), 0.0001) + 0.5;
#endif

	// shadows DEBUG
	c.rgb = deb;

	// Apply fog
	//c.rgb = lerp(fogColNear, c.rgb, input.fog);
	return c;
}