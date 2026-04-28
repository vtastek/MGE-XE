//============================================================================
// MGE XE - Lighting System HLSL Include  
// Provides BRDF functions, material calculations, and lighting models
//============================================================================

#ifndef LIGHTING_HLSL_INCLUDED
#define LIGHTING_HLSL_INCLUDED

#include "common.hlsl"

float minDot = 1e-5;
float dot_c(float3 a, float3 b) { return max(dot(a, b), minDot); }

// Reflected light result
struct LightResult
{
    float3 ambient, diffuse, specular;
};

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

float3 LambertDiffuse(float3 kD)
{
	return kD / PI;
}

// EON (Energy-conserving Oren-Nayar) constants
static const float constant1_FON = 0.5f * (PI - 1.0f);  // ~1.0708
static const float constant2_FON = 0.25f * (PI - 1.0f); // ~0.5354

// FON directional albedo approximation
float E_FON_approx2(float mu, float r)
{
	// Polynomial fit for directional albedo
	float r2 = r * r;
	return 1.0f - r * (0.5f - 0.5f * mu) - r2 * (0.25f - 0.75f * mu + 0.5f * mu * mu);
}

// Energy-conserving Oren-Nayar diffuse (EON) - much faster than classic ON
float3 OrenNayarDiffuse(float3 L, float3 V, float3 N, float roughness, float3 rho)
{
	float NdotL = max(dot(N, L), 0.0f);
	float NdotV = max(dot(N, V), 0.0f);
	float LdotV = dot(L, V);

	float s = LdotV - NdotL * NdotV;  // QON s term
	float sovertF = s > 0.0f ? s / max(NdotV, NdotL) : s;  // FON s/t
	float AF = 1.0f / (1.0f + constant1_FON * roughness);  // FON A coeff
	float3 f_ss = rho * AF * (1.0f + roughness * sovertF);  // single-scatter

	float EFo = E_FON_approx2(NdotV, roughness) * AF;
	float EFi = E_FON_approx2(NdotL, roughness) * AF;

	float avgEF = AF * (1.0f + constant2_FON * roughness);
	float3 rho_ms = (rho * rho) / (1.0f - rho * (1.0f - avgEF));

	// Multi-scatter lobe
	const float eps = 1.0e-7f;
	float3 f_ms = rho_ms * (avgEF * (1.0f - EFo) * (1.0f - EFi) / max(eps, 1.0f - avgEF));

	return (f_ss + f_ms) / PI;
}

LightResult BRDF(float3 N, float3 V, float3 L, float3 albedo, float metalness, float roughness, float roughnessPrime, float radius, float3 F0, int isOrenNayar, float shadows)
{
	LightResult lr;
	lr.ambient = float3(0, 0, 0);  // No ambient in BRDF

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
	//float amask = step(0.05, dot(albedo, 0.33));

	// Separate diffuse and specular
	lr.diffuse = isOrenNayar != 0 ? OrenNayarDiffuse(L, V, N, roughness, kD) : LambertDiffuse(kD);
	lr.specular = NDF * G * F; //* amask * max(0.0, shadows - 0.09);

	return lr;
}

//============================================================================
// Texture-Based Point Light System (only compiled for LIGHT_MODE 3)
//============================================================================
#ifdef USE_TEXTURE_LIGHTS

// Light data texture (3 texels per light)
// Texel 0: [posX, posY, posZ, radius]
// Texel 1: [colorR, colorG, colorB, unused]
// Texel 2: [falloffConstant, falloffLinear, falloffQuadratic, unused]
sampler LightDataSampler : register(s5) = sampler_state {
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = NONE;
    AddressU = CLAMP;
    AddressV = CLAMP;
};

// Light system parameters: (numLights, texelSize, texelOffset, unused)
// texelOffset = per-object starting texel in packed texture
float4 lightDataParams : register(c50);

// Read light data from texture
// params: {numLights, texelSize, texelOffset, unused}
float4 readLightTexel(float4 params, int lightIndex, int texelOffset) {
    float u = (params.z + lightIndex * 3 + texelOffset + 0.5) * params.y;
    return tex2Dlod(LightDataSampler, float4(u, 0.5, 0, 0));
}

// Evaluate point lights from texture with full PBR matching legacy path
// viewPos and light positions are in view-space (matching legacy system)
struct PointLightResult {
    float3 diffuse;
    float3 specular;
    float neglight;
};

PointLightResult evaluatePointLightsPBR(float4 lightParams, float3 viewPos, float3 normal, float3 V, float3 albedo, float metalness, float roughness, float radius, float3 F0) {
    PointLightResult result;
    result.diffuse = float3(0, 0, 0);
    result.specular = float3(0, 0, 0);
    result.neglight = 0.0;

    int numLights = (int)lightParams.x;

    for (int i = 0; i < numLights; i++) {
        float4 posRadius = readLightTexel(lightParams, i, 0);
        float4 color = readLightTexel(lightParams, i, 1);
        float4 falloff = readLightTexel(lightParams, i, 2);

        float3 lightPos = posRadius.xyz;
        float3 toLight = lightPos - viewPos;
        float dist = length(toLight);
        float3 L = toLight / dist;

        float attenuation = pointLightAttenuation(dist);

        LightResult pointLR = BRDF(normal, V, L, albedo, metalness, roughness, roughness, radius, F0, 0, 1.0);

        float dotpoint = dot(normal, L);
        float NdotL_point = max(dotpoint, 0.0);

        float3 pointIntensity = INTENSITY * pow(max(0.0, color.rgb) + EPS, 2.2) * NdotL_point * attenuation;

        result.diffuse += pointLR.diffuse * pointIntensity;
        result.specular += pointLR.specular * pointIntensity;
        float dist2 = max(dist * dist, 1.0);
        result.neglight -= max(0.0, -color.r) * 1 / pow(dist2, 1 / 3.2);
    }

    return result;
}

// Legacy simple diffuse version (kept for compatibility)
float3 evaluatePointLights(float4 lightParams, float3 viewPos, float3 normal) {
    float3 lighting = float3(0, 0, 0);
    int numLights = (int)lightParams.x;

    for (int i = 0; i < numLights; i++) {
        float4 posRadius = readLightTexel(lightParams, i, 0);
        float4 color = readLightTexel(lightParams, i, 1);
        float4 falloff = readLightTexel(lightParams, i, 2);

        float3 lightPos = posRadius.xyz;
        float3 toLight = lightPos - viewPos;
        float dist = length(toLight);
        float3 L = toLight / dist;

        float attenuation = pointLightAttenuation(dist);

        float NdotL = max(0, dot(normal, L));
        lighting += color.rgb * attenuation * NdotL;
    }

    return lighting;
}

#endif // USE_TEXTURE_LIGHTS


#endif // LIGHTING_HLSL_INCLUDED