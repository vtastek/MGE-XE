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

float3 OrenNayarDiffuse(float3 L, float3 V, float3 N, float roughness, float3 kD)
{
	float NdotL = max(dot(N, L), 0.0f);
	float NdotV = max(dot(N, V), 0.0f);
	float3 Vproj = normalize(V - N * NdotV);
	float3 Lproj = normalize(L - N * NdotL);
	float gamma = max(0.0f, dot(Vproj, Lproj));
	float alpha = max(acos(NdotV), acos(NdotL));
	float beta = min(acos(NdotV), acos(NdotL));
	float sigma2 = roughness * roughness;
	float At = 1.0f - 0.5f * sigma2 / (sigma2 + 0.33f);
	float Bt = 0.45f * sigma2 / (sigma2 + 0.09f);
	if (gamma >= 0)
		Bt *= sin(alpha) * clamp(tan(beta), -PI_DIV2, PI_DIV2);
	else
		Bt = 0.0f;
	return (At + Bt) * kD / PI;
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
float4 readLightTexel(int lightIndex, int texelOffset) {
    float u = (lightDataParams.z + lightIndex * 3 + texelOffset + 0.5) * lightDataParams.y;  // texelOffset + local offset
    return tex2Dlod(LightDataSampler, float4(u, 0.5, 0, 0));
}

// Evaluate point lights from texture with full PBR matching legacy path
// viewPos and light positions are in view-space (matching legacy system)
struct PointLightResult {
    float3 diffuse;
    float3 specular;
    float neglight;
};

PointLightResult evaluatePointLightsPBR(float3 viewPos, float3 normal, float3 V, float3 albedo, float metalness, float roughness, float radius, float3 F0) {
    PointLightResult result;
    result.diffuse = float3(0, 0, 0);
    result.specular = float3(0, 0, 0);
    result.neglight = 0.0;

    int numLights = (int)lightDataParams.x;

    for (int i = 0; i < numLights; i++) {
        float4 posRadius = readLightTexel(i, 0);
        float4 color = readLightTexel(i, 1);
        float4 falloff = readLightTexel(i, 2);

        float3 lightPos = posRadius.xyz;
        float3 toLight = lightPos - viewPos;
        float dist = length(toLight);
        float3 L = toLight / dist;

        float falloffValue = 40.0 * falloff.z * dist * dist + falloff.x;
        float t = saturate(dist / 350.0);
        float cutoff = 1.0 - t * t * t * t;
        float attenuation = (falloffValue > 0.0) ? (1.0 / falloffValue) * cutoff : 0.0;

        LightResult pointLR = BRDF(normal, V, L, albedo, metalness, roughness, roughness, radius, F0, 0, 1.0);

        float dotpoint = dot(normal, L);
        float NdotL_point = max(dotpoint, 0.0);

        float3 pointIntensity = 10 * INTENSITY * pow(max(0.0, color.rgb) + EPS, 2.2) * NdotL_point * attenuation;

        result.diffuse += pointLR.diffuse * pointIntensity;
        result.specular += pointLR.specular * pointIntensity;
        result.neglight -= max(0.0, -color.r) * 1 / pow(falloffValue, 1 / 3.2);
    }

    return result;
}

// Legacy simple diffuse version (kept for compatibility)
float3 evaluatePointLights(float3 viewPos, float3 normal) {
    float3 lighting = float3(0, 0, 0);
    int numLights = (int)lightDataParams.x;

    for (int i = 0; i < numLights; i++) {
        float4 posRadius = readLightTexel(i, 0);
        float4 color = readLightTexel(i, 1);
        float4 falloff = readLightTexel(i, 2);

        float3 lightPos = posRadius.xyz;
        float3 toLight = lightPos - viewPos;
        float dist = length(toLight);
        float3 L = toLight / dist;

        float falloffValue = 40.0 * falloff.z * dist * dist + falloff.x;
        float t = saturate(dist / 350.0);
        float cutoff = 1.0 - t * t * t * t;
        float attenuation = (falloffValue > 0.0) ? (1.0 / falloffValue) * cutoff : 0.0;

        float NdotL = max(0, dot(normal, L));
        lighting += color.rgb * attenuation * NdotL;
    }

    return lighting;
}

#endif // USE_TEXTURE_LIGHTS


#endif // LIGHTING_HLSL_INCLUDED