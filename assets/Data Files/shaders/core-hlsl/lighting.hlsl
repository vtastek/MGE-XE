//============================================================================
// MGE XE - Lighting System HLSL Include  
// Provides BRDF functions, material calculations, and lighting models
//============================================================================

#ifndef LIGHTING_HLSL_INCLUDED
#define LIGHTING_HLSL_INCLUDED

#include "common.hlsl"

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

float3 BRDF(float3 N, float3 V, float3 L, float3 albedo, float metalness, float roughness, float roughnessPrime, float radius, float3 F0, int isOrenNayar, float shadows)
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
	return Id + Is * max(0.0,shadows-0.09);
}

#endif // LIGHTING_HLSL_INCLUDED