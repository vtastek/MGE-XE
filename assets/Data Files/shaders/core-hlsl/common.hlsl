//============================================================================
// MGE XE - Common HLSL Constants and Utilities
// This file can be included by other HLSL shaders using #include "common.hlsl"
//============================================================================

#ifndef COMMON_HLSL_INCLUDED
#define COMMON_HLSL_INCLUDED

// Mathematical constants
#define PI 3.14159
#define PI_DIV2 1.57079632679
#define INTENSITY 10.0

// Luminance calculation (ITU-R BT.709)
float luminance(float3 color) {
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}


float3 toLinear(float3 c)
{
    float3 low = c / 12.92;
    float3 high = pow((c + 0.055) / 1.055, 2.4);
    return (c < 0.04045) ? low : high;
    
}

// Gamma correction
float3 linearToGamma(float3 color) {
    return pow(abs(color), 1.0/2.2);
}

float3 gammaToLinear(float3 color) {
    return pow(abs(color), 2.2);
}

// Common texture sampling helper
float4 sampleWithFallback(sampler2D samp, float2 uv, float4 fallback) {
    return tex2D(samp, uv);
}

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
        slope = (1.1);
        power = (1.0);
        sat = 1.3;
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
#define Ax (20.0 / 21.0)  // Rational function parameter for smooth transition

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
		float h = t / (Ax * t + (1.0 - Ax));
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
		float t = y_norm * (1.0 - Ax) / (1.0 - y_norm * Ax);
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

#if defined(HAS_NORMAL) || defined(HAS_PARAMH)
// Parallax height and normal parameters
static const float parallaxScale = 10.2;
static const float parallaxBias = 0.00005;
static const float heightScale = -16;

#ifdef USE_PARALLAX
// Simple offset parallax mapping (no raymarch, just single offset)
// Added channel parameter: 0=r, 1=g, 2=b, 3=a
float2 Parallax(
	sampler2D hmap,
	float2 uv,
	float3 Vts,
	float hs,
	int channel = 3)  // Default to alpha channel for backward compatibility
{
	float4 hmapSample = tex2D(hmap, uv);
	float h;
	if (channel == 0) h = hmapSample.r;
	else if (channel == 1) h = hmapSample.g;
	else if (channel == 2) h = hmapSample.b;
	else h = hmapSample.a;  // _paramh (DXT5) and legacy _nh both store height in alpha

	h = 2 * (1 - h) - 1;
	float2 offset = (h * hs) * (-Vts.xy / max(Vts.z, 1e-3));
	return uv + offset;
}
#endif

// Soft parallax shadowing for parallax-mapped surfaces
// Added channel parameter: 0=r, 1=g, 2=b, 3=a
float ParallaxSoftShadow(
	sampler2D hmap,
	float2 uv,
	float2 lightDirTS,
	float soften,
	float scale,
	int channel = 3)  // Default to alpha channel for backward compatibility
{
	float h0, h;
	float4 sample0 = tex2D(hmap, uv);
	if (channel == 0) h0 = 1.0 - sample0.r;
	else if (channel == 1) h0 = 1.0 - sample0.g;
	else if (channel == 2) h0 = 1.0 - sample0.b;
	else h0 = 1.0 - sample0.a;  // _paramh (DXT5) and legacy _nh both store height in alpha

	h = h0;
	float2 lDir = -lightDirTS * scale;

	// Sample height at multiple points along light direction
	float4 sample1 = tex2D(hmap, uv + 0.20 * lDir);
	float4 sample2 = tex2D(hmap, uv + 0.35 * lDir);
	float4 sample3 = tex2D(hmap, uv + 0.45 * lDir);
	float4 sample4 = tex2D(hmap, uv + 0.55 * lDir);

	if (channel == 0) {
		h = min(1.0, 1.0 - sample1.r);
		h = min(h, 1.0 - sample2.r);
		h = min(h, 1.0 - sample3.r);
		h = min(h, 1.0 - sample4.r);
	} else if (channel == 1) {
		h = min(1.0, 1.0 - sample1.g);
		h = min(h, 1.0 - sample2.g);
		h = min(h, 1.0 - sample3.g);
		h = min(h, 1.0 - sample4.g);
	} else if (channel == 2) {
		h = min(1.0, 1.0 - sample1.b);
		h = min(h, 1.0 - sample2.b);
		h = min(h, 1.0 - sample3.b);
		h = min(h, 1.0 - sample4.b);
	} else {
		h = min(1.0, 1.0 - sample1.a);
		h = min(h, 1.0 - sample2.a);
		h = min(h, 1.0 - sample3.a);
		h = min(h, 1.0 - sample4.a);
	}

	float shadowpara = min(1.0, 1.0 - saturate((h0 - h) * soften));
	return shadowpara;
}

// Phase 8.5: dual-sampler soft parallax shadow. Traces against the height-blended
// surface (base lerp overlay, weighted by mask) so the shadow follows the same
// profile the eye sees (albedo/normal/PBR blend). Alpha channel only — _paramh
// (DXT5) always stores height in alpha.
float ParallaxSoftShadowBlend(
	sampler2D hmapBase,
	sampler2D hmapOverlay,
	float mask,
	float2 uv,
	float2 lightDirTS,
	float soften,
	float scale)
{
	float2 lDir = -lightDirTS * scale;
	float h0 = 1.0 - lerp(tex2D(hmapBase, uv).a,
	                      tex2D(hmapOverlay, uv).a, mask);
	float h1 = 1.0 - lerp(tex2D(hmapBase, uv + 0.20 * lDir).a,
	                      tex2D(hmapOverlay, uv + 0.20 * lDir).a, mask);
	float h2 = 1.0 - lerp(tex2D(hmapBase, uv + 0.35 * lDir).a,
	                      tex2D(hmapOverlay, uv + 0.35 * lDir).a, mask);
	float h3 = 1.0 - lerp(tex2D(hmapBase, uv + 0.45 * lDir).a,
	                      tex2D(hmapOverlay, uv + 0.45 * lDir).a, mask);
	float h4 = 1.0 - lerp(tex2D(hmapBase, uv + 0.55 * lDir).a,
	                      tex2D(hmapOverlay, uv + 0.55 * lDir).a, mask);
	float h = min(min(min(h1, h2), min(h3, h4)), 1.0);
	return min(1.0, 1.0 - saturate((h0 - h) * soften));
}


#endif

#endif // COMMON_HLSL_INCLUDED