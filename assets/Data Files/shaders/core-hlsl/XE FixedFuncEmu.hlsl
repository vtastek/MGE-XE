//---------------------------- PBR ----------------------------
#define PI 3.14159
#define PI_DIV2 1.57079632679

float saturate(float x) { return max(0.0, min(x, 1.0)); }

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
// XE FixedFuncEmu.hlsl  
// MGE XE 0.16.0
// Proper HLSL shader for Morrowind object rendering

//------------------------------------------------------------
// Shared Variables

// Matrices  
matrix proj;
matrix worldview;
matrix vertexBlendPalette[4];
float4 vertexBlendState;

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
float nearFogStart, nearFogRange;

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

// Texture suffix support - using preprocessor defines
//#define HAS_DIFFPARAM
//#define HAS_NORMAL
//#define USE_MIKKT
//#define USE_HQ_UPSCALE
#define USE_SIMPLE_PARALLAX // Enable this for simple offset parallax mapping
//#define USE_POM_PARALLAX // Enable this for raymarching parallax mapping
#ifdef HAS_NORMAL
float2 normres;  // Normal map texture resolution (width, height)
#endif

//------------------------------------------------------------
// Vertex Input/Output

struct VS_INPUT {
    float4 pos : POSITION;
    float3 normal : NORMAL;
    float4 color : COLOR0;
    float2 texcoord : TEXCOORD0;
    float4 blendweights : BLENDWEIGHT;
};

struct VS_OUTPUT {
    float4 position : POSITION;
    float3 normal : TEXCOORD0;
    float4 color : COLOR0;
    float2 texcoord : TEXCOORD1;
    float3 viewPos : TEXCOORD2;  // View space position (renamed to match PSIn)
    float fog : FOG;
};

//------------------------------------------------------------
// Helper Functions

// Skinning function from XE Common.hlsl
float4 skin(float4 pos, float4 blend) {
    float blendState = vertexBlendState.x;

    // Calculate missing blend weights
    if (blendState == 1)
        blend.y = 1 - blend.x;
    else if (blendState == 2)
        blend.z = 1 - (blend.x + blend.y);
    else if (blendState == 3)
        blend.w = 1 - (blend.x + blend.y + blend.z);

    // Weighted blend of matrices - ROW MAJOR (pos * matrix)
    float4 viewpos = mul(pos, vertexBlendPalette[0]) * blend.x;

    if (blendState >= 1)
        viewpos += mul(pos, vertexBlendPalette[1]) * blend.y;
    if (blendState >= 2)
        viewpos += mul(pos, vertexBlendPalette[2]) * blend.z;
    if (blendState >= 3)
        viewpos += mul(pos, vertexBlendPalette[3]) * blend.w;

    return viewpos;
}

// Fog function
float fogMWScalar(float dist) {
    return saturate((nearFogRange - dist) / (nearFogRange - nearFogStart));
}



float3 PBRNeutralToneMapping(float3 color) {
    float startCompression = 0.8 - 0.04;
    float desaturation = 0.15;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    float d = 1. - startCompression;
    float newPeak = 1. - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1. - 1. / (desaturation * (peak - newPeak) + 1.);
    return lerp(color, newPeak, g);
}


// --------------------------------------------------------------------------
// -----------------FILMIC PROCESS ------------------------------------------

static const float _CameraEV = 0.48;
static const float _PrintEV = 0.0;

// Scale factor to keep numbers in safe range for ps_3_0
#define SCALE_FACTOR 10.0f
#define INV_SCALE_FACTOR 0.1f

// Physical Film Constants
static const float D_MIN = 0.0;
static const float D_MAX = 2.0;
static const float NEUTRAL_PRINT_EV = 0.0;

static const float3x3 CROSSTALK_MATRIX = float3x3(
    0.33, 0.09, 0.03,
    0.05, 0.33, 0.07,
    0.05, 0.05, 0.33
);

// Safe logistic function with scaled inputs
float3 SafeLogistic(float3 x, float slope, float offset)
{
    // Clamp input to prevent exp overflow in ps_3_0
    float3 clamped = clamp(x + offset, -10.0, 10.0);
    return 1.0 / (1.0 + exp(-slope * clamped));
}

// Converts linear light value into film dye density (scaled)
float3 CaptureToDensity(float3 scene_linear_scaled)
{
    // Work with scaled values, then convert to log
    float3 safe_input = max(scene_linear_scaled, 1e-3); // Higher minimum for  safety

    float3 log_exp = log2(safe_input) / 3.32193; // log10 using log2 for better precision


    float3 t = SafeLogistic(log_exp, 3.0, 1.0);
    return D_MIN + (D_MAX - D_MIN) * t;
}

// Converts dye density to transmission (scaled)
float3 DevelopFromDensity(float3 density, bool is_paper)
{
    // Clamp density to safe range
    float3 safe_density = clamp(density, -5.0, 5.0);

    if (is_paper) {
        // Paper development with safe normalization
        float3 raw = exp2(-safe_density * 3.32193); // pow(10, -density) using exp2

        // Use simpler normalization to avoid tiny denominators
        float black_ref = exp2(-D_MAX * 0.8 * 3.32193);
        float white_ref = exp2(-D_MIN * 3.32193);

        // Safe range check
        float range = max(white_ref - black_ref, white_ref * 0.1);
        float3 normalized = (raw - black_ref) / range;

        return saturate(normalized) * 1.2;
    }
    else {
        // Negative development - simpler
        return exp2(-safe_density * 3.32193);
    }
}

// Process single channel through film pipeline
float ProcessChannel(float channel_value)
{
    // Scale to safe range
    float scaled = channel_value * SCALE_FACTOR;

    // Apply camera EV
    scaled *= exp2(_CameraEV);

    // Negative capture - single channel logistic
    float log_val = log2(max(scaled, 1e-3)) / 3.32193; // log10 equivalent
    float t1 = 1.0 / (1.0 + exp(-3.0 * (log_val + 1.0)));
    float density_neg = D_MIN + (D_MAX - D_MIN) * t1;

    // Negative transmission
    float transmission = exp2(-density_neg * 3.32193);

    // Print exposure
    transmission *= exp2(_PrintEV + NEUTRAL_PRINT_EV);

    // Paper capture
    float log_print = log2(max(transmission, 1e-3)) / 3.32193;
    float t2 = 1.0 / (1.0 + exp(-3.0 * (log_print + 1.0)));
    float density_paper = D_MIN + (D_MAX - D_MIN) * t2;

    // Paper reflection with safe normalization
    float reflection = exp2(-density_paper * 3.32193);

    // Simple contrast mapping instead of complex normalization
    reflection = saturate(reflection * 1.2);

    // Rescale back
    return reflection * INV_SCALE_FACTOR;
}

float3 filmp(float3 scene_color)
{
    // Apply crosstalk first (affects all channels)
    float3 scene_with_crosstalk = mul(CROSSTALK_MATRIX, scene_color);

    // Process each channel independently
    float r = ProcessChannel(scene_with_crosstalk.r);
    float g = ProcessChannel(scene_with_crosstalk.g);
    float b = ProcessChannel(scene_with_crosstalk.b);

    float3 result = float3(r, g, b);

    // Final artistic processing
    return (saturate(result));
}




#ifdef HAS_NORMAL

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

// Parallax height and normal parameters
static const float parallaxScale = 0.000001;
static const float parallaxBias = 0.0000501;
static const float heightScale = -4;

// Returns parallax-adjusted UV and a view-space normal derived from height.
void ParallaxHeightNormal(
    in float3 viewPos,
    inout float2 uvAdj,
    inout float3 normalVS,
    inout float3 deb)
{
    float3 Nvs = normalize(normalVS);


    // Build per-pixel TBN from screen-space derivs (no mesh tangents needed)
    float3 dpdx = ddx(-viewPos);
    float3 dpdy = ddy(-viewPos);
    float2 dtdx = ddx(uvAdj);
    float2 dtdy = ddy(uvAdj);

    float det = dtdx.x * dtdy.y - dtdx.y * dtdy.x;
    float invDet = (abs(det) > 1e-8) ? (1.0 / det) : 0.0;

    float3 T = normalize((dpdx * dtdy.y - dpdy * dtdx.y) * invDet);
    float3 B = normalize((-dpdx * dtdy.x + dpdy * dtdx.x) * invDet);

    // TBN (rows are T, B, N) — we'll use its transpose for VS<->TS conversion
    float3x3 TBN = float3x3(T, B, Nvs);
    float3x3 TBN_T = transpose(TBN);

    // View dir in view space and tangent space
    float3 Vvs = normalize(viewPos);
    float3 Vts = mul(Vvs, TBN_T);

    // --- Parallax offset (simple) ---
    float2 uv = uvAdj;
    float  h = tex2D(sampTex3, uv).a;
    float  parallax = h * parallaxScale + parallaxBias;
    uv += (parallax * Vts.xy) / max(Vts.z, 1e-3);   // shift along view ray in TS
    uvAdj = uv;

    // --- Height → normal from the same (parallaxed) UV ---
    float2 texel = 1.0 / normres;
    float hL = tex2D(sampTex3, uv + float2(-texel.x, 0)).a;
    float hR = tex2D(sampTex3, uv + float2(texel.x, 0)).a;
    float hD = tex2D(sampTex3, uv + float2(0, -texel.y)).a;
    float hU = tex2D(sampTex3, uv + float2(0, texel.y)).a;

    float dhdu_cd = (hR - hL);
    float dhdv_cd = (hU - hD);

    // Derivative of filtered height
    float hC = tex2D(sampTex3, uv).a;
    float dhdu_ddx = ddx(hC);
    float dhdv_ddy = ddy(hC);

    float2 dudx = ddx(uv) * normres;
    float2 dudy = ddy(uv) * normres;
    float footprint = max(length(dudx), length(dudy));

    // Blend factor: 0 = CD, 1 = ddx/ddy
    float w = saturate(footprint - 1.0); // ~0 when minified, ~1 when magnified

    float dhdu = lerp(dhdu_cd, dhdu_ddx, w);
    float dhdv = lerp(dhdv_cd, dhdv_ddy, w);

    // Tangent-space bump normal
    float3 nTS = normalize(float3(-dhdu * heightScale, -dhdv * heightScale, 1.0));

    // Rotate to view space
    normalVS = normalize(mul(nTS, TBN));
}

// mikkt

float2 DerivFromHeightMap(sampler2D hmap, float2 texST, float2 texDim)
{
    float2 onePixOffs = float2(1.0 / texDim.x, 1.0 / texDim.y);
    float2 st_r = texST + float2(onePixOffs.x, 0.0);
    float2 st_u = texST + float2(0.0, onePixOffs.y);

    float Hr = tex2D(hmap, st_r).x;
    float Hu = tex2D(hmap, st_u).x;
    float Hc = tex2D(hmap, texST).x;

    float2 dHduv = float2(Hr - Hc, Hu - Hc);
    return dHduv;
}

#ifdef USE_POM_PARALLAX
// Simple ps_3_0-compatible parallax occlusion mapping (raymarching)
float2 ParallaxRaymarch(
    sampler2D hmap,
    float2 uv,
    float3 Vts,
    float heightScale,
    int numSteps)
{
    float2 origUV = uv;
    float stepSize = 1.0 / numSteps;
    float2 delta = -Vts.xy / max(Vts.z, 1e-3) * heightScale * stepSize;
    float2 p = uv;
    float prevH = tex2D(hmap, p).a - p.y;
    float currH = prevH;
    for (int i = 0; i < numSteps; ++i) {
        p += delta;
        currH = tex2D(hmap, p).a - (i + 1) * stepSize;
        if (currH < 0) break;
        prevH = currH;
    }
    // Simple linear refinement
    float t = currH / (currH - prevH);
    float2 finalUV = p * t + origUV * (1 - t);
    return finalUV;
}
#endif

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

// SurfgradScaleDependent for ps_3_0
float3 SurfgradScaleDependent(float3 nrmBaseNormal, float3 positionVS, float2 deriv, float2 texST, float2 texDim)
{

    float3 dPdx = ddx(positionVS);
    float3 dPdy = ddy(positionVS);

    float2 dHdST = texDim * deriv;

    float2 texDx = ddx(texST);
    float2 texDy = ddy(texST);
    float dHdx = dHdST.x * texDx.x + dHdST.y * texDx.y;
    float dHdy = dHdST.x * texDy.x + dHdST.y * texDy.y;

    float3 vR1 = cross(dPdy, nrmBaseNormal);
    float3 vR2 = cross(nrmBaseNormal, dPdx);
    float det = dot(dPdx, vR1);

    float eps = 1.192093e-15F;
    float sign_det = det < 0.0 ? -1.0 : 1.0;
    float s = sign_det / max(eps, abs(det));

    return s * (dHdx * vR1 + dHdy * vR2);
}

// ResolveNormalFromSurfaceGradient for ps_3_0
float3 ResolveNormalFromSurfaceGradient(float3 normal, float3 surfGrad)
{
    return normalize(normal - surfGrad);
}

float2 DerivFromHeightMapHQ(sampler2D hmap, float2 texST, float2 texDim)
{
    float2 onePix = float2(1.0 / texDim.x, 1.0 / texDim.y);

    // Sobel filter for higher quality derivatives
    float h00 = tex2D(hmap, texST + onePix * float2(-1, -1)).x;
    float h10 = tex2D(hmap, texST + onePix * float2(0, -1)).x;
    float h20 = tex2D(hmap, texST + onePix * float2(1, -1)).x;
    float h01 = tex2D(hmap, texST + onePix * float2(-1, 0)).x;
    float h11 = tex2D(hmap, texST).x;
    float h21 = tex2D(hmap, texST + onePix * float2(1, 0)).x;
    float h02 = tex2D(hmap, texST + onePix * float2(-1, 1)).x;
    float h12 = tex2D(hmap, texST + onePix * float2(0, 1)).x;
    float h22 = tex2D(hmap, texST + onePix * float2(1, 1)).x;

    float dHdx = (h20 + 2.0 * h21 + h22) - (h00 + 2.0 * h01 + h02);
    float dHdy = (h02 + 2.0 * h12 + h22) - (h00 + 2.0 * h10 + h20);

    return float2(dHdx, dHdy) / 8.0; // Normalize Sobel kernel
}

#endif

//------------------------------------------------------------
// Vertex Shader

VS_OUTPUT vs_main(VS_INPUT input) {
    VS_OUTPUT output;

    // Transform vertex
    float4 viewpos;
    float3 normal;

    if (vertexBlendState.x > 0.5) {
        // Skinned vertex
        viewpos = skin(input.pos, input.blendweights);
        normal = skin(float4(input.normal, 0), input.blendweights).xyz;
    }
    else {
        // Rigid vertex
        viewpos = mul(input.pos, worldview);
        normal = mul(float4(input.normal, 0), worldview).xyz;
    }

    // Project to screen
    output.position = mul(viewpos, proj);

    // Pass through data
    output.normal = normalize(normal);
    output.color = input.color;
    output.texcoord = input.texcoord;
    output.viewPos = viewpos.xyz;

    // Simple fog
    float dist = length(viewpos);
    output.fog = fogMWScalar(dist);

    return output;
}

//------------------------------------------------------------
// Pixel Shader

float4 ps_main(VS_OUTPUT input) : COLOR{
    // Enhanced texture sampling with suffix support
    float3 diffuseParam = float3(1.0, 1.0, 1.0);  // Default white
    float3 deb = 0;
    float2 parallaxUV = input.texcoord;
    float3 normalVS = input.normal;
    float shadowpara = 1.0;

#ifdef HAS_NORMAL
    float bumpIntensity = 5.5;
    float3 T, B, N;
    BuildPerPixelTBN(input.normal, input.viewPos, parallaxUV, T, B, N);

    #if defined(USE_POM_PARALLAX)
    // Parallax Occlusion Mapping (raymarching)
    float3 Vvs = normalize(input.viewPos);
    float3x3 TBN = float3x3(T, B, N);
    float3x3 TBN_T = transpose(TBN);
    float3 Vts = mul(Vvs, TBN_T);
    parallaxUV = ParallaxRaymarch(sampTex3, parallaxUV, Vts, 0.00002, 12); // heightScale, steps
    // After parallax, sample normal as usual
    float2 texel = 1.0 / normres;
    float hL = tex2D(sampTex3, parallaxUV + float2(-texel.x, 0)).a;
    float hR = tex2D(sampTex3, parallaxUV + float2(texel.x, 0)).a;
    float hD = tex2D(sampTex3, parallaxUV + float2(0, -texel.y)).a;
    float hU = tex2D(sampTex3, parallaxUV + float2(0,  texel.y)).a;
    float dhdu = (hR - hL);
    float dhdv = (hU - hD);
    float3 nTS = normalize(float3(-dhdu * heightScale, -dhdv * heightScale, 1.0));
    normalVS = normalize(mul(nTS, TBN));
    // Soft parallax shadowing
    float3 lightDirWS = -lightSunDirection;
    float3 lightDirTS = mul(lightDirWS, TBN_T);
    shadowpara = ParallaxSoftShadow(sampTex3, parallaxUV, lightDirTS.xy, 5.0, 0.04 * 0.75);
#elif defined(USE_SIMPLE_PARALLAX)
    // Simple offset parallax mapping
    float3 Vvs = normalize(input.viewPos);
    float3x3 TBN = float3x3(T, B, N);
    float3x3 TBN_T = transpose(TBN);
    float3 Vts = mul(Vvs, TBN_T);
    parallaxUV = ParallaxSimple(sampTex3, parallaxUV, Vts, 0.000015); // heightScale
    // After parallax, sample normal as usual
    float2 texel = 1.0 / normres;
    float hL = tex2D(sampTex3, parallaxUV + float2(-texel.x, 0)).a;
    float hR = tex2D(sampTex3, parallaxUV + float2(texel.x, 0)).a;
    float hD = tex2D(sampTex3, parallaxUV + float2(0, -texel.y)).a;
    float hU = tex2D(sampTex3, parallaxUV + float2(0,  texel.y)).a;
    float dhdu = (hR - hL);
    float dhdv = (hU - hD);
    float3 nTS = normalize(float3(-dhdu * heightScale, -dhdv * heightScale, 1.0));
    normalVS = normalize(mul(nTS, TBN));
    // Soft parallax shadowing
    float3 lightDirWS = -lightSunDirection;
    float3 lightDirTS = mul(lightDirWS, TBN_T);
    shadowpara = ParallaxSoftShadow(sampTex3, parallaxUV, lightDirTS.xy, 5.0, 0.04 * 0.75);
#elif defined(USE_HQ_UPSCALE)
    float2 dHduv = DerivFromHeightMapHQ(sampTex3, parallaxUV, normres);
    float3 surfGradTS = float3(dHduv.x, dHduv.y, 0);
    float3 surfGradVS = surfGradTS.x * T + surfGradTS.y * B;
    surfGradVS *= bumpIntensity;
    float3 perturbedNormal = normalize(N - surfGradVS);
    normalVS = perturbedNormal;
#elif defined(USE_MIKKT)
    float2 dHduv = DerivFromHeightMap(sampTex3, parallaxUV, normres);
    float3 surfGradTS = float3(dHduv.x, dHduv.y, 0);
    float3 surfGradVS = surfGradTS.x * T + surfGradTS.y * B;
    surfGradVS *= bumpIntensity;
    float3 perturbedNormal = normalize(N - surfGradVS);
    normalVS = perturbedNormal;
#else
    ParallaxHeightNormal(input.viewPos.xyz, parallaxUV, normalVS, deb);
#endif
#endif



    float4 texColor = pow(tex2D(sampTex0, parallaxUV), 2.2); // Will use parallax-corrected UVs if HAS_NORMAL

    #ifdef HAS_DIFFPARAM
    texColor.rgb = pow(tex2D(sampTex2, parallaxUV).rgb, 2.2); // Will use parallax-corrected UVs if HAS_NORMAL
    #endif

    // PBR lighting calculation
    float3 V = normalize(-input.viewPos); // view direction in view space

    float3 albedo = texColor.rgb;
    float roughness = 0.9;
    float metalness = 0.0;
    float ao = 1.0;
    float radius = 1.6;
    float3 F0 = 0.08 * float3(0.5, 0.5, 0.5); // specular reflectance

#ifdef HAS_PARAM
    float4 param = tex2D(sampTex4, parallaxUV);
    metalness = param.x;
    roughness = param.y * param.y;
    F0 = 0.08 * param.z * param.z;
    ao = param.w;
#endif
#ifdef HAS_DIFFPARAM
    roughness = tex2D(sampTex2, parallaxUV).a;
    roughness = roughness * roughness;
#endif


#ifndef NOLIT
    float3 lighting = pow(lightSceneAmbient, 2.2) * ao;

    float3 down = float3(0, 0, -1);
    float3 downV = mul(worldview, down);
    float skylight = max(0.0, dot(down, normalVS));
    lighting *= skylight;
    // Sun light (Oren-Nayar)
    float3 Lsun = -lightSunDirection;
    float sunAtten = shadowpara;
    float3 sunBRDF = BRDF(normalVS, V, Lsun, albedo, metalness, roughness, roughness, radius, F0, 1);
    deb = sunBRDF;
    lighting += pow(lightSunDiffuse, 2.2) * sunBRDF * sunAtten;

#ifndef NO_POINT_LIGHTS
    // Point lights (Lambert)
    for (int i = 0; i < pointLightCount; i++) {
        float3 L = lightPosition[i] - input.viewPos;
        float dist = length(L);
        L = L / dist;
        float falloff = lightFalloffQuadratic[i] * dist * dist + lightFalloffConstant;
        float attenuation = (falloff > 0.0) ? (1.0 / falloff) : 0.0;
        float3 pointBRDF = BRDF(normalVS, V, L, albedo * 0, metalness, roughness, roughness, radius, F0, 0);
        lighting += (pow(lightDiffuse[i].rgb, 2.2) * pointBRDF + lightAmbient[i]) * attenuation;
    }
#endif
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
        effectiveDiffuse = sqrt(input.color.rgb);
        effectiveEmissive = materialEmissive.rgb * materialEmissive.rgb;
        effectiveAlpha = input.color.a;
    }
 else if (materialMode == 3) {
        // Mode 3: Use vertex color for emissive
        effectiveDiffuse = materialDiffuse.rgb;
        effectiveEmissive = input.color.rgb * input.color.rgb;
        effectiveAlpha = materialDiffuse.a;
    }
 else {
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

    c.rgb = PBRNeutralToneMapping(min(1.6e+6f, abs(c.rgb) * 6.14));
    // Apply fog
    //c.rgb = lerp(fogColNear, c.rgb, input.fog);
    c.rgb = pow(c.rgb, 1.0 / 2.2);
    //c.rgb = deb;
    #ifdef HAS_PARAM
    //c.rgb = tex2D(sampTex4, parallaxUV).a;
    //c.rgb = float3(1,0,0);
    #endif
    //c.rgb = F0;
    //c.rgb = c.rgb;

    return c;
}