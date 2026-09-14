
// XE FixedFuncEmu.fx
// MGE XE 0.9
// Replacement shaders for Morrowind's object rendering

#include "XE Common.fx"

shared texture tex4, tex5;
shared float4 materialDiffuse, materialAmbient, materialEmissive;
shared float3 lightSceneAmbient, lightSunDiffuse, lightDiffuse[8];
shared float4 lightAmbient[2];
shared float3 lightSunDirection;
shared float4 lightPosition[6];
shared float4 lightFalloffQuadratic[2], lightFalloffLinear[2];
shared float lightFalloffConstant;
shared matrix texgenTransform;
shared float4 bumpMatrix;
shared float2 bumpLumiScaleBias;
shared float pbrRoughness, pbrF0;
shared int pbrTexFilter, pbrAddressU, pbrAddressV;

sampler sampFFE0 = sampler_state { texture = <tex0>; };
sampler sampFFE1 = sampler_state { texture = <tex1>; };
sampler sampFFE2 = sampler_state { texture = <tex2>; };
sampler sampFFE3 = sampler_state { texture = <tex3>; };
sampler sampFFE4 = sampler_state { texture = <tex4>; };
sampler sampFFE5 = sampler_state { texture = <tex5>; };

sampler sampPBRNormals = sampler_state { texture = <tex4>; SRGBTexture = 0; MinFilter = <pbrTexFilter>; MagFilter = <pbrTexFilter>; MipFilter = linear; MaxAnisotropy = 4; AddressU = <pbrAddressU>; AddressV = <pbrAddressV>; };
sampler sampPBRParameters = sampler_state { texture = <tex5>; SRGBTexture = 0; MinFilter = <pbrTexFilter>; MagFilter = <pbrTexFilter>; MipFilter = linear; MaxAnisotropy = 4; AddressU = <pbrAddressU>; AddressV = <pbrAddressV>; };

//------------------------------------------------------------

#ifdef VERIFY
#define FFE_VB_COUPLING float2 texcoord0 : TEXCOORD0; float4 col : COLOR;
#define FFE_SHADER_COUPLING float4 texcoord01 : TEXCOORD0; float4 col : COLOR;
#define FFE_TRANSFORM_SKIN worldpos = rigidVertex(IN.pos); normal = rigidNormal(IN.nrm);
#define FFE_TEXCOORDS_TEXGEN float3 texgen = texgenReflection(worldpos, normal); texgen = mul(float4(texgen, 1), texgenTransform).xyz; OUT.texcoord01 = float4(IN.texcoord0, texgen.xy);
#define FFE_VERTEX_COLOUR OUT.col = IN.col;
#define FFE_NORMAL_MAPPING normal = normalMap(tex2D(sampFFE4, IN.texcoord01.xy).rgb, normal, IN.view, IN.texcoord01.xy);
#define FFE_REFLECTANCE_MODEL float4 parameters_data = tex2D(sampFFE5, IN.texcoord01.xy); float f_0_nonmetal = pbrF0Range * parameters_data.b; float3 f_0 = lerp(tex2D(sampFFE0, IN.texcoord01.xy), f_0_nonmetal.xxx, parameters_data.r); LightResult lr = calcPBRLighting(IN.lightvec, geom_normal, normal, normalize(IN.view), parameters_data.g, f_0);
#define FFE_LIGHTS_ACTIVE 8
#define FFE_VERTEX_MATERIAL diffuse = vertexMaterialDiffAmb(d, a, IN.col);
#define FFE_TEXTURING c = diffuse + bumpmapLumiStage(sampFFE1, IN.texcoord01.zw, tex2D(sampFFE0, IN.texcoord01.xy)); c.rgb += s * parameters_data.a;
#define FFE_FOG_APPLICATION c.rgb = lerp(FogCol1, c.rgb, fog);
#endif

#ifdef FFE_ERROR_MATERIAL
#define FFE_VB_COUPLING
#define FFE_SHADER_COUPLING
#define FFE_TRANSFORM_SKIN worldpos = rigidVertex(IN.pos); normal = float3(0, 0, 1);
#define FFE_TEXCOORDS_TEXGEN
#define FFE_VERTEX_COLOUR
#define FFE_NORMAL_MAPPING
#define FFE_REFLECTANCE_MODEL LightResult lr;
#define FFE_LIGHTS_ACTIVE 0
#define FFE_VERTEX_MATERIAL diffuse = float4(1, 0, 0.5, 1);
#define FFE_TEXTURING
#define FFE_FOG_APPLICATION c.rgb = lerp(FogCol1, c.rgb, fog);
#endif


//------------------------------------------------------------
// Transform library functions

// Vertex transforms, note that scaled normals are normalized in the pixel shader
float4 rigidVertex(float4 pos) { return mul(pos, world); }
float3 rigidNormal(float3 normal) { return mul(float4(normal, 0), world).xyz; }

float4 skinnedVertex(float4 pos, float4 weights) { return skin(pos, weights); }
float3 skinnedNormal(float3 normal, float4 weights) { return skin(float4(normal, 0), weights).xyz; }

// Texgens with world space inputs, normals must be normalized due to ubiquitous scaling matrices
float3 texgenNormal(float3 normal) { return mul(float4(normalize(normal), 0), view).xyz; }
float3 texgenPosition(float4 pos) { return mul(pos, view).xyz; }
float3 texgenReflection(float4 pos, float3 normal) { float3 r = reflect(normalize(pos.xyz - EyePos), normalize(normal)); return mul(float4(r, 0), view).xyz; }
float3 texgenSphere(float2 tex) { return float3(0.5 * tex + 0.5, 0); }

//------------------------------------------------------------
// sRGB functions

float3 toLinear(float3 c)
{
    float3 low = c / 12.92;
    float3 high = pow((c + 0.055) / 1.055, 2.4);
    return (c < 0.04045) ? low : high;
    
}

float3 toSRGB(float3 c)
{
    float3 low = 12.92 * c;
    float3 high = 1.055 * pow(c, 1.0/2.4) - 0.055;
    return (c < 0.0031308) ? low : high;
}

float4 toLinear4(float4 c)
{
    return float4(toLinear(c.rgb), c.a);
}

float toLuma(float3 c)
{
    // sRGB / Rec. 709 luma
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

//------------------------------------------------------------
// Lighting library functions

// Number of light groups; lights are vectorized into groups of 4
static const int LGs = max(1, ceil(FFE_LIGHTS_ACTIVE / 4.0));

// Reflected light result
struct LightResult
{
    float3 ambient, diffuse, specular;
};

// Point lights
float4 calcLighting4(float4 lightvec[3*LGs], int group, float3 normal)
{
    float4 dist2 = 0, lambert = 0;
    
    // Do four dot products as three mads
    for(int i = 0; i != 3; ++i)
        dist2 += pow(lightvec[3*group + i], 2);
    
    // Same for N.L
    for(int i = 0; i != 3; ++i)
        lambert += normal[i] * lightvec[3*group + i];
    
    // Normalize L after the fact
    float4 dist = sqrt(dist2);
    lambert = saturate(lambert / dist);
    
    // Attenuation
    float4 att = 1.0 / (lightFalloffQuadratic[group] * dist2 + lightFalloffConstant);
    // (slower) float4 att = 1.0 / (lightFalloffQuadratic[group] * dist2 + lightFalloffLinear[group] * dist + lightFalloffConstant);
    return (lambert + lightAmbient[group]) * att;
}

float3 calcPointLighting(uniform int lights, float4 lightvec[3*LGs], float3 normal)
{
    float4 lambert[LGs];
    float3 l = 0;    

    for(int i = 0; i != LGs; ++i)
        lambert[i] = calcLighting4(lightvec, i, normal);
    
    for(int i = 0; i != lights; ++i)
        l += lambert[i/4][i%4] * lightDiffuse[i];
    
    return l;
}

LightResult calcStdLighting(float4 lightvec[3*LGs], float3 normal)
{
    LightResult r;
    
    // Sunlight
    float3 l = -lightSunDirection;
    float n_dot_l = saturate(dot(normal, l));

    r.ambient = lightSceneAmbient;
    r.diffuse = lightSunDiffuse * n_dot_l;
    r.specular = 0;

    // Point lights
    r.diffuse += calcPointLighting(FFE_LIGHTS_ACTIVE, lightvec, normal);

    return r;
}

// Normal mapping, including recovery of TBN cotangent frame
// credit: http://www.thetenthplanet.de/archives/1180
float3x3 cotangentFrame(float3 normal, float3 pos, float2 uv)
{
    // Get gradients across triangle
    // Note negation of ddy due to DirectX viewport inversion of y
    float3 dpos1 = ddx(pos), dpos2 = -ddy(pos);
    float2 duv1 = ddx(uv), duv2 = -ddy(uv);

    // Solve the linear system
    float3 dp2perp = cross(dpos2, normal);
    float3 dp1perp = cross(normal, dpos1);
    float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    float3 B = dp2perp * duv1.y + dp1perp * duv2.y;

    // Construct a scale-invariant frame
    float rcpmax = rsqrt(max(dot(T, T), dot(B, B)));
    return float3x3(rcpmax * T, rcpmax * B, normal);
}

float3 normalMap(float3 map, float3 normal, float3 v, float2 uv)
{
    // Normal map is assumed to be 8-bit, green channel facing up
    map = map * 255.0/127.0 - 128.0/127.0;
    map.y = -map.y;
    return normalize(mul(map, cotangentFrame(normal, -v, uv)));
}

// Static tonemap
float3 tonemap(float3 c)
{
    // Curve maps 0 -> 0, 1.0 -> 0.84, up to 2.2 -> 1.0
    c = clamp(c, 0, 2.2);
    c = (((0.0548303 * c - 0.189786) * c - 0.154732) * c + 1.12969) * c;
    return c;
}

// Vertex material modes
float4 vertexMaterialNone(float3 d, float3 a)
{
    return float4(materialDiffuse.rgb * d + materialAmbient.rgb * a + materialEmissive.rgb, materialDiffuse.a);
}

float4 vertexMaterialDiffAmb(float3 d, float3 a, float4 col)
{
    return float4(col.rgb * (d + a) + materialEmissive.rgb, col.a);
}

float4 vertexMaterialEmissive(float3 d, float3 a, float4 col)
{
    return float4(materialDiffuse.rgb * d + materialAmbient.rgb * a + col.rgb, materialDiffuse.a);
}

//------------------------------------------------------------
// PBR lighting functions

static const float pbrF0Range = 0.2;

float3 fresnelSchlick(float3 f_0, float cosine)
{
    return f_0 + (1 - f_0) * pow(1 - cosine, 5);
}

float3 calcPBRSpecular(float3 normal, float3 l, float3 v, float n_dot_l, float n_dot_v, float alpha2, float3 f_0)
{
    float3 h = normalize(v + l);
    float n_dot_h = saturate(dot(normal, h));
    float h_dot_l = saturate(dot(h, l));

    // Fresnel term for microfacets
    float3 f = fresnelSchlick(f_0, h_dot_l);

    // GGX NDF
    float d = alpha2 / pow((n_dot_h * alpha2 - n_dot_h) * n_dot_h + 1, 2);
    
    // Heitz height-correlated Smith
    float lambda_ggxv = n_dot_l * sqrt((-n_dot_v * alpha2 + n_dot_v) * n_dot_v + alpha2);
    float lambda_ggxl = n_dot_v * sqrt((-n_dot_l * alpha2 + n_dot_l) * n_dot_l + alpha2);
    float vis = 0.5 / (lambda_ggxv + lambda_ggxl);

    // Cook-Torrance microfacet model
    // No division by pi, as Morrowind uses lighting tuned for non-PBR punctual lights
    return d * vis * f;
}

LightResult calcPBREnv(float3 normal, float3 v, float n_dot_v, float alpha2, float3 f_0)
{
    LightResult r;
    float3 rv = reflect(-v, normal);
    float3 adjustedSky = 0.4 * toLinear(SkyCol);
    
    // Analytical ambient environment model
    // upper hemisphere = sky colour, lower hemisphere = ambient light colour
    r.ambient = lerp(lightSceneAmbient, adjustedSky, 0.5 + 0.5 * normal.z);
    r.diffuse = 0;

    // Simplified specular BRDF integrated over sky hemisphere
    float3 k_s = lerp(0, adjustedSky, saturate(0.5 + 0.5 * rv.z / alpha2));
    float3 f = fresnelSchlick(f_0, n_dot_v);
    float vis = n_dot_v / (n_dot_v + 0.5 * alpha2);
    r.specular = k_s * vis * f;
    
    return r;
}

float3 calcPointSpec1(float3 l, float3 col, float falloffQ, float3 geom_normal, float3 normal, float3 v, float n_dot_v, float alpha2, float3 f_0)
{
    float dist2 = dot(l, l);
    l = normalize(l);
    float geom_vis = saturate(8 * dot(geom_normal, l) + 1);
    [branch] if (geom_vis > 0)
    {
        float n_dot_l = saturate(dot(normal, l));
        float att = 1.0 / (falloffQ * dist2 + lightFalloffConstant);
        return col * n_dot_l * att * geom_vis * calcPBRSpecular(normal, l, v, n_dot_l, n_dot_v, alpha2, f_0);
    }
    return 0;
}    

LightResult calcPBRLighting(float4 lightvec[3*LGs], float3 geom_normal, float3 normal, float3 v, float roughness, float3 f_0)
{
    LightResult r;
    r.ambient = 0; r.diffuse = 0; r.specular = 0;
    
    // Calculate NDF parameter alpha^2 = (roughness^2)^2 = roughness^4
    float alpha2 = pow(roughness, 4);
    // Note that interpolated normals can produce an unphysical back-facing normal
    // Add an epsilon to n . v to avoid numeric instability
    float n_dot_v = saturate(dot(normal, v)) + 1e-5;

    // Sunlight
    float3 l = -lightSunDirection;
    
    // Calculate fragment lighting only if not masked by geometry
    float geom_vis = saturate(8 * dot(geom_normal, l) + 1);
    [branch] if (geom_vis > 0)
    {
        float n_dot_l = saturate(dot(normal, l));
        r.diffuse = lightSunDiffuse * geom_vis * n_dot_l;
        r.specular = SunVis * r.diffuse * calcPBRSpecular(normal, l, v, n_dot_l, n_dot_v, alpha2, f_0);
    }

    // Environment
    LightResult e = calcPBREnv(normal, v, n_dot_v, alpha2, f_0);
    r.ambient += e.ambient;
    r.specular += e.specular;

    // Point lights
    r.diffuse += calcPointLighting(FFE_LIGHTS_ACTIVE, lightvec, normal);
    [branch] if (lightFalloffQuadratic[0].x > 0)
    {
        float3 pointLight0 = float3(lightvec[0].x, lightvec[1].x, lightvec[2].x);
        r.specular += calcPointSpec1(pointLight0, lightDiffuse[0], lightFalloffQuadratic[0].x, geom_normal, normal, v, n_dot_v, alpha2, f_0);
    }

    return r;
}

//------------------------------------------------------------
// Data coupling framework

struct FFEVertIn
{
    float4 pos : POSITION;
    float3 nrm : NORMAL;
    
    /* template */ FFE_VB_COUPLING
};

struct FFEPixel
{
    float4 pos : POSITION;
    float4 nrm_fog : NORMAL;
    
    /* template */ FFE_SHADER_COUPLING
    
    float3 view : TEXCOORD2;
    float4 lightvec[3*LGs] : TEXCOORD3;
};

//------------------------------------------------------------
// Shader framework

// Relatively simple, notably passes lighting vectors in interpolators
FFEPixel PerPixelVS(FFEVertIn IN)
{
    FFEPixel OUT;

    // Transforms
    float4 worldpos;
    float3 normal;
    /* template */ FFE_TRANSFORM_SKIN
    
    OUT.pos = mul(worldpos, view);
    float dist = length(OUT.pos);
    OUT.pos = mul(OUT.pos, proj);
    OUT.nrm_fog = float4(normal, fogMWScalar(dist));
    OUT.view = EyePos - worldpos.xyz;
    
    // Texcoord routing and texgen
    /* template */ FFE_TEXCOORDS_TEXGEN
    
    // Vertex colour
    /* template */ FFE_VERTEX_COLOUR
    
    // Point lighting setup, vectorized
    for(int i = 0; i != LGs; ++i)
    {
        OUT.lightvec[3*i + 0] = lightPosition[i + 0] - worldpos.x;
        OUT.lightvec[3*i + 1] = lightPosition[i + 2] - worldpos.y;
        OUT.lightvec[3*i + 2] = lightPosition[i + 4] - worldpos.z;
    }
    
    return OUT;
}

// Bumpmap stages return dUdV alpha channel due to select1 alpha op
float4 bumpmapStage(sampler s, float2 tc, float4 dUdV)
{
    float2 offset = mul(dUdV.rg, float2x2(bumpMatrix.xy, bumpMatrix.zw));
    return float4(tex2D(s, tc + offset).rgb, dUdV.a);
}

float4 bumpmapLumiStage(sampler s, float2 tc, float4 dUdVL)
{
    float4 c = bumpmapStage(s, tc, dUdVL);
    c.rgb *= saturate(dUdVL.b * bumpLumiScaleBias.x + bumpLumiScaleBias.y);
    return c;
}

// Per-pixel lighting augmented with fixed exposure HDR tonemap
// Vectorized lighting reduces instruction count by 50%
float4 PerPixelPS(FFEPixel IN) : COLOR0
{
    float3 geom_normal = normalize(IN.nrm_fog.xyz);
    float3 normal = geom_normal;
    float fog = IN.nrm_fog.w;

    // Normal mapping
    /* template */ FFE_NORMAL_MAPPING
    
    // Standard Morrowind lighting: sun, ambient, and point lights
    /* template */ FFE_REFLECTANCE_MODEL
    float3 a = lr.ambient, d = lr.diffuse, s = lr.specular;
    
    // Material
    float4 diffuse;
    /* template */ FFE_VERTEX_MATERIAL
    
    // Texturing and combinators
    float4 c = diffuse;
    /* template */ FFE_TEXTURING
    
    // Static tonemap and final fogging
    c.rgb = tonemap(c.rgb);
    /* template */ FFE_FOG_APPLICATION
    
    return c;
}

//-----------------------------------------------------------------------------

technique FFE
{
    pass
    {
        VertexShader = compile vs_3_0 PerPixelVS();
        PixelShader = compile ps_3_0 PerPixelPS();
    }
}
