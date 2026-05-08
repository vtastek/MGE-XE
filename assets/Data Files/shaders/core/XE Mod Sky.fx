
// XE Mod Sky.fx
// MGE XE 0.16.0
// Sky and cloud rendering. Can be used as a core mod.

// Ordered dithering matrix
static const float ditherSky[4][4] = { 0.001176, 0.001961, -0.001176, -0.001699, -0.000654, -0.000915, 0.000392, 0.000131, -0.000131, -0.001961, 0.000654, 0.000915, 0.001699, 0.001438, -0.000392, -0.001438 };

//------------------------------------------------------------
// Sky and sky reflections

struct SkyVertOut {
    float4 pos : POSITION;
    float4 color : COLOR0;
    float2 texcoords : TEXCOORD0;
    float4 skypos : TEXCOORD1;
};

SkyVertOut SkyVS(StatVertIn IN) {
    SkyVertOut OUT;
    float4 pos = IN.pos;

    // Screw around with skydome, align default mesh with horizon
    if(!hasAlpha) {
        pos.z = 50 * (IN.pos.z + 200);
    }

    pos = mul(pos, world);
    OUT.skypos = float4(pos.xyz - eyePos, 1);

    pos = mul(pos, view);
    OUT.pos = mul(pos, proj);
    OUT.pos.z = 0.999999 * OUT.pos.w;   // Pin z to far plane so it renders to background
    OUT.color = IN.color;
    OUT.texcoords = IN.texcoords;

    return OUT;
}

float4 SkyPS(SkyVertOut IN, float2 vpos : VPOS) : COLOR0 {
    float4 c = 0;

    if(hasAlpha) {
        if (hasBones) {
            // Sun/moon billboard. Sample texture at lod 0 avoiding mip blurring
            c = tex2Dlod(sampBaseTex, float4(IN.texcoords, 0, 0));
        }
        else {
            // Standard texture filtering
            c = tex2D(sampBaseTex, IN.texcoords);
        }
        c *= IN.color;
    }

#ifdef USE_HLSL_PIPELINE
    // Sky is pre-exposed (matches legacy brightness). Linearize and tonemap
    // but skip intensityScalar and cancel the internal PI exposure bias.
    if (hasAlpha) {
        c.rgb = toLinearSrgb(c.rgb);
    }
    if (hasVCol) {
        float3 dir = normalize(IN.skypos.xyz);
        float3 fogFarLin = toLinearSrgb(fogColFar);
        float3 skyColLin = toLinearSrgb(skyCol);
        float skyT = 1 - pow(saturate(1 - 2.22 * saturate(dir.z - 0.075)), 1.15);
        c.rgb = lerp(fogFarLin, skyColLin, skyT);

        if (dir.z < 0) {
            float sunlightFactor = 1 - pow(1 - sunVis, 2);
            float3 sunColAdjusted = sunCol * sunlightFactor;
            float3 waterDepthCol = sunColAdjusted * float3(0.03, 0.04, 0.05) + (2 * skyCol + fogColFar) * float3(0.075, 0.08, 0.085);
            float3 ambientLin = toLinearSrgb(saturate(sunAmb));
            float3 waterDepthLin = toLinearSrgb(saturate(waterDepthCol));
            float below = saturate(-dir.z);
            float3 lowerSky = lerp(fogFarLin, ambientLin * 0.18, smoothstep(0.0, 0.35, below));
            lowerSky = lerp(lowerSky, waterDepthLin * 0.18, smoothstep(0.35, 1.0, below));
            c.rgb = lowerSky;
        }
        c.rgb += ditherSky[vpos.x % 4][vpos.y % 4];
    }
    c.rgb = ToneMap_AgX_Linear(c.rgb / PI);  // pre-exposed: cancel internal exposure bias
#else
    if(hasVCol) {
        // Moon shadow cutout. Use colour from scattering for sky (but preserves alpha)
        float4 f = fogColourSky(normalize(IN.skypos.xyz));
        c.rgb = f.rgb + ditherSky[vpos.x % 4][vpos.y % 4];
    }
#endif

    return c;
}

//------------------------------------------------------------
// Clouds

SkyVertOut CloudsVS(StatVertIn IN) {
    return SkyVS(IN);
}

float4 CloudsPS(SkyVertOut IN) : COLOR0 {
    float4 c = IN.color * tex2D(sampBaseTex, IN.texcoords);
#ifdef USE_HLSL_PIPELINE
    // Pre-exposed: cancel internal exposure bias to match legacy
    c.rgb = ToneMap_AgX_Linear(toLinearSrgb(c.rgb) / PI);
#endif
    return c;
}
