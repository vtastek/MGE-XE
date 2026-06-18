
// XE Mod Caustics.fx
// MGE XE 0.16.0
// Outdoor water caustics. Can be used as a core mod.


//------------------------------------------------------------
// Caustics post-process

// Note: causticsStrength may be defined in another fx file, so cannot be used here to avoid multiple declarations
static const float causticsSunlightFactor = 1 - pow(1 - sunVis, 2);
static const float causticsStrengthScalar = 0.05 * alphaRef * saturate(0.75 * causticsSunlightFactor + 0.35 * length(fogColFar));

DeferredOut CausticsVS(float4 pos : POSITION, float2 tex : TEXCOORD0, float2 ndc : TEXCOORD1)
{
    DeferredOut OUT;

    // Fix D3D9 half pixel offset
    OUT.pos = float4(ndc.x - rcpRes.x, ndc.y + rcpRes.y, 0, 1);
    OUT.tex = float4(tex, 0, 0);

    // World space reconstruction vector
    OUT.eye = float3(view[0][2], view[1][2], view[2][2]);
    OUT.eye += (ndc.x / proj[0][0]) * float3(view[0][0], view[1][0], view[2][0]);
    OUT.eye += (ndc.y / proj[1][1]) * float3(view[0][1], view[1][1], view[2][1]);
    return OUT;
}

float4 CausticsPS(DeferredOut IN) : COLOR0
{
    float3 c = tex2Dlod(sampBaseTex, IN.tex).rgb;
    float depth = tex2Dlod(sampDepthPoint, IN.tex).r;
    float fog = fogMWScalar(depth);

    clip(nearViewRange - depth);

    float3 uwpos = eyePos + IN.eye * depth;
    uwpos.z -= waterLevel;
    clip(-uwpos.z);

    float3 sunray = uwpos - sunVec * (uwpos.z / sunVec.z);
#ifdef WATER_FLOW_MAP
    // Caustics follow the current: advect the caustic noise UV downstream with the same
    // bounded ping-pong as the water normals. nearStr (river/beach group) drives it; sea
    // and ponds get nearStr 0 → identical to the static sample. The flow uniforms are
    // shared from XE Mod Water.fx (included earlier in this effect) — do NOT redeclare.
    float4 cflow    = tex2Dlod(sampFlow, float4((sunray.xy - flowMapTransform.xy) * flowMapTransform.zw, 0, 0));
    float2 cflowDir = cflow.rg * 2 - 1;
    float  cNearStr = saturate((0.5 - cflow.a) * 2) * flowMapWeight;
    float  crate    = flowScrollSpeed / flowCycleUV;
    float  cph0     = frac(time * crate);
    float  cph1     = frac(time * crate + 0.5);
    float  cbl      = abs(1.0 - 2.0 * cph0);
    float2 cdisp    = cflowDir * cNearStr * flowCycleUV;
    float2 cuv      = sunray.xy / 1104;
    float  caustB   = lerp(tex3D(sampWater3d, float3(cuv - cdisp * cph0, 0.4 * time)).b,
                           tex3D(sampWater3d, float3(cuv - cdisp * cph1, 0.4 * time)).b, cbl);
    float caust = causticsStrengthScalar * caustB;
#else
    float caust = causticsStrengthScalar * tex3D(sampWater3d, float3(sunray.xy / 1104, 0.4 * time)).b;
#endif
    caust *= saturate(125 / depth * min(fwidth(sunray.x), fwidth(sunray.y)));
    c *= 1 + (caust - 0.3) * saturate(exp(uwpos.z / 400)) * saturate(uwpos.z / -30) * fog;

    return float4(c, 1);
}
