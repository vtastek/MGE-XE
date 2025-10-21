// XE HiZ.fx
// MGE XE 0.16.0
// Hierarchical Z-Buffer pyramid generation

// Input depth texture (previous mip level)
texture texDepthInput;
sampler sampDepthInput = sampler_state {
    Texture = <texDepthInput>;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = NONE;
    AddressU = Clamp;
    AddressV = Clamp;
};

// Source texture info: xy = source dimensions, z = source mip level, w = destination mip level
float4 sourceInfo;

//------------------------------------------------------------
// Downsample shader - EXACT D3D9 reference implementation
// Uses VPOS semantic and tex2Dlod for precise mip sampling
//------------------------------------------------------------

struct VS_OUTPUT {
    float4 pos : POSITION;
};

VS_OUTPUT DownsampleVS(float4 pos : POSITION) {
    VS_OUTPUT OUT;
    OUT.pos = pos;
    return OUT;
}

float4 DownsamplePS(float4 screenPos : VPOS) : COLOR0 {
    float sourceWidth = sourceInfo.x;
    float sourceHeight = sourceInfo.y;
    float sourceMip = sourceInfo.z;

    // CRITICAL FIX: Use exact D3D9 reference implementation UV calculation
    // Reference: nCoords0 = float2((PositionSS.x * 2) / width, (PositionSS.y * 2) / height);
    // Note: In D3D9, dividing by texture dimensions gives normalized [0,1] coordinates
    float2 uv0 = float2((screenPos.x * 2.0) / sourceWidth, (screenPos.y * 2.0) / sourceHeight);

    // Add 1 texel offset in normalized coordinates (reference does: nCoords0.x + (1 / width))
    float2 uv1 = float2(uv0.x + (1.0 / sourceWidth), uv0.y);
    float2 uv2 = float2(uv0.x, uv0.y + (1.0 / sourceHeight));
    float2 uv3 = float2(uv1.x, uv2.y);

    // Sample 2x2 block using tex2Dlod with explicit source mip level
    // CRITICAL: Use .r (red channel) not .x for R32F format
    float4 depths;
    depths.x = tex2Dlod(sampDepthInput, float4(uv0, 0, sourceMip)).r;
    depths.y = tex2Dlod(sampDepthInput, float4(uv1, 0, sourceMip)).r;
    depths.z = tex2Dlod(sampDepthInput, float4(uv2, 0, sourceMip)).r;
    depths.w = tex2Dlod(sampDepthInput, float4(uv3, 0, sourceMip)).r;

    // Return MAXIMUM depth for conservative occlusion (reference does same)
    float maxDepth = max(max(depths.x, depths.y), max(depths.z, depths.w));
    return float4(maxDepth, maxDepth, maxDepth, 1.0);
}

//------------------------------------------------------------
// Technique
//------------------------------------------------------------

technique T0 {
    pass P0 {
        ZEnable = false;
        ZWriteEnable = false;
        CullMode = NONE;
        AlphaBlendEnable = false;
        AlphaTestEnable = false;

        VertexShader = compile vs_3_0 DownsampleVS();
        PixelShader = compile ps_3_0 DownsamplePS();
    }
}