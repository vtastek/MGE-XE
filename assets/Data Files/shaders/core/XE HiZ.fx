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

// Texel size for sampling 4 pixels
float2 texelSize;

//------------------------------------------------------------
// Downsample shader - Samples 4 pixels and outputs MAX depth
//------------------------------------------------------------

struct VS_OUTPUT {
    float4 pos : POSITION;
    float2 texcoord : TEXCOORD0;
};

VS_OUTPUT DownsampleVS(float4 pos : POSITION, float2 texcoord : TEXCOORD0) {
    VS_OUTPUT OUT;
    OUT.pos = pos;
    OUT.texcoord = texcoord;
    return OUT;
}

float4 DownsamplePS(float2 texcoord : TEXCOORD0) : COLOR0 {
    // Sample 4 pixels in a 2x2 grid
    float d0 = tex2D(sampDepthInput, texcoord).r;
    float d1 = tex2D(sampDepthInput, texcoord + float2(texelSize.x, 0)).r;
    float d2 = tex2D(sampDepthInput, texcoord + float2(0, texelSize.y)).r;
    float d3 = tex2D(sampDepthInput, texcoord + texelSize).r;

    // Return MAXIMUM depth for conservative occlusion culling
    float maxDepth = max(max(d0, d1), max(d2, d3));

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