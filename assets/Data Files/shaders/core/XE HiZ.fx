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
    // Sample 4x4 grid with dilation to prevent false culling through thin gaps
    // Center the 4x4 samples around the output pixel for proper coverage
    float2 baseUV = texcoord - texelSize * 1.5;

    float maxDepth = 0.0;

    // Sample 4x4 = 16 pixels and take maximum
    [unroll]
    for (int y = 0; y < 4; y++) {
        [unroll]
        for (int x = 0; x < 4; x++) {
            float2 sampleUV = baseUV + float2(x, y) * texelSize;
            float depth = tex2D(sampDepthInput, sampleUV).r;
            maxDepth = max(maxDepth, depth);
        }
    }

    // Return MAXIMUM depth for conservative occlusion culling with dilation
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