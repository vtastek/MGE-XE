// GPU-based Hi-Z Occlusion Culling Shader (D3D9 / Shader Model 3.0)
// Based on AMD/ATI Hierarchical-Z-Buffer technique
// Performs batch occlusion testing of bounding boxes against Hi-Z pyramid

// View and projection matrices (row-major to match D3DX convention)
row_major float4x4 g_mView;
row_major float4x4 g_mProjection;
row_major float4x4 g_mViewProjection;

// View-frustum planes in world space (normals face outward)
float4 g_FrustumPlanes[6];

// Viewport dimensions
float2 g_ViewportSize;       // Width and Height in pixels
float2 g_ResultsSize;        // Results texture dimensions

// Hi-Z pyramid textures (ping-pong: Even and Odd mip levels)
texture g_texHiZEven;
sampler g_sampHiZEven = sampler_state
{
    Texture = <g_texHiZEven>;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
    AddressU = CLAMP;
    AddressV = CLAMP;
};

texture g_texHiZOdd;
sampler g_sampHiZOdd = sampler_state
{
    Texture = <g_texHiZOdd>;
    MinFilter = POINT;
    MagFilter = POINT;
    MipFilter = POINT;
    AddressU = CLAMP;
    AddressV = CLAMP;
};

struct VS_INPUT
{
    // Bounding box min/max in world space
    float3 BBoxMin : POSITION0;
    float3 BBoxMax : POSITION1;
    // Output pixel location in results texture
    float2 ResultPixel : TEXCOORD0;
};

struct PS_INPUT
{
    float4 Position : POSITION0;
    float3 BBoxMin : TEXCOORD0;
    float3 BBoxMax : TEXCOORD1;
};

// Compute signed distance from point to plane
float DistanceToPlane(float4 plane, float3 pt)
{
    return dot(float4(pt, 1.0), plane);
}

// Frustum cull bounding box against 6 frustum planes
// Returns > 0 if visible, <= 0 if completely outside frustum
float CullBox(float4 planes[6], float3 bboxMin, float3 bboxMax)
{
    float minDist = 1e10;

    // Test bbox against each plane
    for (int i = 0; i < 6; i++)
    {
        // Find the positive vertex (furthest in direction of plane normal)
        float3 pVertex;
        pVertex.x = (planes[i].x >= 0) ? bboxMax.x : bboxMin.x;
        pVertex.y = (planes[i].y >= 0) ? bboxMax.y : bboxMin.y;
        pVertex.z = (planes[i].z >= 0) ? bboxMax.z : bboxMin.z;

        float dist = DistanceToPlane(planes[i], pVertex);
        minDist = min(minDist, dist);
    }

    return minDist;
}

PS_INPUT VS(VS_INPUT input)
{
    PS_INPUT output;

    // Calculate pixel location in results texture
    // Convert from result pixel index (already +0.5 offset) to clip space [-1,1]
    output.Position = float4(
        (input.ResultPixel.x / g_ResultsSize.x * 2.0) - 1.0,
        1.0 - (input.ResultPixel.y / g_ResultsSize.y * 2.0),
        0.5,  // Z = 0.5 for mid-depth
        1.0
    );

    // Pass through bounding box
    output.BBoxMin = input.BBoxMin;
    output.BBoxMax = input.BBoxMax;

    return output;
}

float4 PS(PS_INPUT input) : COLOR0
{
    float3 bboxMin = input.BBoxMin;
    float3 bboxMax = input.BBoxMax;

    // Frustum culling first (cheap rejection test)
    float frustumTest = CullBox(g_FrustumPlanes, bboxMin, bboxMax);
    if (frustumTest <= 0)
    {
        return float4(0, 0, 0, 0); // Culled by frustum
    }

    // Compute bbox center and half-extents
    float3 center = (bboxMin + bboxMax) * 0.5;
    float3 extents = (bboxMax - bboxMin) * 0.5;

    // Transform bbox corners to clip space to find screen-space bounds
    float3 corners[8];
    corners[0] = float3(bboxMin.x, bboxMin.y, bboxMin.z);
    corners[1] = float3(bboxMax.x, bboxMin.y, bboxMin.z);
    corners[2] = float3(bboxMin.x, bboxMax.y, bboxMin.z);
    corners[3] = float3(bboxMax.x, bboxMax.y, bboxMin.z);
    corners[4] = float3(bboxMin.x, bboxMin.y, bboxMax.z);
    corners[5] = float3(bboxMax.x, bboxMin.y, bboxMax.z);
    corners[6] = float3(bboxMin.x, bboxMax.y, bboxMax.z);
    corners[7] = float3(bboxMax.x, bboxMax.y, bboxMax.z);

    // Find screen-space bounding rectangle and closest depth
    float2 screenMin = float2(1, 1);
    float2 screenMax = float2(-1, -1);
    float closestDepth = 1e38;  // Large value for view-space depth

    for (int i = 0; i < 8; i++)
    {
        // ROW MAJOR: vector * matrix (matches D3DXVec3Transform and other HLSL shaders)
        float4 clipPos = mul(float4(corners[i], 1.0), g_mViewProjection);

        // Skip if behind camera
        if (clipPos.w <= 0)
            continue;

        float3 ndc = clipPos.xyz / clipPos.w;
        screenMin = min(screenMin, ndc.xy);
        screenMax = max(screenMax, ndc.xy);
        // CRITICAL: Use view-space depth (clipPos.w) to match depth texture format!
        // Depth texture stores clipPos.w (view-space Z), not NDC Z
        closestDepth = min(closestDepth, clipPos.w);
    }

    // Convert NDC [-1,1] to UV [0,1]
    float2 uvMin = screenMin * float2(0.5, -0.5) + 0.5;
    float2 uvMax = screenMax * float2(0.5, -0.5) + 0.5;

    // CRITICAL: Y-flip inverts min/max relationship! Fix it.
    // After Y-flip, uvMin.y might be > uvMax.y, so re-establish min/max
    float2 uvMinCorrected = min(uvMin, uvMax);
    float2 uvMaxCorrected = max(uvMin, uvMax);
    uvMin = uvMinCorrected;
    uvMax = uvMaxCorrected;

    // Clamp to screen bounds
    uvMin = saturate(uvMin);
    uvMax = saturate(uvMax);

    // Compute screen-space width for mip selection
    // CRITICAL: Hi-Z pyramid mip 0 is HALF viewport size (320x240 for 640x480 viewport)
    // Calculate size relative to Hi-Z mip 0, not full viewport
    float2 hiZMip0Size = g_ViewportSize * 0.5;
    float2 screenSize = (uvMax - uvMin) * hiZMip0Size;
    float maxScreenSize = max(screenSize.x, screenSize.y);

    // Choose mip level to match CPU algorithm (renderdepth.cpp:667)
    // Target ~6 pixels at selected mip for good coverage, with -1 mip bias for higher resolution
    float mipLevel = max(0, floor(log2(maxScreenSize / 6.0)) - 1.0);
    mipLevel = clamp(mipLevel, 0, 4); // Clamp to valid mip range (5 mips: 0-4, down to 8x8)

    // Sample Hi-Z at 4 corners of screen-space bbox using tex2Dlod (SM 3.0)
    // Reference: Even mips are in texHiZEven, odd mips are in texHiZOdd
    float4 depths;
    if (int(mipLevel) % 2 == 0) {
        // Even mip level - sample from Even texture
        depths.x = tex2Dlod(g_sampHiZEven, float4(uvMin.x, uvMin.y, 0, mipLevel)).r;
        depths.y = tex2Dlod(g_sampHiZEven, float4(uvMax.x, uvMin.y, 0, mipLevel)).r;
        depths.z = tex2Dlod(g_sampHiZEven, float4(uvMin.x, uvMax.y, 0, mipLevel)).r;
        depths.w = tex2Dlod(g_sampHiZEven, float4(uvMax.x, uvMax.y, 0, mipLevel)).r;
    } else {
        // Odd mip level - sample from Odd texture
        depths.x = tex2Dlod(g_sampHiZOdd, float4(uvMin.x, uvMin.y, 0, mipLevel)).r;
        depths.y = tex2Dlod(g_sampHiZOdd, float4(uvMax.x, uvMin.y, 0, mipLevel)).r;
        depths.z = tex2Dlod(g_sampHiZOdd, float4(uvMin.x, uvMax.y, 0, mipLevel)).r;
        depths.w = tex2Dlod(g_sampHiZOdd, float4(uvMax.x, uvMax.y, 0, mipLevel)).r;
    }

    // Get maximum depth from Hi-Z (furthest occluder)
    float maxOccluderDepth = max(max(depths.x, depths.y), max(depths.z, depths.w));

    // Conservative bias to prevent self-occlusion (matches CPU culling)
    // BBoxes are already expanded by 1.25x + 10 units on CPU side (ffeshader.cpp:3625)
    // Add fixed 20 unit depth bias to match CPU culling (renderdepth.cpp:778)
    float depthBias = 20.0;

    // Compare closest point of bbox against furthest occluder + bias
    // Object is visible if its closest point is in front of or near the occluders
    bool isVisible = closestDepth <= (maxOccluderDepth + depthBias);

    // DEBUG: Encode depth values + mip level for inspection
    // R: closestDepth / 1000.0, G: maxOccluderDepth / 1000.0, B: mipLevel / 10.0, A: isVisible
    return float4(
        saturate(closestDepth / 1000.0),
        saturate(maxOccluderDepth / 1000.0),
        saturate(mipLevel / 10.0),
        isVisible ? 1.0 : 0.0
    );

    // NORMAL: Return visibility as white (visible) or black (culled)
    // return isVisible ? float4(1, 1, 1, 1) : float4(0, 0, 0, 0);
}

technique HiZCull
{
    pass P0
    {
        VertexShader = compile vs_3_0 VS();
        PixelShader = compile ps_3_0 PS();

        ZEnable = FALSE;
        ZWriteEnable = FALSE;
        AlphaBlendEnable = FALSE;
        CullMode = NONE;
    }
}
