//============================================================================
// MGE XE - HLSL Fixed Function Emulation - Vertex Shader
//============================================================================

// Matrices - explicit registers for consistent layout regardless of #ifdefs
matrix proj : register(c0);           // c0-c3
matrix worldview : register(c4);      // c4-c7
matrix world : register(c8);          // c8-c11
matrix view : register(c12);          // c12-c15
matrix vertexBlendPalette[4] : register(c16);      // c16-c31 View-space bone transforms (world * view)
matrix vertexBlendPaletteWorld[4] : register(c32); // c32-c47 World-space bone transforms (world only)
float4 vertexBlendState : register(c48);

// Fog parameters
float nearFogStart : register(c49);
float nearFogRange : register(c50);

// Animation parameters (available for all geometry)
float2 windVec : register(c51);
float time : register(c52);
bool hasAlpha : register(c53);  // For detecting alpha-tested geometry

#ifdef HAS_GRASS
// Additional grass-specific parameters
float3 eyePos : register(c54);
float2 footPos : register(c55);
#endif

#ifdef HAS_SHADOWS
// World-to-shadow matrices for proper shadow coordinate calculation
matrix shadowWorldViewProj[2] : register(c60); // c60-c67
#endif

#ifdef USE_STATELESS_BATCH
// Draw data texture: 8 texels wide x maxDraws tall, A32B32G32R32F
// Texel layout per draw (128 bytes = 8 texels):
//   0-2: World matrix rows (transposed)
//   3: Material diffuse RGBA
//   4: Material ambient RGBA
//   5: Emissive RGB, alphaRef in w
//   6: Light params
//   7: Flags
// Note: Vertex texture sampler uses D3DVERTEXTEXTURESAMPLER0 (sampler 0 in VS)
sampler sampDrawData : register(s0);
float4 drawDataParams : register(c70);  // {1/width=0.125, 1/height, 0, 0}
#endif

//------------------------------------------------------------
// Vertex Input/Output
struct VS_INPUT {
    float4 pos : POSITION;
    float3 normal : NORMAL;
    float4 color : COLOR0;
    float2 texcoord : TEXCOORD0;
    float4 blendweights : BLENDWEIGHT;
#ifdef USE_INSTANCING
    // Instance world matrix columns from stream 1 (using high TEXCOORD indices to avoid conflicts)
    float4 instWorld0 : TEXCOORD8;   // Column 1: {_11, _21, _31, _41}
    float4 instWorld1 : TEXCOORD9;   // Column 2: {_12, _22, _32, _42}
    float4 instWorld2 : TEXCOORD10;  // Column 3: {_13, _23, _33, _43}
#endif
#ifdef USE_STATELESS_BATCH
    // Draw index for texture lookup (stream 1)
    float drawIndex : TEXCOORD8;
#endif
};

struct VS_OUTPUT {
    float4 position : POSITION;
    float3 normal : TEXCOORD0;
    float4 color : COLOR0;
    float2 texcoord : TEXCOORD1;
    float3 viewPos : TEXCOORD2;  // View space position
    float fog : TEXCOORD3;       // Fog factor
#ifdef HAS_SHADOWS
    float4 shadow0pos : TEXCOORD4;  // Shadow map 0 position
    float4 shadow1pos : TEXCOORD5;  // Shadow map 1 position
#endif
#ifdef USE_STATELESS_BATCH
    float drawIndex : TEXCOORD6;  // Draw index for PS material lookup
#endif
};

//------------------------------------------------------------
// Utility Functions

float saturate(float x) { return max(0.0, min(x, 1.0)); }

// Skinning function for view space
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

// Skinning function for world space
float4 skinWorld(float4 pos, float4 blend) {
    float blendState = vertexBlendState.x;

    // Calculate missing blend weights
    if (blendState == 1)
        blend.y = 1 - blend.x;
    else if (blendState == 2)
        blend.z = 1 - (blend.x + blend.y);
    else if (blendState == 3)
        blend.w = 1 - (blend.x + blend.y + blend.z);

    // Weighted blend of world matrices - ROW MAJOR (pos * matrix)
    float4 worldpos = mul(pos, vertexBlendPaletteWorld[0]) * blend.x;

    if (blendState >= 1)
        worldpos += mul(pos, vertexBlendPaletteWorld[1]) * blend.y;
    if (blendState >= 2)
        worldpos += mul(pos, vertexBlendPaletteWorld[2]) * blend.z;
    if (blendState >= 3)
        worldpos += mul(pos, vertexBlendPaletteWorld[3]) * blend.w;

    // Check if the result is valid (not NaN or extremely large)
    // If invalid, fallback to approximation using view space
    // if (any(isnan(worldpos.xyz)) || length(worldpos.xyz) > 100000.0) {
        // // Fallback: transform view position back to approximate world space
        // float4 viewpos = skin(pos, blend);
        // float4x4 viewInverse = transpose(view);
        // worldpos = mul(viewpos, viewInverse);
        // worldpos.xyz += view._41_42_43; // Add camera translation back
    // }

    return worldpos;
}




// Fog calculation
float fogMWScalar(float dist) {
    return saturate((nearFogRange - dist) / (nearFogRange - nearFogStart));
}

#ifdef HAS_GRASS
// Basic noise function for grass animation
float bnoise(float x) {
    float i = floor(x);
    float f = frac(x);
    float s = sign(frac(x / 2.0) - 0.5);
    float k = frac(i * 0.1731);
    return s * f * (f - 1.0) * ((16.0 * k - 4.0) * f * (f - 1.0) - 1.0);
}
#endif

// Grass displacement function based on wind and player proximity
float3 grassDisplacement(float3 worldpos, float h, float speed) {
    float v = length(windVec);
    float2 displace = 2 * v * 0.1 + 0.05;
    float2 harmonics = 0;

    float gtime = time * 0.1 * speed;

    float fi = 5.5;
    float cg = 1.0;
    float bi = 0.05;


    harmonics.x += abs(((fi * 1.0 + 0.03 * v) * sin(-2 * cg * (worldpos.x + worldpos.y + worldpos.z + gtime))) + bi * 1);

    harmonics.y += abs(((fi * 2.0 + 0.044 * v) * sin(-3 * cg * (worldpos.x + worldpos.y + worldpos.z + gtime))) + bi * 0.5);

    float3 stomp = 0;
#ifdef HAS_GRASS
    float d = length(worldpos.xy - footPos.xy);
    //d += pow(0.06 * max(0, footPos.- worldpos.z - 60), 2);


    if (d < 150) {
        stomp.xy = (60 / d - 0.4) * (worldpos.xy - footPos.xy);
    }
    stomp.z = 0;
#endif

    return float3(saturate(0.001 * (abs(h))) * speed * 5 * (harmonics.xy + stomp.xy), 0);
}

//------------------------------------------------------------
// Vertex Shader Main
VS_OUTPUT vs_main(VS_INPUT input) {
    VS_OUTPUT output;

    // Transform vertex
    float4 worldpos;
    float4 viewpos;
    float3 normal;

    // Calculate world position first (needed for shadows)
    worldpos = float4(input.pos.xyz, 1);
	
	// if(vertexBlendState.x > 0.5)
		// input.pos.xyz += 100; 

    // Standard transformation for non-grass
    if (vertexBlendState.x > 0.5) {
        // Skinned vertex
        viewpos = skin(input.pos, input.blendweights);
        normal = skin(float4(input.normal, 0), input.blendweights).xyz;
        // Don't calculate deformed world position - treat as rigid for shadow receiving
        // This simplifies shadow calculations while maintaining visual quality
		// worldpos = skinWorld(input.pos, input.blendweights);
    }
    else {
        // Rigid vertex

#ifdef USE_INSTANCING
        // Hardware instancing: read world matrix from vertex stream
        // Instance data stores columns: world0={_11,_21,_31,_41}, world1={_12,_22,_32,_42}, world2={_13,_23,_33,_43}
        // For pos * world: result.x = dot(pos, column1), etc.
        float3 worldpos3;
        worldpos3.x = dot(input.pos, input.instWorld0);
        worldpos3.y = dot(input.pos, input.instWorld1);
        worldpos3.z = dot(input.pos, input.instWorld2);
        worldpos = float4(worldpos3, 1);

        // Transform normal (use 3x3 rotation part only, w=0)
        float3 nrm3;
        nrm3.x = dot(float4(input.normal, 0), input.instWorld0);
        nrm3.y = dot(float4(input.normal, 0), input.instWorld1);
        nrm3.z = dot(float4(input.normal, 0), input.instWorld2);

        // Transform to view space
        viewpos = mul(worldpos, view);
        normal = mul(float4(nrm3, 0), view).xyz;
#elif defined(USE_STATELESS_BATCH)
        // Stateless batching: read worldView matrix from texture using drawIndex
        // WorldView is pre-combined on CPU for better precision at distance
        // Texture is 8 wide (8 texels per draw), height = maxDraws
        // drawDataParams.x = 1/8 = 0.125 (texel width in UV)
        // drawDataParams.y = 1/height (texel height in UV)
        float drawV = (input.drawIndex + 0.5) * drawDataParams.y;  // Center of row
        float texelW = drawDataParams.x;  // 0.125 for 8-wide texture

        // Sample worldview matrix columns (texels 0, 1, 2) - pre-combined for Z precision
        float4 wv0 = tex2Dlod(sampDrawData, float4(0.5 * texelW, drawV, 0, 0));
        float4 wv1 = tex2Dlod(sampDrawData, float4(1.5 * texelW, drawV, 0, 0));
        float4 wv2 = tex2Dlod(sampDrawData, float4(2.5 * texelW, drawV, 0, 0));
        // Sample flags texel (7) for 4th column: flags.zw = {wv._34, wv._44}
        float4 flagsData = tex2Dlod(sampDrawData, float4(7.5 * texelW, drawV, 0, 0));

        // Transform using dot products - must match mul(pos, worldview) exactly
        // wv0={_11,_21,_31,_41}, wv1={_12,_22,_32,_42}, wv2={_13,_23,_33,_43}
        viewpos.x = dot(input.pos, wv0);
        viewpos.y = dot(input.pos, wv1);
        viewpos.z = dot(input.pos, wv2);
        // Compute w from 4th column: pos.x*_14 + pos.y*_24 + pos.z*_34 + pos.w*_44
        // For affine transforms, _14=_24=0, so: w = pos.z*_34 + pos.w*_44
        viewpos.w = input.pos.z * flagsData.z + input.pos.w * flagsData.w;

        // Transform normal to view space
        normal.x = dot(float4(input.normal, 0), wv0);
        normal.y = dot(float4(input.normal, 0), wv1);
        normal.z = dot(float4(input.normal, 0), wv2);
#else
	#ifdef HAS_GRASS
        // Use grass displacement for grass geometry
        float3 displacement = grassDisplacement(input.pos.xyz, input.pos.z, 2.5);
        input.pos.xy += (1 - input.color.z) * displacement.xy;
        worldpos.xy += (1 - input.color.z) * displacement.xy;
	#else
        // Apply simple wind animation to alpha-tested geometry (trees, bushes, etc.)
        if (hasAlpha && vertexBlendState.x < 0.5) {
            float3 displacement = grassDisplacement(input.pos.xyz, input.pos.z, 1.0);
            input.pos.xyz += displacement;
            worldpos.xyz += displacement;
        }
	#endif

        viewpos = mul(input.pos, worldview);
        normal = mul(float4(input.normal, 0), worldview).xyz;
#endif

    }

    // Project to screen
    output.position = mul(viewpos, proj);

    // Pass through data
    output.normal = normalize(normal);
    output.color = input.color;
    output.texcoord = input.texcoord;
    output.viewPos = viewpos.xyz;

    // Calculate fog
    float dist = length(viewpos);
    output.fog = fogMWScalar(dist);

#ifdef HAS_SHADOWS
    // Use world-to-shadow transformation for proper shadow coordinates
    output.shadow0pos = mul(worldpos, shadowWorldViewProj[0]);
    output.shadow1pos = mul(worldpos, shadowWorldViewProj[1]);
    output.shadow0pos.z = output.shadow0pos.z / output.shadow0pos.w;
    output.shadow1pos.z = output.shadow1pos.z / output.shadow1pos.w;
#endif

#ifdef USE_STATELESS_BATCH
    // Pass draw index to pixel shader for material lookup
    output.drawIndex = input.drawIndex;
#endif

    return output;
}