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

#ifdef HAS_DISPLACEMENT
// Phase 8.4: XY = (R_outer, R_inner) in world units; ZW = world-space camera XY.
// Smoothstep from R_inner->R_outer fades signed displacement to 0 at the outer boundary
// so subdivided tiles keep original terrain height where they meet flat terrain.
float4 displacementFalloffVS : register(c73);
#endif

//------------------------------------------------------------
// Vertex Input/Output
struct VS_INPUT {
    float4 pos : POSITION;
    float3 normal : NORMAL;
    float4 color : COLOR0;
    float2 texcoord : TEXCOORD0;
#ifndef USE_STATELESS_BATCH
    float4 blendweights : BLENDWEIGHT;
#endif
#ifdef USE_STATELESS_BATCH
    // Draw index for texture lookup (appended to vertex data)
    // Note: Using TEXCOORD7 as TEXCOORD8+ may not work reliably on all hardware
    float drawIndex : TEXCOORD7;
#endif
#ifdef HAS_DISPLACEMENT
    // Pre-baked per-vertex signed displacement on the subdivided near-camera patch.
    // x=base displacement, y=overlay displacement. Perimeter edge-locked verts carry 0.
    float2 heights : TEXCOORD1;
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
    float2 screenUV : TEXCOORD7;  // Screen-space UV for forward prepass sampling
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




// Fog calculation. Linear ramp matching DL's near-fog formula (and MW's classic
// linear fog), so the horizon reaches full fog at MW view distance — same point
// DL begins blending in. nearFogStart/nearFogRange come from c49/c50 pushed
// per-replay from s_staging.fogNearStart / fogNearEnd.
float fogMWScalar(float dist) {
    return saturate((nearFogRange - dist) / max(nearFogRange - nearFogStart, 1.0));
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

    // Hoisted so shadow-coord path can undo displacement (see HAS_SHADOWS block).
    // Stays 0 when HAS_DISPLACEMENT is not defined, so the subtract below is a no-op.
    float _displaceH = 0;
#ifdef HAS_DISPLACEMENT
    // CPU path: signed displacements pre-baked into the VB in [-scale, scale];
    // edge-locked verts carry 0 so original terrain height is preserved.
    #ifdef HAS_OVERLAY
    _displaceH = lerp(input.heights.x, input.heights.y, input.color.a);
    #else
    _displaceH = input.heights.x;
    #endif
    // Phase 8.4: planar XY distance from camera, smoothstep fade R_inner->R_outer.
    // Inside R_inner: full signed displacement. Past R_outer: zero, so the outer ring
    // joins the flat non-subdivided terrain at the original height.
    float2 worldXY = mul(float4(input.pos.xyz, 1), world).xy;
    float distXY = length(worldXY - displacementFalloffVS.zw);
    float fall = 1.0 - smoothstep(displacementFalloffVS.y, displacementFalloffVS.x, distXY);
    _displaceH *= fall;

    // Phase 8.3: displace along object-space Z (world up/down — terrain has no
    // rotation) rather than normalize(input.normal). Adjacent patches have
    // distinct per-vertex normals at the shared edge; normalizing them produces
    // different directions for the same value, cracking the seam. Pure Z
    // guarantees identical offsets from identical values -> seamless.
    input.pos.z += _displaceH;
#endif

    // Calculate world position first (needed for shadows)
    worldpos = float4(input.pos.xyz, 1);
	
	// if(vertexBlendState.x > 0.5)
		// input.pos.xyz += 100; 

    // Standard transformation for non-grass
#ifndef USE_STATELESS_BATCH
    if (vertexBlendState.x > 0.5) {
        // Skinned vertex
        viewpos = skin(input.pos, input.blendweights);
        normal = skin(float4(input.normal, 0), input.blendweights).xyz;
        // Don't calculate deformed world position - treat as rigid for shadow receiving
        // This simplifies shadow calculations while maintaining visual quality
		// worldpos = skinWorld(input.pos, input.blendweights);
    }
    else
#endif
    {
        // Rigid vertex

#ifdef USE_STATELESS_BATCH
        // Stateless batching: read worldView matrix from texture using drawIndex
        // WorldView is pre-combined on CPU for better precision at distance
        // Texture is 16 wide (16 texels per draw), height = maxDraws
        // drawDataParams.x = 1/16 = 0.0625 (texel width in UV)
        // drawDataParams.y = 1/height (texel height in UV)
        float drawV = (input.drawIndex + 0.5) * drawDataParams.y;  // Center of row
        float texelW = drawDataParams.x;  // 0.0625 for 16-wide texture

        // Sample visibility flag from texel 8 (normres.w)
        // If visibility == 0, output degenerate position to cull this draw
        float4 normresData = tex2Dlod(sampDrawData, float4(8.5 * texelW, drawV, 0, 0));
        float visibility = normresData.w;

        // Sample worldview matrix columns (texels 0, 1, 2) - pre-combined for Z precision
        float4 wv0 = tex2Dlod(sampDrawData, float4(0.5 * texelW, drawV, 0, 0));
        float4 wv1 = tex2Dlod(sampDrawData, float4(1.5 * texelW, drawV, 0, 0));
        float4 wv2 = tex2Dlod(sampDrawData, float4(2.5 * texelW, drawV, 0, 0));
        // Sample 4th column from texel 7 for complete w computation
        float4 wv3 = tex2Dlod(sampDrawData, float4(7.5 * texelW, drawV, 0, 0));

        // Transform using dot products - must match mul(pos, worldview) exactly
        // wv0={_11,_21,_31,_41}, wv1={_12,_22,_32,_42}, wv2={_13,_23,_33,_43}, wv3={_14,_24,_34,_44}
        viewpos.x = dot(input.pos, wv0);
        viewpos.y = dot(input.pos, wv1);
        viewpos.z = dot(input.pos, wv2);
        // Full 4th column dot product for robust w computation
        viewpos.w = dot(input.pos, wv3);

        // Guard against zero w from texture sampling errors (prevents vertex explosion)
        if (abs(viewpos.w) < 0.001) viewpos.w = 1.0;

        // Cull invisible draws by setting position behind camera (will be clipped)
        // This is more efficient than clip() in PS because entire triangles are culled early
        if (visibility < 0.5) {
            viewpos.z = -10000.0;  // Far behind near plane, will be clipped
        }

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
    output.screenUV = output.position.xy / max(output.position.w, 1e-6) * float2(0.5, -0.5) + 0.5;

    // Pass through data
    output.normal = normalize(normal);
    output.color = input.color;
    output.texcoord = input.texcoord;
    output.viewPos = viewpos.xyz;

    // Calculate fog
    float dist = length(viewpos);
    output.fog = fogMWScalar(dist);

#ifdef HAS_SHADOWS
    // Shadow coordinate calculation. Two-cascade policy for HAS_DISPLACEMENT:
    //   Near (cascade 0): sample at the original terrain height so receiver matches
    //     flat non-subdivided terrain exactly. PCF absorbs residual mismatch.
    //   Far  (cascade 1): keep the displaced worldpos. Far cascade sampler is
    //     essentially single-tap ESM over low-LOD distant-land casters; exact baseline
    //     alignment there exposes LOD averaging as hard stripes.
    #ifdef USE_STATELESS_BATCH
        // TODO: Stateless terrain with displacement also needs _displaceH removed
        // from viewpos before shadow0 transform (displacement is along object-space
        // Z, so subtract _displaceH * wv2 where wv2 is the Z column of worldview).
        // Left as-is for now: the standard path covers the dominant terrain case.
        output.shadow0pos = mul(viewpos, shadowWorldViewProj[0]);

        // Far cascade: use per-draw (world × shadowViewproj[1]) baked into the
        // draw-data texture (texels 9-12). Avoids the worldview→inverseView round
        // trip used by the global c64-c67 constant — that path drops FP32 bits at
        // Morrowind-scale translations and shows up as far-cascade bias artifacts.
        // input.pos is post-displacement (HAS_DISPLACEMENT branch above), matching
        // non-stateless policy of keeping displaced position on the far cascade.
        float texelW_s = drawDataParams.x;
        float drawV_s = (input.drawIndex + 0.5) * drawDataParams.y;
        float4 ws0 = tex2Dlod(sampDrawData, float4( 9.5 * texelW_s, drawV_s, 0, 0));
        float4 ws1 = tex2Dlod(sampDrawData, float4(10.5 * texelW_s, drawV_s, 0, 0));
        float4 ws2 = tex2Dlod(sampDrawData, float4(11.5 * texelW_s, drawV_s, 0, 0));
        float4 ws3 = tex2Dlod(sampDrawData, float4(12.5 * texelW_s, drawV_s, 0, 0));
        output.shadow1pos.x = dot(input.pos, ws0);
        output.shadow1pos.y = dot(input.pos, ws1);
        output.shadow1pos.z = dot(input.pos, ws2);
        output.shadow1pos.w = dot(input.pos, ws3);
    #else
        float4 shadowObjPos0 = worldpos;
        shadowObjPos0.z -= _displaceH;
        // Toward-sun height bias (world +Z): keeps receiver above the softening
        // blur's raised baseline so terrain cannot self-shadow at small scale.
        // Tuned: 12 units clears softening variance without visibly lifting
        // hill-to-valley shadows (their deltas are much larger than this).
        // Note: object-onto-terrain float is a separate world-space sink gap,
        // not a bias issue — see project_terrain_shadow_policy memory.
        shadowObjPos0.z += 12.0;
        output.shadow0pos = mul(shadowObjPos0, shadowWorldViewProj[0]);
        output.shadow1pos = mul(worldpos, shadowWorldViewProj[1]);
    #endif
    output.shadow0pos.z = output.shadow0pos.z / output.shadow0pos.w;
    output.shadow1pos.z = output.shadow1pos.z / output.shadow1pos.w;
#endif

#ifdef USE_STATELESS_BATCH
    // Pass draw index to pixel shader for material lookup
    output.drawIndex = input.drawIndex;
#endif

    return output;
}
