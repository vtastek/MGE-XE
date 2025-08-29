// XE FixedFuncEmu_Simple.hlsl  
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

// Lighting - Basic
float3 lightSceneAmbient;
float3 lightSunDiffuse;
float3 lightSunDirection;

// Lighting - Point lights
float4 lightDiffuse[8];  // Changed to float4 to match D3DXVECTOR4 from C++
float3 lightPosition[8]; // Proper float3 positions
float lightAmbient[8];
float lightFalloffQuadratic[8];
float lightFalloffConstant;
int pointLightCount; // Number of real point lights

// Fog
float3 fogColNear;
float nearFogStart, nearFogRange;

// Textures with explicit register bindings for DX9 HLSL
texture tex0 : register(t0);
texture tex1 : register(t1);
texture tex2 : register(t2); 
texture tex3 : register(t3);
sampler sampTex0 : register(s0) = sampler_state { texture = <tex0>; };
sampler sampTex1 : register(s1) = sampler_state { texture = <tex1>; };
sampler sampTex2 : register(s2) = sampler_state { texture = <tex2>; }; // Diffuse parameter (_diffparam)  
sampler sampTex3 : register(s3) = sampler_state { texture = <tex3>; }; // Normal map (_nh)

// Texture suffix support - using preprocessor defines
// #define HAS_DIFFPARAM
// #define HAS_NORMAL
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
    if(blendState == 1)
        blend.y = 1 - blend.x;
    else if(blendState == 2)
        blend.z = 1 - (blend.x + blend.y);
    else if(blendState == 3)
        blend.w = 1 - (blend.x + blend.y + blend.z);
    
    // Weighted blend of matrices - ROW MAJOR (pos * matrix)
    float4 viewpos = mul(pos, vertexBlendPalette[0]) * blend.x;
    
    if(blendState >= 1)
        viewpos += mul(pos, vertexBlendPalette[1]) * blend.y;
    if(blendState >= 2)
        viewpos += mul(pos, vertexBlendPalette[2]) * blend.z;
    if(blendState >= 3)
        viewpos += mul(pos, vertexBlendPalette[3]) * blend.w;
    
    return viewpos;
}

// Fog function
float fogMWScalar(float dist) {
    return saturate((nearFogRange - dist) / (nearFogRange - nearFogStart));
}

#ifdef HAS_NORMAL
// Parallax height and normal parameters
static const float parallaxScale = 0.000001;
static const float parallaxBias = 0.0000501;
static const float heightScale = 100;
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

    float det   = dtdx.x * dtdy.y - dtdx.y * dtdy.x;
    float invDet = (abs(det) > 1e-8) ? (1.0 / det) : 0.0;

    float3 T = normalize(( dpdx * dtdy.y - dpdy * dtdx.y) * invDet);
    float3 B = normalize((-dpdx * dtdy.x + dpdy * dtdx.x) * invDet);

    // TBN (rows are T, B, N) — we'll use its transpose for VS<->TS conversion
    float3x3 TBN      = float3x3(T, B, Nvs);
    float3x3 TBN_T    = transpose(TBN);

    // View dir in view space and tangent space
    float3 Vvs = normalize(viewPos);
    float3 Vts = mul(Vvs, TBN_T);

    // --- Parallax offset (simple) ---
    float2 uv = uvAdj;
    float  h  = tex2D(sampTex3, uv).a;
    float  parallax = h * parallaxScale + parallaxBias;
    uv += (parallax * Vts.xy) / max(Vts.z, 1e-3);   // shift along view ray in TS
    uvAdj = uv;

    // --- Height → normal from the same (parallaxed) UV ---
    float2 texel = 1.0 / normres;
    float hL = tex2D(sampTex3, uv + float2(-texel.x, 0)).a;
    float hR = tex2D(sampTex3, uv + float2( texel.x, 0)).a;
    float hD = tex2D(sampTex3, uv + float2(0, -texel.y)).a;
    float hU = tex2D(sampTex3, uv + float2(0,  texel.y)).a;

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
    } else {
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

float4 ps_main(VS_OUTPUT input) : COLOR {
    // Enhanced texture sampling with suffix support
    
    float3 diffuseParam = float3(1.0, 1.0, 1.0);  // Default white

    float3 deb = 0;
	
	float2 parallaxUV = input.texcoord;
    float3 normalVS = input.normal;
	
#ifdef HAS_NORMAL
    // Use ParallaxHeightNormal function for cleaner parallax and normal mapping

    ParallaxHeightNormal(input.viewPos, parallaxUV, normalVS, deb);
    
    // Update texture coordinates for main texture sampling

#endif


    
    float4 texColor = tex2D(sampTex0, parallaxUV); // Will use parallax-corrected UVs if HAS_NORMAL
	
	#ifdef HAS_DIFFPARAM
    texColor.rgb = tex2D(sampTex2, parallaxUV).rgb; // Will use parallax-corrected UVs if HAS_NORMAL
	#endif
   
    // Basic lighting calculation
    float3 lighting = lightSceneAmbient;
    
    // Sun light with parallax soft shadows
    float sunDot = saturate(dot(normalVS, -lightSunDirection));
    
    // Sun light contribution
    lighting += lightSunDiffuse * sunDot;
    
    // Point lights 
    float3 pointLightContribution = float3(0, 0, 0);
    for (int i = 0; i < pointLightCount; i++) {
        // Use proper float3 positions 
        float3 L = lightPosition[i] - input.viewPos;
        float dist = length(L);
        L = L / dist;
        
        float NdotL = saturate(dot(normalVS, L));
        
        // Use actual Effect shader falloff formula: 1/(quadratic*dist² + constant) - no linear term!
        float falloff = lightFalloffQuadratic[i] * dist * dist + lightFalloffConstant;
        float attenuation = (falloff > 0.0) ? (1.0 / falloff) : 0.0;
        
        // Add diffuse and ambient components like Effect shader
        pointLightContribution += (lightDiffuse[i].rgb * NdotL + lightAmbient[i]) * attenuation;
    }
    
    lighting += pointLightContribution;
    
    // Material calculation with diffuse parameter modulation
    float3 effectiveDiffuse;
    float3 effectiveEmissive;
    float effectiveAlpha;
    
    int materialMode = (int)shadingMode.z;
    if (materialMode == 2) {
        // Mode 2: Use vertex color for diffuse/ambient
        effectiveDiffuse = input.color.rgb;
        effectiveEmissive = materialEmissive.rgb;
        effectiveAlpha = input.color.a;
    } else if (materialMode == 3) {
        // Mode 3: Use vertex color for emissive
        effectiveDiffuse = materialDiffuse.rgb;
        effectiveEmissive = input.color.rgb;
        effectiveAlpha = materialDiffuse.a;
    } else {
        // Mode 1: Use material constants
        effectiveDiffuse = materialDiffuse.rgb;
        effectiveEmissive = materialEmissive.rgb;
        effectiveAlpha = materialDiffuse.a;
    }
    
    float3 litColor = effectiveDiffuse * lighting;
    litColor += effectiveEmissive;
    
    float4 diffuse = float4(litColor, effectiveAlpha);
    
    // Apply base texture with enhanced material properties
    float4 c = diffuse * texColor;
	
	#ifdef HAS_NORMAL
	//c.rgb = float3(1,0,1);
	#endif
	  
    // Apply fog
    // c.rgb = lerp(fogColNear, c.rgb, input.fog);
    
    return c;
}