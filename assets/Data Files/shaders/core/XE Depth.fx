
// XE Depth.fx
// MGE XE 0.16.0
// Depth buffer render sequence

#include "XE Common.fx"



//------------------------------------------------------------
// Core-mod code is inserted here

#include "XE Mod Shadow.fx"
#include "XE Mod Statics.fx"
#include "XE Mod Landscape.fx"
#include "XE Mod Grass.fx"

//------------------------------------------------------------
// Floating point clears

struct DepthClearOut {
    float4 depth : COLOR0;
    float4 velocity : COLOR1;
};

float4 DepthClearVS(float4 pos : POSITION) : POSITION {
    return pos;
}

DepthClearOut DepthClearPS(float4 pos : POSITION) {
    DepthClearOut OUT;
    OUT.depth = 1.0e38;
    OUT.velocity = float4(0, 0, 0, 0);  // Zero velocity
    return OUT;
}

//------------------------------------------------------------
// Depth render

DepthVertOut DepthMWVS(MorrowindVertIn IN) {
    DepthVertOut OUT;
    float4 viewpos, prevViewpos;
    float4 pos = IN.pos;

    // Apply wind animation to match HLSL pipeline exactly
    if(vertexBlendState < 0.5) { // Rigid vertex (non-skinned)
#ifdef HAS_GRASS
        // Use grass displacement for grass geometry
        float3 displacement = grassDisplacement(IN.pos.xyz, IN.pos.z, 2.5);
        pos.xy += (1 - IN.color.z) * displacement.xy;
#else
        // Apply wind animation to alpha-tested geometry (trees, bushes, etc.)
        if (hasAlpha) {
            float3 displacement = grassDisplacement(IN.pos.xyz, IN.pos.z, 1.0);
            pos.xyz += displacement;
        }
#endif
    }

    // Skin mesh if required
    if(hasBones) {
        viewpos = skin(pos, IN.blendweights);
        // For skinned: track world motion by computing delta between current and prev world position
        // skin() applies bones in view space, so we compute world motion separately
        float4 curOrigin = mul(float4(0,0,0,1), vertexBlendPalette[0]);
        float4 prevOrigin = mul(float4(0,0,0,1), prevVertexBlendPalette[0]);
        float4 worldDelta = curOrigin - prevOrigin;
        prevViewpos = viewpos - worldDelta;  // Apply inverse of world motion
    } else {
        viewpos = mul(pos, vertexBlendPalette[0]);
        prevViewpos = mul(pos, prevVertexBlendPalette[0]);
    }

    // Fragment colour routing
    OUT.alpha = vertexMaterial(IN.color).a;

    // Transform and output depth
    OUT.pos = mul(viewpos, proj);
    OUT.depth = OUT.pos.w;
    OUT.texcoords = IN.texcoords;

    // Velocity: current and previous clip positions
    OUT.curClip = OUT.pos;
    OUT.prevClip = mul(prevViewpos, proj);

    return OUT;
}

struct DepthVelocityOut {
    float4 depth : COLOR0;
    float4 velocity : COLOR1;
};

DepthVelocityOut DepthNearPS(DepthVertOut IN) {
    DepthVelocityOut OUT;

    clip(nearViewRange + 64.0 - IN.depth);

    // Respect alpha test
    if(hasAlpha) {
        float alpha = IN.alpha * tex2D(sampBaseTex, IN.texcoords).a;
        clip(alpha - alphaRef);
    }

    OUT.depth = IN.depth;

    // Compute screen-space velocity (in NDC, range -1 to 1)
    float2 curScreen = IN.curClip.xy / IN.curClip.w;
    float2 prevScreen = IN.prevClip.xy / IN.prevClip.w;
    float2 velocity = curScreen - prevScreen;

    // Encode velocity with pow3 for better small-velocity precision (John Chapman technique)
    // This redistributes precision toward small velocities where banding is most visible
    float2 sign_v = sign(velocity);
    float2 encoded = sign_v * pow(abs(velocity), 3.0);

    // Scale for storage (decoded in post-process shader)
    OUT.velocity = float4(encoded * 50.0, 0, 1);

    return OUT;
}

//------------------------------------------------------------
// Displaced terrain depth (near-patch subdivision path)
//
// Mirrors the displacement block in core-hlsl/XE FixedFuncEmu_VS.hlsl so the
// depth buffer Z matches the color pass exactly. Required for SSAO/DOF on
// displaced ground: with a flat depth, sub-surface rocks would appear in front
// of terrain in depth-only checks, producing AO halos and DoF wrong-occlusion.
//
// Input vertices come from the same VB the color path uses (sp->vb): standard
// FVF + an extra TEXCOORD1 float2 carrying base/overlay heights pre-baked by
// patch_displacement.cpp. `world` and `displacementFalloff` are set per-draw /
// per-frame on the shared effect pool (see renderdepth.cpp).
struct DepthDispVertIn {
    float4 pos : POSITION;
    float4 normal : NORMAL;
    float4 color : COLOR0;
    float2 texcoords : TEXCOORD0;
    float2 heights : TEXCOORD1;
};

DepthVertOut DepthMWDisplacedVS(DepthDispVertIn IN) {
    DepthVertOut OUT;
    float4 pos = IN.pos;

    // Lerp base/overlay heights by alpha grid (matches color VS HAS_OVERLAY path).
    // Edge verts facing non-subdivided neighbors carry zeroed heights, so they
    // collapse to the original surface and stay crack-free.
    float displaceH = lerp(IN.heights.x, IN.heights.y, IN.color.a);

    // Same camera-distance falloff as color VS: full displacement inside R_inner,
    // smooth ramp to zero at R_outer. distXY uses the world matrix set per-draw.
    float2 worldXY = mul(float4(pos.xyz, 1), world).xy;
    float distXY = length(worldXY - displacementFalloff.zw);
    float fall = 1.0 - smoothstep(displacementFalloff.y, displacementFalloff.x, distXY);
    pos.z += displaceH * fall;

    // Rigid transform via worldview (terrain is non-skinned).
    float4 viewpos = mul(pos, vertexBlendPalette[0]);
    float4 prevViewpos = mul(pos, prevVertexBlendPalette[0]);

    OUT.alpha = 1.0;
    OUT.pos = mul(viewpos, proj);
    OUT.depth = OUT.pos.w;
    OUT.texcoords = IN.texcoords;

    // Velocity (terrain is static, so this should be zero)
    OUT.curClip = OUT.pos;
    OUT.prevClip = mul(prevViewpos, proj);

    return OUT;
}

//-----------------------------------------------------------------------------

Technique T0 {
    //------------------------------------------------------------
    // Used for clearing scene depth
    Pass D0 {
        ZEnable = false;
        ZWriteEnable = false;
        CullMode = CW;
        ClipPlaneEnable = 0;
        FillMode = Solid;

        AlphaBlendEnable = false;
        AlphaTestEnable = false;
        StencilEnable = false;
        FogEnable = false;
        Lighting = false;

        VertexShader = compile vs_3_0 DepthClearVS();
        PixelShader = compile ps_3_0 DepthClearPS();
    }
    //------------------------------------------------------------
    // Used for rendering scene depth
    Pass D1 {
        ZEnable = true;
        ZWriteEnable = true;
        ZFunc = LessEqual;
        CullMode = CW;
        ClipPlaneEnable = 0;
        FillMode = Solid;

        AlphaBlendEnable = false;
        AlphaTestEnable = false;
        StencilEnable = false;
        FogEnable = false;
        Lighting = false;

        VertexShader = compile vs_3_0 DepthMWVS();
        PixelShader = compile ps_3_0 DepthNearPS();
    }
    //------------------------------------------------------------
    // Used for rendering distant land depth
    Pass D2 {
        ZEnable = true;
        ZWriteEnable = true;
        CullMode = CW;

        VertexShader = compile vs_3_0 DepthLandVS();
        PixelShader = compile ps_3_0 DepthLandPS();
    }
   //------------------------------------------------------------
   // Used for rendering distant statics depth
    Pass D3 {
        ZEnable = true;
        ZWriteEnable = true;
        CullMode = CW;

        VertexShader = compile vs_3_0 DepthStaticVS();
        PixelShader = compile ps_3_0 DepthStaticPS();
    }
   //------------------------------------------------------------
   // Used for rendering grass depth
    Pass D4i {
        ZEnable = true;
        ZWriteEnable = true;
        CullMode = none;

        AlphaBlendEnable = false;
        AlphaTestEnable = true;
        AlphaFunc = GreaterEqual;
        AlphaRef = 128;

        VertexShader = compile vs_3_0 DepthGrassInstVS();
        PixelShader = compile ps_3_0 DepthGrassInstPS();
    }
   //------------------------------------------------------------
   // Used for rendering displaced near-patch terrain depth
    Pass D5 {
        ZEnable = true;
        ZWriteEnable = true;
        ZFunc = LessEqual;
        CullMode = CW;
        ClipPlaneEnable = 0;
        FillMode = Solid;

        AlphaBlendEnable = false;
        AlphaTestEnable = false;
        StencilEnable = false;
        FogEnable = false;
        Lighting = false;

        VertexShader = compile vs_3_0 DepthMWDisplacedVS();
        PixelShader = compile ps_3_0 DepthNearPS();
    }
   //------------------------------------------------------------
}
