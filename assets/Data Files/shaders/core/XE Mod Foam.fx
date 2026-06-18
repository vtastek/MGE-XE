
// XE Mod Foam.fx
// MGE XE — world-anchored hybrid Voronoi particle foam simulation.
//
// Ported and hybridised from the shadertoy "Voronoi particle tracking"
// (MichaelMoroz, fork of "Voronoi Beach Waves"). Lagrangian particles are stored
// one-per-texel as (position.xy, velocity.zw) and tracked with an 8-neighbour
// Voronoi nearest-particle step. foam = clamp(k * |vorticity|) of the resulting
// velocity field, so foam emerges exactly where the flow converges / shears
// (choke points) — it is an advected 2D buffer field, NOT a 3D-noise lookup, so
// it does not alias.
//
// The hybrid changes vs the demo:
//   - Particle positions are stored in WINDOW TEXEL coordinates and the whole
//     buffer is world-anchored: the C++ driver (simulateFoam) tracks the player
//     with a texel-aligned StretchRect shift, so foam stays world-locked as the
//     camera moves (the demo was screen/mouse anchored). foamShiftPx carries the
//     per-substep texel shift; the advect pass subtracts it from every stored
//     position so window-local coords survive the shift (the "re-bin fixup").
//   - The advecting velocity field is SEEDED FROM THE FLOW MAP (texFlow sampled
//     at the particle's WORLD position) instead of the demo's pressure-only /
//     beach-wave field, so foam rides real river current.
//   - The player injects an outward velocity kick (foamPlayer) so moving through
//     the river spawns foam, not just a ripple wake.
//
// Shares the flow uniforms (texFlow / flowMapTransform / flowMapWeight / sampFlow)
// and foamOrigin declared in XE Mod Water.fx — this file MUST be #included AFTER
// it and must NOT redeclare them (the multiple-declaration / register hazard).

#ifdef WATER_FOAM

// !!! MUST EQUAL DistantLand::foamTexResolution in distantland.h !!! This is a hard cross-file
// coupling: the sim stores particle positions in [0,foamSize] texel space and maps them to world
// via foamOrigin + pos*foamWorldRes, so a value larger than the physical RT scales the whole
// foam window (coordinate mismatch — foam lands at the wrong world position). Keep in lockstep.
static const float  foamRes       = 512.0;
static const float2 foamSize      = float2(512.0, 512.0);
static const float  foamRcpRes    = 1.0 / 512.0;
// World units per foam texel — now a uniform, set per cascade from C++ (foamCascadeWorldRes).
shared float foamWorldRes;
// Per-microstep particle advance in texels (set from C++, kept ≤1 so the 8-neighbour Voronoi
// tracker can always follow the particle). The driver runs ceil(totalAdvance) micro-substeps
// so the per-frame world advance is unchanged — see simulateFoam.
shared float foamAdvance;

// Sim constants (the live, tunable subset rides foamParams from C++).
shared float foamMinDensity;   // respawn when nearest particle > 1/foamMinDensity texels away (FoamDens knob)
shared float foamGaussRadius;  // particle splat radius → foam blob size/softness (FoamGauss knob)

// New sim inputs (the shared flow + foamOrigin uniforms come from XE Mod Water.fx).
shared texture texFoamParticles;   // ping-pong particle buffer: xy = window texel pos, zw = velocity
shared texture texFoamFieldIn;     // smoothed field buffer: xy = velocity, z = density
shared float2  foamShiftPx;        // integer window texel shift this substep (re-bin fixup; 0 except substep 0)
shared float2  foamFieldShift;     // FULL per-frame window shift (texels), every substep: the field buffer
                                   // is NOT re-binned, so it lags the (re-binned) particles by this much. The
                                   // pressure read offsets by it to undo the lag (kills movement-fed brightness).
shared float3  foamPlayer;         // xy = player window texel pos, z = in-water flag (1 = inject)
shared float4  foamParams;         // x = flow force (texels/substep), y = velocity decay, z = pressure, w = foam scale

sampler sampFoamParticles = sampler_state { texture = <texFoamParticles>; minfilter = point;  magfilter = point;  mipfilter = none; addressu = clamp; addressv = clamp; };
sampler sampFoamField     = sampler_state { texture = <texFoamFieldIn>;   minfilter = linear; magfilter = linear; mipfilter = none; addressu = clamp; addressv = clamp; };

// Advected detail-UV offset field (River Editor technique): per-texel world-space offset that is
// transported through the velocity field, so the consume's foam texture follows curved river flow.
shared texture texFoamUVIn;        // ping-pong source (xy = world-unit offset)
shared float   foamUVRate;         // UV transport rate through the velocity field
shared float   foamUVDecay;        // relax offset toward 0 to bound stretch (no hard reset)
sampler sampFoamUV = sampler_state { texture = <texFoamUVIn>; minfilter = linear; magfilter = linear; mipfilter = none; addressu = clamp; addressv = clamp; };

float foamGauss(float2 x, float r)
{
    float d = length(x) / r;
    return exp(-d * d);
}

// Sample the flow map at a foam-window texel position. Returns xy = downstream flow
// DIRECTION (rg*2-1) and z = RIVER/BEACH routing strength = saturate((0.5 - flow.a)*2)
// (the same "near" classifier the normal advection uses). Foam membership keys on z.
//   - NOT length(xy): direction encodes neutral as 0.5, so black/no-data (rg=0) decodes to
//     (-1,-1), |·|=1.41 → reads as MAX flow.
//   - NOT flow.b: B is wave AMPLITUDE and is kFlowWaveDefault (1/3 → saturate(B*3)=1) in
//     dry/uncovered water, so it foams everywhere (the dry-corner "tile" generators).
// A is the routing class: 0.5 neutral/dry, <0.5 river+beach (near), >0.5 sea.
float3 foamFlowAt(float2 ppos)
{
    float2 worldXY = foamOrigin + ppos * foamWorldRes;
    float2 uv = (worldXY - flowMapTransform.xy) * flowMapTransform.zw;
    float4 flow = tex2Dlod(sampFlow, float4(uv, 0, 0));

    // Near (river/beach) routing strength. The soften+expand box-blur is now BAKED into the
    // routing alpha (buildWaterFlowMap step 5c-bis, foamRiverBlur radius), so a single tap of the
    // pre-softened alpha replaces the old runtime 3x3 blur.
    float riverStr = saturate((0.5 - flow.a) * 2.0);

    return float3(flow.rg * 2 - 1, riverStr);
}

//------------------------------------------------------------
// PASS_FOAM_ADVECT — Voronoi particle tracking + flow-seeded advection.

float4 FoamAdvectPS(float2 tex : TEXCOORD0) : COLOR0
{
    float2 pos = tex * foamSize;

    float4 U = tex2Dlod(sampFoamParticles, float4(tex, 0, 0));
    U.xy -= foamShiftPx;   // re-bin: bring the stored position into the shifted window

    // 8-neighbour Voronoi: keep whichever stored particle is closest to this texel.
    [unroll] for (int n = 0; n < 8; ++n)
    {
        const float2 offs[8] = { float2(-1,0), float2(1,0), float2(0,-1), float2(0,1),
                                 float2(-1,-1), float2(1,1), float2(1,-1), float2(-1,1) };
        float4 Un = tex2Dlod(sampFoamParticles, float4(tex + offs[n] * foamRcpRes, 0, 0));
        Un.xy -= foamShiftPx;
        float2 r1 = pos - Un.xy;
        float2 r2 = pos - U.xy;
        if (dot(r1, r1) < dot(r2, r2)) U = Un;
    }

    // Respawn to hold density: if the nearest particle drifted too far, seed one here.
    float2 d = pos - U.xy;
    if (dot(d, d) > (1.0 / foamMinDensity) * (1.0 / foamMinDensity))
        U.xy = pos;

    float2 ppos = U.xy;

    // Flow-seeded advection velocity (river current at the particle's world position).
    // Gate on the per-body intensity (fl.z), NOT length(direction) — black/no-data decodes to
    // a spurious full-magnitude direction, which would drive false advection outside rivers.
    float3 fl        = foamFlowAt(ppos);
    float2 flowDir   = fl.xy;
    float  riverGate = fl.z * flowMapWeight;   // 0 outside river/beach (and when A/B off)
    float2 flowVel   = flowDir * foamParams.x * riverGate;

    // Density pressure (pile-up at narrows): gradient of the field density channel. Sample the
    // (un-re-binned) field at the particle's OLD-window position = ppos + foamFieldShift, so the
    // pressure stays aligned with the scrolled particles — otherwise the lag feeds a spurious
    // velocity under motion that accumulates through the decay integrator (foam self-brightens).
    float2 fuv = (ppos + foamFieldShift) * foamRcpRes;
    float  dL = tex2Dlod(sampFoamField, float4(fuv - float2(foamRcpRes, 0), 0, 0)).z;
    float  dR = tex2Dlod(sampFoamField, float4(fuv + float2(foamRcpRes, 0), 0, 0)).z;
    float  dD = tex2Dlod(sampFoamField, float4(fuv - float2(0, foamRcpRes), 0, 0)).z;
    float  dU = tex2Dlod(sampFoamField, float4(fuv + float2(0, foamRcpRes), 0, 0)).z;
    float2 pressure = float2(dR - dL, dU - dD);

    // Player injection: outward velocity kick → foam ring around the swimmer.
    if (foamPlayer.z > 0.5)
    {
        float  k = foamGauss(ppos - foamPlayer.xy, 8.0);
        float2 outDir = normalize(ppos - foamPlayer.xy + float2(1e-4, 1e-4));
        U.zw = lerp(U.zw, outDir * foamParams.x * 1.5, k);
    }

    // Relax velocity toward the flow current (bounded), add density pile-up.
    float2 vel = lerp(flowVel, U.zw, foamParams.y);
    vel += foamParams.z * pressure;
    // Confine ALL velocity to rivers: outside river/beach the pressure term + residual would
    // keep the field churning (the "action outside the river" in the field debug), feeding back
    // for nothing. Gate by riverGate so particles sit still where there is no flow.
    vel *= riverGate;
    U.zw = vel;                            // FULL velocity drives the curl (foam intensity)

    // Advance position at a slower rate than the stored velocity — tames the visual flow
    // speed without scaling the velocity (which would collapse the curl and kill foam).
    // foamAdvance is the per-microstep texel rate (set per cascade from C++). Hard-cap the
    // resulting step at 1 texel: the 8-neighbour Voronoi search can only track a particle
    // that moves ≤1 texel/substep, so a longer jump would break tracking and force a respawn
    // (the "slow / camera-changes-foam / edge-erasing" failure at fine worldRes). The driver
    // runs enough micro-substeps that the per-frame world advance is preserved.
    float2 step = vel * foamAdvance;
    float  stepLen = length(step);
    step *= min(stepLen, 1.0) / max(stepLen, 1e-5);
    U.xy = ppos + step;
    U.xy = clamp(U.xy, 0.0, foamSize - 1.0);   // world-anchored window: clamp, don't wrap

    return U;
}

//------------------------------------------------------------
// PASS_FOAM_FIELD — smooth the Voronoi velocity + density into a field buffer.

float4 FoamFieldPS(float2 tex : TEXCOORD0) : COLOR0
{
    float2 pos = tex * foamSize;

    float4 P = tex2Dlod(sampFoamParticles, float4(tex, 0, 0));
    float  dens = foamGauss(pos - P.xy, 0.7 * foamGaussRadius);

    // 3x3 velocity blur for a smoother curl field.
    float2 vsum = 0;
    [unroll] for (int dy = -1; dy <= 1; ++dy)
     [unroll] for (int dx = -1; dx <= 1; ++dx)
        vsum += tex2Dlod(sampFoamParticles, float4(tex + float2(dx, dy) * foamRcpRes, 0, 0)).zw;
    float2 vavg = vsum / 9.0;

    return float4(vavg, dens, 0);
}

//------------------------------------------------------------
// PASS_FOAM_EXTRACT — foam = clamp(k * |vorticity|), masked to rivers → texFoam.

float4 FoamExtractPS(float2 tex : TEXCOORD0) : COLOR0
{
    // Measure vorticity over a FIXED WORLD distance (~25u) rather than 1 texel, so foam
    // intensity is invariant to the buffer resolution (at 512/25u this is fd=1 = the
    // original per-texel stencil; at 2048/6.25u it is fd=4 = the same 25u span).
    float  fd = max(1.0, floor(25.0 / foamWorldRes + 0.5));
    float2 ex = float2(fd * foamRcpRes, 0);
    float vyR = tex2Dlod(sampFoamField, float4(tex + ex.xy, 0, 0)).y;
    float vyL = tex2Dlod(sampFoamField, float4(tex - ex.xy, 0, 0)).y;
    float vxU = tex2Dlod(sampFoamField, float4(tex + ex.yx, 0, 0)).x;
    float vxD = tex2Dlod(sampFoamField, float4(tex - ex.yx, 0, 0)).x;
    float vorticity = (vyR - vyL) - (vxU - vxD);

    float foam = saturate(foamParams.w * abs(vorticity));

    // River mask: per-body wave intensity (fl.z), NOT direction magnitude — black/no-data
    // would otherwise decode to full magnitude and foam the no-flow edges (backwards).
    float  riverMask = foamFlowAt(tex * foamSize).z * flowMapWeight;

    return float4((foam * riverMask).xxx, 1);
}

//------------------------------------------------------------
// PASS_FOAM_UV — advect the foam-detail UV offset through the velocity field (River Editor).
// Backward semi-Lagrangian transport + decay relaxation: the stored world-space offset follows
// the curved flow so the consume's foam texture flows along bends without shearing, and the decay
// bounds the offset (no unbounded stretch) without a hard reset/pop.

float4 FoamUVPS(float2 tex : TEXCOORD0) : COLOR0
{
    float2 v = tex2Dlod(sampFoamField, float4(tex, 0, 0)).xy;   // velocity (texels/substep)
    // Fetch the offset from upstream (where this parcel came from), then keep it attached to the
    // moving parcel by subtracting the world displacement it just travelled.
    float2 back   = tex - v * foamRcpRes * foamUVRate;
    float2 off    = tex2Dlod(sampFoamUV, float4(back, 0, 0)).xy;
    float2 dWorld = v * foamWorldRes * foamUVRate;
    off -= dWorld;
    off *= foamUVDecay;                                          // relax toward identity (bounds stretch)
    return float4(off, 0, 0);
}

#endif
