// mgeHost64 — S: the LOCAL WATER SIM (tasks/forge-water-3d.md, FINAL DIRECTION).
//
// "I want to see water locally displace, the realtime sim component, water being pushed and pulled
// by gravity, and the speed being the whitewater driver." (user, 2026-10-09) — and its extent:
// "the detailed push pull sim doesn't need to be kms. more like 20-50 meters."
//
// MODEL: the shallow-water equations on a camera-centred grid (192² at 16 u = 3072 u, ~44 m),
// staggered (MAC): a texel holds the surface height eta at its centre and the velocity on its +x and
// +y faces. Two dispatches per step, ping-ponged like the ripple grids (ripplesim.srt.h):
//   pass 0 (velocity) — gravity pulls each face along the surface slope (-g grad eta), plus the
//                       pressure of wind gusts and of swimmers; scroll + clear happen here.
//   pass 1 (height)   — eta moves by the divergence of the flux u * h_face; whitewater mass and the
//                       horizontal displacement integrate here.
// WET/DRY is LISFLOOD-style: a face flows only where the water above the HIGHER of its two beds is
// deeper than a film, so a wave can run up a beach and drain back, and a cell is never emptied
// below its own ground. The flux uses min(h_face, Hmax) — the depth that SETS the wave speed — so
// open water carries waves at sqrt(g*Hmax) while the shallows slow them (shoaling, refraction).
//
// Bathymetry is gSkyHeight (top-down max of terrain and statics, the GI/sky-AO window): piers and
// rocks are walls. Body weights (gWaterBodies) scale the forcing: sea and beach as tuned, rivers
// gentler, ponds a small wind chop. Rivers do not run a current through the sim (no inflow exists
// at a 44 m window): their chop is a LOOPING flow-advected pattern the sim relaxes toward ("a river
// can be a few frames repeating", user 2026-10-09).
//
// LAYOUT
//   S (RGBA32F): x = eta (surface - water plane; on a DRY cell it is its ground, so the bed is never
//                undercut), y = u on the +x face, z = v on the +y face, w = spare.
//   D (RGBA16F): x,y = horizontal displacement (integrated velocity, relaxing to 0), z = DISPLAY height
//                (eta where wet, 0 where dry — what water.vert lifts the mesh by), w = whitewater mass.
// The finished field is always in index 0 of each pair (pass 1 writes it), so the PerFrame binding
// for water.vert/water.frag is static.
#pragma once

#ifndef SW_MAX_IMPULSES
#define SW_MAX_IMPULSES 16
#endif

STRUCT(SwSimParams)
{
    // x = grid side (texels)   y = units per texel   z = dt (s)   w = pass (0 velocity, 1 height)
    DATA(float4, grid,    None);
    // xy = integer texel shift prev -> this frame's origin (pass 0 of the frame's first step only;
    //      >= grid = CLEAR: every fetch is out of range and returns the rest state)
    // z  = gravity (u/s^2)   w = water plane Z (absolute)
    DATA(float4, scroll,  None);
    // x = Hmax (depth that sets the wave speed)  y = velocity damping (1/s)
    // z = sponge width (texels)  w = sponge relax rate (1/s)
    DATA(float4, phys,    None);
    // xy = ABSOLUTE world XY of texel (0,0)'s corner   z = forcing time (s, wrapped)   w = impulse count
    DATA(float4, origin,  None);
    // gSkyHeight addressing: xy = world origin, z = 1/extent, w = texels per side (0 = no map: deep)
    DATA(float4, skyMap,  None);
    // gWaterBodies / gWaterFlow addressing: xy = world origin, zw = 1/extent (0 = no map: open sea)
    DATA(float4, bodyMap, None);
    // x,y = wind heading (unit)   z = swell amplitude (u) at the sponge   w = swell wavelength (u)
    DATA(float4, wind,    None);
    // x = gust pressure amplitude (u of head)  y = gust scale (u)  z = river chop amp (u)
    // w = river loop period (s; divides the 20 s water clock)
    DATA(float4, chop,    None);
    // whitewater: x = speed threshold (u/s)  y = speed gain (1/s)  z = decay (1/s)
    //             w = convergence gain (per 1/s of convergence)
    DATA(float4, ww,      None);
    // x = displacement relax (1/s)   y = river flow speed (u/s at strength 1)
    // z = DEEP HOLD: share of the sponge rate applied everywhere the water is deeper than 2*Hmax, so
    //     the open water keeps its waves all the way to the eye; the shallows run free (shoaling,
    //     refraction, run-up are theirs)   w = bottom friction coefficient (thin sheets)
    DATA(float4, disp,    None);
    // per-body amplitude (sea, river, pond, beach) — gShadowParams.waterBodyAmp's values
    DATA(float4, bodyAmp, None);
    // BREAKING (user 2026-10-09: "when break happens, it swings in place ... breaking wave speed is
    // enough to travel more there, then swings back as it dissipates. it is okay to penetrate more land")
    //   x = momentum ADVECTION share (0 = the linear v1; 1 = full u.grad u — what lets a broken wave
    //       run on as a bore instead of sloshing in place)
    //   y = breaking ratio: a crest higher than y * still depth breaks
    //   z = whitewater produced per second by a fully breaking cell
    //   w = spare
    DATA(float4, brk,     None);
    // per impulse: xy = position in TEXELS of this frame's grid, z = radius (texels),
    // w = pressure head (u; a moving swimmer is a moving dip in the surface pressure)
    DATA(float4, impulses[SW_MAX_IMPULSES], None);
};

BEGIN_SRT(SwSimSrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_CBUFFER  (Persistent, CBUFFER(SwSimParams), gSwParams)
        DECL_RWTEXTURE(Persistent, RTex2D(float4),  gSwPrevS)
        DECL_RWTEXTURE(Persistent, RTex2D(float4),  gSwPrevD)
        DECL_RWTEXTURE(Persistent, RWTex2D(float4), gSwNextS)
        DECL_RWTEXTURE(Persistent, RWTex2D(float4), gSwNextD)
        DECL_TEXTURE  (Persistent, Tex2D(float),    gSwSkyHeight)   // raw pSkyHeight (bed)
        DECL_TEXTURE  (Persistent, Tex2D(float4),   gSwBodies)      // gWaterBodies
        DECL_TEXTURE  (Persistent, Tex2D(float4),   gSwFlow)        // gWaterFlow
    END_SRT_SET(Persistent)
END_SRT(SwSimSrtData)
