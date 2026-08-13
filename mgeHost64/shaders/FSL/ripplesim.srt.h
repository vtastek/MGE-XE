// mgeHost64 — R2 actor-ripple wave SIMULATION SRT.
//
// WHY A SIM AND NOT THE CLOSED FORM. R1's rain ripples are analytic on purpose (see
// waterripple.h.fsl): a raindrop field covers the whole visible sea, and a grid there cannot
// degrade with distance, so it aliases where a closed form quietly turns itself into roughness.
// Actor wakes are the opposite case. They are LOCAL to the camera, they need dozens of impulses
// alive at once, and evaluating each analytically costs the water frag one loop iteration per
// ripple PER PIXEL. Measured: water=0.31 ms with no ripples against 32.81 ms with them, and the
// spike only lands on frames with a wake in view — which is why it read as frame JITTER rather
// than as a uniform cost. Micro-optimising that loop (hoisting the per-ripple hash host-side,
// tightening the reject to the packet annulus) did not touch the dominant term, because a
// REJECTING iteration still pays a cbuffer load, a dot and a compare for every water pixel.
//
// A sim inverts the cost: two dispatches over a fixed grid, and the water frag does ONE fetch.
// The price is independent of how many ripples exist, so 64 impulses cost what 0 impulses cost.
//
// ⚠ It does NOT, on its own, buy the Kelvin wedge — an earlier version of this comment claimed a
// moving impulse train through "a real wave equation" would interfere into one. Interference is
// necessary but not sufficient: the wedge is a DISPERSION effect and the scheme below has none.
// See the two-grid note further down.
//
// Scheme: evanw/webgl-water's finite difference, via the Mirza Beig Wave-Simulator reference in
// C:\projects\fastwatersim.
//     v += avg(4-neighbourhood) - h;  v *= decay;  h += v * speed;
// ⚠ speed <= 2.0 is a STABILITY limit (CFL), not a taste knob. Above it the explicit integrator
// diverges and the grid detonates into NaN within a few frames.
//
// ⚠ PING-PONG, WHERE THE REFERENCE UPDATES IN PLACE. The reference binds one RWTexture2D and reads
// its neighbours while other threads are writing them — a data race whose result depends on
// dispatch order and occupancy. Diffusion hides it well enough to ship in a demo; it is still
// undefined, and on a different driver it is a different wave equation. Reading prev and writing
// next costs one more 1024² target and removes the hazard outright.
//
// LAYOUT, both targets: .x height   .y velocity   .zw slope (d/dx, d/dy)
// The slope pair is exactly what ripplePacket returns for the analytic path, so water.frag's
// existing normal-perturbation slot takes it unchanged.
//
// ⚠ WHAT MAKES A WAKE A WEDGE, because two attempts got this wrong in two different ways.
//
// A moving source in a NON-dispersive medium has exactly two available shapes: nested circles when
// it is slower than the waves, and a Mach cone of half-angle asin(c/V) when it is faster. Nothing
// in between, and never a fixed-angle wedge. A true Kelvin wedge (19.47 deg regardless of speed) is
// a DISPERSION effect — deep-water gravity waves obey omega^2 = g|k|, long waves outrun short ones,
// group velocity is half phase velocity, and that ratio is the angle.
//
// Attempt 1 (ripplesim.comp) was non-dispersive AND subcritical: 1 texel/step x 2 steps/frame at
// 1 unit/texel is 120 u/s at 60 Hz and 330 u/s at 165, against a swimmer at 80-170 u/s. Circles,
// unavoidably, at every setting of every knob.
// Attempt 2 (ripplewave.comp) made it dispersive, which is the physically correct answer and still
// did not read as a wedge in play.
// What MGE itself shipped (ripplemge.comp) is non-dispersive — the SAME equation as attempt 1 — but
// runs at a FIXED 80 Hz with 2.5 units/texel and a = 0.14, i.e. 75 world units/s, comfortably below
// swim speed. Supercritical source, Mach cone, visible V. The lesson is that the frame-rate-locked
// timestep was never cosmetic: it is what fixes the wave speed, and the wave speed is the effect.
//
// THREE step shaders and one slope pass now share this SRT (hizreduce/reflectmip's pattern):
//   ripplesim.comp    — scroll + inject + integrate, non-dispersive, per-STEP rates.
//   ripplewave.comp   — the dispersive |k| operator (tools/iwave_kernel.py). Seconds, not steps.
//   ripplemge.comp    — MGE's own sim: same Laplacian, fixed 80 Hz, moving pinned-ring obstacle.
//   ripplenormal.comp — central differences over next.x -> next.zw. A separate dispatch because
//                       slopes need the NEW height of the neighbours, which does not exist until
//                       every thread of the step above has retired. Runs over EVERY grid.
//
// ⚠ .y MEANS DIFFERENT THINGS PER STEP SHADER: a velocity in ripplesim/ripplewave, u(t-1) in
// ripplemge (MGE's second-order form). Only .x and .zw are common, and .zw is all water.frag reads,
// so a grid may switch step shaders freely — but its field must be CLEARED when it does.
//
// Resource names are unique across all merged compute SRTs (house rule since the d3d.py aliasing
// bug), and there is exactly ONE SRT per header ([[project_forge_srt_one_per_header]]).
#pragma once

// Impulse cap. Matches IPC::kMaxActorRipples and the host's kMaxActorRipples; entries beyond the
// live count are ignored via gRippleSimParams.impulseCount rather than by zero-filling.
#ifndef RIPPLE_MAX_IMPULSES
#define RIPPLE_MAX_IMPULSES 64
#endif

STRUCT(RippleSimParams)
{
    // x = grid size (texels, square)   y = speed (<= 2.0)   z = decay (0..1 retained per step)
    // w = impulse count this frame
    DATA(float4, sim,       None);
    // Scroll in TEXELS from the previous frame's domain origin to this frame's, as a whole number.
    // xy = shift, zw = unused. The domain origin is snapped to the texel grid host-side precisely
    // so this is an INTEGER: a fractional scroll would need a resample every frame, and resampling
    // a wave field every frame is a low-pass filter running at 60 Hz — the waves would smear away
    // within a second of walking. An integer shift is an exact copy.
    DATA(float4, scroll,    None);
    // R2c (ripplewave.comp) ONLY; ripplesim.comp ignores it.
    //   x = gravity, s^-2. Derived, not tuned: g_SI / (alpha * metresPerTexel), where alpha is the
    //       kernel's measured mean R/|k|. That is what makes the wake's WAVELENGTH physical, and
    //       the wavelength is what the wedge is built out of.
    //   y = this sub-step's dt in SECONDS   z = velocity retention this sub-step = exp(-damp * dt)
    // ⚠ SECONDS, where sim.y/sim.z above are PER STEP. Not a style difference: a per-step rate makes
    // the fine grid evolve 2.75x faster at 165 Hz than at 60, which is survivable for splash detail
    // but not for a wake whose wavelength is set by gravity and the swimmer's speed.
    DATA(float4, wave,      None);
    // R2d (ripplemge.comp) only: (minX, minY, maxX, maxY) in TEXELS, the bounding rectangle of this
    // frame's obstacle rings already expanded by their radius. A rejecting source still costs a
    // length() and a compare, and almost no texel is near any actor, so four compares in front of
    // the loop is what keeps the pin proportional to the disturbed area instead of to the grid.
    DATA(float4, srcBounds, None);
    // Per impulse: xy = position in TEXEL coordinates of THIS frame's grid, z = radius (texels),
    // w = amplitude. Injected once at birth (the client flags births; see RippleSource::slot).
    // Shared by both grids — the same births drive each, at their own texel scale.
    DATA(float4, impulses[RIPPLE_MAX_IMPULSES], None);
};

BEGIN_SRT(RippleSimSrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_CBUFFER  (Persistent, CBUFFER(RippleSimParams), gRippleSimParams)
        DECL_RWTEXTURE(Persistent, RTex2D(float4), gRippleSimPrev)   // read-only this dispatch
        DECL_RWTEXTURE(Persistent, RWTex2D(float4), gRippleSimNext)  // written; RW for the normal pass
    END_SRT_SET(Persistent)
END_SRT(RippleSimSrtData)
