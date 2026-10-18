// mgeHost64 — DEPTH OF FIELD (DoF D0/D1).  tasks/forge-dof.md.
//
// An EYE, not a camera rig: Knu's v12 model (Depth of Field.fx) made physical. The eye accommodates
// to the focus distance s inside its range (infinity down to the near point 1/fpa), and the blur on
// the retina is the PUPIL times the DIOPTRE error:
//     theta = pupil * |D(z) - D_f|  rad (angular blur DIAMETER),  D = 1/distance (m), D_f = clamp(1/s, 0, fpa)
// In pixels that is theta * (px per radian), so nothing here is tuned per resolution or per FOV.
// Deep focus is not a setting, it falls out: focus 10 m and the sky is 0.1 dpt off.
//
// ── TWO PASSES, BOTH COMPUTE (F8 hot-reload, and no late-Z trap — motionblur.srt.h) ──────────────
//   doffocus.comp   ONE group of 8x8    -> gDofFocus[0]   autofocus: centre-window dioptres, eased
//   dofcoc.comp     one thread per px   -> gDofOut        D0: the frame unchanged, or the CoC view
// The bokeh gather (D2) replaces dofcoc's identity arm; the plumbing does not move.
//
// ⚠ THE FOCUS STATE IS A 1-ELEMENT BUFFER, NOT A 1x1 TEXTURE. Forge makes an mHeight == 1 texture
// TEXTURE1D, and a Tex2D/WTex2D view of it reads 0 and drops writes with no error — the trap that
// left the first-person eye lane dead for a week ([[project_forge_1x1_texture_is_1d]]). A buffer has
// no dimension to get wrong, and it copies straight into the readback the heartbeat reads.
//
// ⚠ DEPTH IS RAW REVERSE-Z DEVICE DEPTH (pLinearDepth's name lies; motionblur.srt.h). It is turned
// into a RADIAL distance through the frame's own inverse view-projection — the GTAO/skyvis matrix —
// so there are no projection constants to derive and no far-plane assumption. Radial, not axial,
// because an eye's retina is curved (Knu's z_corr). 0 = no surface = the sky = 0 dioptres.
#pragma once

STRUCT(DofParams)
{
    // Device depth -> camera-relative world (MW units), the SAME matrix skyvis.comp/GTAO use.
    DATA(float4x4, invViewProj, None);
    // xy = THE DELIVERED RECT (the rect read, written and clamped to); zw = THE DEPTH RECT (the
    // render/input rect pLinearDepth was written at). Equal unless an upscaler ran.
    DATA(float4, rects, None);
    // x = PUPIL diameter, metres. y = ACCOMMODATION range, dioptres (near point = 1/y m).
    // z = PX PER RADIAN at the delivered rect (screen centre), so theta * z is the blur in pixels.
    // w = MAX CoC RADIUS, delivered px (the clamp the gather will be sized for).
    DATA(float4, eye, None);
    // x = dt, SECONDS, of the sim clock (0 while a menu pauses the game: the focus holds).
    // y = accommodation RATE, 1/s (D_f moves 1 - exp(-dt*rate) of the way per frame).
    // z = OVERRIDE focus, dioptres (MWSE: the dialogue speaker, the combat target). w = its weight.
    DATA(float4, focus, None);
    // x/y = SHARP HANDS: no blur nearer than x metres, full blur from y (Knu's smoothstep 0.568-0.781;
    //       the first-person arms sit inside it). Also the autofocus ignores anything nearer than y.
    // z = metres per MW unit.  w = RESET: 1 snaps D_f to the target (first frame after a build).
    DATA(float4, hands, None);
    // x = DEBUG mode: 0 off (the identity), 1 the CoC view (far orange, near blue, green = < 1 px).
    // y = the autofocus WINDOW radius, as a fraction of the delivered height.
    // z = the CoC view's full-scale tint, in px of radius.  w reserved (cleared by the host).
    DATA(float4, opts, None);
};

BEGIN_SRT(DofSrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_CBUFFER (Persistent, CBUFFER(DofParams), gDofParams)
        // Raw reverse-Z device depth at the DEPTH rect (rects.zw). Bigger = CLOSER, 0 = sky.
        DECL_TEXTURE (Persistent, Tex2D(float),  gDofDepth)
        // The delivered image: pSceneColor (or the MSAA pre-filter's output), the upscaler's output,
        // or the motion blur's — three set instances, one pointer apart (the resolve's M1 4b trick).
        // ⚠ PREMULTIPLIED: every weight hits all four channels (motionblur.srt.h, resolve.srt.h).
        DECL_TEXTURE (Persistent, Tex2D(float4), gDofColor)
        // [0] = (D_f the eye's focus now, D_t the target, the window's weight sum, 1 = written).
        DECL_RWBUFFER(Persistent, RWBuffer(float4), gDofFocus)
        // pDof, the delivered image after the lens. Its own target, for motionblur.srt.h's reason.
        DECL_RWTEXTURE(Persistent, WTex2D(float4), gDofOut)
    END_SRT_SET(Persistent)
END_SRT(DofSrtData)
