// mgeHost64 — DEPTH OF FIELD (DoF D0/D1).  tasks/forge-dof.md.
//
// An EYE, not a camera rig: Knu's v12 model (Depth of Field.fx) made physical. The eye accommodates
// to the focus distance s inside its range (infinity down to the near point 1/fpa), and the blur on
// the retina is the PUPIL times the DIOPTRE error:
//     theta = pupil * |D(z) - D_f|  rad (angular blur DIAMETER),  D = 1/distance (m), D_f = clamp(1/s, 0, fpa)
// In pixels that is theta * (px per radian), so nothing here is tuned per resolution or per FOV.
// Deep focus is not a setting, it falls out: focus 10 m and the sky is 0.1 dpt off.
//
// ── FIVE PASSES, ALL COMPUTE (F8 hot-reload, and no late-Z trap — motionblur.srt.h) ─────────────
//   doffocus.comp   ONE group of 8x8     -> gDofFocus[0]          autofocus: centre-window dioptres, eased
//   dofcoc.comp     16x16 = one TILE     -> gDofCoc, gDofTile     signed CoC per px, max |CoC| per tile
//   doftile.comp    one thread per tile  -> gDofTileNb            max over +-k tiles (the near reach)
//   dofblurh.comp   one thread per px    -> gDofRe, gDofIm        the disc kernel's rows (complex)
//   dofblurv.comp   one thread per px    -> gDofOut               columns, recombine; the CoC view
// D2 (tasks/forge-dof.md "D2 design"): FULL RES, the kernel radius per tile, and every sample weighted
// by whether its own CoC reaches the pixel (scatter-as-gather), which is what keeps in-focus edges
// from haloing and lets the near field bleed.
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
    // x = DEBUG mode: 0 the blur, 1 the CoC view (far orange, near blue, green = < 1 px),
    //     2 the identity (D0's copy: the plumbing with no lens).
    // y = the autofocus WINDOW radius, as a fraction of the delivered height.
    // z = the CoC view's full-scale tint, in px of radius.
    // w = k, the TILE NEIGHBOURHOOD radius in tiles = ceil(max CoC / 16): how far a CoC can reach.
    DATA(float4, opts, None);
    // x = TARGET CAP (1 on): no pixel is blurred more than the focus TARGET would blur it, so a refocus
    //     only ever eases blur IN — what the eye just turned to is sharp at once, what it left blurs
    //     at the focus rate. Off = the eased focus alone (the image passes through a blurrier state
    //     than either end on a near -> far look).
    // yzw = THE EYE in invViewProj's space (MW units). Not the origin: a park fire's matrix is relative
    //     to the bake eye, so while moving the eye sits a frame's motion away — measured from the
    //     origin, near distances and the px-per-radian came out wrong (blur surging while walking).
    DATA(float4, look, None);
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
        // ─── D2's intermediates: all RW_TEXTURE, all REST in UNORDERED_ACCESS, sequenced by plain UAV
        // barriers (bloom.srt.h / motionblur.srt.h's tile pyramid). R16F and RGBA16F typed UAV loads
        // are the one hardware group the bloom pyramid already requires.
        // Signed CoC RADIUS, delivered px, + far / - near (dofCocPx), at the delivered rect.
        DECL_RWTEXTURE(Persistent, RWTex2D(float),  gDofCoc)
        // max |CoC| per 16x16 tile, and that max over +-k tiles: the kernel radius of every px in it.
        DECL_RWTEXTURE(Persistent, RWTex2D(float),  gDofTile)
        DECL_RWTEXTURE(Persistent, RWTex2D(float),  gDofTileNb)
        // The horizontal pass's complex response, per kernel component (Re/Im = component 0, Re1/Im1 =
        // component 1): Re and Im of the PREMULTIPLIED rgba (every weight hits all four channels,
        // motionblur.srt.h's rule), and of the weight itself (W = Re0, Im0, Re1, Im1), all normalised
        // by component 0's envelope sum so an HDR row stays inside fp16.
        // ⚠ Rows in a SHARP tile are not written; the vertical pass skips them by coverage, never
        // reads them.
        DECL_RWTEXTURE(Persistent, RWTex2D(float4), gDofRe)
        DECL_RWTEXTURE(Persistent, RWTex2D(float4), gDofIm)
        DECL_RWTEXTURE(Persistent, RWTex2D(float4), gDofRe1)
        DECL_RWTEXTURE(Persistent, RWTex2D(float4), gDofIm1)
        DECL_RWTEXTURE(Persistent, RWTex2D(float4), gDofW)
        // pDof, the delivered image after the lens. Its own target, for motionblur.srt.h's reason.
        DECL_RWTEXTURE(Persistent, WTex2D(float4), gDofOut)
    END_SRT_SET(Persistent)
END_SRT(DofSrtData)
