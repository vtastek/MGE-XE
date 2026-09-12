// mgeHost64 — MB-1 OBJECT VELOCITY SRT.  tasks/forge-postprocess.md (motion blur).
//
// The camera-only pass (motionvectors.comp) writes a COMPLETE velocity field by reprojecting
// depth: correct for terrain, statics, grass and sky, and wrong for exactly one population —
// anything that moved under its own power. This pass redraws that population and OVERWRITES its
// pixels in the same texture.
//
// ⚠⚠ IT IS AN OVERWRITE, NOT A SECOND BUFFER, AND THE ORDER IS LOAD-BEARING. The compute must have
// already run when this draws, or the movers' vectors are the ones that get overwritten. There is
// no flag in the texture saying which pixel came from which producer, so nothing downstream can
// detect the inversion — it would simply read as "object blur does nothing", which is also what a
// broken previous-transform looks like. See the dispatch site in forgerender.cpp.
//
// ⚠ NO RENDER TARGET. gObjVelOut is a UAV written from the PIXEL shader, with mRenderTargetCount = 0
// — the same "graphics pipeline, no colour attachment" arrangement the Z-prepass already uses. The
// alternative (make pMotionVectors a RenderTarget) would mean re-plumbing the compute pass that
// writes it as a UAV, to gain nothing: the depth test below already does the only ordering this
// needs.
//
// ⚠ THE DEPTH TEST IS WHAT MAKES THE UAV WRITE SAFE. Two mover fragments that overlap in screen
// space would otherwise race, last-writer-wins with no defined order. They cannot: this pass draws
// with CMP_GEQUAL against the depth the Z-prepass already wrote, so at any pixel exactly one
// fragment — the visible one — survives to write. Remove the depth test and the pass becomes
// order-dependent, which on a GPU means frame-to-frame flicker on every overlap.
#pragma once

// Movers per batch window. 256 x 2 matrices x 64 B = 32 KB, half the 64 KB cbuffer cap, so there is
// room to grow without a second window. Morrowind's moving set is actors + doors + activators +
// bone-attached kit; a crowded cell runs dozens, so this is roughly an order of headroom. Overflow
// is SKIPPED and COUNTED (host logs it) rather than wrapped onto another draw's matrix — a wrapped
// index would give one object a stranger's velocity, which is the single worst failure this pass
// has, and the one the prevWorldFrame pairing key exists to prevent elsewhere.
#define OBJVEL_BATCH 448

STRUCT(ObjVelParams)
{
    // THIS frame's view-projection — the SAME matrix the colour and prepass draws used, jitter and
    // all. It has to be, twice over: the depth test below is GEQUAL against depth those draws wrote,
    // and the vector this pass produces has to agree with the camera pass's convention, which is
    // jittered-previous minus jittered-current (MVJittered).
    DATA(float4x4, viewProj, None);
    // LAST frame's view-projection, re-based to THIS frame's camera-relative origin — byte-for-byte
    // the matrix motionvectors.comp reprojects with (gMvParams.prevViewProjRel). Shared rather than
    // recomputed: two derivations of "last frame's camera" is two places for the origin fold-in to
    // be got wrong, and a disagreement between them would show as movers whose velocity is offset
    // from the static field around them by a constant — i.e. as a blur that does not sit on the
    // object.
    DATA(float4x4, prevViewProjRel, None);
    DATA(float4, screenParams, None);   // xy = render rect px, zw = 1/render rect px
    // x = 1 when prevViewProjRel is valid (there WAS a previous frame). 0 makes every vector zero,
    // which is the correct answer on the first frame and after a camera cut, and is what keeps a
    // teleport from streaking the whole screen.
    //
    // MB-1e: y = 1 in the FIRST-PERSON instance ONLY — "stamp the arm-depth constant into
    // gObjVelDepthOut". See that resource's note. 0 in the world instance, where the same write
    // would replace every mover's DEVICE depth with the near plane and hand water, the APL sky
    // discriminator and next frame's reprojection a lie.
    //
    // MB-2d: z = OBJECT-ONLY BLUR. 1 puts the object's motion RELATIVE TO THE WORLD into
    // gObjVelBlurOut; 0 puts the same total vector gObjVelOut gets, making that target a copy and
    // the blur the classic full-frame one.
    //
    // MB-2g: w = the MOVER-MASK threshold in render-rect px. ⚠ NOT SPARE — this comment said it
    // was, one lane after the shader started reading it as `thr`.
    DATA(float4, opts, None);
    // MB-2m: x = FREEZE THE BLUR FIELD — skip the gObjVelBlurOut store so pMbVelocity keeps the
    // last advancing frame's movers. The camera pass has the matching lane (MvParams opts.w) and
    // the two must be set together: freezing only the camera half leaves this pass writing ~zero
    // over every mover, which is exactly where the held motion was needed.
    // ⚠ THE PASS STILL RUNS. Skipping it instead would be one line in the host and would also skip
    // the FIRST-PERSON DEPTH STAMP (opts.y -> gObjVelDepthOut), so the arm's pixels would carry
    // different depth while paused and the frozen picture would be of a DIFFERENT filter than the
    // one being debugged. yzw reserved, cleared by the host BEFORE the lane above is written.
    DATA(float4, opts2, None);
};

STRUCT(ObjVelBatch)
{
    // ⚠ float4x4 IN A CBUFFER, read with mul(world, float4(pos,1)) — NOT a float4 buffer read as
    // rows. The point is to reproduce opaque.vert's clip position EXACTLY: same packing, same
    // expression, same order of operations, therefore the same bits. That is what lets the depth
    // test match the prepass instead of speckling out on the last mantissa bit. (opaque.srt.h's
    // BatchData carries the same convention note: host uploads D3DXMATRIX bytes as-is, column-major
    // cbuffer, mul(M, v) reproduces D3DX's row-vector v*M.)
    DATA(float4x4, worlds[OBJVEL_BATCH], None);
    // ...and the pose the SAME slot was drawn with last frame, de-absolutized into THIS frame's
    // camera-relative space by the host. Both matrices therefore live in one space and the only
    // difference between them is the object's own motion — which is the entire point.
    DATA(float4x4, prevWorlds[OBJVEL_BATCH], None);
};

// ═══ MB-1b: THE SKINNED HALF ════════════════════════════════════════════════════════════════
// A skinned part has no world matrix. Its position comes from a BONE PALETTE — model->world per
// bone, blended by the vertex's top-4 weights — so the batch above cannot describe it and the
// rigid vertex shader cannot draw it. It needs its own vertex stage (objvelocity_skin.vert) and
// its own pair of palettes, and that is the whole of MB-1b.
//
// ⚠⚠ OUR OWN COPY OF THE *CURRENT* PALETTE, NOT THE SHARED BONE BUFFER — and that is a
// constraint, not a preference. The palette the colour pass skins with lives in opaque.srt.h's
// PerBatch set (gBatch.worlds, rebound per window); this pass has a private SRT with its own
// PerDraw set, and two SRTs cannot both define a set without aliasing each other's registers
// ([[project_forge_srt_one_per_header]]). Copying the palette out of the same wire blob the
// colour walk reads costs tens of KB a frame and keeps this pass self-contained, which is the
// shape MB-1a already has.
//
// ⚠ AND THE COPY IS WHY TRAP 1 GOT WORSE. Because these bytes are copied rather than shared,
// "bit-identical to skinned.vert" is no longer free — it is a property the copy has to preserve
// and the shader has to reproduce. With EARLY_FRAGMENT_TESTS in the frag the depth test now
// genuinely runs BEFORE the shader, and GEQUAL absorbs a fragment that lands a hair NEARER while
// REJECTING one that lands a hair farther: a skinning expression that differs from skinned.vert
// in the last bit loses roughly half the actor's pixels to holes. See objvelocity_skin.vert.fsl,
// where the four mul/add terms are copied character for character.
#define OBJVEL_BONES 1024   // one 64 KB cbuffer window (1024 x 64 B is exactly the CBV cap)

STRUCT(ObjVelBones)
{
    // ⚠ float4x4 IN A CBUFFER, indexed [Base + BoneIdx[j]] — the SAME packing, the SAME
    // expression and the SAME order of operations skinned.vert uses on gBatch.worlds, for the
    // same reason the rigid batch above copies opaque.vert: the depth test compares this pass's
    // clip position against the one the prepass wrote, and the two only agree if the arithmetic
    // producing them is identical.
    DATA(float4x4, bones[OBJVEL_BONES], None);
};

BEGIN_SRT(ObjVelocitySrtData)
    BEGIN_SRT_SET(PerDraw)
        DECL_CBUFFER  (PerDraw, CBUFFER(ObjVelParams), gObjVelParams)
        DECL_CBUFFER  (PerDraw, CBUFFER(ObjVelBatch),  gObjVelBatch)
        // pMotionVectors, in RENDER-rect pixels, previous-minus-current — the same texture and the
        // same convention motionvectors.comp writes. WTex2D because this pass only ever writes it:
        // it has no business reading what the camera pass put there.
        DECL_RWTEXTURE(PerDraw, WTex2D(float2), gObjVelOut)
        // ⚠ A FRAGMENT COUNTER, because "the depth test is rejecting occluded movers" is otherwise
        // an assumption dressed as a design note. [0] counts fragments that SURVIVED the test and
        // wrote a vector. Run once with the test on and once with it off (objVelDepthTest=0): if the
        // two counts are the same, the test is inert and every occluded triangle is painting over
        // whatever is in front of it — which is what "the bed renders over the NPC" looks like from
        // the far side of a picture.
        DECL_RWBUFFER(PerDraw, RWBuffer(uint), gObjVelFrags)
        // MB-1b. ⚠ APPENDED, AND THEY MUST STAY APPENDED. FSL assigns descriptor offsets from one
        // running counter over the set, so inserting a resource ABOVE an existing one renumbers
        // every entry after it while the host's SRT_RES_IDX/updateDescriptorSet order silently
        // keeps the old numbering. Growth goes on the end.
        //
        // THIS frame's bone palettes, packed contiguously by this pass's OWN cursor (see the
        // record loop in forgerender.cpp — it must never touch the shared skinPackNext cursor,
        // whose bit-identical-across-five-walks invariant is what stops a part drawing with a
        // stranger's palette).
        DECL_CBUFFER  (PerDraw, CBUFFER(ObjVelBones), gObjVelBonesCur)
        // ...and LAST frame's palettes for the SAME parts, at the SAME offsets, rebased into THIS
        // frame's camera-relative space by the host.
        //
        // ⚠⚠ THE REBASE IS NOT OPTIONAL AND IT IS NOT VISIBLE WHEN YOU STAND STILL. Bone matrices
        // are model->world in CAMERA-RELATIVE coordinates, so last frame's palette is relative to
        // last frame's eye, while prevViewProjRel expects a point in THIS frame's relative space
        // (it folds dBake = eye_now - eye_prev into itself). Every previous bone's translation row
        // therefore needs -dBake applied at upload. Parked, dBake is exactly zero and a missing
        // correction is undetectable; at speed and close in it slides the whole actor's previous
        // pose by the frame's camera travel ([[project_park_restamp_eye_origin]]).
        //
        // ⚠ Written at the SAME base as gObjVelBonesCur, which is why the per-instance stream
        // carries ONE Base and not a Base/PrevBase pair: this pass packs both windows itself, in
        // one pass, off one cursor. The pairing that CAN differ between frames — where last
        // frame's palette sat — is resolved on the HOST (HostMesh::prevBoneBase, recorded per
        // slot) and never reaches the shader.
        DECL_CBUFFER  (PerDraw, CBUFFER(ObjVelBones), gObjVelBonesPrev)
        // ═══ MB-1e: THE ARM'S DEPTH, FOR MB-2's SOFT-DEPTH WEIGHT ═══════════════════════════
        // ⚠ APPENDED, for the reason stated above — FSL numbers descriptors off ONE running
        // counter per set, so a resource inserted above this one silently renumbers it.
        //
        // pLinearDepth, whose name is a leftover: it holds RAW REVERSE-Z DEVICE depth (
        // linearizedepth.comp.fsl says so outright). MB-2's gather reads it as gMbDepth, and the
        // arms are not in it — pLinearDepth's last write is the seam re-linearize, long before the
        // FP pass draws — so over an arm pixel mbDepthWeight compares the WALL BEHIND THE ARM
        // against its neighbours, goes inert, and only the velocity-agreement weights hold the
        // silhouette. That is the reported bleed across the arms when strafing with hands still.
        //
        // ⚠⚠ WHAT IS WRITTEN IS A CONSTANT, **NOT THE ARM'S OWN PROJECTED DEPTH**. The arm camera
        // has its own near/far (near=1.0 far=7168) and the arms sit INSIDE the world camera's near
        // plane — the very fact that made MB-1d's first acceptance probe blow up. Two projections
        // in one buffer are only accidentally ordered and they INVERT when a world surface is very
        // close. The constant is unconditionally true instead: the FP pass draws after a depth
        // clear, so the arms are in front of everything by construction. mbDepthWeight is a pure
        // RATIO with no projection constants, so it needs nothing else, and it then answers
        // correctly in both directions (a world centre rejects an arm sample as "in front of me";
        // an arm centre accepts world samples as "behind me"). The only thing lost is depth
        // discrimination BETWEEN arm parts, which at arm scale is what you want anyway.
        DECL_RWTEXTURE(PerDraw, WTex2D(float), gObjVelDepthOut)
        // MB-2d — THE BLUR'S VELOCITY FIELD, which is NOT gObjVelOut and must never be merged with
        // it. gObjVelOut is the TOTAL motion of each pixel and belongs to the upscaler: a temporal
        // backend handed an object-only field would reject history for the entire static world on
        // every camera turn. This one is what the blur reads, and at opts.z it holds the object's
        // motion relative to the world — zero for everything the camera merely swept past, which is
        // what makes a spin stop smearing the room.
        //
        // ⚠ ITS STATIC PIXELS ARE WRITTEN BY motionvectors.comp, NOT HERE. This pass only rasterises
        // movers, so every pixel it does not cover keeps whatever was there; the camera pass writes
        // the whole rect (zero in object-only mode) immediately before. That pairing IS the clear,
        // and it costs one store rather than a separate full-screen pass.
        DECL_RWTEXTURE(PerDraw, WTex2D(float2), gObjVelBlurOut)
    END_SRT_SET(PerDraw)
END_SRT(ObjVelocitySrtData)
