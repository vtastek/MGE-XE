// mgeHost64 — G7 GRASS PLASTICITY: the world-locked CRUSH FIELD. One SRT for all three passes.
//
// WHY THIS EXISTS AT ALL, and it is a gameplay bug rather than a look. Host-owned grass (G1-G6) is
// denser and taller than MGE's ever was, and nothing in the world displaces it — so a dead body
// lying in a sward is simply invisible, including a quest body that in vanilla lies in plain view.
// That is a regression the render takeover introduced.
//
// ⚠ THE PRIOR ART CANNOT FIX IT, and each of its four defects is fatal on its own. MGE's stomp
// (`core/XE Mod Grass.fx:20-26`) is
//     d = length(worldpos.xy - footPos.xy);  if (d < 150) stomp = (60/d - 0.4) * (worldpos.xy - footPos.xy);
// which (1) knows exactly ONE point and that point is the PLAYER — no NPC, no creature, above all no
// CORPSE; (2) pushes SIDEWAYS, never down, which parts grass around a pole where a body PRESSES it;
// (3) has no SHAPE, and a body is ~130x40 units whose silhouette is exactly what must be cleared;
// (4) has no MEMORY, so grass springs back the instant the foot leaves. Real trampled grass stays
// down, and that persistence is the "plasticity" this ticket is named for.
//
// ─── THE MODEL ────────────────────────────────────────────────────────────────────────────────────
//
// Built exactly the way the caustics are built, and the analogy is STRUCTURAL rather than a mood: a
// camera-following domain snapped to whole texels, a scatter through a `uint` atomic accumulator, and
// a resolve into the texture consumers sample. See causticsplat.h.fsl for the accumulator pattern and
// RippleGrid / advanceRippleGrid (forgerender.cpp) for the scroll-and-snap discipline.
//
// ⚠⚠ WHAT THE FIELD STORES IS A CLEARANCE HEIGHT, NOT A STOMP AMOUNT, AND THAT ONE CHOICE DOES ALL
// THE WORK. Per texel: the absolute world Z of the LOWEST OCCUPANT SURFACE over that texel. A blade
// whose tip would rise above that height is folded until the tip sits at it. Out of that single
// scalar, with no case analysis anywhere:
//
//   corpse lying down   -> clearance ~= the ground over the whole silhouette (the spine sits ~12 up
//                          and DROP takes it to the body's underside), so the patch goes flat and the
//                          body stands clear of it.
//   standing NPC        -> ankle bones sit at ground level, pelvis at ~70, head at ~130, so grass is
//                          crushed AT THE FEET and merely brushed under the torso. Grass does stand
//                          between your legs — thinned outward by the skirt, not mown.
//   kneeling / sitting  -> no case needed; the skeleton says where the low surfaces are.
//   a dropped item      -> likewise.
//
// ⚠ NO "IS THIS ACTOR DEAD?" QUERY IS EVER NEEDED, and death state is not on the wire and does not
// have to be: a prone skeleton IS a low skeleton, and the field only ever asks how low.
//
// ⚠ PLASTICITY IS THE ASYMMETRY: FAST DOWN, SLOW UP. The splat is a `min` toward the target; the
// release is an order of magnitude slower. That asymmetry is also what makes the quest case safe — a
// body you have never approached is pressed flat within a few frames of entering the skinned list,
// long before you can walk to it.
//
// ─── THE SPRINGBACK, AND WHY IT LEFT THE HEIGHT AXIS ──────────────────────────────────────────────
//
// The heal used to raise the stored CLEARANCE at a fixed rate in world units per second. That is a
// conveyor belt, not a spring: one speed the whole way, no acceleration, no overshoot, an abrupt
// stop. It was also blade-height dependent for no reason anyone chose — the same 12 u/s takes four
// times as long to release a 251-unit blade as a 60-unit one, because what a blade responds to is
// clearance OVER ITS OWN HEIGHT, not clearance.
//
// ⚠⚠ SO .w STOPPED BEING A 0/1 FLAG AND BECAME THE CLOCK. It is a RELEASE ENVELOPE: pinned at 1 while
// a crusher sits on the texel, falling linearly to 0 over healTime once nothing does. u = 1 - w is
// then normalised time since release, and the blade's whole response is a CLOSED FORM the vertex
// shader evaluates from that one number. No velocity to store, no second texture, no ping-pong, no
// per-blade state anywhere.
//
// ⚠ AND THE ELASTIC HALF IS THE MAIN MOTION, NOT A PERTURBATION ON THE PLASTIC ONE. The first build
// of this had it backwards — a slow plastic ramp with a small ring hung off it — and it does not
// spring at all: the ring dies inside the first eighth of the heal, during which the blade is still
// lying flat, so the fold only ever moved a quarter either side of "pressed". Released grass is the
// other way round: the elastic recovery is nearly all of the travel and it is fast, and what remains
// afterwards is a small plastic residue that relaxes slowly. grass.vert carries the arithmetic.
//
// ⚠ AND IT COSTS NOTHING IN SAFETY, which is why this channel and not another. .w = 0 still means
// "no crush", so an UNBOUND texture still reads as the pre-G7 image. The flag was only ever using two
// of the values a float can hold; the envelope uses the rest of them.
//
// ⚠ AND THE CLOCK RUNS FROM 2, NOT 1, BECAUSE A RELEASED BLADE DOES NOT MOVE AT ONCE. Grass held
// under a load takes a set, and the set takes time to let go of — so the region .w in [1,2] is a
// DWELL in which the blade is still fully pressed and nothing is recovering yet, and only [0,1] is
// the springback. The vertex shader needs no case for it: u = saturate(1 - w) is already 0 for every
// w >= 1, so the dwell falls out of the saturate that was there for other reasons.
//
// How much dwell a texel gets is CHARGED while it is held (grassCrushChargeTime) and scaled by the
// crusher's WEIGHT, so a heavy thing resting a long while leaves grass that stays down after it
// goes, and a foot passing through does not. The charge is an AtomicMax over the ordered-uint
// encoding — every value here is >= 1 so the bit pattern is monotone — which keeps the scatter
// commutative: the heaviest, longest-resting crusher over a texel wins it, in any order.
//
// ⚠ THE SEED'S STEP DOUBLES ABOVE 1.0, AND THAT IS fp16'S EXPONENT, NOT A FUDGE. ulp is 2^-11 in
// [0.5,1) and 2^-10 in [1,2), so a step that is one ulp in the recovery region is HALF an ulp in the
// dwell region and would be swallowed whole — the clock would stall at whatever charge it reached
// and the imprint would never begin to heal. Doubling it there is walking the value in its own
// representable steps.
//
// ⚠ THE CLEARANCE NEVER MOVES ON ITS OWN. .x is the HELD clearance — where the occupant pressed it
// to — and it stays there until the envelope retires the texel. All the healing is in .w.
// One consequence worth stating plainly: the heal is now the same DURATION for every blade where it
// used to be the same SPEED in world units, and duration is what a viewer actually perceives.
//
// ─── THE PROFILE OF ONE CRUSHER: A DOME WITH A SKIRT, NOT A FLAT DISC ─────────────────────────────
//
// The first build splatted each bone as a flat disc at the bone's own Z, and the argument for it was
// that a skeleton is a point cloud of low surfaces whose union IS the silhouette. That is true of the
// silhouette and false of everything around it, and the failure it produced was specific and
// reproducible: BIG ANIMALS READ WELL AND HUMANOID NPCS DID NOT.
//
// The reason is that the pressed patch is only ever as wide as the BONE CLOUD'S XY EXTENT plus one
// disc radius, and the two body plans differ in exactly that. A guar's bones are spread over ~200 x
// 80 units, so the union of discs is a patch far larger than any blade is tall and the animal stands
// clear of it. A standing humanoid's bones are a nearly VERTICAL LINE: forty-odd bones stacked over
// one ~30 x 30 footprint, of which only the ankles are low enough to press anything at all. The
// result is a ~50-unit clearing around the feet with full-height grass at its rim — grass that, from
// any normal camera, stands in front of the legs and the torso. "Grass sprouting from them."
//
// And widening the disc did not fix it, which is the useful half of the observation: a wider FLAT
// disc is a crop circle. It has a hard rim at whatever radius it is given, and it presses everything
// inside that radius equally hard whether the body is touching it or not. The knob existed, it moved,
// and it made the image worse. What was missing was not radius, it was PROFILE.
//
// So a crusher is now a shape rather than a stamp, and it has two parts:
//
//   DROP, inside the contact radius. A bone translation is a JOINT CENTRE — a point on the limb's
//   centre line — and this field means the lowest occupant SURFACE. The underside of a limb is its
//   translation minus its radius, so that is where the clearance goes. grassCrushBoneRadius now does
//   both halves of one job: how far the limb reaches in XY, and how far its surface sits below the
//   bone. A prone corpse's spine sits ~12 units up and the body's underside is on the ground; before
//   this, the grass under it was pressed to 12 and stood up THROUGH the body.
//
//   SLOPE, outside it. Beyond the contact radius the clearance RISES, linearly, out to a falloff
//   distance. This is the half that was missing entirely, and it is not a fudge: a body moving
//   through grass leans and parts the blades it never touches, and the disturbance tapers with
//   distance. A cone is the cheapest honest shape for that taper, and it self-limits — once the
//   clearance climbs past a blade's own height that blade is simply not folded, so the skirt fades
//   out on its own instead of ending at a rim.
//
// The two together are what a standing humanoid needed: fully flat where the feet actually are, and
// a broad graded bowl around them that thins the canopy without mowing it. Falloff = 0 restores the
// flat disc exactly, which is the A/B.
//
// ⚠ AND THE SKIRT FIXES THE LAY DIRECTION FOR FREE. The resolve reads lay off grad(z), so inside a
// flat disc the gradient VANISHES and every blade in the patch compressed straight down with no
// direction at all; only the rim texels had a gradient to read. A sloped skirt has a radial gradient
// everywhere, so the whole bowl now lays away from the body — which is what a body pushing through
// grass does, and it is why the mat reads as pressed rather than as mown.
//
// ─── THE PRESS IS NOT INSTANT ─────────────────────────────────────────────────────────────────────
//
// A `min` straight to the target lands the whole press in ONE FRAME, and a one-frame press is
// invisible: the eye is never shown grass going down, only grass that is already down, which reads as
// a bald patch that was always there rather than as something a body did. So the splat eases toward
// its target instead — `target + remaining * exp(-dt / pressTime)` — which is fast enough to satisfy
// the quest case (a tenth of a second) and slow enough to be legible as a crush.
//
// ⚠ THE EASE IS STILL A `min`, AND IT HAS TO BE. The rate-limited value is MONOTONE in the target, so
// the minimum over overlapping crushers of the eased values equals the eased value of the minimum —
// two bones over one texel still commute, still need no ordering, and still need no lock. That is the
// only property this whole design rests on, and the ease was chosen to preserve it.
//
// ⚠ THE DESCENT NEEDS LAST FRAME'S CLEARANCE, and the splat cannot read it from the field texture:
// the texture is SCROLLED (the address it wants is not the address it has) and pass 3 writes the very
// texels other threads would be reading. Pass 1 already does that fetch, in the one pass where it is
// safe, so it parks the answer in a second lane of the accumulator. See gGrassCrushAccum below.
//
// ─── THE ORDERED-UINT KEY ─────────────────────────────────────────────────────────────────────────
//
// For NON-NEGATIVE IEEE floats the bit pattern is monotone under unsigned compare, so a plain
// integer AtomicMin over `asuint(z + BIAS)` is a float min. BIAS is 32768, which puts every Morrowind
// Z (the deepest map geometry is a few thousand units below zero) safely positive; the clamp at zero
// is belt-and-braces against a wild value rather than a case that happens.
//
// The "nothing here" SENTINEL is z = 60000, above any MW geometry and below the bias's headroom. It
// is a value and not a flag precisely so that `min` needs no special case: an empty texel loses every
// comparison it is ever in.
//
// ─── WHY .x IS STORED RELATIVE TO A SNAPPED crushOriginZ ──────────────────────────────────────────
//
// The published texture is RGBA16F. Absolute MW Z reaches ~15,000 and fp16's ulp there is 8 units,
// which would quantise a 15-unit clearance into nothing at all. Relative to an origin snapped to a
// 512-unit grid around the eye the range is +-512 and the ulp is <= 0.125 near the eye. SNAPPED
// rather than tracking the eye exactly, so the re-bias fires only on a boundary crossing and fp16
// rounding cannot accumulate one frame's error into the next.
//
// ⚠ AND VALIDITY RIDES IN .w RATHER THAN BEING INFERRED FROM A MAGNITUDE. "No crush here" cannot be
// encoded as "a very large .x": the stored value is CLAMPED to the fp16-friendly +-512 window, and a
// blade standing on a hillside 900 units above the eye would then read a sentinel as a clearance
// BELOW its own root and be crushed by nothing. So .w is 1 where a crusher was actually found and 0
// everywhere else, and grass.vert tests it before it does anything at all. An unbound texture reads
// zero, which is .w = 0 = no crush = exactly the pre-G7 image — the house convention, pointing the
// right way for once without an inversion trick (contrast gCausticField, which had to store gain-1).
//
// ⚠ EMPTY TEXELS STILL CARRY .x = +REL_MAX, and that is not redundancy with .w. grass.vert samples
// BILINEARLY, so a tap on the edge of a pressed patch blends both channels and what .x holds on the
// empty side is what decides the halo around every imprint. Zero there would read as "clearance = the
// snapped origin" — an arbitrary height up to 512 units below the eye — and the ring around each
// imprint would be crushed HARDER than the imprint, by an amount that changes as the player walks
// uphill. +REL_MAX fades the crush out instead, and does it independently of where the camera is.
//
// ─── NO PING-PONG ANYWHERE ────────────────────────────────────────────────────────────────────────
//
// Unlike RippleGrid, which needs a pair because its step reads a neighbourhood of the field it is
// writing. Here the read and the write are DIFFERENT RESOURCES at every step: pass 1 reads the
// texture and writes the buffer, pass 3 reads the buffer and writes the texture. One texture, one
// buffer, no parity to track and no descriptor-set orientation to get wrong.
//
// Resource names are unique across all merged compute SRTs (house rule since the d3d.py aliasing
// bug), which is why the compute-side texture is gGrassCrushField and the graphics-side SRV in
// opaque.srt.h is gGrassCrush. There is exactly ONE SRT per header ([[project_forge_srt_one_per_header]]).
#pragma once

// The bias that makes the float bit pattern a valid unsigned sort key, and the empty-texel sentinel.
// ⚠ NOT MIRRORED — the host DERIVES kGrassCrushZBias / kGrassCrushSentinel / kGrassCrushZSnap from
// these three macros, because this header is on its include list too. Nothing can static_assert
// across a C++/FSL boundary, and a mismatch here would read as "the whole field is crushed to the
// floor" rather than as an error, so the only safe arrangement is one definition and no copy.
#define GRASS_CRUSH_ZBIAS     32768.0f
#define GRASS_CRUSH_SENTINEL  60000.0f
// How far from the snapped origin the published .x may reach. fp16 ulp at 512 is 0.25; the deepest
// crush anyone can author is a body on the ground under an eye at the top of its 512-unit cell, i.e.
// well inside this. Clamping rather than letting fp16 do it keeps the value meaningful at the edge.
#define GRASS_CRUSH_REL_MAX   512.0f
// How far ABOVE its target the press may start when it lands on a texel that carried no crush at all.
// The press is rate-limited (see THE PRESS IS NOT INSTANT below) and a virgin texel reads the 60000
// sentinel, so without a bound the first descent would start 60000 units up and take a minute to
// arrive. This is the height the descent starts from instead, and the only requirement on it is that
// it be TALLER THAN THE TALLEST BLADE: the fold is driven by clearance-versus-blade-height, so a
// start above every blade is indistinguishable from a start at infinity, while a start BELOW one
// means that blade is ALREADY PARTLY FOLDED on the frame the press begins — a pop, not a press.
// ⚠ 192 was wrong for exactly that reason. The window log measures this bake at bladeH up to 251
// (def AABB top x per-blade scale), so the tallest blades snapped to 24% folded before the ease even
// started. 320 clears the measured maximum with room for a taller pack.
#define GRASS_CRUSH_PRESS_START 320.0f

// ⚠ THE ENCODE/DECODE PAIR LIVES IN grasscrushkey.h.fsl, NOT HERE, and the split is forced: this
// header is included by forgerender.cpp as well (that is how the host gets GrassCrushParams and
// SRT_RES_IDX), and asuint/asfloat are HLSL intrinsics a C++ compiler has never heard of. The three
// #defines above are plain float literals, so they cross the boundary happily — and the host DERIVES
// its own constants from them rather than restating them, which is what makes the two sides
// unable to disagree in the first place.

STRUCT(GrassCrushParams)
{
    // x,y = domain origin in ABSOLUTE WORLD units, snapped to whole texels (see advanceRippleGrid for
    //       why the snap is what makes the scroll an integer, i.e. an exact copy rather than a
    //       resample). z = world units per texel. w = grid size in texels.
    DATA(float4, domain, None);
    // x,y = the scroll from the PREVIOUS frame's domain origin to this one, in whole TEXELS.
    // z     = this frame's HEAL STEP: how much of the release envelope to retire, dt / healTime,
    //         clamped host-side. ⚠ It used to be a springback in WORLD units per second applied to
    //         the clearance; see THE SPRINGBACK below for why the heal left the height axis.
    // w     = crusher count for the splat dispatch.
    DATA(float4, sim,    None);
    // x = this frame's crushOriginZ (world). y = the PREVIOUS frame's. The stored .x is relative to
    //     the frame that wrote it, so pass 1 adds y to get absolute and pass 3 subtracts x to store.
    // z = the largest texel radius a single crusher may claim (cost bound; see the splat).
    // w = the LAY-GRADIENT clamp in world units — how far above a texel's own clearance a neighbour
    //     is allowed to count when the resolve differences them. Without it a body's edge texel
    //     differences against a 60000 sentinel and every lay direction in the field is the same
    //     saturated vector; with it the gradient is dominated by real structure.
    DATA(float4, bias,   None);
    // THE SHAPE OF ONE CRUSHER, and the two things it adds to a flat disc — see THE PROFILE above.
    // x = DROP: how far below its own translation a bone's underside sits, in world units. A bone is
    //     a centre LINE and the field means the lowest occupant SURFACE, so the clearance under a
    //     joint is its translation minus the limb's radius, not the translation.
    // y = SLOPE: world units of clearance gained per world unit of horizontal distance, OUTSIDE the
    //     contact radius. This is the skirt, and it is what a body wading through grass actually does
    //     to the blades it does not touch.
    // z = FALLOFF: how far that skirt reaches beyond the contact radius, in world units. 0 restores
    //     the flat disc exactly.
    // w = PRESS DECAY: the per-frame multiplier on the remaining distance to the target,
    //     exp(-dt / pressTime), computed host-side. 0 = the press lands in one frame (the original
    //     behaviour); toward 1 = slower. See THE PRESS IS NOT INSTANT.
    DATA(float4, shape,  None);
    // G7i — THE DWELL: the delay between a crusher leaving and the blade starting to come back up.
    // x = CHARGE STEP: how much of a full set one frame of continuous contact lays down, dt/chargeTime.
    // y = DWELL STEP: how fast the dwell region of the clock is retired, dt/dwellTime. ⚠ Floored at
    //     1/1024 host-side, which is ONE fp16 ulp in [1,2) — exactly twice the floor the recovery
    //     region needs, because that is exactly how fp16's exponent works across that boundary.
    // z = the WEIGHT REFERENCE in world units: the bone-volume proxy that counts as a full-weight
    //     crusher. Below it a crusher both charges more slowly and saturates lower.
    // w spare.
    DATA(float4, dwell,  None);
};

BEGIN_SRT(GrassCrushSrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_CBUFFER  (Persistent, CBUFFER(GrassCrushParams), gGrassCrushParams)
        // The crushers, TWO float4s each (stride 2, so crusher k is at 2k and 2k+1):
        //   [2k]   xyz = ABSOLUTE world position of one bone (or the player's feet)
        //          w   = the radius of the disc it presses, in world units
        //   [2k+1] x   = WEIGHT, 0..1 — a bone-volume proxy (the part's model bounding radius over
        //                grassCrushWeightRef). It drives the dwell only, never the footprint:
        //                ⚠ scaling the radius by it instead would be one knob doing two jobs, and the
        //                contact silhouette is not negotiable for a legibility parameter.
        //          yzw spare.
        // A BUFFER and not a cbuffer array because 4096 float4s is already the whole 64 KB D3D12
        // cbuffer limit with no room for the params block beside it, and this is twice that.
        DECL_BUFFER   (Persistent, Buffer(float4), gGrassCrushers)
        // The ordered-uint accumulator, 2 x grid² entries in TWO LANES. Written by pass 1 (the seed)
        // and pass 2 (the AtomicMin scatter), read by pass 3. Overlap between crushers is free and
        // unordered by construction — `min` is idempotent and commutative, so a bone palette shipped
        // twenty times over would produce the identical field (it is deduped host-side for COST, not
        // correctness).
        //
        // ⚠ LANE 0 (entry i) is the TARGET the passes min into. LANE 1 (entry grid² + i) is LAST
        // FRAME'S published clearance, written by the seed and never touched again — it exists only
        // so the splat can rate-limit its descent. LANE 2 (entry 2*grid² + i) is the RELEASE ENVELOPE:
        // the seed writes last frame's minus this frame's heal step, and the splat stores a plain 1.0
        // wherever it covers — every writer stores the same bits, so that race is benign and needs no
        // atomic. The splat cannot read the field TEXTURE for either of those:
        // it is scrolled, and pass 3 writes the very texels other threads would be reading, which is
        // an unsynchronised read/write across one dispatch. The seed already does that fetch, in the
        // one pass where it is safe, so it simply parks the answer here.
        DECL_RWBUFFER (Persistent, RWBuffer(uint), gGrassCrushAccum)
        // The published field. .x = crushZ - crushOriginZ, .yz = lay direction, .w = release envelope.
        // RW rather than an SRV/UAV pair because pass 1 LOADS it and pass 3 WRITES it in different
        // dispatches — see the no-ping-pong note above.
        DECL_RWTEXTURE(Persistent, RWTex2D(float4), gGrassCrushField)
    END_SRT_SET(Persistent)
END_SRT(GrassCrushSrtData)
