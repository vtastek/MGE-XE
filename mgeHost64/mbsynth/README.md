# mbsynth — synthetic tests for the MB-2 motion blur filter

`python3 mbsynth.py` — no dependencies but numpy. Runs in about three minutes, needs no game,
no GPU and no save.

## Why

Every question asked of this filter until now was asked with a **whole-frame aggregate** —
`changed%` out of `gMbStats` — against a screenshot. A frame mean cannot resolve a defect that
lives on a silhouette: those are order 1% of the pixels, inside a ±0.1 spread. The `mbSoftZ`
sweep that came back 0.287 vs 0.288 over a 10× change was not an inert knob, it was a **blind
metric**, and two earlier A/Bs (the objvel depth bias, the first `pcdrive` comparison) had the
same flaw. A frame-wide mean cannot test a silhouette-local hypothesis.

Here the scene is analytic, so there is a **ground truth**: a rect moving at a known velocity,
averaged over the exposure, *is* the correct motion blur. `ground_truth()` integrates the scene
over the centred shutter window at 513 sub-samples. The filter is then scored per pixel against
the right answer instead of against an opinion, and the score can be **split by what the
receiving pixel is** — which is how a silhouette-local defect becomes a number.

## What is transcribed

`mbtilemax.comp.fsl`, `mbneighbormax.comp.fsl`, `mbgather.comp.fsl` and `mbcommon.h.fsl`, line
for line, as of MB-2h. Conventions preserved: `gMbDepth` is raw reverse-Z device depth (bigger =
closer, `d = 1/z` here); `gMbVelocity` is previous-minus-current in delivered px per frame and is
**quantised to fp16**, as RG16F actually is; `shutter` is `exposure_ms / frameDt_ms`, not
`angle/360`; the exposure window is centred, so taps span ±|v|·shutter/2.

⚠ **This is a transcription, not the shader.** If the FSL changes, this does not. The rig's
authority is that T0 reproduces the velocity floor's bit-exact pass-through and T1 reproduces the
streak length to 3 px in 239 — it is not a substitute for a GPU capture.

## The tests

| | what it asks | status |
|---|---|---|
| T0 | a still frame must come out BIT-IDENTICAL | PASS, max diff exactly 0 |
| T1 | streak length and shape vs ground truth | extent within 3 px of GT |
| T2 | blade in FRONT, static skirt behind | over-delivers, 2.96 |
| T3 | blade in FRONT, skirt behind **and slowly moving** | **amputated, 0.15** |
| T3b | control: blade BEHIND the skirt | GT cuts it too — correct |
| T4 | `mbSoftZ` sweep, localised to the silhouette | exactly inert, and provably so |
| T5 | does a slow object soften its own edge | 6 px vs GT's 7 px — fine |
| T6 | **tap-count invariance** | **FAILS: 0.95 at 6 taps, 1.51 at 64** |
| T6b | how fast must a receiver be to block a smear | **cliff at 0.6 px/frame** |
| T7 | tile grid vs `mbTileJitter` | does not reproduce the grid — see below |
| T8 | two bodies at right angles vs `mbTwoDir` | no seam in either arm |
| T9 | candidate fix: weight taps by arc length | (a) null by construction, (b) partial |

`--explain X,Y` prints the tap-by-tap arithmetic for one pixel: which axis each tap walked, what
it landed on, and all three weight terms. That is what turned T3 from a correlation into a
mechanism.

## What it found

**1. The amputation is `mbTwoDir`, and it is a cliff.** A blade in front of a *static* receiver
smears across it (over-delivering, 2.96). Move the receiver by **0.6 px per frame** and delivery
collapses to 0.159 — a 16× drop from a tenth of a pixel. The threshold is exactly `lenSelf >= 1.0`,
MB-2h's two-direction gate. With `mbTwoDir=0` the cliff is gone and the falloff is smooth
(2.96 → 0.60 over the same sweep).

The mechanism, from `--explain`: when twoDir arms, half the taps walk the *receiver's own* axis.
For a slow receiver those taps all land within its own 5 px streak, where both cones and both
cylinders are ≈1, so each scores up to `1 + 1 + 2 = 4` — against ~0.6 for a tap that actually
found the blade. Sixteen self-taps carried **61.6** weight of skirt against **2.5** of blade:
94.6% of the answer was the receiver re-sampling itself, sixteen times, within 2.4 px.

This also explains why it appeared in the same play session that confirmed MB-2h fixed the seams.

**2. The filter's answer depends on its own sample count.** Delivery vs ground truth on a lone
mover: 0.954 at 6 taps, 1.181 at 12, 1.415 at the shipped 32, 1.513 at 64. A reconstruction filter
may get noisier when undersampled; it must not get *browner*. At the shipped setting the blur is
**1.4× too strong**, which is the residue of "motion blur on bodies are too much" that MB-2f's
streak clamp did not reach — MB-2f fixed the cone's *shape*, this is its *amount*.

T9(a) refutes the obvious explanation: arc-length weighting is algebraically a no-op on one axis
(the factor cancels from numerator and denominator), so this is **not** a sampling-density bug.
The real asymmetry is that background taps score exactly 0 — `cone(dist, 0) = 0` both ways round —
so however long the streak, **the background is represented exactly once**, by the centre's fixed
weight of 1, while the mover is represented once per tap.

**3. `mbSoftZ` cannot matter for a static receiver, and that is provable rather than measured.**
The `b` term is the only one carrying `softZ` in that direction, and it is multiplied by
`mbCone(dist, selfStreak)` with `selfStreak = 0`. Zero times anything. The in-game null result was
correct; the knob was never the lever.

## What it does NOT show

T7 and T8 **do not reproduce the tile grid or the direction seam**, in any arm, jitter on or off.
That is a statement about these scenes, not about MB-2h: a rigid translating rect has one velocity
everywhere, so the dilation has nothing to quantise, and the agreement weights already keep the
grid out of the static background (a background pixel runs the gather and comes out unchanged).
The grid was measured directly from `squaretiles.png` — K=96 phase 0 on both axes, 1.26×/1.30× the
other phases — and that measurement stands on its own. Reproducing it here needs a scene closer to
what produced it: a body with a velocity gradient *against* static world, at real resolution.

In T7 `mbTileJitter` is very slightly **worse** against ground truth (RMSE 0.0891 → 0.0897). Small,
one scene, and the scene does not contain the artifact jitter was built for — so it is not evidence
against jitter, only a note that its benefit is unproven here.
