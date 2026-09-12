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

`mbtilemax.comp.fsl`, `mbcoveru`/`mbcoverv.comp.fsl` (`mbneighbormax.comp.fsl` until MB-2l),
`mbgather.comp.fsl` and `mbcommon.h.fsl`, line for line, as of MB-2l. Conventions preserved: `gMbDepth` is raw reverse-Z device depth (bigger =
closer, `d = 1/z` here); `gMbVelocity` is previous-minus-current in delivered px per frame and is
**quantised to fp16**, as RG16F actually is; `shutter` is `exposure_ms / frameDt_ms`, not
`angle/360`; the exposure window is centred, so taps span ±|v|·shutter/2.

Arms: `--fix ship` is the pre-MB-2j weighting, `--fix c7 --gain 2` is MB-2j, `--fix c9` adds
MB-2k. `--tapjitter 0` puts every pixel's taps in phase, so the difference from 1 is by
construction the entire contribution of the tap hash. Both of those last two are rig-only levers.

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
| T10 | where each pixel's answer came from (near / own / behind) | 94.8% self over a moving skirt |
| T11 | every candidate weighting scored against GT | **c6/c7 = MB-2j, shipped** |
| T12 | **THIN fast mover: the refraction and the dither** | **MB-2k, shipped** — see below |
| T13 | can a mover smear something that is NOT moving? | no: `same%` and `far%` are exactly 0 |
| T14 | **the halftone at a mover's silhouette** | **`mbTileJitter` flips a binary gate** |
| T15 | **small tiles AND long streaks** | **MB-2l, shipped** — reach is R tiles, not 1 |
| T16 | is a wide mover's revealed background EARNED? | **yes, to 0.02** — the amount is right |
| T17 | why does the residue read as a SMEAR? | **the streak is undersampled** — MB-2n |

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

## What T12 found — the two look problems reported after MB-2j shipped

A thin fast sword read as a **refraction**, and a small share of pixels looked **dithered**. The
existing scenes could not test either: every blade in them is 160 px across, and a wide body's
taps mostly land back on the body, so where the background comes from hardly matters. At 16 px
almost every tap lands on background instead.

**Four hypotheses went in. The first three were mine, and all three were wrong — each cost
one run of the rig rather than one build and one play session:**

- **"The sword is too transparent" — refuted.** MB-2j's blade share on the mover's own pixels
  matches ground truth's exposure coverage to within 4% at every width (0.083 vs 0.080 at 8 px,
  0.850 vs 0.844 at 160). The pre-MB-2j filter was 4.3× *too solid*. A thin fast object really is
  mostly background; the user was seeing correct transparency for the first time.
- **"The cap flattened the cone, so lower the gain" — refuted.** Across gain 1.0→2.4 the dither
  moves 0.0059→0.0071 while lone-mover delivery moves 0.687→1.000. The gain is not the dither's
  lever, and 2.0 is where delivery is exactly right. (2.0 and 2.4 are bit-identical — every tap
  inside the streak has already saturated.)
- **"Use blue noise for the tap phase" — a measured null, twice.** Same variance, spectrum
  shaped high: RMSE identical to four decimals at every width, and `lpRMSE` (error surviving a
  small blur — the half a viewer sees) *slightly worse*. It cannot help while the hash decides
  4% of the error.
- **The refraction is real, and it is a PROVENANCE error.** `shown_background()` backs the
  background out of the filter's own output using GT's own decomposition
  `out = cov*blade + (1-cov)*bg`, with `cov` measured exactly by a white-on-black copy of the
  scene. Ground truth scores 0 there by construction. MB-2j's background had **0.26 of the true
  contrast and correlation −0.98** — averaging a texture over ±50 px does not blur it, it
  *inverts* it. MB-2j did not cause this; it made it visible, by correctly raising the
  background's share of a thin mover's pixels from 0.47 to 0.83.

**MB-2k** weights the revealed-background *mixture* by `1/dist⁴` and leaves its *total* alone: the
offsets are a search for somewhere the background can be seen unoccluded, so the nearest place it
was found wins. On a 16 px blade at 60 px/frame: contrast 0.34 → 0.97, correlation −0.96 → +0.81,
RMSE on the mover's pixels 0.2053 → 0.0803, `lpRMSE` 0.161 → 0.075. The amputation, over-blur,
two-direction and static-world guards are **exactly unchanged** — it is a no-op wherever the
background behind a mover is locally flat.

⚠ **A first attempt scaled a cone by the streak instead, and it is instructive that it failed.**
Its best width tracks the *mover's own width* (0.15 of the streak at 8 px, 0.25 at 32), which the
gather cannot know, and when it is too narrow **no tap qualifies at all**: the bucket's weight sum
collapses and the background handed back is black — RMSE 0.0414 on a 160 px body against 0.0045.
`1/dist^p` has no radius and never reaches zero, so it cannot starve. The exponent is an optimum
rather than a trend: 0.0559 at 3, **0.0544 at 4**, 0.0579 at 6, 0.4161 at 12 where it has
collapsed onto the single nearest tap.

⚠ **It trades a large low-frequency error for a smaller high-frequency one.** A sharper estimate
depends on which taps landed where, so the hash decides more of it (`dither` 0.0074 → 0.0296).
But `excessHF` — high-frequency energy ground truth does not have — *falls* (0.0240 → 0.0156),
because an inverted texture was itself high-frequency error. Better on every visibility metric.

## What T15 found — the tile size and the streak ceiling were never one quantity

T14 halved the halftone by shrinking K and had to pay for it in the blur itself (RMSE 0.00281 →
0.04073 in the first run of it, 0.00409 → 0.04078 at the arm settings T15 uses), because the
streak clamp **was** K: 28 px tiles also meant a 28 px ceiling on a mover that wanted 60. The
filter documented that coupling as a law — *"K IS ALSO THE MAXIMUM BLUR LENGTH, and that is a
property of the algorithm"* — and it is a property of the **3×3 search**, not of the algorithm. A
tile may be told about motion within the dilation's reach; a 3×3 reaches one tile. Cover R tiles
and the reach is R·K.

`max` is separable, so R costs `2(2R+1)` taps per tile instead of `(2R+1)²`. At a matched 96 px
reach, arm over sky:

| K | R | tiles | maxLen | dilation taps/px | blurred% | speckle | RMSE vs GT |
|---|---|---|---|---|---|---|---|
| 96 | 1 | 8×6 | 96 | 0.0007 | 46.788 | 43239 | 0.00409 |
| 28 | 1 | 28×19 | 28 | 0.0077 | 13.159 | 8927 | 0.04078 |
| 28 | 4 | 28×19 | 112 | 0.0230 | 40.608 | 15240 | 0.00409 |
| **24** | **4** | **32×22** | **96** | **0.0312** | **34.272** | **12172** | **0.00409** |
| 16 | 6 | 48×32 | 96 | 0.1016 | 33.397 | 7889 | 0.00409 |
| 12 | 8 | 64×43 | 96 | 0.2361 | 31.644 | 5671 | 0.00409 |

Equal reach is equal blur to five decimals; the speckle falls with K; and the **gather gets
cheaper** (`blurred` 46.8% → 31.6%) because a tighter dilation stops dragging static pixels into
the search at all. `t_reach` asserts `cover_max(R=1) == neighbour_max` bit-for-bit before it
reports any of this, which is what makes the rows a comparison rather than two unrelated filters.

Shipped as MB-2l: `mbTileK` 96 → 24, `mbTileReach` 4, `opts2.z` carries the ceiling.

## What T16 found — the wash at an arm's silhouette is physically earned

The remaining report after MB-2l: *"a moving arm's edges are a magenta gradient, it is smearing the
foliage IN instead of the arm OUT"*. T13 had already proved the gather cannot move a static pixel's
colour, so the defect has to live on the **mover's own pixels**, in the band at its silhouette where
the exposure is only partly covered. Two candidates, with different fixes — the filter reveals MORE
background than the shutter did (a weighting bug), or it reveals the right amount of the WRONG
background (unfixable from one frame).

A 160 px mover at 50 px/frame (83 px streak), white on black so ground truth returns exposure
coverage directly:

| px inside the leading edge | GT | ship | MB-2j | MB-2k |
|---|---|---|---|---|
| 2 | 0.524 | 0.838 | 0.518 | 0.518 |
| 16 | 0.692 | 0.923 | 0.686 | 0.686 |
| 32 | 0.883 | 0.979 | 0.877 | 0.877 |

Over the whole partial band (50 560 px, 0.02 < GT cov < 0.98): **MB-2j mean error −0.0000, |mean|
0.0165, RMSE 0.0211**; `ship` +0.3579 / 0.4020. So it is the second case. The half-transparent band
is what a real shutter produces, MB-2j gets its size right to 2%, and the pre-MB-2j weighting is not
a refuge — it was 36% too opaque.

What is wrong is only WHICH background fills it. The true answer is the background **at that pixel**,
which is occluded in this frame and is not recoverable from it: the mean (MB-2j) combs it into 1D
streaks along the motion, and the nearest sample (MB-2k) synthesises confident high-contrast detail
out of an estimate that carries no information (T12's frequency sweep). Both are guesses about the
same missing data. The lever that exists is a **colour history** — one frame earlier the arm was a
streak-length back and that background WAS visible — which is what the published pipelines use.

## What T17 found — the combing is undersampling, not the estimator

Having established (T16) that the amount of revealed background is right and its content is a
guess, the obvious next move was to make the guess's error isotropic instead of directional: keep
MB-2j's weights exactly and read background taps from a 2D blur of the source (`c11`). That is
measurable, so it was measured — against a metric built for the artefact, `anisotropy`, the rms
image gradient ACROSS the motion over the rms gradient ALONG it, where 1.0 is isotropic and the
reported "stripes along the arm's direction" show up as a value **below** 1 (a train of displaced
copies varies along the motion).

It works, and it is the wrong fix. On the foliage scene, 67 486 px in the mover's partial band:

| arm | RMSE band | combing |
|---|---|---|
| MB-2j (the mean) | 0.0470 | 0.761 |
| c11, blur 4, per tap | 0.0474 | **1.004** |
| c11, blur 8, per tap | 0.0479 | 1.015 |
| c11, blur 8, at centre | 0.1253 | 0.739 |

and the competing explanation wins outright:

| MB-2j maxTaps | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| RMSE band | 0.0518 | 0.0470 | **0.0454** | 0.0454 |
| combing | 1.101 | 0.761 | **0.964** | 1.005 |
| tap spacing | 5.42 px | 2.71 px | 1.35 px | 1.00 px |

32 taps over an 87 px streak is one sample every 2.7 px against foliage 3–11 px wide, so every tap
lays down a discrete displaced copy and the copies are spaced along the motion. **64 taps reaches
both the RMSE floor and isotropy; 128 buys nothing.** It saturates at ~1.4 px — the spacing at which
consecutive taps stop skipping over the background's own features. c11 reaches the same isotropy at
a worse error, so it pays with error for what sampling density gives free, and is kept only as a
measured dead end (`--fix c11`, `bg_blur`).

`tapJitter` is visible doing its job on the way there: at 16 taps, combing 1.101 dithered against
0.641 in phase — it converts the comb into grain, which is why undersampling reads as noise rather
than as a ladder.

⚠ **The two probes disagree about MB-2k and the disagreement is not resolved.** The shown-background
probe (T12) says nearest-sample is worse on foliage; this composited band RMSE says it is better
(0.0369 vs 0.0470 at 32 taps). They measure different things — the estimate in isolation against the
final pixel — and MB-2k's combing gets *worse* as taps rise (0.889 → 1.339 → 1.543) where MB-2j's
improves. `mbProv` stays off on that last point alone, and it is a live knob.

## What it does NOT show

⚠ **T7's null had a cause, and it was the rig, not the scene — found 2026-09-11 by T14.** The rig
runs 768×512 at K=96, which is **8×6 tiles**. The game runs **27×17**. A mover that covers two tiles
here covers a quarter of the frame, so a mover's 3×3-dilated tile boundary falls OFF SCREEN and the
jitter has nothing to straddle. Shrink K until the tiles-per-frame ratio matches the game and the
artefact appears in one run. **Any test of anything tile-shaped must match the game's tile COUNT,
not its tile SIZE.** The note below stands as written but is no longer the whole story.

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
