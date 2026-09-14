# pbrsynth — synthetic tests for height-derived normals (Forge PBR materials)

`python3 pbrsynth.py` — no dependencies but numpy. The synthetic table (T0–T10a) runs in about
fifteen seconds, needs no game, no GPU and no save. T10 reads the shipped `_paramh` maps and needs
`--paramh "<Data Files>/textures"`; it scans all 210 in well under a minute.

Plan: `tasks/forge-pbr-materials.md`, Track A.

## Why

Morrowind meshes carry no tangents, so this project stores **height** in the alpha of a DXT5
`_paramh` and differentiates it in the shader. Two questions follow and neither had been measured:

1. Should the derivative be taken **offline** at full precision and stored (BC5 — two independent
   BC4 blocks), or at **runtime** from a filtered, 8-bit field?
2. Does the magnification complaint that motivates 4096-square `_paramh` maps want **resolution**,
   or a **C1 reconstruction**? The live 4.38 GB `_paramh` set rides on the answer.

## Ground truth is closed form

The surface is `P(u,v) = O + u·Tu + v·Tv + disp·h(u,v)·n` for an analytic `h`, so the true normal is
`normalize(cross(∂P/∂u, ∂P/∂v))` exactly, with `∂h/∂u`, `∂h/∂v` written out by hand. There is no
integration error to argue about and no reference implementation that could itself be wrong. T1
checks the hand-written gradients against finite differences of `h` to 1e-9.

## What is transcribed

- **Live DX9 shader** (`Data Files/shaders/core-hlsl/XE FixedFuncEmu_PS.hlsl`, May 2026 — ⚠ which
  exists ONLY in the deploy tree; see the plan): the central difference at ±1.5 **base** texels,
  `heightScale = 16` over the raw difference, and the rotate through `BuildPerPixelTBN`'s cotangent
  frame (`common.hlsl`).
- **Rescued Aug–Sep 2025 stash** (`core-hlsl/do not delete/`, commit 47cf8ef5):
  `DerivFromHeightMap` (forward), `DerivFromHeightMapHQ` (Sobel), `ParallaxHeightNormal`'s
  footprint blend, `SurfgradScaleDependent` (Mikkelsen surface gradient).
- **Codecs are the real bit layouts**: BC4 packs to 8 bytes and back (two endpoints, sixteen
  3-bit indices, both interpolation modes); BC5 is two independent BC4 blocks; every mip is
  compressed separately, as a DDS stores it.

**New arms, proposed rather than transcribed:** `bicubic` (Catmull-Rom derivative, the plan's C1
candidate), `bspline` (cubic B-spline derivative, C2), `cdbs` (the live central difference with
B-spline taps — found by the rig), `deriv5` / `deriv5c` (a baked BC5 derivative map, bilinear /
Catmull-Rom filtered). No shader in this tree stores a derivative map.

⚠ **This is a transcription, not the shader.** If the FSL or the HLSL changes, this does not. Its
authority is T0 (the codec round-trips the real bit layout, and the fast path equals the byte-level
path exactly) and T1 (ground truth to 1e-9). It is not a substitute for a GPU capture.

Every arm returns `∂h/∂u` in the SAME per-uv units, so the comparison is about reconstruction
rather than gain. The live shader does NOT normalise, and that is measured separately (T9).

## The tests

| | what it asks | status |
|---|---|---|
| T0 | codecs are the real bit layouts and round-trip | **PASS** — constant block exact; fast == byte-level to 0.0 |
| T1 | the closed-form gradient IS the gradient | **PASS** — ≤ 2.4e-9 relative on all four scenes |
| T2 | angular error by gradient arm, magnified 8× | measured — `deriv5` 0.31° vs `cd` 1.99° on `noise` |
| T3 | **the facet metric**: curvature ON texel-cell edges | **every difference arm AT the ceiling (16.0)**; `bspline` 1.07, `cdbs` 1.85 |
| T4 | magnification sweep 1/16 → 16 tpp | measured; no row skipped (window now always fits) |
| T5 | **does 4096 buy anything a better reconstruction would not?** | **answered, with a control** — see below |
| T6 | 8-bit floor (`shallow`, 3/255 of range) | **u8 alone flattens it: retention 0.000**; `deriv5` 1.001 |
| T7 | cotangent TBN vs surface gradient, orthonormal vs skewed | identical on orthonormal; **TBN floors at ~2.7° under skew** |
| T8 | the "mip bug": do base-texel taps collapse the bump? | **hypothesis REFUTED** — the proposed fix is worse at every distance |
| T9 | `heightScale = 16` is resolution-coupled | measured — slope 7.06× / 4.00× / 2.00× / 1.00× at 512/1K/2K/4K |
| T10a | calibration of T10's noise estimator | **bitstream floor PASSES (empty band 0.61–0.96); corner floor FAILS** |
| T10 | **real art**: top octave above codec noise? | **135 of 208 4096 maps: top octave at least half codec noise** |
| T11 | **what the BAKE must read**: source vs shipped height | **`deriv5q` is WORSE than not baking**; `deriv5s` (fixed gain) fails outright |

## What it found

**1. Resolution does not touch the facets. Reconstruction does.** (T3, T5) Facet enrichment is
`mean(2nd difference² | on a cell edge) / mean(it anywhere)`; its ceiling is `spp/2` = 16, reached
when all the normal's curvature sits on the cell edges — which is what a kink is. `cd`, `fwd`,
`sobel`, `ddx` and `deriv5` read 15.5–16.2 at **every** resolution from 512 to 4096. Doubling
resolution halves the facet PITCH (32 → 16 → 8 → 4 px) and nothing else: 4× the memory for 2× the
facet density. The plan's C1 candidate is one order short — Catmull-Rom is C1 in the HEIGHT, which
leaves the NORMAL C0 and still kinked (10.3). A C2 height (`bspline`, 1.07) or a C1 derivative
(`deriv5c`, 2.43) removes the kinks; so does `cdbs` (1.85).

**2. Catmull-Rom is the best reconstruction of the field and the worst of a quantised one.** At
infinite precision it wins (0.031° vs `cd` 0.227° on `noise`); at 8 bits it is the worst runtime arm
(5.3° vs 1.2°), at BC4 worse still (8.8°). Its negative lobes sharpen the quantiser's staircase.
This is why the plan's bicubic arm fails T5 and why the B-spline, with non-negative weights, does not.

**3. The runtime winner is `cdbs`, and it wins only on the content most maps have.** The live
±1.5-texel baseline averages the 8-bit staircase; bilinear taps put a kink at every edge. B-spline
taps on the same baseline keep both properties: on `noise` 1.46° / facets 1.85 against `cd`'s
1.99° / 15.6, and it beats `cd` at every resolution (4096: 3.59° vs 5.23°, p99 15.3 vs 21.2). Its
cost is 16 bilinear fetches against `cd`'s 4, and it rounds sharp creases (`bevel` 5.28° vs 5.11°).
On the `fine` control — detail near the source's own Nyquist — its baseline low-passes the detail
(retention 0.36 at 2048) and the narrow-baseline `bspline` is the better runtime arm (8.6° at 4096).
**No single runtime arm wins everywhere**, which is the measured reason `pbrGradMode` ships as a
live knob.

**4. Baking the derivative wins by an order of magnitude, at every resolution.** `deriv5` is 0.12°
at 4096 against `cd`'s 5.23°, and **512 + `deriv5` beats 4096 + `cd`** (0.87° vs 5.23°) on the
resampled scene. It differentiates BEFORE quantising; every runtime arm differentiates after. On
`shallow` (3/255 of range) 8-bit storage alone flattens the gradient to exactly zero for every
runtime arm (T6) while `deriv5` keeps it (1.001). What it cannot do is parallax, soft parallax
shadows or height-blended overlays, all of which need the height itself.

**5. The 4096 tier is mostly holding codec noise.** (T5 `step`/`flat`, T10) On a resampled surface
the RMS texel-to-texel step falls 3.33 → 1.69 → 0.91 → 0.61 LSB and the share of neighbour pairs
that quantise to the SAME code rises 16% → 65%: two thirds of 4096's central differences come out
exactly zero. On the shipped art, T10 measures each map's top octave (the band a 2048 map cannot
hold) against that map's own BC4 noise, read from its bitstream and corrected by T10a's calibration
at the map's own gradient regime:

| of the 208 shipped maps at 4096 | |
|---|---|
| top octave at least HALF codec noise (SNR < 1) | **135** |
| top octave mostly real detail (SNR ≥ 3) | 26 |
| the octave below is ALSO SNR < 1 | 14 |
| median top-octave share of all gradient energy | **44%** |

So for most maps the 4096 level is holding the codec's own error, and that error is the part the
shader's derivative amplifies most (the gradient weights a band by frequency²). The per-map table
(`--paramh-csv`) feeds Track B's `content_height_cap`.

**6. The mip "bug" is not one.** (T8) The plan said base-texel taps collapse the bump at distance.
They don't: the taps sample a continuous (trilinear) function, so their difference is the slope of
the FILTERED field. `LIVE/own` stays within 4% of 1 through tpp 4 and bottoms at 0.73 at tpp 16
(straddling mip cells once the mip nears its own Nyquist). The proposed fix — offsets in mip texels
— is a second low-pass on top of the chain's own and is worse at every distance (0.37 at tpp 16).

**7. Under a skewed UV island the frame is the bottleneck, not the gradient.** (T7) TBN and the
surface gradient agree to the digit on an orthonormal map. On a 1.6:1 + 20° shear island the
cotangent frame floors at ~2.7° whatever the gradient — `deriv5`/TBN 2.68° against `deriv5`/surfgrad
0.03°, `cdbs`/TBN 2.73° against 0.52°. A better gradient under the live frame buys nothing on such
islands.

**8. `heightScale = 16` makes the bump depth resolution-dependent.** (T9) It multiplies the RAW
3-texel difference, so the per-uv gain is `16·N/3`: the same art reads 2× shallower at 4096 than at
2048. Re-encoding a map changes its apparent depth with no edit anywhere. Retuning the constant is
not a fix for facets — it changes how deep they look, not how many there are.

**9. Two derivative-map designs failed before the third worked, and the rig killed both cheaply.**
The bake (`mgeHost64/pbrbake`) exists because of finding 4, and the arms here are what decided its
format:

| arm | what it bakes | shallow (terracing) | noise |
|---|---|---|---|
| `cd` | nothing — the runtime baseline | 0.159°, retention **0.000** | 1.99° |
| `deriv5q` | the derivative of the **shipped 8-bit height** | 0.159°, retention **0.000** | **3.08° — worse than cd** |
| `deriv5s` | the 16-bit source, **fixed gain 2**, BC5 | 0.159°, retention **0.000** | **3.47° — worse than cd** |
| *(shipped)* | the 16-bit source, **per-texture range**, BC5 | **0.009°, retention 0.955** | **0.76°** |
| `deriv5` | the analytic truth (the concept, not a file) | 0.000°, retention 1.001 | 0.31° |

`deriv5q` says the SOURCE is the whole feature: differentiating an already-quantised staircase at
full precision captures its spikes exactly and then requantises them. `deriv5s` says the RANGE is
not optional: BC4's per-block endpoints sit on a **global** 8-bit grid, so a block whose derivatives
all lie within 1/127 of each other collapses to one constant value — and a shallow map is nothing
but such blocks. Both were plausible, both were simpler than what shipped, and both cost one run.

## Instruments that were wrong first, and are recorded as such

- **T3's first lattice was assumed, not searched**: it classified crossings at integer texel
  coordinates, but `cd`'s ±1.5-texel taps kink on HALF-integers, so the live shader scored 0.00 —
  "more continuous than a C1 spline". Every phase is now tried.
- **T3's first ratio divided by zero**: across/within variance reported 1e24 for `ddx`, whose
  within-cell variance is nil. Replaced by a bounded enrichment.
- **T5 first read the facet metric at each row's own magnification**, whose ceiling moves with it;
  4096's ceiling was 2.0, and `cd`'s 1.98 looked like a pass. Fixed at 32 samples per texel.
- **The dome was first measured on its flat peak** (retention divided by ~0) **and the bevel's crease
  was off-screen** (every arm 0.000°). Scenes now carry their analysis window.
- **T8's first verdict was written before its numbers** ("LIVE/own stays near 1.0" — it is 0.73 at
  tpp 16). The verdict is now computed from the rows.
- **T10's corner noise floor failed its calibration**: it reads 1.18 on a map whose top octave holds
  3.4× the codec energy. It prints, and has no vote.

## Limits

- Isotropic footprints only: anisotropic filtering at grazing angles is not modelled.
- The hardware's 8-bit bilinear weight precision is not modelled.
- The rig's BC4 encoder is min/max endpoints; texconv refines endpoints, so real error is a little
  lower than the rig's — which makes the rig's runtime arms, if anything, pessimistic.
- T10's noise floor treats BC4 error as white; T10a shows how far that holds (an empty band reads
  0.61–0.96) and T10 corrects each map at its own regime rather than with one constant.
- `bevel` is one sharp crease; T2 is the only place it is scored.
