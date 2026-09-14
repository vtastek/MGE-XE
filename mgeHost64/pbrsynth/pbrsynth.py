#!/usr/bin/env python3
"""
pbrsynth -- synthetic tests for height-derived normals in the Forge PBR material path.

WHY THIS EXISTS.  Morrowind meshes carry no tangents (io_scene_mw's NiGeometryData defines the
whole vertex payload as vertices / normals / vertex_colors / uv_sets and there is no tangent
field), so this project stores HEIGHT and differentiates it in the shader rather than storing a
normal map.  Two questions follow from that and neither has ever been measured here:

  1. Should the derivative be taken OFFLINE at full precision and stored (BC5, two independent
     BC4 blocks), or at RUNTIME from a filtered 8-bit height field?
  2. Does the magnification complaint that motivates 4096-square _paramh maps want RESOLUTION,
     or does it want a C1 RECONSTRUCTION?

Question 2 has a specific shape.  normal-from-height differentiates a BILINEARLY INTERPOLATED
field, and bilinear is C0 but not C1: its gradient is discontinuous across texel-cell boundaries.
Doubling resolution does not remove that discontinuity, it makes the facets smaller -- 4x the
memory for 2x the facet density, with the seams still there for a closer camera.  So the rig
measures the facets DIRECTLY (T3) rather than inferring them from an angular error that a smooth
scene can hide.

GROUND TRUTH IS CLOSED FORM, which is the one place this rig is better off than mbsynth.  The
surface is P(u,v) = origin + u*Tu + v*Tv + s*h(u,v)*n for an analytic h, so the true normal is
normalize(cross(dP/du, dP/dv)) exactly, with dh/du and dh/dv written out by hand.  There is no
integration to argue about and no reference implementation to be wrong.

WHAT IS TRANSCRIBED.  From the LIVE DX9 shader (Data Files/shaders/core-hlsl/XE FixedFuncEmu_PS.hlsl,
May 2026): the fixed central difference at +-1.5 BASE-LEVEL texels, `heightScale = 16`, the
tangent-space assembly normalize(float3(-dhdu*hs, -dhdv*hs, 1)) and the rotate through
BuildPerPixelTBN's cotangent frame (common.hlsl).  From the RESCUED Aug-Sep 2025 stash
(core-hlsl/do not delete/XE FixedFuncEmu.hlsl, commit 47cf8ef5): DerivFromHeightMap (forward
difference), DerivFromHeightMapHQ (Sobel 3x3), ParallaxHeightNormal's footprint-adaptive blend of
central difference and ddx, and SurfgradScaleDependent (Mikkelsen surface gradient, no TBN).

CONVENTIONS PRESERVED.  Height lives in the ALPHA channel of a DXT5 _paramh (metal/rough/IOR in
RGB); texel centres are at integer texel coordinates, so uv * N - 0.5 is the sampling position;
the central difference offsets in BASE-LEVEL texels regardless of which mip the sampler picks,
which is a bug and is measured as one in T8.

WHAT IS NOT TRANSCRIBED, and it matters: `deriv5`/`deriv5c` are NEW -- no shader in this tree
stores a derivative map, because until this week the host could not load BC4 or BC5 at all.  Those
two arms are a PROPOSAL being scored, not a description of anything that runs.

!! THIS IS A TRANSCRIPTION, NOT THE SHADER.  If the FSL or the HLSL changes, this does not.  Its
authority is that T0 round-trips the real BC4 bit layout and T1 reproduces the closed-form
gradient to 1e-7; it is not a substitute for a GPU capture.

Run:  python3 pbrsynth.py                 (the whole table)
      python3 pbrsynth.py --test T5       (one row)
      python3 pbrsynth.py --list
"""

import argparse
import math
import sys

import numpy as np

# =======================================================================================
# 1.  BLOCK CODECS -- the real bit layouts, packed to bytes and unpacked again
# =======================================================================================
#
# BC3's alpha block and BC4 are the same codec: two 8-bit endpoints plus 3-bit per-texel indices,
# i.e. 8 levels on the block's own ramp.  BC1's colour block gives two RGB565 endpoints and 2-bit
# indices -- 4 levels, with all three channels chained to ONE shared line in RGB space, which is
# why two independent normal axes packed into its RG co-vary inside every 4x4 block.  That is the
# artifact height-in-alpha was chosen to escape, and BC5 escapes it a second way: two INDEPENDENT
# BC4 blocks, so a stored derivative map has exactly the per-channel fidelity of a height map.
#
# BC1 is deliberately not implemented.  The DXT1 packing it would model (height in green) is
# already known abandoned -- see the note in texturematcher/CLAUDE.md, which documents it and is
# wrong to.


def _bc4_ramp(e0, e1, signed):
    """The 8 decoded values of a BC4 block, as floats, for endpoints (e0, e1).

    Two modes, exactly as the hardware: e0 > e1 gives a 6-interpolant ramp between them; e0 <= e1
    gives 4 interpolants plus the two extremes of the range.  Index 0 always selects e0 and index 1
    always selects e1, under BOTH modes -- which is what makes a constant block (e0 == e1, indices
    all zero) exact and mode-independent.
    """
    lo, hi = (-1.0, 1.0) if signed else (0.0, 1.0)
    t = np.empty(8, dtype=np.float64)
    t[0], t[1] = e0, e1
    if e0 > e1:
        for k in range(6):
            t[2 + k] = ((6 - k) * e0 + (1 + k) * e1) / 7.0
    else:
        for k in range(4):
            t[2 + k] = ((4 - k) * e0 + (1 + k) * e1) / 5.0
        t[6], t[7] = lo, hi
    return t


def _pack_indices(idx16):
    """16 three-bit indices -> 6 bytes, low index first (the DDS bit order)."""
    bits = 0
    for i in range(16):
        bits |= (int(idx16[i]) & 0x7) << (3 * i)
    return bytes((bits >> (8 * k)) & 0xFF for k in range(6))


def _unpack_indices(six):
    bits = 0
    for k in range(6):
        bits |= six[k] << (8 * k)
    return np.array([(bits >> (3 * i)) & 0x7 for i in range(16)], dtype=np.int64)


def bc4_encode_block(vals16, signed=False):
    """One 4x4 block of scalars -> the 8 bytes a BC4 block actually is.

    Endpoint selection is min/max with the 6-interpolant mode, which is what every encoder does for
    a smooth field; the point of the rig is the 3-bit INDEX quantisation, not endpoint search.
    """
    v = np.asarray(vals16, dtype=np.float64).reshape(16)
    if signed:
        q = np.clip(np.round(v * 127.0), -127, 127)          # -128 is reserved in BC4_SNORM
        e0, e1 = int(q.max()), int(q.min())
        f0, f1 = e0 / 127.0, e1 / 127.0
        b0, b1 = e0 & 0xFF, e1 & 0xFF                        # int8 two's complement
    else:
        q = np.clip(np.round(v * 255.0), 0, 255)
        e0, e1 = int(q.max()), int(q.min())
        f0, f1 = e0 / 255.0, e1 / 255.0
        b0, b1 = e0, e1
    if e0 == e1:                                             # constant block: index 0 everywhere
        return bytes([b0, b1]) + _pack_indices(np.zeros(16, dtype=np.int64))
    ramp = _bc4_ramp(f0, f1, signed)
    src = q / (127.0 if signed else 255.0)
    idx = np.argmin(np.abs(src[:, None] - ramp[None, :]), axis=1)
    return bytes([b0, b1]) + _pack_indices(idx)


def bc4_decode_block(eight, signed=False):
    if signed:
        r0 = eight[0] - 256 if eight[0] > 127 else eight[0]
        r1 = eight[1] - 256 if eight[1] > 127 else eight[1]
        f0, f1 = r0 / 127.0, r1 / 127.0
        ramp = _bc4_ramp(f0, f1, True)
    else:
        f0, f1 = eight[0] / 255.0, eight[1] / 255.0
        ramp = _bc4_ramp(f0, f1, False)
    return ramp[_unpack_indices(eight[2:8])]


def _blocks(img):
    """(H, W) -> (nblocks, 16), 4x4 blocks in raster order, texels in raster order within."""
    h, w = img.shape
    return img.reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 16)


def _unblocks(b, h, w):
    return b.reshape(h // 4, w // 4, 4, 4).transpose(0, 2, 1, 3).reshape(h, w)


def _ramps(f0, f1, signed):
    """_bc4_ramp for every block at once: (nb,) endpoint arrays -> (nb, 8) decode tables."""
    lo, hi = (-1.0, 1.0) if signed else (0.0, 1.0)
    nb = f0.shape[0]
    t = np.empty((nb, 8), dtype=np.float64)
    t[:, 0], t[:, 1] = f0, f1
    k6 = np.arange(6, dtype=np.float64)
    k4 = np.arange(4, dtype=np.float64)
    six = ((6 - k6)[None, :] * f0[:, None] + (1 + k6)[None, :] * f1[:, None]) / 7.0
    four = ((4 - k4)[None, :] * f0[:, None] + (1 + k4)[None, :] * f1[:, None]) / 5.0
    four = np.concatenate([four, np.full((nb, 1), lo), np.full((nb, 1), hi)], axis=1)
    t[:, 2:8] = np.where((f0 > f1)[:, None], six, four)
    return t


def bc4_roundtrip(img, signed=False):
    """Encode and decode a whole 2D scalar image through BC4, vectorised over blocks.

    The SAME endpoint and index rule as bc4_encode_block/bc4_decode_block -- T0 asserts the two
    agree exactly -- because the byte-level pair is what proves the bit layout, while this one is
    what makes a 4096-square map tractable.  Edge-padded to a multiple of 4, as an encoder does for
    a 2x2 or 1x1 mip, and cropped back.
    """
    h, w = img.shape
    ph, pw = (-h) % 4, (-w) % 4
    src = np.pad(img, ((0, ph), (0, pw)), mode="edge") if (ph or pw) else img
    B = _blocks(np.asarray(src, dtype=np.float64))
    sc = 127.0 if signed else 255.0
    q = np.clip(np.round(B * sc), -127.0 if signed else 0.0, 127.0 if signed else 255.0)
    e0, e1 = q.max(axis=1), q.min(axis=1)
    ramp = _ramps(e0 / sc, e1 / sc, signed)
    idx = np.argmin(np.abs((q / sc)[:, :, None] - ramp[:, None, :]), axis=2)
    idx[e0 == e1] = 0                                        # constant block: index 0 everywhere
    dec = np.take_along_axis(ramp, idx, axis=1)
    return _unblocks(dec, h + ph, w + pw)[:h, :w]


def bc4_roundtrip_scalar(img, signed=False):
    """The byte-level path over a whole image: every block really is packed to 8 bytes and
    unpacked again.  Slow, and kept only so T0 can hold the fast path to it."""
    h, w = img.shape
    out = np.empty((h, w), dtype=np.float64)
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            blk = img[by:by + 4, bx:bx + 4].reshape(16)
            out[by:by + 4, bx:bx + 4] = bc4_decode_block(bc4_encode_block(blk, signed), signed).reshape(4, 4)
    return out


def bc4_decode_file_blocks(raw8):
    """Decode real BC4 / DXT5-alpha blocks straight from the file: (nb, 8) uint8 -> (nb, 16)
    float in [0, 1].  The endpoints come from the bytes, so both of BC4's modes occur as the
    encoder chose them -- which the rig's own encoder never does (it always picks e0 > e1)."""
    a0 = raw8[:, 0].astype(np.float64) / 255.0
    a1 = raw8[:, 1].astype(np.float64) / 255.0
    bits = np.zeros(raw8.shape[0], dtype=np.uint64)
    for k in range(6):
        bits |= raw8[:, 2 + k].astype(np.uint64) << np.uint64(8 * k)
    sh = (np.uint64(3) * np.arange(16, dtype=np.uint64))[None, :]
    idx = ((bits[:, None] >> sh) & np.uint64(7)).astype(np.int64)
    return np.take_along_axis(_ramps(a0, a1, False), idx, axis=1)


def bc5_roundtrip(imgR, imgG, signed=True):
    """BC5 is literally two BC4 blocks, red then green, INDEPENDENTLY.  That independence is the
    whole reason a stored derivative map costs a height map's fidelity per channel rather than
    BC1's shared-line compromise."""
    return bc4_roundtrip(imgR, signed), bc4_roundtrip(imgG, signed)


def quant_u8(img):
    """Plain 8-bit, the ceiling BC4 is measured against."""
    return np.round(np.clip(img, 0.0, 1.0) * 255.0) / 255.0


CODECS = {
    "none": lambda h: h,                       # infinite precision: isolates RECONSTRUCTION
    "u8":   quant_u8,                          # 8-bit, no block structure
    "bc4":  lambda h: bc4_roundtrip(h, False),  # what a _paramh's alpha block actually is
}


# =======================================================================================
# 2.  SCENES -- analytic height fields with closed-form gradients
# =======================================================================================
#
# Each is chosen to isolate ONE failure, so that a method which is good on average can still be
# caught.  h is a function of TEXTURE COORDINATE (u, v) in [0,1] and returns values in [0,1];
# dh returns (dh/du, dh/dv) in the same units.  Nothing here is sampled, filtered or quantised --
# these ARE the truth, and every other quantity in the rig is scored against them.


class Scene(object):
    # `centre` is where the analysis window sits, and it is part of the scene rather than a knob
    # because the wrong centre silently deletes the test: a window on the dome's PEAK sees a
    # gradient of ~0, so every retention figure divides by nothing and reads 1.9 for arms that are
    # within 2% of correct, and a window off the bevel sees no crease at all.
    def __init__(self, name, h, dh, why, centre=(0.5, 0.5)):
        self.name, self.h, self.dh, self.why = name, h, dh, why
        self.centre = centre


def _dome(u, v):
    # A smooth radial cosine bump: C-infinity inside, and C1 at its rim because the derivative
    # goes to zero there.  No quantisation stress, no edges -- so an arm that fails HERE fails at
    # reconstruction and nothing else.
    r = np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2) / 0.35
    rc = np.clip(r, 0.0, 1.0)
    return 0.5 + 0.5 * np.cos(np.pi * rc) * 0.9


def _dome_d(u, v):
    du, dv = u - 0.5, v - 0.5
    r = np.sqrt(du * du + dv * dv)
    rn = r / 0.35
    inside = rn < 1.0
    k = np.where(r > 1e-12, -0.45 * np.pi / 0.35 * np.sin(np.pi * np.clip(rn, 0, 1)) / np.maximum(r, 1e-12), 0.0)
    k = np.where(inside, k, 0.0)
    return k * du, k * dv


_BEVEL_W = 0.012          # 12 texels at 1024 -- a chamfer, not a wall and not a slow ramp
_BEVEL_RISE = 0.35


def _bevel(u, v):
    # A chiselled bevel: flat, a linear ramp, flat again, with the whole feature 12 texels wide at
    # the 1024 reference so BOTH creases sit inside the analysis window.  The gradient is piecewise
    # CONSTANT with two jumps -- the worst case for anything that assumes smoothness, and the case
    # a plain difference is best at.
    t = np.clip((u - (0.5 - 0.5 * _BEVEL_W)) / _BEVEL_W, 0.0, 1.0)
    return 0.3 + _BEVEL_RISE * t


def _bevel_d(u, v):
    lo, hi = 0.5 - 0.5 * _BEVEL_W, 0.5 + 0.5 * _BEVEL_W
    inside = (u > lo) & (u < hi)
    return np.where(inside, _BEVEL_RISE / _BEVEL_W, 0.0), np.zeros_like(v)


_NOISE_FREQ = np.array([[13.0, 5.0], [7.0, -17.0], [23.0, 11.0], [3.0, 29.0], [19.0, -2.0]])
_NOISE_AMP = np.array([0.09, 0.07, 0.05, 0.04, 0.06])
_NOISE_PH = np.array([0.3, 1.1, 2.4, 0.7, 1.9])


def _noise(u, v):
    # Band-limited pseudo-noise: five sinusoids, the highest at 29 cycles across the texture.  At a
    # 512-texel source that is ~8.8 texels per cycle and comfortably resolved; the point is that it
    # has energy at many scales, so mip selection and footprint filtering have something to destroy.
    s = np.zeros_like(u)
    for (fx, fy), a, p in zip(_NOISE_FREQ, _NOISE_AMP, _NOISE_PH):
        s = s + a * np.sin(2.0 * np.pi * (fx * u + fy * v) + p)
    return 0.5 + s


def _noise_d(u, v):
    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    for (fx, fy), a, p in zip(_NOISE_FREQ, _NOISE_AMP, _NOISE_PH):
        c = a * np.cos(2.0 * np.pi * (fx * u + fy * v) + p) * 2.0 * np.pi
        du = du + c * fx
        dv = dv + c * fy
    return du, dv


def _shallow(u, v):
    # A gentle dome whose whole peak-to-trough range is 3/255.  Every quantiser in the chain sees
    # a handful of distinct codes, so this is where TERRACING lives -- the defect that is invisible
    # in the stored scalar and unmissable in its derivative.
    return 0.5 + (3.0 / 255.0) * (0.5 + 0.5 * np.cos(np.pi * np.clip(
        np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2) / 0.4, 0.0, 1.0)))


def _shallow_d(u, v):
    du, dv = u - 0.5, v - 0.5
    r = np.sqrt(du * du + dv * dv)
    rn = r / 0.4
    k = np.where((rn < 1.0) & (r > 1e-12),
                 -(3.0 / 255.0) * 0.5 * np.pi / 0.4 * np.sin(np.pi * np.clip(rn, 0, 1)) / np.maximum(r, 1e-12),
                 0.0)
    return k * du, k * dv


_FINE_K = 24.0            # frequency multiplier: ~6 texels per cycle at 4096, aliased below 2048
_FINE_A = 1.0 / 10.0      # amplitude divisor, so the SLOPES stay in the same range as `noise`


def _fine(u, v):
    s = np.zeros_like(u)
    for (fx, fy), a, p in zip(_NOISE_FREQ, _NOISE_AMP, _NOISE_PH):
        s = s + a * _FINE_A * np.sin(2.0 * np.pi * _FINE_K * (fx * u + fy * v) + p)
    return 0.5 + s


def _fine_d(u, v):
    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    for (fx, fy), a, p in zip(_NOISE_FREQ, _NOISE_AMP, _NOISE_PH):
        c = a * _FINE_A * np.cos(2.0 * np.pi * _FINE_K * (fx * u + fy * v) + p) * 2.0 * np.pi * _FINE_K
        du = du + c * fx
        dv = dv + c * fy
    return du, dv


SCENES = {
    # The dome and the shallow dome are sampled on their FLANK (r ~ half the radius), where the
    # slope is near its maximum.  Their peaks are flat by construction and a window there measures
    # nothing.
    "dome":    Scene("dome",    _dome,    _dome_d,
                     "smooth, C1 everywhere -- pure reconstruction test", centre=(0.675, 0.5)),
    "bevel":   Scene("bevel",   _bevel,   _bevel_d,
                     "sharp crease -- edge behaviour", centre=(0.5, 0.5)),
    "noise":   Scene("noise",   _noise,   _noise_d,
                     "band-limited high frequency -- aliasing and mips", centre=(0.5, 0.5)),
    "shallow": Scene("shallow", _shallow, _shallow_d,
                     "3/255 of range -- the 8-bit terracing floor", centre=(0.7, 0.5)),
    # `noise` scaled up to ~6 texels per cycle at 4096.  The CONTROL for the resolution test: here
    # 4096 carries an octave 2048 physically cannot hold, which is the case `noise` cannot speak to
    # and the case that would justify the 4096 default if anything does.
    "fine":    Scene("fine",    _fine,    _fine_d,
                     "near Nyquist at 4096, aliased at 1024 -- the control for T5", centre=(0.5, 0.5)),
}


# =======================================================================================
# 3.  THE TEXTURE, AND HOW IT IS SAMPLED
# =======================================================================================


def build_texture(scene, N, M, u0, v0, ss=4):
    """An M x M window of an N x N texture of `scene`, AREA-AVERAGED over each texel.

    Area-averaging rather than point-sampling because that is what a resampler does and because it
    is the half of the argument that favours storing height: height is LINEAR under filtering, so
    averaging it is meaningful.  (A normal map is not, which is a separate reason this project went
    to height and is not what the rig is measuring.)

    Only a window is materialised.  The resolution sweep changes the TEXEL SIZE, so a fixed world
    window costs M = window * N texels -- 64 at 512, 512 at 4096 -- and nothing here ever allocates
    a 4096-square array.
    """
    tx = 1.0 / N
    # texel centres of the window
    i = np.arange(M, dtype=np.float64)
    # sub-samples inside each texel, at (k+0.5)/ss of its extent
    k = (np.arange(ss, dtype=np.float64) + 0.5) / ss - 0.5
    uu = ((u0 + i[:, None] + 0.5) + k[None, :]).reshape(-1) * tx
    vv = ((v0 + i[:, None] + 0.5) + k[None, :]).reshape(-1) * tx
    U, V = np.meshgrid(uu, vv, indexing="xy")
    H = scene.h(U, V)
    # average the ss x ss block belonging to each texel
    H = H.reshape(M, ss, M, ss).mean(axis=(1, 3))
    return H


def bilinear(tex, x, y):
    """Texel-space bilinear.  x, y are in TEXEL COORDINATES with centres at integers, which is the
    convention uv*N - 0.5 produces.  Clamped at the window edge: the window is padded by the
    callers so no test ever reads the clamp."""
    h, w = tex.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    x0 = np.clip(x0, 0, w - 2)
    y0 = np.clip(y0, 0, h - 2)
    a = tex[y0, x0]
    b = tex[y0, x0 + 1]
    c = tex[y0 + 1, x0]
    d = tex[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _cr_w(t):
    """Catmull-Rom basis at fractional position t, for the 4 taps at -1, 0, +1, +2."""
    t2, t3 = t * t, t * t * t
    return (np.stack([
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ], axis=0))


def _cr_dw(t):
    """d/dt of the Catmull-Rom basis -- the ANALYTIC derivative of the interpolant, not a
    difference of it.  This is the whole C1 candidate: Catmull-Rom is C1, so this derivative is
    CONTINUOUS across texel-cell boundaries, where bilinear's is not."""
    t2 = t * t
    return (np.stack([
        -1.5 * t2 + 2.0 * t - 0.5,
        4.5 * t2 - 5.0 * t,
        -4.5 * t2 + 4.0 * t + 0.5,
        1.5 * t2 - 1.0 * t,
    ], axis=0))


def _bs_w(t):
    """Uniform cubic B-spline basis for the 4 taps at -1, 0, +1, +2.  APPROXIMATING, not
    interpolating: it does not pass through the texel values, it smooths them.  All four weights
    are non-negative, so it cannot ring -- and that is the property that matters once the texels
    are 8-bit, because Catmull-Rom's negative lobes are exactly what amplifies the quantiser."""
    t2, t3 = t * t, t * t * t
    return np.stack([
        (1.0 - t) ** 3 / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ], axis=0)


def _bs_dw(t):
    """d/dt of the B-spline basis.  The spline is C2, so this derivative is C1 -- continuous AND
    smooth across cell boundaries, which is one order more than Catmull-Rom gives, and one order
    more is what the NORMAL needs: the normal is a function of the gradient, so a C1 height (CR)
    gives a C0 normal that still KINKS at every texel edge.  dw0, dw1 <= 0 and dw2, dw3 >= 0 on
    [0,1], so each same-sign pair folds into one bilinear fetch: 4 taps per axis on hardware."""
    t2 = t * t
    return np.stack([
        -0.5 * (1.0 - t) ** 2,
        1.5 * t2 - 2.0 * t,
        -1.5 * t2 + t + 0.5,
        0.5 * t2,
    ], axis=0)


def bicubic_value_and_grad(tex, x, y, kernel="cr"):
    """Cubic interpolated value and its exact partial derivatives, per texel.  kernel = "cr"
    (Catmull-Rom, interpolating, C1) or "bs" (uniform B-spline, approximating, C2)."""
    h, w = tex.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    W, DW = (_cr_w, _cr_dw) if kernel == "cr" else (_bs_w, _bs_dw)
    wx, dwx = W(fx), DW(fx)
    wy, dwy = W(fy), DW(fy)
    val = np.zeros_like(x, dtype=np.float64)
    gx = np.zeros_like(x, dtype=np.float64)
    gy = np.zeros_like(x, dtype=np.float64)
    for j in range(4):
        yy = np.clip(y0 + j - 1, 0, h - 1)
        for i in range(4):
            xx = np.clip(x0 + i - 1, 0, w - 1)
            t = tex[yy, xx]
            val += t * wx[i] * wy[j]
            gx += t * dwx[i] * wy[j]
            gy += t * wx[i] * dwy[j]
    return val, gx, gy            # gx, gy are per TEXEL


def mip_chain(tex, levels):
    """Box-filtered mip chain, the 2x2 average the hardware generates."""
    out = [tex]
    cur = tex
    for _ in range(levels - 1):
        h, w = cur.shape
        if h < 2 or w < 2:
            break
        cur = cur[:h - h % 2, :w - w % 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        out.append(cur)
    return out


def sample_trilinear(mips, x, y, lod):
    """Trilinear across a box-filtered chain.  lod is log2(texels per pixel), clamped at 0."""
    lod = max(0.0, float(lod))
    l0 = int(math.floor(lod))
    l1 = min(l0 + 1, len(mips) - 1)
    l0 = min(l0, len(mips) - 1)
    f = lod - math.floor(lod)
    s0 = bilinear(mips[l0], (x + 0.5) / (2 ** l0) - 0.5, (y + 0.5) / (2 ** l0) - 0.5)
    if l1 == l0 or f == 0.0:
        return s0
    s1 = bilinear(mips[l1], (x + 0.5) / (2 ** l1) - 0.5, (y + 0.5) / (2 ** l1) - 0.5)
    return s0 * (1 - f) + s1 * f


# =======================================================================================
# 4.  AXIS 1 -- GETTING dH/duv OUT OF THE TEXTURE.  This is where magnification lives.
# =======================================================================================
#
# Every arm returns (dh/du, dh/dv) in PER-UV units, so they are all estimating the same quantity
# and the comparison is about RECONSTRUCTION rather than about gain.
#
# !! THE LIVE SHADER DOES NOT NORMALISE, AND THAT IS A SEPARATE FINDING, NOT A HANDICAP APPLIED
# HERE.  It uses `dhdu = hR - hL` raw with a static `heightScale = 16` and offsets of +-1.5 BASE
# texels, so its effective per-uv gain is 16 * N / 3 -- PROPORTIONAL TO RESOLUTION.  Re-encoding a
# 4096 _paramh to 2048 therefore HALVES its apparent bump depth on top of any reconstruction
# change, which would contaminate the resolution A/B beyond reading.  T9 measures that coupling on
# its own; every other test normalises it away so that what is left is the reconstruction.


class Sampling(object):
    """One screen's worth of sampling geometry over a texture window."""

    def __init__(self, N, M, u0, v0, tpp, pixels, pad=6.0):
        self.N, self.M, self.u0, self.v0, self.tpp = N, M, u0, v0, tpp
        self.lod = math.log2(tpp) if tpp > 0 else 0.0
        i = np.arange(pixels, dtype=np.float64)
        # CENTRED on the window, not anchored at its corner.  The first version anchored at `pad`
        # and the screen then covered texels 6..30 of a 128-texel window -- so the `bevel` scene's
        # crease, which sits at the CENTRE of the texture by construction, was never on screen and
        # every arm scored exactly 0.000 deg on it.  Four arms agreeing perfectly on an edge test
        # is the tell that the edge was not in the frame.
        xt = 0.5 * M + (i - 0.5 * pixels + 0.5) * tpp
        self.xt, self.yt = np.meshgrid(xt, xt, indexing="xy")     # window-texel coords
        self.u = (self.u0 + self.xt + 0.5) / N
        self.v = (self.v0 + self.yt + 0.5) / N
        self.du_dx = tpp / N                                      # ddx(uv).x
        self.dv_dy = tpp / N                                      # ddy(uv).y
        self.pixels = pixels

    def fits(self):
        # 4 texels of slack for the widest stencil in the rig (Catmull-Rom's -1..+2 plus the
        # central difference's 1.5), taken at the COARSEST mip any arm will select.
        m = 2.0 ** max(0.0, self.lod)
        return self.xt.max() + 4.0 * m < self.M and self.xt.min() - 4.0 * m > 0.0


def _quad_ddx(a):
    """HLSL ddx at QUAD granularity: both columns of a 2x2 quad get the same difference.  Faithful
    because it is the reason the ddx arm is blocky at a 2-pixel scale rather than smooth."""
    out = np.empty_like(a)
    out[:, 0::2] = a[:, 1::2] - a[:, 0::2]
    out[:, 1::2] = a[:, 1::2] - a[:, 0::2]
    return out


def _quad_ddy(a):
    out = np.empty_like(a)
    out[0::2, :] = a[1::2, :] - a[0::2, :]
    out[1::2, :] = a[1::2, :] - a[0::2, :]
    return out


def arm_cd(ctx, deriv=1.5, mip_aware=False):
    """LIVE SHADER.  Fixed central difference at +-`deriv` texels, bilinear/trilinear taps.

    `mip_aware` is the FIX, not the shipped behaviour: the live code offsets by 1/normres, i.e.
    BASE-level texels, while the hardware samples whichever mip the footprint selected.  At a mip
    2 levels down, four taps 1.5 base texels apart land inside ~0.4 of a mip texel -- all four
    inside one filtered footprint -- and the bump collapses far faster than the filtering itself
    would take it.  Parallax() directly above it in common.hlsl does the same job correctly with
    tex2Dgrad.  T8 is the measurement.
    """
    s = ctx.s
    off = deriv * (2.0 ** max(0.0, s.lod)) if mip_aware else deriv
    hL = sample_trilinear(ctx.mips, s.xt - off, s.yt, s.lod)
    hR = sample_trilinear(ctx.mips, s.xt + off, s.yt, s.lod)
    hD = sample_trilinear(ctx.mips, s.xt, s.yt - off, s.lod)
    hU = sample_trilinear(ctx.mips, s.xt, s.yt + off, s.lod)
    k = s.N / (2.0 * off)
    return (hR - hL) * k, (hU - hD) * k


def arm_fwd(ctx):
    """DerivFromHeightMap -- one-texel forward difference.  Half a texel of phase error by
    construction: it estimates the slope at u + 0.5 texels and reports it at u."""
    s = ctx.s
    hC = sample_trilinear(ctx.mips, s.xt, s.yt, s.lod)
    hR = sample_trilinear(ctx.mips, s.xt + 1.0, s.yt, s.lod)
    hU = sample_trilinear(ctx.mips, s.xt, s.yt + 1.0, s.lod)
    return (hR - hC) * s.N, (hU - hC) * s.N


def arm_sobel(ctx):
    """DerivFromHeightMapHQ -- Sobel 3x3, normalised by 8.  Nine taps buy noise rejection and cost
    a low-pass: it is a central difference pre-blurred along the perpendicular axis."""
    s = ctx.s
    g = {}
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            g[(dx, dy)] = sample_trilinear(ctx.mips, s.xt + dx, s.yt + dy, s.lod)
    dHdx = (g[(1, -1)] + 2 * g[(1, 0)] + g[(1, 1)]) - (g[(-1, -1)] + 2 * g[(-1, 0)] + g[(-1, 1)])
    dHdy = (g[(-1, 1)] + 2 * g[(0, 1)] + g[(1, 1)]) - (g[(-1, -1)] + 2 * g[(0, -1)] + g[(1, -1)])
    return dHdx / 8.0 * s.N, dHdy / 8.0 * s.N


def arm_ddx(ctx):
    """ParallaxHeightNormal's ddx/ddy branch: the SCREEN-SPACE derivative of the sampled height.

    Tracks the mip for free -- it differentiates whatever the sampler returned -- but its scale is
    the PIXEL, not the texel, so under magnification it is a difference of two nearly-equal 8-bit
    samples spread over a 2x2 quad, which is quantisation noise amplified by 1/tpp."""
    s = ctx.s
    hC = sample_trilinear(ctx.mips, s.xt, s.yt, s.lod)
    return _quad_ddx(hC) / s.du_dx, _quad_ddy(hC) / s.dv_dy


def arm_fp(ctx):
    """ParallaxHeightNormal as written: lerp(cd, ddx, saturate(footprint - 1)).

    !! ITS OWN COMMENT IS INVERTED AND THE CODE IS RIGHT.  The source says "~0 when minified, ~1
    when magnified"; saturate(footprint - 1) is 0 when footprint < 1, which IS magnification, and
    reaches 1 only past 2 texels per pixel, which is minification.  Transcribed from the code.
    """
    cdu, cdv = arm_cd(ctx)
    dxu, dxv = arm_ddx(ctx)
    w = min(max(ctx.s.tpp - 1.0, 0.0), 1.0)
    return cdu * (1 - w) + dxu * w, cdv * (1 - w) + dxv * w


def arm_bicubic(ctx):
    """NEW -- the C1 candidate.  The ANALYTIC derivative of a Catmull-Rom interpolant.

    Catmull-Rom is C1, so this gradient is continuous ACROSS texel-cell boundaries where bilinear's
    is not.  That is the whole claim, and T3 is where it is tested rather than asserted: resolution
    shrinks the facets, C1 reconstruction removes them.  Taken on the mip the footprint selects
    (floor, not trilinear -- 32 taps across two levels is not a shader anyone would ship)."""
    s = ctx.s
    l = int(max(0.0, math.floor(s.lod)))
    l = min(l, len(ctx.mips) - 1)
    sc = 2.0 ** l
    _, gx, gy = bicubic_value_and_grad(ctx.mips[l], (s.xt + 0.5) / sc - 0.5, (s.yt + 0.5) / sc - 0.5)
    return gx * s.N / sc, gy * s.N / sc


def arm_bspline(ctx):
    """NEW -- the C2 candidate, and the one T3 showed was missing.  The analytic derivative of a
    uniform cubic B-spline over the height texels.  Two differences from `bicubic`, both deliberate:
    it is C2 rather than C1, so the NORMAL is continuous in its slope as well as its value; and its
    weights never go negative, so it low-passes the 8-bit staircase instead of sharpening it.  The
    price is that it is approximating -- it slightly flattens real detail near the source's own
    Nyquist -- which is what `retention` below 1 on the `fine` control measures."""
    s = ctx.s
    l = min(int(max(0.0, math.floor(s.lod))), len(ctx.mips) - 1)
    sc = 2.0 ** l
    _, gx, gy = bicubic_value_and_grad(ctx.mips[l], (s.xt + 0.5) / sc - 0.5, (s.yt + 0.5) / sc - 0.5,
                                       kernel="bs")
    return gx * s.N / sc, gy * s.N / sc


def arm_cdbs(ctx, deriv=1.5):
    """NEW -- the live central difference, with each of its four taps read through the cubic
    B-spline instead of bilinear.  Found by the rig rather than proposed by the plan, and it is the
    best RUNTIME arm on the table.

    Why it works where both of its parents half-fail.  `cd` has the right BASELINE (+-1.5 texels,
    so its difference averages the 8-bit staircase over 3 texels) on the wrong INTERPOLANT (bilinear
    taps, so every cell edge is a kink: facet enrichment 16, the ceiling).  `bspline` has the right
    interpolant (C2, facets ~1) on too narrow a baseline (its analytic derivative is effectively a
    ~1.3-texel difference, so it lets roughly twice the quantiser noise through).  The difference of
    two C2 samples is itself C2, so this keeps cd's baseline AND the spline's continuity.

    Cost: each tap is a B-spline read -- 4 bilinear fetches with the linear-filtering trick -- so 16
    fetches against cd's 4.  The price of a sharp crease: the spline rounds it, which is where this
    loses to cd (the `bevel` rows)."""
    s = ctx.s
    l = min(int(max(0.0, math.floor(s.lod))), len(ctx.mips) - 1)
    sc = 2.0 ** l

    def tap(dx, dy):
        return bicubic_value_and_grad(ctx.mips[l], (s.xt + dx + 0.5) / sc - 0.5,
                                      (s.yt + dy + 0.5) / sc - 0.5, kernel="bs")[0]
    k = s.N / (2.0 * deriv)
    return (tap(deriv, 0.0) - tap(-deriv, 0.0)) * k, (tap(0.0, deriv) - tap(0.0, -deriv)) * k


def arm_deriv5(ctx, cubic=False):
    """NEW -- the OFFLINE candidate.  dH/duv baked at full precision, stored as BC5, filtered
    directly.

    The argument for it is one line: differentiation and box-filtering COMMUTE, so the derivative
    of the averaged height IS the average of the derivative -- but the quantisation does not
    commute, and the baked map quantises AFTER differentiating while the runtime arms differentiate
    AFTER quantising.  Offline therefore wins on the gradient by construction.  What it cannot do
    is parallax, soft parallax shadows or height blending, all of which need the height field
    itself; that is the trade this rig prices rather than settles."""
    s = ctx.s
    mips = ctx.dmips_c if cubic else ctx.dmips
    if cubic:
        l = min(int(max(0.0, math.floor(s.lod))), len(mips[0]) - 1)
        sc = 2.0 ** l
        du = bicubic_value_and_grad(mips[0][l], (s.xt + 0.5) / sc - 0.5, (s.yt + 0.5) / sc - 0.5)[0]
        dv = bicubic_value_and_grad(mips[1][l], (s.xt + 0.5) / sc - 0.5, (s.yt + 0.5) / sc - 0.5)[0]
    else:
        du = sample_trilinear(mips[0], s.xt, s.yt, s.lod)
        dv = sample_trilinear(mips[1], s.xt, s.yt, s.lod)
    return du * ctx.drange, dv * ctx.drange


GRAD_ARMS = {
    "cd":       lambda c: arm_cd(c),
    "fwd":      arm_fwd,
    "sobel":    arm_sobel,
    "ddx":      arm_ddx,
    "fp":       arm_fp,
    "bicubic":  arm_bicubic,
    "bspline":  arm_bspline,
    "cdbs":     lambda c: arm_cdbs(c),
    "deriv5":   lambda c: arm_deriv5(c, False),
    "deriv5c":  lambda c: arm_deriv5(c, True),
}
GRAD_ORDER = ["cd", "fwd", "sobel", "ddx", "fp", "bicubic", "bspline", "cdbs", "deriv5", "deriv5c"]


# =======================================================================================
# 5.  AXIS 2 -- TURNING dH/duv INTO A NORMAL.  This is where the missing tangents live.
# =======================================================================================
#
# MW meshes have no tangents, so both arms here RECOVER a frame from screen-space derivatives.
# They agree exactly on an orthonormal UV map -- verified in T7 -- and the question is what happens
# when the mapping is skewed or anisotropically scaled, which on MW art is the normal case rather
# than the exception (UV islands packed by hand, non-uniform object scale on the node above).


class Geom(object):
    """A plane with a possibly-skewed UV map, displaced along its normal by disp * h."""

    def __init__(self, Tu, Tv, disp):
        self.Tu = np.asarray(Tu, dtype=np.float64)
        self.Tv = np.asarray(Tv, dtype=np.float64)
        n = np.cross(self.Tu, self.Tv)
        self.n = n / np.linalg.norm(n)
        self.disp = float(disp)
        # The one scalar the tangent-space arm has to stand in for a frame.  Correct when the map
        # is orthonormal; under skew there is no value that is right for both axes at once, and
        # that is the point of T7 rather than a calibration this rig gets to choose.
        self.hs = disp / (0.5 * (np.linalg.norm(self.Tu) + np.linalg.norm(self.Tv)))


ORTHO = Geom((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.06)
# 1.6:1 anisotropy plus a 20-degree shear -- a packed UV island, not a pathological case.
SKEW = Geom((1.6, 0.0, 0.0), (0.58, 1.0, 0.0), 0.06)


def true_normal(geom, dhdu, dhdv):
    """Closed form.  P = O + u*Tu + v*Tv + disp*h*n, so dP/du = Tu + disp*dh/du*n exactly."""
    dPu = geom.Tu[None, None, :] + (geom.disp * dhdu)[:, :, None] * geom.n[None, None, :]
    dPv = geom.Tv[None, None, :] + (geom.disp * dhdv)[:, :, None] * geom.n[None, None, :]
    nn = np.cross(dPu, dPv)
    return nn / np.linalg.norm(nn, axis=2, keepdims=True)


def frame_tbn(geom, s, dhdu, dhdv):
    """LIVE SHADER: BuildPerPixelTBN's cotangent frame, then rotate the tangent-space normal.

    The frame is scale-invariant by construction (rcpmax over max(|T|,|B|)), which is what makes it
    usable with no mesh tangents -- and is also what makes it lose the ANISOTROPY it just
    normalised away."""
    dp1 = geom.Tu * s.du_dx           # ddx(viewPos)
    dp2 = geom.Tv * s.dv_dy           # ddy(viewPos)
    duv1 = np.array([s.du_dx, 0.0])
    duv2 = np.array([0.0, s.dv_dy])
    dp2perp = np.cross(dp2, geom.n)
    dp1perp = np.cross(geom.n, dp1)
    T = dp2perp * duv1[0] + dp1perp * duv2[0]
    B = dp2perp * duv1[1] + dp1perp * duv2[1]
    rcpmax = 1.0 / math.sqrt(max(T.dot(T), B.dot(B)))
    T, B = T * rcpmax, B * rcpmax
    nx = -dhdu * geom.hs
    ny = -dhdv * geom.hs
    nz = np.ones_like(nx)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx * inv, ny * inv, nz * inv
    out = (nx[:, :, None] * T[None, None, :] +
           ny[:, :, None] * B[None, None, :] +
           nz[:, :, None] * geom.n[None, None, :])
    return out / np.linalg.norm(out, axis=2, keepdims=True)


def frame_surfgrad(geom, s, dhdu, dhdv):
    """SurfgradScaleDependent + ResolveNormalFromSurfaceGradient (Mikkelsen).  No TBN, no
    orthonormality assumption: it works in WORLD height units and divides by the real Jacobian
    determinant, so an anisotropic or sheared map is carried rather than normalised away."""
    dHdu = geom.disp * dhdu
    dHdv = geom.disp * dhdv
    dPdx = geom.Tu * s.du_dx
    dPdy = geom.Tv * s.dv_dy
    dHdx = dHdu * s.du_dx            # dot(dHduv, ddx(uv)); ddx(uv) = (du_dx, 0)
    dHdy = dHdv * s.dv_dy            # ddy(uv) = (0, dv_dy)
    vR1 = np.cross(dPdy, geom.n)
    vR2 = np.cross(geom.n, dPdx)
    det = float(dPdx.dot(vR1))
    sc = (1.0 if det >= 0.0 else -1.0) / max(1.192093e-15, abs(det))
    sg = (sc * dHdx)[:, :, None] * vR1[None, None, :] + (sc * dHdy)[:, :, None] * vR2[None, None, :]
    out = geom.n[None, None, :] - sg
    return out / np.linalg.norm(out, axis=2, keepdims=True)


FRAME_ARMS = {"tbn": frame_tbn, "surfgrad": frame_surfgrad}


# =======================================================================================
# 6.  THE CONTEXT: one (scene, resolution, codec, magnification) configuration
# =======================================================================================


def window_for(N, tpp, pixels):
    """The smallest world window whose texel grid holds the whole screen plus the widest stencil
    in the rig at the coarsest mip that tpp selects.  T4 used to pick a fixed window and SKIP any
    row that did not fit, which silently dropped tpp = 16 from the sweep."""
    m = 2.0 ** max(0.0, math.log2(tpp)) if tpp > 0 else 1.0
    need = pixels * tpp + 2.0 * (4.0 * m + 8.0)
    need = int(math.ceil(need / 64.0)) * 64                  # a multiple of 64: every mip is BC4-able
    return min(1.0, need / float(N))


class Ctx(object):
    def __init__(self, scene, N, tpp, codec="bc4", pixels=192, window_world=0.125,
                 u0f=None, v0f=None):
        if u0f is None:
            u0f, v0f = scene.centre
        self.scene = scene
        self.N = N
        self.codec = codec
        # Matched WORLD size across the resolution sweep: the window covers the same patch of
        # surface at every N, so M scales with N and the comparison is not secretly about how much
        # of the scene is on screen.
        M = int(round(window_world * N))
        self.M = M
        self.u0 = int(round(u0f * N)) - M // 2
        self.v0 = int(round(v0f * N)) - M // 2
        h = build_texture(scene, N, M, self.u0, self.v0)
        # A DDS stores EVERY MIP AS ITS OWN BLOCKS: the chain is built from the full-precision
        # source and each level is compressed independently.  The first version box-filtered the
        # already-compressed base instead, which handed the coarse levels a precision no shipped
        # file has -- and those levels are exactly where the minification tests live.
        self.mips = [CODECS[codec](m) for m in mip_chain(h, 6)]
        self.height = self.mips[0]

        # The BAKED derivative map: analytic dh/duv, area-averaged over each texel exactly as the
        # height was, then scaled into [-1,1] by a per-texture range and pushed through real BC5
        # (two independent BC4_SNORM blocks).  Quantised AFTER differentiating -- which is the
        # whole offline argument in one line.
        du, dv = self._bake_derivative(M)
        self.drange = float(max(np.abs(du).max(), np.abs(dv).max(), 1e-12))
        # Same rule as the height: filter the full-precision derivative, THEN compress each level.
        # Filtering first is exact here -- the box filter commutes with differentiation, which is
        # the whole reason a derivative map mips correctly and a normal map does not.
        cu, cv = mip_chain(du / self.drange, 6), mip_chain(dv / self.drange, 6)
        if codec == "none":
            self.dmips = (cu, cv)
        else:
            pairs = [bc5_roundtrip(a, b, signed=True) for a, b in zip(cu, cv)]
            self.dmips = ([p[0] for p in pairs], [p[1] for p in pairs])
        self.dmips_c = self.dmips

        self.s = Sampling(N, M, self.u0, self.v0, tpp, pixels)

    def _bake_derivative(self, M, ss=4):
        tx = 1.0 / self.N
        i = np.arange(M, dtype=np.float64)
        k = (np.arange(ss, dtype=np.float64) + 0.5) / ss - 0.5
        uu = ((self.u0 + i[:, None] + 0.5) + k[None, :]).reshape(-1) * tx
        vv = ((self.v0 + i[:, None] + 0.5) + k[None, :]).reshape(-1) * tx
        U, V = np.meshgrid(uu, vv, indexing="xy")
        du, dv = self.scene.dh(U, V)
        return (du.reshape(M, ss, M, ss).mean(axis=(1, 3)),
                dv.reshape(M, ss, M, ss).mean(axis=(1, 3)))

    def truth_point(self):
        return self.scene.dh(self.s.u, self.s.v)

    def truth_footprint(self, ss=8):
        """The analytic gradient AVERAGED OVER THE PIXEL FOOTPRINT.  Under minification the point
        normal is not what a correct renderer should produce, so scoring against it would punish
        filtering for being filtering.  This is the reference the retention metric uses instead."""
        s = self.s
        k = (np.arange(ss, dtype=np.float64) + 0.5) / ss - 0.5
        acc_u = np.zeros_like(s.u)
        acc_v = np.zeros_like(s.v)
        for a in k:
            for b in k:
                du, dv = self.scene.dh(s.u + a * s.tpp / s.N, s.v + b * s.tpp / s.N)
                acc_u += du
                acc_v += dv
        return acc_u / (ss * ss), acc_v / (ss * ss)


# =======================================================================================
# 7.  METRICS
# =======================================================================================


def angular_error_deg(a, b):
    d = np.clip(np.sum(a * b, axis=2), -1.0, 1.0)
    return np.degrees(np.arccos(d))


def score(ctx, gradarm, framearm, geom=ORTHO, footprint_truth=False):
    dhdu, dhdv = GRAD_ARMS[gradarm](ctx)
    tu, tv = ctx.truth_footprint() if footprint_truth else ctx.truth_point()
    nrec = FRAME_ARMS[framearm](geom, ctx.s, dhdu, dhdv)
    ntru = true_normal(geom, tu, tv)
    e = angular_error_deg(nrec, ntru)
    # Bump RETENTION: the reconstructed slope magnitude against the reference's, RMS.  1.0 is
    # correct; below 1 the bump has collapsed early, above 1 it has been over-sharpened.
    mrec = np.sqrt(np.mean(dhdu ** 2 + dhdv ** 2))
    mtru = np.sqrt(np.mean(tu ** 2 + tv ** 2))
    return {
        "mean": float(np.mean(e)),
        "p99": float(np.percentile(e, 99.0)),
        "retention": float(mrec / mtru) if mtru > 1e-12 else float("nan"),
    }


def facet_ratio(scene, N, gradarm, codec="bc4", spp=32, geom=ORTHO, phases=16):
    """THE C1 METRIC, and the one the resolution question turns on.

    Scan a line across the surface at `spp` samples per texel, reconstruct the normal, take its
    second difference, and ask how much of that energy sits ON texel-cell boundaries rather than
    spread evenly.  Report the ENRICHMENT:

        mean(d2^2 | on a boundary) / mean(d2^2 | anywhere)

    1.0 means a boundary looks like everywhere else -- a continuous gradient, no facets.  The
    ceiling is 1/(fraction of samples on a boundary), about spp/2, reached when ALL the curvature
    is concentrated at the cell edges, which is what a kink IS.  An enrichment ratio rather than
    across/within because the within-cell variance goes to zero for a piecewise-constant arm like
    `ddx`, and a ratio with zero underneath reported 1e24 and said nothing.

    !! THE BOUNDARY LATTICE IS SEARCHED, NOT ASSUMED, and the first version of this test was wrong
    for exactly that reason.  It classified crossings of INTEGER texel coordinates, which is where
    the kinks are for an arm whose taps sit at integer offsets (fwd, sobel, deriv5) -- but the live
    central difference taps at +-1.5 texels, so its kinks land on HALF-integers and it scored 0.00,
    i.e. more continuous than a C1 interpolant.  A metric that has to be told where to look will
    find nothing wherever it was pointed wrongly.  So every phase is tried and the worst is
    reported; a genuinely C1 arm has no phase that lights up.
    """
    M = 96
    tpp = 1.0 / spp
    ctx = Ctx(scene, N, tpp, codec=codec, pixels=8, window_world=float(M) / N)
    # A 1-D scan, done as a thin 2-D strip so every sampler in the rig is reused unchanged.
    n = (M - 16) * spp
    xs = 8.0 + np.arange(n, dtype=np.float64) * tpp
    ys = np.full(n, 8.0 + 0.5 * tpp)
    ctx.s.xt = xs[None, :].repeat(2, axis=0)
    ctx.s.yt = ys[None, :].repeat(2, axis=0)
    ctx.s.u = (ctx.u0 + ctx.s.xt + 0.5) / N
    ctx.s.v = (ctx.v0 + ctx.s.yt + 0.5) / N
    ctx.s.pixels = n
    dhdu, dhdv = GRAD_ARMS[gradarm](ctx)
    nrec = FRAME_ARMS["tbn"](geom, ctx.s, dhdu, dhdv)
    nx = nrec[0, :, 0]
    d2 = nx[2:] - 2.0 * nx[1:-1] + nx[:-2]
    e2 = d2 * d2
    all_mean = float(np.mean(e2))
    if all_mean <= 1e-30:
        return float("nan")
    xc = xs[1:-1]
    best = 0.0
    for k in range(phases):
        phi = k / float(phases)
        crosses = np.floor(xc - tpp - phi) != np.floor(xc + tpp - phi)
        if crosses.sum() < 8:
            continue
        best = max(best, float(np.mean(e2[crosses])) / all_mean)
    return best


def facet_ceiling(spp):
    """What `facet_ratio` would report if every scrap of curvature sat on the cell edges."""
    return spp / 2.0


# =======================================================================================
# 8.  THE TESTS
# =======================================================================================

TESTS = []


def test(tid, title):
    def deco(fn):
        fn.tid, fn.title = tid, title
        TESTS.append(fn)
        return fn
    return deco


def _row(label, d, extra=""):
    print("  %-22s mean %7.3f deg   p99 %8.3f deg   retention %6.3f %s"
          % (label, d["mean"], d["p99"], d["retention"], extra))


@test("T0", "the block codecs are the real bit layouts, and they round-trip")
def t0():
    rng = np.random.default_rng(7)
    img = rng.random((16, 16))
    dec = bc4_roundtrip(img, False)
    err = float(np.abs(dec - img).max())
    # 8 levels on a per-block min/max ramp: the worst case is half a step of the block's own range.
    print("  BC4_UNORM random field      max abs error %.5f   (8 levels on the block range)" % err)
    # A CONSTANT block must be EXACT under both interpolation modes -- this is the property the
    # host-side --forge-scene gate leans on to make its four cases unambiguous.
    c = np.full((4, 4), 200.0 / 255.0)
    cd = bc4_roundtrip(c, False)
    print("  BC4_UNORM constant block    max abs error %.6f   (must be 0)" % float(np.abs(cd - c).max()))
    s = rng.random((16, 16)) * 2.0 - 1.0
    sd = bc4_roundtrip(s, True)
    print("  BC4_SNORM random field      max abs error %.5f" % float(np.abs(sd - s).max()))
    # The packing itself, not just the values: 16 three-bit indices survive a byte round-trip.
    idx = rng.integers(0, 8, 16)
    ok = np.array_equal(_unpack_indices(_pack_indices(idx)), idx)
    print("  3-bit index pack/unpack     %s" % ("exact" if ok else "MISMATCH"))
    b = bc4_encode_block(np.full(16, 0.5), False)
    print("  block size                  %d bytes (BC4) / %d bytes (BC5, two of them)" % (len(b), 2 * len(b)))
    # The vectorised bulk codec every other test uses must be the byte-level one, exactly.
    for signed in (False, True):
        f = rng.random((32, 32)) * (2.0 if signed else 1.0) - (1.0 if signed else 0.0)
        d = float(np.abs(bc4_roundtrip(f, signed) - bc4_roundtrip_scalar(f, signed)).max())
        print("  fast vs byte-level (%s)  max abs difference %.1e   (must be 0)"
              % ("SNORM" if signed else "UNORM", d))
    # ...and the FILE decoder, which reads endpoints the way a real encoder wrote them, including
    # the e0 <= e1 mode the rig's own encoder never emits.
    raw = np.array([[40, 200, 0x88, 0xC6, 0xFA, 0xFF, 0x00, 0x00]], dtype=np.uint8)
    ref = bc4_decode_block(bytes(raw[0]), False)
    got = bc4_decode_file_blocks(raw)[0]
    print("  file decoder, e0<=e1 block  max abs difference %.1e   (must be 0)"
          % float(np.abs(ref - got).max()))


@test("T1", "the closed-form gradient IS the gradient (no reference implementation to be wrong)")
def t1():
    for name in ("dome", "bevel", "noise", "shallow"):
        sc = SCENES[name]
        rng = np.random.default_rng(3)
        u = rng.random((64, 64)) * 0.6 + 0.2
        v = rng.random((64, 64)) * 0.6 + 0.2
        if name == "bevel":
            u = u * 0.03 + 0.48                      # stay inside the ramp, away from its corners
        e = 1e-6
        fdu = (sc.h(u + e, v) - sc.h(u - e, v)) / (2 * e)
        fdv = (sc.h(u, v + e) - sc.h(u, v - e)) / (2 * e)
        au, av = sc.dh(u, v)
        sc_mag = max(float(np.abs(au).max()), 1e-9)
        err = max(float(np.abs(au - fdu).max()), float(np.abs(av - fdv).max())) / sc_mag
        print("  %-8s  max relative disagreement %.2e   (%s)" % (name, err, sc.why))


@test("T2", "angular error by GRADIENT arm, magnified 8x (tpp = 0.125), BC4 height")
def t2():
    for name in ("dome", "bevel", "noise", "shallow"):
        print(" %s:" % name)
        ctx = Ctx(SCENES[name], 1024, 0.125, codec="bc4")
        for a in GRAD_ORDER:
            _row(a, score(ctx, a, "tbn"))


@test("T3", "THE FACET METRIC: how much of the normal's curvature sits ON texel-cell boundaries")
def t3():
    print("  Enrichment = mean(2nd difference squared | on a cell boundary) / mean(it anywhere).")
    print("  1.0 = a boundary looks like everywhere else, i.e. a continuous gradient and no")
    print("  facets. The CEILING is spp/2 = %.1f here, reached when ALL the curvature is at the"
          % facet_ceiling(32))
    print("  cell edges -- which is what a kink is. The boundary lattice is SEARCHED over every")
    print("  phase, so an arm cannot score well by having its kinks somewhere unexpected.")
    for name in ("dome", "noise"):
        print(" %s @ 1024, 32 samples per texel (ceiling %.1f):" % (name, facet_ceiling(32)))
        for a in GRAD_ORDER:
            r = facet_ratio(SCENES[name], 1024, a, spp=32)
            print("  %-22s enrichment %7.2f  (%.0f%% of ceiling)" % (a, r, 100.0 * r / facet_ceiling(32)))


@test("T4", "magnification sweep, 1/16 to 16 texels per pixel")
def t4():
    print("  tpp <= 1 is MAGNIFICATION and scores against the POINT normal.")
    print("  tpp >  1 is MINIFICATION and scores against the FOOTPRINT-AVERAGED normal, because")
    print("  the point normal is not what a correct renderer should produce there.")
    for tpp in (1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0, 2.0, 4.0, 8.0, 16.0):
        fp = tpp > 1.0
        pix = 128 if tpp <= 1.0 else 48
        ctx = Ctx(SCENES["noise"], 1024, tpp, codec="bc4", pixels=pix,
                  window_world=window_for(1024, tpp, pix))
        assert ctx.s.fits(), "window_for() must always fit (tpp %.3f)" % tpp
        print(" tpp = %6.3f  (%s):" % (tpp, "minified, footprint truth" if fp else "magnified"))
        for a in GRAD_ORDER:
            _row(a, score(ctx, a, "tbn", footprint_truth=fp))


@test("T5", "THE RESOLUTION QUESTION: does 4096 buy anything a better reconstruction would not?")
def t5():
    print("  Same world patch, same screen, same content -- only the texel size and the arm move.")
    print("")
    print("  !! WHAT THIS ROW DOES AND DOES NOT ANSWER.  The height field is the SAME SURFACE at")
    print("  every resolution, so this prices 4096 as a RESAMPLE of 2048.  If a 4096 map carries")
    print("  an octave of detail a 2048 map physically cannot hold, that octave is a separate")
    print("  purchase and this test says nothing about it.  The `step` column is what tells the")
    print("  two cases apart on real art: it is the median height change from one texel to the")
    print("  next, in units of the 8-bit quantiser.  Below about 1.0 the stored field no longer")
    print("  resolves its own slope and every runtime derivative of it is mostly quantiser noise,")
    print("  whatever the arm -- that is the resolution at which more texels stop being more")
    print("  information.")
    print("")
    print("  The facet metric is taken at a FIXED 32 samples per texel for every row, because its")
    print("  ceiling is spp/2: reading it at the row's own magnification made the 4096 ceiling 2.0")
    print("  and cd's 1.98 look like a pass when it was saturation.  Resolution's real contribution")
    print("  is the facet PITCH -- how big each facet is on screen -- and that is its own column.")
    ww = 0.03125
    pixels = 512
    print("")
    hdr = ("   %-6s %-9s %9s %10s %11s %13s %12s %8s %8s" %
           ("N", "arm", "mean deg", "p99 deg", "retention", "facet enrich", "facet pitch",
            "step", "flat"))
    print(" -- `noise`: 4096 is a RESAMPLE of the same surface --")
    print(hdr)
    for N in (512, 1024, 2048, 4096):
        tpp = ww * N / pixels
        ctx = Ctx(SCENES["noise"], N, tpp, codec="bc4", pixels=pixels, window_world=ww)
        dstep = np.abs(np.diff(ctx.height, axis=1))
        step = float(np.sqrt(np.mean(dstep ** 2))) * 255.0
        flat = 100.0 * float(np.mean(dstep < 0.5 / 255.0))
        for a in ("cd", "sobel", "bicubic", "bspline", "cdbs", "deriv5", "deriv5c"):
            d = score(ctx, a, "tbn")
            fr = facet_ratio(SCENES["noise"], N, a, spp=32)
            print("   %-6d %-9s %9.3f %10.3f %11.3f %13.2f %9.1f px %8.2f %7.1f%%" %
                  (N, a, d["mean"], d["p99"], d["retention"], fr, 1.0 / tpp, step, flat))
    print("")
    print(" -- `fine`: 4096 carries an octave 2048 CANNOT hold (the control) --")
    print(hdr)
    for N in (512, 1024, 2048, 4096):
        tpp = ww * N / pixels
        ctx = Ctx(SCENES["fine"], N, tpp, codec="bc4", pixels=pixels, window_world=ww)
        dstep = np.abs(np.diff(ctx.height, axis=1))
        step = float(np.sqrt(np.mean(dstep ** 2))) * 255.0
        flat = 100.0 * float(np.mean(dstep < 0.5 / 255.0))
        for a in ("cd", "sobel", "bicubic", "bspline", "cdbs", "deriv5", "deriv5c"):
            d = score(ctx, a, "tbn")
            fr = facet_ratio(SCENES["fine"], N, a, spp=32)
            print("   %-6d %-9s %9.3f %10.3f %11.3f %13.2f %9.1f px %8.2f %7.1f%%" %
                  (N, a, d["mean"], d["p99"], d["retention"], fr, 1.0 / tpp, step, flat))
    print("")
    print("   step = RMS height change from one texel to the next, in 8-bit quantiser units.")
    print("   flat = the share of neighbouring texel pairs that quantise to the SAME code, i.e.")
    print("   the share of central differences that come out EXACTLY ZERO however good the arm is.")
    print("   facet enrichment ceiling %.1f;  memory per _paramh (DXT5): 4096 = 21.3 MB," % facet_ceiling(32))
    print("   2048 = 5.3 MB, 1024 = 1.3 MB, 512 = 0.33 MB.  The live set is 208 x 4096 = 4.38 GB.")


@test("T6", "quantisation: none / u8 / BC4, on the field that lives at the 8-bit floor")
def t6():
    print("  `shallow` spans 3/255 of the range end to end. A defect invisible in the stored")
    print("  scalar is a staircase in its derivative, which is the sensitivity height has and")
    print("  roughness does not -- roughness is never differentiated.")
    for codec in ("none", "u8", "bc4"):
        print(" codec = %s:" % codec)
        ctx = Ctx(SCENES["shallow"], 1024, 0.125, codec=codec)
        for a in ("cd", "sobel", "bicubic", "deriv5"):
            _row(a, score(ctx, a, "tbn"))


@test("T7", "AXIS 2: cotangent TBN against the Mikkelsen surface gradient, orthonormal vs SKEWED")
def t7():
    print("  On an orthonormal map the two are algebraically identical and must agree to")
    print("  floating point. The question is the 1.6:1 + 20-degree-shear island.")
    for gname, geom in (("orthonormal", ORTHO), ("skewed 1.6:1 + shear", SKEW)):
        print(" %s:" % gname)
        ctx = Ctx(SCENES["dome"], 1024, 0.125, codec="bc4")
        for a in ("cd", "cdbs", "bicubic", "deriv5"):
            for f in ("tbn", "surfgrad"):
                _row("%s / %s" % (a, f), score(ctx, a, f, geom=geom))


@test("T8", "THE 'MIP BUG' HYPOTHESIS: do BASE-texel taps collapse the bump at distance?")
def t8():
    print("  The hypothesis, from the plan: the live code offsets its taps by 1/normres -- BASE")
    print("  texels -- while the sampler reads whichever mip it picked, so at distance all four")
    print("  taps land inside one filtered footprint and the bump collapses faster than it should.")
    print("")
    print("  What it gets wrong, and this table measures: the taps sample a CONTINUOUS function.")
    print("  Trilinear filtering of the mip chain is piecewise-linear between mip texels, so two")
    print("  taps 3 base texels apart difference that function across 3 base texels and return")
    print("  its slope -- the slope of the FILTERED field, which is exactly what a filtered")
    print("  surface should show.  Landing inside one footprint is not the same as landing on one")
    print("  VALUE.  The live gain (16 over a fixed 3-base-texel span) is a CONSTANT at every mip,")
    print("  so it scales these ratios by nothing.")
    print("")
    print("  `own`   = the exact slope of what the sampler returns (taps 0.02 base texels apart).")
    print("  `truth` = the analytic slope averaged over the pixel footprint (a perfect box).")
    print("  `LIVE`  = +-1.5 BASE texels, as shipped.   `aware` = +-1.5 MIP texels, the proposed fix.")
    print("  All are mean slope magnitudes in the same per-uv units.")
    print("")
    print("   %5s %11s %11s %11s %13s %12s" %
          ("tpp", "own/truth", "LIVE/own", "aware/own", "LIVE/truth", "aware/truth"))
    res = []
    for tpp in (1.0, 2.0, 4.0, 8.0, 16.0):
        pix = 48
        ctx = Ctx(SCENES["noise"], 1024, tpp, codec="bc4", pixels=pix,
                  window_world=window_for(1024, tpp, pix))
        assert ctx.s.fits()
        tu, tv = ctx.truth_footprint()
        mag = lambda a: float(np.mean(np.sqrt(a[0] ** 2 + a[1] ** 2)))
        truth = mag((tu, tv))
        own = mag(arm_cd(ctx, deriv=0.02))
        live = mag(arm_cd(ctx))
        aware = mag(arm_cd(ctx, mip_aware=True))
        print("   %5.1f %11.3f %11.3f %11.3f %13.3f %12.3f" %
              (tpp, own / truth, live / own, aware / own, live / truth, aware / truth))
        res.append((tpp, own / truth, live / own, aware / truth, live / truth))
    print("")
    # The verdict is COMPUTED from the rows above, not written in advance.  The first draft of
    # this paragraph said "LIVE/own stays near 1.0", and at tpp 16 it is 0.73.
    beats = all(r[4] >= r[3] for r in res)
    worst = min(res, key=lambda r: r[2])
    print("   LIVE beats the mip-aware 'fix' against truth at every distance: %s." %
          ("YES" if beats else "NO"))
    print("   LIVE/own is within %.0f%% of 1 through tpp 4 and bottoms at %.2f (tpp %.0f)." %
          (100.0 * max(abs(1.0 - r[2]) for r in res if r[0] <= 4.0), worst[2], worst[0]))
    print("   That residual loss is the taps STRADDLING mip-cell boundaries once the selected mip")
    print("   is near its own Nyquist (noise's 29 cycles over a 64-texel mip 4 is ~2.2 texels per")
    print("   cycle), and own/truth -- the mip chain's own attenuation -- stacks on it.  Widening")
    print("   the stencil to MIP texels is a second low-pass on top of both, which is why every")
    print("   'aware' ratio is lower.  So there is no collapse the offset can be blamed for: the")
    print("   base-texel offset is the better of the two at every distance measured.")
    print("   (Isotropic footprints only; anisotropic filtering at grazing angles and the")
    print("   hardware's 8-bit bilinear weight precision are not modelled.)")


@test("T9", "heightScale = 16 is RESOLUTION-COUPLED: the same art reads as a different depth")
def t9():
    print("  The live shader multiplies the RAW 3-texel difference by a static 16 with no division")
    print("  by the tap spacing, so its effective per-uv gain is 16*N/3 -- PROPORTIONAL TO N.")
    print("  Re-encoding a 4096 map to 2048 therefore halves its apparent bump depth with no edit")
    print("  to any shader or ini.  Two consequences: the resolution A/B in T5 has to normalise")
    print("  this away or it is measuring gain rather than reconstruction, and 'retune heightScale'")
    print("  is not a fix for a gradient defect -- it changes how DEEP the facets look, not how")
    print("  many of them there are.")
    print("")
    print("  `slope` is the mean tangent-space slope, which is linear in the gain; `tilt` is its")
    print("  angle, which is not, and is shown because that is what the eye reads.")
    rows = []
    for N in (512, 1024, 2048, 4096):
        ctx = Ctx(SCENES["dome"], N, 0.125, codec="bc4", pixels=128, window_world=0.03125)
        sp = ctx.s
        hL = sample_trilinear(ctx.mips, sp.xt - 1.5, sp.yt, sp.lod)
        hR = sample_trilinear(ctx.mips, sp.xt + 1.5, sp.yt, sp.lod)
        hD = sample_trilinear(ctx.mips, sp.xt, sp.yt - 1.5, sp.lod)
        hU = sample_trilinear(ctx.mips, sp.xt, sp.yt + 1.5, sp.lod)
        nx, ny = (hR - hL) * 16.0, (hU - hD) * 16.0          # LIVE, un-normalised
        slope = float(np.mean(np.sqrt(nx * nx + ny * ny)))
        rows.append((N, slope, float(np.mean(np.degrees(np.arctan(np.sqrt(nx * nx + ny * ny)))))))
    ref = rows[-1][1]
    print("   %-6s %12s %14s %14s" % ("N", "mean slope", "vs 4096", "mean tilt deg"))
    for N, slope, tilt in rows:
        print("   %-6d %12.4f %13.2fx %14.2f" % (N, slope, slope / ref, tilt))
    print("")
    print("   The slope column is linear in N by construction -- 512 reads 8x the depth of 4096")
    print("   for the same surface.  The tilt column compresses it through arctan, which is why")
    print("   the coupling is easy to mistake for 'the low-res map just looks a bit rougher'.")


# =======================================================================================
# 9.  REAL ART -- the shipped _paramh maps (optional: --paramh DIR)
# =======================================================================================
#
# Every test above is synthetic, and T5's control made the limit of that plain: whether 4096 buys
# anything depends on whether the map HAS content at 4096's Nyquist, which no synthetic scene can
# say about the shipped art.  This measures it on the shipped art directly.
#
# The question per map: does the top octave -- the band a 2048 map cannot hold -- carry height
# detail ABOVE THE MAP'S OWN CODEC NOISE?  If the energy in that band is what BC4 quantisation
# alone would put there, halving the map loses nothing but noise.
#
# Measured in GRADIENT energy, not height energy, because the shader differentiates: a band's
# contribution to the normal scales with frequency squared, which is also why quantisation noise
# (flat spectrum) dominates the top of the gradient spectrum long before it matters to the height.
#
# The noise estimate is read from the bitstream, block by block: a BC4 block quantises to its own
# ramp, step (e0 - e1)/7 (or /5 in the other mode), so its index error variance is step^2/12 --
# plus 1/12 LSB^2 for the 8-bit rounding texturematcher applies before compression
# (convert_to_8bit_single_channel).  !! It is treated as WHITE.  It is not exactly: the error is
# block-structured and puts some energy at the 4-texel period, which sits at the edge of the top
# band.  The ratio below is therefore an estimate, and a map near R = 1-2 is "mostly noise", not
# "provably noise".


def _calib_base(N=1024, seed=11):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:N, 0:N].astype(np.float64) / N
    base = np.zeros((N, N))
    for _ in range(24):                                      # band-limited below N/8 = 128 cycles
        fx, fy = rng.integers(-100, 101, 2)
        base += rng.normal(0.0, 1.0) / (1.0 + math.hypot(fx, fy)) * np.sin(
            2 * np.pi * (fx * x + fy * y) + rng.random() * 6.3)
    return base / np.abs(base).max(), x, y, rng


_EMPTY_CURVE = None


def empty_band_curve():
    """What the bitstream floor reads on a top octave that holds NOTHING but codec error, as a
    function of the MEASURED neighbour step -- the same statistic T10 reports per map, so a map is
    corrected by the calibration at its own regime.  The floor over-states the codec's share by a
    regime-dependent amount (most at the gentlest gradients, where rounding error forms terraces
    and is redder than white), and a single constant would be right at one end and wrong at the
    other.  Deterministic; computed once per run."""
    global _EMPTY_CURVE
    if _EMPTY_CURVE is None:
        base, _, _, _ = _calib_base()
        d = np.diff(base, axis=1)
        srms = float(np.sqrt(np.mean(d * d))) * 255.0
        pts = []
        for target in (0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.0, 3.0, 6.0):
            f = np.clip(0.5 + base * (target / srms), 0.0, 1.0)
            dec = bc4_roundtrip(f, False)
            B = _blocks(np.clip(np.round(f * 255.0), 0, 255))
            r = octave_analysis(dec * 255.0, B.max(axis=1), B.min(axis=1))
            pts.append((r["step_rms"], r["R_top1"], r["R_top2"], r["flat"]))
        pts.sort()
        _EMPTY_CURVE = np.array(pts)
    return _EMPTY_CURVE


def empty_r_at(step, band=1):
    c = empty_band_curve()
    return float(np.interp(step, c[:, 0], c[:, band]))


def _read_dxt5_alpha(path):
    import struct
    d = open(path, "rb").read()
    if len(d) < 128 or d[:4] != b"DDS ":
        return None, "not a DDS"
    h = struct.unpack_from("<I", d, 12)[0]
    w = struct.unpack_from("<I", d, 16)[0]
    fourcc = d[84:88]
    off = 128
    if fourcc == b"DX10":
        dxgi = struct.unpack_from("<I", d, 128)[0]
        if dxgi not in (77, 78):
            return None, "DX10 DXGI %d, not BC3" % dxgi
        off = 148
    elif fourcc != b"DXT5":
        return None, "FourCC %r, not DXT5" % fourcc
    bw, bh = w // 4, h // 4
    nb = bw * bh
    if off + nb * 16 > len(d):
        return None, "truncated"
    raw = np.frombuffer(d, dtype=np.uint8, count=nb * 16, offset=off).reshape(nb, 16)[:, :8]
    return (w, h, bw, bh, raw), None


def paramh_octave_scan(path):
    got, why = _read_dxt5_alpha(path)
    if got is None:
        return {"error": why}
    w, h, bw, bh, raw = got
    if w != h or w < 64 or (w & (w - 1)):
        return {"error": "not a square power of two (%dx%d)" % (w, h)}
    # Decode in chunks of block rows -- a 4096 map is a million blocks.
    H = np.empty((h, w), dtype=np.float64)
    rows = 64
    for r0 in range(0, bh, rows):
        r1 = min(bh, r0 + rows)
        dec = bc4_decode_file_blocks(raw[r0 * bw:r1 * bw])
        H[r0 * 4:r1 * 4, :] = _unblocks(dec, (r1 - r0) * 4, w)
    H *= 255.0                                               # LSB units from here on
    return octave_analysis(H, raw[:, 0].astype(np.float64), raw[:, 1].astype(np.float64))


def octave_analysis(H, a0, a1):
    """H in LSB units (N x N), a0/a1 the per-block BC4 endpoints in LSB.  Shared by the real-art
    scan and by its calibration, which is the point: the calibration can only vouch for the
    estimator if it runs the SAME code."""
    w = H.shape[1]
    step = np.where(a0 > a1, (a0 - a1) / 7.0, (a1 - a0) / 5.0)
    sigma2 = float(np.mean(step * step) / 12.0) + 1.0 / 12.0

    dx = np.diff(H, axis=1)
    step_rms = float(np.sqrt(np.mean(dx * dx)))
    flat = 100.0 * float(np.mean(np.abs(dx) < 0.5))

    N = w
    F = np.fft.rfft2(H - H.mean())
    P = (F.real ** 2 + F.imag ** 2) / float(N) ** 4
    kx = np.abs(np.fft.fftfreq(N) * N)[:, None]
    ky = (np.fft.rfftfreq(N) * N)[None, :]
    wgt = np.where((ky == 0) | (ky == N // 2), 1.0, 2.0)   # rfft stores half the plane
    g = (2.0 * np.pi / N) ** 2 * (kx * kx + ky * ky)        # gradient weight, per native texel
    kmax = np.maximum(kx, ky)
    # A SECOND, INDEPENDENT noise floor, read off the spectrum itself: the mean power in the
    # outermost sixteenth of the band before Nyquist, where hand-authored and scanned height art
    # has least content, taken as the level of a flat floor.  The bitstream estimate above knows
    # the codec's TOTAL error but must assume its spectrum is white; this one assumes nothing about
    # the codec but must assume that corner is empty of art.  Their biases point the same way --
    # both can only UNDER-state R -- so where they agree the verdict is not an artefact of either.
    corner = kmax > 0.4375 * N
    sigma2_c = float(np.mean(P[corner])) * float(N) ** 2
    out = {"N": N, "sigma2": sigma2, "sigma2_c": sigma2_c, "step_rms": step_rms, "flat": flat,
           "std": float(H.std())}
    for name, lo, hi in (("top1", N / 4.0, N / 2.0 + 1), ("top2", N / 8.0, N / 4.0)):
        band = (kmax > lo) & (kmax <= hi)
        gs = float(np.sum((P * g * wgt)[band]))
        gn = sigma2 / float(N) ** 2 * float(np.sum((g * wgt)[band]))
        out["R_" + name] = gs / gn if gn > 0 else float("inf")
        out["Rc_" + name] = gs / (gn * sigma2_c / sigma2) if gn > 0 and sigma2_c > 0 else float("inf")
        out["G_" + name] = gs
    gtot = float(np.sum(P * g * wgt))
    out["top1_share"] = out["G_top1"] / gtot if gtot > 0 else float("nan")
    return out


@test("T10a", "CALIBRATION of T10's estimator: a known-empty and a known-full top octave")
def t10a():
    print("  T10's verdict rests on one estimator, so it is run first on maps whose answer is KNOWN,")
    print("  through the same 8-bit + BC4 path texturematcher uses and the same analysis code.")
    print("  EMPTY: all content below N/8 -- the top octave holds nothing but codec error, so R must")
    print("  read ~1.  FULL: the same plus a component placed inside the top octave -- R must read")
    print("  well above 1.  A floor that is off by a factor shows up here as R off by that factor.")
    N = 1024
    base, x, y, rng = _calib_base(N)
    base = 0.5 + 0.18 * base
    top = np.zeros((N, N))
    for _ in range(12):                                      # inside the top octave: 300..480 cycles
        fx, fy = rng.integers(300, 481, 2) * rng.choice([-1, 1], 2)
        top += np.sin(2 * np.pi * (fx * x + fy * y) + rng.random() * 6.3)
    top = top / np.abs(top).max()
    for label, amp in (("EMPTY", 0.0), ("FULL, top-octave amplitude 1.5 LSB", 1.5 / 255.0),
                       ("FULL, top-octave amplitude 4 LSB", 4.0 / 255.0)):
        f = np.clip(base + amp * top, 0.0, 1.0)
        dec = bc4_roundtrip(f, False)
        B = _blocks(np.clip(np.round(f * 255.0), 0, 255))
        r = octave_analysis(dec * 255.0, B.max(axis=1), B.min(axis=1))
        print("   %-38s R_top1 %6.2f (bitstream) %6.2f (corner)   R_top2 %6.2f" %
              (label, r["R_top1"], r["Rc_top1"], r["R_top2"]))
    print("   (EMPTY's R_top2 is also a codec-only band here, since the art stops at N/8.)")
    print("")
    print("  !! THE CORNER FLOOR FAILS ITS CALIBRATION: it reads 1.18 on the 4-LSB map, whose top")
    print("  octave carries 3.4x the codec energy.  Art that reaches the Nyquist corner raises the")
    print("  very floor it is measured against.  T10 therefore reports it as a column and gives it")
    print("  NO vote in the verdict.")
    print("")
    print("  The bitstream floor, swept across the gradient regimes the shipped maps actually live")
    print("  in (T10's median neighbour step is ~1 LSB at 4096, where rounding error forms")
    print("  correlated TERRACES rather than white noise -- the case its white assumption fears):")
    c = empty_band_curve()
    for st, r1, r2, fl in c:
        print("   EMPTY at step %4.2f LSB (flat %4.1f%%)   R_top1 %5.2f   R_top2 %5.2f" % (st, fl, r1, r2))
    print("  An empty top octave reads %.2f-%.2f across that range: the floor OVER-states the" %
          (float(c[:, 1].min()), float(c[:, 1].max())))
    print("  codec's share by a regime-dependent amount.  So T10 does not use one constant: each map")
    print("  is divided by this curve read at ITS OWN measured step, which turns R into the band's")
    print("  real-detail-to-codec SNR in the regime that map actually lives in.")


@test("T10", "REAL ART: does each shipped _paramh's top octave carry height above its codec noise?")
def t10():
    import glob
    import os
    if not ARGS.paramh:
        print("  SKIPPED -- pass --paramh DIR (the folder holding the *_paramh.dds files).")
        return
    files = sorted(glob.glob(os.path.join(ARGS.paramh, "*_paramh*.dds")))
    if ARGS.paramh_limit:
        files = files[:ARGS.paramh_limit]
    print("  R = measured gradient energy in the band / what the map's own BC4 noise predicts, from")
    print("  the bitstream.  SNR = R / R_empty(step) - 1: the band's real-detail energy over its")
    print("  codec energy, with the floor's calibrated bias at THIS map's own step taken out (T10a).")
    print("  SNR < 1: the octave is at least HALF codec noise.  top1 = the octave a 2048 map cannot")
    print("  hold; top2 = the next one down.")
    print("  (The corner floor is printed per map in --paramh-verbose and the CSV; it failed its")
    print("  calibration in T10a and has no vote here.)")
    print("")
    rows = []
    for f in files:
        r = paramh_octave_scan(f)
        name = os.path.basename(f)
        if "error" in r:
            print("   %-44s  SKIPPED: %s" % (name[:44], r["error"]))
            continue
        rows.append((name, r))
        if ARGS.paramh_verbose:
            print("   %-40s N=%4d step %5.2f flat %5.1f%% sigma %4.2f/%4.2f  R_top1 %6.2f/%6.2f"
                  "  R_top2 %7.2f  top1 %4.1f%% of grad" %
                  (name[:40], r["N"], r["step_rms"], r["flat"], math.sqrt(r["sigma2"]),
                   math.sqrt(r["sigma2_c"]), r["R_top1"], r["Rc_top1"], r["R_top2"],
                   100.0 * r["top1_share"]))
    if not rows:
        print("  no readable maps")
        return
    if ARGS.paramh_csv:
        with open(ARGS.paramh_csv, "w") as fh:
            fh.write("file,N,step_rms_lsb,flat_pct,sigma_bitstream_lsb,sigma_corner_lsb,"
                     "R_top1,Rc_top1,R_top2,Rc_top2,top1_gradient_share,snr_top1,snr_top2\n")
            for name, r in rows:
                fh.write("%s,%d,%.4f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f,%.4f,%.4f\n" %
                         (name, r["N"], r["step_rms"], r["flat"], math.sqrt(r["sigma2"]),
                          math.sqrt(r["sigma2_c"]), r["R_top1"], r["Rc_top1"], r["R_top2"],
                          r["Rc_top2"], r["top1_share"],
                          r["R_top1"] / empty_r_at(r["step_rms"], 1) - 1.0,
                          r["R_top2"] / empty_r_at(r["step_rms"], 2) - 1.0))
        print("  wrote %s" % ARGS.paramh_csv)
    R1 = np.array([r["R_top1"] for _, r in rows])
    R2 = np.array([r["R_top2"] for _, r in rows])
    st = np.array([r["step_rms"] for _, r in rows])
    fl = np.array([r["flat"] for _, r in rows])
    sh = np.array([r["top1_share"] for _, r in rows])
    q = lambda a: "p10 %7.2f  median %7.2f  p90 %7.2f" % tuple(np.percentile(a, [10, 50, 90]))
    print("   maps scanned        %d" % len(rows))
    print("   neighbour step      %s LSB" % q(st))
    print("   flat neighbours     %s %%" % q(fl))
    S1 = np.array([r["R_top1"] / empty_r_at(r["step_rms"], 1) - 1.0 for _, r in rows])
    S2 = np.array([r["R_top2"] / empty_r_at(r["step_rms"], 2) - 1.0 for _, r in rows])
    print("   R_top1 (4096->2048) %s" % q(R1))
    print("   R_top2 (2048->1024) %s" % q(R2))
    print("   SNR top1            %s" % q(S1))
    print("   SNR top2            %s" % q(S2))
    print("   top-octave share of all gradient energy  %s" % q(100.0 * sh))
    big = np.array([r["N"] for _, r in rows]) >= 4096
    nb = int(big.sum())
    print("")
    print("   of the %d maps at 4096:" % nb)
    print("     top octave at least HALF codec noise (SNR < 1)   %3d" % int(np.sum(big & (S1 < 1.0))))
    print("     top octave mostly real detail (SNR >= 3)          %3d" % int(np.sum(big & (S1 >= 3.0))))
    print("     ...and the octave below ALSO SNR < 1              %3d" % int(np.sum(big & (S1 < 1.0) & (S2 < 1.0))))
    print("   A map whose top octave is SNR < 1 loses, by halving, mostly the codec's own error --")
    print("   and that error is what the shader's derivative amplifies most, since the gradient")
    print("   weights a band by frequency squared.")


ARGS = None


def main():
    ap = argparse.ArgumentParser(description="pbrsynth -- height-derived normal tests")
    ap.add_argument("--test", help="run one test by id, e.g. T5")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--paramh", help="T10: folder of shipped *_paramh.dds files (real-art scan)")
    ap.add_argument("--paramh-limit", type=int, default=0, help="T10: scan only the first N maps")
    ap.add_argument("--paramh-csv", help="T10: write the per-map table here")
    ap.add_argument("--paramh-verbose", action="store_true", help="T10: print every map")
    a = ap.parse_args()
    global ARGS
    ARGS = a
    if a.list:
        for t in TESTS:
            print("%-4s %s" % (t.tid, t.title))
        return 0
    sel = [t for t in TESTS if (a.test is None or t.tid.lower() == a.test.lower())]
    if not sel:
        print("no such test: %s" % a.test)
        return 2
    for t in sel:
        print("")
        print("=" * 96)
        print("%s  %s" % (t.tid, t.title))
        print("=" * 96)
        t()
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
