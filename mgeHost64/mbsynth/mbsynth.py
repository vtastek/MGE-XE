#!/usr/bin/env python3
"""
mbsynth -- synthetic tests for the MB-2 motion blur reconstruction filter.

WHY THIS EXISTS.  Every question asked of this filter so far has been asked with a
whole-frame aggregate (`changed%` out of gMbStats) against a screenshot, and a frame mean
cannot resolve a defect that lives on a silhouette -- order 1% of the pixels, inside a
+-0.1 spread.  A 10x sweep of mbSoftZ came back at 0.287 vs 0.288 for exactly that reason:
the metric was blind, not the knob inert.

Here the scene is analytic, so there is a GROUND TRUTH: a rect moving at a known velocity,
averaged over the exposure, IS the correct motion blur.  The filter can then be scored per
pixel against the right answer instead of against an opinion.

WHAT IS TRANSCRIBED.  mbtilemax.comp.fsl, mbneighbormax.comp.fsl, mbgather.comp.fsl and
mbcommon.h.fsl, line for line, as of MB-2h (tile jitter + two-direction sampling).  The
conventions that matter:
  * gMbDepth is RAW REVERSE-Z DEVICE DEPTH -- bigger is CLOSER.  d = 1/z here.
  * gMbVelocity is PREVIOUS MINUS CURRENT, in delivered px per FRAME, and is RG16F, so the
    field is quantised to fp16 exactly as the texture is.
  * `shutter` is exposure_ms / frameDt_ms, NOT angle/360.  180 deg at 30 fps ref = 16.67 ms,
    so at ~100 fps it is ~1.667.
  * The exposure window is CENTRED: taps span +- |v|*shutter/2 about now.  Ground truth
    integrates over exactly that window.

Run:  python3 mbsynth.py            (all scenes)
      python3 mbsynth.py --scene blade_front --dump
"""

import argparse
import math
import os
import struct
import sys
import zlib

import numpy as np

# ---------------------------------------------------------------------------------------
# mbcommon.h.fsl
# ---------------------------------------------------------------------------------------

def saturate(x):
    return np.clip(x, 0.0, 1.0)


def mb_velocity_texel(delivered_px, delivered_rect, mv_rect):
    """int2 mbVelocityTexel(float2, float2, float2) -- delivered pixel -> input texel."""
    n = (delivered_px + 0.5) / np.maximum(delivered_rect, 1.0)
    q = np.trunc(n * mv_rect).astype(np.int64)          # HLSL int() truncates toward zero
    return np.clip(q, 0, (mv_rect - 1).astype(np.int64))


def mb_depth_weight(d_centre, d_sample, extent):
    """saturate(1 + (dCentre/dSample - 1)/extent).  Both depths floored at 1e-7."""
    rel = np.maximum(d_centre, 1.0e-7) / np.maximum(d_sample, 1.0e-7) - 1.0
    return saturate(1.0 + rel / max(extent, 1.0e-4))


def mb_cone(dist, streak):
    return saturate(1.0 - dist / np.maximum(streak, 1.0e-4))


def smoothstep(a, b, x):
    t = saturate((x - a) / np.maximum(b - a, 1.0e-20))
    return t * t * (3.0 - 2.0 * t)


def mb_cylinder(dist, streak):
    L = np.maximum(streak, 1.0e-4)
    return 1.0 - smoothstep(0.95 * L, 1.05 * L, dist)


def mb_dither(px):
    """Interleaved-gradient noise, Jimenez 2014.  fp32, as the shader runs it."""
    p = px.astype(np.float32)
    d = (p[..., 0] * np.float32(0.06711056) + p[..., 1] * np.float32(0.00583715))
    inner = d - np.floor(d)
    v = np.float32(52.9829189) * inner
    return (v - np.floor(v)).astype(np.float64)


# ---------------------------------------------------------------------------------------
# The scene: axis-aligned opaque rects at constant view depth, moving at constant velocity.
# ---------------------------------------------------------------------------------------

class Layer(object):
    def __init__(self, name, rect, z, colour, motion=(0.0, 0.0), tex=None):
        self.name = name
        self.tex = tex                                # sampled in OBJECT space, so it rides
        self.rect = tuple(float(v) for v in rect)     # x0, y0, x1, y1 at s = 0
        self.z = float(z)                             # view distance; device depth = 1/z
        self.colour = np.array(colour, dtype=np.float64)
        self.motion = np.array(motion, dtype=np.float64)   # FORWARD screen px per frame

    @property
    def vfield(self):
        """What the velocity buffer holds: previous minus current."""
        return -self.motion


def rasterize(layers, W, H, s=0.0):
    """Point-sample the scene with each layer displaced by vfield*s.  Painter's, far first."""
    colour = np.zeros((H, W, 3), dtype=np.float64)
    depth = np.zeros((H, W), dtype=np.float64)          # reverse-Z: far = 0
    vel = np.zeros((H, W, 2), dtype=np.float64)

    xs = np.arange(W, dtype=np.float64)[None, :]
    ys = np.arange(H, dtype=np.float64)[:, None]

    for L in sorted(layers, key=lambda l: -l.z):        # far first, near paints over
        off = L.vfield * s
        x0, y0, x1, y1 = L.rect
        m = ((xs >= x0 + off[0]) & (xs < x1 + off[0]) &
             (ys >= y0 + off[1]) & (ys < y1 + off[1]))
        if L.tex is None:
            colour[m] = L.colour
        else:
            th, tw, _ = L.tex.shape
            iy, ix = np.nonzero(np.broadcast_to(m, (H, W)))
            colour[iy, ix] = L.tex[(iy - int(round(off[1]))) % th,
                                   (ix - int(round(off[0]))) % tw]
        depth[m] = 1.0 / L.z
        vel[m] = L.vfield
    return colour, depth, vel


def ground_truth(layers, W, H, shutter, nsub=513):
    """The correct answer: the scene averaged over the CENTRED exposure window."""
    acc = np.zeros((H, W, 3), dtype=np.float64)
    for k in range(nsub):
        s = (k + 0.5) / nsub - 0.5                      # in [-0.5, 0.5)
        c, _, _ = rasterize(layers, W, H, s * shutter)
        acc += c
    return acc / nsub


# ---------------------------------------------------------------------------------------
# mbtilemax.comp.fsl / mbneighbormax.comp.fsl
# ---------------------------------------------------------------------------------------

def tile_max(vel, P):
    K, W, H = P.K, P.W, P.H
    tx, ty = P.tiles
    out = np.zeros((ty, tx, 2), dtype=np.float64)
    for j in range(ty):
        for i in range(tx):
            y0, y1 = j * K, min((j + 1) * K, H)
            x0, x1 = i * K, min((i + 1) * K, W)
            blk = vel[y0:y1, x0:x1] * P.out_scale
            l2 = blk[..., 0] ** 2 + blk[..., 1] ** 2
            idx = np.unravel_index(np.argmax(l2), l2.shape)
            out[j, i] = blk[idx]
    return out


def neighbour_max(tiles):
    ty, tx, _ = tiles.shape
    out = np.zeros_like(tiles)
    for j in range(ty):
        for i in range(tx):
            best, bestl = np.zeros(2), -1.0
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    q = tiles[min(max(j + dj, 0), ty - 1), min(max(i + di, 0), tx - 1)]
                    l2 = q[0] ** 2 + q[1] ** 2
                    if l2 > bestl:
                        bestl, best = l2, q
            out[j, i] = best
    return out


# ---------------------------------------------------------------------------------------
# mbgather.comp.fsl
# ---------------------------------------------------------------------------------------

class Params(object):
    def __init__(self, W, H, K=96, shutter=1.667, max_taps=32, floor_px=0.5,
                 soft_z=0.1, tile_jitter=1.0, two_dir=True, out_scale=1.0,
                 arclen=False, fix='ship', gain=1.0, tap_jitter=1.0):
        self.W, self.H, self.K = W, H, K
        self.shutter, self.max_taps = shutter, max_taps
        self.floor_px, self.soft_z = floor_px, soft_z
        self.tile_jitter, self.two_dir = tile_jitter, two_dir
        self.out_scale = out_scale
        # RIG ONLY -- no shader has this.  Scales the per-pixel TAP phase hash (mbDither in
        # mbgather).  Setting it to 0 puts every pixel's taps at the same phase, so the
        # difference between 1 and 0 is, by construction, the ENTIRE contribution the hash
        # makes to the image: that difference IS the dither the user can see.
        self.tap_jitter = tap_jitter
        # RIG ONLY.  'white' is mbDither, the hash the shader actually uses.  'blue' is the
        # same uniform distribution with its LOW frequencies removed, which does not change
        # how big the sampling error is -- only where in the spectrum it sits.
        self.tap_noise = 'white'
        self.blue = None
        # ── c9: HOW MUCH of the background, and WHICH background, are different questions ──
        # The `b * cone(dist, selfStreak)` term answers the first one: what fraction of the
        # exposure am I swept off this pixel, so that whatever is behind me shows. c5's cap
        # makes that fraction come out right. But the COLOUR it fetches comes from OFFSETS
        # along the streak, and the thing being revealed sits AT THIS PIXEL -- for a static
        # background it never moved at all. The offsets are a SEARCH for somewhere the
        # background can be seen unoccluded, not a description of where it is, so the search
        # should prefer the NEAREST place it found one instead of averaging all of them
        # equally. c9 keeps the total (the amount) and re-weights only the mixture (the
        # provenance) by a cone `prox` times as wide as the streak.
        self.prox = 0.25
        # MB-2h's tile-fetch jitter REPLACES this pixel's tile with one up to K/2 away, and
        # `blurred` is a BINARY gate on the fetched speed -- so at the edge of a mover's dilated
        # neighbourhood the jitter decides per pixel between a full blur and none at all. That is
        # a stipple, and it is what the halftone band at an arm's silhouette is made of.
        # 'max' keeps the jitter but takes whichever of (own tile, jittered tile) is LONGER, so a
        # jittered fetch can only ever EXTEND the search region, never punch a hole in it.
        self.tile_jitter_mode = 'replace'
        # 'cone' scales the mixing cone by prox * selfStreak.  Measured: the best prox is
        # WIDTH-DEPENDENT (0.15 for an 8 px mover, 0.25 for a 32 px one) and a value too small
        # for the mover STARVES the bucket -- no tap is close enough to qualify, so the
        # estimate collapses onto one or two taps and its contrast overshoots past the truth
        # (std 1.93 at prox 0.08, width 32).  That is a radius that has to know how wide the
        # mover is, which the gather does not.
        # 'idw' weights by 1 / dist**p instead.  It has NO radius: whatever the nearest
        # unoccluded background sample turns out to be, it wins, and the weight never reaches
        # zero, so the bucket cannot starve however wide the mover is.
        self.prox_mode = 'cone'
        self.prox_p = 2.0
        # EXPERIMENTAL, not in any shader.  See t_arclen: weight each tap by the ARC LENGTH
        # of streak it stands for (|dir|/n) instead of giving every tap weight 1.  A tap is a
        # sampling site, not a vote; 16 taps crammed into a 5 px streak currently outvote 16
        # taps spread over 96 px by ~19x purely because nothing divides by sampling density.
        self.arclen = arclen
        # Which weighting to run. 'ship' must stay BIT-IDENTICAL to the deployed shader, and
        # T11(a) is the standing guard on that.
        self.fix = fix
        # ── WHY A GAIN OF 2 IS NOT A FUDGE ───────────────────────────────────────────────
        # mbCone(d, L) = 1 - d/L falls linearly to zero at d == L, so averaged over its own
        # support it is 0.5 -- it is a DENSITY, not a coverage probability. But the streak
        # spans +-L/2 about now, so the surface at distance d actually passes over the centre
        # iff d <= L/2, which as a function of d is a BOX of half-width L/2. And
        #     min(2 * (1 - d/L), 1)  ==  1 for d <= L/2, tapering to 0 at d == L
        # is exactly that box with a soft outer edge. So under the c5 cap, gain 2 converts
        # McGuire's density into the coverage function the integral actually wants. The sweep
        # below is the check that 2 is the optimum rather than a coincidence.
        self.gain = gain
        # c8 only: a correction on the composited coverage. Summing per-tap coverage
        # under-counts because the cone tapers at the ends of its support; whether ONE
        # constant repairs that across scenes is a question for measurement, not taste.
        self.alpha_scale = 1.0
        self.delivered = np.array([W, H], dtype=np.float64)
        self.mv_rect = np.array([W, H], dtype=np.float64)
        self.tiles = (int(math.ceil(W / float(K))), int(math.ceil(H / float(K))))


def gather(colour, depth, vel, tilev, P):
    H, W = depth.shape
    K = P.K
    shutter, soft_z = P.shutter, P.soft_z
    deliver_n = np.array([W, H], dtype=np.int64)

    ys, xs = np.mgrid[0:H, 0:W]
    pc = np.stack([xs, ys], axis=-1).astype(np.float64)

    vq = mb_velocity_texel(pc, P.delivered, P.mv_rect)
    v_self = vel[vq[..., 1], vq[..., 0]] * P.out_scale

    # MB-2h: jittered tile fetch, a SECOND hash, not the tap dither.
    if P.tile_jitter > 0.0:
        r1 = mb_dither(pc + 17.0)
        r2 = mb_dither(pc + 37.0)
        tj = (np.stack([r1, r2], -1) - 0.5) * (K * P.tile_jitter)
    else:
        tj = np.zeros_like(pc)
    tq = np.trunc((pc + tj) / float(max(K, 1))).astype(np.int64)
    tq[..., 0] = np.clip(tq[..., 0], 0, max(P.tiles[0] - 1, 0))
    tq[..., 1] = np.clip(tq[..., 1], 0, max(P.tiles[1] - 1, 0))
    v_tile = tilev[tq[..., 1], tq[..., 0]]
    if P.tile_jitter_mode == 'max' and P.tile_jitter > 0.0:
        to = np.trunc(pc / float(max(K, 1))).astype(np.int64)
        to[..., 0] = np.clip(to[..., 0], 0, max(P.tiles[0] - 1, 0))
        to[..., 1] = np.clip(to[..., 1], 0, max(P.tiles[1] - 1, 0))
        v_own = tilev[to[..., 1], to[..., 0]]
        take_own = (v_own[..., 0] ** 2 + v_own[..., 1] ** 2) > (v_tile[..., 0] ** 2 + v_tile[..., 1] ** 2)
        v_tile = np.where(take_own[..., None], v_own, v_tile)

    speed = np.linalg.norm(v_tile, axis=-1)
    self_streak = np.minimum(np.linalg.norm(v_self, axis=-1) * shutter, float(K))

    d = v_tile * shutter
    ln = np.linalg.norm(d, axis=-1)
    over = ln > float(K)
    d = np.where(over[..., None], d * (float(K) / np.maximum(ln, 1e-6))[..., None], d)
    ln = np.where(over, float(K), ln)

    blurred = (speed >= P.floor_px) & (ln >= 1.0e-4)
    taps = np.clip(np.ceil(ln), 3, max(P.max_taps, 3)).astype(np.int64)

    if P.tap_noise == 'blue':
        if P.blue is None or P.blue.shape != (H, W):
            P.blue = blue_noise(H, W)
        jitter = (P.blue - 0.5) * P.tap_jitter
    else:
        jitter = (mb_dither(pc) - 0.5) * P.tap_jitter
    d_self = v_self * shutter
    len_self = np.linalg.norm(d_self, axis=-1)
    o = len_self > float(K)
    d_self = np.where(o[..., None], d_self * (float(K) / np.maximum(len_self, 1e-6))[..., None], d_self)
    len_self = np.where(o, float(K), len_self)

    two_dir = (P.two_dir) & (len_self >= 1.0) & (taps >= 4)
    n_half = np.maximum(taps >> 1, 1)
    d_centre = depth[vq[..., 1], vq[..., 0]]

    acc = colour.copy()
    wsum = np.ones((H, W), dtype=np.float64)
    accB = np.zeros_like(colour)                 # c9: the revealed-background bucket, by
    wB = np.zeros((H, W), dtype=np.float64)      # PROXIMITY, kept apart from its own total
    covB = np.zeros((H, W), dtype=np.float64)
    accN = np.zeros_like(colour)                 # NEARER taps, kept apart so they composite
    wN = np.zeros((H, W), dtype=np.float64)
    accR = colour.copy()                         # everything else, centre included at weight 1
    wR = np.ones((H, W), dtype=np.float64)
    w_near = np.zeros((H, W), dtype=np.float64)   # tap NEARER than us  (mode 3 RED)
    w_same = np.zeros((H, W), dtype=np.float64)   # tap on OUR OWN surface
    w_far = np.zeros((H, W), dtype=np.float64)    # tap genuinely BEHIND us
    if P.arclen:
        # The centre stands for one tap's worth of the TILE axis, so the scale cancels and
        # the answer stops depending on the tap count.
        w0 = ln / np.maximum(taps.astype(np.float64), 1.0)
        acc = colour * w0[..., None]
        wsum = w0.copy()

    for i in range(int(taps.max())):
        active = blurred & (i < taps)
        if not active.any():
            break
        use_self = two_dir & ((i & 1) == 1)
        dirv = np.where(use_self[..., None], d_self, d)
        n = np.where(two_dir, n_half, taps).astype(np.float64)
        jj = np.where(two_dir, i >> 1, i).astype(np.float64)
        t = (jj + 0.5 + jitter) / n - 0.5
        off = dirv * t[..., None]
        sp = pc + off
        spi = np.trunc(sp + 0.5).astype(np.int64)
        spi[..., 0] = np.clip(spi[..., 0], 0, deliver_n[0] - 1)
        spi[..., 1] = np.clip(spi[..., 1], 0, deliver_n[1] - 1)
        dist = np.linalg.norm(off, axis=-1)

        dq = mb_velocity_texel(spi.astype(np.float64), P.delivered, P.mv_rect)
        ds = depth[dq[..., 1], dq[..., 0]]
        vs = vel[dq[..., 1], dq[..., 0]] * P.out_scale
        sample_streak = np.minimum(np.linalg.norm(vs, axis=-1) * shutter, float(K))

        f = mb_depth_weight(ds, d_centre, soft_z)
        b = mb_depth_weight(d_centre, ds, soft_z)
        cs = mb_cone(dist, sample_streak)
        cc = mb_cone(dist, self_streak)
        cy = 2.0 * mb_cylinder(dist, sample_streak) * mb_cylinder(dist, self_streak)
        a = f * cs + b * cc + cy

        # ── C1: A TAP ON MY OWN SURFACE IS NOT THREE KINDS OF EVIDENCE ──────────────────
        # At equal depth f == b == 1 and both cylinders are ~1 inside the streak, so a tap on
        # the centre's OWN surface scores up to 1 + 1 + 2 = 4 while a genuinely foreground tap
        # scores at most 1. The receiver outvotes the smear 4:1 PER TAP, before the tap COUNT
        # is even considered. The `b` term ("I am blurry and swept over something behind me")
        # and the cylinder term ("two different blurry things mix") are both statements about
        # TWO surfaces; neither is true of one surface sampled twice. Keep f*cone, which is
        # the self-blur that genuinely exists.
        if P.fix in ('c1', 'c6', 'c7', 'c8', 'c9'):
            same = np.abs(ds / np.maximum(d_centre, 1e-12) - 1.0) <= 0.01
            a = np.where(same, f * cs, a)

        if P.arclen:
            a = a * (np.linalg.norm(dirv, axis=-1) / np.maximum(n, 1.0))
        a = np.where(active, a, 0.0)

        # ── C5: A TAP IS A SLICE OF THE EXPOSURE, NOT A VOTE ────────────────────────────
        # Each tap stands for 1/n of the shutter and can cover the centre for at most all of
        # that slice, so its weight is bounded by 1/n and the total by 1. The centre then takes
        # whatever the taps did NOT cover, instead of a fixed 1 that the taps can outvote
        # arbitrarily. That is what makes the answer independent of the tap count.
        if P.fix in ('c5', 'c6', 'c7', 'c8', 'c9'):
            a = np.minimum(a * P.gain, 1.0) / np.maximum(n, 1.0)

        rel = ds / np.maximum(d_centre, 1e-12) - 1.0
        w_near += np.where(rel > 0.01, a, 0.0)
        w_same += np.where(np.abs(rel) <= 0.01, a, 0.0)
        w_far += np.where(rel < -0.01, a, 0.0)

        if P.fix == 'c8':
            isN = (rel > 0.01)
            accN += colour[spi[..., 1], spi[..., 0]] * np.where(isN, a, 0.0)[..., None]
            wN += np.where(isN, a, 0.0)
            accR += colour[spi[..., 1], spi[..., 0]] * np.where(isN, 0.0, a)[..., None]
            wR += np.where(isN, 0.0, a)

        if P.fix == 'c9':
            isB = (rel < -0.01)
            if P.prox_mode == 'idw':
                pw = 1.0 / np.maximum(dist, 1.0) ** P.prox_p
            else:
                pw = mb_cone(dist, self_streak * P.prox)
            pw = pw * np.where(isB, 1.0, 0.0)
            accB += colour[spi[..., 1], spi[..., 0]] * (a * pw)[..., None]
            wB += a * pw
            covB += np.where(isB, a, 0.0)
            a = np.where(isB, 0.0, a)

        acc += colour[spi[..., 1], spi[..., 0]] * a[..., None]
        wsum += a

    if P.fix == 'c8':
        # ── TWO SURFACES CANNOT BOTH OWN THE SAME SLICE OF THE SHUTTER ───────────────────
        # Everything above averages every accepted tap together, so a receiver that claims
        # coverage and a foreground that claims the SAME coverage simply dilute each other in
        # proportion to how much weight each piled up. But the foreground is in FRONT: while it
        # covers this pixel the receiver is HIDDEN, and cannot be claiming anything. So resolve
        # the two by COMPOSITING instead of averaging -- the near bucket's accumulated coverage
        # is an alpha, and everything else shows through 1 - alpha.
        cN = accN / np.maximum(wN, 1e-12)[..., None]
        cR = accR / np.maximum(wR, 1e-12)[..., None]
        alpha = np.minimum(wN * P.alpha_scale, 1.0)[..., None]
        out = cN * alpha + cR * (1.0 - alpha)
        changed = wsum > 1.0001
        stats = dict(hit=float(blurred.mean()), changed=float(changed.mean()),
                     max_len=float(ln[blurred].max()) if blurred.any() else 0.0,
                     avg_taps=float(taps[blurred].mean()) if blurred.any() else 0.0)
        r = dict(blurred=blurred, changed=changed, v_tile=v_tile, ln=ln,
                 near=w_near, same=w_same, far=w_far, wsum=wsum)
        return out, stats, r

    if P.fix in ('c5', 'c6', 'c7', 'c9'):
        # acc/wsum currently carry the centre at weight 1; back it out and re-add it at the
        # weight the taps left unclaimed.
        wt = wsum - 1.0 + covB
        w_centre = np.maximum(0.0, 1.0 - wt)
        acc = (acc - colour) + colour * w_centre[..., None]
        wsum = wt + w_centre
    if P.fix == 'c9':
        # The background comes in with the coverage it earned, but wearing the colour the
        # NEAREST unoccluded sample of it had, rather than the mean of the whole streak.
        bg = accB / np.maximum(wB, 1e-12)[..., None]
        acc = acc + bg * covB[..., None]
    out = acc / np.maximum(wsum, 1e-12)[..., None]
    changed = (wsum > 1.0001) if not P.arclen else (wsum > w0 * 1.0001)
    aux_comp = dict(near=w_near, same=w_same, far=w_far, wsum=wsum)
    stats = dict(hit=float(blurred.mean()), changed=float(changed.mean()),
                 max_len=float(ln[blurred].max()) if blurred.any() else 0.0,
                 avg_taps=float(taps[blurred].mean()) if blurred.any() else 0.0)
    r = dict(blurred=blurred, changed=changed, v_tile=v_tile, ln=ln)
    r.update(aux_comp)
    return out, stats, r


def run_filter(layers, P, quantise_fp16=True):
    colour, depth, vel = rasterize(layers, P.W, P.H, 0.0)
    if quantise_fp16:
        vel = vel.astype(np.float16).astype(np.float64)   # gMbVelocity is RG16F
    tiles = neighbour_max(tile_max(vel, P))
    out, stats, aux = gather(colour, depth, vel, tiles, P)
    aux['source'] = colour
    aux['depth'] = depth
    aux['vel'] = vel
    return out, stats, aux


# ---------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------

def png_write(path, img):
    a = np.clip(img, 0.0, 1.0) ** (1.0 / 2.2)
    b = (a * 255.0 + 0.5).astype(np.uint8)
    h, w, _ = b.shape
    raw = b.tobytes()
    rows = b''.join(b'\x00' + raw[y * w * 3:(y + 1) * w * 3] for y in range(h))

    def chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data +
                struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))

    hdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    with open(path, 'wb') as fh:
        fh.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', hdr) +
                 chunk(b'IDAT', zlib.compress(rows, 6)) + chunk(b'IEND', b''))


# ---------------------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------------------

W_DEF, H_DEF = 768, 512

BG = (0.040, 0.045, 0.055)
BLADE = (0.780, 0.800, 0.860)       # a bright steel blade
SKIRT = (0.180, 0.060, 0.240)       # the dark purple skirt of the report
WALL = (0.300, 0.280, 0.250)


def scene_static():
    return [Layer('bg', (-4000, -4000, 4000, 4000), 2000.0, BG),
            Layer('box', (300, 200, 460, 312), 200.0, WALL)]


def scene_lone():
    return [Layer('bg', (-4000, -4000, 4000, 4000), 2000.0, BG),
            Layer('blade', (300, 200, 460, 312), 200.0, BLADE, motion=(50.0, 0.0))]


def scene_blade_front(skirt_motion=(0.0, 0.0)):
    """The reported case: a FAST blade in FRONT, a slow/static skirt BEHIND it.
    Ground truth says the blade's smear must cross the skirt, partially transparent."""
    return [Layer('bg', (-4000, -4000, 4000, 4000), 2000.0, BG),
            Layer('skirt', (470, 120, 560, 400), 300.0, SKIRT, motion=skirt_motion),
            Layer('blade', (300, 200, 460, 312), 100.0, BLADE, motion=(60.0, 0.0))]


def scene_blade_behind():
    """The control: the blade is BEHIND the skirt.  Ground truth says the skirt occludes it
    and the smear IS cut at the silhouette -- correct behaviour, not a defect."""
    return [Layer('bg', (-4000, -4000, 4000, 4000), 2000.0, BG),
            Layer('skirt', (470, 120, 560, 400), 100.0, SKIRT),
            Layer('blade', (300, 200, 460, 312), 300.0, BLADE, motion=(60.0, 0.0))]


def scene_slow():
    return [Layer('bg', (-4000, -4000, 4000, 4000), 2000.0, BG),
            Layer('body', (300, 180, 500, 340), 200.0, BLADE, motion=(6.0, 0.0))]


def scene_thin(W, H, width=16.0, speed=60.0, z=100.0, flat=False):
    """A THIN FAST mover over a TEXTURED STATIC background.

    Every blade in the scenes above is 160 px across, which is exactly what HIDES this case:
    most taps along a wide body's own streak land back ON the body, so where the background
    is fetched from hardly matters.  At 16 px almost every tap lands on BACKGROUND instead,
    and then the fetch position is the whole answer.

    The background is TEXTURED and STATIC on purpose.  Ground truth keeps it SHARP through
    the mover -- a thin fast object really is nearly transparent, and what shows through is
    the background AT THIS PIXEL, which never moved.  A gather can only fetch background from
    OFFSETS along the streak, so it shows a BLURRED background instead.  Same amount of
    background, wrong provenance: transparency reads as frosted glass."""
    tex = None if flat else texture(W, H, scale=11.0)
    cx = 0.5 * W
    return [Layer('bg', (-4000, -4000, 4000, 4000), 2000.0, (0.5, 0.5, 0.5), tex=tex),
            Layer('blade', (cx - 0.5 * width, 96, cx + 0.5 * width, H - 96), z, BLADE,
                  motion=(speed, 0.0))]


def scene_thin_coverage(W, H, width=16.0, speed=60.0, z=100.0):
    """The same geometry with the mover WHITE on a BLACK ground, so ground_truth returns the
    mover's exposure COVERAGE per pixel directly -- the fraction of the shutter it is really
    there for.  That is the number the filter's own blade share has to match."""
    cx = 0.5 * W
    return [Layer('bg', (-4000, -4000, 4000, 4000), 2000.0, (0.0, 0.0, 0.0)),
            Layer('blade', (cx - 0.5 * width, 96, cx + 0.5 * width, H - 96), z,
                  (1.0, 1.0, 1.0), motion=(speed, 0.0))]


def scene_foliage(W, H, speed=52.0, ang=28.0, nbar=26, seed=11):
    """A THICK fast mover across HIGH-CONTRAST STATIC CLUTTER AT MANY DEPTHS.

    From play, with a picture: a first-person arm sweeping past a tree, and the tree's foliage
    combed into regular stripes along the arm's direction at both edges of the sweep. Every other
    scene in this file puts ONE flat background plane behind the mover, which cannot produce that
    for two reasons -- a flat plane has no high-frequency detail to comb, and one depth means the
    same-surface test never has to decide between two different static things.

    Thin dark bars on bright sky, at depths spread over 600..1400, is alpha-tested foliage in the
    only respects that matter here: maximum local contrast, structure at the pixel scale, and a
    depth that changes discontinuously between neighbouring pixels."""
    rng = np.random.default_rng(seed)
    L = [Layer('sky', (-4000, -4000, 4000, 4000), 4000.0, (0.62, 0.74, 0.92))]
    for i in range(nbar):
        z = 600.0 + 800.0 * rng.random()
        x0 = rng.uniform(-40.0, W - 20.0)
        w = rng.uniform(3.0, 11.0)
        y0 = rng.uniform(-40.0, H - 60.0)
        h = rng.uniform(60.0, H * 0.9)
        g = rng.uniform(0.05, 0.22)
        L.append(Layer('leaf%d' % i, (x0, y0, x0 + w, y0 + h), z, (g * 0.9, g, g * 0.55)))
    a = math.radians(ang)
    L.append(Layer('arm', (0.30 * W, 0.18 * H, 0.30 * W + 150.0, 0.18 * H + 420.0), 80.0,
                   (0.30, 0.16, 0.11),
                   motion=(speed * math.cos(a), speed * math.sin(a))))
    return L


def noisy_bg(W, H, seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.10, 0.55, size=(H, W, 3))


def texture(W, H, scale=17.0):
    """A SMOOTH texture, deliberately not noise.  Broadband noise has enormous edge energy of
    its own and would bury the thing being measured; a smooth texture makes a tile seam show
    up as what it is -- a DISCONTINUITY where there should be none."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    a = 0.55 + 0.30 * np.sin(xs / scale) * np.sin(ys / (scale * 1.37))
    b = 0.55 + 0.30 * np.sin((xs + 0.7 * ys) / (scale * 0.81))
    return np.stack([a, 0.5 * (a + b), b], axis=-1)


def scene_swing(W, H, nband=12, vmax=62.0, z=200.0, x0=232, x1=568, y0=104, y1=416):
    """A body whose velocity VARIES ACROSS IT -- a swinging limb, in piecewise bands.

    ⚠ A RIGID TRANSLATING RECT CANNOT SHOW A TILE SEAM AT ALL.  Every tile that touches it
    reports the same vector, so the dilation has nothing to quantise and the K grid cannot
    appear however wrong the filter is.  The seam needs a velocity GRADIENT inside the
    dilation reach -- which is why the report came from BODIES (limbs at different speeds and
    angles) and never from a wall."""
    tex = texture(W, H)
    L = [Layer('bg', (-4000, -4000, 4000, 4000), 2000.0, (0.02, 0.02, 0.03))]
    h = (y1 - y0) / float(nband)
    for i in range(nband):
        frac = (i + 0.5) / float(nband)
        spd = vmax * (0.15 + 0.85 * frac)
        ang = math.radians(-42.0 + 84.0 * frac)
        L.append(Layer('band%d' % i, (x0, y0 + i * h, x1, y0 + (i + 1) * h), z,
                       (0.8, 0.8, 0.8), tex=tex,
                       motion=(spd * math.cos(ang), spd * math.sin(ang))))
    return L


def scene_twodir(W, H):
    """Two bodies sharing a vertical boundary, moving at right angles to each other."""
    tex = texture(W, H)
    return [Layer('bg', (-4000, -4000, 4000, 4000), 2000.0, (0.02, 0.02, 0.03)),
            Layer('left', (180, 140, 384, 380), 200.0, (0.8, 0.8, 0.8), tex=tex, motion=(46.0, 0.0)),
            Layer('right', (384, 140, 588, 380), 200.0, (0.8, 0.8, 0.8), tex=tex, motion=(0.0, -46.0))]


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------

def delivery(filt, gt, src, mask):
    """How much of the change ground truth demands did the filter actually deliver?

    Projection of (filter - source) onto (gt - source) over `mask`, summed over the region
    so bright pixels are not drowned by dim ones.  1.0 = exactly right, 0.0 = the filter
    left the pixel alone where GT says it must change, >1 = over-delivered."""
    if not mask.any():
        return float('nan'), 0
    a = (filt - src)[mask]
    b = (gt - src)[mask]
    den = float((b * b).sum())
    if den <= 1e-12:
        return float('nan'), int(mask.sum())
    return float((a * b).sum() / den), int(mask.sum())


def rmse(a, b, mask=None):
    d = (a - b) ** 2
    if mask is not None:
        d = d[mask]
    return float(math.sqrt(d.mean())) if d.size else float('nan')


def phase_energy(img, K, y0, y1, x0, x1, axis=1):
    """Edge energy binned by phase against period K -- the metric that identified the K=96
    grid in squaretiles.png.  Returns (ratio of phase 0 to the mean of the rest, z score).
    axis=1 bins COLUMNS (a vertical seam line), axis=0 bins ROWS."""
    band = img[y0:y1, x0:x1].mean(axis=2)
    if axis == 1:
        ex = np.abs(np.diff(band, axis=1)).mean(axis=0)     # per-column edge energy
        cols = np.arange(x0, x0 + ex.size)
    else:
        ex = np.abs(np.diff(band, axis=0)).mean(axis=1)     # per-row edge energy
        cols = np.arange(y0, y0 + ex.size)
    ph = cols % K
    means = np.array([ex[ph == p].mean() if (ph == p).any() else np.nan for p in range(K)])
    p0 = means[0]
    rest = means[1:]
    mu, sd = np.nanmean(rest), np.nanstd(rest)
    return float(p0 / mu), float((p0 - mu) / sd if sd > 0 else 0.0)


def edge_width(profile, lo=0.10, hi=0.90):
    """10-90% rise distance of a monotone-ish transect, in pixels."""
    p = np.asarray(profile, dtype=np.float64)
    a, b = p.min(), p.max()
    if b - a < 1e-9:
        return float('nan')
    n = (p - a) / (b - a)
    idx_lo = np.argmax(n >= lo)
    idx_hi = np.argmax(n >= hi)
    return float(abs(idx_hi - idx_lo))


def blue_noise(H, W, seed=3, iters=12):
    """Uniform values in [0,1) whose spectrum is EMPTY at low frequency -- alternating
    projection between "spectrum is high-pass" and "histogram is uniform".  Same variance as
    white noise, so it does not make the sampling error smaller; it moves that error to
    frequencies a viewer (and any later filter) does not resolve."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(H, W))
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    hp = r / r.max()
    for _ in range(iters):
        v = np.real(np.fft.ifft2(np.fft.fft2(v) * hp))
        order = np.argsort(v, axis=None)
        ranks = np.empty(v.size, dtype=np.float64)
        ranks[order] = np.arange(v.size, dtype=np.float64)
        v = (ranks.reshape(H, W) + 0.5) / float(v.size) - 0.5
    return v + 0.5


def boxblur(img, times=2):
    a = img.copy()
    for _ in range(times):
        p = np.pad(a, ((1, 1), (1, 1), (0, 0)) if a.ndim == 3 else 1, mode='edge')
        a = sum(p[dy:dy + img.shape[0], dx:dx + img.shape[1]]
                for dy in range(3) for dx in range(3)) / 9.0
    return a


def shown_background(out, cov, blade_rgb, mask, cov_max=0.5):
    """Back out the BACKGROUND the filter is showing THROUGH the mover.

    out = cov*blade + (1-cov)*bg is exactly ground truth's own decomposition, and cov is
    measured exactly (scene_thin_coverage), so this isolates the one thing the amount-of-blur
    metrics cannot see: WHICH background the filter put there.  Ground truth scores 0 by
    construction.  Restricted to cov < cov_max so the division stays conditioned."""
    m = mask & (cov < cov_max)
    if not m.any():
        return None, None, m
    k = np.maximum(1.0 - cov[m], 1e-6)[..., None]
    return (out[m] - cov[m][..., None] * np.array(blade_rgb)) / k, None, m


def highpass(img):
    """img minus its own 3x3 box mean -- the band a 1-px dither pattern lives in."""
    a = img.mean(axis=2) if img.ndim == 3 else img
    p = np.pad(a, 1, mode='edge')
    box = sum(p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
              for dy in range(3) for dx in range(3)) / 9.0
    return a - box


def speckle(mask):
    """Pixels whose BINARY gate disagrees with 3 or more of their 4 neighbours.

    A clean boundary scores ~0; a stipple is made entirely of these. This exists because every
    magnitude metric in this file is blind to the artefact it counts: across a tile-jitter sweep
    that takes speckle from 0 to 43239, RMSE against ground truth does not move in the fifth
    decimal. Structure and magnitude are different questions."""
    m = mask.astype(np.int8)
    n = (np.roll(m, 1, 0) + np.roll(m, -1, 0) + np.roll(m, 1, 1) + np.roll(m, -1, 1))
    return int((((m == 1) & (n <= 1)) | ((m == 0) & (n >= 3))).sum())


def error_structure(err, mask, maxshift=10):
    """How much of the error REPEATS, as opposed to being random.

    ⚠ THE REASON THIS EXISTS. Two error images with identical RMSE can look completely
    different: white noise sinks into the picture and a regular comb does not. RMSE, lpRMSE and
    excessHF are all MAGNITUDES and are blind to that distinction by construction -- which is how
    "blue noise is a measured null" was concluded from three metrics, none of which could see the
    one property blue noise changes. mbDither is interleaved-gradient noise, which is DESIGNED to
    be structured (Jimenez 2014 expects a temporal filter to resolve it); asked for a single-frame
    coverage estimate across a high-contrast edge, that structure is the artefact.

    Returns (peak |correlation| of the error with itself at a 1..maxshift pixel offset, the shift
    and axis that peaked). ~0 for noise; large for anything periodic.

    ⚠⚠ AND THIS ONE DOES NOT DISCRIMINATE YET -- it is kept, flagged, because a metric that was
    quietly wrong is worse than one that is loudly unfinished. On scene_foliage it returns
    0.83-0.98 for EVERY arm including the one that is 20x closer to ground truth, because a smooth
    error correlates at lag 1 just as strongly as a comb does. It is measuring SMOOTHNESS. To
    separate a repeating pattern from a smooth one it has to look for a peak AWAY from lag 1 --
    a power spectrum with DC and the low frequencies removed, or an alternating-sign
    autocorrelation. Third metric in this file's history to be blind to the hypothesis it was
    built for; see the note at the head of the file."""
    e = err.mean(axis=2) if err.ndim == 3 else err
    if mask.sum() < 200:
        return 0.0, 0, 0
    best = (0.0, 0, 0)
    for ax in (0, 1):
        for k in range(1, maxshift + 1):
            m2 = mask & np.roll(mask, k, axis=ax)
            if m2.sum() < 200:
                continue
            a = e[m2]
            b = np.roll(e, k, axis=ax)[m2]
            a = a - a.mean()
            b = b - b.mean()
            den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
            c = float((a * b).sum() / den) if den > 1e-20 else 0.0
            if abs(c) > best[0]:
                best = (abs(c), k, ax)
    return best


def hf_rms(img, mask=None):
    h = highpass(img)
    return float(math.sqrt((h[mask] ** 2).mean() if mask is not None else (h ** 2).mean()))


def grad_mag(img, mask=None):
    """Mean |gradient| of an image.

    ⚠ NOT a smear metric on an image that also carries NOISE -- dither raises |gradient| for
    the same reason blur lowers it, and on these scenes the two were the same size, so the
    first version of T12(b) read "sharper than ground truth" for a filter that was in fact
    smearing badly. Use shown_background() for provenance; this stays for edge profiles."""
    a = img.mean(axis=2) if img.ndim == 3 else img
    gx = np.abs(np.diff(a, axis=1, append=a[:, -1:]))
    gy = np.abs(np.diff(a, axis=0, append=a[-1:, :]))
    g = 0.5 * (gx + gy)
    return float(g[mask].mean() if mask is not None else g.mean())


# ---------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------

def hdr(title):
    print('')
    print('=' * 86)
    print(title)
    print('=' * 86)


def dump(outdir, name, **imgs):
    if not outdir:
        return
    for k, v in imgs.items():
        png_write(os.path.join(outdir, '%s_%s.png' % (name, k)), v)


def t_static(P, outdir):
    hdr('T0  STATIC SCENE -- the velocity floor must pass the frame through BIT-IDENTICALLY')
    L = scene_static()
    out, st, aux = run_filter(L, P)
    md = float(np.abs(out - aux['source']).max())
    print('  blurred%%=%.4f  changed%%=%.4f  max|out-src|=%.3e' % (100 * st['hit'], 100 * st['changed'], md))
    print('  %s  (a still frame that softens is the most visible failure this pass has)'
          % ('PASS' if md == 0.0 else 'FAIL'))
    return md == 0.0


def t_lone(P, outdir):
    hdr('T1  LONE MOVER vs GROUND TRUTH -- does the streak have the right length and shape?')
    L = scene_lone()
    out, st, aux = run_filter(L, P)
    gt = ground_truth(L, P.W, P.H, P.shutter)
    src = aux['source']
    row = 256
    thr = 0.02
    gm = np.abs(gt[row] - src[row]).max(axis=1) > thr
    fm = np.abs(out[row] - src[row]).max(axis=1) > thr
    gx = np.where(gm)[0]
    fx = np.where(fm)[0]
    print('  streak predicted = |v|*shutter = %.1f px  (K = %d)' % (50.0 * P.shutter, P.K))
    print('  ground truth  extent x=[%d..%d]  width %d' % (gx.min(), gx.max(), np.ptp(gx) + 1))
    print('  filter        extent x=[%d..%d]  width %d' % (fx.min(), fx.max(), np.ptp(fx) + 1))
    print('  RMSE(filter, gt) whole frame = %.4f   over the smear = %.4f'
          % (rmse(out, gt), rmse(out, gt, np.abs(gt - src).max(axis=2) > thr)))
    dump(outdir, 'lone', src=src, filt=out, gt=gt, err=np.abs(out - gt) * 6.0)
    return True


def t_bladeskirt(P, outdir, name, layers, note):
    hdr('%s  %s' % (name, note))
    out, st, aux = run_filter(layers, P)
    gt = ground_truth(layers, P.W, P.H, P.shutter)
    src = aux['source']

    # Which pixels is ground truth asking to change, and what is showing there NOW?
    want = np.abs(gt - src).max(axis=2) > 0.02
    skirt_px = np.all(np.abs(src - np.array(SKIRT)) < 1e-9, axis=2)
    bg_px = np.all(np.abs(src - np.array(BG)) < 1e-9, axis=2)

    on_skirt = want & skirt_px
    on_bg = want & bg_px

    d_sk, n_sk = delivery(out, gt, src, on_skirt)
    d_bg, n_bg = delivery(out, gt, src, on_bg)

    print('  blurred%%=%.2f changed%%=%.2f maxLen=%.1f avgTaps=%.1f'
          % (100 * st['hit'], 100 * st['changed'], st['max_len'], st['avg_taps']))
    print('  GT wants %d px changed:  %d on the SKIRT, %d on the BACKGROUND' %
          (int(want.sum()), n_sk, n_bg))
    print('  delivery ON THE SKIRT      = %7.3f   (1.0 = exactly right, 0.0 = amputated)  n=%d' % (d_sk, n_sk))
    print('  delivery ON THE BACKGROUND = %7.3f   (the control: same blade, nothing in the way) n=%d' % (d_bg, n_bg))
    print('  RMSE on skirt = %.4f   RMSE on background = %.4f'
          % (rmse(out, gt, on_skirt), rmse(out, gt, on_bg)))

    # A transect straight through the overlap, printed so the shape is visible.
    ys = np.where(on_skirt.any(axis=1))[0]
    if ys.size:
        row = int(np.median(ys))
        xs = np.where(on_skirt[row] | on_bg[row])[0]
        if xs.size:
            lo, hi = max(xs.min() - 6, 0), min(xs.max() + 7, P.W)
            print('  transect row y=%d, x=%d..%d  (luma)' % (row, lo, hi - 1))
            step = max(1, (hi - lo) // 26)
            cols = list(range(lo, hi, step))
            print('    x    : ' + ' '.join('%5d' % x for x in cols))
            for tag, im in (('src', src), ('gt ', gt), ('filt', out)):
                print('    %-5s: ' % tag + ' '.join('%5.3f' % im[row, x].mean() for x in cols))
    dump(outdir, name.lower(), src=src, filt=out, gt=gt, err=np.abs(out - gt) * 6.0)
    return d_sk, d_bg


def t_softz_sweep(P, outdir):
    hdr('T4  softZ SWEEP ON THE BLADE/SKIRT CASE -- localised, which the frame mean could not be')
    layers = scene_blade_front()
    gt = ground_truth(layers, P.W, P.H, P.shutter)
    src, _, _ = rasterize(layers, P.W, P.H, 0.0)
    want = np.abs(gt - src).max(axis=2) > 0.02
    skirt_px = np.all(np.abs(src - np.array(SKIRT)) < 1e-9, axis=2)
    bg_px = np.all(np.abs(src - np.array(BG)) < 1e-9, axis=2)
    on_skirt, on_bg = want & skirt_px, want & bg_px
    print('  softZ   delivery(skirt)  delivery(bg)   RMSE(skirt)  changed%')
    for sz in (0.02, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00):
        P.soft_z = sz
        out, st, _ = run_filter(layers, P)
        d_sk, _ = delivery(out, gt, src, on_skirt)
        d_bg, _ = delivery(out, gt, src, on_bg)
        print('  %5.2f   %13.3f  %12.3f   %10.4f  %7.3f'
              % (sz, d_sk, d_bg, rmse(out, gt, on_skirt), 100 * st['changed']))
    P.soft_z = 0.10


def t_slow(P, outdir):
    hdr('T5  SLOW MOVER -- does an object with a SHORT streak soften its own edge at all?')
    L = scene_slow()
    out, st, aux = run_filter(L, P)
    gt = ground_truth(L, P.W, P.H, P.shutter)
    src = aux['source']
    row = 256
    x0, x1 = 480, 530                      # across the leading edge at x = 500
    print('  streak = |v|*shutter = %.1f px' % (6.0 * P.shutter))
    print('  10-90%% edge width:  source %.1f px   ground truth %.1f px   filter %.1f px'
          % (edge_width(src[row, x0:x1].mean(axis=1)[::-1]),
             edge_width(gt[row, x0:x1].mean(axis=1)[::-1]),
             edge_width(out[row, x0:x1].mean(axis=1)[::-1])))
    print('  RMSE(filter, gt) over the softened band = %.4f'
          % rmse(out, gt, np.abs(gt - src).max(axis=2) > 0.02))
    dump(outdir, 'slow', src=src, filt=out, gt=gt, err=np.abs(out - gt) * 6.0)


def t_tapcount(P, outdir):
    hdr('T6  TAP-COUNT INVARIANCE -- a reconstruction filter must not depend on HOW FINELY it samples')
    L = scene_lone()
    gt = ground_truth(L, P.W, P.H, P.shutter)
    src, _, _ = rasterize(L, P.W, P.H, 0.0)
    want = np.abs(gt - src).max(axis=2) > 0.02
    keep = P.max_taps
    print('  maxTaps   avgTaps   delivery vs GT   RMSE(smear)   (1.000 = the correct amount of blur)')
    for mt in (6, 8, 12, 16, 24, 32, 48, 64):
        P.max_taps = mt
        out, st, _ = run_filter(L, P)
        d, _ = delivery(out, gt, src, want)
        print('  %7d   %7.1f   %14.3f   %11.4f' % (mt, st['avg_taps'], d, rmse(out, gt, want)))
    P.max_taps = keep


def t_skirtspeed(P, outdir):
    hdr('T6b  HOW FAST MUST THE RECEIVER BE TO BLOCK THE SMEAR?  ("maybe skirt is too slow")')
    print('  A blade in FRONT, smearing across a skirt BEHIND it.  Ground truth does not care')
    print('  how fast the skirt is -- the blade is in front, so its smear must cross regardless.')
    print('  twoDir arms at lenSelf >= 1.0 px, i.e. skirt speed >= %.3f px/frame.' % (1.0 / P.shutter))
    print('')
    print('  skirt speed   streak    delivery(skirt)      delivery(skirt)')
    print('   px/frame      px       twoDir=ON (ship)     twoDir=OFF')
    keep = P.two_dir
    for spd in (0.0, 0.3, 0.6, 1.2, 3.0, 6.0, 12.0, 24.0):
        layers = scene_blade_front(skirt_motion=(0.0, spd))
        gt = ground_truth(layers, P.W, P.H, P.shutter, nsub=257)
        s0, _, _ = rasterize(layers, P.W, P.H, 0.0)
        want = np.abs(gt - s0).max(axis=2) > 0.02
        skirt_px = np.all(np.abs(s0 - np.array(SKIRT)) < 1e-9, axis=2)
        on_skirt = want & skirt_px
        row = []
        for td in (True, False):
            P.two_dir = td
            out, _, _ = run_filter(layers, P)
            d, _ = delivery(out, gt, s0, on_skirt)
            row.append(d)
        print('  %9.1f   %7.1f   %16.3f   %18.3f' % (spd, spd * P.shutter, row[0], row[1]))
    P.two_dir = keep


def t_arclen(P, outdir):
    hdr('T9  CANDIDATE FIX -- weight each tap by the ARC LENGTH it stands for (NOT in any shader)')
    print('  Multiply every tap by |dir|/n -- the arc length of streak it stands for -- and give')
    print('  the centre one tap-length.  EXPERIMENT in this rig only; no shader has this.')
    print('')
    print('  MEASURED RESULT, and (a) is the more useful half BECAUSE it is null:')
    print('   (a) is a NO-OP BY CONSTRUCTION and that refutes the density story for T6.  On one')
    print('       axis |dir|/n is the same constant for every tap AND for the centre, so it')
    print('       cancels out of the ratio exactly.  T6 is therefore NOT a sampling-density bug:')
    print('       mover weight is sum(cone) over the taps that land on it, which grows with the')
    print('       tap count, while THE BACKGROUND IS REPRESENTED EXACTLY ONCE however long the')
    print('       streak is -- background taps score 0 (cone(dist, 0) = 0 both ways round).')
    print('   (b) DOES move, because twoDir is the one place two different |dir|/n values exist')
    print('       (5/16 px vs 96/16 px, a 19x ratio), so the scale no longer cancels.')
    print('')
    keep_a, keep_t = P.arclen, P.max_taps

    L = scene_lone()
    gt = ground_truth(L, P.W, P.H, P.shutter)
    s0, _, _ = rasterize(L, P.W, P.H, 0.0)
    want = np.abs(gt - s0).max(axis=2) > 0.02
    print('  (a) TAP-COUNT INVARIANCE, lone mover.   delivery vs GT, 1.000 = correct')
    print('      maxTaps      shipped     arc-length')
    for mt in (6, 12, 24, 32, 48, 64):
        P.max_taps = mt
        row = []
        for al in (False, True):
            P.arclen = al
            out, _, _ = run_filter(L, P)
            row.append(delivery(out, gt, s0, want)[0])
        print('      %7d   %10.3f   %12.3f' % (mt, row[0], row[1]))
    P.max_taps = keep_t

    print('')
    print('  (b) THE AMPUTATION, blade in front of a skirt behind.   delivery on the skirt')
    print('      skirt px/frame      shipped     arc-length')
    for spd in (0.0, 0.3, 0.6, 3.0, 12.0):
        layers = scene_blade_front(skirt_motion=(0.0, spd))
        g = ground_truth(layers, P.W, P.H, P.shutter, nsub=257)
        s1, _, _ = rasterize(layers, P.W, P.H, 0.0)
        w = (np.abs(g - s1).max(axis=2) > 0.02) & np.all(np.abs(s1 - np.array(SKIRT)) < 1e-9, axis=2)
        row = []
        for al in (False, True):
            P.arclen = al
            out, _, _ = run_filter(layers, P)
            row.append(delivery(out, g, s1, w)[0])
        print('      %14.1f   %10.3f   %12.3f' % (spd, row[0], row[1]))
    P.arclen = keep_a


def t_composition(P, outdir):
    hdr('T10  WHO WON THE PIXEL -- the same split the in-game mode 3 draws, as numbers')
    print('  Reported from play: the sword bleeds STRONG MAGENTA over the ground (red = weight')
    print('  accepted from something NEARER, blue = the weight the pixel kept of itself) and')
    print('  goes GREEN over the skirt.  Green is weight from taps at or behind our own depth --')
    print('  which the shader cannot yet separate from taps on OUR OWN SURFACE.  Here it can.')
    print('')
    print('  receiver     twoDir   |  near%%  same%%   far%%  centre%%  | delivery')
    for spd in (0.0, 3.0, 8.0, 16.0, 32.0):
        layers = scene_blade_front(skirt_motion=(0.0, spd))
        gt = ground_truth(layers, P.W, P.H, P.shutter, nsub=257)
        s0, _, _ = rasterize(layers, P.W, P.H, 0.0)
        want = np.abs(gt - s0).max(axis=2) > 0.02
        skirt_px = np.all(np.abs(s0 - np.array(SKIRT)) < 1e-9, axis=2)
        bg_px = np.all(np.abs(s0 - np.array(BG)) < 1e-9, axis=2)
        keep = P.two_dir
        for td in (True, False):
            P.two_dir = td
            out, _, aux = run_filter(layers, P)
            for label, m in (('skirt', want & skirt_px), ('ground', want & bg_px)):
                if not m.any():
                    continue
                tot = aux['wsum'][m]
                n_, s_, f_ = aux['near'][m] / tot, aux['same'][m] / tot, aux['far'][m] / tot
                d, _ = delivery(out, gt, s0, m)
                print('  %-6s %4.0f  %-6s  | %5.1f  %5.1f  %5.1f   %5.1f    | %6.3f'
                      % (label, spd, td, 100 * n_.mean(), 100 * s_.mean(), 100 * f_.mean(),
                         100 * (1.0 / tot).mean(), d))
        P.two_dir = keep
        print('')


def t_fix(P, outdir):
    hdr('T11  THE FIX -- every candidate scored against GROUND TRUTH on the whole matrix')
    print('  ship = what is deployed.  c1 = a tap on my own surface scores f*cone only, not')
    print('  f*cone + b*cone + 2*cyl*cyl.  c5 = each tap is a 1/n slice of the shutter capped at')
    print('  1, and the centre takes what the taps did not claim.  c6 = both.')
    print('  1.000 = exactly the blur ground truth demands.')
    print('')
    arms = ('ship', 'c1', 'c5', 'c6')
    keep = P.fix

    print('  (a) STATIC WORLD -- must be BIT-IDENTICAL or the candidate is not shippable')
    print('      The SHIPPED configuration is the last two rows: c7 at gain 2 is MB-2j, and c9')
    print('      adds MB-2k. A still frame that softens is the most visible failure this pass has,')
    print('      so every arm that has ever been deployed stays in this guard.')
    L0 = scene_static()
    keep_gain, keep_pm, keep_pp = P.gain, P.prox_mode, P.prox_p
    P.prox_mode, P.prox_p = 'idw', 4.0
    for fx, gn in [(a, 1.0) for a in arms] + [('c7', 2.0), ('c9', 2.0)]:
        P.fix, P.gain = fx, gn
        out, _, aux = run_filter(L0, P)
        d = float(np.abs(out - aux['source']).max())
        print('      %-5s gain %.1f  max|out-src| = %.3e   %s'
              % (fx, gn, d, 'PASS' if d == 0.0 else 'FAIL'))
    P.gain, P.prox_mode, P.prox_p = keep_gain, keep_pm, keep_pp

    print('')
    print('  (b) THE SMEAR ITSELF, lone mover over static ground')
    L1 = scene_lone()
    gt1 = ground_truth(L1, P.W, P.H, P.shutter)
    s1, _, _ = rasterize(L1, P.W, P.H, 0.0)
    w1 = np.abs(gt1 - s1).max(axis=2) > 0.02
    for fx in arms:
        P.fix = fx
        out, _, _ = run_filter(L1, P)
        print('      %-5s delivery = %6.3f   RMSE = %.4f'
              % (fx, delivery(out, gt1, s1, w1)[0], rmse(out, gt1, w1)))

    print('')
    print('  (c) TAP-COUNT INVARIANCE -- delivery must not move with maxTaps')
    kt = P.max_taps
    print('      arm      6 taps  16 taps  32 taps  64 taps   spread')
    for fx in arms:
        P.fix = fx
        row = []
        for mt in (6, 16, 32, 64):
            P.max_taps = mt
            out, _, _ = run_filter(L1, P)
            row.append(delivery(out, gt1, s1, w1)[0])
        P.max_taps = kt
        print('      %-5s %8.3f %8.3f %8.3f %8.3f %8.3f'
              % (fx, row[0], row[1], row[2], row[3], max(row) - min(row)))

    print('')
    print('  (d) THE AMPUTATION -- delivery onto a skirt that a faster blade is crossing')
    print('      receiver px/frame:    0.0     3.0     8.0    16.0    32.0')
    cache = {}
    for spd in (0.0, 3.0, 8.0, 16.0, 32.0):
        layers = scene_blade_front(skirt_motion=(0.0, spd))
        g = ground_truth(layers, P.W, P.H, P.shutter, nsub=257)
        s0, _, _ = rasterize(layers, P.W, P.H, 0.0)
        m = (np.abs(g - s0).max(axis=2) > 0.02) & np.all(np.abs(s0 - np.array(SKIRT)) < 1e-9, axis=2)
        cache[spd] = (layers, g, s0, m)
    for fx in arms:
        P.fix = fx
        row = []
        for spd in (0.0, 3.0, 8.0, 16.0, 32.0):
            layers, g, s0, m = cache[spd]
            out, _, _ = run_filter(layers, P)
            row.append(delivery(out, g, s0, m)[0])
        print('      %-5s          ' % fx + ' '.join('%7.3f' % v for v in row))

    print('')
    print('  (e) THE GREEN IN MODE 3 -- same-surface share over the moving skirt, 8 px/frame')
    layers, g, s0, m = cache[8.0]
    for fx in arms:
        P.fix = fx
        out, _, aux = run_filter(layers, P)
        tot = aux['wsum'][m]
        cen = (tot - aux['near'][m] - aux['same'][m] - aux['far'][m]) / tot
        print('      %-5s near=%5.1f%%  same=%5.1f%%  far=%4.1f%%  centre=%5.1f%%'
              % (fx, 100 * (aux['near'][m] / tot).mean(), 100 * (aux['same'][m] / tot).mean(),
                 100 * (aux['far'][m] / tot).mean(), 100 * cen.mean()))
    P.fix = keep


def thin_case(P, width, speed=60.0, nsub=257, flat=False):
    """Everything a thin-mover arm needs, built once and reused across arms."""
    L = scene_thin(P.W, P.H, width=width, speed=speed, flat=flat)
    src, _, _ = rasterize(L, P.W, P.H, 0.0)
    gt = ground_truth(L, P.W, P.H, P.shutter, nsub=nsub)
    cov = ground_truth(scene_thin_coverage(P.W, P.H, width=width, speed=speed),
                       P.W, P.H, P.shutter, nsub=nsub)[..., 0]
    body = np.all(np.abs(src - np.array(BLADE)) < 1e-9, axis=2)    # the mover's OWN pixels
    changed = np.abs(gt - src).max(axis=2) > 0.02
    return dict(L=L, src=src, gt=gt, cov=cov, body=body, changed=changed)


def t_thin(P, outdir):
    hdr('T12  THIN FAST MOVER -- the reported "refraction" and "dithered" look')
    print('  Reported after MB-2j landed: a thin fast sword can read as a REFRACTION (the')
    print('  background smears where the sword should be), and a small share of pixels look')
    print('  DITHERED.  Neither can appear on the 160 px blades above -- see scene_thin.')
    print('')
    keep = (P.fix, P.gain, P.tap_jitter, P.max_taps)
    arms = (('ship', 'ship', 1.0), ('MB-2j', 'c7', 2.0), ('MB-2k', 'c9', 2.0))
    P.prox_mode, P.prox_p = 'idw', 4.0

    def run(case, fx, gn, tj=None):
        P.fix, P.gain = fx, gn
        if tj is not None:
            P.tap_jitter = tj
        out, _, aux = run_filter(case['L'], P)
        P.tap_jitter = keep[2]
        return out, aux

    widths = (8.0, 16.0, 32.0, 64.0, 160.0)
    cases = {w: thin_case(P, w) for w in widths}

    print('  (a) IS THE SWORD TOO TRANSPARENT?  On the mover\'s OWN pixels, how much of the')
    print('      answer is the mover, against the coverage ground truth says it has.')
    print('      A thin fast object IS mostly background -- cov is the CORRECT blade share.')
    print('      width   cov(GT)      arm     blade share    centre kept   bg taps')
    for w in widths:
        c = cases[w]
        m = c['body']
        covm = float(c['cov'][m].mean())
        for name, fx, gn in arms:
            out, aux = run(c, fx, gn)
            tot = aux['wsum'][m]
            blade = (aux['same'][m] + (tot - aux['near'][m] - aux['same'][m] - aux['far'][m])) / tot
            cen = (tot - aux['near'][m] - aux['same'][m] - aux['far'][m]) / tot
            print('      %5.0f   %7.3f   %-6s      %7.3f       %7.3f   %7.3f'
                  % (w, covm, name, blade.mean(), cen.mean(), (aux['far'][m] / tot).mean()))

    print('')
    print('  (b) WHICH BACKGROUND IS IT SHOWING?  Back out bg from out = cov*blade+(1-cov)*bg')
    print('      and compare to the background REALLY behind that pixel.  A gather can only')
    print('      fetch background from OFFSETS along the streak, and this background never')
    print('      moved, so any error here is pure provenance.  GT scores 0.00 by construction.')
    print('      corr 1.00 / std 1.00 = the right background.  std < 1 = smeared (refraction).')
    print('      width      arm     RMSE(bg)   corr    std ratio')
    for w in widths:
        c = cases[w]
        bg_true, _, _ = rasterize([c['L'][0]], P.W, P.H, 0.0)      # the scene without the mover
        for name, fx, gn in arms:
            out, _ = run(c, fx, gn)
            shown, _, m = shown_background(out, c['cov'], BLADE, c['body'])
            if shown is None:
                print('      %5.0f   %-6s      (no pixel below cov 0.5)' % (w, name)); continue
            truth = bg_true[m]
            a = shown.mean(axis=1) - shown.mean()
            b = truth.mean(axis=1) - truth.mean()
            corr = float((a * b).sum() / max(math.sqrt((a * a).sum() * (b * b).sum()), 1e-12))
            print('      %5.0f   %-6s       %.4f   %6.3f     %6.3f'
                  % (w, name, float(math.sqrt(((shown - truth) ** 2).mean())), corr,
                     float(shown.mean(axis=1).std() / max(truth.mean(axis=1).std(), 1e-12))))

    print('')
    print('  (c) THE DITHER.  dither = RMS of (taps jittered) - (taps at a fixed phase), which')
    print('      is EXACTLY how much of the image the per-pixel hash decides.  excessHF is')
    print('      high-frequency energy the filter has that GROUND TRUTH does not.')
    print('      width      arm     dither    excessHF    RMSE(body)')
    for w in widths:
        c = cases[w]
        m = c['body']
        for name, fx, gn in arms:
            oj, _ = run(c, fx, gn, tj=1.0)
            o0, _ = run(c, fx, gn, tj=0.0)
            d = float(math.sqrt(((oj - o0) ** 2)[m].mean()))
            print('      %5.0f   %-6s   %.5f     %.5f      %.4f'
                  % (w, name, d, hf_rms(oj, m) - hf_rms(c['gt'], m), rmse(oj, c['gt'], m)))

    print('')
    print('  (d) THE GAIN.  The cap is min(a*gain, 1).  At gain 2 every tap inside the streak')
    print('      SATURATES, so the taper that used to grade near background over far is gone')
    print('      and the centre keeps nothing.  Lowering it trades that back against the')
    print('      over-blur gain 2 was measured to cure -- lone = the wide-open smear.')
    c16 = cases[16.0]
    L1 = scene_lone()
    gt1 = ground_truth(L1, P.W, P.H, P.shutter)
    s1, _, _ = rasterize(L1, P.W, P.H, 0.0)
    w1 = np.abs(gt1 - s1).max(axis=2) > 0.02
    bg16d, _, _ = rasterize([c16['L'][0]], P.W, P.H, 0.0)
    print('      gain   thin: blade  bgStd   dither  RMSE     lone: delivery   RMSE')
    for gn in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4):
        oj, aux = run(c16, 'c7', gn, tj=1.0)
        o0, _ = run(c16, 'c7', gn, tj=0.0)
        m = c16['body']
        tot = aux['wsum'][m]
        blade = ((aux['same'][m] + (tot - aux['near'][m] - aux['same'][m] - aux['far'][m])) / tot).mean()
        d = float(math.sqrt(((oj - o0) ** 2)[m].mean()))
        P.fix, P.gain = 'c7', gn
        ol, _, _ = run_filter(L1, P)
        sh, _, mm = shown_background(oj, c16['cov'], BLADE, c16['body'])
        std = float(sh.mean(axis=1).std() / max(bg16d[mm].mean(axis=1).std(), 1e-12))
        print('      %4.1f      %7.3f  %6.3f  %.5f  %.4f       %7.3f  %.4f'
              % (gn, blade, std, d, rmse(oj, c16['gt'], m),
                 delivery(ol, gt1, s1, w1)[0], rmse(ol, gt1, w1)))
    print('      cov(GT) for the blade share above = %.3f' % float(c16['cov'][c16['body']].mean()))

    print('')
    print('  (e) THE NOISE ITSELF.  The dither in (c) is the sampling error of estimating')
    print('      coverage from 32 jittered taps, and no weighting can remove it -- but mbDither')
    print('      is WHITE, so that error sits at every frequency including the ones a viewer')
    print('      resolves.  Blue noise has the same variance and almost none of it low.')
    print('      lpRMSE is the error left after a small blur -- the part that is actually seen.')
    print('      Run at the SHIPPED arm: under MB-2j the hash decided 4% of the error and blue')
    print('      noise could not have registered whatever its merits. MB-2k raises that share,')
    print('      so the question is asked again where it can actually be answered.')
    print('      width    noise    dither    RMSE     lpRMSE    excessHF')
    for w in widths:
        c = cases[w]
        m = c['body']
        gl = boxblur(c['gt'])
        for nz in ('white', 'blue'):
            P.tap_noise = nz
            oj, _ = run(c, 'c9', 2.0, tj=1.0)
            o0, _ = run(c, 'c9', 2.0, tj=0.0)
            print('      %5.0f   %-6s   %.5f  %.4f   %.5f    %.5f'
                  % (w, nz, float(math.sqrt(((oj - o0) ** 2)[m].mean())), rmse(oj, c['gt'], m),
                     rmse(boxblur(oj), gl, m), hf_rms(oj, m) - hf_rms(c['gt'], m)))
        P.tap_noise = 'white'

    print('')
    print('  (f) MB-2k: THE EXPONENT.  1/dist**p weights the revealed-background MIXTURE; the')
    print('      AMOUNT of background is untouched.  p is an optimum, not a trend -- past it the')
    print('      estimate collapses onto the single nearest tap, whose position the hash picks,')
    print('      and both the contrast (std past 1.0) and the dither run away.  skirt/lone are')
    print('      the guards: MB-2k must not undo the amputation fix or bring the over-blur back.')
    L1 = scene_lone()
    gt1 = ground_truth(L1, P.W, P.H, P.shutter)
    s1, _, _ = rasterize(L1, P.W, P.H, 0.0)
    w1 = np.abs(gt1 - s1).max(axis=2) > 0.02
    Ls = scene_blade_front(skirt_motion=(0.0, 8.0))
    gts = ground_truth(Ls, P.W, P.H, P.shutter, nsub=257)
    ss, _, _ = rasterize(Ls, P.W, P.H, 0.0)
    ms = (np.abs(gts - ss).max(axis=2) > 0.02) & np.all(np.abs(ss - np.array(SKIRT)) < 1e-9, axis=2)
    c16 = cases[16.0]
    bg16, _, _ = rasterize([c16['L'][0]], P.W, P.H, 0.0)
    print('      p       thin16: std   corr   RMSE(bg)  dither |  skirt   lone')
    for pp in (0.0, 2.0, 3.0, 4.0, 6.0, 12.0):
        P.prox_mode, P.prox_p = 'idw', pp
        fx = 'c7' if pp == 0.0 else 'c9'
        oj, _ = run(c16, fx, 2.0, tj=1.0)
        o0, _ = run(c16, fx, 2.0, tj=0.0)
        shown, _, m = shown_background(oj, c16['cov'], BLADE, c16['body'])
        truth = bg16[m]
        aa = shown.mean(axis=1) - shown.mean()
        bb = truth.mean(axis=1) - truth.mean()
        corr = float((aa * bb).sum() / max(math.sqrt((aa * aa).sum() * (bb * bb).sum()), 1e-12))
        P.fix, P.gain = fx, 2.0
        osk, _, _ = run_filter(Ls, P)
        olo, _, _ = run_filter(L1, P)
        print('      %-6s        %5.3f %6.3f    %.4f   %.5f | %6.3f %6.3f'
              % ('MB-2j' if pp == 0.0 else '%.0f' % pp,
                 float(shown.mean(axis=1).std() / max(truth.mean(axis=1).std(), 1e-12)), corr,
                 float(math.sqrt(((shown - truth) ** 2).mean())),
                 float(math.sqrt(((oj - o0) ** 2)[c16['body']].mean())),
                 delivery(osk, gts, ss, ms)[0], delivery(olo, gt1, s1, w1)[0]))
    P.prox_mode, P.prox_p = 'idw', 4.0

    if outdir:
        c = cases[16.0]
        o_ship, _ = run(c, 'ship', 1.0)
        o_j, _ = run(c, 'c7', 2.0)
        dump(outdir, 't12_thin', src=c['src'], gt=c['gt'], ship=o_ship, mb2j=o_j)
    P.fix, P.gain, P.tap_jitter, P.max_taps = keep


def t_foliage(P, outdir):
    hdr('T13  STATIC CLUTTER AT MANY DEPTHS -- can a MOVER smear something that is NOT MOVING?')
    print('  From play, with a picture: an arm sweeping past a tree, the foliage combed into')
    print('  streaks along the arm\'s direction at both edges of the sweep. Every other scene here')
    print('  puts ONE flat plane behind the mover, which can show neither the high-frequency')
    print('  detail that combs nor a same-surface test having to choose between two static things.')
    print('')
    print('  THE STRUCTURAL CLAIM UNDER TEST: a static pixel has selfStreak = 0, and every weight')
    print('  term except `f * cone(dist, sampleStreak)` carries a cone or cylinder of selfStreak.')
    print('  So a static pixel should be able to receive colour ONLY from a mover -- never from')
    print('  other static geometry, at any depth spread or contrast. same% and far% must be 0.')
    L = scene_foliage(P.W, P.H)
    src, _, vel = rasterize(L, P.W, P.H, 0.0)
    gt = ground_truth(L, P.W, P.H, P.shutter, nsub=257)
    arm = np.all(np.abs(src - np.array((0.30, 0.16, 0.11))) < 1e-9, axis=2)
    delta = np.abs(gt - src).max(axis=2)
    frozen = (delta <= 0.002) & ~arm
    vq = np.linalg.norm(vel.astype(np.float16).astype(np.float64), axis=-1) * P.shutter
    print('')
    print('  %d px that ground truth says must not move; their max selfStreak = %.4f px'
          % (int(frozen.sum()), float(vq[frozen].max())))
    print('  arm      max|leak|   px>0.02 | where the weight on those pixels came from')
    keep = (P.fix, P.gain, P.prox_p)
    for name, fx, gn, pp in (('ship', 'ship', 1.0, 1.0), ('MB-2j', 'c7', 2.0, 1.0),
                             ('MB-2k', 'c9', 2.0, 4.0)):
        P.fix, P.gain, P.prox_mode, P.prox_p = fx, gn, 'idw', pp
        out, _, aux = run_filter(L, P)
        err = np.abs(out - src).max(axis=2)
        lk = frozen & (err > 0.02)
        tot = np.maximum(aux['wsum'][lk], 1e-12)
        cen = (tot - aux['near'][lk] - aux['same'][lk] - aux['far'][lk]) / tot
        print('  %-6s   %.4f      %6d | near=%5.1f%% same=%5.1f%% far=%5.1f%% centre=%5.1f%%'
              % (name, float(err[frozen].max()), int(lk.sum()),
                 100 * (aux['near'][lk] / tot).mean(), 100 * (aux['same'][lk] / tot).mean(),
                 100 * (aux['far'][lk] / tot).mean(), 100 * cen.mean()))
    P.fix, P.gain, P.prox_p = keep
    print('')
    print('  MEASURED: same% and far% are EXACTLY 0.0 for every arm, and the only pixels that')
    print('  move are ones a mover genuinely reached (343 of 295725, at the 0.002 boundary of the')
    print('  frozen mask itself). The claim holds, so THE GATHER CANNOT PRODUCE THE REPORTED')
    print('  PICTURE with a static tree -- the foliage has to be carrying velocity, which is a')
    print('  question for gMbVelocity and mbDebug mode 1, not for this file.')


def t_stipple(P, outdir):
    hdr('T14  THE HALFTONE AT A MOVER\'S SILHOUETTE -- mbTileJitter flips a BINARY gate')
    print('  From play, magnified: the arm\'s edge against bright sky is a band of discrete dots')
    print('  ~30-40 px wide, not a gradient. Binary, so not tap-count quantisation (which grades')
    print('  1/32, 2/32, ...). `blurred` IS binary -- speed >= floor -- and MB-2h\'s jitter')
    print('  REPLACES this pixel\'s tile with one up to K/2 away, so at the edge of a mover\'s')
    print('  dilated neighbourhood it decides per pixel between a full blur and none at all.')
    print('')
    print('  ⚠ AND THIS IS WHY EVERY EARLIER TILE TEST WAS A NULL. The rig ran 768x512 at K=96 --')
    print('  EIGHT BY SIX tiles, against the game\'s 27x17. A mover covering two tiles here covers')
    print('  a quarter of the frame, so the dilated boundary falls OFF SCREEN and the jitter has')
    print('  nothing to straddle. T7 could not reproduce the K grid for the same reason.')
    print('')
    keep = (P.K, P.tiles, P.tile_jitter, P.tile_jitter_mode, P.fix, P.gain)
    print('  K   tiles   mode      jitter | blurred%  speckle   RMSE vs GT')
    for K in (96, 28):
        P.K = K
        P.tiles = (int(math.ceil(P.W / float(K))), int(math.ceil(P.H / float(K))))
        L = [Layer('sky', (-4000, -4000, 4000, 4000), 4000.0, (0.62, 0.74, 0.92)),
             Layer('arm', (300, 150, 390, 380), 80.0, (0.30, 0.16, 0.11), motion=(48.0, 26.0))]
        gt = ground_truth(L, P.W, P.H, P.shutter, nsub=257)
        for mode, tj in (('replace', 0.0), ('replace', 0.5), ('replace', 1.0), ('max', 1.0)):
            P.tile_jitter, P.tile_jitter_mode = tj, mode
            P.fix, P.gain, P.prox_mode, P.prox_p = 'c9', 2.0, 'idw', 4.0
            out, st, aux = run_filter(L, P)
            print('  %-3d %-7s %-9s  %4.2f  |  %6.3f  %7d    %.5f'
                  % (K, '%dx%d' % P.tiles, mode, tj, 100.0 * st['hit'],
                     speckle(aux['blurred']), rmse(out, gt)))
    P.K, P.tiles, P.tile_jitter, P.tile_jitter_mode, P.fix, P.gain = keep
    print('')
    print('  MEASURED: speckle is EXACTLY 0 at jitter 0 and rises linearly with it, while RMSE')
    print('  does not move in the fifth decimal at any setting. `max` -- take the LONGER of (own')
    print('  tile, jittered tile), so a jittered fetch can only EXTEND the search -- halves it and')
    print('  no more, because only half the flips are holes punched INSIDE the region; the other')
    print('  half are pixels switched ON outside it, and those it cannot touch. A stochastic')
    print('  answer to a BINARY question is a stipple however it is clipped, so the fix has to')
    print('  make the dilation smooth and DETERMINISTIC rather than make the coin fairer.')


def t_tilegrid(P, outdir):
    hdr('T7  TILE GRID -- MB-2h jitter, on a body whose velocity VARIES (a rigid rect cannot show it)')
    L = scene_swing(P.W, P.H)
    gt = ground_truth(L, P.W, P.H, P.shutter, nsub=257)
    keep = P.tile_jitter
    print('  tileJitter   phase-0 cols      phase-0 rows      RMSE vs GT   changed%')
    for jit in (0.0, 0.5, 1.0):
        P.tile_jitter = jit
        out, st, aux = run_filter(L, P)
        rc, zc = phase_energy(out, P.K, 120, 400, 260, 540, axis=1)
        rr, zr = phase_energy(out, P.K, 120, 400, 260, 540, axis=0)
        print('  %10.1f   %5.3fx z=%+5.2f   %5.3fx z=%+5.2f   %10.4f   %7.3f'
              % (jit, rc, zc, rr, zr, rmse(out, gt), 100 * st['changed']))
        dump(outdir, 'tilegrid_j%02d' % int(jit * 10), filt=out, gt=gt,
             err=np.abs(out - gt) * 6.0)
    P.tile_jitter = keep


def t_twodir(P, outdir):
    hdr('T8  TWO BODIES AT RIGHT ANGLES -- MB-2h two-direction sampling, judged against GT')
    L = scene_twodir(P.W, P.H)
    gt = ground_truth(L, P.W, P.H, P.shutter, nsub=257)
    keep = P.two_dir
    print('  twoDir   seam edge energy   body interior   ratio    RMSE vs GT')
    for td in (False, True):
        P.two_dir = td
        out, st, aux = run_filter(L, P)
        seam = float(np.abs(np.diff(out[170:350, 376:394].mean(axis=2), axis=1)).mean())
        body = float(np.abs(np.diff(out[170:350, 230:340].mean(axis=2), axis=1)).mean())
        print('  %-6s   %16.5f   %13.5f   %5.2f   %11.4f'
              % (td, seam, body, seam / max(body, 1e-9), rmse(out, gt)))
        dump(outdir, 'twodir_%d' % int(td), filt=out, gt=gt, err=np.abs(out - gt) * 6.0)
    P.two_dir = keep


def explain(P, layers, px, py):
    """Print the tap-by-tap arithmetic for ONE pixel.  The whole-frame statistics can say a
    number moved; only this can say WHY, and which of the three weight terms carried it."""
    hdr('EXPLAIN  pixel (%d, %d)' % (px, py))
    colour, depth, vel = rasterize(layers, P.W, P.H, 0.0)
    vel = vel.astype(np.float16).astype(np.float64)
    tiles = neighbour_max(tile_max(vel, P))
    K, sh = float(P.K), P.shutter

    v_self = vel[py, px]
    r1, r2 = mb_dither(np.array([px + 17.0, py + 17.0])), mb_dither(np.array([px + 37.0, py + 37.0]))
    tj = (np.array([r1, r2]) - 0.5) * (K * P.tile_jitter) if P.tile_jitter > 0 else np.zeros(2)
    tq = np.clip(np.trunc((np.array([px, py]) + tj) / K).astype(int),
                 [0, 0], [P.tiles[0] - 1, P.tiles[1] - 1])
    v_tile = tiles[tq[1], tq[0]]

    self_streak = min(np.linalg.norm(v_self) * sh, K)
    d = v_tile * sh
    ln = np.linalg.norm(d)
    if ln > K:
        d, ln = d * (K / ln), K
    taps = int(np.clip(np.ceil(ln), 3, max(P.max_taps, 3)))
    jitter = mb_dither(np.array([float(px), float(py)])) - 0.5
    d_self = v_self * sh
    len_self = np.linalg.norm(d_self)
    if len_self > K:
        d_self, len_self = d_self * (K / len_self), K
    two_dir = P.two_dir and (len_self >= 1.0) and (taps >= 4)
    n_half = max(taps >> 1, 1)
    d_centre = depth[py, px]

    print('  centre   colour=%s  z=%.1f  vSelf=(%.2f,%.2f) selfStreak=%.2f px'
          % (np.round(colour[py, px], 3), 1.0 / max(d_centre, 1e-9), v_self[0], v_self[1], self_streak))
    print('  tile     vTile=(%.2f,%.2f)  len=%.2f  taps=%d  twoDir=%s  (nHalf=%d)'
          % (v_tile[0], v_tile[1], ln, taps, two_dir, n_half))
    print('')
    print('  tap axis   dist    lands on      sampleStreak      f      b   cone_s   cone_c    cyl*cyl*2        a')
    acc, wsum = colour[py, px].copy(), 1.0
    tally = {}
    for i in range(taps):
        use_self = two_dir and ((i & 1) == 1)
        dirv = d_self if use_self else d
        n = n_half if two_dir else taps
        j = (i >> 1) if two_dir else i
        t = (j + 0.5 + jitter) / n - 0.5
        off = dirv * t
        spi = np.clip(np.trunc(np.array([px, py]) + off + 0.5).astype(int),
                      [0, 0], [P.W - 1, P.H - 1])
        dist = np.linalg.norm(off)
        ds = depth[spi[1], spi[0]]
        vs = vel[spi[1], spi[0]]
        ss = min(np.linalg.norm(vs) * sh, K)
        f = mb_depth_weight(ds, d_centre, P.soft_z)
        b = mb_depth_weight(d_centre, ds, P.soft_z)
        cs, cc = mb_cone(dist, ss), mb_cone(dist, self_streak)
        cy = 2.0 * mb_cylinder(dist, ss) * mb_cylinder(dist, self_streak)
        a = f * cs + b * cc + cy
        c = colour[spi[1], spi[0]]
        who = 'BLADE' if c.mean() > 0.5 else ('skirt' if c.mean() > 0.1 else 'bg')
        tally[who] = tally.get(who, 0.0) + a
        acc = acc + c * a
        wsum += a
        print('  %3d %-5s %6.1f   %-8s z=%7.1f  %8.2f  %5.2f  %5.2f  %7.3f  %7.3f  %11.3f  %7.3f'
              % (i, 'SELF' if use_self else 'tile', dist, who, 1.0 / max(ds, 1e-9), ss,
                 f, b, cs, cc, cy, a))
    print('')
    print('  centre carries weight 1.000 (fixed, by construction)')
    for k in sorted(tally, key=lambda z: -tally[z]):
        print('  %-6s taps carry weight %8.3f  = %5.1f%% of the answer' % (k, tally[k], 100 * tally[k] / wsum))
    print('  wsum = %.3f   ->  out = %s' % (wsum, np.round(acc / wsum, 3)))


def main():
    ap = argparse.ArgumentParser(description='Synthetic tests for the MB-2 motion blur filter.')
    ap.add_argument('--scene', default='all')
    ap.add_argument('--width', type=int, default=W_DEF)
    ap.add_argument('--height', type=int, default=H_DEF)
    ap.add_argument('--K', type=int, default=96)
    ap.add_argument('--shutter', type=float, default=1.667)
    ap.add_argument('--softz', type=float, default=0.10)
    ap.add_argument('--maxtaps', type=int, default=32)
    ap.add_argument('--minpx', type=float, default=0.5)
    ap.add_argument('--jitter', type=float, default=1.0)
    ap.add_argument('--no-twodir', action='store_true')
    ap.add_argument('--gain', type=float, default=1.0)
    ap.add_argument('--tapjitter', type=float, default=1.0,
                    help='RIG ONLY: scale the per-pixel tap phase hash (0 = every pixel in phase)')
    ap.add_argument('--fix', default='ship', choices=('ship', 'c1', 'c5', 'c6', 'c7', 'c8', 'c9'),
                    help='which candidate weighting to run')
    ap.add_argument('--arclen', action='store_true',
                    help='EXPERIMENT: weight taps by the arc length they represent')
    ap.add_argument('--dump', metavar='DIR', default=None, help='write PNGs here')
    ap.add_argument('--explain', default=None, metavar='X,Y',
                    help='print the tap-by-tap arithmetic for one pixel of --scene')
    ap.add_argument('--skirt-speed', type=float, default=3.0,
                    help='receiver speed for the blade/skirt scenes, px per frame')
    a = ap.parse_args()

    P = Params(a.width, a.height, K=a.K, shutter=a.shutter, max_taps=a.maxtaps,
               floor_px=a.minpx, soft_z=a.softz, tile_jitter=a.jitter,
               two_dir=not a.no_twodir, arclen=a.arclen, fix=a.fix, gain=a.gain,
               tap_jitter=a.tapjitter)
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)

    print('mbsynth -- %dx%d  K=%d  shutter=%.3f  maxTaps=%d  minPx=%.2f  softZ=%.2f  jitter=%.2f  twoDir=%s'
          % (P.W, P.H, P.K, P.shutter, P.max_taps, P.floor_px, P.soft_z, P.tile_jitter, P.two_dir))
    print('tiles = %dx%d   velocity quantised to fp16 (RG16F), depth = 1/z reverse-Z' % P.tiles)

    if a.explain:
        px, py = (int(v) for v in a.explain.split(','))
        sc = {'blade_front': lambda: scene_blade_front(),
              'blade_front_slow': lambda: scene_blade_front(skirt_motion=(0.0, a.skirt_speed)),
              'blade_behind': scene_blade_behind,
              'lone': scene_lone}.get(a.scene, scene_lone)
        explain(P, sc(), px, py)
        return

    s = a.scene
    if s in ('all', 'static'):
        t_static(P, a.dump)
    if s in ('all', 'lone'):
        t_lone(P, a.dump)
    if s in ('all', 'blade_front'):
        t_bladeskirt(P, a.dump, 'T2', scene_blade_front(),
                     'BLADE IN FRONT, STATIC SKIRT BEHIND -- the reported "smear cut by the skirt"')
    if s in ('all', 'blade_front_slow'):
        t_bladeskirt(P, a.dump, 'T3', scene_blade_front(skirt_motion=(0.0, 3.0)),
                     'BLADE IN FRONT, SKIRT BEHIND AND SLOWLY MOVING -- "maybe skirt is too slow"')
    if s in ('all', 'blade_behind'):
        t_bladeskirt(P, a.dump, 'T3b', scene_blade_behind(),
                     'CONTROL: BLADE BEHIND THE SKIRT -- GT itself cuts the smear, correctly')
    if s in ('all', 'softz'):
        t_softz_sweep(P, a.dump)
    if s in ('all', 'slow'):
        t_slow(P, a.dump)
    if s in ('all', 'taps'):
        t_tapcount(P, a.dump)
    if s in ('all', 'skirtspeed'):
        t_skirtspeed(P, a.dump)
    if s in ('all', 'arclen'):
        t_arclen(P, a.dump)
    if s in ('all', 'composition'):
        t_composition(P, a.dump)
    if s in ('all', 'fix'):
        t_fix(P, a.dump)
    if s in ('all', 'thin'):
        t_thin(P, a.dump)
    if s in ('all', 'foliage'):
        t_foliage(P, a.dump)
    if s in ('all', 'stipple'):
        t_stipple(P, a.dump)
    if s in ('all', 'tilegrid'):
        t_tilegrid(P, a.dump)
    if s in ('all', 'twodir'):
        t_twodir(P, a.dump)
    print('')


if __name__ == '__main__':
    main()
