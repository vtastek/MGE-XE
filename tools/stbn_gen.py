#!/usr/bin/env python3
"""Spatiotemporal blue-noise mask generator (Wolfe et al., EGSR 2022) via 3D void-and-cluster.

The property that makes this different from plain 3D blue noise: the energy kernel is DECOUPLED,
not isotropic. Two voxels only interact if they share a t-slice (spatial gaussian on dxy) or share
an (x,y) column (temporal gaussian on dt). A 3D-isotropic kernel gives a volume whose SLICES are
not individually blue -- which is exactly the failure mode STBN exists to avoid.

Toroidal in all three axes, so screen tiling and the frame-index wrap are both seamless.
"""
import numpy as np, sys

X, Y, T = 64, 64, 16
SIG_S, SIG_T = 1.9, 1.9
RS = 6                                  # spatial kernel half-width (~3.2 sigma)

def build_kernels():
    dx = np.arange(-RS, RS + 1)
    gs = np.exp(-(dx[:, None] ** 2 + dx[None, :] ** 2) / (2 * SIG_S ** 2))
    gs[RS, RS] = 1.0
    dt = np.arange(T)
    dt = np.minimum(dt, T - dt)          # toroidal time distance
    gt = np.exp(-(dt ** 2) / (2 * SIG_T ** 2))
    gt[0] = 1.0
    return gs.astype(np.float64), gt.astype(np.float64)

GS, GT = build_kernels()
SX = (np.arange(-RS, RS + 1)[:, None] * np.ones(2 * RS + 1, int)).ravel()
SY = (np.ones(2 * RS + 1, int)[:, None] * np.arange(-RS, RS + 1)).ravel()
GSF = GS.ravel()

def splat(E, x, y, t, sign):
    xs = (x + SX) % X
    ys = (y + SY) % Y
    np.add.at(E[:, :, t], (xs, ys), sign * GSF)      # spatial, same slice
    ts = (t + np.arange(T)) % T
    E[x, y, ts] += sign * GT                          # temporal, same column

def energy(pattern):
    E = np.zeros((X, Y, T))
    for (x, y, t) in zip(*np.nonzero(pattern)):
        splat(E, x, y, t, +1.0)
    return E

def tightest(E, pattern):                             # max energy among ONES
    m = np.where(pattern, E, -np.inf)
    return np.unravel_index(np.argmax(m), m.shape)

def largest_void(E, pattern):                         # min energy among ZEROS
    m = np.where(pattern, np.inf, E)
    return np.unravel_index(np.argmin(m), m.shape)

def generate(seed):
    rng = np.random.default_rng(seed)
    N = X * Y * T
    n0 = N // 10
    pattern = np.zeros((X, Y, T), bool)
    idx = rng.choice(N, n0, replace=False)
    pattern.ravel()[idx] = True
    E = energy(pattern)

    # Phase 0 -- relax the random seed pattern into a blue-noise one.
    for _ in range(4 * n0):
        cx, cy, ct = tightest(E, pattern)
        pattern[cx, cy, ct] = False; splat(E, cx, cy, ct, -1.0)
        vx, vy, vt = largest_void(E, pattern)
        if (vx, vy, vt) == (cx, cy, ct):
            pattern[cx, cy, ct] = True; splat(E, cx, cy, ct, +1.0)
            break
        pattern[vx, vy, vt] = True; splat(E, vx, vy, vt, +1.0)

    rank = np.full((X, Y, T), -1, np.int64)

    # Phase 1 -- rank the initial ones downward by repeatedly removing the tightest cluster.
    work = pattern.copy(); Ew = E.copy()
    for r in range(n0 - 1, -1, -1):
        cx, cy, ct = tightest(Ew, work)
        work[cx, cy, ct] = False; splat(Ew, cx, cy, ct, -1.0)
        rank[cx, cy, ct] = r

    # Phase 2 -- fill the rest upward by repeatedly inserting into the largest void.
    work = pattern.copy(); Ew = E.copy()
    for r in range(n0, N):
        vx, vy, vt = largest_void(Ew, work)
        work[vx, vy, vt] = True; splat(Ew, vx, vy, vt, +1.0)
        rank[vx, vy, vt] = r
        if r % 8192 == 0:
            print(f"  seed {seed}: {r}/{N}", flush=True)

    assert (rank >= 0).all()
    return (rank * 256 // N).astype(np.uint8)          # rank -> uint8, uniform histogram

def report(vol, name):
    v = vol.astype(np.float64) / 255.0
    # Spatial: FFT of one slice, radially averaged. Blue = low energy at low frequency.
    sl = np.fft.fftshift(np.abs(np.fft.fft2(v[:, :, 0] - v[:, :, 0].mean())))
    cy_, cx_ = X // 2, Y // 2
    yy, xx = np.mgrid[0:X, 0:Y]
    rr = np.hypot(yy - cy_, xx - cx_).astype(int)
    prof = np.bincount(rr.ravel(), sl.ravel()) / np.maximum(np.bincount(rr.ravel()), 1)
    lo = prof[1:5].mean(); hi = prof[12:28].mean()
    # Temporal: per-pixel sequence, same test along t.
    ft = np.abs(np.fft.fft(v - v.mean(axis=2, keepdims=True), axis=2)).mean(axis=(0, 1))
    tlo = ft[1:3].mean(); thi = ft[6:9].mean()
    print(f"{name}: mean {v.mean():.4f} (want 0.5)")
    print(f"  SPATIAL  low-freq {lo:.3f}  high-freq {hi:.3f}  ratio {hi/max(lo,1e-9):.2f}x  (>1 = blue)")
    print(f"  TEMPORAL low-freq {tlo:.3f}  high-freq {thi:.3f}  ratio {thi/max(tlo,1e-9):.2f}x  (>1 = blue)")

if __name__ == "__main__":
    out = np.zeros((X, Y, T, 2), np.uint8)
    for ch, seed in enumerate((12345, 67890)):
        print(f"channel {ch} (seed {seed})...", flush=True)
        vol = generate(seed)
        report(vol, f"channel {ch}")
        out[:, :, :, ch] = vol
    out.tofile(sys.argv[1] if len(sys.argv) > 1 else "stbn_64x64x16_rg8.bin")
    print(f"wrote {out.size} bytes ({X}x{Y}x{T} RG8)")
