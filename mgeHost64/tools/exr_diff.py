#!/usr/bin/env python3
"""Compare two host HDR dumps (hdrdump/mge_NNNN.exr) pixel by pixel.

Reads only the layout writeExrHalfRGBA writes: uncompressed single-part scanline, 4 x HALF, planes
A,B,G,R per line, fixed-size blocks at the END of the file. No OpenEXR dependency.

    python3 exr_diff.py A.exr B.exr [tilesX tilesY]

Prints the exact-equal fraction, relative-error percentiles on RGB, and a tile grid of the fraction
of pixels whose relative error exceeds 1e-2, so a difference that lives in one place (an animated
NPC, a flame) reads apart from one that covers the frame (a per-frame constant arrived wrong).

A refactor that is meant to be inert is judged against a NOISE PAIR: two dumps of the unchanged
build at the same save and frame. The candidate's numbers should sit inside that pair's.
"""
import struct
import sys

import numpy as np


def read_exr(path):
    with open(path, "rb") as f:
        data = f.read()
    if struct.unpack_from("<i", data, 0)[0] != 0x01312F76:
        raise SystemExit(f"{path}: not an EXR")
    # dataWindow lives in the header; find it by name rather than walking every attribute.
    k = data.index(b"dataWindow\0box2i\0")
    x0, y0, x1, y1 = struct.unpack_from("<4i", data, k + len(b"dataWindow\0box2i\0") + 4)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    line = w * 8
    block = 8 + line
    start = len(data) - h * block
    raw = np.frombuffer(data, dtype=np.uint8, count=h * block, offset=start).reshape(h, block)
    planes = raw[:, 8:].copy().view(np.float16).reshape(h, 4, w)   # A, B, G, R
    rgb = np.stack([planes[:, 3], planes[:, 2], planes[:, 1]], axis=-1).astype(np.float32)
    return rgb, planes[:, 0].astype(np.float32)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    tx = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    ty = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    a, aa = read_exr(sys.argv[1])
    b, ba = read_exr(sys.argv[2])
    if a.shape != b.shape:
        raise SystemExit(f"size differs: {a.shape} vs {b.shape}")
    h, w, _ = a.shape
    finite = np.isfinite(a).all(-1) & np.isfinite(b).all(-1)
    eq = (a == b).all(-1) & (aa == ba)
    lum_a = a.mean(-1)
    lum_b = b.mean(-1)
    rel = np.abs(a - b).max(-1) / np.maximum(np.maximum(np.abs(a).max(-1), np.abs(b).max(-1)), 1e-4)
    rel = np.where(finite, rel, np.inf)
    print(f"{w}x{h}  exact-equal={eq.mean() * 100:.3f}%  nonfinite={(~finite).sum()}")
    print(f"mean lum A={lum_a[finite].mean():.6f} B={lum_b[finite].mean():.6f} "
          f"ratio={lum_b[finite].mean() / max(lum_a[finite].mean(), 1e-12):.5f}")
    for p in (50, 90, 99, 99.9):
        print(f"  rel err p{p:<5} = {np.percentile(rel[finite], p):.3e}")
    print(f"  pixels rel>1e-3: {(rel > 1e-3).mean() * 100:.3f}%   rel>1e-2: {(rel > 1e-2).mean() * 100:.3f}%"
          f"   rel>1e-1: {(rel > 1e-1).mean() * 100:.3f}%")
    print(f"tile grid {tx}x{ty}: % of pixels with rel>1e-2 (row 0 = top)")
    for j in range(ty):
        row = []
        for i in range(tx):
            t = rel[j * h // ty:(j + 1) * h // ty, i * w // tx:(i + 1) * w // tx]
            row.append(f"{(t > 1e-2).mean() * 100:5.1f}")
        print("  " + " ".join(row))


if __name__ == "__main__":
    main()
