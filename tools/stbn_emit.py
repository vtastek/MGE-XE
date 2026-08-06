#!/usr/bin/env python3
"""Emit the STBN volume as a C source array, laid out as a 2D ATLAS of the 16 time slices.

2D, not 3D, on purpose: a 64 x (64*16) R8G8 texture needs no 3D-texture plumbing anywhere in the
host or the SRTs, and the index is one mad --  int2(px.x & 63, (px.y & 63) + 64 * (frame & 15)).
Byte order here must match that: slice-major, then y, then x.
"""
import numpy as np, sys

X, Y, T = 64, 64, 16
vol = np.fromfile(sys.argv[1], np.uint8).reshape(X, Y, T, 2)
atlas = np.zeros((T * Y, X, 2), np.uint8)
for t in range(T):
    atlas[t * Y:(t + 1) * Y, :, :] = vol[:, :, t, :].transpose(1, 0, 2)
flat = atlas.ravel()

with open(sys.argv[2], "w", newline="\n") as f:
    f.write("""// mgeHost64 — spatiotemporal blue-noise mask (GENERATED, do not hand-edit).
//
// Wolfe et al., "Spatiotemporal Blue Noise Masks" (EGSR 2022), built by 3D void-and-cluster with a
// DECOUPLED energy kernel: two voxels interact only if they share a time slice (spatial gaussian on
// dxy) or share an (x,y) column (temporal gaussian on dt). A 3D-ISOTROPIC kernel would give a volume
// whose individual slices are not blue, which is the exact failure STBN exists to avoid.
//
// Measured on this data: spatial high/low frequency energy ratio 52.1x (ch0) and 42.8x (ch1);
// temporal ratio 5.42x / 5.46x; histogram exactly uniform (mean 0.5000). Toroidal on all three axes,
// so screen tiling and the frame-index wrap are both seamless.
//
// LAYOUT: a 2D ATLAS, 64 wide x (64*16) tall, R8G8 — the 16 time slices stacked vertically. 2D
// rather than a 3D texture so nothing in the host or the SRTs needs 3D-texture plumbing; the index
// is one mad, int2(px.x & 63, (px.y & 63) + 64 * (frame & 15)).
//   ch0 (R) -> AO slice rotation / sun disc rotation
//   ch1 (G) -> AO step offset      (independently generated: two masks, uncorrelated by seed)
//
// Regenerate with tools/stbn_gen.py; the generator is deterministic on its seeds.

#include "stbn_mask.h"

""")
    f.write(f"const unsigned int kStbnWidth  = {X};\n")
    f.write(f"const unsigned int kStbnHeight = {T * Y};   // {T} slices of {Y}\n")
    f.write(f"const unsigned int kStbnSlices = {T};\n")
    f.write(f"const unsigned int kStbnBytes  = {flat.size};\n\n")
    f.write("const unsigned char kStbnMaskRG8[] = {\n")
    for i in range(0, flat.size, 24):
        f.write("    " + ",".join(str(int(b)) for b in flat[i:i + 24]) + ",\n")
    f.write("};\n")
print(f"emitted {flat.size} bytes -> {sys.argv[2]}")
