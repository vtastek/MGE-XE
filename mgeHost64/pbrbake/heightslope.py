#!/usr/bin/env python3
"""What slope does a land _paramh actually ask for?

`depth` in pbrPerturb* is a pure slope multiplier: the world height is depth*L*h and the world
horizontal is L*duv, so L (world units per UV unit) CANCELS and

    tangent-space slope = depth * dH/duv

That is why the convention is "height range as a fraction of one UV unit" and why the same number
means the same thing on a 7.3 m land tile and a 1.8 m wall tile.

So the question "is 0.025 weak" is really "what is dH/duv in these maps", which is measurable.
Decodes BC3 alpha (= height) at mip 0, builds the mip chain by box filter (what the GPU samples),
and reports the central-difference gradient arm 4 would see at each level.
"""
import os, struct, sys, glob
import numpy as np

TEXDD = "/mnt/c/mgem/morrowind64/Data Files/texdd"

def dds_info(b):
    assert b[:4] == b"DDS ", "not a dds"
    h = struct.unpack_from("<7I", b, 4)
    size, flags, height, width, pitch, depth, mips = h
    fourcc = b[84:88]
    return width, height, max(mips, 1), fourcc

def bc3_alpha(b, off, w, h):
    """Decode just the alpha (height) plane of a BC3 surface."""
    bw, bh = (w + 3) // 4, (h + 3) // 4
    out = np.zeros((bh * 4, bw * 4), dtype=np.uint8)
    blocks = np.frombuffer(b, dtype=np.uint8, count=bw * bh * 16, offset=off)
    blocks = blocks.reshape(bh, bw, 16)
    a0 = blocks[:, :, 0].astype(np.uint16)
    a1 = blocks[:, :, 1].astype(np.uint16)
    # the 48 index bits, little-endian across bytes 2..7
    bits = np.zeros((bh, bw), dtype=np.uint64)
    for i in range(6):
        bits |= blocks[:, :, 2 + i].astype(np.uint64) << np.uint64(8 * i)
    # the 8-entry palette, both BC3 modes
    pal = np.zeros((bh, bw, 8), dtype=np.float64)
    pal[:, :, 0] = a0
    pal[:, :, 1] = a1
    six = a0 > a1
    for i in range(1, 7):                       # 6-value mode: 6 interpolants
        pal[:, :, i + 1] = np.where(six, ((6 - i) * a0 + i * a1) / 6.0, 0)
    for i in range(1, 5):                       # 4-value mode: 4 interpolants + 0 and 255
        m = ~six
        pal[:, :, i + 1] = np.where(m, ((4 - i) * a0 + i * a1) / 4.0, pal[:, :, i + 1])
    pal[:, :, 6] = np.where(six, pal[:, :, 6], 0.0)
    pal[:, :, 7] = np.where(six, pal[:, :, 7], 255.0)
    for py in range(4):
        for px in range(4):
            k = py * 4 + px
            idx = ((bits >> np.uint64(3 * k)) & np.uint64(7)).astype(np.intp)
            out[py::4, px::4] = np.take_along_axis(pal, idx[:, :, None], axis=2)[:, :, 0].astype(np.uint8)
    return out[:h, :w]

def grad_at_level(hh, lvl):
    """Box-downsample to `lvl`, then arm 4's +-1-texel central difference, in dH/duv."""
    a = hh.astype(np.float64) / 255.0
    for _ in range(lvl):
        if a.shape[0] < 2 or a.shape[1] < 2:
            break
        a = 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])
    n = a.shape[0]
    # wrap, as the sampler does
    du = (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) * (n * 0.5)
    dv = (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) * (n * 0.5)
    return np.hypot(du, dv), a, n

names = sys.argv[1:]
if not names:
    names = sorted(glob.glob(os.path.join(TEXDD, "tx_a?_*_paramh.dds")))[:14]

print("%-34s %5s %6s %6s  %s" % ("land texture", "W", "hmin", "hmax", "dH/duv (median) -> deg at depth 0.025"))
print("-" * 118)
rows = []
for p in names:
    b = open(p, "rb").read()
    w, h, mips, fourcc = dds_info(b)
    if fourcc != b"DXT5":
        print("%-34s  SKIP fourcc=%s" % (os.path.basename(p)[:34], fourcc))
        continue
    hh = bc3_alpha(b, 128, w, h)
    line = []
    for lvl in (0, 2, 4, 6):
        g, a, n = grad_at_level(hh, lvl)
        med = float(np.median(g))
        deg = np.degrees(np.arctan(med * 0.025))
        line.append("L%d %7.1f/%4.1f deg" % (lvl, med, deg))
        if lvl == 0:
            rows.append(med)
    print("%-34s %5d %6.3f %6.3f  %s" % (os.path.basename(p).replace("_paramh.dds", "")[:34],
          w, hh.min() / 255.0, hh.max() / 255.0, "  ".join(line)))

if rows:
    m = float(np.median(rows))
    print("-" * 118)
    print("median dH/duv at mip 0 over %d land maps: %.1f" % (len(rows), m))
    for d in (0.025, 0.05, 0.1, 0.2, 0.4):
        print("   depth %-6.3f -> %5.1f deg at mip 0" % (d, np.degrees(np.arctan(m * d))))
    for target in (20.0, 30.0, 40.0):
        print("   %4.0f deg at mip 0 needs depth %.4f" % (target, np.tan(np.radians(target)) / m))
