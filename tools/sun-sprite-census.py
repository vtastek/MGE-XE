#!/usr/bin/env python3
"""Census a MW sky sprite (default tx_sun_05.dds) the way R0_candleflame.dds was measured.

WHY THIS EXISTS (tasks/forge-physical-sky.md, P2b step 0).
The Forge host is about to render MW's sun disc as PHYSICAL RADIANCE rather than pinning it to
its authored display value.  A uniform gain cannot do that: `tx_sun_05` is an ALREADY-EXPOSED
glare sprite, so multiplying it re-expands core and skirt by the same factor and a soft sprite
renders as a solid bright square ([[project_sdr_exposed_not_radiance]]).  The operator for that
is scenecolor.h.fsl's expandExposedEmissive(), whose ONE free parameter is a falloff exponent p
acting on `w = saturate(max(rgb) * a)`.

**The exponent follows from the texture, not from taste** — so this script measures the texture:

  * how much of the sprite is CLIPPED (max(RGB) >= 0.99) and how much of that is transparent,
    i.e. is the shape in RGB or in ALPHA (the flame family's defining property);
  * the distribution of the operator's own measure `w`, because p only has leverage where w is
    strictly between 0 and 1 — a sprite whose w is bimodal at {0, 1} cannot be shaped by ANY p;
  * `f(w) = fMin + (1-fMin)*w^p` at candidate exponents, evaluated on the real histogram, as the
    core:skirt gain RATIO the sprite would actually receive;
  * the radial profile of the exposed value, which is what says "core + glare skirt" out loud;
  * the sprite's total exposed ENERGY (mean of rgb*a), which is the quantity the physical sun has
    to match once L = E_normal / Omega_sprite is applied.

Usage:  python3 tools/sun-sprite-census.py [path-to.dds] [--p 1 2 3 4]
Self-contained: DXT1/3/5 + uncompressed BGRA are decoded here, so it runs with numpy alone.
"""

import struct
import sys

import numpy as np


# ---------------------------------------------------------------------------------------------
# DDS -> RGBA8, top mip only.  Enough of the format for MW's texture set (DXT1/DXT3/DXT5 and the
# uncompressed 32/24-bit masks); anything else is refused loudly rather than decoded wrong.
# ---------------------------------------------------------------------------------------------
def _dxt_colors(block):
    c0, c1 = struct.unpack('<HH', block[:4])
    bits = struct.unpack('<I', block[4:8])[0]

    def rgb565(c):
        return (((c >> 11) & 31) * 255 // 31,
                ((c >> 5) & 63) * 255 // 63,
                (c & 31) * 255 // 31)

    a, b = rgb565(c0), rgb565(c1)
    if c0 > c1:
        pal = [a, b,
               tuple((2 * a[i] + b[i]) // 3 for i in range(3)),
               tuple((a[i] + 2 * b[i]) // 3 for i in range(3))]
        opaque = [True] * 4
    else:
        pal = [a, b,
               tuple((a[i] + b[i]) // 2 for i in range(3)),
               (0, 0, 0)]
        opaque = [True, True, True, False]
    out = np.zeros((4, 4, 4), np.uint8)
    for py in range(4):
        for px in range(4):
            idx = (bits >> (2 * (4 * py + px))) & 3
            out[py, px, 0:3] = pal[idx]
            out[py, px, 3] = 255 if opaque[idx] else 0
    return out


def _dxt5_alpha(block):
    a0, a1 = block[0], block[1]
    bits = int.from_bytes(block[2:8], 'little')
    if a0 > a1:
        pal = [a0, a1] + [((6 - i) * a0 + (1 + i) * a1) // 7 for i in range(6)]
    else:
        pal = [a0, a1] + [((4 - i) * a0 + (1 + i) * a1) // 5 for i in range(4)] + [0, 255]
    out = np.zeros((4, 4), np.uint8)
    for py in range(4):
        for px in range(4):
            out[py, px] = pal[(bits >> (3 * (4 * py + px))) & 7]
    return out


def load_dds(path):
    raw = open(path, 'rb').read()
    if raw[:4] != b'DDS ':
        raise SystemExit('%s: not a DDS' % path)
    _, _, h, w, _pitch, _d, mips = struct.unpack('<7I', raw[4:32])
    pf_flags = struct.unpack('<I', raw[80:84])[0]
    fourcc = raw[84:88]
    rgbbits = struct.unpack('<I', raw[88:92])[0]
    masks = struct.unpack('<4I', raw[92:108])
    data = raw[128:]

    img = np.zeros((h, w, 4), np.uint8)
    if pf_flags & 0x4:                                   # DDPF_FOURCC
        if fourcc not in (b'DXT1', b'DXT3', b'DXT5'):
            raise SystemExit('%s: unsupported fourcc %r' % (path, fourcc))
        stride = 8 if fourcc == b'DXT1' else 16
        off = 0
        for by in range(0, h, 4):
            for bx in range(0, w, 4):
                blk = data[off:off + stride]
                off += stride
                if fourcc == b'DXT1':
                    tile = _dxt_colors(blk)
                elif fourcc == b'DXT3':
                    tile = _dxt_colors(blk[8:16])
                    ab = int.from_bytes(blk[0:8], 'little')
                    for py in range(4):
                        for px in range(4):
                            tile[py, px, 3] = ((ab >> (4 * (4 * py + px))) & 15) * 255 // 15
                else:
                    tile = _dxt_colors(blk[8:16])
                    tile[:, :, 3] = _dxt5_alpha(blk[0:8])
                img[by:by + 4, bx:bx + 4] = tile[:min(4, h - by), :min(4, w - bx)]
        return img, w, h, fourcc.decode(), mips

    bypp = rgbbits // 8                                  # uncompressed
    if bypp not in (3, 4):
        raise SystemExit('%s: unsupported %u-bit uncompressed' % (path, rgbbits))
    flat = np.frombuffer(data[:w * h * bypp], np.uint8).reshape(h, w, bypp)
    val = np.zeros((h, w), np.uint32)
    for i in range(bypp):
        val |= flat[:, :, i].astype(np.uint32) << (8 * i)

    def chan(mask):
        if not mask:
            return np.full((h, w), 255, np.uint8)
        sh = (mask & -mask).bit_length() - 1
        return (((val & mask) >> sh) * 255 // (mask >> sh)).astype(np.uint8)

    img[:, :, 0], img[:, :, 1] = chan(masks[0]), chan(masks[1])
    img[:, :, 2], img[:, :, 3] = chan(masks[2]), chan(masks[3])
    return img, w, h, '%ubit' % rgbbits, mips


def main():
    args = [a for a in sys.argv[1:]]
    powers = [1.0, 2.0, 3.0, 4.0]
    if '--p' in args:
        i = args.index('--p')
        powers = [float(x) for x in args[i + 1:]]
        args = args[:i]
    path = args[0] if args else '/mnt/c/mgem/morrowind64/Data Files/textures/tx_sun_05.dds'

    img, w, h, fmt, mips = load_dds(path)
    rgb = img[:, :, 0:3].astype(np.float64) / 255.0
    a = img[:, :, 3].astype(np.float64) / 255.0
    mx = rgb.max(axis=2)
    n = float(w * h)

    print('=== %s  %ux%u %s, %u mips ===' % (path, w, h, fmt, mips))

    # (1) IS THE SHAPE IN RGB OR IN ALPHA?  The flame family's signature: a flat bright RGB plane
    # whose transparent texels are still ~white, so any measure taken from rgb alone reads ~1 over
    # the whole quad and hands the full gain to the corners.
    clipped = mx >= 0.99
    transparent = a < 0.02
    print('\n[1] shape location')
    print('    max(RGB) >= 0.99            : %6.2f%% of texels' % (100.0 * clipped.mean()))
    print('    ...of the TRANSPARENT ones  : %6.2f%%   (flame family reads 97%%)'
          % (100.0 * (clipped & transparent).sum() / max(transparent.sum(), 1)))
    print('    alpha: mean %.3f  median %.3f  below 0.02 %.1f%%'
          % (a.mean(), float(np.median(a)), 100.0 * transparent.mean()))
    print('    rgb  : mean %.3f  median %.3f' % (mx.mean(), float(np.median(mx))))

    # (2) THE OPERATOR'S OWN MEASURE.  p only has leverage where w is strictly interior; a sprite
    # whose w is bimodal at {0,1} cannot be shaped by any exponent and needs a different operator.
    wmeas = np.clip(mx * a, 0.0, 1.0)
    print('\n[2] expandExposedEmissive measure  w = saturate(max(rgb) * a)')
    print('    mean %.4f  median %.4f' % (wmeas.mean(), float(np.median(wmeas))))
    edges = [0.0, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.01]
    hist, _ = np.histogram(wmeas, bins=edges)
    for i in range(len(hist)):
        print('    w in [%.2f, %.2f)  %7u texels  %5.2f%%'
              % (edges[i], edges[i + 1], hist[i], 100.0 * hist[i] / n))
    interior = ((wmeas > 0.02) & (wmeas < 0.99)).mean()
    print('    INTERIOR (0.02 < w < 0.99)  : %5.2f%%   <- the fraction any exponent can shape'
          % (100.0 * interior))

    # (3) WHAT AN EXPONENT ACTUALLY BUYS.  f(w) = fMin + (1-fMin)*w^p with fMin = 1/max(lumaE,1);
    # at a gain L the skirt returns to ~authored and the core keeps ~L.  Reported as the ratio
    # between the mean gain over the CORE (w > 0.9) and over the SKIRT (0.02 < w < 0.5) — the
    # number that decides whether the sprite still reads as a disc with a halo or as a square.
    print('\n[3] falloff exponent -> core:skirt gain ratio  (fMin from a candidate L)')
    core = wmeas > 0.9
    skirt = (wmeas > 0.02) & (wmeas < 0.5)
    print('    core (w>0.9) %5.2f%% of texels, skirt (0.02<w<0.5) %5.2f%%'
          % (100.0 * core.mean(), 100.0 * skirt.mean()))
    for L in (10.0, 100.0, 1000.0):
        fmin = 1.0 / max(L, 1.0)
        row = []
        for p in powers:
            f = fmin + (1.0 - fmin) * np.power(wmeas, p)
            fc = f[core].mean() if core.any() else float('nan')
            fs = f[skirt].mean() if skirt.any() else float('nan')
            row.append('p=%.1f core %.3f skirt %.4f (%.0fx)' % (p, fc, fs, fc / max(fs, 1e-9)))
        print('    L=%-6g fMin=%.4f  %s' % (L, fmin, ' | '.join(row)))

    # (4) THE RADIAL PROFILE.  A glare sprite is a small saturated core inside a wide skirt; a
    # disc-with-hard-edge is not, and would want pinning rather than expansion.
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (min(w, h) / 2.0)
    print('\n[4] radial profile of the exposed value (rgb*a), r normalised to the half-width')
    print('    %-12s %8s %8s %8s' % ('r', 'mean w', 'mean a', 'mean rgb'))
    for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                   (0.5, 0.7), (0.7, 0.9), (0.9, 1.2), (1.2, 9.9)):
        m = (r >= lo) & (r < hi)
        if not m.any():
            continue
        print('    [%.1f, %.1f)   %8.4f %8.4f %8.4f'
              % (lo, hi, wmeas[m].mean(), a[m].mean(), mx[m].mean()))

    # (5) ENERGY.  The physical sun matches the sprite's ENERGY, not its peak, which is what makes
    # the disc dim at sunset and bright at noon on its own.  Reported in the LINEAR domain, since
    # that is where the renderer multiplies (sRGB decode, premultiplied by coverage).
    def srgb_to_linear(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    lin = srgb_to_linear(rgb) * a[:, :, None]
    luma = lin[:, :, 0] * 0.2126 + lin[:, :, 1] * 0.7152 + lin[:, :, 2] * 0.0722
    print('\n[5] exposed energy (LINEAR, premultiplied)')
    print('    mean over the quad          : %.6f' % luma.mean())
    print('    sum / (w*h)  = coverage-mean: %.6f' % (luma.sum() / n))
    print('    peak texel                  : %.6f' % luma.max())
    print('    peak:mean                   : %.1fx' % (luma.max() / max(luma.mean(), 1e-12)))
    print('    mean chroma (R:G:B)         : %.3f : %.3f : %.3f'
          % tuple(lin[:, :, c].mean() / max(luma.mean(), 1e-12) for c in range(3)))
    print('\n    NOTE: the sprite covers Omega_sprite steradians, NOT the sun\'s 6.8e-5 sr.')
    print('    The host computes Omega from the draw\'s world transform and PRINTS it; the ratio')
    print('    to 6.8e-5 is a finding, not a constant to assume.')


if __name__ == '__main__':
    main()
