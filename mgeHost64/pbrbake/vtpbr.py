#!/usr/bin/env python3
"""vtpbr.py -- convert the textures/vt PBR TEST SET to the shipping _paramh / _paramd spec.

vt/ is the hand-built test rig: roughness ladders (gold*, greendia*), parameter spheres, and the
A/B objects w_nord_waraxe.nif draws. It predates the spec the host now reads, so it carries the
old pair -- `<base>_spec.dds` (DXT1) and `<base>_n.dds` (uncompressed R5G6B5 normal map) -- and no
_paramh at all, which means every object in it currently shades on the NON-PBR path.

    <base>_paramh.dds   DXT5   R metal   G roughness   B IOR   A height
    <base>_paramd.dds   BC5    dH/du, dH/dv, per-texture range in dwReserved1  (pbrbake's writer)

    python3 vtpbr.py [--vt DIR] [--dry] [--only NAME]

THE _spec MAPS ARE ALREADY THE RGB HALF OF _paramh, which the pixels say and the names do not:
    gold0_spec   R255 G  0 B164      greendia0_spec   R0 G  0 B131
    gold1_spec   R255 G254 B164      greendia1_spec   R0 G254 B131
    gold2_spec   R255 G 52 B164      greendia25_spec  R0 G 64 B131
    gold35_spec  R255 G 89 B164      greendia5_spec   R0 G129 B131
    gold7_spec   R255 G178 B164      greendia75_spec  R0 G190 B131
R is metal (255 for gold, 0 for the dielectric), B is IOR, and G is roughness read straight off:
52/255 = 0.20, 89/255 = 0.35, 178/255 = 0.70, 64/255 = 0.25, 190/255 = 0.75.

!! SO THE LADDER NAMES LIE, AND ONLY IN ONE PLACE. `gold1` and `greendia1` are roughness 1.0
(G = 254), NOT 0.1 -- every other rung reads as the decimal its digits spell, which is exactly the
pattern that makes the exception invisible. Reading roughness off the name would have put the
roughest rung of each ladder at the second-smoothest position and left the ladder looking wrong in
a way no single object would reveal. The RGB is copied from the _spec map, never re-derived.

Height, in priority order, and what each costs:
  1. the _spec map's own ALPHA where it is DXT5 (greensphere_spec) -- already this spec, just
     misnamed. 4445 distinct levels over a full 0..1 range: a genuine relief map.
  2. the _n normal map, INTEGRATED (Frankot-Chellappa, FFT) -- real relief at full float precision,
     but ONLY when the map survives its own storage format. See the gate below.
  3. flat 0.5 -- no relief, and such an object CANNOT show terracing. Reported, not hidden.

!! THE _n MAPS IN vt/ DO NOT SURVIVE THE GATE, AND THAT IS THE POINT OF HAVING ONE. They are
uncompressed R5G6B5, which gives n.x five bits over [-1,1] -- one code step is 0.0645. The content
is smaller than that: measured n.x runs -0.0323..+0.0323, exactly HALF a code either way, so every
_n in vt/ holds precisely TWO levels in R, TWO in G and ONE in B. It is not a normal map any more,
it is a 1-bit dither pattern; the relief it once had was destroyed on the way into the file.

Integrating it would have produced height out of pure quantiser noise -- and stair-stepped height
is exactly the artifact this rig exists to measure, so the fixture would have been manufacturing
its own signal. So the gate is on CONTENT, not on the file existing: an _n with <= 4 distinct
levels in either tangent channel is refused and the reason is printed. Supply a real normal map
(BC5, or 8-bit RGB) and the path works.

!! THE INTEGRATION'S SIGN IS CHECKED, NOT ASSUMED. A tangent-space normal map has two live
conventions (+Y up / +Y down), and picking wrong inverts every bump. So the result is
re-differentiated by central difference and correlated against the gradients the normal map
actually implies; a negative correlation flips it, and the correlation is printed either way. The
rig cannot see its own conventions unless it looks -- see feedback_a_frame_the_rig_cannot_see.

!! _paramd IS BAKED FROM THE FLOAT HEIGHT, not from the 8-bit alpha that goes into _paramh. That is
the case pbrbake normally cannot get: it refuses an 8-bit source because differentiating an
already-quantised staircase captures its spikes exactly and measures WORSE than not baking at all
(3.47 deg vs 1.99 for the runtime central difference). Here the height exists at full precision
before anything quantises it, so these objects carry the good arm and the bad one over identical
geometry -- which is the comparison the rig is for.

The `def` twins (graydef, whitedef, MBcarddef) are SKIPPED. They are byte-identical to their
partners -- verified, not assumed -- and exist to be the unmodified reference the new material is
judged against. Give them a _paramh and the A/B compares nothing.
"""
import argparse
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pbrbake as PB

if sys.platform == 'win32':
    DEFAULT_VT = r"C:\mgem\morrowind64\Data Files\textures\vt"
    TEXCONV = r"C:\projects\texturematcher\texconv.exe"
else:
    DEFAULT_VT = "/mnt/c/mgem/morrowind64/Data Files/textures/vt"
    TEXCONV = "/mnt/c/projects/texturematcher/texconv.exe"

SUFFIXES = ('_n', '_spec', '_param', '_paramh', '_paramd')
DEFAULT_METAL, DEFAULT_ROUGH, DEFAULT_IOR = 0, 179, 128   # the host's own 1x1 default, in codes


def find_source(vt, stem, suffix):
    """A parameter source for <stem>, looked for in vt/ AND IN ITS PARENT.

    ⚠ THE PARENT IS NOT OPTIONAL. Searching only vt/ is how FH_suit_01_d lost its relief: its
    real normal map (2048^2, 32/64/25 distinct levels, |n| = 1.000 -- nothing like the two-level
    dithers inside vt/) sits in textures/, and only the albedo had been copied across. The object
    that could carry the best height in the whole rig got a flat one, and nothing said so.
    """
    for d in (vt, os.path.dirname(os.path.normpath(vt))):
        cand = os.path.join(d, stem + suffix + '.dds')
        if os.path.exists(cand):
            return cand
    return None


def win_path(p):
    if sys.platform == 'win32':
        return p
    out = subprocess.run(['wslpath', '-w', p], capture_output=True, text=True)
    return out.stdout.strip()


# ---------------------------------------------------------------- readers

def dds_header(path):
    d = open(path, 'rb').read(128)
    if d[:4] != b'DDS ':
        raise ValueError('not a DDS')
    h, w = struct.unpack_from('<II', d, 12)
    return w, h, d[84:88], struct.unpack_from('<I', d, 88)[0]


def load_rgb(path, max_size=0):
    """Level-0 (or a mip at/under max_size) RGB, for DXT1/3/5 and uncompressed R5G6B5."""
    w, h, fourcc, bits = dds_header(path)
    if fourcc in (b'DXT1', b'DXT3', b'DXT5'):
        sys.path.insert(0, os.path.join(HERE, '..', 'pbrsynth'))
        return _load_dxt_rgb(path, max_size)
    if fourcc == b'\0\0\0\0' and bits == 16:
        d = open(path, 'rb').read()
        a = np.frombuffer(d, '<u2', count=w * h, offset=128).reshape(h, w)
        return np.stack([((a >> 11) & 0x1F).astype(np.float32) * (255.0 / 31.0),
                         ((a >> 5) & 0x3F).astype(np.float32) * (255.0 / 63.0),
                         (a & 0x1F).astype(np.float32) * (255.0 / 31.0)], -1)
    raise ValueError('unsupported pixel format %r bits=%d' % (fourcc, bits))


def _bc1(blob, n):
    c0 = blob[:, 0].astype(np.uint16) | (blob[:, 1].astype(np.uint16) << 8)
    c1 = blob[:, 2].astype(np.uint16) | (blob[:, 3].astype(np.uint16) << 8)
    bits = (blob[:, 4].astype(np.uint32) | (blob[:, 5].astype(np.uint32) << 8) |
            (blob[:, 6].astype(np.uint32) << 16) | (blob[:, 7].astype(np.uint32) << 24))

    def u565(c):
        return np.stack([((c >> 11) & 0x1F).astype(np.float32) * (255.0 / 31.0),
                         ((c >> 5) & 0x3F).astype(np.float32) * (255.0 / 63.0),
                         (c & 0x1F).astype(np.float32) * (255.0 / 31.0)], -1)

    e0, e1 = u565(c0), u565(c1)
    pal = np.zeros((n, 4, 3), np.float32)
    pal[:, 0], pal[:, 1] = e0, e1
    wide = (c0 > c1)[:, None]
    pal[:, 2] = np.where(wide, (2 * e0 + e1) / 3.0, (e0 + e1) / 2.0)
    pal[:, 3] = np.where(wide, (e0 + 2 * e1) / 3.0, 0.0)
    idx = np.zeros((n, 16), np.uint8)
    for i in range(16):
        idx[:, i] = ((bits >> (2 * i)) & 3).astype(np.uint8)
    return np.take_along_axis(pal, idx[:, :, None].astype(np.intp), 1).reshape(n, 4, 4, 3)


def _load_dxt_rgb(path, max_size=0):
    d = open(path, 'rb').read()
    h, w = struct.unpack_from('<II', d, 12)
    stride = {b'DXT1': 8, b'DXT3': 16, b'DXT5': 16}[d[84:88]]
    mips, = struct.unpack_from('<I', d, 28)
    off = 128
    if max_size > 0:
        for _ in range(max(0, mips - 1)):
            if max(w, h) <= max_size:
                break
            off += max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * stride
            w, h = max(1, w // 2), max(1, h // 2)
    bw, bh = (w + 3) // 4, (h + 3) // 4
    body = np.frombuffer(d, np.uint8, count=bw * bh * stride, offset=off).reshape(bw * bh, stride)
    blk = _bc1(np.ascontiguousarray(body[:, stride - 8:]), bw * bh)
    img = blk.reshape(bh, bw, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 3)
    return np.clip(img[:h, :w], 0, 255).astype(np.float32)


# ---------------------------------------------------------------- height

def integrate_normal(nrm):
    """Frankot-Chellappa: least-squares height whose gradient best matches the normal map.

    Returns (height in 0..1, sign-check correlation). The height is periodic, which is right here:
    every one of these maps tiles."""
    n = nrm / 127.5 - 1.0
    nz = np.where(np.abs(n[..., 2]) < 1e-3, 1e-3, n[..., 2])
    p, q = -n[..., 0] / nz, -n[..., 1] / nz          # dH/dx, dH/dy in texel units
    h, w = p.shape
    wx = 2.0 * np.pi * np.fft.fftfreq(w)[None, :]
    wy = 2.0 * np.pi * np.fft.fftfreq(h)[:, None]
    denom = wx ** 2 + wy ** 2
    denom[0, 0] = 1.0
    Z = (-1j * wx * np.fft.fft2(p) - 1j * wy * np.fft.fft2(q)) / denom
    Z[0, 0] = 0.0
    z = np.real(np.fft.ifft2(Z))
    # SIGN CHECK: re-differentiate and correlate against the gradients the normal map implies.
    gx = 0.5 * (np.roll(z, -1, 1) - np.roll(z, 1, 1))
    gy = 0.5 * (np.roll(z, -1, 0) - np.roll(z, 1, 0))
    a = np.stack([gx, gy]).ravel()
    b = np.stack([p, q]).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    corr = float((a * b).sum() / d) if d > 1e-12 else 0.0
    if corr < 0:
        z, corr = -z, -corr
    rng = z.max() - z.min()
    return ((z - z.min()) / rng if rng > 1e-9 else np.full_like(z, 0.5)), corr, float(rng)


def resize_nn(a, size):
    """Resample to size x size. DOWNSCALING AREA-AVERAGES, it does not point-sample.

    Point-sampling a 4096 height down to 1024 throws away three of every four texels and aliases
    what is left, which shows up as per-texel jumps the derivative bake then has to encode -- it
    put greensphere's range at 2^0 when the real relief needs far less. A height is a signal; it
    gets filtered like one."""
    if a.shape[0] == size and a.shape[1] == size:
        return a
    if a.shape[0] > size and a.shape[0] % size == 0 and a.shape[1] == a.shape[0]:
        k = a.shape[0] // size
        sh = (size, k, size, k) + a.shape[2:]
        return a.reshape(sh).mean(axis=(1, 3))
    yi = (np.arange(size) * (a.shape[0] / float(size))).astype(np.intp)
    xi = (np.arange(size) * (a.shape[1] / float(size))).astype(np.intp)
    return a[yi][:, xi]


# ---------------------------------------------------------------- emit

def write_paramh(out_dds, rgba, dry):
    from PIL import Image
    png = out_dds[:-4] + '.png'
    Image.fromarray(rgba, 'RGBA').save(png)
    if dry:
        return 'dry (png only)'
    # texconv.exe runs directly under WSL interop; a powershell wrapper only adds a quoting layer
    # that breaks on the space in "Data Files". Its exit code is not a reliable success signal
    # (it returns 2 on a clean write), so the OUTPUT FILE is what decides.
    cmd = [TEXCONV, '-nologo', '-y', '-f', 'BC3_UNORM',
           '-o', win_path(os.path.dirname(out_dds)), win_path(png)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    made = out_dds[:-4] + '.DDS'
    for cand in (made, out_dds):
        if os.path.exists(cand):
            if cand != out_dds:
                os.replace(cand, out_dds)
            os.remove(png)
            return 'ok'
    return 'texconv FAILED: %s' % ((r.stdout + r.stderr).strip()[-160:] or 'no output')


def write_paramd(out_dds, height01, dry):
    """Central-difference the FULL-PRECISION height and quantise the RESULT -- pbrbake's contract."""
    H = height01.astype(np.float64)
    du = 0.5 * (np.roll(H, -1, 1) - np.roll(H, 1, 1))
    dv = 0.5 * (np.roll(H, -1, 0) - np.roll(H, 1, 0))
    peak = float(np.percentile(np.maximum(np.abs(du), np.abs(dv)), PB.DERIV_PCT))
    if peak <= 1e-9:
        return 'skipped (height is flat)'
    exp = int(np.ceil(np.log2(peak / 0.5)))
    exp = max(-128, min(127, exp))
    rng = 2.0 ** exp
    levels, n0, w, h = [], H.shape[1], H.shape[1], H.shape[0]
    while True:
        levels.append(PB.bc5_encode_snorm(np.clip(du / rng, -1, 1), np.clip(dv / rng, -1, 1)))
        if max(w, h) <= 4:
            break
        w, h = max(1, w // 2), max(1, h // 2)
        du, dv = PB.box_down(du, w), PB.box_down(dv, w)
    if dry:
        return 'dry (%d mips, 2^%d)' % (len(levels), exp)
    PB.write_dds_bc5(out_dds, levels, exp, n0)
    why = PB.verify_dds_bc5(out_dds, n0, len(levels), exp)
    if why:
        os.remove(out_dds)
        return 'REJECTED: %s' % why
    return 'ok (%d mips, range 2^%d)' % (len(levels), exp)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vt', default=DEFAULT_VT)
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--only', default='')
    a = ap.parse_args()

    files = [f for f in sorted(os.listdir(a.vt)) if f.lower().endswith('.dds')]
    bases = []
    for f in files:
        stem = f[:-4]
        if any(stem.lower().endswith(s) for s in SUFFIXES) or ' - Copy' in stem:
            continue
        if stem.lower().endswith('def'):
            print('%-24s SKIP - the unmodified A/B reference twin' % stem)
            continue
        if a.only and a.only.lower() not in stem.lower():
            continue
        bases.append(stem)
    print('\n%d base textures to convert\n' % len(bases))

    print('%-22s %-9s %-26s %-28s %s' % ('base', 'size', 'material', 'height', 'paramd'))
    for stem in bases:
        base = os.path.join(a.vt, stem)
        # `_param` is the authored DXT1 material map (metal/rough/IOR, no height); `_spec` is the
        # older name for the same packing. Either is EVIDENCE and is copied, never re-derived.
        spec = find_source(a.vt, stem, '_param') or find_source(a.vt, stem, '_spec')
        nrm = find_source(a.vt, stem, '_n')

        # ---- material RGB
        src_size = None
        if spec:
            sw, sh, sfc, _ = dds_header(spec)
            src_size = min(sw, 2048)
            rgb = resize_nn(load_rgb(spec, src_size), src_size)
            mat = 'from %s (%s %d)' % (os.path.basename(spec).replace(stem, ''),
                                       sfc.decode('ascii', 'replace'), sw)
        else:
            src_size = 64
            rgb = np.zeros((64, 64, 3), np.float32)
            rgb[..., 0], rgb[..., 1], rgb[..., 2] = DEFAULT_METAL, DEFAULT_ROUGH, DEFAULT_IOR
            mat = 'DEFAULT (no _spec)'

        # ---- height
        hf, hnote = None, ''
        if spec and dds_header(spec)[2] == b'DXT5':
            H, why = PB.read_paramh_alpha(spec)
            if H is not None:
                # ⚠ NO /255 HERE. pbrsynth's bc4_decode_file_blocks returns floats already in
                # [0,1], so read_paramh_alpha does too. Dividing again squashed greensphere's
                # height to 1/255 of its range -- it wrote out FLAT, and its _paramd range came out
                # 255x too small. Same class as the pbrbake header bug: the repair was right, the
                # quantity was not. The flatness guard below is what now catches it.
                hf = resize_nn(H.astype(np.float64), src_size)
                hnote = 'from _spec ALPHA (already this spec)'
        if hf is None and nrm:
            nim = load_rgb(nrm)
            lv = (len(np.unique(nim[..., 0])), len(np.unique(nim[..., 1])))
            if min(lv) <= 4:
                hnote = '_n REFUSED (%d/%d levels - dither, not relief)' % lv
            else:
                nw = min(dds_header(nrm)[0], 2048)
                nim = resize_nn(nim, nw)
                if nw != src_size:
                    src_size = nw
                    rgb = resize_nn(rgb, src_size)
                h01, corr, rng = integrate_normal(nim)
                hf = h01
                hnote = 'integrated _n (corr %+.3f, span %.4f)' % (corr, rng)
        if hf is None:
            hf = np.full((src_size, src_size), 0.5, np.float64)
            if not hnote:
                hnote = 'FLAT - cannot show terracing'

        rgba = np.zeros((src_size, src_size, 4), np.uint8)
        rgba[..., :3] = np.clip(resize_nn(rgb, src_size), 0, 255).astype(np.uint8)
        rgba[..., 3] = np.clip(np.rint(hf * 255.0), 0, 255).astype(np.uint8)

        # ⚠ A HEIGHT THAT VARIED AT THE SOURCE MUST STILL VARY IN THE BYTES WE ARE ABOUT TO WRITE.
        # This is the check that was missing when a double /255 quietly flattened greensphere: the
        # source read std 0.10 over 4445 levels and the file came out with two. Checking the OUTPUT
        # is the only way to catch a scaling error, because every input was fine.
        if hf.std() > 1e-3 and rgba[..., 3].std() < 1.0:
            print('%-22s ABORT - height varied at source (std %.4f) but writes FLAT'
                  % (stem, hf.std()))
            continue

        r1 = write_paramh(base + '_paramh.dds', rgba, a.dry)
        flat = float(hf.max() - hf.min()) < 1e-9
        r2 = 'skipped (flat height)' if flat else write_paramd(base + '_paramd.dds', hf, a.dry)
        print('%-22s %-9s %-26s %-28s %s' % (stem, '%d^2' % src_size, mat, hnote, r1 if r1 != 'ok' else r2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
