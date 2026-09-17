#!/usr/bin/env python3
"""
pbrbake -- bake a BC5 DERIVATIVE MAP (<base>_paramd.dds) from a 16-bit source displacement map.

WHY, in one line: differentiate BEFORE quantising instead of after.

The shipped `<base>_paramh.dds` carries height in a DXT5 alpha block -- 8 bits, and
texturematcher rounds to 8 bits BEFORE compressing, so the 16-bit source is already gone by the
time the file exists.  The shader then DIFFERENTIATES that field, and differentiation is where
quantisation stops being invisible: a staircase in the stored scalar is a train of spikes in its
slope.  That is the terracing, and mgeHost64/pbrsynth measured that NO runtime reconstruction
survives it -- on a height field spanning 3/255 of range, cd / sobel / bicubic / B-spline every
one flattens to gradient retention EXACTLY 0.000, while a derivative baked at full precision and
then quantised keeps 1.001.

⚠⚠ AND BAKING FROM THE SHIPPED `_paramh` IS WORSE THAN NOT BAKING AT ALL.  Measured, pbrsynth
`deriv5q`: 3.08 deg against cd's 1.99 on the noise scene, 0.88 against 0.66 on the dome, and on
the terracing scene it is bit-for-bit as bad as cd (retention 0.000).  Differentiating a
staircase at full precision captures its spikes EXACTLY and then requantises them.  So this tool
reads the 16-bit SOURCE or it does not run: a `_paramh` whose source PNG is gone gets no
derivative map, and the runtime arms (pbrGradMode 0 with pbrGradRadius) remain the answer for it.

WHAT IT WRITES.  <base>_paramd.dds, DX10 header, DXGI_FORMAT_BC5_SNORM (84): R = dH/du, G = dH/dv,
in PER-UV units divided by a per-texture range, with a full mip chain.  ⚠ PER UV, and every level
differentiated at its own size -- the first format stored per-TEXEL and averaged the derivative to
make mips, which is 2x too small at every level and arrives on screen as a hard line at each mip
boundary.  See the comment in bake_one; it is the seam, and the reason the shader now needs no LOD
reconstruction at all.  BC5 is two INDEPENDENT BC4 blocks,
so each derivative axis gets the same 8-level-per-block ramp the height had to itself -- which is
the whole reason a stored derivative is affordable at all, and why this is not the abandoned
"two axes in a BC1 colour block" packing (there they would share one line in RGB space).

UNITS: dH/dtexel divided by a PER-TEXTURE RANGE, and the range is a power of two carried in the
file.  The difference is taken WRAPPED, because MW's default address mode is WRAP_S_WRAP_T and
this art tiles (on a CLAMP texture the wrap is wrong for exactly the one-texel border, which is
where such a texture holds its edge texel anyway).

⚠⚠ THE RANGE IS THE WHOLE FEATURE, AND TWO EARLIER DESIGNS WITHOUT ONE BOTH FAILED THE
MEASUREMENT.  A fixed gain of 2 is unclippable -- a wrapped central difference of a [0,1] field is
bounded by +-0.5 -- and it is USELESS: pbrsynth scores it at 0.159 deg with gradient retention
0.000 on the terracing scene, which is bit for bit as bad as the runtime central difference it was
meant to replace.  The reason is that BC4 is not the fully adaptive quantiser it looks like: each
4x4 block carries its own endpoints, but those endpoints are themselves on a GLOBAL 8-bit grid, so
a block whose values all sit within 1/127 of each other collapses to e0 == e1 -- one constant value
for the whole block, gradient zero.  A shallow height map's derivative is 1e-5 in global units and
every block of it collapses.  With a per-texture range the same scene scores 0.009 deg at retention
0.955, and `noise` goes from 3.47 deg (worse than not baking) to 0.76 (2.6x better than not
baking).  There is no version of this format without a per-texture range that is worth shipping.

WHERE THE RANGE LIVES.  A POWER OF TWO, as a signed exponent, written into the DDS header's own
dwReserved1 area behind a magic -- and ALSO recoverable without it, because the tool names the
file <base>_paramd.dds and the exponent is the only thing the host needs.  A power of two rather
than the exact range costs at most one bit of the eight (the range rounds up, so at worst half the
codes go unused) and buys the host a lane it can pack beside the texture slot instead of a second
float and a per-slot table.  Rounding UP also guarantees nothing clips.

MIPS.  The HEIGHT is box-filtered at full precision and each level differentiated AT ITS OWN SIZE,
then compressed separately.  Filtering commutes with differentiation -- but only for the same units:
the average of a per-TEXEL derivative is still in the finer level's texel units, which is exactly
the trap the first format fell into.  In per-UV units the statement holds as written, and that is
why a derivative map mips correctly where a normal map does not.

ALIGNMENT IS VERIFIED, NOT ASSUMED.  The source PNG reaches the shipped `_paramh` through
texturematcher's db.json (texture -> thumbnail -> staging file) and an optional rotation/tiling
transform.  Rather than reimplement that pipeline and hope, every bake is checked against the
shipped height it must line up with: correlation below --min-corr is REFUSED and reported.  That
catches a wrong thumbnail match, a rotated source and a tiled source alike, none of which would
be visible in the output on its own.

Run:  python3 pbrbake.py --dry-run        (match, verify and report; writes nothing)
      python3 pbrbake.py                  (bake)
      python3 pbrbake.py --only tx_ac_dirt_01
"""

import argparse
import json
import os
import struct
import sys
import zlib

import numpy as np

# The DDS dwReserved1 magic and the exponent bias -- the three numbers this tool, parseDds and
# pbrmaterial.h.fsl must agree on.  The exponent is stored biased so it survives as an unsigned byte
# in the instance lane; range = 2^(exp) with exp in [-128, 127].
DERIV_MAGIC = 0x4445474D            # 'MGED' little-endian
DERIV_EXP_BIAS = 128
# The percentile of |dH/dtexel| the range is fitted to. p99.9 rather than the max: measured
# marginally better on the noise scene (0.756 deg against 0.891) because the top 0.1% of texels --
# the steepest cliff edges -- otherwise stretch the range for everything else. They clamp.
DERIV_PCT = 99.9

DEFAULT_GAME_TEX = "/mnt/c/mgem/morrowind64/Data Files/textures"
DEFAULT_TM       = "/mnt/c/projects/texturematcher"
DEFAULT_MANIFEST = "/mnt/c/projects/texturematcher/texel_density_manifest.csv"
# The project's own PNG decoder, used only where Pillow cannot represent the file (_texconv_r16).
TEXCONV          = "/mnt/c/projects/texturematcher/texconv.exe"


# =======================================================================================
# BC4 / BC5 -- a vectorised ENCODER, held to pbrsynth's byte-level one by selftest()
# =======================================================================================
#
# pbrsynth owns the reference implementation and T0 proves it is the real bit layout; this is the
# same rule written to run over a million blocks at a time.  They are cross-checked at startup
# rather than trusted to stay in step, because "the encoder in the tool" and "the encoder in the
# rig" silently diverging is exactly how a format bug ships with a green test suite.


def _blocks(img):
    h, w = img.shape
    return img.reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 16)


def bc4_encode_snorm(img):
    """(H, W) float in [-1, 1] -> (nblocks, 8) uint8 of real BC4_SNORM blocks.

    Endpoints are the block's min/max in the 6-interpolant mode (e0 > e1), which is what every
    encoder picks for a smooth field; the rig measured the index quantisation, not endpoint search,
    and endpoint search is not where a derivative map's error lives.
    """
    q = np.clip(np.rint(np.asarray(img, dtype=np.float64) * 127.0), -127, 127)
    B = _blocks(q)
    e0 = B.max(axis=1)
    e1 = B.min(axis=1)
    const = (e0 == e1)
    # The 8 decode values of each block, as the hardware builds them for e0 > e1.
    k = np.arange(6, dtype=np.float64)
    ramp = np.empty((B.shape[0], 8), dtype=np.float64)
    ramp[:, 0] = e0
    ramp[:, 1] = e1
    ramp[:, 2:8] = ((6 - k)[None, :] * e0[:, None] + (1 + k)[None, :] * e1[:, None]) / 7.0
    idx = np.argmin(np.abs(B[:, :, None] - ramp[:, None, :]), axis=2).astype(np.uint64)
    idx[const] = 0
    out = np.zeros((B.shape[0], 8), dtype=np.uint8)
    out[:, 0] = (e0.astype(np.int64) & 0xFF).astype(np.uint8)      # int8 two's complement
    out[:, 1] = (e1.astype(np.int64) & 0xFF).astype(np.uint8)
    bits = np.zeros(B.shape[0], dtype=np.uint64)
    for t in range(16):
        bits |= idx[:, t] << np.uint64(3 * t)
    for b in range(6):
        out[:, 2 + b] = ((bits >> np.uint64(8 * b)) & np.uint64(0xFF)).astype(np.uint8)
    return out


def bc5_encode_snorm(du, dv):
    """BC5 is two BC4 blocks, red then green, interleaved per block."""
    r = bc4_encode_snorm(du)
    g = bc4_encode_snorm(dv)
    return np.concatenate([r, g], axis=1).reshape(-1)


def selftest():
    """Hold the fast encoder to pbrsynth's byte-level one, and the whole chain to a round-trip."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pbrsynth"))
    import pbrsynth as P
    rng = np.random.default_rng(4)
    img = rng.random((16, 16)) * 2.0 - 1.0
    mine = bc4_encode_snorm(img)
    ref = _blocks(img)
    bad = 0
    for i in range(ref.shape[0]):
        if bytes(mine[i]) != P.bc4_encode_block(ref[i], True):
            bad += 1
    print("  selftest: BC4_SNORM bytes vs pbrsynth's reference encoder -- %d of %d blocks differ"
          % (bad, ref.shape[0]))
    dec = P.bc4_roundtrip(img, True)
    print("  selftest: round-trip max abs error %.5f (8 levels on the block's own range)"
          % float(np.abs(dec - img).max()))
    # A constant block must be EXACT: it is what a flat region of a derivative map is made of, and
    # a flat region reading back as anything but zero is a bump on a flat wall.
    c = np.zeros((4, 4))
    print("  selftest: constant-zero block round-trips to %.6f (must be 0)"
          % float(np.abs(P.bc4_roundtrip(c, True)).max()))
    return bad == 0


# =======================================================================================
# PNG (16-bit) and DDS
# =======================================================================================


def png_ihdr(path):
    """(width, height, bitdepth, colourtype) read from the FILE, not from a decoder.

    The file's own header is the only thing that says how many bits the art has. Asking a decoder
    what it returned answers a different question, and the difference is what broke this tool --
    see load_png_gray.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(26)
        if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
            return None
        w, h = struct.unpack(">II", head[16:24])
        return w, h, head[24], head[25]
    except Exception:
        return None


def _texconv_r16(path):
    """Decode a 16-bit PNG to raw R16_UNORM through texconv, because Pillow cannot.

    ⚠⚠ PILLOW SILENTLY HALVES THESE FILES, AND THIS TOOL BELIEVED IT. Pillow has no
    16-bit-per-channel RGB mode: a PNG with IHDR bit depth 16 and colour type 2 opens as mode
    'RGB', dtype uint8, with the LOW BYTE OF EVERY SAMPLE DISCARDED and no warning. numpy then
    sees uint8, and the check below concluded "source is 8-bit" and refused to bake -- for 77 of
    209 maps, every one of which had a real 16-bit source sitting on disk the whole time. The FILE
    was 16-bit; the READER was 8-bit, and the report named the wrong one.

    Measured on coast_sand_02_disp_4k.png: 233 distinct values through Pillow, 55705 through this.
    That is the 8 bits this entire tool exists to keep.

    texconv rather than a hand-rolled decoder: these PNGs are Paeth-filtered on 4095 of 4096 rows,
    and Paeth is a per-byte recurrence along the row that numpy cannot vectorise -- a Python loop
    over 16.7M groups per image. texconv is already the project's decoder and does it natively in
    ~2.5 s. Returns None (never a half-precision guess) if it is missing or fails.
    """
    import shutil
    import subprocess
    import tempfile
    exe = TEXCONV
    if not os.path.isfile(exe):
        return None
    work = tempfile.mkdtemp(prefix="pbrbake16_")
    try:
        def w(p):
            if sys.platform == "win32":
                return p
            return subprocess.run(["wslpath", "-w", p], capture_output=True, text=True).stdout.strip()
        r = subprocess.run([exe, "-nologo", "-y", "-m", "1", "-ft", "dds", "-f", "R16_UNORM",
                            "-o", w(work), w(path)], capture_output=True, text=True)
        if r.returncode != 0:
            return None
        stem = os.path.basename(path)[:-4]
        got = None
        for name in os.listdir(work):
            if name.lower() == (stem + ".dds").lower():
                got = os.path.join(work, name)
                break
        if not got:
            return None
        b = open(got, "rb").read()
        if b[:4] != b"DDS ":
            return None
        hsz = struct.unpack("<I", b[4:8])[0]
        hh = struct.unpack("<I", b[12:16])[0]
        ww = struct.unpack("<I", b[16:20])[0]
        off = 4 + hsz + (20 if b[84:88] == b"DX10" else 0)
        need = ww * hh * 2
        if len(b) - off < need:
            return None
        return np.frombuffer(b[off:off + need], dtype="<u2").reshape(hh, ww)
    except Exception:
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def load_png_gray(path):
    """16- or 8-bit PNG -> float64 in [0,1], first channel only.  Pillow reads 16-bit grey as
    mode 'I;16'/'I', which numpy sees correctly; RGB(A) sources take channel 0, because a
    displacement map's three channels are the same field.

    ⚠ THE SECOND RETURN VALUE IS THE FILE'S BIT DEPTH, NOT THE DECODER'S. Those are different
    numbers whenever Pillow cannot represent the file -- see _texconv_r16 -- and reporting the
    decoder's was this tool refusing 77 perfectly good sources.
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        a = np.array(im)
    if a.ndim == 3:
        a = a[..., 0]
    if a.dtype == np.uint16:
        return a.astype(np.float64) / 65535.0, 16
    if a.dtype in (np.int32, np.uint32):
        return a.astype(np.float64) / 65535.0, 16
    hdr = png_ihdr(path)
    if hdr is not None and hdr[2] == 16:
        full = _texconv_r16(path)
        if full is not None:
            return full.astype(np.float64) / 65535.0, 16
        # Fall through reporting 8 rather than handing back Pillow's truncation as 16-bit: a
        # refusal is recoverable, a map baked from half a file is silently worse than none.
    return a.astype(np.float64) / 255.0, 8


def box_down(a, n):
    """Area-average `a` down to n x n.  Only used for integer ratios, which is every real case
    here (4096 source -> a power-of-two target)."""
    h, w = a.shape
    assert h % n == 0 and w % n == 0, "non-integer downscale %dx%d -> %d" % (w, h, n)
    f = h // n
    return a.reshape(n, f, n, f).mean(axis=(1, 3))


def read_paramh_alpha(path):
    """Decode the height (alpha) plane of a DXT5 _paramh, via pbrsynth's file-block decoder."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pbrsynth"))
    import pbrsynth as P
    got, why = P._read_dxt5_alpha(path)
    if got is None:
        return None, why
    w, h, bw, bh, raw = got
    H = np.empty((h, w), dtype=np.float64)
    for r0 in range(0, bh, 64):
        r1 = min(bh, r0 + 64)
        dec = P.bc4_decode_file_blocks(raw[r0 * bw:r1 * bw])
        H[r0 * 4:r1 * 4, :] = P._unblocks(dec, (r1 - r0) * 4, w)
    return H, None


def write_dds_bc5(path, levels, exp, n0):
    """DX10 header + BC5_SNORM (DXGI 84), tightly packed mip chain -- the layout parseDds walks.

    dwReserved1 (offset 32, eleven uint32 the DDS spec leaves to the writer -- nvtt and others put
    their own signatures here) carries the magic and the range exponent.  The host reads it ONLY
    when the magic matches and otherwise assumes nothing, so an ordinary BC5 file from any other
    tool still loads; it simply has no range and is not usable as a derivative map."""
    # ⚠ n0 IS PASSED IN, NOT RECOVERED FROM levels[0]. The first version derived it as
    # `levels[0].shape[0] * 4` on the belief that a level was a 2-D block array; bc5_encode_snorm
    # returns a FLAT byte array, so that read the byte COUNT and stamped a 2048 map as 16777216
    # square. D3D12 refuses a texture that size, hands back a null resource, and The Forge's
    # updateTexture then calls GetDesc on it — a null-deref that kills the host with no log line
    # (see [[project_forge_addbuffer_null_deref_av]]). 101 files shipped with that header because
    # nothing re-read them; verify_dds_bc5 below is the fix for the CLASS, not just the typo.
    hdr = bytearray(148)
    hdr[0:4] = b"DDS "
    struct.pack_into("<I", hdr, 4, 124)                       # dwSize
    struct.pack_into("<I", hdr, 8, 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000)  # caps|h|w|pixelformat|mipcount|linearsize
    struct.pack_into("<I", hdr, 12, n0)                       # dwHeight
    struct.pack_into("<I", hdr, 16, n0)                       # dwWidth
    struct.pack_into("<I", hdr, 20, len(levels[0]))           # dwPitchOrLinearSize (top level bytes)
    struct.pack_into("<I", hdr, 28, len(levels))              # dwMipMapCount
    struct.pack_into("<I", hdr, 32, DERIV_MAGIC)              # dwReserved1[0] = 'MGED'
    struct.pack_into("<i", hdr, 36, exp)                      # dwReserved1[1] = range exponent
    struct.pack_into("<I", hdr, 76, 32)                       # ddspf dwSize
    struct.pack_into("<I", hdr, 80, 0x4)                      # DDPF_FOURCC
    hdr[84:88] = b"DX10"
    struct.pack_into("<I", hdr, 108, 0x1000 | 0x400000 | 0x8) # DDSCAPS_TEXTURE|MIPMAP|COMPLEX
    struct.pack_into("<I", hdr, 128, 84)                      # DXGI_FORMAT_BC5_SNORM
    struct.pack_into("<I", hdr, 132, 3)                       # D3D10_RESOURCE_DIMENSION_TEXTURE2D
    struct.pack_into("<I", hdr, 140, 1)                       # arraySize
    with open(path, "wb") as f:
        f.write(bytes(hdr))
        for lv in levels:
            f.write(lv.tobytes())


def verify_dds_bc5(path, n0, mips, exp):
    """Re-read what we just wrote and check the host can walk it. Returns None on success, else why.

    ⚠ THIS EXISTS BECAUSE THE TOOL SHIPPED 101 FILES WITH A GARBAGE HEADER AND REPORTED SUCCESS.
    Every other check in this tool looks at the INPUTS — is the source 16-bit, does it align with the
    shipped height, does the range clip — and not one of them looked at the OUTPUT. The GPU-side sign
    test did not catch it either, because the probe builds its own synthetic derivative map in C++:
    it tested the FORMAT CONTRACT while leaving this writer on no tested path at all.

    The walk below is deliberately the HOST'S, not a restatement of the writer's intent: dimensions
    and mip count out of the header, 16 bytes per 4x4 block per level (ddsTightMipBytes), and the sum
    has to be exactly the payload. A writer bug cannot agree with it by construction."""
    d = open(path, "rb").read()
    if len(d) < 148 or d[:4] != b"DDS ":
        return "not a DDS"
    h = struct.unpack_from("<I", d, 12)[0]
    w = struct.unpack_from("<I", d, 16)[0]
    m = struct.unpack_from("<I", d, 28)[0]
    if (w, h) != (n0, n0):
        return "header says %dx%d, baked %dx%d" % (w, h, n0, n0)
    if m != mips:
        return "header says %d mips, wrote %d" % (m, mips)
    if d[84:88] != b"DX10" or struct.unpack_from("<I", d, 128)[0] != 84:
        return "not DX10/BC5_SNORM"
    if struct.unpack_from("<I", d, 32)[0] != DERIV_MAGIC:
        return "derivative magic missing"
    if struct.unpack_from("<i", d, 36)[0] != exp:
        return "range exponent did not round-trip"
    need = 0
    for i in range(m):
        mw, mh = max(1, w >> i), max(1, h >> i)
        need += max(1, (mw + 3) // 4) * max(1, (mh + 3) // 4) * 16
    have = len(d) - 148
    if need != have:
        return "mip chain is %d bytes, file holds %d (delta %+d)" % (need, have, have - need)
    return None


# =======================================================================================
# Matching a shipped _paramh to its 16-bit source
# =======================================================================================


MAX_CANDS_PER_THUMB = 8


def _pick_sources(files, dn, limit=MAX_CANDS_PER_THUMB):
    """EVERY staging displacement whose name starts with thumbnail `dn`, exact stem first.

    ⚠⚠ NO NAMING RULE DECIDES THIS, AND BOTH OF THE ONES TRIED HERE LOST TEXTURES. The original
    rule took the LONGEST prefix match, which chose a different asset whenever one shared the
    prefix -- "Rock Wall" took rock_wall_09 while rock_wall sat in the same folder. Replacing it
    with the obvious correction, requiring the suffix to follow the name exactly, then broke SIX
    textures that had been baking at correlation 1.000: their export really was built from the
    longer-named sibling (tx_wood_blackwood's "Bark Willow" among them). Measured, not argued: the
    exact-stem rule gained 14 and lost 6.

    So the name does not decide, it only NARROWS. Hand back the whole family and let the
    correlation against the shipped height pick, which is the one test that cannot be wrong about
    this. Exact stem first so the common case is checked first and the cache stays warm; `_disp`
    before `_height` because the displacement is the 16-bit map and `_height` is often the 8-bit
    convenience export.
    """
    out = []
    for infix in ("_disp_", "_height_"):
        exact = sorted(f for f in files if f.casefold().startswith(dn + infix))
        loose = sorted((f for f in files
                        if f.casefold().startswith(dn) and infix in f.casefold() and f not in exact),
                       key=lambda f: (len(f), f))
        for f in exact + loose:
            if f not in out:
                out.append(f)
    return out[:limit]


def build_source_index(tm_dir):
    """texture base name -> EVERY staging displacement PNG its selected thumbnails point at.

    ⚠⚠ THIS USED TO RETURN ONE PATH -- the first thumbnail that had a staged file -- AND THAT IS A
    GUESS, NOT A LOOKUP. A texture can carry several selected_thumbnails, and the `_paramh` on disk
    was built from whichever one texturematcher last exported, which is not in general the first.
    Measured: of twenty textures whose bake was refused as "source does not match the shipped
    height", THIRTEEN matched one of their OTHER thumbnails at correlation 1.000, identity, no
    transform. The art was right there and the index looked at the wrong entry.

    So hand back all of them and let the correlation decide. That check already existed; it was
    being used only to REJECT a guess when it could have been used to MAKE the choice.

    The route is texturematcher's own: db.json maps `textures\\<base>_result.png` to a selected
    THUMBNAIL name, and the staging file is that name lower-cased with underscores plus a
    `_disp_` / `_height_` infix (find_file in modules/texture_operations.py).  Reproduced rather
    than imported because importing that module pulls cv2, which is not installed here -- and the
    mapping is two lines.  Every match it produces is verified against the shipped height anyway.
    """
    db_path = os.path.join(tm_dir, "db.json")
    staging = os.path.join(tm_dir, "staging")
    db = json.load(open(db_path, encoding="utf-8"))["textures"]
    files = os.listdir(staging)
    index, multi = {}, 0
    for key, val in db.items():
        base = os.path.basename(key.replace("\\", "/"))
        if base.lower().endswith("_result.png"):
            base = base[:-len("_result.png")]
        base = base.lower()
        thumbs = val.get("selected_thumbnails") or []
        if len(thumbs) > 1:
            multi += 1
        for th in thumbs:
            dn = th.get("name", "").lower().replace(" ", "_")
            if not dn:
                continue
            for got in _pick_sources(files, dn):
                index.setdefault(base, []).append(os.path.join(staging, got))
    return index, multi


def load_targets(manifest):
    """Track B's per-texture height target -- the size to bake at.  pbrsynth T5 measured that a
    derivative map at 512 beats a runtime central difference at 4096, so the derivative map does
    NOT need the height's resolution, and baking at the geometry's own texel density is what keeps
    it from re-spending the memory the resolution work just freed."""
    out = {}
    if not manifest or not os.path.exists(manifest):
        return out
    import csv
    for r in csv.DictReader(open(manifest, encoding="utf-8")):
        t = r.get("height_target_final") or r.get("height_target")
        if t:
            try:
                out[r["texture"].lower()] = int(t)
            except ValueError:
                pass
    return out


# The eight symmetries of the square. A source exported under a different axis convention lands on
# one of these EXACTLY -- a correlation of 1.000, not 0.9 -- so trying them costs one comparison each
# and either finds the art or proves the match is simply wrong. Anything needing an arbitrary angle
# is not a convention mismatch, it is a different texture.
DIHEDRAL = (
    ("identity",       lambda a: a),
    ("rot90",          lambda a: np.rot90(a, 1)),
    ("rot180",         lambda a: np.rot90(a, 2)),
    ("rot270",         lambda a: np.rot90(a, 3)),
    ("flipV",          lambda a: a[::-1, :]),
    ("flipH",          lambda a: a[:, ::-1]),
    ("transpose",      lambda a: a.T),
    ("anti-transpose", lambda a: np.rot90(a, 2).T),
)


def _corr(a, b):
    av, bv = a - a.mean(), b - b.mean()
    den = float(np.sqrt((av * av).sum() * (bv * bv).sum()))
    return float((av * bv).sum() / den) if den > 1e-20 else 0.0


_SMALL_CACHE = {}


def _source_small(path, n=256):
    """An n x n box-downsample of a candidate source, cached by path.

    Choosing among candidates needs only a low-resolution correlation, but getting there still costs
    a full decode (and for the 16-bit RGB files, a texconv round trip). Sources repeat heavily
    across textures -- quarry_wall_02 is a candidate for a dozen -- so caching the SMALL array turns
    a quadratic pile of decodes back into one per file, without holding 134 MB per 4096 map.

    Returns (small, bits, why) with small None when the candidate cannot be compared at all.
    """
    hit = _SMALL_CACHE.get(path)
    if hit is not None:
        return hit
    try:
        S, bits = load_png_gray(path)
    except Exception as e:
        out = (None, 0, "unreadable: %s" % e)
        _SMALL_CACHE[path] = out
        return out
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        out = (None, bits, "not square (%dx%d)" % (S.shape[1], S.shape[0]))
    elif bits < 16:
        out = (None, bits, "8-bit")
    else:
        m = min(S.shape[0], n)
        m = 1 << (int(m).bit_length() - 1)
        out = (box_down(S, m), bits, None)
    _SMALL_CACHE[path] = out
    return out


def bake_one(src_png, paramh_path, out_path, size, min_corr, dry):
    """Returns a dict of what happened; never raises on bad data."""
    cands = [src_png] if isinstance(src_png, str) else list(src_png)
    r = {"src": os.path.basename(cands[0]) if cands else "-"}
    H8, why = read_paramh_alpha(paramh_path)
    if H8 is None:
        r["skip"] = "paramh unreadable: %s" % why
        return r

    # ── WHICH candidate, decided by MEASUREMENT rather than by order ────────────────────────────
    # Every selected thumbnail is a candidate; the one that correlates with the shipped height is
    # the one that made it. Dihedral transforms are tried here too, so a rotated source competes on
    # the same footing as an upright one instead of losing to a wrong-but-upright sibling.
    picked, why8 = None, []
    for p in cands:
        small, bits, bad = _source_small(p)
        if small is None:
            why8.append("%s: %s" % (os.path.basename(p), bad))
            continue
        m = small.shape[0]
        hb = box_down(H8, m) if H8.shape[0] % m == 0 else None
        if hb is None:
            why8.append("%s: size mismatch" % os.path.basename(p))
            continue
        # ⚠ AND THE SOURCE MAY BE TILED INTO THE EXPORT. texturematcher records a per-thumbnail
        # `scale` in db.json and honours it by repeating the art: tx_cobblestone_01 carries
        # scale 2.0, and its shipped height is cobblestone_floor_13 tiled 2x2 -- correlation 1.000
        # tiled, 0.148 untiled, which reads as a wrong source if you never try it. Measured rather
        # than read from db.json: the field is absent on most entries, and a correlation that
        # confirms the tiling is worth more than a number that asserts it.
        scored = []
        for t in (1, 2, 3, 4):
            if m % t:
                continue
            cand_s = np.tile(box_down(small, m // t), (t, t)) if t > 1 else small
            for nm, fn in DIHEDRAL:
                scored.append((_corr(np.ascontiguousarray(fn(cand_s)), hb), nm, t))
        scored.sort(key=lambda x: -x[0])
        if picked is None or scored[0][0] > picked[0]:
            picked = (scored[0][0], scored[0][1], p, bits, scored[1][0], scored[0][2])
    if picked is None:
        r["bits"] = 8
        r["skip"] = ("no usable candidate: %s" % "; ".join(why8)) if why8 else "no source"
        return r
    corr0, tname, src_png, bits, runner, tile = picked
    r["src"], r["bits"], r["cands"] = os.path.basename(src_png), bits, len(cands)
    if tile > 1:
        r["tile"] = tile
    S, bits = load_png_gray(src_png)
    if bits < 16:
        # The entire argument for this tool is the 8 bits the _paramh threw away. An 8-bit source
        # has nothing more to give than the file we already ship, and pbrsynth measured that baking
        # from an 8-bit field is WORSE than not baking (deriv5q). Refuse rather than produce a map
        # that is confidently worse.
        r["skip"] = "source is 8-bit: baking from it measures worse than not baking (deriv5q)"
        return r
    if S.shape[0] != S.shape[1]:
        r["skip"] = "source not square (%dx%d)" % (S.shape[1], S.shape[0])
        return r

    if tile > 1:
        # Rebuild the field the export actually holds: the art at 1/tile of the size, repeated.
        # Do it BEFORE the derivative, for the same reason the rotation is applied first -- the
        # gradient then comes out in the frame and at the texel pitch the shader will sample.
        k = S.shape[0] // tile
        S = np.tile(box_down(S, k), (tile, tile))

    # ── ALIGNMENT, against the height this map has to line up with ──────────────────────────────
    n = min(S.shape[0], H8.shape[0], 512)
    n = 1 << (int(n).bit_length() - 1)
    try:
        a = box_down(S, n)
        b = box_down(H8, n)
    except AssertionError as e:
        r["skip"] = "cannot compare: %s" % e
        return r
    if tname != "identity":
        for nm, fn in DIHEDRAL:
            if nm == tname:
                S = np.ascontiguousarray(fn(S))
                a = box_down(S, n)
                break
        r["transform"], r["margin"] = tname, corr0 - runner
    corr = _corr(a, b)
    r["corr"] = corr
    if corr < min_corr:
        # ⚠ THREE DIFFERENT FAULTS USED TO LAND HERE AS ONE SENTENCE -- a wrong thumbnail, a ROTATED
        # source, and a genuinely different picture -- and they have three different owners. The
        # selection above now separates them, because it tries every candidate under every dihedral
        # transform before anything reaches this line. Measured over all 209 maps: THIRTEEN were the
        # wrong thumbnail (recovered at correlation 1.000, identity, from a thumbnail that was
        # already in db.json), exactly ONE was a real rotation (tx_gl_rock_01, rot180, 1.000 against
        # a runner-up of 0.524), and the remainder are a different image outright -- 0.05 to 0.40
        # under every transform and every cyclic shift, winner tied to runner-up inside noise.
        r["skip"] = ("correlation %.3f < %.2f — WRONG IMAGE: no candidate, no transform matches"
                     " (best %.3f via %s%s)"
                     % (corr, min_corr, corr0, tname,
                        ", %d candidates tried" % len(cands) if len(cands) > 1 else ""))
        return r

    # ── The bake ────────────────────────────────────────────────────────────────────────────────
    # DOWNSCALE FIRST, THEN DIFFERENTIATE — and the order is the whole reason the derivative is
    # taken here rather than at the source's 4096. dH/dtexel is a per-TEXEL quantity, so halving the
    # resolution doubles it: differentiating at 4096 and then averaging the derivative down would
    # store the 4096 map's slopes under the 1024 map's texel size, which is a factor of 4 wrong.
    # Averaging the HEIGHT and then differentiating is the quantity the shader will ask for.
    if size and size < S.shape[0]:
        S = box_down(S, size)
    N = S.shape[0]
    r["size"] = N

    # ⚠⚠ PER UV, NOT PER TEXEL, AND EVERY LEVEL DIFFERENTIATED AT ITS OWN SIZE.
    #
    # The first format stored dH/dTEXEL and built the chain by AVERAGING THE DERIVATIVE. Both halves
    # of that are wrong together, and they produced the seam play reported. Averaging a per-texel
    # derivative keeps it in LEVEL 0's texel units -- one texel of level L spans two of level L-1,
    # so the true per-texel slope there is 2x the average, not the average. The shader then
    # multiplied by THAT LEVEL's width, as if the value were already per-texel-of-L.
    #
    # Measured against ground truth (box-down the height to each level, differentiate there):
    # truth/reconstruction ran 0.99, 1.52, 2.53, 4.27, 7.75, 14.3, 28.3 -- a clean DOUBLING per
    # level. The relief was exact at level 0 and halved with every mip, and because the sampler
    # blends the stored value smoothly while `floor(log2(footprint))` steps the scale in a jump,
    # the loss arrived as a visible LINE at each mip boundary rather than as a gentle fade.
    #
    # Per-uv fixes both at once and deletes the reconstruction: dH/duv means the same thing at every
    # resolution, so a level describes itself, a trilinear blend between two levels is a blend
    # between two correct values, and the shader needs no width, no footprint and no LOD -- just the
    # range. There is nothing left to step.
    def _grad_peruv(H):
        """Wrapped central difference at FULL PRECISION, scaled to per-UV by the level's own width."""
        W = H.shape[0]
        return (0.5 * (np.roll(H, -1, axis=1) - np.roll(H, 1, axis=1)) * W,
                0.5 * (np.roll(H, -1, axis=0) - np.roll(H, 1, axis=0)) * W)

    chain, H = [], S
    while True:
        chain.append(_grad_peruv(H))
        if H.shape[0] <= 4:
            break
        H = box_down(H, H.shape[0] // 2)

    du, dv = chain[0]
    r["dmax"] = float(max(np.abs(du).max(), np.abs(dv).max()))
    fit = float(np.percentile(np.abs(np.concatenate([du.ravel(), dv.ravel()])), DERIV_PCT))
    if fit <= 0.0:
        r["skip"] = "the source is perfectly flat — a derivative map would be all zero"
        return r
    # Round the range UP to a power of two, so nothing below the percentile can clip and the host
    # can carry the range as an exponent rather than a float.
    exp = int(np.ceil(np.log2(fit)))
    exp = max(-128, min(127, exp))
    rng = float(2.0 ** exp)
    r["exp"], r["range"] = exp, rng
    # Clip measured over the WHOLE chain, not just the top level -- the range is fitted on level 0
    # and every other level has to live inside it too.
    r["clip"] = float(np.mean([float(np.mean((np.abs(u) > rng) | (np.abs(v) > rng)))
                               for u, v in chain]))
    levels = [bc5_encode_snorm(np.clip(u / rng, -1.0, 1.0), np.clip(v / rng, -1.0, 1.0))
              for u, v in chain]
    r["mips"] = len(levels)
    r["bytes"] = int(sum(len(l) for l in levels))
    if not dry:
        write_dds_bc5(out_path, levels, r["exp"], N)
        bad = verify_dds_bc5(out_path, N, len(levels), r["exp"])
        if bad:
            # Refuse to leave a file the host cannot load. A bake that reports success and writes a
            # nonsense header is worse than one that fails, because the failure surfaces as a crash
            # in someone else's code an hour later.
            os.remove(out_path)
            r["skip"] = "WROTE A BAD FILE and removed it: %s" % bad
            return r
        r["wrote"] = os.path.basename(out_path)
    return r


def main():
    ap = argparse.ArgumentParser(description="bake BC5 derivative maps from 16-bit displacement sources")
    ap.add_argument("--game-tex", default=DEFAULT_GAME_TEX, help="folder holding the shipped *_paramh.dds")
    ap.add_argument("--tm", default=DEFAULT_TM, help="texturematcher folder (db.json + staging/)")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Track B manifest, for per-texture sizes")
    ap.add_argument("--out", default=None, help="output folder (default: alongside the _paramh)")
    ap.add_argument("--size", type=int, default=0, help="force one size for every map (0 = the manifest's)")
    ap.add_argument("--cap", type=int, default=2048, help="never bake larger than this")
    ap.add_argument("--min-corr", type=float, default=0.90, help="alignment threshold against the shipped height")
    ap.add_argument("--only", default=None, help="substring filter on the texture base name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="match, verify and report; write nothing")
    a = ap.parse_args()

    print("pbrbake -- BC5 derivative maps: wrapped dH/dtexel, per-texture power-of-two range"
          " fitted to p%.1f" % DERIV_PCT)
    if not selftest():
        print("  !! encoder selftest FAILED — refusing to bake")
        return 2

    index, multi = build_source_index(a.tm)
    targets = load_targets(a.manifest)
    print("  db.json -> staging: %d texture bases matched a source candidate"
          " (%d have several thumbnails; the CORRELATION picks, not the order)" % (len(index), multi))
    print("  manifest: %d per-texture height targets" % len(targets))

    phs = sorted(f for f in os.listdir(a.game_tex) if f.lower().endswith("_paramh.dds"))
    if a.only:
        phs = [f for f in phs if a.only.lower() in f.lower()]
    if a.limit:
        phs = phs[:a.limit]
    print("  %d _paramh maps to consider%s\n" % (len(phs), "  (DRY RUN — nothing will be written)" if a.dry_run else ""))

    done, skipped, nosrc, bytes_out = 0, [], 0, 0
    for fn in phs:
        base = fn[:-len("_paramh.dds")].lower()
        src = index.get(base)
        if not src:
            nosrc += 1
            continue
        size = a.size or targets.get(base, 0) or a.cap
        size = min(size, a.cap)
        out = os.path.join(a.out or a.game_tex, base + "_paramd.dds")
        r = bake_one(src, os.path.join(a.game_tex, fn), out, size, a.min_corr, a.dry_run)
        if "skip" in r:
            skipped.append((base, r))
            print("  SKIP %-38s %s" % (base[:38], r["skip"]))
        else:
            done += 1
            bytes_out += r["bytes"]
            print("  bake %-38s %4d^2 x%2d mips  corr %.3f  range 2^%-4d (max |dH/texel| %.4f)"
                  "  clip %.3f%%  %6.2f MB%s"
                  % (base[:38], r["size"], r["mips"], r["corr"], r["exp"], r["dmax"],
                     100.0 * r["clip"], r["bytes"] / 1048576.0,
                     (("   ROTATED %s (margin %.3f)" % (r["transform"], r.get("margin", 0.0))
                       if r.get("transform") else "")
                      + ("   TILED %dx%d" % (r["tile"], r["tile"]) if r.get("tile") else "")
                      + ("   [%d candidates]" % r["cands"] if r.get("cands", 1) > 1 else ""))))

    print("\n  baked %d, skipped %d, no 16-bit source %d, of %d _paramh maps" % (done, len(skipped), nosrc, len(phs)))
    print("  total derivative-map bytes: %.1f MB" % (bytes_out / 1048576.0))
    if nosrc:
        print("  ⚠ the %d maps with no source keep the runtime arms (pbrGradMode 0 + pbrGradRadius)."
              " Baking from their shipped 8-bit height measures WORSE than not baking (pbrsynth deriv5q)." % nosrc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
