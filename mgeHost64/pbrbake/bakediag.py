#!/usr/bin/env python3
"""
bakediag -- WHY a `_paramh` has no baked derivative map, one line per texture.

pbrbake refuses a bake for two reasons and reports both as a sentence. That is right for a bake
log and useless for fixing the inputs, because the two reasons have completely different owners:

  * "source is 8-bit"  -> the SOURCE ART is wrong. Nothing in the bake can fix it; the 16-bit
    displacement has to be fetched. But 8-bit is not one thing: the file on disk may be an 8-bit
    PNG because Poly Haven only published an 8-bit map for that asset, or because the download
    took a `_height` where a 16-bit `_disp` sat beside it, or because something re-saved it. Those
    need different fixes, so this reports the PNG's own IHDR and every candidate in staging.

  * "correlation 0.004" -> the MATCH is wrong, and a near-zero correlation says the two images
    have nothing to do with each other rather than that they are slightly misaligned. A ROTATED or
    FLIPPED source scores exactly this, and so does a source that tiles at a different phase. So
    every failure is re-tested under all EIGHT dihedral transforms and under the best cyclic shift
    (FFT phase correlation), and the transform that recovers it is named.

⚠ THE DIHEDRAL TEST IS NOT FREE OF FALSE POSITIVES, so it is reported with the runner-up. On a
texture with 4-fold symmetry several transforms score alike and the winner means nothing; the
margin over the second-best is what says a transform was actually found.

Reads only. Writes nothing. Run:  python3 bakediag.py [--only substr] [--limit N]
"""
import argparse
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pbrbake as PB


def png_ihdr(path):
    """(width, height, bitdepth, colourtype) straight from the file, not from a decoder."""
    try:
        with open(path, "rb") as f:
            head = f.read(33)
        if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
            return None
        w, h = struct.unpack(">II", head[16:24])
        return w, h, head[24], head[25]
    except Exception:
        return None


COLOURTYPE = {0: "grey", 2: "rgb", 3: "palette", 4: "grey+a", 6: "rgba"}

# The eight symmetries of the square. A source that came out of a different tool's axis convention
# lands on one of these exactly; anything else is a wrong match, not a transform.
DIHEDRAL = [
    ("identity",      lambda a: a),
    ("rot90",         lambda a: np.rot90(a, 1)),
    ("rot180",        lambda a: np.rot90(a, 2)),
    ("rot270",        lambda a: np.rot90(a, 3)),
    ("flipV",         lambda a: a[::-1, :]),
    ("flipH",         lambda a: a[:, ::-1]),
    ("transpose",     lambda a: a.T),
    ("anti-transpose", lambda a: np.rot90(a, 2).T),
]


def corr(a, b):
    av, bv = a - a.mean(), b - b.mean()
    den = float(np.sqrt((av * av).sum() * (bv * bv).sum()))
    return float((av * bv).sum() / den) if den > 1e-20 else 0.0


def best_shift_corr(a, b):
    """Highest correlation over every CYCLIC shift, via FFT cross-correlation.

    These textures tile, so a source that is the right art at the wrong phase correlates at zero
    under a direct comparison and at ~1 here. Normalising by the zero-mean energies makes the peak
    a real correlation coefficient rather than an unbounded dot product.
    """
    av, bv = a - a.mean(), b - b.mean()
    den = float(np.sqrt((av * av).sum() * (bv * bv).sum()))
    if den <= 1e-20:
        return 0.0, (0, 0)
    c = np.fft.irfft2(np.fft.rfft2(av) * np.conj(np.fft.rfft2(bv)), s=av.shape) / den
    k = int(np.argmax(c))
    dy, dx = divmod(k, c.shape[1])
    if dy > c.shape[0] // 2: dy -= c.shape[0]
    if dx > c.shape[1] // 2: dx -= c.shape[1]
    return float(c.max()), (dy, dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-tex", default=PB.DEFAULT_GAME_TEX)
    ap.add_argument("--tm", default=PB.DEFAULT_TM)
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    index, _ = PB.build_source_index(a.tm)
    staging = os.path.join(a.tm, "staging")
    files = os.listdir(staging)

    names = sorted(f for f in os.listdir(a.game_tex) if f.lower().endswith("_paramh.dds"))
    rows = []
    for i, f in enumerate(names):
        base = f[: -len("_paramh.dds")]
        if a.only and a.only.lower() not in base.lower():
            continue
        if a.limit and len(rows) >= a.limit:
            break
        src = index.get(base.lower())
        row = {"base": base, "src": src}
        if not src:
            row["verdict"] = "NO SOURCE  (no db.json thumbnail -> staging match)"
            rows.append(row); continue

        ih = png_ihdr(src)
        row["ihdr"] = ih
        if ih and ih[2] < 16:
            # Is a 16-bit file for the SAME asset sitting beside it? That is the difference between
            # "Poly Haven has no 16-bit map" and "we downloaded the wrong one of two".
            stem = os.path.basename(src).lower()
            for infix in ("_disp_", "_height_"):
                if infix in stem:
                    pre = stem.split(infix)[0]
                    break
            else:
                pre = stem[:12]
            alts = []
            for g in files:
                gl = g.lower()
                if gl.startswith(pre) and gl.endswith(".png") and g != os.path.basename(src):
                    gh = png_ihdr(os.path.join(staging, g))
                    if gh and gh[2] >= 16:
                        alts.append((g, gh))
            row["alts"] = alts
            row["verdict"] = "8-BIT SOURCE" + ("  *** a 16-bit sibling EXISTS ***" if alts else "  (no 16-bit file staged)")
            rows.append(row); continue

        H8, why = PB.read_paramh_alpha(os.path.join(a.game_tex, f))
        if H8 is None:
            row["verdict"] = "paramh unreadable: %s" % why
            rows.append(row); continue
        try:
            S, bits = PB.load_png_gray(src)
        except Exception as e:
            row["verdict"] = "source unreadable: %s" % e
            rows.append(row); continue
        if S.shape[0] != S.shape[1]:
            row["verdict"] = "SOURCE NOT SQUARE %dx%d" % (S.shape[1], S.shape[0])
            rows.append(row); continue

        n = min(S.shape[0], H8.shape[0], 256)
        n = 1 << (int(n).bit_length() - 1)
        try:
            sa, hb = PB.box_down(S, n), PB.box_down(H8, n)
        except AssertionError as e:
            row["verdict"] = "cannot compare: %s" % e
            rows.append(row); continue

        scores = []
        for name, fn in DIHEDRAL:
            t = np.ascontiguousarray(fn(sa))
            direct = corr(t, hb)
            shifted, off = best_shift_corr(t, hb)
            scores.append((max(direct, shifted), name, direct, shifted, off))
        scores.sort(reverse=True)
        row["scores"] = scores
        best, bname, bdirect, bshift, boff = scores[0]
        runner = scores[1][0]
        row["verdict"] = ("OK (identity)" if bname == "identity" and bdirect >= 0.90 else
                          "RECOVERED by %s" % bname if best >= 0.90 else
                          "NO TRANSFORM HELPS (best %.3f)" % best)
        row["best"] = (best, bname, bdirect, bshift, boff, runner)
        rows.append(row)

    # ── report ─────────────────────────────────────────────────────────────────────────────────
    import collections
    tally = collections.Counter()
    print("%-38s %s" % ("TEXTURE", "VERDICT"))
    print("-" * 110)
    for r in rows:
        v = r["verdict"]
        tally[v.split("(")[0].split("***")[0].strip()] += 1
        extra = ""
        if "best" in r:
            best, bname, bdirect, bshift, boff, runner = r["best"]
            if not v.startswith("OK"):
                extra = ("   best %.3f via %-14s (direct %.3f, shifted %.3f at %s) runner-up %.3f"
                         % (best, bname, bdirect, bshift, boff, runner))
        if r.get("alts"):
            extra = "   16-bit sibling: %s" % ", ".join(g for g, _ in r["alts"][:2])
        if r.get("ihdr"):
            w, h, bd, ct = r["ihdr"]
            extra += "   [%dx%d %d-bit %s]" % (w, h, bd, COLOURTYPE.get(ct, ct))
        if not v.startswith("OK"):
            print("%-38s %s%s" % (r["base"], v, extra))
    print("-" * 110)
    for k, v in tally.most_common():
        print("  %-52s %d" % (k, v))
    print("  %-52s %d" % ("TOTAL", len(rows)))


if __name__ == "__main__":
    main()
