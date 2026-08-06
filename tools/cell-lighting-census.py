#!/usr/bin/env python3
"""Census of AUTHORED per-cell lighting in the ESM masters (tasks/forge-postprocess.md step 2 aside).

Morrowind gives an interior exactly three colour slots — ambient, "sunlight", fog — plus a fog
density. There is no sun in an interior, so the sunlight slot is a PHANTOM: a fill or key light
standing in for sources the engine could not afford to make matter. This tool reports what the
vanilla artists actually put in those slots, which is the input side of the linear/PBR/GI work
(the output side is the host's [forge-hb] apl: instrument).

  python3 tools/cell-lighting-census.py                 # full census
  python3 tools/cell-lighting-census.py --find akulakhan # look up cells by name substring

TES3 FORMAT, AND THE TWO TRAPS THAT COST A WRONG ANSWER
-------------------------------------------------------
Record:    name[4], size u32, header u32, flags u32, then `size` bytes of subrecords.
Subrecord: name[4], size u32, data[size].

A CELL record carries its ENTIRE REFERENCE LIST INLINE — one FRMR plus a NAME plus a DATA for every
object placed in the cell. So a naive "last subrecord wins" scan gets both of these wrong:

  * NAME — the last one is an object id (`ingred_emerald_01`), not the cell name. Take the FIRST.
  * DATA — a reference's DATA is 24 bytes of position/rotation; the cell's is 12 bytes of
    (flags, gridX, gridY). Taking the last one puts float bits in `flags`, so an `interior` test
    reads a coordinate's low bit. This silently changed the POPULATION: filtering on the corrupt
    flag gave 594 cells instead of 1335, dropping the dark cells hardest and hiding Akulakhan's
    Chamber entirely. Take the FIRST 12-byte DATA — and prefer AMBI presence as the interior test,
    since only interiors carry AMBI at all.

Colours are 4 bytes, R,G,B,unused.
"""
import struct, os, sys
from collections import Counter, defaultdict

DATA_DIR = os.environ.get("MW_DATA", "/mnt/c/mgem/morrowind64/Data Files")
MASTERS = ["Morrowind.esm", "Tribunal.esm", "Bloodmoon.esm"]

# The Construction Set's untouched default. Present in test cells (`ken's test hole`, `ToddTest`,
# `Mark's Vampire Test Cell`) and a few unedited shells. It is NOT authorship and it sits above every
# real value, so it must be excluded before ranking or it buries the actual brightest cell.
CS_DEFAULT_SUN = (242, 217, 217)

CAVE_HINTS = ("cave", "grotto", "cavern", "mine", "lava", "sewer", "den", "burial", "tomb")


def records(path):
    with open(path, "rb") as f:
        blob = f.read()
    p, n = 0, len(blob)
    while p + 16 <= n:
        name = blob[p:p + 4]
        size = struct.unpack_from("<I", blob, p + 4)[0]
        yield name, blob[p + 16:p + 16 + size]
        p += 16 + size


def subrecords(body):
    p, n = 0, len(body)
    while p + 8 <= n:
        name = body[p:p + 4]
        size = struct.unpack_from("<I", body, p + 4)[0]
        yield name, body[p + 8:p + 8 + size]
        p += 8 + size


def luma(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def load(files):
    """-> [(file, name, flags, ambient, sun, fog, fogDensity)] for every cell carrying AMBI."""
    out = []
    for fn in files:
        path = os.path.join(DATA_DIR, fn)
        if not os.path.exists(path):
            print(f"  (missing {fn})", file=sys.stderr)
            continue
        for rname, body in records(path):
            if rname != b"CELL":
                continue
            cname, flags, ambi, gotData = None, 0, None, False
            for sname, sdata in subrecords(body):
                if sname == b"NAME" and cname is None:          # FIRST — see module docstring
                    cname = sdata.split(b"\0")[0].decode("cp1252", "replace")
                elif sname == b"DATA" and len(sdata) == 12 and not gotData:
                    flags = struct.unpack_from("<I", sdata, 0)[0]
                    gotData = True
                elif sname == b"AMBI" and len(sdata) >= 16:
                    ambi = (tuple(sdata[0:3]), tuple(sdata[4:7]), tuple(sdata[8:11]),
                            struct.unpack_from("<f", sdata, 12)[0])
            if ambi:
                out.append((fn, cname or "", flags, *ambi))
    return out


def find(pats):
    cells = load(MASTERS + ["Patch for Purists.esm"])
    pats = [p.lower() for p in pats]
    for fn, name, flags, a, s, f, fd in cells:
        if not any(p in name.lower() for p in pats):
            continue
        print(f"[{fn}] {'INT' if flags & 1 else 'ext'} {name}")
        print(f"      ambient ={str(a):16s} luma={luma(a):6.1f}")
        print(f"      sunlight={str(s):16s} luma={luma(s):6.1f}")
        print(f"      fog     ={str(f):16s} density={fd:.3f}")


def census():
    cells = load(MASTERS)
    nflag = sum(1 for c in cells if c[2] & 1)
    print(f"=== cells with AMBI: {len(cells)} across {', '.join(MASTERS)} "
          f"({nflag} also carry the interior flag) ===\n")

    amb = Counter(c[3] for c in cells)
    sun = Counter(c[4] for c in cells)
    print(f"DISTINCT ambient colours: {len(amb)}     DISTINCT sunlight colours: {len(sun)}\n")

    print("--- top 12 ambient values ---")
    for v, n in amb.most_common(12):
        print(f"  {str(v):18s} x{n:5d}  {100.0*n/len(cells):5.1f}%   luma={luma(v):6.1f}")
    print("\n--- top 8 sunlight values (the phantom) ---")
    for v, n in sun.most_common(8):
        print(f"  {str(v):18s} x{n:5d}  {100.0*n/len(cells):5.1f}%   luma={luma(v):6.1f}")

    groups = defaultdict(list)
    for c in cells:
        nm = c[1].lower()
        groups["cave-ish" if any(h in nm for h in CAVE_HINTS) else "built"].append(c)
    print("\n--- ambient luma by group ---")
    for g, lst in sorted(groups.items()):
        ls = sorted(luma(c[3]) for c in lst)
        pct = lambda q: ls[min(len(ls) - 1, int(q * len(ls)))]
        print(f"  {g:9s} n={len(lst):5d}  min={ls[0]:5.1f}  p25={pct(.25):5.1f}  "
              f"median={pct(.5):5.1f}  p75={pct(.75):5.1f}  max={ls[-1]:5.1f}")

    # The DDGI target list: cells where the phantom sun is a KEY light rather than domestic fill.
    nd = [c for c in cells if c[4] != CS_DEFAULT_SUN]
    print(f"\n(excluded {len(cells) - len(nd)} cells at the CS default sun {CS_DEFAULT_SUN})")
    print("--- brightest DELIBERATELY authored sunlights ---")
    for c in sorted(nd, key=lambda c: -luma(c[4]))[:10]:
        print(f"  sun luma={luma(c[4]):6.1f} sun={str(c[4]):16s} "
              f"amb={str(c[3]):15s}({luma(c[3]):5.1f})  {c[1][:44]}")
    key = [c for c in nd if luma(c[4]) > 2 * luma(c[3]) and luma(c[4]) > 60]
    print(f"\nkey-light shape (sun > 2x ambient AND luma > 60): {len(key)} cells "
          f"({100.0*len(key)/len(cells):.1f}%) — the DDGI validation set")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--find":
        find(sys.argv[2:])
    else:
        census()
