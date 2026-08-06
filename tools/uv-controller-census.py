#!/usr/bin/env python3
"""Census of every NiUVController in the Morrowind data, for the heart_akulakhan finding.

The question: heart_akulakhan.nif's forcefield UV controller is authored cycle=CLAMP, so the host
(which free-runs t = simTime and clamps) freezes it, while MW loops it. Before "treat CLAMP as
LOOP" can be considered, we need the POPULATION: how many UV controllers are CLAMP, and are their
tracks SEAMLESS (net offset a whole number of texture wraps, so restarting is invisible)?

A seamless CLAMP track is proof the artist intended a loop and relied on MW's animation manager
restarting the controller. A NON-seamless CLAMP track would be a genuine play-once scroll, which
looping would break -- so the count of those is the risk.

Parses NIF 4.0.0.2 by the same block-token framing as nif-controller-census.py; a NiUVController's
uvData link is the last i32 of its body, and blocks are numbered in file order.
"""
import os, re, struct, sys
from collections import Counter, defaultdict

DATA = os.environ.get("MW_DATA", "/mnt/c/mgem/morrowind64/Data Files")
_TOKEN_RE = re.compile(rb"([\x03-\x3c])\x00\x00\x00([A-Z][A-Za-z0-9_]{2,59})")
_NON_NI = {"RootCollisionNode", "AvoidNode", "BSMirroredNode", "BoundingBox", "TES3ObjectEXTRA"}
CYCLE = {0: "LOOP", 1: "REVERSE", 2: "CLAMP"}


def iter_bsa_nifs(path):
    with open(path, "rb") as f:
        d = f.read()
    ver, hash_off, n = struct.unpack_from("<III", d, 0)
    if ver != 0x100:
        return
    rec, noff = 12, 12 + n * 8
    ntab, dbase = noff + n * 4, 12 + hash_off + n * 8
    for i in range(n):
        size, off = struct.unpack_from("<II", d, rec + i * 8)
        s = ntab + struct.unpack_from("<I", d, noff + i * 4)[0]
        name = d[s:d.index(b"\x00", s)].decode("cp1252", "replace")
        if name.lower().endswith(".nif"):
            yield name, d[dbase + off:dbase + off + size]


def iter_all_nifs():
    meshes = os.path.join(DATA, "Meshes")
    seen = set()
    for root, _d, files in os.walk(meshes):
        for fn in files:
            if fn.lower().endswith(".nif"):
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, meshes).replace("\\", "/").lower()
                seen.add(rel)
                with open(p, "rb") as f:
                    yield rel, f.read()
    for b in sorted(os.listdir(DATA)):
        if b.lower().endswith(".bsa"):
            for name, blob in iter_bsa_nifs(os.path.join(DATA, b)):
                rel = name.replace("\\", "/").lower()
                rel = rel[len("meshes/"):] if rel.startswith("meshes/") else rel
                if rel not in seen:          # a loose file overrides the BSA copy
                    seen.add(rel)
                    yield rel, blob


def blocks(blob):
    toks = []
    for m in _TOKEN_RE.finditer(blob):
        nm = m.group(2).decode("ascii")
        if m.group(1)[0] == len(nm) and (nm.startswith("Ni") or nm in _NON_NI):
            toks.append((m.start(), m.end(), nm))
    return [(nm, e, toks[i + 1][0] if i + 1 < len(toks) else len(blob))
            for i, (s, e, nm) in enumerate(toks)]


def read_group(blob, p, end):
    """One NiUVData KeyGroup<float> -> (keys, new_p) or (None, end) if unparsable."""
    if p + 4 > end:
        return None, end
    n = struct.unpack_from("<I", blob, p)[0]
    p += 4
    if n == 0:
        return [], p
    if p + 4 > end:
        return None, end
    interp = struct.unpack_from("<I", blob, p)[0]
    p += 4
    stride = {1: 8, 2: 16, 3: 20}.get(interp)
    if stride is None or p + n * stride > end:
        return None, end
    ks = [struct.unpack_from("<2f", blob, p + i * stride) for i in range(n)]
    return ks, p + n * stride


def main():
    ctl = Counter()
    tiling = 0
    clamp_seamless, clamp_ragged = [], []
    loop_seamless, loop_ragged = 0, 0
    per_file = defaultdict(list)
    nfiles = 0

    for name, blob in iter_all_nifs():
        if not blob.startswith(b"NetImmerse File Format"):
            continue
        nfiles += 1
        bl = blocks(blob)
        for i, (t, s, e) in enumerate(bl):
            if t != "NiUVController":
                continue
            if e - s < 32:
                continue
            flags = struct.unpack_from("<H", blob, s + 4)[0]
            cyc = (flags & 6) >> 1
            active = (flags >> 3) & 1
            link = struct.unpack_from("<i", blob, s + 28)[0]
            ctl[(CYCLE.get(cyc, cyc), bool(active))] += 1
            if not (0 <= link < len(bl)) or bl[link][0] != "NiUVData":
                continue
            ds, de = bl[link][1], bl[link][2]
            p = ds
            groups = []
            for _ in range(4):
                g, p = read_group(blob, p, de)
                groups.append(g)
            u, v, ut, vt = groups
            if (ut and len(ut) > 1) or (vt and len(vt) > 1):
                tiling += 1
                continue
            # Seamless == the net offset over the track is a whole number of texture wraps, so a
            # restart lands on an identical image. That is the artist relying on MW's manager.
            nets = [g[-1][1] - g[0][1] for g in (u, v) if g and len(g) > 1]
            if not nets:
                continue
            # Seamless == every component's net offset is a WHOLE number of texture wraps. Zero
            # counts: a track that returns to its start value restarts continuously, so it is the
            # most seamless case of all. (Requiring |net| >= 1 here mis-filed 18 of 21 "ragged"
            # CLAMPs -- the daylight panels, the sky window shimmers and puz_basin all sit at
            # net 0 on at least one axis and integer on the other.)
            seamless = all(abs(d - round(d)) < 1e-3 for d in nets)
            if cyc == 2:
                (clamp_seamless if seamless else clamp_ragged).append((name, nets))
            elif cyc == 0:
                if seamless:
                    loop_seamless += 1
                else:
                    loop_ragged += 1
            per_file[name].append((CYCLE.get(cyc, cyc), seamless, nets))

    print(f"=== {nfiles} NIFs scanned under {DATA} ===\n")
    print("NiUVController by (cycleType, active):")
    for k, n in sorted(ctl.items(), key=lambda kv: -kv[1]):
        print(f"  {str(k):24s} x{n}")
    total = sum(ctl.values())
    print(f"  TOTAL {total}\n")
    print(f"animated TILING (client declines the takeover, engine keeps driving): {tiling}\n")
    print("OFFSET-only tracks, by cycle type and whether the net offset is a WHOLE number of wraps:")
    print(f"  CLAMP  seamless (loop was intended, engine restarts it): {len(clamp_seamless)}")
    print(f"  CLAMP  ragged   (a genuine play-once scroll would break): {len(clamp_ragged)}")
    print(f"  LOOP   seamless: {loop_seamless}")
    print(f"  LOOP   ragged  : {loop_ragged}")

    if clamp_ragged:
        print("\n--- every ragged CLAMP (the risk set for treating CLAMP as LOOP) ---")
        for nm, nets in sorted(clamp_ragged)[:40]:
            print(f"  {nm:58s} net={['%+.3f' % d for d in nets]}")
        if len(clamp_ragged) > 40:
            print(f"  ... {len(clamp_ragged) - 40} more")
    print("\n--- a sample of seamless CLAMPs ---")
    for nm, nets in sorted(clamp_seamless)[:15]:
        print(f"  {nm:58s} net={['%+.3f' % d for d in nets]}")


if __name__ == "__main__":
    main()
