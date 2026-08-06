#!/usr/bin/env python3
"""List the placed references of a named cell, resolved to their NIF models.

Answers "is in_lava_1024_01 actually in Akulakhan's Chamber?" -- which decides whether that mesh
reaches the host by the interior NEAR path (cached mesh + gUVAnim) or only by the DISTANT-statics
path (statics.vert.fsl's hardcoded 0.08 V-scroll).

A CELL record carries its reference list inline as repeated FRMR (u32 index) + NAME (object id)
[+ optional per-ref subrecords]. Object id -> model is resolved from the STAT/ACTI/DOOR/LIGH/etc
records' NAME + MODL pair. Both are matched case-insensitively; later masters override earlier.

Usage: cellrefs.py "akulakhan"            # refs of every cell whose name matches
       cellrefs.py --model in_lava        # which cells place a model matching this
"""
import os, struct, sys
from collections import Counter, defaultdict

DATA = os.environ.get("MW_DATA", "/mnt/c/mgem/morrowind64/Data Files")
MASTERS = ["Morrowind.esm", "Tribunal.esm", "Bloodmoon.esm"]
# Record types that own a placeable NIF via MODL.
MODELED = {b"STAT", b"ACTI", b"DOOR", b"LIGH", b"CONT", b"MISC", b"WEAP", b"ARMO", b"BOOK",
           b"CLOT", b"INGR", b"APPA", b"LOCK", b"PROB", b"REPA", b"ALCH", b"NPC_", b"CREA"}


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
        size = struct.unpack_from("<I", body, p + 8 - 4)[0]
        yield name, body[p + 8:p + 8 + size]
        p += 8 + size


def zstr(b):
    return b.split(b"\0")[0].decode("cp1252", "replace")


def load():
    models = {}                      # lowercase object id -> model path
    cells = []                       # (cellName, isInterior, [objectIds])
    for fn in MASTERS:
        path = os.path.join(DATA, fn)
        if not os.path.exists(path):
            continue
        for rname, body in records(path):
            if rname in MODELED:
                oid = mdl = None
                for sname, sdata in subrecords(body):
                    if sname == b"NAME" and oid is None:
                        oid = zstr(sdata)
                    elif sname == b"MODL" and mdl is None:
                        mdl = zstr(sdata)
                if oid and mdl:
                    models[oid.lower()] = mdl
            elif rname == b"CELL":
                cname, interior, refs, gotData = None, False, [], False
                for sname, sdata in subrecords(body):
                    if sname == b"NAME" and cname is None:
                        cname = zstr(sdata)
                    elif sname == b"DATA" and len(sdata) == 12 and not gotData:
                        interior = bool(struct.unpack_from("<I", sdata, 0)[0] & 1)
                        gotData = True
                    elif sname == b"AMBI":
                        interior = True          # only interiors carry AMBI
                    elif sname == b"NAME" and cname is not None:
                        refs.append(zstr(sdata))
                if cname:
                    cells.append((cname, interior, refs))
    return models, cells


def main():
    models, cells = load()
    if len(sys.argv) > 2 and sys.argv[1] == "--model":
        pat = sys.argv[2].lower()
        hits = defaultdict(int)
        for cname, interior, refs in cells:
            for r in refs:
                m = models.get(r.lower())
                if m and pat in m.lower():
                    hits[(cname, "INT" if interior else "ext", m)] += 1
        for (cname, kind, m), n in sorted(hits.items(), key=lambda kv: -kv[1]):
            print(f"  {kind} {cname[:46]:46s} x{n:3d}  {m}")
        print(f"\n{len(hits)} (cell, model) pairs place a model matching '{pat}'")
        return

    pat = (sys.argv[1] if len(sys.argv) > 1 else "akulakhan").lower()
    for cname, interior, refs in cells:
        if pat not in cname.lower():
            continue
        print(f"\n=== {'INT' if interior else 'ext'} {cname} — {len(refs)} refs ===")
        c = Counter()
        for r in refs:
            c[(r, models.get(r.lower(), "(no model)"))] += 1
        for (oid, mdl), n in sorted(c.items(), key=lambda kv: kv[0][1].lower()):
            print(f"  x{n:3d}  {oid[:34]:34s} {mdl}")


if __name__ == "__main__":
    main()
