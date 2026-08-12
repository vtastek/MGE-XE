#!/usr/bin/env python3
"""Census of NiAlphaProperty test state across every MW mesh (loose + BSA).

THE QUESTION: our wire encodes "no alpha test" as alphaRef == 0 (packTexAlpha quantises
alphaRef*255 and every consumer treats byte 0 as "test off"). MW's own encoding has a
TEST_ENABLE bit that is independent of the reference, and `GREATER 0` — test on, ref 0,
"discard only the fully transparent texels" — is a completely ordinary authoring idiom.
Those two meet as: test ON + ref 0 arrives at the host as test OFF.

Counts how many shapes hit that, split by whether they also blend, because a blended one
lands on the sorted-alpha path (where it still reads as translucent) while an unblended one
lands on the OPAQUE path and renders as a solid card.

Usage: nif-alpharef-census.py
"""
import collections
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

_dump = import_module("nif-block-dump")

TESTFUNC = {0: "ALWAYS", 1: "LESS", 2: "EQUAL", 3: "LEQUAL",
            4: "GREATER", 5: "NOTEQUAL", 6: "GEQUAL", 7: "NEVER"}


def alpha_props(blob):
    """-> [(flags, ref)] for every NiAlphaProperty in the file."""
    out = []
    for t, s, e in _dump.blocks(blob):
        if t != "NiAlphaProperty" or e - s < 3:
            continue
        fl, thr = struct.unpack_from("<HB", blob, e - 3)
        out.append((fl, thr))
    return out


def main():
    data = _dump.DATA
    seen = set()
    sources = []

    meshes = os.path.join(data, "Meshes")
    for root, _d, files in os.walk(meshes):
        for fn in files:
            if fn.lower().endswith(".nif"):
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, meshes).replace("\\", "/").lower()
                seen.add(rel)
                with open(p, "rb") as f:
                    sources.append(("loose", rel, f.read()))
    loose_count = len(sources)

    for b in sorted(os.listdir(data)):
        if not b.lower().endswith(".bsa"):
            continue
        for name, blob in _dump.iter_bsa_nifs(os.path.join(data, b)):
            rel = name.replace("\\", "/").lower()
            if rel.startswith("meshes/"):
                rel = rel[len("meshes/"):]
            if rel in seen:          # a loose file WINS over the archive
                continue
            seen.add(rel)
            sources.append((b, rel, blob))

    files = props = 0
    kinds = collections.Counter()
    funcs = collections.Counter()
    broken_opaque = []
    broken_blend = []

    for src, rel, blob in sources:
        try:
            aps = alpha_props(blob)
        except Exception:
            continue
        if aps:
            files += 1
        for fl, ref in aps:
            props += 1
            blend = bool(fl & 1)
            test = bool(fl & (1 << 9))
            func = (fl >> 10) & 7
            kinds[(blend, test, ref == 0)] += 1
            if test:
                funcs[TESTFUNC.get(func, func)] += 1
            if test and ref == 0:
                (broken_blend if blend else broken_opaque).append((src, rel, fl, func))

    print(f"scanned {len(sources)} unique meshes ({loose_count} loose override the archives)")
    print(f"{files} carry a NiAlphaProperty; {props} properties total\n")

    print("  blend  test  ref==0   count")
    for (blend, test, z), n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {int(blend):5d}  {int(test):4d}  {int(z):6d}   {n:6d}")

    print("\ntest function, where TEST_ENABLE is set:")
    for f, n in funcs.most_common():
        print(f"  {f:9s} {n:6d}")

    tot = len(broken_opaque) + len(broken_blend)
    print(f"\n>>> TEST ENABLED WITH REF 0 (arrives as 'no test'): {tot} properties")
    print(f"      unblended -> OPAQUE path, renders as a solid card: {len(broken_opaque)}")
    print(f"      blended   -> sorted-alpha path, still translucent: {len(broken_blend)}")
    for label, lst in (("OPAQUE", broken_opaque), ("BLENDED", broken_blend)):
        print(f"\n    first 25 {label}:")
        for src, rel, fl, func in lst[:25]:
            print(f"      [{src:>16s}] {rel:52s} flags=0x{fl:04X} {TESTFUNC.get(func, func)}")


if __name__ == "__main__":
    main()
