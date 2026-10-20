#!/usr/bin/env python3
"""Point meshes/vt/spheresm.nif's textures at textures\\vt\\<name>.dds.

MW raised `tex not found "vt\\<name>.tga"` + `Texture Load Error!` for every sphere texture: the NIF's
paths lack the `textures\\` prefix. Proven on the grid first (2026-10-18): make-pbrgrid.py's
`vt\\pbrgrid\\x.dds` -> `textures\\vt\\pbrgrid\\x.dds` took its 44 Warnings.txt lines to 0 with the
cell loaded, nothing else changed. The .tga originals (RLE, which MW cannot read) were moved out
earlier, so the paths name the .dds directly.

Rewrites ONLY the NiSourceTexture file names, through make-pbrgrid.py's writer, which is first
required to rebuild the untouched file bit-identically. Backs the original up beside it as
spheresm.nif.bak (once; an existing backup is kept).

Usage: python3 tools/fix-spheres-texpaths.py ["<Data Files>"]
"""
import importlib.util
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("make_pbrgrid", os.path.join(HERE, "make-pbrgrid.py"))
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)

DATA = sys.argv[1] if len(sys.argv) > 1 else mp.DATA
NIF = os.path.join(DATA, "meshes", "vt", "spheresm.nif")


def fixed(name):
    p = name.replace("/", "\\")
    if not p.lower().startswith("textures\\"):
        p = "textures\\" + p
    stem, ext = os.path.splitext(p)
    return stem + ".dds" if ext.lower() == ".tga" else p


def main():
    src = open(NIF, "rb").read()
    hdr, ver, blocks, roots = mp.parse_nif(src)
    if mp.write_nif(hdr, ver, blocks, roots) != src:
        raise SystemExit("self-test FAILED: the writer does not rebuild spheresm.nif; nothing written")
    changed = 0
    for b in blocks:
        if b["type"] == "NiSourceTexture" and b.get("file"):
            new = fixed(b["file"])
            if new != b["file"]:
                print("%-30s -> %s" % (b["file"], new))
                b["file"] = new
                changed += 1
    if not changed:
        print("nothing to change")
        return
    out = mp.write_nif(hdr, ver, blocks, roots)
    _, _, chk, roots2 = mp.parse_nif(out)
    assert len(chk) == len(blocks) and roots2 == roots
    bak = NIF + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(NIF, bak)
    open(NIF, "wb").write(out)
    print("spheresm.nif: %d paths rewritten, %d -> %d bytes (backup %s)" % (changed, len(src), len(out), bak))


if __name__ == "__main__":
    main()
