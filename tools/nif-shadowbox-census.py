#!/usr/bin/env python3
"""Census: what actually lives under a node whose NAME trips isLegacyShadowGeometry?

scenegraph_geometry_cache.cpp walk() prunes the WHOLE SUBTREE at any node whose name matches
`[Tri ]shadow[box][digits]`. That is correct when the node wraps only MW's fake blob shadow, and
silently deletes real geometry when a mesh author parents something else under it (found via
LIght_Com_Candle_10.nif, whose entire silver body sits under a node called "ShadowBox").

Prints, per offending mesh, the geometry we currently drop that a SHAPE-level prune would keep.

Usage: nif-shadowbox-census.py [--all]     (--all also lists meshes where the prune is correct)
"""
import io, os, re, struct, sys, tempfile

sys.path.insert(0, "/mnt/c/projects/mgexe/MGE-XE/io_scene_mw-master/lib")
from es3 import nif
from es3.nif import NiStream

DATA = "/mnt/c/mgem/morrowind64/Data Files"


def is_legacy_shadow(name):
    """Byte-for-byte mirror of isLegacyShadowGeometry (scenegraph_geometry_cache.cpp:2618)."""
    if not name:
        return False
    n = name
    if n[:4].lower() == "tri ":
        n = n[4:]
    if n[:6].lower() != "shadow":
        return False
    n = n[6:]
    if n[:3].lower() == "box":
        n = n[3:]
    return n.lstrip("0123456789") == ""


def iter_bsa_nifs(path):
    with open(path, "rb") as f:
        data = f.read()
    version, hash_offset, file_count = struct.unpack_from("<III", data, 0)
    if version != 0x100:
        return
    rec_base = 12
    name_off_base = rec_base + file_count * 8
    name_table_base = name_off_base + file_count * 4
    data_base = 12 + hash_offset + file_count * 8
    for i in range(file_count):
        size, offset = struct.unpack_from("<II", data, rec_base + i * 8)
        name_off = struct.unpack_from("<I", data, name_off_base + i * 4)[0]
        ns = name_table_base + name_off
        ne = data.index(b"\x00", ns)
        name = data[ns:ne].decode("cp1252", "replace")
        if name.lower().endswith(".nif"):
            yield name, data[data_base + offset:data_base + offset + size]


def sources():
    meshes = os.path.join(DATA, "Meshes")
    for root, _d, files in os.walk(meshes):
        for fn in files:
            if fn.lower().endswith(".nif"):
                p = os.path.join(root, fn)
                yield "loose", os.path.relpath(p, meshes), p, None
    for fn in sorted(os.listdir(DATA)):
        if fn.lower().endswith(".bsa"):
            for name, blob in iter_bsa_nifs(os.path.join(DATA, fn)):
                yield fn, name, None, blob


def load(path, blob):
    s = NiStream()
    if blob is None:
        s.load(path)
        return s
    tmp = tempfile.NamedTemporaryFile(suffix=".nif", delete=False)
    try:
        tmp.write(blob)
        tmp.close()
        s.load(tmp.name)
        return s
    finally:
        os.unlink(tmp.name)


def subtree_shapes(o, seen=None):
    """(name, kind) for every NiTriBasedGeom at or under o."""
    if seen is None:
        seen = set()
    if id(o) in seen:
        return
    seen.add(id(o))
    if isinstance(o, nif.NiTriBasedGeom):
        yield (getattr(o, "name", "") or "", type(o).__name__)
    for c in (getattr(o, "children", None) or []):
        if c is not None:
            yield from subtree_shapes(c, seen)


def main():
    show_all = "--all" in sys.argv
    nPruners = nOverPrune = nFiles = 0
    lostTotal = 0
    for src, name, path, blob in sources():
        nFiles += 1
        try:
            s = load(path, blob)
        except Exception:
            continue
        for o in s.objects():
            if not isinstance(o, nif.NiAVObject):
                continue
            if not is_legacy_shadow(getattr(o, "name", None)):
                continue
            nPruners += 1
            shapes = list(subtree_shapes(o))
            # What a SHAPE-level prune would keep that the SUBTREE prune deletes.
            kept = [sh for sh in shapes if not is_legacy_shadow(sh[0])]
            if kept:
                nOverPrune += 1
                lostTotal += len(kept)
                print(f"[{src}] {name}")
                print(f"    node '{o.name}' ({type(o).__name__}) drops {len(kept)} real shape(s):")
                for shname, kind in kept:
                    print(f"        {kind} '{shname}'")
            elif show_all:
                print(f"[{src}] {name}: node '{o.name}' -> {len(shapes)} shadow shape(s), prune correct")
    print(f"\n=== {nFiles} NIFs scanned | {nPruners} shadow-named node(s) | "
          f"{nOverPrune} OVER-PRUNE site(s) losing {lostTotal} real shape(s) ===")


main()
