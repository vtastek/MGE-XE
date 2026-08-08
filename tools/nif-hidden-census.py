#!/usr/bin/env python3
"""Census of AUTHORED-HIDDEN geometry: NiTriShape/NiTriStrips whose NiAVObject flags carry
bit 0 (APP_CULLED) *in the file*.

Why this matters: MW's engine also sets appCulled at runtime for off-screen culling, so the
cache's normal walk stops at it either way. But the post-purge INTERIOR walk deliberately walks
THROUGH app-cull (bypassCullDeep) to reach off-screen fixtures — and there is no way to tell
"engine culled it this frame" from "the artist shipped it hidden" by reading the flag. Anything
listed here is geometry the deep walk un-hides and the host then draws forever.

Usage: nif-hidden-census.py [name-substring ...]     (no args = whole corpus)
"""
import os, re, struct, sys, collections

DATA = "/mnt/c/mgem/morrowind64/Data Files"
_TOKEN_RE = re.compile(rb"([\x03-\x3c])\x00\x00\x00([A-Z][A-Za-z0-9_]{2,59})")
_NON_NI_TYPES = {"RootCollisionNode", "AvoidNode", "BSMirroredNode", "BoundingBox", "TES3ObjectEXTRA"}
_GEOM = ("NiTriShape", "NiTriStrips", "NiAutoNormalParticles", "NiRotatingParticles")


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


def iter_corpus(pats):
    meshes = os.path.join(DATA, "Meshes")
    for root, _d, files in os.walk(meshes):
        for fn in files:
            if not fn.lower().endswith(".nif"):
                continue
            p = os.path.relpath(os.path.join(root, fn), meshes)
            if pats and not any(x.lower() in p.lower() for x in pats):
                continue
            with open(os.path.join(root, fn), "rb") as f:
                yield "loose", p, f.read()
    for b in sorted(os.listdir(DATA)):
        if not b.lower().endswith(".bsa"):
            continue
        for name, blob in iter_bsa_nifs(os.path.join(DATA, b)):
            if pats and not any(x.lower() in name.lower() for x in pats):
                continue
            yield b, name, blob


def blocks(blob):
    toks = []
    for m in _TOKEN_RE.finditer(blob):
        name = m.group(2).decode("ascii")
        if m.group(1)[0] != len(name):
            continue
        if not name.startswith("Ni") and name not in _NON_NI_TYPES:
            continue
        toks.append((m.start(), m.end(), name))
    return [(n, e, toks[i + 1][0] if i + 1 < len(toks) else len(blob))
            for i, (s, e, n) in enumerate(toks)]


def hidden_shapes(blob):
    """-> [(type, name, flags)] for every geometry block flagged APP_CULLED in the file."""
    out = []
    for t, s, _e in blocks(blob):
        if t not in _GEOM:
            continue
        n = struct.unpack_from("<I", blob, s)[0]
        if n > 200:
            continue
        nm = blob[s + 4:s + 4 + n].decode("cp1252", "replace")
        p = s + 4 + n + 8                       # extraData ref, controller ref
        flags = struct.unpack_from("<H", blob, p)[0]
        if flags & 1:
            out.append((t, nm, flags))
    return out


def main():
    pats = sys.argv[1:]
    by_name = collections.Counter()
    by_nif = {}
    scanned = 0
    for src, name, blob in iter_corpus(pats):
        scanned += 1
        try:
            h = hidden_shapes(blob)
        except Exception:
            continue
        if h:
            by_nif[f"{src}:{name}"] = h
            for _t, nm, _f in h:
                by_name[nm] += 1

    print(f"scanned {scanned} nifs; {len(by_nif)} carry authored-hidden geometry "
          f"({sum(len(v) for v in by_nif.values())} shapes)\n")
    print("=== most common hidden shape names ===")
    for nm, c in by_name.most_common(40):
        print(f"  {c:5d}  {nm}")
    if pats:
        print("\n=== per-nif ===")
        for k, v in sorted(by_nif.items()):
            print(f"  {k}")
            for t, nm, f in v:
                print(f"      {t:12s} flags=0x{f:04X}  {nm}")


if __name__ == "__main__":
    main()
