#!/usr/bin/env python3
"""
NIF controller census for Morrowind data.

Walks every .nif in a Morrowind "Data Files" tree -- loose Meshes\\ AND inside the
BSAs (Morrowind/Tribunal/Bloodmoon + any mod .bsa) -- and tallies the NI* block
types used, with a focus on NiTimeController-derived controllers and particle
systems. The point: know which controllers exist and how frequently, so we can
decide what else to host-migrate the way NiUVController was (2026-07-17).

Why a raw token scan works: MW NIFs are version 4.0.0.2, which frames every block
as `uint32 nameLen + ASCII type name` inline (no header string table). We collect
every length-prefixed ASCII token that looks like a block type ("Ni..." etc). The
`len == token length` + printable-ASCII constraint makes false positives from
string-valued fields astronomically unlikely for the type names we care about.

Usage:
    python3 nif-controller-census.py [DATA_FILES_DIR] [--csv out.csv] [--top N]

Default DATA_FILES_DIR = /mnt/c/mgem/morrowind64/Data Files
"""
import os
import re
import sys
import struct
import argparse
from collections import defaultdict

# ---- block-type framing (regex, C-speed) --------------------------------------
# A block type token is a uint32 length L (3..60, so high 3 bytes are 0) followed
# by exactly L bytes of [A-Za-z0-9_] starting uppercase. The regex finds every
# candidate `len-byte 00 00 00 <name>`; we then confirm the length byte equals the
# name length so string-valued fields can't masquerade as a type token. Matching in
# the C regex engine instead of a per-byte Python loop is ~100x faster.
_TOKEN_RE = re.compile(rb"([\x03-\x3c])\x00\x00\x00([A-Z][A-Za-z0-9_]{2,59})")


def scan_block_types(blob: bytes):
    """Yield block-type name strings found via length-prefixed framing."""
    if len(blob) < 40 or not blob.startswith(b"NetImmerse File Format"):
        return
    for m in _TOKEN_RE.finditer(blob):
        if m.group(1)[0] == len(m.group(2)):
            yield m.group(2).decode("ascii")


# ---- MW BSA reader ------------------------------------------------------------
def iter_bsa_nifs(path: str):
    """Yield (internal_name, bytes) for every .nif inside a MW-format BSA."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 12:
        return
    version, hash_offset, file_count = struct.unpack_from("<III", data, 0)
    if version != 0x100:
        sys.stderr.write(f"  [skip] {os.path.basename(path)}: not a MW BSA (v={version:#x})\n")
        return
    # Layout after the 12-byte header:
    #   file_count * {uint32 size, uint32 offset}
    #   file_count * uint32 nameOffset (into name table)
    #   name table (null-terminated)
    #   hash table at hash_offset (from end of the 12-byte header)
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
        if not name.lower().endswith(".nif"):
            continue
        start = data_base + offset
        yield name, data[start:start + size]


def iter_loose_nifs(meshes_dir: str):
    for root, _dirs, files in os.walk(meshes_dir):
        for fn in files:
            if fn.lower().endswith(".nif"):
                p = os.path.join(root, fn)
                try:
                    with open(p, "rb") as f:
                        yield os.path.relpath(p, meshes_dir), f.read()
                except OSError as e:
                    sys.stderr.write(f"  [skip] {p}: {e}\n")


# ---- verdict table ------------------------------------------------------------
# migratable to the host like NiUVController, or must be replaced / left alone.
VERDICT = {
    "NiUVController":              "DONE  host-migrated 2026-07-17 (UV scroll from sim time)",
    "NiKeyframeController":        "NEXT  rigid TRS anim -> host matrix from sim time (banners/doors/machines)",
    "NiGeomMorpherController":     "PARTIAL  vertex morph; heads reship by design, statics could bake",
    "NiVisController":             "CHEAP  per-key visibility -> host draw-cull flag",
    "NiFlipController":            "CHEAP  texture-flip -> host bindless slot swap from sim time",
    "NiAlphaController":           "CHEAP  animated alpha -> host per-draw alpha from sim time",
    "NiMaterialColorController":   "CHEAP  animated material colour -> host material from sim time",
    "NiPathController":            "MED   path-follow TRS -> host, needs NiPosData+NiFloatData",
    "NiRollController":            "CHEAP  spin about axis -> host matrix from sim time",
    "NiParticleSystemController":  "REPLACE  particle sim+sort; not migratable -> procedural shader (Part C, deferred)",
    "NiBSPArrayController":        "REPLACE  particle array; not migratable -> procedural shader (Part C, deferred)",
}
PARTICLE_TYPES = {
    "NiParticleSystemController", "NiBSPArrayController", "NiAutoNormalParticles",
    "NiRotatingParticles", "NiParticles", "NiParticlesData", "NiAutoNormalParticlesData",
    "NiRotatingParticlesData", "NiParticleColorModifier", "NiParticleGrowFade",
    "NiParticleRotation", "NiGravity", "NiParticleBomb", "NiPlanarCollider",
    "NiParticleColorModifier", "NiBSPArrayController",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?",
                    default="/mnt/c/mgem/morrowind64/Data Files")
    ap.add_argument("--csv", default=None, help="write per-type CSV here")
    ap.add_argument("--top", type=int, default=0, help="print top-N non-controller types too")
    args = ap.parse_args()

    meshes = os.path.join(args.data_dir, "Meshes")
    bsas = [os.path.join(args.data_dir, f) for f in sorted(os.listdir(args.data_dir))
            if f.lower().endswith(".bsa")]

    # per-type: total block count, set of distinct nifs, up to 5 example nifs
    total = defaultdict(int)
    nifs = defaultdict(set)
    examples = defaultdict(list)
    file_count = 0

    def account(nif_name, blob):
        nonlocal file_count
        file_count += 1
        seen = set()
        for t in scan_block_types(blob):
            total[t] += 1
            seen.add(t)
        for t in seen:
            s = nifs[t]
            if nif_name not in s:
                s.add(nif_name)
                if len(examples[t]) < 5:
                    examples[t].append(nif_name)

    print(f"[census] data dir: {args.data_dir}")
    if os.path.isdir(meshes):
        print(f"[census] scanning loose Meshes ...")
        for name, blob in iter_loose_nifs(meshes):
            account("loose:" + name, blob)
    for bsa in bsas:
        print(f"[census] scanning {os.path.basename(bsa)} ...")
        for name, blob in iter_bsa_nifs(bsa):
            account(f"{os.path.basename(bsa)}:{name}", blob)

    print(f"\n[census] {file_count} NIFs scanned, {len(total)} distinct block types\n")

    def is_ctrl(t):
        return t.endswith("Controller")

    def row(t):
        return (t, total[t], len(nifs[t]))

    controllers = sorted((t for t in total if is_ctrl(t)),
                         key=lambda t: total[t], reverse=True)
    particles = sorted((t for t in total if t in PARTICLE_TYPES),
                       key=lambda t: total[t], reverse=True)

    def dump(title, types):
        print(f"== {title} ==")
        print(f"{'type':<32}{'blocks':>9}{'nifs':>8}   verdict / examples")
        for t in types:
            v = VERDICT.get(t, "")
            ex = "" if v else ("  e.g. " + ", ".join(os.path.basename(x) for x in examples[t][:3]))
            print(f"{t:<32}{total[t]:>9}{len(nifs[t]):>8}   {v}{ex}")
        print()

    dump("CONTROLLERS (NiTimeController-derived)", controllers)
    dump("PARTICLE SYSTEM blocks", particles)

    if args.top:
        others = sorted((t for t in total if not is_ctrl(t) and t not in PARTICLE_TYPES),
                        key=lambda t: total[t], reverse=True)[:args.top]
        dump(f"TOP {args.top} OTHER block types", others)

    if args.csv:
        # Only controllers + particle blocks are committed: their type names never
        # collide with node names, so these rows are exact. The raw "other" histogram
        # is polluted by inline node-name tokens (e.g. "NightDaySwitch") and is left
        # to the stdout --top view for eyeballing, not the artifact.
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("type,category,blocks,nifs,verdict,examples\n")
            for t in sorted(total, key=lambda t: total[t], reverse=True):
                if is_ctrl(t):
                    cat = "controller"
                elif t in PARTICLE_TYPES:
                    cat = "particle"
                else:
                    continue
                v = VERDICT.get(t, "").replace(",", ";")
                ex = "; ".join(examples[t])
                f.write(f'{t},{cat},{total[t]},{len(nifs[t])},"{v}","{ex}"\n')
        print(f"[census] CSV (controllers+particles) -> {args.csv}")


if __name__ == "__main__":
    main()
