#!/usr/bin/env python3
"""
Hero distant-statics authoring (MGE XE supported mod, NOT part of the release).

Generates the `<model>_herodist.nif` copies that MGEgui's distant-land bake prefers over the
low-poly `_dist` stand-in. A hero copy is a byte-for-byte copy of the REAL multi-layer NIF; its
mere presence is the opt-in, so the bake captures that mesh's real per-subset UV keys + alpha
blending (see MGEfuncs/NifConverter.cpp hero mode). We copy from the user's own Morrowind.bsa
rather than committing Bethesda meshes into the repo.

Scope (user decision): ghostfence + lava only.

Usage:
    python3 make_herodist.py <Morrowind.bsa> <dest-Data-Files-dir>
e.g.
    python3 make_herodist.py "C:/mgem/morrowind64/Data Files/Morrowind.bsa" "C:/mgem/morrowind64/Data Files"
"""
import sys, struct, os

# Internal BSA names of the real meshes to shadow with a _herodist copy.
HERO_MESHES = [
    "meshes\\x\\ex_gg_fence_s_01.nif",
    "meshes\\x\\ex_gg_fence_s_02.nif",
    "meshes\\x\\ex_gg_fence_s_03.nif",
    "meshes\\x\\ex_gg_fence_s_04.nif",
    "meshes\\x\\ex_gg_fence_s_h_01.nif",
    "meshes\\i\\in_lava_1024.nif",
    "meshes\\i\\in_lava_1024_01.nif",
    "meshes\\i\\in_lava_256.nif",
    "meshes\\i\\in_lava_256a.nif",
    "meshes\\i\\in_lava_512.nif",
    "meshes\\i\\in_lava_oval.nif",
]

def load_bsa(path):
    f = open(path, 'rb')
    ver, hashOff, count = struct.unpack('<III', f.read(12))
    if ver != 0x100:
        raise SystemExit("not a Morrowind BSA (version %s)" % hex(ver))
    recs = [struct.unpack('<II', f.read(8)) for _ in range(count)]        # (size, offset)
    nameOffs = [struct.unpack('<I', f.read(4))[0] for _ in range(count)]
    nameBufLen = hashOff - (count*8 + count*4)
    nameBuf = f.read(nameBufLen)
    names = []
    for no in nameOffs:
        e = nameBuf.index(b'\x00', no)
        names.append(nameBuf[no:e].decode('cp1252').lower())
    dataStart = 12 + hashOff + count*8
    return f, names, recs, dataStart

def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    bsa, dataFiles = sys.argv[1], sys.argv[2]
    f, names, recs, dataStart = load_bsa(bsa)
    idx = {n: i for i, n in enumerate(names)}
    for internal in HERO_MESHES:
        key = internal.lower()
        if key not in idx:
            print("MISSING in BSA: %s" % internal); continue
        size, off = recs[idx[key]]
        f.seek(dataStart + off); data = f.read(size)
        # meshes\x\foo.nif -> <dataFiles>\meshes\x\foo_herodist.nif
        rel = internal[:-4] + "_herodist.nif"        # strip .nif, append suffix
        out = os.path.join(dataFiles, rel.replace('\\', os.sep))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, 'wb').write(data)
        print("wrote %6d bytes -> %s" % (size, out))

if __name__ == '__main__':
    main()
