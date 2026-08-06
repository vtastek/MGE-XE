#!/usr/bin/env python3
"""Dump the block structure of a MW (NIF 4.0.0.2) mesh, loose or from a BSA.

Block bodies have per-type layouts, but block TYPE NAMES are length-prefixed and appear in file
order, so the token scan gives us exact body byte-ranges without a full parser. We then decode only
the fields we care about inside each range.

Usage: nifdump.py <name-substring> [...]
"""
import os, re, struct, sys

DATA = "/mnt/c/mgem/morrowind64/Data Files"
_TOKEN_RE = re.compile(rb"([\x03-\x3c])\x00\x00\x00([A-Z][A-Za-z0-9_]{2,59})")


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


def find_nif(pat):
    """-> [(source, name, blob)] for every nif whose path contains `pat`."""
    hits = []
    meshes = os.path.join(DATA, "Meshes")
    for root, _d, files in os.walk(meshes):
        for fn in files:
            if fn.lower().endswith(".nif") and pat.lower() in fn.lower():
                p = os.path.join(root, fn)
                with open(p, "rb") as f:
                    hits.append(("loose", os.path.relpath(p, meshes), f.read()))
    for b in sorted(os.listdir(DATA)):
        if b.lower().endswith(".bsa"):
            for name, blob in iter_bsa_nifs(os.path.join(DATA, b)):
                if pat.lower() in name.lower():
                    hits.append((b, name, blob))
    return hits


# A node NAME is also a length-prefixed ASCII string, so "Bip01"/"Dummy18" match the token regex
# and split a real block in two. Restrict to actual MW block types: everything starts with "Ni"
# except this short list. Without this, body ranges (and so every field offset) silently shift.
_NON_NI_TYPES = {"RootCollisionNode", "AvoidNode", "BSMirroredNode", "BoundingBox", "TES3ObjectEXTRA"}


def blocks(blob):
    """-> [(type, body_start, body_end)] in file order."""
    toks = []
    for m in _TOKEN_RE.finditer(blob):
        name = m.group(2).decode("ascii")
        if m.group(1)[0] != len(name):
            continue
        if not name.startswith("Ni") and name not in _NON_NI_TYPES:
            continue
        toks.append((m.start(), m.end(), name))
    out = []
    for i, (s, e, name) in enumerate(toks):
        end = toks[i + 1][0] if i + 1 < len(toks) else len(blob)
        out.append((name, e, end))
    return out


def sized_string(blob, off):
    n = struct.unpack_from("<I", blob, off)[0]
    if n > 200:
        return None, off
    return blob[off + 4:off + 4 + n].decode("cp1252", "replace"), off + 4 + n


def avobject_xform(blob, off):
    """NiObjectNET(name, extra, ctrl) + NiAVObject(flags, trans, rot3x3, scale) -> (name, det, scale)."""
    nm, p = sized_string(blob, off)
    if nm is None:
        return None, None, None
    p += 8                                   # extraData ref, controller ref
    flags = struct.unpack_from("<H", blob, p)[0]; p += 2
    t = struct.unpack_from("<3f", blob, p); p += 12
    r = struct.unpack_from("<9f", blob, p); p += 36
    sc = struct.unpack_from("<f", blob, p)[0]
    det = (r[0] * (r[4] * r[8] - r[5] * r[7])
           - r[1] * (r[3] * r[8] - r[5] * r[6])
           + r[2] * (r[3] * r[7] - r[4] * r[6]))
    return nm, det, sc


DRAWMODE = {0: "CCW_OR_BOTH(default)", 1: "DRAW_CCW", 2: "DRAW_CW(!! reversed)", 3: "DRAW_BOTH"}
CYCLE = {0: "LOOP", 1: "REVERSE", 2: "CLAMP", 3: "?"}
INTERP = {1: "LINEAR", 2: "QUADRATIC/bezier", 3: "TBC"}


def det3(r):
    return (r[0] * (r[4] * r[8] - r[5] * r[7])
            - r[1] * (r[3] * r[8] - r[5] * r[6])
            + r[2] * (r[3] * r[7] - r[4] * r[6]))


def dump_skindata(blob, s, e):
    """NiSkinData 4.0.0.2: overall trafo, numBones, skinPartition link, then per-bone
    (rot3x3, pos, scale, boundSphere, numVerts, verts). The BIND transform is what MGE's bone
    palette multiplies through, so a negative determinant here is a real reflection."""
    p = s
    r = struct.unpack_from("<9f", blob, p); p += 36
    p += 12
    sc = struct.unpack_from("<f", blob, p)[0]; p += 4
    nb = struct.unpack_from("<I", blob, p)[0]; p += 4
    p += 4                                   # NiSkinPartition link (present at 4.0.0.2)
    print(f"      overall det={det3(r):+.4f} scale={sc:g}  bones={nb}")
    neg = []
    for b in range(nb):
        if p + 70 > e:
            print(f"      <ran out at bone {b}>")
            return
        br = struct.unpack_from("<9f", blob, p); p += 36
        p += 12
        bsc = struct.unpack_from("<f", blob, p)[0]; p += 4
        p += 16                              # bound sphere
        nv = struct.unpack_from("<H", blob, p)[0]; p += 2
        p += nv * 6
        d = det3(br)
        if b < 4 or d < 0:
            print(f"        bone[{b:2d}] det={d:+.4f} scale={bsc:g} verts={nv}"
                  + ("   <<< NEGATIVE" if d < 0 else ""))
        if d < 0:
            neg.append(b)
    print(f"      negative-determinant bind transforms: {len(neg)}/{nb} {neg[:12]}")


def dump_uvdata(blob, s, e):
    """NiUVData = 4 KeyGroup<float>: U-offset, V-offset, U-tiling, V-tiling."""
    p = s
    for label in ("U-offset", "V-offset", "U-tiling", "V-tiling"):
        if p + 4 > e:
            print(f"      {label:9s} <truncated>")
            return
        n = struct.unpack_from("<I", blob, p)[0]
        p += 4
        if n == 0:
            print(f"      {label:9s} 0 keys")
            continue
        interp = struct.unpack_from("<I", blob, p)[0]
        p += 4
        stride = {1: 8, 2: 16, 3: 20}.get(interp)
        if stride is None or p + n * stride > e:
            print(f"      {label:9s} {n} keys interp={interp} <unparsable>")
            return
        ks = [struct.unpack_from("<" + "f" * (stride // 4), blob, p + i * stride) for i in range(n)]
        p += n * stride
        span = ks[-1][0] - ks[0][0]
        d = ks[-1][1] - ks[0][1]
        print(f"      {label:9s} {n:3d} keys  {INTERP.get(interp, interp):16s} "
              f"t=[{ks[0][0]:.3f}..{ks[-1][0]:.3f}] v=[{ks[0][1]:+.3f}..{ks[-1][1]:+.3f}] "
              f"net={d:+.3f} rate={d/span if span else 0:+.3f}/s")
        for k in ks[:6]:
            print("          " + "  ".join(f"{x:+.4f}" for x in k))
        if len(ks) > 6:
            print(f"          ... {len(ks)-6} more")


def dump(src, name, blob):
    print(f"\n=== [{src}] {name}  ({len(blob)} bytes) ===")
    bl = blocks(blob)
    hdr_blocks = struct.unpack_from("<I", blob, blob.index(b"\n") + 5)[0]
    print(f"    header says {hdr_blocks} blocks; token scan found {len(bl)}")
    for i, (t, s, e) in enumerate(bl):
        size = e - s
        extra = ""
        if t in ("NiNode", "NiTriShape", "NiTriStrips", "NiSourceTexture", "NiTexturingProperty",
                 "NiStencilProperty", "NiMaterialProperty", "NiAlphaProperty", "NiZBufferProperty",
                 "NiVertexColorProperty", "NiWireframeProperty", "NiShadeProperty",
                 "NiAutoNormalParticles", "NiRotatingParticles", "NiBillboardNode", "RootCollisionNode",
                 "AvoidNode", "NiLODNode", "NiSwitchNode", "NiCamera", "NiTextureEffect"):
            nm, _ = sized_string(blob, s)
            if nm is not None:
                extra = f' name="{nm}"'
            if t in ("NiNode", "NiTriShape", "NiTriStrips", "NiBillboardNode", "RootCollisionNode",
                     "AvoidNode", "NiAutoNormalParticles", "NiRotatingParticles"):
                nm2, det, sc = avobject_xform(blob, s)
                if det is not None:
                    flag = "  <<< MIRRORED (det<0)" if det < 0 else ""
                    extra += f"  det={det:+.4f} scale={sc:g}{flag}"
        if t == "NiStencilProperty":
            dm = struct.unpack_from("<I", blob, e - 4)[0]
            extra += f"   drawMode={dm} {DRAWMODE.get(dm, '?')}"
        if t.endswith("Controller") and size >= 26:
            nxt, flags, freq, phase, start, stop, tgt = struct.unpack_from("<IHffffi", blob, s)
            extra += (f"   flags=0x{flags:04X} cycle={CYCLE[(flags >> 1) & 3]} "
                      f"active={(flags >> 3) & 1} freq={freq:g} phase={phase:g} "
                      f"t=[{start:g}..{stop:g}]")
            if t == "NiUVController" and size >= 32:
                ts, ref = struct.unpack_from("<Hi", blob, s + 26)
                extra += f" textureSet={ts} uvData=#{ref}"
        print(f"  [{i:3d}] {t:32s} body={size:6d}{extra}")
        if t == "NiZBufferProperty":
            # 4.0.0.2 body = name + extraData ref + controller ref + flags u16 (= 14 bytes for an
            # unnamed one). bit0 = z-TEST enable, bit1 = z-WRITE enable; 3 = both (the default).
            zf = struct.unpack_from("<H", blob, e - 2)[0]
            print(f"        flags=0x{zf:04X}  test={zf & 1}  WRITE={(zf >> 1) & 1}")
        if t == "NiAlphaProperty":
            fl, thr = struct.unpack_from("<HB", blob, e - 3)
            bits = []
            if fl & 1:
                bits.append("BLEND")
            if fl & (1 << 9):
                bits.append("TESTREF")
            if fl & (1 << 13):
                bits.append("NO_SORTER")
            print(f"        flags=0x{fl:04X} [{' '.join(bits) or 'opaque'}] "
                  f"src={(fl >> 1) & 15} dst={(fl >> 5) & 15} testFunc={(fl >> 10) & 7} ref={thr}")
        if t == "NiMaterialProperty":
            gloss, alpha = struct.unpack_from("<ff", blob, e - 8)
            emis = struct.unpack_from("<3f", blob, e - 20)
            print(f"        emissive=({emis[0]:.3f},{emis[1]:.3f},{emis[2]:.3f}) "
                  f"gloss={gloss:g} alpha={alpha:g}")
        if t == "NiFloatData":
            n, interp = struct.unpack_from("<II", blob, s)
            stride = {1: 8, 2: 16, 3: 20}.get(interp, 0)
            if stride:
                ks = [struct.unpack_from("<2f", blob, s + 8 + i * stride) for i in range(n)]
                print(f"        {n} keys {INTERP.get(interp)}  "
                      + "  ".join(f"({t_:.2f}→{v:.3f})" for t_, v in ks[:10]))
        if t == "NiUVData":
            dump_uvdata(blob, s, e)
        if t == "NiSkinData":
            dump_skindata(blob, s, e)


if __name__ == "__main__":
    for pat in sys.argv[1:]:
        hits = find_nif(pat)
        if not hits:
            print(f"(no nif matching '{pat}')")
        for h in hits:
            dump(*h)
