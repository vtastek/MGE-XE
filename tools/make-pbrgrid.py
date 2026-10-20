#!/usr/bin/env python3
"""Build the PBR sphere grid: a metal row and a dielectric row, roughness 0..1 in 0.1 steps.

Writes, into the dev install's Data Files:
  meshes/vt/pbrgrid.nif               22 spheres, two rows, one static
  textures/vt/pbrgrid/{m,d}_rNN.dds   constant albedo (DXT1)
  textures/vt/pbrgrid/{m,d}_rNN_paramh.dds  constant R metal, G rough, B 0.5, A 0.5 (DXT5)
  pbrgrid.esp                         STAT vtpbrgrid + one reference beside sphere.ESP's

WHY THE MESH IS NOT AUTHORED FROM SCRATCH: MW's NetImmerse 4.0.0.2 framing fails silently when
it is wrong (see make-alpha-probe-nif.py). The sphere geometry and the material are taken BYTE FOR
BYTE from the user's meshes/vt/spheresm.nif, and the blocks this writes (NiNode, NiTriShape,
NiTexturingProperty, NiSourceTexture) use one writer that is first proven by rebuilding
spheresm.nif itself bit-identically from its parse. If that self-test fails, nothing is written.

The reference copies vtspheres' position, rotation and scale exactly, and the rows are offset in
the NIF's own X, so they sit parallel to the existing row whatever MW's rotation convention is.
Move it in the Construction Set if it lands in a wall.

Usage: python3 tools/make-pbrgrid.py ["<Data Files>"]
"""
import os, struct, sys

DATA = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/mgem/morrowind64/Data Files"
SRC_NIF = os.path.join(DATA, "meshes", "vt", "spheresm.nif")
SRC_ESP = os.path.join(DATA, "sphere.ESP")
MASTER = os.path.join(DATA, "Morrowind.esm")

STEPS = [i / 10 for i in range(11)]
SPACING = 220.0                    # NIF units along Y (sphere radius 100; ref scale 0.5)
ROW_X = {"m": 300.0, "d": 550.0}   # NIF units off the existing row
ALBEDO = {"m": (250, 250, 250),    # sRGB: ~0.95 linear, a silver-ish F0
          "d": (128, 128, 128)}    # sRGB mid grey, ~0.22 linear
TEX = 64                           # constant maps: size only feeds the mip chain


# ── NIF 4.0.0.2 ──────────────────────────────────────────────────────────────────────────────
class R:
    def __init__(s, d, i=0): s.d, s.i = d, i
    def take(s, fmt):
        v = struct.unpack_from("<" + fmt, s.d, s.i); s.i += struct.calcsize("<" + fmt)
        return v if len(v) > 1 else v[0]
    def str(s):
        n = s.take("I"); v = s.d[s.i:s.i + n]; s.i += n
        return v.decode("latin1")


def s_(x): return struct.pack("<I", len(x)) + x.encode("latin1")


def parse_nif(d):
    r = R(d, d.index(b"\n") + 1)
    ver, nb = r.take("II")
    blocks = []
    for _ in range(nb):
        start = r.i; t = r.str(); o = {"type": t}
        if t in ("NiNode", "NiTriShape"):
            o.update(name=r.str(), extra=r.take("i"), ctrl=r.take("i"), flags=r.take("H"),
                     trans=r.take("3f"), rot=r.take("9f"), scale=r.take("f"), vel=r.take("3f"))
            o["props"] = [r.take("i") for _ in range(r.take("I"))]
            assert r.take("I") == 0, "bounding box"
            if t == "NiNode":
                o["children"] = [r.take("i") for _ in range(r.take("I"))]
                o["effects"] = [r.take("i") for _ in range(r.take("I"))]
            else:
                o["data"], o["skin"] = r.take("ii")
        elif t == "NiTexturingProperty":
            o.update(name=r.str(), extra=r.take("i"), ctrl=r.take("i"), flags=r.take("H"),
                     apply=r.take("I"))
            o["tex"] = []
            for k in range(r.take("I")):
                if r.take("I"):
                    assert k != 4, "bump slot"
                    o["tex"].append(r.take("iIIIHHH"))
                else:
                    o["tex"].append(None)
        elif t == "NiSourceTexture":
            o.update(name=r.str(), extra=r.take("i"), ctrl=r.take("i"))
            assert r.take("B") == 1, "embedded texture"
            o.update(file=r.str(), pix=r.take("I"), mip=r.take("I"), alphaf=r.take("I"),
                     static=r.take("B"))
        elif t == "NiMaterialProperty":
            r.str(); r.take("ii"); r.take("H"); r.take("14f")
        elif t == "NiTriShapeData":
            nv = r.take("H")
            if r.take("I"): r.i += 12 * nv
            if r.take("I"): r.i += 12 * nv
            r.take("4f")
            if r.take("I"): r.i += 16 * nv
            nuv = r.take("H")
            if r.take("I"): r.i += 8 * nv * nuv
            nt = r.take("H"); r.take("I"); r.i += 6 * nt
            for _ in range(r.take("H")): r.i += 2 * r.take("H")
        else:
            raise SystemExit("unknown block " + t)
        o["raw"] = d[start:r.i]
        blocks.append(o)
    roots = [r.take("i") for _ in range(r.take("I"))]
    assert r.i == len(d), "trailing bytes"
    return d[:d.index(b"\n") + 1], ver, blocks, roots


def write_block(o):
    t = o["type"]
    if t not in ("NiNode", "NiTriShape", "NiTexturingProperty", "NiSourceTexture"):
        return o["raw"]
    b = s_(t) + s_(o["name"]) + struct.pack("<ii", o["extra"], o["ctrl"])
    if t in ("NiNode", "NiTriShape"):
        b += struct.pack("<H3f9ff3f", o["flags"], *o["trans"], *o["rot"], o["scale"], *o["vel"])
        b += struct.pack("<I", len(o["props"])) + b"".join(struct.pack("<i", p) for p in o["props"])
        b += struct.pack("<I", 0)
        if t == "NiNode":
            for lst in (o["children"], o["effects"]):
                b += struct.pack("<I", len(lst)) + b"".join(struct.pack("<i", c) for c in lst)
        else:
            b += struct.pack("<ii", o["data"], o["skin"])
    elif t == "NiTexturingProperty":
        b += struct.pack("<HII", o["flags"], o["apply"], len(o["tex"]))
        for td in o["tex"]:
            b += struct.pack("<I", 0) if td is None else struct.pack("<IiIIIHHH", 1, *td)
    else:
        b += struct.pack("<B", 1) + s_(o["file"]) + struct.pack("<IIIB", o["pix"], o["mip"], o["alphaf"], o["static"])
    return b


def write_nif(hdr, ver, blocks, roots):
    out = hdr + struct.pack("<II", ver, len(blocks)) + b"".join(write_block(o) for o in blocks)
    return out + struct.pack("<I", len(roots)) + b"".join(struct.pack("<i", x) for x in roots)


# ── DDS: constant colour, full mip chain ─────────────────────────────────────────────────────
def rgb565(c):
    r, g, b = c
    return ((r * 31 + 127) // 255) << 11 | ((g * 63 + 127) // 255) << 5 | ((b * 31 + 127) // 255)


def dds(fourcc, block, size):
    mips, s, n = 0, size, 0
    while True:
        mips += 1; n += max(1, s // 4) ** 2
        if s == 1: break
        s //= 2
    bs = len(block)
    hdr = struct.pack("<4sIIIIIII44xII4s20xIIII", b"DDS ", 124, 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000,
                      size, size, max(1, size // 4) ** 2 * bs, 0, mips,
                      32, 0x4, fourcc, 0x1000 | 0x8 | 0x400000, 0, 0, 0)
    hdr += struct.pack("<I", 0)
    assert len(hdr) == 128
    return hdr + block * n


def dxt1(c):
    v = rgb565(c)
    return dds(b"DXT1", struct.pack("<HHI", v, v, 0), TEX)


def dxt5(c, a):
    v = rgb565(c)
    return dds(b"DXT5", struct.pack("<BB6xHHI", a, a, v, v, 0), TEX)


# ── ESP ──────────────────────────────────────────────────────────────────────────────────────
def sub(tag, payload): return tag + struct.pack("<I", len(payload)) + payload
def rec(tag, subs):
    body = b"".join(subs)
    return tag + struct.pack("<III", len(body), 0, 0) + body
def z(x): return x.encode("latin1") + b"\0"


def esp_records(d):
    i, out = 0, []
    while i < len(d):
        t = d[i:i + 4]; n, = struct.unpack_from("<I", d, i + 4); body = d[i + 16:i + 16 + n]; j, subs = 0, []
        while j < len(body):
            st = body[j:j + 4]; sn, = struct.unpack_from("<I", body, j + 4)
            subs.append((st, body[j + 8:j + 8 + sn])); j += 8 + sn
        out.append((t, subs)); i += 16 + n
    return out


def main():
    src = open(SRC_NIF, "rb").read()
    hdr, ver, blocks, roots = parse_nif(src)
    if write_nif(hdr, ver, blocks, roots) != src:
        raise SystemExit("self-test FAILED: the writer does not rebuild spheresm.nif; nothing written")
    print("self-test: spheresm.nif rebuilt bit-identically")

    shape0 = blocks[1]
    mat = blocks[shape0["props"][0]]
    tprop = next(blocks[p] for p in shape0["props"] if blocks[p]["type"] == "NiTexturingProperty")
    stex = blocks[tprop["tex"][0][0]]
    data = blocks[shape0["data"]]
    assert mat["type"] == "NiMaterialProperty" and data["type"] == "NiTriShapeData"

    # 0 root, 1 material (shared), 2 tri data (shared), then shape/texprop/source per sphere.
    root = dict(blocks[0], name="pbrgrid.nif", children=[])
    out = [root, mat, data]
    tdir = os.path.join(DATA, "textures", "vt", "pbrgrid")
    os.makedirs(tdir, exist_ok=True)
    for row in ("m", "d"):
        for k, rough in enumerate(STEPS):
            stem = "%s_r%02d" % (row, round(rough * 10))
            base = len(out)
            root["children"].append(base)
            out.append(dict(shape0, name="pbr_" + stem, props=[1, base + 1], data=2,
                            trans=(ROW_X[row], (k - 5) * SPACING, 0.0)))
            t = dict(tprop); t["tex"] = [(base + 2,) + tuple(tprop["tex"][0][1:])] + list(tprop["tex"][1:])
            out.append(t)
            out.append(dict(stex, file="textures\\vt\\pbrgrid\\%s.dds" % stem))
            metal = 255 if row == "m" else 0
            g = round(rough * 255)
            open(os.path.join(tdir, stem + ".dds"), "wb").write(dxt1(ALBEDO[row]))
            open(os.path.join(tdir, stem + "_paramh.dds"), "wb").write(dxt5((metal, g, 128), 128))
            print("%s  metal %d  rough %.3f (G565 %d/63)" % (stem, metal // 255, rough, (g * 63 + 127) // 255))

    nif = write_nif(hdr, ver, out, [0])
    hdr2, ver2, chk, roots2 = parse_nif(nif)
    assert len(chk) == len(out) and roots2 == [0]
    open(os.path.join(DATA, "meshes", "vt", "pbrgrid.nif"), "wb").write(nif)
    print("pbrgrid.nif: %d blocks, %d bytes" % (len(out), len(nif)))

    # The reference: vtspheres' own placement, from sphere.ESP.
    cell = next(s for t, s in esp_records(open(SRC_ESP, "rb").read()) if t == b"CELL")
    head = [(t, v) for t, v in cell if t in (b"NAME", b"DATA", b"RGNN")][:3]
    head.append((b"NAM0", struct.pack("<I", 1)))   # this plugin's reference count in the cell
    place = [v for t, v in cell if t == b"DATA" and len(v) == 24][0]
    xscl = [v for t, v in cell if t == b"XSCL"][0]
    master_size = os.path.getsize(MASTER)
    hedr = struct.pack("<fI32s256sI", 1.3, 0, b"MGE-XE tools/make-pbrgrid.py",
                       b"PBR sphere grid: metal and dielectric rows, roughness 0..1", 2)
    esp = rec(b"TES3", [sub(b"HEDR", hedr), sub(b"MAST", z("Morrowind.esm")), sub(b"DATA", struct.pack("<Q", master_size))])
    esp += rec(b"STAT", [sub(b"NAME", z("vtpbrgrid")), sub(b"MODL", z("vt\\pbrgrid.nif"))])
    esp += rec(b"CELL", [sub(t, v) for t, v in head] +
               [sub(b"FRMR", struct.pack("<I", 1)), sub(b"NAME", z("vtpbrgrid")), sub(b"XSCL", xscl), sub(b"DATA", place)])
    open(os.path.join(DATA, "pbrgrid.esp"), "wb").write(esp)
    print("pbrgrid.esp: placed at", struct.unpack("<6f", place), "scale", struct.unpack("<f", xscl)[0])


if __name__ == "__main__":
    main()
