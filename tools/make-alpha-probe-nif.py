#!/usr/bin/env python3
"""Build a purpose-made NiAlphaController test mesh for ControllerProbe.

WHY NOT AUTHOR A NIF FROM SCRATCH: MW's NetImmerse 4.0.0.2 framing is unforgiving
(bools are 32-bit at this version but 8-bit later, block order defines the index space,
the footer carries the root list) and a malformed file simply fails to load with no useful
diagnostic. There is also no way to test-load one from here.

WHAT THIS DOES INSTEAD: start from meshes/f/bm_forcefield.nif -- the one mesh already
PROVEN to render and animate as a spawned static in this exact setup -- and rewrite only
FLOAT VALUES IN PLACE. Every edit is a same-width overwrite, so the block sizes, the index
space and the footer are all untouched by construction. It cannot be malformed; the only
thing that can be wrong is the numbers, which is exactly what we want to control.

Three edits, each fixing something that made the vanilla candidates unreadable:
  1. NiFloatData curves -> a slow, brutal square blink (1.0 held, then 0.0 held).
     Vanilla's 0.20..0.60 wobble was too subtle to judge; this is fully on / fully off.
  2. NiAlphaController start/stop -> 0..CYCLE so every controller runs the same cycle.
     The stock file staggers them (8.00..8.90, 0.00..8.93) which desynchronises the sheet.
  3. NiMaterialProperty alpha -> 1.0 and emissive -> white. Base alpha 1.0 means a FROZEN
     capture reads as visible-and-constant rather than invisible, and self-illumination
     means it is readable in a dark interior without depending on local lighting.

The root stays NiBSAnimationNode, which is what makes MW start the controllers at all.
"""
import os, struct, sys

BSA = "/mnt/c/mgem/morrowind64/Data Files/Bloodmoon.bsa"
SOURCE = "meshes/f/bm_forcefield.nif"
CYCLE = 4.0          # seconds per blink cycle
STRIDE = {1: 8, 2: 16, 5: 8}


def tok(n):
    return struct.pack("<I", len(n)) + n.encode("ascii")


def read_from_bsa(path, want):
    with open(path, "rb") as f:
        blob = f.read()
    ver, hash_off, count = struct.unpack_from("<III", blob, 0)
    names_off = 12 + count * 8
    table_off = names_off + count * 4
    data_off = 12 + hash_off + count * 8
    for i in range(count):
        size, off = struct.unpack_from("<II", blob, 12 + i * 8)
        noff, = struct.unpack_from("<I", blob, names_off + i * 4)
        end = blob.index(b"\0", table_off + noff)
        nm = blob[table_off + noff:end].decode("cp1252").replace("\\", "/").lower()
        if nm == want:
            return bytearray(blob[data_off + off:data_off + off + size])
    raise SystemExit(f"{want} not found in {path}")


def curve_at(t_span):
    """A square blink over t_span seconds: opaque, snap out, hold, snap back."""
    return [(0.00 * t_span, 1.0), (0.35 * t_span, 1.0), (0.40 * t_span, 0.0),
            (0.70 * t_span, 0.0), (0.75 * t_span, 1.0), (1.00 * t_span, 1.0)]


def patch_float_data(buf):
    needle, at, n = tok("NiFloatData"), 0, 0
    while True:
        at = buf.find(needle, at)
        if at < 0:
            return n
        p = at + len(needle)
        nkeys, ktype = struct.unpack_from("<II", buf, p)
        stride = STRIDE.get(ktype)
        if not stride or not (0 < nkeys < 4096):
            at += len(needle)
            continue
        keys, p2 = curve_at(CYCLE), p + 8
        # Same key COUNT, same stride -> same bytes. Resample our curve onto the slots the
        # file already has; a tangent-keyed curve keeps its tangents zeroed (linear).
        for k in range(nkeys):
            frac = k / max(nkeys - 1, 1)
            t = frac * CYCLE
            v = 1.0
            for i in range(len(keys) - 1):
                t0, v0 = keys[i]
                t1, v1 = keys[i + 1]
                if t0 <= t <= t1:
                    v = v0 if t1 == t0 else v0 + (v1 - v0) * (t - t0) / (t1 - t0)
                    break
            struct.pack_into("<ff", buf, p2 + k * stride, t, v)
            if stride == 16:
                struct.pack_into("<ff", buf, p2 + k * stride + 8, 0.0, 0.0)
        n += 1
        at = p2 + nkeys * stride


def patch_controllers(buf):
    needle, at, n = tok("NiAlphaController"), 0, 0
    while True:
        at = buf.find(needle, at)
        if at < 0:
            return n
        p = at + len(needle)
        # next:i32 flags:u16 freq:f phase:f start:f stop:f target:i32 data:i32
        struct.pack_into("<f", buf, p + 6, 1.0)        # frequency
        struct.pack_into("<f", buf, p + 10, 0.0)       # phase
        struct.pack_into("<f", buf, p + 14, 0.0)       # start
        struct.pack_into("<f", buf, p + 18, CYCLE)     # stop
        flags, = struct.unpack_from("<H", buf, p + 4)
        struct.pack_into("<H", buf, p + 4, flags | 8)  # force Active
        n += 1
        at = p + 30


def patch_materials(buf):
    needle, at, n = tok("NiMaterialProperty"), 0, 0
    while True:
        at = buf.find(needle, at)
        if at < 0:
            return n
        p = at + len(needle)
        nlen, = struct.unpack_from("<I", buf, p)
        if nlen > 256:
            at += len(needle)
            continue
        q = p + 4 + nlen + 4 + 4 + 2      # name, extraData, controller, flags
        struct.pack_into("<3f", buf, q + 36, 1.0, 1.0, 1.0)   # emissive -> white
        struct.pack_into("<f", buf, q + 52, 1.0)              # alpha -> fully opaque
        n += 1
        at = q + 56


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    if not out:
        raise SystemExit("usage: make-alpha-probe-nif.py <output.nif>")
    buf = read_from_bsa(BSA, SOURCE)
    before = len(buf)
    nf = patch_float_data(buf)
    nc = patch_controllers(buf)
    nm = patch_materials(buf)
    assert len(buf) == before, "size changed - the in-place invariant is broken"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(buf)
    print(f"{SOURCE} -> {out}")
    print(f"  {before} bytes in, {len(buf)} bytes out (must match)")
    print(f"  patched {nf} NiFloatData curves, {nc} NiAlphaControllers, {nm} materials")
    print(f"  cycle {CYCLE}s: opaque -> snap invisible -> hold -> snap back")


if __name__ == "__main__":
    main()
