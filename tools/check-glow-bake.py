#!/usr/bin/env python3
"""
Verify the GitD day/night window variants in a distant-land statics bake.

Parses Data Files/distantland/statics/static_meshes positionally, exactly as
mgeHost64 loadDistantStatics() does:

  per DistantStatic { u32 numSubsets; f32 radius; f32[3] center; u8 type;
    per subset { f32 r; f32[3] c; f32[3] aabbMin; f32[3] aabbMax; i32 verts; i32 faces;
                 vbytes[verts*20]; u16 idx[faces*3]; u8 flags[2]; u16 pathsize; char name[pathsize] } }

flags[1]: bit0 = hasUVController, bit1 = night-only, bit2 = day-only, bit3 = multiply layer.
A vertex is 20 B: FLOAT16_4 pos (8) | UBYTE4 normal, .w = emissive (4) | D3DCOLOR (4) | FLOAT16_2 uv (4).
"""
import struct
import sys
from collections import Counter

# A lit GitD window is a multi-map material: 2-3 layers, each on its own UV set, multiplied
# together. The BASE layer is often a near-neutral glow sheet, so a lit window that baked ONLY
# that layer reads as a white rectangle at distance (the whole "still white" symptom). With the
# layer bake these names are expected and correct — they are the base the layers multiply onto.
# They only signal a fault when the bake carries NO multiply layers at all.
GLOW_SHEETS = ("tex02_dark1", "lightnessgrading", "ray_alpha")
# Hero meshes (ghostfence/lava) must NEVER be day/night split: their subsets join back to
# hero_anim.data by ordinal, so splitting shifts every anim slot and the fence loses its glow.
HERO_TEXTURES = ("tx_gg_fence",)

path = sys.argv[1] if len(sys.argv) > 1 else \
    r"/mnt/c/mgem/morrowind64/Data Files/distantland/statics/static_meshes"

blob = open(path, "rb").read()
n = len(blob)
p = 0

statics = 0
subsets = 0
night = 0
day = 0
uvctrl = 0
unknown_bits = Counter()
night_emissive = Counter()   # emissive byte histogram over night-variant vertices
night_tex = Counter()        # which texture each night variant actually baked
hero_split = Counter()       # hero subsets that got day/night split (must stay empty)
mul_layers = 0                # multi-map layers stacked onto a base (bit3)
layered_statics = 0           # statics carrying at least one such layer
paired_statics = 0

while p + 21 <= n:
    numSubsets, = struct.unpack_from("<I", blob, p)
    p += 4 + 4 + 12 + 1                     # numSubsets + radius + center + type
    if numSubsets > 100000:
        print("!! implausible numSubsets=%d at %d - parser desync" % (numSubsets, p))
        sys.exit(2)
    statics += 1
    has_night = has_day = has_layer = False
    for _ in range(numSubsets):
        if p + 44 > n:
            print("!! truncated subset header at %d" % p)
            sys.exit(2)
        p += 4 + 12 + 12 + 12               # sphere + aabbMin + aabbMax
        verts, faces = struct.unpack_from("<ii", blob, p)
        p += 8
        vb_off = p
        p += verts * 20
        p += faces * 6
        if p + 4 > n:
            print("!! truncated subset body at %d" % p)
            sys.exit(2)
        f0, f1 = blob[p], blob[p + 1]
        p += 2
        pathsize, = struct.unpack_from("<H", blob, p)
        p += 2
        name = blob[p:p + pathsize].split(b"\0")[0].decode("latin-1")
        p += pathsize

        subsets += 1
        if f1 & 0x1: uvctrl += 1
        if f1 & 0x2:
            night += 1
            has_night = True
            for v in range(verts):
                night_emissive[blob[vb_off + v * 20 + 8 + 3]] += 1
            night_tex[name] += 1
        if f1 & 0x6:
            if any(h in name.lower() for h in HERO_TEXTURES):
                hero_split[name] += 1
        if f1 & 0x4:
            day += 1
            has_day = True
        if f1 & 0x8:
            mul_layers += 1
            has_layer = True
        if f1 & ~0xF:
            unknown_bits[f1] += 1
    if has_night and has_day:
        paired_statics += 1
    if has_layer:
        layered_statics += 1

print("parsed %d statics / %d subsets, consumed %d of %d bytes" % (statics, subsets, p, n))
if p != n:
    print("!! %d trailing bytes - parser desync" % (n - p))
    sys.exit(2)
print("  uvCtrl (bit0)     : %d" % uvctrl)
print("  night-only (bit1) : %d" % night)
print("  day-only   (bit2) : %d" % day)
print("  statics with BOTH : %d" % paired_statics)
print("  multiply layers   : %d across %d statics" % (mul_layers, layered_statics))
if unknown_bits:
    print("  !! undefined flag bits set: %s" % dict(unknown_bits))
if night:
    lit = sum(c for e, c in night_emissive.items() if e >= 200)
    tot = sum(night_emissive.values())
    print("  night-variant vertex emissive: %d/%d bytes >= 200 (%.1f%%)" % (lit, tot, 100.0 * lit / tot))
    print("  top emissive bytes: %s" % night_emissive.most_common(5))
    print("  night-variant textures (base + every multiplied layer):")
    for k, v in night_tex.most_common(10):
        flag = ("  <== glow sheet (correct as a layer BASE)" if mul_layers
                else "  <== GLOW SHEET and no layers, reads white") \
               if any(g in k.lower() for g in GLOW_SHEETS) else ""
        print("     %4d  %s%s" % (v, k, flag))
    # With layered baking a glow sheet is legitimate - it is the BASE the layers multiply onto.
    # It is only a fault if NO layers were baked at all, i.e. the stack collapsed to the sheet.
    bad = 0 if mul_layers else sum(v for k, v in night_tex.items()
                                   if any(g in k.lower() for g in GLOW_SHEETS))
else:
    bad = 0
    print("  (no night variants: this is a pre-regen bake, or no GitD meshes are installed)")

if hero_split:
    print("  !! HERO subsets were day/night split - anim slots shift, ghostfence loses its glow:")
    for k, v in hero_split.most_common():
        print("     %4d  %s" % (v, k))
else:
    print("  hero meshes split: none (correct)")

if night and sum(c for e, c in night_emissive.items() if e >= 200) == 0:
    print("  !! night variants carry NO emissive - they would bake dark, not glowing")
    sys.exit(3)
if bad:
    print("  !! %d night subsets baked a glow sheet - texture slot fell back to base" % bad)
    sys.exit(3)
if hero_split:
    sys.exit(4)
