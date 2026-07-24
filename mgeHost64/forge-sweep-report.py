#!/usr/bin/env python3
"""Parse a forge-scene-sweep output directory into one comparison table.

Reads the per-scene mgeHost64.log / mgeXE.log copies the sweep left behind and reports the MEDIAN of
each scene's heartbeat samples (median, not mean: the first heartbeat after a load carries the
warm-up spike and would drag a mean).

The table is meant to be read as deltas over the two floor scenes (justsky = no world geometry,
justwall = minimal interior). A phase that costs the same in the floor as in a heavy scene is fixed
per-frame overhead and optimising it pays everywhere; a phase that only appears in one scene class
is scene-driven and its priority depends on how often that class is actually played.
"""
import os
import re
import sys
from statistics import median

# Ordered floors-first; must match forge-scene-sweep.sh.
SCENES = ["vjustsky", "vjustwall", "vlightdense", "vmanylights", "vdensecity", "vheavydistantland"]

GPU_SPLIT = re.compile(
    r"gpu split: cull=(?P<cull>-?[\d.]+) prepass=(?P<prepass>-?[\d.]+) shadow=(?P<shadow>-?[\d.]+) "
    r"\(st=(?P<shst>-?[\d.]+) dyn=(?P<shdyn>-?[\d.]+)\) postdepth=(?P<postdepth>-?[\d.]+) "
    r"\(lin=(?P<lin>-?[\d.]+) ao=(?P<ao>-?[\d.]+) mask=(?P<mask>-?[\d.]+)\) "
    r"reflect=(?P<reflect>-?[\d.]+) color=(?P<color>-?[\d.]+) water=(?P<water>-?[\d.]+) "
    r"resolve=(?P<resolve>-?[\d.]+) ms.*?shadowCasters=(?P<casters>\d+).*?"
    r"nearTris=(?P<tris>-?[\d.]+)M atDraws=(?P<atdraws>\d+)")

HOST_SPLIT = re.compile(
    r"host split: setup=(?P<setup>-?[\d.]+) cull=(?P<hcull>-?[\d.]+) record=(?P<record>-?[\d.]+) "
    r"gpu=(?P<gpu>-?[\d.]+) gpuOverlap=(?P<overlap>-?[\d.]+) post=(?P<post>-?[\d.]+) "
    r"total=(?P<total>-?[\d.]+)ms.*?survivors=(?P<survivors>\d+)")

COLOR_SUB = re.compile(r"nearfrox=[\d.]+\((?P<froxon>\w+),n=(?P<lights>\d+)\)")

POOLS = re.compile(
    r"pools: arenaVB=(?P<vb>\d+)/(?P<vbtot>\d+) MB \((?P<vbpct>\d+)%.*?frag=(?P<vbfrag>\d+)\) "
    r"arenaIB=(?P<ib>\d+)/(?P<ibtot>\d+) MB.*?grows=(?P<grows>\d+) \| skipped=(?P<skipped>\d+)")

TEXRES = re.compile(
    r"tex residency: slots=(?P<slots>\d+)/(?P<cap>\d+) resident=(?P<resident>\d+) \| "
    r"recycles=(?P<recycles>\d+) thrash=(?P<thrash>\d+)")

CLIENT_HB = re.compile(r"\[hb\] \d+ frames avg:.*? dt=(?P<dt>-?[\d.]+) ")


def collect(path, rx, keys):
    """Median of each named field across every match in the file."""
    if not os.path.exists(path):
        return None
    vals = {k: [] for k in keys}
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = rx.search(line)
            if m:
                for k in keys:
                    vals[k].append(float(m.group(k)))
    if not vals[keys[0]]:
        return None
    return {k: median(v) for k, v in vals.items()}, len(vals[keys[0]])


def fmt(v, nd=2):
    return "-" if v is None else f"{v:.{nd}f}"


def main(outdir):
    rows = []
    for tag in SCENES:
        host = os.path.join(outdir, f"{tag}.host.log")
        client = os.path.join(outdir, f"{tag}.client.log")
        gpu = collect(host, GPU_SPLIT, ["cull", "prepass", "shadow", "postdepth", "lin", "mask",
                                        "reflect", "color", "water", "resolve", "casters",
                                        "tris", "atdraws"])
        hostp = collect(host, HOST_SPLIT, ["setup", "hcull", "record", "gpu", "overlap", "post",
                                           "total", "survivors"])
        lights = collect(host, COLOR_SUB, ["lights"])
        pools = collect(host, POOLS, ["vb", "vbtot", "vbpct", "vbfrag", "ib", "grows", "skipped"])
        tex = collect(client, TEXRES, ["slots", "cap", "resident", "recycles", "thrash"])
        dt = collect(client, CLIENT_HB, ["dt"])
        rows.append((tag, gpu, hostp, lights, pools, tex, dt))

    def cell(bundle, key, nd=2):
        return "-" if not bundle else fmt(bundle[0][key], nd)

    print("## Host GPU phases (median ms per frame)\n")
    hdr = ("| scene | n | lights | casters | tris M | GPU tot | mask | prepass | color | reflect "
           "| water | shadow | cull | resolve |")
    print(hdr)
    print("|" + "---|" * 14)
    for tag, gpu, hostp, lights, _, _, _ in rows:
        n = gpu[1] if gpu else 0
        print(f"| {tag} | {n} | {cell(lights,'lights',0)} | {cell(gpu,'casters',0)} | "
              f"{cell(gpu,'tris')} | {cell(hostp,'gpu')} | {cell(gpu,'mask')} | "
              f"{cell(gpu,'prepass')} | {cell(gpu,'color')} | {cell(gpu,'reflect')} | "
              f"{cell(gpu,'water')} | {cell(gpu,'shadow')} | {cell(gpu,'cull')} | "
              f"{cell(gpu,'resolve')} |")

    print("\n## Host CPU split (median ms) + client frame\n")
    print("| scene | survivors | setup | cull | record | gpuWait | overlap | post | total | client dt |")
    print("|" + "---|" * 10)
    for tag, _, hostp, _, _, _, dt in rows:
        print(f"| {tag} | {cell(hostp,'survivors',0)} | {cell(hostp,'setup')} | "
              f"{cell(hostp,'hcull')} | {cell(hostp,'record')} | {cell(hostp,'gpu')} | "
              f"{cell(hostp,'overlap')} | {cell(hostp,'post')} | {cell(hostp,'total')} | "
              f"{cell(dt,'dt')} |")

    print("\n## Pools (H2 occupancy)\n")
    print("| scene | arenaVB MB | % | frag | arenaIB MB | grows | skipped | tex slots | resident | recycles | thrash |")
    print("|" + "---|" * 11)
    for tag, _, _, _, pools, tex, _ in rows:
        vb = "-" if not pools else f"{pools[0]['vb']:.0f}/{pools[0]['vbtot']:.0f}"
        slots = "-" if not tex else f"{tex[0]['slots']:.0f}/{tex[0]['cap']:.0f}"
        print(f"| {tag} | {vb} | {cell(pools,'vbpct',0)} | {cell(pools,'vbfrag',0)} | "
              f"{cell(pools,'ib',0)} | {cell(pools,'grows',0)} | {cell(pools,'skipped',0)} | "
              f"{slots} | {cell(tex,'resident',0)} | {cell(tex,'recycles',0)} | "
              f"{cell(tex,'thrash',0)} |")

    # The floor is the whole point of the two baseline saves — state it explicitly rather than
    # leaving every reader to subtract by hand.
    floor = next((r for r in rows if r[0] == "vjustsky" and r[1]), None)
    if floor:
        print("\n## Delta over the empty-sky floor (host GPU ms)\n")
        print("| scene | GPU tot | d mask | d prepass | d color |")
        print("|" + "---|" * 5)
        fg, fh = floor[1][0], (floor[2][0] if floor[2] else None)
        for tag, gpu, hostp, _, _, _, _ in rows:
            if not gpu or tag == "vjustsky":
                continue
            dgpu = (hostp[0]["gpu"] - fh["gpu"]) if (hostp and fh) else None
            print(f"| {tag} | {fmt(dgpu)} | {fmt(gpu[0]['mask']-fg['mask'])} | "
                  f"{fmt(gpu[0]['prepass']-fg['prepass'])} | {fmt(gpu[0]['color']-fg['color'])} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/forge-sweep")
