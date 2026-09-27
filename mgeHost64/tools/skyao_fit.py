#!/usr/bin/env python3
"""Sky-AO calibrator (tasks/forge-skyao-oracle.md O2).

Reads hdrdump/skyoracle_NNNN.{bin,json} written by the host's ground-truth tool:
  A = (sum vis, sum inCascade, sum N.z, frames)   B = (sum worldAbs.xyz, sum GPU model)
  C = (sum GTAO vis, sum AO armed, 0, frames)  [v2]                            + height map
and
  1. ports skyamb.h.fsl::skyAOVisibility to numpy and CHECKS the port against the GPU's own answer
     (B.w / frames) — a fit run on a port that disagrees with the shader fits the wrong model;
  2. scores any model (knobs or a variant) against V_true = sum vis / sum inCascade, by bucket;
  3. sweeps knobs;
  4. writes side-by-side PNGs (oracle | model | error).

Usage:
  skyao_fit.py DUMP check                 port check + score of the knobs the run used
  skyao_fit.py DUMP images [--variant v]  oracle/model/error PNGs next to the dump
  skyao_fit.py DUMP sweep                 knob sweep, best by floor-weighted error
DUMP = a number (hdrdump/skyoracle_NNNN) or a path stem.
"""
import json
import os
import struct
import sys
import zlib

import numpy as np

ROOT = "/mnt/c/mgem/morrowind64/hdrdump"
DIRS = np.array([[0.98769, 0.15643], [0.15643, 0.98769], [-0.89101, 0.45399],
                 [-0.70711, -0.70711], [0.45399, -0.89101]], np.float32)   # skyamb.h.fsl's five


# ─── loading ────────────────────────────────────────────────────────────────────────────────────
def load(dump):
    stem = dump if not dump.isdigit() else os.path.join(ROOT, "skyoracle_%04d" % int(dump))
    stem = stem[:-4] if stem.endswith((".bin", ".json")) else stem
    meta = json.load(open(stem + ".json"))
    buf = open(stem + ".bin", "rb").read()
    ver = {b"SKYORCL1": 1, b"SKYORCL2": 2, b"SKYORCL3": 3}.get(buf[:8])
    assert ver, "bad magic"
    W, H, mapRes, K, drawn = struct.unpack("<5I", buf[8:28])
    hf = struct.unpack("<14f", buf[28:84])
    o = 84
    A = np.frombuffer(buf, "<f4", W * H * 4, o).reshape(H, W, 4); o += W * H * 16
    B = np.frombuffer(buf, "<f4", W * H * 4, o).reshape(H, W, 4); o += W * H * 16
    C = None
    if ver >= 2:   # + GTAO: (vis, armed, 0, frames)
        C = np.frombuffer(buf, "<f4", W * H * 4, o).reshape(H, W, 4); o += W * H * 16
    hmap = np.frombuffer(buf, "<f4", mapRes * mapRes, o).reshape(mapRes, mapRes) if mapRes else None
    o += mapRes * mapRes * 4
    # v3: the LOW layer as stored (-lowest static surface; -30000 sentinel = no static), turned into
    # the lowest surface itself: +30000 where no static covers the texel.
    hlow = -np.frombuffer(buf, "<f4", mapRes * mapRes, o).reshape(mapRes, mapRes) if (ver >= 3 and mapRes) else None
    frames = A[..., 3]
    have = frames > 0.5
    fr = np.where(have, frames, 1.0)
    inc = A[..., 1]
    d = dict(meta=meta, W=W, H=H, K=K, drawn=drawn, hmap=hmap, hlow=hlow,
             origin=np.array(hf[0:2], np.float32), extent=hf[2], eye=np.array(hf[3:6]),
             knobs=dict(strength=hf[6], inner=hf[7], outer=hf[8], taps=hf[9],
                        overhang=hf[10], overhangFade=hf[11], overhangFloor=hf[12],
                        blur=float(meta.get("skyAO", {}).get("blur", 0)) * 1.5),
             have=have,
             # V_true over the directions each pixel was INSIDE a cascade for.
             vtrue=np.where(inc > 0.5, A[..., 0] / np.maximum(inc, 1.0), np.nan),
             coverage=inc / fr,           # fraction of directions that saw this pixel
             nz=A[..., 2] / fr,
             wa=B[..., :3] / fr[..., None],
             gpu=B[..., 3] / fr,
             # GTAO visibility the ambient ALSO multiplies by (1 where unknown / AO off).
             gtao=(C[..., 0] / fr) if C is not None else np.ones((H, W), np.float32),
             ao_on=bool(C is not None and (C[..., 1] > 0.5).any()))
    return stem, d


# ─── the model, ported ──────────────────────────────────────────────────────────────────────────
def _bilinear(hmap, u, v):
    """D3D bilinear, clamp addressing, texel centres at (i + 0.5) / N."""
    n = hmap.shape[0]
    x = u * n - 0.5; y = v * n - 0.5
    x0 = np.floor(x); y0 = np.floor(y)
    fx = (x - x0).astype(np.float32); fy = (y - y0).astype(np.float32)
    x0 = x0.astype(np.int64); y0 = y0.astype(np.int64)
    xa = np.clip(x0, 0, n - 1); xb = np.clip(x0 + 1, 0, n - 1)
    ya = np.clip(y0, 0, n - 1); yb = np.clip(y0 + 1, 0, n - 1)
    h00 = hmap[ya, xa]; h10 = hmap[ya, xb]; h01 = hmap[yb, xa]; h11 = hmap[yb, xb]
    return (h00 * (1 - fx) + h10 * fx) * (1 - fy) + (h01 * (1 - fx) + h11 * fx) * fy


def _point(hmap, u, v):
    n = hmap.shape[0]
    x = np.clip(np.floor(u * n).astype(np.int64), 0, n - 1)
    y = np.clip(np.floor(v * n).astype(np.int64), 0, n - 1)
    return hmap[y, x]


_BLUR_CACHE = {}


def _blurred(d, sigma, radius=4):
    """EXACTLY skyheightblur.comp.fsl: a (2R+1)^2 Gaussian, clamp addressing, EMPTY texels (sentinel
    -30000) excluded from the weights, and an empty texel stays empty. Cached per dump."""
    key = (id(d), sigma, radius)
    if key not in _BLUR_CACHE:
        h = d["hmap"].astype(np.float64)
        valid = h > -20000
        hv = np.where(valid, h, 0.0)
        n = h.shape[0]
        num = np.zeros_like(h); den = np.zeros_like(h)
        for dy in range(-radius, radius + 1):
            ys = np.clip(np.arange(n) + dy, 0, n - 1)
            for dx in range(-radius, radius + 1):
                xs = np.clip(np.arange(n) + dx, 0, n - 1)
                w = np.float32(np.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma)))
                num += w * hv[np.ix_(ys, xs)]
                den += w * valid[np.ix_(ys, xs)]
        out = np.where(valid, num / np.maximum(den, 1e-12), d["hmap"])
        _BLUR_CACHE[key] = out.astype(np.float32)
    return _BLUR_CACHE[key]


def model(d, wa, knobs, variant=None):
    """skyAOVisibility at absolute world positions wa (N,3).

    variant (dict, all optional; {} / None = the SHIPPED shader, bit-for-bit in intent):
      blur    = sigma in texels: march a Gaussian-smoothed map (the raster staircase -> contours)
      pcfself = True: the self "covered" test is filtered (bilinear weights over the 2x2 texels'
                BOOLEAN covered-ness) instead of point sampled — PCF, not blended heights
      enclose = D (world u): a COVERED receiver's direction only keeps its sky if the march finds an
                EXIT (a column below the receiver) within D; the trust lerp toward open is dropped
      near    = first march distance for `enclose`'s exit search (default 16 u)
    """
    v = variant or {}
    if v.get("interval"):
        return model_interval(d, wa, knobs, v)
    if v.get("hybrid") is not None:
        # OPTION 2: the two-layer bitmask ONLY where this receiver's own column has a static
        # floating above it (lowest static surface > me + gap), faded in over `hfade`; the fitted
        # single-layer model everywhere else. Stacked statics mostly fail the test (the receiver sits
        # ON a static, so its own column's lowest surface is below it) and keep the fit.
        base = model(d, wa, knobs, {k2: v2 for k2, v2 in v.items() if k2 not in ("hybrid", "hfade", "ik")})
        ik = dict(knobs, **v.get("ik", {}))
        two = model_interval(d, wa, ik, {"interval": 1})
        inv = 1.0 / d["extent"]
        uv = (wa[:, :2] - d["origin"]) * inv
        low = _point(d["hlow"], uv[:, 0], uv[:, 1])
        gap = low - wa[:, 2] - float(v["hybrid"])
        w = np.where(low < 20000, np.clip(gap / float(v.get("hfade", 128.0)), 0.0, 1.0), 0.0)
        return base + (two - base) * w
    sig = v.get("blur", knobs.get("blur", 0.0))
    hmap = _blurred(d, sig) if sig else d["hmap"]
    raw = d["hmap"]
    inv = 1.0 / d["extent"]
    uv = (wa[:, :2] - d["origin"]) * inv
    e = np.abs(uv - 0.5) * 2.0
    edge = np.clip((1.0 - np.maximum(e[:, 0], e[:, 1])) * 8.0, 0.0, 1.0)
    inner, outer = knobs["inner"], knobs["outer"]
    steps = int(max(knobs["taps"], 1.0))
    log_ratio = np.log2(max(outer / max(inner, 1.0), 1.0))
    my_h = wa[:, 2]
    if v.get("pcfself"):
        n = raw.shape[0]
        x = uv[:, 0] * n - 0.5; y = uv[:, 1] * n - 0.5
        x0 = np.floor(x); y0 = np.floor(y); fx = x - x0; fy = y - y0
        x0 = x0.astype(np.int64); y0 = y0.astype(np.int64)
        def cov(ix, iy):
            hh = raw[np.clip(iy, 0, n - 1), np.clip(ix, 0, n - 1)]
            a = hh - my_h
            return np.clip((a - knobs["overhang"]) / max(knobs["overhangFade"], 1.0), 0.0, 1.0)
        c = (cov(x0, y0) * (1 - fx) + cov(x0 + 1, y0) * fx) * (1 - fy) + \
            (cov(x0, y0 + 1) * (1 - fx) + cov(x0 + 1, y0 + 1) * fx) * fy
        covered = c
    else:
        above = _point(raw, uv[:, 0], uv[:, 1]) - my_h
        covered = np.clip((above - knobs["overhang"]) / max(knobs["overhangFade"], 1.0), 0.0, 1.0)
    trust = 1.0 - (1.0 - knobs["overhangFloor"]) * covered

    max_s = np.zeros((wa.shape[0], 5), np.float32)
    for k in range(steps):
        t = (k + 1.0) / steps
        dist = inner * 2.0 ** (t * log_ratio)
        r = dist * inv
        for j in range(5):
            h = _bilinear(hmap, uv[:, 0] + DIRS[j, 0] * r, uv[:, 1] + DIRS[j, 1] * r)
            dh = h - my_h
            max_s[:, j] = np.maximum(max_s[:, j], dh / np.sqrt(dh * dh + dist * dist))
    s = np.clip(max_s, 0.0, 1.0)
    vis_dir = 1.0 - s * s                                   # (N, 5)

    if v.get("enclose"):
        D = float(v["enclose"])
        near = float(v.get("near", 16.0))
        n_ex = 12
        lr = np.log2(max(outer / near, 1.0))
        exit_d = np.full((wa.shape[0], 5), np.inf, np.float32)
        for k in range(n_ex):
            dist = near * 2.0 ** (((k + 1.0) / n_ex) * lr)
            r = dist * inv
            for j in range(5):
                h = _bilinear(raw, uv[:, 0] + DIRS[j, 0] * r, uv[:, 1] + DIRS[j, 1] * r)
                open_ = (h - my_h) < knobs["overhang"]           # this column no longer roofs me
                exit_d[:, j] = np.where(np.isinf(exit_d[:, j]) & open_, dist, exit_d[:, j])
        w_exit = np.clip(1.0 - exit_d / D, 0.0, 1.0)            # inf -> 0: no way out, no sky
        # Blend by how covered the receiver is: open receivers keep the shipped wedge.
        vis_dir = vis_dir * (1.0 - covered[:, None]) + vis_dir * w_exit * covered[:, None]
        ao = vis_dir.mean(axis=1)
    else:
        ao = vis_dir.mean(axis=1)
        ao = 1.0 + (ao - 1.0) * trust
    return np.where(edge <= 0.0, 1.0, 1.0 + (ao - 1.0) * knobs["strength"] * edge)


def _texels(hmap, u, v):
    """The four bilinear texels and their weights (D3D centres, clamp)."""
    n = hmap.shape[0]
    x = u * n - 0.5; y = v * n - 0.5
    x0 = np.floor(x); y0 = np.floor(y)
    fx = (x - x0).astype(np.float32); fy = (y - y0).astype(np.float32)
    x0 = x0.astype(np.int64); y0 = y0.astype(np.int64)
    xa = np.clip(x0, 0, n - 1); xb = np.clip(x0 + 1, 0, n - 1)
    ya = np.clip(y0, 0, n - 1); yb = np.clip(y0 + 1, 0, n - 1)
    return [(ya, xa, (1 - fx) * (1 - fy)), (ya, xb, fx * (1 - fy)),
            (yb, xa, (1 - fx) * fy), (yb, xb, fx * fy)]


def _band_bits(lo, hi, my_h, dist, nbins):
    """Blocked bins (bool, N x nbins) for a column occupying [lo, hi] seen at horizontal `dist`.
    Bins are equal cosine-weighted sky mass: bin b spans sin^2(elevation) in [b, b+1) / nbins."""
    dh = hi - my_h
    dl = lo - my_h
    s_hi = np.where(dh > 0, dh * dh / (dh * dh + dist * dist), -1.0)
    s_lo = np.where(dl > 0, dl * dl / (dl * dl + dist * dist), 0.0)
    c = (np.arange(nbins, dtype=np.float32) + 0.5) / nbins
    return (c[None, :] >= s_lo[:, None]) & (c[None, :] <= s_hi[:, None])


def model_interval(d, wa, knobs, variant=None):
    """TWO-LAYER receiver: each column is terrain-or-static up to `top`, except that where a STATIC's
    lowest surface `low` is above the receiver the column is open below it (a roof, a deck, a cap).
    Per direction a visibility BITMASK over sin^2(elevation) accumulates the union of blocked bands.
    variant: nbins (32), ndirs (5 = the shipped odd set; 7/9 = rotated odd sets), filt ('vote'|'bilinear')."""
    v = variant or {}
    nb = int(v.get("nbins", 32))
    filt = v.get("filt", "vote")
    top, low = d["hmap"], d["hlow"]
    inv = 1.0 / d["extent"]
    uv = (wa[:, :2] - d["origin"]) * inv
    e = np.abs(uv - 0.5) * 2.0
    edge = np.clip((1.0 - np.maximum(e[:, 0], e[:, 1])) * 8.0, 0.0, 1.0)
    inner, outer = knobs["inner"], knobs["outer"]
    steps = int(max(knobs["taps"], 1.0))
    log_ratio = np.log2(max(outer / max(inner, 1.0), 1.0))
    my_h = wa[:, 2]
    nd = int(v.get("ndirs", 5))
    if nd == 5:
        dirs = DIRS
    else:
        a = np.deg2rad(9.0) + 2 * np.pi * np.arange(nd) / nd
        dirs = np.stack([np.cos(a), np.sin(a)], 1).astype(np.float32)
    vis = np.zeros(wa.shape[0], np.float32)
    for j in range(nd):
        mask = np.zeros((wa.shape[0], nb), bool)
        for k in range(steps):
            t = k / max(steps - 1, 1)                      # first tap AT inner, last AT outer
            dist = inner * 2.0 ** (t * log_ratio)
            r = dist * inv
            uu = uv[:, 0] + dirs[j, 0] * r; vv = uv[:, 1] + dirs[j, 1] * r
            if filt == "bilinear":
                tp = _bilinear(top, uu, vv)
                lw = _bilinear(np.where(low > 20000, -30000.0, low).astype(np.float32), uu, vv)
                mask |= _band_bits(lw, tp, my_h, dist, nb)
            else:
                acc = np.zeros((wa.shape[0], nb), np.float32)
                for (iy, ix, w) in _texels(top, uu, vv):
                    tp = top[iy, ix]
                    lw = low[iy, ix]
                    lw = np.where(lw > 20000, -1.0e9, lw)      # no static: solid from below
                    acc += w[:, None] * _band_bits(lw, tp, my_h, dist, nb)
                mask |= acc >= 0.5
        vis += 1.0 - mask.mean(axis=1)
    ao = vis / nd
    return np.where(edge <= 0.0, 1.0, 1.0 + (ao - 1.0) * knobs["strength"] * edge)


# ─── scoring ────────────────────────────────────────────────────────────────────────────────────
def masks(d):
    ok = d["have"] & (d["coverage"] > 0.9) & np.isfinite(d["vtrue"])
    floor = ok & (d["nz"] > 0.7)
    vt = d["vtrue"]
    return dict(all=ok, floor=floor,
                enclosed=floor & (vt < 0.15), partial=floor & (vt >= 0.15) & (vt < 0.7),
                open=floor & (vt >= 0.7))


def score(d, pred, label):
    m = masks(d)
    vt = d["vtrue"]
    print("  %-22s" % label, end="")
    for name in ("floor", "enclosed", "partial", "open"):
        s = m[name]
        if s.sum() == 0:
            print(" | %s n=0" % name, end=""); continue
        err = pred[s] - vt[s]
        print(" | %s n=%d bias %+.3f mae %.3f" % (name, s.sum(), err.mean(), np.abs(err).mean()), end="")
    print()


def eval_model(d, knobs, variant=None):
    ok = d["have"]
    pred = np.full(d["gpu"].shape, np.nan, np.float32)
    pred[ok] = model(d, d["wa"][ok], knobs, variant)
    return pred


# ─── sweep ──────────────────────────────────────────────────────────────────────────────────────
def subsample(d, step=4):
    """Floor pixels on a regular grid, plus their right/down neighbours for the roughness term."""
    m = masks(d)["floor"]
    H, W = m.shape
    ys, xs = np.mgrid[0:H - 1:step, 0:W - 1:step]
    ys = ys.ravel(); xs = xs.ravel()
    keep = m[ys, xs] & m[ys, xs + 1] & m[ys + 1, xs]
    ys, xs = ys[keep], xs[keep]
    # A neighbour across a depth edge is a different surface; its step is not the model's fault.
    wa = d["wa"]
    near = (np.linalg.norm(wa[ys, xs + 1] - wa[ys, xs], axis=1) < 64) & \
           (np.linalg.norm(wa[ys + 1, xs] - wa[ys, xs], axis=1) < 64)
    return ys[near], xs[near]


def fit_eval(d, sub, knobs, variant=None):
    ys, xs = sub
    pts = np.concatenate([d["wa"][ys, xs], d["wa"][ys, xs + 1], d["wa"][ys + 1, xs]])
    pr = model(d, pts, knobs, variant).reshape(3, -1)
    g = d["gtao"]
    tot = pr[0] * g[ys, xs]
    vt = d["vtrue"][ys, xs]
    err = tot - vt
    enc = vt < 0.15
    par = (vt >= 0.15) & (vt < 0.7)
    # roughness: how many more screen-space steps the MODEL takes than the oracle does
    dm = np.abs(pr[1] - pr[0]) + np.abs(pr[2] - pr[0])
    dv = np.abs(d["vtrue"][ys, xs + 1] - vt) + np.abs(d["vtrue"][ys + 1, xs] - vt)
    return dict(mae=np.abs(err).mean(), bias=err.mean(),
                enc=np.abs(err[enc]).mean() if enc.any() else np.nan,
                encb=err[enc].mean() if enc.any() else np.nan,
                par=np.abs(err[par]).mean() if par.any() else np.nan,
                rough=(dm.mean() - dv.mean()) * 100.0)


def sweep(dumps, variant=None):
    ds = [load(x)[1] for x in dumps]
    subs = [subsample(d) for d in ds]
    base = dict(ds[0]["knobs"])
    rows = []
    grid = [(i, o, t, ov, fl) for i in (32, 64, 128, 256, 445) for o in (1024, 2048, 4096, 8192)
            for t in (4, 8) for ov in (128, 2048) for fl in (0.25, 1.0)]
    for n, (i, o, t, ov, fl) in enumerate(grid):
        k = dict(base, inner=float(i), outer=float(o), taps=float(t), overhang=float(ov),
                 overhangFade=1024.0, overhangFloor=float(fl), strength=1.0)
        r = [fit_eval(d, sb, k, variant) for d, sb in zip(ds, subs)]
        rows.append((np.mean([x["mae"] for x in r]), k, r))
        if n % 20 == 0:
            print("  ... %d/%d" % (n, len(grid)), flush=True)
    rows.sort(key=lambda x: x[0])
    for i, d in enumerate(dumps):
        r0 = fit_eval(ds[i], subs[i], dict(ds[i]["knobs"]), variant)
        print("RUN KNOBS  dump %s: mae %.3f bias %+.3f | enclosed mae %.3f bias %+.3f | partial %.3f | rough %+.2f"
              % (d, r0["mae"], r0["bias"], r0["enc"], r0["encb"], r0["par"], r0["rough"]))
    print("BEST 12 (mean floor MAE over dumps %s):" % dumps)
    for m, k, r in rows[:12]:
        print("  mae %.3f | in %4.0f out %5.0f taps %.0f ovStart %4.0f floor %.2f | " % (
            m, k["inner"], k["outer"], k["taps"], k["overhang"], k["overhangFloor"]) +
            " ; ".join("bias %+.3f enc %.3f(%+.3f) par %.3f rough %+.2f" % (
                x["bias"], x["enc"], x["encb"], x["par"], x["rough"]) for x in r))
    return rows


# ─── images ─────────────────────────────────────────────────────────────────────────────────────
def png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\0" + rgb[y].tobytes() for y in range(h))
    def chunk(t, c):
        return struct.pack(">I", len(c)) + t + c + struct.pack(">I", zlib.crc32(t + c) & 0xffffffff)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def grey(v, step=2):
    g = (np.clip(np.nan_to_num(v, nan=0.0), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)[::step, ::step]
    return np.stack([g, g, g], -1)


def diverge(err, step=2):
    e = np.nan_to_num(err, nan=0.0)[::step, ::step]
    out = np.zeros(e.shape + (3,), np.uint8)
    out[..., 0] = (np.clip(e, 0, 0.5) * 510).astype(np.uint8)    # red = model too BRIGHT
    out[..., 2] = (np.clip(-e, 0, 0.5) * 510).astype(np.uint8)   # blue = model too DARK
    return out


def images(stem, d, pred, tag):
    top = np.concatenate([grey(d["vtrue"]), grey(pred)], 1)
    bot = np.concatenate([diverge(pred - d["vtrue"]), grey(d["nz"] > 0.7)], 1)
    path = "%s_%s.png" % (stem, tag)
    png(path, np.concatenate([top, bot], 0))
    print("wrote", path, "(TL oracle | TR model | BL error red=model brighter | BR floor mask)")


# ─── main ───────────────────────────────────────────────────────────────────────────────────────

# ─── sky-visibility maps (Twister-style; tasks/forge-skyao-oracle.md "TWISTER") ─────────────────
def load_svm(dump):
    """hdrdump/skyvis_NNNN.bin, written beside skyoracle_NNNN by the same run."""
    stem = dump if not dump.isdigit() else os.path.join(ROOT, "skyvis_%04d" % int(dump))
    buf = open(stem + ".bin", "rb").read()
    assert buf[:8] == b"SKYVIS01", "bad magic"
    K, res, drawn = struct.unpack("<3I", buf[8:20])
    ext, dh, ex, ey, ez = struct.unpack("<5f", buf[20:40])
    o = 40
    tiles = []
    for _ in range(K):
        f = struct.unpack("<19fI", buf[o:o + 80]); o += 80
        tiles.append(dict(vp=np.array(f[:16], np.float64).reshape(4, 4), eye=np.array(f[16:19], np.float64),
                          frame=f[19]))
    maps = np.frombuffer(buf, "<f4", K * res * res, o).reshape(K, res, res)
    return dict(K=K, res=res, drawn=drawn, ext=ext, dh=dh, eyeShadow=np.array([ex, ey, ez]),
                tiles=tiles, maps=maps)


def svm_visibility(sv, wa, bias=16.0, pcf=1, nz=None):
    """Fraction of the K maps that see each world point. wa: (N,3) absolute world positions.
    bias: world units along the direction. pcf: half-width in texels of a box of bilinear
    compare taps (0 = one bilinear tap). A point outside a map's box counts as seen."""
    res = sv["res"]
    vis = np.zeros(len(wa), np.float64)
    for t, m in zip(sv["tiles"], sv["maps"]):
        rel = wa.astype(np.float64) - t["eye"]
        clip = rel @ t["vp"][:3, :] + t["vp"][3, :]
        u = (clip[:, 0] + 1.0) * 0.5 * res - 0.5
        v = (1.0 - clip[:, 1]) * 0.5 * res - 0.5
        z = clip[:, 2] + bias / (2.0 * sv["dh"])
        acc = np.zeros(len(wa)); n = 0
        for dy in range(-pcf, pcf + 1):
            for dx in range(-pcf, pcf + 1):
                uu, vv = u + dx, v + dy
                x0 = np.floor(uu).astype(np.int64); y0 = np.floor(vv).astype(np.int64)
                fx = uu - x0; fy = vv - y0
                def lit(xi, yi):
                    inside = (xi >= 0) & (xi < res) & (yi >= 0) & (yi < res)
                    d = m[np.clip(yi, 0, res - 1), np.clip(xi, 0, res - 1)]
                    return np.where(inside, z >= d, True).astype(np.float64)
                acc += ((1 - fx) * (1 - fy) * lit(x0, y0) + fx * (1 - fy) * lit(x0 + 1, y0)
                        + (1 - fx) * fy * lit(x0, y0 + 1) + fx * fy * lit(x0 + 1, y0 + 1))
                n += 1
        vis += acc / n
    return vis / len(sv["tiles"])


def svm_score(d, sv, bias=16.0, pcf=1, step=4):
    ys, xs = subsample(d, step)
    pts = np.concatenate([d["wa"][ys, xs], d["wa"][ys, xs + 1], d["wa"][ys + 1, xs]])
    pr = svm_visibility(sv, pts, bias, pcf).reshape(3, -1)
    vt = d["vtrue"][ys, xs]
    err = pr[0] - vt
    enc = vt < 0.15
    par = (vt >= 0.15) & (vt < 0.7)
    dm = np.abs(pr[1] - pr[0]) + np.abs(pr[2] - pr[0])
    dv = np.abs(d["vtrue"][ys, xs + 1] - vt) + np.abs(d["vtrue"][ys + 1, xs] - vt)
    return dict(mae=np.abs(err).mean(), bias=err.mean(),
                enc=np.abs(err[enc]).mean() if enc.any() else np.nan,
                encb=err[enc].mean() if enc.any() else np.nan,
                par=np.abs(err[par]).mean() if par.any() else np.nan,
                rough=(dm.mean() - dv.mean()) * 100.0)


def main():
    stem, d = load(sys.argv[1])
    cmd = sys.argv[2] if len(sys.argv) > 2 else "check"
    print("dump %s: %dx%d, %d/%d directions, knobs %s" % (os.path.basename(stem), d["W"], d["H"],
                                                         d["drawn"], d["K"], d["knobs"]))
    if cmd in ("check", "images"):
        pred = eval_model(d, d["knobs"])
        ok = d["have"]
        diff = np.abs(pred[ok] - d["gpu"][ok])
        print("PORT CHECK vs GPU channel: max |diff| %.5f, p99 %.5f, mean %.6f over %d px"
              % (diff.max(), np.percentile(diff, 99), diff.mean(), ok.sum()))
        score(d, d["gpu"], "GPU model (run knobs)")
        score(d, pred, "numpy port")
        score(d, np.ones_like(pred), "no sky AO (vis=1)")
        print("  GTAO armed: %s, floor GTAO median %.3f" % (d["ao_on"], np.median(d["gtao"][masks(d)["floor"]])))
        score(d, d["gpu"] * d["gtao"], "GPU model x GTAO")
        if cmd == "images":
            images(stem, d, d["gpu"] * d["gtao"], "gpu")
    elif cmd == "svm":   # skyao_fit.py DUMP svm [bias pcf]: score skyvis_DUMP against the oracle
        sv = load_svm(sys.argv[1])
        print("svm K=%d res=%d ext=%.0f texel=%.2f | eye check: tile0 %s vs oracle %s" % (
            sv["K"], sv["res"], sv["ext"], 2 * sv["ext"] / sv["res"], sv["tiles"][0]["eye"], sv["eyeShadow"]))
        grid = [(float(sys.argv[3]), int(sys.argv[4]))] if len(sys.argv) > 4 else \
               [(b, p) for b in (4.0, 16.0, 48.0) for p in (0, 1, 2)]
        for b, p in grid:
            r = svm_score(d, sv, b, p)
            print("  bias %5.1f pcf %d | mae %.3f bias %+.3f | enclosed %.3f (%+.3f) | partial %.3f | rough %+.1f"
                  % (b, p, r["mae"], r["bias"], r["enc"], r["encb"], r["par"], r["rough"]))
    elif cmd == "sweep":
        sweep([sys.argv[1]] + sys.argv[3:])


if __name__ == "__main__":
    main()
