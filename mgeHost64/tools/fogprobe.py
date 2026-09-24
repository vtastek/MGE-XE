#!/usr/bin/env python3
"""fogprobe — measurements over hdrdump EXRs for the fog rewrite (tasks/forge-atmosphere.md S3.0).

Every dump the host writes (numpad 1 or the dumpAtFrame knob) now carries a JSON sidecar with the
camera it was rasterised with (forgerender.cpp hdrDumpIfArmed). This tool turns the picture + that
camera into the numbers S3 is judged by, so "the horizon line follows the eye" or "the far edge is
half fogged" become rows and ratios instead of impressions.

  fogprobe.py geom    DUMP                  where the eye plane, the world edge and the sun land
  fogprobe.py skyline DUMP [--png OUT]      the sky -> ground step along the skyline, per column
  fogprobe.py diff    A B [--rect x0 y0 x1 y1] [--png OUT]
                                            what changed between two dumps: row profile (lines),
                                            radial profile around the sun (lobes)

DUMP is a number (52), a stem (mge_0052) or a path; the default directory is the dev install's
hdrdump. Radiance is LINEAR scene radiance, pre-exposure; RGB is premultiplied by coverage (alpha),
so every read divides by A where A > 0 (the log line's own warning).

The EXR reader handles exactly what the host writes: uncompressed half RGBA scanlines, planes
stored alphabetically (A, B, G, R) — see memory project_forge_hdr_exr_dump.
"""
import argparse
import json
import math
import os
import struct
import sys

import numpy as np

DEFAULT_DIR = "/mnt/c/mgem/morrowind64/hdrdump"


# ─── I/O ────────────────────────────────────────────────────────────────────────────────────────────

def resolve(dump, root):
    if os.path.exists(dump):
        base = dump
    elif dump.isdigit():
        base = os.path.join(root, "mge_%04d" % int(dump))
    else:
        base = os.path.join(root, dump)
    stem = base[:-4] if base.endswith((".exr", ".tga", ".json")) else base
    return stem


def read_exr(path):
    buf = open(path, "rb").read()
    i = 8
    hdr = {}
    while True:
        j = buf.index(b"\0", i); name = buf[i:j].decode(); i = j + 1
        if not name:
            break
        j = buf.index(b"\0", i); i = j + 1
        size = struct.unpack("<i", buf[i:i + 4])[0]; i += 4
        hdr[name] = buf[i:i + size]; i += size
    if hdr.get("compression", b"\0") != b"\0":
        sys.exit("fogprobe: only uncompressed EXRs (what the host writes) are supported")
    dw = struct.unpack("<4i", hdr["dataWindow"])
    W = dw[2] - dw[0] + 1; H = dw[3] - dw[1] + 1
    off = np.frombuffer(buf[i:i + 8 * H], dtype="<u8")
    img = np.zeros((H, W, 4), np.float32)
    for y in range(H):
        o = int(off[y]); ly, sz = struct.unpack("<ii", buf[o:o + 8])
        img[ly - dw[1]] = np.frombuffer(buf[o + 8:o + 8 + sz], dtype="<f2").reshape(4, W).T
    a = img[..., 0]
    rgb = img[..., [3, 2, 1]]
    # Un-premultiply where there is coverage; leave zero-coverage texels as they are.
    cov = np.where(a > 1e-4, a, 1.0)[..., None]
    return rgb / cov, a


def load(dump, root):
    stem = resolve(dump, root)
    rgb, a = read_exr(stem + ".exr")
    cam = None
    if os.path.exists(stem + ".json"):
        cam = json.load(open(stem + ".json"))
    return os.path.basename(stem), rgb, a, cam


def lum(rgb):
    return rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)


# ─── CAMERA ─────────────────────────────────────────────────────────────────────────────────────────

class Camera:
    """The sidecar's matrix: D3D row-vector convention, clip = [x y z w] * M, positions relative to
    eyeAbs. A DIRECTION is w = 0 (rotation rows only), a POINT is w = 1."""

    def __init__(self, cam, W, H):
        self.M = np.array(cam["viewProj"], np.float64).reshape(4, 4)
        self.eye = np.array(cam["eyeAbs"], np.float64)
        self.W, self.H = W, H
        self.cam = cam

    def project(self, v, w):
        c = np.append(np.asarray(v, np.float64), w) @ self.M
        if c[3] <= 1e-9:
            return None
        nx, ny = c[0] / c[3], c[1] / c[3]
        return ((nx + 1.0) * 0.5 * self.W, (1.0 - ny) * 0.5 * self.H)

    def _ring_rows(self, make):
        rows = []
        for k in range(720):
            phi = math.radians(k * 0.5)
            p = make(phi)
            if p and 0.0 <= p[0] < self.W:
                rows.append(p[1])
        return float(np.median(rows)) if rows else None

    def eye_row(self):
        """The eye's horizontal plane on screen (MW has no roll, so it is one row)."""
        return self._ring_rows(lambda f: self.project((math.cos(f), math.sin(f), 0.0), 0.0))

    def elevation_row(self, deg):
        e = math.radians(deg)
        return self._ring_rows(lambda f: self.project(
            (math.cos(f) * math.cos(e), math.sin(f) * math.cos(e), math.sin(e)), 0.0))

    def edge_row(self, dist, ground_z=0.0):
        """Where the world's visible edge lands: ground level `ground_z` at horizontal distance `dist`."""
        dz = ground_z - self.eye[2]
        return self._ring_rows(lambda f: self.project(
            (dist * math.cos(f), dist * math.sin(f), dz), 1.0))

    def px_per_deg(self):
        a, b = self.elevation_row(-1.0), self.elevation_row(1.0)
        return None if a is None or b is None else (a - b) / 2.0

    def sun_px(self):
        s = self.cam.get("toSun")
        return self.project(s, 0.0) if s else None


def need_cam(name, cam):
    if cam is None:
        sys.exit("fogprobe: %s has no JSON sidecar — it predates the S3.0 host build; re-dump it" % name)


def row_to_elev(camera, row):
    eye, ppd = camera.eye_row(), camera.px_per_deg()
    return None if eye is None or not ppd else (eye - row) / ppd


# ─── COMMANDS ───────────────────────────────────────────────────────────────────────────────────────

def cmd_geom(args):
    name, rgb, a, cam = load(args.dump, args.dir)
    need_cam(name, cam)
    H, W = rgb.shape[:2]
    c = Camera(cam, W, H)
    fog = cam.get("fog", [0, 0])
    dist = args.dist if args.dist else fog[1]
    eye, edge, ppd, sun = c.eye_row(), c.edge_row(dist, args.ground), c.px_per_deg(), c.sun_px()
    print("%s  %dx%d  weather=%s" % (name, W, H, cam.get("weather")))
    print("  eye abs z %.1f | fog start/end %.0f / %.0f units" % (c.eye[2], fog[0], fog[1]))
    print("  px per degree (vertical, screen centre): %.2f" % ppd if ppd else "  px/deg: n/a")
    print("  EYE-PLANE row:  %s" % ("%.1f" % eye if eye is not None else "off screen"))
    if edge is not None and eye is not None and ppd:
        print("  WORLD EDGE row: %.1f  (ground z=%.0f at %.0f units) -> %.1f px = %.2f deg BELOW the eye plane"
              % (edge, args.ground, dist, edge - eye, (edge - eye) / ppd))
    else:
        print("  WORLD EDGE row: off screen")
    print("  SUN: %s" % ("(%.0f, %.0f)" % sun if sun else "behind the camera"))
    return 0


def cmd_skyline(args):
    """Per column: the sharpest log-luminance step in a band around the horizon, and the ratio of the
    radiance just below it to just above it. 1.0 = the ground melts into the sky exactly; the S3 goal
    is >= 0.95. Columns with no step sharper than --min-step are left out (a tree, a flat sky)."""
    name, rgb, a, cam = load(args.dump, args.dir)
    H, W = rgb.shape[:2]
    L = np.log(np.maximum(lum(rgb), 1e-7))
    if args.band:
        y0, y1 = args.band
    elif cam:
        c = Camera(cam, W, H)
        eye, ppd = c.eye_row(), c.px_per_deg() or 20.0
        edge = c.edge_row(cam.get("fog", [0, 0])[1] or 1e6, args.ground)
        lo = min(v for v in (eye, edge) if v is not None) if (eye or edge) else H * 0.4
        hi = max(v for v in (eye, edge) if v is not None) if (eye or edge) else H * 0.6
        y0, y1 = int(lo - 3 * ppd), int(hi + 3 * ppd)
    else:
        y0, y1 = int(H * 0.3), int(H * 0.7)
    y0, y1 = max(y0, 6), min(y1, H - 6)
    k = args.gap
    dy = L[y0 + 1:y1 + 1] - L[y0:y1]                 # row-to-row step
    rows, ratios, textures = [], [], []
    for x in range(0, W, args.stride):
        col = dy[:, x]
        j = int(np.argmax(np.abs(col)))
        if abs(col[j]) < args.min_step:
            continue
        y = y0 + j
        above = rgb[y - k - 2:y - k + 1, x].mean(axis=0)
        below = rgb[y + k:y + k + 3, x].mean(axis=0)
        la, lb = lum(above[None])[0], lum(below[None])[0]
        if la <= 0:
            continue
        rows.append(y)
        ratios.append(lb / la)
        # Texture just below the step, across a few neighbouring columns: a fogged far edge is smooth,
        # a near tree or rock is not. Row-to-row log steps, RMS.
        xs0, xs1 = max(x - 2, 0), min(x + 3, W)
        blk = L[y + k + 1:y + k + 12, xs0:xs1]
        textures.append(float(np.sqrt(np.mean(np.diff(blk, axis=0) ** 2))) if blk.shape[0] > 1 else 1.0)
    if not ratios:
        print("%s: no skyline step found in rows %d..%d" % (name, y0, y1))
        return 1
    r = np.array(ratios)
    rows = np.array(rows)
    tex = np.array(textures)
    print("%s  skyline band rows %d..%d, %d/%d columns stepped" % (name, y0, y1, len(r), len(range(0, W, args.stride))))

    def report(label, sel):
        if not sel.any():
            print("  %s: none" % label)
            return
        rr = r[sel]
        print("  %s (%d cols): step row median %.0f | ground/sky median %.3f  p10 %.3f  p90 %.3f  min %.3f"
              % (label, sel.sum(), np.median(rows[sel]), np.median(rr), np.percentile(rr, 10),
                 np.percentile(rr, 90), rr.min()))

    # ⚠ THE FAR EDGE IS THE QUESTION, NOT EVERY SILHOUETTE. A near tree against the sky is SUPPOSED to
    # step; only the world's edge (geometry at the draw distance) should melt into the sky. So steps
    # within --edge-win degrees of the predicted edge row are the metric, the rest are reported apart.
    if cam:
        c = Camera(cam, W, H)
        ppd = c.px_per_deg() or 20.0
        edge = c.edge_row(args.edge_dist or cam.get("fog", [0, 0])[1] or 1e6, args.ground)
        if edge is not None:
            far = (np.abs(rows - edge) <= args.edge_win * ppd) & (tex <= args.max_texture)
            print("  predicted world-edge row %.0f (window +-%.1f deg, texture below <= %.3f)"
                  % (edge, args.edge_win, args.max_texture))
            report("FAR EDGE   (goal >= 0.95)", far)
            report("silhouettes (near objects, expected to step)", ~far)
            if far.any():
                el = row_to_elev(c, float(np.median(rows[far])))
                if el is not None:
                    print("  the far edge sits %.2f deg %s the eye plane" % (abs(el), "above" if el > 0 else "below"))
        else:
            report("all steps", np.ones_like(r, dtype=bool))
    else:
        report("all steps (no sidecar: cannot separate the far edge)", np.ones_like(r, dtype=bool))
    if args.png:
        from PIL import Image
        img = np.clip(np.power(np.clip(rgb / max(np.percentile(lum(rgb), 99.5), 1e-6), 0, 1), 1 / 2.2) * 255, 0, 255).astype(np.uint8)
        mark = np.array(img)
        for x in range(0, W, args.stride):
            col = dy[:, x]
            j = int(np.argmax(np.abs(col)))
            if abs(col[j]) < args.min_step:
                continue
            y = y0 + j
            mark[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = (255, 0, 255)
        Image.fromarray(mark).save(args.png)
        print("  marked -> %s" % args.png)
    return 0


def cmd_diff(args):
    """A - B, relative to B. The ROW profile finds lines (a fog rule keyed on elevation draws one row
    across everything); the RADIAL profile around the sun finds lobes."""
    na, ra, aa, cam = load(args.a, args.dir)
    nb, rb, ab, _ = load(args.b, args.dir)
    H, W = ra.shape[:2]
    La, Lb = lum(ra), lum(rb)
    rel = (La - Lb) / np.maximum(Lb, 1e-6)
    x0, y0, x1, y1 = args.rect if args.rect else (0, 0, W, H)
    region = rel[y0:y1, x0:x1]
    print("%s - %s  (relative to %s), rect %s" % (na, nb, nb, (x0, y0, x1, y1)))
    print("  mean %+.4f | p1 %+.4f | p99 %+.4f | max |d| %.4f"
          % (region.mean(), np.percentile(region, 1), np.percentile(region, 99), np.abs(region).max()))
    prof = np.median(region, axis=1)                        # per row, robust to texture
    kern = np.ones(5) / 5.0
    sm = np.convolve(prof, kern, mode="same")
    d2 = np.abs(np.convolve(sm, [1, -2, 1], mode="same"))
    d2[:4] = d2[-4:] = 0
    jr = int(np.argmax(d2))
    print("  row profile: sharpest CORNER at row %d (|d2| %.5f)" % (y0 + jr, d2[jr]))
    if cam:
        c = Camera(cam, W, H)
        eye, ppd = c.eye_row(), c.px_per_deg()
        if eye is not None and ppd:
            print("  ...which is %.2f deg %s the eye-plane row %.0f"
                  % (abs((eye - (y0 + jr)) / ppd), "above" if y0 + jr < eye else "below", eye))
        sun = c.sun_px()
        if sun:
            yy, xx = np.mgrid[0:H, 0:W]
            rpx = np.hypot(xx - sun[0], yy - sun[1])
            print("  radial profile around the sun at (%.0f, %.0f), mean relative diff by ring:" % sun)
            for r0, r1 in ((0, 50), (50, 100), (100, 200), (200, 400), (400, 800)):
                m = (rpx >= r0) & (rpx < r1)
                if m.any():
                    print("    %4d-%4d px: %+.4f   (%d px)" % (r0, r1, rel[m].mean(), m.sum()))
    if args.png:
        from PIL import Image
        s = np.percentile(np.abs(rel), 99.5) or 1.0
        v = np.zeros((H, W, 3), np.uint8)
        v[..., 0] = np.clip(rel / s * 255, 0, 255)
        v[..., 2] = np.clip(-rel / s * 255, 0, 255)
        v[..., 1] = np.clip(np.power(np.clip(Lb / max(np.percentile(Lb, 99.5), 1e-6), 0, 1), 1 / 2.2) * 80, 0, 255)
        Image.fromarray(v).save(args.png)
        print("  diff (red = A brighter, blue = A darker) -> %s" % args.png)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=DEFAULT_DIR, help="hdrdump directory (default: dev install)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geom"); g.add_argument("dump")
    g.add_argument("--dist", type=float, default=0.0, help="world-edge distance, units (default: fog end)")
    g.add_argument("--ground", type=float, default=0.0, help="ground/sea z of the edge (default 0 = sea level)")
    s = sub.add_parser("skyline"); s.add_argument("dump")
    s.add_argument("--band", type=int, nargs=2, help="rows to search (default: eye plane .. world edge +-3 deg)")
    s.add_argument("--ground", type=float, default=0.0)
    s.add_argument("--stride", type=int, default=4, help="column stride")
    s.add_argument("--gap", type=int, default=2, help="rows skipped either side of the step")
    s.add_argument("--min-step", type=float, default=0.08, help="min |dlogL| per row to call it a skyline")
    s.add_argument("--edge-win", type=float, default=1.0, help="deg either side of the world edge = far skyline")
    s.add_argument("--edge-dist", type=float, default=0.0, help="world-edge distance, units (default: fog end)")
    s.add_argument("--max-texture", type=float, default=0.03,
                   help="RMS row-to-row log step below the step; above it the column is a near silhouette")
    s.add_argument("--png")
    d = sub.add_parser("diff"); d.add_argument("a"); d.add_argument("b")
    d.add_argument("--rect", type=int, nargs=4)
    d.add_argument("--png")
    args = ap.parse_args()
    return {"geom": cmd_geom, "skyline": cmd_skyline, "diff": cmd_diff}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
