#!/usr/bin/env python3
"""Derive the discrete |k| convolution kernel behind mgeHost64/shaders/FSL/ripplewave.comp.fsl,
and regenerate that shader's tap table.

    python3 tools/iwave_kernel.py            # report only
    python3 tools/iwave_kernel.py --emit     # rewrite ripplewave.comp.fsl

WHY THIS EXISTS. The R2 actor wake needs a Kelvin wedge, and a wedge is a DISPERSION effect. The
first wake sim (ripplesim.comp) integrates `v += avg(4-neighbours) - h`, the plain wave equation, in
which every wavelength travels at the same speed: a moving source there can only draw nested circles
or a Mach cone, never a fixed-angle wedge. Deep-water gravity waves obey omega^2 = g|k| instead, and
that square root is the whole phenomenon.

WHAT HAS TO BE ACCURATE, which is not the obvious thing. For omega ~ k^p the wake half-angle is
arcsin((1-p)/(1+p)): p=1/2 gives Kelvin's 19.47 deg, p=1 (a Laplacian) gives 0. So the quantity that
must be right is the LOG-SLOPE of the response, not the response. Fitting the kernel to |k| by
weighted least squares gets R(k) within 10% and still swings the implied wedge across 12..62 deg,
because a small ripple in R is a large error in dR/dk. Construct the kernel instead:

    kernel = IFFT2(|k| * W(|k|))  ->  spatial taper  ->  exact DC removal

The taper suppresses the truncation leakage that produced the ripple; the DC removal enforces
R(0) = |0| = 0, without which flat water accelerates.

CHOOSING THE RADIUS. |k| has a 1/r^3 real-space tail, so a truncated kernel always loses the low-k
(long-wavelength) end, and cost goes as (2P+1)^2 per texel. Radius 10 holds p in 0.42..0.73 over
wavelengths 4..24 texels, which at the shipped 8 world units/texel is precisely the swim-speed band
(60 u/s -> 4.1 texels, 100 -> 11.5, 140 -> 22.5). Radius 6 halves the cost and loses the fast end.
"""
import argparse
import numpy as np

P_DEFAULT, KC_DEFAULT, TAPER_DEFAULT, THREADS = 10, 9.0, 0.5, 16
MW_UNIT_M = 0.014224          # 1 Morrowind unit in metres (64 units = 1 yard)
G_SI = 9.81


def build(P, kc, taper_pow, N=512):
    """Central (2P+1)^2 taps of the periodised |k|*W(k) kernel, tapered and DC-corrected."""
    f = np.fft.fftfreq(N, d=1.0) * 2.0 * np.pi
    KX, KY = np.meshgrid(f, f, indexing="xy")
    KM = np.hypot(KX, KY)
    ker = np.fft.fftshift(np.real(np.fft.ifft2(KM * np.exp(-(KM / kc) ** 2))))
    c = N // 2
    G = ker[c - P:c + P + 1, c - P:c + P + 1].copy()
    jj, ii = np.mgrid[-P:P + 1, -P:P + 1]
    r = np.hypot(ii, jj)
    G *= np.cos(0.5 * np.pi * np.minimum(r / (P + 0.5), 1.0)) ** taper_pow
    G -= G.sum() / G.size
    G[P, P] -= G.sum()        # make it exact after the taper
    return G


def response(G, kx, ky):
    P = (G.shape[0] - 1) // 2
    jj, ii = np.mgrid[-P:P + 1, -P:P + 1]
    kx, ky = np.atleast_1d(kx), np.atleast_1d(ky)
    return (np.cos(kx[..., None, None] * ii + ky[..., None, None] * jj) * G).sum(axis=(-1, -2))


def report(G, upt):
    P = (G.shape[0] - 1) // 2
    g = np.linspace(-np.pi, np.pi, 385)
    KX, KY = np.meshgrid(g, g)
    R = response(G, KX.ravel(), KY.ravel())
    ang = np.linspace(0.0, np.pi / 4, 9)
    print(f"kernel radius {P}  ({2*P+1}^2 = {(2*P+1)**2} taps)")
    print(f"  sum {G.sum():+.2e}   R over BZ [{R.min():+.5f}, {R.max():.5f}]"
          f"   {'OK' if R.min() > -1e-4 else '*** NEGATIVE -> UNSTABLE ***'}")
    print("   lambda(tex)      k    R/|k|       p    wedge    aniso")
    alphas = []
    for lam in (40, 30, 24, 18, 14, 11, 9, 7, 5.5, 4.5, 3.5):
        k, h = 2 * np.pi / lam, 0.04
        r0 = response(G, k*np.exp(-h)*np.cos(ang), k*np.exp(-h)*np.sin(ang)).mean()
        r1 = response(G, k*np.exp( h)*np.cos(ang), k*np.exp( h)*np.sin(ang)).mean()
        rd = response(G, k*np.cos(ang), k*np.sin(ang))
        p = 0.5 * (np.log(max(r1, 1e-12)) - np.log(max(r0, 1e-12))) / (2 * h)
        ratio = (1 - p) / (1 + p)
        wedge = np.degrees(np.arcsin(min(max(ratio, 0.0), 1.0))) if ratio > 0 else 0.0
        if 4.0 <= lam <= 24.0:
            alphas.append(rd.mean() / k)
        print(f"   {lam:9.1f}  {k:6.3f}  {rd.mean()/k:6.3f}  {p:6.3f}  {wedge:6.2f}deg"
              f"  {(rd.max()-rd.min())/max(abs(rd.mean()),1e-9)*100:5.1f}%")
    alpha = float(np.mean(alphas))
    grav = G_SI / (alpha * upt * MW_UNIT_M)
    print(f"\n  alpha (mean R/|k| over lambda 4..24 texels) = {alpha:.4f}")
    print(f"  at {upt} world units/texel: gravity {grav:.2f} s^-2, "
          f"CFL dt_max {2.0/np.sqrt(grav*R.max()):.4f} s")
    print("  Kelvin transverse wavelength vs swim speed:")
    for V in (60, 80, 100, 120, 140):
        vt = V / upt
        lam = 2 * np.pi * vt * vt / grav
        print(f"     V={V:4d} u/s -> {lam:5.1f} texels = {lam*upt:6.1f} units")
    return R.max(), alpha


def fold(G, tile):
    """Group taps into 8-fold-symmetry orbits: one multiply per distinct radius."""
    P = (G.shape[0] - 1) // 2
    orb = {}
    for j in range(-P, P + 1):
        for i in range(-P, P + 1):
            orb.setdefault((max(abs(i), abs(j)), min(abs(i), abs(j))), []).append((i, j))
    out = []
    for key in sorted(orb):
        mem = sorted(set(orb[key]))
        coef = G[mem[0][1] + P, mem[0][0] + P]
        taps = []
        for (i, j) in mem:
            off = j * tile + i
            taps.append(f"g_wave[base + {off}]" if off >= 0 else f"g_wave[base - {-off}]")
        out.append((coef, taps))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=P_DEFAULT)
    ap.add_argument("--kc", type=float, default=KC_DEFAULT)
    ap.add_argument("--taper", type=float, default=TAPER_DEFAULT)
    ap.add_argument("--upt", type=float, default=8.0, help="world units per texel (host constant)")
    ap.add_argument("--emit", action="store_true", help="rewrite ripplewave.comp.fsl's tap table")
    a = ap.parse_args()

    G = build(a.radius, a.kc, a.taper)
    rmax, alpha = report(G, a.upt)
    if a.emit:
        import pathlib
        tile = THREADS + 2 * a.radius
        body = "\n".join(f"    acc += {c:+.8f}f * ({' + '.join(t)});" for c, t in fold(G, tile))
        path = (pathlib.Path(__file__).resolve().parent.parent
                / "mgeHost64" / "shaders" / "FSL" / "ripplewave.comp.fsl")
        src = path.read_text()
        head, _, rest = src.partition("    float acc  = 0.0f;\n")
        _, _, tail = rest.partition("\n\n    // ---- integrate")
        path.write_text(f"{head}    float acc  = 0.0f;\n{body}\n\n    // ---- integrate{tail}",
                        newline="\n")
        print(f"\nemitted {(2*a.radius+1)**2} taps into {path}")
        print(f"  ⚠ WAVE_RMAX in that shader must read {rmax:.4f}f")
        print(f"  ⚠ kWakeKernelAlpha in forgerender.cpp must read {alpha:.4f}f")
