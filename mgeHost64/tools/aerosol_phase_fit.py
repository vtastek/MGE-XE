#!/usr/bin/env python3
"""S2l/P0 — the aerosol PHASE FUNCTION, measured against a real aerosol before the shader moves.

tasks/forge-atmosphere.md §S2l. numpy only (WSL has no scipy / miepython), so the Mie solver is a
BHMIE port (Bohren & Huffman, appendix A) evaluated over OPAC-style lognormal size distributions.

What this answers, in order:
  1. REFERENCE phase functions at 450/550/650 nm for three aerosols — continental, maritime (MW is
     coastal) and a coarse wet haze (the high-Mie weathers). Each is checked to integrate to 1 over
     the sphere against its ANALYTIC scattering cross-section (a non-trivial check: the numerator is
     a quadrature of |S1|^2+|S2|^2, the denominator is the Q_sca series), and its asymmetry is
     reported against the literature's ~0.6-0.75 at 550 nm.
  2. FITS of the analytic candidates the shader could carry — single HG (today), two-term HG, and
     Cornette-Shanks — each normalised to 1, with P at the angles that matter and the log-error.
  3. The WASHOUT, predicted: the zenith's single-scatter Mie/Rayleigh ratio for a sun at 41.34 deg
     (zenith 48.66 deg from it — the gate) and 78.6 deg (11.4 deg — Sadrith Mora, where x40 went
     white), per candidate, at x1 and x40.

Pass condition for a candidate: P(11.4 deg) AND P(90 deg) both within ~20% of the reference.
HG g 0.8 fails the first, HG g 0.45 fails the second — that is what S2c measured.

Usage: python3 aerosol_phase_fit.py            (prints the tables; ~1 min)
"""
import math
import numpy as np

PI = math.pi

# ─── BHMIE ───────────────────────────────────────────────────────────────────────────────────────
def mie_ab(m, x):
    """Mie coefficients a_n, b_n (n = 1..nstop) for a sphere of size parameter x, index m."""
    nstop = int(x + 4.0 * x ** (1.0 / 3.0) + 2.0)
    y = m * x
    nmx = int(max(nstop, abs(y)) + 16)
    # logarithmic derivative D_n(mx), DOWNWARD recurrence (stable for absorbing m)
    D = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / y - 1.0 / (D[n] + n / y)
    an = np.zeros(nstop, dtype=complex)
    bn = np.zeros(nstop, dtype=complex)
    psi0, psi1 = math.cos(x), math.sin(x)
    chi0, chi1 = -math.sin(x), math.cos(x)
    xi1 = complex(psi1, -chi1)
    for n in range(1, nstop + 1):
        f = (2.0 * n - 1.0) / x
        psi = f * psi1 - psi0
        chi = f * chi1 - chi0
        xi = complex(psi, -chi)
        da = D[n] / m + n / x
        db = D[n] * m + n / x
        an[n - 1] = (da * psi - psi1) / (da * xi - xi1)
        bn[n - 1] = (db * psi - psi1) / (db * xi - xi1)
        psi0, psi1 = psi1, psi
        chi0, chi1 = chi1, chi
        xi1 = complex(psi1, -chi1)
    return an, bn


def pitau(mu, nmax):
    """Angular functions pi_n, tau_n (n = 1..nmax) at every mu: arrays [nmax, len(mu)]."""
    P = np.zeros((nmax, mu.size))
    T = np.zeros((nmax, mu.size))
    p_nm2 = np.zeros_like(mu)   # pi_0
    p_nm1 = np.ones_like(mu)    # pi_1
    P[0] = p_nm1
    T[0] = mu
    for n in range(2, nmax + 1):
        p_n = ((2.0 * n - 1.0) / (n - 1.0)) * mu * p_nm1 - (n / (n - 1.0)) * p_nm2
        P[n - 1] = p_n
        T[n - 1] = n * mu * p_n - (n + 1.0) * p_nm1
        p_nm2, p_nm1 = p_nm1, p_n
    return P, T


# ─── ANGLE GRIDS ─────────────────────────────────────────────────────────────────────────────────
# Gauss-Legendre in mu for the sphere integrals: nodes crowd the poles, so the forward diffraction
# peak (width ~1/x rad, 0.14 deg at the coarsest mode's tail) is sampled at ~0.05 deg there.
NGL = 6000
GL_MU, GL_W = np.polynomial.legendre.leggauss(NGL)
PROBE_DEG = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 11.4, 20.0, 30.0, 48.66, 60.0, 90.0, 120.0, 150.0, 180.0])
FIT_DEG = np.linspace(0.0, 180.0, 721)                      # the curve the fits are scored on
MU_ALL = np.concatenate([GL_MU, np.cos(np.radians(PROBE_DEG)), np.cos(np.radians(FIT_DEG))])
NMAX = 620     # the coarse sea-salt tail: r 38 um at 450 nm is x 528 -> 562 terms
print("building angular functions ...", flush=True)
PI_N, TAU_N = pitau(MU_ALL, NMAX)


def s11_for(m, x):
    """(|S1|^2+|S2|^2)/2 at MU_ALL, and Q_sca."""
    an, bn = mie_ab(m, x)
    nn = an.size
    if nn > NMAX:
        raise RuntimeError("x=%.1f needs %d terms > NMAX" % (x, nn))
    n = np.arange(1, nn + 1)
    c = (2.0 * n + 1.0) / (n * (n + 1.0))
    ca, cb = c * an, c * bn
    S1 = ca @ PI_N[:nn] + cb @ TAU_N[:nn]
    S2 = ca @ TAU_N[:nn] + cb @ PI_N[:nn]
    s11 = 0.5 * (np.abs(S1) ** 2 + np.abs(S2) ** 2)
    qsca = (2.0 / x ** 2) * np.sum((2.0 * n + 1.0) * (np.abs(an) ** 2 + np.abs(bn) ** 2))
    return s11, qsca


# ─── AEROSOL COMPONENTS (OPAC, Hess et al. 1998, at the humidity each is used at) ─────────────────
# (name, r_mod um of the NUMBER distribution, sigma, refractive index at 550 nm).
# The index is held constant over 450-650 nm: across the visible OPAC's own tables move n by <1% and
# k by a few %, which is far below what separates the phase-function candidates here.
# ⚠ BH's CONVENTION: m = n + ik, k > 0 ABSORBS. OPAC prints n - ik; feeding that sign into BHMIE makes
# every absorbing grain a GAIN medium — the first run did, and the dust mode came out with g 0.21
# and a backscatter of 3.1 /sr. Checked against BH's own m=1.55, x=5.213 case (Qsca 3.1050).
COMPONENTS = {
    # water-soluble (sulphate/organics) at 70% RH
    "WASO": (0.0262, 2.24, complex(1.450, 0.0040)),
    # insoluble mineral (dust) — dry, humidity-independent
    "INSO": (0.4710, 2.51, complex(1.530, 0.0080)),
    # soot
    "SOOT": (0.0118, 2.00, complex(1.750, 0.4400)),
    # water-soluble at 80% RH (maritime)
    "WASO80": (0.0311, 2.24, complex(1.400, 0.0020)),
    # sea salt, accumulation / coarse modes at 80% RH
    "SSAM": (0.3780, 2.03, complex(1.381, 1.0e-8)),
    "SSCM": (3.1700, 2.03, complex(1.381, 1.0e-8)),
    # a wet coarse haze mode (mist / fog's leading edge): swollen droplets, nearly water
    "HAZE": (1.0000, 1.80, complex(1.340, 1.0e-8)),
}
# Aerosol types as number densities (1/cm^3) of the components above — OPAC's "continental average"
# and "maritime clean", and a haze of continental background plus the wet coarse mode.
AEROSOLS = {
    "continental": {"WASO": 7000.0, "INSO": 0.4, "SOOT": 8300.0},
    "maritime":    {"WASO80": 1500.0, "SSAM": 20.0, "SSCM": 3.2e-3},
    "haze":        {"WASO": 7000.0, "INSO": 0.4, "SOOT": 8300.0, "HAZE": 1.0},
}
WAVES_UM = [0.450, 0.550, 0.650]
NBIN = 90


def component_optics(name, lam):
    """Size-integrated sum of w*S11/k^2 (um^2/sr) and w*C_sca (um^2) per unit number density."""
    rmod, sigma, m = COMPONENTS[name]
    ls = math.log(sigma)
    lnr = np.linspace(math.log(rmod) - 3.5 * ls, math.log(rmod) + 3.5 * ls, NBIN)
    dlnr = lnr[1] - lnr[0]
    k = 2.0 * PI / lam
    s11_sum = np.zeros(MU_ALL.size)
    csca_sum = 0.0
    for L in lnr:
        r = math.exp(L)
        w = math.exp(-0.5 * ((L - math.log(rmod)) / ls) ** 2) / (math.sqrt(2.0 * PI) * ls) * dlnr
        x = k * r
        s11, qsca = s11_for(m, x)
        s11_sum += w * s11 / k ** 2
        csca_sum += w * qsca * PI * r * r
    return s11_sum, csca_sum


def split(arr):
    a = arr[:NGL]
    b = arr[NGL:NGL + PROBE_DEG.size]
    c = arr[NGL + PROBE_DEG.size:]
    return a, b, c


# ─── ANALYTIC CANDIDATES (each integrates to 1 over the sphere) ──────────────────────────────────
def hg(mu, g):
    d = 1.0 + g * g - 2.0 * g * mu
    return (1.0 - g * g) / (4.0 * PI) / (d * np.sqrt(d))


def tthg(mu, g1, g2, w):
    return w * hg(mu, g1) + (1.0 - w) * hg(mu, g2)


def cs(mu, g):
    d = 1.0 + g * g - 2.0 * g * mu
    return 3.0 / (8.0 * PI) * (1.0 - g * g) * (1.0 + mu * mu) / ((2.0 + g * g) * d * np.sqrt(d))



# The shipped candidate (S2l): a sharp forward SPIKE on a smooth Cornette-Shanks BODY. The spike's g
# is one constant for every aerosol — free fits put it at 0.965-0.972 for all three references —
# so the per-weather freedom is two numbers: the body's g and the spike's weight.
SPIKE_G = 0.97


def hgcs(mu, g, a):
    return a * hg(mu, SPIKE_G) + (1.0 - a) * cs(mu, g)


def rayleigh(mu):
    return 3.0 / (16.0 * PI) * (1.0 + mu * mu)


def sphere_integral(f_mu):
    return 2.0 * PI * np.sum(GL_W * f_mu)


# ─── A SMALL NELDER-MEAD (no scipy) ──────────────────────────────────────────────────────────────
def nelder_mead(f, x0, step, iters=4000, tol=1e-12):
    n = len(x0)
    pts = [np.array(x0, float)]
    for i in range(n):
        p = np.array(x0, float)
        p[i] += step[i]
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        cen = np.mean(pts[:-1], axis=0)
        xr = cen + (cen - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = cen + 2.0 * (cen - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen)
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, len(pts)):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def main():
    fit_mu = np.cos(np.radians(FIT_DEG))
    probe_mu = np.cos(np.radians(PROBE_DEG))
    # SCORED ON theta >= 3 deg, sin-weighted (+0.05 so the poles still count). Inside ~3 deg no LUT
    # here can hold anything — the sky-view rows are ~3.3 deg apart near the zenith — so a fit that
    # bends to match the diffraction spike at 0.3 deg is buying accuracy nobody can see with error
    # everyone can. The sin weight is the solid angle: it is what the sky-view image and the SH
    # integral both weight by.
    sel = FIT_DEG >= 3.0
    wsel = np.sin(np.radians(FIT_DEG[sel])) + 0.05

    refs = {}
    print("\n=== 1. REFERENCE AEROSOLS (BHMIE over OPAC lognormals) ===")
    comp_cache = {}
    for aname, mix in AEROSOLS.items():
        refs[aname] = {}
        for lam in WAVES_UM:
            s11 = np.zeros(MU_ALL.size)
            csca = 0.0
            for cname, N in mix.items():
                key = (cname, lam)
                if key not in comp_cache:
                    comp_cache[key] = component_optics(cname, lam)
                cs_, cc_ = comp_cache[key]
                s11 += N * cs_
                csca += N * cc_
            P = s11 / csca                      # normalised by the ANALYTIC cross-section
            gl, pr, fit = split(P)
            norm = sphere_integral(gl)
            gasym = sphere_integral(gl * GL_MU) / norm
            f3 = 2.0 * PI * np.sum((GL_W * gl)[GL_MU >= math.cos(math.radians(3.0))])
            f10 = 2.0 * PI * np.sum((GL_W * gl)[GL_MU >= math.cos(math.radians(10.0))])
            refs[aname][lam] = dict(probe=pr, fit=fit, g=gasym, gl=gl, f3=f3, f10=f10)
            print("  %-12s %3d nm   integral %.5f   g %.3f   within 3deg %.3f   within 10deg %.3f"
                  % (aname, int(lam * 1000), norm, gasym, f3, f10))

    header = "  ".join("%6.1f" % d for d in PROBE_DEG)
    print("\n  P(theta) /sr at 550 nm; theta deg:\n                 " + header)
    for aname in AEROSOLS:
        print("  %-14s " % aname + "  ".join("%6.3f" % v for v in refs[aname][0.550]["probe"]))
    print("  %-14s " % "HG g0.80" + "  ".join("%6.3f" % v for v in hg(probe_mu, 0.80)))
    print("  %-14s " % "HG g0.45" + "  ".join("%6.3f" % v for v in hg(probe_mu, 0.45)))
    print("  %-14s " % "Rayleigh" + "  ".join("%6.3f" % v for v in rayleigh(probe_mu)))

    print("\n=== 2. FITS at 550 nm, sin-weighted log error over theta >= 3 deg ===")
    print("  HG+CS = a*HG(%.2f) + (1-a)*CS(g): a sharp diffraction SPIKE over a smooth BODY" % SPIKE_G)
    clamp = lambda g: max(-0.95, min(0.95, g))
    sig = lambda t: 1.0 / (1.0 + math.exp(-t))
    ang = [3.0, 11.4, 30.0, 48.66, 90.0, 180.0]
    ix = [list(PROBE_DEG).index(d) for d in ang]
    chosen = {}
    for aname in AEROSOLS:
        ref = refs[aname][0.550]
        lref = np.log(ref["fit"][sel])
        m_sel = fit_mu[sel]

        def err(model):
            v = model(m_sel)
            if np.any(v <= 0) or not np.all(np.isfinite(v)):
                return 1e9
            return float(np.sqrt(np.sum(wsel * (np.log(v) - lref) ** 2) / np.sum(wsel)))

        res = {}
        p, e = nelder_mead(lambda q: err(lambda mu: hg(mu, clamp(q[0]))), [0.7], [0.1])
        res["HG"] = (e, [clamp(p[0])], lambda mu, p=p: hg(mu, clamp(p[0])))
        p, e = nelder_mead(lambda q: err(lambda mu: cs(mu, clamp(q[0]))), [0.6], [0.1])
        res["CS"] = (e, [clamp(p[0])], lambda mu, p=p: cs(mu, clamp(p[0])))

        def tt_par(q):
            return clamp(q[0]), clamp(q[1]), sig(q[2])
        best = None
        for g1 in (0.95, 0.85, 0.7):
            for g2 in (-0.5, 0.0, 0.4):
                for w in (-1.0, 0.5, 2.0):
                    p, e = nelder_mead(lambda q: err(lambda mu: tthg(mu, *tt_par(q))),
                                       [g1, g2, w], [0.03, 0.1, 0.5])
                    if best is None or e < best[1]:
                        best = (p, e)
        p, e = best
        res["TTHG"] = (e, list(tt_par(p)), lambda mu, p=p: tthg(mu, *tt_par(p)))

        def hc_par(q):
            return clamp(q[0]), sig(q[1])
        best = None
        for a0 in (-4.0, -1.5, 0.0):
            p, e = nelder_mead(lambda q: err(lambda mu: hgcs(mu, *hc_par(q))), [0.6, a0], [0.1, 0.5])
            if best is None or e < best[1]:
                best = (p, e)
        p, e = best
        res["HG+CS"] = (e, list(hc_par(p)), lambda mu, p=p: hgcs(mu, *hc_par(p)))
        chosen[aname] = res

        print("\n  %s (reference g %.3f)" % (aname, ref["g"]))
        print("    %-6s %-22s %6s %6s  " % ("model", "params", "rmsLog", "g_eff")
              + " ".join("%7s" % ("r%g" % d) for d in ang))
        rows = [(m, e, ",".join("%.3f" % v for v in par), fn) for m, (e, par, fn) in res.items()]
        rows += [("HG", float("nan"), "0.800 today", lambda mu: hg(mu, 0.80)),
                 ("HG", float("nan"), "0.450 S2c alt", lambda mu: hg(mu, 0.45))]
        for mname, e, par, fn in rows:
            pv = fn(probe_mu)
            geff = sphere_integral(fn(GL_MU) * GL_MU)
            r = [pv[i] / ref["probe"][i] for i in ix]
            ok = "PASS" if abs(r[1] - 1) <= 0.2 and abs(r[4] - 1) <= 0.2 else "fail"
            print("    %-6s %-22s %6.3f %6.3f  " % (mname, par, e, geff)
                  + " ".join("%7.2f" % v for v in r) + "  " + ok)

    print("\n=== 3. THE WASHOUT, PREDICTED — zenith single scatter, Clear row (mieScale 0.60) ===")
    # Optically thin zenith: L_c ∝ beta_c * H_c * P_c(theta). Ratios only — the absolute is the
    # renderer's job. Channels are the packed primaries; the Mie column is grey (tint 1,1,1).
    betaR = np.array([5.802e-6, 13.558e-6, 33.100e-6])
    HR, HM = 8000.0, 1200.0
    betaM1 = 3.996e-6 * 0.60
    for sun_el in (41.34, 78.6):
        th = 90.0 - sun_el
        mu = math.cos(math.radians(th))
        print("\n  sun %.2f deg -> zenith %.2f deg from the sun" % (sun_el, th))
        print("    %-24s %6s   %-20s %-20s" % ("phase", "P", "Mie/Ray RGB x1", "Mie/Ray RGB x40"))
        cands = [("HG g0.80 (today)", lambda m_: hg(m_, 0.80))]
        for aname in AEROSOLS:
            ref = refs[aname][0.550]
            cands.append(("ref " + aname, lambda m_, ref=ref: np.interp(
                math.degrees(math.acos(max(-1.0, min(1.0, float(m_))))), FIT_DEG, ref["fit"])))
            cands.append(("HG+CS fit " + aname, chosen[aname]["HG+CS"][2]))
        for name, fn in cands:
            P = float(np.asarray(fn(np.array(mu))).reshape(-1)[0])
            ray = betaR * HR * rayleigh(mu)
            r1 = betaM1 * HM * P / ray
            print("    %-24s %6.3f   %-20s %-20s" % (name, P, "%.2f %.2f %.2f" % tuple(r1),
                                                    "%.2f %.2f %.2f" % tuple(40.0 * r1)))

    print("\n=== 4. ROWS WITH NO MEASURED REFERENCE: CS g that keeps the AUTHORED asymmetry ===")
    # Ash (mineral dust) and Snow/Blizzard (ice) have no sphere reference that means anything — ice
    # is not a sphere, and dust is barely one. Their authored HG g is the only statement of intent
    # there is, so the new model keeps it as an ASYMMETRY: the CS g whose own <cos> equals it.
    for g_auth in (0.80, 0.85, 0.65, 0.62, 0.60, 0.55):
        p, e = nelder_mead(lambda q: (sphere_integral(cs(GL_MU, clamp(q[0])) * GL_MU) - g_auth) ** 2,
                           [g_auth], [0.05], tol=1e-16)
        print("  authored HG g %.2f  ->  CS g %.3f" % (g_auth, clamp(p[0])))

if __name__ == "__main__":
    main()
