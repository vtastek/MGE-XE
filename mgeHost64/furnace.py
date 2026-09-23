"""
White-furnace tests for the Forge host's BRDF lobes.

Every function below is transcribed LINE FOR LINE from pbrmaterial.h.fsl, not paraphrased, so a
failure here is a failure of the shipped shader and not of a model of it. No GPU, no game: the
furnace is a hemisphere integral and Python does it exactly as well as a compute pass would.

⚠ THE 1/pi CONVENTION. This project drops the 1/pi on BOTH lobes — GGX's D has no 1/pi and the
diffuse is `albedo * NoL` rather than `albedo/pi * NoL`. So a shader value f is pi * f_physical, and
the directional albedo is E(V) = (1/pi) * integral of f * cos(theta_L) dw_L. Lambert checks this:
f = albedo constant, integral of cos dw = pi, so E = albedo exactly.

WHAT A PASS MEANS
  - specular, f0 = 1: E is the single-scatter directional albedo. It is < 1 by construction (GGX
    loses energy to masking), and 1 - E is exactly what pbrSpecMulti exists to give back.
  - the split-sum fit (pbrEnvAB) must AGREE with that integral. pbrSpecMulti is fed `ab` from the
    fit, so if the fit disagrees with the real lobe the compensation is wrong by the difference.
  - EON diffuse at rho = 1 must integrate to 1 at every sigma. That is what "energy preserving"
    claims, and it is the one property the lobe was adopted for.
"""
import math

# ── transcribed from pbrmaterial.h.fsl ───────────────────────────────────────────────────────────
EON_C1, EON_K, EON_C1MK = 0.287793409, 0.072488212, 0.215305197

def saturate(x): return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def pbrSpecTerms(NoL, NoV, NoH, HoL, alpha2):
    m  = 1.0 - saturate(HoL)
    m5 = (m * m) * (m * m) * m
    dd = (NoH * alpha2 - NoH) * NoH + 1.0
    D  = alpha2 / max(dd * dd, 1e-12)
    lv = NoL * math.sqrt((-NoV * alpha2 + NoV) * NoV + alpha2)
    ll = NoV * math.sqrt((-NoL * alpha2 + NoL) * NoL + alpha2)
    DV = D * (0.5 / max(lv + ll, 1e-6))
    return DV * (1.0 - m5), DV * m5

def pbrEnvAB(NoV, rough):
    c0 = (-1.0, -0.0275, -0.572, 0.022); c1 = (1.0, 0.0425, 1.04, -0.04)
    r  = [rough * c0[i] + c1[i] for i in range(4)]
    a004 = min(r[0] * r[0], 2.0 ** (-9.28 * NoV)) * r[0] + r[1]
    return -1.04 * a004 + r[2], 1.04 * a004 + r[3]

def pbrSpecMulti(f0, ab, strength):
    ess = max(ab[0] + ab[1], 1e-2)
    return 1.0 + (strength * (1.0 / ess - 1.0)) * f0

def eonAB(sigma):
    A = 1.0 / (1.0 + EON_C1 * sigma)
    return A, sigma * A

def eonSingle(ab, NoL, NoV, LoV):
    s = LoV - NoL * NoV
    t = max(max(NoL, NoV), 1e-4) if s > 0.0 else 1.0
    return ab[0] + ab[1] * (s / t)

def eonG(mu):
    u = 1.0 - saturate(mu)
    return EON_C1 * u * (0.132551 + u * (2.136720 + u * (-1.914293 + u * 0.645021)))

def eonMsWeight(mu):           return EON_C1 - eonG(mu)
def eonMsScale(ab, sigma, NoV): return sigma * ab[0] * eonMsWeight(NoV) * (1.0 / EON_C1MK)
def eonMsAlbedo(rho, eAvg):    return rho * rho * eAvg / max(1.0 - rho * (1.0 - eAvg), 1e-4)
def eonEavg(ab, sigma):        return ab[0] * (1.0 + sigma * EON_K)

# ── the furnace: E(V) = (1/pi) * integral over the hemisphere of f * cos(theta_L) dw ─────────────
def integrate(f, NoV, NT=256, NP=512):
    """f(NoL, NoH, HoL, LoV) -> shader-convention BRDF value. V is in the xz plane."""
    sinV = math.sqrt(max(0.0, 1.0 - NoV * NoV))
    V = (sinV, 0.0, NoV)
    total = 0.0
    for i in range(NT):
        th = (i + 0.5) * (math.pi * 0.5) / NT
        ct, st = math.cos(th), math.sin(th)
        for j in range(NP):
            ph = (j + 0.5) * (2.0 * math.pi) / NP
            L = (st * math.cos(ph), st * math.sin(ph), ct)
            NoL = ct
            if NoL <= 0.0: continue
            hx, hy, hz = V[0] + L[0], V[1] + L[1], V[2] + L[2]
            hl = math.sqrt(hx * hx + hy * hy + hz * hz)
            if hl < 1e-9: continue
            H = (hx / hl, hy / hl, hz / hl)
            NoH = saturate(H[2])
            HoL = saturate(H[0] * L[0] + H[1] * L[1] + H[2] * L[2])
            LoV = L[0] * V[0] + L[1] * V[1] + L[2] * V[2]
            total += f(NoL, NoH, HoL, LoV) * NoL * st
    return total * (math.pi * 0.5 / NT) * (2.0 * math.pi / NP) / math.pi
