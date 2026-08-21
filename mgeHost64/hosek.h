// mgeHost64 — the physical sky: Hosek-Wilkie radiance, host side (tasks/forge-physical-sky.md P2).
//
// P1 SCALED MW'S AUTHORED SKY. THIS REPLACES IT WITH A GENERATOR. The anchor was a correction
// applied to a display code, and it inherited two problems it could not solve: it scaled the whole
// sky pass (clouds and moons included, which P2b forbids), and MW's 2-colour dome cannot hold a real
// sky's SHAPE — §0.5 measured up to 5x circumsolar brightening and a minimum 60-90 degrees from the
// sun that dips BELOW the zenith, and `lerp(fogColNear, skyZenith, saturate(dir.z))` returns one
// number for that whole ring.
//
// What lives here is one radiance field L(w) in ABSOLUTE units, and it is the single source for
// three things that used to be three different definitions (§0.6):
//   - the sky the player sees            (skyhw.frag, the same closed form in hosek.h.fsl)
//   - the ambient's SHAPE                (publishSkyAmbientSH's SH-L1 projection)
//   - the ambient's LEVEL and the SUN    (skyPhysicalLighting, below)
// They cannot drift apart because they are the same object. That is the whole point of the step;
// everything else here is bookkeeping around it.
//
// ⚠ THE DATASET IS TRANSCRIBED PRIOR ART AND IS NOT TRUSTED — IT IS MEASURED.
// hosek_data.h is a verbatim copy of the authors' release, and a mistyped constant in 3600 of them
// would not fail to build; it would make the sky "look a bit off", which is indistinguishable from
// every other reason a sky looks a bit off. So selfTest() below re-derives three cooked
// configurations and twelve radiances at init and compares them against values produced by the
// authors' OWN implementation, and says so on the log. Nothing downstream of this file means
// anything until that line reads OK. [[feedback_prior_art_constants_dont_transfer]]
//
// ⚠ UNITS ARE MEASURED, NOT ASSUMED. The RGB dataset turns out to be in absolute W/(m^2 sr) per
// linear sRGB primary — verified against the CIE XYZ dataset, which the authors document as
// "Y * 683 = luminance in lm": converting XYZ -> linear sRGB reproduces the RGB dataset to 0.7% per
// channel and `luma709 * 683` matches `Y * 683` to 0.02%. See hosek_data.h. This is what makes the
// step possible at all — a model that states its own level does not need to be told one.
//
// ⚠ THE SOLAR DISC IS **NOT** FROM THE H-W SOLAR FUNCTION, AND THAT IS A MEASURED DECISION.
// The plan asked for `E_sun = L_solar(theta_s) * Omega_sun` from the model's own solar-radiance
// term. The 2013 release ships that function in SPECTRAL FORM ONLY ("CAVEAT #1: in this release,
// this function is only provided in spectral form! RGB/XYZ versions to follow at a later date") —
// 11 wavebands, and turning 11 wavebands into linear sRGB needs the CIE colour-matching functions,
// which the release does not ship. The obvious dodge — recover the spectral->XYZ quadrature by
// least squares from the two datasets that DO overlap — was tried and FALSIFIED: 27% max held-out
// error with wildly alternating signs, because the shipped XYZ data is not a linear map of the
// 11-band spectral FIT and sky spectra are nearly collinear anyway. Do not re-derive that.
// What is here instead is the standard clear-sky direct-beam attenuation (Kasten air mass, Rayleigh
// + Angstrom aerosol at the sRGB primaries), whose SHAPE is published prior art and whose absolute
// SCALE is solved at init against our own §0.1 measurement. Corroboration, all of it after the fact:
// the anchor lands the reference clear day at 94 klx direct-normal and peaks at 109 klx with the sun
// overhead, against a textbook sea-level maximum of ~110 klx.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "hosek_data.h"

namespace Hosek {

    constexpr double kPi     = 3.14159265358979323846;
    constexpr double kHalfPi = kPi * 0.5;

    // ─── THE MODEL ───────────────────────────────────────────────────────────────────────────────
    // Nine coefficients + one radiance term per channel, all functions of THREE scalars, so this is
    // a per-frame cook and never a per-pixel one. The same nine reach the shader through gSkyView.
    struct State {
        float cfg[3][9];      // A..I per channel, cooked for (turbidity, albedo, solar elevation)
        float rad[3];         // the model's per-channel radiance scale
        float toSun[3];       // unit vector TOWARD the sun (world, MW Z up)
        float elevation;      // solar elevation in radians, clamped to the model's domain
        float turbidity;
        float albedo;
        bool  valid;          // false = never cooked; every consumer must treat that as "no sky"
    };

    // The model's domain. Outside it the dataset indexing walks off its own array, so these are
    // hard clamps and not preferences: turbidity indexes a 10-entry table directly, and the
    // elevation warp `pow(elev / (pi/2), 1/3)` is undefined for a sun below the horizon.
    constexpr double kTurbidityMin = 1.0;
    constexpr double kTurbidityMax = 10.0;
    constexpr double kAlbedoMin    = 0.0;
    constexpr double kAlbedoMax    = 1.0;
    // The lowest solar elevation the model is evaluated at. Not a taste value: below roughly this
    // the fit degrades badly (the reference implementation's own dataset is piecewise in
    // `(2*elev/pi)^(1/3)`, which crowds every piece into the last degree), and beneath 0 it is
    // undefined outright. Night is handled by RAMPING THE WHOLE MODEL OUT over the last few degrees
    // (skyNightRamp) rather than by evaluating it somewhere it does not hold.
    constexpr double kElevationMin = 0.0;

    // ─── COOKING, mirroring ArHosekSkyModel.c's ArHosekSkyModel_CookConfiguration EXACTLY ────────
    // Quintic Bezier in the CUBE ROOT of normalised elevation, bilinear in (turbidity, albedo).
    // Written out rather than restructured: this is the one place where "equivalent" is not good
    // enough, because selfTest() compares against the reference implementation to 1e-5 and any
    // reordering shows up there as a failure nobody can attribute.
    inline void bezierAccum(const double* m, double t, double w, double* dst, int stride, int n)
    {
        const double u  = 1.0 - t;
        const double b0 = u*u*u*u*u;
        const double b1 = 5.0 * u*u*u*u * t;
        const double b2 = 10.0 * u*u*u * t*t;
        const double b3 = 10.0 * u*u * t*t*t;
        const double b4 = 5.0 * u * t*t*t*t;
        const double b5 = t*t*t*t*t;
        for (int i = 0; i < n; ++i) {
            dst[i] += w * (b0 * m[i]
                         + b1 * m[i + stride]
                         + b2 * m[i + 2 * stride]
                         + b3 * m[i + 3 * stride]
                         + b4 * m[i + 4 * stride]
                         + b5 * m[i + 5 * stride]);
        }
    }

    // dataset layout (per channel): [albedo 0 | albedo 1] x [turbidity 1..10] x [6 Bezier control
    // points] x [n coefficients]. n = 9 for the shape dataset, 1 for the radiance dataset, and the
    // stride between control points is n — which is why one routine serves both.
    inline void cookGroup(const double* dataset, int n, double turbidity, double albedo,
                          double elevWarped, double* out)
    {
        for (int i = 0; i < n; ++i) { out[i] = 0.0; }
        const int    it   = (int)turbidity;
        const double trem = turbidity - (double)it;
        const int    blk  = n * 6;                       // one turbidity's control points
        const int    alb1 = blk * 10;                    // ...and the albedo-1 half of the table
        bezierAccum(dataset + blk * (it - 1),        elevWarped, (1.0 - albedo) * (1.0 - trem), out, n, n);
        bezierAccum(dataset + alb1 + blk * (it - 1), elevWarped, albedo         * (1.0 - trem), out, n, n);
        if (it == 10) { return; }                        // the table's last row has no successor
        bezierAccum(dataset + blk * it,              elevWarped, (1.0 - albedo) * trem,         out, n, n);
        bezierAccum(dataset + alb1 + blk * it,       elevWarped, albedo         * trem,         out, n, n);
    }

    inline void cook(double turbidity, double albedo, double solarElevation, State& s)
    {
        turbidity      = std::max(kTurbidityMin, std::min(kTurbidityMax, turbidity));
        albedo         = std::max(kAlbedoMin,    std::min(kAlbedoMax,    albedo));
        solarElevation = std::max(kElevationMin, std::min(kHalfPi,       solarElevation));
        const double ew = std::pow(solarElevation / kHalfPi, 1.0 / 3.0);
        for (int c = 0; c < 3; ++c) {
            double cfg[9], rad[1];
            cookGroup(kHosekDatasets[c],    9, turbidity, albedo, ew, cfg);
            cookGroup(kHosekDatasetsRad[c], 1, turbidity, albedo, ew, rad);
            for (int i = 0; i < 9; ++i) { s.cfg[c][i] = (float)cfg[i]; }
            s.rad[c] = (float)rad[0];
        }
        s.elevation = (float)solarElevation;
        s.turbidity = (float)turbidity;
        s.albedo    = (float)albedo;
        s.valid     = true;
    }

    // ─── EVALUATION — the closed form, and its SECOND COPY lives in hosek.h.fsl ──────────────────
    // ⚠ TWO EVALUATIONS OF ONE MODEL, WHICH IS THE FOURTH-COPY PROBLEM publishSkyAmbientSH ALREADY
    // NAMES BY NAME. The C++ side feeds the SH (and therefore the light); the FSL side feeds the
    // pixels. A silent divergence between the sky you SEE and the light it CASTS is exactly the
    // failure this whole step exists to remove, so it is not left to discipline: hosekcheck.comp
    // evaluates the shader form at a fixed direction set once at startup, the host evaluates the
    // same set here, and the max deviation is logged. If you edit this expression, edit that one.
    //
    // cosTheta = the view direction's z (MW is Z-up, so the zenith angle's cosine IS dir.z);
    // gamma    = the angle between the view direction and the SUN.
    inline float evalNative(const State& s, float cosTheta, float gamma, int ch)
    {
        const float* c = s.cfg[ch];
        // Below the horizon the model is not merely inaccurate, it INVERTS: c[1] is negative, so
        // once cosTheta drops under -0.01 the exponent's sign flips and the term runs away. Clamped
        // to the horizon rather than gated, so the field is continuous across the skyline and the
        // fog target below it is the horizon haze — which is what a surface seen from above melts
        // into, and what re-establishes the "looking down from a height must not move" control that
        // an alpha-1 sky copy would otherwise have broken (skydome.h.fsl).
        const float ct   = std::max(0.0f, cosTheta);
        const float cg   = std::cos(gamma);
        const float expM = std::exp(c[4] * gamma);
        const float rayM = cg * cg;
        const float den  = 1.0f + c[8] * c[8] - 2.0f * c[8] * cg;
        const float mieM = (1.0f + cg * cg) / (den * std::sqrt(std::max(1.0e-8f, den)));
        const float zen  = std::sqrt(ct);
        return (1.0f + c[0] * std::exp(c[1] / (ct + 0.01f)))
             * (c[2] + c[3] * expM + c[5] * rayM + c[6] * mieM + c[7] * zen)
             * s.rad[ch];
    }

    // Radiance along `dir` (unit, world, MW Z up) in the model's native units: W/(m^2 sr) per linear
    // sRGB primary. NOT scene units — skyPhysicalLighting owns that conversion, in one place.
    inline void radiance(const State& s, const float dir[3], float out[3])
    {
        if (!s.valid) { out[0] = out[1] = out[2] = 0.0f; return; }
        float d = dir[0] * s.toSun[0] + dir[1] * s.toSun[1] + dir[2] * s.toSun[2];
        d = std::max(-1.0f, std::min(1.0f, d));
        const float gamma = std::acos(d);
        for (int c = 0; c < 3; ++c) { out[c] = std::max(0.0f, evalNative(s, dir[2], gamma, c)); }
    }

    // ─── THE QUADRATURE, shared so the anchor and the runtime ambient cannot disagree ────────────
    // Fibonacci sphere: z is uniform over [-1,1], which IS uniform in solid angle, so every sample
    // carries the same weight 4*pi/N and the integral is a plain sum. This is verbatim the table
    // publishSkyAmbientSH builds — same golden angle, same offset — because the anchor solved here
    // and the irradiance measured there have to be the SAME number or the sun:sky split drifts by
    // whatever the two quadratures disagree about. Measured against a 400x800 tabulated integral:
    // 0.06% at N = 128.
    inline void fibDir(int i, int n, float d[3])
    {
        constexpr float kGolden = 2.39996322972865332f;
        const float z   = 1.0f - (2.0f * (float)i + 1.0f) / (float)n;
        const float r   = std::sqrt(std::max(0.0f, 1.0f - z * z));
        const float phi = kGolden * (float)i;
        d[0] = r * std::cos(phi);
        d[1] = r * std::sin(phi);
        d[2] = z;
    }
    constexpr int kFibDirs = 128;

    // Cosine-weighted UPPER-hemisphere irradiance of the sky, native units (W/m^2 per primary).
    // This is the sky's whole contribution to the light on flat ground — the quantity §0.1's
    // albedo-free invariant is stated against, and the one the sun is anchored to.
    inline void skyIrradiance(const State& s, float out[3])
    {
        out[0] = out[1] = out[2] = 0.0f;
        if (!s.valid) { return; }
        const float w = 4.0f * (float)kPi / (float)kFibDirs;
        for (int i = 0; i < kFibDirs; ++i) {
            float d[3];
            fibDir(i, kFibDirs, d);
            if (d[2] <= 0.0f) { continue; }
            float L[3];
            radiance(s, d, L);
            for (int c = 0; c < 3; ++c) { out[c] += L[c] * d[2] * w; }
        }
    }

    inline float luma709(const float v[3])
    {
        return 0.2126f * v[0] + 0.7152f * v[1] + 0.0722f * v[2];
    }

    // ─── THE SUN ─────────────────────────────────────────────────────────────────────────────────
    // Clear-sky direct-beam transmittance at the sRGB primaries. Rayleigh + Angstrom aerosol over a
    // Kasten air mass — the standard pairing for a Preetham/Hosek sky, and prior art whose SHAPE is
    // all that is taken from it; the LEVEL is solved below against our own measurement.
    // ⚠ Ozone and water vapour are omitted. Both are small across the visible band at these air
    // masses, and including them would add two more atmospheric parameters MW has nothing to drive
    // with — turbidity is already the one knob until P4.
    // ⚠ THE SUN'S COLOUR IS ENTIRELY EXTINCTION. The extraterrestrial beam is taken as neutral in
    // linear sRGB, which is a simplification (it is a ~5800 K body, faintly warm) but the visible
    // effect — the reddening from ~(1.09, 1.00, 0.81) at the reference to ~(1.46, 1.00, 0.40) at 9
    // degrees — is the extinction's, not the source's.
    inline void sunTransmittance(double solarElevation, double turbidity, float out[3])
    {
        constexpr double kLambdaUm[3] = { 0.611, 0.549, 0.464 };   // sRGB primary dominant wavelengths
        const double elev = std::max(1.0e-4, std::min(kHalfPi, solarElevation));
        const double zDeg = 90.0 - elev * 180.0 / kPi;
        // Kasten 1966. sec(z) diverges at the horizon and this does not: the correction term is what
        // keeps a setting sun finite instead of infinitely extinguished one frame before it sets.
        const double m    = 1.0 / (std::sin(elev) + 0.15 * std::pow(93.885 - zDeg, -1.253));
        const double beta = std::max(0.0, 0.04608 * turbidity - 0.04586);   // Angstrom, from turbidity
        for (int c = 0; c < 3; ++c) {
            const double tauR = 0.008735 * std::pow(kLambdaUm[c], -4.08);
            const double tauA = beta     * std::pow(kLambdaUm[c], -1.3);
            out[c] = (float)std::exp(-(tauR + tauA) * m);
        }
    }

    // ─── THE ONE ABSOLUTE ANCHOR, SOLVED AT INIT AND NEVER WRITTEN DOWN ──────────────────────────
    // The transmittance above is a shape; something has to say how bright the beam is. That number
    // is solved once, at the REFERENCE configuration, against the only measurement we own that
    // constrains it: §0.1's clear-sky energy split.
    //
    //   E_sun / E_total = 0.80  (median of 8 clear sunlit HDRIs; p25..p75 = 0.73..0.84)
    //
    // The sky's own E is exact (the model states it), so the split pins the sun. Everything else
    // about the sun — how it falls at dusk, how it reddens, how it responds to turbidity — is the
    // transmittance model's, and this constant does not touch any of it.
    //
    // ⚠⚠ 41.34 DEGREES IS NO LONGER "MW'S SUN", AND THE ORIGINAL JUSTIFICATION FOR IT IS DEAD.
    // It was picked as MW's clear-weather `sunDir.z = -0.66`, i.e. the elevation of the LIGHT — and
    // the model is cooked at the DISC now, which on that very same frame sits at 69.9 degrees, 29
    // degrees higher (see skyPhysicalMeasure: MW has two suns and they do not agree). So the number
    // survives on a different argument, and it is worth stating rather than leaving as an
    // archaeological coincidence:
    //
    //   §0.1's `E_sun/E_total = 0.80` is a POPULATION median over eight clear sunlit HDRIs whose
    //   solar elevations are unknown. Sun share is strongly elevation-dependent — this model gives
    //   0.35 at 10 degrees rising to 0.86 by 70 — so the constraint has to be applied at SOME
    //   elevation, and a mid-range one minimises the extrapolation to either end. Anchoring at 70
    //   instead would pin 0.80 at a high sun and push every low sun further down.
    //
    // ⚠ CONSEQUENCE, REPORTED NOT HIDDEN: at the harness save's actual disc elevation of 69.9 the
    // model then reads sun share 0.859, just above the measured p75 of 0.84. That is the model
    // saying a 70-degree sun is sunnier than the population median, which is physically right, but
    // it is out of the measured band and the `sun%=` field on the [sky] heartbeat carries the band
    // beside it so it stays visible. If it ever needs to move, move THIS constant — not the
    // per-frame maths.
    constexpr double kRefTurbidity = 3.0;
    constexpr double kRefAlbedo    = 0.10;
    constexpr double kRefElevation = 41.34 * kPi / 180.0;
    constexpr double kRefSunShare  = 0.80;

    // ...and what ONE SCENE UNIT is worth, which is the OTHER absolute number and the one P1 left
    // behind. P1 derived `unitCd = kClearDayLux / pi / Epi` per frame from MW's own lighting; that
    // cannot survive into P2, because P2 SETS the lighting and the derivation would close on itself.
    // So it is pinned here, to exactly the value P1 measured on the reference save, out of the same
    // two named numbers it used. Pinning is not a downgrade — it is the requirement: the emissive
    // and lantern calibration is stated in scene units, and it only keeps meaning what it meant if
    // the unit stops moving. [[project_emissive_kref_rederived]] [[project_emissive_area_ref_measured]]
    constexpr double kClearDayLux = 90000.0;    // a real clear day's horizontal illuminance (P1)
    constexpr double kRefEpi      = 0.9046;     // P1's MEASURED E_total/pi on the reference save
    constexpr double kSceneUnitCd = kClearDayLux / kPi / kRefEpi;   // ~= 31,670 cd/m^2 per scene 1.0
    // Native model units (W/(m^2 sr), linear sRGB) -> SCENE units. 683 lm/W is the photometric
    // constant the authors' own units note is stated against; dividing by kSceneUnitCd lands the
    // result where the rest of the renderer already lives.
    constexpr double kNativeToScene = 683.0 / kSceneUnitCd;

    // Solved once (idempotent), because it costs a cook + 128 evaluations and never changes.
    inline double sunBeamScale()
    {
        static double s_scale = 0.0;
        if (s_scale > 0.0) { return s_scale; }
        State ref = {};
        cook(kRefTurbidity, kRefAlbedo, kRefElevation, ref);
        ref.toSun[0] = (float)std::cos(kRefElevation);
        ref.toSun[1] = 0.0f;
        ref.toSun[2] = (float)std::sin(kRefElevation);
        float Esky[3];
        skyIrradiance(ref, Esky);
        float tr[3];
        sunTransmittance(kRefElevation, kRefTurbidity, tr);
        const double denom = luma709(tr) * std::sin(kRefElevation);
        s_scale = (denom > 1.0e-9)
                ? (kRefSunShare / (1.0 - kRefSunShare)) * (double)luma709(Esky) / denom
                : 0.0;
        return s_scale;
    }

    // Direct-NORMAL solar irradiance, native units. Multiply by the ground's N.L to get what lands
    // on a surface; `sunCol` in this renderer is E_normal/pi, because `lit = amb + sun*ndl` IS E/pi
    // (§0.1) and that identity is the reason the shading term and the measurement are comparable.
    inline void sunIrradianceNormal(double solarElevation, double turbidity, float out[3])
    {
        float tr[3];
        sunTransmittance(solarElevation, turbidity, tr);
        const float s = (float)sunBeamScale();
        for (int c = 0; c < 3; ++c) { out[c] = s * tr[c]; }
    }

    // ─── NIGHT ───────────────────────────────────────────────────────────────────────────────────
    // H-W degrades below a few degrees of solar elevation and is undefined beneath 0, so the model
    // is ramped OUT rather than extrapolated. 1 = fully physical, 0 = the model contributes nothing.
    //
    // ⚠ THE SAME RAMP DOES TWO DIFFERENT THINGS, AND THAT IS DELIBERATE. On the SKY it is a fade to
    // black — space between the stars is supposed to be black, and since P2b there IS a star field
    // to hand over to. On the LIGHTING it fades the physical blend back to MW's authored night
    // ambient — because `calTarget()`'s night row only means something if the servo has a lit ground
    // to meter, and "fade the model out" would otherwise read as "delete the world's only light
    // source".
    //
    // ⚠ THE BAND IS BELOW THE HORIZON (P2b), AND IT WAS 0..5 DEGREES. Ramping out across the first
    // five degrees ABOVE the horizon meant the model was already gone while the sun was still up:
    // measured in play, elev 0.33 deg read ramp 0.01 against elev 6.80 deg reading 1.00, i.e. the
    // sky blacked out roughly twenty game-minutes early at each end of the day and came back the
    // same way. Running the ramp to -4..0 instead lets H-W hold the sky all the way DOWN to the
    // horizon and fade over the first few degrees beneath it, which is what civil twilight does —
    // and it hands over to the stars rather than to nothing.
    //
    // ⚠ THE **COOK'S** ELEVATION STAYS CLAMPED AT 0 (skyPhysicalMeasure's std::max(0.0, elev)) while
    // this ramp runs negative. That split is the whole trick: below the horizon the model would be
    // EXTRAPOLATION, so the sky holds its horizon-elevation colour and fades out, rather than being
    // asked for a configuration Hosek-Wilkie does not define.
    constexpr float kNightRampLoDeg = -4.0f;
    constexpr float kNightRampHiDeg =  0.0f;
    inline float nightRamp(float solarElevationRad)
    {
        const float deg = solarElevationRad * 180.0f / (float)kPi;
        const float t = (deg - kNightRampLoDeg) / (kNightRampHiDeg - kNightRampLoDeg);
        const float u = std::max(0.0f, std::min(1.0f, t));
        return u * u * (3.0f - 2.0f * u);   // smoothstep: C1 at both ends, so no step at dawn
    }

    // ─── THE COEFFICIENT GATE ────────────────────────────────────────────────────────────────────
    // Three configurations and twelve radiances, produced by the AUTHORS' implementation compiled
    // against their own release and captured here. Between them they exercise every path in cook():
    // both albedo halves, both turbidity neighbours (integer and fractional), the elevation warp
    // near both ends, and the `it == 10` early-out's neighbour.
    struct RefProbe { float cosTheta, gamma, L[3]; };
    struct RefCase  {
        float    turbidity, albedo, elevation;
        float    cfg[3][9];
        float    rad[3];
        RefProbe probe[4];
    };
    static const RefCase kRefCases[] = {
    { 3.000000f, 0.100000f, 0.721519113f,   // the reference clear day: T 3, ash albedo, sunDir.z = -0.66
      {
        { -1.069756992e+00f, -1.553866817e-01f, 1.445002057e+00f, 1.993388011e+00f, -2.740731354e+00f, 1.229670445e+00f, 1.772794315e-01f, 9.741822577e-01f, 7.045681465e-01f, },
        { -1.080544502e+00f, -1.759380090e-01f, 1.133385235e+00f, 1.117433410e+00f, -4.739784242e+00f, 1.068522293e+00f, 1.282281414e-01f, 1.909424548e+00f, 6.798772456e-01f, },
        { -1.083206859e+00f, -2.187370107e-01f, 6.566736039e-01f, 1.314122883e-02f, -1.066906378e+00f, 7.268108641e-01f, 6.904622987e-02f, 2.642481308e+00f, 6.594939413e-01f, },
      },
      { 6.787016469e+00f, 1.013846726e+01f, 1.528985985e+01f },
      {
        { 1.000000000e+00f, 8.492772140e-01f, { 2.106288222e+00f, 3.705107844e+00f, 7.529882321e+00f, } },
        { 1.570731731e-02f, 1.570796327e+00f, { 1.144671805e+01f, 1.464196195e+01f, 1.575246998e+01f, } },
        { 6.605259884e-01f, 2.000000000e-02f, { 1.957413890e+01f, 2.148955019e+01f, 2.346139991e+01f, } },
        { 7.071067812e-01f, 2.356194490e+00f, { 2.775733580e+00f, 5.208588035e+00f, 1.008280216e+01f, } },
      },
    },
    { 6.400000f, 0.350000f, 0.139626340f,   // hazy + bright ground + low sun: BOTH interpolations at once
      {
        { -1.184469513e+00f, -2.449870588e-01f, 1.894302502e-01f, 1.759528546e+00f, -2.563094062e+00f, 3.376313232e-01f, 2.676839025e-01f, 1.572652070e+00f, 6.803130550e-01f, },
        { -1.195307459e+00f, -2.814678437e-01f, 1.041370058e-01f, 1.208606825e+00f, -2.621687834e+00f, 3.913647604e-01f, 1.701923012e-01f, 2.324596861e+00f, 6.772719272e-01f, },
        { -1.243977610e+00f, -4.032285230e-01f, -7.636176478e-01f, 8.999832943e-01f, -7.738251709e-02f, 4.595209340e-01f, 5.788631352e-02f, 2.642363903e+00f, 6.752650018e-01f, },
      },
      { 8.662877371e+00f, 7.917405588e+00f, 6.412576106e+00f },
      {
        { 1.000000000e+00f, 1.431169987e+00f, { 1.226093933e+00f, 1.953442031e+00f, 2.902180205e+00f, } },
        { 1.570731731e-02f, 1.570796327e+00f, { 4.930690242e+00f, 4.051640846e+00f, 2.548814235e+00f, } },
        { 1.391731010e-01f, 2.000000000e-02f, { 1.275508363e+02f, 8.165285245e+01f, 2.908050545e+01f, } },
        { 7.071067812e-01f, 2.356194490e+00f, { 2.456423394e+00f, 3.547881710e+00f, 4.593837072e+00f, } },
      },
    },
    { 1.000000f, 0.000000f, 1.569050998f,   // the corner of the domain: T and albedo at their minima
      {
        { -1.138678934e+00f, -1.795871679e-01f, 1.928208882e+00f, 6.775179235e+00f, -2.374024189e+00f, -1.052677391e+00f, 1.709069976e-01f, 1.523515328e+00f, 5.019304395e-01f, },
        { -1.075507379e+00f, -1.244499090e-01f, 1.427508423e+00f, 8.783094997e+00f, -2.933514199e+00f, 1.486318741e+00f, 3.253195916e-02f, 3.879971717e+00f, 5.001565263e-01f, },
        { -1.087242311e+00f, -1.886708135e-01f, 8.158553668e-01f, 3.078422035e-01f, -2.150873494e+00f, 1.420415309e+00f, 9.663357691e-02f, 3.126358824e+00f, 5.003509269e-01f, },
      },
      { 4.324782720e+00f, 8.086709728e+00f, 1.390166108e+01f },
      {
        { 1.000000000e+00f, 1.745329252e-03f, { 2.411591853e+00f, 6.383705304e+00f, 9.836067376e+00f, } },
        { 1.570731731e-02f, 1.570796327e+00f, { 1.038521413e+01f, 1.623359432e+01f, 1.788283919e+01f, } },
        { 9.999984769e-01f, 2.000000000e-02f, { 2.352974444e+00f, 6.202017652e+00f, 9.816296318e+00f, } },
        { 7.071067812e-01f, 2.356194490e+00f, { 1.376119801e+00f, 4.231696370e+00f, 9.614013445e+00f, } },
      },
    },
    };

    // Returns the worst RELATIVE deviation seen. `log` is handed a formatted line per case so the
    // caller decides where it goes (LOG::logline in the host, printf in a bare test).
    template <typename LogFn>
    inline double selfTest(LogFn log)
    {
        double worstCfg = 0.0, worstL = 0.0;
        for (const RefCase& rc : kRefCases) {
            State s = {};
            cook(rc.turbidity, rc.albedo, rc.elevation, s);
            double wc = 0.0;
            for (int c = 0; c < 3; ++c) {
                for (int i = 0; i < 9; ++i) {
                    const double d = std::fabs(s.cfg[c][i] - rc.cfg[c][i])
                                   / std::max(1.0e-6, std::fabs((double)rc.cfg[c][i]));
                    wc = std::max(wc, d);
                }
                const double dr = std::fabs(s.rad[c] - rc.rad[c])
                                / std::max(1.0e-6, std::fabs((double)rc.rad[c]));
                wc = std::max(wc, dr);
            }
            double wl = 0.0;
            for (const RefProbe& p : rc.probe) {
                for (int c = 0; c < 3; ++c) {
                    const double v = evalNative(s, p.cosTheta, p.gamma, c);
                    wl = std::max(wl, std::fabs(v - p.L[c]) / std::max(1.0e-6, std::fabs((double)p.L[c])));
                }
            }
            log(rc.turbidity, rc.albedo, rc.elevation, wc, wl);
            worstCfg = std::max(worstCfg, wc);
            worstL   = std::max(worstL,   wl);
        }
        return std::max(worstCfg, worstL);
    }

}   // namespace Hosek
