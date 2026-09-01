// mgeHost64 — THE ATMOSPHERE, host side: the one number the medium cannot make up for itself.
// tasks/forge-atmosphere.md, phase S2.
//
// ─── WHY THIS TU EXISTS, AND WHY IT IS THE ONLY THING IN IT ──────────────────────────────────────
//
// A participating medium computes everything it shows from its own coefficients — EXCEPT the light
// entering it. Bruneton's integral is `L = f(medium) x E_TOA`, and E_TOA, the solar irradiance at
// the top of the atmosphere, is an astronomical measurement rather than a rendering parameter.
//
// ⚠⚠ AND THIS IS THE CONSTANT THAT MAKES THE S2 GATE A TEST INSTEAD OF AN IDENTITY.
// Hosek's `sunBeamScale()` SOLVED its beam so that the reference configuration lands at
// `E_sun/E_total == 0.80` by construction — so the heartbeat's `sun%=0.80` was an identity, not a
// measurement, and the 94 klx it also reported was that same solve seen from another angle. Under
// the medium the beam is `E_TOA x transmittance(h_cam, theta_sun)`, with NO free scale anywhere in
// it. That is what turns `sun%`, `q_zenith` and the 94 klx into genuine predictions that CAN FAIL —
// and the whole reason S2b is a gate rather than a formality.
//
// ⚠ SO E_TOA IS NEVER FITTED, IN EITHER DIRECTION. If the gate misses, the honest lever is the
// medium's AEROSOL OPTICAL DEPTH — which moves the beam and the sky together, as it physically must
// — and never this number, and never `kSceneUnitCd` ([[project_forge_exposure_couples_every_level]]).
// A solar constant tuned to make a gate pass is a solve wearing a measurement's clothes, which is
// exactly the thing being deleted.
//
// ─── WHAT IS INTEGRATED, AND WHAT THAT BUYS ──────────────────────────────────────────────────────
//
// Two published pieces of prior art, composed:
//
//   1. THE SOURCE — a Planck spectral distribution at the sun's effective temperature (5772 K, the
//      IAU 2015 nominal value), normalised so its FULL-spectrum integral is the solar constant
//      1361 W/m^2 (IAU 2015 nominal total solar irradiance). The normalisation is analytic
//      (Stefan-Boltzmann), so there is no wide numerical integral to get subtly wrong.
//   2. THE OBSERVER — the CIE 1931 2-degree colour-matching functions, in Wyman/Sloan/Shirley's
//      published multi-lobe piecewise-Gaussian fit (JCGT 2013), whose stated accuracy is ~1% of
//      peak. A fit rather than the 471-entry table, deliberately: eight lines of arithmetic cannot
//      be mistyped in a way that builds and merely looks "a bit off", which is exactly the failure
//      hosek_data.h needed a 3600-constant self-test to guard against.
//
// The product, integrated over 360..830 nm and mapped XYZ -> linear sRGB, gives E_TOA per PRIMARY in
// the renderer's native convention — the one scenecal.h states: a triple is in native units iff
// `luma709(rgb) * 683` is its photometric value. That identity is EXACT here rather than assumed,
// because the 709 luma weights ARE the middle row of the sRGB->XYZ matrix, so luma709(RGB) == Y by
// construction. The Hosek dataset had to be MEASURED to satisfy the same convention; this one
// satisfies it structurally, and the two therefore share kSceneUnitCd without a second conversion.
//
// ⚠ IT IS CORROBORATED, NOT ASSERTED. solarReport() carries three independent falsification tests to
// the log, none of which anything is tuned to:
//   * luminous efficacy of the source, which for a ~5800 K body must land near 93 lm/W;
//   * the resulting extraterrestrial illuminance, against the published 127-134 klx;
//   * the observer's own normalisation, `integral(ybar) d(lambda) ~= 106.86`, which is a property of
//     the CIE table and therefore a direct check on the FIT rather than on the sun.
// A blackbody is an idealisation of the real solar spectrum — it has no Fraunhofer lines and is
// slightly blue-rich against a measured AM0 curve — and the report is where that shows if it
// matters. What it must not become is a place to nudge a number until a gate goes green.

#include "atmosphere.h"

#include <cmath>

namespace Atmosphere {
namespace {

    // --- The source ------------------------------------------------------------------------------
    constexpr double kSolarConstantWm2 = 1361.0;      // IAU 2015 nominal total solar irradiance
    constexpr double kSolarEffTempK    = 5772.0;      // IAU 2015 nominal solar effective temperature
    constexpr double kStefanBoltzmann  = 5.670374419e-8;
    constexpr double kPlanckH          = 6.62607015e-34;
    constexpr double kBoltzmannK       = 1.380649e-23;
    constexpr double kLightC           = 2.99792458e8;

    // Spectral radiance of a blackbody, W / (m^2 sr m). lambda in METRES.
    double planckRadiance(double lambdaM, double tempK)
    {
        const double a = 2.0 * kPlanckH * kLightC * kLightC;
        const double b = (kPlanckH * kLightC) / (lambdaM * kBoltzmannK * tempK);
        const double l5 = lambdaM * lambdaM * lambdaM * lambdaM * lambdaM;
        return a / (l5 * (std::exp(b) - 1.0));
    }

    // --- The observer: Wyman, Sloan & Shirley 2013, multi-lobe piecewise-Gaussian fit to the
    // CIE 1931 2-degree colour-matching functions. lambda in NANOMETRES. -------------------------
    double lobe(double x, double mu, double s1, double s2)
    {
        const double t = (x - mu) * ((x < mu) ? (1.0 / s1) : (1.0 / s2));
        return std::exp(-0.5 * t * t);
    }
    double cieX(double l)
    {
        return 1.056 * lobe(l, 599.8, 37.9, 31.0)
             + 0.362 * lobe(l, 442.0, 16.0, 26.7)
             - 0.065 * lobe(l, 501.1, 20.4, 26.2);
    }
    double cieY(double l)
    {
        return 0.821 * lobe(l, 568.8, 46.9, 40.5)
             + 0.286 * lobe(l, 530.9, 16.3, 31.1);
    }
    double cieZ(double l)
    {
        return 1.217 * lobe(l, 437.0, 11.8, 36.0)
             + 0.681 * lobe(l, 459.0, 26.0, 13.8);
    }

    SolarReport s_report = {};
    bool        s_done   = false;

    void integrateOnce()
    {
        if (s_done) { return; }
        s_done = true;

        // The blackbody's FULL-spectrum radiance integral is sigma*T^4/pi (Stefan-Boltzmann), so the
        // scale that turns the shape into an irradiance totalling the solar constant is analytic.
        // Doing it this way rather than by a wide numerical sum is what keeps the visible band's
        // FRACTION of the total honest: a truncated numerical normaliser would silently hand the
        // visible band all of the sun's energy.
        const double shapeTotal = kStefanBoltzmann * std::pow(kSolarEffTempK, 4.0) / kPi;
        const double scale      = kSolarConstantWm2 / shapeTotal;     // W/m^2 per (W/m^2/sr/m)

        double X = 0.0, Y = 0.0, Z = 0.0, ybarIntegral = 0.0;
        for (int nm = 360; nm <= 830; ++nm) {
            const double l  = (double)nm;
            // E_lambda in W/(m^2 nm): the shape is per METRE of wavelength, and d(lambda) = 1e-9 nm.
            const double El = planckRadiance(l * 1.0e-9, kSolarEffTempK) * scale * 1.0e-9;
            const double yb = cieY(l);
            X += El * cieX(l);
            Y += El * yb;
            Z += El * cieZ(l);
            ybarIntegral += yb;                       // the observer's own normalisation check
        }

        // XYZ (D65-referred sRGB primaries) -> linear sRGB. The MIDDLE ROW of the inverse of this is
        // (0.2126, 0.7152, 0.0722), which is why luma709(rgb) == Y exactly and why the native-unit
        // convention scenecal.h states holds by construction rather than by measurement.
        const double r =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
        const double g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
        const double b =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;

        s_report.rgb[0] = (float)((r > 0.0) ? r : 0.0);
        s_report.rgb[1] = (float)((g > 0.0) ? g : 0.0);
        s_report.rgb[2] = (float)((b > 0.0) ? b : 0.0);
        s_report.visibleWm2   = Y;                    // NOT the visible-band power; Y is photopic-weighted
        s_report.lux          = 683.0 * Y;
        s_report.efficacyLmW  = s_report.lux / kSolarConstantWm2;
        s_report.totalWm2     = kSolarConstantWm2;
        s_report.tempK        = kSolarEffTempK;
        s_report.ybarIntegral = ybarIntegral;
        const double sum = X + Y + Z;
        s_report.chromX = (sum > 0.0) ? (X / sum) : 0.0;
        s_report.chromY = (sum > 0.0) ? (Y / sum) : 0.0;
    }

}   // namespace

const float* solarIrradianceTOA()
{
    integrateOnce();
    return s_report.rgb;
}

SolarReport solarReport()
{
    integrateOnce();
    return s_report;
}

}   // namespace Atmosphere
