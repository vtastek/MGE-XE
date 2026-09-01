// mgeHost64 — THE SCENE'S CALIBRATION LAYER: what one scene unit is worth, and the small utility
// set that every absolute lane in the renderer is stated against (tasks/forge-atmosphere.md S2f).
//
// ─── WHY THIS FILE EXISTS AT ALL ─────────────────────────────────────────────────────────────────
//
// Every constant below used to live in `hosek.h`, and NOT ONE OF THEM WAS EVER HOSEK'S. They are
// the renderer's own: what a scene 1.0 means in cd/m², the photometric constant that converts a
// model's radiance into it, the measured clear-day illuminance the whole calibration closes on, the
// Fibonacci quadrature the ambient is integrated over, the luma weights, pi. Hosek-Wilkie merely
// happened to be the first physical sky, so they were written down beside it.
//
// That accident becomes a hazard the moment the sky model is replaced. `forgerender.cpp` reaches
// into `Hosek::` twelve times for kPi alone and six for luma709, and deleting the model while those
// references stand would mean doing two things in one change: swapping the physics AND moving the
// calibration. If the result then looked wrong, nobody could say which half did it. So the move
// happens FIRST, on its own, and it is BEHAVIOUR-FREE BY CONSTRUCTION — every value here is
// byte-identical to the one it replaced, and `hosek.h` now aliases these rather than defining its
// own, so the two cannot drift apart while both exist.
//
// ⚠ kSceneUnitCd IS PINNED AND STAYS PINNED. It is the single number the emissive k_ref/area_ref,
// every calTarget() row and the exposure servo are all stated against
// ([[project_forge_exposure_couples_every_level]]). A new sky model that does not reproduce the
// reference illuminance is a model to fix, NOT a reason to re-pin this — re-pinning would move every
// absolute lane in the renderer at once under the guise of a sky change.
#pragma once

#include <algorithm>
#include <cmath>

namespace SceneCal {

    constexpr double kPi     = 3.14159265358979323846;
    constexpr double kHalfPi = kPi * 0.5;

    // ─── WHAT ONE SCENE UNIT IS WORTH ────────────────────────────────────────────────────────────
    // P1 derived `unitCd = kClearDayLux / pi / Epi` per frame from MW's own lighting; that could not
    // survive into P2, because P2 SETS the lighting and the derivation would close on itself. So it
    // is pinned here, to exactly the value P1 measured on the reference save, out of the same two
    // named numbers it used. Pinning is not a downgrade — it is the requirement: the emissive and
    // lantern calibration is stated in scene units and only keeps meaning what it meant if the unit
    // stops moving. [[project_emissive_kref_rederived]] [[project_emissive_area_ref_measured]]
    constexpr double kClearDayLux = 90000.0;    // a real clear day's horizontal illuminance (P1)
    constexpr double kRefEpi      = 0.9046;     // P1's MEASURED E_total/pi on the reference save
    constexpr double kSceneUnitCd = kClearDayLux / kPi / kRefEpi;   // ~= 31,670 cd/m^2 per scene 1.0
    // Native radiance units (W/(m^2 sr), linear sRGB primaries) -> SCENE units. 683 lm/W is the
    // photometric constant; dividing by kSceneUnitCd lands the result where the renderer lives.
    //
    // ⚠ THE SAME CONVENTION BINDS EVERY PHYSICAL SOURCE, which is what makes one constant enough:
    // a triple (R,G,B) is in native units iff luma709(RGB) * 683 is its photometric luminance. The
    // Hosek dataset was MEASURED to satisfy that; the atmosphere's own integral satisfies it by
    // construction, because its solar irradiance is projected through the CIE observer onto the same
    // primaries (atmosphere.cpp). Two models, one unit, no second conversion.
    constexpr double kNativeToScene = 683.0 / kSceneUnitCd;

    // ─── THE REFERENCE CONFIGURATION ─────────────────────────────────────────────────────────────
    // The one configuration every gate in the programme is measured at, and the reason the
    // atmosphere table's row 0 is not allowed to be a look.
    //
    // ⚠ 41.34 DEGREES SURVIVES ON A DIFFERENT ARGUMENT THAN THE ONE IT WAS PICKED FOR. It was MW's
    // clear-weather `sunDir.z = -0.66`, i.e. the elevation of the LIGHT — and the sky is cooked at
    // the DISC now, which on that same frame sits 29 degrees higher ([[project_mw_two_suns]]). What
    // keeps it is that §0.1's `E_sun/E_total = 0.80` is a POPULATION median over eight clear sunlit
    // HDRIs whose solar elevations are unknown, sun share is strongly elevation-dependent, and the
    // constraint therefore has to be applied at SOME elevation — a mid-range one minimises the
    // extrapolation to either end.
    constexpr double kRefAlbedo    = 0.10;      // Vvardenfell is ash
    constexpr double kRefElevation = 41.34 * kPi / 180.0;
    constexpr double kRefSunShare  = 0.80;      // §0.1's measured clear-sky energy split (MEDIAN)
    // ...and its measured spread, over the same eight clear sunlit HDRIs. ⚠ THE GATE CHECKS AGAINST
    // THE BAND, NOT AGAINST THE MEDIAN +- SOMETHING. A tolerance invented around a median is exactly
    // the kind of number [[feedback_prior_art_constants_dont_transfer]] warns about: it looks like a
    // measurement and is nobody's. q_zenith below has always been treated this way and sun% now
    // matches it, which also means the two rows of the gate are read the same way.
    constexpr double kSunShareLo   = 0.73;      // p25
    constexpr double kSunShareHi   = 0.84;      // p75
    // The measured clear-sky band for the albedo-free invariant q = L_zenith / (E_total/pi).
    // ⚠ REPORTED, NEVER TARGETED. The moment anything tunes to it, it stops being a test.
    constexpr double kQZenithLo    = 0.093;     // p25 of the measured population
    constexpr double kQZenithRef   = 0.122;     // its median
    constexpr double kQZenithHi    = 0.148;     // p75

    inline float luma709(const float v[3])
    {
        return 0.2126f * v[0] + 0.7152f * v[1] + 0.0722f * v[2];
    }

    // ─── THE QUADRATURE, SHARED SO NO TWO INTEGRALS CAN DISAGREE ─────────────────────────────────
    // Fibonacci sphere: z is uniform over [-1,1], which IS uniform in solid angle, so every sample
    // carries the same weight 4*pi/N and the integral is a plain sum. Every consumer — the ambient's
    // SH projection, the sky's irradiance, and now the GPU's own SH pass (atmos_sh.comp reproduces
    // this exact sequence) — walks the same table, because a fraction of a percent of disagreement
    // between two quadratures over one field is the kind of drift nobody can ever attribute.
    // Measured against a 400x800 tabulated integral: 0.06% at N = 128.
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

    // ─── NIGHT ───────────────────────────────────────────────────────────────────────────────────
    // ⚠⚠ THIS RAMP DOES TWO DIFFERENT JOBS WITH ONE NUMBER, AND S2e SPLITS THEM.
    //
    //   the SKY job     — fade the drawn atmosphere to black, because Hosek-Wilkie degrades below a
    //                     few degrees of solar elevation and is undefined beneath 0, so it had to be
    //                     ramped OUT rather than extrapolated.
    //   the LIGHTING job — hand the physical blend back to MW's authored night ambient, because
    //                     calTarget()'s night row and the moonlit trim are calibrated against MW's
    //                     own night and only mean something if the servo has that to meter.
    //
    // The participating medium removes the FIRST reason entirely: a sun below the horizon is a legal
    // configuration for a medium, so twilight is computed rather than faded, and ozone is what makes
    // it violet/indigo instead of muddy brown. The SECOND reason is untouched and the ramp survives
    // for it verbatim — deleting it wholesale would silently move the night calibration.
    //
    // ⚠ THE BAND IS BELOW THE HORIZON (-4..0 deg), AND THAT WAS MEASURED. Ramping across the first
    // five degrees ABOVE it meant the model was already gone while the sun was still up: elev 0.33
    // deg read ramp 0.01 against elev 6.80 deg reading 1.00, i.e. roughly twenty game-minutes of
    // blackout at each end of the day.
    constexpr float kNightRampLoDeg = -4.0f;
    constexpr float kNightRampHiDeg =  0.0f;
    inline float nightRamp(float solarElevationRad)
    {
        const float deg = solarElevationRad * 180.0f / (float)kPi;
        const float t = (deg - kNightRampLoDeg) / (kNightRampHiDeg - kNightRampLoDeg);
        const float u = std::max(0.0f, std::min(1.0f, t));
        return u * u * (3.0f - 2.0f * u);   // smoothstep: C1 at both ends, so no step at dawn
    }

}   // namespace SceneCal
