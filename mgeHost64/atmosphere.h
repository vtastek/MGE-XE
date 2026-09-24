// mgeHost64 — THE ATMOSPHERE: one participating medium for sky, fog and cloud.
// tasks/forge-atmosphere.md, phase S1 (the parameter field; no pixels move yet).
//
// ─── WHAT THIS REPLACES, AND WHY IT IS A STRUCT AND NOT A SLIDER ─────────────────────────────────
//
// The physical sky (P2, hosek.h) is a CLEAR-SKY GENERATOR WITH A FIXED TURBIDITY KNOB, and MW's
// weather has never been connected to it. Four things reported wrong in play are that one gap seen
// from four angles: the overcast sky is a painted grey mesh sitting on a permanently blue field;
// below the horizon is a frozen ring because Hosek inverts there; the fog is one colour and the
// horizon is another because they are two different atmospheres joined by a knee; and there is no
// volume at all. None of those are shader defects. They are all "the sky is a dome, not a medium".
//
// A medium fixes them by construction rather than by tuning, because a single scattering integral
// generates the sky, the aerial perspective and the light on the clouds — so they cannot disagree.
// What that integral needs is not a look; it is a small set of PHYSICAL parameters. This header is
// those parameters, plus the one table in the renderer where a Morrowind weather index is allowed
// to become physics.
//
// ⚠ THE TABLE IS THE ONLY PLACE A WEATHER INDEX BECOMES PHYSICS. If a `curWeather == Ash` test ever
// appears anywhere else in the host, the weather has stopped being a parameter and gone back to
// being a special case — which is the state this work exists to leave. Add a row, or add a field to
// a row; do not add a branch.
//
// ⚠ PARAMETERS ARE A FIELD FROM THE FIRST LINE. Everything reads the atmosphere through ONE
// sampler, atmosphereAt(worldX, worldY). Through S5 the map is 1x1 and the sampler returns the
// global row every time; S6 makes it a real low-res texture advected by the wind, and regional
// weather (blight sitting over the Ashlands whether or not you are standing in it, an ash storm
// arriving FROM a direction) drops in instead of forcing a rewrite. The cost of that promise is
// paid entirely by consumers: a march must integrate THROUGH the sampler and must never hoist one
// call out of its loop. That is the whole discipline, and it is cheap only while it is kept.
//
// ⚠ NOTHING HERE RENDERS ANYTHING IN S1. The row is shipped, lerped and logged. The LUTs
// (transmittance / multiscatter / sky-view / aerial perspective) land in S2 alongside
// atmosphere.cpp, and until they do the sky is still Hosek's. That is deliberate: a wire and a
// table that move no pixels are separately verifiable, and the calibration gate that S2 has to pass
// (E_sun/E_total ~= 0.80, q_zenith ~= 0.122, direct-normal ~= 94 klx) means nothing if the weather
// feeding it is not already known-good.
#pragma once

#include <algorithm>
#include <cmath>

#include "ipc/weatherwire.h"
#include "scenecal.h"

namespace Atmosphere {

    using SceneCal::kPi;

    // ─── THE REFERENCE ATMOSPHERE ────────────────────────────────────────────────────────────────
    // Earth at sea level, per metre, at the linear sRGB primaries. These are Bruneton's published
    // constants in Hillaire's packaging, NOT values fitted to anything of ours — every weather row
    // below is a MULTIPLIER on them, so "clear" is literally Earth and the whole table is readable
    // as departures from a known atmosphere. [[feedback_prior_art_constants_dont_transfer]] applies
    // to what we DERIVE from them, not to the coefficients themselves, which are optical
    // properties of air and dust rather than someone's tuning.
    //
    // ⚠ These are here rather than in the shader so the S2 gate can be computed on the CPU against
    // the same numbers the GPU integrates. Two copies of a coefficient is exactly the defect the SH
    // readback exists to delete from the Hosek path; do not start a second one here.
    constexpr float kRayleighSeaLevel[3] = { 5.802e-6f, 13.558e-6f, 33.100e-6f };  // 1/m
    constexpr float kRayleighHeightKm    = 8.0f;
    // Aerosol at 550 nm, single-scattering albedo 0.9 — the 0.9 is where kMieAbsorbFractionClear
    // comes from and it is the number every row's `mieAbsorption` is a departure from.
    //
    // ⚠⚠ THIS IS THE ONE COEFFICIENT ON THIS PAGE THAT IS NOT BRUNETON'S, AND THE UNIT THAT SETTLES
    // IT IS METEOROLOGICAL VISIBILITY. Bruneton's value is 3.996e-6 /m of scattering (4.44e-6
    // extinction) at a 1.2 km scale height — his kMieAngstromBeta 5.328e-3 divided by that height —
    // and Koschmieder turns any extinction into a distance a human can check:
    //
    //     visibility = 3.912 / beta_ext        (the 2% contrast threshold)
    //
    // 4.44e-6 /m is a visibility of 881 km. That is not a clear day, it is not a mountain top, it is
    // not anywhere: the clearest air ever measured at sea level is ~100 km and the WMO's top
    // category ("exceptionally clear") starts at 50. Applied across the whole table it made an ASH
    // STORM read 44 km and a clear day 1469 — every weather Morrowind has, from a light haze to a
    // blight storm, sitting between 33 and 1469 km. So the ROWS were authored sanely as relative
    // departures and the BASE they departed from was ~40x below any real atmosphere.
    //
    // x40 is measured, not chosen. It is the value at which the clear-sky gate's own rows land, and
    // they are rows nothing here is fitted to — the sweep, after the two multiscatter fixes that had
    // to come first (see atmos_multiscatter.comp.fsl):
    //
    //     mieMul     1      8     20     32     40     48
    //     hor/zen   8.38   5.23   3.10   2.28   1.98   1.77    band 2.0-4.0
    //     zenith    1218   1387   1668   1933   2102   2264    band 2000-8000 cd/m2
    //     q         .051   .059   .072   .084   .092   .100    band 0.093-0.148
    //     sun%      .915   .889   .848   .810   .786   .763    band 0.73-0.84
    //     beam klx  103.3   99.8   94.1   88.6   85.2   81.9   ~94 expected
    //     ENERGY    90.1%  89.5%  88.5%  87.3%  86.5%  85.6%   must be <= 100
    //
    // At 40 the beam, sun%, zenith and ENERGY rows are all in band and the last two — q and hor/zen —
    // miss by 0.9% and 1.0% IN OPPOSITE DIRECTIONS, so no aerosol satisfies both and chasing the
    // residual would be fitting. What it buys in the table's own units: Clear 37 km, Cloudy 22,
    // Overcast 18, Rain 12, Thunder 7.9, Snow 7.8, Blizzard 1.9, Foggy 4.0, Ash 1.1, Blight 0.8.
    // Every row lands where meteorology puts that weather, and the ORDER the author intended is
    // untouched because no row's multiplier moved.
    //
    // ⚠⚠ AND x40 WAS SHIPPED, PHOTOGRAPHED AND REVERTED THE SAME DAY. THE PICTURE REFUTED IT, AND
    // THE REASON IS THE PHASE FUNCTION RATHER THAN THE OPTICAL DEPTH. The gate is measured at a
    // 41.34 deg sun, where the zenith sits 49 deg from the sun and the aerosol's forward lobe is
    // irrelevant. Play is not: at Sadrith Mora, sun 78.6 deg, the ZENITH IS 11 DEGREES FROM THE SUN,
    // and Henyey-Greenstein at g 0.80 returns P(11.4 deg) = 1.483 /sr against Rayleigh's 0.117 —
    // 12.7x per unit scattering, on top of a Mie column that x40 made larger than the Rayleigh one.
    // Measured, same save, same frame, only this constant different:
    //
    //                        zenith cd/m2            q      sky lx   the picture
    //     Bruneton   (1399,  2128,  4154)   deep blue    0.058     7071   blue sky, read shadows
    //     x40        (20650, 17737, 16926)  WHITE        0.510    18148   pale, flat, washed
    //
    // A real sky does have a bright aureole around a high sun, but HG smears the diffraction peak
    // across 10-30 degrees where a real polydisperse aerosol confines it to a few — so at any
    // realistic optical depth HG paints the whole near-solar sky white. And no g fixes it: lowering
    // it trades the aureole for side-scatter, which whitens the deep blue instead. THE OPTICAL DEPTH
    // AND THE PHASE FUNCTION ARE ONE DECISION, and this file only has an honest value for one of
    // them. Raising the AOD is blocked on the phase function, not on more sweeping.
    //
    // The Koschmieder argument above stands and is why this is recorded rather than deleted: 881 km
    // is not a visibility Earth has, and the whole table is 33-1469 km. The measurement that closes
    // it needs a phase function a real aerosol would recognise. Until then Bruneton's coefficient
    // ships, `atmosMieMul` is the arm, and the sweep above is the table to resume from.
    // [[feedback_the_standin_was_doing_the_real_job]] in reverse: here the physically-correct number
    // was the one the picture rejected, and the gate could not see it because its ONE sun angle
    // never puts the zenith near the sun.
    //
    // ⚠⚠ FOLDED 2026-09-24: x30, UNDER THE S2l PHASE (spike + Cornette-Shanks, atmosMiePhase 1).
    // The phase function WAS the blocker, as the note above said. With a phase fitted to BHMIE over
    // real aerosols, the S2l P3 sweep found x30 the ONLY optical depth that passes every row of both
    // clear frames (the 41.34 deg gate and the 78.6 deg Sadrith Mora frame); the window is ~x29-31,
    // and Koschmieder puts Clear at 49 km, inside the 20-50 km cross-check. The user chose it from the
    // pictures (hdrdump/pics/s31-ebonheart, S3.1: "more dramatic, as it should be") over the deep-blue
    // x1. It is also the medium S3's aerial perspective will put on the ground, so this sets the look
    // of both. Bruneton's 3.996e-6 x 30. A/B: atmosMieMul 0.0333 + atmosMiePhase 0 = the old sky.
    constexpr float kMieScatterSeaLevel  = 3.996e-6f * 30.0f;   // 1/m  (Bruneton x30, S2l/S3.1)
    constexpr float kMieHeightKm         = 1.2f;
    constexpr float kMieAbsorbFractionClear = 0.10f;    // 1 - single-scattering albedo
    // The aerosol's diffraction SPIKE (S2l). One constant for every weather because free fits put it
    // at 0.965-0.972 for all three BHMIE references; a row says how much light is in it (mieSpike),
    // never how sharp it is. ⚠ The sky-view LUT resolves ~3 deg near the zenith, so the inner
    // degrees of this lobe are integrated rather than drawn — that is S2l/P4's business, not a reason
    // to blunt it here.
    constexpr float kMieSpikeG           = 0.97f;
    // Ozone: a stratospheric TENT, peak absorption at 25 km, zero by 10 and 40 km. It has no
    // scattering term at all — it only removes light, and it removes it where Rayleigh does not,
    // which is the entire reason twilight is violet rather than brown. Hosek has no ozone term.
    constexpr float kOzoneAbsorb[3]      = { 0.650e-6f, 1.881e-6f, 0.085e-6f };    // 1/m at peak
    constexpr float kOzoneCentreKm       = 25.0f;
    constexpr float kOzoneWidthKm        = 15.0f;       // half-width; the tent spans centre +- this

    // ─── THE CLOUD DECK'S OPTICS (S4a) ───────────────────────────────────────────────────────────
    // ⚠⚠ THESE FIVE FIELDS WERE CARRIED BY THE WIRE, INTERPOLATED, PRINTED ON THE HEARTBEAT AND READ
    // BY NO SHADER FOR THE WHOLE OF S1-S3, and the ten-weather sweep (2026-08-31) is what that cost:
    // direct-normal illuminance moved 109364 -> 107623 lx, i.e. 1.6%, going from a clear sky to a
    // TOTAL OVERCAST. Aerosol cannot stand in for a lid — Overcast's `mie 2.0x abs 0.30` is an
    // optical depth of order 0.05 against a real deck's 10-50, three orders short, and closing the
    // gap by raising absorption DARKENS a lid that should be BRIGHT.
    //
    // So the deck is a fourth density profile inside the same integral (atmosphere.h.fsl), and these
    // are the constants that turn a weather row's cover/type/geometry/precipitation into it. Every
    // one of them is an optical property of water cloud rather than a fitted look:
    //
    //   tau        a stratus overcast runs 10-50 in the literature; 24 at full cover is a solid,
    //              ordinary lid, and precipitation thickens it toward a storm's.
    //   omega      0.9999. ⚠ A WATER CLOUD IS BRIGHT BECAUSE IT SCATTERS, NOT BECAUSE IT GLOWS, and
    //              it barely absorbs at all in the visible. Rain and snow add a trace of absorption
    //              (larger drops, and the drizzle below the base), which is what makes a storm deck
    //              read dark from underneath while an overcast one reads luminous.
    //   g          0.84-0.88. Droplets are hundreds of wavelengths across: a strong forward lobe.
    //
    // ⚠ AND cover ENTERS AS cover^k, k > 1, BECAUSE A HORIZONTALLY-HOMOGENEOUS MEDIUM CAN ONLY EVER
    // REPRESENT PARTIAL COVER AS A MEAN FIELD — a uniformly thinner lid, never broken cumulus with
    // blue between it. A linear mean field would grey the entire sky at `cover 0.35` (Cloudy), which
    // is exactly what a fair-weather-cumulus day does not look like. Broken cloud is what S4b's
    // volumetric pass is for; this exponent is the honest bound on what S4a can claim.
    constexpr float kCloudTauRef        = 24.0f;    // vertical optical depth at full cover, dry
    constexpr float kCloudTauPrecip     = 16.0f;    // + this x precipitation
    // ⚠⚠ RETIRED 2026-09-09 AND KEPT ONLY AS THIS NOTE. `cover^2.5` existed for exactly one reason
    // — "so a light cover does not grey the whole sky the way a linear mean field would" — and it
    // was a mitigation for a model that could only express partial cover as A THINNER LID. The cover
    // MIXTURE removes the need: cover is a blend weight between a clear sky and a full-depth deck,
    // so a light cover leaves most of the sky untouched instead of veiling all of it faintly.
    //
    // ⚠ AND THE EXPONENT COULD NOT HAVE BEEN TUNED OUT OF THE PROBLEM, WHICH IS WHY IT IS DELETED
    // RATHER THAN RAISED. It set the sky's greyness and the beam's dimming with ONE number, and at
    // Cloudy the beam was already RIGHT (0.616 measured against the 0.65 that 35% cover implies).
    // Raising it to de-grey the sky would have broken the light to do it.
    // [[feedback_one_knob_two_jobs]]
    // constexpr float kCloudCoverExponent = 2.5f;   // cover^k — the MEAN-FIELD exponent
    constexpr float kCloudSsa           = 0.9999f;  // single-scattering albedo, dry
    constexpr float kCloudSsaPrecip     = 0.015f;   // - this x precipitation
    constexpr float kCloudGStratus      = 0.84f;    // HG asymmetry at cloudType 0
    constexpr float kCloudGDeep         = 0.88f;    // ...and at cloudType 1
    // ⚠ THE PROFILE'S SHOULDER AND ITS MEAN ARE ONE FACT WRITTEN TWICE, AND THE SECOND IS DERIVED
    // FROM THE FIRST SO THEY CANNOT DRIFT. atmosCloudProfile() blends a soft-shouldered slab
    // (smoothstep from 1 down to `shoulder`, whose mean over the layer is exactly (1+shoulder)/2)
    // into a raised cosine (mean exactly 1/2). packParams divides tau by that mean, so tau is tau
    // whatever the shape and whatever the thickness — which is what makes "a deck of optical depth
    // 24" a statement anyone can check against a textbook instead of a coefficient nobody can.
    // MUST match the 0.60f in atmosCloudProfile().
    constexpr float kCloudProfileShoulder = 0.60f;
    constexpr float kCloudProfileMeanFlat = 0.5f * (1.0f + kCloudProfileShoulder);   // 0.80
    constexpr float kCloudProfileMeanBell = 0.5f;

    // ─── THE PROFILE ITSELF, HOST-SIDE — A SECOND COPY, DECLARED AS ONE ──────────────────────────
    // ⚠⚠ THIS FILE'S SIBLING (atmosphere.h.fsl) EXISTS SO THAT THERE IS EXACTLY ONE MEDIUM, and
    // this function is a deliberate exception to that rule rather than an oversight. S4a/A2 solves
    // the deck's transport HOST-SIDE (deckSolve below) and hands the shader a small table indexed by
    // ALTITUDE, so the host has to know where inside the deck a given altitude sits in OPTICAL
    // DEPTH — which is the profile's cumulative integral, and there is no way to ask the GPU for it.
    //
    // The exception is bounded three ways, and all three are checkable:
    //   1. it is the SHAPE only — no coefficient, no scale height, no phase function crosses over;
    //   2. deckSolve NORMALISES by this function's own total, so a scale error cannot survive;
    //   3. deckSolve computes this function's MEAN and compares it against kCloudProfileMeanFlat /
    //      ...Bell above, which are the analytic means of the shader's shapes. That ratio is
    //      reported on the gate line as `prof`, and it is 1.000 exactly while the two agree.
    // Row 3 is the cross-check the top of atmosphere.h.fsl says a second copy must come with.
    // MUST match atmosCloudProfile() in atmosphere.h.fsl, including the 0.60f shoulder.
    inline double cloudProfileHost(double x, double shape)
    {
        const double ax = std::min(std::fabs(x), 1.0);
        const double t  = std::max(0.0, std::min(1.0, (1.0 - ax) / (1.0 - (double)kCloudProfileShoulder)));
        const double flat = t * t * (3.0 - 2.0 * t);
        const double bell = 0.5 + 0.5 * std::cos(3.14159265358979323846 * ax);
        return flat + std::max(0.0, std::min(1.0, shape)) * (bell - flat);
    }


    // Planet geometry, metres. Vvardenfell is Earth-sized as far as an atmosphere is concerned —
    // nothing in MW states otherwise and the horizon distance a player can see is set by the
    // terrain, not by the curvature.
    constexpr float kGroundRadiusM = 6360000.0f;
    constexpr float kAtmoRadiusM   = 6460000.0f;        // 100 km top of atmosphere

    // ─── THE MEDIUM AT A POINT ───────────────────────────────────────────────────────────────────
    // What atmosphereAt() returns. Two halves, and the split matters:
    //
    //   - the MEDIUM lanes are physics and are consumed by the scattering integral;
    //   - the mw* lanes are Morrowind's own authored scalars, carried UNCONSUMED in S1.
    //
    // ⚠ THE mw* LANES ARE DELIBERATELY NOT FOLDED INTO THE PHYSICS YET, and that is a measurement
    // decision rather than laziness. Combining MW's `cloudsMaxPercent` with the table's
    // `cloudCoverage` needs a rule, and any rule written today would be an invented constant: the
    // live values MW actually ships per weather are not known from this side of the wire, and the
    // vanilla ini and every weather mod disagree about them. S1's job is to PUT THEM ON THE
    // HEARTBEAT so the rule can be read off a real storm instead of guessed. Fog closes in S3,
    // clouds in S4, and each states its coupling where it consumes it.
    struct Params {
        // -- molecular ------------------------------------------------------------------------
        float rayleighScale;      // x kRayleighSeaLevel. 1 = Earth. Weather barely moves this.
        // -- aerosol: the lane that actually distinguishes MW's weathers ------------------------
        float mieScale;           // x kMieScatterSeaLevel. 1 = clean air, 10 = a dust storm.
        float mieAbsorption;      // ABSORBED FRACTION of aerosol extinction, 0..1. 0.10 = clean air
                                  // (albedo 0.9). High values de-blue and DIM, which is what turns
                                  // a bright haze into an overcast lid or an ash sky.
        float mieG;               // Henyey-Greenstein asymmetry. 0.8 = clean air's forward lobe;
                                  // large mineral grains and ice scatter more broadly.
                                  // ⚠ READ ONLY BY THE SINGLE-HG ARM (atmosMiePhase 0) since S2l.
        // -- the aerosol's PHASE since S2l: a smooth BODY plus a sharp diffraction SPIKE ---------
        // ⚠ NOT A SECOND WAY TO SAY mieG. Single HG is the wrong SHAPE at any g (see
        // atmosPhaseAerosol in atmosphere.h.fsl), so these are fitted to BHMIE over real aerosols by
        // mgeHost64/tools/aerosol_phase_fit.py, not tuned. The fit's finding is that the body barely
        // moves between aerosol types (CS g ~0.60 for continental AND wet haze) while the spike
        // weight is what separates them: dry fine particles 0.00, a wet coarse mode 0.12. Rows with
        // no sphere reference (ash: irregular dust; snow: ice) keep their AUTHORED asymmetry — the
        // CS g whose own <cos> equals the row's HG g (script section 4) — and no spike.
        float mieBodyG;           // Cornette-Shanks shape g (NOT <cos>, which runs ~0.06 higher)
        float mieSpike;           // weight of the HG(kMieSpikeG) diffraction spike, 0..1
        float mieHeightKm;        // aerosol scale height. Fog and ash HUG THE GROUND; this is the
                                  // lane that says so, and it is why fog is not just "more Mie".
        float mieTint[3];         // per-primary multiplier on aerosol EXTINCTION, 1,1,1 = grey.
                                  // Real aerosols are spectrally selective (that is the Angstrom
                                  // exponent, made explicit per primary instead of fitted), and it
                                  // is the ONLY thing separating ash from fog in colour. Crimson,
                                  // Burnt Orange and Lavender Grey are configurations of this.
        // -- stratospheric --------------------------------------------------------------------
        float ozoneScale;         // x kOzoneAbsorb.
        // -- the cloud deck: S4a consumes it OPTICALLY, S4b will draw it -----------------------
        // ⚠ THESE FIVE WERE CARRIED AND CONSUMED BY NOTHING FOR THREE PHASES, and the ten-weather
        // sweep is what that cost: a TOTAL OVERCAST dimmed the sun by 1.6%. S4a turned them into a
        // fourth density profile inside the same integral — see kCloudTauRef above and
        // atmosCloudProfile() in atmosphere.h.fsl. They are still not DRAWN; that is S4b.
        // ⚠⚠ cover IS A MIX WEIGHT SINCE 2026-09-09, AND ITS MEANING CHANGED UNDER THE AUTHORED
        // VALUES. It used to THIN the deck (tau = 24 cover^2.5, applied over the whole sky); it now
        // BLENDS a full-depth cloud against a clear sky. The old reading made 0.95 and 1.00 nearly
        // the same thing (tau 21.1 vs 24.0); the new one makes them 5% blue sky versus none, which
        // is the difference between an overcast day with shadows and one without. Every row was
        // re-read against the new meaning when the mixture landed — see Overcast and Snow below.
        float cloudCoverage;      // 0..1 — the FRACTION OF SKY under cloud (a blend weight)
        float cloudType;          // 0 stratus .. 0.5 cumulus .. 1 cumulonimbus (the density profile)
        float cloudBottomKm;      // base altitude
        float cloudThicknessKm;   // top - base
        float precipitation;      // 0..1; darkens and thickens the deck, and feeds S3's near fog
        // -- Morrowind's own authored scalars, UNCONSUMED in S1 (see the warning above) --------
        // ⚠ mwCloudsMaxPercent IS NOT A COVERAGE, DESPITE ITS NAME — MEASURED, 2026-08-30.
        // Across all ten weathers in the reference install it reads 1.000 for eight of them and
        // 0.660 for Rain and Thunderstorm, so it does not even separate Clear from Blizzard.
        // OpenMW settles what it is: `cloudBlendFactor = transitionRatio / cloudsMaximumPercent`,
        // the rate MW cross-fades one cloud TEXTURE into the next. S4a's cover comes from
        // `cloudCoverage` above and from nothing else. Had S1 folded this lane into the physics on
        // the strength of its name — which is exactly what "cloud coverage" invites — overcast and
        // clear would have shipped the same deck and the error would have been invisible until the
        // clouds were already built on top of it.
        float mwCloudsMaxPercent;
        float mwCloudsSpeed;
        float mwWindSpeed;
        float mwLandFogDay;
        float mwLandFogNight;
    };

    // The non-spatial half of the live weather: state that belongs to the FRAME rather than to a
    // point in the world, so it has no business in a field sampler. Kept beside Params because the
    // heartbeat has to report both to be worth anything.
    struct Live {
        bool  valid;              // false ⇒ interior / menu / no world. Everything else is zero.
        int   cur, next;          // TES3::WeatherType, -1 when absent
        float transition;         // 0 = cur, 1 = next
        float thunderFlash;       // live engine counter, 0 when not flashing
        float sunglareVis;        // 0..1
        bool  sunOccluded;
        float skyColRef[3];       // ⚠ REFERENCES. Never drive a pixel with these — see
        float fogColRef[3];       //   ipc/weatherwire.h. They exist so the log can compare.
    };

    // ─── THE PER-WEATHER TABLE ───────────────────────────────────────────────────────────────────
    // One row per TES3::WeatherType (TES3WeatherController.h: Clear, Cloudy, Foggy, Overcast, Rain,
    // Thunder, Ash, Blight, Snow, Blizzard). Read as departures from Earth-at-sea-level: row 0 IS
    // the reference atmosphere, and every other row says what that weather does to the air.
    //
    // ⚠ OZONE IS 1.0 IN EVERY ROW, AND THAT IS PHYSICS RATHER THAN AN UNFILLED COLUMN. The ozone
    // layer sits at 25 km; no weather Morrowind has reaches it. The lane exists because S6's
    // regional field may want it (and because a twilight tuned by moving ozone would otherwise be
    // tuned by moving something that is not ozone), not because the ten rows should differ.
    //
    // ⚠ Cloud altitudes are in KILOMETRES and are real ones. They are not scaled to MW's world:
    // a 2 km cloud base is ~143000 MW units, far beyond anything the player reaches, which is
    // exactly right — the deck is scenery at infinity, and pretending otherwise is what makes a
    // painted dome look painted.
    constexpr int kWeatherCount = 10;

    constexpr Params kWeatherTable[kWeatherCount] = {
        // Clear — the reference atmosphere, unmodified. Every gate S2 has to pass (q_zenith,
        // E_sun/E_total, direct-normal illuminance) is measured HERE, so this row is not a look and
        // must not be tuned: if clear looks wrong the model is wrong.
        { /*ray*/ 1.00f, /*mie*/ 0.60f, /*abs*/ 0.10f, /*g*/ 0.80f, /*body*/ 0.60f, /*spike*/ 0.00f, /*hkm*/ 1.20f,
          /*tint*/ { 1.00f, 1.00f, 1.00f }, /*ozone*/ 1.00f,
          /*cover*/ 0.00f, /*type*/ 0.30f, /*bot*/ 2.00f, /*thick*/ 0.40f, /*precip*/ 0.00f,
          0,0,0,0,0 },
        // Cloudy — fair-weather cumulus over otherwise clean air. Slightly more aerosol than clear
        // because a sky with cumulus in it is a sky with moisture in it.
        { 1.00f, 1.00f, 0.10f, 0.80f, 0.60f, 0.00f, 1.20f, { 1.00f, 1.00f, 1.00f }, 1.00f,
          0.35f, 0.50f, 1.80f, 0.80f, 0.00f, 0,0,0,0,0 },
        // Foggy — NOT "more haze". Fog is a thick, near-white, ground-hugging aerosol: the defining
        // lane is mieHeightKm 0.25, which puts almost all of it below the player. Droplets are large
        // and nearly non-absorbing (albedo ~0.99), so fog is BRIGHT — it hides the sun without
        // darkening the world, which is exactly how real fog reads.
        //
        // ⚠⚠ C5 TRIED cover 0.40 -> 0.00 HERE AND THE PICTURE REFUTED IT — REVERTED, DELIBERATELY,
        // AND THE PLAN'S "one should go" IS ANSWERED "NEITHER, NOT YET". The overlap is real: the
        // deck sits 50-350 m, inside the 250 m scale height of the aerosol on the line above. But
        // removing the deck does not leave fog behind, it leaves A CLEAR DAY — measured, deck-a2 vs
        // deck-c5 on this row:
        //
        //     sunNormal   52,175 -> 108,745 lx      (the full unobstructed beam)
        //     q           0.5491 -> 0.1421          (back INSIDE the CLEAR-sky band 0.093-0.148)
        //     zenith      (15677,13767,15506) -> (2523,5704,18142)   grey-white -> BLUE
        //
        // ⚠ THE REASON IS THAT THIS ROW'S AEROSOL IS NOT FOG. Its column optical depth is
        // mieScale x hkm = 6.00 x 0.25 = 1.5 against Clear's 0.60 x 1.20 = 0.72 — barely TWICE a
        // clear day. Real fog is a horizontal visibility of tens of metres, i.e. an extinction near
        // 0.05-0.1 /m, which over its own 250 m is an optical depth of 12-25. So the deck was not
        // double-counting the fog; IT WAS DOING THE FOG, and the aerosol lane has been decorative.
        // Thickening it is the actual fix and it is a look change with its own picture, not part of
        // a rollback. Until then the deck stays and the overlap is the lesser error.
        { 1.00f, 6.00f, 0.02f, 0.85f, 0.60f, 0.12f, 0.25f, { 1.00f, 1.00f, 1.00f }, 1.00f,
          0.40f, 0.00f, 0.05f, 0.30f, 0.00f, 0,0,0,0,0 },
        // Overcast — ⚠⚠ C5 ROLLED THIS ROW BACK, AND ITS OLD COMMENT IS THE CONFESSION. It read:
        // "THE LID ... High Mie with real absorption de-blues AND dims the whole atmosphere, so MW's
        // painted cloud layer stops sitting on a bright blue field." That was a FAKE LID built out of
        // aerosol because no real one existed yet (S2), and since S4a a real one does: cover 0.95
        // gives tau 21.1 through a 1.2 km deck. Leaving both in place is the DOUBLE-DARK — the lid
        // twice, once as cloud and once as dirty air.
        //   was  mie 2.00  abs 0.30  g 0.78  hkm 1.60
        //   now  mie 1.20  abs 0.10  g 0.80  hkm 1.20
        // What is left is the air an overcast day actually has UNDER its cloud: humid and slightly
        // hazier than fair weather (Cloudy is 1.00), non-absorbing, at clean air's forward lobe and
        // scale height. ⚠ THE ABSORPTION IS THE TELL, not the density: 0.30 is what DIMS, and no real
        // sub-cloud aerosol absorbs a third of what it intercepts.
        // ⚠ C6: cover 0.95 -> 1.00, FORCED BY THE COVER MIXTURE and not a look change. Under the old
        // thinned-lid reading 0.95 gave tau 21.1 against a full lid's 24 — indistinguishable. Under
        // the mixture it means 5% of the sky is CLEAR, which measured as a 5,506 lx direct beam and
        // sun% 0.159: an overcast day casting shadows. "Overcast" is 8/8 oktas by definition, so the
        // authored intent was always 1.00 and 0.95 was an artefact of a semantics that rounded it off.
        { 1.00f, 1.20f, 0.10f, 0.80f, 0.60f, 0.00f, 1.20f, { 1.00f, 1.00f, 1.00f }, 1.00f,
          1.00f, 0.00f, 1.00f, 1.20f, 0.00f, 0,0,0,0,0 },
        // Rain — full cover, a low wet base, and the aerosol below the deck that rain actually is.
        // ⚠ C5: the DENSITY was always real and the ABSORPTION never was. Falling rain is large water
        // droplets — the same material as fog, which this table already puts at albedo 0.98
        // (abs 0.02) and a strong forward lobe (g 0.85). abs 0.35 was the fake lid's dimming, doing
        // in the aerosol what cover 1.00 (tau 24 through 2.5 km) now does as cloud.
        //   was  mie 2.50  abs 0.35  g 0.76        now  mie 2.00  abs 0.03  g 0.85
        { 1.00f, 2.00f, 0.03f, 0.85f, 0.60f, 0.12f, 1.20f, { 1.00f, 1.00f, 1.00f }, 1.00f,
          1.00f, 0.35f, 0.70f, 2.50f, 0.60f, 0,0,0,0,0 },
        // Thunder — cumulonimbus: the same medium as rain with a deck six kilometres deep, which is
        // what makes a storm cloud dark underneath and bright on top. The underlit-at-sunset case
        // on the brief is this row plus S4's lighting, and no new code.
        // ⚠ C5, same rollback as Rain and for the same reason — heavier rain, still water droplets.
        // "Dark underneath" is the DECK's job now and it does it properly: type 1.00 is the raised
        // cosine, 6 km deep at tau 24, so the base really is starved while the top is lit.
        //   was  mie 3.00  abs 0.40  g 0.74        now  mie 3.00  abs 0.03  g 0.85
        { 1.00f, 3.00f, 0.03f, 0.85f, 0.60f, 0.12f, 1.20f, { 1.00f, 1.00f, 1.00f }, 1.00f,
          1.00f, 1.00f, 0.60f, 6.00f, 1.00f, 0,0,0,0,0 },
        // Ash — Vvardenfell's signature, and the clearest case for the tint lane. Mineral dust is a
        // heavy, strongly ABSORBING, ground-hugging aerosol that scatters broadly (large irregular
        // grains, so a weaker forward lobe than clean air) and removes blue far harder than red.
        // The result is a dim brown-orange sky that darkens the ground rather than glowing, which
        // is what separates an ash storm from fog of the same density.
        { 1.00f, 10.00f, 0.55f, 0.65f, 0.58f, 0.00f, 0.80f, { 1.00f, 0.80f, 0.55f }, 1.00f,
          0.50f, 0.20f, 1.20f, 1.00f, 0.00f, 0,0,0,0,0 },
        // Blight — ash, sicker: denser, more absorbing, and pushed further toward red so the sky
        // reads diseased rather than merely dusty. Same medium, different point in it.
        { 1.00f, 12.00f, 0.60f, 0.62f, 0.55f, 0.00f, 0.80f, { 1.00f, 0.62f, 0.45f }, 1.00f,
          0.60f, 0.20f, 1.20f, 1.20f, 0.00f, 0,0,0,0,0 },
        // Snow — a deck like overcast, but the medium below it is ice crystals: they scatter almost
        // without absorbing and much more isotropically than droplets, which is why falling snow is
        // luminous grey rather than dark.
        // ⚠⚠ C5 DELIBERATELY LEFT SNOW AND BLIZZARD ALONE, AGAINST THE PLAN'S OWN LIST, and the
        // reason is one number. The tell for a fake lid is ABSORPTION, because absorption is the only
        // lane that can DIM — Overcast 0.30, Rain 0.35, Thunder 0.40 were all built to darken a sky
        // that had no cloud in it. These two sit at **0.05**. They cannot darken anything, so there
        // is no double-DARK here to remove; their density is falling precipitation, which is real and
        // is not what the deck represents. What they can be is double-BRIGHT, and that is a picture
        // question rather than a physics one — it is what the C5 picture set is for.
        // ⚠ C6: cover 0.90 -> 1.00, same migration as Overcast. Snow falls out of nimbostratus, which
        // is a complete deck; 10% of clear sky over falling snow is a configuration weather does not
        // have. Ash (0.50), Blight (0.60), Foggy (0.40) and Cloudy (0.35) were left ALONE — their
        // cover is genuinely partial, and for those rows the mixture is the whole point.
        { 1.00f, 3.00f, 0.05f, 0.60f, 0.53f, 0.00f, 1.00f, { 1.00f, 1.00f, 1.00f }, 1.00f,
          1.00f, 0.00f, 0.90f, 1.20f, 0.50f, 0,0,0,0,0 },
        // Blizzard — snow's medium at storm density and pulled down to the ground. Non-absorbing
        // like snow, so a whiteout is BRIGHT; the ash rows are the dark counterpart and the pair is
        // the clearest demonstration that density and absorption are two different lanes.
        { 1.00f, 12.00f, 0.05f, 0.55f, 0.48f, 0.00f, 0.40f, { 1.00f, 1.00f, 1.00f }, 1.00f,
          1.00f, 0.00f, 0.80f, 1.50f, 1.00f, 0,0,0,0,0 },
    };

    inline const char* weatherName(int w) {
        static const char* kNames[kWeatherCount] = {
            "Clear", "Cloudy", "Foggy", "Overcast", "Rain",
            "Thunder", "Ash", "Blight", "Snow", "Blizzard"
        };
        return (w >= 0 && w < kWeatherCount) ? kNames[w] : "none";
    }

    // ─── LIVE STATE ──────────────────────────────────────────────────────────────────────────────
    // File-local by intent: there is NO public accessor for "the global row". The only way to read
    // the atmosphere is through atmosphereAt(), because the moment a consumer can reach past the
    // sampler, S6 becomes a rewrite instead of a drop-in. [[project_forge_no_ini_flips]] is the same
    // lesson from a different direction.
    namespace detail {
        inline Params s_row  = kWeatherTable[0];
        inline Live   s_live = {};

        inline float lerpf(float a, float b, float t) { return a + t * (b - a); }

        inline Params lerpRow(const Params& a, const Params& b, float t) {
            Params o = a;
            o.rayleighScale    = lerpf(a.rayleighScale,    b.rayleighScale,    t);
            o.mieScale         = lerpf(a.mieScale,         b.mieScale,         t);
            o.mieAbsorption    = lerpf(a.mieAbsorption,    b.mieAbsorption,    t);
            o.mieG             = lerpf(a.mieG,             b.mieG,             t);
            // ⚠ LERPING THE PARAMETERS, NOT THE PHASE FUNCTIONS — and for the spike weight those are
            // the same thing (the mixture is linear in it). For the body g they are not, but both
            // ends share g ~0.60 wherever a reference exists, so the walk barely leaves the fit.
            o.mieBodyG         = lerpf(a.mieBodyG,         b.mieBodyG,         t);
            o.mieSpike         = lerpf(a.mieSpike,         b.mieSpike,         t);
            o.mieHeightKm      = lerpf(a.mieHeightKm,      b.mieHeightKm,      t);
            for (int c = 0; c < 3; ++c) {
                o.mieTint[c]   = lerpf(a.mieTint[c],       b.mieTint[c],       t);
            }
            o.ozoneScale       = lerpf(a.ozoneScale,       b.ozoneScale,       t);
            o.cloudCoverage    = lerpf(a.cloudCoverage,    b.cloudCoverage,    t);
            // ⚠ cloudType is lerped like everything else, and that IS the intended behaviour: the
            // lane is a continuous density-profile parameter (stratus -> cumulus -> cumulonimbus),
            // not an enum. A storm growing out of an overcast sky is a walk along it. If it ever
            // becomes an index, the walk becomes a cross-fade and this whole design is lost.
            o.cloudType        = lerpf(a.cloudType,        b.cloudType,        t);
            o.cloudBottomKm    = lerpf(a.cloudBottomKm,    b.cloudBottomKm,    t);
            o.cloudThicknessKm = lerpf(a.cloudThicknessKm, b.cloudThicknessKm, t);
            o.precipitation    = lerpf(a.precipitation,    b.precipitation,    t);
            return o;
        }
    }

    // Called once per frame from the render RPC with MW's live weather row (ipc/weatherwire.h).
    // Turns a weather INDEX PAIR into a medium, which is the one place in the renderer allowed to
    // do that. An invalid row (interior, menu, no world) parks the medium at Clear and clears the
    // frame state; nothing consumes it there, and a parked-at-Clear medium is a safer stale value
    // than a parked-at-Blizzard one.
    inline void setWeather(const IPC::WeatherWire& w) {
        Live lv = {};
        lv.valid = (w.valid != 0);
        lv.cur   = w.cur;
        lv.next  = w.next;
        lv.transition = std::max(0.0f, std::min(1.0f, w.transition));
        if (!lv.valid) {
            lv.cur = lv.next = -1;
            detail::s_live = lv;
            detail::s_row  = kWeatherTable[0];
            return;
        }
        lv.thunderFlash = w.thunderFlash;
        lv.sunglareVis  = w.sunglareVis;
        lv.sunOccluded  = (w.sunOccluded != 0);
        for (int c = 0; c < 3; ++c) {
            lv.skyColRef[c] = w.skyColRef[c];
            lv.fogColRef[c] = w.fogColRef[c];
        }

        // Out-of-range indices fall back to Clear rather than to the last row: a mod that adds an
        // eleventh weather must degrade to "ordinary air", not to "blizzard".
        const int a = (w.cur  >= 0 && w.cur  < kWeatherCount) ? w.cur  : 0;
        const int b = (w.next >= 0 && w.next < kWeatherCount) ? w.next : a;
        Params row = detail::lerpRow(kWeatherTable[a], kWeatherTable[b], lv.transition);

        // MW's authored scalars ride along already interpolated (the client holds both Weather
        // objects; see renderprocess.cpp). Carried, not consumed — see the warning on Params.
        row.mwCloudsMaxPercent = w.cloudsMaxPercent;
        row.mwCloudsSpeed      = w.cloudsSpeed;
        row.mwWindSpeed        = w.windSpeed;
        row.mwLandFogDay       = w.landFogDay;
        row.mwLandFogNight     = w.landFogNight;

        detail::s_live = lv;
        detail::s_row  = row;
    }

    // ⚠ THE ONE SAMPLER. Every consumer of the atmosphere — the LUT cooks, the froxel march, the
    // cloud lighting, the heartbeat — reads it through here and through nothing else.
    //
    // S1-S5: the map is 1x1, so the arguments are ignored and the global row comes back every time.
    // They are in the signature ANYWAY, from the first line, because the difference between S6
    // being a drop-in and S6 being a rewrite is whether the call sites already pass a position.
    //
    // ⚠ A CONSUMER MUST NOT HOIST THIS OUT OF A LOOP. A froxel march that samples once at the
    // camera and integrates a constant is not integrating a field, and it will keep working
    // perfectly until the day the map stops being 1x1 — at which point regional weather is a
    // rewrite of every march instead of a change to this function. That is the entire cost of the
    // promise, and it is paid here or it is paid ten times later.
    inline Params atmosphereAt(float worldX, float worldY) {
        (void)worldX; (void)worldY;   // S6: sample the advected weather map here.
        return detail::s_row;
    }

    // The frame-scoped half: weather identity, transition, and the engine counters that belong to
    // the frame rather than to a place. Separate from atmosphereAt() precisely because these do NOT
    // become a field in S6 — MW runs one global weather at the player and S6 is constrained to
    // agree with it there, which is what makes the sub-simulation a sub-simulation.
    inline const Live& live() { return detail::s_live; }


    // ═══ S2 — WHAT THE MEDIUM NEEDS FROM OUTSIDE ITSELF, AND HOW IT REACHES THE GPU ══════════════

    // ─── THE ONLY CONSTANT THE INTEGRAL CANNOT MAKE UP ───────────────────────────────────────────
    // Solar irradiance at the top of the atmosphere, W/m^2 per linear sRGB primary, integrated ONCE
    // from a Planck source at the sun's effective temperature against the CIE observer — see
    // atmosphere.cpp for the whole derivation and for the three corroborations it prints.
    //
    // ⚠ NEVER FITTED. This is what turns S2's gate from an identity into a prediction: Hosek SOLVED
    // its beam scale so that `E_sun/E_total == 0.80` held by construction, and here the beam is
    // `E_TOA x transmittance` with no free parameter at all. If the gate misses, the lever is the
    // medium's aerosol optical depth — which moves the beam and the sky together — and never this.
    const float* solarIrradianceTOA();

    struct SolarReport {
        float  rgb[3];        // E_TOA per linear sRGB primary, W/m^2 (native units)
        double lux;           // = 683 * Y. Published extraterrestrial illuminance: 127-134 klx
        double efficacyLmW;   // luminous efficacy of the source. A ~5800 K body must land near 93
        double totalWm2;      // the solar constant the source is normalised to
        double visibleWm2;    // the photopic-weighted integral Y (NOT the visible band's power)
        double tempK;
        double ybarIntegral;  // integral of ybar over 360..830 nm. The CIE table's own value: 106.86
        double chromX, chromY;// the source's chromaticity, for eyeballing against the real AM0 sun
    };
    SolarReport solarReport();

    // ─── THE UNIT BRIDGE, IN EXACTLY ONE PLACE ───────────────────────────────────────────────────
    // A Morrowind unit is 1/64 yard (forgerender.cpp: "~1.4 cm"). The LUTs are in METRES because the
    // coefficients above are per metre and because a scale height expressed in MW units would be a
    // number nobody could check against a textbook. Every conversion in the phase goes through here.
    constexpr float kMwUnitToMetre = 0.9144f / 64.0f;    // 0.0142875 m

    // ⚠ THE CAMERA MAY NEVER SIT EXACTLY ON THE GROUND — a float32 fact, not a taste. Every
    // ray-sphere test in the medium computes `dot(ro,ro) - R*R`, and at planet scale both terms are
    // ~4.05e13 against a 24-bit mantissa, so at exactly r == Rg the difference is NOISE OF EITHER
    // SIGN — and that sign decides whether a ray one degree ABOVE the horizon is reported as hitting
    // the ground ten metres away. MUST match ATMOS_GROUND_EPS_M in atmosphere.h.fsl.
    constexpr double kGroundEpsM = 10.0;

    // ─── THE GPU-SIDE PARAMETER BLOCK ────────────────────────────────────────────────────────────
    // The float layout of `AtmosphereParams` in shaders/FSL/atmosphere.h.fsl, filled HERE and
    // nowhere else. The table turns a weather index into a Params; this turns a Params into the
    // coefficients the integral runs on. Two steps, two sites, and neither of them is a branch.
    constexpr int kGpuParamFloats = 17 * 4;     // 17 float4 rows = 272 B (the CBV is 512 B)

    // What the deck came out as, for the heartbeat and the gate. A report, not a second derivation:
    // it recomputes nothing the pack does not, and nothing reads it to shade with.
    // ⚠⚠ `tau` IS THE FULL DECK'S OPTICAL DEPTH SINCE THE COVER MIXTURE (2026-09-09), NOT THE
    // COVER-THINNED ONE, AND THE TWO ARE DIFFERENT NUMBERS. Partial cover used to enter here as
    // `cover^2.5` multiplying tau — a real cloud replaced by a thin veil over the WHOLE sky — and
    // that is what made a 35%-cover Cloudy day grey: measured, its zenith went from a deep blue
    // (0.134, 0.310, 1.000) to a near-neutral (0.838, 0.774, 1.000) at 2.2x the luma, while the
    // horizon dropped 3.2x, flattening a 7.2x gradient to 1.02x. Cover is now a BLEND WEIGHT and
    // this is the cloud's own depth, so anything printing `tau` must print `cover` beside it.
    // [[feedback_a_printed_identity_outlives_its_term]]
    struct DeckReport {
        float tau;        // vertical optical depth OF THE CLOUD ITSELF (cover-independent)
        float cover;      // ...and the fraction of sky it covers, which is now a mix weight
        float ext;        // beta_ext, 1/m
        float ssa;        // single-scattering albedo
        float g;          // HG asymmetry
        float baseM;      // the deck's base and top, metres above the ground
        float topM;
    };
    inline DeckReport deckReport(const Params& p, float deckMul) {
        DeckReport d = {};
        const float cover  = std::max(0.0f, std::min(1.0f, p.cloudCoverage));
        const float shape  = std::max(0.0f, std::min(1.0f, p.cloudType));
        const float precip = std::max(0.0f, std::min(1.0f, p.precipitation));
        const float thickM = std::max(0.0f, p.cloudThicknessKm) * 1000.0f;
        const float meanD  = kCloudProfileMeanFlat
                           + shape * (kCloudProfileMeanBell - kCloudProfileMeanFlat);
        d.cover = cover;
        d.tau   = (kCloudTauRef + kCloudTauPrecip * precip) * std::max(0.0f, deckMul);
        // ⚠ THE cover > 0 GATE IS LOAD-BEARING AND IS NOT AN OPTIMISATION. Cover no longer thins
        // tau, so without it a clear row would publish a full-strength deck weighted zero — and
        // atmosDeckSpan()/atmosMarchPlan() gate on beta_ext, so the clear sky's QUADRATURE would
        // silently gain the deck's extra segment and stop being bit-identical to the pre-S4a march.
        // The A/B control arm has to reduce node for node, not merely to the same answer.
        d.ext   = (cover > 0.0f && thickM > 1.0f && d.tau > 0.0f && meanD > 1.0e-3f)
                ? (d.tau / (meanD * thickM)) : 0.0f;
        d.ssa   = std::max(0.0f, std::min(1.0f, kCloudSsa - kCloudSsaPrecip * precip));
        d.g     = kCloudGStratus + shape * (kCloudGDeep - kCloudGStratus);
        d.baseM = std::max(0.0f, p.cloudBottomKm) * 1000.0f;
        d.topM  = d.baseM + thickM;
        return d;
    }

    // ─── S4a/A2: THE DECK'S OWN TRANSPORT, AS A BOUNDED SLAB ─────────────────────────────────────
    //
    // ⚠⚠ WHAT THIS REPLACES AND WHY IT IS NOT A RETUNE. C1-C3 fed the deck's diffuse source from the
    // multiple-scattering LUT, and that source is `sca * ms * sInt` = `ssa * ms * (1 - stepT)`.
    // At a cloud's ssa of 0.9999 the deck's OWN OPTICAL DEPTH CANCELS OUT OF ITS OWN SOURCE, so the
    // term saturates at `ms` and stops depending on tau — measured across a 16x tau sweep, `f` was
    // identical to five decimals and E_sky floored at 55% of the incoming flux however thick the lid
    // got. At tau 3 the ground received 4.6x the energy arriving at the top of the atmosphere.
    // That is not a mistune, it is a MISSING CONSERVATION LAW: `F = 1/(1-f)` has nothing bounding it,
    // and `ms` was read off an altitude axis of 3.2 km per texel — wider than the whole deck — so
    // the in-cloud lookup interpolated between a ground sample and one above the lid, NEITHER OF
    // THEM INSIDE THE CLOUD. atmos_multiscatter.comp.fsl names this as the first thing to suspect.
    //
    // The replacement is the fallback that file reserved and refused to slip in: a delta-Eddington
    // TWO-STREAM SLAB, whose defining property is that R + T <= 1 BY CONSTRUCTION. It is solved here
    // in double precision, once per frame, because it is a function of (tau, ssa, g, mu_sun, albedo)
    // and every one of those is uniform over the whole LUT chain. The shader gets a table.
    //
    // ─── DELTA SCALING, WHICH IS ALSO THE FIX TO THE BEAM ────────────────────────────────────────
    // A cloud's g is 0.84-0.88, so about three quarters of what it scatters goes into a forward peak
    // physically indistinguishable from unscattered light. Delta scaling truncates that peak and
    // moves it into the direct beam:
    //
    //     f = g^2      tau' = (1 - ssa*f) tau      ssa' = ssa(1-f)/(1-ssa*f)      g' = g/(1+g)
    //
    // At the Cloudy row (tau 1.74, g 0.84) that takes tau to 0.512 and the beam transmittance from
    // 0.176 to 0.527 — A 3x RECOVERY OF THE SUN, which is the whole of "there are no shadows". It is
    // physics and not a dial: the same scaling this repo already applies to the water medium
    // (waterfog.h.fsl) and the standard treatment in every atmospheric radiation model since 1976.
    //
    // ⚠ THE SCALING IS APPLIED HERE, AT THE ONE SITE WHERE A Params BECOMES COEFFICIENTS, so all
    // three marches see ONE consistent scaled medium. Scaling only the transmittance would leave the
    // sky-view march phase-shifting with the UNSCALED g against a beam attenuated with the scaled
    // tau, and delta-Eddington is only correct as a triple.
    constexpr int    kDeckMsTaps      = 8;        // table entries, deck TOP -> deck BASE
    constexpr double kDeckDiffusivity = 7.0 / 4.0;  // Eddington's gamma1 at ssa = 0

    struct DeckSolution {
        bool   active;      // false = no deck this frame; every lane below is then 0
        float  tau, tauP;   // vertical optical depth, geometric and delta-scaled
        float  ssaP, gP;    // the delta-scaled optics
        float  extP;        // delta-scaled beta_ext, 1/m — what the shaders integrate
        // ⚠ THE THREE BELOW ARE THE BLACK-BOUNDARY SLAB, not the one the table was built from.
        // R + T <= 1 is a statement about THE MEDIUM; with the ground's albedo under it a slab can
        // return more than arrived (down, bounce, back up) and the sum reads ~1.27 — true, and not
        // a conservation law. The bound is only a bound where it is stated. All / (mu0*F0).
        float  R;           // slab reflectance
        float  Tdif, Tdir;  // diffuse / direct transmittance
        float  cover;       // the MIX WEIGHT: 0 = this row's sky is clear, 1 = a full lid
        float  profMean;    // cloudProfileHost's numeric mean / the analytic one — must be 1.000
        float  ms[kDeckMsTaps];  // isotropic MULTIPLE-scattering radiance / F0, top -> base
    };

    // The slab, solved. `muSun` is the sun's zenith cosine; `groundAlbedo` closes the lower boundary
    // (a cloud base over bright ground IS brighter, and leaving it out is a real error, not a
    // simplification). Everything returned is dimensionless — the shader multiplies by the solar
    // irradiance and by the air's transmittance down to the deck's lid, so the deck reddens at a low
    // sun and goes dark below the horizon without this function knowing anything about either.
    inline DeckSolution deckSolve(const Params& p, float deckMul, float muSun, float groundAlbedo)
    {
        DeckSolution s = {};
        const DeckReport d = deckReport(p, deckMul);
        s.tau   = d.tau;
        s.cover = d.cover;
        if (!(d.ext > 0.0f) || !(d.tau > 0.0f)) { return s; }   // clear sky: every lane stays 0

        // --- delta scaling -------------------------------------------------------------------
        const double ssa = std::max(0.0, std::min(0.999999, (double)d.ssa));
        const double g   = std::max(0.0, std::min(0.95,     (double)d.g));
        const double fwd = g * g;
        const double den = std::max(1.0e-6, 1.0 - ssa * fwd);
        const double tau = den * (double)d.tau;
        const double ssaP = std::max(0.0, std::min(0.999999, ssa * (1.0 - fwd) / den));
        const double gP   = g / (1.0 + g);
        s.tauP = (float)tau;  s.ssaP = (float)ssaP;  s.gP = (float)gP;
        s.extP = (float)(den * (double)d.ext);

        // --- the Eddington two-stream, on the scaled medium ------------------------------------
        // t runs DOWNWARD from the deck's lid. u = diffuse up, v = diffuse down.
        //     du/dt = g1 u - g2 v - S g3 exp(-t/mu0)
        //     dv/dt = g2 u - g1 v + S g4 exp(-t/mu0)
        // g3 is the UPSCATTER fraction of the direct beam and g4 = 1 - g3 the downscatter; at ssa 1
        // g1 == g2 and the net flux is conserved exactly, which is the identity this whole exercise
        // is about. (Verified against a brute-force RK4 integration of the same ODEs to 6 digits at
        // every configuration in the weather table.)
        const double mu0 = std::max(0.02, std::min(1.0, (double)muSun));
        const double ag  = std::max(0.0,  std::min(1.0, (double)groundAlbedo));
        const double g1 = (7.0 - ssaP * (4.0 + 3.0 * gP)) * 0.25;
        double       g2 = -(1.0 - ssaP * (4.0 - 3.0 * gP)) * 0.25;
        const double g3 = (2.0 - 3.0 * gP * mu0) * 0.25;
        const double g4 = 1.0 - g3;
        // g2 crosses zero only at ssa ~ 0.38, far below any cloud; the clamp is an underflow floor
        // for the eigenvector below, not a physical statement.
        if (std::fabs(g2) < 1.0e-6) { g2 = (g2 < 0.0) ? -1.0e-6 : 1.0e-6; }
        const double lam = std::sqrt(std::max(1.0e-12, g1 * g1 - g2 * g2));
        const double Gam = (g1 - lam) / g2;
        // The particular solution resonates when the beam's slant rate meets the diffusion rate;
        // nudging one off the other is the standard treatment and moves nothing measurable.
        double k = 1.0 / mu0;
        if (std::fabs(k - lam) < 1.0e-4) { k = lam + 1.0e-4; }
        auto  ex = [](double a) { return std::exp(std::max(-60.0, std::min(60.0, a))); };
        const double S = ssaP, Fs = mu0;             // F0 == 1: everything here is per unit beam
        const double dd = k * k - lam * lam;
        const double U  = -S * (g3 * (g1 - k) + g2 * g4) / dd;
        const double V  = -S * (g4 * (g1 + k) + g2 * g3) / dd;
        // Boundaries: no diffuse light enters the lid; the lower one returns `alb` x what reaches it.
        // ⚠ ONLY C1/C2 DEPEND ON THE LOWER BOUNDARY, so it is solved TWICE and the two answers do
        // two different jobs. The `ag` pair drives the table, because a cloud base over albedo-0.3
        // ground really is brighter and leaving that out is an error rather than a simplification.
        // The `0` pair is the REPORT, because R + T <= 1 is a statement about THE MEDIUM: with a
        // reflecting floor under it a slab can legitimately return more than arrived (light goes
        // down, bounces, comes back up), so a conservation row measured through the ground bounce
        // would read 1.27 and mean nothing. The bound has to be stated where it is actually a bound.
        const double E = ex(lam * tau), Em = ex(-lam * tau), Ek = ex(-k * tau);
        double C1 = 0.0, C2 = 0.0, C1b = 0.0, C2b = 0.0;
        for (int pass = 0; pass < 2; ++pass) {
            const double alb = (pass == 0) ? ag : 0.0;
            const double a11 = Gam, a12 = 1.0, b1 = -V;
            const double a21 = E * (1.0 - alb * Gam), a22 = Em * (Gam - alb);
            const double b2  = Ek * (alb * (V + Fs) - U);
            const double det = a11 * a22 - a12 * a21;
            if (std::fabs(det) < 1.0e-12) { return s; }
            const double c1 = (b1 * a22 - a12 * b2) / det;
            const double c2 = (a11 * b2 - b1 * a21) / det;
            if (pass == 0) { C1 = c1; C2 = c2; } else { C1b = c1; C2b = c2; }
        }
        auto uC = [&](double c1, double c2, double t) {
            return c1 * ex(lam * t) + c2 * Gam * ex(-lam * t) + U * ex(-k * t);
        };
        auto vC = [&](double c1, double c2, double t) {
            return c1 * Gam * ex(lam * t) + c2 * ex(-lam * t) + V * ex(-k * t);
        };
        auto uF = [&](double t) { return uC(C1, C2, t); };
        auto vF = [&](double t) { return vC(C1, C2, t); };

        // --- FIRST-ORDER scattering alone, so it can be taken back out -------------------------
        // ⚠⚠ WITHOUT THIS SUBTRACTION THE DECK IS COUNTED TWICE. The sky-view march already carries
        // the cloud's single scattering with the real HG lobe and the real sun direction — that term
        // is angularly correct and stays. The slab's total diffuse field CONTAINS first-order light,
        // and at the Cloudy row first order is ~100% of it, so handing the march the total would
        // double the deck's whole contribution. So: the same beam source transported with the
        // pure-extinction diffusivity and NO diffuse-diffuse coupling is exactly first order, and
        // total - first = orders >= 2, which is what `ms` has always meant.
        double kk = k;
        if (std::fabs(kDeckDiffusivity - kk) < 1.0e-4) { kk = kDeckDiffusivity + 1.0e-4; }
        auto v1F = [&](double t) {
            return S * g4 * (ex(-kk * t) - ex(-kDeckDiffusivity * t)) / (kDeckDiffusivity - kk);
        };
        const double P1   = S * g3 / (kDeckDiffusivity + kk);
        // Written as a decay from the LOWER boundary rather than exp(+D*t) with a tiny coefficient:
        // at tau' ~ 28 the two factors are 1e21 and 1e-21 and float64 keeps only one of them.
        const double Ctop = ag * (v1F(tau) + Fs * ex(-kk * tau)) - P1 * ex(-kk * tau);
        auto u1F = [&](double t) { return Ctop * ex(-kDeckDiffusivity * (tau - t)) + P1 * ex(-kk * t); };

        // --- the boundary fluxes, for the gate's conservation row ------------------------------
        // The BLACK-boundary pair: the medium's own albedo and transmittance, which is the only
        // form in which R + T <= 1 is a bound rather than an observation. See the note above.
        s.R    = (float)(uC(C1b, C2b, 0.0) / Fs);
        s.Tdif = (float)(vC(C1b, C2b, tau) / Fs);
        s.Tdir = (float)ex(-tau / mu0);

        // --- the table, sampled on NORMALISED ALTITUDE ----------------------------------------
        // The shader marches in altitude and has no cheap way to reach optical depth, so the table's
        // axis is altitude and the mapping between the two is done here, where the profile's
        // cumulative integral is affordable. 8 taps carry the emergent radiance at the deck base —
        // the only part of this field a camera under the lid can see — to within 3.2% over the whole
        // sweep of sun elevations, deck shapes and optical depths the weather table can produce.
        const double shape = std::max(0.0f, std::min(1.0f, p.cloudType));
        const int    nSub  = 512;
        double cum[nSub + 1];
        cum[0] = 0.0;
        for (int i = 0; i < nSub; ++i) {
            // x = +1 at the lid, -1 at the base, matching atmosCloudProfile's coordinate.
            const double x = 1.0 - 2.0 * ((double)i + 0.5) / (double)nSub;
            cum[i + 1] = cum[i] + cloudProfileHost(x, shape);
        }
        const double total = std::max(1.0e-9, cum[nSub]);
        const double meanA = (double)kCloudProfileMeanFlat
                           + shape * ((double)kCloudProfileMeanBell - (double)kCloudProfileMeanFlat);
        s.profMean = (float)((total / (double)nSub) / std::max(1.0e-9, meanA));
        const double inv2pi = 1.0 / (2.0 * 3.14159265358979323846);
        for (int i = 0; i < kDeckMsTaps; ++i) {
            const double xn = (double)i / (double)(kDeckMsTaps - 1);   // 0 = lid, 1 = base
            const double t  = tau * cum[(int)(xn * (double)nSub + 0.5)] / total;
            // Two hemispheres of isotropic radiance: F = pi L each, so the sphere mean is F/(2 pi).
            const double all = (uF(t) + vF(t)) * inv2pi;
            const double one = (u1F(t) + v1F(t)) * inv2pi;
            s.ms[i] = (float)std::max(0.0, all - one);
        }
        s.active = true;
        return s;
    }

    // ⚠⚠ THE mieTint COUPLING, STATED WHERE IT IS CONSUMED — AND IT IS NOT WHAT THE FIELD'S OWN
    // COMMENT SAYS. Params::mieTint is documented as "a per-primary multiplier on aerosol
    // EXTINCTION", and applying it that way is BACKWARDS for the two rows that use it: ash ships
    // tint (1.00, 0.80, 0.55), so scaling extinction would make blue the LEAST extinguished channel
    // — a bluer sun through an ash storm, and a sky that de-reddens as the storm thickens. The row's
    // own prose asks for the opposite ("removes blue far harder than red ... a dim brown-orange sky
    // that darkens the ground rather than glowing").
    //
    // What produces that, and what mineral dust actually is, is a nearly GREY extinction (large
    // irregular grains have an Angstrom exponent near zero) with a wavelength-dependent SINGLE-
    // SCATTERING ALBEDO — dust scatters ~0.9 of the red it intercepts and ~0.5 of the blue, absorbing
    // the rest. So the tint scales SCATTERING and leaves extinction grey:
    //
    //     ext[c] = base / (1 - mieAbsorption)          (grey: the aerosol's optical depth)
    //     sca[c] = ext[c] * (1 - mieAbsorption) * tint[c]
    //     abs[c] = ext[c] - sca[c]                     (implicit; blue-heavy wherever tint < 1)
    //
    // Both readings are identical on the eight rows whose tint is (1,1,1), so this decides ash and
    // blight alone — which is exactly where the plan says the lane earns its keep.
    inline void packParams(const Params& p,
                           float groundAlbedo,
                           const float toSun[3],
                           float cameraAltitudeM,
                           const float toMoon[3],
                           float moonIrradiance,
                           float airglow,
                           float mieScaleMul,
                           int   miePhaseModel,
                           float ozoneMul,
                           float deckMul,
                           float msMul,
                           float deckDownMul,
                           float skyViewSteps,
                           float msSteps,
                           float msDirs,
                           float deckSteps,
                           const float lutDims[6],
                           float* dst)
    {
        const float* Etoa = solarIrradianceTOA();

        // ⚠ mieScaleMul IS A CALIBRATION OF kMieScatterSeaLevel, NOT A SECOND LOOK KNOB, and the
        // distinction is why it multiplies HERE rather than being a per-row field. `g_skyTurbidity`
        // was retired so there would not be a second, disagreeing way to say what the table already
        // says; a UNIFORM multiplier on the sea-level coefficient says something the table cannot
        // say at all — that the aerosol column's UNITS are wrong for every weather at once.
        //
        // ⚠⚠ ITS ANSWER HAS BEEN FOLDED (2026-09-09) AND IT IS BACK TO BEING AN A/B ARM. The three
        // readings it was cut for — q_zenith over the p75, the horizon at 8.8x the zenith against a
        // real 2-4x, and a horizon far too BLUE — were ONE symptom and it was not the aerosol: the
        // multiscatter LUT was manufacturing the diffuse field (a 4pi in atmos_multiscatter, and a
        // probe placed exactly ON the ground sphere in atmosMultiScatterParams). With both fixed the
        // aerosol lever became measurable for the first time, the sweep ran, and x40 went into
        // kMieScatterSeaLevel where the note above derives it from Koschmieder visibility.
        //
        // ⚠ THAT ORDER WAS NOT OPTIONAL. Swept BEFORE the fixes, more aerosol made every row WORSE —
        // at x32 the sky doubled to E_sky 53 klx and the ENERGY row went to 134.8% — because the
        // manufactured term scaled with the scattering coefficient it was multiplying. A calibration
        // taken on top of a source that creates energy measures the bug, not the medium.
        //
        // ⚠⚠ AND THE FOLD WAS REVERTED THE SAME DAY, BY THE PICTURE. Read the long note at
        // kMieScatterSeaLevel: the gate's single 41 deg sun cannot see the solar aureole, and at a
        // 78 deg sun x40 turned the zenith white. So this is back to being the SWEEP HANDLE, and the
        // value that survives is still not known — it is blocked on the aerosol PHASE FUNCTION, not
        // on more sweeping. [[project_forge_no_ini_flips]]
        // ⚠ RESOLVED 2026-09-24: the S2l phase unblocked it and x30 is folded into kMieScatterSeaLevel
        // (see there). This is the A/B arm again: 1.0 = what ships (x30), 1/30 = Bruneton.
        const float mieBase = kMieScatterSeaLevel * std::max(0.0f, p.mieScale)
                            * std::max(0.0f, mieScaleMul);
        const float ssa     = 1.0f - std::max(0.0f, std::min(0.999f, p.mieAbsorption));
        const float mieExt  = (ssa > 1.0e-4f) ? (mieBase / ssa) : mieBase;

        // ─── THE CLOUD DECK, WHERE FIVE CARRIED FIELDS FINALLY BECOME COEFFICIENTS (S4a) ─────────
        // ⚠ THIS IS THE ONLY SITE, exactly as mieScaleMul is. The table says what the weather IS;
        // this says what that means optically; the shader integrates it. Three steps, three places,
        // and none of them a branch on a weather index.
        //
        // ⚠ deckMul IS AN A/B GATE, NOT A LOOK SLIDER — the same treatment mieScaleMul documents for
        // itself. 0 disarms the deck completely, which is byte-for-byte the pre-S4a sky and is the
        // control arm for every measurement in this phase; 1 is the shipped medium. It gets deleted
        // when the phase is accepted. [[project_forge_no_ini_flips]]
        // ⚠⚠ AND S4a/A2 MOVED THE ARITHMETIC OUT OF HERE INTO deckSolve(), WHICH IS THE SAME
        // DISCIPLINE ONE STEP FURTHER ON. The deck's tau/ssa/g were derived here AND in deckReport()
        // — two copies, kept honest by nothing — and A2 needs a third reading of them (the slab) plus
        // the delta scaling that all three marches have to share. So there is now exactly one
        // derivation: deckReport() says what the weather IS optically, deckSolve() delta-scales it
        // and solves its transport, and this writes the answer down. The rows below carry the
        // SCALED coefficients, which is what makes the sky-view march's phase function, the
        // transmittance LUT's beam and the slab's source three views of one medium instead of three.
        const float shape  = std::max(0.0f, std::min(1.0f, p.cloudType));
        const float baseM  = std::max(0.0f, p.cloudBottomKm)    * 1000.0f;
        const float thickM = std::max(0.0f, p.cloudThicknessKm) * 1000.0f;
        const DeckSolution deck = deckSolve(p, deckMul, toSun[2], groundAlbedo);
        const float cloudExt = deck.extP;                // delta-scaled beta_ext (1/m)
        const float cloudSsa = deck.ssaP;                // ...and its delta-scaled albedo
        const float cloudG   = deck.gP;                  // ...and its delta-scaled asymmetry

        int i = 0;
        // row 0: Rayleigh scattering (1/m) + its scale height (m)
        for (int c = 0; c < 3; ++c) { dst[i++] = kRayleighSeaLevel[c] * std::max(0.0f, p.rayleighScale); }
        dst[i++] = kRayleighHeightKm * 1000.0f;
        // row 1: Mie SCATTERING (1/m, tinted) + its scale height (m)
        for (int c = 0; c < 3; ++c) { dst[i++] = mieExt * ssa * std::max(0.0f, p.mieTint[c]); }
        dst[i++] = std::max(1.0f, p.mieHeightKm * 1000.0f);
        // row 2: Mie EXTINCTION (1/m, grey) + the HG asymmetry
        for (int c = 0; c < 3; ++c) { dst[i++] = mieExt; }
        dst[i++] = std::max(-0.95f, std::min(0.95f, p.mieG));
        // row 3: ozone absorption at the tent's peak (1/m)
        // ⚠ ozoneMul IS A UNIFORM MULTIPLIER ON THE COLUMN, AND IT IS THE ONLY LEVER IN THIS FILE
        // THAT MAKES A SKY LESS GREEN. Bruneton's per-primary absorption is (0.650, 1.881, 0.085) —
        // green 2.9x red and 22x blue — so ozone is what turns a CYAN Rayleigh sky into a blue-violet
        // one, and it is the reason twilight is violet rather than brown. Hosek had no ozone term at
        // all. Every row's `ozoneScale` is 1.00 on purpose (the layer sits at 25 km; no weather
        // Morrowind has reaches it), so a global multiplier is the right shape for this and a
        // per-weather column would be fiction.
        //
        // ⚠⚠ IT IS A LOOK KNOB WEARING A PHYSICAL NAME, AND THE UNIT SAYS SO. x1 is ~300 Dobson,
        // Earth's mean; Earth's whole range is 200-500, i.e. x0.7-x1.7. Measured against a target
        // sky colour supplied from play, the hue lands at **x3.2 — about 960 DU**, three times any
        // atmosphere Earth has. So a value above ~1.7 is a STYLISTIC choice about Vvardenfell's air
        // and must be labelled as one; it is not a calibration and no gate will ever endorse it.
        for (int c = 0; c < 3; ++c) {
            dst[i++] = kOzoneAbsorb[c] * std::max(0.0f, p.ozoneScale) * std::max(0.0f, ozoneMul);
        }
        // .w: the aerosol spike's HG g (S2l) — riding a spare lane, not an ozone property.
        dst[i++] = kMieSpikeG;
        // row 4: the geometry of the two BOUNDED profiles, metres above the ground — the ozone tent
        // and the cloud deck. ⚠ A DECK WITH NO EXTINCTION STILL PUBLISHES ITS GEOMETRY, and that is
        // deliberate: atmosDeckSpan() gates on beta_ext, so a clear sky costs one compare rather than
        // a shell intersection, and the geometry stays readable in a capture.
        dst[i++] = kOzoneCentreKm * 1000.0f;
        dst[i++] = kOzoneWidthKm  * 1000.0f;
        dst[i++] = baseM + 0.5f * thickM;    // deck centre
        dst[i++] = 0.5f * thickM;            // deck half-thickness
        // row 5: planet geometry + the ground the medium sits on + where the camera is in it
        dst[i++] = kGroundRadiusM;
        dst[i++] = kAtmoRadiusM;
        dst[i++] = std::max(0.0f, std::min(1.0f, groundAlbedo));
        dst[i++] = kGroundRadiusM + std::max(0.0f, std::min(kAtmoRadiusM - kGroundRadiusM - 1.0f,
                                                            cameraAltitudeM));
        // row 6: the direction TOWARD the sun (world, MW Z up) + its zenith cosine
        dst[i++] = toSun[0]; dst[i++] = toSun[1]; dst[i++] = toSun[2];
        dst[i++] = std::max(-1.0f, std::min(1.0f, toSun[2]));
        // row 7: E_TOA per primary
        dst[i++] = Etoa[0]; dst[i++] = Etoa[1]; dst[i++] = Etoa[2];
        dst[i++] = 0.0f;
        // row 8: the MOON as a second source (S2e) — direction + its own top-of-atmosphere
        // irradiance in the SAME native units, so it enters the identical integral. 0 disarms it.
        dst[i++] = toMoon[0]; dst[i++] = toMoon[1]; dst[i++] = toMoon[2];
        dst[i++] = std::max(0.0f, moonIrradiance);
        // row 9: the airglow floor (native radiance) — the night sky's own emission, which is what
        // stops "no sun" from meaning "no photons" once the sun is genuinely below the horizon.
        dst[i++] = std::max(0.0f, airglow);
        // .yzw: THE AEROSOL PHASE (S2l) — body g, spike weight, model. Model 0 is the single-HG A/B
        // arm on row 2's .w and the shader evaluates exactly the pre-S2l expression there; the two
        // lanes before it are then unread. Clamped like mieG, for the same reason: |g| -> 1 is a
        // delta function the march cannot integrate.
        dst[i++] = std::max(-0.95f, std::min(0.95f, p.mieBodyG));
        dst[i++] = std::max(0.0f, std::min(1.0f, p.mieSpike));
        dst[i++] = (miePhaseModel != 0) ? 1.0f : 0.0f;
        // row 10: march budgets. .w is the DECK's own step budget — the quadrature fix S4a needed,
        // because all three marches are tuned for smooth exponentials and a 1.2 km lid is not one.
        // ⚠ 0 HERE DISARMS THE DECK IN EVERY MARCH, whatever the coefficients say, so it is a second
        // and completely independent way to reach the pre-S4a sky. Kept independent on purpose: if a
        // deck frame looks wrong, "is it the optics or the quadrature" is the first bisection.
        dst[i++] = std::max(4.0f, skyViewSteps);
        dst[i++] = std::max(4.0f, msSteps);
        dst[i++] = std::max(2.0f, msDirs);
        dst[i++] = (cloudExt > 0.0f) ? std::max(0.0f, std::min(64.0f, deckSteps)) : 0.0f;
        // rows 11-12: LUT dimensions (sky-view, transmittance, multiscatter)
        dst[i++] = lutDims[0]; dst[i++] = lutDims[1];
        dst[i++] = lutDims[2]; dst[i++] = lutDims[3];
        dst[i++] = lutDims[4]; dst[i++] = lutDims[5];
        dst[i++] = 0.0f; dst[i++] = 0.0f;
        // row 13: the deck's optics. GREY — droplets are hundreds of wavelengths across, so cloud
        // extinction is spectrally flat and every colour an overcast sky has comes from the air
        // above and below it, which the three species above already carry.
        dst[i++] = cloudExt * cloudSsa;      // beta_sca (1/m)
        dst[i++] = cloudExt;                 // beta_ext (1/m)
        dst[i++] = std::max(-0.95f, std::min(0.95f, cloudG));
        dst[i++] = shape;                    // the profile's shape blend
        // rows 14-15: THE DECK'S OWN MULTIPLE SCATTERING (S4a/A2) — the two-stream slab's isotropic
        // orders>=2 radiance per unit incident beam, sampled on normalised altitude from the lid
        // (tap 0) to the base (tap 7). ⚠ THIS IS WHAT THE MULTISCATTER LUT NO LONGER SUPPLIES for
        // the cloud species: a 3.2 km/texel altitude axis could not see a 1.2 km deck, and the
        // source it fed back was unbounded in tau. All zero on a clear sky, so the pre-S4a A/B is
        // untouched. See deckSolve().
        for (int t = 0; t < kDeckMsTaps; ++t) { dst[i++] = deck.ms[t]; }
        // row 16: THE COVER MIXTURE (2026-09-09). x is the fraction of sky the deck covers, and it
        // is a BLEND WEIGHT rather than a thinner: the sky-view march runs a clear arm and a
        // full-depth deck arm and lerps them by this. Everything above describes ONE FULL CLOUD; how
        // much of the sky has one is this number and nothing else.
        dst[i++] = deck.cover;
        // y: THE AIR'S MULTIPLE-SCATTERING A/B GATE. 1 is the shipped medium; 0 leaves the sky-view
        // march with its FIRST ORDER only, which a CPU quadrature can reproduce independently. It is
        // a measurement arm and not a look knob, exactly as deckMul is, and it exists because the
        // clear sky reads 3.3x (red) to 8.2x (blue) over its own single-scattering value and the
        // gate had no way to say which of the loop's two additive sources carries that.
        dst[i++] = std::max(0.0f, msMul);
        // ─── z: S2i — THE DOME'S RE-MIX WEIGHT, AND IT IS A DIFFERENT QUESTION FROM `cover` ──────
        //
        // `cover` above is the weight for the LIGHT, where it is exact: the SH pass integrates the
        // sky over directions, and the expected radiance of a direction that is cloud with
        // probability `cover` IS the lerp. The IMAGE samples ONE direction, where the same number is
        // simply wrong — a direction is cloud or gap, never `cover` of both. Shipping the light's
        // mean field to the dome put a uniform 35% grey veil over a cumulus day, gaps included: the
        // gaps measured R/B 0.73 against a clear sky's 0.24, which is the whole of "cloudy is washed
        // out, clear and cloudy should look pretty much the same".
        //
        // ⚠ THE PER-DIRECTION STRUCTURE IS ALREADY DRAWN, WHICH IS WHY THIS IS A DOUBLE COUNT AND
        // NOT A MISSING FEATURE. MW's cloud MESH paints the individual clouds and is already
        // anchored to this deck's radiance (sky.frag.fsl, SKY_CLASS_CLOUD). For a BROKEN field the
        // deck therefore reaches the camera through the mesh, and the dome's job is the sky BETWEEN
        // the clouds. For a CLOSED lid there are no gaps, the mesh does not cover the sky on its own
        // (measured: Overcast with the deck off renders BLUE), and the dome must carry the deck.
        //
        // So the regime, not the coverage, is what the dome needs — and the table splits cleanly:
        //
        //     cover 0.35 Cloudy · 0.40 Foggy · 0.50 Ash · 0.60 Blight     gaps exist
        //     cover 1.00 Overcast · Rain · Thunder · Snow · Blizzard      closed lid
        //
        // Nothing is authored between 0.60 and 1.00, so the smoothstep below is never evaluated on
        // the inside by a settled weather — it exists so a TRANSITION between the two regimes
        // crossfades instead of popping.
        //
        // ⚠ EXPRESSED AS A FRACTION OF THE LIGHT'S MIX, not as a second cover. The dome lerps
        // between the two published FIELDS (gap and mixed), so to land on a true weight `w` it needs
        // w/cover — and that keeps the cover=1 path byte-identical to the old one-LUT dome, which is
        // the regression this change has to protect. cover=0 gives 0, and there both fields are the
        // same expression anyway, so clear weather is unmoved twice over.
        {
            const float c = std::max(0.0f, std::min(1.0f, deck.cover));
            const float t = std::max(0.0f, std::min(1.0f, (c - 0.60f) / 0.40f));
            const float w = t * t * (3.0f - 2.0f * t);          // smoothstep(0.60, 1.00, cover)
            dst[i++] = (c > 1.0e-4f) ? std::min(1.0f, w / c) : 0.0f;
        }
        // ─── w: S2h — THE A/B ON THE DECK'S LIGHT REACHING THE AIR BENEATH IT ───────────────────
        // 1 ships the term; 0 reproduces the pre-S2h march term for term, which is what makes the
        // orange horizon a measurable claim rather than an argued one. Same kind of arm as
        // `atmosDeck` and `atmosMs`, and it exists for the same reason they do.
        dst[i++] = std::max(0.0f, deckDownMul);
    }

    // The REFERENCE configuration the S2 gate is measured at, as a Params. Row 0 verbatim — the
    // table's own note says row 0 is not a look and must not be tuned, and this is why.
    inline Params referenceRow() { return kWeatherTable[0]; }

    // ─── THE S4a GATE's SECOND CONFIGURATION: THE LID ────────────────────────────────────────────
    // ⚠⚠ ROW 0's AIR WITH A FULL DECK OVER IT, AND NOT THE OVERCAST ROW. That choice is the whole
    // reason the second gate frame is worth running. Two things follow from it:
    //
    //   1. IT ISOLATES THE DECK. The only difference from referenceRow() is the lid — same aerosol,
    //      same ozone, same everything — so every number the gate reports is a statement about the
    //      cloud model rather than about one table entry's aerosol. Measuring against the Overcast
    //      row instead would have folded its stand-in Mie into the answer, which is exactly the
    //      double-count C5 exists to remove.
    //   2. IT SURVIVES C5. The Overcast row's aerosol is scheduled to be rolled back this phase; a
    //      gate anchored to it would move when that lands and the before/after would be unreadable.
    //
    // Geometry from the Overcast row (base 1.0 km, 1.2 km thick, stratus) because that is what an
    // overcast lid is; cover forced to 1.00 because the gate's bands are stated at FULL cover.
    inline Params overcastGateRow() {
        Params p = kWeatherTable[0];
        p.cloudCoverage    = 1.00f;
        p.cloudType        = 0.00f;   // stratus: a flat slab, not a convective tower
        p.cloudBottomKm    = 1.00f;
        p.cloudThicknessKm = 1.20f;
        p.precipitation    = 0.00f;   // an overcast day, not a storm
        return p;
    }

}   // namespace Atmosphere
