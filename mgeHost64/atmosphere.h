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
    // Aerosol at 550 nm: extinction 4.44e-6 /m with single-scattering albedo 0.9, i.e. scattering
    // 3.996e-6 and absorption 0.444e-6. The 0.9 albedo is where kMieAbsorbFractionClear comes from
    // and it is the number every row's `mieAbsorption` is a departure from.
    constexpr float kMieScatterSeaLevel  = 3.996e-6f;   // 1/m
    constexpr float kMieHeightKm         = 1.2f;
    constexpr float kMieAbsorbFractionClear = 0.10f;    // 1 - single-scattering albedo
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
    constexpr float kCloudCoverExponent = 2.5f;     // cover^k — the MEAN-FIELD exponent
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
        float cloudCoverage;      // 0..1
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
        { /*ray*/ 1.00f, /*mie*/ 0.60f, /*abs*/ 0.10f, /*g*/ 0.80f, /*hkm*/ 1.20f,
          /*tint*/ { 1.00f, 1.00f, 1.00f }, /*ozone*/ 1.00f,
          /*cover*/ 0.00f, /*type*/ 0.30f, /*bot*/ 2.00f, /*thick*/ 0.40f, /*precip*/ 0.00f,
          0,0,0,0,0 },
        // Cloudy — fair-weather cumulus over otherwise clean air. Slightly more aerosol than clear
        // because a sky with cumulus in it is a sky with moisture in it.
        { 1.00f, 1.00f, 0.10f, 0.80f, 1.20f, { 1.00f, 1.00f, 1.00f }, 1.00f,
          0.35f, 0.50f, 1.80f, 0.80f, 0.00f, 0,0,0,0,0 },
        // Foggy — NOT "more haze". Fog is a thick, near-white, ground-hugging aerosol: the defining
        // lane is mieHeightKm 0.25, which puts almost all of it below the player. Droplets are large
        // and nearly non-absorbing (albedo ~0.99), so fog is BRIGHT — it hides the sun without
        // darkening the world, which is exactly how real fog reads.
        { 1.00f, 6.00f, 0.02f, 0.85f, 0.25f, { 1.00f, 1.00f, 1.00f }, 1.00f,
          0.40f, 0.00f, 0.05f, 0.30f, 0.00f, 0,0,0,0,0 },
        // Overcast — THE LID, and the row that closes the reported "overcast sky is gray from the
        // clouds texture, horizon has hosek's bluer sky" before real clouds exist (S2). High Mie
        // with real absorption de-blues AND dims the whole atmosphere, so MW's painted cloud layer
        // stops sitting on a bright blue field.
        { 1.00f, 2.00f, 0.30f, 0.78f, 1.60f, { 1.00f, 1.00f, 1.00f }, 1.00f,
          0.95f, 0.00f, 1.00f, 1.20f, 0.00f, 0,0,0,0,0 },
        // Rain — full cover, a low wet base, and the aerosol below the deck that rain actually is.
        { 1.00f, 2.50f, 0.35f, 0.76f, 1.20f, { 1.00f, 1.00f, 1.00f }, 1.00f,
          1.00f, 0.35f, 0.70f, 2.50f, 0.60f, 0,0,0,0,0 },
        // Thunder — cumulonimbus: the same medium as rain with a deck six kilometres deep, which is
        // what makes a storm cloud dark underneath and bright on top. The underlit-at-sunset case
        // on the brief is this row plus S4's lighting, and no new code.
        { 1.00f, 3.00f, 0.40f, 0.74f, 1.20f, { 1.00f, 1.00f, 1.00f }, 1.00f,
          1.00f, 1.00f, 0.60f, 6.00f, 1.00f, 0,0,0,0,0 },
        // Ash — Vvardenfell's signature, and the clearest case for the tint lane. Mineral dust is a
        // heavy, strongly ABSORBING, ground-hugging aerosol that scatters broadly (large irregular
        // grains, so a weaker forward lobe than clean air) and removes blue far harder than red.
        // The result is a dim brown-orange sky that darkens the ground rather than glowing, which
        // is what separates an ash storm from fog of the same density.
        { 1.00f, 10.00f, 0.55f, 0.65f, 0.80f, { 1.00f, 0.80f, 0.55f }, 1.00f,
          0.50f, 0.20f, 1.20f, 1.00f, 0.00f, 0,0,0,0,0 },
        // Blight — ash, sicker: denser, more absorbing, and pushed further toward red so the sky
        // reads diseased rather than merely dusty. Same medium, different point in it.
        { 1.00f, 12.00f, 0.60f, 0.62f, 0.80f, { 1.00f, 0.62f, 0.45f }, 1.00f,
          0.60f, 0.20f, 1.20f, 1.20f, 0.00f, 0,0,0,0,0 },
        // Snow — a deck like overcast, but the medium below it is ice crystals: they scatter almost
        // without absorbing and much more isotropically than droplets, which is why falling snow is
        // luminous grey rather than dark.
        { 1.00f, 3.00f, 0.05f, 0.60f, 1.00f, { 1.00f, 1.00f, 1.00f }, 1.00f,
          0.90f, 0.00f, 0.90f, 1.20f, 0.50f, 0,0,0,0,0 },
        // Blizzard — snow's medium at storm density and pulled down to the ground. Non-absorbing
        // like snow, so a whiteout is BRIGHT; the ash rows are the dark counterpart and the pair is
        // the clearest demonstration that density and absorption are two different lanes.
        { 1.00f, 12.00f, 0.05f, 0.55f, 0.40f, { 1.00f, 1.00f, 1.00f }, 1.00f,
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
    constexpr int kGpuParamFloats = 14 * 4;     // 14 float4 rows = 224 B (the CBV is 512 B)

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
                           float deckMul,
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
        // It exists because three independent readings point at one number: q_zenith sits over the
        // measured p75 (0.158 vs 0.148), the horizon reads 8.8x the zenith against a real 2-4x, and
        // the horizon is far too BLUE (B/R 3.55 against a real ~1.2-1.5). Too bright and not white
        // enough at the horizon is one symptom, not two — aerosol is what whitens a horizon and what
        // bounds it, and at Bruneton's coefficients Mie is ~2% of the vertical optical depth here.
        //
        // ⚠⚠ IT IS NOT A SHIPPING KNOB. Whatever value survives belongs FOLDED INTO
        // kMieScatterSeaLevel (or into the rows, if the answer turns out to be per-weather), and
        // this parameter deleted. A calibration multiplier left in the build is how a constant stops
        // having one home. [[project_forge_no_ini_flips]]
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
        const float cover   = std::max(0.0f, std::min(1.0f, p.cloudCoverage));
        const float shape   = std::max(0.0f, std::min(1.0f, p.cloudType));
        const float precip  = std::max(0.0f, std::min(1.0f, p.precipitation));
        const float baseM   = std::max(0.0f, p.cloudBottomKm)    * 1000.0f;
        const float thickM  = std::max(0.0f, p.cloudThicknessKm) * 1000.0f;
        // cover^k, the mean-field exponent — see the long note at kCloudCoverExponent.
        const float coverK  = std::pow(cover, kCloudCoverExponent);
        const float tauDeck = (kCloudTauRef + kCloudTauPrecip * precip)
                            * coverK * std::max(0.0f, deckMul);
        // The profile's mean over its own layer, exact for the two shapes atmosCloudProfile blends.
        // Dividing by it is what makes tau the VERTICAL OPTICAL DEPTH rather than a coefficient
        // whose meaning drifts with the deck's thickness and type.
        const float meanD   = kCloudProfileMeanFlat
                            + shape * (kCloudProfileMeanBell - kCloudProfileMeanFlat);
        const float cloudExt = (thickM > 1.0f && tauDeck > 0.0f && meanD > 1.0e-3f)
                             ? (tauDeck / (meanD * thickM)) : 0.0f;
        const float cloudSsa = std::max(0.0f, std::min(1.0f, kCloudSsa - kCloudSsaPrecip * precip));
        const float cloudG   = kCloudGStratus + shape * (kCloudGDeep - kCloudGStratus);

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
        for (int c = 0; c < 3; ++c) { dst[i++] = kOzoneAbsorb[c] * std::max(0.0f, p.ozoneScale); }
        dst[i++] = 0.0f;
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
        dst[i++] = 0.0f; dst[i++] = 0.0f; dst[i++] = 0.0f;
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
    }

    // What the deck came out as, for the heartbeat and the gate. A report, not a second derivation:
    // it recomputes nothing the pack does not, and nothing reads it to shade with.
    struct DeckReport {
        float tau;        // vertical optical depth
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
        d.tau   = (kCloudTauRef + kCloudTauPrecip * precip)
                * std::pow(cover, kCloudCoverExponent) * std::max(0.0f, deckMul);
        d.ext   = (thickM > 1.0f && d.tau > 0.0f && meanD > 1.0e-3f) ? (d.tau / (meanD * thickM)) : 0.0f;
        d.ssa   = std::max(0.0f, std::min(1.0f, kCloudSsa - kCloudSsaPrecip * precip));
        d.g     = kCloudGStratus + shape * (kCloudGDeep - kCloudGStratus);
        d.baseM = std::max(0.0f, p.cloudBottomKm) * 1000.0f;
        d.topM  = d.baseM + thickM;
        return d;
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
