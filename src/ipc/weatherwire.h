#pragma once

#include <cstdint>

// ─── S1 ATMOSPHERE — MW'S WEATHER, ON THE WIRE AT LAST (tasks/forge-atmosphere.md) ───────────────
//
// ⚠ THE SKY HAS NEVER KNOWN WHAT THE WEATHER IS. The physical sky (P2) generates a CLEAR sky from a
// fixed turbidity slider, and this struct is the whole reason it could not do anything else:
// nothing about MW's weather reached the host at all. The one weather read on the client
// (renderprocess.cpp, MWBridge::getWeatherState) took two particle counts for the rain ripples and
// threw the rest away — so "overcast sky is gray from the clouds texture, horizon has hosek's bluer
// sky" is not a shader defect, it is a missing wire.
//
// ⚠ THESE ARE PARAMETERS OF A MEDIUM, NOT A LOOK. Everything here is either an index into the
// host's per-weather physics table (cur/next/transition, mgeHost64/atmosphere.h) or one of MW's own
// authored scalars that the medium consumes directly (cloud cover, fog depth, wind). Nothing here
// is a colour a shader may draw with — see the two that ARE colours, below.
//
// ⚠ skyColRef / fogColRef ARE CALIBRATION REFERENCES AND MUST NEVER DRIVE ANYTHING. They are MW's
// authored display codes, already blended for the hour and the transition, and the whole P2 finding
// was that a display code used as radiance cannot make sky and land agree
// ([[project_forge_sky_is_a_display_code]]). They ride so the host can REPORT how far the generated
// atmosphere has moved from MW's intent — the same job `mwRef amb=/sun=` does on the
// [forge-hb][sky] line — and for no other purpose.
//
// This lives in its OWN header, not bridge.h, so the x64 host can include it without dragging in
// bridge.h's d3d9header.h. Same reasoning as ipc/hostframetimings.h and ipc/geomwire.h.
//
// LAYOUT IS THE CONTRACT: shared by layout between an x86 client and an x64 host, where a mismatch
// is silent corruption rather than a link error. Hence int32/float only — no double, size_t, bool
// or pointers, all of which differ or pad differently across the two. Append new fields at the END,
// and always rebuild + deploy BOTH binaries together.
namespace IPC {

    struct WeatherWire {
        // Read FIRST. 0 ⇒ no live weather — an interior, a menu, or before the world exists — and
        // every other lane is zeroed. This is the idiom the sun, the sky-AO and the sky-ambient
        // lanes already use: the gate is CLIENT-side, so the host needs no exterior test of its own
        // and a stale exterior row can never rain indoors.
        std::int32_t valid;
        std::int32_t cur;              // TES3::WeatherType 0..9 (Clear..Blizzard), -1 if absent
        std::int32_t next;             // == cur when no transition is running
        float transition;              // 0 = cur, 1 = next. MW's transitionScalar; lerp(cur,next,t).

        // MW's own authored scalars for the CURRENT atmosphere, already interpolated cur->next by
        // `transition` on the client (see renderprocess.cpp). They are interpolated THERE rather
        // than here because MWBridge reads them off the two Weather objects and only the client
        // holds both; shipping just the current one would step at the swap, which is precisely the
        // "texture transition" this work exists to stop being.
        // ⚠ NOT A COVERAGE FRACTION, DESPITE THE NAME. Measured 2026-08-30 across all ten
        // weathers in this install: it is 1.000 for eight of them and 0.660 for Rain and
        // Thunderstorm — i.e. it does not discriminate Clear from Blizzard and cannot be a cover.
        // OpenMW settles what it actually is: `cloudBlendFactor = transitionRatio /
        // cloudsMaximumPercent`, the rate at which MW cross-fades one cloud TEXTURE into the next.
        // So S4 must take its cover from the per-weather physics table and never from this lane.
        // This is precisely the invented constant the "carry it, do not consume it" rule existed to
        // prevent, and the measurement caught it before a single line consumed it.
        float cloudsMaxPercent;        // Weather::cloudsMaxPercent — a cross-fade RATE, not a cover
        float cloudsSpeed;             // cloud layer scroll speed
        float windSpeed;               // Weather::windSpeed. ⚠ NOT the live wind VECTOR, which is
                                       // already on lighting[36..37]: this is the weather's own
                                       // AUTHORED strength, and S6 advects the parameter field by it
        float landFogDay;              // Weather::landFogDayDepth
        float landFogNight;            // Weather::landFogNightDepth

        // Live engine state, not authored: these move WITHIN a single weather.
        float thunderFlash;            // activeThunderFlashIntensity, 0 when not flashing
        float sunglareVis;             // smoothedSunglareVis 0..1
        std::int32_t sunOccluded;      // 1 = MW considers the sun occluded this frame

        // ⚠ REFERENCES ONLY. See the warning above. Linear-ish display codes as MW authored them.
        float skyColRef[3];
        float fogColRef[3];
    };

    // 4 header lanes + 5 authored scalars + 3 live + 6 colour = 18. Update the count when appending
    // a field: the point is that a stray double/pointer (or padding from one) can never slip in
    // unnoticed, because this struct is interpreted by two differently-sized processes.
    static_assert(sizeof(WeatherWire) == 18 * sizeof(float),
                  "WeatherWire must stay tightly packed 4-byte lanes - it crosses the x86/x64 wire");

}
