// mgeHost64 — the knob registry (tasks/forge-host-decomposition.md, Phase 1).
//
// ⚠ THIS FILE EXISTS TO UNBLOCK A FILE LAYOUT, NOT TO IMPROVE A DATA STRUCTURE. forgerender.cpp is
// 57k lines, and the reason is not that anyone wanted it that way: a dev knob could only be
// DECLARED in one place. The four `{name, pointer}` tables in that file are static arrays, so every
// knob global had to be a file-scope object visible at the line where the arrays are initialised —
// the file's own comment says so ("Verified before moving: all 255 knob globals are declared above
// this point"). A subsystem in its own .cpp therefore could not register a knob AT ALL without
// editing forgerender.cpp, and since every feature here is measured through a knob (an env token
// for the minimized harness, a panel row for the eye), every feature for a year was born in that
// file. terrain.cpp, atmosphere.cpp, upscale.cpp and vkrender.cpp stayed small partly because they
// own almost no knobs.
//
// Self-registration removes that constraint and nothing else. A knob declares itself next to the
// code that reads it, in whatever translation unit that is:
//
//     static float g_cloudDensity = 0.5f;
//     MGE_KNOB(g_cloudDensity, "cloudDensity");
//
// ⚠ ORDER IS NOT OBSERVABLE, and that is what makes static initialisers safe here. It is worth
// writing down because it looks like it should be, and the decomposition plan originally assumed it
// was. The tables have exactly three readers: applyKnobSpec (name -> pointer), knobNameOf (pointer
// -> name, for the panel's SAVE), and applyEnvOverrides (which calls the first). NONE of them
// enumerates for display. The dev panel's row order comes from initDevUI's explicit TabBuilder
// sequence, not from this table, so adding a TU cannot reshuffle the panel. The only two ways
// position could matter are a duplicate NAME (applyKnobSpec takes the first) and a duplicate
// POINTER (knobNameOf takes the first) — so `--knob-dump` reports both, and they are the thing to
// watch rather than ordering.
//
// ⚠ ABI: this TU is built on the host's DEFAULT MSVC settings, NOT The Forge's (_HAS_EXCEPTIONS=0
// plus IMemory.h's global new/delete override) — the same arrangement terrain.cpp documents. The
// registry's container is allocated and freed entirely inside knobs.cpp; every type that crosses
// this header is POD or a raw pointer. Callers must never free anything owned here.

#pragma once

#include <cstddef>
#include <cstdint>

namespace Knobs {

    // The four kinds, kept as a tag rather than as four containers. They exist because a float
    // cannot honestly carry a mode (0.9 is not a mode) and a bool cannot carry six values; that is
    // a reason for four INTERPRETATIONS, never for four tables.
    enum Kind : uint8_t { KindF = 0, KindB = 1, KindU = 2, KindS = 3 };

    struct Entry {
        const char* name;     // the stable MGE_HOST_KNOBS name; must outlive the process
        void*       p;        // float* / bool* / uint32_t* / char*, per `kind`
        uint32_t    umax;     // KindU: the inclusive clamp. Other kinds: 0
        uint32_t    cap;      // KindS: the buffer size for snprintf. Other kinds: 0
        Kind        kind;
    };

    // Registration. Returns true so the call can initialise a file-scope object; the value carries
    // no information and nothing should test it. A duplicate name is accepted and counted rather
    // than rejected — see dupNames() — because refusing at static-init time has nowhere to report.
    bool add(const char* name, float*    p);
    bool add(const char* name, bool*     p);
    bool add(const char* name, uint32_t* p, uint32_t umax);
    bool add(const char* name, char*     p, size_t cap);

    // name -> entry, or nullptr. Linear; called at startup and when the panel is built or saved.
    const Entry* find(const char* name);

    // pointer -> the stable name, or nullptr. The panel's SAVE direction. Linear, same reason.
    const char* nameOf(const void* p);

    // Enumeration, in REGISTRATION order. Callers that report to a human sort it themselves —
    // registration order is stable within one build but says nothing across builds, and a caller
    // that wants to be diffed must not depend on it.
    size_t       count();
    const Entry* at(size_t i);

    // How many registrations collided on a name already present. Non-zero means some knob cannot
    // be reached from a spec string, which is silent in every other way.
    int dupNames();

}  // namespace Knobs

// Declare a knob next to the code that owns it. `var` must be a file-scope object.
#define MGE_KNOB(var, name) static const bool k_knobreg_##var = ::Knobs::add(name, &var)
// The uint form needs its clamp, and the string form needs its buffer size.
#define MGE_KNOB_U(var, name, umax) static const bool k_knobreg_##var = ::Knobs::add(name, &var, umax)
#define MGE_KNOB_S(var, name) static const bool k_knobreg_##var = ::Knobs::add(name, var, sizeof(var))
