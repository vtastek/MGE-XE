#pragma once

#include <cstdint>

// Per-stage draw-call counter.
//
// Goal: attribute every GPU draw MGE issues this frame to the rendering stage
// that issued it, so the per-stage totals can be summed and compared against
// the single draw-call number DXVK reports in its HUD. Both Morrowind's scene
// draws (forwarded through ProxyDevice) and MGE's own passes (raw device->Draw*
// and the VisibleSet::Render leaf in quadtree.h) land on the same real D3D9
// device, so we count at every leaf draw site and tag the *current* stage with
// a ScopedStage guard at each high-level pass entry.
//
// Attribution rule: a draw counts under the innermost stage active when it is
// issued. Stages can nest (e.g. distant land drawn during a water reflection),
// in which case the inner tag wins — but every draw is counted exactly once, so
// the SUM across stages is the exact frame draw-call count regardless of how the
// buckets split. That total is the apples-to-apples number versus DXVK.
//
// Thread note: g_stage is thread_local so the render thread's depth pass and the
// main thread's scene draws never corrupt each other's current stage. The shared
// counter arrays are plain (unsynchronised) ints — a lost increment under a race
// is acceptable for a diagnostic, and avoids atomics on the hot draw path.
namespace DrawStats {

    enum Stage : int {
        Scene0, Scene1, Scene2, UI,        // Morrowind scenes (set in BeginScene)
        Depth, Shadow, Reflection,          // pre-passes
        Land, Statics, Grass, Water, Sky,   // distant-land content
        Post, Debug, Other,
        COUNT
    };

    inline const char* name(Stage s) {
        static const char* const n[COUNT] = {
            "scene0", "scene1", "scene2", "ui",
            "depth", "shadow", "refl",
            "land", "statics", "grass", "water", "sky",
            "post", "dbg", "other"
        };
        return n[s];
    }

    inline thread_local Stage g_stage = Other;
    inline std::uint32_t g_calls[COUNT] = {};
    inline std::uint32_t g_prims[COUNT] = {};

    // Count one draw call (and optionally its primitive count) under the current
    // stage. Call immediately before forwarding the draw to the device.
    inline void count(unsigned primCount = 0) {
        g_calls[g_stage] += 1;
        g_prims[g_stage] += primCount;
    }

    inline void reset() {
        for (int i = 0; i < COUNT; ++i) { g_calls[i] = 0; g_prims[i] = 0; }
    }

    // Scope the current stage; restores the previous stage on exit so nested
    // passes (reflection -> land) unwind correctly.
    struct ScopedStage {
        Stage prev;
        explicit ScopedStage(Stage s) : prev(g_stage) { g_stage = s; }
        ~ScopedStage() { g_stage = prev; }
        ScopedStage(const ScopedStage&) = delete;
        ScopedStage& operator=(const ScopedStage&) = delete;
    };

    // Called once per frame at Present: emits per-frame Tracy plots, logs the
    // 60-frame breakdown (gated on LogDistantPipeline), then resets the counters.
    // Defined in mged3d8device.cpp to keep this header free of heavy includes.
    void logFrame();

} // namespace DrawStats
