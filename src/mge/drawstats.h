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
        // Depth pre-pass, split by content source. Depth = misc (cleardepth /
        // renderDepthRecorded); DepthCache = renderDepthFromCache (cache geometry,
        // frustum-only post-Phase-1); DepthLand/Statics/Grass = the distant-land
        // depth replays. So DepthCache is the apples-to-apples vs scene0.
        Depth, DepthCache, DepthLand, DepthStatics, DepthGrass,
        // Shadow caster pass, split by source. Shadow = caster misc (per-cascade stencil
        // cube + soften passes; distant-land caster still lands in Land — shares
        // renderDistantLand's tag); ShadowCache = renderShadowFromCache (cache casters,
        // near cascade 0 only); ShadowDL = distant-statics casters (both cascades).
        // ShadowRecv = the receiver re-draw (renderShadow over recordMW: the engine's
        // near scene shaded as shadow receivers). Split from Shadow so the receiver cost
        // is visible separately from the caster passes and never hides inside scene0.
        // In CACHE mode the cache-covered receiver is folded into the color passes
        // (zero extra draws), so ShadowRecv there is only the non-cache remainder.
        Shadow, ShadowCache, ShadowDL, ShadowRecv,
        // Reflection, split by source. Reflection = misc (reflected sky; reflected
        // LOD land still lands in Land — it shares renderDistantLand's tag);
        // ReflStatics = reflected distant statics (the "DL culled" set, near band
        // shader-clipped but still issued); ReflCacheColor / ReflCacheShadow /
        // ReflCacheTerrain = the three cache-injection passes (lit color, the shadow
        // re-draw, and near terrain) — split out so the color/shadow ~2x doubling and
        // terrain weight are visible separately.
        Reflection, ReflStatics, ReflCacheColor, ReflCacheShadow, ReflCacheTerrain,
        Land, Statics, Grass, Water, Sky,   // distant-land content
        Post, Debug, Other,
        // CACHE-mode near scene (scene 0), split from Scene0 so the cache passes don't
        // hide inside the engine's scene count. CacheOpaque = renderCachedOpaque (objects
        // + skinned NPCs from the visKeys set); CacheTerrain = renderCachedTerrain (near
        // landscape patches). In ENGINE mode both are 0 (the engine draws scene 0), so
        // the scene subtotal stays directly comparable between the two modes.
        CacheOpaque, CacheTerrain,
        COUNT
    };

    inline const char* name(Stage s) {
        static const char* const n[COUNT] = {
            "scene0", "scene1", "scene2", "ui",
            "depth", "d.cache", "d.land", "d.stat", "d.grass",
            "shadow", "s.cache", "s.dl", "s.recv",
            "refl", "r.stat", "r.c.color", "r.c.shadow", "r.c.terr",
            "land", "statics", "grass", "water", "sky",
            "post", "dbg", "other",
            "c.opaque", "c.terr"
        };
        return n[s];
    }

    // Persistent "draws:<stage>" plot-name literals. Tracy stores the plot-name
    // POINTER (never copies it), so the name must outlive the capture — a stack
    // buffer would leave Tracy reading freed memory and render the plot label as
    // mojibake. These literals have static lifetime, so the labels stay valid.
    inline const char* plotName(Stage s) {
        static const char* const n[COUNT] = {
            "draws:scene0", "draws:scene1", "draws:scene2", "draws:ui",
            "draws:depth", "draws:d.cache", "draws:d.land", "draws:d.statics", "draws:d.grass",
            "draws:shadow", "draws:s.cache", "draws:s.dl", "draws:s.recv",
            "draws:refl", "draws:r.statics", "draws:r.c.color", "draws:r.c.shadow", "draws:r.c.terr",
            "draws:land", "draws:statics", "draws:grass", "draws:water", "draws:sky",
            "draws:post", "draws:dbg", "draws:other",
            "draws:c.opaque", "draws:c.terr"
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
