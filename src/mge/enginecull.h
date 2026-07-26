#pragma once

// EngineCull — MGE's own NiAVObject::CullShow traversal, absorbed from the
// msoc-plugin so the engine-driven discovery feed survives MSOC's retirement.
//
// WHY THIS EXISTS. MSOC has two outputs and only one is redundant:
//   * the occlusion verdict — DEAD. The Forge host's two-phase Hi-Z GPU cull
//     replaced it, and MSOC.log confirms it does literally zero occlusion for us
//     (rasterized=0, classifyUs=0, queryOccluded=0/0 every frame).
//   * the DISCOVERY feed — load-bearing. It is how MGE learns which scene-graph
//     leaves the engine actually drew this frame. Removing it (hostCullOnly=true)
//     costs a 7.05 ms full cache walk. See tasks/msoc-retirement.md.
// So retiring MSOC means REPLACING the feed, not deleting the consumer. This
// module is that replacement: msoc's already-shipping engine-faithful traversal
// with every occlusion branch removed.
//
// STATUS: D3 — the feed is live. install() takes the prologue (D2) and
// classifyNow() runs the traversal EARLY, at BeginScene(0), deferring geometry
// display to the engine's own pass. That is what replaces msoc's discovery feed.
// Plan: tasks/msoc-detour-absorb.md.
//
// THE FRAME, in order:
//   BeginScene(0)  beginFrame()   — roots, owned flags, previous frame's stats
//                  classifyNow()  — deferring traversal from g_topLevelRoot;
//                                   fires the visible-geom sink with the drawn set
//   engine's own top-level CullShow  — matched by root identity, displays the
//                                   collected leaves minus the ones Forge covers
// Neither half is optional: the classify without the display draws nothing, and
// the display without the classify is a stale (freed) leaf list. beginFrame()
// re-arms both and counts any frame where the pair did not complete.
//
// THE SHARP EDGE. Installation is a 5-byte prologue overwrite at 0x6EB480 with
// NO trampoline — the original CullShow is *gone* once patched, so this body must
// be behaviourally identical to the engine's for all seven of its direct callers,
// not just the main scene. The culling-plane bitfield save/restore is the part to
// review hardest: a missed unflip corrupts engine culling state for every later
// traversal in the frame, and there is nothing to fall back to.

#include <cstdint>

namespace NI {
    struct AVObject;
    struct Camera;
    struct Node;
}

namespace MGE::EngineCull {

    // Who renders a given leaf while the Forge composite owns part of the frame.
    // Mirrors msoc's EngineCoverage 1:1 — the rules are load-bearing and the
    // rationale for each lives on classifyEngineCoverage in the .cpp.
    //
    // BOTH skippable classes MUST stay strict subsets of what MGE's proxy
    // actually rejects and the host actually redraws. A leaf classified skippable
    // that nothing redraws is a visual hole. Any doubt resolves to EngineDraws.
    enum class Coverage : uint8_t {
        EngineDraws     = 0,   // the engine's display() must run
        OpaqueRedundant = 1,   // proxy rejects every colour pass per-DIP
        AlphaCovered    = 2,   // the host's sorted-alpha pass (AT1) redraws it
    };

    // Owned display-skip flags, pushed once per frame before the traversal. Bit
    // meanings were kept identical to msoc's kOwnedOpaque / kOwnedAlpha so the D2
    // A/B compared like with like; earlyClassifyMainScene is the only caller now.
    constexpr int kOwnedOpaque = 1 << 0;
    constexpr int kOwnedAlpha  = 1 << 1;

    // Per-frame setup, called from earlyClassifyMainScene immediately before
    // classifyNow(). Latches the owned flags, re-reads the sky/landscape/weather
    // roots the coverage classifier needs, and re-arms the classify/display pair.
    // Cheap; safe to call on frames where the traversal never runs.
    void beginFrame(int ownedFlags);

    // Run the world-camera traversal NOW and hand the drawn set to the sink,
    // deferring every geometry leaf to the engine's own top-level CullShow.
    // `camera` may be the world camera (validated) or null (we resolve it) — the
    // signature is msoc's classifyMainSceneNow, kept when the two producers had to
    // be drop-in comparable.
    //
    // Returns 0 on success, else the guard that declined. Codes are msoc's,
    // so one legend covers both producers:
    //   1 re-entrant  3 noRootCaptured  4 alreadyClassified  5 sceneHandled
    //   6 noMainCamera  7 menuMode  8 wrongCamera  9 noDataHandler
    //   11 staleRoot  12 notInstalled (ours; msoc has no equivalent)
    // msoc's 2 (!inRenderMainScene) and 10 (detourDeclined) have no counterpart:
    // this call IS the traversal driver, so there is no second gate to decline
    // and no separate scene-phase flag to consult.
    int classifyNow(void* camera);

    // Sink for the current-frame drawn set. Signature inherited from msoc's
    // FnVisibleGeomCallback, which is what let D5 delete the msoc feed without
    // touching DistantLand's onVisibleGeom at all.
    using FnVisibleGeom = void(__cdecl*)(void* const* shapes, const float* bounds, int count);
    void setVisibleGeomCallback(FnVisibleGeom cb);

    // Coverage verdict for one deferred leaf, memoised per NiTriShape until
    // invalidateCoverageCache(). Exposed for D2's A/B: it can be diffed against
    // msoc's classification leaf-by-leaf without installing the detour.
    // Precondition: obj is an NiTriBasedGeom (the classifier casts for
    // skinInstance, exactly as msoc's does).
    Coverage classifyCoverage(NI::AVObject* obj);

    // Drop the memoised coverage classes. Coverage is stable for a leaf within a
    // cell (it is a function of properties + root membership), so this is a
    // cell-change hook, not a per-frame one.
    void invalidateCoverageCache();

    // The engine-faithful CullShow body — the detour target once installed, and
    // the only genuinely foreign code in this module. Declared here so D2 can
    // take its address; nothing else should call it directly.
    void __fastcall cullShowBody(NI::AVObject* self, void* edx, NI::Camera* camera);

    // Display the leaves the traversal collected, minus the ones the Forge side
    // already covers. Pairs with a deferring traversal; a no-op otherwise.
    void displayDeferred();

    // Install the prologue patch at NiAVObject::CullShow (0x6EB480). Refuses and
    // returns false if that prologue is already detoured — two detours on one
    // 5-byte patch is unrecoverable, so the check is not optional.
    // isInstalled() doubles as "which producer owns the feed this session":
    // install() refuses whenever msoc could own the prologue, so the two are
    // mutually exclusive by construction.
    bool install();
    bool isInstalled();

    // Diagnostics for the D4 A/B. Counters mirror the MSOC.log names so the two
    // paths can be compared directly: deferred / ownedOpaqueSkipped /
    // ownedAlphaSkipped must match msoc at the same camera.
    struct Stats {
        uint32_t deferred;            // NiTriBasedGeom leaves collected
        uint32_t appCulled;
        uint32_t frustumCulled;
        uint32_t recursiveCalls;
        uint32_t fed;                 // leaves handed to the sink
        uint32_t ownedOpaqueSkipped;
        uint32_t ownedAlphaSkipped;
        uint32_t displayed;
        // Frames where classifyNow() collected leaves but the engine's display
        // pass never arrived to draw them. Should be 0. Anything else means the
        // root-identity match is missing the engine's real top-level pass, which
        // is a frame of missing world geometry, not a slow frame.
        uint32_t missedDisplays;
    };
    const Stats& lastFrameStats();

}
