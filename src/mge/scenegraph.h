#pragma once

// Per-frame scene-graph snapshot owned MGE-side. The DataHandler pointer
// is self-sourced from Morrowind's engine global at 0x7C67E0 (matching
// what TES3::DataHandler::get() reads on the MWSE side) on first call
// to getDataHandler(); a per-frame trigger from DistantLand::renderStage0
// drives onFrameReady() exactly once per frame (renderStage0 is
// stage0Complete-gated by mged3d8device, so it never fires twice in the
// same frame). MGE walks the scene roots (worldObjectRoot +
// worldPickObjectRoot + sgSunlight) recursively, classifies each visited
// node by RTTI, and extracts a POD view per subtype. Currently exposes
// NiPointLight (FFE many-lights consumer) and NiDirectionalLight (sun
// direction + colour, no consumer yet — staged for the dynamic-shadows
// work). Future per-subtype POD vectors (geometry casters, water-
// reflection candidates, etc.) plug in at the same dispatch with no
// extra traversal cost.
//
// All public accessors return references to internal storage that
// remains valid until the next onFrameReady() call. Consumers that
// want to outlive a frame must copy.
//
// Previously the bridge was driven from MWSE via MGEAPIv4::setDataHandler
// + MGEAPIv4::onSceneGraphReady. MWSE removed its side of that ABI on the
// sharedse-ni-unification branch (commit d2a92c596d on MWSE); MGE no
// longer expects MWSE to push either signal — both are self-sourced.

#include <cstdint>
#include <vector>

namespace MGE::SceneGraph {

    // Returns the engine's TES3::DataHandler pointer (typed as void* —
    // MGE consumes it via offset-based access in datahandler_view.h, not
    // by including the TES3 type). Self-sources from the engine global
    // at 0x7C67E0 on first call once DataHandler is constructed; returns
    // nullptr before that point.
    void* getDataHandler();

    // Trigger called from DistantLand::renderStage0 once per frame. Walks
    // the scene graph (synchronously or via the async worker depending on
    // Configuration.UseAsyncSceneGraphWalk) and refreshes the snapshot.
    // No-op if DataHandler hasn't been constructed yet.
    void  onFrameReady();

    // POD view of point lights, suitable for consumers that do not
    // include SharedSE NI headers (e.g. ffeshader.cpp, which can't
    // take the prelude due to d3dx9 SDK conflicts). Fields are raw
    // NI::PointLight values — diffuse pre-multiplied by dimmer,
    // attenuation coefficients copied verbatim, radius read from the
    // engine's overloaded specular.r slot. No engine-state branching:
    // the patterns the master constant-array path decodes are markers
    // Morrowind sets on its D3DLIGHT9 state via SetLight, not on the
    // raw NI fields. Consumers apply the standard 1/(k0 + k1·d + k2·d²)
    // attenuation formula directly.
    struct PointLight {
        float worldPos[3];   // worldTransform.translation
        float diffuse[3];    // pl->diffuse.rgb * pl->dimmer
        float falloff[3];    // (constantAttenuation, linearAttenuation, quadraticAttenuation)
        float radius;        // specular.r — Bethesda's modder-set fade radius
    };
    const std::vector<PointLight>& pointLights();

    // POD view of directional lights. In vanilla MW the only entry is the
    // sun (the sgSunlight DataHandler root is itself a NiDirectionalLight);
    // the walk classifies by RTTI so any modder-added directional lights
    // would surface here too. World-space direction is computed as
    // worldTransform.rotation * NI::DirectionalLight::direction — the sun's
    // local direction is fixed at NIF authoring time and the day/night
    // controller drives worldTransform.rotation, so the multiply is what
    // captures the animated sun vector. Diffuse is pre-multiplied by
    // dimmer to match the PointLight convention; ambient is raw (the
    // engine doesn't scale sun ambient by dimmer either).
    struct DirectionalLight {
        float worldDir[3];   // worldTransform.rotation * direction
        float diffuse[3];    // dl->diffuse.rgb * dl->dimmer
        float ambient[3];    // dl->ambient.rgb
    };
    const std::vector<DirectionalLight>& directionalLights();

    // Snapshot lock helpers — required when the async walk is active
    // (Use Async Scene Graph Walk INI flag). The worker thread swaps
    // the public vectors atomically under this lock; consumers that
    // want race-free access to pointLights() / directionalLights() must
    // hold the lock for the duration of their reads.
    //
    // In synchronous mode (default), the lock is a no-op — the walk
    // and consumer reads are on the same thread by construction. The
    // helpers can be called unconditionally; correctness is unchanged.
    //
    // RAII pattern recommended:
    //   {
    //       MGE::SceneGraph::SnapshotReadLock lk;
    //       const auto& lights = MGE::SceneGraph::pointLights();
    //       // ... use lights ...
    //   }   // lock released
    //
    // Worker swap is microseconds; consumer reads typically <1 ms. Lock
    // contention is near-zero in practice.
    void lockSnapshot();
    void unlockSnapshot();

    class SnapshotReadLock {
    public:
        SnapshotReadLock()  { lockSnapshot();   }
        ~SnapshotReadLock() { unlockSnapshot(); }
        SnapshotReadLock(const SnapshotReadLock&) = delete;
        SnapshotReadLock& operator=(const SnapshotReadLock&) = delete;
    };

    // Increments each time onFrameReady() actually rebuilds the cache
    // (skipped rebuilds do not bump it). Consumers can use this as a
    // cheap "should I re-derive my own derived state" key.
    uint64_t frameRevision();

    // Cumulative count of frames walked since the bridge was stamped (one
    // walk per onFrameReady call when the INI knob is on). Mainly for
    // diagnostic use — divide cumulative per-call counters by this to
    // get per-frame averages.
    uint64_t frameCount();

}
