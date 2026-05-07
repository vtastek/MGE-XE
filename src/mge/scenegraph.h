#pragma once

// Per-frame scene-graph snapshot owned MGE-side. MWSE pushes the
// TES3::DataHandler pointer once on first onSceneGraphReady, then
// signals the start of each frame's safe-to-walk window. MGE walks
// the scene roots (worldObjectRoot + worldPickObjectRoot + sgSunlight)
// recursively, holds NI::Pointer<> refs for the duration of the frame,
// and exposes flat typed views for consumers (FFE many-lights shader
// today; future: water reflection, shadow maps, custom post effects).
//
// All public accessors return references to internal storage that
// remains valid until the next onFrameReady() call. Consumers that
// want to outlive a frame must copy.

#include <cstdint>
#include <vector>

namespace NI {
    struct AVObject;
    struct Light;
}

namespace MGE::SceneGraph {

    // Bridge wiring — called from MGEAPIv4 impl.
    void  setDataHandler(void* dataHandler);
    void* getDataHandler();
    void  onFrameReady();

    // Consumer views over the typed scene-graph capture. References
    // remain valid until the next onFrameReady().
    const std::vector<NI::Light*>&    lights();
    const std::vector<NI::AVObject*>& nodes();

    // Pre-extracted POD view of point lights, suitable for consumers
    // that do not include SharedSE NI headers (e.g. ffeshader.cpp,
    // which can't take the prelude due to d3dx9 SDK conflicts). All
    // fields are computed during onFrameReady() with the same engine
    // branching the 8-light constant path applies (standard / MCP
    // magic / projectile / spell-effect light variants), so the
    // texture-light path matches what the engine would have done for
    // these same lights.
    struct PointLight {
        float worldPos[3];   // worldTransform.translation
        float diffuse[3];    // engine-derived diffuse, pre-multiplied by dimmer
        float ambient;       // engine-derived per-light ambient term
        float falloff[3];    // engine-derived (constant, linear, quadratic)
        float radius;        // specular.r — Bethesda's modder-set fade radius
    };
    const std::vector<PointLight>& pointLights();

    // Increments each time onFrameReady() actually rebuilds the cache
    // (skipped rebuilds do not bump it). Consumers can use this as a
    // cheap "should I re-derive my own derived state" key.
    uint64_t frameRevision();

}
