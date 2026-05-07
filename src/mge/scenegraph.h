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

    // Consumer views. References remain valid until the next onFrameReady().
    const std::vector<NI::Light*>&    lights();
    const std::vector<NI::AVObject*>& nodes();

    // Increments each time onFrameReady() actually rebuilds the cache
    // (skipped rebuilds do not bump it). Consumers can use this as a
    // cheap "should I re-derive my own derived state" key.
    uint64_t frameRevision();

}
