#pragma once

// Typed offset-based reader for TES3::WorldController — the sibling of
// datahandler_view.h, same contract: MGE consumes engine state by reading known
// offsets and never includes TES3 game-class headers. NI-typed pointers come back
// typed (MGE has SharedSE); TES3-typed pointers stay void*.
//
// Unlike DataHandlerView, the WorldController pointer is self-sourced here
// (engine global 0x7C67DC) because every field below is reached through it and
// callers would otherwise all repeat the same null check. Each accessor returns
// null/false when the global is not yet populated (pre-load, main menu).
//
// Offsets are MWSE-provided (MWSE/TES3WorldController.h, TES3WeatherController.h),
// which is what keeps this inside PRIME DIRECTIVE 6 — no disassembly of our own.
//
// The weather/sky roots exist for the absorbed CullShow traversal's owned-coverage
// classifier (enginecull.cpp): leaves under them must keep engine-displaying
// because MW's own scene-0 draws are what feed MGE's sky capture and the AT3
// sorted-alpha capture. Getting one of these null does NOT mean "not under a
// weather root" — it means unknown, and the classifier treats unknown as
// EngineDraws.

#include <cstddef>
#include <cstdint>

namespace NI {
    struct Node;
    struct Camera;
}

namespace MGE::WorldControllerView {

    // Engine global holding the WorldController*.
    constexpr uintptr_t ADDR_worldController        = 0x7C67DC;

    // WorldController fields.
    constexpr size_t OFF_weatherController          = 0x58;
    constexpr size_t OFF_worldCameraData            = 0x124 + 0x10;  // CameraData::camera
    constexpr size_t OFF_flagMenuMode               = 0xD6;

    // WeatherController fields (NI::Pointer<NI::Node> — raw pointer is the first
    // and only member, so reading a Node** at the offset is the same load).
    constexpr size_t OFF_sgSkyRoot                  = 0x4C;
    constexpr size_t OFF_sgRainRoot                 = 0x5C;
    constexpr size_t OFF_sgSnowRoot                 = 0x60;
    constexpr size_t OFF_sgStormRoot                = 0x6C;

    inline void* worldController() {
        return *reinterpret_cast<void**>(ADDR_worldController);
    }

    inline void* weatherController() {
        void* wc = worldController();
        if (!wc) return nullptr;
        return *reinterpret_cast<void**>(static_cast<unsigned char*>(wc) + OFF_weatherController);
    }

    // The engine's main world camera. MWBridge::getWorldCamera() reads the same
    // field; this exists so the traversal TU doesn't pull in mwbridge.h just for
    // one pointer, and so the camera and the roots come from one view.
    inline NI::Camera* worldCamera() {
        void* wc = worldController();
        if (!wc) return nullptr;
        return *reinterpret_cast<NI::Camera**>(static_cast<unsigned char*>(wc) + OFF_worldCameraData);
    }

    inline bool menuMode() {
        void* wc = worldController();
        if (!wc) return false;
        return *reinterpret_cast<bool*>(static_cast<unsigned char*>(wc) + OFF_flagMenuMode);
    }

    namespace detail {
        inline NI::Node* weatherRoot(size_t offset) {
            void* wtr = weatherController();
            if (!wtr) return nullptr;
            return *reinterpret_cast<NI::Node**>(static_cast<unsigned char*>(wtr) + offset);
        }
    }

    inline NI::Node* sgSkyRoot()   { return detail::weatherRoot(OFF_sgSkyRoot); }
    inline NI::Node* sgRainRoot()  { return detail::weatherRoot(OFF_sgRainRoot); }
    inline NI::Node* sgSnowRoot()  { return detail::weatherRoot(OFF_sgSnowRoot); }
    inline NI::Node* sgStormRoot() { return detail::weatherRoot(OFF_sgStormRoot); }

}
