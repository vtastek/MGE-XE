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
    constexpr size_t OFF_gvarGameHour               = 0xA8;          // TES3::GlobalVariable* GameHour

    // TES3::GlobalVariable::value.
    constexpr size_t OFF_globalValue                = 0x34;

    // WeatherController fields (NI::Pointer<NI::Node> — raw pointer is the first
    // and only member, so reading a Node** at the offset is the same load).
    constexpr size_t OFF_sgSkyRoot                  = 0x4C;
    constexpr size_t OFF_sgRainRoot                 = 0x5C;
    constexpr size_t OFF_sgSnowRoot                 = 0x60;
    constexpr size_t OFF_sgStormRoot                = 0x6C;

    // WeatherController sun-hour schedule. These are the boundaries Glow in the
    // Dahrk derives its day/night window switch from (GlowInTheDahrk/interop.lua
    // getSunHours), which is why they are read here rather than approximated from
    // the sun elevation — a derived elevation disagrees with the mod's near-field
    // windows by tens of game-minutes at dawn and dusk.
    constexpr size_t OFF_sunriseHour                = 0xDC;
    constexpr size_t OFF_sunsetHour                 = 0xE0;
    constexpr size_t OFF_sunriseDuration            = 0xE4;
    constexpr size_t OFF_sunsetDuration             = 0xE8;
    constexpr size_t OFF_sunPreSunriseTime          = 0x120;
    constexpr size_t OFF_sunPostSunsetTime          = 0x12C;

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

    // How far the current game hour is INTO the period where Glow in the Dahrk shows a
    // window mesh's lit "on" child: positive = lit, negative = dark, magnitude in game
    // hours from the nearest boundary. The host adds a per-instance stagger to this and
    // picks the night or day variant of a distant window subset from the sign, which is
    // what makes distant land light up at night instead of freezing on the unlit bake.
    //
    // GitD's rule is `hour < sunriseStart || hour > sunsetStop` with
    //   sunriseStart = sunriseHour - sunPreSunriseTime
    //   sunsetStop   = sunsetHour + sunsetDuration + sunPostSunsetTime
    // (GlowInTheDahrk/interop.lua getSunHours + main.lua). Reading the real boundaries is
    // what keeps distant windows in step with the mod's near ones.
    //
    // Returns false, leaving `out` untouched, before a world exists (main menu, pre-load)
    // — MWBridge::getGameHour() is the unguarded twin of the same read, so the global is
    // dereferenced here only after both it and the weather controller are known non-null.
    inline bool glowLitMargin(float& out) {
        void* wc = worldController();
        if (!wc) return false;
        void* wtr = *reinterpret_cast<void**>(static_cast<unsigned char*>(wc) + OFF_weatherController);
        if (!wtr) return false;
        void* gvar = *reinterpret_cast<void**>(static_cast<unsigned char*>(wc) + OFF_gvarGameHour);
        if (!gvar) return false;

        auto weatherFloat = [wtr](size_t offset) {
            return *reinterpret_cast<float*>(static_cast<unsigned char*>(wtr) + offset);
        };
        const float hour = *reinterpret_cast<float*>(static_cast<unsigned char*>(gvar) + OFF_globalValue);
        const float sunriseStart = weatherFloat(OFF_sunriseHour) - weatherFloat(OFF_sunPreSunriseTime);
        const float sunsetStop = weatherFloat(OFF_sunsetHour) + weatherFloat(OFF_sunsetDuration)
                               + weatherFloat(OFF_sunPostSunsetTime);

        // Distance past whichever boundary we are outside of; when inside the lit day both
        // terms are negative and the larger (nearer boundary) is the one that matters.
        const float toDawn = sunriseStart - hour;
        const float toDusk = hour - sunsetStop;
        out = (toDawn > toDusk) ? toDawn : toDusk;
        return true;
    }

}
