#pragma once

// Typed offset-based reader for TES3::DataHandler. MGE-XE consumes only the
// fields it needs from MWSE-side state by reading at known engine offsets;
// it never includes TES3 game-class headers. NI-typed pointer fields are
// returned typed (MGE has SharedSE); TES3-typed pointers are returned as
// void* opaque — useful for identity comparison without dragging in TES3.
//
// The DataHandler pointer itself is self-sourced from the engine global
// at 0x7C67E0 via MGE::SceneGraph::getDataHandler(). All accessors below
// assume the caller has a non-null dh; callers that don't are
// responsible for the null check.
//
// Adding a new field is one constant + one inline. Fields the consumer
// doesn't read remain inert at zero cost.

#include <cstddef>

namespace NI {
    struct Node;
    struct DirectionalLight;
}

namespace MGE::DataHandlerView {

    // Scene roots + globals.
    constexpr size_t OFF_worldObjectRoot           = 0x8C;
    constexpr size_t OFF_worldPickObjectRoot       = 0x90;
    constexpr size_t OFF_worldLandscapeRoot        = 0x94;
    constexpr size_t OFF_sgSunlight                = 0x98;
    constexpr size_t OFF_sgFogProperty             = 0x9C;

    // Cell context.
    constexpr size_t OFF_centralGridX              = 0xA0;
    constexpr size_t OFF_centralGridY              = 0xA4;
    constexpr size_t OFF_cellChanged               = 0xA8;
    constexpr size_t OFF_currentInteriorCell       = 0xAC;
    constexpr size_t OFF_lastExteriorCellPositionX = 0xB8;
    constexpr size_t OFF_lastExteriorCellPositionY = 0xBC;
    constexpr size_t OFF_currentCell               = 0xB540;
    constexpr size_t OFF_lastExteriorCell          = 0xB544;

    inline NI::Node* worldObjectRoot(void* dh) {
        return *reinterpret_cast<NI::Node**>(static_cast<unsigned char*>(dh) + OFF_worldObjectRoot);
    }
    inline NI::Node* worldPickObjectRoot(void* dh) {
        return *reinterpret_cast<NI::Node**>(static_cast<unsigned char*>(dh) + OFF_worldPickObjectRoot);
    }
    inline NI::Node* worldLandscapeRoot(void* dh) {
        return *reinterpret_cast<NI::Node**>(static_cast<unsigned char*>(dh) + OFF_worldLandscapeRoot);
    }
    inline NI::DirectionalLight* sgSunlight(void* dh) {
        return *reinterpret_cast<NI::DirectionalLight**>(static_cast<unsigned char*>(dh) + OFF_sgSunlight);
    }
    // NI::FogProperty header isn't in MGE's SharedSE consumption set yet.
    // Returned as void* until a consumer needs it typed.
    inline void* sgFogProperty(void* dh) {
        return *reinterpret_cast<void**>(static_cast<unsigned char*>(dh) + OFF_sgFogProperty);
    }

    inline int centralGridX(void* dh) {
        return *reinterpret_cast<int*>(static_cast<unsigned char*>(dh) + OFF_centralGridX);
    }
    inline int centralGridY(void* dh) {
        return *reinterpret_cast<int*>(static_cast<unsigned char*>(dh) + OFF_centralGridY);
    }
    inline bool cellChanged(void* dh) {
        return *reinterpret_cast<bool*>(static_cast<unsigned char*>(dh) + OFF_cellChanged);
    }
    inline void* currentInteriorCell(void* dh) {
        return *reinterpret_cast<void**>(static_cast<unsigned char*>(dh) + OFF_currentInteriorCell);
    }
    inline int lastExteriorCellPositionX(void* dh) {
        return *reinterpret_cast<int*>(static_cast<unsigned char*>(dh) + OFF_lastExteriorCellPositionX);
    }
    inline int lastExteriorCellPositionY(void* dh) {
        return *reinterpret_cast<int*>(static_cast<unsigned char*>(dh) + OFF_lastExteriorCellPositionY);
    }
    inline void* currentCell(void* dh) {
        return *reinterpret_cast<void**>(static_cast<unsigned char*>(dh) + OFF_currentCell);
    }
    inline void* lastExteriorCell(void* dh) {
        return *reinterpret_cast<void**>(static_cast<unsigned char*>(dh) + OFF_lastExteriorCell);
    }

}
