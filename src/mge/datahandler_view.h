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

    // Active exterior cell set. exteriorCellData is an array of NINE POINTERS (0x04, stride 4)
    // to ExteriorCellData { u8 state; Cell* cell; void* landRenderData; } — the engine's own
    // 3x3 residency table. A slot only counts as loaded when the pointer is non-null, state ==
    // Loaded (1) and cell is non-null (MWSE's DataHandler::ExteriorCellData::isFullyLoaded).
    // The slot ORDER is TES3::CellGrid (NW,N,NE,W,C,E,SW,S,SE), but nothing here depends on it:
    // each cell's own gridX/gridY is read out of the Cell record, so the mapping cannot be got
    // wrong. Read-only field reads — no detour (MWSE already hooks the cell-attach path;
    // a second hook there is the double-hook hazard).
    constexpr size_t OFF_exteriorCellData          = 0x04;
    constexpr size_t EXT_CELL_DATA_COUNT           = 9;
    constexpr size_t OFF_ecdState                  = 0x00;   // ExteriorDataLoadingState, u8; 1 = Loaded
    constexpr size_t OFF_ecdCell                   = 0x04;
    constexpr unsigned char EXT_CELL_STATE_LOADED  = 1;
    // TES3::Cell::variantData.exterior (union @0x1C): { PackedColor, Land*, int gridX, int gridY }.
    constexpr size_t OFF_cellExteriorGridX         = 0x24;
    constexpr size_t OFF_cellExteriorGridY         = 0x28;

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

    // exteriorCellData[i], i in [0,9). Null when the engine has no cell in that grid slot.
    inline void* exteriorCellData(void* dh, size_t i) {
        return *(reinterpret_cast<void**>(static_cast<unsigned char*>(dh) + OFF_exteriorCellData) + i);
    }
    // A slot is LOADED only when it exists, its state is Loaded and it carries a cell record.
    // Anything else (background-loading, pending commit, unloading, absent) reads as not loaded —
    // the fail-safe direction for every consumer: "the engine is not drawing this cell yet".
    inline bool exteriorCellLoaded(void* ecd) {
        if (!ecd) { return false; }
        const unsigned char state = *(static_cast<unsigned char*>(ecd) + OFF_ecdState);
        if (state != EXT_CELL_STATE_LOADED) { return false; }
        return *reinterpret_cast<void**>(static_cast<unsigned char*>(ecd) + OFF_ecdCell) != nullptr;
    }
    inline void* exteriorCellRecord(void* ecd) {
        return *reinterpret_cast<void**>(static_cast<unsigned char*>(ecd) + OFF_ecdCell);
    }
    // Exterior cell grid coords, read from the cell record itself (valid only for exterior cells —
    // the union holds interior lighting otherwise).
    inline int cellExteriorGridX(void* cell) {
        return *reinterpret_cast<int*>(static_cast<unsigned char*>(cell) + OFF_cellExteriorGridX);
    }
    inline int cellExteriorGridY(void* cell) {
        return *reinterpret_cast<int*>(static_cast<unsigned char*>(cell) + OFF_cellExteriorGridY);
    }

}
