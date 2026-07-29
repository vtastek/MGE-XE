// mgeHost64 — host-owned terrain: the LAND record loader (tasks/forge-terrain.md, T0).
//
// Morrowind's terrain is a hand-authored heightfield living in the plugin files, not in any
// baked artifact. This module reads it directly, in the x64 host, at startup — replacing the
// MGEgui offline "distant land" world bake, whose every OOM was 32-bit address space and whose
// only quality knob was the same knob that triggered the OOM.
//
// It parses `Morrowind.ini [Game Files]` (host cwd is the install dir) and walks each plugin for
// LAND + LTEX, keeping heights (int16, VHGT units), VCLR (hand-painted, unrecoverable) and VTEX
// (16x16 texture indices). VNML is deliberately DROPPED: it is per-original-vertex and becomes
// wrong the moment displacement lands, so normals are recomputed from the heightfield instead.
//
// The whole world is resident: ~81 MB for 3910 cells, so there is no streaming scheme here and
// there does not need to be one.
//
// ABI: this TU is built with the host's DEFAULT MSVC settings, NOT The Forge's
// (_HAS_EXCEPTIONS=0 + IMemory.h's global new/delete override). So every container stays inside
// terrain.cpp and this header exposes POD + raw pointers only — forgerender.cpp must never
// allocate or free anything owned here.

#pragma once

#include <cstdint>

namespace Terrain {

    constexpr int   kCellVerts   = 65;                             // LAND grid is 65x65 vertices
    constexpr int   kCellQuads   = kCellVerts - 1;                 // ...= 64 quads across
    constexpr int   kCellTexRes  = 16;                             // VTEX is 16x16 per cell
    constexpr float kVertSpacing = 128.0f;                         // world units between vertices
    constexpr float kCellSize    = kVertSpacing * kCellQuads;      // 8192 world units per cell
    constexpr float kHeightScale = 8.0f;                           // VHGT units -> world units

    // One exterior cell's decoded LAND record. Vertex (x,y) sits at world
    //   (cellX*kCellSize + x*kVertSpacing, cellY*kCellSize + y*kVertSpacing, height[]*kHeightScale).
    struct LandCell {
        int32_t  cellX, cellY;
        int16_t  height[kCellVerts * kCellVerts];        // [y*65 + x], VHGT units
        uint8_t  color [kCellVerts * kCellVerts * 3];    // [ (y*65 + x)*3 ], VCLR (white if absent)
        uint16_t tex   [kCellTexRes * kCellTexRes];      // [ty*16 + tx], GLOBAL texture id (0 = default)
        int16_t  minHeight, maxHeight;                   // VHGT units, for the per-cell cull bound
        uint8_t  hasColor;                               // 0 = VCLR absent, color[] filled white
        uint8_t  pad[3];
    };

    // Kick the loader off on a background thread. Returns immediately; call once, before the
    // client handshake, so terrain parse overlaps MW's own load. Idempotent.
    void beginLoadAsync();

    // True once the background parse has finished (successfully or not).
    bool isLoaded();

    // Block until the parse finishes, up to timeoutMs (0 = poll). Returns isLoaded().
    bool waitLoaded(uint32_t timeoutMs);

    // --- Results. Only valid once isLoaded(); the data is then immutable for the process life. ---

    uint32_t        cellCount();
    const LandCell* cells();                        // cellCount() entries, plugin discovery order
    const LandCell* cellAt(int32_t x, int32_t y);   // nullptr if that cell has no LAND record
    int32_t         slotAt(int32_t x, int32_t y);   // index into cells(), or -1 (neighbour tables)

    // Inclusive grid extent over every loaded cell (all zero when cellCount() == 0).
    void extent(int32_t& minX, int32_t& minY, int32_t& maxX, int32_t& maxY);

    // Global land-texture table. Id 0 is the built-in default (`_land_default.tga`); ids are
    // assigned in first-seen order across the whole load, deduped by lowercased file name.
    uint32_t    texCount();
    const char* texName(uint32_t id);               // "" if out of range

    // Resident bytes of the decoded cell store, for the log/budget line.
    uint64_t residentBytes();

    // --- GPU staging pack -------------------------------------------------------------------
    // The renderer wants the heightfield as flat, uint-aligned arrays it can hand to one buffer
    // upload each, not as an array of structs. Heights pack two int16 per uint (so a cell's 4225
    // vertices occupy kHeightStrideUints, padded to stay uint-aligned); colour is one 0x00BBGGRR
    // uint per vertex. Both index as `slot * stride + (y*65 + x)`.
    //
    // buildGpuPack() is idempotent and costs one pass over the cell store; releaseGpuPack() frees
    // the staging copies once they are on the GPU (the decoded LandCell store stays — the cull
    // bounds and VTEX live there).
    constexpr uint32_t kHeightStrideUints = (kCellVerts * kCellVerts + 1) / 2;   // 2113
    constexpr uint32_t kColorStrideUints  = kCellVerts * kCellVerts;             // 4225
    // ONE uint per texture square, not two packed uint16. The renderer rewrites these LTEX ids into
    // (bucket<<16)|layer texture slots before upload, and a slot needs the full 32 bits — packing
    // two per uint silently truncated every real texture to 0 (= white) the first time it was tried.
    // 256 uints per cell is 4 MB for the whole world; there was never anything to save here.
    constexpr uint32_t kTexStrideUints    = kCellTexRes * kCellTexRes;           // 256

    bool            buildGpuPack();
    void            releaseGpuPack();
    const uint32_t* packedHeights();   // cellCount() * kHeightStrideUints, or nullptr
    const uint32_t* packedColors();    // cellCount() * kColorStrideUints,  or nullptr
    const uint32_t* packedTex();       // cellCount() * kTexStrideUints,    or nullptr
                                       // (one global texture id per uint, indexed [ty*16 + tx])

    // --- Land texture files -----------------------------------------------------------------
    // LTEX records a `.tga` name but installs ship `.dds`, mods use subpaths (`hf\lnd\…`) and mixed
    // case, and the vanilla textures are inside BSAs. Resolution order matches the DL generator's
    // (MGEgui/DistantLand/BSA.cs) so the same file wins as before: loose .dds, loose as-recorded,
    // BSA .dds, BSA as-recorded.
    //
    // Returns a pointer to the file bytes, valid ONLY until the next call (one internal buffer —
    // the caller parses/uploads and moves on). nullptr + *sizeOut = 0 when unresolvable.
    const uint8_t* readLandTextureFile(uint32_t texId, uint32_t* sizeOut);

    // Build the case-insensitive `Data Files\Textures\**` index + the BSA directory. Idempotent;
    // readLandTextureFile calls it on demand. Counts are for the residency log.
    void     buildTextureIndex();
    uint32_t looseTextureCount();
    uint32_t bsaTextureCount();

    // On the MO2 / Wrye Mash "the ini isn't what the game loaded" failure: the plan wanted a plugin
    // list from the client to diff against ours. That check is not reachable — MW's real load order
    // lives in the engine, SharedSE does not expose it, and digging it out is disassembly (PD6). The
    // client reading the same ini back to us proves nothing. So the failure is caught by its
    // CONSEQUENCE instead, which is both reachable and strictly more general:
    //   - a plugin the ini lists but disk lacks  -> named in the load log ("plugin missing: ...")
    //   - a cell with no LAND record anywhere    -> named in the load log (heightless / no terrain)
    //   - a gap within one cell of the camera    -> g_terrainEyeCellMissing in forgerender.cpp:
    //                                               logs the cell and hands the near field to MW
    // The last one is the one that matters, because a name diff can come back clean and still leave
    // a hole: it catches every cause, including ones no plugin list would show.

}
