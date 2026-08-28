// grass.bin -- the distant grass density field.
//
// Why this file exists at all: measured across lush3_ai.esp's 193,951 references against the real
// LAND heightfield, a grass placement carries ONE bit of information -- "a blade here, of family F".
// z is a constant +16 sink, pitch/roll ARE the terrain normal (tilt - slope = +0.3 deg median),
// yaw is uniform over [0,2pi), scale is uniform [0.9,1.3], and the median cell uses its whole mesh
// palette. So the 34-byte-per-placement record stores almost nothing, and 5.44M of them (185 MB)
// collapse to per-cell tile counts. See tasks/dl-gen-ownership.md section 3.
//
// Resolution is measured, not guessed: quantising an authored field to 256-unit tiles and
// re-scattering loses 6% of its open ground at D=256u, against 34% at 1024u. 32x32 it is.
#pragma once

#include <cstdint>

namespace grassfmt {

constexpr char     kMagic[4]  = { 'M', 'G', 'G', 'R' };
constexpr uint32_t kVersion   = 1;
constexpr uint32_t kTileDim   = 32;          // tiles per cell edge -> 256 world units per tile
constexpr float    kCellSize  = 8192.0f;

// Defaults are the measured values, carried in the file so the host never hardcodes them and a
// future pack with different authoring can ship its own.
constexpr float kDefaultSinkZ    = 16.0f;    // z - terrain_z, p50 over 193,951 refs
constexpr float kDefaultScaleMin = 0.90f;
constexpr float kDefaultScaleMax = 1.30f;
constexpr float kDefaultTiltJitterDeg = 5.0f; // tilt - terrain slope, p5..p95 = -4.8 .. +5.5

#pragma pack(push, 1)

struct Header {
    char     magic[4];
    uint32_t version;
    uint32_t tileDim;          // kTileDim; planes carry tileDim*tileDim bytes
    float    cellSize;

    uint32_t meshCount;        // entries in the mesh-name table
    uint32_t paletteCount;
    uint32_t planeCount;

    uint32_t meshTableOff;     // byte offsets from file start
    uint32_t paletteOff;
    uint32_t planeOff;

    float    sinkZ;
    float    scaleMin;
    float    scaleMax;
    float    tiltJitterDeg;

    uint64_t totalBlades;      // observability: must equal the sum of every plane's counts
    uint32_t clampedTiles;     // tiles whose count exceeded 255 and was clamped (expect 0)
    uint32_t reserved;
};

// Planes are fixed-size, so the host can index straight into the array after a binary search of the
// (cy, cx) sort order. A plane is one (cell, palette) pair; the median cell has 2.
struct PlaneHeader {
    int16_t  cx, cy;
    uint16_t paletteId;
    uint16_t _pad;
    uint32_t blades;           // sum of count[], for a cheap load-time cross-check
    // followed by uint8 count[tileDim * tileDim]
};

#pragma pack(pop)

// Mesh table: meshCount x { uint16 len; char path[len]; }  -- lowercase model path as it appears in
// the plugin's STAT MODL, e.g. "grass\az\luri05.nif". Names rather than static_meshes ordinals on
// purpose: the ordinal is whatever order the NIF pass happened to load in, and a bake that reorders
// would silently repaint the world with the wrong plants.
//
// Palette table: paletteCount x { uint16 n; uint16 meshIdx[n]; uint8 weight[n]; } with weights
// normalised so the host can pick a mesh by hashing into their prefix sum.

inline uint32_t planeStride(uint32_t tileDim) {
    return (uint32_t)sizeof(PlaneHeader) + tileDim * tileDim;
}

}  // namespace grassfmt
