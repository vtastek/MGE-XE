#pragma once

// Wire format for the M1 opaque-geometry seam (32-bit MW cache -> 64-bit Forge host).
// Deliberately free of STL and D3D9/D3D12 headers so BOTH the host TU (forgerender.cpp,
// which keeps STL/D3D out for Forge ABI reasons) and the 32-bit client can include it.
//
// A geometry upload batch is a flat byte blob shipped through a shared byte vector:
//   [GeomPartWire][verts: GeomVertexWire * vertexCount][indices: uint16 * indexCount]
//   ... repeated partCount times (partCount carried in the RPC params).
// Parts are slot-indexed: the client assigns each cached NiTriShape* a dense slot, and
// the host stores meshes in a flat array indexed by that slot (no host-side hashing).
// Per-frame draw lists (M1c) reference the same slots.

#include <cstdint>

namespace IPC {

#pragma pack(push, 4)

    // Model-space vertex: position + normal. M1 is flat/normal-shaded (no UV, no colour
    // yet — M2/M3 extend this). 24 bytes.
    struct GeomVertexWire {
        float px, py, pz;
        float nx, ny, nz;
    };

    // M-Skinning: bind-pose skinned vertex — position + normal + top-4 bone influences
    // (weights + packed UBYTE4 palette indices). Same flat fidelity as GeomVertexWire
    // (UV/colour omitted). The GPU palette-skins this with the per-frame bone window.
    // `indices` packs idx0 in the low byte, matching D3D9 SkinnedVertex.indices so the
    // FSL R8G8B8A8_UINT unpacks in the same order. 44 bytes.
    struct SkinnedVertexWire {
        float px, py, pz;
        float nx, ny, nz;
        float w0, w1, w2, w3;
        std::uint32_t indices;       // packed UBYTE4 bone palette indices (idx0 = low byte)
    };

    // GeomPartWire::flags bits.
    constexpr std::uint16_t kGeomFlagSkinned = 0x1;   // part vertices are SkinnedVertexWire (stride 44)

    // Per-part header preceding the part's vertex+index data in the batch blob. When
    // (flags & kGeomFlagSkinned), the part's vertices are SkinnedVertexWire (stride 44)
    // and numBones is the part's bone count; otherwise GeomVertexWire (stride 24). The
    // index-buffer layout is unchanged either way.
    struct GeomPartWire {
        std::uint32_t slot;          // dense host array index assigned by the client
        std::uint16_t revisionID;    // cache revisionID at capture (host re-uploads on change)
        std::uint16_t flags;         // kGeomFlag* bits (was pad)
        std::uint32_t vertexCount;   // GeomVertexWire / SkinnedVertexWire count
        std::uint32_t indexCount;    // uint16 index count (= triangleCount * 3)
        std::uint16_t numBones;      // skinned parts: bone count (<= kMaxBones); 0 otherwise
        std::uint16_t pad2;
    };

    // M1c per-frame draw item: which uploaded mesh (slot) to draw, with its current
    // model->world transform (D3DXMATRIX bytes, row-major — uploaded straight into the
    // host's gObject cbuffer; see opaque.srt.h for the no-transpose convention). The
    // per-frame draw list is an array of these in a chunked byte vec, with the camera
    // view*proj carried inline in the RenderFrame RPC params. 68 bytes.
    struct DrawItemWire {
        std::uint32_t slot;
        float         world[16];
    };

    // M-Skinning per-frame draw item: which uploaded skinned mesh (slot) to draw, the
    // part's bone count, and its mirror flag (left-side parts reuse the right mesh via a
    // negative-scale bone → inside-out without the mirror pipeline). There is NO per-draw
    // world matrix — the bone palette is already world-space. A SkinnedDrawWire is
    // followed INLINE by numBones * 64 palette bytes (each bone a model->world D3DXMATRIX,
    // row-major). A skinned-draw blob = repeated [SkinnedDrawWire][palette]. 12 bytes + palette.
    struct SkinnedDrawWire {
        std::uint32_t slot;
        std::uint32_t numBones;
        std::uint32_t mirror;        // 1 = mirrored (negative-determinant); pick CW pipeline
    };

#pragma pack(pop)

    // Transport chunk for the shared upload vector. The IPC Vec reserves
    // maxSize * windowBytes bytes (vec.cpp), so a byte-element vector with a multi-MB
    // window overflows uint32 — exactly the trap the occlusion-mask path avoids by
    // using a big chunk element (few elements, large each). We mirror that: an 8MB
    // window is 8 chunks of 1MB, giving a modest 64MB reservation. Both sides alloc /
    // read the vector as GeomChunk so size() and &vec[0] stay consistent; the exact
    // byte length travels in the RPC params. assign_bytes flat-copies the blob in.
    constexpr unsigned kGeomChunkBytes = 1u << 20;             // 1 MB
    constexpr unsigned kGeomChunks     = 8;                    // 8 MB window
    constexpr unsigned kGeomWindowBytes = kGeomChunkBytes * kGeomChunks;
    struct GeomChunk { char b[kGeomChunkBytes]; };

}
