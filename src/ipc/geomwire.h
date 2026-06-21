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

    // Per-part header preceding the part's vertex+index data in the batch blob.
    struct GeomPartWire {
        std::uint32_t slot;          // dense host array index assigned by the client
        std::uint16_t revisionID;    // cache revisionID at capture (host re-uploads on change)
        std::uint16_t pad;
        std::uint32_t vertexCount;   // GeomVertexWire count
        std::uint32_t indexCount;    // uint16 index count (= triangleCount * 3)
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
