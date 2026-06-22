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

    // Model-space vertex: position + normal + base-map UV (set e.baseUV). 32 bytes.
    // (Texturing Phase 1 added u,v; multi-map extra UV sets are a later milestone.)
    struct GeomVertexWire {
        float px, py, pz;
        float nx, ny, nz;
        float u, v;
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
        std::uint32_t texIndex;   // bindless gTextures[] slot for the base map (0 = default white)
    };

    // Texture-residency upload (Phase 2 bindless texturing). The client resolves each unique
    // texture to a dense slot, reads its RAW DDS bytes via BSA::loadFileBytes, and ships
    // [TexUploadWire][dds bytes] entries through the geometry channel's chunked vec. The host
    // parses the DDS (BCn/uncompressed + mip chain) into gTextures[slot]. slot 0 is the host's
    // default white texture (never uploaded). Sent once per unique texture (no re-upload).
    struct TexUploadWire {
        std::uint32_t slot;
        std::uint32_t byteLen;    // length of the DDS blob that follows inline
    };

    // Bindless texture-array capacity (client residency cap == host gTextures[] size; the host
    // mirrors this as MAX_TEXTURES in opaque.srt.h / kMaxTextures in forgerender.cpp).
    // MUST stay in lock-step with host MAX_TEXTURES. Capped at 1024: Forge's addDescriptorSet
    // crashes building a bindless descriptor table >1024 entries on this stack (2048 faults in
    // consume_descriptor_handles; 1024 verified OK). Beyond 1024 unique textures in a session
    // fall back to white — a real limit pending LRU eviction or SM6.6 ResourceDescriptorHeap.
    constexpr std::uint32_t kMaxTextures = 1024;

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
    // using a big chunk element (few elements, large each). We mirror that: the window
    // is kGeomChunks chunks of 1MB, so reservation = maxSize(=window) * 1MB stays well
    // under uint32. Both sides alloc / read the vector as GeomChunk so size() and
    // &vec[0] stay consistent; the exact byte length travels in the RPC params.
    // assign_bytes flat-copies the blob in.
    //
    // Geometry and textures have SEPARATE window sizes (both use the GeomChunk element type,
    // just different chunk counts). Geometry batches are small (vertex+index data, packed per
    // window) — 8MB is ample. Textures need a big window: a single DDS must fit in ONE window
    // or resolveTextureSlot/flushTextures drop it to white. 4096x4096 replacers (DXT1 ~11MB,
    // DXT5/BC7+mips ~22MB) need ~32MB. Keeping them separate avoids paying the 32MB reservation
    // on the geom vec too. NOTE the d3d8.dll client is 32-bit: each vec RESERVES (not commits) a
    // window of address space, so geom(8) + tex(32) = 40MB reserved — budget the 2-4GB space.
    constexpr unsigned kGeomChunkBytes = 1u << 20;             // 1 MB (shared element size)
    constexpr unsigned kGeomChunks     = 8;                    // 8 MB geometry window
    constexpr unsigned kTexChunks      = 32;                   // 32 MB texture window
    constexpr unsigned kGeomWindowBytes = kGeomChunkBytes * kGeomChunks;
    constexpr unsigned kTexWindowBytes  = kGeomChunkBytes * kTexChunks;
    struct GeomChunk { char b[kGeomChunkBytes]; };

}
