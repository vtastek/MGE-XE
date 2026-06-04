#pragma once

// Shared layout for the occlusion-mask blob shipped from the 32-bit Morrowind
// process (msoc.dll builds the MOC mask) to the 64-bit host (mgeHost64 runs the
// distant-statics cull). The blob is [Header][raw MOC ZTile buffer]; the host
// reconstructs a MaskedOcclusionCulling instance from it and runs the EXACT same
// TestRect() queries the plugin runs in-process — bit-identical verdicts, no
// re-rasterization.
//
// THIS STRUCT MUST STAY BYTE-FOR-BYTE IDENTICAL to msoc-plugin's
// msoc::patch::occlusion::MaskBlobHeader (PatchOcclusionCulling.h). Both are
// fixed-width + pointer-free + pack(4) so the 32-bit producer and 64-bit
// consumer agree. Bump kVersion on any layout change and reject mismatches.

#include <cstdint>

namespace OcclusionMask {

    constexpr std::uint32_t kVersion = 1;

#pragma pack(push, 4)
    struct Header {
        std::uint32_t version;        // == kVersion; consumer rejects mismatch
        std::int32_t  impl;           // MaskedOcclusionCulling::Implementation tier
        std::int32_t  maskW, maskH;   // mask resolution
        std::uint32_t zbufBytes;      // raw ZTile buffer size (follows header)
        std::uint32_t headerBytes;    // sizeof(Header)
        float viewProj[16];           // snapshot world->clip, Intel column-major
        float ndcRadiusX, ndcRadiusY; // sphere NDC half-extent = r*ndcRadius/cw
        float wGradMag;               // near-surface clip-w offset = r*wGradMag
        float nearClipW;              // near-clip w threshold (straddle → visible)
        float depthSlack;             // world-unit depth bias used for wMin
        float minRadius;              // skip-tiny world-radius threshold (0 = none)
        std::int32_t  ready;          // 1 if snapshot fresh (else keep everything)
        std::uint64_t ageMs;          // snapshot wall-clock age
    };
#pragma pack(pop)

    // Raw ZTile buffer sits immediately after the header.
    inline const void* zbuffer(const Header* h) {
        return reinterpret_cast<const char*>(h) + h->headerBytes;
    }

    // The blob travels in a shared-memory Vec. We can't use a Vec<char> sized to
    // the blob: the Vec reserves maxSize*windowBytes address space, which for a
    // single 192KB window of byte elements is ~38GB (the server's whole-range
    // MapViewOfFile then fails). So the element is a 64KB CHUNK — the vec holds
    // just kBlobChunks of them (one window), keeping the reservation tiny while
    // staying a single contiguous region the blob memcpy's into.
    constexpr int kChunkBytes  = 65536;
    constexpr int kBlobChunks  = 3;                       // 192KB ≥ any mask blob
    constexpr int kBlobBytes   = kChunkBytes * kBlobChunks;
    struct MaskChunk { char b[kChunkBytes]; };

} // namespace OcclusionMask
