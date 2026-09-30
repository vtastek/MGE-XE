#pragma once

// Dependency-free on purpose: renderprocess.cpp reads this ledger and must not pull in the D3D8
// proxy's header stack to do it. Written by ProxyDevice::CreateTexture / ProxyTexture::Release.

#include <atomic>
#include <cstdint>

// MW-SIDE TEXTURE LEDGER (tasks/forge-memory-shape.md, alarm 4). Every texture Morrowind creates
// through this proxy in a GPU pool, by live bytes and size class. Morrowind.exe was measured holding
// 2355 MB of VRAM with no counter anywhere that could say of what; this is the meter on its own
// textures. Live = created minus released-to-zero (ProxyTexture::Release). Bytes are computed from
// the format and the real level count, i.e. the GPU footprint, not the file size.
struct ProxyTexLedger {
    std::atomic<uint64_t> liveBytes{0};
    std::atomic<uint32_t> liveCount{0};
    std::atomic<uint64_t> bigBytes{0};      // >= 4 MB: 2048^2 and up
    std::atomic<uint32_t> bigCount{0};
    std::atomic<uint64_t> midBytes{0};      // 1-4 MB
    std::atomic<uint32_t> midCount{0};
    std::atomic<uint64_t> createdBytes{0};  // session totals
    std::atomic<uint32_t> createdCount{0};
    // The map cap (g_proxyTexCapDim, see d3d8texture.h), session totals. savedBytes is what the
    // dropped top mips would have cost. filled counts capped textures Morrowind then wrote a KEPT
    // level of: if it lags capped, Morrowind is filling them some way the cap does not redirect
    // and those textures are blank.
    std::atomic<uint32_t> capCount{0};
    std::atomic<uint64_t> capSavedBytes{0};
    std::atomic<uint32_t> capFilled{0};
    std::atomic<uint32_t> capScratchLocks{0};   // writes to a dropped level, discarded
    std::atomic<uint32_t> capScratchSurfaces{0};// GetSurfaceLevel on a dropped level
};
ProxyTexLedger& proxyTexLedger();

// Called from ProxyTexture::Release when Morrowind drops its LAST reference to a texture, with the
// real D3D9 texture pointer (a key only — the texture is already gone). The Forge feed hangs its
// "host mirrors Morrowind" texture release on it (GeometryCache::onTextureDestroyed). Null = nobody
// listening. A plain function pointer keeps the proxy free of the mge headers.
extern void (*g_onProxyTextureDestroyed)(void* realTexture);

// Largest level-0 edge Morrowind's own mip-mapped MANAGED textures are created at; 0 = off. Armed
// at device creation, only under the Forge takeover (d3d8texture.cpp).
extern uint32_t g_proxyTexCapDim;
