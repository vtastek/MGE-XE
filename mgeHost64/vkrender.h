#pragma once

#include <cstdint>

// Minimal raw-Vulkan offscreen renderer for the present-seam spike (Milestone A).
// Renders one hardcoded triangle into a fixed-size offscreen RGBA8 (B8G8R8A8) image,
// then reads it back into a host buffer so the 32-bit client can blit it onto MW's
// backbuffer. No swapchain / WSI — strictly offscreen + readback. Milestone B will
// replace the readback with a shared external image bound to a D3D9Ex render target.
namespace VKRender {
    // Bring up instance/device/pipeline for a width*height target.
    // sharedHandle0 != null (Milestone B): import the D3D9Ex KMT shared render-target
    // handle(s) as Vulkan external memory and render directly into them — no readback.
    // sharedHandle1 != null double-buffers (Milestone C): render into buffer N&1 while
    // the consumer composites N-1. sharedHandle0 == null (Milestone A): offscreen + readback.
    // Idempotent for the same dimensions/handles; re-inits otherwise.
    bool init(std::uint32_t width, std::uint32_t height,
              void* sharedHandle0 = nullptr, void* sharedHandle1 = nullptr);

    // Render the triangle into buffer `targetIndex` (0 for A / single-buffered). On the A
    // path, copy the result into outPixels (outBytes must be width*height*4, rows tightly
    // packed). cpuMs optionally reports submit-to-fence time.
    bool renderFrame(void* outPixels, std::uint32_t outBytes, std::uint32_t targetIndex, double* cpuMs = nullptr);

    void shutdown();
    bool isReady();
}
