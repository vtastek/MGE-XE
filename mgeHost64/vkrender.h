#pragma once

#include <cstdint>

// Minimal raw-Vulkan offscreen renderer for the present-seam spike (Milestone A).
// Renders one hardcoded triangle into a fixed-size offscreen RGBA8 (B8G8R8A8) image,
// then reads it back into a host buffer so the 32-bit client can blit it onto MW's
// backbuffer. No swapchain / WSI — strictly offscreen + readback. Milestone B will
// replace the readback with a shared external image bound to a D3D9Ex render target.
namespace VKRender {
    // Bring up instance/device/offscreen image/pipeline for a width*height target.
    // Idempotent for the same dimensions; re-inits if the size changes. Returns false
    // on any failure (the spike then simply produces no frame; the game is unaffected).
    bool init(std::uint32_t width, std::uint32_t height);

    // Render the triangle and copy the result into outPixels. outBytes must be
    // width*height*4. Rows are tightly packed (stride = width*4). Optionally reports
    // the CPU submit-to-readback time in milliseconds via cpuMs.
    bool renderFrame(void* outPixels, std::uint32_t outBytes, double* cpuMs = nullptr);

    void shutdown();
    bool isReady();
}
