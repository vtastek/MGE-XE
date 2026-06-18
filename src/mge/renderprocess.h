#pragma once

#include "proxydx/d3d9header.h"

namespace IPC { class Client; }

// Present-seam spike (client side). Coordinates the out-of-process 64-bit Vulkan
// renderer: maps the host's framebuffer, drives the per-frame RenderFrame RPC, and
// composites the result as a non-destructive corner quad on MW's backbuffer just
// before Present. All gated by Configuration.UseRenderProcess; the per-frame blit is
// additionally toggled live with F11. Off = the game frame is untouched.
namespace RenderProcess {
    // Called once after the IPC host is up (DistantLand::initDistantStaticsClient).
    // No-op unless Configuration.UseRenderProcess. Issues RenderInit and maps the
    // shared framebuffer.
    void init(IPC::Client* client);

    // Called from the present hook (mged3d8device.cpp), right after ImGuiWater::onPresent.
    void onPresent(IDirect3DDevice9* device);

    void shutdown();
}
