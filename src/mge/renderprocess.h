#pragma once

#include "proxydx/d3d9header.h"
#include "ipc/geomwire.h"

#include <cstdint>

namespace IPC { class Client; }

// Present-seam spike (client side). Coordinates the out-of-process 64-bit Vulkan
// renderer: maps the host's framebuffer, drives the per-frame RenderFrame RPC, and
// composites the result as a non-destructive corner quad on MW's backbuffer just
// before Present. All gated by Configuration.UseRenderProcess; the per-frame blit is
// additionally toggled live with F11. Off = the game frame is untouched.
namespace RenderProcess {
    // Called once after the IPC host is up, from inside DistantLand::init() — i.e.
    // UNDER the "...MGE XE..." loading bar, before the first menu present. No-op unless
    // Configuration.UseRenderProcess. Brings up the seam synchronously here (9On12
    // side-device + RenderInit RPC + shared-RT open) so the cost hides behind the bar
    // instead of stalling the first menu frame. Needs the live device for the bring-up.
    void init(IPC::Client* client, IDirect3DDevice9* device);

    // Called from the present hook (mged3d8device.cpp), right after ImGuiWater::onPresent.
    void onPresent(IDirect3DDevice9* device);

    void shutdown();

    // --- M1b: opaque-geometry capture (32-bit cache -> 64-bit Forge host) ---------
    // Cheap gate the cache checks before building capture buffers: true only when the
    // seam is live and wants static opaque geometry. Avoids any cost when off.
    bool wantsGeometryCapture();

    // Called from the cache upload path (scenegraph_geometry_cache.cpp) for each
    // non-skinned opaque part when its model-space geometry is (re)built. Assigns the
    // part a dense host slot (keyed on the cache key), packs pos+normal+indices into a
    // pending blob, and ships it to the host on the next onPresent flush. Re-uploads
    // only when revision changes. verts/indices are model-space; world transform +
    // camera arrive per-frame in M1c.
    void captureGeometry(std::uint32_t key, std::uint16_t revision,
                         const IPC::GeomVertexWire* verts, std::uint32_t vertexCount,
                         const std::uint16_t* indices, std::uint32_t indexCount);

    // M-Skinning: capture a skinned part's bind-pose VB (model-space pos/normal +
    // per-vertex weights + packed bone indices). Like captureGeometry it assigns/reuses a
    // dense host slot (shared g_keySlot) and packs the part into the geometry blob with the
    // SKINNED flag + numBones; re-uploads only on revision change. The per-frame bone
    // palette is shipped separately (built in onPresent from the cache entry).
    void captureSkinnedGeometry(std::uint32_t key, std::uint16_t revision,
                                const IPC::SkinnedVertexWire* verts, std::uint32_t vertexCount,
                                const std::uint16_t* indices, std::uint32_t indexCount,
                                std::uint32_t numBones);
}
