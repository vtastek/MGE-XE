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

    // Drive the host + composite the Forge layer into MW's frame. Called from EndScene at the
    // end of scene 0 (after renderStageBlend, before scene 1) so Forge's opaque world lands
    // BEHIND s1's sorted-alpha + first-person and post-process works on the whole composite.
    // (Was onPresent — moved earlier so the opaque layer composites in scene order.)
    //
    // FUSED form (UseAsyncHostFrame=0): kickoff+finish back-to-back — the exact pre-split
    // serial behaviour for A/B.
    void onStage0Composite(IDirect3DDevice9* device);

    // Async-frame split (UseAsyncHostFrame=1). Kickoff: flush geom/tex, build the draw
    // lists + frame params, and START the host RenderFrame without waiting — the host
    // renders frame N while MW's own frame-N work continues. Finish: wait for the host
    // (GPU-complete fence), copy the shared RT, composite over MW's backbuffer. The two
    // must be called as a pair within one scene 0; between them NO IPC may run on either
    // channel (the client refuses it loudly — see IPC::Client::renderSceneKickoff). Every
    // kickoff early-out (seam down, F11 off, no scene data) makes the paired Finish a
    // no-op, so the pair degrades to exactly the old whole-function early-outs.
    void onStage0CompositeKickoff(IDirect3DDevice9* device);
    void onStage0CompositeFinish(IDirect3DDevice9* device);

    // True while a kicked-off host RenderFrame awaits its Finish (the async window is
    // open). The EndScene call site uses this to skip its late kickoff when the Phase 2
    // early kickoff (BeginScene(0), gated by DistantLand::earlyForgeKickoff) already
    // fired this frame.
    bool kickoffPending();

    void shutdown();

    // --- M1b: opaque-geometry capture (32-bit cache -> 64-bit Forge host) ---------
    // Cheap gate the cache checks before building capture buffers: true only when the
    // seam is live and wants static opaque geometry. Avoids any cost when off.
    bool wantsGeometryCapture();

    // SK1 sky takeover: true when the Forge seam is live AND compositing AND the Forge sky pass
    // is toggled ON (F7). Gates the cache's skyRoot walk + the per-frame sky draw list. Default
    // OFF → MW's own sky is untouched (clean A/B). Independent of wantsGeometryCapture so the sky
    // walk only runs when the host will actually draw it.
    bool wantsSkyCapture();

    // WT1 Forge water takeover: true when the seam is live AND compositing (F11) AND the Forge water
    // pass is toggled ON (F7). No geometry capture (the host generates the geo-clipmap mesh); this
    // gates the per-frame water-params crossing and, later (WT3), suppression of MGE's own water +
    // reflection passes. Default OFF → MGE water draws (clean A/B).
    bool wantsWaterCapture();

    // True when the Forge seam is live AND compositing (F11 on): the host renders the opaque
    // world and the full-screen composite overwrites MW's frame at present. While true, the
    // engine's own scene-0 covered-opaque draw is redundant (overwritten) — DistantLand
    // suppresses it so we don't pay for double rendering. MWSE does NOT consume the API v5
    // no-op signal, so this in-MGE suppression is the lever. Off = engine draws scene 0 normally.
    bool ownsOpaqueWorld();

    // True when the Forge seam is live AND compositing (F11 on): the host draws the exterior
    // distant land + distant statics into the same Forge frame, which the full-screen composite
    // lays over MW. While true, MW's own main-view DL color (renderDistantLand / renderDistantStatics
    // in renderStage0) is redundant — the composite overwrites exactly those pixels — so DistantLand
    // skips those COLOR draws. The depth pre-pass and the statics cull are kept (MW effects still
    // sample the distant depth). Exterior-only at the call site (Forge DL is exterior-only). Off =
    // MW draws its own DL normally, giving a clean F11 A/B and a safe fallback if Forge DL has a gap.
    bool ownsDistantLand();

    // Called from the cache upload path (scenegraph_geometry_cache.cpp) for each
    // non-skinned opaque part when its model-space geometry is (re)built. Assigns the
    // part a dense host slot (keyed on the cache key), packs pos+normal+indices into a
    // pending blob, and ships it to the host on the next onPresent flush. verts/indices
    // are model-space; world transform + camera arrive per-frame in M1c.
    //
    // `modelId` = the part's NI GeometryData pointer (object identity). The dedup is on
    // (modelId, vertexCount, revision), NOT revision alone: the cache key is an NiTriShape*
    // that Morrowind RECYCLES across cell transitions, so a new object can reuse a freed
    // object's key+slot. Static meshes share revisionID (often 0), so revision-only dedup
    // skipped the re-upload and the slot kept the previous mesh ("barrel for head"). The
    // GeometryData ptr differs on reuse → forces the re-upload. Morphed parts keep their
    // data ptr but bump revisionID, so they still refresh.
    // `forceReupload` (SK1 sky) bypasses the (modelId, vertexCount, revision) dedup so a part whose
    // vertex data changes WITHOUT a revisionID bump (the sky dome's per-frame gradient) re-ships
    // every frame. Default false keeps every existing caller's dedup behaviour.
    void captureGeometry(std::uint32_t key, std::uint16_t revision, std::uint32_t modelId,
                         const IPC::GeomVertexWire* verts, std::uint32_t vertexCount,
                         const std::uint16_t* indices, std::uint32_t indexCount,
                         bool forceReupload = false);

    // M-Skinning: capture a skinned part's bind-pose VB (model-space pos/normal +
    // per-vertex weights + packed bone indices). Like captureGeometry it assigns/reuses a
    // dense host slot (shared g_keySlot) and packs the part into the geometry blob with the
    // SKINNED flag + numBones; dedup on (modelId, vertexCount, revision) — see captureGeometry
    // for why identity (not revision alone) is required. The per-frame bone palette is
    // shipped separately (built in onPresent from the cache entry).
    void captureSkinnedGeometry(std::uint32_t key, std::uint16_t revision, std::uint32_t modelId,
                                const IPC::SkinnedVertexWire* verts, std::uint32_t vertexCount,
                                const std::uint16_t* indices, std::uint32_t indexCount,
                                std::uint32_t numBones);

    // Tier 4 multi-map: capture a STATIC opaque part carrying dark/detail/glow sibling maps,
    // with up to 4 per-vertex UV sets (GeomVertexWireMM). Like captureGeometry it assigns/reuses
    // a dense host slot (shared g_keySlot) and packs the part into the geometry blob with the
    // MULTIMAP flag; dedup on (modelId, vertexCount, revision). The per-frame ordered stage list
    // (texture slots + ops + UV sets) is built separately in buildMultiMapDrawList (onPresent).
    void captureMultiMapGeometry(std::uint32_t key, std::uint16_t revision, std::uint32_t modelId,
                                 const IPC::GeomVertexWireMM* verts, std::uint32_t vertexCount,
                                 const std::uint16_t* indices, std::uint32_t indexCount);
}
