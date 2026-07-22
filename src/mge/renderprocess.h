#pragma once

#include "proxydx/d3d9header.h"
#include "ipc/geomwire.h"

#include <cstdint>

namespace IPC { class Client; }
struct RenderedState;
struct FragmentState;

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

    // Frame-start kick (async OVERLAP mode + early-kickoff frames only): dispatch the produce the
    // instant frameSetupEarly has built its inputs, instead of after frameSetupEarly returns. The
    // paired finish still runs from onStage0CompositeKickoff moments later, under the build.
    // No-op on every other path — those kick from the dispatcher as before.
    // In produce mode 3 (PARK) the dispatched job is BUILD-ONLY: it parks the payload
    // client-private and the next frame's fireParked ships it.
    void kickProduceEarly(IDirect3DDevice9* device);

    // Produce mode 3 "PARK-AND-FIRE" fire point. Called from frameSetupEarly at the START of the
    // frame — right after the camera read + grass cull, before classify/walk/build — to ship the
    // payload the worker parked LAST frame, restamped with THIS frame's camera. The host then
    // owns essentially the whole client frame (frame = max(client CPU, host wall), not the sum).
    // No-op outside mode 3 / frame-ahead / early-kickoff frames; drops the park (one repeated
    // composite frame) on a cell-epoch change between build and fire.
    void fireParked(IDirect3DDevice9* device);

    // Produce-worker OVERLAP mode (NUMPAD8 -> 2): drains the async produce kicked at BeginScene(0)
    // so it completes within the quiescent scene-0 window (before the finish reads g_kick and before
    // mwstart(N+1) mutates the live scene graph the worker read). No-op in the OFF/FENCED modes and
    // whenever no async produce is in flight — safe to call unconditionally at the EndScene(0)
    // finish boundary, which is the ONLY place it must run.
    void waitProduce();

    // True while a kicked-off host RenderFrame awaits its Finish (the async window is
    // open). The EndScene call site uses this to skip its late kickoff when the Phase 2
    // early kickoff (BeginScene(0), gated by DistantLand::earlyForgeKickoff) already
    // fired this frame.
    bool kickoffPending();

    // Frame-ahead pipelining (ForgeFrameAhead ini, numpad-* live A/B). On early-kickoff
    // frames the kickoff marks its finish DEFERRED: the EndScene(0) composite point only
    // blits the PREVIOUS host frame (onFrameAheadBlit — zero IPC, zero wait), and the
    // finish + RT copy run at the NEXT frame's BeginScene(0) (onFrameAheadCollect —
    // before frameSetupEarly, so the IPC window closes before any of the new frame's
    // RPCs need the channel). The host D3D12 frame thus overlaps the whole MW frame;
    // the composited world lags input by one frame (UI stays current). Late/warm-up
    // frames keep the same-frame finish, which also primes g_mainTex before the first
    // deferred blit. Collect additionally owns the once-per-frame dev-key poll (deduped
    // against the finish via onFramePresented's frame serial), so call it on EVERY
    // scene-0 frame, not just deferred ones.
    void onFrameAheadCollect(IDirect3DDevice9* device);
    void onFrameAheadBlit(IDirect3DDevice9* device);

    // Phase 0 (deferred wait): the previous frame's deferred finish no longer runs inside
    // onFrameAheadCollect. On NON-early / not-ready frames the caller invokes this after
    // frameSetupEarly has latched earlyForgeKickoff — it finishes+copies the pending frame at
    // BeginScene, before the MGE pipeline's scene-0 IPC needs the channel. On EARLY-kickoff
    // frames the caller skips this; the finish moves into onStage0CompositeKickoff (run late,
    // just before the kickoff reuses the channel, so the host had the whole BeginScene window to
    // finish and the exposed wait collapses to ~0). No-op unless a deferred finish is pending.
    void collectDeferredFinish(IDirect3DDevice9* device);

    // True when the current kickoff deferred its finish to the next frame's collect:
    // the EndScene composite point must blit (onFrameAheadBlit) instead of finishing.
    bool finishDeferred();

    // Display-frame tick, called from the proxy Present's scene-reset block. Drives the
    // once-per-frame dedup of the dev-key poll across the collect/finish call sites.
    void onFramePresented();

    // A0 Cut-4 probe: called right after the engine's Present returns. The span from
    // here to the next BeginScene(0) is MW's un-zoned frame start (input/sim/AI/
    // animation); onFrameAheadCollect closes it and the [hb] heartbeat reports the
    // average as mwstart=. Its magnitude decides whether a frame-start cut exists.
    void noteEnginePresentReturn();

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

    // FP1a first-person takeover: true when the seam is live AND compositing AND the Forge FP
    // pass is enabled (ForgeFPPass ini) AND the player is in FIRST person. Gates the cache's
    // armCamera-root walk + the per-frame FP draw lists + the FP camera crossing. Default off
    // (ini) → MW's own first-person rendering is untouched.
    bool wantsFPCapture();

    // FP1b: true when the host FP pass is live (wantsFPCapture + camera math validated) AND
    // suppression is on (ForgeFPSuppress ini, flipped live with numpad-/). The cache walk
    // applies it by force-culling MW's arm-scene root each frame (restored once on release),
    // so MW's own first-person draws no-op while the host draws the arms.
    bool wantsFPSuppression();

    // FP camera ground-truth diagnostic (offset-from-center triage): the proxy arms this at
    // the z-only clear MW issues before its first-person scene, then forwards the next
    // view/proj it SUBMITS (post camEffects/proj-edit — exactly what rasterizes the native
    // arms). buildFPFrame diffs the latched pair against the built fpView/fpProj and logs.
    void noteFPZClear();
    void noteFPSceneTransform(bool isProj, const D3DMATRIX* m);

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
    // `uvAnim`/`uvAnimBytes` (NiUVController takeover): optional GeomUVAnimWire key-track blob
    // appended after the part's indices (flags |= kGeomFlagUVAnim); default null = no payload.
    void captureGeometry(std::uint32_t key, std::uint16_t revision, std::uint32_t modelId,
                         const IPC::GeomVertexWire* verts, std::uint32_t vertexCount,
                         const std::uint16_t* indices, std::uint32_t indexCount,
                         bool forceReupload = false,
                         const std::uint8_t* uvAnim = nullptr, std::uint16_t uvAnimBytes = 0);

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
                                 const std::uint16_t* indices, std::uint32_t indexCount,
                                 const std::uint8_t* uvAnim = nullptr, std::uint16_t uvAnimBytes = 0);

    // AT3 captured-alpha: called from the ForgeAlphaSuppressS1 reject gate
    // (distantland.cpp inspectIndexedPrimitive) for each alpha-BLENDED DIP MW is about to skip
    // in scenes >= 1. MW already billboarded + sorted it; we Lock the live VB/IB, copy the final
    // verts + rebased indices into pending scratch, and record the draw metadata so the next
    // buildGeometryDrawLists ships the geometry and draws it in the Forge post-water sorted-alpha
    // pass — restoring NiParticles (smoke/flames) + multimap/decal/untextured blends the host
    // cache pass doesn't own. Silent no-op on any guard miss (flag off, HW-skinned, non-tri-list,
    // caps, dedup); reject at the gate is unchanged whether or not this captures.
    void captureAlphaDraw(const RenderedState* rs, const FragmentState* frs);

    // Upload cost accounting (Part A, upload-debug). The geometry cache calls noteUpload()
    // once per host geom reship, tagged by cause so the [uploads] heartbeat line can break the
    // per-frame cost down and the host panel can show total KB/parts per frame. Categories are
    // derived for free from the content-gate's changed-component diff + isSkinned — no extra NIF
    // walking. `bytes` = the model-space wire payload shipped (VB verts + IB indices).
    enum UploadCat : std::uint8_t {
        kUpNew = 0,    // first upload of a part (not a reship)
        kUpSkin,       // reship on a skinned actor (NPC/creature) — expected
        kUpMorph,      // non-skinned changed=pos (morphing statics + particle-system regen)
        kUpVcol,       // changed=vcol
        kUpTopo,       // vertex/tri/uvset count change → full re-alloc
        kUpUvLeak,     // changed=uv after the UVController takeover (should be ~0)
        kUpOther,      // anything else (normals, etc.)
        kUpCount
    };
    void noteUpload(std::uint8_t cat, std::uint32_t bytes);
}
