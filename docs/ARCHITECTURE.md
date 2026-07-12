# Forge architecture

Master design document for the "Forge" out-of-process renderer on the `forge` branch.
Describes how the system is designed and why; for where the work currently stands see
[dev/STATUS.md](dev/STATUS.md), for known problems see
[dev/SHORTCOMINGS.md](dev/SHORTCOMINGS.md), and for the best-practice end-state we
steer toward see [TARGET-ARCHITECTURE.md](TARGET-ARCHITECTURE.md).

Line references in this document are relative to tag `v0.19.1-rc4` (commit `8d2508f`)
and will drift.

## 1. What Forge is

Fork of Hrnchamd/MGE-XE. Lineage: descawed's `distantland64` shared-memory IPC work
(32-bit Morrowind process ↔ 64-bit host process) was merged into `master`, then the
`forge` branch built an out-of-process 64-bit render host on top: `mgeHost64.exe`,
using a vendored ConfettiFX **The-Forge** framework on the **D3D12** backend
(Agility SDK 715), running headless. Shaders are written in FSL
(`mgeHost64/shaders/FSL/*.fsl`) and transpiled to `shaders/gen/DIRECT3D12/`. An older
Vulkan spike exists (`mgeHost64/vkrender.cpp`, ~600 lines); the live path is
Forge/D3D12 (`mgeHost64/forgerender.cpp`, ~13.7k lines).

Morrowind itself runs 32-bit on DXVK (`d3d9_dxvk.dll`, shipped and Forge-on-by-default
since v0.19.1-rc2). MGE XE in-process is demoted to a *feeder*: when Forge ownership
gates are on, it does not render the opaque world, sky, water, or distant land
(`src/mge/distantland.cpp` gates on `RenderProcess::ownsOpaqueWorld()` /
`wantsSkyCapture()` / `wantsWaterCapture()`).

## 2. Process and IPC architecture

Two shared-memory channels: a main RPC channel and a bulk geometry/texture channel
(`src/ipc/server.cpp` dispatch; `GeomUpload`/`TexUpload` ride the geometry channel).
Wire structs are 32/64-safe via `ptr32`/`ptr64` templates and `#pragma pack(4)`
(`src/ipc/bridge.h`).

Geometry wire format (`src/ipc/geomwire.h`): flat blobs of
`[GeomPartWire][verts][uint16 indices]`, slot-indexed — the client assigns each cached
`NiTriShape*` a dense slot; the host stores meshes in a flat array by slot, no
host-side hashing. Vertex formats: 36-byte static (pos/normal/uv/packed color),
52-byte skinned (top-4 bone influences), 60-byte multi-map (`GeomVertexWireMM`, four
UV sets) for dark/detail/glow siblings. (Nearby comments claiming skinned strides of
56/44 are stale; the struct is 52 bytes.) Vertex color is written only for `DiffAmb`
material routing; otherwise white, so the host runs a single shading path.

On the Forge path, distant land and distant statics are **host-owned** — the host
loads the DL files from disk itself (the "Phase 1a/1b LIVE distant land" blocks in
`forgerender.cpp`), so near cache geometry is the only mesh data crossing the wire.
The IPC layer additionally still carries the **legacy distantland64 service**
inherited from the merged lineage (`src/ipc/dlshare.cpp`): host-side
placement/quadtree metadata plus cull queries answered with client-owned D3D9 buffer
handles (`ptr32<IDirect3DVertexBuffer9>` in `RenderMesh` records, `bridge.h:37`).
That path predates Forge ownership; which parts remain live when Forge is on has not
been established.

## 3. Client frame (Morrowind process)

- Two walkers cover the NI scene graph — `WorldObjectRoot`, `WorldPickObjectRoot`,
  plus loose `worldRoot` siblings (projectiles etc.). A **worker thread**
  (`src/mge/scenegraph.cpp`) walks it with a byte-compare/revision scheme to feed the
  **light snapshot**. The **geometry cache** walk
  (`scenegraph_geometry_cache.cpp`, per-shape `revisionID` + transform compare) runs
  **synchronously on the main thread**, driven from `DistantLand::renderStage0` and
  the depth pass — so the geometry feeding cost sits on the main thread.
  [`SCENEGRAPH.md`](../SCENEGRAPH.md) documents the engine scene-graph structure and
  the DataHandler offsets.
- At end of Morrowind's scene 0, `RenderProcess::onStage0CompositeKickoff`
  (`src/mge/renderprocess.cpp:2251`): cell-change detection (interior-cell pointer
  change OR single-frame eye teleport beyond `kCellTeleportDist`) bumps an epoch that
  evicts light identities; then `buildGeometryDrawLists` re-walks the cache and packs
  per-frame draw lists into scratch blobs (`kCellTeleportDist` = 8192 units — exactly
  one exterior cell). Lanes: static opaque, skinned, multi-map,
  alpha, captured-alpha (MW particle/flame draws intercepted at `DistantLand`'s
  `captureAlphaDraw`), sky shapes, lights, and a WIP first-person frame.
- First-sight meshes are captured lazily *during* the build, so the geometry flush
  runs after the build, then lazy first-seen texture flush, then `renderSceneKickoff`
  starts the host RenderFrame **without waiting**.
- **Async split:** the host renders frame N while MW does its own frame-N work;
  `onStage0CompositeFinish` (`renderprocess.cpp:2695`) waits for GPU-complete, copies,
  composites. A fused serial mode exists for A/B. Between kickoff and finish, the
  geometry channel refuses IPC (enforced loudly). Dev-key toggles (F7/F8/F9/F11/F12)
  are polled at finish so every per-frame ownership gate sees one value per frame —
  toggles are one-frame latched by design.

## 4. Host frame (`renderScene`, `mgeHost64/forgerender.cpp:5694`)

Pass order, from the section banners in the code:

1. **GPU distant-statics cull** — count → prefix-sum → scatter compute chain
   (`cull.comp` family), one thread per instance, survivors written camera-relative,
   drawn via a handful of `cmdExecuteIndirect` calls per (mirror, batch) group.
2. **Shadow-light occlusion cull** — candidate shadow lights tested against the Hi-Z
   pyramid before atlas slots are spent (rc4 addition).
3. **Z-prepass** (depth-only): static opaques (PS-less fast path when alphaRef == 0;
   alpha-tested variant otherwise), then skinned (fills bone windows), then multi-map.
4. **Point-light shadow faces** — 32-slot cube-face atlas (`kMaxShadowLights = 32`,
   must match `MAX_SHADOW_SLOTS`). Two-layer composite: cached STATIC atlas (tiles
   re-rendered only on per-slot invalidation; "never evicts and never re-renders on
   camera motion") + DYNAMIC mover atlas per frame; movers classified by source.
   Reverse-Z into the atlas; light positions camera-relative.
5. **Linearize depth → GTAO → bilateral blur** (compute). AO carries world-space bent
   normal in RGB, visibility in A.
6. **Shadow mask compute** (`shadowmask.comp.fsl`) — per pixel: reconstruct
   camera-relative world position from depth, analytic cube-face atlas test per active
   slot, manual bilinear PCF (Forge has no comparison samplers; binary comparisons are
   bilerped), 4-bit visibility per light packed into `R32G32B32A32_UINT`, capped at
   the K=8 strongest slots per pixel. Hot-reloadable (F8 / dxil mtime poll).
7. **Reflection pass** — 1024², for water: sky first, then reflected land + statics
   drawn over it (WV2 stage, default on via `g_drawReflectGeo`).
8. **Colour pass**, early-Z `CMP_EQUAL`, no depth write: sky first (blended into the
   cleared target; opaque replace-blend overwrites it), then live distant land,
   indirect distant statics, near static/skinned/multi-map. Near static opaques are
   drawn via CPU-filled `cmdExecuteIndirect` args in both prepass and colour pass
   (one per mirror/alpha-test/batch group); skinned, multi-map, sky, alpha, water,
   FP, and shadow-caster lanes are recorded per-draw. Forward shading,
   deliberately FFE-faithful: ports `XE FixedFuncEmu.fx` per-pixel lighting and
   tonemap verbatim — sun N·L + scene ambient + brute-force per-pixel loop over the
   light cbuffer (no clustering), FFE `vColSource` material routing, AO modulates
   ambient only, optional bent-normal substitution.
9. **Hi-Z mip-0 fill** from scene depth, mid-pass, with a barrier dance around the
   colour pass.
10. **Water** — drawn last so the completed opaque frame is copied out as its
    refraction source; geo-clipmap surface + the reflection RT.
11. **Sorted-alpha pass** (captured MW alpha draws), then **first-person pass**
    (FP1a/1b, explicitly WIP; MW arm suppression toggleable live via numpad-/).
12. **Hi-Z prologue tail submit** — mips 1..N reduced on a separate command buffer
    after the main frame, consumed by *next* frame's statics cull and shadow-light
    cull. GPU occlusion is one frame stale by construction. The client-side MSOC
    plugin is additionally integrated into the distant-statics cull.

### Conventions (hold everywhere)

- **Camera-relative world matrices** — translation shifted by −eye everywhere,
  including lights and bone translations — with translation-free viewProj.
- **Reverse-Z** with `CMP_GEQUAL`.
- **Premultiplied-alpha** output for the composite.
- MSAA is supported host-side by rendering into an MSAA colour/depth pair and
  resolving into the shared RT (the shared RT cannot be MSAA).

## 5. Composite seam back to Morrowind

Host renders into a cross-process shared D3D12 RT (B8G8R8A8, shared heap,
`D3D12_TEXTURE_LAYOUT_UNKNOWN`), exports an NT handle. The client imports that handle
into DXVK's Vulkan device as external memory, `vkCmdCopyImage`s into a DXVK-tracked
destination image under DXVK's submission-queue lock with a client-side
`WaitForFences`, then composites that image as a premultiplied-alpha quad over MW's
backbuffer at end of scene 0 — so MW's scene-1 sorted alpha, first person, and MGE
post-process all run on top.

**Doc drift:** the `src/mge/renderprocess.h` header comment (:20-21) and
`distantinit.cpp:551` still describe a "9On12 side-device" route. Audited 2026-07-13:
the route is **fully removed** — no `D3D9On12` code or fallback exists anywhere, and
`renderprocess.cpp:273` explicitly negates it ("no native d3d9, no D3D9On12, no
cross-backend share"). The stale comments are cleanup candidates, nothing more.

## 6. Navigation pointers

- Key files: `mgeHost64/forgerender.cpp` (host; `renderScene` at :5694),
  `mgeHost64/main.cpp` (standalone probes: `--forge-probe`, `--forge-render`,
  `--forge-dl`, `--forge-statics`, `--forge-view` — useful for isolating host from
  integration), `src/mge/renderprocess.cpp` (client seam; kickoff :2251, finish
  :2695), `src/ipc/` (bridge/geomwire/server), `src/mge/scenegraph_geometry_cache.cpp`
  (capture), `SCENEGRAPH.md` (engine scene-graph reference).
- Shaders: `mgeHost64/shaders/FSL/` (source of truth; hot-reloadable compute via F8),
  `shaders.list` for the build set.
- Dev keys at runtime: F11 composite toggle, F12 debug-view cycle (AO, bent normal,
  albedo, lit, ambient, world normal, light count, shadow mask, shadow atlases…),
  F9 host overlay, F8 compute hot-reload, F7 water A/B, numpad-/ FP arm suppression.
- Branches of note: `forge` (live), `master` (fork base incl. distantland64 merge),
  plus historical spikes (`hdrthreaded`, `scene-walk-v2`, `motionblur`).
- Tracy zones (`MGE_ZoneScopedN`) exist throughout the **client** (`src/mge`, ~82
  sites). The host has no Tracy instrumentation — it uses its own GPU timestamp
  phase timers (`kGpuPhase*`) surfaced by the F9 overlay.

### A note on plan codenames

Code comments and commit messages reference plan documents in a `tasks/` folder
(`tasks/todo.md`, `tasks/forge-phase1.md`, `tasks/forge-depth-ssao.md`,
`tasks/async-frame-split.md`, `tasks/forge-gpu-occlusion.md`, …) that was **never
committed** to the repo. The codenames survive in history: Phase 1a/1b = host-owned
distant land/statics, Phase 2 = bindless texturing (host) *and separately* the
planned GPU-resident scene (client), Phase 3 = Hi-Z occlusion prologue, W* = walk
cost cuts, AT* = alpha takeover, C*/P* = point-light shadows, SK* = sky, FP* =
first-person takeover. Where a "Phase" number appears, check which subsystem's plan
it belongs to — the numbering is per-plan, not global.
