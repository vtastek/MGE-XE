# Forge shortcomings

**As of:** 2026-07-13, `forge` @ `8d2508f` (tag `v0.19.1-rc4`).
**Audited:** 2026-07-13 — every claim below was verified against the code by a
four-way claim audit; corrections from that audit are noted in place.

Prioritized list of known structural problems, ordered roughly by structural weight.
Each item carries a verification tag: **[verified]** (confirmed in code) or
**[inferred]** (judgment from the architecture, not confirmed by code, comments, or
measurement). Nothing here is measured — no profiling has been run. Line references
are relative to the commit above.

When an item is fixed, move it to *Resolved* at the bottom with the fixing commit —
don't delete it. Items retired by audit stay in place, marked. Each structural item
maps to a target in [../TARGET-ARCHITECTURE.md](../TARGET-ARCHITECTURE.md): items
1-2 → T6, 4 → T4/T5/T6, 5 → T1, 6 → T2, 7-8 → T7, 9 → cleanup, 10 → T3, 12 → T1,
13-14 → T4/T5, 15-16 → T8, 17 → in-flight FP work.

## Scaling and memory

1. **No distance/LRU geometry eviction.** On arena-full the arena *grows by
   doubling* (`growArenaBuffer`: new buffer, GPU copy-in-place, offsets preserved)
   up to a hard cap; past the cap parts are **dropped** (logged as e.g. "geomUpload
   built 973/1693"). A free-list does reclaim ranges on re-upload/shape-change and
   release sentinels — but nothing evicts by distance or age. Comment at
   `mgeHost64/forgerender.cpp:1353`: "true eviction across a 40000-cell world needs
   a client free-slot signal (follow-up)". Long sessions crossing many cells push
   toward the cap and then silently lose geometry. [verified]
2. **Texture eviction is re-upload-only.** Fixed bindless array
   (`MAX_TEXTURES = 896`, `opaque.srt.h:18`), lazy first-seen upload. A release path
   *does* exist — re-uploading a slot frees the prior texture
   (`forgerender.cpp:9955`, `:10232`) — but there is no LRU/camera-driven eviction,
   so the number of *distinct* textures per session is bounded by the fixed array.
   [verified; audit corrected the earlier "no release path" claim]
3. ~~**uint16 indices per wire part.**~~ **Retired by audit — non-issue by
   construction.** No >64k guard exists on the main capture path, but index values
   come straight from NI's native `uint16` triangle lists, so Morrowind meshes
   cannot exceed the cap. The captured-alpha path has an explicit oversize guard
   (drops whole record when rebased indices exceed 0xFFFF,
   `renderprocess.cpp:3155-3164`); a whole-part >8 MB window drop exists at flush
   (`renderprocess.cpp:1017-1020`). [verified]
4. **Hard caps:** 32 shadow slots (`kMaxShadowLights`), K=8 shadowed lights/pixel
   (`SHADOW_MAX_SLOTS_PER_PIXEL`), `kMaxSkyDraws = 64`, `kMaxMultiMap = 256`, light
   cbuffer `MAX_POINT_LIGHTS = 128`, bindless `MAX_TEXTURES = 896`, mesh-arena hard
   cap. Comment at `forgerender.cpp:4945` describes an observed shadow-slot
   "eviction fight" in dense scenes — slot pressure is already real. [verified]

## Per-frame CPU cost (the feeding tax)

5. **Full draw-list rebuild every frame, on the client's main thread.**
   `buildGeometryDrawLists` re-walks the cache and repacks all lanes each frame, and
   the geometry-cache walk itself is synchronous main-thread work (only the light
   snapshot walk is on a worker thread). The code calls this "the MGE→Forge feeding
   cost — Phase 2 makes this GPU-resident so it goes to 0"
   (`src/mge/renderprocess.cpp:2317`) — acknowledged, unimplemented. Likely the
   dominant steady-state CPU overhead. [verified design; "dominant" is inferred,
   unmeasured]
6. **Per-draw CPU recording for the non-static lanes.** Near static opaques *are*
   indirect — CPU-filled `cmdExecuteIndirect` args in both prepass and colour pass —
   but shadow-face casters (the "camera-shaped + single-buffered, so indirect here
   would fight the main pass" comment at `forgerender.cpp:7332` belongs to *this*
   pass), skinned, multi-map, sky, alpha, water, and FP lanes are recorded per-draw.
   Recording cost scales with scene complexity in those lanes — most acutely
   shadow-caster re-bakes. [verified; audit corrected the earlier "near-scene draws
   are not indirect" claim, which was false]
7. **Synchronous finish chain:** fence wait → Vulkan copy → `WaitForFences`, under
   DXVK's queue lock. If the host frame outlasts MW's remaining frame work, MW
   stalls; the copy is a hard sync point every frame. [structure verified; impact
   unmeasured]

## Bandwidth / seam

8. **Per-frame copy chain:** host colour target → (MSAA resolve) → shared D3D12 RT →
   Vulkan-imported copy → DXVK image → blended quad. Two-plus full-res copies per
   frame; meaningful at high resolution. [structure verified; cost inferred]
9. **Stale 9On12 seam comments** (`src/mge/renderprocess.h:20-21`,
   `distantinit.cpp:551`). Audit settled the open question: the route is **fully
   removed** — no code or fallback remains, and `renderprocess.cpp:273` explicitly
   negates it. Pure comment cleanup. [verified]

## Correctness / visual

10. **One-frame-stale occlusion.** Hi-Z tail submit feeds next frame's culls; camera
    cuts and load doors will pop. rc4's temporal fade-in for shadow pops is a symptom
    patch, not a fix. [verified design; pop severity inferred]
11. **Two alpha layers in two processes.** Host draws captured/sorted alpha; MW's
    scene-1 alpha composites unconditionally on top. Host alpha can never sort in
    front of MW alpha. [inferred from composite ordering; needs an in-game
    counterexample to confirm severity]
12. **Cell-change heuristic edge cases.** `kCellTeleportDist = 8192` — exactly one
    exterior cell per frame (`renderprocess.cpp:198`); trigger = interior-cell
    pointer change OR single-frame displacement > 8192 units. Audit confirmed the
    code guards neither direction beyond that: **false positives** from any >8192
    single-frame move — long falls, levitate, scripted `SetPos`/`PositionCell` —
    cause spurious light-identity eviction; **false negatives** when a transition
    moves the eye < 8192 units without changing the interior pointer (stale light
    identities). Consequence is bounded: only `g_lightTracks` is evicted / epoch
    bumped — geometry is unaffected. [verified; consequences bounded per code]
13. **Brute-force per-pixel light loop**, no clustering/tiling. Bounded only by the
    cbuffer cap (`MAX_POINT_LIGHTS = 128`) and the shadow mask's K=8 cap. Fine at
    vanilla light counts; dense modded interiors scale badly. [verified]
14. **Shadow-mask cost scales with in-range slots:** (2R+1)² × 4 atlas loads per
    in-range slot per pixel at full res, plus a 128-bit/pixel mask. Worst case = many
    overlapping lights near the camera. [formula verified; impact inferred]

## Robustness

15. **Device-removed is terminal.** Entry guard at `forgerender.cpp:5711-5716`
    returns false every subsequent frame ("every subsequent frame stays black …
    explains 'back to interior, still black'"). No restart/reattach path. [verified]
16. **No recovery from host loss; wedged host stalls up to 60 s.** Audit refined
    this: host *death* IS detected immediately — the host process handle sits in
    every client wait set, so a crashed host returns `WakeReason::ServerLost` at
    once (`src/ipc/client.cpp:730`, `:775`), and the frame is skipped. But there is
    no restart-after-loss, no watchdog, no runtime `isServerActive()` polling —
    after `ServerLost` the game continues without Forge, silently skipping frames'
    host work. A *wedged-but-alive* host is undetected and stalls the game up to
    `IPC::MaxWait = 60000` ms (`src/ipc/bridge.h:74`) per blocking wait. Teardown
    exists only at process exit (`~Client` → `TerminateProcess`). [verified]
17. **First-person takeover half-done.** Shipped default mixes MW-rendered first
    person over the Forge world with a separate camera oracle; projection/fog
    mismatches likely until FP1 lands. FP1c (torch flame / enchant glow) is reserved
    in the wire format but unimplemented — the client explicitly skips blend-enabled
    FP entries (`renderprocess.cpp:2169`). [WIP status verified; mismatch prediction
    inferred]

## Resolved

*(nothing yet — item 3 above was retired as a non-issue by the 2026-07-13 audit, not
fixed)*
