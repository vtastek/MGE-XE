# Forge development status

**As of:** 2026-07-13, `forge` @ `8d2508f` (tag `v0.19.1-rc4`, released 2026-07-12).
**Audited:** 2026-07-13 — claims verified against the code by a four-way claim
audit; several open questions were settled by it (see below).

Snapshot document — update the stamp above when revising. Design reference:
[../ARCHITECTURE.md](../ARCHITECTURE.md). Known problems:
[SHORTCOMINGS.md](SHORTCOMINGS.md). Where we're steering:
[../TARGET-ARCHITECTURE.md](../TARGET-ARCHITECTURE.md).

## Release line

| Tag | Content |
|---|---|
| v0.19.1-rc1 | GitHub Actions release workflow |
| v0.19.1-rc2 | Forge on by default + ship `d3d9_dxvk.dll` |
| v0.19.1-rc3 | FSL watcher tweaks |
| v0.19.1-rc4 | 18 commits: point-light shadow work (atlas slot management, occlusion-culled bakes, mover classification, lantern-specific behaviour, debug views), first-person takeover WIP (FP0–FP1b), showcase interiors |

## Workstream state

**Landed and stable** (per commit trail through rc4):

- Host-owned distant land + GPU-driven distant statics (Phase 1a/1b), Hi-Z occlusion
  prologue (Phase 3), bindless texturing (host Phase 2).
- Async host-frame split (host renders frame N while MW works frame N), with serial
  fused mode kept for A/B.
- Alpha takeover (AT1–AT3): captured MW blended draws render host-side after water.
- Sky takeover through SK4 (live vertex colours; host-gradient dome retired).
- Client walk-cost cuts (W1.5, W3, "dense-city serial-chain cuts"): active-cell radius
  gate, live-read at build, fused draw lists.
- Point-light shadows P1→P7 + C2→C4 + rc4 deliverables: 32-slot cached/dynamic
  cube-face atlas, screen-space shadow mask with per-pixel K=8 cap, skinned casters,
  translucency-aware casters, occlusion-culled bakes, slot reclaim with temporal
  fade-in, lantern polish (soft shadows, per-texel emissive carve, flicker
  classification).
- Actor fidelity extras: lip-sync/blink head morphs driven from
  `AnimationData::headMorphTiming`.

**In flight (WIP in the shipped build):**

- **First-person takeover** — FP0/FP1a/FP1b landed (host FP arm pass, MW arm
  suppression via numpad-/, per-scene camera oracle, interior live lighting).
  FP1c is reserved in the wire format but unimplemented (torch flame / enchant glow;
  `src/ipc/bridge.h:402`, `alphaList` at :295) — audit confirmed the client
  explicitly skips blend-enabled FP entries (`renderprocess.cpp:2169`). No FP2
  references exist yet.
- **Interiors** — "showcase interiors" work landed 2026-07-10 (arena caps + capture
  drain, fixture shadows, interior far clip, PS-less prepass, async split in
  worldspace interiors); interior water and interior live lighting fixes followed.

**Acknowledged but not started:**

- **Client Phase 2 — GPU-resident scene.** The per-frame `buildGeometryDrawLists`
  rebuild (main-thread) is called "the MGE→Forge feeding cost — Phase 2 makes this
  GPU-resident so it goes to 0" (`src/mge/renderprocess.cpp:2317`, :2444). This is
  the acknowledged big structural next step on the client side.

## Inferred direction

*This section is judgment from the commit trail and code markers, not stated
anywhere by the author.*

1. **The current arc is "interiors as the showcase."** The last three weeks converge
   from three sides on fully Forge-rendered interior scenes: the point-light shadow
   system (interiors are where Morrowind's lighting is densest), explicit interior
   support work, and first-person takeover (arms are always in frame indoors).
   Release-day commits are lantern polish — the aesthetic target is candle/lantern-lit
   interiors with real shadows.
2. **Next on deck is likely FP1c and finishing first-person**, given the reserved
   wire fields and the WIP labels — the shipped default still mixes MW-rendered
   first person over the Forge world.
3. **Client Phase 2 (GPU-resident scene) is the next structural milestone** after the
   feature arc: it is repeatedly named in comments as the fix for the dominant
   steady-state CPU cost, and the walk-cut commits (W*) read as interim mitigations.
4. **Working method:** the author plans in uncommitted `tasks/*.md` documents, works
   in small plan-coded increments (P1→P7, C2→C4, AT1→AT3, FP0→FP1b), validates
   manually in-game ("user-verified" appears in commit messages), and keeps A/B
   toggles + debug views for everything. Expect future work to follow the same
   pattern: a lettered/numbered plan, default-on once user-verified.

## Settled by the 2026-07-13 audit

Formerly open questions, now answered from the code:

- **>64k-index meshes:** no guard on the main capture path, but unreachable — NI
  triangle lists are natively `uint16`, so meshes can't exceed the cap. The
  captured-alpha path has an explicit oversize drop guard.
- **Texture release path:** exists, but only on slot re-upload (prior texture freed,
  `forgerender.cpp:9955`, `:10232`). No LRU/camera eviction; `MAX_TEXTURES = 896`.
- **9On12 route:** fully removed from code; only stale comments remain
  (`renderprocess.h:20-21`, `distantinit.cpp:551`).
- **IPC behaviour on host death:** detected immediately (host process handle in
  every wait set → `ServerLost`), frame skipped; no restart/recovery path. A
  wedged-but-alive host stalls the game up to `IPC::MaxWait` = 60 s per wait; no
  watchdog exists.
- **Near-scene indirect draws:** near static opaques *are* ExecuteIndirect in
  prepass + colour; the direct-draw lanes are shadow casters, skinned, multi-map,
  sky, alpha, water, FP.
- **rc3→rc4 commit count** is 18 (an earlier analysis said 19).

## Open questions

Still unresolved as of the stamp above:

- Actual measured cost of the feeding path and the finish-chain stalls — no
  profiling has been run. Tracy zones (`MGE_ZoneScopedN`, ~82 sites) exist in the
  **client only**; the host has GPU-phase timestamp timers (F9 overlay) but no CPU
  zones.
- Which parts of the legacy distantland64 service (`src/ipc/dlshare.cpp`,
  client-owned D3D9 buffers + host cull queries) remain live when Forge is on.
- `MGEfuncs`, `MGEgui`, grass, and the pre-existing MGE post-process chain — not
  examined at all.
- Correctness of the FFE port against `XE FixedFuncEmu.fx` — the tonemap comment
  says "ported verbatim", not diffed.
