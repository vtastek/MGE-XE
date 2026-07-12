# Forge target architecture

Guidance document: the end-state we steer toward when operating on the code. Where
[ARCHITECTURE.md](ARCHITECTURE.md) describes the system as it *is* (audited
2026-07-13 at `8d2508f`), this describes the system as it *should be*, calibrated
against current industry best practice. Every change we make should either move a
subsystem toward its target below or explicitly note why it doesn't.

Targets are stated per subsystem as **current → target**, with rationale, prior
art, and acceptance criteria. Sequencing is at the end and is deliberately not
feature-ordered — see the reasoning there.

## 1. Fixed constraints (we don't fight these)

These are load-bearing and out of scope to change; every target below is designed
within them.

- **Morrowind stays a 32-bit process we don't own.** The NI scene graph must be
  scraped, not instrumented; address space in the client is scarce (the original
  motivation for the out-of-process split).
- **The client renders through DXVK.** The composite seam is D3D12 (host) ↔
  Vulkan-under-D3D9 (client). Interop options are bounded by what DXVK's
  `ID3D9VkInteropDevice` surface exposes.
- **The host output composites *under* MW's scene-1 alpha, first person (until FP
  takeover completes), UI, and MGE post-process.** Forge cannot own final present.
- **Shading stays FFE-faithful.** Visual fidelity target is "Morrowind, correct" —
  the `XE FixedFuncEmu.fx` port is a feature, not debt. No PBR, no deferred.
- **The out-of-process split is established for distant content.** Its founding
  justification — 32-bit address-space exhaustion — is rigorous for host-owned
  distant land and statics. The *mechanisms* the split relies on have solid prior
  art (cross-process shared surfaces and sync are how browser GPU processes and
  VR compositors ship), and we adopt the obligations that come with them (T7,
  T8 — notably: a process your frame depends on owes a restart path). The
  *pattern* itself — two processes rendering one shared 3D scene in per-frame
  lockstep, composited with interleaved depth/alpha — has **no direct prior
  art** we know of: browsers composite independent layers and degrade gracefully
  when one is late, which Forge structurally cannot. The novel part is the
  unvalidated part; that is one more reason the near-scene scope is not treated
  as fixed here and is validated by measurement at M0 (see §4 and P0 in §6).

## 2. Design principles

1. **The host owns frames; the client owns deltas.** Steady state, the client
   should send only what changed. Any per-frame cost proportional to *scene size*
   (rather than *change size*) on the client is debt.
2. **GPU-driven by default for opaque world content**; per-draw CPU recording is
   acceptable only for lanes that are small by construction (sky, FP) or
   structurally hostile to indirection (captured alpha).
3. **Staleness must be repaired, not faded.** Temporal masking (fade-ins) is
   acceptable as polish on top of a correct re-test, never as the mechanism.
4. **Every cap has a measured headroom story.** A hard cap is fine; a hard cap
   whose overflow behaviour is silent content loss is not.
5. **A dependency that can die gets a recovery path.** A frame that depends on
   another process requires insurance proportional to that dependency: detection,
   bounded waits, restart, and a degraded-but-playable fallback. (This is the one
   obligation we import wholesale from the systems that ship this mechanism —
   browser GPU processes and VR compositors both treat it as non-negotiable.)
6. **Measure before restructuring.** Targets T1 and T7 rest on unmeasured cost
   assumptions (flagged in SHORTCOMINGS as inferred), and the near-scene scope of
   the split itself deserves the same rigor (P0 in §6). M0 exists to convert
   assumptions to numbers before the big moves — both to tune the work and to
   confirm the premises underneath it.

## 3. Targets by subsystem

### T1. Scene feeding — retained GPU-resident scene (client "Phase 2")

**Current:** geometry cache is retained and delta-updated (good), but
`buildGeometryDrawLists` repacks *all* draw lanes every frame on the client main
thread and re-ships them; host rebuilds per-frame instance data from the blob.

**Target:** persistent host-side scene tables (instance transforms, material
routing, mesh slot refs) mirroring the geometry cache, keyed by the same slots.
Per frame the client sends: transform deltas, add/remove, visibility epoch — sized
by *change*, not by scene. Draw-list construction moves host-side (CPU incremental
or GPU compaction over the resident tables). The wire's per-frame geometry payload
in a static scene approaches zero bytes.

**Prior art:** GPU-driven scene submission (AC Unity lineage), UE retained proxy
scenes. This is also the author's own stated Phase 2 ("makes this GPU-resident so
it goes to 0", `renderprocess.cpp:2317`) — we are aligning with intent, not
overriding it.

**Acceptance:** standing still in Balmora: client kickoff CPU time and geometry
wire bytes both flat and near-zero, independent of scene complexity; no visual
diff vs the repack path (A/B toggle retained during migration).

### T2. Draw submission — indirect where it pays

**Current:** distant statics fully GPU-driven; near static opaques indirect
(prepass + colour); skinned, multi-map, shadow casters, sky, alpha, water, FP
recorded per-draw.

**Target:** skinned and multi-map lanes join the indirect path once T1's resident
tables exist (they become natural extensions of the same instance tables). Shadow
casters move to indirect *per atlas face* (biggest per-draw volume today given
32 slots × movers). Sky, captured alpha, water, FP stay direct — small by
construction (principle 2).

**Acceptance:** host draw-recording CPU per frame flat as caster/skinned counts
scale in a dense interior; measured via T9 zones.

### T3. Visibility — two-phase occlusion

**Current:** single-phase cull against the previous frame's Hi-Z (tail-submitted).
Camera cuts and load doors pop; rc4 adds temporal fade-in for shadow pops
(masking, per principle 3).

**Target:** standard two-phase scheme. Phase A: cull with last frame's Hi-Z, draw
survivors, build this frame's Hi-Z. Phase B: re-test the rejected set against the
*fresh* Hi-Z, draw the disoccluded remainder. Statics cull and shadow-light cull
both ride it. Fades become optional polish on genuinely new-to-frustum content
only.

**Prior art:** Nanite and essentially every shipped GPU-driven pipeline since
~2015.

**Acceptance:** teleport/load-door into a dense scene shows no one-frame world or
shadow pop with fades disabled; steady-state cull cost within noise of current.

### T4. Lighting — clustered binning, FFE shading preserved

**Current:** brute-force per-pixel loop over up to `MAX_POINT_LIGHTS = 128`
lights; bounded otherwise only by the shadow mask's K=8 cap. Fine at vanilla
counts; the current showcase target (dense modded interiors) is exactly where it
degrades.

**Target:** a froxel/tile light-binning compute pass; the colour pass iterates the
cluster's light list instead of the global cbuffer. The FFE per-light math is
unchanged — binning changes *which* lights a pixel iterates, not how they shade
(constraint: FFE-faithful). K-strongest shadow selection folds naturally into
per-cluster ordering. Raise or remove the 128 cap once iteration is bounded per
cluster.

**Prior art:** Forward+ / clustered forward (Doom 2016 onward; the default for
forward renderers at this light count).

**Acceptance:** GPU lighting cost scales with lights *visible in a cluster*, not
scene light count; a 300-light modded interior renders within budget; vanilla
scenes show zero visual diff (bit-identical shading path per light).

### T5. Shadows — already at target shape; scale the mask

**Current:** cached static atlas + per-frame dynamic mover layer, per-slot
invalidation, occlusion-culled bakes — this *is* best practice (id-Tech-style
atlas caching). Remaining gaps are scale knobs: 32 slots with an observed
"eviction fight" in dense scenes, and mask cost of (2R+1)²×4 loads per in-range
slot per pixel.

**Target:** keep the architecture. Slot count becomes budget-driven (grow the
atlas or tile size adaptively under pressure rather than fighting at 32). Mask
sampling gains an early-out from T4's clusters (only test slots relevant to the
cluster). Bake scheduling stays invalidation-driven.

**Acceptance:** the :4945 eviction-fight scenario holds stable slot assignments
across frames; mask GPU time bounded in the many-overlapping-lights worst case.

### T6. Residency — budgets and eviction, no silent loss

**Current:** mesh arena grows by doubling to a hard cap then silently drops parts;
textures freed only on slot re-upload (`MAX_TEXTURES = 896` fixed); no
distance/age eviction anywhere. Failure mode is silent content loss in long
sessions (principle 4 violation — the least defensible current gap).

**Target:** explicit budgets per pool (mesh arena, texture array, atlas). The
client emits free-slot signals on cache eviction / cell transition (the exact
"client free-slot signal (follow-up)" the code names at `forgerender.cpp:1353`);
the host reclaims eagerly. Distance/LRU eviction as backstop when budget pressure
hits despite signals. Overflow behaviour is *never* silent: on-screen dev warning
+ log with counts. Texture slots ref-counted against live mesh parts, freed with
them.

**Acceptance:** a scripted long play session crossing 50+ cells holds arena and
texture occupancy under budget with zero dropped-part log lines; slot-reuse
correctness verified by revisit (return to first cell renders identically).

### T7. Composite seam — GPU-to-GPU sync, minimal copies

**Current:** fence wait → `vkCmdCopyImage` under DXVK's submission-queue lock →
client `WaitForFences` (CPU), every frame; chain is host RT → (MSAA resolve) →
shared RT → Vulkan copy → DXVK image → blended quad.

**Target:** cross-API timeline sync — D3D12 shared fence imported as a Vulkan
external semaphore — so the wait moves GPU-side; the client CPU never blocks on
the host GPU in steady state. Copy count driven to the minimum DXVK's interop
allows: ideally the imported image is sampled directly by the composite draw
(zero-copy), else exactly one tracked copy. Requires a spike to establish what
`ID3D9VkInteropDevice` permits — the constraint is DXVK's surface, not D3D12/VK.

**Prior art:** Chromium shared images + sync tokens; standard D3D12↔VK external
semaphore interop.

**Acceptance:** finish-chain CPU wait ~0 in steady state (measured, T9); at most
one full-res copy per frame; no DXVK validation or hazard regressions across
alt-tab/resize.

### T8. Robustness — the recovery ladder

**Current:** host death detected immediately (`ServerLost`) but never recovered;
wedged-alive host stalls the game up to 60 s per wait; device-removed is terminal
black with a comment naming the state.

**Target,** as a ladder every failure walks down instead of falling off:

1. **Bounded waits:** steady-state IPC waits drop from 60 s to ~2 s with a
   watchdog escalation (a frame-level wait that hits 2 s is already a failure).
2. **Device-removed reattach:** host tears down the D3D12 device, recreates,
   re-imports the shared RT, and replays resident state. Client re-uploads the
   geometry cache from its retained copy (it already has one — this is cheap
   after T1) and continues.
3. **Host restart:** on `ServerLost` or watchdog escalation, the client respawns
   `mgeHost64.exe`, re-handshakes, re-uploads, continues — Chromium's GPU-process
   restart, applied here.
4. **Graceful fallback:** while the ladder runs (or if restart loops), the client
   falls back to MGE's own in-process rendering path — which still exists behind
   the ownership gates — rather than black frames. Degraded, playable, loud.

**Acceptance:** `taskkill /f` on `mgeHost64.exe` mid-play recovers to Forge
rendering within a few seconds with no game restart; a synthetic device-removed
(via dev toggle) recovers likewise; a synthetic host wedge (dev pause toggle)
never stalls the game thread more than the bounded wait.

### T9. Observability — one timeline, both processes

**Current:** client has ~82 Tracy zones; host has GPU-phase timestamp timers + F9
overlay but zero CPU zones. The two cost stories can't be correlated, and the
biggest SHORTCOMINGS items (5, 7) are inferred rather than measured because of it.

**Target:** Tracy (or equivalent) zones host-side covering IPC dispatch, upload
parse, draw recording, submit, and present-fence waits; client and host clocks
correlated (Tracy supports multi-process capture) so one timeline shows kickoff →
host frame → finish wait. GPU phase timers feed the same view. This is the
prerequisite for every acceptance criterion above that says "measured".

**Acceptance:** a single capture answers "where did this frame's 16 ms go" across
both processes; feeding cost (item 5) and finish stalls (item 7) get real numbers.

## 4. Sequencing

Ordered by dependency and risk, not by feature appeal. Notably this *diverges*
from the current commit trajectory (interior/FP polish) — deliberately: the next
structural layers all want measurement and safety first.

| Milestone | Content | Unlocks |
|---|---|---|
| **M0** | Premise validation (P0 below) + T9 observability + a T7 interop spike (what does DXVK allow?) | Confirms the near-scene scope; real numbers for items 5/7; re-scope M3/M5 with data |
| **M1** | T8 robustness ladder (bounded waits → restart → fallback) | Safe iteration for everything after; user-facing stability |
| **M2** | T6 residency (free-slot signal, budgets, eviction) | Long sessions; removes silent-loss failure mode |
| **M3** | T1 GPU-resident scene (Phase 2) | Kills the feeding tax; precondition for T2, cheapens T8 replay |
| **M4** | T3 two-phase occlusion + T2 indirect lanes | Kills pops; recording cost flat |
| **M5** | T4 clustered lighting + T5 mask/slot scaling | Dense-interior showcase actually scales |
| **M6** | T7 seam (shared fence, copy reduction) | Steady-state latency/CPU win; scope set by M0 spike |

Rationale for the two most debatable calls:

- **Robustness before Phase 2 (M1 < M3):** every subsequent milestone involves
  destabilizing the host repeatedly during development; the restart/fallback
  ladder pays for itself in iteration speed alone, and it's the only current gap
  that loses a *user's session* rather than milliseconds.
- **Residency before Phase 2 (M2 < M3):** the free-slot signal defines the
  client↔host lifecycle contract that Phase 2's resident tables must also honor —
  designing eviction after the tables exist means retrofitting the contract.

## 5. Non-goals

- Deferred shading, PBR, or any departure from FFE-faithful output.
- Moving rendering back in-process, or replacing DXVK on the client.
- A general asset-streaming system beyond the residency contract in T6.
- Multi-GPU, HDR output, or VR — nothing here should preclude them, but none is a
  target.
- Rewriting the vendored The-Forge framework; we live with its idioms (e.g. no
  comparison samplers — the manual-PCF workaround stays).

## 6. Open decisions (resolve before the affected milestone)

- **P0 — premise validation (resolve at M0, before further structural
  investment).** The architecture's benefits for *distant* content rest on the
  well-founded 64-bit argument. The **near-scene takeover** rests on a different
  argument — access to compute (GTAO, point-light shadows, unified lighting),
  which D3D9 cannot provide — and that argument, while plausible, has not yet
  been confirmed by measurement or weighed against lighter alternatives. It also
  carries most of the catalogued costs (near assets are resident twice — in the
  game process and in the host arenas — plus the feeding path, the capture
  lanes, and the seam traffic; SHORTCOMINGS items 1-2, 5, 11, 17 all trace to
  it). This is very likely simply work that outpaced its paperwork — the fork
  kept every switch needed to settle it cheaply. **Deliverable:** an A/B
  benchmark, Forge-on vs. in-process MGE XE (the path still present behind the
  ownership gates), measuring frame time and input latency in the scenarios
  Forge targets (dense city, lantern-lit interior) on at least one high-end and
  one low-end GPU. A clear win confirms the premise and turns it into a
  documented strength; a null or negative result would argue for re-scoping
  toward host-owned distant content + the cheapest available compute path for
  AO/shadows, before M2+ investment compounds. Either outcome, the numbers go
  in STATUS.md and this item closes.
- **M3:** host-side draw-list build — CPU-incremental vs GPU compaction? (Decide
  with M0 numbers; GPU compaction only pays if instance counts demand it.)
- **M5:** froxel grid vs screen-space tiles for clustering? (Froxels compose
  better with the existing camera-relative + reverse-Z conventions; verify against
  interior far-clip behaviour from the rc4 work.)
- **M6:** if DXVK's interop can't import a D3D12 fence, fallback design — host
  pre-signals via a shared-memory sequence + client polls on the GPU timeline, or
  keep one CPU wait but move it off the critical path?
- **M1:** fallback threshold — after how many restart failures do we latch to
  MGE-only rendering for the session?
