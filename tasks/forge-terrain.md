# Terrain overhaul — retire the DL generator, load LAND at launch, one terrain

**Status:** **T0–T4 LANDED** (2026-07-29). The DL world bake is gone: host, shaders, and the MGEgui
generator. Terrain is the world's one surface, near and far, main view and reflection.
Supersedes the terrain half of `tasks/forge-near-far-unify.md` (whose "MGE clipped
land, we sink it" paragraph is corrected there, in the CORRECTION 2026-07-29 section).

---

## LANDED 2026-07-29 — T0: the LAND loader runs in the host at launch

`mgeHost64/terrain.cpp` + `.h`. Reads `Morrowind.ini [Game Files]` directly (host cwd is the install
dir), memory-maps each plugin, and walks LAND + LTEX. Started from `main()` on a background thread
**before** the client handshake, so the parse overlaps MW's own load. No disk artifact, no
read-back, nothing to regenerate when the mod list changes.

Ported verbatim from `MGEgui/DistantLand/DistantLandForm.cs:496-573`, including both silent-
corruption traps: the **VTEX 4×4-block interleave** and the **per-plugin LTEX index space** (each
plugin's cells are resolved against ITS table at plugin end, which also makes an LTEX record that
appears *after* a LAND record resolve correctly — MGEgui gets that for free by sharing one mutable
dictionary per plugin). VNML is parsed past and dropped; VCLR is kept.

**Verified against the plan's census** (`mgeHost64.exe --terrain-census`, a probe that runs the
loader with no GPU/IPC so this needs no game session):

```
>> [terrain] plugins: 30 loaded, 0 MISSING (329.1 MB walked)
>> [terrain] LAND: 3898 cells  x[-28..49] y[-58..33]  (4132 records, 130 overrides, 104 no-heights,
             0 out-of-range, 0 no-VHGT, 114 no-VCLR)
>> [terrain] LTEX: 499 unique textures (847 records, 0 unresolved VTEX indices)
>> [terrain] heightless LAND records: 104 (12 are cells with NO terrain: (-28,18) (-25,14) (-24,14)
             (-23,14) (-19,28) (-18,28) (-17,9) (-12,4) (-10,-9) (-10,-8) (-9,-14) (-9,-13))
>> [terrain] height range -9728..21168 world units;  resident 80 MB;  parsed in 84 ms
```

Extent and the 499 LTEX match the plan exactly. The cell count is **3898, not 3910** — and the
missing 12 are named in the log above. They are cells whose only LAND record has `DATA` bit 0 clear
(no vertex heights); MGEgui skips those and so do we. The plan's Python census counted every LAND
record's `(x,y)` without that gate. **This matters for T3:** those 12 are exactly the cells that will
have no terrain once MW's near land is gone, so they are enumerated rather than counted.

84 ms warm, well inside the "a few seconds at launch" budget, and 80 MB resident — so, as predicted,
streaming is a non-issue.

**RESOLVED (2026-07-29), not by wiring it — by replacing it.** `Terrain::checkClientPluginList()` was
written to diff MW's loaded-master list against the ini's. It is now **deleted**, because that check
is not reachable and was not the right one anyway:

- **Not reachable.** MW's real load order lives in the engine; SharedSE does not expose it, and
  extracting it is disassembly (PD6). Having the *client* read the same `Morrowind.ini` back to us
  proves nothing about what the engine loaded.
- **Not sufficient.** A name diff can come back perfectly clean and the world still have a hole — a
  plugin that parsed but whose LAND we dropped, a heightless cell, an extent bug. The plugin list is
  a proxy for the thing we actually care about.

So the failure is caught by its **consequence**, which is both reachable and strictly more general.
Three layers, all live:

| symptom | caught by | where |
|---|---|---|
| ini lists a plugin, disk lacks it | named at load | `terrain.cpp` — `"plugin missing: %s"` |
| cell has no LAND record anywhere | enumerated at load | the heightless/no-terrain list above |
| gap within one cell of the camera | per-frame tripwire | `g_terrainEyeCellMissing`, below |

The third is the one that closes the plan's "T3 is the irreversible one" risk, and it does more than
log: **it hands the near field back to MW.** See the T3 section.

## IMPLEMENTED 2026-07-29 — T1: heightfield residency + terrain draw

`terrain.vert/.frag.fsl`, the terrain block in `forgerender.cpp`, `gTerrainHeights`/`gTerrainColor`
appended to the PerFrame SRT set. **Two deviations from the plan as written, both forced:**

1. **Heights are a typed `Buffer<uint>`, not a `Texture2DArray`.** D3D12 caps a Texture2DArray at
   2048 slices (`D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION`) and there are 3898 cells, so the array
   form cannot hold the world without being split. A buffer has no such cap, is an exact integer
   fetch (the VS samples the lattice at grid points — nothing to filter), and stays one SRV. Two
   int16 pack per uint: 33 MB heights + 66 MB VCLR (RGBA8) = **99 MB VRAM**, uploaded once in 8 MB
   chunks at first exterior.
2. **LOD seams are closed by EDGE STITCHING, not skirts.** A boundary vertex whose neighbour cell
   drew at a coarser stride resamples its height on that neighbour's lattice, so the shared edge is
   literally the same polyline — crack-free by construction, with no skirt geometry, no skirt-depth
   constant to tune, and no skirt wall hanging out of a cliff silhouette. Corners never move (0 and
   64 are multiples of every power-of-two stride), so the two axes can never fight over one vertex.
   The instance row carries the four neighbour slots and their strides; a neighbour culled away this
   frame is stamped out and treated as equal stride, because there is no edge to match.

The rest is as planned: one shared 65×65 lattice VB in grid units (no baked positions), one IB
holding all six strides back to back, one 40-byte instance row per visible cell, six
`cmdDrawIndexedInstanced` calls. Lit by VCLR × sun/ambient with the two-scale detail map, MSM sun
shadow and clustered baked point lights — deliberately the same lighting shape as
`distantland.frag`, so the A/B compares geometry and not two lighting models. **No z-sink.**

Normals are already computed from the heightfield (central difference at the cell's own LOD step,
crossing into the neighbour cell's data so cell boundaries do not get a lighting seam) — that half
of T2 came for free and makes the T1 A/B readable.

Panel toggles, next to `Draw: reflect land+statics`: `Draw: terrain`, `Terrain: near cut (leave MW's
own land alone)`, `Terrain: wireframe`, and `Draw: distant land (OLD DL bake — A/B vs terrain)`.
The old bake now defaults OFF and terrain ON; if terrain residency fails to come up the host flips
that pair back and says so in the log, so a failure is a visible downgrade rather than an empty
horizon. Heartbeat: `terrain=drawn/inRange cells (nearCut=N) X.XXM tris (lod a/b/c/d/e/f)` on the
gpu-split line — the cull has three independent limiters (DrawDist cap, near cut, frustum) that look
identical from outside, so the funnel is carried per frame rather than inferred.

### First harness run (2026-07-29) — works, but the scene proves nothing

`forge-perf-run.sh`, unpinned save. Residency uploaded (`3898 cells … 94 MB VRAM`), terrain drawing,
no device removal, `gpuCull MATCH` on every sample but the first (that one is the pre-existing
statics warm-up frame with `hizOccl=2`, not terrain — terrain does not touch the statics cull).

But only **19 cells** drew, none past LOD 1. The funnel says why, and it is not our cap:
`3898 resident -> 591 in range (drawDist=16.0, maxView=131072) -> 591 after nearCut -> 17 in
frustum`. So DrawDist and the near cut drop nothing; the frustum does the cutting. With
`nearTris=0.02M` and `static-instances=17` on the same frames, the scene itself is near-empty — the
harness loads the newest `.ess` and warned that nothing was pinned. **The numbers are consistent
with a camera looking at almost nothing, and they do not yet demonstrate the LOD ladder.** Re-measure
on a real vista (and pin the save) before drawing any conclusion about cell counts or `dl=` cost.

**The near cut is T1-only scaffolding.** MW still draws its own land, so host terrain skips cells
that lie wholly inside MW's coverage. It uses the 3D distance, not an XY column — an XY-only cut
keeps cutting when the camera climbs away and MW has already stopped drawing
(`[[project_statics_nearcull_height]]`). Straddling cells are kept, so the two surfaces overlap in a
thin ring rather than gapping; they are the same heightfield at the same stride there, so the cost
is a ring of untextured-vs-textured shading instead of a hole. T3 deletes the cut, MW's near land,
and the ring together.

## IMPLEMENTED 2026-07-29 — T2: real land textures, detail map deleted

User on the T1 build: *"geometry looks good. textures are all snow or the detail map is overlayed.
detail map is not needed if we get higher quality texturing, it was a trick for the low quality
atlas."* Both correct — T1 had no LTEX at all, so albedo was VCLR (near-white) × detail. **The
detail map is gone**, not made optional: its two-scale modulation existed to fake variation the
44-texel-per-cell atlas could not carry, and that reason has stopped being true.

**The texture layout is Morrowind's own, and MGE's generator documents it exactly.**
`DistantLandTextures.cs::CalcWeights` ("match Morrowind rounding") assigns vertex (x,y) the texture
square `(floor((x-2)/4), ceil((y-2)/4))` — the floor/ceil asymmetry is the engine's — wrapping into
the neighbour cell via `ModCell` when the index leaves 0..15, and blends by interpolating the
resulting per-vertex 0/1 weights. `CellTexCreator` sets `cv.u = x/4`, so **a land texture tiles once
per 512-unit texture square**. `terrain.frag` reproduces that per pixel: take the four bracketing
vertices, look up each one's square, blend with bilinear weights. One pass, four taps, and a single
tap on the common interior case — no four-layer "texture bank" (that was a fixed-function limit).

The neighbour reach is served by a **world cell-lookup grid** (`gTerrainCellGrid`, local grid coord
→ slot+1, 78×92 here) rather than per-neighbour slots in the instance row. It reaches diagonally,
which the square wrap needs at cell corners, and it let the VS drop its four neighbour slots too —
the instance row stayed 32 bytes while getting strictly more capable.

VTEX ships as resolved `(bucket<<16)|layer` **slots**, not LTEX ids: the host remaps the pack after
building the arrays, so the frag's hot path is one load per tap instead of a load plus an
indirection.

### Measured on this install (the plan asked for this before trusting the 1024 cap)

```
texture index: 16638 loose under Data Files\Textures, 4783 in 3 BSA(s), 50 ms
bucket=1 1024x1024 slices=87  mips=5  (full=11)   <-- SHORT CHAIN
bucket=2 1024x1024 slices=370 mips=11 (full=11)
bucket=3 1024x1024 slices=7   mips=4  (full=11)   <-- SHORT CHAIN
bucket=4 256x256   slices=29  mips=9  (full=9)
bucket=5 128x128   slices=2   mips=5  (full=8)    <-- SHORT CHAIN
bucket=6 1024x512  slices=1   mips=11 (full=11)
bucket=7 512x512   slices=2   mips=10 (full=10)
499 textures: 498 uploaded, 2 missing, 0 overflow; 7 buckets (cap 1024),
short-chain slices=96, ~311 MB VRAM
```

**~311 MB at cap 1024**, close to the plan's 333 MB estimate — because this install is an HD
retexture set: **463 of 499 land textures are 1024²**. Dropping the cap to 512 (the statics value)
would quarter that to roughly 80 MB. That is a quality/VRAM call, not a correctness one, and
`kTerrainTexCap` is the single constant.

Three things the measurement caught that would otherwise have shipped silently:

1. **Bucketing must key on MIP COUNT, not just (format, size).** A Texture2DArray has one mip count
   for all slices, so with the statics key the chain is the MIN over members — and ONE truncated
   DDS cut all 463 1024² textures to 4 mips of 11, i.e. nothing below 128². The whole world would
   have shimmered past that distance. Keying on mips costs two extra buckets (of 32) and isolates
   the offenders: 370 slices now carry the full chain, 96 do not. Those 96 are **truncated on
   disk** — `[[project_dl_mipfix_source]]` (MGEgui's source-mipmap fixer) is the tool for them.
2. **Texture id 0 was never loaded.** `_land_default.tga` is the fallback ground every unpainted
   texture square uses — one of the most-drawn textures in the world — and the pass-1 loop started
   at id 1, drawing all of it white. Fixed; upload count went 497 → 498.
3. **Two LTEX records name textures nobody ships** (`tx_ma_sandstone02.tga`, `tx_lavacrust00.tga` —
   vanilla content gaps; the latter appears only inside mesh data in Morrowind.bsa, never as a file).
   They now fall back to the DEFAULT ground, as the engine does, rather than to white — white would
   make a content gap look like a renderer bug.

Normals were already recomputed from the heightfield in T1, so that half of T2 was done early.

### Land-texture mip repair (2026-07-29, user: "run MGEgui's source-mipmap fixer")

The fixer existed but **did not cover land textures**: `StaticTexCreator.FixSourceMips` is only
reachable from the statics texture path, and `LTEX.LoadTexture` — a different class entirely — has no
mip-fix branch. Running it as shipped would have repaired nothing here.

So `FixSourceMips` became `internal static` (it used no instance state) and a new
`MGEgui.DirectX.LandMipFixer` drives it over the LTEX set, called from `workerLoadPlugins` once the
whole set is known. Same guarantees as the statics path — append-only, authored mips byte-for-byte,
original backed up to `Data Files\distantland\mipfix_backup`, BSA-packed vanilla assets reported but
never rewritten, idempotent. Counts and per-file paths go into the wizard's warnings.

**To run it: MGEXEgui → Distant Land → Plugins tab → Run.** Only the plugin step is needed; the
repair happens during the LTEX walk, so nothing that this plan is deleting has to be generated.

The host now NAMES the short-chain sources in `mgeHost64.log` rather than only counting them —
`_land_default.tga` is among them, i.e. the single most-drawn texture in the world ships a truncated
chain. (`DistantLandForm.cs` is cp1252; patched via latin-1 per
`[[project_cp1252_edit_tool_corruption]]` — verified 0 replacement chars, original high bytes intact.)

**Result of the first run: 92 files repaired, short-chain slices 96 → 3.** The three left were
`_land_default.tga` and two `water\water01.*`. The default ground was a gap in the collector — it is
constructed in `workerLoadPlugins` rather than by an LTEX record, so gathering only from the record
branch skipped the single most-drawn land texture in the world. Now added explicitly; re-run the
plugin step to pick it up.

**Still open:** `kTerrainTexCap` is 1024 (~311 MB). 512 would be ~80 MB. Not yet decided.

### The white-terrain bug (same day, found from the user's report)

User, on the first T2 build: *"okay, so I should see textured terrain and not white terrain? because
I see white terrain."* Yes — and the cause was in the VTEX remap, not the sampling.

`gTerrainTex` packed **two** entries per uint (16 bits each), carried over from the heights/VCLR
layout. But the renderer rewrites LTEX ids into `(bucket<<16)|layer` texture SLOTS before upload, and
a slot needs the full 32 bits: any real texture (bucket ≥ 1) is ≥ 65536. The guard that was supposed
to catch an out-of-range slot — "clamp loudly to bucket 0 rather than alias into a wrong texture" —
therefore clamped **every** texture in the world to slot 0, which is the 4×4 white fill. It also was
not loud: it clamped silently, which is exactly what the comment claimed it would not do.

Fixed by storing ONE slot per uint (cell stride 256; 4 MB for the whole world — there was never
anything to save), and by making the remap actually report: `VTEX remap: N squares -> slots, U
unresolved (white)`, with an explicit `<-- ALL WHITE, remap is broken` when U == N. That line reads
`0 unresolved` now, and would have caught this in the first harness run.

### T1/T2 harness (real scene this time)

`terrain=159/597 cells (nearCut=0) 0.18M tris (lod 13/19/48/79/0/0)`, `gpuCull MATCH`, no device
removal. The LOD ladder is doing its job — 159 cells for 0.18 M triangles. Strides 16/32 never fire
because `Draw Distance=16` cells and their thresholds are 24/48; headroom, not a bug.

**`nearCut=0` every frame, which is worth knowing before T3:** a cell sphere's radius is ≥5793, so
"wholly inside 7168 of the eye" is essentially never true, and host terrain is *already* covering the
near field on top of MW's land. It wins the depth test (drawn later, GEQUAL passes at equal depth)
and the user reports the geometry looks right — so the T3 removal is mostly bookkeeping rather than
a visual change, and the near cut is closer to dead code than to load-bearing scaffolding.

## IMPLEMENTED 2026-07-29 — T3: MW's near land suppressed, and the LOD-tied sampling fixed

### Where MW's terrain actually came from

Worth recording, because it is not where the plan said to look. MW's own terrain **DIP is already
rejected** under the seam — `inspectIndexedPrimitive`'s `isCoveredOpaque || isLandSplat` gate has
swallowed it since S4. What was still drawing MW's terrain was **the host**, replaying the geometry
cache's captured `isLandscape` entries. So T3's "suppress at the proxy reject gate" was already
done; the live change is one line in `buildGeometryDrawLists`.

### The ownership handshake (the part that is not a one-liner)

Suppression cannot be a client setting. If the client stops emitting MW's land while the host is NOT
drawing terrain — still loading, load failed, panel toggle off — the near field is a **hole**, which
is strictly worse than the double-draw it replaces. So the host reports it:

- `HostFrameTimings::terrainOwned` (appended; static_assert 15 → 16 floats). `float`, not `bool`,
  per that header's layout contract — it is shared BY LAYOUT across the x86/x64 wire.
- Host sets it from `g_terrainReady && g_drawTerrain`, so a panel flip hands the near field back on
  the very next frame.
- Client mirrors it into `g_hostOwnsTerrain`, ANDed with `forgeOwnsFrame()` — when the seam drops
  there is no RPC, so the flag would otherwise read stale-true.

**No new toggle and no new key.** The host's existing `Draw: terrain` checkbox IS the A/B: unchecking
it drops `terrainOwned`, which restores MW's land immediately. The keyspace is full anyway (free
numpad keys collide with the water-flow handlers' edge polls).

`g_terrainNearCut` now defaults **off** — the host owns near and far on one ladder, so the cut would
carve a hole rather than avoid an overlap. Kept as a checkbox only because it is the fastest way to
prove the near field is ours: with it on, anything still drawn near the camera came from elsewhere.

Verified: `>> [seam] host terrain ownership ON — MW near land SUPPRESSED`, `nearTris=0.00M` on an
otherwise-empty frame, no device removal, `terrain=231/550 cells 0.20M tris`.

**Deliberately NOT done yet:** the landscape *capture* (`g_walkingLandscape`, the `g_landRoot` walk)
still runs. Keeping it is what makes the A/B instant and reversible — toggling the host checkbox
restores MW's land with no re-walk. Removing it is a pure CPU/memory saving and belongs after the
suppression is visually proven, not before.

### VCLR and normals were tied to mesh LOD (user-reported, fixed here)

User: *"visually, vcol loses quality since it is tied to mesh resolution. there are thin roads
painted everywhere."* Correct, and it is the same structural mistake heights would have had if they
had been baked into geometry — **the sample rate of shading data was inherited from the mesh.**

- **VCLR now samples per PIXEL**, bilinearly over the full 65×65 field, in `terrain.frag`. A
  vertex-stage fetch only sees the vertices the current LOD draws: at stride 8 that drops seven of
  every eight painted values and smears the survivors over 1024 world units, so thin hand-painted
  features (roads) wash out and crawl as the LOD changes underfoot. It reuses the `x0/x1/y0/y1/fx/fy`
  the texture blend already derived, so the whole fix is four buffer loads and three lerps. Edge
  behaviour is a clamp, which is faithful — each MW cell interpolates its own vertex colours.
- **Normals now use a stride-1 central difference always**, never the LOD stride. At stride 16 the
  old code averaged the gradient over 2048 units and shaded the terrain flat at exactly the distance
  you see most of it. Stride 1 gives each drawn vertex its TRUE normal — the same one MW's own
  per-vertex normal carries — for the same four loads. Free, and parity rather than embellishment.

The `Color` interpolator left `VSOutput` entirely as a result.

### Scale headroom (user: "half of Tamriel Rebuilt, and there is also skyrim landmasses")

Measured per cell: heights 8.45 KB + VCLR 16.9 KB + VTEX 1 KB ≈ **26 KB**. Today's 3898 cells = 98 MB.
A world that doubles is ~210 MB of buffers — still nothing, and still no streaming scheme needed.
**VCLR is now the dominant per-cell cost**, 2× heights, and it is the one that could be halved
(RGB565 / a palette) if that ever matters. It does not yet.

Textures are the real axis: 311 MB at cap 1024 for 499 LTEX, and every added landmass adds LTEX to
the same dominant (DXT1, 1024², full-chain) bucket. Two guards went in for that:

- **Bucket rollover at 2048 slices.** `D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION` is 2048; the biggest
  bucket holds 370 today. A key whose bucket is full now starts a fresh bucket instead of failing
  resource creation at some future install size — which would have surfaced as "those textures are
  white", a long way from the cause.
- **Overflow is loud.** Running out of buckets (`MAX_TERRAIN_BUCKETS`, 32) now logs the count and
  names the constant to raise. Currently `6/31 buckets`.

### The coverage tripwire — T3's safety valve (added after the above)

This is what makes T3 reversible in the field rather than only on the bench, and it replaces the
deleted plugin-list diff (see T0). The plan's stated risk was: *"Removing MW's near land means any
cell the host fails to load is a hole in the world, not a quality drop… fail loud on a missing cell
rather than silently drawing nothing."* That is now enforced, in `terrainCullAndBuild`:

- Every frame, test `Terrain::slotAt()` over the **3×3 cells around the eye** — not just the cell the
  camera is in. `terrainOwned` reaches the client one RPC late, so testing only the current cell
  reports the hole on the frame you are already standing in it: one frame of visible hole per
  crossing. A cell is 8192 units and you cannot cross one in a frame, so the ring buys a full cell of
  warning and the handover completes before anything is missing on screen.
- On a gap, `g_terrainEyeCellMissing` goes true and is ANDed **out** of `terrainOwned` — so MW's near
  land comes straight back. Overlap in the neighbouring cells is the cheaper failure, the same
  trade the near-cut straddler rule already makes: *overlap costs a ring of shading, a gap costs a
  hole.*
- **Logged once per distinct cell**, not per frame — a hole you can walk in and out of would
  otherwise bury the log — and the line names the cell `(x,y)` and the resident cell count, which is
  what identifies the missing plugin.
- **Visible in game, not only in the log:** a coverage line on the Draw tab under the terrain
  toggles, red when the near field has been handed back. It distinguishes the three ways host
  terrain can stop covering the near field — not resident / toggle off / real gap — which are
  otherwise indistinguishable on screen.
- Cleared with the residency it describes, so a reload cannot come up claiming a hole it never
  tested for (the logged-once set being sticky would otherwise hide the re-test).

Cost is one hash lookup per frame in the common case.

## LANDED 2026-07-29 — T4: the old pipeline deleted

User: *"delete old, like you said we are 370x of it, there is no comparison."*

### T4a first — the reflection had no terrain (found before deleting anything)

T3 wired terrain into the MAIN view only; `terrainCullAndBuild` ran under `if (T.primary)`, and the
water reflection was still drawing the DL bake. Deleting the bake first would have left reflections
with **no ground at all**, so the reflection got terrain before anything was removed.

`TerrainView` now holds the whole per-view cull state — visible list, LOD arrays, per-LOD draw
counts, instance ring — with `g_terrainMain` / `g_terrainRefl` instances, mirroring the existing
`DlCullTargets` split. Two things that had to be per-view, not shared:

- **The LOD arrays.** The edge stitch reads a NEIGHBOUR's chosen stride. One view's LOD decisions
  leaking into the other's stitch tears cell edges in whichever view records second.
- **The per-LOD histogram.** It is what `base[]` is computed from when packing instances into the
  ring. The first refactor left `base[]` reading the primary-only diagnostic histogram, which for
  the reflection is all zeros — every LOD would have written at offset 0, on top of each other.

`terrainRecord(cmd, view, frameSet)` takes the descriptor set as its only per-view difference: main
passes `pPerFrameSet`, reflection passes `pPerFrameSetReflectGeo` (mirror-about-water + below-water
clip). Terrain is CULL_NONE, so unlike statics the mirror's winding flip needs no second PSO. The
near cut is main-view only — it exists to leave MW's own land alone, and the reflection has no MW
land to leave alone.

**This is what closes the shoreline gap** `[[project_forge_reflection_content]]` describes: the
reflection now sees the same surface at the same resolution the camera does, so a coastline reflects
its real shape instead of a 915-tri approximation clipped at `waterLevel-1` to hide the mismatch.

### T4b — host

Deleted: `LandMeshGPU` + `g_landMeshes` (16384 slots) + `g_landMeshCount` + `g_landLoaded`,
`buildLandPath`, `loadDistantLand`, `g_pLandShader/g_pLandPipeline`, both land cull loops and both
land draws (main + reflect), `g_liveLandVisible{,Refl}`, `g_liveLastLand`, the `Draw: distant land`
toggle, the three atlas slot constants, and `distantland.vert/.frag` (FSL source, `shaders.list`
entry, gen/, bin/, and the stale copies in the deployed `DIRECT3D12/`).

Three things that were entangled with it and had to be re-pointed rather than deleted:

- **`g_landLoaded` was the DL-resident gate for STATICS too**, not just land. Now `g_dlLiveInit`.
- **`lodParams.w` survives** — `statics.vert` still gates the hero near-cut on nearViewRange. Only
  `.xyz` (the atlas slots) died. Left as zeroed padding; repacking would shift every field below it.
- **`loadDistantLand` also loaded the baked lights.** `dlLoadBakedLights()` is now called directly
  from the live init and the viewer.

The lazy init's fallback inverted: terrain used to fall back to the bake, so a terrain failure was a
downgrade. There is nothing to fall back to now, so a failure disables distant land outright and
says so. `--forge-view` was re-pointed at terrain (bounds from the cull spheres). `--forge-dl` is
**retired**: that probe is deliberately absolute-space (`lodEye = 0`) and terrain instances are
camera-relative, so it cannot host them; `--forge-statics` survives, statics-only.

### T4c — NOT done, deliberately: the landscape capture stays

`captureLandMesh` was already gone. The remaining piece — the `g_landRoot` walk and
`g_walkingLandscape` — **must stay**, and the plan asking for its removal is now wrong. It produces
the `isLandscape` entries that `renderprocess.cpp:2405` replays whenever `terrainOwned` is false.
That IS the fallback the T3 coverage tripwire depends on: drop the capture and every coverage gap
goes back to being a hole in the world. The CPU/memory saving does not buy that back.

### T4d — MGEgui

The generator no longer produces `distantland\world`, `world.dds` or `world_n.dds`. Removed: the
LAND record parse and `LandMap`, `AtlasSetup` + `AtlasRegion` + the atlas fields, `workerCreate
Textures`/`workerCreateMeshes` and their completion handlers, `GenerateWorldMesh`, `GetTex`, and
`TextureBank` / `CellTexCreator` / `WorldTexCreator` (~24 KB of `DistantLandTextures.cs`).
`workerFLoadPlugins` now hands straight to the statics page.

**Kept, and it is the reason the LTEX parse survives:** `LandMipFixer`. The host renders terrain
from those loose land textures directly and buckets them by mip count, where ONE truncated source
caps the chain of every texture sharing its bucket.

Three follow-ons that would each have been a silent breakage:
- `MainForm.cs` gated "enable distant land" on the three world files existing — it would have
  refused forever. Now `fn_dlver` only, in both places.
- Same for the wizard's finish handler.
- `bTexRun_Click` / `bMeshRun_Click` are kept as forwarding stubs: the designer wires the buttons to
  them, so deleting the methods breaks the build.

`Exists` was already computed from `fn_dlver` alone, so statics-skip still works.

Verified: mgeHost64 (Release-Fast), MGEXEgui, and mgecore.dll all build clean; shaders recompile with
no distantland artifacts; host + shaders deployed.

---

**Original direction below (2026-07-26).**

## The user's framing

> "terrain is fully static, the height (normals?) data comes from plugin files. So it is a one time
> thing. DL generator is an archaic concept though, OpenMW doesn't have one. They just launch into
> the game. So the scope is more like, DL generator overhaul. For terrain generation, the texture
> atlas is never enough. Mesh quality is too low. DL gen OOM error with this much land and ultra
> high, and it ends up as bad terrain quality anyways. […] Morrowind terrain is hand made, hand
> textured with brushes. So most terrain implementations are no use, they assume procedural."

Prior art: **terrain subdivision + displacement with height-based blending, already implemented by
the user on branch `scene-walk`.** Port from there, don't re-derive.

## What the pipeline actually does today (VERIFIED 2026-07-26)

**The good news first: the ESM parser already exists and is complete.**
`MGEgui/DistantLand/DistantLandForm.cs:496-573` walks every plugin and reads the full LAND record —
`INTV` (cell x/y), `DATA` (bit 0 = uses vertex heights), `VHGT` (65×65 int8 deltas + float offset,
row-relative — note `offset = land.Heights[0,y]` at :522), `VNML` (65×65 sbyte normals), `VCLR`
(65×65 hand-painted vertex colour), `VTEX` (16×16 texture indices, stored **interleaved as 4×4
blocks of 4×4** — :544-552, an easy thing to get wrong in a port), plus LTEX name resolution with a
missing-index fallback to `default.dds`. **This is the piece to port, verbatim, to the x64 host.**

Everything downstream of it is what has to go:

1. **Mesh = greedy decimation to a vertical tolerance.** `GenerateWorldMesh` (:2365) picks
   `tolerance` from `{15, 70, 125, 180, 235}` world units by detail index and calls
   `NativeMethods.TessellateLandscapeAtlased`. Even "ultra" throws away everything under a 15-unit
   vertical error on a grid whose source spacing is 128 units. The output is a **fixed decimated
   mesh** — no continuous LOD, so it can never agree with the near terrain, which is the
   near/far seam.

2. **Colour = ONE global atlas covering the entire map span.** `WorldTexCreator(Res, AtlasSpanX,
   AtlasSpanY)` (DistantLandTextures.cs:996), `Res = 128 << comboIndex` (default 2048), capped at
   `MaxTexSize`. Texel density is `Res / mapSpan`, so **quality falls as landmass grows** — add
   Tamriel Rebuilt and every existing texel gets coarser. Normals are a second atlas at `NormRes`
   (default **1024**), i.e. coarser still.

3. **The OOM is structural, not a leak.** Two independent quadratic allocations in a 32-bit process:
   - `WorldTexCreator` holds **three** `Res²` textures at once (:1008-1010) — a DEFAULT-pool render
     target, a DXT1 sysmem copy, and an **X8R8G8B8 sysmem copy**. At Res 8192 that uncompressed
     copy is ~268 MB before mips; at 16384 it is ~1 GB. Built once for colour and again for normals.
   - `height_data = new float[DataSpanX * DataSpanY]` per atlas region (:2400), where
     `DataSpan = regionCells * 64` — a 100×100-cell region is 6400² floats = 164 MB.
   - …on top of `LandMap[,]` holding a full `LAND` object (65×65 heights + normals + colours +
     VTEX) for **every cell in the world simultaneously**.

   So "OOM at ultra with this much land" is exactly right, and **raising the resolution to fix the
   quality is what triggers the OOM** — the two failure modes are the same knob. That is the whole
   case for deleting this pipeline rather than tuning it.

4. **Any content change invalidates the bake.** The generator is offline and disk-backed
   (`Data Files\distantland\world`, `world.dds`, `world_n.dds`, loaded at
   `distantinit.cpp:978-1009`). Mod churn = re-bake. OpenMW has no such step because it reads the
   plugins at load.

## Target

Read LAND from the plugin list **at launch, in the x64 host**, and keep it as a heightfield —
never as a decimated mesh, never as a global atlas.

- **Geometry:** GPU subdivision + displacement from the heightfield. LOD becomes a continuous
  tessellation parameter, so there is no near/far mesh to reconcile — this is what makes
  `forge-near-far-unify.md` fall out for free rather than needing a cross-fade.
- **Texturing:** per-cell `VTEX` indices → the host's existing bindless texture array, sampling the
  **real** land textures at full resolution, blended by height (the `scene-walk` work). No global
  atlas exists, so there is no density-vs-memory tradeoff left to lose.
- **Normals:** recompute from the subdivided heightfield. `VNML` is per-original-vertex and would be
  a downgrade after displacement. **`VCLR` is hand-authored and must be kept** — it carries artist
  intent that cannot be recovered procedurally.
- **Hand-painted, not procedural:** `VTEX` is a discrete index map, not splat weights. Height-based
  blending is a *refinement between the indexed layers*, not a replacement rule. Terrain literature
  that assumes procedural placement does not apply here — the user's explicit warning.

**Why x64 host, not the x86 client:** every OOM above is a 32-bit address-space failure. The host
has the headroom, already owns DL, and already has the bindless array the textures land in.

## Open questions

- Budget: the full-world LAND set in memory (heights + VTEX + VCLR, dropping VNML). Estimate before
  designing the residency scheme — it may simply all fit, which would make streaming a non-issue.
- ~~What replaces `captureLandMesh`~~ — **VOID (2026-07-26).** That constraint came from
  `dx9-retirement.md` premise 4, which is stale: `renderexterior.cpp` and the MSOC horizon occluder
  were deleted in S4, and `DistantLand::landMeshes` now has **no reader at all**. Nothing depends on
  the DL land VB as a lookup key, so a heightfield-based host terrain has no D3D9 buffer to keep
  alive. `captureLandMesh` is itself deletable dead weight.
- Does `MGEgui`'s generator survive at all for statics/lights, or does that follow terrain out?
- Plugin load order + LAND overrides across masters/plugins: the existing parser processes files in
  order and later records win — confirm that matches engine behaviour before porting.

## Interiors (RELATED, but a CULLING problem — not this plan)

Verified: **interiors are fully resident in the scene graph, no streaming.** The geometry cache's
post-purge path DEEP-walks both `g_objRoot` and `g_pickRoot` with the distance gate disabled and
`bypassCullDeep=true`, on the explicit reasoning "An interior is bounded, so a full capture is safe
and correct" (`scenegraph_geometry_cache.cpp:2482-2504`). Designers split large interiors into
loading zones by hand; the engine has no interior streaming to hook.

The user's report — MW draws roughly 100 m of a big interior, we draw all of it and pay for it —
is consistent with what the code shows: **there is no interior view-distance cut anywhere on our
path.** The only distance input shaped like one is `mfd[51]` (`nearViewRange`), which
`forgerender.cpp:7824` documents as *stale in interiors* because the DL cull early-returns there,
and floors to 4096 as a workaround. Two-phase Hi-Z occlusion culling is the right weapon for
interior walls and already exists — measure it in a big interior before adding a distance cut.
Track separately from terrain: interiors have no terrain at all.
