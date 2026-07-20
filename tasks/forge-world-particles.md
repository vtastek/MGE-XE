# World particle takeover + MW world-traversal suppression

Branch: `forge-leadin-shrink`. **Status: DEPRIORITISED — measured payoff ~0.75ms, see below.**

## What suppression is

`GeometryCache::applyWorldSuppression` sets `appCulled` on MW's world roots so the ENGINE
stops traversing them. Phase 2 stopped MGE drawing and the proxy reject gate drops MW's
draws, but MW still walked its whole scene graph and issued every call first. Controlled by
`DistantLand::mwWorldSuppress`, a **bitmask of independent bits**
(`kSuppressLand|Pick|Objects`), set from the Forge Dev imgui panel — no key, the keyspace is
full.

Independent bits, NOT a ladder: a cumulative level cannot attribute a symptom to a root.
"Smoke vanished at level 2" only ever *implied* pick, because level 2 culled landscape AND
pick together. That inference was presented as a measurement once; don't repeat it.

## What breaks, and why (measured 2026-07-20)

| root suppressed | lost |
|---|---|
| landscape | nothing visible |
| pick objects | smoke / particle flames |
| world objects | fireplace fire (also alpha-blended **particles**) |

Every lost item is **alpha-blended particles**, and every one reaches the host only because
`captureAlphaDraw` (distantland.cpp:1671) intercepts a draw MW really issued — MW's particle
renderer generates the billboarded quads *at draw time*
(scenegraph_geometry_cache.cpp:1796), so the cache walk never sees them. No traversal ⇒ no
draw ⇒ no capture ⇒ absent. There is no ordering trick that recovers it.

Corrections to earlier readings in this file:
- The fireplace fire is **particles, not a NiTriShape** — so there is no "capture it in the
  walk instead" shortcut for it. It is the same problem as the smoke.
- Roots are not scenes. A pick object with an alpha-blended part is sorted and drawn in the
  scene-1 sorted pass like anything else; pick objects render in the normal scenes.

## Why it is deprioritised: zone time is not frame time

```
mask 5 (land+obj suppressed, pick RENDERING):  dt=13.01  feed=4.23  build=3.18  host=8.59  render=0.17
mask 7 (+pick suppressed):                     dt=12.26  feed=3.31  build=2.42  host=7.37  render=0.05
```

- `mwsky` fell **3ms -> 0.5ms** when pick was suppressed, but **dt moved only 13.01 -> 12.26**.
  ~2.5ms of zone time produced ~0.75ms of frame time. This matches the long-standing note
  that the `MW sky` zone (BeginScene(0) -> first world draw) absorbs a GPU touch rather than
  CPU work — see the `MWsky:DIP` sub-zone at mged3d8device:1060, added to split exactly this.
  **Do not price work off a Tracy zone shrinking.** (Caveat: one hb window at mask 7 vs five
  at mask 5.)
- `mwdraws` stayed **2.1ms with ALL THREE roots suppressed**. Whatever costs 2.1ms there is
  NOT the world roots — it lives in the worldRoot siblings we never suppress
  (precipitation/storm/projectile/spell/VFX, sky), or it is another wait wearing a zone name.
  Identify that before assuming the draw loop is attackable at all.

So the whole suppression line buys ~0.75ms and costs smoke + sorted alpha. The world particle
takeover below is what would let us keep the content — but it is now priced against 0.75ms,
not the ~5ms the zone times suggested.

## The takeover, if it is ever worth doing

Generate the particles client-side instead of intercepting MW's draws. `buildFPParticleQuads()`
(scenegraph_geometry_cache.cpp:1328) already does this for first person against the arm camera
basis (`tasks/forge-fp-particles.md`); the world equivalent needs the same against
`MWBridge::getWorldCamera()` (added 2026-07-20 for the billboard re-face — the `wc+0x124`
counterpart of `getArmCamera`).

## Verify

Toggle ONE root at a time in the Forge Dev panel. Weather/VFX/projectile/spell roots are
worldRoot siblings and are never suppressed, so ashstorms/rain/spell effects must be
unaffected in every combination.

## Landed alongside (independent, keep)

**Billboard walk-order fix** (commit `edd711d`). World `NiBillboardNode`s shipped the PREVIOUS
frame's facing: our cache walk runs at BeginScene(0), MW's cull pass — where a billboard
re-faces itself — runs later in scene 0, and the host draws what we captured. Proven with only
landscape suppressed, where MW still traversed `objRoot` and re-faced normally and world flames
were stale anyway; FP never showed it because FP1c already re-faced explicitly. Fixed by gating
that re-face on `ownsOpaqueWorld()` rather than on suppression. **Cherry-pick to `forge`** — it
is a real bug there too.

**Lights-walk root bypass.** `SceneGraph::runWalk` starts AT objRoot/pickRoot and `walk()`
early-returns on `appCulled`, so suppression blanked the point-light snapshot ("only the near
lights changed" — distant baked lights come from the host froxel path). It now bypasses the
root's own flag for roots we suppressed, testing the REQUESTED mask: that walk runs on its own
worker thread and can fire before the flags go up or after Present clears them, so the applied
value is not race-free.
