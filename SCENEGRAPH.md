# Morrowind Scenegraph Structure

## Top-Level Scene Graphs

The engine root NiNode "Morrowind World Scene Graph" has 6 child scene graphs:

| Index | Name |
|-------|------|
| 0 | Morrowind World Scene Graph |
| 1 | Morrowind Arm Scene Graph |
| 2 | Morrowind Menu Scene Graph |
| 3 | Morrowind Inventory Scene Graph |
| 4 | Morrowind Map Scene Graph |
| 5 | Morrowind Shadow Scene Graph |

**Important**: The Arm Scene Graph is a completely separate top-level sibling — it is NOT what DataHandler::worldPickObjectRoot (offset 0x90) points to.

---

## World Scene Graph (index 0) Internal Structure

```
NiBSAnimationManager "World Scene Graph Root"
├── NiNode "skyRoot"
└── NiNode "worldRoot"
    ├── Properties: NiFogProperty, NiVertexColorProperty, NiWireframeProperty
    ├── NiNode "Precipitation Rain Root"  [Culled]
    ├── NiNode "Precipitation Snow Root"  [Culled]
    ├── NiNode "Storm Root"
    ├── NiNode "WorldProjectileRoot"
    ├── NiNode "WorldObjectRoot"          ← DataHandler+0x8C
    ├── NiNode "WorldPickObjectRoot"      ← DataHandler+0x90
    ├── NiNode "WorldLandscapeRoot"       ← DataHandler+0x94
    ├── NiNode "WorldSpellRoot"
    ├── NiNode "WorldVFXRoot"
    └── NiNode "Water Node"
```

The DataHandler offsets are pointers into `worldRoot`'s children, NOT into the top-level 6 graphs.

---

## WorldObjectRoot Structure

```
NiNode "WorldObjectRoot"
├── NiAmbientLight "activation ambientLight"
├── NiNode "CLONE PlayerSaveGame"  [Culled]
├── NiBSAnimationManager "Cell 'Bitter Coast Region' (-3, -10)"
│   ├── NiNode "CLONE terrain_rock_bc_18"  [worldTranslate, worldScale]
│   │   └── NiTriShape "Tri Terrain_rock_BC_18 0"
│   │       ├── Attributes: worldTranslate, worldRotate, worldScale, worldBound
│   │       └── Properties: NiTexturingProperty, NiMaterialProperty
│   └── ... (more CLONE nodes per object reference)
└── NiBSAnimationManager "Cell ..."  (one per loaded cell)
```

Objects are grouped: each loaded cell has a **NiBSAnimationManager** node. Under that, each placed object instance is a **NiNode** named "CLONE \<base-id\>". The actual geometry sits inside as one or more **NiTriShape** children.

Objects that can be picked (ingredients, containers, creatures, NPCs, signposts with scripts) are in WorldObjectRoot. All statics and animated statics are here too. Picking/selection does not remove objects from the world — WorldPickObjectRoot is a separate structure for raycasting.

---

## WorldPickObjectRoot

Inside the world's `worldRoot`, sibling to WorldObjectRoot and WorldLandscapeRoot. Holds pickable/interactive world objects (ingredients, weapons, books, containers, etc.) — geometry that can be activated or picked up by the player. These objects do NOT appear in WorldObjectRoot.

The Arm Scene Graph is a completely separate top-level graph at index 1 — it has nothing to do with this node.

**Now walked** by the geometry cache (`onFrameReady`).

---

## WorldLandscapeRoot Structure

```
NiNode "WorldLandscapeRoot"
└── NiNode "LAND (-3, -10)"  [at (-24576, -81920, 0)]
    └── NiNode "(null)"  [at (1024, 1024, 0)]
        ├── NiTriShape "(null)"  [NiTexturingProperty: slot 0 + slot 6]
        ├── NiTriShape "(null)"
        ├── ...  (many per cell, typically 16–36 terrain patches)
        └── NiTriShape "(null)"  [NiTexturingProperty: slot 0 only]
```

Terrain NiTriShapes are always named "(null)". They hold:
- **Slot 0**: base terrain texture (m_uiTexCoord = 0, UV set 0)
- **Slot 6**: decal/blend overlay from adjacent patch (present on some shapes)

**Terrain uniqueID is 0** — all terrain NiTriShapes have `GeometryData::uniqueID == 0`. The geometry cache `if (!uid) return;` guard skips all terrain. This is why exterior depth is entirely missing.

---

## NiTriShape Attributes (object example: "Tri Ingred_Comberry_01 0")

| Field | Value |
|-------|-------|
| m_bAppCulled | false (visible) |
| m_worldTranslate | (35.509, -114.273, -60.299) — world-space |
| m_worldRotate | 3×3 rotation matrix |
| m_worldScale | 1.19 |
| m_kWorldBound | center + radius (NI::Bound) |

UV: `m_uiTexCoord = 0` → UV set 0 in `NiGeometryData::textureCoords`. The highlighted field in the screenshot confirms UV set 0 is the correct index to read from `data->textureCoords`.

NiTexturingProperty on objects:
- Slot 0: base texture (CLAMP_S_CLAMP_T or WRAP, FILTER_TRILERP)
- Slot 6: terrain decal only (not on regular objects)

---

## Key Findings for Geometry Cache

### 1. Terrain uniqueID == 0 → exterior totally missing
All terrain NiTriShapes have `GeometryData::uniqueID == 0` and are skipped by `if (!uid) return;` in visitGeometry. The terrain walk in `onFrameReady` registers texture names via `registerAllMaps` but never populates VBs. Exterior scenes are mostly terrain → depth buffer is nearly empty outside.

**Fix**: Cache is now keyed by `NiTriShape*` pointer (cast to uint32_t on x86) — no more uid==0 skip. Each placed instance is a distinct node in the tree so pointer identity is unique and stable.

### 2. WorldObjectRoot objects also have uniqueID == 0
All Morrowind geometry data has uniqueID == 0. The uid-based key has been replaced by the NiTriShape pointer key.

### 3. WorldPickObjectRoot purpose unclear
It's inside the world's worldRoot, not the Arm Scene Graph. Do not walk it in the geometry cache until its contents are confirmed — could contain pick proxies (AABBs, spheres) not suitable for depth rendering.

### 4. Cell structure
Objects are under NiBSAnimationManager "Cell '...'" nodes. These are NiNode children of WorldObjectRoot. The geometry cache walk handles this correctly because `walk()` recurses through all NiNode children.

### 5. DataHandler offset summary

| Field | Offset | Points to |
|-------|--------|-----------|
| worldObjectRoot | 0x8C | NiNode "WorldObjectRoot" inside worldRoot |
| worldPickObjectRoot | 0x90 | NiNode "WorldPickObjectRoot" inside worldRoot |
| worldLandscapeRoot | 0x94 | NiNode "WorldLandscapeRoot" inside worldRoot |

The Arm Scene Graph is the top-level "Morrowind Arm Scene Graph" (index 1) — none of the DataHandler offsets point to it directly.
