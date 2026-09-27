--- ControllerProbe — spawn NIF-controller test cases in front of the player.
---
--- Why this exists: verifying a controller migration means finding a mesh that exercises
--- exactly one controller type and then WALKING to it. The good vanilla cases are in
--- Bloodmoon and Mournhold. This puts them at arm's length instead.
---
--- Why real references and not a hand-attached scene node: `tes3.loadMesh():clone()` +
--- attachChild gives you a node with NO TES3 reference, so the geometry cache's
--- referenceLiveKind() returns None and the entry classifies differently from the thing
--- that actually broke. Spawning a real object id exercises the same capture path the bug
--- lives on. A test that dodges the code under test is worse than no test.
---
--- Keys (raw DirectInput scan codes, so there is no scanCode-name guessing):
---   Numpad 1  cycle to the next case (announces it; does not spawn)
---   Numpad 2  spawn the current case in front of you
---   Numpad 3  delete everything this mod spawned
---
--- Numpad 1/2/3 are chosen because MGE itself already claims most of the keypad and a
--- collision here is not harmless: this mod's first draft used Numpad 8, which is (was)
--- MGE's produce-worker mode cycle, and one press moved it to OFF (inline) — a mode that
--- deadlocks the async frame-split and freezes the image with the game still running.
--- MGE currently claims numpad 0, 4, + - * / . and Scroll Lock; 1/2/3/5/6/7 are free.
--- Check renderprocess.cpp's GetAsyncKeyState block before adding another binding.
---
--- Spawned refs are auto-deleted on cell change and on load, so nothing litters the save.

local KEY_CYCLE = 0x4F -- Numpad 1
local KEY_SPAWN = 0x50 -- Numpad 2
local KEY_CLEAR = 0x51 -- Numpad 3
local KEY_DRIVE = 0x4C -- Numpad 5 (MGE claims Numpad 4; 5 is free)

local SPAWN_DISTANCE = 300 -- units in front of the player

--- Each case names the controller it isolates, what a WORKING render looks like, and what
--- the FROZEN-at-capture failure looks like. The failure line is the point: "looks a bit
--- dull" is not a verdict, "never flickers once in ten seconds" is.
--- Ordered best-instrument-first. Three rules govern selection here, two of them learned
--- the hard way after picking cases that produced no readable result:
---
--- 1. THE ROOT MUST BE NiBSAnimationNode. This is what makes MW start and tick a static's
---    controllers at all. Under a plain NiNode root the controllers are never started, so
---    the mesh sits at its authored material values forever - in the DX9 path and the
---    Forge path alike. magic_reflect.nif and magic_area_drain.nif have plain roots, which
---    is why they animated in neither.
--- 2. BASE MATERIAL ALPHA MUST NOT BE 0.0. Several VFX meshes are authored fully
---    transparent and rely on the controller to reveal them (magic_reflect: 0.000 on BOTH
---    materials). Combined with rule 1 that renders as nothing at all - and "frozen at
---    capture" then looks exactly like "failed to spawn", which is a useless instrument.
---    Starting at 1.0 means frozen = visible-and-constant, which is unmistakable.
--- 3. The curve must REACH 0.00, so working = blinks fully out. A 0.20..0.60 wobble is
---    parity-checkable but not eyeball-checkable (see the forcefield, last).
---
--- Placement matters too: the area VFX are flat horizontal discs authored at ground
--- level, so they need zOffset to clear the floor or they read as "did not display".
---
--- All the VFX_* ids are ordinary STAT records - Morrowind stores its magic effect meshes
--- as plain statics, so there is no gameplay state to be "off" and no activation to wait
--- for. Verify a candidate with scratchpad matalpha.py before adding it here.
--- Our own fixture, and the only case here designed rather than found. Built by
--- tools/make-alpha-probe-nif.py from bm_forcefield.nif -- the one mesh proven to render
--- AND animate as a spawned static in this setup -- with only float values rewritten in
--- place (same byte count, so the NIF cannot be structurally malformed). Base alpha 1.0,
--- emissive white, all three controllers synced to one 4s cycle that holds fully opaque,
--- snaps fully invisible, holds, and snaps back.
---
--- Registered at runtime via tes3.createObject, so this needs no ESP -- just the loose
--- mesh under Data Files/meshes/.
local PROBE_ID = "mge_alphaprobe"
local PROBE_MESH = [[mgeprobe\alphaprobe.nif]]

local CASES = {
    {
        id = PROBE_ID,
        custom = true,
        mesh = "mgeprobe/alphaprobe.nif (built by tools/make-alpha-probe-nif.py)",
        dist = 320,
        controllers = "NiAlphaController x3, NiUVController x3, NiKeyframeController x1",
        watch = "A big glowing sheet on a FOUR SECOND cycle: solid for ~1.5s, then gone "
             .. "completely for ~1s, then back. Slow enough to count out loud.",
        broken = "A sheet that is permanently there and never once disappears. It starts "
              .. "at alpha 1.0 and is self-illuminated, so frozen-at-capture is always "
              .. "VISIBLE-and-constant - it can never be confused with a failed spawn, "
              .. "which is what sank every vanilla candidate.",
    },
    {
        id = "VFX_DestructArea",
        mesh = "e/magic_area_dst.nif",
        dist = 180,
        zOffset = 60,
        controllers = "NiAlphaController x2, NiMaterialColorController x1, "
                   .. "NiGeomMorpherController x1, NiKeyframeController x5, NiUVController x1",
        watch = "A flat disc that blinks FULLY OUT every 0.87s: 1.00 -> 1.00 -> 0.00, "
             .. "looping. Base material alpha is 1.0 on all five materials, so it is "
             .. "plainly visible the moment it spawns.",
        broken = "A disc that just sits there at one opacity and never blinks out. "
              .. "Because it starts at 1.0, frozen means VISIBLE-AND-CONSTANT, which is "
              .. "unmistakable next to a 0.87s blink - unlike a mesh authored at 0.0, "
              .. "where frozen and 'failed to spawn' look identical.",
    },
    {
        id = "VFX_DefaultArea",
        mesh = "e/magic_area.nif",
        dist = 180,
        zOffset = 60,
        controllers = "NiAlphaController x2, NiMaterialColorController x1, "
                   .. "NiGeomMorpherController x1, NiKeyframeController x5, NiUVController x1",
        watch = "0.87s loop ending at 0.00. Also the ONLY case here that exercises "
             .. "NiMaterialColorController - the animated material TINT, which rides the "
             .. "same matAnimated flag as alpha and is otherwise untested.",
        broken = "No blink (alpha) and/or a colour that never shifts (material colour).",
    },
    {
        id = "bm_aesliipforcefield",
        mesh = "f/bm_forcefield.nif",
        controllers = "NiAlphaController x3, NiUVController x3, NiKeyframeController x1",
        watch = "Steady 0.20 for EIGHT seconds, then a flicker (0.40/0.30/0.60) in the "
             .. "last 0.9s.",
        broken = "PARITY CHECK ONLY - do not try to read a verdict off this one. The swing "
              .. "is 0.20..0.60, which is too subtle to judge by eye; it is here because "
              .. "it should look IDENTICAL under F11 either way.",
    },
}

local current = 1
local spawned = {}

local function log(fmt, ...)
    mwse.log("[ctrlprobe] " .. fmt, ...)
end

local function describe(case)
    return string.format("%s  (%s)", case.id, case.mesh)
end

local function clearSpawned(quiet)
    local n = 0
    for i = #spawned, 1, -1 do
        local ref = spawned[i]
        -- The ref can already be gone (cell unloaded, player deleted it): guard rather
        -- than trust the list.
        if ref and not ref.deleted then
            ref:disable()
            ref:delete()
            n = n + 1
        end
        spawned[i] = nil
    end
    if n > 0 then
        log("cleared %d spawned reference(s)", n)
        if not quiet then
            tes3.messageBox("ControllerProbe: cleared %d", n)
        end
    end
    return n
end

--- Register the custom fixture's STAT record on first use. Done lazily rather than at
--- "initialized" because createObject needs the data handler up. Idempotent: an existing
--- record (created earlier this session, or restored from the save) is reused.
local function ensureProbeObject()
    local existing = tes3.getObject(PROBE_ID)
    if existing then return existing end
    local ok, obj = pcall(tes3.createObject, {
        objectType = tes3.objectType.static,
        id = PROBE_ID,
        mesh = PROBE_MESH,
    })
    if not ok or not obj then
        log("FAILED to register '%s' (mesh %s): %s", PROBE_ID, PROBE_MESH, tostring(obj))
        tes3.messageBox("ControllerProbe: could not register '%s' - is Data Files\\meshes\\%s present?",
                        PROBE_ID, PROBE_MESH)
        return nil
    end
    log("registered static '%s' -> %s", PROBE_ID, PROBE_MESH)
    return obj
end

-- ---------------------------------------------------------------------------------------
-- Lua-driven alpha (Numpad 5) — the test that does not depend on MW's controllers.
--
-- Established the hard way: MW does not reliably run NiAlphaControllers on a spawned
-- static. It ran the UV scroll on ONE of the probe's three layers and no alpha at all, so
-- MW's own render never blinked either — which means there was nothing for the host to
-- diverge FROM, and no authoring of the curve could have produced a verdict.
--
-- So drive NiMaterialProperty::alpha directly from Lua instead. That is precisely the
-- value the fix re-reads each frame (refreshAnimatedMaterial), so this tests the changed
-- code path end to end with no engine machinery in between:
--   MW's own render follows it (fixed function reads the property at draw time)
--   the host follows it ONLY if the per-frame material refresh works
-- Toggle F11 while it pulses: both sides pulsing = fixed. Host frozen while MW pulses =
-- the bug, reproduced deliberately and on demand.
--
-- The scene node is re-read from the reference EVERY frame and never latched. MWSE hands
-- out NI objects as raw pointers with no addref, so holding one across frames is a
-- use-after-free waiting for the cell to unload.
local driving = false
local driveTime = 0
local driveLogAt = 0

local function collectMaterials(node, out)
    if not node then return out end
    local mp = node.materialProperty
    if mp then out[#out + 1] = mp end
    local kids = node.children
    if kids then
        for i = 1, #kids do
            local c = kids[i]
            if c then collectMaterials(c, out) end
        end
    end
    return out
end

local function driveAlpha(e)
    if not driving then return end
    driveTime = driveTime + (e.delta or 0)
    -- 0.10 .. 1.00 at roughly half a hertz: slow enough to watch, deep enough that a
    -- frozen host is obvious next to a pulsing MW.
    local a = 0.10 + 0.90 * (0.5 + 0.5 * math.sin(driveTime * 3.0))
    local touched = 0
    for i = 1, #spawned do
        local ref = spawned[i]
        if ref and not ref.deleted then
            for _, m in ipairs(collectMaterials(ref.sceneNode, {})) do
                m.alpha = a
                touched = touched + 1
            end
        end
    end
    if driveTime - driveLogAt >= 1.0 then
        driveLogAt = driveTime
        log("driving alpha=%.2f across %d material(s) on %d reference(s)",
            a, touched, #spawned)
        if touched == 0 then
            log("  NOTHING TOUCHED - spawn a case first (Numpad 2), or the scene node has "
                .. "no NiMaterialProperty to drive")
        end
    end
end

local function toggleDrive()
    driving = not driving
    driveTime, driveLogAt = 0, 0
    log("Lua alpha drive %s", driving and "ON" or "OFF")
    if driving then
        tes3.messageBox("Alpha drive ON - MW should pulse. Toggle F11: if the host stays "
                     .. "frozen while MW pulses, the per-frame material refresh is broken.")
    else
        tes3.messageBox("Alpha drive OFF")
    end
end

local function spawnCurrent()
    local case = CASES[current]
    local player = tes3.player
    if not player then return end

    if case.custom and not ensureProbeObject() then return end

    -- Forward from the player's yaw. MW yaw 0 faces +Y, so forward = (sin, cos, 0).
    -- Per-case distance: the VFX_* meshes are sized to wrap a caster, so they read badly
    -- at wall range; a forcefield sheet needs the room. Falls back to the default.
    local dist = case.dist or SPAWN_DISTANCE
    -- zOffset lifts a case clear of the floor. The area VFX are FLAT HORIZONTAL DISCS
    -- authored at ground level, so spawning one at the player's own z puts it coplanar
    -- with the floor, where it reads as "did not display" rather than "is in the floor".
    -- A tall vertical sheet (the forcefield) needs no lift, hence per-case.
    local yaw = player.orientation.z
    local p = player.position
    local pos = tes3vector3.new(
        p.x + math.sin(yaw) * dist,
        p.y + math.cos(yaw) * dist,
        p.z + (case.zOffset or 0))

    local params = {
        object = case.id,
        position = pos,
        -- +pi so the sheet's front faces back at the player rather than away.
        orientation = tes3vector3.new(0, 0, yaw + math.pi),
    }
    -- The cell argument is only required for interiors (tes3.createReference docs).
    local cell = player.cell
    if cell and cell.isInterior then
        params.cell = cell
    end

    local ok, ref = pcall(tes3.createReference, params)
    if not ok or not ref then
        -- Almost always "that object id is not in the loaded masters" — e.g. asking for a
        -- Bloodmoon activator with Bloodmoon.esm not loaded. Say which, plainly.
        log("FAILED to spawn '%s': %s", case.id, tostring(ref))
        tes3.messageBox("ControllerProbe: could not spawn '%s' (master not loaded?)", case.id)
        return
    end

    table.insert(spawned, ref)
    log("spawned %s at (%.0f, %.0f, %.0f) in %s",
        describe(case), pos.x, pos.y, pos.z, cell and cell.editorName or "?")
    log("  controllers: %s", case.controllers)
    log("  WORKING:     %s", case.watch)
    log("  BROKEN:      %s", case.broken)
    tes3.messageBox("Spawned %s\n\nWORKING: %s\n\nBROKEN: %s",
                    case.id, case.watch, case.broken)
end

local function cycleCase()
    current = current % #CASES + 1
    local case = CASES[current]
    log("selected [%d/%d] %s", current, #CASES, describe(case))
    tes3.messageBox("ControllerProbe [%d/%d]: %s\n%s",
                    current, #CASES, case.id, case.controllers)
end

event.register("keyDown", spawnCurrent, { filter = KEY_SPAWN })
event.register("keyDown", cycleCase, { filter = KEY_CYCLE })
event.register("keyDown", function() clearSpawned(false) end, { filter = KEY_CLEAR })
event.register("keyDown", toggleDrive, { filter = KEY_DRIVE })
event.register("simulate", driveAlpha)

-- Spawned test objects are real world references and would otherwise persist into the
-- save. Drop them the moment they stop being useful.
event.register("cellChanged", function() clearSpawned(true) end)
event.register("loaded", function()
    for i = #spawned, 1, -1 do spawned[i] = nil end
end)

event.register("initialized", function()
    log("ready - %d cases. Numpad1 cycle / Numpad2 spawn / Numpad3 clear / Numpad5 alpha-drive.",
        #CASES)
    log("selected [1/%d] %s", #CASES, describe(CASES[1]))
end)
