-- Spell Probe (dev aid for "the light spell's hit form pulses between correct and bright squares")
--
-- WHY: spell/VFX content is the one class the Forge takeover can only see indirectly. The
-- projectile lives under WorldProjectileRoot, which the geometry cache never walks — MGE only
-- sweeps those roots per-frame to REGISTER their texture names, and the draws themselves ride the
-- AT3 captured-alpha path. The "hit" form is different: Enhanced Light's onCollision/onTick create
-- a real LGHT_OJ_EL_LightAnimated *cell reference*, so it goes through the normal cache walk.
-- Those two paths refresh texture identity on completely different schedules, and the reported
-- symptom (projectile fine, hit form strobing white quads) splits exactly along that seam. Before
-- touching C++ we need the mesh's actual structure: which shapes carry a NiFlipController (kurp's
-- meshes flip through a 300-frame `sray` book), which carry a DARK/DETAIL/GLOW map (the host's
-- captured-alpha frag samples the BASE map only), and which are alpha-blended.
--
-- WHAT: `delay` real seconds after load, spawn the hit form in front of the player exactly the way
-- the mod does (its own functions.createAnimatedLight), then dump the scene node: every NiTriShape
-- and NiParticles leaf with its texture maps, alpha/z state, and attached controllers. Hold it in
-- view for `hold` seconds so mgeXE.log's [hb] tex-residency heartbeat spans the flipbook, then say
-- "done" so the harness can stop the run.
--
-- The verdict is in mgeXE.log, not here:
--   [hb] tex residency: slots=N/890 ... thrash=T   -- T>0 => the working set does not fit and
--                                                     textures ARE drawing white (the strobe)
--   [alpha-cap] no source name for tex=...(white)  -- AT3 draw whose texture was never registered
--   [cap-diag] <name> ... stageOps=[..]            -- stage1+ != 1 => multi-map, base-only = too bright
--
-- CONFIG (Data Files/MWSE/config/SpellProbe.json):
--   enabled   false = never registers a handler. DEFAULT OFF, like GhostProbe/AutoTurn360.
--   delay     real seconds after `loaded` before spawning (let the cell finish streaming in).
--   hold      real seconds to keep it in view before declaring done.
--   radius    light radius passed to createAnimatedLight (magnitude*25 in the mod).
--   distance  how far in front of the player to place the orb.
--   height    vertical offset from the player's feet, so it sits in the middle of the view.

-- `count` > 1 is the CRASH repro, not just a stress knob. One orb kept the reverse
-- texture-name map's insert traffic low enough to be survivable; six orbs' worth of flip books
-- rehashing it under the produce worker, while MAIN reads it per captured DIP, is what turned a
-- latent data race into an access violation (garbage `[tex] not found:` names in mgeXE.log were
-- the corrupted reads). Keep a multi-orb run in the loop whenever this path changes.
local defaults = {
    enabled = false,
    delay = 8,
    hold = 30,
    radius = 400,
    distance = 220,
    height = 90,
    count = 1,
    spread = 160,
}

local cfg = mwse.loadConfig("SpellProbe", defaults)

local function log(fmt, ...)
    mwse.log("[spellprobe] " .. fmt, ...)
end

-- ---------------------------------------------------------------------------
-- Scene-node dump
-- ---------------------------------------------------------------------------
-- Property access differs per MWSE build (`node.texturingProperty` vs getProperty), and a leaf may
-- carry none at all, so every read is defensive: a missing field must degrade to "?" and never
-- abort the dump half way through the tree.
local function safeGet(fn, ...)
    local ok, v = pcall(fn, ...)
    if ok then return v end
    return nil
end

local function texturingOf(node)
    local p = safeGet(function() return node.texturingProperty end)
    if p then return p end
    return safeGet(function() return node:getProperty(ni.propertyType.texturing) end)
end

local function alphaOf(node)
    local p = safeGet(function() return node.alphaProperty end)
    if p then return p end
    return safeGet(function() return node:getProperty(ni.propertyType.alpha) end)
end

-- The map slots we care about, in NiTexturingProperty order. DARK/DETAIL/GLOW are the ones the
-- host's captured-alpha fragment shader does NOT sample — a blend that relies on them renders at
-- full base brightness, which is what "bright squares" would look like.
local mapSlots = { "baseMap", "darkMap", "detailMap", "glowMap" }

local function describeMaps(texProp)
    if not texProp then return "(no texturing property)" end
    local parts = {}
    for _, slot in ipairs(mapSlots) do
        local m = safeGet(function() return texProp[slot] end)
        local t = m and safeGet(function() return m.texture end) or nil
        local name = t and safeGet(function() return t.fileName end) or nil
        if name then
            table.insert(parts, string.format("%s=%s", slot, name))
        end
    end
    if #parts == 0 then return "(no maps)" end
    return table.concat(parts, " ")
end

-- Controllers are the whole point of the dump: a NiFlipController is what swaps the bound
-- NiSourceTexture without ever touching NiGeometryData, so the cache's revisionID-driven material
-- re-extract never fires for it. Report the flip book's SIZE too — that number is what has to fit
-- in the client's 890-slot bindless residency.
local function describeControllers(obj, label)
    local out = {}
    local c = safeGet(function() return obj.controller end)
    local guard = 0
    while c and guard < 32 do
        guard = guard + 1
        local cls = safeGet(function() return c.__type and c.__type.name end)
                 or safeGet(function() return tostring(c) end) or "?"
        local extra = ""
        local srcs = safeGet(function() return c.sources end)
        if srcs then
            local n = safeGet(function() return #srcs end)
            extra = string.format(" sources=%s", tostring(n))
        end
        local freq = safeGet(function() return c.frequency end)
        if freq then extra = extra .. string.format(" freq=%.3f", freq) end
        table.insert(out, cls .. extra)
        c = safeGet(function() return c.nextController end)
    end
    if #out == 0 then return nil end
    return label .. "{" .. table.concat(out, ", ") .. "}"
end

local shapeCount = 0

local function dumpNode(node, depth)
    if not node or depth > 12 then return end
    local name = safeGet(function() return node.name end) or "(unnamed)"
    local cls = safeGet(function() return node.__type and node.__type.name end) or "?"
    local indent = string.rep("  ", depth)

    local isLeaf = safeGet(function() return node.data ~= nil end) and cls ~= "niNode"
    if isLeaf then
        shapeCount = shapeCount + 1
        local texProp = texturingOf(node)
        local alpha = alphaOf(node)
        local aflags = alpha and safeGet(function() return alpha.flags end) or nil
        -- Bit 0 of NiAlphaProperty flags is alpha-blend enable; that bit is what routes the leaf
        -- to the alpha rules in MGE's coverage classifier.
        local blend = aflags and ((aflags % 2) == 1) or false
        local ctrl = describeControllers(node, "geomCtrl")
        local pctrl = texProp and describeControllers(texProp, "texCtrl") or nil
        log("%s%s [%s] blend=%s alphaFlags=%s", indent, name, cls, tostring(blend),
            aflags and string.format("0x%X", aflags) or "?")
        log("%s    %s", indent, describeMaps(texProp))
        if ctrl then log("%s    %s", indent, ctrl) end
        if pctrl then log("%s    %s", indent, pctrl) end
    else
        local ctrl = describeControllers(node, "nodeCtrl")
        log("%s%s [%s]%s", indent, name, cls, ctrl and (" " .. ctrl) or "")
    end

    local kids = safeGet(function() return node.children end)
    if kids then
        for _, c in pairs(kids) do
            if c then dumpNode(c, depth + 1) end
        end
    end
end

-- ---------------------------------------------------------------------------
-- Spawn + hold
-- ---------------------------------------------------------------------------
-- Spawn through the mod's OWN function, not a hand-rolled createReference: the hit form is a
-- dynamic LGHT reference with a runtime-assigned radius, and reproducing that by hand would be a
-- different object path from the one the bug was reported against.
local function spawnHitForm(index)
    local fns = include("OperatorJack.EnhancedLight.functions")
    local player = tes3.player
    if not player then log("no player"); return nil end

    -- Fan the orbs across the view rather than stacking them, so every one is separately
    -- visible and separately drawn (a stack would z-fight into one effective draw).
    local n = math.max(1, cfg.count)
    local offset = (index - (n + 1) / 2) * cfg.spread
    local facing = player.orientation.z
    local fwd = tes3vector3.new(-math.sin(facing), math.cos(facing), 0)
    local right = tes3vector3.new(math.cos(facing), math.sin(facing), 0)
    local pos = tes3vector3.new(
        player.position.x + fwd.x * cfg.distance + right.x * offset,
        player.position.y + fwd.y * cfg.distance + right.y * offset,
        player.position.z + cfg.height)

    local ref
    if fns and fns.createAnimatedLight then
        ref = fns.createAnimatedLight(pos, player.cell, cfg.radius)
    else
        -- Enhanced Light not installed / module layout changed: fall back to the raw object so the
        -- run still says something useful instead of silently doing nothing.
        log("!! OperatorJack.EnhancedLight.functions unavailable - falling back to createReference")
        local obj = tes3.getObject("LGHT_OJ_EL_LightAnimated")
        if not obj then log("!! LGHT_OJ_EL_LightAnimated missing - is Enhanced Light.ESP active?"); return nil end
        ref = tes3.createReference({ object = obj, position = pos, cell = player.cell })
    end
    return ref
end

local function runProbe()
    local ref
    for i = 1, math.max(1, cfg.count) do
        local r = spawnHitForm(i)
        if r then
            ref = ref or r
            log("spawned #%d '%s' at (%.0f,%.0f,%.0f) radius=%d", i, r.id,
                r.position.x, r.position.y, r.position.z, cfg.radius)
        end
    end
    if not ref then
        log("SPAWN FAILED - nothing to observe")
        log("done sampling")
        return
    end

    -- One frame later: the scene node exists but MW has not necessarily instantiated every
    -- controller target yet on the spawn frame itself.
    timer.delayOneFrame(function()
        local node = ref.sceneNode
        if not node then log("!! no scene node"); return end
        log("--- scene node dump for %s ---", ref.id)
        shapeCount = 0
        dumpNode(node, 0)
        log("--- %d leaf shapes ---", shapeCount)
    end)

    timer.start({
        type = timer.real, duration = cfg.hold, iterations = 1,
        callback = function()
            log("held %d s; check mgeXE.log for [hb] tex residency / [alpha-cap] / [cap-diag]", cfg.hold)
            log("done sampling")
        end,
    })
end

local function onLoaded()
    if not cfg.enabled then
        log("disabled (enabled=false)")
        return
    end
    local cell = tes3.getPlayerCell()
    log("loaded into '%s' - spawning the hit form in %d s",
        cell and (cell.editorName or cell.id) or "(unknown cell)", cfg.delay)
    timer.start({ type = timer.real, duration = cfg.delay, iterations = 1, callback = runProbe })
end

event.register("loaded", onLoaded)
