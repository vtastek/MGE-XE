-- Auto Zip (dev aid: exterior cell-change STRESS for the Forge feed — texture prefetch/eviction,
-- geometry purge + release backlog, host residency; tasks/forge-pipeline-depth.md)
--
-- WHY: AutoWalk crosses a handful of borders at walking pace. This hammers the two grid events as
-- fast as MW will do them, hundreds of times, so leaks, backlogs and races that need volume show up:
--   FAR   tes3.positionCell to a random exterior cell anywhere on the map: MW's loading path (the
--         same one a door or fast travel takes) — whole grid replaced, client purge + post-load window.
--   GLIDE 1-3 cells in a random direction at `speed` u/s, position written every frame: the grid
--         SLIDES a row at a time, faster than any player can run.
-- Deterministic: every choice comes from math.random seeded with `seed`, so a run repeats exactly.
--
-- Ground truth: every cellChanged MW fires is counted and logged; the run ends at `target` changes.
-- God mode is switched on (tgm) so a random landing in a fight or a long fall cannot end the run.
--
-- CONFIG (Data Files/MWSE/config/AutoZip.json) — written by mgeHost64/forge-zip-run.sh, removed on
-- exit. DEFAULT OFF: it takes the controls away and teleports the player around the map.
--   enabled, delay (real s before the first leg), target (cell changes), seed,
--   speed (glide u/s), farChance (0..1 per leg), pause (real s between legs), settle (real s at end).
--
-- mode = "hop": the DOOR HOP instead (user: "I like fast load of last cell, interior<->exterior").
-- `hops` round trips of: exterior spot -> `interior` (dwell s) -> back to the SAME exterior spot
-- (dwell s). Each leg is logged with the real seconds positionCell took, so the return trip's cost
-- can be compared across builds; mgeXE.log carries the texture/geometry side.

local defaults = {
    enabled = false,
    mode = "zip",
    delay = 12.0,
    target = 500,
    seed = 1,
    speed = 16384.0,
    farChance = 0.4,
    pause = 0.5,
    settle = 20.0,
    hops = 10,
    dwell = 4.0,
    interior = "Seyda Neen, Census and Excise Office",
}

local cfg = mwse.loadConfig("AutoZip", defaults)
local kCell = 8192.0
local kMaxLegs = 5000   -- a run that stops producing cell changes must still end

local function log(fmt, ...)
    mwse.log("[autozip] " .. fmt, ...)
end

local running = false
local changes, legs, farLegs, glideLegs = 0, 0, 0, 0
local legKind = "-"
local exteriors = {}      -- { {x, y}, ... } every exterior cell record
local lastZ = 0.0
-- glide state
local gliding = false
local gStart, gDirX, gDirY, gHeading, gDist, gWalked = nil, 0.0, 0.0, 0.0, 0.0, 0.0
local snapPending = false
local t0 = 0.0

local function groundZ(x, y)
    local hit = tes3.rayTest({
        position = tes3vector3.new(x, y, 40000),
        direction = tes3vector3.new(0, 0, -1),
        root = tes3.game.worldLandscapeRoot,
        maxDistance = 80000,
    })
    if hit and hit.intersection then
        lastZ = math.max(hit.intersection.z, 0.0)
    end
    return lastZ
end

local nextLeg
local onSimulate

local function finish(reason)
    if not running then return end
    running = false
    gliding = false
    event.unregister("simulate", onSimulate)
    log("ZIP DONE (%s): %d cell changes in %d legs (%d far, %d glide), %.0fs real",
        reason, changes, legs, farLegs, glideLegs, os.difftime(os.time(), t0))
    timer.start({ type = timer.real, duration = cfg.settle, iterations = 1,
        callback = function() log("SETTLED: %.1fs after the run", cfg.settle) end })
end

local function schedule()
    if not running then return end
    if changes >= cfg.target then finish("target reached") return end
    if legs >= kMaxLegs then finish("!! leg cap hit - cell changes stopped arriving") return end
    timer.start({ type = timer.real, duration = math.max(cfg.pause, 0.001), iterations = 1, callback = nextLeg })
end

onSimulate = function(e)
    if not running then return end
    if snapPending then
        -- The far jump lands in the air at a guessed height; the cells are loaded now, so drop the
        -- player onto the real ground before the pause starts.
        snapPending = false
        local p = tes3.player.position
        tes3.player.position = tes3vector3.new(p.x, p.y, groundZ(p.x, p.y))
        schedule()
        return
    end
    if not gliding then return end
    gWalked = math.min(gWalked + cfg.speed * e.delta, gDist)
    local x = gStart.x + gDirX * gWalked
    local y = gStart.y + gDirY * gWalked
    tes3.player.position = tes3vector3.new(x, y, groundZ(x, y))
    tes3.player.orientation = tes3vector3.new(0, 0, gHeading)
    if gWalked >= gDist then
        gliding = false
        schedule()
    end
end

nextLeg = function()
    if not running then return end
    legs = legs + 1
    if #exteriors > 0 and math.random() < cfg.farChance then
        farLegs = farLegs + 1
        legKind = "far"
        local c = exteriors[math.random(#exteriors)]
        local x = c[1] * kCell + kCell * 0.5
        local y = c[2] * kCell + kCell * 0.5
        local face = math.random() * 2 * math.pi
        lastZ = 0.0
        local ok, err = pcall(tes3.positionCell, {
            reference = tes3.player,
            position = { x, y, 12000 },
            orientation = { 0, 0, face },
            suppressFader = true,
            teleportCompanions = false,
        })
        if not ok then log("!! positionCell to (%d,%d) failed: %s", c[1], c[2], tostring(err)) end
        snapPending = true   -- onSimulate snaps to the ground, then schedules the next leg
    else
        glideLegs = glideLegs + 1
        legKind = "glide"
        gStart = tes3.player.position:copy()
        gHeading = math.random() * 2 * math.pi
        gDirX, gDirY = math.sin(gHeading), math.cos(gHeading)
        gDist = kCell * math.random(1, 3)
        gWalked = 0.0
        lastZ = gStart.z
        gliding = true
    end
end

local onDoorArrived   -- door mode (below)

local function onCellChanged(e)
    if not running then return end
    if cfg.mode == "door" then onDoorArrived(e) return end
    local c = e.cell
    if c and c.isInterior then return end
    changes = changes + 1
    log("#%d %-5s -> (%s,%s) %s", changes, legKind,
        c and tostring(c.gridX) or "?", c and tostring(c.gridY) or "?", c and c.editorName or "?")
end

local function begin()
    if tes3.player.cell.isInterior then
        log("!! player is in an interior - start from an exterior save")
        return
    end
    for _, c in pairs(tes3.dataHandler.nonDynamicData.cells) do
        if not c.isInterior then exteriors[#exteriors + 1] = { c.gridX, c.gridY } end
    end
    math.randomseed(cfg.seed)
    tes3.runLegacyScript({ command = "tgm" })
    changes, legs, farLegs, glideLegs = 0, 0, 0, 0
    t0 = os.time()
    running = true
    event.register("simulate", onSimulate)
    log("ZIP START: target %d cell changes, seed %d, %d exterior cells, glide %.0f u/s, far %.0f%%, pause %.2fs",
        cfg.target, cfg.seed, #exteriors, cfg.speed, cfg.farChance * 100, cfg.pause)
    nextLeg()
end

-- ---- DOOR HOP mode ------------------------------------------------------------------------------
local hopHome, hopHomeRot, hopInPos = nil, nil, nil
local hopsDone = 0

local function hopLeg(inside)
    if not running then return end
    local t = os.clock()
    local ok, err
    if inside then
        ok, err = pcall(tes3.positionCell, { reference = tes3.player, cell = cfg.interior, position = hopInPos,
                                             suppressFader = true, teleportCompanions = false })
    else
        ok, err = pcall(tes3.positionCell, { reference = tes3.player, position = hopHome, orientation = hopHomeRot,
                                             suppressFader = true, teleportCompanions = false })
    end
    local cpu = os.clock() - t
    if not ok then
        log("!! positionCell (%s) failed: %s", inside and "in" or "out", tostring(err))
        finish("positionCell failed")
        return
    end
    if not inside then hopsDone = hopsDone + 1 end
    log("hop %d %s: positionCell %.0f ms cpu -> %s", hopsDone + (inside and 1 or 0), inside and "IN " or "OUT",
        cpu * 1000, tes3.player.cell and tes3.player.cell.editorName or "?")
    if not inside and hopsDone >= cfg.hops then
        finish("hops done")
        return
    end
    timer.start({ type = timer.real, duration = math.max(cfg.dwell, 0.001), iterations = 1,
        callback = function() hopLeg(not inside) end })
end

local function beginHop()
    if tes3.player.cell.isInterior then
        log("!! player is in an interior - start from an exterior save")
        return
    end
    local cell = tes3.getCell({ id = cfg.interior })
    if not cell then
        log("!! interior '%s' not found", cfg.interior)
        return
    end
    -- Land on the first reference in the room (a spot inside it), lifted clear of the floor.
    hopInPos = { 0, 0, 0 }
    for ref in cell:iterateReferences() do
        local p = ref.position
        hopInPos = { p.x, p.y, p.z + 64 }
        break
    end
    hopHome = tes3.player.position:copy()
    hopHomeRot = tes3.player.orientation:copy()
    tes3.runLegacyScript({ command = "tgm" })
    hopsDone = 0
    t0 = os.time()
    running = true
    event.register("simulate", onSimulate)   -- finish() unregisters it; nothing else runs in it here
    log("HOP START: %d round trips to '%s' (land at %.0f,%.0f,%.0f), dwell %.1fs", cfg.hops, cfg.interior,
        hopInPos[1], hopInPos[2], hopInPos[3], cfg.dwell)
    hopLeg(true)
end

-- ---- REAL DOOR mode -----------------------------------------------------------------------------
-- mode = "door": `hops` round trips through the actual DOOR references (user, 2026-10-03: "small
-- interior, in and out, always loading bar and fade"). The hop above uses positionCell with the fader
-- suppressed, which is not the path a player takes: activating a door runs MW's fade and its loading
-- menu. Here the player activates the exterior door whose destination is `interior`, then the
-- interior door that leads back out. Each leg logs the real ms from the activate to cellChanged;
-- mgeXE.log's `[loadbar] up` says whether MW raised its loading menu for that leg.
local doorLegInside, doorLegT = false, 0.0

local function findDoor(cells, wantInterior)
    for _, cell in ipairs(cells) do
        for ref in cell:iterateReferences(tes3.objectType.door) do
            local d = ref.destination
            if d and d.cell and not ref.disabled then
                if wantInterior and d.cell.id == cfg.interior then return ref end
                if not wantInterior and not d.cell.isInterior then return ref end
            end
        end
    end
    return nil
end

local function doorLeg(inside)
    if not running then return end
    local door = inside and findDoor(tes3.getActiveCells(), true) or findDoor({ tes3.player.cell }, false)
    if not door then
        finish(string.format("!! no door %s", inside and ("to '" .. cfg.interior .. "'") or "back outside"))
        return
    end
    doorLegInside, doorLegT = inside, os.clock()
    tes3.player:activate(door)
end

onDoorArrived = function(e)
    local c = e.cell
    if not c or (c.isInterior ~= doorLegInside) then return end   -- not this leg's arrival
    if not doorLegInside then hopsDone = hopsDone + 1 end
    log("door %d %s: %.0f ms activate -> cellChanged (%s)", hopsDone + (doorLegInside and 1 or 0),
        doorLegInside and "IN " or "OUT", (os.clock() - doorLegT) * 1000, c.editorName or "?")
    if not doorLegInside and hopsDone >= cfg.hops then
        finish("doors done")
        return
    end
    local nextInside = not doorLegInside
    timer.start({ type = timer.real, duration = math.max(cfg.dwell, 0.001), iterations = 1,
        callback = function() doorLeg(nextInside) end })
end

local function beginDoor()
    -- A save made INSIDE the interior starts with the way out (the user's own loop began there).
    local startInside = tes3.player.cell.isInterior
    if startInside and tes3.player.cell.id ~= cfg.interior then
        log("!! player is in '%s', not '%s'", tes3.player.cell.id, cfg.interior)
        return
    end
    tes3.runLegacyScript({ command = "tgm" })
    hopsDone = 0
    t0 = os.time()
    running = true
    event.register("simulate", onSimulate)   -- finish() unregisters it; nothing else runs in it here
    log("DOOR START: %d round trips through the door to '%s', dwell %.1fs, starting %s", cfg.hops,
        cfg.interior, cfg.dwell, startInside and "inside" or "outside")
    doorLeg(not startInside)
end

local function onLoaded()
    if not cfg.enabled then
        log("disabled (enabled=false) - the player will not be moved")
        return
    end
    if cfg.mode == "door" then
        log("armed (door): delay=%.1fs hops=%d dwell=%.1fs interior='%s'", cfg.delay, cfg.hops, cfg.dwell, cfg.interior)
        timer.start({ type = timer.real, duration = math.max(cfg.delay, 0.001), iterations = 1, callback = beginDoor })
        return
    end
    if cfg.mode == "hop" then
        log("armed (hop): delay=%.1fs hops=%d dwell=%.1fs interior='%s'", cfg.delay, cfg.hops, cfg.dwell, cfg.interior)
        timer.start({ type = timer.real, duration = math.max(cfg.delay, 0.001), iterations = 1, callback = beginHop })
        return
    end
    log("armed: delay=%.1fs target=%d seed=%d", cfg.delay, cfg.target, cfg.seed)
    timer.start({ type = timer.real, duration = math.max(cfg.delay, 0.001), iterations = 1, callback = begin })
end

event.register("loaded", onLoaded)
event.register("cellChanged", onCellChanged)
