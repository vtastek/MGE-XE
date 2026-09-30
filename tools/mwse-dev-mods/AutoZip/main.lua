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

local defaults = {
    enabled = false,
    delay = 12.0,
    target = 500,
    seed = 1,
    speed = 16384.0,
    farChance = 0.4,
    pause = 0.5,
    settle = 20.0,
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

local function onCellChanged(e)
    if not running then return end
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

local function onLoaded()
    if not cfg.enabled then
        log("disabled (enabled=false) - the player will not be moved")
        return
    end
    log("armed: delay=%.1fs target=%d seed=%d", cfg.delay, cfg.target, cfg.seed)
    timer.start({ type = timer.real, duration = math.max(cfg.delay, 0.001), iterations = 1, callback = begin })
end

event.register("loaded", onLoaded)
event.register("cellChanged", onCellChanged)
