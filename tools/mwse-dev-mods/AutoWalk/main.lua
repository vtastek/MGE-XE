-- Auto Walk (dev aid: cell-border crossings for the texture prefetch / eviction work,
-- tasks/forge-pipeline-depth.md "Fast-turn fps drop")
--
-- WHY: a grid move is the one event the load-and-turn harness never produces. Crossing a border
-- slides MW's 3x3 active grid, loads a row of cells ahead and drops one behind, and every texture
-- system keyed on the grid reacts at once: the prefetch rescans and re-pins, the eviction releases
-- what only the dropped row used, the geometry cache evicts by cell. Nobody walks a minimized game,
-- so this does.
--
-- WHAT: on load, waits `delay` real seconds, then glides the player `distance` units along
-- `heading` at `speed` units/s, then holds still for `settle` seconds. The glide writes the
-- position every simulate frame and follows the landscape with a downward ray (never below sea
-- level), so it is not stopped by collision and does not fall. The camera faces the way it walks.
--
-- Ground truth, as in AutoTurn360: every cell MW actually entered is logged from the cellChanged
-- event, and a glide whose position MW did not take is reported (`writes-ignored`), so a run that
-- never crossed a border says so.
--
-- CONFIG (Data Files/MWSE/config/AutoWalk.json) — written by mgeHost64/forge-walk-run.sh, removed on
-- exit:
--   enabled   false = never registers a per-frame handler. DEFAULT OFF: it takes the controls away.
--   delay     real seconds after `loaded` before the walk starts (let the post-load work finish).
--   speed     units per simulated second. A cell is 8192 units; MW running is ~300-500.
--   distance  total units to glide. 24576 = three cells, i.e. at least two border crossings.
--   heading   degrees (0 = north, 90 = east, MW's facing convention); negative = the current facing.
--   settle    real seconds to hold still afterwards. Longer than 600 frames (the stale-texture
--             eviction age) so a release of anything still in the grid would show in the log.
--   fly       true = keep the save's camera (pitch included) and glide along its 3D view direction,
--             instead of following the ground and facing the walk; `heading` is ignored. Stops 1000
--             units above the ground. For a save that looks down on a town from above: the town
--             crosses MW's near reach the way it does when the player flies down to it.
--   z         fly only: start altitude instead of the save's (absent = the save's).
--   pitch     fly only: degrees below the horizon to look and glide along, with `heading`, instead
--             of the save's camera (absent = the save's camera).

local defaults = {
    enabled = false,
    delay = 12.0,
    speed = 600.0,
    distance = 24576.0,
    heading = -1.0,
    settle = 20.0,
    fly = false,
}

local cfg = mwse.loadConfig("AutoWalk", defaults)
local kCell = 8192.0

local function log(fmt, ...)
    mwse.log("[autowalk] " .. fmt, ...)
end

local walking = false
local walked = 0.0        -- units covered
local simTime = 0.0
local startPos = nil
local dirX, dirY, dirZ, heading = 0.0, 0.0, 0.0, 0.0
local lastZ = 0.0
local nextMark = 0.0
local wroteBad = 0
local crossings = 0

local function gridOf(x, y)
    return math.floor(x / kCell), math.floor(y / kCell)
end

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
    return lastZ   -- no land under the ray (cell not loaded yet): hold the last height
end

local onSimulate

local function finish()
    if not walking then return end
    walking = false
    event.unregister("simulate", onSimulate)
    local p = tes3.player.position
    local gx, gy = gridOf(p.x, p.y)
    log("WALK DONE: %.0f units in %.2fs sim | now grid (%d,%d) | cell changes=%d | writes-ignored=%d",
        walked, simTime, gx, gy, crossings, wroteBad)
    if crossings == 0 then
        log("!! MW reported no cell change - this run did not cross a border")
    end
    if wroteBad > 0 then
        log("!! the engine did NOT take %d position writes", wroteBad)
    end
    timer.start({
        type = timer.real, duration = cfg.settle, iterations = 1,
        callback = function()
            log("SETTLED: %.1fs after the walk", cfg.settle)
        end,
    })
end

onSimulate = function(e)
    if not walking then return end
    simTime = simTime + e.delta
    walked = math.min(walked + cfg.speed * e.delta, cfg.distance)
    local x = startPos.x + dirX * walked
    local y = startPos.y + dirY * walked
    local z = groundZ(x, y)
    if cfg.fly then
        local fz = startPos.z + dirZ * walked
        if fz < z + 1000.0 then
            log("fly: 1000 units above the ground at %.0f units - stopping", walked)
            cfg.distance = walked
        end
        z = math.max(fz, z + 1000.0)
        if cfg.pitch then
            tes3.player.orientation = tes3vector3.new(math.rad(cfg.pitch), 0, heading)
        end
    end
    tes3.player.position = tes3vector3.new(x, y, z)
    if not cfg.fly then
        tes3.player.orientation = tes3vector3.new(0, 0, heading)
    end

    local got = tes3.player.position
    if math.abs(got.x - x) > 1.0 or math.abs(got.y - y) > 1.0 then
        wroteBad = wroteBad + 1
    end
    if walked >= nextMark then
        local gx, gy = gridOf(x, y)
        log("at %6.0f units (sim %.1fs): pos (%.0f, %.0f, %.0f) grid (%d,%d)", walked, simTime, x, y, z, gx, gy)
        nextMark = nextMark + kCell / 4
    end
    if walked >= cfg.distance then finish() end
end

local function onCellChanged(e)
    if not walking then return end
    crossings = crossings + 1
    local c = e.cell
    log("CELL CHANGED #%d at %.0f units (sim %.1fs): -> %s (%s,%s)", crossings, walked, simTime,
        c and c.editorName or "?", c and tostring(c.gridX) or "?", c and tostring(c.gridY) or "?")
end

local function beginWalk()
    if tes3.player.cell.isInterior then
        log("!! player is in an interior - nothing to walk across")
        return
    end
    startPos = tes3.player.position:copy()
    heading = cfg.heading >= 0 and math.rad(cfg.heading) or tes3.mobilePlayer.facing
    -- MW facing: 0 = +Y (north), increasing clockwise, so east (+X) is pi/2.
    dirX, dirY, dirZ = math.sin(heading), math.cos(heading), 0.0
    if cfg.fly and cfg.pitch then
        local p = math.rad(cfg.pitch)
        dirX, dirY, dirZ = math.sin(heading) * math.cos(p), math.cos(heading) * math.cos(p), -math.sin(p)
        if cfg.z then startPos.z = cfg.z end
        log("fly: heading %.1f pitch %.1f from z %.0f", math.deg(heading) % 360, cfg.pitch, startPos.z)
    elseif cfg.fly then
        local v = tes3.getCameraVector()
        dirX, dirY, dirZ = v.x, v.y, v.z
        heading = math.atan2(v.x, v.y)
        if cfg.z then startPos.z = cfg.z end
        log("fly: camera vector (%.3f, %.3f, %.3f)", v.x, v.y, v.z)
    end
    lastZ = startPos.z
    if cfg.fly then
        -- A bare position write does not hold against the player's own update (gridwalk, 2026-09-19).
        tes3.mobilePlayer.movementCollision = false
        tes3.mobilePlayer.isFlying = true
    end
    walked, simTime, nextMark, wroteBad, crossings = 0.0, 0.0, 0.0, 0, 0
    local gx, gy = gridOf(startPos.x, startPos.y)
    log("WALK START: %.0f units at %.0f u/s, heading %.1f deg, from (%.0f, %.0f, %.0f) grid (%d,%d)%s",
        cfg.distance, cfg.speed, math.deg(heading) % 360, startPos.x, startPos.y, startPos.z, gx, gy,
        cfg.fly and " FLYING" or "")
    walking = true
    event.register("simulate", onSimulate)
end

local function onLoaded()
    if not cfg.enabled then
        log("disabled (enabled=false) - the player will not be moved")
        return
    end
    log("armed: delay=%.1fs speed=%.0f distance=%.0f heading=%.1f settle=%.1fs",
        cfg.delay, cfg.speed, cfg.distance, cfg.heading, cfg.settle)
    timer.start({ type = timer.real, duration = math.max(cfg.delay, 0.001), iterations = 1, callback = beginWalk })
end

event.register("loaded", onLoaded)
event.register("cellChanged", onCellChanged)
