-- MGE XE - EXTERIOR GRID-WALK FIXTURE (test fixture, NOT a gameplay mod).
--
-- WHY THIS EXISTS. Texture residency (tasks/forge-memory-shape.md) is driven by Morrowind's active
-- cell grid: a cell entering the grid brings its textures, a cell leaving it should take them away.
-- cellchurn drives DOOR crossings, i.e. whole-grid swaps; this drives the other half, the continuous
-- exterior grid shift a player produces by travelling overland, which only an exterior border
-- crossing can produce. It flies a closed loop, so the SAME cells come back every lap: residency that
-- keeps climbing lap over lap is a leak, residency that swings and returns is eviction working.
--
-- ⚠⚠ THE FIXTURE MUST NOT MANUFACTURE THE SIGNAL. The motion is per frame at kSpeed (a fast
-- levitating player, ~15 m/s), never a hop: the client treats a one-frame move of a cell or more as a
-- TELEPORT and purges the whole cache, which would fake exactly the churn under test. Height follows
-- the TERRAIN (a ray against worldLandscapeRoot only), so the camera sees what a player there would.
--
-- ⚠ HOW TO MOVE THE PLAYER IS NOT OBVIOUS, SO THE FIXTURE FINDS OUT. The first version wrote
-- tes3.player.position every simulate and the player did not move at all (eye drifted 1 unit in
-- five minutes, game unpaused): something in the player's own update puts it back. So the movers
-- are tried in order, each for kProbeSecs, and the first that actually displaces the player is kept
-- and LOGGED — a run whose mover never worked says so instead of reporting a flat residency curve
-- as "no growth". Last resort is a positionCell hop per cell, which IS a teleport to the client
-- (whole-cache purge per hop) and is logged as such, so the run is labelled as the weaker test.
--
-- ⚠⚠ INERT UNLESS THE MARKER FILE EXISTS. This moves the player continuously; during real play it is
-- indistinguishable from possession. No marker, no event registration. Arm before a run, DISARM
-- after - even if the run crashed.
local markerPath = "Data Files/MWSE/mods/mgexe/gridwalk/ACTIVE"
local marker = io.open(markerPath, "r")
if not marker then return end
marker:close()

mwse.log("[gridwalk] ARMED - marker present; this session WILL be flown around the exterior grid")

local kCell      = 8192.0
local kSpeed     = 1024.0   -- units/s: a cell every ~8 s
local kHover     = 96.0     -- above the terrain
local kSettle    = 20.0     -- seconds after load before moving (first-sight upload of the start area)
local kProbeSecs = 3.0      -- per mover, while finding one that works
local kHopSecs   = 8.0      -- last-resort mover: one positionCell hop per cell, at the same pace

-- Cell-grid waypoints (cell centres). Balmora -> east -> north (Red Mountain's western flank) ->
-- west (West Gash) -> south -> back: ~30 cells a lap, all land.
local route = {
    { -3, -2 }, {  4, -2 }, {  4,  5 }, { -4,  5 }, { -4, -2 }, { -3, -2 },
}

local function cellCentre(i)
    local w = route[i]
    return w[1] * kCell + kCell * 0.5, w[2] * kCell + kCell * 0.5
end

-- Movers, tried in order. Each takes (player, mobile, nx, ny, nz, vx, vy, vz).
local movers = {
    { name = "reference position write", fn = function(pl, mp, nx, ny, nz)
        pl.position = tes3vector3.new(nx, ny, nz)
    end },
    { name = "noclip+flying, position write", fn = function(pl, mp, nx, ny, nz)
        mp.movementCollision = false
        mp.isFlying = true
        pl.position = tes3vector3.new(nx, ny, nz)
    end },
    { name = "noclip+flying, velocity", fn = function(pl, mp, nx, ny, nz, vx, vy, vz)
        mp.movementCollision = false
        mp.isFlying = true
        mp.velocity = tes3vector3.new(vx, vy, vz)
    end },
}
local mover = 1
local locked = false
local hopMode = false
local probeT, probeX, probeY = 0.0, nil, nil

local leg = 1          -- heading from route[leg] to route[leg + 1]
local lap = 0
local moving = false
local t = 0.0
local hopT = 0.0
local lastGrid = nil
local crossings = 0
local visited = {}
local distinct = 0

local function landHeight(x, y)
    local root = tes3.game and tes3.game.worldLandscapeRoot
    if not root then return nil end
    local hit = tes3.rayTest({
        position  = tes3vector3.new(x, y, 30000.0),
        direction = tes3vector3.new(0, 0, -1),
        root      = root,
    })
    return hit and hit.intersection and hit.intersection.z or nil
end

local function noteCrossing(x, y)
    local gx, gy = math.floor(x / kCell), math.floor(y / kCell)
    local key = gx .. "," .. gy
    if key == lastGrid then return end
    if lastGrid then crossings = crossings + 1 end
    if not visited[key] then visited[key] = true; distinct = distinct + 1 end
    local cell = tes3.getPlayerCell()
    local cx, cy = cell and cell.gridX or 0, cell and cell.gridY or 0
    -- MW's own idea of the player's cell, beside ours: if they disagree for more than the moment of
    -- the crossing, the player is not really where the fixture thinks and the run is void.
    mwse.log("[gridwalk] t=%.1f cross -> (%d,%d) mw=(%d,%d) lap=%d crossings=%d distinct=%d",
             t, gx, gy, cx, cy, lap, crossings, distinct)
    lastGrid = key
end

-- Advance the route target; returns the (tx, ty) currently headed for.
local function target(px, py, reach)
    local tx, ty = cellCentre(leg + 1)
    local dx, dy = tx - px, ty - py
    if math.sqrt(dx * dx + dy * dy) <= reach then
        leg = leg + 1
        if leg >= #route then
            leg = 1
            lap = lap + 1
            mwse.log("[gridwalk] lap %d done t=%.0fs crossings=%d distinct=%d", lap, t, crossings, distinct)
        end
        tx, ty = cellCentre(leg + 1)
    end
    return tx, ty
end

local function hop(player)
    -- Last resort: one hop per kHopSecs to the next cell along the route (a cell change, so the
    -- client sees a teleport). Still exercises grid load/unload and the loop's revisits.
    local p = player.position
    local tx, ty = target(p.x, p.y, kCell)
    local dx, dy = tx - p.x, ty - p.y
    local dist = math.sqrt(dx * dx + dy * dy)
    local s = math.min(kCell, dist)
    local nx, ny = p.x + dx / dist * s, p.y + dy / dist * s
    tes3.positionCell({ reference = player, position = tes3vector3.new(nx, ny, (landHeight(nx, ny) or 2000.0) + kHover),
                        suppressFader = true })
    noteCrossing(nx, ny)
end

local function onSimulate(e)
    if not moving then return end
    if tes3.menuMode() then return end
    local player, mp = tes3.player, tes3.mobilePlayer
    if not player or not mp then return end
    t = t + e.delta

    -- Cheap insurance against the one thing that can end an unattended run early.
    local hp = mp.health
    if hp and hp.current < hp.base then
        tes3.setStatistic({ reference = player, name = "health", current = hp.base })
    end

    if hopMode then
        hopT = hopT + e.delta
        if hopT >= kHopSecs then hopT = 0.0; hop(player) end
        return
    end

    local p = player.position
    if not locked then
        if probeX == nil then probeT, probeX, probeY = t, p.x, p.y end
        if t - probeT >= kProbeSecs then
            local moved = math.sqrt((p.x - probeX) ^ 2 + (p.y - probeY) ^ 2)
            local want = kSpeed * (t - probeT)
            if moved >= 0.3 * want then
                locked = true
                mwse.log("[gridwalk] mover LOCKED: %s (moved %.0f of %.0f u in %.1f s)",
                         movers[mover].name, moved, want, t - probeT)
            else
                mwse.log("[gridwalk] mover FAILED: %s (moved %.0f of %.0f u in %.1f s)",
                         movers[mover].name, moved, want, t - probeT)
                mover = mover + 1
                probeX = nil
                if mover > #movers then
                    hopMode = true
                    mwse.log("[gridwalk] no continuous mover works - FALLING BACK to one positionCell hop"
                             .. " per %.0f s (the client sees each hop as a TELEPORT)", kHopSecs)
                    return
                end
            end
        end
    end

    local step = kSpeed * e.delta
    local tx, ty = target(p.x, p.y, step)
    local dx, dy = tx - p.x, ty - p.y
    local dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1.0 then return end
    local ux, uy = dx / dist, dy / dist
    local nx, ny = p.x + ux * step, p.y + uy * step
    local z = landHeight(nx, ny)
    local nz = z and (z + kHover) or p.z
    local vz = e.delta > 0 and (nz - p.z) / math.max(e.delta, 0.001) or 0.0
    movers[mover].fn(player, mp, nx, ny, nz, ux * kSpeed, uy * kSpeed, vz)
    player.facing = math.atan2(ux, uy)
    noteCrossing(p.x, p.y)
end

local function onLoaded()
    mwse.log("[gridwalk] loaded - settling %.0f s, then %.0f u/s around %d waypoints",
             kSettle, kSpeed, #route)
    t = 0.0
    event.register("simulate", onSimulate)
    timer.start({
        duration = kSettle, type = timer.real, iterations = 1,
        callback = function()
            local cell = tes3.getPlayerCell()
            if cell and cell.isInterior then
                -- Start outdoors: Balmora's centre, high enough to clear any roof; the per-frame
                -- terrain ray brings it down once the cells have loaded.
                local x, y = cellCentre(1)
                tes3.positionCell({ reference = tes3.player, position = tes3vector3.new(x, y, 4000.0) })
                mwse.log("[gridwalk] started indoors - moved to the exterior at the first waypoint")
            end
            moving = true
            mwse.log("[gridwalk] moving")
        end,
    })
end

event.register("loaded", onLoaded)
