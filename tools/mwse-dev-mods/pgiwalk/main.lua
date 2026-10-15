-- MGE XE - PROBE-GI WALK-IN FIXTURE (test fixture, NOT a gameplay mod).
--
-- WHY THIS EXISTS. The probe field (tasks/forge-gi.md "Runtime probe GI v1") is built around the
-- camera, nearest first. A save that loads AT a spot builds that spot's probes before anything is
-- looked at; the user's reports were about WALKING IN: "tunnel looks dark from outside ... walk
-- inside and it lits up", "I walk to another mushroom at walk speed, and it is slow". So this takes
-- the player AWAY from the save's spot — far enough that its probe columns leave the 24-column window
-- and are thrown away — and walks back in at walking pace, then holds still with the save's own view.
-- Dumps come from the host knobs (dumpAtFrame / dumpEvery / dumpCount), not from here.
--
-- MOTION: noclip + flying + a position write per frame (a bare position write does not move the
-- player, see mgeHost64/gridwalk). The camera's orientation is never touched, so the hold frames see
-- exactly what the save looked at. The path is a straight line at the save's height: it may pass
-- through rock or terrain, which the probe scheduler does not care about (it follows the eye's XY).
--
-- ⚠⚠ INERT UNLESS THE MARKER FILE EXISTS. This moves the player; during real play it is
-- indistinguishable from possession. Arm before a run, DISARM after - even if the run crashed.
local markerPath = "Data Files/MWSE/mods/mgexe/pgiwalk/ACTIVE"
local marker = io.open(markerPath, "r")
if not marker then return end
marker:close()

mwse.log("[pgiwalk] ARMED - marker present; this session WILL be moved away and walked back")

local kSettle = 6.0      -- seconds after load before leaving
local kAway   = 4000.0   -- units behind the save's view: > 12 columns x 256, so its columns are dropped
local kFast   = 2000.0   -- units/s going out (below the client's teleport threshold per frame)
local kHoldOut = 4.0     -- seconds held away (the window recentres, the old columns reset)
-- units/s coming back: a walk/jog by default; PGIWALK_SPEED overrides (the user's run was ~580)
local kWalk   = tonumber(os.getenv("PGIWALK_SPEED") or "") or 200.0

local phase = "settle"
local t, phaseT = 0.0, 0.0
local sx, sy, sz, ax, ay = 0.0, 0.0, 0.0, 0.0, 0.0

local followGround = false   -- PGIWALK_TO walks descend: the save's height -> the land at the target
local zFrom, zTo = 0.0, 0.0

local function landHeight(x, y)
    local root = tes3.game and tes3.game.worldLandscapeRoot
    if not root then return nil end
    local hit = tes3.rayTest({ position = tes3vector3.new(x, y, 30000.0), direction = tes3vector3.new(0, 0, -1),
                               root = root })
    return hit and hit.intersection and hit.intersection.z or nil
end

local function place(x, y)
    local pl, mp = tes3.player, tes3.mobilePlayer
    mp.movementCollision = false
    mp.isFlying = true
    local z = sz
    if followGround then
        -- Height along the walk by distance covered, so a descent from a raised save spot (a wall
        -- walk, a ramp) is replayed as the user's eye track did it, not snapped to the land below.
        local L = math.sqrt((sx - ax) ^ 2 + (sy - ay) ^ 2)
        local d = math.sqrt((x - ax) ^ 2 + (y - ay) ^ 2)
        local f = (L > 1.0) and math.min(1.0, d / L) or 1.0
        z = zFrom + (zTo - zFrom) * f
    end
    pl.position = tes3vector3.new(x, y, z)
    if followGround then   -- look where we walk, level: the user's approach, not the save's own view
        pl.orientation = tes3vector3.new(0.0, 0.0, math.atan2(sx - ax, sy - ay))
    end
end

local function toward(tx, ty, speed, dt)
    local p = tes3.player.position
    local dx, dy = tx - p.x, ty - p.y
    local d = math.sqrt(dx * dx + dy * dy)
    local step = speed * dt
    if d <= step then place(tx, ty); return true end
    place(p.x + dx / d * step, p.y + dy / d * step)
    return false
end

local function enter(name)
    phase = name
    phaseT = 0.0
    local p = tes3.player.position
    mwse.log("[pgiwalk] t=%.1f phase %s at (%.0f, %.0f, %.0f)", t, name, p.x, p.y, p.z)
end

local function onSimulate(e)
    if tes3.menuMode() then return end
    local mp = tes3.mobilePlayer
    if not tes3.player or not mp then return end
    t = t + e.delta
    phaseT = phaseT + e.delta
    local hp = mp.health
    if hp and hp.current < hp.base then
        tes3.setStatistic({ reference = tes3.player, name = "health", current = hp.base })
    end
    if phase == "settle" then
        if phaseT >= kSettle then enter("out") end
    elseif phase == "out" then
        if toward(ax, ay, kFast, e.delta) then enter("holdout") end
    elseif phase == "holdout" then
        place(ax, ay)
        if phaseT >= kHoldOut then enter("walk") end
    elseif phase == "walk" then
        if toward(sx, sy, kWalk, e.delta) then enter("hold") end
    elseif phase == "hold" then
        place(sx, sy)
    end
end

-- PGIWALK_TO="x,y": instead of away-and-back, walk FROM the save's spot TO that point (at the save's
-- height, kWalk u/s) and hold there — a user's own approach replayed from their log's eye track.
local walkTo = os.getenv("PGIWALK_TO")

local function onLoaded()
    local p = tes3.player.position
    local f = tes3.player.orientation.z
    sx, sy, sz = p.x, p.y, p.z
    ax, ay = sx - math.sin(f) * kAway, sy - math.cos(f) * kAway   -- straight back from the view
    if walkTo then
        -- "x_y" (the perf harness splits its client env on commas), "x,y" also accepted
        local tx, ty = walkTo:match("^%s*(-?[%d%.]+)%s*[,_]%s*(-?[%d%.]+)%s*$")
        if tx then
            -- "away" is the save's spot (where we already are) and "home" the target: the out leg is
            -- empty, the walk leg goes save -> target.
            ax, ay = sx, sy
            sx, sy = tonumber(tx), tonumber(ty)
            followGround = true
            zFrom = sz
            zTo = landHeight(sx, sy) or sz
        end
    end
    t = 0.0
    phase = "settle"; phaseT = 0.0
    mwse.log("[pgiwalk] loaded at (%.0f, %.0f, %.0f) facing %.2f; away point (%.0f, %.0f)", sx, sy, sz, f, ax, ay)
    event.register("simulate", onSimulate)
end

event.register("loaded", onLoaded)
