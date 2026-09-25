-- MGE XE - BACK-STEP FIXTURE (test fixture, NOT a gameplay mod).
--
-- WHY THIS EXISTS. A near/far handover hole reproduced by the user only AFTER moving: load a save,
-- walk back ~20 m, walk back to the same spot, and castle walls that were drawn on load are gone.
-- A pinned save alone never shows it. This drives exactly that: after kSettle it backs the player
-- off kDist units along the reverse of its facing at walking pace, holds kPause, walks back to the
-- EXACT start position, and parks there with the facing untouched, so a dump taken afterwards views
-- the same scene the load did.
--
-- Motion is per frame at kSpeed (a walk, never a hop: a one-frame move of a cell or more is a
-- TELEPORT to the client and purges the cache, which would fake the history under test). Height
-- follows the terrain under the path at the start's own clearance. Mover = noclip + flying +
-- position write, the one gridwalk found works (a bare position write is undone by the player's
-- own update).
--
-- ⚠⚠ INERT UNLESS THE MARKER FILE EXISTS. Arm before a run, DISARM after, even if the run crashed.
local markerPath = "Data Files/MWSE/mods/mgexe/backstep/ACTIVE"
local marker = io.open(markerPath, "r")
if not marker then return end
marker:close()

mwse.log("[backstep] ARMED - marker present; this session WILL walk the player back and forth")

local kSettle = 12.0     -- seconds after load before moving
local kDist   = 1430.0   -- ~20 m (MW unit ~1.4 cm)
local kSpeed  = 250.0    -- units/s, about a walk
local kPause  = 3.0      -- seconds held at the far point

-- FORWARD mode (a second marker file, FORWARD, beside ACTIVE): instead of out-and-back, walk straight
-- ahead kFwdDist at kFwdSpeed and park. That is the approach the user reports a one-frame blip on —
-- crossing a cell border toward the fort, where MW shifts its grid and loads a new row.
local fwdMarker = io.open("Data Files/MWSE/mods/mgexe/backstep/FORWARD", "r")
local forward = fwdMarker ~= nil
local fwdSpeedText = fwdMarker and fwdMarker:read("*l") or nil   -- FORWARD's first line = speed
if fwdMarker then fwdMarker:close() end
local kFwdDist  = 12000.0
local kFwdSettle = 10.0  -- seconds at the teleport target before walking
local kFwdOver  = 4000.0 -- ...and how far PAST the save spot it keeps walking
local kFwdSpeed = tonumber(fwdSpeedText or "") or 400.0  -- about a run unless FORWARD says otherwise

local phase = "idle"     -- idle -> out -> pause -> back -> parked
local t, phaseT = 0.0, 0.0
local sx, sy, sz, clear, facing
local bx, by             -- unit vector backwards

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

local function place(mp, x, y)
    local g = landHeight(x, y)
    local z = g and (g + clear) or sz
    mp.movementCollision = false
    mp.isFlying = true
    tes3.player.position = tes3vector3.new(x, y, z)
    tes3.player.facing = facing
end

-- Scene-state dump of every reference whose id matches kProbe in the active cells: is it attached,
-- app-culled, and where is its world bound — per reference root and per geometry leaf. Taken on
-- arrival and again once parked, so a hole can be read as a difference between the two.
local kProbe = "ex_imp_wall"
local function dumpNode(tag, node, depth)
    if not node or depth > 6 then return end
    local o = node.worldBoundOrigin
    local chain, p = "", node.parent
    for _ = 1, 3 do
        if not p then break end
        chain = chain .. " <- " .. tostring(p.name)
        p = p.parent
    end
    mwse.log("[backstep][%s] %s%s culled=%s bound=(%.0f,%.0f,%.0f) r=%.0f%s", tag,
             string.rep("  ", depth), tostring(node.name), tostring(node.appCulled),
             o and o.x or 0, o and o.y or 0, o and o.z or 0, node.worldBoundRadius or -1,
             depth == 0 and chain or "")
    if node.children then
        for _, c in ipairs(node.children) do
            if c then dumpNode(tag, c, depth + 1) end
        end
    end
end

local function dumpProbe(tag)
    for _, cell in ipairs(tes3.getActiveCells()) do
        for ref in cell:iterateReferences(tes3.objectType.static) do
            if ref.object.id:lower():find(kProbe, 1, true) then
                local q = ref.position
                mwse.log("[backstep][%s] REF %s cell=(%d,%d) pos=(%.0f,%.0f,%.0f) disabled=%s deleted=%s node=%s",
                         tag, ref.object.id, cell.gridX, cell.gridY, q.x, q.y, q.z,
                         tostring(ref.disabled), tostring(ref.deleted), tostring(ref.sceneNode ~= nil))
                if ref.sceneNode then dumpNode(tag, ref.sceneNode, 0) end
            end
        end
    end
end

local function setPhase(p)
    phase, phaseT = p, 0.0
    local q = tes3.player.position
    mwse.log("[backstep] t=%.1f phase=%s pos=(%.1f,%.1f,%.1f)", t, p, q.x, q.y, q.z)
end

local function onSimulate(e)
    if phase == "idle" or phase == "parked" then return end
    if tes3.menuMode() then return end
    local mp = tes3.mobilePlayer
    if not mp then return end
    t = t + e.delta
    phaseT = phaseT + e.delta

    if phase == "fwd" then
        local d = math.min(kFwdDist + kFwdOver, kFwdSpeed * phaseT)
        place(mp, sx - bx * d, sy - by * d)
        if d >= kFwdDist + kFwdOver then
            setPhase("parked")
            timer.start({ duration = 2.0, type = timer.real, iterations = 1,
                          callback = function() dumpProbe("after") end })
        end
    elseif phase == "out" then
        local d = math.min(kDist, kSpeed * phaseT)
        place(mp, sx + bx * d, sy + by * d)
        if d >= kDist then setPhase("pause") end
    elseif phase == "pause" then
        place(mp, sx + bx * kDist, sy + by * kDist)
        if phaseT >= kPause then setPhase("back") end
    elseif phase == "back" then
        local d = math.max(0.0, kDist - kSpeed * phaseT)
        place(mp, sx + bx * d, sy + by * d)
        if d <= 0.0 then
            tes3.player.position = tes3vector3.new(sx, sy, sz)
            setPhase("parked")
            mwse.log("[backstep] PARKED at start - the dump from here on is the repro frame")
            timer.start({ duration = 2.0, type = timer.real, iterations = 1,
                          callback = function() dumpProbe("after") end })
        end
    end
end

local function onLoaded()
    phase, t = "idle", 0.0
    timer.start({
        duration = kSettle, type = timer.real, iterations = 1,
        callback = function()
            local p = tes3.player.position
            sx, sy, sz = p.x, p.y, p.z
            facing = tes3.player.facing
            local g = landHeight(sx, sy)
            clear = g and (sz - g) or 0.0
            bx, by = -math.sin(facing), -math.cos(facing)
            mwse.log("[backstep] start=(%.1f,%.1f,%.1f) facing=%.3f clear=%.1f back=(%.3f,%.3f)",
                     sx, sy, sz, facing, clear, bx, by)
            dumpProbe("before")
            if not forward then
                setPhase("out")
                return
            end
            -- FORWARD: teleport kFwdDist BEHIND the save spot (a cell change: the client purges and
            -- re-captures there), settle, then walk forward back to the save spot, so the fort ENTERS
            -- MW's range mid-walk — the case the user sees, which a load already inside range hides.
            local fx, fy = sx + bx * kFwdDist, sy + by * kFwdDist
            local g2 = landHeight(fx, fy)
            local fz = g2 and (g2 + clear) or sz
            tes3.positionCell({ reference = tes3.player, position = tes3vector3.new(fx, fy, fz),
                                orientation = tes3vector3.new(0, 0, facing), suppressFader = true })
            mwse.log("[backstep] FORWARD: teleported to (%.1f,%.1f,%.1f), settling %.0f s", fx, fy, fz, kFwdSettle)
            timer.start({ duration = kFwdSettle, type = timer.real, iterations = 1, callback = function()
                sx, sy = fx, fy
                tes3.player.facing = facing
                setPhase("fwd")
            end })
        end,
    })
end

event.register("simulate", onSimulate)
event.register("loaded", onLoaded)
