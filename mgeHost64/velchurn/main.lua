-- MGE XE — MB-1b CHURN DRIVER (test fixture, NOT a gameplay mod).
--
-- WHY THIS EXISTS. MB-1b pairs a skinned part's bone palette with the SAME slot's palette from the
-- previous frame, and the pairing key exists to survive SLOT RECYCLING. Recycling is a rare EVENT —
-- a cell transition, a part leaving and re-entering the drawn set — so on a pinned static save the
-- key's rejection counters read zero whether it works or is dead code. The harness runs minimized
-- with a fixed save and cannot walk, so the one thing that would exercise the key was the one thing
-- that could not be scripted. This drives it: rotate (parts leave/re-enter the frustum -> `stale`),
-- cross load doors (cell epoch wipes every record -> `first`, and re-assigns slots -> `count`).
--
-- ⚠⚠ INERT UNLESS THE MARKER FILE EXISTS. This mod teleports the player through load doors and
-- spins the camera; running during real play would be indistinguishable from possession. It does
-- NOTHING at all unless `ACTIVE` sits beside this file, which only the churn harness creates and
-- which it deletes again on exit. No marker, no event registration, no timer.
local markerPath = "Data Files/MWSE/mods/mgexe/velchurn/ACTIVE"
local marker = io.open(markerPath, "r")
if not marker then return end
marker:close()

mwse.log("[velchurn] ARMED — marker present; this session WILL be driven")

local step = 0

-- Every load door in the current cell, with the destination the engine itself would use. Taking the
-- door's own destination rather than a hand-written coordinate is what keeps this from wedging the
-- player inside geometry: the marker is where MW puts you when you walk through.
local function findLoadDoor()
    local cell = tes3.getPlayerCell()
    if not cell then return nil end
    for ref in cell:iterateReferences(tes3.objectType.door) do
        if ref.destination and ref.destination.cell then
            return ref
        end
    end
    return nil
end

local function drive()
    -- Never act through a menu or a loading screen: a teleport issued mid-load is how a fixture
    -- turns into a corrupt save, and the point of this is to stress the RENDERER, not the engine.
    if tes3.menuMode() then return end
    local player = tes3.player
    if not player or not tes3.mobilePlayer then return end

    step = step + 1
    local phase = step % 24

    if phase < 12 then
        -- ROTATION — the strongest churn for the DRAWN set, and the cheapest. Sweeping the cell
        -- through the frustum makes skinned parts leave the drawn set and come back, and a part
        -- gone for one frame has a palette from further back: that is exactly the `stale` clause,
        -- the one that tears a limb if unguarded.
        --
        -- ⚠ 10 deg, NOT 30. This fixture feeds an instrument (`maxBoneDelta`) whose whole job is to
        -- separate animation-scale motion from cell-scale motion, so the fixture must not itself
        -- produce cell-scale motion. A violent yaw also swamps `mv final: max` with a perfectly
        -- correct ~1134 px camera vector, under which a real tear would hide.
        local o = player.orientation
        player.orientation = tes3vector3.new(o.x, o.y, o.z + math.rad(10))
    elseif phase < 16 then
        -- ⚠⚠ 16 UNITS, NOT 128, AND THE FIRST VERSION'S 128 IS WHY THIS COMMENT EXISTS. A hard
        -- position write is a TELEPORT, not a walk: 128 units in one 90 Hz frame is ~160 m/s, and
        -- the player's own skinned body legitimately moves that far, so `maxBoneDelta` peaked at
        -- exactly 128.09 u with the pairing key INTACT. The fixture had manufactured the very
        -- signal the instrument was built to detect, and the reading was correct — it was the
        -- motion that was fake. 16 u/tick is ~20 m/s, a brisk run, and leaves the animation-scale
        -- band intact.
        local f = tes3.mobilePlayer.facing
        player.position = player.position + tes3vector3.new(math.sin(f) * 16, math.cos(f) * 16, 0)
    elseif phase < 20 then
        local f = tes3.mobilePlayer.facing
        player.position = player.position - tes3vector3.new(math.sin(f) * 16, math.cos(f) * 16, 0)
    elseif phase == 20 then
        -- JUMP — vertical motion the reprojection handles but that moves every bone at once.
        tes3.mobilePlayer.velocity = tes3vector3.new(0, 0, 400)   -- physics-driven, so its speed is real
    elseif phase == 22 then
        -- ⚠ THE CELL TRANSITION, the event this fixture is really for. It bumps the client's cell
        -- epoch, which wipes every caster record AND (MB-1b) every bone-palette record, then
        -- re-populates slots from scratch in the new cell — i.e. it recycles slots wholesale, which
        -- is the one condition under which pairing a palette with a stranger's is possible at all.
        local door = findLoadDoor()
        if door then
            local d = door.destination
            local ok = pcall(tes3.positionCell, {
                reference   = player,
                cell        = d.cell,
                position    = d.marker.position,
                orientation = d.marker.orientation,
            })
            mwse.log("[velchurn] step %d: load door -> %s (%s)", step, d.cell.editorName,
                     ok and "ok" or "FAILED")
        else
            mwse.log("[velchurn] step %d: no load door in %s", step,
                     tes3.getPlayerCell() and tes3.getPlayerCell().editorName or "?")
        end
    end
end

local function onLoaded()
    mwse.log("[velchurn] loaded — driving every 0.4 s (rotate x12, walk, jump, load door)")
    timer.start({ duration = 0.4, callback = drive, type = timer.real, iterations = -1 })
end

event.register("loaded", onLoaded)
