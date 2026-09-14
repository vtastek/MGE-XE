-- MGE XE - LONG-SESSION SOAK FIXTURE (test fixture, NOT a gameplay mod).
--
-- WHY THIS EXISTS, AND WHY IT IS NOT velchurn. velchurn drives continuously (rotate/walk/jump every
-- 0.4 s, a load door every ~10 s) because it hunts an EVENT — slot recycling. This one hunts the
-- opposite: a regime change that appears over TIME and then STICKS. The client's present-seam flush
-- was observed jumping 6.5 -> 38 ms mid-session, flipping both ways, with every GPU timestamp flat
-- across the flip; the two mechanisms with that signature (VRAM over-commit, EcoQoS throttling) are
-- both cumulative, not event-driven.
--
-- ⚠ SO THE FIXTURE MUST GO QUIET. A continuous driver would keep loading cells, keep growing the
-- texture residency set, and keep feeding the very quantity under test — and then "residency grew
-- and the flush got worse" would be a statement about the fixture, not the renderer. This does ONE
-- cell transition (so the session is measuring a real, populated scene rather than a load screen)
-- and then does NOTHING for the rest of the run. The idle is the experiment.
--
-- ⚠⚠ INERT UNLESS THE MARKER FILE EXISTS. This teleports the player through a load door; running it
-- during real play would be indistinguishable from possession. No marker, no event registration, no
-- timer. The harness creates `ACTIVE` before the run and deletes it after, even if the run failed.
local markerPath = "Data Files/MWSE/mods/mgexe/soakcell/ACTIVE"
local marker = io.open(markerPath, "r")
if not marker then return end
marker:close()

mwse.log("[soakcell] ARMED - marker present; this session WILL cross one load door")

-- Seconds to wait after the save loads before crossing. Long enough that first-sight texture and
-- geometry upload for the STARTING cell has settled, so the transition is a clean step rather than
-- landing on top of the load burst it would otherwise be confused with.
local kCrossAfter = 30
local done = false

-- The door's OWN destination, never a hand-written coordinate: the marker is where MW itself puts
-- the player, so it cannot wedge them inside geometry. (velchurn's rule, and it earned it.)
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

local function crossOnce()
    if done then return end
    -- Never act through a menu or a loading screen: a teleport issued mid-load is how a fixture
    -- turns into a corrupt save. Retry on the next tick instead of forcing it.
    if tes3.menuMode() then return end
    if not tes3.player or not tes3.mobilePlayer then return end

    local from = tes3.getPlayerCell()
    local door = findLoadDoor()
    if not door then
        mwse.log("[soakcell] no load door in %s - staying put, soak continues",
                 from and from.editorName or "?")
        done = true
        return
    end

    local d = door.destination
    local ok = pcall(tes3.positionCell, {
        reference   = tes3.player,
        cell        = d.cell,
        position    = d.marker.position,
        orientation = d.marker.orientation,
    })
    mwse.log("[soakcell] crossed %s -> %s (%s); going idle for the rest of the run",
             from and from.editorName or "?", d.cell.editorName, ok and "ok" or "FAILED")
    done = true
end

local function onLoaded()
    mwse.log("[soakcell] loaded - will cross one load door in %d s, then idle", kCrossAfter)
    -- iterations = -1 with an internal `done` latch rather than a one-shot timer: the cross can be
    -- refused (menu up, mid-load), and a one-shot would silently never happen. The latch makes the
    -- retry free and the "it happened" record single.
    timer.start({ duration = kCrossAfter, callback = crossOnce, type = timer.real, iterations = -1 })
end

event.register("loaded", onLoaded)
