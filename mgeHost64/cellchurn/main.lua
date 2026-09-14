-- MGE XE - RAPID CELL-CHURN FIXTURE (test fixture, NOT a gameplay mod).
--
-- WHY THIS EXISTS. The 22-minute soak (soakcell) proved the thing it was built to test does NOT
-- happen: with the player parked, texture residency froze at 261/872 for the whole run, VRAM took
-- exactly two distinct values, and the seam never changed regime. Residency only moves when the
-- player ENTERS A CELL IT HAS NOT SEEN. So the variable to drive is cells-visited, not seconds.
--
-- ⚠ WHY NOT velchurn / soakcell. Both take `the first load door in the cell`, which from an
-- interior is the door back out - so they PING-PONG between two cells and residency plateaus after
-- one crossing. That is fine for their job (velchurn wants slot recycling, which two cells will
-- produce) and useless for this one. This fixture keeps a visited set and actively prefers a door
-- whose destination it has never seen, so the walk keeps finding NEW textures.
--
-- ⚠⚠ THE FIXTURE MUST NOT MANUFACTURE THE SIGNAL. Teleporting through doors IS the quantity under
-- test here, so driving it is legitimate - but the rate has to stay in the band a player could
-- actually produce, or "residency climbed" becomes a statement about the timer. kPeriod is 3 s,
-- roughly a brisk player clearing a street of shops; it is NOT one crossing per frame. And every
-- crossing uses the door's OWN destination marker, never a hand-written coordinate, so the player
-- lands where MW itself would put them and cannot be wedged into geometry.
--
-- ⚠⚠ INERT UNLESS THE MARKER FILE EXISTS. This teleports the player continuously; during real play
-- it would be indistinguishable from possession. No marker, no event registration, no timer. Arm
-- before a run, DISARM after - even if the run crashed.
local markerPath = "Data Files/MWSE/mods/mgexe/cellchurn/ACTIVE"
local marker = io.open(markerPath, "r")
if not marker then return end
marker:close()

mwse.log("[cellchurn] ARMED - marker present; this session WILL be driven through cells")

local kPeriod   = 3.0    -- seconds between crossings (see the rate warning above)
local kSettle   = 20     -- seconds after load before the first crossing, so first-sight upload
                         -- for the STARTING cell is not folded into the first measured step

local visited = {}       -- destination cell name -> true
local distinct = 0
local steps = 0

local function cellName(cell)
    if not cell then return "?" end
    return cell.editorName or cell.id or "?"
end

-- Every load door in the current cell, split into "leads somewhere new" and "leads somewhere seen".
-- Preferring the novel one is the whole point; falling back to a seen one keeps the walk alive once
-- a wing of the city is exhausted, rather than parking the fixture silently.
local function pickDoor()
    local cell = tes3.getPlayerCell()
    if not cell then return nil end
    local fresh, seen = {}, {}
    for ref in cell:iterateReferences(tes3.objectType.door) do
        local d = ref.destination
        if d and d.cell and d.marker then
            if visited[cellName(d.cell)] then
                seen[#seen + 1] = ref
            else
                fresh[#fresh + 1] = ref
            end
        end
    end
    if #fresh > 0 then return fresh[math.random(#fresh)], true end
    if #seen  > 0 then return seen[math.random(#seen)],  false end
    return nil, false
end

local function drive()
    -- Never act through a menu or a loading screen: a teleport issued mid-load is how a fixture
    -- turns into a corrupt save. Skip this tick and try the next one.
    if tes3.menuMode() then return end
    if not tes3.player or not tes3.mobilePlayer then return end

    local door, isNew = pickDoor()
    if not door then
        mwse.log("[cellchurn] step %d: no load door in %s - stuck, will retry",
                 steps, cellName(tes3.getPlayerCell()))
        return
    end

    local d = door.destination
    local name = cellName(d.cell)
    local ok = pcall(tes3.positionCell, {
        reference   = tes3.player,
        cell        = d.cell,
        position    = d.marker.position,
        orientation = d.marker.orientation,
    })
    if not ok then
        mwse.log("[cellchurn] step %d: positionCell FAILED -> %s", steps, name)
        return
    end

    steps = steps + 1
    if not visited[name] then
        visited[name] = true
        distinct = distinct + 1
    end
    -- Logged EVERY crossing, with the distinct count, because the residency trend is only readable
    -- against how many NEW cells produced it - "slots climbed" against an unknown number of cells
    -- is the same unfalsifiable claim as a perf number without a named scene.
    mwse.log("[cellchurn] step %d: -> %s (%s) | distinct=%d",
             steps, name, isNew and "NEW" or "revisit", distinct)
end

local function onLoaded()
    math.randomseed(os.time())
    mwse.log("[cellchurn] loaded - settling %d s, then a crossing every %.1f s", kSettle, kPeriod)
    timer.start({
        duration = kSettle, type = timer.real, iterations = 1,
        callback = function()
            timer.start({ duration = kPeriod, type = timer.real, iterations = -1, callback = drive })
        end,
    })
end

event.register("loaded", onLoaded)
