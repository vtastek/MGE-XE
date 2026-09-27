-- MGE XE — DOOR HOP (test fixture, NOT a gameplay mod). Sunny 16 P3's exposure servo is judged at
-- door crossings, which a pinned save cannot produce. From an interior save: wait, go out through the
-- cell's load door, wait, come back in through the exterior door that leads to the start cell, and
-- repeat. The host's [exp-door] trace records every servo step around each switch.
--
-- ⚠⚠ INERT UNLESS `ACTIVE` SITS BESIDE THIS FILE (same contract as velchurn): no marker, no event,
-- no timer. Arm before the run, delete the marker after — even if the run crashed.
local markerPath = "Data Files/MWSE/mods/mgexe/doorhop/ACTIVE"
local marker = io.open(markerPath, "r")
if not marker then return end
marker:close()

mwse.log("[doorhop] ARMED — marker present; this session WILL be driven")

local homeCell = nil
local hops, maxHops = 0, 4        -- out, in, out, in
local wait = 8.0                  -- seconds on each side (the servo settles in ~2)

local function doorTo(filter)
    local cell = tes3.getPlayerCell()
    if not cell then return nil end
    for ref in cell:iterateReferences(tes3.objectType.door) do
        local d = ref.destination
        if d and d.cell and filter(d.cell) then return ref end
    end
    return nil
end

local function hop()
    if tes3.menuMode() then return end
    if hops >= maxHops then return end
    local here = tes3.getPlayerCell()
    local door
    if here == homeCell then
        door = doorTo(function(c) return c ~= homeCell end)
    else
        -- the exterior can hold many doors; take the one back to where we started
        door = doorTo(function(c) return c == homeCell end)
    end
    if not door then
        mwse.log("[doorhop] hop %d: no suitable door in %s", hops, here and here.editorName or "?")
        return
    end
    local d = door.destination
    local ok = pcall(tes3.positionCell, {
        reference = tes3.player, cell = d.cell,
        position = d.marker.position, orientation = d.marker.orientation,
    })
    hops = hops + 1
    mwse.log("[doorhop] hop %d: %s -> %s (%s)", hops, here.editorName, d.cell.editorName,
             ok and "ok" or "FAILED")
    frameLog, lastClock, hopClock = 12, os.clock(), os.clock()
end

-- Real frame times for the frames after each hop: a transition's hitch is what decides whether MW
-- puts up its black fade, and it is a handful of frames no heartbeat resolves.
frameLog, lastClock, hopClock = 0, 0, 0
event.register("enterFrame", function()
    if frameLog <= 0 then return end
    local now = os.clock()
    mwse.log("[doorhop]   frame +%.0f ms (dt %.1f ms)", (now - hopClock) * 1000, (now - lastClock) * 1000)
    lastClock = now
    frameLog = frameLog - 1
end)

local function onLoaded()
    homeCell = tes3.getPlayerCell()
    mwse.log("[doorhop] loaded in %s — hopping every %.0f s, %d hops", homeCell and homeCell.editorName or "?",
             wait, maxHops)
    timer.start({ duration = wait, callback = hop, type = timer.real, iterations = maxHops })
end

event.register("loaded", onLoaded)
