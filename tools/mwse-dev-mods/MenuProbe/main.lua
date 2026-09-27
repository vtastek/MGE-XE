-- Menu Probe (dev aid for the DX9-retirement work, tasks/dx9-retirement.md)
--
-- WHY: "in menus, FPS is low". After S2, menu frames take the same early-Forge-kickoff shape as
-- play frames, so the residual cost has to be measured, not guessed. MGE's [s2] counter already
-- splits mean frame period play-vs-menu — but the perf harness runs Morrowind MINIMIZED off a fixed
-- save and never opens a menu, so the menu bucket always reads 0 and nobody can watch the window to
-- drive one by hand.
--
-- WHAT: cycles menu mode open/closed on a REAL-time timer so the harness collects both buckets
-- unattended. Everything it does is logged to mwse.log, so a run is verifiable from logs alone —
-- no need to see the game window.
--
-- Reads as ground truth, not intent: enterMenuMode's return value is recorded, and the NEXT tick
-- re-reads tes3ui.menuMode() and logs what actually happened. A menu that refused to open says so.
--
-- Turn OFF (set ENABLED = false) once the menu cost is characterised — it makes the game unplayable
-- by design.

local mod = "Menu Probe"
local version = "1.0"

local ENABLED = false   -- OFF by default: it drives menus on a timer and makes the game unplayable.

-- Real seconds in each state. Long enough that MGE's 600-frame [s2] window catches whole runs of
-- menu frames rather than a smear across the open/close transitions.
local CLOSED_SECS = 8
local OPEN_SECS = 8

-- Tried in order; the first that actually reaches menu mode wins and is reused thereafter.
-- MenuInventory is the one users mean by "menus" (the heaviest UI: paperdoll + item tile grid).
local CANDIDATES = { "MenuInventory", "MenuStat", "MenuMap" }

local function log(fmt, ...)
    mwse.log("[%s %s] " .. fmt, mod, version, ...)
end

local state = "closed"      -- "closed" | "opening" | "open" | "closing"
local elapsed = 0.0
local candidateIx = 1
local chosen = nil          -- latched once something works
local cycles = 0
local menuFrames, playFrames = 0, 0

-- Per-frame ground truth: requesting menu mode and being in it are different things, so count what
-- actually happened. Must be the `frame` event, NOT `simulate` — simulate does not fire in menu
-- mode, so it would report zero menu frames no matter how well the probe worked. The event is named
-- `enterFrame` (MWSE LuaFrameEvent.cpp:7) — registering "frame" silently never fires, which is
-- exactly how the first run reported menu=0 play=0 while the menus were demonstrably opening.
local function onFrame(e)
    if e.menuMode then
        menuFrames = menuFrames + 1
    else
        playFrames = playFrames + 1
    end
end

local function tryOpen()
    local id = chosen or CANDIDATES[candidateIx]
    local ok, ret = pcall(tes3ui.enterMenuMode, id)
    if not ok then
        log("enterMenuMode('%s') ERRORED: %s", tostring(id), tostring(ret))
        return false, id
    end
    return ret and true or false, id
end

local function onTick()
    elapsed = elapsed + 1.0

    if state == "closed" then
        if elapsed >= CLOSED_SECS then
            local requested, id = tryOpen()
            log("open requested id='%s' -> %s", tostring(id), tostring(requested))
            state = "opening"
            elapsed = 0.0
        end

    elseif state == "opening" then
        -- Ground truth one tick later: did menu mode actually engage?
        if tes3ui.menuMode() then
            chosen = chosen or CANDIDATES[candidateIx]
            log("MENU MODE CONFIRMED via '%s' (cycle %d)", tostring(chosen), cycles + 1)
            state = "open"
            elapsed = 0.0
        else
            -- That candidate did not take. Advance and retry next cycle.
            log("menu mode did NOT engage for '%s'", tostring(CANDIDATES[candidateIx]))
            if not chosen then
                candidateIx = candidateIx + 1
                if candidateIx > #CANDIDATES then
                    candidateIx = 1
                    log("!! all candidates failed; will keep retrying from the top")
                end
            end
            state = "closed"
            elapsed = 0.0
        end

    elseif state == "open" then
        if elapsed >= OPEN_SECS then
            local ok, ret = pcall(tes3ui.leaveMenuMode)
            log("close requested -> %s", tostring(ok and ret))
            state = "closing"
            elapsed = 0.0
        end

    elseif state == "closing" then
        cycles = cycles + 1
        local total = menuFrames + playFrames
        log("cycle %d done | frames since load: menu=%d play=%d (menu share %.1f%%) | still in menu mode: %s",
            cycles, menuFrames, playFrames,
            total > 0 and (100.0 * menuFrames / total) or 0.0,
            tostring(tes3ui.menuMode()))
        state = "closed"
        elapsed = 0.0
    end
end

local function onLoaded()
    if not ENABLED then
        log("disabled (ENABLED=false) - no menus will be driven")
        return
    end
    -- Defer a frame: at `loaded` the UI/scene graph is still settling (same reason
    -- ToggleWorldOnStart defers its console command).
    timer.delayOneFrame(function()
        log("armed: %ds closed / %ds open, candidates=%s",
            CLOSED_SECS, OPEN_SECS, table.concat(CANDIDATES, ","))
        -- REAL time, not simulate: simulate timers stop ticking in menu mode, which would open a
        -- menu and then never close it.
        timer.start({
            type = timer.real,
            duration = 1.0,
            iterations = -1,
            callback = onTick,
        })
    end)
end

event.register("loaded", onLoaded)
-- Gated too: a disabled probe must cost NOTHING per frame, not just skip driving menus.
if ENABLED then
    event.register("enterFrame", onFrame)
end
