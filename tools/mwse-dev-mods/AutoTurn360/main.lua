-- Auto Turn 360 (dev aid for the post-load residency work, tasks/ + memory forge-postload-residency-walk)
--
-- WHY: the behind-camera pop-in bug only shows when the player TURNS after a load — the cache is
-- populated from the engine's frustum-limited classify set, so everything behind you is first-sight
-- geometry that arrives as a capture burst (rotation hitch) or a 32/frame trickle (pop-in). The perf
-- harness runs Morrowind minimized off a fixed save and never touches the mouse, so the one input
-- that triggers the bug is exactly the one nobody can supply unattended. This drives it.
--
-- WHAT: on load, waits `delay` real seconds, then sweeps the player through `degrees` over
-- `duration` seconds, then holds still for `settle` seconds. Every phase is logged to mwse.log so a
-- run is verifiable from logs alone — the game window is minimized and nobody is watching it.
--
-- Reads as ground truth, not intent: the sweep WRITES tes3.player.orientation and then reads back
-- tes3.mobilePlayer.facing, logging both. If the engine ignores the write (mouse-look fighting it,
-- API drift) the log says so instead of the run quietly measuring a camera that never moved.
--
-- CONFIG (Data Files/MWSE/config/AutoTurn360.json) — driven by mgeHost64/forge-popin-run.sh, which
-- rewrites it per repro and restores it on exit:
--   enabled  false = costs nothing, never registers a per-frame handler. DEFAULT OFF: this mod
--            takes the controls away from the player, so it must never engage during normal play.
--   delay    real seconds after `loaded` before the sweep starts. 0 = turn immediately (repro a);
--            12 = past the 600-frame kCaptureEpochGraceFrames, the worst case today (repro b).
--   duration real seconds for the whole sweep. ~3s is a brisk but human 360.
--   degrees  sweep size. 360 returns you to the start heading, so the run ends framing the same
--            view it began with — the cache/entry counts are then directly comparable end-to-end.
--   settle   real seconds to hold still afterwards, so a couple more [gc]/[hb] heartbeats land
--            AFTER the turn (that is where "did anything still trickle in?" is answered).
--   reload   reload the save ONCE before turning (repro c, "a far load, then turn"). This is the
--            path that actually hurts: a FIRST load runs its loading frames before the seam owns
--            them, so the full refresh walk still runs and pre-populates the cache (measured:
--            11705 entries already cached at the first [cell-purge]). A RELOAD has the host
--            already up, so those frames are liveDrawBuild and walk nothing — checkCellEpochAndPurge
--            purges to empty and the cache refills frustum-only. Same code path as a door /
--            fast-travel, and it needs no destination coordinates to automate.

local mod = "Auto Turn 360"
local version = "1.0"

local defaults = {
    enabled = false,
    delay = 2.0,
    duration = 3.0,
    degrees = 360,
    settle = 6.0,
    reload = false,
    -- Save basename (no .ess) for reload mode. Supplied by the harness, which pinned the save in the
    -- first place, rather than asked of the API: tes3.getLastLoadedFileName does not exist, and the
    -- error took the whole `loaded` callback down before it could log a word about why.
    saveFile = "",
}

-- Survives the reload: `loaded` fires again for the reloaded save, and only that second pass turns.
local didReload = false

local cfg = mwse.loadConfig("AutoTurn360", defaults)

local function log(fmt, ...)
    mwse.log("[autoturn] " .. fmt, ...)
end

local TWO_PI = 2 * math.pi

local sweeping = false
local swept = 0.0            -- simulated seconds accumulated into the sweep
local startFacing = 0.0
local totalRad = 0.0
local nextMark = 0.125       -- log at each eighth of the sweep
local wroteBad = 0           -- times the read-back facing disagreed with what we wrote

local function facingNow()
    local mp = tes3.mobilePlayer
    return mp and mp.facing or 0.0
end

-- Forward local: finish() unregisters the handler by identity, and the handler calls finish().
-- Declared local (not a bare global function) so nothing leaks into the shared MWSE environment.
local onSimulate

local function finish()
    if not sweeping then return end
    sweeping = false
    event.unregister("simulate", onSimulate)
    log("SWEEP DONE: %.0f deg in %.2fs sim | facing start=%.1f end=%.1f deg | writes-ignored=%d",
        cfg.degrees, swept, math.deg(startFacing) % 360, math.deg(facingNow()) % 360, wroteBad)
    if wroteBad > 0 then
        log("!! the engine did NOT take %d orientation writes - this run did not actually turn", wroteBad)
    end
    -- Hold still so post-turn heartbeats land in the log before the harness kills us.
    timer.start({
        type = timer.real,
        duration = cfg.settle,
        iterations = 1,
        callback = function()
            log("SETTLED: %.1fs after the sweep, facing=%.1f deg", cfg.settle, math.deg(facingNow()) % 360)
        end,
    })
end

onSimulate = function(e)
    if not sweeping then return end
    swept = swept + e.delta
    local t = swept / cfg.duration
    if t > 1.0 then t = 1.0 end

    local want = (startFacing + totalRad * t) % TWO_PI
    -- Player pitch lives on the camera, not the reference: keep x/y at zero so a standing player
    -- stays upright while only the heading sweeps.
    tes3.player.orientation = tes3vector3.new(0, 0, want)

    -- Ground truth: did that write take? Compare on the circle (wrap-safe).
    local got = facingNow()
    local err = math.abs(((want - got + math.pi) % TWO_PI) - math.pi)
    if err > math.rad(5) then
        wroteBad = wroteBad + 1
    end

    if t >= nextMark then
        log("sweep %3.0f%%: wrote=%.1f read=%.1f deg (sim %.2fs)",
            t * 100, math.deg(want) % 360, math.deg(got) % 360, swept)
        nextMark = nextMark + 0.125
    end

    if t >= 1.0 then finish() end
end

local function beginSweep()
    startFacing = facingNow()
    totalRad = math.rad(cfg.degrees)
    swept = 0.0
    nextMark = 0.125
    wroteBad = 0
    sweeping = true
    log("SWEEP START: %.0f deg over %.1fs, from facing=%.1f deg",
        cfg.degrees, cfg.duration, math.deg(startFacing) % 360)
    event.register("simulate", onSimulate)
end

local function onLoaded()
    if not cfg.enabled then
        log("disabled (enabled=false) - the player will not be turned")
        return
    end
    local save = cfg.saveFile
    local wantReload = cfg.reload and not didReload and save ~= nil and save ~= ""
    if cfg.reload and not didReload and not wantReload then
        log("!! reload requested but saveFile is empty - turning without the reload instead")
    end
    if wantReload then
        didReload = true
        log("RELOAD FIRST: reloading '%s' to force the purge path, then turning on the way back in", save)
        -- Defer: this handler is itself running inside the loader (the instant-load mod calls
        -- loadGame, which triggers `loaded`), so reloading from here re-enters the load machinery.
        timer.start({
            type = timer.real, duration = 1.0, iterations = 1,
            callback = function()
                local ok, err = pcall(tes3.loadGame, save)
                if not ok then log("!! loadGame('%s') FAILED: %s", save, tostring(err)) end
            end,
        })
        return
    end

    log("armed: delay=%.1fs duration=%.1fs degrees=%.0f settle=%.1fs reloaded=%s",
        cfg.delay, cfg.duration, cfg.degrees, cfg.settle, tostring(didReload))
    -- REAL time, not simulate: `delay` is meant to be wall-clock ("turn immediately" vs "wait past
    -- the capture grace window"), and it must keep ticking through the load's frame stalls.
    timer.start({
        type = timer.real,
        duration = math.max(cfg.delay, 0.001),
        iterations = 1,
        callback = beginSweep,
    })
end

event.register("loaded", onLoaded)
