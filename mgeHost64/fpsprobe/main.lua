-- MGE XE - CLIENT FRAME-TIME PROBE (test fixture, NOT a gameplay mod).
--
-- WHY THIS EXISTS. Comparing our build against Greatness7's DX9 fork needs ONE number that exists
-- on BOTH sides. Ours reports `gpu split:` from mgeHost64.log; theirs has no such line, and its own
-- internal timings have no counterpart here. Every per-build instrument measures a different frame.
-- What both builds unambiguously share is the CLIENT frame: Morrowind.exe asking for the next frame
-- and getting it. The interval between successive `enterFrame` callbacks is that frame, timed by the
-- same MWSE build in both installs, so the two rows differ by the renderer under test, not the ruler.
--
-- ⚠ `min` IS THE HEADLINE, not p50. A median over a window measures how many STALLED frames that
-- window happened to catch, not what the work costs - one wandering ~2 ms stall wearing a different
-- pass's name each frame is exactly how a whole table of per-phase medians turned out to be fiction
-- (see feedback_one_stall_wearing_every_passs_name). The floor is the closest thing to "what this
-- renderer costs when nothing external is in the way", so the floor is what gets compared. p50/p95
-- are printed alongside because a build that wins on the floor and loses badly at p95 is a
-- different result from one that wins on both, and the table should be able to say which.
--
-- ⚠ NOT MARKER-GATED, unlike cellchurn/velchurn. Those teleport the player, so an un-noticed arming
-- would be indistinguishable from possession. This one only READS - a table append per frame and a
-- log line every kWindow frames - so gating it would add a way to lose a whole measurement run to a
-- forgotten file, and buy nothing.
--
-- ⚠ MENU FRAMES ARE NOT FRAMES OF THE SCENE. Inventory and dialogue draw a fraction of the world;
-- folding them in would let a run that spent time in a menu report a better floor. Dropped, and
-- counted separately so a window that was mostly menu is visible as such rather than silently thin.
--
-- ⚠⚠ NOT `enterFrame.delta`. That field comes from MW's own WorldController, whose clock is
-- millisecond-granular: the first run of this probe reported min=14.00 p50=15.00 p95=16.00 -- every
-- statistic an exact integer. On a ~15 ms frame that is a 6.7% quantiser, and it is BIASED for the
-- headline: a floor read through it is the floor rounded DOWN, so two builds whose true floors are
-- 14.2 and 14.6 ms both print 14.00 and the comparison silently reports a tie. os.getHighPrecisionClock
-- is MWSE's profiling clock; the interval between successive enterFrame callbacks is the same frame,
-- measured with a ruler that can tell those two builds apart.
local kWindow = 600      -- frames per reported line; ~7 s at 85 fps. PRINTED in the line so the
                         -- "both rows summarise the same number of frames" check reads the
                         -- artifact rather than trusting two copies of this source.
local kSettle = 10.0     -- seconds after `loaded` before sampling starts, so first-sight texture
                         -- upload and shader warm-up for the starting cell stay out of the floor
local kHitch  = 1.0      -- seconds; deltas above this are load screens / alt-tabs, not frames.
                         -- Excluded from the statistics but COUNTED, because a window that
                         -- silently dropped a third of its samples is not the same measurement.

-- Guarded rather than called blind: if a future MWSE drops the profiling clock, this must say so in
-- the log and fall back, not quietly resume reporting integers under the same header.
local hpc = os.getHighPrecisionClock
if not hpc then
    mwse.log("[fpsprobe] WARNING: os.getHighPrecisionClock absent - falling back to enterFrame.delta, "
             .. "which is MILLISECOND-QUANTISED. Treat every number below as +/- 1 ms.")
end

local samples = {}
local nMenu, nHitch = 0, 0
local sampling = false
local startCell = nil
local last = nil        -- clock reading at the previous COUNTED frame; nil after any gap

local kWeatherName = {
    [0] = "clear", [1] = "cloudy", [2] = "foggy",  [3] = "overcast", [4] = "rain",
    [5] = "thunder", [6] = "ash",  [7] = "blight", [8] = "snow",     [9] = "blizzard",
}

local function cellName()
    local c = tes3.getPlayerCell()
    if not c then return "?" end
    return c.editorName or c.id or "?"
end

-- The weather is read HERE rather than left to each build's own log because the two builds do not
-- log the same things, and because loading a save RE-ROLLS the weather (project_forge_apl_weather_roll).
-- A pair of rows taken under clear and under ashstorm is not a comparison, and the only way to throw
-- such a pair away afterwards is for both rows to have carried their weather from the same source.
local function weatherName()
    local wc = tes3.worldController and tes3.worldController.weatherController
    local w = wc and wc.currentWeather
    if not w then return "?" end
    return kWeatherName[w.index] or tostring(w.index)
end

local function report()
    table.sort(samples)
    local n = #samples
    local sum = 0
    for i = 1, n do sum = sum + samples[i] end
    -- Nearest-rank percentiles: no interpolation, so every printed value is a frame that actually
    -- happened rather than a blend of two that did not.
    local p50 = samples[math.ceil(0.50 * n)]
    local p95 = samples[math.ceil(0.95 * n)]
    local cell = cellName()
    mwse.log("[fpsprobe] n=%d min=%.2f p50=%.2f p95=%.2f mean=%.2f menu=%d hitch=%d weather=%s cell=%q%s",
             n, samples[1], p50, p95, sum / n, nMenu, nHitch, weatherName(), cell,
             (startCell and cell ~= startCell) and (' moved_from=%q'):format(startCell) or "")
end

local function onFrame(e)
    if not sampling then return end
    -- `last` is cleared on every skip. An interval that spans a menu, a hitch or the settle period
    -- is not a frame time, and letting one through would put a multi-second sample in the window
    -- that only p95 and mean would ever show.
    if e.menuMode then nMenu = nMenu + 1 last = nil return end

    local dt
    if hpc then
        local now = hpc()
        dt = last and (now - last) or nil
        last = now
        if not dt then return end
    else
        dt = e.delta
    end
    if not dt or dt <= 0 then return end
    if dt > kHitch then nHitch = nHitch + 1 last = nil return end

    if #samples == 0 then startCell = cellName() end
    samples[#samples + 1] = dt * 1000.0
    if #samples >= kWindow then
        report()
        samples, nMenu, nHitch = {}, 0, 0
        startCell = nil
    end
end

local function onLoaded()
    -- Reset on EVERY load, not just the first. The harness pins one save per run, but a run that
    -- reloads would otherwise carry the previous scene's frames into the first window of the next.
    samples, nMenu, nHitch = {}, 0, 0
    startCell, last = nil, nil
    sampling = false
    mwse.log("[fpsprobe] loaded - settling %.0f s, then one line per %d play frames", kSettle, kWindow)
    timer.start({
        duration = kSettle, type = timer.real, iterations = 1,
        callback = function()
            sampling = true
            last = nil
            mwse.log("[fpsprobe] sampling STARTED in %q", cellName())
        end,
    })
end

-- "enterFrame", not "frame": registering "frame" silently never fires, which is how an earlier
-- probe reported zero of everything while the thing it watched was demonstrably happening.
event.register("enterFrame", onFrame)
event.register("loaded", onLoaded)
