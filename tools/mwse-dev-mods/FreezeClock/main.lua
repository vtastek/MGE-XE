-- FreezeClock: pin MW's game clock for image A/B oracles. Inert unless MGE_FREEZE_CLOCK=1 or
-- MGE_SET_HOUR=<h> is in the game's environment (forge-perf-run.sh clientEnv), so leaving it deployed
-- changes nothing.
--
-- WHY: a dump at a fixed FRAME is not a dump at a fixed TIME. Load frames advance sim time, so an arm
-- that loads faster reaches frame 2000 earlier in game time — a different sun angle and weather blend,
-- i.e. a frame-wide difference that has nothing to do with the change being tested (geometry dedup:
-- ON dumped at weather t=0.272 vs OFF's 0.281/0.282). Timescale 0 at `loaded` holds gameHour (and the
-- weather transition, which runs on game time) at the save's own value in every arm.
--
-- MGE_SET_HOUR=<0..24> moves the clock to that hour at `loaded` first: one save, any time of day
-- (forge-weather-pics.sh's time-of-day sweeps). It implies the freeze, so the hour stays put.

local freeze = os.getenv("MGE_FREEZE_CLOCK") == "1"
local setHour = tonumber(os.getenv("MGE_SET_HOUR") or "")
if not freeze and not setHour then
    return
end

local function onLoaded()
    local wc = tes3.worldController
    if setHour then
        mwse.log("[freezeclock] gameHour %.4f -> %.4f (MGE_SET_HOUR)", wc.hour.value, setHour)
        wc.hour.value = setHour
    end
    mwse.log("[freezeclock] gameHour=%.4f timescale %.2f -> 0", wc.hour.value, wc.timescale.value)
    wc.timescale.value = 0
end

event.register("loaded", onLoaded)
