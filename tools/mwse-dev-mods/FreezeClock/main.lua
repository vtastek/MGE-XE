-- FreezeClock: pin MW's game clock for image A/B oracles. Inert unless MGE_FREEZE_CLOCK=1 is in the
-- game's environment (forge-perf-run.sh clientEnv), so leaving it deployed changes nothing.
--
-- WHY: a dump at a fixed FRAME is not a dump at a fixed TIME. Load frames advance sim time, so an arm
-- that loads faster reaches frame 2000 earlier in game time — a different sun angle and weather blend,
-- i.e. a frame-wide difference that has nothing to do with the change being tested (geometry dedup:
-- ON dumped at weather t=0.272 vs OFF's 0.281/0.282). Timescale 0 at `loaded` holds gameHour (and the
-- weather transition, which runs on game time) at the save's own value in every arm.

if os.getenv("MGE_FREEZE_CLOCK") ~= "1" then
    return
end

local function onLoaded()
    local wc = tes3.worldController
    mwse.log("[freezeclock] gameHour=%.4f timescale %.2f -> 0", wc.hour.value, wc.timescale.value)
    wc.timescale.value = 0
end

event.register("loaded", onLoaded)
