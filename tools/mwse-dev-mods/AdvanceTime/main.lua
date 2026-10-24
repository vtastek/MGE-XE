-- AdvanceTime: a wait/rest from a script, for the minimized harness — anything that has to survive the
-- clock jumping hours at once (exposure, sky, weather) without someone at the rest menu. Uses MW's own
-- rest step (tes3.advanceTime, one hour per call of the engine's rest update), so the environment is
-- updated the way a real wait updates it. Inert unless MGE_ADVANCE_TIME=<hours>:<delay s> is in the
-- game's environment (forge-perf-run.sh clientEnv); the jump happens <delay> real seconds after load.

local spec = os.getenv("MGE_ADVANCE_TIME")
if not spec then
    return
end
local hours, delay = spec:match("^([%d%.]+):([%d%.]+)$")
if not hours then
    mwse.log("[advancetime] !! bad MGE_ADVANCE_TIME '%s' (want hours:delaySeconds)", spec)
    return
end
hours, delay = tonumber(hours), tonumber(delay)

local function onLoaded()
    mwse.log("[advancetime] armed: %.2f h in %.1f s (hour now %.2f)", hours, delay, tes3.worldController.hour.value)
    timer.start({
        type = timer.real, duration = math.max(delay, 0.001), iterations = 1,
        callback = function()
            local before = tes3.worldController.hour.value
            local passed = tes3.advanceTime({ hours = hours, resting = false })
            mwse.log("[advancetime] waited %.2f h: hour %.2f -> %.2f", passed, before, tes3.worldController.hour.value)
        end,
    })
end

event.register("loaded", onLoaded)
