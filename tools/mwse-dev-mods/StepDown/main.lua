-- StepDown: lower the player through the water surface in fixed steps, for a dump series that shows
-- how the lighting changes with height (the "extra darkening underwater" sweep). Inert unless
-- MGE_STEP_DOWN is in the game's environment (forge-perf-run.sh clientEnv):
--
--     MGE_STEP_DOWN="delay:interval:step:below[:startAbove]"     e.g. "14:3:70:140:700"
--
--   delay       real seconds after load before the first step (let the weather ring park the sky)
--   interval    real seconds between steps (the host's dumpEvery must give >= 1 dump per step)
--   step        world units per step (70 ~ 1 m)
--   below       stop once the CAMERA is this far under the water level
--   startAbove  optional: at the first step, drop straight to the camera this far ABOVE the water
--               (a save high in the air would otherwise need a hundred steps to reach the surface)
--
-- The player is HELD at the target height every frame (a save in the air would otherwise fall), so
-- each step is an exact height. Every step is logged with the camera Z, and every dump's sidecar
-- carries the eye, so the pictures order themselves by height.

local cfg = os.getenv("MGE_STEP_DOWN")
if not cfg or cfg == "" or cfg == "0" then
    return
end
local delay, interval, step, below, startAbove =
    cfg:match("^([%d%.]+):([%d%.]+):([%d%.]+):([%d%.]+):?([%d%.]*)$")
delay, interval, step, below = tonumber(delay), tonumber(interval), tonumber(step), tonumber(below)
startAbove = tonumber(startAbove)
if not (delay and interval and step and below) then
    mwse.log("[stepdown] bad MGE_STEP_DOWN '%s' — expected delay:interval:step:below", cfg)
    return
end

local targetZ = nil
local steps = 0
local stepTimer = nil

local function waterLevel()
    local cell = tes3.player.cell
    return (cell and cell.waterLevel) or 0
end

local function onSimulate()
    if targetZ then
        tes3.player.position.z = targetZ
        if tes3.mobilePlayer then
            tes3.mobilePlayer.velocity = tes3vector3.new(0, 0, 0)
        end
    end
end

local function doStep()
    local cam = tes3.getCameraPosition()
    local wl = waterLevel()
    if steps == 0 and startAbove then
        -- One drop to the start height: the camera rides the player at a fixed offset.
        local camOff = cam.z - targetZ
        targetZ = wl + startAbove - camOff
        steps = 1
        mwse.log("[stepdown] dropped to start: playerZ=%.1f (camera %.0f above the water)", targetZ, startAbove)
        return
    end
    mwse.log("[stepdown] step %d playerZ=%.1f camZ=%.1f water=%.1f camBelow=%.1f",
             steps, targetZ, cam.z, wl, wl - cam.z)
    if wl - cam.z >= below then
        mwse.log("[stepdown] done: camera %.1f below the water after %d steps", wl - cam.z, steps)
        if stepTimer then stepTimer:cancel() end
        return
    end
    steps = steps + 1
    targetZ = targetZ - step
end

local function onLoaded()
    targetZ = tes3.player.position.z
    mwse.log("[stepdown] armed: start playerZ=%.1f delay=%.1fs interval=%.1fs step=%.0f below=%.0f",
             targetZ, delay, interval, step, below)
    timer.start({ type = timer.real, duration = delay, callback = function()
        doStep()
        stepTimer = timer.start({ type = timer.real, duration = interval, iterations = -1, callback = doStep })
    end })
end

event.register("loaded", onLoaded)
event.register("simulate", onSimulate)
