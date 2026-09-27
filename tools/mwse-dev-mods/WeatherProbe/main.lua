-- Weather Probe (dev aid for the sky/weather takeover)
--
-- WHY: the submerged sky is dim, and F11 proves it is dim in the DX9 baseline too — so nothing the
-- Forge host does is responsible. MW runs ONE global weather state, switches part of it to
-- underwater values the instant the eye goes under, and the sky inherits that. The client then
-- copies the result to the host, faithfully, including the dim.
--
-- Every host-side fix tried against this failed, and they failed for the same reason: they were
-- corrections applied to an input that was already wrong. This probe goes at the input instead.
--
-- The MGE side of the probe is in mgeXE.log as `[wthr]` (needs LogDistantPipeline): it logs
-- currentSkyColor/currentFogColor NEXT TO the sky meshes' live per-vertex colours, which the engine
-- derives separately, so it can tell WHICH of the two the water tint lands on. This side does the
-- thing that log cannot: it WRITES.
--
-- ── THE CONTROL EXPERIMENT (`uwWeight`) ────────────────────────────────────────────────────────
-- MW's underwater tint is a plain linear blend with two authored operands, both writable from here:
--
--     colour' = colour * (1 - underwaterColorWeight) + underwaterColor * underwaterColorWeight
--
-- with Morrowind.ini shipping UnderwaterColor=012,030,037 and UnderwaterColorWeight=0.85. Both are
-- INPUTS to MW's own per-frame colour update, so setting the weight to 0 before that update runs
-- disables the blend at source — no inversion, no latch, no ordering subtlety.
--
--   sky stops dimming with uwWeight=0  -> confirmed, and the fix is this knob (or the host reading
--                                        past it). Everything else can be deleted.
--   sky still dims                     -> the dim is NOT this blend, and the `[wthr]` sky-vs-vertex
--                                        columns in mgeXE.log say where else to look.
--
-- Inverting the blend after the fact was considered and rejected: at w=0.85 the above-water signal
-- is 15% of an 8-bit vertex colour, so undoing it multiplies quantisation by 6.7 and bands the sky.
--
-- ── THE SECOND EXPERIMENT (`holdSky`) ──────────────────────────────────────────────────────────
-- Latch currentSkyColor/currentFogColor above water, write them back every frame while submerged.
-- ⚠ This one may legitimately do NOTHING, and that is a result rather than a failure: MW recomputes
-- both values inside its own colour update, so whether the write survives depends on where
-- `simulate` lands relative to that update. If holdSky visibly works, the write lands after MW's
-- update; if it does not, it lands before, and only input-side knobs like uwWeight can ever work.
-- Do not read a null result here as "the weather state is not the cause".
--
-- ── WHAT THE PASSIVE LOG IS FOR ────────────────────────────────────────────────────────────────
-- Beyond the dive: the standing question of whether MW keeps simulating weather underwater. Rain
-- and thunder are reported to continue while submerged, which would mean only the presentation
-- changes. `particles`, `rainActive` and the weather transition events answer that directly — and a
-- takeover has to reproduce whatever they say.
--
-- It also logs the input set a GENERATED sky would actually need: sun angle, cloud coverage and
-- speed, fog depth, wind. In a physical sky every authored COLOUR below becomes an optional tint.
--
-- CONFIG (Data Files/MWSE/config/WeatherProbe.json):
--   enabled    false = no handlers at all.
--   interval   real seconds between passive samples (crossings always log, regardless).
--   uwWeight   -1 = leave MW's underwaterColorWeight alone. 0 = the control experiment above.
--   holdSky    hold pre-dive currentSkyColor/currentFogColor while submerged.
--   keyWeight  toggles uwWeight between -1 and 0 in game. Default Numpad 6.
--   keyHold    toggles holdSky in game. Default Numpad 7.
-- Both hotkeys exist so the A/B is one keypress with the water in frame, instead of a reload.
--
-- Raw DirectInput scan codes, on ControllerProbe's precedent — no scanCode-name guessing, and the
-- keypad because MGE's own bindings are enumerated there: MGE claims Numpad 0, 4, + - * / . and
-- Scroll Lock, ControllerProbe took 1/2/3/5, which leaves 6 and 7. Check renderprocess.cpp's
-- GetAsyncKeyState block before moving these.

local defaults = {
    enabled = true,
    interval = 0.25,
    uwWeight = -1,
    holdSky = false,
    keyWeight = 0x4D,   -- Numpad 6
    keyHold = 0x47,     -- Numpad 7
}

local cfg = mwse.loadConfig("WeatherProbe", defaults)

local function log(fmt, ...)
    mwse.log("[wthrprobe] " .. fmt, ...)
end

local function v3(c)
    if not c then return "(nil)" end
    return string.format("(%.3f %.3f %.3f)", c.r or c.x, c.g or c.y, c.b or c.z)
end

-- Two underwater answers, deliberately kept apart. `eye` is the geometric test MGE itself uses
-- (camera below the cell's water plane), `snd` is the engine's OWN underwater bit, set by MW when it
-- swaps to the submerged ambient loop. If they ever disagree, the dim is keyed to whichever one
-- moved with it — and MGE's fog swap and waterParams[7] both ride on the geometric one.
local function underwater()
    local cell = tes3.player and tes3.player.cell
    local eye = false
    if cell and cell.hasWater then
        local p = tes3.getPlayerEyePosition()
        if p then eye = p.z < (cell.waterLevel or 0) end
    end
    local wc = tes3.worldController and tes3.worldController.weatherController
    local w = wc and wc.currentWeather
    local snd = w and w.underwaterSoundState or false
    return eye, snd
end

-- Restored on surfacing so a session that ends submerged does not leave MW's ini value clobbered
-- in the save's weather state.
local savedWeight = nil
local heldSky, heldFog = nil, nil

local function applyOverrides(wc, isUnder)
    if isUnder then
        if cfg.uwWeight >= 0 then
            if savedWeight == nil then
                savedWeight = wc.underwaterColorWeight
                log("uwWeight OVERRIDE %.2f -> %.2f", savedWeight, cfg.uwWeight)
            end
            wc.underwaterColorWeight = cfg.uwWeight
        end
        if cfg.holdSky and heldSky then
            wc.currentSkyColor = heldSky
            wc.currentFogColor = heldFog
        end
    else
        if savedWeight ~= nil then
            wc.underwaterColorWeight = savedWeight
            log("uwWeight restored -> %.2f", savedWeight)
            savedWeight = nil
        end
        -- Latch every above-water frame, so the held value is genuinely "the sky one frame before
        -- the dive" rather than whatever the weather was when the mod loaded.
        local s, f = wc.currentSkyColor, wc.currentFogColor
        heldSky = tes3vector3.new(s.r, s.g, s.b)
        heldFog = tes3vector3.new(f.r, f.g, f.b)
    end
end

-- The live precipitation count, and the whole reason it is here: if this keeps moving while the
-- camera is submerged, MW never stopped simulating the weather and only the look changed.
-- particlesActive is an NI iterated list; `#` is the cheap path and iteration the fallback, both
-- guarded because a binding that does not support either must not take the probe down with it.
local function countParticles(wc)
    local list = wc.particlesActive
    if not list then return -1 end
    local ok, n = pcall(function() return #list end)
    if ok and type(n) == "number" then return n end
    n = 0
    ok = pcall(function()
        for _ in pairs(list) do n = n + 1 end
    end)
    return ok and n or -1
end

local lastSample = 0
local lastUnder = nil
local lastWeather = nil

local function sample(force, tag)
    local wc = tes3.worldController and tes3.worldController.weatherController
    if not wc then return end
    local w = wc.currentWeather
    if not w then return end

    local eye, snd = underwater()
    applyOverrides(wc, eye)

    local now = os.clock()
    local edge = (lastUnder ~= nil and lastUnder ~= eye)
    if not (force or edge) and (now - lastSample) < cfg.interval then
        lastUnder = eye
        return
    end
    lastSample = now
    lastUnder = eye

    local nx = wc.nextWeather
    log("%s under=%d/%d w=%s->%s t=%.2f | sky=%s fog=%s | uwCol=%s uwW=%.2f"
        .. " | rain=%s particles=%d hour=%.2f",
        edge and "EDGE" or (tag or "    "),
        eye and 1 or 0, snd and 1 or 0,
        w.name or "?", nx and nx.name or "?",
        wc.transitionScalar,
        v3(wc.currentSkyColor), v3(wc.currentFogColor),
        v3(wc.underwaterColor), wc.underwaterColorWeight,
        tostring(w.rainActive), countParticles(wc),
        tes3.worldController.hour.value)

    -- The slow half, printed only when it can have changed. This is the physical-sky input set:
    -- sun angle plus cloud coverage carry the whole look, and the colours below are the tints a
    -- generated sky would optionally honour.
    if force or edge or lastWeather ~= w.index then
        lastWeather = w.index
        local light = wc.sceneSkyLight
        log("     inputs: clouds=%.1f%% spd=%.2f tex='%s' | landFog day=%.2f night=%.2f wind=%.2f"
            .. " | skyDay=%s fogDay=%s | skyLight amb=%s diff=%s"
            .. " | uwFog rise=%.1f day=%.1f set=%.1f night=%.1f indoor=%.1f",
            w.cloudsMaxPercent, w.cloudsSpeed, w.cloudTexture or "(none)",
            w.landFogDayDepth, w.landFogNightDepth, w.windSpeed,
            v3(w.skyDayColor), v3(w.fogDayColor),
            light and v3(light.ambient) or "(no light)",
            light and v3(light.diffuse) or "(no light)",
            wc.underwaterSunriseFog, wc.underwaterDayFog, wc.underwaterSunsetFog,
            wc.underwaterNightFog, wc.underwaterIndoorFog)
    end
end

local function onSimulate()
    sample(false, nil)
end

-- Weather CONTINUES while submerged or it does not; these three events are the witnesses that do
-- not depend on anything being visible. A transition that starts and finishes underwater settles it.
local function onTransitionStarted()
    local eye = underwater()
    log("EVENT transition STARTED (under=%d)", eye and 1 or 0)
    sample(true, "evt ")
end

local function onTransitionFinished()
    local eye = underwater()
    log("EVENT transition FINISHED (under=%d)", eye and 1 or 0)
    sample(true, "evt ")
end

local function onWeatherCycled()
    log("EVENT weather CYCLED")
    sample(true, "evt ")
end

local function onKeyDown(e)
    if e.keyCode == cfg.keyWeight then
        cfg.uwWeight = (cfg.uwWeight >= 0) and -1 or 0
        if cfg.uwWeight < 0 then
            -- Put MW's own value back immediately; waiting for the next surfacing would leave the
            -- A/B showing the override for as long as the player stays under.
            local wc = tes3.worldController and tes3.worldController.weatherController
            if wc and savedWeight ~= nil then
                wc.underwaterColorWeight = savedWeight
                savedWeight = nil
            end
        end
        log("HOTKEY uwWeight = %s", cfg.uwWeight < 0 and "off (MW's own)" or tostring(cfg.uwWeight))
        tes3.messageBox("WeatherProbe: underwater tint %s",
            cfg.uwWeight < 0 and "ON (MW default)" or "OFF (weight 0)")
        sample(true, "key ")
    elseif e.keyCode == cfg.keyHold then
        cfg.holdSky = not cfg.holdSky
        log("HOTKEY holdSky = %s", tostring(cfg.holdSky))
        tes3.messageBox("WeatherProbe: hold pre-dive sky %s", cfg.holdSky and "ON" or "OFF")
        sample(true, "key ")
    end
end

local function onLoaded()
    if not cfg.enabled then
        log("disabled (enabled=false)")
        return
    end
    lastUnder, lastWeather, savedWeight = nil, nil, nil
    heldSky, heldFog = nil, nil
    log("armed: interval=%.2fs uwWeight=%s holdSky=%s (F3 / F4 toggle)",
        cfg.interval, tostring(cfg.uwWeight), tostring(cfg.holdSky))
    sample(true, "load")
    event.register("simulate", onSimulate)
    event.register("keyDown", onKeyDown)
    event.register("weatherTransitionStarted", onTransitionStarted)
    event.register("weatherTransitionFinished", onTransitionFinished)
    event.register("weatherCycled", onWeatherCycled)
end

event.register("loaded", onLoaded)
