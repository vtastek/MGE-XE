-- Toggle World On Start
--
-- Dev aid for the Forge renderer takeover: on every game load, sends two console commands so the
-- scene is static and MW's world is hidden while its sky stays live:
--   ToggleWorld (TW)  - toggles MW's own WORLD geometry rendering OFF but KEEPS the sky, so only the
--                       Forge D3D12 composite shows over MW's (still-live) sky.
--   ToggleAI    (TAI) - freezes all actor AI, so autospawned fish (water is everywhere) and other
--                       creatures stop moving -> no per-frame dynamic-mesh uploads polluting perf.
-- Runs on every load because MW resets both toggles back ON when a save loads, so re-applying them on
-- each load lands world OFF + AI OFF every time.

local mod = "Toggle World On Start"
local version = "1.1"

local function onLoaded()
    -- Defer one frame: at the `loaded` event the world scene graph is still settling; running the
    -- console commands a frame later applies reliably.
    timer.delayOneFrame(function()
        tes3.runLegacyScript({ command = "ToggleWorld" })
        tes3.runLegacyScript({ command = "ToggleAI" })
        mwse.log("[%s %s] sent ToggleWorld (TW) + ToggleAI (TAI) on load.", mod, version)
    end)
end

event.register("loaded", onLoaded)
