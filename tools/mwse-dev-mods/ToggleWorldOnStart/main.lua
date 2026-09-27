-- Toggle AI On Start (was Toggle World On Start)
--
-- Dev aid for the Forge renderer takeover: on every game load, sends one console command:
--   ToggleAI (TAI) - freezes all actor AI, so autospawned fish (water is everywhere) and other
--                    creatures stop moving -> no per-frame dynamic-mesh uploads polluting perf.
-- ToggleWorld was REMOVED 2026-07-02: the auto-load test scene is now the densest city view and
-- MW's world must render (it also starved the async host-frame seam — tasks/async-frame-split.md).
-- Runs on every load because MW resets the toggle back ON when a save loads.

local mod = "Toggle AI On Start"
local version = "1.2"

local function onLoaded()
    -- Defer one frame: at the `loaded` event the world scene graph is still settling; running the
    -- console command a frame later applies reliably.
    timer.delayOneFrame(function()
        tes3.runLegacyScript({ command = "ToggleAI" })
        mwse.log("[%s %s] sent ToggleAI (TAI) on load.", mod, version)
    end)
end

event.register("loaded", onLoaded)
