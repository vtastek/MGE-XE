-- MGE XE - NO-AI FIXTURE (test fixture, NOT a gameplay mod).
--
-- An unattended harness run stands the player still for minutes; on a save near hostiles (Ebonheart's
-- castle) something attacks it, and the run measures a fight, or a death screen, instead of the scene.
-- This runs the console's ToggleAI ("tai", user 2026-10-12: "use tai, to stop getting attacked") once
-- after every load.
--
-- ⚠ INERT UNLESS MGE_NOAI=1 is in the game's environment — only the perf harness sets it (clientEnv),
-- so a normal launch never has AI switched off under the player.
if os.getenv("MGE_NOAI") ~= "1" then return end

mwse.log("[noai] ARMED (MGE_NOAI=1) - ToggleAI runs after every load")

local function onLoaded()
    tes3.runLegacyScript({ command = "ToggleAI" })
    mwse.log("[noai] ToggleAI run")
end

event.register("loaded", onLoaded)
