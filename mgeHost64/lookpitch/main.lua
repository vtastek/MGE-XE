-- MGE XE — GRASS OCCLUSION PROBE (test fixture, NOT a gameplay mod).
--
-- WHY THIS EXISTS. Grass is ~3.5 ms of the exterior frame, the single largest term, and its cull is
-- range + frustum + nearcut + vis-group. Whether the Hi-Z occlusion stage rejects ANY grass cannot
-- be read out of the log as it stands: the `[grass] cpu-check` line re-runs range+frustum ONLY (a
-- CPU cannot sample the depth pyramid), and `hizOccl` is a counter shared with the statics lane.
--
-- The discriminator is therefore CPU `pass*64` (range+frustum) against GPU `drawn`
-- (range+frustum+Hi-Z). On a level camera those read 3392 +/- 470 vs 3242 — consistent with
-- occlusion rejecting nothing, but the 1-in-64 stride's noise is wider than the effect, so it
-- proves nothing on its own.
--
-- Pitching the camera into the ground makes near terrain occlude essentially everything behind it.
-- If Hi-Z runs on grass, `drawn` MUST fall away from `pass*64`. If the two stay locked together at
-- the same ratio they have level, the occlusion stage is rejecting nothing.
--
-- ⚠ THE FRUSTUM CONFOUND, AND WHY THE COMPARISON IS A RATIO. Looking down also shrinks the frustum
-- survivor set, so `drawn` collapsing is NOT by itself evidence of occlusion — a pure frustum cull
-- would do the same. The CPU line moves with the frustum too, because it re-runs the same frustum
-- test. So the quantity to read is drawn / (pass*64): frustum cancels, and only Hi-Z can move it.
--
-- ⚠⚠ INERT UNLESS THE MARKER FILE EXISTS. This mod pins the player's view direction; running it
-- during real play would be indistinguishable from possession. It does NOTHING unless `ACTIVE` sits
-- beside this file. Same contract as mgexe/velchurn. The marker's CONTENTS are the pitch in degrees
-- (negative = down, e.g. "-80"); an empty or unparseable marker means 0 = level, which is the
-- control arm rather than a failure.
local markerPath = "Data Files/MWSE/mods/mgexe/lookpitch/ACTIVE"
local marker = io.open(markerPath, "r")
if not marker then return end
local body = marker:read("*a") or ""
marker:close()

local pitchDeg = tonumber(body:match("-?%d+%.?%d*") or "") or 0.0

mwse.log("[lookpitch] ARMED — marker present; view will be PINNED at pitch=%.1f deg", pitchDeg)

-- MW stores the player's look pitch on the mobile actor, not on the reference: writing
-- reference.orientation.x is silently overwritten by the camera every frame. Both are set and both
-- are READ BACK BELOW, because a fixture that silently no-ops produces a "no effect" result that is
-- indistinguishable from "the feature does nothing" — the exact confusion this probe exists to
-- settle. If the readback does not track the request, the arm is INVALID, not negative.
local function applyPitch()
    if tes3.menuMode() then return end
    local player = tes3.player
    local mob = tes3.mobilePlayer
    if not player or not mob then return end

    local rad = math.rad(pitchDeg)
    -- Preferred: the mobile actor's own look pitch.
    pcall(function() mob.viewAngle = rad end)
    -- Belt and braces: the reference orientation, which some MWSE builds route to the same field.
    local o = player.orientation
    pcall(function() player.orientation = tes3vector3.new(rad, o.y, o.z) end)
end

local ticks = 0
local function report()
    local mob = tes3.mobilePlayer
    local player = tes3.player
    if not mob or not player then return end
    local va = nil
    pcall(function() va = mob.viewAngle end)
    mwse.log("[lookpitch] want=%.1f deg | mob.viewAngle=%s | ref.orientation.x=%.1f deg | cell=%s",
             pitchDeg,
             va and string.format("%.1f deg", math.deg(va)) or "UNAVAILABLE",
             math.deg(player.orientation.x),
             tes3.getPlayerCell() and tes3.getPlayerCell().editorName or "?")
end

local function onLoaded()
    mwse.log("[lookpitch] loaded — pinning pitch every 0.1 s, reporting every 2 s")
    timer.start({ duration = 0.1, callback = applyPitch, type = timer.real, iterations = -1 })
    timer.start({ duration = 2.0, callback = report,     type = timer.real, iterations = -1 })
end

event.register("loaded", onLoaded)
