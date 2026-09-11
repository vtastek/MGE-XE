-- MGE XE — MB-2g THIRD-PERSON TRAVERSAL DRIVER (test fixture, NOT a gameplay mod).
--
-- WHY THIS EXISTS. Object-only motion blur writes motion relative to the WORLD, and the one
-- population where that is not what a viewer sees is geometry the CAMERA MOVES WITH. The arms were
-- the first (MB-2e); the THIRD-PERSON PLAYER is the second, and it is the one the harness could
-- never reach: the fixed test save loads in first person and never walks, so the body that proves
-- or disproves the rule is not on screen and is not translating.
--
-- The acceptance test is a contrast, not a number, and it needs BOTH phases in one run:
--   TRAVERSE — the PC runs in a straight line with the camera tracking it. Its world motion is
--              large and uniform across every bone; its SCREEN motion is ~0. Correct = the body
--              stays sharp and only the limbs smear. The defect = "blurred everywhere, slow parts
--              or fast parts alike", which is what was reported from play.
--   HOLD     — the PC stands still and animates in place. World motion == screen motion, so both
--              rules agree and the picture must be UNCHANGED between them. This is the control:
--              a fix that also changes HOLD changed something it had no business touching.
--
-- ⚠⚠ INERT UNLESS THE MARKER FILE EXISTS, for the reason velchurn/main.lua gives at length: this
-- moves the player, and doing that during real play is indistinguishable from possession. No
-- marker, no event registration, no timer, no POV change.
local markerPath = "Data Files/MWSE/mods/mgexe/pcdrive/ACTIVE"
local marker = io.open(markerPath, "r")
if not marker then return end
marker:close()

mwse.log("[pcdrive] ARMED — marker present; this session WILL be driven (3rd person + traversal)")

-- ⚠ PER FRAME, NOT ON A TIMER, and that is the whole difference between a run and a teleport.
-- velchurn records the mistake in its own comment: a 128 u hop in one frame is ~160 m/s and
-- manufactures exactly the cell-scale motion the instruments exist to distinguish from animation.
-- Scaling by the frame delta keeps the PER-FRAME displacement at whatever 250 u/s really is on this
-- machine, which is a brisk run and is the quantity the blur integrates.
local RUN_SPEED = 250.0       -- units/sec, ~MW run
local PHASE_SECS = 4.0        -- forward / hold / back / hold

local t = 0.0
local phase = -1

local function phaseName(p)
    if p == 0 then return "TRAVERSE forward" end
    if p == 1 then return "HOLD (animate in place)" end
    if p == 2 then return "TRAVERSE backward" end
    return "HOLD (animate in place)"
end

local function onSimulate(e)
    if tes3.menuMode() then return end
    local player = tes3.player
    if not player or not tes3.mobilePlayer then return end

    -- Held every frame rather than set once: anything in the game may flip the POV back, and a
    -- fixture whose premise silently lapses reports a pass for a test that never ran.
    if not tes3.is3rdPerson() then tes3.force3rdPerson() end

    t = t + e.delta
    local p = math.floor(t / PHASE_SECS) % 4
    if p ~= phase then
        phase = p
        mwse.log("[pcdrive] t=%.1f phase -> %s", t, phaseName(p))
    end

    if p == 0 or p == 2 then
        -- ⚠ A DIRECT POSITION WRITE, SO THERE IS NO COLLISION. Deliberate and bounded: the two
        -- traverse phases are equal and opposite, so the net displacement over a cycle is ~0 and the
        -- fixture cannot walk the player off across the map while it runs unattended. It can still
        -- pass through a wall — which for this test is harmless, because what is under test is the
        -- TRANSLATION, not where it ends up.
        local f = tes3.mobilePlayer.facing
        local dir = (p == 0) and 1.0 or -1.0
        local d = RUN_SPEED * e.delta * dir
        player.position = player.position + tes3vector3.new(math.sin(f) * d, math.cos(f) * d, 0)
    end
end

local function onLoaded()
    mwse.log("[pcdrive] loaded — 3rd person forced; %.0f s phases at %.0f u/s", PHASE_SECS, RUN_SPEED)
    t = 0.0
    phase = -1
    event.register("simulate", onSimulate)
end

event.register("loaded", onLoaded)
