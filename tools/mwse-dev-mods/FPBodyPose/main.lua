-- FP Body Pose (MGE XE companion: tasks/forge-first-person.md FP2)
--
-- WHY: in true 1st person MW stops animating the player's 3rd-person body altogether. Its
-- AnimationData clocks stay at 0 and its bones never move (MGE_FP2_PROBE, 2026-10-19). MGE draws
-- that body into the player's SHADOW and REFLECTIONS in 1st person, but only while it is live; a
-- frozen body would be a statue. This mod gives it a pose: every frame in 1st person it plays the
-- ARMS' current animation on the body. The arms are only read, never written, so the 1st-person
-- view is exactly MW's.
--
-- HOW, per layer (lower body, upper body, left arm: AnimationData's three):
--   1. The group each 1st-person layer is playing (currentAnimGroups) and its clock (timings).
--   2. The same group in the body's AnimationData. The 1st- and 3rd-person .kf files put a group
--      at different times (idle ~206 s vs ~0-2 s), so the clock is mapped PIECEWISE through each
--      group's own action key times (start, loop start, loop stop, stop, ...).
--   3. The body's controllers are evaluated at the mapped time: the whole body at the lower
--      layer's, then the upper-body subtree at the upper's, then the left arm at its own.
--   4. Root motion: walking keys move the movement root ("Bip01") forward through the cycle. MW
--      takes that out as the actor's movement; here its horizontal position is put back, or the
--      body would slide ahead and snap back every stride.
--
-- CONFIG (Data Files/MWSE/config/FPBodyPose.json): enabled (default true), verbose (logs one
-- line every 2 s while posing).

local defaults = { enabled = true, verbose = false }
local config = mwse.loadConfig("FPBodyPose", defaults)

local kUpperBone = "Bip01 Spine1"
local kLeftArmBone = "Bip01 L Clavicle"

-- The time inside `to`'s group that matches `t` inside `from`'s: piecewise-linear through the
-- action key times both groups share. nil when either group is missing or has no keys.
local function mapTime(from, to, t)
    if not from or not to then return nil end
    local a, b = from.actionTimings, to.actionTimings
    local n = math.min(from.actionCount or 0, to.actionCount or 0)
    if n < 2 or not a or not b then return nil end
    if t <= a[1] then return b[1] end
    for k = 1, n - 1 do
        local a0, a1 = a[k], a[k + 1]
        if t <= a1 then
            local span = a1 - a0
            local f = span > 0 and (t - a0) / span or 0
            return b[k] + f * (b[k + 1] - b[k])
        end
    end
    return b[n]
end

-- The body's time for 1st-person layer `i` (1-based), or nil if the body has no such group.
local function layerTime(armsAnim, bodyAnim, i)
    local group = armsAnim.currentAnimGroups[i]
    if group == nil or group == 255 then return nil end
    local from = armsAnim.animationGroups[group + 1]
    local to = bodyAnim.animationGroups[group + 1]
    return mapTime(from, to, armsAnim.timings[i])
end

local lastLog = 0

local function onSimulated()
    if not config.enabled then return end
    local mobile = tes3.mobilePlayer
    if not mobile or mobile.is3rdPerson then return end   -- 3rd person: MW animates the body itself
    local fp = tes3.player1stPerson
    local body = tes3.player and tes3.player.sceneNode
    if not fp or not body then return end
    local armsAnim = fp.attachments and fp.attachments.animation
    local bodyAnim = tes3.player.attachments and tes3.player.attachments.animation
    if not armsAnim or not bodyAnim then return end

    local tLower = layerTime(armsAnim, bodyAnim, 1)
    if not tLower then return end
    local tUpper = layerTime(armsAnim, bodyAnim, 2) or tLower
    local tLeft = layerTime(armsAnim, bodyAnim, 3) or tUpper

    -- The body's clocks follow the mapped times. MGE's draw gate reads exactly this ("did the
    -- body's clock move?"), and a switch to 3rd person resumes from a matching phase.
    local timings = bodyAnim.timings
    timings[1], timings[2], timings[3] = tLower, tUpper, tLeft

    local root = bodyAnim.movementRootNode
    local rx, ry
    if root then rx, ry = root.translation.x, root.translation.y end

    body:update({ controllers = true, time = tLower })
    if tUpper ~= tLower then
        local upper = body:getObjectByName(kUpperBone)
        if upper then upper:update({ controllers = true, time = tUpper }) end
    end
    if tLeft ~= tUpper then
        local left = body:getObjectByName(kLeftArmBone)
        if left then left:update({ controllers = true, time = tLeft }) end
    end

    if root then
        root.translation = tes3vector3.new(rx, ry, root.translation.z)
        body:update()   -- re-propagate world transforms from the restored root (no controllers)
    end

    if config.verbose then
        local now = os.clock()
        if now - lastLog >= 2 then
            lastLog = now
            mwse.log("[FPBodyPose] groups %s,%s,%s arms t=%.3f -> body t=%.3f,%.3f,%.3f",
                tostring(armsAnim.currentAnimGroups[1]), tostring(armsAnim.currentAnimGroups[2]),
                tostring(armsAnim.currentAnimGroups[3]), armsAnim.timings[1], tLower, tUpper, tLeft)
        end
    end
end

event.register("initialized", function()
    event.register("simulated", onSimulated)
    mwse.log("[FPBodyPose] initialized (enabled=%s)", tostring(config.enabled))
end)
