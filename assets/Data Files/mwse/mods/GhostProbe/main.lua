-- Ghost Probe (dev aid for the "disabled reference is drawn by the Forge host" bug)
--
-- WHY: MW has no separate "hidden" flag — TES3::Reference::disable() app-culls the reference's
-- scene node and leaves it parented (MWSE/TES3Reference.cpp:474). MGE's interior post-load walk
-- runs with bypassCullDeep, which un-hides app-culled subtrees on purpose, so a DISABLED quest prop
-- (the airborne TR_m3_OE_MG_FlyingChair) can be captured, uploaded and drawn by the host forever
-- while MW itself draws nothing. Skipping disabled subtrees at capture fixes that — but ONLY if the
-- reference is already disabled when the walk runs. If the cell instead loads the prop ENABLED and
-- a local script disables it a frame or two later, the walk captures it legitimately and the entry
-- has to be EVICTED on the disable instead. Those are different fixes, so the timing has to be
-- measured, not guessed.
--
-- WHAT: on load, and then on every frame for `frames` frames, logs each watched reference's
-- disabled flag and its scene node's appCulled flag. The frame index is the answer:
--   frame 0 already disabled=true  -> the prop is never enabled at render time; capture-side skip
--                                     is the whole fix.
--   disabled=false for frames 0..k -> the script disables it AFTER capture; eviction-on-disable is
--                                     also required.
-- Also censuses every disabled reference in the cell, so we learn how many ghosts a normal TR
-- interior actually carries rather than reasoning from one chair.
--
-- CONFIG (Data Files/MWSE/config/GhostProbe.json):
--   enabled  false = never registers a handler. DEFAULT OFF, like AutoTurn360.
--   frames   how many frames after `loaded` to sample (the window MGE's post-load walk runs in).
--   watch    extra reference IDs to track by name, on top of the defaults below.
--
-- DISABLE TEST (`disableAfter` > 0): the second half of the bug. A reference disabled AFTER MGE
-- captured it is a different failure from one that was already disabled at load — reachability
-- cannot retire it (disable leaves the node parented), so anything in MGE's mover-candidate set
-- (NPCs, activators) keeps being re-emitted from the cache and ghosts until the cell is re-entered.
-- Plain statics do not ghost, because they are only ever drawn from the engine's visible set.
-- So the test has to disable a MOVER, not a crate: this picks a live NPC in the cell, disables it,
-- and leaves the game running long enough for an eviction sweep to render a verdict. The receipt is
-- in mgeXE.log, not here: `[evict] ... byDisabled=N` plus the cache `entries=` dropping by the
-- actor's part count. Console `disable` takes the same engine path, so this stands in for it.
--   disableAfter  real seconds after load before disabling (0 = off).
--   disableId     reference ID to disable; empty = pick the nearest enabled NPC in the cell.

local defaults = {
    enabled = false,
    frames = 12,
    watch = { "TR_m3_FlyingChair_01", "TR_m3_SittingChair_01" },
    disableAfter = 0,
    disableId = "",
}

local cfg = mwse.loadConfig("GhostProbe", defaults)

local function log(fmt, ...)
    mwse.log("[ghostprobe] " .. fmt, ...)
end

local frame = 0
local onSimulate   -- forward local: the handler unregisters itself by identity

-- Both flags matter and they are NOT the same question. `disabled` is the TES3 reference state MGE
-- now reads; `appCulled` is what the scene walk actually sees. If they ever disagree, the capture
-- rule is reading the wrong one.
local function describe(ref)
    if not ref then return "(no such reference)" end
    local node = ref.sceneNode
    return string.format("disabled=%s appCulled=%s pos=(%.0f,%.0f,%.0f) base=%s",
        tostring(ref.disabled),
        node and tostring(node.appCulled) or "(no scene node)",
        ref.position.x, ref.position.y, ref.position.z,
        ref.baseObject and ref.baseObject.id or "?")
end

local function sample(tag)
    for _, id in ipairs(cfg.watch) do
        log("%s %-24s %s", tag, id, describe(tes3.getReference(id)))
    end
end

-- One-shot census: every disabled reference in the cell that still has a scene node is a candidate
-- ghost, because a scene node is exactly what MGE's deep walk can reach.
local function census()
    local cell = tes3.getPlayerCell()
    if not cell then log("census: no player cell"); return end
    local total, disabled, withNode = 0, 0, 0
    local shown = 0
    for ref in cell:iterateReferences() do
        total = total + 1
        if ref.disabled then
            disabled = disabled + 1
            if ref.sceneNode then
                withNode = withNode + 1
                if shown < 16 then
                    shown = shown + 1
                    log("census  DISABLED %-28s %s", ref.id, describe(ref))
                end
            end
        end
    end
    log("census '%s': %d refs, %d disabled, %d of those still have a scene node (= reachable by the deep walk)",
        cell.editorName or cell.id, total, disabled, withNode)
end

onSimulate = function()
    sample(string.format("frame %-2d", frame))
    frame = frame + 1
    if frame >= cfg.frames then
        event.unregister("simulate", onSimulate)
        log("done sampling after %d frames", frame)
        census()
    end
end

-- Pick the disable victim: a MOVER is the whole point (see the header), so prefer an NPC. The
-- player is excluded for the obvious reason.
local function findVictim()
    if cfg.disableId ~= "" then return tes3.getReference(cfg.disableId) end
    local cell = tes3.getPlayerCell()
    if not cell then return nil end
    local player = tes3.player
    for ref in cell:iterateReferences(tes3.objectType.npc) do
        if ref ~= player and not ref.disabled and ref.sceneNode then return ref end
    end
    return nil
end

local function runDisableTest()
    local ref = findVictim()
    if not ref then
        log("DISABLE TEST: no enabled NPC found in the cell - nothing to disable")
        return
    end
    log("DISABLE TEST: disabling '%s' %s", ref.id, describe(ref))
    ref:disable()
    log("DISABLE TEST: disabled '%s' -> %s", ref.id, describe(ref))
    log("DISABLE TEST: watch mgeXE.log for [evict] byDisabled= and a matching entries= drop")
    -- Re-read a few seconds later: if the reference somehow re-enabled itself (an AI package, a
    -- script), the mgeXE.log verdict would be about a different world state than we think.
    timer.start({
        type = timer.real, duration = 6.0, iterations = 1,
        callback = function()
            log("DISABLE TEST: 6s later '%s' %s", ref.id, describe(ref))
        end,
    })
end

local function onLoaded()
    if not cfg.enabled then
        log("disabled (enabled=false)")
        return
    end
    local cell = tes3.getPlayerCell()
    log("loaded into '%s' - sampling %d frames",
        cell and (cell.editorName or cell.id) or "(unknown cell)", cfg.frames)
    -- Sample BEFORE any simulate tick: this is the closest we can stand to the state MGE's first
    -- post-load walk frame sees, and it is the frame the whole question turns on.
    sample("onLoaded ")
    frame = 0
    event.register("simulate", onSimulate)

    if cfg.disableAfter > 0 then
        timer.start({
            type = timer.real, duration = cfg.disableAfter, iterations = 1,
            callback = runDisableTest,
        })
    end
end

event.register("loaded", onLoaded)
