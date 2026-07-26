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

local defaults = {
    enabled = false,
    frames = 12,
    watch = { "TR_m3_FlyingChair_01", "TR_m3_SittingChair_01" },
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
end

event.register("loaded", onLoaded)
