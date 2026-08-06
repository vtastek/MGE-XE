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

-- DEATH WATCH (`deathWatch`): the THIRD way a reference leaves the world, after "already disabled at
-- load" and "disabled after capture". Some creatures do not leave a corpse — a dwarven specter dies
-- and the body is replaced by an ectoplasm pile. Reported 2026-08-01: the host goes on drawing the
-- body's last pose PERMANENTLY (cell re-entry is the only thing that clears it, i.e. purgeAll), while
-- the ectoplasm captures fine. Permanent is the tell — a creature is a Mover, so it is re-emitted
-- from the cache every frame independent of the engine's visible set, and only an eviction verdict
-- can ever retire it. So the sweep is looking at the body and concluding it is still alive.
--
-- The question this answers is exactly the one the disable bug turned on, one bit over: when the
-- engine removes that body, what does it leave behind? MGE reads the Disabled bit (0x800). There is
-- also a Delete bit (0x20, MWSE/TES3Object.h:100) which MGE tests NOWHERE, and `deleted`/`disabled`
-- are independent flags. Three outcomes, three different fixes:
--   deleted=true, node still parented   -> a verdict hole; the sweep needs to ask about Delete too.
--   node detached / no scene node       -> NOT a hole; the sweep should already say gone, so the bug
--                                          is upstream (the entry is being kept by something else).
--   neither, body simply still there    -> the engine keeps it and MW hides it some other way; the
--                                          whole premise is wrong and this is not an eviction bug.
-- The climb below deliberately mirrors the sweep's own parent climb, so its verdict is directly
-- comparable to `[evict] byDetach=`/`byDisabled=` in mgeXE.log rather than merely suggestive.

local defaults = {
    enabled = false,
    frames = 12,
    watch = { "TR_m3_FlyingChair_01", "TR_m3_SittingChair_01" },
    disableAfter = 0,
    disableId = "",
    deathWatch = false,
    deathFrames = 240,
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

-- Mirror of the eviction sweep's parentVerdict climb (scenegraph_geometry_cache.cpp): walk parentNode
-- up to the depth cap and report where it ended. MGE evicts on "chain ended without reaching a known
-- root"; it keeps on "reached g_objRoot". We cannot name MGE's roots from Lua, so report the SHAPE of
-- the chain — depth reached and whether it terminated in nil — which is the same discriminator.
local kMaxParentDepth = 32
local function climb(node)
    if not node then return "(no node)" end
    local p = node.parent
    if not p then return "detached(depth0)" end
    for depth = 1, kMaxParentDepth do
        local next = p.parent
        if not next then
            -- Terminated. A chain ending at the world root is normal and means REACHABLE; MGE
            -- distinguishes them by root identity, which is why the top node's name is printed.
            return string.format("ends depth=%d top='%s' culled=%s",
                depth, p.name or "(unnamed)", tostring(p.appCulled))
        end
        p = next
    end
    return "depth-capped"
end

-- Latched at death: the REFERENCE only, never its scene node. MWSE hands out NI objects as raw
-- pointers without taking a reference, so a node held across frames is a use-after-free the moment the
-- engine releases the body — which is precisely the event being measured, so the crash would be
-- reliable rather than rare. Re-read ref.sceneNode every sample instead: "it went nil" is one of the
-- three answers anyway, so nothing is lost. (MW keeps deleted references in the cell list until the
-- cell unloads, so the reference itself stays addressable.)
local deathWatch = nil

local function deathSample(tag)
    local w = deathWatch
    if not w then return end
    local ref = w.ref
    local node = ref and ref.sceneNode or nil
    log("%s %-24s deleted=%-5s disabled=%-5s sceneNode=%-4s parent=%-5s appCulled=%-5s climb=%s",
        tag, w.id,
        ref and tostring(ref.deleted) or "?",
        ref and tostring(ref.disabled) or "?",
        node and "yes" or "nil",
        node and tostring(node.parent ~= nil) or "-",
        node and tostring(node.appCulled) or "-",
        node and climb(node) or "(no node)")
end

local function onDeathSimulate()
    local w = deathWatch
    if not w then return end
    -- Every frame while it matters, then thin out: the interesting transition is within a second or
    -- two of death, but the whole point is that this state PERSISTS, so keep sampling long enough to
    -- outlast several 30-frame eviction sweeps and prove it never changes.
    w.frame = w.frame + 1
    if w.frame <= 30 or (w.frame % 30) == 0 then
        deathSample(string.format("death+%-3d", w.frame))
    end
    if w.frame >= cfg.deathFrames then
        event.unregister("simulate", onDeathSimulate)
        deathSample("death-END")
        log("death watch done after %d frames - compare with [evict] in mgeXE.log", w.frame)
        deathWatch = nil
    end
end

local function onDeath(e)
    if not cfg.deathWatch or deathWatch then return end   -- first death only; one clean trace
    local ref = e.reference
    if not ref or not ref.sceneNode then return end
    deathWatch = { ref = ref, id = ref.id, frame = 0 }
    log("DEATH '%s' base=%s - watching %d frames",
        ref.id, ref.baseObject and ref.baseObject.id or "?", cfg.deathFrames)
    deathSample("death+0  ")
    event.register("simulate", onDeathSimulate)
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
-- Registered unconditionally; onDeath itself gates on cfg.deathWatch, so toggling the config does not
-- depend on having reloaded the save since.
event.register("death", onDeath)
