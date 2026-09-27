-- Gobo Inspect (dev aid for tasks/forge-light-gobo.md)
--
-- WHY: every baked light fixture model has a gobo (hdrdump/gobo_sheet.tga, written by the host with
-- MGE_HOST_KNOBS=goboDump=1). A gobo is only right if the fixture's own mesh is — a one-sided bottom
-- plate, a missing cap, a light that sits inside a solid mesh all show up as a wrong pattern, and
-- the same asset bug shows in the LIVE near shadow. This walks the fixtures one at a time so each
-- can be looked at live and compared against its sheet cell.
--
-- WHAT: press ] to spawn the NEXT fixture light in front of you (Shift+] = previous). The previous
-- spawn is deleted. The message names the list position, the gobo layer, its sheet cell (col,row),
-- the baked openness (0 = sealed, 1 = casts nothing), the LIGH id, the mesh and the plugin it came
-- from (vanilla / not), since non-vanilla meshes are the suspects.
--
-- ORDER: the gobo sheet's layer order (hdrdump/gobo_sheet.txt), so ] steps through the sheet cells
-- left to right, top to bottom. One LIGH id per model (a placeable, on-by-default one if there is).
-- Without the txt it falls back to every meshed light, grouped by model, sorted by mesh path.
--
-- Stand near a wall and a floor: the light spawns at about head height, so the cage/bowl shadow
-- lands on both.

local kKey      = tes3.scanCode.closeSquareBracket
local kDistance = 128.0   -- units in front of the player
local kHeight   = 110.0   -- above the player's feet
local kSheet    = "hdrdump\\gobo_sheet.txt"

local vanilla = { ["morrowind.esm"] = true, ["tribunal.esm"] = true, ["bloodmoon.esm"] = true }

local entries = nil   -- { light=, layer=, col=, row=, open=, mesh= }
local cursor  = 0
local spawned = nil

local function readSheet()
    local f = io.open(kSheet, "r")
    if not f then return nil end
    local byMesh, order = {}, {}
    for line in f:lines() do
        local col, row, layer, open, mesh =
            line:match("cell%s+(%d+),%s*(%d+)%s+layer%s+(%d+)%s+open%s+([%d%.]+)%s+(.+)$")
        if mesh then
            mesh = mesh:lower():gsub("%s+$", "")
            local e = { layer = tonumber(layer), col = tonumber(col), row = tonumber(row),
                        open = tonumber(open), mesh = mesh }
            byMesh[mesh] = e
            table.insert(order, e)
        end
    end
    f:close()
    return byMesh, order
end

local function better(a, b)
    -- Prefer a fixed, on-by-default light: that is what the world places and what the baker baked.
    local function score(l) return (l.isOffByDefault and 2 or 0) + (l.canCarry and 1 or 0) end
    return score(a) < score(b)
end

local function build()
    local lightByMesh = {}
    for obj in tes3.iterateObjects(tes3.objectType.light) do
        local mesh = obj.mesh
        if mesh and mesh ~= "" then
            mesh = mesh:lower()
            local cur = lightByMesh[mesh]
            if not cur or better(obj, cur) then lightByMesh[mesh] = obj end
        end
    end

    entries = {}
    local _, order = readSheet()
    if order then
        for _, e in ipairs(order) do
            local l = lightByMesh[e.mesh]
            if l then
                e.light = l
                table.insert(entries, e)
            else
                mwse.log("[GoboInspect] no LIGH object for sheet model %s (layer %d)", e.mesh, e.layer)
            end
        end
        mwse.log("[GoboInspect] %d fixtures from %s", #entries, kSheet)
    else
        for mesh, l in pairs(lightByMesh) do
            table.insert(entries, { light = l, mesh = mesh })
        end
        table.sort(entries, function(a, b) return a.mesh < b.mesh end)
        mwse.log("[GoboInspect] %s missing (run the host with goboDump=1) - %d meshed lights, no layer info",
                 kSheet, #entries)
    end
end

local function show(e)
    local l = e.light
    local src = l.sourceMod or "?"
    local origin = vanilla[src:lower()] and "vanilla" or "NON-VANILLA"
    local where = e.layer and string.format("layer %d  cell %d,%d  open %.2f", e.layer, e.col, e.row, e.open)
                           or "no gobo info"
    local msg = string.format("[%d/%d] %s\n%s\n%s\n%s (%s)", cursor, #entries, where, l.id, e.mesh, src, origin)
    tes3.messageBox(msg)
    mwse.log("[GoboInspect] %s", msg:gsub("\n", " | "))
end

local function spawn(step)
    if not entries then build() end
    if #entries == 0 then
        tes3.messageBox("[GoboInspect] no fixture lights found")
        return
    end
    cursor = ((cursor - 1 + step) % #entries) + 1
    if spawned then
        spawned:delete()
        spawned = nil
    end
    local e = entries[cursor]
    local p = tes3.player
    local yaw = p.orientation.z
    local pos = p.position + tes3vector3.new(math.sin(yaw) * kDistance, math.cos(yaw) * kDistance, kHeight)
    spawned = tes3.createReference({
        object      = e.light,
        position    = pos,
        orientation = tes3vector3.new(0, 0, yaw + math.pi),   -- face the player
        cell        = p.cell,
    })
    show(e)
end

local function onKeyDown(ev)
    if tes3.menuMode() then return end
    spawn(ev.isShiftDown and -1 or 1)
end

-- AUTO mode (unattended harness runs): armed ONLY by an `AUTO` file beside this main.lua. Each
-- line is one STEP: comma-separated object ids (or mesh-path substrings), each optionally `id*N`
-- for N copies, all spawned side by side in front of the player for kAutoHold seconds, then
-- deleted. A `hold <seconds>` line changes the hold (e.g. to line up a host dumpAtFrame), and a
-- `goto <cell>;x,y,z;yawDegrees` step teleports the player instead of spawning, a `key <scancode>`
-- step taps a key, `seam` toggles the composite (F11), `shot <file>` saves MW's backbuffer,
-- `relight` unequips and re-equips the player's light, and a
-- trailing `@<seconds>` gives any step its own duration. Key/seam/shot steps keep spawns. Logs
-- `[GoboInspect] AUTO done` at the end. Disarm by deleting the file.
local kAutoFile = "Data Files\\MWSE\\mods\\GoboInspect\\AUTO"
local kAutoHold = 5.0
local kAutoWait = 10.0
local kAutoGap  = 48.0   -- units between neighbours in a step

local function readAuto()
    local f = io.open(kAutoFile, "r")
    if not f then return nil end
    local list = {}
    for line in f:lines() do
        local dur = line:match("@%s*([%d%.]+)%s*$")      -- per-step duration override
        line = line:gsub("@%s*[%d%.]+%s*$", "")
        local key = line:match("^%s*key%s+(%w+)")
        local shot = line:match("^%s*shot%s+(%S+)")
        local seam = line:match("^%s*seam%s*$")
        local relight = line:match("^%s*relight%s*$")
        local hold = line:match("^%s*hold%s+([%d%.]+)")
        local gCell, gx, gy, gz, gyaw = line:match("^%s*goto%s+([^;]+);%s*([-%d%.]+),%s*([-%d%.]+),%s*([-%d%.]+);%s*([-%d%.]+)")
        if hold then
            kAutoHold = tonumber(hold)
        elseif shot then
            -- `shot <file.png>` saves MW's backbuffer (no UI) via mge.saveScreenshot
            table.insert(list, { shot = shot, dur = tonumber(dur) })
        elseif relight then
            -- `relight` unequips the player's equipped light and re-equips it 1 s later
            table.insert(list, { relight = true, dur = tonumber(dur) })
        elseif seam then
            -- `seam` toggles the Forge composite like F11 (the client consumes mge_seam_toggle)
            table.insert(list, { seam = true, dur = tonumber(dur) })
        elseif key then
            -- `key <scancode>` taps a key (e.g. 0x57 = F11 seam toggle, 0xB7 = PrintScreen)
            table.insert(list, { key = tonumber(key), dur = tonumber(dur) })
        elseif gCell then
            -- `goto <cell>;x,y,z;yawDegrees` — a step that moves the player instead of spawning
            table.insert(list, { move = { cell = gCell, pos = { tonumber(gx), tonumber(gy), tonumber(gz) },
                                          yaw = math.rad(tonumber(gyaw)) }, dur = tonumber(dur) })
        else
            local step = {}
            for tok in line:gmatch("[^,]+") do
                local id, n = tok:match("^%s*([^%s%*]+)%s*%*?%s*(%d*)")
                if id then table.insert(step, { id = id, n = tonumber(n) or 1 }) end
            end
            step.dur = tonumber(dur)
            if #step > 0 then table.insert(list, step) end
        end
    end
    f:close()
    return list
end

local function findObject(id)
    local obj = tes3.getObject(id)
    if obj then return obj end
    local want = id:lower()   -- not an id: the first light whose MESH path contains it
    for l in tes3.iterateObjects(tes3.objectType.light) do
        if l.mesh and l.mesh:lower():find(want, 1, true) then return l end
    end
    return nil
end

local autoSpawned = {}
local function autoClear()
    for _, r in ipairs(autoSpawned) do r:delete() end
    autoSpawned = {}
end

local function autoRun(list)
    mwse.log("[GoboInspect] AUTO ARMED: %d steps, hold %.0fs", #list, kAutoHold)
    local i = 0
    local step
    step = function()
        i = i + 1
        local items = list[i]
        if not items then
            autoClear()
            mwse.log("[GoboInspect] AUTO done")
            return
        end
        local nextIn = items.dur or kAutoHold
        timer.start({ type = timer.real, duration = nextIn, callback = step })
        if items.shot then
            local ok, err = pcall(mge.saveScreenshot, { path = items.shot, captureWithUI = false })
            mwse.log("[GoboInspect] AUTO step %d: shot %s %s", i, items.shot, ok and "ok" or tostring(err))
            return
        end
        if items.relight then
            local st = tes3.getEquippedItem({ actor = tes3.player, objectType = tes3.objectType.light })
            if not st then
                mwse.log("[GoboInspect] AUTO step %d: relight: no light equipped", i)
                return
            end
            local obj = st.object
            tes3.mobilePlayer:unequip({ item = obj })
            mwse.log("[GoboInspect] AUTO step %d: relight: unequipped %s", i, obj.id)
            timer.start({ type = timer.real, duration = 1.0, callback = function()
                tes3.mobilePlayer:equip({ item = obj })
                mwse.log("[GoboInspect] AUTO relight: re-equipped %s", obj.id)
            end })
            return
        end
        if items.seam then
            local f = io.open("mge_seam_toggle", "w")
            if f then f:write("1") f:close() end
            mwse.log("[GoboInspect] AUTO step %d: seam toggle requested", i)
            return
        end
        if items.key then
            tes3.tapKey(items.key)
            mwse.log("[GoboInspect] AUTO step %d: key 0x%X", i, items.key)
            return   -- a key step keeps the previous step's spawns (screenshot / A/B of them)
        end
        autoClear()
        if items.move then
            local g = items.move
            tes3.positionCell({ reference = tes3.player, cell = g.cell,
                                position = g.pos, orientation = { 0, 0, g.yaw } })
            -- Hold the spot: a teleport into a stairwell would otherwise fall or be pushed out.
            tes3.mobilePlayer.movementCollision = false
            tes3.mobilePlayer.isFlying = true
            mwse.log("[GoboInspect] AUTO step %d: goto %s (%.0f,%.0f,%.0f) yaw %.0f", i, g.cell,
                     g.pos[1], g.pos[2], g.pos[3], math.deg(g.yaw))
            return
        end
        local spawnList = {}
        for _, it in ipairs(items) do
            local obj = findObject(it.id)
            if obj then
                for _ = 1, it.n do table.insert(spawnList, obj) end
                mwse.log("[GoboInspect] AUTO step %d: %s x%d (%s, %s)", i, it.id, it.n,
                         tostring(obj.mesh), tostring(obj.sourceMod))
            else
                mwse.log("[GoboInspect] AUTO step %d: %s: no such object", i, it.id)
            end
        end
        local p = tes3.player
        local yaw = p.orientation.z
        local fwd = tes3vector3.new(math.sin(yaw), math.cos(yaw), 0)
        local side = tes3vector3.new(math.cos(yaw), -math.sin(yaw), 0)
        for k, obj in ipairs(spawnList) do
            local off = (k - (#spawnList + 1) / 2) * kAutoGap
            local pos = p.position + fwd * kDistance + side * off + tes3vector3.new(0, 0, kHeight)
            table.insert(autoSpawned, tes3.createReference({
                object = obj, position = pos,
                orientation = tes3vector3.new(0, 0, yaw + math.pi), cell = p.cell,
            }))
        end
    end
    timer.start({ type = timer.real, duration = kAutoWait, callback = function()
        step()   -- each step schedules the next after its own duration
    end })
end

local function onLoaded()
    -- A reload drops the old spawn with the save; forget it rather than delete a stale handle.
    spawned = nil
    autoSpawned = {}
    if not entries then build() end   -- logs the fixture count now, not on the first key press
    local list = readAuto()
    if list then autoRun(list) end
end

event.register(tes3.event.initialized, function()
    event.register(tes3.event.keyDown, onKeyDown, { filter = kKey })
    event.register(tes3.event.loaded, onLoaded)
    mwse.log("[GoboInspect] ready: ] = next fixture light, Shift+] = previous")
end)
