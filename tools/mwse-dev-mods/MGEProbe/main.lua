-- MGE Probe (dev component: ask the ENGINE, not our own bookkeeping)
--
-- WHY: the client and host only know what reached them — what was drawn, captured, resolved. When a
-- number like "1500 textures in the active grid" drives a design decision, it has to be confirmed
-- from Morrowind's own scene graph, or we are measuring our instrument instead of the game. Each
-- probe here reads engine state through MWSE and logs one `[probe]` block to MWSE.log, so a
-- minimized harness run is verifiable from logs alone.
--
-- PROBES
--   texcensus  walks worldObjectRoot / worldPickRoot / worldLandscapeRoot and every niAVObject's
--              texturingProperty maps. Reports unique texture FILES (what the host needs a bindless
--              slot for; names normalised as MW's loader does) per root and in total, sizes and
--              full-chain bytes from the loose files' DDS headers, loose/BSA split, and the
--              active-cell / reference counts that give the numbers scale. Nothing is filtered by
--              visibility: MW loads a cell's textures when the cell loads, so this is the set a
--              stationary turn can ever reveal. First run (ibodragonstareast360, 2026-09-30):
--              1084 files over 9 cells, 1087 after a 720 deg turn; every niPixelData nil (MW frees
--              the CPU copy once D3D has it), hence the header reads.
--
-- CONFIG (Data Files/MWSE/config/MGEProbe.json):
--   enabled  false = registers nothing. DEFAULT OFF: dev component, never ships active.
--   at       real seconds after `loaded` at which to run the probes; a list, so the same census can
--            be taken before and after an AutoTurn360 sweep to show the set does not change.
--   probes   which probes to run, e.g. { "texcensus" }.
--   top      how many of the largest textures to list by name.

local defaults = {
    enabled = false,
    at = { 5.0 },
    probes = { "texcensus" },
    top = 12,
}

local cfg = mwse.loadConfig("MGEProbe", defaults)

local function log(fmt, ...)
    mwse.log("[probe] " .. fmt, ...)
end

-- ---- texcensus -------------------------------------------------------------------------------

local function sizeBucket(px)
    if px <= 0 then return "unknown" end
    if px <= 128 then return "<=128" end
    if px <= 256 then return "256" end
    if px <= 512 then return "512" end
    if px <= 1024 then return "1024" end
    if px <= 2048 then return "2048" end
    return "4096+"
end
local bucketOrder = { "<=128", "256", "512", "1024", "2048", "4096+", "unknown" }

-- MW keys textures the way its loader does: relative to Data Files\textures, and a .tga/.bmp request
-- is served by a .dds of the same stem when one exists. Normalise to that, or one file counts twice
-- ("textures\x.dds" and "x.dds") and the census overstates the set.
local function normalise(fileName)
    local n = string.lower(fileName):gsub("/", "\\")
    n = n:gsub("^data files\\", ""):gsub("^textures\\", "")
    return n
end

local function u32(s, at)
    local a, b, c, d = string.byte(s, at + 1, at + 4)
    return a + b * 256 + c * 65536 + d * 16777216
end

-- Bytes per texel for the formats Morrowind content ships in; the full-mip-chain factor is 4/3.
local fourCCBpt = { DXT1 = 0.5, DXT3 = 1, DXT5 = 1, ATI1 = 0.5, BC4U = 0.5, ATI2 = 1, BC5U = 1 }

-- Size from the DDS header. niPixelData cannot answer this: MW frees a static texture's CPU pixels
-- once D3D holds them (census: every pixelData nil). Loose files only; BSA contents are not readable
-- from Lua, so those are counted by source and left unsized.
local function ddsInfo(rel)
    local stem = rel:gsub("%.[^.\\]+$", "")
    for _, cand in ipairs({ stem .. ".dds", rel }) do
        local src, resolved = tes3.getFileSource("textures\\" .. cand)
        if src == "file" then
            local f = io.open(resolved, "rb") or io.open("Data Files\\textures\\" .. cand, "rb")
            if f then
                local hdr = f:read(148) or ""
                f:close()
                if #hdr >= 128 and hdr:sub(1, 4) == "DDS " then
                    local h, w = u32(hdr, 12), u32(hdr, 16)
                    local mips = math.max(u32(hdr, 28), 1)
                    local fourCC = hdr:sub(85, 88)
                    local bpt = fourCCBpt[fourCC]
                    if fourCC == "DX10" then bpt = 1 end   -- BC7/BC6H; others are rare in MW content
                    if not bpt then bpt = u32(hdr, 88) / 8 end   -- uncompressed: RGBBitCount
                    local bytes = w * h * bpt * (mips > 1 and 4 / 3 or 1)
                    return "loose", w, h, bytes, fourCC
                end
            end
            return "loose", 0, 0, 0, "?"
        elseif src == "bsa" then
            return "bsa", 0, 0, 0, "?"
        end
    end
    return "missing", 0, 0, 0, "?"
end

-- One root: unique (normalised) texture files reached from it, with the map slot that first named it.
local function walkRoot(root, files, stats)
    if not root then return end
    local stack = { root }
    while #stack > 0 do
        local obj = table.remove(stack)
        stats.nodes = stats.nodes + 1
        local tp = obj.texturingProperty
        if tp then
            local maps = tp.maps
            for i = 1, #maps do
                local map = maps[i]
                local tex = map and map.texture
                if tex and tex.fileName then
                    local name = normalise(tex.fileName)
                    if not files[name] then files[name] = { slot = i } end
                end
            end
        end
        if obj:isInstanceOfType(ni.type.NiNode) then
            local children = obj.children
            for i = 1, #children do
                local c = children[i]
                if c then stack[#stack + 1] = c end
            end
        end
    end
end

local function count(t)
    local n = 0
    for _ in pairs(t) do n = n + 1 end
    return n
end

local function texcensus(label)
    local t0 = os.clock()
    local game = tes3.game
    local roots = {
        { "object",    game.worldObjectRoot },
        { "pick",      game.worldPickRoot },
        { "landscape", game.worldLandscapeRoot },
    }
    local all = {}
    local nodesTotal = 0
    for _, r in ipairs(roots) do
        local files, stats = {}, { nodes = 0 }
        walkRoot(r[2], files, stats)
        nodesTotal = nodesTotal + stats.nodes
        for k, v in pairs(files) do all[k] = v end
        log("texcensus %s | root %-9s nodes=%d uniqueFiles=%d", label, r[1], stats.nodes, count(files))
    end

    -- Size from each file's DDS header (see ddsInfo). bytes = full mip chain, i.e. what the host
    -- holds for it at full resolution.
    local hist, byMap, bySource = {}, {}, { loose = 0, bsa = 0, missing = 0 }
    for _, b in ipairs(bucketOrder) do hist[b] = 0 end
    local sized, texels, bytes, sizedN = {}, 0, 0, 0
    for name, v in pairs(all) do
        local src, w, h, b, fmt = ddsInfo(name)
        bySource[src] = bySource[src] + 1
        local bk = sizeBucket(math.max(w, h))
        hist[bk] = hist[bk] + 1
        byMap[v.slot] = (byMap[v.slot] or 0) + 1
        if w > 0 then
            sizedN = sizedN + 1
            texels = texels + w * h
            bytes = bytes + b
        end
        sized[#sized + 1] = { name = name, px = w * h, w = w, h = h, bytes = b, fmt = fmt }
    end
    local parts = {}
    for _, b in ipairs(bucketOrder) do parts[#parts + 1] = string.format("%s:%d", b, hist[b]) end
    local mapParts = {}
    for slot, n in pairs(byMap) do mapParts[#mapParts + 1] = string.format("map%d:%d", slot, n) end
    table.sort(mapParts)

    local cells = tes3.getActiveCells() or {}
    local refs = 0
    for _, cell in ipairs(cells) do
        for _ in cell:iterateReferences() do refs = refs + 1 end
    end

    log("texcensus %s | TOTAL uniqueFiles=%d (loose %d, bsa %d, missing %d) | cells=%d refs=%d nodes=%d | %.1f ms",
        label, count(all), bySource.loose, bySource.bsa, bySource.missing, #cells, refs, nodesTotal,
        (os.clock() - t0) * 1000)
    log("texcensus %s | sized %d loose files: %.1f Mtexels top mip, %.0f MB full chains (avg %.2f MB)",
        label, sizedN, texels / 1e6, bytes / 1048576, sizedN > 0 and bytes / 1048576 / sizedN or 0)
    log("texcensus %s | size (max edge): %s", label, table.concat(parts, " "))
    log("texcensus %s | first seen in map slot (1=base 2=dark 3=detail 4=gloss 5=glow 6=bump 7=decal): %s",
        label, table.concat(mapParts, " "))
    table.sort(sized, function(a, b) return a.bytes > b.bytes end)
    for i = 1, math.min(cfg.top, #sized) do
        local s = sized[i]
        log("texcensus %s |   %4dx%-4d %-4s %5.1f MB %s", label, s.w, s.h, s.fmt, s.bytes / 1048576, s.name)
    end
end

-- ---- driver ----------------------------------------------------------------------------------

local probes = { texcensus = texcensus }

local function runAll(label)
    for _, p in ipairs(cfg.probes) do
        local fn = probes[p]
        if fn then
            local ok, err = pcall(fn, label)
            if not ok then log("!! probe %s FAILED: %s", p, tostring(err)) end
        else
            log("!! unknown probe '%s'", tostring(p))
        end
    end
end

local function onLoaded()
    if not cfg.enabled then return end
    log("armed: probes=%s at=%s", table.concat(cfg.probes, ","), table.concat(cfg.at, ","))
    for _, secs in ipairs(cfg.at) do
        timer.start({
            type = timer.real,
            duration = math.max(secs, 0.001),
            iterations = 1,
            callback = function() runAll(string.format("t=%.0fs", secs)) end,
        })
    end
end

event.register("loaded", onLoaded)
