-- RotateRef: turn one reference about its vertical axis after load, for image A/Bs that need to know
-- whether a dark patch is attached to the object (a shadow its own geometry casts moves with it) or
-- to the world (a gobo, a mask texel, a screen-space term stays put). Inert unless
-- MGE_ROTATE_REF=<object id>:<degrees> is in the game's environment (forge-perf-run.sh clientEnv).
-- Picks the reference with that id nearest the player in the player's cell.

local spec = os.getenv("MGE_ROTATE_REF")
if not spec then
    return
end
local id, deg = spec:match("^(.-):([%-%d%.]+)$")
if not id then
    mwse.log("[rotateref] !! bad MGE_ROTATE_REF '%s' (want id:degrees)", spec)
    return
end
deg = tonumber(deg)

local function onLoaded()
    local best, bestD = nil, math.huge
    for ref in tes3.player.cell:iterateReferences() do
        if ref.object.id:lower() == id:lower() then
            local d = ref.position:distance(tes3.player.position)
            if d < bestD then best, bestD = ref, d end
        end
    end
    if not best then
        mwse.log("[rotateref] !! no '%s' in %s", id, tes3.player.cell.id)
        return
    end
    local o = best.orientation
    best.orientation = tes3vector3.new(o.x, o.y, o.z + math.rad(deg))
    mwse.log("[rotateref] %s at (%.0f, %.0f, %.0f), %.0f u from the player: z %.1f -> %.1f deg",
        best.id, best.position.x, best.position.y, best.position.z, bestD,
        math.deg(o.z), math.deg(best.orientation.z))
end

event.register("loaded", onLoaded)
