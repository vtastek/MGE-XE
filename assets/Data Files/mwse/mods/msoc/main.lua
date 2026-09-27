-- msoc: native plugin loader + MCM bootstrap.
--
-- Loads the C++ plugin, pushes the values from msoc.json into the
-- native statics via plugin.configure(table), then registers the MCM
-- page once MWSE's mcm module is ready.
--
-- SHIPPED DISABLED with MGE XE. Loading msoc.dll installs its engine
-- hooks unconditionally (EnableMSOC=false only skips its resources), and
-- its CullShow detour makes MGE refuse to install its own traversal and
-- object-discovery feed. So the gate is here, BEFORE anything include()s
-- the DLL — config.lua and mcm.lua both do. `LoadPlugin` is a new key on
-- purpose: existing msoc.json files carry EnableMSOC=true and would
-- otherwise keep loading it. Turning it on in the MCM takes a restart.

local gate = mwse.loadConfig("msoc", { LoadPlugin = false })

if not gate.LoadPlugin then
    mwse.log("[msoc] not loaded (LoadPlugin=false): MGE XE's own culling is active. "
        .. "Enable it in the Mod Config menu and restart to use msoc instead.")
    event.register("modConfigReady", function()
        local template = mwse.mcm.createTemplate({ name = "MSOC" })
        template:saveOnClose("msoc", gate)
        local page = template:createSideBarPage({
            label = "MSOC",
            description = "Masked software occlusion culling. Disabled by default: "
                .. "MGE XE's renderer does its own culling, and msoc replaces it when loaded.",
        })
        page:createOnOffButton({
            label = "Load the msoc plugin (restart required)",
            description = "Replaces MGE XE's own scene traversal and culling with msoc's. "
                .. "Takes effect the next time the game starts.",
            variable = mwse.mcm.createTableVariable({ id = "LoadPlugin", table = gate }),
        })
        template:register()
    end)
    return
end

local msoc = include("msoc")

if not msoc then
    mwse.log("[msoc] msoc.dll not loaded. If you're using a mod manager, "
        .. "check that .dll files weren't filtered out of the install.")
    return
end

mwse.log("[msoc] plugin loaded, version=%s, mocLink=%s",
    tostring(msoc.version), tostring(msoc.mocLink))

local cfg = require("msoc.config")
cfg.syncToNative(msoc)

mwse.log("[msoc] config synced from msoc.json: EnableMSOC=%s, ExteriorCull=%s",
    tostring(cfg.config.EnableMSOC),
    tostring(cfg.config.OcclusionEnableExterior))

-- mcm.lua registers its own modConfigReady handler; require at top
-- level so that handler is installed before the event fires.
require("msoc.mcm")
