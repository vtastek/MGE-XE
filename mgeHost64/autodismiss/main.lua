-- mgeHost64 perf harness — AUTO-DISMISS the startup confirmation dialogs.
--
-- WHY THIS EXISTS. A minimized harness run has nobody at the keyboard, and Morrowind stacks
-- confirmation dialogs during startup that block the load until each is answered. The host then
-- reaches `[seam] render init ok ... sceneReady=0`, never receives a scene, never prints a
-- `gpu split:` heartbeat, and the run times out with 0 samples -- which in the log is
-- indistinguishable from a host hang, and cost several terrain-PBR A/B runs to a hunt for a
-- rendering bug that did not exist.
--
-- !! WHAT THE DIALOGS ACTUALLY ARE, measured rather than assumed: 26 per launch of
-- "Texture Load Error!: vt\<name>.tga", with Yes / No / Yes To All, from the `vt\` PBR test-rig
-- textures. Fixing those references at the source would remove the dialogs entirely; this mod is
-- the harness's insurance, not a substitute for that.
--
-- !! THE SAVE WAS A RED HERRING, AND IT COST FOUR WRONG THEORIES. MWSE logs
-- "Local count for script 'sleeperScript' (Patch for Purists.esm)' differs from local count for
-- saved reference data ... in function 'loadGame'" on the stuck launches -- and also on launches
-- that load fine, so it never was the blocker. On the way to finding that out:
--   1. "It is a MenuMessage" -- right, but the handler never fired, so the silence was read as
--      absence. See the timer note in onMenuMessage: that is why it never fired.
--   2. "There is no dialog, it is not a MenuMessage" -- drawn from a Win32 window probe that ran
--      45 s after launch, by which time the user had pressed space and the game had LOADED. A
--      probe of the rescued state says nothing about the stuck state.
--   3. "It is a native #32770, dismiss it from outside" -- the dismisser was written in UTF-8 with
--      em-dashes, PowerShell 5.1 reads .ps1 as ANSI, and it never parsed. Its silence looked like
--      "no dialog found" too.
--   4. "The save is drifted, retry until it loads" -- a retry wrapper for a failure that repeats
--      every launch.
-- What settled it was one UNFILTERED uiActivated log naming `MenuMessage`, then one dump of that
-- menu's element tree. Instrument before theorising; three of those four theories were built on a
-- silence that had a bug behind it.
--
-- !! GATED ON AN ENV VAR, AND DELIBERATELY NOT ALWAYS ON. A mod that clicks the affirmative button
-- of every confirmation box is a footgun in normal play -- "overwrite this save", "are you sure you
-- want to rest". It arms only when the harness sets MGE_AUTODISMISS=1, and it disarms itself the
-- moment the save is loaded, so it can never reach an in-game dialog.
-- MGE_AUTODISMISS_VERBOSE=1 adds the unfiltered menu-name log that cracked this.

local ARMED = (os.getenv("MGE_AUTODISMISS") == "1")

if not ARMED then
    return
end

mwse.log("[autodismiss] armed (MGE_AUTODISMISS=1) — will click through startup dialogs until loaded")

local disarmed = false
local clicks = 0

-- Click the dialog's button.
--
-- !! THE FIRST VERSION OF THIS NEVER CLICKED ANYTHING, and the reason is worth keeping: it looked
-- for `child.widget ~= nil and child.widget.state ~= nil` as a "this is a button" test, which is
-- not how MWSE exposes buttons, so the walk always returned nil. The mod logged nothing, and the
-- silence was then misread as "no MenuMessage appears at all" -- which sent me to a Win32 window
-- probe, a native-dialog dismisser and two wrong theories before an UNFILTERED uiActivated log
-- showed `MenuMessage` arriving exactly where it always had.
--
-- So: no cleverness about what a button is. Walk every descendant, and trigger mouseClick on any
-- element that carries TEXT, deepest-last. MW's MenuMessage holds its prompt in a label and its
-- choices in buttons, and clicking a label is a no-op, so trying all of them is harmless and
-- cannot be defeated by a widget-type guess being wrong again.
local function dumpAndClick(menu)
    if menu == nil then
        return false
    end
    -- What these dialogs actually are, now that one has been read instead of guessed:
    --
    --   MenuMessage_message      "Texture Load Error!: vt\\metalshinysphere.tga ..."
    --   MenuMessage_button_layout
    --     text='Yes'  text='No'  text='Yes To All'
    --
    -- 26 of them per launch, all from the `vt\` PBR test-rig textures, NOT from the save at all.
    -- The `sleeperScript` local-count warning that this whole hunt was built around is a red
    -- herring: it fires on loads that succeed too. The blocker is a stack of missing-texture
    -- prompts, which is why pressing space cleared them one at a time.
    --
    -- So prefer "Yes To All" -- one click retires the entire stack -- and fall back to the usual
    -- affirmatives. Then STOP: clicking a button destroys the menu, and the previous version kept
    -- walking its remaining collected elements afterwards, which is where
    -- "sol: received nil for 'self'" came from. One click, one return.
    local prefer = { "Yes To All", "Yes", "Ok", "OK", "Continue" }
    local found = {}
    local function walk(el, depth)
        if el == nil or depth > 8 then
            return
        end
        for _, child in ipairs(el.children or {}) do
            local txt
            if pcall(function() txt = child.text end) and txt ~= nil and txt ~= "" then
                if found[txt] == nil then
                    found[txt] = child
                end
            end
            walk(child, depth + 1)
        end
    end
    walk(menu, 1)

    for _, want in ipairs(prefer) do
        local el = found[want]
        if el ~= nil then
            local ok = pcall(function() el:triggerEvent("mouseClick") end)
            if ok then
                clicks = clicks + 1
                if clicks <= 3 or (clicks % 10) == 0 then
                    mwse.log("[autodismiss] clicked '%s' (#%d)", want, clicks)
                end
                return true
            end
        end
    end
    -- Nothing recognised: name what was there, so the next new dialog shape is one log line away
    -- from being handled rather than another round of theories.
    local names = {}
    for txt, _ in pairs(found) do
        table.insert(names, "'" .. tostring(txt) .. "'")
    end
    mwse.log("[autodismiss] UNHANDLED dialog, buttons seen: %s", table.concat(names, " "))
    return false
end

local function onMenuMessage(e)
    if disarmed then
        return
    end
    -- !! CLICKED INLINE, NOT FROM A TIMER, AND THAT IS THE WHOLE BUG.
    -- The previous version deferred this by `timer.start{ duration = 0.05, type = timer.real }`
    -- "for one frame's grace so the menu has finished building". But this dialog appears while the
    -- game is blocked inside a MODAL loop in tes3.loadGame: no frames advance, so MWSE's timers
    -- never tick, so the callback never ran. The mod logged nothing at all -- and that silence was
    -- read as "no MenuMessage exists", which is what sent me to a Win32 window probe and a
    -- native-dialog dismisser for a dialog that was in-engine the whole time.
    -- `uiActivated` with newlyCreated=true already fires after the element tree is built, so the
    -- grace period was not buying anything it cost three theories to discover.
    dumpAndClick(e.element)
end

-- DIAGNOSTIC ARM. The MenuMessage filter above never fired on a stuck launch, and a Win32 probe of
-- the stuck process (mgeHost64/probe-stuck-windows.ps1, caught at the right moment this time) found
-- NO dialog window of any class -- only the main 'Morrowind' window, responding. So whatever is
-- blocking the load is neither an OS dialog nor a MenuMessage, and guessing its id has now cost
-- three wrong theories. This logs EVERY menu MWSE activates during startup, so the next stuck run
-- names it instead of me inferring it.
local VERBOSE = (os.getenv("MGE_AUTODISMISS_VERBOSE") == "1")

local function onAnyMenu(e)
    if disarmed or not VERBOSE then
        return
    end
    mwse.log("[autodismiss] uiActivated: '%s'", tostring(e.newlyCreated) .. " " .. tostring(e.element and e.element.name or "?"))
end

-- `loaded` fires once the save is actually in. From then on every dialog is an in-game one and is
-- none of this mod's business.
local function onLoaded()
    disarmed = true
    event.unregister("uiActivated", onMenuMessage, { filter = "MenuMessage" })
    event.unregister("uiActivated", onAnyMenu)
    event.unregister("loaded", onLoaded)
    mwse.log("[autodismiss] disarmed after load (%d dialogs dismissed)", clicks)
end

event.register("uiActivated", onAnyMenu)
event.register("uiActivated", onMenuMessage, { filter = "MenuMessage" })
event.register("loaded", onLoaded)
