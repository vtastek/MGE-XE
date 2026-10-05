-- Mouse Probe (dev aid: why a hand mouse turn looks jittery when a scripted AutoTurn360 is smooth)
--
-- WHAT: every simulate frame, append one CSV row to Data Files/MWSE/MouseProbe.csv:
--   frame, clock_ms (wall clock), delta_ms (MW's frame time), lX, lY (the raw DirectInput deltas MW
--   read this frame), yaw / pitch (player orientation z / x, radians), px, py (position: a forward
--   run moves by speed * delta, so its step against the turn step shows two clocks), sensX
-- Read it with mgexe-devkit/tools/mouse-probe.py. The questions it answers:
--   yaw / lX constant?        MW turns by the raw count (no dt scaling, no smoothing) — or not.
--   lX per real ms steady?    a steady hand at a steady rate should give lX proportional to the
--                             frame's real interval; a count that alternates (1,2,1,2 reports per
--                             frame) is mouse-poll quantisation beating against the frame rate.
--   clock vs delta            MW's dt against the wall clock between the frames it samples.
--
-- CONFIG (Data Files/MWSE/config/MouseProbe.json): { "enabled": true, "maxRows": 30000 }.
-- DEFAULT OFF. Nothing moves the player: the person at the mouse does the turning.

local cfg = mwse.loadConfig("MouseProbe", { enabled = false, maxRows = 30000 })

local file = nil
local rows = 0
local frame = 0

-- Wall clock in ms. QueryPerformanceCounter through the LuaJIT FFI; os.clock (MSVC clock(): 1 ms
-- steps, too coarse against 13 ms frames) only if the FFI is unavailable.
local nowMs = function() return os.clock() * 1000.0 end
local okFfi, ffi = pcall(require, "ffi")
if okFfi then
    pcall(ffi.cdef, [[
        int QueryPerformanceCounter(int64_t* count);
        int QueryPerformanceFrequency(int64_t* freq);
    ]])
    local freq = ffi.new("int64_t[1]")
    local cnt = ffi.new("int64_t[1]")
    if ffi.C.QueryPerformanceFrequency(freq) ~= 0 then
        local f = tonumber(freq[0]) / 1000.0
        nowMs = function()
            ffi.C.QueryPerformanceCounter(cnt)
            return tonumber(cnt[0]) / f
        end
    end
end

local function onSimulate(e)
    if not file then return end
    frame = frame + 1
    local ms = tes3.worldController.inputController.mouseState
    local o, p = tes3.player.orientation, tes3.player.position
    file:write(string.format("%d,%.3f,%.3f,%d,%d,%.6f,%.6f,%.2f,%.2f,%.5f\n", frame, nowMs(), e.delta * 1000.0,
        ms.x, ms.y, o.z, o.x, p.x, p.y, tes3.worldController.mouseSensitivityX))
    rows = rows + 1
    if rows % 300 == 0 then file:flush() end
    if rows >= cfg.maxRows then
        file:close()
        file = nil
        mwse.log("[mouseprobe] stopped at %d rows", rows)
    end
end

local function onLoaded()
    if not cfg.enabled then return end
    if file then return end
    file = io.open("Data Files/MWSE/MouseProbe.csv", "w")
    if not file then
        mwse.log("[mouseprobe] !! could not open Data Files/MWSE/MouseProbe.csv")
        return
    end
    file:write("frame,clock_ms,delta_ms,lX,lY,yaw,pitch,px,py,sensX\n")
    mwse.log("[mouseprobe] recording up to %d frames to Data Files/MWSE/MouseProbe.csv", cfg.maxRows)
    event.register("simulate", onSimulate)
end

event.register("loaded", onLoaded)
