#pragma once

class HighResolutionTimer {
public:
    static void init();
    static int getMicroseconds();
    static int getMilliseconds();
};

#ifdef TRACY_ENABLE
// Diagnostic probe for the engine frame-limiter. MGE redirects Morrowind's
// timeGetTime call sites to HighResolutionTimer::getMilliseconds (mwbridge.cpp
// patchFrameTimer), so every engine time read passes through here. We tally
// calls + their return addresses per frame; Present (mged3d8device.cpp) reads
// and resets these once per frame to plot/log them. Reveals whether the engine
// pacer busy-waits (thousands of calls/frame) vs sleeps, and which engine call
// site is the limiter loop. Compiled only in Tracy builds.
namespace TimerProbe {
    struct CallerStat { void* addr; unsigned count; };
    // Total getMilliseconds calls since the last take; resets to 0.
    unsigned takeFrameCallCount();
    // Copy up to maxOut distinct (caller, count) buckets; resets the histogram.
    int takeCallerStats(CallerStat* out, int maxOut);
}
#endif
