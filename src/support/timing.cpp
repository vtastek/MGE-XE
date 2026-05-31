
#include "stdint.h"
#include "timing.h"
#include "winheader.h"

#ifdef TRACY_ENABLE
#include <intrin.h>
#endif



// Calculations use doubles, as int64 ops bring in an excessive amount of library code on a 32-bit target

static double reciprocalFreq;
static LARGE_INTEGER initialTime;

#ifdef TRACY_ENABLE
namespace {
    // Per-frame tally of engine timer reads + a small caller histogram keyed by
    // return address. Single-threaded (engine main loop + Present both run on
    // the main thread), so plain statics are fine.
    constexpr int kMaxCallers = 8;
    unsigned s_callCount = 0;
    void*    s_callerAddr[kMaxCallers]  = {};
    unsigned s_callerCount[kMaxCallers] = {};

    inline void recordCaller(void* ra) {
        ++s_callCount;
        for (int i = 0; i < kMaxCallers; ++i) {
            if (s_callerAddr[i] == ra)      { ++s_callerCount[i]; return; }
            if (s_callerAddr[i] == nullptr) { s_callerAddr[i] = ra; s_callerCount[i] = 1; return; }
        }
        ++s_callerCount[0];  // histogram full: lump leftovers into slot 0
    }
}

namespace TimerProbe {
    unsigned takeFrameCallCount() {
        unsigned c = s_callCount;
        s_callCount = 0;
        return c;
    }

    int takeCallerStats(CallerStat* out, int maxOut) {
        int n = 0;
        for (int i = 0; i < kMaxCallers; ++i) {
            if (s_callerCount[i] != 0 && n < maxOut) {
                out[n].addr  = s_callerAddr[i];
                out[n].count = s_callerCount[i];
                ++n;
            }
            s_callerCount[i] = 0;
            s_callerAddr[i]  = nullptr;
        }
        return n;
    }
}
#endif

void HighResolutionTimer::init() {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&initialTime);
    reciprocalFreq = 1.0 / frequency.QuadPart;
}

int HighResolutionTimer::getMicroseconds() {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);

    double x = 1000000 * (double(t.QuadPart - initialTime.QuadPart) * reciprocalFreq);
    return int(int64_t(x));
}

int HighResolutionTimer::getMilliseconds() {
#ifdef TRACY_ENABLE
    recordCaller(_ReturnAddress());
#endif
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);

    double x = 1000 * (double(t.QuadPart - initialTime.QuadPart) * reciprocalFreq);
    return int(int64_t(x));
}
