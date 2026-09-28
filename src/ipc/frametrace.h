#pragma once

// Frame timeline trace (MGE_FRAME_TRACE=1): a span recorder shared by the client (mgecore, x86)
// and the Forge host (mgeHost64, x64), for looking at the two processes and the host GPU as
// OVERLAPPING BOXES on one time axis instead of as per-window averages. Render the dumps with
// mgexe-devkit/tools/frametrace-html.py.
//
// ONE CLOCK: every timestamp is QueryPerformanceCounter in ms. QPC is system-wide, so the two
// processes' spans line up without any handshake; host GPU timestamps are mapped onto it with
// ID3D12CommandQueue::GetClockCalibration by the host before they are recorded here.
//
// Each process keeps a ring of the most recent kCap spans and REWRITES its whole file from the ring
// on every dump() (the callers dump once per 300-frame heartbeat). Nothing waits for a clean exit —
// the harness kills the game — so the file on disk is always the last complete ring. With ~40 spans
// a frame the ring holds well over 300 frames, so the two processes' last dumps always overlap.
//
// Off (the default) costs one predictable branch per call site. Names and lanes must be string
// LITERALS (pointers are stored, not copied).

#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace FrameTrace {

    struct Span {
        const char*  lane;
        const char*  name;
        double       t0, t1;    // QPC ms
        std::int64_t frame;     // the shared-fence value this span belongs to; -1 = unknown
    };

    constexpr std::uint32_t kCap = 32768;

    struct State {
        Span          ring[kCap];
        std::uint32_t head = 0;     // next write
        std::uint32_t count = 0;
        std::mutex    mx;
    };

    inline State& state() { static State* s = new State(); return *s; }

    inline bool enabled() {
        static const bool on = [] {
            char v[8] = {};
            return GetEnvironmentVariableA("MGE_FRAME_TRACE", v, sizeof(v)) > 0 && v[0] == '1';
        }();
        return on;
    }

    inline double qpcMs() {
        static const double freq = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return (double)f.QuadPart; }();
        LARGE_INTEGER c; QueryPerformanceCounter(&c);
        return 1000.0 * (double)c.QuadPart / freq;
    }

    inline void span(const char* lane, const char* name, double t0, double t1, std::int64_t frame = -1) {
        if (!enabled() || !(t1 >= t0)) { return; }
        State& s = state();
        std::lock_guard<std::mutex> lk(s.mx);
        s.ring[s.head] = Span{ lane, name, t0, t1, frame };
        s.head = (s.head + 1) % kCap;
        if (s.count < kCap) { ++s.count; }
    }

    // `clockOffsetMs` is added to every stored time: the host records with hostNowMs() (steady_clock),
    // and passes (qpcMs() - hostNowMs()) so the file is in QPC ms like the client's.
    inline void dump(const char* path, const char* process, double clockOffsetMs = 0.0) {
        if (!enabled()) { return; }
        State& s = state();
        std::lock_guard<std::mutex> lk(s.mx);
        FILE* f = nullptr;
        if (fopen_s(&f, path, "wb") != 0 || !f) { return; }
        std::fprintf(f, "# frametrace process=%s spans=%u\nlane,name,t0,t1,frame\n", process, s.count);
        const std::uint32_t first = (s.head + kCap - s.count) % kCap;
        for (std::uint32_t i = 0; i < s.count; ++i) {
            const Span& p = s.ring[(first + i) % kCap];
            std::fprintf(f, "%s,%s,%.4f,%.4f,%lld\n", p.lane, p.name,
                         p.t0 + clockOffsetMs, p.t1 + clockOffsetMs, (long long)p.frame);
        }
        std::fclose(f);
    }

} // namespace FrameTrace
