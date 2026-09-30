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
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>

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
    //
    // OFF THE CALLER'S THREAD. This used to fprintf the whole ring (~2 MB of text) under the ring lock,
    // on the calling thread: the host's render thread and MW's main thread. That was a stall every 300
    // frames that only existed while tracing — frame x00 waited ~22 ms on the host, x01/x02 ran ~35-43
    // ms against ~14 — and every traced measurement carried it (a walk run without the trace: worst
    // quiet-window frame 18-28 ms). Now the caller only copies the ring (a memcpy under the lock) and a
    // writer thread formats it. The file is written beside the target and renamed over it, so a kill
    // mid-write (the harness always kills) leaves the previous complete file instead of a torn row.
    // One write in flight at a time; a dump that finds the writer busy is skipped — the next one
    // carries the same ring, 300 frames later.
    inline void dump(const char* path, const char* process, double clockOffsetMs = 0.0) {
        if (!enabled()) { return; }
        static std::atomic<bool> s_busy{ false };
        if (s_busy.exchange(true)) { return; }
        struct Job {
            Span*         spans;
            std::uint32_t count;
            const char*   path;      // literals at every call site
            const char*   process;
            double        offset;
        };
        Job* job = new (std::nothrow) Job{ new (std::nothrow) Span[kCap], 0u, path, process, clockOffsetMs };
        if (!job || !job->spans) {
            if (job) { delete job; }
            s_busy.store(false);
            return;
        }
        {
            State& s = state();
            std::lock_guard<std::mutex> lk(s.mx);
            const std::uint32_t first = (s.head + kCap - s.count) % kCap;
            const std::uint32_t tail = (kCap - first < s.count) ? kCap - first : s.count;
            std::memcpy(job->spans, s.ring + first, tail * sizeof(Span));
            std::memcpy(job->spans + tail, s.ring, (s.count - tail) * sizeof(Span));
            job->count = s.count;
        }
        HANDLE h = CreateThread(nullptr, 0, [](LPVOID p) -> DWORD {
            Job* j = static_cast<Job*>(p);
            char tmp[MAX_PATH];
            std::snprintf(tmp, sizeof(tmp), "%s.tmp", j->path);
            FILE* f = nullptr;
            if (fopen_s(&f, tmp, "wb") == 0 && f) {
                std::fprintf(f, "# frametrace process=%s spans=%u\nlane,name,t0,t1,frame\n", j->process, j->count);
                for (std::uint32_t i = 0; i < j->count; ++i) {
                    const Span& s = j->spans[i];
                    std::fprintf(f, "%s,%s,%.4f,%.4f,%lld\n", s.lane, s.name,
                                 s.t0 + j->offset, s.t1 + j->offset, (long long)s.frame);
                }
                const bool ok = std::fclose(f) == 0;
                if (ok) { MoveFileExA(tmp, j->path, MOVEFILE_REPLACE_EXISTING); }
            }
            delete[] j->spans;
            delete j;
            s_busy.store(false);
            return 0;
        }, job, 0, nullptr);
        if (h) {
            CloseHandle(h);
        } else {
            delete[] job->spans;
            delete job;
            s_busy.store(false);
        }
    }

} // namespace FrameTrace
