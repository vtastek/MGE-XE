#pragma once

// Lightweight phase-timing instrumentation. Reports gated by the
// "Log Distant Pipeline" mge.ini flag — off by default, so the
// scoped timers accumulate but no per-frame log output is produced
// unless the user opts into pipeline logging.
//
// Usage: at the top of any scope you want to measure, write
//     MGE_SCOPED_TIMER("cullDistantStatics");
// and the RAII destructor will add the elapsed microseconds into a
// named accumulator at scope exit. MGEPhaseTimers::report() flushes a
// one-line-per-bucket summary into mgeXE.log and resets the counters;
// it's called from the per-frame diagnostic block in cullDistantStatics
// when LogDistantPipeline is enabled.
//
// Everything lives in two files and is grep-visible via the
// MGE_SCOPED_TIMER / MGEPhaseTimers tokens, so the instrumentation
// can be removed cleanly if it's ever no longer needed.

#include <chrono>
#include <cstdint>

namespace MGEPhaseTimers {
    // Add `us` microseconds to the bucket identified by `name`. The
    // pointer is stored verbatim — it must point to string-literal or
    // otherwise-stable storage that outlives the next `report()` call.
    void add(const char* name, std::uint64_t us);

    // Emit one line per bucket (sorted by total descending) into the
    // MGE-XE log, then zero all buckets for the next sampling window.
    // Safe to call even when no buckets have recorded yet (no-op).
    void report();
}

// RAII scope timer. Uses steady_clock (monotonic, high-res on Win32).
struct MGEScopedTimer {
    const char* name;
    std::chrono::steady_clock::time_point start;

    explicit MGEScopedTimer(const char* n)
        : name(n), start(std::chrono::steady_clock::now()) {}

    ~MGEScopedTimer() {
        using namespace std::chrono;
        const auto us = duration_cast<microseconds>(steady_clock::now() - start).count();
        MGEPhaseTimers::add(name, static_cast<std::uint64_t>(us));
    }

    // Non-copyable / non-movable — lifetime is strictly lexical scope.
    MGEScopedTimer(const MGEScopedTimer&) = delete;
    MGEScopedTimer& operator=(const MGEScopedTimer&) = delete;
};

#define MGE_TIMER_CONCAT_INNER(a, b) a##b
#define MGE_TIMER_CONCAT(a, b) MGE_TIMER_CONCAT_INNER(a, b)
// Concatenating with __LINE__ lets multiple timers live in the same
// function without colliding on the variable name.
#define MGE_SCOPED_TIMER(name) MGEScopedTimer MGE_TIMER_CONCAT(_mgePhaseTimer_, __LINE__)(name)
