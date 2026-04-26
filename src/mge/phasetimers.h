#pragma once

// MOREFPS: temporary lightweight phase-timing instrumentation.
//
// Usage: at the top of any scope you want to measure, write
//     MGE_SCOPED_TIMER("cullDistantStatics");
// and the RAII destructor will add the elapsed microseconds into a
// named accumulator at scope exit. Call MGEPhaseTimers::report() once
// per second (e.g., piggy-backed on an existing 60-frame diagnostic)
// to flush a one-line-per-bucket summary into mgeXE.log and reset.
//
// Explicitly "temporary" — everything lives in two files and is
// grep-visible via the MGE_SCOPED_TIMER / MGEPhaseTimers tokens.
// Can be ripped out in a single commit once we're done profiling.

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
