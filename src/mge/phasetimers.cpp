#include "phasetimers.h"
#include "support/log.h"

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

namespace MGEPhaseTimers {

// One bucket per named scope. Keyed by `const char*` pointer identity —
// the macro always passes string literals so same-string calls collapse
// into the same bucket (literals are pooled by the linker). Hash/equal
// use pointer value directly; good enough when everything resolves to
// the literal pool in the .rdata section.
struct Bucket {
    std::uint64_t totalUs = 0;
    std::uint64_t calls   = 0;
};

static std::unordered_map<const char*, Bucket> g_buckets;

void add(const char* name, std::uint64_t us) {
    auto& b = g_buckets[name];
    b.totalUs += us;
    ++b.calls;
}

void report() {
    if (g_buckets.empty()) {
        return;
    }

    // Sort buckets by total time descending so the expensive scopes
    // surface at the top of the log line block.
    std::vector<std::pair<const char*, Bucket>> sorted;
    sorted.reserve(g_buckets.size());
    for (const auto& kv : g_buckets) {
        sorted.emplace_back(kv);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                  return a.second.totalUs > b.second.totalUs;
              });

    for (const auto& [name, b] : sorted) {
        const std::uint64_t avg = b.calls > 0 ? (b.totalUs / b.calls) : 0;
        LOG::logline("-- PHASE: %-40s total=%llu us  calls=%llu  avg=%llu us",
                     name,
                     static_cast<unsigned long long>(b.totalUs),
                     static_cast<unsigned long long>(b.calls),
                     static_cast<unsigned long long>(avg));
    }

    // Reset every bucket for the next sampling window. Using clear()
    // would drop the string-pointer keys and re-hash next call; erase-
    // in-place with zero is slightly nicer (preserves allocator state),
    // but clear is simpler and the overhead is negligible.
    g_buckets.clear();
}

} // namespace MGEPhaseTimers
