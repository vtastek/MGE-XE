// SharedSE NI consumer prelude. Must come before any SharedSE include.
#include <windows.h>
#include <iterator>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#define span_CONFIG_SELECT_SPAN span_SPAN_NONSTD
#include <nonstd/span.hpp>

// MGE consumer identity + read-only allocator gate.
#define SE_IS_MGE 1
#define MWSE_NO_CUSTOM_ALLOC 1

#include "scenegraph.h"

namespace MGE::SceneGraph {

    namespace {
        void*    g_dataHandler   = nullptr;
        uint64_t g_frameRevision = 0;

        std::vector<NI::Light*>    g_lights;
        std::vector<NI::AVObject*> g_nodes;
    }

    void setDataHandler(void* dh) {
        g_dataHandler = dh;
    }

    void* getDataHandler() {
        return g_dataHandler;
    }

    void onFrameReady() {
        // Skeleton: actual walk + three-trigger throttle land in the next
        // commit. For now this is a no-op so the bridge wires up cleanly
        // and consumers see empty arrays.
        if (!g_dataHandler) {
            return;
        }
    }

    const std::vector<NI::Light*>&    lights()         { return g_lights; }
    const std::vector<NI::AVObject*>& nodes()          { return g_nodes; }
    uint64_t                          frameRevision()  { return g_frameRevision; }

}
