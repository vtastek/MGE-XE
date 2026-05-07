// MGE-side consumer prelude — defines SE_IS_MGE / MWSE_NO_CUSTOM_ALLOC and
// pulls the prelude SharedSE headers expect. Force-included via the vcxproj
// on SharedSE-compiled TUs as well, so compile flags stay consistent across
// every translation unit that touches NI types.
#include "mge_se_prelude.h"

// SharedSE NI types.
#include "NIAVObject.h"
#include "NINode.h"
#include "NILight.h"
#include "NIDirectionalLight.h"
#include "NIPointer.h"
#include "NIRTTIDefines.h"

#include "datahandler_view.h"
#include "scenegraph.h"

namespace MGE::SceneGraph {

    namespace {
        // Bridge state.
        void*    g_dataHandler   = nullptr;
        uint64_t g_frameRevision = 0;

        // Output caches. Public accessors return const refs to these.
        // Pointers stay valid until the next rebuild because g_pinned
        // holds NI::Pointer<> refs for the same objects.
        std::vector<NI::Light*>                g_lights;
        std::vector<NI::AVObject*>             g_nodes;

        // Lifetime anchor — refcount-holds every NI::AVObject we exposed
        // so consumers can dereference safely between rebuilds even if
        // Lua scripts detach the underlying engine objects mid-frame.
        std::vector<NI::Pointer<NI::AVObject>> g_pinned;

        // Throttle state.
        constexpr int kForceRebuildEveryNFrames = 5;
        void* g_lastSeenCurrentCell  = nullptr;
        int   g_lastWorldObjChildCnt = -1;
        int   g_lastPickObjChildCnt  = -1;
        int   g_framesSinceLastWalk  = kForceRebuildEveryNFrames; // force first walk

        // Recursive descent into one subtree. Skips AppCulled subtrees
        // (engine/script-hidden — not "in" the scene this frame).
        void walk(NI::AVObject* av) {
            if (!av) return;
            if (av->getAppCulled()) return;

            g_pinned.emplace_back(av);
            g_nodes.push_back(av);

            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiLight)) {
                g_lights.push_back(static_cast<NI::Light*>(av));
            }

            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) {
                auto node = static_cast<NI::Node*>(av);
                const auto count = node->children.getFilledCount();
                for (size_t i = 0; i < count; ++i) {
                    walk(node->children.at(i));
                }
            }
        }

        bool needsRebuild() {
            if (g_framesSinceLastWalk >= kForceRebuildEveryNFrames) {
                return true;
            }

            void* cur = MGE::DataHandlerView::currentCell(g_dataHandler);
            if (cur != g_lastSeenCurrentCell) {
                return true;
            }

            auto wor  = MGE::DataHandlerView::worldObjectRoot(g_dataHandler);
            auto pick = MGE::DataHandlerView::worldPickObjectRoot(g_dataHandler);
            const int worCnt  = wor  ? static_cast<int>(wor->children.getFilledCount())  : 0;
            const int pickCnt = pick ? static_cast<int>(pick->children.getFilledCount()) : 0;
            if (worCnt != g_lastWorldObjChildCnt || pickCnt != g_lastPickObjChildCnt) {
                return true;
            }

            return false;
        }

        void rebuild() {
            // Releasing g_pinned drops the refcount on the previous frame's
            // NI::AVObjects. Clearing g_nodes/g_lights afterwards is just
            // bookkeeping (the raw pointers were aliases).
            g_pinned.clear();
            g_nodes.clear();
            g_lights.clear();

            walk(MGE::DataHandlerView::worldObjectRoot(g_dataHandler));
            walk(MGE::DataHandlerView::worldPickObjectRoot(g_dataHandler));

            // sgSunlight is a sibling root of the two NiNodes — the global
            // directional light. Treat as a synthetic single-node subtree.
            if (auto sun = MGE::DataHandlerView::sgSunlight(g_dataHandler)) {
                walk(reinterpret_cast<NI::AVObject*>(sun));
            }

            // Update throttle trackers AFTER the walk so the child counts
            // reflect what we actually saw.
            g_lastSeenCurrentCell = MGE::DataHandlerView::currentCell(g_dataHandler);
            auto wor  = MGE::DataHandlerView::worldObjectRoot(g_dataHandler);
            auto pick = MGE::DataHandlerView::worldPickObjectRoot(g_dataHandler);
            g_lastWorldObjChildCnt = wor  ? static_cast<int>(wor->children.getFilledCount())  : 0;
            g_lastPickObjChildCnt  = pick ? static_cast<int>(pick->children.getFilledCount()) : 0;

            g_framesSinceLastWalk = 0;
            ++g_frameRevision;
        }
    }

    void setDataHandler(void* dh) {
        g_dataHandler = dh;
    }

    void* getDataHandler() {
        return g_dataHandler;
    }

    void onFrameReady() {
        if (!g_dataHandler) return;

        if (needsRebuild()) {
            rebuild();
        } else {
            ++g_framesSinceLastWalk;
        }
    }

    const std::vector<NI::Light*>&    lights()         { return g_lights; }
    const std::vector<NI::AVObject*>& nodes()          { return g_nodes; }
    uint64_t                          frameRevision()  { return g_frameRevision; }

}
