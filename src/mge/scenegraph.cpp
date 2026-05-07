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
#include "NIPointLight.h"
#include "NIPointer.h"
#include "NIRTTIDefines.h"

#include <cmath>

#include "configuration.h"
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
        std::vector<PointLight>                g_pointLights;

        // Lifetime anchor — refcount-holds every NI::AVObject we exposed
        // so consumers can dereference safely between rebuilds even if
        // Lua scripts detach the underlying engine objects mid-frame.
        std::vector<NI::Pointer<NI::AVObject>> g_pinned;

        // Mirrors the engine's per-light falloff branching from master's
        // ffeshader.cpp constant-array path. NI::PointLight gives us the
        // raw attenuation values; the engine interprets them as four
        // distinct light kinds based on which fields are non-zero.
        PointLight extractPointLight(const NI::PointLight* pl) {
            PointLight out{};

            out.worldPos[0] = pl->worldTransform.translation.x;
            out.worldPos[1] = pl->worldTransform.translation.y;
            out.worldPos[2] = pl->worldTransform.translation.z;

            const float k0     = pl->constantAttenuation;
            const float k1     = pl->linearAttenuation;
            const float k2     = pl->quadraticAttenuation;
            const float dimmer = pl->dimmer;

            out.diffuse[0] = pl->diffuse.r * dimmer;
            out.diffuse[1] = pl->diffuse.g * dimmer;
            out.diffuse[2] = pl->diffuse.b * dimmer;
            out.ambient    = 0.0f;
            out.falloff[0] = k0;
            out.falloff[1] = k1;
            out.falloff[2] = k2;

            if (k0 > 0.0f) {
                // Standard point light source — pass-through.
            } else if (k2 > 0.0f) {
                // Morrowind-Code-Patch magic light (only quadratic set).
                // Engine path uses the carryover bufferFalloffConstant from
                // the previous standard light; we use the same default it
                // initialises to (0.33).
                constexpr float kMcpDefaultK0 = 0.33f;
                out.diffuse[0] *= kMcpDefaultK0;
                out.diffuse[1] *= kMcpDefaultK0;
                out.diffuse[2] *= kMcpDefaultK0;
                out.ambient    = 1.0f + 1e-4f / std::sqrt(k2);
                out.falloff[0] = kMcpDefaultK0;
                out.falloff[1] = 0.0f;
                out.falloff[2] = kMcpDefaultK0 * k2;
            } else if (k1 == 0.10000001f) {
                // Projectile light — engine sets exactly { 0, 3*(1/30), 0 }.
                // Replacement falloff is significantly brighter to look cool.
                out.falloff[0] = 0.0f;
                out.falloff[1] = 0.0f;
                out.falloff[2] = 5e-5f;
            } else if (k1 > 0.0f) {
                // Light-magic spell effect: { 0, 3 / (22 * magnitude), 0 }.
                // Approximated as half-lambert weight + quadratic falloff.
                const float brightness = 0.25f + 1e-4f / k1;
                out.diffuse[0]  = brightness;
                out.diffuse[1]  = brightness;
                out.diffuse[2]  = brightness;
                out.ambient     = 1.0f;
                out.falloff[0]  = 0.0f;
                out.falloff[1]  = 0.0f;
                out.falloff[2]  = 0.5555f * k1 * k1;
                out.worldPos[2] += 25.0f;  // lift out of ground embedment
            }

            // Bethesda overloads NI::Light::specular.r as the modder-set
            // fade radius (verified by NIPointLight.h:28 comment).
            out.radius = pl->specular.r;
            return out;
        }

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
                auto* light = static_cast<NI::Light*>(av);
                g_lights.push_back(light);

                if (av->isInstanceOfType(NI::RTTIStaticPtr::NiPointLight)) {
                    g_pointLights.push_back(
                        extractPointLight(static_cast<const NI::PointLight*>(av)));
                }
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
            g_pointLights.clear();

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

        // Default-off: ensure the walk has zero overhead on installs that
        // haven't opted in. When the flag is off, output caches are kept
        // empty so consumers (FFE texture-light variant) never activate.
        if (!Configuration.UseSceneGraphSnapshot) {
            if (!g_pinned.empty() || !g_nodes.empty() || !g_lights.empty() || !g_pointLights.empty()) {
                g_pinned.clear();
                g_nodes.clear();
                g_lights.clear();
                g_pointLights.clear();
                ++g_frameRevision;
            }
            return;
        }

        if (needsRebuild()) {
            rebuild();
        } else {
            ++g_framesSinceLastWalk;
        }
    }

    const std::vector<NI::Light*>&    lights()         { return g_lights; }
    const std::vector<NI::AVObject*>& nodes()          { return g_nodes; }
    const std::vector<PointLight>&    pointLights()    { return g_pointLights; }
    uint64_t                          frameRevision()  { return g_frameRevision; }

}
