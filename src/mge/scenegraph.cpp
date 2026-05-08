// MGE-side consumer prelude — defines SE_IS_MGE / MWSE_NO_CUSTOM_ALLOC and
// pulls the prelude SharedSE headers expect. Force-included via the vcxproj
// on SharedSE-compiled TUs as well, so compile flags stay consistent across
// every translation unit that touches NI types.
#include "mge_se_prelude.h"

// SharedSE NI types.
#include "NIAVObject.h"
#include "NINode.h"
#include "NILight.h"
#include "NIPointLight.h"
#include "NIRTTIDefines.h"

#include "configuration.h"
#include "datahandler_view.h"
#include "scenegraph.h"
#include "support/log.h"

namespace MGE::SceneGraph {

    namespace {
        // Bridge state.
        void*    g_dataHandler   = nullptr;
        uint64_t g_frameRevision = 0;

        // Output cache — POD only. No NI pointers held past walk return:
        // every dereference of a visited NI::PointLight happens inside walk()
        // and produces scalar field copies in `extractPointLight`. PODs don't
        // dangle, so no NI::Pointer<> pinning is needed and no g_pinned
        // anchor vector. Future consumers that need typed NI access (e.g.
        // shadow map per-light list) add their own typed pinned vector then.
        std::vector<PointLight> g_pointLights;

        // Ships raw NI::PointLight fields. NO engine-state branching —
        // the magic patterns master's constant-array path decodes
        // (`{0, 0.1, 0}` projectile, `{0, k1, 0}` spell, `{0, 0, k2}`
        // MCP magic) are markers Morrowind sets on D3DLIGHT9 state via
        // SetLight, not on the raw NI fields. Pattern-matching on raw
        // NI values misclassifies static lanterns on installs whose
        // Morrowind.ini favours linear-dominant falloff (a common
        // configuration). msoc-plugin's deleted producer + 7d49548's
        // texture-light shader both shipped raw fields and applied
        // the standard 1/(k0 + k1·d + k2·d²) formula directly; that's
        // what we mirror here.
        PointLight extractPointLight(const NI::PointLight* pl) {
            PointLight out{};

            out.worldPos[0] = pl->worldTransform.translation.x;
            out.worldPos[1] = pl->worldTransform.translation.y;
            out.worldPos[2] = pl->worldTransform.translation.z;

            const float dimmer = pl->dimmer;
            out.diffuse[0] = pl->diffuse.r * dimmer;
            out.diffuse[1] = pl->diffuse.g * dimmer;
            out.diffuse[2] = pl->diffuse.b * dimmer;

            out.falloff[0] = pl->constantAttenuation;
            out.falloff[1] = pl->linearAttenuation;
            out.falloff[2] = pl->quadraticAttenuation;

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

            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiPointLight)) {
                auto* pl = static_cast<const NI::PointLight*>(av);
                // Skip lights with no engine-set radius. Morrowind stores the
                // modder-set Radius in NI::Light::specular.r (Bethesda overload)
                // and the engine's own per-object selection at 0x4D2F40
                // (game_dynamicLightTest) culls the light when
                //   `objectToLightDist - objectRadius > specular.r`.
                // A `specular.r == 0` light is excluded by every object in
                // vanilla — it never enters any effect list, never gets pushed
                // via SetLight, never lights anything. Filtering here saves
                // a texLightData pool slot and a per-mesh sphere-AABB test
                // for a light that would have been dropped anyway.
                if (pl->specular.r > 0.0f) {
                    g_pointLights.push_back(extractPointLight(pl));
                }
            }

            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) {
                auto node = static_cast<NI::Node*>(av);
                // Iterate [0, endIndex), null-skip in walk(). NITArray is
                // sparse — filledCount counts non-null entries, but they
                // may sit at non-contiguous slots up to endIndex. Using
                // filledCount as the loop bound silently drops any light
                // whose slot index is >= filledCount (visible as missing
                // lights / dark patches in dense interiors). msoc's walk
                // used endIndex; mirror that here.
                const auto count = node->children.getEndIndex();
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
            const int worCnt  = wor  ? static_cast<int>(wor->children.getEndIndex())  : 0;
            const int pickCnt = pick ? static_cast<int>(pick->children.getEndIndex()) : 0;
            if (worCnt != g_lastWorldObjChildCnt || pickCnt != g_lastPickObjChildCnt) {
                return true;
            }

            return false;
        }

        void rebuild() {
            g_pointLights.clear();

            walk(MGE::DataHandlerView::worldObjectRoot(g_dataHandler));
            walk(MGE::DataHandlerView::worldPickObjectRoot(g_dataHandler));
            // sgSunlight is a sibling root of the two NiNodes — the global
            // directional light. NiDirectionalLight is not a NiPointLight,
            // so the walk's RTTI test naturally skips it for g_pointLights.
            // Visited only so future per-subtype POD vectors (e.g.
            // g_directionalLights) get populated by the same pass.
            if (auto sun = MGE::DataHandlerView::sgSunlight(g_dataHandler)) {
                walk(reinterpret_cast<NI::AVObject*>(sun));
            }

            // Update throttle trackers AFTER the walk so the child counts
            // reflect what we actually saw.
            g_lastSeenCurrentCell = MGE::DataHandlerView::currentCell(g_dataHandler);
            auto wor  = MGE::DataHandlerView::worldObjectRoot(g_dataHandler);
            auto pick = MGE::DataHandlerView::worldPickObjectRoot(g_dataHandler);
            g_lastWorldObjChildCnt = wor  ? static_cast<int>(wor->children.getEndIndex())  : 0;
            g_lastPickObjChildCnt  = pick ? static_cast<int>(pick->children.getEndIndex()) : 0;

            g_framesSinceLastWalk = 0;
            ++g_frameRevision;
        }
    }

    void setDataHandler(void* dh) {
        const bool firstStamp = (g_dataHandler == nullptr) && (dh != nullptr);
        g_dataHandler = dh;
        if (firstStamp) {
            LOG::logline("-- [SCENEGRAPH] DataHandler stamped: %p", dh);
        }
    }

    void* getDataHandler() {
        return g_dataHandler;
    }

    void onFrameReady() {
        if (!g_dataHandler) return;

        // Default-off: ensure the walk has zero overhead on installs that
        // haven't opted in. When the flag is off, the output cache is kept
        // empty so consumers (FFE texture-light variant) never activate.
        if (!Configuration.UseSceneGraphSnapshot) {
            if (!g_pointLights.empty()) {
                g_pointLights.clear();
                ++g_frameRevision;
            }
            return;
        }

        if (needsRebuild()) {
            rebuild();

            static uint64_t s_lastLoggedRev = (uint64_t)-1;
            if (g_frameRevision != s_lastLoggedRev && (g_frameRevision % 30) == 0) {
                LOG::logline("-- [SCENEGRAPH] rev=%llu pointLights=%zu",
                    (unsigned long long)g_frameRevision,
                    g_pointLights.size());
                s_lastLoggedRev = g_frameRevision;
            }
        } else {
            ++g_framesSinceLastWalk;
        }
    }

    const std::vector<PointLight>& pointLights()    { return g_pointLights; }
    uint64_t                       frameRevision()  { return g_frameRevision; }

}
