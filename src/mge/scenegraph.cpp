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
#include "NIDirectionalLight.h"
#include "NIRTTIDefines.h"

#include "configuration.h"
#include "datahandler_view.h"
#include "scenegraph.h"
#include "support/log.h"

#include <cstring>

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

        // Previous-frame snapshot for change detection. Same shape as
        // g_pointLights; swapped in/out of g_pointLights each rebuild so we
        // can byte-compare the new walk's output against the prior one and
        // bump g_frameRevision only when something actually moved.
        std::vector<PointLight> g_lastWalkedLights;

        // Directional-light output + previous-frame snapshot. Mirrors the
        // PointLight pair. Vanilla MW always has exactly one entry (the
        // sun); kept as a vector so the same swap+memcmp change-detection
        // path works without a special-case for size 0/1.
        std::vector<DirectionalLight> g_directionalLights;
        std::vector<DirectionalLight> g_lastWalkedDirectionalLights;

        DirectionalLight extractDirectionalLight(const NI::DirectionalLight* dl) {
            DirectionalLight out{};

            // World-space direction: rotation * local direction. The local
            // direction is set at NIF authoring time (typically a fixed
            // axis on the sun anchor) and stays constant; the day/night
            // controller drives worldTransform.rotation. Manual 3x3 multiply
            // avoids depending on whether SharedSE's Matrix33::operator*
            // (Vector3) is linked into MGE-XE.
            const auto& M = dl->worldTransform.rotation;
            const auto& d = dl->direction;
            out.worldDir[0] = M.m0.x * d.x + M.m0.y * d.y + M.m0.z * d.z;
            out.worldDir[1] = M.m1.x * d.x + M.m1.y * d.y + M.m1.z * d.z;
            out.worldDir[2] = M.m2.x * d.x + M.m2.y * d.y + M.m2.z * d.z;

            const float dimmer = dl->dimmer;
            out.diffuse[0] = dl->diffuse.r * dimmer;
            out.diffuse[1] = dl->diffuse.g * dimmer;
            out.diffuse[2] = dl->diffuse.b * dimmer;

            out.ambient[0] = dl->ambient.r;
            out.ambient[1] = dl->ambient.g;
            out.ambient[2] = dl->ambient.b;
            return out;
        }

        // Walk-cost instrumentation. Cumulative since process start; the
        // counters tick once per onFrameReady() call when the INI knob is
        // on. Periodic dump runs every 1800 walks (~30 sec at 60 FPS).
        // Walk count is exposed via frameCount() so consumers can divide
        // their cumulative counters by it for per-frame averages.
        LARGE_INTEGER       g_qpcFreq         = {};
        unsigned long long  g_walks           = 0;
        unsigned long long  g_walksTotalNs    = 0;
        unsigned long long  g_walksWithChange = 0;
        unsigned long long  g_lastReportWalks = 0;

        // Recursive descent into one subtree. Skips AppCulled subtrees
        // (engine/script-hidden — not "in" the scene this frame).
        void walk(NI::AVObject* av) {
            if (!av) return;
            if (av->getAppCulled()) return;

            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiPointLight)) {
                auto* pl = static_cast<const NI::PointLight*>(av);
                // Mirror two engine filters that drop lights vanilla wouldn't
                // render either:
                //
                //   1. specular.r > 0 — Morrowind stores the modder-set Radius
                //      in NI::Light::specular.r (Bethesda overload). The
                //      engine's per-object selection at 0x4D2F40
                //      (game_dynamicLightTest) culls when
                //      `objectToLightDist - objectRadius > specular.r`. With
                //      radius=0 every non-overlapping object misses; the
                //      light is excluded from every effect list and never
                //      reaches SetLight.
                //
                //   2. affectedNodes is non-empty — the engine populates this
                //      bidirectional list when a light is attached to a
                //      reference's scene node (game_dynamicLightTest line
                //      122-127). MWSE's tes3reference:deleteDynamicLight-
                //      Attachment() (0x4E50F0) clears it via
                //      detachDynamicLightFromAffectedNodes(). The Midnight
                //      Oil mod's "turn lantern off" path goes through that
                //      MWSE call but does NOT detach the NiLight from its
                //      parent NiNode (default `removeLightFromParent=false`),
                //      so the light still appears in our walk despite being
                //      logically off. A light with empty affectedNodes is
                //      one the engine considers "not lighting anything";
                //      vanilla draw paths skip it.
                //
                // Filtering here saves a texLightData pool slot and a
                // per-mesh sphere-AABB test for a light that vanilla would
                // not have lit either. Visually no change; just stops
                // texture-light from rendering toggled-off lanterns.
                if (pl->specular.r > 0.0f
                    && !pl->affectedNodes.empty()) {
                    g_pointLights.push_back(extractPointLight(pl));
                }
            }

            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiDirectionalLight)) {
                auto* dl = static_cast<const NI::DirectionalLight*>(av);
                g_directionalLights.push_back(extractDirectionalLight(dl));
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

        // Walk every frame — no periodic / cell / child-count throttle.
        //
        // The previous 5-frame periodic was visibly aliasing NPC-held torches:
        // an NPC walking ~5 units/frame would be drawn at one position while
        // the torch's worldTransform.translation in our snapshot lagged 1-4
        // frames behind, producing a scatter of mis-lit ground around the
        // NPC. Cell-change and root-child-count short-circuits don't catch
        // intra-cell motion (NPC walking with a torch doesn't change either
        // signal). The fix is to walk every frame and detect change at the
        // POD level: byte-compare the fresh walk against the previous one
        // and only bump frameRevision when something actually moved. The
        // FFE consumer keys its texLightData re-upload on frameRevision
        // (plus its own view-matrix diff), so unchanged data still costs
        // zero re-upload.
        //
        // Walk cost is dominated by NI tree traversal + RTTI dispatch
        // (~200-300 µs/frame in dense interior scenes per the prior
        // measurement). The byte-compare is ~28 bytes/light at memcmp speed
        // — sub-microsecond for 86 lights.
        void rebuild() {
            // Swap last-frame's data out of each output vector, then walk
            // into now-empty vectors. Avoids any allocation in steady
            // state — both pairs keep capacity across frames.
            g_lastWalkedLights.swap(g_pointLights);
            g_pointLights.clear();
            g_lastWalkedDirectionalLights.swap(g_directionalLights);
            g_directionalLights.clear();

            walk(MGE::DataHandlerView::worldObjectRoot(g_dataHandler));
            walk(MGE::DataHandlerView::worldPickObjectRoot(g_dataHandler));
            // sgSunlight is a sibling root of the two NiNodes — the global
            // directional light. The walk's NiDirectionalLight RTTI arm
            // pushes it into g_directionalLights.
            if (auto sun = MGE::DataHandlerView::sgSunlight(g_dataHandler)) {
                walk(reinterpret_cast<NI::AVObject*>(sun));
            }

            const bool pointsChanged =
                g_pointLights.size() != g_lastWalkedLights.size()
                || (!g_pointLights.empty()
                    && std::memcmp(g_pointLights.data(),
                                   g_lastWalkedLights.data(),
                                   g_pointLights.size() * sizeof(PointLight)) != 0);
            const bool dirsChanged =
                g_directionalLights.size() != g_lastWalkedDirectionalLights.size()
                || (!g_directionalLights.empty()
                    && std::memcmp(g_directionalLights.data(),
                                   g_lastWalkedDirectionalLights.data(),
                                   g_directionalLights.size() * sizeof(DirectionalLight)) != 0);
            if (pointsChanged || dirsChanged) {
                ++g_frameRevision;
            }
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
            if (!g_pointLights.empty() || !g_directionalLights.empty()) {
                g_pointLights.clear();
                g_lastWalkedLights.clear();
                g_directionalLights.clear();
                g_lastWalkedDirectionalLights.clear();
                ++g_frameRevision;
            }
            return;
        }

        if (g_qpcFreq.QuadPart == 0) QueryPerformanceFrequency(&g_qpcFreq);

        const uint64_t prevRev = g_frameRevision;
        LARGE_INTEGER tsBegin, tsEnd;
        QueryPerformanceCounter(&tsBegin);
        rebuild();
        QueryPerformanceCounter(&tsEnd);

        const unsigned long long deltaTicks =
            static_cast<unsigned long long>(tsEnd.QuadPart - tsBegin.QuadPart);
        const unsigned long long deltaNs =
            (g_qpcFreq.QuadPart > 0)
                ? deltaTicks * 1000000000ULL / static_cast<unsigned long long>(g_qpcFreq.QuadPart)
                : 0ULL;
        ++g_walks;
        g_walksTotalNs += deltaNs;
        if (g_frameRevision != prevRev) ++g_walksWithChange;

        if (g_walks - g_lastReportWalks >= 1800) {
            const double avgNs       = (double)g_walksTotalNs / (double)g_walks;
            const double changeRate  = 100.0 * (double)g_walksWithChange / (double)g_walks;
            LOG::logline("-- [SCENEGRAPH] walks=%llu totalNs=%llu (avg=%.0fns) changes=%llu (%.0f%%) lights=%zu sun=%zu",
                g_walks,
                g_walksTotalNs,
                avgNs,
                g_walksWithChange,
                changeRate,
                g_pointLights.size(),
                g_directionalLights.size());
            g_lastReportWalks = g_walks;
        }
    }

    const std::vector<PointLight>&        pointLights()        { return g_pointLights; }
    const std::vector<DirectionalLight>&  directionalLights()  { return g_directionalLights; }
    uint64_t                              frameRevision()      { return g_frameRevision; }
    uint64_t                              frameCount()         { return g_walks; }

}
