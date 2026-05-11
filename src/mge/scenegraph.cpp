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

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>

#include "configuration.h"
#include "datahandler_view.h"
#include "scenegraph.h"
#include "support/log.h"

namespace MGE::SceneGraph {

    namespace {
        // Bridge state.
        void*    g_dataHandler   = nullptr;
        uint64_t g_frameRevision = 0;

        //
        // Public output buffers — consumers read these. Worker swaps them
        // atomically under g_snapshotMtx; sync rebuild writes directly.
        //
        std::vector<PointLight>       g_pointLights;
        std::vector<DirectionalLight> g_directionalLights;

        //
        // Worker write buffers. Worker writes into these while walking;
        // on completion, swaps with the public buffers under
        // g_snapshotMtx. Empty / unused in sync mode.
        //
        std::vector<PointLight>       g_pointLightsW;
        std::vector<DirectionalLight> g_directionalLightsW;

        //
        // Last-walked snapshot (sync mode only). Used for the byte-compare
        // change-detection that gates frameRevision bumps. Async mode
        // always bumps revision per walk — see rebuildAsync().
        //
        std::vector<PointLight>       g_lastWalkedLights;
        std::vector<DirectionalLight> g_lastWalkedDirectionalLights;

        //
        // Walk targets — set by the rebuild dispatcher, read by walk()
        // during recursion. Single-thread-by-construction: the thread
        // running walk() sets these immediately before invoking it and
        // reads them through the recursion. Sync rebuild sets them on
        // main; async worker sets them on the worker. No concurrent
        // access from any other thread.
        //
        std::vector<PointLight>*       gw_pointLights       = nullptr;
        std::vector<DirectionalLight>* gw_directionalLights = nullptr;

        //
        // Snapshot lock. Always taken (uncontended in sync mode, briefly
        // contended in async mode). Held by:
        //   - worker during the publish swap (~microseconds)
        //   - sync rebuild during its writes (~hundreds of microseconds,
        //     irrelevant since the only "consumer" on the same thread is
        //     past the lock by then)
        //   - consumers via SnapshotReadLock RAII helper, for the duration
        //     of their reads of pointLights() / directionalLights()
        //     (~microseconds typical)
        //
        std::mutex g_snapshotMtx;

        //
        // Worker thread infrastructure. Lazy-spawned on first signal
        // when UseAsyncSceneGraphWalk is on; stays parked on the
        // condvar otherwise. Never explicitly joined — leaked at process
        // exit, which is fine for a render plugin (no shared resources
        // to leak; OS reclaims thread on process death).
        //
        std::thread             g_workerThread;
        std::mutex              g_workerSignalMtx;
        std::condition_variable g_workerCv;
        int                     g_workerPending = 0;     // collapsed walk requests
        bool                    g_workerStop    = false; // never set today
        bool                    g_workerStarted = false; // lazy-init flag

        //
        // Walk-cost instrumentation. Cumulative since process start.
        // Periodic dump every 1800 walks (~30 sec at 60 FPS).
        //
        LARGE_INTEGER       g_qpcFreq         = {};
        unsigned long long  g_walks           = 0;
        unsigned long long  g_walksTotalNs    = 0;
        unsigned long long  g_walksWithChange = 0;
        unsigned long long  g_lastReportWalks = 0;

        //
        // Extraction helpers — pure POD copies from NI memory.
        //

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

            out.radius = pl->specular.r;
            return out;
        }

        DirectionalLight extractDirectionalLight(const NI::DirectionalLight* dl) {
            DirectionalLight out{};

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

        //
        // Recursive scene-graph descent. Reads from gw_* destination
        // pointers set by the rebuild dispatcher. Skips AppCulled
        // subtrees.
        //
        void walk(NI::AVObject* av) {
            if (!av) return;
            if (av->getAppCulled()) return;

            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiPointLight)) {
                auto* pl = static_cast<const NI::PointLight*>(av);
                // Mirror two engine filters that drop lights vanilla
                // wouldn't render either:
                //   1. specular.r > 0 — Bethesda overload of NI::Light::
                //      specular.r as the modder-set fade radius. The
                //      engine's per-object selection at 0x4D2F40 culls
                //      lights with radius=0 from every object.
                //   2. affectedNodes non-empty — engine populates this
                //      bidirectional list when a light is attached to a
                //      reference's scene node. MWSE's
                //      tes3reference:deleteDynamicLightAttachment()
                //      clears it without detaching the NiLight, leaving
                //      logically-off lanterns visible to our walk.
                if (pl->specular.r > 0.0f && !pl->affectedNodes.empty()) {
                    gw_pointLights->push_back(extractPointLight(pl));
                }
            }

            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiDirectionalLight)) {
                auto* dl = static_cast<const NI::DirectionalLight*>(av);
                gw_directionalLights->push_back(extractDirectionalLight(dl));
            }

            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) {
                auto node = static_cast<NI::Node*>(av);
                // Iterate [0, endIndex), null-skip in walk(). NITArray is
                // sparse — filledCount counts non-null entries, but they
                // may sit at non-contiguous slots up to endIndex. Using
                // filledCount as the loop bound silently drops any light
                // whose slot index is >= filledCount.
                const auto count = node->children.getEndIndex();
                for (size_t i = 0; i < count; ++i) {
                    walk(node->children.at(i));
                }
            }
        }

        //
        // Walk into the destination buffers from the three scene roots.
        // Caller has already set gw_* targets and prepped the
        // destination vectors (cleared / swapped as needed).
        //
        void runWalk() {
            walk(MGE::DataHandlerView::worldObjectRoot(g_dataHandler));
            walk(MGE::DataHandlerView::worldPickObjectRoot(g_dataHandler));
            // sgSunlight is a sibling root of the two NiNodes — the
            // global directional light. The walk's NiDirectionalLight
            // RTTI arm pushes it into gw_directionalLights.
            if (auto sun = MGE::DataHandlerView::sgSunlight(g_dataHandler)) {
                walk(reinterpret_cast<NI::AVObject*>(sun));
            }
        }

        //
        // Synchronous rebuild — runs on the main thread when async is
        // off. Writes directly to the public buffers, with the byte-
        // compare change detection that gates frameRevision bumps. Lock
        // is uncontended (only this thread touches the public buffers
        // in sync mode).
        //
        void rebuildSync() {
            std::lock_guard<std::mutex> lk(g_snapshotMtx);

            g_lastWalkedLights.swap(g_pointLights);
            g_pointLights.clear();
            g_lastWalkedDirectionalLights.swap(g_directionalLights);
            g_directionalLights.clear();

            gw_pointLights       = &g_pointLights;
            gw_directionalLights = &g_directionalLights;

            runWalk();

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

        //
        // Async rebuild — runs on the worker thread. Writes into private
        // W buffers (no lock needed during walk; no other thread reads
        // them). On completion, takes the snapshot lock and swaps W
        // buffers with public. Always bumps revision (no change
        // detection — would require extra copy or third buffer; the
        // FFE consumer's per-frame re-upload is cheaper than the copy).
        //
        void rebuildAsync() {
            g_pointLightsW.clear();
            g_directionalLightsW.clear();

            gw_pointLights       = &g_pointLightsW;
            gw_directionalLights = &g_directionalLightsW;

            runWalk();

            std::lock_guard<std::mutex> lk(g_snapshotMtx);
            g_pointLights.swap(g_pointLightsW);
            g_directionalLights.swap(g_directionalLightsW);
            ++g_frameRevision;
        }

        void workerLoop() {
            while (true) {
                {
                    std::unique_lock<std::mutex> lk(g_workerSignalMtx);
                    g_workerCv.wait(lk, []{
                        return g_workerStop || g_workerPending > 0;
                    });
                    if (g_workerStop) return;
                    g_workerPending = 0; // collapse multiple signals into one walk
                }

                if (g_qpcFreq.QuadPart == 0) QueryPerformanceFrequency(&g_qpcFreq);
                LARGE_INTEGER tsBegin, tsEnd;
                QueryPerformanceCounter(&tsBegin);

                rebuildAsync();

                QueryPerformanceCounter(&tsEnd);
                const unsigned long long deltaTicks =
                    static_cast<unsigned long long>(tsEnd.QuadPart - tsBegin.QuadPart);
                const unsigned long long deltaNs =
                    (g_qpcFreq.QuadPart > 0)
                        ? deltaTicks * 1000000000ULL / static_cast<unsigned long long>(g_qpcFreq.QuadPart)
                        : 0ULL;
                ++g_walks;
                g_walksTotalNs += deltaNs;
                ++g_walksWithChange; // async always bumps revision

                if (g_walks - g_lastReportWalks >= 1800) {
                    const double avgNs      = (double)g_walksTotalNs / (double)g_walks;
                    const double changeRate = 100.0 * (double)g_walksWithChange / (double)g_walks;
                    LOG::logline("-- [SCENEGRAPH async] walks=%llu totalNs=%llu (avg=%.0fns) changes=%llu (%.0f%%) lights=%zu sun=%zu",
                        g_walks, g_walksTotalNs, avgNs, g_walksWithChange, changeRate,
                        g_pointLights.size(), g_directionalLights.size());
                    g_lastReportWalks = g_walks;
                }
            }
        }

        void ensureWorker() {
            if (!g_workerStarted) {
                g_workerStarted = true;
                g_workerThread = std::thread(workerLoop);
                LOG::logline("-- [SCENEGRAPH] async worker spawned");
            }
        }

        void signalWorker() {
            {
                std::lock_guard<std::mutex> lk(g_workerSignalMtx);
                ++g_workerPending;
            }
            g_workerCv.notify_one();
        }
    }

    void* getDataHandler() {
        // Self-source from the engine global at 0x7C67E0 (the same address
        // TES3::DataHandler::get() reads on the MWSE side). Lazy: returns
        // null on early frames before the engine has constructed the
        // singleton; subsequent calls retry until the read returns
        // non-null, at which point we cache and log once.
        //
        // Previously this pointer was pushed in by MWSE through
        // MGEAPIv4::setDataHandler. MWSE dropped that ABI on the
        // sharedse-ni-unification branch (commit d2a92c596d), so we
        // resolve it ourselves to remove the cross-DLL handoff.
        if (!g_dataHandler) {
            void* dh = *reinterpret_cast<void**>(0x7C67E0);
            if (dh) {
                g_dataHandler = dh;
                LOG::logline("-- [SCENEGRAPH] DataHandler resolved from engine global 0x7C67E0: %p", dh);
            }
        }
        return g_dataHandler;
    }

    void onFrameReady() {
        if (!getDataHandler()) return;

        if (!Configuration.UseSceneGraphSnapshot) {
            // Disabled: drop any lingering data so flipping the flag off
            // mid-session releases the snapshot promptly. Lock taken so
            // a worker spawned earlier (and still running its loop) can't
            // race against this clear.
            std::lock_guard<std::mutex> lk(g_snapshotMtx);
            if (!g_pointLights.empty() || !g_directionalLights.empty()) {
                g_pointLights.clear();
                g_lastWalkedLights.clear();
                g_directionalLights.clear();
                g_lastWalkedDirectionalLights.clear();
                g_pointLightsW.clear();
                g_directionalLightsW.clear();
                ++g_frameRevision;
            }
            return;
        }

        // Async path: signal the worker, return immediately. Walk runs
        // concurrent with engine render. Snapshot becomes available 1
        // frame stale (by next onFrameReady at the latest).
        if (Configuration.UseAsyncSceneGraphWalk) {
            ensureWorker();
            signalWorker();
            return;
        }

        // Synchronous path. Walk on the main thread; consumer reads see
        // fresh data on the same frame.
        if (g_qpcFreq.QuadPart == 0) QueryPerformanceFrequency(&g_qpcFreq);

        const uint64_t prevRev = g_frameRevision;
        LARGE_INTEGER tsBegin, tsEnd;
        QueryPerformanceCounter(&tsBegin);
        rebuildSync();
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
            const double avgNs      = (double)g_walksTotalNs / (double)g_walks;
            const double changeRate = 100.0 * (double)g_walksWithChange / (double)g_walks;
            LOG::logline("-- [SCENEGRAPH sync] walks=%llu totalNs=%llu (avg=%.0fns) changes=%llu (%.0f%%) lights=%zu sun=%zu",
                g_walks, g_walksTotalNs, avgNs, g_walksWithChange, changeRate,
                g_pointLights.size(), g_directionalLights.size());
            g_lastReportWalks = g_walks;
        }
    }

    const std::vector<PointLight>&        pointLights()        { return g_pointLights; }
    const std::vector<DirectionalLight>&  directionalLights()  { return g_directionalLights; }
    uint64_t                              frameRevision()      { return g_frameRevision; }
    uint64_t                              frameCount()         { return g_walks; }

    void lockSnapshot()   { g_snapshotMtx.lock(); }
    void unlockSnapshot() { g_snapshotMtx.unlock(); }

}
