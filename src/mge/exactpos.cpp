// ExactPos — see exactpos.h for why this exists and the rule it applies.

#include "mge_se_prelude.h"

#include "NIAVObject.h"
#include "NICamera.h"
#include "NINode.h"
#include "NITransform.h"

#include "exactpos.h"
#include "mwbridge.h"
#include "worldcontroller_view.h"
#include "support/log.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace MGE::ExactPos {

    namespace {

        // ---- mode -------------------------------------------------------------------------

        unsigned readMode() {
            char buf[16] = {};
            const DWORD n = GetEnvironmentVariableA("MGE_EXACT_POS", buf, sizeof(buf));
            unsigned m = kAll;
            if (n > 0 && n < sizeof(buf)) {
                char* end = nullptr;
                const unsigned long v = std::strtoul(buf, &end, 0);
                if (end != buf) m = static_cast<unsigned>(v) & kAll;
            }
            LOG::logline(">> [exactpos] MGE_EXACT_POS=%s -> mode %u (eye %s, rigid+lights %s, skinned %s, fp camera %s)",
                         n > 0 ? buf : "(unset)", m,
                         (m & kEye) ? "EXACT" : "float", (m & kRigid) ? "EXACT" : "float",
                         (m & kSkinned) ? "EXACT" : "float", (m & kFP) ? "EXACT" : "float");
            return m;
        }

        // ---- per-frame, per-thread memo ----------------------------------------------------
        //
        // Open addressing keyed on the node pointer. Retiring a frame is O(1): bump the
        // generation, and every slot stamped with an older one reads as empty. A hit must also
        // match the node's stored translation bit for bit, so a node MW moves between two
        // queries in the same frame is recomposed instead of served stale.

        std::atomic<uint32_t> g_frameId{ 1 };

        struct Slot {
            const NI::AVObject* key;
            uint32_t gen;
            float    stored[3];
            double   exact[3];
        };

        struct Memo {
            std::vector<Slot> slots;
            uint32_t mask = 0;
            uint32_t gen = 0;
            uint32_t used = 0;
            uint32_t frame = 0;

            static uint32_t hash(const NI::AVObject* p) {
                return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p) >> 4) * 2654435761u;
            }

            void retireIfStale() {
                const uint32_t f = g_frameId.load(std::memory_order_relaxed);
                if (frame == f && !slots.empty()) return;
                frame = f;
                if (slots.empty()) {
                    slots.assign(4096, Slot{});
                    mask = 4095;
                }
                if (++gen == 0) {                    // wrapped: stale stamps could alias gen 0
                    for (Slot& s : slots) s.gen = 0;
                    gen = 1;
                }
                used = 0;
            }

            Slot* find(const NI::AVObject* key) {
                for (uint32_t i = hash(key) & mask;; i = (i + 1) & mask) {
                    Slot& s = slots[i];
                    if (s.gen != gen) return nullptr;
                    if (s.key == key) return &s;
                }
            }

            void grow() {
                std::vector<Slot> old;
                old.swap(slots);
                slots.assign(old.size() * 2, Slot{});
                mask = static_cast<uint32_t>(slots.size()) - 1;
                const uint32_t g = gen;
                gen = 1;
                used = 0;
                for (const Slot& s : old) {
                    if (s.gen == g) insert(s.key, s.stored, s.exact);
                }
            }

            void insert(const NI::AVObject* key, const float stored[3], const double exact[3]) {
                if ((used + 1) * 2 > slots.size()) grow();
                for (uint32_t i = hash(key) & mask;; i = (i + 1) & mask) {
                    Slot& s = slots[i];
                    if (s.gen != gen || s.key == key) {
                        if (s.gen != gen) ++used;
                        s.key = key;
                        s.gen = gen;
                        std::memcpy(s.stored, stored, sizeof(s.stored));
                        std::memcpy(s.exact, exact, sizeof(s.exact));
                        return;
                    }
                }
            }
        };

        thread_local Memo t_memo;

        // ---- counters (any thread) ---------------------------------------------------------

        std::atomic<uint32_t> g_composed{ 0 };
        std::atomic<uint32_t> g_anchored{ 0 };

        // Max of non-negative floats kept as their bit patterns: for x >= 0 the IEEE ordering
        // and the unsigned ordering agree, so a CAS on the bits is a lock-free float max.
        void atomicMax(std::atomic<uint32_t>& a, float v) {
            uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            uint32_t cur = a.load(std::memory_order_relaxed);
            while (bits > cur && !a.compare_exchange_weak(cur, bits, std::memory_order_relaxed)) {}
        }
        float takeMax(std::atomic<uint32_t>& a) {
            const uint32_t bits = a.exchange(0, std::memory_order_relaxed);
            float v;
            std::memcpy(&v, &bits, sizeof(v));
            return v;
        }

        // Which nodes anchor, named once each (first 12) — the E0 question "what does MW set by
        // copy?" answered from the log instead of from guesses.
        std::mutex g_anchorLogMx;
        std::vector<const NI::AVObject*> g_anchorLogged;

        void noteAnchor(const NI::AVObject* n, const float stored[3], const double exact[3]) {
            std::lock_guard<std::mutex> lk(g_anchorLogMx);
            if (g_anchorLogged.size() >= 12) return;
            for (const auto* p : g_anchorLogged) if (p == n) return;
            g_anchorLogged.push_back(n);
            const char* nm = n->name ? n->name : "(unnamed)";
            const char* pn = (n->parentNode && n->parentNode->name) ? n->parentNode->name : "(unnamed)";
            LOG::logline(">> [exactpos] anchored '%s' (parent '%s'): stored (%.4f %.4f %.4f) vs composed "
                         "(%.4f %.4f %.4f) — d=(%.4f %.4f %.4f)",
                         nm, pn, stored[0], stored[1], stored[2], exact[0], exact[1], exact[2],
                         exact[0] - stored[0], exact[1] - stored[1], exact[2] - stored[2]);
        }

        // 16 float ulps at the largest component's magnitude — G7's per-level tolerance. MW's own
        // float composition stays well inside it (a few half-ulp roundings per level); a node
        // placed by any other rule does not.
        double tolerance(const float s[3]) {
            float m = std::fabs(s[0]);
            if (std::fabs(s[1]) > m) m = std::fabs(s[1]);
            if (std::fabs(s[2]) > m) m = std::fabs(s[2]);
            const float ulp = std::nextafter(m, std::numeric_limits<float>::max()) - m;
            return 16.0 * static_cast<double>(ulp);
        }

        void compose(const NI::AVObject* n, double out[3]) {
            const NI::Point3& st = n->worldTransform.translation;
            const float stored[3] = { st.x, st.y, st.z };

            Memo& memo = t_memo;
            if (const Slot* s = memo.find(n)) {
                if (std::memcmp(s->stored, stored, sizeof(stored)) == 0) {
                    out[0] = s->exact[0]; out[1] = s->exact[1]; out[2] = s->exact[2];
                    return;
                }
            }

            const NI::AVObject* p = n->parentNode;
            if (!p) {
                // A root: its stored translation IS its local one, and exact by definition.
                out[0] = stored[0]; out[1] = stored[1]; out[2] = stored[2];
            } else {
                double pt[3];
                compose(p, pt);
                const NI::Matrix33& R = p->worldTransform.rotation;
                const double sc = p->worldTransform.scale;
                const double lx = sc * n->localTranslate.x;
                const double ly = sc * n->localTranslate.y;
                const double lz = sc * n->localTranslate.z;
                out[0] = pt[0] + R.m0.x * lx + R.m0.y * ly + R.m0.z * lz;
                out[1] = pt[1] + R.m1.x * lx + R.m1.y * ly + R.m1.z * lz;
                out[2] = pt[2] + R.m2.x * lx + R.m2.y * ly + R.m2.z * lz;

                const double tol = tolerance(stored);
                if (std::fabs(out[0] - stored[0]) > tol || std::fabs(out[1] - stored[1]) > tol
                    || std::fabs(out[2] - stored[2]) > tol) {
                    noteAnchor(n, stored, out);
                    out[0] = stored[0]; out[1] = stored[1]; out[2] = stored[2];
                    g_anchored.fetch_add(1, std::memory_order_relaxed);
                } else {
                    g_composed.fetch_add(1, std::memory_order_relaxed);
                }
            }
            memo.insert(n, stored, out);
        }

        // ---- eye ----------------------------------------------------------------------------

        double g_eyeExact[3] = {};     // truth, every mode (the diagnostic's reference)
        double g_eyeShip[3]  = {};     // what the payload subtracts (exact under kEye)

        // Beyond this the "exact" eye disagrees with the view MW is rendering with by more than
        // any float rounding can explain: the node is not the render camera this frame (vanity,
        // a scripted camera). Fall back to the view inverse rather than shift the scene.
        constexpr double kEyeSanity = 4.0;

        // ---- arm camera ---------------------------------------------------------------------

        std::mutex g_armMx;
        bool   g_armValid = false;
        float  g_armStored[3] = {};
        double g_armExact[3]  = {};

        // ---- diagnostic accumulators --------------------------------------------------------

        struct Window {
            uint32_t frames = 0;
            uint32_t headHits = 0;         // exact source engaged (first person, bit-equal camera)
            uint32_t headSeen = 0;         // first-person frames with a head camera at all
            uint32_t eyeRejects = 0;       // sanity fallback fired
            double   maxViewInvErr = 0.0;  // |viewInv - exact|
            double   maxCamErr = 0.0;      // |camFloat - exact|
            double   maxCamHead = 0.0;     // |camFloat - head stored| (0 = MW copied the head)
            double   maxCamHeadExact = 0.0;// |camFloat - head exact| (~ulp = copy + one rounding)
        } g_win;

        std::atomic<uint32_t> g_shipMax[2][2];     // [fp][cls] max error (float bits)
        std::atomic<uint32_t> g_shipCount[2][2];   // samples inside 500 units
        std::atomic<uint32_t> g_fpCamMisses{ 0 };  // arm camera moved between walk and build

        thread_local bool   t_fpCamValid = false;
        thread_local float  t_fpCamShipped[3] = {};
        thread_local double t_fpCamExact[3] = {};

        std::chrono::steady_clock::time_point g_lastLog;

        double dist3(const double a[3], const double b[3]) {
            const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
            return std::sqrt(dx * dx + dy * dy + dz * dz);
        }

        void logWindow() {
            const auto now = std::chrono::steady_clock::now();
            if (now - g_lastLog < std::chrono::seconds(1)) return;
            g_lastLog = now;
            const uint32_t composed = g_composed.exchange(0, std::memory_order_relaxed);
            const uint32_t anchored = g_anchored.exchange(0, std::memory_order_relaxed);
            float mx[2][2];
            uint32_t cnt[2][2];
            for (int f = 0; f < 2; ++f) for (int c = 0; c < 2; ++c) {
                mx[f][c] = takeMax(g_shipMax[f][c]);
                cnt[f][c] = g_shipCount[f][c].exchange(0, std::memory_order_relaxed);
            }
            const uint32_t fpMiss = g_fpCamMisses.exchange(0, std::memory_order_relaxed);
            const Window w = g_win;
            g_win = Window{};
            if (w.frames == 0) return;
            LOG::logline(">> [exactpos] mode=%u frames=%u eye(%.1f %.1f %.1f) | head src %u/%u 1st-person "
                         "frames, sanity rejects %u | max |viewInv-exact|=%.4f |camFloat-exact|=%.4f "
                         "|cam-headStored|=%.4f |cam-headExact|=%.4f | nodes/frame composed %.0f anchored %.1f",
                         mode(), w.frames, g_eyeExact[0], g_eyeExact[1], g_eyeExact[2],
                         w.headHits, w.headSeen, w.eyeRejects, w.maxViewInvErr, w.maxCamErr,
                         w.maxCamHead, w.maxCamHeadExact,
                         composed / (double)w.frames, anchored / (double)w.frames);
            LOG::logline(">> [exactpos] shipped-vs-exact within 500u: rigid %.5f (%u) skinned %.5f (%u) | "
                         "fp rigid %.5f (%u) fp skinned %.5f (%u) | fp cam misses %u",
                         mx[0][0], cnt[0][0], mx[0][1], cnt[0][1], mx[1][0], cnt[1][0], mx[1][1], cnt[1][1],
                         fpMiss);
        }
    }

    unsigned mode() {
        static const unsigned s_mode = readMode();
        return s_mode;
    }

    void worldT(const NI::AVObject* node, double out[3]) {
        t_memo.retireIfStale();
        compose(node, out);
    }

    void composeBone(const NI::AVObject* bone, const NI::Transform& offset, double out[3]) {
        double bt[3];
        worldT(bone, bt);
        const NI::Matrix33& R = bone->worldTransform.rotation;
        const double sc = bone->worldTransform.scale;
        const double lx = sc * offset.translation.x;
        const double ly = sc * offset.translation.y;
        const double lz = sc * offset.translation.z;
        out[0] = bt[0] + R.m0.x * lx + R.m0.y * ly + R.m0.z * lz;
        out[1] = bt[1] + R.m1.x * lx + R.m1.y * ly + R.m1.z * lz;
        out[2] = bt[2] + R.m2.x * lx + R.m2.y * ly + R.m2.z * lz;
    }

    void beginFrame(const float viewInvEye[3]) {
        mode();   // log the mode once, early
        g_frameId.fetch_add(1, std::memory_order_relaxed);
        t_memo.retireIfStale();

        const double viewInv[3] = { viewInvEye[0], viewInvEye[1], viewInvEye[2] };
        double exact[3] = { viewInv[0], viewInv[1], viewInv[2] };

        NI::Camera* cam = MGE::WorldControllerView::worldCamera();
        if (cam) {
            const NI::Point3& ct = cam->worldTransform.translation;
            const double camFloat[3] = { ct.x, ct.y, ct.z };
            auto* bridge = MWBridge::get();
            NI::Camera* head = bridge->is3rdPerson() ? nullptr : bridge->getPlayerHeadCamera();
            if (head) {
                ++g_win.headSeen;
                const NI::Point3& hs = head->worldTransform.translation;
                const double hsd[3] = { hs.x, hs.y, hs.z };
                double hx[3];
                worldT(head, hx);
                const double a = dist3(camFloat, hsd), b = dist3(camFloat, hx);
                if (a > g_win.maxCamHead) g_win.maxCamHead = a;
                if (b > g_win.maxCamHeadExact) g_win.maxCamHeadExact = b;
            }
            // The exact source applies only when MW verifiably copied the camera from the head
            // node this frame: bit-identical stored translations. Then eye = camFloat + (exact
            // head - stored head), i.e. the head's exact composition.
            if (head && std::memcmp(&head->worldTransform.translation, &ct, sizeof(NI::Point3)) == 0) {
                double h[3];
                worldT(head, h);
                const NI::Point3& hs = head->worldTransform.translation;
                exact[0] = camFloat[0] + (h[0] - hs.x);
                exact[1] = camFloat[1] + (h[1] - hs.y);
                exact[2] = camFloat[2] + (h[2] - hs.z);
                ++g_win.headHits;
            } else {
                worldT(cam, exact);
            }
            if (dist3(exact, viewInv) > kEyeSanity) {
                exact[0] = viewInv[0]; exact[1] = viewInv[1]; exact[2] = viewInv[2];
                ++g_win.eyeRejects;
            }
            const double ce = dist3(camFloat, exact);
            if (ce > g_win.maxCamErr) g_win.maxCamErr = ce;
        }
        const double ve = dist3(viewInv, exact);
        if (ve > g_win.maxViewInvErr) g_win.maxViewInvErr = ve;
        ++g_win.frames;

        std::memcpy(g_eyeExact, exact, sizeof(g_eyeExact));
        if (on(kEye)) std::memcpy(g_eyeShip, exact, sizeof(g_eyeShip));
        else          std::memcpy(g_eyeShip, viewInv, sizeof(g_eyeShip));

        logWindow();
    }

    const double* eye() {
        return g_eyeShip;
    }

    void captureArmCamera(const NI::AVObject* armCamera) {
        std::lock_guard<std::mutex> lk(g_armMx);
        g_armValid = (armCamera != nullptr);
        if (!armCamera) return;
        const NI::Point3& t = armCamera->worldTransform.translation;
        g_armStored[0] = t.x; g_armStored[1] = t.y; g_armStored[2] = t.z;
        worldT(armCamera, g_armExact);
    }

    bool armCameraExact(const float storedPos[3], double out[3]) {
        std::lock_guard<std::mutex> lk(g_armMx);
        if (!g_armValid || std::memcmp(storedPos, g_armStored, sizeof(g_armStored)) != 0) {
            g_fpCamMisses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        out[0] = g_armExact[0]; out[1] = g_armExact[1]; out[2] = g_armExact[2];
        return true;
    }

    void setFPCamera(const float shippedRel[3], const double exactT[3]) {
        t_fpCamValid = (exactT != nullptr);
        if (!exactT) return;
        std::memcpy(t_fpCamShipped, shippedRel, sizeof(t_fpCamShipped));
        std::memcpy(t_fpCamExact, exactT, sizeof(t_fpCamExact));
    }

    void noteShipped(int cls, bool fp, const float shippedRel[3], const double exactT[3]) {
        double truth[3], shipped[3];
        if (fp) {
            // An arm part is drawn under the arm camera, so what matters is its position
            // RELATIVE TO THAT CAMERA — a common offset of both cancels in the FP view.
            if (!t_fpCamValid) return;
            for (int i = 0; i < 3; ++i) {
                truth[i]   = exactT[i] - t_fpCamExact[i];
                shipped[i] = static_cast<double>(shippedRel[i]) - t_fpCamShipped[i];
            }
        } else {
            for (int i = 0; i < 3; ++i) {
                truth[i]   = exactT[i] - g_eyeExact[i];
                shipped[i] = shippedRel[i];
            }
        }
        if (truth[0] * truth[0] + truth[1] * truth[1] + truth[2] * truth[2] > 500.0 * 500.0) return;
        // Accumulated per thread and folded into the shared atomics every 1024 samples: a
        // skinned crowd is thousands of bones a frame, too many for an atomic each.
        thread_local float    t_max[2][2] = {};
        thread_local uint32_t t_cnt[2][2] = {};
        thread_local uint32_t t_total = 0;
        const int f = fp ? 1 : 0;
        const float err = static_cast<float>(dist3(truth, shipped));
        if (err > t_max[f][cls]) t_max[f][cls] = err;
        ++t_cnt[f][cls];
        if (++t_total >= 1024) {
            t_total = 0;
            for (int a = 0; a < 2; ++a) for (int c = 0; c < 2; ++c) {
                if (t_cnt[a][c] == 0) continue;
                atomicMax(g_shipMax[a][c], t_max[a][c]);
                g_shipCount[a][c].fetch_add(t_cnt[a][c], std::memory_order_relaxed);
                t_max[a][c] = 0.0f;
                t_cnt[a][c] = 0;
            }
        }
    }
}
