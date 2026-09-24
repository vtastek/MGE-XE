#pragma once

// ExactPos — double-precision world translations and eye for the Forge path.
//
// Far from the origin (Dragonstar East, x ~ -921,700) a float32 holds a world position to
// 0.0625 units, and MW composes every node's world transform in float. The Forge payload is
// camera-relative, but it used to subtract an already-rounded eye from already-rounded
// translations, so each kept up to ~0.03-0.06 units of error — 1-3 px at arm's length, changing
// frame to frame: NPCs, arms and nearby props visibly shook. Worse, the eye was recovered by
// inverting MW's float view matrix, so its error depended on camera ROTATION and the whole scene
// shifted as you turned.
//
// The fix recomposes world translations in double from the scene-graph parent chain. Each node's
// local translation is small and exact; only MW's float composition into world rounds. Rotation
// error is relative, not magnitude-dependent, so the parent's stored float rotation/scale are
// used as-is. Every level is VALIDATED: a node whose stored world translation is further than
// 16 float ulps from the standard composition is not composed by the standard rule (billboards,
// roots set by copy, controllers writing world directly), so it is ANCHORED to its stored value
// and its descendants continue exactly from there. Fails closed — never worse than float.
//
// No hooks and no addresses beyond what SharedSE/MWSE provide (PRIME DIRECTIVE 6). The host needs
// no change: it already receives camera-relative translations.
//
// MGE_EXACT_POS (env, read once) selects which consumers take the exact values — a bitmask so one
// build can A/B each phase: 1 eye, 2 rigid entries + point lights, 4 skinned palettes, 8 the
// first-person arm camera. Unset = all on (15); 0 = the float path, byte-identical to before.
// The [exactpos] diagnostic runs in every mode, so mode 0 measures the error being fixed.
//
// Threading: worldT() may be called from the main thread (cache walk) and from the produce worker
// (ensureLive) or the scene-graph light worker. Its memo is thread-local and retired per frame,
// so there is no shared mutable state on that path.

#include <cstdint>

namespace NI {
    struct AVObject;
    struct Transform;
}

namespace MGE::ExactPos {

    enum : unsigned {
        kEye     = 1u,
        kRigid   = 2u,   // rigid cache entries (static/multimap/alpha) + point lights
        kSkinned = 4u,   // skinned bone palettes
        kFP      = 8u,   // first-person arm camera
        kAll     = 15u,
    };

    unsigned mode();
    inline bool on(unsigned bit) { return (mode() & bit) != 0; }

    // Main thread, once per frame from DistantLand::setView (the scene graph is posed for the
    // frame there). Retires every thread's memo, resolves this frame's eye, and emits the
    // once-a-second [exactpos] diagnostic. `viewInvEye` = the eye recovered from MW's view
    // matrix (the pre-ExactPos source), kept for the A/B and the diagnostic.
    //
    // The eye is worldT(world camera). A head-node source (PlayerAnimationController::
    // firstPersonHeadCameraNode, corrected by its own rounding) engages only when the camera's
    // stored translation equals the head's bit for bit — measured 2026-09-24 it NEVER does: the
    // camera sits ~1e-3 from the head standing still and up to ~0.3 away mid-turn, so it is not a
    // copy. The camera chain composes to its own stored value exactly, so the eye keeps MW's own
    // camera rounding (<= half an ulp per axis) and loses the view-inverse's rotation-coupled error.
    void beginFrame(const float viewInvEye[3]);

    // The eye every camera-relative subtraction uses this frame: the exact eye under kEye, else
    // the view-inverse eye widened to double (so mode 0 reproduces the float path bit for bit).
    const double* eye();

    // Exact world translation of a node (see the header comment for the rule and the anchor).
    void worldT(const NI::AVObject* node, double out[3]);

    // Exact translation of (bone world) * offset — NI::Transform::operator* in double:
    // worldT(bone) + R_bone · (s_bone · offset.translation).
    void composeBone(const NI::AVObject* bone, const NI::Transform& offset, double out[3]);

    // Camera-relative translation for the wire: float(T - eye()), where T is the exact
    // translation when `exact` is set, else the stored float one widened. Mode 0 == float(a - b).
    inline void rel(float out[3], const float stored[3], const double exact[3], bool useExact) {
        const double* e = eye();
        for (int i = 0; i < 3; ++i) {
            out[i] = static_cast<float>((useExact ? exact[i] : static_cast<double>(stored[i])) - e[i]);
        }
    }

    // First-person arm camera. captureArmCamera runs on main right after the FP walk (the same
    // instant the arm parts are read); armCameraExact is asked by buildFPFrame on the worker with
    // the position it read, and answers with the exact world position only if that position is
    // bit-identical to the one captured — otherwise false (keep the float path).
    void captureArmCamera(const NI::AVObject* armCamera);
    bool armCameraExact(const float storedPos[3], double out[3]);

    // Diagnostic: what was shipped vs the exact camera-relative truth (exact T - exact eye).
    // Called from the emitters; only |exactRel| < 500 counts. cls: 0 rigid, 1 skinned.
    // First-person parts are judged against the arm camera (setFPCamera, same thread, first).
    enum ShipClass { kShipRigid = 0, kShipSkinned = 1 };
    void noteShipped(int cls, bool fp, const float shippedRel[3], const double exactT[3]);
    void setFPCamera(const float shippedRel[3], const double exactT[3]);
}
