// mgeHost64 — the C++/FSL Hosek-Wilkie CROSS-CHECK (tasks/forge-physical-sky.md P2, step 3).
//
// The physical sky evaluates ONE model TWICE: Hosek::evalNative() in C++ feeds the SH and therefore
// the light, hosekRadiance() in hosek.h.fsl feeds the pixels. That is the documented fourth-copy
// drift hazard, and the failure it produces is the exact thing the whole step exists to remove — a
// sky whose colour and whose light have quietly stopped agreeing. A comment is not a mitigation for
// that, so this is: one dispatch at startup evaluates the SHADER form at a fixed direction set, the
// host evaluates the C++ form at the same set, and the max relative deviation is logged.
//
// It runs ONCE, costs 64 threads, and is the first thing on the log that anything else here can be
// believed against. `[forge-hosek] cross-check` — see forgerender.cpp.
#pragma once

#include "skyview.h.fsl"   // SkyViewData — the SAME cbuffer the sky pass reads, deliberately

BEGIN_SRT(HosekCheckSrtData)
    BEGIN_SRT_SET(PerBatch)
        // ⚠ THE SKY PASS'S OWN CBUFFER, bound here as well rather than a copy filled for the test.
        // A check that evaluated its own coefficients would prove the ARITHMETIC matches and say
        // nothing about whether the coefficients reaching the frag are the ones the host cooked —
        // which is half of what can go wrong (a mis-sized struct, a lane written at the wrong
        // offset, an unbound set). Sharing the buffer folds both questions into one number.
        DECL_CBUFFER(PerBatch, CBUFFER(SkyViewData), gSkyView)
        // The directions to evaluate, xyz = a unit world direction, w unused. One float4 per thread.
        DECL_BUFFER(PerBatch, Buffer(float4), gHosekDirs)
        // ...and where the answers go: 4 uints per direction — asuint(R), asuint(G), asuint(B), and
        // a literal 1 as a "this thread ran" tell, so a dispatch that silently did nothing stays
        // distinguishable from one that agreed. uint element type + asuint() rather than a typed
        // float UAV, matching gAplOut and gInstOut: the merged compute rootsig has no float-typed
        // RWBuffer and a self-test is not the place to be the first.
        DECL_RWBUFFER(PerBatch, RWBuffer(uint), gHosekOut)
    END_SRT_SET(PerBatch)
END_SRT(HosekCheckSrtData)
