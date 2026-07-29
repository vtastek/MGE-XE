#pragma once

#include <cstdint>

// Host frame CPU/GPU timing block, forwarded in the RenderFrame completion so the CLIENT can plot
// the host's split in Tracy (tasks/forge-host-gpu-lane.md, Tier 1).
//
// WHY THIS EXISTS: the Tracy fiber lane "Forge Host Frame (inflight)" spans
// [kickoff RPC issued -> completion drained] — IPC + host CPU + host GPU + drain. Reading its width
// as GPU time yields a false "we are GPU bound": measured dense scene, a ~9.5ms box over only
// ~4.0ms of real GPU work, i.e. GPU busy ~26% of a 15.3ms frame, with the host's CPU and GPU halves
// SERIAL (they add). The lever is overlap, not shader work — but that is only obvious once the
// split is visible in the capture instead of hand-reconstructed from mgeHost64.log's `gpu split`.
// The host already computed every field per frame (real D3D12 timestamp pool); it was merely
// logged 1-in-300 and never sent.
//
// This lives in its OWN header, not bridge.h, so the x64 host can include it without dragging in
// bridge.h's d3d9header.h. Same reasoning as ipc/geomwire.h.
//
// LAYOUT IS THE CONTRACT: shared by layout between an x86 client and an x64 host, where a mismatch
// is silent corruption rather than a link error. Hence float only — no double, size_t, bool or
// pointers, all of which differ or pad differently across the two. Append new fields at the END,
// and always rebuild + deploy BOTH binaries together.
namespace IPC {

    struct HostFrameTimings {
        // WHOLE command buffer GPU EXECUTION (host kGpuPhaseFrame). The one honest "is the GPU
        // actually busy?" number — unlike gpuWaitMs it excludes submit/fence latency.
        float gpuFrameMs;

        // Per-pass GPU execution, for locating cost once gpuFrameMs says the GPU matters.
        float gpuCullMs;
        float gpuPrepassMs;
        float gpuShadowMs;
        float gpuPostDepthMs;
        float gpuReflectMs;
        float gpuColorMs;
        float gpuWaterMs;
        float gpuResolveMs;

        // Host CPU phases. These are SERIAL with the GPU segment, so they add into the inflight
        // box — which is why they, not the shaders, dominate it today.
        float cpuSetupMs;
        float cpuCullMs;
        float cpuRecordMs;
        float cpuPostMs;

        // Host CPU blocked awaiting the GPU (>= gpuFrameMs; the delta is submit/fence overhead).
        float gpuWaitMs;
        // setup + cull + record + gpuWait + post, host-measured. Compare against the inflight
        // box width: the remainder is IPC + drain latency.
        float totalMs;

        // --- NOT a timing: host TERRAIN ownership (tasks/forge-terrain.md T3). ---------------
        // 1.0 = the host has the world's LAND heightfield resident and is drawing it; 0.0 = it is
        // not (still loading, failed to load, or the debug toggle is off — the host then falls back
        // to the old DL world bake).
        //
        // The client uses this to decide whether to stop drawing MW's OWN near terrain. That has to
        // be the host's call, not a client setting: if the host is not drawing terrain and the
        // client has already suppressed MW's, the near field is a HOLE, and a hole is much worse
        // than the double-draw it replaces. Riding the frame timings rather than a new channel
        // because this block already flows back every frame and is already layout-contract'd.
        //
        // float, not bool — see the layout note above: bool differs in size/padding across the
        // x86/x64 wire, and this struct is shared BY LAYOUT.
        float terrainOwned;
    };

    // 9 GPU + 4 CPU + gpuWait + total + terrainOwned. Update the count when appending a field: the
    // point is that a stray double/pointer (or padding from one) can never slip in unnoticed,
    // because this struct is interpreted by two differently-sized processes.
    static_assert(sizeof(HostFrameTimings) == 16 * sizeof(float),
                  "HostFrameTimings must stay tightly packed floats - it crosses the x86/x64 wire");

}
