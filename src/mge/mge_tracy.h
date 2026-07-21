#pragma once
#include "tracy/Tracy.hpp"
#include "tracy/TracyC.h"   // always included: defines TracyCZoneCtx (const void* when Tracy is off)

extern bool g_tracyActive;

#ifdef TRACY_ENABLE
#define MGE_ZoneScopedN(name) ZoneNamedN(___tracy_scoped_zone, name, g_tracyActive)
#define MGE_ZoneScoped ZoneNamed(___tracy_scoped_zone, g_tracyActive)
#define MGE_FrameMark do { if(g_tracyActive) tracy::Profiler::SendFrameMark(nullptr); } while(0)
#define MGE_TracyPlot(name, val) do { if(g_tracyActive) TracyPlot(name, val); } while(0)
#define MGE_TracyMessage(msg, len) do { if(g_tracyActive) TracyMessage(msg, len); } while(0)
// Cross-process host-frame lane. The Forge host renders in a SEPARATE process, so it has no real
// thread in this capture — a Tracy FIBER gives it a virtual lane so its inflight window
// [kickoff RPC issued -> completion drained] shows as a box aligned with the real client thread
// lanes, on the client's own clock. Requires TRACY_FIBERS (defined in the Release-Tracy config).
//
// The lane is named "(inflight)" ON PURPOSE: the box is the whole client-side round trip —
// IPC + host CPU (setup/cull/record/post) + host GPU + completion drain — NOT GPU execution.
// It was called "Forge Host GPU", which invited reading its width as GPU time and concluding
// "GPU bound": a ~9.5ms box against ~4.0ms of summed [forge-hb] gpu-split timestamps, i.e. the
// GPU is under half of it and ~26% of a 15.3ms frame. There is no real GPU lane in this capture
// (the host has no Tracy of its own); read per-pass GPU cost from mgeHost64.log's `gpu split`.
// Begin and End may run on DIFFERENT OS threads (the produce worker kicks the host; the main
// thread drains it) — the fiber rebinds the zone context by name regardless of the caller thread.
#define MGE_TracyHostFrameBegin(ctxLValue) do { if (g_tracyActive) { \
        TracyCFiberEnter("Forge Host Frame (inflight)"); \
        static const struct ___tracy_source_location_data ___mge_hostsl = \
            { "host frame", __func__, __FILE__, (uint32_t)__LINE__, 0x00B87333 }; \
        (ctxLValue) = ___tracy_emit_zone_begin(&___mge_hostsl, 1); \
        TracyCFiberLeave; } } while(0)
#define MGE_TracyHostFrameEnd(ctxRValue) do { if (g_tracyActive) { \
        TracyCFiberEnter("Forge Host Frame (inflight)"); \
        ___tracy_emit_zone_end(ctxRValue); \
        TracyCFiberLeave; } } while(0)
#define MGE_TracyNameThread(name) do { if (g_tracyActive) tracy::SetThreadName(name); } while(0)
#else
#define MGE_ZoneScopedN(name)
#define MGE_ZoneScoped
#define MGE_FrameMark
#define MGE_TracyPlot(name, val)
#define MGE_TracyMessage(msg, len)
#define MGE_TracyHostFrameBegin(ctxLValue) do {} while(0)
#define MGE_TracyHostFrameEnd(ctxRValue)   do {} while(0)
#define MGE_TracyNameThread(name)          do {} while(0)
#endif
