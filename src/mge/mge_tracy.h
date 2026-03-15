#pragma once
#include "tracy/Tracy.hpp"

extern bool g_tracyActive;

#ifdef TRACY_ENABLE
#define MGE_ZoneScopedN(name) ZoneNamedN(___tracy_scoped_zone, name, g_tracyActive)
#define MGE_ZoneScoped ZoneNamed(___tracy_scoped_zone, g_tracyActive)
#define MGE_FrameMark do { if(g_tracyActive) tracy::Profiler::SendFrameMark(nullptr); } while(0)
#define MGE_TracyPlot(name, val) do { if(g_tracyActive) TracyPlot(name, val); } while(0)
#define MGE_TracyMessage(msg, len) do { if(g_tracyActive) TracyMessage(msg, len); } while(0)
#else
#define MGE_ZoneScopedN(name)
#define MGE_ZoneScoped
#define MGE_FrameMark
#define MGE_TracyPlot(name, val)
#define MGE_TracyMessage(msg, len)
#endif
