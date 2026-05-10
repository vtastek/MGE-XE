#pragma once

// MGE-XE targets Morrowind.exe (same address space as MWSE).
// Shim parallels NIConfig.h — selects the engine's vtable address table.
//
// SharedSE/NIDefines.h includes "NIVirtualTableDefines.h" with no path
// prefix, expecting each consumer to provide its own per-binary shim on
// the include path. MWSE/ and CSSE/ each ship their own (selecting the
// .Morrowind / .TESConstructionSet variants respectively); MGE-XE picks
// the Morrowind variant since it's loaded into Morrowind.exe.

#include "NIVirtualTableDefines.Morrowind.h"
