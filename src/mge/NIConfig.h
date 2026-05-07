#pragma once

// MGE-XE targets Morrowind.exe (same address space as MWSE).
// SharedSE expects each consumer to provide this shim header on its
// include path so that <SharedSE/NIAVObject.h> etc. can pick up the
// right per-target FNADDR table without dragging in MWSE-private
// preprocessor flags.
//
// MGE-XE intentionally does NOT define:
//   SE_IS_MWSE  — would pull in MWSE-private type enrichments
//   SE_USE_LUA  — would pull in sol/lua bindings
// MGE-XE DOES define (in its precompiled environment, not here):
//   SE_IS_MGE 1                — selects the MGE arm of three-arm gates
//                                in NIExtraData.h / NIObjectNET.h
//   MWSE_NO_CUSTOM_ALLOC 1     — neutralises engine-allocator references
//                                in TArray / IteratedList / StlList; safe
//                                because MGE never constructs or destroys
//                                engine-owned containers (read-only).

#include "NIConfig.Morrowind.h"
