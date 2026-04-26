#pragma once

// Distant-static instancing. Kept in its own translation unit so the
// rest of the codebase (notably the MSOC pipeline) doesn't reach into
// instancing internals.

#include "ipc/dlshare.h"
#include "ipc/vecwrap.h"
#include "proxydx/d3d8header.h"

namespace StaticInstancing {

// Lifecycle — called from distantinit.cpp's existing init / teardown
// pair. initVB allocates the per-frame instance VB; shutdownVB releases
// it. Idempotent on failure paths.
bool initVB(IDirect3DDevice9* device);
void shutdownVB();

// Per-frame: pack the MSOC-filtered visible set into the instance VB
// and group consecutive same-state meshes into draw batches. Reads
// DistantLand::msocOccluded for per-instance cull decisions.
template<class T>
void buildVB(VisibleSet<T>& staticSet);

// Instanced color pass (PASS_RENDERSTATICSEXTERIOR_INST or _INTERIOR).
// Caller is expected to have already invoked BeginPass.
void renderColor();

// Instanced depth pre-pass (PASS_RENDERSTATICSDEPTH_INST). Caller is
// expected to have already invoked BeginPass on effectDepth.
void renderDepth();

} // namespace StaticInstancing
