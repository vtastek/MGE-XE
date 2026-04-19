// Phase 8C: R16F _paramh VTF cache.
// Converts DXT-compressed _paramh.green (height) to a VTF-sampleable R16F texture,
// cached by source texture pointer, built lazily on first request, and freed on
// cell transition alongside the other landscape caches.
//
// D3D9 vertex textures cannot sample DXT formats directly — R16F is universally
// supported for vertex texture fetch on SM3 hardware and is the tightest layout
// for a single-channel heightmap.
#pragma once

#include "proxydx/d3d9header.h"

namespace ParamHVTF {

// Return a VTF-sampleable R16F texture derived from the source _paramh's green
// channel. Returns nullptr if conversion failed or VTF R16F is unsupported.
// Cached by source texture pointer; subsequent calls with the same source are O(1).
IDirect3DTexture9* getOrBuild(IDirect3DDevice9* device, IDirect3DBaseTexture9* paramhSource);

// Evict all cached VTF textures. Called from FixedFunctionShader::clearAllCellBatchCaches
// on cell / interior-exterior transitions, mirroring PatchDisplacement::clearAll.
void clearAll();

// Evict a single entry when its source texture is being released.
void onTextureReleased(IDirect3DBaseTexture9* tex);

// Total time spent building VTF textures (ms). For ImGui telemetry.
double getTotalBuildMs();

// Number of build operations. For ImGui telemetry.
unsigned int getBuildCount();

// Whether D3DFMT_R16F is VTF-samplable on this device. Cached on first query.
bool isSupported(IDirect3DDevice9* device);

} // namespace ParamHVTF
