#pragma once

// Force-included on SharedSE TUs that MGE compiles and on MGE TUs that
// consume SharedSE NI types. Sets the consumer identity gates (SE_IS_MGE +
// MWSE_NO_CUSTOM_ALLOC) and pulls in the standard headers SharedSE expects
// from its consumer's PCH.
//
// MWSE force-includes its stdafx.h on every TU; CSSE force-includes pch.h.
// MGE-XE doesn't have a PCH, so this header plays the equivalent role on
// the narrow set of TUs that touch SharedSE — keeping the rest of MGE-XE's
// build untouched.

// Windows + D3D includes go BEFORE the near/far #undef. Both windows.h
// and d3d9.h's DEFINE_GUID expansions reference the legacy 16-bit
// memory-model macros (NEAR/FAR — and lowercase aliases on some SDK
// versions) during macro expansion of EXTERN_C const GUID FAR name.
// Undefining them up here, before the SDK chain gets to expand those
// DEFINE_GUID lines, leaves trailing identifiers without their FAR
// qualifier and breaks parsing of every IID definition in d3d9.h.
//
// proxydx/d3d9header.h hits the same constraint — it includes d3d9.h
// THEN undefs. We mirror that order. SharedSE's NICamera.h has fields
// literally named `near` / `far` (the only reason we undef at all),
// so the undefs MUST happen between the SDK chain and SharedSE.
//
// proxydx/d3d8header.h is included rather than <d3d8.h> because
// modern Windows SDKs (10.0.22621+ verified) no longer ship d3d8.h.
// Proxydx provides the d3d8 type shims (D3DCAPS8, D3DADAPTER_-
// IDENTIFIER8, D3DPRESENT_PARAMETERS8, IDirect3DDevice8 typedef etc.)
// that SharedSE/NIDX8Renderer.h depends on. Their layouts match the
// real d3d8 SDK (verified: sizeof(DX8DeviceDesc) == 0xF4 holds with
// proxydx's D3DCAPS8). It also transitively pulls d3d9.h via
// proxydx/d3d9header.h.
#include <windows.h>
#include "proxydx/d3d8header.h"
#undef near
#undef far

// SharedSE/NIDX8Renderer.h declares fields like
//   D3DPRESENT_PARAMETERS d3dPresentParameters;
// expecting the legacy d3d8 layout (52 bytes / 0x34). Modern d3d9.h's
// D3DPRESENT_PARAMETERS is 56 bytes (has the extra MultiSampleQuality
// field d3d9 added). Without correction the offset of every subsequent
// field shifts by 4 and the engine-truth size_validation static_assert
// at the end of DX8Renderer fails. proxydx provides the correctly-sized
// D3DPRESENT_PARAMETERS8 alias; mapping the d3d9 name to it, scoped to
// SharedSE TUs only (this prelude is force-included only on those),
// keeps the layout matching the engine without touching MGE's own
// d3d9-aware code.
#define D3DPRESENT_PARAMETERS    D3DPRESENT_PARAMETERS8

#include <cassert>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define span_CONFIG_SELECT_SPAN span_SPAN_NONSTD
#include <nonstd/span.hpp>

#define SE_IS_MGE 1
#define MWSE_NO_CUSTOM_ALLOC 1

// Engine allocator entry points on Morrowind.exe. MGE-XE shares this process,
// so the addresses are valid; in practice MGE never invokes these (it is a
// read-only consumer), but a few SharedSE call sites reference se::memory::_new
// outside of MWSE_NO_CUSTOM_ALLOC's gate, so the symbols must resolve at compile
// time. Same values MWSE's stdafx.h ships with.
#define SE_MEMORY_FNADDR_NEW     0x727692
#define SE_MEMORY_FNADDR_DELETE  0x727530
#define SE_MEMORY_FNADDR_MALLOC  0x727738
#define SE_MEMORY_FNADDR_FREE    0x727732
#define SE_MEMORY_FNADDR_REALLOC 0x746288
