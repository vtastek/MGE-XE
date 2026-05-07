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

#include <windows.h>
// Windows.h defines `near` and `far` as macros (legacy 16-bit memory model);
// SharedSE NICamera.h has fields named `near` / `far` on NI::Frustum.
#undef near
#undef far

#include <filesystem>
#include <iomanip>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
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
