// The Forge renderer bootstrap for mgeHost64 (Milestone D).
//
// This header is deliberately Forge-free and STL-free: it only declares plain
// entry points so the rest of the host (which compiles with the default MSVC
// ABI: exceptions + RTTI on) can call into the Forge glue. forgerender.cpp and
// all vendored Forge TUs compile with _HAS_EXCEPTIONS=0 / no-RTTI to match The
// Forge's required ABI — keeping that boundary at this one C-style seam avoids
// STL ODR mismatches across the link.
#pragma once

namespace ForgeRender {
    // D1 probe: bring the full Forge stack up (mem/filesystem/log → GPU config →
    // Renderer → graphics queue → resource loader), log the selected GPU, then
    // tear everything back down. Returns true if a Renderer was created.
    // Proves the vendored build manifest compiles, links, and initialises a
    // device under DXVK/native — no rendering yet.
    bool probe();
}
