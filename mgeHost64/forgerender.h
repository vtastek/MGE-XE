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

    // D3 render probe: same bring-up, then load the precompiled FSL triangle
    // shaders + root signature, create a Forge-OWNED 640x360 render target,
    // draw the hardcoded triangle, read the result back to system memory and
    // verify a coloured (non-clear) centre pixel. No MW present seam yet — this
    // proves the full Forge render path (shader/pipeline/RT/cmd/submit/readback)
    // standalone before D4 re-attaches the shared-image seam. Returns true if
    // the triangle rendered (centre pixel != clear colour).
    bool renderTriangle();

    // D4 host half: like renderTriangle, but the render target is a CROSS-PROCESS
    // SHARED D3D12 resource (D3D12_HEAP_FLAG_SHARED + B8G8R8A8_UNORM +
    // LAYOUT_UNKNOWN) adopted into a Forge RT, with an exported NT handle. Proves
    // the shared-resource creation + render path in isolation before the IPC
    // handoff + MW-side D3D9Ex ingestion are wired. Returns true if the triangle
    // rendered into the shared RT.
    bool renderTriangleShared();

    // --- D4 live path: persistent out-of-process renderer ---------------------
    // Unlike the one-shot --forge-* probes above, these keep the Forge stack +
    // shared render target alive across frames, driven by the IPC server's
    // RenderInit / RenderFrame / shutdown.

    // Bring up Forge and create a SHARED D3D12 render target (width x height,
    // B8G8R8A8) with an exported NT handle. Returns true on success. Idempotent
    // re-init tears down a prior instance first.
    // sampleCount is the requested MSAA level (1 = none). When >1 the scene renders
    // into an internal MSAA color+depth and resolves into the shared (single-sample)
    // RT before handoff; if the device can't support the count the host logs and uses 1.
    // anisoLevel is MGE's Configuration.AnisoLevel (0 = off/linear, else max anisotropy);
    // it parameterises the texture sampler built in the Phase 2 texturing path.
    bool init(unsigned width, unsigned height, unsigned sampleCount, unsigned anisoLevel);

    // The exported NT shared-RT handle — valid in the HOST process. Null until a
    // successful init(). The IPC server DuplicateHandles this into MW's process.
    void* sharedHandle();

    // True if the M1c opaque scene path (depth RT + opaque pipeline + descriptor sets)
    // built successfully in init(). False ⇒ renderScene returns false and the seam
    // falls back to the triangle. The server logs this so the buffered host stdout
    // isn't needed to know whether the scene path is live.
    bool sceneReady();

    // M1b: build a D3D12 VB/IB per part from a packed upload blob (model-space
    // pos+normal+indices, slot-indexed; see ipc/geomwire.h) and store each in the
    // host's slot-indexed mesh array. blobBytes points at the contiguous batch;
    // byteCount is its length; partCount is the number of parts to parse. Returns
    // the number of parts successfully built. Safe to call before/independently of
    // renderFrame (M1c will draw these). No-op if Forge isn't live.
    unsigned uploadGeometry(const void* blobBytes, unsigned byteCount, unsigned partCount);

    // Phase 2 bindless texturing: parse [TexUploadWire][dds bytes]* and decode each DDS into
    // gTextures[slot] (BCn/uncompressed + mip chain), rebinding that bindless descriptor.
    // count entries, byteCount total. Returns the number built. No-op until the opaque path
    // (PerFrame set) exists. The sampler is built from the AF level passed to init().
    unsigned uploadTextures(const void* blob, unsigned byteCount, unsigned count);

    // Render one frame (clear + triangle) into the shared RT and leave it in a
    // state MW's D3D9Ex StretchRect can read (COMMON). Blocking: waits on the GPU
    // fence before returning, so the RPC reply implies the frame is ready.
    bool renderFrame(unsigned frameIndex);

    // M1c: render the cached opaque scene into the shared RT (depth-tested), then
    // leave it COMMON for MW's StretchRect. viewProj is 16 floats (D3DXMATRIX bytes,
    // row-major). drawBlob is an array of IPC::DrawItemWire (slot + world[16]);
    // drawCount entries, drawBytes total. Parts whose slot has no uploaded mesh are
    // skipped. Returns false if the opaque path isn't built (caller can fall back to
    // renderFrame's triangle). Blocking (fence-waits like renderFrame).
    // M-Skinning: skinnedBlob is a sequence of [SkinnedDrawWire][palette]* (geomwire.h);
    // skinnedCount items, skinnedBytes total. The host GPU palette-skins each (no per-draw
    // world matrix — the palette is world-space). drawBlob/skinnedBlob may each be null.
    // lighting = 24 floats (6 × float4): sunDir, sunCol, ambCol, fogColNear, fogParams
    // (x=fogNearStart, y=fogNearEnd), eyePos — uploaded into gFrameData after viewProj.
    // May be null (then lighting stays whatever the cbuffer last held).
    // Tier 4: multiMapBlob is an array of IPC::MultiMapDrawWire (static opaque parts with
    // dark/detail/glow sibling maps), drawn through the host's wide multimap pipeline;
    // multiMapCount items, multiMapBytes total. May be null / 0 (no multi-map parts this frame).
    // Tier 3a: lightBlob is an array of IPC::PointLightWire (world-space point lights);
    // lightCount items, lightBytes total. Uploaded into the host light cbuffer for the
    // per-pixel FFE evalOnePointLight loop. May be null / 0 (no point lights this frame).
    bool renderScene(const float* viewProj, const float* lighting, const void* drawBlob,
                     unsigned drawCount, unsigned drawBytes,
                     const void* skinnedBlob, unsigned skinnedCount, unsigned skinnedBytes,
                     const void* multiMapBlob, unsigned multiMapCount, unsigned multiMapBytes,
                     const void* lightBlob, unsigned lightCount, unsigned lightBytes);

    // Parts actually drawn (slot valid) in the last renderScene — for diagnostics.
    unsigned lastDrawn();

    // Skinned parts actually drawn in the last renderScene — for diagnostics.
    unsigned lastSkinnedDrawn();

    // Multi-map (Tier 4) parts actually drawn in the last renderScene — for diagnostics.
    unsigned lastMultiMapDrawn();

    // Standalone scene-path exercise (init → uploadGeometry → renderScene with a dummy
    // mesh) so host-side printf/asserts are visible in a terminal. Run via --forge-scene.
    bool sceneProbe();

    // Debug: read back the live shared RT centre pixel and printf it (BGRA). Used by the
    // --forge-scene probe to ground-truth the fragment output offline.
    void debugReadbackCenterPixel();

    // Tear down the persistent renderer (shared RT, pipeline, Forge stack).
    void shutdown();
}
