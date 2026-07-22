// The Forge renderer bootstrap for mgeHost64 (Milestone D).
//
// This header is deliberately Forge-free and STL-free: it only declares plain
// entry points so the rest of the host (which compiles with the default MSVC
// ABI: exceptions + RTTI on) can call into the Forge glue. forgerender.cpp and
// all vendored Forge TUs compile with _HAS_EXCEPTIONS=0 / no-RTTI to match The
// Forge's required ABI — keeping that boundary at this one C-style seam avoids
// STL ODR mismatches across the link.
#pragma once

// Forward-declared, not included: ipc/hostframetimings.h pulls <cstdint>, and this header is
// deliberately dependency-free (see above). forgerender.cpp includes the real definition.
namespace IPC { struct HostFrameTimings; }

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

    // Live render-scale (supersampling): set the CURRENT internal render resolution for the
    // next renderScene, clamped to the allocation size passed at init() (ceiling scale x
    // backbuffer). The scene viewports/dispatches to this size while every size-dependent RT
    // stays allocated at the ceiling — so a scale change is a per-frame viewport move, no
    // reallocation. w==0 || h==0 ⇒ render at the full allocation size (the default).
    void setRenderSize(unsigned w, unsigned h);

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
    // SK1: skyBlob is an array of IPC::SkyDrawWire (alpha-blended sky shapes, SK1 = the gradient
    // dome); skyCount items, skyBytes total. Drawn FIRST in the colour pass with depth off + alpha
    // blend so it sits behind the opaque world. May be null / 0 (Forge sky toggle off → MW sky).
    // AT1: alphaBlob is an array of IPC::AlphaDrawWire (the scene-1 sorted-alpha world shapes,
    // CLIENT-sorted back-to-front); alphaCount items, alphaBytes total. Drawn AFTER water with
    // depth GEQUAL test + no write, per-draw blend PSO from the captured (src,dst) pair. May be
    // null / 0 (Forge Alpha Pass off → MW draws its own sorted alpha).
    // FP1a: the per-frame first-person bundle — the arms/weapon lists MW renders in its
    // own post-z-clear scene, with the ARM camera's viewProj (client-built, same
    // camera-relative convention as the main viewProj; the host applies the same
    // reverse-Z + half-pixel fixups). drawBlob = DrawItemWire[] (rigid parts),
    // skinnedBlob = [SkinnedDrawWire][palette]*. Drawn by the dedicated FP pass after
    // sorted-alpha: fresh reverse-Z depth clear, world screen-space AO/shadow masks
    // neutralized. Null ⇒ no FP pass this frame.
    struct FPScene {
        const float* viewProj = nullptr;      // 16 floats
        const void*  drawBlob = nullptr;
        unsigned     drawCount = 0;
        unsigned     drawBytes = 0;
        const void*  skinnedBlob = nullptr;
        unsigned     skinnedCount = 0;
        unsigned     skinnedBytes = 0;
        // FP1c: blended FP parts (torch flame, enchant glow) — AlphaDrawWire[] client-sorted
        // back-to-front vs the ARM camera, drawn last in the FP pass with the main alpha PSOs.
        const void*  alphaBlob = nullptr;
        unsigned     alphaCount = 0;
        unsigned     alphaBytes = 0;
    };

    bool renderScene(const float* viewProj, const float* lighting, const void* drawBlob,
                     unsigned drawCount, unsigned drawBytes,
                     const void* skinnedBlob, unsigned skinnedCount, unsigned skinnedBytes,
                     const void* multiMapBlob, unsigned multiMapCount, unsigned multiMapBytes,
                     const void* lightBlob, unsigned lightCount, unsigned lightBytes,
                     const void* skyBlob = nullptr, unsigned skyCount = 0, unsigned skyBytes = 0,
                     const void* alphaBlob = nullptr, unsigned alphaCount = 0, unsigned alphaBytes = 0,
                     // AT3: captured-alpha geometry — one blob [verts (GeomVertexWire)][indices (uint16)]
                     // (indices at capturedVertBytes). Referenced by AlphaDrawWire items with
                     // slot == IPC::kAlphaSlotCaptured. Null/0 ⇒ no captured geometry this frame.
                     const void* capturedAlphaBlob = nullptr, unsigned capturedVertBytes = 0, unsigned capturedIdxBytes = 0,
                     // WT1: per-frame water params (12 floats; see bridge.h RenderFrameParameters)
                     // + the F7 water-enable gate. Null/0 ⇒ no Forge water pass this frame.
                     const float* waterParams = nullptr, unsigned waterEnabled = 0,
                     // FP1a: first-person bundle (see FPScene above). Null ⇒ no FP pass.
                     const FPScene* fp = nullptr);

    // F12 debug view: 0 = normal, 1 = depth (world-distance grayscale), 2 = scatter (client-side).
    // Stored in a host global and written into FrameData.debugParams.x each renderScene.
    void setDebugMode(unsigned m);

    // Dev overlay input bridge (Stage 2): the client forwards polled mouse each frame; the host
    // pushes it into Forge UI via uiSetExternalInput. buttons bitmask: bit0=L,bit1=R,bit2=M.
    void setDevInput(int x, int y, unsigned buttons, float wheel, unsigned uiVisible);
    // Frame-ahead observability: client-forwarded last-frame timings + the host-side idle
    // gap (server-measured, ms between RenderFrame exits/entries) for the Stats panel.
    void setClientStats(unsigned frameAhead, float clientWaitMs, float clientDtMs,
                        float clientMwStartMs, double hostIdleMs);

    // Dev hot-reload (F8): rebuild the compute pipelines (gtao + linearize) from the on-disk dxil,
    // no game restart. Recompile + redeploy the *_0.dxil first. Idles the queue before swapping.
    void reloadComputeShaders();

    // Perf A/B (numpad -): flip the baked distant point-light loop on/off live. Same host bool the
    // dev panel "Dist lights: enable" checkbox drives — diff the gpu-split `dl=` bracket with it.
    void toggleDistLights();

    // Last frame's CPU/GPU phase split, copied into the RenderFrame completion so the client can
    // plot it in Tracy (tasks/forge-host-gpu-lane.md, Tier 1). Pure reads of values renderScene
    // already computed. See ipc/hostframetimings.h for what each field means and why it exists.
    void fillFrameTimings(IPC::HostFrameTimings& out);

    // Parts actually drawn (slot valid) in the last renderScene — for diagnostics.
    unsigned lastDrawn();

    // Skinned parts actually drawn in the last renderScene — for diagnostics.
    unsigned lastSkinnedDrawn();

    // Multi-map (Tier 4) parts actually drawn in the last renderScene — for diagnostics.
    unsigned lastMultiMapDrawn();

    // Standalone scene-path exercise (init → uploadGeometry → renderScene with a dummy
    // mesh) so host-side printf/asserts are visible in a terminal. Run via --forge-scene.
    bool sceneProbe();

    // Arm PIX programmatic GPU capture — MUST be called before sceneProbe()/init() so
    // WinPixGpuCapturer.dll hooks d3d12 device creation. sceneProbe wraps one renderScene.
    bool enablePixCapture();

    // Arm RenderDoc in-application capture — MUST be called before sceneProbe()/init() so
    // renderdoc.dll hooks d3d12 device creation. sceneProbe wraps one renderScene in
    // StartFrameCapture/EndFrameCapture (no Present needed). Best launched via the RenderDoc UI.
    bool enableRdocCapture();

    // Debug: read back the live shared RT centre pixel and printf it (BGRA). Used by the
    // --forge-scene probe to ground-truth the fragment output offline.
    void debugReadbackCenterPixel();

    // Tier 2 diag: read back pAO (RGBA16F) and printf 3 texels — GTAO write vs graphics read.
    void debugReadbackAO();

    // Phase 1a standalone probe (--forge-dl): bring Forge up, build the land pipeline, load the
    // host-owned distant land (Data Files\distantland\world + atlas) from the cwd, render it from a
    // synthetic camera into the owned RT and read back. No MW / IPC — proves the DL loader +
    // pipeline + atlas in isolation. Returns true if any land covered the frame. Run from morrowind64.
    bool renderDistantLandProbe();

    // Phase 1b standalone probe (--forge-statics): like renderDistantLandProbe, but also loads the
    // host-owned distant STATICS (static_meshes library + usage.data placements), builds the mega
    // VB/IB + per-instance stream + indirect-args for the densest exterior cell, and draws them with
    // a single cmdExecuteIndirect (instancing + bindless per-subset textures + execute-indirect) over
    // the distant land. Dumps forge_statics.tga. Proves the GPU-driven statics path in isolation.
    bool renderDistantStaticsProbe();

    // Standalone interactive world viewer (--forge-view): a real Win32 window + Forge swapchain
    // showing the host-owned distant land + statics that you fly around (WASD + arrows, Shift = fast,
    // ESC = quit). No Morrowind / IPC / client — isolates Forge renderer bugs from integration bugs
    // and iterates fast. Renders into the headless pRT (probe path) then copies pRT -> backbuffer ->
    // present, in a message loop. tasks/forge-viewer.md.
    bool worldViewer();

    // Tear down the persistent renderer (shared RT, pipeline, Forge stack).
    void shutdown();
}
