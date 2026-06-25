// The Forge renderer bootstrap for mgeHost64 — Milestones D1 (probe), D3 (render),
// D4 (shared render target).
//
// Brings the full Forge stack up headlessly (no window system): replicate just
// the subsystem init that WindowsBase.cpp's app main does — initMemAlloc →
// initFileSystem → initLog — then GPU config → Renderer → graphics queue →
// resource loader. D1 (probe) stops there and logs the GPU. D3 (renderTriangle)
// loads the precompiled FSL triangle shaders + root signature, creates a
// Forge-OWNED render target, draws the triangle and verifies a coloured centre
// pixel. D4 (renderTriangleShared) does the same but into a CROSS-PROCESS SHARED
// D3D12 render target (created per the D3D9Ex-ingestion rules: D3D12_HEAP_FLAG_
// SHARED + DXGI_FORMAT_B8G8R8A8_UNORM + D3D12_TEXTURE_LAYOUT_UNKNOWN) and exports
// an NT handle — the host half of the route-1 present seam.
//
// Compiled with _HAS_EXCEPTIONS=0 / no-RTTI to match The Forge's ABI (see
// forgerender.h). Keep host (STL) headers OUT of this TU — talk to the rest of
// the host only through the plain decls in forgerender.h.

#include "forgerender.h"
#include "ipc/geomwire.h"
#include "support/log.h"   // LOG::logline -> mgeHost64.log (LOGF goes to uncaptured stdout)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>

#include "OS/Interfaces/IOperatingSystem.h"
#include "Utilities/Interfaces/IFileSystem.h"
#include "Utilities/Log/Log.h"
#include "Graphics/GraphicsConfig.h"
#include "Graphics/Interfaces/IGraphics.h"
#include "Resources/ResourceLoader/Interfaces/IResourceLoader.h"
#include "Utilities/Interfaces/ILog.h"
#if defined(ENABLE_GRAPHICS_VALIDATION)
// ID3D12InfoQueue (break-on-severity control). d3d12.h is already pulled in by
// IGraphics.h for the D3D12 backend; this only adds the debug-layer interface.
// Include BEFORE IMemory.h (which overrides new/delete/malloc) to avoid the
// allocator macros mangling the system header.
#include <d3d12sdklayers.h>
#endif
// IMemory.h overrides new/delete/malloc — Forge convention: include it LAST.
#include "Utilities/Interfaces/IMemory.h"
// M1c opaque-scene SRT. defaults.h provides the C++ definitions of the FSL macros
// (STRUCT/DATA/BEGIN_SRT/DECL_CBUFFER/SRT_SET_DESC/SRT_RES_IDX); the .srt.h then
// declares SRT_SrtData + the gFrameData/gWorlds descriptor indices. Per the Forge
// convention (06_MaterialPlayground) these come AFTER IMemory.h.
#include "Graphics/FSL/defaults.h"
#include "shaders/FSL/opaque.srt.h"

// App-layer callback normally provided by WindowsBase.cpp (which we exclude — it
// drags in the window system). The backend calls this on device-lost. Headless:
// no swapchain to rebuild, so record nothing. The header's extern "C" block
// gives this C linkage to match the (compiled-as-C) Direct3D12.c reference.
void requestReset(const ResetDesc* pResetDesc) { (void)pResetDesc; }

namespace {
    const char* kAppName = "mgeHost64";

    inline uint32_t roundUp(uint32_t v, uint32_t a) { return a ? ((v + a - 1) / a) * a : v; }
    inline uint64_t roundUp64(uint64_t v, uint32_t a) { return a ? ((v + a - 1) / a) * a : v; }

    // Negative determinant of the world matrix's upper-left 3x3 ⇒ the transform flips
    // triangle winding (a mirrored part). Such draws need the opposite front-face pipeline
    // or they cull the wrong face and render inside-out. Row-major D3DXMATRIX (item.world).
    // Mirrors the cache's isMirroredMatrix so host + D3D9 agree on the sign.
    inline bool worldMirrored(const float m[16]) {
        const float det = m[0] * (m[5] * m[10] - m[6] * m[9])
                        - m[1] * (m[4] * m[10] - m[6] * m[8])
                        + m[2] * (m[4] * m[9]  - m[5] * m[8]);
        return det < 0.0f;
    }

    // If the D3D12 device has been removed, log WHY (the DXGI_ERROR reason code) and
    // return true. Available WITHOUT the debug layer — GetDeviceRemovedReason is always
    // present. With ENABLE_GRAPHICS_VALIDATION on, the preceding InfoQueue messages name
    // the exact offending call. Call after batches of GPU work (upload, draw submit) so a
    // removal is pinned to the operation that caused it instead of surfacing as silent black.
    bool g_deviceRemovedLogged = false;   // latch: log the removal once, not every frame
    bool logDeviceRemoved(Renderer* R, const char* where) {
        if (!R || !R->mDx.pDevice) {
            return false;
        }
        HRESULT reason = R->mDx.pDevice->GetDeviceRemovedReason();
        if (reason != S_OK) {
            if (!g_deviceRemovedLogged) {
                std::printf("[forge] !! DEVICE REMOVED at %s — reason 0x%08lX\n", where, (unsigned long)reason);
                LOGF(eERROR, "[forge] !! DEVICE REMOVED at %s — reason 0x%08lX", where, (unsigned long)reason);
                g_deviceRemovedLogged = true;
            }
            return true;
        }
        return false;
    }

    // Shared bring-up: mem → filesystem → resource dirs → log → GPU config →
    // Renderer → graphics queue → resource loader. On success *ppRenderer and
    // *ppQueue are live and the resource loader is initialised. On failure tears
    // down whatever it created and returns false.
    bool forgeBringUp(Renderer** ppRenderer, Queue** ppQueue) {
        *ppRenderer = nullptr;
        *ppQueue = nullptr;

        std::printf("[forge] initMemAlloc...\n");
        if (!initMemAlloc(kAppName)) {
            std::printf("[forge] initMemAlloc FAILED\n");
            return false;
        }

        std::printf("[forge] initFileSystem...\n");
        FileSystemInitDesc fsDesc = {};
        fsDesc.pAppName = kAppName;
        // Headless: we set resource dirs explicitly below, so skip the bundled
        // PathStatement.txt the app framework normally ships (its absence is
        // what makes initFileSystem fail otherwise).
        fsDesc.mIsTool = true;
        if (!initFileSystem(&fsDesc)) {
            std::printf("[forge] initFileSystem FAILED\n");
            exitMemAlloc();
            return false;
        }

        // Resource dirs must be set BEFORE initLog (initLog opens its file under
        // RD_LOG). All point at the exe's working dir. RD_SHADER_BINARIES = "" so
        // loadShader finds DIRECT3D12/<name> beside the exe; gpu.cfg/gpu.data sit
        // under RD_GPU_CONFIG.
        std::printf("[forge] setting resource dirs...\n");
        fsSetPathForResourceDir(pSystemFileIO, RD_LOG, "");
        fsSetPathForResourceDir(pSystemFileIO, RD_GPU_CONFIG, "");
        fsSetPathForResourceDir(pSystemFileIO, RD_OTHER_FILES, "");
        fsSetPathForResourceDir(pSystemFileIO, RD_SHADER_BINARIES, "");
        fsSetPathForResourceDir(pSystemFileIO, RD_PIPELINE_CACHE, "");

        std::printf("[forge] initLog...\n");
        initLog(kAppName, DEFAULT_LOG_LEVEL);

        // FORGE_DEBUG turns on The Forge's internal ASSERTs, which on Windows pop a MODAL
        // MessageBox ("Display more asserts? Yes/No") that FREEZES this headless host mid-frame
        // and blocks the game. Disable interactive mode so a failed assert returns silently
        // instead of blocking — the D3D12 validation layer (InfoQueue -> LOGF) and the
        // GetDeviceRemovedReason logging still record the actual fault to mgeHost64.log.
        _EnableInteractiveMode(false);

        Renderer*    pRenderer = nullptr;
        RendererDesc settings = {};
        std::printf("[forge] initGPUConfiguration...\n");
        initGPUConfiguration(settings.pExtendedSettings);
        std::printf("[forge] initRenderer...\n");
        initRenderer(kAppName, &settings, &pRenderer);
        if (!pRenderer) {
            std::printf("[forge] initRenderer FAILED (no compatible GPU?)\n");
            exitGPUConfiguration();
            exitLog();
            exitFileSystem();
            exitMemAlloc();
            return false;
        }
        setupGPUConfigurationPlatformParameters(pRenderer, settings.pExtendedSettings);

        const char* gpuName = pRenderer->pGpu ? pRenderer->pGpu->mGpuVendorPreset.mGpuName : "(unknown)";
        std::printf("[forge] initRenderer OK — GPU: %s\n", gpuName);
        LOGF(eINFO, "[forge] initRenderer OK — GPU: %s", gpuName);

#if defined(ENABLE_GRAPHICS_VALIDATION)
        // The D3D12 debug/validation layer is compiled in. Forge registers an InfoQueue
        // message callback (DebugMessageCallback -> LOGF) so validation errors land in
        // mgeHost64.log, but it ALSO sets break-on-CORRUPTION/ERROR. In this headless host
        // (no debugger attached) a break raises EXCEPTION_BREAKPOINT and kills the process
        // BEFORE we can read the message. Disable the breaks so validation messages are
        // LOGGED but never abort — exactly what we want for diagnosing the exterior fault.
        {
            ID3D12InfoQueue* pInfoQueue = nullptr;
            if (SUCCEEDED(pRenderer->mDx.pDevice->QueryInterface(IID_PPV_ARGS(&pInfoQueue)))) {
                pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, FALSE);
                pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, FALSE);
                pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, FALSE);
                pInfoQueue->Release();
                std::printf("[forge] D3D12 validation layer ON (breaks disabled; errors -> mgeHost64.log)\n");
                LOGF(eINFO, "[forge] D3D12 validation layer ON (breaks disabled)");
            } else {
                std::printf("[forge] validation compiled in but InfoQueue unavailable (debug layer not installed?)\n");
                LOGF(eWARNING, "[forge] validation compiled in but InfoQueue unavailable");
            }
        }
#endif

        Queue*    pQueue = nullptr;
        QueueDesc queueDesc = {};
        queueDesc.mType = QUEUE_TYPE_GRAPHICS;
        initQueue(pRenderer, &queueDesc, &pQueue);
        if (!pQueue) {
            std::printf("[forge] initQueue FAILED\n");
            exitRenderer(pRenderer);
            exitGPUConfiguration();
            exitLog();
            exitFileSystem();
            exitMemAlloc();
            return false;
        }
        std::printf("[forge] graphics queue created\n");

        initResourceLoaderInterface(pRenderer);
        std::printf("[forge] resource loader up\n");

        *ppRenderer = pRenderer;
        *ppQueue = pQueue;
        return true;
    }

    void forgeTearDown(Renderer* pRenderer, Queue* pQueue) {
        exitResourceLoaderInterface(pRenderer);
        if (pQueue) {
            exitQueue(pRenderer, pQueue);
        }
        exitRenderer(pRenderer);
        exitGPUConfiguration();
        exitLog();
        exitFileSystem();
        exitMemAlloc();
    }

    // Load the global graphics root signature (from the compiled default.rootsig)
    // and the triangle shaders. Returns the shader, or null on failure.
    Shader* loadTriangleShader(Renderer* pRenderer) {
        std::printf("[forge] initRootSignature (default.rootsig)...\n");
        RootSignatureDesc rsDesc = {};
        rsDesc.pGraphicsFileName = "default.rootsig";
        initRootSignature(pRenderer, &rsDesc);

        std::printf("[forge] addShader (tri.vert/tri.frag)...\n");
        ShaderLoadDesc shaderDesc = {};
        shaderDesc.mVert.pFileName = "tri.vert";
        shaderDesc.mFrag.pFileName = "tri.frag";
        Shader* pShader = nullptr;
        addShader(pRenderer, &shaderDesc, &pShader);
        if (!pShader) {
            std::printf("[forge] addShader FAILED (shader binaries missing?)\n");
            exitRootSignature(pRenderer);
        }
        return pShader;
    }

    // Graphics pipeline for the hardcoded triangle: no vertex layout (verts come
    // from SV_VertexID), no depth, CULL_NONE, colour format taken from the RT.
    Pipeline* buildTrianglePipeline(Renderer* pRenderer, RenderTarget* pRT, Shader* pShader) {
        RasterizerStateDesc rasterDesc = {};
        rasterDesc.mCullMode = CULL_MODE_NONE;

        PipelineDesc pipelineDescOuter = {};
        pipelineDescOuter.mType = PIPELINE_TYPE_GRAPHICS;
        GraphicsPipelineDesc& g = pipelineDescOuter.mGraphicsDesc;
        g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
        g.mRenderTargetCount = 1;
        g.pColorFormats = &pRT->mFormat;
        g.mSampleCount = SAMPLE_COUNT_1;
        g.mSampleQuality = 0;
        g.pDepthState = nullptr;
        g.pVertexLayout = nullptr;
        g.pRasterizerState = &rasterDesc;
        g.pShaderProgram = pShader;
        Pipeline* pPipeline = nullptr;
        addPipeline(pRenderer, &pipelineDescOuter, &pPipeline);
        return pPipeline;
    }

    // Record clear + draw into pRT, submit, read the RT back to system memory and
    // verify the triangle rasterised (coloured centre pixel, black corner). Owns
    // and tears down its own cmd pool / cmd / fence / readback buffer.
    bool drawTriangleAndVerify(Renderer* pRenderer, Queue* pQueue, RenderTarget* pRT, Pipeline* pPipeline) {
        const uint32_t kWidth = pRT->mWidth;
        const uint32_t kHeight = pRT->mHeight;

        CmdPool*    pCmdPool = nullptr;
        CmdPoolDesc cmdPoolDesc = {};
        cmdPoolDesc.pQueue = pQueue;
        initCmdPool(pRenderer, &cmdPoolDesc, &pCmdPool);
        Cmd*    pCmd = nullptr;
        CmdDesc cmdDesc = {};
        cmdDesc.pPool = pCmdPool;
        initCmd(pRenderer, &cmdDesc, &pCmd);
        Fence* pFence = nullptr;
        initFence(pRenderer, &pFence);

        const uint32_t rowAlign = (pRenderer->pGpu->mUploadBufferTextureRowAlignment > 1u)
                                      ? pRenderer->pGpu->mUploadBufferTextureRowAlignment : 1u;
        const uint32_t texAlign = (pRenderer->pGpu->mUploadBufferTextureAlignment > 1u)
                                      ? pRenderer->pGpu->mUploadBufferTextureAlignment : 1u;
        const uint32_t rowPitch = roundUp(kWidth * 4u, rowAlign);
        const uint64_t bufSize = roundUp64((uint64_t)rowPitch * kHeight, texAlign);

        BufferLoadDesc bufDesc = {};
        bufDesc.mDesc.mSize = bufSize;
        bufDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_TO_CPU;
        bufDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        bufDesc.mDesc.mStartState = RESOURCE_STATE_COPY_DEST;
        bufDesc.mDesc.mQueueType = QUEUE_TYPE_TRANSFER;
        Buffer* pReadback = nullptr;
        bufDesc.ppBuffer = &pReadback;
        addResource(&bufDesc, nullptr);
        waitForAllResourceLoads();

        std::printf("[forge] recording draw...\n");
        resetCmdPool(pRenderer, pCmdPool);
        beginCmd(pCmd);

        BindRenderTargetsDesc bind = {};
        bind.mRenderTargetCount = 1;
        bind.mRenderTargets[0] = { pRT, LOAD_ACTION_CLEAR };
        cmdBindRenderTargets(pCmd, &bind);
        cmdSetViewport(pCmd, 0.0f, 0.0f, (float)kWidth, (float)kHeight, 0.0f, 1.0f);
        cmdSetScissor(pCmd, 0, 0, kWidth, kHeight);
        cmdBindPipeline(pCmd, pPipeline);
        cmdDraw(pCmd, 3, 0);
        cmdBindRenderTargets(pCmd, nullptr);

        RenderTargetBarrier rtBarrier = {};
        rtBarrier.pRenderTarget = pRT;
        rtBarrier.mCurrentState = RESOURCE_STATE_RENDER_TARGET;
        rtBarrier.mNewState = RESOURCE_STATE_COPY_SOURCE;
        cmdResourceBarrier(pCmd, 0, nullptr, 0, nullptr, 1, &rtBarrier);
        endCmd(pCmd);

        QueueSubmitDesc submitDesc = {};
        submitDesc.mCmdCount = 1;
        submitDesc.ppCmds = &pCmd;
        submitDesc.pSignalFence = pFence;
        submitDesc.mSubmitDone = true;
        std::printf("[forge] queueSubmit...\n");
        queueSubmit(pQueue, &submitDesc);
        waitForFences(pRenderer, 1, &pFence);
        std::printf("[forge] render submit complete\n");

        TextureCopyDesc copyDesc = {};
        copyDesc.pTexture = pRT->pTexture;
        copyDesc.pBuffer = pReadback;
        copyDesc.pWaitSemaphore = nullptr;
        copyDesc.mTextureState = RESOURCE_STATE_COPY_SOURCE;
        copyDesc.mQueueType = QUEUE_TYPE_GRAPHICS;
        SyncToken copyToken = {};
        copyResource(&copyDesc, &copyToken);
        waitForToken(&copyToken);

        bool ok = false;
        const uint8_t* pixels = (const uint8_t*)pReadback->pCpuMappedAddress;
        if (pixels) {
            const uint8_t* centre = &pixels[(uint64_t)(kHeight / 2) * rowPitch + (uint64_t)(kWidth / 2) * 4u];
            const uint8_t* corner = &pixels[(uint64_t)2 * rowPitch + (uint64_t)2 * 4u];
            std::printf("[forge] centre pixel = %u,%u,%u,%u\n", centre[0], centre[1], centre[2], centre[3]);
            std::printf("[forge] corner pixel = %u,%u,%u,%u\n", corner[0], corner[1], corner[2], corner[3]);
            ok = (centre[0] | centre[1] | centre[2]) != 0;
            std::printf("[forge] triangle rendered: %s\n", ok ? "YES" : "NO");
        } else {
            std::printf("[forge] readback buffer not mapped!\n");
        }

        removeResource(pReadback);
        exitFence(pRenderer, pFence);
        exitCmd(pRenderer, pCmd);
        exitCmdPool(pRenderer, pCmdPool);
        return ok;
    }
}

namespace ForgeRender {
    bool probe() {
        // Unbuffered: a Forge ASSERT/abort would otherwise swallow piped stdout.
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        std::printf("[forge] probe: bringing up The Forge (D3D12)...\n");

        Renderer* pRenderer = nullptr;
        Queue*    pQueue = nullptr;
        if (!forgeBringUp(&pRenderer, &pQueue)) {
            return false;
        }

        forgeTearDown(pRenderer, pQueue);
        std::printf("[forge] probe complete — full bring-up + teardown OK\n");
        return true;
    }

    // Standalone exercise of the M1c opaque scene path (init → uploadGeometry →
    // renderScene) with a dummy triangle mesh, so the host-side printf/asserts are
    // visible in a terminal. Isolates a buildOpaquePath/draw crash from the IPC seam.
    bool sceneProbe() {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        std::printf("[forge] scene-probe: init...\n");
        if (!init(640, 360, 4, 16)) {   // exercise the MSAA path (resolve into the shared RT); falls back to 1x if unsupported
            std::printf("[forge] scene-probe: init FAILED\n");
            return false;
        }

        // Fullscreen triangle in NDC (viewProj + world identity) so it actually covers the RT
        // centre — the old 0..100 triangle landed off-screen, so the probe never visually tested.
        IPC::GeomVertexWire verts[3] = {
            { -1.0f, -1.0f, 0.5f,  0,0,1,  0,0,  0xFFFFFFFFu },
            {  3.0f, -1.0f, 0.5f,  0,0,1,  2,0,  0xFFFFFFFFu },
            { -1.0f,  3.0f, 0.5f,  0,0,1,  0,2,  0xFFFFFFFFu },
        };
        uint16_t idx[3] = { 0, 1, 2 };
        uint8_t blob[sizeof(IPC::GeomPartWire) + sizeof(verts) + sizeof(idx)];
        IPC::GeomPartWire hdr = {};
        hdr.slot = 0; hdr.revisionID = 0; hdr.vertexCount = 3; hdr.indexCount = 3;
        std::memcpy(blob, &hdr, sizeof(hdr));
        std::memcpy(blob + sizeof(hdr), verts, sizeof(verts));
        std::memcpy(blob + sizeof(hdr) + sizeof(verts), idx, sizeof(idx));
        std::printf("[forge] scene-probe: uploadGeometry...\n");
        unsigned built = uploadGeometry(blob, (unsigned)sizeof(blob), 1);
        std::printf("[forge] scene-probe: built %u/1\n", built);

        // Pack a dummy SKINNED part into a second blob (slot 1): 3 verts fully weighted to
        // bone 0, 1 bone. Exercises the skinned VB upload + skinned pipeline + draw.
        IPC::SkinnedVertexWire skVerts[3] = {
            {   0.0f,   0.0f, 0.0f,  0,0,1,  1,0,0,0,  0,  0.0f,0.0f },
            { 100.0f,   0.0f, 0.0f,  0,0,1,  1,0,0,0,  0,  1.0f,0.0f },
            {   0.0f, 100.0f, 0.0f,  0,0,1,  1,0,0,0,  0,  0.0f,1.0f },
        };
        uint16_t skIdx[3] = { 0, 1, 2 };
        uint8_t skBlob[sizeof(IPC::GeomPartWire) + sizeof(skVerts) + sizeof(skIdx)];
        IPC::GeomPartWire skHdr = {};
        skHdr.slot = 1; skHdr.revisionID = 0; skHdr.flags = IPC::kGeomFlagSkinned;
        skHdr.vertexCount = 3; skHdr.indexCount = 3; skHdr.numBones = 1;
        std::memcpy(skBlob, &skHdr, sizeof(skHdr));
        std::memcpy(skBlob + sizeof(skHdr), skVerts, sizeof(skVerts));
        std::memcpy(skBlob + sizeof(skHdr) + sizeof(skVerts), skIdx, sizeof(skIdx));
        std::printf("[forge] scene-probe: uploadGeometry (skinned)...\n");
        unsigned skBuilt = uploadGeometry(skBlob, (unsigned)sizeof(skBlob), 1);
        std::printf("[forge] scene-probe: skinned built %u/1\n", skBuilt);

        // Tier 4: a dummy MULTI-MAP part (slot 2): a fullscreen triangle with 4 UV sets (set 1
        // = set 0 here). Exercises the wide VB upload + multimap pipeline + draw.
        IPC::GeomVertexWireMM mmVerts[3] = {
            { -1.0f, -1.0f, 0.5f,  0,0,1,  0xFFFFFFFFu, { {0,0},{0,0},{0,0},{0,0} } },
            {  3.0f, -1.0f, 0.5f,  0,0,1,  0xFFFFFFFFu, { {2,0},{2,0},{0,0},{0,0} } },
            { -1.0f,  3.0f, 0.5f,  0,0,1,  0xFFFFFFFFu, { {0,2},{0,2},{0,0},{0,0} } },
        };
        uint16_t mmIdx[3] = { 0, 1, 2 };
        uint8_t mmBlob[sizeof(IPC::GeomPartWire) + sizeof(mmVerts) + sizeof(mmIdx)];
        IPC::GeomPartWire mmHdr = {};
        mmHdr.slot = 2; mmHdr.revisionID = 0; mmHdr.flags = IPC::kGeomFlagMultiMap;
        mmHdr.vertexCount = 3; mmHdr.indexCount = 3;
        std::memcpy(mmBlob, &mmHdr, sizeof(mmHdr));
        std::memcpy(mmBlob + sizeof(mmHdr), mmVerts, sizeof(mmVerts));
        std::memcpy(mmBlob + sizeof(mmHdr) + sizeof(mmVerts), mmIdx, sizeof(mmIdx));
        std::printf("[forge] scene-probe: uploadGeometry (multi-map)...\n");
        unsigned mmBuilt = uploadGeometry(mmBlob, (unsigned)sizeof(mmBlob), 1);
        std::printf("[forge] scene-probe: multi-map built %u/1\n", mmBuilt);

        // Identity viewProj + identity world, one static draw item at slot 0.
        float vp[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
        // Tier 1 lighting (6×float4): sun OFF, ambient 0.5, no fog — so the SLOT-0 white-texture
        // guard below stays a clean normal-independent value (white × 0.5, tonemapped ≈ 129).
        float lt[24] = {
            0,0,-1,0,           // sunDir (unused, sunCol=0)
            0,0,0,0,            // sunCol = 0
            0.5f,0.5f,0.5f,0,   // ambCol
            0,0,0,0,            // fogColNear
            0, 1e9f, 0,0,       // fogParams: start=0, end=1e9 -> fog ≈ 1 (clear)
            0,0,0,0             // eyePos
        };
        IPC::DrawItemWire item = {};
        item.slot = 0;
        // Tier 2b: white diffuse/ambient, no emissive, vColSource 0 (const material) — the
        // "no material" default. Keeps the SLOT-0 guard at ~129 (white * 0.5 ambient, tonemapped);
        // a zero material would make vertexMaterialNone render black.
        item.matDiffuse[0] = item.matDiffuse[1] = item.matDiffuse[2] = 1.0f;
        item.matAmbient[0] = item.matAmbient[1] = item.matAmbient[2] = 1.0f;
        item.vColSource = 0;
        float ident[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
        std::memcpy(item.world, ident, sizeof(ident));

        // Skinned draw blob: [SkinnedDrawWire][1 identity bone matrix].
        IPC::SkinnedDrawWire skItem = {};
        skItem.slot = 1; skItem.numBones = 1; skItem.mirror = 0;
        uint8_t skDraw[sizeof(IPC::SkinnedDrawWire) + 64];
        std::memcpy(skDraw, &skItem, sizeof(skItem));
        std::memcpy(skDraw + sizeof(skItem), ident, sizeof(ident));   // 16 floats = 64 bytes

        // First render builds the lazy opaque path (incl. the bindless PerFrame set) so the
        // texture upload below has somewhere to land. texIndex 0 here = default white.
        std::printf("[forge] scene-probe: renderScene #1 (builds path)...\n");
        bool ok = renderScene(vp, lt, &item, 1, (unsigned)sizeof(item),
                              skDraw, 1, (unsigned)sizeof(skDraw),
                              nullptr, 0, 0,    // no multi-map this pass
                              nullptr, 0, 0);   // no point lights (CENTRE stays ~129)

        // SLOT-0 regression guard: sample gTextures[0] (default white) THROUGH the shader. Skinned
        // parts (TexIndex 0) and oversize-texture fallbacks all use slot 0, so it MUST sample white
        // (160,160,160), not black — a 1x1 default white regressed this (see buildOpaquePath).
        item.texIndex = 0;
        renderScene(vp, lt, &item, 1, (unsigned)sizeof(item), skDraw, 1, (unsigned)sizeof(skDraw), nullptr, 0, 0, nullptr, 0, 0);
        std::printf("[forge] scene-probe: SLOT-0 (default white) sample -> expect ~129 (white x 0.5 ambient, tonemapped), NOT 0\n");
        debugReadbackCenterPixel();

        // Synthetic 4x4 all-white DXT1 DDS → texture slot 1. Exercises parseDds (BC1) + the
        // create-empty + mip-upload + bindless-descriptor path offline.
        uint8_t dds[128 + 8] = {};
        dds[0] = 'D'; dds[1] = 'D'; dds[2] = 'S'; dds[3] = ' ';
        *(uint32_t*)(dds + 4)  = 124;          // dwSize
        *(uint32_t*)(dds + 12) = 4;            // dwHeight
        *(uint32_t*)(dds + 16) = 4;            // dwWidth
        *(uint32_t*)(dds + 28) = 1;            // dwMipMapCount
        *(uint32_t*)(dds + 76) = 32;           // ddspf dwSize
        *(uint32_t*)(dds + 80) = 0x4;          // DDPF_FOURCC
        *(uint32_t*)(dds + 84) = 0x31545844u;  // 'DXT1'
        dds[128] = 0xFF; dds[129] = 0xFF; dds[130] = 0xFF; dds[131] = 0xFF;  // c0=c1=white, idx 0
        uint8_t texBlob[sizeof(IPC::TexUploadWire) + sizeof(dds)];
        IPC::TexUploadWire th{ 1, (uint32_t)sizeof(dds) };
        std::memcpy(texBlob, &th, sizeof(th));
        std::memcpy(texBlob + sizeof(th), dds, sizeof(dds));
        std::printf("[forge] scene-probe: uploadTextures...\n");
        unsigned texBuilt = uploadTextures(texBlob, (unsigned)sizeof(texBlob), 1);
        std::printf("[forge] scene-probe: textures built %u/1\n", texBuilt);

        // Second render samples the uploaded texture (slot 1).
        item.texIndex = 1;
        std::printf("[forge] scene-probe: renderScene #2 (samples slot 1)...\n");
        ok = renderScene(vp, lt, &item, 1, (unsigned)sizeof(item),
                         skDraw, 1, (unsigned)sizeof(skDraw),
                         nullptr, 0, 0,
                         nullptr, 0, 0);
        std::printf("[forge] scene-probe: renderScene returned %d (skinnedDrawn=%u)\n",
                    (int)ok, lastSkinnedDrawn());

        debugReadbackCenterPixel();   // ground-truth the frag output (defined after g_live)

        // Third render: MULTI-MAP ONLY (slot 2) — 2 stages over the now-resident white texture
        // (slot 1): BASE on UV set 0 + GLOW ADD on UV set 1. base = white*lit (lit = white*(0.5
        // ambient) = 0.5); glow ADD white → 1.5 → tonemap ≈ 0.97 → CENTRE ≈ 248. Proves the MM
        // pipeline builds + draws (multiMapDrawn=1) with no device removal.
        {
            IPC::MultiMapDrawWire mm = {};
            mm.slot = 2;
            std::memcpy(mm.world, ident, sizeof(ident));
            mm.matDiffuse[0] = mm.matDiffuse[1] = mm.matDiffuse[2] = 1.0f;
            mm.matAmbient[0] = mm.matAmbient[1] = mm.matAmbient[2] = 1.0f;
            mm.vColSource = 0;
            mm.alphaRef = 0.0f;
            mm.stageCount = 2;
            mm.stages[0] = IPC::packMMStage(1, 0, IPC::kMMOpBase);   // base, white slot 1, UV 0
            mm.stages[1] = IPC::packMMStage(1, 1, IPC::kMMOpAdd);    // glow ADD, white slot 1, UV 1
            std::printf("[forge] scene-probe: renderScene #3 (multi-map only)...\n");
            ok = renderScene(vp, lt, nullptr, 0, 0,
                             nullptr, 0, 0,
                             &mm, 1, (unsigned)sizeof(mm),
                             nullptr, 0, 0);
            std::printf("[forge] scene-probe: multi-map renderScene returned %d (multiMapDrawn=%u) -> expect CENTRE ~248\n",
                        (int)ok, lastMultiMapDrawn());
            debugReadbackCenterPixel();
        }

        shutdown();
        std::printf("[forge] scene-probe complete — %s\n", ok ? "OK" : "FAILED");
        return ok;
    }

    bool renderTriangle() {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        std::printf("[forge] render: bringing up The Forge (D3D12)...\n");

        Renderer* pRenderer = nullptr;
        Queue*    pQueue = nullptr;
        if (!forgeBringUp(&pRenderer, &pQueue)) {
            return false;
        }

        Shader* pShader = loadTriangleShader(pRenderer);
        if (!pShader) {
            forgeTearDown(pRenderer, pQueue);
            return false;
        }

        const uint32_t kWidth = 640, kHeight = 360;
        std::printf("[forge] addRenderTarget %ux%u...\n", kWidth, kHeight);
        RenderTargetDesc rtDesc = {};
        rtDesc.mWidth = kWidth;
        rtDesc.mHeight = kHeight;
        rtDesc.mDepth = 1;
        rtDesc.mArraySize = 1;
        rtDesc.mMipLevels = 1;
        rtDesc.mSampleCount = SAMPLE_COUNT_1;
        rtDesc.mFormat = TinyImageFormat_R8G8B8A8_UNORM;
        rtDesc.mStartState = RESOURCE_STATE_RENDER_TARGET;
        rtDesc.mClearValue.r = 0.0f;
        rtDesc.mClearValue.g = 0.0f;
        rtDesc.mClearValue.b = 0.0f;
        rtDesc.mClearValue.a = 1.0f;
        rtDesc.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
        rtDesc.pName = "triRT";
        RenderTarget* pRT = nullptr;
        addRenderTarget(pRenderer, &rtDesc, &pRT);

        std::printf("[forge] addPipeline...\n");
        Pipeline* pPipeline = buildTrianglePipeline(pRenderer, pRT, pShader);
        bool ok = false;
        if (pPipeline) {
            ok = drawTriangleAndVerify(pRenderer, pQueue, pRT, pPipeline);
            removePipeline(pRenderer, pPipeline);
        } else {
            std::printf("[forge] addPipeline FAILED\n");
        }

        removeRenderTarget(pRenderer, pRT);
        removeShader(pRenderer, pShader);
        exitRootSignature(pRenderer);
        forgeTearDown(pRenderer, pQueue);

        std::printf("[forge] render complete — %s\n", ok ? "triangle verified" : "FAILED");
        return ok;
    }

    bool renderTriangleShared() {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        std::printf("[forge] render-shared: bringing up The Forge (D3D12)...\n");

        Renderer* pRenderer = nullptr;
        Queue*    pQueue = nullptr;
        if (!forgeBringUp(&pRenderer, &pQueue)) {
            return false;
        }

        Shader* pShader = loadTriangleShader(pRenderer);
        if (!pShader) {
            forgeTearDown(pRenderer, pQueue);
            return false;
        }

        const uint32_t kWidth = 640, kHeight = 360;

        // --- Create the CROSS-PROCESS SHARED D3D12 render target manually. Forge's
        // addRenderTarget can't do this (no CreateSharedHandle; EXPORT_BIT is a
        // no-op on D3D12), so we create the ID3D12Resource per the D3D9Ex-ingestion
        // rules and adopt it into a Forge RenderTarget via pNativeHandle.
        //   - heap flag  D3D12_HEAP_FLAG_SHARED      (shareable across processes)
        //   - format     DXGI_FORMAT_B8G8R8A8_UNORM  (strict; == D3DFMT_A8R8G8B8)
        //   - layout     D3D12_TEXTURE_LAYOUT_UNKNOWN (driver retiles legacy-readable)
        ID3D12Device* pDevice = pRenderer->mDx.pDevice;

        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC resDesc = {};
        resDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        resDesc.Alignment = 0;
        resDesc.Width = kWidth;
        resDesc.Height = kHeight;
        resDesc.DepthOrArraySize = 1;
        resDesc.MipLevels = 1;
        resDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        resDesc.SampleDesc.Count = 1;
        resDesc.SampleDesc.Quality = 0;
        resDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

        D3D12_CLEAR_VALUE clearVal = {};
        clearVal.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        clearVal.Color[0] = 0.0f;
        clearVal.Color[1] = 0.0f;
        clearVal.Color[2] = 0.0f;
        clearVal.Color[3] = 1.0f;

        std::printf("[forge] CreateCommittedResource (SHARED, B8G8R8A8_UNORM, LAYOUT_UNKNOWN)...\n");
        ID3D12Resource* pSharedRes = nullptr;
        HRESULT hr = pDevice->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_SHARED, &resDesc,
            D3D12_RESOURCE_STATE_RENDER_TARGET, &clearVal, IID_PPV_ARGS(&pSharedRes));
        if (FAILED(hr) || !pSharedRes) {
            std::printf("[forge] CreateCommittedResource FAILED 0x%08lX\n", (unsigned long)hr);
            removeShader(pRenderer, pShader);
            exitRootSignature(pRenderer);
            forgeTearDown(pRenderer, pQueue);
            return false;
        }

        std::printf("[forge] CreateSharedHandle (NT)...\n");
        HANDLE ntHandle = nullptr;
        hr = pDevice->CreateSharedHandle(pSharedRes, nullptr, GENERIC_ALL, nullptr, &ntHandle);
        if (FAILED(hr) || !ntHandle) {
            std::printf("[forge] CreateSharedHandle FAILED 0x%08lX\n", (unsigned long)hr);
            pSharedRes->Release();
            removeShader(pRenderer, pShader);
            exitRootSignature(pRenderer);
            forgeTearDown(pRenderer, pQueue);
            return false;
        }
        std::printf("[forge] shared NT handle = %p (host-process value)\n", (void*)ntHandle);

        // --- Adopt the shared resource into a Forge RenderTarget. Forge takes the
        // pointer (no AddRef) and SAFE_RELEASEs it on removeRenderTarget, so we hand
        // ownership over and must NOT Release pSharedRes ourselves.
        std::printf("[forge] addRenderTarget (adopt pNativeHandle)...\n");
        RenderTargetDesc rtDesc = {};
        rtDesc.mWidth = kWidth;
        rtDesc.mHeight = kHeight;
        rtDesc.mDepth = 1;
        rtDesc.mArraySize = 1;
        rtDesc.mMipLevels = 1;
        rtDesc.mSampleCount = SAMPLE_COUNT_1;
        rtDesc.mFormat = TinyImageFormat_B8G8R8A8_UNORM;
        rtDesc.mStartState = RESOURCE_STATE_RENDER_TARGET;
        rtDesc.mClearValue.r = 0.0f;
        rtDesc.mClearValue.g = 0.0f;
        rtDesc.mClearValue.b = 0.0f;
        rtDesc.mClearValue.a = 1.0f;
        rtDesc.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
        rtDesc.pNativeHandle = (void*)pSharedRes;
        rtDesc.pName = "sharedTriRT";
        RenderTarget* pRT = nullptr;
        addRenderTarget(pRenderer, &rtDesc, &pRT);

        std::printf("[forge] addPipeline...\n");
        Pipeline* pPipeline = buildTrianglePipeline(pRenderer, pRT, pShader);
        bool ok = false;
        if (pPipeline) {
            ok = drawTriangleAndVerify(pRenderer, pQueue, pRT, pPipeline);
            removePipeline(pRenderer, pPipeline);
        } else {
            std::printf("[forge] addPipeline FAILED\n");
        }

        CloseHandle(ntHandle);
        removeRenderTarget(pRenderer, pRT);  // releases the adopted shared resource
        removeShader(pRenderer, pShader);
        exitRootSignature(pRenderer);
        forgeTearDown(pRenderer, pQueue);

        std::printf("[forge] render-shared complete — %s\n", ok ? "shared RT verified" : "FAILED");
        return ok;
    }
}

// --- D4 live path: persistent out-of-process renderer ------------------------
namespace {
    // Persistent renderer state, alive between RenderInit and shutdown. Distinct
    // from the one-shot probes: the shared RT + pipeline + cmd infra survive across
    // renderFrame calls.
    struct LiveRenderer {
        Renderer*       pRenderer = nullptr;
        Queue*          pQueue = nullptr;
        Shader*         pShader = nullptr;
        RenderTarget*   pRT = nullptr;
        Pipeline*       pPipeline = nullptr;
        CmdPool*        pCmdPool = nullptr;
        Cmd*            pCmd = nullptr;
        Fence*          pFence = nullptr;
        ID3D12Resource* pSharedRes = nullptr;   // owned by pRT (released on removeRenderTarget)
        HANDLE          ntHandle = nullptr;     // host-process NT shared handle
        uint32_t        width = 0, height = 0;
        bool            firstFrame = true;

        // MSAA: requested sample count (1 = off, validated against the device in init).
        // >1 ⇒ the scene renders into pMSAAColor (+ MSAA pDepth) and resolves into the
        // shared single-sample pRT before the D3D9 handoff (the shared RT can't be MSAA).
        uint32_t        sampleCount = 1;
        RenderTarget*   pMSAAColor = nullptr;     // internal MSAA color (null when sampleCount==1)
        uint32_t        anisoLevel = 0;           // texture sampler max anisotropy (0 = linear); Phase 2

        // --- M1c opaque scene path (GPU-driven: one PerFrame set, structured buffer) ---
        RenderTarget*  pDepth = nullptr;          // depth buffer for the scene
        Shader*        pOpaqueShader = nullptr;
        Pipeline*      pOpaquePipeline = nullptr;        // FRONT_FACE_CCW (non-mirrored)
        Pipeline*      pOpaquePipelineMirror = nullptr;  // FRONT_FACE_CW (negative-determinant world)
        // Phase 1 depth-takeover (tasks/forge-depth-ssao.md): Z-prepass. Same opaque.vert + the
        // depth-only alpha frag (depthonly.frag), GEQUAL + depthWrite, NO colour target. Run
        // first so depth is complete before the colour pass (which switches to EQUAL + no-write).
        Shader*        pDepthOnlyShader = nullptr;
        Pipeline*      pOpaquePrepassPipeline = nullptr;        // FRONT_FACE_CCW, depth-only
        Pipeline*      pOpaquePrepassPipelineMirror = nullptr;  // FRONT_FACE_CW, depth-only
        DescriptorSet* pPerFrameSet = nullptr;    // gFrameData cbuffer (viewProj), 1 instance
        DescriptorSet* pPerBatchSet = nullptr;    // gBatch cbuffer window, kMaxBatches instances
        Buffer*        pFrameCbv = nullptr;        // gFrameData cbuffer (viewProj), persistent-mapped
        // Tier 3a point lights: per-frame light cbuffer + its own descriptor set (read by the
        // shared opaque.frag for both the static and skinned paths).
        Buffer*        pLightCbv = nullptr;        // gLights cbuffer, persistent-mapped
        DescriptorSet* pPerLightsSet = nullptr;    // gLights, 1 instance
        Buffer*        pWorldsBuf[16] = {};        // gBatch windows: one 64KB cbuffer PER batch, persistent-mapped
        // Per-batch instance-rate VB of uint2 { .x = identity DrawIndex/Base, .y = per-draw
        // texIndex }. CPU-mapped so the static loop writes texIndex per frame; .x stays identity
        // (filled once) so skinned still reads Base from .x via firstInstance.
        Buffer*        pInstanceBuf[16] = {};
        uint32_t       maxDraws = 0;

        // --- Phase 2 bindless texturing -------------------------------------------------
        // No dynamic sampler: the frag uses the FSL static sampler gSamplerAnisotropic (see
        // opaque.srt.h for why a dynamic sampler can't share the array's set).
        Texture*       pDefaultWhite = nullptr;    // gTextures[0] / fill for unloaded slots
        Texture*       pTextures[MAX_TEXTURES] = {}; // bindless base maps; unloaded == pDefaultWhite
        uint32_t       texHigh = 0;                // highest assigned slot + 1 (for teardown)
        DescriptorSet* pPersistentSet = nullptr;   // bindless gTextures[] (bound once per frame)

        // --- M-Skinning: GPU palette skinning path -----------------------------------
        // Reuses the SAME SrtData/default.rootsig as the static path: gBatch.worlds[1024]
        // is read as a 64KB BONE window (32 parts * 32 bones). Only a new skinned vertex
        // shader + layout + a parallel set of bone cbuffers + a skinned PerBatch set.
        Shader*        pSkinnedShader = nullptr;
        Pipeline*      pSkinnedPipeline = nullptr;       // FRONT_FACE_CCW (non-mirrored)
        Pipeline*      pSkinnedPipelineMirror = nullptr; // FRONT_FACE_CW (mirrored, neg-determinant bones)
        // Tier 1b Z-prepass: skinned.vert + the shared depthonly.frag (skinned VSOutput == opaque's),
        // GEQUAL + depthWrite, 0 RTs. Skinned COLOUR pipelines flip to CMP_EQUAL + no-write. The
        // depth shader MUST pair skinned.vert (not opaque.vert) with the skinned layout.
        Shader*        pSkinnedDepthShader = nullptr;
        Pipeline*      pSkinnedPrepassPipeline = nullptr;
        Pipeline*      pSkinnedPrepassPipelineMirror = nullptr;
        Buffer*        pBonesBuf[16] = {};               // bone windows: one 64KB cbuffer per window, persistent-mapped
        DescriptorSet* pPerBatchSetSkin = nullptr;       // gBatch bound to pBonesBuf[], kMaxBatches instances
        // Skinned instance-rate VB of uint2 { .x = Base (bone offset in window), .y = texIndex }.
        // Indexed by the global skinnedDrawn (0..kMaxSkinned-1) via firstInstance, so each drawn
        // part gets a UNIQUE entry — no per-part overwrite hazard (can't reuse pInstanceBuf[0],
        // which the static loop fills with its own texIndices). Written per frame in the skinned loop.
        Buffer*        pInstanceBufSkin = nullptr;

        // --- Tier 4: multi-map (dark/detail/glow) path -------------------------------
        // Wide vertex (GeomVertexWireMM, 4 UV sets) + its own pipeline. Shares the SAME
        // SrtData/default.rootsig + opaque frag-side lighting (duplicated in multimap.frag).
        // gBatch is bound to ONE 64KB world window (kMaxMultiMap=256 <= 1024 matrices).
        Shader*        pMultiMapShader = nullptr;
        Pipeline*      pMultiMapPipeline = nullptr;        // FRONT_FACE_CCW (non-mirrored)
        Pipeline*      pMultiMapPipelineMirror = nullptr;  // FRONT_FACE_CW (mirrored world)
        // Tier 1b Z-prepass: multimap.vert + depthonly_mm.frag (own VSOutput, base-stage alpha
        // test), GEQUAL + depthWrite, 0 RTs. Multi-map COLOUR pipelines flip to CMP_EQUAL + no-write.
        Shader*        pMultiMapDepthShader = nullptr;
        Pipeline*      pMultiMapPrepassPipeline = nullptr;
        Pipeline*      pMultiMapPrepassPipelineMirror = nullptr;
        Buffer*        pMMWorldsBuf = nullptr;             // gBatch: one 64KB world window, persistent-mapped
        DescriptorSet* pPerBatchSetMM = nullptr;           // gBatch bound to pMMWorldsBuf, 1 instance
        // Per-draw instance VB (kMMInstU32 uint32 slots): { Meta, stages[4], matDiff3, matAmb3, matEmis3 },
        // one entry per drawn part (indexed by multiMapDrawn via firstInstance). CPU-mapped, per frame.
        Buffer*        pInstanceBufMM = nullptr;

        // --- Buffer consolidation: one shared mega VB + IB for STATIC non-skinned parts ----
        // Kills the per-draw VB/IB binds (the DX9-shaped bottleneck): bind these ONCE, draw each
        // part with firstVertex(BaseVertexLocation)/firstIndex offsets into them. Free-list
        // suballocated so streaming geometry can add/free regions as the player moves.
        Buffer*        pArenaVB = nullptr;   // GPU_ONLY, GeomVertexWire (stride 32)
        Buffer*        pArenaIB = nullptr;   // GPU_ONLY, UINT16 indices

        // GPU-driven draws: per-frame indirect-argument buffer (one IndirectDrawIndexArguments
        // per static arena part). The ~2667 cmdDrawIndexedInstanced API calls collapse into a
        // handful of cmdExecuteIndirect (one per (mirror,batch) group) — kills the per-draw-CALL
        // CPU record cost. CPU_TO_GPU upload heap = GENERIC_READ (includes INDIRECT_ARGUMENT),
        // so no per-frame barrier; single-buffered (host render is lockstep, waitForFences gates).
        Buffer*        pIndirectArgs = nullptr;
    };
    LiveRenderer g_live;

    // Mega-arena sizes. Holds the resident static-opaque working set (engine ~100m + MGE LOD
    // ~16 cells). Generous; arena-full logs + skips (no fallback). True eviction across a
    // 40000-cell world needs a client free-slot signal (follow-up); the free-list already
    // reclaims on re-upload/shape-change.
    constexpr uint64_t kArenaVBBytes = 256ull << 20;   // 256 MB
    constexpr uint64_t kArenaIBBytes = 64ull << 20;    // 64 MB

    // First-fit free-list suballocator over a fixed byte arena (the mega VB or IB). VB allocs
    // are vertexCount*32, IB allocs indexCount*2 — both inherently aligned — so region offsets
    // stay aligned without explicit padding. free() coalesces adjacent regions.
    struct FreeList {
        struct Region { uint64_t off; uint64_t len; };
        std::vector<Region> regions;     // sorted ascending by off; non-adjacent (coalesced)
        uint64_t total = 0;

        void init(uint64_t size) { regions.assign(1, {0, size}); total = size; }

        // Returns byte offset, or UINT64_MAX if no region fits.
        uint64_t alloc(uint64_t len) {
            if (len == 0) return UINT64_MAX;
            for (size_t i = 0; i < regions.size(); ++i) {
                if (regions[i].len >= len) {
                    uint64_t off = regions[i].off;
                    if (regions[i].len == len) { regions.erase(regions.begin() + i); }
                    else { regions[i].off += len; regions[i].len -= len; }
                    return off;
                }
            }
            return UINT64_MAX;
        }

        void release(uint64_t off, uint64_t len) {   // NOT 'free' — that's a Forge IMemory macro
            if (len == 0) return;
            // insert sorted, then coalesce with neighbours
            size_t i = 0;
            while (i < regions.size() && regions[i].off < off) ++i;
            regions.insert(regions.begin() + i, {off, len});
            // coalesce with next
            if (i + 1 < regions.size() && regions[i].off + regions[i].len == regions[i + 1].off) {
                regions[i].len += regions[i + 1].len;
                regions.erase(regions.begin() + i + 1);
            }
            // coalesce with prev
            if (i > 0 && regions[i - 1].off + regions[i - 1].len == regions[i].off) {
                regions[i - 1].len += regions[i].len;
                regions.erase(regions.begin() + i);
            }
        }
    };
    FreeList g_arenaVB;   // suballocates pArenaVB (bytes)
    FreeList g_arenaIB;   // suballocates pArenaIB (bytes)

    // Per-draw transform: the PROVEN column-major cbuffer convention, BATCHED to beat the
    // 64KB cbuffer cap. CRITICAL: a UNIFORM_BUFFER resource > 64KB makes Forge build an
    // oversized full CBV (> D3D12's 65536 max) which REMOVES THE DEVICE — so each batch is
    // its OWN exactly-64KB cbuffer (a valid full CBV, no sub-ranges). kMaxBatches separate
    // 64KB buffers hold kBatchSize matrices each; a PerBatch descriptor set instance b binds
    // buffer b. Draws issue in batches of kBatchSize, the per-instance DrawIndex selecting
    // within the bound window. (This is why every "descriptor count" scaling attempt failed:
    // the backing buffer crossed 64KB, not the descriptor count.)
    constexpr uint32_t kBatchSize  = 1024;                 // matrices per 64KB cbuffer (must match OPAQUE_BATCH)
    constexpr uint32_t kBatchBytes = kBatchSize * 64;      // 65536 = exactly the D3D12 CBV max
    constexpr uint32_t kMaxBatches = 8;                    // 8 * 1024 = 8192 draws/frame
    constexpr uint32_t kMaxDraws   = kBatchSize * kMaxBatches;

    // M-Skinning palette packing: a FIXED 32-matrix stride per skinned part (kMaxBonesPerPart,
    // matches the cache's kMaxBones). One 64KB bone window (float4x4[1024]) holds 1024/32 = 32
    // parts; reusing the kMaxBatches windows → 256 skinned parts/frame. base = (p%32)*32 ∈
    // {0,32,...,992} < 1024, so the identity instance buffer gives instanceBuf[base]==base.
    constexpr uint32_t kMaxBonesPerPart = 32;
    constexpr uint32_t kSkinnedPerWindow = kBatchSize / kMaxBonesPerPart;   // 32
    constexpr uint32_t kMaxSkinned = kSkinnedPerWindow * kMaxBatches;       // 256

    // Tier 4 multi-map: ONE 64KB world window holds kBatchSize(1024) matrices; cap at 256
    // parts/frame (< 1024 → a single window, no batching). Per-instance VB stride (uint32 slots):
    //   [0] Meta (DrawIndex|stageCount|vColSource|alphaRef)   [1..4] stages[4]
    //   [5..7] matDiffuse.rgb   [8..10] matAmbient.rgb   [11..13] matEmissive.rgb
    // 14 * 4 = 56 bytes. Rare geometry → a flat 256-cap (log-drop, no fallback) is ample.
    constexpr uint32_t kMaxMultiMap = 256;
    constexpr uint32_t kMMInstU32   = 14;

    // Tier 3a point lights: per-frame cbuffer of MAX_POINT_LIGHTS lights (3 float4 each) +
    // a float4 header (count). MUST match IPC::kMaxPointLights / MAX_POINT_LIGHTS (opaque.srt.h).
    // 16 + 128*3*16 = 6160 B, rounded up to a 256-byte CBV multiple.
    constexpr uint32_t kMaxPointLights = 128;
    constexpr uint32_t kLightCbvBytes  = ((16 + kMaxPointLights * 3 * 16) + 255) & ~255u;  // 6400

    // Bindless base-map texture array size (must match MAX_TEXTURES in opaque.srt.h).
    constexpr uint32_t kMaxTextures = MAX_TEXTURES;

    // Static per-instance VB stride (uint32 slots). Tier 2b grew it past the old uint2:
    //   [0] DrawIndex (identity, set once)   [1] TexAlpha (tex|alphaRef|vColSource, per-frame)
    //   [2..4] matDiffuse.rgb (float)        [5..7] matAmbient.rgb (float)   [8..10] matEmissive.rgb (float)
    //   [11] OverlayIndex (terrain DECAL_1 bindless slot, 0 = no decal; per-frame)
    // 12 * 4 = 48 bytes. The material + overlay ride the instance VB (not a new cbuffer/descriptor
    // set) to avoid the FSL descriptor-offset gotcha that hoisted the sampler — see opaque.srt.h.
    constexpr uint32_t kStaticInstU32 = 12;

    // Pack the per-draw instance .y: texIndex in the low 16 bits (slots < kMaxTextures=1024,
    // so ≤10 bits), the alpha-test reference quantised to a byte in bits 16-23, and the
    // vertex-colour routing (0/1/2) in bits 24-25. The vert shaders unpack: TexIndex = packed
    // & 0xFFFF, AlphaRef = ((packed>>16)&0xFF)/255, VColSource = (packed>>24)&0x3. alphaRef 0 →
    // frag's strict a<ref never discards (opaque-safe). See opaque.vert/skinned.vert/opaque.frag.
    inline uint32_t packTexAlpha(uint32_t texIndex, float alphaRef, uint32_t vColSource = 0u) {
        const uint32_t tex = texIndex < kMaxTextures ? texIndex : 0u;
        float r = alphaRef < 0.0f ? 0.0f : (alphaRef > 1.0f ? 1.0f : alphaRef);
        const uint32_t aref = (uint32_t)(r * 255.0f + 0.5f) & 0xFFu;
        return tex | (aref << 16) | ((vColSource & 0x3u) << 24);
    }

    // Submit + wait the resource loader's UPLOAD ENGINE. beginUpdateResource/endUpdateResource
    // record texture copies on the upload engine (pUploadEngines), which waitForAllResourceLoads
    // (async loader) does NOT flush — without this the copies never execute and textures stay
    // black. flushResourceUpdates does streamerFlush + returns the copy fence to wait on.
    void flushTextureUploads(Renderer* R) {
        FlushResourceUpdateDesc fd = {};
        flushResourceUpdates(&fd);
        if (fd.pOutFence) {
            waitForFences(R, 1, &fd.pOutFence);
        }
    }

    // Build the M1c opaque scene path: depth target, opaque shader + pipeline (pos+normal
    // per-vertex + per-instance DrawIndex layout, depth test/write), the viewProj CBV +
    // gFrameData cbuffer (viewProj) + gWorlds structured buffer (one PerFrame set), and the identity
    // instance buffer. The opaque shaders share the global default.rootsig (regenerated from the
    // SRT in opaque.srt.h). On any failure tears down what it made and returns false
    // (the triangle path stays usable).
    bool buildOpaquePath(Renderer* R, uint32_t width, uint32_t height) {
        // Depth target (reverse-Z not needed for M1c; standard LEQUAL + clear to 1.0).
        RenderTargetDesc dDesc = {};
        dDesc.mWidth = width;
        dDesc.mHeight = height;
        dDesc.mDepth = 1;
        dDesc.mArraySize = 1;
        dDesc.mMipLevels = 1;
        dDesc.mSampleCount = (SampleCount)g_live.sampleCount;   // MSAA depth must match the color RT
        dDesc.mFormat = TinyImageFormat_D32_SFLOAT;
        dDesc.mStartState = RESOURCE_STATE_DEPTH_WRITE;
        // REVERSE-Z: clear to 0.0 (the far plane). With a float32 depth buffer, reverse-Z
        // gives near-uniform precision across the whole range — the fix for "different-but-
        // close object" z-fighting (standard-Z + D32 starves precision near the far plane,
        // exactly where MW's huge far/near ratio bites). Paired with CMP_GEQUAL below and a
        // viewProj that maps near->1 / far->0 (renderScene applies it). near is now 1.0.
        dDesc.mClearValue.depth = 0.0f;
        dDesc.mClearValue.stencil = 0;
        dDesc.pName = "sceneDepth";
        addRenderTarget(R, &dDesc, &g_live.pDepth);
        if (!g_live.pDepth) {
            return false;
        }

        // MSAA: internal multisampled color target. The scene renders here; it's resolved
        // into the shared single-sample pRT in renderScene. Only when sampleCount > 1 — at 1x
        // the scene renders straight into pRT exactly as before (pMSAAColor stays null).
        if (g_live.sampleCount > 1) {
            RenderTargetDesc cDesc = {};
            cDesc.mWidth = width;
            cDesc.mHeight = height;
            cDesc.mDepth = 1;
            cDesc.mArraySize = 1;
            cDesc.mMipLevels = 1;
            cDesc.mSampleCount = (SampleCount)g_live.sampleCount;
            cDesc.mFormat = TinyImageFormat_B8G8R8A8_UNORM;   // same as pRT → resolve is format-compatible
            cDesc.mStartState = RESOURCE_STATE_RENDER_TARGET;
            cDesc.mClearValue.r = 0.0f;
            cDesc.mClearValue.g = 0.0f;
            cDesc.mClearValue.b = 0.0f;
            cDesc.mClearValue.a = 0.0f;   // transparent bg: resolves into pRT's coverage-mask alpha
            cDesc.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
            cDesc.pName = "sceneMSAAColor";
            addRenderTarget(R, &cDesc, &g_live.pMSAAColor);
            if (!g_live.pMSAAColor) {
                std::printf("[forge] addRenderTarget(MSAA color %ux) FAILED\n", g_live.sampleCount);
                return false;
            }
        }

        // Opaque shaders (share the global root signature already loaded for the triangle).
        ShaderLoadDesc sDesc = {};
        sDesc.mVert.pFileName = "opaque.vert";
        sDesc.mFrag.pFileName = "opaque.frag";
        addShader(R, &sDesc, &g_live.pOpaqueShader);
        if (!g_live.pOpaqueShader) {
            std::printf("[forge] addShader(opaque) FAILED\n");
            return false;
        }

        // Vertex layout: binding 0 = mesh (IPC::GeomVertexWire, per-vertex: pos@0,
        // normal@12, UV@24, color@32, stride 36); binding 1 = per-INSTANCE DrawIndex (uint,
        // stride 4), fed by the identity instance buffer + firstInstance to index gWorlds.
        // UV takes TEXCOORD0, so DrawIndex moved to TEXCOORD1 (matches opaque.vert).
        VertexLayout vl = {};
        vl.mBindingCount = 2;
        vl.mBindings[0].mStride = sizeof(IPC::GeomVertexWire);
        vl.mBindings[0].mRate = VERTEX_BINDING_RATE_VERTEX;
        vl.mBindings[1].mStride = kStaticInstU32 * sizeof(uint32_t);   // {DrawIndex, TexAlpha, matDiff3, matAmb3, matEmis3}
        vl.mBindings[1].mRate = VERTEX_BINDING_RATE_INSTANCE;
        vl.mAttribCount = 10;
        vl.mAttribs[0].mSemantic = SEMANTIC_POSITION;
        vl.mAttribs[0].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
        vl.mAttribs[0].mBinding = 0;
        vl.mAttribs[0].mLocation = 0;
        vl.mAttribs[0].mOffset = 0;
        vl.mAttribs[1].mSemantic = SEMANTIC_NORMAL;
        vl.mAttribs[1].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
        vl.mAttribs[1].mBinding = 0;
        vl.mAttribs[1].mLocation = 1;
        vl.mAttribs[1].mOffset = 12;
        vl.mAttribs[2].mSemantic = SEMANTIC_TEXCOORD0;       // UV (per-vertex)
        vl.mAttribs[2].mFormat = TinyImageFormat_R32G32_SFLOAT;
        vl.mAttribs[2].mBinding = 0;
        vl.mAttribs[2].mLocation = 2;
        vl.mAttribs[2].mOffset = 24;
        vl.mAttribs[3].mSemantic = SEMANTIC_COLOR;           // per-vertex colour (DiffAmb)
        vl.mAttribs[3].mFormat = TinyImageFormat_B8G8R8A8_UNORM;   // D3DCOLOR byte order (B,G,R,A)
        vl.mAttribs[3].mBinding = 0;
        vl.mAttribs[3].mLocation = 3;
        vl.mAttribs[3].mOffset = 32;
        vl.mAttribs[4].mSemantic = SEMANTIC_TEXCOORD1;       // DrawIndex (per-instance .x)
        vl.mAttribs[4].mFormat = TinyImageFormat_R32_UINT;
        vl.mAttribs[4].mBinding = 1;
        vl.mAttribs[4].mLocation = 4;
        vl.mAttribs[4].mOffset = 0;
        vl.mAttribs[5].mSemantic = SEMANTIC_TEXCOORD2;       // TexAlpha (per-instance .y: tex|alphaRef|vColSource)
        vl.mAttribs[5].mFormat = TinyImageFormat_R32_UINT;
        vl.mAttribs[5].mBinding = 1;
        vl.mAttribs[5].mLocation = 5;
        vl.mAttribs[5].mOffset = sizeof(uint32_t);
        vl.mAttribs[6].mSemantic = SEMANTIC_TEXCOORD3;       // matDiffuse.rgb (per-instance)
        vl.mAttribs[6].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
        vl.mAttribs[6].mBinding = 1;
        vl.mAttribs[6].mLocation = 6;
        vl.mAttribs[6].mOffset = 2 * sizeof(uint32_t);
        vl.mAttribs[7].mSemantic = SEMANTIC_TEXCOORD4;       // matAmbient.rgb (per-instance)
        vl.mAttribs[7].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
        vl.mAttribs[7].mBinding = 1;
        vl.mAttribs[7].mLocation = 7;
        vl.mAttribs[7].mOffset = 5 * sizeof(uint32_t);
        vl.mAttribs[8].mSemantic = SEMANTIC_TEXCOORD5;       // matEmissive.rgb (per-instance)
        vl.mAttribs[8].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
        vl.mAttribs[8].mBinding = 1;
        vl.mAttribs[8].mLocation = 8;
        vl.mAttribs[8].mOffset = 8 * sizeof(uint32_t);
        vl.mAttribs[9].mSemantic = SEMANTIC_TEXCOORD6;       // OverlayIndex (per-instance: terrain DECAL_1 slot)
        vl.mAttribs[9].mFormat = TinyImageFormat_R32_UINT;
        vl.mAttribs[9].mBinding = 1;
        vl.mAttribs[9].mLocation = 9;
        vl.mAttribs[9].mOffset = 11 * sizeof(uint32_t);

        // COLOUR-pass depth (Phase 1 early-Z): the Z-prepass already wrote every opaque pixel's
        // depth, so the colour pass only MATCHES it — CMP_EQUAL + depthWrite OFF = true early-Z,
        // zero shaded overdraw. Correct only because prepass and colour share opaque.vert, so
        // SV_Position is bit-identical (same matrices + reverse-Z post-mul). See the prepass
        // pipeline below (GEQUAL + write) and tasks/forge-depth-ssao.md.
        DepthStateDesc depthDesc = {};
        depthDesc.mDepthTest = true;
        depthDesc.mDepthWrite = false;
        depthDesc.mDepthFunc = CMP_EQUAL;

        RasterizerStateDesc rasterDesc = {};
        // Backface culling — the proven D3D9 cache color pass culls (drawEntry: mirrored ?
        // D3DCULL_CCW : D3DCULL_CW). CULL_MODE_NONE drew BOTH faces of every thin/double-
        // sided mesh → two near-coincident surfaces → z-fighting on movers (thin geometry),
        // statics (thick) unaffected. Same VB/IB + viewProj as the proven path; the scene
        // already renders correctly oriented, so front faces are CCW → keep CCW, cull back.
        // (Mirrored / negative-determinant parts will show inside-out until a per-draw mirror
        // pipeline is added — a minority; the global winding is validated by this build.)
        rasterDesc.mCullMode = CULL_MODE_BACK;
        rasterDesc.mFrontFace = FRONT_FACE_CCW;

        PipelineDesc pd = {};
        pd.mType = PIPELINE_TYPE_GRAPHICS;
        GraphicsPipelineDesc& g = pd.mGraphicsDesc;
        g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
        g.mRenderTargetCount = 1;
        g.pColorFormats = &g_live.pRT->mFormat;   // B8G8R8A8 — same for pRT and the MSAA color
        g.mSampleCount = (SampleCount)g_live.sampleCount;
        g.mSampleQuality = 0;
        g.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
        g.pDepthState = &depthDesc;
        g.pVertexLayout = &vl;
        g.pRasterizerState = &rasterDesc;
        g.pShaderProgram = g_live.pOpaqueShader;
        addPipeline(R, &pd, &g_live.pOpaquePipeline);
        if (!g_live.pOpaquePipeline) {
            std::printf("[forge] addPipeline(opaque) FAILED\n");
            return false;
        }

        // Mirror variant: same pipeline with the OPPOSITE front face (CW). Negative-
        // determinant (mirrored) world transforms flip triangle winding, so their front
        // faces become CW; without this they'd cull the wrong face and render INSIDE-OUT
        // (matches the proven D3D9 drawEntry's `mirrored ? D3DCULL_CCW : D3DCULL_CW`). The
        // draw loop picks per draw by det(world) sign — no IPC wire change needed.
        RasterizerStateDesc rasterMirror = rasterDesc;
        rasterMirror.mFrontFace = FRONT_FACE_CW;
        g.pRasterizerState = &rasterMirror;
        addPipeline(R, &pd, &g_live.pOpaquePipelineMirror);
        g.pRasterizerState = &rasterDesc;   // restore for any later use of pd
        if (!g_live.pOpaquePipelineMirror) {
            std::printf("[forge] addPipeline(opaque mirror) FAILED\n");
            return false;
        }

        // --- Phase 1 Z-prepass pipelines (opaque.vert + depthonly.frag) -----------------
        // Depth-only: GEQUAL + depthWrite ON, NO colour target (mRenderTargetCount = 0, the
        // Forge depth-pass convention). Reuses the SAME vertex layout (vl) + rasterizer states
        // as the colour pipelines, so SV_Position is bit-identical → the colour pass's CMP_EQUAL
        // matches. Runs first in renderScene to complete opaque depth before any shading.
        {
            ShaderLoadDesc dpDesc = {};
            dpDesc.mVert.pFileName = "opaque.vert";
            dpDesc.mFrag.pFileName = "depthonly.frag";
            addShader(R, &dpDesc, &g_live.pDepthOnlyShader);
            if (!g_live.pDepthOnlyShader) {
                std::printf("[forge] addShader(depthonly) FAILED\n");
                return false;
            }
            DepthStateDesc preDepth = {};
            preDepth.mDepthTest = true;
            preDepth.mDepthWrite = true;
            preDepth.mDepthFunc = CMP_GEQUAL;   // REVERSE-Z write — fills depth for the colour EQUAL test

            PipelineDesc ppd = {};
            ppd.mType = PIPELINE_TYPE_GRAPHICS;
            GraphicsPipelineDesc& pg = ppd.mGraphicsDesc;
            pg.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
            pg.mRenderTargetCount = 0;          // depth-only — no colour attachment
            pg.pColorFormats = nullptr;
            pg.mSampleCount = (SampleCount)g_live.sampleCount;
            pg.mSampleQuality = 0;
            pg.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
            pg.pDepthState = &preDepth;
            pg.pVertexLayout = &vl;
            pg.pRasterizerState = &rasterDesc;   // CCW (non-mirrored)
            pg.pShaderProgram = g_live.pDepthOnlyShader;
            addPipeline(R, &ppd, &g_live.pOpaquePrepassPipeline);
            if (!g_live.pOpaquePrepassPipeline) {
                std::printf("[forge] addPipeline(opaque prepass) FAILED\n");
                return false;
            }
            pg.pRasterizerState = &rasterMirror;   // CW (negative-determinant world)
            addPipeline(R, &ppd, &g_live.pOpaquePrepassPipelineMirror);
            if (!g_live.pOpaquePrepassPipelineMirror) {
                std::printf("[forge] addPipeline(opaque prepass mirror) FAILED\n");
                return false;
            }
        }

        // gFrameData: small viewProj cbuffer, persistent-mapped (256 = min CBV size).
        BufferLoadDesc fb = {};
        fb.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        fb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        fb.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        fb.mDesc.mSize = 256;
        fb.mDesc.pName = "frameCbv";
        fb.pData = nullptr;
        fb.ppBuffer = &g_live.pFrameCbv;
        addResource(&fb, nullptr);

        // Tier 3a: gLights cbuffer, persistent-mapped (kLightCbvBytes ~6.4KB < 64KB CBV max).
        BufferLoadDesc lb = {};
        lb.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        lb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        lb.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        lb.mDesc.mSize = kLightCbvBytes;
        lb.mDesc.pName = "lightCbv";
        lb.pData = nullptr;
        lb.ppBuffer = &g_live.pLightCbv;
        addResource(&lb, nullptr);

        // World buffers: ONE exactly-64KB cbuffer per batch (a valid full CBV — a single
        // >64KB uniform buffer removes the device). Each holds kBatchSize float4x4.
        for (uint32_t b = 0; b < kMaxBatches; ++b) {
            BufferLoadDesc wb = {};
            wb.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            wb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            wb.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            wb.mDesc.mSize = kBatchBytes;
            wb.mDesc.pName = "worldBatchCbv";
            wb.pData = nullptr;
            wb.ppBuffer = &g_live.pWorldsBuf[b];
            addResource(&wb, nullptr);
            if (b == 0) {
                // Probe the BufferDesc fields that decide CBV-vs-SRV + element layout — the
                // fields that distinguish this 64KB UNIFORM_BUFFER (CBV) workaround from the
                // structured-buffer (SRV) path that would lift the per-batch cap entirely.
                std::printf("[forge] worldBuf[0] desc: mDescriptors=0x%X mSize=%llu mMemoryUsage=%d mStructStride=%u mElementCount=%u\n",
                            (unsigned)wb.mDesc.mDescriptors, (unsigned long long)wb.mDesc.mSize,
                            (int)wb.mDesc.mMemoryUsage, (unsigned)wb.mDesc.mStructStride,
                            (unsigned)wb.mDesc.mElementCount);
                LOGF(eINFO, "[forge] worldBuf[0] desc: mDescriptors=0x%X mSize=%llu mMemoryUsage=%d mStructStride=%u mElementCount=%u",
                     (unsigned)wb.mDesc.mDescriptors, (unsigned long long)wb.mDesc.mSize,
                     (int)wb.mDesc.mMemoryUsage, (unsigned)wb.mDesc.mStructStride,
                     (unsigned)wb.mDesc.mElementCount);
            }
        }

        // Per-batch instance buffers (kStaticInstU32 uint32 slots/entry): { [0] identity DrawIndex,
        // [1] per-draw TexAlpha, [2..10] per-draw material rgb }. CPU-mapped: the static loop writes
        // slots [1..10] each frame; [0] is initialised to the identity here and never clobbered, so a
        // draw with firstInstance=l reads DrawIndex = l.
        for (uint32_t b = 0; b < kMaxBatches; ++b) {
            BufferLoadDesc ib = {};
            ib.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
            ib.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            ib.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            ib.mDesc.mSize = (uint64_t)kBatchSize * kStaticInstU32 * sizeof(uint32_t);
            ib.mDesc.pName = "instanceVB";
            ib.pData = nullptr;
            ib.ppBuffer = &g_live.pInstanceBuf[b];
            addResource(&ib, nullptr);
        }

        // Per-frame indirect-args buffer (GPU-driven draws). One IndirectDrawIndexArguments
        // (20 B) per arena part, up to kMaxDraws. CPU_TO_GPU persistent-mapped: written each
        // frame, read by cmdExecuteIndirect (GENERIC_READ covers INDIRECT_ARGUMENT — no barrier).
        {
            BufferLoadDesc ad = {};
            ad.mDesc.mDescriptors = DESCRIPTOR_TYPE_INDIRECT_BUFFER;
            ad.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            ad.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            ad.mDesc.mSize = (uint64_t)kMaxDraws * sizeof(IndirectDrawIndexArguments);
            ad.mDesc.pName = "indirectArgs";
            ad.pData = nullptr;
            ad.ppBuffer = &g_live.pIndirectArgs;
            addResource(&ad, nullptr);
        }

        waitForAllResourceLoads();
        if (!g_live.pFrameCbv || !g_live.pLightCbv || !g_live.pWorldsBuf[0] || !g_live.pInstanceBuf[0]
            || !g_live.pIndirectArgs) {
            return false;
        }
        // Zero the light cbuffer so a frame with no lightBlob (lightParams.x = 0) does nothing.
        std::memset(g_live.pLightCbv->pCpuMappedAddress, 0, kLightCbvBytes);
        for (uint32_t b = 0; b < kMaxBatches; ++b) {
            uint32_t* inst = (uint32_t*)g_live.pInstanceBuf[b]->pCpuMappedAddress;
            for (uint32_t i = 0; i < kBatchSize; ++i) {
                inst[i * kStaticInstU32 + 0] = i;   // [0] identity (DrawIndex)
                for (uint32_t k = 1; k < kStaticInstU32; ++k)
                    inst[i * kStaticInstU32 + k] = 0;   // [1] TexAlpha + [2..10] material (per-frame)
            }
        }
        g_live.maxDraws = kMaxDraws;

        // PerFrame set (1 instance): gFrameData viewProj.
        DescriptorSetDesc pfDesc = SRT_SET_DESC(SrtData, PerFrame, 1, 0);
        addDescriptorSet(R, &pfDesc, &g_live.pPerFrameSet);
        // Point-light set (1 instance): gLights cbuffer (Tier 3a) — rides the PerDraw frequency.
        DescriptorSetDesc plDesc = SRT_SET_DESC(SrtData, PerDraw, 1, 0);
        addDescriptorSet(R, &plDesc, &g_live.pPerLightsSet);
        // PerBatch set (kMaxBatches instances): gBatch = window b of the world buffer.
        DescriptorSetDesc pbDesc = SRT_SET_DESC(SrtData, PerBatch, kMaxBatches, 0);
        addDescriptorSet(R, &pbDesc, &g_live.pPerBatchSet);
        // Persistent set (1 instance): bindless gTextures[] (sampler is static, in the root sig).
        DescriptorSetDesc psDesc = SRT_SET_DESC(SrtData, Persistent, 1, 0);
        addDescriptorSet(R, &psDesc, &g_live.pPersistentSet);
        if (!g_live.pPerFrameSet || !g_live.pPerLightsSet || !g_live.pPerBatchSet || !g_live.pPersistentSet) {
            return false;
        }
        {
            DescriptorData p = {};
            p.mIndex = SRT_RES_IDX(SrtData, PerFrame, gFrameData);
            p.ppBuffers = &g_live.pFrameCbv;
            updateDescriptorSet(R, 0, g_live.pPerFrameSet, 1, &p);
        }
        {
            DescriptorData p = {};
            p.mIndex = SRT_RES_IDX(SrtData, PerDraw, gLights);
            p.ppBuffers = &g_live.pLightCbv;
            updateDescriptorSet(R, 0, g_live.pPerLightsSet, 1, &p);
        }

        // --- Phase 2: bindless texture array (default white), bound once. ---
        // No dynamic sampler: the frag samples with the FSL static sampler gSamplerAnisotropic
        // (anisotropic 8x, WRAP, baked into the root sig). A dynamic sampler sharing the Persistent
        // set with the 1024-entry array gets reflected mOffset=1024 (FSL single per-set counter),
        // so its bind lands at sampler-table slot 1024 while the shader reads s0 (null sampler =>
        // point filter + tiled-black). See opaque.srt.h / opaque.frag.fsl.
        {
            // 4x4 white default for gTextures[0] and every not-yet-uploaded slot. MUST be >= 4x4:
            // a 1x1 texture sampled through the bindless array with the static ANISOTROPIC sampler
            // returns BLACK on this stack (verified via the --forge-scene slot-0 probe — 1x1 gave
            // CENTRE 0,0,0, 4x4 gave 160). So skinned parts (TexIndex 0) and oversize-texture
            // fallbacks (resolveTextureSlot -> 0) rendered black until this was bumped to 4x4.
            TextureDesc wd = {};
            wd.mWidth = 4; wd.mHeight = 4; wd.mDepth = 1;
            wd.mArraySize = 1; wd.mMipLevels = 1;
            wd.mSampleCount = SAMPLE_COUNT_1;
            wd.mFormat = TinyImageFormat_R8G8B8A8_UNORM;
            wd.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
            wd.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
            wd.pName = "defaultWhite";
            TextureLoadDesc wld = {};
            wld.ppTexture = &g_live.pDefaultWhite;
            wld.pDesc = &wd;
            addResource(&wld, nullptr);
            waitForAllResourceLoads();
            if (!g_live.pDefaultWhite) {
                std::printf("[forge] default texture creation FAILED\n");
                return false;
            }
            {
                TextureUpdateDesc wu = {};
                wu.pTexture = g_live.pDefaultWhite;
                wu.mBaseMipLevel = 0; wu.mMipLevels = 1;
                wu.mBaseArrayLayer = 0; wu.mLayerCount = 1;
                wu.mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;   // created in & returned to this state
                beginUpdateResource(&wu);
                TextureSubresourceUpdate s = wu.getSubresourceUpdateDesc(0, 0);
                for (uint32_t row = 0; row < s.mRowCount; ++row) {
                    uint32_t* dst = (uint32_t*)(s.pMappedData + (size_t)row * s.mDstRowStride);
                    for (uint32_t px = 0; px < 4; ++px) { dst[px] = 0xFFFFFFFFu; }
                }
                endUpdateResource(&wu);
                flushTextureUploads(R);   // submit the upload-engine copy (else texture stays black)
            }

            for (uint32_t i = 0; i < kMaxTextures; ++i) {
                g_live.pTextures[i] = g_live.pDefaultWhite;
            }
            g_live.texHigh = 0;

            // Bind the whole bindless array (every slot = default white for now). Individual
            // slots are re-bound as textures upload (uploadTextures, via mArrayOffset).
            DescriptorData td = {};
            td.mIndex = SRT_RES_IDX(SrtData, Persistent, gTextures);
            td.mCount = kMaxTextures;
            td.ppTextures = g_live.pTextures;
            updateDescriptorSet(R, 0, g_live.pPersistentSet, 1, &td);
        }

        for (uint32_t b = 0; b < kMaxBatches; ++b) {
            DescriptorData p = {};
            p.mIndex = SRT_RES_IDX(SrtData, PerBatch, gBatch);
            p.ppBuffers = &g_live.pWorldsBuf[b];   // full 64KB buffer = the CBV (no sub-range)
            updateDescriptorSet(R, b, g_live.pPerBatchSet, 1, &p);
        }

        // --- M-Skinning: skinned shader + pipelines + bone cbuffers + descriptor set ---
        // Shares the global default.rootsig (same SrtData) and opaque.frag; only the vertex
        // shader + layout differ. gBatch is bound to the bone windows (pBonesBuf) via a
        // separate PerBatch descriptor set (pPerBatchSetSkin).
        {
            ShaderLoadDesc skDesc = {};
            skDesc.mVert.pFileName = "skinned.vert";
            skDesc.mFrag.pFileName = "opaque.frag";   // reuse flat N.L shading
            addShader(R, &skDesc, &g_live.pSkinnedShader);
            if (!g_live.pSkinnedShader) {
                std::printf("[forge] addShader(skinned) FAILED\n");
                return false;
            }

            // Skinned vertex layout: binding 0 = mesh (IPC::SkinnedVertexWire, stride 52:
            // pos@0, normal@12, weights@24 float4, indices@40 UBYTE4→R8G8B8A8_UINT, uv@44);
            // binding 1 = per-INSTANCE uint2 { Base @0 (palette offset in the bound window),
            // texIndex @4 }, fed by pInstanceBufSkin + firstInstance=skinnedDrawn.
            VertexLayout svl = {};
            svl.mBindingCount = 2;
            svl.mBindings[0].mStride = sizeof(IPC::SkinnedVertexWire);
            svl.mBindings[0].mRate = VERTEX_BINDING_RATE_VERTEX;
            svl.mBindings[1].mStride = 2 * sizeof(uint32_t);   // instance uint2; Base @0, texIndex @4
            svl.mBindings[1].mRate = VERTEX_BINDING_RATE_INSTANCE;
            svl.mAttribCount = 7;
            svl.mAttribs[0].mSemantic = SEMANTIC_POSITION;
            svl.mAttribs[0].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
            svl.mAttribs[0].mBinding = 0;
            svl.mAttribs[0].mLocation = 0;
            svl.mAttribs[0].mOffset = 0;
            svl.mAttribs[1].mSemantic = SEMANTIC_NORMAL;
            svl.mAttribs[1].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
            svl.mAttribs[1].mBinding = 0;
            svl.mAttribs[1].mLocation = 1;
            svl.mAttribs[1].mOffset = 12;
            svl.mAttribs[2].mSemantic = SEMANTIC_TEXCOORD0;       // Weights (float4)
            svl.mAttribs[2].mFormat = TinyImageFormat_R32G32B32A32_SFLOAT;
            svl.mAttribs[2].mBinding = 0;
            svl.mAttribs[2].mLocation = 2;
            svl.mAttribs[2].mOffset = 24;
            svl.mAttribs[3].mSemantic = SEMANTIC_TEXCOORD1;       // BoneIdx (UBYTE4 → uint4)
            svl.mAttribs[3].mFormat = TinyImageFormat_R8G8B8A8_UINT;
            svl.mAttribs[3].mBinding = 0;
            svl.mAttribs[3].mLocation = 3;
            svl.mAttribs[3].mOffset = 40;
            svl.mAttribs[4].mSemantic = SEMANTIC_TEXCOORD2;       // Base (per-instance uint)
            svl.mAttribs[4].mFormat = TinyImageFormat_R32_UINT;
            svl.mAttribs[4].mBinding = 1;
            svl.mAttribs[4].mLocation = 4;
            svl.mAttribs[4].mOffset = 0;
            svl.mAttribs[5].mSemantic = SEMANTIC_TEXCOORD3;       // Uv (per-vertex, base map)
            svl.mAttribs[5].mFormat = TinyImageFormat_R32G32_SFLOAT;
            svl.mAttribs[5].mBinding = 0;
            svl.mAttribs[5].mLocation = 5;
            svl.mAttribs[5].mOffset = 44;
            svl.mAttribs[6].mSemantic = SEMANTIC_TEXCOORD4;       // TexIndex (per-instance uint)
            svl.mAttribs[6].mFormat = TinyImageFormat_R32_UINT;
            svl.mAttribs[6].mBinding = 1;
            svl.mAttribs[6].mLocation = 6;
            svl.mAttribs[6].mOffset = 4;

            // BISECT (Tier 1b skinned-only): skinned in the Z-prepass → colour matches with EQUAL.
            DepthStateDesc skDepth = {};
            skDepth.mDepthTest = true;
            skDepth.mDepthWrite = false;
            skDepth.mDepthFunc = CMP_EQUAL;

            RasterizerStateDesc skRaster = {};
            skRaster.mCullMode = CULL_MODE_BACK;
            skRaster.mFrontFace = FRONT_FACE_CCW;

            PipelineDesc skPd = {};
            skPd.mType = PIPELINE_TYPE_GRAPHICS;
            GraphicsPipelineDesc& sg = skPd.mGraphicsDesc;
            sg.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
            sg.mRenderTargetCount = 1;
            sg.pColorFormats = &g_live.pRT->mFormat;
            sg.mSampleCount = (SampleCount)g_live.sampleCount;
            sg.mSampleQuality = 0;
            sg.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
            sg.pDepthState = &skDepth;
            sg.pVertexLayout = &svl;
            sg.pRasterizerState = &skRaster;
            sg.pShaderProgram = g_live.pSkinnedShader;
            addPipeline(R, &skPd, &g_live.pSkinnedPipeline);
            if (!g_live.pSkinnedPipeline) {
                std::printf("[forge] addPipeline(skinned) FAILED\n");
                return false;
            }
            // Mirror variant (CW) for negative-determinant (mirrored left-side) parts.
            RasterizerStateDesc skRasterMirror = skRaster;
            skRasterMirror.mFrontFace = FRONT_FACE_CW;
            sg.pRasterizerState = &skRasterMirror;
            addPipeline(R, &skPd, &g_live.pSkinnedPipelineMirror);
            if (!g_live.pSkinnedPipelineMirror) {
                std::printf("[forge] addPipeline(skinned mirror) FAILED\n");
                return false;
            }

            // Tier 1b skinned Z-prepass pipelines: skinned.vert + the shared depthonly.frag
            // (skinned VSOutput == opaque's, so it links), GEQUAL + depthWrite, 0 RTs. Same svl
            // so SV_Position matches the colour pass's EQUAL test. CRITICAL: this MUST use a
            // skinned.vert shader, NOT pDepthOnlyShader (opaque.vert) — pairing opaque.vert with
            // the skinned layout reads bone-indices/weights as UV/drawindex → garbage TexIndex →
            // gTextures[garbage] OOB bindless access → GPU hang. (That was the Tier 1b hang.)
            {
                ShaderLoadDesc skDpDesc = {};
                skDpDesc.mVert.pFileName = "skinned.vert";
                skDpDesc.mFrag.pFileName = "depthonly.frag";
                addShader(R, &skDpDesc, &g_live.pSkinnedDepthShader);
                if (!g_live.pSkinnedDepthShader) {
                    std::printf("[forge] addShader(skinned depth) FAILED\n");
                    return false;
                }
                DepthStateDesc skPreDepth = {};
                skPreDepth.mDepthTest = true;
                skPreDepth.mDepthWrite = true;
                skPreDepth.mDepthFunc = CMP_GEQUAL;

                PipelineDesc skpPd = {};
                skpPd.mType = PIPELINE_TYPE_GRAPHICS;
                GraphicsPipelineDesc& spg = skpPd.mGraphicsDesc;
                spg.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
                spg.mRenderTargetCount = 0;
                spg.pColorFormats = nullptr;
                spg.mSampleCount = (SampleCount)g_live.sampleCount;
                spg.mSampleQuality = 0;
                spg.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
                spg.pDepthState = &skPreDepth;
                spg.pVertexLayout = &svl;
                spg.pRasterizerState = &skRaster;            // CCW
                spg.pShaderProgram = g_live.pSkinnedDepthShader;
                addPipeline(R, &skpPd, &g_live.pSkinnedPrepassPipeline);
                if (!g_live.pSkinnedPrepassPipeline) {
                    std::printf("[forge] addPipeline(skinned prepass) FAILED\n");
                    return false;
                }
                spg.pRasterizerState = &skRasterMirror;       // CW
                addPipeline(R, &skpPd, &g_live.pSkinnedPrepassPipelineMirror);
                if (!g_live.pSkinnedPrepassPipelineMirror) {
                    std::printf("[forge] addPipeline(skinned prepass mirror) FAILED\n");
                    return false;
                }
            }

            // Bone windows: ONE exactly-64KB cbuffer per window (same CBV rule as worlds).
            for (uint32_t b = 0; b < kMaxBatches; ++b) {
                BufferLoadDesc bb = {};
                bb.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                bb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
                bb.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
                bb.mDesc.mSize = kBatchBytes;
                bb.mDesc.pName = "boneWindowCbv";
                bb.pData = nullptr;
                bb.ppBuffer = &g_live.pBonesBuf[b];
                addResource(&bb, nullptr);
            }
            // Skinned instance buffer: kMaxSkinned uint2 { Base, texIndex }, one entry per
            // drawn part (indexed by skinnedDrawn via firstInstance). CPU-mapped, written per
            // frame in the skinned loop. Dedicated (not pInstanceBuf[0]) so the static loop's
            // per-frame texIndices can't clobber it.
            {
                BufferLoadDesc sib = {};
                sib.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
                sib.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
                sib.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
                sib.mDesc.mSize = (uint64_t)kMaxSkinned * 2 * sizeof(uint32_t);
                sib.mDesc.pName = "instanceVBSkin";
                sib.pData = nullptr;
                sib.ppBuffer = &g_live.pInstanceBufSkin;
                addResource(&sib, nullptr);
            }

            waitForAllResourceLoads();
            if (!g_live.pBonesBuf[0] || !g_live.pInstanceBufSkin) {
                return false;
            }

            // Skinned PerBatch set: instance b binds bone window b (pBonesBuf[b]) as gBatch.
            DescriptorSetDesc sbDesc = SRT_SET_DESC(SrtData, PerBatch, kMaxBatches, 0);
            addDescriptorSet(R, &sbDesc, &g_live.pPerBatchSetSkin);
            if (!g_live.pPerBatchSetSkin) {
                return false;
            }
            for (uint32_t b = 0; b < kMaxBatches; ++b) {
                DescriptorData p = {};
                p.mIndex = SRT_RES_IDX(SrtData, PerBatch, gBatch);
                p.ppBuffers = &g_live.pBonesBuf[b];
                updateDescriptorSet(R, b, g_live.pPerBatchSetSkin, 1, &p);
            }
        }

        // --- Tier 4: multi-map shader + pipelines + world window + instance VB + descriptor set ---
        // Wide vertex (4 UV sets) + per-draw stage list. Shares default.rootsig + SrtData; gBatch
        // is bound to ONE 64KB world window (pMMWorldsBuf), drawn with firstInstance = the part's
        // world index. Mirrors the static/skinned block structure.
        {
            ShaderLoadDesc mmDesc = {};
            mmDesc.mVert.pFileName = "multimap.vert";
            mmDesc.mFrag.pFileName = "multimap.frag";
            addShader(R, &mmDesc, &g_live.pMultiMapShader);
            if (!g_live.pMultiMapShader) {
                std::printf("[forge] addShader(multimap) FAILED\n");
                return false;
            }

            // Multi-map vertex layout: binding 0 = mesh (IPC::GeomVertexWireMM, stride 60:
            // pos@0, normal@12, color@24, uv0@28, uv1@36, uv2@44, uv3@52); binding 1 = per-INSTANCE
            // { Meta @0, Stages uint4 @4, matDiffuse @20, matAmbient @32, matEmissive @44 } (stride 56).
            VertexLayout mvl = {};
            mvl.mBindingCount = 2;
            mvl.mBindings[0].mStride = sizeof(IPC::GeomVertexWireMM);
            mvl.mBindings[0].mRate = VERTEX_BINDING_RATE_VERTEX;
            mvl.mBindings[1].mStride = kMMInstU32 * sizeof(uint32_t);
            mvl.mBindings[1].mRate = VERTEX_BINDING_RATE_INSTANCE;
            mvl.mAttribCount = 12;
            mvl.mAttribs[0].mSemantic = SEMANTIC_POSITION;
            mvl.mAttribs[0].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
            mvl.mAttribs[0].mBinding = 0; mvl.mAttribs[0].mLocation = 0; mvl.mAttribs[0].mOffset = 0;
            mvl.mAttribs[1].mSemantic = SEMANTIC_NORMAL;
            mvl.mAttribs[1].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
            mvl.mAttribs[1].mBinding = 0; mvl.mAttribs[1].mLocation = 1; mvl.mAttribs[1].mOffset = 12;
            mvl.mAttribs[2].mSemantic = SEMANTIC_COLOR;
            mvl.mAttribs[2].mFormat = TinyImageFormat_B8G8R8A8_UNORM;   // D3DCOLOR byte order
            mvl.mAttribs[2].mBinding = 0; mvl.mAttribs[2].mLocation = 2; mvl.mAttribs[2].mOffset = 24;
            mvl.mAttribs[3].mSemantic = SEMANTIC_TEXCOORD0;
            mvl.mAttribs[3].mFormat = TinyImageFormat_R32G32_SFLOAT;
            mvl.mAttribs[3].mBinding = 0; mvl.mAttribs[3].mLocation = 3; mvl.mAttribs[3].mOffset = 28;
            mvl.mAttribs[4].mSemantic = SEMANTIC_TEXCOORD1;
            mvl.mAttribs[4].mFormat = TinyImageFormat_R32G32_SFLOAT;
            mvl.mAttribs[4].mBinding = 0; mvl.mAttribs[4].mLocation = 4; mvl.mAttribs[4].mOffset = 36;
            mvl.mAttribs[5].mSemantic = SEMANTIC_TEXCOORD2;
            mvl.mAttribs[5].mFormat = TinyImageFormat_R32G32_SFLOAT;
            mvl.mAttribs[5].mBinding = 0; mvl.mAttribs[5].mLocation = 5; mvl.mAttribs[5].mOffset = 44;
            mvl.mAttribs[6].mSemantic = SEMANTIC_TEXCOORD3;
            mvl.mAttribs[6].mFormat = TinyImageFormat_R32G32_SFLOAT;
            mvl.mAttribs[6].mBinding = 0; mvl.mAttribs[6].mLocation = 6; mvl.mAttribs[6].mOffset = 52;
            mvl.mAttribs[7].mSemantic = SEMANTIC_TEXCOORD4;       // Meta (per-instance uint)
            mvl.mAttribs[7].mFormat = TinyImageFormat_R32_UINT;
            mvl.mAttribs[7].mBinding = 1; mvl.mAttribs[7].mLocation = 7; mvl.mAttribs[7].mOffset = 0;
            mvl.mAttribs[8].mSemantic = SEMANTIC_TEXCOORD5;       // Stages (per-instance uint4)
            mvl.mAttribs[8].mFormat = TinyImageFormat_R32G32B32A32_UINT;
            mvl.mAttribs[8].mBinding = 1; mvl.mAttribs[8].mLocation = 8; mvl.mAttribs[8].mOffset = sizeof(uint32_t);
            mvl.mAttribs[9].mSemantic = SEMANTIC_TEXCOORD6;       // MatDiffuse (per-instance)
            mvl.mAttribs[9].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
            mvl.mAttribs[9].mBinding = 1; mvl.mAttribs[9].mLocation = 9; mvl.mAttribs[9].mOffset = 5 * sizeof(uint32_t);
            mvl.mAttribs[10].mSemantic = SEMANTIC_TEXCOORD7;      // MatAmbient (per-instance)
            mvl.mAttribs[10].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
            mvl.mAttribs[10].mBinding = 1; mvl.mAttribs[10].mLocation = 10; mvl.mAttribs[10].mOffset = 8 * sizeof(uint32_t);
            mvl.mAttribs[11].mSemantic = SEMANTIC_TEXCOORD8;      // MatEmissive (per-instance)
            mvl.mAttribs[11].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
            mvl.mAttribs[11].mBinding = 1; mvl.mAttribs[11].mLocation = 11; mvl.mAttribs[11].mOffset = 11 * sizeof(uint32_t);

            DepthStateDesc mmDepth = {};
            mmDepth.mDepthTest = true;
            mmDepth.mDepthWrite = false;       // Tier 1b: multimap now in the Z-prepass → early-Z
            mmDepth.mDepthFunc = CMP_EQUAL;     // prepass wrote this pixel's depth; colour matches it

            RasterizerStateDesc mmRaster = {};
            mmRaster.mCullMode = CULL_MODE_BACK;
            mmRaster.mFrontFace = FRONT_FACE_CCW;

            PipelineDesc mmPd = {};
            mmPd.mType = PIPELINE_TYPE_GRAPHICS;
            GraphicsPipelineDesc& mg = mmPd.mGraphicsDesc;
            mg.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
            mg.mRenderTargetCount = 1;
            mg.pColorFormats = &g_live.pRT->mFormat;
            mg.mSampleCount = (SampleCount)g_live.sampleCount;
            mg.mSampleQuality = 0;
            mg.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
            mg.pDepthState = &mmDepth;
            mg.pVertexLayout = &mvl;
            mg.pRasterizerState = &mmRaster;
            mg.pShaderProgram = g_live.pMultiMapShader;
            addPipeline(R, &mmPd, &g_live.pMultiMapPipeline);
            if (!g_live.pMultiMapPipeline) {
                std::printf("[forge] addPipeline(multimap) FAILED\n");
                return false;
            }
            RasterizerStateDesc mmRasterMirror = mmRaster;
            mmRasterMirror.mFrontFace = FRONT_FACE_CW;
            mg.pRasterizerState = &mmRasterMirror;
            addPipeline(R, &mmPd, &g_live.pMultiMapPipelineMirror);
            if (!g_live.pMultiMapPipelineMirror) {
                std::printf("[forge] addPipeline(multimap mirror) FAILED\n");
                return false;
            }

            // Tier 1b multi-map Z-prepass: multimap.vert + depthonly_mm.frag (own VSOutput,
            // base-stage alpha test), GEQUAL + depthWrite, 0 RTs. Same mvl → SV_Position matches
            // the colour pass's EQUAL test.
            {
                ShaderLoadDesc mmdDesc = {};
                mmdDesc.mVert.pFileName = "multimap.vert";
                mmdDesc.mFrag.pFileName = "depthonly_mm.frag";
                addShader(R, &mmdDesc, &g_live.pMultiMapDepthShader);
                if (!g_live.pMultiMapDepthShader) {
                    std::printf("[forge] addShader(depthonly_mm) FAILED\n");
                    return false;
                }
                DepthStateDesc mmPreDepth = {};
                mmPreDepth.mDepthTest = true;
                mmPreDepth.mDepthWrite = true;
                mmPreDepth.mDepthFunc = CMP_GEQUAL;

                PipelineDesc mmpPd = {};
                mmpPd.mType = PIPELINE_TYPE_GRAPHICS;
                GraphicsPipelineDesc& mpg = mmpPd.mGraphicsDesc;
                mpg.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
                mpg.mRenderTargetCount = 0;
                mpg.pColorFormats = nullptr;
                mpg.mSampleCount = (SampleCount)g_live.sampleCount;
                mpg.mSampleQuality = 0;
                mpg.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
                mpg.pDepthState = &mmPreDepth;
                mpg.pVertexLayout = &mvl;
                mpg.pRasterizerState = &mmRaster;            // CCW
                mpg.pShaderProgram = g_live.pMultiMapDepthShader;
                addPipeline(R, &mmpPd, &g_live.pMultiMapPrepassPipeline);
                if (!g_live.pMultiMapPrepassPipeline) {
                    std::printf("[forge] addPipeline(multimap prepass) FAILED\n");
                    return false;
                }
                mpg.pRasterizerState = &mmRasterMirror;       // CW
                addPipeline(R, &mmpPd, &g_live.pMultiMapPrepassPipelineMirror);
                if (!g_live.pMultiMapPrepassPipelineMirror) {
                    std::printf("[forge] addPipeline(multimap prepass mirror) FAILED\n");
                    return false;
                }
            }

            // One 64KB world window (a valid full CBV; kMaxMultiMap=256 <= 1024 matrices).
            BufferLoadDesc mwb = {};
            mwb.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            mwb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            mwb.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            mwb.mDesc.mSize = kBatchBytes;
            mwb.mDesc.pName = "mmWorldsCbv";
            mwb.pData = nullptr;
            mwb.ppBuffer = &g_live.pMMWorldsBuf;
            addResource(&mwb, nullptr);

            // Per-draw instance VB: kMaxMultiMap entries of kMMInstU32 uint32 (indexed by
            // multiMapDrawn via firstInstance). CPU-mapped, written per frame in the MM loop.
            BufferLoadDesc mib = {};
            mib.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
            mib.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            mib.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            mib.mDesc.mSize = (uint64_t)kMaxMultiMap * kMMInstU32 * sizeof(uint32_t);
            mib.mDesc.pName = "instanceVBMM";
            mib.pData = nullptr;
            mib.ppBuffer = &g_live.pInstanceBufMM;
            addResource(&mib, nullptr);

            waitForAllResourceLoads();
            if (!g_live.pMMWorldsBuf || !g_live.pInstanceBufMM) {
                return false;
            }

            // Multi-map PerBatch set (1 instance): gBatch = the single world window.
            DescriptorSetDesc mbDesc = SRT_SET_DESC(SrtData, PerBatch, 1, 0);
            addDescriptorSet(R, &mbDesc, &g_live.pPerBatchSetMM);
            if (!g_live.pPerBatchSetMM) {
                return false;
            }
            DescriptorData mp = {};
            mp.mIndex = SRT_RES_IDX(SrtData, PerBatch, gBatch);
            mp.ppBuffers = &g_live.pMMWorldsBuf;
            updateDescriptorSet(R, 0, g_live.pPerBatchSetMM, 1, &mp);
        }

        std::printf("[forge] opaque scene path ready (depth %ux%u, maxDraws=%u, batched %ux%u, maxSkinned=%u, maxMultiMap=%u)\n",
                    width, height, kMaxDraws, kMaxBatches, kBatchSize, kMaxSkinned, kMaxMultiMap);
        return true;
    }

    // --- M1b: slot-indexed static opaque mesh store ------------------------------
    // The client assigns each cached NiTriShape* a dense slot; we keep meshes in a
    // flat array indexed by that slot (no hashing). Grown on demand; freed on
    // shutdown. M1c iterates a per-frame visible list of slots to draw these.
    // Dynamic-geometry ring depth. Animated (morph) meshes re-upload their VB/IB every
    // frame. Instead of recreating the GPU_ONLY buffer + fencing each time (~6ms), a slot
    // that re-uploads at the same shape is promoted to a ring of persistently-mapped
    // CPU_TO_GPU buffers we just memcpy into (no recreate, no fence). The ring lets a write
    // land on a buffer the GPU isn't reading — explicit double-buffering, the modern
    // replacement for D9's MANAGED/vb[2] driver magic. 2 is enough while renderScene is
    // GPU-blocking (each frame completes before the next upload); bump if render goes async.
    constexpr uint32_t kGeomRing = 2;

    struct HostMesh {
        Buffer*  vb;            // per-mesh VB: skinned + dynamic-morph only (null when inArena)
        Buffer*  ib;
        uint32_t vertexCount;
        uint32_t indexCount;
        bool     valid;
        bool     skinned;       // M-Skinning: VB is SkinnedVertexWire (stride 56)
        bool     multimap;      // Tier 4: VB is GeomVertexWireMM (stride 60); own per-mesh VB
        // Buffer consolidation: static non-skinned parts live in the shared mega arena (no
        // per-mesh VB/IB) and draw with byte offsets. Skinned + multimap + dynamic-morph parts
        // are NOT in the arena (vb/ib above). Exactly one of: inArena | skinned | multimap | dynamic.
        bool     inArena;
        uint64_t vbOff;         // byte offset into pArenaVB (valid when inArena)
        uint64_t ibOff;         // byte offset into pArenaIB (valid when inArena)
        // --- dynamic (re-uploaded / animated) path ---
        bool     dynamic;       // in the persistent upload-heap ring (true morph only)
        uint8_t  ring;          // next ring index to write
        Buffer*  dynVb[kGeomRing];
        Buffer*  dynIb[kGeomRing];
        // Frequency gate: promote to the upload-heap ring ONLY for a mesh that re-uploads on
        // CONSECUTIVE frames (a real per-vertex morph). Cell-transition churn (recycled slots)
        // re-uploads a slot once, not frame-after-frame, so it must NOT promote — an upload-heap
        // VB has high-latency uncached GPU vertex fetch, and mass-promoting churn tanks render.
        uint32_t lastUploadFrame;
        uint16_t uploadStreak;  // consecutive-frame re-uploads at the same shape
    };
    HostMesh* g_meshes   = nullptr;
    uint32_t  g_meshCap  = 0;   // allocated slot count
    uint32_t  g_meshHigh = 0;   // highest slot+1 ever populated
    uint32_t  g_renderFrame = 0;   // monotonic, ++ per renderScene; drives the dynamic-promote streak
    uint32_t  g_dynamicCount = 0;  // meshes currently in the upload-heap ring (should stay tiny)
    // Split the host render cost into CPU command-recording (the per-draw bind+draw loop) vs
    // GPU execution (submit→fence). If record >> gpu, the renderer is CPU-bound on per-draw
    // binds (D3D9-style) and the DX12 win needs buffer consolidation / batched draws.
    double    g_recAccum = 0.0;     // summed cmd-record ms over the heartbeat window
    double    g_gpuAccum = 0.0;     // summed submit→fence ms over the heartbeat window

    inline double hostNowMs() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // Consecutive same-shape re-uploads before a slot migrates to the upload-heap ring.
    // 2 = promote on the 3rd consecutive frame (a couple of static recreates at morph onset,
    // then fence-free). Cell churn never reaches it (one-off re-uploads).
    constexpr uint16_t kDynPromoteStreak = 2;
    unsigned  g_lastDrawn = 0;  // static parts actually drawn in the last renderScene
    unsigned  g_lastSkinnedDrawn = 0;  // skinned parts actually drawn in the last renderScene
    unsigned  g_lastMultiMapDrawn = 0; // multi-map parts actually drawn in the last renderScene
    unsigned  g_lastLightCount = 0;    // point lights uploaded in the last renderScene
    uint32_t  g_debugMode = 0;         // F12 debug view: 0=normal, 1=depth, 2=scatter (written to FrameData.debugParams.x)

    bool ensureMeshSlot(uint32_t slot) {
        if (slot < g_meshCap) {
            return true;
        }
        uint32_t newCap = g_meshCap ? g_meshCap * 2 : 1024;
        while (newCap <= slot) {
            newCap *= 2;
        }
        HostMesh* n = (HostMesh*)tf_calloc(newCap, sizeof(HostMesh));
        if (!n) {
            return false;
        }
        if (g_meshes) {
            std::memcpy(n, g_meshes, (size_t)g_meshCap * sizeof(HostMesh));
            tf_free(g_meshes);
        }
        g_meshes  = n;
        g_meshCap = newCap;
        return true;
    }

    // Release every buffer a slot owns. Static slots own vb/ib directly; dynamic slots own
    // the ring (dynVb/dynIb[]) and vb/ib merely alias the current ring entry, so the ring is
    // the authority — null vb/ib first to avoid a double-free of an aliased pointer.
    void releaseMeshBuffers(HostMesh& m) {
        if (m.inArena) {
            // Arena parts own no D3D12 resources — just free the suballocations.
            g_arenaVB.release(m.vbOff, (uint64_t)m.vertexCount * sizeof(IPC::GeomVertexWire));
            g_arenaIB.release(m.ibOff, (uint64_t)m.indexCount * sizeof(uint16_t));
            m.inArena = false;
            m.vbOff = m.ibOff = 0;
        } else if (m.dynamic) {
            for (uint32_t r = 0; r < kGeomRing; ++r) {
                if (m.dynVb[r]) { removeResource(m.dynVb[r]); m.dynVb[r] = nullptr; }
                if (m.dynIb[r]) { removeResource(m.dynIb[r]); m.dynIb[r] = nullptr; }
            }
            m.vb = m.ib = nullptr;   // were aliases into the ring
            if (g_dynamicCount) { --g_dynamicCount; }
        } else {
            if (m.vb) { removeResource(m.vb); m.vb = nullptr; }
            if (m.ib) { removeResource(m.ib); m.ib = nullptr; }
        }
        m.dynamic = false;
        m.ring = 0;
    }

    void freeMeshStore() {
        if (g_meshes) {
            for (uint32_t i = 0; i < g_meshHigh; ++i) {
                if (g_meshes[i].valid) {
                    releaseMeshBuffers(g_meshes[i]);
                }
            }
            tf_free(g_meshes);
        }
        g_meshes   = nullptr;
        g_meshCap  = 0;
        g_meshHigh = 0;
    }
}

namespace ForgeRender {
    bool init(unsigned width, unsigned height, unsigned sampleCount, unsigned anisoLevel) {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        if (g_live.pRenderer) {
            shutdown();
        }
        std::printf("[forge] live init %ux%u (%ux MSAA requested, AF %u)...\n", width, height, sampleCount, anisoLevel);

        if (!forgeBringUp(&g_live.pRenderer, &g_live.pQueue)) {
            return false;
        }
        Renderer* R = g_live.pRenderer;

        // MSAA: validate the requested count against the device for the shared RT format.
        // Unsupported ⇒ log loudly and fall back to single-sample (the agreed policy — no
        // silent substitution of a different count). Counts are 1/2/4/8 (D3DMULTISAMPLE).
        uint32_t reqSamples = sampleCount < 1 ? 1u : sampleCount;
        if (reqSamples > 1) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS msq = {};
            msq.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            msq.SampleCount = reqSamples;
            HRESULT qr = R->mDx.pDevice->CheckFeatureSupport(
                D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &msq, sizeof(msq));
            if (FAILED(qr) || msq.NumQualityLevels == 0) {
                std::printf("[forge] MSAA %ux unsupported for B8G8R8A8_UNORM — rendering single-sample\n", reqSamples);
                LOGF(eWARNING, "[forge] MSAA %ux unsupported for B8G8R8A8_UNORM — rendering single-sample", reqSamples);
                reqSamples = 1;
            }
        }
        g_live.sampleCount = reqSamples;
        g_live.anisoLevel  = anisoLevel;   // consumed by the Phase 2 texture sampler
        std::printf("[forge] MSAA sample count = %u, AF = %u\n", g_live.sampleCount, g_live.anisoLevel);

        g_live.pShader = loadTriangleShader(R);
        if (!g_live.pShader) {
            forgeTearDown(R, g_live.pQueue);
            g_live = LiveRenderer{};
            return false;
        }

        // Shared D3D12 render target per the D3D9Ex-ingestion rules (see
        // renderTriangleShared for the why): SHARED heap, B8G8R8A8_UNORM,
        // LAYOUT_UNKNOWN. Created manually + adopted via pNativeHandle.
        ID3D12Device* pDevice = R->mDx.pDevice;

        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC resDesc = {};
        resDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        resDesc.Width = width;
        resDesc.Height = height;
        resDesc.DepthOrArraySize = 1;
        resDesc.MipLevels = 1;
        resDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        resDesc.SampleDesc.Count = 1;
        resDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

        D3D12_CLEAR_VALUE clearVal = {};
        clearVal.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        clearVal.Color[3] = 0.0f;   // transparent bg: alpha = coverage mask for the alpha-blend composite

        std::printf("[forge] live CreateCommittedResource (SHARED)...\n");
        HRESULT hr = pDevice->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_SHARED, &resDesc,
            D3D12_RESOURCE_STATE_RENDER_TARGET, &clearVal, IID_PPV_ARGS(&g_live.pSharedRes));
        if (FAILED(hr) || !g_live.pSharedRes) {
            std::printf("[forge] live CreateCommittedResource FAILED 0x%08lX\n", (unsigned long)hr);
            shutdown();
            return false;
        }

        hr = pDevice->CreateSharedHandle(g_live.pSharedRes, nullptr, GENERIC_ALL, nullptr, &g_live.ntHandle);
        if (FAILED(hr) || !g_live.ntHandle) {
            std::printf("[forge] live CreateSharedHandle FAILED 0x%08lX\n", (unsigned long)hr);
            shutdown();
            return false;
        }
        std::printf("[forge] live shared NT handle = %p\n", (void*)g_live.ntHandle);

        RenderTargetDesc rtDesc = {};
        rtDesc.mWidth = width;
        rtDesc.mHeight = height;
        rtDesc.mDepth = 1;
        rtDesc.mArraySize = 1;
        rtDesc.mMipLevels = 1;
        rtDesc.mSampleCount = SAMPLE_COUNT_1;
        rtDesc.mFormat = TinyImageFormat_B8G8R8A8_UNORM;
        rtDesc.mStartState = RESOURCE_STATE_RENDER_TARGET;
        rtDesc.mClearValue.r = 0.0f;
        rtDesc.mClearValue.g = 0.0f;
        rtDesc.mClearValue.b = 0.0f;
        rtDesc.mClearValue.a = 0.0f;   // transparent bg: alpha = coverage mask for the composite
        rtDesc.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
        rtDesc.pNativeHandle = (void*)g_live.pSharedRes;
        rtDesc.pName = "liveSharedRT";
        addRenderTarget(R, &rtDesc, &g_live.pRT);

        g_live.pPipeline = buildTrianglePipeline(R, g_live.pRT, g_live.pShader);
        if (!g_live.pPipeline) {
            std::printf("[forge] live addPipeline FAILED\n");
            shutdown();
            return false;
        }

        CmdPoolDesc cmdPoolDesc = {};
        cmdPoolDesc.pQueue = g_live.pQueue;
        initCmdPool(R, &cmdPoolDesc, &g_live.pCmdPool);
        CmdDesc cmdDesc = {};
        cmdDesc.pPool = g_live.pCmdPool;
        initCmd(R, &cmdDesc, &g_live.pCmd);
        initFence(R, &g_live.pFence);

        // M1c opaque scene path (depth RT + opaque pipeline + scene CBVs) is built
        // LAZILY on the first renderScene — NOT here. Building it during init (before
        // any geometry upload) was corrupting the resource loader and hanging/crashing
        // the first uploadGeometry. Deferring it lets geometry upload run on a clean
        // loader, and isolates any opaque-path issue to the F11 scene path.

        g_live.width = width;
        g_live.height = height;
        g_live.firstFrame = true;
        std::printf("[forge] live init OK\n");
        return true;
    }

    void* sharedHandle() {
        return (void*)g_live.ntHandle;
    }

    bool sceneReady() {
        return g_live.pOpaquePipeline != nullptr;
    }

    bool renderFrame(unsigned frameIndex) {
        if (!g_live.pRenderer || !g_live.pPipeline) {
            return false;
        }
        Renderer* R = g_live.pRenderer;

        resetCmdPool(R, g_live.pCmdPool);
        beginCmd(g_live.pCmd);

        // Steady state between frames is COMMON (handed off to D3D9). Transition
        // back to RENDER_TARGET for the draw — except the very first frame, which
        // the resource was created in RENDER_TARGET state for.
        if (!g_live.firstFrame) {
            RenderTargetBarrier toRT = {};
            toRT.pRenderTarget = g_live.pRT;
            toRT.mCurrentState = RESOURCE_STATE_COMMON;
            toRT.mNewState = RESOURCE_STATE_RENDER_TARGET;
            cmdResourceBarrier(g_live.pCmd, 0, nullptr, 0, nullptr, 1, &toRT);
        }

        BindRenderTargetsDesc bind = {};
        bind.mRenderTargetCount = 1;
        bind.mRenderTargets[0] = { g_live.pRT, LOAD_ACTION_CLEAR };
        cmdBindRenderTargets(g_live.pCmd, &bind);
        cmdSetViewport(g_live.pCmd, 0.0f, 0.0f, (float)g_live.width, (float)g_live.height, 0.0f, 1.0f);
        cmdSetScissor(g_live.pCmd, 0, 0, g_live.width, g_live.height);
        cmdBindPipeline(g_live.pCmd, g_live.pPipeline);
        cmdDraw(g_live.pCmd, 3, 0);
        cmdBindRenderTargets(g_live.pCmd, nullptr);

        // Hand the shared RT back to COMMON so MW's D3D9Ex can read it.
        RenderTargetBarrier toCommon = {};
        toCommon.pRenderTarget = g_live.pRT;
        toCommon.mCurrentState = RESOURCE_STATE_RENDER_TARGET;
        toCommon.mNewState = RESOURCE_STATE_COMMON;
        cmdResourceBarrier(g_live.pCmd, 0, nullptr, 0, nullptr, 1, &toCommon);
        endCmd(g_live.pCmd);

        QueueSubmitDesc submitDesc = {};
        submitDesc.mCmdCount = 1;
        submitDesc.ppCmds = &g_live.pCmd;
        submitDesc.pSignalFence = g_live.pFence;
        submitDesc.mSubmitDone = true;
        queueSubmit(g_live.pQueue, &submitDesc);
        waitForFences(R, 1, &g_live.pFence);

        g_live.firstFrame = false;
        (void)frameIndex;
        return true;
    }

    bool renderScene(const float* viewProj, const float* lighting, const void* drawBlob,
                     unsigned drawCount, unsigned drawBytes,
                     const void* skinnedBlob, unsigned skinnedCount, unsigned skinnedBytes,
                     const void* multiMapBlob, unsigned multiMapCount, unsigned multiMapBytes,
                     const void* lightBlob, unsigned lightCount, unsigned lightBytes) {
        if (!g_live.pRenderer) {
            return false;
        }
        Renderer* R = g_live.pRenderer;

        // If the device was already removed on a PRIOR frame (e.g. during a dense exterior),
        // every subsequent frame stays black no matter the scene — this names that state
        // instead of silently rendering nothing (explains "back to interior, still black").
        if (logDeviceRemoved(R, "renderScene/entry")) {
            return false;
        }

        // Lazy one-time opaque-path build (depth RT + pipeline + descriptor sets),
        // deferred out of init so geometry upload runs first on a clean resource loader.
        if (!g_live.pOpaquePipeline) {
            std::printf("[forge] renderScene: lazy buildOpaquePath...\n");
            if (!buildOpaquePath(R, g_live.width, g_live.height)) {
                std::printf("[forge] lazy buildOpaquePath FAILED — scene path disabled\n");
                return false;   // caller falls back to triangle
            }
        }
        if (!g_live.pDepth) {
            return false;
        }
        ++g_renderFrame;   // drives the dynamic-promote consecutive-frame streak in uploadGeometry

        const uint32_t n = (drawCount < g_live.maxDraws) ? drawCount : g_live.maxDraws;
        const uint32_t haveBytes = drawBytes / (uint32_t)sizeof(IPC::DrawItemWire);
        const uint32_t count = (n < haveBytes) ? n : haveBytes;
        const IPC::DrawItemWire* items = (const IPC::DrawItemWire*)drawBlob;

        // REVERSE-Z: post-multiply the received row-major viewProj by Z_rev (maps clip z'
        // = w - z, i.e. near->1 / far->0). On a row-major matrix that ONLY touches column 2:
        // m[i*4+2] := m[i*4+3] - m[i*4+2]. The shader reads the cbuffer column-major (== the
        // transpose) so mul(viewProj, worldPos) applies viewProj*Z_rev in row-vector terms,
        // i.e. Z_rev acts on the post-projection clip coords — exactly reverse-Z. No shader
        // change. Pairs with depth clear 0.0 + CMP_GEQUAL above.
        float rzViewProj[16];
        std::memcpy(rzViewProj, viewProj, 16 * sizeof(float));
        rzViewProj[2]  = rzViewProj[3]  - rzViewProj[2];
        rzViewProj[6]  = rzViewProj[7]  - rzViewProj[6];
        rzViewProj[10] = rzViewProj[11] - rzViewProj[10];
        rzViewProj[14] = rzViewProj[15] - rzViewProj[14];

        // HALF-PIXEL ALIGN: MW's own layers (sky/water/distant land/UI) are drawn by DXVK as
        // D3D9, which applies the D3D9 half-pixel rasterization offset. Our Forge layer is D3D12
        // (pixel centre at +0.5, no offset), so it lands ~½px off in BOTH axes from every
        // MW-native layer → a ~1px whole-image shift in the composite. Re-introduce the D3D9
        // offset on the Forge projection so the layers register. Shift screen-space by half a
        // pixel: NDC dx = -1/width, dy = +1/height (y is flipped screen↔NDC). On the row-major
        // viewProj clip.x = col0 (idx 0,4,8,12), clip.y = col1 (1,5,9,13), clip.w = col3
        // (3,7,11,15); add d*col3 to the matching col so the shift scales with w (post-divide
        // constant). kHalfPixelSign flips the whole correction in one place for F5/F6 A/B.
        {
            const float kHalfPixelSign = -1.0f;  // -1 pushes Forge toward bottom-right (cancels the D3D9/D3D12 top-left mismatch)
            const float dx = kHalfPixelSign * (-1.0f / (float)g_live.width);
            const float dy = kHalfPixelSign * ( 1.0f / (float)g_live.height);
            rzViewProj[0]  += dx * rzViewProj[3];
            rzViewProj[4]  += dx * rzViewProj[7];
            rzViewProj[8]  += dx * rzViewProj[11];
            rzViewProj[12] += dx * rzViewProj[15];
            rzViewProj[1]  += dy * rzViewProj[3];
            rzViewProj[5]  += dy * rzViewProj[7];
            rzViewProj[9]  += dy * rzViewProj[11];
            rzViewProj[13] += dy * rzViewProj[15];
        }

        // viewProj → the persistent-mapped frame cbuffer. world[i] → window (i/kBatchSize)
        // at local slot (i%kBatchSize): byte offset (i/kBatchSize)*kBatchBytes + (i%kBatchSize)*64.
        // Index i aligns with the draw loop below (batch+local select the same matrix).
        std::memcpy(g_live.pFrameCbv->pCpuMappedAddress, rzViewProj, 16 * sizeof(float));
        // Tier 1 lighting block (6 × float4 = 24 floats) right after viewProj. gFrameData layout:
        // viewProj(64B) | sunDir | sunCol | ambCol | fogColNear | fogParams | eyePos. The pFrameCbv
        // is 256B (min CBV), so 64 + 96 = 160B fits. Null lighting leaves the prior values.
        if (lighting) {
            std::memcpy((uint8_t*)g_live.pFrameCbv->pCpuMappedAddress + 16 * sizeof(float),
                        lighting, 24 * sizeof(float));
        }
        // F12 debug view: debugParams.x at float index 40 (160B = viewProj 16f + 6×float4 24f).
        // 0=normal (frag is byte-for-byte unchanged), 1=depth world-distance grayscale.
        ((float*)g_live.pFrameCbv->pCpuMappedAddress)[40] = (float)g_debugMode;

        // Tier 3a: upload this frame's point lights into gLights. Each PointLightWire is exactly
        // 3 float4 (== one cbuffer light entry), so the blob copies straight in after the float4
        // lightParams header. lightParams.x = the active count the frag loops. Clamp to the cap +
        // to what actually arrived. Count 0 leaves the loop a no-op (header still written = 0).
        {
            uint32_t nL = lightCount;
            if (nL > kMaxPointLights) { nL = kMaxPointLights; }
            const uint32_t haveLights = lightBytes / (uint32_t)sizeof(IPC::PointLightWire);
            if (nL > haveLights) { nL = haveLights; }
            uint8_t* lc = (uint8_t*)g_live.pLightCbv->pCpuMappedAddress;
            ((float*)lc)[0] = (float)nL;   // lightParams.x = count (.yzw already 0)
            ((float*)lc)[1] = 0.0f; ((float*)lc)[2] = 0.0f; ((float*)lc)[3] = 0.0f;
            if (nL && lightBlob) {
                std::memcpy(lc + 16, lightBlob, (size_t)nL * sizeof(IPC::PointLightWire));
            }
            g_lastLightCount = nL;
        }
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t batch = i / kBatchSize;
            const uint32_t local = i % kBatchSize;
            uint8_t* dst = (uint8_t*)g_live.pWorldsBuf[batch]->pCpuMappedAddress;
            std::memcpy(dst + (size_t)local * 64, items[i].world, 64);
            // Per-draw texIndex + alpha-test ref + vColSource packed into instance slot [1]; [0]
            // stays the identity DrawIndex set at creation. Tier 2b material rgb in slots [2..10]
            // (float). Unloaded/unknown slots fall back to 0 (white texture).
            uint32_t* inst = (uint32_t*)g_live.pInstanceBuf[batch]->pCpuMappedAddress;
            inst[local * kStaticInstU32 + 1] = packTexAlpha(items[i].texIndex, items[i].alphaRef, items[i].vColSource);
            float* finst = (float*)inst;
            finst[local * kStaticInstU32 + 2] = items[i].matDiffuse[0];
            finst[local * kStaticInstU32 + 3] = items[i].matDiffuse[1];
            finst[local * kStaticInstU32 + 4] = items[i].matDiffuse[2];
            finst[local * kStaticInstU32 + 5] = items[i].matAmbient[0];
            finst[local * kStaticInstU32 + 6] = items[i].matAmbient[1];
            finst[local * kStaticInstU32 + 7] = items[i].matAmbient[2];
            finst[local * kStaticInstU32 + 8] = items[i].matEmissive[0];
            finst[local * kStaticInstU32 + 9] = items[i].matEmissive[1];
            finst[local * kStaticInstU32 + 10] = items[i].matEmissive[2];
            // Terrain DECAL_1 overlay slot (0 = no decal → frag splat gated off, non-terrain unchanged).
            inst[local * kStaticInstU32 + 11] = items[i].overlayTexIndex;
        }

        resetCmdPool(R, g_live.pCmdPool);
        beginCmd(g_live.pCmd);
        const double tRec0 = hostNowMs();   // start of CPU command recording

        // Color target: the MSAA color when antialiasing is on (resolved into pRT at the end),
        // else the shared pRT directly. The MSAA target is left in RENDER_TARGET between frames
        // (transitioned back after the resolve), so it needs no begin barrier — only the direct
        // pRT path transitions COMMON (steady state) -> RENDER_TARGET (first frame it was
        // created RENDER_TARGET). Depth was created DEPTH_WRITE and stays there.
        RenderTarget* colorTarget = (g_live.sampleCount > 1) ? g_live.pMSAAColor : g_live.pRT;
        if (g_live.sampleCount == 1 && !g_live.firstFrame) {
            RenderTargetBarrier toRT = {};
            toRT.pRenderTarget = g_live.pRT;
            toRT.mCurrentState = RESOURCE_STATE_COMMON;
            toRT.mNewState = RESOURCE_STATE_RENDER_TARGET;
            cmdResourceBarrier(g_live.pCmd, 0, nullptr, 0, nullptr, 1, &toRT);
        }

        // Phase 1 depth-takeover: classify + fill the indirect args FIRST, with NO render target
        // bound yet. BOTH the Z-prepass and the colour pass replay the same arena groups, so the
        // classification is computed once here; each pass binds its own targets/pipeline below
        // (prepass depth-only first, then colour). (tasks/forge-depth-ssao.md)

        // GPU-driven static draws (ExecuteIndirect). Consolidation already put every static
        // non-skinned part in the shared mega VB/IB; now we also collapse the ~2667 per-part
        // cmdDrawIndexedInstanced API CALLS (the residual CPU record cost) into a handful of
        // cmdExecuteIndirect — one per (mirror, batch) group. ExecuteIndirect can't switch PSO
        // or rebind descriptors/buffers mid-execution, so a single call spans exactly one group
        // (same opaque/mirror PSO + PerBatch[batch] + arena VB + pInstanceBuf[batch] + arena IB);
        // each part is just a 20-byte IndirectDrawIndexArguments record in pIndirectArgs.
        // The 1-4 dynamic-morph parts (own VB/IB, not in the arena) can't ride the shared-VB
        // indirect call, so they stay inline cmdDrawIndexedInstanced after the groups.
        const uint32_t vStride = (uint32_t)sizeof(IPC::GeomVertexWire);
        const uint32_t iStride = (uint32_t)(kStaticInstU32 * sizeof(uint32_t));

        // Classify the draw list once: arena parts → tmp records (for arg-buffer fill),
        // dynamic-morph parts → a small inline list. (worldMirrored + the slot lookup are
        // computed here once instead of twice.)
        struct ArenaTmp { uint32_t indexCount, firstIndex, firstVertex, local; uint8_t mirror, batch; };
        static std::vector<ArenaTmp> s_arena;     // reused across frames (render is single-threaded)
        static std::vector<uint32_t> s_dynamic;   // draw indices i of dynamic-morph parts
        s_arena.clear();
        s_dynamic.clear();
        if (s_arena.capacity() < count) s_arena.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t slot = items[i].slot;
            if (slot >= g_meshHigh || !g_meshes[slot].valid) {
                continue;   // mesh not uploaded yet (or evicted)
            }
            const HostMesh& m = g_meshes[slot];
            if (m.inArena) {
                ArenaTmp t;
                t.indexCount  = m.indexCount;
                t.firstIndex  = (uint32_t)(m.ibOff / sizeof(uint16_t));
                t.firstVertex = (uint32_t)(m.vbOff / sizeof(IPC::GeomVertexWire));
                t.local       = i % kBatchSize;
                t.mirror      = worldMirrored(items[i].world) ? 1 : 0;
                t.batch       = (uint8_t)(i / kBatchSize);
                s_arena.push_back(t);
            } else {
                s_dynamic.push_back(i);   // own VB/IB → inline draw below
            }
        }
        const uint32_t drawn = (uint32_t)(s_arena.size() + s_dynamic.size());

        // Count then prefix-sum the (mirror, batch) groups (mirror outer → ≤2 PSO binds).
        uint32_t groupCount[2][kMaxBatches] = {};
        for (const ArenaTmp& t : s_arena) { ++groupCount[t.mirror][t.batch]; }
        uint32_t groupOff[2][kMaxBatches] = {};
        uint32_t running = 0;
        for (uint32_t mir = 0; mir < 2; ++mir) {
            for (uint32_t b = 0; b < kMaxBatches; ++b) {
                groupOff[mir][b] = running;
                running += groupCount[mir][b];
            }
        }
        // Fill the indirect-args buffer in group order. firstInstance=local → the per-instance
        // Base attr reads pInstanceBuf[batch][local].x = local → gBatch.worlds[local] (+ texIndex).
        IndirectDrawIndexArguments* args =
            (IndirectDrawIndexArguments*)g_live.pIndirectArgs->pCpuMappedAddress;
        uint32_t cursor[2][kMaxBatches];
        std::memcpy(cursor, groupOff, sizeof(cursor));
        for (const ArenaTmp& t : s_arena) {
            IndirectDrawIndexArguments& a = args[cursor[t.mirror][t.batch]++];
            a.mIndexCount    = t.indexCount;
            a.mInstanceCount = 1;
            a.mStartIndex    = t.firstIndex;
            a.mVertexOffset  = t.firstVertex;   // BaseVertexLocation (shifts 0-based indices)
            a.mStartInstance = t.local;
        }

        // ===================== Z-PREPASS (depth-only, opaque set) =====================
        // Replay the SAME arena groups + dynamic-morph parts through the depth-only pipeline
        // (opaque.vert + depthonly.frag, GEQUAL + depthWrite, NO colour target), so opaque depth
        // is complete before the colour pass. The colour pass then LOADs this depth and tests
        // CMP_EQUAL (true early-Z). No depth barrier: depth stays DEPTH_WRITE (skinned/multimap
        // still write it in the colour pass) and is read as a DSV, not an SRV — coherent across
        // the two passes. (Skinned/multimap are NOT in the prepass yet — Tier 1b; they keep
        // their combined depth+colour draw, which still renders correctly.)
        {
            BindRenderTargetsDesc pbind = {};
            pbind.mRenderTargetCount = 0;
            pbind.mDepthStencil = { g_live.pDepth, LOAD_ACTION_CLEAR };
            cmdBindRenderTargets(g_live.pCmd, &pbind);
            cmdSetViewport(g_live.pCmd, 0.0f, 0.0f, (float)g_live.width, (float)g_live.height, 0.0f, 1.0f);
            cmdSetScissor(g_live.pCmd, 0, 0, g_live.width, g_live.height);
            cmdBindPipeline(g_live.pCmd, g_live.pOpaquePrepassPipeline);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);     // gFrameData viewProj
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);   // bindless gTextures (alpha test)

            int preMirror = 0;
            for (uint32_t mir = 0; mir < 2; ++mir) {
                bool anyInMirror = false;
                for (uint32_t b = 0; b < kMaxBatches; ++b) { if (groupCount[mir][b]) { anyInMirror = true; break; } }
                if (!anyInMirror) continue;
                if ((int)mir != preMirror) {
                    cmdBindPipeline(g_live.pCmd, mir ? g_live.pOpaquePrepassPipelineMirror
                                                     : g_live.pOpaquePrepassPipeline);
                    cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
                    cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
                    preMirror = (int)mir;
                }
                for (uint32_t b = 0; b < kMaxBatches; ++b) {
                    const uint32_t c = groupCount[mir][b];
                    if (!c) continue;
                    cmdBindDescriptorSet(g_live.pCmd, b, g_live.pPerBatchSet);
                    Buffer*  vbs[2]     = { g_live.pArenaVB, g_live.pInstanceBuf[b] };
                    uint32_t strides[2] = { vStride, iStride };
                    cmdBindVertexBuffer(g_live.pCmd, 2, vbs, strides, nullptr);
                    cmdBindIndexBuffer(g_live.pCmd, g_live.pArenaIB, INDEX_TYPE_UINT16, 0);
                    cmdExecuteIndirect(g_live.pCmd, INDIRECT_DRAW_INDEX, c, g_live.pIndirectArgs,
                                       (uint64_t)groupOff[mir][b] * sizeof(IndirectDrawIndexArguments),
                                       nullptr, 0);
                }
            }
            for (uint32_t i : s_dynamic) {
                const uint32_t slot  = items[i].slot;
                const uint32_t batch = i / kBatchSize;
                const uint32_t local = i % kBatchSize;
                const int mirror = worldMirrored(items[i].world) ? 1 : 0;
                if (mirror != preMirror) {
                    cmdBindPipeline(g_live.pCmd, mirror ? g_live.pOpaquePrepassPipelineMirror
                                                        : g_live.pOpaquePrepassPipeline);
                    cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
                    cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
                    preMirror = mirror;
                }
                cmdBindDescriptorSet(g_live.pCmd, batch, g_live.pPerBatchSet);
                HostMesh& m = g_meshes[slot];
                Buffer*  vbs[2]     = { m.vb, g_live.pInstanceBuf[batch] };
                uint32_t strides[2] = { vStride, iStride };
                cmdBindVertexBuffer(g_live.pCmd, 2, vbs, strides, nullptr);
                cmdBindIndexBuffer(g_live.pCmd, m.ib, INDEX_TYPE_UINT16, 0);
                cmdDrawIndexedInstanced(g_live.pCmd, m.indexCount, 0, 1, 0, local);
            }

            // --- BISECT: skinned Z-prepass (depth-only). Fill bone+instance buffers and draw
            // depth-only; the skinned COLOUR loop below re-walks the blob and re-fills the SAME
            // buffers (identical data → same SV_Position) then draws EQUAL. Double-fill is a
            // deliberate simplification for the bisect (records come back once this is proven).
            if (skinnedBlob && skinnedCount && skinnedBytes &&
                g_live.pSkinnedPrepassPipeline && g_live.pSkinnedPrepassPipelineMirror) {
                cmdBindPipeline(g_live.pCmd, g_live.pSkinnedPrepassPipeline);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
                const uint8_t* sp  = (const uint8_t*)skinnedBlob;
                const uint8_t* sEnd = sp + skinnedBytes;
                uint32_t boundWindow = UINT32_MAX;
                int      boundSkinMirror = 0;
                uint32_t preDrawn = 0;
                for (uint32_t k = 0; k < skinnedCount; ++k) {
                    if (sp + sizeof(IPC::SkinnedDrawWire) > sEnd) { break; }
                    IPC::SkinnedDrawWire item;
                    std::memcpy(&item, sp, sizeof(item));
                    const uint8_t* palette = sp + sizeof(item);
                    const uint32_t bones = (item.numBones < kMaxBonesPerPart) ? item.numBones : kMaxBonesPerPart;
                    const uint64_t paletteBytes = (uint64_t)item.numBones * 64;
                    if (palette + paletteBytes > sEnd) { break; }
                    sp = palette + paletteBytes;
                    if (preDrawn >= kMaxSkinned) { continue; }
                    const uint32_t slot = item.slot;
                    if (slot >= g_meshHigh || !g_meshes[slot].valid || !g_meshes[slot].skinned) { continue; }
                    const uint32_t window = preDrawn / kSkinnedPerWindow;
                    const uint32_t base   = (preDrawn % kSkinnedPerWindow) * kMaxBonesPerPart;
                    uint8_t* dst = (uint8_t*)g_live.pBonesBuf[window]->pCpuMappedAddress;
                    std::memcpy(dst + (size_t)base * 64, palette, (size_t)bones * 64);
                    uint32_t* sinst = (uint32_t*)g_live.pInstanceBufSkin->pCpuMappedAddress;
                    sinst[preDrawn * 2 + 0] = base;
                    sinst[preDrawn * 2 + 1] = packTexAlpha(item.texIndex, item.alphaRef);
                    const int mirror = item.mirror ? 1 : 0;
                    if (mirror != boundSkinMirror) {
                        cmdBindPipeline(g_live.pCmd, mirror ? g_live.pSkinnedPrepassPipelineMirror : g_live.pSkinnedPrepassPipeline);
                        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
                        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
                        boundSkinMirror = mirror;
                        boundWindow = UINT32_MAX;
                    }
                    if (window != boundWindow) {
                        cmdBindDescriptorSet(g_live.pCmd, window, g_live.pPerBatchSetSkin);
                        boundWindow = window;
                    }
                    HostMesh& sm = g_meshes[slot];
                    Buffer*  pvbs[2]     = { sm.vb, g_live.pInstanceBufSkin };
                    uint32_t pstrides[2] = { (uint32_t)sizeof(IPC::SkinnedVertexWire), (uint32_t)(2 * sizeof(uint32_t)) };
                    cmdBindVertexBuffer(g_live.pCmd, 2, pvbs, pstrides, nullptr);
                    cmdBindIndexBuffer(g_live.pCmd, sm.ib, INDEX_TYPE_UINT16, 0);
                    cmdDrawIndexedInstanced(g_live.pCmd, sm.indexCount, 0, 1, 0, preDrawn);
                    ++preDrawn;
                }
            }

            // --- BISECT: multi-map Z-prepass (depth-only). Same double-fill pattern as skinned:
            // fill pMMWorldsBuf + pInstanceBufMM and draw depth-only with depthonly_mm.frag (base-
            // stage alpha test). The multi-map COLOUR loop below re-walks the blob and re-fills the
            // SAME buffers with identical data → bit-identical SV_Position → colour matches EQUAL.
            // depthonly_mm.frag needs gTextures (Persistent) only; NO gLights (no shading).
            if (multiMapBlob && multiMapCount && multiMapBytes &&
                g_live.pMultiMapPrepassPipeline && g_live.pMultiMapPrepassPipelineMirror) {
                cmdBindPipeline(g_live.pCmd, g_live.pMultiMapPrepassPipeline);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerBatchSetMM);   // single world window
                const uint32_t haveMM = multiMapBytes / (uint32_t)sizeof(IPC::MultiMapDrawWire);
                const uint32_t nMM = (multiMapCount < haveMM) ? multiMapCount : haveMM;
                const IPC::MultiMapDrawWire* mmItems = (const IPC::MultiMapDrawWire*)multiMapBlob;
                int      boundMMMirror = 0;
                uint32_t preDrawnMM = 0;
                for (uint32_t k = 0; k < nMM; ++k) {
                    if (preDrawnMM >= kMaxMultiMap) { break; }   // no fallback (PD1); matches colour cap
                    const IPC::MultiMapDrawWire& it = mmItems[k];
                    const uint32_t slot = it.slot;
                    if (slot >= g_meshHigh || !g_meshes[slot].valid || !g_meshes[slot].multimap) { continue; }
                    const uint32_t idx = preDrawnMM;

                    uint8_t* dst = (uint8_t*)g_live.pMMWorldsBuf->pCpuMappedAddress;
                    std::memcpy(dst + (size_t)idx * 64, it.world, 64);

                    uint32_t* inst = (uint32_t*)g_live.pInstanceBufMM->pCpuMappedAddress;
                    uint32_t* e = inst + (size_t)idx * kMMInstU32;
                    const uint32_t sc  = (it.stageCount > 4u) ? 4u : it.stageCount;
                    float ar = it.alphaRef < 0.0f ? 0.0f : (it.alphaRef > 1.0f ? 1.0f : it.alphaRef);
                    const uint32_t aref = (uint32_t)(ar * 255.0f + 0.5f) & 0xFFu;
                    const uint32_t vcs  = it.vColSource & 0x3u;
                    e[0] = (idx & 0xFFu) | (sc << 8u) | (vcs << 11u) | (aref << 16u);   // Meta
                    e[1] = it.stages[0]; e[2] = it.stages[1]; e[3] = it.stages[2]; e[4] = it.stages[3];
                    float* fe = (float*)e;
                    fe[5]  = it.matDiffuse[0];  fe[6]  = it.matDiffuse[1];  fe[7]  = it.matDiffuse[2];
                    fe[8]  = it.matAmbient[0];  fe[9]  = it.matAmbient[1];  fe[10] = it.matAmbient[2];
                    fe[11] = it.matEmissive[0]; fe[12] = it.matEmissive[1]; fe[13] = it.matEmissive[2];

                    const int mirror = worldMirrored(it.world) ? 1 : 0;
                    if (mirror != boundMMMirror) {
                        cmdBindPipeline(g_live.pCmd, mirror ? g_live.pMultiMapPrepassPipelineMirror
                                                            : g_live.pMultiMapPrepassPipeline);
                        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
                        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
                        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerBatchSetMM);
                        boundMMMirror = mirror;
                    }

                    HostMesh& mm = g_meshes[slot];
                    Buffer*  vbs[2]     = { mm.vb, g_live.pInstanceBufMM };
                    uint32_t strides[2] = { (uint32_t)sizeof(IPC::GeomVertexWireMM), (uint32_t)(kMMInstU32 * sizeof(uint32_t)) };
                    cmdBindVertexBuffer(g_live.pCmd, 2, vbs, strides, nullptr);
                    cmdBindIndexBuffer(g_live.pCmd, mm.ib, INDEX_TYPE_UINT16, 0);
                    cmdDrawIndexedInstanced(g_live.pCmd, mm.indexCount, 0, 1, 0, idx);   // firstInstance=idx
                    ++preDrawnMM;
                }
            }
        }

        // ===================== COLOUR PASS (early-Z: CMP_EQUAL, no depth write) =====================
        BindRenderTargetsDesc bind = {};
        bind.mRenderTargetCount = 1;
        bind.mRenderTargets[0] = { colorTarget, LOAD_ACTION_CLEAR };
        bind.mDepthStencil = { g_live.pDepth, LOAD_ACTION_LOAD };   // LOAD the prepass depth (no clear)
        cmdBindRenderTargets(g_live.pCmd, &bind);
        cmdSetViewport(g_live.pCmd, 0.0f, 0.0f, (float)g_live.width, (float)g_live.height, 0.0f, 1.0f);
        cmdSetScissor(g_live.pCmd, 0, 0, g_live.width, g_live.height);
        // Bind a pipeline FIRST — cmdBindPipeline establishes the root signature the descriptor
        // binds need. Start on the non-mirror colour pipeline; the loop switches per draw.
        cmdBindPipeline(g_live.pCmd, g_live.pOpaquePipeline);
        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);    // Tier 3a point lights
        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);   // bindless gTextures (static sampler)

        // Emit: one cmdExecuteIndirect per non-empty group. The arena VB/IB + instance buffer
        // are bound once per group; PerFrame/Persistent persist across the (same-root-sig) PSO
        // switch but are rebound on switch defensively (matches the skinned path).
        int boundMirror = 0;   // matches the initial cmdBindPipeline(pOpaquePipeline) above
        for (uint32_t mir = 0; mir < 2; ++mir) {
            bool anyInMirror = false;
            for (uint32_t b = 0; b < kMaxBatches; ++b) { if (groupCount[mir][b]) { anyInMirror = true; break; } }
            if (!anyInMirror) continue;
            if ((int)mir != boundMirror) {
                cmdBindPipeline(g_live.pCmd, mir ? g_live.pOpaquePipelineMirror
                                                 : g_live.pOpaquePipeline);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
                boundMirror = (int)mir;
            }
            for (uint32_t b = 0; b < kMaxBatches; ++b) {
                const uint32_t c = groupCount[mir][b];
                if (!c) continue;
                cmdBindDescriptorSet(g_live.pCmd, b, g_live.pPerBatchSet);
                Buffer*  vbs[2]     = { g_live.pArenaVB, g_live.pInstanceBuf[b] };
                uint32_t strides[2] = { vStride, iStride };
                cmdBindVertexBuffer(g_live.pCmd, 2, vbs, strides, nullptr);
                cmdBindIndexBuffer(g_live.pCmd, g_live.pArenaIB, INDEX_TYPE_UINT16, 0);
                cmdExecuteIndirect(g_live.pCmd, INDIRECT_DRAW_INDEX, c, g_live.pIndirectArgs,
                                   (uint64_t)groupOff[mir][b] * sizeof(IndirectDrawIndexArguments),
                                   nullptr, 0);
            }
        }

        // Dynamic-morph parts (own CPU_TO_GPU VB/IB): inline draws after the indirect groups.
        // Only ~1-4 per frame, so per-draw binds here are negligible.
        for (uint32_t i : s_dynamic) {
            const uint32_t slot  = items[i].slot;
            const uint32_t batch = i / kBatchSize;
            const uint32_t local = i % kBatchSize;
            const int mirror = worldMirrored(items[i].world) ? 1 : 0;
            if (mirror != boundMirror) {
                cmdBindPipeline(g_live.pCmd, mirror ? g_live.pOpaquePipelineMirror
                                                    : g_live.pOpaquePipeline);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
                boundMirror = mirror;
            }
            cmdBindDescriptorSet(g_live.pCmd, batch, g_live.pPerBatchSet);
            HostMesh& m = g_meshes[slot];
            Buffer*  vbs[2]     = { m.vb, g_live.pInstanceBuf[batch] };
            uint32_t strides[2] = { vStride, iStride };
            cmdBindVertexBuffer(g_live.pCmd, 2, vbs, strides, nullptr);
            cmdBindIndexBuffer(g_live.pCmd, m.ib, INDEX_TYPE_UINT16, 0);
            cmdDrawIndexedInstanced(g_live.pCmd, m.indexCount, 0, 1, 0, local);
        }

        // --- M-Skinning: skinned draw loop (GPU palette skinning) --------------------
        // The blob is [SkinnedDrawWire][palette]* (palette = numBones * 64 bytes, each a
        // model->world matrix). Each drawn part p packs its palette into bone window p/32
        // at base = (p%32)*32, then draws with firstInstance=base so the per-instance Base
        // attribute selects gBatch.worlds[base + BoneIdx]. Capped at kMaxSkinned (256).
        uint32_t skinnedDrawn = 0;
        if (skinnedBlob && skinnedCount && skinnedBytes &&
            g_live.pSkinnedPipeline && g_live.pSkinnedPipelineMirror) {
            cmdBindPipeline(g_live.pCmd, g_live.pSkinnedPipeline);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);

            const uint8_t* sp  = (const uint8_t*)skinnedBlob;
            const uint8_t* sEnd = sp + skinnedBytes;
            uint32_t boundWindow = UINT32_MAX;
            int      boundSkinMirror = 0;
            bool     dropLogged = false;
            for (uint32_t k = 0; k < skinnedCount; ++k) {
                if (sp + sizeof(IPC::SkinnedDrawWire) > sEnd) {
                    break;
                }
                IPC::SkinnedDrawWire item;
                std::memcpy(&item, sp, sizeof(item));
                const uint8_t* palette = sp + sizeof(item);
                const uint32_t bones = (item.numBones < kMaxBonesPerPart)
                                     ? item.numBones : kMaxBonesPerPart;
                const uint64_t paletteBytes = (uint64_t)item.numBones * 64;
                if (palette + paletteBytes > sEnd) {
                    break;   // truncated palette
                }
                sp = palette + paletteBytes;   // advance regardless of whether we draw

                if (skinnedDrawn >= kMaxSkinned) {
                    if (!dropLogged) {
                        LOGF(eWARNING, "[forge] skinned over cap %u — dropping extra parts (count=%u)",
                             kMaxSkinned, skinnedCount);
                        std::printf("[forge] skinned over cap %u — dropping extra parts (count=%u)\n",
                                    kMaxSkinned, skinnedCount);
                        dropLogged = true;
                    }
                    continue;
                }
                const uint32_t slot = item.slot;
                if (slot >= g_meshHigh || !g_meshes[slot].valid || !g_meshes[slot].skinned) {
                    continue;   // mesh not uploaded yet / not a skinned mesh
                }

                const uint32_t window = skinnedDrawn / kSkinnedPerWindow;
                const uint32_t base   = (skinnedDrawn % kSkinnedPerWindow) * kMaxBonesPerPart;
                uint8_t* dst = (uint8_t*)g_live.pBonesBuf[window]->pCpuMappedAddress;
                std::memcpy(dst + (size_t)base * 64, palette, (size_t)bones * 64);

                uint32_t* sinst = (uint32_t*)g_live.pInstanceBufSkin->pCpuMappedAddress;
                sinst[skinnedDrawn * 2 + 0] = base;
                sinst[skinnedDrawn * 2 + 1] = packTexAlpha(item.texIndex, item.alphaRef);

                const int mirror = item.mirror ? 1 : 0;
                if (mirror != boundSkinMirror) {
                    cmdBindPipeline(g_live.pCmd, mirror ? g_live.pSkinnedPipelineMirror
                                                        : g_live.pSkinnedPipeline);
                    boundSkinMirror = mirror;
                    boundWindow = UINT32_MAX;
                }
                if (window != boundWindow) {
                    cmdBindDescriptorSet(g_live.pCmd, window, g_live.pPerBatchSetSkin);
                    boundWindow = window;
                }

                HostMesh& sm = g_meshes[slot];
                Buffer*  vbs[2]     = { sm.vb, g_live.pInstanceBufSkin };
                uint32_t strides[2] = { (uint32_t)sizeof(IPC::SkinnedVertexWire), (uint32_t)(2 * sizeof(uint32_t)) };
                cmdBindVertexBuffer(g_live.pCmd, 2, vbs, strides, nullptr);
                cmdBindIndexBuffer(g_live.pCmd, sm.ib, INDEX_TYPE_UINT16, 0);
                cmdDrawIndexedInstanced(g_live.pCmd, sm.indexCount, 0, 1, 0, skinnedDrawn);
                ++skinnedDrawn;
            }
        }

        // --- Tier 4: multi-map draw loop (dark/detail/glow) --------------------------
        // MultiMapDrawWire[]: each part writes its world into pMMWorldsBuf[idx] and its per-draw
        // stage/material into pInstanceBufMM[idx], then draws with firstInstance=idx so the
        // per-instance Meta.DrawIndex selects gBatch.worlds[idx]. Capped at kMaxMultiMap (256).
        uint32_t multiMapDrawn = 0;
        if (multiMapBlob && multiMapCount && multiMapBytes &&
            g_live.pMultiMapPipeline && g_live.pMultiMapPipelineMirror) {
            cmdBindPipeline(g_live.pCmd, g_live.pMultiMapPipeline);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerBatchSetMM);   // single world window

            const uint32_t haveMM = multiMapBytes / (uint32_t)sizeof(IPC::MultiMapDrawWire);
            const uint32_t nMM = (multiMapCount < haveMM) ? multiMapCount : haveMM;
            const IPC::MultiMapDrawWire* mmItems = (const IPC::MultiMapDrawWire*)multiMapBlob;
            int  boundMMMirror = 0;
            bool mmDropLogged = false;
            for (uint32_t k = 0; k < nMM; ++k) {
                if (multiMapDrawn >= kMaxMultiMap) {
                    if (!mmDropLogged) {
                        LOGF(eWARNING, "[forge] multi-map over cap %u — dropping extra parts (count=%u)",
                             kMaxMultiMap, multiMapCount);
                        std::printf("[forge] multi-map over cap %u — dropping extra parts (count=%u)\n",
                                    kMaxMultiMap, multiMapCount);
                        mmDropLogged = true;
                    }
                    break;   // no fallback (PD1)
                }
                const IPC::MultiMapDrawWire& it = mmItems[k];
                const uint32_t slot = it.slot;
                if (slot >= g_meshHigh || !g_meshes[slot].valid || !g_meshes[slot].multimap) {
                    continue;   // mesh not uploaded yet / not a multi-map mesh
                }
                const uint32_t idx = multiMapDrawn;

                uint8_t* dst = (uint8_t*)g_live.pMMWorldsBuf->pCpuMappedAddress;
                std::memcpy(dst + (size_t)idx * 64, it.world, 64);

                uint32_t* inst = (uint32_t*)g_live.pInstanceBufMM->pCpuMappedAddress;
                uint32_t* e = inst + (size_t)idx * kMMInstU32;
                const uint32_t sc  = (it.stageCount > 4u) ? 4u : it.stageCount;
                float ar = it.alphaRef < 0.0f ? 0.0f : (it.alphaRef > 1.0f ? 1.0f : it.alphaRef);
                const uint32_t aref = (uint32_t)(ar * 255.0f + 0.5f) & 0xFFu;
                const uint32_t vcs  = it.vColSource & 0x3u;
                e[0] = (idx & 0xFFu) | (sc << 8u) | (vcs << 11u) | (aref << 16u);   // Meta
                e[1] = it.stages[0]; e[2] = it.stages[1]; e[3] = it.stages[2]; e[4] = it.stages[3];
                float* fe = (float*)e;
                fe[5]  = it.matDiffuse[0];  fe[6]  = it.matDiffuse[1];  fe[7]  = it.matDiffuse[2];
                fe[8]  = it.matAmbient[0];  fe[9]  = it.matAmbient[1];  fe[10] = it.matAmbient[2];
                fe[11] = it.matEmissive[0]; fe[12] = it.matEmissive[1]; fe[13] = it.matEmissive[2];

                const int mirror = worldMirrored(it.world) ? 1 : 0;
                if (mirror != boundMMMirror) {
                    cmdBindPipeline(g_live.pCmd, mirror ? g_live.pMultiMapPipelineMirror
                                                        : g_live.pMultiMapPipeline);
                    cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
                    cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);
                    cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
                    cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerBatchSetMM);
                    boundMMMirror = mirror;
                }

                HostMesh& mm = g_meshes[slot];
                Buffer*  vbs[2]     = { mm.vb, g_live.pInstanceBufMM };
                uint32_t strides[2] = { (uint32_t)sizeof(IPC::GeomVertexWireMM), (uint32_t)(kMMInstU32 * sizeof(uint32_t)) };
                cmdBindVertexBuffer(g_live.pCmd, 2, vbs, strides, nullptr);
                cmdBindIndexBuffer(g_live.pCmd, mm.ib, INDEX_TYPE_UINT16, 0);
                cmdDrawIndexedInstanced(g_live.pCmd, mm.indexCount, 0, 1, 0, idx);   // firstInstance=idx
                ++multiMapDrawn;
            }
        }

        cmdBindRenderTargets(g_live.pCmd, nullptr);

        if (g_live.sampleCount > 1) {
            // MSAA: resolve the multisampled color into the shared single-sample RT, then leave
            // the shared RT in COMMON for MW's D3D9Ex StretchRect. Forge has no RESOLVE resource
            // states, so this is driven natively (the shared RT is already managed natively).
            ID3D12GraphicsCommandList* cl = g_live.pCmd->mDx.pCmdList;
            ID3D12Resource* msaaRes = g_live.pMSAAColor->pTexture->mDx.pResource;
            ID3D12Resource* dstRes  = g_live.pSharedRes;

            D3D12_RESOURCE_BARRIER pre[2] = {};
            pre[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            pre[0].Transition.pResource = msaaRes;
            pre[0].Transition.Subresource = 0;
            pre[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            pre[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
            pre[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            pre[1].Transition.pResource = dstRes;
            pre[1].Transition.Subresource = 0;
            // First frame the shared RT was created RENDER_TARGET; steady state is COMMON.
            pre[1].Transition.StateBefore = g_live.firstFrame ? D3D12_RESOURCE_STATE_RENDER_TARGET
                                                              : D3D12_RESOURCE_STATE_COMMON;
            pre[1].Transition.StateAfter  = D3D12_RESOURCE_STATE_RESOLVE_DEST;
            cl->ResourceBarrier(2, pre);

            cl->ResolveSubresource(dstRes, 0, msaaRes, 0, DXGI_FORMAT_B8G8R8A8_UNORM);

            D3D12_RESOURCE_BARRIER post[2] = {};
            post[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            post[0].Transition.pResource = dstRes;
            post[0].Transition.Subresource = 0;
            post[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_DEST;
            post[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_COMMON;   // hand off to D3D9Ex
            post[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            post[1].Transition.pResource = msaaRes;
            post[1].Transition.Subresource = 0;
            post[1].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
            post[1].Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;   // ready for next frame
            cl->ResourceBarrier(2, post);
        } else {
            // No MSAA: hand the shared RT (rendered into directly) back to COMMON for StretchRect.
            RenderTargetBarrier toCommon = {};
            toCommon.pRenderTarget = g_live.pRT;
            toCommon.mCurrentState = RESOURCE_STATE_RENDER_TARGET;
            toCommon.mNewState = RESOURCE_STATE_COMMON;
            cmdResourceBarrier(g_live.pCmd, 0, nullptr, 0, nullptr, 1, &toCommon);
        }
        endCmd(g_live.pCmd);
        const double tRec1 = hostNowMs();   // end of CPU recording (record = tRec1 - tRec0)

        QueueSubmitDesc submitDesc = {};
        submitDesc.mCmdCount = 1;
        submitDesc.ppCmds = &g_live.pCmd;
        submitDesc.pSignalFence = g_live.pFence;
        submitDesc.mSubmitDone = true;
        queueSubmit(g_live.pQueue, &submitDesc);
        waitForFences(R, 1, &g_live.pFence);
        g_recAccum += (tRec1 - tRec0);            // CPU per-draw bind+draw recording
        g_gpuAccum += (hostNowMs() - tRec1);      // GPU execute (submit→fence)
        // A dense exterior frame is the suspected trigger; pin a removal to the draw submit.
        logDeviceRemoved(R, "renderScene/submit");

        g_live.firstFrame = false;
        g_lastDrawn = drawn;
        g_lastSkinnedDrawn = skinnedDrawn;
        g_lastMultiMapDrawn = multiMapDrawn;

        // Host-side heartbeat to mgeHost64.log (LOG::logline; LOGF goes to uncaptured stdout).
        // dynamic = meshes in the upload-heap ring (must stay tiny — hundreds = over-promotion);
        // meshHigh = total slots ever populated (monotonic leak check). Lets us correlate the
        // client's [hb] frame cost with what the Forge renderer is actually drawing.
        if ((g_renderFrame % 300u) == 0u) {
            LOG::logline(">> [forge-hb] frame=%u drawn=%u skinned=%u multimap=%u dynamic=%u meshHigh=%u "
                         "| record=%.2fms gpu=%.2fms (avg/frame over 300)",
                         g_renderFrame, drawn, skinnedDrawn, multiMapDrawn, g_dynamicCount, g_meshHigh,
                         g_recAccum / 300.0, g_gpuAccum / 300.0);
            g_recAccum = 0.0;
            g_gpuAccum = 0.0;
        }
        return true;
    }

    void setDebugMode(unsigned m) { g_debugMode = m; }

    unsigned lastDrawn() { return g_lastDrawn; }
    unsigned lastSkinnedDrawn() { return g_lastSkinnedDrawn; }
    unsigned lastMultiMapDrawn() { return g_lastMultiMapDrawn; }

    void debugReadbackCenterPixel() {
        if (!g_live.pRenderer || !g_live.pRT) {
            return;
        }
        Renderer* R = g_live.pRenderer;

        // Direct texture readback: copy the default-white texture (4x4 RGBA, uploaded via
        // TextureUpdateDesc) straight to a buffer — proves whether the UPLOAD worked, bypassing
        // the shader/descriptor binding entirely. White => upload ok (bug is binding); black =>
        // upload broken.
        if (g_live.pDefaultWhite) {
            const uint32_t rowA = (R->pGpu->mUploadBufferTextureRowAlignment > 1u) ? R->pGpu->mUploadBufferTextureRowAlignment : 1u;
            const uint32_t texA = (R->pGpu->mUploadBufferTextureAlignment > 1u) ? R->pGpu->mUploadBufferTextureAlignment : 1u;
            const uint32_t rp = roundUp(4u * 4u, rowA);   // default white is 4x4 RGBA8
            BufferLoadDesc tb = {};
            tb.mDesc.mSize = roundUp64((uint64_t)rp * 4u, texA);
            tb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_TO_CPU;
            tb.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            tb.mDesc.mStartState = RESOURCE_STATE_COPY_DEST;
            tb.mDesc.mQueueType = QUEUE_TYPE_TRANSFER;
            Buffer* pTexRb = nullptr; tb.ppBuffer = &pTexRb;
            addResource(&tb, nullptr); waitForAllResourceLoads();
            resetCmdPool(R, g_live.pCmdPool);
            beginCmd(g_live.pCmd);
            TextureBarrier tbar = { g_live.pDefaultWhite, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_COPY_SOURCE };
            cmdResourceBarrier(g_live.pCmd, 0, nullptr, 1, &tbar, 0, nullptr);
            endCmd(g_live.pCmd);
            QueueSubmitDesc s2 = {}; s2.mCmdCount = 1; s2.ppCmds = &g_live.pCmd; s2.pSignalFence = g_live.pFence; s2.mSubmitDone = true;
            queueSubmit(g_live.pQueue, &s2); waitForFences(R, 1, &g_live.pFence);
            TextureCopyDesc tc = {}; tc.pTexture = g_live.pDefaultWhite; tc.pBuffer = pTexRb;
            tc.mTextureState = RESOURCE_STATE_COPY_SOURCE; tc.mQueueType = QUEUE_TYPE_GRAPHICS;
            SyncToken tk = {}; copyResource(&tc, &tk); waitForToken(&tk);
            const uint8_t* tp = (const uint8_t*)pTexRb->pCpuMappedAddress;
            if (tp) { std::printf("[forge] scene-probe: defaultWhite texel (RGBA) = %u,%u,%u,%u\n", tp[0], tp[1], tp[2], tp[3]); }
            removeResource(pTexRb);
        }

        const uint32_t W = g_live.width, H = g_live.height;
        const uint32_t rowAlign = (R->pGpu->mUploadBufferTextureRowAlignment > 1u) ? R->pGpu->mUploadBufferTextureRowAlignment : 1u;
        const uint32_t texAlign = (R->pGpu->mUploadBufferTextureAlignment > 1u) ? R->pGpu->mUploadBufferTextureAlignment : 1u;
        const uint32_t rowPitch = roundUp(W * 4u, rowAlign);
        const uint64_t bufSize  = roundUp64((uint64_t)rowPitch * H, texAlign);
        BufferLoadDesc bd = {};
        bd.mDesc.mSize = bufSize;
        bd.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_TO_CPU;
        bd.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        bd.mDesc.mStartState = RESOURCE_STATE_COPY_DEST;
        bd.mDesc.mQueueType = QUEUE_TYPE_TRANSFER;
        Buffer* pReadback = nullptr;
        bd.ppBuffer = &pReadback;
        addResource(&bd, nullptr);
        waitForAllResourceLoads();
        resetCmdPool(R, g_live.pCmdPool);
        beginCmd(g_live.pCmd);
        RenderTargetBarrier b = {};
        b.pRenderTarget = g_live.pRT; b.mCurrentState = RESOURCE_STATE_COMMON; b.mNewState = RESOURCE_STATE_COPY_SOURCE;
        cmdResourceBarrier(g_live.pCmd, 0, nullptr, 0, nullptr, 1, &b);
        endCmd(g_live.pCmd);
        QueueSubmitDesc sd = {}; sd.mCmdCount = 1; sd.ppCmds = &g_live.pCmd; sd.pSignalFence = g_live.pFence; sd.mSubmitDone = true;
        queueSubmit(g_live.pQueue, &sd);
        waitForFences(R, 1, &g_live.pFence);
        TextureCopyDesc cd = {};
        cd.pTexture = g_live.pRT->pTexture; cd.pBuffer = pReadback;
        cd.mTextureState = RESOURCE_STATE_COPY_SOURCE; cd.mQueueType = QUEUE_TYPE_GRAPHICS;
        SyncToken ct = {};
        copyResource(&cd, &ct);
        waitForToken(&ct);
        const uint8_t* px = (const uint8_t*)pReadback->pCpuMappedAddress;
        if (px) {
            const uint8_t* c = &px[(uint64_t)(H / 2) * rowPitch + (uint64_t)(W / 2) * 4u];
            std::printf("[forge] scene-probe: CENTRE pixel (BGRA) = %u,%u,%u,%u\n", c[0], c[1], c[2], c[3]);
        } else {
            std::printf("[forge] scene-probe: readback not mapped\n");
        }
        removeResource(pReadback);
    }

    // --- Phase 2: DDS parse + bindless texture upload ---------------------------------
    static uint32_t ddsRd32(const uint8_t* p) {
        return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    }
    struct DdsInfo {
        TinyImageFormat fmt = TinyImageFormat_UNDEFINED;
        uint32_t width = 0, height = 0, mipLevels = 0, dataOffset = 0;
        bool ok = false;
    };
    // Parse the formats MW ships: DXT1/3/5 (BCn), uncompressed 32-bit (treated BGRA), and DX10
    // BC1/2/3 / BGRA / RGBA. Anything else → ok=false (slot stays default white).
    static DdsInfo parseDds(const uint8_t* d, uint32_t size) {
        DdsInfo r;
        if (size < 128 || ddsRd32(d) != 0x20534444u) { return r; }   // 'DDS '
        const uint32_t height  = ddsRd32(d + 12);
        const uint32_t width   = ddsRd32(d + 16);
        uint32_t       mips    = ddsRd32(d + 28);
        const uint32_t pfFlags = ddsRd32(d + 80);
        const uint32_t fourCC  = ddsRd32(d + 84);
        if (mips == 0) { mips = 1; }
        uint32_t dataOffset = 128;
        TinyImageFormat fmt = TinyImageFormat_UNDEFINED;
        const uint32_t DDPF_FOURCC = 0x4;
        if (pfFlags & DDPF_FOURCC) {
            switch (fourCC) {
                case 0x31545844u: fmt = TinyImageFormat_DXBC1_RGBA_UNORM; break;   // 'DXT1'
                case 0x33545844u: fmt = TinyImageFormat_DXBC2_UNORM;      break;   // 'DXT3'
                case 0x35545844u: fmt = TinyImageFormat_DXBC3_UNORM;      break;   // 'DXT5'
                case 0x30315844u: {                                               // 'DX10'
                    if (size < 148) { return r; }
                    const uint32_t dxgi = ddsRd32(d + 128);
                    dataOffset = 148;
                    switch (dxgi) {
                        case 71: case 72: fmt = TinyImageFormat_DXBC1_RGBA_UNORM; break;  // BC1(_SRGB)
                        case 74: case 75: fmt = TinyImageFormat_DXBC2_UNORM;      break;  // BC2
                        case 77: case 78: fmt = TinyImageFormat_DXBC3_UNORM;      break;  // BC3
                        case 87: case 91: fmt = TinyImageFormat_B8G8R8A8_UNORM;   break;  // BGRA8
                        case 28: case 29: fmt = TinyImageFormat_R8G8B8A8_UNORM;   break;  // RGBA8
                        default: break;
                    }
                } break;
                default: break;
            }
        } else {
            const uint32_t bits = ddsRd32(d + 88);   // dwRGBBitCount
            if (bits == 32) { fmt = TinyImageFormat_B8G8R8A8_UNORM; }   // MW A8R8G8B8
        }
        if (fmt == TinyImageFormat_UNDEFINED || width == 0 || height == 0) { return r; }
        r.fmt = fmt; r.width = width; r.height = height; r.mipLevels = mips;
        r.dataOffset = dataOffset; r.ok = true;
        return r;
    }

    // Parse [TexUploadWire][dds bytes]* and decode each into gTextures[slot]. slot 0 is reserved
    // (default white). Returns the number of textures successfully built. No-op if Forge/the
    // opaque path isn't live yet (textures arrive after the first scene builds the PerFrame set).
    unsigned uploadTextures(const void* blob, unsigned byteCount, unsigned count) {
        // The PerFrame set (which owns the bindless array) is built lazily on the first
        // renderScene. Until then, signal "not ready" (0xFFFFFFFF) so the client keeps the
        // batch and retries next frame, rather than silently dropping textures.
        if (g_live.pRenderer && !g_live.pPersistentSet) {
            return 0xFFFFFFFFu;
        }
        if (!g_live.pRenderer || !blob || !byteCount || !count) {
            return 0;
        }
        Renderer* R = g_live.pRenderer;
        const uint8_t* p   = (const uint8_t*)blob;
        const uint8_t* end = p + byteCount;
        unsigned built = 0;

        for (unsigned i = 0; i < count; ++i) {
            if (p + sizeof(IPC::TexUploadWire) > end) { break; }
            IPC::TexUploadWire hdr;
            std::memcpy(&hdr, p, sizeof(hdr));
            const uint8_t* dds    = p + sizeof(hdr);
            const uint8_t* ddsEnd = dds + hdr.byteLen;
            if (ddsEnd > end) { break; }
            p = ddsEnd;   // advance regardless of whether this one decodes

            if (hdr.slot == 0 || hdr.slot >= kMaxTextures) { continue; }   // 0 reserved; OOB dropped
            DdsInfo info = parseDds(dds, hdr.byteLen);
            if (!info.ok) {
                LOGF(eWARNING, "[forge] tex slot %u: unsupported DDS format (slot stays white)", hdr.slot);
                continue;
            }

            Texture* tex = nullptr;
            TextureDesc td = {};
            td.mWidth = info.width; td.mHeight = info.height; td.mDepth = 1;
            td.mArraySize = 1; td.mMipLevels = info.mipLevels;
            td.mSampleCount = SAMPLE_COUNT_1;
            td.mFormat = info.fmt;
            td.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
            td.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
            td.pName = "mwTexture";
            TextureLoadDesc tld = {};
            tld.ppTexture = &tex;
            tld.pDesc = &td;
            addResource(&tld, nullptr);
            waitForAllResourceLoads();
            if (!tex) { continue; }

            // Upload the tightly-packed DDS mip chain into the (row-aligned) GPU texture.
            const uint8_t* src = dds + info.dataOffset;
            TextureUpdateDesc upd = {};
            upd.pTexture = tex;
            upd.mBaseMipLevel = 0; upd.mMipLevels = info.mipLevels;
            upd.mBaseArrayLayer = 0; upd.mLayerCount = 1;
            upd.mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;   // created in & returned to this state
            beginUpdateResource(&upd);
            for (uint32_t m = 0; m < info.mipLevels; ++m) {
                TextureSubresourceUpdate s = upd.getSubresourceUpdateDesc(m, 0);
                const uint32_t mipBytes = s.mRowCount * s.mSrcRowStride;
                if (src + mipBytes > ddsEnd) { break; }   // truncated; stop copying mips
                for (uint32_t row = 0; row < s.mRowCount; ++row) {
                    std::memcpy(s.pMappedData + (size_t)row * s.mDstRowStride,
                                src + (size_t)row * s.mSrcRowStride, s.mSrcRowStride);
                }
                src += mipBytes;
            }
            endUpdateResource(&upd);
            // NOTE: the upload-engine copy is submitted ONCE for the whole batch after the loop
            // (flushTextureUploads below) — not per texture. A per-texture GPU fence stalled the
            // whole batch N times (the "slow" with hundreds of new textures on area load).

            // Replace any prior texture in this slot (revision re-upload), store, rebind the
            // single bindless descriptor. The rebind only copies CPU descriptor handles; the GPU
            // doesn't read them until renderScene (its own fence). Safe between frames.
            if (g_live.pTextures[hdr.slot] != g_live.pDefaultWhite) {
                removeResource(g_live.pTextures[hdr.slot]);
            }
            g_live.pTextures[hdr.slot] = tex;
            if (hdr.slot + 1 > g_live.texHigh) { g_live.texHigh = hdr.slot + 1; }

            DescriptorData dd = {};   // rebind just this slot in the bindless array
            dd.mIndex = SRT_RES_IDX(SrtData, Persistent, gTextures);
            dd.mArrayOffset = hdr.slot;
            dd.mCount = 1;
            dd.ppTextures = &g_live.pTextures[hdr.slot];
            updateDescriptorSet(R, 0, g_live.pPersistentSet, 1, &dd);
            ++built;
        }
        // One upload-engine flush for the whole batch (else textures stay black). Replaces the
        // former per-texture fence — the batch is window-bounded so this is a single short wait.
        if (built) { flushTextureUploads(R); }
        return built;
    }

    // Create the shared mega VB/IB on first use. Must NOT live in buildOpaquePath: geometry
    // uploads run BEFORE the lazy buildOpaquePath (first renderScene), so the arena has to exist
    // at upload time. Idempotent.
    bool ensureArena() {
        if (g_live.pArenaVB && g_live.pArenaIB) { return true; }
        if (!g_live.pRenderer) { return false; }
        BufferLoadDesc av = {};
        av.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
        av.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        av.mDesc.mSize        = kArenaVBBytes;
        av.mDesc.mStartState  = RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        av.mDesc.pName        = "arenaVB";
        av.pData              = nullptr;
        av.ppBuffer           = &g_live.pArenaVB;
        addResource(&av, nullptr);

        BufferLoadDesc ai = {};
        ai.mDesc.mDescriptors = DESCRIPTOR_TYPE_INDEX_BUFFER;
        ai.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        ai.mDesc.mSize        = kArenaIBBytes;
        ai.mDesc.mStartState  = RESOURCE_STATE_INDEX_BUFFER;
        ai.mDesc.pName        = "arenaIB";
        ai.pData              = nullptr;
        ai.ppBuffer           = &g_live.pArenaIB;
        addResource(&ai, nullptr);

        waitForAllResourceLoads();
        if (!g_live.pArenaVB || !g_live.pArenaIB) { return false; }
        g_arenaVB.init(kArenaVBBytes);
        g_arenaIB.init(kArenaIBBytes);
        return true;
    }

    unsigned uploadGeometry(const void* blobBytes, unsigned byteCount, unsigned partCount) {
        if (!g_live.pRenderer || !blobBytes || !byteCount || !partCount) {
            return 0;
        }
        if (!ensureArena()) { return 0; }
        const uint8_t* p   = (const uint8_t*)blobBytes;
        const uint8_t* end = p + byteCount;
        unsigned built = 0;
        bool anyStatic = false;   // any per-mesh addResource (skinned) -> waitForAllResourceLoads
        bool anyArena  = false;   // any arena begin/endUpdateResource -> flushResourceUpdates

        for (unsigned i = 0; i < partCount; ++i) {
            if (p + sizeof(IPC::GeomPartWire) > end) {
                break;
            }
            IPC::GeomPartWire hdr;
            std::memcpy(&hdr, p, sizeof(hdr));
            p += sizeof(hdr);

            const bool isSkinned  = (hdr.flags & IPC::kGeomFlagSkinned) != 0;
            const bool isMultiMap = (hdr.flags & IPC::kGeomFlagMultiMap) != 0;
            const uint64_t vStride = isSkinned  ? sizeof(IPC::SkinnedVertexWire)
                                   : isMultiMap ? sizeof(IPC::GeomVertexWireMM)
                                                : sizeof(IPC::GeomVertexWire);
            const uint64_t vbBytes = (uint64_t)hdr.vertexCount * vStride;
            const uint64_t ibBytes = (uint64_t)hdr.indexCount * sizeof(uint16_t);
            if (p + vbBytes + ibBytes > end) {
                break;  // malformed / truncated blob
            }
            const void* verts   = p; p += vbBytes;
            const void* indices = p; p += ibBytes;
            if (!hdr.vertexCount || !hdr.indexCount) {
                continue;
            }
            if (!ensureMeshSlot(hdr.slot)) {
                continue;
            }

            HostMesh& m = g_meshes[hdr.slot];

            // A re-upload at the SAME shape (vertex/index count + skinned flag) MIGHT be an
            // animated morph. But only a CONSECUTIVE-frame streak proves it — promote those to
            // the upload-heap ring (fence-free memcpy). Non-consecutive re-uploads (cell churn /
            // recycled slots) reset the streak and stay GPU_ONLY: an upload-heap VB has
            // high-latency uncached GPU vertex fetch, so mass-promoting churn tanks render.
            const bool sameShape = m.valid && m.vertexCount == hdr.vertexCount
                                && m.indexCount == hdr.indexCount && m.skinned == isSkinned
                                && m.multimap == isMultiMap;
            const bool consecutive = m.valid && (g_renderFrame == m.lastUploadFrame + 1);

            if (m.valid && !sameShape) {
                releaseMeshBuffers(m);
                m.valid = false;
                m.uploadStreak = 0;
            }

            if (sameShape) {
                m.uploadStreak    = consecutive ? (uint16_t)(m.uploadStreak + 1) : 0;
                m.lastUploadFrame = g_renderFrame;

                if (m.dynamic) {
                    // Already in the ring — just memcpy (no recreate, no fence).
                    const uint8_t r = m.ring;
                    std::memcpy(m.dynVb[r]->pCpuMappedAddress, verts,   (size_t)vbBytes);
                    std::memcpy(m.dynIb[r]->pCpuMappedAddress, indices, (size_t)ibBytes);
                    m.vb   = m.dynVb[r];
                    m.ib   = m.dynIb[r];
                    m.ring = (uint8_t)((r + 1) % kGeomRing);
                    ++built;
                    continue;
                }

                if (m.uploadStreak >= kDynPromoteStreak) {
                    // PROMOTE: a proven per-frame morph. Free the static GPU_ONLY buffers, build
                    // the persistent-mapped ring (one-time fence here only), then memcpy.
                    releaseMeshBuffers(m);
                    bool ringOk = true;
                    for (uint32_t r = 0; r < kGeomRing; ++r) {
                        BufferLoadDesc dv = {};
                        dv.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
                        dv.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
                        dv.mDesc.mFlags       = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
                        dv.mDesc.mSize        = vbBytes;
                        dv.mDesc.pName        = "geomVBdyn";
                        dv.pData              = nullptr;
                        dv.ppBuffer           = &m.dynVb[r];
                        addResource(&dv, nullptr);

                        BufferLoadDesc di = {};
                        di.mDesc.mDescriptors = DESCRIPTOR_TYPE_INDEX_BUFFER;
                        di.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
                        di.mDesc.mFlags       = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
                        di.mDesc.mSize        = ibBytes;
                        di.mDesc.pName        = "geomIBdyn";
                        di.pData              = nullptr;
                        di.ppBuffer           = &m.dynIb[r];
                        addResource(&di, nullptr);
                    }
                    waitForAllResourceLoads();   // one-time, on promotion only
                    for (uint32_t r = 0; r < kGeomRing; ++r) {
                        if (!m.dynVb[r] || !m.dynIb[r]) { ringOk = false; }
                    }
                    if (!ringOk) { releaseMeshBuffers(m); m.valid = false; m.uploadStreak = 0; continue; }
                    m.dynamic = true;
                    m.ring    = 0;
                    ++g_dynamicCount;
                    const uint8_t r = m.ring;
                    std::memcpy(m.dynVb[r]->pCpuMappedAddress, verts,   (size_t)vbBytes);
                    std::memcpy(m.dynIb[r]->pCpuMappedAddress, indices, (size_t)ibBytes);
                    m.vb   = m.dynVb[r];
                    m.ib   = m.dynIb[r];
                    m.ring = (uint8_t)((r + 1) % kGeomRing);
                    ++built;
                    continue;
                }

                // Same shape but not yet a proven morph: rebuild as static GPU_ONLY (fast reads).
                // A couple of recreate+fence frames at morph onset, then it promotes. (Cell churn
                // never gets here twice — the streak resets, so it just rebuilds static once.)
                releaseMeshBuffers(m);
                m.valid = false;
            }

            // --- static build: PLAIN STATIC -> shared mega arena (bind-once, offset draws);
            //     SKINNED / MULTI-MAP -> per-mesh GPU_ONLY (wide/odd stride, own VB) ---
            m.skinned  = isSkinned;
            m.multimap = isMultiMap;

            if (!isSkinned && !isMultiMap) {
                // Sub-allocate the mega VB + IB; write the sub-ranges via BufferUpdateDesc.
                uint64_t vbo = g_arenaVB.alloc(vbBytes);
                uint64_t ibo = g_arenaIB.alloc(ibBytes);
                if (vbo == UINT64_MAX || ibo == UINT64_MAX) {
                    if (vbo != UINT64_MAX) { g_arenaVB.release(vbo, vbBytes); }
                    if (ibo != UINT64_MAX) { g_arenaIB.release(ibo, ibBytes); }
                    static bool warned = false;
                    if (!warned) {
                        LOG::logline("!! [forge] geometry arena FULL (need %llu VB / %llu IB) — "
                                     "part skipped (no fallback); eviction is the follow-up",
                                     (unsigned long long)vbBytes, (unsigned long long)ibBytes);
                        warned = true;
                    }
                    continue;   // part won't draw this slot — logged, no fallback (PD1)
                }
                BufferUpdateDesc uv = {};
                uv.pBuffer    = g_live.pArenaVB;
                uv.mDstOffset = vbo;
                uv.mSize      = vbBytes;
                beginUpdateResource(&uv);
                std::memcpy(uv.pMappedData, verts, (size_t)vbBytes);
                endUpdateResource(&uv);

                BufferUpdateDesc ui = {};
                ui.pBuffer    = g_live.pArenaIB;
                ui.mDstOffset = ibo;
                ui.mSize      = ibBytes;
                beginUpdateResource(&ui);
                std::memcpy(ui.pMappedData, indices, (size_t)ibBytes);
                endUpdateResource(&ui);

                m.inArena = true;
                m.vbOff   = vbo;
                m.ibOff   = ibo;
                m.vb = m.ib = nullptr;
                anyArena = true;
            } else {
                m.inArena = false;
                BufferLoadDesc vbDesc = {};
                vbDesc.mDesc.mDescriptors  = DESCRIPTOR_TYPE_VERTEX_BUFFER;
                vbDesc.mDesc.mMemoryUsage  = RESOURCE_MEMORY_USAGE_GPU_ONLY;
                vbDesc.mDesc.mSize         = vbBytes;
                vbDesc.mDesc.mStartState   = RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
                vbDesc.mDesc.pName         = "geomVBskin";
                vbDesc.pData               = verts;
                vbDesc.ppBuffer            = &m.vb;
                addResource(&vbDesc, nullptr);

                BufferLoadDesc ibDesc = {};
                ibDesc.mDesc.mDescriptors  = DESCRIPTOR_TYPE_INDEX_BUFFER;
                ibDesc.mDesc.mMemoryUsage  = RESOURCE_MEMORY_USAGE_GPU_ONLY;
                ibDesc.mDesc.mSize         = ibBytes;
                ibDesc.mDesc.mStartState   = RESOURCE_STATE_INDEX_BUFFER;
                ibDesc.mDesc.pName         = "geomIBskin";
                ibDesc.pData               = indices;
                ibDesc.ppBuffer            = &m.ib;
                addResource(&ibDesc, nullptr);
                anyStatic = true;
            }

            m.vertexCount    = hdr.vertexCount;
            m.indexCount     = hdr.indexCount;
            m.valid          = true;
            m.dynamic        = false;
            m.lastUploadFrame = g_renderFrame;   // seed the consecutive-frame streak detector
            if (hdr.slot + 1 > g_meshHigh) {
                g_meshHigh = hdr.slot + 1;
            }
            ++built;
        }

        // Only fence/log when a static GPU_ONLY build happened. The steady-state dynamic path
        // (animated meshes) is pure memcpy into persistent buffers — no loader work to wait on,
        // and logging it every frame is the very per-frame cost we're removing.
        if (anyStatic) {
            waitForAllResourceLoads();   // flush per-mesh (skinned) addResource loads
        }
        if (anyArena) {
            // begin/endUpdateResource records on the loader's UPDATE stream, which
            // waitForAllResourceLoads does NOT flush — must flushResourceUpdates + fence (same
            // gotcha as texture uploads), else arena geometry stays zero.
            flushTextureUploads(g_live.pRenderer);
        }
        if (anyStatic || anyArena) {
            // Exterior uploads ship ~1000+ parts in one batch; this is where a prior in-game
            // test went DEVICE_REMOVED. Pin a removal to the upload (vs the later draw).
            logDeviceRemoved(g_live.pRenderer, "uploadGeometry/flush");
            LOGF(eINFO, "[forge] uploadGeometry: built %u/%u parts (%u bytes), meshHigh=%u",
                 built, partCount, byteCount, g_meshHigh);
            std::printf("[forge] uploadGeometry: built %u/%u parts (%u bytes), meshHigh=%u\n",
                        built, partCount, byteCount, g_meshHigh);
        }
        return built;
    }

    void shutdown() {
        Renderer* R = g_live.pRenderer;
        if (!R) {
            freeMeshStore();
            g_live = LiveRenderer{};
            return;
        }
        freeMeshStore();   // release VB/IB before the resource loader goes down
        // M1c opaque path teardown.
        if (g_live.pPerFrameSet)    { removeDescriptorSet(R, g_live.pPerFrameSet); }
        if (g_live.pPerLightsSet)   { removeDescriptorSet(R, g_live.pPerLightsSet); }
        if (g_live.pPerBatchSet)    { removeDescriptorSet(R, g_live.pPerBatchSet); }
        if (g_live.pPersistentSet)  { removeDescriptorSet(R, g_live.pPersistentSet); }
        if (g_live.pFrameCbv)       { removeResource(g_live.pFrameCbv); }
        if (g_live.pLightCbv)       { removeResource(g_live.pLightCbv); }
        if (g_live.pArenaVB)        { removeResource(g_live.pArenaVB); }
        if (g_live.pArenaIB)        { removeResource(g_live.pArenaIB); }
        if (g_live.pIndirectArgs)   { removeResource(g_live.pIndirectArgs); }
        g_arenaVB = FreeList{};
        g_arenaIB = FreeList{};
        for (uint32_t b = 0; b < kMaxBatches; ++b) {
            if (g_live.pWorldsBuf[b]) { removeResource(g_live.pWorldsBuf[b]); }
        }
        for (uint32_t b = 0; b < kMaxBatches; ++b) {
            if (g_live.pInstanceBuf[b]) { removeResource(g_live.pInstanceBuf[b]); }
        }
        // Phase 2 texture teardown: distinct uploaded textures, then the shared default.
        for (uint32_t i = 0; i < g_live.texHigh; ++i) {
            if (g_live.pTextures[i] && g_live.pTextures[i] != g_live.pDefaultWhite) {
                removeResource(g_live.pTextures[i]);
            }
        }
        if (g_live.pDefaultWhite)   { removeResource(g_live.pDefaultWhite); }
        if (g_live.pOpaquePipeline) { removePipeline(R, g_live.pOpaquePipeline); }
        if (g_live.pOpaquePipelineMirror) { removePipeline(R, g_live.pOpaquePipelineMirror); }
        if (g_live.pOpaquePrepassPipeline)       { removePipeline(R, g_live.pOpaquePrepassPipeline); }
        if (g_live.pOpaquePrepassPipelineMirror) { removePipeline(R, g_live.pOpaquePrepassPipelineMirror); }
        if (g_live.pDepthOnlyShader) { removeShader(R, g_live.pDepthOnlyShader); }
        if (g_live.pOpaqueShader)   { removeShader(R, g_live.pOpaqueShader); }
        // M-Skinning teardown.
        if (g_live.pPerBatchSetSkin) { removeDescriptorSet(R, g_live.pPerBatchSetSkin); }
        for (uint32_t b = 0; b < kMaxBatches; ++b) {
            if (g_live.pBonesBuf[b]) { removeResource(g_live.pBonesBuf[b]); }
        }
        if (g_live.pInstanceBufSkin) { removeResource(g_live.pInstanceBufSkin); }
        if (g_live.pSkinnedPipeline)       { removePipeline(R, g_live.pSkinnedPipeline); }
        if (g_live.pSkinnedPipelineMirror) { removePipeline(R, g_live.pSkinnedPipelineMirror); }
        if (g_live.pSkinnedPrepassPipeline)       { removePipeline(R, g_live.pSkinnedPrepassPipeline); }
        if (g_live.pSkinnedPrepassPipelineMirror) { removePipeline(R, g_live.pSkinnedPrepassPipelineMirror); }
        if (g_live.pSkinnedDepthShader)    { removeShader(R, g_live.pSkinnedDepthShader); }
        if (g_live.pSkinnedShader)         { removeShader(R, g_live.pSkinnedShader); }
        // Tier 4 multi-map teardown.
        if (g_live.pPerBatchSetMM)          { removeDescriptorSet(R, g_live.pPerBatchSetMM); }
        if (g_live.pMMWorldsBuf)            { removeResource(g_live.pMMWorldsBuf); }
        if (g_live.pInstanceBufMM)          { removeResource(g_live.pInstanceBufMM); }
        if (g_live.pMultiMapPipeline)       { removePipeline(R, g_live.pMultiMapPipeline); }
        if (g_live.pMultiMapPipelineMirror) { removePipeline(R, g_live.pMultiMapPipelineMirror); }
        if (g_live.pMultiMapPrepassPipeline)       { removePipeline(R, g_live.pMultiMapPrepassPipeline); }
        if (g_live.pMultiMapPrepassPipelineMirror) { removePipeline(R, g_live.pMultiMapPrepassPipelineMirror); }
        if (g_live.pMultiMapDepthShader)    { removeShader(R, g_live.pMultiMapDepthShader); }
        if (g_live.pMultiMapShader)         { removeShader(R, g_live.pMultiMapShader); }
        if (g_live.pMSAAColor)      { removeRenderTarget(R, g_live.pMSAAColor); }
        if (g_live.pDepth)          { removeRenderTarget(R, g_live.pDepth); }
        if (g_live.pFence)    { exitFence(R, g_live.pFence); }
        if (g_live.pCmd)      { exitCmd(R, g_live.pCmd); }
        if (g_live.pCmdPool)  { exitCmdPool(R, g_live.pCmdPool); }
        if (g_live.pPipeline) { removePipeline(R, g_live.pPipeline); }
        if (g_live.ntHandle)  { CloseHandle(g_live.ntHandle); }
        if (g_live.pRT)       { removeRenderTarget(R, g_live.pRT); }  // releases pSharedRes
        if (g_live.pShader)   { removeShader(R, g_live.pShader); exitRootSignature(R); }
        forgeTearDown(R, g_live.pQueue);
        g_live = LiveRenderer{};
        std::printf("[forge] live shutdown complete\n");
    }
}
