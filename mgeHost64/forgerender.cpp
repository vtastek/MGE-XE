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

#include <cstdio>
#include <cstdint>
#include <cstring>

#include "OS/Interfaces/IOperatingSystem.h"
#include "Utilities/Interfaces/IFileSystem.h"
#include "Utilities/Log/Log.h"
#include "Graphics/GraphicsConfig.h"
#include "Graphics/Interfaces/IGraphics.h"
#include "Resources/ResourceLoader/Interfaces/IResourceLoader.h"
#include "Utilities/Interfaces/ILog.h"
// IMemory.h overrides new/delete/malloc — Forge convention: include it LAST.
#include "Utilities/Interfaces/IMemory.h"
// M1c opaque-scene SRT. defaults.h provides the C++ definitions of the FSL macros
// (STRUCT/DATA/BEGIN_SRT/DECL_CBUFFER/SRT_SET_DESC/SRT_RES_IDX); the .srt.h then
// declares SRT_SrtData + the gFrameData/gObject descriptor indices. Per the Forge
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

        // --- M1c opaque scene path ---
        RenderTarget*  pDepth = nullptr;          // depth buffer for the scene
        Shader*        pOpaqueShader = nullptr;
        Pipeline*      pOpaquePipeline = nullptr;
        DescriptorSet* pPerFrameSet = nullptr;    // gFrameData (viewProj)
        DescriptorSet* pPerDrawSet = nullptr;     // gObject (world), one instance per draw
        Buffer*        pFrameCbv = nullptr;       // viewProj, persistent-mapped
        Buffer*        pObjectCbv = nullptr;      // maxDraws * kObjStride, persistent-mapped
        uint32_t       maxDraws = 0;
    };
    LiveRenderer g_live;

    // CBV instances must be 256-byte aligned; pad the 64-byte world matrix up.
    // kMaxDraws temporarily small while bisecting the uploadGeometry crash (large
    // PerDraw descriptor sets suspected). Raise once the per-draw path is proven.
    constexpr uint32_t kObjStride = 256;
    constexpr uint32_t kMaxDraws  = 256;

    // Build the M1c opaque scene path: depth target, opaque shader + pipeline (pos+normal
    // vertex layout, depth test/write), and the PerFrame (viewProj) + PerDraw (world)
    // descriptor sets backed by persistent-mapped CBVs. The opaque shaders share the
    // global default.rootsig (regenerated from the SRT in opaque.srt.h). On any failure
    // tears down what it made and returns false (the triangle path stays usable).
    bool buildOpaquePath(Renderer* R, uint32_t width, uint32_t height) {
        // Depth target (reverse-Z not needed for M1c; standard LEQUAL + clear to 1.0).
        RenderTargetDesc dDesc = {};
        dDesc.mWidth = width;
        dDesc.mHeight = height;
        dDesc.mDepth = 1;
        dDesc.mArraySize = 1;
        dDesc.mMipLevels = 1;
        dDesc.mSampleCount = SAMPLE_COUNT_1;
        dDesc.mFormat = TinyImageFormat_D32_SFLOAT;
        dDesc.mStartState = RESOURCE_STATE_DEPTH_WRITE;
        dDesc.mClearValue.depth = 1.0f;
        dDesc.mClearValue.stencil = 0;
        dDesc.pName = "sceneDepth";
        addRenderTarget(R, &dDesc, &g_live.pDepth);
        if (!g_live.pDepth) {
            return false;
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

        // Vertex layout matches IPC::GeomVertexWire: pos float3 @0, normal float3 @12, stride 24.
        VertexLayout vl = {};
        vl.mBindingCount = 1;
        vl.mBindings[0].mStride = sizeof(IPC::GeomVertexWire);
        vl.mAttribCount = 2;
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

        DepthStateDesc depthDesc = {};
        depthDesc.mDepthTest = true;
        depthDesc.mDepthWrite = true;
        depthDesc.mDepthFunc = CMP_LEQUAL;

        RasterizerStateDesc rasterDesc = {};
        rasterDesc.mCullMode = CULL_MODE_NONE;   // M1c: no winding assumptions yet

        PipelineDesc pd = {};
        pd.mType = PIPELINE_TYPE_GRAPHICS;
        GraphicsPipelineDesc& g = pd.mGraphicsDesc;
        g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
        g.mRenderTargetCount = 1;
        g.pColorFormats = &g_live.pRT->mFormat;
        g.mSampleCount = SAMPLE_COUNT_1;
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

        // PerFrame viewProj CBV (one), persistent-mapped.
        BufferLoadDesc fb = {};
        fb.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        fb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        fb.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        fb.mDesc.mSize = kObjStride;   // 256, holds one float4x4
        fb.pData = nullptr;
        fb.ppBuffer = &g_live.pFrameCbv;
        addResource(&fb, nullptr);

        // PerDraw world CBV array (kMaxDraws slices of kObjStride), persistent-mapped.
        BufferLoadDesc ob = {};
        ob.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ob.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        ob.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        ob.mDesc.mSize = (uint64_t)kObjStride * kMaxDraws;
        ob.pData = nullptr;
        ob.ppBuffer = &g_live.pObjectCbv;
        addResource(&ob, nullptr);
        waitForAllResourceLoads();
        if (!g_live.pFrameCbv || !g_live.pObjectCbv) {
            return false;
        }
        g_live.maxDraws = kMaxDraws;

        // Descriptor sets from the SRT (PerFrame: 1 instance; PerDraw: kMaxDraws).
        DescriptorSetDesc pfDesc = SRT_SET_DESC(SrtData, PerFrame, 1, 0);
        addDescriptorSet(R, &pfDesc, &g_live.pPerFrameSet);
        DescriptorSetDesc pdDesc = SRT_SET_DESC(SrtData, PerDraw, kMaxDraws, 0);
        addDescriptorSet(R, &pdDesc, &g_live.pPerDrawSet);
        if (!g_live.pPerFrameSet || !g_live.pPerDrawSet) {
            return false;
        }

        // Bind the frame CBV to PerFrame instance 0.
        {
            DescriptorData p = {};
            p.mIndex = SRT_RES_IDX(SrtData, PerFrame, gFrameData);
            p.ppBuffers = &g_live.pFrameCbv;
            updateDescriptorSet(R, 0, g_live.pPerFrameSet, 1, &p);
        }
        // Bind each PerDraw instance i to object-buffer slice [i*kObjStride, +kObjStride).
        for (uint32_t i = 0; i < kMaxDraws; ++i) {
            DescriptorDataRange range = {};
            range.mOffset = i * kObjStride;
            range.mSize = kObjStride;
            DescriptorData p = {};
            p.mIndex = SRT_RES_IDX(SrtData, PerDraw, gObject);
            p.ppBuffers = &g_live.pObjectCbv;
            p.pRanges = &range;
            updateDescriptorSet(R, i, g_live.pPerDrawSet, 1, &p);
        }

        std::printf("[forge] opaque scene path ready (depth %ux%u, maxDraws=%u)\n", width, height, kMaxDraws);
        return true;
    }

    // --- M1b: slot-indexed static opaque mesh store ------------------------------
    // The client assigns each cached NiTriShape* a dense slot; we keep meshes in a
    // flat array indexed by that slot (no hashing). Grown on demand; freed on
    // shutdown. M1c iterates a per-frame visible list of slots to draw these.
    struct HostMesh {
        Buffer*  vb;
        Buffer*  ib;
        uint32_t vertexCount;
        uint32_t indexCount;
        bool     valid;
    };
    HostMesh* g_meshes   = nullptr;
    uint32_t  g_meshCap  = 0;   // allocated slot count
    uint32_t  g_meshHigh = 0;   // highest slot+1 ever populated
    unsigned  g_lastDrawn = 0;  // parts actually drawn in the last renderScene

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

    void freeMeshStore() {
        if (g_meshes) {
            for (uint32_t i = 0; i < g_meshHigh; ++i) {
                if (g_meshes[i].valid) {
                    if (g_meshes[i].vb) { removeResource(g_meshes[i].vb); }
                    if (g_meshes[i].ib) { removeResource(g_meshes[i].ib); }
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
    bool init(unsigned width, unsigned height) {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        if (g_live.pRenderer) {
            shutdown();
        }
        std::printf("[forge] live init %ux%u...\n", width, height);

        if (!forgeBringUp(&g_live.pRenderer, &g_live.pQueue)) {
            return false;
        }
        Renderer* R = g_live.pRenderer;

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
        clearVal.Color[3] = 1.0f;

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
        // M1c debug: teal clear so F11 distinguishes "composite shows the RT but
        // geometry is off-screen" (teal screen) from "composite/RT broken" (black).
        rtDesc.mClearValue.r = 0.0f;
        rtDesc.mClearValue.g = 0.30f;
        rtDesc.mClearValue.b = 0.35f;
        rtDesc.mClearValue.a = 1.0f;
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

    bool renderScene(const float* viewProj, const void* drawBlob,
                     unsigned drawCount, unsigned drawBytes) {
        if (!g_live.pRenderer) {
            return false;
        }
        Renderer* R = g_live.pRenderer;

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

        const uint32_t n = (drawCount < g_live.maxDraws) ? drawCount : g_live.maxDraws;
        const uint32_t haveBytes = drawBytes / (uint32_t)sizeof(IPC::DrawItemWire);
        const uint32_t count = (n < haveBytes) ? n : haveBytes;
        const IPC::DrawItemWire* items = (const IPC::DrawItemWire*)drawBlob;

        // Upload camera + per-draw world matrices into the persistent-mapped CBVs.
        std::memcpy(g_live.pFrameCbv->pCpuMappedAddress, viewProj, 16 * sizeof(float));
        uint8_t* objBase = (uint8_t*)g_live.pObjectCbv->pCpuMappedAddress;
        for (uint32_t i = 0; i < count; ++i) {
            std::memcpy(objBase + (size_t)i * kObjStride, items[i].world, 16 * sizeof(float));
        }

        resetCmdPool(R, g_live.pCmdPool);
        beginCmd(g_live.pCmd);

        // Shared RT: COMMON (steady state) -> RENDER_TARGET. First frame it was created
        // RENDER_TARGET; depth created DEPTH_WRITE and stays there.
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
        bind.mDepthStencil = { g_live.pDepth, LOAD_ACTION_CLEAR };
        cmdBindRenderTargets(g_live.pCmd, &bind);
        cmdSetViewport(g_live.pCmd, 0.0f, 0.0f, (float)g_live.width, (float)g_live.height, 0.0f, 1.0f);
        cmdSetScissor(g_live.pCmd, 0, 0, g_live.width, g_live.height);
        cmdBindPipeline(g_live.pCmd, g_live.pOpaquePipeline);
        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);

        uint32_t drawn = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t slot = items[i].slot;
            if (slot >= g_meshHigh || !g_meshes[slot].valid) {
                continue;   // mesh not uploaded yet (or evicted)
            }
            HostMesh& m = g_meshes[slot];
            cmdBindDescriptorSet(g_live.pCmd, i, g_live.pPerDrawSet);
            uint32_t stride = (uint32_t)sizeof(IPC::GeomVertexWire);
            cmdBindVertexBuffer(g_live.pCmd, 1, &m.vb, &stride, nullptr);
            cmdBindIndexBuffer(g_live.pCmd, m.ib, INDEX_TYPE_UINT16, 0);
            cmdDrawIndexed(g_live.pCmd, m.indexCount, 0, 0);
            ++drawn;
        }

        cmdBindRenderTargets(g_live.pCmd, nullptr);

        // Hand the shared RT back to COMMON for MW's D3D9Ex StretchRect.
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
        g_lastDrawn = drawn;
        return true;
    }

    unsigned lastDrawn() { return g_lastDrawn; }

    unsigned uploadGeometry(const void* blobBytes, unsigned byteCount, unsigned partCount) {
        if (!g_live.pRenderer || !blobBytes || !byteCount || !partCount) {
            return 0;
        }
        const uint8_t* p   = (const uint8_t*)blobBytes;
        const uint8_t* end = p + byteCount;
        unsigned built = 0;

        for (unsigned i = 0; i < partCount; ++i) {
            if (p + sizeof(IPC::GeomPartWire) > end) {
                break;
            }
            IPC::GeomPartWire hdr;
            std::memcpy(&hdr, p, sizeof(hdr));
            p += sizeof(hdr);

            const uint64_t vbBytes = (uint64_t)hdr.vertexCount * sizeof(IPC::GeomVertexWire);
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
            if (m.valid) {  // re-upload on revision change: drop the old buffers
                if (m.vb) { removeResource(m.vb); }
                if (m.ib) { removeResource(m.ib); }
                m.vb = m.ib = nullptr;
                m.valid = false;
            }

            BufferLoadDesc vbDesc = {};
            vbDesc.mDesc.mDescriptors  = DESCRIPTOR_TYPE_VERTEX_BUFFER;
            vbDesc.mDesc.mMemoryUsage  = RESOURCE_MEMORY_USAGE_GPU_ONLY;
            vbDesc.mDesc.mSize         = vbBytes;
            vbDesc.mDesc.mStartState   = RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
            vbDesc.pData               = verts;
            vbDesc.ppBuffer            = &m.vb;
            addResource(&vbDesc, nullptr);

            BufferLoadDesc ibDesc = {};
            ibDesc.mDesc.mDescriptors  = DESCRIPTOR_TYPE_INDEX_BUFFER;
            ibDesc.mDesc.mMemoryUsage  = RESOURCE_MEMORY_USAGE_GPU_ONLY;
            ibDesc.mDesc.mSize         = ibBytes;
            ibDesc.mDesc.mStartState   = RESOURCE_STATE_INDEX_BUFFER;
            ibDesc.pData               = indices;
            ibDesc.ppBuffer            = &m.ib;
            addResource(&ibDesc, nullptr);

            m.vertexCount = hdr.vertexCount;
            m.indexCount  = hdr.indexCount;
            m.valid       = true;
            if (hdr.slot + 1 > g_meshHigh) {
                g_meshHigh = hdr.slot + 1;
            }
            ++built;
        }

        waitForAllResourceLoads();
        LOGF(eINFO, "[forge] uploadGeometry: built %u/%u parts (%u bytes), meshHigh=%u",
             built, partCount, byteCount, g_meshHigh);
        std::printf("[forge] uploadGeometry: built %u/%u parts (%u bytes), meshHigh=%u\n",
                    built, partCount, byteCount, g_meshHigh);
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
        if (g_live.pPerDrawSet)     { removeDescriptorSet(R, g_live.pPerDrawSet); }
        if (g_live.pFrameCbv)       { removeResource(g_live.pFrameCbv); }
        if (g_live.pObjectCbv)      { removeResource(g_live.pObjectCbv); }
        if (g_live.pOpaquePipeline) { removePipeline(R, g_live.pOpaquePipeline); }
        if (g_live.pOpaqueShader)   { removeShader(R, g_live.pOpaqueShader); }
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
