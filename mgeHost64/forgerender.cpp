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
#include "mge/configuration.h"   // Configuration.DL.* for the host-owned live distant-land cull
#include "support/log.h"   // LOG::logline -> mgeHost64.log (LOGF goes to uncaptured stdout)

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
#include <algorithm>

#include "OS/Interfaces/IOperatingSystem.h"
#include "Utilities/Interfaces/IFileSystem.h"
#include "Utilities/Log/Log.h"
#include "Graphics/GraphicsConfig.h"
#include "Graphics/Interfaces/IGraphics.h"
#include "Resources/ResourceLoader/Interfaces/IResourceLoader.h"
#include "Utilities/Interfaces/ILog.h"
// Dev overlay (Stage 1): Forge's own IUI/IFont rendered into pRT each frame. The headless host
// has no window/InputSystem, so input is bridged from the MW client over IPC and injected via the
// vendored UI.cpp shim (uiSetExternalInput) — see [[project_forge_dev_overlay]].
#include "Application/Interfaces/IUI.h"
#include "Application/Interfaces/IFont.h"
#include "Utilities/ThirdParty/OpenSource/bstrlib/bstrlib.h"   // Phase 0 panel: DynamicTextWidget live stats
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
// Tier 2 depth-takeover compute SRTs (linearize/resolve + GTAO). Each declares its own
// SRT_<name> + descriptor indices; the AOParams cbuffer struct comes from gtao.srt.h. They share
// the merged ComputeRootSignature (compute.rootsig). See [[project_forge_depth_prepass]].
#include "shaders/FSL/linearizedepth.srt.h"
#include "shaders/FSL/gtao.srt.h"
#include "shaders/FSL/aoblur.srt.h"
// Stage B (M1) GPU statics cull SRT (CullSrtData: gCullParams + gCullInst + gCullCount). Also shares
// the merged ComputeRootSignature. Names its element CullInstance (not GpuCullInstance) to avoid
// redefining the host C++ struct when STRUCT(T) expands to `struct T` in this TU.
#include "shaders/FSL/cull.srt.h"

// App-layer callback normally provided by WindowsBase.cpp (which we exclude — it
// drags in the window system). The backend calls this on device-lost. Headless:
// no swapchain to rebuild, so record nothing. The header's extern "C" block
// gives this C linkage to match the (compiled-as-C) Direct3D12.c reference.
void requestReset(const ResetDesc* pResetDesc) { (void)pResetDesc; }

// Dev overlay (Stage 1/2). The platform*UserInterface / platform*FontSystem functions live in The
// Forge's UI.cpp / FontSystem.cpp with no public header — normally the app framework
// (WindowsBase::initBaseSubsystems) calls platformInit*/platformUpdate*/platformExit* around the
// app's init/exit. The headless host excludes that framework, so it MUST drive them itself.
// platformInitFontSystem in particular sizes the font atlas (from DPI) and creates the FONS
// context; skipping it leaves the atlas 0x0 and initFontSystem crashes. uiSetExternalInput is the
// vendored UI.cpp input-bridge setter (FORGE_HOST_EXTERNAL_INPUT) the host pushes mouse state into.
extern bool platformInitFontSystem();
extern bool platformInitUserInterface();
extern void platformUpdateUserInterface(float deltaTime);
extern void platformExitFontSystem();
extern void platformExitUserInterface();
extern "C" void uiSetExternalInput(float x, float y, float wheel, bool l, bool r, bool m, bool enabled);

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

    // Row-major 4x4 multiply C = A·B (out[i*4+j] = sum_k A[i*4+k]·B[k*4+j]). Same row-major /
    // row-vector convention as the rest of the host (a point transforms p' = p·M). WT2 builds the
    // reflection matrix as Mirror·viewProj (mirror world geometry about the water plane, then the
    // normal camera). out may alias neither A nor B.
    void mul4x4(const float a[16], const float b[16], float out[16]) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                out[i*4+j] = a[i*4+0]*b[0*4+j] + a[i*4+1]*b[1*4+j]
                           + a[i*4+2]*b[2*4+j] + a[i*4+3]*b[3*4+j];
            }
        }
    }

    // Full 4x4 inverse (cofactor / adjugate). Tier 2 GTAO reconstructs world position from device
    // depth via the inverse of the reverse-Z world->clip matrix; the host has no D3DX, so invert
    // here. Layout-agnostic (operates on the raw 16 floats); the caller feeds rzViewProj bytes and
    // uploads the result as-is — the shader's mul(M,v) reproduces the row-vector transform either
    // way. Returns false (and leaves out untouched) for a singular matrix.
    bool invert4x4(const float m[16], float out[16]) {
        float inv[16];
        inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
        inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
        inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
        inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
        inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
        inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
        inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
        inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
        inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
        inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
        inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
        inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
        inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
        inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
        inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
        inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];
        float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
        if (det > -1e-12f && det < 1e-12f) { return false; }
        det = 1.0f / det;
        for (int i = 0; i < 16; ++i) { out[i] = inv[i] * det; }
        return true;
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
        // Dev overlay font lives in morrowind64/fonts/ (deployed beside the exe).
        fsSetPathForResourceDir(pSystemFileIO, RD_FONTS, "fonts");

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
        std::printf("[forge] initRootSignature (default.rootsig + compute.rootsig)...\n");
        RootSignatureDesc rsDesc = {};
        rsDesc.pGraphicsFileName = "default.rootsig";
        // Tier 2: the host's first COMPUTE root signature (linearize + GTAO). One global compute
        // rootsig serves every PIPELINE_TYPE_COMPUTE shader (fsl.py merged both compute SRTs into it).
        rsDesc.pComputeFileName = "compute.rootsig";
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

namespace {
    // --- PIX programmatic GPU capture (the headless probe never Presents, so PIX's hotkey
    // capture can't trigger). enablePixCapture() loads WinPixGpuCapturer.dll BEFORE the device
    // is created (it hooks d3d12 on load); sceneProbe then wraps one renderScene in begin/end. ---
    // Modern WinPixGpuCapturer exports (dumpbin-verified): BeginProgrammaticGpuCapture(params),
    // EndProgrammaticGpuCapture(). params = &{ PWSTR fileName } (PIXCaptureParameters::GpuCaptureParameters).
    typedef long (__stdcall *PFN_PIXBeginGpu)(const void*);
    typedef long (__stdcall *PFN_PIXEndGpu)(void);
    static PFN_PIXBeginGpu g_pixBegin = nullptr;
    static PFN_PIXEndGpu   g_pixEnd   = nullptr;

    static HMODULE findWinPixGpuCapturer() {
        HMODULE m = LoadLibraryW(L"WinPixGpuCapturer.dll");   // next to exe / on PATH first
        if (m) { return m; }
        // Scan the PIX install dir for the newest version: C:\Program Files\Microsoft PIX\<ver>\.
        WIN32_FIND_DATAW fd = {};
        wchar_t best[MAX_PATH] = {};
        HANDLE h = FindFirstFileW(L"C:\\Program Files\\Microsoft PIX\\*", &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && fd.cFileName[0] != L'.') {
                    // Lexicographically-latest version dir wins (PIX names are date/semver-ish).
                    if (wcscmp(fd.cFileName, best) > 0) { wcscpy_s(best, fd.cFileName); }
                }
            } while (FindNextFileW(h, &fd));
            FindClose(h);
        }
        if (best[0]) {
            wchar_t path[MAX_PATH];
            swprintf_s(path, L"C:\\Program Files\\Microsoft PIX\\%s\\WinPixGpuCapturer.dll", best);
            m = LoadLibraryW(path);
        }
        return m;
    }

    // --- RenderDoc in-application capture (alternative to PIX; the user found PIX's descriptor
    // tables un-inspectable). Same shape: load renderdoc.dll BEFORE device creation so its hooks
    // install, fetch the API table via RENDERDOC_GetAPI, then StartFrameCapture/EndFrameCapture
    // wrap one renderScene. RenderDoc's Pipeline State → CS → UAVs resolves root-param → heap →
    // resource (the view PIX denied us) so we can finally see whether gtao's space-3 u0 points at
    // pAO. We declare the API table prefix by hand (renderdoc_app.h not vendored): the layout is
    // append-only and STABLE up to EndFrameCapture across every version ≥ 1.1.2, so requesting
    // eRENDERDOC_API_Version_1_1_2 (10102) and using only the prefix fields is safe. CC = __cdecl. --
    typedef void* RDOC_DevicePtr;
    typedef void* RDOC_WindowHandle;
    typedef int      (__cdecl *PFN_RDOC_GetAPI)(uint32_t version, void** outAPIPointers);
    typedef void     (__cdecl *PFN_RDOC_SetCaptureFilePathTemplate)(const char* pathtemplate);
    typedef void     (__cdecl *PFN_RDOC_StartFrameCapture)(RDOC_DevicePtr, RDOC_WindowHandle);
    typedef uint32_t (__cdecl *PFN_RDOC_EndFrameCapture)(RDOC_DevicePtr, RDOC_WindowHandle);

    // Function-pointer table in renderdoc_app.h order; void* for fields we don't call so the
    // offsets to SetCaptureFilePathTemplate / StartFrameCapture / EndFrameCapture stay exact.
    struct RDOC_API_Table {
        void* GetAPIVersion;
        void* SetCaptureOptionU32;
        void* SetCaptureOptionF32;
        void* GetCaptureOptionU32;
        void* GetCaptureOptionF32;
        void* SetFocusToggleKeys;
        void* SetCaptureKeys;
        void* GetOverlayBits;
        void* MaskOverlayBits;
        void* RemoveHooks;
        void* UnloadCrashHandler;
        PFN_RDOC_SetCaptureFilePathTemplate SetCaptureFilePathTemplate;
        void* GetCaptureFilePathTemplate;
        void* GetNumCaptures;
        void* GetCapture;
        void* TriggerCapture;
        void* IsTargetControlConnected;
        void* LaunchReplayUI;
        void* SetActiveWindow;
        PFN_RDOC_StartFrameCapture StartFrameCapture;
        void* IsFrameCapturing;
        PFN_RDOC_EndFrameCapture EndFrameCapture;
    };
    static RDOC_API_Table* g_rdoc = nullptr;

    static HMODULE findRenderDoc() {
        HMODULE m = GetModuleHandleW(L"renderdoc.dll");        // already injected by the RenderDoc UI?
        if (m) { return m; }
        m = LoadLibraryW(L"renderdoc.dll");                    // next to exe / on PATH
        if (m) { return m; }
        return LoadLibraryW(L"C:\\Program Files\\RenderDoc\\renderdoc.dll");
    }
}

namespace ForgeRender {
    // Public: call ONCE before sceneProbe()/init() so the capturer hooks d3d12 device creation.
    bool enablePixCapture() {
        HMODULE m = findWinPixGpuCapturer();
        if (!m) { std::printf("[forge] PIX: WinPixGpuCapturer.dll NOT found (install PIX or copy the dll next to the exe)\n"); return false; }
        g_pixBegin = (PFN_PIXBeginGpu)GetProcAddress(m, "BeginProgrammaticGpuCapture");
        g_pixEnd   = (PFN_PIXEndGpu)GetProcAddress(m, "EndProgrammaticGpuCapture");
        std::printf("[forge] PIX capturer loaded (begin=%p end=%p)\n", (void*)g_pixBegin, (void*)g_pixEnd);
        return g_pixBegin && g_pixEnd;
    }

    // Public: call ONCE before sceneProbe()/init() so renderdoc.dll hooks d3d12 device creation.
    bool enableRdocCapture() {
        HMODULE m = findRenderDoc();
        if (!m) { std::printf("[forge] RDOC: renderdoc.dll NOT found (launch via RenderDoc UI or install it)\n"); return false; }
        PFN_RDOC_GetAPI getApi = (PFN_RDOC_GetAPI)GetProcAddress(m, "RENDERDOC_GetAPI");
        if (!getApi) { std::printf("[forge] RDOC: RENDERDOC_GetAPI export missing\n"); return false; }
        const uint32_t eRENDERDOC_API_Version_1_1_2 = 10102;
        int ok = getApi(eRENDERDOC_API_Version_1_1_2, (void**)&g_rdoc);
        std::printf("[forge] RDOC: GetAPI -> %d (table=%p)\n", ok, (void*)g_rdoc);
        if (!ok || !g_rdoc) { g_rdoc = nullptr; return false; }
        // Frame number gets appended; the .rdc lands next to the game so the user can open it.
        g_rdoc->SetCaptureFilePathTemplate("C:\\mgem\\morrowind64\\forge_gtao");
        std::printf("[forge] RDOC capture armed (start=%p end=%p)\n",
                    (void*)g_rdoc->StartFrameCapture, (void*)g_rdoc->EndFrameCapture);
        return true;
    }

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
        float lt[28] = {
            0,0,-1,0,           // sunDir (unused, sunCol=0)
            0,0,0,0,            // sunCol = 0
            0.5f,0.5f,0.5f,0,   // ambCol
            0,0,0,0,            // fogColNear
            0, 1e9f, 0,0,       // fogParams: start=0, end=1e9 -> fog ≈ 1 (clear)
            0,0,0,0,            // eyePos
            0,0,0,0             // realEye.xyz + isExterior (0 -> no host-owned DL in the scene-probe)
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
            // PIX: capture THIS renderScene (linearize + GTAO dispatches + colour pass) if armed.
            if (g_pixBegin) {
                struct GpuCapParams { const wchar_t* fileName; } params = { L"C:\\mgem\\morrowind64\\forge_gtao.wpix" };
                long hr = g_pixBegin(&params);
                std::printf("[forge] PIX: BeginProgrammaticGpuCapture -> hr=0x%08lX\n", hr);
            }
            if (g_rdoc) {
                g_rdoc->StartFrameCapture(nullptr, nullptr);   // NULL device = capture all the host's queues
                std::printf("[forge] RDOC: StartFrameCapture\n");
            }
            ok = renderScene(vp, lt, nullptr, 0, 0,
                             nullptr, 0, 0,
                             &mm, 1, (unsigned)sizeof(mm),
                             nullptr, 0, 0);
            if (g_rdoc) {
                uint32_t cap = g_rdoc->EndFrameCapture(nullptr, nullptr);
                std::printf("[forge] RDOC: EndFrameCapture -> %u (capture: C:\\mgem\\morrowind64\\forge_gtao_frameNNN.rdc)\n", cap);
            }
            if (g_pixEnd) {
                long hr = g_pixEnd();
                std::printf("[forge] PIX: EndProgrammaticGpuCapture -> hr=0x%08lX (capture: C:\\mgem\\morrowind64\\forge_gtao.wpix)\n", hr);
            }
            std::printf("[forge] scene-probe: multi-map renderScene returned %d (multiMapDrawn=%u) -> expect CENTRE ~248\n",
                        (int)ok, lastMultiMapDrawn());
            debugReadbackCenterPixel();
        }

        debugReadbackAO();   // Tier 2 diag: ground-truth pAO contents (GTAO write vs graphics read)

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
        QueryPool*      pGpuQueryPool = nullptr;   // GPU timestamp pool (per-phase 4ms breakdown)
        double          gpuTickFreq = 0.0;         // timestamp ticks/sec (getTimestampFrequency)
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

        // --- Tier 2 depth-takeover: depth-as-SRV + GTAO compute (AO buffer in isolation) ---
        // First compute pipelines / UAVs / depth->SRV barrier in the host. Two passes:
        //   (1) linearize: resolve sample 0 of pDepth (MSAA-robust) into single-sample pLinearDepth.
        //   (2) gtao:      horizon-search AO from pLinearDepth -> pAO (rgb bent normal, a visibility).
        // Sequenced between the Z-prepass and the colour pass in renderScene. Tier 2 feeds ONLY the
        // F12 debug views (modes 3/4); Tier 3 will read pAO in the colour frags.
        Texture*       pLinearDepth = nullptr;    // single-sample R32F resolved DEVICE depth (SRV+UAV)
        Texture*       pAO = nullptr;             // RGBA16F AO (rgb = bent normal, a = visibility) (SRV+UAV)
        Texture*       pAOBlur = nullptr;         // RGBA16F bilateral-blurred AO (the frags' gAO) (SRV+UAV)
        Shader*        pLinearizeShader = nullptr;
        Pipeline*      pLinearizePipeline = nullptr;
        Shader*        pGtaoShader = nullptr;
        Pipeline*      pGtaoPipeline = nullptr;
        Shader*        pAOBlurShader = nullptr;
        Pipeline*      pAOBlurPipeline = nullptr;
        Buffer*        pAOParamsCbv = nullptr;    // gAOParams (invViewProj/screen/knobs/eye), persistent-mapped
        DescriptorSet* pLinearizeSet = nullptr;   // LinDepthSrtData PerBatch: gSceneDepth + gLinearDepthOut
        DescriptorSet* pGtaoBatchSet = nullptr;   // AOSrtData PerBatch:  gLinearDepthIn + gAOOut
        DescriptorSet* pAOBlurSet = nullptr;      // AOBlurSrtData PerFrame: gBlurParams + gAOSrc + gBlurDepthIn + gAODst
        // --- Stage B (M1) GPU statics cull (B2 = COUNT-only validation) ------------------------
        // cull.comp tests every resident GpuCullInstance vs the per-frame CullParams (planes/eye/
        // ranges) and atomic-adds numSubsets into pCullCountBuf[0]. The host resets it from a zero
        // upload buffer and reads it back (raw D3D12 CopyBufferRegion — Forge exposes no buffer->
        // buffer copy) to compare against the CPU cull's g_liveLastInst. No draws (B3 adds scatter).
        Buffer*        pCullInstBuf = nullptr;     // GPU_ONLY structured SRV: g_cullInst (96B/inst), uploaded once
        Buffer*        pCullCountBuf = nullptr;    // GPU_ONLY structured RW (uint[4]): [0] = survivor count
        Buffer*        pCullCountReadback = nullptr; // GPU_TO_CPU, persistent-mapped (post-fence read)
        Buffer*        pCullCountZero = nullptr;   // CPU_TO_GPU zeros (per-frame reset source)
        Buffer*        pCullParamsCbv = nullptr;   // gCullParams (planes/eye/ranges/count), persistent-mapped
        Shader*        pCullShader = nullptr;
        Pipeline*      pCullPipeline = nullptr;
        DescriptorSet* pCullSet = nullptr;         // CullSrtData PerBatch: gCullParams + gCullInst + gCullCount
        uint32_t       cullInstCount = 0;          // g_ws0Count at upload (dispatch + bounds guard)
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
        DescriptorSet* pPersistentSet = nullptr;   // bindless gTextures[] + gStaticsArrays (bound once per frame)
        Texture*       pStaticsWhiteArray = nullptr; // 4x4x2 white Tex2DArray; fills every gStaticsArrays slot until the
                                                     // distant-statics buckets build (and stays in the unused tail).

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

        // --- SK1: sky pass (alpha-blended, depth off) --------------------------------
        // The host's FIRST blend pipeline. Reuses the SAME GeomVertexWire layout (vl) + SrtData/
        // default.rootsig + bindless gTextures as the static opaque path; sky.vert reads gBatch.worlds
        // (one window, like multimap) and sky.frag outputs real RGBA (no forced alpha=1) so the
        // present-seam premultiplied composite lays the dome over MW. Drawn FIRST in the colour pass.
        Shader*        pSkyShader = nullptr;
        Pipeline*      pSkyPipeline = nullptr;             // SRCALPHA/INVSRCALPHA, depth test+write OFF
        Pipeline*      pSkyPipelineAdd = nullptr;          // SK2: SRCALPHA/ONE additive variant (glare/cloud groundwork)
        Buffer*        pSkyWorldsBuf = nullptr;            // gBatch: one 64KB world window, persistent-mapped
        DescriptorSet* pPerBatchSetSky = nullptr;          // gBatch bound to pSkyWorldsBuf, 1 instance
        Buffer*        pSkyInstanceBuf = nullptr;          // per-draw instance VB (kStaticInstU32 slots), CPU-mapped

        // --- WT1: Forge water takeover (host-generated geo-clipmap surface) -----------
        // Reuses the SAME SrtData/default.rootsig: the per-LOD-level worlds + packed params + invVP
        // ride gBatch.worlds (pWaterWorldsBuf, one window) exactly like the sky path; the 4 water SRVs
        // (gWaterNormalVol/gRefractColor/gSceneLinDepth/gReflectColor) append to the PerFrame set and
        // are bound once into pPerFrameSet. Drawn AFTER the distant land (depth-write reverse-Z GEQUAL,
        // cull NONE) into the same colour+depth, then the present-seam composites the whole frame.
        Shader*        pWaterShader = nullptr;
        Pipeline*      pWaterPipeline = nullptr;           // depth GEQUAL + write, cull NONE, no blend
        Buffer*        pWaterWorldsBuf = nullptr;          // gBatch: worlds[0..5]=LOD levels, [6]=params, [7]=invVP
        DescriptorSet* pPerBatchSetWater = nullptr;        // gBatch bound to pWaterWorldsBuf, 1 instance
        Buffer*        pWaterInstanceBuf = nullptr;        // per-draw instance VB: DrawIndex=level (kMaxWaterLevels)
        Buffer*        pWaterVB = nullptr;                 // GPU_ONLY clipmap verts (float3, stride 12)
        Buffer*        pWaterIB = nullptr;                 // GPU_ONLY clipmap indices (uint16)
        Texture*       pRefractColor = nullptr;            // screen copy of the pre-water colour (refraction src)
        Texture*       pWaterNormalVol = nullptr;          // water_NRM.dds 3D animated-normal volume
        bool           waterReady = false;                 // all water resources built (gates the pass)

        // --- WT2: real Forge reflection RT (replaces WT1's flat stand-in) ----------------
        // A fixed 1024² RT (matches MGE's texReflection budget; sampled with normalized UV so the
        // resolution is decoupled from the screen). Rendered by mirroring world geometry about the
        // water plane (mirrorVP = Mirror·viewProj) and drawing into it with the NORMAL camera, so it
        // lands in main-screen space → the water frag samples it at its own screen UV. Stage 1 draws
        // SKY ONLY (own decoupled buffers; validates the mirror matrix without the DL cull machinery).
        RenderTarget*  pReflectColor = nullptr;            // 1024² B8G8R8A8, alpha = coverage
        RenderTarget*  pReflectDepth = nullptr;            // 1024² D32 reverse-Z (mirror pass depth)
        Buffer*        pReflectFrameCbv = nullptr;         // gFrameData for the mirror view (own viewProj)
        DescriptorSet* pPerFrameSetReflect = nullptr;      // PerFrame set bound to pReflectFrameCbv (+ gAO + water SRVs)
        Buffer*        pReflectSkyWorldsBuf = nullptr;     // reflect sky gBatch window (own; filled in the reflect pass)
        DescriptorSet* pPerBatchSetReflectSky = nullptr;   // gBatch bound to pReflectSkyWorldsBuf
        Buffer*        pReflectSkyInstanceBuf = nullptr;   // reflect sky per-draw instance VB
        bool           reflectReady = false;               // all reflection resources built (gates the pass)

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

    // GPU timestamp phases (the ~4ms gpu breakdown). Index = QueryDesc index in renderScene.
    enum { kGpuPhasePrepass = 0, kGpuPhaseGtao, kGpuPhaseReflect, kGpuPhaseColor,
           kGpuPhaseWater, kGpuPhaseResolve, kGpuPhaseCount };

    // Tier 4 multi-map: ONE 64KB world window holds kBatchSize(1024) matrices; cap at 256
    // parts/frame (< 1024 → a single window, no batching). Per-instance VB stride (uint32 slots):
    //   [0] Meta (DrawIndex|stageCount|vColSource|alphaRef)   [1..4] stages[4]
    //   [5..7] matDiffuse.rgb   [8..10] matAmbient.rgb   [11..13] matEmissive.rgb
    // 14 * 4 = 56 bytes. Rare geometry → a flat 256-cap (log-drop, no fallback) is ample.
    constexpr uint32_t kMaxMultiMap = 256;
    constexpr uint32_t kMMInstU32   = 14;

    // SK1 sky: per-frame sky draw cap (must match IPC::kMaxSkyDraws). SK1 draws only the dome;
    // the full sky subtree is ~15 shapes (SK2). One 64KB world window (< 1024 matrices) holds them.
    constexpr uint32_t kMaxSkyDraws = 64;

    // WT1 Forge water: the geo-clipmap (port of MGE initWaterLodMesh, distantinit.cpp:1047) — 6 LOD
    // levels, finest cell 128u, 64 cells/side, T-junction stitch + 4 trim variants/level. The host
    // generates verts/indices once and draws one cmdDrawIndexedInstanced per level (DrawIndex=level
    // selects gBatch.worlds[level]). worlds[0..5] = the 6 levels, [6] = packed params, [7] = invVP.
    constexpr uint32_t kMaxWaterLevels = 6;
    constexpr uint32_t kWaterLevels    = 6;
    // WT2 reflection RT side (matches MGE texReflection's 1024² budget; sampled with normalized UV).
    constexpr uint32_t kReflectSize = 1024;
    constexpr float    kWaterCell0     = 128.0f;
    constexpr int      kWaterGrid      = 64;     // cells per side per level (even)
    // One per-level draw record: which IB sub-range each of the 4 trim variants occupies.
    struct WaterLodLevelHost {
        float    cellSize;
        uint32_t vertBase;        // first vertex of this level in pWaterVB
        uint32_t vertCount;       // verts in this level
        uint32_t ibStart[4];      // index of first index for variant v
        uint32_t triCount[4];     // triangles for variant v
        uint32_t numVariants;     // 1 (level 0) or 4
    };
    WaterLodLevelHost g_waterLevels[kWaterLevels] = {};
    uint32_t          g_waterVertTotal = 0;

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

    // SK2: D3DBLEND_* (the SkyDrawWire blend factors, captured from NiAlphaProperty) — the host
    // keeps D3D headers out, so these are the literal d3d9.h enum values. Used to classify a
    // captured (src,dst) pair against the prebuilt sky PSO set.
    enum {
        kD3DBLEND_ZERO        = 1,
        kD3DBLEND_ONE         = 2,
        kD3DBLEND_SRCALPHA    = 5,
        kD3DBLEND_INVSRCALPHA = 6,
    };

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

    // Little-endian 32-bit read (DDS header fields). Defined here (ahead of the water loader AND
    // parseDds, both of which use it).
    static uint32_t ddsRd32(const uint8_t* p) {
        return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    }

    // WT1: build the geo-clipmap water VB/IB (CPU port of MGE DistantLand::initWaterLodMesh,
    // distantinit.cpp:1047). Local integer lattice [-half,half] on z=0 (height added in the VS later);
    // 6 LOD levels, each with a stitched outer annulus (T-junction fix to the coarser ring) and (for
    // levels 1..5) 4 trim variants whose central hole shifts by eye parity to nest the finer level.
    // Fills g_waterLevels + the two GPU_ONLY buffers via BufferLoadDesc pData. Idempotent.
    bool buildWaterMesh(Renderer* R) {
        if (g_live.pWaterVB && g_live.pWaterIB) { return true; }
        const int   m       = kWaterGrid;
        const int   verts1D = m + 1;
        const int   half    = m / 2;
        const float c0      = kWaterCell0;
        const int   L       = (int)kWaterLevels;

        g_waterVertTotal = (uint32_t)(L * verts1D * verts1D);
        std::vector<float> verts;                  // float3 per vertex
        verts.reserve((size_t)g_waterVertTotal * 3);
        for (int k = 0; k < L; ++k) {
            for (int gy = 0; gy <= m; ++gy) {
                for (int gx = 0; gx <= m; ++gx) {
                    verts.push_back((float)(gx - half));
                    verts.push_back((float)(gy - half));
                    verts.push_back(0.0f);         // z = 0 (height in the VS, follow-up)
                }
            }
        }

        std::vector<uint16_t> indices;
        indices.reserve(400000);
        int vertBase = 0;
        auto vidx = [&](int gx, int gy) -> uint16_t { return (uint16_t)(vertBase + gy * verts1D + gx); };
        // Emit one triangle forcing CCW winding on the z=0 plane (signed area decides orientation).
        auto addTri = [&](int ax, int ay, int bx, int by, int cx, int cy) {
            long cross = (long)(bx - ax) * (cy - ay) - (long)(by - ay) * (cx - ax);
            uint16_t ia = vidx(ax, ay), ib = vidx(bx, by), ic = vidx(cx, cy);
            if (cross < 0) { uint16_t t = ib; ib = ic; ic = t; }
            indices.push_back(ia); indices.push_back(ib); indices.push_back(ic);
        };
        auto addCell = [&](int cx, int cy) {
            addTri(cx, cy, cx + 1, cy, cx + 1, cy + 1);
            addTri(cx, cy, cx + 1, cy + 1, cx, cy + 1);
        };
        auto addEdgeBlock = [&](int ax, int ay, int tx, int ty, int nx, int ny) {
            int Ax = ax,                Ay = ay;
            int Cx = ax + 2 * tx,       Cy = ay + 2 * ty;
            int Apx = ax + nx,          Apy = ay + ny;
            int Bpx = ax + tx + nx,     Bpy = ay + ty + ny;
            int Cpx = ax + 2 * tx + nx, Cpy = ay + 2 * ty + ny;
            addTri(Ax, Ay, Cx, Cy, Bpx, Bpy);
            addTri(Ax, Ay, Bpx, Bpy, Apx, Apy);
            addTri(Cx, Cy, Cpx, Cpy, Bpx, Bpy);
        };
        auto addCorner = [&](int cgx, int cgy, int dx, int dy) {
            int Ox = cgx,          Oy = cgy;
            int Bx = cgx + 2 * dx, By = cgy;
            int Tx = cgx,          Ty = cgy + 2 * dy;
            int Mx = cgx + dx,     My = cgy + dy;
            int Rx = cgx + 2 * dx, Ry = cgy + dy;
            int Sx = cgx + 2 * dx, Sy = cgy + 2 * dy;
            int Ux = cgx + dx,     Uy = cgy + 2 * dy;
            addTri(Ox, Oy, Bx, By, Mx, My);
            addTri(Ox, Oy, Mx, My, Tx, Ty);
            addTri(Bx, By, Rx, Ry, Mx, My);
            addTri(Rx, Ry, Sx, Sy, Mx, My);
            addTri(Mx, My, Sx, Sy, Ux, Uy);
            addTri(Mx, My, Ux, Uy, Tx, Ty);
        };
        auto addOuterStitch = [&]() {
            addCorner(0, 0, +1, +1);
            addCorner(m, 0, -1, +1);
            addCorner(0, m, +1, -1);
            addCorner(m, m, -1, -1);
            for (int c = 2; c <= m - 4; c += 2) {
                addEdgeBlock(c, 0, 1, 0,  0,  1);
                addEdgeBlock(c, m, 1, 0,  0, -1);
                addEdgeBlock(0, c, 0, 1,  1,  0);
                addEdgeBlock(m, c, 0, 1, -1,  0);
            }
        };

        for (int k = 0; k < L; ++k) {
            const bool stitch = (k < L - 1);
            const int  numVar = (k == 0) ? 1 : 4;
            WaterLodLevelHost& lvl = g_waterLevels[k];
            lvl.cellSize    = c0 * (float)(1 << k);
            lvl.vertBase    = (uint32_t)vertBase;
            lvl.vertCount   = (uint32_t)(verts1D * verts1D);
            lvl.numVariants = (uint32_t)numVar;
            for (int i = 0; i < 4; ++i) { lvl.ibStart[i] = 0; lvl.triCount[i] = 0; }
            for (int vrt = 0; vrt < numVar; ++vrt) {
                const int ex = vrt & 1;
                const int ey = (vrt >> 1) & 1;
                const int holeLoX = (m / 4) + ex, holeHiX = (3 * m / 4) + ex;
                const int holeLoY = (m / 4) + ey, holeHiY = (3 * m / 4) + ey;
                const size_t startIdx = indices.size();
                lvl.ibStart[vrt] = (uint32_t)startIdx;
                for (int cy = 0; cy < m; ++cy) {
                    for (int cx = 0; cx < m; ++cx) {
                        if (stitch) {
                            const bool corner = (cx <= 1 || cx >= m - 2) && (cy <= 1 || cy >= m - 2);
                            const bool bedge  = (cy == 0 || cy == m - 1) && (cx >= 2 && cx <= m - 3);
                            const bool vedge  = (cx == 0 || cx == m - 1) && (cy >= 2 && cy <= m - 3);
                            if (corner || bedge || vedge) continue;
                        }
                        if (k > 0 && cx >= holeLoX && cx < holeHiX && cy >= holeLoY && cy < holeHiY) continue;
                        addCell(cx, cy);
                    }
                }
                if (stitch) { addOuterStitch(); }
                lvl.triCount[vrt] = (uint32_t)((indices.size() - startIdx) / 3);
            }
            vertBase += verts1D * verts1D;
        }

        BufferLoadDesc wv = {};
        wv.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
        wv.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        wv.mDesc.mSize        = (uint64_t)verts.size() * sizeof(float);
        wv.mDesc.mStartState  = RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        wv.mDesc.pName        = "waterVB";
        wv.pData              = verts.data();
        wv.ppBuffer           = &g_live.pWaterVB;
        addResource(&wv, nullptr);

        BufferLoadDesc wi = {};
        wi.mDesc.mDescriptors = DESCRIPTOR_TYPE_INDEX_BUFFER;
        wi.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        wi.mDesc.mSize        = (uint64_t)indices.size() * sizeof(uint16_t);
        wi.mDesc.mStartState  = RESOURCE_STATE_INDEX_BUFFER;
        wi.mDesc.pName        = "waterIB";
        wi.pData              = indices.data();
        wi.ppBuffer           = &g_live.pWaterIB;
        addResource(&wi, nullptr);

        waitForAllResourceLoads();
        if (!g_live.pWaterVB || !g_live.pWaterIB) { return false; }
        std::printf("[forge][water] clipmap mesh: %d levels, %u verts, %zu tris\n",
                    L, g_waterVertTotal, indices.size() / 3);
        return true;
    }

    // WT1: load water_NRM.dds as a 3D animated-normal volume. The host runs from the morrowind64 cwd,
    // so it reads the file directly. water_NRM is uncompressed 32-bit BGRA (DDS depth at offset 24).
    // The host's parseDds (uploadTextures) is 2D-only, so this is a dedicated 3D path: parse the
    // header inline, create a Texture3D, then slice/row-copy each mip (mDstSliceStride per Z slice).
    bool loadWaterNormalVolume(Renderer* R) {
        if (g_live.pWaterNormalVol) { return true; }
        const char* path = "Data Files\\textures\\MGE\\water_NRM.dds";
        FILE* f = std::fopen(path, "rb");
        if (!f) { std::printf("[forge][water] water_NRM.dds not found (%s)\n", path); return false; }
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (sz < 128) { std::fclose(f); return false; }
        std::vector<uint8_t> bytes((size_t)sz);
        size_t rd = std::fread(bytes.data(), 1, (size_t)sz, f);
        std::fclose(f);
        if (rd != (size_t)sz) { return false; }
        const uint8_t* d = bytes.data();
        if (ddsRd32(d) != 0x20534444u) { std::printf("[forge][water] bad DDS magic\n"); return false; }
        const uint32_t height = ddsRd32(d + 12);
        const uint32_t width  = ddsRd32(d + 16);
        uint32_t       depth  = ddsRd32(d + 24);
        uint32_t       mips   = ddsRd32(d + 28);
        const uint32_t pfFlags= ddsRd32(d + 80);
        const uint32_t bits   = ddsRd32(d + 88);
        if (mips == 0) { mips = 1; }
        if (depth == 0) { depth = 1; }
        // water_NRM is an uncompressed 32-bit volume (no FourCC). Treat as BGRA8 (MW A8R8G8B8).
        if ((pfFlags & 0x4) != 0 || bits != 32) {
            std::printf("[forge][water] water_NRM.dds not 32-bit uncompressed (pfFlags=%u bits=%u) — unsupported\n",
                        pfFlags, bits);
            return false;
        }
        const uint32_t dataOffset = 128;

        TextureDesc td = {};
        td.mWidth = width; td.mHeight = height; td.mDepth = depth;
        td.mArraySize = 1; td.mMipLevels = mips;
        td.mSampleCount = SAMPLE_COUNT_1;
        td.mFormat = TinyImageFormat_B8G8R8A8_UNORM;
        td.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        td.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
        td.pName = "waterNormalVol";
        TextureLoadDesc tld = {};
        tld.ppTexture = &g_live.pWaterNormalVol;
        tld.pDesc = &td;
        addResource(&tld, nullptr);
        waitForAllResourceLoads();
        if (!g_live.pWaterNormalVol) { std::printf("[forge][water] addResource(volume) FAILED\n"); return false; }

        const uint8_t* src    = d + dataOffset;
        const uint8_t* srcEnd = d + sz;
        TextureUpdateDesc upd = {};
        upd.pTexture = g_live.pWaterNormalVol;
        upd.mBaseMipLevel = 0; upd.mMipLevels = mips;
        upd.mBaseArrayLayer = 0; upd.mLayerCount = 1;
        upd.mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
        beginUpdateResource(&upd);
        for (uint32_t mip = 0; mip < mips; ++mip) {
            TextureSubresourceUpdate s = upd.getSubresourceUpdateDesc(mip, 0);
            const uint32_t sliceBytes = s.mRowCount * s.mSrcRowStride;
            // Slice count for this mip = max(1, depth >> mip).
            uint32_t mipDepth = depth >> mip; if (mipDepth == 0) mipDepth = 1;
            for (uint32_t z = 0; z < mipDepth; ++z) {
                if (src + sliceBytes > srcEnd) { mip = mips; break; }   // truncated → stop
                uint8_t* dstSlice = s.pMappedData + (size_t)z * s.mDstSliceStride;
                for (uint32_t row = 0; row < s.mRowCount; ++row) {
                    std::memcpy(dstSlice + (size_t)row * s.mDstRowStride,
                                src + (size_t)row * s.mSrcRowStride, s.mSrcRowStride);
                }
                src += sliceBytes;
            }
        }
        endUpdateResource(&upd);
        std::printf("[forge][water] volume %ux%ux%u, %u mips (BGRA8) loaded\n", width, height, depth, mips);
        return true;
    }

    // WT1/WT2: (re)build the water shader + graphics pipeline ONLY (not the resources/buffers/sets,
    // which persist across a shader edit). Called from buildOpaquePath at startup AND from the
    // hot-reload path so water.vert/.frag edits go live without a full relaunch. Idempotent: tears
    // down the existing shader/pipeline first. Position-only layout (pos float3 + per-instance
    // DrawIndex), depth GEQUAL+write (reverse-Z), cull NONE, no blend.
    bool buildWaterPipeline(Renderer* R) {
        waitQueueIdle(g_live.pQueue);
        if (g_live.pWaterPipeline) { removePipeline(R, g_live.pWaterPipeline); g_live.pWaterPipeline = nullptr; }
        if (g_live.pWaterShader)   { removeShader(R, g_live.pWaterShader);     g_live.pWaterShader = nullptr; }

        ShaderLoadDesc wsDesc = {};
        wsDesc.mVert.pFileName = "water.vert";
        wsDesc.mFrag.pFileName = "water.frag";
        addShader(R, &wsDesc, &g_live.pWaterShader);
        if (!g_live.pWaterShader) {
            LOG::logline("!! [forge][water] addShader(water) FAILED (dxil missing?)"); LOG::flush();
            return false;
        }

        VertexLayout wvl = {};
        wvl.mBindingCount = 2;
        wvl.mBindings[0].mStride = 12;
        wvl.mBindings[1].mStride = sizeof(uint32_t);
        wvl.mBindings[1].mRate   = VERTEX_BINDING_RATE_INSTANCE;
        wvl.mAttribCount = 2;
        wvl.mAttribs[0].mSemantic = SEMANTIC_POSITION;
        wvl.mAttribs[0].mFormat   = TinyImageFormat_R32G32B32_SFLOAT;
        wvl.mAttribs[0].mBinding  = 0;
        wvl.mAttribs[0].mLocation = 0;
        wvl.mAttribs[0].mOffset   = 0;
        wvl.mAttribs[1].mSemantic = SEMANTIC_TEXCOORD1;       // DrawIndex (matches water.vert)
        wvl.mAttribs[1].mFormat   = TinyImageFormat_R32_UINT;
        wvl.mAttribs[1].mBinding  = 1;
        wvl.mAttribs[1].mLocation = 1;
        wvl.mAttribs[1].mOffset   = 0;

        DepthStateDesc wDepth = {};
        wDepth.mDepthTest = true;
        wDepth.mDepthWrite = true;
        wDepth.mDepthFunc = CMP_GEQUAL;        // reverse-Z: water writes + occludes

        RasterizerStateDesc wRaster = {};
        wRaster.mCullMode = CULL_MODE_NONE;    // matches MGE water (P6/P7, double-sided plane)
        wRaster.mFrontFace = FRONT_FACE_CCW;

        PipelineDesc wPd = {};
        wPd.mType = PIPELINE_TYPE_GRAPHICS;
        GraphicsPipelineDesc& wg = wPd.mGraphicsDesc;
        wg.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
        wg.mRenderTargetCount = 1;
        wg.pColorFormats = &g_live.pRT->mFormat;
        wg.mSampleCount = (SampleCount)g_live.sampleCount;
        wg.mSampleQuality = 0;
        wg.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
        wg.pDepthState = &wDepth;
        wg.pVertexLayout = &wvl;
        wg.pRasterizerState = &wRaster;
        wg.pShaderProgram = g_live.pWaterShader;
        addPipeline(R, &wPd, &g_live.pWaterPipeline);
        if (!g_live.pWaterPipeline) {
            LOG::logline("!! [forge][water] addPipeline(water) FAILED"); LOG::flush();
            return false;
        }
        return true;
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
        // Tier 2: depth must be sampleable by the linearize compute. Add SRV capability (keeps the
        // DEPTH_WRITE start state; renderScene barriers DEPTH_WRITE→SHADER_RESOURCE before the
        // compute reads it, then back to DEPTH_WRITE for the colour EQUAL load). MSAA-safe: the
        // SRV is a Texture2DMS view the linearize shader Loads sample 0 of.
        dDesc.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
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

        // --- Tier 2: single-sample linear depth + AO targets (SRV+UAV). Created here so the
        // graphics PerFrame set (below) can bind pAO as its gAO SRV. Both start UNORDERED_ACCESS
        // (the compute writes them first each frame); renderScene barriers them to SHADER_RESOURCE
        // after the dispatch so the colour pass / F12 debug can sample. ---
        {
            TextureDesc ld = {};
            ld.mWidth = width; ld.mHeight = height; ld.mDepth = 1;
            ld.mArraySize = 1; ld.mMipLevels = 1;
            ld.mSampleCount = SAMPLE_COUNT_1;                       // ALWAYS single-sample (the resolve target)
            ld.mFormat = TinyImageFormat_R32_SFLOAT;
            ld.mStartState = RESOURCE_STATE_UNORDERED_ACCESS;
            ld.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
            ld.pName = "linearDepth";
            TextureLoadDesc lld = {};
            lld.ppTexture = &g_live.pLinearDepth;
            lld.pDesc = &ld;
            addResource(&lld, nullptr);

            TextureDesc ad = {};
            ad.mWidth = width; ad.mHeight = height; ad.mDepth = 1;
            ad.mArraySize = 1; ad.mMipLevels = 1;
            ad.mSampleCount = SAMPLE_COUNT_1;
            ad.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;       // rgb bent normal, a visibility (option A)
            ad.mStartState = RESOURCE_STATE_UNORDERED_ACCESS;
            ad.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
            ad.pName = "aoBuffer";
            TextureLoadDesc ald = {};
            ald.ppTexture = &g_live.pAO;
            ald.pDesc = &ad;
            addResource(&ald, nullptr);

            // pAOBlur: bilateral-blur destination, identical format/desc to pAO. The colour frags
            // sample THIS (not raw pAO) as gAO; pAO/pLinearDepth feed the blur as SRVs.
            TextureDesc abd = ad;             // same RGBA16F, SRV+UAV, starts UNORDERED_ACCESS
            abd.pName = "aoBlurBuffer";
            TextureLoadDesc abld = {};
            abld.ppTexture = &g_live.pAOBlur;
            abld.pDesc = &abd;
            addResource(&abld, nullptr);

            // gAOParams cbuffer (invViewProj + screen + knobs + eye), uploaded per frame.
            // UNIFORM_BUFFER (CBV), NOT DESCRIPTOR_TYPE_BUFFER: a structured-SRV view over a
            // CPU_TO_GPU (UPLOAD-heap) resource is illegal in D3D12 and silently removes the device
            // (the fault only surfaces lazily at the next PSO create — see Tier 2 bisect).
            BufferLoadDesc cb = {};
            cb.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            cb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            cb.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            cb.mDesc.mSize = 256;                                   // >= sizeof(AOParams) (112B), CBV-aligned
            cb.mDesc.pName = "aoParamsCbv";
            cb.pData = nullptr;
            cb.ppBuffer = &g_live.pAOParamsCbv;
            addResource(&cb, nullptr);

            waitForAllResourceLoads();
            if (!g_live.pLinearDepth || !g_live.pAO || !g_live.pAOBlur || !g_live.pAOParamsCbv) {
                std::printf("[forge] Tier 2 AO resource alloc FAILED\n");
                return false;
            }
            // Diagnostic: does the GPU support typed UAV store for the AO format? R32_SFLOAT (lin depth)
            // is base-guaranteed; R16G16B16A16_SFLOAT needs the additional-format cap. 0x8 = READ_WRITE,
            // 0x4 = WRITE. If the AO format lacks WRITE/READ_WRITE, gtao's Write2D silently no-ops.
            std::printf("[forge] FORMAT CAPS: R32_SFLOAT=0x%X  R16G16B16A16_SFLOAT=0x%X  R32G32B32A32_SFLOAT=0x%X\n",
                        (unsigned)R->pGpu->mFormatCaps[TinyImageFormat_R32_SFLOAT],
                        (unsigned)R->pGpu->mFormatCaps[TinyImageFormat_R16G16B16A16_SFLOAT],
                        (unsigned)R->pGpu->mFormatCaps[TinyImageFormat_R32G32B32A32_SFLOAT]);
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
            // PerFrame set: gFrameData CBV + gAO SRV. The frags read the BILATERAL-BLURRED AO
            // (pAOBlur), not raw pAO; pAOBlur's per-frame UAV<->SHADER_RESOURCE ping-pong (renderScene)
            // leaves it SHADER_RESOURCE before the colour pass samples it. (Raw pAO still feeds the
            // blur as an SRV, and the DebugTextures/readback paths still inspect pAO directly.)
            DescriptorData p[2] = {};
            p[0].mIndex = SRT_RES_IDX(SrtData, PerFrame, gFrameData);
            p[0].ppBuffers = &g_live.pFrameCbv;
            p[1].mIndex = SRT_RES_IDX(SrtData, PerFrame, gAO);
            p[1].mCount = 1;   // single-texture SRV needs explicit count (else binds nothing)
            p[1].ppTextures = &g_live.pAOBlur;
            updateDescriptorSet(R, 0, g_live.pPerFrameSet, 2, p);
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

            // gStaticsArrays (distant-statics buckets) shares the Persistent table. Initialize EVERY
            // element to a 4x4x2 white Texture2DArray so the table holds no uninitialized descriptors
            // before the buckets build (first exterior). buildStaticsTextureArrays re-binds the real
            // buckets; the unused tail keeps pointing here. mArraySize=2 forces a 2DArray SRV (a
            // 1-slice texture reflects as a plain Tex2D, the wrong descriptor dimension for the array).
            {
                TextureDesc sw = {};
                sw.mWidth = 4; sw.mHeight = 4; sw.mDepth = 1;
                sw.mArraySize = 2; sw.mMipLevels = 1;
                sw.mSampleCount = SAMPLE_COUNT_1;
                sw.mFormat = TinyImageFormat_R8G8B8A8_UNORM;
                sw.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
                sw.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
                sw.pName = "staticsWhiteArray";
                TextureLoadDesc swl = {};
                swl.ppTexture = &g_live.pStaticsWhiteArray;
                swl.pDesc = &sw;
                addResource(&swl, nullptr);
                waitForAllResourceLoads();
                if (!g_live.pStaticsWhiteArray) { std::printf("[forge] statics white array creation FAILED\n"); return false; }
                for (uint32_t layer = 0; layer < 2; ++layer) {
                    TextureUpdateDesc wu = {};
                    wu.pTexture = g_live.pStaticsWhiteArray;
                    wu.mBaseMipLevel = 0; wu.mMipLevels = 1;
                    wu.mBaseArrayLayer = layer; wu.mLayerCount = 1;
                    wu.mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
                    beginUpdateResource(&wu);
                    TextureSubresourceUpdate s = wu.getSubresourceUpdateDesc(0, layer);
                    for (uint32_t row = 0; row < s.mRowCount; ++row) {
                        uint32_t* dst = (uint32_t*)(s.pMappedData + (size_t)row * s.mDstRowStride);
                        for (uint32_t px = 0; px < 4; ++px) { dst[px] = 0xFFFFFFFFu; }
                    }
                    endUpdateResource(&wu);
                }
                flushTextureUploads(R);

                std::vector<Texture*> whites(MAX_STATICS_BUCKETS, g_live.pStaticsWhiteArray);
                DescriptorData sd = {};
                sd.mIndex = SRT_RES_IDX(SrtData, Persistent, gStaticsArrays);
                sd.mCount = MAX_STATICS_BUCKETS;
                sd.ppTextures = whites.data();
                updateDescriptorSet(R, 0, g_live.pPersistentSet, 1, &sd);
            }
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

        // --- SK1: sky shader + pipeline (FIRST blend PSO, depth off) + world window + instance VB ---
        // Reuses the opaque GeomVertexWire layout (vl) + SrtData/default.rootsig + bindless gTextures;
        // sky.vert reads gBatch.worlds as ONE 64KB window (pSkyWorldsBuf, like multimap), drawn with
        // firstInstance = the part's world index. The dome is alpha-blended into the transparent-
        // cleared colour target → premultiplied; the present-seam composites it over MW. Depth test+
        // write OFF (pure background): drawn first, the opaque world (replace blend) overwrites it
        // where geometry exists. Built unconditionally (unused/zero-cost when the SK1 toggle is off).
        {
            ShaderLoadDesc skDesc = {};
            skDesc.mVert.pFileName = "sky.vert";
            skDesc.mFrag.pFileName = "sky.frag";
            addShader(R, &skDesc, &g_live.pSkyShader);
            if (!g_live.pSkyShader) {
                std::printf("[forge] addShader(sky) FAILED\n");
                return false;
            }

            DepthStateDesc skyDepth = {};
            skyDepth.mDepthTest = false;
            skyDepth.mDepthWrite = false;

            // Standard transparency blend (the host's first). Colour target is cleared to (0,0,0,0)
            // so result.rgb = src.rgb*a (premultiplied); alpha channel ONE/INVSRCALPHA accumulates
            // coverage → result.a = a. Exactly what the present-seam ONE/INVSRCALPHA composite wants.
            BlendStateDesc skyBlend = {};
            skyBlend.mSrcFactors[0]      = BC_SRC_ALPHA;
            skyBlend.mDstFactors[0]      = BC_ONE_MINUS_SRC_ALPHA;
            skyBlend.mSrcAlphaFactors[0] = BC_ONE;
            skyBlend.mDstAlphaFactors[0] = BC_ONE_MINUS_SRC_ALPHA;
            skyBlend.mBlendModes[0]      = BM_ADD;
            skyBlend.mBlendAlphaModes[0] = BM_ADD;
            skyBlend.mColorWriteMasks[0] = COLOR_MASK_ALL;
            skyBlend.mRenderTargetMask   = BLEND_STATE_TARGET_0;
            skyBlend.mIndependentBlend   = false;

            // Cull NONE: the dome is a single-sided shell viewed from inside; MW draws it without a
            // depth-tested winding dependency. NONE keeps SK1 winding-agnostic (draw order owns it).
            RasterizerStateDesc skyRaster = {};
            skyRaster.mCullMode = CULL_MODE_NONE;
            skyRaster.mFrontFace = FRONT_FACE_CCW;

            PipelineDesc skPd = {};
            skPd.mType = PIPELINE_TYPE_GRAPHICS;
            GraphicsPipelineDesc& sg = skPd.mGraphicsDesc;
            sg.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
            sg.mRenderTargetCount = 1;
            sg.pColorFormats = &g_live.pRT->mFormat;
            sg.mSampleCount = (SampleCount)g_live.sampleCount;
            sg.mSampleQuality = 0;
            sg.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
            sg.pDepthState = &skyDepth;
            sg.pBlendState = &skyBlend;
            sg.pVertexLayout = &vl;                 // reuse the opaque GeomVertexWire layout
            sg.pRasterizerState = &skyRaster;
            sg.pShaderProgram = g_live.pSkyShader;
            addPipeline(R, &skPd, &g_live.pSkyPipeline);
            if (!g_live.pSkyPipeline) {
                std::printf("[forge] addPipeline(sky) FAILED\n");
                return false;
            }

            // SK2: a second, ADDITIVE sky PSO (SRCALPHA/ONE) — groundwork for the deferred sun
            // glare + cloud layers. Identical to the alpha-over PSO except dst colour = ONE
            // (and dst alpha = ONE so coverage accumulates). skyPipelineFor() picks per draw by
            // the captured (src,dst) blend pair; the SK2 elements (sun/moons/stars) are all
            // alpha-over and use pSkyPipeline. Built unconditionally (zero cost when unused).
            BlendStateDesc skyBlendAdd = skyBlend;
            skyBlendAdd.mDstFactors[0]      = BC_ONE;
            skyBlendAdd.mDstAlphaFactors[0] = BC_ONE;
            sg.pBlendState = &skyBlendAdd;
            addPipeline(R, &skPd, &g_live.pSkyPipelineAdd);
            if (!g_live.pSkyPipelineAdd) {
                std::printf("[forge] addPipeline(sky additive) FAILED\n");
                return false;
            }

            // One 64KB world window + a per-draw instance VB (kStaticInstU32 slots, like opaque).
            BufferLoadDesc swb = {};
            swb.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            swb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            swb.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            swb.mDesc.mSize = kBatchBytes;
            swb.mDesc.pName = "skyWorldsCbv";
            swb.pData = nullptr;
            swb.ppBuffer = &g_live.pSkyWorldsBuf;
            addResource(&swb, nullptr);

            BufferLoadDesc sib = {};
            sib.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
            sib.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            sib.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            sib.mDesc.mSize = (uint64_t)kMaxSkyDraws * kStaticInstU32 * sizeof(uint32_t);
            sib.mDesc.pName = "instanceVBSky";
            sib.pData = nullptr;
            sib.ppBuffer = &g_live.pSkyInstanceBuf;
            addResource(&sib, nullptr);

            waitForAllResourceLoads();
            if (!g_live.pSkyWorldsBuf || !g_live.pSkyInstanceBuf) {
                return false;
            }

            // Sky PerBatch set (1 instance): gBatch = the single sky world window.
            DescriptorSetDesc skbDesc = SRT_SET_DESC(SrtData, PerBatch, 1, 0);
            addDescriptorSet(R, &skbDesc, &g_live.pPerBatchSetSky);
            if (!g_live.pPerBatchSetSky) {
                return false;
            }
            DescriptorData skp = {};
            skp.mIndex = SRT_RES_IDX(SrtData, PerBatch, gBatch);
            skp.ppBuffers = &g_live.pSkyWorldsBuf;
            updateDescriptorSet(R, 0, g_live.pPerBatchSetSky, 1, &skp);
        }

        // --- WT1: Forge water (host-generated geo-clipmap surface) ------------------------------
        // Own position-only vertex layout (float3 + per-instance DrawIndex=level). Reuses the shared
        // SrtData/default.rootsig: worlds+params ride gBatch (pWaterWorldsBuf, one window, like sky),
        // and the 4 water SRVs append to the PerFrame set (bound below). Depth GEQUAL + write
        // (reverse-Z, water occludes / is occluded), cull NONE (MGE P6/P7), no blend, frag alpha=1.
        {
            // Build the clipmap mesh + load the animated-normal volume. If either fails, leave water
            // unbuilt (waterReady stays false) — the rest of the path is unaffected (clean A/B).
            const bool meshOk = buildWaterMesh(R);
            const bool volOk  = loadWaterNormalVolume(R);

            // Water shader + graphics pipeline (extracted so the hot-reload path can rebuild them
            // when water.vert/.frag change on disk — graphics shaders aren't reloaded by the
            // compute-only checkShaderHotReload otherwise).
            if (!buildWaterPipeline(R)) {
                return false;
            }

            // pRefractColor: screen-sized copy target of the pre-water colour (refraction source).
            // SRV only (the copy is a CopyResource/ResolveSubresource, not a render target).
            {
                TextureDesc rd = {};
                rd.mWidth = width; rd.mHeight = height; rd.mDepth = 1;
                rd.mArraySize = 1; rd.mMipLevels = 1;
                rd.mSampleCount = SAMPLE_COUNT_1;
                rd.mFormat = TinyImageFormat_B8G8R8A8_UNORM;       // matches the shared RT
                rd.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
                rd.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
                rd.pName = "refractColor";
                TextureLoadDesc rld = {};
                rld.ppTexture = &g_live.pRefractColor;
                rld.pDesc = &rd;
                addResource(&rld, nullptr);
            }

            // gBatch window for water (worlds[0..5]=levels, [6]=params, [7]=invVP) + per-draw instance
            // VB (DrawIndex per level). Same layout as the sky window.
            BufferLoadDesc wwb = {};
            wwb.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            wwb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            wwb.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            wwb.mDesc.mSize = kBatchBytes;
            wwb.mDesc.pName = "waterWorldsCbv";
            wwb.pData = nullptr;
            wwb.ppBuffer = &g_live.pWaterWorldsBuf;
            addResource(&wwb, nullptr);

            BufferLoadDesc wib = {};
            wib.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
            wib.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            wib.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            wib.mDesc.mSize = (uint64_t)kMaxWaterLevels * sizeof(uint32_t);
            wib.mDesc.pName = "instanceVBWater";
            wib.pData = nullptr;
            wib.ppBuffer = &g_live.pWaterInstanceBuf;
            addResource(&wib, nullptr);

            waitForAllResourceLoads();
            if (!g_live.pRefractColor || !g_live.pWaterWorldsBuf || !g_live.pWaterInstanceBuf) {
                std::printf("[forge] water resource alloc FAILED\n");
                return false;
            }
            // Fill the per-draw instance VB once: DrawIndex[level] = level (selects gBatch.worlds[level]).
            {
                uint32_t* wi = (uint32_t*)g_live.pWaterInstanceBuf->pCpuMappedAddress;
                for (uint32_t i = 0; i < kMaxWaterLevels; ++i) { wi[i] = i; }
            }

            DescriptorSetDesc wbDesc = SRT_SET_DESC(SrtData, PerBatch, 1, 0);
            addDescriptorSet(R, &wbDesc, &g_live.pPerBatchSetWater);
            if (!g_live.pPerBatchSetWater) { return false; }
            DescriptorData wbp = {};
            wbp.mIndex = SRT_RES_IDX(SrtData, PerBatch, gBatch);
            wbp.ppBuffers = &g_live.pWaterWorldsBuf;
            updateDescriptorSet(R, 0, g_live.pPerBatchSetWater, 1, &wbp);

            // waterReady gates the per-frame pass: needs the mesh, pipeline, worlds window AND the
            // animated-normal volume (the frag samples a Tex3D — binding a 2D fallback to a 3D slot is
            // an illegal type mismatch, so the volume is mandatory; it's a shipped MGE asset).
            g_live.waterReady = meshOk && volOk && g_live.pWaterNormalVol && g_live.pWaterPipeline
                              && g_live.pWaterWorldsBuf && g_live.pWaterVB && g_live.pWaterIB;

            // Bind the 4 water SRVs into the (already-created) PerFrame set — only when fully ready, so
            // the Tex3D slot never gets a null/2D binding. These are STABLE views; the refraction/scene-
            // depth CONTENTS change per frame (the descriptors don't). gSceneLinDepth = pLinearDepth (RAW
            // reverse-Z device depth, Tier 2 AO block above). gReflectColor is the WT1 stand-in — bind
            // pRefractColor as a valid placeholder (the frag ignores it in WT1; WT2 rebinds the mirror RT).
            if (g_live.waterReady) {
                DescriptorData wp[4] = {};
                wp[0].mIndex = SRT_RES_IDX(SrtData, PerFrame, gWaterNormalVol);
                wp[0].mCount = 1; wp[0].ppTextures = &g_live.pWaterNormalVol;
                wp[1].mIndex = SRT_RES_IDX(SrtData, PerFrame, gRefractColor);
                wp[1].mCount = 1; wp[1].ppTextures = &g_live.pRefractColor;
                wp[2].mIndex = SRT_RES_IDX(SrtData, PerFrame, gSceneLinDepth);
                wp[2].mCount = 1; wp[2].ppTextures = &g_live.pLinearDepth;
                wp[3].mIndex = SRT_RES_IDX(SrtData, PerFrame, gReflectColor);
                wp[3].mCount = 1; wp[3].ppTextures = &g_live.pRefractColor;   // WT1 stand-in
                updateDescriptorSet(R, 0, g_live.pPerFrameSet, 4, wp);
            }
            std::printf("[forge][water] build: mesh=%d vol=%d ready=%d\n",
                        (int)meshOk, (int)volOk, (int)g_live.waterReady);
        }

        // --- WT2: reflection RT (1024²) + mirror-view frame cbuffer/set + reflect sky buffers ------
        // Stage 1 renders SKY ONLY into pReflectColor with the mirror matrix; the water frag samples
        // it (gReflectColor re-bound to pReflectColor below, replacing the WT1 stand-in). Reuses the
        // sky pipeline (cull NONE → winding-agnostic, so the mirror's handedness flip is a no-op here).
        {
            // 1024² colour RT (alpha = coverage, cleared transparent) + matching D32 reverse-Z depth.
            RenderTargetDesc rcd = {};
            rcd.mWidth = kReflectSize; rcd.mHeight = kReflectSize; rcd.mDepth = 1;
            rcd.mArraySize = 1; rcd.mMipLevels = 1; rcd.mSampleCount = SAMPLE_COUNT_1;
            rcd.mFormat = TinyImageFormat_B8G8R8A8_UNORM;
            rcd.mStartState = RESOURCE_STATE_SHADER_RESOURCE;   // resting; pass flips to RENDER_TARGET
            rcd.mClearValue.r = 0.0f;
            rcd.mClearValue.g = 0.0f;
            rcd.mClearValue.b = 0.0f;
            rcd.mClearValue.a = 0.0f;   // transparent: alpha = coverage (premultiplied, composited over fog)
            rcd.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
            rcd.pName = "reflectColor";
            addRenderTarget(R, &rcd, &g_live.pReflectColor);

            RenderTargetDesc rdd = {};
            rdd.mWidth = kReflectSize; rdd.mHeight = kReflectSize; rdd.mDepth = 1;
            rdd.mArraySize = 1; rdd.mMipLevels = 1; rdd.mSampleCount = SAMPLE_COUNT_1;
            rdd.mFormat = TinyImageFormat_D32_SFLOAT;
            rdd.mStartState = RESOURCE_STATE_DEPTH_WRITE;
            rdd.mClearValue.depth = 0.0f;   // reverse-Z clear
            rdd.mClearValue.stencil = 0;
            rdd.pName = "reflectDepth";
            addRenderTarget(R, &rdd, &g_live.pReflectDepth);

            // Mirror-view frame cbuffer (own viewProj; same 256B layout as gFrameData).
            BufferLoadDesc rfb = {};
            rfb.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            rfb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            rfb.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            rfb.mDesc.mSize = 256;
            rfb.mDesc.pName = "reflectFrameCbv";
            rfb.pData = nullptr;
            rfb.ppBuffer = &g_live.pReflectFrameCbv;
            addResource(&rfb, nullptr);

            // Reflect sky gBatch window + instance VB (own copies; the reflect pass fills + draws them
            // BEFORE the main sky pass, so they must be decoupled from pSkyWorldsBuf/pSkyInstanceBuf).
            BufferLoadDesc rsw = {};
            rsw.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            rsw.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            rsw.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            rsw.mDesc.mSize = kBatchBytes;
            rsw.mDesc.pName = "reflectSkyWorldsCbv";
            rsw.pData = nullptr;
            rsw.ppBuffer = &g_live.pReflectSkyWorldsBuf;
            addResource(&rsw, nullptr);

            BufferLoadDesc rsi = {};
            rsi.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
            rsi.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            rsi.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            rsi.mDesc.mSize = (uint64_t)kMaxSkyDraws * kStaticInstU32 * sizeof(uint32_t);
            rsi.mDesc.pName = "instanceVBReflectSky";
            rsi.pData = nullptr;
            rsi.ppBuffer = &g_live.pReflectSkyInstanceBuf;
            addResource(&rsi, nullptr);

            waitForAllResourceLoads();
            if (!g_live.pReflectColor || !g_live.pReflectDepth || !g_live.pReflectFrameCbv
                || !g_live.pReflectSkyWorldsBuf || !g_live.pReflectSkyInstanceBuf) {
                std::printf("[forge] reflection resource alloc FAILED\n");
                return false;
            }

            // pPerFrameSetReflect: a SECOND PerFrame set bound to pReflectFrameCbv (so the reflection
            // pass uses the mirror viewProj while the main pass keeps rzViewProj — two matrices in one
            // command buffer need two cbuffers). gAO + the 4 water SRVs are bound to the same textures
            // as the main set (the sky frag ignores them; the set layout just needs them valid).
            DescriptorSetDesc rpfDesc = SRT_SET_DESC(SrtData, PerFrame, 1, 0);
            addDescriptorSet(R, &rpfDesc, &g_live.pPerFrameSetReflect);
            DescriptorSetDesc rsbDesc = SRT_SET_DESC(SrtData, PerBatch, 1, 0);
            addDescriptorSet(R, &rsbDesc, &g_live.pPerBatchSetReflectSky);
            if (!g_live.pPerFrameSetReflect || !g_live.pPerBatchSetReflectSky) { return false; }
            {
                Texture* vol = g_live.pWaterNormalVol ? g_live.pWaterNormalVol : g_live.pDefaultWhite;
                DescriptorData p[6] = {};
                p[0].mIndex = SRT_RES_IDX(SrtData, PerFrame, gFrameData);
                p[0].ppBuffers = &g_live.pReflectFrameCbv;
                p[1].mIndex = SRT_RES_IDX(SrtData, PerFrame, gAO);
                p[1].mCount = 1; p[1].ppTextures = &g_live.pAOBlur;
                p[2].mIndex = SRT_RES_IDX(SrtData, PerFrame, gWaterNormalVol);
                p[2].mCount = 1; p[2].ppTextures = &vol;
                p[3].mIndex = SRT_RES_IDX(SrtData, PerFrame, gRefractColor);
                p[3].mCount = 1; p[3].ppTextures = &g_live.pRefractColor;
                p[4].mIndex = SRT_RES_IDX(SrtData, PerFrame, gSceneLinDepth);
                p[4].mCount = 1; p[4].ppTextures = &g_live.pLinearDepth;
                p[5].mIndex = SRT_RES_IDX(SrtData, PerFrame, gReflectColor);
                p[5].mCount = 1; p[5].ppTextures = &g_live.pRefractColor;   // reflect set never reads this
                updateDescriptorSet(R, 0, g_live.pPerFrameSetReflect, 6, p);
            }
            {
                DescriptorData rbp = {};
                rbp.mIndex = SRT_RES_IDX(SrtData, PerBatch, gBatch);
                rbp.ppBuffers = &g_live.pReflectSkyWorldsBuf;
                updateDescriptorSet(R, 0, g_live.pPerBatchSetReflectSky, 1, &rbp);
            }

            g_live.reflectReady = g_live.waterReady && g_live.pReflectColor && g_live.pReflectDepth
                                && g_live.pReflectFrameCbv && g_live.pPerFrameSetReflect
                                && g_live.pPerBatchSetReflectSky;

            // Re-point gReflectColor (in the MAIN PerFrame set) at the real reflection RT, replacing
            // WT1's pRefractColor stand-in. Only when ready, so the water frag samples a valid RT.
            if (g_live.reflectReady && g_live.waterReady) {
                Texture* reflTex = g_live.pReflectColor->pTexture;
                DescriptorData rp = {};
                rp.mIndex = SRT_RES_IDX(SrtData, PerFrame, gReflectColor);
                rp.mCount = 1; rp.ppTextures = &reflTex;
                updateDescriptorSet(R, 0, g_live.pPerFrameSet, 1, &rp);
            }
            std::printf("[forge][reflect] build ready=%d (%u²)\n", (int)g_live.reflectReady, kReflectSize);
        }

        // --- Tier 2: compute pipelines (linearize + GTAO) + their descriptor sets. First
        // PIPELINE_TYPE_COMPUTE in the host; both use the global compute root signature. ---
        {
            // Linearize: pick the MSAA / non-MSAA variant by the live sample count. Only sc1/sc4
            // are compiled (the user runs 4x, probe-verified); any sampleCount>1 maps to sc4.
            const char* linName = (g_live.sampleCount > 1) ? "linearizedepth_sc4.comp"
                                                           : "linearizedepth_sc1.comp";
            ShaderLoadDesc lsd = {};
            lsd.mComp.pFileName = linName;
            addShader(R, &lsd, &g_live.pLinearizeShader);
            ShaderLoadDesc gsd = {};
            gsd.mComp.pFileName = "gtao.comp";
            addShader(R, &gsd, &g_live.pGtaoShader);
            ShaderLoadDesc absd = {};
            absd.mComp.pFileName = "aoblur.comp";
            addShader(R, &absd, &g_live.pAOBlurShader);
            if (!g_live.pLinearizeShader || !g_live.pGtaoShader || !g_live.pAOBlurShader) {
                std::printf("[forge] addShader(compute %s/gtao.comp/aoblur.comp) FAILED\n", linName);
                return false;
            }

            PipelineDesc lpd = {};
            lpd.mType = PIPELINE_TYPE_COMPUTE;
            lpd.mComputeDesc.pShaderProgram = g_live.pLinearizeShader;
            addPipeline(R, &lpd, &g_live.pLinearizePipeline);
            PipelineDesc gpd = {};
            gpd.mType = PIPELINE_TYPE_COMPUTE;
            gpd.mComputeDesc.pShaderProgram = g_live.pGtaoShader;
            addPipeline(R, &gpd, &g_live.pGtaoPipeline);
            PipelineDesc abpd = {};
            abpd.mType = PIPELINE_TYPE_COMPUTE;
            abpd.mComputeDesc.pShaderProgram = g_live.pAOBlurShader;
            addPipeline(R, &abpd, &g_live.pAOBlurPipeline);
            if (!g_live.pLinearizePipeline || !g_live.pGtaoPipeline || !g_live.pAOBlurPipeline) {
                std::printf("[forge] addPipeline(compute linearize/gtao/aoblur) FAILED\n");
                return false;
            }

            // Linearize PerBatch set: gSceneDepth (pDepth SRV) + gLinearDepthOut (pLinearDepth UAV).
            DescriptorSetDesc lset = SRT_SET_DESC(LinDepthSrtData, PerBatch, 1, 0);
            addDescriptorSet(R, &lset, &g_live.pLinearizeSet);
            // GTAO PerFrame set: gAOParams cbuffer. PerBatch set: gLinearDepthIn SRV + gAOOut UAV.
            DescriptorSetDesc gbset = SRT_SET_DESC(AOSrtData, PerDraw, 1, 0);
            addDescriptorSet(R, &gbset, &g_live.pGtaoBatchSet);
            // AO blur PerFrame set (root index distinct from PerBatch/PerDraw — no rebind collision).
            DescriptorSetDesc abset = SRT_SET_DESC(AOBlurSrtData, PerFrame, 1, 0);
            addDescriptorSet(R, &abset, &g_live.pAOBlurSet);
            if (!g_live.pLinearizeSet || !g_live.pGtaoBatchSet || !g_live.pAOBlurSet) {
                std::printf("[forge] addDescriptorSet(compute) FAILED\n");
                return false;
            }
            {
                // mCount=1 is REQUIRED for single-texture binds (SRV and UAV) — without it the
                // descriptor count is 0 and the slot binds NOTHING (UAV writes/SRV reads vanish).
                // Buffers/CBVs default to 1, which is why the opaque path's bindless array (mCount set)
                // and CBVs worked but these first single textures did not.
                DescriptorData d[2] = {};
                d[0].mIndex = SRT_RES_IDX(LinDepthSrtData, PerBatch, gSceneDepth);
                d[0].mCount = 1;
                d[0].ppTextures = &g_live.pDepth->pTexture;   // depth target's underlying texture (SRV)
                d[1].mIndex = SRT_RES_IDX(LinDepthSrtData, PerBatch, gLinearDepthOut);
                d[1].mCount = 1;
                d[1].ppTextures = &g_live.pLinearDepth;
                updateDescriptorSet(R, 0, g_live.pLinearizeSet, 2, d);
            }
            {
                // PerDraw set (root 0, distinct from linearize's PerBatch root 1): CBV + SRV + UAV.
                DescriptorData d[3] = {};
                d[0].mIndex = SRT_RES_IDX(AOSrtData, PerDraw, gAOParams);
                d[0].ppBuffers = &g_live.pAOParamsCbv;
                d[1].mIndex = SRT_RES_IDX(AOSrtData, PerDraw, gLinearDepthIn);
                d[1].mCount = 1;
                d[1].ppTextures = &g_live.pLinearDepth;
                d[2].mIndex = SRT_RES_IDX(AOSrtData, PerDraw, gAOOut);
                d[2].mCount = 1;
                d[2].ppTextures = &g_live.pAO;
                updateDescriptorSet(R, 0, g_live.pGtaoBatchSet, 3, d);
            }
            {
                // AO blur PerFrame set: CBV (shared pAOParamsCbv) + 2 SRV (pAO, pLinearDepth) + UAV (pAOBlur).
                DescriptorData d[4] = {};
                d[0].mIndex = SRT_RES_IDX(AOBlurSrtData, PerFrame, gBlurParams);
                d[0].ppBuffers = &g_live.pAOParamsCbv;
                d[1].mIndex = SRT_RES_IDX(AOBlurSrtData, PerFrame, gAOSrc);
                d[1].mCount = 1;
                d[1].ppTextures = &g_live.pAO;
                d[2].mIndex = SRT_RES_IDX(AOBlurSrtData, PerFrame, gBlurDepthIn);
                d[2].mCount = 1;
                d[2].ppTextures = &g_live.pLinearDepth;
                d[3].mIndex = SRT_RES_IDX(AOBlurSrtData, PerFrame, gAODst);
                d[3].mCount = 1;
                d[3].ppTextures = &g_live.pAOBlur;
                updateDescriptorSet(R, 0, g_live.pAOBlurSet, 4, d);
            }
            std::printf("[forge] SET HANDLES: linBatch root=%u handle=%llu stride=%u | gtaoBatch root=%u handle=%llu stride=%u\n",
                        (unsigned)g_live.pLinearizeSet->mDx.mCbvSrvUavRootIndex, (unsigned long long)g_live.pLinearizeSet->mDx.mCbvSrvUavHandle, (unsigned)g_live.pLinearizeSet->mDx.mCbvSrvUavStride,
                        (unsigned)g_live.pGtaoBatchSet->mDx.mCbvSrvUavRootIndex, (unsigned long long)g_live.pGtaoBatchSet->mDx.mCbvSrvUavHandle, (unsigned)g_live.pGtaoBatchSet->mDx.mCbvSrvUavStride);
            std::printf("[forge] Tier 2 compute ready (linearize=%s, gtao.comp; pLinearDepth R32F, pAO RGBA16F)\n",
                        linName);
        }

        // GPU timestamp pool for the per-phase 4ms breakdown (kGpuPhaseCount begin/end pairs).
        // Failure is non-fatal — the breakdown just stays 0 (don't gate the scene on profiling).
        {
            QueryPoolDesc qd = {};
            qd.pName = "GpuPhaseTimestamps";
            qd.mType = QUERY_TYPE_TIMESTAMP;
            qd.mQueryCount = kGpuPhaseCount;
            initQueryPool(R, &qd, &g_live.pGpuQueryPool);
            if (g_live.pGpuQueryPool) {
                getTimestampFrequency(g_live.pQueue, &g_live.gpuTickFreq);
                std::printf("[forge] GPU timestamp pool ready (%u phases, freq=%.0f ticks/s)\n",
                            (unsigned)kGpuPhaseCount, g_live.gpuTickFreq);
            }
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
    unsigned  g_lastSkyDrawn = 0;      // SK1 sky parts actually drawn in the last renderScene
    unsigned  g_lastLightCount = 0;    // point lights uploaded in the last renderScene
    // Phase 0 panel readouts: extra per-frame counters + single-frame timings (the 300-frame
    // accumulators g_recAccum/g_gpuAccum are for the heartbeat; these are this frame's values).
    unsigned  g_lastReflSkyDrawn = 0;  // sky shapes drawn into the reflection RT (WT2)
    unsigned  g_lastWaterLevels  = 0;  // water clipmap LOD levels drawn (WT1)
    double    g_lastRecMs = 0.0;       // this frame's CPU command-record ms (beginCmd→endCmd)
    double    g_lastGpuMs = 0.0;       // this frame's GPU submit→fence ms (incl. resolve)
    // Pre-record CPU split (the ~2ms the client saw that record+gpu didn't account for): the host
    // frame is setup + cull + record + gpu. setup = per-draw world/instance memcpy + cbuffers;
    // cull = dlLiveCullAndBuild (CPU frustum/tier cull + ring fill + lazy resource/texture load,
    // MUST run pre-beginCmd). post = after the fence (panel/heartbeat). total = whole renderScene.
    double    g_lastSetupMs = 0.0;
    double    g_lastCullMs  = 0.0;
    double    g_lastPostMs  = 0.0;
    double    g_lastTotalMs = 0.0;
    // GPU-side per-phase ms (breaks down the ~4ms gpu via timestamp queries). kGpuPhase* enum
    // is defined up near kMaxDraws (used in buildOpaquePath, earlier in the TU than this block).
    double    g_lastGpuPhaseMs[kGpuPhaseCount] = {};
    // Live DL cull survivors (set in dlLiveCullAndBuild). Declared here (not in the DL section lower
    // in the TU) so the Phase 0 panel readout in drawDevUI can see them.
    uint32_t  g_liveLastInst = 0, g_liveLastSubsets = 0, g_liveLastLand = 0;
    uint32_t  g_lastCullExamined = 0;   // instances the cull TESTED this frame (cull-ms normalizer)
    uint32_t  g_lastGpuCullCount = 0;   // Stage B (B2): GPU cull survivor count (Σ numSubsets), read
                                        // back post-fence; must equal g_liveLastInst (CPU) once correct.

    // Phase 0 panel toggles: turn a Forge subsystem's draw OFF to reveal MW's own version through the
    // premultiplied composite (instant visual A/B). All default ON (no behaviour change). Host-side —
    // they gate the host render blocks; MGE's counterparts are gated separately on the client (F-keys).
    bool g_drawSky       = true;
    bool g_drawDLLand    = true;
    bool g_drawDLStatics = true;
    bool g_drawWater     = true;
    bool g_drawReflect   = true;
    uint32_t  g_debugMode = 0;         // F12 debug view: 0=normal, 1=depth, 2=scatter (written to FrameData.debugParams.x)

    // ---- Dev overlay (Forge IUI) state ----
    bool          g_uiInited  = false;
    bool          g_uiVisible = true;          // F9 (client-forwarded) toggles; default on
    uint32_t      g_uiFontId  = 0;
    UIComponent*  g_uiPanel   = nullptr;
    double        g_uiLastMs  = 0.0;           // for the per-frame ImGui delta-time
    // Forwarded mouse from the MW client (set in renderScene, pushed to UI.cpp::uiSetExternalInput).
    float         g_inMouseX = 0.0f, g_inMouseY = 0.0f, g_inWheel = 0.0f;
    bool          g_inLBtn = false, g_inRBtn = false, g_inMBtn = false;
    int32_t       g_uiDropdownMode = 0;        // DebugTexturesWidget/Dropdown mirror of g_debugMode
    // Phase 0 live-stats readout: a DynamicTextWidget backed by a fixed char array (Forge pattern,
    // 01_Transformations). drawDevUI bformat()s the current per-frame counts into it each frame.
    unsigned char g_statsBuf[1024] = {};
    bstring       g_statsText = bfromarr(g_statsBuf);
    float4        g_statsColor = { 0.70f, 1.0f, 0.70f, 1.0f };

    // ---- AO knobs (Stage 3): promoted from the hardcoded constants so sliders drive them live.
    // ap[20..23] read these each frame, so a slider move takes effect next frame. aoThickness is
    // the GTAO horizon self-occlusion bias (Stage 4) — was reserved/unused before.
    float g_aoRadius    = 1.5f;    // world-unit search extent (in-game keeper)
    float g_aoFalloff   = 20.0f;   // world-unit distance falloff (in-game keeper)
    float g_aoIntensity = 4.0f;    // occlusion scale (in-game keeper; darker than before)
    float g_aoThickness = 0.4f;    // horizon self-occlusion bias (in-game keeper; kills flat-ground matcap)
    float g_aoBlurPx    = 2.0f;    // bilateral blur spatial sigma in pixels (0 = passthrough)
    float g_aoBlurDepth = 5.0f;    // bilateral blur range sigma in WORLD units (rejects across silhouettes)

    // AO contribution toggles → FrameData.debugParams.w bitmask (bit0 AO, bit1 bent normal, bit2 ambient=white).
    bool  g_aoEnable         = true;    // AO visibility modulates ambient (matches current behaviour)
    bool  g_bentNormalEnable = false;   // use the AO bent normal as the lighting normal (A/B; off = geometric N)
    bool  g_ambientWhite     = false;   // debug: force ambient term to 1.0 so AO darkening is visible (pair w/ Diffuse=0)

    // Dev panel intensity modifiers → FrameData.dbgScales (float index 44..47). All 1.0 = no-op.
    // Scale Forge per-component output only; surfaces Forge doesn't draw are unaffected, so cranking
    // one is a quick way to spot what's still going through MW's own path.
    float g_ambScale     = 1.0f;   // ambient term
    float g_litScale     = 1.0f;   // diffuse (sun + point) term
    float g_albedoScale  = 1.0f;   // albedo (texture) term
    float g_overallScale = 1.0f;   // final output

    // SK2 "ownership tell": the Forge sky takeover is byte-identical to MW's own sky, so there's
    // no way to tell it's live. These tint the Forge sky toward magenta (gFrameData.skyParams,
    // read only by sky.frag) — crank the slider and the sky goes magenta IFF Forge is drawing it
    // (F7 on). Pulse animates it (MW's sky never pulses) for an unmistakable confirmation. 0 = off.
    float g_skyDebugTint = 0.0f;   // 0 = identical to MW (clean A/B); 1 = full magenta
    bool  g_skyTintPulse = false;  // animate the tint so it's obviously Forge-owned

    // WT2 water debug: output one isolated water term instead of the composited surface, so the
    // reflection / refraction contents are directly inspectable. Driven by two panel CHECKBOXES
    // (the proven widget type — AO toggles use them; the dropdown's pData write is unreliable here).
    // Combined into a mode (1 = reflection RT only / raw gReflectColor, 2 = refraction only) routed
    // into the water params (worlds[6] group 3); water.frag branches on it. Both off = normal water.
    bool g_waterReflOnly = false;
    bool g_waterRefrOnly = false;

    // F12 debug-view names; index = debugParams.x. The dropdown writes g_debugMode directly so the
    // overlay selector and the F12 key cycle stay unified.
    const char* const kDebugModeNames[] = { "0 normal", "1 depth", "2 scatter", "3 AO", "4 bent-normal",
                                            "5 albedo", "6 lit", "7 ambient" };

    // Build the dev overlay once. The headless host has no Load/Unload reload split, so font-system
    // + UI-system init AND their pipeline load happen together here, right after pRT exists.
    void initDevUI(Renderer* R, uint32_t width, uint32_t height, uint32_t colorFmt) {
        if (g_uiInited) {
            return;
        }

        LOG::logline(">> [devui] initDevUI begin (%ux%u fmt=%u)", width, height, colorFmt); LOG::flush();

        // Framework-level init the headless host must drive itself (normally WindowsBase does it):
        // platformInitFontSystem sizes the atlas from DPI + builds the FONS context; without it the
        // atlas is 0x0 and initFontSystem crashes. Must precede fntDefineFonts (which needs FONS).
        if (!platformInitFontSystem()) {
            LOG::logline("!! [devui] platformInitFontSystem FAILED"); LOG::flush();
            return;
        }
        if (!platformInitUserInterface()) {
            LOG::logline("!! [devui] platformInitUserInterface FAILED"); LOG::flush();
            return;
        }
        LOG::logline(">> [devui] platformInit{Font,UI} ok"); LOG::flush();

        FontDesc font = {};
        font.pFontName = "MGEDev";
        font.pFontPath = "ComicRelief.ttf";   // RD_FONTS = morrowind64/fonts/
        fntDefineFonts(&font, 1, &g_uiFontId);
        LOG::logline(">> [devui] fntDefineFonts ok (id=%u)", g_uiFontId); LOG::flush();

        FontSystemDesc fontDesc = {};
        fontDesc.pRenderer = R;
        if (!initFontSystem(&fontDesc)) {
            LOG::logline("!! [devui] initFontSystem FAILED (font missing?)"); LOG::flush();
            return;
        }
        LOG::logline(">> [devui] initFontSystem ok"); LOG::flush();

        UserInterfaceDesc uiDesc = {};
        uiDesc.pRenderer = R;
        uiDesc.mEnableRemoteUI = false;   // headless: no remote-UI socket
        initUserInterface(&uiDesc);
        LOG::logline(">> [devui] initUserInterface ok"); LOG::flush();

        FontSystemLoadDesc fontLoad = {};
        fontLoad.mLoadType = RELOAD_TYPE_ALL;
        fontLoad.mColorFormat = colorFmt;
        fontLoad.mWidth = width;
        fontLoad.mHeight = height;
        loadFontSystem(&fontLoad);
        LOG::logline(">> [devui] loadFontSystem ok"); LOG::flush();

        UserInterfaceLoadDesc uiLoad = {};
        uiLoad.mLoadType = RELOAD_TYPE_ALL;
        uiLoad.mColorFormat = colorFmt;
        uiLoad.mWidth = width;
        uiLoad.mHeight = height;
        loadUserInterface(&uiLoad);
        LOG::logline(">> [devui] loadUserInterface ok"); LOG::flush();

        UIComponentDesc cd = {};
        cd.mStartPosition = vec2(16.0f, 16.0f);
        cd.mStartSize = vec2(360.0f, 420.0f);
        cd.mFontID = g_uiFontId;
        uiAddComponent("MGE Dev", &cd, &g_uiPanel);

        LabelWidget lbl = {};
        uiAddComponentWidget(g_uiPanel, "Forge dev overlay (F9 toggles)", &lbl, WIDGET_TYPE_LABEL);

        DropdownWidget dd = {};
        dd.pData = &g_debugMode;
        dd.pNames = kDebugModeNames;
        dd.mCount = (uint32_t)(sizeof(kDebugModeNames) / sizeof(kDebugModeNames[0]));
        uiAddComponentWidget(g_uiPanel, "Fullscreen buffer", &dd, WIDGET_TYPE_DROPDOWN);

        SliderFloatWidget sR = {}; sR.pData = &g_aoRadius;    sR.mMin = 1.0f;  sR.mMax = 64.0f;  sR.mStep = 0.5f;
        uiAddComponentWidget(g_uiPanel, "AO radius (world)", &sR, WIDGET_TYPE_SLIDER_FLOAT);
        SliderFloatWidget sF = {}; sF.pData = &g_aoFalloff;   sF.mMin = 1.0f;  sF.mMax = 200.0f; sF.mStep = 1.0f;
        uiAddComponentWidget(g_uiPanel, "AO falloff (world)", &sF, WIDGET_TYPE_SLIDER_FLOAT);
        SliderFloatWidget sI = {}; sI.pData = &g_aoIntensity; sI.mMin = 0.0f;  sI.mMax = 8.0f;   sI.mStep = 0.05f;
        uiAddComponentWidget(g_uiPanel, "AO intensity", &sI, WIDGET_TYPE_SLIDER_FLOAT);
        SliderFloatWidget sT = {}; sT.pData = &g_aoThickness; sT.mMin = 0.0f;  sT.mMax = 0.5f;   sT.mStep = 0.005f;
        uiAddComponentWidget(g_uiPanel, "AO horizon bias", &sT, WIDGET_TYPE_SLIDER_FLOAT);
        SliderFloatWidget sBl = {}; sBl.pData = &g_aoBlurPx;    sBl.mMin = 0.0f; sBl.mMax = 4.0f;   sBl.mStep = 0.1f;
        uiAddComponentWidget(g_uiPanel, "AO blur spatial (px)", &sBl, WIDGET_TYPE_SLIDER_FLOAT);
        SliderFloatWidget sBd = {}; sBd.pData = &g_aoBlurDepth; sBd.mMin = 1.0f; sBd.mMax = 256.0f; sBd.mStep = 1.0f;
        uiAddComponentWidget(g_uiPanel, "AO blur range (world)", &sBd, WIDGET_TYPE_SLIDER_FLOAT);

        // AO contribution toggles.
        CheckboxWidget cAO = {}; cAO.pData = &g_aoEnable;
        uiAddComponentWidget(g_uiPanel, "AO enable", &cAO, WIDGET_TYPE_CHECKBOX);
        CheckboxWidget cBN = {}; cBN.pData = &g_bentNormalEnable;
        uiAddComponentWidget(g_uiPanel, "Bent normal enable", &cBN, WIDGET_TYPE_CHECKBOX);
        CheckboxWidget cAW = {}; cAW.pData = &g_ambientWhite;
        uiAddComponentWidget(g_uiPanel, "Ambient = white (debug)", &cAW, WIDGET_TYPE_CHECKBOX);

        // Intensity modifiers (dbgScales) — crank one to see which surfaces respond (= Forge-drawn).
        SliderFloatWidget sAmb = {}; sAmb.pData = &g_ambScale;     sAmb.mMin = 0.0f; sAmb.mMax = 4.0f; sAmb.mStep = 0.02f;
        uiAddComponentWidget(g_uiPanel, "Ambient intensity", &sAmb, WIDGET_TYPE_SLIDER_FLOAT);
        SliderFloatWidget sLit = {}; sLit.pData = &g_litScale;     sLit.mMin = 0.0f; sLit.mMax = 4.0f; sLit.mStep = 0.02f;
        uiAddComponentWidget(g_uiPanel, "Diffuse intensity", &sLit, WIDGET_TYPE_SLIDER_FLOAT);
        SliderFloatWidget sAlb = {}; sAlb.pData = &g_albedoScale;  sAlb.mMin = 0.0f; sAlb.mMax = 4.0f; sAlb.mStep = 0.02f;
        uiAddComponentWidget(g_uiPanel, "Albedo intensity", &sAlb, WIDGET_TYPE_SLIDER_FLOAT);
        SliderFloatWidget sOvr = {}; sOvr.pData = &g_overallScale; sOvr.mMin = 0.0f; sOvr.mMax = 4.0f; sOvr.mStep = 0.02f;
        uiAddComponentWidget(g_uiPanel, "Overall intensity", &sOvr, WIDGET_TYPE_SLIDER_FLOAT);

        // SK2 sky-takeover "ownership tell". The Forge sky is byte-identical to MW's, so crank this
        // (F7 on) to tint the Forge sky magenta — it only moves if Forge is drawing the sky. Pulse
        // animates it for an unmistakable confirmation. Leave at 0 for a clean vanilla-vs-Forge A/B.
        LabelWidget skyLbl = {};
        uiAddComponentWidget(g_uiPanel, "-- Sky takeover (F7) --", &skyLbl, WIDGET_TYPE_LABEL);
        SliderFloatWidget sSky = {}; sSky.pData = &g_skyDebugTint; sSky.mMin = 0.0f; sSky.mMax = 1.0f; sSky.mStep = 0.02f;
        uiAddComponentWidget(g_uiPanel, "Sky tint (Forge tell)", &sSky, WIDGET_TYPE_SLIDER_FLOAT);
        CheckboxWidget cSkyP = {}; cSkyP.pData = &g_skyTintPulse;
        uiAddComponentWidget(g_uiPanel, "Sky tint pulse", &cSkyP, WIDGET_TYPE_CHECKBOX);

        // WT2 water debug: isolate the reflection / refraction inputs so they can be inspected
        // directly (the composited surface can hide a black/empty reflection RT). 0 = normal water.
        LabelWidget watLbl = {};
        uiAddComponentWidget(g_uiPanel, "-- Water takeover (F7) --", &watLbl, WIDGET_TYPE_LABEL);
        CheckboxWidget cWRefl = {}; cWRefl.pData = &g_waterReflOnly;
        uiAddComponentWidget(g_uiPanel, "Water: reflection only", &cWRefl, WIDGET_TYPE_CHECKBOX);
        CheckboxWidget cWRefr = {}; cWRefr.pData = &g_waterRefrOnly;
        uiAddComponentWidget(g_uiPanel, "Water: refraction only", &cWRefr, WIDGET_TYPE_CHECKBOX);

        // -- Phase 0: per-subsystem draw toggles (off → MW's version shows through the composite) --
        LabelWidget subLbl = {};
        uiAddComponentWidget(g_uiPanel, "-- Forge subsystems (off = show MW) --", &subLbl, WIDGET_TYPE_LABEL);
        CheckboxWidget cDSky = {}; cDSky.pData = &g_drawSky;
        uiAddComponentWidget(g_uiPanel, "Draw: sky", &cDSky, WIDGET_TYPE_CHECKBOX);
        CheckboxWidget cDLand = {}; cDLand.pData = &g_drawDLLand;
        uiAddComponentWidget(g_uiPanel, "Draw: distant land", &cDLand, WIDGET_TYPE_CHECKBOX);
        CheckboxWidget cDStat = {}; cDStat.pData = &g_drawDLStatics;
        uiAddComponentWidget(g_uiPanel, "Draw: distant statics", &cDStat, WIDGET_TYPE_CHECKBOX);
        CheckboxWidget cDWat = {}; cDWat.pData = &g_drawWater;
        uiAddComponentWidget(g_uiPanel, "Draw: water", &cDWat, WIDGET_TYPE_CHECKBOX);
        CheckboxWidget cDRef = {}; cDRef.pData = &g_drawReflect;
        uiAddComponentWidget(g_uiPanel, "Draw: reflection", &cDRef, WIDGET_TYPE_CHECKBOX);

        // -- Phase 0: live per-frame stats (updated each frame in drawDevUI) --
        LabelWidget statLbl = {};
        uiAddComponentWidget(g_uiPanel, "-- Stats (this frame) --", &statLbl, WIDGET_TYPE_LABEL);
        DynamicTextWidget statsW = {};
        statsW.pText = &g_statsText;
        statsW.pColor = &g_statsColor;
        uiAddComponentWidget(g_uiPanel, "", &statsW, WIDGET_TYPE_DYNAMIC_TEXT);

        g_uiInited = true;
        LOG::logline(">> [devui] ready (%ux%u fmt=%u)", width, height, colorFmt); LOG::flush();
    }

    // Per-frame: push forwarded input, build the ImGui frame, and draw it into pRT. Called from
    // renderScene with pRT in RENDER_TARGET state (caller restores COMMON for the DXVK handoff).
    void drawDevUI() {
        if (!g_uiInited) {
            return;
        }

        // Attach the buffer-viewer once the AO targets exist (they're built lazily on first
        // renderScene, after initDevUI). The widget clones a pointer to this persistent array.
        static const Texture* s_debugTex[2] = {};
        static bool s_texWidgetAdded = false;
        if (!s_texWidgetAdded && g_live.pAO && g_live.pLinearDepth) {
            s_debugTex[0] = g_live.pAO;
            s_debugTex[1] = g_live.pLinearDepth;
            DebugTexturesWidget dbg = {};
            dbg.pTextures = s_debugTex;
            dbg.mTexturesCount = 2;
            dbg.mTextureDisplaySize = float2(320.0f, 180.0f);
            uiAddComponentWidget(g_uiPanel, "AO  |  LinearDepth", &dbg, WIDGET_TYPE_DEBUG_TEXTURES);
            s_texWidgetAdded = true;
        }

        // Phase 0: refresh the live-stats text from this frame's counters (Forge bformat pattern).
        bformat(&g_statsText,
                "near opaque %u | skinned %u | multimap %u\n"
                "sky %u | reflect-sky %u | lights %u\n"
                "DL land %u | DL statics %u inst / %u subsets\n"
                "water levels %u\n"
                "host %.2f ms = setup %.2f + cull %.2f + rec %.2f + gpu %.2f + post %.2f\n"
                "gpu: prepass %.2f gtao %.2f reflect %.2f color %.2f water %.2f resolve %.2f",
                g_lastDrawn, g_lastSkinnedDrawn, g_lastMultiMapDrawn,
                g_lastSkyDrawn, g_lastReflSkyDrawn, g_lastLightCount,
                g_liveLastLand, g_liveLastInst, g_liveLastSubsets,
                g_lastWaterLevels,
                g_lastTotalMs, g_lastSetupMs, g_lastCullMs, g_lastRecMs, g_lastGpuMs, g_lastPostMs,
                g_lastGpuPhaseMs[kGpuPhasePrepass], g_lastGpuPhaseMs[kGpuPhaseGtao],
                g_lastGpuPhaseMs[kGpuPhaseReflect], g_lastGpuPhaseMs[kGpuPhaseColor],
                g_lastGpuPhaseMs[kGpuPhaseWater], g_lastGpuPhaseMs[kGpuPhaseResolve]);

        uiSetExternalInput(g_inMouseX, g_inMouseY, g_inWheel, g_inLBtn, g_inRBtn, g_inMBtn, g_uiVisible);
        uiSetComponentActive(g_uiPanel, g_uiVisible);

        const double now = hostNowMs();
        const float  dt  = (g_uiLastMs > 0.0) ? (float)((now - g_uiLastMs) / 1000.0) : 0.016f;
        g_uiLastMs = now;
        platformUpdateUserInterface(dt);

        BindRenderTargetsDesc bind = {};
        bind.mRenderTargetCount = 1;
        bind.mRenderTargets[0] = { g_live.pRT, LOAD_ACTION_LOAD };
        cmdBindRenderTargets(g_live.pCmd, &bind);
        cmdSetViewport(g_live.pCmd, 0.0f, 0.0f, (float)g_live.width, (float)g_live.height, 0.0f, 1.0f);
        cmdSetScissor(g_live.pCmd, 0, 0, g_live.width, g_live.height);
        cmdDrawUserInterface(g_live.pCmd);
        cmdBindRenderTargets(g_live.pCmd, nullptr);
    }

    void exitDevUI() {
        if (!g_uiInited) {
            return;
        }
        unloadFontSystem(RELOAD_TYPE_ALL);
        unloadUserInterface(RELOAD_TYPE_ALL);
        exitFontSystem();
        exitUserInterface();
        // Mirror WindowsBase::exitBaseSubsystems order (UI before fonts) for the framework-level teardown.
        platformExitUserInterface();
        platformExitFontSystem();
        g_uiInited = false;
        g_uiPanel  = nullptr;
    }

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

        // Dev overlay: Forge IUI rendered into pRT each frame. UI is single-sample and matches
        // pRT's B8G8R8A8 (no MSAA UI pipeline). Failure here is non-fatal — the scene still renders.
        initDevUI(R, width, height, (uint32_t)g_live.pRT->mFormat);

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

    void checkShaderHotReload();   // defined below; polls the gtao dxil mtime for auto hot-reload

    // Phase 1a/1b LIVE distant land — defined below (with the DL globals), forward-declared so
    // renderScene (earlier in the file) can drive them. dlSetFrameEye/dlLogHeartbeat wrap the DL
    // global state renderScene touches so those globals can stay next to their definitions.
    void dlSetFrameEye(float x, float y, float z, bool exterior);
    void dlLiveCullAndBuild(Renderer* R, const float* rzViewProj);
    void dlLiveRecord();
    void dlLogHeartbeat();
    void dlLogGpuSlow(double gpuMs, double recMs, unsigned drawn);   // per-frame GPU spike (reads DL counts)

    bool renderScene(const float* viewProj, const float* lighting, const void* drawBlob,
                     unsigned drawCount, unsigned drawBytes,
                     const void* skinnedBlob, unsigned skinnedCount, unsigned skinnedBytes,
                     const void* multiMapBlob, unsigned multiMapCount, unsigned multiMapBytes,
                     const void* lightBlob, unsigned lightCount, unsigned lightBytes,
                     const void* skyBlob, unsigned skyCount, unsigned skyBytes,
                     const float* waterParams, unsigned waterEnabled) {
        if (!g_live.pRenderer) {
            return false;
        }
        Renderer* R = g_live.pRenderer;
        const double tEntry = hostNowMs();   // host-frame split: start of renderScene
        checkShaderHotReload();   // auto-reload compute pipelines if a recompiled dxil landed on disk

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
            // Phase 1a/1b: lighting[24..27] = realEye.xyz + isExterior (appended by the client; the
            // scene-probe passes 0s). The near scene is camera-relative (eyePos=0); resident DL is in
            // ABSOLUTE coords, so the live DL cull/build shifts it by -realEye. lodEye -> gFrameData[56..59].
            dlSetFrameEye(lighting[24], lighting[25], lighting[26], lighting[27] != 0.0f);
        }
        // F12 debug view: debugParams.x at float index 40 (160B = viewProj 16f + 6×float4 24f).
        // 0=normal (frag is byte-for-byte unchanged), 1=depth world-distance grayscale,
        // 3=AO, 4=bent normal (both sample gAO at SV_Position * debugParams.yz = invScreen).
        {
            float* dp = (float*)g_live.pFrameCbv->pCpuMappedAddress;
            dp[40] = (float)g_debugMode;
            dp[41] = 1.0f / (float)g_live.width;    // invScreen.x (F12 AO/bent-normal gAO sample)
            dp[42] = 1.0f / (float)g_live.height;   // invScreen.y
            dp[43] = (float)((g_aoEnable ? 1u : 0u) | (g_bentNormalEnable ? 2u : 0u)
                           | (g_ambientWhite ? 4u : 0u)); // AO toggles
            // dbgScales (float index 44..47): dev panel intensity modifiers.
            dp[44] = g_ambScale; dp[45] = g_litScale; dp[46] = g_albedoScale; dp[47] = g_overallScale;
            // skyParams (float index 60..63): SK2 "ownership tell" — sky.frag tints the Forge sky
            // toward magenta by dp[60], optionally pulsing (dp[62]) over host time dp[61]. 0 = no-op.
            dp[60] = g_skyDebugTint;
            dp[61] = (float)(hostNowMs() * 0.001);
            dp[62] = g_skyTintPulse ? 1.0f : 0.0f;
            dp[63] = 0.0f;
        }

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

        // Phase 1a/1b LIVE distant land: lazy resident load + per-frame frustum/tier cull + ring fill
        // + lazy texture load. MUST run here — BEFORE beginCmd — because it can create GPU resources
        // and load textures (addResource/updateDescriptorSet), which can't happen mid-command-buffer.
        // The recorded draws (dlLiveRecord) go in after the near colour pass. rzViewProj is the
        // relative, reverse-Z, extended-far matrix already in gFrameData.
        const double tCull0 = hostNowMs();   // end of per-draw setup, start of the DL cull
        dlLiveCullAndBuild(R, rzViewProj);
        const double tCull1 = hostNowMs();   // end of the DL cull (+ lazy resource/texture load)

        resetCmdPool(R, g_live.pCmdPool);
        beginCmd(g_live.pCmd);
        const double tRec0 = hostNowMs();   // start of CPU command recording
        g_lastSetupMs = tCull0 - tEntry;    // hot-reload check + cbuffers + per-draw memcpy loop
        g_lastCullMs  = tCull1 - tCull0;    // dlLiveCullAndBuild (CPU cull + ring fill + lazy loads)

        // GPU per-phase timestamps: wrap each render phase with a begin/end query. cmdBeginQuery
        // writes a GPU timestamp at phase start, cmdEndQuery at phase end; resolved before endCmd,
        // read back after the fence (gpuMs). Empty/gated phases (reflection/water off) measure ~0.
        auto gpuPhaseBegin = [&](uint32_t i) {
            if (g_live.pGpuQueryPool) { QueryDesc q = {}; q.mIndex = i; cmdBeginQuery(g_live.pCmd, g_live.pGpuQueryPool, &q); }
        };
        auto gpuPhaseEnd = [&](uint32_t i) {
            if (g_live.pGpuQueryPool) { QueryDesc q = {}; q.mIndex = i; cmdEndQuery(g_live.pCmd, g_live.pGpuQueryPool, &q); }
        };

        // ===================== Stage B (B2): GPU statics cull — COUNT-only validation ===========
        // Independent compute pass at the very start of recording (no render target bound). Resets
        // the survivor counter from a zero upload buffer, dispatches cull.comp (one thread/instance,
        // atomic-adds numSubsets on survival), and copies the count into a readback buffer for the
        // post-fence compare against g_liveLastInst. Raw D3D12 CopyBufferRegion (Forge has no buffer->
        // buffer copy); Forge BufferBarriers keep the DEFAULT-heap counter's state tracked. No draws.
        if (g_live.pCullPipeline && g_live.cullInstCount) {
            ID3D12GraphicsCommandList* cl = g_live.pCmd->mDx.pCmdList;
            BufferBarrier bb = {};
            bb.pBuffer = g_live.pCullCountBuf;
            // reset: UAV -> COPY_DEST, copy zeros in, COPY_DEST -> UAV
            bb.mCurrentState = RESOURCE_STATE_UNORDERED_ACCESS; bb.mNewState = RESOURCE_STATE_COPY_DEST;
            cmdResourceBarrier(g_live.pCmd, 1, &bb, 0, nullptr, 0, nullptr);
            cl->CopyBufferRegion(g_live.pCullCountBuf->mDx.pResource, 0,
                                 g_live.pCullCountZero->mDx.pResource, 0, sizeof(uint32_t));
            bb.mCurrentState = RESOURCE_STATE_COPY_DEST; bb.mNewState = RESOURCE_STATE_UNORDERED_ACCESS;
            cmdResourceBarrier(g_live.pCmd, 1, &bb, 0, nullptr, 0, nullptr);
            // dispatch the cull (one thread per resident instance)
            cmdBeginDebugMarker(g_live.pCmd, 0.9f, 0.9f, 0.2f, "CULL (count survivors -> validate)");
            cmdBindPipeline(g_live.pCmd, g_live.pCullPipeline);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pCullSet);
            cmdDispatch(g_live.pCmd, (g_live.cullInstCount + 63u) / 64u, 1, 1);
            cmdEndDebugMarker(g_live.pCmd);
            // readback: UAV -> COPY_SOURCE, copy to readback, COPY_SOURCE -> UAV
            bb.mCurrentState = RESOURCE_STATE_UNORDERED_ACCESS; bb.mNewState = RESOURCE_STATE_COPY_SOURCE;
            cmdResourceBarrier(g_live.pCmd, 1, &bb, 0, nullptr, 0, nullptr);
            cl->CopyBufferRegion(g_live.pCullCountReadback->mDx.pResource, 0,
                                 g_live.pCullCountBuf->mDx.pResource, 0, sizeof(uint32_t));
            bb.mCurrentState = RESOURCE_STATE_COPY_SOURCE; bb.mNewState = RESOURCE_STATE_UNORDERED_ACCESS;
            cmdResourceBarrier(g_live.pCmd, 1, &bb, 0, nullptr, 0, nullptr);
        }

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
        gpuPhaseBegin(kGpuPhasePrepass);
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

        gpuPhaseEnd(kGpuPhasePrepass);
        gpuPhaseBegin(kGpuPhaseGtao);
        // ===================== TIER 2: LINEARIZE + GTAO COMPUTE =====================
        // First compute work in the host. Sits between the depth-complete prepass and the colour
        // pass: resolve pDepth (sample 0, MSAA-robust) -> single-sample pLinearDepth, then GTAO
        // -> pAO (bent normal + visibility). Tier 2 feeds ONLY the F12 debug views; the colour
        // pass reads pAO unconditionally is Tier 3. pLinearDepth/pAO live in UNORDERED_ACCESS at
        // frame start (created state on frame 0; returned to SHADER_RESOURCE at the end of each
        // frame, so they're flipped back to UAV here on every frame after the first).
        // Tier 2 master toggle: runs the linearize + GTAO dispatches and the pDepth
        // DEPTH_WRITE<->SHADER_RESOURCE ping-pong each frame. With it off, behaviour is pure Tier 1
        // (pDepth stays DEPTH_WRITE; pLinearDepth/pAO stay in their created UAV state, untouched).
        static const bool g_aoComputeEnable = true;
        static bool s_aoDispatchLogged = false;
        if (!s_aoDispatchLogged) {
            std::printf("[forge] AO dispatch GATE: enable=%d linPipe=%p gtaoPipe=%p linSet=%p gBatchSet=%p pAO=%p pLinDepth=%p firstFrame=%d\n",
                        (int)g_aoComputeEnable, (void*)g_live.pLinearizePipeline, (void*)g_live.pGtaoPipeline,
                        (void*)g_live.pLinearizeSet, (void*)g_live.pGtaoBatchSet,
                        (void*)g_live.pAO, (void*)g_live.pLinearDepth, (int)g_live.firstFrame);
            s_aoDispatchLogged = true;
        }
        if (g_aoComputeEnable && g_live.pLinearizePipeline && g_live.pGtaoPipeline) {
            // Build gAOParams: invViewProj (from the SAME rzViewProj geometry used, incl. the
            // half-pixel offset) + screen + knobs + eye. Seeds follow scene-walk (WORLD-unit knobs;
            // may need MW-scale tuning — change here). eye is read back from the frame cbuffer
            // (floats 36..38 = gFrameData.eyePos), which holds the latest value across null-lighting frames.
            float invVP[16];
            if (!invert4x4(rzViewProj, invVP)) {
                for (int i = 0; i < 16; ++i) { invVP[i] = (i % 5 == 0) ? 1.0f : 0.0f; }  // identity guard
            }
            // AO knobs are now dev-overlay sliders (g_ao*); the per-frame upload reads them live.
            const float* fcbv = (const float*)g_live.pFrameCbv->pCpuMappedAddress;
            float* ap = (float*)g_live.pAOParamsCbv->pCpuMappedAddress;
            std::memcpy(ap, invVP, 16 * sizeof(float));
            ap[16] = (float)g_live.width;  ap[17] = (float)g_live.height;
            ap[18] = 1.0f / (float)g_live.width; ap[19] = 1.0f / (float)g_live.height;
            ap[20] = g_aoRadius; ap[21] = g_aoFalloff; ap[22] = g_aoIntensity; ap[23] = g_aoThickness;
            ap[24] = fcbv[36]; ap[25] = fcbv[37]; ap[26] = fcbv[38]; ap[27] = 0.0f;
            ap[28] = g_aoBlurPx; ap[29] = g_aoBlurDepth; ap[30] = 0.0f; ap[31] = 0.0f;  // gBlurParams.blurParams

            const uint32_t gx = (g_live.width + 7u) / 8u;
            const uint32_t gy = (g_live.height + 7u) / 8u;

            // End the prepass render pass, then pDepth DEPTH_WRITE -> SHADER_RESOURCE (first depth
            // -> SRV transition in the host) and flip pLinearDepth back to UAV (skip on frame 0).
            cmdBindRenderTargets(g_live.pCmd, nullptr);
            {
                RenderTargetBarrier rtb = {};
                rtb.pRenderTarget = g_live.pDepth;
                rtb.mCurrentState = RESOURCE_STATE_DEPTH_WRITE;
                rtb.mNewState = RESOURCE_STATE_SHADER_RESOURCE;
                TextureBarrier tb[1] = {};
                uint32_t nt = 0;
                if (!g_live.firstFrame) {
                    tb[nt].pTexture = g_live.pLinearDepth;
                    tb[nt].mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
                    tb[nt].mNewState = RESOURCE_STATE_UNORDERED_ACCESS;
                    ++nt;
                }
                cmdResourceBarrier(g_live.pCmd, 0, nullptr, nt, tb, 1, &rtb);
            }

            // (1) Linearize/resolve dispatch.
            cmdBeginDebugMarker(g_live.pCmd, 0.2f, 0.6f, 1.0f, "LINEARIZE (writes pLinearDepth)");
            cmdBindPipeline(g_live.pCmd, g_live.pLinearizePipeline);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pLinearizeSet);
            cmdDispatch(g_live.pCmd, gx, gy, 1);
            cmdEndDebugMarker(g_live.pCmd);

            // pLinearDepth UAV -> SRV (GTAO reads it); pAO SRV -> UAV (skip on frame 0).
            {
                TextureBarrier tb[2] = {};
                uint32_t nt = 0;
                tb[nt].pTexture = g_live.pLinearDepth;
                tb[nt].mCurrentState = RESOURCE_STATE_UNORDERED_ACCESS;
                tb[nt].mNewState = RESOURCE_STATE_SHADER_RESOURCE;
                ++nt;
                if (!g_live.firstFrame) {
                    tb[nt].pTexture = g_live.pAO;
                    tb[nt].mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
                    tb[nt].mNewState = RESOURCE_STATE_UNORDERED_ACCESS;
                    ++nt;
                }
                cmdResourceBarrier(g_live.pCmd, 0, nullptr, nt, tb, 0, nullptr);
            }

            // (2) GTAO dispatch (single PerDraw set holds cbuffer + depth SRV + AO UAV).
            cmdBeginDebugMarker(g_live.pCmd, 1.0f, 0.3f, 0.2f, "GTAO (pLinearDepth -> pAO: bent normal + visibility)");
            cmdBindPipeline(g_live.pCmd, g_live.pGtaoPipeline);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pGtaoBatchSet);
            cmdDispatch(g_live.pCmd, gx, gy, 1);
            cmdEndDebugMarker(g_live.pCmd);
            static bool s_gtaoDispatched = false;
            if (!s_gtaoDispatched) { std::printf("[forge] GTAO dispatch ISSUED gx=%u gy=%u (w=%u h=%u)\n", gx, gy, g_live.width, g_live.height); s_gtaoDispatched = true; }

            // pAO UAV -> SRV (blur + F12 debug read it); pAOBlur SRV -> UAV (blur writes it, skip f0);
            // pDepth SRV -> DEPTH_WRITE (colour LOADs it). pLinearDepth STAYS SRV — the blur reads it.
            {
                RenderTargetBarrier rtb = {};
                rtb.pRenderTarget = g_live.pDepth;
                rtb.mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
                rtb.mNewState = RESOURCE_STATE_DEPTH_WRITE;
                TextureBarrier tb[2] = {};
                uint32_t nt = 0;
                tb[nt].pTexture = g_live.pAO;
                tb[nt].mCurrentState = RESOURCE_STATE_UNORDERED_ACCESS;
                tb[nt].mNewState = RESOURCE_STATE_SHADER_RESOURCE;
                ++nt;
                if (!g_live.firstFrame) {
                    tb[nt].pTexture = g_live.pAOBlur;
                    tb[nt].mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
                    tb[nt].mNewState = RESOURCE_STATE_UNORDERED_ACCESS;
                    ++nt;
                }
                cmdResourceBarrier(g_live.pCmd, 0, nullptr, nt, tb, 1, &rtb);
            }

            // (3) Bilateral AO blur: pAO + pLinearDepth (both SRV) -> pAOBlur (UAV). PerFrame set
            // (root distinct from gtao/linearize). Depth-aware, so it denoises without silhouette
            // bleed. The colour frags sample pAOBlur as gAO.
            {
                cmdBeginDebugMarker(g_live.pCmd, 0.6f, 1.0f, 0.4f, "AO BILATERAL BLUR (pAO -> pAOBlur)");
                cmdBindPipeline(g_live.pCmd, g_live.pAOBlurPipeline);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pAOBlurSet);
                cmdDispatch(g_live.pCmd, gx, gy, 1);
                cmdEndDebugMarker(g_live.pCmd);
                // pAOBlur UAV -> SRV for the colour pass.
                TextureBarrier tb = {};
                tb.pTexture = g_live.pAOBlur;
                tb.mCurrentState = RESOURCE_STATE_UNORDERED_ACCESS;
                tb.mNewState = RESOURCE_STATE_SHADER_RESOURCE;
                cmdResourceBarrier(g_live.pCmd, 0, nullptr, 1, &tb, 0, nullptr);
            }
        }

        gpuPhaseEnd(kGpuPhaseGtao);
        gpuPhaseBegin(kGpuPhaseReflect);
        // ===================== WT2: REFLECTION PASS (sky-only, Stage 1) =====================
        // Render the mirrored scene into pReflectColor (1024²) BEFORE the main colour pass, so the
        // water frag can sample it. Stage 1 = SKY ONLY: mirror world geometry about the water plane
        // (mirrorVP = Mirror·viewProj) and draw the sky with the NORMAL camera → it lands in main-
        // screen space. Own frame cbuffer (mirror matrix) + own sky buffers (filled here, decoupled
        // from the main sky pass). Gated by reflectReady (build) + waterEnabled (F7) + a sky list.
        {
            static uint32_t s_reflGateLog = 0;
            if ((s_reflGateLog++ % 120) == 0) {
                LOG::logline(">> [forge][reflect] GATE ready=%d waterEn=%u params=%d skyBlob=%d skyCount=%u skyBytes=%u",
                             (int)g_live.reflectReady, waterEnabled, (int)(waterParams != nullptr),
                             (int)(skyBlob != nullptr), skyCount, skyBytes);
                LOG::flush();
            }
        }
        g_lastReflSkyDrawn = 0;   // Phase 0 panel: 0 unless the reflection pass runs below
        if (g_drawReflect && g_live.reflectReady && waterEnabled && waterParams && skyBlob && skyCount && skyBytes) {
            const float* fcbvR = (const float*)g_live.pFrameCbv->pCpuMappedAddress;
            const float eyeAbsZ = fcbvR[58];                       // lodEye.z (absolute camera)
            const float waterLevelAbs = waterParams[0];
            const float dRel = (waterLevelAbs - 1.0f) - eyeAbsZ;   // water plane, camera-relative (logged)

            // STAGE 1 (sky): mirror about the CAMERA's horizontal plane (z = 0 camera-relative), NOT
            // the water plane. The sky is an infinite, camera-attached dome; reflecting a FINITE dome
            // about the water plane (~|dRel| below) displaces it ~2·|dRel| below the camera → when the
            // dome radius is smaller than that the camera leaves the dome and it only fills the lower
            // screen ("ends at 5m then black", horizon mismatched), and billboards (the sun disc) are
            // viewed obliquely (squashed). Mirroring about z = 0 keeps the dome centered on the camera
            // (fills the screen at any height) and flips its pitch = the reflected sky directions.
            // (Stage 2 static distant land will mirror about the water plane dRel — different plane.)
            float M[16] = { 1,0,0,0,  0,1,0,0,  0,0,-1,0,  0,0,0,1 };
            float mirrorVP[16];
            mul4x4(M, viewProj, mirrorVP);   // mirror world geom, then the ORIGINAL (pre-reverse-Z) camera
            // Same reverse-Z + half-pixel edits the main path applies to viewProj (so the reflection
            // RT is in main-screen space → the water frag samples it at its own screen UV).
            mirrorVP[2]  = mirrorVP[3]  - mirrorVP[2];
            mirrorVP[6]  = mirrorVP[7]  - mirrorVP[6];
            mirrorVP[10] = mirrorVP[11] - mirrorVP[10];
            mirrorVP[14] = mirrorVP[15] - mirrorVP[14];
            {
                const float dx = -1.0f * (-1.0f / (float)g_live.width);
                const float dy = -1.0f * ( 1.0f / (float)g_live.height);
                mirrorVP[0]  += dx * mirrorVP[3];  mirrorVP[4]  += dx * mirrorVP[7];
                mirrorVP[8]  += dx * mirrorVP[11]; mirrorVP[12] += dx * mirrorVP[15];
                mirrorVP[1]  += dy * mirrorVP[3];  mirrorVP[5]  += dy * mirrorVP[7];
                mirrorVP[9]  += dy * mirrorVP[11]; mirrorVP[13] += dy * mirrorVP[15];
            }
            // Reflect frame cbuffer = a copy of the main frame data (identical lighting/fog/skyParams)
            // with ONLY the viewProj replaced by the mirror matrix.
            std::memcpy(g_live.pReflectFrameCbv->pCpuMappedAddress, fcbvR, 256);
            std::memcpy(g_live.pReflectFrameCbv->pCpuMappedAddress, mirrorVP, 64);

            // pReflectColor SHADER_RESOURCE -> RENDER_TARGET (pReflectDepth stays DEPTH_WRITE).
            {
                RenderTargetBarrier rb = {};
                rb.pRenderTarget = g_live.pReflectColor;
                rb.mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
                rb.mNewState = RESOURCE_STATE_RENDER_TARGET;
                cmdResourceBarrier(g_live.pCmd, 0, nullptr, 0, nullptr, 1, &rb);
            }
            BindRenderTargetsDesc rbind = {};
            rbind.mRenderTargetCount = 1;
            rbind.mRenderTargets[0] = { g_live.pReflectColor, LOAD_ACTION_CLEAR };
            rbind.mDepthStencil = { g_live.pReflectDepth, LOAD_ACTION_CLEAR };
            cmdBindRenderTargets(g_live.pCmd, &rbind);
            cmdSetViewport(g_live.pCmd, 0.0f, 0.0f, (float)kReflectSize, (float)kReflectSize, 0.0f, 1.0f);
            cmdSetScissor(g_live.pCmd, 0, 0, kReflectSize, kReflectSize);

            // Sky draw (mirror): fill the reflect sky buffers + replay the shapes. Cull NONE in the
            // sky PSO makes the mirror's winding flip irrelevant. Same per-shape data as the main sky
            // pass (only the bound frame cbuffer differs).
            const uint32_t haveSkyR = skyBytes / (uint32_t)sizeof(IPC::SkyDrawWire);
            uint32_t nSkyR = (skyCount < haveSkyR) ? skyCount : haveSkyR;
            if (nSkyR > kMaxSkyDraws) { nSkyR = kMaxSkyDraws; }
            const IPC::SkyDrawWire* skyItemsR = (const IPC::SkyDrawWire*)skyBlob;
            cmdBindPipeline(g_live.pCmd, g_live.pSkyPipeline);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSetReflect);   // MIRROR matrix
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerBatchSetReflectSky);
            Pipeline* curReflSky = g_live.pSkyPipeline;
            uint32_t reflSkyDrawn = 0;
            for (uint32_t k = 0; k < nSkyR; ++k) {
                const IPC::SkyDrawWire& it = skyItemsR[k];
                const uint32_t slot = it.slot;
                if (slot >= g_meshHigh || !g_meshes[slot].valid) continue;
                HostMesh& m = g_meshes[slot];
                if (m.skinned || m.multimap) continue;
                const uint32_t idx = reflSkyDrawn;
                Pipeline* want = g_live.pSkyPipeline;
                if (it.srcBlend == kD3DBLEND_SRCALPHA && it.destBlend == kD3DBLEND_ONE) {
                    want = g_live.pSkyPipelineAdd;
                }
                if (want != curReflSky) { cmdBindPipeline(g_live.pCmd, want); curReflSky = want; }

                uint8_t* dst = (uint8_t*)g_live.pReflectSkyWorldsBuf->pCpuMappedAddress;
                std::memcpy(dst + (size_t)idx * 64, it.world, 64);
                // WT2 sun-disc fix: the client faces the sun to the MAIN camera, so after the mirror
                // (M = flip world-z) it faces the REFLECTED camera while we view from the main camera
                // → foreshortened/squashed. Pre-flip the z-component of its 3 basis rows here; M then
                // un-flips the rotation (→ faces the view = round) but still mirrors the translation
                // (→ correct reflected position). Only the flagged sun shape (moons look fine as-is).
                if (it.isSunDisc) {
                    float* w = (float*)(dst + (size_t)idx * 64);
                    w[2]  = -w[2];    // +X basis z
                    w[6]  = -w[6];    // +Y basis z
                    w[10] = -w[10];   // +Z basis z
                }
                uint32_t* inst = (uint32_t*)g_live.pReflectSkyInstanceBuf->pCpuMappedAddress;
                inst[idx * kStaticInstU32 + 0] = idx;
                inst[idx * kStaticInstU32 + 1] = packTexAlpha(it.texIndex, it.alphaRef, it.vColSource);
                float* finst = (float*)inst;
                finst[idx * kStaticInstU32 + 2] = it.matColor[0];
                finst[idx * kStaticInstU32 + 3] = it.matColor[1];
                finst[idx * kStaticInstU32 + 4] = it.matColor[2];
                finst[idx * kStaticInstU32 + 11] = it.matAlpha;

                Buffer*  meshVb     = m.inArena ? g_live.pArenaVB : m.vb;
                Buffer*  meshIb     = m.inArena ? g_live.pArenaIB : m.ib;
                uint32_t firstVertex = m.inArena ? (uint32_t)(m.vbOff / sizeof(IPC::GeomVertexWire)) : 0u;
                uint32_t firstIndex  = m.inArena ? (uint32_t)(m.ibOff / sizeof(uint16_t)) : 0u;
                if (!meshVb || !meshIb) continue;
                Buffer*  vbs[2]     = { meshVb, g_live.pReflectSkyInstanceBuf };
                uint32_t strides[2] = { vStride, iStride };
                cmdBindVertexBuffer(g_live.pCmd, 2, vbs, strides, nullptr);
                cmdBindIndexBuffer(g_live.pCmd, meshIb, INDEX_TYPE_UINT16, 0);
                cmdDrawIndexedInstanced(g_live.pCmd, m.indexCount, firstIndex, 1, firstVertex, idx);
                ++reflSkyDrawn;
            }
            g_lastReflSkyDrawn = reflSkyDrawn;   // Phase 0 panel

            static uint32_t s_reflDrawLog = 0;
            if ((s_reflDrawLog++ % 120) == 0) {
                LOG::logline(">> [forge][reflect] PASS RAN: nSky=%u drawn=%u waterLvl=%.1f eyeZ=%.1f dRel=%.1f mVP[0,5,10]=%.3f,%.3f,%.4f",
                             nSkyR, reflSkyDrawn, waterLevelAbs, eyeAbsZ, dRel,
                             mirrorVP[0], mirrorVP[5], mirrorVP[10]);
                LOG::flush();
            }

            // End the reflect pass; pReflectColor RENDER_TARGET -> SHADER_RESOURCE (water samples it).
            cmdBindRenderTargets(g_live.pCmd, nullptr);
            {
                RenderTargetBarrier rb = {};
                rb.pRenderTarget = g_live.pReflectColor;
                rb.mCurrentState = RESOURCE_STATE_RENDER_TARGET;
                rb.mNewState = RESOURCE_STATE_SHADER_RESOURCE;
                cmdResourceBarrier(g_live.pCmd, 0, nullptr, 0, nullptr, 1, &rb);
            }
        }

        gpuPhaseEnd(kGpuPhaseReflect);
        gpuPhaseBegin(kGpuPhaseColor);
        // ===================== COLOUR PASS (early-Z: CMP_EQUAL, no depth write) =====================
        BindRenderTargetsDesc bind = {};
        bind.mRenderTargetCount = 1;
        bind.mRenderTargets[0] = { colorTarget, LOAD_ACTION_CLEAR };
        bind.mDepthStencil = { g_live.pDepth, LOAD_ACTION_LOAD };   // LOAD the prepass depth (no clear)
        cmdBindRenderTargets(g_live.pCmd, &bind);
        cmdSetViewport(g_live.pCmd, 0.0f, 0.0f, (float)g_live.width, (float)g_live.height, 0.0f, 1.0f);
        cmdSetScissor(g_live.pCmd, 0, 0, g_live.width, g_live.height);

        // --- SK1: sky pass (FIRST in the colour pass, depth off, alpha blend) ----------------
        // Draw the sky shapes (SK1 = the gradient dome) BEFORE the opaque world so the opaque
        // replace-blend draws overwrite them where geometry exists → sky ends up behind. The colour
        // target was just cleared to (0,0,0,0); SRCALPHA/INVSRCALPHA gives premultiplied output for
        // the present-seam composite. Each item writes its world into pSkyWorldsBuf[idx] + its
        // DrawIndex/TexAlpha into pSkyInstanceBuf[idx], drawn with firstInstance=idx. Sky meshes are
        // captured as ordinary GeomVertexWire statics, so they live in the arena (or the dynamic ring
        // after the per-frame re-upload promotes them) — handle both. Capped at kMaxSkyDraws.
        uint32_t skyDrawn = 0;
        if (g_drawSky && skyBlob && skyCount && skyBytes && g_live.pSkyPipeline && g_live.pSkyWorldsBuf) {
            const uint32_t haveSky = skyBytes / (uint32_t)sizeof(IPC::SkyDrawWire);
            uint32_t nSky = (skyCount < haveSky) ? skyCount : haveSky;
            if (nSky > kMaxSkyDraws) { nSky = kMaxSkyDraws; }
            const IPC::SkyDrawWire* skyItems = (const IPC::SkyDrawWire*)skyBlob;

            // Bind a sky pipeline FIRST (establishes the shared default.rootsig the descriptor
            // binds need); the loop switches between the alpha-over / additive PSOs per draw.
            cmdBindPipeline(g_live.pCmd, g_live.pSkyPipeline);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerBatchSetSky);
            Pipeline* curSkyPipe = g_live.pSkyPipeline;

            for (uint32_t k = 0; k < nSky; ++k) {
                const IPC::SkyDrawWire& it = skyItems[k];
                const uint32_t slot = it.slot;
                if (slot >= g_meshHigh || !g_meshes[slot].valid) {
                    continue;   // mesh not uploaded yet
                }
                HostMesh& m = g_meshes[slot];
                if (m.skinned || m.multimap) {
                    continue;   // sky uses the lean GeomVertexWire layout only
                }
                const uint32_t idx = skyDrawn;

                // SK2: pick the blend PSO from the captured (src,dst) pair. SK2 elements
                // (sun/moons/stars) are all standard alpha-over → pSkyPipeline; SRCALPHA/ONE →
                // the additive variant (deferred glare/clouds). Unhandled pairs fall back to
                // alpha-over and log once (surfaces e.g. an unexpected moon-shadow blend). The
                // skyOrder sort groups like blends, so rebinds are rare.
                Pipeline* want = g_live.pSkyPipeline;
                if (it.srcBlend == kD3DBLEND_SRCALPHA && it.destBlend == kD3DBLEND_ONE) {
                    want = g_live.pSkyPipelineAdd;
                } else if (!(it.srcBlend == kD3DBLEND_SRCALPHA && it.destBlend == kD3DBLEND_INVSRCALPHA)) {
                    static bool warnedSkyBlend = false;
                    if (!warnedSkyBlend) {
                        std::printf("[forge][sky] unhandled blend pair src=%u dst=%u -> alpha-over fallback\n",
                                    it.srcBlend, it.destBlend);
                        warnedSkyBlend = true;
                    }
                }
                if (want != curSkyPipe) {
                    cmdBindPipeline(g_live.pCmd, want);
                    curSkyPipe = want;
                }

                uint8_t* dst = (uint8_t*)g_live.pSkyWorldsBuf->pCpuMappedAddress;
                std::memcpy(dst + (size_t)idx * 64, it.world, 64);

                uint32_t* inst = (uint32_t*)g_live.pSkyInstanceBuf->pCpuMappedAddress;
                inst[idx * kStaticInstU32 + 0] = idx;   // DrawIndex → gBatch.worlds[idx]
                inst[idx * kStaticInstU32 + 1] = packTexAlpha(it.texIndex, it.alphaRef, it.vColSource);
                // SK2 FFP modulation: material diffuse rgb in slots [2..4] (TEXCOORD3) and the
                // per-element alpha fade bit-cast into the spare overlay slot [11] (TEXCOORD6) —
                // reuses the opaque vl unchanged. sky.frag does c = tex * base; c.a *= matAlpha.
                float* finst = (float*)inst;
                finst[idx * kStaticInstU32 + 2] = it.matColor[0];
                finst[idx * kStaticInstU32 + 3] = it.matColor[1];
                finst[idx * kStaticInstU32 + 4] = it.matColor[2];
                finst[idx * kStaticInstU32 + 11] = it.matAlpha;

                // Arena (bind-once + offsets) vs dynamic-ring (own VB/IB) source.
                Buffer*  meshVb     = m.inArena ? g_live.pArenaVB : m.vb;
                Buffer*  meshIb     = m.inArena ? g_live.pArenaIB : m.ib;
                uint32_t firstVertex = m.inArena ? (uint32_t)(m.vbOff / sizeof(IPC::GeomVertexWire)) : 0u;
                uint32_t firstIndex  = m.inArena ? (uint32_t)(m.ibOff / sizeof(uint16_t)) : 0u;
                if (!meshVb || !meshIb) {
                    continue;
                }
                Buffer*  vbs[2]     = { meshVb, g_live.pSkyInstanceBuf };
                uint32_t strides[2] = { vStride, iStride };
                cmdBindVertexBuffer(g_live.pCmd, 2, vbs, strides, nullptr);
                cmdBindIndexBuffer(g_live.pCmd, meshIb, INDEX_TYPE_UINT16, 0);
                cmdDrawIndexedInstanced(g_live.pCmd, m.indexCount, firstIndex, 1, firstVertex, idx);
                ++skyDrawn;
            }
        }

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

        // Phase 1a/1b LIVE distant land: draw land + statics into the SAME colorTarget + pDepth as the
        // near scene (still bound here), depth-write reverse-Z GEQUAL so DL is occluded behind the near
        // scene and fills the horizon. Cull/ring-fill already ran before recording (dlLiveCullAndBuild).
        dlLiveRecord();

        gpuPhaseEnd(kGpuPhaseColor);
        gpuPhaseBegin(kGpuPhaseWater);
        // ===================== WT1: FORGE WATER SURFACE =====================
        // Drawn LAST (after near scene + DL) so the whole opaque frame is its refraction/scene-depth
        // source. Sequence: end the colour pass → copy colorTarget into pRefractColor (refraction src)
        // → re-bind the colour pass (LOAD/LOAD) → upload the per-LOD-level worlds + packed params +
        // invVP → draw one indexed-instanced call per clipmap level (DrawIndex=level). Depth GEQUAL +
        // write so water occludes / is occluded correctly. Gated by waterReady (build) + waterEnabled (F7).
        g_lastWaterLevels = 0;   // Phase 0 panel: 0 unless the water pass runs below
        if (g_live.waterReady && waterEnabled && g_drawWater) {
            // (1) End the colour pass; copy colorTarget → pRefractColor. CopyResource for 1x; for MSAA
            // the colour is multisampled → ResolveSubresource into the single-sample refraction copy.
            cmdBindRenderTargets(g_live.pCmd, nullptr);
            ID3D12GraphicsCommandList* cl = g_live.pCmd->mDx.pCmdList;
            ID3D12Resource* colRes  = colorTarget->pTexture->mDx.pResource;
            ID3D12Resource* refrRes = g_live.pRefractColor->mDx.pResource;
            const bool msaa = (g_live.sampleCount > 1);
            {
                D3D12_RESOURCE_BARRIER pre[2] = {};
                pre[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                pre[0].Transition.pResource = colRes;
                pre[0].Transition.Subresource = 0;
                pre[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
                pre[0].Transition.StateAfter  = msaa ? D3D12_RESOURCE_STATE_RESOLVE_SOURCE
                                                     : D3D12_RESOURCE_STATE_COPY_SOURCE;
                pre[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                pre[1].Transition.pResource = refrRes;
                pre[1].Transition.Subresource = 0;
                pre[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                                              | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                pre[1].Transition.StateAfter  = msaa ? D3D12_RESOURCE_STATE_RESOLVE_DEST
                                                     : D3D12_RESOURCE_STATE_COPY_DEST;
                cl->ResourceBarrier(2, pre);
            }
            if (msaa) { cl->ResolveSubresource(refrRes, 0, colRes, 0, DXGI_FORMAT_B8G8R8A8_UNORM); }
            else      { cl->CopyResource(refrRes, colRes); }
            {
                D3D12_RESOURCE_BARRIER post[2] = {};
                post[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                post[0].Transition.pResource = colRes;
                post[0].Transition.Subresource = 0;
                post[0].Transition.StateBefore = msaa ? D3D12_RESOURCE_STATE_RESOLVE_SOURCE
                                                      : D3D12_RESOURCE_STATE_COPY_SOURCE;
                post[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
                post[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                post[1].Transition.pResource = refrRes;
                post[1].Transition.Subresource = 0;
                post[1].Transition.StateBefore = msaa ? D3D12_RESOURCE_STATE_RESOLVE_DEST
                                                      : D3D12_RESOURCE_STATE_COPY_DEST;
                post[1].Transition.StateAfter  = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                                               | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                cl->ResourceBarrier(2, post);
            }

            // (2) Build worlds[0..5] (camera-relative, eye-snapped per level), worlds[6]=packed params,
            // worlds[7]=invVP. The absolute eye for snapping = gFrameData.lodEye (fcbv float 56..58),
            // valid across null-lighting frames (same source the AO block uses for the eye).
            const float* fcbv = (const float*)g_live.pFrameCbv->pCpuMappedAddress;
            const float eyeAbsX = fcbv[56], eyeAbsY = fcbv[57], eyeAbsZ = fcbv[58];
            const float waterLevelAbs = waterParams ? waterParams[0] : 0.0f;
            const bool  underwater    = waterParams && waterParams[7] > 0.5f;
            const float waterZ = waterLevelAbs + (underwater ? 5.0f : -5.0f);

            uint8_t* wbuf = (uint8_t*)g_live.pWaterWorldsBuf->pCpuMappedAddress;
            for (uint32_t k = 0; k < kWaterLevels; ++k) {
                const float cell = g_waterLevels[k].cellSize;
                const float snap = 2.0f * cell;
                const float originX = std::floor(eyeAbsX / snap) * snap;
                const float originY = std::floor(eyeAbsY / snap) * snap;
                // Camera-relative, D3DX row-major: scale(cell,cell,1) · translate(origin-eye, waterZ-eye.z).
                float m[16] = { cell, 0, 0, 0,  0, cell, 0, 0,  0, 0, 1, 0,
                                originX - eyeAbsX, originY - eyeAbsY, waterZ - eyeAbsZ, 1 };
                std::memcpy(wbuf + (size_t)k * 64, m, 64);
            }
            // worlds[6] = packed params (4 float4 groups, row-major; the frag transposes to read).
            {
                float p[16] = {};
                p[0] = waterLevelAbs - eyeAbsZ;                 // waterLevelRel (water-cut feature)
                p[1] = waterParams ? waterParams[1] : 0.013f;  // windFactor
                p[2] = waterParams ? waterParams[2] : 24.0f;   // shoreDepthBias
                // Animation time for the normal volume's W axis. MUST be wrapped on the HOST (double)
                // before the float cast: hostNowMs() is steady_clock-since-BOOT, so seconds is ~1e5-1e6
                // → float32 ULP ~0.02-0.13s quantizes t into coarse steps ("low frame rate" normals).
                // The frag does t = 0.4*time and the volume W is REPEAT, so wrapping at period 2.5 keeps
                // t in [0,1) at full precision AND wraps seamlessly (0.4*2.5 = exactly one W cycle).
                p[3] = (float)std::fmod(hostNowMs() * 0.001, 2.5);
                p[4] = waterParams ? waterParams[3] : 0.0f;    // depthBaseColor.r
                p[5] = waterParams ? waterParams[4] : 0.0f;    // .g
                p[6] = waterParams ? waterParams[5] : 0.0f;    // .b
                p[7] = underwater ? 1.0f : 0.0f;
                p[8]  = waterParams ? waterParams[8]  : 0.0f;  // camFwd.x
                p[9]  = waterParams ? waterParams[9]  : 0.0f;  // camFwd.y
                p[10] = waterParams ? waterParams[10] : 1.0f;  // camFwd.z
                p[11] = waterParams ? waterParams[6] : 0.0f;   // nearViewRange
                p[12] = g_waterReflOnly ? 1.0f : (g_waterRefrOnly ? 2.0f : 0.0f);  // water debug view
                std::memcpy(wbuf + 6 * 64, p, 64);
            }
            // worlds[7] = invViewProj of the SAME rzViewProj (incl. half-pixel) the GTAO block inverts;
            // uploaded RAW (the frag does mul(invVP, ndc), identical convention to gtao.comp).
            {
                float invVP[16];
                if (!invert4x4(rzViewProj, invVP)) {
                    for (int i = 0; i < 16; ++i) { invVP[i] = (i % 5 == 0) ? 1.0f : 0.0f; }
                }
                std::memcpy(wbuf + 7 * 64, invVP, 64);
            }

            // (3) Re-bind the colour pass (LOAD colour, LOAD depth — pDepth is still DEPTH_WRITE) and
            // draw each clipmap level. Trim variant = eye parity (port of renderwater.cpp:947).
            BindRenderTargetsDesc wbind = {};
            wbind.mRenderTargetCount = 1;
            wbind.mRenderTargets[0] = { colorTarget, LOAD_ACTION_LOAD };
            wbind.mDepthStencil = { g_live.pDepth, LOAD_ACTION_LOAD };
            cmdBindRenderTargets(g_live.pCmd, &wbind);
            cmdSetViewport(g_live.pCmd, 0.0f, 0.0f, (float)g_live.width, (float)g_live.height, 0.0f, 1.0f);
            cmdSetScissor(g_live.pCmd, 0, 0, g_live.width, g_live.height);
            cmdBindPipeline(g_live.pCmd, g_live.pWaterPipeline);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);    // gFrameData + gAO + 4 water SRVs
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerBatchSetWater);
            Buffer*  wvbs[2]     = { g_live.pWaterVB, g_live.pWaterInstanceBuf };
            uint32_t wstrides[2] = { 12, (uint32_t)sizeof(uint32_t) };
            cmdBindVertexBuffer(g_live.pCmd, 2, wvbs, wstrides, nullptr);
            cmdBindIndexBuffer(g_live.pCmd, g_live.pWaterIB, INDEX_TYPE_UINT16, 0);
            for (uint32_t k = 0; k < kWaterLevels; ++k) {
                const WaterLodLevelHost& lvl = g_waterLevels[k];
                uint32_t variant = 0;
                if (lvl.numVariants > 1) {
                    const int ex = (int)(((long long)std::floor(eyeAbsX / lvl.cellSize)) & 1);
                    const int ey = (int)(((long long)std::floor(eyeAbsY / lvl.cellSize)) & 1);
                    variant = (uint32_t)(ey * 2 + ex);
                }
                if (!lvl.triCount[variant]) continue;
                // firstInstance = k → the instance VB's DrawIndex[k] = k → gBatch.worlds[k].
                // BaseVertexLocation = 0: the indices ALREADY include each level's vertBase (the vidx
                // lambda bakes it in, like MGE's DrawIndexedPrimitive with BaseVertexIndex=0). Passing
                // vertBase here too DOUBLE-offset levels 1..5 → only level 0 drew (~1 cell of coverage).
                cmdDrawIndexedInstanced(g_live.pCmd, lvl.triCount[variant] * 3,
                                        lvl.ibStart[variant], 1, 0, k);
                ++g_lastWaterLevels;   // Phase 0 panel
            }
        }

        cmdBindRenderTargets(g_live.pCmd, nullptr);

        gpuPhaseEnd(kGpuPhaseWater);
        gpuPhaseBegin(kGpuPhaseResolve);
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

            // Resolve dst back to RENDER_TARGET (not COMMON yet) so the dev overlay can draw into
            // it; the final COMMON transition for the D3D9Ex handoff happens after drawDevUI().
            D3D12_RESOURCE_BARRIER post[2] = {};
            post[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            post[0].Transition.pResource = dstRes;
            post[0].Transition.Subresource = 0;
            post[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_DEST;
            post[0].Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;
            post[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            post[1].Transition.pResource = msaaRes;
            post[1].Transition.Subresource = 0;
            post[1].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
            post[1].Transition.StateAfter  = D3D12_RESOURCE_STATE_RENDER_TARGET;   // ready for next frame
            cl->ResourceBarrier(2, post);

            // Dev overlay into the resolved pRT (RENDER_TARGET), then hand off to COMMON natively.
            drawDevUI();
            D3D12_RESOURCE_BARRIER toCommon = {};
            toCommon.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            toCommon.Transition.pResource = dstRes;
            toCommon.Transition.Subresource = 0;
            toCommon.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
            toCommon.Transition.StateAfter  = D3D12_RESOURCE_STATE_COMMON;   // hand off to D3D9Ex
            cl->ResourceBarrier(1, &toCommon);
        } else {
            // No MSAA: dev overlay into pRT (still RENDER_TARGET), then hand back to COMMON for StretchRect.
            drawDevUI();
            RenderTargetBarrier toCommon = {};
            toCommon.pRenderTarget = g_live.pRT;
            toCommon.mCurrentState = RESOURCE_STATE_RENDER_TARGET;
            toCommon.mNewState = RESOURCE_STATE_COMMON;
            cmdResourceBarrier(g_live.pCmd, 0, nullptr, 0, nullptr, 1, &toCommon);
        }
        gpuPhaseEnd(kGpuPhaseResolve);
        // Resolve all GPU phase timestamps into the readback buffer (valid after the fence below).
        if (g_live.pGpuQueryPool) {
            cmdResolveQuery(g_live.pCmd, g_live.pGpuQueryPool, 0, kGpuPhaseCount);
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
        const double gpuMs = hostNowMs() - tRec1;
        g_recAccum += (tRec1 - tRec0);            // CPU per-draw bind+draw recording
        g_gpuAccum += gpuMs;                       // GPU execute (submit→fence)
        g_lastRecMs = tRec1 - tRec0;               // Phase 0 panel: this frame's single values
        g_lastGpuMs = gpuMs;
        // GPU per-phase breakdown of gpuMs: read back the timestamps (valid now the fence signalled).
        if (g_live.pGpuQueryPool && g_live.gpuTickFreq > 0.0) {
            for (uint32_t i = 0; i < kGpuPhaseCount; ++i) {
                QueryData qd = {};
                getQueryData(R, g_live.pGpuQueryPool, i, &qd);
                const uint64_t b = qd.mBeginTimestamp, e = qd.mEndTimestamp;
                g_lastGpuPhaseMs[i] = (e > b) ? ((double)(e - b) / g_live.gpuTickFreq) * 1000.0 : 0.0;
            }
        }
        // Stage B (B2): the GPU cull survivor count is valid now the fence signalled. Compare to the
        // CPU cull's g_liveLastInst in the heartbeat — they MUST match once the GPU logic is correct.
        if (g_live.pCullCountReadback && g_live.pCullCountReadback->pCpuMappedAddress) {
            g_lastGpuCullCount = *(const uint32_t*)g_live.pCullCountReadback->pCpuMappedAddress;
        }
        // Per-frame slow-GPU self-report (the 300-frame average can't surface a fresh dense-area
        // stall at 0.2 fps). Pairs with [dl-slow] (CPU cull) to localize a hitch to CPU vs GPU.
        if (gpuMs > 30.0) {
            dlLogGpuSlow(gpuMs, tRec1 - tRec0, drawn);
        }
        // A dense exterior frame is the suspected trigger; pin a removal to the draw submit.
        logDeviceRemoved(R, "renderScene/submit");

        g_live.firstFrame = false;
        g_lastDrawn = drawn;
        g_lastSkinnedDrawn = skinnedDrawn;
        g_lastMultiMapDrawn = multiMapDrawn;
        g_lastSkyDrawn = skyDrawn;

        // Host-side heartbeat to mgeHost64.log (LOG::logline; LOGF goes to uncaptured stdout).
        // dynamic = meshes in the upload-heap ring (must stay tiny — hundreds = over-promotion);
        // meshHigh = total slots ever populated (monotonic leak check). Lets us correlate the
        // client's [hb] frame cost with what the Forge renderer is actually drawing.
        // Post-fence + whole-frame totals (host-internal; the server also wall-times renderScene for
        // the client's hostMs — these let us see WHERE that wall goes: setup+cull+record+gpu+post).
        g_lastPostMs  = hostNowMs() - (tRec1 + gpuMs);
        g_lastTotalMs = hostNowMs() - tEntry;
        if ((g_renderFrame % 300u) == 0u) {
            LOG::logline(">> [forge-hb] frame=%u drawn=%u skinned=%u multimap=%u sky=%u dynamic=%u meshHigh=%u "
                         "| record=%.2fms gpu=%.2fms (avg/frame over 300)",
                         g_renderFrame, drawn, skinnedDrawn, multiMapDrawn, skyDrawn, g_dynamicCount, g_meshHigh,
                         g_recAccum / 300.0, g_gpuAccum / 300.0);
            // Host-frame split (this frame's instantaneous values) so the ~2ms "unaccounted inside the
            // host" the client saw is localized: setup (per-draw memcpy) + cull (DL CPU cull + lazy
            // loads) + record + gpu + post = total. cull is the prime pre-record suspect.
            LOG::logline(">> [forge-hb] host split: setup=%.2f cull=%.2f record=%.2f gpu=%.2f post=%.2f total=%.2fms"
                         " | cull examined=%u survivors=%u (%.3f us/1k examined) | gpuCull=%u %s",
                         g_lastSetupMs, g_lastCullMs, g_lastRecMs, g_lastGpuMs, g_lastPostMs, g_lastTotalMs,
                         g_lastCullExamined, g_liveLastInst,
                         g_lastCullExamined ? (g_lastCullMs * 1000.0 / (g_lastCullExamined / 1000.0)) : 0.0,
                         g_lastGpuCullCount, (g_lastGpuCullCount == g_liveLastInst) ? "MATCH" : "MISMATCH");
            LOG::logline(">> [forge-hb] gpu split: prepass=%.2f gtao=%.2f reflect=%.2f color=%.2f water=%.2f resolve=%.2f ms",
                         g_lastGpuPhaseMs[kGpuPhasePrepass], g_lastGpuPhaseMs[kGpuPhaseGtao],
                         g_lastGpuPhaseMs[kGpuPhaseReflect], g_lastGpuPhaseMs[kGpuPhaseColor],
                         g_lastGpuPhaseMs[kGpuPhaseWater], g_lastGpuPhaseMs[kGpuPhaseResolve]);
            dlLogHeartbeat();
            g_recAccum = 0.0;
            g_gpuAccum = 0.0;
        }
        return true;
    }

    void setDebugMode(unsigned m) { g_debugMode = m; }

    void setDevInput(int x, int y, unsigned buttons, float wheel, unsigned uiVisible) {
        g_inMouseX = (float)x;
        g_inMouseY = (float)y;
        g_inLBtn   = (buttons & 0x1u) != 0;
        g_inRBtn   = (buttons & 0x2u) != 0;
        g_inMBtn   = (buttons & 0x4u) != 0;
        g_inWheel  = wheel;
        g_uiVisible = (uiVisible != 0);
    }

    // Dev hot-reload (F8): rebuild the compute pipelines (gtao + linearize) from the dxil currently
    // on disk — no game restart. The descriptor sets stay valid (bound to the root signature, which
    // is unchanged), so only shader+pipeline are swapped. Recompile externally with fsl.py + redeploy
    // the *_0.dxil first, then trigger this. Queue is idled so no in-flight dispatch uses the old PSO.
    void reloadComputeShaders() {
        Renderer* R = g_live.pRenderer;
        if (!R || !g_live.pGtaoShader) {
            return;
        }
        waitQueueIdle(g_live.pQueue);

        removePipeline(R, g_live.pGtaoPipeline);       g_live.pGtaoPipeline = nullptr;
        removeShader(R, g_live.pGtaoShader);           g_live.pGtaoShader = nullptr;
        removePipeline(R, g_live.pLinearizePipeline);  g_live.pLinearizePipeline = nullptr;
        removeShader(R, g_live.pLinearizeShader);      g_live.pLinearizeShader = nullptr;
        removePipeline(R, g_live.pAOBlurPipeline);     g_live.pAOBlurPipeline = nullptr;
        removeShader(R, g_live.pAOBlurShader);         g_live.pAOBlurShader = nullptr;

        const char* linName = (g_live.sampleCount > 1) ? "linearizedepth_sc4.comp"
                                                       : "linearizedepth_sc1.comp";
        ShaderLoadDesc lsd = {};
        lsd.mComp.pFileName = linName;
        addShader(R, &lsd, &g_live.pLinearizeShader);
        ShaderLoadDesc gsd = {};
        gsd.mComp.pFileName = "gtao.comp";
        addShader(R, &gsd, &g_live.pGtaoShader);
        ShaderLoadDesc absd = {};
        absd.mComp.pFileName = "aoblur.comp";
        addShader(R, &absd, &g_live.pAOBlurShader);
        if (!g_live.pLinearizeShader || !g_live.pGtaoShader || !g_live.pAOBlurShader) {
            LOG::logline("!! [forge] hot-reload addShader FAILED (dxil missing on disk?)"); LOG::flush();
            return;
        }
        PipelineDesc lpd = {};
        lpd.mType = PIPELINE_TYPE_COMPUTE;
        lpd.mComputeDesc.pShaderProgram = g_live.pLinearizeShader;
        addPipeline(R, &lpd, &g_live.pLinearizePipeline);
        PipelineDesc gpd = {};
        gpd.mType = PIPELINE_TYPE_COMPUTE;
        gpd.mComputeDesc.pShaderProgram = g_live.pGtaoShader;
        addPipeline(R, &gpd, &g_live.pGtaoPipeline);
        PipelineDesc abpd = {};
        abpd.mType = PIPELINE_TYPE_COMPUTE;
        abpd.mComputeDesc.pShaderProgram = g_live.pAOBlurShader;
        addPipeline(R, &abpd, &g_live.pAOBlurPipeline);
        LOG::logline(">> [forge] compute shaders hot-reloaded (gtao.comp + aoblur.comp + %s)", linName); LOG::flush();

        // GRAPHICS hot-reload: also rebuild the water pipeline so water.vert/.frag edits go live on
        // the same trigger (the fsl.py watcher recompiles ALL dxil and bumps gtao's mtime LAST, so
        // water's fresh dxil is already on disk here). The persistent water resources/buffers/sets
        // are untouched — only the shader + PSO are swapped. Skipped cleanly if water isn't built.
        if (g_live.pWaterPipeline || g_live.pWaterShader) {
            if (buildWaterPipeline(R)) {
                LOG::logline(">> [forge][water] water graphics pipeline hot-reloaded (water.vert + water.frag)"); LOG::flush();
            }
        }
    }

    // Auto hot-reload: poll the gtao dxil's mtime (CWD = morrowind64, same dir the loader reads from)
    // every ~20 frames; when it changes, an external fsl.py watcher has landed a recompile, so rebuild
    // the compute pipelines. Edit .fsl -> save -> live, no key press. (F8 stays as a manual trigger.)
    void checkShaderHotReload() {
        static unsigned long long s_lastWrite = 0;
        static unsigned s_tick = 0;
        if ((s_tick++ % 20) != 0) {
            return;
        }
        WIN32_FILE_ATTRIBUTE_DATA fad = {};
        if (!GetFileAttributesExA("DIRECT3D12\\gtao.comp_0.dxil", GetFileExInfoStandard, &fad)) {
            return;
        }
        unsigned long long w = ((unsigned long long)fad.ftLastWriteTime.dwHighDateTime << 32) |
                               fad.ftLastWriteTime.dwLowDateTime;
        if (s_lastWrite == 0) {
            s_lastWrite = w;   // baseline: don't reload the first time we see the file
            return;
        }
        if (w != s_lastWrite) {
            s_lastWrite = w;
            LOG::logline(">> [forge] gtao dxil changed on disk — auto hot-reload"); LOG::flush();
            reloadComputeShaders();
        }
    }

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

    // Tier 2 diag: read pAO (R16G16B16A16_SFLOAT) back to CPU and log 3 texels across the row.
    // With the gtao gradient diag active, pAO should hold R=uv.x (≈.25/.5/.75 L/C/R), G=uv.y(.5),
    // B=1, A=uv.x. All-zero ⇒ GTAO write/dispatch broken; gradient ⇒ the graphics gAO READ is broken.
    void debugReadbackAO() {
        if (!g_live.pRenderer || !g_live.pAO) { std::printf("[forge] AO readback: no pAO\n"); return; }
        Renderer* R = g_live.pRenderer;
        // --- pLinearDepth readback (R32F, 4bpp) — does the LINEARIZE compute UAV write land? ---
        if (g_live.pLinearDepth) {
            const uint32_t W2 = g_live.width, H2 = g_live.height, bpp2 = 4;
            const uint32_t ra = (R->pGpu->mUploadBufferTextureRowAlignment > 1u) ? R->pGpu->mUploadBufferTextureRowAlignment : 1u;
            const uint32_t ta = (R->pGpu->mUploadBufferTextureAlignment > 1u) ? R->pGpu->mUploadBufferTextureAlignment : 1u;
            const uint32_t rp2 = roundUp(W2 * bpp2, ra);
            BufferLoadDesc bd2 = {};
            bd2.mDesc.mSize = roundUp64((uint64_t)rp2 * H2, ta);
            bd2.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_TO_CPU;
            bd2.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            bd2.mDesc.mStartState = RESOURCE_STATE_COPY_DEST;
            bd2.mDesc.mQueueType = QUEUE_TYPE_TRANSFER;
            Buffer* pRb2 = nullptr; bd2.ppBuffer = &pRb2;
            addResource(&bd2, nullptr); waitForAllResourceLoads();
            resetCmdPool(R, g_live.pCmdPool);
            beginCmd(g_live.pCmd);
            TextureBarrier tb2 = { g_live.pLinearDepth, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_COPY_SOURCE };
            cmdResourceBarrier(g_live.pCmd, 0, nullptr, 1, &tb2, 0, nullptr);
            endCmd(g_live.pCmd);
            QueueSubmitDesc s2 = {}; s2.mCmdCount = 1; s2.ppCmds = &g_live.pCmd; s2.pSignalFence = g_live.pFence; s2.mSubmitDone = true;
            queueSubmit(g_live.pQueue, &s2); waitForFences(R, 1, &g_live.pFence);
            TextureCopyDesc c2 = {}; c2.pTexture = g_live.pLinearDepth; c2.pBuffer = pRb2;
            c2.mTextureState = RESOURCE_STATE_COPY_SOURCE; c2.mQueueType = QUEUE_TYPE_GRAPHICS;
            SyncToken t2 = {}; copyResource(&c2, &t2); waitForToken(&t2);
            const uint8_t* b2 = (const uint8_t*)pRb2->pCpuMappedAddress;
            if (b2) {
                const float* cf = (const float*)(b2 + (uint64_t)(H2/2)*rp2 + (uint64_t)(W2/2)*bpp2);
                const float* lf = (const float*)(b2 + (uint64_t)(H2/2)*rp2 + (uint64_t)(W2/4)*bpp2);
                std::printf("[forge] LinDepth readback left=%.4f centre=%.4f\n", *lf, *cf);
            }
            removeResource(pRb2);
        }
        const uint32_t W = g_live.width, H = g_live.height;
        const uint32_t bpp = 8;   // R16G16B16A16_SFLOAT
        const uint32_t rowAlign = (R->pGpu->mUploadBufferTextureRowAlignment > 1u) ? R->pGpu->mUploadBufferTextureRowAlignment : 1u;
        const uint32_t texAlign = (R->pGpu->mUploadBufferTextureAlignment > 1u) ? R->pGpu->mUploadBufferTextureAlignment : 1u;
        const uint32_t rowPitch = roundUp(W * bpp, rowAlign);
        const uint64_t bufSize  = roundUp64((uint64_t)rowPitch * H, texAlign);
        BufferLoadDesc bd = {};
        bd.mDesc.mSize = bufSize;
        bd.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_TO_CPU;
        bd.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        bd.mDesc.mStartState = RESOURCE_STATE_COPY_DEST;
        bd.mDesc.mQueueType = QUEUE_TYPE_TRANSFER;
        Buffer* pRb = nullptr; bd.ppBuffer = &pRb;
        addResource(&bd, nullptr); waitForAllResourceLoads();
        resetCmdPool(R, g_live.pCmdPool);
        beginCmd(g_live.pCmd);
        // pAO ends each renderScene in SHADER_RESOURCE (post-GTAO barrier) → COPY_SOURCE.
        TextureBarrier tb = { g_live.pAO, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_COPY_SOURCE };
        cmdResourceBarrier(g_live.pCmd, 0, nullptr, 1, &tb, 0, nullptr);
        endCmd(g_live.pCmd);
        QueueSubmitDesc sd = {}; sd.mCmdCount = 1; sd.ppCmds = &g_live.pCmd; sd.pSignalFence = g_live.pFence; sd.mSubmitDone = true;
        queueSubmit(g_live.pQueue, &sd); waitForFences(R, 1, &g_live.pFence);
        TextureCopyDesc cd = {};
        cd.pTexture = g_live.pAO; cd.pBuffer = pRb;
        cd.mTextureState = RESOURCE_STATE_COPY_SOURCE; cd.mQueueType = QUEUE_TYPE_GRAPHICS;
        SyncToken ct = {}; copyResource(&cd, &ct); waitForToken(&ct);
        const uint8_t* base = (const uint8_t*)pRb->pCpuMappedAddress;
        if (base) {
            auto half2f = [](uint16_t h) -> float {
                uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
                uint32_t exp  = (h >> 10) & 0x1Fu;
                uint32_t man  = h & 0x3FFu;
                uint32_t f;
                if (exp == 0u) {
                    if (man == 0u) { f = sign; }
                    else { int e = 127 - 15 + 1; while (!(man & 0x400u)) { man <<= 1; --e; } man &= 0x3FFu; f = sign | ((uint32_t)e << 23) | (man << 13); }
                } else if (exp == 31u) { f = sign | 0x7F800000u | (man << 13); }
                else { f = sign | ((exp - 15u + 127u) << 23) | (man << 13); }
                float r; std::memcpy(&r, &f, 4); return r;
            };
            auto logpx = [&](uint32_t x, uint32_t y, const char* name) {
                const uint16_t* p = (const uint16_t*)(base + (uint64_t)y * rowPitch + (uint64_t)x * bpp);
                std::printf("[forge] AO readback %-6s (%u,%u) = R%.3f G%.3f B%.3f A%.3f\n",
                            name, x, y, half2f(p[0]), half2f(p[1]), half2f(p[2]), half2f(p[3]));
            };
            logpx(W / 4u, H / 2u, "left");
            logpx(W / 2u, H / 2u, "centre");
            logpx(3u * W / 4u, H / 2u, "right");
            // Offset-independent: scan the WHOLE copied buffer for any non-zero byte (decouples
            // "pAO truly zero" from "my pixel-offset math is wrong").
            uint64_t nz = 0, firstNz = 0; bool found = false;
            for (uint64_t i = 0; i < bufSize; ++i) { if (base[i] != 0) { ++nz; if (!found) { firstNz = i; found = true; } } }
            std::printf("[forge] AO buffer scan: %llu non-zero bytes / %llu (firstNz=%llu)\n",
                        (unsigned long long)nz, (unsigned long long)bufSize, (unsigned long long)firstNz);
        } else {
            std::printf("[forge] AO readback not mapped\n");
        }
        removeResource(pRb);
    }

    // --- Phase 2: DDS parse + bindless texture upload --------------------------------- (ddsRd32
    // is defined above, ahead of the WT1 water volume loader which also needs it.)
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

    // ===================== Phase 1a: host-owned distant land =============================
    // The distant land is loaded + culled + drawn entirely host-side (the DL files are static
    // per worldspace and the host's cwd is morrowind64), so it renders WITHOUT Morrowind — see
    // [[project_forge_dl_host_owned]] / tasks/forge-phase1.md. Resident LandElem VBs (16 B:
    // float3 pos + SHORT2N uv) + the 3 atlas textures (base/normal/detail) feed a 16 B land
    // pipeline that reuses the opaque SrtData/default.rootsig + the bindless gTextures array.
    // The --forge-dl probe renders it from a synthetic camera to ground-truth the loader/pipeline
    // /atlas in isolation; the same loader+pipeline+draw wire into renderScene for the live path.

    constexpr uint32_t kMaxLandMeshes  = 16384;
    // DL LAND atlas lives in the TOP 3 slots of gTextures[] (the client caps its own bottom-up
    // residency below kDlReserve; see IPC::kDlReserve) so host-owned land never stomps the client's
    // near-scene textures. Distant STATICS no longer use gTextures — they live in gStaticsArrays
    // (the bucketed Texture2DArray residency, below).
    constexpr uint32_t kLandBaseSlot   = MAX_TEXTURES - 1;   // 895
    constexpr uint32_t kLandNormalSlot = MAX_TEXTURES - 2;   // 894
    constexpr uint32_t kLandDetailSlot = MAX_TEXTURES - 3;   // 893

    struct LandMeshGPU {
        Buffer*  vb;
        Buffer*  ib;
        uint32_t indexCount;
        bool     large;             // 32-bit indices
        float    cx, cy, cz, r;     // bounding sphere (world)
    };
    LandMeshGPU g_landMeshes[kMaxLandMeshes];
    uint32_t    g_landMeshCount = 0;
    Shader*     g_pLandShader   = nullptr;
    Pipeline*   g_pLandPipeline = nullptr;
    bool        g_landLoaded    = false;

    // --- Phase 1b distant STATICS (host-owned, GPU-driven: instancing + bindless + execute-indirect) ---
    // The unique mesh LIBRARY (static_meshes) is packed ONCE into a mega VB (StaticElem, 20 B) + a
    // mega 16-bit IB; each subset records its (vbBase, ibBase, indexCount) into those. Placements
    // (usage.data worldspace 0 = exterior) expand to per-(instance×subset) entries grouped by subset
    // into one instance VB (binding 1: world matrix rows + bindless texSlot/flags). One indirect-arg
    // record per subset (with StartInstanceLocation = its instance run) drives a single
    // cmdExecuteIndirect. See [[project_forge_dl_statics_format]].
    constexpr uint32_t kStaticsBuckets    = MAX_STATICS_BUCKETS;  // gStaticsArrays element count (128)
    constexpr uint32_t kStaticsTexCap     = 512;   // cap the long side: extract that mip + chain (raw copy)
    constexpr uint32_t kStaticsInstStride = 80;    // 4 world rows (64 B) + params float4 (16 B)
    constexpr float    kStaticsScopeR     = 6144.0f; // probe: draw instances within this radius of the densest cell

    struct StaticsSubsetGPU {
        uint32_t vbBase;       // first vertex in the mega-VB (BaseVertexLocation)
        uint32_t ibBase;       // first index  in the mega-IB (StartIndexLocation)
        uint32_t indexCount;   // faces*3
        uint32_t texSlot;      // bindless gTextures slot (0 until its texture is loaded in scope)
        uint32_t flags;        // bit1 = hasAlpha cutout
    };
    struct StaticsDefCPU { uint32_t firstSubset, numSubsets; float radius; uint8_t type; }; // mirrors DistantStatic

    Buffer*   g_pStaticsVB       = nullptr;   // mega per-vertex (GPU_ONLY, stride 20)
    Buffer*   g_pStaticsIB       = nullptr;   // mega 16-bit index (GPU_ONLY)
    Buffer*   g_pStaticsInst     = nullptr;   // per-instance stream for the current scope (GPU_ONLY)
    Buffer*   g_pStaticsArgs     = nullptr;   // IndirectDrawIndexArguments[] for the current scope
    Shader*   g_pStaticsShader   = nullptr;
    Pipeline* g_pStaticsPipeline = nullptr;
    std::vector<StaticsSubsetGPU> g_staticsSubsets;
    std::vector<StaticsDefCPU>    g_staticsDefs;
    std::vector<std::string>      g_staticsSubsetTex;   // per-subset basename (lowercased)
    // DL statics texture residency = gStaticsArrays: a descriptor-array of Texture2DArrays, one
    // element per (format, capped-size) BUCKET. Every statics texture is uploaded ONCE at load into
    // its bucket as an array slice (raw mip-extract: the largest mip with long side <= kStaticsTexCap,
    // plus the chain below it — no decode/resize/encode). All resident => no eviction, no white,
    // no per-frame streaming. subset.texSlot = (bucket<<16)|layer. Bucket 0 is reserved = white.
    struct StaticsTexBucket {
        TinyImageFormat fmt = TinyImageFormat_UNDEFINED;
        uint32_t w = 0, h = 0, mips = 0;   // uniform across the bucket's slices
        Texture* tex = nullptr;            // the Texture2DArray (mArraySize = count)
        uint32_t count = 0;                // assigned layers
    };
    std::vector<StaticsTexBucket>             g_staticsBuckets;     // [0] = white; reals from [1]
    bool                                      g_staticsTexReady = false;
    std::vector<uint8_t>          g_usageData;          // resident usage.data
    // Canonical per-instance cull data, precomputed ONCE at load (statics never move). Replaces the
    // per-frame trig/mat-mul (Stage A) AND the per-frame tier/effR compute — the CPU cull now just
    // reads this. It is ALSO the exact resident struct the Stage B GPU compute cull reads (96 B,
    // 16-aligned). world = absolute (S·Rz·Ry·Rx·T); -eye is applied per frame. rangeEndIdx 0/1/2 =
    // near/far/veryFar tier; 0xFFFFFFFF = skip (grass). Indexed by ws0 instance index.
    // The 96 B size is deliberate: an HLSL StructuredBuffer element holding a float4x4 has 16-byte
    // alignment, so the shader struct rounds 80 -> 96; an explicit posZ + _pad keep the C++ upload
    // byte-identical to that stride and let the cull shader test the sphere z without decoding world.
    struct GpuCullInstance {
        float    world[16];        // absolute world matrix (64 B)
        float    posX, posY, posZ; // absolute position (xy = horizontal dist test, z = frustum sphere)
        float    effR;             // frustum sphere radius (def.radius * scale)            -> 80 B
        uint32_t rangeEndIdx;      // tier: 0=nearEnd 1=farEnd 2=vfarEnd ; 0xFFFFFFFF = skip
        uint32_t firstSubset;
        uint32_t numSubsets;
        uint32_t _pad;             // -> 96 B (matches the HLSL StructuredBuffer element stride)
    };
    std::vector<GpuCullInstance>  g_cullInst;
    uint64_t  g_ws0Off    = 0;                          // byte offset of the first exterior instance record
    uint32_t  g_ws0Count  = 0;                          // exterior instance count
    uint32_t  g_staticsDrawCount = 0;                   // EI arg count (subsets with scoped instances)
    uint32_t  g_staticsInstTotal = 0;                   // total scoped draw-instances
    bool      g_staticsLoaded    = false;

    // --- Phase 1a/1b LIVE distant land (host-owned cull, persistent rings) -------------------
    // The probe loads/draws DL in isolation; the LIVE path wires the same loaders + pipelines into
    // renderScene (after the near colour pass, shared depth). It must NOT addResource per frame (the
    // probe's buildStaticsScope pattern would stutter), so the per-frame instance + indirect-arg
    // streams ride PERSISTENT CPU_TO_GPU rings (created once). A uniform grid over the resident
    // exterior placements makes the per-frame frustum cull cheap. lodEye shifts DL into the
    // camera-relative space the near scene uses. See tasks/forge-phase1.md.
    enum { DL_STATIC_AUTO = 0, DL_STATIC_NEAR, DL_STATIC_FAR, DL_STATIC_VERY_FAR,
           DL_STATIC_GRASS, DL_STATIC_TREE, DL_STATIC_BUILDING };   // mirrors dlformat.h StaticType
    constexpr uint32_t kLiveMaxInst    = 65536;        // per-frame draw-instance cap (ring size)
    constexpr uint32_t kLiveMaxSubsets = 16384;        // per-frame indirect-arg cap (ring size)
    constexpr float    kLiveGridCell   = 8192.0f;      // uniform-grid cell (one MW cell) for the cull
    Buffer*   g_pStaticsInstRing = nullptr;            // CPU_TO_GPU per-frame instance stream (80 B/inst)
    Buffer*   g_pStaticsArgsRing = nullptr;            // CPU_TO_GPU per-frame IndirectDrawIndexArguments[]
    struct LiveGridCell {
        std::vector<uint32_t> inst;                    // instance indices into g_usageData ws0 records
        float minx, miny, minz, maxx, maxy, maxz;      // AABB of member placements (padded for cull)
    };
    std::vector<LiveGridCell> g_liveGrid;
    bool      g_dlExterior   = false;                  // per-frame: client's isExterior gate
    bool      g_dlLiveInit   = false;                  // resident load + grid + rings done
    bool      g_staticsLiveOk = false;                 // statics library + rings ready
    float     g_dlEye[3]     = { 0, 0, 0 };            // realEye this frame (DL shift origin)
    // (Statics textures are all resident from load via gStaticsArrays — no per-frame load budget.)
    std::vector<uint32_t> g_liveLandVisible;           // land mesh indices surviving the frustum cull
    // g_liveLastInst/Subsets/Land declared up with the other per-frame counters (Phase 0 panel needs
    // them in drawDevUI, which is defined earlier in the TU).

    // Read an entire file into `out`. Uses std::vector (Forge's IMemory bans raw malloc).
    bool dlReadWholeFile(const char* path, std::vector<uint8_t>& out) {
        out.clear();
        std::FILE* f = std::fopen(path, "rb");
        if (!f) { return false; }
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (sz <= 0) { std::fclose(f); return false; }
        out.resize((size_t)sz);
        size_t rd = std::fread(out.data(), 1, (size_t)sz, f);
        std::fclose(f);
        if (rd != (size_t)sz) { out.clear(); return false; }
        return true;
    }

    // Decode a DDS file from disk into bindless gTextures[slot] (reuses parseDds + the
    // uploadTextures create/upload/rebind path). Returns true on success.
    bool dlLoadAtlas(Renderer* R, const char* path, uint32_t slot) {
        std::vector<uint8_t> dds;
        if (!dlReadWholeFile(path, dds)) { std::printf("[forge][dl] atlas missing: %s\n", path); return false; }
        DdsInfo info = parseDds(dds.data(), (unsigned)dds.size());
        if (!info.ok) { std::printf("[forge][dl] atlas unsupported DDS: %s\n", path); return false; }

        Texture* tex = nullptr;
        TextureDesc td = {};
        td.mWidth = info.width; td.mHeight = info.height; td.mDepth = 1;
        td.mArraySize = 1; td.mMipLevels = info.mipLevels;
        td.mSampleCount = SAMPLE_COUNT_1;
        td.mFormat = info.fmt;
        td.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
        td.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
        td.pName = "dlAtlas";
        TextureLoadDesc tld = {};
        tld.ppTexture = &tex;
        tld.pDesc = &td;
        addResource(&tld, nullptr);
        waitForAllResourceLoads();
        if (!tex) { return false; }

        const uint8_t* src    = dds.data() + info.dataOffset;
        const uint8_t* ddsEnd = dds.data() + dds.size();
        TextureUpdateDesc upd = {};
        upd.pTexture = tex;
        upd.mBaseMipLevel = 0; upd.mMipLevels = info.mipLevels;
        upd.mBaseArrayLayer = 0; upd.mLayerCount = 1;
        upd.mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
        beginUpdateResource(&upd);
        for (uint32_t m = 0; m < info.mipLevels; ++m) {
            TextureSubresourceUpdate s = upd.getSubresourceUpdateDesc(m, 0);
            const uint32_t mipBytes = s.mRowCount * s.mSrcRowStride;
            if (src + mipBytes > ddsEnd) { break; }
            for (uint32_t row = 0; row < s.mRowCount; ++row) {
                std::memcpy(s.pMappedData + (size_t)row * s.mDstRowStride,
                            src + (size_t)row * s.mSrcRowStride, s.mSrcRowStride);
            }
            src += mipBytes;
        }
        endUpdateResource(&upd);
        flushTextureUploads(R);

        if (g_live.pTextures[slot] != g_live.pDefaultWhite) { removeResource(g_live.pTextures[slot]); }
        g_live.pTextures[slot] = tex;
        if (slot + 1 > g_live.texHigh) { g_live.texHigh = slot + 1; }
        DescriptorData dd = {};
        dd.mIndex = SRT_RES_IDX(SrtData, Persistent, gTextures);
        dd.mArrayOffset = slot;
        dd.mCount = 1;
        dd.ppTextures = &g_live.pTextures[slot];
        updateDescriptorSet(R, 0, g_live.pPersistentSet, 1, &dd);
        std::printf("[forge][dl] atlas slot %u <- %s (%ux%u, %u mips)\n",
                    slot, path, info.width, info.height, info.mipLevels);
        return true;
    }

    // Build the land pipeline: distantland.vert/.frag, 16 B vertex layout (pos + SHORT2N uv),
    // depth-write + reverse-Z GEQUAL (land owns its depth — no prepass, terrain is fully opaque),
    // drawn after the near colour pass. Reuses default.rootsig + the opaque descriptor sets.
    bool buildLandPath(Renderer* R) {
        if (g_pLandPipeline) { return true; }
        ShaderLoadDesc sd = {};
        sd.mVert.pFileName = "distantland.vert";
        sd.mFrag.pFileName = "distantland.frag";
        addShader(R, &sd, &g_pLandShader);
        if (!g_pLandShader) { std::printf("[forge][dl] addShader(distantland) FAILED\n"); return false; }

        VertexLayout vl = {};
        vl.mBindingCount = 1;
        vl.mBindings[0].mStride = 16;               // LandElem: float3 pos + SHORT2N uv
        vl.mBindings[0].mRate = VERTEX_BINDING_RATE_VERTEX;
        vl.mAttribCount = 2;
        vl.mAttribs[0].mSemantic = SEMANTIC_POSITION;
        vl.mAttribs[0].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
        vl.mAttribs[0].mBinding = 0;
        vl.mAttribs[0].mLocation = 0;
        vl.mAttribs[0].mOffset = 0;
        vl.mAttribs[1].mSemantic = SEMANTIC_TEXCOORD0;
        vl.mAttribs[1].mFormat = TinyImageFormat_R16G16_SNORM;   // SHORT2N -> [-1,1], as the D3D9 decl
        vl.mAttribs[1].mBinding = 0;
        vl.mAttribs[1].mLocation = 1;
        vl.mAttribs[1].mOffset = 12;

        DepthStateDesc ds = {};
        ds.mDepthTest = true;
        ds.mDepthWrite = true;
        ds.mDepthFunc = CMP_GEQUAL;                 // reverse-Z (near->1, far->0); pDepth clears to 0

        RasterizerStateDesc rs = {};
        rs.mCullMode = CULL_MODE_NONE;              // terrain heightfield: don't risk culling LOD faces
        rs.mFrontFace = FRONT_FACE_CCW;

        PipelineDesc pd = {};
        pd.mType = PIPELINE_TYPE_GRAPHICS;
        GraphicsPipelineDesc& g = pd.mGraphicsDesc;
        g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
        g.mRenderTargetCount = 1;
        g.pColorFormats = &g_live.pRT->mFormat;
        g.mSampleCount = (SampleCount)g_live.sampleCount;
        g.mSampleQuality = 0;
        g.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
        g.pDepthState = &ds;
        g.pVertexLayout = &vl;
        g.pRasterizerState = &rs;
        g.pShaderProgram = g_pLandShader;
        addPipeline(R, &pd, &g_pLandPipeline);
        if (!g_pLandPipeline) { std::printf("[forge][dl] addPipeline(distantland) FAILED\n"); return false; }
        std::printf("[forge][dl] land pipeline built\n");
        return true;
    }

    // Load the 3 atlas textures + parse the distantland\world container into resident Forge
    // VB/IB + bounding spheres. Idempotent. Requires buildOpaquePath (pPersistentSet) already built.
    bool loadDistantLand(Renderer* R) {
        if (g_landLoaded) { return true; }
        bool a0 = dlLoadAtlas(R, "Data Files\\distantland\\world.dds",          kLandBaseSlot);
        bool a1 = dlLoadAtlas(R, "Data Files\\distantland\\world_n.dds",        kLandNormalSlot);
        bool a2 = dlLoadAtlas(R, "Data Files\\textures\\MGE\\world_detail.dds", kLandDetailSlot);
        if (!a0 || !a1 || !a2) { std::printf("[forge][dl] atlas incomplete (a0=%d a1=%d a2=%d)\n", a0, a1, a2); return false; }

        std::vector<uint8_t> file;
        if (!dlReadWholeFile("Data Files\\distantland\\world", file) || file.size() < 4) {
            std::printf("[forge][dl] world container missing/empty\n");
            return false;
        }
        const uint8_t* p   = file.data();
        const uint8_t* end = file.data() + file.size();
        uint32_t meshCount = 0;
        std::memcpy(&meshCount, p, 4); p += 4;
        std::printf("[forge][dl] world container: %u meshes (%zu bytes)\n", meshCount, file.size());

        uint32_t built = 0;
        for (uint32_t i = 0; i < meshCount && g_landMeshCount < kMaxLandMeshes; ++i) {
            // Per-mesh header: radius(4) + center(12) + boxMin(12) + boxMax(12) = 40, then verts(4)+faces(4).
            if (p + 48 > end) { break; }
            float radius; float center[3];
            std::memcpy(&radius, p, 4);
            std::memcpy(center, p + 4, 12);
            p += 40;                                 // skip boxMin/boxMax (sphere suffices for the cull)
            uint32_t verts = 0, faces = 0;
            std::memcpy(&verts, p, 4); std::memcpy(&faces, p + 4, 4); p += 8;

            bool large = (verts > 0xFFFFu || faces > 0xFFFFu);
            size_t vbBytes = (size_t)verts * 16;
            size_t ibBytes = (size_t)faces * (large ? 12 : 6);
            if (p + vbBytes + ibBytes > end) { std::printf("[forge][dl] mesh %u truncated\n", i); break; }
            const uint8_t* vbSrc = p;
            const uint8_t* ibSrc = p + vbBytes;
            p += vbBytes + ibBytes;
            if (!verts || !faces) { continue; }

            LandMeshGPU& m = g_landMeshes[g_landMeshCount];
            m.vb = nullptr; m.ib = nullptr;
            BufferLoadDesc vbd = {};
            vbd.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
            vbd.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
            vbd.mDesc.mSize = vbBytes;
            vbd.pData = vbSrc;
            vbd.ppBuffer = &m.vb;
            addResource(&vbd, nullptr);
            BufferLoadDesc ibd = {};
            ibd.mDesc.mDescriptors = DESCRIPTOR_TYPE_INDEX_BUFFER;
            ibd.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
            ibd.mDesc.mSize = ibBytes;
            ibd.pData = ibSrc;
            ibd.ppBuffer = &m.ib;
            addResource(&ibd, nullptr);
            waitForAllResourceLoads();
            if (!m.vb || !m.ib) { std::printf("[forge][dl] mesh %u buffer alloc FAILED\n", i); continue; }

            m.indexCount = faces * 3;
            m.large = large;
            m.cx = center[0]; m.cy = center[1]; m.cz = center[2]; m.r = radius;
            ++g_landMeshCount;
            ++built;
        }
        std::printf("[forge][dl] resident land meshes: %u\n", g_landMeshCount);
        g_landLoaded = (built > 0);
        return g_landLoaded;
    }

    void dlMul(const float a[16], const float b[16], float out[16]);   // defined with the camera helpers below

    // Build the statics pipeline: statics.vert/.frag, two vertex bindings (0 = StaticElem 20 B,
    // per-vertex; 1 = per-INSTANCE world matrix rows + params, 80 B), depth-write + reverse-Z
    // GEQUAL (shares pDepth with land/near). Reuses default.rootsig + the opaque descriptor sets.
    bool buildStaticsPath(Renderer* R) {
        if (g_pStaticsPipeline) { return true; }
        ShaderLoadDesc sd = {};
        sd.mVert.pFileName = "statics.vert";
        sd.mFrag.pFileName = "statics.frag";
        addShader(R, &sd, &g_pStaticsShader);
        if (!g_pStaticsShader) { std::printf("[forge][dl] addShader(statics) FAILED\n"); return false; }

        VertexLayout vl = {};
        vl.mBindingCount = 2;
        vl.mBindings[0].mStride = 20;                       // StaticElem: FLOAT16_4 pos, UBYTE4N nrm, D3DCOLOR, FLOAT16_2 uv
        vl.mBindings[0].mRate   = VERTEX_BINDING_RATE_VERTEX;
        vl.mBindings[1].mStride = kStaticsInstStride;       // per-instance: 4 world rows + params
        vl.mBindings[1].mRate   = VERTEX_BINDING_RATE_INSTANCE;
        vl.mAttribCount = 9;
        // binding 0 (per-vertex)
        vl.mAttribs[0].mSemantic = SEMANTIC_POSITION;
        vl.mAttribs[0].mFormat   = TinyImageFormat_R16G16B16A16_SFLOAT;  // FLOAT16_4
        vl.mAttribs[0].mBinding  = 0; vl.mAttribs[0].mLocation = 0; vl.mAttribs[0].mOffset = 0;
        vl.mAttribs[1].mSemantic = SEMANTIC_NORMAL;
        vl.mAttribs[1].mFormat   = TinyImageFormat_R8G8B8A8_UNORM;       // UBYTE4N (xyz=normal, w=emissive)
        vl.mAttribs[1].mBinding  = 0; vl.mAttribs[1].mLocation = 1; vl.mAttribs[1].mOffset = 8;
        vl.mAttribs[2].mSemantic = SEMANTIC_COLOR;
        vl.mAttribs[2].mFormat   = TinyImageFormat_B8G8R8A8_UNORM;       // D3DCOLOR byte order
        vl.mAttribs[2].mBinding  = 0; vl.mAttribs[2].mLocation = 2; vl.mAttribs[2].mOffset = 12;
        vl.mAttribs[3].mSemantic = SEMANTIC_TEXCOORD0;
        vl.mAttribs[3].mFormat   = TinyImageFormat_R16G16_SFLOAT;        // FLOAT16_2
        vl.mAttribs[3].mBinding  = 0; vl.mAttribs[3].mLocation = 3; vl.mAttribs[3].mOffset = 16;
        // binding 1 (per-instance): 4 world rows (TEXCOORD1..4) + params (TEXCOORD5)
        for (uint32_t k = 0; k < 5; ++k) {
            vl.mAttribs[4 + k].mSemantic = (ShaderSemantic)(SEMANTIC_TEXCOORD1 + k);
            vl.mAttribs[4 + k].mFormat   = TinyImageFormat_R32G32B32A32_SFLOAT;
            vl.mAttribs[4 + k].mBinding  = 1;
            vl.mAttribs[4 + k].mLocation = 4 + k;
            vl.mAttribs[4 + k].mOffset   = k * 16;
        }

        DepthStateDesc ds = {};
        ds.mDepthTest = true; ds.mDepthWrite = true; ds.mDepthFunc = CMP_GEQUAL;
        RasterizerStateDesc rs = {};
        // Single-sided, front = CCW: under MW's live view+proj the facing is INVERTED from the
        // --forge-dl probe's synthetic LH camera (the probe's FRONT_FACE_CW showed BACK faces =
        // "inside out"; CULL_NONE confirmed the geometry is otherwise correct). CCW is the live
        // front, so back-cull saves the back-face fill. (MGE itself draws distant statics single-
        // sided; its foliage LODs are crossed quads, so single-sided is correct for this geometry.)
        rs.mCullMode = CULL_MODE_BACK; rs.mFrontFace = FRONT_FACE_CCW;

        PipelineDesc pd = {};
        pd.mType = PIPELINE_TYPE_GRAPHICS;
        GraphicsPipelineDesc& g = pd.mGraphicsDesc;
        g.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
        g.mRenderTargetCount = 1;
        g.pColorFormats = &g_live.pRT->mFormat;
        g.mSampleCount = (SampleCount)g_live.sampleCount;
        g.mSampleQuality = 0;
        g.mDepthStencilFormat = TinyImageFormat_D32_SFLOAT;
        g.pDepthState = &ds;
        g.pVertexLayout = &vl;
        g.pRasterizerState = &rs;
        g.pShaderProgram = g_pStaticsShader;
        addPipeline(R, &pd, &g_pStaticsPipeline);
        if (!g_pStaticsPipeline) { std::printf("[forge][dl] addPipeline(statics) FAILED\n"); return false; }
        std::printf("[forge][dl] statics pipeline built\n");
        return true;
    }

    // Parse static_meshes into the mega VB/IB + per-subset records, and load usage.data resident.
    // The unique geometry is uploaded ONCE; textures + instances are bound later per scope. Idempotent.
    bool loadDistantStatics(Renderer* R) {
        if (g_staticsLoaded) { return true; }

        // usage.data: DWORD DistantStaticCount, DWORD dynamicVisGroupCount, [visgroups 130 B each],
        // then per-worldspace { DWORD count; (ws>0) char[64] name; records[count*34] }. ws0 = exterior.
        if (!dlReadWholeFile("Data Files\\distantland\\statics\\usage.data", g_usageData) || g_usageData.size() < 8) {
            std::printf("[forge][dl] usage.data missing\n"); return false;
        }
        uint32_t distantStaticCount = 0, visGroupCount = 0;
        std::memcpy(&distantStaticCount, &g_usageData[0], 4);
        std::memcpy(&visGroupCount,      &g_usageData[4], 4);
        uint64_t uoff = 8 + (uint64_t)visGroupCount * 130;
        if (uoff + 4 > g_usageData.size()) { std::printf("[forge][dl] usage.data truncated header\n"); return false; }
        std::memcpy(&g_ws0Count, &g_usageData[uoff], 4); uoff += 4;
        g_ws0Off = uoff;                                  // ws0 has no name; records follow immediately
        if (g_ws0Off + (uint64_t)g_ws0Count * 34 > g_usageData.size()) {
            std::printf("[forge][dl] usage.data ws0 overrun (%u instances)\n", g_ws0Count); return false;
        }

        // static_meshes: per DistantStatic { DWORD numSubsets; float r; float3 c; byte type; per subset
        //   { float r; float3 c; float3 amin; float3 amax; int verts; int faces; vbytes[verts*20];
        //     idx[faces*3]u16; bool[2]{hasAlpha,uvCtrl}; u16 pathsize; char name[pathsize] } }.
        std::vector<uint8_t> file;
        if (!dlReadWholeFile("Data Files\\distantland\\statics\\static_meshes", file) || file.size() < 4) {
            std::printf("[forge][dl] static_meshes missing\n"); return false;
        }
        const uint8_t* p   = file.data();
        const uint8_t* end = file.data() + file.size();

        std::vector<uint8_t> megaVB, megaIB;             // packed library geometry
        megaVB.reserve(160u << 20); megaIB.reserve(64u << 20);
        g_staticsSubsets.clear(); g_staticsDefs.clear(); g_staticsSubsetTex.clear();
        g_staticsDefs.reserve(distantStaticCount);

        for (uint32_t s = 0; s < distantStaticCount; ++s) {
            if (p + 21 > end) { break; }
            uint32_t numSubsets = 0; std::memcpy(&numSubsets, p, 4);
            float    sradius = 0.0f; std::memcpy(&sradius, p + 4, 4);   // model bounding radius
            uint8_t  stype   = *(p + 4 + 4 + 12);                       // StaticType (dlformat.h)
            p += 4 + 4 + 12 + 1;                          // numSubsets + radius + center + type
            StaticsDefCPU def; def.firstSubset = (uint32_t)g_staticsSubsets.size(); def.numSubsets = numSubsets;
            def.radius = sradius; def.type = stype;
            for (uint32_t ss = 0; ss < numSubsets; ++ss) {
                if (p + 44 > end) { p = end; break; }
                p += 4 + 12 + 12 + 12;                    // subset sphere + aabbMin + aabbMax
                int verts = 0, faces = 0;
                std::memcpy(&verts, p, 4); std::memcpy(&faces, p + 4, 4); p += 8;
                size_t vbBytes = (size_t)verts * 20;
                size_t ibBytes = (size_t)faces * 6;       // 16-bit indices, faces*3
                if (verts < 0 || faces < 0 || p + vbBytes + ibBytes + 4 > end) { p = end; break; }
                const uint8_t* vbSrc = p;       p += vbBytes;
                const uint8_t* ibSrc = p;       p += ibBytes;
                uint8_t hasAlpha = *p; p += 2;            // bool[2] {hasAlpha, hasUVController}
                uint16_t pathsize = 0; std::memcpy(&pathsize, p, 2); p += 2;
                if (p + pathsize > end) { p = end; break; }
                // texname -> path RELATIVE to statics\textures\, lowercased, .dds. The LOD library
                // MIRRORS the source texture subfolders (tr\, hr\, sum\, glow\, ... for TR/mod packs):
                // 226 of 1557 DDS live in subfolders, so basename-only lookup loses the folder and
                // ~250 statics go white. Keep the subpath: strip trailing NUL/space, normalize slashes
                // + lowercase, drop a leading "data files\" then "textures\" prefix, force .dds (the
                // library is all .dds; static_meshes stores the original, often .tga, source name).
                std::string name;
                {
                    const char* np = (const char*)p; uint16_t n = pathsize;
                    while (n > 0 && (np[n-1] == 0 || np[n-1] == ' ')) { --n; }
                    std::string full;
                    for (int i = 0; i < (int)n; ++i) {
                        char c = np[i];
                        if (c == '/') { c = '\\'; }
                        if (c >= 'A' && c <= 'Z') { c = (char)(c - 'A' + 'a'); }
                        full.push_back(c);
                    }
                    if (full.rfind("data files\\", 0) == 0) { full.erase(0, 11); }
                    if (full.rfind("textures\\", 0) == 0)    { full.erase(0, 9); }
                    size_t dot = full.find_last_of('.');
                    if (dot != std::string::npos) { full.erase(dot); }
                    name = full + ".dds";
                }
                p += pathsize;

                StaticsSubsetGPU sub = {};
                sub.vbBase     = (uint32_t)(megaVB.size() / 20);
                sub.ibBase     = (uint32_t)(megaIB.size() / 2);
                sub.indexCount = (uint32_t)faces * 3;
                sub.texSlot    = 0;                       // assigned lazily when first drawn in scope
                sub.flags      = hasAlpha ? 0x2u : 0u;
                if (verts > 0 && faces > 0) {
                    megaVB.insert(megaVB.end(), vbSrc, vbSrc + vbBytes);
                    megaIB.insert(megaIB.end(), ibSrc, ibSrc + ibBytes);
                }
                g_staticsSubsets.push_back(sub);
                g_staticsSubsetTex.push_back(name);
            }
            g_staticsDefs.push_back(def);
        }
        std::printf("[forge][dl] statics library: %zu statics, %zu subsets, megaVB %.1fMB megaIB %.1fMB\n",
                    g_staticsDefs.size(), g_staticsSubsets.size(), megaVB.size()/1e6, megaIB.size()/1e6);
        if (megaVB.empty() || megaIB.empty()) { return false; }

        BufferLoadDesc vbd = {};
        vbd.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
        vbd.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        vbd.mDesc.mSize = megaVB.size();
        vbd.mDesc.pName = "staticsVB";
        vbd.pData = megaVB.data();
        vbd.ppBuffer = &g_pStaticsVB;
        addResource(&vbd, nullptr);
        BufferLoadDesc ibd = {};
        ibd.mDesc.mDescriptors = DESCRIPTOR_TYPE_INDEX_BUFFER;
        ibd.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        ibd.mDesc.mSize = megaIB.size();
        ibd.mDesc.pName = "staticsIB";
        ibd.pData = megaIB.data();
        ibd.ppBuffer = &g_pStaticsIB;
        addResource(&ibd, nullptr);
        waitForAllResourceLoads();
        if (!g_pStaticsVB || !g_pStaticsIB) { std::printf("[forge][dl] statics mega buffer alloc FAILED\n"); return false; }

        g_staticsLoaded = true;
        return true;
    }

    // Tightly-packed (DDS on-disk) byte size of one mip surface at (w,h) for the given format. BCn
    // is block-packed (min 1 block); uncompressed is 32-bit. Matches util_get_surface_info's tight
    // mSrcRowStride*mRowCount, so it correctly advances over skipped top mips.
    static uint32_t ddsTightMipBytes(TinyImageFormat fmt, uint32_t w, uint32_t h) {
        uint32_t bw, bh;
        switch (fmt) {
            case TinyImageFormat_DXBC1_RGBA_UNORM:
                bw = (w + 3) / 4; bh = (h + 3) / 4; if (!bw) bw = 1; if (!bh) bh = 1;
                return bw * bh * 8;
            case TinyImageFormat_DXBC2_UNORM:
            case TinyImageFormat_DXBC3_UNORM:
                bw = (w + 3) / 4; bh = (h + 3) / 4; if (!bw) bw = 1; if (!bh) bh = 1;
                return bw * bh * 16;
            default:   // BGRA8 / RGBA8 (uncompressed 32-bit)
                return w * h * 4;
        }
    }

    // Build the distant-statics texture residency (gStaticsArrays): one Texture2DArray per
    // (format, capped-size) bucket, every unique statics texture uploaded ONCE as a slice via raw
    // mip-extract (the largest mip with long side <= kStaticsTexCap + the chain below it — no
    // decode/resize/encode). Each subset's texSlot resolves to (bucket<<16)|layer. All resident:
    // no eviction, no per-frame streaming, no white-on-traverse (the old LRU thrashed at high
    // DrawDist because a single view's working set exceeds the shared descriptor table). Idempotent.
    // Requires loadDistantStatics + the Persistent set (g_live.pStaticsWhiteArray). Runs ONCE at the
    // first-exterior init (a one-time multi-hundred-ms hitch — reading ~1.5k DDS + GPU uploads).
    bool buildStaticsTextureArrays(Renderer* R) {
        if (g_staticsTexReady) { return true; }
        if (!g_staticsLoaded || !g_live.pPersistentSet || !g_live.pStaticsWhiteArray) { return false; }
        const std::string texDir = "Data Files\\distantland\\statics\\textures\\";

        struct UTexPlan { TinyImageFormat fmt; uint32_t cw, ch, capStep, availMips, bucket, layer; };
        std::unordered_map<std::string, UTexPlan> plan;        // unique name -> plan (zero = white/bucket 0)
        std::unordered_map<uint64_t, uint32_t>    keyToBucket; // (fmt,cw,ch) -> bucket index
        auto bucketKey = [](TinyImageFormat f, uint32_t w, uint32_t h) -> uint64_t {
            return ((uint64_t)(uint32_t)f << 40) | ((uint64_t)(w & 0xFFFFF) << 20) | (uint64_t)(h & 0xFFFFF);
        };

        g_staticsBuckets.clear();
        g_staticsBuckets.push_back(StaticsTexBucket{ TinyImageFormat_R8G8B8A8_UNORM, 4, 4, 1,
                                                     g_live.pStaticsWhiteArray, 2 });   // [0] = white
        std::vector<std::vector<std::string>> bucketMembers; bucketMembers.emplace_back();

        // Pass 1: header scan -> bucket plan (dedup by name).
        uint32_t missing = 0, overflow = 0;
        for (const std::string& nm : g_staticsSubsetTex) {
            if (nm.empty() || plan.find(nm) != plan.end()) { continue; }
            std::vector<uint8_t> hdr;
            DdsInfo info = dlReadWholeFile((texDir + nm).c_str(), hdr)
                         ? parseDds(hdr.data(), (uint32_t)hdr.size()) : DdsInfo{};
            if (!info.ok) { plan[nm] = UTexPlan{}; ++missing; continue; }   // -> white (bucket 0)
            uint32_t k = 0, w = info.width, h = info.height;
            while ((w > kStaticsTexCap || h > kStaticsTexCap) && (k + 1) < info.mipLevels) {
                w = (w > 1) ? w >> 1 : 1; h = (h > 1) ? h >> 1 : 1; ++k;    // clamp to leave >=1 mip
            }
            UTexPlan up; up.fmt = info.fmt; up.cw = w; up.ch = h; up.capStep = k;
            up.availMips = info.mipLevels - k;
            const uint64_t key = bucketKey(info.fmt, w, h);
            auto bit = keyToBucket.find(key);
            if (bit == keyToBucket.end()) {
                if (g_staticsBuckets.size() >= kStaticsBuckets) { plan[nm] = UTexPlan{}; ++overflow; continue; }
                const uint32_t b = (uint32_t)g_staticsBuckets.size();
                keyToBucket[key] = b;
                StaticsTexBucket nb; nb.fmt = info.fmt; nb.w = w; nb.h = h; nb.mips = up.availMips;
                g_staticsBuckets.push_back(nb);
                bucketMembers.emplace_back();
                bit = keyToBucket.find(key);
            }
            up.bucket = bit->second;
            StaticsTexBucket& b = g_staticsBuckets[up.bucket];
            if (up.availMips < b.mips) { b.mips = up.availMips; }   // uniform chain = min over members
            up.layer = b.count++;
            bucketMembers[up.bucket].push_back(nm);
            plan[nm] = up;
        }

        // Pass 2: create + bind one Texture2DArray per real bucket.
        for (uint32_t b = 1; b < g_staticsBuckets.size(); ++b) {
            StaticsTexBucket& bk = g_staticsBuckets[b];
            TextureDesc td = {};
            td.mWidth = bk.w; td.mHeight = bk.h; td.mDepth = 1;
            td.mArraySize = bk.count; td.mMipLevels = bk.mips;
            td.mSampleCount = SAMPLE_COUNT_1;
            td.mFormat = bk.fmt;
            td.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
            td.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
            td.pName = "staticsBucket";
            TextureLoadDesc tld = {}; tld.ppTexture = &bk.tex; tld.pDesc = &td;
            addResource(&tld, nullptr);
        }
        waitForAllResourceLoads();
        {
            std::vector<Texture*> texs(g_staticsBuckets.size());
            for (uint32_t b = 0; b < g_staticsBuckets.size(); ++b) {
                texs[b] = g_staticsBuckets[b].tex ? g_staticsBuckets[b].tex : g_live.pStaticsWhiteArray;
            }
            DescriptorData sd = {};
            sd.mIndex = SRT_RES_IDX(SrtData, Persistent, gStaticsArrays);
            sd.mArrayOffset = 0; sd.mCount = (uint32_t)g_staticsBuckets.size();
            sd.ppTextures = texs.data();
            updateDescriptorSet(R, 0, g_live.pPersistentSet, 1, &sd);
        }

        // Pass 3: upload each unique texture's capped mip range into its slice.
        uint32_t uploaded = 0;
        for (uint32_t b = 1; b < g_staticsBuckets.size(); ++b) {
            StaticsTexBucket& bk = g_staticsBuckets[b];
            if (!bk.tex) { continue; }
            for (uint32_t layer = 0; layer < (uint32_t)bucketMembers[b].size(); ++layer) {
                const std::string& nm = bucketMembers[b][layer];
                std::vector<uint8_t> dds;
                if (!dlReadWholeFile((texDir + nm).c_str(), dds)) { continue; }
                DdsInfo info = parseDds(dds.data(), (uint32_t)dds.size());
                if (!info.ok) { continue; }
                const UTexPlan& up = plan[nm];
                const uint8_t* src    = dds.data() + info.dataOffset;
                const uint8_t* ddsEnd = dds.data() + dds.size();
                uint32_t sw = info.width, sh = info.height;                 // advance over skipped top mips
                for (uint32_t i = 0; i < up.capStep; ++i) {
                    src += ddsTightMipBytes(info.fmt, sw, sh);
                    sw = (sw > 1) ? sw >> 1 : 1; sh = (sh > 1) ? sh >> 1 : 1;
                }
                TextureUpdateDesc upd = {};
                upd.pTexture = bk.tex;
                upd.mBaseMipLevel = 0; upd.mMipLevels = bk.mips;
                upd.mBaseArrayLayer = layer; upd.mLayerCount = 1;
                upd.mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
                beginUpdateResource(&upd);
                for (uint32_t m = 0; m < bk.mips; ++m) {
                    TextureSubresourceUpdate s = upd.getSubresourceUpdateDesc(m, layer);
                    const uint32_t mipBytes = s.mRowCount * s.mSrcRowStride;
                    if (src + mipBytes > ddsEnd) { break; }
                    for (uint32_t row = 0; row < s.mRowCount; ++row) {
                        std::memcpy(s.pMappedData + (size_t)row * s.mDstRowStride,
                                    src + (size_t)row * s.mSrcRowStride, s.mSrcRowStride);
                    }
                    src += mipBytes;
                }
                endUpdateResource(&upd);
                ++uploaded;
            }
            flushTextureUploads(R);   // submit per bucket (bounds the staging ring)
        }

        // Resolve every subset's texSlot once (missing/overflow -> bucket 0 = white).
        for (uint32_t sid = 0; sid < g_staticsSubsets.size(); ++sid) {
            auto it = plan.find(g_staticsSubsetTex[sid]);
            g_staticsSubsets[sid].texSlot = (it != plan.end())
                                          ? ((it->second.bucket << 16) | (it->second.layer & 0xFFFF)) : 0u;
        }

        uint64_t vram = 0;
        for (uint32_t b = 1; b < g_staticsBuckets.size(); ++b) {
            const StaticsTexBucket& bk = g_staticsBuckets[b];
            uint32_t mw = bk.w, mh = bk.h;
            for (uint32_t m = 0; m < bk.mips; ++m) {
                vram += (uint64_t)ddsTightMipBytes(bk.fmt, mw, mh) * bk.count;
                mw = (mw > 1) ? mw >> 1 : 1; mh = (mh > 1) ? mh >> 1 : 1;
            }
        }
        std::printf("[forge][dl] statics textures: %zu buckets, %u uploaded, %u missing, %u overflow, ~%lluMB resident\n",
                    g_staticsBuckets.size() - 1, uploaded, missing, overflow, (unsigned long long)(vram >> 20));
        g_staticsTexReady = true;
        return true;
    }

    // Build the per-scope instance + indirect-arg buffers: collect exterior placements within
    // kStaticsScopeR of camera target T, expand to (instance×subset) grouped by subset, load each
    // referenced texture (bindless), and emit one IndirectDrawIndexArguments per non-empty subset.
    bool buildStaticsScope(Renderer* R, const float T[3]) {
        if (!g_staticsLoaded) { return false; }
        const float r2 = kStaticsScopeR * kStaticsScopeR;

        // Per-subset accumulator of instance rows (80 B each: 16 floats world + 4 floats params).
        std::vector<std::vector<float>> bySubset(g_staticsSubsets.size());
        uint32_t scoped = 0;
        const uint8_t* rec = &g_usageData[g_ws0Off];
        for (uint32_t i = 0; i < g_ws0Count; ++i, rec += 34) {
            uint32_t staticRef; std::memcpy(&staticRef, rec, 4);
            float pos[3], yaw, pitch, roll, scale;
            std::memcpy(pos, rec + 6, 12);
            std::memcpy(&yaw, rec + 18, 4); std::memcpy(&pitch, rec + 22, 4);
            std::memcpy(&roll, rec + 26, 4); std::memcpy(&scale, rec + 30, 4);
            float dx = pos[0] - T[0], dy = pos[1] - T[1];
            if (dx*dx + dy*dy > r2) { continue; }
            if (staticRef >= g_staticsDefs.size()) { continue; }
            ++scoped;

            // transform = Scale * RotZ(-roll) * RotY(-pitch) * RotX(-yaw) * Translate(pos)  (D3DX row-vec).
            float cz=std::cos(-roll),  sz=std::sin(-roll);
            float cy=std::cos(-pitch), sy=std::sin(-pitch);
            float cx=std::cos(-yaw),   sx=std::sin(-yaw);
            float S[16]  = { scale,0,0,0, 0,scale,0,0, 0,0,scale,0, 0,0,0,1 };
            float Rz[16] = { cz,sz,0,0, -sz,cz,0,0, 0,0,1,0, 0,0,0,1 };
            float Ry[16] = { cy,0,-sy,0, 0,1,0,0, sy,0,cy,0, 0,0,0,1 };
            float Rx[16] = { 1,0,0,0, 0,cx,sx,0, 0,-sx,cx,0, 0,0,0,1 };
            float Tm[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, pos[0],pos[1],pos[2],1 };
            float m0[16], m1[16], m2[16], W[16];
            dlMul(S,  Rz, m0);
            dlMul(m0, Ry, m1);
            dlMul(m1, Rx, m2);
            dlMul(m2, Tm, W);                             // world (row-major D3DX)

            const StaticsDefCPU& def = g_staticsDefs[staticRef];
            for (uint32_t k = 0; k < def.numSubsets; ++k) {
                uint32_t sid = def.firstSubset + k;
                // texSlot was resolved once in buildStaticsTextureArrays ((bucket<<16)|layer).
                std::vector<float>& dst = bySubset[sid];
                dst.insert(dst.end(), W, W + 16);
                dst.push_back((float)g_staticsSubsets[sid].texSlot);
                dst.push_back((float)g_staticsSubsets[sid].flags);
                dst.push_back(0.0f); dst.push_back(0.0f);
            }
        }

        // Flatten grouped instances + build one indirect-arg record per non-empty subset.
        std::vector<float> instAll;
        std::vector<IndirectDrawIndexArguments> args;
        uint32_t instBase = 0;
        for (uint32_t sid = 0; sid < g_staticsSubsets.size(); ++sid) {
            const std::vector<float>& v = bySubset[sid];
            if (v.empty()) { continue; }
            uint32_t instCount = (uint32_t)(v.size() / 20);   // 20 floats per instance (80 B)
            instAll.insert(instAll.end(), v.begin(), v.end());
            IndirectDrawIndexArguments a = {};
            a.mIndexCount    = g_staticsSubsets[sid].indexCount;
            a.mInstanceCount = instCount;
            a.mStartIndex    = g_staticsSubsets[sid].ibBase;
            a.mVertexOffset  = g_staticsSubsets[sid].vbBase;
            a.mStartInstance = instBase;
            args.push_back(a);
            instBase += instCount;
        }
        g_staticsInstTotal = instBase;
        g_staticsDrawCount = (uint32_t)args.size();
        std::printf("[forge][dl] statics scope T(%.0f,%.0f,%.0f) r%.0f: %u placements -> %u draw-instances, %u subsets, %zu tex-buckets\n",
                    T[0], T[1], T[2], kStaticsScopeR, scoped, g_staticsInstTotal, g_staticsDrawCount,
                    g_staticsBuckets.empty() ? (size_t)0 : g_staticsBuckets.size() - 1);
        if (g_staticsDrawCount == 0) { return false; }

        BufferLoadDesc ibl = {};
        ibl.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
        ibl.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        ibl.mDesc.mSize = instAll.size() * sizeof(float);
        ibl.mDesc.pName = "staticsInst";
        ibl.pData = instAll.data();
        ibl.ppBuffer = &g_pStaticsInst;
        addResource(&ibl, nullptr);
        BufferLoadDesc adl = {};
        adl.mDesc.mDescriptors = DESCRIPTOR_TYPE_INDIRECT_BUFFER;
        adl.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        adl.mDesc.mSize = args.size() * sizeof(IndirectDrawIndexArguments);
        adl.mDesc.pName = "staticsArgs";
        adl.pData = args.data();
        adl.ppBuffer = &g_pStaticsArgs;
        addResource(&adl, nullptr);
        waitForAllResourceLoads();
        if (!g_pStaticsInst || !g_pStaticsArgs) { std::printf("[forge][dl] statics scope buffer FAILED\n"); return false; }
        return true;
    }

    // Pick the densest 8192-grid cell among exterior placements as the probe camera target
    // (deterministic, dense scene). out[2] = the cell's mean z.
    void pickStaticsTarget(float out[3]) {
        const float G = 8192.0f;
        std::vector<uint64_t> keys; std::vector<uint32_t> cnt;
        std::vector<double> zsum;
        const uint8_t* rec = &g_usageData[g_ws0Off];
        for (uint32_t i = 0; i < g_ws0Count; ++i, rec += 34) {
            float pos[3]; std::memcpy(pos, rec + 6, 12);
            int32_t ix = (int32_t)std::floor(pos[0] / G);
            int32_t iy = (int32_t)std::floor(pos[1] / G);
            uint64_t key = ((uint64_t)(uint32_t)ix << 32) | (uint32_t)iy;
            // linear probe into the small histogram (few thousand cells)
            uint32_t j = 0; for (; j < keys.size(); ++j) { if (keys[j] == key) { break; } }
            if (j == keys.size()) { keys.push_back(key); cnt.push_back(0); zsum.push_back(0.0); }
            cnt[j] += 1; zsum[j] += pos[2];
        }
        uint32_t best = 0;
        for (uint32_t j = 1; j < cnt.size(); ++j) { if (cnt[j] > cnt[best]) { best = j; } }
        int32_t ix = (int32_t)(keys[best] >> 32), iy = (int32_t)(uint32_t)keys[best];
        out[0] = (ix + 0.5f) * G;
        out[1] = (iy + 0.5f) * G;
        out[2] = (float)(zsum[best] / (cnt[best] ? cnt[best] : 1));
        std::printf("[forge][dl] densest cell (%d,%d) %u placements -> target (%.0f,%.0f,%.0f)\n",
                    ix, iy, cnt[best], out[0], out[1], out[2]);
    }

    // --- tiny row-major (D3D LH, v*M) matrix helpers for the probe's synthetic camera ---
    void dlNorm3(float v[3]) {
        float l = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if (l > 1e-8f) { v[0] /= l; v[1] /= l; v[2] /= l; }
    }
    void dlCross(const float a[3], const float b[3], float out[3]) {
        out[0] = a[1]*b[2] - a[2]*b[1];
        out[1] = a[2]*b[0] - a[0]*b[2];
        out[2] = a[0]*b[1] - a[1]*b[0];
    }
    float dlDot3(const float a[3], const float b[3]) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
    void dlLookAtLH(const float eye[3], const float at[3], const float up[3], float m[16]) {
        float z[3] = { at[0]-eye[0], at[1]-eye[1], at[2]-eye[2] }; dlNorm3(z);
        float x[3]; dlCross(up, z, x); dlNorm3(x);
        float y[3]; dlCross(z, x, y);
        m[0]=x[0]; m[1]=y[0]; m[2]=z[0]; m[3]=0;
        m[4]=x[1]; m[5]=y[1]; m[6]=z[1]; m[7]=0;
        m[8]=x[2]; m[9]=y[2]; m[10]=z[2]; m[11]=0;
        m[12]=-dlDot3(x,eye); m[13]=-dlDot3(y,eye); m[14]=-dlDot3(z,eye); m[15]=1;
    }
    void dlPerspLH(float yScale, float aspect, float zn, float zf, float m[16]) {
        for (int i = 0; i < 16; ++i) { m[i] = 0; }
        m[0] = yScale / aspect;
        m[5] = yScale;
        m[10] = zf / (zf - zn);
        m[11] = 1.0f;
        m[14] = -zn * zf / (zf - zn);
    }
    void dlMul(const float a[16], const float b[16], float out[16]) {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                out[i*4+j] = a[i*4+0]*b[0*4+j] + a[i*4+1]*b[1*4+j]
                           + a[i*4+2]*b[2*4+j] + a[i*4+3]*b[3*4+j];
    }

    // Standalone --forge-dl probe: bring Forge up, build the opaque path's descriptor sets + the
    // land pipeline, load distant land from disk, render it from a synthetic camera over the
    // terrain bounds into the owned RT, read back, and report coverage. No MW / IPC — proves the
    // host-owned DL loader + pipeline + atlas in isolation ("DL renders without Morrowind").
    bool renderDLProbe(bool withStatics) {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        std::printf("[forge][dl] --forge-%s probe (cwd must be morrowind64)\n", withStatics ? "statics" : "dl");
        const unsigned W = 1280, H = 720;
        if (!init(W, H, 1, 8)) { std::printf("[forge][dl] init FAILED\n"); return false; }
        Renderer* R = g_live.pRenderer;
        if (!buildOpaquePath(R, g_live.width, g_live.height)) {
            std::printf("[forge][dl] buildOpaquePath FAILED\n"); shutdown(); return false;
        }
        if (!buildLandPath(R))     { shutdown(); return false; }
        if (!loadDistantLand(R))   { shutdown(); return false; }
        if (g_landMeshCount == 0)  { std::printf("[forge][dl] no land meshes\n"); shutdown(); return false; }

        // Phase 1b: load the statics library + build the scoped instance/indirect buffers around the
        // densest exterior cell (the probe camera then frames that cluster).
        float staticsT[3] = { 0, 0, 0 };
        if (withStatics) {
            if (!buildStaticsPath(R))         { shutdown(); return false; }
            if (!loadDistantStatics(R))       { shutdown(); return false; }
            if (!buildStaticsTextureArrays(R)){ shutdown(); return false; }
            pickStaticsTarget(staticsT);
            if (!buildStaticsScope(R, staticsT)) { shutdown(); return false; }
        }

        // Terrain bounds from the loaded sphere centres.
        float mnX=3.4e38f,mnY=3.4e38f,mnZ=3.4e38f,mxX=-3.4e38f,mxY=-3.4e38f,mxZ=-3.4e38f;
        for (uint32_t i=0;i<g_landMeshCount;++i){
            const LandMeshGPU& m=g_landMeshes[i];
            mnX = (m.cx-m.r<mnX)?m.cx-m.r:mnX; mxX=(m.cx+m.r>mxX)?m.cx+m.r:mxX;
            mnY = (m.cy-m.r<mnY)?m.cy-m.r:mnY; mxY=(m.cy+m.r>mxY)?m.cy+m.r:mxY;
            mnZ = (m.cz-m.r<mnZ)?m.cz-m.r:mnZ; mxZ=(m.cz+m.r>mxZ)?m.cz+m.r:mxZ;
        }
        float cx=0.5f*(mnX+mxX), cy=0.5f*(mnY+mxY), cz=0.5f*(mnZ+mxZ);
        float extX=mxX-mnX, extY=mxY-mnY, ext=(extX>extY)?extX:extY;
        std::printf("[forge][dl] bounds c(%.0f,%.0f,%.0f) ext %.0f z[%.0f,%.0f]\n", cx,cy,cz,ext,mnZ,mxZ);

        // Synthetic camera. Land-only: high above the terrain top near the centre, looking across it.
        // Statics: a ground-level oblique view of the dense cluster so the instanced statics fill the
        // frame (land still draws behind, sharing depth). MW is Z-up.
        float eye[3], at[3];
        if (withStatics) {
            eye[0] = staticsT[0] - 0.9f*kStaticsScopeR;
            eye[1] = staticsT[1] - 0.3f*kStaticsScopeR;
            eye[2] = staticsT[2] + 0.55f*kStaticsScopeR;
            at[0]  = staticsT[0]; at[1] = staticsT[1]; at[2] = staticsT[2] + 600.0f;
        } else {
            eye[0] = cx - 0.35f*ext; eye[1] = cy; eye[2] = mxZ + 0.10f*ext + 4000.0f;
            at[0]  = cx + 0.25f*ext; at[1] = cy; at[2] = cz;
        }
        float up[3]  = { 0.0f, 0.0f, 1.0f };
        float zn=8.0f, zf=2.0f*ext + 40000.0f;
        const float yScale = 1.732051f;        // 1/tan(30deg) → 60° vertical FOV
        float view[16], proj[16], vp[16];
        dlLookAtLH(eye, at, up, view);
        dlPerspLH(yScale, (float)W/(float)H, zn, zf, proj);
        dlMul(view, proj, vp);
        // reverse-Z munge (matches renderScene): col2 := col3 - col2 on the row-major matrix.
        vp[2]=vp[3]-vp[2]; vp[6]=vp[7]-vp[6]; vp[10]=vp[11]-vp[10]; vp[14]=vp[15]-vp[14];

        // Fill gFrameData directly (opaque.srt.h FrameData layout, float indices).
        float* fd = (float*)g_live.pFrameCbv->pCpuMappedAddress;
        std::memcpy(fd, vp, 16*sizeof(float));
        float sun[3] = { 0.4f, 0.2f, -0.9f }; dlNorm3(sun);          // world sun TRAVEL dir (to-sun = -sun)
        fd[16]=sun[0]; fd[17]=sun[1]; fd[18]=sun[2]; fd[19]=0;       // sunDir
        fd[20]=1.0f;   fd[21]=0.96f;  fd[22]=0.86f; fd[23]=0;        // sunCol
        fd[24]=0.34f;  fd[25]=0.38f;  fd[26]=0.46f; fd[27]=0;        // ambCol
        fd[28]=0.60f;  fd[29]=0.66f;  fd[30]=0.78f; fd[31]=0;        // fogColNear
        fd[32]=0.30f*zf; fd[33]=0.95f*zf; fd[34]=0; fd[35]=0;        // fogParams (start, end)
        fd[36]=eye[0]; fd[37]=eye[1]; fd[38]=eye[2]; fd[39]=0;       // eyePos
        fd[40]=0; fd[41]=1.0f/(float)W; fd[42]=1.0f/(float)H; fd[43]=0; // debugParams
        fd[44]=1; fd[45]=1; fd[46]=1; fd[47]=1;                      // dbgScales
        fd[48]=(float)kLandBaseSlot; fd[49]=(float)kLandNormalSlot;  // lodParams.xy
        fd[50]=(float)kLandDetailSlot; fd[51]=7168.0f;               // lodParams.zw (detail slot, nearViewRange)
        fd[52]=0.34f; fd[53]=0.38f; fd[54]=0.46f; fd[55]=0;          // lodSunAmb (= ambCol for 1a)
        fd[56]=0; fd[57]=0; fd[58]=0; fd[59]=0;                      // lodEye = 0 (probe camera is absolute)

        // Readback buffer (mirror drawTriangleAndVerify).
        const uint32_t rowAlign = (R->pGpu->mUploadBufferTextureRowAlignment > 1u) ? R->pGpu->mUploadBufferTextureRowAlignment : 1u;
        const uint32_t texAlign = (R->pGpu->mUploadBufferTextureAlignment > 1u) ? R->pGpu->mUploadBufferTextureAlignment : 1u;
        const uint32_t rowPitch = roundUp(W*4u, rowAlign);
        const uint64_t bufSize  = roundUp64((uint64_t)rowPitch*H, texAlign);
        BufferLoadDesc rbd = {};
        rbd.mDesc.mSize = bufSize;
        rbd.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_TO_CPU;
        rbd.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        rbd.mDesc.mStartState = RESOURCE_STATE_COPY_DEST;
        rbd.mDesc.mQueueType = QUEUE_TYPE_TRANSFER;
        Buffer* pReadback = nullptr;
        rbd.ppBuffer = &pReadback;
        addResource(&rbd, nullptr);
        waitForAllResourceLoads();

        resetCmdPool(R, g_live.pCmdPool);
        beginCmd(g_live.pCmd);
        BindRenderTargetsDesc bind = {};
        bind.mRenderTargetCount = 1;
        bind.mRenderTargets[0] = { g_live.pRT, LOAD_ACTION_CLEAR };
        bind.mDepthStencil = { g_live.pDepth, LOAD_ACTION_CLEAR };
        cmdBindRenderTargets(g_live.pCmd, &bind);
        cmdSetViewport(g_live.pCmd, 0.0f, 0.0f, (float)W, (float)H, 0.0f, 1.0f);
        cmdSetScissor(g_live.pCmd, 0, 0, W, H);
        cmdBindPipeline(g_live.pCmd, g_pLandPipeline);
        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);
        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerBatchSet);
        uint64_t drawnTris = 0;
        for (uint32_t i=0;i<g_landMeshCount;++i){
            LandMeshGPU& m = g_landMeshes[i];
            Buffer*  vbs[1]     = { m.vb };
            uint32_t strides[1] = { 16 };
            cmdBindVertexBuffer(g_live.pCmd, 1, vbs, strides, nullptr);
            cmdBindIndexBuffer(g_live.pCmd, m.ib, m.large ? INDEX_TYPE_UINT32 : INDEX_TYPE_UINT16, 0);
            cmdDrawIndexedInstanced(g_live.pCmd, m.indexCount, 0, 1, 0, 0);
            drawnTris += m.indexCount / 3;
        }

        // Statics: one cmdExecuteIndirect over the mega VB/IB + per-instance stream. The arg buffer
        // holds one IndirectDrawIndexArguments per scoped subset (StartInstanceLocation selects its
        // instance run); StartIndexLocation/BaseVertexLocation select its slice of the mega buffers.
        if (withStatics && g_staticsDrawCount > 0) {
            cmdBindPipeline(g_live.pCmd, g_pStaticsPipeline);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerBatchSet);
            Buffer*  svbs[2]     = { g_pStaticsVB, g_pStaticsInst };
            uint32_t sstrides[2] = { 20, kStaticsInstStride };
            cmdBindVertexBuffer(g_live.pCmd, 2, svbs, sstrides, nullptr);
            cmdBindIndexBuffer(g_live.pCmd, g_pStaticsIB, INDEX_TYPE_UINT16, 0);
            cmdExecuteIndirect(g_live.pCmd, INDIRECT_DRAW_INDEX, g_staticsDrawCount, g_pStaticsArgs, 0, nullptr, 0);
        }

        cmdBindRenderTargets(g_live.pCmd, nullptr);
        RenderTargetBarrier rtb = {};
        rtb.pRenderTarget = g_live.pRT;
        rtb.mCurrentState = RESOURCE_STATE_RENDER_TARGET;
        rtb.mNewState = RESOURCE_STATE_COPY_SOURCE;
        cmdResourceBarrier(g_live.pCmd, 0, nullptr, 0, nullptr, 1, &rtb);
        endCmd(g_live.pCmd);

        QueueSubmitDesc submit = {};
        submit.mCmdCount = 1;
        submit.ppCmds = &g_live.pCmd;
        submit.pSignalFence = g_live.pFence;
        submit.mSubmitDone = true;
        queueSubmit(g_live.pQueue, &submit);
        waitForFences(R, 1, &g_live.pFence);

        TextureCopyDesc copyDesc = {};
        copyDesc.pTexture = g_live.pRT->pTexture;
        copyDesc.pBuffer = pReadback;
        copyDesc.mTextureState = RESOURCE_STATE_COPY_SOURCE;
        copyDesc.mQueueType = QUEUE_TYPE_GRAPHICS;
        SyncToken tok = {};
        copyResource(&copyDesc, &tok);
        waitForToken(&tok);

        bool ok = false;
        const uint8_t* px = (const uint8_t*)pReadback->pCpuMappedAddress;
        if (px) {
            uint64_t nonClear = 0;
            for (uint32_t y=0;y<H;y+=16) {
                for (uint32_t x=0;x<W;x+=16) {
                    const uint8_t* s=&px[(uint64_t)y*rowPitch + (uint64_t)x*4];
                    if (s[0]|s[1]|s[2]) { ++nonClear; }
                }
            }
            const uint8_t* c=&px[(uint64_t)(H/2)*rowPitch + (uint64_t)(W/2)*4];
            std::printf("[forge][dl] drew %u land meshes / %llu tris + %u static draw-instances (%u subsets); centre=%u,%u,%u,%u; non-clear samples=%llu\n",
                        g_landMeshCount, (unsigned long long)drawnTris, g_staticsInstTotal, g_staticsDrawCount,
                        c[0],c[1],c[2],c[3], (unsigned long long)nonClear);
            ok = nonClear > 0;
            std::printf("[forge][dl] rendered: %s\n", ok ? "YES" : "NO");
            // Dump the RT to an uncompressed TGA (forge_dl.tga in cwd) so the terrain look can be
            // eyeballed: atlas UV fidelity, lighting, fog. TGA stores BGR == our BGRA source order.
            const char* tgaName = withStatics ? "forge_statics.tga" : "forge_dl.tga";
            std::FILE* tf = std::fopen(tgaName, "wb");
            if (tf) {
                uint8_t hdr[18] = {0};
                hdr[2]  = 2;                                   // uncompressed true-color
                hdr[12] = (uint8_t)(W & 0xFF); hdr[13] = (uint8_t)(W >> 8);
                hdr[14] = (uint8_t)(H & 0xFF); hdr[15] = (uint8_t)(H >> 8);
                hdr[16] = 24;                                  // bpp
                hdr[17] = 0x20;                                // top-left origin
                std::fwrite(hdr, 1, 18, tf);
                std::vector<uint8_t> rowBuf((size_t)W * 3);
                for (uint32_t y = 0; y < H; ++y) {
                    const uint8_t* row = &px[(uint64_t)y * rowPitch];
                    for (uint32_t x = 0; x < W; ++x) {
                        rowBuf[x*3+0] = row[x*4+0];            // B
                        rowBuf[x*3+1] = row[x*4+1];            // G
                        rowBuf[x*3+2] = row[x*4+2];            // R
                    }
                    std::fwrite(rowBuf.data(), 1, rowBuf.size(), tf);
                }
                std::fclose(tf);
                std::printf("[forge][dl] wrote %s (%ux%u)\n", tgaName, W, H);
            }
        } else {
            std::printf("[forge][dl] readback not mapped\n");
        }
        removeResource(pReadback);
        shutdown();
        return ok;
    }

    bool renderDistantLandProbe()    { return renderDLProbe(false); }
    bool renderDistantStaticsProbe() { return renderDLProbe(true);  }

    // ---- Standalone interactive world viewer (--forge-view) ---------------------------------------
    // A real Win32 window + Forge swapchain showing the host-owned world (distant land + statics) that
    // you fly around with WASD + arrow keys. NO Morrowind, NO IPC, NO client — so it isolates Forge
    // RENDERER bugs from INTEGRATION bugs (a hang here = Forge; a hang only in-game = the seam/IPC)
    // and iterates far faster than launching the game. Renders into the existing headless pRT (reusing
    // the probe's land+statics record), then CopyResource pRT -> swapchain backbuffer -> present.
    // tasks/forge-viewer.md. V1 = land+statics+fly-cam; sky (V2) and water (V3) layer on the loop.
    static bool s_viewerQuit = false;
    static LRESULT CALLBACK viewerWndProc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
        switch (msg) {
            case WM_CLOSE: case WM_DESTROY: s_viewerQuit = true; return 0;
            case WM_KEYDOWN: if (wp == VK_ESCAPE) { s_viewerQuit = true; } return 0;
        }
        return DefWindowProcW(h, msg, wp, lp);
    }

    bool worldViewer() {
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        const unsigned W = 1280, H = 720;
        std::printf("[forge][view] world viewer (cwd must be morrowind64) — WASD+arrows fly, Shift fast, ESC quit\n");

        // --- Win32 window ---
        HINSTANCE hInst = GetModuleHandleW(nullptr);
        WNDCLASSEXW wc = {}; wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = viewerWndProc; wc.hInstance = hInst;
        wc.lpszClassName = L"ForgeWorldViewer"; wc.hCursor = LoadCursorW(nullptr, (LPCWSTR)IDC_ARROW);
        RegisterClassExW(&wc);
        RECT wr = { 0, 0, (LONG)W, (LONG)H };
        AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
        HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"Forge World Viewer  (WASD + arrows, Shift = fast, ESC = quit)",
            WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, wr.right - wr.left, wr.bottom - wr.top,
            nullptr, nullptr, hInst, nullptr);
        if (!hwnd) { std::printf("[forge][view] CreateWindow FAILED\n"); return false; }
        ShowWindow(hwnd, SW_SHOW); s_viewerQuit = false;

        // --- init renderer (single-sample so the pRT->backbuffer copy is 1:1) + load land + statics ---
        if (!init(W, H, 1, 8)) { std::printf("[forge][view] init FAILED\n"); return false; }
        Renderer* R = g_live.pRenderer;
        if (!buildOpaquePath(R, g_live.width, g_live.height) || !buildLandPath(R) || !loadDistantLand(R)) {
            std::printf("[forge][view] DL load FAILED\n"); shutdown(); return false;
        }
        float staticsT[3] = { 0, 0, 0 };
        bool haveStatics = buildStaticsPath(R) && loadDistantStatics(R) && buildStaticsTextureArrays(R);
        if (haveStatics) { pickStaticsTarget(staticsT); haveStatics = buildStaticsScope(R, staticsT); }
        std::printf("[forge][view] loaded: %u land meshes, statics=%d (%u draws)\n",
                    g_landMeshCount, (int)haveStatics, g_staticsDrawCount);

        // --- swapchain on the window + present sync ---
        SwapChain* pSwap = nullptr;
        SwapChainDesc scd = {};
        scd.mWindowHandle.type = WINDOW_HANDLE_TYPE_WIN32;
        scd.mWindowHandle.window = hwnd;
        scd.ppPresentQueues = &g_live.pQueue; scd.mPresentQueueCount = 1;
        scd.mImageCount = 2; scd.mWidth = W; scd.mHeight = H;
        scd.mColorFormat = TinyImageFormat_B8G8R8A8_UNORM;
        scd.mEnableVsync = true;
        addSwapChain(R, &scd, &pSwap);
        if (!pSwap) { std::printf("[forge][view] addSwapChain FAILED\n"); shutdown(); return false; }
        Semaphore* pImgSem = nullptr; Semaphore* pRenderSem = nullptr;
        initSemaphore(R, &pImgSem); initSemaphore(R, &pRenderSem);

        // --- terrain bounds → camera start (the probe's land overview), then free-fly ---
        float mnX=3.4e38f,mnY=3.4e38f,mnZ=3.4e38f,mxX=-3.4e38f,mxY=-3.4e38f,mxZ=-3.4e38f;
        for (uint32_t i=0;i<g_landMeshCount;++i){ const LandMeshGPU& m=g_landMeshes[i];
            mnX=(m.cx-m.r<mnX)?m.cx-m.r:mnX; mxX=(m.cx+m.r>mxX)?m.cx+m.r:mxX;
            mnY=(m.cy-m.r<mnY)?m.cy-m.r:mnY; mxY=(m.cy+m.r>mxY)?m.cy+m.r:mxY;
            mnZ=(m.cz-m.r<mnZ)?m.cz-m.r:mnZ; mxZ=(m.cz+m.r>mxZ)?m.cz+m.r:mxZ; }
        float cx=0.5f*(mnX+mxX), cy=0.5f*(mnY+mxY), cz=0.5f*(mnZ+mxZ);
        float ext=((mxX-mnX)>(mxY-mnY))?(mxX-mnX):(mxY-mnY);
        float eye[3] = { cx - 0.35f*ext, cy, mxZ + 0.10f*ext + 4000.0f };
        // Aim at the terrain centre.
        float aim[3] = { cx - eye[0], cy - eye[1], cz - eye[2] };
        float aimLen = std::sqrt(aim[0]*aim[0]+aim[1]*aim[1]+aim[2]*aim[2]) + 1e-6f;
        aim[0]/=aimLen; aim[1]/=aimLen; aim[2]/=aimLen;
        float yaw = std::atan2(aim[1], aim[0]);
        float pitch = std::asin(aim[2] < -1.0f ? -1.0f : (aim[2] > 1.0f ? 1.0f : aim[2]));
        const float zn = 8.0f, zf = 2.0f*ext + 40000.0f;
        const float yScale = 1.732051f;                       // 60° vertical FOV
        const float kPi = 3.14159265f;
        LARGE_INTEGER qf, t0; QueryPerformanceFrequency(&qf); QueryPerformanceCounter(&t0);

        // --- render loop ---
        while (!s_viewerQuit) {
            MSG m;
            while (PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&m); DispatchMessageW(&m); }
            if (s_viewerQuit) break;

            // dt for frame-rate-independent movement.
            LARGE_INTEGER t1; QueryPerformanceCounter(&t1);
            float dt = (float)(t1.QuadPart - t0.QuadPart) / (float)qf.QuadPart; t0 = t1;
            if (dt > 0.1f) dt = 0.1f;
            auto down = [](int vk){ return (GetAsyncKeyState(vk) & 0x8000) != 0; };
            float moveSpd = ext * 0.10f * dt * (down(VK_SHIFT) ? 6.0f : 1.0f);   // ~world-scaled
            float lookSpd = 1.4f * dt;
            if (down(VK_LEFT))  yaw   -= lookSpd;
            if (down(VK_RIGHT)) yaw   += lookSpd;
            if (down(VK_UP))    pitch += lookSpd;
            if (down(VK_DOWN))  pitch -= lookSpd;
            if (pitch >  1.55f) pitch =  1.55f;
            if (pitch < -1.55f) pitch = -1.55f;
            float cp = std::cos(pitch), sp = std::sin(pitch), cyw = std::cos(yaw), syw = std::sin(yaw);
            float fwd[3] = { cp*cyw, cp*syw, sp };
            float rgt[3] = { syw, -cyw, 0.0f };               // right = fwd × worldUp(Z), normalized in XY
            if (down('W')) { eye[0]+=fwd[0]*moveSpd; eye[1]+=fwd[1]*moveSpd; eye[2]+=fwd[2]*moveSpd; }
            if (down('S')) { eye[0]-=fwd[0]*moveSpd; eye[1]-=fwd[1]*moveSpd; eye[2]-=fwd[2]*moveSpd; }
            if (down('D')) { eye[0]+=rgt[0]*moveSpd; eye[1]+=rgt[1]*moveSpd; }
            if (down('A')) { eye[0]-=rgt[0]*moveSpd; eye[1]-=rgt[1]*moveSpd; }
            if (down(VK_SPACE))   eye[2]+=moveSpd;
            if (down(VK_CONTROL)) eye[2]-=moveSpd;
            (void)kPi;

            // viewProj from the camera (reverse-Z munge matches renderScene/probe).
            float at[3] = { eye[0]+fwd[0], eye[1]+fwd[1], eye[2]+fwd[2] };
            float up[3] = { 0.0f, 0.0f, 1.0f };
            float view[16], proj[16], vp[16];
            dlLookAtLH(eye, at, up, view);
            dlPerspLH(yScale, (float)W/(float)H, zn, zf, proj);
            dlMul(view, proj, vp);
            vp[2]=vp[3]-vp[2]; vp[6]=vp[7]-vp[6]; vp[10]=vp[11]-vp[10]; vp[14]=vp[15]-vp[14];

            float* fd = (float*)g_live.pFrameCbv->pCpuMappedAddress;
            std::memcpy(fd, vp, 16*sizeof(float));
            float sun[3] = { 0.4f, 0.2f, -0.9f }; dlNorm3(sun);
            fd[16]=sun[0]; fd[17]=sun[1]; fd[18]=sun[2]; fd[19]=0;
            fd[20]=1.0f; fd[21]=0.96f; fd[22]=0.86f; fd[23]=0;
            fd[24]=0.34f; fd[25]=0.38f; fd[26]=0.46f; fd[27]=0;
            fd[28]=0.60f; fd[29]=0.66f; fd[30]=0.78f; fd[31]=0;
            fd[32]=0.30f*zf; fd[33]=0.95f*zf; fd[34]=0; fd[35]=0;
            fd[36]=eye[0]; fd[37]=eye[1]; fd[38]=eye[2]; fd[39]=0;
            fd[40]=0; fd[41]=1.0f/(float)W; fd[42]=1.0f/(float)H; fd[43]=0;
            fd[44]=1; fd[45]=1; fd[46]=1; fd[47]=1;
            fd[48]=(float)kLandBaseSlot; fd[49]=(float)kLandNormalSlot;
            fd[50]=(float)kLandDetailSlot; fd[51]=7168.0f;
            fd[52]=0.34f; fd[53]=0.38f; fd[54]=0.46f; fd[55]=0;
            fd[56]=0; fd[57]=0; fd[58]=0; fd[59]=0;                // lodEye = 0 (camera absolute)

            uint32_t idx = 0;
            acquireNextImage(R, pSwap, pImgSem, nullptr, &idx);
            RenderTarget* bb = pSwap->ppRenderTargets[idx];

            resetCmdPool(R, g_live.pCmdPool);
            beginCmd(g_live.pCmd);
            BindRenderTargetsDesc bind = {};
            bind.mRenderTargetCount = 1;
            bind.mRenderTargets[0] = { g_live.pRT, LOAD_ACTION_CLEAR };
            bind.mDepthStencil = { g_live.pDepth, LOAD_ACTION_CLEAR };
            cmdBindRenderTargets(g_live.pCmd, &bind);
            cmdSetViewport(g_live.pCmd, 0.0f, 0.0f, (float)W, (float)H, 0.0f, 1.0f);
            cmdSetScissor(g_live.pCmd, 0, 0, W, H);
            cmdBindPipeline(g_live.pCmd, g_pLandPipeline);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerBatchSet);
            for (uint32_t i=0;i<g_landMeshCount;++i){ LandMeshGPU& mm=g_landMeshes[i];
                Buffer* vbs[1]={mm.vb}; uint32_t st[1]={16};
                cmdBindVertexBuffer(g_live.pCmd, 1, vbs, st, nullptr);
                cmdBindIndexBuffer(g_live.pCmd, mm.ib, mm.large?INDEX_TYPE_UINT32:INDEX_TYPE_UINT16, 0);
                cmdDrawIndexedInstanced(g_live.pCmd, mm.indexCount, 0, 1, 0, 0);
            }
            if (haveStatics && g_staticsDrawCount > 0) {
                cmdBindPipeline(g_live.pCmd, g_pStaticsPipeline);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
                cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerBatchSet);
                Buffer* svbs[2]={g_pStaticsVB, g_pStaticsInst}; uint32_t sst[2]={20, kStaticsInstStride};
                cmdBindVertexBuffer(g_live.pCmd, 2, svbs, sst, nullptr);
                cmdBindIndexBuffer(g_live.pCmd, g_pStaticsIB, INDEX_TYPE_UINT16, 0);
                cmdExecuteIndirect(g_live.pCmd, INDIRECT_DRAW_INDEX, g_staticsDrawCount, g_pStaticsArgs, 0, nullptr, 0);
            }
            cmdBindRenderTargets(g_live.pCmd, nullptr);

            // Copy the rendered pRT into the acquired backbuffer, then present.
            RenderTargetBarrier cpb[2] = {};
            cpb[0].pRenderTarget = g_live.pRT; cpb[0].mCurrentState = RESOURCE_STATE_RENDER_TARGET; cpb[0].mNewState = RESOURCE_STATE_COPY_SOURCE;
            cpb[1].pRenderTarget = bb;         cpb[1].mCurrentState = RESOURCE_STATE_PRESENT;       cpb[1].mNewState = RESOURCE_STATE_COPY_DEST;
            cmdResourceBarrier(g_live.pCmd, 0, nullptr, 0, nullptr, 2, cpb);
            g_live.pCmd->mDx.pCmdList->CopyResource(bb->pTexture->mDx.pResource, g_live.pRT->pTexture->mDx.pResource);
            cpb[0].mCurrentState = RESOURCE_STATE_COPY_SOURCE; cpb[0].mNewState = RESOURCE_STATE_RENDER_TARGET;
            cpb[1].mCurrentState = RESOURCE_STATE_COPY_DEST;   cpb[1].mNewState = RESOURCE_STATE_PRESENT;
            cmdResourceBarrier(g_live.pCmd, 0, nullptr, 0, nullptr, 2, cpb);
            endCmd(g_live.pCmd);

            QueueSubmitDesc sub = {};
            sub.mCmdCount = 1; sub.ppCmds = &g_live.pCmd;
            sub.pSignalFence = g_live.pFence;
            sub.mWaitSemaphoreCount = 1; sub.ppWaitSemaphores = &pImgSem;
            sub.mSignalSemaphoreCount = 1; sub.ppSignalSemaphores = &pRenderSem;
            queueSubmit(g_live.pQueue, &sub);
            QueuePresentDesc pres = {};
            pres.pSwapChain = pSwap; pres.mIndex = (uint8_t)idx;
            pres.mWaitSemaphoreCount = 1; pres.ppWaitSemaphores = &pRenderSem;
            pres.mSubmitDone = true;
            queuePresent(g_live.pQueue, &pres);
            waitForFences(R, 1, &g_live.pFence);
        }

        waitQueueIdle(g_live.pQueue);
        exitSemaphore(R, pImgSem); exitSemaphore(R, pRenderSem);
        removeSwapChain(R, pSwap);
        DestroyWindow(hwnd);
        UnregisterClassW(wc.lpszClassName, hInst);
        shutdown();
        std::printf("[forge][view] viewer closed\n");
        return true;
    }

    // ===================== Phase 1a/1b LIVE distant land (wired into renderScene) =============

    // Latch this frame's realEye + exterior gate (called from renderScene's lighting block, which
    // sits earlier in the file than the DL globals).
    void dlSetFrameEye(float x, float y, float z, bool exterior) {
        g_dlEye[0] = x; g_dlEye[1] = y; g_dlEye[2] = z;
        g_dlExterior = exterior;
    }

    // Per-frame GPU-spike log (from renderScene), reads the DL draw counts living below it.
    void dlLogGpuSlow(double gpuMs, double recMs, unsigned drawn) {
        LOG::logline(">> [gpu-slow] gpu=%.1fms record=%.1fms drawn=%u dl-land=%u dl-inst=%u dl-subsets=%u",
                     gpuMs, recMs, drawn, g_liveLastLand, g_liveLastInst, g_liveLastSubsets);
    }

    // Per-300-frame DL heartbeat (from renderScene's heartbeat block).
    void dlLogHeartbeat() {
        if (!g_dlLiveInit) { return; }
        LOG::logline(">> [forge-hb][dl] exterior=%d land=%u/%u static-instances=%u subsets=%u tex-buckets=%zu",
                     (int)g_dlExterior, g_liveLastLand, g_landMeshCount, g_liveLastInst,
                     g_liveLastSubsets, g_staticsBuckets.empty() ? 0 : g_staticsBuckets.size() - 1);
    }

    // Build a uniform grid (cell = one MW cell) over the resident exterior placements: gridCell ->
    // [instance indices into g_usageData ws0]. One-time; the per-frame cull then visits only the
    // grid cells whose AABB intersects the frustum. Each cell's AABB bounds the member placement
    // POSITIONS (padded for per-static extent in the coarse reject; the per-instance test is exact).
    void buildStaticsGrid() {
        g_liveGrid.clear();
        if (!g_staticsLoaded || g_ws0Count == 0) { return; }
        // Precompute the canonical per-instance cull struct (same iteration as the grid). Mirrors the
        // exact tier/effR rule the old per-frame hot loop used (dlshare.h:184) so survivors are
        // byte-identical; only the per-frame distance + frustum tests + -eye stay in the loop.
        g_cullInst.assign((size_t)g_ws0Count, GpuCullInstance{});
        const float gFarMin  = Configuration.DL.FarStaticMinSize;
        const float gVfarMin = Configuration.DL.VeryFarStaticMinSize;
        std::unordered_map<uint64_t, uint32_t> cellMap;
        cellMap.reserve(4096);
        const uint8_t* rec = &g_usageData[g_ws0Off];
        for (uint32_t i = 0; i < g_ws0Count; ++i, rec += 34) {
            float pos[3]; std::memcpy(pos, rec + 6, 12);
            GpuCullInstance& gi = g_cullInst[i];
            {
                uint32_t staticRef; std::memcpy(&staticRef, rec, 4);
                float yaw, pitch, roll, scale;
                std::memcpy(&yaw, rec + 18, 4); std::memcpy(&pitch, rec + 22, 4);
                std::memcpy(&roll, rec + 26, 4); std::memcpy(&scale, rec + 30, 4);
                // Absolute world matrix (S·Rz·Ry·Rx·T) — same math as the old hot loop, minus -eye.
                float cz=std::cos(-roll),  sz=std::sin(-roll);
                float cyf=std::cos(-pitch),syf=std::sin(-pitch);
                float cxf=std::cos(-yaw),  sxf=std::sin(-yaw);
                float S[16]  = { scale,0,0,0, 0,scale,0,0, 0,0,scale,0, 0,0,0,1 };
                float Rz[16] = { cz,sz,0,0, -sz,cz,0,0, 0,0,1,0, 0,0,0,1 };
                float Ry[16] = { cyf,0,-syf,0, 0,1,0,0, syf,0,cyf,0, 0,0,0,1 };
                float Rx[16] = { 1,0,0,0, 0,cxf,sxf,0, 0,-sxf,cxf,0, 0,0,0,1 };
                float Tm[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, pos[0],pos[1],pos[2],1 };
                float m0[16], m1[16], m2[16];
                dlMul(S,  Rz, m0); dlMul(m0, Ry, m1); dlMul(m1, Rx, m2); dlMul(m2, Tm, gi.world);
                gi.posX = pos[0]; gi.posY = pos[1]; gi.posZ = pos[2];
                // tier/effR/subset range — precomputed (constant per session: type, size, DL config).
                if (staticRef >= g_staticsDefs.size()) { gi.rangeEndIdx = 0xFFFFFFFFu; }
                else {
                    const StaticsDefCPU& def = g_staticsDefs[staticRef];
                    gi.effR = def.radius * scale;
                    gi.firstSubset = def.firstSubset; gi.numSubsets = def.numSubsets;
                    float tierR = (def.type == DL_STATIC_BUILDING) ? gi.effR * 2.0f : gi.effR;
                    switch (def.type) {
                        case DL_STATIC_GRASS:    gi.rangeEndIdx = 0xFFFFFFFFu; break;  // skip
                        case DL_STATIC_NEAR:     gi.rangeEndIdx = 0; break;
                        case DL_STATIC_FAR:      gi.rangeEndIdx = 1; break;
                        case DL_STATIC_VERY_FAR: gi.rangeEndIdx = 2; break;
                        default: gi.rangeEndIdx = (tierR <= gFarMin) ? 0 : (tierR <= gVfarMin ? 1 : 2); break;
                    }
                }
            }
            int32_t ix = (int32_t)std::floor(pos[0] / kLiveGridCell);
            int32_t iy = (int32_t)std::floor(pos[1] / kLiveGridCell);
            uint64_t key = ((uint64_t)(uint32_t)ix << 32) | (uint32_t)iy;
            auto it = cellMap.find(key);
            uint32_t ci;
            if (it == cellMap.end()) {
                ci = (uint32_t)g_liveGrid.size();
                cellMap.emplace(key, ci);
                LiveGridCell c;
                c.minx = c.miny = c.minz =  3.4e38f;
                c.maxx = c.maxy = c.maxz = -3.4e38f;
                g_liveGrid.push_back(std::move(c));
            } else {
                ci = it->second;
            }
            LiveGridCell& c = g_liveGrid[ci];
            c.inst.push_back(i);
            c.minx = std::min(c.minx, pos[0]); c.maxx = std::max(c.maxx, pos[0]);
            c.miny = std::min(c.miny, pos[1]); c.maxy = std::max(c.maxy, pos[1]);
            c.minz = std::min(c.minz, pos[2]); c.maxz = std::max(c.maxz, pos[2]);
        }
        std::printf("[forge][dl] live grid: %zu cells over %u placements\n", g_liveGrid.size(), g_ws0Count);
    }

    // Create the persistent per-frame instance + indirect-arg rings (CPU_TO_GPU, mapped). Replaces
    // the probe's per-scope addResource/removeResource (too slow per frame). Idempotent.
    bool dlCreateLiveRings(Renderer* R) {
        if (g_pStaticsInstRing && g_pStaticsArgsRing) { return true; }
        BufferLoadDesc ir = {};
        ir.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
        ir.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        ir.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        ir.mDesc.mSize = (uint64_t)kLiveMaxInst * kStaticsInstStride;
        ir.mDesc.pName = "staticsInstRing";
        ir.pData = nullptr;
        ir.ppBuffer = &g_pStaticsInstRing;
        addResource(&ir, nullptr);
        BufferLoadDesc ar = {};
        ar.mDesc.mDescriptors = DESCRIPTOR_TYPE_INDIRECT_BUFFER;
        ar.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        ar.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        ar.mDesc.mSize = (uint64_t)kLiveMaxSubsets * sizeof(IndirectDrawIndexArguments);
        ar.mDesc.pName = "staticsArgsRing";
        ar.pData = nullptr;
        ar.ppBuffer = &g_pStaticsArgsRing;
        addResource(&ar, nullptr);
        waitForAllResourceLoads();
        return g_pStaticsInstRing && g_pStaticsArgsRing;
    }

    // Stage B (M1) GPU statics cull resources (B2 = COUNT-only validation). One-time, AFTER
    // buildStaticsGrid fills g_cullInst: upload the canonical instance struct as a structured SRV,
    // create the survivor-count UAV + readback/zero staging + per-frame CullParams cbuffer, and the
    // cull.comp pipeline/descriptor set. Idempotent; non-fatal if it fails (validation just won't run).
    bool dlCreateCullResources(Renderer* R) {
        if (g_live.pCullPipeline) { return true; }
        if (g_cullInst.empty())   { return false; }
        g_live.cullInstCount = (uint32_t)g_cullInst.size();   // == g_ws0Count

        // (1) Resident instance SRV (GPU_ONLY structured buffer, uploaded once from g_cullInst).
        //     Structured SRV over an UPLOAD heap is illegal in D3D12 (silent device-remove) → GPU_ONLY.
        BufferLoadDesc ib = {};
        ib.mDesc.mDescriptors  = DESCRIPTOR_TYPE_BUFFER;
        ib.mDesc.mMemoryUsage  = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        ib.mDesc.mStructStride = sizeof(GpuCullInstance);     // 96
        ib.mDesc.mElementCount = g_live.cullInstCount;
        ib.mDesc.mSize         = (uint64_t)ib.mDesc.mStructStride * ib.mDesc.mElementCount;
        ib.mDesc.mStartState   = RESOURCE_STATE_SHADER_RESOURCE;
        ib.mDesc.pName         = "cullInstBuf";
        ib.pData               = g_cullInst.data();           // staged upload at load
        ib.ppBuffer            = &g_live.pCullInstBuf;
        addResource(&ib, nullptr);

        // (2) Survivor-count UAV (uint[4]; [0] = Σ numSubsets). DEFAULT heap (UAV can't be upload/readback).
        BufferLoadDesc cb = {};
        cb.mDesc.mDescriptors  = DESCRIPTOR_TYPE_RW_BUFFER;
        cb.mDesc.mMemoryUsage  = RESOURCE_MEMORY_USAGE_GPU_ONLY;
        cb.mDesc.mStructStride = sizeof(uint32_t);
        cb.mDesc.mElementCount = 4;
        cb.mDesc.mSize         = (uint64_t)cb.mDesc.mStructStride * cb.mDesc.mElementCount;
        cb.mDesc.mStartState   = RESOURCE_STATE_UNORDERED_ACCESS;
        cb.mDesc.pName         = "cullCountBuf";
        cb.pData               = nullptr;
        cb.ppBuffer            = &g_live.pCullCountBuf;
        addResource(&cb, nullptr);

        // (3) Readback (GPU_TO_CPU) + zero-reset (CPU_TO_GPU) staging, persistent-mapped. Forge has no
        //     buffer->buffer copy, so the per-frame reset/readback uses raw D3D12 CopyBufferRegion.
        BufferLoadDesc rb = {};
        rb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_TO_CPU;
        rb.mDesc.mFlags       = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        rb.mDesc.mSize        = 16;
        rb.mDesc.mStartState  = RESOURCE_STATE_COPY_DEST;
        rb.mDesc.pName        = "cullCountReadback";
        rb.ppBuffer           = &g_live.pCullCountReadback;
        addResource(&rb, nullptr);
        BufferLoadDesc zb = {};
        zb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        zb.mDesc.mFlags       = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        zb.mDesc.mSize        = 16;
        zb.mDesc.pName        = "cullCountZero";
        zb.ppBuffer           = &g_live.pCullCountZero;
        addResource(&zb, nullptr);

        // (4) Per-frame CullParams cbuffer (planes/eye/ranges/count), filled in dlLiveCullAndBuild.
        BufferLoadDesc pb = {};
        pb.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
        pb.mDesc.mFlags       = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
        pb.mDesc.mSize        = 256;                          // >= sizeof(CullParams) (144B), CBV-aligned
        pb.mDesc.pName        = "cullParamsCbv";
        pb.ppBuffer           = &g_live.pCullParamsCbv;
        addResource(&pb, nullptr);

        waitForAllResourceLoads();
        if (g_live.pCullCountZero && g_live.pCullCountZero->pCpuMappedAddress) {
            std::memset(g_live.pCullCountZero->pCpuMappedAddress, 0, 16);  // reset source (never changes)
        }
        if (!g_live.pCullInstBuf || !g_live.pCullCountBuf || !g_live.pCullCountReadback
            || !g_live.pCullCountZero || !g_live.pCullParamsCbv) {
            std::printf("[forge][cull] Stage B resource alloc FAILED\n");
            return false;
        }

        // (5) cull.comp pipeline (merged ComputeRootSignature) + its PerBatch descriptor set.
        ShaderLoadDesc csd = {};
        csd.mComp.pFileName = "cull.comp";
        addShader(R, &csd, &g_live.pCullShader);
        if (!g_live.pCullShader) { std::printf("[forge][cull] addShader(cull.comp) FAILED\n"); return false; }
        PipelineDesc cpd = {};
        cpd.mType = PIPELINE_TYPE_COMPUTE;
        cpd.mComputeDesc.pShaderProgram = g_live.pCullShader;
        addPipeline(R, &cpd, &g_live.pCullPipeline);
        if (!g_live.pCullPipeline) { std::printf("[forge][cull] addPipeline(cull.comp) FAILED\n"); return false; }

        DescriptorSetDesc cset = SRT_SET_DESC(CullSrtData, PerBatch, 1, 0);
        addDescriptorSet(R, &cset, &g_live.pCullSet);
        if (!g_live.pCullSet) { std::printf("[forge][cull] addDescriptorSet FAILED\n"); return false; }
        {
            DescriptorData d[3] = {};
            d[0].mIndex     = SRT_RES_IDX(CullSrtData, PerBatch, gCullParams);
            d[0].ppBuffers  = &g_live.pCullParamsCbv;
            d[1].mIndex     = SRT_RES_IDX(CullSrtData, PerBatch, gCullInst);
            d[1].mCount     = 1;
            d[1].ppBuffers  = &g_live.pCullInstBuf;
            d[2].mIndex     = SRT_RES_IDX(CullSrtData, PerBatch, gCullCount);
            d[2].mCount     = 1;
            d[2].ppBuffers  = &g_live.pCullCountBuf;
            updateDescriptorSet(R, 0, g_live.pCullSet, 3, d);
        }
        std::printf("[forge][cull] Stage B ready: %u instances (struct=%zuB)\n",
                    g_live.cullInstCount, sizeof(GpuCullInstance));
        return true;
    }

    // Extract the 6 frustum planes (normalized) from a ROW-MAJOR viewProj used as clip = v*M (the
    // same convention the host uploads, including the reverse-Z munge). Gribb-Hartmann; planeN =
    // (a,b,c,d) with inside == a*x+b*y+c*z+d >= 0. m[i*4+j] = M[row i][col j]; colK = rows' Kth col.
    void dlExtractFrustum(const float* m, float planes[6][4]) {
        // colX=...j0, colY=...j1, colZ=...j2, colW=...j3 across rows i=0..3.
        auto setp = [&](int idx, float a, float b, float c, float d) {
            float inv = 1.0f / std::sqrt(a*a + b*b + c*c + 1e-20f);
            planes[idx][0] = a*inv; planes[idx][1] = b*inv; planes[idx][2] = c*inv; planes[idx][3] = d*inv;
        };
        // left = colW + colX
        setp(0, m[3]+m[0], m[7]+m[4], m[11]+m[8],  m[15]+m[12]);
        // right = colW - colX
        setp(1, m[3]-m[0], m[7]-m[4], m[11]-m[8],  m[15]-m[12]);
        // bottom = colW + colY
        setp(2, m[3]+m[1], m[7]+m[5], m[11]+m[9],  m[15]+m[13]);
        // top = colW - colY
        setp(3, m[3]-m[1], m[7]-m[5], m[11]-m[9],  m[15]-m[13]);
        // near = colZ (DX z in [0,w]; reverse-Z swaps which physical plane this is — volume identical)
        setp(4, m[2],      m[6],      m[10],       m[14]);
        // far = colW - colZ
        setp(5, m[3]-m[2], m[7]-m[6], m[11]-m[10], m[15]-m[14]);
    }

    // Sphere (relative-space center + radius) vs frustum. true = at least partly inside.
    bool dlSphereInFrustum(const float planes[6][4], float cx, float cy, float cz, float r) {
        for (int i = 0; i < 6; ++i) {
            float d = planes[i][0]*cx + planes[i][1]*cy + planes[i][2]*cz + planes[i][3];
            if (d < -r) { return false; }
        }
        return true;
    }

    // Per-frame cull + ring fill (runs BEFORE command recording — it lazily creates GPU resources +
    // loads newly-visible textures, which must not happen mid-command-buffer). Fills g_liveLandVisible
    // + the instance/args rings, and writes the DL fields of gFrameData (lodParams/lodSunAmb/lodEye).
    // rzViewProj = the relative, reverse-Z, extended-far viewProj already in gFrameData.
    void dlLiveCullAndBuild(Renderer* R, const float* rzViewProj) {
        g_liveLandVisible.clear();
        g_liveLastInst = g_liveLastSubsets = g_liveLastLand = 0;
        if (!g_dlExterior) { return; }
        const double tCull0 = hostNowMs();   // CPU cull+build cost (NOT in the host record/gpu metrics)

        // Lazy one-time resident load (first exterior frame): land + statics library + grid + rings.
        if (!g_dlLiveInit) {
            if (!buildLandPath(R) || !loadDistantLand(R)) {
                std::printf("[forge][dl] live land load FAILED — DL disabled\n");
                g_dlExterior = false; return;
            }
            g_staticsLiveOk = buildStaticsPath(R) && loadDistantStatics(R)
                            && buildStaticsTextureArrays(R) && dlCreateLiveRings(R);
            if (g_staticsLiveOk) {
                buildStaticsGrid();
                // Stage B (B2): upload g_cullInst + create the GPU cull pipeline (validation only;
                // non-fatal — the CPU cull stays authoritative until B3's draw cutover).
                dlCreateCullResources(R);
            }
            else { std::printf("[forge][dl] live statics unavailable — land only\n"); }
            g_dlLiveInit = true;
            std::printf("[forge][dl] live init done (land meshes=%u, statics=%d)\n",
                        g_landMeshCount, (int)g_staticsLiveOk);
        }

        // DL frame constants into gFrameData (mirrors the probe's fd[48..59] block).
        float* fd = (float*)g_live.pFrameCbv->pCpuMappedAddress;
        fd[48] = (float)kLandBaseSlot; fd[49] = (float)kLandNormalSlot;
        fd[50] = (float)kLandDetailSlot; fd[51] = 7168.0f;     // lodParams.zw (detail slot, nearViewRange)
        fd[52] = fd[24]; fd[53] = fd[25]; fd[54] = fd[26]; fd[55] = 0.0f;  // lodSunAmb = ambCol
        fd[56] = g_dlEye[0]; fd[57] = g_dlEye[1]; fd[58] = g_dlEye[2]; fd[59] = 0.0f;  // lodEye

        float planes[6][4];
        dlExtractFrustum(rzViewProj, planes);
        const float eye[3] = { g_dlEye[0], g_dlEye[1], g_dlEye[2] };

        // Land: frustum-cull the resident sphere set in relative space.
        for (uint32_t i = 0; i < g_landMeshCount; ++i) {
            const LandMeshGPU& m = g_landMeshes[i];
            if (dlSphereInFrustum(planes, m.cx - eye[0], m.cy - eye[1], m.cz - eye[2], m.r)) {
                g_liveLandVisible.push_back(i);
            }
        }
        g_liveLastLand = (uint32_t)g_liveLandVisible.size();

        if (!g_staticsLiveOk) { return; }

        // Statics: per-cell coarse reject, then per-instance tier/distance/MinSize + frustum cull.
        // Survivors expand to (instance x subset), grouped by subset into the instance ring.
        static std::vector<std::vector<float>> s_bySubset;   // reused; capacity retained across frames
        static std::vector<uint32_t> s_touched;              // subset ids that got instances this frame
        if (s_bySubset.size() != g_staticsSubsets.size()) { s_bySubset.assign(g_staticsSubsets.size(), {}); }
        s_touched.clear();

        const float nearEnd = Configuration.DL.NearStaticEnd     * 8192.0f;   // cell -> world distance
        const float farEnd  = Configuration.DL.FarStaticEnd      * 8192.0f;
        const float vfarEnd = Configuration.DL.VeryFarStaticEnd  * 8192.0f;
        // (FarStaticMinSize/VeryFarStaticMinSize are now consumed at LOAD — tier is precomputed into
        //  g_cullInst.rangeEndIdx, so the per-frame hot loop no longer needs the MinSize thresholds.)
        // Near cutoff: the Forge NEAR path (cache opaque) already draws statics within the near
        // scene, so a DL static drawn there is a DUPLICATE → z-fight ("double rendering, near and DL").
        // Skip DL statics closer than the handover (MGE clips its distant statics at nearViewRange-768).
        // lodParams.w (fd[51]) = nearViewRange; per-origin cut (a large mesh straddling the band can
        // still gap — acceptable for the additive phase; MGE uses a slab clip plane there).
        const float nearCut  = fd[51] - 768.0f;
        const float nearCut2 = nearCut * nearCut;

        uint32_t cellsHit = 0, examined = 0;             // slow-frame diagnostics
        for (const LiveGridCell& c : g_liveGrid) {
            // Cell sphere (relative): center of the padded AABB, radius = half-diagonal + extent pad.
            float pad = kLiveGridCell;   // generous static-extent margin (per-instance test is exact)
            float ccx = 0.5f*(c.minx+c.maxx) - eye[0];
            float ccy = 0.5f*(c.miny+c.maxy) - eye[1];
            float ccz = 0.5f*(c.minz+c.maxz) - eye[2];
            float hx = 0.5f*(c.maxx-c.minx)+pad, hy = 0.5f*(c.maxy-c.miny)+pad, hz = 0.5f*(c.maxz-c.minz)+pad;
            float cr = std::sqrt(hx*hx + hy*hy + hz*hz);
            if (!dlSphereInFrustum(planes, ccx, ccy, ccz, cr)) { continue; }
            // Cell-level DISTANCE reject (Stage A.5): the per-instance test rejects anything past its
            // tier range (≤ vfarEnd, the max). If the cell's NEAREST horizontal point is already beyond
            // vfarEnd, every instance in it is out of range → skip the whole cell. The grid's frustum
            // test alone passes cells out to the far plane (>> vfarEnd), so ~all the 96%-rejected
            // examined instances live in too-far cells. Nearest-point dist to the xy AABB (+pad margin).
            {
                float nx = eye[0] < c.minx ? c.minx : (eye[0] > c.maxx ? c.maxx : eye[0]);
                float ny = eye[1] < c.miny ? c.miny : (eye[1] > c.maxy ? c.maxy : eye[1]);
                float ndx = eye[0] - nx, ndy = eye[1] - ny;
                float farLimit = vfarEnd + pad;   // pad for static extent (the per-instance test is exact)
                if (ndx*ndx + ndy*ndy > farLimit*farLimit) { continue; }
            }
            ++cellsHit;

            for (uint32_t ii : c.inst) {
                ++examined;
                // All per-instance cull data is precomputed (g_cullInst). Only the per-frame tests
                // (distance vs eye, frustum) + the -eye shift remain. world[12..14] == absolute pos.
                const GpuCullInstance& gi = g_cullInst[ii];
                if (gi.rangeEndIdx == 0xFFFFFFFFu) { continue; }   // grass / invalid → skip
                float rangeEnd = (gi.rangeEndIdx == 0) ? nearEnd : (gi.rangeEndIdx == 1) ? farEnd : vfarEnd;
                float dx = gi.posX - eye[0], dy = gi.posY - eye[1];
                float d2 = dx*dx + dy*dy;
                if (d2 < nearCut2 || d2 > rangeEnd*rangeEnd) { continue; }   // near-owned or beyond tier

                // Frustum-cull the instance sphere (relative space). z = world[14] (absolute pos.z).
                if (!dlSphereInFrustum(planes, dx, dy, gi.world[14]-eye[2], gi.effR)) { continue; }

                // World matrix: copy the precomputed absolute matrix, then the ONLY per-frame change —
                // the camera-relative -eye shift on the translation row.
                float W[16];
                std::memcpy(W, gi.world, 16 * sizeof(float));
                W[12] -= eye[0]; W[13] -= eye[1]; W[14] -= eye[2];   // camera-relative shift

                for (uint32_t k = 0; k < gi.numSubsets; ++k) {
                    uint32_t sid = gi.firstSubset + k;
                    // texSlot = (bucket<<16)|layer, resolved once in buildStaticsTextureArrays; all
                    // statics textures are resident in gStaticsArrays (no per-frame re-resolve/stream).
                    uint32_t ts = g_staticsSubsets[sid].texSlot;   // <= 127<<16 -> exact as float
                    std::vector<float>& dst = s_bySubset[sid];
                    if (dst.empty()) { s_touched.push_back(sid); }
                    dst.insert(dst.end(), W, W + 16);
                    dst.push_back((float)ts);
                    dst.push_back((float)g_staticsSubsets[sid].flags);
                    dst.push_back(0.0f); dst.push_back(0.0f);
                }
            }
        }

        // Flatten the touched subsets into the persistent rings (one indirect-arg per subset).
        float* instMap = (float*)g_pStaticsInstRing->pCpuMappedAddress;
        IndirectDrawIndexArguments* argMap = (IndirectDrawIndexArguments*)g_pStaticsArgsRing->pCpuMappedAddress;
        uint32_t instBase = 0, drawCount = 0;
        bool overflow = false;
        for (uint32_t sid : s_touched) {
            std::vector<float>& v = s_bySubset[sid];
            uint32_t instCount = (uint32_t)(v.size() / 20);   // 20 floats (80 B) per instance
            if (instBase + instCount > kLiveMaxInst || drawCount >= kLiveMaxSubsets) {
                overflow = true; v.clear(); continue;
            }
            std::memcpy(instMap + (size_t)instBase * 20, v.data(), v.size() * sizeof(float));
            IndirectDrawIndexArguments a = {};
            a.mIndexCount    = g_staticsSubsets[sid].indexCount;
            a.mInstanceCount = instCount;
            a.mStartIndex    = g_staticsSubsets[sid].ibBase;
            a.mVertexOffset  = g_staticsSubsets[sid].vbBase;
            a.mStartInstance = instBase;
            argMap[drawCount++] = a;
            instBase += instCount;
            v.clear();   // reset for next frame (capacity retained)
        }
        g_liveLastInst = instBase;
        g_liveLastSubsets = drawCount;
        g_lastCullExamined = examined;   // instances tested this frame — normalizes cull ms across scenes

        // Stage B (B2): mirror the EXACT planes/eye/ranges this CPU cull used into the GPU cull
        // cbuffer (the validation dispatch runs during command recording, post-beginCmd). The GPU
        // tests every instance with this identical rule → its Σ numSubsets must equal g_liveLastInst.
        if (g_live.pCullParamsCbv && g_live.pCullParamsCbv->pCpuMappedAddress) {
            float* cp = (float*)g_live.pCullParamsCbv->pCpuMappedAddress;
            for (int p = 0; p < 6; ++p) { std::memcpy(cp + p * 4, planes[p], 4 * sizeof(float)); } // 0..23
            cp[24] = eye[0]; cp[25] = eye[1]; cp[26] = eye[2]; cp[27] = 0.0f;                       // eye
            cp[28] = nearEnd * nearEnd; cp[29] = farEnd * farEnd;                                   // ranges
            cp[30] = vfarEnd * vfarEnd; cp[31] = nearCut2;
            cp[32] = (float)g_live.cullInstCount; cp[33] = cp[34] = cp[35] = 0.0f;                  // misc.x = count
        }
        if (overflow) {
            static bool warned = false;
            if (!warned) { std::printf("[forge][dl] live ring overflow (inst cap %u / subset cap %u) — clamped\n",
                                       kLiveMaxInst, kLiveMaxSubsets); warned = true; }
        }

        // Slow-frame self-report: this CPU cull+build runs in the blocking RPC but OUTSIDE the host's
        // record/gpu metrics, so a stall here is invisible to the 300-frame heartbeat (which never
        // updates at 0.2 fps). Log EVERY frame the cull alone exceeds ~20 ms, with the breakdown
        // (cells/instances examined, survivors). Statics textures are all resident (no per-frame loads).
        const double cullMs = hostNowMs() - tCull0;
        if (cullMs > 20.0) {
            LOG::logline(">> [dl-slow] cull=%.1fms cellsHit=%u examined=%u land=%u inst=%u subsets=%u buckets=%zu",
                         cullMs, cellsHit, examined, g_liveLastLand, g_liveLastInst, g_liveLastSubsets,
                         g_staticsBuckets.empty() ? 0 : g_staticsBuckets.size() - 1);
        }
    }

    // Record the live DL draws into g_live.pCmd. Runs AFTER the near colour pass with colorTarget +
    // pDepth still bound: land + statics depth-write with reverse-Z GEQUAL, so the near scene (larger
    // reverse-Z) correctly occludes DL behind it, and DL fills the empty sky/horizon. (No GTAO on DL:
    // GTAO already ran on the near-only depth before this.)
    void dlLiveRecord() {
        if (!g_dlExterior || !g_dlLiveInit || !g_landLoaded) { return; }
        cmdBeginDebugMarker(g_live.pCmd, 0.3f, 0.8f, 0.5f, "DISTANT LAND (live)");

        if (g_drawDLLand) {   // Phase 0 panel toggle
        cmdBindPipeline(g_live.pCmd, g_pLandPipeline);
        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);
        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
        cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerBatchSet);
        for (uint32_t idx : g_liveLandVisible) {
            LandMeshGPU& m = g_landMeshes[idx];
            Buffer*  vbs[1]     = { m.vb };
            uint32_t strides[1] = { 16 };
            cmdBindVertexBuffer(g_live.pCmd, 1, vbs, strides, nullptr);
            cmdBindIndexBuffer(g_live.pCmd, m.ib, m.large ? INDEX_TYPE_UINT32 : INDEX_TYPE_UINT16, 0);
            cmdDrawIndexedInstanced(g_live.pCmd, m.indexCount, 0, 1, 0, 0);
        }
        }

        if (g_drawDLStatics && g_staticsLiveOk && g_liveLastSubsets > 0) {
            cmdBindPipeline(g_live.pCmd, g_pStaticsPipeline);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerFrameSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerLightsSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPersistentSet);
            cmdBindDescriptorSet(g_live.pCmd, 0, g_live.pPerBatchSet);
            Buffer*  svbs[2]     = { g_pStaticsVB, g_pStaticsInstRing };
            uint32_t sstrides[2] = { 20, kStaticsInstStride };
            cmdBindVertexBuffer(g_live.pCmd, 2, svbs, sstrides, nullptr);
            cmdBindIndexBuffer(g_live.pCmd, g_pStaticsIB, INDEX_TYPE_UINT16, 0);
            cmdExecuteIndirect(g_live.pCmd, INDIRECT_DRAW_INDEX, g_liveLastSubsets,
                               g_pStaticsArgsRing, 0, nullptr, 0);
        }
        cmdEndDebugMarker(g_live.pCmd);
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
        exitDevUI();       // Forge IUI/IFont teardown while the renderer is still alive
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
        // SK1 sky teardown.
        if (g_live.pPerBatchSetSky)         { removeDescriptorSet(R, g_live.pPerBatchSetSky); }
        if (g_live.pSkyWorldsBuf)           { removeResource(g_live.pSkyWorldsBuf); }
        if (g_live.pSkyInstanceBuf)         { removeResource(g_live.pSkyInstanceBuf); }
        if (g_live.pSkyPipeline)            { removePipeline(R, g_live.pSkyPipeline); }
        if (g_live.pSkyPipelineAdd)         { removePipeline(R, g_live.pSkyPipelineAdd); }
        if (g_live.pSkyShader)              { removeShader(R, g_live.pSkyShader); }
        // WT1 water teardown.
        if (g_live.pPerBatchSetWater)       { removeDescriptorSet(R, g_live.pPerBatchSetWater); }
        if (g_live.pWaterWorldsBuf)         { removeResource(g_live.pWaterWorldsBuf); }
        if (g_live.pWaterInstanceBuf)       { removeResource(g_live.pWaterInstanceBuf); }
        if (g_live.pWaterVB)                { removeResource(g_live.pWaterVB); }
        if (g_live.pWaterIB)                { removeResource(g_live.pWaterIB); }
        if (g_live.pRefractColor)           { removeResource(g_live.pRefractColor); }
        if (g_live.pWaterNormalVol)         { removeResource(g_live.pWaterNormalVol); }
        if (g_live.pWaterPipeline)          { removePipeline(R, g_live.pWaterPipeline); }
        if (g_live.pWaterShader)            { removeShader(R, g_live.pWaterShader); }
        // WT2 reflection teardown.
        if (g_live.pPerFrameSetReflect)     { removeDescriptorSet(R, g_live.pPerFrameSetReflect); }
        if (g_live.pPerBatchSetReflectSky)  { removeDescriptorSet(R, g_live.pPerBatchSetReflectSky); }
        if (g_live.pReflectFrameCbv)        { removeResource(g_live.pReflectFrameCbv); }
        if (g_live.pReflectSkyWorldsBuf)    { removeResource(g_live.pReflectSkyWorldsBuf); }
        if (g_live.pReflectSkyInstanceBuf)  { removeResource(g_live.pReflectSkyInstanceBuf); }
        if (g_live.pReflectColor)           { removeRenderTarget(R, g_live.pReflectColor); }
        if (g_live.pReflectDepth)           { removeRenderTarget(R, g_live.pReflectDepth); }
        // Phase 1a distant-land teardown (atlas textures freed by the bindless loop below).
        for (uint32_t i = 0; i < g_landMeshCount; ++i) {
            if (g_landMeshes[i].vb) { removeResource(g_landMeshes[i].vb); g_landMeshes[i].vb = nullptr; }
            if (g_landMeshes[i].ib) { removeResource(g_landMeshes[i].ib); g_landMeshes[i].ib = nullptr; }
        }
        g_landMeshCount = 0;
        g_landLoaded = false;
        if (g_pLandPipeline) { removePipeline(R, g_pLandPipeline); g_pLandPipeline = nullptr; }
        if (g_pLandShader)   { removeShader(R, g_pLandShader);     g_pLandShader = nullptr; }
        // Phase 1b distant-statics teardown (per-subset textures freed by the bindless loop below).
        if (g_pStaticsArgs)     { removeResource(g_pStaticsArgs);     g_pStaticsArgs = nullptr; }
        if (g_pStaticsInst)     { removeResource(g_pStaticsInst);     g_pStaticsInst = nullptr; }
        if (g_pStaticsVB)       { removeResource(g_pStaticsVB);       g_pStaticsVB = nullptr; }
        if (g_pStaticsIB)       { removeResource(g_pStaticsIB);       g_pStaticsIB = nullptr; }
        if (g_pStaticsPipeline) { removePipeline(R, g_pStaticsPipeline); g_pStaticsPipeline = nullptr; }
        if (g_pStaticsShader)   { removeShader(R, g_pStaticsShader);   g_pStaticsShader = nullptr; }
        g_staticsSubsets.clear(); g_staticsDefs.clear(); g_staticsSubsetTex.clear();
        // Bucketed statics texture arrays (gStaticsArrays). Bucket 0 aliases the white array — free
        // it once, separately, below; don't double-free here.
        for (size_t b = 1; b < g_staticsBuckets.size(); ++b) {
            if (g_staticsBuckets[b].tex) { removeResource(g_staticsBuckets[b].tex); }
        }
        g_staticsBuckets.clear(); g_staticsTexReady = false;
        if (g_live.pStaticsWhiteArray) { removeResource(g_live.pStaticsWhiteArray); g_live.pStaticsWhiteArray = nullptr; }
        g_usageData.clear();
        g_staticsDrawCount = 0; g_staticsInstTotal = 0; g_staticsLoaded = false;
        // Phase 1a/1b LIVE distant-land teardown (persistent rings + grid + flags).
        if (g_pStaticsArgsRing) { removeResource(g_pStaticsArgsRing); g_pStaticsArgsRing = nullptr; }
        if (g_pStaticsInstRing) { removeResource(g_pStaticsInstRing); g_pStaticsInstRing = nullptr; }
        // Stage B (M1) GPU cull resources.
        if (g_live.pCullSet)           { removeDescriptorSet(R, g_live.pCullSet);    g_live.pCullSet = nullptr; }
        if (g_live.pCullPipeline)      { removePipeline(R, g_live.pCullPipeline);    g_live.pCullPipeline = nullptr; }
        if (g_live.pCullShader)        { removeShader(R, g_live.pCullShader);        g_live.pCullShader = nullptr; }
        if (g_live.pCullParamsCbv)     { removeResource(g_live.pCullParamsCbv);      g_live.pCullParamsCbv = nullptr; }
        if (g_live.pCullCountZero)     { removeResource(g_live.pCullCountZero);      g_live.pCullCountZero = nullptr; }
        if (g_live.pCullCountReadback) { removeResource(g_live.pCullCountReadback);  g_live.pCullCountReadback = nullptr; }
        if (g_live.pCullCountBuf)      { removeResource(g_live.pCullCountBuf);       g_live.pCullCountBuf = nullptr; }
        if (g_live.pCullInstBuf)       { removeResource(g_live.pCullInstBuf);        g_live.pCullInstBuf = nullptr; }
        g_live.cullInstCount = 0;
        g_liveGrid.clear(); g_liveLandVisible.clear();
        g_dlLiveInit = false; g_staticsLiveOk = false; g_dlExterior = false;
        g_liveLastInst = g_liveLastSubsets = g_liveLastLand = 0;
        // Tier 2 compute (linearize + GTAO + AO bilateral blur).
        if (g_live.pAOBlurSet)      { removeDescriptorSet(R, g_live.pAOBlurSet); }
        if (g_live.pGtaoBatchSet)   { removeDescriptorSet(R, g_live.pGtaoBatchSet); }
        if (g_live.pLinearizeSet)   { removeDescriptorSet(R, g_live.pLinearizeSet); }
        if (g_live.pAOBlurPipeline) { removePipeline(R, g_live.pAOBlurPipeline); }
        if (g_live.pGtaoPipeline)   { removePipeline(R, g_live.pGtaoPipeline); }
        if (g_live.pLinearizePipeline) { removePipeline(R, g_live.pLinearizePipeline); }
        if (g_live.pAOBlurShader)   { removeShader(R, g_live.pAOBlurShader); }
        if (g_live.pGtaoShader)     { removeShader(R, g_live.pGtaoShader); }
        if (g_live.pLinearizeShader){ removeShader(R, g_live.pLinearizeShader); }
        if (g_live.pAOParamsCbv)    { removeResource(g_live.pAOParamsCbv); }
        if (g_live.pAOBlur)         { removeResource(g_live.pAOBlur); }
        if (g_live.pAO)             { removeResource(g_live.pAO); }
        if (g_live.pLinearDepth)    { removeResource(g_live.pLinearDepth); }
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
