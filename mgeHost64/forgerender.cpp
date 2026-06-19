// The Forge renderer bootstrap for mgeHost64 — Milestone D1.
//
// Brings the full Forge stack up headlessly (no window system): replicate just
// the subsystem init that WindowsBase.cpp's app main does — initMemAlloc →
// initFileSystem → initLog — then GPU config → Renderer → graphics queue →
// resource loader, log the selected GPU, and tear back down. This proves the
// vendored Vulkan-only build manifest compiles, links, and initialises a real
// device. No rendering yet (that's D3).
//
// Compiled with _HAS_EXCEPTIONS=0 / no-RTTI to match The Forge's ABI (see
// forgerender.h). Keep host (STL) headers OUT of this TU — talk to the rest of
// the host only through the plain decls in forgerender.h.

#include "forgerender.h"

#include <cstdio>

#include "OS/Interfaces/IOperatingSystem.h"
#include "Utilities/Interfaces/IFileSystem.h"
#include "Utilities/Log/Log.h"
#include "Graphics/GraphicsConfig.h"
#include "Graphics/Interfaces/IGraphics.h"
#include "Resources/ResourceLoader/Interfaces/IResourceLoader.h"
#include "Utilities/Interfaces/ILog.h"
// IMemory.h overrides new/delete/malloc — Forge convention: include it LAST.
#include "Utilities/Interfaces/IMemory.h"

// App-layer callback normally provided by WindowsBase.cpp (which we exclude — it
// drags in the window system). Vulkan.c calls this on device-lost. Headless: no
// swapchain to rebuild, so record nothing. The header's extern "C" block gives
// this C linkage to match Vulkan.c's (compiled-as-C) reference.
void requestReset(const ResetDesc* pResetDesc) { (void)pResetDesc; }

namespace {
    const char* kAppName = "mgeHost64";
}

namespace ForgeRender {
    bool probe() {
        // Unbuffered: a Forge ASSERT/abort would otherwise swallow piped stdout.
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        std::printf("[forge] probe: bringing up The Forge (D3D12)...\n");

        std::printf("[forge] initMemAlloc...\n");
        if (!initMemAlloc(kAppName)) {
            std::printf("[forge] initMemAlloc FAILED\n");
            return false;
        }

        std::printf("[forge] initFileSystem...\n");
        FileSystemInitDesc fsDesc = {};
        fsDesc.pAppName = kAppName;
        // Headless: we set resource dirs explicitly below, so skip the bundled
        // PathStatement.txt the app framework normally ships (its absence is what
        // makes initFileSystem fail otherwise).
        fsDesc.mIsTool = true;
        if (!initFileSystem(&fsDesc)) {
            std::printf("[forge] initFileSystem FAILED\n");
            exitMemAlloc();
            return false;
        }

        // Resource dirs must be set BEFORE initLog — initLog opens its log file
        // under RD_LOG, and an unconfigured mount asserts. All point at the exe's
        // working dir for now. GPU config (gpu.cfg/gpu.data) is non-fatal if absent
        // (falls back to first GPU + LOW preset); we ship those properly in D2.
        std::printf("[forge] setting resource dirs...\n");
        fsSetPathForResourceDir(pSystemFileIO, RD_LOG, "");
        fsSetPathForResourceDir(pSystemFileIO, RD_GPU_CONFIG, "");
        fsSetPathForResourceDir(pSystemFileIO, RD_OTHER_FILES, "");
        fsSetPathForResourceDir(pSystemFileIO, RD_SHADER_BINARIES, "");
        fsSetPathForResourceDir(pSystemFileIO, RD_PIPELINE_CACHE, "");

        std::printf("[forge] initLog...\n");
        initLog(kAppName, DEFAULT_LOG_LEVEL);
        std::printf("[forge] initLog OK\n");

        Renderer*   pRenderer = nullptr;
        RendererDesc settings = {};
        std::printf("[forge] initGPUConfiguration...\n");
        initGPUConfiguration(settings.pExtendedSettings);
        std::printf("[forge] initRenderer...\n");
        initRenderer(kAppName, &settings, &pRenderer);
        std::printf("[forge] initRenderer returned (pRenderer=%p)\n", (void*)pRenderer);
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
        } else {
            std::printf("[forge] graphics queue created\n");
        }

        initResourceLoaderInterface(pRenderer);
        std::printf("[forge] resource loader up\n");

        // Tear everything back down in reverse order.
        exitResourceLoaderInterface(pRenderer);
        if (pQueue) {
            exitQueue(pRenderer, pQueue);
        }
        exitRenderer(pRenderer);
        exitGPUConfiguration();
        exitLog();
        exitFileSystem();
        exitMemAlloc();

        std::printf("[forge] probe complete — full bring-up + teardown OK\n");
        return true;
    }
}
