#include "renderprocess.h"
#include "configuration.h"
#include "ipc/client.h"
#include "ipc/geomwire.h"
#include "support/log.h"
#include "dxvk_interop.h"
#include "distantland.h"
#include "scenegraph_geometry_cache.h"
#include "scenegraph.h"
#include "morrowindbsa.h"

#include <windows.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
    // Seam target size. Set at bring-up to the live backbuffer resolution (the host's
    // Forge render target is created at the same size via RenderInit), so the composite
    // is a 1:1 full-screen blit. Falls back to 640x360 if the backbuffer can't be queried.
    UINT g_w = 640;
    UINT g_h = 360;

    IPC::Client* g_client = nullptr;
    bool   g_initOk  = false;
    bool   g_enabled = true;           // composite ON by default; F11 toggles it OFF/ON
    int    g_debugMode = 0;            // F12 diagnostic cycle: 0=normal, 1=depth (world-distance), 2=scatter, 3=AO, 4=bent normal
    unsigned g_frame = 0;

    // --- Feeding-side spike logging --------------------------------------------------
    // Periodic FPS dips are suspected to come from the client feed (geometry/texture
    // uploads + the blocking host RPC), not the host GPU. Time each phase of onPresent
    // with QPC and, when the whole feed exceeds kSpikeMs, emit ONE breakdown line to
    // mgeXE.log so the periodic culprit (almost certainly texture streaming) is visible.
    constexpr double kSpikeMs = 5.0;   // ~ one 165Hz frame budget; tune as needed
    double g_lastPresentMs = 0.0;      // for the inter-present delta (dip magnitude)

    // Baseline heartbeat: the spike log only fires on >=kSpikeMs frames, so it's BLIND to the
    // normal per-frame cost ("always slow" lives in the baseline, not the spikes). Accumulate
    // every composited frame and log avg/max over a window so the true standing-still cost and
    // its breakdown are visible without spamming.
    constexpr unsigned kHeartbeatFrames = 300;
    struct Accum { double feed, render, host, copy, dt; double maxFeed, maxDt; unsigned n; };
    Accum g_hb = {};

    inline double nowMs() {
        static LARGE_INTEGER freq = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f; }();
        LARGE_INTEGER c; QueryPerformanceCounter(&c);
        return 1000.0 * (double)c.QuadPart / (double)freq.QuadPart;
    }

    // --- M1b: opaque-geometry capture + upload -----------------------------------
    // The cache hands us model-space parts (pos+normal+indices). We assign each a
    // dense host slot, accumulate them into a pending byte blob, and flush them to
    // the Forge host in window-sized whole-part chunks at present time (a safe point
    // with no other RPC in flight). Slots are stable per cache key; re-upload only on
    // revision change. The host stores meshes slot-indexed for M1c's per-frame draw.
    // Chunked shared vec (see ipc/geomwire.h): an 8MB window as 8x1MB chunks, so the
    // Vec reservation (maxSize*windowBytes) stays small. Chunk cap = the full window.
    constexpr std::uint32_t kGeomChunkCap = IPC::kGeomWindowBytes;  // <= window (assign_bytes)

    std::optional<IPC::VecView<IPC::GeomChunk>> g_geomVec;          // persistent geometry upload vec
    std::optional<IPC::VecView<IPC::GeomChunk>> g_drawVec;          // persistent per-frame draw-list vec
    std::optional<IPC::VecView<IPC::GeomChunk>> g_skinnedVec;       // persistent per-frame skinned draw-list vec
    std::optional<IPC::VecView<IPC::GeomChunk>> g_multiMapVec;      // persistent per-frame multi-map draw-list vec (Tier 4)
    std::optional<IPC::VecView<IPC::GeomChunk>> g_lightVec;         // persistent per-frame point-light vec (Tier 3a)
    std::vector<std::uint8_t>                 g_pendingBlob;        // packed parts awaiting flush
    std::uint32_t                             g_pendingParts = 0;
    std::unordered_map<std::uint32_t, std::uint32_t> g_keySlot;    // cache key -> host slot
    // Last-shipped identity per cache key. Dedup is on (modelId, vc, rev), NOT rev alone:
    // the key is a recycled NiTriShape*, so a new object can inherit a freed key+slot; the
    // GeometryData ptr (modelId) + vertexCount disambiguate it (see captureGeometry).
    struct UploadSig { std::uint32_t id; std::uint32_t vc; std::uint16_t rev; };
    std::unordered_map<std::uint32_t, UploadSig> g_uploadedRev;    // cache key -> last sent identity
    std::uint32_t                             g_nextSlot = 0;
    std::vector<std::uint8_t>                 g_drawScratch;        // packed DrawItemWire[] this frame
    std::vector<std::uint8_t>                 g_skinnedScratch;     // packed [SkinnedDrawWire][palette]* this frame
    std::vector<std::uint8_t>                 g_multiMapScratch;    // packed MultiMapDrawWire[] this frame (Tier 4)
    std::vector<std::uint8_t>                 g_lightScratch;       // packed PointLightWire[] this frame

    // --- Phase 2 bindless texture residency (client) ---
    // Each unique texture (by normalized name) gets a dense bindless slot; its raw DDS bytes
    // are loaded once via BSA::loadFileBytes and shipped to the host (which decodes into
    // gTextures[slot]). Misses/oversize map to slot 0 (host default white) and are cached so
    // we don't retry. Rides the geometry channel's chunked vec (g_texVec).
    std::optional<IPC::VecView<IPC::GeomChunk>> g_texVec;           // persistent texture upload vec
    std::unordered_map<std::string, std::uint32_t> g_texSlot;       // normalized name -> bindless slot
    std::uint32_t                             g_nextTexSlot = 1;    // 0 = host default white
    std::vector<std::uint8_t>                 g_texPendingBlob;     // [TexUploadWire][dds]* awaiting flush
    std::uint32_t                             g_texPendingCount = 0;

    // Per-frame draw list rides a 4-chunk (4MB) vec: ~61K DrawItemWire, well over the
    // host's kMaxDraws cap. Geometry vec stays 8 chunks (8MB).
    constexpr unsigned kDrawChunks = 4;

    void initSceneVecs();   // defined below; called from lazyInit
    void flushGeometry();
    void flushTextures();
    std::uint32_t resolveTextureSlot(const char* textureName);

    // --- DXVK Vulkan-interop seam ---
    // MW's MAIN device is DXVK (Vulkan-backed). DXVK exposes ID3D9VkInteropDevice, which
    // hands us its own VkInstance/VkPhysicalDevice/VkDevice/VkQueue. We:
    //   - import the Forge host's shared D3D12 render target (an NT handle) as external
    //     memory bound to a VkImage on DXVK's device, and
    //   - create a DXVK-owned D3D9 render-target texture (via ID3D9VkInteropDevice::
    //     CreateImage, with TRANSFER_DST usage) whose VkImage we copy INTO each frame
    //     with a raw vkCmdCopyImage on DXVK's queue.
    // The MAIN device then StretchRects that D3D9 texture to its backbuffer (unchanged).
    // One Vulkan device throughout: no native d3d9, no D3D9On12, no cross-backend share.
    // The D3D12->Vulkan import handle type (D3D12_RESOURCE / D3D11_TEXTURE) is chosen at
    // bring-up by querying the physical device, so the same binary adapts per vendor.

    ID3D9VkInteropDevice* g_vki = nullptr;       // QI of MW's DXVK device (owned ref)

    VkInstance       g_inst   = VK_NULL_HANDLE;  // DXVK's handles (borrowed; do not destroy)
    VkPhysicalDevice g_phys   = VK_NULL_HANDLE;
    VkDevice         g_dev    = VK_NULL_HANDLE;
    VkQueue          g_queue  = VK_NULL_HANDLE;
    uint32_t         g_qFamily = 0;

    HANDLE           g_hostHandle = nullptr;     // Forge RT shared NT handle (we own it)
    VkExternalMemoryHandleTypeFlagBits g_htype = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
    VkImage          g_importImg = VK_NULL_HANDLE;  // imported host RT (owned)
    VkDeviceMemory   g_importMem = VK_NULL_HANDLE;  // imported external memory (owned)

    IDirect3DTexture9* g_mainTex = nullptr;      // DXVK D3D9 RT texture; copy dst + blit src
    VkImage            g_dstImg = VK_NULL_HANDLE;   // its backing VkImage (borrowed)
    VkImageLayout      g_dstLayout = VK_IMAGE_LAYOUT_GENERAL;  // DXVK's resting layout for g_mainTex

    VkCommandPool   g_cmdPool = VK_NULL_HANDLE;  // owned
    VkCommandBuffer g_cmd     = VK_NULL_HANDLE;
    VkFence         g_fence   = VK_NULL_HANDLE;  // owned

    HMODULE g_vulkanDll = nullptr;

    // Dynamically-resolved Vulkan entry points (DXVK's loader, via vulkan-1.dll).
    struct VkApi {
        PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
        PFN_vkGetDeviceProcAddr   GetDeviceProcAddr;
        PFN_vkGetPhysicalDeviceImageFormatProperties2 GetPhysicalDeviceImageFormatProperties2;
        PFN_vkGetPhysicalDeviceMemoryProperties       GetPhysicalDeviceMemoryProperties;
        PFN_vkCreateImage                CreateImage;
        PFN_vkDestroyImage               DestroyImage;
        PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
        PFN_vkAllocateMemory             AllocateMemory;
        PFN_vkFreeMemory                 FreeMemory;
        PFN_vkBindImageMemory            BindImageMemory;
        PFN_vkGetMemoryWin32HandlePropertiesKHR GetMemoryWin32HandlePropertiesKHR;
        PFN_vkCreateCommandPool          CreateCommandPool;
        PFN_vkDestroyCommandPool         DestroyCommandPool;
        PFN_vkAllocateCommandBuffers     AllocateCommandBuffers;
        PFN_vkBeginCommandBuffer         BeginCommandBuffer;
        PFN_vkEndCommandBuffer           EndCommandBuffer;
        PFN_vkResetCommandBuffer         ResetCommandBuffer;
        PFN_vkCmdPipelineBarrier         CmdPipelineBarrier;
        PFN_vkCmdCopyImage               CmdCopyImage;
        PFN_vkQueueSubmit                QueueSubmit;
        PFN_vkCreateFence                CreateFence;
        PFN_vkDestroyFence               DestroyFence;
        PFN_vkWaitForFences              WaitForFences;
        PFN_vkResetFences                ResetFences;
    } vk = {};

    bool loadVulkan() {
        if (!g_vulkanDll) {
            g_vulkanDll = LoadLibraryA("vulkan-1.dll");
        }
        if (!g_vulkanDll) {
            LOG::logline("!! [seam] LoadLibrary(vulkan-1.dll) failed — seam disabled");
            return false;
        }
        vk.GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)GetProcAddress(g_vulkanDll, "vkGetInstanceProcAddr");
        if (!vk.GetInstanceProcAddr) {
            LOG::logline("!! [seam] vkGetInstanceProcAddr missing — seam disabled");
            return false;
        }
        vk.GetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vk.GetInstanceProcAddr(g_inst, "vkGetDeviceProcAddr");

        bool ok = vk.GetDeviceProcAddr != nullptr;
        #define INST(name) do { vk.name = (PFN_vk##name)vk.GetInstanceProcAddr(g_inst, "vk" #name); ok = ok && vk.name; } while (0)
        #define DEV(name)  do { vk.name = (PFN_vk##name)vk.GetDeviceProcAddr(g_dev, "vk" #name);   ok = ok && vk.name; } while (0)
        INST(GetPhysicalDeviceImageFormatProperties2);
        INST(GetPhysicalDeviceMemoryProperties);
        DEV(CreateImage);
        DEV(DestroyImage);
        DEV(GetImageMemoryRequirements);
        DEV(AllocateMemory);
        DEV(FreeMemory);
        DEV(BindImageMemory);
        DEV(GetMemoryWin32HandlePropertiesKHR);
        DEV(CreateCommandPool);
        DEV(DestroyCommandPool);
        DEV(AllocateCommandBuffers);
        DEV(BeginCommandBuffer);
        DEV(EndCommandBuffer);
        DEV(ResetCommandBuffer);
        DEV(CmdPipelineBarrier);
        DEV(CmdCopyImage);
        DEV(QueueSubmit);
        DEV(CreateFence);
        DEV(DestroyFence);
        DEV(WaitForFences);
        DEV(ResetFences);
        #undef INST
        #undef DEV
        if (!ok) {
            LOG::logline("!! [seam] failed to resolve required Vulkan entry points — seam disabled");
        }
        return ok;
    }

    uint32_t pickMemoryType(uint32_t typeBits, VkMemoryPropertyFlags want) {
        VkPhysicalDeviceMemoryProperties mp = {};
        vk.GetPhysicalDeviceMemoryProperties(g_phys, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
                return i;
            }
        }
        return UINT32_MAX;
    }

    // Is a given external handle type importable for our RT format on this physical device?
    bool handleTypeImportable(VkExternalMemoryHandleTypeFlagBits htype) {
        VkPhysicalDeviceExternalImageFormatInfo ext = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO };
        ext.handleType = htype;
        VkPhysicalDeviceImageFormatInfo2 fi = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2 };
        fi.pNext  = &ext;
        fi.format = VK_FORMAT_B8G8R8A8_UNORM;
        fi.type   = VK_IMAGE_TYPE_2D;
        fi.tiling = VK_IMAGE_TILING_OPTIMAL;
        fi.usage  = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VkExternalImageFormatProperties efp = { VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES };
        VkImageFormatProperties2 ifp = { VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2 };
        ifp.pNext = &efp;
        if (vk.GetPhysicalDeviceImageFormatProperties2(g_phys, &fi, &ifp) != VK_SUCCESS) {
            return false;
        }
        return (efp.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
    }

    // Import the host's shared NT handle as a VkImage on DXVK's device.
    bool importHostImage() {
        VkExternalMemoryImageCreateInfo extImg = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
        extImg.handleTypes = g_htype;
        VkImageCreateInfo ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ici.pNext         = &extImg;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = VK_FORMAT_B8G8R8A8_UNORM;
        ici.extent        = { g_w, g_h, 1 };
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vk.CreateImage(g_dev, &ici, nullptr, &g_importImg) != VK_SUCCESS) {
            LOG::logline("!! [seam] vkCreateImage (imported host RT) failed");
            return false;
        }

        VkMemoryRequirements mr = {};
        vk.GetImageMemoryRequirements(g_dev, g_importImg, &mr);

        VkMemoryWin32HandlePropertiesKHR whp = { VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR };
        uint32_t handleTypeBits = mr.memoryTypeBits;
        if (vk.GetMemoryWin32HandlePropertiesKHR(g_dev, g_htype, g_hostHandle, &whp) == VK_SUCCESS) {
            handleTypeBits &= whp.memoryTypeBits;
        }
        uint32_t typeIdx = pickMemoryType(handleTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (typeIdx == UINT32_MAX) {
            typeIdx = pickMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        if (typeIdx == UINT32_MAX) {
            LOG::logline("!! [seam] no device-local memory type for imported handle");
            return false;
        }

        VkMemoryDedicatedAllocateInfo ded = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
        ded.image = g_importImg;
        VkImportMemoryWin32HandleInfoKHR imp = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR };
        imp.pNext      = &ded;
        imp.handleType = g_htype;
        imp.handle     = g_hostHandle;
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.pNext           = &imp;
        mai.allocationSize  = mr.size;
        mai.memoryTypeIndex = typeIdx;
        VkResult r = vk.AllocateMemory(g_dev, &mai, nullptr, &g_importMem);
        if (r != VK_SUCCESS) {
            LOG::logline("!! [seam] *** vkAllocateMemory (import) FAILED VkResult=%d *** (handle type %d not importable on this GPU)", (int)r, (int)g_htype);
            return false;
        }
        if (vk.BindImageMemory(g_dev, g_importImg, g_importMem, 0) != VK_SUCCESS) {
            LOG::logline("!! [seam] vkBindImageMemory (imported host RT) failed");
            return false;
        }
        return true;
    }

    // Create the DXVK D3D9 RT texture (copy destination + StretchRect source) and capture
    // its backing VkImage + resting layout. Priming with ColorFill forces DXVK to settle
    // the image into a defined, content-preserving layout (so our copy isn't discarded).
    bool createDstTexture() {
        D3D9VkExtImageDesc d = {};
        d.Type               = D3DRTYPE_TEXTURE;
        d.Width              = g_w;
        d.Height             = g_h;
        d.Depth              = 1;
        d.MipLevels          = 1;
        d.Usage              = D3DUSAGE_RENDERTARGET;
        d.Format             = D3DFMT_A8R8G8B8;
        d.Pool               = D3DPOOL_DEFAULT;
        d.MultiSample        = D3DMULTISAMPLE_NONE;
        d.MultiSampleQuality = 0;
        d.Discard            = false;
        d.IsAttachmentOnly   = false;
        d.IsLockable         = false;
        d.ImageUsage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT;   // we vkCmdCopyImage INTO it

        IDirect3DResource9* res = nullptr;
        if (FAILED(g_vki->CreateImage(&d, &res)) || !res) {
            LOG::logline("!! [seam] ID3D9VkInteropDevice::CreateImage (dst RT) failed");
            return false;
        }
        HRESULT hr = res->QueryInterface(__uuidof(IDirect3DTexture9), (void**)&g_mainTex);
        res->Release();
        if (FAILED(hr) || !g_mainTex) {
            LOG::logline("!! [seam] dst resource is not a Texture9");
            return false;
        }

        // Prime the layout: ColorFill the surface so DXVK transitions+tracks a real layout.
        IDirect3DSurface9* surf = nullptr;
        if (SUCCEEDED(g_mainTex->GetSurfaceLevel(0, &surf)) && surf) {
            // g_vki's device is MW's device; reach it through the surface's device.
            IDirect3DDevice9* dev = nullptr;
            if (SUCCEEDED(surf->GetDevice(&dev)) && dev) {
                dev->ColorFill(surf, nullptr, D3DCOLOR_ARGB(255, 0, 0, 0));
                dev->Release();
            }
            surf->Release();
        }
        g_vki->FlushRenderingCommands();

        ID3D9VkInteropTexture* vkt = nullptr;
        if (FAILED(g_mainTex->QueryInterface(__uuidof(ID3D9VkInteropTexture), (void**)&vkt)) || !vkt) {
            LOG::logline("!! [seam] dst texture has no ID3D9VkInteropTexture");
            return false;
        }
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        hr = vkt->GetVulkanImageInfo(&g_dstImg, &layout, nullptr);
        vkt->Release();
        if (FAILED(hr) || g_dstImg == VK_NULL_HANDLE) {
            LOG::logline("!! [seam] GetVulkanImageInfo (dst) failed");
            return false;
        }
        // Restore target must preserve contents; never transition back to UNDEFINED.
        g_dstLayout = (layout == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_IMAGE_LAYOUT_GENERAL : layout;
        LOG::logline(">> [seam] dst DXVK texture VkImage=%p resting layout=%d", (void*)g_dstImg, (int)g_dstLayout);
        return true;
    }

    bool createCopyInfra() {
        VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = g_qFamily;
        if (vk.CreateCommandPool(g_dev, &pci, nullptr, &g_cmdPool) != VK_SUCCESS) {
            return false;
        }
        VkCommandBufferAllocateInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cbi.commandPool        = g_cmdPool;
        cbi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        if (vk.AllocateCommandBuffers(g_dev, &cbi, &g_cmd) != VK_SUCCESS) {
            return false;
        }
        VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        return vk.CreateFence(g_dev, &fci, nullptr, &g_fence) == VK_SUCCESS;
    }

    void releaseAll() {
        if (g_dev) {
            if (g_fence)     { vk.DestroyFence(g_dev, g_fence, nullptr); g_fence = VK_NULL_HANDLE; }
            if (g_cmdPool)   { vk.DestroyCommandPool(g_dev, g_cmdPool, nullptr); g_cmdPool = VK_NULL_HANDLE; g_cmd = VK_NULL_HANDLE; }
            if (g_importImg) { vk.DestroyImage(g_dev, g_importImg, nullptr); g_importImg = VK_NULL_HANDLE; }
            if (g_importMem) { vk.FreeMemory(g_dev, g_importMem, nullptr); g_importMem = VK_NULL_HANDLE; }
        }
        if (g_mainTex)     { g_mainTex->Release(); g_mainTex = nullptr; }
        g_dstImg = VK_NULL_HANDLE;
        if (g_hostHandle)  { CloseHandle(g_hostHandle); g_hostHandle = nullptr; }
        if (g_vki)         { g_vki->Release(); g_vki = nullptr; }
        g_inst = VK_NULL_HANDLE; g_phys = VK_NULL_HANDLE; g_dev = VK_NULL_HANDLE; g_queue = VK_NULL_HANDLE;
    }

    // Seam bring-up (called from RenderProcess::init, under the loading bar / live by menu).
    void lazyInit(IDirect3DDevice9* device) {
        if (FAILED(device->QueryInterface(__uuidof(ID3D9VkInteropDevice), (void**)&g_vki)) || !g_vki) {
            LOG::logline("!! [seam] main device is not DXVK (no ID3D9VkInteropDevice) — seam disabled");
            return;
        }
        g_vki->GetVulkanHandles(&g_inst, &g_phys, &g_dev);
        uint32_t queueIndex = 0;
        g_vki->GetSubmissionQueue(&g_queue, &queueIndex, &g_qFamily);
        if (!g_inst || !g_phys || !g_dev || !g_queue) {
            LOG::logline("!! [seam] DXVK returned null Vulkan handles — seam disabled");
            releaseAll();
            return;
        }
        if (!loadVulkan()) {
            releaseAll();
            return;
        }

        // Size the seam to the live backbuffer so the composite is a 1:1 full-screen blit.
        {
            IDirect3DSurface9* bb = nullptr;
            if (SUCCEEDED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &bb)) && bb) {
                D3DSURFACE_DESC sd = {};
                if (SUCCEEDED(bb->GetDesc(&sd)) && sd.Width && sd.Height) {
                    g_w = sd.Width;
                    g_h = sd.Height;
                }
                bb->Release();
            }
            LOG::logline(">> [seam] backbuffer %ux%u — shared RT sized to match", g_w, g_h);
        }

        // Host brings up Forge + creates the shared RT; returns the NT handle already
        // duplicated into THIS process.
        HANDLE hostHandle = nullptr;
        // MSAA: Configuration.AALevel is the D3DMULTISAMPLE value (0/2/4/8); map 0 -> 1 sample.
        const std::uint32_t sampleCount = Configuration.AALevel > 0 ? (std::uint32_t)Configuration.AALevel : 1u;
        // AF: Configuration.AnisoLevel (0 = off, else max anisotropy) — host sampler (Phase 2).
        const std::uint32_t anisoLevel = (std::uint32_t)Configuration.AnisoLevel;
        if (!g_client->renderInitBlocking(g_w, g_h, sampleCount, anisoLevel, nullptr, nullptr, &hostHandle) || hostHandle == nullptr) {
            LOG::logline("!! [seam] renderInit RPC failed or no shared handle; seam disabled");
            releaseAll();
            return;
        }
        g_hostHandle = hostHandle;
        LOG::logline(">> [seam] host shared-RT NT handle (this process) = %p", hostHandle);

        // Cross-vendor: pick the first external handle type the GPU can import.
        const VkExternalMemoryHandleTypeFlagBits candidates[] = {
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT,
        };
        bool picked = false;
        for (VkExternalMemoryHandleTypeFlagBits c : candidates) {
            if (handleTypeImportable(c)) { g_htype = c; picked = true; break; }
        }
        if (!picked) {
            LOG::logline("!! [seam] GPU reports no importable D3D12/D3D11 external image type — seam disabled");
            releaseAll();
            return;
        }
        LOG::logline(">> [seam] importing host RT as external handle type %d", (int)g_htype);

        if (!importHostImage()) {
            releaseAll();
            return;
        }
        if (!createDstTexture()) {
            releaseAll();
            return;
        }
        if (!createCopyInfra()) {
            LOG::logline("!! [seam] failed to create Vulkan copy infrastructure — seam disabled");
            releaseAll();
            return;
        }

        g_initOk = true;
        LOG::logline(">> [seam] DXVK Vulkan-interop seam ready (%ux%u). F11 toggles the composite.", g_w, g_h);

        // M1b/M1c: bring up the geometry upload + per-frame draw-list channels.
        initSceneVecs();
    }

    // Copy the imported host RT -> the DXVK D3D9 texture's VkImage, on DXVK's own queue.
    // Blocking: CPU-waits the copy so the subsequent StretchRect reads finished pixels.
    bool copyHostRtToDst() {
        // Flush any DXVK rendering that touches the dst image before we use its queue.
        g_vki->FlushRenderingCommands();

        vk.ResetCommandBuffer(g_cmd, 0);
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk.BeginCommandBuffer(g_cmd, &bi);

        const VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        // dst: DXVK resting layout -> TRANSFER_DST
        VkImageMemoryBarrier toDst = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        toDst.srcAccessMask       = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        toDst.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        toDst.oldLayout           = g_dstLayout;
        toDst.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.image               = g_dstImg;
        toDst.subresourceRange    = range;
        vk.CmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &toDst);

        VkImageCopy region = {};
        region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.extent         = { g_w, g_h, 1 };
        vk.CmdCopyImage(g_cmd, g_importImg, VK_IMAGE_LAYOUT_GENERAL,
                        g_dstImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        // dst: TRANSFER_DST -> DXVK resting layout (so DXVK's tracking stays valid)
        VkImageMemoryBarrier toRest = toDst;
        toRest.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toRest.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        toRest.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toRest.newLayout     = g_dstLayout;
        vk.CmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &toRest);

        vk.EndCommandBuffer(g_cmd);

        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &g_cmd;

        g_vki->LockSubmissionQueue();
        VkResult r = vk.QueueSubmit(g_queue, 1, &si, g_fence);
        if (r == VK_SUCCESS) {
            vk.WaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX);
            vk.ResetFences(g_dev, 1, &g_fence);
        }
        g_vki->ReleaseSubmissionQueue();
        return r == VK_SUCCESS;
    }

    // Allocate the persistent shared vecs: geometry upload (8 chunks) + per-frame draw
    // list (4 chunks). Both chunk-element typed to dodge the Vec uint32 reservation trap.
    void initSceneVecs() {
        auto gv = g_client->allocVecBlocking<IPC::GeomChunk>(
            IPC::kGeomChunks, IPC::kGeomChunks, IPC::kGeomChunks);
        if (!gv) {
            LOG::logline("!! [seam] geometry upload vec alloc failed — capture disabled");
            return;
        }
        g_geomVec.emplace(std::move(*gv));

        auto dv = g_client->allocVecBlocking<IPC::GeomChunk>(kDrawChunks, kDrawChunks, kDrawChunks);
        if (!dv) {
            LOG::logline("!! [seam] draw-list vec alloc failed — scene path disabled (triangle only)");
        } else {
            g_drawVec.emplace(std::move(*dv));
        }

        // Skinned draw list rides its own 4-chunk vec. The blob is [SkinnedDrawWire]
        // [palette]* — bounded at the host's 256-part cap (256 * (12 + 32*64) ≈ 527KB),
        // well under 4MB.
        auto sv = g_client->allocVecBlocking<IPC::GeomChunk>(kDrawChunks, kDrawChunks, kDrawChunks);
        if (!sv) {
            LOG::logline("!! [seam] skinned draw-list vec alloc failed — skinned path disabled");
        } else {
            g_skinnedVec.emplace(std::move(*sv));
        }

        // Multi-map draw list (Tier 4) rides its own 1-chunk vec. MultiMapDrawWire[] bounded at
        // the host's 256-part cap (256 * 132B ≈ 34KB), well under 1MB.
        auto mm = g_client->allocVecBlocking<IPC::GeomChunk>(1, 1, 1);
        if (!mm) {
            LOG::logline("!! [seam] multi-map draw-list vec alloc failed — multi-map path disabled");
        } else {
            g_multiMapVec.emplace(std::move(*mm));
        }

        // Point-light list (Tier 3a) rides its own 1-chunk vec — kMaxPointLights * 48B ≈ 6KB,
        // far under 1MB. PointLightWire[] world-space, rebuilt each frame from the SceneGraph snapshot.
        auto lv = g_client->allocVecBlocking<IPC::GeomChunk>(1, 1, 1);
        if (!lv) {
            LOG::logline("!! [seam] light vec alloc failed — point lights disabled");
        } else {
            g_lightVec.emplace(std::move(*lv));
        }

        // Texture upload vec: kTexChunks (32MB window) — larger than geometry because a single DDS
        // must fit one window (oversize textures are dropped to white in resolveTextureSlot).
        auto tv = g_client->allocVecBlocking<IPC::GeomChunk>(
            IPC::kTexChunks, IPC::kTexChunks, IPC::kTexChunks);
        if (!tv) {
            LOG::logline("!! [seam] texture upload vec alloc failed — textures disabled (white)");
        } else {
            g_texVec.emplace(std::move(*tv));
        }
        LOG::logline(">> [seam] scene vecs ready (geom %u, draw %u, skinned %u, multimap 1, light 1, tex %u chunks)",
                     IPC::kGeomChunks, kDrawChunks, kDrawChunks, IPC::kTexChunks);
    }

    // Normalize an NI SourceTexture::fileName to the bare name BSA::loadFileBytes expects
    // (it re-adds "textures\"). Lower-cased (BSA hashing + case-insensitive loose lookup),
    // backslash-separated, with any leading "data files\" then "textures\" stripped.
    static std::string normalizeTextureName(const char* fileName) {
        if (!fileName) {
            return std::string();
        }
        std::string s(fileName);
        for (auto& c : s) {
            c = (char)std::tolower((unsigned char)c);
            if (c == '/') { c = '\\'; }
        }
        if (s.size() >= 11 && s.compare(0, 11, "data files\\") == 0) { s.erase(0, 11); }
        if (s.size() >= 9  && s.compare(0, 9,  "textures\\")  == 0) { s.erase(0, 9); }
        return s;
    }

    // Map a texture name to its bindless slot, loading + queueing its DDS on first sight.
    // Misses / oversize / residency-full → slot 0 (host default white), cached so we don't retry.
    std::uint32_t resolveTextureSlot(const char* textureName) {
        if (!textureName || !*textureName || !g_texVec) {
            return 0;
        }
        std::string name = normalizeTextureName(textureName);
        if (name.empty()) {
            return 0;
        }
        auto it = g_texSlot.find(name);
        if (it != g_texSlot.end()) {
            return it->second;   // already resolved (slot or cached-miss 0)
        }
        if (g_nextTexSlot >= IPC::kMaxTextures) {
            static bool logged = false;
            if (!logged) { LOG::logline("!! [tex] residency full (%u slots) — extra textures = white", IPC::kMaxTextures); logged = true; }
            g_texSlot.emplace(name, 0);
            return 0;
        }

        void* data = nullptr;
        unsigned size = 0;
        // skipDistantStatics=true: the Forge near path must NOT pick the distantland\statics
        // downscaled-LOD copies (they blur near geometry) — resolve loose Data Files -> BSA.
        if (!BSA::loadFileBytes(name.c_str(), &data, &size, true) || !data || size == 0) {
            static int misses = 0;
            if (misses < 20) { LOG::logline("!! [tex] not found: %s (white)", name.c_str()); ++misses; }
            if (data) { std::free(data); }
            g_texSlot.emplace(name, 0);
            return 0;
        }

        const std::size_t windowBytes = (std::size_t)IPC::kTexWindowBytes;
        if (sizeof(IPC::TexUploadWire) + size > windowBytes) {
            LOG::logline("!! [tex] %s too large (%u bytes) for %zu window — white", name.c_str(), size, windowBytes);
            std::free(data);
            g_texSlot.emplace(name, 0);
            return 0;
        }

        const std::uint32_t slot = g_nextTexSlot++;
        g_texSlot.emplace(name, slot);
        IPC::TexUploadWire hdr{ slot, size };
        const std::size_t at = g_texPendingBlob.size();
        g_texPendingBlob.resize(at + sizeof(hdr) + size);
        std::memcpy(g_texPendingBlob.data() + at, &hdr, sizeof(hdr));
        std::memcpy(g_texPendingBlob.data() + at + sizeof(hdr), data, size);
        std::free(data);
        ++g_texPendingCount;
        return slot;
    }

    // Ship queued texture uploads to the host in window-sized batches on whole-entry
    // boundaries (each entry is guaranteed <= window by resolveTextureSlot). Blocking RPCs.
    void flushTextures() {
        if (!g_texVec || g_texPendingBlob.empty() || g_texPendingCount == 0) {
            return;
        }
        const std::size_t windowBytes = (std::size_t)IPC::kTexWindowBytes;
        const std::uint8_t* base = g_texPendingBlob.data();
        const std::size_t total = g_texPendingBlob.size();
        std::size_t off = 0;
        while (off < total) {
            std::size_t batchEnd = off;
            std::uint32_t batchCount = 0;
            while (batchEnd < total) {
                IPC::TexUploadWire h;
                std::memcpy(&h, base + batchEnd, sizeof(h));
                const std::size_t entryBytes = sizeof(h) + h.byteLen;
                if ((batchEnd - off) + entryBytes > windowBytes) { break; }   // window full
                batchEnd += entryBytes;
                ++batchCount;
            }
            if (batchCount == 0) { break; }   // safety (each entry <= window)
            const std::uint32_t bytes = (std::uint32_t)(batchEnd - off);
            std::uint32_t uploaded = 0;
            if (g_texVec->assign_bytes(base + off, bytes)) {
                g_client->texUploadBlocking(g_texVec->id(), batchCount, bytes, &uploaded);
            }
            if (uploaded == 0xFFFFFFFFu) {
                // Host opaque path not built yet (first scene frame) — keep the whole queue and
                // retry next frame (slots are already assigned; draws show white until then).
                return;
            }
            off = batchEnd;
        }
        g_texPendingBlob.clear();
        g_texPendingCount = 0;
    }

    // Drain g_pendingBlob to the host in window-sized whole-part chunks. Parts are
    // self-describing (GeomPartWire carries vert/index counts), so we walk part
    // boundaries to never split a part across a chunk. Blocking RPCs — called at
    // present time. Static cost: each part ships once per (key,revision).
    void flushGeometry() {
        if (!g_geomVec || g_pendingBlob.empty() || g_pendingParts == 0) {
            return;
        }
        // Geometry now ships on the client's DEDICATED geometry IPC channel
        // (geomUploadBlocking → its own Parameters + events), so it no longer contends
        // with the one-at-a-time cull/scene RPCs on the main channel. That contention used
        // to starve this flush at present time and leave exteriors black (the draw list
        // referenced slots whose geometry never shipped). The old isRpcPending() bail —
        // and its clobber hazard — is gone because there is no shared Parameters union to
        // interleave. The host services the geometry channel on its single thread (WFMO),
        // so this upload just waits its turn behind any in-flight cull instead of being
        // skipped, and can never race renderScene.
        const std::uint8_t* const data = g_pendingBlob.data();
        const std::uint32_t total = static_cast<std::uint32_t>(g_pendingBlob.size());

        std::uint32_t off = 0;
        while (off < total) {
            std::uint32_t chunkBytes = 0;
            std::uint32_t chunkParts = 0;
            std::uint32_t cursor = off;
            while (cursor < total) {
                IPC::GeomPartWire hdr;
                memcpy(&hdr, data + cursor, sizeof(hdr));
                const std::size_t vStride = (hdr.flags & IPC::kGeomFlagSkinned)
                    ? sizeof(IPC::SkinnedVertexWire)
                    : (hdr.flags & IPC::kGeomFlagMultiMap)
                        ? sizeof(IPC::GeomVertexWireMM) : sizeof(IPC::GeomVertexWire);
                const std::uint32_t partSize = static_cast<std::uint32_t>(
                    sizeof(IPC::GeomPartWire)
                    + (std::uint64_t)hdr.vertexCount * vStride
                    + (std::uint64_t)hdr.indexCount * sizeof(std::uint16_t));
                if (chunkBytes != 0 && chunkBytes + partSize > kGeomChunkCap) {
                    break;  // close this chunk on a part boundary
                }
                chunkBytes += partSize;
                ++chunkParts;
                cursor += partSize;
            }
            if (chunkParts == 0) {
                break;  // single part exceeds the cap (shouldn't happen) — bail
            }

            if (!g_geomVec->assign_bytes(data + off, chunkBytes)) {
                LOG::logline("!! [seam] geometry chunk assign_bytes failed (%u bytes)", chunkBytes);
                break;
            }
            // Blocking upload. With the dynamic-VB ring the host side is now a cheap memcpy
            // (animated re-uploads) or a one-time static build, so blocking no longer carries
            // the old recreate+fence cost — and it keeps the present pipeline simple (async
            // kickoff was reverted: it cost the framerate cap without helping the steady state).
            std::uint32_t uploaded = 0;
            if (!g_client->geomUploadBlocking(g_geomVec->id(), chunkParts, chunkBytes, &uploaded)) {
                LOG::logline("!! [seam] geomUpload RPC built %u/%u parts", uploaded, chunkParts);
            }
            off = cursor;
        }

        g_pendingBlob.clear();
        g_pendingParts = 0;
    }

    // Gather this frame's visible opaque parts into g_drawScratch as DrawItemWire[]:
    // each is the part's host slot + its current world transform. Source is the cache's
    // current-frame frustum-visible key set (built in renderStage0); only keys we've
    // assigned a host slot (uploaded, non-skinned, non-landscape) are included. Returns
    // the packed item count (0 if nothing to draw / scene path unavailable).
    std::uint32_t buildDrawList() {
        if (!g_drawVec) {
            return 0;
        }
        // Source = the engine's MSOC drawn set (visibleCacheKeys / s_prevVisibleKeys), NOT
        // the frustum set. The frustum-fallback set iterates the WHOLE cache and keeps every
        // cached LOD level of an object — so an object whose original AND lod meshes are both
        // cached gets drawn TWICE (coincident). Statics resolve to one (invisible), but movers
        // sit near the camera where multiple LOD levels coexist → Z-FIGHTING. The MSOC set is
        // exactly what the engine DREW: one LOD per object, occlusion-correct — the same source
        // the reflection cache path uses for dynamic things (no double there). It's the prior
        // frame's set (1-frame lag) but the world transform is read fresh from the cache below.
        const auto& keys = DistantLand::visibleCacheKeys();
        const auto& cacheMap = MGE::GeometryCache::cache();

        g_drawScratch.clear();
        g_drawScratch.reserve(keys.size() * sizeof(IPC::DrawItemWire));

        IPC::DrawItemWire item;
        std::uint32_t count = 0;
        for (std::uint32_t key : keys) {
            auto ks = g_keySlot.find(key);
            if (ks == g_keySlot.end()) {
                continue;   // not an uploaded opaque part (skinned/landscape/not yet sent)
            }
            auto ce = cacheMap.find(key);
            if (ce == cacheMap.end()) {
                continue;   // evicted since the visible-set build
            }
            const auto& e = ce->second;
            // Mirror the PROVEN D3D9 cache color pass's per-entry filters EXACTLY (drawEntry
            // in rendercachedcolor.cpp) so the Forge draw list draws the same set:
            //   - !d3dTexture: drops UNTEXTURED entries — the worldPickObjectRoot collision
            //     proxies that mirror each visual object. Drawing those gave a second
            //     near-coincident copy; statics overlapped exactly (invisible) but MOVING
            //     objects' visual vs proxy transforms diverged a frame → Z-FIGHTING. This
            //     is THE de-dup. (Do NOT also filter isPickRoot — legit movers like dropped
            //     items/projectiles LIVE in the pick root and ARE textured; the proven path
            //     keeps them, dropping them blanked all movers.)
            //   - blendEnable: alpha-blended OBJECTS stay on the engine/alpha path. But
            //     terrain is the exception — the D9 oracle (renderCachedTerrain) draws ALL
            //     isLandscape regardless of blendEnable (alpha-splat trishapes included),
            //     and at flat-shaded fidelity blend-vs-opaque is invisible. So only filter
            //     blendEnable for non-landscape.
            //   - skinned: ALL skinned parts go through buildSkinnedDrawList (their own
            //     stride-44 VB + GPU palette pipeline). They now share g_keySlot with the
            //     static path (so they HAVE a slot here), so this static loop MUST exclude
            //     every skinned entry — drawing a stride-44 skinned VB through the stride-24
            //     static pipeline would garble it.
            // Terrain now draws (near worldLandscapeRoot patches; flat-shaded geometry).
            if (!e.d3dTexture) continue;
            if (!e.isLandscape && e.blendEnable) continue;
            if (e.isSkinned) continue;
            //   - multi-map: parts with dark/detail/glow siblings go through
            //     buildMultiMapDrawList (their own stride-60 wide VB + multimap pipeline).
            //     They share g_keySlot with the static path (so they HAVE a slot here), so this
            //     static loop MUST exclude them — drawing a stride-60 wide VB through the
            //     stride-36 static pipeline would garble it. Matches the capture-side condition
            //     (non-landscape; landscape is forced single-UV / single-map).
            if (!e.isLandscape && (e.d3dDark || e.d3dDetail || e.d3dGlow)) continue;

            item.slot = ks->second;
            item.texIndex = resolveTextureSlot(e.textureName);   // bindless base map (0 = white)
            // Terrain DECAL_1 overlay (second land texture). resolveTextureSlot ships its DDS
            // bytes the same way as the base map. Non-landscape / single-texture draws get 0,
            // which gates the frag's splat off → byte-for-byte unchanged.
            item.overlayTexIndex = (e.isLandscape && e.d3dOverlay && e.overlayTextureName)
                ? resolveTextureSlot(e.overlayTextureName) : 0u;
            item.alphaRef = e.alphaTest ? e.alphaRef : 0.0f;     // alpha-test cutout (0 = no test)
            // Tier 2b material: ship the captured MaterialProperty colours + the vertex-colour
            // routing, replicating buildCacheReflectionState/buildCacheMainState EXACTLY. useVCol
            // = mesh has colours AND its VertexColorProperty says to use them (else real material
            // colours get white-washed); when off we send vColSource 0 so the frag uses the
            // constant material (vertexMaterialNone), ignoring the VB's colour slot.
            item.matDiffuse[0]  = e.matDiffuse[0];  item.matDiffuse[1]  = e.matDiffuse[1];  item.matDiffuse[2]  = e.matDiffuse[2];
            item.matAmbient[0]  = e.matAmbient[0];  item.matAmbient[1]  = e.matAmbient[1];  item.matAmbient[2]  = e.matAmbient[2];
            item.matEmissive[0] = e.matEmissive[0]; item.matEmissive[1] = e.matEmissive[1]; item.matEmissive[2] = e.matEmissive[2];
            item.vColSource = (e.hasVertexColor && e.vColSource != 0) ? e.vColSource : 0u;
            memcpy(item.world, e.worldTransformD3D, 16 * sizeof(float));
            // CAMERA-RELATIVE rendering: subtract the camera world position from the world
            // translation so vertices reach the shader near the origin. At MW's exterior
            // coordinates (|eye| ~150k) absolute world positions quantise to ~0.01-0.1 units
            // in float32, and the combined viewProj's per-vertex cancellation then produces
            // orientation-dependent stretching. Shifting world + viewProj + lights + eyePos by
            // -eye keeps all vertex math small/precise. (viewProj is built translation-free below.)
            item.world[12] -= DistantLand::eyePos.x;
            item.world[13] -= DistantLand::eyePos.y;
            item.world[14] -= DistantLand::eyePos.z;
            // F12 diagnostic: displace each object by a deterministic per-slot vector. The
            // world matrix is row-major D3DX (translation in m[12..14]); a fixed offset per
            // slot means duplicates of one object (same or different slot) appear as two
            // separated copies. Hash the slot to a pseudo-random ±range.
            if (g_debugMode == 2) {
                const std::uint32_t s = ks->second;
                std::uint32_t h = s * 2654435761u;        // Knuth multiplicative hash
                auto axis = [&](std::uint32_t shift) {
                    std::uint32_t v = (h >> shift) & 0x3FF;   // 10 bits
                    return (static_cast<float>(v) / 1023.0f - 0.5f) * 100.0f;  // ±50 units
                };
                item.world[12] += axis(0);
                item.world[13] += axis(10);
                item.world[14] += axis(20);
            }
            const std::size_t at = g_drawScratch.size();
            g_drawScratch.resize(at + sizeof(item));
            memcpy(g_drawScratch.data() + at, &item, sizeof(item));
            ++count;
        }
        return count;
    }

    // M-Skinning: gather this frame's visible SKINNED parts into g_skinnedScratch as a
    // sequence of [SkinnedDrawWire][palette]. Same visible-set source as buildDrawList
    // (DistantLand::visibleCacheKeys); skinned keys are EXCLUDED from buildDrawList's static
    // loop (it requires the per-draw world matrix; skinned has none), so the two lists are
    // disjoint over the same set. There is no per-draw world transform — the bone palette
    // (read fresh from the cache entry each frame, that IS the animation) is world-space.
    // Returns the packed item count (0 if nothing skinned / skinned path unavailable).
    std::uint32_t buildSkinnedDrawList() {
        if (!g_skinnedVec) {
            return 0;
        }
        const auto& keys = DistantLand::visibleCacheKeys();
        const auto& cacheMap = MGE::GeometryCache::cache();

        g_skinnedScratch.clear();
        std::uint32_t count = 0;
        for (std::uint32_t key : keys) {
            auto ks = g_keySlot.find(key);
            if (ks == g_keySlot.end()) {
                continue;   // not an uploaded part (or not yet shipped)
            }
            auto ce = cacheMap.find(key);
            if (ce == cacheMap.end()) {
                continue;   // evicted since the visible-set build
            }
            const auto& e = ce->second;
            // Only GPU-skinnable parts: a built skinned VB, bones within the palette cap,
            // and a current bone palette of the expected size.
            if (!e.isSkinned || e.skinnedUnsupported || e.numBones == 0) {
                continue;
            }
            if (e.bonePalette.size() < (std::size_t)e.numBones * 16) {
                continue;   // palette not yet built this frame
            }

            IPC::SkinnedDrawWire item;
            item.slot     = ks->second;
            item.numBones = e.numBones;
            item.mirror   = e.mirrored ? 1u : 0u;
            item.texIndex = resolveTextureSlot(e.textureName);   // bindless base map (0 = white)
            item.alphaRef = e.alphaTest ? e.alphaRef : 0.0f;     // alpha-test cutout (0 = no test)

            const std::size_t paletteBytes = (std::size_t)e.numBones * 64;  // numBones * 16 floats
            const std::size_t at = g_skinnedScratch.size();
            g_skinnedScratch.resize(at + sizeof(item) + paletteBytes);
            std::uint8_t* dst = g_skinnedScratch.data() + at;
            memcpy(dst, &item, sizeof(item));                  dst += sizeof(item);
            memcpy(dst, e.bonePalette.data(), paletteBytes);
            // CAMERA-RELATIVE: the bone palette is world-space; shift each bone matrix's
            // translation by -eye so the skinned vertices land near the origin, consistent
            // with the translation-free viewProj + shifted lights (see buildDrawList).
            {
                float* pal = reinterpret_cast<float*>(dst);
                for (std::uint32_t b = 0; b < e.numBones; ++b) {
                    pal[b * 16 + 12] -= DistantLand::eyePos.x;
                    pal[b * 16 + 13] -= DistantLand::eyePos.y;
                    pal[b * 16 + 14] -= DistantLand::eyePos.z;
                }
            }
            ++count;
        }
        return count;
    }

    // Tier 4 multi-map: gather this frame's visible STATIC multi-map parts (dark/detail/glow
    // siblings) into g_multiMapScratch as MultiMapDrawWire[]. Same visible-set source as
    // buildDrawList; multi-map keys are EXCLUDED from buildDrawList's static loop (they need the
    // wide stride-60 VB + multimap pipeline), so the two lists are disjoint over the same set.
    // The ORDERED stage list is built here on the CLIENT, replicating rendercachedcolor.cpp::
    // buildCacheStages EXACTLY (present maps pushed with their op + UV set, stable-sorted by
    // texCoordSet ascending — MW assigns the D3D stage index = texCoordSet, not the map slot).
    // Returns the packed item count (0 if nothing multi-map / path unavailable).
    std::uint32_t buildMultiMapDrawList() {
        if (!g_multiMapVec) {
            return 0;
        }
        const auto& keys = DistantLand::visibleCacheKeys();
        const auto& cacheMap = MGE::GeometryCache::cache();

        g_multiMapScratch.clear();
        std::uint32_t count = 0;
        for (std::uint32_t key : keys) {
            auto ks = g_keySlot.find(key);
            if (ks == g_keySlot.end()) {
                continue;   // not an uploaded part (or not yet shipped)
            }
            auto ce = cacheMap.find(key);
            if (ce == cacheMap.end()) {
                continue;   // evicted since the visible-set build
            }
            const auto& e = ce->second;
            // Must match the capture-side isMultiMap condition + the static loop's filters:
            // non-landscape, textured, opaque, non-skinned, with a dark/detail/glow sibling.
            if (e.isLandscape || e.isSkinned || !e.d3dTexture) continue;
            if (e.blendEnable) continue;
            if (!(e.d3dDark || e.d3dDetail || e.d3dGlow)) continue;

            // Build the ordered stage list, replicating buildCacheStages. A present map is usable
            // only if the wide VB carries the UV set it samples (uv < uvSetCount) — cacheMapActive.
            // Base is pushed unconditionally (e.d3dTexture); the others gated by cacheMapActive.
            struct Stage { const char* name; std::uint8_t uv; std::uint32_t op; };
            Stage st[4];
            int ns = 0;
            st[ns++] = { e.textureName, e.baseUV, IPC::kMMOpBase };
            if (e.d3dDark   && e.darkUV   < e.uvSetCount) st[ns++] = { e.darkTextureName,   e.darkUV,   IPC::kMMOpMod };
            if (e.d3dDetail && e.detailUV < e.uvSetCount) st[ns++] = { e.detailTextureName, e.detailUV, IPC::kMMOpMod2X };
            if (e.d3dGlow   && e.glowUV   < e.uvSetCount) st[ns++] = { e.glowTextureName,   e.glowUV,   IPC::kMMOpAdd };
            // Stable insertion sort by UV ascending (<=4 stages; keeps slot order on ties).
            for (int a = 1; a < ns; ++a) {
                Stage tmp = st[a];
                int b = a - 1;
                while (b >= 0 && st[b].uv > tmp.uv) { st[b + 1] = st[b]; --b; }
                st[b + 1] = tmp;
            }

            IPC::MultiMapDrawWire item = {};
            item.slot = ks->second;
            memcpy(item.world, e.worldTransformD3D, 16 * sizeof(float));
            // CAMERA-RELATIVE: shift translation by -eye (see buildDrawList).
            item.world[12] -= DistantLand::eyePos.x;
            item.world[13] -= DistantLand::eyePos.y;
            item.world[14] -= DistantLand::eyePos.z;
            item.matDiffuse[0]  = e.matDiffuse[0];  item.matDiffuse[1]  = e.matDiffuse[1];  item.matDiffuse[2]  = e.matDiffuse[2];
            item.matAmbient[0]  = e.matAmbient[0];  item.matAmbient[1]  = e.matAmbient[1];  item.matAmbient[2]  = e.matAmbient[2];
            item.matEmissive[0] = e.matEmissive[0]; item.matEmissive[1] = e.matEmissive[1]; item.matEmissive[2] = e.matEmissive[2];
            item.vColSource = (e.hasVertexColor && e.vColSource != 0) ? e.vColSource : 0u;
            item.alphaRef   = e.alphaTest ? e.alphaRef : 0.0f;   // base-stage alpha test
            item.stageCount = (std::uint32_t)ns;
            for (int s = 0; s < ns; ++s) {
                const std::uint32_t tex = resolveTextureSlot(st[s].name);   // bindless slot (0 = white)
                item.stages[s] = IPC::packMMStage(tex, st[s].uv, st[s].op);
            }
            const std::size_t at = g_multiMapScratch.size();
            g_multiMapScratch.resize(at + sizeof(item));
            memcpy(g_multiMapScratch.data() + at, &item, sizeof(item));
            ++count;
        }
        return count;
    }

    // Tier 3a: gather this frame's point lights into g_lightScratch as PointLightWire[]. Source is
    // the MGE scene-graph snapshot (MGE::SceneGraph::pointLights()) — the same NI::PointLight POD
    // the FFE many-lights path consumes — read under the SnapshotReadLock (the async walk swaps the
    // vector atomically). World-space, no culling (Tier 3a is correctness-first; the frag loops the
    // whole set). Diffuse is already dimmer-scaled; pointLightMult is baked here (1.0 on the main
    // cache path → identity). Clamped to kMaxPointLights (logged once if exceeded). Returns the count.
    std::uint32_t buildLightList() {
        if (!g_lightVec) {
            return 0;
        }
        // The main cache color path (rendercachedcolor.cpp) renders the SAME opaque set Forge does
        // and passes pointLightMult = 1.0f, so bake 1.0 (kept explicit for future per-frame scaling).
        constexpr float pointLightMult = 1.0f;

        MGE::SceneGraph::SnapshotReadLock lk;
        const auto& lights = MGE::SceneGraph::pointLights();

        g_lightScratch.clear();
        std::uint32_t count = 0;
        for (const auto& pl : lights) {
            if (count >= IPC::kMaxPointLights) {
                static bool logged = false;
                if (!logged) {
                    LOG::logline("!! [light] %zu point lights this frame > cap %u — extra dropped (Tier 3b clustering lifts this)",
                                 lights.size(), IPC::kMaxPointLights);
                    logged = true;
                }
                break;
            }
            IPC::PointLightWire w;
            // CAMERA-RELATIVE: light positions are compared against the (now camera-relative)
            // WorldPos in the frag, so shift them by -eye too (see buildDrawList).
            w.posRadius[0] = pl.worldPos[0] - DistantLand::eyePos.x;
            w.posRadius[1] = pl.worldPos[1] - DistantLand::eyePos.y;
            w.posRadius[2] = pl.worldPos[2] - DistantLand::eyePos.z;
            w.posRadius[3] = pl.radius;
            w.color[0] = pl.diffuse[0] * pointLightMult;
            w.color[1] = pl.diffuse[1] * pointLightMult;
            w.color[2] = pl.diffuse[2] * pointLightMult;
            w.color[3] = 0.0f;
            w.falloff[0] = pl.falloff[0];
            w.falloff[1] = pl.falloff[1];
            w.falloff[2] = pl.falloff[2];
            w.falloff[3] = 0.0f;
            const std::size_t at = g_lightScratch.size();
            g_lightScratch.resize(at + sizeof(w));
            memcpy(g_lightScratch.data() + at, &w, sizeof(w));
            ++count;
        }
        return count;
    }
}

namespace RenderProcess {
    void init(IPC::Client* client, IDirect3DDevice9* device) {
        if (!Configuration.UseRenderProcess) {
            return;
        }
        g_client = client;
        if (!device) {
            LOG::logline("!! [seam] no device at init; seam disabled");
            return;
        }
        // Bring up the seam NOW, under the "...MGE XE..." loading bar (this runs inside
        // DistantLand::init()), so the shared texture is live by the main menu.
        lazyInit(device);
    }

    void onStage0Composite(IDirect3DDevice9* device) {
        if (!device || !g_initOk) {
            return;
        }

        const double tStart = nowMs();
        const double dtPresent = (g_lastPresentMs > 0.0) ? (tStart - g_lastPresentMs) : 0.0;
        g_lastPresentMs = tStart;

        // Ship any geometry the cache captured this frame (independent of the F11
        // composite toggle, so the host's mesh store is ready when we turn it on).
        const std::uint32_t geomParts = g_pendingParts;            // snapshot (flush clears it)
        const std::size_t   geomBytes = g_pendingBlob.size();
        flushGeometry();
        const double tGeomFlush = nowMs();

        // Live toggle (debug key). Edge-triggered.
        if (GetAsyncKeyState(VK_F11) & 0x0001) {
            g_enabled = !g_enabled;
            LOG::logline(">> [seam] composite %s", g_enabled ? "ON" : "OFF");
        }
        // F12 diagnostic: scatter each object by a fixed per-slot offset so any object that
        // is drawn more than once appears as TWO separated copies of the same mesh (a single
        // draw just looks displaced). Reveals duplicate draws regardless of source.
        if (GetAsyncKeyState(VK_F12) & 0x0001) {
            // Tier 2 GTAO added modes 3 (AO buffer) + 4 (bent normal); cycle is now %5.
            g_debugMode = (g_debugMode + 1) % 5;
            const char* name = (g_debugMode == 1) ? "DEPTH" : (g_debugMode == 2) ? "SCATTER"
                             : (g_debugMode == 3) ? "AO" : (g_debugMode == 4) ? "BENT NORMAL" : "NORMAL";
            LOG::logline(">> [seam] debug mode %d (%s)", g_debugMode, name);
        }
        if (!g_enabled) {
            return;
        }

        const unsigned frame = g_frame++;

        // Drive the host renderer into the shared RT. Blocking — the host fence-waits
        // before replying, so the draw is GPU-complete and the resource is quiescent
        // before our copy. M1c: build this frame's visible draw list (camera + slots +
        // world transforms) and render the cached opaque SCENE; fall back to the
        // triangle if the scene path or draw data isn't available.
        double hostMs = 0.0;
        bool ok = false;
        const std::uint32_t drawCount = buildDrawList();
        const std::uint32_t skinnedCount = buildSkinnedDrawList();
        const std::uint32_t multiMapCount = buildMultiMapDrawList();
        const std::uint32_t lightCount = buildLightList();
        const double tBuild = nowMs();

        // Ship any textures newly referenced this frame BEFORE the scene draw that uses them
        // (buildDrawList queued their DDS via resolveTextureSlot). Lazy: only first-seen textures.
        const std::uint32_t texCount = g_texPendingCount;          // snapshot (flush clears it)
        const std::size_t   texBytes = g_texPendingBlob.size();
        flushTextures();
        const double tTexFlush = nowMs();

        const bool haveDraw = g_drawVec && drawCount > 0
            && g_drawVec->assign_bytes(g_drawScratch.data(), (std::uint32_t)g_drawScratch.size());

        IPC::VecId   skinnedId = IPC::InvalidVector;
        std::uint32_t skinnedBytes = 0;
        if (g_skinnedVec && skinnedCount > 0
            && g_skinnedVec->assign_bytes(g_skinnedScratch.data(), (std::uint32_t)g_skinnedScratch.size())) {
            skinnedId    = g_skinnedVec->id();
            skinnedBytes = (std::uint32_t)g_skinnedScratch.size();
        }

        IPC::VecId   multiMapId = IPC::InvalidVector;
        std::uint32_t multiMapBytes = 0;
        if (g_multiMapVec && multiMapCount > 0
            && g_multiMapVec->assign_bytes(g_multiMapScratch.data(), (std::uint32_t)g_multiMapScratch.size())) {
            multiMapId    = g_multiMapVec->id();
            multiMapBytes = (std::uint32_t)g_multiMapScratch.size();
        }

        IPC::VecId   lightId = IPC::InvalidVector;
        std::uint32_t lightBytes = 0;
        if (g_lightVec && lightCount > 0
            && g_lightVec->assign_bytes(g_lightScratch.data(), (std::uint32_t)g_lightScratch.size())) {
            lightId    = g_lightVec->id();
            lightBytes = (std::uint32_t)g_lightScratch.size();
        }
        const double tAssign = nowMs();

        // Only drive + composite the host when there's actual scene data this frame. With no
        // draw list (loading doors, menus, empty cells) we must NOT fall back to the bring-up
        // triangle and composite it — that flashes the debug triangle over MW's loading/menu
        // frame. Skip the seam entirely and let MW present its own (fixed-function) frame.
        if (!haveDraw && skinnedId == IPC::InvalidVector && multiMapId == IPC::InvalidVector) {
            return;
        }

        // CAMERA-RELATIVE viewProj: zero the view matrix's translation row so the camera sits at
        // the origin (the world translations above are pre-shifted by -eye, so this cancels exactly:
        // eyePos == inverse(mwView)·origin, hence mwView's translation row == -eye·R). This keeps the
        // whole vertex pipeline near the origin and eliminates the float32 large-world stretching.
        D3DXMATRIX viewRel = DistantLand::mwView;
        viewRel._41 = viewRel._42 = viewRel._43 = 0.0f;
        D3DXMATRIX viewProj;
        D3DXMatrixMultiply(&viewProj, &viewRel, &DistantLand::mwProj);

        // Tier 1 lighting (6 × float4): MW sun/ambient/fog for this frame, uploaded into the
        // host gFrameData after viewProj. sunVec is the world-space sun TRAVEL direction (the
        // shader does dot(N, -sunDir)); fogParams = (fogNearStart, fogNearEnd); dist for fog is
        // |worldPos - eyePos| in-shader.
        //
        // Sun/ambient use the CANONICAL MGE PPL/DL formula (distantland.cpp:780-792, :1141):
        //   sun     = lightSunMult * sunCol
        //   ambient = lightAmbMult * (sunAmb + ambCol)   // ambCol alone == globalAmbient, often
        //                                                 // ~0 in exteriors; the ambient fill is
        //                                                 // mostly the sun light's sunAmb term.
        const RGBVECTOR sunColEff = DistantLand::lightSunMult * DistantLand::sunCol;
        const RGBVECTOR ambColEff = DistantLand::lightAmbMult * (DistantLand::sunAmb + DistantLand::ambCol);
        const float lighting[24] = {
            DistantLand::sunVec.x,     DistantLand::sunVec.y,     DistantLand::sunVec.z,     0.0f,
            sunColEff.r,               sunColEff.g,               sunColEff.b,               0.0f,
            ambColEff.r,               ambColEff.g,               ambColEff.b,               0.0f,
            DistantLand::nearFogCol.r, DistantLand::nearFogCol.g, DistantLand::nearFogCol.b, 0.0f,
            DistantLand::fogNearStart, DistantLand::fogNearEnd,   0.0f,                      0.0f,
            // CAMERA-RELATIVE: WorldPos reaches the shader already relative to the eye, so the
            // eyePos used for the per-vertex fog distance |worldPos - eyePos| is the origin (0).
            0.0f,                      0.0f,                      0.0f,                      0.0f,
        };

        ok = g_client->renderSceneBlocking(frame, (const float*)&viewProj, lighting,
                 haveDraw ? g_drawVec->id() : IPC::InvalidVector,
                 haveDraw ? drawCount : 0,
                 haveDraw ? (std::uint32_t)g_drawScratch.size() : 0,
                 skinnedId, skinnedCount, skinnedBytes,
                 multiMapId, (multiMapId != IPC::InvalidVector) ? multiMapCount : 0, multiMapBytes,
                 lightId, (lightId != IPC::InvalidVector) ? lightCount : 0, lightBytes,
                 (std::uint32_t)g_debugMode, &hostMs);
        const double tRender = nowMs();
        if (!ok) {
            return;
        }

        if (!copyHostRtToDst()) {
            static bool logged = false;
            if (!logged) { LOG::logline("!! [seam] copyHostRtToDst failed"); logged = true; }
            return;
        }
        const double tCopy = nowMs();

        // Composite the Forge layer (g_mainTex) OVER MW's frame as a full-screen textured quad
        // with PREMULTIPLIED alpha blend. The Forge RT clears to alpha=0 and geometry writes
        // alpha=1, so alpha is a coverage mask: sky/distant land (already on the backbuffer from
        // renderStage0) show through alpha=0 regions and alpha-test holes. Premultiplied
        // (SRCBLEND=ONE, DESTBLEND=INVSRCALPHA) — NOT SRCALPHA — because MSAA resolve leaves
        // edge pixels premultiplied (rgb already scaled by partial coverage); SRCALPHA would
        // darken edges. State-blocked so nothing leaks into MW's scene 1 (every DistantLand
        // stage does this). -0.5 px offset + POINT filter = the 1:1 texel mapping StretchRect
        // gave (the geometry half-pixel was already corrected host-side).
        {
            IDirect3DStateBlock9* sb = nullptr;
            device->CreateStateBlock(D3DSBT_ALL, &sb);

            IDirect3DSurface9* backbuffer = nullptr;
            if (SUCCEEDED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backbuffer)) && backbuffer) {
                device->SetRenderTarget(0, backbuffer);
                backbuffer->Release();
            }

            device->SetPixelShader(nullptr);
            device->SetVertexShader(nullptr);
            device->SetFVF(D3DFVF_XYZRHW | D3DFVF_TEX1);
            device->SetTexture(0, g_mainTex);

            device->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
            device->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_ONE);
            device->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA);
            device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
            device->SetRenderState(D3DRS_ZENABLE, FALSE);
            device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
            device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
            device->SetRenderState(D3DRS_LIGHTING, FALSE);
            device->SetRenderState(D3DRS_FOGENABLE, FALSE);
            device->SetRenderState(D3DRS_STENCILENABLE, FALSE);
            device->SetRenderState(D3DRS_COLORWRITEENABLE, 0x0F);

            device->SetTextureStageState(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1);
            device->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
            device->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1);
            device->SetTextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
            device->SetTextureStageState(1, D3DTSS_COLOROP, D3DTOP_DISABLE);
            device->SetTextureStageState(1, D3DTSS_ALPHAOP, D3DTOP_DISABLE);
            // CRITICAL: disable texture-coordinate transformation. MW leaves a VIEW-DEPENDENT
            // texture matrix active for environment/sphere-map reflections; without this the FF
            // pipeline would transform our blit UVs by that matrix, shearing the composited image
            // as the camera rotates (host output g_mainTex is correct; only the sampled blit skews).
            device->SetTextureStageState(0, D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_DISABLE);
            device->SetTextureStageState(1, D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_DISABLE);
            // CRITICAL: MW leaves environment/sphere-map TEXGEN active (D3DTSS_TEXCOORDINDEX carries
            // a TCI_CAMERASPACE* flag). With texgen the FF pipeline SYNTHESISES texcoords from
            // camera-space position/normal and IGNORES the vertex UVs — disabling the texture matrix
            // alone isn't enough (the generated coords are the zoom/skew gradient we saw). Force the
            // stage to read vertex texcoord set 0 with no texgen.
            device->SetTextureStageState(0, D3DTSS_TEXCOORDINDEX, 0);
            // Belt-and-suspenders: also neutralise the texture matrix itself. Identity == passthrough.
            {
                D3DMATRIX ident = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
                device->SetTransform(D3DTS_TEXTURE0, &ident);
            }
            device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
            device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
            device->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
            device->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);

            const float fw = (float)g_w, fh = (float)g_h;
            struct CV { float x, y, z, rhw, u, v; };
            const CV quad[4] = {
                { -0.5f,      -0.5f,      0.0f, 1.0f, 0.0f, 0.0f },
                { fw - 0.5f,  -0.5f,      0.0f, 1.0f, 1.0f, 0.0f },
                { -0.5f,      fh - 0.5f,  0.0f, 1.0f, 0.0f, 1.0f },
                { fw - 0.5f,  fh - 0.5f,  0.0f, 1.0f, 1.0f, 1.0f },
            };
            device->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, quad, sizeof(CV));

            device->SetTexture(0, nullptr);
            if (sb) { sb->Apply(); sb->Release(); }
        }
        const double tEnd = nowMs();

        // Spike log: one breakdown line when the client feed blew the budget. render =
        // the blocking host RPC (host's own GPU time = host[]); render - host = IPC/wait
        // stall. texflush is the prime suspect for the *periodic* dips (textures stream in
        // as you cross into new cells). dt = inter-present delta (the visible dip).
        const double feed = tEnd - tStart;

        // Baseline heartbeat over every composited frame (sees the <kSpikeMs majority).
        g_hb.feed += feed; g_hb.render += (tRender - tAssign); g_hb.host += hostMs;
        g_hb.copy += (tCopy - tRender); g_hb.dt += dtPresent;
        if (feed > g_hb.maxFeed) g_hb.maxFeed = feed;
        if (dtPresent > g_hb.maxDt) g_hb.maxDt = dtPresent;
        if (++g_hb.n >= kHeartbeatFrames) {
            LOG::logline(">> [hb] %u frames avg: feed=%.2f render=%.2f[host=%.2f] copy=%.2f dt=%.2f | "
                         "max feed=%.2f dt=%.2f (~%.0f fps)",
                         g_hb.n, g_hb.feed / g_hb.n, g_hb.render / g_hb.n, g_hb.host / g_hb.n,
                         g_hb.copy / g_hb.n, g_hb.dt / g_hb.n, g_hb.maxFeed, g_hb.maxDt,
                         g_hb.dt > 0.0 ? 1000.0 * g_hb.n / g_hb.dt : 0.0);
            g_hb = Accum{};
        }

        if (feed >= kSpikeMs) {
            LOG::logline("!! [spike] frame %u feed=%.2fms (geomflush=%.2f build=%.2f texflush=%.2f "
                         "assign=%.2f render=%.2f[host=%.2f] copy=%.2f blit=%.2f) "
                         "draws=%u skin=%u mm=%u light=%u geom+=%u/%uKB tex+=%u/%uKB dt=%.2fms",
                         frame, feed,
                         tGeomFlush - tStart, tBuild - tGeomFlush, tTexFlush - tBuild,
                         tAssign - tTexFlush, tRender - tAssign, hostMs, tCopy - tRender, tEnd - tCopy,
                         drawCount, skinnedCount, multiMapCount, lightCount,
                         geomParts, (unsigned)(geomBytes >> 10), texCount, (unsigned)(texBytes >> 10),
                         dtPresent);
        }
    }

    bool wantsGeometryCapture() {
        return g_initOk && g_geomVec.has_value();
    }

    bool ownsOpaqueWorld() {
        // Forge composites a full-screen blit over MW's frame when enabled; the engine's
        // scene-0 opaque draw underneath is then pure wasted cost. Gate on the live composite
        // toggle so F11-off restores normal engine rendering (clean A/B of the double cost).
        return g_initOk && g_enabled;
    }

    void captureGeometry(std::uint32_t key, std::uint16_t revision, std::uint32_t modelId,
                         const IPC::GeomVertexWire* verts, std::uint32_t vertexCount,
                         const std::uint16_t* indices, std::uint32_t indexCount) {
        if (!wantsGeometryCapture() || !verts || !indices || !vertexCount || !indexCount) {
            return;
        }
        // Skip only if the SAME object (modelId), same shape (vertexCount) and same revision
        // was already shipped. Keying on revision alone aliased recycled NiTriShape* keys (a
        // freed object's key+slot inherited by a new mesh with a colliding revisionID).
        auto rev = g_uploadedRev.find(key);
        if (rev != g_uploadedRev.end() && rev->second.id == modelId &&
            rev->second.vc == vertexCount && rev->second.rev == revision) {
            return;
        }
        // Stable host slot per cache key (reused on re-upload so the host frees
        // and rebuilds in place).
        std::uint32_t slot;
        auto ks = g_keySlot.find(key);
        if (ks != g_keySlot.end()) {
            slot = ks->second;
        } else {
            slot = g_nextSlot++;
            g_keySlot.emplace(key, slot);
        }

        IPC::GeomPartWire hdr = {};
        hdr.slot        = slot;
        hdr.revisionID  = revision;
        hdr.vertexCount = vertexCount;
        hdr.indexCount  = indexCount;

        const std::size_t vbBytes = (std::size_t)vertexCount * sizeof(IPC::GeomVertexWire);
        const std::size_t ibBytes = (std::size_t)indexCount * sizeof(std::uint16_t);
        const std::size_t at = g_pendingBlob.size();
        g_pendingBlob.resize(at + sizeof(hdr) + vbBytes + ibBytes);
        std::uint8_t* dst = g_pendingBlob.data() + at;
        memcpy(dst, &hdr, sizeof(hdr));            dst += sizeof(hdr);
        memcpy(dst, verts, vbBytes);               dst += vbBytes;
        memcpy(dst, indices, ibBytes);

        ++g_pendingParts;
        g_uploadedRev[key] = { modelId, vertexCount, revision };
    }

    void captureSkinnedGeometry(std::uint32_t key, std::uint16_t revision, std::uint32_t modelId,
                                const IPC::SkinnedVertexWire* verts, std::uint32_t vertexCount,
                                const std::uint16_t* indices, std::uint32_t indexCount,
                                std::uint32_t numBones) {
        if (!wantsGeometryCapture() || !verts || !indices || !vertexCount || !indexCount || numBones == 0) {
            return;
        }
        // Skip only if the same object+shape+revision was already shipped (shared map with
        // captureGeometry); identity (modelId,vc) guards against recycled NiTriShape* keys.
        auto rev = g_uploadedRev.find(key);
        if (rev != g_uploadedRev.end() && rev->second.id == modelId &&
            rev->second.vc == vertexCount && rev->second.rev == revision) {
            return;
        }
        // Stable host slot per cache key (shared slot map / nextSlot with the static path).
        std::uint32_t slot;
        auto ks = g_keySlot.find(key);
        if (ks != g_keySlot.end()) {
            slot = ks->second;
        } else {
            slot = g_nextSlot++;
            g_keySlot.emplace(key, slot);
        }

        IPC::GeomPartWire hdr = {};
        hdr.slot        = slot;
        hdr.revisionID  = revision;
        hdr.flags       = IPC::kGeomFlagSkinned;
        hdr.vertexCount = vertexCount;
        hdr.indexCount  = indexCount;
        hdr.numBones    = static_cast<std::uint16_t>(numBones);

        const std::size_t vbBytes = (std::size_t)vertexCount * sizeof(IPC::SkinnedVertexWire);
        const std::size_t ibBytes = (std::size_t)indexCount * sizeof(std::uint16_t);
        const std::size_t at = g_pendingBlob.size();
        g_pendingBlob.resize(at + sizeof(hdr) + vbBytes + ibBytes);
        std::uint8_t* dst = g_pendingBlob.data() + at;
        memcpy(dst, &hdr, sizeof(hdr));            dst += sizeof(hdr);
        memcpy(dst, verts, vbBytes);               dst += vbBytes;
        memcpy(dst, indices, ibBytes);

        ++g_pendingParts;
        g_uploadedRev[key] = { modelId, vertexCount, revision };
    }

    void captureMultiMapGeometry(std::uint32_t key, std::uint16_t revision, std::uint32_t modelId,
                                 const IPC::GeomVertexWireMM* verts, std::uint32_t vertexCount,
                                 const std::uint16_t* indices, std::uint32_t indexCount) {
        if (!wantsGeometryCapture() || !verts || !indices || !vertexCount || !indexCount) {
            return;
        }
        // Dedup on (modelId, vertexCount, revision) — shared map with captureGeometry; identity
        // guards against recycled NiTriShape* keys (see captureGeometry).
        auto rev = g_uploadedRev.find(key);
        if (rev != g_uploadedRev.end() && rev->second.id == modelId &&
            rev->second.vc == vertexCount && rev->second.rev == revision) {
            return;
        }
        // Stable host slot per cache key (shared slot map / nextSlot with the static path).
        std::uint32_t slot;
        auto ks = g_keySlot.find(key);
        if (ks != g_keySlot.end()) {
            slot = ks->second;
        } else {
            slot = g_nextSlot++;
            g_keySlot.emplace(key, slot);
        }

        IPC::GeomPartWire hdr = {};
        hdr.slot        = slot;
        hdr.revisionID  = revision;
        hdr.flags       = IPC::kGeomFlagMultiMap;
        hdr.vertexCount = vertexCount;
        hdr.indexCount  = indexCount;

        const std::size_t vbBytes = (std::size_t)vertexCount * sizeof(IPC::GeomVertexWireMM);
        const std::size_t ibBytes = (std::size_t)indexCount * sizeof(std::uint16_t);
        const std::size_t at = g_pendingBlob.size();
        g_pendingBlob.resize(at + sizeof(hdr) + vbBytes + ibBytes);
        std::uint8_t* dst = g_pendingBlob.data() + at;
        memcpy(dst, &hdr, sizeof(hdr));            dst += sizeof(hdr);
        memcpy(dst, verts, vbBytes);               dst += vbBytes;
        memcpy(dst, indices, ibBytes);

        ++g_pendingParts;
        g_uploadedRev[key] = { modelId, vertexCount, revision };
    }

    void shutdown() {
        releaseAll();
        g_geomVec.reset();
        g_drawVec.reset();
        g_skinnedVec.reset();
        g_multiMapVec.reset();
        g_lightVec.reset();
        g_texVec.reset();
        g_pendingBlob.clear();
        g_pendingBlob.shrink_to_fit();
        g_drawScratch.clear();
        g_drawScratch.shrink_to_fit();
        g_skinnedScratch.clear();
        g_skinnedScratch.shrink_to_fit();
        g_multiMapScratch.clear();
        g_multiMapScratch.shrink_to_fit();
        g_lightScratch.clear();
        g_lightScratch.shrink_to_fit();
        g_texPendingBlob.clear();
        g_texPendingBlob.shrink_to_fit();
        g_pendingParts = 0;
        g_texPendingCount = 0;
        g_keySlot.clear();
        g_uploadedRev.clear();
        g_texSlot.clear();
        g_nextSlot = 0;
        g_nextTexSlot = 1;
        g_initOk = false;
        g_enabled = false;
    }
}
