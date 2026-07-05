#include "renderprocess.h"
#include "configuration.h"
#include "ipc/client.h"
#include "ipc/geomwire.h"
#include "support/log.h"
#include "dxvk_interop.h"
#include "distantland.h"
#include "mwbridge.h"
#include "scenegraph_geometry_cache.h"
#include "scenegraph.h"
#include "morrowindbsa.h"
#include "mge_tracy.h"

#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    bool   g_skyEnabled = true;        // Sky takeover is DONE → Forge sky is now ALWAYS ON (no longer toggled).
    bool   g_waterEnabled = true;      // Forge water takeover is DONE → ON by default ("always on"); F7 still toggles OFF for A/B vs MW water
    int    g_debugMode = 0;            // F12 diagnostic cycle: 0=normal, 1=depth (world-distance), 2=scatter, 3=AO, 4=bent normal
    unsigned g_frame = 0;

    // Dev overlay (Stage 2): F9 toggles the in-host Forge panel; mouse is polled each frame and
    // forwarded over the renderScene RPC. The host injects it into Forge UI (uiSetExternalInput).
    bool   g_devUiVisible = false;     // F9; default off so it never blocks normal play
    HWND   g_devHwnd = nullptr;        // MW focus window (cached from device creation params)
    bool   g_reloadShadersPending = false; // F8 latched at composite finish, consumed by the next kickoff

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
    //
    // `overlap` = wall time between kickoff-return and finish-entry: MW frame work the host
    // render ran UNDER (recovered time). It is NOT part of `feed` — feed stays the client's
    // own cost (kickoff prep + residual wait + copy + blit), so feed is comparable across
    // fused/async modes and IS the perf baseline once the wait bucket collapses.
    constexpr unsigned kHeartbeatFrames = 300;
    struct Accum { double feed, geom, build, render, host, overlap, copy, blit, dt; double maxFeed, maxDt; double captured; unsigned n, earlyN; };
    Accum g_hb = {};

    // Async-frame split: everything the finish half needs from the kickoff half. Reset at
    // every kickoff entry; rpcPending=true only when renderSceneKickoff actually started the
    // host (finish no-ops otherwise, so every kickoff early-out stays a whole-frame no-op).
    struct KickState {
        bool rpcPending;
        bool early;     // fired from the BeginScene(0) site (DistantLand::earlyForgeKickoff latch)
        unsigned frame;
        std::uint32_t drawCount, skinnedCount, multiMapCount, lightCount, skyCount, alphaCount;
        std::uint32_t capturedCount;   // AT3: captured blended DIPs merged into the alpha list this frame
        std::uint32_t geomParts, texCount;
        std::size_t geomBytes, texBytes;
        double dtPresent;
        double tStart, tGeomFlush, tBuild, tTexFlush, tAssign, tKick;
    };
    KickState g_kick = {};

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
    std::optional<IPC::VecView<IPC::GeomChunk>> g_skyVec;           // persistent per-frame sky draw-list vec (SK1)
    std::optional<IPC::VecView<IPC::GeomChunk>> g_alphaVec;         // persistent per-frame sorted-alpha draw-list vec (AT1)
    std::vector<std::uint8_t>                 g_pendingBlob;        // packed parts awaiting flush
    std::uint32_t                             g_pendingParts = 0;
    // Cache key -> host slot, plus per-key cached bindless texture slots so the
    // per-frame draw-list build doesn't re-run resolveTextureSlot's normalize
    // (heap string) + string-hash find for every item every frame. A cached slot
    // is valid iff BOTH hold:
    //   - the entry's texture-name POINTER is unchanged (NiSourceTexture fileName
    //     storage is stable; NiFlipController animation swaps to a different
    //     SourceTexture = different pointer, so animated textures still re-resolve)
    //   - the epoch matches g_texEpoch (bumped when the LRU recycles a bindless
    //     slot to a new texture, which invalidates every cached slot value).
    // The cached fast path still refreshes g_slotLastUsed so the LRU stays exact.
    struct SlotInfo {
        std::uint32_t slot = 0;              // host geometry slot (stable per key)
        const char*   baseNamePtr = nullptr; // texture-name identity for baseSlot
        std::uint32_t baseSlot = 0;
        std::uint32_t baseEpoch = 0;
        const char*   ovNamePtr = nullptr;   // texture-name identity for ovSlot
        std::uint32_t ovSlot = 0;
        std::uint32_t ovEpoch = 0;           // separate epochs: one field re-validating
                                             // the other's stale slot after a recycle
                                             // would alias textures
    };
    std::unordered_map<std::uint32_t, SlotInfo> g_keySlot;
    std::uint32_t g_texEpoch = 0;            // bumped on bindless-slot LRU recycle
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
    std::vector<std::uint8_t>                 g_skyScratch;         // packed SkyDrawWire[] this frame (SK1)
    std::vector<std::uint8_t>                 g_alphaScratch;       // packed AlphaDrawWire[] this frame (AT1, back-to-front)

    // --- AT3 captured-alpha (particles/smoke/flames restored under Forge) ------------
    // MW already did the CPU billboarding + sort for these blended DIPs before the reject gate
    // (distantland.cpp inspectIndexedPrimitive). captureAlphaDraw Locks the live VB/IB, copies
    // the final verts + rebased indices into pending scratch DURING frame N, then the NEXT
    // buildGeometryDrawLists (kickoff N+1) merges them into the sorted-alpha list (ONE sort with
    // the cached blends) and ships the geometry to the host. Single-buffered: one pending scratch,
    // consumed + cleared at the next consume. 1-frame-late particle positions are accepted (they
    // were invisible before this feature); camera-relative subtraction uses the CURRENT eyePos at
    // emit so there is no swim.
    struct CapturedAlphaRec {
        std::uint32_t vertexBase, vertexCount;   // into g_capVertScratch
        std::uint32_t indexBase, indexCount;     // into g_capIdxScratch
        float world[16];                          // ABSOLUTE model->world (emit subtracts eyePos)
        float centroid[3];                        // ABSOLUTE world-space centroid (depth sort)
        std::uint32_t texIndex;
        std::uint32_t srcBlend, destBlend;
        float alphaRef;
        std::uint32_t vColSource;
        float matDiffuse[3], matAmbient[3], matEmissive[3], matAlpha;
    };
    std::vector<IPC::GeomVertexWire> g_capVertScratch;   // pending captured verts (frame N)
    std::vector<std::uint16_t>       g_capIdxScratch;    // pending captured indices (frame N)
    std::vector<CapturedAlphaRec>    g_capRecs;          // pending captured records (frame N)
    std::optional<IPC::VecView<IPC::GeomChunk>> g_capturedVec;   // shipped at kickoff ([verts][indices])
    // Old-msoc double-draw guard: (d3dTexture<<32 | vertCount) of every cached blended shape the
    // host already draws this frame; a captured DIP that matches one is a duplicate and is skipped
    // (built in buildGeometryDrawLists while pushing alphaCands; consumed at the gate this frame).
    std::unordered_set<std::uint64_t> g_alphaDedup;
    // Per-frame texture-slot memo for captureAlphaDraw (cleared at consume): rs.texture ptr -> slot.
    std::unordered_map<IDirect3DTexture9*, std::uint32_t> g_capTexMemo;
    // Drop counters (one-shot logged): lock fail / cap overflow / INDEX32 rebase / no-name-white.
    std::uint32_t g_capDropLock = 0, g_capDropCap = 0, g_capDropIdx32 = 0, g_capNoName = 0;
    // Captured blended DIPs merged into the alpha list this frame (heartbeat cap=N). Set in
    // buildGeometryDrawLists before the records are cleared; read into KickState at kickoff.
    std::uint32_t g_capturedEmitted = 0;

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
    // LRU eviction over the client's bindless range [1, kMaxTextures-kDlReserve). Residency is
    // cumulative all session (no per-cell reset), so without eviction a long traversal exhausts the
    // slots and every NEW near texture goes white ("near white far from spawn"). Recycle the
    // least-recently-used slot instead. g_frame is the LRU clock.
    std::vector<std::string>                  g_slotName;           // slot -> name (for eviction; size kMaxTextures)
    std::vector<std::uint32_t>                g_slotLastUsed;       // slot -> last g_frame it was referenced

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

        // Sky draw list (SK1) rides its own 1-chunk vec — kMaxSkyDraws * 88B ≈ 5.6KB, far under 1MB.
        // SkyDrawWire[] alpha-blended sky shapes, rebuilt each frame from the cache's isSky entries.
        auto kv = g_client->allocVecBlocking<IPC::GeomChunk>(1, 1, 1);
        if (!kv) {
            LOG::logline("!! [seam] sky draw-list vec alloc failed — Forge sky (SK1) disabled");
        } else {
            g_skyVec.emplace(std::move(*kv));
        }

        // Sorted-alpha draw list (AT1) rides its own 1-chunk vec — kMaxAlphaDraws * 128B = 128KB,
        // well under 1MB. AlphaDrawWire[] back-to-front sorted, rebuilt each frame from the
        // blendEnable cache entries the opaque lists skip.
        auto av = g_client->allocVecBlocking<IPC::GeomChunk>(1, 1, 1);
        if (!av) {
            LOG::logline("!! [seam] alpha draw-list vec alloc failed — Forge alpha (AT1) disabled");
        } else {
            g_alphaVec.emplace(std::move(*av));
        }

        // AT3 captured-alpha geometry vec: one 1-chunk (1MB) vec carrying [captured verts][captured
        // indices] — 20000 verts (720KB) + 60000 uint16 (120KB) = 840KB fits one chunk (see
        // kMaxCapturedAlpha* in geomwire.h). A 1-chunk vec sidesteps the IPC uint32-reservation hazard.
        auto cv = g_client->allocVecBlocking<IPC::GeomChunk>(1, 1, 1);
        if (!cv) {
            LOG::logline("!! [seam] captured-alpha vec alloc failed — Forge captured alpha (AT3) disabled");
        } else {
            g_capturedVec.emplace(std::move(*cv));
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
        LOG::logline(">> [seam] scene vecs ready (geom %u, draw %u, skinned %u, multimap 1, light 1, sky 1, alpha 1, tex %u chunks)",
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
        const std::uint32_t cap = IPC::kMaxTextures - IPC::kDlReserve;   // client range [1, cap)
        if (g_slotName.size() != IPC::kMaxTextures) {
            g_slotName.assign(IPC::kMaxTextures, std::string());
            g_slotLastUsed.assign(IPC::kMaxTextures, 0u);
        }
        auto it = g_texSlot.find(name);
        if (it != g_texSlot.end()) {
            if (it->second != 0) { g_slotLastUsed[it->second] = g_frame; }   // refresh LRU age
            return it->second;   // already resolved (slot or cached-miss 0)
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

        // Assign a slot: grow while the range has room, else recycle the least-recently-used slot.
        std::uint32_t slot;
        if (g_nextTexSlot < cap) {
            slot = g_nextTexSlot++;
        } else {
            std::uint32_t lru = 1, best = 0xFFFFFFFFu;
            for (std::uint32_t s = 1; s < cap; ++s) {
                if (g_slotLastUsed[s] < best) { best = g_slotLastUsed[s]; lru = s; }
            }
            if (best == g_frame) {
                static bool warned = false;   // working set > capacity this frame: unavoidable thrash
                if (!warned) { LOG::logline("!! [tex] working set exceeds %u client slots — thrashing (white)", cap); warned = true; }
            }
            g_texSlot.erase(g_slotName[lru]);   // evict the recycled name
            slot = lru;
            // The recycled slot now means a different texture: every SlotInfo-cached
            // slot value is suspect. Epoch bump forces per-key re-resolve (one-off).
            ++g_texEpoch;
        }
        g_texSlot[name] = slot;
        g_slotName[slot] = name;
        g_slotLastUsed[slot] = g_frame;
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
    // Resolve a bindless texture slot through a SlotInfo cache field (see SlotInfo).
    // Fast path = pointer-identity + epoch check, no string normalize/hash. The
    // LRU age refresh matches what resolveTextureSlot's memoized path would do.
    std::uint32_t resolveCachedSlot(const char* name, const char*& namePtr,
                                    std::uint32_t& slotVal, std::uint32_t& slotEpoch) {
        if (name == namePtr && slotEpoch == g_texEpoch) {
            if (slotVal != 0) { g_slotLastUsed[slotVal] = g_frame; }
            return slotVal;
        }
        slotVal = resolveTextureSlot(name);
        namePtr = name;
        slotEpoch = g_texEpoch;
        return slotVal;
    }

    // Emit one STATIC opaque draw (pre-filtered by buildGeometryDrawLists — the
    // per-entry filter rationale lives there). Mirrors the PROVEN D3D9 cache color
    // pass's per-entry packing (drawEntry in rendercachedcolor.cpp) so the Forge
    // draw list draws the same set. Terrain draws too (near worldLandscapeRoot
    // patches; flat-shaded geometry).
    // TEMP normal-flip diagnostic (remove after triage): one-shot per (name,tag) dump of the world
    // 3x3 determinant sign. Negative det = MIRRORED placement — mul((float3x3)world, N) then flips
    // the normal relative to the surface (green<->purple in F12 mode 8), which mis-lights the shape.
    // Called from BOTH the opaque (STATIC) and sorted-alpha (ALPHA) emit so a wall's sign can be
    // compared directly to a tapestry's.
    void diagWorldDet(const char* tag, const char* name, const float* w) {
        static std::unordered_set<std::string> s_seen;
        std::string key = std::string(tag) + "|" + (name ? name : "(null)");
        if (!s_seen.insert(key).second || s_seen.size() > 128) return;
        const float det = w[0] * (w[5]*w[10] - w[6]*w[9])
                        - w[1] * (w[4]*w[10] - w[6]*w[8])
                        + w[2] * (w[4]*w[9]  - w[5]*w[8]);
        LOG::logline(">> [norm-diag] %s %s det=%.3f %s", tag, name ? name : "(null)",
                     det, det < 0.0f ? "MIRRORED" : "normal");
    }

    void emitStaticDraw(SlotInfo& si, const MGE::GeometryCache::CachedGeometry& e,
                        std::uint32_t& count) {
            IPC::DrawItemWire item;
            item.slot = si.slot;
            diagWorldDet("STATIC", e.textureName, e.worldTransformD3D);
            item.texIndex = resolveCachedSlot(e.textureName, si.baseNamePtr, si.baseSlot, si.baseEpoch);
            // Terrain DECAL_1 overlay (second land texture). resolveTextureSlot ships its DDS
            // bytes the same way as the base map. Non-landscape / single-texture draws get 0,
            // which gates the frag's splat off → byte-for-byte unchanged.
            item.overlayTexIndex = (e.isLandscape && e.d3dOverlay && e.overlayTextureName)
                ? resolveCachedSlot(e.overlayTextureName, si.ovNamePtr, si.ovSlot, si.ovEpoch) : 0u;
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
                const std::uint32_t s = si.slot;
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

    // M-Skinning: emit one visible SKINNED part into g_skinnedScratch as
    // [SkinnedDrawWire][palette] (pre-filtered by buildGeometryDrawLists; skinned
    // keys are EXCLUDED from the static list — stride-44 VB + GPU palette pipeline).
    // There is no per-draw world transform — the bone palette (read fresh from the
    // cache entry each frame, that IS the animation) is world-space.
    void emitSkinnedDraw(SlotInfo& si, const MGE::GeometryCache::CachedGeometry& e,
                         std::uint32_t& count) {
            // Only GPU-skinnable parts: a built skinned VB, bones within the palette cap,
            // and a current bone palette of the expected size.
            if (e.skinnedUnsupported || e.numBones == 0) {
                return;
            }
            if (e.bonePalette.size() < (std::size_t)e.numBones * 16) {
                return;   // palette not yet built this frame
            }

            IPC::SkinnedDrawWire item;
            item.slot     = si.slot;
            item.numBones = e.numBones;
            item.mirror   = e.mirrored ? 1u : 0u;
            item.texIndex = resolveCachedSlot(e.textureName, si.baseNamePtr, si.baseSlot, si.baseEpoch);
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

    // Tier 4 multi-map: emit one visible STATIC multi-map part (dark/detail/glow
    // siblings) into g_multiMapScratch as a MultiMapDrawWire (pre-filtered by
    // buildGeometryDrawLists; multi-map keys are EXCLUDED from the static list —
    // wide stride-60 VB + multimap pipeline). The ORDERED stage list is built here
    // on the CLIENT, replicating rendercachedcolor.cpp::buildCacheStages EXACTLY
    // (present maps pushed with their op + UV set, stable-sorted by texCoordSet
    // ascending — MW assigns the D3D stage index = texCoordSet, not the map slot).
    // Stage textures resolve through resolveTextureSlot directly (multi-map items
    // are rare — a handful per frame — not worth SlotInfo fields for 4 maps).
    void emitMultiMapDraw(SlotInfo& si, const MGE::GeometryCache::CachedGeometry& e,
                          std::uint32_t& count) {
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
            item.slot = si.slot;
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

    // AT1 sorted-alpha: emit one blended STATIC part into g_alphaScratch as an AlphaDrawWire.
    // Called AFTER the visible-set loop in back-to-front order (the collect/sort lives in
    // buildGeometryDrawLists), so the wire order IS the draw order — the host never re-sorts.
    // Field packing mirrors emitStaticDraw (texture SlotInfo cache, material, vColSource,
    // camera-relative world) plus the SkyDrawWire blend fields and the material alpha.
    void emitAlphaDraw(SlotInfo& si, const MGE::GeometryCache::CachedGeometry& e,
                       std::uint32_t& count) {
            IPC::AlphaDrawWire item;
            item.slot      = si.slot;
            diagWorldDet("ALPHA", e.textureName, e.worldTransformD3D);
            item.texIndex  = resolveCachedSlot(e.textureName, si.baseNamePtr, si.baseSlot, si.baseEpoch);
            item.srcBlend  = e.srcBlend;
            item.destBlend = e.destBlend;
            item.alphaRef  = e.alphaTest ? e.alphaRef : 0.0f;
            item.matDiffuse[0]  = e.matDiffuse[0];  item.matDiffuse[1]  = e.matDiffuse[1];  item.matDiffuse[2]  = e.matDiffuse[2];
            item.matAlpha       = e.matDiffuse[3];   // MaterialProperty::alpha (the FFE per-draw fade)
            item.matAmbient[0]  = e.matAmbient[0];  item.matAmbient[1]  = e.matAmbient[1];  item.matAmbient[2]  = e.matAmbient[2];
            item.matEmissive[0] = e.matEmissive[0]; item.matEmissive[1] = e.matEmissive[1]; item.matEmissive[2] = e.matEmissive[2];
            item.vColSource = (e.hasVertexColor && e.vColSource != 0) ? e.vColSource : 0u;
            memcpy(item.world, e.worldTransformD3D, 16 * sizeof(float));
            // CAMERA-RELATIVE: shift translation by -eye (see emitStaticDraw).
            item.world[12] -= DistantLand::eyePos.x;
            item.world[13] -= DistantLand::eyePos.y;
            item.world[14] -= DistantLand::eyePos.z;
            // AT3 captured-geometry locators unused for a cached-mesh (slot) item — the host
            // reads them only when slot == kAlphaSlotCaptured. Zero so they never alias garbage.
            item.vertexBase = item.indexBase = item.indexCount = 0;
            // Cull mode from the shape's live NiStencilProperty (DRAW_BOTH → CULL_NONE) + winding
            // from the mirror flag, so the host draws it exactly as MW does (single-sided alpha
            // like the draped altar cloth gets CULL_BACK, hiding its back/interior faces).
            item.cullFlags = (e.twoSided ? IPC::kAlphaCullTwoSided : 0u)
                           | (e.mirrored ? IPC::kAlphaCullMirrored : 0u);
            const std::size_t at = g_alphaScratch.size();
            g_alphaScratch.resize(at + sizeof(item));
            memcpy(g_alphaScratch.data() + at, &item, sizeof(item));
            ++count;
    }

    // AT3: emit one captured blended DIP as a sentinel-slot AlphaDrawWire. Unlike emitAlphaDraw
    // there is no uploaded mesh slot — slot = kAlphaSlotCaptured and vertexBase/indexBase/indexCount
    // locate the geometry in the shared captured VB/IB the kickoff ships. Camera-relative world
    // subtraction uses the CURRENT eyePos (rec.world is absolute), so 1-frame-old records don't swim.
    void emitCapturedAlphaDraw(const CapturedAlphaRec& rec, std::uint32_t& count) {
            IPC::AlphaDrawWire item;
            item.slot      = IPC::kAlphaSlotCaptured;
            item.texIndex  = rec.texIndex;
            item.srcBlend  = rec.srcBlend;
            item.destBlend = rec.destBlend;
            item.alphaRef  = rec.alphaRef;
            item.matDiffuse[0]  = rec.matDiffuse[0];  item.matDiffuse[1]  = rec.matDiffuse[1];  item.matDiffuse[2]  = rec.matDiffuse[2];
            item.matAlpha       = rec.matAlpha;
            item.matAmbient[0]  = rec.matAmbient[0];  item.matAmbient[1]  = rec.matAmbient[1];  item.matAmbient[2]  = rec.matAmbient[2];
            item.matEmissive[0] = rec.matEmissive[0]; item.matEmissive[1] = rec.matEmissive[1]; item.matEmissive[2] = rec.matEmissive[2];
            item.vColSource = rec.vColSource;
            memcpy(item.world, rec.world, 16 * sizeof(float));
            item.world[12] -= DistantLand::eyePos.x;
            item.world[13] -= DistantLand::eyePos.y;
            item.world[14] -= DistantLand::eyePos.z;
            item.vertexBase = rec.vertexBase;
            item.indexBase  = rec.indexBase;
            item.indexCount = rec.indexCount;
            // Captured DIPs are billboarded particles (smoke/flames) — keep CULL_NONE (two-sided)
            // so they render exactly as today; culling a camera-facing quad by winding is fragile.
            item.cullFlags  = IPC::kAlphaCullTwoSided;
            const std::size_t at = g_alphaScratch.size();
            g_alphaScratch.resize(at + sizeof(item));
            memcpy(g_alphaScratch.data() + at, &item, sizeof(item));
            ++count;
    }

    // Cut 2 (dense-city frame attack): ONE pass over the visible set builds all three
    // geometry draw lists. Previously three functions each re-iterated
    // frustumVisibleKeys with their own g_keySlot + cache-map finds — 3x the hash
    // work on the frame's hottest client loop — and re-resolved every texture name
    // through resolveTextureSlot's normalize (heap string) + string-hash find every
    // frame (now SlotInfo-cached, see resolveCachedSlot). Emission order within each
    // list is unchanged (same key iteration order), so the wire bytes are identical.
    //
    // Source = MGE's OWN current-frame frustum-visible set (frustumVisibleKeys /
    // buildFrustumVisibleSet, built in renderStage0) — the SAME source the proven D3D9
    // cache color path (renderCachedOpaque) consumes. This is adaptive and decouples the
    // Forge near draw from the MWSE occlusion plugin entirely:
    //   - plugin ON  → buildFrustumVisibleSet fills it from the engine's drawn set
    //     (s_visibleKeys: de-duped, one LOD per object, occlusion-correct, CURRENT frame).
    //   - plugin OFF → frustum-only fallback (whole-cache frustum walk). Near still renders.
    //
    // Dispatch replicates the three old loops' per-entry filters EXACTLY:
    //   - skinned first (stride-44 VB + GPU palette pipeline), regardless of texture.
    //   - !d3dTexture: drops UNTEXTURED entries — the worldPickObjectRoot collision
    //     proxies that mirror each visual object; drawing them z-fights movers. This is
    //     THE de-dup. (Do NOT also filter isPickRoot — legit movers like dropped
    //     items/projectiles LIVE in the pick root and ARE textured.)
    //   - blendEnable (non-landscape): alpha-blended OBJECTS stay on the engine/alpha
    //     path. Terrain is the exception — the D9 oracle draws ALL isLandscape
    //     regardless of blendEnable (alpha-splat trishapes included).
    //   - non-landscape parts with dark/detail/glow siblings → multi-map pipeline
    //     (stride-60 wide VB); everything else (terrain included) → static pipeline.
    void buildGeometryDrawLists(std::uint32_t& drawCount, std::uint32_t& skinnedCount,
                                std::uint32_t& multiMapCount, std::uint32_t& alphaCount) {
        drawCount = 0;
        skinnedCount = 0;
        multiMapCount = 0;
        alphaCount = 0;
        // Cut 2B fold: on fold frames buildFrustumVisibleSet deferred its ensureLive
        // loop here (foldVisibleKeys = the raw classify set) instead of running it AND
        // having this function re-hash the same keys — ONE pass does live-freshen +
        // emit. Non-fold frames iterate frustumVisibleKeys exactly as before (entries
        // already freshened by buildFrustumVisibleSet or the full walk).
        const auto* foldKeys = DistantLand::foldVisibleKeys();
        const auto& keys = DistantLand::frustumVisibleKeys();
        const auto& cacheMap = MGE::GeometryCache::cache();

        g_drawScratch.clear();
        g_drawScratch.reserve((foldKeys ? foldKeys->size() : keys.size()) * sizeof(IPC::DrawItemWire));
        g_skinnedScratch.clear();
        g_multiMapScratch.clear();
        g_alphaScratch.clear();

        const bool wantStatic  = (bool)g_drawVec;
        const bool wantSkinned = (bool)g_skinnedVec;
        const bool wantMM      = (bool)g_multiMapVec;
        // AT1 sorted-alpha: gated by the bring-up ini flag on top of the vec (capture + emit +
        // host draw all ride this one gate; off = the blended set stays engine-drawn as before).
        const bool wantAlpha   = (bool)g_alphaVec && Configuration.ForgeAlphaPass;
        if (!wantStatic && !wantSkinned && !wantMM && !wantAlpha) {
            // AT3 defensive clear: drop any captured records/geometry so an idle frame (menu/
            // loading) can't leave last frame's captures to accumulate or ship stale.
            g_capturedEmitted = 0;
            g_capRecs.clear();
            g_capVertScratch.clear();
            g_capIdxScratch.clear();
            g_capTexMemo.clear();
            g_alphaDedup.clear();
            return;
        }

        // AT1 collect: blended shapes are gathered (not emitted) during the loop, then sorted
        // back-to-front and packed AFTER it — MW's sorter criterion is bound-center view depth.
        // Pointers into the two unordered_maps are element-stable across ensureLive inserts.
        // AT1 cached-mesh (si+e) OR AT3 captured-geometry (cap) candidate — ONE sort covers both
        // so cross-set blend order is correct. Exactly one of {e, cap} is non-null per entry.
        struct AlphaCand {
            float depth;
            SlotInfo* si;                                       // cached: valid; captured: nullptr
            const MGE::GeometryCache::CachedGeometry* e;        // cached: valid; captured: nullptr
            const CapturedAlphaRec* cap;                        // captured: valid; cached: nullptr
        };
        static std::vector<AlphaCand> alphaCands;   // single-threaded; reused frame-to-frame
        alphaCands.clear();
        // AT3 old-msoc double-draw guard set — rebuilt THIS frame from the cached blends the host
        // draws, then read by captureAlphaDraw during this frame's scene>=1 (see g_alphaDedup).
        g_alphaDedup.clear();
        // World-space view forward (mwView's 3rd column) for the depth key; eye-relative so the
        // key is invariant to the camera-relative world shift emitAlphaDraw applies later.
        const float fwdX = DistantLand::mwView._13;
        const float fwdY = DistantLand::mwView._23;
        const float fwdZ = DistantLand::mwView._33;

        // Per-entry dispatch, identical on both paths (see the filter contract above).
        // slot is mutable — the emit helpers update its cached texture SlotInfo.
        auto dispatch = [&](auto& slot, const auto& e) {
            if (e.isSky) return;   // sky rides the Forge alpha-blend sky pass (buildSkyDrawList)
            if (e.isSkinned) {
                if (wantSkinned) emitSkinnedDraw(slot, e, skinnedCount);
                return;
            }
            if (!e.d3dTexture) return;
            if (!e.isLandscape && e.blendEnable) {
                // AT1: non-landscape blended shapes ride the host alpha pass (v1 = static
                // single-map tri shapes only; multi-map blends keep today's behavior — skipped —
                // and skinned blends were already routed to the skinned path above).
                if (wantAlpha && !(e.d3dDark || e.d3dDetail || e.d3dGlow)) {
                    const float* w = e.worldTransformD3D;
                    const float cx = e.boundsCenter[0], cy = e.boundsCenter[1], cz = e.boundsCenter[2];
                    const float wx = cx * w[0] + cy * w[4] + cz * w[8]  + w[12] - DistantLand::eyePos.x;
                    const float wy = cx * w[1] + cy * w[5] + cz * w[9]  + w[13] - DistantLand::eyePos.y;
                    const float wz = cx * w[2] + cy * w[6] + cz * w[10] + w[14] - DistantLand::eyePos.z;
                    alphaCands.push_back({ wx * fwdX + wy * fwdY + wz * fwdZ, &slot, &e, nullptr });
                    // AT3: record (GPU texture, vertexCount) so captureAlphaDraw skips this exact
                    // blend if MW's own DIP for it still reaches the reject gate (old-msoc dll:
                    // opaque-only skip → cached single-map blends both DIP AND ride the host).
                    if (e.d3dTexture) {
                        g_alphaDedup.insert(((std::uint64_t)(std::uintptr_t)e.d3dTexture << 32)
                                            | (std::uint64_t)e.vertexCount);
                    }
                }
                return;
            }
            if (!e.isLandscape && (e.d3dDark || e.d3dDetail || e.d3dGlow)) {
                if (wantMM) emitMultiMapDraw(slot, e, multiMapCount);
            } else if (wantStatic) {
                emitStaticDraw(slot, e, drawCount);
            }
        };

        if (foldKeys) {
            for (std::uint32_t key : *foldKeys) {
                // Freshen (or lazily capture) straight off the live NiTriShape — a
                // classify key is engine-drawn THIS frame, so the pointer is valid by
                // construction. MUST run before the g_keySlot probe: a first-sight key
                // has no slot until ensureLive's capture registers it (same frame).
                const auto* e = MGE::GeometryCache::ensureLive(key);
                if (!e) continue;                     // no model data / capture failed
                // The unusable-skinned filter buildFrustumVisibleSet applies on
                // non-fold frames (never drawn; trips the bound helper).
                if (e->isSkinned && (e->skinnedUnsupported || e->numBones == 0)) continue;
                auto ks = g_keySlot.find(key);
                if (ks == g_keySlot.end()) {
                    continue;   // not an uploaded part (or not yet shipped)
                }
                dispatch(ks->second, *e);
            }
        } else {
            for (std::uint32_t key : keys) {
                auto ks = g_keySlot.find(key);
                if (ks == g_keySlot.end()) {
                    continue;   // not an uploaded part (or not yet shipped)
                }
                auto ce = cacheMap.find(key);
                if (ce == cacheMap.end()) {
                    continue;   // evicted since the visible-set build
                }
                dispatch(ks->second, ce->second);
            }
        }

        // AT3: merge the captured blended DIPs (particles/smoke/flames + multimap/decal/untextured
        // blends the host cache pass doesn't own) captured LAST frame (g_capRecs) into the SAME
        // candidate list so ONE sort orders the whole blended set back-to-front. Depth uses the
        // CURRENT eye + forward against the record's absolute world-space centroid (no swim from the
        // 1-frame latency). The records are consumed (cleared) after emit; the geometry bytes stay
        // in g_capVertScratch/g_capIdxScratch for the kickoff to ship.
        if (wantAlpha) {
            for (const auto& rec : g_capRecs) {
                const float dx = rec.centroid[0] - DistantLand::eyePos.x;
                const float dy = rec.centroid[1] - DistantLand::eyePos.y;
                const float dz = rec.centroid[2] - DistantLand::eyePos.z;
                alphaCands.push_back({ dx * fwdX + dy * fwdY + dz * fwdZ, nullptr, nullptr, &rec });
            }
        }

        // AT1/AT3: back-to-front (descending view depth — farthest drawn first, MW's sort) then pack.
        if (!alphaCands.empty()) {
            std::sort(alphaCands.begin(), alphaCands.end(),
                      [](const AlphaCand& a, const AlphaCand& b) { return a.depth > b.depth; });
            std::size_t first = 0;
            if (alphaCands.size() > IPC::kMaxAlphaDraws) {
                // Over cap: drop the FARTHEST (sorted front) — the near draws matter most.
                first = alphaCands.size() - IPC::kMaxAlphaDraws;
                static bool logged = false;
                if (!logged) {
                    LOG::logline("!! [alpha] %zu blended shapes this frame > cap %u — farthest dropped",
                                 alphaCands.size(), IPC::kMaxAlphaDraws);
                    logged = true;
                }
            }
            g_alphaScratch.reserve((alphaCands.size() - first) * sizeof(IPC::AlphaDrawWire));
            for (std::size_t i = first; i < alphaCands.size(); ++i) {
                if (alphaCands[i].cap) {
                    emitCapturedAlphaDraw(*alphaCands[i].cap, alphaCount);
                } else {
                    emitAlphaDraw(*alphaCands[i].si, *alphaCands[i].e, alphaCount);
                }
            }
            // Bring-up diagnostic: name the captured set once (what the alpha list actually IS —
            // if the in-game "sorted" the eye notices isn't in here, it's an AT3 leftover, not a
            // draw bug). Remove after verify.
            static bool s_alphaNamed = false;
            if (!s_alphaNamed) {
                s_alphaNamed = true;
                LOG::logline(">> [alpha-cap] %zu blended shapes this frame (%zu captured AT3):",
                             alphaCands.size(), g_capRecs.size());
                const std::size_t nDump = (alphaCands.size() < 16u) ? alphaCands.size() : 16u;
                for (std::size_t i = 0; i < nDump; ++i) {
                    const auto& c = alphaCands[i];
                    if (c.cap) {
                        LOG::logline(">> [alpha-cap] #%zu depth=%.0f CAPTURED tex=%u vc=%u tri=%u blend=%u/%u matA=%.2f vcs=%u",
                                     i, c.depth, c.cap->texIndex, c.cap->vertexCount, c.cap->indexCount / 3,
                                     c.cap->srcBlend, c.cap->destBlend, c.cap->matAlpha, c.cap->vColSource);
                    } else {
                        const auto& e = *c.e;
                        LOG::logline(">> [alpha-cap] #%zu depth=%.0f tex=%s vc=%u tri=%u blend=%u/%u matA=%.2f",
                                     i, c.depth, e.textureName ? e.textureName : "(null)",
                                     e.vertexCount, e.triangleCount, e.srcBlend, e.destBlend, e.matDiffuse[3]);
                    }
                }
            }
        }

        // AT3: the captured RECORDS are now consumed (emitted as wire items). Clear them + the
        // per-frame texture memo so this frame's scene>=1 captures start fresh; the geometry BYTES
        // (g_capVertScratch/g_capIdxScratch) stay for the kickoff assign to ship, then are cleared
        // there. On !wantAlpha frames the records were never merged — drop them so they can't
        // accumulate (and drop the bytes too, since nothing will ship them).
        g_capturedEmitted = wantAlpha ? (std::uint32_t)g_capRecs.size() : 0u;
        g_capRecs.clear();
        g_capTexMemo.clear();
        if (!wantAlpha) {
            g_capVertScratch.clear();
            g_capIdxScratch.clear();
        }
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

        // Frustum cull (host GPU shrink): only the NEAR scene consumes point lights
        // (opaque.frag/multimap.frag loop them; statics/distantland don't), and a light's
        // contribution is EXACTLY zero beyond 2·radius (the shader's smoothstep cutoff).
        // So a light whose 2r sphere misses the game frustum can't touch any lit pixel —
        // drop it before it costs every pixel a loop iteration. This also makes the
        // kMaxPointLights cap meaningful: cull first, THEN cap, so a dense cell keeps the
        // lights that can actually show (was: arbitrary first-128 of the snapshot order).
        D3DXMATRIX lightVP;
        D3DXMatrixMultiply(&lightVP, &DistantLand::mwView, &DistantLand::mwProj);
        const ViewFrustum lightFrustum(&lightVP);

        g_lightScratch.clear();
        std::uint32_t count = 0;
        std::uint32_t culled = 0;
        for (const auto& pl : lights) {
            BoundingSphere ls;
            ls.center = D3DXVECTOR3(pl.worldPos[0], pl.worldPos[1], pl.worldPos[2]);
            ls.radius = 2.0f * pl.radius;
            if (lightFrustum.ContainsSphere(ls) == ViewFrustum::OUTSIDE) {
                ++culled;
                continue;
            }
            if (count >= IPC::kMaxPointLights) {
                static bool logged = false;
                if (!logged) {
                    LOG::logline("!! [light] %zu in-frustum point lights this frame > cap %u — extra dropped (Tier 3b clustering lifts this)",
                                 lights.size() - culled, IPC::kMaxPointLights);
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

    // SK1 sky takeover: gather this frame's sky parts into g_skyScratch as SkyDrawWire[]. Source is
    // the WHOLE geometry cache (sky shapes are NOT in DistantLand::visibleCacheKeys — that's the MSOC
    // world-object drawn set), filtered to isSky entries that have a host slot. SK1 emits ONLY the
    // dome: the untextured vertex-colour shape (isSky && !d3dTexture); SK2 adds the textured shapes
    // (sun/moons/clouds/stars). Camera-relative shift matches the host's translation-free viewProj
    // (the sky is camera-attached). Only runs when the Forge sky pass is toggled on (F7), so the
    // full-cache scan is paid only during the A/B. Returns the packed item count.
    std::uint32_t buildSkyDrawList() {
        if (!g_skyVec || !g_skyEnabled) {
            return 0;
        }
        const auto& cacheMap = MGE::GeometryCache::cache();

        g_skyScratch.clear();
        // SK2: gather every isSky entry that has a host slot (dome + textured sun/moons/stars),
        // then sort by skyOrder (MW's back-to-front subtree order) so the alpha-blended shapes
        // layer correctly — the cache is an unordered_map, so we can't rely on iteration order.
        struct SkyCand { std::uint16_t order; const MGE::GeometryCache::CachedGeometry* e; std::uint32_t slot; };
        static std::vector<SkyCand> cands;   // single-threaded; reused frame-to-frame
        cands.clear();
        // Eviction is a periodic sweep now, so this whole-cache scan must skip stale
        // entries itself: a sky shape the walk stopped visiting (moon set, weather
        // change → appCulled) would otherwise keep drawing until the next sweep.
        const auto cacheFrame = MGE::GeometryCache::currentFrame();
        for (const auto& kv : cacheMap) {
            const auto& e = kv.second;
            if (!e.isSky) continue;
            if (e.lastFrame != cacheFrame) continue;   // stale (awaiting eviction sweep)
            auto ks = g_keySlot.find(kv.first);
            if (ks == g_keySlot.end()) {
                continue;   // not yet uploaded to the host
            }
            cands.push_back({ e.skyOrder, &e, ks->second.slot });
        }
        std::sort(cands.begin(), cands.end(),
                  [](const SkyCand& a, const SkyCand& b) { return a.order < b.order; });

        IPC::SkyDrawWire item;
        std::uint32_t count = 0;
        for (const auto& c : cands) {
            const auto& e = *c.e;
            item.slot      = c.slot;
            // Dome stays 0 (vertex-colour only → host default white); textured shapes resolve
            // their DDS to a bindless slot via the existing residency path.
            item.texIndex  = e.d3dTexture ? resolveTextureSlot(e.textureName) : 0u;
            item.srcBlend  = e.srcBlend;
            item.destBlend = e.destBlend;
            item.alphaRef  = e.alphaTest ? e.alphaRef : 0.0f;
            // SK2 FFP modulation: material diffuse rgb + per-element alpha fade + vcol routing,
            // exactly as buildDrawList ships for opaques.
            item.matColor[0] = e.matDiffuse[0];
            item.matColor[1] = e.matDiffuse[1];
            item.matColor[2] = e.matDiffuse[2];
            item.matAlpha    = e.matDiffuse[3];
            {
                const std::uint32_t baseVCol = (e.hasVertexColor && e.vColSource != 0) ? e.vColSource : 0u;
                // C3: the untextured atmosphere dome (texIndex 0, vertex-coloured) is now host-coloured
                // with a vertical gradient. Tag it with the dedicated vColSource 3 ("host gradient dome")
                // so sky.frag computes fogColNear->skyZenith from the vertex direction instead of using
                // the (no-longer-re-uploaded) baked per-vertex gradient. SK2 textured shapes keep 0/1/2.
                item.vColSource = (item.texIndex == 0u && baseVCol != 0u) ? 3u : baseVCol;
            }
            memcpy(item.world, e.worldTransformD3D, 16 * sizeof(float));
            // SK2 billboard fix (SUN ONLY): the sun disc hangs under a NiBillboardNode that MW
            // re-faces to the camera each frame via rotateToCamera — but that runs AFTER our
            // onFrameReady scene walk, so e.worldTransformD3D still holds the billboard's BASE
            // celestial-sphere (tangent) orientation. Replayed verbatim the quad is a flat plate
            // tangent to the sky sphere: round looking straight at it (zenith), squashed at grazing
            // angles (near the horizon). The two-part MOONS are also 4-vert/2-tri textured quads but
            // arrive ALREADY camera-faced in their captured transform (re-billboarding them broke
            // their facing in-game), so leave those alone — gate on the sun's base-texture name
            // ("tx_sun_05" — the moons are tx_masser/tx_secunda/tx_mooncircle, no "sun"). Rebuild a
            // camera-facing basis: preserve position (translation) + per-axis size; orient model
            // +X -> camera right, +Y -> camera up (spherical / full-facing → always round = vanilla).
            bool isSunDisc = false;
            if (e.d3dTexture && e.textureName && e.vertexCount == 4 && e.triangleCount == 2) {
                // case-insensitive substring "sun" (|32 lowercases ASCII letters; loop guard keeps
                // the p[1]/p[2] look-ahead inside the null-terminated string).
                for (const char* p = e.textureName; p[0] && p[1] && p[2]; ++p) {
                    if ((p[0] | 32) == 's' && (p[1] | 32) == 'u' && (p[2] | 32) == 'n') { isSunDisc = true; break; }
                }
            }
            if (isSunDisc) {
                const D3DXMATRIX& V = DistantLand::mwView;  // row-vector view: columns = world camera axes
                const float* m = item.world;
                const float sx = sqrtf(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);     // model +X length (width)
                const float sy = sqrtf(m[4]*m[4] + m[5]*m[5] + m[6]*m[6]);     // model +Y length (height)
                const float sz = sqrtf(m[8]*m[8] + m[9]*m[9] + m[10]*m[10]);   // model +Z length (normal)
                const float Rx = V._11, Ry = V._21, Rz = V._31;   // camera right  (world)
                const float Ux = V._12, Uy = V._22, Uz = V._32;   // camera up     (world)
                const float Fx = V._13, Fy = V._23, Fz = V._33;   // camera forward(world, into scene)
                item.world[0] =  sx*Rx; item.world[1] =  sx*Ry; item.world[2]  =  sx*Rz;   // +X -> right
                item.world[4] =  sy*Ux; item.world[5] =  sy*Uy; item.world[6]  =  sy*Uz;   // +Y -> up
                item.world[8] = -sz*Fx; item.world[9] = -sz*Fy; item.world[10] = -sz*Fz;   // +Z -> toward camera
            }
            // WT2: tell the Forge reflection pass which shape is the sun, so it can re-face the disc
            // for the mirrored view (the world above faces the MAIN camera → squashed once mirrored).
            item.isSunDisc = isSunDisc ? 1u : 0u;
            // CAMERA-RELATIVE: the sky is camera-attached; shift by -eye to match the host's
            // translation-free viewProj (see buildDrawList) and keep vertex math near the origin.
            item.world[12] -= DistantLand::eyePos.x;
            item.world[13] -= DistantLand::eyePos.y;
            item.world[14] -= DistantLand::eyePos.z;
            const std::size_t at = g_skyScratch.size();
            g_skyScratch.resize(at + sizeof(item));
            memcpy(g_skyScratch.data() + at, &item, sizeof(item));
            if (++count >= IPC::kMaxSkyDraws) {
                break;
            }
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

    void onStage0CompositeKickoff(IDirect3DDevice9* device) {
        // Whole-frame no-op guarantee: rpcPending stays false on every early-out below, so
        // the paired Finish returns immediately.
        g_kick = KickState{};
        if (!device || !g_initOk) {
            return;
        }
        // Client-side Forge driving phase, kickoff half: build draw lists → geom flush → tex
        // flush → start the host RenderFrame WITHOUT waiting. The host renders frame N while
        // MW's own frame-N work continues; onStage0CompositeFinish drains the completion and
        // composites. Zoned so the Tracy frame has NO unaccounted gap here.
        MGE_ZoneScopedN("Forge composite kickoff");

        const double tStart = nowMs();
        const double dtPresent = (g_lastPresentMs > 0.0) ? (tStart - g_lastPresentMs) : 0.0;
        g_lastPresentMs = tStart;

        // (Edge-triggered dev-key polls — F11/F12/F9/F7/F8 — moved to
        // onStage0CompositeFinish so every per-frame ownership gate, including the
        // frameSetupEarly early-kickoff latch that runs BEFORE this function, sees
        // one consistent value per frame. See the comment there.)
        if (!g_enabled) {
            // Still ship any geometry the cache captured this frame (walk-driven while
            // the composite is off), so the host's mesh store is ready when F11 turns
            // it on.
            MGE_ZoneScopedN("Forge geom flush");
            flushGeometry();
            // AT3 defensive clear: nothing will consume/ship captured alpha while off — drop it so
            // a stale record can't leak in on the next F11-on frame (captures require g_enabled).
            g_capRecs.clear();
            g_capVertScratch.clear();
            g_capIdxScratch.clear();
            g_capTexMemo.clear();
            return;
        }

        const unsigned frame = g_frame++;

        // Drive the host renderer into the shared RT. Async — the host fence-waits before
        // signalling completion, so at renderSceneFinish the draw is GPU-complete and the
        // resource quiescent before our copy. M1c: build this frame's visible draw list
        // (camera + slots + world transforms) and render the cached opaque SCENE.
        bool ok = false;
        // Re-walk the geometry cache to build the host's draw lists (the MGE→Forge feeding cost —
        // Phase 2 makes this GPU-resident so it goes to 0).
        std::uint32_t drawCount, skinnedCount, multiMapCount, lightCount, skyCount, alphaCount;
        {
            MGE_ZoneScopedN("Forge build draw lists");
            buildGeometryDrawLists(drawCount, skinnedCount, multiMapCount, alphaCount);
            lightCount = buildLightList();
            skyCount = buildSkyDrawList();
        }
        const double tBuild = nowMs();

        // Ship this frame's captured geometry AFTER the build (Cut 2B): on fold frames
        // first-sight parts are captured DURING buildGeometryDrawLists (ensureLive lazy
        // capture), and the host must have the mesh bytes before the RenderFrame RPC
        // draws them — same reason textures flush after the build. Cheap on steady
        // frames (geom+=0KB), spikes on cell loads; goes to ~0 with Phase 2 (resident
        // GPU geometry — nothing to ship).
        const std::uint32_t geomParts = g_pendingParts;            // snapshot (flush clears it)
        const std::size_t   geomBytes = g_pendingBlob.size();
        {
            MGE_ZoneScopedN("Forge geom flush");
            flushGeometry();
        }
        const double tGeomFlush = nowMs();

        // Ship any textures newly referenced this frame BEFORE the scene draw that uses them
        // (buildDrawList queued their DDS via resolveTextureSlot). Lazy: only first-seen textures.
        const std::uint32_t texCount = g_texPendingCount;          // snapshot (flush clears it)
        const std::size_t   texBytes = g_texPendingBlob.size();
        {
            MGE_ZoneScopedN("Forge tex flush");
            flushTextures();
        }
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

        IPC::VecId   skyId = IPC::InvalidVector;
        std::uint32_t skyBytes = 0;
        if (g_skyVec && skyCount > 0
            && g_skyVec->assign_bytes(g_skyScratch.data(), (std::uint32_t)g_skyScratch.size())) {
            skyId    = g_skyVec->id();
            skyBytes = (std::uint32_t)g_skyScratch.size();
        }

        IPC::VecId   alphaId = IPC::InvalidVector;
        std::uint32_t alphaBytes = 0;
        if (g_alphaVec && alphaCount > 0
            && g_alphaVec->assign_bytes(g_alphaScratch.data(), (std::uint32_t)g_alphaScratch.size())) {
            alphaId    = g_alphaVec->id();
            alphaBytes = (std::uint32_t)g_alphaScratch.size();
        }

        // AT3 captured-alpha: ship the geometry buildGeometryDrawLists left in the scratch
        // (referenced by the sentinel-slot AlphaDrawWire items just packed). One 1-chunk vec
        // holds [verts (GeomVertexWire)][indices (uint16)] contiguously — indices begin at
        // capturedVertBytes. Then clear the scratch so this frame's fresh captures start at base 0
        // (the wire bases match the shipped layout because we ship the whole scratch verbatim).
        IPC::VecId   capturedId = IPC::InvalidVector;
        std::uint32_t capVertBytes = 0, capIdxBytes = 0;
        if (g_capturedVec && !g_capVertScratch.empty()) {
            capVertBytes = (std::uint32_t)(g_capVertScratch.size() * sizeof(IPC::GeomVertexWire));
            capIdxBytes  = (std::uint32_t)(g_capIdxScratch.size() * sizeof(std::uint16_t));
            static std::vector<std::uint8_t> capBlob;   // reused staging blob (contiguous verts+indices)
            capBlob.resize((std::size_t)capVertBytes + capIdxBytes);
            memcpy(capBlob.data(), g_capVertScratch.data(), capVertBytes);
            if (capIdxBytes) { memcpy(capBlob.data() + capVertBytes, g_capIdxScratch.data(), capIdxBytes); }
            if (g_capturedVec->assign_bytes(capBlob.data(), (std::uint32_t)capBlob.size())) {
                capturedId = g_capturedVec->id();
            } else {
                capVertBytes = capIdxBytes = 0;   // over one window (shouldn't happen at these caps)
            }
        }
        g_capVertScratch.clear();
        g_capIdxScratch.clear();
        const double tAssign = nowMs();

        // The MGE→Forge FEEDING cost (the bulk of the "unaccounted" client gap): every frame MGE
        // walks its scene-graph cache to rebuild the host's draw lists + uploads geom/tex over IPC.
        // This is exactly the dependence to drive toward 0 — a resident flat GPU scene (Phase 2) lets
        // the host keep geometry across frames and the GPU cull build draw lists, removing both.
        // Cut 2B bucket order: build runs FIRST now (geom flush moved after it), and on
        // fold frames the build bucket includes the ensureLive cost that used to bill
        // to buildFrustumVisibleSet (cross-run comparison caveat).
        MGE_TracyPlot("Forge prep: buildLists ms", tBuild - tStart);
        MGE_TracyPlot("Forge prep: geomFlush ms", tGeomFlush - tBuild);
        MGE_TracyPlot("Forge prep: texFlush ms", tTexFlush - tGeomFlush);
        MGE_TracyPlot("Forge prep: assign ms", tAssign - tTexFlush);

        // Only drive + composite the host when there's actual scene data this frame. With no
        // draw list (loading doors, menus, empty cells) we must NOT fall back to the bring-up
        // triangle and composite it — that flashes the debug triangle over MW's loading/menu
        // frame. Skip the seam entirely and let MW present its own (fixed-function) frame.
        if (!haveDraw && skinnedId == IPC::InvalidVector && multiMapId == IPC::InvalidVector
            && skyId == IPC::InvalidVector && alphaId == IPC::InvalidVector) {
            return;
        }

        // CAMERA-RELATIVE viewProj: zero the view matrix's translation row so the camera sits at
        // the origin (the world translations above are pre-shifted by -eye, so this cancels exactly:
        // eyePos == inverse(mwView)·origin, hence mwView's translation row == -eye·R). This keeps the
        // whole vertex pipeline near the origin and eliminates the float32 large-world stretching.
        D3DXMATRIX viewRel = DistantLand::mwView;
        viewRel._41 = viewRel._42 = viewRel._43 = 0.0f;

        // UNIFIED FAR PROJECTION (Phase 1a/1b): push the far plane out to the distant-land draw
        // distance so near opaque + host-owned DL share ONE reverse-Z depth mapping (statics get
        // occluded behind distant terrain in the shared depth). Near geometry is unaffected — only
        // the far plane moves; reverse-Z keeps near precision. mwProj is the NEAR projection; recover
        // its near plane (zn = -_43/_33 for a standard D3D perspective) and re-edit only z.
        D3DXMATRIX farProj = DistantLand::mwProj;
        const float zn = (farProj._33 != 0.0f) ? (-farProj._43 / farProj._33) : 4.0f;
        DistantLand::editProjectionZ(&farProj, zn, Configuration.DL.DrawDist * DistantLand::kCellSize);
        D3DXMATRIX viewProj;
        D3DXMatrixMultiply(&viewProj, &viewRel, &farProj);

        const bool isExterior = MWBridge::get()->IsExterior();

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
        // Host-computed sky dome (C2): the current interpolated ZENITH sky colour MW already blended
        // this frame from the Weather_*_Sky_*_Color ini keys (getCurrentWeatherSkyCol; MW does the
        // time-of-day blend internally). The host colours the dome with a vertical gradient
        // fogColNear(horizon) -> skyZenith(zenith), so we ship ONLY this colour — the dome geometry
        // itself no longer re-uploads (see scenegraph_geometry_cache SK1). Null-guarded to black.
        // Guard on CellHasWeather like DistantLand::update (the weather-struct pointer is only valid
        // then); fall back to the horizon (nearFogCol) so a no-weather cell yields a flat dome. The
        // dome only draws in weather exteriors anyway, so skyZenith is otherwise unconsumed.
        MWBridge* mwb = MWBridge::get();
        const RGBVECTOR* skyColPtr = mwb->CellHasWeather() ? mwb->getCurrentWeatherSkyCol() : nullptr;
        const float skyZenithR = skyColPtr ? skyColPtr->r : DistantLand::nearFogCol.r;
        const float skyZenithG = skyColPtr ? skyColPtr->g : DistantLand::nearFogCol.g;
        const float skyZenithB = skyColPtr ? skyColPtr->b : DistantLand::nearFogCol.b;
        const float lighting[32] = {
            DistantLand::sunVec.x,     DistantLand::sunVec.y,     DistantLand::sunVec.z,     0.0f,
            sunColEff.r,               sunColEff.g,               sunColEff.b,               0.0f,
            ambColEff.r,               ambColEff.g,               ambColEff.b,               0.0f,
            DistantLand::nearFogCol.r, DistantLand::nearFogCol.g, DistantLand::nearFogCol.b, 0.0f,
            DistantLand::fogNearStart, DistantLand::fogNearEnd,   0.0f,                      0.0f,
            // CAMERA-RELATIVE: WorldPos reaches the shader already relative to the eye, so the
            // eyePos used for the per-vertex fog distance |worldPos - eyePos| is the origin (0).
            0.0f,                      0.0f,                      0.0f,                      0.0f,
            // Phase 1a/1b: the REAL absolute camera eye (host shifts resident DL by -realEye to
            // match the camera-relative near scene) + isExterior gate (1 = feed host-owned DL).
            DistantLand::eyePos.x,     DistantLand::eyePos.y,     DistantLand::eyePos.z,     isExterior ? 1.0f : 0.0f,
            // C2 skyZenith (float4 28..31): zenith sky colour for the host dome gradient. Host reads
            // it into FrameData.skyZenith; only sky.frag (dome branch) consumes it.
            skyZenithR,                skyZenithG,                skyZenithB,                0.0f,
        };

        // Dev overlay input (Stage 2): poll the mouse in MW client-space pixels (1:1 with the host
        // render target) + L/R/M buttons, and forward with the F9 visibility flag. Only meaningful
        // when the panel is up; otherwise uiVisible=0 leaves the host overlay hidden/inert.
        IPC::DevInput devInput;
        devInput.uiVisible = g_devUiVisible ? 1u : 0u;
        // F8 one-shot host compute-shader hot-reload: the edge poll happens at composite
        // finish (see there); consume the latched request into THIS frame's DevInput.
        if (g_reloadShadersPending) {
            devInput.reloadShaders = 1u;
            g_reloadShadersPending = false;
        }
        if (g_devUiVisible) {
            if (!g_devHwnd) {
                D3DDEVICE_CREATION_PARAMETERS cp = {};
                if (SUCCEEDED(device->GetCreationParameters(&cp))) {
                    g_devHwnd = cp.hFocusWindow;
                }
            }
            POINT p;
            if (GetCursorPos(&p) && g_devHwnd && ScreenToClient(g_devHwnd, &p)) {
                devInput.x = p.x;
                devInput.y = p.y;
            }
            if (GetAsyncKeyState(VK_LBUTTON) & 0x8000) devInput.buttons |= 0x1u;
            if (GetAsyncKeyState(VK_RBUTTON) & 0x8000) devInput.buttons |= 0x2u;
            if (GetAsyncKeyState(VK_MBUTTON) & 0x8000) devInput.buttons |= 0x4u;
        }

        // WT1 Forge water: per-frame surface params (no geometry — the host generates the
        // geo-clipmap mesh). waterOn (F7) gates the host water pass. depthBaseColor mirrors XE Mod
        // Water.fx:24 using available DistantLand colours as the skyCol/fogColFar proxies; windFactor
        // is a calm constant (windVec isn't exposed to MGE — tune later). camFwd = mwView's 3rd column
        // (world-space view forward) for the slant→perpendicular shoreline depth correction.
        float waterParams[12] = {};
        // CellHasWater: waterless interiors (most of them) must not draw the host water —
        // MGE's own water path always gated on this and the Forge crossing lost it, so the
        // host drew the geo-clipmap at a stale WaterLevel() in dry cells. Per-frame cell
        // state gates only THIS wire flag (host skips water + its reflection pass);
        // wantsWaterCapture() stays the mode gate (forgeOwnsDepth / fold / WT3 suppression
        // must not flip per cell). Exteriors always have water (CellHasWater true there).
        const std::uint32_t waterOn =
            (wantsWaterCapture() && MWBridge::get()->CellHasWater()) ? 1u : 0u;
        if (waterOn) {
            MWBridge* mw = MWBridge::get();
            const float sunlightFactor = 1.0f - (1.0f - DistantLand::sunVis) * (1.0f - DistantLand::sunVis);
            const RGBVECTOR sunAdj = sunlightFactor * DistantLand::sunCol;
            const RGBVECTOR skyC = DistantLand::horizonCol;     // skyCol proxy
            const RGBVECTOR fogF = DistantLand::nearFogCol;     // fogColFar proxy
            waterParams[0]  = mw->WaterLevel();
            waterParams[1]  = 0.013f;                            // windFactor (calm; tune later)
            waterParams[2]  = 24.0f;                             // shoreDepthBias (XE Mod Water.fx:27)
            waterParams[3]  = sunAdj.r * 0.03f + (2.0f * skyC.r + fogF.r) * 0.075f;
            waterParams[4]  = sunAdj.g * 0.04f + (2.0f * skyC.g + fogF.g) * 0.080f;
            waterParams[5]  = sunAdj.b * 0.05f + (2.0f * skyC.b + fogF.b) * 0.085f;
            waterParams[6]  = DistantLand::nearViewRange;
            waterParams[7]  = mw->IsUnderwater(DistantLand::eyePos.z) ? 1.0f : 0.0f;
            waterParams[8]  = DistantLand::mwView._13;
            waterParams[9]  = DistantLand::mwView._23;
            waterParams[10] = DistantLand::mwView._33;
            waterParams[11] = 0.0f;
        }

        // Async kickoff: copy the frame params into shared memory and start the host, then
        // RETURN — the host renders while MW's frame-N work continues. All the pointer args
        // (viewProj/lighting/devInput/waterParams) are memcpy'd into the IPC block before
        // renderSceneKickoff returns, so these stack locals can die here.
        {
            MGE_ZoneScopedN("Forge renderSceneKickoff");
            ok = g_client->renderSceneKickoff(frame, (const float*)&viewProj, lighting,
                     haveDraw ? g_drawVec->id() : IPC::InvalidVector,
                     haveDraw ? drawCount : 0,
                     haveDraw ? (std::uint32_t)g_drawScratch.size() : 0,
                     skinnedId, skinnedCount, skinnedBytes,
                     multiMapId, (multiMapId != IPC::InvalidVector) ? multiMapCount : 0, multiMapBytes,
                     lightId, (lightId != IPC::InvalidVector) ? lightCount : 0, lightBytes,
                     skyId, (skyId != IPC::InvalidVector) ? skyCount : 0, skyBytes,
                     alphaId, (alphaId != IPC::InvalidVector) ? alphaCount : 0, alphaBytes,
                     capturedId, capVertBytes, capIdxBytes,
                     (std::uint32_t)g_debugMode, &devInput, waterParams, waterOn);
        }
        if (!ok) {
            return;     // rpcPending stays false → Finish no-ops
        }

        // Hand everything the finish half needs across the overlap window.
        g_kick.rpcPending    = true;
        g_kick.early         = DistantLand::earlyForgeKickoff;  // BeginScene(0) site vs late (EndScene)
        g_kick.frame         = frame;
        g_kick.drawCount     = drawCount;
        g_kick.skinnedCount  = skinnedCount;
        g_kick.multiMapCount = multiMapCount;
        g_kick.lightCount    = lightCount;
        g_kick.skyCount      = skyCount;
        g_kick.alphaCount    = alphaCount;
        g_kick.capturedCount = g_capturedEmitted;
        g_kick.geomParts     = geomParts;
        g_kick.texCount      = texCount;
        g_kick.geomBytes     = geomBytes;
        g_kick.texBytes      = texBytes;
        g_kick.dtPresent     = dtPresent;
        g_kick.tStart        = tStart;
        g_kick.tGeomFlush    = tGeomFlush;
        g_kick.tBuild        = tBuild;
        g_kick.tTexFlush     = tTexFlush;
        g_kick.tAssign       = tAssign;
        g_kick.tKick         = nowMs();
    }

    void onStage0CompositeFinish(IDirect3DDevice9* device) {
        // Poll the seam's edge-triggered dev keys HERE, at composite finish — after every
        // consumer of the ownership gates has run this frame. Phase 2 latches the early-
        // kickoff decision (DistantLand::earlyForgeKickoff) at BeginScene(0), BEFORE the
        // kickoff; polling in the kickoff flipped g_enabled/g_waterEnabled between that
        // latch and the stage-0 suppression gates — a mid-frame mixed state that could
        // e.g. fire the shadow/reflection RPCs inside the still-open async window. Costs
        // one frame of latency on dev toggles; every gate now sees one value per frame.
        if (g_initOk) {
            // F11: live composite toggle.
            if (GetAsyncKeyState(VK_F11) & 0x0001) {
                g_enabled = !g_enabled;
                LOG::logline(">> [seam] composite %s", g_enabled ? "ON" : "OFF");
            }
            // F12 diagnostic: scatter each object by a fixed per-slot offset so any object that
            // is drawn more than once appears as TWO separated copies of the same mesh (a single
            // draw just looks displaced). Reveals duplicate draws regardless of source.
            if (GetAsyncKeyState(VK_F12) & 0x0001) {
                // Tier 2 GTAO added 3 (AO) + 4 (bent normal); dev panel added 5 (albedo) 6 (lit)
                // 7 (ambient) shading-isolation views; 8 (world normal) 9 (point-light count);
                // P1 shadows added 10 (shadow mask — the host panel's face-id/atlas checkboxes
                // pick what it displays) — cycle is now %11.
                g_debugMode = (g_debugMode + 1) % 11;
                const char* name = (g_debugMode == 1) ? "DEPTH" : (g_debugMode == 2) ? "SCATTER"
                                 : (g_debugMode == 3) ? "AO" : (g_debugMode == 4) ? "BENT NORMAL"
                                 : (g_debugMode == 5) ? "ALBEDO" : (g_debugMode == 6) ? "LIT"
                                 : (g_debugMode == 7) ? "AMBIENT" : (g_debugMode == 8) ? "WORLD NORMAL"
                                 : (g_debugMode == 9) ? "LIGHT COUNT"
                                 : (g_debugMode == 10) ? "SHADOW MASK" : "NORMAL";
                LOG::logline(">> [seam] debug mode %d (%s)", g_debugMode, name);
            }
            // F9 toggles the in-host dev overlay.
            if (GetAsyncKeyState(VK_F9) & 0x0001) {
                g_devUiVisible = !g_devUiVisible;
                LOG::logline(">> [seam] dev overlay %s", g_devUiVisible ? "ON" : "OFF");
            }
            // F7 toggles the Forge WATER takeover. The sky takeover is finished, so the sky
            // pass is always on now and F7 was freed — it drives water (WT1). ON (default) →
            // the host draws its geo-clipmap water surface. OFF → MW's own water (clean A/B).
            if (GetAsyncKeyState(VK_F7) & 0x0001) {
                g_waterEnabled = !g_waterEnabled;
                LOG::logline(">> [seam] Forge water (WT1) %s", g_waterEnabled ? "ON" : "OFF");
            }
            // F8: one-shot host compute-shader hot-reload — rebuild gtao/linearize from the
            // dxil on disk (recompile + redeploy first). Latched into the NEXT kickoff's
            // DevInput. Independent of panel visibility.
            if (GetAsyncKeyState(VK_F8) & 0x0001) {
                g_reloadShadersPending = true;
                LOG::logline(">> [seam] compute shader hot-reload requested (F8)");
            }
        }

        if (!g_kick.rpcPending) {
            return;
        }
        g_kick.rpcPending = false;

        MGE_ZoneScopedN("Forge composite finish");

        // overlap = the MW frame work the host render ran under. In fused mode (Finish
        // called straight after Kickoff) this is ~0 and every bucket reduces to the old
        // serial breakdown — the exact A/B.
        const double tWait0 = nowMs();
        const double overlap = tWait0 - g_kick.tKick;

        double hostMs = 0.0;
        bool ok;
        {
            MGE_ZoneScopedN("Forge renderSceneFinish (host wait)");
            ok = g_client->renderSceneFinish(&hostMs);
        }
        const double tRender = nowMs();
        // Split the wait: hostMs = host self-timed cost; (residual wait + kickoff cost - hostMs)
        // = IPC/sync/host-present overhead. Once the kickoff is hoisted to frame start, the
        // residual wait collapsing toward 0 is the whole point of the async split.
        MGE_TracyPlot("Forge client wait ms", tRender - tWait0);
        MGE_TracyPlot("Forge overlap ms", overlap);
        MGE_TracyPlot("Forge host ms", hostMs);
        if (!ok) {
            return;
        }

        bool copyOk;
        {
            MGE_ZoneScopedN("Forge RT copy");
            copyOk = copyHostRtToDst();
        }
        if (!copyOk) {
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
            MGE_ZoneScopedN("Forge composite blit");
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
        // kickoff issue + residual host wait (host's own GPU time = host[]); overlap =
        // MW frame work the host ran under (NOT client cost). texflush is the prime
        // suspect for the *periodic* dips (textures stream in as you cross into new
        // cells). dt = inter-present delta (the visible dip).
        //
        // feed EXCLUDES the overlap window — it's the client's own cost, comparable
        // across fused/async: (kickoff prep) + (residual wait + copy + blit).
        const double feed = (g_kick.tKick - g_kick.tStart) + (tEnd - tWait0);
        const double renderBucket = (g_kick.tKick - g_kick.tAssign) + (tRender - tWait0);

        // Baseline heartbeat over every composited frame (sees the <kSpikeMs majority).
        // Cut 2B: build first, then geom flush (fold frames capture during build).
        g_hb.feed += feed; g_hb.geom += (g_kick.tGeomFlush - g_kick.tBuild);
        g_hb.build += (g_kick.tBuild - g_kick.tStart);
        g_hb.render += renderBucket; g_hb.host += hostMs; g_hb.overlap += overlap;
        g_hb.copy += (tCopy - tRender); g_hb.blit += (tEnd - tCopy); g_hb.dt += g_kick.dtPresent;
        g_hb.captured += g_kick.capturedCount;
        if (feed > g_hb.maxFeed) g_hb.maxFeed = feed;
        if (g_kick.dtPresent > g_hb.maxDt) g_hb.maxDt = g_kick.dtPresent;
        if (g_kick.early) ++g_hb.earlyN;
        if (++g_hb.n >= kHeartbeatFrames) {
            LOG::logline(">> [hb] %u frames avg: feed=%.2f geom=%.2f build=%.2f render=%.2f[host=%.2f] "
                         "overlap=%.2f copy=%.2f blit=%.2f dt=%.2f early=%u cap=%.1f | max feed=%.2f dt=%.2f (~%.0f fps)",
                         g_hb.n, g_hb.feed / g_hb.n, g_hb.geom / g_hb.n, g_hb.build / g_hb.n,
                         g_hb.render / g_hb.n, g_hb.host / g_hb.n, g_hb.overlap / g_hb.n,
                         g_hb.copy / g_hb.n, g_hb.blit / g_hb.n, g_hb.dt / g_hb.n, g_hb.earlyN,
                         g_hb.captured / g_hb.n,
                         g_hb.maxFeed, g_hb.maxDt,
                         g_hb.dt > 0.0 ? 1000.0 * g_hb.n / g_hb.dt : 0.0);
            g_hb = Accum{};
        }

        if (feed >= kSpikeMs) {
            LOG::logline("!! [spike] frame %u feed=%.2fms (geomflush=%.2f build=%.2f texflush=%.2f "
                         "assign=%.2f render=%.2f[host=%.2f] overlap=%.2f copy=%.2f blit=%.2f) "
                         "draws=%u skin=%u mm=%u light=%u alpha=%u cap=%u geom+=%u/%uKB tex+=%u/%uKB dt=%.2fms",
                         g_kick.frame, feed,
                         g_kick.tGeomFlush - g_kick.tBuild, g_kick.tBuild - g_kick.tStart,
                         g_kick.tTexFlush - g_kick.tGeomFlush, g_kick.tAssign - g_kick.tTexFlush,
                         renderBucket, hostMs, overlap, tCopy - tRender, tEnd - tCopy,
                         g_kick.drawCount, g_kick.skinnedCount, g_kick.multiMapCount, g_kick.lightCount,
                         g_kick.alphaCount, g_kick.capturedCount,
                         g_kick.geomParts, (unsigned)(g_kick.geomBytes >> 10),
                         g_kick.texCount, (unsigned)(g_kick.texBytes >> 10),
                         g_kick.dtPresent);
        }
    }

    void onStage0Composite(IDirect3DDevice9* device) {
        // Fused kickoff+finish — the exact pre-split serial behaviour (A/B reference,
        // UseAsyncHostFrame=0). overlap ≈ 0 by construction.
        onStage0CompositeKickoff(device);
        onStage0CompositeFinish(device);
    }

    bool kickoffPending() {
        return g_kick.rpcPending;
    }

    bool wantsGeometryCapture() {
        return g_initOk && g_geomVec.has_value();
    }

    bool wantsSkyCapture() {
        // Sky takeover is DONE: walk skyRoot whenever the seam is live, geometry capture is up, and
        // the composite is ON (F11). g_skyEnabled is now permanently true (F7 freed for water).
        return g_initOk && g_geomVec.has_value() && g_enabled && g_skyEnabled;
    }

    bool wantsWaterCapture() {
        // WT1 Forge water: true when the seam is live, the composite is ON (F11), and the Forge water
        // pass is toggled ON (F7). No geometry capture (the host generates the mesh); this gate drives
        // the per-frame water-params crossing and (WT3) the MGE water-pass suppression.
        return g_initOk && g_enabled && g_waterEnabled;
    }

    bool ownsOpaqueWorld() {
        // Forge composites a full-screen blit over MW's frame when enabled; the engine's
        // scene-0 opaque draw underneath is then pure wasted cost. Gate on the live composite
        // toggle so F11-off restores normal engine rendering (clean A/B of the double cost).
        return g_initOk && g_enabled;
    }

    bool ownsDistantLand() {
        // The host draws exterior distant land + statics into the Forge frame; the composite
        // lays it over MW. So MW's own main-view DL color draw is redundant when the seam is
        // compositing. Same gate as ownsOpaqueWorld (F11 = g_enabled) for a clean A/B; the
        // caller adds the exterior check (Forge DL is exterior-only). See header.
        return g_initOk && g_enabled;
    }

    void captureAlphaDraw(const RenderedState* rs, const FragmentState* frs) {
        if (!Configuration.ForgeAlphaCapture) return;
        if (!g_initOk || !g_enabled || !g_capturedVec) return;
        if (!rs || !frs) return;
        // HW-skinned blends (ghosts) excluded — the bind-pose VB here is the wrong pose (a
        // separate follow-up). Need an indexed TRIANGLELIST with a real stride (TRISTRIP/FAN and
        // non-indexed particle DIPs are dropped; a counter would reveal if MW emits any).
        if (rs->vertexBlendState != 0) return;
        if (rs->primType != D3DPT_TRIANGLELIST) return;
        if (!rs->vb || !rs->ib || rs->vbStride == 0) return;
        if ((rs->fvf & D3DFVF_POSITION_MASK) != D3DFVF_XYZ) return;   // untransformed XYZ only
        if (rs->vertCount == 0 || rs->primCount == 0) return;

        // Old-msoc double-draw guard: this DIP is a duplicate iff the host already draws a cached
        // blended shape with the same (GPU texture, vertexCount) — the cache draws from its own VB
        // copies so rs->vb never matches, but (tex, vertCount) does (the vertCount term kills the
        // shared-texture false positive — particle counts vary). g_alphaDedup is built this frame
        // in buildGeometryDrawLists while pushing the cached alphaCands.
        const std::uint64_t dedupKey =
            ((std::uint64_t)(std::uintptr_t)rs->texture << 32) | (std::uint64_t)rs->vertCount;
        if (g_alphaDedup.find(dedupKey) != g_alphaDedup.end()) return;

        // Caps: keep the shared captured buffers within their single-chunk budget. Drop WHOLE
        // (never partial) on overflow so an index range can't dangle past the shipped vertex window.
        const std::uint32_t idxCount = rs->primCount * 3;
        if (g_capRecs.size() >= IPC::kMaxCapturedAlphaDraws
            || g_capVertScratch.size() + rs->vertCount > IPC::kMaxCapturedAlphaVerts
            || g_capIdxScratch.size() + idxCount > IPC::kMaxCapturedAlphaIndices) {
            if (g_capDropCap++ == 0) {
                LOG::logline("!! [alpha-cap] captured-alpha cap hit (draws/verts/indices) — dropping rest this session-window");
            }
            return;
        }

        // Texture slot: rs->texture is the proxy realTexture pointer, identical to the GPU texture
        // the cache walk registered in g_textureNameMap. Resolve name -> bindless slot (per-frame
        // memo). No name -> slot 0 (host default white): a visible white particle beats an invisible
        // one, and NiFlipController textures self-correct after the first walk registers them.
        std::uint32_t texIndex;
        auto mit = g_capTexMemo.find(rs->texture);
        if (mit != g_capTexMemo.end()) {
            texIndex = mit->second;
        } else {
            const char* name = MGE::GeometryCache::resolveTextureName(rs->texture);
            texIndex = name ? resolveTextureSlot(name) : 0u;
            if (!name && g_capNoName++ < 20) {
                LOG::logline("!! [alpha-cap] no source name for tex=%p (white)", (void*)rs->texture);
            }
            g_capTexMemo.emplace(rs->texture, texIndex);
        }

        // FVF offsets within the source stride (XYZ at 0; then NORMAL, PSIZE, DIFFUSE, SPECULAR,
        // TEX in D3D order). We read pos/normal/diffuse/UV0.
        const UINT stride = rs->vbStride;
        UINT off = 12;   // past XYZ
        const bool hasNorm = (rs->fvf & D3DFVF_NORMAL) != 0; const UINT normOff = off; if (hasNorm) off += 12;
        if (rs->fvf & D3DFVF_PSIZE) off += 4;
        const bool hasCol = (rs->fvf & D3DFVF_DIFFUSE) != 0;  const UINT colOff = off; if (hasCol) off += 4;
        if (rs->fvf & D3DFVF_SPECULAR) off += 4;
        const UINT texCount = (rs->fvf & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT;
        const bool hasUV = texCount >= 1; const UINT uvOff = off;

        // vColSource forward map (inverse proven at rendercachedcolor.cpp:141-142). The captured
        // frag reads real vertex color for source 1 and vertex alpha for any nonzero source, so we
        // write the REAL captured D3DCOLOR whenever vColSource != 0 (particles fade via vertex alpha).
        std::uint32_t vColSource = 0;
        if (hasCol) {
            if (rs->matSrcDiffuse == D3DMCS_COLOR1) vColSource = 2;
            else if (rs->matSrcEmissive == D3DMCS_COLOR1) vColSource = 1;
        }

        // TEMP AT3 lighting-mismatch diagnostic (remove after triage): dump each first-seen captured
        // source-name's full state — FVF layout (rule out a wrong normal offset), lighting/vColSource/
        // material (the frag's lit path), and whether stage 1+ carries a texture (multi-map: the frag
        // only samples the BASE map, so a dark/detail/glow blend renders wrong). Keyed on the name ptr.
        {
            static std::unordered_set<const void*> s_capDiagSeen;
            const char* dname = MGE::GeometryCache::resolveTextureName(rs->texture);
            const void* dkey = dname ? (const void*)dname : (const void*)rs->texture;
            if (s_capDiagSeen.insert(dkey).second && s_capDiagSeen.size() <= 64) {
                const int st1 = (int)frs->stage[1].colorOp, st2 = (int)frs->stage[2].colorOp, st3 = (int)frs->stage[3].colorOp;
                LOG::logline(">> [cap-diag] %s fvf=0x%X stride=%u norm=%d@%u col=%d@%u uv=%d@%u "
                             "vcs=%u lit=%u matD=(%.2f,%.2f,%.2f) matA=%.2f stageOps=[%d,%d,%d,%d]",
                             dname ? dname : "(null)", (unsigned)rs->fvf, (unsigned)stride,
                             (int)hasNorm, (unsigned)normOff, (int)hasCol, (unsigned)colOff, (int)hasUV, (unsigned)uvOff,
                             vColSource, (unsigned)rs->useLighting,
                             frs->material.diffuse.r, frs->material.diffuse.g, frs->material.diffuse.b,
                             frs->material.diffuse.a, (int)frs->stage[0].colorOp, st1, st2, st3);
            }
        }

        // Lock VB whole-remainder from vbOffset + IB (computeBoundingBox's exact flags); graceful drop.
        void* pVerts = nullptr;
        if (FAILED(rs->vb->Lock(rs->vbOffset, 0, &pVerts, D3DLOCK_READONLY | D3DLOCK_NOSYSLOCK))) {
            if (g_capDropLock++ == 0) LOG::logline("!! [alpha-cap] VB lock failed — dropping");
            return;
        }
        void* pIdx = nullptr;
        if (FAILED(rs->ib->Lock(0, 0, &pIdx, D3DLOCK_READONLY | D3DLOCK_NOSYSLOCK))) {
            rs->vb->Unlock();
            if (g_capDropLock++ == 0) LOG::logline("!! [alpha-cap] IB lock failed — dropping");
            return;
        }
        D3DINDEXBUFFER_DESC ibd; rs->ib->GetDesc(&ibd);
        const bool is16 = (ibd.Format == D3DFMT_INDEX16);

        // Copy the vertex window [baseIndex+minIndex, +vertCount) → captured VB, building the
        // model-space bbox for the centroid. (The host adds vertexBase to each index, so indices
        // are rebased to 0 within this window below.)
        const std::uint32_t vBase = (std::uint32_t)g_capVertScratch.size();
        const UINT srcVBase = rs->baseIndex + rs->minIndex;
        float mn[3] = { 1e30f, 1e30f, 1e30f }, mx[3] = { -1e30f, -1e30f, -1e30f };
        for (UINT j = 0; j < rs->vertCount; ++j) {
            const BYTE* v = (const BYTE*)pVerts + (size_t)(srcVBase + j) * stride;
            const float* p = (const float*)v;
            IPC::GeomVertexWire out;
            out.px = p[0]; out.py = p[1]; out.pz = p[2];
            if (hasNorm) { const float* n = (const float*)(v + normOff); out.nx = n[0]; out.ny = n[1]; out.nz = n[2]; }
            else { out.nx = 0.0f; out.ny = 0.0f; out.nz = 1.0f; }
            out.color = (vColSource != 0) ? *(const std::uint32_t*)(v + colOff) : 0xFFFFFFFFu;
            if (hasUV) { const float* t = (const float*)(v + uvOff); out.u = t[0]; out.v = t[1]; }
            else { out.u = 0.0f; out.v = 0.0f; }
            g_capVertScratch.push_back(out);
            for (int c = 0; c < 3; ++c) { mn[c] = std::min(mn[c], p[c]); mx[c] = std::max(mx[c], p[c]); }
        }

        // Copy primCount*3 indices from startIndex, rebased by -minIndex → [0, vertCount). Any
        // rebased value out of uint16 → drop the whole record (roll back the pushed verts+indices).
        const std::uint32_t iBase = (std::uint32_t)g_capIdxScratch.size();
        bool idxDrop = false;
        for (UINT i = 0; i < idxCount; ++i) {
            const UINT raw = is16 ? ((const WORD*)pIdx)[rs->startIndex + i]
                                  : ((const DWORD*)pIdx)[rs->startIndex + i];
            const long rebased = (long)raw - (long)rs->minIndex;
            if (rebased < 0 || rebased > 0xFFFF) { idxDrop = true; break; }
            g_capIdxScratch.push_back((std::uint16_t)rebased);
        }
        rs->ib->Unlock();
        rs->vb->Unlock();

        if (idxDrop) {
            g_capVertScratch.resize(vBase);
            g_capIdxScratch.resize(iBase);
            if (g_capDropIdx32++ == 0) LOG::logline("!! [alpha-cap] index rebase out of uint16 — dropping");
            return;
        }
        if (mn[0] > mx[0]) { g_capVertScratch.resize(vBase); g_capIdxScratch.resize(iBase); return; }

        // World-space centroid via worldTransforms[0] (model-space bbox center → world). The
        // camera-relative subtraction is applied at emit with the CURRENT eyePos (no swim).
        const float cx = 0.5f * (mn[0] + mx[0]), cy = 0.5f * (mn[1] + mx[1]), cz = 0.5f * (mn[2] + mx[2]);
        const D3DXMATRIX& w = rs->worldTransforms[0];

        CapturedAlphaRec rec;
        rec.vertexBase = vBase; rec.vertexCount = rs->vertCount;
        rec.indexBase = iBase;  rec.indexCount = idxCount;
        memcpy(rec.world, &w, 16 * sizeof(float));   // D3DXMATRIX is row-major (host convention)
        rec.centroid[0] = cx * w._11 + cy * w._21 + cz * w._31 + w._41;
        rec.centroid[1] = cx * w._12 + cy * w._22 + cz * w._32 + w._42;
        rec.centroid[2] = cx * w._13 + cy * w._23 + cz * w._33 + w._43;
        rec.texIndex = texIndex;
        rec.srcBlend = rs->srcBlend; rec.destBlend = rs->destBlend;
        rec.alphaRef = rs->alphaTest
            ? (float)(rs->alphaRef + (rs->alphaFunc == D3DCMP_GREATER ? 1 : 0)) / 255.0f : 0.0f;
        rec.vColSource = vColSource;
        // FFP-unlit approximation (verify in-game): unlit draws (particles) get zero diffuse/ambient
        // + white emissive so the frag emits texture * vcol; lit draws pass the material through.
        if (rs->useLighting == 0) {
            rec.matDiffuse[0] = rec.matDiffuse[1] = rec.matDiffuse[2] = 0.0f;
            rec.matAmbient[0] = rec.matAmbient[1] = rec.matAmbient[2] = 0.0f;
            rec.matEmissive[0] = rec.matEmissive[1] = rec.matEmissive[2] = 1.0f;
        } else {
            rec.matDiffuse[0]  = frs->material.diffuse.r;  rec.matDiffuse[1]  = frs->material.diffuse.g;  rec.matDiffuse[2]  = frs->material.diffuse.b;
            rec.matAmbient[0]  = frs->material.ambient.r;  rec.matAmbient[1]  = frs->material.ambient.g;  rec.matAmbient[2]  = frs->material.ambient.b;
            rec.matEmissive[0] = frs->material.emissive.r; rec.matEmissive[1] = frs->material.emissive.g; rec.matEmissive[2] = frs->material.emissive.b;
        }
        rec.matAlpha = frs->material.diffuse.a;
        g_capRecs.push_back(rec);
    }

    void captureGeometry(std::uint32_t key, std::uint16_t revision, std::uint32_t modelId,
                         const IPC::GeomVertexWire* verts, std::uint32_t vertexCount,
                         const std::uint16_t* indices, std::uint32_t indexCount,
                         bool forceReupload) {
        if (!wantsGeometryCapture() || !verts || !indices || !vertexCount || !indexCount) {
            return;
        }
        // Skip only if the SAME object (modelId), same shape (vertexCount) and same revision
        // was already shipped. Keying on revision alone aliased recycled NiTriShape* keys (a
        // freed object's key+slot inherited by a new mesh with a colliding revisionID).
        // forceReupload (SK1 sky dome) bypasses this — its vertex colours change every frame
        // without a revisionID bump, so the dedup would otherwise freeze the gradient.
        auto rev = g_uploadedRev.find(key);
        if (!forceReupload && rev != g_uploadedRev.end() && rev->second.id == modelId &&
            rev->second.vc == vertexCount && rev->second.rev == revision) {
            return;
        }
        // Stable host slot per cache key (reused on re-upload so the host frees
        // and rebuilds in place).
        std::uint32_t slot;
        auto ks = g_keySlot.find(key);
        if (ks != g_keySlot.end()) {
            slot = ks->second.slot;
        } else {
            slot = g_nextSlot++;
            g_keySlot.emplace(key, SlotInfo{ slot });
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
            slot = ks->second.slot;
        } else {
            slot = g_nextSlot++;
            g_keySlot.emplace(key, SlotInfo{ slot });
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
            slot = ks->second.slot;
        } else {
            slot = g_nextSlot++;
            g_keySlot.emplace(key, SlotInfo{ slot });
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
        g_skyVec.reset();
        g_alphaVec.reset();
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
        g_skyScratch.clear();
        g_skyScratch.shrink_to_fit();
        g_alphaScratch.clear();
        g_alphaScratch.shrink_to_fit();
        g_texPendingBlob.clear();
        g_texPendingBlob.shrink_to_fit();
        g_pendingParts = 0;
        g_texPendingCount = 0;
        g_keySlot.clear();
        g_uploadedRev.clear();
        g_texSlot.clear();
        g_slotName.clear();
        g_slotLastUsed.clear();
        g_nextSlot = 0;
        g_nextTexSlot = 1;
        g_initOk = false;
        g_enabled = false;
    }
}
