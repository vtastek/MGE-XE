#pragma once

#include "proxydx/d3d9header.h"
#include "mge/dlmath.h"

#include <cstddef>
#include <cstdint>

// we could use the MS extensions __ptr32 and __ptr64 instead of this conditional definition,
// but that makes the value appear as a pointer on both sides, which might give the
// impression that the pointer is valid on both sides. with the condition, we can represent
// non-shared pointers as opaque integers on the remote side, so there's no chance of confusion.
#ifdef MGE64_HOST
template<typename T> using ptr32 = std::uint32_t;
template<typename T> using ptr64 = T*;
#else
template<typename T> using ptr32 = T*;
template<typename T> using ptr64 = std::uint64_t;
#endif
// handles, on the other hand, are always opaque, so we wouldn't try to dereference one. also,
// a 32-bit handle could be valid on the 64-bit side if it's inherited. so for those reasons,
// we will use __ptr32 here.
typedef void* __ptr32 HANDLE32;

constexpr DWORD VIS_NEAR =     0x01;
constexpr DWORD VIS_FAR =      0x02;
constexpr DWORD VIS_VERY_FAR = 0x04;
constexpr DWORD VIS_GRASS =    0x08;
constexpr DWORD VIS_LAND =     0x10;
constexpr DWORD VIS_STATIC = VIS_NEAR | VIS_FAR | VIS_VERY_FAR;

// ensure consistent layout between 32-bit and 64-bit processes
#pragma pack(push, 4)
struct RenderMesh {
    bool enabled, hasAlpha, animateUV;

    ptr32<IDirect3DTexture9> tex;
    D3DXMATRIX transform;
    int verts;
    ptr32<IDirect3DVertexBuffer9> vBuffer;
    int faces;
    ptr32<IDirect3DIndexBuffer9> iBuffer;

    // World-space bounding sphere (post-transform). Promoted from
    // QuadTreeMesh so it survives the IPC wire copy — consumers need
    // real per-mesh bounds for occlusion tests and anything else that
    // can't rely on a conservative global radius. 16 bytes at pack(4).
    BoundingSphere sphere;

    // World-space oriented bounding box (post-transform). Promoted
    // alongside sphere so the occlusion path can escalate to a tighter
    // 12-triangle test for giants whose sphere is too loose to resolve
    // as OCCLUDED. center + (vx, vy, vz) axis half-extents; 52 bytes.
    BoundingBox box;
};

struct ViewFrustum {
    D3DXPLANE frustum[6];
    enum Containment { INSIDE, OUTSIDE, INTERSECTS };

    ViewFrustum(const D3DXMATRIX* viewProj);

    Containment ContainsSphere(const BoundingSphere& sphere) const;
    Containment ContainsBox(const BoundingBox& box) const;
};

enum VisibleSetSort : std::uint8_t {
    None,
    ByState,
    ByTexture,
};

namespace IPC {
    constexpr DWORD MaxWait = 60000;

    typedef std::uint32_t VecId;
    constexpr VecId InvalidVector = static_cast<VecId>(-1);

    static inline void CleanupHandle(HANDLE& h) {
        if (h != INVALID_HANDLE_VALUE && h != NULL) {
            CloseHandle(h);
        }
        h = INVALID_HANDLE_VALUE;
    }

    // these APIs aren't supported until Windows 8 or 10, so we load them dynamically
    typedef decltype(&::MapViewOfFileNuma2) MapViewOfFileNuma2_t;
    typedef decltype(&::MapViewOfFile3) MapViewOfFile3_t;
    typedef decltype(&::UnmapViewOfFileEx) UnmapViewOfFileEx_t;
    typedef decltype(&::UnmapViewOfFile2) UnmapViewOfFile2_t;
    typedef decltype(&::VirtualAlloc2) VirtualAlloc2_t;

    extern MapViewOfFileNuma2_t MapViewOfFileNuma2;
    extern MapViewOfFile3_t MapViewOfFile3;
    extern UnmapViewOfFileEx_t UnmapViewOfFileEx;
    extern UnmapViewOfFile2_t UnmapViewOfFile2;
    extern VirtualAlloc2_t VirtualAlloc2;

    extern bool initImports();

    class Client;
    class Server;

    enum WakeReason {
        Update,
        Complete,
        ServerLost,
        Timeout,
        Error
    };

    enum Command: std::uint32_t {
        None,
        AllocVec,
        FreeVec,
        Exit,
        UpdateDynVis,
        InitDistantStatics,
        InitLandscape,
        SetWorldSpace,
        GetVisibleMeshesCoarse,
        GetVisibleMeshes,
        SortVisibleSet,
        // Batched variant of GetVisibleMeshes: runs all 3 distant-statics
        // range queries (Near/Far/VeryFar) plus the sort in one RPC,
        // collapsing 4 sequential client-server round trips into 1. Used
        // by DistantLand::cullDistantStatics_kickoff to let the server-
        // side quadtree work overlap with shadow / curtain rendering.
        GetVisibleMeshesAllRanges,

        // --- Present-seam spike (out-of-process 64-bit renderer) ---
        // RenderInit: bring up the host-side Vulkan renderer for a fixed WxH
        // target and bind the shared framebuffer vec that carries pixels back
        // to the 32-bit side (Milestone A, CPU staging). RenderFrame: render one
        // frame into that vec. Both no-ops unless the spike is enabled client-side.
        RenderInit,
        RenderFrame,

        // M1b: upload a batch of static opaque meshes (model-space pos+normal+indices,
        // slot-indexed) into the Forge host's mesh store. The blob vec holds
        // [GeomPartWire+verts+indices]*partCount (see ipc/geomwire.h).
        GeomUpload,

        // Phase 2 bindless texturing: upload a batch of base-map textures. The blob vec holds
        // [TexUploadWire][dds bytes]*texCount (see ipc/geomwire.h); the host decodes each DDS
        // into gTextures[slot]. Rides the same dedicated geometry channel as GeomUpload.
        TexUpload,
    };

    struct AllocVecParameters {
        IN std::uint32_t maxCapacityInElements;
        IN std::uint32_t windowSizeInElements;
        IN std::uint32_t elementSize;
        IN std::uint32_t initialCapacity;

        OUT std::uint32_t reservedBytes;
        OUT std::uint32_t windowBytes;
        OUT std::uint32_t headerBytes;
        OUT ptr32<void> header32;
        OUT VecId id;
    };

    struct FreeVecParameters {
        IN VecId id;

        OUT bool wasFreed;
    };

    struct DynVisFlag {
        std::uint16_t groupIndex;
        bool enable;
    };

    struct DynVisParameters {
        IN VecId id;
    };

    struct DistantStaticParameters {
        IN VecId distantStatics;
        IN VecId distantSubsets;
    };

    struct LandscapeBuffers {
        ptr32<IDirect3DVertexBuffer9> vb;
        ptr32<IDirect3DIndexBuffer9> ib;
    };

    struct InitLandscapeParameters {
        IN VecId buffers;
        IN ptr32<IDirect3DTexture9> texWorldColour;
    };

    struct SetWorldSpaceParameters {
        IN char cellname[64];

        OUT bool cellFound;
    };

    struct GetMeshesParameters {
        IN VecId visibleSet;
        IN VisibleSetSort sort;
        IN ViewFrustum viewFrustum;
        IN DWORD setFlags;
        IN D3DXVECTOR4 viewSphere;
    };

    // Batched 3-range variant. rangeCount selects how many entries of the
    // arrays are live (1..3). Total size ≈ 3 × (96 + 16 + 4) + small =
    // ~360 bytes, well under the union budget.
    //
    // Optionally piggybacks a 4th, independent reflection-statics query into
    // the same RPC (reflFlags != 0). The reflection set writes a SEPARATE
    // output vec (reflSet) with its own frustum/sphere/sort, so the reflection
    // server-cull overlaps the kickoff→drain head-start window instead of
    // being a sequential worker RPC. reflFlags == 0 ⇒ no reflection query.
    // Adds ~124 B (VecId + ViewFrustum + D3DXVECTOR4 + DWORD + sort) → ~484 B,
    // still well under the union budget.
    struct GetMeshesAllRangesParameters {
        IN VecId visibleSet;
        IN VisibleSetSort sort;
        IN std::uint8_t rangeCount;
        IN ViewFrustum viewFrustum[3];
        IN D3DXVECTOR4 viewSphere[3];
        IN DWORD setFlags[3];

        IN VecId reflSet;
        IN DWORD reflFlags;          // 0 ⇒ no reflection query this RPC
        IN ViewFrustum reflFrustum;
        IN D3DXVECTOR4 reflSphere;
        IN VisibleSetSort reflSort;

        // Host-side occlusion cull. occlusionMask is a Vec holding the
        // [OcclusionMask::Header][raw MOC ZTile buffer] blob shipped from
        // msoc.dll; the server reconstructs the mask and TestRect-culls each
        // frustum survivor before PushBack, so only visible statics cross the
        // wire. InvalidVector ⇒ no host cull (behaves exactly as before).
        IN VecId occlusionMask;
    };

    // --- Present-seam spike params ---
    // Milestone A uses a dedicated flat shared mapping (not a windowed Vec) for the
    // framebuffer: a W*H*4 blob must stay contiguous on both sides, and the windowed
    // Vec's reservedBytes = maxSize*windowBytes term overflows uint32 for a ~1MB
    // single-window allocation. The host creates the mapping and duplicates the handle
    // back into the client process (the same cross-process handle-sharing Vec::init
    // does), so the client maps it read-only-ish for the blit. Milestone B swaps this
    // CPU mapping for a shared GPU texture handle in the same OUT slot.
    struct RenderInitParameters {
        IN std::uint32_t width;
        IN std::uint32_t height;
        // MSAA: MGE's Configuration.AALevel (the D3DMULTISAMPLE value 0/2/4/8) mapped to a
        // GPU sample count (>=1). 1 ⇒ no antialiasing (host renders straight into the shared
        // single-sample RT, as before). >1 ⇒ host renders into an internal MSAA color+depth
        // and resolves down into the shared RT before the D3D9 handoff. If the device doesn't
        // support the requested count the host logs and drops to 1.
        IN std::uint32_t sampleCount;
        // Anisotropic filtering: MGE's Configuration.AnisoLevel (0 = off/linear, else the max
        // anisotropy 2..16). The host builds its texture sampler from this (Phase 2 texturing).
        IN std::uint32_t anisoLevel;
        // Milestone B/C: D3D9Ex shared render-target HANDLE(s) (KMT/global) the client created.
        // [0] non-null ⇒ the host imports them as Vulkan external memory and renders directly
        // (zero-copy). [1] non-null double-buffers (C). Both null ⇒ Milestone A CPU readback.
        IN HANDLE32 sharedTextureHandles[2];

        // Route 1 (Forge D3D12 host): the host's SHARED D3D12 render-target NT handle,
        // duplicated into the CLIENT process. MW ingests it via D3D9Ex CreateTexture.
        // (Formerly the Milestone-A file-mapping handle; the IN sharedTextureHandles[]
        // above are now ignored — the host owns the RT. Field name kept to avoid churn.)
        OUT HANDLE32 framebufferHandle;
        OUT bool ok;
    };

    // Dev overlay input snapshot forwarded client -> host each frame (Stage 2). Mouse is in host
    // render-target pixels (client already did ScreenToClient). buttons: bit0=L,bit1=R,bit2=M.
    struct DevInput {
        std::int32_t  x = 0;
        std::int32_t  y = 0;
        std::uint32_t buttons = 0;
        float         wheel = 0.0f;
        std::uint32_t uiVisible = 0;
        std::uint32_t reloadShaders = 0;   // one-shot (F8 edge): host rebuilds compute pipelines from disk
    };

    struct RenderFrameParameters {
        IN std::uint32_t frameIndex;     // for logging
        IN std::uint32_t targetIndex;    // which shared buffer to render into (double-buffer, C)

        // M1c scene path: when drawList != InvalidVector the host renders the cached
        // opaque scene (camera below + DrawItemWire[] in the drawList vec) instead of
        // the triangle. viewProj is D3DXMATRIX bytes (row-major) — uploaded straight to
        // the host's gFrameData cbuffer (no transpose, see opaque.srt.h).
        IN float viewProj[16];
        // Tier 1 lighting: 6 × float4 = 24 floats (sunDir, sunCol, ambCol, fogColNear,
        // fogParams[x=fogNearStart,y=fogNearEnd], eyePos), uploaded into gFrameData after
        // viewProj. From DistantLand each frame (see RenderProcess::onPresent).
        IN float lighting[24];
        IN VecId drawList;               // chunked byte vec of DrawItemWire[]; Invalid ⇒ triangle
        IN std::uint32_t drawCount;
        IN std::uint32_t drawBytes;

        // M-Skinning: parallel per-frame skinned draw list. skinnedList is a chunked byte
        // vec of [SkinnedDrawWire][palette]* (geomwire.h); Invalid ⇒ no skinned draws. The
        // scene path runs when EITHER drawList OR skinnedList is valid.
        IN VecId skinnedList;
        IN std::uint32_t skinnedCount;
        IN std::uint32_t skinnedBytes;

        // Tier 4 multi-map: parallel per-frame draw list of MultiMapDrawWire[] (geomwire.h) —
        // static opaque parts carrying dark/detail/glow sibling maps, drawn through the host's
        // wide multimap pipeline. Invalid ⇒ no multi-map draws. The scene path runs when ANY of
        // drawList / skinnedList / multiMapList is valid.
        IN VecId multiMapList;
        IN std::uint32_t multiMapCount;
        IN std::uint32_t multiMapBytes;

        // Tier 3a point lights: a chunked byte vec of PointLightWire[] (geomwire.h),
        // world-space, uploaded into the host's light cbuffer for the per-pixel FFE
        // evalOnePointLight loop. Invalid / count 0 ⇒ no point lights this frame.
        IN VecId lightList;
        IN std::uint32_t lightCount;
        IN std::uint32_t lightBytes;

        // F12 debug view: 0 = normal, 1 = depth (world-distance grayscale), 2 = scatter.
        // Appended after the light-list fields so existing field offsets are unchanged.
        IN std::uint32_t debugMode;

        // Dev overlay input bridge (Stage 2): the headless host has no window/InputSystem, so the
        // client polls the mouse (client-space pixels) + a visibility toggle and forwards it here
        // each frame. The host pushes these into Forge UI via uiSetExternalInput. devMouseButtons
        // is a bitmask: bit0=L, bit1=R, bit2=M. devUiVisible nonzero shows/activates the panel.
        IN std::int32_t  devMouseX;
        IN std::int32_t  devMouseY;
        IN std::uint32_t devMouseButtons;
        IN float         devMouseWheel;
        IN std::uint32_t devUiVisible;
        IN std::uint32_t devReloadShaders;   // one-shot (F8 edge): host rebuilds compute pipelines from disk

        OUT std::uint32_t bytesWritten;
        OUT double renderMs;             // host-side render+readback time
    };

    // M1b geometry upload. blob = a byte VecId holding partCount packed parts
    // (GeomPartWire+verts+indices each, ipc/geomwire.h). The host builds a D3D12
    // VB/IB per part and stores it in its slot-indexed mesh array.
    struct GeomUploadParameters {
        IN VecId blob;
        IN std::uint32_t partCount;
        IN std::uint32_t byteCount;

        OUT std::uint32_t partsUploaded;
    };

    // Phase 2 texturing: blob vec of [TexUploadWire][dds bytes]*texCount (ipc/geomwire.h).
    struct TexUploadParameters {
        IN VecId blob;
        IN std::uint32_t texCount;
        IN std::uint32_t byteCount;

        OUT std::uint32_t texturesUploaded;
    };

	struct Parameters {
        Command command;
        union {
            AllocVecParameters allocVecParams;
            FreeVecParameters freeVecParams;
            DynVisParameters dynVisParams;
            DistantStaticParameters distantStaticParams;
            InitLandscapeParameters initLandscapeParams;
            SetWorldSpaceParameters worldSpaceParams;
            GetMeshesParameters meshParams;
            GetMeshesAllRangesParameters meshAllRangesParams;
            RenderInitParameters renderInitParams;
            RenderFrameParameters renderFrameParams;
            GeomUploadParameters geomUploadParams;
            TexUploadParameters texUploadParams;
        } params;
	};
}
#pragma pack(pop)