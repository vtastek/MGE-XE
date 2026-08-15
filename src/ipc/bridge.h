#pragma once

#include "proxydx/d3d9header.h"
#include "mge/dlmath.h"
#include "ipc/hostframetimings.h"   // IPC::HostFrameTimings (RenderFrameParameters OUT block)

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

    // R2 actor-ripple wire cap (see RenderFrameParameters::actorRipples). This is a PER-PIXEL LOOP
    // BOUND before it is a bandwidth number: the water frag tests every entry against every water
    // pixel.
    //
    // 64 because 32 measurably clipped. MW's pool ran pegged at 75/75 with births peaking at 80/s
    // — a single swimmer lays down life*speed/14.2 ≈ 21 rings for its own wake, so 32 shared across
    // the player plus any nearby actor starved everyone. 64 covers the player and two or three
    // others; the pool's own 75 is the hard ceiling above that, and rain competes for the same
    // slots unless Morrowind.ini's `Rain Ripples` is off.
    //
    // Sized to the GPU side as well: entries land in the water pass's spare worlds[] matrices, 4
    // ripples per float4x4, so 64 is exactly 16 slots out of ~1012 free.
    constexpr std::uint32_t kMaxActorRipples = 64;

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

        // (MSOC retirement D5: an IN VecId occlusionMask followed, carrying the
        // shipped MOC mask blob the server culled distant statics against. No
        // caller ever set it — the host's Hi-Z GPU cull owns occlusion — so the
        // field and the whole server-side filter behind it are gone.)
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

        // Tier 1 (tasks/forge-host-gpu-lane.md): the host's SHARED, monotonic D3D12 frame FENCE,
        // duplicated into the CLIENT process. The client imports it as a Vulkan timeline semaphore
        // and waits on it in its RT copy, which is what lets the host stop CPU-blocking on its own
        // fence (the RPC reply then no longer means "GPU-complete"). Null ⇒ the host could not
        // create a shared fence; the client must keep the old blocking contract. Appended after
        // `ok` so every existing field offset is unchanged.
        OUT HANDLE32 frameFenceHandle;
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
        std::uint32_t distLightsToggle = 0; // one-shot (numpad- edge): host flips baked distant-light loop (perf A/B)
        std::uint32_t gpuCapture = 0;      // one-shot (numpad0 edge): N host frames of RenderDoc capture
        std::uint32_t dumpHdr = 0;         // one-shot (numpad1 edge): dump the LINEAR scene target as .exr + .tga
        // Frame-ahead observability: the client's own last-frame timings, shown live in the
        // host Stats panel. Purely informational — no host behaviour keys off these.
        std::uint32_t frameAhead = 0;      // 1 = deferred-finish pipelining live (default on; numpad-*)
        float clientWaitMs = 0.0f;         // last residual collect wait (the pipeline success metric)
        float clientDtMs = 0.0f;           // last whole MW frame delta (kickoff-to-kickoff)
        float clientMwStartMs = 0.0f;      // last Present-return -> BeginScene(0) gap (engine sim)
    };

    // FP1a: the per-frame first-person bundle handed to renderSceneKickoff as ONE optional
    // pointer (nullptr ⇒ fpEnabled=0, no FP pass) instead of another 10 positional args.
    // Field meanings match the fp* wire fields in RenderFrameParameters below.
    struct FPFrame {
        float viewProj[16] = {};
        VecId drawList = InvalidVector;
        std::uint32_t drawCount = 0;
        std::uint32_t drawBytes = 0;
        VecId skinnedList = InvalidVector;
        std::uint32_t skinnedCount = 0;
        std::uint32_t skinnedBytes = 0;
        VecId alphaList = InvalidVector;      // FP1c
        std::uint32_t alphaCount = 0;
        std::uint32_t alphaBytes = 0;
    };

    struct RenderFrameParameters {
        IN std::uint32_t frameIndex;     // for logging
        IN std::uint32_t targetIndex;    // which shared buffer to render into (double-buffer, C)

        // M1c scene path: when drawList != InvalidVector the host renders the cached
        // opaque scene (camera below + DrawItemWire[] in the drawList vec) instead of
        // the triangle. viewProj is D3DXMATRIX bytes (row-major) — uploaded straight to
        // the host's gFrameData cbuffer (no transpose, see opaque.srt.h).
        IN float viewProj[16];
        // Tier 1 lighting: 6 × float4 (sunDir, sunCol, ambCol, fogColNear,
        // fogParams[x=fogNearStart,y=fogNearEnd], eyePos), uploaded into gFrameData after
        // viewProj. From DistantLand each frame (see RenderProcess::onPresent).
        // Phase 1a/1b appends a 7th float4 (realEye.xyz, isExterior) for the host-owned LIVE
        // distant-land path: realEye = DistantLand::eyePos (absolute camera, since the camera-relative
        // frame zeroes the shipped eyePos), isExterior gates DL feeding. Host reads it into
        // FrameData.lodEye + g_dlExterior.
        // C2 appends an 8th float4 (skyZenith.rgb, _) — the current interpolated zenith sky colour for
        // the host-computed dome gradient (FrameData.skyZenith).
        // A 9th float4 carries [32] = MW's SIMULATION time in seconds (mwBridge->simulationTime()),
        // which drives the UV scroll of UV-animated distant statics (ghostfence). It must be MW's sim
        // clock, not the host's: sim time does not advance in menus, and it is the exact value MGE's
        // DX9 path feeds its `time` shader uniform (distantland.cpp SetFloat(ehTime, simulationTime())),
        // so both renderers scroll in step under an F11 A/B. 36 floats total.
        IN float lighting[36];
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

        // SK1 sky takeover: per-frame sky draw list of SkyDrawWire[] (geomwire.h) — alpha-blended
        // sky shapes (SK1 = the gradient atmosphere dome), drawn FIRST in the host colour pass with
        // depth off so they sit behind the opaque world. Invalid / count 0 ⇒ no Forge sky this frame
        // (Forge sky toggle off → MW's own sky shows). Grouped with the other per-frame lists.
        IN VecId skyList;
        IN std::uint32_t skyCount;
        IN std::uint32_t skyBytes;

        // F12 debug view: 0 = normal, 1 = depth (world-distance grayscale), 2 = scatter.
        // Appended after the light-list fields so existing field offsets are unchanged.
        IN std::uint32_t debugMode;

        // Forge water takeover (WT1): per-frame water surface params (NO geometry — the host
        // generates the geo-clipmap mesh itself). waterEnabled gates the host water pass (cell has water).
        //   [0] waterLevel (absolute world Z)      [6] nearViewRange
        //   [1] windFactor                         [7] underwater (0/1)
        //   [2] shoreDepthBias                     [8..10] camFwd.xyz (world view forward)
        //   [3..5] depthBaseColor.rgb              [11] MW GameHour (unified water fog's ToD density)
        //   [12] rainParticles   [13] snowParticles — R1 impulse-ripple rate. RAW engine counters,
        //        deliberately not a 0..1 density: the host owns the mapping so its reference count
        //        can be dialled in the dev panel against real weather, which is the only place the
        //        number can honestly be found ([[feedback_prior_art_constants_dont_transfer]]).
        // ⚠ GROWING THIS ARRAY MOVES EVERY FIELD BELOW IT. There is no protocol version or size
        // assert on this struct, so a client/host pair built from different revisions of it will
        // silently read garbage from devMouseX down. mgecore.dll and mgeHost64.exe ship together.
        IN float waterParams[14];
        IN std::uint32_t waterEnabled;

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
        IN std::uint32_t devDistLightsToggle; // one-shot (numpad- edge): host flips baked distant-light loop (perf A/B)
        // One-shot (numpad0 edge): capture N whole HOST frames with RenderDoc, bracketed host-side.
        // The host never Presents — the client does, a frame or two later — so RenderDoc's own
        // hotkey can never land on the frame being looked at. This arms it from the game instead.
        IN std::uint32_t devGpuCapture;

        // AT1 sorted-alpha takeover: per-frame alpha draw list of AlphaDrawWire[] (geomwire.h) —
        // the scene-1 blended world shapes, CLIENT-sorted back-to-front, drawn by the host after
        // water (depth-tested GEQUAL, no write). Invalid / count 0 ⇒ no Forge alpha this frame.
        // Appended after the dev-input fields so every existing field offset is unchanged.
        IN VecId alphaList;
        IN std::uint32_t alphaCount;
        IN std::uint32_t alphaBytes;

        // AT3 captured-alpha: a 1-chunk GeomChunk vec holding [captured verts][captured indices]
        // (indices at capturedVertBytes). Referenced by AlphaDrawWire items with
        // slot == kAlphaSlotCaptured (the host binds pCapAlphaVB/pCapAlphaIB). Invalid / both
        // byte counts 0 ⇒ no captured geometry this frame. Appended after the alpha fields so
        // every existing offset is unchanged.
        IN VecId capturedAlpha;
        IN std::uint32_t capturedVertBytes;
        IN std::uint32_t capturedIdxBytes;

        // FP1a first-person takeover: the arms/weapon draw lists MW renders in its own
        // post-z-clear scene, shipped with the ARM camera's own viewProj (same D3DXMATRIX
        // bytes + camera-relative convention as viewProj above; the host applies the same
        // reverse-Z + half-pixel fixups). fpDrawList = DrawItemWire[] (rigid parts),
        // fpSkinnedList = [SkinnedDrawWire][palette]* — both the exact main-pass wire
        // formats, drawn by the host's dedicated FP pass (fresh depth clear, world
        // screen-space AO/shadow masks neutralized). fpAlphaList = AlphaDrawWire[]
        // reserved for FP1c (torch flame / enchant glow); count 0 until then.
        // fpEnabled == 0 ⇒ no FP pass this frame (all lists ignored). Appended after the
        // captured-alpha fields so every existing IN field offset is unchanged.
        IN float fpViewProj[16];
        IN VecId fpDrawList;
        IN std::uint32_t fpDrawCount;
        IN std::uint32_t fpDrawBytes;
        IN VecId fpSkinnedList;
        IN std::uint32_t fpSkinnedCount;
        IN std::uint32_t fpSkinnedBytes;
        IN VecId fpAlphaList;
        IN std::uint32_t fpAlphaCount;
        IN std::uint32_t fpAlphaBytes;
        IN std::uint32_t fpEnabled;

        // Frame-ahead observability (host Stats panel): the client's own last-frame timings,
        // forwarded from DevInput each kickoff. Purely informational. Appended after the FP
        // fields so every existing IN field offset is unchanged.
        IN std::uint32_t devFrameAhead;
        IN float devClientWaitMs;
        IN float devClientDtMs;
        IN float devClientMwStartMs;

        // Live render-scale (supersampling): the CURRENT internal render resolution the host
        // should draw this frame, always <= the allocation size passed at RenderInit (ceiling
        // scale x backbuffer). The host viewports/dispatches to renderWidth x renderHeight while
        // all size-dependent RTs stay allocated at the ceiling — so a scale change is a per-frame
        // viewport move, no reallocation. 0 ⇒ render at the full allocation size (default 1x path
        // before any scale is set). Appended after the dev fields so every existing IN offset is
        // unchanged. See tasks/forge-seam-resize.md (render-scale, per-frame viewport).
        IN std::uint32_t renderWidth;
        IN std::uint32_t renderHeight;

        // Statics near/far handover: MW's ACTIVE exterior cell set + how far its own cull reaches,
        // so the host can tell which distant statics the NEAR path is already drawing at full
        // detail (drawing both is the handover z-fight). nearCellX/Y = DataHandler centralGridX/Y;
        // nearCellMask bit (dy+1)*3 + (dx+1) is set for each LOADED neighbour (engine residency
        // table, exteriorCellData[9]); nearCellReach = MW's view distance this frame, the radius
        // its bounding-sphere cull reaches. The CENTRE bit doubles as the valid flag: clear ⇒ the
        // host falls back to its old fixed near-cut distance. Appended after the render-scale
        // fields so every existing IN offset is unchanged.
        IN std::int32_t  nearCellX;
        IN std::int32_t  nearCellY;
        IN std::uint32_t nearCellMask;
        IN float         nearCellReach;

        // Tier 1 (tasks/forge-host-gpu-lane.md): 1 ⇒ the client HAS imported the host's shared frame
        // fence as a Vulkan timeline semaphore and will wait frameFenceValue before touching the
        // shared RT, so the host may return from renderScene without waiting its own fence and let
        // frame N's GPU work overlap the reply, the RT copy and MW's frame.
        //
        // 0 ⇒ the client has NO sync object (no shared fence on this device, or the import failed on
        // this DXVK build) and cannot make the copy safe. The host must then keep its old
        // end-of-frame fence wait, which restores the "reply implies GPU-complete" contract. This
        // field is the ONLY thing standing between such a machine and a torn frame every frame, so
        // it is fail-SAFE by construction: anything but an explicit 1 means "host must block".
        // Appended after the near-cell fields so every existing IN offset is unchanged.
        IN std::uint32_t clientSyncsOnFence;

        // (eyeNow - bakeEye): the mode-3 PARK delta the sky payload was pre-cancelled by, in world
        // units. Zero on the serial paths, where bakeEye == eyePos.
        //
        // ⚠ THE HOST CANNOT DERIVE THIS. `lighting[24..26]` (-> FrameData.lodEye) is deliberately the
        // BAKE eye, because it has to match the relative space the payload's positions live in; the
        // fire-time eye only ever existed folded into the restamped viewProj's translation row, which
        // is not separable without inverting the rotation. So it has to ride the wire.
        //
        // Needed because the REFLECT pass mirrors the sky about "z = 0 camera-relative", and that
        // plane is bake-relative like everything else in the payload — i.e. it sits at the BAKE
        // camera's height. The sky pre-cancel makes the payload camera-attached in the MAIN view, and
        // the mirror then re-introduces the delta DOUBLED (a reflection doubles any plane offset).
        // Whenever the camera's height changes between bake and fire — walking a slope, stairs, a
        // jump — the reflected sky steps by 2*dz. Appended after clientSyncsOnFence so every existing
        // IN offset is unchanged.
        IN float skyParkEyeDelta[4];

        // R2 ACTOR RIPPLES. MW keeps a 75-slot pool of ripple decals and switches one on for every
        // actor moving in water — player, NPC and creature alike, with no owner field and no way to
        // ask for one. Measured 2026-08-12: one impulse per ~14.2 units of travel (p25 12.4, median
        // 14.2, p75 14.9 over 360 births), a DISTANCE rule rather than a timer, out to 7249 units
        // from the player. The host superposes a ripplePacket per entry; a moving actor therefore
        // lays down an overlapping train of rings, which IS the wake — the same closed form R1
        // already uses, with no second wave model.
        //
        // ⚠ Liveness is the decal's APP_CULLED flag, NOT the `isActive` byte MWSE names
        // (see MWBridge::getRippleState). That byte reads 0 always.
        //
        // Culled and capped CLIENT-side: the pool is global and reaches far past anything that
        // resolves on screen, so shipping all 75 would spend wire and per-pixel loop iterations on
        // ripples smaller than a texel. Entries are the nearest kMaxActorRipples to the eye.
        //
        // Appended at the very end of the IN block so every existing offset is unchanged — the same
        // rule skyParkEyeDelta above followed, and the reason waterParams could NOT carry this.
        IN std::uint32_t actorRippleCount;
        IN float actorRipples[kMaxActorRipples * 4];   // xy = world XY, z = age 0..1, w = MW scale

        // One-shot (numpad1 edge): dump the host's LINEAR scene target to hdrdump/mge_NNNN.exr plus
        // the composited BGRA8 frame to mge_NNNN.tga, both in the install dir. Follows devGpuCapture
        // exactly — same latch, same one-frame arm — because the target it reads (pMSAAColor, fp16,
        // pre-exposure and pre-curve) exists only inside the host and every other way out of the
        // process goes through resolve.frag, which exposes, tone-maps, encodes and dithers into 8
        // bits. An EXR of the raw radiance is what makes MW's sky/sun/interior levels comparable
        // against real HDRIs instead of eyeballed. Appended at the very end of the IN block so every
        // existing offset is unchanged — the same rule actorRipples above followed.
        IN std::uint32_t devDumpHdr;

        OUT std::uint32_t bytesWritten;
        OUT double renderMs;             // host-side render+readback time

        // Host frame CPU/GPU split, forwarded every frame so the client can PLOT it in Tracy
        // (tasks/forge-host-gpu-lane.md, Tier 1). Definition + the full rationale live in
        // ipc/hostframetimings.h, which the x64 host includes WITHOUT this header's d3d9 baggage.
        // Appended at the end so every existing field offset is unchanged — but the struct is
        // shared by LAYOUT across x86/x64, so rebuild and deploy both binaries together.
        OUT HostFrameTimings hostTimings;

        // Tier 1 (tasks/forge-host-gpu-lane.md): the value the host signalled on the SHARED frame
        // fence for THIS frame's submit. The host no longer blocks on its own fence, so this reply
        // arrives while the GPU may still be drawing — and this number is what makes that safe: the
        // client waits it on the imported timeline semaphore before its RT copy reads the shared RT.
        // 0 ⇒ no shared fence on this host (or the client failed to import it); the client then has
        // no sync object and must not overlap. Appended at the end so every existing OUT offset is
        // unchanged; uint64 because a D3D12 fence value is 64-bit and the wire is layout-shared.
        OUT std::uint64_t frameFenceValue;
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