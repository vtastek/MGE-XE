
#include "distantland.h"
#include "distantshader.h"
#include "drawstats.h"
#include "configuration.h"
#include "msocclient.h"
#include "mwbridge.h"
#include "renderprocess.h"
#include "phasetimers.h"
#include "proxydx/d3d8header.h"
#include "proxydx/devicelock.h"
#include "support/log.h"
#include "statusoverlay.h"
#include "terrain_horizon_occluder.h"
#include "mge_tracy.h"
#include "imgui.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
// Horizon-curtain workspace — kept for shutdownHorizonWorkspace; no longer populated.
thc_horizon_t           g_horizon{};
thc_simplify_workspace_t g_horizonWs{};
bool                    g_horizonInitialized = false;
} // namespace

// Numpad8/Numpad2: raise/lower MSOC low-static cutoff height (steps of 256 units).
static float g_msocCutoffHeight = 2500.0f;

// Contiguous, reused snapshot of the distant-statics visible set.
// applyMSOCToDistantStatics materializes the windowed IPC view into this
// buffer ONCE per frame; every subsequent pass (partition + survivor gather)
// reads it instead of re-traversing the remapping window — a single windowed
// walk, the rest cache-friendly sequential access. DistantLand::
// visDistantSurvivors holds const RenderMesh* into this stable storage, so the
// survivor set is just a pointer subset (no second value copy). Reused across
// frames; ~12k * 156B ~= 1.9MB, trivial for the process heap.
static std::vector<RenderMesh> g_meshValues;

// Basin (watershed) pre-cull state. g_basinOccluded mirrors msocOccluded:
// per-static, 1 = terrain-occluded by the watershed spill-height surface,
// indexed in lockstep with g_meshValues. Filled by applyBasinCullToDistant-
// Statics; consumed for the Phase-1 comparison diag and (when g_basinCullActive)
// the front filter in the MSOC partition.
static std::vector<std::uint8_t> g_basinOccluded;
// True on frames where MGE shipped a ready mask to the host, so the host ran the
// occlusion cull during its quadtree walk and visDistantShared already holds only
// survivors. When set, applyMSOCToDistantStatics skips its own (now redundant)
// group+sphere cull and just materializes + sorts. Reset each frame in the
// kickoff. Same-build deploy guarantees the host loads any mask MGE ships ready.
static bool g_hostCullActiveThisFrame = false;
// Phase 1 (default): compute the basin verdict + comparison diag but do not
// cull with it (proves conservativeness before any geometry is dropped).
static bool g_basinEnabled    = true;
// Basin front filter: drop basin-hidden statics before MSOC's grouping/sphere
// passes and the depth replay. DISABLED — the watershed only reached ~1% beyond
// MSOC (basinFront ~69 of ~6600 survivors), not worth the pass. The terrain-box
// occluders (contributeTerrainBoxOccluders) now carry the terrain-occlusion role
// through MSOC instead. Kept as a flag so the cull site reads the same way.
static bool g_basinCullActive = false;
// Basin debug visualization (watershed surface + culled-static boxes), Numpad6.
// Separate on/off from culling because the extra draws add real draw calls.
static bool g_drawBasinDebug = false;
// Lock the eye fed into the basin flood + verdict (Numpad0) so a free camera can
// fly into a basin to inspect it without the watershed recomputing/shifting.
static bool        g_basinLockEye   = false;
static D3DXVECTOR4 g_basinLockedEye = {0, 0, 0, 0};

// Basin watershed surface: per-basinGrid-cell minimax spill height required to
// reach the cell from the camera cell. Stored as a DENSE camera-relative grid
// (not a hash map) — at fine resolutions the domain is ~10^5 cells, and an
// unordered_map rebuild + per-static hash lookups cost >100ms. The dense array
// is indexed by (cellX - originX, cellY - originY); the verdict bounds-checks
// and reads it in O(1). Cached; rebuilt only when the camera cell changes, the
// grid resolution changes, or the barrier map is rebuilt. Declared here (ahead of
// renderBasinDebug) so the visualizer can read it.
static std::vector<float> g_basinReqArr;   // dense req grid, g_basinReqDim^2
static int    g_basinReqOriginX = 0;       // world cell of g_basinReqArr[0]
static int    g_basinReqOriginY = 0;
static int    g_basinReqDim     = 0;
static int    g_basinCamCellX  = INT_MIN;
static int    g_basinCamCellY  = INT_MIN;
static float  g_basinReqGrid   = 0.0f;
static size_t g_basinReqSrcSize = (size_t)-1;
static bool   g_basinReqValid  = false;

// Tunable basin barrier grid (world units). The watershed flood + the per-cell
// "full-cell wall" test run at this resolution, decoupled from kCellSize. Finer
// = the wall condition is achievable for real ridges (more conservative culling)
// at higher flood cost + more frequent recompute. Numpad / halves, Numpad *
// doubles, clamped to [kBasinGridMin, kBasinGridMax]. The conservativeness proof
// holds at any resolution: a cell whose entire terrain min is above the sight-
// line is a real wall; a sub-cell gap lowers that min and opens free passage.
static constexpr float kBasinGridMin = 512.0f;
static constexpr float kBasinGridMax = 8192.0f;
static float g_basinGrid = 1024.0f;

namespace {
struct MSOCBasinDebugBox {
    float minX, maxX, minY, maxY, minZ, maxZ;
    MSOCClient::TestResult verdict;
};

static bool g_drawMSOCBasinBounds = false;
static std::vector<MSOCBasinDebugBox> g_msocBasinDebugBoxes;

// Water-reflection proxy visualizer (Numpad5). isReflectionWaterVisible() fills
// these — one box per tested water cell slab, coloured by MSOC verdict (green
// visible / red occluded / blue view-culled). Lets us see why the gate decides
// water is/isn't visible. (Numpad5 also drives the mask dump in runMSOCDebugTail,
// but only when LogDistantPipeline is on; with it off, this is the sole reader.)
static std::vector<MSOCBasinDebugBox> g_waterProxyDebugBoxes;

struct MSOCLineVertex {
    float x, y, z;
    DWORD color;
};

constexpr DWORD fvfMSOCLine = D3DFVF_XYZ | D3DFVF_DIFFUSE;

// Curtain debug overlay (Numpad3 cycle, 3rd state "ON + curtain debug").
// contributeDistantLandOccluders stashes the pre-fixup curtain triangles here
// (NDC x, NDC y, view-depth w) when capture is on; renderCurtainDebug draws them
// as a screen-space overlay coloured by depth (near=red -> far=blue), then clears
// the buffer. An empty buffer draws nothing, so the OFF and plain-ON cycle states
// naturally show no overlay without any extra flag plumbing.
struct CurtainDbgVert { float ndcX, ndcY, depthW; };
static std::vector<CurtainDbgVert> g_curtainDbg;

// Terrain box occluders (Numpad3 cycle, "boxes" states). Instead of the screen-
// space horizon curtain, we voxelize the coarse terrain into per-cell boxes whose
// TOP = the cell's minimum height (so each box is buried inside the real terrain
// and never occludes a static sitting on the surface above it) and feed them to
// MSOC as world-space occluders. Real 3D geometry → no per-vertex silhouette
// aliasing, no single-wall pancake; the existing sphere/OBB tests cull against them
// unchanged. Box footprint reuses the basin grid (g_basinMinH), extent is a radius
// in game cells around the camera.
static float g_boxRadiusCells = 4.0f;   // voxelization radius, in game cells
struct BoxDbgAABB { float x0, y0, x1, y1, zTop, zBot; };
static std::vector<BoxDbgAABB> g_boxDbg;

// S3: the reflection-statics pipeline diagnostics, the fine water-silhouette mask
// (256x128 NDC bitmask + its rasteriser) and the per-tile visibility hysteresis sets
// lived here. All of it existed to cull what the water reflection would draw.

// Pre-transformed (pixel-space) coloured vertex for the curtain overlay. The
// curtain is a screen-space construct, so XYZRHW is the honest representation —
// it lands exactly where the wall sits on this frame's image.
struct CurtainScreenVertex { float x, y, z, rhw; DWORD color; };
constexpr DWORD fvfCurtainScreen = D3DFVF_XYZRHW | D3DFVF_DIFFUSE;

DWORD colorForMSOCVerdict(MSOCClient::TestResult verdict) {
    switch (verdict) {
    case MSOCClient::ResultOccluded:   return D3DCOLOR_XRGB(255, 64, 64);
    case MSOCClient::ResultViewCulled: return D3DCOLOR_XRGB(64, 128, 255);
    case MSOCClient::ResultNotReady:   return D3DCOLOR_XRGB(255, 224, 64);
    default:                           return D3DCOLOR_XRGB(64, 255, 96);
    }
}

float distanceSqToAABB2D(float px, float py, float minX, float maxX, float minY, float maxY) {
    const float dx = (px < minX) ? (minX - px) : (px > maxX) ? (px - maxX) : 0.0f;
    const float dy = (py < minY) ? (minY - py) : (py > maxY) ? (py - maxY) : 0.0f;
    return dx * dx + dy * dy;
}

// ---------------------------------------------------------------------------
// Dedicated MSOC cull worker.
//
// applyMSOCToDistantStatics is pure compute over an already-ready IPC result
// (visDistantShared). Tracy showed it sitting at ~892us on the main critical
// path inside renderDepth, after the engine's 1.46ms sky pass — yet its input
// (the statics visible set) was ready during the sky window (finish:wait was
// only 33us). Running the verdict on a dedicated worker lets it overlap the
// sky pass; cullDistantStatics_finish then collapses to a ~0 join.
//
// Synchronization mirrors scenegraph.cpp's worker idiom: one mutex + condvar
// + bool flags. Three sync points (the old single maskDone fence is split so
// the early statics verdict isn't coupled to the late reflection cull):
//   - channelDrained: the worker has finished its tryWaitForCompletion drain
//     of the (now reflection-folded) statics RPC, so the single-channel
//     ipcClient is free for the main thread's land/grass/shadow/water RPCs.
//     renderStage0 waits on this at entry before any main-thread ipcClient touch.
//   - staticsDone: the verdict core has finished writing msocOccluded.
//     cullDistantStatics_finish waits on this in place of wait+applyMSOC. Signalled
//     right after the verdict, BEFORE the reflection gate/cull, so the statics
//     consumer (renderDepth) no longer waits on the reflection work.
//   - reflDone: the reflection gate + survivor cull have finished. Joined just
//     before renderWaterReflection (~10 passes later), where reflVisible /
//     reflectionSurvivors are consumed.
std::thread             g_cullThread;
std::mutex              g_cullMtx;
std::condition_variable g_cullCv;
bool g_cullPending        = false; // a finish job has been signalled
bool g_cullChannelDrained = false; // statics RPC drained off ipcClient
bool g_cullStaticsDone    = false; // verdict core finished; msocOccluded stable
bool g_cullReflDone       = false; // reflection gate + survivor cull finished
bool g_cullStop           = false; // shutdown request
bool g_cullStarted        = false; // lazy-init flag

// View-projection (screen space, mwView*mwProj) captured on the main thread at
// signalCullFinish — frame-stable by then (GetTransform ran in frameSetupEarly).
// Read by the worker's terrain-box occluder pass so it never touches the live
// mwView/mwProj globals off the main thread.
static D3DXMATRIX g_boxWorkerViewProj;

// Per-frame summary stashed by the verdict core (worker) and read by the
// main-thread debug tail's 60-frame log block. Stable once staticsDone joins.
struct MSOCCullDiag {
    bool     valid = false;
    unsigned setSize = 0;
    unsigned nSphere = 0;
    unsigned groupCount = 0;
    int      groupsOccluded = 0;
    int      lowCulled = 0;
    int      sphereCulled = 0;
    int      basinFrontCulled = 0; // statics removed by the Phase-2 basin filter
                                   // (not counted in low/sphere — they continue
                                   // before both passes)
    // Basin pre-cull comparison vs MSOC (Phase 1 diagnostic).
    float    basinGrid         = 0; // barrier grid resolution (world units)
    unsigned basinReqCells     = 0; // cells in the watershed surface
    int      basinCull         = 0; // statics the basin verdict would cull
    int      basinCull_MSOCkeep = 0; // basin beyond MSOC (headroom OR false-cull
                                     // — the cell-min flood is conservative, so
                                     // these should be truly hidden; verify by
                                     // looking for pop-in with Phase 2 active)
    int      basinCull_MSOCcull = 0; // agreement (basin + MSOC both cull)
    int      basinKeep_MSOCcull = 0; // MSOC headroom the basin can't reach
};
MSOCCullDiag g_msocCullDiag;
} // namespace

// True when this frame's verdict pass was dispatched to the worker (set by
// signalCullFinish, consumed by the wait/finish helpers). Main-thread only.
static bool s_cullOnWorker = false;



// S4: renderSky() lived here — MGE's atmosphere-scattering sky, replayed from the
// recordSky list captured in inspectIndexedPrimitive. The Forge host owns the sky
// (SK3/SK4), so both the pass and the record list are gone.

// S4: renderDistantLand() — the DX9 LOD-landscape colour pass — is gone; the Forge host
// draws LOD land itself. renderDistantLandZ (the depth twin) survives to S5.

void DistantLand::contributeDistantLandOccluders(bool captureDebug) {
    MGE_ZoneScopedN("contributeOccluders");
    MGE_SCOPED_TIMER("contributeDistantLandOccluders");

    // Reset the debug-overlay capture every call; renderCurtainDebug consumes and
    // clears it, but clear here too so a frame that bails early leaves no stale wall.
    g_curtainDbg.clear();

    static int diagGuardFrame = 0;
    // Diagnostic loglines fire every 60th call AND only when the user
    // opted into pipeline logging via mge.ini [Misc] "Log Distant Pipeline".
    const bool emitDiag = ((diagGuardFrame++ % 60) == 0) && Configuration.LogDistantPipeline;

    if (!Configuration.UseOcclusionCulling) {
        if (emitDiag) LOG::logline("-- MSOC occluder: horizon skipped — UseOcclusionCulling=false");
        return;
    }
    if (!MSOCClient::isAvailable()) {
        if (emitDiag) LOG::logline("-- MSOC occluder: horizon skipped — MSOCClient not available");
        return;
    }
    if (landMeshes.empty()) {
        if (emitDiag) LOG::logline("-- MSOC occluder: horizon skipped — landMeshes map is empty");
        return;
    }

    // Horizon state persists across frames (reset at the top of each
    // contribution). One allocation at first-call, reused thereafter.
    constexpr int   kHorizonResolution = 512;
    constexpr int   kMaxSamples        = 60;
    constexpr float kEpsH              = 0.01f;  // ~1% of clip-y range per sample
    constexpr float kEpsD              = 1.0e30f; // effectively disable depth-driven splits
    constexpr int   kTileAlign         = 16;     // 32-px HiZ tiles on a 512-col horizon
    constexpr float kNdcYBottom        = -1.1f;  // curtain bottom slightly off-screen

    // Conservatism knobs — both push the curtain in the "under-cull"
    // direction. Statics close to the silhouette or close to terrain's
    // far depth stay tested instead of culled.
    //   - y safety: lower the curtain top by this clip-y amount (NDC).
    //     0.04 ≈ 5 pixels on a 256-row mask — enough to cover projection
    //     noise + ROAM tessellation edge approximations.
    //   - w safety: multiply max clip-w (depth) by this factor so the
    //     curtain rasterizes "deeper" than measured, preventing culls
    //     when a static happens to share terrain's far depth.
    constexpr float kYSafetyMargin = 0.04f;
    constexpr float kWSafetyFactor = 1.10f;

    if (!g_horizonInitialized) {
        if (thc_horizon_init(&g_horizon, kHorizonResolution, nullptr, nullptr) != 0) {
            LOG::logline("-- MSOC occluder: thc_horizon_init failed; disabling horizon path");
            return;
        }
        g_horizonWs.size_bytes = thc_simplify_workspace_required_bytes(kHorizonResolution, kMaxSamples);
        g_horizonWs.memory = malloc(g_horizonWs.size_bytes);
        if (!g_horizonWs.memory) {
            LOG::logline("-- MSOC occluder: simplify workspace alloc failed; disabling horizon path");
            return;
        }
        g_horizonInitialized = true;
    }
    thc_horizon_reset(&g_horizon);

    // World-to-clip matrix (row-major D3DX convention, row-vector * M).
    // clip = world * view * proj; we pre-multiply once per frame.
    const D3DXMATRIX viewProj = mwView * mwProj;

    // Per-vertex horizon build (replaces per-triangle for cost).
    //
    // Each tile's projected verts are dropped into a per-tile per-column
    // scratch (max-y, max-w per column the tile spans). After the scan,
    // submit one horizon update per active column. Per-vertex misses
    // triangle-interior silhouette contributions (a vert mid-edge of a
    // peak triangle isn't sampled), but that's strictly UNDER-cull —
    // safer direction. ROAM puts vertex density on silhouettes, so the
    // visual result is essentially the same with a fraction of the work.
    int visibleLandTiles = 0;
    int lookupMisses = 0;
    int verticesFed = 0;
    int columnsUpdated = 0;
    int columnsPruned = 0;

    // Per-tile column scratch. Sized to the horizon resolution so any
    // tile, however wide, fits without resizing. Reset per tile by only
    // touching the columns this tile contributes to (tracked in
    // tileTouchedCols).
    static std::vector<float>    tileMaxY;       // [resolution]
    static std::vector<float>    tileMaxW;       // [resolution]
    static std::vector<uint16_t> tileTouchedCols; // sparse list of cols updated this tile
    if ((int)tileMaxY.size() < kHorizonResolution) {
        tileMaxY.assign(kHorizonResolution, -1e30f);
        tileMaxW.assign(kHorizonResolution, -1e30f);
        tileTouchedCols.reserve(kHorizonResolution);
    }

    const float halfSpan = 0.5f * (float)(kHorizonResolution - 1);

    auto feedTile = [&](const RenderMesh& m) {
        ++visibleLandTiles;
        auto it = landMeshes.find(m.vBuffer);
        if (it == landMeshes.end()) {
            ++lookupMisses;
            return;
        }
        const LandMeshCache& mesh = it->second;
        if (mesh.positions.empty()) return;

        tileTouchedCols.clear();

        // Project each vertex; bin into per-column max-y / max-w for
        // this tile.
        for (const D3DXVECTOR3& p : mesh.positions) {
            // D3DX row-major: clip = (p.x, p.y, p.z, 1) * viewProj
            const float cx = p.x * viewProj._11 + p.y * viewProj._21 + p.z * viewProj._31 + viewProj._41;
            const float cy = p.x * viewProj._12 + p.y * viewProj._22 + p.z * viewProj._32 + viewProj._42;
            const float cw = p.x * viewProj._14 + p.y * viewProj._24 + p.z * viewProj._34 + viewProj._44;
            if (cw <= 1.0e-4f) continue;       // behind camera

            const float inv = 1.0f / cw;
            const float ndcX = cx * inv;
            const float ndcY = cy * inv;

            if (ndcX < -1.0f || ndcX > 1.0f) continue;  // off-screen X
            if (ndcY < -1.0f) continue;                  // below screen
            const float clampedY = (ndcY > 1.0f) ? 1.0f : ndcY;

            int col = (int)floorf((ndcX + 1.0f) * halfSpan);
            if (col < 0) col = 0;
            if (col >= kHorizonResolution) col = kHorizonResolution - 1;

            ++verticesFed;
            // First-touch tracking: empty marker is -1e30, anything
            // above counts as "touched this tile."
            if (tileMaxY[col] < -1e29f) {
                tileTouchedCols.push_back((uint16_t)col);
                tileMaxY[col] = clampedY;
                tileMaxW[col] = cw;
            } else {
                if (clampedY > tileMaxY[col]) tileMaxY[col] = clampedY;
                if (cw > tileMaxW[col])       tileMaxW[col] = cw;
            }
        }

        // Submit one horizon update per touched column. Reset scratch
        // back to sentinel as we go so the next tile starts clean.
        for (uint16_t col : tileTouchedCols) {
            thc_node_projection_t proj;
            proj.c0        = (int)col;
            proj.c1        = (int)col;
            proj.y_upper   = tileMaxY[col];
            proj.far_depth = tileMaxW[col] * kWSafetyFactor;

            if (thc_horizon_test_and_update(&g_horizon, &proj)) {
                ++columnsPruned;
            } else {
                ++columnsUpdated;
            }
            tileMaxY[col] = -1e30f;   // reset for next tile
            tileMaxW[col] = -1e30f;
        }
    };

    if (Configuration.UseSharedMemory) {
        visLandShared.Reset();
        while (!visLandShared.AtEnd()) feedTile(visLandShared.Next());
    } else {
        visLand.Reset();
        while (!visLand.AtEnd()) feedTile(visLand.Next());
    }

    if (verticesFed == 0) {
        if (emitDiag) {
            LOG::logline(
                "-- MSOC occluder: horizon empty — visibleTiles=%d lookupMisses=%d",
                visibleLandTiles, lookupMisses);
        }
        return;
    }

    // Simplify to ~60 adaptive samples, snapping to 16-column (~32 pixel)
    // alignment so each curtain quad spans a whole HiZ tile column.
    static thc_sample_t samples[kMaxSamples];
    const int nSamples = thc_horizon_simplify(
        &g_horizon, &g_horizonWs, samples, kMaxSamples, kEpsH, kEpsD, kTileAlign);
    if (nSamples < 2) return;

    // Apply y safety margin: pull the silhouette top down by a small
    // clip-y amount so the curtain hangs strictly below the measured
    // horizon. Statics whose tops are within `kYSafetyMargin` of the
    // silhouette stay tested instead of culled. Skip the sentinel
    // (THC_Y_BELOW) — those columns saw no terrain, no margin needed.
    for (int i = 0; i < nSamples; ++i) {
        if (samples[i].h > -1e29f) {
            samples[i].h -= kYSafetyMargin;
        }
    }

    // Emit curtain triangles in the reference's {x, y, z_depth, w=1} layout.
    // Capacity: 6 verts per segment × (nSamples - 1) segments.
    static std::vector<thc_curtain_vertex_t> curtainVerts;
    curtainVerts.resize(6 * (nSamples - 1));
    const int triCount = thc_emit_curtains(
        samples, nSamples, kNdcYBottom,
        curtainVerts.data(), (int)curtainVerts.size());
    if (triCount <= 0) return;

    // Layout fixup for MOC: the reference emits NDC x,y with w=1 and the
    // curtain depth in z. MOC with matrix=nullptr expects clip-space
    // (x, y, w) where x/w = NDC_x internally, and depth = 1/w. Convert:
    //   stored_x = ndc_x * depth_w
    //   stored_y = y_top * depth_w
    //   stored_w = depth_w
    // (z slot ignored by MOC with the default stride=16, offW=12 layout.)
    const int vtxCount = triCount * 3;

    // Stash the pre-fixup curtain (NDC x, NDC y, view-depth in z) for the in-world
    // overlay before we overwrite the layout for MOC. Captured only when the
    // Numpad3 cycle is in its debug state, so the cost is zero in normal use.
    if (captureDebug) {
        g_curtainDbg.reserve(vtxCount);
        for (int i = 0; i < vtxCount; ++i) {
            const thc_curtain_vertex_t& v = curtainVerts[i];
            g_curtainDbg.push_back({ v.x, v.y, v.z });
        }
    }

    for (int i = 0; i < vtxCount; ++i) {
        thc_curtain_vertex_t& v = curtainVerts[i];
        const float d = v.z;
        v.x *= d;
        v.y *= d;
        v.w  = d;
    }

    // Build index buffer once (straight 0..vtxCount-1 — emit_curtains
    // writes tris as consecutive 3-vert groups, no index sharing).
    static std::vector<uint32_t> curtainIdx;
    curtainIdx.resize(vtxCount);
    for (int i = 0; i < vtxCount; ++i) curtainIdx[i] = (uint32_t)i;

    const bool ok = MSOCClient::addPreTransformedOccluder(
        reinterpret_cast<const float*>(curtainVerts.data()), vtxCount,
        /*stride*/ 16, /*offY*/ 4, /*offW*/ 12,
        curtainIdx.data(), triCount);

    static int diagFrameCounter = 0;
    if (Configuration.LogDistantPipeline && (diagFrameCounter++ % 60) == 0) {
        LOG::logline("-- MSOC occluder: horizon %s — tiles=%d verts=%d colsUpdated=%d colsPruned=%d samples=%d curtainTris=%d",
                     ok ? "submitted" : "REJECTED",
                     visibleLandTiles, verticesFed,
                     columnsUpdated, columnsPruned,
                     nSamples, triCount);
    }
}

// Free the horizon-curtain workspace allocated lazily in
// contributeDistantLandOccluders. Safe to call when uninitialized.
// Called from DistantLand::release().
void DistantLand::shutdownHorizonWorkspace() {
    if (!g_horizonInitialized) {
        return;
    }
    thc_horizon_free(&g_horizon, nullptr);
    free(g_horizonWs.memory);
    g_horizonWs.memory = nullptr;
    g_horizonWs.size_bytes = 0;
    g_horizonInitialized = false;
}

void DistantLand::renderDistantLandZ() {
    MGE_SCOPED_TIMER("renderDistantLandZ");
    DrawStats::ScopedStage _ds(DrawStats::DepthLand);   // distant-land depth replay
    D3DXMATRIX world;

    D3DXMatrixIdentity(&world);
    effect->SetMatrix(DistantLand::ehWorld, &world);
    effectDepth->CommitChanges();

    // Draw with cached vis set
    device->SetVertexDeclaration(LandDecl);
    if (Configuration.UseSharedMemory) {
        visLandShared.Render(device, SIZEOFLANDVERT);
    } else {
        visLand.Render(device, SIZEOFLANDVERT);
    }
}

// Cross-phase scratch for the non-IPC diagnostic counters. Populated by
// _kickoff (which performs the synchronous quadtree queries when the
// IPC path is disabled) and consumed by _finish's log block.
namespace {
    unsigned g_cullDiagNearCount    = 0;
    unsigned g_cullDiagFarCount     = 0;
    unsigned g_cullDiagVeryFarCount = 0;
}

namespace {
// Worker entry. Parks on the condvar until signalCullFinish wakes it, then:
//   1. drains the (reflection-folded) statics RPC off the single IPC channel
//      and — while still owning the channel — materializes the reflection set,
//      then signals channelDrained (frees the channel for the main thread),
//   2. runs the pure verdict core over the freshly-drained set and signals
//      staticsDone (early fence; statics consumer no longer waits on reflection),
//   3. runs the reflection gate + survivor cull and signals reflDone (late fence,
//      joined just before renderWaterReflection).
void cullWorkerLoop() {
#ifdef TRACY_ENABLE
    tracy::SetThreadName("MGE MSOC cull");
#endif
    while (true) {
        {
            std::unique_lock<std::mutex> lk(g_cullMtx);
            g_cullCv.wait(lk, []{ return g_cullStop || g_cullPending; });
            if (g_cullStop) return;
            g_cullPending = false;
        }

        // Drain the statics RPC — which now also carries the folded reflection
        // query (visExtraShared), so one drain covers both. Guarded
        // (tryWaitForCompletion): an interleaved drain — there shouldn't be one
        // before renderStage0's channel-free gate, but stay defensive — may
        // already have completed it, in which case there's no pending RPC.
        {
            MGE_ZoneScopedN("cullWorker:drain");
            DistantLand::ipcClient.tryWaitForCompletion();
        }

        {
            std::lock_guard<std::mutex> lk(g_cullMtx);
            g_cullChannelDrained = true;
        }
        g_cullCv.notify_all();

        // Pure verdict core. Writes msocOccluded (+ g_msocBasinDebugBoxes /
        // g_msocCullDiag); reads visDistantShared, eyePos, g_msocCutoffHeight,
        // WaterLevel(). All inputs are frame-stable by signal time.
        {
            MGE_ZoneScopedN("cullWorker:verdict");
            DistantLand::applyMSOCToDistantStatics(DistantLand::visDistantShared);
        }

        // Early fence: statics verdict done. Signalled BEFORE the reflection
        // gate/cull AND before the terrain-box pass so cullDistantStatics_finish
        // (renderDepth, ~10 passes before the reflection draw) joins on the verdict
        // alone — it must not wait on the box occluders (those feed the mask two
        // frames out; there is no in-frame consumer of them on the main thread).
        {
            std::lock_guard<std::mutex> lk(g_cullMtx);
            g_cullStaticsDone = true;
        }
        g_cullCv.notify_all();

        // Contribute the terrain-box occluders to MSOC (next mask build, N+2
        // latency). Off the main thread AND past the staticsDone fence, so its
        // ~0.4ms lands in the reflection window's slack (finish:joinReflMask is
        // ~0) instead of on the verdict-join critical path. Uses the frame-stable
        // view-proj captured at signalCullFinish.
        {
            MGE_ZoneScopedN("cullWorker:terrainBoxes");
            DistantLand::contributeTerrainBoxOccluders(
                g_boxWorkerViewProj, DistantLand::boxOccluderDebug);
        }

        // S3: the reflection gate + survivor cull ran here, between the terrain-box pass
        // and the late fence. The fence itself stays -- joinCullWorker and the worker
        // loop still hand off on it -- it just has no reflection work in front of it now.

        // Late fence: reflection cull done.
        {
            std::lock_guard<std::mutex> lk(g_cullMtx);
            g_cullReflDone = true;
        }
        g_cullCv.notify_all();
    }
}

void ensureCullWorker() {
    if (!g_cullStarted) {
        g_cullStarted = true;
        g_cullThread = std::thread(cullWorkerLoop);
        LOG::logline("-- [MSOC cull] worker spawned");
    }
}

// Block until the worker has finished the verdict core (msocOccluded stable).
void waitCullStaticsReady() {
    MGE_ZoneScopedN("finish:joinMask");
    std::unique_lock<std::mutex> lk(g_cullMtx);
    g_cullCv.wait(lk, []{ return g_cullStaticsDone; });
}

// Main-thread debug tail for the MSOC verdict: the per-60-frame cull summary.
// Touches LOG, so it stays off the worker; reads g_msocCullDiag, stable after
// the join. (The Numpad5 mask dump moved to DistantLand::debugDumpMSOCMask,
// called after contributeDistantLandOccluders so the dump captures this frame's
// horizon curtain instead of the pre-curtain mask.)
void runMSOCDebugTail() {
    if (!Configuration.LogDistantPipeline)
        return;

    static int diagFrameCounter = 0;
    if (g_msocCullDiag.valid && (diagFrameCounter++ % 60) == 0) {
        const MSOCCullDiag& d = g_msocCullDiag;
        // total = everything removed from the render/depth set: group + sphere
        // culls PLUS the basin front-culls (which skip both passes). survivors =
        // setSize - total is what actually enters depth replay + render.
        const int totalCulled = d.lowCulled + d.sphereCulled + d.basinFrontCulled;
        const int survivors   = (int)d.setSize - totalCulled;
        LOG::logline(
            "-- MSOC cull: statics=%u  low=%u(groups=%u occ=%d culled=%d)"
            "  sphere=%u(culled=%d)  basinFront=%d  total=%d(%d%%)  survivors=%d",
            d.setSize,
            d.setSize - d.nSphere, d.groupCount,
            d.groupsOccluded, d.lowCulled,
            d.nSphere, d.sphereCulled,
            d.basinFrontCulled,
            totalCulled,
            d.setSize > 0 ? (totalCulled * 100) / d.setSize : 0,
            survivors);
        // Basin runs last on MSOC's survivors, so basinCull is the REAL number of
        // statics MSOC would have drawn that the basin removes — its headroom over
        // MSOC. Reported as a fraction of the pre-basin survivors (what MSOC kept).
        const int msocCulled = d.basinKeep_MSOCcull;          // group + sphere culls
        const int msocKept   = (int)d.setSize - msocCulled;   // survivors before basin
        LOG::logline(
            "-- MSOC basin: grid=%.0f  reqCells=%u  beyondMSOC=%d (%d%% of %d MSOC-survivors, "
            "verify no pop)  MSOCculled=%d  [%s]",
            d.basinGrid,
            d.basinReqCells,
            d.basinCull,
            msocKept > 0 ? (d.basinCull * 100) / msocKept : 0,
            msocKept,
            msocCulled,
            g_basinLockEye ? "eye:LOCK" : "eye:live");
    }
}
} // namespace

// Numpad5: dump the MSOC occlusion mask to a .pfm depth image. Called from
// renderStage0 AFTER contributeDistantLandOccluders so the dump contains this
// frame's freshly-submitted horizon curtain — the pre-curtain dump (where this
// used to live, in cullDistantStatics_finish) could only ever show the previous
// frame's stale curtain. Gated on LogDistantPipeline like the rest of the diag.
void DistantLand::debugDumpMSOCMask() {
    if (!Configuration.LogDistantPipeline)
        return;

    static bool dumpPending = false;
    static int  dumpIndex   = 0;
    if (GetAsyncKeyState(VK_NUMPAD5) & 0x0001)
        dumpPending = true;
    if (dumpPending) {
        char exePath[MAX_PATH] = {};
        GetModuleFileNameA(NULL, exePath, MAX_PATH);
        if (char* lastSlash = strrchr(exePath, '\\'))
            *lastSlash = '\0';
        char fullPath[MAX_PATH];
        std::snprintf(fullPath, sizeof(fullPath), "%s\\msoc_mask_%03d.pfm", exePath, dumpIndex++);
        const bool ok = MSOCClient::dumpMask(fullPath);
        LOG::logline("-- MSOC: Numpad5 mask dump %s %s", ok ? "->" : "FAILED for", fullPath);
        dumpPending = false;
    }
}

// Numpad8/2: raise/lower the MSOC low-static cutoff height (256-unit steps).
// Read on the main thread so g_msocCutoffHeight is final before the verdict
// core (worker or inline) reads it.
// Flow-map re-bake signal (main → cull worker) and the runtime-tunable knobs.
// Declared here so updateMSOCCutoffInput (NUMPAD7/6/3) can write them; consumed by
// buildWaterFlowMap on the worker. Plain writes ordered before the g_flowRebake
// release store, which the worker reads with acquire — benign for debug tuning.
// S3: the flow-map bake knobs and their rebake signal lived here.

void DistantLand::updateMSOCCutoffInput() {
    // In-world overlays are compacted onto one cycling key (numpad +). One press
    // advances which single overlay is active; we derive the per-overlay
    // capture/render bools here. Read on the main thread BEFORE the cull worker
    // dispatch so a press this frame is captured this frame (the worker fills the
    // water-proxy / basin / box debug buffers gated on these bools).
    if (GetAsyncKeyState(VK_ADD) & 0x0001) {
        // S3 dropped states 1 (water proxy boxes) and 5 (reflection frustum) with the
        // water reflection; the surviving overlays keep their historical numbers so
        // muscle memory and every existing note still line up.
        debugOverlayCycle = (debugOverlayCycle + 1) % 5;
        if (debugOverlayCycle == 1) { debugOverlayCycle = 2; }
        static const char* const names[5] = {
            "OFF", "(retired)", "Box occluders", "Basin watershed", "MSOC basin boxes" };
        char msg[80];
        std::snprintf(msg, sizeof(msg), "Overlay [+]: %s", names[debugOverlayCycle]);
        StatusOverlay::setStatus(msg);
    }
    boxOccluderDebug       = (debugOverlayCycle == 2);
    g_drawBasinDebug       = (debugOverlayCycle == 3);
    g_drawMSOCBasinBounds  = (debugOverlayCycle == 4);

    // S3: NUMPAD1 (reflection fine mask), NUMPAD0 (water LOD mesh A/B), NUMPAD9 (flow-map
    // debug cycle), NUMPAD2 (foam panels) and the NUMPAD8/6/3 flow+foam knob tuner were
    // here. Every one of them tuned MGE's water renderer.
    // (Retired debug hot keys: MSOC cutoff height (NUMPAD8/2), basin eye lock
    // (NUMPAD0), basin barrier grid (NUMPAD / and *). Those values are settled;
    // g_msocCutoffHeight / g_basinLockEye / g_basinGrid keep their defaults.
    // NUMPAD8 is now the flow-map knob cycle above.)
}

// S3: DrawFlowFoamPanel (the F10 Water Flow / Foam imgui panel) was here.

// Dispatch the verdict pass to the cull worker. Called from frameSetupEarly
// right after cullDistantStatics_kickoff issues the statics RPC.
void DistantLand::signalCullFinish() {
    ensureCullWorker();
    // Snapshot the screen view-proj for the worker's terrain-box occluder pass.
    // mwView/mwProj were finalized in frameSetupEarly (GetTransform) just above
    // this call, so this is frame-stable; the worker never reads the live globals.
    g_boxWorkerViewProj = mwView * mwProj;
    {
        std::lock_guard<std::mutex> lk(g_cullMtx);
        g_cullChannelDrained = false;
        g_cullStaticsDone    = false;
        g_cullReflDone       = false;
        g_cullPending        = true;
    }
    s_cullOnWorker = true;
    g_cullCv.notify_one();
}

// Block at renderStage0 entry until the worker has drained the statics RPC,
// so no main-thread ipcClient call (land/grass/shadow/water) races the drain
// on the single-channel client. No-op when the worker path is inactive.
void DistantLand::waitCullChannelFree() {
    if (!s_cullOnWorker)
        return;
    MGE_ZoneScopedN("waitCullChannelFree");
    std::unique_lock<std::mutex> lk(g_cullMtx);
    g_cullCv.wait(lk, []{ return g_cullChannelDrained; });
}

// Block until the worker has finished the reflection gate + survivor cull
// (reflVisible / reflectionSurvivors stable). Joined just before
// renderWaterReflection, ~10 passes after the early staticsDone fence — so in
// steady state this collapses to ~0. No-op when the worker path is inactive.
// S3: waitCullReflReady (the late reflection fence) was here. Its only caller was
// renderStage0's reflection update.

// Tear the worker thread down on renderer release. Mirrors the SceneGraph
// worker shutdown; resets state so a later init re-spawns cleanly.
void DistantLand::joinCullWorker() {
    if (!g_cullStarted)
        return;
    {
        std::lock_guard<std::mutex> lk(g_cullMtx);
        g_cullStop = true;
    }
    g_cullCv.notify_all();
    if (g_cullThread.joinable())
        g_cullThread.join();
    g_cullStarted        = false;
    g_cullStop           = false;
    g_cullPending        = false;
    g_cullChannelDrained = false;
    g_cullStaticsDone    = false;
    g_cullReflDone       = false;
    s_cullOnWorker       = false;
}

void DistantLand::cullDistantStatics_kickoff(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    MGE_ZoneScopedN("cullDistantStatics_kickoff");
    MGE_SCOPED_TIMER("cullDistantStatics:kickoff");

    // Default this frame to the inline path. signalCullFinish (worker path)
    // flips it back to true right after this kickoff. Resetting here — rather
    // than at the end of _finish — keeps the flag fresh even on frames where
    // _finish is skipped (e.g. underwater: the statics depth/color passes and
    // their join don't run, but the worker was still dispatched). The kickoff
    // always runs before the renderDepth channel-free gate and before _finish,
    // so both read a current-frame value.
    s_cullOnWorker = false;
    g_hostCullActiveThisFrame = false;   // set true below iff a ready mask ships

    D3DXMATRIX ds_proj = *proj, ds_viewproj;
    D3DXVECTOR4 viewsphere(eyePos.x, eyePos.y, eyePos.z, 0);
    float zn = nearViewRange - 768.0f, zf = zn;
    float cullDist = fogEnd;

    if (Configuration.UseSharedMemory) {
        visDistantShared.RemoveAll();
    } else {
        visDistant.RemoveAll();
    }

    g_cullDiagNearCount = g_cullDiagFarCount = g_cullDiagVeryFarCount = 0;

    const bool useIpc = Configuration.UseSharedMemory;

    // Pack the per-range parameters once; we use them either to populate
    // the batched RPC (IPC path) or to drive the local quadtree queries
    // sequentially (non-IPC path).
    struct Range {
        DWORD flag;
        ViewFrustum frustum;
        D3DXVECTOR4 sphere;
        bool active;
    };
    Range ranges[3] = {
        { VIS_NEAR,     ViewFrustum(&ds_viewproj), viewsphere, false },
        { VIS_FAR,      ViewFrustum(&ds_viewproj), viewsphere, false },
        { VIS_VERY_FAR, ViewFrustum(&ds_viewproj), viewsphere, false },
    };
    const float rangeEnds[3] = {
        Configuration.DL.NearStaticEnd     * kCellSize,
        Configuration.DL.FarStaticEnd      * kCellSize,
        Configuration.DL.VeryFarStaticEnd  * kCellSize,
    };
    for (int i = 0; i < 3; ++i) {
        zf = std::min(rangeEnds[i], cullDist);
        if (zn < zf) {
            editProjectionZ(&ds_proj, zn, zf);
            ds_viewproj = (*view) * ds_proj;
            ranges[i].frustum = ViewFrustum(&ds_viewproj);
            ranges[i].sphere.w = zf;
            ranges[i].active = true;
        }
    }

    if (useIpc) {
        // One batched RPC for all 3 fetches + sort. The server-side work
        // (~265 us median) runs in parallel with renderShadowMap /
        // renderDistantLand / contributeDistantLandOccluders, which add
        // up to >1.5 ms of main-thread work — so cullDistantStatics_finish
        // typically sees the server already done.
        DWORD setFlags[3] = {
            ranges[0].active ? ranges[0].flag : 0,
            ranges[1].active ? ranges[1].flag : 0,
            ranges[2].active ? ranges[2].flag : 0,
        };
        ViewFrustum frustums[3] = {
            ranges[0].frustum, ranges[1].frustum, ranges[2].frustum,
        };
        D3DXVECTOR4 spheres[3] = {
            ranges[0].sphere,  ranges[1].sphere,  ranges[2].sphere,
        };
        // Cull-then-sort: the server no longer sorts the full ~13k visible
        // set (~1.35ms of drain wasted on meshes MSOC then occludes ~95% of).
        // applyMSOCToDistantStatics sorts only the ~500 survivors client-side.
        //
        // Fold the reflection-statics query (prepared in
        // prepareReflectionCullForWorker just above) into this same RPC as a 4th,
        // independent query writing visExtraShared. Its server-cull overlaps the
        // kickoff→drain head-start window; the worker then drains both statics and
        // reflection in one drain. reflStaticsWanted=false (no near-static
        // reflections, or inline-fallback kickoff) ⇒ reflFlags=0, no fold.
        // Host-side occlusion cull: ship msoc.dll's snapshot mask into the
        // single-window blob vec so the server TestRect-culls each frustum
        // survivor before PushBack — only visible survivors cross the wire.
        // Gated (default off) + needs the plugin's copy export + a ready mask.
        // occlMaskId stays InvalidVector otherwise, and the server behaves
        // exactly as before. When a ready mask ships, g_hostCullActiveThisFrame
        // tells applyMSOCToDistantStatics to skip its now-redundant cull.
        IPC::VecId occlMaskId = IPC::InvalidVector;
        if (Configuration.UseHostOcclusionCull && MSOCClient::hasMaskExport() &&
            MSOCClient::isMaskReady() && maskBlobSharedId != IPC::InvalidVector) {
            static std::vector<char> maskScratch;
            const int need = MSOCClient::copyMaskBlob(nullptr, 0);
            if (need > 0) {
                if ((int)maskScratch.size() < need) maskScratch.resize(need);
                const int wrote = MSOCClient::copyMaskBlob(maskScratch.data(), (int)maskScratch.size());
                if (wrote > 0 &&
                    maskBlobShared.assign_bytes(maskScratch.data(), (std::uint32_t)wrote)) {
                    occlMaskId = maskBlobSharedId;
                    g_hostCullActiveThisFrame = true;
                }
            }
        }

        // S3: this call took a 4th folded query (visExtraShared) carrying the
        // water-reflection statics, gated on reflStaticsWanted. With the reflection
        // gone only the 3 main ranges remain, so the branch collapsed to one call.
        ipcClient.getVisibleMeshesAllRanges(
            visDistantSharedId, 3, frustums, spheres, setFlags,
            VisibleSetSort::None,
            IPC::InvalidVector, 0, nullptr, nullptr,
            VisibleSetSort::None, occlMaskId);
    } else {
        // Non-IPC path: synchronous quadtree work happens on the main
        // thread here. No pipelining benefit, but the bucket counters are
        // available right away.
        unsigned prevSize = 0;
        if (ranges[0].active) {
            DistantLandShare::currentWorldSpace->NearStatics->GetVisibleMeshes(
                ranges[0].frustum, ranges[0].sphere, visDistant);
            g_cullDiagNearCount = (unsigned)visDistant.Size() - prevSize;
            prevSize = (unsigned)visDistant.Size();
        }
        if (ranges[1].active) {
            DistantLandShare::currentWorldSpace->FarStatics->GetVisibleMeshes(
                ranges[1].frustum, ranges[1].sphere, visDistant);
            g_cullDiagFarCount = (unsigned)visDistant.Size() - prevSize;
            prevSize = (unsigned)visDistant.Size();
        }
        if (ranges[2].active) {
            DistantLandShare::currentWorldSpace->VeryFarStatics->GetVisibleMeshes(
                ranges[2].frustum, ranges[2].sphere, visDistant);
            g_cullDiagVeryFarCount = (unsigned)visDistant.Size() - prevSize;
        }
        // No SortByState here: applyMSOCToDistantStatics compacts and sorts
        // the survivor set (visDistantSurvivors) for this path too.
    }
}

void DistantLand::cullDistantStatics_finish() {
    MGE_ZoneScopedN("cullDistantStatics_finish");
    MGE_SCOPED_TIMER("cullDistantStatics:finish");

    if (s_cullOnWorker) {
        // Verdict pass was dispatched to the cull worker at frameSetupEarly;
        // it drained the statics RPC and ran the verdict core during the sky
        // window. Here we only join — collapses to ~0 on the critical path.
        // (The Numpad cutoff was read on main in frameSetupEarly.) This is the
        // early staticsDone fence — it no longer waits on the reflection cull
        // (that joins later at waitCullReflReady before renderWaterReflection).
        waitCullStaticsReady();
    } else {
        // Inline path: non-IPC (synchronous quadtree cull already populated
        // visDistant in _kickoff), or the IPC fallback kickoff (menus /
        // not-ready early frames) where frameSetupEarly didn't dispatch.
        updateMSOCCutoffInput();

        if (Configuration.UseSharedMemory) {
            {
                MGE_ZoneScopedN("finish:wait");
                MGE_SCOPED_TIMER("cullDistantStatics:finishWait");
                // Guarded wait: an interleaved RPC between kickoff and here
                // (e.g. cullGrass's getVisibleMeshesCoarse, which awaits its
                // own result) may already have drained the AllRanges
                // completion. In that case the statics data is ready and no
                // RPC is pending — an unconditional waitForCompletion would
                // block on an event nobody signals and time out at 60s.
                ipcClient.tryWaitForCompletion();
            }
        }

        // MSOC verdict pass runs unconditionally so both instanced AND
        // non-instanced render paths consume the same cull mask. Must run
        // after the sort so msocOccluded[idx] aligns with each render
        // path's visible-set iteration order.
        {
            MGE_ZoneScopedN("finish:applyMSOC");
            if (Configuration.UseSharedMemory) {
                applyMSOCToDistantStatics(visDistantShared);
            } else {
                applyMSOCToDistantStatics(visDistant);
            }
        }

        // Inline path runs entirely on the main thread; the worker path
        // contributes the terrain boxes on the cull thread instead.
        contributeTerrainBoxOccluders(mwView * mwProj, boxOccluderDebug);
    }

    // Main-thread debug tail: Numpad5 mask dump + the per-60-frame MSOC cull
    // summary. Reads results stable after the join / inline run; touches
    // input / MSOCClient::dumpMask / LOG, so it stays on the main thread.
    runMSOCDebugTail();

    // Per-frame cull summary + phase-timer flush. Gated by the
    // LogDistantPipeline config flag — off by default. IPC path logs
    // total only (bucket split is now collapsed inside the batched RPC);
    // non-IPC logs the breakdown captured during _kickoff. Flushed after the
    // verdict so this frame's applyMSOCToDistantStatics timing is included.
    if (Configuration.LogDistantPipeline) {
        static int diagFrameCounter = 0;
        if ((diagFrameCounter++ % 60) == 0) {
            const bool useIpc = Configuration.UseSharedMemory;
            const unsigned total = useIpc
                ? (unsigned)visDistantShared.Size()
                : (unsigned)visDistant.Size();
            if (useIpc) {
                LOG::logline("-- DL statics: total=%u (IPC; bucket split not available)", total);
            } else {
                LOG::logline("-- DL statics: total=%u  near=%u  far=%u  veryFar=%u",
                             total,
                             g_cullDiagNearCount,
                             g_cullDiagFarCount,
                             g_cullDiagVeryFarCount);
            }
            MGEPhaseTimers::report();
        }
    }
    // (s_cullOnWorker is reset per-frame at the top of cullDistantStatics_kickoff,
    // which always runs before this finish; no reset needed here.)
}

// S4: renderDistantStatics() — the DX9 LOD-statics colour pass — is gone; the Forge host
// draws LOD statics itself.

void DistantLand::renderMSOCBasinBoundsDebug(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    DrawStats::ScopedStage _ds(DrawStats::Debug);
    // Toggle is now overlay-cycle state 4 (g_drawMSOCBasinBounds derived in
    // updateMSOCCutoffInput).
    if (!g_drawMSOCBasinBounds || g_msocBasinDebugBoxes.empty() || !view || !proj)
        return;

    IDirect3DStateBlock9* stateSaved = nullptr;
    if (FAILED(device->CreateStateBlock(D3DSBT_ALL, &stateSaved)) || !stateSaved)
        return;

    static std::vector<MSOCLineVertex> lineVerts;
    lineVerts.clear();
    lineVerts.reserve(g_msocBasinDebugBoxes.size() * 24);

    auto pushLine = [](float ax, float ay, float az,
                       float bx, float by, float bz,
                       DWORD color) {
        lineVerts.push_back({ax, ay, az, color});
        lineVerts.push_back({bx, by, bz, color});
    };

    for (const auto& b : g_msocBasinDebugBoxes) {
        const DWORD color = colorForMSOCVerdict(b.verdict);

        pushLine(b.minX, b.minY, b.minZ, b.maxX, b.minY, b.minZ, color);
        pushLine(b.maxX, b.minY, b.minZ, b.maxX, b.maxY, b.minZ, color);
        pushLine(b.maxX, b.maxY, b.minZ, b.minX, b.maxY, b.minZ, color);
        pushLine(b.minX, b.maxY, b.minZ, b.minX, b.minY, b.minZ, color);

        pushLine(b.minX, b.minY, b.maxZ, b.maxX, b.minY, b.maxZ, color);
        pushLine(b.maxX, b.minY, b.maxZ, b.maxX, b.maxY, b.maxZ, color);
        pushLine(b.maxX, b.maxY, b.maxZ, b.minX, b.maxY, b.maxZ, color);
        pushLine(b.minX, b.maxY, b.maxZ, b.minX, b.minY, b.maxZ, color);

        pushLine(b.minX, b.minY, b.minZ, b.minX, b.minY, b.maxZ, color);
        pushLine(b.maxX, b.minY, b.minZ, b.maxX, b.minY, b.maxZ, color);
        pushLine(b.maxX, b.maxY, b.minZ, b.maxX, b.maxY, b.maxZ, color);
        pushLine(b.minX, b.maxY, b.minZ, b.minX, b.maxY, b.maxZ, color);
    }

    D3DXMATRIX identity;
    D3DXMatrixIdentity(&identity);

    device->SetVertexDeclaration(nullptr);
    device->SetVertexShader(nullptr);
    device->SetPixelShader(nullptr);
    device->SetFVF(fvfMSOCLine);
    device->SetTransform(D3DTS_WORLD, &identity);
    device->SetTransform(D3DTS_VIEW, view);
    device->SetTransform(D3DTS_PROJECTION, proj);
    device->SetTexture(0, nullptr);
    device->SetRenderState(D3DRS_LIGHTING, FALSE);
    device->SetRenderState(D3DRS_FOGENABLE, FALSE);
    device->SetRenderState(D3DRS_ZENABLE, FALSE);
    device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    device->SetRenderState(D3DRS_CLIPPLANEENABLE, 0);

    DrawStats::count((unsigned)(lineVerts.size() / 2));
    device->DrawPrimitiveUP(D3DPT_LINELIST, (UINT)(lineVerts.size() / 2),
                            lineVerts.data(), sizeof(MSOCLineVertex));

    stateSaved->Apply();
    stateSaved->Release();
}

// Basin watershed visualizer (Numpad6). Drawn over the live camera view:
//   - cull-eligible watershed cells (req > eyeZ) as translucent quads at z=req,
//     coloured yellow→red by how far the spill height rises above the (possibly
//     locked) basin eye — the regions where statics CAN be basin-culled;
//   - a red wireframe box around every basin-occluded static — what IS culled.
// The gap between the two is statics sitting in cull-eligible cells whose own top
// still pokes above req (or that the grid is too coarse to resolve). Z-test off so
// basins hidden behind walls stay visible. Numpad0 locks the basin eye so a free
// camera can roam without the field shifting. Mirrors renderMSOCBasinBoundsDebug.
void DistantLand::renderBasinDebug(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    DrawStats::ScopedStage _ds(DrawStats::Debug);
    if (!g_drawBasinDebug || !view || !proj) return;
    if (!g_basinReqValid || g_basinReqDim == 0) return;

    IDirect3DStateBlock9* stateSaved = nullptr;
    if (FAILED(device->CreateStateBlock(D3DSBT_ALL, &stateSaved)) || !stateSaved)
        return;

    const float eyeZ = (g_basinLockEye ? g_basinLockedEye : eyePos).z;
    const float grid = g_basinReqGrid;
    const int   dim  = g_basinReqDim;
    const int   oX   = g_basinReqOriginX;
    const int   oY   = g_basinReqOriginY;

    // --- Surface: translucent quad per cull-eligible (req > eyeZ) cell ---
    static std::vector<MSOCLineVertex> triVerts;
    triVerts.clear();
    auto pushTri = [](float ax, float ay, float az, float bx, float by, float bz,
                      float cx, float cy, float cz, DWORD col) {
        triVerts.push_back({ax, ay, az, col});
        triVerts.push_back({bx, by, bz, col});
        triVerts.push_back({cx, cy, cz, col});
    };
    for (int ly = 0; ly < dim; ++ly) {
        for (int lx = 0; lx < dim; ++lx) {
            const float req = g_basinReqArr[(size_t)ly * dim + lx];
            if (!std::isfinite(req) || req <= eyeZ) continue;  // only basins above the eye
            const float s = std::min(1.0f, (req - eyeZ) / 4096.0f);
            const DWORD col = D3DCOLOR_ARGB(110, 255, (int)(255 * (1.0f - s)), 0);
            const float x0 = (float)(oX + lx) * grid, x1 = x0 + grid;
            const float y0 = (float)(oY + ly) * grid, y1 = y0 + grid;
            pushTri(x0, y0, req, x1, y0, req, x1, y1, req, col);
            pushTri(x0, y0, req, x1, y1, req, x0, y1, req, col);
        }
    }

    // --- Culled-static boxes (red wireframe) ---
    static std::vector<MSOCLineVertex> lineVerts;
    lineVerts.clear();
    auto pushLine = [](float ax, float ay, float az, float bx, float by, float bz, DWORD c) {
        lineVerts.push_back({ax, ay, az, c});
        lineVerts.push_back({bx, by, bz, c});
    };
    const DWORD boxCol = D3DCOLOR_XRGB(255, 48, 48);
    const size_t nBoxes = std::min(g_basinOccluded.size(), g_meshValues.size());
    for (size_t i = 0; i < nBoxes; ++i) {
        if (!g_basinOccluded[i]) continue;
        const D3DXVECTOR3& c = g_meshValues[i].sphere.center;
        const float r = g_meshValues[i].sphere.radius;
        const float x0 = c.x - r, x1 = c.x + r, y0 = c.y - r, y1 = c.y + r, z0 = c.z - r, z1 = c.z + r;
        pushLine(x0, y0, z0, x1, y0, z0, boxCol); pushLine(x1, y0, z0, x1, y1, z0, boxCol);
        pushLine(x1, y1, z0, x0, y1, z0, boxCol); pushLine(x0, y1, z0, x0, y0, z0, boxCol);
        pushLine(x0, y0, z1, x1, y0, z1, boxCol); pushLine(x1, y0, z1, x1, y1, z1, boxCol);
        pushLine(x1, y1, z1, x0, y1, z1, boxCol); pushLine(x0, y1, z1, x0, y0, z1, boxCol);
        pushLine(x0, y0, z0, x0, y0, z1, boxCol); pushLine(x1, y0, z0, x1, y0, z1, boxCol);
        pushLine(x1, y1, z0, x1, y1, z1, boxCol); pushLine(x0, y1, z0, x0, y1, z1, boxCol);
    }

    D3DXMATRIX identity;
    D3DXMatrixIdentity(&identity);
    device->SetVertexDeclaration(nullptr);
    device->SetVertexShader(nullptr);
    device->SetPixelShader(nullptr);
    device->SetFVF(fvfMSOCLine);
    device->SetTransform(D3DTS_WORLD, &identity);
    device->SetTransform(D3DTS_VIEW, view);
    device->SetTransform(D3DTS_PROJECTION, proj);
    device->SetTexture(0, nullptr);
    device->SetRenderState(D3DRS_LIGHTING, FALSE);
    device->SetRenderState(D3DRS_FOGENABLE, FALSE);
    device->SetRenderState(D3DRS_ZENABLE, FALSE);
    device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    device->SetRenderState(D3DRS_CLIPPLANEENABLE, 0);

    if (!triVerts.empty()) {
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
        device->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);
        device->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA);
        DrawStats::count((unsigned)(triVerts.size() / 3));
        device->DrawPrimitiveUP(D3DPT_TRIANGLELIST, (UINT)(triVerts.size() / 3),
                                triVerts.data(), sizeof(MSOCLineVertex));
    }
    if (!lineVerts.empty()) {
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
        DrawStats::count((unsigned)(lineVerts.size() / 2));
        device->DrawPrimitiveUP(D3DPT_LINELIST, (UINT)(lineVerts.size() / 2),
                                lineVerts.data(), sizeof(MSOCLineVertex));
    }

    stateSaved->Apply();
    stateSaved->Release();
}

// Curtain overlay (Numpad3 cycle, "ON + curtain debug"). Draws the horizon-curtain
// triangles that were submitted to MSOC this frame as a translucent screen-space
// fill, coloured by the wall's view depth — near walls red, far walls blue — so you
// can see which hills produced a curtain and how far back each wall hangs. The
// curtain is a screen-space construct built from this frame's view, so an XYZRHW
// overlay lands exactly on the hills it represents. Consumes and clears g_curtainDbg,
// so the OFF / plain-ON cycle states (which never fill it) draw nothing.
void DistantLand::renderCurtainDebug() {
    if (g_curtainDbg.empty() || !device) {
        g_curtainDbg.clear();
        return;
    }

    DrawStats::ScopedStage _ds(DrawStats::Debug);

    // XYZRHW is post-viewport, so map NDC -> pixels with the live viewport.
    D3DVIEWPORT9 vp;
    if (FAILED(device->GetViewport(&vp))) { g_curtainDbg.clear(); return; }

    // Depth normaliser: far reference = the distant-land draw distance.
    const float farRef = std::max(1.0f, (float)(Configuration.DL.DrawDist * kCellSize));

    static std::vector<CurtainScreenVertex> sv;
    sv.clear();
    sv.reserve(g_curtainDbg.size());
    for (const CurtainDbgVert& c : g_curtainDbg) {
        const float px = (c.ndcX * 0.5f + 0.5f) * (float)vp.Width;       // NDC x -> pixel
        const float py = (1.0f - (c.ndcY * 0.5f + 0.5f)) * (float)vp.Height; // NDC y up -> pixel y down

        float t = c.depthW / farRef;                 // 0 near .. 1 far
        if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
        const DWORD r = (DWORD)(255.0f * (1.0f - t));
        const DWORD b = (DWORD)(255.0f * t);
        const DWORD color = (0x80u << 24) | (r << 16) | (0x20u << 8) | b;   // ARGB, 50% alpha

        sv.push_back({ px, py, 0.5f, 1.0f, color });
    }

    IDirect3DStateBlock9* stateSaved = nullptr;
    if (FAILED(device->CreateStateBlock(D3DSBT_ALL, &stateSaved)) || !stateSaved) {
        g_curtainDbg.clear();
        return;
    }

    device->SetVertexDeclaration(nullptr);
    device->SetVertexShader(nullptr);
    device->SetPixelShader(nullptr);
    device->SetFVF(fvfCurtainScreen);
    device->SetTexture(0, nullptr);
    device->SetRenderState(D3DRS_LIGHTING, FALSE);
    device->SetRenderState(D3DRS_FOGENABLE, FALSE);
    device->SetRenderState(D3DRS_ZENABLE, FALSE);
    device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
    device->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);
    device->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    device->SetRenderState(D3DRS_CLIPPLANEENABLE, 0);

    DrawStats::count((unsigned)(sv.size() / 3));
    device->DrawPrimitiveUP(D3DPT_TRIANGLELIST, (UINT)(sv.size() / 3),
                            sv.data(), sizeof(CurtainScreenVertex));

    stateSaved->Apply();
    stateSaved->Release();
    g_curtainDbg.clear();
}

// Terrain box occluder overlay (Numpad3 cycle, "boxes + debug"). Draws the top lip
// of each voxel box submitted to MSOC this frame as a world-space wireframe, colour-
// coded by the box's distance from the eye (near=red -> far=blue), so you can see the
// voxelized terrain the occlusion is actually built from. Consumes/clears g_boxDbg.
void DistantLand::renderBoxOccluderDebug(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    if (g_boxDbg.empty() || !view || !proj || !device) {
        g_boxDbg.clear();
        return;
    }

    DrawStats::ScopedStage _ds(DrawStats::Debug);

    const float farRef = std::max(1.0f, (float)(Configuration.DL.DrawDist * kCellSize));

    static std::vector<MSOCLineVertex> lineVerts;
    lineVerts.clear();
    lineVerts.reserve(g_boxDbg.size() * 24);

    auto pushLine = [&](float ax, float ay, float az, float bx, float by, float bz, DWORD color) {
        lineVerts.push_back({ ax, ay, az, color });
        lineVerts.push_back({ bx, by, bz, color });
    };

    for (const BoxDbgAABB& b : g_boxDbg) {
        const float cx = 0.5f * (b.x0 + b.x1), cy = 0.5f * (b.y0 + b.y1);
        const float ddx = cx - eyePos.x, ddy = cy - eyePos.y;
        float t = sqrtf(ddx * ddx + ddy * ddy) / farRef;
        if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
        const DWORD r  = (DWORD)(255.0f * (1.0f - t));
        const DWORD bl = (DWORD)(255.0f * t);
        const DWORD color = 0xFF000000u | (r << 16) | (0x20u << 8) | bl;

        const float zt = b.zTop;
        const float zd = b.zBot;   // full box extent (down to the occluder floor)
        // top rectangle
        pushLine(b.x0, b.y0, zt, b.x1, b.y0, zt, color);
        pushLine(b.x1, b.y0, zt, b.x1, b.y1, zt, color);
        pushLine(b.x1, b.y1, zt, b.x0, b.y1, zt, color);
        pushLine(b.x0, b.y1, zt, b.x0, b.y0, zt, color);
        // bottom rectangle
        pushLine(b.x0, b.y0, zd, b.x1, b.y0, zd, color);
        pushLine(b.x1, b.y0, zd, b.x1, b.y1, zd, color);
        pushLine(b.x1, b.y1, zd, b.x0, b.y1, zd, color);
        pushLine(b.x0, b.y1, zd, b.x0, b.y0, zd, color);
        // corner drops
        pushLine(b.x0, b.y0, zt, b.x0, b.y0, zd, color);
        pushLine(b.x1, b.y0, zt, b.x1, b.y0, zd, color);
        pushLine(b.x1, b.y1, zt, b.x1, b.y1, zd, color);
        pushLine(b.x0, b.y1, zt, b.x0, b.y1, zd, color);
    }

    IDirect3DStateBlock9* stateSaved = nullptr;
    if (FAILED(device->CreateStateBlock(D3DSBT_ALL, &stateSaved)) || !stateSaved) {
        g_boxDbg.clear();
        return;
    }

    D3DXMATRIX identity;
    D3DXMatrixIdentity(&identity);
    device->SetVertexDeclaration(nullptr);
    device->SetVertexShader(nullptr);
    device->SetPixelShader(nullptr);
    device->SetFVF(fvfMSOCLine);
    device->SetTransform(D3DTS_WORLD, &identity);
    device->SetTransform(D3DTS_VIEW, view);
    device->SetTransform(D3DTS_PROJECTION, proj);
    device->SetTexture(0, nullptr);
    device->SetRenderState(D3DRS_LIGHTING, FALSE);
    device->SetRenderState(D3DRS_FOGENABLE, FALSE);
    device->SetRenderState(D3DRS_ZENABLE, FALSE);
    device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    device->SetRenderState(D3DRS_CLIPPLANEENABLE, 0);

    DrawStats::count((unsigned)(lineVerts.size() / 2));
    device->DrawPrimitiveUP(D3DPT_LINELIST, (UINT)(lineVerts.size() / 2),
                            lineVerts.data(), sizeof(MSOCLineVertex));

    stateSaved->Apply();
    stateSaved->Release();
    g_boxDbg.clear();
}

// Water-reflection proxy visualizer (Numpad1). Draws the per-cell water slabs
// tested by isReflectionWaterVisible(), coloured by MSOC verdict (green visible,
// red occluded, blue view-culled), depth-disabled so occluded slabs are still
// inspectable behind buildings. Mirrors renderMSOCBasinBoundsDebug. Moved off
// Numpad5 (which fires the MSOC mask dump) so the two no longer collide.
// S3: renderWaterProxyBoundsDebug + renderReflectionFrustumDebug (overlay cycle
// states 1 and 5) went with the water reflection they visualised.

template<class T>
void DistantLand::applyMSOCToDistantStatics(VisibleSet<T>& staticSet) {
    MGE_SCOPED_TIMER("applyMSOCToDistantStatics");

    msocOccluded.clear();
    g_msocBasinDebugBoxes.clear();
    g_msocCullDiag.valid = false;

    // Materialize the windowed IPC view into contiguous storage ONCE. This is
    // the only traversal of the remapping window; the partition pass and the
    // survivor gather below both read g_meshValues sequentially (cache-friendly,
    // no window remap). Survivor pointers reference this stable buffer.
    g_meshValues.clear();
    g_meshValues.reserve(staticSet.Size());
    staticSet.Reset();
    while (!staticSet.AtEnd())
        g_meshValues.push_back(staticSet.Next());
    const unsigned setSize = (unsigned)g_meshValues.size();

    // Basin (watershed) pre-cull — DISABLED (g_basinCullActive=false). The
    // terrain-box occluders now carry the terrain-occlusion role through MSOC.
    // Only flood the spill-height surface when the basin cull (Pass 3) or its
    // debug viz is actually on; otherwise the Dijkstra is wasted work. The box
    // pass owns g_basinMinH independently, so skipping this doesn't starve it.
    if (g_basinCullActive || g_drawBasinDebug)
        buildBasinRequiredHeight();
    g_basinOccluded.assign(setSize, 0);

    // Cull-then-sort tail. Build visDistantSurvivors as the pointer subset of
    // g_meshValues that survived (msocOccluded[idx]==0, or every mesh when the
    // mask is empty), then sort only those (~500 ptrs, contiguous deref → fast).
    // Both the depth and color static passes iterate the result. Defined as a
    // lambda so the no-cull early-out and the normal exit share one impl.
    auto gatherSurvivors = []() {
        MGE_SCOPED_TIMER("applyMSOCToDistantStatics:gatherSort");
        const bool haveMask = !msocOccluded.empty();
        visDistantSurvivors.RemoveAll();
        visDistantSurvivors.visible_set.reserve((std::uint32_t)g_meshValues.size());
        for (size_t i = 0; i < g_meshValues.size(); ++i) {
            if (!haveMask || msocOccluded[i] == 0)
                visDistantSurvivors.PushBack(g_meshValues[i]);
        }
        visDistantSurvivors.SortByState();
    };

    // Compute the occlusion mask only when culling is active in an exterior
    // with a ready mask and a non-empty set; otherwise every mesh survives.
    // When the host already culled this frame (g_hostCullActiveThisFrame),
    // staticSet is the survivor set — skip the now-redundant group+sphere cull
    // and just materialize + sort. This is where the verdict cost collapses.
    const bool cullByOcclusion =
        setSize != 0
        && !g_hostCullActiveThisFrame
        && Configuration.UseOcclusionCulling
        && MSOCClient::isAvailable()
        && MSOCClient::isMaskReady()
        && MWBridge::get()->IsExterior();
    if (!cullByOcclusion) {
        gatherSurvivors();
        return;
    }

    msocOccluded.resize(setSize, 0);

    // Numpad8/2 live cutoff adjust is read on the main thread before this
    // pure core runs (updateMSOCCutoffInput, from frameSetupEarly on the
    // worker path or cullDistantStatics_finish on the inline path), so
    // g_msocCutoffHeight is final here. GetAsyncKeyState / StatusOverlay
    // must not be touched off the main thread, hence the split.

    // Statics whose bounding sphere top sits at or below waterLevel + kTallness are
    // bucketed into flat per-cell-group OBBs and tested as a batch (~33 calls).
    // Statics above the cutoff (hillside objects, towers) go through the original
    // per-static sphere batch — they are never grouped and never falsely culled.
    const float kTallness  = g_msocCutoffHeight;
    const float waterLevel = MWBridge::get()->WaterLevel();
    const float cutoffZ    = waterLevel + kTallness;

    // ---------------------------------------------------------------------
    // Fixed grid grouping.
    //
    // DL is entirely static: a static's group cell is a pure function of its
    // fixed world position — floor(center / gSize) with a single FIXED cell
    // size (no per-frame distance-LOD). The grid is an implicit, constant
    // structure; it never changes frame to frame, so there is nothing to
    // rebuild. Per frame we only bucket the *visible* low statics into their
    // (fixed) cells via an O(1) camera-relative dense-grid lookup, replacing
    // the old O(visible × groups) linear scan. Group bounds derive directly
    // from the cell coords; the camera-dependent OBB verdicts are computed in
    // Pass 2a.
    //
    // Fixed group cell size, in Morrowind cells. Tunable: larger = fewer/safer
    // OBB tests (less popping, less culling); smaller = tighter culling.
    constexpr int kMSOCGroupCells = 2;
    const int   gCells = kMSOCGroupCells;
    const float gSize  = gCells * kCellSize;

    // Close low statics stay on the per-static sphere path (grouping is too
    // coarse near the camera); only statics beyond this band are grouped.
    const float groupCullStartDist   = nearViewRange + kCellSize;
    const float groupCullStartDistSq = groupCullStartDist * groupCullStartDist;

    // Camera-relative dense grid: maps a group cell to a group-list slot in
    // O(1) with no hashing. Origin is the eye's group cell; radius spans the
    // draw distance plus a margin. Stored value is groupIdx+1 (0 = empty).
    const int eyeGx = (int)floorf(eyePos.x / gSize);
    const int eyeGy = (int)floorf(eyePos.y / gSize);
    const int gridRadius = (int)ceilf((float)Configuration.DL.DrawDist / (float)gCells) + 2;
    const int gridDim    = 2 * gridRadius + 1;

    struct GroupEntry {
        int gx, gy;
        float minX, maxX, minY, maxY;
        MSOCClient::TestResult verdict;
    };
    static std::vector<GroupEntry>              groups;
    // Per-static group index; 0xFFFF = handled by sphere batch.
    static std::vector<std::uint16_t>           staticGroupIdx;
    // Camera-relative cell → groupIdx+1 (0 = empty), cleared each frame.
    static std::vector<std::uint16_t>           gridCellToGroup;
    // Sphere batch for high statics, near low statics, AND the members of
    // VISIBLE groups (two-level hierarchy: the group OBB is only a cheap reject;
    // anything its slab can't prove occluded falls through to a precise per-static
    // sphere test, so the survivor set is the sphere-precise set regardless of the
    // cutoff). sphereIdx maps each 4-float sphere back to its static index, so the
    // result mapping is order-independent (no fragile 0xFFFF-in-iteration scan).
    static std::vector<float>                   sphereBatch;
    static std::vector<unsigned>                sphereIdx;
    static std::vector<MSOCClient::TestResult>  sphereResults;

    groups.clear();
    groups.reserve(64);
    sphereBatch.clear();
    sphereIdx.clear();
    staticGroupIdx.assign(setSize, 0xFFFF);
    gridCellToGroup.assign((size_t)gridDim * gridDim, 0);

    // Basin verdict (used as the MIDDLE pipeline stage in Pass 2b). Conservative
    // test req > max(eyeZ, top), with top = the OBB z-max (tight) not the loose
    // sphere top, and eyeZ possibly locked (Numpad0) for inspection.
    const bool  basinReady = g_basinEnabled && g_basinReqValid && g_basinReqDim > 0;
    const float basinEyeZ  = (g_basinLockEye ? g_basinLockedEye : eyePos).z;
    const float bGrid = g_basinReqGrid;
    const int   bDim  = g_basinReqDim, bOX = g_basinReqOriginX, bOY = g_basinReqOriginY;
    auto basinHidden = [&](unsigned idx) -> bool {
        if (!basinReady) return false;
        const RenderMesh& m = g_meshValues[idx];
        const int lx = (int)floorf(m.sphere.center.x / bGrid) - bOX;
        const int ly = (int)floorf(m.sphere.center.y / bGrid) - bOY;
        if (lx < 0 || lx >= bDim || ly < 0 || ly >= bDim) return false;
        const float req = g_basinReqArr[(size_t)ly * bDim + lx];
        if (!std::isfinite(req)) return false;
        const D3DXVECTOR3& bc = m.box.center;
        const float top = bc.z + fabsf(m.box.vx.z) + fabsf(m.box.vy.z) + fabsf(m.box.vz.z);
        return req > std::max(basinEyeZ, top);
    };

    // Pass 1: partition statics into group cells. Low in-grid → fixed cell group
    // (O(1)). High / near-handoff / out-of-grid stay 0xFFFF (sphere candidates).
    // No sphere batch is built here — that happens in Pass 2b AFTER the group
    // cheap-reject and the basin filter, so neither cheap-rejected nor basin-
    // hidden statics ever enter the (expensive) sphere batch.
    int diagBasinFrontCulled = 0;
    {
        MGE_SCOPED_TIMER("applyMSOCToDistantStatics:partition");
        for (unsigned idx = 0; idx < setSize; ++idx) {
            const RenderMesh& m = g_meshValues[idx];
            const float sx = m.sphere.center.x, sy = m.sphere.center.y;
            const float sz = m.sphere.center.z, sr = m.sphere.radius;

            if (sz + sr <= cutoffZ) {
                // Low static. Its grid cell is a pure function of fixed position.
                const int gx = (int)floorf(sx / gSize);
                const int gy = (int)floorf(sy / gSize);
                const float x0 = (float)gx * gSize;
                const float y0 = (float)gy * gSize;

                // Near handoff / out-of-grid → stays a sphere candidate (0xFFFF).
                const float minDistSq = distanceSqToAABB2D(
                    eyePos.x, eyePos.y, x0, x0 + gSize, y0, y0 + gSize);
                const int lx = gx - eyeGx + gridRadius;
                const int ly = gy - eyeGy + gridRadius;
                if (minDistSq <= groupCullStartDistSq ||
                    lx < 0 || lx >= gridDim || ly < 0 || ly >= gridDim) {
                    continue;
                }

                // O(1) cell → group lookup (no linear scan).
                const size_t cell = (size_t)ly * gridDim + lx;
                std::uint16_t slot = gridCellToGroup[cell];
                if (slot == 0) {
                    slot = (std::uint16_t)(groups.size() + 1);
                    groups.push_back({gx, gy, x0, x0 + gSize, y0, y0 + gSize,
                                      MSOCClient::ResultVisible});
                    gridCellToGroup[cell] = slot;
                }
                staticGroupIdx[idx] = (std::uint16_t)(slot - 1);
            }
            // else: high static → sphere candidate (staticGroupIdx stays 0xFFFF).
        }
    }

    // Pass 2a: raster test each low group against its projected bbox footprint.
    {
        MGE_SCOPED_TIMER("applyMSOCToDistantStatics:obbTests");
        const float slabCZ = (waterLevel + cutoffZ) * 0.5f;
        const float slabHZ = kTallness * 0.5f;
        for (auto& g : groups) {
            const float cx = (g.minX + g.maxX) * 0.5f;
            const float cy = (g.minY + g.maxY) * 0.5f;
            const float hx = (g.maxX - g.minX) * 0.5f;
            const float hy = (g.maxY - g.minY) * 0.5f;
            g.verdict = MSOCClient::classifyOBB(
                cx, cy, slabCZ,
                hx,  0,      0,
                 0, hy,      0,
                 0,  0, slabHZ);
            if (g_drawMSOCBasinBounds) {
                g_msocBasinDebugBoxes.push_back({
                    g.minX, g.maxX, g.minY, g.maxY, waterLevel, cutoffZ, g.verdict
                });
            }
        }
    }

    // Pass 2b: group cheap-reject → sphere batch. Occluded group ⇒ slab fully
    // hidden ⇒ cull every member (free). Otherwise (visible-group member or
    // 0xFFFF high/near) ⇒ append to the precise sphere batch. Basin runs LAST
    // (Pass 3), so it isn't involved here.
    int diagLowCulled = 0;
    {
        MGE_SCOPED_TIMER("applyMSOCToDistantStatics:groupResolve");
        for (unsigned idx = 0; idx < setSize; ++idx) {
            const std::uint16_t gIdx = staticGroupIdx[idx];
            if (gIdx < groups.size() &&
                groups[gIdx].verdict == MSOCClient::ResultOccluded) {
                msocOccluded[idx] = 1;          // group cheap-reject
                ++diagLowCulled;
                continue;
            }
            const RenderMesh& m = g_meshValues[idx];
            sphereBatch.push_back(m.sphere.center.x);
            sphereBatch.push_back(m.sphere.center.y);
            sphereBatch.push_back(m.sphere.center.z);
            sphereBatch.push_back(m.sphere.radius);
            sphereIdx.push_back(idx);
        }
    }

    // Pass 2c: classify the sphere batch (group-cull + basin survivors) and map
    // each result back via sphereIdx (order-independent — robust to the partition
    // and Pass-2b gaps where statics were group- or basin-culled).
    int diagSphereCulled = 0;
    if (!sphereBatch.empty()) {
        MGE_SCOPED_TIMER("applyMSOCToDistantStatics:sphereBatch");
        const int nSpheres = (int)(sphereBatch.size() / 4);
        sphereResults.assign(nSpheres, MSOCClient::ResultVisible);
        MSOCClient::classifySphereBatch(sphereBatch.data(), nSpheres, sphereResults.data());
        for (int si = 0; si < nSpheres; ++si) {
            if (sphereResults[si] == MSOCClient::ResultOccluded) {
                msocOccluded[sphereIdx[si]] = 1;
                ++diagSphereCulled;
            }
        }
    }

    // Pass 3: BASIN LAST — run only on statics MSOC KEPT (msocOccluded==0). Every
    // basin cull here is a static the group + sphere tests would have DRAWN, so
    // diagBasinFrontCulled is the genuine "culls beyond MSOC" headroom — no
    // circular self-comparison. It removes them from the survivor set (fewer draw
    // calls, the main-thread saving); it no longer offloads sphere tests. The
    // box-top conservative test (req > max(eyeZ, OBB-top)) is unchanged.
    if (g_basinCullActive && basinReady) {
        MGE_SCOPED_TIMER("applyMSOCToDistantStatics:basin");
        for (unsigned idx = 0; idx < setSize; ++idx) {
            if (msocOccluded[idx]) continue;        // already culled by group/sphere
            if (basinHidden(idx)) {
                g_basinOccluded[idx] = 1;
                msocOccluded[idx]    = 1;
                ++diagBasinFrontCulled;
            }
        }
    }

    // Stash the per-frame summary for the main-thread debug tail. The
    // Numpad5 mask dump and the 60-frame "MSOC cull" log line touch input /
    // MSOCClient::dumpMask / LOG and run in cullDistantStatics_finish (main),
    // reading these stable values after the worker join. Only computed when
    // LogDistantPipeline is on — the group-occluded scan is otherwise wasted.
    if (Configuration.LogDistantPipeline) {
        int diagGroupsOccluded = 0;
        for (const auto& g : groups)
            if (g.verdict == MSOCClient::ResultOccluded) ++diagGroupsOccluded;
        g_msocCullDiag.valid          = true;
        g_msocCullDiag.setSize        = setSize;
        g_msocCullDiag.nSphere        = (unsigned)(sphereBatch.size() / 4);
        g_msocCullDiag.groupCount     = (unsigned)groups.size();
        g_msocCullDiag.groupsOccluded = diagGroupsOccluded;
        g_msocCullDiag.lowCulled      = diagLowCulled;
        g_msocCullDiag.sphereCulled   = diagSphereCulled;
        g_msocCullDiag.basinFrontCulled = diagBasinFrontCulled;

        // Basin runs LAST on MSOC survivors, so every basin cull is a static MSOC
        // KEPT → all of it is real "beyond-MSOC" headroom, none "agree". No
        // circular scan needed; the counts are exact by construction.
        g_msocCullDiag.basinGrid          = g_basinReqGrid;
        g_msocCullDiag.basinReqCells      = (unsigned)((size_t)g_basinReqDim * g_basinReqDim);
        g_msocCullDiag.basinCull          = diagBasinFrontCulled;
        g_msocCullDiag.basinCull_MSOCkeep = diagBasinFrontCulled;          // = beyond MSOC
        g_msocCullDiag.basinCull_MSOCcull = 0;
        g_msocCullDiag.basinKeep_MSOCcull = diagLowCulled + diagSphereCulled; // MSOC's own
    }

    // Compact + sort the ~500 survivors now that msocOccluded is final.
    gatherSurvivors();
}

template void DistantLand::applyMSOCToDistantStatics(VisibleSet<StlVector>& staticSet);
template void DistantLand::applyMSOCToDistantStatics(VisibleSet<IpcClientVector>& staticSet);

// ---- Terrain min-height maps for the water-reflection gate ----
//
// Per-cell (8192) and fine (kWaterFineGrid) minimum terrain height, world-space,
// built once from the captured distant-land mesh. The land world transform is
// identity (distantinit.cpp), so LandMeshCache::positions are already world-space
// — bin each vertex by floor(xy / grid) and keep the min Z.
//
// Terrain — not MSOC — is the decider for the reflection gate. Occlusion alone
// can't tell "visible water" from "visible land sitting at ~water level" (a
// street that doesn't rise above the slab is unoccluded → false-VISIBLE, the bug
// that defeated the MSOC version). Terrain height is unambiguous: a tile has
// water only where terrain dips below WaterLevel.
static constexpr float kWaterFineGrid = 512.0f;
static std::unordered_map<uint64_t, float> g_cellMinH;   // key: 8192-cell -> min Z
static std::unordered_map<uint64_t, float> g_fineMinH;   // key: 512-cell -> min Z (far mask + presence gate)
static size_t g_terrainMinHSrcSize = (size_t)-1;         // landMeshes.size() built from

// ---- Runtime near-camera terrain height grid (close reflection-mask silhouette) ----
//
// Vertex-binning into a fixed grid (g_fineMinH) can't beat the captured LOD mesh's own
// vertex spacing: where the mesh is decimated coarser than the grid, cells with no
// vertex stay empty and the mask checkerboards. The fix is to RASTERIZE the real
// terrain triangles into a fine dense grid covering the 3x3 cells around the camera —
// every cell a triangle covers gets the interpolated surface height, so coverage is
// continuous (no holes) and finer than the vertex spacing. Rebuilt only when the eye
// crosses a cell (infrequent), on the cull worker (off the main thread).
static constexpr float kNearGrid = 128.0f;               // near silhouette resolution
static constexpr int   kNearCellsPerWorldCell = (int)(DistantLand::kCellSize / kNearGrid); // 64
static constexpr int   kNearW = 3 * kNearCellsPerWorldCell;   // 3x3 world cells → 192 grid cells/side
struct NearHeightGrid {
    int   gx0 = 0, gy0 = 0;          // grid-cell index of the region origin (floor(worldOrigin/kNearGrid))
    int   builtCellX = INT_MIN, builtCellY = INT_MIN;  // eye cell this was built for
    bool  valid = false;
    std::vector<float> minZ;         // kNearW*kNearW, +inf = no terrain
};
static NearHeightGrid g_nearGrid;

// Per distant-land tile: its source mesh + XY AABB, so the near-grid rebuild only
// iterates triangles of tiles overlapping the 3x3 region (not the whole worldspace).
// Built alongside g_fineMinH (same landMeshes-changed trigger).
struct TileTriEntry {
    const DistantLand::LandMeshCache* mesh;
    float minx, miny, maxx, maxy;
};
static std::vector<TileTriEntry> g_tileTris;

// Per-g_basinGrid terrain minimum (the watershed barrier map). Rebuilt when the
// grid resolution changes or landMeshes changes; far cheaper than re-flooding,
// so it's kept separate from the per-camera-cell flood cache below.
static std::unordered_map<uint64_t, float> g_basinMinH;  // key: basinGrid-cell -> min Z
static float  g_basinMinHGrid    = 0.0f;                 // grid the map was built at
static size_t g_basinMinHSrcSize = (size_t)-1;           // landMeshes.size() built from

static inline uint64_t waterGridKey(int gx, int gy) {
    return ((uint64_t)(uint32_t)gx << 32) | (uint32_t)gy;
}

// S3: buildTerrainMinHeight, buildWaterFlowMap, updateFlowMapTexture and
// getFlowMapTransform lived here -- the baked river/coast flow field MGE's water
// shader sampled. The Forge host never consumed a flow map (its water shader carries
// its own wave model), and the only caller of buildTerrainMinHeight was the
// water-visibility gate, so the whole stack went at once.

static void buildNearHeightGrid(int eyeCellX, int eyeCellY) {
    if (g_nearGrid.valid && g_nearGrid.builtCellX == eyeCellX && g_nearGrid.builtCellY == eyeCellY)
        return;   // still current

    g_nearGrid.gx0 = (eyeCellX - 1) * kNearCellsPerWorldCell;
    g_nearGrid.gy0 = (eyeCellY - 1) * kNearCellsPerWorldCell;
    g_nearGrid.builtCellX = eyeCellX;
    g_nearGrid.builtCellY = eyeCellY;
    g_nearGrid.minZ.assign((size_t)kNearW * kNearW, 1e30f);

    // Region world bounds (for the per-tile AABB reject).
    const float rx0 = g_nearGrid.gx0 * kNearGrid;
    const float ry0 = g_nearGrid.gy0 * kNearGrid;
    const float rx1 = rx0 + kNearW * kNearGrid;
    const float ry1 = ry0 + kNearW * kNearGrid;

    for (const TileTriEntry& t : g_tileTris) {
        if (t.maxx < rx0 || t.minx > rx1 || t.maxy < ry0 || t.miny > ry1) continue;  // tile outside region
        const auto& pos = t.mesh->positions;
        const auto& idx = t.mesh->indices;
        const std::uint32_t nv = (std::uint32_t)pos.size();
        for (size_t i = 0; i + 3 <= idx.size(); i += 3) {
            const std::uint32_t i0 = idx[i], i1 = idx[i + 1], i2 = idx[i + 2];
            if (i0 >= nv || i1 >= nv || i2 >= nv) continue;   // malformed index guard
            const D3DXVECTOR3& a = pos[i0];
            const D3DXVECTOR3& b = pos[i1];
            const D3DXVECTOR3& c = pos[i2];
            // Triangle XY bbox → grid-cell range, clipped to the region.
            float bminx = std::min(a.x, std::min(b.x, c.x));
            float bmaxx = std::max(a.x, std::max(b.x, c.x));
            float bminy = std::min(a.y, std::min(b.y, c.y));
            float bmaxy = std::max(a.y, std::max(b.y, c.y));
            int cx0 = (int)floorf(bminx / kNearGrid) - g_nearGrid.gx0;
            int cx1 = (int)floorf(bmaxx / kNearGrid) - g_nearGrid.gx0;
            int cy0 = (int)floorf(bminy / kNearGrid) - g_nearGrid.gy0;
            int cy1 = (int)floorf(bmaxy / kNearGrid) - g_nearGrid.gy0;
            if (cx1 < 0 || cy1 < 0 || cx0 >= kNearW || cy0 >= kNearW) continue;
            if (cx0 < 0) cx0 = 0; if (cy0 < 0) cy0 = 0;
            if (cx1 >= kNearW) cx1 = kNearW - 1; if (cy1 >= kNearW) cy1 = kNearW - 1;
            // Barycentric setup (XY); skip degenerate triangles.
            const float dx1 = b.x - a.x, dy1 = b.y - a.y;
            const float dx2 = c.x - a.x, dy2 = c.y - a.y;
            const float den = dx1 * dy2 - dx2 * dy1;
            if (fabsf(den) < 1e-6f) continue;
            const float invDen = 1.0f / den;
            for (int gy = cy0; gy <= cy1; ++gy) {
                const float wy = (g_nearGrid.gy0 + gy + 0.5f) * kNearGrid;
                float* row = &g_nearGrid.minZ[(size_t)gy * kNearW];
                for (int gx = cx0; gx <= cx1; ++gx) {
                    const float wx = (g_nearGrid.gx0 + gx + 0.5f) * kNearGrid;
                    // Point-in-triangle via barycentric; interpolate z if inside.
                    const float px = wx - a.x, py = wy - a.y;
                    const float w1 = (px * dy2 - dx2 * py) * invDen;
                    const float w2 = (dx1 * py - px * dy1) * invDen;
                    const float w0 = 1.0f - w1 - w2;
                    if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) continue;
                    const float z = w0 * a.z + w1 * b.z + w2 * c.z;
                    if (z < row[gx]) row[gx] = z;
                }
            }
        }
    }
    g_nearGrid.valid = true;
    LOG::logline("-- [water-gate] near height grid rebuilt for cell (%d,%d)", eyeCellX, eyeCellY);
}

// Build the basin barrier map (terrain MIN per g_basinGrid cell) at the given
// resolution. Rebuilt only on grid-change or landMeshes-change (key press / land
// reload), so the per-frame flood never re-bins vertices.
static void buildBasinMinHeight(
    const std::unordered_map<IDirect3DVertexBuffer9*, DistantLand::LandMeshCache>& landMeshes,
    float grid) {
    g_basinMinH.clear();
    for (const auto& kv : landMeshes) {
        for (const D3DXVECTOR3& p : kv.second.positions) {
            const uint64_t k = waterGridKey((int)floorf(p.x / grid), (int)floorf(p.y / grid));
            auto it = g_basinMinH.find(k);
            if (it == g_basinMinH.end()) g_basinMinH.emplace(k, p.z);
            else if (p.z < it->second)   it->second = p.z;
        }
    }
    g_basinMinHGrid    = grid;
    g_basinMinHSrcSize = landMeshes.size();
    LOG::logline("-- [basin] barrier map built: grid=%.0f, %zu cells", grid, g_basinMinH.size());
}

// Terrain box occluders — voxelize the terrain min-height grid into boxes (top =
// cell min height, so each box is buried inside the real terrain and never occludes
// a static on the surface) and submit them to MSOC. Two reductions keep the batch
// inside the occluder budget while preserving a fine grid:
//   1. FRUSTUM CULL — only cells inside the lateral view frustum are considered, so
//      the full radius disk collapses to the visible wedge.
//   2. GREEDY MERGE — adjacent in-frustum cells in the same height bucket are merged
//      into larger rectangles (one box each), so flat terrain costs a handful of
//      boxes instead of thousands. Merged box top = MIN over its cells (conservative).
// Owns the min-height map (builds g_basinMinH itself, cached on grid/landMeshes —
// no dependency on the basin flood, which is now disabled). Submitted via
// addPreTransformedOccluder (the path that lands in the queryable mask). The
// caller passes a frame-stable view-proj so this can run on the cull worker.
// captureDebug stashes the merged AABBs for the in-world overlay.
void DistantLand::contributeTerrainBoxOccluders(const D3DXMATRIX& viewProj, bool captureDebug) {
    MGE_ZoneScopedN("contributeTerrainBoxes");
    MGE_SCOPED_TIMER("contributeTerrainBoxOccluders");

    g_boxDbg.clear();

    if (!Configuration.UseOcclusionCulling || !MSOCClient::isAvailable())
        return;
    if (!MWBridge::get()->IsExterior())
        return;

    // Ensure the min-height barrier map is current (cheap; rebuilt only on grid /
    // landMeshes change). The boxes are this map voxelized.
    const float grid = g_basinGrid;
    if (g_basinMinHGrid != grid || g_basinMinHSrcSize != landMeshes.size())
        buildBasinMinHeight(landMeshes, grid);
    if (g_basinMinH.empty())
        return;

    const int   camGX  = (int)floorf(eyePos.x / grid);
    const int   camGY  = (int)floorf(eyePos.y / grid);
    const int   radius = std::max(1, (int)ceilf(g_boxRadiusCells * kCellSize / grid));
    const int   W = 2 * radius + 1, H = 2 * radius + 1;
    const int   baseGX = camGX - radius, baseGY = camGY - radius;

    constexpr float kEmpty   = -1e30f;     // "no terrain / frustum-culled" sentinel

    // Box floor. The box only needs to reach far enough below the terrain min to
    // cover statics sitting at/near water level behind a ridge — the OBB occlusion
    // test needs the whole occludee bounds covered. A flat 500 units below water
    // does it; the old 16k pillar was wasteful depth (and fed the looking-down
    // frustum problem). The min with zTop keeps the box non-degenerate where the
    // terrain min itself dips below water. Still conservative: the top is the cell
    // min, buried in the heightfield, so the box only occludes things genuinely
    // behind terrain.
    const float waterLevel    = (float)MWBridge::get()->WaterLevel();
    constexpr float kBoxFloorBelowWater = 500.0f;
    auto boxFloor = [&](float zTop) {
        return std::min(zTop - kBoxFloorBelowWater, waterLevel - kBoxFloorBelowWater);
    };

    // Lateral frustum planes (left/right/bottom/top) from the passed view-proj
    // (screen-space mwView*mwProj). We skip the near/far planes: MOC has no far
    // plane (depth = 1/w), so distant boxes beyond the far plane must still be
    // kept — only off-screen-sideways cells are culled.
    const D3DXMATRIX& M = viewProj;
    const float planes[4][4] = {
        { M._14 + M._11, M._24 + M._21, M._34 + M._31, M._44 + M._41 },  // left
        { M._14 - M._11, M._24 - M._21, M._34 - M._31, M._44 - M._41 },  // right
        { M._14 + M._12, M._24 + M._22, M._34 + M._32, M._44 + M._42 },  // bottom
        { M._14 - M._12, M._24 - M._22, M._34 - M._32, M._44 - M._42 },  // top
    };
    auto boxOutside = [&](float x0, float y0, float x1, float y1, float zb, float zt) {
        for (const auto& p : planes) {
            const float px = p[0] > 0 ? x1 : x0;
            const float py = p[1] > 0 ? y1 : y0;
            const float pz = p[2] > 0 ? zt : zb;
            if (p[0] * px + p[1] * py + p[2] * pz + p[3] < 0.0f) return true;  // fully outside
        }
        return false;
    };

    // Dense local height grid over the radius window; kEmpty where there is no terrain
    // or the cell box is outside the frustum.
    static std::vector<float> cellH;
    cellH.assign((size_t)W * H, kEmpty);
    for (int ly = 0; ly < H; ++ly) {
        for (int lx = 0; lx < W; ++lx) {
            const int gx = baseGX + lx, gy = baseGY + ly;
            auto it = g_basinMinH.find(waterGridKey(gx, gy));
            if (it == g_basinMinH.end()) continue;
            const float zt = it->second;
            const float x0 = gx * grid, x1 = x0 + grid, y0 = gy * grid, y1 = y0 + grid;
            // Frustum-cull by the box TOP FACE only, not the buried floor. The floor
            // drops ~16k units (boxFloor) purely to extend downward occlusion for
            // statics behind ridges; it is always below the local terrain min, so it
            // never forms a visible silhouette and must not vote on visibility. When
            // the camera pitches down, the tilted screen bottom/top planes widen with
            // depth and the deep column's far-Z corner falls INSIDE the downward cone
            // even when the surface is laterally off-screen — that is why looking at
            // your feet kept all 546 boxes. Testing the slab at zt culls strictly more
            // (for any plane with a negative z-coeff the p-vertex moves up to zt,
            // lowering the dot), and removing a box only ever drops occlusion → more
            // draws, never a visual gap. So the cull stays conservative.
            if (boxOutside(x0, y0, x1, y1, zt, zt)) continue;
            // DL-only exclusion: terrain boxes are DISTANT-LAND occluders. Within the
            // near view distance the real terrain is rasterized full-res (aggregate
            // terrain), so a box there is redundant AND harmful — its vertical wall is
            // a cliff vs the gradual slope, near-clips into the view, and over-occludes
            // upslope statics (the original flicker). So skip any box whose footprint is
            // within the near view range of the eye; boxes only exist where DL takes
            // over. (max with one grid cell keeps the minimum near-clip guard if the
            // view distance is ever tuned very low.)
            const float dlNearExcl = std::max(grid, nearViewRange);
            if (distanceSqToAABB2D(eyePos.x, eyePos.y, x0, x1, y0, y1) <= dlNearExcl * dlNearExcl)
                continue;
            cellH[(size_t)ly * W + lx] = zt;
        }
    }

    // Clip-space projection + near-plane clip emit (the proven curtain path).
    // emitBox below projects with `viewProj` (the parameter) directly.
    constexpr float kWNear = 1.0f;
    static const int kBoxTris[36] = {
        4,5,6, 4,6,7,  0,2,1, 0,3,2,  0,1,5, 0,5,4,
        1,2,6, 1,6,5,  2,3,7, 2,7,6,  3,0,4, 3,4,7,
    };
    static std::vector<float>        verts;
    static std::vector<unsigned int> tris;
    verts.clear();
    tris.clear();

    struct ClipV { float x, y, w; };
    auto appendVert = [&](const ClipV& v) {
        const unsigned int idx = (unsigned int)(verts.size() / 4);
        verts.push_back(v.x); verts.push_back(v.y); verts.push_back(0.0f); verts.push_back(v.w);
        tris.push_back(idx);
    };
    auto emitTri = [&](const ClipV& a, const ClipV& b, const ClipV& c) {
        const ClipV in[3] = { a, b, c };
        ClipV poly[4];
        int n = 0;
        for (int i = 0; i < 3; ++i) {
            const ClipV& cur = in[i];
            const ClipV& nxt = in[(i + 1) % 3];
            const bool curIn = cur.w >= kWNear, nxtIn = nxt.w >= kWNear;
            if (curIn) poly[n++] = cur;
            if (curIn != nxtIn) {
                const float t = (kWNear - cur.w) / (nxt.w - cur.w);
                poly[n++] = { cur.x + t * (nxt.x - cur.x), cur.y + t * (nxt.y - cur.y), kWNear };
            }
        }
        for (int i = 1; i + 1 < n; ++i) {
            appendVert(poly[0]); appendVert(poly[i]); appendVert(poly[i + 1]);
        }
    };
    auto emitBox = [&](float x0, float y0, float x1, float y1, float zTop) {
        // Inflate the footprint 1.25x about its centre to close grid-vs-terrain
        // misalignment gaps (adjacent boxes now overlap ~0.25 cell). Conservative: the
        // top stays at the cell minimum (buried below the surface), so a wider box still
        // only occludes things genuinely behind terrain — it just seals the seams.
        {
            const float cx = 0.5f * (x0 + x1), cy = 0.5f * (y0 + y1);
            const float ex = 0.625f * (x1 - x0), ey = 0.625f * (y1 - y0);  // 1.25x half-extent
            x0 = cx - ex; x1 = cx + ex;
            y0 = cy - ey; y1 = cy + ey;
        }
        const float zBot = boxFloor(zTop);
        const float corners[8][3] = {
            {x0,y0,zBot},{x1,y0,zBot},{x1,y1,zBot},{x0,y1,zBot},
            {x0,y0,zTop},{x1,y0,zTop},{x1,y1,zTop},{x0,y1,zTop},
        };
        ClipV cc[8];
        for (int k = 0; k < 8; ++k) {
            const float wx = corners[k][0], wy = corners[k][1], wz = corners[k][2];
            cc[k].x = wx * viewProj._11 + wy * viewProj._21 + wz * viewProj._31 + viewProj._41;
            cc[k].y = wx * viewProj._12 + wy * viewProj._22 + wz * viewProj._32 + viewProj._42;
            cc[k].w = wx * viewProj._14 + wy * viewProj._24 + wz * viewProj._34 + viewProj._44;
        }
        for (int t = 0; t < 36; t += 3)
            emitTri(cc[kBoxTris[t]], cc[kBoxTris[t + 1]], cc[kBoxTris[t + 2]]);
    };

    // Distance-graded coarsening (1024 near -> 2048 far). Boxes near the DL
    // boundary keep the base grid; further out we collapse each world-aligned 2x2
    // block onto one 2x-coarser box by stamping the block's MIN top onto all four
    // members, so the greedy merge below fuses them. The outer rings dominate the
    // disc area, so this roughly halves the box/tri count and keeps the batch under
    // the shared occluder-tri budget. World-aligned (gx>>1, arithmetic shift floors
    // in C++20) so the supercell lattice is frame-stable — array-space grouping
    // would jitter distant boxes and pop distant statics in/out of occlusion. A
    // supercell with ANY near member is left untouched (no near coarsening). MIN
    // top keeps it at or below every member's terrain min, so still buried.
    {
        const float coarsenDist   = std::max(2.5f * kCellSize, nearViewRange + 1.5f * kCellSize);
        const float coarsenDistSq = coarsenDist * coarsenDist;
        auto cellDistSq = [&](int lx, int ly) {
            const int gx = baseGX + lx, gy = baseGY + ly;
            const float x0 = gx * grid, x1 = x0 + grid, y0 = gy * grid, y1 = y0 + grid;
            return distanceSqToAABB2D(eyePos.x, eyePos.y, x0, x1, y0, y1);
        };
        static std::unordered_map<uint64_t, float> superMin;     // supercell -> min top
        static std::unordered_set<uint64_t>        superBlocked; // supercell has a near member
        superMin.clear();
        superBlocked.clear();
        for (int ly = 0; ly < H; ++ly)
            for (int lx = 0; lx < W; ++lx) {
                const size_t idx = (size_t)ly * W + lx;
                if (cellH[idx] <= kEmpty + 1.0f) continue;
                const uint64_t sk = waterGridKey((baseGX + lx) >> 1, (baseGY + ly) >> 1);
                if (cellDistSq(lx, ly) <= coarsenDistSq) { superBlocked.insert(sk); continue; }
                auto it = superMin.find(sk);
                if (it == superMin.end()) superMin.emplace(sk, cellH[idx]);
                else if (cellH[idx] < it->second) it->second = cellH[idx];
            }
        for (int ly = 0; ly < H; ++ly)
            for (int lx = 0; lx < W; ++lx) {
                const size_t idx = (size_t)ly * W + lx;
                if (cellH[idx] <= kEmpty + 1.0f) continue;
                if (cellDistSq(lx, ly) <= coarsenDistSq) continue;
                const uint64_t sk = waterGridKey((baseGX + lx) >> 1, (baseGY + ly) >> 1);
                if (superBlocked.count(sk)) continue;
                cellH[idx] = superMin[sk];
            }
    }

    // Greedy merge: group adjacent cells in the same height bucket into rectangles.
    constexpr float kBucket = 1024.0f;   // merge tolerance (coarser = fewer boxes)
    static std::vector<char> visited;
    visited.assign((size_t)W * H, 0);

    int boxes = 0;
    for (int ly = 0; ly < H; ++ly) {
        for (int lx = 0; lx < W; ++lx) {
            const size_t idx = (size_t)ly * W + lx;
            if (visited[idx] || cellH[idx] <= kEmpty + 1.0f) continue;
            const int bk = (int)floorf(cellH[idx] / kBucket);

            // grow the run rightward, then the block downward, within the same bucket
            int w = 1;
            while (lx + w < W) {
                const size_t i = (size_t)ly * W + lx + w;
                if (visited[i] || cellH[i] <= kEmpty + 1.0f || (int)floorf(cellH[i] / kBucket) != bk) break;
                ++w;
            }
            int h = 1;
            bool grow = true;
            while (ly + h < H && grow) {
                for (int i = 0; i < w; ++i) {
                    const size_t j = (size_t)(ly + h) * W + lx + i;
                    if (visited[j] || cellH[j] <= kEmpty + 1.0f || (int)floorf(cellH[j] / kBucket) != bk) { grow = false; break; }
                }
                if (grow) ++h;
            }

            // mark visited, take MIN top over the rectangle (conservative)
            float minTop = cellH[idx];
            for (int yy = 0; yy < h; ++yy)
                for (int xx = 0; xx < w; ++xx) {
                    const size_t j = (size_t)(ly + yy) * W + lx + xx;
                    if (cellH[j] < minTop) minTop = cellH[j];
                    visited[j] = 1;
                }

            const float x0 = (baseGX + lx) * grid, x1 = (baseGX + lx + w) * grid;
            const float y0 = (baseGY + ly) * grid, y1 = (baseGY + ly + h) * grid;
            emitBox(x0, y0, x1, y1, minTop);
            ++boxes;
            if (captureDebug) g_boxDbg.push_back({ x0, y0, x1, y1, minTop, boxFloor(minTop) });
        }
    }

    if (verts.empty())
        return;

    const int vtxCount = (int)(verts.size() / 4);
    const int triCount = (int)(tris.size() / 3);
    const bool ok = MSOCClient::addPreTransformedOccluder(
        verts.data(), vtxCount, /*stride*/ 16, /*offY*/ 4, /*offW*/ 12,
        tris.data(), triCount);

    static int diagFrame = 0;
    if (Configuration.LogDistantPipeline && (diagFrame++ % 60) == 0) {
        LOG::logline("-- MSOC occluder: terrain boxes %s — grid=%.0f radius=%d merged=%d tris=%d",
                     ok ? "submitted" : "REJECTED", grid, radius, boxes, triCount);
    }
}

// ---- Basin (watershed) occlusion pre-cull ----
//
// requiredHeight(cell) = min over all paths from the camera cell of
//                        ( max barrier height along the path )
// = the lowest full-cell wall separating the camera's basin from `cell`. A
// minimax flood (Dijkstra with `max` as the path operator) over the coarse map.
//
// Barrier height per cell = the cell's terrain MINIMUM, NOT its max/ridge.
// Conservativeness: req(C) > max(eyeZ, top) must imply the static is truly
// hidden, never visible. With cell-MIN, req(C) > S means every path E->C
// crosses a cell whose *entire* terrain (even its lowest point) sits above S —
// a solid full-cell wall taller than both the eye and the static top, that the
// flood proves you cannot route around. A straight eye->top sightline stays
// within (-inf, max(eyeZ,top)], so such a wall always blocks it ⇒ occluded.
// (Cell-MAX would be non-conservative: a sightline can thread a low saddle of
// a basin-grid-wide cell, so a peak there doesn't prove occlusion — that over-
// culled visible statics. The trade is weaker culling at coarser resolution,
// which is why the barrier grid (g_basinGrid) is tunable: finer makes the full-
// cell-wall condition achievable for real ridges. It never hides a visible
// static at any resolution.)
//
// Cells with no terrain data use barrier = -inf (free passage → lower required
// height → conservative, never over-culls). Cached on the camera cell + grid;
// runs on the cull worker, overlapping the main thread.
void DistantLand::buildBasinRequiredHeight() {
    MGE_ZoneScopedN("buildBasinRequiredHeight");

    const float grid = g_basinGrid;

    // Ensure the barrier map is current for this resolution + land set.
    if (g_basinMinHGrid != grid || g_basinMinHSrcSize != landMeshes.size()) {
        buildBasinMinHeight(landMeshes, grid);
    }
    if (g_basinMinH.empty()) {
        g_basinReqValid = false;
        return;
    }

    // Eye fed into the basin: live, or a frozen snapshot while locked (Numpad0)
    // so a free camera can inspect basins without the watershed shifting.
    const D3DXVECTOR4 be = g_basinLockEye ? g_basinLockedEye : eyePos;
    const int camX = (int)floorf(be.x / grid);
    const int camY = (int)floorf(be.y / grid);

    // Cache: recompute only on camera-cell change, grid change, or map rebuild.
    if (g_basinReqValid && camX == g_basinCamCellX && camY == g_basinCamCellY &&
        g_basinReqGrid == grid && g_basinReqSrcSize == g_basinMinHSrcSize) {
        return;
    }

    // Domain: cells within DrawDist of the camera (+margin). DrawDist is in
    // Morrowind cells (== kCellSize), converted to basin-grid cells.
    const int radius   = (int)ceilf(Configuration.DL.DrawDist * kCellSize / grid) + 2;
    const int dim      = 2 * radius + 1;
    const int originX  = camX - radius;
    const int originY  = camY - radius;

    constexpr float kInf = std::numeric_limits<float>::infinity();
    static std::vector<float>   barrierArr;   // dense terrain-min over the domain
    static std::vector<uint8_t> closed;
    g_basinReqArr.assign((size_t)dim * dim, kInf);
    barrierArr.assign((size_t)dim * dim, -kInf);
    closed.assign((size_t)dim * dim, 0);
    std::vector<float>& reqArr = g_basinReqArr;

    // Scatter the sparse barrier map into the dense domain array ONCE (iterate
    // the map, not 8 hash lookups per cell). Cells outside the domain are skipped;
    // unsampled cells stay -inf (free passage). This is the only g_basinMinH walk.
    for (const auto& kv : g_basinMinH) {
        const int gx = (int)(int32_t)(kv.first >> 32);
        const int gy = (int)(int32_t)(kv.first & 0xffffffffu);
        const int lx = gx - originX;
        const int ly = gy - originY;
        if (lx < 0 || lx >= dim || ly < 0 || ly >= dim) continue;
        barrierArr[(size_t)ly * dim + lx] = kv.second;
    }

    struct Node { float req; int idx; };
    struct Cmp  { bool operator()(const Node& a, const Node& b) const { return a.req > b.req; } };
    std::priority_queue<Node, std::vector<Node>, Cmp> pq;

    const int srcIdx = radius * dim + radius;
    reqArr[srcIdx] = barrierArr[srcIdx];
    pq.push({reqArr[srcIdx], srcIdx});

    while (!pq.empty()) {
        const Node n = pq.top();
        pq.pop();
        if (closed[n.idx]) continue;     // stale heap entry
        closed[n.idx] = 1;
        const int lx = n.idx % dim;
        const int ly = n.idx / dim;
        // 8-connectivity, dense neighbour indices (barrier read = array, no hash).
        for (int dy = -1; dy <= 1; ++dy) {
            const int nly = ly + dy;
            if (nly < 0 || nly >= dim) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                const int nlx = lx + dx;
                if (nlx < 0 || nlx >= dim) continue;
                const int ni = nly * dim + nlx;
                if (closed[ni]) continue;
                const float cand = std::max(n.req, barrierArr[ni]);
                if (cand < reqArr[ni]) {
                    reqArr[ni] = cand;
                    pq.push({cand, ni});
                }
            }
        }
    }

    g_basinReqOriginX = originX;
    g_basinReqOriginY = originY;
    g_basinReqDim     = dim;
    g_basinCamCellX   = camX;
    g_basinCamCellY   = camY;
    g_basinReqGrid    = grid;
    g_basinReqSrcSize = g_basinMinHSrcSize;
    g_basinReqValid   = true;
}

// Water-reflection gate (Phase A). renderStage0 called renderWaterReflection
// whenever the cell *contains* water (CellHasWater) — even facing into city
// streets with the river off-frame. That pass measured ~1ms (0.64ms of it the
// reflected-statics IPC wait, 3651 statics) for a reflection nothing samples.
// This returns true only if some water surface is actually in view, letting the
// caller clearReflection() + skip otherwise.
//
// Two-stage decider: terrain height first, MSOC occlusion second.
//   1. Terrain (cheap, reliable): for each water tile (cell-sized far, subdivided
//      near the player) frustum-cull the thin water box, then ask the min-height
//      maps whether terrain in that footprint dips below WaterLevel. Street/ground
//      at ~water level → terrain >= water → dry, dropped; riverbed → water present.
//      A tile with no height samples is very distant unsampled terrain → dropped.
//   2. MSOC (only on the surviving water tiles): box-test the slab against the
//      occlusion mask and cull water that's present + in frustum but hidden behind
//      buildings/terrain. Cheap because it runs on the few real-water tiles, not
//      every cell. When MSOC is unavailable the survivor is kept (can't prove it
//      hidden), so the gate still works on terrain alone.
//
// Returns true (reflect, unchanged behavior) only when there's no terrain data at
// all (e.g. a worldspace without distant land).
// S3: isReflectionWaterVisible (the water-visibility gate + screen-rect build),
// materializeReflectionMeshes, cullReflectionSurvivors, reflWaterMaskTestNDC,
// prepareReflectionCullForWorker and workerReflectionGateAndMask were all here.
// They culled the reflection draw set; there is no reflection draw any more.
