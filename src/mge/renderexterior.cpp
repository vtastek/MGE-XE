
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

// Same idea for the reflection-statics set: the cull worker materializes
// visExtraShared into this stable buffer ONCE (before releasing the IPC channel),
// and reflectionSurvivors holds pointers into it so the main-thread draw never
// traverses the live IPC window concurrently with the worker / main RPCs.
static std::vector<RenderMesh> g_reflMeshValues;

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
static bool g_drawWaterProxyBounds = false;
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

// [REFL PIPE] diagnostic: reflection-statics pipeline stage counts. Filled by
// isReflectionWaterVisible (water-tile stages) and cullReflectionSurvivors (statics
// in/out), read on the main thread by logReflStaticNearFar to pin WHERE the count
// diverges on a jump frame. Plain ints; the worker fills, main reads — a benign
// diagnostic race (no tearing concern at this granularity).
struct ReflPipeDiag {
    int tilesTested = 0, waterPresent = 0, waterOccluded = 0, rects = 0;
    int queried = 0, survivors = 0;
    bool msocUsable = false;
    // Flip diagnostics: tiles whose in-range visible-state changed since last frame
    // (the flickering ones). farness = horizontal distance (cells); tiltedness =
    // elevation angle of the eye->tile ray above the water plane (grazing = small).
    int   flips = 0;
    float flipDistMinCells = 0, flipDistMaxCells = 0;
    float flipElevMinDeg = 0, flipElevMaxDeg = 0;
};
static ReflPipeDiag g_reflPipe;

// Runtime A/B toggle (Numpad1): enable the stage-2 fine water-silhouette mask (the
// runtime near-grid raster + per-object mask test). OFF → skip the near-grid build and
// the raster, leave the mask invalid so reflWaterMaskTestNDC keeps all → rect-only
// (stage-1) cull, exactly the pre-mask behavior. Lets us bisect whether stage-2 is
// implicated in a crash without swapping DLLs. Read on the main thread (updateMSOCCutoffInput)
// and by the gate (worker or inline) — a benign cross-thread bool, like the other debug flags.
static bool g_reflFineMaskEnabled = true;

// True once isReflectionWaterVisible ran its tile loop this frame (i.e. it did NOT
// early-out on interior / no-terrain-data). Lets cullReflectionSurvivors tell apart
// "gate ran and found no in-range water → cull statics" from "gate couldn't compute
// → keep all (legacy fallback)". Reset per frame at the top of the gate.
static bool g_reflRectsComputed = false;

// ---- Reflection water-silhouette mask (stage-2 fine cull) ----
//
// A tight binary screen-NDC mask of where VISIBLE water actually lands, built by
// software-rastering the WET fine cells (g_fineMinH < waterZ) of the tiles that
// already survived to emit a reflectionWaterRect. reflectionWaterRects is the cheap
// COARSE stage (one loose AABB per wet tile); this mask is the FINE stage that drops
// objects FLANKING a thin/diagonal river — near a wet tile (so they pass the rect)
// but off the actual water silhouette. Water lies on the mirror plane, so its main-
// cam NDC equals the reflection-cam NDC; the mask lives in the same screen space both
// reflection culls test in. Binary (no Z): the reflection clips all geometry to one
// side of the water plane, so nothing sits between camera and plane to depth-
// disambiguate — a "water here" bit cannot mis-cull (a visible reflection lands on a
// water pixel by definition). 256x128 cells packed as uint64 words (4 KB), file-scope
// so there is no per-frame allocation; cleared each frame in the gate. kReflMaskW/H is
// the silhouette tightness knob (bump to 512x256 if a far river reads blocky).
static constexpr int kReflMaskW = 256;
static constexpr int kReflMaskH = 128;
static constexpr int kReflMaskWords = (kReflMaskW * kReflMaskH) / 64;
static uint64_t g_reflWaterMask[kReflMaskWords];
// false on the interior / no-terrain-data early-outs (cull then keeps all, mirroring
// the empty-rects fallback); set true once the tile loop runs. Cleared per frame.
static bool g_reflWaterMaskValid = false;

// Map an NDC [-1,1] AABB to the inclusive mask cell range. Returns false if the AABB
// lies fully outside the mask (off-screen); clamps a straddling AABB to the edges.
static inline bool reflMaskCellRange(float nminx, float nminy, float nmaxx, float nmaxy,
                                     int& ix0, int& iy0, int& ix1, int& iy1) {
    ix0 = (int)floorf((nminx * 0.5f + 0.5f) * kReflMaskW);
    ix1 = (int)floorf((nmaxx * 0.5f + 0.5f) * kReflMaskW);
    iy0 = (int)floorf((nminy * 0.5f + 0.5f) * kReflMaskH);
    iy1 = (int)floorf((nmaxy * 0.5f + 0.5f) * kReflMaskH);
    if (ix1 < 0 || iy1 < 0 || ix0 >= kReflMaskW || iy0 >= kReflMaskH) return false;
    if (ix0 < 0) ix0 = 0;
    if (iy0 < 0) iy0 = 0;
    if (ix1 >= kReflMaskW) ix1 = kReflMaskW - 1;
    if (iy1 >= kReflMaskH) iy1 = kReflMaskH - 1;
    return true;
}

// Set every mask bit covered by an NDC AABB. Loose (the AABB of a grazing-angle water
// quad balloons inland) — used only as the behind-near-plane fallback in the rasterizer
// where the proper quad fill can't run.
static inline void reflMaskSetRange(float nminx, float nminy, float nmaxx, float nmaxy) {
    int ix0, iy0, ix1, iy1;
    if (!reflMaskCellRange(nminx, nminy, nmaxx, nmaxy, ix0, iy0, ix1, iy1)) return;
    for (int y = iy0; y <= iy1; ++y) {
        const int base = y * kReflMaskW;
        for (int x = ix0; x <= ix1; ++x) {
            const int bit = base + x;
            g_reflWaterMask[bit >> 6] |= (uint64_t(1) << (bit & 63));
        }
    }
}

// Set one mask bit (mask-cell coords, no bounds check by caller responsibility).
static inline void reflMaskSetCell(int cx, int cy) {
    if (cx < 0 || cy < 0 || cx >= kReflMaskW || cy >= kReflMaskH) return;
    const int bit = cy * kReflMaskW + cx;
    g_reflWaterMask[bit >> 6] |= (uint64_t(1) << (bit & 63));
}

// Scanline-fill a triangle given in mask-cell float coords (edge-function rasterizer,
// pixel-centre sampling). Tight: covers exactly the projected footprint, no AABB slop.
static inline void reflMaskFillTri(float x0, float y0, float x1, float y1, float x2, float y2) {
    // Reject non-finite verts (degenerate edge-on projection near the eye), then clamp
    // the bbox in FLOAT space before the int cast — so a wildly off-screen corner can
    // never produce an out-of-range (UB) int and an out-of-bounds mask write.
    if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) ||
        !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2)) return;
    float fminx = std::max(0.0f, std::min((float)kReflMaskW, std::min(x0, std::min(x1, x2))));
    float fmaxx = std::max(0.0f, std::min((float)kReflMaskW, std::max(x0, std::max(x1, x2))));
    float fminy = std::max(0.0f, std::min((float)kReflMaskH, std::min(y0, std::min(y1, y2))));
    float fmaxy = std::max(0.0f, std::min((float)kReflMaskH, std::max(y0, std::max(y1, y2))));
    int minx = (int)floorf(fminx);
    int maxx = (int)ceilf (fmaxx);
    int miny = (int)floorf(fminy);
    int maxy = (int)ceilf (fmaxy);
    if (maxx > kReflMaskW) maxx = kReflMaskW;
    if (maxy > kReflMaskH) maxy = kReflMaskH;
    if (minx >= maxx || miny >= maxy) return;
    auto edge = [](float ax, float ay, float bx, float by, float px, float py) {
        return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
    };
    const float area = edge(x0, y0, x1, y1, x2, y2);
    if (fabsf(area) < 1e-6f) return;   // degenerate
    const float s = (area < 0.0f) ? -1.0f : 1.0f;   // normalize winding
    for (int py = miny; py < maxy; ++py) {
        const float sy = py + 0.5f;
        const int base = py * kReflMaskW;
        for (int px = minx; px < maxx; ++px) {
            const float sx = px + 0.5f;
            const float w0 = edge(x1, y1, x2, y2, sx, sy) * s;
            const float w1 = edge(x2, y2, x0, y0, sx, sy) * s;
            const float w2 = edge(x0, y0, x1, y1, sx, sy) * s;
            if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
                const int bit = base + px;
                g_reflWaterMask[bit >> 6] |= (uint64_t(1) << (bit & 63));
            }
        }
    }
}

// Fill a convex quad given as 4 mask-cell float coords in tileScreenRect corner order
// (0=(-,-) 1=(+,-) 2=(-,+) 3=(+,+)); perimeter 0-1-3-2 → triangles (0,1,3),(0,3,2).
// Also sets the centroid cell so a sub-cell-thin quad (whose edges miss every pixel
// centre) still leaves a bit — no hole that would falsely cull an object over it.
static inline void reflMaskFillQuad(const float mx[4], const float my[4]) {
    reflMaskFillTri(mx[0], my[0], mx[1], my[1], mx[3], my[3]);
    reflMaskFillTri(mx[0], my[0], mx[3], my[3], mx[2], my[2]);
    const float cxf = 0.25f * (mx[0] + mx[1] + mx[2] + mx[3]);
    const float cyf = 0.25f * (my[0] + my[1] + my[2] + my[3]);
    // Range-compare (NaN-false) bounds the value so the int cast is always defined.
    if (cxf >= 0.0f && cxf < (float)kReflMaskW && cyf >= 0.0f && cyf < (float)kReflMaskH)
        reflMaskSetCell((int)cxf, (int)cyf);
}

// 1-frame hysteresis on per-water-tile visibility. A borderline horizon tile can flip
// occluded↔visible every frame (snapshot wobble / a box-occluder occasionally missing
// its drain), and because each tile's rect catches hundreds of reflection statics that
// makes the survivor count oscillate ~149↔576. Require a tile to read visible THIS
// frame AND the previous frame before it contributes a static-cull rect: single-frame
// visible blips are filtered, genuine sustained water passes (1 frame late). Keyed by
// quantized tile-centre world position (absolute, so a cell change needs no reset — the
// set is rebuilt every frame). Only gates rect emission; the anyVisible gate (terrain/
// sky reflection) still uses the raw per-frame verdict.
static std::unordered_set<int64_t> g_reflWaterVisLast;
static std::unordered_set<int64_t> g_reflWaterVisThis;

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

// [REFL PIPE] accessor — defined outside the anonymous namespace so it satisfies the
// DistantLand member declaration; reads the file-static counters filled by the
// reflection cull (worker) for the main-thread diagnostic line.
void DistantLand::getReflPipeDiag(int& tilesTested, int& waterPresent, int& waterOccluded,
                                  int& rects, int& queried, int& survivors, bool& msocUsable) {
    tilesTested   = g_reflPipe.tilesTested;
    waterPresent  = g_reflPipe.waterPresent;
    waterOccluded = g_reflPipe.waterOccluded;
    rects         = g_reflPipe.rects;
    queried       = g_reflPipe.queried;
    survivors     = g_reflPipe.survivors;
    msocUsable    = g_reflPipe.msocUsable;
}

// [REFL PIPE FLIP] accessor — farness/tiltedness of the tiles that flipped this frame.
void DistantLand::getReflPipeFlip(int& flips, float& distMinCells, float& distMaxCells,
                                  float& elevMinDeg, float& elevMaxDeg) {
    flips        = g_reflPipe.flips;
    distMinCells = g_reflPipe.flipDistMinCells;
    distMaxCells = g_reflPipe.flipDistMaxCells;
    elevMinDeg   = g_reflPipe.flipElevMinDeg;
    elevMaxDeg   = g_reflPipe.flipElevMaxDeg;
}

// True when this frame's verdict pass was dispatched to the worker (set by
// signalCullFinish, consumed by the wait/finish helpers). Main-thread only.
static bool s_cullOnWorker = false;



// renderSky - Render atmosphere scattering sky layer and other recorded draw calls on top
void DistantLand::renderSky() {
    MGE_ZoneScopedN("renderSky");
    DrawStats::ScopedStage _ds(DrawStats::Sky);
    // Recorded renders
    const auto& recordSky_const = recordSky;
    const int standardCloudVerts = 65, standardCloudTris = 112;
    const int standardMoonVerts = 4, standardMoonTris = 2;

    // Render sky without clouds first
    effect->BeginPass(PASS_RENDERSKY);
    for (const auto& i : recordSky_const) {
        // Skip clouds
        if (i.texture && i.vertCount == standardCloudVerts && i.primCount == standardCloudTris) {
            continue;
        }

        // Set variables in main effect; variables are shared via effect pool
        effect->SetTexture(ehTex0, i.texture);
        if (i.texture) {
            // Textured object; draw as normal in shader, with exceptions:
            // - Sun/moon billboards do not use mipmaps
            // - Moon shadow cutout (prevents stars shining through moons)
            //   which requires colour to be replaced with atmosphere scattering colour
            bool isBillboard = (i.vertCount == standardMoonVerts && i.primCount == standardMoonTris);
            bool isMoonShadow = i.destBlend == D3DBLEND_INVSRCALPHA && !i.useLighting;

            effect->SetBool(ehHasAlpha, true);
            effect->SetBool(ehHasBones, isBillboard);
            effect->SetBool(ehHasVCol, isMoonShadow);
            device->SetRenderState(D3DRS_ALPHABLENDENABLE, 1);
            device->SetRenderState(D3DRS_SRCBLEND, i.srcBlend);
            device->SetRenderState(D3DRS_DESTBLEND, i.destBlend);
            device->SetRenderState(D3DRS_ALPHATESTENABLE, 1);
        } else {
            // Sky; perform atmosphere scattering in shader
            effect->SetBool(ehHasAlpha, false);
            effect->SetBool(ehHasVCol, true);
            device->SetRenderState(D3DRS_ALPHABLENDENABLE, 0);
            device->SetRenderState(D3DRS_ALPHATESTENABLE, 0);
        }

        effect->SetMatrix(ehWorld, &i.worldTransforms[0]);
        effect->CommitChanges();

        device->SetStreamSource(0, i.vb, i.vbOffset, i.vbStride);
        device->SetIndices(i.ib);
        device->SetFVF(i.fvf);
        DrawStats::count(i.primCount);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);
    }
    effect->EndPass();

    // Render clouds with a separate shader
    effect->BeginPass(PASS_RENDERCLOUDS);
    for (const auto& i : recordSky_const) {
        // Clouds only
        if (!(i.texture && i.vertCount == standardCloudVerts && i.primCount == standardCloudTris)) {
            continue;
        }

        effect->SetTexture(ehTex0, i.texture);
        effect->SetBool(ehHasAlpha, true);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, 1);
        device->SetRenderState(D3DRS_SRCBLEND, i.srcBlend);
        device->SetRenderState(D3DRS_DESTBLEND, i.destBlend);
        device->SetRenderState(D3DRS_ALPHATESTENABLE, 1);
        effect->SetMatrix(ehWorld, &i.worldTransforms[0]);
        effect->CommitChanges();

        device->SetStreamSource(0, i.vb, i.vbOffset, i.vbStride);
        device->SetIndices(i.ib);
        device->SetFVF(i.fvf);
        DrawStats::count(i.primCount);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);
    }
    effect->EndPass();
}

void DistantLand::renderDistantLand(ID3DXEffect* e, const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    MGE_SCOPED_TIMER("renderDistantLand");
    DrawStats::ScopedStage _ds(DrawStats::Land);
    D3DXMATRIX world, viewproj = (*view) * (*proj);
    D3DXVECTOR4 viewsphere(eyePos.x, eyePos.y, eyePos.z, Configuration.DL.DrawDist * kCellSize);

    // Cull and draw
    ViewFrustum frustum(&viewproj);

    if (Configuration.UseSharedMemory) {
        // kick the operation off early so we can do some additional work while it runs
        visLandShared.RemoveAll();
        ipcClient.getVisibleMeshes(visLandSharedId, frustum, viewsphere, VIS_LAND);
    }

    D3DXMatrixIdentity(&world);
    effect->SetMatrix(ehWorld, &world);

    effect->SetTexture(ehTex0, texWorldColour);
    effect->SetTexture(ehTex1, texWorldNormals);
    effect->SetTexture(ehTex2, texWorldDetail);
    e->CommitChanges();

    if (!Configuration.UseSharedMemory) {
        visLand.RemoveAll();
        DistantLandShare::LandQuadTree.GetVisibleMeshes(frustum, viewsphere, visLand);
    }

    device->SetVertexDeclaration(LandDecl);

    if (Configuration.UseSharedMemory) {
        visLandShared.Render(device, SIZEOFLANDVERT, true);
    } else {
        visLand.Render(device, SIZEOFLANDVERT);
    }

    // Visible land-tile count diagnostic.
    // Visible land-tile count diagnostic. Gated by LogDistantPipeline.
    // Logged AFTER Render, because in IPC mode getVisibleMeshes is
    // async and the size isn't unpacked until the parallel-read pass
    // inside Render. Sampling before that returns 0 regardless of
    // how many tiles are actually visible.
    if (Configuration.LogDistantPipeline) {
        static int diagFrameCounter = 0;
        if ((diagFrameCounter++ % 60) == 0) {
            const unsigned tiles = Configuration.UseSharedMemory
                ? (unsigned)visLandShared.Size()
                : (unsigned)visLand.Size();
            LOG::logline("-- DL land: tiles=%u", tiles);
        }
    }

    // Horizon-curtain occluder contribution lives in renderStage0 (the
    // main exterior render path), NOT here, because renderDistantLand
    // also fires for the water-reflection and shadow-cast passes with
    // different view matrices. The contribution always projects through
    // mwView, so running it once per frame from the main path is enough
    // — running it from here too would submit duplicate curtains.
}

// Horizon-curtain terrain occluder.
//
// Instead of rasterizing raw terrain triangles (which over-culls statics
// sitting ON the hill surface, because MOC's min-depth test on a filled
// hill-top volume buries anything with a nearer center), we build a 1D
// screen-space silhouette and emit a ~120-triangle "curtain" that hangs
// from the silhouette to the screen bottom at the terrain's FAR depth.
//
// Why this is conservative in the right direction: a static on top of a
// hill projects ABOVE the silhouette (y > h[c]) so is never tested
// against the curtain. A static behind the hill projects BELOW the
// silhouette and has center depth < terrain-far, so it's correctly
// reported OCCLUDED.
//
// Algorithm: reference impl in terrain_horizon_occluder.{h,cpp} (dropped
// from D:/Modding/horizon/). We drive the "horizon build" phase ourselves
// by projecting each cached tile's triangles through the current view-
// projection and feeding `thc_horizon_test_and_update` once per triangle,
// then call the reference's simplify + emit, fix up the vertex layout for
// MOC's consumption, and submit via mwse_addPreTransformedOccluder.
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

        // Materialize the reflection set out of its IPC window NOW, while the
        // worker still owns the channel (before channelDrained releases it to the
        // main thread). Preserves the flicker fix's "materialize before release"
        // invariant; main never touches the live reflection window.
        if (DistantLand::reflStaticsWanted) {
            DistantLand::materializeReflectionMeshes();
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

        // Reflection gate + survivor cull: pure CPU over the materialized
        // reflection set and the frame-stable terrain/MSOC inputs. Produces
        // reflVisible + reflectionSurvivors, joined by main at the reflDone fence
        // just before renderWaterReflection.
        DistantLand::workerReflectionGateAndMask();

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
static std::atomic<bool> g_flowRebake{false};
static std::atomic<int> g_flowDebugBake{0};  // 0 flow, 1 CLASSIFY colours, 2 DIRECTION (flow-angle hue)
// Classification is GEOMETRIC (by water-body thinness), not depth-based: a river is narrow
// water (land close on every side); the sea is wide; the beach is the narrow fringe of the
// sea. RiverWidth = how many cells from shore still counts as "narrow"; BeachReach = how many
// cells out from open (wide) water stays beach before becoming sea.
static float g_flowRiverWidth   = 3.5f;    // max cells-to-nearest-land for water to be a river
static float g_flowBeachReach   = 10.0f;   // cells from open water that remain beach (else sea)
static float g_flowBeachExtend  = 30.0f;   // cells to grow beach OUTWARD into open sea (SEA→BEACH only)
static float g_flowBeachMul     = 0.5f;    // beach wave intensity
static float g_flowRiverInten   = 0.7f;    // river/inlet base wave intensity
static float g_flowDirRadius    = 40.0f;   // box-blur radius (cells) for the macro coastline normal
static float g_flowDirIters     = 3.0f;    // box-blur iterations (≈Gaussian) for beach direction
// River foam de-block: SAME separable box blur as the beach direction above, run on a binary
// river mask so the 512u square category source becomes a smooth continuous strength ramp.
static float g_flowRiverBlurRadius = 1.0f; // box-blur radius (cells) for the river foam ramp
static float g_flowRiverBlurIters  = 1.0f; // box-blur iterations (≈Gaussian) for the river ramp
static float g_flowRiverGain       = 4.0f; // post-blur gain: pushes narrow-river cores back to full
static float g_flowDilate          = 3.0f; // expand baked field this many cells past the shore (FlowExpand)

void DistantLand::updateMSOCCutoffInput() {
    // In-world overlays are compacted onto one cycling key (numpad +). One press
    // advances which single overlay is active; we derive the per-overlay
    // capture/render bools here. Read on the main thread BEFORE the cull worker
    // dispatch so a press this frame is captured this frame (the worker fills the
    // water-proxy / basin / box debug buffers gated on these bools).
    if (GetAsyncKeyState(VK_ADD) & 0x0001) {
        debugOverlayCycle = (debugOverlayCycle + 1) % 6;
        static const char* const names[6] = {
            "OFF", "Water proxy boxes", "Box occluders", "Basin watershed",
            "MSOC basin boxes", "Reflection frustum + cache" };
        char msg[80];
        std::snprintf(msg, sizeof(msg), "Overlay [+]: %s", names[debugOverlayCycle]);
        StatusOverlay::setStatus(msg);
    }
    g_drawWaterProxyBounds = (debugOverlayCycle == 1);
    boxOccluderDebug       = (debugOverlayCycle == 2);
    g_drawBasinDebug       = (debugOverlayCycle == 3);
    g_drawMSOCBasinBounds  = (debugOverlayCycle == 4);
    debugReflFrustum       = (debugOverlayCycle == 5);
    // Cleared each frame; renderReflectionsFromCache re-validates it only if a
    // water reflection actually runs (else the overlay draws no frustum).
    reflDbgValid = false;

    // Numpad1: A/B toggle the stage-2 reflection fine-mask (near-grid raster + mask test).
    // OFF → rect-only reflection cull (pre-mask behavior); for bisecting crashes.
    if (GetAsyncKeyState(VK_NUMPAD1) & 0x0001) {
        g_reflFineMaskEnabled = !g_reflFineMaskEnabled;
        StatusOverlay::setStatus(g_reflFineMaskEnabled
            ? "Reflection fine mask: ON" : "Reflection fine mask: OFF (rect-only)");
    }
    // Numpad0: A/B toggle the world-snapped LOD water mesh (clipmap) vs the legacy
    // camera-locked radial fan. Clipmap = vertices on a stable world lattice (no
    // swimming) + flow-steered 3D crests; radial = the pre-LOD look for comparison.
    if ((GetAsyncKeyState(VK_NUMPAD0) & 0x0001) && Configuration.UseWaterFlowMap) {
        waterLodMeshOn = !waterLodMeshOn;
        StatusOverlay::setStatus(waterLodMeshOn
            ? "Water mesh: LOD clipmap (world-snapped, 3D crests)"
            : "Water mesh: radial fan (legacy)");
    }
    // Numpad9: cycle the water flow map mode. OFF → neutral (today's uniform water look);
    // FLOW → per-body directional waves + storm-calm ponds; CLASSIFY → flat category colours
    // (green=river, red=pond, yellow=beach, blue=sea); DIRECTION → flow-angle hue wheel (rivers
    // downstream, beaches onshore). Each mode re-bakes the map into the matching encoding.
    if ((GetAsyncKeyState(VK_NUMPAD9) & 0x0001) && Configuration.UseWaterFlowMap) {
        // Cycle of water-flow debug views. Two tiers:
        //   BAKED source views (2..7): the flow texture is re-baked into a debug encoding, so the
        //     flow/foam effect is OFF (the texture no longer holds real flow) — inspect the data.
        //   SHADER views (8..11): NORMAL bake + effect ON, the shader overlays an in-pipeline value
        //     (filtered rmask/fdir, the fbm, the eroded far foam) at its last line — foam KEPT live.
        static const int NMODES = 12;
        static int mode = 1;
        mode = (mode + 1) % NMODES;
        // per-mode: bake encoding id (0 = normal flow), effect (flow weight) on?
        static const int  bakeId[NMODES]  = { 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0 };
        static const bool effectOn[NMODES] = { false, true, false, false, false, false, false, false,
                                               true, true, true, true };
        waterFlowDebugOn  = effectOn[mode];
        waterFlowDebugView = (mode >= 2) ? mode : 0;            // shader overlay id (0 = none)
        g_flowDebugBake.store(bakeId[mode], std::memory_order_release);
        g_flowRebake.store(true, std::memory_order_release);   // re-bake into the new encoding
        static const char* const nm[NMODES] = {
            "Water flow: OFF (neutral)",
            "Water flow: FLOW",
            "Water flow: CLASSIFY (grn river / red pond / yel beach / blu sea)",
            "Water flow: DIRECTION (hue = flow angle)  [baked, effect off]",
            "Water flow: STRENGTH (near river/beach routing, white=full)  [baked, effect off]",
            "Water flow: INTENSITY (wave amplitude)  [baked, effect off]",
            "Water flow: DIST RAW (banded dist-to-sea; jagged bands = quantized)  [baked, effect off]",
            "Water flow: DIST SMOOTH (banded blurred dist; smooth bands = de-blocked)  [baked, effect off]",
            "Water flow: SH RMASK (shader filtered river/beach strength)  [effect ON]",
            "Water flow: SH FDIR (shader filtered flow direction; R=x G=y)  [effect ON]",
            "Water flow: SH FBM (ridged foam noise)  [effect ON]",
            "Water flow: SH FARFOAM (final eroded far foam)  [effect ON]" };
        StatusOverlay::setStatus(nm[mode]);
    }
    // Numpad2: A/B toggle the world-anchored particle foam (sim + render). OFF leaves
    // the water identical to the no-foam look (the sim is also skipped, saving its cost).
    // Foam is always on (gated only by UseWaterFlowMap). NUMPAD2 now toggles the foam
    // debug panels (raw particle/field buffers blitted to the screen corner) instead of
    // the old A/B foam toggle, which was a confusion point during sim bring-up.
    if ((GetAsyncKeyState(VK_NUMPAD2) & 0x0001) && Configuration.UseWaterFlowMap) {
        foamDebugView = !foamDebugView;
        StatusOverlay::setStatus(foamDebugView
            ? "Foam debug panels: ON (particles | field)"
            : "Foam debug panels: OFF");
    }
    // Flow-map knob tuning. NUMPAD8 cycles which knob is selected; NUMPAD6/NUMPAD3
    // increase/decrease it. "Scroll" is a live shader uniform (instant). The other
    // four are baked into the flow map, so adjusting them sets g_flowRebake and the
    // cull worker re-bakes on the next water view.
    if (Configuration.UseWaterFlowMap) {
        static int knob = 0;  // 0 RiverSpeed 1 SeaSpeed 2 RiverWidth 3 RiverInten 4 BeachMul 5 BeachReach 6 BeachExtend 7 DirRadius 8 DirIters 9 CycleUV 10 SeaRefract 11 WaveAmp 12 WaveLen 13 WaveSpeed 14 CrestSpread 15 FoamForce 16 FoamDecay 17 FoamPress 18 FoamScale 19 DetTile 20 FoamSpeed 21 ErodeThr 22 FarAmt 23 RiverBlurR 24 RiverBlurIt 25 RiverGain 26 FlowWarp 27 FlowExpand 28 FoamMix 29 FoamGauss 30 FoamDens 31 FoamUVDcy 32 SimSpeed 33 VortGain 34 FineScale 35 FineAmt 36 CoarseScale 37 CoarseAmt
        static const char* const knobName[38] = {
            "RiverSpeed", "SeaSpeed", "RiverWidth", "RiverInten", "BeachMul", "BeachReach", "BeachExtend",
            "DirRadius", "DirIters", "CycleUV", "SeaRefract", "WaveAmp", "WaveLen", "WaveSpeed", "CrestSpread",
            "FoamForce", "FoamDecay", "FoamPress", "FoamScale",         // carrier sim tuning
            "DetTile", "FoamSpeed", "ErodeThr", "FarAmt",             // two-layer foam (FoamSpeed = far advect ×river flow)
            "RiverBlurR", "RiverBlurIt", "RiverGain",                // baked river-foam box blur (mirrors DirRadius/DirIters)
            "FlowWarp", "FlowExpand",                                // FlowWarp = shader grid-break; FlowExpand = bake shore dilation
            "FoamMix",                                               // near/sim → far modulation (0 far only, 1 sim drives)
            "FoamGauss", "FoamDens", "FoamUVDcy",                    // XE Mod Foam sim: blob size, particle density, UV-stretch decay
            "SimSpeed", "VortGain",                                  // SimSpeed = sim advance; VortGain = static curl concentration (far)
            "FineScale", "FineAmt",                                  // multi-scale foam erosion (perforating octave)
            "CoarseScale", "CoarseAmt" };                            // multi-scale foam erosion (clumping octave)
        const bool inc    = (GetAsyncKeyState(VK_NUMPAD6) & 0x0001) != 0;
        const bool dec    = (GetAsyncKeyState(VK_NUMPAD3) & 0x0001) != 0;
        const bool cycle  = (GetAsyncKeyState(VK_NUMPAD8) & 0x0001) != 0;
        const bool fineTgl= (GetAsyncKeyState(VK_MULTIPLY) & 0x0001) != 0;   // numpad * = fine-tune toggle
        static bool fine = false;
        if (fineTgl) fine = !fine;
        if (cycle) {
            knob = (knob + 1) % 38;
            fine = false;                               // each new knob starts at the coarse step
        }
        if (inc || dec) {
            const float fineK = fine ? 0.25f : 1.0f;            // additive steps → quarter in fine mode
            const float s = (inc ? 1.0f : -1.0f) * fineK;
            const float mB = fine ? 1.12f : 1.5f;              // gentler multiplicative step in fine mode
            const float m = inc ? mB : 1.0f / mB;
            bool rebake = true;
            switch (knob) {
            case 0:  // RiverSpeed: directional advection rate, live (no rebake)
                waterFlowScroll = std::max(0.001f, std::min(1.0f, waterFlowScroll * m));
                rebake = false;
                break;
            case 1:  // SeaSpeed: base wave animation rate, live (no rebake)
                waterFlowSeaSpeed = std::max(0.05f, std::min(8.0f, waterFlowSeaSpeed * m));
                rebake = false;
                break;
            case 2:  g_flowRiverWidth   = std::max(0.5f, std::min(16.0f, g_flowRiverWidth + s * 0.5f)); break;
            case 3:  g_flowRiverInten   = std::max(0.0f, std::min(3.0f, g_flowRiverInten + s * 0.10f)); break;
            case 4:  g_flowBeachMul     = std::max(0.0f, std::min(1.0f, g_flowBeachMul   + s * 0.10f)); break;
            case 5:  g_flowBeachReach   = std::max(0.0f, std::min(16.0f, g_flowBeachReach + s * 0.5f)); break;
            case 6:  g_flowBeachExtend  = std::max(0.0f, std::min(32.0f, g_flowBeachExtend + s * 0.5f)); break;
            case 7:  g_flowDirRadius    = std::max(1.0f, std::min(64.0f, g_flowDirRadius + s * 2.0f)); break;
            case 8:  g_flowDirIters     = std::max(1.0f, std::min(8.0f, g_flowDirIters + s * 1.0f)); break;
            case 9:  // CycleUV: bounded per-cycle UV displacement (bigger = longer cycle, less pulsing), live
                waterFlowCycleUV = std::max(0.02f, std::min(4.0f, waterFlowCycleUV * m));
                rebake = false;
                break;
            case 10: // SeaRefract: sea far-wave (refraction) strength multiplier, live (no rebake)
                waterFlowSeaRefract = std::max(0.0f, std::min(4.0f, waterFlowSeaRefract + s * 0.25f));
                rebake = false;
                break;
            case 11: // WaveAmp: 3D crest amplitude (world units), live (no rebake)
                waterWaveAmp = std::max(0.0f, std::min(64.0f, waterWaveAmp + s * 2.0f));
                rebake = false;
                break;
            case 12: // WaveLen: base wavelength along the flow (world units), live (no rebake)
                waterWaveLen = std::max(64.0f, std::min(4096.0f, waterWaveLen * m));
                rebake = false;
                break;
            case 13: // WaveSpeed: crest travel speed scale, live (no rebake)
                waterWaveSpeed = std::max(0.05f, std::min(8.0f, waterWaveSpeed * m));
                rebake = false;
                break;
            case 14: // CrestSpread: directional fan (small = longer crest lines), live (no rebake)
                waterCrestSpread = std::max(0.0f, std::min(1.0f, waterCrestSpread + s * 0.05f));
                rebake = false;
                break;
            case 15: // FoamForce: carrier river advection force, live (no rebake)
                foamFlowForce[0] = std::max(0.0f, std::min(8.0f, foamFlowForce[0] + s * 0.25f));
                rebake = false;
                break;
            case 16: // FoamDecay: carrier velocity decay toward the flow current, live
                foamDecay[0] = std::max(0.5f, std::min(0.99f, foamDecay[0] + s * 0.01f));
                rebake = false;
                break;
            case 17: // FoamPress: carrier density pile-up coefficient (narrows), live
                foamPressure[0] = std::max(0.0f, std::min(2.0f, foamPressure[0] + s * 0.05f));
                rebake = false;
                break;
            case 18: // FoamScale: carrier vorticity → foam intensity scale, live
                foamScale[0] = std::max(0.0f, std::min(400.0f, foamScale[0] * m));
                rebake = false;
                break;
            case 19: // DetTile: indirection fbm cell size (world units, smaller = crisper), live
                foamDetailTile = std::max(16.0f, std::min(4096.0f, foamDetailTile * m));
                rebake = false;
                break;
            case 20: // FoamSpeed: foam detail scroll speed as a multiplier on river flow (1 = match), live
                foamDetailSpeed = std::max(0.0f, std::min(8.0f, foamDetailSpeed + s * 0.1f));
                rebake = false;
                break;
            case 21: // ErodeThr: shared erosion cut (higher = tighter foam streaks), live
                foamErodeThreshold = std::max(0.0f, std::min(1.0f, foamErodeThreshold + s * 0.02f));
                rebake = false;
                break;
            case 22: // FarAmt: far flow-map layer strength (0 = near-only), live
                foamFarAmount = std::max(0.0f, std::min(2.0f, foamFarAmount + s * 0.1f));
                rebake = false;
                break;
            case 23: // RiverBlurR: river-foam box-blur radius (cells), mirrors DirRadius → rebake
                g_flowRiverBlurRadius = std::max(1.0f, std::min(64.0f, g_flowRiverBlurRadius + s * 2.0f));
                break;
            case 24: // RiverBlurIt: river-foam box-blur iterations, mirrors DirIters → rebake
                g_flowRiverBlurIters  = std::max(1.0f, std::min(8.0f, g_flowRiverBlurIters + s * 1.0f));
                break;
            case 25: // RiverGain: post-blur gain (narrow-river core strength) → rebake
                g_flowRiverGain       = std::max(0.5f, std::min(16.0f, g_flowRiverGain + s * 0.5f));
                break;
            case 26: // FlowWarp: shader domain-warp amount (world u), live (no rebake)
                waterFlowWarp = std::max(0.0f, std::min(1024.0f, waterFlowWarp + s * 64.0f));
                rebake = false;
                break;
            case 27: // FlowExpand: dilate baked field past the shore (cells) → rebake
                g_flowDilate = std::max(0.0f, std::min(16.0f, g_flowDilate + s * 1.0f));
                break;
            case 28: // FoamMix: near/sim → far modulation strength, live (no rebake)
                foamMix = std::max(0.0f, std::min(1.0f, foamMix + s * 0.1f));
                rebake = false;
                break;
            case 29: // FoamGauss: sim particle splat radius → foam blob size, live
                foamGaussRadius = std::max(0.5f, std::min(8.0f, foamGaussRadius + s * 0.5f));
                rebake = false;
                break;
            case 30: // FoamDens: sim particle respawn density, live
                foamMinDensity = std::max(0.2f, std::min(3.0f, foamMinDensity + s * 0.1f));
                rebake = false;
                break;
            case 31: // FoamUVDcy: sim UV-offset decay (bounds detail stretch), live
                foamUVDecay = std::max(0.5f, std::min(0.999f, foamUVDecay + s * 0.01f));
                rebake = false;
                break;
            case 32: // SimSpeed: sim particle advance rate (slow near foam to match far), live
                foamSimSpeed = std::max(0.05f, std::min(2.0f, foamSimSpeed + s * 0.1f));
                rebake = false;
                break;
            case 33: // VortGain: static vorticity (curl) concentration for the far foam, live
                foamVortGain = std::max(0.0f, std::min(64.0f, foamVortGain + s * 1.0f));
                rebake = false;
                break;
            case 34: // FineScale: multi-scale erosion perforating-octave ratio, live
                foamFineScale = std::max(1.5f, std::min(16.0f, foamFineScale + s * 0.5f));
                rebake = false;
                break;
            case 35: // FineAmt: multi-scale fine-perforation strength, live
                foamFineAmt = std::max(0.0f, std::min(1.0f, foamFineAmt + s * 0.1f));
                rebake = false;
                break;
            case 36: // CoarseScale: multi-scale erosion clumping-octave ratio, live
                foamCoarseScale = std::max(1.5f, std::min(16.0f, foamCoarseScale + s * 0.5f));
                rebake = false;
                break;
            case 37: // CoarseAmt: multi-scale coarse-clumping strength, live
                foamCoarseAmt = std::max(0.0f, std::min(1.0f, foamCoarseAmt + s * 0.1f));
                rebake = false;
                break;
            }
            if (rebake) g_flowRebake.store(true, std::memory_order_release);
        }
        // Report the current selection/value whenever a tuning key is pressed (incl. the * fine toggle).
        if (cycle || inc || dec || fineTgl) {
            const float val[38] = { waterFlowScroll, waterFlowSeaSpeed, g_flowRiverWidth,
                                   g_flowRiverInten, g_flowBeachMul, g_flowBeachReach, g_flowBeachExtend,
                                   g_flowDirRadius, g_flowDirIters, waterFlowCycleUV, waterFlowSeaRefract,
                                   waterWaveAmp, waterWaveLen, waterWaveSpeed, waterCrestSpread,
                                   foamFlowForce[0], foamDecay[0], foamPressure[0], foamScale[0],
                                   foamDetailTile, foamDetailSpeed, foamErodeThreshold, foamFarAmount,
                                   g_flowRiverBlurRadius, g_flowRiverBlurIters, g_flowRiverGain,
                                   waterFlowWarp, g_flowDilate, foamMix,
                                   foamGaussRadius, foamMinDensity, foamUVDecay, foamSimSpeed, foamVortGain,
                                   foamFineScale, foamFineAmt,
                                   foamCoarseScale, foamCoarseAmt };
            char msg[112];
            std::snprintf(msg, sizeof(msg), "Flow [8] %s = %.3f  (6 +/ 3 -)  [* %s]",
                          knobName[knob], val[knob], fine ? "FINE" : "coarse");
            StatusOverlay::setStatus(msg);
        }
    }
    // (Retired debug hot keys: MSOC cutoff height (NUMPAD8/2), basin eye lock
    // (NUMPAD0), basin barrier grid (NUMPAD / and *). Those values are settled;
    // g_msocCutoffHeight / g_basinLockEye / g_basinGrid keep their defaults.
    // NUMPAD8 is now the flow-map knob cycle above.)
}

// ImGui Water Flow / Foam panel (F10). Defined here — not in imgui_water.cpp — so it can reach
// every setting: DistantLand:: public statics AND the file-static g_flow* bake knobs + rebake.
// Baked sliders re-bake once on release (IsItemDeactivatedAfterEdit), like the NUMPAD8 knobs.
void DrawFlowFoamPanel() {
    ImGui::SetNextWindowSize(ImVec2(370, 0), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Water Flow / Foam")) { ImGui::End(); return; }

    auto baked = [](const char* label, float* v, float lo, float hi, const char* fmt) {
        ImGui::SliderFloat(label, v, lo, hi, fmt);
        if (ImGui::IsItemDeactivatedAfterEdit())
            g_flowRebake.store(true, std::memory_order_release);
    };

    if (ImGui::CollapsingHeader("Classify  (rebake)", ImGuiTreeNodeFlags_DefaultOpen)) {
        baked("RiverWidth",  &g_flowRiverWidth,  0.5f, 16.0f, "%.1f");
        baked("RiverInten",  &g_flowRiverInten,  0.0f, 3.0f,  "%.2f");
        baked("BeachMul",    &g_flowBeachMul,    0.0f, 1.0f,  "%.2f");
        baked("BeachReach",  &g_flowBeachReach,  0.0f, 16.0f, "%.1f");
        baked("BeachExtend", &g_flowBeachExtend, 0.0f, 32.0f, "%.1f");
        baked("DirRadius",   &g_flowDirRadius,   1.0f, 64.0f, "%.0f");
        baked("DirIters",    &g_flowDirIters,    1.0f, 8.0f,  "%.0f");
        baked("FlowExpand",  &g_flowDilate,      0.0f, 16.0f, "%.0f");
        baked("RiverBlurR",  &g_flowRiverBlurRadius, 1.0f, 64.0f, "%.0f");
        baked("RiverBlurIt", &g_flowRiverBlurIters,  1.0f, 8.0f,  "%.0f");
        baked("RiverGain",   &g_flowRiverGain,   0.5f, 16.0f, "%.2f");
        if (ImGui::Button("Rebake now"))
            g_flowRebake.store(true, std::memory_order_release);
    }
    if (ImGui::CollapsingHeader("Animation  (live)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("RiverSpeed",  &DistantLand::waterFlowScroll,    0.001f, 1.0f, "%.3f");
        ImGui::SliderFloat("SeaSpeed",    &DistantLand::waterFlowSeaSpeed,  0.05f, 8.0f, "%.2f");
        ImGui::SliderFloat("CycleUV",     &DistantLand::waterFlowCycleUV,   0.02f, 4.0f, "%.2f");
        ImGui::SliderFloat("SeaRefract",  &DistantLand::waterFlowSeaRefract,0.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("WaveAmp",     &DistantLand::waterWaveAmp,       0.0f, 64.0f, "%.1f");
        ImGui::SliderFloat("WaveLen",     &DistantLand::waterWaveLen,       64.0f, 4096.0f, "%.0f");
        ImGui::SliderFloat("WaveSpeed",   &DistantLand::waterWaveSpeed,     0.05f, 8.0f, "%.2f");
        ImGui::SliderFloat("CrestSpread", &DistantLand::waterCrestSpread,   0.0f, 1.0f, "%.2f");
    }
    if (ImGui::CollapsingHeader("Far foam  (live)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("DetTile",   &DistantLand::foamDetailTile,     16.0f, 4096.0f, "%.0f");
        ImGui::SliderFloat("FoamSpeed", &DistantLand::foamDetailSpeed,    0.0f, 8.0f, "%.2f");
        ImGui::SliderFloat("ErodeThr",  &DistantLand::foamErodeThreshold, 0.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("FarAmt",    &DistantLand::foamFarAmount,      0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("FoamMix",   &DistantLand::foamMix,            0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("VortGain",  &DistantLand::foamVortGain,       0.0f, 64.0f, "%.1f");
        ImGui::SliderFloat("FineScale", &DistantLand::foamFineScale,      1.5f, 16.0f, "%.1f");
        ImGui::SliderFloat("FineAmt",   &DistantLand::foamFineAmt,        0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("CoarseScale", &DistantLand::foamCoarseScale,  1.5f, 16.0f, "%.1f");
        ImGui::SliderFloat("CoarseAmt", &DistantLand::foamCoarseAmt,      0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("FlowWarp",  &DistantLand::waterFlowWarp,      0.0f, 1024.0f, "%.0f");
    }
    if (ImGui::CollapsingHeader("Sim foam  (live)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("FoamForce", &DistantLand::foamFlowForce[0], 0.0f, 8.0f, "%.2f");
        ImGui::SliderFloat("FoamDecay", &DistantLand::foamDecay[0],     0.5f, 0.99f, "%.2f");
        ImGui::SliderFloat("FoamPress", &DistantLand::foamPressure[0],  0.0f, 2.0f, "%.2f");
        ImGui::SliderFloat("FoamScale", &DistantLand::foamScale[0],     0.0f, 400.0f, "%.1f");
        ImGui::SliderFloat("FoamGauss", &DistantLand::foamGaussRadius,  0.5f, 8.0f, "%.2f");
        ImGui::SliderFloat("FoamDens",  &DistantLand::foamMinDensity,   0.2f, 3.0f, "%.2f");
        ImGui::SliderFloat("FoamUVDcy", &DistantLand::foamUVDecay,      0.5f, 0.999f, "%.3f");
        ImGui::SliderFloat("SimSpeed",  &DistantLand::foamSimSpeed,     0.05f, 2.0f, "%.2f");
    }
    if (ImGui::CollapsingHeader("Debug view")) {
        static const char* const views[12] = {
            "OFF", "FLOW", "CLASSIFY", "DIRECTION", "STRENGTH", "INTENSITY",
            "DIST RAW", "DIST SMOOTH", "SH RMASK", "SH FDIR", "SH FBM", "SH FARFOAM" };
        static const int  bakeId[12] = { 0,0,1,2,3,4,5,6,0,0,0,0 };
        static const bool effOn[12]  = { false,true,false,false,false,false,false,false,true,true,true,true };
        static int mode = 1;
        if (ImGui::Combo("view", &mode, views, 12)) {
            DistantLand::waterFlowDebugOn   = effOn[mode];
            DistantLand::waterFlowDebugView = (mode >= 2) ? mode : 0;
            g_flowDebugBake.store(bakeId[mode], std::memory_order_release);
            g_flowRebake.store(true, std::memory_order_release);
        }
    }
    ImGui::End();
}

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
void DistantLand::waitCullReflReady() {
    if (!s_cullOnWorker)
        return;
    MGE_ZoneScopedN("finish:joinReflMask");
    std::unique_lock<std::mutex> lk(g_cullMtx);
    g_cullCv.wait(lk, []{ return g_cullReflDone; });
}

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

        if (reflStaticsWanted) {
            ViewFrustum reflFrustum(&reflCullViewProj);
            ipcClient.getVisibleMeshesAllRanges(
                visDistantSharedId, 3, frustums, spheres, setFlags,
                VisibleSetSort::None,
                visExtraSharedId, VIS_STATIC, &reflFrustum, &reflCullViewSphere,
                VisibleSetSort::ByState, occlMaskId);
        } else {
            ipcClient.getVisibleMeshesAllRanges(
                visDistantSharedId, 3, frustums, spheres, setFlags,
                VisibleSetSort::None,
                IPC::InvalidVector, 0, nullptr, nullptr,
                VisibleSetSort::None, occlMaskId);
        }
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

void DistantLand::renderDistantStatics() {
    MGE_ZoneScopedN("renderDistantStatics");
    MGE_SCOPED_TIMER("renderDistantStatics");
    DrawStats::ScopedStage _ds(DrawStats::Statics);
    // Handover band near-cut clip plane is set by the caller (renderStage0), bracketed
    // with the distant projection these statics are drawn with so the view-z slab cuts
    // at the intended band start. (Was an interior-only mwProj clip here — moved out so
    // the plane matches distProj and covers exteriors too.)

    device->SetVertexDeclaration(StaticDecl);

    // Cull-then-sort: iterate the compacted, state-sorted survivor set built
    // by applyMSOCToDistantStatics (IPC and non-IPC paths alike). The old
    // skipMask plumbing is gone — occluded instances are already absent.
    visDistantSurvivors.Render(device, effect, effect, &ehTex0, nullptr, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT, false);
}

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
void DistantLand::renderWaterProxyBoundsDebug(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    DrawStats::ScopedStage _ds(DrawStats::Debug);
    // Toggle is now overlay-cycle state 1 (g_drawWaterProxyBounds derived in
    // updateMSOCCutoffInput).
    if (!g_drawWaterProxyBounds || g_waterProxyDebugBoxes.empty() || !view || !proj)
        return;

    IDirect3DStateBlock9* stateSaved = nullptr;
    if (FAILED(device->CreateStateBlock(D3DSBT_ALL, &stateSaved)) || !stateSaved)
        return;

    static std::vector<MSOCLineVertex> lineVerts;
    lineVerts.clear();
    lineVerts.reserve(g_waterProxyDebugBoxes.size() * 24);

    auto pushLine = [](float ax, float ay, float az,
                       float bx, float by, float bz,
                       DWORD color) {
        lineVerts.push_back({ax, ay, az, color});
        lineVerts.push_back({bx, by, bz, color});
    };

    for (const auto& b : g_waterProxyDebugBoxes) {
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

// Reflection-frustum + cache survivor overlay (overlay-cycle state 5). Drawn in
// the MAIN camera view: wireframes the reflection cull frustum (inverse of the
// stashed reflection view*proj — it sits BELOW the water mirror, looking up) plus
// every GeometryCache reflection candidate as a box, RED = drawn into the
// reflection now, GREEN = its mirrored sphere lands on a visible water rect (would
// survive a water-rect cull — the over-draw / fix preview). Mirrors
// renderMSOCBasinBoundsDebug's state save / LINELIST / restore; Z-test off so the
// below-water frustum and boxes behind buildings stay visible.
void DistantLand::renderReflectionFrustumDebug(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    DrawStats::ScopedStage _ds(DrawStats::Debug);
    if (!debugReflFrustum || !reflDbgValid || !view || !proj)
        return;

    IDirect3DStateBlock9* stateSaved = nullptr;
    if (FAILED(device->CreateStateBlock(D3DSBT_ALL, &stateSaved)) || !stateSaved)
        return;

    static std::vector<MSOCLineVertex> lineVerts;
    lineVerts.clear();
    lineVerts.reserve(reflCacheDbg.size() * 24 + 24);

    auto pushLine = [](float ax, float ay, float az, float bx, float by, float bz, DWORD color) {
        lineVerts.push_back({ax, ay, az, color});
        lineVerts.push_back({bx, by, bz, color});
    };
    auto pushWireBox = [&](float x0, float y0, float z0, float x1, float y1, float z1, DWORD color) {
        pushLine(x0,y0,z0, x1,y0,z0, color); pushLine(x1,y0,z0, x1,y1,z0, color);
        pushLine(x1,y1,z0, x0,y1,z0, color); pushLine(x0,y1,z0, x0,y0,z0, color);
        pushLine(x0,y0,z1, x1,y0,z1, color); pushLine(x1,y0,z1, x1,y1,z1, color);
        pushLine(x1,y1,z1, x0,y1,z1, color); pushLine(x0,y1,z1, x0,y0,z1, color);
        pushLine(x0,y0,z0, x0,y0,z1, color); pushLine(x1,y0,z0, x1,y0,z1, color);
        pushLine(x1,y1,z0, x1,y1,z1, color); pushLine(x0,y1,z0, x0,y1,z1, color);
    };

    // Reflection frustum: 8 world corners = inverse(reflDbgViewProj) of the D3D NDC
    // cube (x,y in [-1,1], z in [0,1]). near face yellow, far magenta, sides cyan.
    D3DXMATRIX invVP;
    if (D3DXMatrixInverse(&invVP, nullptr, &reflDbgViewProj)) {
        static const D3DXVECTOR3 ndc[8] = {
            {-1,-1,0},{ 1,-1,0},{ 1, 1,0},{-1, 1,0},   // near 0..3
            {-1,-1,1},{ 1,-1,1},{ 1, 1,1},{-1, 1,1}};  // far  4..7
        D3DXVECTOR3 c[8];
        for (int i = 0; i < 8; ++i) D3DXVec3TransformCoord(&c[i], &ndc[i], &invVP);
        const DWORD kYellow = D3DCOLOR_XRGB(255,255,0), kMagenta = D3DCOLOR_XRGB(255,0,255), kCyan = D3DCOLOR_XRGB(0,255,255);
        auto edge = [&](int a, int b, DWORD col){ pushLine(c[a].x,c[a].y,c[a].z, c[b].x,c[b].y,c[b].z, col); };
        edge(0,1,kYellow); edge(1,2,kYellow); edge(2,3,kYellow); edge(3,0,kYellow);     // near quad
        edge(4,5,kMagenta); edge(5,6,kMagenta); edge(6,7,kMagenta); edge(7,4,kMagenta); // far quad
        edge(0,4,kCyan); edge(1,5,kCyan); edge(2,6,kCyan); edge(3,7,kCyan);             // side edges
    }

    // Cache reflection candidates: RED = drawn now, GREEN = would survive water-rect cull.
    const DWORD kRed = D3DCOLOR_XRGB(255,64,64), kGreen = D3DCOLOR_XRGB(64,255,96);
    for (const ReflCacheDbgBox& b : reflCacheDbg) {
        const DWORD col = b.keep ? kGreen : kRed;
        pushWireBox(b.center.x - b.radius, b.center.y - b.radius, b.center.z - b.radius,
                    b.center.x + b.radius, b.center.y + b.radius, b.center.z + b.radius, col);
    }

    if (!lineVerts.empty()) {
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
    }

    // Stage-2 mask overlay: draw the rasterized water silhouette (g_reflWaterMask) as
    // translucent screen-space cells, so the fine raster the cache/statics cull tests
    // against is directly inspectable. XYZRHW (post-viewport), one quad per set bit.
    if (g_reflWaterMaskValid) {
        D3DVIEWPORT9 vp;
        if (SUCCEEDED(device->GetViewport(&vp))) {
            static std::vector<CurtainScreenVertex> mv;
            mv.clear();
            const float cellNdcW = 2.0f / (float)kReflMaskW;
            const float cellNdcH = 2.0f / (float)kReflMaskH;
            const DWORD maskColor = 0x600080FFu;   // ARGB ~38% alpha, cyan-blue water
            for (int cy = 0; cy < kReflMaskH; ++cy) {
                for (int cx = 0; cx < kReflMaskW; ++cx) {
                    const int bit = cy * kReflMaskW + cx;
                    if (!(g_reflWaterMask[bit >> 6] & (uint64_t(1) << (bit & 63)))) continue;
                    // Cell NDC AABB → pixel quad (NDC y up → pixel y down).
                    const float nx0 = -1.0f + cx * cellNdcW, nx1 = nx0 + cellNdcW;
                    const float ny0 = -1.0f + cy * cellNdcH, ny1 = ny0 + cellNdcH;
                    const float px0 = (nx0 * 0.5f + 0.5f) * (float)vp.Width;
                    const float px1 = (nx1 * 0.5f + 0.5f) * (float)vp.Width;
                    const float py0 = (1.0f - (ny1 * 0.5f + 0.5f)) * (float)vp.Height;
                    const float py1 = (1.0f - (ny0 * 0.5f + 0.5f)) * (float)vp.Height;
                    mv.push_back({ px0, py0, 0.5f, 1.0f, maskColor });
                    mv.push_back({ px1, py0, 0.5f, 1.0f, maskColor });
                    mv.push_back({ px1, py1, 0.5f, 1.0f, maskColor });
                    mv.push_back({ px0, py0, 0.5f, 1.0f, maskColor });
                    mv.push_back({ px1, py1, 0.5f, 1.0f, maskColor });
                    mv.push_back({ px0, py1, 0.5f, 1.0f, maskColor });
                }
            }
            if (!mv.empty()) {
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
                DrawStats::count((unsigned)(mv.size() / 3));
                device->DrawPrimitiveUP(D3DPT_TRIANGLELIST, (UINT)(mv.size() / 3),
                                        mv.data(), sizeof(CurtainScreenVertex));
            }
        }
    }

    stateSaved->Apply();
    stateSaved->Release();
}

// MSOC verdict pass — populates `msocOccluded` with a per-instance
// cull mask consumed by both the color and depth render paths.
//
// Walks the visible set, runs the batched sphere query, optionally
// escalates large statics to the OBB test, then applies far-distance
// gate / engine-handoff gate / temporal hysteresis to produce the
// final per-instance cull decision.
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

static void buildTerrainMinHeight(
    const std::unordered_map<IDirect3DVertexBuffer9*, DistantLand::LandMeshCache>& landMeshes) {
    g_cellMinH.clear();
    g_fineMinH.clear();
    g_tileTris.clear();
    g_tileTris.reserve(landMeshes.size());
    g_nearGrid.valid = false;   // region/source changed → force a near-grid rebuild
    g_nearGrid.builtCellX = g_nearGrid.builtCellY = INT_MIN;
    for (const auto& kv : landMeshes) {
        // Per-tile XY AABB for the near-grid triangle cull.
        float tminx = 1e30f, tminy = 1e30f, tmaxx = -1e30f, tmaxy = -1e30f;
        for (const D3DXVECTOR3& p : kv.second.positions) {
            auto upd = [](std::unordered_map<uint64_t, float>& m, uint64_t k, float z) {
                auto it = m.find(k);
                if (it == m.end()) m.emplace(k, z);
                else if (z < it->second) it->second = z;
            };
            upd(g_cellMinH, waterGridKey((int)floorf(p.x / DistantLand::kCellSize),
                                         (int)floorf(p.y / DistantLand::kCellSize)), p.z);
            upd(g_fineMinH, waterGridKey((int)floorf(p.x / kWaterFineGrid),
                                         (int)floorf(p.y / kWaterFineGrid)), p.z);
            tminx = std::min(tminx, p.x); tmaxx = std::max(tmaxx, p.x);
            tminy = std::min(tminy, p.y); tmaxy = std::max(tmaxy, p.y);
        }
        if (!kv.second.indices.empty() && tmaxx >= tminx)
            g_tileTris.push_back({ &kv.second, tminx, tminy, tmaxx, tmaxy });
    }
    g_terrainMinHSrcSize = landMeshes.size();
    LOG::logline("-- [water-gate] terrain min-height built: %zu cells, %zu fine tiles, %zu tiled meshes",
                 g_cellMinH.size(), g_fineMinH.size(), g_tileTris.size());
}

// ---- Water flow map (per-water-body directional waves & storm-calm ponds) ----
//
// Baked once on the cull worker. RASTERIZES the captured terrain triangles (g_tileTris)
// into a dense min-height grid — vertex-binning g_fineMinH leaves decimation holes that
// checkerboard the classification (sparse "squares" in rivers, dotted/black open sea).
// Cells with NO triangle coverage are deep OPEN SEA (water outside the terrain mesh) and
// are treated as full-intensity sea. Then: flood-fill wet cells into bodies, run a
// distance-to-sea BFS seeded from the open-sea cells, derive a per-cell flow field, and
// encode RGBA8 (R,G = downstream dir, B = wave intensity, A = directionality). The water
// shader advects the normal map downstream (rivers) and scales amplitude per body (calm
// ponds/shallows). The texture upload happens on the main thread.
//
// Channel semantics (see XE Mod Water.fx getFinalWaterNormal):
//   open sea / near-shore : dir 0, intensity 1, directionality 0  → unchanged look
//   river/inlet (dist>fr) : dir = -grad(dist) (downstream), intensity ~0.7, dir'ity 1
//   isolated lake/pond    : dir 0, intensity by cellCount (~0.2 pond .. 0.5 lake)
//   beach/shore (any)     : intensity *= kFlowBeachMul where adjacent to dry land / shallow
static std::vector<uint32_t> g_flowMapBytes;          // A8R8G8B8, W*H
static int   g_flowMapW = 0, g_flowMapH = 0;
static float g_flowOriginX = 0.0f, g_flowOriginY = 0.0f;   // world XY of grid corner (0,0)
static float g_flowInvSizeX = 0.0f, g_flowInvSizeY = 0.0f; // 1/(W*grid), 1/(H*grid)
static std::atomic<bool> g_flowMapDirty{false};            // worker → main upload signal

// Tunables (Open knobs — adjust at review).
// (Runtime-tunable knobs g_flow* + g_flowRebake are declared above
//  updateMSOCCutoffInput so the input handler can see them.)
static constexpr int   kFlowBlurPasses   = 1;       // light box-blur passes on intensity (0 = off)
// B channel = per-body 3D-wave amplitude mask (encoded /3, decoded *3 in WaterVS): pond
// dead calm, river ~0.5x, beach/default 1x, open sea 3x. The pixel shader reuses B as the
// surface normal strength (saturate(flow.b*3)) → calm bodies smoother, sea/beach choppier.
static constexpr float kFlowWaveNone     = 0.0f;        // pond/lake → 0 waves
static constexpr float kFlowWaveRiver    = 1.0f / 6.0f; // river → ~0.5x ("less in rivers")
static constexpr float kFlowWaveDefault  = 1.0f / 3.0f; // beach / default / uncovered → 1x
static constexpr float kFlowWaveSea      = 1.0f;        // open sea → 3x
static constexpr float kFlowSeaRefract   = 1.5f;    // sea far-wave (refraction) strength near coasts
static constexpr int64_t kFlowMaxCells   = 4 * 1024 * 1024; // sanity bound on W*H
static constexpr float kFlowGrid         = 512.0f;  // flow-map cell size (world units)

void DistantLand::buildWaterFlowMap(float waterZ) {
    g_flowMapW = g_flowMapH = 0;
    if (g_tileTris.empty()) {
        return;
    }

    // 1. World bbox over the captured terrain tiles → integer flow-grid bbox.
    float wminx = 1e30f, wminy = 1e30f, wmaxx = -1e30f, wmaxy = -1e30f;
    for (const TileTriEntry& t : g_tileTris) {
        wminx = std::min(wminx, t.minx); wmaxx = std::max(wmaxx, t.maxx);
        wminy = std::min(wminy, t.miny); wmaxy = std::max(wmaxy, t.maxy);
    }
    const int minFx = (int)floorf(wminx / kFlowGrid);
    const int minFy = (int)floorf(wminy / kFlowGrid);
    const int maxFx = (int)floorf(wmaxx / kFlowGrid);
    const int maxFy = (int)floorf(wmaxy / kFlowGrid);
    const int W = maxFx - minFx + 1;
    const int H = maxFy - minFy + 1;
    if (W <= 0 || H <= 0 || (int64_t)W * H > kFlowMaxCells) {
        LOG::logline("!! [water-flow] grid out of range (%dx%d) — flow map skipped", W, H);
        return;
    }
    const size_t N = (size_t)W * H;

    // 2. RASTERIZE terrain triangles into a dense min-height grid (continuous coverage,
    //    no decimation holes). 1e30 = no triangle covers the cell = OPEN SEA.
    std::vector<float> minH(N, 1e30f);
    const float gx0w = minFx * kFlowGrid;
    const float gy0w = minFy * kFlowGrid;
    for (const TileTriEntry& t : g_tileTris) {
        const auto& pos = t.mesh->positions;
        const auto& idx = t.mesh->indices;
        const std::uint32_t nv = (std::uint32_t)pos.size();
        for (size_t i = 0; i + 3 <= idx.size(); i += 3) {
            const std::uint32_t i0 = idx[i], i1 = idx[i + 1], i2 = idx[i + 2];
            if (i0 >= nv || i1 >= nv || i2 >= nv) continue;
            const D3DXVECTOR3& a = pos[i0];
            const D3DXVECTOR3& b = pos[i1];
            const D3DXVECTOR3& c = pos[i2];
            float bminx = std::min(a.x, std::min(b.x, c.x));
            float bmaxx = std::max(a.x, std::max(b.x, c.x));
            float bminy = std::min(a.y, std::min(b.y, c.y));
            float bmaxy = std::max(a.y, std::max(b.y, c.y));
            int cx0 = (int)floorf(bminx / kFlowGrid) - minFx;
            int cx1 = (int)floorf(bmaxx / kFlowGrid) - minFx;
            int cy0 = (int)floorf(bminy / kFlowGrid) - minFy;
            int cy1 = (int)floorf(bmaxy / kFlowGrid) - minFy;
            if (cx1 < 0 || cy1 < 0 || cx0 >= W || cy0 >= H) continue;
            if (cx0 < 0) cx0 = 0; if (cy0 < 0) cy0 = 0;
            if (cx1 >= W) cx1 = W - 1; if (cy1 >= H) cy1 = H - 1;
            const float dx1 = b.x - a.x, dy1 = b.y - a.y;
            const float dx2 = c.x - a.x, dy2 = c.y - a.y;
            const float den = dx1 * dy2 - dx2 * dy1;
            if (fabsf(den) < 1e-6f) continue;
            const float invDen = 1.0f / den;
            for (int gy = cy0; gy <= cy1; ++gy) {
                const float wy = gy0w + (gy + 0.5f) * kFlowGrid;
                float* row = &minH[(size_t)gy * W];
                for (int gx = cx0; gx <= cx1; ++gx) {
                    const float wx = gx0w + (gx + 0.5f) * kFlowGrid;
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

    // 3. Classify. openSea = no terrain data (deep ocean) → wet + BFS seed. data &
    //    below water → wet (coast/river/lake bed). data & above water → dry land.
    std::vector<uint8_t> wet(N, 0), openSea(N, 0);
    for (size_t i = 0; i < N; ++i) {
        if (minH[i] >= 1e29f)      { wet[i] = 1; openSea[i] = 1; }   // no data → open ocean
        else if (minH[i] < waterZ) { wet[i] = 1; }                   // submerged terrain
    }

    static const int nb[4][2] = { {1,0},{-1,0},{0,1},{0,-1} };

    // 4a. Connected components (4-conn flood) over wet cells (isolated-lake sizing).
    std::vector<int> label(N, -1);
    std::vector<int> compCount;
    std::vector<int> stk;
    int numComp = 0;
    for (size_t s = 0; s < N; ++s) {
        if (!wet[s] || label[s] != -1) continue;
        const int comp = numComp++;
        int count = 0;
        stk.clear();
        stk.push_back((int)s);
        label[s] = comp;
        while (!stk.empty()) {
            const int c = stk.back(); stk.pop_back();
            ++count;
            const int cx = c % W, cy = c / W;
            for (auto& d : nb) {
                const int nx = cx + d[0], ny = cy + d[1];
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                const int ni = ny * W + nx;
                if (wet[ni] && label[ni] == -1) { label[ni] = comp; stk.push_back(ni); }
            }
        }
        compCount.push_back(count);
    }

    // 4b. Distance-to-sea BFS seeded from OPEN-SEA cells. Rivers/inlets share the sea's
    //     component, so seeding the whole component would zero their distance (no gradient
    //     → no flow); seeding from open ocean gives a real distance ramp inland. Lakes
    //     never reach an open-sea cell → stay INF (isolated). Fallback: if there's no
    //     open sea at all, seed the largest component (keeps behavior reasonable).
    std::vector<int> dist(N, INT_MAX);
    std::vector<int> q;
    q.reserve(N);
    int seaSeed = 0;
    for (size_t i = 0; i < N; ++i) {
        if (openSea[i]) { dist[i] = 0; q.push_back((int)i); ++seaSeed; }
    }
    if (seaSeed == 0) {
        int seaComp = -1, seaMax = -1;
        for (int i = 0; i < numComp; ++i) {
            if (compCount[i] > seaMax) { seaMax = compCount[i]; seaComp = i; }
        }
        for (size_t i = 0; i < N; ++i) {
            if (label[i] == seaComp) { dist[i] = 0; q.push_back((int)i); ++seaSeed; }
        }
    }
    for (size_t head = 0; head < q.size(); ++head) {
        const int c = q[head];
        const int cx = c % W, cy = c / W;
        const int nd = dist[c] + 1;
        for (auto& d : nb) {
            const int nx = cx + d[0], ny = cy + d[1];
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            const int ni = ny * W + nx;
            if (wet[ni] && dist[ni] == INT_MAX) { dist[ni] = nd; q.push_back(ni); }
        }
    }

    // 4c. Distance-to-SHORE BFS: cells from the nearest LAND (dry) cell. This is the
    //     thinness/width measure that drives river detection — narrow water (rivers) stays
    //     small in every direction; open sea grows large. Depth is deliberately ignored.
    //     Grid-edge neighbours are open ocean (the bbox has a fringe), never shore.
    std::vector<int> distShore(N, INT_MAX);
    q.clear();
    for (size_t i = 0; i < N; ++i) {
        if (!wet[i]) continue;
        const int cx = (int)(i % W), cy = (int)(i / W);
        bool touchesLand = false;
        for (auto& d : nb) {
            const int nx = cx + d[0], ny = cy + d[1];
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;   // grid edge = ocean, not land
            if (!wet[ny * W + nx]) { touchesLand = true; break; }
        }
        if (touchesLand) { distShore[i] = 1; q.push_back((int)i); }
    }
    for (size_t head = 0; head < q.size(); ++head) {
        const int c = q[head];
        const int cx = c % W, cy = c / W;
        const int nd = distShore[c] + 1;
        for (auto& d : nb) {
            const int nx = cx + d[0], ny = cy + d[1];
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            const int ni = ny * W + nx;
            if (wet[ni] && distShore[ni] == INT_MAX) { distShore[ni] = nd; q.push_back(ni); }
        }
    }

    // 4d. Wide water = nearest shore farther than the river half-width (open water). Beach =
    //     narrow water within beachReach cells of wide water (the sea's thin fringe). River =
    //     narrow water NOT near any wide water (thin in every direction). distWide is a bounded
    //     dilation of the wide set into the narrow shore band.
    const float riverWidth = g_flowRiverWidth;
    const int   beachReach = (int)(g_flowBeachReach + 0.5f);
    std::vector<uint8_t> wide(N, 0);
    std::vector<int> distWide(N, INT_MAX);
    q.clear();
    for (size_t i = 0; i < N; ++i) {
        if (wet[i] && distShore[i] != INT_MAX && (float)distShore[i] > riverWidth) {
            wide[i] = 1; distWide[i] = 0; q.push_back((int)i);
        }
    }
    for (size_t head = 0; head < q.size(); ++head) {
        const int c = q[head];
        if (distWide[c] >= beachReach) continue;   // stop expanding past the beach band
        const int cx = c % W, cy = c / W;
        const int nd = distWide[c] + 1;
        for (auto& d : nb) {
            const int nx = cx + d[0], ny = cy + d[1];
            if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
            const int ni = ny * W + nx;
            if (wet[ni] && distWide[ni] == INT_MAX) { distWide[ni] = nd; q.push_back(ni); }
        }
    }

    // 5. Per-cell flow field → float channels (so an optional blur is well-defined).
    // chCat is the debug classification (CLASSIFY view): 0 none/dry, 1 sea, 2 river,
    // 3 pond/lake, 4 beach.
    // chDir holds the provisional NEAR-flow strength (river/beach); chFar the SEA far/refraction
    // strength. A final pass packs both into chDir as the alpha routing channel (0.5 = neutral,
    // <0.5 = near group, >0.5 = far group). chInten stays the wave amplitude (pond-calm).
    std::vector<float> chDirX(N, 0.0f), chDirY(N, 0.0f), chInten(N, 0.0f), chDir(N, 0.0f), chFar(N, 0.0f);
    std::vector<float> chDistSmooth(N, 0.0f);   // 5c-bis blurred distance-to-sea (DIST SMOOTH debug view)
    std::vector<uint8_t> chCat(N, 0);
    enum { CAT_NONE = 0, CAT_SEA = 1, CAT_RIVER = 2, CAT_POND = 3, CAT_BEACH = 4 };
    auto distAt = [&](int x, int y) -> int {
        if (x < 0 || y < 0 || x >= W || y >= H) return INT_MAX;
        const int i = y * W + x;
        return wet[i] ? dist[i] : INT_MAX;
    };
    // Central-difference gradient component with one-sided / INF fallback.
    auto gradComp = [](int neg, int pos, int center) -> float {
        const bool hn = neg != INT_MAX, hp = pos != INT_MAX;
        if (hn && hp) return 0.5f * (float)(pos - neg);
        if (hp)       return (float)(pos - center);
        if (hn)       return (float)(center - neg);
        return 0.0f;
    };

    for (size_t i = 0; i < N; ++i) {
        if (!wet[i]) continue;   // dry → all-zero (sampled outside any body)
        const int cx = (int)(i % W), cy = (int)(i / W);
        float dirX = 0.0f, dirY = 0.0f, intensity = 0.0f, directionality = 0.0f;
        uint8_t cat;

        if (dist[i] == INT_MAX) {
            // Not connected to the sea → isolated pond/lake: dead calm, no 3D waves.
            cat = CAT_POND;
            intensity = kFlowWaveNone;
        } else if (wide[i]) {
            // Open water: wide → tallest isotropic swell (3x).
            cat = CAT_SEA;
            intensity = kFlowWaveSea;
        } else if (distWide[i] != INT_MAX && distWide[i] <= beachReach) {
            // Narrow fringe of open water → beach (1x; gets the flow-steered directional crests).
            cat = CAT_BEACH;
            intensity = kFlowWaveDefault;
        } else {
            // Narrow, sea-connected, away from open water → river. Downstream = -grad(distToSea).
            cat = CAT_RIVER;
            const float gx = gradComp(distAt(cx - 1, cy), distAt(cx + 1, cy), dist[i]);
            const float gy = gradComp(distAt(cx, cy - 1), distAt(cx, cy + 1), dist[i]);
            const float fx = -gx, fy = -gy;
            const float len = sqrtf(fx * fx + fy * fy);
            if (len > 1e-3f) { dirX = fx / len; dirY = fy / len; directionality = 1.0f; }
            intensity = kFlowWaveRiver;   // ~0.5x: less than open water
        }

        chDirX[i] = dirX; chDirY[i] = dirY; chInten[i] = intensity; chDir[i] = directionality;
        chCat[i] = cat;
    }

    // 5b. BeachExtend: grow beach seaward into open water by beachExtend cells. Expands ONLY
    //     into SEA cells (reclassifying them to beach), so it can neither create nor shorten
    //     rivers — it widens the coastal beach band independently of RiverWidth/BeachReach.
    const int beachExtend = (int)(g_flowBeachExtend + 0.5f);
    if (beachExtend > 0) {
        std::vector<int> dB(N, INT_MAX);
        q.clear();
        for (size_t i = 0; i < N; ++i) {
            if (chCat[i] == CAT_BEACH) { dB[i] = 0; q.push_back((int)i); }
        }
        for (size_t head = 0; head < q.size(); ++head) {
            const int c = q[head];
            if (dB[c] >= beachExtend) continue;   // cap distance from the original beach front
            const int cx = c % W, cy = c / W;
            const int nd = dB[c] + 1;
            for (auto& d : nb) {
                const int nx = cx + d[0], ny = cy + d[1];
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                const int ni = ny * W + nx;
                if (wet[ni] && dB[ni] == INT_MAX && chCat[ni] == CAT_SEA) {
                    dB[ni] = nd; q.push_back(ni);
                    chCat[ni] = CAT_BEACH;
                    chInten[ni] = kFlowWaveDefault;   // beach → 1x
                    chDirX[ni] = chDirY[ni] = chDir[ni] = 0.0f;
                }
            }
        }
    }

    // 5c. Beach flow ramps SMOOTHLY across the band, from the open-sea front (calm, no flow) to
    //     the land edge (full onshore flow). Per the design: the coastline's local ups/downs must
    //     NOT bend the general direction, and there must be no sudden calm-sea→violent-beach step.
    //     We build a normalized cross-band coordinate phi in [0,1] from two BFS distance fields over
    //     the beach cells: dSea (cells from the seaward front) and dLand (cells from the landward
    //     front); phi = dSea/(dSea+dLand) → 0 at the sea side, 1 at the land side. phi is blurred so
    //     border wiggles average out, then:
    //       direction      = normalize(grad phi)              (general onshore, not border-following)
    //       directionality = phi                              (flow accelerates 0 → full toward land)
    //       intensity      = lerp(1.0 [sea], beachMul, phi)   (amplitude blends, no step)
    {
        std::vector<int> dSea(N, INT_MAX), dLand(N, INT_MAX);
        // seaward front: beach cells adjacent to a SEA cell
        q.clear();
        for (size_t i = 0; i < N; ++i) {
            if (chCat[i] != CAT_BEACH) continue;
            const int cx = (int)(i % W), cy = (int)(i / W);
            for (auto& d : nb) {
                const int nx = cx + d[0], ny = cy + d[1];
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                if (chCat[ny * W + nx] == CAT_SEA) { dSea[i] = 0; q.push_back((int)i); break; }
            }
        }
        for (size_t head = 0; head < q.size(); ++head) {
            const int c = q[head];
            const int cx = c % W, cy = c / W;
            const int nd = dSea[c] + 1;
            for (auto& d : nb) {
                const int nx = cx + d[0], ny = cy + d[1];
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                const int ni = ny * W + nx;
                if (chCat[ni] == CAT_BEACH && dSea[ni] == INT_MAX) { dSea[ni] = nd; q.push_back(ni); }
            }
        }
        // landward front: beach cells adjacent to dry land
        q.clear();
        for (size_t i = 0; i < N; ++i) {
            if (chCat[i] != CAT_BEACH) continue;
            const int cx = (int)(i % W), cy = (int)(i / W);
            for (auto& d : nb) {
                const int nx = cx + d[0], ny = cy + d[1];
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                if (!wet[ny * W + nx]) { dLand[i] = 0; q.push_back((int)i); break; }
            }
        }
        for (size_t head = 0; head < q.size(); ++head) {
            const int c = q[head];
            const int cx = c % W, cy = c / W;
            const int nd = dLand[c] + 1;
            for (auto& d : nb) {
                const int nx = cx + d[0], ny = cy + d[1];
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                const int ni = ny * W + nx;
                if (chCat[ni] == CAT_BEACH && dLand[ni] == INT_MAX) { dLand[ni] = nd; q.push_back(ni); }
            }
        }
        // normalized cross-band coordinate
        std::vector<float> phi(N, 0.0f);
        for (size_t i = 0; i < N; ++i) {
            if (chCat[i] != CAT_BEACH) continue;
            const float a = (dSea[i]  == INT_MAX) ? (float)beachReach : (float)dSea[i];
            const float b = (dLand[i] == INT_MAX) ? (float)beachReach : (float)dLand[i];
            const float s = a + b;
            phi[i] = (s > 0.0f) ? (a / s) : 0.5f;
        }
        // blur phi over beach neighbours so the coastline's ups/downs don't steer the direction
        for (int pass = 0; pass < kFlowBlurPasses + 4; ++pass) {
            std::vector<float> tP(phi);
            for (size_t i = 0; i < N; ++i) {
                if (chCat[i] != CAT_BEACH) continue;
                const int cx = (int)(i % W), cy = (int)(i / W);
                float sP = tP[i]; int n = 1;
                for (auto& d : nb) {
                    const int nx = cx + d[0], ny = cy + d[1];
                    if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                    const int ni = ny * W + nx;
                    if (chCat[ni] == CAT_BEACH) { sP += tP[ni]; ++n; }
                }
                phi[i] = sP / (float)n;
            }
        }
        // DIRECTION = the MACRO coastline normal: gradient of a heavily-blurred land(1)/water(0)
        //     field. Blurring over a wide radius collapses the whole grid into a smooth land/sea
        //     ramp, so grad() at a beach cell is the broad orientation of that coast — south of the
        //     island it points north (waves come from the south), the west coast points east, etc.
        //     Fully averaged across cells: the flow map shows none of the per-cell boxy source. phi
        //     drives only the STRENGTH ramp.
        std::vector<float> lf(N), tmp(N);
        for (size_t i = 0; i < N; ++i) lf[i] = wet[i] ? 0.0f : 1.0f;
        {
            std::vector<float> pre(std::max(W, H) + 1);
            const int dirIters = std::max(1, (int)(g_flowDirIters + 0.5f));
            const int R = std::max(1, (int)(g_flowDirRadius + 0.5f));
            for (int it = 0; it < dirIters; ++it) {
                for (int y = 0; y < H; ++y) {                       // horizontal box blur, radius R
                    const int row = y * W;
                    pre[0] = 0.0f;
                    for (int x = 0; x < W; ++x) pre[x + 1] = pre[x] + lf[row + x];
                    for (int x = 0; x < W; ++x) {
                        const int lo = std::max(0, x - R), hi = std::min(W - 1, x + R);
                        tmp[row + x] = (pre[hi + 1] - pre[lo]) / (float)(hi - lo + 1);
                    }
                }
                for (int x = 0; x < W; ++x) {                       // vertical box blur, radius R
                    pre[0] = 0.0f;
                    for (int y = 0; y < H; ++y) pre[y + 1] = pre[y] + tmp[y * W + x];
                    for (int y = 0; y < H; ++y) {
                        const int lo = std::max(0, y - R), hi = std::min(H - 1, y + R);
                        lf[y * W + x] = (pre[hi + 1] - pre[lo]) / (float)(hi - lo + 1);
                    }
                }
            }
        }
        const int R = std::max(1, (int)(g_flowDirRadius + 0.5f));
        for (size_t i = 0; i < N; ++i) {
            const bool isBeach = (chCat[i] == CAT_BEACH);
            const bool isSea   = (chCat[i] == CAT_SEA);
            if (!isBeach && !isSea) continue;
            const int cx = (int)(i % W), cy = (int)(i / W);
            const int xl = std::max(0, cx - 1), xr = std::min(W - 1, cx + 1);
            const int yd = std::max(0, cy - 1), yu = std::min(H - 1, cy + 1);
            const float gx = lf[cy * W + xr] - lf[cy * W + xl];    // toward more land = onshore
            const float gy = lf[yu * W + cx] - lf[yd * W + cx];
            const float len = sqrtf(gx * gx + gy * gy);
            if (len > 1e-6f) { chDirX[i] = gx / len; chDirY[i] = gy / len; }
            if (isBeach) {
                chDir[i]   = phi[i];                               // near-flow strength ramps 0 → full
                chInten[i] = kFlowWaveDefault;                     // beach amplitude = 1x (flat mask)
            } else {
                // Sea: far-wave (refraction) strength fades offshore. The blurred-field gradient
                // magnitude ~1/R near a coast and ~0 in deep sea; len*R*boost normalises that to
                // a 0..1 ramp so swells refract toward shore only within the coastal band.
                if (len > 1e-6f) chFar[i] = std::min(1.0f, len * (float)R * kFlowSeaRefract);
            }
        }
    }

    // 5c-bis. De-block the river foam the SAME way the beach band is smoothed (5c): a wide
    //     separable box blur. Two fields get it — both knobbed by RiverBlurR / RiverBlurIt / RiverGain
    //     (#23/24/25, rebake):
    //       (a) DIRECTION (the square look): the river flow vector in step 5 is the central
    //           difference of the INTEGER distance field, so its angle quantizes to a few values
    //           per cell (the diagonal staircase) — and the far foam advects the fbm ALONG it
    //           (fadv = fdir·time·speed), so neighbouring cells scroll the texture in different
    //           constant directions → square seams. Beach has none of this because its normal is
    //           grad(wide-blurred field). We do the SAME for rivers: blur the SCALAR distance field
    //           first, then take its gradient → a continuous heading. Rewrites the source direction,
    //           so the debug view, DIRECTION encoding and downstream foam all see de-blocked data.
    //       (b) STRENGTH: the binary river mask blurred into a smooth ramp, so the consume's erosion
    //           contour (ErodeThr) is a rounded curve, not the square mask edge. Replaces the hard
    //           near-strength (=1); beach phi kept via max(); gain restores a narrow river's core.
    //     Sea alpha is overwritten by chFar in 5d, so this shapes river/beach/pond only.
    {
        const int rbR  = std::max(1, (int)(g_flowRiverBlurRadius + 0.5f));
        const int rbIt = std::max(1, (int)(g_flowRiverBlurIters + 0.5f));
        std::vector<float> pre(std::max(W, H) + 1), tmp(N);
        auto boxBlur = [&](std::vector<float>& f) {            // rbIt separable box passes, radius rbR
            for (int it = 0; it < rbIt; ++it) {
                for (int y = 0; y < H; ++y) {                  // horizontal
                    const int row = y * W;
                    pre[0] = 0.0f;
                    for (int x = 0; x < W; ++x) pre[x + 1] = pre[x] + f[row + x];
                    for (int x = 0; x < W; ++x) {
                        const int lo = std::max(0, x - rbR), hi = std::min(W - 1, x + rbR);
                        tmp[row + x] = (pre[hi + 1] - pre[lo]) / (float)(hi - lo + 1);
                    }
                }
                for (int x = 0; x < W; ++x) {                  // vertical
                    pre[0] = 0.0f;
                    for (int y = 0; y < H; ++y) pre[y + 1] = pre[y] + tmp[y * W + x];
                    for (int y = 0; y < H; ++y) {
                        const int lo = std::max(0, y - rbR), hi = std::min(H - 1, y + rbR);
                        f[y * W + x] = (pre[hi + 1] - pre[lo]) / (float)(hi - lo + 1);
                    }
                }
            }
        };

        // (a) river direction = -grad(SMOOTH distance-to-sea): blur the SCALAR distance field
        //     (normalized convolution over connected water), then take its gradient. The integer
        //     distance field's raw per-cell gradient quantizes to a few angles (the diagonal
        //     staircase); blurring the scalar first gives a continuous field → a continuous heading.
        std::vector<float> df(N, 0.0f), dw(N, 0.0f);
        for (size_t i = 0; i < N; ++i)
            if (wet[i] && dist[i] != INT_MAX) { df[i] = (float)dist[i]; dw[i] = 1.0f; }
        boxBlur(df); boxBlur(dw);
        std::vector<float>& ds = chDistSmooth;   // retained for the DIST SMOOTH debug view
        for (size_t i = 0; i < N; ++i) ds[i] = (dw[i] > 1e-6f) ? (df[i] / dw[i]) : 0.0f;
        auto dsAt = [&](int x, int y, float fallback) -> float {   // one-sided fallback past the blur reach
            const int j = y * W + x;
            return (dw[j] > 1e-3f) ? ds[j] : fallback;
        };

        // (b) river strength: binary mask → smooth ramp
        std::vector<float> rf(N, 0.0f);
        for (size_t i = 0; i < N; ++i) rf[i] = (chCat[i] == CAT_RIVER) ? 1.0f : 0.0f;
        boxBlur(rf);

        const float gain = g_flowRiverGain;
        for (size_t i = 0; i < N; ++i) {
            if (!wet[i]) continue;
            if (chCat[i] == CAT_RIVER) {                           // smooth heading from grad(ds)
                const int cx = (int)(i % W), cy = (int)(i / W);
                const int xl = std::max(0, cx - 1), xr = std::min(W - 1, cx + 1);
                const int yd = std::max(0, cy - 1), yu = std::min(H - 1, cy + 1);
                const float c  = ds[i];
                const float gx = dsAt(xr, cy, c) - dsAt(xl, cy, c);
                const float gy = dsAt(cx, yu, c) - dsAt(cx, yd, c);
                const float fx = -gx, fy = -gy;                    // downstream = decreasing dist-to-sea
                const float len = sqrtf(fx * fx + fy * fy);
                if (len > 1e-4f) { chDirX[i] = fx / len; chDirY[i] = fy / len; }
            }
            const float beachPart   = (chCat[i] == CAT_BEACH) ? chDir[i] : 0.0f;
            const float riverSmooth = std::min(1.0f, rf[i] * gain);   // smooth ramp, not the boxy 1
            chDir[i] = std::max(beachPart, riverSmooth);
        }
    }

    // 5d. Pack the alpha routing channel: 0.5 = neutral; <0.5 = NEAR group (river/beach), with
    //     near-strength = (0.5 - a) * 2 advecting the close normal; >0.5 = FAR group (sea), with
    //     far-strength = (a - 0.5) * 2 advecting the far normal (refraction). chDir held the
    //     provisional near strength (river=1, beach=phi, sea/pond=0); chFar the sea far strength.
    for (size_t i = 0; i < N; ++i) {
        if (!wet[i]) continue;
        if (chCat[i] == CAT_SEA) chDir[i] = 0.5f + 0.5f * std::min(1.0f, std::max(0.0f, chFar[i]));
        else                     chDir[i] = 0.5f - 0.5f * std::min(1.0f, std::max(0.0f, chDir[i]));
    }

    // 6. Light box-blur of ALL channels (intensity + flow dir + directionality), so the
    // piecewise-constant per-cell field doesn't show hard square edges through the water's
    // linear texture filter. Averaged only over wet neighbours (dry stays zero). Direction
    // is blurred un-normalized: shorter vectors near body edges → gentler advection, which
    // is what we want at transitions. Bends average opposing vectors, but 1 pass is mild.
    for (int pass = 0; pass < kFlowBlurPasses; ++pass) {
        std::vector<float> tI(chInten), tX(chDirX), tY(chDirY), tD(chDir);
        for (size_t i = 0; i < N; ++i) {
            if (!wet[i]) continue;
            const int cx = (int)(i % W), cy = (int)(i / W);
            float sI = tI[i], sX = tX[i], sY = tY[i], sD = tD[i]; int n = 1;
            for (auto& d : nb) {
                const int nx = cx + d[0], ny = cy + d[1];
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                const int ni = ny * W + nx;
                if (wet[ni]) { sI += tI[ni]; sX += tX[ni]; sY += tY[ni]; sD += tD[ni]; ++n; }
            }
            const float inv = 1.0f / (float)n;
            chInten[i] = sI * inv; chDirX[i] = sX * inv; chDirY[i] = sY * inv; chDir[i] = sD * inv;
        }
    }

    // 6b. Dilate the baked field outward into the dry border (FlowExpand cells) so the water EDGE
    //     samples valid flow on BOTH sides — no zero/gray cells to interpolate toward and no foam
    //     cut at the 512u grid line. Nearest-wet propagation via BFS; cells past the margin keep
    //     dryDefault. hasData marks wet + dilated cells; the encoder below uses it instead of wet.
    std::vector<uint8_t> hasData(N, 0);
    for (size_t i = 0; i < N; ++i) hasData[i] = wet[i] ? 1 : 0;
    const int dilateCells = std::max(0, (int)(g_flowDilate + 0.5f));
    if (dilateCells > 0) {
        std::vector<int> dDil(N, INT_MAX);
        q.clear();
        for (size_t i = 0; i < N; ++i) if (wet[i]) { dDil[i] = 0; q.push_back((int)i); }
        for (size_t head = 0; head < q.size(); ++head) {
            const int c = q[head];
            if (dDil[c] >= dilateCells) continue;       // cap expansion at the margin
            const int cx = c % W, cy = c / W;
            const int nd = dDil[c] + 1;
            for (auto& d : nb) {
                const int nx = cx + d[0], ny = cy + d[1];
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                const int ni = ny * W + nx;
                if (!hasData[ni]) {                      // first (= nearest-wet) reach wins
                    dDil[ni] = nd;
                    chDirX[ni] = chDirX[c]; chDirY[ni] = chDirY[c];
                    chInten[ni] = chInten[c]; chDir[ni] = chDir[c];
                    chCat[ni] = chCat[c];                // carry category so debug views show the expansion
                    hasData[ni] = 1;
                    q.push_back(ni);
                }
            }
        }
    }

    // 7. Encode RGBA8 (A8R8G8B8 in-memory = 0xAARRGGBB).
    auto enc8 = [](float v01) -> uint32_t {
        int b = (int)(v01 * 255.0f + 0.5f);
        return (uint32_t)std::max(0, std::min(255, b));
    };
    g_flowMapBytes.assign(N, 0);
    const int debugBake = g_flowDebugBake.load(std::memory_order_acquire);
    if (debugBake == 1) {
        // CLASSIFY view: flat category colours (sea=blue, river=green, pond=red, beach=yellow).
        // 0xAARRGGBB; A unused here (shader keys on flowDebugView, tints by RGB).
        static const uint32_t catCol[5] = {
            0x00000000,  // none/dry
            0x000000FF,  // sea   = blue
            0x0000FF00,  // river = green
            0x00FF0000,  // pond  = red
            0x00FFFF00,  // beach = yellow
        };
        for (size_t i = 0; i < N; ++i) {
            if (!hasData[i]) continue;
            g_flowMapBytes[i] = catCol[chCat[i]];
        }
    } else if (debugBake == 2) {
        // DIRECTION view: colour by flow angle (hue wheel). Directional cells (rivers
        // downstream, beaches onshore) get a saturated hue; non-directional water (sea/pond)
        // is dim grey. East=red, North=green, West=cyan, South=magenta-ish.
        for (size_t i = 0; i < N; ++i) {
            if (!hasData[i]) continue;
            if (chDirX[i] != 0.0f || chDirY[i] != 0.0f) {   // any directional cell (river/beach/sea-refract)
                const float h = atan2f(chDirY[i], chDirX[i]) * 0.1591549f + 0.5f;   // 1/(2pi)
                const float r = std::min(1.0f, std::max(0.0f, fabsf(h * 6 - 3) - 1));
                const float g = std::min(1.0f, std::max(0.0f, 2 - fabsf(h * 6 - 2)));
                const float b = std::min(1.0f, std::max(0.0f, 2 - fabsf(h * 6 - 4)));
                g_flowMapBytes[i] = (enc8(r) << 16) | (enc8(g) << 8) | enc8(b);
            } else {
                g_flowMapBytes[i] = 0x00303030;   // grey = no direction
            }
        }
    } else if (debugBake >= 3 && debugBake <= 6) {
        // Grayscale source views (RGB = value; the shader tints by RGB, masking near-black).
        //   3 STRENGTH   = near river/beach routing strength = saturate((0.5 - A) * 2)
        //   4 INTENSITY  = wave amplitude (B channel)
        //   5 DIST RAW   = banded raw integer dist-to-sea  (jagged bands = quantized direction source)
        //   6 DIST SMOOTH= banded blurred dist-to-sea (chDistSmooth) (smooth bands = de-blocked)
        const float distBand = 8.0f;   // cells per contour band
        auto gray = [&](float v) -> uint32_t {
            const uint32_t g = enc8(std::max(0.0f, std::min(1.0f, v)));
            return (g << 16) | (g << 8) | g;
        };
        for (size_t i = 0; i < N; ++i) {
            if (!hasData[i]) continue;
            float v = 0.0f;
            if (debugBake == 3) {
                v = std::max(0.0f, std::min(1.0f, (0.5f - chDir[i]) * 2.0f));
            } else if (debugBake == 4) {
                v = chInten[i];
            } else {   // 5 / 6: distance bands, connected water only
                if (dist[i] == INT_MAX) continue;
                const float d = (debugBake == 5) ? (float)dist[i] : chDistSmooth[i];
                v = (d - std::floor(d / distBand) * distBand) / distBand;   // sawtooth contour bands
            }
            g_flowMapBytes[i] = gray(v);
        }
    } else {
        // Dry / out-of-coverage cells default to 1x ambient waves (B = kFlowWaveDefault),
        // not 0: the LOD water mesh reaches far past this wet-cell bbox and clamp-samples
        // the border, so a 0 here left the distant water flat. Direction/routing neutral
        // (no heading → the shader keeps those cells isotropic). Wet cells keep their bake.
        const uint32_t dryDefault = (enc8(0.5f) << 24) | (enc8(0.5f) << 16) | (enc8(0.5f) << 8) | enc8(kFlowWaveDefault);
        for (size_t i = 0; i < N; ++i) {
            if (!hasData[i]) { g_flowMapBytes[i] = dryDefault; continue; }   // wet + dilated border
            const uint32_t R = enc8(chDirX[i] * 0.5f + 0.5f);
            const uint32_t G = enc8(chDirY[i] * 0.5f + 0.5f);
            const uint32_t B = enc8(chInten[i]);
            const uint32_t A = enc8(chDir[i]);
            g_flowMapBytes[i] = (A << 24) | (R << 16) | (G << 8) | B;
        }
    }

    g_flowMapW = W; g_flowMapH = H;
    g_flowOriginX = minFx * kFlowGrid;
    g_flowOriginY = minFy * kFlowGrid;
    g_flowInvSizeX = 1.0f / (W * kFlowGrid);
    g_flowInvSizeY = 1.0f / (H * kFlowGrid);
    g_flowMapDirty.store(true, std::memory_order_release);

    LOG::logline("-- [water-flow] flow map built: %dx%d cells, %d bodies, sea seed=%d cells, waterZ=%.0f",
                 W, H, numComp, seaSeed, waterZ);
}

// Main thread: lazily (re)create texFlow and upload the worker-baked bytes when
// dirty. Returns true if texFlow is valid to bind. Same managed Lock/Unlock upload
// pattern as the error texture / ripple textures.
bool DistantLand::updateFlowMapTexture() {
    if (g_flowMapW <= 0 || g_flowMapH <= 0) {
        return texFlow != nullptr;
    }
    if (g_flowMapDirty.exchange(false, std::memory_order_acquire)) {
        // Recreate if the grid dimensions changed (e.g. land set reloaded).
        if (texFlow) {
            D3DSURFACE_DESC desc;
            if (texFlow->GetLevelDesc(0, &desc) == D3D_OK &&
                ((int)desc.Width != g_flowMapW || (int)desc.Height != g_flowMapH)) {
                texFlow->Release();
                texFlow = nullptr;
            }
        }
        if (!texFlow) {
            if (device->CreateTexture(g_flowMapW, g_flowMapH, 1, g_spikeForceDefaultPool ? D3DUSAGE_DYNAMIC : 0, D3DFMT_A8R8G8B8,
                                      g_spikeForceDefaultPool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &texFlow, NULL) != D3D_OK) {
                texFlow = nullptr;
                LOG::logline("!! [water-flow] CreateTexture failed (%dx%d)", g_flowMapW, g_flowMapH);
                return false;
            }
        }
        D3DLOCKED_RECT lr;
        if (texFlow->LockRect(0, &lr, NULL, 0) == D3D_OK) {
            const uint8_t* src = (const uint8_t*)g_flowMapBytes.data();
            uint8_t* dst = (uint8_t*)lr.pBits;
            const int rowBytes = g_flowMapW * 4;
            for (int y = 0; y < g_flowMapH; ++y) {
                memcpy(dst + (size_t)y * lr.Pitch, src + (size_t)y * rowBytes, rowBytes);
            }
            texFlow->UnlockRect(0);
        }
    }
    return texFlow != nullptr;
}

void DistantLand::getFlowMapTransform(float out[4]) {
    out[0] = g_flowOriginX;
    out[1] = g_flowOriginY;
    out[2] = g_flowInvSizeX;
    out[3] = g_flowInvSizeY;
}

// Rebuild the dense near-camera terrain height grid for the 3x3 world cells around the
// given eye cell, by rasterizing real terrain triangles (continuous coverage → no
// vertex-spacing holes). No-op if already built for this cell. Runs on the cull worker
// during isReflectionWaterVisible; cost is bounded to the few tiles overlapping the
// region (per-tile AABB cull) and only paid on a cell crossing.
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
bool DistantLand::isReflectionWaterVisible() {
    MGE_ZoneScopedN("isReflectionWaterVisible");

    // Cleared up front so the interior / no-terrain-data early-outs below leave it
    // false (cullReflectionSurvivors then keeps all — the legacy fallback). Set true
    // only after the tile loop actually runs.
    g_reflRectsComputed = false;

    // Water cull off by default; enabled below only when the screen-space projection is
    // reliable (exterior, terrain data, eye well above water). Interior / no-data / swimming
    // leave it false → both consumers keep all reflection candidates.
    reflWaterCullActive = false;

    // Stage-2 fine mask: clear bits + invalidate up front, so the same early-outs
    // leave the mask invalid (reflWaterMaskTestNDC then keeps all). Validated at the
    // same point as the rects, once the tile loop has run.
    g_reflWaterMaskValid = false;
    memset(g_reflWaterMask, 0, sizeof(g_reflWaterMask));

    // The terrain-height gate is an exterior-only optimization: the min-height
    // maps describe the exterior worldspace, and landMeshes persists across cell
    // changes (built at init, cleared at release). In an interior those maps are
    // stale relative to the interior's own water, so the gate would test interior
    // water tiles against unrelated exterior terrain and wrongly cull. Key on
    // IsExterior (not isDistantCell): interiors that ship generated distant land
    // are still distant cells, but their water must not be gated by exterior
    // terrain. Reflect unconditionally in any interior.
    if (!MWBridge::get()->IsExterior()) {
        return true;
    }

    const bool debug = g_drawWaterProxyBounds;
    if (debug) g_waterProxyDebugBoxes.clear();

    // Build/refresh the terrain maps if the captured land set changed (init or
    // release). Built once per session in practice.
    if (g_terrainMinHSrcSize != landMeshes.size()) {
        buildTerrainMinHeight(landMeshes);
        // Same one-time gate (CPU only, on the cull worker). The wet/dry source is
        // g_fineMinH, just (re)built above. waterZ captured at build time.
        if (Configuration.UseWaterFlowMap) {
            buildWaterFlowMap(MWBridge::get()->WaterLevel());
        }
    } else if (Configuration.UseWaterFlowMap &&
               g_flowRebake.exchange(false, std::memory_order_acquire)) {
        // Knob tuning (NUMPAD7/6/3) asked for a re-bake. Terrain triangles persist,
        // so only the flow map is rebuilt; same CPU-only worker path.
        buildWaterFlowMap(MWBridge::get()->WaterLevel());
    }
    if (g_cellMinH.empty()) {
        return true;   // no terrain data → gate inert, reflect as before
    }

    const bool msocUsable = MSOCClient::isAvailable() && MSOCClient::isMaskReady();

    // Current main-view frustum out to the water-visible range (fogEnd).
    D3DXMATRIX waterProj = mwProj;
    editProjectionZ(&waterProj, 4.0f, Configuration.DL.DrawDist * kCellSize);
    D3DXMATRIX viewProj = mwView * waterProj;
    ViewFrustum frustum(&viewProj);

    const float waterZ = MWBridge::get()->WaterLevel();
    const float R = fogEnd;                 // world units; water visible to fog (the GATE range)
    const float half = 0.5f * kCellSize;    // cell half-extent (4096)
    const float slabHZ = 4.0f;              // thin water box half-height
    const float eyeAboveWater = eyePos.z - waterZ;  // for the grazing-angle static-rect cull
    constexpr float kReflGrazeTan2 = 0.00191f;      // tan^2(2.5 deg): drop static rects below ~2.5 deg elevation
    const float grazeFarMinSq = (2.0f * kCellSize) * (2.0f * kCellSize);  // grazing cull applies only beyond 2 cells

    // Stage-2 fine mask is only well-conditioned when the eye is clearly ABOVE the water
    // plane. Swimming/underwater the eye sits ON the mirror plane, so water-plane cells
    // project edge-on: cells near the eye blow up to wild NDC (rasterizer stress) and the
    // edge-on silhouette wrongly culls peripheral objects. When that happens, OMIT the two
    // silhouette grids (near 128u + far 512u) — skip the raster, leave the mask invalid so
    // reflWaterMaskTestNDC keeps all — and let the coarse rect stage cull alone. The
    // reflection system stays on; only the fine refinement is suspended near the surface.
    constexpr float kMaskMinEyeHeight = 64.0f;      // below this above-water height → skip the fine mask
    // Screen-space water projection is reliable only when the eye is clearly above the
    // plane. Below it (swimming/wading) BOTH the rects and the mask collapse edge-on and
    // over-cull — so disable the whole water cull (keep all). The fine mask additionally
    // honors the Numpad1 A/B toggle.
    const bool projReliable = eyeAboveWater > kMaskMinEyeHeight;
    reflWaterCullActive = projReliable;             // gates rects AND mask in both consumers
    const bool maskUsable = g_reflFineMaskEnabled && projReliable;

    // Reflection STATICS only reflect out to NearStaticEnd (see renderReflectedStatics
    // / prepareReflectionCullForWorker), but the gate tests water to fogEnd. Far horizon
    // tiles beyond the static range still produce thin horizon rects, and the 2D rect
    // test then spuriously keeps distant statics whose mirror projects to the horizon —
    // a single borderline far tile flipping (occluded↔visible) swings the survivor count
    // by hundreds. So contribute static-cull rects ONLY for tiles within the static
    // range; far tiles still set anyVisible (terrain/sky reflection gate unchanged).
    const float staticReflRange   = std::min(fogEnd, Configuration.DL.NearStaticEnd * kCellSize);
    const float staticReflRangeSq = staticReflRange * staticReflRange;

    bool anyVisible = false;
    reflectionWaterRects.clear();

    // [REFL PIPE] reset water-tile stage counters for this frame.
    g_reflPipe.tilesTested = g_reflPipe.waterPresent = g_reflPipe.waterOccluded = 0;
    g_reflPipe.msocUsable = msocUsable;

    // Start this frame's visible-tile set (hysteresis); swapped into ...Last at the end.
    g_reflWaterVisThis.clear();

    // Warmup: the rect-emit gate normally requires a tile visible THIS frame AND last
    // (1-frame anti-flicker). At load / after a cell change the last-frame set is empty,
    // so NO rects emit for one frame → every reflection object is culled (a visible flash
    // of empty water reflection). When the last set is empty, drop the last-frame
    // requirement so rects emit immediately. Anti-flicker is unaffected: the flickering
    // tiles live in already-watery areas (last non-empty), so this only relaxes the very
    // first frame entering water, never the sustained case.
    const bool reflWarmup = g_reflWaterVisLast.empty();

    // Project a water tile footprint (at waterZ) to a main-view NDC AABB. Corners
    // behind the near plane are clamped to the plane (cw=eps) so a straddling near
    // tile yields a bounded — if loose — screen rect rather than garbage. Returns
    // false if the tile projects fully off-screen.
    auto tileScreenRect = [&](float tcx, float tcy, float thalf, D3DXVECTOR4& out) -> bool {
        const float cxs[4] = { tcx - thalf, tcx + thalf, tcx - thalf, tcx + thalf };
        const float cys[4] = { tcy - thalf, tcy - thalf, tcy + thalf, tcy + thalf };
        float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;
        for (int i = 0; i < 4; ++i) {
            const float wx = cxs[i], wy = cys[i], wz = waterZ;
            float cx = wx * viewProj._11 + wy * viewProj._21 + wz * viewProj._31 + viewProj._41;
            float cy = wx * viewProj._12 + wy * viewProj._22 + wz * viewProj._32 + viewProj._42;
            float cw = wx * viewProj._14 + wy * viewProj._24 + wz * viewProj._34 + viewProj._44;
            if (cw < 1e-3f) cw = 1e-3f;     // clamp behind-plane corners
            const float inv = 1.0f / cw;
            float nx = cx * inv, ny = cy * inv;
            nx = std::max(-1.5f, std::min(1.5f, nx));
            ny = std::max(-1.5f, std::min(1.5f, ny));
            minX = std::min(minX, nx); maxX = std::max(maxX, nx);
            minY = std::min(minY, ny); maxY = std::max(maxY, ny);
        }
        // Clamp to screen; reject if no on-screen overlap.
        minX = std::max(-1.0f, minX); minY = std::max(-1.0f, minY);
        maxX = std::min( 1.0f, maxX); maxY = std::min( 1.0f, maxY);
        if (minX > maxX || minY > maxY) return false;
        out = D3DXVECTOR4(minX, minY, maxX, maxY);
        return true;
    };

    // Water-presence verdict: 1 = water (terrain below WaterLevel), 0 = dry
    // (terrain at/above water), 2 = no data (conservatively treated as water).
    auto cellWater = [&](int cx, int cy) -> int {
        auto it = g_cellMinH.find(waterGridKey(cx, cy));
        if (it == g_cellMinH.end()) return 2;
        return (it->second < waterZ) ? 1 : 0;
    };
    auto footprintWater = [&](float minX, float maxX, float minY, float maxY) -> int {
        const int fx0 = (int)floorf(minX / kWaterFineGrid);
        const int fx1 = (int)floorf(maxX / kWaterFineGrid);
        const int fy0 = (int)floorf(minY / kWaterFineGrid);
        const int fy1 = (int)floorf(maxY / kWaterFineGrid);
        bool any = false;
        for (int fy = fy0; fy <= fy1; ++fy) {
            for (int fx = fx0; fx <= fx1; ++fx) {
                auto it = g_fineMinH.find(waterGridKey(fx, fy));
                if (it == g_fineMinH.end()) continue;
                any = true;
                if (it->second < waterZ) return 1;   // a wet sub-tile here
            }
        }
        return any ? 0 : 2;
    };

    // Stage-2 raster primitive: project one wet grid cell (centre ccx,ccy, half-extent
    // cellHalf) onto the water plane to mask-cell coords and scanline-fill the actual
    // quad — tight, no AABB inland spill. Water on the mirror plane ⇒ main-cam NDC ==
    // reflection NDC. Corner order matches tileScreenRect: 0=(-,-) 1=(+,-) 2=(-,+) 3=(+,+).
    auto fillWetCellQuad = [&](float ccx, float ccy, float cellHalf) {
        const float cxs[4] = { ccx - cellHalf, ccx + cellHalf, ccx - cellHalf, ccx + cellHalf };
        const float cys[4] = { ccy - cellHalf, ccy - cellHalf, ccy + cellHalf, ccy + cellHalf };
        float mx[4], my[4];
        for (int i = 0; i < 4; ++i) {
            const float wx = cxs[i], wy = cys[i], wz = waterZ;
            const float cx = wx*viewProj._11 + wy*viewProj._21 + wz*viewProj._31 + viewProj._41;
            const float cy = wx*viewProj._12 + wy*viewProj._22 + wz*viewProj._32 + viewProj._42;
            const float cw = wx*viewProj._14 + wy*viewProj._24 + wz*viewProj._34 + viewProj._44;
            if (cw < 1e-3f) {
                // Rare near-plane straddle: fall back to the loose AABB so there's no
                // hole at the camera's feet (safe over-cover, not the common path).
                D3DXVECTOR4 cell;
                if (tileScreenRect(ccx, ccy, cellHalf, cell))
                    reflMaskSetRange(cell.x, cell.y, cell.z, cell.w);
                return;
            }
            const float inv = 1.0f / cw;
            mx[i] = (cx * inv * 0.5f + 0.5f) * (float)kReflMaskW;
            my[i] = (cy * inv * 0.5f + 0.5f) * (float)kReflMaskH;
        }
        reflMaskFillQuad(mx, my);
    };

    // FAR tiles: walk the 512u g_fineMinH cells in the tile footprint, fill the wet ones.
    // Far cells are sub-pixel on screen, so the coarse grid is plenty.
    auto rasterTileWetCells = [&](float tcx, float tcy, float thalf) {
        const float grid = kWaterFineGrid;
        const int fx0 = (int)floorf((tcx - thalf) / grid);
        const int fx1 = (int)floorf((tcx + thalf) / grid);
        const int fy0 = (int)floorf((tcy - thalf) / grid);
        const int fy1 = (int)floorf((tcy + thalf) / grid);
        const float cellHalf = 0.5f * grid;
        for (int fy = fy0; fy <= fy1; ++fy)
            for (int fx = fx0; fx <= fx1; ++fx) {
                auto it = g_fineMinH.find(waterGridKey(fx, fy));
                if (it == g_fineMinH.end() || it->second >= waterZ) continue;   // dry / no data
                fillWetCellQuad((fx + 0.5f) * grid, (fy + 0.5f) * grid, cellHalf);
            }
    };

    // CLOSE tiles: walk the dense 128u runtime near-grid (triangle-rasterized real
    // terrain → continuous coverage, no vertex-spacing holes) over the sub-tile
    // footprint, fill the wet cells. This is the tight near-camera silhouette.
    auto rasterCloseTile = [&](float tcx, float tcy, float thalf) {
        if (!g_nearGrid.valid) { rasterTileWetCells(tcx, tcy, thalf); return; }   // fallback
        const float cellHalf = 0.5f * kNearGrid;
        int cx0 = (int)floorf((tcx - thalf) / kNearGrid) - g_nearGrid.gx0;
        int cx1 = (int)floorf((tcx + thalf) / kNearGrid) - g_nearGrid.gx0;
        int cy0 = (int)floorf((tcy - thalf) / kNearGrid) - g_nearGrid.gy0;
        int cy1 = (int)floorf((tcy + thalf) / kNearGrid) - g_nearGrid.gy0;
        if (cx1 < 0 || cy1 < 0 || cx0 >= kNearW || cy0 >= kNearW) return;
        if (cx0 < 0) cx0 = 0; if (cy0 < 0) cy0 = 0;
        if (cx1 >= kNearW) cx1 = kNearW - 1; if (cy1 >= kNearW) cy1 = kNearW - 1;
        for (int gy = cy0; gy <= cy1; ++gy) {
            const float* row = &g_nearGrid.minZ[(size_t)gy * kNearW];
            for (int gx = cx0; gx <= cx1; ++gx) {
                if (row[gx] >= waterZ) continue;   // dry / no terrain (+inf)
                fillWetCellQuad((g_nearGrid.gx0 + gx + 0.5f) * kNearGrid,
                                (g_nearGrid.gy0 + gy + 0.5f) * kNearGrid, cellHalf);
            }
        }
    };

    // Test one water tile: frustum-cull the thin box, terrain water-presence,
    // then MSOC occlusion on the survivor. A surviving (visible water) tile
    // contributes its screen rect to reflectionWaterRects for the Phase-B static
    // cull. No early-out — the full survivor set is the cull input. Visualizer:
    //   green  = water present and not occluded → reflects
    //   red    = water present but MSOC-occluded → culled
    //   yellow = dry land at/above water level → no water
    //   (no-data tiles are very distant; dropped and not drawn)
    auto testTile = [&](float tcx, float tcy, float thalf, bool closeTile) {
        BoundingBox wbox(
            D3DXVECTOR3(tcx - thalf, tcy - thalf, waterZ - slabHZ),
            D3DXVECTOR3(tcx + thalf, tcy + thalf, waterZ + slabHZ));
        if (frustum.ContainsBox(wbox) == ViewFrustum::OUTSIDE) return;
        ++g_reflPipe.tilesTested;   // [REFL PIPE] in-frustum water-tile candidates

        const int w = (thalf >= half - 1.0f)
            ? cellWater((int)floorf(tcx / kCellSize), (int)floorf(tcy / kCellSize))
            : footprintWater(tcx - thalf, tcx + thalf, tcy - thalf, tcy + thalf);

        // No height data → very distant unsampled terrain; drop entirely.
        if (w == 2) return;

        bool vis;
        MSOCClient::TestResult dbg;
        if (w == 0) {
            // Dry land at/above water level — no water surface here.
            vis = false;
            dbg = MSOCClient::ResultNotReady;     // yellow
        } else {
            // Water present — cull if occlusion proves it hidden. MSOC unavailable
            // → keep it (can't prove occluded), gate falls back to terrain only.
            ++g_reflPipe.waterPresent;   // [REFL PIPE] tiles with water surface
            MSOCClient::TestResult r = msocUsable
                ? MSOCClient::classifyOBB(tcx, tcy, waterZ,
                                          thalf, 0, 0,  0, thalf, 0,  0, 0, slabHZ)
                : MSOCClient::ResultVisible;
            vis = (r != MSOCClient::ResultOccluded);
            if (!vis) ++g_reflPipe.waterOccluded;   // [REFL PIPE] water MSOC-culled
            dbg = vis ? MSOCClient::ResultVisible : MSOCClient::ResultOccluded; // green/red
            if (vis) {
                anyVisible = true;   // gate (terrain/sky reflection) — full fogEnd range
                // Static-cull rect only for water within the reflection-static range,
                // not too grazing, AND only with 1-frame hysteresis (visible this and
                // last). Grazing cull (measured: the flickering tiles sit at ~1.9 deg
                // elevation, 3.3 cells out): such far/tilted cells are coarse tiles
                // straddling the close-terrain ridge silhouette, so their occlusion
                // verdict flickers and swings the survivor count by ~100 statics. We
                // drop only their STATIC-cull contribution; the sky/horizon reflection
                // gate (anyVisible, above) is untouched. elev < thresh <=> eyeAboveWater
                // < dist*tan(thresh); squared to avoid the sqrt. Eye-height-relative, so
                // far water from a hill keeps a steeper angle and is retained.
                //   Gated on dist > 2 cells: while SWIMMING the eye sits at water level
                // (eyeAboveWater ~ 0), so every tile reads grazing — but near water (and
                // the subdivided 3x3 block) must still reflect. Only the far coarse tiles
                // that actually straddle a ridge are culled.
                const float ddx = tcx - eyePos.x, ddy = tcy - eyePos.y;
                const float dist2 = ddx * ddx + ddy * ddy;
                const bool tooGrazing = dist2 > grazeFarMinSq
                    && eyeAboveWater > 0.0f
                    && eyeAboveWater * eyeAboveWater < dist2 * kReflGrazeTan2;
                if (dist2 <= staticReflRangeSq && !tooGrazing) {
                    const int64_t tkey = ((int64_t)lroundf(tcx / 64.0f) << 32)
                                       ^ (int64_t)(uint32_t)lroundf(tcy / 64.0f);
                    g_reflWaterVisThis.insert(tkey);
                    if (reflWarmup || g_reflWaterVisLast.count(tkey) != 0) {   // visible last frame too (or warmup)
                        D3DXVECTOR4 rect;
                        if (tileScreenRect(tcx, tcy, thalf, rect)) {
                            reflectionWaterRects.push_back(rect);
                            // Stage-2: raster this tile's wet cells into the fine mask
                            // (same gate as the rect → "raster the rect-emitting tiles only").
                            // Close tiles use the dense runtime near-grid (triangle-rastered
                            // real terrain) for a tight, hole-free near silhouette. Omitted
                            // while swimming (degenerate edge-on projection).
                            if (maskUsable) {
                                if (closeTile) rasterCloseTile(tcx, tcy, thalf);
                                else           rasterTileWetCells(tcx, tcy, thalf);
                            }
                        }
                    }
                }
            }
        }

        if (debug) {
            g_waterProxyDebugBoxes.push_back({
                tcx - thalf, tcx + thalf, tcy - thalf, tcy + thalf,
                waterZ - 128.0f, waterZ + 128.0f, dbg
            });
        }
    };

    const int cx0 = (int)floorf((eyePos.x - R) / kCellSize);
    const int cx1 = (int)floorf((eyePos.x + R) / kCellSize);
    const int cy0 = (int)floorf((eyePos.y - R) / kCellSize);
    const int cy1 = (int)floorf((eyePos.y + R) / kCellSize);
    const int eyeCellX = (int)floorf(eyePos.x / kCellSize);
    const int eyeCellY = (int)floorf(eyePos.y / kCellSize);

    // Refresh the dense near-camera height grid (triangle-rastered real terrain) for the
    // 3x3 cells the close-tile silhouette raster reads. No-op unless the eye crossed a cell.
    // Skipped while swimming (mask omitted) — no point building what won't be read.
    if (maskUsable)
        buildNearHeightGrid(eyeCellX, eyeCellY);

    for (int cy = cy0; cy <= cy1; ++cy) {
        for (int cx = cx0; cx <= cx1; ++cx) {
            const float ccx = (cx + 0.5f) * kCellSize;
            const float ccy = (cy + 0.5f) * kCellSize;

            // Range gate on cell centre (loop bounds are a square; trim to R).
            const float dx = ccx - eyePos.x, dy = ccy - eyePos.y;
            if (dx * dx + dy * dy > (R + half) * (R + half)) continue;

            // Close cells (the 3x3 block around the player) split into a 9x9 fine
            // sub-tile grid so sub-cell water (a canal beside a street) resolves
            // separately and near tiles below the view get frustum-culled.
            const bool close = (abs(cx - eyeCellX) <= 1) && (abs(cy - eyeCellY) <= 1);

            if (close) {
                const float subHalf = half / 9.0f;     // ~455
                const float step    = 2.0f * subHalf;  // kCellSize / 9
                for (int sy = -4; sy <= 4; ++sy) {
                    for (int sx = -4; sx <= 4; ++sx) {
                        testTile(ccx + sx * step, ccy + sy * step, subHalf, true);
                    }
                }
            } else {
                testTile(ccx, ccy, half, false);
            }
        }
    }

    g_reflPipe.rects = (int)reflectionWaterRects.size();   // [REFL PIPE] visible water rects
    g_reflRectsComputed = true;   // tile loop ran → empty rects now means "no in-range water"
    // Stage-2 mask is meaningful only when it was actually rastered (eye above water).
    // Swimming → leave it invalid so reflWaterMaskTestNDC keeps all (rect-only cull).
    g_reflWaterMaskValid = maskUsable;

    // [REFL PIPE FLIP] Diagnose the flickering tiles: any in-range tile whose visible-
    // state changed since last frame (symmetric difference of the visible sets). Report
    // their farness (horizontal distance, in cells) and tiltedness (elevation angle of
    // the eye->tile ray above the water plane; grazing/tilted = small angle). Decodes
    // the 64-unit-quantized tile centre back out of the hysteresis key.
    {
        int flips = 0;
        float dMinC = 1e9f, dMaxC = -1e9f, eMin = 1e9f, eMax = -1e9f;
        auto note = [&](int64_t tkey) {
            const float tcx = (float)(int)(tkey >> 32) * 64.0f;
            const float tcy = (float)(int)(int32_t)(uint32_t)tkey * 64.0f;
            const float dx = tcx - eyePos.x, dy = tcy - eyePos.y;
            const float dist = sqrtf(dx * dx + dy * dy);
            const float distCells = dist / kCellSize;
            const float elevDeg = atan2f(eyeAboveWater, std::max(1.0f, dist)) * (180.0f / 3.14159265f);
            ++flips;
            dMinC = std::min(dMinC, distCells); dMaxC = std::max(dMaxC, distCells);
            eMin = std::min(eMin, elevDeg);     eMax = std::max(eMax, elevDeg);
        };
        for (int64_t k : g_reflWaterVisThis) if (g_reflWaterVisLast.count(k) == 0) note(k);
        for (int64_t k : g_reflWaterVisLast) if (g_reflWaterVisThis.count(k) == 0) note(k);
        g_reflPipe.flips            = flips;
        g_reflPipe.flipDistMinCells = flips ? dMinC : 0.0f;
        g_reflPipe.flipDistMaxCells = flips ? dMaxC : 0.0f;
        g_reflPipe.flipElevMinDeg   = flips ? eMin  : 0.0f;
        g_reflPipe.flipElevMaxDeg   = flips ? eMax  : 0.0f;
    }

    g_reflWaterVisLast.swap(g_reflWaterVisThis);   // this frame's visible tiles → last
    return anyVisible;
}

// Materialize visExtraShared (windowed IPC view) into stable contiguous storage
// ONE time. Must run while the channel is owned by this caller (no concurrent
// IPC traffic). Mirrors the distant-statics materialize in applyMSOCToDistantStatics.
void DistantLand::materializeReflectionMeshes() {
    g_reflMeshValues.clear();
    g_reflMeshValues.reserve(visExtraShared.Size());
    visExtraShared.Reset();
    while (!visExtraShared.AtEnd())
        g_reflMeshValues.push_back(visExtraShared.Next());
}

// Cull the materialized reflection meshes into reflectionSurvivors: keep a static
// only if its reflection (projected via the reflection view*proj) lands on a
// surviving water tile's screen rect. Pure CPU over the stable copy + rects — no
// IPC window access, so it's safe on the worker after channelDrained.
void DistantLand::cullReflectionSurvivors(const D3DXMATRIX& viewProj, const D3DXMATRIX& proj) {
    reflectionSurvivors.RemoveAll();
    reflectionSurvivors.visible_set.reserve((std::uint32_t)g_reflMeshValues.size());

    const auto& waterRects = reflectionWaterRects;
    const bool noRects = waterRects.empty();
    // Empty rects has two meanings: the gate RAN and found no in-range water (cull all
    // statics — none have near water to reflect in), or the gate couldn't compute
    // (interior / no terrain data → keep all, legacy fallback). g_reflRectsComputed
    // distinguishes them. With rects (the common case) the per-static test below decides.
    const bool keepWhenNoRects = !g_reflRectsComputed;
    for (const RenderMesh& m : g_reflMeshValues) {
        // Water cull unreliable (interior / no data / swimming) → keep all.
        if (!reflWaterCullActive) { reflectionSurvivors.PushBack(m); continue; }
        bool keep = keepWhenNoRects;
        if (!noRects) {
            const D3DXVECTOR3& c = m.sphere.center;
            const float rad = m.sphere.radius;
            const float cx = c.x*viewProj._11 + c.y*viewProj._21 + c.z*viewProj._31 + viewProj._41;
            const float cy = c.x*viewProj._12 + c.y*viewProj._22 + c.z*viewProj._32 + viewProj._42;
            const float cw = c.x*viewProj._14 + c.y*viewProj._24 + c.z*viewProj._34 + viewProj._44;
            if (cw < 1e-3f) {
                keep = true;   // behind near plane → conservatively keep
            } else {
                const float inv = 1.0f / cw;
                const float nx = cx * inv, ny = cy * inv;
                const float rx = rad * proj._11 * inv;   // sphere radius in NDC
                const float ry = rad * proj._22 * inv;
                keep = false;
                for (const D3DXVECTOR4& wr : waterRects) {   // stage 1: coarse rect
                    if (nx + rx >= wr.x && nx - rx <= wr.z &&
                        ny + ry >= wr.y && ny - ry <= wr.w) { keep = true; break; }
                }
                // Stage 2: fine water-silhouette mask refines the rect survivors,
                // dropping objects flanking a thin/diagonal river. Only run on rect
                // survivors; fail → cull.
                if (keep) keep = reflWaterMaskTestNDC(nx, ny, rx, ry);
            }
        }
        if (keep) reflectionSurvivors.PushBack(m);
    }
    reflectionSurvivors.SortByState();
    g_reflPipe.queried   = (int)g_reflMeshValues.size();       // [REFL PIPE] frustum-query input
    g_reflPipe.survivors = (int)reflectionSurvivors.Size();    // [REFL PIPE] after water-rect cull
    MGE_TracyPlot("reflStatics:count", double(g_reflMeshValues.size()));
    MGE_TracyPlot("reflStatics:culled", double(g_reflMeshValues.size() - reflectionSurvivors.Size()));
}

// Stage-2 fine cull (shared by statics + cache). Keep (true) if the footprint NDC
// AABB overlaps any set water-silhouette bit, OR the mask is invalid (interior /
// no-data fallback — mirrors the empty-rects keep). A fully off-screen footprint maps
// to no cells → no water → cull. Water-on-plane means the caller's reflection-
// projected footprint indexes the same mask the main camera built.
bool DistantLand::reflWaterMaskTestNDC(float nx, float ny, float rx, float ry) {
    if (!g_reflWaterMaskValid) return true;
    int ix0, iy0, ix1, iy1;
    if (!reflMaskCellRange(nx - rx, ny - ry, nx + rx, ny + ry, ix0, iy0, ix1, iy1))
        return false;
    for (int y = iy0; y <= iy1; ++y) {
        const int base = y * kReflMaskW;
        for (int x = ix0; x <= ix1; ++x) {
            const int bit = base + x;
            if (g_reflWaterMask[bit >> 6] & (uint64_t(1) << (bit & 63))) return true;
        }
    }
    return false;
}

// Debug: popcount of the current silhouette mask (0 when invalid). Reported in
// [REFL DBG] to quantify the rasterized water area.
int DistantLand::reflWaterMaskSetBits() {
    if (!g_reflWaterMaskValid) return 0;
    int n = 0;
    for (int i = 0; i < kReflMaskWords; ++i) {
        uint64_t w = g_reflWaterMask[i];
        while (w) { w &= (w - 1); ++n; }
    }
    return n;
}

// Main (frameSetupEarly): decide whether the cull worker should handle the
// reflection this frame, and stash the reflection cull frustum/projection. Only
// called when the worker was dispatched (USE_DISTANT_STATICS path), so
// reflGateWanted=true implies the worker will run.
void DistantLand::prepareReflectionCullForWorker() {
    reflGateWanted = false;
    reflStaticsWanted = false;

    auto mwBridge = MWBridge::get();
    if (!isDistantCell() || !mwBridge->CellHasWater()) return;
    // Forge owns the water surface (F7): MGE water is suppressed (renderStageWater early-returns),
    // so its reflection RT has no visible consumer. Skip the ENTIRE reflection cull this frame —
    // worker gate, the statics RPC fold, and (at the draw site) the draw — by leaving reflGateWanted
    // false. No fence to consume, no inline cull. F7-off restores the full MGE reflection (clean A/B).
    if (RenderProcess::wantsWaterCapture()) return;
    reflGateWanted = true;   // worker runs the reflect-vs-clear gate

    // Reflection statics RPC only when near-static reflections are enabled.
    if (!(Configuration.MGEFlags & REFLECT_NEAR)) return;

    const float zn = 4.0f;
    const float zf = std::min(fogEnd, Configuration.DL.NearStaticEnd * kCellSize);
    if (zf <= zn) return;

    // Mirror the view across the water plane (matches renderWaterReflection).
    D3DXMATRIX reflView;
    D3DXPLANE plane(0, 0, 1.0f, -(mwBridge->WaterLevel() - 1.0f));
    D3DXMatrixReflect(&reflView, &plane);
    D3DXMatrixMultiply(&reflView, &reflView, &mwView);

    D3DXMATRIX ds_proj = mwProj;
    editProjectionZ(&ds_proj, zn, zf);

    reflCullProj       = ds_proj;
    reflCullViewProj   = reflView * ds_proj;
    reflCullViewSphere = D3DXVECTOR4(eyePos.x, eyePos.y, eyePos.z, zf);
    reflStaticsWanted  = true;

    // The reflection query is folded into the batched statics RPC issued by
    // cullDistantStatics_kickoff (called right after this), so clear the output
    // vec here before the RPC writes it (client convention: RemoveAll pre-RPC).
    visExtraShared.RemoveAll();
}

// Worker step (after the statics verdict, before reflDone): run the water-visible
// gate and build the reflection skipMask. Both are pure CPU over frame-stable
// inputs (terrain maps, MSOC mask, stashed reflection matrices).
void DistantLand::workerReflectionGateAndMask() {
    if (!reflGateWanted) return;
    {
        MGE_ZoneScopedN("cullWorker:reflGate");
        reflVisible = isReflectionWaterVisible();   // fills reflectionWaterRects
    }
    if (reflStaticsWanted) {
        MGE_ZoneScopedN("cullWorker:reflMask");
        cullReflectionSurvivors(reflCullViewProj, reflCullProj);
    }
}
