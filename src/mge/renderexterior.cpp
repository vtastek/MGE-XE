
#include "distantland.h"
#include "distantshader.h"
#include "configuration.h"
#include "msocclient.h"
#include "mwbridge.h"
#include "phasetimers.h"
#include "proxydx/d3d8header.h"
#include "support/log.h"
#include "statusoverlay.h"
#include "terrain_horizon_occluder.h"
#include "mge_tracy.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>
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
// + bool flags. Two sync points:
//   - channelDrained: the worker has finished its tryWaitForCompletion drain
//     of the statics RPC, so the single-channel ipcClient is free for the
//     main thread's land/grass/shadow/water RPCs. renderStage0 waits on this
//     at entry before any main-thread ipcClient touch.
//   - maskDone: the verdict core has finished writing msocOccluded.
//     cullDistantStatics_finish waits on this in place of wait+applyMSOC.
std::thread             g_cullThread;
std::mutex              g_cullMtx;
std::condition_variable g_cullCv;
bool g_cullPending        = false; // a finish job has been signalled
bool g_cullChannelDrained = false; // statics RPC drained off ipcClient
bool g_cullMaskDone       = false; // verdict core finished; msocOccluded stable
bool g_cullStop           = false; // shutdown request
bool g_cullStarted        = false; // lazy-init flag

// Per-frame summary stashed by the verdict core (worker) and read by the
// main-thread debug tail's 60-frame log block. Stable once maskDone joins.
struct MSOCCullDiag {
    bool     valid = false;
    unsigned setSize = 0;
    unsigned nSphere = 0;
    unsigned groupCount = 0;
    int      groupsOccluded = 0;
    int      lowCulled = 0;
    int      sphereCulled = 0;
};
MSOCCullDiag g_msocCullDiag;
} // namespace

// True when this frame's verdict pass was dispatched to the worker (set by
// signalCullFinish, consumed by the wait/finish helpers). Main-thread only.
static bool s_cullOnWorker = false;



// renderSky - Render atmosphere scattering sky layer and other recorded draw calls on top
void DistantLand::renderSky() {
    MGE_ZoneScopedN("renderSky");
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
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);
    }
    effect->EndPass();
}

void DistantLand::renderDistantLand(ID3DXEffect* e, const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    MGE_SCOPED_TIMER("renderDistantLand");
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
void DistantLand::contributeDistantLandOccluders() {
    MGE_ZoneScopedN("contributeOccluders");
    MGE_SCOPED_TIMER("contributeDistantLandOccluders");

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
//   1. drains the statics RPC off the single IPC channel (frees the channel
//      for the main thread's other RPCs) and signals channelDrained,
//   2. runs the pure verdict core over the freshly-drained set and signals
//      maskDone.
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

        // Drain the statics RPC. Guarded (tryWaitForCompletion): an
        // interleaved drain — there shouldn't be one before renderStage0's
        // channel-free gate, but stay defensive — may already have completed
        // it, in which case there's no pending RPC to wait on.
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
        {
            std::lock_guard<std::mutex> lk(g_cullMtx);
            g_cullMaskDone = true;
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
void waitCullMaskReady() {
    MGE_ZoneScopedN("finish:joinMask");
    std::unique_lock<std::mutex> lk(g_cullMtx);
    g_cullCv.wait(lk, []{ return g_cullMaskDone; });
}

// Main-thread debug tail for the MSOC verdict: Numpad5 mask dump + the
// per-60-frame cull summary. Touches input / MSOCClient::dumpMask / LOG, so
// it stays off the worker; reads g_msocCullDiag, stable after the join.
void runMSOCDebugTail() {
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

    static int diagFrameCounter = 0;
    if (g_msocCullDiag.valid && (diagFrameCounter++ % 60) == 0) {
        const MSOCCullDiag& d = g_msocCullDiag;
        const int totalCulled = d.lowCulled + d.sphereCulled;
        LOG::logline(
            "-- MSOC cull: statics=%u  low=%u(groups=%u occ=%d culled=%d)"
            "  sphere=%u(culled=%d)  total=%d(%d%%)",
            d.setSize,
            d.setSize - d.nSphere, d.groupCount,
            d.groupsOccluded, d.lowCulled,
            d.nSphere, d.sphereCulled,
            totalCulled,
            d.setSize > 0 ? (totalCulled * 100) / d.setSize : 0);
    }
}
} // namespace

// Numpad8/2: raise/lower the MSOC low-static cutoff height (256-unit steps).
// Read on the main thread so g_msocCutoffHeight is final before the verdict
// core (worker or inline) reads it.
void DistantLand::updateMSOCCutoffInput() {
    if (GetAsyncKeyState(VK_NUMPAD8) & 0x0001) {
        g_msocCutoffHeight += 256.0f;
        char msg[64];
        std::snprintf(msg, sizeof(msg), "MSOC cutoff: %.0f units", g_msocCutoffHeight);
        StatusOverlay::setStatus(msg);
    }
    if (GetAsyncKeyState(VK_NUMPAD2) & 0x0001) {
        g_msocCutoffHeight = std::max(0.0f, g_msocCutoffHeight - 256.0f);
        char msg[64];
        std::snprintf(msg, sizeof(msg), "MSOC cutoff: %.0f units", g_msocCutoffHeight);
        StatusOverlay::setStatus(msg);
    }
}

// Dispatch the verdict pass to the cull worker. Called from frameSetupEarly
// right after cullDistantStatics_kickoff issues the statics RPC.
void DistantLand::signalCullFinish() {
    ensureCullWorker();
    {
        std::lock_guard<std::mutex> lk(g_cullMtx);
        g_cullChannelDrained = false;
        g_cullMaskDone       = false;
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
    g_cullMaskDone       = false;
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
        ipcClient.getVisibleMeshesAllRanges(
            visDistantSharedId, 3, frustums, spheres, setFlags,
            VisibleSetSort::None);
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
        // (The Numpad cutoff was read on main in frameSetupEarly.)
        waitCullMaskReady();
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
    if (!MWBridge::get()->IsExterior()) {
        // Set clipping to stop large architectural meshes (that don't match exactly)
        // from visible overdrawing and causing z-buffer occlusion
        float clipAt = nearViewRange - 768.0f;
        D3DXPLANE clipPlane(0, 0, clipAt, -(mwProj._33 * clipAt + mwProj._43));
        device->SetClipPlane(0, clipPlane);
        device->SetRenderState(D3DRS_CLIPPLANEENABLE, 1);
    }

    device->SetVertexDeclaration(StaticDecl);

    // Cull-then-sort: iterate the compacted, state-sorted survivor set built
    // by applyMSOCToDistantStatics (IPC and non-IPC paths alike). The old
    // skipMask plumbing is gone — occluded instances are already absent.
    visDistantSurvivors.Render(device, effect, effect, &ehTex0, nullptr, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT, false);

    device->SetRenderState(D3DRS_CLIPPLANEENABLE, 0);
}

void DistantLand::renderMSOCBasinBoundsDebug(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    if (GetAsyncKeyState(VK_NUMPAD9) & 0x0001) {
        g_drawMSOCBasinBounds = !g_drawMSOCBasinBounds;
        char msg[64];
        std::snprintf(msg, sizeof(msg), "MSOC basin boxes: %s",
                      g_drawMSOCBasinBounds ? "ON" : "OFF");
        StatusOverlay::setStatus(msg);
    }

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

    device->DrawPrimitiveUP(D3DPT_LINELIST, (UINT)(lineVerts.size() / 2),
                            lineVerts.data(), sizeof(MSOCLineVertex));

    stateSaved->Apply();
    stateSaved->Release();
}

// Water-reflection proxy visualizer (Numpad5). Draws the per-cell water slabs
// tested by isReflectionWaterVisible(), coloured by MSOC verdict (green visible,
// red occluded, blue view-culled), depth-disabled so occluded slabs are still
// inspectable behind buildings. Mirrors renderMSOCBasinBoundsDebug.
void DistantLand::renderWaterProxyBoundsDebug(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    if (GetAsyncKeyState(VK_NUMPAD5) & 0x0001) {
        g_drawWaterProxyBounds = !g_drawWaterProxyBounds;
        char msg[64];
        std::snprintf(msg, sizeof(msg), "Water proxy boxes: %s",
                      g_drawWaterProxyBounds ? "ON" : "OFF");
        StatusOverlay::setStatus(msg);
    }

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

    device->DrawPrimitiveUP(D3DPT_LINELIST, (UINT)(lineVerts.size() / 2),
                            lineVerts.data(), sizeof(MSOCLineVertex));

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
    const bool cullByOcclusion =
        setSize != 0
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
    // Sphere batch for high statics and near low statics.
    static std::vector<float>                   sphereBatch;
    static std::vector<MSOCClient::TestResult>  sphereResults;

    groups.clear();
    groups.reserve(64);
    sphereBatch.clear();
    staticGroupIdx.assign(setSize, 0xFFFF);
    gridCellToGroup.assign((size_t)gridDim * gridDim, 0);

    // Pass 1: partition statics. Low → fixed cell group (O(1)).  High → sphere batch.
    // Iterates the contiguous g_meshValues snapshot (not the windowed view).
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

                // Near handoff / out-of-grid → per-static sphere path.
                const float minDistSq = distanceSqToAABB2D(
                    eyePos.x, eyePos.y, x0, x0 + gSize, y0, y0 + gSize);
                const int lx = gx - eyeGx + gridRadius;
                const int ly = gy - eyeGy + gridRadius;
                if (minDistSq <= groupCullStartDistSq ||
                    lx < 0 || lx >= gridDim || ly < 0 || ly >= gridDim) {
                    sphereBatch.push_back(sx);
                    sphereBatch.push_back(sy);
                    sphereBatch.push_back(sz);
                    sphereBatch.push_back(sr);
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
            } else {
                // High static: per-static sphere test.
                sphereBatch.push_back(sx);
                sphereBatch.push_back(sy);
                sphereBatch.push_back(sz);
                sphereBatch.push_back(sr);
            }
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

    // Pass 2b: sphere batch for high statics and near low statics.
    // Results land directly in msocOccluded via staticGroupIdx == 0xFFFF slots.
    int diagSphereCulled = 0;
    if (!sphereBatch.empty()) {
        MGE_SCOPED_TIMER("applyMSOCToDistantStatics:sphereBatch");
        const int nSpheres = (int)(sphereBatch.size() / 4);
        sphereResults.assign(nSpheres, MSOCClient::ResultVisible);
        MSOCClient::classifySphereBatch(sphereBatch.data(), nSpheres, sphereResults.data());
        // Map results back: sphere-tested statics appear in staticGroupIdx order (0xFFFF slots).
        int si = 0;
        for (unsigned idx = 0; idx < setSize; ++idx) {
            if (staticGroupIdx[idx] == 0xFFFF) {
                if (sphereResults[si] == MSOCClient::ResultOccluded) {
                    msocOccluded[idx] = 1;
                    ++diagSphereCulled;
                }
                ++si;
            }
        }
    }

    // Pass 3: propagate group verdicts — direct array lookup, no recomputation.
    int diagLowCulled = 0;
    {
        for (unsigned idx = 0; idx < setSize; ++idx) {
            const std::uint16_t gIdx = staticGroupIdx[idx];
            if (gIdx != 0xFFFF && groups[gIdx].verdict == MSOCClient::ResultOccluded) {
                msocOccluded[idx] = 1;
                ++diagLowCulled;
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
static std::unordered_map<uint64_t, float> g_fineMinH;   // key: fine-cell -> min Z
static size_t g_terrainMinHSrcSize = (size_t)-1;         // landMeshes.size() built from

static inline uint64_t waterGridKey(int gx, int gy) {
    return ((uint64_t)(uint32_t)gx << 32) | (uint32_t)gy;
}

static void buildTerrainMinHeight(
    const std::unordered_map<IDirect3DVertexBuffer9*, DistantLand::LandMeshCache>& landMeshes) {
    g_cellMinH.clear();
    g_fineMinH.clear();
    for (const auto& kv : landMeshes) {
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
        }
    }
    g_terrainMinHSrcSize = landMeshes.size();
    LOG::logline("-- [water-gate] terrain min-height built: %zu cells, %zu fine tiles",
                 g_cellMinH.size(), g_fineMinH.size());
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

    const bool debug = g_drawWaterProxyBounds;
    if (debug) g_waterProxyDebugBoxes.clear();

    // Build/refresh the terrain maps if the captured land set changed (init or
    // release). Built once per session in practice.
    if (g_terrainMinHSrcSize != landMeshes.size()) {
        buildTerrainMinHeight(landMeshes);
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
    const float R = fogEnd;                 // world units; water visible to fog
    const float half = 0.5f * kCellSize;    // cell half-extent (4096)
    const float slabHZ = 4.0f;              // thin water box half-height

    bool anyVisible = false;
    reflectionWaterRects.clear();

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

    // Test one water tile: frustum-cull the thin box, terrain water-presence,
    // then MSOC occlusion on the survivor. A surviving (visible water) tile
    // contributes its screen rect to reflectionWaterRects for the Phase-B static
    // cull. No early-out — the full survivor set is the cull input. Visualizer:
    //   green  = water present and not occluded → reflects
    //   red    = water present but MSOC-occluded → culled
    //   yellow = dry land at/above water level → no water
    //   (no-data tiles are very distant; dropped and not drawn)
    auto testTile = [&](float tcx, float tcy, float thalf) {
        BoundingBox wbox(
            D3DXVECTOR3(tcx - thalf, tcy - thalf, waterZ - slabHZ),
            D3DXVECTOR3(tcx + thalf, tcy + thalf, waterZ + slabHZ));
        if (frustum.ContainsBox(wbox) == ViewFrustum::OUTSIDE) return;

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
            MSOCClient::TestResult r = msocUsable
                ? MSOCClient::classifyOBB(tcx, tcy, waterZ,
                                          thalf, 0, 0,  0, thalf, 0,  0, 0, slabHZ)
                : MSOCClient::ResultVisible;
            vis = (r != MSOCClient::ResultOccluded);
            dbg = vis ? MSOCClient::ResultVisible : MSOCClient::ResultOccluded; // green/red
            if (vis) {
                anyVisible = true;
                D3DXVECTOR4 rect;
                if (tileScreenRect(tcx, tcy, thalf, rect))
                    reflectionWaterRects.push_back(rect);
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
                        testTile(ccx + sx * step, ccy + sy * step, subHalf);
                    }
                }
            } else {
                testTile(ccx, ccy, half);
            }
        }
    }

    return anyVisible;
}
