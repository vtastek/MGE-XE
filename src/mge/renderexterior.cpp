
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
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {
// Horizon-curtain workspace — kept for shutdownHorizonWorkspace; no longer populated.
thc_horizon_t           g_horizon{};
thc_simplify_workspace_t g_horizonWs{};
bool                    g_horizonInitialized = false;
} // namespace

// Numpad8/Numpad2: raise/lower MSOC low-static cutoff height (steps of 256 units).
static float g_msocCutoffHeight = 2500.0f;

namespace {
struct MSOCBasinDebugBox {
    float minX, maxX, minY, maxY, minZ, maxZ;
    MSOCClient::TestResult verdict;
};

static bool g_drawMSOCBasinBounds = false;
static std::vector<MSOCBasinDebugBox> g_msocBasinDebugBoxes;

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
} // namespace



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
    MGE_ZoneScopedN("renderDistantLand");
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

void DistantLand::cullDistantStatics_kickoff(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    MGE_ZoneScopedN("cullDistantStatics_kickoff");
    MGE_SCOPED_TIMER("cullDistantStatics:kickoff");

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
        ipcClient.getVisibleMeshesAllRanges(
            visDistantSharedId, 3, frustums, spheres, setFlags,
            VisibleSetSort::ByState);
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
        visDistant.SortByState();
    }
}

void DistantLand::cullDistantStatics_finish() {
    MGE_ZoneScopedN("cullDistantStatics_finish");
    MGE_SCOPED_TIMER("cullDistantStatics:finish");

    if (Configuration.UseSharedMemory) {
        {
            MGE_SCOPED_TIMER("cullDistantStatics:finishWait");
            ipcClient.waitForCompletion();
        }
    }

    // Per-frame cull summary + phase-timer flush. Gated by the
    // LogDistantPipeline config flag — off by default. IPC path logs
    // total only (bucket split is now collapsed inside the batched RPC);
    // non-IPC logs the breakdown captured during _kickoff.
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

    // MSOC verdict pass runs unconditionally so both instanced AND
    // non-instanced render paths consume the same cull mask. Must run
    // after the sort so msocOccluded[idx] aligns with each render
    // path's visible-set iteration order.
    if (Configuration.UseSharedMemory) {
        applyMSOCToDistantStatics(visDistantShared);
    } else {
        applyMSOCToDistantStatics(visDistant);
    }
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

    // Pass the prebuilt MSOC cull mask through the render helper's
    // optional skipMask parameter. Empty mask = no culling.
    const std::uint8_t* skipMask = msocOccluded.empty() ? nullptr : msocOccluded.data();
    if (Configuration.UseSharedMemory) {
        visDistantShared.Render(device, effect, effect, &ehTex0, nullptr, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT, false, skipMask);
    } else {
        visDistant.Render(device, effect, effect, &ehTex0, nullptr, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT, false, skipMask);
    }

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
    if (staticSet.Empty())
        return;

    if (!MWBridge::get()->IsExterior())
        return;

    const bool cullByOcclusion =
        Configuration.UseOcclusionCulling
        && MSOCClient::isAvailable()
        && MSOCClient::isMaskReady();
    if (!cullByOcclusion)
        return;

    const unsigned setSize = (unsigned)staticSet.Size();
    msocOccluded.resize(setSize, 0);

    // Numpad8/2: adjust cutoff height live.
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

    // Statics whose bounding sphere top sits at or below waterLevel + kTallness are
    // bucketed into flat per-cell-group OBBs and tested as a batch (~33 calls).
    // Statics above the cutoff (hillside objects, towers) go through the original
    // per-static sphere batch — they are never grouped and never falsely culled.
    const float kTallness  = g_msocCutoffHeight;
    const float waterLevel = MWBridge::get()->WaterLevel();
    const float cutoffZ    = waterLevel + kTallness;

    // Distance thresholds for group-size LOD (squared, avoids sqrt per static).
    // Near  (<4 cells): 1x1-cell OBBs.
    // Mid   (<8 cells): 2x2-cell OBBs.
    // Far   (8+ cells): 3x3-cell OBBs.
    const float kNearDistSq = (4.0f * kCellSize) * (4.0f * kCellSize);
    const float kMidDistSq  = (8.0f * kCellSize) * (8.0f * kCellSize);
    const float groupCullStartDist = nearViewRange + kCellSize;
    const float groupCullStartDistSq = groupCullStartDist * groupCullStartDist;

    // Flat group list — typically ≤ 64 entries, linear search is cache-friendly
    // and cheaper than unordered_map for this count.
    struct GroupEntry {
        int gx, gy, gCells;
        float minX, maxX, minY, maxY;
        MSOCClient::TestResult verdict;
    };
    static std::vector<GroupEntry>              groups;
    // Per-static group index; 0xFFFF = handled by sphere batch.
    static std::vector<std::uint16_t>           staticGroupIdx;
    // Sphere batch for high statics and near low statics.
    static std::vector<float>                   sphereBatch;
    static std::vector<MSOCClient::TestResult>  sphereResults;

    groups.clear();
    groups.reserve(64);
    sphereBatch.clear();
    staticGroupIdx.assign(setSize, 0xFFFF);

    // Pass 1: partition statics. Low → cell group (linear search).  High → sphere batch.
    {
        MGE_SCOPED_TIMER("applyMSOCToDistantStatics:partition");
        staticSet.Reset();
        unsigned idx = 0;
        while (!staticSet.AtEnd()) {
            const auto& m = staticSet.Next();
            const float sx = m.sphere.center.x, sy = m.sphere.center.y;
            const float sz = m.sphere.center.z, sr = m.sphere.radius;

            if (sz + sr <= cutoffZ) {
                // Low static: use grouped OBBs only after the near handoff band.
                // Close low statics use the original per-static sphere path.
                const float dx  = sx - eyePos.x;
                const float dy  = sy - eyePos.y;
                const float dSq = dx * dx + dy * dy;
                const int gCells = (dSq < kNearDistSq) ? 1 : (dSq < kMidDistSq) ? 2 : 3;
                const float gSize = gCells * kCellSize;
                const int gx = (int)floorf(sx / gSize);
                const int gy = (int)floorf(sy / gSize);
                const float x0 = (float)gx * gSize;
                const float y0 = (float)gy * gSize;
                const float minDistSq = distanceSqToAABB2D(
                    eyePos.x, eyePos.y, x0, x0 + gSize, y0, y0 + gSize);
                if (minDistSq <= groupCullStartDistSq) {
                    sphereBatch.push_back(sx);
                    sphereBatch.push_back(sy);
                    sphereBatch.push_back(sz);
                    sphereBatch.push_back(sr);
                    ++idx;
                    continue;
                }

                // Linear scan — groups count is tiny (~33), fits in a cache line or two.
                std::uint16_t gIdx = (std::uint16_t)groups.size();
                for (std::uint16_t i = 0; i < (std::uint16_t)groups.size(); ++i) {
                    if (groups[i].gx == gx && groups[i].gy == gy && groups[i].gCells == gCells) {
                        gIdx = i;
                        break;
                    }
                }
                if (gIdx == (std::uint16_t)groups.size()) {
                    groups.push_back({gx, gy, gCells, x0, x0 + gSize, y0, y0 + gSize,
                                      MSOCClient::ResultVisible});
                }
                staticGroupIdx[idx] = gIdx;
            } else {
                // High static: per-static sphere test.
                sphereBatch.push_back(sx);
                sphereBatch.push_back(sy);
                sphereBatch.push_back(sz);
                sphereBatch.push_back(sr);
            }
            ++idx;
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

    if (Configuration.LogDistantPipeline) {
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
        if ((diagFrameCounter++ % 60) == 0) {
            int diagGroupsOccluded = 0;
            for (const auto& g : groups)
                if (g.verdict == MSOCClient::ResultOccluded) ++diagGroupsOccluded;
            const int diagTotalCulled = diagLowCulled + diagSphereCulled;
            const unsigned nSphere = (unsigned)(sphereBatch.size() / 4);
            LOG::logline(
                "-- MSOC cull: statics=%u  low=%u(groups=%u occ=%d culled=%d)"
                "  sphere=%u(culled=%d)  total=%d(%d%%)",
                setSize,
                setSize - nSphere, (unsigned)groups.size(),
                diagGroupsOccluded, diagLowCulled,
                nSphere, diagSphereCulled,
                diagTotalCulled,
                setSize > 0 ? (diagTotalCulled * 100) / setSize : 0);
        }
    }
}

template void DistantLand::applyMSOCToDistantStatics(VisibleSet<StlVector>& staticSet);
template void DistantLand::applyMSOCToDistantStatics(VisibleSet<IpcClientVector>& staticSet);
