
#include "distantland.h"
#include "distantshader.h"
#include "configuration.h"
#include "msocclient.h"
#include "mwbridge.h"
#include "phasetimers.h"
#include "proxydx/d3d8header.h"
// MOREFPS-INSTANCING: drop this include + the StaticInstancing::buildVB dispatch in cullDistantStatics to remove instancing.
#include "staticinstancing.h"
#include "support/log.h"
#include "terrain_horizon_occluder.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

// MSOC temporal hysteresis.
// MSOC verdicts can be unstable for small / distant statics whose
// projected rect is a few pixels wide — a 1-2 px camera jitter flips
// the verdict frame-to-frame, producing visible flicker.
//
// Fix: track per-static "consecutive OCCLUDED" counter. Only cull once
// the static has been reported OCCLUDED for the user-configured number
// of frames in a row (mge.ini [Misc] "Occlusion Hysteresis Frames",
// default 8). A single VISIBLE verdict resets the counter, so brief
// cull spikes don't translate into visible flicker.
//
// Key = packed hash of sphere center XYZ. Statics don't move, so the
// position is stable across frames and across the IPC boundary (sphere
// data comes from the wire format, identical on both sides). Collisions
// would mean two statics at the exact same XYZ share a counter — rare
// in practice; worst outcome is "they cull together," still safer than
// flicker.
namespace {
struct MsocHysteresisState {
    std::uint8_t  consecutiveOccluded;
    std::uint32_t lastSeenFrame;
};
std::unordered_map<std::uint64_t, MsocHysteresisState> g_msocHysteresis;
std::uint32_t g_msocHysteresisFrame = 0;
constexpr std::uint32_t kHysteresisAgeOutFrames = 60;

inline std::uint64_t hashStaticKey(float x, float y, float z) {
    std::uint32_t ix, iy, iz;
    std::memcpy(&ix, &x, 4);
    std::memcpy(&iy, &y, 4);
    std::memcpy(&iz, &z, 4);
    // splitmix-style mixing. Three primes are odd / coprime; XOR
    // combine spreads bits across the 64-bit hash so close-XYZ
    // statics don't collapse into the same bucket.
    std::uint64_t h = (std::uint64_t)ix * 0x9E3779B97F4A7C15ull;
    h ^= (std::uint64_t)iy * 0xBF58476D1CE4E5B9ull;
    h ^= (std::uint64_t)iz * 0x94D049BB133111EBull;
    return h;
}

// Horizon-curtain workspace. Lives at file scope so
// DistantLand::shutdownHorizonWorkspace can free it on release. Lazily
// initialized on first call to contributeDistantLandOccluders.
thc_horizon_t           g_horizon{};
thc_simplify_workspace_t g_horizonWs{};
bool                    g_horizonInitialized = false;
} // namespace



// renderSky - Render atmosphere scattering sky layer and other recorded draw calls on top
void DistantLand::renderSky() {
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

void DistantLand::cullDistantStatics(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    MGE_SCOPED_TIMER("cullDistantStatics");
    D3DXMATRIX ds_proj = *proj, ds_viewproj;
    D3DXVECTOR4 viewsphere(eyePos.x, eyePos.y, eyePos.z, 0);
    float zn = nearViewRange - 768.0f, zf = zn;
    float cullDist = fogEnd;

    if (Configuration.UseSharedMemory) {
        visDistantShared.RemoveAll();
    } else {
        visDistant.RemoveAll();
    }

    // Per-bucket diagnostics: counts only available in the
    // non-IPC path (GetVisibleMeshes is synchronous). In IPC mode
    // getVisibleMeshes is async and the totals only materialize after
    // waitForCompletion below — we then log the total without a
    // bucket breakdown. Snapshotted via size deltas across the three
    // range fetches.
    const bool useIpc = Configuration.UseSharedMemory;
    unsigned diagNearCount = 0, diagFarCount = 0, diagVeryFarCount = 0;
    unsigned prevSize = 0;

    zf = std::min(Configuration.DL.NearStaticEnd * kCellSize, cullDist);
    if (zn < zf) {
        editProjectionZ(&ds_proj, zn, zf);
        ds_viewproj = (*view) * ds_proj;
        ViewFrustum range_frustum(&ds_viewproj);
        viewsphere.w = zf;
        if (useIpc) {
            ipcClient.getVisibleMeshes(visDistantSharedId, range_frustum, viewsphere, VIS_NEAR);
        } else {
            DistantLandShare::currentWorldSpace->NearStatics->GetVisibleMeshes(range_frustum, viewsphere, visDistant);
            diagNearCount = (unsigned)visDistant.Size() - prevSize;
            prevSize = (unsigned)visDistant.Size();
        }
    }

    zf = std::min(Configuration.DL.FarStaticEnd * kCellSize, cullDist);
    if (zn < zf) {
        editProjectionZ(&ds_proj, zn, zf);
        ds_viewproj = (*view) * ds_proj;
        ViewFrustum range_frustum(&ds_viewproj);
        viewsphere.w = zf;
        if (useIpc) {
            ipcClient.getVisibleMeshes(visDistantSharedId, range_frustum, viewsphere, VIS_FAR);
        } else {
            DistantLandShare::currentWorldSpace->FarStatics->GetVisibleMeshes(range_frustum, viewsphere, visDistant);
            diagFarCount = (unsigned)visDistant.Size() - prevSize;
            prevSize = (unsigned)visDistant.Size();
        }
    }

    zf = std::min(Configuration.DL.VeryFarStaticEnd * kCellSize, cullDist);
    if (zn < zf) {
        editProjectionZ(&ds_proj, zn, zf);
        ds_viewproj = (*view) * ds_proj;
        ViewFrustum range_frustum(&ds_viewproj);
        viewsphere.w = zf;
        if (useIpc) {
            ipcClient.getVisibleMeshes(visDistantSharedId, range_frustum, viewsphere, VIS_VERY_FAR);
        } else {
            DistantLandShare::currentWorldSpace->VeryFarStatics->GetVisibleMeshes(range_frustum, viewsphere, visDistant);
            diagVeryFarCount = (unsigned)visDistant.Size() - prevSize;
            prevSize = (unsigned)visDistant.Size();
        }
    }

    if (useIpc) {
        ipcClient.sortVisibleSet(visDistantSharedId, VisibleSetSort::ByState);
        ipcClient.waitForCompletion();
    } else {
        visDistant.SortByState();
    }

    // Per-frame cull summary + phase-timer flush. Gated by the
    // LogDistantPipeline config flag — off by default. IPC path logs
    // total only (async fetches mean per-bucket isn't meaningful at
    // delivery time); non-IPC logs the breakdown.
    if (Configuration.LogDistantPipeline) {
        static int diagFrameCounter = 0;
        if ((diagFrameCounter++ % 60) == 0) {
            const unsigned total = useIpc
                ? (unsigned)visDistantShared.Size()
                : (unsigned)visDistant.Size();
            if (useIpc) {
                LOG::logline("-- DL statics: total=%u (IPC; bucket split not available)", total);
            } else {
                LOG::logline("-- DL statics: total=%u  near=%u  far=%u  veryFar=%u",
                             total, diagNearCount, diagFarCount, diagVeryFarCount);
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

    // Build the per-frame instance VB when instancing is enabled.
    // MOREFPS-INSTANCING: drop this whole block to remove instancing.
    if (Configuration.UseStaticInstancing) {
        if (Configuration.UseSharedMemory) {
            StaticInstancing::buildVB(visDistantShared);
        } else {
            StaticInstancing::buildVB(visDistant);
        }
    }
}

void DistantLand::renderDistantStatics() {
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

// Distant-statics instancing implementations live in
// staticinstancing.cpp (namespace StaticInstancing). Only
// applyMSOCToDistantStatics (below) lives here — it's path-agnostic
// and consumed by both the instanced and non-instanced render paths.

// MSOC verdict pass — populates `msocOccluded` with a per-instance
// cull mask. Both the instanced and non-instanced render paths consume
// this mask, so culling behaves the same regardless of which rendering
// pipeline is active.
//
// Walks the visible set, runs the batched sphere query, optionally
// escalates large statics to the OBB test, then applies far-distance
// gate / engine-handoff gate / temporal hysteresis to produce the
// final per-instance cull decision.
template<class T>
void DistantLand::applyMSOCToDistantStatics(VisibleSet<T>& staticSet) {
    MGE_SCOPED_TIMER("applyMSOCToDistantStatics");

    msocOccluded.clear();
    if (staticSet.Empty()) {
        return;
    }

    // Skip MSOC entirely in interior cells. The horizon curtain
    // produces nothing useful indoors (no terrain horizon), and the
    // ~228 us/frame the verdict pass costs is pure waste. The
    // distant-statics rendering path is still active in interiors
    // (via NoInteriorDL=False) but we don't bother culling it.
    if (!MWBridge::get()->IsExterior()) {
        return;
    }

    const bool cullByOcclusion =
        Configuration.UseOcclusionCulling
        && MSOCClient::isAvailable()
        && MSOCClient::isMaskReady();
    if (!cullByOcclusion) {
        return;   // empty mask = nothing culled
    }

    const unsigned setSize = (unsigned)staticSet.Size();
    msocOccluded.resize(setSize, 0);

    // Constants — same values as the original implementation in
    // buildStaticInstanceVB; kept here as the single source of truth.
    //
    // OBB escalation disabled (threshold bumped to "never"). The OBB
    // test was added to resolve giants whose loose bounding sphere
    // couldn't return OCCLUDED, but for tall-thin statics like the
    // Telvanni mushroom towers, the OBB box may exclude the spire (or
    // have tight Z), so the OBB triangles project entirely within the
    // curtain region while the real geometry extends above the
    // silhouette. Result: tower wrongly reports OCCLUDED while the
    // sphere correctly reports VISIBLE. Sphere-only is more
    // conservative and under-culls in the right direction; the cull
    // rate hit is small now that the horizon curtain gives the sphere
    // test enough mask coverage to resolve real occlusion.
    constexpr float kOBBRadiusThreshold = std::numeric_limits<float>::max();
    constexpr float kHandoffMargin      = 512.0f;
    // User-tunable knobs (mge.ini, [Misc]):
    //   "Occlusion Sphere Inflate"   — radius multiplier for verdict stability
    //   "Occlusion Hysteresis Frames" — consecutive OCCLUDED frames before cull
    const float kSphereInflate          = Configuration.OcclusionSphereInflate;
    const std::uint8_t hysteresisFrames =
        (std::uint8_t)Configuration.OcclusionHysteresisFrames;
    const float farMSOCLimit   = Configuration.DL.FarStaticEnd * kCellSize;
    const float farMSOCLimitSq = farMSOCLimit * farMSOCLimit;

    // Bump per-frame counter once per MSOC pass for hysteresis aging.
    ++g_msocHysteresisFrame;

    // Batched sphere test — pack all spheres into one buffer, issue
    // a single DLL call instead of N. Saves the per-sphere
    // boundary-crossing overhead.
    static std::vector<float> sphereScratch;
    static std::vector<MSOCClient::TestResult> verdictScratch;
    sphereScratch.clear();
    verdictScratch.clear();
    sphereScratch.reserve(setSize * 4);
    verdictScratch.resize(setSize, MSOCClient::ResultVisible);
    {
        MGE_SCOPED_TIMER("applyMSOCToDistantStatics:sphereBatch");
        staticSet.Reset();
        while (!staticSet.AtEnd()) {
            const auto& m = staticSet.Next();
            sphereScratch.push_back(m.sphere.center.x);
            sphereScratch.push_back(m.sphere.center.y);
            sphereScratch.push_back(m.sphere.center.z);
            sphereScratch.push_back(m.sphere.radius * kSphereInflate);
        }
        MSOCClient::classifySphereBatch(
            sphereScratch.data(), (int)setSize, verdictScratch.data());
    }

    int diagTested = 0, diagVisible = 0, diagOccluded = 0;
    int diagViewCulled = 0, diagNotReady = 0;
    int diagOBBCalls = 0, diagOBBOccluded = 0;
    int diagHandoffSpared = 0, diagFarSpared = 0, diagHysteresisSpared = 0;

    staticSet.Reset();
    unsigned idx = 0;
    while (!staticSet.AtEnd()) {
        const auto& m = staticSet.Next();
        ++diagTested;

        MSOCClient::TestResult verdict = verdictScratch[idx];

        // OBB escalation for giants (radius > kOBBRadiusThreshold).
        if (m.sphere.radius > kOBBRadiusThreshold
            && verdict == MSOCClient::ResultVisible)
        {
            ++diagOBBCalls;
            const auto obbVerdict = MSOCClient::classifyOBB(
                m.box.center.x, m.box.center.y, m.box.center.z,
                m.box.vx.x * kSphereInflate, m.box.vx.y * kSphereInflate, m.box.vx.z * kSphereInflate,
                m.box.vy.x * kSphereInflate, m.box.vy.y * kSphereInflate, m.box.vy.z * kSphereInflate,
                m.box.vz.x * kSphereInflate, m.box.vz.y * kSphereInflate, m.box.vz.z * kSphereInflate);
            if (obbVerdict != MSOCClient::ResultNotReady) {
                verdict = obbVerdict;
                if (verdict == MSOCClient::ResultOccluded) ++diagOBBOccluded;
            }
        }

        switch (verdict) {
        case MSOCClient::ResultVisible:    ++diagVisible;    break;
        case MSOCClient::ResultOccluded:   ++diagOccluded;   break;
        case MSOCClient::ResultViewCulled: ++diagViewCulled; break;
        case MSOCClient::ResultNotReady:   ++diagNotReady;   break;
        }

        // Hysteresis state lookup.
        const std::uint64_t hKey = hashStaticKey(
            m.sphere.center.x, m.sphere.center.y, m.sphere.center.z);
        auto& hState = g_msocHysteresis[hKey];
        hState.lastSeenFrame = g_msocHysteresisFrame;

        if (verdict != MSOCClient::ResultOccluded) {
            hState.consecutiveOccluded = 0;
            // msocOccluded[idx] stays 0 (render).
            ++idx;
            continue;
        }

        // verdict == OCCLUDED. Apply gates + hysteresis to decide
        // whether to actually cull.
        const float dx = m.sphere.center.x - eyePos.x;
        const float dy = m.sphere.center.y - eyePos.y;
        const float dz = m.sphere.center.z - eyePos.z;
        const float distSq = dx * dx + dy * dy + dz * dz;

        if (distSq > farMSOCLimitSq) {
            ++diagFarSpared;
            // Far gate spares — render.
        }
        else {
            const float thresh = nearViewRange + m.sphere.radius + kHandoffMargin;
            if (distSq < thresh * thresh) {
                ++diagHandoffSpared;
                // Handoff gate spares — render.
            }
            else if (hState.consecutiveOccluded < hysteresisFrames) {
                ++hState.consecutiveOccluded;
                ++diagHysteresisSpared;
                // Hysteresis still in warmup — render.
            }
            else {
                // Sustained OCCLUDED past all gates — actually cull.
                msocOccluded[idx] = 1;
            }
        }
        ++idx;
    }

    // Hysteresis age-prune. Drop entries not touched in the last
    // kHysteresisAgeOutFrames frames. Run once per 60 frames so per-
    // frame cost stays trivial.
    if ((g_msocHysteresisFrame % 60) == 0) {
        for (auto it = g_msocHysteresis.begin(); it != g_msocHysteresis.end();) {
            if (g_msocHysteresisFrame - it->second.lastSeenFrame > kHysteresisAgeOutFrames) {
                it = g_msocHysteresis.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Numpad-5 mask dump (debug feature, gated by LogDistantPipeline).
    // Latched once-per-frame and only services on a frame where the
    // mask is ready, so a tap that lands during a non-cull frame still
    // dumps on the next eligible one.
    if (Configuration.LogDistantPipeline) {
        static bool dumpPending = false;
        static int  dumpIndex   = 0;
        if (GetAsyncKeyState(VK_NUMPAD5) & 0x0001) {
            dumpPending = true;
        }
        if (dumpPending) {
            char exePath[MAX_PATH] = {};
            GetModuleFileNameA(NULL, exePath, MAX_PATH);
            if (char* lastSlash = strrchr(exePath, '\\')) {
                *lastSlash = '\0';
            }
            char fullPath[MAX_PATH];
            std::snprintf(fullPath, sizeof(fullPath), "%s\\msoc_mask_%03d.pfm",
                          exePath, dumpIndex++);
            const bool ok = MSOCClient::dumpMask(fullPath);
            LOG::logline("-- MSOC: Numpad5 mask dump %s %s",
                         ok ? "->" : "FAILED for", fullPath);
            dumpPending = false;
        }
    }

    static int diagFrameCounter = 0;
    if (Configuration.LogDistantPipeline && (diagFrameCounter++ % 60) == 0) {
        const int rateNum = diagTested > 0 ? (diagOccluded * 100) / diagTested : 0;
        LOG::logline(
            "-- MSOC cull: tested=%d  occluded=%d (%d%%)  visible=%d  viewCulled=%d  notReady=%d  obb=%d/%d  handoffSpared=%d  farSpared=%d  hystSpared=%d  hystSize=%u",
            diagTested, diagOccluded, rateNum,
            diagVisible, diagViewCulled, diagNotReady,
            diagOBBOccluded, diagOBBCalls, diagHandoffSpared, diagFarSpared,
            diagHysteresisSpared, (unsigned)g_msocHysteresis.size());
    }
}

template void DistantLand::applyMSOCToDistantStatics(VisibleSet<StlVector>& staticSet);
template void DistantLand::applyMSOCToDistantStatics(VisibleSet<IpcClientVector>& staticSet);

// (Instanced render path moved to StaticInstancing::renderColor /
// renderDepth in staticinstancing.cpp.)
