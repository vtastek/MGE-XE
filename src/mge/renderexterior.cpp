
#include "distantland.h"
#include "distantshader.h"
#include "configuration.h"
#include "msocclient.h"
#include "mwbridge.h"
#include "phasetimers.h"
#include "proxydx/d3d8header.h"
#include "support/log.h"

#include <algorithm>
#include <vector>



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

    // MOREFPS diagnostic: land-tile count after frustum culling.
    // Logged AFTER Render, because in IPC mode getVisibleMeshes is
    // async and the size isn't unpacked until the parallel-read pass
    // inside Render. Sampling before that returns 0 regardless of
    // how many tiles are actually visible.
    {
        static int diagFrameCounter = 0;
        if ((diagFrameCounter++ % 60) == 0) {
            const unsigned tiles = Configuration.UseSharedMemory
                ? (unsigned)visLandShared.Size()
                : (unsigned)visLand.Size();
            LOG::logline("-- DL land: tiles=%u", tiles);
        }
    }

    // MOREFPS phase 7: feed tile OBBs into MSOC's mask for NEXT frame.
    // Runs after Render to guarantee (in IPC mode) that visLandShared's
    // async fetch has completed. Submissions queued here become part of
    // frame N+1's mask, matching the existing double-buffer snapshot
    // one-frame-latency contract.
    contributeDistantLandOccluders();
}

// Phase D (replaces the subsample-based Phase C): submit each visible
// tile's NATIVE triangle mesh verbatim into the MSOC mask. Phase C's
// 9x9 subsample produced either a "canopy" over-cull or a banded flat
// mask that missed real terrain silhouettes. Root cause: MGE-XE's
// ROAM-tessellated distant-land is adaptive and cache-reordered, so
// any regular sampling pattern fights the data. Submitting the real
// triangles eliminates that entire failure mode — the mask gets the
// actual terrain surface, exactly where the tessellator put detail.
//
// Budget: plugin caps external tris at OcclusionOccluderMaxTriangles
// (default 4096). Closest-first greedy fill: take each nearest tile's
// whole mesh until the running total would exceed the cap, then stop.
// A typical Morrowind draw distance puts 1-9 tiles (32768x32768 each)
// in view, so budget usually covers most or all of them.
void DistantLand::contributeDistantLandOccluders() {
    MGE_SCOPED_TIMER("contributeDistantLandOccluders");

    // Rate-limited diagnostic: log which early-return guard fired so
    // we can tell why no occluder submission is happening. Sampled
    // every 60th invocation to stay quiet on the steady state.
    static int diagGuardFrame = 0;
    const bool emitDiag = (diagGuardFrame++ % 60) == 0;

    if (!Configuration.UseOcclusionCulling) {
        if (emitDiag) LOG::logline("-- MSOC occluder: land skipped — UseOcclusionCulling=false");
        return;
    }
    if (!MSOCClient::isAvailable()) {
        if (emitDiag) LOG::logline("-- MSOC occluder: land skipped — MSOCClient not available");
        return;
    }
    if (landMeshes.empty()) {
        if (emitDiag) LOG::logline("-- MSOC occluder: land skipped — landMeshes map is empty");
        return;
    }

    // Small safety margin under the plugin cap so other consumers (if
    // any arrive) still have room to submit. 3800 / 4096 ≈ 93%.
    constexpr int kTriBudget = 3800;

    // Gather each visible tile + its planar distance, so we can pick
    // the closest-N under the tri budget. Tiles without a captured
    // mesh (shouldn't happen for land tiles, but defensive) fall
    // through — one fewer contribution, no correctness risk.
    struct Candidate {
        const LandMeshCache* mesh;
        float                distSq;
    };
    static std::vector<Candidate> candidates;
    candidates.clear();

    int visibleLandTiles = 0;
    int lookupMisses = 0;
    auto collectWithDiag = [&](const RenderMesh& m) {
        ++visibleLandTiles;
        auto it = landMeshes.find(m.vBuffer);
        if (it == landMeshes.end()) {
            ++lookupMisses;
            return;
        }
        const float dx = m.sphere.center.x - eyePos.x;
        const float dy = m.sphere.center.y - eyePos.y;
        candidates.push_back({ &it->second, dx * dx + dy * dy });
    };

    if (Configuration.UseSharedMemory) {
        visLandShared.Reset();
        while (!visLandShared.AtEnd()) collectWithDiag(visLandShared.Next());
    } else {
        visLand.Reset();
        while (!visLand.AtEnd()) collectWithDiag(visLand.Next());
    }

    if (candidates.empty()) {
        if (emitDiag) {
            LOG::logline(
                "-- MSOC occluder: land skipped — no candidates (visible=%d lookupMisses=%d mapSize=%u)",
                visibleLandTiles, lookupMisses, (unsigned)landMeshes.size());
        }
        return;
    }

    // Closest-first sort — nearest silhouettes matter most for
    // occluding statics at mid-distance; far tiles are discretionary.
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.distSq < b.distSq; });

    // Build one combined vertex + index buffer across selected tiles.
    // Static scratch to avoid reallocation each frame.
    static std::vector<float>    vertScratch;   // (x, y, z, w) stride 16
    static std::vector<uint32_t> triScratch;    // 3 indices per triangle
    vertScratch.clear();
    triScratch.clear();

    int totalTris  = 0;
    int tilesTaken = 0;
    for (const auto& cand : candidates) {
        const LandMeshCache& mesh = *cand.mesh;
        const int tileTris = (int)(mesh.indices.size() / 3);
        if (tileTris <= 0) continue;

        // Greedy cap: once a tile would push us past budget, stop.
        // Far tiles beyond that point contribute nothing this frame,
        // which is safe — missing horizon coverage is under-cull, not
        // over-cull.
        if (totalTris + tileTris > kTriBudget) break;

        const uint32_t baseVtx = (uint32_t)(vertScratch.size() / 4);

        // Append positions padded to (x, y, z, w=1) matching the
        // stride=16 / offY=4 / offW=12 layout we pass to addOccluder.
        for (const auto& p : mesh.positions) {
            vertScratch.push_back(p.x);
            vertScratch.push_back(p.y);
            vertScratch.push_back(p.z);
            vertScratch.push_back(1.0f);
        }

        // Rebase and copy the index list into the combined buffer.
        // baseVtx is the offset of this tile's first vertex in the
        // combined vertex array.
        for (uint32_t idx : mesh.indices) {
            triScratch.push_back(baseVtx + idx);
        }

        totalTris  += tileTris;
        tilesTaken += 1;
    }

    if (totalTris == 0) return;

    const int vtxCount = (int)(vertScratch.size() / 4);
    const bool ok = MSOCClient::addOccluder(
        vertScratch.data(), vtxCount,
        /*stride*/ 16, /*offY*/ 4, /*offW*/ 12,
        triScratch.data(), totalTris,
        /*modelMatrix*/ nullptr);   // positions are world-space already

    static int diagFrameCounter = 0;
    if ((diagFrameCounter++ % 60) == 0) {
        LOG::logline("-- MSOC occluder: land %s — tiles=%d/%u tris=%d verts=%d",
                     ok ? "submitted" : "REJECTED",
                     tilesTaken, (unsigned)candidates.size(),
                     totalTris, vtxCount);
    }
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

    // MOREFPS diagnostics: per-bucket counts only available in the
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

    // Per-frame cull summary. IPC path logs total only (async fetches
    // mean per-bucket isn't meaningful at delivery time); non-IPC logs
    // the breakdown.
    {
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
            // MOREFPS temporary phase-timer flush. Piggy-backs on the
            // 60-frame diagnostic cadence already in use above. Dumps
            // each scope's accumulated µs + call count as `-- PHASE: ...`
            // lines in mgeXE.log, then resets for the next sampling
            // window. Remove these two lines (and the MGE_SCOPED_TIMER
            // call sites) to rip the instrumentation out entirely.
            MGEPhaseTimers::report();
        }
    }

    // MOREFPS phase 4: build the per-frame instance VB when instancing is enabled.
    // Must run after the sort so groups form over consecutive same-state meshes.
    if (Configuration.UseStaticInstancing) {
        if (Configuration.UseSharedMemory) {
            buildStaticInstanceVB(visDistantShared);
        } else {
            buildStaticInstanceVB(visDistant);
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

    if (Configuration.UseSharedMemory) {
        visDistantShared.Render(device, effect, effect, &ehTex0, nullptr, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT);
    } else {
        visDistant.Render(device, effect, effect, &ehTex0, nullptr, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT);
    }

    device->SetRenderState(D3DRS_CLIPPLANEENABLE, 0);
}

// MOREFPS phase 2: distant-statics instancing helpers.
//
// buildStaticInstanceVB walks the sorted visible set, groups consecutive
// meshes sharing (vBuffer, tex, hasAlpha, animateUV), and packs each
// group's transforms as transposed 4x3 matrices into vbStaticInstances.
// batchedStatics is populated with (firstMesh, instanceCount) pairs.
//
// Mirrors buildGrassInstanceVB (rendergrass.cpp). Difference: grass
// groups only on vBuffer since grass shares texture/pipeline state;
// statics require a wider group key because each static can have its
// own texture/alpha/uv-anim state.
//
// NOTE: the sort key in QuadTreeMesh::CompareByState is currently
// (tex, vBuffer). For optimal grouping, phase 4 will extend it to
// (tex, vBuffer, hasAlpha, animateUV). Until then, this function
// will still produce correct output, just with smaller batches when
// hasAlpha/animateUV vary within a (tex, vBuffer) cluster.
template<class T>
void DistantLand::buildStaticInstanceVB(VisibleSet<T>& staticSet) {
    MGE_SCOPED_TIMER("buildStaticInstanceVB");
    batchedStatics.clear();

    if (staticSet.Empty()) {
        return;
    }

    if (staticSet.Size() > MaxStaticInstances) {
        static bool warnOnce = true;
        if (warnOnce) {
            LOG::logline("!! Too many distant-static instances. (%d elements, limit %d)", staticSet.Size(), MaxStaticInstances);
            LOG::logline("!! Some distant statics will not render. Reduce draw distance or lower static density.");
            warnOnce = false;
        }
        staticSet.Truncate(MaxStaticInstances);
    }

    // Snapshot the group head BY VALUE. We can't hold a pointer into the
    // source visible-set: with shared-memory IPC, visDistantShared iterates
    // through a remapping window (see IpcClientVector::next()), so any
    // reference into the source is invalidated on the next advance.
    RenderMesh groupHead{};
    bool haveGroupHead = false;
    float* vbwrite = nullptr;
    int instanceCount = 0;

    HRESULT hr = vbStaticInstances->Lock(0, StaticInstStride * staticSet.Size(), (void**)&vbwrite, D3DLOCK_DISCARD);
    if (hr != D3D_OK || vbwrite == 0) {
        return;
    }

    // MOREFPS: decide ONCE whether we'll consult the msoc.dll mask this
    // frame. The readiness flag flips false at the start of every frame
    // (before the plugin re-renders occluders), so a stale-read here is
    // safe either way: either we skip the test, or we cache "ready" and
    // every test returns kNotReady → visible.
    //
    // Conservative radius: RenderMesh doesn't carry bounding-sphere data
    // through the IPC wire format (only QuadTreeMesh does, which is on the
    // server side). 1024 is ~half a cell and comfortably larger than
    // typical static bounds — it under-culls rather than over-culls, so
    // false positives (incorrectly hiding a visible mesh) can't happen.
    // TODO: plumb QuadTreeMesh::sphere.radius through RenderMesh for a
    // tighter test and higher cull rate.
    const bool cullByOcclusion =
        Configuration.UseOcclusionCulling
        && MSOCClient::isAvailable()
        && MSOCClient::isMaskReady();

    // MOREFPS debug: Numpad 5 dumps the MSOC mask to a PFM in the install
    // root. Numpad 5 is picked because nothing in Morrowind or MGE-XE
    // consumes it — F10 (previous binding) doubled as Win32's menu-
    // focus key and was flaky.
    //
    // Detection uses GetAsyncKeyState's 0x0001 bit — the "has been
    // pressed since the last call" transient — instead of the 0x8000
    // "is down right now" bit. The 0x0001 latch survives between polls,
    // so a tap that happens during a gap in polling (interior cell,
    // menu open, cull pass skipped) still registers on the next call.
    // The 0x8000 approach we used before could silently miss presses.
    //
    // Latches a "pending" flag and services it only on a frame where
    // the mask is actually ready — the plugin flips g_maskReady true
    // mid-frame, so dispatching immediately could fail "mask not ready".
    // One press → one dump regardless of which frame the key registered.
    if (MSOCClient::isAvailable()) {
        static bool dumpPending = false;
        static int  dumpIndex = 0;
        if (GetAsyncKeyState(VK_NUMPAD5) & 0x0001) {
            dumpPending = true;
        }

        if (dumpPending && cullByOcclusion) {
            char exePath[MAX_PATH] = {};
            GetModuleFileNameA(NULL, exePath, MAX_PATH);
            if (char* lastSlash = strrchr(exePath, '\\')) {
                *lastSlash = '\0';
            }
            char fullPath[MAX_PATH];
            snprintf(fullPath, sizeof(fullPath), "%s\\msoc_mask_%03d.pfm",
                     exePath, dumpIndex++);
            const bool ok = MSOCClient::dumpMask(fullPath);
            LOG::logline("-- MSOC: Numpad5 mask dump %s %s",
                         ok ? "->" : "FAILED for", fullPath);
            dumpPending = false;
        }
    }

    // Radius above which the sphere test is too loose to resolve big
    // architectural statics (Vivec Ministry, Telvanni towers, etc.) —
    // escalate to the 12-triangle OBB test for these. Below threshold
    // the sphere path is fast (~100ns) and accurate enough. Tuned
    // conservatively — raise if OBB call cost starts dominating.
    constexpr float kOBBRadiusThreshold = 2048.0f;

    // MOREFPS diagnostics: bucket plugin verdicts so we can see why the
    // cull rate is low. Sphere and OBB paths are tracked separately so
    // we can judge whether the giants escalation is actually producing
    // culls (a zero obbOccluded with nonzero obbCalls signals a problem
    // in the OBB code path or the box data feeding it).
    int diagTested = 0, diagVisible = 0, diagOccluded = 0;
    int diagViewCulled = 0, diagNotReady = 0;
    int diagOBBCalls = 0, diagOBBOccluded = 0;

    // Batch sphere test (Phase A). Pre-walk the set once to pack the
    // (x, y, z, r) tuples into a scratch buffer, call the plugin once,
    // then consult the results array during the main pack loop. Saves
    // ~2000 DLL-boundary crossings per cull pass. When batch export is
    // unavailable or occlusion is off, the scratch buffers stay empty
    // and the pack loop falls back to per-instance classifySphere calls.
    //
    // The scratch vectors are function-static to avoid per-frame heap
    // allocation — capacity grows to peak set size once and stays put.
    static std::vector<float> sphereScratch;
    static std::vector<MSOCClient::TestResult> verdictScratch;
    sphereScratch.clear();
    verdictScratch.clear();
    const unsigned sizeAfterTruncate = (unsigned)staticSet.Size();

    if (cullByOcclusion) {
        MGE_SCOPED_TIMER("buildStaticInstanceVB:sphereBatch");
        sphereScratch.reserve(sizeAfterTruncate * 4);
        verdictScratch.resize(sizeAfterTruncate, MSOCClient::ResultVisible);
        staticSet.Reset();
        while (!staticSet.AtEnd()) {
            const auto& m = staticSet.Next();
            sphereScratch.push_back(m.sphere.center.x);
            sphereScratch.push_back(m.sphere.center.y);
            sphereScratch.push_back(m.sphere.center.z);
            sphereScratch.push_back(m.sphere.radius);
        }
        MSOCClient::classifySphereBatch(
            sphereScratch.data(),
            (int)sizeAfterTruncate,
            verdictScratch.data());
    }

    staticSet.Reset();
    unsigned idx = 0;
    while (!staticSet.AtEnd()) {
        const auto& m = staticSet.Next();

        // Per-instance occlusion decision. Sphere verdict (from the batch
        // pass above) is the starting point. Giants with radius above
        // kOBBRadiusThreshold escalate to the tighter OBB test when
        // their sphere said Visible — OBB's tighter silhouette may
        // resolve them as Occluded even when the loose sphere couldn't.
        // Occluded / VIEW_CULLED from sphere are kept as-is: already a
        // cull-equivalent outcome, no need for the more expensive test.
        if (cullByOcclusion) {
            ++diagTested;
            MSOCClient::TestResult verdict = verdictScratch[idx];

            if (m.sphere.radius > kOBBRadiusThreshold
                && verdict == MSOCClient::ResultVisible)
            {
                ++diagOBBCalls;
                const auto obbVerdict = MSOCClient::classifyOBB(
                    m.box.center.x, m.box.center.y, m.box.center.z,
                    m.box.vx.x, m.box.vx.y, m.box.vx.z,
                    m.box.vy.x, m.box.vy.y, m.box.vy.z,
                    m.box.vz.x, m.box.vz.y, m.box.vz.z);
                // Only upgrade to OBB's verdict if the plugin actually
                // answered — ResultNotReady means OBB export is absent
                // in an older plugin, in which case the sphere verdict
                // stands.
                if (obbVerdict != MSOCClient::ResultNotReady) {
                    verdict = obbVerdict;
                    if (verdict == MSOCClient::ResultOccluded) {
                        ++diagOBBOccluded;
                    }
                }
            }

            switch (verdict) {
            case MSOCClient::ResultVisible:    ++diagVisible;    break;
            case MSOCClient::ResultOccluded:   ++diagOccluded;   break;
            case MSOCClient::ResultViewCulled: ++diagViewCulled; break;
            case MSOCClient::ResultNotReady:   ++diagNotReady;   break;
            }
            if (verdict == MSOCClient::ResultOccluded) {
                ++idx;
                continue;
            }
        }

        if (!haveGroupHead) {
            groupHead = m;
            haveGroupHead = true;
        } else if (groupHead.vBuffer != m.vBuffer
                   || groupHead.tex != m.tex
                   || groupHead.hasAlpha != m.hasAlpha
                   || groupHead.animateUV != m.animateUV) {
            batchedStatics.push_back(std::make_pair(groupHead, instanceCount));
            groupHead = m;
            instanceCount = 0;
        }

        // Pack per-instance transform as a transposed 4x3 matrix (12 floats).
        // Layout matches vbGrassInstances, so GrassDecl's stream 1 elements
        // (TEXCOORD1/2/3, three FLOAT4 rows) can consume it directly.
        const D3DMATRIX* world = &m.transform;
        vbwrite[0]  = world->_11; vbwrite[1]  = world->_21; vbwrite[2]  = world->_31; vbwrite[3]  = world->_41;
        vbwrite[4]  = world->_12; vbwrite[5]  = world->_22; vbwrite[6]  = world->_32; vbwrite[7]  = world->_42;
        vbwrite[8]  = world->_13; vbwrite[9]  = world->_23; vbwrite[10] = world->_33; vbwrite[11] = world->_43;
        vbwrite += 12;
        instanceCount++;
        ++idx;
    }
    if (haveGroupHead) {
        batchedStatics.push_back(std::make_pair(groupHead, instanceCount));
    }

    vbStaticInstances->Unlock();

    // MOREFPS: log the per-frame cull breakdown. Rate-limited to one
    // line every 60 cull passes (≈ once per second at 60fps) so it's
    // non-spammy. Remove once the cull rate is understood.
    if (cullByOcclusion) {
        static int diagFrameCounter = 0;
        if ((diagFrameCounter++ % 60) == 0) {
            const int rateNum = diagTested > 0 ? (diagOccluded * 100) / diagTested : 0;
            // Batch-size stats over the instanced draws we're about to
            // issue (min/max/avg instance count per DrawIndexedPrimitive
            // call). High avg = efficient instancing; many small batches
            // = sort/group isn't coalescing as well as it could.
            int batchMin = 0, batchMax = 0, batchSum = 0;
            if (!batchedStatics.empty()) {
                batchMin = batchedStatics[0].second;
                batchMax = batchedStatics[0].second;
                for (const auto& b : batchedStatics) {
                    if (b.second < batchMin) batchMin = b.second;
                    if (b.second > batchMax) batchMax = b.second;
                    batchSum += b.second;
                }
            }
            const int batchAvg = !batchedStatics.empty()
                ? batchSum / (int)batchedStatics.size() : 0;
            LOG::logline(
                "-- MSOC cull: tested=%d  occluded=%d (%d%%)  visible=%d  viewCulled=%d  notReady=%d  obb=%d/%d",
                diagTested, diagOccluded, rateNum,
                diagVisible, diagViewCulled, diagNotReady,
                diagOBBOccluded, diagOBBCalls);
            LOG::logline(
                "-- DL batches: count=%u  instances=%d  min=%d  avg=%d  max=%d",
                (unsigned)batchedStatics.size(),
                batchSum, batchMin, batchAvg, batchMax);
        }
    }
}

// Explicit instantiations for the two VisibleSet container variants (StlVector / IpcClientVector).
// Mirrors buildGrassInstanceVB's implicit instantiation via direct callers in rendergrass.cpp.
// Not strictly required yet (no callers), but keeps phase 4 wire-up from producing surprise
// linker errors and lets the linker strip this code as dead until that wire-up lands.
template void DistantLand::buildStaticInstanceVB(VisibleSet<StlVector>& staticSet);
template void DistantLand::buildStaticInstanceVB(VisibleSet<IpcClientVector>& staticSet);

// Common core of the two instanced-statics entry points.
// Iterates batchedStatics, rebinds per-group state (texture, alpha test,
// animate-uv), and issues one instanced DrawIndexedPrimitive per group.
//
// Effect-variable writes go through `effect` (the main effect that owns
// the shared effect-pool parameters ehTex0 / ehHasVCol). CommitChanges
// is called on `e` so the currently-running pass (which may live in
// `effect` OR `effectDepth`) picks them up. Mirrors the pattern used in
// renderGrassCommon.
void DistantLand::renderDistantStaticsInstancedCommon(ID3DXEffect* e) {
    MGE_SCOPED_TIMER("renderDistantStaticsInstancedCommon");
    device->SetVertexDeclaration(GrassDecl);

    // Reset animate_uv so the first group picks up a fresh state.
    effect->SetBool(ehHasVCol, false);
    e->CommitChanges();

    IDirect3DTexture9* lastTexture = nullptr;
    bool lastAnimateUV = false;
    bool lastHasAlpha = false;
    int firstInstance = 0;

    for (const auto& batch : batchedStatics) {
        const RenderMesh& mesh = batch.first;
        int count = batch.second;

        // Per-group state setup (rebind-filtered by the sort + group boundary rules).
        if (lastTexture != mesh.tex) {
            effect->SetTexture(ehTex0, mesh.tex);
            device->SetRenderState(D3DRS_ALPHATESTENABLE, mesh.hasAlpha);
            lastTexture = mesh.tex;
            lastHasAlpha = mesh.hasAlpha;
        } else if (lastHasAlpha != mesh.hasAlpha) {
            device->SetRenderState(D3DRS_ALPHATESTENABLE, mesh.hasAlpha);
            lastHasAlpha = mesh.hasAlpha;
        }

        if (lastAnimateUV != mesh.animateUV) {
            effect->SetBool(ehHasVCol, mesh.animateUV);
            lastAnimateUV = mesh.animateUV;
        }

        e->CommitChanges();

        // Bind per-group static VB + shared instance VB and issue one instanced draw.
        device->SetIndices(mesh.iBuffer);
        device->SetStreamSourceFreq(0, D3DSTREAMSOURCE_INDEXEDDATA | count);
        device->SetStreamSource(0, mesh.vBuffer, 0, SIZEOFSTATICVERT);
        device->SetStreamSourceFreq(1, D3DSTREAMSOURCE_INSTANCEDATA | 1);
        device->SetStreamSource(1, vbStaticInstances, StaticInstStride * firstInstance, StaticInstStride);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, mesh.verts, 0, mesh.faces);

        firstInstance += count;
    }

    // Restore non-instanced state so later unrelated draws don't inherit the instance freq.
    device->SetStreamSourceFreq(0, 1);
    device->SetStreamSourceFreq(1, 1);
    device->SetStreamSource(1, NULL, 0, 0);
}

// Render the sets built by buildStaticInstanceVB as instanced draws in the
// main color pass. Caller is expected to have invoked BeginPass with either
// PASS_RENDERSTATICSEXTERIOR_INST or PASS_RENDERSTATICSINTERIOR_INST.
// Uses GrassDecl (the layout static-stream + per-instance-transform-stream
// is identical to grass's).
void DistantLand::renderDistantStaticsInstanced() {
    MGE_SCOPED_TIMER("renderDistantStaticsInstanced");
    if (batchedStatics.empty()) {
        return;
    }

    if (!MWBridge::get()->IsExterior()) {
        // Same architectural clip-plane workaround as renderDistantStatics.
        float clipAt = nearViewRange - 768.0f;
        D3DXPLANE clipPlane(0, 0, clipAt, -(mwProj._33 * clipAt + mwProj._43));
        device->SetClipPlane(0, clipPlane);
        device->SetRenderState(D3DRS_CLIPPLANEENABLE, 1);
    }

    renderDistantStaticsInstancedCommon(effect);

    device->SetRenderState(D3DRS_CLIPPLANEENABLE, 0);
}

// Render the sets built by buildStaticInstanceVB into the depth pre-pass.
// Caller is expected to have invoked effectDepth->BeginPass with
// PASS_RENDERSTATICSDEPTH_INST. No clip-plane handling here — the
// existing non-instanced depth path doesn't do that either.
void DistantLand::renderDistantStaticsInstancedZ() {
    MGE_SCOPED_TIMER("renderDistantStaticsInstancedZ");
    if (batchedStatics.empty()) {
        return;
    }

    renderDistantStaticsInstancedCommon(effectDepth);
}
