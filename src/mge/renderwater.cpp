
#include "distantland.h"
#include "distantshader.h"
#include "drawstats.h"
#include "configuration.h"
#include "doublesurface.h"
#include "ffeshader.h"
#include "mwbridge.h"
#include "postshaders.h"
#include "scenegraph_geometry_cache.h"
#include "support/log.h"
#include "mge_tracy.h"

#include <algorithm>
#include <cstdint>
#include <vector>



// Tweakable: the cache-driven reflection covers near objects within this fraction
// of Morrowind's view distance (nearViewRange). Beyond it, distant-land statics
// take over (dissolved in via ehNearViewRange). 1.0 = full MW view distance; lower
// (down to ~0.5) shrinks the lit near-field. TODO: promote to a config/Numpad knob.
static float s_reflectionCacheNearFactor = 1.0f;

// Downward bias (world units) for the cache reflection below-water clip. The clip at
// the exact waterline cuts a thin gap at terrain/water intersections (background
// shows through) and slices straddling characters open at the surface (hollow
// interior visible). Lowering the clip a few units keeps a sliver of near-waterline
// geometry: it fills the terrain gap and sinks the character cut below the surface,
// where the water shading (fresnel/refraction/fog) masks it. Larger = less
// aggressive (more below-water reflection); 0 = clip exactly at the true level.
// TODO: promote to a config/Numpad knob.
static float s_reflWaterClipBias = 3.0f;

// Water-surface snap offsets (world units) applied to the rendered water mesh height,
// chosen by the engine underwater state. A single fixed height can't satisfy both
// sides: above water the wave crests poke up and intersect the near-surface camera,
// and a low surface renders below the camera once submerged. So push the mesh DOWN
// when above water (crests stay below the eye) and keep it at/above the level when
// underwater (surface stays overhead). The snap happens exactly at the IsUnderwater
// threshold, where the camera is at the surface and the jump is hidden. Fine-tune to
// the wave height. TODO: promote to config/Numpad knobs.
static float s_waterMeshSnapAbove      = -5.0f;   // above water: push down
static float s_waterMeshSnapUnderwater =  5.0f;   // underwater: at WaterLevel (default)

// NOTE: the cache-driven color machinery (the cacheWorldBounds /
// cacheSkinnedWorldBounds / cacheHandoverFade helpers, the synthetic-state
// builders, and renderReflectionsFromCache / renderReflectionShadowsFromCache /
// renderReflectionTerrainFromCache / renderCachedOpaque) lives in
// rendercachedcolor.cpp. renderWaterReflection below just calls those methods.

// Diagnostic (gated on LogDistantPipeline): bucket the reflected distant-statics
// survivors by object-origin distance vs cacheNearDist. The near band is collapsed
// to nothing by the staticNearCull vertex-shader clip (XE Mod Statics.fx) so the
// cache can draw those objects lit — but the draw call is still ISSUED here, so the
// near count is wasted draw calls that mirror the cache's lit near-statics. The far
// band is what actually rasterizes. Confirms the dl.stat/cache double-issue.
static void logReflStaticNearFar(VisibleSet<StlVector>& vs, const D3DXVECTOR3& eye, float cacheNearDist) {
    if (!Configuration.LogDistantPipeline || cacheNearDist <= 0.0f) return;
    static unsigned s_near = 0, s_far = 0, s_calls = 0;
    for (const RenderMesh* m : vs.visible_set) {
        const D3DXVECTOR3 d = m->sphere.center - eye;
        if (D3DXVec3Length(&d) < cacheNearDist) ++s_near; else ++s_far;
    }
    constexpr unsigned kIv = 300;
    if ((++s_calls % kIv) == 0) {
        LOG::logline("-- [REFL DL.STAT] per-frame: near(<%.0f, shader-clipped waste)=%u far(rasterized)=%u total=%u",
                     cacheNearDist, s_near / kIv, s_far / kIv, (s_near + s_far) / kIv);
        s_near = s_far = 0;
    }

    // [REFL PIPE] On a significant survivor swing, dump the full reflection pipeline so
    // we can see WHERE it diverges: water-tile stages (tested→present→occluded→rects)
    // feed the water-rect cull (queried→survivors). If 'rects' moves with 'surv', it's
    // step-3 water-tile occlusion (snapshot) flipping; if 'rects' is steady but 'surv'
    // jumps, it's the step-4 reflection cull; if 'queried' jumps, the frustum query.
    static int s_prevSurv = -1;
    int tiles, water, occl, rects, queried, surv; bool msoc;
    DistantLand::getReflPipeDiag(tiles, water, occl, rects, queried, surv, msoc);

    // [REFL PIPE FLIP] Farness/tiltedness of the tiles that flipped this frame — the
    // test for "really far and really tilted". dist in cells, elev = grazing angle.
    int flips; float fDistMin, fDistMax, fElevMin, fElevMax;
    DistantLand::getReflPipeFlip(flips, fDistMin, fDistMax, fElevMin, fElevMax);

    const int survDelta = surv > s_prevSurv ? surv - s_prevSurv : s_prevSurv - surv;
    if (s_prevSurv >= 0 && survDelta > 40) {
        LOG::logline("-- [REFL PIPE] JUMP surv %d->%d | waterTiles tested=%d present=%d occluded=%d rects=%d msoc=%d | statics queried=%d surv=%d | flips=%d dist=%.1f..%.1f cells elev=%.1f..%.1f deg",
                     s_prevSurv, surv, tiles, water, occl, rects, msoc ? 1 : 0, queried, surv,
                     flips, fDistMin, fDistMax, fElevMin, fElevMax);
    } else if (flips > 0) {
        // Capture flip geometry even without a big survivor swing, rate-limited.
        static int s_flipThrottle = 0;
        if ((s_flipThrottle++ % 30) == 0)
            LOG::logline("-- [REFL PIPE FLIP] flips=%d dist=%.1f..%.1f cells elev=%.1f..%.1f deg | rects=%d surv=%d",
                         flips, fDistMin, fDistMax, fElevMin, fElevMax, rects, surv);
    }
    s_prevSurv = surv;
}

void DistantLand::renderWaterReflection(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    MGE_ZoneScopedN("renderWaterReflection");
    DrawStats::ScopedStage _ds(DrawStats::Reflection);
    auto mwBridge = MWBridge::get();

    // Switch to render target
    RenderTargetSwitcher rtsw(texReflection, surfReflectionZ);
    device->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, horizonCol, 1.0, 0);

    // Calculate reflected view matrix, mirror plane at water mesh level
    D3DXMATRIX reflView;
    D3DXPLANE plane(0, 0, 1.0f, -(mwBridge->WaterLevel() - 1.0f));
    D3DXMatrixReflect(&reflView, &plane);
    D3DXMatrixMultiply(&reflView, &reflView, view);
    effect->SetMatrix(ehView, &reflView);

    // Calculate new projection
    D3DXMATRIX reflProj = *proj;
    editProjectionZ(&reflProj, 4.0, Configuration.DL.DrawDist * kCellSize);
    effect->SetMatrix(ehProj, &reflProj);

    // Clipping setup
    D3DXMATRIX clipMat;

    // Clip geometry on opposite side of water plane
    plane *= mwBridge->IsUnderwater(eyePos.z) ? -1.0f : 1.0f;

    // If using dynamic ripples, the water level can be lowered by up to 0.5 * waveheight
    // so move clip plane downwards at the cost of some reflection errors
    if (Configuration.MGEFlags & DYNAMIC_RIPPLES) {
        plane.d += 0.5f * Configuration.DL.WaterWaveHeight;
    }

    // Doing inverses separately is a lot more numerically stable
    D3DXMatrixInverse(&clipMat, 0, &reflView);
    D3DXMatrixTranspose(&clipMat, &clipMat);
    D3DXPlaneTransform(&plane, &plane, &clipMat);

    D3DXMatrixInverse(&clipMat, 0, &reflProj);
    D3DXMatrixTranspose(&clipMat, &clipMat);
    D3DXPlaneTransform(&plane, &plane, &clipMat);

    if (visDistant.Empty()) {
        // Workaround for a Direct3D bug with clipping planes, where SetClipPlane
        // has no effect on the shader pipeline if the last rendered draw call was using
        // the fixed function pipeline. This is usually covered by distant statics, but
        // not in compact interiors where all distant statics may be culled.
        // Provoking a DrawPrimitive with shader here makes the following SetClipPlane work.
        effect->BeginPass(PASS_WORKAROUND);
        device->SetVertexDeclaration(WaterDecl);
        device->SetStreamSource(0, vbFullFrame, 0, 12);
        DrawStats::count(2);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
        effect->EndPass();
    }

    device->SetClipPlane(0, plane);
    device->SetRenderState(D3DRS_CLIPPLANEENABLE, 1);

    // Per-pixel below-water clip plane for the cache reflection passes, at the TRUE
    // water level. The device clip plane above is lowered by 0.5*waveHeight for
    // dynamic ripples, which lets a band of below-water geometry bleed into the now
    // detailed cache reflection. Same world plane, transformed only into reflected-
    // view space (the first half of the device-plane transform above, minus the
    // projection step) so the cache pixel shaders clip per-fragment against viewpos.
    {
        D3DXPLANE wpln(0, 0, 1.0f, -(mwBridge->WaterLevel() - 1.0f));
        wpln *= mwBridge->IsUnderwater(eyePos.z) ? -1.0f : 1.0f;
        // Lower the clip by s_reflWaterClipBias on the kept side (extends the kept
        // half-space). Increasing d keeps geometry farther past the boundary; the
        // sign-flip above already orients the normal toward the kept side, so this
        // works for both the above-water and underwater reflection cases.
        wpln.d += s_reflWaterClipBias;
        D3DXMATRIX itRefl;
        D3DXMatrixInverse(&itRefl, 0, &reflView);
        D3DXMatrixTranspose(&itRefl, &itRefl);
        D3DXPlaneTransform(&wpln, &wpln, &itRefl);
        effect->SetVector(ehReflWaterClip, reinterpret_cast<const D3DXVECTOR4*>(&wpln));
    }

    // Near-field distance the cache owns in the reflection (Morrowind's view
    // distance by default). Statics, terrain, shadows and lights all hand off to
    // distant land at this distance.
    const float cacheNearDist = Configuration.UseSceneGraphSnapshot
                                ? nearViewRange * s_reflectionCacheNearFactor : 0.0f;

    // Rendering
    if (mwBridge->IsExterior() && (Configuration.MGEFlags & REFLECTIVE_WATER)) {
        // Draw land reflection, with opposite culling. Near-clip the LOD land so the
        // cache draws the real near terrain at full resolution; DL owns beyond.
        effect->SetFloat(ehLandNearCull, cacheNearDist);
        effect->BeginPass(PASS_RENDERLANDREFL);
        device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);
        renderDistantLand(effect, &reflView, &reflProj);
        effect->EndPass();
        effect->SetFloat(ehLandNearCull, 0);   // off for the main-view land later this frame

        // Real near terrain from the cache (two-texture AlphaGrid splat).
        renderReflectionTerrainFromCache(&reflView, &reflProj, cacheNearDist);
    }

    if (isDistantCell() && (Configuration.MGEFlags & REFLECT_NEAR)) {
        // Draw statics reflection, with opposite culling and no dissolve.
        // Per-object handover: distant statics whose object origin is within
        // cacheNearDist are culled whole (staticNearCull) — the cache draws those,
        // lit. DL owns everything beyond. Complementary to the cache's near-distance
        // test (same metric, same threshold) => no duplicates, no z-fight, no slice.
        DWORD p = (mwBridge->CellHasWeather() && !mwBridge->IsUnderwater(eyePos.z)) ? PASS_RENDERSTATICSEXTERIOR : PASS_RENDERSTATICSINTERIOR;
        effect->SetFloat(ehNearViewRange, 0);
        effect->SetFloat(ehStaticNearCull, cacheNearDist);
        effect->BeginPass(p);
        device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);
        renderReflectedStatics(&reflView, &reflProj);
        effect->EndPass();
        effect->SetFloat(ehStaticNearCull, 0);   // off for the main-view statics later this frame
        effect->SetFloat(ehNearViewRange, nearViewRange);
    }

    // Phase 0.5-B/C: inject the GeometryCache near-field set (NPCs + dynamic +
    // static objects within cacheNearDist, excluding terrain) into the reflection
    // with full FFE color, driven from the cache walk. Additive — runs inside the
    // clip-plane setup, after distant statics. renderMorrowind manages its own
    // effectFFE bracket (the distant-land effect is mid-Begin but not in a pass
    // here). No-op when the snapshot cache is empty/disabled.
    //
    // Sun shadows are folded INTO this color pass (the applyCacheShadow branch in
    // XE FixedFuncEmu.fx): each fragment computes its receiver term inline, which
    // replaces the standalone renderReflectionShadowsFromCache re-draw of the same
    // geometry (~1000 draws/frame). Reflected cache TERRAIN lost its receiver
    // re-draw with that pass (its color comes from the distant-land effect, not
    // FFE) — accepted for now; the DL LOD beyond the handover never had shadows.
    //
    // The fold gates/scales by surface lit-ness dot(normal, -sunVecView). The
    // cache geometry is rasterized in REFLECTED-view space (normals * reflView), so
    // sunVecView must be the sun in reflected-view space too — otherwise the dot is
    // inconsistent and flat up-facing surfaces fail the lit-ness term and receive
    // no shadow. setupCommonEffect set the main-view sun; swap to the reflected
    // sun for this pass, then restore for the rest of the frame.
    D3DXVECTOR3 sunVecViewRefl, sunVecViewMain;
    D3DXVec3TransformNormal(&sunVecViewRefl, reinterpret_cast<const D3DXVECTOR3*>(&sunVec), &reflView);
    D3DXVec3TransformNormal(&sunVecViewMain, reinterpret_cast<const D3DXVECTOR3*>(&sunVec), view);
    effect->SetFloatArray(ehSunVecView, sunVecViewRefl, 3);
    renderReflectionsFromCache(&reflView, &reflProj, cacheNearDist);
    effect->SetFloatArray(ehSunVecView, sunVecViewMain, 3);   // restore for later passes

    // Restore pass-all so the cache color shader (shared with the main reactive
    // scene) and any later passes don't clip against the water plane.
    const D3DXVECTOR4 reflClipPassAll(0, 0, 0, 1);
    effect->SetVector(ehReflWaterClip, &reflClipPassAll);

    if ((Configuration.MGEFlags & REFLECT_SKY) && !recordSky.empty() && !mwBridge->IsUnderwater(eyePos.z)) {
        // Draw sky reflection, with opposite culling
        renderReflectedSky();
    }

    // Restore view state
    device->SetRenderState(D3DRS_CLIPPLANEENABLE, 0);
    effect->SetMatrix(ehView, view);
    effect->SetMatrix(ehProj, proj);
}

void DistantLand::renderReflectedSky() {
    // Sky objects are not correctly positioned at infinity, so correction is required
    const float adjustZ = -2.0f * eyePos.z;
    D3DXMATRIX skyScale, worldTransform;
    D3DXMatrixScaling(&skyScale, 1e6, 1e6, 1e6);

    // Recorded renders
    const auto& recordSky_const = recordSky;
    const int standardCloudVerts = 65, standardCloudTris = 112;
    const int standardMoonVerts = 4, standardMoonTris = 2;

    // Render sky without clouds first
    effect->BeginPass(PASS_RENDERSKY);
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);

    for (const auto& i : recordSky_const) {
        // Skip clouds
        if (i.texture && i.vertCount == standardCloudVerts && i.primCount == standardCloudTris) {
            continue;
        }

        // Adjust world transform, as skydome objects are positioned relative to the viewer
        worldTransform = i.worldTransforms[0];
        worldTransform._43 += adjustZ;
        if (i.texture == nullptr) {
            // Inflate sky mesh towards infinity, makes skypos in shader calculate correctly
            D3DXMatrixMultiply(&worldTransform, &skyScale, &worldTransform);
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

        effect->SetMatrix(ehWorld, &worldTransform);
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
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_CCW);

    for (const auto& i : recordSky_const) {
        // Clouds only
        if (!(i.texture && i.vertCount == standardCloudVerts && i.primCount == standardCloudTris)) {
            continue;
        }

        // Adjust world transform, as skydome objects are positioned relative to the viewer
        worldTransform = i.worldTransforms[0];
        worldTransform._43 += adjustZ;

        effect->SetTexture(ehTex0, i.texture);
        effect->SetBool(ehHasAlpha, true);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, 1);
        device->SetRenderState(D3DRS_SRCBLEND, i.srcBlend);
        device->SetRenderState(D3DRS_DESTBLEND, i.destBlend);
        device->SetRenderState(D3DRS_ALPHATESTENABLE, 1);
        effect->SetMatrix(ehWorld, &worldTransform);
        effect->CommitChanges();

        device->SetStreamSource(0, i.vb, i.vbOffset, i.vbStride);
        device->SetIndices(i.ib);
        device->SetFVF(i.fvf);
        DrawStats::count(i.primCount);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);
    }
    effect->EndPass();
}

void DistantLand::renderReflectedStatics(const D3DXMATRIX* view, const D3DXMATRIX* proj) {
    DrawStats::ScopedStage _ds(DrawStats::ReflStatics);   // reflected distant statics (the "DL culled" set)
    // Select appropriate static clipping distance
    D3DXMATRIX ds_proj = *proj, ds_viewproj;
    float zn = 4.0f, zf = Configuration.DL.NearStaticEnd * kCellSize;

    // Don't draw beyond fully fogged distance; early out if frustum is empty
    zf = std::min(fogEnd, zf);
    if (zf <= zn) {
        return;
    }

    // Create a clipping frustum for visibility determination
    editProjectionZ(&ds_proj, zn, zf);
    ds_viewproj = (*view) * ds_proj;

    // Cull sort and draw
    ViewFrustum range_frustum(&ds_viewproj);
    D3DXVECTOR4 viewsphere(eyePos.x, eyePos.y, eyePos.z, zf);

    // Same near radius the cache reflection owns (renderWaterReflection), recomputed
    // here for the diagnostic near/far split below.
    const D3DXVECTOR3 eye(eyePos.x, eyePos.y, eyePos.z);
    const float diagCacheNearDist = Configuration.UseSceneGraphSnapshot
                                    ? nearViewRange * s_reflectionCacheNearFactor : 0.0f;

    if (Configuration.UseSharedMemory) {
        // Worker path: the cull worker already issued the reflection RPC,
        // materialized it, and culled it to reflectionSurvivors during the sky
        // window. Just draw the stable survivor set — no RPC, no wait, no IPC
        // window traversal on the main thread.
        if (reflStaticsWanted) {
            logReflStaticNearFar(reflectionSurvivors, eye, diagCacheNearDist);
            device->SetVertexDeclaration(StaticDecl);
            reflectionSurvivors.Render(device, effect, effect, &ehTex0, nullptr, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT);
            return;
        }

        // Inline fallback (worker not dispatched): issue + wait + materialize +
        // cull + draw, all on main. Single-threaded, so the live IPC window is
        // stable across the steps.
        visExtraShared.RemoveAll();
        ipcClient.getVisibleMeshes(visExtraSharedId, range_frustum, viewsphere, VIS_STATIC, VisibleSetSort::ByState);
        ipcClient.waitForCompletion();

        {
            MGE_ZoneScopedN("reflStatics:cull");
            DistantLand::materializeReflectionMeshes();
            // reflectionWaterRects was filled by the inline gate in renderStage0.
            DistantLand::cullReflectionSurvivors(ds_viewproj, ds_proj);
        }

        logReflStaticNearFar(reflectionSurvivors, eye, diagCacheNearDist);
        device->SetVertexDeclaration(StaticDecl);
        reflectionSurvivors.Render(device, effect, effect, &ehTex0, nullptr, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT);
    } else {
        VisibleSet<StlVector> visReflected((StlVector()));

        DistantLandShare::currentWorldSpace->NearStatics->GetVisibleMeshes(range_frustum, viewsphere, visReflected);
        DistantLandShare::currentWorldSpace->FarStatics->GetVisibleMeshes(range_frustum, viewsphere, visReflected);
        DistantLandShare::currentWorldSpace->VeryFarStatics->GetVisibleMeshes(range_frustum, viewsphere, visReflected);
        visReflected.SortByState();

        logReflStaticNearFar(visReflected, eye, diagCacheNearDist);
        device->SetVertexDeclaration(StaticDecl);
        visReflected.Render(device, effect, effect, &ehTex0, nullptr, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT);
    }
}

void DistantLand::clearReflection() {
    auto mwBridge = MWBridge::get();
    IDirect3DSurface9* target;
    DWORD baseColour;

    texReflection->GetSurfaceLevel(0, &target);
    if (mwBridge->CellHasWeather() || mwBridge->IsUnderwater(eyePos.z)) {
        // Use fog colour as reflection
        baseColour = (DWORD)horizonCol;
    } else {
        // Interior fog colour is typically too bright
        // Guess a reflection colour based on cell lighting parameters
        const BYTE* sun = mwBridge->getInteriorSun();
        RGBVECTOR c(sun[0] / 255.0f, sun[1] / 255.0f, sun[2] / 255.0f);
        c += ambCol;
        baseColour = (DWORD)c;
    }
    device->ColorFill(target, 0, baseColour);
    target->Release();
}

void DistantLand::simulateDynamicWaves() {
    MGE_ZoneScopedN("simulateDynamicWaves");
    DrawStats::ScopedStage _ds(DrawStats::Water);
    auto mwBridge = MWBridge::get();

    static bool resetRippleSurface = true;
    static float remainingWaveTime = 0;
    static const float waveStep = 0.0125f;  // time per wave simulation step (1/80 sec)

    // Simulation paused in menu mode
    if (mwBridge->IsMenu()) {
        return;
    }

    device->SetFVF(fvfWave);
    device->SetStreamSource(0, vbWaveSim, 0, 32);

    // Calc number of wave iterations to run this frame
    float frameTime = std::min(mwBridge->frameTime(), 0.5f);
    remainingWaveTime += frameTime;
    int numWaveSteps = (int)(remainingWaveTime / waveStep);
    remainingWaveTime -= numWaveSteps * waveStep;

    // Preciptation (rain/snow) ripples
    if (mwBridge->CellHasWeather()) {
        static float remainingRipples = 0;

        // Reset surface when not needed next time
        resetRippleSurface = true;

        // Weather types: rain = 4; thunderstorm = 5; snow = 8; blizzard = 9
        // Thunderstorm causes 50% more ripples
        int w0 = mwBridge->GetCurrentWeather(), w1 = mwBridge->GetNextWeather();
        float precipitation0 = (w0 == 4 || w0 == 5 || w0 == 8 || w0 == 9) ? 1.0f : -1.5f;
        float precipitation1 = (w1 == 4 || w1 == 5 || w1 == 8 || w1 == 9) ? 1.0f : -1.5f;
        precipitation0 += (w0 == 5) ? 0.5f : 0;
        precipitation1 += (w1 == 5) ? 0.5f : 0;

        // 150 drops per second for normal precipitation
        float precipitation = (1.0f - mwBridge->GetWeatherRatio()) * precipitation0 + mwBridge->GetWeatherRatio() * precipitation1;
        float rippleFrequency = 150.0f * precipitation;

        if (rippleFrequency > 0) {
            static double randomizer = 0.546372819;
            int ripplePos[2];
            RECT drop;

            remainingRipples += rippleFrequency * frameTime;
            int n = int(std::floor(remainingRipples));
            remainingRipples -= n;

            while (n-- > 0) {
                // Place rain ripple at random location
                for (int i = 0; i != 2; ++i) {
                    randomizer = randomizer * (1337.134511337451 + 0.0001 * rand()) + 0.12351523;
                    randomizer -= floor(randomizer);
                    ripplePos[i] = (int)(randomizer * waveTexResolution);
                }

                drop.left = ripplePos[0] - 2;
                drop.right = ripplePos[0] + 2;
                drop.top = ripplePos[1] - 1;
                drop.bottom = ripplePos[1] + 1;
                device->ColorFill(surfRain, &drop, 0x6060);

                drop.left = ripplePos[0] - 1;
                drop.right = ripplePos[0] + 1;
                drop.top = ripplePos[1] - 2;
                drop.bottom = ripplePos[1] + 2;
                device->ColorFill(surfRain, &drop, 0x6060);

                drop.left = ripplePos[0] - 1;
                drop.right = ripplePos[0] + 1;
                drop.top = ripplePos[1] - 1;
                drop.bottom = ripplePos[1] + 1;
                device->ColorFill(surfRain, &drop, 0x4040);
            }
        }

        // Apply wave equation numWaveSteps times
        // Uses double buffering to avoid reads and writes on the same target
        RenderTargetSwitcher rtsw(surfRippleBuffer, NULL);
        SurfaceDoubleBuffer doublebuffer;
        doublebuffer.init(texRain, surfRain, texRippleBuffer, surfRippleBuffer);

        effect->BeginPass(PASS_WAVESTEP);
        for (int i = 0; i != numWaveSteps; ++i) {
            device->SetRenderTarget(0, doublebuffer.sinkSurface());
            effect->SetTexture(ehTex4, doublebuffer.sourceTexture());
            effect->CommitChanges();

            DrawStats::count(1);
            device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 1);
            doublebuffer.cycle();
        }
        effect->EndPass();

        if (doublebuffer.sourceSurface() != surfRain) {
            device->StretchRect(surfRippleBuffer, 0, surfRain, 0, D3DTEXF_NONE);
        }
    } else if (resetRippleSurface) {
        // No weather - clear rain ripples
        device->ColorFill(surfRain, NULL, 0);
        resetRippleSurface = false;
    }

    // Player local ripples
    // Move ripple texture with player; lock to texel alignment to prevent visible jitter
    const D3DXVECTOR3* playerPos = (const D3DXVECTOR3*)mwBridge->PlayerPositionPointer();
    static int lastXpos = (int)floor(playerPos->x / waveTexWorldRes);
    static int lastYpos = (int)floor(playerPos->y / waveTexWorldRes);

    int newXpos = (int)floor(playerPos->x / waveTexWorldRes);
    int newYpos = (int)floor(playerPos->y / waveTexWorldRes);

    int shiftX = newXpos - lastXpos;
    int shiftY = newYpos - lastYpos;

    lastXpos = newXpos;
    lastYpos = newYpos;

    int shiftXp = (shiftX > 0) ? +shiftX : 0;
    int shiftXn = (shiftX < 0) ? -shiftX : 0;
    int shiftYp = (shiftY > 0) ? +shiftY : 0;
    int shiftYn = (shiftY < 0) ? -shiftY : 0;

    // Shift texture by (shiftX, shiftY) pixels
    RECT source;
    source.left = 1 + shiftXp;
    source.right = waveTexResolution - shiftXn;
    source.top = 1 + shiftYp;
    source.bottom = waveTexResolution - shiftYn;

    RECT target;
    target.left = 1 + shiftXn;
    target.right = waveTexResolution - shiftXp;
    target.top = 1 + shiftYn;
    target.bottom = waveTexResolution - shiftYp;

    device->ColorFill(surfRippleBuffer, 0, 0);
    device->StretchRect(surfRipples, &source, surfRippleBuffer, &target, D3DTEXF_NONE);

    // Water simulation; realigned water starts in surfRippleBuffer
    // Uses double buffering to avoid reads and writes on the same target
    RenderTargetSwitcher rtsw(surfRipples, NULL);
    SurfaceDoubleBuffer doublebuffer;
    doublebuffer.init(texRippleBuffer, surfRippleBuffer, texRipples, surfRipples);

    float rippleOrigin[2];
    float dz = playerPos->z - mwBridge->WaterLevel();
    if (dz < 0 && dz > -128.0f * mwBridge->PlayerHeight()) {
        // Create waves around the player
        effect->BeginPass(PASS_PLAYERWAVE);
        for (int i = 0; i != numWaveSteps; ++i) {
            // Interpolate between starting and ending point, so that low framerates do not cause less waves
            float w = -(float)i / (float)numWaveSteps / (float)waveTexResolution;
            rippleOrigin[0] = w * shiftX + 0.5f;
            rippleOrigin[1] = w * shiftY + 0.5f;

            device->SetRenderTarget(0, doublebuffer.sinkSurface());
            effect->SetTexture(ehTex4, doublebuffer.sourceTexture());
            effect->SetFloatArray(ehRippleOrigin, rippleOrigin, 2);
            effect->CommitChanges();

            DrawStats::count(1);
            device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 1);
            doublebuffer.cycle();
        }
        effect->EndPass();
    }

    // Apply wave equation numWaveSteps times
    effect->BeginPass(PASS_WAVESTEP);
    for (int i = 0; i != numWaveSteps; ++i) {
        device->SetRenderTarget(0, doublebuffer.sinkSurface());
        effect->SetTexture(ehTex4, doublebuffer.sourceTexture());
        effect->CommitChanges();

        DrawStats::count(1);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 1);
        doublebuffer.cycle();
    }
    effect->EndPass();

    if (doublebuffer.sourceSurface() != surfRipples) {
        device->StretchRect(surfRippleBuffer, 0, surfRipples, 0, D3DTEXF_NONE);
    }

    // Set wave texture world origin
    static float halfWaveTexWorldSize = 0.5f * waveTexWorldRes * waveTexResolution;
    rippleOrigin[0] = lastXpos*waveTexWorldRes - halfWaveTexWorldSize;
    rippleOrigin[1] = lastYpos*waveTexWorldRes - halfWaveTexWorldSize;
    effect->SetFloatArray(ehRippleOrigin, rippleOrigin, 2);

    // Set weather-dependent wave height
    effect->SetFloat(ehWaveHeight, (float)Configuration.DL.WaterWaveHeight);
}

void DistantLand::simulateFoam() {
    MGE_ZoneScopedN("simulateFoam");
    DrawStats::ScopedStage _ds(DrawStats::Water);
    auto mwBridge = MWBridge::get();

    // Paused in menu; needs the flow field bound (it IS the advecting velocity field).
    if (mwBridge->IsMenu()) {
        return;
    }
    if (!updateFlowMapTexture()) {
        return;
    }

    // Substep cadence (independent of the ripple sim; capped to bound the GPU cost).
    static float remainingFoamTime = 0;
    static const float foamStep = 1.0f / 60.0f;
    float frameTime = std::min(mwBridge->frameTime(), 0.5f);
    remainingFoamTime += frameTime;
    int numSteps = (int)(remainingFoamTime / foamStep);
    remainingFoamTime -= numSteps * foamStep;
    numSteps = std::max(1, std::min(numSteps, 3));

    device->SetFVF(fvfWave);
    device->SetStreamSource(0, vbFoamSim, 0, 32);   // foam's own RT-sized fullscreen triangle (shared by all cascades)

    const D3DXVECTOR3* playerPos = (const D3DXVECTOR3*)mwBridge->PlayerPositionPointer();

    // Shared flow uniforms the foam passes read (sampFlow at world pos) — cascade-invariant,
    // set once. The foam tuning (foamParams) is per-cascade and set inside the loop below.
    float ft[4];
    getFlowMapTransform(ft);
    effect->SetTexture(ehFlow, texFlow);
    effect->SetFloatArray(ehFlowTransform, ft, 4);
    effect->SetFloat(ehFlowWeight, waterFlowDebugOn ? 1.0f : 0.0f);
    // (river-mask soften/expand is now baked into the routing alpha — see buildWaterFlowMap 5c-bis)
    // UV-advection (River Editor): rate tied to the FoamSpeed knob; decay bounds the offset stretch.
    effect->SetFloat(ehFoamUVRate, foamDetailSpeed);
    effect->SetFloat(ehFoamUVDecay, foamUVDecay);
    effect->SetFloat(ehFoamGaussRadius, foamGaussRadius);
    effect->SetFloat(ehFoamMinDensity, foamMinDensity);

    // Player in-water flag (window texel position is cascade-dependent, computed per cascade).
    float dz = playerPos->z - mwBridge->WaterLevel();
    bool inWater = (dz < 0 && dz > -128.0f * mwBridge->PlayerHeight());

    // Save the frame RT once; the passes below retarget manually and it is restored on scope exit.
    RenderTargetSwitcher rtsw(surfFoamP_A[0], NULL);

    // Force the viewport to the foam RT size after every retarget (matches SetRenderTarget's
    // auto-size; defends against a driver leaving it stale).
    D3DVIEWPORT9 savedVp;
    device->GetViewport(&savedVp);
    const D3DVIEWPORT9 foamVp = { 0, 0, (DWORD)foamTexResolution, (DWORD)foamTexResolution, 0.0f, 1.0f };

    // The reset is one shared flag across cascades — capture it, clear once after the loop,
    // so every cascade clears on the resetting frame.
    const bool doReset = foamSimReset;

    // Single world-anchored low-res carrier (foamCascades == 1; the loop carries over from the
    // cascade split and now runs once). The water shader's procedural detail supplies the
    // up-close crispness, so this sim stays cheap and coarse.
    for (int c = 0; c < foamCascades; ++c) {
        const float worldRes = foamCascadeWorldRes[c];
        effect->SetFloat(ehFoamWorldRes, worldRes);

        // Per-cascade sim tuning (force/decay/pressure/scale).
        float foamParams[4] = { foamFlowForce[c], foamDecay[c], foamPressure[c], foamScale[c] };
        effect->SetFloatArray(ehFoamParams, foamParams, 4);

        // Micro-substep the advection so each step moves ≤1 texel (the 8-neighbour Voronoi
        // tracker's hard limit). advTotal is the world-relative per-substep texel advance
        // (= 3.125/worldRes — constant world speed across cascades); split it into `micro`
        // equal micro-steps each ≤1 texel. Coarse cascade (advTotal 0.25) → micro 1 (no cost
        // change); fine cascade (advTotal 2.0 @ 1.5625u) → micro 2. The advect loop runs
        // numSteps*micro iterations so the per-frame world advance is preserved.
        const float advTotal = 3.125f * foamSimSpeed / worldRes;   // SimSpeed scales sim-foam visual speed
        const int   micro    = std::max(1, (int)ceilf(advTotal));
        const float advStep  = advTotal / micro;
        effect->SetFloat(ehFoamAdvance, advStep);

        // World-anchor: track the player at texel granularity so foam stays world-locked
        // (the window scrolls, its contents do not). Mirrors the ripple sim's tracking.
        int newXpos = (int)floor(playerPos->x / worldRes);
        int newYpos = (int)floor(playerPos->y / worldRes);
        int shiftX = newXpos - foamLastXpos[c];
        int shiftY = newYpos - foamLastYpos[c];
        foamLastXpos[c] = newXpos;
        foamLastYpos[c] = newYpos;

        // First step (or device reset): clear the buffers — the advect respawn self-seeds
        // particles into every texel within a frame.
        if (doReset) {
            device->ColorFill(surfFoamP_A[c], 0, 0);
            device->ColorFill(surfFoamP_B[c], 0, 0);
            device->ColorFill(surfFoamField[c], 0, 0);
            device->ColorFill(surfFoam[c], 0, 0);
            device->ColorFill(surfFoamUV_A[c], 0, 0);
            device->ColorFill(surfFoamUV_B[c], 0, 0);
            shiftX = shiftY = 0;
        }

        // Shift the persistent particle buffer (A) into the scratch (B) by the integer texel
        // offset; the advect pass then subtracts foamShiftPx from each stored position so the
        // window-local coords survive the shift (the re-bin fixup). Clear B so streamed-in
        // border texels are empty (they respawn).
        int shiftXp = (shiftX > 0) ? +shiftX : 0;
        int shiftXn = (shiftX < 0) ? -shiftX : 0;
        int shiftYp = (shiftY > 0) ? +shiftY : 0;
        int shiftYn = (shiftY < 0) ? -shiftY : 0;
        RECT source, target;
        source.left = shiftXp;   source.right  = foamTexResolution - shiftXn;
        source.top  = shiftYp;   source.bottom = foamTexResolution - shiftYn;
        target.left = shiftXn;   target.right  = foamTexResolution - shiftXp;
        target.top  = shiftYn;   target.bottom = foamTexResolution - shiftYp;
        device->ColorFill(surfFoamP_B[c], 0, 0);
        device->StretchRect(surfFoamP_A[c], &source, surfFoamP_B[c], &target, D3DTEXF_NONE);

        // Window world origin (min corner): foam particle (tx,ty) → world = origin + (tx,ty)*res.
        const float halfWorld = 0.5f * worldRes * foamTexResolution;
        float foamOrigin[2] = { newXpos * worldRes - halfWorld,
                                newYpos * worldRes - halfWorld };
        effect->SetFloatArray(ehFoamOrigin, foamOrigin, 2);   // transient per-cascade for the sim
        foamOriginC[c][0] = foamOrigin[0];                    // saved for the consume bind
        foamOriginC[c][1] = foamOrigin[1];

        // Player injection input: window texel position + in-water flag.
        float foamPlayer[3] = { (playerPos->x - foamOrigin[0]) / worldRes,
                                (playerPos->y - foamOrigin[1]) / worldRes,
                                inWater ? 1.0f : 0.0f };
        effect->SetFloatArray(ehFoamPlayer, foamPlayer, 3);

        // Full per-frame window shift (texels) for the pressure-field read. The field buffer is
        // not re-binned, so it lags the scrolled particles by the whole frame's shift on EVERY
        // substep (unlike foamShiftPx, which re-bins the particles once on substep 0). The advect
        // offsets the field sample by this to kill the movement-fed brightness drift.
        float fieldShift[2] = { (float)shiftX, (float)shiftY };
        effect->SetFloatArray(ehFoamFieldShift, fieldShift, 2);

        device->SetViewport(&foamVp);

        // PASS_FOAM_ADVECT — Voronoi particle step, ping-pong B→A. Source is the realigned B.
        SurfaceDoubleBuffer pb;
        pb.init(texFoamP_B[c], surfFoamP_B[c], texFoamP_A[c], surfFoamP_A[c]);
        effect->BeginPass(PASS_FOAM_ADVECT);
        const int totalIters = numSteps * micro;
        for (int i = 0; i < totalIters; ++i) {
            // The re-bin shift applies once (first iteration only); later steps do not re-shift.
            float shiftPx[2] = { (i == 0) ? (float)shiftX : 0.0f, (i == 0) ? (float)shiftY : 0.0f };
            device->SetRenderTarget(0, pb.sinkSurface());
            device->SetViewport(&foamVp);
            effect->SetTexture(ehFoamParticles, pb.sourceTexture());
            effect->SetTexture(ehFoamFieldIn, texFoamField[c]);   // previous frame's field (pressure source)
            effect->SetFloatArray(ehFoamShift, shiftPx, 2);
            effect->CommitChanges();
            DrawStats::count(1);
            device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 1);
            pb.cycle();
        }
        effect->EndPass();
        // Ensure the canonical particle buffer (A) holds the final result.
        if (pb.sourceSurface() != surfFoamP_A[c]) {
            device->StretchRect(pb.sourceSurface(), 0, surfFoamP_A[c], 0, D3DTEXF_NONE);
        }

        // PASS_FOAM_FIELD — smooth Voronoi velocity + density from A into the field buffer.
        device->SetRenderTarget(0, surfFoamField[c]);
        device->SetViewport(&foamVp);
        effect->BeginPass(PASS_FOAM_FIELD);
        effect->SetTexture(ehFoamParticles, texFoamP_A[c]);
        effect->CommitChanges();
        DrawStats::count(1);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 1);
        effect->EndPass();

        // PASS_FOAM_EXTRACT — foam = clamp(scale·|vorticity|), river-masked, into texFoam.
        device->SetRenderTarget(0, surfFoam[c]);
        device->SetViewport(&foamVp);
        effect->BeginPass(PASS_FOAM_EXTRACT);
        effect->SetTexture(ehFoamFieldIn, texFoamField[c]);
        effect->CommitChanges();
        DrawStats::count(1);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 1);
        effect->EndPass();

        // PASS_FOAM_UV — advect the detail UV offset field through the velocity (River Editor).
        // Re-bin the canonical UV (A) into B by the window shift (plain scroll; the stored offsets
        // are displacements, so their values need no fixup — only the buffer content scrolls), then
        // advect B → A reading the current velocity field.
        device->ColorFill(surfFoamUV_B[c], 0, 0);
        device->StretchRect(surfFoamUV_A[c], &source, surfFoamUV_B[c], &target, D3DTEXF_NONE);
        device->SetRenderTarget(0, surfFoamUV_A[c]);
        device->SetViewport(&foamVp);
        effect->BeginPass(PASS_FOAM_UV);
        effect->SetTexture(ehFoamUVIn, texFoamUV_B[c]);
        effect->SetTexture(ehFoamFieldIn, texFoamField[c]);
        effect->CommitChanges();
        DrawStats::count(1);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 1);
        effect->EndPass();
    }

    foamSimReset = false;
    device->SetViewport(&savedVp);   // restore the frame viewport for subsequent rendering
}

void DistantLand::renderWaterPlane() {
    DrawStats::ScopedStage _ds(DrawStats::Water);
    D3DXMATRIX m;
    IDirect3DTexture9* texRefract = PostShaders::borrowBuffer(0);

    // Snap the surface height by underwater state so it never intersects the camera
    // at the transition (see s_waterMeshSnap*). The snap lands at the IsUnderwater
    // threshold where the camera is at the surface, hiding the jump.
    const bool underwater = MWBridge::get()->IsUnderwater(eyePos.z);
    const float waterZ = MWBridge::get()->WaterLevel()
                       + (underwater ? s_waterMeshSnapUnderwater : s_waterMeshSnapAbove);
    effect->SetTexture(ehTex0, texReflection);
    effect->SetTexture(ehTex1, texWater);
    effect->SetTexture(ehTex2, texRefract);
    effect->SetTexture(ehTex3, texDepthFrame);
    if (Configuration.MGEFlags & DYNAMIC_RIPPLES) {
        effect->SetTexture(ehTex4, texRain);
        effect->SetTexture(ehTex5, texRipples);
    }
    if (Configuration.UseWaterFlowMap && updateFlowMapTexture()) {
        float ft[4];
        getFlowMapTransform(ft);
        effect->SetTexture(ehFlow, texFlow);
        effect->SetFloatArray(ehFlowTransform, ft, 4);
        // Debug A/B (VK_NUMPAD9): 1 = flow map active, 0 = neutral (today's look).
        effect->SetFloat(ehFlowWeight, waterFlowDebugOn ? 1.0f : 0.0f);
        effect->SetFloat(ehFlowScroll, waterFlowScroll);
        effect->SetFloat(ehFlowSeaSpeed, waterFlowSeaSpeed);
        effect->SetFloat(ehFlowCycleUV, waterFlowCycleUV);
        effect->SetFloat(ehFlowSeaRefract, waterFlowSeaRefract);
        effect->SetFloat(ehFlowDebugView, (float)waterFlowDebugView);
        effect->SetFloat(ehFlowWarp, waterFlowWarp);
        // World-anchored particle foam carrier: bind the single low-res buffer + its window
        // origin (saved by simulateFoam). foamWeight is the runtime A/B (0 → legacy water).
        effect->SetTexture(ehFoam0, texFoam[0]);
        effect->SetTexture(ehFoamUVTex, texFoamUV_A[0]);   // advected detail-UV offset field
        effect->SetFloatArray(ehFoamOrigin0, foamOriginC[0], 2);
        effect->SetFloat(ehFoamWeight, waterFoamOn ? 1.0f : 0.0f);
        // Indirection detail (Phase 1): procedural fbm modulating the carrier (live-tuned).
        float foamDetail[4] = { foamDetailTile, foamDetailSpeed, foamErodeThreshold, foamMix };
        effect->SetFloatArray(ehFoamDetail, foamDetail, 4);
        effect->SetFloat(ehFoamFarAmount, foamFarAmount);
        effect->SetFloat(ehFoamVortGain, foamVortGain);
        effect->SetFloat(ehFoamFineScale, foamFineScale);
        effect->SetFloat(ehFoamFineAmt, foamFineAmt);
        effect->SetFloat(ehFoamCoarseScale, foamCoarseScale);
        effect->SetFloat(ehFoamCoarseAmt, foamCoarseAmt);
    }
    if (Configuration.UseWaterFlowMap) {
        // Flow-steered crest displacement knobs (WATER_LOD_MESH). Set every frame the
        // LOD shader is compiled — even before the flow texture is ready — so waveLen
        // is never the 0 default (which would divide-by-zero in the displacement VS).
        effect->SetFloat(ehWaveAmp, waterWaveAmp);
        effect->SetFloat(ehWaveLen, waterWaveLen);
        effect->SetFloat(ehWaveSpeed, waterWaveSpeed);
        effect->SetFloat(ehCrestSpread, waterCrestSpread);
    }

    device->SetVertexDeclaration(WaterDecl);

    // Force LINEAR min/mag on the pixel-stage samplers. The water effect's sampler_state
    // blocks declare linear, but Begin() runs with D3DXFX_DONOTSAVESTATE and the effect's
    // internal state-cache can desync across passes, leaving a register at the device-default
    // POINT — which showed as pixellated flow-driven normals/foam. Set after BeginPass (done by
    // the caller) and after the last CommitChanges so these win for the draw. All water samplers
    // want linear; vertex-texture-fetch stages (256+) are untouched (point-only by hardware).
    for (DWORD s = 0; s < 10; ++s) {
        device->SetSamplerState(s, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
        device->SetSamplerState(s, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
    }

    // World-snapped LOD path (clipmap): one snapped world matrix + draw per level so
    // vertices land on a stable world lattice (no swimming). Else the radial mesh.
    const bool useLod = waterLodMeshOn && Configuration.UseWaterFlowMap
                        && vbWaterLod && !waterLodLevels.empty();
    if (useLod) {
        device->SetStreamSource(0, vbWaterLod, 0, 12);
        device->SetIndices(ibWaterLod);
        for (const WaterLodLevel& lvl : waterLodLevels) {
            // Snap the level origin to 2*cellSize so each finer level's boundary lands
            // on the coarser grid lines (shared world-lattice points → no seam).
            const float snap = 2.0f * lvl.cellSize;
            const float originX = floorf(eyePos.x / snap) * snap;
            const float originY = floorf(eyePos.y / snap) * snap;
            // World = scale(cellSize, cellSize, 1) * translate(origin, waterZ).
            D3DXMatrixScaling(&m, lvl.cellSize, lvl.cellSize, 1.0f);
            m._41 = originX;
            m._42 = originY;
            m._43 = waterZ;
            effect->SetMatrix(ehWorld, &m);
            effect->CommitChanges();

            // Flexible-trim variant: shift the hole by eye parity so it nests exactly
            // over the finer level's snapped footprint (kills the coverage-gap flicker).
            int variant = 0;
            if (lvl.numVariants > 1) {
                const int ex = int(((long long)floorf(eyePos.x / lvl.cellSize)) & 1);
                const int ey = int(((long long)floorf(eyePos.y / lvl.cellSize)) & 1);
                variant = ey * 2 + ex;
            }

            DrawStats::count(lvl.triCount[variant]);
            device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, lvl.vertBase, lvl.vertCount,
                                         lvl.ibStart[variant], lvl.triCount[variant]);
        }
    } else {
        D3DXMatrixTranslation(&m, eyePos.x, eyePos.y, waterZ);
        effect->SetMatrix(ehWorld, &m);
        effect->CommitChanges();

        device->SetStreamSource(0, vbWater, 0, 12);
        device->SetIndices(ibWater);
        DrawStats::count(numWaterTris);
        device->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, numWaterVerts, 0, numWaterTris);
    }
}
