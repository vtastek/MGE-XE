#pragma once

#include "quadtree.h"
#include "ffeshader.h"
#include "mwbridge.h"
#include "ipc/client.h"
#include "ipc/dlshare.h"
#include "ipc/occlusionmask.h"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <memory>



struct MGEShader;

class DistantLand {
public:
    struct DynamicVisGroup {
        enum class DataSource : uint8_t {
            Journal = 1,
            Global = 2,
            UniqueObject = 3
        };
        struct Range {
            int begin, end;
        };

        DataSource source;
        bool enabled;
        const void *gameObject;
        std::string id;
        std::vector<Range> ranges;
        std::vector<QuadTreeMesh*> references;

        void push_back(QuadTreeMesh* mesh) {
            references.push_back(mesh);
        }
    };

    static constexpr float kCellSize = 8192.0f;
    static constexpr float kDistantZBias = 5e-6f;
    static constexpr float kDistantNearPlane = 4.0f;
    static constexpr float kMoonTag = 88888.0f;

    static bool ready;
    static bool isRenderCached;
    static bool isPPLActive;
    // Phase 1 Milestone 1 A/B toggle (NUMPAD7). false = ENGINE (untouched
    // S4: cacheOpaqueMode / cacheOnlyMode (the NUMPAD7 opaque-source cycle) went with
    // rendercachedcolor.cpp — the Forge host is the opaque takeover CACHE mode reached for.
    // Set by frameSetupEarly() when the GeometryCache walk ran at BeginScene(0);
    // read by renderStage0 to skip its own redundant fallback walk.
    static bool earlyWalkedCache;
    // Async full-frame overlap (Phase 2): latched by frameSetupEarly() when THIS frame
    // runs the early Forge kickoff at BeginScene(0) — i.e. the Forge baseline is up
    // (seam compositing + Forge water), one warm-up frame after any transition.
    // Covers exterior distant cells (2a: the distant-statics cull is gated off — the
    // host renders those statics) AND plain interiors / non-distant cells (2b: no
    // scene-0 RPC exists there; the interior cache walk + visible set are hoisted to
    // frameSetupEarly). Interior DISTANT cells stay on the late kickoff — MGE still
    // draws their distant statics, so its cull keeps the channel. While latched, the
    // whole scene 0 is the IPC-free async window; the grass cull runs pre-kickoff
    // instead of in renderDepth. Read by mged3d8device (fire the kickoff after
    // frameSetupEarly), renderStage0 (skip the fallback statics kickoff) and
    // renderDepth (skip cullDistantStatics_finish). Stable for the frame: the seam's
    // The F11 toggle is polled at composite-finish time, after every consumer.
    static bool earlyForgeKickoff;
    // MENU FREEZE (USE_MENU_CACHING, "pause world during menus"). While latched, the world is not
    // re-rendered at all: frameSetupEarly returns before the grass cull / park fire / statics cull /
    // cache walk / classify / visible set / produce kick, mged3d8device skips the host kickoff, and
    // the composite blit re-shows the LAST host frame from g_mainTex. MW's own UI/HUD still draws
    // over it, so menus stay fully interactive on a frozen world — which is free, because menu mode
    // pauses simulation.
    //
    // Implies earlyForgeKickoff (never latches on warm-up / late-kickoff frames, where EndScene
    // would kick anyway). NOT a hard freeze: the world still refreshes on a cadence (every 8th
    // menu frame) plus a short burst after each click, so a change made from a menu — or anything
    // evolving on its own — appears within a few frames and the host's per-frame settle/expiry
    // logic keeps advancing. Requires a valid composited frame to re-show
    // (RenderProcess::hasCompositeFrame).
    static bool menuFreeze;
    // Phase 1 (MW-only pipeline): route the Forge produce's visible set OFF the engine
    // MSOC classify. When true, liveDrawBuild is forced off (onFrameReady's full refresh
    // walk discovers newly-visible objects instead of the classify's lazy-capture) and
    // buildFrustumVisibleSet takes the frustum-only branch (whole-cache frustum cull, no
    // s_visibleKeys dependency) — the host's two-phase Hi-Z GPU cull then owns ALL
    // occlusion. A/B live via VK_SCROLL. Boot default off (= known-good MSOC path).
    // DISPROVEN as a deletion route (2026-07-26): flipping this on costs a 7.05 ms full
    // cache walk (runRefreshWalks over 12,572 entries) because the classify is not just an
    // occlusion verdict — it is the engine-driven DISCOVERY feed. earlyClassifyMainScene
    // therefore STAYS; retiring MSOC means absorbing the CullShow detour that produces the
    // feed, not deleting the consumer. See tasks/msoc-detour-absorb.md. Keep this knob: it
    // is the A/B baseline (frustum-only) the absorbed path is measured against.
    static bool hostCullOnly;
    // MW-ONLY-UI: BITMASK of the world roots the engine is forbidden to traverse
    // (GeometryCache::kSuppressLand/Pick/Objects). Independent bits, not a ladder — a cumulative
    // level cannot attribute a missing effect to a root. Set from the Forge Dev imgui panel; no
    // key, the keyspace is full and the state needs to be readable.
    static int  mwWorldSuppress;

    static IDirect3DDevice9* device;
    static ID3DXEffect* effect;
    static ID3DXEffectPool* effectPool;
    static IDirect3DVertexDeclaration9* LandDecl;
    static IDirect3DVertexDeclaration9* StaticDecl;
    // S5a: PosOnlyDecl (the position-only decl for the fullscreen quad) went with the
    // depth pre-pass's float-depth clear, its last user.


    static IPC::Client ipcClient;
    static std::vector<DynamicVisGroup> dynamicVisGroups;
    static void* lastDistantVisCell;
    static bool isDistantLandLoaded;

    // S4b: visLand / visDistant / visDistantSurvivors and their IPC twins are gone with
    // MGE's distant-land renderer. The Forge host owns distant land and statics.
    // visLand was never populated at all — nothing ever requested VIS_LAND.
    static IPC::VecView<IPC::DynVisFlag> dynVisFlagsShared;
    // Single-window chunk vec carrying the occlusion-mask blob shipped to the
    // host each frame (host-side cull). Empty/InvalidVector when disabled.
    static IPC::VecView<OcclusionMask::MaskChunk> maskBlobShared;

    static IPC::VecId dynVisFlagsSharedId;
    static IPC::VecId maskBlobSharedId;

    // Number of z-writing draws MW has issued in the current scene. Bumped in
    // inspectIndexedPrimitive, reset at scene 0 (renderStage0 / renderStage1). Its one
    // consumer is the sky predicate there: MW's sky is the first blended geometry of a
    // weather cell's scene 0, i.e. the one drawn while this is still 0.
    // (S5a: was recordMW, a vector<RecordedState> that AddRef'd VB/IB/texture per DIP so
    // the draws could be replayed into the depth texture. Nothing replays them now.)
    static unsigned recordMWCount;

    // (Removed: LandMeshCache / landMeshes — the CPU-side copy of every distant-land
    // tile's triangle mesh, ~20 MB per worldspace. Its only consumer was
    // contributeDistantLandOccluders, which fed the plugin's MSOC horizon curtain; that
    // went out with renderexterior.cpp in S4, leaving the map written and never read.
    // See tasks/msoc-retirement.md.)

    static IDirect3DTexture9* texWorldColour, *texWorldNormals, *texWorldDetail;
    static IDirect3DTexture9* texMenuCache;
    // S3: texReflection / surfReflectionZ (water reflection RT), texWater (the animated
    // volume normal map), vbWater / ibWater (the radial water mesh) went with
    // renderwater.cpp -- the Forge host owns the water surface.

    // S3: the geo-clipmap water LOD mesh, dynamic-ripple/wave sim targets, the baked
    // water flow map and the whole two-cascade Voronoi foam simulation lived here.
    // None of it ever reached the host -- the Forge water shader carries its own
    // wave and foam model -- so the entire stack died with MGE's water renderer.

    // S3: the cascaded shadow atlas (texShadow / texSoftShadow / surfShadowZ) and the
    // frustum clip cube it projected went with rendershadow.cpp. S5a: vbFullFrame, the
    // generic fullscreen quad, went with the depth pre-pass that was its last user.

    static D3DXMATRIX mwView, mwProj;
    static D3DXMATRIX smView[2], smProj[2], smViewproj[2];
    static D3DXVECTOR4 eyeVec, eyePos, sunVec, sunPos;
    static float sunVis;
    static RGBVECTOR sunCol, sunAmb, ambCol;
    static RGBVECTOR nearFogCol, horizonCol;
    static RGBVECTOR atmOutscatter, atmInscatter;
    static D3DXVECTOR4 atmSkylightScatter;
    static float fogStart, fogEnd;
    static float fogExpStart, fogExpDivisor;
    static float fogNearStart, fogNearEnd;
    static float nearViewRange;
    static float windScaling, niceWeather;
    static float lightSunMult, lightAmbMult;

    static D3DXHANDLE ehRcpRes, ehShadowRcpRes;
    static D3DXHANDLE ehWorld, ehView, ehProj;
    static D3DXHANDLE ehShadowViewproj;
    static D3DXHANDLE ehVertexBlendState, ehVertexBlendPalette;
    static D3DXHANDLE ehBoneMatrices;
    static D3DXHANDLE ehAlphaRef, ehMaterialAlpha;
    static D3DXHANDLE ehHasAlpha, ehHasBones, ehHasVCol;
    static D3DXHANDLE ehTex0, ehTex1, ehTex2, ehTex3, ehTex4, ehTex5;
    static D3DXHANDLE ehEyePos, ehFootPos;
    static D3DXHANDLE ehSunCol, ehSunAmb, ehSunVec, ehSunVecView;
    static D3DXHANDLE ehSkyCol, ehFogColNear, ehFogColFar;
    static D3DXHANDLE ehSunPos, ehSunVis;
    static D3DXHANDLE ehOutscatter, ehInscatter, ehSkyScatterFar;
    static D3DXHANDLE ehFogStart, ehFogRange;
    static D3DXHANDLE ehFogNearStart, ehFogNearRange;
    static D3DXHANDLE ehNearViewRange;
    static D3DXHANDLE ehStaticNearCull;
    static D3DXHANDLE ehLandNearCull;
    // Texture-light path handles on the distant-land effect (cache terrain point
    // lights). Bound per-patch from selectTextureLights output in renderCachedTerrain.
    static D3DXHANDLE ehLightData, ehLightDataParams, ehLightIndices, ehTexLightView;
    static D3DXHANDLE ehWindVec;
    static D3DXHANDLE ehNiceWeather;
    static D3DXHANDLE ehTime;

    static std::function<void(IDirect3DSurface9*)> captureScreenHandler;
    static bool captureScreenWithUI;

    static bool init();
    static bool initIpc();
    static bool initShader();
    static bool initLandscapeClient();
    static bool initLandscape();
    static bool initDistantStaticsClient();
    static void loadVisGroupsClient(HANDLE h);
    template<class T, class U>
    static bool loadStaticMeshes(HANDLE h, T& distantStatics, U& distantSubsets);
    template<class T, class U>
    static bool loadDistantStaticsClient(T& distantStatics, U& distantSubsets);
    static bool reloadShaders();
    static void release();

    static void editProjectionZ(D3DMATRIX* m, float zn, float zf);
    static bool selectDistantCell();
    static bool isDistantCell();
    static void resolveDynamicVisGroups();
    static void scanDynamicVisGroups();

    static void setView(const D3DMATRIX* m);
    static void setProjection(D3DMATRIX* proj);
    static void setHorizonColour(const RGBVECTOR& c);
    static void setAmbientColour(const RGBVECTOR& c);
    static void setSunLight(const D3DLIGHT8* s);
    static void setScattering(const RGBVECTOR& out, const RGBVECTOR& in);
    static void adjustFog();
    static bool inspectIndexedPrimitive(int sceneCount, const RenderedState* rs, const FragmentState* frs, LightState* lightrs);

    static void beginSkyZone();
    // Called from BeginScene(scene 0): runs the statics-cull prerequisites
    // (selectDistantCell + camera/fog setup) and kicks off the distant-statics
    // cull early, so its ~4ms server-side work overlaps the engine's sky pass.
    // renderStage0 detects the early run and skips the redundant work.
    static void frameSetupEarly();
    static void renderStage0();
    static void beginDrawsZone();
    static void renderStage1();

    static void setupCommonEffect(const D3DXMATRIX* view,const  D3DXMATRIX* proj);


    // In-world debug overlays are compacted onto ONE cycling key (numpad +,
    // VK_ADD). debugOverlayCycle selects which single overlay is active; the
    // per-overlay capture/render bools (boxOccluderDebug, g_drawWaterProxyBounds,
    // g_drawBasinDebug, g_drawMSOCBasinBounds, debugReflFrustum) are DERIVED from
    // it each frame in updateMSOCCutoffInput (early, before the cull worker reads
    // them). 0=off 1=water-proxy 2=box-occluders 3=basin-watershed 4=msoc-basin
    // 5=reflection-frustum+cache. Advanced by updateMSOCCutoffInput.
    static int debugOverlayCycle;
    // S3: overlay cycle state 5 (reflection cull frustum + cache reflection draw set)
    // went with the water reflection it visualised.




    // S4b: the basin (watershed) pre-cull, the MSOC verdict pass, the cull worker, the
    // horizon curtain and the terrain-box occluders all fed one consumer — the distant-
    // statics visible set — and it is gone. The worker had in fact stopped being signalled
    // when the early-Forge-kickoff gate landed, so none of it had run in Forge play since.

    // S3: the water-reflection render (mirrored sky / statics / cache objects / cache
    // terrain, plus their sun-shadow re-draws) went with renderwater.cpp.
    // Phase 1 Milestone 1: draw the simple-opaque subset of the GeometryCache
    // (NPCs + dynamic + near statics, excluding terrain) into the MAIN view with
    // full FFE color, driven authoritatively from the cache walk instead of the
    // engine's reactive per-draw path. Sibling of renderReflectionsFromCache with
    // the main view/proj, documented CW base winding, no water clip plane, and its
    // own z (ZWRITE on, ZFUNC LESSEQUAL). Run from renderStage0 in CACHE mode.
    // S3: clearReflection, the water-visibility gate, the flow-map bake, the reflection
    // screen-rect / silhouette-mask cull, the reflection-statics worker cull, the dynamic
    // wave + foam simulations and renderWaterPlane were all declared here.

    // S5a: MGE's depth pre-pass was declared here — renderDepth, renderDepthAdditional,
    // renderDepthRecorded, renderDepthFromCache, renderCacheDepthToMainZ and the
    // render-thread pair (snapshotVisibleKeysForThread / renderThreadDepthCacheJob).
    // The depth texture they produced had exactly one consumer, the DX9 post chain's
    // SSAO/DOF, which is skipped whenever the Forge seam owns the frame.

    // Build the deterministic current-frame frustum-visible set (s_frustumVisibleKeys)
    // over the full GeometryCache, from the game view*proj. Called early
    // (frameSetupEarly, after the cache walk), or from renderStage0's fallback on frames
    // where frameSetupEarly did not walk. Consumed by the Forge kickoff's draw-list build.
    static void buildFrustumVisibleSet(const D3DXMATRIX* view, const D3DXMATRIX* proj);
    static void updateVisibleSet(void* const* shapes, int count);
    // Stage 2 early classify (main thread, BeginScene(0) via frameSetupEarly, before
    // buildFrustumVisibleSet). Asks the plugin to run the world-camera occlusion
    // classify NOW so the visible/occluded callbacks fire with the current-frame set;
    // latches whether it ran so buildFrustumVisibleSet can drive the cache passes off
    // the engine's exact drawn set (engine-set mode). worldCamera may be null (the
    // plugin resolves it). No-op if the plugin lacks the export or self-declines.
    static void earlyClassifyMainScene(void* worldCamera);
    // The deterministic current-frame frustum-visible set (s_frustumVisibleKeys),
    // built by buildFrustumVisibleSet. Keys = GeometryCache keys.
    static const std::vector<uint32_t>& frustumVisibleKeys();
    // Cut 2B fold: when this frame's buildFrustumVisibleSet DEFERRED its ensureLive
    // loop into the kickoff draw-list build, returns the raw classify-visible keys (s_visibleKeys) for
    // buildGeometryDrawLists to iterate directly — ensureLive + emit in ONE pass.
    // nullptr on non-fold frames (iterate frustumVisibleKeys() as before).
    // CONSUME-ONCE: returns non-null at most once per latched frame (the classify
    // pointers are only valid the frame they were latched).
    static const std::vector<uint32_t>* foldVisibleKeys();
    // Diagnostic: count of cache entries the last buildFrustumVisibleSet dropped via
    // MSOC occlusion refinement (0 when disarmed). For the LogDistantPipeline line.
    static unsigned lastRefineCulled();

    // S3: the whole cascaded sun-shadow renderer (build, per-cascade clear/layer, the
    // recorded-draw receiver overlay, the cache caster pass and the debug view) went with
    // rendershadow.cpp, along with the reflection-statics pipeline diagnostics.

    static void postProcess();
    static void updatePostShader(MGEShader* shader);

    static void requestCapture(std::function<void(IDirect3DSurface9*)> handler, bool captureWithUI);
    static void checkCaptureScreenshot(bool isUIDrawn);
    static IDirect3DSurface9* captureScreenshot();
};

// S5a: RenderTargetSwitcher (scoped SetRenderTarget/SetDepthStencilSurface with restore)
// lived here. Every remaining user was a pass that rendered into one of MGE's own render
// targets; the depth pre-pass was the last of them.
