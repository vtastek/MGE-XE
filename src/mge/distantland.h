#pragma once

#include "quadtree.h"
#include "ffeshader.h"
#include "mwbridge.h"
#include "specificrender.h"
#include "ipc/client.h"
#include "ipc/dlshare.h"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
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

    struct RecordedState : RenderedState {
        RecordedState(const RenderedState&);
        ~RecordedState();
        RecordedState(const RecordedState&) = delete;
        RecordedState(RecordedState&&) noexcept;
    };

    static constexpr DWORD fvfWave = D3DFVF_XYZRHW | D3DFVF_TEX2;
    static constexpr int waveTexResolution = 512;
    static constexpr float waveTexWorldRes = 2.5f;
    static constexpr int GrassInstStride = 48;
    static constexpr int MaxGrassElements = 8192;
    static constexpr float kCellSize = 8192.0f;
    static constexpr float kDistantZBias = 5e-6f;
    static constexpr float kDistantNearPlane = 4.0f;
    static constexpr float kMoonTag = 88888.0f;

    static bool ready;
    static bool isRenderCached;
    static bool isPPLActive;
    // Set by frameSetupEarly() when the GeometryCache walk ran at BeginScene(0);
    // read by renderDepth (different TU) to skip its own redundant walk.
    static bool earlyWalkedCache;
    // Set by frameSetupEarly() when it kicked the render-thread depth-cache job
    // this frame; read by renderStage0() to fence (RenderThread::wait) before any
    // main-thread device/effect work, and by renderDepth() to skip the Clear /
    // float-depth clear / cache pass the worker already wrote into texDepthFrame.
    static bool renderThreadJobKicked;
    static int numWaterVerts, numWaterTris;

    static IDirect3DDevice9* device;
    static ID3DXEffect* effect;
    static ID3DXEffect* effectShadow;
    static ID3DXEffect* effectDepth;
    static ID3DXEffectPool* effectPool;
    static IDirect3DVertexDeclaration9* LandDecl;
    static IDirect3DVertexDeclaration9* StaticDecl;
    static IDirect3DVertexDeclaration9* WaterDecl;
    static IDirect3DVertexDeclaration9* GrassDecl;

    static VendorSpecificRendering vsr;

    static IPC::Client ipcClient;
    static std::vector<DynamicVisGroup> dynamicVisGroups;
    static void* lastDistantVisCell;
    static bool isDistantLandLoaded;

    static VisibleSet<StlVector> visLand;
    static VisibleSet<StlVector> visDistant;
    static VisibleSet<StlVector> visGrass;

    // Cull-then-sort survivor set for distant statics. The server no longer
    // sorts the full visible set; instead applyMSOCToDistantStatics compacts
    // the MSOC survivors (msocOccluded[idx]==0) into a contiguous owned buffer
    // (g_survivorStorage in renderexterior.cpp) and sorts only those (~500 vs
    // ~13k). Both the depth and color static passes iterate this set. Kept
    // separate from visDistant (the non-IPC raw cull output) to avoid
    // overloading its meaning.
    static VisibleSet<StlVector> visDistantSurvivors;

    static VisibleSet<IpcClientVector> visLandShared;
    static VisibleSet<IpcClientVector> visDistantShared;
    static VisibleSet<IpcClientVector> visGrassShared;
    static VisibleSet<IpcClientVector> visExtraShared;
    static IPC::VecView<IPC::DynVisFlag> dynVisFlagsShared;

    static IPC::VecId visLandSharedId;
    static IPC::VecId visDistantSharedId;
    static IPC::VecId visGrassSharedId;
    static IPC::VecId visExtraSharedId;
    static IPC::VecId dynVisFlagsSharedId;

    static std::vector<RecordedState> recordMW;
    static std::vector<RecordedState> recordSky;
    static std::vector< std::pair<const RenderMesh*, int> > batchedGrass;

    // CPU-side copy of each distant-land tile's triangle mesh, captured
    // during initLandscape before the VB/IB Unlocks. Used by
    // contributeDistantLandOccluders to feed the plugin's MSOC mask the
    // real terrain surface — sampled subsets (regular grids etc.) fight
    // ROAM's adaptive tessellation and produce poor silhouettes, so we
    // store the full mesh. Cost: ~20 MB extra RAM per worldspace, well
    // within budget for a 32-bit process with a 2-4 GB heap.
    //
    // Keyed by the tile's VB pointer (the vBuffer field each RenderMesh
    // in visLand / visLandShared carries).
    //
    // Indices are stored uniformly as uint32 regardless of the on-disk
    // format (16-bit for small tiles, 32-bit for large) so the runtime
    // rebase path doesn't have to branch.
    struct LandMeshCache {
        std::vector<D3DXVECTOR3>   positions;   // POSITION float3 only; UVs discarded
        std::vector<std::uint32_t> indices;     // promoted to uint32 uniformly
    };
    static std::unordered_map<IDirect3DVertexBuffer9*, LandMeshCache> landMeshes;

    static IDirect3DTexture9* texWorldColour, *texWorldNormals, *texWorldDetail;
    static IDirect3DTexture9* texDepthFrame;
    static IDirect3DSurface9* surfDepthDepth;
    static IDirect3DTexture9* texDistantBlend;
    static IDirect3DTexture9* texReflection;
    static IDirect3DSurface9* surfReflectionZ;
    static IDirect3DVolumeTexture9* texWater;
    static IDirect3DVertexBuffer9* vbWater;
    static IDirect3DIndexBuffer9* ibWater;
    static IDirect3DVertexBuffer9* vbGrassInstances;

    static IDirect3DTexture9* texRain, *texRipples, *texRippleBuffer;
    static IDirect3DSurface9* surfRain, *surfRipples, *surfRippleBuffer;
    static IDirect3DVertexBuffer9* vbWaveSim;

    static IDirect3DTexture9* texShadow, *texSoftShadow;
    static IDirect3DSurface9* surfShadowZ;
    static IDirect3DVertexBuffer9* vbFullFrame, *vbClipCube;

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
    static D3DXHANDLE ehWindVec;
    static D3DXHANDLE ehNiceWeather;
    static D3DXHANDLE ehTime;
    static D3DXHANDLE ehRippleOrigin;
    static D3DXHANDLE ehWaveHeight;

    static std::function<void(IDirect3DSurface9*)> captureScreenHandler;
    static bool captureScreenWithUI;

    static bool init();
    static bool initIpc();
    static bool initShader();
    static bool initDepth();
    static bool initWater();
    static bool initDynamicWaves();
    static bool initLandscapeClient();
    static bool initLandscape();
    static bool initDistantStaticsClient();
    static bool initShadow();
    static bool initGrass();
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

    static void renderSky();
    static void beginSkyZone();
    // Called from BeginScene(scene 0): runs the statics-cull prerequisites
    // (selectDistantCell + camera/fog setup) and kicks off the distant-statics
    // cull early, so its ~4ms server-side work overlaps the engine's sky pass.
    // renderStage0 detects the early run and skips the redundant work.
    static void frameSetupEarly();
    static void renderStage0();
    static void beginDrawsZone();
    static void renderStage1();
    static void renderStage2();
    static void renderStageBlend();
    static void renderStageWater();

    static void setupCommonEffect(const D3DXMATRIX* view,const  D3DXMATRIX* proj);

    static void renderDistantLand(ID3DXEffect* e, const D3DXMATRIX* view, const D3DXMATRIX* proj);
    static void renderDistantLandZ();
    // Build a horizon-curtain occluder from the visible distant-land
    // tiles and feed it to the plugin's MSOC mask via the pre-
    // transformed occluder ABI. Without this, the upper half of the
    // mask is empty and giants at the skyline never cull. Called from
    // renderStage0 after visLand is materialized; submissions land in
    // the next frame's mask (one-frame latency by design).
    static void contributeDistantLandOccluders();

    // Free the horizon-curtain workspace (the lazily-allocated state in
    // renderexterior.cpp). Called from release() so the malloc'd buffers
    // don't leak across renderer init/release cycles.
    static void shutdownHorizonWorkspace();
    // Two-phase culling so the IPC server's quadtree work overlaps with
    // the rest of the frame instead of blocking the main thread.
    //
    //   _kickoff issues the batched 3-range visibility RPC (or the
    //   equivalent synchronous quadtree query in the non-IPC path) and
    //   returns immediately. Must be called once mwView is finalized.
    //
    //   _finish blocks until the visible set is populated and then runs
    //   applyMSOCToDistantStatics over it. Must be called before any
    //   consumer of the visible set (renderDepth:statics,
    //   renderDistantStatics, water-reflection statics).
    static void cullDistantStatics_kickoff(const D3DXMATRIX* view, const D3DXMATRIX* proj);
    static void cullDistantStatics_finish();

    // --- Dedicated MSOC cull worker -------------------------------------
    // The distant-statics verdict pass (drain the statics RPC + the
    // partition/OBB/sphere/propagate core in applyMSOCToDistantStatics) is
    // pure compute that depends only on an already-ready IPC result. On the
    // IPC + early-kickoff path it is dispatched to a dedicated worker by
    // frameSetupEarly so it overlaps the engine's sky pass instead of
    // stalling cullDistantStatics_finish on the main critical path.
    //
    //   updateMSOCCutoffInput  — read the Numpad8/2 live cutoff (main thread)
    //                            so g_msocCutoffHeight is final before the
    //                            worker reads it.
    //   signalCullFinish       — dispatch the verdict pass to the worker.
    //   waitCullChannelFree    — block until the worker has drained the
    //                            statics RPC off the single IPC channel;
    //                            called at renderStage0 entry so no main-
    //                            thread ipcClient call races the drain.
    //   joinCullWorker         — tear the worker thread down (release()).
    static void updateMSOCCutoffInput();
    static void signalCullFinish();
    static void waitCullChannelFree();
    static void joinCullWorker();
    static void renderDistantStatics();
    static void renderMSOCBasinBoundsDebug(const D3DXMATRIX* view, const D3DXMATRIX* proj);
    static void renderWaterProxyBoundsDebug(const D3DXMATRIX* view, const D3DXMATRIX* proj);

    // MSOC occlusion verdict pass — walks the visible set, runs the
    // batched sphere query, applies far/handoff gates and temporal
    // hysteresis, fills msocOccluded with a per-instance cull mask
    // (1 = cull, 0 = render). Both the instanced and non-instanced
    // render paths consume this mask, so MSOC works the same way
    // regardless of which rendering path is active.
    template<class T>
    static void applyMSOCToDistantStatics(VisibleSet<T>& staticSet);

    // Per-instance MSOC verdict, indexed in lockstep with the visible
    // set's iteration order. Sized = visible set size, filled by
    // applyMSOCToDistantStatics. 1 = cull this instance, 0 = render.
    // Empty when MSOC is unavailable / occlusion disabled.
    static std::vector<std::uint8_t> msocOccluded;
    static void cullGrass(const D3DXMATRIX* view, const D3DXMATRIX* proj);
    template<class T>
    static void buildGrassInstanceVB(VisibleSet<T>& grassSet);
    static bool hasVisibleGrass();
    static void renderGrassInst();
    static void renderGrassInstZ();
    static void renderGrassCommon(ID3DXEffect* e);

    static void renderWaterReflection(const D3DXMATRIX* view, const D3DXMATRIX* proj);
    static void renderReflectedSky();
    static void renderReflectedStatics(const D3DXMATRIX* view, const D3DXMATRIX* proj);
    static void clearReflection();
    // Water-reflection occlusion gate: true if any water surface is actually
    // visible in the main view (terrain height + MSOC), so the ~1ms reflection
    // pass can be skipped when water is fully occluded / out of frame. See
    // tasks/todo.md Phase A. Side effect: fills reflectionWaterRects with the
    // surviving water tiles' main-view NDC screen rects, consumed by
    // renderReflectedStatics to cull reflection statics (Phase B).
    static bool isReflectionWaterVisible();
    // Surviving water tile screen rects (main-view NDC AABBs: x=minX, y=minY,
    // z=maxX, w=maxY). Where visible water samples texReflection on screen.
    static std::vector<D3DXVECTOR4> reflectionWaterRects;

    // --- Reflection-statics cull dispatched to the MSOC cull worker ---
    // The gate (isReflectionWaterVisible), the reflection IPC query and the
    // per-static skipMask are all pure-CPU/IPC and frame-stable, so they run on
    // the cull worker during the sky window, leaving only the reflection draw on
    // the main thread. See tasks/todo.md.
    static bool reflGateWanted;     // worker runs the water-visible gate this frame
    static bool reflStaticsWanted;  // worker issues the reflection RPC + culls
    static bool reflVisible;        // gate result: any water visible (worker-written)
    static D3DXMATRIX  reflCullViewProj;   // reflection view*proj (RPC frustum + cull projection)
    static D3DXMATRIX  reflCullProj;       // reflection proj (NDC radius scale)
    static D3DXVECTOR4 reflCullViewSphere; // reflection cull sphere (eye + range)
    // Compacted reflection-static survivors: pointers into a stable copy of the
    // IPC visible set, so the MAIN thread draws without traversing the live IPC
    // window (which the worker + main RPCs would race). Mirrors visDistantSurvivors.
    static VisibleSet<StlVector> reflectionSurvivors;
    // Main (frameSetupEarly): stash the reflection cull inputs for the worker.
    static void prepareReflectionCullForWorker();
    // Worker: issue the reflection RPC + materialize the result (before
    // channelDrained), then after the verdict run the gate + cull to survivors.
    static void workerReflectionRPC();
    static void workerReflectionGateAndMask();
    // Materialize visExtraShared into stable storage (one IPC-window traversal).
    static void materializeReflectionMeshes();
    // Cull the materialized reflection meshes into reflectionSurvivors using
    // reflectionWaterRects. Shared by the worker and the non-worker fallback.
    static void cullReflectionSurvivors(const D3DXMATRIX& viewProj, const D3DXMATRIX& proj);
    static void simulateDynamicWaves();
    static void renderWaterPlane();

    static void renderDepth();
    static void renderDepthAdditional();
    static void renderDepthRecorded();
    // visibleOverride: when non-null, iterate this key list instead of the live
    // s_prevVisibleKeys set. Used by the render-thread job, which reads a
    // main-thread snapshot (snapshotVisibleKeysForThread) so it never touches the
    // set concurrently with updateVisibleSet.
    static void renderDepthFromCache(const D3DXMATRIX* gameView,
                                     const std::vector<uint32_t>* visibleOverride = nullptr);
    // Copy s_prevVisibleKeys into the render-thread snapshot. Main-thread only,
    // called at kick (frameSetupEarly) before the job can read it.
    static void snapshotVisibleKeysForThread();
    // Render-thread depth-cache job: under the device lock, save full device
    // state (engine is mid-sky), bind texDepthFrame/surfDepthDepth, Clear +
    // float-depth clear pass + renderDepthFromCache (its own effectDepth
    // bracket), restore RT + device state. Produces the same depth content
    // renderDepth's serial cache path would, just during the sky window.
    static void renderThreadDepthCacheJob();
    static void updateVisibleSet(void* const* shapes, int count);

    static void renderShadowMap();
    static void renderShadowFromCache(int layer, const D3DXMATRIX* viewproj);
    // Clears one cascade's region of the shadow atlas (depth + stencil
    // + the float-encoded "far depth" sentinel). Viewport-clipped so it
    // touches only [layer*res, 0, res, res] of the atlas — the other
    // cascade's region is preserved. Used by the adaptive shadow
    // scheduler to refresh only the cascade(s) being re-rendered.
    static void clearShadowCascade(int layer);
    template<class T>
    static void renderShadowLayerGeneric(MWBridge* mwBridge, int layer, const D3DXMATRIX* inverseCameraProj, D3DXMATRIX* view, D3DXMATRIX* proj, VisibleSet<T>& visible_set);
    static void renderShadowLayer(int layer, float radius, const D3DXMATRIX* inverseCameraProj);
    static void renderShadow();
    static void renderShadowDebug();

    static void postProcess();
    static void updatePostShader(MGEShader* shader);

    static void requestCapture(std::function<void(IDirect3DSurface9*)> handler, bool captureWithUI);
    static void checkCaptureScreenshot(bool isUIDrawn);
    static IDirect3DSurface9* captureScreenshot();
};

class RenderTargetSwitcher {
    IDirect3DSurface9* savedTarget, *savedDepthStencil;
    void init(IDirect3DSurface9* target, IDirect3DSurface9* targetDepthStencil);

public:
    RenderTargetSwitcher(IDirect3DSurface9* target, IDirect3DSurface9* targetDepthStencil);
    RenderTargetSwitcher(IDirect3DTexture9* targetTex, IDirect3DSurface9* targetDepthStencil);
    ~RenderTargetSwitcher();
};
