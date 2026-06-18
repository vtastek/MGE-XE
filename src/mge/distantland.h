#pragma once

#include "quadtree.h"
#include "ffeshader.h"
#include "mwbridge.h"
#include "specificrender.h"
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

    struct RecordedState : RenderedState {
        RecordedState(const RenderedState&);
        ~RecordedState();
        RecordedState(const RecordedState&) = delete;
        RecordedState(RecordedState&&) noexcept;
    };

    static constexpr DWORD fvfWave = D3DFVF_XYZRHW | D3DFVF_TEX2;
    static constexpr int waveTexResolution = 512;
    static constexpr float waveTexWorldRes = 2.5f;
    // World-anchored hybrid particle foam sim (WATER_FOAM). Two cascades, each a full
    // world-anchored 1024 sim at a different world-scale: cascade 0 (fine/near) and
    // cascade 1 (coarse/far). The water shader samples the fine cascade near the camera
    // and blends to the coarse one at cascade 0's window edge. Resolution is shared by
    // all cascades (the foam passes reuse one fullscreen triangle, vbFoamSim).
    // Single low-res CARRIER buffer (the cascades collapsed to one — Phase 2 of the indirection
    // redesign). The carrier marks WHERE foam is (vorticity); the procedural detail in the water
    // shader supplies the up-close crispness, so the sim itself can stay cheap and coarse.
    // foamCascades kept = 1 so the world-anchor loop/arrays carry over unchanged (one iteration).
    static constexpr int foamCascades = 1;
    static constexpr int foamTexResolution = 512;     // half-size sim: ~2MB/RT × 6 RTs ≈ 12MB VRAM (¼ the fill of 1024)
    static constexpr float foamCascadeWorldRes[foamCascades] = { 25.0f };    // 512 * 25 = 12800u window (kept; texel doubled)
    static constexpr int GrassInstStride = 48;
    static constexpr int MaxGrassElements = 8192;
    static constexpr float kCellSize = 8192.0f;
    static constexpr float kDistantZBias = 5e-6f;
    static constexpr float kDistantNearPlane = 4.0f;
    static constexpr float kMoonTag = 88888.0f;

    static bool ready;
    static bool isRenderCached;
    static bool isPPLActive;
    // Phase 1 Milestone 1 A/B toggle (NUMPAD7). false = ENGINE (untouched
    // reactive path). true = CACHE (renderCachedOpaque draws the simple-opaque
    // subset authoritatively from the GeometryCache walk in renderStage0, and
    // inspectIndexedPrimitive suppresses the engine's covered opaque draws).
    // Read once per frame at renderStage0 entry so inspectIndexedPrimitive sees
    // a stable value for the whole frame.
    static bool cacheOpaqueMode;
    // NUMPAD7 third state (CACHE-ONLY): when true, cacheOpaqueMode is also true, but
    // inspectIndexedPrimitive additionally suppresses EVERY other colour draw across ALL
    // scenes (reactive PPL, the fixed-function fallback, first-person hands, alpha-sorted,
    // UI) so only what renderCachedOpaque/renderCachedTerrain produce is visible. A
    // diagnostic to read cache coverage at a glance: anything not owned by the cache goes
    // black. (sky stage is separate from inspectIndexedPrimitive, so it still backdrops.)
    static bool cacheOnlyMode;
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
    // Single-window chunk vec carrying the occlusion-mask blob shipped to the
    // host each frame (host-side cull). Empty/InvalidVector when disabled.
    static IPC::VecView<OcclusionMask::MaskChunk> maskBlobShared;

    static IPC::VecId visLandSharedId;
    static IPC::VecId visDistantSharedId;
    static IPC::VecId visGrassSharedId;
    static IPC::VecId visExtraSharedId;
    static IPC::VecId dynVisFlagsSharedId;
    static IPC::VecId maskBlobSharedId;

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

    // World-snapped nested-grid (geo-clipmap) water mesh. Built once when
    // UseWaterFlowMap is on (initWaterLodMesh); drawn per-level in
    // renderWaterPlane with a per-level snapped world matrix so vertices land on
    // a stable world lattice every frame (kills the radial mesh's swimming).
    // Vertices are stored in local integer grid units (cell size supplied by the
    // per-frame world matrix); each level records its index range and cell size.
    struct WaterLodLevel {
        float cellSize;     // world units per grid cell at this level
        int   vertBase;     // first vertex of this level's block in vbWaterLod
        int   vertCount;    // vertices owned by this level
        int   numVariants;  // flexible-trim hole variants (1 for solid level 0, else 4)
        int   ibStart[4];   // first index of each trim variant's triangle list
        int   triCount[4];  // triangles in each trim variant
    };
    static IDirect3DVertexBuffer9* vbWaterLod;
    static IDirect3DIndexBuffer9* ibWaterLod;
    static int numWaterLodVerts;
    static std::vector<WaterLodLevel> waterLodLevels;
    static bool waterLodMeshOn;        // runtime A/B (VK_NUMPAD0): true = clipmap, false = radial
    static float waterWaveAmp;         // live shader uniform: Gerstner crest amplitude (world units)
    static float waterWaveLen;         // live shader uniform: base wavelength along the flow (world units)
    static float waterWaveSpeed;       // live shader uniform: crest travel speed scale
    static float waterCrestSpread;     // live shader uniform: directional fan (small = longer crest lines)

    static IDirect3DTexture9* texRain, *texRipples, *texRippleBuffer;
    static IDirect3DSurface9* surfRain, *surfRipples, *surfRippleBuffer;
    static IDirect3DVertexBuffer9* vbWaveSim;
    static IDirect3DVertexBuffer9* vbFoamSim;   // fullscreen triangle sized to foamTexResolution

    // Water flow map (UseWaterFlowMap): low-res baked RGBA8 covering the exterior
    // island. R,G = downstream flow dir (encoded), B = wave intensity, A =
    // directionality. Built once on the cull worker (buildWaterFlowMap), lazily
    // uploaded on the main thread (updateFlowMapTexture). VK_NUMPAD9 A/B.
    static IDirect3DTexture9* texFlow;
    static bool waterFlowDebugOn;
    static int  waterFlowDebugView; // 0 none; >0 = debug overlay id passed to the shader (2..11)
    static float waterFlowScroll;   // live shader uniform: river directional advection rate (NUMPAD8/6/3 tuning)
    static float waterFlowSeaSpeed; // live shader uniform: base wave animation rate scale (1 = stock)
    static float waterFlowCycleUV;  // live shader uniform: bounded per-cycle UV displacement (Valve flow map)
    static float waterFlowSeaRefract; // live shader uniform: sea far-wave (refraction) strength multiplier
    static float waterFlowWarp;       // live shader uniform: domain-warp amount (world u) to break the 512u flow grid

    // World-anchored hybrid Voronoi particle foam (WATER_FOAM, gated by UseWaterFlowMap).
    // texFoamP_A/B: ping-pong particle buffer (xy = window texel pos, zw = velocity).
    // texFoamField: smoothed velocity + density field (the vorticity source).
    // texFoam: extracted foam intensity sampled by the water shader. fp16 throughout.
    // Per-cascade arrays (foamCascades): cascade 0 = fine/near, cascade 1 = coarse/far.
    static IDirect3DTexture9* texFoamP_A[foamCascades], *texFoamP_B[foamCascades], *texFoamField[foamCascades], *texFoam[foamCascades];
    static IDirect3DSurface9* surfFoamP_A[foamCascades], *surfFoamP_B[foamCascades], *surfFoamField[foamCascades], *surfFoam[foamCascades];
    // Advected detail-UV offset field (River Editor): per-texel world-space offset transported
    // through the velocity field so the consume's foam texture follows curved river flow. Ping-pong.
    static IDirect3DTexture9* texFoamUV_A[foamCascades], *texFoamUV_B[foamCascades];
    static IDirect3DSurface9* surfFoamUV_A[foamCascades], *surfFoamUV_B[foamCascades];
    static int   foamLastXpos[foamCascades], foamLastYpos[foamCascades];   // per-cascade world-anchor window tracking (texels)
    static float foamOriginC[foamCascades][2];   // per-cascade window world min-corner (saved for the consume bind)
    static bool foamSimReset;          // clear/seed the particle buffers on the next sim step
    static bool waterFoamOn;           // runtime A/B: true = foam sim + render, false = legacy water
    static bool foamDebugView;         // NUMPAD2: blit raw foam particle/field buffers to screen corner
    // Per-cascade live tuning (fine/near = [0], coarse/far = [1]). The two cascades have
    // very different texel sizes, so vorticity/density respond differently to the same
    // values — they are tuned independently to match the two foam looks.
    static float foamFlowForce[foamCascades];   // live: river advection force (texels/substep)
    static float foamDecay[foamCascades];       // live: velocity decay toward the flow current
    static float foamPressure[foamCascades];    // live: density pile-up coefficient (narrows)
    static float foamScale[foamCascades];       // live: vorticity → foam intensity scale
    // Two-layer foam (Phase 2): erosion + ridged-fbm shared by the far (flow-map) and near
    // (sim carrier) layers. Live-tuned via the NUMPAD8 cycle.
    static float foamDetailTile;       // world units per fbm cell (smaller = crisper)
    static float foamDetailSpeed;      // far-layer flow advect rate (also the sim UV-advect rate)
    static float foamErodeThreshold;   // erosion cut: higher = tighter foam streaks [0,1]
    static float foamFarAmount;        // far flow-map layer strength (0 = near-only)
    static float foamMix;              // near/sim → far modulation in the carrier window (foamDetail.w); 0 = far only
    static float foamGaussRadius;      // sim particle splat radius → foam blob size (FoamGauss)
    static float foamMinDensity;       // sim particle respawn density (FoamDens)
    static float foamUVDecay;          // sim UV-offset decay → bounds detail stretch (FoamUVDcy)
    static float foamSimSpeed;         // scales sim particle advance rate → sim-foam visual speed (SimSpeed)
    static float foamVortGain;         // static vorticity (curl) concentration for the far foam (VortGain)
    static float foamFineScale;        // multi-scale erosion: perforating octave scale ratio (FineScale)
    static float foamFineAmt;          // multi-scale erosion: fine-perforation strength (FineAmt)
    static float foamCoarseScale;      // multi-scale erosion: clumping octave scale ratio (CoarseScale)
    static float foamCoarseAmt;        // multi-scale erosion: coarse-clumping strength (CoarseAmt)

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
    static D3DXHANDLE ehStaticNearCull;
    static D3DXHANDLE ehShadowReflMult;
    static D3DXHANDLE ehLandNearCull;
    static D3DXHANDLE ehReflWaterClip;
    // Texture-light path handles on the distant-land effect (cache terrain point
    // lights). Bound per-patch from selectTextureLights output in renderCachedTerrain.
    static D3DXHANDLE ehLightData, ehLightDataParams, ehLightIndices, ehTexLightView;
    static D3DXHANDLE ehWindVec;
    static D3DXHANDLE ehNiceWeather;
    static D3DXHANDLE ehTime;
    static D3DXHANDLE ehRippleOrigin;
    static D3DXHANDLE ehWaveHeight;
    static D3DXHANDLE ehFlow, ehFlowTransform, ehFlowWeight, ehFlowScroll, ehFlowSeaSpeed, ehFlowCycleUV, ehFlowSeaRefract, ehFlowDebugView, ehFlowWarp;
    static D3DXHANDLE ehWaveAmp, ehWaveLen, ehWaveSpeed, ehCrestSpread;
    static D3DXHANDLE ehFoamParticles, ehFoamFieldIn, ehFoamOrigin, ehFoamShift, ehFoamFieldShift, ehFoamPlayer, ehFoamParams, ehFoamWorldRes, ehFoamAdvance;
    static D3DXHANDLE ehFoam0, ehFoamOrigin0, ehFoamWeight, ehFoamDetail, ehFoamFarAmount;   // ehFoam0/Origin0 = the single carrier
    static D3DXHANDLE ehFoamUVIn, ehFoamUVRate, ehFoamUVDecay;   // UV-advection sim uniforms
    static D3DXHANDLE ehFoamGaussRadius, ehFoamMinDensity;       // sim look knobs (blob size, particle density)
    static D3DXHANDLE ehFoamVortGain;                            // static vorticity concentration (far foam)
    static D3DXHANDLE ehFoamFineScale, ehFoamFineAmt;            // multi-scale foam erosion
    static D3DXHANDLE ehFoamCoarseScale, ehFoamCoarseAmt;       // multi-scale foam erosion (coarse clumping)
    static D3DXHANDLE ehFoamUVTex;                               // consume: advected UV offset field

    static std::function<void(IDirect3DSurface9*)> captureScreenHandler;
    static bool captureScreenWithUI;

    static bool init();
    static bool initIpc();
    static bool initShader();
    static bool initDepth();
    static bool initWater();
    static bool initWaterLodMesh();
    static bool initDynamicWaves();
    static bool initFoamSim();
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
    static void contributeDistantLandOccluders(bool captureDebug = false);
    // Voxelize the min-height terrain into merged boxes and feed them to MSOC as
    // real 3D occluders. Runs on the cull worker every frame (off the main thread)
    // using a frame-stable view-proj. captureDebug stashes the AABBs for the
    // Numpad3 in-world overlay.
    static void contributeTerrainBoxOccluders(const D3DXMATRIX& viewProj, bool captureDebug = false);
    // Numpad3: draw the terrain-box occluder overlay (capture is otherwise off).
    static bool boxOccluderDebug;

    // In-world debug overlays are compacted onto ONE cycling key (numpad +,
    // VK_ADD). debugOverlayCycle selects which single overlay is active; the
    // per-overlay capture/render bools (boxOccluderDebug, g_drawWaterProxyBounds,
    // g_drawBasinDebug, g_drawMSOCBasinBounds, debugReflFrustum) are DERIVED from
    // it each frame in updateMSOCCutoffInput (early, before the cull worker reads
    // them). 0=off 1=water-proxy 2=box-occluders 3=basin-watershed 4=msoc-basin
    // 5=reflection-frustum+cache. Advanced by updateMSOCCutoffInput.
    static int debugOverlayCycle;
    // Cycle state 5: draw the reflection cull frustum + the GeometryCache
    // reflection draw set as boxes, RED = drawn now, GREEN = its mirrored sphere
    // lands on a visible water rect (would survive a water-rect cull — the fix
    // preview). debugReflFrustum gates capture in renderReflectionsFromCache.
    static bool debugReflFrustum;
    struct ReflCacheDbgBox { D3DXVECTOR3 center; float radius; bool keep; };
    static std::vector<ReflCacheDbgBox> reflCacheDbg;
    // The reflection render view*proj, stashed by renderReflectionsFromCache when
    // the overlay is active, so renderReflectionFrustumDebug (drawn in the MAIN
    // view) can invert it to wireframe the reflection camera frustum. reflDbgValid
    // is false when no water reflection ran this frame (overlay then draws nothing).
    static D3DXMATRIX reflDbgViewProj;
    static bool reflDbgValid;
    static void renderReflectionFrustumDebug(const D3DXMATRIX* view, const D3DXMATRIX* proj);

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
    //   waitCullReflReady      — block until the worker has finished the
    //                            reflection gate + survivor cull (late fence,
    //                            joined just before renderWaterReflection).
    static void waitCullReflReady();
    static void joinCullWorker();
    static void renderDistantStatics();
    static void renderMSOCBasinBoundsDebug(const D3DXMATRIX* view, const D3DXMATRIX* proj);
    static void debugDumpMSOCMask();   // Numpad5 mask dump, post-curtain
    static void renderCurtainDebug();  // Numpad3 cycle: in-world curtain overlay
    static void renderBoxOccluderDebug(const D3DXMATRIX* view, const D3DXMATRIX* proj);
    static void renderBasinDebug(const D3DXMATRIX* view, const D3DXMATRIX* proj);
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

    // --- Basin (watershed) occlusion pre-cull for distant statics ---
    // View-direction-independent terrain pre-cull run ahead of MSOC on the
    // cull worker. buildBasinRequiredHeight floods a minimax (watershed)
    // spill-height surface out from the camera's coarse cell — the lowest
    // full-cell wall (per-cell terrain MIN, so it's a provable barrier) that
    // must be cleared to reach each cell. The MSOC pass's middle stage then marks
    // any static whose OBB top sits below that wall (dense O(1) per static).
    // The provably conservative test is req > max(eyeZ, top): a low camera
    // collapses to the static top (aggressive discard), an elevated camera
    // tightens (ridge must clear the eye), so a visible static is never hidden.
    // See tasks/todo.md (basin pre-cull). The per-static verdict runs inside the
    // MSOC pass (group cheap-reject → basin → sphere), so only the flood builder
    // is exposed here.
    static void buildBasinRequiredHeight();
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
    // Phase 0.5: inject the GeometryCache dynamic set (NPCs + dynamic statics)
    // into the water reflection with full color via FixedFunctionShader::
    // renderMorrowind, driven from the cache walk (not engine draws). Additive —
    // no engine suppression. Non-skinned opaque parts only in 0.5-B.
    static void renderReflectionsFromCache(const D3DXMATRIX* view, const D3DXMATRIX* proj, float nearDist);
    // Phase 1 Milestone 1: draw the simple-opaque subset of the GeometryCache
    // (NPCs + dynamic + near statics, excluding terrain) into the MAIN view with
    // full FFE color, driven authoritatively from the cache walk instead of the
    // engine's reactive per-draw path. Sibling of renderReflectionsFromCache with
    // the main view/proj, documented CW base winding, no water clip plane, and its
    // own z (ZWRITE on, ZFUNC LESSEQUAL). Run from renderStage0 in CACHE mode.
    static void renderCachedOpaque(const D3DXMATRIX* view, const D3DXMATRIX* proj);
    // Phase 0.5: apply sun shadows to the cache reflection objects by re-drawing
    // them with the shadow-receiver shader. The sun shadow map is world-space, so
    // reflected geometry samples it correctly via shadowViewProj = inverse(reflView)
    // * smViewproj. Non-skinned only for now (skinned receiver needs skinIndexed).
    static void renderReflectionShadowsFromCache(const D3DXMATRIX* view, const D3DXMATRIX* proj, float nearDist);
    // Phase 0.5: draw the real near terrain from the cache into the reflection
    // (two-texture AlphaGrid splat), replacing the coarse distant-land LOD inside
    // nearDist. The DL land pass is near-clipped (landNearCull) to hand off.
    static void renderReflectionTerrainFromCache(const D3DXMATRIX* view, const D3DXMATRIX* proj, float nearDist);
    // Phase 1 Milestone 2.1: main-view sibling of renderReflectionTerrainFromCache.
    // Draws the real near terrain (the engine's submitted landscape, two-texture
    // AlphaGrid splat) from the cache into the MAIN view, so MGE owns ALL opaque
    // (objects + terrain) in CACHE mode. No near-dist handover (the cache IS the
    // engine's near terrain; DL LOD owns beyond via the distant projection), no
    // water clip, main-view CW winding. Run from renderStage0 after renderCachedOpaque.
    static void renderCachedTerrain(const D3DXMATRIX* view, const D3DXMATRIX* proj);
    static void clearReflection();
    // Water-reflection occlusion gate: true if any water surface is actually
    // visible in the main view (terrain height + MSOC), so the ~1ms reflection
    // pass can be skipped when water is fully occluded / out of frame. See
    // tasks/todo.md Phase A. Side effect: fills reflectionWaterRects with the
    // surviving water tiles' main-view NDC screen rects, consumed by
    // renderReflectedStatics to cull reflection statics (Phase B).
    static bool isReflectionWaterVisible();

    // Water flow map. buildWaterFlowMap bakes the per-body flow field on the cull
    // worker (CPU only). updateFlowMapTexture (main thread) lazily (re)creates and
    // uploads texFlow when the worker marks it dirty; returns true if texFlow is
    // valid to bind. getFlowMapTransform fills {origin.x, origin.y, invSizeX, invSizeY}.
    static void buildWaterFlowMap(float waterZ);
    static bool updateFlowMapTexture();
    static void getFlowMapTransform(float out[4]);

    // Surviving water tile screen rects (main-view NDC AABBs: x=minX, y=minY,
    // z=maxX, w=maxY). Where visible water samples texReflection on screen.
    static std::vector<D3DXVECTOR4> reflectionWaterRects;
    // Whether the screen-space water cull (rects + silhouette mask) is reliable this
    // frame. FALSE → both consumers (cullReflectionSurvivors, renderReflectionsFromCache)
    // keep ALL reflection candidates. Set false on interior / no terrain data, and while
    // SWIMMING (eye on the water plane → water projects edge-on → rects AND mask collapse
    // to slivers and over-cull). Written by isReflectionWaterVisible.
    static bool reflWaterCullActive;

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
    // The reflection RPC is folded into the batched statics RPC (kickoff), so
    // there is no separate worker RPC step — the worker materializes the folded
    // result after the single drain, then runs the gate + cull to survivors.
    static void prepareReflectionCullForWorker();
    static void workerReflectionGateAndMask();
    // Materialize visExtraShared into stable storage (one IPC-window traversal).
    static void materializeReflectionMeshes();
    // Cull the materialized reflection meshes into reflectionSurvivors using
    // reflectionWaterRects. Shared by the worker and the non-worker fallback.
    static void cullReflectionSurvivors(const D3DXMATRIX& viewProj, const D3DXMATRIX& proj);
    // Stage-2 reflection cull: fine water-silhouette mask test. Given a footprint's
    // main-view NDC AABB (centre nx,ny ± half-extents rx,ry), returns true (keep) if
    // it overlaps any rasterized visible-water bit. Returns true unconditionally when
    // the mask is invalid (interior / no terrain data — mirrors the empty-rects keep).
    // Water lies on the mirror plane, so a reflection-projected footprint shares the
    // main-cam NDC the mask was built in. Built by isReflectionWaterVisible, consumed
    // by cullReflectionSurvivors (statics) and renderReflectionsFromCache (cache).
    static bool reflWaterMaskTestNDC(float nx, float ny, float rx, float ry);
    // Debug: number of set bits in the current water-silhouette mask (0 if invalid).
    static int reflWaterMaskSetBits();
    static void simulateDynamicWaves();
    // World-anchored hybrid particle foam sim (WATER_FOAM). Runs in the same effect
    // Begin bracket as simulateDynamicWaves (reuses vbWaveSim + WaveVS), gated by
    // UseWaterFlowMap && waterFoamOn. Tracks the player with a texel-aligned StretchRect
    // shift (world-locked foam), advects Voronoi particles along the flow map, and
    // writes foam intensity into texFoam for the water shader.
    static void simulateFoam();
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
    // Build the deterministic current-frame frustum-visible set (s_frustumVisibleKeys)
    // over the full GeometryCache, from the game view*proj. Called early
    // (frameSetupEarly, after the cache walk) on the IPC path and at the renderDepth
    // walk site on the non-IPC path. The cache depth + opaque color passes drive off
    // this set instead of the frame-lagged engine-MSOC verdict. Frustum-only (Phase 1).
    static void buildFrustumVisibleSet(const D3DXMATRIX* view, const D3DXMATRIX* proj);
    // Copy s_frustumVisibleKeys into the render-thread snapshot. Main-thread only,
    // called at kick (frameSetupEarly) before the job can read it.
    static void snapshotVisibleKeysForThread();
    // Render-thread depth-cache job: under the device lock, save full device
    // state (engine is mid-sky), bind texDepthFrame/surfDepthDepth, Clear +
    // float-depth clear pass + renderDepthFromCache (its own effectDepth
    // bracket), restore RT + device state. Produces the same depth content
    // renderDepth's serial cache path would, just during the sky window.
    static void renderThreadDepthCacheJob();
    static void updateVisibleSet(void* const* shapes, int count);
    // Stage 2 early classify (main thread, BeginScene(0) via frameSetupEarly, before
    // buildFrustumVisibleSet). Asks the plugin to run the world-camera occlusion
    // classify NOW so the visible/occluded callbacks fire with the current-frame set;
    // latches whether it ran so buildFrustumVisibleSet can drive the cache passes off
    // the engine's exact drawn set (engine-set mode). worldCamera may be null (the
    // plugin resolves it). No-op if the plugin lacks the export or self-declines.
    static void earlyClassifyMainScene(void* worldCamera);
    // The main-view MSOC visible set (s_prevVisibleKeys), keyed on
    // NiTriBasedGeometry* == GeometryCache keys. No longer drives the cache
    // depth/opaque paths (they consume the early frustum set, buildFrustumVisibleSet)
    // — kept for the distant-statics path and as the foundation for Phase 3 (an
    // early MGE-driven MSOC mask over the cache, replacing this lagged verdict).
    static const std::unordered_set<uint32_t>& visibleCacheKeys();
    // The deterministic current-frame frustum-visible set (s_frustumVisibleKeys),
    // built by buildFrustumVisibleSet. The cache opaque color pass consumes it so it
    // stays in lockstep with the depth pre-pass (same set). Keys = GeometryCache keys.
    static const std::vector<uint32_t>& frustumVisibleKeys();
    // Diagnostic: count of cache entries the last buildFrustumVisibleSet dropped via
    // MSOC occlusion refinement (0 when disarmed). For the LogDistantPipeline line.
    static unsigned lastRefineCulled();

    // Diagnostic: the reflection-statics pipeline stage counts from the last
    // isReflectionWaterVisible + cullReflectionSurvivors run, surfaced so the main
    // thread (logReflStaticNearFar) can show WHERE the count diverges on a jump.
    static void getReflPipeDiag(int& tilesTested, int& waterPresent, int& waterOccluded,
                                int& rects, int& queried, int& survivors, bool& msocUsable);
    static void getReflPipeFlip(int& flips, float& distMinCells, float& distMaxCells,
                                float& elevMinDeg, float& elevMaxDeg);

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
    // skipCacheCovered: in CACHE mode (scene 0), skip the recordMW entries the cache
    // owns (textured opaque) — their shadow receiver is folded into the cache color
    // passes (applyCacheShadow in renderCachedOpaque, cacheTerrainSunShadow in
    // renderCachedTerrain) at the snapshot pose, so the async-stale snapshot doesn't
    // flicker against a live-pose receiver on animated geometry.
    static void renderShadow(bool skipCacheCovered = false);
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
