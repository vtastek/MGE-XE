#pragma once

#include "quadtree.h"
#include "ffeshader.h"
#include "mwbridge.h"
#include "specificrender.h"
#include "ipc/client.h"
#include "ipc/dlshare.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>

// Geometry hash function (defined in hiz_culling.cpp)
// Computes a content-based hash from vertex buffer data for identifying identical meshes
size_t computeGeometryHash(IDirect3DVertexBuffer9* vb, UINT offset, UINT stride,
                           UINT vertCount, DWORD fvf);

struct MGEShader;

// DLContext is now defined in ffeshader.h (shared by FrameBuffer and DistantLand)

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

    // RecordedState is now an alias for RecordedMWState (defined in ffeshader.h)
    using RecordedState = RecordedMWState;

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

    static IDirect3DTexture9* texWorldColour, *texWorldNormals, *texWorldDetail;
    static IDirect3DTexture9* texDepthFrame;
    static IDirect3DSurface9* surfDepthFrameMSAA; // Phase A: MSAA render target for depth frame
    static IDirect3DSurface9* surfDepthDepth;
    static IDirect3DSurface9* surfDepthDepthResolved; // Non-MSAA depth buffer for direct writes into texDepthFrame
    static IDirect3DTexture9* texVelocity;
    static IDirect3DSurface9* surfVelocityMSAA; // MSAA render target for velocity buffer
    static bool velocityBufferEnabled;
    static IDirect3DTexture9* texForwardSSAORaw;
    static IDirect3DSurface9* surfForwardSSAORaw;
    static IDirect3DTexture9* texForwardSSAO;
    static IDirect3DSurface9* surfForwardSSAO;
    static IDirect3DTexture9* texForwardSSAONoise;
    static IDirect3DPixelShader9* psForwardSSAO;
    static IDirect3DPixelShader9* psForwardSSAOBlur;
    static IDirect3DVertexBuffer9* vbForwardPrepass;
    static bool forwardSSAOActive;
    static bool forwardSSAOEnabled;
    static bool forwardSSAOBendNormals;
    static IDirect3DTexture9* texCullDepth; // Cull-only depth (recordMW only, for Hi-Z)
    static IDirect3DTexture9* texHiZ; // Hi-Z pyramid - even mips (0,2,4...) for ping-pong generation
    static IDirect3DTexture9* texHiZPrev; // Hi-Z pyramid - odd mips (1,3,5...) for ping-pong generation
    static IDirect3DTexture9* texHiZStaging; // Staging texture for CPU readback - frame N (async copy target)
    static IDirect3DTexture9* texHiZStaging2; // Staging texture - frame N-1 (1 frame old, still copying)
    static IDirect3DTexture9* texHiZStagingPrev; // Staging texture - frame N-2 (2 frames old, safe to lock)
    static IDirect3DQuery9* queryHiZCopy; // D3D9 Event Query to track completion of GetRenderTargetData
    static IDirect3DQuery9* queryHiZCopy2; // Query for staging2 buffer
    static IDirect3DQuery9* queryHiZCopyPrev; // Query for stagingPrev buffer
    static D3DLOCKED_RECT hiZLockedRects[16]; // Pre-locked rects for each mip (locked in Present, unlocked next frame)
    static int hiZLockedMips; // Number of mips currently locked (0 = none, hiZLevels = all)
    static ID3DXEffect* effectHiZ; // Shader effect for Hi-Z downsample
    static IDirect3DVertexShader9* vsHiZ; // Cached Hi-Z vertex shader (compiled once)
    static IDirect3DPixelShader9* psHiZ; // Cached Hi-Z pixel shader (compiled once)
    static int hiZLevels; // Number of levels in Hi-Z pyramid (total texture mips)
    static int hiZValidMips; // Number of actually generated mips (stops at 8x8 minimum)

    static IDirect3DTexture9* texDistantBlend;
    static IDirect3DTexture9* texMenuCache;
    static IDirect3DSurface9* surfMenuCache;
    static bool menuCacheValid;
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
    static IDirect3DTexture9* texBlueNoise;  // 34x1 R16G16F blue noise for shadow PCF
    static IDirect3DSurface9* surfShadowZ;
    static IDirect3DVertexBuffer9* vbFullFrame, *vbClipCube;

    // Texture-based lighting system for HLSL
    struct SceneLight {
        DWORD id;                    // D3D light index for deduplication
        D3DXVECTOR3 position;        // World-space position
        D3DCOLORVALUE diffuse;       // Color (may pulse for dynamic lights)
        float radius;                // Computed from attenuation parameters
        D3DXVECTOR3 falloff;         // (constant, linear, quadratic) attenuation
        bool isVisible;              // After Hi-Z culling
        int lastSeenFrame;           // Frame number when light was last updated
    };
    static std::vector<SceneLight> sceneLights;       // All unique lights in scene
    static std::unordered_map<int, size_t> sceneLightIndexMap;  // ID -> index for O(1) lookup
    static IDirect3DTexture9* texLightData;           // GPU texture with light data

    static D3DXHANDLE ehRcpRes, ehShadowRcpRes;
    static D3DXHANDLE ehWorld, ehView, ehProj;
    static D3DXHANDLE ehShadowViewproj;
    static D3DXHANDLE ehVertexBlendState, ehVertexBlendPalette, ehPrevVertexBlendPalette;
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
    static D3DXHANDLE ehDisplacementFalloff;
    static D3DXHANDLE ehIntensityScalar;
    static D3DXHANDLE ehDebugMode;

    static std::function<void(IDirect3DSurface9*)> captureScreenHandler;
    static bool captureScreenWithUI;

    static bool init();
    static bool initIpc();
    static bool initShader();
    static bool initDepth();
    static bool initForwardPrepass();
    static bool reloadForwardPrepass();
    static bool initHiZ();
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

    // Staging context — setters write here, captureContext() snapshots to DLContext
    static DLContext s_staging;
    static DLContext captureContext();

    static void renderSky(const std::vector<RecordedMWState>& sky, bool useAtmScatter = true);
    static DLContext captureStage0Context();
    static void renderStage0GPU(DLContext* ctx, FixedFunctionShader::FrameBuffer* fb = nullptr);
    static DLContext renderStage0();
    static void renderStage1(DLContext* ctx, FixedFunctionShader::FrameBuffer* fb = nullptr);
    static void renderStage2(DLContext* ctx, FixedFunctionShader::FrameBuffer* fb = nullptr);
    static void renderStageBlend(DLContext* ctx, FixedFunctionShader::FrameBuffer* fb = nullptr);
    static void renderStageWater(DLContext* ctx);

    static void setupCommonEffect(DLContext* ctx, const D3DXMATRIX* view, const D3DXMATRIX* proj);

    static void renderDistantLand(DLContext* ctx, ID3DXEffect* e, const D3DXMATRIX* view, const D3DXMATRIX* proj);
    static void renderDistantLandZ();
    static void cullDistantStatics(DLContext* ctx, const D3DXMATRIX* view, const D3DXMATRIX* proj);
    static void renderDistantStatics(DLContext* ctx);
    static void cullGrass(DLContext* ctx, const D3DXMATRIX* view, const D3DXMATRIX* proj);
    template<class T>
    static void buildGrassInstanceVB(VisibleSet<T>& grassSet);
    static bool hasVisibleGrass();
    static void renderGrassInst(DLContext* ctx);
    static void renderGrassInstZ();
    static void renderGrassCommon(ID3DXEffect* e);

    static void renderWaterReflection(DLContext* ctx, const D3DXMATRIX* view, const D3DXMATRIX* proj, const std::vector<RecordedMWState>* sky = nullptr);
    static void renderReflectedSky(DLContext* ctx, const std::vector<RecordedMWState>& sky);
    static void renderReflectedStatics(DLContext* ctx, const D3DXMATRIX* view, const D3DXMATRIX* proj);
    static void clearReflection(DLContext* ctx);
    static void simulateDynamicWaves();
    static void renderWaterPlane(DLContext* ctx);

    static void renderDepth(DLContext* ctx, const std::vector<RecordedMWState>& recMW, int sceneFilter = -1, FixedFunctionShader::FrameBuffer* fb = nullptr);
    static void renderDepthDistantLand(DLContext* ctx);
    static void renderDepthAdditional(DLContext* ctx, const std::vector<RecordedMWState>& recMW, int sceneFilter = -1, const D3DXMATRIX* viewOverride = nullptr,
        IDirect3DSurface9* targetOverride = nullptr, IDirect3DSurface9* depthStencilOverride = nullptr, bool clearZ = false);
    static void renderDepthRecorded(const std::vector<RecordedMWState>& recMW, int sceneFilter = -1, const D3DXMATRIX* gameView = nullptr, FixedFunctionShader::FrameBuffer* fb = nullptr);
    // Phase 8.4 displaced-depth pass. Iterates recMW, only emits displaced
    // terrain tiles using the color-path VB so the new DepthMWDisplacedVS can
    // apply the same camera-distance falloff. Caller must have entered
    // PASS_RENDERMWDEPTHDISPLACED and set displacementFalloff.
    static void renderDepthRecordedDisplaced(const std::vector<RecordedMWState>& recMW, FixedFunctionShader::FrameBuffer* fb);
    static void renderForwardPrepassChain(DLContext* ctx, const PostProcessData* ppd = nullptr);
    static void generateHiZMipsGPU();
    static void copyHiZToStaging();
    static void lockRemainingHiZMips();
    static void saveHiZSnapshot(); // Save Hi-Z pyramid to temp folder (all mip levels)

    // CPU-based Hi-Z culling
    static bool cullAgainstHiZ(const D3DXVECTOR3& bboxMin, const D3DXVECTOR3& bboxMax, const D3DXMATRIX& worldViewProj, bool debugLog = false);
    // Texture-based lighting system
    static float computeLightRadius(float constant, float linear, float quadratic);

    static void renderShadowMap(DLContext* ctx);
    template<class T>
    static void renderShadowLayerGeneric(DLContext* ctx, MWBridge* mwBridge, int layer, const D3DXMATRIX* inverseCameraProj, D3DXMATRIX* view, D3DXMATRIX* proj, VisibleSet<T>& visible_set);
    static void renderShadowLayer(DLContext* ctx, int layer, float radius, const D3DXMATRIX* inverseCameraProj);
    static void renderShadow(DLContext* ctx);
    static void renderShadowDebug(DLContext* ctx);

    static void postProcess(DLContext* ctx);
    static void postProcess(DLContext* ctx, const PostProcessData& ppd);
    static void updatePostShader(MGEShader* shader);
    static void logWaterDiagnostics(const char* tag, const DLContext* ctx, const PostProcessData* ppd = nullptr);

    static void requestCapture(std::function<void(IDirect3DSurface9*)> handler, bool captureWithUI);
    static void checkCaptureScreenshot(bool isUIDrawn);
    static IDirect3DSurface9* captureScreenshot();
};

// Instrumentation for DXVK render pass break tracking
struct PassBreakCounters {
    // Categorized MGE counters
    int mw_clear;           // Morrowind Clear calls (via proxy)
    int mge_depthRT;        // MGE depth pass RT switches
    int mge_shadowRT;       // MGE shadow map RT switches
    int mge_waterRT;        // MGE water reflection RT switches
    int mge_postRT;         // MGE post-process RT/DS switches
    int mge_stretchRect;    // MGE StretchRect calls
    int mge_otherRT;        // MGE other RT switches (sky, blend, etc.)

    // Raw totals from ALL device calls (MGE + Morrowind)
    int raw_setRT;          // All SetRenderTarget calls
    int raw_setDS;          // All SetDepthStencilSurface calls
    int raw_clear;          // All Clear calls
    int raw_stretchRect;    // All StretchRect calls

    void reset() { memset(this, 0, sizeof(*this)); }
    int categorized() const { return mw_clear + mge_depthRT + mge_shadowRT + mge_waterRT + mge_postRT + mge_stretchRect + mge_otherRT; }
};

extern PassBreakCounters g_passBreaks;

class RenderTargetSwitcher {
    IDirect3DSurface9* savedTarget, *savedDepthStencil;
    void init(IDirect3DSurface9* target, IDirect3DSurface9* targetDepthStencil);

public:
    RenderTargetSwitcher(IDirect3DSurface9* target, IDirect3DSurface9* targetDepthStencil);
    RenderTargetSwitcher(IDirect3DTexture9* targetTex, IDirect3DSurface9* targetDepthStencil);
    ~RenderTargetSwitcher();
};
