#pragma once

#include "proxydx/d3d8header.h"
#include "softwareocclusion.h"
#include "d3dcommandbuffer.h"
#include "renderstate.h"
#include "meshkey.h"
#include "mgedevicehelpers.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>
#include <atomic>

// Per-frame rendering context — snapshotted at start of Stage0, passed through all stages.
struct DLContext {
    // Camera
    D3DXMATRIX mwView, mwProj;
    D3DXVECTOR4 eyeVec, eyePos;

    // Lighting
    D3DXVECTOR4 sunVec, sunPos;
    float sunVis;
    RGBVECTOR sunCol, sunAmb, ambCol;
    RGBVECTOR horizonCol, nearFogCol;

    // Atmosphere
    RGBVECTOR atmOutscatter, atmInscatter;
    D3DXVECTOR4 atmSkylightScatter;

    // Fog
    float fogStart, fogEnd;
    float fogExpStart, fogExpDivisor;
    float fogNearStart, fogNearEnd;
    float nearViewRange;
    float windScaling, niceWeather;
    float lightSunMult, lightAmbMult;

    // Shadow (written by renderShadowMap in Stage0, read by Stage1/2)
    D3DXMATRIX smView[2], smProj[2], smViewproj[2];

    // Flags
    bool isRenderCached;
    bool isPPLActive;
};

// Captured MWBridge state for postProcess — allows render thread execution without MWBridge access
struct PostProcessData {
    int envFlags;
    float frameTime, simulationTime, waterLevel;
    bool isMenu, isInterior, isUnderwater;
};

// Render bin classification for Tracy profiling and debug visualization
enum class RenderBin : uint8_t {
    Terrain,      // Landscape verts (future: detected by land splat pattern)
    Opaque,       // Z-write, no blend, no skin, no grass
    Skinning,     // sk.usesSkinning
    Grass,        // sk.hasGrass
    AlphaTested,  // alphaTest enabled, no blend
    Blending,     // blendEnable
    Count
};

// Dirty flags for performance mode (dirty tracking between frames)
enum DirtyFlags : DWORD {
    DIRTY_NONE      = 0,
    DIRTY_TRANSFORM = 1 << 0,  // worldTransforms[0] changed (object moved)
    DIRTY_LIGHT     = 1 << 1,  // LightState pointer differs
    DIRTY_MATERIAL  = 1 << 2,  // material diffuse/ambient/emissive differs
    DIRTY_SHADER    = 1 << 3,  // ShaderKey differs
    DIRTY_BLEND     = 1 << 4,  // blendEnable/srcBlend/destBlend changed
    DIRTY_TEXTURE   = 1 << 5,  // base texture pointer changed
    DIRTY_ALL       = 0xFFFFFFFF  // new object or mode disabled
};

// DeviceStateSnapshot is in mgedevicehelpers.h

// Expected device state for debug mode (state leak detection)
struct ExpectedDeviceState {
    DWORD alphaBlendEnable, alphaTestEnable;
    DWORD zEnable, zWriteEnable;
    DWORD fogEnable, cullMode;
    DWORD srcBlend, destBlend;
    float depthBias, slopeScaledDepthBias;
    DWORD samplerAddressU[2], samplerAddressV[2];  // stages 0-1
    IDirect3DBaseTexture9* textures[8];  // for leak detection on unused stages
    bool captured;

    ExpectedDeviceState() : captured(false) {}
};

// State contract for explicit state handoffs between pipeline phases.
// Consolidates scattered GetRenderState/SetRenderState calls into a single structure.
// Each phase transition has defined preconditions and postconditions using this contract.
struct StateContract {
    // Depth state
    DWORD zEnable = D3DZB_TRUE;
    DWORD zWriteEnable = TRUE;
    DWORD zFunc = D3DCMP_LESSEQUAL;

    // Blending state
    DWORD alphaBlendEnable = FALSE;
    DWORD srcBlend = D3DBLEND_ONE;
    DWORD destBlend = D3DBLEND_ZERO;

    // Alpha test state
    DWORD alphaTestEnable = FALSE;
    DWORD alphaFunc = D3DCMP_ALWAYS;
    DWORD alphaRef = 0;

    // Culling and fog
    DWORD cullMode = D3DCULL_CW;
    DWORD fogEnable = FALSE;

    // Specular and lighting (legacy FFE state)
    DWORD specularEnable = FALSE;
    DWORD localViewer = FALSE;
    DWORD normalizeNormals = FALSE;

    // Sampler states for stages 0-1 (MW doesn't use 2+)
    struct SamplerState {
        DWORD minFilter = D3DTEXF_LINEAR;
        DWORD magFilter = D3DTEXF_LINEAR;
        DWORD mipFilter = D3DTEXF_LINEAR;
        DWORD addressU = D3DTADDRESS_WRAP;
        DWORD addressV = D3DTADDRESS_WRAP;
    };
    SamplerState samplers[2];

    // Transforms
    D3DMATRIX world = {};
    D3DMATRIX view = {};
    D3DMATRIX projection = {};

    // Capture all state from device (uses GetXXX calls)
    void captureFrom(IDirect3DDevice9* dev);

    // Capture state from MWStateTracker (no device calls - enables async)
    void captureFromTracker(const struct MWStateTracker& tracker);

    // Apply all state to device
    void applyTo(IDirect3DDevice9* dev) const;

    // Apply only render states (not samplers/transforms) to device
    void applyRenderStatesTo(IDirect3DDevice9* dev) const;

    // Apply only sampler states to device
    void applySamplersTo(IDirect3DDevice9* dev) const;

    // Apply only transforms to device
    void applyTransformsTo(IDirect3DDevice9* dev) const;

#ifdef _DEBUG
    // Validate device matches expected state (returns false on mismatch, logs diffs)
    bool validate(IDirect3DDevice9* dev, const char* context) const;
#endif
};


// Named transition points between pipeline phases.
// State is captured at each transition for validation.
enum class PhaseTransition {
    // Main scene boundaries
    RecordingEntry,    // BeginScene 0 - MW starting state
    RecordingExit,     // EndScene 0 pre-GPU - MW ending state (what MW expects after Scene 0)

    // GPU phase sub-stages (renderStage0GPU)
    Stage0Entry,       // Start of GPU phase (shadows, distant land)
    ShadowEntry,       // Before shadow map render
    ShadowExit,        // After shadow map render
    SkyEntry,          // Before sky render
    SkyExit,           // After sky render
    WaterReflEntry,    // Before water reflection render
    WaterReflExit,     // After water reflection render
    Stage0Exit,        // End of renderStage0GPU

    // Offscreen rendering
    OffscreenEntry,    // Start of offscreen (local map, inventory)
    OffscreenExit,     // End of offscreen

    // Stage 1 (grass, shadows over near, depth)
    Stage1Entry,       // Before renderStage1
    DepthEntry,        // Before depth render
    DepthExit,         // After depth render
    Stage1Exit,        // After renderStage1

    // Stage Blend (distant land blend, water plane)
    StageBlendEntry,   // Before renderStageBlend
    WaterPlaneEntry,   // Before water plane render
    WaterPlaneExit,    // After water plane render
    StageBlendExit,    // After renderStageBlend

    // HLSL replay
    ReplayEntry,       // Before HLSL replay
    ReplayExit,        // After HLSL replay

    // Post-GPU boundaries
    GpuExit,           // After all GPU work - must match RecordingExit
    Scene1Entry,       // BeginScene 1 - validates state was restored correctly
    UIEntry,           // UI BeginScene - after all 3D scenes

    Count
};

const char* getPhaseTransitionName(PhaseTransition trans);

// Save/load state baselines to files for comparison
void saveStateBaseline(PhaseTransition trans, const StateContract& state);
bool loadStateBaseline(PhaseTransition trans, StateContract* outState);
bool compareToBaseline(PhaseTransition trans, const StateContract& current);

// RenderedState, RecordedMWState, FragmentState, LightState are in renderstate.h

// Pipeline state diagnostic snapshot — captured at Present() before reset
struct PipelineDiag {
    // Per-frame DIP counters (end-of-frame totals)
    int dipScene0, dipScene1, dipOffscreen, dipUI, dipStencilShadow, dipUnknown;
    // Pipeline flags at end of frame
    int sceneCount;
    bool isMainView, rendertargetNormal, stage0Complete, isFrameComplete;
    bool isHUDComplete, isHUDready, isStencilScene, isAmbientWhite;
    // DistantLand state
    bool distantLandReady, isPPLActive;
    // View matrix at last SetTransform(VIEW) — raw values for detectMenu analysis
    float view_11, view_12, view_13, view_41, view_42, view_43;
    // MWBridge state
    bool mwLoaded;
    DWORD mwCellAddr;
};
extern PipelineDiag g_pipelineDiag;

// Global callback for texture release notification (set by FixedFunctionShader::init)
extern void (*g_onTextureReleased)(IDirect3DTexture9* realTexture);

class FixedFunctionShader {
    struct ShaderKey {
        DWORD uvSets : 4;
        DWORD usesSkinning : 1;
        DWORD vertexColour : 1;
        DWORD heavyLighting : 1;
        DWORD useLighting : 1;
        DWORD lightMode : 2;  // 0=sun only, 1=single point, 2=few (loop up to 8), 3=texture (>8)
        DWORD vertexMaterial : 2;
        DWORD fogMode : 2;
        DWORD activeStages : 3;
        DWORD usesBumpmap : 1;
        DWORD bumpmapStage : 3;
        DWORD usesTexgen : 1;
        DWORD projectiveTexgen : 1;
        DWORD texgenStage : 3;
        DWORD hasDiffParam : 1;        // Has diffuse parameter texture (_diffparam)
        DWORD hasParamH : 1;           // Has parameter texture (_paramh: metallic/roughness|height/IOR)
        DWORD hasParamX : 1;           // Has anisotropic texture (_paramx: aniso rotation/strength/metallic)
        DWORD hasShadows : 1;          // Has shadow mapping enabled
        DWORD hasGrass : 1;            // Is grass texture (enables vertex animation and A2C)
        DWORD hasDetail : 1;           // Has detail texture (conditional binding to minimize overhead)

        struct Stage {
            DWORD colorOp : 6;
            DWORD colorArg1 : 6;
            DWORD colorArg2 : 6;
            DWORD colorArg0 : 6;
            DWORD alphaOpMatched : 1;
            DWORD alphaOpSelect1 : 1;
            DWORD texcoordIndex : 2;
            DWORD texcoordGen : 4;
        } stage[8];

        ShaderKey() {}
        ShaderKey(const RenderedState* rs, const FragmentState* frs, const LightState* lightrs);
        bool operator<(const ShaderKey& other) const;
        bool operator==(const ShaderKey& other) const;
        void log() const;

        struct hasher {
            std::size_t operator()(const ShaderKey& k) const;
        };
    };

    struct ShaderLRU {
        ID3DXEffect* effect;
        FixedFunctionShader::ShaderKey last_sk;
    };

    static IDirect3DDevice* device;
    static ID3DXEffectPool* constantPool;
    static std::unordered_map<ShaderKey, ID3DXEffect*, ShaderKey::hasher> cacheEffects;
    static ShaderLRU shaderLRU;
    static ID3DXEffect* effectDefaultPurple;

    static D3DXHANDLE ehWorld, ehWorldView, ehView;
    static D3DXHANDLE ehVertexBlendState, ehVertexBlendPalette;
    static D3DXHANDLE ehTex0, ehTex1, ehTex2, ehTex3, ehTex4, ehTex5;
    static D3DXHANDLE ehMaterialDiffuse, ehMaterialAmbient, ehMaterialEmissive;
    static D3DXHANDLE ehLightSceneAmbient, ehLightSunDiffuse, ehLightSunDirection;
    static D3DXHANDLE ehLightDiffuse, ehLightAmbient, ehLightPosition;
    static D3DXHANDLE ehLightFalloffQuadratic, ehLightFalloffLinear, ehLightFalloffConstant;
    static D3DXHANDLE ehTexgenTransform, ehBumpMatrix, ehBumpLumiScaleBias;

    static float sunMultiplier, ambMultiplier;

    static ID3DXEffect* generateMWShader(const ShaderKey& sk);

public:
    // Register offset for a shader constant (resolved at compile time from handle)
    static const UINT REG_INVALID = 0xFFFF;
    struct ConstReg {
        UINT reg;
        UINT count;  // Number of float4 registers
        UINT regSet; // 0=FLOAT4, 1=INT4, 2=BOOL (from D3DXREGISTER_SET)
        ConstReg() : reg(REG_INVALID), count(0), regSet(0) {}
    };

private:
    // HLSL Pipeline structures and functions
    struct HLSLShader {
        IDirect3DVertexShader9* vertexShader;
        IDirect3DPixelShader9* pixelShader;
        ID3DXConstantTable* vsConstantTable;
        ID3DXConstantTable* psConstantTable;

        // Cached vertex shader constant handles (avoid per-draw string lookups)
        D3DXHANDLE hWorldViewProj;
        D3DXHANDLE hView;
        D3DXHANDLE hProj;
        UINT projRegister;  // Cached register index for direct SetVertexShaderConstantF fallback
        D3DXHANDLE hWorld;
        D3DXHANDLE hWorldView;
        D3DXHANDLE hVertexBlendPalette;
        D3DXHANDLE hVertexBlendState;
        D3DXHANDLE hShadowWorldViewProj;

        // Resolved register offsets for VS constants (for command buffer path)
        ConstReg regWorldViewProj, regView, regProj, regWorld, regWorldView;
        ConstReg regVertexBlendPalette, regVertexBlendState, regShadowWorldViewProj;

        // Cached pixel shader constant handles
        D3DXHANDLE hMaterialDiffuse;
        D3DXHANDLE hMaterialAmbient;
        D3DXHANDLE hMaterialEmissive;

        // Additional cached lighting constant handles (avoid per-draw string lookups)
        D3DXHANDLE hLightSunDirection;
        D3DXHANDLE hLightSunDiffuse;
        D3DXHANDLE hLightSceneAmbient;
        D3DXHANDLE hShadowRcpRes;
        D3DXHANDLE hPCFFilterSize;

        // Cached point light constant handles (per-object uniform path, lightMode 1-2)
        D3DXHANDLE hLightDiffuse, hLightPosition, hLightAmbient;
        D3DXHANDLE hPointLightCount, hLightFalloffQuadratic, hLightFalloffConstant;

        // Resolved register offsets for PS constants (for command buffer path)
        ConstReg regMaterialDiffuse, regMaterialAmbient, regMaterialEmissive;
        ConstReg regLightSunDirection, regLightSunDiffuse, regLightSceneAmbient;
        ConstReg regShadowRcpRes, regPCFFilterSize;
        ConstReg regLightDiffuse, regLightPosition, regLightAmbient;
        ConstReg regPointLightCount, regLightFalloffQuadratic, regLightFalloffConstant;

        // Dynamic PS constants resolved on first use
        ConstReg regShadingMode, regFogColNear, regMaterialAlpha, regAlphaRef;
        ConstReg regHasVCol, regHasAlpha, regHasBones, regHasAlphaVS;
        ConstReg regTexgenTransform, regBumpMatrix, regBumpLumiScaleBias;
        ConstReg regPCFPenumbraScale, regPCFMinPenumbra, regPCFMaxPenumbra;
        ConstReg regPCFBias, regPCFBias2, regPCFSlopeBias;
        ConstReg regWindVec, regTime, regNormres;
        bool dynamicConstsResolved;

        // Suffix texture support
        IDirect3DTexture9* diffparamTexture;
        IDirect3DTexture9* normalTexture;
        bool hasSuffixSupport;
    };

    struct HLSLShaderLRU {
        HLSLShader shader;
        FixedFunctionShader::ShaderKey last_sk;
    };

    static std::unordered_map<ShaderKey, HLSLShader, ShaderKey::hasher> cacheHLSLShaders;
    static HLSLShaderLRU hlslShaderLRU;
    static HLSLShader hlslShaderDefaultPurple;

    // Diagnostic: track which precached variants get hit (temporary)
    static std::unordered_set<ShaderKey, ShaderKey::hasher> diagHitKeys;

    static SRWLOCK hlslCacheLock;  // Protects cacheHLSLShaders (zero-init = valid)
    static HANDLE precacheThread;  // Joinable precache thread handle

    // Material state cache to minimize redundant SetRenderState calls
    struct MaterialStateCache {
        bool initialized;

        // Depth and culling states
        DWORD zEnable, zWriteEnable, zFunc, cullMode;
        bool zEnableValid, zWriteEnableValid, zFuncValid, cullModeValid;

        // Alpha blending states
        DWORD alphaBlendEnable, srcBlend, destBlend;
        bool alphaBlendEnableValid, srcBlendValid, destBlendValid;

        // Alpha testing states
        DWORD alphaTestEnable, alphaFunc, alphaRef;
        bool alphaTestEnableValid, alphaFuncValid, alphaRefValid;

        // DX8 specular states (always disabled in HLSL)
        DWORD specularEnable, localViewer, normalizeNormals;
        bool specularEnableValid, localViewerValid, normalizeNormalsValid;

        // Vertex format
        DWORD fvf;
        bool fvfValid;

        MaterialStateCache() : initialized(false),
            zEnableValid(false), zWriteEnableValid(false), zFuncValid(false), cullModeValid(false),
            alphaBlendEnableValid(false), srcBlendValid(false), destBlendValid(false),
            alphaTestEnableValid(false), alphaFuncValid(false), alphaRefValid(false),
            specularEnableValid(false), localViewerValid(false), normalizeNormalsValid(false),
            fvfValid(false) {}

        void reset() {
            initialized = false;
            zEnableValid = zWriteEnableValid = zFuncValid = cullModeValid = false;
            alphaBlendEnableValid = srcBlendValid = destBlendValid = false;
            alphaTestEnableValid = alphaFuncValid = alphaRefValid = false;
            specularEnableValid = localViewerValid = normalizeNormalsValid = false;
            fvfValid = false;
        }
    };
    static MaterialStateCache materialCache;

    // Texture binding cache to minimize redundant SetTexture calls and preserve sampler states
    struct TextureBindingCache {
        IDirect3DTexture9* boundTextures[8];  // Track bound textures for slots 0-7
        bool textureValid[8];                 // Track which slots have valid cached values

        // Sampler state preservation for suffix texture binding
        struct SamplerState {
            DWORD addressU, addressV;
            bool addressUValid, addressVValid;
        } samplerStates[8];

        TextureBindingCache() {
            reset();
        }

        void reset() {
            for (int i = 0; i < 8; i++) {
                boundTextures[i] = nullptr;
                textureValid[i] = false;
                samplerStates[i].addressUValid = false;
                samplerStates[i].addressVValid = false;
            }
        }

        bool needsUpdate(DWORD stage, IDirect3DTexture9* texture) {
            if (stage >= 8) return true;  // Outside cache range
            if (!textureValid[stage]) return true;  // No cached value
            return boundTextures[stage] != texture;  // Different texture
        }

        void updateCache(DWORD stage, IDirect3DTexture9* texture) {
            if (stage < 8) {
                boundTextures[stage] = texture;
                textureValid[stage] = true;
            }
        }

        void cacheSamplerState(DWORD stage, D3DSAMPLERSTATETYPE type, DWORD value) {
            if (stage >= 8) return;
            if (type == D3DSAMP_ADDRESSU) {
                samplerStates[stage].addressU = value;
                samplerStates[stage].addressUValid = true;
            } else if (type == D3DSAMP_ADDRESSV) {
                samplerStates[stage].addressV = value;
                samplerStates[stage].addressVValid = true;
            }
        }

        void restoreSamplerStates(IDirect3DDevice9* device, DWORD stage) {
            if (stage >= 8) return;
            if (samplerStates[stage].addressUValid) {
                device->SetSamplerState(stage, D3DSAMP_ADDRESSU, samplerStates[stage].addressU);
            }
            if (samplerStates[stage].addressVValid) {
                device->SetSamplerState(stage, D3DSAMP_ADDRESSV, samplerStates[stage].addressV);
            }
        }
    };
    static TextureBindingCache textureCache;

    // State contracts for phase transitions (replaces scattered GetRenderState calls)
    static StateContract preRecordingContract;   // MW state before recording starts
    static StateContract postRecordingContract;  // MW state at end of Scene 0 (restored after replay)

    // Phase transition tracking - captured state at each boundary
    static StateContract transitionState[static_cast<int>(PhaseTransition::Count)];

    // Exterior texture binding optimizations
    static bool isExteriorShadowBound;
    static bool isDetailTextureBound;

    // Shadow matrix caching to avoid per-draw calculations
    static D3DXMATRIX cachedViewMatrix;
    static D3DXMATRIX cachedInverseView;
    static D3DXMATRIX cachedViewToShadow[2];
    static bool shadowMatricesValid;

    // Default textures to avoid null binds that cause DXVK descriptor updates
    static IDirect3DTexture9* defaultWhiteTexture;
    static IDirect3DTexture9* defaultBlackTexture;
    static IDirect3DTexture9* defaultNormalTexture;  // 128,128,255,255 for flat normal

    // Original detail texture storage for frequency optimization
    static IDirect3DBaseTexture9* savedOriginalDetailTexture;

    // Per-object light packing for mode 3 (saturated Morrowind assignment)
    struct PerObjectLightInfo {
        int texelOffset;   // starting texel index in packed texture
        int lightCount;    // number of nearby lights for this object
    };
    static std::vector<PerObjectLightInfo> perObjectLightInfo;
    static float perObjectTexelSize;  // 1/totalTexels, computed once per frame
    static IDirect3DTexture9* texPerObjectLightData;  // Per-object packed light texture


    // Async compilation system
    struct AsyncShaderRequest {
        ShaderKey key;
        std::atomic<bool> completed{false};
        HLSLShader result;
        
        AsyncShaderRequest(const ShaderKey& k) : key(k) {}
    };
    
    static std::queue<std::shared_ptr<AsyncShaderRequest>> compilationQueue;
    static std::mutex queueMutex;
    static std::condition_variable queueCondition;
    static std::thread compilerThread;
    static std::atomic<bool> shutdownCompiler;
    static std::unordered_map<ShaderKey, std::shared_ptr<AsyncShaderRequest>, ShaderKey::hasher> pendingCompilations;

    // Vertex shader caching (vertex shaders don't use texture suffix defines)
    struct VertexShaderKey {
        DWORD useLighting : 1;
        DWORD lightMode : 2;  // matches ShaderKey::lightMode
        DWORD usesSkinning : 1;
        DWORD vertexColour : 1;
        
        bool operator==(const VertexShaderKey& other) const {
            return memcmp(this, &other, sizeof(VertexShaderKey)) == 0;
        }
        
        struct hasher {
            std::size_t operator()(const VertexShaderKey& k) const {
                return std::hash<uint32_t>{}(*(uint32_t*)&k);
            }
        };
    };
    
    static std::unordered_map<VertexShaderKey, IDirect3DVertexShader9*, VertexShaderKey::hasher> vertexShaderCache;

    static HLSLShader generateMWShaderHLSL(const ShaderKey& sk);
    static HLSLShader createPurpleErrorShader();
    static void captureAndDumpTexture(IDirect3DTexture9* texture);

    // HLSL Render Dispatch Recording System with proper resource management
    struct RecordedRenderedState : RenderedState {
        RecordedRenderedState(const RenderedState&);
        ~RecordedRenderedState();
        RecordedRenderedState(const RecordedRenderedState&) = delete;
        RecordedRenderedState(RecordedRenderedState&&) noexcept;
    };

    struct RecordedFragmentState : FragmentState {
        RecordedFragmentState(const FragmentState& frs) : FragmentState(frs) {}
    };

    struct RecordedLightState : LightState {
        RecordedLightState(const LightState& lightrs) : LightState(lightrs) {}
    };

public:
    // Mesh identifier for bbox caching (VB + IB + FVF combo uniquely identifies object-space mesh)
    // Type aliases for mesh identification (definitions in meshkey.h)
    using MeshKey = ::MeshKey;
    using MeshKeyHash = ::MeshKeyHash;
    using ObjectSpaceBBox = ::ObjectSpaceBBox;

    struct HLSLRecordedCall {
        RecordedRenderedState rs;
        RecordedFragmentState frs;
        std::shared_ptr<LightState> lightrs;  // Shared pointer to avoid redundant copies
        ShaderKey sk;

        // Captured sampler states for each texture stage
        struct SamplerState {
            DWORD addressU = D3DTADDRESS_WRAP;
            DWORD addressV = D3DTADDRESS_WRAP;
            bool captured = false;
        };
        SamplerState samplerStates[8];  // D3D9 supports up to 8 texture stages

        // Bounding box for Hi-Z culling
        D3DXVECTOR3 bboxMin;
        D3DXVECTOR3 bboxMax;
        bool hasBoundingBox;

        // Index into recordMW for visibility lookup (-1 if not in recordMW, e.g., alpha objects)
        int recordMWIndex;

        // Whether this call has been through the prepare phase (shader key, bbox, etc.)
        bool prepared;

        // Render bin classification
        RenderBin bin;

        // Dirty tracking for performance mode
        DWORD dirtyFlags;

        // Whether this call should be rendered (set by cull pass)
        bool shouldRender = true;

        // Scene number this call was recorded in (0=world, 1=particles, 2=hands)
        int sceneNum = 0;

        // Expected device state for debug mode (state leak detection)
        ExpectedDeviceState expectedState;

        // Complete device state snapshot for async replay (no assumptions)
        DeviceStateSnapshot deviceState;

        // Vertex/Index snapshots for dynamic VB draws (Scene 1/2 particles)
        // Captures actual vertex bytes at record time since shared VB may be overwritten
        std::vector<BYTE> vertexSnapshot;
        std::vector<BYTE> indexSnapshot;
        bool usesSnapshot = false;  // True for Scene 1/2, false for Scene 0 static geometry
        UINT stagingVBOffset = 0;   // Offset into staging VB for this call's data
        UINT stagingIBOffset = 0;   // Offset into staging IB for this call's data

        // Constructor to capture render state data with proper resource management
        HLSLRecordedCall(const RenderedState* rs_, const FragmentState* frs_, std::shared_ptr<LightState> lightrs_, const ShaderKey& sk_, int recordMWIdx = -1);
        // Implementation moved to cpp file to handle sampler state capture
    };

    // VB+IB key for bbox lookup between recordedCalls and recordMW
    // Type aliases for VB+IB key (definitions in meshkey.h)
    using VBIBKey = ::VBIBKey;
    using VBIBKeyHash = ::VBIBKeyHash;

    // Buffer lifecycle states for triple-buffered pipeline
    enum class BufferState {
        Available,      // Free for recording
        Recording,      // Main thread writing draw calls
        ReadyToCull,    // Recording complete, waiting for cull thread
        Culling,        // Cull thread processing (Hi-Z, shouldRender, shader keys)
        ReadyToRender,  // Cull complete, waiting for render thread
        Rendering       // Render thread submitting GPU calls
    };

    struct FrameBuffer {
        std::vector<HLSLRecordedCall> recordedCalls;        // Scene 0 (main world)
        std::vector<HLSLRecordedCall> recordedCallsScene1;  // Scene 1 (particles, alpha sorted)
        std::vector<HLSLRecordedCall> recordedCallsScene2;  // Scene 2 (hands, skinned)

        // Matrices captured at recording time (per-scene)
        D3DXMATRIX view, proj;           // Scene 0 (world)
        D3DXMATRIX viewScene1, projScene1;  // Scene 1 (particles) - may differ if camera moves during Scene 0
        D3DXMATRIX viewScene2, projScene2;  // Scene 2 (hands) - different view matrix
        D3DXMATRIX shadowViewproj[2];

        // Matrices stamped at Present() for render pass (fresh camera)
        D3DXMATRIX currentView, currentProj;
        D3DXMATRIX currentShadowViewproj[2];

        // Per-buffer occluder set (no cross-buffer sharing)
        std::unordered_set<MeshKey, MeshKeyHash> rasterizedOccluderMeshes;

        // Per-buffer bbox lookup (rebuilt each frame from recordedCalls)
        std::unordered_map<VBIBKey, ObjectSpaceBBox, VBIBKeyHash> bboxLookup;

        // Per-buffer LightState cache for recording
        std::shared_ptr<LightState> lastLightState;

        // Per-buffer scene data (HLSL mode only — isolates record/render phases)
        std::vector<RecordedMWState> recordMW;
        std::vector<RecordedMWState> recordSky;
        DLContext dlContext;
        bool waterSeen = false;
        IDirect3DTexture9* texDistantBlend = nullptr;
        StateContract stateContract;  // MW device state at end of Scene 0 (restored after replay)

        // HLSL replay command buffer — built by replayRecordedCalls, replayed in executeGpuPhase
        D3DCommandBuffer hlslCmds;

        // Captured MWBridge state for postProcess (render thread safe)
        PostProcessData postProcessData = {};

        // Snapshot of MWStateTracker at end of recording (for render thread state fix)
        MWStateTracker trackerSnapshot;

        // Staging buffers for Scene 1/2 dynamic VB snapshots
        // Persistent across frames (resized as needed, not released in clear())
        IDirect3DVertexBuffer9* particleStagingVB = nullptr;
        IDirect3DIndexBuffer9* particleStagingIB = nullptr;
        UINT stagingVBSize = 0;  // Current allocated size in bytes
        UINT stagingIBSize = 0;

        bool valid;
        BufferState state;
        int frameNumber = -1;  // Frame number when this buffer was recorded (-1 = never)

        FrameBuffer() : valid(false), state(BufferState::Available), frameNumber(-1) {}

        void clear() {
            recordedCalls.clear();
            recordedCallsScene1.clear();
            recordedCallsScene2.clear();
            rasterizedOccluderMeshes.clear();
            bboxLookup.clear();
            lastLightState.reset();
            recordMW.clear();
            recordSky.clear();
            waterSeen = false;
            texDistantBlend = nullptr;
            stateContract = StateContract();  // Reset to default state
            hlslCmds.clear();
            postProcessData = {};
            trackerSnapshot.clear();
            dlContext = DLContext();  // Reset DLContext to prevent garbage values
            valid = false;
            // Initialize matrices to identity to prevent garbage if capture functions aren't called
            D3DXMatrixIdentity(&view);
            D3DXMatrixIdentity(&proj);
            D3DXMatrixIdentity(&viewScene1);
            D3DXMatrixIdentity(&projScene1);
            D3DXMatrixIdentity(&viewScene2);
            D3DXMatrixIdentity(&projScene2);
            memset(shadowViewproj, 0, sizeof(shadowViewproj));
        }

        void reserve() {
            recordedCalls.reserve(4000);
            recordedCallsScene1.reserve(100);   // Particles: moderate draws
            recordedCallsScene2.reserve(50);    // Hands: far fewer draws
        }
    };

    // Triple buffering for N / N-1 / N-2 pipeline:
    // recordingBuffer: Frame N - main thread records draw calls
    // prepBuffer:      Frame N-1 - CPU prep thread processes (shader keys, bins, cull)
    // renderBuffer:    Frame N-2 - GPU thread renders (what actually displays)
    static FrameBuffer frameBuffers[3];
    static int recordingBuffer;   // Index for frame N (main thread recording)
    static int prepBuffer;        // Index for frame N-1 (CPU prep)
    static int renderBuffer;      // Index for frame N-2 (GPU render)
    static bool n1Ready;          // True after first frame completes (N-1 data available)
    static bool n2Ready;          // True after second frame completes (N-2 data available)
    static int swapCount;         // Track swaps for warm-up (moved from static local)

public:
    // Pipeline phase tracking for GPU call separation verification
    enum class PipelinePhase {
        Idle,           // Between frames or non-HLSL
        FrameCapture,   // captureStage0Context: effect uniforms, camera reads
        Recording,      // Main thread capturing DIP calls (should be CPU-only)
        CpuPrepare,     // Hi-Z culling, shader key computation (CPU-only)
        GpuRender       // Stages + replay (GPU calls expected here)
    };
    static PipelinePhase currentPhase;

    struct PhaseCallCounts {
        int deviceReads;    // Get* calls (safe, just reading state)
        int deviceWrites;   // Set*, effect->Set* (state changes, need to defer for threading)
        int deviceSubmits;  // Draw*, Clear, StretchRect, CreateStateBlock (GPU work, must not happen)
        void reset() { deviceReads = deviceWrites = deviceSubmits = 0; }
    };
    static PhaseCallCounts frameCaptureGpuCalls;
    static PhaseCallCounts recordingGpuCalls;

private:
    static bool isRecording;
    static bool isReplaying;
    static bool manualRecordingControl;  // When true, user controls recording via K key
    static bool recordingEnabled;  // Global toggle for entire recording system
    static bool recordingCompletedThisFrame;  // Prevents restarting recording after Scene 0
    static int currentRecordingScene;  // Scene number being recorded (0=world, 1=particles, 2=hands)
    static bool hiZBuiltThisFrame;  // Prevents rebuilding Hi-Z pyramid multiple times per frame
    static bool dumpRequested;  // When true, preserve calls for dump

    // Bbox cache: maps mesh identifier to object-space bbox (persists across frames)
    // Protected by bboxCacheMutex for thread safety (main thread + CpuPrepThread access)
    static std::unordered_map<MeshKey, ObjectSpaceBBox, MeshKeyHash> bboxCache;
    static std::mutex bboxCacheMutex;

    // Bbox lookup: maps VB+IB to world-space bbox (rebuilt each frame from recordedCalls)
    static std::unordered_map<VBIBKey, ObjectSpaceBBox, VBIBKeyHash> bboxLookup;

    // Visibility results (index-based): stores culling results indexed by draw order
    static std::vector<int8_t> visibilityResults;

    // Visibility lookup (map-based): stores culling results from depth pass for replay to reuse
    // Key is MeshKey, value is whether ANY instance of that mesh is visible
    // Uses OR-logic: if any instance visible, key = true (conservative, avoids missing objects)
    static std::unordered_map<MeshKey, bool, MeshKeyHash> visibilityLookup;

    // Previous frame camera tracking for velocity-based bbox expansion
    static D3DXVECTOR3 prevCameraPos;
    static D3DXMATRIX prevCameraView;
    static bool hasPrevCamera;

    // LightState cache: reuse shared_ptr for identical lighting states to avoid redundant allocations
    static std::shared_ptr<LightState> lastLightState;
    static const LightState* lastLightStatePtr;  // Raw pointer for fast O(1) comparison
    static bool compareLightStates(const LightState* a, const LightState* b);

    static void startRecording();
    static void stopRecordingAndReplay();
    static void recordRenderCall(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, const ShaderKey& sk, int recordMWIdx = -1);
    static void replayRecordedCalls(int sceneCount, D3DCommandBuffer* cmdBuf = nullptr);
    static void renderMorrowindHLSL_Internal(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, DWORD dirtyFlags = DIRTY_ALL, int callIndex = -1, D3DCommandBuffer* cmdBuf = nullptr, const DeviceStateSnapshot* capturedState = nullptr, const HLSLRecordedCall* replayCall = nullptr);
    static void validateDeviceState(const ExpectedDeviceState& expected, int callIndex);
    static void markAllCallsDirty();
    static ShaderKey computeShaderKeyWithSuffixes(const RenderedState* rs, const FragmentState* frs, LightState* lightrs);
    static bool computeBoundingBox(const RenderedState* rs, D3DXVECTOR3& bboxMin, D3DXVECTOR3& bboxMax);

public:
    static void prepareRecordedCalls();  // CPU-only: compute shader keys, bins (can run async)
    static void finalizeBatchAndReplay(int sceneCount = 0); // Call when HLSL rendering session is complete
    static void finalizeBatchAndSubmitCull();  // Stop recording, submit to cull thread (before renderStage1)
    static void waitCullAndReplay();           // Wait for cull, replay, restore state (after renderStageBlend)
    static void capturePostRecordingState();   // Capture MW device state at end of Scene 0 (recording continues)
    static void restorePostRecordingState();   // Clean up device state for Scene 1/2 (shaders, textures)
    static void finalizeAndRender(DLContext* frameCtx, bool waterSeen); // Prepare + render phase at frame finalize point
    static void finalizeAndRenderAllScenes(DLContext* frameCtx, bool waterSeen); // Full deferred GPU phase at UI BeginScene
    static void executeGpuPhase(); // GPU render block — called by render thread or inline
    static void replayScene1And2(FrameBuffer* fb);  // Replay Scene 1/2 at UI BeginScene (after recording)

    // Scene lifecycle for triple-buffered pipeline
    static void markSceneStart(int sceneNum, bool isUI = false);
    static void markSceneEnd();

    // Single-buffer pipeline: split cull (CPU) and render (GPU) passes
    static void executeCullPass();
    static void executeRenderPass();

    // Async GPU path: runs on render thread during safe zone
    // Combines finalizeAndRenderAllScenes + postProcess without UI state save/restore
    static void renderFullFrameAsync();

    // Triple buffer accessors
    static FrameBuffer& getRecordingBuffer() { return frameBuffers[recordingBuffer]; }  // Frame N
    static FrameBuffer& getPrepBuffer() { return frameBuffers[prepBuffer]; }            // Frame N-1
    static FrameBuffer& getRenderingBuffer() { return frameBuffers[renderBuffer]; }     // Frame N-2
    static void swapBuffers();  // Called at Present() to rotate buffers
    static bool isN1Ready() { return n1Ready; }
    static bool isN2Ready() { return n2Ready; }

    // Debug controls for record/replay system
    static bool getIsRecording() { return isRecording; }
    static bool getIsReplaying() { return isReplaying; }
    static void setRecordingState(bool recording) { isRecording = recording; }
    static void resetRecordingCompletedFlag() { recordingCompletedThisFrame = false; currentRecordingScene = 0; }
    static void setCurrentRecordingScene(int scene) { currentRecordingScene = scene; }
    static int getCurrentRecordingScene() { return currentRecordingScene; }
    static void captureScene1Matrices();  // Capture particles view/proj at Scene 1 start
    static void captureScene2Matrices();  // Capture hands view/proj at Scene 2 start
    static void saveOffscreenState();     // Save blend state before offscreen rendering
    static void restoreOffscreenState();  // Restore blend state after offscreen rendering
    static void resetHiZBuiltFlag() { hiZBuiltThisFrame = false; }
    static void setReplayingState(bool replaying) { isReplaying = replaying; }
    static void setManualRecordingControl(bool manual) { manualRecordingControl = manual; }
    static bool getManualRecordingControl() { return manualRecordingControl; }
    static size_t getRecordedCallsCount() { return frameBuffers[recordingBuffer].recordedCalls.size(); }

    // Visibility results for depth pass (indexed by recordMW)
    static const std::vector<int8_t>& getVisibilityResults() { return visibilityResults; }

    // Global recording system toggle
    static bool getRecordingEnabled() { return recordingEnabled; }
    static void setRecordingEnabled(bool enabled) { recordingEnabled = enabled; }

    // Dump control
    static void requestDump() { dumpRequested = true; }

    // Pipeline phase tracking
    static PipelinePhase getPhase() { return currentPhase; }
    static void setPhase(PipelinePhase phase) { currentPhase = phase; }
    static void trackDeviceRead(const char* callName);   // Get* calls
    static void trackDeviceWrite(const char* callName);  // Set*, effect->Set* calls
    static void trackDeviceSubmit(const char* callName); // Draw*, Clear, StretchRect, CreateStateBlock
    // Convenience: legacy name for backwards compat
    static void trackGpuCall(const char* callName) { trackDeviceSubmit(callName); }

    static const std::vector<HLSLRecordedCall>& getRecordedCalls() { return frameBuffers[recordingBuffer].recordedCalls; }

    // Scene handover debugging (particle bug investigation)
    static void logSceneHandoverState(const char* label);

    // Phase transition tracking - captures state and validates against expected
    // Returns true if state matches expected (or no expected state to compare)
    static bool transitionTo(PhaseTransition trans);

    // Get captured state at a transition (for comparison)
    static const StateContract& getTransitionState(PhaseTransition trans);

    // Baseline management - save/load known-good states
    static void saveCurrentAsBaseline();   // Save all current transition states as baselines
    static void validateAgainstBaselines(); // Compare current states to saved baselines

private:

public:
    // Software occlusion culler for CPU-based Hi-Z depth buffer generation (public for depth culling)
    static SoftwareOcclusionCuller softwareOcclusionCuller;

    // Set of meshes rasterized as occluders - these must never be culled by Hi-Z
    static std::unordered_set<MeshKey, MeshKeyHash> rasterizedOccluderMeshes;

    // Hi-Z culling: split into CPU-only test pass and lightweight filter pass
    // executeHiZCulling: bbox computation, occluder rasterization, Hi-Z pyramid, visibility testing.
    //   Pure CPU work (no D3D device access) — draw thread candidate.
    //   Populates visibilityResults[] and sets shouldRender on recordedCalls.
    static void executeHiZCulling(const D3DXMATRIX& currentView, const D3DXMATRIX& currentProj);
    // applyVisibilityAndFilterRecordMW: filters recordMW using visibilityResults from executeHiZCulling.
    static void applyVisibilityAndFilterRecordMW();

    static bool init(IDirect3DDevice* d, ID3DXEffectPool* pool);
    static void startEarlyPrecache(IDirect3DDevice* d);
    static void updateLighting(float sunMult, float ambMult);
    static void renderMorrowind(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, int recordMWIdx = -1);
    static void renderMorrowindHLSL(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, int recordMWIdx = -1);
    static void release();
    static void invalidateShaderSourceCache();
    static void startAsyncCompiler();
    static void stopAsyncCompiler();
    static void queueShaderCompilation(const ShaderKey& key);
    static void processAsyncCompletions();
    static void resetHLSLCaches(); // Reset material/texture caches for HLSL rendering session
    static void setCachedTexture(IDirect3DDevice9* device, DWORD stage, IDirect3DTexture9* texture); // Cached texture binding
    static void setCachedTextureWithSamplerPreservation(IDirect3DDevice9* device, DWORD stage, IDirect3DTexture9* texture); // Cached texture binding that preserves sampler states
    static void captureSamplerStates(IDirect3DDevice9* device, DWORD stage); // Capture current sampler states before texture binding
    static void createDefaultTextures(); // Create default textures to avoid null binds
    static void bindShaderTextures(const ShaderKey& sk, const RenderedState* rs); // Smart texture binding for shader

    // Pipeline diagnostic snapshot hotkeys (called from Present())
    static void checkSnapshotHotkeys();
};
