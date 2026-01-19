#pragma once

#include "proxydx/d3d8header.h"
#include "softwareocclusion.h"

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



struct RenderedState {
    IDirect3DTexture9* texture;
    IDirect3DVertexBuffer9* vb;
    UINT vbOffset, vbStride;
    IDirect3DIndexBuffer9* ib;
    DWORD ibBase;
    DWORD fvf;
    DWORD zWrite, cullMode;
    DWORD vertexBlendState;
    D3DXMATRIX worldTransforms[4];
    D3DXMATRIX viewTransform;
    D3DXMATRIX worldViewTransforms[4];
    D3DXMATRIX shadowWorldViewProj[2];  // Complete shadow world-view-projection matrices at time of recording
    D3DCOLORVALUE diffuseMaterial;
    BYTE blendEnable, srcBlend, destBlend;
    BYTE alphaTest, alphaFunc, alphaRef;
    BYTE useLighting, useFog, matSrcDiffuse, matSrcEmissive;

    D3DPRIMITIVETYPE primType;
    UINT baseIndex, minIndex, vertCount, startIndex, primCount;

    // Bounding box for occlusion culling during depth rendering
    D3DXVECTOR3 bboxMin, bboxMax;
    bool hasBoundingBox;
};

struct FragmentState {
    struct Stage {
        BYTE colorOp, colorArg1, colorArg2;
        BYTE alphaOp, alphaArg1, alphaArg2;
        BYTE colorArg0, alphaArg0, resultArg;
        DWORD texcoordIndex;
        DWORD texTransformFlags;
        float bumpEnvMat[2][2];
        float bumpLumiScale, bumpLumiBias;
    } stage[8];

    struct Material {
        D3DCOLORVALUE diffuse, ambient, emissive;
    } material;
};

struct LightState {
    struct Light {
        D3DLIGHTTYPE type;
        D3DCOLORVALUE diffuse;
        D3DVECTOR position;     // position / normalized direction
        D3DVECTOR viewspacePos;
        union {
            D3DVECTOR falloff;  // constant, linear, quadratic
            D3DVECTOR ambient;  // for directional lights
        };
    };

    D3DCOLORVALUE globalAmbient;
    std::unordered_map<DWORD, Light> lights;
    std::unordered_map<DWORD, bool> lightsTransformed;
    std::vector<DWORD> active;
};

class FixedFunctionShader {
    struct ShaderKey {
        DWORD uvSets : 4;
        DWORD usesSkinning : 1;
        DWORD vertexColour : 1;
        DWORD heavyLighting : 1;
        DWORD useLighting : 1;
        DWORD noPointLights : 1;
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
        D3DXHANDLE hWorld;
        D3DXHANDLE hWorldView;
        D3DXHANDLE hVertexBlendPalette;
        D3DXHANDLE hVertexBlendState;
        D3DXHANDLE hShadowWorldViewProj;

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

    // Shader source caching for hot reload support
    struct CachedShaderSource {
        char* source;
        DWORD size;
        FILETIME lastWriteTime;
    };
    static std::unordered_map<std::string, CachedShaderSource> shaderSourceCache;
    static bool needsCacheReset;  // Flag to trigger cache reset after file changes

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
        DWORD noPointLights : 1;
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

    static char* loadShaderFile(const char* filename, DWORD* outFileSize);
    static bool detectDXVK();
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

    // Mesh identifier for bbox caching (VB + IB + FVF combo uniquely identifies object-space mesh)
    struct MeshKey {
        IDirect3DVertexBuffer9* vb;
        IDirect3DIndexBuffer9* ib;
        DWORD fvf;
        UINT baseIndex;
        UINT vertCount;
        UINT startIndex;
        UINT primCount;

        bool operator==(const MeshKey& other) const {
            return vb == other.vb && ib == other.ib && fvf == other.fvf &&
                   baseIndex == other.baseIndex && vertCount == other.vertCount &&
                   startIndex == other.startIndex && primCount == other.primCount;
        }
    };

    // Hash function for MeshKey
    struct MeshKeyHash {
        std::size_t operator()(const MeshKey& k) const {
            std::size_t h1 = std::hash<void*>{}(k.vb);
            std::size_t h2 = std::hash<void*>{}(k.ib);
            std::size_t h3 = std::hash<DWORD>{}(k.fvf);
            std::size_t h4 = std::hash<UINT>{}(k.baseIndex);
            std::size_t h5 = std::hash<UINT>{}(k.vertCount);
            std::size_t h6 = std::hash<UINT>{}(k.startIndex);
            std::size_t h7 = std::hash<UINT>{}(k.primCount);
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5) ^ (h7 << 6);
        }
    };

    // Cached object-space bounding box
    struct ObjectSpaceBBox {
        D3DXVECTOR3 bboxMin;
        D3DXVECTOR3 bboxMax;
    };

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

        // Constructor to capture render state data with proper resource management
        HLSLRecordedCall(const RenderedState* rs_, const FragmentState* frs_, std::shared_ptr<LightState> lightrs_, const ShaderKey& sk_);
        // Implementation moved to cpp file to handle sampler state capture
    };

    static std::vector<HLSLRecordedCall> recordedCalls;
    static std::vector<HLSLRecordedCall> previousFrameCalls;  // Previous frame for raycast targeting
    static bool isRecording;
    static bool isReplaying;
    static bool manualRecordingControl;  // When true, user controls recording via K key
    static bool recordingEnabled;  // Global toggle for entire recording system
    static bool recordingCompletedThisFrame;  // Prevents restarting recording after Scene 0
    static bool hiZBuiltThisFrame;  // Prevents rebuilding Hi-Z pyramid multiple times per frame
    static bool dumpRequested;  // When true, preserve calls for dump

    // Consistent matrices for entire recording session
    static D3DXMATRIX recordingDeviceView, recordingDeviceProj;
    static D3DXMATRIX recordingShadowViewproj[2];

    // Bbox cache: maps mesh identifier to object-space bbox (persists across frames)
    static std::unordered_map<MeshKey, ObjectSpaceBBox, MeshKeyHash> bboxCache;

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
    static void recordRenderCall(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, const ShaderKey& sk);
    static void replayRecordedCalls(int sceneCount);
    static void renderMorrowindHLSL_Internal(const RenderedState* rs, const FragmentState* frs, LightState* lightrs);
    static ShaderKey computeShaderKeyWithSuffixes(const RenderedState* rs, const FragmentState* frs, LightState* lightrs);
    static bool computeBoundingBox(const RenderedState* rs, D3DXVECTOR3& bboxMin, D3DXVECTOR3& bboxMax);

public:
    static void finalizeBatchAndReplay(int sceneCount = 0); // Call when HLSL rendering session is complete

    // Debug controls for record/replay system
    static bool getIsRecording() { return isRecording; }
    static bool getIsReplaying() { return isReplaying; }
    static void setRecordingState(bool recording) { isRecording = recording; }
    static void resetRecordingCompletedFlag() { recordingCompletedThisFrame = false; }
    static void resetHiZBuiltFlag() { hiZBuiltThisFrame = false; }
    static void setReplayingState(bool replaying) { isReplaying = replaying; }
    static void setManualRecordingControl(bool manual) { manualRecordingControl = manual; }
    static bool getManualRecordingControl() { return manualRecordingControl; }
    static size_t getRecordedCallsCount() { return recordedCalls.size(); }

    // Global recording system toggle
    static bool getRecordingEnabled() { return recordingEnabled; }
    static void setRecordingEnabled(bool enabled) { recordingEnabled = enabled; }

    // Dump control
    static void requestDump() { dumpRequested = true; }

    static const std::vector<HLSLRecordedCall>& getRecordedCalls() { return recordedCalls; }

private:

public:
    // Software occlusion culler for CPU-based Hi-Z depth buffer generation (public for depth culling)
    static SoftwareOcclusionCuller softwareOcclusionCuller;

    // Set of meshes rasterized as occluders - these must never be culled by Hi-Z
    static std::unordered_set<MeshKey, MeshKeyHash> rasterizedOccluderMeshes;

    // Prepare occlusion culling for depth rendering (build Hi-Z and filter recordMW)
    static void prepareOcclusionCullingForDepth();

    static bool init(IDirect3DDevice* d, ID3DXEffectPool* pool);
    static void startEarlyPrecache(IDirect3DDevice* d);
    static void precacheAsync();
    static void updateLighting(float sunMult, float ambMult);
    static void renderMorrowind(const RenderedState* rs, const FragmentState* frs, LightState* lightrs);
    static void renderMorrowindHLSL(const RenderedState* rs, const FragmentState* frs, LightState* lightrs);
    static void release();
    static void invalidateShaderSourceCache();
    static void checkForShaderFileChanges();
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
};
