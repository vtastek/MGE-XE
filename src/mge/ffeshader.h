#pragma once

#include "proxydx/d3d8header.h"

#include <unordered_map>
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
    D3DCOLORVALUE diffuseMaterial;
    BYTE blendEnable, srcBlend, destBlend;
    BYTE alphaTest, alphaFunc, alphaRef;
    BYTE useLighting, useFog, matSrcDiffuse, matSrcEmissive;

    D3DPRIMITIVETYPE primType;
    UINT baseIndex, minIndex, vertCount, startIndex, primCount;
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
        DWORD hasNormal : 1;           // Has normal map texture (_nh)
        DWORD hasParam : 1;            // Has parameter texture (_param)

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
    static HLSLShader generateMWShaderHLSL(const ShaderKey& sk);
    static HLSLShader createPurpleErrorShader();
    static void captureAndDumpTexture(IDirect3DTexture9* texture);

public:
    static bool init(IDirect3DDevice* d, ID3DXEffectPool* pool);
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
};
