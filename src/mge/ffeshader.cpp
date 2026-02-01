
#include "ffeshader.h"
#include "tracy/Tracy.hpp"
#include "configuration.h"
#include "support/log.h"
#include "mwbridge.h"
#include "morrowindbsa.h"
#include "statusoverlay.h"
#include "distantland.h"
#include "imgui_manager.h"
#include "hlsl_shader_manager.h"

#include <algorithm>
#include <sstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>
#include <unordered_map>
#include <random>

using std::string;
using std::stringstream;
using std::unordered_map;

IDirect3DDevice* FixedFunctionShader::device;
ID3DXEffectPool* FixedFunctionShader::constantPool;
unordered_map<FixedFunctionShader::ShaderKey, ID3DXEffect*, FixedFunctionShader::ShaderKey::hasher> FixedFunctionShader::cacheEffects;
FixedFunctionShader::ShaderLRU FixedFunctionShader::shaderLRU;
ID3DXEffect* FixedFunctionShader::effectDefaultPurple;

D3DXHANDLE FixedFunctionShader::ehWorld, FixedFunctionShader::ehWorldView;
D3DXHANDLE FixedFunctionShader::ehVertexBlendState, FixedFunctionShader::ehVertexBlendPalette;
D3DXHANDLE FixedFunctionShader::ehTex0, FixedFunctionShader::ehTex1, FixedFunctionShader::ehTex2, FixedFunctionShader::ehTex3, FixedFunctionShader::ehTex4, FixedFunctionShader::ehTex5;
D3DXHANDLE FixedFunctionShader::ehMaterialDiffuse, FixedFunctionShader::ehMaterialAmbient, FixedFunctionShader::ehMaterialEmissive;
D3DXHANDLE FixedFunctionShader::ehLightSceneAmbient, FixedFunctionShader::ehLightSunDiffuse, FixedFunctionShader::ehLightDiffuse;
D3DXHANDLE FixedFunctionShader::ehLightSunDirection, FixedFunctionShader::ehLightPosition, FixedFunctionShader::ehLightAmbient;
D3DXHANDLE FixedFunctionShader::ehLightFalloffQuadratic, FixedFunctionShader::ehLightFalloffLinear, FixedFunctionShader::ehLightFalloffConstant;
D3DXHANDLE FixedFunctionShader::ehTexgenTransform, FixedFunctionShader::ehBumpMatrix, FixedFunctionShader::ehBumpLumiScaleBias;

float FixedFunctionShader::sunMultiplier, FixedFunctionShader::ambMultiplier;

// DXVK detection static variables
static bool dxvkDetectionCached = false;
static bool dxvkDetectionResult = false;

// HLSL Pipeline static variables
unordered_map<FixedFunctionShader::ShaderKey, FixedFunctionShader::HLSLShader, FixedFunctionShader::ShaderKey::hasher> FixedFunctionShader::cacheHLSLShaders;
FixedFunctionShader::HLSLShaderLRU FixedFunctionShader::hlslShaderLRU;
FixedFunctionShader::HLSLShader FixedFunctionShader::hlslShaderDefaultPurple;

// HLSL Render Dispatch Recording System
std::vector<FixedFunctionShader::HLSLRecordedCall> FixedFunctionShader::recordedCalls;
std::vector<FixedFunctionShader::HLSLRecordedCall> FixedFunctionShader::previousFrameCalls;
bool FixedFunctionShader::isRecording = false;
bool FixedFunctionShader::isReplaying = false;
bool FixedFunctionShader::manualRecordingControl = false;
bool FixedFunctionShader::recordingEnabled = true;
bool FixedFunctionShader::recordingCompletedThisFrame = false;
bool FixedFunctionShader::hiZBuiltThisFrame = false;
bool FixedFunctionShader::dumpRequested = false;

// Consistent matrices for entire recording session
D3DXMATRIX FixedFunctionShader::recordingDeviceView;
D3DXMATRIX FixedFunctionShader::recordingDeviceProj;
D3DXMATRIX FixedFunctionShader::recordingShadowViewproj[2];

// Bbox cache: persists across frames for fast object-space bbox lookup
std::unordered_map<FixedFunctionShader::MeshKey, FixedFunctionShader::ObjectSpaceBBox, FixedFunctionShader::MeshKeyHash> FixedFunctionShader::bboxCache;

// Bbox lookup: maps VB+IB to world-space bbox (rebuilt each frame from recordedCalls)
std::unordered_map<FixedFunctionShader::VBIBKey, FixedFunctionShader::ObjectSpaceBBox, FixedFunctionShader::VBIBKeyHash> FixedFunctionShader::bboxLookup;

// Visibility lookup: stores culling results from HLSL replay for depth pass to reuse
std::vector<int8_t> FixedFunctionShader::visibilityResults;

// Visibility lookup (map-based): stores culling results keyed by MeshKey for replay to reuse
// Uses OR-logic: if any instance of a mesh is visible, the key is marked visible
std::unordered_map<FixedFunctionShader::MeshKey, bool, FixedFunctionShader::MeshKeyHash> FixedFunctionShader::visibilityLookup;

// Software occlusion culler for CPU-based Hi-Z depth buffer generation
SoftwareOcclusionCuller FixedFunctionShader::softwareOcclusionCuller;

// Set of meshes rasterized as occluders - these must never be culled by Hi-Z
std::unordered_set<FixedFunctionShader::MeshKey, FixedFunctionShader::MeshKeyHash> FixedFunctionShader::rasterizedOccluderMeshes;

// Previous frame camera tracking for velocity-based bbox expansion
D3DXVECTOR3 FixedFunctionShader::prevCameraPos;
D3DXMATRIX FixedFunctionShader::prevCameraView;
bool FixedFunctionShader::hasPrevCamera = false;

std::shared_ptr<LightState> FixedFunctionShader::lastLightState;
const LightState* FixedFunctionShader::lastLightStatePtr = nullptr;

// Material state cache static member
FixedFunctionShader::MaterialStateCache FixedFunctionShader::materialCache;

// Texture binding cache static member
FixedFunctionShader::TextureBindingCache FixedFunctionShader::textureCache;

FixedFunctionShader::SavedRenderStates FixedFunctionShader::preRecordingState = {};

// Exterior texture binding optimization flags
bool FixedFunctionShader::isExteriorShadowBound = false;
bool FixedFunctionShader::isDetailTextureBound = false;

// Shadow matrix caching optimization
D3DXMATRIX FixedFunctionShader::cachedViewMatrix;
D3DXMATRIX FixedFunctionShader::cachedInverseView;
D3DXMATRIX FixedFunctionShader::cachedViewToShadow[2];
bool FixedFunctionShader::shadowMatricesValid = false;

// Default textures static members
IDirect3DTexture9* FixedFunctionShader::defaultWhiteTexture = nullptr;
IDirect3DTexture9* FixedFunctionShader::defaultBlackTexture = nullptr;
IDirect3DTexture9* FixedFunctionShader::defaultNormalTexture = nullptr;

// Original detail texture storage
IDirect3DBaseTexture9* FixedFunctionShader::savedOriginalDetailTexture = nullptr;

std::unordered_map<std::string, FixedFunctionShader::CachedShaderSource> FixedFunctionShader::shaderSourceCache;
bool FixedFunctionShader::needsCacheReset = false;

// Async compilation system static variables
std::queue<std::shared_ptr<FixedFunctionShader::AsyncShaderRequest>> FixedFunctionShader::compilationQueue;
std::mutex FixedFunctionShader::queueMutex;
std::condition_variable FixedFunctionShader::queueCondition;
std::thread FixedFunctionShader::compilerThread;
std::atomic<bool> FixedFunctionShader::shutdownCompiler(false);
std::unordered_map<FixedFunctionShader::ShaderKey, std::shared_ptr<FixedFunctionShader::AsyncShaderRequest>, FixedFunctionShader::ShaderKey::hasher> FixedFunctionShader::pendingCompilations;
std::unordered_map<FixedFunctionShader::VertexShaderKey, IDirect3DVertexShader9*, FixedFunctionShader::VertexShaderKey::hasher> FixedFunctionShader::vertexShaderCache;

// Current suffix texture flags for shader variant generation
struct SuffixTextureFlags {
    bool hasDiffParam = false;
    bool hasParamH = false;
    bool hasParamX = false;
    bool hasGrass = false;
} static suffixFlags;

// Cache for per-texture suffix flags to avoid repeated hash calculations
static std::unordered_map<IDirect3DTexture9*, SuffixTextureFlags> textureSuffixCache;

// Cache for texture resolutions to avoid repeated GetLevelDesc calls
static std::unordered_map<IDirect3DTexture9*, D3DXVECTOR2> textureResolutionCache;

// Texture resolution cache to avoid repeated expensive operations
struct TextureSuffixResolutionCache {
    BSA::TextureRuntimeHash hash;
    std::string textureName;
    bool hasValidName;
    const BSA::TextureSuffixVariants* variants;
    
    TextureSuffixResolutionCache() : hasValidName(false), variants(nullptr) {}
};
static std::unordered_map<IDirect3DTexture9*, TextureSuffixResolutionCache> textureSuffixResolutionCache;

// Suffix texture binding cache to prevent repeated binding operations
struct SuffixBindingState {
    IDirect3DTexture9* lastBaseTexture;  // Texture pointer for fast comparison
    std::string currentBaseTextureName;
    IDirect3DTexture9* boundDiffParam;
    IDirect3DTexture9* boundParamH;
    IDirect3DTexture9* boundParamX;
    
    SuffixBindingState() : lastBaseTexture(nullptr), boundDiffParam(nullptr), boundParamH(nullptr), boundParamX(nullptr) {}
};
static SuffixBindingState bindingCache;

// Get suffix flags for a specific texture (per-texture, not global)
static SuffixTextureFlags getSuffixFlagsForTexture(IDirect3DDevice9* device, IDirect3DTexture9* texture) {
    SuffixTextureFlags flags = {false, false, false, false};
    
    if (!texture || Configuration.PerPixelLightFlags != 2) {
        return flags;
    }
    
    // Check cache first
    auto cacheIt = textureSuffixCache.find(texture);
    if (cacheIt != textureSuffixCache.end()) {
        return cacheIt->second;  // Return cached result
    }
    
    // Not in cache - calculate hash and determine suffix flags (disable caching for unique hashes)
    BSA::TextureRuntimeHash texHash = BSA::calculateTextureHash(device, texture, false);
    
    if (texHash.crc32 != 0) {
        // Try to resolve texture name from hash
        const std::string* textureName = BSA::resolveTextureNameFromHash(texHash);
        if (textureName) {
            // Hash lookup successful
            LOG::logline("RUNTIME HASH MATCH: %08x -> %s", texHash.crc32, textureName->c_str());
            
            // Look up suffix variants for this specific texture
            const BSA::TextureSuffixVariants* variants = BSA::getTextureSuffixVariants(textureName->c_str());
            if (variants && (variants->hasDiffParam() || variants->hasParamH() || variants->hasParamX() || variants->hasGrass())) {
                // Set flags based on available suffix variants (combinations allowed)
                flags.hasDiffParam = variants->hasDiffParam() || variants->hasDiffParamT();
                flags.hasParamH = variants->hasParamH();
                flags.hasParamX = variants->hasParamX();
                flags.hasGrass = variants->hasGrass();
                
                // DEBUG: Log suffix selection decision and the actual variant paths
                //LOG::logline("DEBUG SUFFIX SELECTION: %s -> selected: diffparam=%d normal=%d param=%d", 
                //           textureName->c_str(), flags.hasDiffParam, flags.hasNormal, flags.hasParam);
                // LOG::logline("DEBUG AVAILABLE VARIANTS: diffparam='%s' normal='%s' param='%s'",
                //          variants->diffparam.c_str(), variants->normal.c_str(), variants->param.c_str());
                
                //const char* selectedType = flags.hasDiffParam ? "diffparam+normal" : 
                //                         (flags.hasParam ? "param+normal" : "normal-only");
                // LOG::logline("PER-TEXTURE SUFFIX: %s -> SELECTED: %s", textureName->c_str(), selectedType);
            }
        } else {
            // Hash lookup failed
            LOG::logline("RUNTIME HASH FAILED: %08x -> NO MATCH FOUND", texHash.crc32);
        }
    }
    
    // Cache the result
    textureSuffixCache[texture] = flags;
    
    return flags;
}

static string buildArgString(DWORD arg, const string& mask, const string& sampler);

// DXVK detection function
static bool isDXVK() {
    if (dxvkDetectionCached) {
        return dxvkDetectionResult;
    }

    dxvkDetectionCached = true;
    dxvkDetectionResult = false;

    // Get d3d9.dll version info to detect DXVK
    DWORD dwHandle = 0;
    DWORD dwSize = GetFileVersionInfoSizeA("d3d9.dll", &dwHandle);
    if (dwSize == 0) {
        return false;
    }

    std::vector<BYTE> versionInfo(dwSize);
    if (!GetFileVersionInfoA("d3d9.dll", dwHandle, dwSize, versionInfo.data())) {
        return false;
    }

    // Try multiple language/codepage combinations
    const char* langCodes[] = {
        "040904b0", // US English
        "040904E4", // US English (another variant)
        "04090000", // English (neutral)
        "000004b0", // Neutral language, US English codepage
        "00000000"  // Neutral language, neutral codepage
    };

    std::string productName;
    bool foundProductName = false;

    for (const char* langCode : langCodes) {
        std::string queryPath = std::string("\\StringFileInfo\\") + langCode + "\\ProductName";
        LPVOID lpBuffer = nullptr;
        UINT uLen = 0;

        if (VerQueryValueA(versionInfo.data(), queryPath.c_str(), &lpBuffer, &uLen) && lpBuffer && uLen > 0) {
            productName = static_cast<char*>(lpBuffer);
            foundProductName = true;
            break;
        }
    }

    if (!foundProductName) {
        // Try to enumerate available language/codepage combinations
        struct LANGANDCODEPAGE {
            WORD wLanguage;
            WORD wCodePage;
        } *lpTranslate;

        UINT cbTranslate = 0;
        if (VerQueryValueA(versionInfo.data(), "\\VarFileInfo\\Translation", (LPVOID*)&lpTranslate, &cbTranslate)) {
            for (size_t i = 0; i < (cbTranslate / sizeof(LANGANDCODEPAGE)); i++) {
                char langCode[9];
                std::snprintf(langCode, sizeof(langCode), "%04x%04x", lpTranslate[i].wLanguage, lpTranslate[i].wCodePage);

                std::string queryPath = std::string("\\StringFileInfo\\") + langCode + "\\ProductName";
                LPVOID lpBuffer = nullptr;
                UINT uLen = 0;

                if (VerQueryValueA(versionInfo.data(), queryPath.c_str(), &lpBuffer, &uLen) && lpBuffer && uLen > 0) {
                    productName = static_cast<char*>(lpBuffer);
                    foundProductName = true;
                    break;
                }
            }
        }

        if (!foundProductName) {
            return false;
        }
    }

    // Check if this is DXVK
    if (productName.find("DXVK") != std::string::npos) {
        dxvkDetectionResult = true;
        LOG::logline("-- DXVK detected, fast compilation enabled");
    }

    return dxvkDetectionResult;
}

bool FixedFunctionShader::init(IDirect3DDevice* d, ID3DXEffectPool* pool) {
    device = d;
    constantPool = pool;

    // Create last resort shader when a generated shader fails somehow
    const D3DXMACRO generateDefault[] = { "FFE_ERROR_MATERIAL", "", 0, 0 };
    ID3DXEffect* effect;
    ID3DXBuffer* errors;

    HRESULT hr = D3DXCreateEffectFromFile(device, "Data Files\\shaders\\core\\XE FixedFuncEmu.fx", generateDefault, 0, D3DXSHADER_OPTIMIZATION_LEVEL3|D3DXFX_LARGEADDRESSAWARE, constantPool, &effect, &errors);
    if (hr != D3D_OK) {
        if (errors) {
            LOG::write("!! Shader compile errors:\n");
            LOG::write(reinterpret_cast<const char*>(errors->GetBufferPointer()));
            LOG::write("\n");
            errors->Release();
        }
        return false;
    }

    // Use it to bind shared parameters too
    ehWorld = effect->GetParameterByName(0, "world");
    ehVertexBlendState = effect->GetParameterByName(0, "vertexBlendState");
    ehVertexBlendPalette = effect->GetParameterByName(0, "vertexBlendPalette");
    ehTex0 = effect->GetParameterByName(0, "tex0");
    ehTex1 = effect->GetParameterByName(0, "tex1");
    ehTex2 = effect->GetParameterByName(0, "tex2");
    ehTex3 = effect->GetParameterByName(0, "tex3");
    ehTex4 = effect->GetParameterByName(0, "tex4");
    ehTex5 = effect->GetParameterByName(0, "tex5");

    ehWorldView = effect->GetParameterByName(0, "worldview");
    ehMaterialDiffuse = effect->GetParameterByName(0, "materialDiffuse");
    ehMaterialAmbient = effect->GetParameterByName(0, "materialAmbient");
    ehMaterialEmissive = effect->GetParameterByName(0, "materialEmissive");
    ehLightSceneAmbient = effect->GetParameterByName(0, "lightSceneAmbient");
    ehLightSunDiffuse = effect->GetParameterByName(0, "lightSunDiffuse");
    ehLightSunDirection = effect->GetParameterByName(0, "lightSunDirection");
    ehLightDiffuse = effect->GetParameterByName(0, "lightDiffuse");
    ehLightAmbient = effect->GetParameterByName(0, "lightAmbient");
    ehLightPosition = effect->GetParameterByName(0, "lightPosition");
    ehLightFalloffQuadratic = effect->GetParameterByName(0, "lightFalloffQuadratic");
    ehLightFalloffLinear = effect->GetParameterByName(0, "lightFalloffLinear");
    ehLightFalloffConstant = effect->GetParameterByName(0, "lightFalloffConstant");
    ehTexgenTransform = effect->GetParameterByName(0, "texgenTransform");
    ehBumpMatrix = effect->GetParameterByName(0, "bumpMatrix");
    ehBumpLumiScaleBias = effect->GetParameterByName(0, "bumpLumiScaleBias");

    effectDefaultPurple = effect;
    sunMultiplier = ambMultiplier = 1.0;

    // Clear cache and LRU, important if the renderer resets
    shaderLRU.effect = nullptr;
    shaderLRU.last_sk = ShaderKey();
    cacheEffects.clear();

    // Initialize HLSL pipeline if enabled (PerPixelLightFlags == 2 means HLSL)
    if (Configuration.PerPixelLightFlags == 2) {
        LOG::logline("-- Initializing HLSL compilation pipeline");
        
        // Clear HLSL cache and LRU
        hlslShaderLRU.shader = {};
        hlslShaderLRU.last_sk = ShaderKey();
        cacheHLSLShaders.clear();
        
        // Create default error shader for HLSL pipeline
        hlslShaderDefaultPurple = createPurpleErrorShader();
        
        // Compile essential shaders synchronously to prevent startup regression
        LOG::logline("-- Compiling essential HLSL shaders synchronously");
        
        // Most basic variants needed for immediate rendering
        struct EssentialVariant {
            int lighting; int noPointLights; int vertexCol; int skinning;
            int hasDiffParam; int hasParamH; int hasParamX; int fogMode; int stages;
        };
        
        EssentialVariant essentials[] = {
            // Unlit base case
            {0, 1, 0, 0, 0, 0, 0, 1, 1},
            // Basic lit case (sun only, no vertex color, no skinning)
            {1, 1, 0, 0, 0, 0, 0, 1, 1},
            // Basic lit with vertex color
            {1, 1, 1, 0, 0, 0, 0, 1, 1},
        };
        
        for (const auto& variant : essentials) {
            ShaderKey sk;
            memset(&sk, 0, sizeof(sk));
            sk.uvSets = 1;
            sk.useLighting = variant.lighting;
            sk.noPointLights = variant.noPointLights;
            sk.vertexColour = variant.vertexCol;
            sk.vertexMaterial = variant.vertexCol + 1;
            sk.usesSkinning = variant.skinning;
            sk.hasDiffParam = variant.hasDiffParam;
            sk.hasParamH = variant.hasParamH;
            sk.hasParamX = variant.hasParamX;
            sk.fogMode = variant.fogMode;
            sk.activeStages = variant.stages;
            sk.hasShadows = ((Configuration.MGEFlags & USE_SHADOWS) && (Configuration.MGEFlags & USE_DISTANT_LAND)) ? 1 : 0;
            
            HLSLShader shader = generateMWShaderHLSL(sk);
            if (shader.vertexShader && shader.pixelShader) {
                cacheHLSLShaders[sk] = shader;
                LOG::logline("-- Essential HLSL shader compiled: lighting=%d noPointLights=%d vertexCol=%d",
                            variant.lighting, variant.noPointLights, variant.vertexCol);
            }
        }
        
        // Start async compiler for on-demand compilation of remaining variants
        startAsyncCompiler();
    }

    // Start shader precaching immediately after init - moved earlier for faster startup
    if (Configuration.MGEFlags & USE_FFESHADER) {
        LOG::logline("-- Starting early shader precaching");
        precacheAsync();
    }

    // Create default textures to avoid null binds that cause DXVK descriptor updates
    createDefaultTextures();

    // Initialize software occlusion culler for CPU-based Hi-Z depth buffer
    D3DDISPLAYMODE dm;
    device->GetDisplayMode(0, &dm);

    // Use current resolution or reasonable default
    UINT width = (dm.Width > 0) ? dm.Width : 1920;
    UINT height = (dm.Height > 0) ? dm.Height : 1080;

    // Cap Hi-Z buffer at 256 horizontal resolution (mip 0 = half-res = 128)
    // This significantly reduces rasterization cost while maintaining good culling
    const UINT MAX_HIZ_WIDTH = 256;
    if (width > MAX_HIZ_WIDTH) {
        float scale = (float)MAX_HIZ_WIDTH / (float)width;
        height = (UINT)(height * scale);
        width = MAX_HIZ_WIDTH;
    }

    softwareOcclusionCuller.init(width, height);
    LOG::logline("-- Software occlusion culler initialized (%dx%d)", width, height);

    return true;
}

void FixedFunctionShader::startEarlyPrecache(IDirect3DDevice* d) {
    // Set device early and start immediate precaching
    if (!device && d) {
        device = d;
        LOG::logline("-- Starting immediate HLSL shader precaching");
        
        // Start the full precaching immediately in the background
        // This runs the same code as precacheAsync but starts much earlier
        std::thread immediateThread([]() {
            bool hlslMode = (Configuration.PerPixelLightFlags == 2);
            
            if (hlslMode) {
                LOG::logline("-- Immediate HLSL shader precaching started");
                
                int hlslVariants = 0;
                
                // Progress tracking for status overlay
                auto updateStatus = [&](int current, int total) {
                    char progressText[128];
                    std::snprintf(progressText, sizeof(progressText), "Compiling HLSL shaders: %d/%d", current, total);
                    StatusOverlay::setStatus(progressText);
                };
                
                // Same essential variants as in the main precaching
                struct ShaderVariant {
                    int lighting;
                    int noPointLights;
                    int vertexCol;
                    int skinning;
                    int hasDiffParam;
                    int hasParamH;
                    int hasParamX;
                    int fogMode;
                    int stages;
                };
                
                ShaderVariant variants[] = {
                    // Basic unlit variants (2) - format: {lighting, noPointLights, vertexCol, skinning, hasDiffParam, hasParamH, hasParamX, fogMode, stages}
                    {0, 1, 0, 0, 0, 0, 0, 1, 1}, {0, 1, 1, 0, 0, 0, 0, 1, 1},

                    // Basic lit variants - sun only (4)
                    {1, 1, 0, 0, 0, 0, 0, 1, 1}, {1, 1, 1, 0, 0, 0, 0, 1, 1}, {1, 1, 0, 1, 0, 0, 0, 1, 1}, {1, 1, 1, 1, 0, 0, 0, 1, 1},

                    // Lit with point lights (4)
                    {1, 0, 0, 0, 0, 0, 0, 1, 1}, {1, 0, 1, 0, 0, 0, 0, 1, 1}, {1, 0, 0, 1, 0, 0, 0, 1, 1}, {1, 0, 1, 1, 0, 0, 0, 1, 1},

                    // Diffparam variants - most important for terrain (8)
                    {1, 1, 0, 0, 1, 0, 0, 1, 1}, {1, 1, 1, 0, 1, 0, 0, 1, 1}, {1, 0, 0, 0, 1, 0, 0, 1, 1}, {1, 0, 1, 0, 1, 0, 0, 1, 1},
                    {1, 1, 0, 1, 1, 0, 0, 1, 1}, {1, 1, 1, 1, 1, 0, 0, 1, 1}, {1, 0, 0, 1, 1, 0, 0, 1, 1}, {1, 0, 1, 1, 1, 0, 0, 1, 1},

                    // ParamH variants (8) - replaces normal map variants
                    {1, 1, 0, 0, 0, 1, 0, 1, 1}, {1, 1, 1, 0, 0, 1, 0, 1, 1}, {1, 0, 0, 0, 0, 1, 0, 1, 1}, {1, 0, 1, 0, 0, 1, 0, 1, 1},
                    {1, 1, 0, 1, 0, 1, 0, 1, 1}, {1, 1, 1, 1, 0, 1, 0, 1, 1}, {1, 0, 0, 1, 0, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 1, 0, 1, 1},

                    // Diffparam + ParamH combinations (8)
                    {1, 1, 0, 0, 1, 1, 0, 1, 1}, {1, 1, 1, 0, 1, 1, 0, 1, 1}, {1, 0, 0, 0, 1, 1, 0, 1, 1}, {1, 0, 1, 0, 1, 1, 0, 1, 1},
                    {1, 1, 0, 1, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 1, 1, 0, 1, 1}, {1, 0, 0, 1, 1, 1, 0, 1, 1}, {1, 0, 1, 1, 1, 1, 0, 1, 1},

                    // ParamX variants (8) - replaces param variants
                    {1, 1, 0, 0, 0, 0, 1, 1, 1}, {1, 1, 1, 0, 0, 0, 1, 1, 1}, {1, 0, 0, 0, 0, 0, 1, 1, 1}, {1, 0, 1, 0, 0, 0, 1, 1, 1},
                    {1, 1, 0, 1, 0, 0, 1, 1, 1}, {1, 1, 1, 1, 0, 0, 1, 1, 1}, {1, 0, 0, 1, 0, 0, 1, 1, 1}, {1, 0, 1, 1, 0, 0, 1, 1, 1},

                    // Full combination variants - diffparam + paramh + paramx (8)
                    {1, 1, 0, 0, 1, 1, 1, 1, 1}, {1, 1, 1, 0, 1, 1, 1, 1, 1}, {1, 0, 0, 0, 1, 1, 1, 1, 1}, {1, 0, 1, 0, 1, 1, 1, 1, 1},
                    {1, 1, 0, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 0, 0, 1, 1, 1, 1, 1, 1}, {1, 0, 1, 1, 1, 1, 1, 1, 1},
                    
                    // Dual texture variants - basic (8)
                    {1, 1, 0, 0, 0, 0, 0, 1, 2}, {1, 1, 1, 0, 0, 0, 0, 1, 2}, {1, 0, 0, 0, 0, 0, 0, 1, 2}, {1, 0, 1, 0, 0, 0, 0, 1, 2},
                    {1, 1, 0, 1, 0, 0, 0, 1, 2}, {1, 1, 1, 1, 0, 0, 0, 1, 2}, {1, 0, 0, 1, 0, 0, 0, 1, 2}, {1, 0, 1, 1, 0, 0, 0, 1, 2},
                    
                    // Dual texture + diffparam (8) 
                    {1, 1, 0, 0, 1, 0, 0, 1, 2}, {1, 1, 1, 0, 1, 0, 0, 1, 2}, {1, 0, 0, 0, 1, 0, 0, 1, 2}, {1, 0, 1, 0, 1, 0, 0, 1, 2},
                    {1, 1, 0, 1, 1, 0, 0, 1, 2}, {1, 1, 1, 1, 1, 0, 0, 1, 2}, {1, 0, 0, 1, 1, 0, 0, 1, 2}, {1, 0, 1, 1, 1, 0, 0, 1, 2},
                    
                    // Dual texture + normal (8)
                    {1, 1, 0, 0, 0, 1, 0, 1, 2}, {1, 1, 1, 0, 0, 1, 0, 1, 2}, {1, 0, 0, 0, 0, 1, 0, 1, 2}, {1, 0, 1, 0, 0, 1, 0, 1, 2},
                    {1, 1, 0, 1, 0, 1, 0, 1, 2}, {1, 1, 1, 1, 0, 1, 0, 1, 2}, {1, 0, 0, 1, 0, 1, 0, 1, 2}, {1, 0, 1, 1, 0, 1, 0, 1, 2},
                    
                    // No fog variants (8)
                    {1, 1, 0, 0, 0, 0, 0, 0, 1}, {1, 1, 1, 0, 0, 0, 0, 0, 1}, {1, 0, 0, 0, 0, 0, 0, 0, 1}, {1, 0, 1, 0, 0, 0, 0, 0, 1},
                    {1, 1, 0, 0, 1, 0, 0, 0, 1}, {1, 1, 1, 0, 1, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {1, 0, 1, 0, 1, 0, 0, 0, 1},
                    
                    // Fog mode 2 variants (8)
                    {1, 1, 0, 0, 0, 0, 0, 2, 1}, {1, 1, 1, 0, 0, 0, 0, 2, 1}, {1, 0, 0, 0, 0, 0, 0, 2, 1}, {1, 0, 1, 0, 0, 0, 0, 2, 1},
                    {1, 1, 0, 0, 1, 0, 0, 2, 1}, {1, 1, 1, 0, 1, 0, 0, 2, 1}, {1, 0, 0, 0, 1, 0, 0, 2, 1}, {1, 0, 1, 0, 1, 0, 0, 2, 1},
                    
                    // Special cases - no fog, dual texture (8)
                    {1, 1, 0, 0, 0, 0, 0, 0, 2}, {1, 1, 1, 0, 0, 0, 0, 0, 2}, {1, 0, 0, 0, 0, 0, 0, 0, 2}, {1, 0, 1, 0, 0, 0, 0, 0, 2},
                    {1, 1, 0, 1, 0, 0, 0, 0, 2}, {1, 1, 1, 1, 0, 0, 0, 0, 2}, {1, 0, 0, 1, 0, 0, 0, 0, 2}, {1, 0, 1, 1, 0, 0, 0, 0, 2},
                    
                    // Fog mode 2, dual texture (8)
                    {1, 1, 0, 0, 0, 0, 0, 2, 2}, {1, 1, 1, 0, 0, 0, 0, 2, 2}, {1, 0, 0, 0, 0, 0, 0, 2, 2}, {1, 0, 1, 0, 0, 0, 0, 2, 2},
                    {1, 1, 0, 1, 0, 0, 0, 2, 2}, {1, 1, 1, 1, 0, 0, 0, 2, 2}, {1, 0, 0, 1, 0, 0, 0, 2, 2}, {1, 0, 1, 1, 0, 0, 0, 2, 2},
                    
                    // Additional edge cases for complete coverage (4)  
                    {0, 1, 0, 1, 0, 0, 0, 1, 1}, {0, 1, 1, 1, 0, 0, 0, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 1, 1}, {0, 0, 1, 0, 0, 0, 0, 1, 1},
                    
                    // CRITICAL FIX: Add the 8 universal base combinations that the fallback system expects
                    // These are guaranteed to be cached and should always be available for fallback
                    // Format: {lighting, noPointLights, vertexCol, skinning, hasDiffParam, hasNormal, hasParam, fogMode, stages}
                    {1, 1, 0, 0, 0, 0, 0, 1, 1}, // lit, no points, no vertcol, no skinning, no suffixes
                    {1, 1, 1, 0, 0, 0, 0, 1, 1}, // lit, no points, vertcol, no skinning, no suffixes  
                    {1, 1, 0, 1, 0, 0, 0, 1, 1}, // lit, no points, no vertcol, skinning, no suffixes
                    {1, 1, 1, 1, 0, 0, 0, 1, 1}, // lit, no points, vertcol, skinning, no suffixes
                    {1, 0, 0, 0, 0, 0, 0, 1, 1}, // lit, points, no vertcol, no skinning, no suffixes
                    {1, 0, 1, 0, 0, 0, 0, 1, 1}, // lit, points, vertcol, no skinning, no suffixes
                    {1, 0, 0, 1, 0, 0, 0, 1, 1}, // lit, points, no vertcol, skinning, no suffixes
                    {1, 0, 1, 1, 0, 0, 0, 1, 1}, // lit, points, vertcol, skinning, no suffixes
                };
                
                const int totalVariants = sizeof(variants) / sizeof(variants[0]);
                LOG::logline("-- Immediate precaching %d essential HLSL shader variants", totalVariants);
                
                for (int i = 0; i < totalVariants; i++) {
                    const auto& v = variants[i];
                    
                    // Update progress every few variants
                    if (i % 5 == 0 || i == totalVariants - 1) {
                        updateStatus(i + 1, totalVariants);
                    }
                    
                    ShaderKey sk;
                    memset(&sk, 0, sizeof(sk));
                    sk.uvSets = 1;
                    sk.useLighting = v.lighting;
                    sk.noPointLights = v.noPointLights;
                    sk.vertexColour = v.vertexCol;
                    sk.vertexMaterial = v.vertexCol + 1;
                    sk.usesSkinning = v.skinning;
                    sk.hasDiffParam = v.hasDiffParam;
                    sk.hasParamH = v.hasParamH;
                    sk.hasParamX = v.hasParamX;
                    sk.hasGrass = 0; // Precache without grass specific variants
                    sk.fogMode = v.fogMode;
                    sk.activeStages = v.stages;
                    
                    // Standard texture stage setup
                    sk.stage[0] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 1, 0, 0, 0 };
                    if (v.stages > 1) {
                        sk.stage[1] = { D3DTOP_ADD, D3DTA_TEXTURE, D3DTA_CURRENT, D3DTA_CURRENT, 0, 0, 0, 0 };
                    }
                    memset(&sk.stage[v.stages], 0, sizeof(sk.stage[0]) * (8 - v.stages));
                    
                    // Compile if not already cached
                    if (cacheHLSLShaders.find(sk) == cacheHLSLShaders.end()) {
                        cacheHLSLShaders[sk] = generateMWShaderHLSL(sk);
                        hlslVariants++;
                    }
                }
                
                StatusOverlay::setStatus("Immediate HLSL shader precaching complete");
                LOG::logline("-- Immediate HLSL precaching completed: %d shaders compiled", hlslVariants);
            }
        });
        
        immediateThread.detach();
    }
}

void FixedFunctionShader::precacheAsync() {
    // Move precaching to a separate thread - essential variants to prevent stuttering
    std::thread precacheThread([]() {
        // Check lighting mode to prioritize compilation order
        bool hlslMode = (Configuration.PerPixelLightFlags == 2);
        
        // If HLSL is current mode, compile HLSL shaders first for better startup performance
        // Skip if immediate precaching already handled the essential variants
        if (hlslMode && cacheHLSLShaders.size() < 20) {
            LOG::logline("-- Starting HLSL shader precaching (supplemental)");
            
            int hlslVariants = 0;
            
            // Progress tracking for status overlay
            auto updateStatus = [&](int current, int total) {
                char progressText[128];
                std::snprintf(progressText, sizeof(progressText), "Compiling HLSL shaders: %d/%d", current, total);
                StatusOverlay::setStatus(progressText);
            };
            
                // Essential variants that cover most real-world cases
                struct ShaderVariant {
                    int lighting;
                    int noPointLights; 
                    int vertexCol;
                    int skinning;
                    int hasDiffParam;
                    int hasParamH;
                    int hasParamX;
                    int fogMode;
                    int stages;
                };
                
                ShaderVariant variants[] = {
                    // Same 108 variants as immediate precaching - keep both in sync
                    // Basic unlit variants (2)
                    {0, 1, 0, 0, 0, 0, 0, 1, 1}, {0, 1, 1, 0, 0, 0, 0, 1, 1},
                    
                    // Basic lit variants - sun only (4)
                    {1, 1, 0, 0, 0, 0, 0, 1, 1}, {1, 1, 1, 0, 0, 0, 0, 1, 1}, {1, 1, 0, 1, 0, 0, 0, 1, 1}, {1, 1, 1, 1, 0, 0, 0, 1, 1},
                    
                    // Lit with point lights (4)
                    {1, 0, 0, 0, 0, 0, 0, 1, 1}, {1, 0, 1, 0, 0, 0, 0, 1, 1}, {1, 0, 0, 1, 0, 0, 0, 1, 1}, {1, 0, 1, 1, 0, 0, 0, 1, 1},
                    
                    // Diffparam variants - most important for terrain (8)
                    {1, 1, 0, 0, 1, 0, 0, 1, 1}, {1, 1, 1, 0, 1, 0, 0, 1, 1}, {1, 0, 0, 0, 1, 0, 0, 1, 1}, {1, 0, 1, 0, 1, 0, 0, 1, 1},
                    {1, 1, 0, 1, 1, 0, 0, 1, 1}, {1, 1, 1, 1, 1, 0, 0, 1, 1}, {1, 0, 0, 1, 1, 0, 0, 1, 1}, {1, 0, 1, 1, 1, 0, 0, 1, 1},
                    
                    // Normal map variants (8)
                    {1, 1, 0, 0, 0, 1, 0, 1, 1}, {1, 1, 1, 0, 0, 1, 0, 1, 1}, {1, 0, 0, 0, 0, 1, 0, 1, 1}, {1, 0, 1, 0, 0, 1, 0, 1, 1},
                    {1, 1, 0, 1, 0, 1, 0, 1, 1}, {1, 1, 1, 1, 0, 1, 0, 1, 1}, {1, 0, 0, 1, 0, 1, 0, 1, 1}, {1, 0, 1, 1, 0, 1, 0, 1, 1},
                    
                    // Diffparam + normal combinations (8)
                    {1, 1, 0, 0, 1, 1, 0, 1, 1}, {1, 1, 1, 0, 1, 1, 0, 1, 1}, {1, 0, 0, 0, 1, 1, 0, 1, 1}, {1, 0, 1, 0, 1, 1, 0, 1, 1},
                    {1, 1, 0, 1, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 1, 1, 0, 1, 1}, {1, 0, 0, 1, 1, 1, 0, 1, 1}, {1, 0, 1, 1, 1, 1, 0, 1, 1},
                    
                    // Param variants (8)
                    {1, 1, 0, 0, 0, 0, 1, 1, 1}, {1, 1, 1, 0, 0, 0, 1, 1, 1}, {1, 0, 0, 0, 0, 0, 1, 1, 1}, {1, 0, 1, 0, 0, 0, 1, 1, 1},
                    {1, 1, 0, 1, 0, 0, 1, 1, 1}, {1, 1, 1, 1, 0, 0, 1, 1, 1}, {1, 0, 0, 1, 0, 0, 1, 1, 1}, {1, 0, 1, 1, 0, 0, 1, 1, 1},
                    
                    // Full combination variants (8)
                    {1, 1, 0, 0, 1, 1, 1, 1, 1}, {1, 1, 1, 0, 1, 1, 1, 1, 1}, {1, 0, 0, 0, 1, 1, 1, 1, 1}, {1, 0, 1, 0, 1, 1, 1, 1, 1},
                    {1, 1, 0, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 0, 0, 1, 1, 1, 1, 1, 1}, {1, 0, 1, 1, 1, 1, 1, 1, 1},
                    
                    // Dual texture variants - basic (8)
                    {1, 1, 0, 0, 0, 0, 0, 1, 2}, {1, 1, 1, 0, 0, 0, 0, 1, 2}, {1, 0, 0, 0, 0, 0, 0, 1, 2}, {1, 0, 1, 0, 0, 0, 0, 1, 2},
                    {1, 1, 0, 1, 0, 0, 0, 1, 2}, {1, 1, 1, 1, 0, 0, 0, 1, 2}, {1, 0, 0, 1, 0, 0, 0, 1, 2}, {1, 0, 1, 1, 0, 0, 0, 1, 2},
                    
                    // Dual texture + diffparam (8) 
                    {1, 1, 0, 0, 1, 0, 0, 1, 2}, {1, 1, 1, 0, 1, 0, 0, 1, 2}, {1, 0, 0, 0, 1, 0, 0, 1, 2}, {1, 0, 1, 0, 1, 0, 0, 1, 2},
                    {1, 1, 0, 1, 1, 0, 0, 1, 2}, {1, 1, 1, 1, 1, 0, 0, 1, 2}, {1, 0, 0, 1, 1, 0, 0, 1, 2}, {1, 0, 1, 1, 1, 0, 0, 1, 2},
                    
                    // Dual texture + normal (8)
                    {1, 1, 0, 0, 0, 1, 0, 1, 2}, {1, 1, 1, 0, 0, 1, 0, 1, 2}, {1, 0, 0, 0, 0, 1, 0, 1, 2}, {1, 0, 1, 0, 0, 1, 0, 1, 2},
                    {1, 1, 0, 1, 0, 1, 0, 1, 2}, {1, 1, 1, 1, 0, 1, 0, 1, 2}, {1, 0, 0, 1, 0, 1, 0, 1, 2}, {1, 0, 1, 1, 0, 1, 0, 1, 2},
                    
                    // No fog variants (8)
                    {1, 1, 0, 0, 0, 0, 0, 0, 1}, {1, 1, 1, 0, 0, 0, 0, 0, 1}, {1, 0, 0, 0, 0, 0, 0, 0, 1}, {1, 0, 1, 0, 0, 0, 0, 0, 1},
                    {1, 1, 0, 0, 1, 0, 0, 0, 1}, {1, 1, 1, 0, 1, 0, 0, 0, 1}, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {1, 0, 1, 0, 1, 0, 0, 0, 1},
                    
                    // Fog mode 2 variants (8)
                    {1, 1, 0, 0, 0, 0, 0, 2, 1}, {1, 1, 1, 0, 0, 0, 0, 2, 1}, {1, 0, 0, 0, 0, 0, 0, 2, 1}, {1, 0, 1, 0, 0, 0, 0, 2, 1},
                    {1, 1, 0, 0, 1, 0, 0, 2, 1}, {1, 1, 1, 0, 1, 0, 0, 2, 1}, {1, 0, 0, 0, 1, 0, 0, 2, 1}, {1, 0, 1, 0, 1, 0, 0, 2, 1},
                    
                    // Special cases - no fog, dual texture (8)
                    {1, 1, 0, 0, 0, 0, 0, 0, 2}, {1, 1, 1, 0, 0, 0, 0, 0, 2}, {1, 0, 0, 0, 0, 0, 0, 0, 2}, {1, 0, 1, 0, 0, 0, 0, 0, 2},
                    {1, 1, 0, 1, 0, 0, 0, 0, 2}, {1, 1, 1, 1, 0, 0, 0, 0, 2}, {1, 0, 0, 1, 0, 0, 0, 0, 2}, {1, 0, 1, 1, 0, 0, 0, 0, 2},
                    
                    // Fog mode 2, dual texture (8)
                    {1, 1, 0, 0, 0, 0, 0, 2, 2}, {1, 1, 1, 0, 0, 0, 0, 2, 2}, {1, 0, 0, 0, 0, 0, 0, 2, 2}, {1, 0, 1, 0, 0, 0, 0, 2, 2},
                    {1, 1, 0, 1, 0, 0, 0, 2, 2}, {1, 1, 1, 1, 0, 0, 0, 2, 2}, {1, 0, 0, 1, 0, 0, 0, 2, 2}, {1, 0, 1, 1, 0, 0, 0, 2, 2},
                    
                    // Additional edge cases for complete coverage (4)
                    {0, 1, 0, 1, 0, 0, 0, 1, 1}, {0, 1, 1, 1, 0, 0, 0, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 1, 1}, {0, 0, 1, 0, 0, 0, 0, 1, 1},
                    
                    // CRITICAL FIX: Add the 8 universal base combinations that the fallback system expects
                    // These are guaranteed to be cached and should always be available for fallback
                    // Format: {lighting, noPointLights, vertexCol, skinning, hasDiffParam, hasNormal, hasParam, fogMode, stages}
                    {1, 1, 0, 0, 0, 0, 0, 1, 1}, // lit, no points, no vertcol, no skinning, no suffixes
                    {1, 1, 1, 0, 0, 0, 0, 1, 1}, // lit, no points, vertcol, no skinning, no suffixes  
                    {1, 1, 0, 1, 0, 0, 0, 1, 1}, // lit, no points, no vertcol, skinning, no suffixes
                    {1, 1, 1, 1, 0, 0, 0, 1, 1}, // lit, no points, vertcol, skinning, no suffixes
                    {1, 0, 0, 0, 0, 0, 0, 1, 1}, // lit, points, no vertcol, no skinning, no suffixes
                    {1, 0, 1, 0, 0, 0, 0, 1, 1}, // lit, points, vertcol, no skinning, no suffixes
                    {1, 0, 0, 1, 0, 0, 0, 1, 1}, // lit, points, no vertcol, skinning, no suffixes
                    {1, 0, 1, 1, 0, 0, 0, 1, 1}, // lit, points, vertcol, skinning, no suffixes
                };
            
            const int totalVariants = sizeof(variants) / sizeof(variants[0]);
            LOG::logline("-- Precaching %d essential HLSL shader variants", totalVariants);
            
            for (int i = 0; i < totalVariants; i++) {
                const ShaderVariant& v = variants[i];
                
                // Update progress every few variants
                if (i % 5 == 0 || i == totalVariants - 1) {
                    updateStatus(i + 1, totalVariants);
                }
                
                ShaderKey sk;
                memset(&sk, 0, sizeof(sk));
                sk.uvSets = 1;
                sk.useLighting = v.lighting;
                sk.noPointLights = v.noPointLights;
                sk.vertexColour = v.vertexCol;
                sk.vertexMaterial = v.vertexCol + 1;
                sk.usesSkinning = v.skinning;
                sk.hasDiffParam = v.hasDiffParam;
                sk.hasGrass = 0; // Precache without grass specific variants
                sk.fogMode = v.fogMode;
                sk.activeStages = v.stages;
                
                // Standard texture stage setup
                sk.stage[0] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 1, 0, 0, 0 };
                if (v.stages > 1) {
                    sk.stage[1] = { D3DTOP_ADD, D3DTA_TEXTURE, D3DTA_CURRENT, D3DTA_CURRENT, 0, 0, 0, 0 };
                }
                memset(&sk.stage[v.stages], 0, sizeof(sk.stage[0]) * (8 - v.stages));
                
                // Compile if not already cached
                if (cacheHLSLShaders.find(sk) == cacheHLSLShaders.end()) {
                    cacheHLSLShaders[sk] = generateMWShaderHLSL(sk);
                    hlslVariants++;
                }
            }
            
            StatusOverlay::setStatus("HLSL shader precaching complete");
            LOG::logline("-- HLSL precaching completed: %d shaders compiled", hlslVariants);
        }
        
        LOG::logline("-- Starting collapsed HLSL shader precaching (optimized permutations)");

        // Collapsed precaching using modern engine techniques:
        // 1. Lighting loop collapse: One shader handles 0-N lights dynamically
        // 2. Unified param flavors: All texture suffixes handled with #ifdef guards  
        // 3. Skinning in VS only: Pixel shader doesn't care about skinning
        // 4. Runtime normal selection: Shader samples normal or derives from height

        struct CoreShaderVariant {
            bool lighting, vertCol, skinning;
            const char* desc;
        };
        
        CoreShaderVariant coreVariants[] = {
            {true, false, false, "lit"},
            {true, true, false, "lit+vertColor"}, 
            {true, false, true, "lit+skinned"},
            {true, true, true, "lit+vertColor+skinned"},
            {false, false, false, "unlit"},
            {false, true, false, "unlit+vertColor"},
            {false, false, true, "unlit+skinned"}, 
            {false, true, true, "unlit+vertColor+skinned"}
        };

        int compiledVariants = 0;
        
        for (auto& variant : coreVariants) {
            // Generate both standard fog (fogMode=1) and alpha blend fog (fogMode=2) variants
            for (int fogMode = 1; fogMode <= 2; ++fogMode) {
                ShaderKey sk;
                memset(&sk, 0, sizeof(sk));
                
                // Core shader properties
                sk.uvSets = 1;
                sk.useLighting = variant.lighting;
                sk.heavyLighting = 0;
                sk.vertexColour = variant.vertCol;
                sk.vertexMaterial = variant.vertCol + 1;  
                sk.usesSkinning = variant.skinning;
                
                // Lighting collapse: Use dynamic loop (no separate shaders for light counts)
                sk.noPointLights = 0; // Always allow point lights, handled dynamically
                
                // Param flavor unification: Always enable all texture suffixes
                // Shader uses #ifdef HAS_DIFFPARAM, #ifdef HAS_NORMAL, #ifdef HAS_PARAM
                // Note: These are compile-time defines for the collapsed shader that handles all combinations
                sk.hasDiffParam = 1;
                sk.hasGrass = 0; // Let grass shaders compile on demand
                
                // Fog mode: 1=standard, 2=alpha blending (diffparam textures)
                sk.fogMode = fogMode;
                
                // Standard texture stage
                sk.activeStages = 1;
                sk.usesTexgen = 0;
                sk.stage[0] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 1, 0, 0, 0 };
                memset(&sk.stage[1], 0, sizeof(sk.stage[1]));
                
                cacheHLSLShaders[sk] = generateMWShaderHLSL(sk);
                compiledVariants++;
                
                const char* fogDesc = (fogMode == 2) ? "+alphaBlend" : "";
                LOG::logline("-- Compiled collapsed shader: %s%s", variant.desc, fogDesc);
            }
        }

        LOG::logline("-- Collapsed HLSL precaching completed: %d core shaders compiled (reduced from 120+ permutations)", compiledVariants);
    });

    precacheThread.detach();
}

void FixedFunctionShader::updateLighting(float sunMult, float ambMult) {
    sunMultiplier = sunMult;
    ambMultiplier = ambMult;
}

void FixedFunctionShader::renderMorrowind(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, int recordMWIdx) {
    // Use HLSL pipeline if mode is set to HLSL (PerPixelLightFlags == 2)
    if (Configuration.PerPixelLightFlags == 2) {
        renderMorrowindHLSL(rs, frs, lightrs, recordMWIdx);
        return;
    }
    
    ID3DXEffect* effectFFE;

    // Check if state matches last used effect
    ShaderKey sk(rs, frs, lightrs);

    if (sk == shaderLRU.last_sk) {
        effectFFE = shaderLRU.effect;
    } else {
        // Read from shader cache / generate
        decltype(cacheEffects)::const_iterator iEffect = cacheEffects.find(sk);

        if (iEffect != cacheEffects.end()) {
            effectFFE = iEffect->second;
        } else {
            effectFFE = generateMWShader(sk);
        }

        shaderLRU.effect = effectFFE;
        shaderLRU.last_sk = sk;
    }

    // Set up material
    effectFFE->SetVector(ehMaterialDiffuse, (D3DXVECTOR4*)&frs->material.diffuse);
    effectFFE->SetVector(ehMaterialAmbient, (D3DXVECTOR4*)&frs->material.ambient);
    effectFFE->SetVector(ehMaterialEmissive, (D3DXVECTOR4*)&frs->material.emissive);

    // Set up lighting
    const size_t MaxLights = 8;
    D3DXVECTOR4 bufferDiffuse[MaxLights];
    float bufferAmbient[MaxLights];
    float bufferPosition[3 * MaxLights];
    float bufferFalloffQuadratic[MaxLights], bufferFalloffLinear[MaxLights], bufferFalloffConstant;

    memset(&bufferDiffuse, 0, sizeof(bufferDiffuse));
    memset(&bufferAmbient, 0, sizeof(bufferAmbient));
    memset(&bufferPosition, 0, sizeof(bufferPosition));
    memset(&bufferFalloffQuadratic, 0, sizeof(bufferFalloffQuadratic));
    memset(&bufferFalloffLinear, 0, sizeof(bufferFalloffLinear));
    bufferFalloffConstant = 0.33;

    // Check each active light
    RGBVECTOR sunDiffuse(0, 0, 0), ambient = lightrs->globalAmbient;
    size_t n = std::min(lightrs->active.size(), MaxLights), pointLightCount = 0;
    for (; n --> 0; ) {
        DWORD i = lightrs->active[n];
        const LightState::Light* light = &lightrs->lights.find(i)->second;

        // Transform to view space if not transformed this frame
        if (lightrs->lightsTransformed.find(i) == lightrs->lightsTransformed.end()) {
            if (light->type == D3DLIGHT_DIRECTIONAL) {
                D3DXVec3TransformNormal((D3DXVECTOR3*)&light->viewspacePos, (D3DXVECTOR3*)&light->position, &rs->viewTransform);
            } else {
                D3DXVec3TransformCoord((D3DXVECTOR3*)&light->viewspacePos, (D3DXVECTOR3*)&light->position, &rs->viewTransform);
            }

            lightrs->lightsTransformed[i] = true;
        }

        if (light->type == D3DLIGHT_POINT) {
            memcpy(&bufferDiffuse[pointLightCount], &light->diffuse, sizeof(light->diffuse));

            // Scatter position vectors for vectorization
            bufferPosition[pointLightCount] = light->viewspacePos.x;
            bufferPosition[pointLightCount + MaxLights] = light->viewspacePos.y;
            bufferPosition[pointLightCount + 2*MaxLights] = light->viewspacePos.z;

            // Scatter attenuation factors for vectorization
            if (light->falloff.x > 0) {
                // Standard point light source (falloffConstant doesn't vary per light)
                bufferFalloffConstant = light->falloff.x;
                bufferFalloffLinear[pointLightCount] = light->falloff.y;
                bufferFalloffQuadratic[pointLightCount] = light->falloff.z;
            } else if (light->falloff.z > 0) {
                // Probably a magic light source patched by Morrowind Code Patch
                // Patched falloff calculation is quadratic only, which needs to be
                // modified to account for the standard falloffConstant
                // Diffuse colour is correctly specified with the patch
                // Some overbrightness is applied to diffuse to cause glowing
                bufferDiffuse[pointLightCount].x *= bufferFalloffConstant;
                bufferDiffuse[pointLightCount].y *= bufferFalloffConstant;
                bufferDiffuse[pointLightCount].z *= bufferFalloffConstant;
                bufferAmbient[pointLightCount] = 1.0f + 1e-4f / sqrt(light->falloff.z);
                bufferFalloffQuadratic[pointLightCount] = bufferFalloffConstant * light->falloff.z;
            } else if (light->falloff.y == 0.10000001f) {
                // Projectile light source, normally hard coded by Morrowind to { 0, 3 * (1/30), 0 }
                // This falloff value cannot be produced by other magic effects
                // Replacement falloff is significantly brighter to look cool
                // Avoids modifying colour or position
                bufferFalloffQuadratic[pointLightCount] = 5e-5;
            } else if (light->falloff.y > 0) {
                // Light magic effect, falloffs calculated by { 0, 3 / (22 * spell magnitude), 0 }
                // A mix of ambient (falloff but no N.L component) and over-bright diffuse lighting
                // It is approximated with a half-lambert weight + quadratic falloff
                // Light colour is altered to avoid variable brightness from Morrowind bugs
                // The point source is moved up slightly as it is often embedded in the ground
                float brightness = 0.25f + 1e-4f / light->falloff.y;
                bufferDiffuse[pointLightCount].x = brightness;
                bufferDiffuse[pointLightCount].y = brightness;
                bufferDiffuse[pointLightCount].z = brightness;
                bufferAmbient[pointLightCount] = 1.0;
                bufferFalloffQuadratic[pointLightCount] = 0.5555f * light->falloff.y * light->falloff.y;
                bufferPosition[pointLightCount + 2*MaxLights] += 25.0;
            }
            ++pointLightCount;
        } else if (light->type == D3DLIGHT_DIRECTIONAL) {
            effectFFE->SetFloatArray(ehLightSunDirection, (const float*)&light->viewspacePos, 3);

            sunDiffuse = light->diffuse;
            ambient.r += light->ambient.x;
            ambient.g += light->ambient.y;
            ambient.b += light->ambient.z;
        }
    }

    // Apply light multipliers, for HDR light levels
    sunDiffuse *= sunMultiplier;
    ambient *= ambMultiplier;

    // Special case, check if ambient state is pure white (distant land does not record this for a reason)
    // Morrowind temporarily sets this for full-bright particle effects, but just adding it
    // to other ambient sources above would cause over-brightness
    DWORD checkAmbient;
    device->GetRenderState(D3DRS_AMBIENT, &checkAmbient);
    if (checkAmbient == 0xffffffff) {
        // Set lighting to result in full-bright equivalent after tonemapping
        ambient.r = ambient.g = ambient.b = 1.25;
        sunDiffuse.r = sunDiffuse.g = sunDiffuse.b = 0.0;
    }

    effectFFE->SetFloatArray(ehLightSceneAmbient, ambient, 3);
    effectFFE->SetFloatArray(ehLightSunDiffuse, sunDiffuse, 3);
    effectFFE->SetVectorArray(ehLightDiffuse, bufferDiffuse, MaxLights);
    effectFFE->SetFloatArray(ehLightAmbient, bufferAmbient, MaxLights);
    effectFFE->SetFloatArray(ehLightPosition, bufferPosition, 3 * MaxLights);
    effectFFE->SetFloatArray(ehLightFalloffQuadratic, bufferFalloffQuadratic, MaxLights);
    effectFFE->SetFloatArray(ehLightFalloffLinear, bufferFalloffLinear, MaxLights);
    effectFFE->SetFloat(ehLightFalloffConstant, bufferFalloffConstant);

    // Bump mapping state
    if (sk.usesBumpmap) {
        const FragmentState::Stage& bumpStage = frs->stage[sk.bumpmapStage];
        effectFFE->SetFloatArray(ehBumpMatrix, &bumpStage.bumpEnvMat[0][0], 4);
        effectFFE->SetFloatArray(ehBumpLumiScaleBias, &bumpStage.bumpLumiScale, 2);
    }

    // Texgen texture matrix
    if (sk.usesTexgen) {
        D3DXMATRIX m;
        device->GetTransform((D3DTRANSFORMSTATETYPE)(D3DTS_TEXTURE0 + sk.texgenStage), &m);
        effectFFE->SetMatrix(ehTexgenTransform, &m);
    }

    // Copy texture bindings from fixed function pipe
    const D3DXHANDLE ehIndex[] = { ehTex0, ehTex1, ehTex2, ehTex3, ehTex4, ehTex5 };
    for (n = 0; n != std::min((int)sk.activeStages, 6); ++n) {
        IDirect3DBaseTexture9* tex;
        device->GetTexture(n, &tex);
        effectFFE->SetTexture(ehIndex[n], tex);
        if (tex) {
            tex->Release();
        }
    }

    // Set common state and render
    effectFFE->SetInt(ehVertexBlendState, rs->vertexBlendState);
    if (rs->vertexBlendState) {
        effectFFE->SetMatrixArray(ehVertexBlendPalette, rs->worldViewTransforms, 4);
    } else {
        effectFFE->SetMatrix(ehWorld, &rs->worldTransforms[0]);
        effectFFE->SetMatrix(ehWorldView, &rs->worldViewTransforms[0]);
    }

    UINT passes;
    effectFFE->Begin(&passes, D3DXFX_DONOTSAVESTATE);
    effectFFE->BeginPass(0);
    device->DrawIndexedPrimitive(rs->primType, rs->baseIndex, rs->minIndex, rs->vertCount, rs->startIndex, rs->primCount);
    effectFFE->EndPass();
    effectFFE->End();

    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);
}

ID3DXEffect* FixedFunctionShader::generateMWShader(const ShaderKey& sk) {
    string genVBCoupling, genPSCoupling, genTransform, genTexcoords, genVertexColour, genLightCount, genMaterial, genTexturing, genFog;
    stringstream buf;

    // Identify output texcoords and check for texgen; supports max. one per shader
    int texGen = 0, texGenSrcIndex = 0, texGenOutputIndex = sk.uvSets, totalOutputCoords = sk.uvSets;
    if (sk.usesTexgen) {
        texGen = sk.stage[sk.texgenStage].texcoordGen;
        texGenSrcIndex = sk.stage[sk.texgenStage].texcoordIndex;

        ++totalOutputCoords;
        if (sk.projectiveTexgen) {
            ++totalOutputCoords;
        }
    }

    if (totalOutputCoords > 4) {
        LOG::logline("!! Shader generator error: excessive texcoord usage (%d).", totalOutputCoords);
        sk.log();
        LOG::flush();

        effectDefaultPurple->AddRef();
        cacheEffects[sk] = effectDefaultPurple;
        return effectDefaultPurple;
    }

    // Pack 2d texcoords into interpolators and map to stages
    const char* strInterpolators[] = { "01", "23" };
    const char* strTexcoordPacking[] = { ".xy", ".zw" };
    string texcoordNames[8], texSamplers[8];

    for (int i = 0; i != sk.activeStages; ++i) {
        bool isTexGen = bool(sk.stage[i].texcoordGen);
        int x = isTexGen ? texGenOutputIndex : sk.stage[i].texcoordIndex;

        buf.str(string());
        buf << "IN.texcoord" << strInterpolators[x >> 1] << strTexcoordPacking[x & 1];
        if (isTexGen && sk.projectiveTexgen) {
            buf << " / IN.texcoord" << strInterpolators[(x+1) >> 1] << strTexcoordPacking[(x+1) & 1];
        }
        texcoordNames[i] = buf.str();
        buf.str(string());
        buf << "tex2D(sampFFE" << i << ", " << texcoordNames[i] << ")";
        texSamplers[i] = buf.str();
    }

    // Vertex format coupling, generate equivalent struct to input FVF
    buf.str(string());

    if (sk.usesSkinning) {
        buf << "float4 blendweights : BLENDWEIGHT; ";
    }
    if (sk.vertexColour) {
        buf << "float4 col : COLOR; ";
    }
    for (int i = 0; i != sk.uvSets; ++i) {
        buf << "float2 texcoord" << i << " : TEXCOORD" << i << "; ";
    }

    genVBCoupling = buf.str();

    // Pixel shader coupling, passes texcoords and colours
    buf.str(string());

    if (sk.vertexColour) {
        buf << "centroid float4 col : COLOR; ";
    }
    if (totalOutputCoords == 1) {
        buf << "float2 texcoord01 : TEXCOORD0; ";
    } else if (totalOutputCoords > 1) {
        buf << "float4 texcoord01 : TEXCOORD0; ";
    }
    if (totalOutputCoords == 3) {
        buf << "float2 texcoord23 : TEXCOORD1; ";
    } else if (totalOutputCoords == 4) {
        buf << "float4 texcoord23 : TEXCOORD1; ";
    }

    genPSCoupling = buf.str();

    // Transform / skinning
    buf.str(string());

    if (sk.usesSkinning) {
        buf << "viewpos = skinnedVertex(IN.pos, IN.blendweights); normal = skinnedNormal(IN.nrm, IN.blendweights);";
    } else {
        buf << "viewpos = rigidVertex(IN.pos); normal = rigidNormal(IN.nrm);";
    }

    genTransform = buf.str();

    // Texcoord routing and texgen
    string texRouting[4];
    for (int i = 0; i != sk.uvSets; ++i) {
        buf.str(string());
        buf << "IN.texcoord" << i;
        texRouting[i] = buf.str();
    }

    buf.str(string());

    if (texGen) {
        buf << "float3 texgen = ";
        switch (texGen) {
        case D3DTSS_TCI_CAMERASPACENORMAL >> 16:
            buf << "texgenNormal(normal); ";
            break;
        case D3DTSS_TCI_CAMERASPACEPOSITION >> 16:
            buf << "texgenPosition(viewpos); ";
            break;
        case D3DTSS_TCI_CAMERASPACEREFLECTIONVECTOR >> 16:
            buf << "texgenReflection(viewpos, normal); ";
            break;
        case D3DTSS_TCI_SPHEREMAP >> 16:
            buf << "texgenSphere(" << texRouting[texGenSrcIndex] << "); ";
            break;
        }
        buf << "texgen = mul(float4(texgen, 1), texgenTransform).xyz; ";
        texRouting[texGenOutputIndex] = "texgen.xy";
        if (sk.projectiveTexgen) {
            texRouting[texGenOutputIndex + 1] = "texgen.zz";
        }
    }

    if (totalOutputCoords == 1) {
        buf << "OUT.texcoord01 = " << texRouting[0] << ";";
    } else if (totalOutputCoords > 1) {
        buf << "OUT.texcoord01 = float4(" << texRouting[0] << ", " << texRouting[1] << "); ";
    }
    if (totalOutputCoords == 3) {
        buf << "OUT.texcoord23 = " << texRouting[2] << ";";
    } else if (totalOutputCoords == 4) {
        buf << "OUT.texcoord23 = float4(" << texRouting[2] << ", " << texRouting[3] << ");";
    }

    genTexcoords = buf.str();

    // Vertex colour routing
    buf.str(string());
    if (sk.vertexColour) {
        buf << "OUT.col = IN.col;";
    }
    genVertexColour = buf.str();

    // Lighting
    if (sk.vertexMaterial == 0) {
        genLightCount = "0";
    } else {
        genLightCount = sk.heavyLighting ? "8" : "4";
    }

    // Vertex material
    buf.str(string());
    switch (sk.vertexMaterial) {
    case 0:
        buf << "diffuse = " << (sk.vertexColour ? "IN.col;" : "1.0;");
        break;
    case 1:
        buf << "diffuse = vertexMaterialNone(d, a);";
        break;
    case 2:
        buf << "diffuse = vertexMaterialDiffAmb(d, a, IN.col);";
        break;
    case 3:
        buf << "diffuse = vertexMaterialEmissive(d, a, IN.col);";
        break;
    }
    genMaterial = buf.str();

    // Texture and shading operations
    buf.str(string());
    string arg1, arg2, arg3;

    for (int i = 0; i != sk.activeStages; ++i) {
        const ShaderKey::Stage& s = sk.stage[i];
        const string dest = s.alphaOpMatched ? "c = " : "c.rgb = ";
        const string mask = s.alphaOpMatched ? "" : ".rgb";

        arg1 = buildArgString(s.colorArg1, mask, texSamplers[i]);
        arg2 = buildArgString(s.colorArg2, mask, texSamplers[i]);

        switch (s.colorOp) {
        case D3DTOP_SELECTARG1:
            buf << dest << arg1 << ";";
            break;

        case D3DTOP_SELECTARG2:
            buf << dest << arg2 << ";";
            break;

        case D3DTOP_MODULATE:
            buf << dest << arg1 << " * " << arg2 << ";";
            break;

        case D3DTOP_MODULATE2X:
            buf << dest << "2 * " << arg1 << " * " << arg2 << ";";
            break;

        case D3DTOP_MODULATE4X:
            buf << dest << "4 * " << arg1 << " * " << arg2 << ";";
            break;

        case D3DTOP_ADD:
            buf << dest << arg1 << " + " << arg2 << ";";
            break;

        case D3DTOP_ADDSIGNED:
            buf << dest << arg1 << " + " << arg2 << " - 0.5;";
            break;

        case D3DTOP_ADDSIGNED2X:
            buf << dest << "2 * (" << arg1 << "+" << arg2 << ") - 1;";
            break;

        case D3DTOP_SUBTRACT:
            buf << dest << arg1 << " - " << arg2 << ";";
            break;

        case D3DTOP_BLENDDIFFUSEALPHA:
            buf << dest << "lerp(" << arg1 << ", " << arg2 << ", diffuse.a);";
            break;

        case D3DTOP_BLENDTEXTUREALPHA:
            arg3 = buildArgString(D3DTA_TEXTURE, "", texSamplers[i]);
            buf << "float4 temp" << i << " = " << arg3 << "; lerp(" << arg1 << ", " << arg1 << ", temp" << i <<".a);";
            break;

        case D3DTOP_BUMPENVMAP:
            arg3 = buildArgString(D3DTA_TEXTURE, "", texSamplers[i]);
            buf << "float4 bump = bumpmapStage(sampFFE" << i+1 << ", " << texcoordNames[i+1] << ", " << arg3 << ");";
            texSamplers[i+1] = "bump";
            break;

        case D3DTOP_BUMPENVMAPLUMINANCE:
            arg3 = buildArgString(D3DTA_TEXTURE, "", texSamplers[i]);
            buf << "float4 bump = bumpmapLumiStage(sampFFE" << i+1 << ", " << texcoordNames[i+1] << ", " << arg3 << ");";
            texSamplers[i+1] = "bump";
            break;

        case D3DTOP_DOTPRODUCT3:
            arg1 = buildArgString(s.colorArg1, ".rgb", texSamplers[i]);
            arg2 = buildArgString(s.colorArg2, ".rgb", texSamplers[i]);
            buf << "c.rgb = dot(" << arg1 << ", " << arg2 << ");";
            break;

        case D3DTOP_MULTIPLYADD:
            arg1 = buildArgString(s.colorArg1, ".rgb", texSamplers[i]);
            arg2 = buildArgString(s.colorArg2, ".rgb", texSamplers[i]);
            arg3 = buildArgString(s.colorArg0, ".rgb", texSamplers[i]);
            buf << "c.rgb = " << arg1 << " * " << arg2 << " + " << arg3 << ";";
            break;

        default:
            buf << "unsupported";
            break;
        }

        if (s.alphaOpSelect1) {
            // Alpha Select1 op, assumes alpha args are the same as color args
            switch (s.colorArg1) {
            case D3DTA_DIFFUSE:
                buf << "c.a = diffuse.a";
                break;

            case D3DTA_TEXTURE:
                // The HLSL compiler is able to optimize this repeated sampler use and does not generate an extra texld.
                buf << "c.a = " << texSamplers[i] << ".a;";
                break;
            }
        }

        buf << " \\\n";
    }

    genTexturing = buf.str();

    // Final fog application
    buf.str(string());

    switch (sk.fogMode) {
    case 0:     // Fog disabled
        break;
    case 1:     // Standard fog mode
        buf << "c.rgb = lerp(fogColNear, c.rgb, fog); ";
        break;
    case 2:     // Additive objects should fog towards black, which preserves the destination correctly
        buf << "c.rgb *= fog; ";
        break;
    }

    genFog = buf.str();

    // Compile HLSL through insertions into a template file
    const D3DXMACRO generatedCode[] = {
        "FFE_VB_COUPLING", genVBCoupling.c_str(),
        "FFE_SHADER_COUPLING", genPSCoupling.c_str(),
        "FFE_TRANSFORM_SKIN", genTransform.c_str(),
        "FFE_TEXCOORDS_TEXGEN", genTexcoords.c_str(),
        "FFE_VERTEX_COLOUR", genVertexColour.c_str(),
        "FFE_LIGHTS_ACTIVE", genLightCount.c_str(),
        "FFE_VERTEX_MATERIAL", genMaterial.c_str(),
        "FFE_TEXTURING", genTexturing.c_str(),
        "FFE_FOG_APPLICATION", genFog.c_str(),
        0, 0
    };

    // Create effect while pooling constants with everything else
    ID3DXEffect* effectFFE;
    ID3DXBuffer* errors;

    //LOG::logline("-- Generating replacement fixed function shader");
    //sk.log();

    HRESULT hr = D3DXCreateEffectFromFile(device, "Data Files\\shaders\\core\\XE FixedFuncEmu.fx", generatedCode, 0, D3DXSHADER_OPTIMIZATION_LEVEL3|D3DXFX_LARGEADDRESSAWARE, constantPool, &effectFFE, &errors);

    if (hr != D3D_OK) {
        LOG::logline("!! Generating FFE shader: compile error %xh", hr);
        if (errors) {
            LOG::write("!! Shader compile errors:\n");
            LOG::write(reinterpret_cast<const char*>(errors->GetBufferPointer()));
            LOG::write("\n");
            errors->Release();
        }
        LOG::write("\n");
        effectDefaultPurple->AddRef();
        effectFFE = effectDefaultPurple;
    }

    cacheEffects[sk] = effectFFE;
    return effectFFE;
}

string buildArgString(DWORD arg, const string& mask, const string& sampler) {
    stringstream s;

    switch (arg) {
    case D3DTA_DIFFUSE:
        s << "diffuse" << mask;
        break;
    case D3DTA_CURRENT:
        s << "c" << mask;
        break;
    case D3DTA_TEXTURE:
        s << sampler << mask;
        break;
    default:
        s << "unsupported";
        break;
    }

    return s.str();
}

// Material state cache helper functions
static inline void setCachedRenderState(IDirect3DDevice9* device, D3DRENDERSTATETYPE state, DWORD value, DWORD& cachedValue, bool& cacheValid) {
    if (!cacheValid || cachedValue != value) {
        device->SetRenderState(state, value);
        cachedValue = value;
        cacheValid = true;
    }
}

static inline void setCachedFVF(IDirect3DDevice9* device, DWORD fvf, DWORD& cachedFVF, bool& cacheValid) {
    if (!cacheValid || cachedFVF != fvf) {
        device->SetFVF(fvf);
        cachedFVF = fvf;
        cacheValid = true;
    }
}

// Texture binding cache helper function with DXVK optimization
void FixedFunctionShader::setCachedTexture(IDirect3DDevice9* device, DWORD stage, IDirect3DTexture9* texture) {
    // Replace null textures with appropriate defaults to avoid DXVK descriptor updates
    IDirect3DTexture9* actualTexture = texture;
    if (!texture) {
        switch (stage) {
            case 0: // Base texture slot
            case 2: // ParamH slot (metallic/roughness)
                actualTexture = defaultWhiteTexture;
                break;
            case 1: // Normal texture slot
            case 3: // ParamX slot (anisotropic/normal)
                actualTexture = defaultNormalTexture;
                break;
            case 4: // Shadow slot
            default:
                actualTexture = defaultBlackTexture;
                break;
        }
    }

    if (textureCache.needsUpdate(stage, actualTexture)) {
        device->SetTexture(stage, actualTexture);
        textureCache.updateCache(stage, actualTexture);
    }
}

// Create default textures to avoid null binds that cause DXVK descriptor updates
void FixedFunctionShader::createDefaultTextures() {
    if (!device) return;

    // Create 1x1 white texture (for missing diffuse/param textures)
    if (device->CreateTexture(1, 1, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &defaultWhiteTexture, nullptr) == S_OK) {
        D3DLOCKED_RECT lockedRect;
        if (defaultWhiteTexture->LockRect(0, &lockedRect, nullptr, 0) == S_OK) {
            *(DWORD*)lockedRect.pBits = 0xFFFFFFFF; // White ARGB
            defaultWhiteTexture->UnlockRect(0);
        }
    }

    // Create 1x1 black texture (for missing specular/height textures)
    if (device->CreateTexture(1, 1, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &defaultBlackTexture, nullptr) == S_OK) {
        D3DLOCKED_RECT lockedRect;
        if (defaultBlackTexture->LockRect(0, &lockedRect, nullptr, 0) == S_OK) {
            *(DWORD*)lockedRect.pBits = 0xFF000000; // Black ARGB (alpha=1, rgb=0)
            defaultBlackTexture->UnlockRect(0);
        }
    }

    // Create 1x1 normal texture (128,128,255,255 for flat normal map)
    if (device->CreateTexture(1, 1, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &defaultNormalTexture, nullptr) == S_OK) {
        D3DLOCKED_RECT lockedRect;
        if (defaultNormalTexture->LockRect(0, &lockedRect, nullptr, 0) == S_OK) {
            *(DWORD*)lockedRect.pBits = 0xFF8080FF; // Normal map: A=255, R=128, G=128, B=255
            defaultNormalTexture->UnlockRect(0);
        }
    }

    LOG::logline("-- Created default textures for DXVK optimization");
}

// Capture current sampler states before replacing textures
void FixedFunctionShader::captureSamplerStates(IDirect3DDevice9* device, DWORD stage) {
    if (stage >= 8) return;

    DWORD addressU, addressV;
    if (device->GetSamplerState(stage, D3DSAMP_ADDRESSU, &addressU) == S_OK) {
        textureCache.cacheSamplerState(stage, D3DSAMP_ADDRESSU, addressU);
    }
    if (device->GetSamplerState(stage, D3DSAMP_ADDRESSV, &addressV) == S_OK) {
        textureCache.cacheSamplerState(stage, D3DSAMP_ADDRESSV, addressV);
    }
}

// Cached texture binding that preserves sampler states for suffix textures
void FixedFunctionShader::setCachedTextureWithSamplerPreservation(IDirect3DDevice9* device, DWORD stage, IDirect3DTexture9* texture) {
    // Capture current sampler states if this is the first time binding to this stage
    if (!textureCache.textureValid[stage] || textureCache.boundTextures[stage] != texture) {
        captureSamplerStates(device, stage);
    }

    // Bind the texture using normal caching
    setCachedTexture(device, stage, texture);

    // Restore the original sampler states after texture binding
    textureCache.restoreSamplerStates(device, stage);
}

// Smart texture binding - only binds slots that the shader actually uses
void FixedFunctionShader::bindShaderTextures(const ShaderKey& sk, const RenderedState* rs) {
    // Original slot assignment (keeping existing layout):
    // Slot 0: Base texture (always bound)
    // Slot 1: Detail texture (conditional with ifdef)
    // Slot 2: ParamH metallic/roughness
    // Slot 3: ParamX anisotropic
    // Slot 4: Shadow map

    // Slot 0: Base texture or diffparam replacement (always used by HLSL shaders)
    IDirect3DTexture9* baseTexture = rs->texture;

    // Check for diffparam replacement if shader supports it
    if (sk.hasDiffParam) {
        auto cacheIt = textureSuffixResolutionCache.find(rs->texture);
        if (cacheIt != textureSuffixResolutionCache.end() &&
            cacheIt->second.hasValidName && cacheIt->second.variants) {

            // Use cached diffparam replacement or load once per texture change
            if (bindingCache.currentBaseTextureName == cacheIt->second.textureName &&
                bindingCache.boundDiffParam) {
                // Use cached replacement
                baseTexture = bindingCache.boundDiffParam;
            } else {
                // Load replacement texture once for this base texture
                IDirect3DTexture9* replacementTexture = nullptr;
                if (cacheIt->second.variants->hasDiffParamT()) {
                    replacementTexture = BSA::loadSuffixTexture((IDirect3DDevice9*)device,
                                                              *cacheIt->second.variants, "diffparam_t");
                }
                if (!replacementTexture && cacheIt->second.variants->hasDiffParam()) {
                    replacementTexture = BSA::loadSuffixTexture((IDirect3DDevice9*)device,
                                                              *cacheIt->second.variants, "diffparam");
                }

                if (replacementTexture) {
                    baseTexture = replacementTexture;
                    // Cache this replacement for future use with same base texture
                    bindingCache.boundDiffParam = replacementTexture;
                }
            }
        }
    }

    // Use sampler preservation if base texture is a diffparam replacement
    if (sk.hasDiffParam && baseTexture != rs->texture) {
        setCachedTextureWithSamplerPreservation(device, 0, baseTexture);
    } else {
        setCachedTexture(device, 0, baseTexture);
    }

    // Slot 1: Detail texture (conditional only - with ifdef support)
    if (sk.hasDetail && savedOriginalDetailTexture) {
        setCachedTexture(device, 1, static_cast<IDirect3DTexture9*>(savedOriginalDetailTexture));
    }

    // Slots 2-3: Suffix textures (only if shader has suffix support)
    if (sk.hasDiffParam || sk.hasParamH || sk.hasParamX) {
        // Fast check: if same texture pointer, skip all expensive operations
        if (bindingCache.lastBaseTexture != rs->texture) {
            // Check texture suffix resolution cache first
            auto cacheIt = textureSuffixResolutionCache.find(rs->texture);
            if (cacheIt == textureSuffixResolutionCache.end()) {
                // Not in cache, perform expensive resolution
                TextureSuffixResolutionCache entry;
                entry.hash = BSA::calculateTextureHash((IDirect3DDevice9*)device, rs->texture, false);
                const std::string* textureName = BSA::resolveTextureNameFromHash(entry.hash);
                if (textureName && entry.hash.crc32 != 0) {
                    entry.textureName = *textureName;
                    entry.hasValidName = true;
                    entry.variants = BSA::getTextureSuffixVariants(textureName->c_str());
                } else {
                    entry.hasValidName = false;
                    entry.variants = nullptr;
                }

                // Cache the result
                textureSuffixResolutionCache[rs->texture] = entry;
                cacheIt = textureSuffixResolutionCache.find(rs->texture);
            }

            if (cacheIt->second.hasValidName) {
                if (bindingCache.currentBaseTextureName != cacheIt->second.textureName) {
                    // Reset cache when texture changes
                    bindingCache.currentBaseTextureName = cacheIt->second.textureName;
                    bindingCache.boundDiffParam = nullptr;
                    bindingCache.boundParamH = nullptr;
                    bindingCache.boundParamX = nullptr;

                    if (cacheIt->second.variants) {
                        // Slot 2: ParamH (metallic/roughness) - load once per texture change
                        if (sk.hasParamH && cacheIt->second.variants->hasParamH()) {
                            if (!bindingCache.boundParamH) {
                                bindingCache.boundParamH = BSA::loadSuffixTexture((IDirect3DDevice9*)device, *cacheIt->second.variants, "paramh");
                            }
                            if (bindingCache.boundParamH) {
                                setCachedTextureWithSamplerPreservation(device, 2, bindingCache.boundParamH);
                            }
                        }

                        // Slot 3: ParamX (anisotropic) - load once per texture change
                        if (sk.hasParamX && cacheIt->second.variants->hasParamX()) {
                            if (!bindingCache.boundParamX) {
                                bindingCache.boundParamX = BSA::loadSuffixTexture((IDirect3DDevice9*)device, *cacheIt->second.variants, "paramx");
                            }
                            if (bindingCache.boundParamX) {
                                setCachedTextureWithSamplerPreservation(device, 3, bindingCache.boundParamX);
                            }
                        }
                    }
                }
                bindingCache.lastBaseTexture = rs->texture;
            }
        }
    }

    // Slot 4: Shadow map
    if (sk.hasShadows) {
        setCachedTexture(device, 4, DistantLand::texSoftShadow);
    }

    // Slot 5: Light data texture (for texture-based point lighting)
    if (DistantLand::texLightData) {
        setCachedTexture(device, 5, DistantLand::texLightData);
    }
}


// Helper function to compute ShaderKey with texture suffix detection
FixedFunctionShader::ShaderKey FixedFunctionShader::computeShaderKeyWithSuffixes(const RenderedState* rs, const FragmentState* frs, LightState* lightrs) {

    // Step 1: Determine texture suffix availability
    bool hasDiffParam = false, hasParamH = false, hasParamX = false, hasGrass = false;

    if (rs->texture) {
        // Check texture suffix resolution cache first
        auto cacheIt = textureSuffixResolutionCache.find(rs->texture);
        if (cacheIt == textureSuffixResolutionCache.end()) {
            // Not in cache, perform expensive resolution
            TextureSuffixResolutionCache entry;

            entry.hash = BSA::calculateTextureHash((IDirect3DDevice9*)device, rs->texture, false);

            const std::string* textureName = BSA::resolveTextureNameFromHash(entry.hash);
            if (textureName && entry.hash.crc32 != 0) {
                entry.textureName = *textureName;
                entry.hasValidName = true;
                entry.variants = BSA::getTextureSuffixVariants(textureName->c_str());
            } else {
                entry.hasValidName = false;
                entry.variants = nullptr;
            }

            cacheIt = textureSuffixResolutionCache.emplace(rs->texture, std::move(entry)).first;
        }

        // Set texture suffix flags based on actual availability
        if (cacheIt->second.hasValidName && cacheIt->second.variants) {
            hasDiffParam = cacheIt->second.variants->hasDiffParam() || cacheIt->second.variants->hasDiffParamT();
            hasParamH = cacheIt->second.variants->hasParamH();
            hasParamX = cacheIt->second.variants->hasParamX();
            hasGrass = cacheIt->second.variants->hasGrass();
        }
    }

    // Step 2: Create ShaderKey with suffix flags
    ShaderKey sk(rs, frs, lightrs);
    sk.hasDiffParam = hasDiffParam;
    sk.hasParamH = hasParamH;
    sk.hasParamX = hasParamX;
    sk.hasGrass = hasGrass;

    // Set shadow flag based on MGE configuration
    sk.hasShadows = ((Configuration.MGEFlags & USE_SHADOWS) && (Configuration.MGEFlags & USE_DISTANT_LAND)) ? 1 : 0;

    return sk;
}

// HLSL Pipeline Implementation
void FixedFunctionShader::renderMorrowindHLSL(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, int recordMWIdx) {
    // Skip if we're in replay mode to avoid recursion
    if (isReplaying) {
        // During replay mode, perform actual rendering with this specific call
        renderMorrowindHLSL_Internal(rs, frs, lightrs);
        return;
    }

    // Start recording at first HLSL call if not already recording (unless under manual control or disabled)
    // Don't restart recording if it already completed this frame (Scene 0 finished)
    if (!isRecording && !isReplaying && !manualRecordingControl && recordingEnabled && !recordingCompletedThisFrame && ImGuiManager::GetEnableRecording()) {
        startRecording();
    }

    // If recording is active, record the call for batched replay
    if (isRecording && ImGuiManager::GetEnableRecording()) {
        // Create a copy of rs and add CURRENT shadow world-view-projection matrices for this draw call
        // During recording, use current matrices; during replay, these will be the "recorded" matrices
        RenderedState rsWithShadows = *rs;

        // Use current shadow matrices for this specific draw call during recording
        rsWithShadows.shadowWorldViewProj[0] = rs->worldTransforms[0] * DistantLand::smViewproj[0];
        rsWithShadows.shadowWorldViewProj[1] = rs->worldTransforms[0] * DistantLand::smViewproj[1];

        // Defer shader key computation to prepare phase (avoid texture hash lookups during recording)
        ShaderKey sk;
        memset(&sk, 0, sizeof(sk));  // Placeholder — computed in prepareRecordedCalls()
        recordRenderCall(&rsWithShadows, frs, lightrs, sk, recordMWIdx);
        return;
    }

    // Normal rendering path (when not recording or replaying) - renders immediately
    if (!ImGuiManager::GetEnableImmediateRendering()) {
        return;  // Skip immediate rendering if disabled
    }

    // Set empty light parameters for immediate rendering (hands/UI don't use texture lights)
    float lightParams[4] = { 0.0f, 0.0f, 0.0f, 0.0f };  // numLights = 0
    device->SetPixelShaderConstantF(50, lightParams, 1);

    renderMorrowindHLSL_Internal(rs, frs, lightrs);
}

// Internal rendering function that does the actual HLSL rendering
void FixedFunctionShader::renderMorrowindHLSL_Internal(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, DWORD dirtyFlags) {

    // Process any completed async shader compilations
    processAsyncCompletions();

    HLSLShader hlslShader;

    // Get ShaderKey with texture suffix detection
    ShaderKey sk;
    if (isReplaying) {
        // During replay, use the recorded ShaderKey with original suffix flags
        for (const auto& call : recordedCalls) {
            if (&call.rs == rs) {
                sk = call.sk;
                break;
            }
        }
    } else {
        // During normal rendering, compute ShaderKey with texture suffix detection
        sk = computeShaderKeyWithSuffixes(rs, frs, lightrs);
    }

    // Check if Morrowind bound a detail texture to slot 1 (frequency optimization)
    // Save the original detail texture before we start binding our own textures
    if (savedOriginalDetailTexture) {
        savedOriginalDetailTexture->Release(); // Release previous frame's texture
        savedOriginalDetailTexture = nullptr;
    }
    device->GetTexture(1, &savedOriginalDetailTexture);
    sk.hasDetail = (savedOriginalDetailTexture != nullptr) ? 1 : 0;

    if (sk == hlslShaderLRU.last_sk) {
        hlslShader = hlslShaderLRU.shader;
    } else {
        // Read from shader cache / generate
        decltype(cacheHLSLShaders)::const_iterator iShader = cacheHLSLShaders.find(sk);

        if (iShader != cacheHLSLShaders.end()) {
            hlslShader = iShader->second;
            // LOG::logline("DEBUG CACHE HIT: Using cached shader with flags diffparam=%d normal=%d param=%d", 
            //           sk.hasDiffParam, sk.hasNormal, sk.hasParam);
        } else {
            // Cache miss - try smart fallback before using purple
            queueShaderCompilation(sk);
            
            // Smart fallback hierarchy: try alternative point light counts
            ShaderKey fallbackSk = sk;
            HLSLShader fallbackShader = {};
            bool foundFallback = false;
            
            // Priority fallback order: prefer exact match first, then alternatives
            int currentCount = sk.noPointLights ? 0 : 1;  // 0 = no lights, 1 = has lights
            int fallbackPointLights[] = {0, 1, 4, 7};
            for (int i = 0; i < 4 && !foundFallback; ++i) {
                int fallbackCount = fallbackPointLights[i];
                if (fallbackCount == currentCount) continue; // Skip exact match, already tried
                
                // Correct the noPointLights assignment: 0 lights -> noPointLights=1, 1+ lights -> noPointLights=0
                fallbackSk.noPointLights = (fallbackCount == 0) ? 1 : 0;
                auto fallbackIter = cacheHLSLShaders.find(fallbackSk);
                if (fallbackIter != cacheHLSLShaders.end()) {
                    fallbackShader = fallbackIter->second;
                    foundFallback = true;
                }
            }
            
            // If no point light fallback found, try texture suffix fallbacks
            if (!foundFallback && (sk.hasDiffParam || sk.hasParamH || sk.hasParamX)) {
                // Try progressively simpler texture combinations
                ShaderKey textureFallbackSk = sk;
                
                // Step 1: Remove diffparam but keep normal+param
                if (sk.hasDiffParam && !foundFallback) {
                    textureFallbackSk.hasDiffParam = 0;
                    auto fallbackIter = cacheHLSLShaders.find(textureFallbackSk);
                    if (fallbackIter != cacheHLSLShaders.end()) {
                        fallbackShader = fallbackIter->second;
                        foundFallback = true;
                    }
                }
                
                // Step 2: Remove normal but keep param
                if (sk.hasParamH && !foundFallback) {
                    textureFallbackSk = sk;
                    textureFallbackSk.hasDiffParam = 0;
                    textureFallbackSk.hasParamH = 0;
                    textureFallbackSk.hasParamX = 0;
                    auto fallbackIter = cacheHLSLShaders.find(textureFallbackSk);
                    if (fallbackIter != cacheHLSLShaders.end()) {
                        fallbackShader = fallbackIter->second;
                        foundFallback = true;
                    }
                }
                
                // Step 3: Base texture only (no suffixes)
                if (!foundFallback) {
                    textureFallbackSk = sk;
                    textureFallbackSk.hasDiffParam = 0;
                    textureFallbackSk.hasParamH = 0;
                    textureFallbackSk.hasParamX = 0;
                    auto fallbackIter = cacheHLSLShaders.find(textureFallbackSk);
                    if (fallbackIter != cacheHLSLShaders.end()) {
                        fallbackShader = fallbackIter->second;
                        foundFallback = true;
                    }
                }
            }
            
            // Final universal fallback: try the 8 guaranteed base combinations
            if (!foundFallback) {
                // Start with exact copy of requested shader, then simplify
                ShaderKey universalSk = sk;

                // Force base texture settings
                universalSk.heavyLighting = 0;
                universalSk.hasDiffParam = 0;
                universalSk.hasParamH = 0;
                universalSk.hasParamX = 0;

                // Try all 8 universal combinations: point lights (0/1+) × vertex color (0/1) × skinning (0/1)
                for (int pointLightMode = 0; pointLightMode <= 1 && !foundFallback; ++pointLightMode) {
                    for (int vertCol = 0; vertCol <= 1 && !foundFallback; ++vertCol) {
                        for (int skinning = 0; skinning <= 1 && !foundFallback; ++skinning) {
                            universalSk.noPointLights = pointLightMode; // pointLightMode 0 = no lights, so noPointLights = 0
                            universalSk.vertexColour = vertCol;
                            universalSk.vertexMaterial = vertCol + 1;
                            universalSk.usesSkinning = skinning;

                            auto fallbackIter = cacheHLSLShaders.find(universalSk);
                            if (fallbackIter != cacheHLSLShaders.end()) {
                                fallbackShader = fallbackIter->second;
                                foundFallback = true;
                            }
                        }
                    }
                }
            }
            
            if (foundFallback) {
                hlslShader = fallbackShader;
            } else {
                hlslShader = hlslShaderDefaultPurple;
            }
        }

        hlslShaderLRU.shader = hlslShader;
        hlslShaderLRU.last_sk = sk;
    }
    
    // DXVK-optimized texture binding - only touches slots the shader actually uses
    bindShaderTextures(sk, rs);

    // Get current view matrix and compute inverse (needed for texture lights and shadows)
    D3DXMATRIX currentView;
    device->GetTransform(D3DTS_VIEW, &currentView);

    // Only recalculate inverse view when view changes
    if (!shadowMatricesValid || memcmp(&currentView, &cachedViewMatrix, sizeof(D3DXMATRIX)) != 0) {
        cachedViewMatrix = currentView;
        D3DXMatrixInverse(&cachedInverseView, NULL, &currentView);
        shadowMatricesValid = true;
    }

    // Set viewInverse matrix for texture-based lighting (always needed)
    device->SetPixelShaderConstantF(18, (float*)&cachedInverseView, 4); // c18

    // Set shadow matrices if shadows are enabled
    if (sk.hasShadows) {
        // Compute shadow transform matrices using cached inverse view
        static D3DXMATRIX cachedViewToShadowLocal[2];
        static bool shadowTransformValid = false;

        if (!shadowTransformValid || !shadowMatricesValid) {
            cachedViewToShadowLocal[0] = cachedInverseView * DistantLand::smViewproj[0];
            cachedViewToShadowLocal[1] = cachedInverseView * DistantLand::smViewproj[1];
            shadowTransformValid = shadowMatricesValid;
        }

        device->SetVertexShaderConstantF(20, (float*)&cachedViewToShadowLocal[0], 4); // c20-c23
        device->SetVertexShaderConstantF(24, (float*)&cachedViewToShadowLocal[1], 4); // c24-c27

        // Set shadow resolution parameter
        float shadowRcp = 1.0f / Configuration.DL.ShadowResolution;
        device->SetPixelShaderConstantF(10, &shadowRcp, 1); // c10
    }


    // Save current render states before modifying them (only states that HLSL actually changes)
    DWORD savedAlphaBlendEnable = 0, savedAlphaTestEnable = 0;
    DWORD savedZEnable = 0, savedZWriteEnable = 0;
    DWORD savedSpecularEnable = 0, savedLocalViewer = 0, savedNormalizeNormals = 0;
    device->GetRenderState(D3DRS_ALPHABLENDENABLE, &savedAlphaBlendEnable);
    device->GetRenderState(D3DRS_ALPHATESTENABLE, &savedAlphaTestEnable);
    device->GetRenderState(D3DRS_ZENABLE, &savedZEnable);
    device->GetRenderState(D3DRS_ZWRITEENABLE, &savedZWriteEnable);
    device->GetRenderState(D3DRS_SPECULARENABLE, &savedSpecularEnable);
    device->GetRenderState(D3DRS_LOCALVIEWER, &savedLocalViewer);
    device->GetRenderState(D3DRS_NORMALIZENORMALS, &savedNormalizeNormals);
    
    // Set shaders
    device->SetVertexShader(hlslShader.vertexShader);
    device->SetPixelShader(hlslShader.pixelShader);
    
    // Use cached render state setting to minimize redundant SetRenderState calls
    // Depth and culling states
    DWORD zEnable, zWriteEnable;
    if (rs->blendEnable) {
        zEnable = TRUE;
        zWriteEnable = FALSE;
    } else {
        zEnable = rs->zWrite ? TRUE : FALSE;
        zWriteEnable = rs->zWrite ? TRUE : FALSE;
    }

    setCachedRenderState(device, D3DRS_ZENABLE, zEnable, materialCache.zEnable, materialCache.zEnableValid);
    setCachedRenderState(device, D3DRS_ZWRITEENABLE, zWriteEnable, materialCache.zWriteEnable, materialCache.zWriteEnableValid);
    setCachedRenderState(device, D3DRS_ZFUNC, D3DCMP_LESSEQUAL, materialCache.zFunc, materialCache.zFuncValid);
    setCachedRenderState(device, D3DRS_CULLMODE, rs->cullMode, materialCache.cullMode, materialCache.cullModeValid);

    // Disable DX8 specular pipeline that Morrowind.exe might have enabled - HLSL handles specular internally
    setCachedRenderState(device, D3DRS_SPECULARENABLE, FALSE, materialCache.specularEnable, materialCache.specularEnableValid);
    setCachedRenderState(device, D3DRS_LOCALVIEWER, FALSE, materialCache.localViewer, materialCache.localViewerValid);
    setCachedRenderState(device, D3DRS_NORMALIZENORMALS, FALSE, materialCache.normalizeNormals, materialCache.normalizeNormalsValid);

    // Alpha blending states
    setCachedRenderState(device, D3DRS_ALPHABLENDENABLE, rs->blendEnable, materialCache.alphaBlendEnable, materialCache.alphaBlendEnableValid);
    if (rs->blendEnable) {
        setCachedRenderState(device, D3DRS_SRCBLEND, rs->srcBlend, materialCache.srcBlend, materialCache.srcBlendValid);
        setCachedRenderState(device, D3DRS_DESTBLEND, rs->destBlend, materialCache.destBlend, materialCache.destBlendValid);
    }

    // Alpha testing states
    setCachedRenderState(device, D3DRS_ALPHATESTENABLE, rs->alphaTest, materialCache.alphaTestEnable, materialCache.alphaTestEnableValid);
    if (rs->alphaTest) {
        setCachedRenderState(device, D3DRS_ALPHAFUNC, rs->alphaFunc, materialCache.alphaFunc, materialCache.alphaFuncValid);
        setCachedRenderState(device, D3DRS_ALPHAREF, rs->alphaRef, materialCache.alphaRef, materialCache.alphaRefValid);
    }

    // Set vertex format (legacy DX8 FVF - HLSL input semantics handle layout internally)
    setCachedFVF(device, rs->fvf, materialCache.fvf, materialCache.fvfValid);
    
    // Set vertex and index buffers like the original system
    device->SetStreamSource(0, rs->vb, rs->vbOffset, rs->vbStride);
    if (rs->ib) {
        device->SetIndices(rs->ib);
    }

    // Set up matrices using constant tables (like the Combined shader expects)
    D3DXMATRIX projMatrix, viewMatrix, worldMatrix;

    // During replay, use ALL recorded matrices to avoid stale matrix issues
    // During normal rendering, get them from the device
    if (isReplaying) {
        projMatrix = recordingDeviceProj;
        viewMatrix = recordingDeviceView;
        worldMatrix = rs->worldTransforms[0];
    } else {
        device->GetTransform(D3DTS_PROJECTION, &projMatrix);
        device->GetTransform(D3DTS_VIEW, &viewMatrix);
        device->GetTransform(D3DTS_WORLD, &worldMatrix);
    }
    
    // During replay, use recorded combined matrices; during normal rendering, calculate them
    D3DXMATRIX worldViewProj, worldView;
    if (isReplaying) {
        // Use pre-recorded combined matrices to avoid any matrix timing issues
        worldViewProj = rs->worldTransforms[0] * recordingDeviceView * recordingDeviceProj;
        worldView = rs->worldTransforms[0] * recordingDeviceView;
    } else {
        // Normal rendering - calculate from current matrices
        worldViewProj = worldMatrix * viewMatrix * projMatrix;
        worldView = worldMatrix * viewMatrix;
    }
    
    // Use constant tables to set matrices with cached handles (no per-draw string lookups)
    if (hlslShader.vsConstantTable) {
        try {
            if (hlslShader.hWorldViewProj) {
                HRESULT hr = hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hWorldViewProj, &worldViewProj);
                if (FAILED(hr)) {
                    // Shader may have been invalidated by file edit - use fallback
                    return;
                }
            }

            if (hlslShader.hView) {
                HRESULT hr = hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hView, &viewMatrix);
                if (FAILED(hr)) {
                    return;
                }
            }

            if (hlslShader.hProj) {
                HRESULT hr = hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hProj, &projMatrix);
                if (FAILED(hr)) {
                    return;
                }
            }

            if (hlslShader.hWorld) {
                // Use recorded world matrix for each object, not current device world matrix
                HRESULT hr = hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hWorld, &rs->worldTransforms[0]);
                if (FAILED(hr)) {
                    return;
                }
            }

            if (hlslShader.hWorldView) {
                HRESULT hr = hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hWorldView, &worldView);
                if (FAILED(hr)) {
                    return;
                }
            }
            // Set up vertex blend palette for skinning using Morrowind's actual data
            if (hlslShader.hVertexBlendPalette) {
                if (rs->vertexBlendState > 0) {
                    // For skinned objects, recombine recorded world matrices with current view matrix
                    // rs->worldViewTransforms contains old view matrix, causing one-frame delay
                    D3DXMATRIX currentWorldViewTransforms[4];
                    for (int i = 0; i < 4; i++) {
                        currentWorldViewTransforms[i] = rs->worldTransforms[i] * viewMatrix;
                    }
                    HRESULT hr = hlslShader.vsConstantTable->SetMatrixArray(device, hlslShader.hVertexBlendPalette, currentWorldViewTransforms, 4);
                    if (FAILED(hr)) {
                        return;
                    }
                } else {
                    // For rigid objects, set first matrix to worldview and clear others
                    D3DXMATRIX blendMatrices[4];
                    blendMatrices[0] = worldView;
                    memset(&blendMatrices[1], 0, sizeof(D3DXMATRIX) * 3);
                    HRESULT hr = hlslShader.vsConstantTable->SetMatrixArray(device, hlslShader.hVertexBlendPalette, blendMatrices, 4);
                    if (FAILED(hr)) {
                        return;
                    }
                }
            }

            if (hlslShader.hVertexBlendState) {
                D3DXVECTOR4 blendState((float)rs->vertexBlendState, 0, 0, 0);
                HRESULT hr = hlslShader.vsConstantTable->SetVector(device, hlslShader.hVertexBlendState, &blendState);
                if (FAILED(hr)) {
                    return;
                }
            }

            // Set shadow world-to-shadow matrices for proper shadow coordinate calculation
            if (hlslShader.hShadowWorldViewProj) {
                // Use recorded complete shadow world-view-projection matrices directly
                // This avoids any stale matrix issues by using exact matrices from recording time
                HRESULT hr = hlslShader.vsConstantTable->SetMatrixArray(device, hlslShader.hShadowWorldViewProj, rs->shadowWorldViewProj, 2);
                if (FAILED(hr)) {
                    return;
                }
            }
        } catch (...) {
            // Shader invalidated during file edit - return early
            LOG::logline("!! HLSL Vertex shader constant table access failed - shader may have been edited");
            return;
        }
    }
    // Set pixel shader constants using cached handles (no per-draw string lookups)
    if (hlslShader.psConstantTable) {
        try {
            if (hlslShader.hMaterialDiffuse) {
                HRESULT hr = hlslShader.psConstantTable->SetVector(device, hlslShader.hMaterialDiffuse, (D3DXVECTOR4*)&frs->material.diffuse);
                if (FAILED(hr)) {
                    LOG::logline("!! HLSL Pixel shader constant table access failed - shader may have been edited");
                    return;
                }
            }

            if (hlslShader.hMaterialAmbient) {
                HRESULT hr = hlslShader.psConstantTable->SetVector(device, hlslShader.hMaterialAmbient, (D3DXVECTOR4*)&frs->material.ambient);
                if (FAILED(hr)) {
                    return;
                }
            }

            if (hlslShader.hMaterialEmissive) {
                HRESULT hr = hlslShader.psConstantTable->SetVector(device, hlslShader.hMaterialEmissive, (D3DXVECTOR4*)&frs->material.emissive);
                if (FAILED(hr)) {
                    return;
                }
            }
        // Set up lighting using the same logic as the original renderMorrowind
        const size_t MaxLights = 8;
        D3DXVECTOR4 bufferDiffuse[MaxLights];
        float bufferAmbient[MaxLights];
        float bufferPosition[3 * MaxLights];
        float bufferFalloffQuadratic[MaxLights], bufferFalloffLinear[MaxLights], bufferFalloffConstant;

        memset(&bufferDiffuse, 0, sizeof(bufferDiffuse));
        memset(&bufferAmbient, 0, sizeof(bufferAmbient));
        memset(&bufferPosition, 0, sizeof(bufferPosition));
        memset(&bufferFalloffQuadratic, 0, sizeof(bufferFalloffQuadratic));
        memset(&bufferFalloffLinear, 0, sizeof(bufferFalloffLinear));
        bufferFalloffConstant = 0.33;

        // Check each active light
        RGBVECTOR sunDiffuse(0, 0, 0), ambient = lightrs->globalAmbient;
        D3DVECTOR sunDirection = {0, 0, 1};
        size_t n = std::min(lightrs->active.size(), MaxLights), pointLightCount = 0;
        for (; n --> 0; ) {
            DWORD i = lightrs->active[n];
            const LightState::Light* light = &lightrs->lights.find(i)->second;

            // Transform to view space if not transformed this frame
            if (lightrs->lightsTransformed.find(i) == lightrs->lightsTransformed.end()) {
                if (light->type == D3DLIGHT_DIRECTIONAL) {
                    D3DXVec3TransformNormal((D3DXVECTOR3*)&light->viewspacePos, (D3DXVECTOR3*)&light->position, &rs->viewTransform);
                } else {
                    D3DXVec3TransformCoord((D3DXVECTOR3*)&light->viewspacePos, (D3DXVECTOR3*)&light->position, &rs->viewTransform);
                }

                lightrs->lightsTransformed[i] = true;
            }

            if (light->type == D3DLIGHT_POINT) {
                memcpy(&bufferDiffuse[pointLightCount], &light->diffuse, sizeof(light->diffuse));

                // Scatter position vectors for vectorization
                bufferPosition[pointLightCount] = light->viewspacePos.x;
                bufferPosition[pointLightCount + MaxLights] = light->viewspacePos.y;
                bufferPosition[pointLightCount + 2*MaxLights] = light->viewspacePos.z;

                // Scatter attenuation factors for vectorization (match Effect path)
                if (light->falloff.x > 0) {
                    // Standard point light source (falloffConstant doesn't vary per light)
                    bufferFalloffConstant = light->falloff.x;
                    bufferFalloffLinear[pointLightCount] = light->falloff.y;
                    bufferFalloffQuadratic[pointLightCount] = light->falloff.z;
                } else if (light->falloff.z > 0) {
                    // Probably a magic light source patched by Morrowind Code Patch
                    bufferDiffuse[pointLightCount].x *= bufferFalloffConstant;
                    bufferDiffuse[pointLightCount].y *= bufferFalloffConstant;
                    bufferDiffuse[pointLightCount].z *= bufferFalloffConstant;
                    bufferAmbient[pointLightCount] = 1.0f + 1e-4f / sqrt(light->falloff.z);
                    bufferFalloffQuadratic[pointLightCount] = bufferFalloffConstant * light->falloff.z;
                } else if (light->falloff.y == 0.10000001f) {
                    // Projectile light source, normally hard coded by Morrowind to { 0, 3 * (1/30), 0 }
                    // This falloff value cannot be produced by other magic effects
                    // Replacement falloff is significantly brighter to look cool
                    // Avoids modifying colour or position
                    bufferFalloffQuadratic[pointLightCount] = 5e-5;
                } else if (light->falloff.y > 0) {
                    // Light magic effect, falloffs calculated by { 0, 3 / (22 * spell magnitude), 0 }
                    // A mix of ambient (falloff but no N.L component) and over-bright diffuse lighting
                    // It is approximated with a half-lambert weight + quadratic falloff
                    // Light colour is altered to avoid variable brightness from Morrowind bugs
                    // The point source is moved up slightly as it is often embedded in the ground
                    float brightness = 0.25f + 1e-4f / light->falloff.y;
                    bufferDiffuse[pointLightCount].x = brightness;
                    bufferDiffuse[pointLightCount].y = brightness;
                    bufferDiffuse[pointLightCount].z = brightness;
                    bufferAmbient[pointLightCount] = 1.0;
                    bufferFalloffQuadratic[pointLightCount] = 0.5555f * light->falloff.y * light->falloff.y;
                    bufferPosition[pointLightCount + 2*MaxLights] += 25.0;
                }

                ++pointLightCount;
            } else if (light->type == D3DLIGHT_DIRECTIONAL) {
                sunDiffuse = light->diffuse;
                sunDirection = light->viewspacePos;  // Already transformed to view space
                // Add directional light ambient to global ambient like the original
                ambient.r += light->ambient.x;
                ambient.g += light->ambient.y;
                ambient.b += light->ambient.z;
            }
        }
        
        // Apply light multipliers, for HDR light levels
        sunDiffuse *= sunMultiplier;
        ambient *= ambMultiplier;
        
        // Special case, check if ambient state is pure white (distant land does not record this for a reason)
        // Morrowind temporarily sets this for full-bright particle effects
        DWORD checkAmbient;
        device->GetRenderState(D3DRS_AMBIENT, &checkAmbient);
        if (checkAmbient == 0xffffffff) {
            // Set lighting to result in full-bright equivalent after tonemapping
            ambient.r = ambient.g = ambient.b = 1.25;
            sunDiffuse.r = sunDiffuse.g = sunDiffuse.b = 0.0;
        }
        
        // Set lighting constants using the same format as the original system
        if (hlslShader.hLightSunDirection) {
            hlslShader.psConstantTable->SetFloatArray(device, hlslShader.hLightSunDirection, (const float*)&sunDirection, 3);
        } else {
            // logline("!! lightSunDirection constant not found in pixel shader");
        }

        if (hlslShader.hLightSunDiffuse) {
            hlslShader.psConstantTable->SetFloatArray(device, hlslShader.hLightSunDiffuse, (const float*)&sunDiffuse, 3);
        } else {
            // LOG::logline("!! lightSunDiffuse constant not found in pixel shader");
        }

        if (hlslShader.hLightSceneAmbient) {
            hlslShader.psConstantTable->SetFloatArray(device, hlslShader.hLightSceneAmbient, (const float*)&ambient, 3);
        } else {
            // LOG::logline("!! lightSceneAmbient constant not found in pixel shader");
        }
        
        // Set light arrays in pixel shader (same as Effect shader approach)
        D3DXHANDLE hLightDiffuse = hlslShader.psConstantTable->GetConstantByName(NULL, "lightDiffuse");
        if (hLightDiffuse) {
            hlslShader.psConstantTable->SetVectorArray(device, hLightDiffuse, bufferDiffuse, MaxLights);
        } else {
            // LOG::logline("!! lightDiffuse array constant not found in pixel shader");
        }
        
        D3DXHANDLE hLightPosition = hlslShader.psConstantTable->GetConstantByName(NULL, "lightPosition");
        if (hLightPosition) {
            // HLSL expects float3 array, but we have packed data - need to convert
            D3DXVECTOR3 hlslLightPositions[MaxLights];
            for (int i = 0; i < MaxLights; i++) {
                hlslLightPositions[i].x = bufferPosition[i];
                hlslLightPositions[i].y = bufferPosition[i + MaxLights];
                hlslLightPositions[i].z = bufferPosition[i + 2*MaxLights];
            }
            hlslShader.psConstantTable->SetFloatArray(device, hLightPosition, (float*)hlslLightPositions, 3 * MaxLights);
        } else {
            // LOG::logline("!! HLSL ERROR: lightPosition array constant not found in pixel shader");
        }
        
        // Set light ambient array
        D3DXHANDLE hLightAmbient = hlslShader.psConstantTable->GetConstantByName(NULL, "lightAmbient");
        if (hLightAmbient) {
            hlslShader.psConstantTable->SetFloatArray(device, hLightAmbient, bufferAmbient, MaxLights);
        } else {
            // LOG::logline("!! lightAmbient array constant not found in pixel shader");
        }
        
        // Set pointLightCount uniform for HLSL (CRITICAL FIX)
        D3DXHANDLE hPointLightCount = hlslShader.psConstantTable->GetConstantByName(NULL, "pointLightCount");
        if (hPointLightCount) {
            hlslShader.psConstantTable->SetInt(device, hPointLightCount, (int)pointLightCount);
            // LOG::logline("HLSL: Set pointLightCount to %d", (int)pointLightCount);
        } else {
            // LOG::logline("!! HLSL ERROR: pointLightCount constant not found in pixel shader");
        }
        
        // Set falloff constants using Effect shader approach (quadratic + constant only)
        D3DXHANDLE hLightFalloffQuadratic = hlslShader.psConstantTable->GetConstantByName(NULL, "lightFalloffQuadratic");
        if (hLightFalloffQuadratic) {
            // Pack quadratic falloffs into 2 float4 vectors (8 lights total, 4 per vector)
            D3DXVECTOR4 quadraticData[2];
            for (int i = 0; i < 4; i++) {
                quadraticData[0][i] = ((size_t)i < pointLightCount) ? bufferFalloffQuadratic[i] : 0.0f;
                quadraticData[1][i] = ((size_t)(i + 4) < pointLightCount) ? bufferFalloffQuadratic[i + 4] : 0.0f;
            }
            hlslShader.psConstantTable->SetVectorArray(device, hLightFalloffQuadratic, quadraticData, 2);
        }
        
        D3DXHANDLE hLightFalloffConstant = hlslShader.psConstantTable->GetConstantByName(NULL, "lightFalloffConstant");
        if (hLightFalloffConstant) {
            hlslShader.psConstantTable->SetFloat(device, hLightFalloffConstant, bufferFalloffConstant);
        }
        
        // Note: HLSL uses same falloff as Effect shader - quadratic + constant only, no linear term
        // Set shading mode from actual material mode calculation
        D3DXHANDLE hShadingMode = hlslShader.psConstantTable->GetConstantByName(NULL, "shadingMode");
        if (hShadingMode) {
            float shadingModeData[4] = {0, 0, (float)sk.vertexMaterial, 0};
            hlslShader.psConstantTable->SetFloatArray(device, hShadingMode, shadingModeData, 4);
        } else {
            // LOG::logline("!! shadingMode constant not found in pixel shader");
        }
        
        // Set shadow resolution for pixel shader
        if (hlslShader.hShadowRcpRes) {
            float shadowRcp = 1.0f / Configuration.DL.ShadowResolution;
            hlslShader.psConstantTable->SetFloat(device, hlslShader.hShadowRcpRes, shadowRcp);
        }

        // Set PCF parameters from ImGui debug interface
        if (hlslShader.hPCFFilterSize) {
            hlslShader.psConstantTable->SetFloat(device, hlslShader.hPCFFilterSize, ImGuiManager::GetPCFFilterSize());
        }
        
        D3DXHANDLE hPCFPenumbraScale = hlslShader.psConstantTable->GetConstantByName(NULL, "PCF_penumbraScale");
        if (hPCFPenumbraScale) {
            hlslShader.psConstantTable->SetFloat(device, hPCFPenumbraScale, ImGuiManager::GetPCFPenumbraScale());
        }
        
        D3DXHANDLE hPCFMinPenumbra = hlslShader.psConstantTable->GetConstantByName(NULL, "PCF_minPenumbra");
        if (hPCFMinPenumbra) {
            hlslShader.psConstantTable->SetFloat(device, hPCFMinPenumbra, ImGuiManager::GetPCFMinPenumbra());
        }
        
        D3DXHANDLE hPCFMaxPenumbra = hlslShader.psConstantTable->GetConstantByName(NULL, "PCF_maxPenumbra");
        if (hPCFMaxPenumbra) {
            hlslShader.psConstantTable->SetFloat(device, hPCFMaxPenumbra, ImGuiManager::GetPCFMaxPenumbra());
        }
        
        D3DXHANDLE hPCFBias = hlslShader.psConstantTable->GetConstantByName(NULL, "PCF_bias");
        if (hPCFBias) {
            hlslShader.psConstantTable->SetFloat(device, hPCFBias, ImGuiManager::GetPCFBias());
        }
        
        D3DXHANDLE hPCFBias2 = hlslShader.psConstantTable->GetConstantByName(NULL, "PCF_bias2");
        if (hPCFBias2) {
            hlslShader.psConstantTable->SetFloat(device, hPCFBias2, ImGuiManager::GetPCFBias2());
        }
        
        D3DXHANDLE hPCFSlopeBias = hlslShader.psConstantTable->GetConstantByName(NULL, "PCF_slopeBias");
        if (hPCFSlopeBias) {
            hlslShader.psConstantTable->SetFloat(device, hPCFSlopeBias, ImGuiManager::GetPCFSlopeBias());
        }
        
        // Set fog color
        DWORD fogColorDword = 0x808080FF;
        device->GetRenderState(D3DRS_FOGCOLOR, &fogColorDword);
        D3DXVECTOR4 fogColor(
            ((fogColorDword >> 16) & 0xFF) / 255.0f,
            ((fogColorDword >> 8) & 0xFF) / 255.0f,
            (fogColorDword & 0xFF) / 255.0f,
            1.0f
        );
        
        D3DXHANDLE hFogColNear = hlslShader.psConstantTable->GetConstantByName(NULL, "fogColNear");
        if (hFogColNear) {
            hlslShader.psConstantTable->SetVector(device, hFogColNear, &fogColor);
        }
        
        // NOTE: Texture binding and suffix processing is now handled by bindShaderTextures() above
        // This includes frequency-optimized slot assignment and proper cache management.
        // Removed redundant texture copying loop that was interfering with optimized binding.
        
        // HLSL samplers are declared in shader with proper filtering/addressing - no need for manual sampler states
        
        // Set missing constants that the Combined shader expects
        D3DXHANDLE hTexgenTransform = hlslShader.vsConstantTable->GetConstantByName(NULL, "texgenTransform");
        if (hTexgenTransform) {
            D3DXMATRIX identity;
            D3DXMatrixIdentity(&identity);
            hlslShader.vsConstantTable->SetMatrix(device, hTexgenTransform, &identity);
        }
        
        D3DXHANDLE hBumpMatrix = hlslShader.psConstantTable->GetConstantByName(NULL, "bumpMatrix");
        if (hBumpMatrix) {
            D3DXVECTOR4 bumpMatrix(1, 0, 0, 1);  // Identity 2x2 matrix
            hlslShader.psConstantTable->SetVector(device, hBumpMatrix, &bumpMatrix);
        }
        
        D3DXHANDLE hBumpLumiScaleBias = hlslShader.psConstantTable->GetConstantByName(NULL, "bumpLumiScaleBias");
        if (hBumpLumiScaleBias) {
            D3DXVECTOR2 scaleBias(1, 0);  // Scale=1, Bias=0
            hlslShader.psConstantTable->SetFloatArray(device, hBumpLumiScaleBias, (float*)&scaleBias, 2);
        }
        
        // Set critical shared variables that the Combined shader needs
        D3DXHANDLE hHasAlpha = hlslShader.psConstantTable->GetConstantByName(NULL, "hasAlpha");
        if (hHasAlpha) {
            hlslShader.psConstantTable->SetBool(device, hHasAlpha, false);
        }
        
        D3DXHANDLE hHasBones = hlslShader.vsConstantTable->GetConstantByName(NULL, "hasBones");
        if (hHasBones) {
            hlslShader.vsConstantTable->SetBool(device, hHasBones, rs->vertexBlendState > 0);
        }
        
        // Alpha testing flag for wind animation
        D3DXHANDLE hHasAlphaVS = hlslShader.vsConstantTable->GetConstantByName(NULL, "hasAlpha");
        if (hHasAlphaVS) {
            hlslShader.vsConstantTable->SetBool(device, hHasAlphaVS, rs->alphaTest);
        }
        
        // Wind vector for animations
        D3DXHANDLE hWindVec = hlslShader.vsConstantTable->GetConstantByName(NULL, "windVec");
        if (hWindVec) {
            static float smoothWind[2] = {0, 0};
            if (!MWBridge::get()->IsMenu()) {
                const float f = 0.02f;
                const float windScaling = 1.0f; // Same as distant land
                const float* wind = MWBridge::get()->GetWindVector();
                smoothWind[0] += f * (windScaling * wind[0] - smoothWind[0]);
                smoothWind[1] += f * (windScaling * wind[1] - smoothWind[1]);
            }
            hlslShader.vsConstantTable->SetFloatArray(device, hWindVec, smoothWind, 2);
        }
        
        // Time for animations
        D3DXHANDLE hTime = hlslShader.vsConstantTable->GetConstantByName(NULL, "time");
        if (hTime) {
            hlslShader.vsConstantTable->SetFloat(device, hTime, MWBridge::get()->simulationTime());
        }
        
        D3DXHANDLE hHasVCol = hlslShader.psConstantTable->GetConstantByName(NULL, "hasVCol");
        if (hHasVCol) {
            hlslShader.psConstantTable->SetBool(device, hHasVCol, (rs->fvf & D3DFVF_DIFFUSE) != 0);
        }
        
        D3DXHANDLE hMaterialAlpha = hlslShader.psConstantTable->GetConstantByName(NULL, "materialAlpha");
        if (hMaterialAlpha) {
            hlslShader.psConstantTable->SetFloat(device, hMaterialAlpha, frs->material.diffuse.a);
        }
        
        D3DXHANDLE hAlphaRef = hlslShader.psConstantTable->GetConstantByName(NULL, "alphaRef");
        if (hAlphaRef) {
            hlslShader.psConstantTable->SetFloat(device, hAlphaRef, rs->alphaRef / 255.0f);
        }
        
        // Set normres constant if HAS_PARAMH is defined and paramh texture is bound
        if (sk.hasParamH) {
            IDirect3DBaseTexture9* normalTexture;
            device->GetTexture(2, &normalTexture);  // Get texture from slot 2
            if (normalTexture && normalTexture->GetType() == D3DRTYPE_TEXTURE) {
                IDirect3DTexture9* tex = static_cast<IDirect3DTexture9*>(normalTexture);
                
                // Check cache first
                D3DXVECTOR2 normres;
                auto cacheIt = textureResolutionCache.find(tex);
                if (cacheIt != textureResolutionCache.end()) {
                    normres = cacheIt->second;
                } else {
                    // Not in cache - get dimensions and cache them
                    D3DSURFACE_DESC desc;
                    if (SUCCEEDED(tex->GetLevelDesc(0, &desc))) {
                        normres = D3DXVECTOR2((float)desc.Width, (float)desc.Height);
                        textureResolutionCache[tex] = normres;
                    } else {
                        normres = D3DXVECTOR2(1.0f, 1.0f);  // Default fallback
                    }
                }
                
                // Set normres constant if it exists in pixel shader
                D3DXHANDLE hNormres = hlslShader.psConstantTable->GetConstantByName(NULL, "normres");
                if (hNormres) {
                    hlslShader.psConstantTable->SetFloatArray(device, hNormres, (float*)&normres, 2);
                } else {
                    // LOG::logline("HLSL: normres constant not found in pixel shader");
                }
                
                normalTexture->Release();
            }
        }
        } catch (...) {
            // Shader invalidated during file edit - return early
            LOG::logline("!! HLSL Pixel shader constant table access failed - shader may have been edited");
            return;
        }
    }
    
    // Error checking for vertex/index buffers
    if (!rs->vb) {
        // LOG::logline("!! HLSL pipeline: null vertex buffer, skipping draw call");
        return;
    }
    
    // Set vertex declaration and stream sources with error checking
    HRESULT hr = device->SetFVF(rs->fvf);
    if (FAILED(hr)) {
        LOG::logline("!! HLSL pipeline: failed to set FVF %x, hr=%x", rs->fvf, hr);
        return;
    }
    
    hr = device->SetStreamSource(0, rs->vb, rs->vbOffset, rs->vbStride);
    if (FAILED(hr)) {
        LOG::logline("!! HLSL pipeline: failed to set vertex buffer, hr=%x", hr);
        return;
    }

    // Phase A: Add small depth bias to resolve Z-fighting with depth prepass
    // Use a very small bias to push HLSL geometry slightly forward
    device->SetRenderState(D3DRS_DEPTHBIAS, *(DWORD*)&(const float&)-1e-6f);
    device->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, *(DWORD*)&(const float&)-1e-6f);

    // Execute the draw call with proper error checking
    if (rs->ib) {
        hr = device->SetIndices(rs->ib);
        if (FAILED(hr)) {
            LOG::logline("!! HLSL pipeline: failed to set index buffer, hr=%x", hr);
            return;
        }
        device->DrawIndexedPrimitive(rs->primType, rs->baseIndex, rs->minIndex, rs->vertCount, rs->startIndex, rs->primCount);
    } else {
        device->DrawPrimitive(rs->primType, rs->startIndex, rs->primCount);
    }

    // Reset depth bias after drawing
    device->SetRenderState(D3DRS_DEPTHBIAS, 0);
    device->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, 0);

    // Restore device state after HLSL rendering (like the original system does)
    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);

    // During replay, texture slots are managed by bindShaderTextures and cleaned up
    // at batch end in stopRecordingAndReplay. Per-call clearing would break the
    // bindingCache optimization (consecutive same-texture calls skip rebinding).
    if (!isReplaying) {
        // Clear HLSL texture bindings to prevent leaks across calls
        FixedFunctionShader::setCachedTexture(device, 2, nullptr);  // Clear paramH/suffix texture
        FixedFunctionShader::setCachedTexture(device, 3, nullptr);  // Clear any legacy MGE effects shadow binding
        FixedFunctionShader::setCachedTexture(device, 4, nullptr);  // Clear HLSL shadow binding

        // Restore render states that HLSL rendering may have changed
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, savedAlphaBlendEnable);
        device->SetRenderState(D3DRS_ALPHATESTENABLE, savedAlphaTestEnable);
        device->SetRenderState(D3DRS_ZENABLE, savedZEnable);
        device->SetRenderState(D3DRS_ZWRITEENABLE, savedZWriteEnable);
        device->SetRenderState(D3DRS_SPECULARENABLE, savedSpecularEnable);
        device->SetRenderState(D3DRS_LOCALVIEWER, savedLocalViewer);
        device->SetRenderState(D3DRS_NORMALIZENORMALS, savedNormalizeNormals);
    }
}

FixedFunctionShader::HLSLShader FixedFunctionShader::createPurpleErrorShader() {
    HLSLShader errorShader = {};
    
    // Minimal vertex shader - just transforms position
    const char* vsCode = 
        "float4x4 proj;\n"
        "float4x4 worldview;\n"
        "struct VS_INPUT { float4 pos : POSITION; };\n"
        "struct VS_OUTPUT { float4 position : POSITION; };\n"
        "VS_OUTPUT vs_main(VS_INPUT input) {\n"
        "    VS_OUTPUT output;\n"
        "    float4 worldPos = mul(input.pos, worldview);\n"
        "    output.position = mul(worldPos, proj);\n"
        "    return output;\n"
        "}\n";
    
    // Minimal pixel shader - just returns purple and includes lighting constants to prevent errors
    const char* psCode = 
        "// Dummy lighting constants to prevent lookup errors\n"
        "float3 lightSceneAmbient;\n"
        "float3 lightSunDiffuse;\n"
        "float3 lightSunDirection;\n"
        "float4 lightDiffuse[8];\n"
        "float3 lightPosition[8];\n"
        "float lightAmbient[8];\n"
        "int pointLightCount;\n"
        "struct VS_OUTPUT { float4 position : POSITION; };\n"
        "float4 ps_main(VS_OUTPUT input) : COLOR {\n"
        "    return float4(1.0, 0.0, 1.0, 1.0); // Purple error color\n"
        "}\n";
    
    // Compile vertex shader
    ID3DBlob* vsBlob = nullptr;
    ID3DBlob* vsErrors = nullptr;
    DWORD vsCompileFlags = isDXVK() ? D3DCOMPILE_OPTIMIZATION_LEVEL1 : D3DCOMPILE_OPTIMIZATION_LEVEL3;
    HRESULT hr = D3DCompile(vsCode, strlen(vsCode), "ErrorShader.hlsl", nullptr, nullptr, 
                           "vs_main", "vs_3_0", vsCompileFlags, 0, &vsBlob, &vsErrors);
    
    if (SUCCEEDED(hr)) {
        hr = device->CreateVertexShader(reinterpret_cast<DWORD*>(vsBlob->GetBufferPointer()), &errorShader.vertexShader);
        if (SUCCEEDED(hr)) {
            D3DXGetShaderConstantTable(reinterpret_cast<DWORD*>(vsBlob->GetBufferPointer()), &errorShader.vsConstantTable);

            // Cache vertex shader constant handles for error shader
            if (errorShader.vsConstantTable) {
                errorShader.hWorldViewProj = errorShader.vsConstantTable->GetConstantByName(NULL, "worldViewProj");
                errorShader.hView = errorShader.vsConstantTable->GetConstantByName(NULL, "view");
                errorShader.hProj = errorShader.vsConstantTable->GetConstantByName(NULL, "proj");
                errorShader.hWorld = errorShader.vsConstantTable->GetConstantByName(NULL, "world");
                errorShader.hWorldView = errorShader.vsConstantTable->GetConstantByName(NULL, "worldview");
                errorShader.hVertexBlendPalette = errorShader.vsConstantTable->GetConstantByName(NULL, "vertexBlendPalette");
                errorShader.hVertexBlendState = errorShader.vsConstantTable->GetConstantByName(NULL, "vertexBlendState");
                errorShader.hShadowWorldViewProj = errorShader.vsConstantTable->GetConstantByName(NULL, "shadowWorldViewProj");
            }
        }
        vsBlob->Release();
    }
    if (vsErrors) vsErrors->Release();
    
    // Compile pixel shader
    ID3DBlob* psBlob = nullptr;
    ID3DBlob* psErrors = nullptr;
    DWORD psCompileFlags = isDXVK() ? D3DCOMPILE_OPTIMIZATION_LEVEL1 : D3DCOMPILE_OPTIMIZATION_LEVEL3;
    hr = D3DCompile(psCode, strlen(psCode), "ErrorShader.hlsl", nullptr, nullptr, 
                   "ps_main", "ps_3_0", psCompileFlags, 0, &psBlob, &psErrors);
    
    if (SUCCEEDED(hr)) {
        hr = device->CreatePixelShader(reinterpret_cast<DWORD*>(psBlob->GetBufferPointer()), &errorShader.pixelShader);
        if (SUCCEEDED(hr)) {
            D3DXGetShaderConstantTable(reinterpret_cast<DWORD*>(psBlob->GetBufferPointer()), &errorShader.psConstantTable);

            // Cache pixel shader constant handles for error shader
            if (errorShader.psConstantTable) {
                errorShader.hMaterialDiffuse = errorShader.psConstantTable->GetConstantByName(NULL, "materialDiffuse");
                errorShader.hMaterialAmbient = errorShader.psConstantTable->GetConstantByName(NULL, "materialAmbient");
                errorShader.hMaterialEmissive = errorShader.psConstantTable->GetConstantByName(NULL, "materialEmissive");

                // Cache additional handles for error shader
                errorShader.hLightSunDirection = errorShader.psConstantTable->GetConstantByName(NULL, "lightSunDirection");
                errorShader.hLightSunDiffuse = errorShader.psConstantTable->GetConstantByName(NULL, "lightSunDiffuse");
                errorShader.hLightSceneAmbient = errorShader.psConstantTable->GetConstantByName(NULL, "lightSceneAmbient");
                errorShader.hShadowRcpRes = errorShader.psConstantTable->GetConstantByName(NULL, "shadowRcpRes");
                errorShader.hPCFFilterSize = errorShader.psConstantTable->GetConstantByName(NULL, "PCFFilterSize");
            }
        }
        psBlob->Release();
    }
    if (psErrors) psErrors->Release();
    
    LOG::logline("-- Created HLSL purple error shader");
    return errorShader;
}

// Helper function to load shader source from file
char* FixedFunctionShader::loadShaderFile(const char* filename, DWORD* outFileSize) {
    std::string key(filename);
    
    // Check if we have cached version
    auto it = shaderSourceCache.find(key);
    if (it != shaderSourceCache.end()) {
        // Check if file has been modified since we cached it
        WIN32_FIND_DATAA findData;
        HANDLE hFind = FindFirstFileA(filename, &findData);
        if (hFind != INVALID_HANDLE_VALUE) {
            FindClose(hFind);
            if (CompareFileTime(&it->second.lastWriteTime, &findData.ftLastWriteTime) == 0) {
                // File unchanged, return cached version
                if (outFileSize) *outFileSize = it->second.size;
                char* cachedCopy = new char[it->second.size + 1];
                memcpy(cachedCopy, it->second.source, it->second.size);
                cachedCopy[it->second.size] = '\0';
                return cachedCopy;
            } else {
                // File changed! Mark for recompilation but don't clear cache immediately
                delete[] it->second.source;
                shaderSourceCache.erase(it);
                // Set flag to clear cache after current compilation operations complete
                needsCacheReset = true;
            }
        }
    }
    
    // Get file attributes for initial caching
    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA(filename, &findData);
    if (hFind == INVALID_HANDLE_VALUE) {
        LOG::logline("!! HLSL file not found: %s", filename);
        return nullptr;
    }
    FindClose(hFind);
    
    // Read file from disk
    HANDLE hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        LOG::logline("!! HLSL file read error: %s", filename);
        return nullptr;
    }
    
    DWORD fileSize = GetFileSize(hFile, nullptr);
    char* shaderSource = new char[fileSize + 1];
    DWORD bytesRead;
    ReadFile(hFile, shaderSource, fileSize, &bytesRead, nullptr);
    shaderSource[fileSize] = '\0';
    CloseHandle(hFile);
    
    // Cache the result
    CachedShaderSource cached;
    cached.source = new char[fileSize + 1];
    memcpy(cached.source, shaderSource, fileSize);
    cached.source[fileSize] = '\0';
    cached.size = fileSize;
    cached.lastWriteTime = findData.ftLastWriteTime;
    shaderSourceCache[key] = cached;
    
    if (outFileSize) *outFileSize = fileSize;
    return shaderSource;
}

void FixedFunctionShader::invalidateShaderSourceCache() {
    // Clear shader source cache for hot reloading
    for (auto& i : shaderSourceCache) {
        delete[] i.second.source;
    }
    shaderSourceCache.clear();
    
    // Also clear compiled shader cache to force recompilation
    for (auto& i : cacheHLSLShaders) {
        if (i.second.vertexShader) i.second.vertexShader->Release();
        if (i.second.pixelShader) i.second.pixelShader->Release();
        if (i.second.vsConstantTable) i.second.vsConstantTable->Release();
        if (i.second.psConstantTable) i.second.psConstantTable->Release();
    }
    cacheHLSLShaders.clear();
}

void FixedFunctionShader::startAsyncCompiler() {
    shutdownCompiler = false;
    compilerThread = std::thread([]() {
        LOG::logline("-- HLSL Async compiler thread started");
        
        while (!shutdownCompiler) {
            std::shared_ptr<AsyncShaderRequest> request;
            
            // Wait for work or shutdown signal
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCondition.wait(lock, []() { 
                    return !compilationQueue.empty() || shutdownCompiler; 
                });
                
                if (shutdownCompiler && compilationQueue.empty()) {
                    break;
                }
                
                if (!compilationQueue.empty()) {
                    request = compilationQueue.front();
                    compilationQueue.pop();
                }
            }
            
            if (request) {
                // Compile shader in background thread
                request->result = generateMWShaderHLSL(request->key);
                request->completed = true;
                
                // LOG::logline("-- HLSL Async compiled shader with flags diffparam=%d normal=%d param=%d", 
                //            request->key.hasDiffParam, request->key.hasNormal, request->key.hasParam);
            }
        }
        
        LOG::logline("-- HLSL Async compiler thread stopped");
    });
}

void FixedFunctionShader::stopAsyncCompiler() {
    shutdownCompiler = true;
    queueCondition.notify_all();
    
    if (compilerThread.joinable()) {
        compilerThread.join();
    }
    
    // Clear any pending work
    std::lock_guard<std::mutex> lock(queueMutex);
    while (!compilationQueue.empty()) {
        compilationQueue.pop();
    }
    pendingCompilations.clear();
}

void FixedFunctionShader::queueShaderCompilation(const ShaderKey& key) {
    // Check if already pending
    if (pendingCompilations.find(key) != pendingCompilations.end()) {
        return;
    }
    
    auto request = std::make_shared<AsyncShaderRequest>(key);
    pendingCompilations[key] = request;
    
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        compilationQueue.push(request);
    }
    queueCondition.notify_one();
    
    // LOG::logline("-- HLSL Queued async compilation for shader with flags diffparam=%d normal=%d param=%d", 
    //            key.hasDiffParam, key.hasNormal, key.hasParam);
}

void FixedFunctionShader::processAsyncCompletions() {
    // Check for completed compilations and move them to the main cache
    auto it = pendingCompilations.begin();
    while (it != pendingCompilations.end()) {
        if (it->second->completed) {
            // Move completed shader to main cache
            cacheHLSLShaders[it->first] = it->second->result;
            
            // LOG::logline("-- HLSL Async shader ready, added to cache with flags diffparam=%d normal=%d param=%d", 
            //            it->first.hasDiffParam, it->first.hasNormal, it->first.hasParam);
            
            it = pendingCompilations.erase(it);
        } else {
            ++it;
        }
    }
}

void FixedFunctionShader::checkForShaderFileChanges() {
    // Check if any cached shader files have been modified
    const char* shaderFiles[] = {
        "Data Files\\shaders\\core-hlsl\\XE FixedFuncEmu_VS.hlsl",
        "Data Files\\shaders\\core-hlsl\\XE FixedFuncEmu_PS.hlsl"
    };
    
    bool anyChanged = false;
    for (const char* filename : shaderFiles) {
        std::string key(filename);
        auto it = shaderSourceCache.find(key);
        if (it != shaderSourceCache.end()) {
            // Check file modification time
            WIN32_FIND_DATAA findData;
            HANDLE hFind = FindFirstFileA(filename, &findData);
            if (hFind != INVALID_HANDLE_VALUE) {
                FindClose(hFind);
                if (CompareFileTime(&it->second.lastWriteTime, &findData.ftLastWriteTime) != 0) {
                    anyChanged = true;
                    break;
                }
            }
        }
    }
    
    if (anyChanged) {
        LOG::logline("-- HLSL shader files changed, invalidating cache for hot reload");
        invalidateShaderSourceCache();
    }
}

FixedFunctionShader::HLSLShader FixedFunctionShader::generateMWShaderHLSL(const ShaderKey& sk) {
    // Check if we need to clear the shader cache due to file changes
    if (needsCacheReset) {
        cacheHLSLShaders.clear();
        needsCacheReset = false;
    }
    
    HLSLShader hlslShader = {};
    
    // Shader entry points
    const char* vertexShaderName = "vs_main";
    const char* pixelShaderName = "ps_main";
    
    // Load separate vertex and pixel shader files
    DWORD vsFileSize = 0, psFileSize = 0;
    char* vertexShaderSource = loadShaderFile("Data Files\\shaders\\core-hlsl\\XE FixedFuncEmu_VS.hlsl", &vsFileSize);
    char* pixelShaderSource = loadShaderFile("Data Files\\shaders\\core-hlsl\\XE FixedFuncEmu_PS.hlsl", &psFileSize);
    
    if (!vertexShaderSource || !pixelShaderSource) {
        if (vertexShaderSource) delete[] vertexShaderSource;
        if (pixelShaderSource) delete[] pixelShaderSource;
        return hlslShaderDefaultPurple;
    }

    // Build shader defines based on ShaderKey
    D3D_SHADER_MACRO defines[10] = {};  // Increased to 10 for USE_TEXTURE_LIGHTS
    int defineCount = 0;

    // Always enable texture-based lighting system
    defines[defineCount++] = {"USE_TEXTURE_LIGHTS", "1"};

    if (sk.hasDiffParam) {
        defines[defineCount++] = {"HAS_DIFFPARAM", "1"};
        // LOG::logline("HLSL: Compiling with HAS_DIFFPARAM define");
    }
    if (sk.hasParamH) {
        defines[defineCount++] = {"HAS_PARAMH", "1"};
        // LOG::logline("HLSL: Compiling with HAS_PARAMH define");
    }
    if (sk.hasParamX) {
        defines[defineCount++] = {"HAS_PARAMX", "1"};
        // LOG::logline("HLSL: Compiling with HAS_PARAMX define");
    }
    if (sk.hasGrass) {
        defines[defineCount++] = {"HAS_GRASS", "1"};
        // LOG::logline("HLSL: Compiling with HAS_GRASS define");
    }
    if (sk.hasShadows) {
        defines[defineCount++] = {"HAS_SHADOWS", "1"};
        // LOG::logline("HLSL: Compiling with HAS_SHADOWS define");
    }
    if (sk.hasDetail) {
        defines[defineCount++] = {"HAS_DETAIL", "1"};
        // LOG::logline("HLSL: Compiling with HAS_DETAIL define");
    }
    if (!sk.useLighting) {
        defines[defineCount++] = {"NOLIT", "1"};
        // LOG::logline("HLSL: Compiling with NOLIT define (unlit shader)");
    }
    if (sk.noPointLights && sk.useLighting) {
        defines[defineCount++] = {"NO_POINT_LIGHTS", "1"};
        // LOG::logline("HLSL: Compiling with NO_POINT_LIGHTS define (wilderness shader)");
    }
    defines[defineCount] = {nullptr, nullptr}; // Null terminator
    
    // LOG::logline("HLSL: Compiling shader with %d defines", defineCount);
    
    // Compile vertex shader
    // TODO: Optimize - vertex shaders don't use texture suffix defines (HAS_DIFFPARAM, HAS_NORMAL, HAS_PARAM)
    // Many vertex shaders could be cached and reused across pixel shader variants
    ID3DBlob* vsBlob = nullptr;
    ID3DBlob* vsErrors = nullptr;

    // Match D3DX9 effect compilation - no IEEE_STRICTNESS for invariance with depth pass
    DWORD vsCompileFlags = D3DCOMPILE_PREFER_FLOW_CONTROL;
    if (isDXVK()) {
        vsCompileFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL1;
    } else {
        vsCompileFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
    }
    
    HRESULT hr = D3DCompile(
        vertexShaderSource,
        vsFileSize,
        "XE FixedFuncEmu_VS.hlsl",
        defines, // Pass suffix texture defines
        HLSLShaderManager::getIncludeHandler(), // Include handler for #include support
        vertexShaderName,
        "vs_3_0",
        vsCompileFlags,
        0,
        &vsBlob,
        &vsErrors
    );
    
    if (FAILED(hr)) {
        if (vsErrors) {
            LOG::write("!! HLSL Vertex Shader compile errors:\n");
            LOG::write(reinterpret_cast<const char*>(vsErrors->GetBufferPointer()));
            LOG::write("\n");
            vsErrors->Release();
        }
        LOG::logline("!! HLSL Vertex Shader compilation failed, using default");
        delete[] vertexShaderSource;
        delete[] pixelShaderSource;
        return hlslShaderDefaultPurple;
    }
    
    // Create vertex shader
    hr = device->CreateVertexShader(
        reinterpret_cast<DWORD*>(vsBlob->GetBufferPointer()),
        &hlslShader.vertexShader
    );
    
    if (FAILED(hr)) {
        LOG::logline("!! Failed to create HLSL vertex shader");
        vsBlob->Release();
        delete[] vertexShaderSource;
        delete[] pixelShaderSource;
        return hlslShaderDefaultPurple;
    }
    
    // Get constant table for vertex shader
    hr = D3DXGetShaderConstantTable(
        reinterpret_cast<DWORD*>(vsBlob->GetBufferPointer()),
        &hlslShader.vsConstantTable
    );

    // Cache vertex shader constant handles to avoid per-draw string lookups
    if (hlslShader.vsConstantTable) {
        hlslShader.hWorldViewProj = hlslShader.vsConstantTable->GetConstantByName(NULL, "worldViewProj");
        hlslShader.hView = hlslShader.vsConstantTable->GetConstantByName(NULL, "view");
        hlslShader.hProj = hlslShader.vsConstantTable->GetConstantByName(NULL, "proj");
        hlslShader.hWorld = hlslShader.vsConstantTable->GetConstantByName(NULL, "world");
        hlslShader.hWorldView = hlslShader.vsConstantTable->GetConstantByName(NULL, "worldview");
        hlslShader.hVertexBlendPalette = hlslShader.vsConstantTable->GetConstantByName(NULL, "vertexBlendPalette");
        hlslShader.hVertexBlendState = hlslShader.vsConstantTable->GetConstantByName(NULL, "vertexBlendState");
        hlslShader.hShadowWorldViewProj = hlslShader.vsConstantTable->GetConstantByName(NULL, "shadowWorldViewProj");
    }

    // Log VS blob size before releasing
    // LOG::logline("-- HLSL VS blob size: %u bytes", vsBlob->GetBufferSize());

    vsBlob->Release();
    
    // Compile pixel shader using pixel shader source
    ID3DBlob* psBlob = nullptr;
    ID3DBlob* psErrors = nullptr;

    // Match D3DX9 effect compilation - no IEEE_STRICTNESS for invariance with depth pass
    DWORD psCompileFlags = D3DCOMPILE_PREFER_FLOW_CONTROL;
    if (isDXVK()) {
        psCompileFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL1;
    } else {
        psCompileFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
    }
    
    hr = D3DCompile(
        pixelShaderSource,
        psFileSize,
        "XE FixedFuncEmu_PS.hlsl",
        defines, // Pass same suffix texture defines
        HLSLShaderManager::getIncludeHandler(), // Include handler for #include support
        pixelShaderName,
        "ps_3_0",
        psCompileFlags,
        0,
        &psBlob,
        &psErrors
    );
    
    if (FAILED(hr)) {
        if (psErrors) {
            LOG::write("!! HLSL Pixel Shader compile errors:\n");
            LOG::write(reinterpret_cast<const char*>(psErrors->GetBufferPointer()));
            LOG::write("\n");
            psErrors->Release();
        }
        LOG::logline("!! HLSL Pixel Shader compilation failed, using default");
        // Clean up vertex shader
        if (hlslShader.vertexShader) hlslShader.vertexShader->Release();
        if (hlslShader.vsConstantTable) hlslShader.vsConstantTable->Release();
        delete[] vertexShaderSource;
        delete[] pixelShaderSource;
        return hlslShaderDefaultPurple;
    }
    
    // Create pixel shader
    hr = device->CreatePixelShader(
        reinterpret_cast<DWORD*>(psBlob->GetBufferPointer()),
        &hlslShader.pixelShader
    );
    
    if (FAILED(hr)) {
        LOG::logline("!! Failed to create HLSL pixel shader");
        psBlob->Release();
        // Clean up vertex shader
        if (hlslShader.vertexShader) hlslShader.vertexShader->Release();
        if (hlslShader.vsConstantTable) hlslShader.vsConstantTable->Release();
        delete[] vertexShaderSource;
        delete[] pixelShaderSource;
        return hlslShaderDefaultPurple;
    }
    
    // Get constant table for pixel shader
    hr = D3DXGetShaderConstantTable(
        reinterpret_cast<DWORD*>(psBlob->GetBufferPointer()),
        &hlslShader.psConstantTable
    );

    if (FAILED(hr) || !hlslShader.psConstantTable) {
        LOG::logline("HLSL: Failed to extract pixel shader constant table, hr=%x", hr);
    } else {
        // Cache pixel shader constant handles to avoid per-draw string lookups
        hlslShader.hMaterialDiffuse = hlslShader.psConstantTable->GetConstantByName(NULL, "materialDiffuse");
        hlslShader.hMaterialAmbient = hlslShader.psConstantTable->GetConstantByName(NULL, "materialAmbient");
        hlslShader.hMaterialEmissive = hlslShader.psConstantTable->GetConstantByName(NULL, "materialEmissive");

        // Cache additional lighting and shader constant handles
        hlslShader.hLightSunDirection = hlslShader.psConstantTable->GetConstantByName(NULL, "lightSunDirection");
        hlslShader.hLightSunDiffuse = hlslShader.psConstantTable->GetConstantByName(NULL, "lightSunDiffuse");
        hlslShader.hLightSceneAmbient = hlslShader.psConstantTable->GetConstantByName(NULL, "lightSceneAmbient");
        hlslShader.hShadowRcpRes = hlslShader.psConstantTable->GetConstantByName(NULL, "shadowRcpRes");
        hlslShader.hPCFFilterSize = hlslShader.psConstantTable->GetConstantByName(NULL, "PCF_filterSize");
    }
    
    // Log compilation details for debugging (before releasing blobs)
    // LOG::logline("-- HLSL shader compiled successfully: VS=%s PS=%s", vertexShaderName, pixelShaderName);
    // LOG::logline("-- HLSL PS blob size: %u bytes", psBlob->GetBufferSize());
    
    psBlob->Release();
    
    // Clean up shader sources
    delete[] vertexShaderSource;
    delete[] pixelShaderSource;
    
    // Log shader key details
    // LOG::logline("-- HLSL ShaderKey: hasDiffParam=%d hasNormal=%d hasParam=%d", sk.hasDiffParam, sk.hasNormal, sk.hasParam);
    
    // Cache the compiled shader
    cacheHLSLShaders[sk] = hlslShader;
    
    return hlslShader;
}

void FixedFunctionShader::release() {
    // Stop async compiler thread
    stopAsyncCompiler();
    
    // Clean up D3DXEffect cache
    for (auto& i : cacheEffects) {
        if (i.second) {
            i.second->Release();
        }
    }

    shaderLRU.effect = nullptr;
    shaderLRU.last_sk = ShaderKey();
    cacheEffects.clear();
    if (effectDefaultPurple) {
    effectDefaultPurple->Release();
        effectDefaultPurple = nullptr;
    }
    
    // Clean up HLSL cache with safety checks
    for (auto& i : cacheHLSLShaders) {
        try {
            // Extra safety: validate pointers before release using COM object validation
            if (i.second.vertexShader) {
                i.second.vertexShader->Release();
                i.second.vertexShader = nullptr;
            }

            if (i.second.pixelShader) {
                i.second.pixelShader->Release();
                i.second.pixelShader = nullptr;
            }
            if (i.second.vsConstantTable) {
                ULONG refCount = i.second.vsConstantTable->AddRef();
                if (refCount > 1) {
                    i.second.vsConstantTable->Release(); // Remove our AddRef
                    i.second.vsConstantTable->Release(); // Original release
                } else {
                    i.second.vsConstantTable->Release(); // Just remove our AddRef
                }
                i.second.vsConstantTable = nullptr;
            }

            if (i.second.psConstantTable) {
                i.second.psConstantTable->Release();
                i.second.psConstantTable = nullptr;
            }
        } catch (...) {
            // Ignore exceptions during cleanup
        }
    }

    hlslShaderLRU.shader = {};
    hlslShaderLRU.last_sk = ShaderKey();
    cacheHLSLShaders.clear();
    
    // Clean up default HLSL shader with null checks
    if (hlslShaderDefaultPurple.vertexShader) {
        hlslShaderDefaultPurple.vertexShader->Release();
        hlslShaderDefaultPurple.vertexShader = nullptr;
    }
    if (hlslShaderDefaultPurple.pixelShader) {
        hlslShaderDefaultPurple.pixelShader->Release();
        hlslShaderDefaultPurple.pixelShader = nullptr;
    }
    if (hlslShaderDefaultPurple.vsConstantTable) {
        hlslShaderDefaultPurple.vsConstantTable->Release();
        hlslShaderDefaultPurple.vsConstantTable = nullptr;
    }
    if (hlslShaderDefaultPurple.psConstantTable) {
        hlslShaderDefaultPurple.psConstantTable->Release();
        hlslShaderDefaultPurple.psConstantTable = nullptr;
    }
    
    // Clean up shader source cache
    for (auto& i : shaderSourceCache) {
        delete[] i.second.source;
    }
    shaderSourceCache.clear();

    // Reset material state cache
    materialCache.reset();

}

void FixedFunctionShader::resetHLSLCaches() {
    // Note: Recording/replay happens within same frame during HLSL pipeline
    // This function should NOT interfere with the recording system

    // Reset material cache for HLSL rendering session to avoid stale state
    materialCache.reset();

    // Reset texture binding cache for HLSL rendering session
    textureCache.reset();

    // Reset exterior texture binding optimizations
    isExteriorShadowBound = false;
    isDetailTextureBound = false;

    // Reset shadow matrix cache validity
    shadowMatricesValid = false;

    // Reset suffix binding cache to prevent stale texture pointers across frames
    bindingCache.lastBaseTexture = nullptr;
    bindingCache.currentBaseTextureName.clear();
    bindingCache.boundDiffParam = nullptr;
    bindingCache.boundParamH = nullptr;
    bindingCache.boundParamX = nullptr;
}



// ShaderKey - Captures a generatable shader configuration

FixedFunctionShader::ShaderKey::ShaderKey(const RenderedState* rs, const FragmentState* frs, const LightState* lightrs) {
    memset(this, 0, sizeof(ShaderKey));         // Clear padding bits for compares

    uvSets = (rs->fvf & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT;
    usesSkinning = rs->vertexBlendState ? 1 : 0;
    vertexColour = (rs->fvf & D3DFVF_DIFFUSE) ? 1 : 0;
    useLighting = rs->useLighting ? 1 : 0;
    
    // Count point lights specifically (exclude directional lights like sun)
    int pointLightCount = 0;
    int directionalLightCount = 0;
    if (rs->useLighting) {
        for (DWORD lightId : lightrs->active) {
            auto lightIt = lightrs->lights.find(lightId);
            if (lightIt != lightrs->lights.end()) {
                if (lightIt->second.type == D3DLIGHT_POINT) {
                    pointLightCount++;
                } else if (lightIt->second.type == D3DLIGHT_DIRECTIONAL) {
                    directionalLightCount++;
                }
            }
        }
    }
    noPointLights = (rs->useLighting && pointLightCount == 0) ? 1 : 0;

    // Match constant material, diffuse+ambient vcol, or emissive vcol
    if (rs->useLighting) {
        heavyLighting = (lightrs->active.size() > 4) ? 1 : 0;
        vertexMaterial = 1;

        if (vertexColour) {
            if (rs->matSrcDiffuse == D3DMCS_COLOR1) {
                vertexMaterial = 2;
            } else if (rs->matSrcEmissive == D3DMCS_COLOR1) {
                vertexMaterial = 3;
            }
        }
    }

    if (rs->useFog) {
        // Match premultipled alpha or additive blending
        if (rs->blendEnable && (rs->srcBlend == D3DBLEND_ONE || rs->destBlend == D3DBLEND_ONE)) {
            fogMode = 2;
        } else {
            fogMode = 1;
        }
    }

    DWORD maxTexcoordIndex = 0;
    bool bumpStageFixup = false;

    for (int i = 0; i != 8; ++i) {
        const FragmentState::Stage& s = frs->stage[i];

        if (s.colorOp == D3DTOP_DISABLE) {
            activeStages = i;
            break;
        }

        stage[i].colorOp = s.colorOp;
        stage[i].colorArg1 = s.colorArg1;
        stage[i].colorArg2 = s.colorArg2;
        stage[i].colorArg0 = s.colorArg0;
        stage[i].alphaOpMatched = (s.alphaOp == s.colorOp);
        stage[i].alphaOpSelect1 = (s.alphaOp == D3DTOP_SELECTARG1 && s.alphaArg1 == s.colorArg1);
        stage[i].texcoordIndex = s.texcoordIndex & 3;
        stage[i].texcoordGen = s.texcoordIndex >> 16;
        maxTexcoordIndex = std::max(maxTexcoordIndex, (DWORD)stage[i].texcoordIndex);

        if (s.colorOp == D3DTOP_BUMPENVMAP || s.colorOp == D3DTOP_BUMPENVMAPLUMINANCE) {
            usesBumpmap = 1;
            bumpmapStage = i;
            stage[i].alphaOpMatched = false;
            stage[i].alphaOpSelect1 = false;
            bumpStageFixup = true;
        } else if (bumpStageFixup) {
            stage[i].alphaOpMatched = false;
            stage[i].alphaOpSelect1 = false;
            bumpStageFixup = false;
        }

        if (stage[i].texcoordGen) {
            usesTexgen = 1;
            projectiveTexgen = (s.texTransformFlags == (D3DTTFF_COUNT3 | D3DTTFF_PROJECTED)) ? 1 : 0;
            texgenStage = i;
        }
    }

    // Generate based on actual UV sets available and used
    DWORD usedUVSets = maxTexcoordIndex + 1;
    uvSets = std::min((DWORD)uvSets, usedUVSets);
}

bool FixedFunctionShader::ShaderKey::operator<(const ShaderKey& other) const {
    return memcmp(this, &other, sizeof(ShaderKey)) < 0;
}

bool FixedFunctionShader::ShaderKey::operator==(const ShaderKey& other) const {
    return memcmp(this, &other, sizeof(ShaderKey)) == 0;
}

std::size_t FixedFunctionShader::ShaderKey::hasher::operator()(const ShaderKey& k) const {
    DWORD z[9];
    memcpy(&z, &k, sizeof(z));
    return (z[0] << 16) ^ z[1] ^ z[2] ^ z[3] ^ z[4] ^ z[5] ^ z[6] ^ z[7] ^ z[8];
}

void FixedFunctionShader::ShaderKey::log() const {
    const char* opSymbols[] = { "?", "disable", "select1", "select2", "mul", "mul2x", "mul4x", "add", "addsigned", "addsigned2x", "sub", "?", "blend.diffuse", "blend.texture", "?", "?", "?", "?", "?", "?", "?", "?", "bump", "bump.l", "dp3", "mad", "?" };
    const char* argSymbols[] = { "diffuse", "current", "texture", "tfactor", "specular", "temp", "constant" };
    const char* texgenSymbols[] = { "none", "normal", "position", "reflection", "sphere" };

    const unsigned char *dump = (const unsigned char*)this;
    stringstream stream;
    stream << "   Hex: ";
    for(int i = 0; i < sizeof *this; ++i) {
        char hex[4];
        snprintf(hex, sizeof hex, "%02x ", dump[i]);
        stream << hex;
    }
    // LOG::logline("%s", stream.str().c_str());

    // LOG::logline("   Input state: UVs:%d skin:%d vcol:%d lights:%d vmat:%d fogm:%d", uvSets, usesSkinning, vertexColour, vertexMaterial ? (heavyLighting ? 8 : 4) : 0, vertexMaterial, fogMode);
    // LOG::logline("   Texture stages:");
    for (int i = 0; i != activeStages; ++i) {
        const auto& s = stage[i];
        if (s.colorOp != D3DTOP_MULTIPLYADD) { // or D3DTOP_LERP (unused)
            LOG::logline("    [%d] %s % 12s    %s, %s            uv %d texgen %s", i,
                         s.alphaOpMatched ? "RGBA" : "RGB ",
                         opSymbols[s.colorOp], argSymbols[s.colorArg1], argSymbols[s.colorArg2],
                         s.texcoordIndex, texgenSymbols[s.texcoordGen]);
        } else {
            LOG::logline("    [%d] %s % 12s    %s, %s, %s   uv %d texgen %s", i,
                         s.alphaOpMatched ? "RGBA" : "RGB ",
                         opSymbols[s.colorOp], argSymbols[s.colorArg1], argSymbols[s.colorArg2], argSymbols[s.colorArg0],
                         s.texcoordIndex, texgenSymbols[s.texcoordGen]);
        }
        if (s.alphaOpSelect1) {
            LOG::logline("           A % 12s    %s", opSymbols[D3DTOP_SELECTARG1], argSymbols[s.colorArg1]);
        }
    }
    // LOG::logline("");
}

// HLSL Render Dispatch Recording System Implementation

// Frame counter for light persistence (prevent flickering)
static int g_currentFrame = 0;

// Sampler state cache: maps texture pointer to (addressU, addressV) pair
// Reduces D3D API calls from ~24 to 0-2 per draw call
static std::unordered_map<IDirect3DBaseTexture9*, std::pair<DWORD, DWORD>> samplerCache;

void FixedFunctionShader::startRecording() {
    // Save render states before recording so we can restore after replay
    device->GetRenderState(D3DRS_ALPHABLENDENABLE, &preRecordingState.alphaBlendEnable);
    device->GetRenderState(D3DRS_ALPHATESTENABLE, &preRecordingState.alphaTestEnable);
    device->GetRenderState(D3DRS_ZENABLE, &preRecordingState.zEnable);
    device->GetRenderState(D3DRS_ZWRITEENABLE, &preRecordingState.zWriteEnable);
    device->GetRenderState(D3DRS_CULLMODE, &preRecordingState.cullMode);
    device->GetRenderState(D3DRS_SRCBLEND, &preRecordingState.srcBlend);
    device->GetRenderState(D3DRS_DESTBLEND, &preRecordingState.destBlend);
    device->GetRenderState(D3DRS_FOGENABLE, &preRecordingState.fogEnable);
    device->GetRenderState(D3DRS_SPECULARENABLE, &preRecordingState.specularEnable);
    device->GetRenderState(D3DRS_LOCALVIEWER, &preRecordingState.localViewer);
    device->GetRenderState(D3DRS_NORMALIZENORMALS, &preRecordingState.normalizeNormals);
    device->GetRenderState(D3DRS_ZFUNC, &preRecordingState.zFunc);
    device->GetRenderState(D3DRS_ALPHAFUNC, &preRecordingState.alphaFunc);
    device->GetRenderState(D3DRS_ALPHAREF, &preRecordingState.alphaRef);

    // Reset HLSL caches for new recording session
    resetHLSLCaches();

    recordedCalls.clear();
    recordedCalls.reserve(4000);  // Pre-allocate to avoid reallocation spikes
    samplerCache.clear();  // Clear sampler cache for new frame
    bboxLookup.clear();  // Clear for new frame (populated during recording)
    // NOTE: Do NOT clear recordMW here - it's populated by inspectIndexedPrimitive()
    // BEFORE startRecording() is called. recordMW is cleared at end of renderStage1/2.
    lastLightState.reset();  // Clear LightState cache for new recording
    lastLightStatePtr = nullptr;  // Clear pointer cache

    // Clear lights from previous frame (simple approach, no persistence)
    DistantLand::sceneLights.clear();
    DistantLand::sceneLightIndexMap.clear();
    DistantLand::visibleLights.clear();

    isRecording = true;
    isReplaying = false;
    recordingCompletedThisFrame = false;  // Allow recording to proceed

    // Clear CPU depth buffer for new frame (replaces GPU Hi-Z readback)
    softwareOcclusionCuller.clear();

    // Capture view/projection matrices once at start of recording
    // Note: World transforms are captured per-call in each RenderedState
    device->GetTransform(D3DTS_VIEW, &recordingDeviceView);
    device->GetTransform(D3DTS_PROJECTION, &recordingDeviceProj);
    recordingShadowViewproj[0] = DistantLand::smViewproj[0];
    recordingShadowViewproj[1] = DistantLand::smViewproj[1];

    LOG::logline("HLSL Recording: Started recording render dispatches");
}

// prepareOcclusionCullingForDepth - Build Hi-Z pyramid and filter recordMW before depth rendering
// Called at the start of renderStage1(), BEFORE renderDepth() executes
void FixedFunctionShader::prepareOcclusionCullingForDepth() {
    ZoneScopedN("Prepare Occlusion Culling for Depth");

    // Resize visibility results for this frame (indexed by draw order)
    visibilityResults.assign(DistantLand::recordMW.size(), -1);  // -1 = not yet tested

    // Phase 2a: Compute deferred bboxes and rasterize occluders from recorded HLSL calls
    // This must happen before Hi-Z build so the depth pass benefits from culling
    if (!recordedCalls.empty() && !hiZBuiltThisFrame) {
        // Compute bounding boxes for calls that missed the cache during recording
        {
            ZoneScopedN("Prepare: BBox Cache Misses");
            for (auto& call : recordedCalls) {
                if (!call.hasBoundingBox) {
                    call.hasBoundingBox = computeBoundingBox(&call.rs, call.bboxMin, call.bboxMax);
                    if (call.hasBoundingBox) {
                        VBIBKey key{call.rs.vb, call.rs.ib};
                        bboxLookup[key] = {call.bboxMin, call.bboxMax};
                    }
                }
            }
        }

        // Occluder selection and rasterization
        {
            ZoneScopedN("Prepare: Occluder Rasterization");
            int occluderCount = 0;
            const int MAX_OCCLUDERS = ImGuiManager::GetOccluderMaxCount();
            const float MIN_SCREEN_COVERAGE = 0.01f;
            rasterizedOccluderMeshes.clear();

            D3DXMATRIX viewProj = recordingDeviceView * recordingDeviceProj;

            for (auto& call : recordedCalls) {
                if (occluderCount >= MAX_OCCLUDERS) break;

                const RenderedState* rs = &call.rs;
                bool isTransparent = (rs->alphaTest || rs->blendEnable);
                if (!rs->vb || !rs->ib || isTransparent || !call.hasBoundingBox) continue;

                float bboxSizeX = call.bboxMax.x - call.bboxMin.x;
                float bboxSizeY = call.bboxMax.y - call.bboxMin.y;
                float bboxSizeZ = call.bboxMax.z - call.bboxMin.z;
                float bboxSize = std::max({bboxSizeX, bboxSizeY, bboxSizeZ});

                const UINT BASE_TRIANGLES = 2;
                const UINT MAX_TRIANGLES = 400;
                const float MAX_SIZE = 3000.0f;
                float sizeRatio = std::min(bboxSize / MAX_SIZE, 1.0f);
                UINT allowedTriangles = BASE_TRIANGLES + (UINT)(sizeRatio * (MAX_TRIANGLES - BASE_TRIANGLES));

                if (rs->primCount > allowedTriangles) continue;

                // Screen coverage test
                D3DXVECTOR3 corners[8] = {
                    D3DXVECTOR3(call.bboxMin.x, call.bboxMin.y, call.bboxMin.z),
                    D3DXVECTOR3(call.bboxMax.x, call.bboxMin.y, call.bboxMin.z),
                    D3DXVECTOR3(call.bboxMin.x, call.bboxMax.y, call.bboxMin.z),
                    D3DXVECTOR3(call.bboxMax.x, call.bboxMax.y, call.bboxMin.z),
                    D3DXVECTOR3(call.bboxMin.x, call.bboxMin.y, call.bboxMax.z),
                    D3DXVECTOR3(call.bboxMax.x, call.bboxMin.y, call.bboxMax.z),
                    D3DXVECTOR3(call.bboxMin.x, call.bboxMax.y, call.bboxMax.z),
                    D3DXVECTOR3(call.bboxMax.x, call.bboxMax.y, call.bboxMax.z)
                };

                float minX = FLT_MAX, maxX = -FLT_MAX;
                float minY = FLT_MAX, maxY = -FLT_MAX;
                bool allBehindCamera = true;

                for (int i = 0; i < 8; i++) {
                    D3DXVECTOR4 clipPos;
                    D3DXVec3Transform(&clipPos, &corners[i], &viewProj);
                    if (clipPos.w > 0.0f) {
                        allBehindCamera = false;
                        float ndcX = std::max(-1.0f, std::min(1.0f, clipPos.x / clipPos.w));
                        float ndcY = std::max(-1.0f, std::min(1.0f, clipPos.y / clipPos.w));
                        minX = std::min(minX, ndcX);
                        maxX = std::max(maxX, ndcX);
                        minY = std::min(minY, ndcY);
                        maxY = std::max(maxY, ndcY);
                    }
                }

                if (allBehindCamera) continue;
                float screenCoverage = ((maxX - minX) * (maxY - minY)) / 4.0f;
                if (screenCoverage < MIN_SCREEN_COVERAGE) continue;

                MeshKey occluderKey;
                occluderKey.vb = rs->vb;
                occluderKey.ib = rs->ib;
                occluderKey.fvf = rs->fvf;
                occluderKey.baseIndex = rs->baseIndex;
                occluderKey.vertCount = rs->vertCount;
                occluderKey.startIndex = rs->startIndex;
                occluderKey.primCount = rs->primCount;
                rasterizedOccluderMeshes.insert(occluderKey);

                softwareOcclusionCuller.rasterizeMesh(
                    rs->vb, rs->vbOffset, rs->vbStride,
                    rs->ib, rs->baseIndex, rs->startIndex, rs->primCount, rs->primType,
                    rs->fvf, rs->worldTransforms[0],
                    recordingDeviceView, recordingDeviceProj
                );
                occluderCount++;
            }
        }
    }

    // Build Hi-Z mipmap pyramid from rasterized occluders (only once per frame)
    if (!hiZBuiltThisFrame) {
        ZoneScopedN("Build Hi-Z Pyramid");
        softwareOcclusionCuller.buildHiZPyramid();
        if (ImGuiManager::GetShowHiZInterface()) {
            softwareOcclusionCuller.uploadHiZToTexture(reinterpret_cast<IDirect3DDevice9*>(device), ImGuiManager::GetHiZDisplayMip(), ImGuiManager::GetHiZBrightness(), ImGuiManager::GetHiZGamma(), ImGuiManager::GetHiZInvert(), ImGuiManager::GetHiZShowRaycastGrid(), ImGuiManager::GetHiZRaycastStep());
        }
        hiZBuiltThisFrame = true;
    }

    // Only do Hi-Z culling in HLSL mode (PerPixelLightFlags == 2)
    // Non-HLSL modes don't have bboxCache populated, so skip culling
    if (Configuration.PerPixelLightFlags != 2) {
        LOG::logline(">> Depth: %d objects (no culling - non-HLSL mode)",
                     (int)DistantLand::recordMW.size());
        return;
    }

    // Direct Hi-Z culling for depth pass using bboxCache + current matrices
    {
        ZoneScopedN("Filter recordMW with Hi-Z Culling");

        // Get current matrices at depth-render time (not recording time!)
        D3DXMATRIX currentView, currentProj;
        device->GetTransform(D3DTS_VIEW, &currentView);
        device->GetTransform(D3DTS_PROJECTION, &currentProj);

        std::vector<DistantLand::RecordedState> filteredRecordMW;
        filteredRecordMW.reserve(DistantLand::recordMW.size());

        int visibleCount = 0, culledCount = 0, noBboxCount = 0;

        for (size_t i = 0; i < DistantLand::recordMW.size(); i++) {
            auto& obj = DistantLand::recordMW[i];
            bool isVisible = true;  // Default to visible

            // Compute world-space bbox for this entry using bboxCache
            D3DXVECTOR3 worldBboxMin, worldBboxMax;
            bool hasBbox = computeBoundingBox(&obj, worldBboxMin, worldBboxMax);

            if (hasBbox) {
                // Check if this is an occluder (never cull occluders)
                MeshKey key{obj.vb, obj.ib, obj.fvf, obj.baseIndex,
                            obj.vertCount, obj.startIndex, obj.primCount};
                bool isOccluder = (rasterizedOccluderMeshes.count(key) > 0);

                if (!isOccluder) {
                    // Hi-Z test using CURRENT matrices (same as bbox visualization)
                    isVisible = softwareOcclusionCuller.testBoundingBox(
                        worldBboxMin, worldBboxMax, currentView, currentProj);
                }

                // Store visibility by index
                visibilityResults[i] = isVisible ? 1 : 0;

                if (isVisible) visibleCount++; else culledCount++;
            } else {
                // No bbox - keep visible (conservative), mark as tested
                visibilityResults[i] = 1;
                noBboxCount++;
            }

            if (isVisible) {
                filteredRecordMW.push_back(std::move(obj));
            }
        }

        DistantLand::recordMW = std::move(filteredRecordMW);

        LOG::logline(">> Depth Hi-Z: %d visible, %d culled, %d no bbox (of %d total)",
                     visibleCount, culledCount, noBboxCount,
                     visibleCount + culledCount + noBboxCount);
    }
}

// Dirty tracking: match current frame calls to previous frame by MeshKey
void FixedFunctionShader::matchPreviousFrameCalls() {
    ZoneScopedN("matchPreviousFrameCalls");

    if (previousFrameCalls.empty()) {
        // First frame or no previous data — all dirty
        for (auto& call : recordedCalls) {
            call.dirtyFlags = DIRTY_ALL;
        }
        return;
    }

    // Build lookup from previous frame
    std::unordered_map<MeshKey, int, MeshKeyHash> prevLookup;
    prevLookup.reserve(previousFrameCalls.size());
    for (int i = 0; i < (int)previousFrameCalls.size(); ++i) {
        auto& prev = previousFrameCalls[i];
        MeshKey key;
        key.vb = prev.rs.vb;
        key.ib = prev.rs.ib;
        key.fvf = prev.rs.fvf;
        key.baseIndex = prev.rs.baseIndex;
        key.vertCount = prev.rs.vertCount;
        key.startIndex = prev.rs.startIndex;
        key.primCount = prev.rs.primCount;
        prevLookup[key] = i;  // Last wins for duplicates
    }

    for (auto& call : recordedCalls) {
        MeshKey key;
        key.vb = call.rs.vb;
        key.ib = call.rs.ib;
        key.fvf = call.rs.fvf;
        key.baseIndex = call.rs.baseIndex;
        key.vertCount = call.rs.vertCount;
        key.startIndex = call.rs.startIndex;
        key.primCount = call.rs.primCount;

        auto it = prevLookup.find(key);
        if (it == prevLookup.end()) {
            call.dirtyFlags = DIRTY_ALL;
            continue;
        }

        auto& prev = previousFrameCalls[it->second];
        DWORD flags = DIRTY_NONE;

        // Compare world transform (object moved?)
        if (memcmp(&call.rs.worldTransforms[0], &prev.rs.worldTransforms[0], sizeof(D3DXMATRIX)) != 0) {
            flags |= DIRTY_TRANSFORM;
        }

        // Compare light state (pointer comparison — shared_ptr reuse)
        if (call.lightrs.get() != prev.lightrs.get()) {
            flags |= DIRTY_LIGHT;
        }

        // Compare material
        if (memcmp(&call.frs.material, &prev.frs.material, sizeof(FragmentState::Material)) != 0) {
            flags |= DIRTY_MATERIAL;
        }

        // Compare shader key
        if (!(call.sk == prev.sk)) {
            flags |= DIRTY_SHADER;
        }

        // Compare blend state
        if (call.rs.blendEnable != prev.rs.blendEnable ||
            call.rs.srcBlend != prev.rs.srcBlend ||
            call.rs.destBlend != prev.rs.destBlend) {
            flags |= DIRTY_BLEND;
        }

        // Compare base texture
        if (call.rs.texture != prev.rs.texture) {
            flags |= DIRTY_TEXTURE;
        }

        call.dirtyFlags = flags;
    }
}

// Phase 2b: Prepare shader keys — runs after recording completes, before replay.
// BBox computation and occluder rasterization already done in prepareOcclusionCullingForDepth (Phase 2a).
void FixedFunctionShader::prepareRecordedCalls() {
    ZoneScopedN("prepareRecordedCalls");

    if (recordedCalls.empty()) {
        return;
    }

    // Compute shader keys (deferred from renderMorrowindHLSL recording path)
    {
        ZoneScopedN("Prepare: Shader Keys");
        for (auto& call : recordedCalls) {
            call.sk = computeShaderKeyWithSuffixes(&call.rs, &call.frs, call.lightrs.get());
            call.prepared = true;
        }
    }

    // Performance mode: match against previous frame for dirty tracking
    if (ImGuiManager::GetPerformanceMode()) {
        matchPreviousFrameCalls();
    }
}

void FixedFunctionShader::stopRecordingAndReplay() {
    if (!isRecording) {
        return;
    }

    isRecording = false;

    // Capture device state NOW — this is Morrowind's last mesh state (correct end-of-Scene-0 state).
    // We restore this after replay instead of preRecordingState (which was the FIRST mesh's state
    // and could have different alpha test/blend settings that corrupt the sky).
    SavedRenderStates postRecordingState;
    device->GetRenderState(D3DRS_ALPHABLENDENABLE, &postRecordingState.alphaBlendEnable);
    device->GetRenderState(D3DRS_ALPHATESTENABLE, &postRecordingState.alphaTestEnable);
    device->GetRenderState(D3DRS_ZENABLE, &postRecordingState.zEnable);
    device->GetRenderState(D3DRS_ZWRITEENABLE, &postRecordingState.zWriteEnable);
    device->GetRenderState(D3DRS_CULLMODE, &postRecordingState.cullMode);
    device->GetRenderState(D3DRS_SRCBLEND, &postRecordingState.srcBlend);
    device->GetRenderState(D3DRS_DESTBLEND, &postRecordingState.destBlend);
    device->GetRenderState(D3DRS_FOGENABLE, &postRecordingState.fogEnable);
    device->GetRenderState(D3DRS_SPECULARENABLE, &postRecordingState.specularEnable);
    device->GetRenderState(D3DRS_LOCALVIEWER, &postRecordingState.localViewer);
    device->GetRenderState(D3DRS_NORMALIZENORMALS, &postRecordingState.normalizeNormals);
    device->GetRenderState(D3DRS_ZFUNC, &postRecordingState.zFunc);
    device->GetRenderState(D3DRS_ALPHAFUNC, &postRecordingState.alphaFunc);
    device->GetRenderState(D3DRS_ALPHAREF, &postRecordingState.alphaRef);

    // Phase 2b: Prepare shader keys (bbox + occluders already done in prepareOcclusionCullingForDepth)
    prepareRecordedCalls();

    // Phase 3: Replay all prepared calls (Hi-Z already built by prepareOcclusionCullingForDepth)
    replayRecordedCalls(0);

    // Clear recorded calls after replay
    recordedCalls.clear();

    // Clean up HLSL-only texture slots to prevent DXVK descriptor bloat
    // Use raw SetTexture to truly unbind (setCachedTexture substitutes default textures)
    // Slots 0-1 are used by Morrowind normally, don't touch them
    // Slots 2-5 are HLSL-specific (paramH, paramX, shadow, lightData)
    for (int i = 2; i < 6; i++) {
        device->SetTexture(i, NULL);
        textureCache.updateCache(i, nullptr);
        textureCache.textureValid[i] = false;
    }
    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);

    // Restore Morrowind's end-of-Scene-0 state (last mesh state, not first mesh state).
    // This undoes any state changes from replay/HLSL rendering, and also cleans up
    // leaked state from previous frame's Scene 1+ immediate rendering.
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, postRecordingState.alphaBlendEnable);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, postRecordingState.alphaTestEnable);
    device->SetRenderState(D3DRS_ZENABLE, postRecordingState.zEnable);
    device->SetRenderState(D3DRS_ZWRITEENABLE, postRecordingState.zWriteEnable);
    device->SetRenderState(D3DRS_CULLMODE, postRecordingState.cullMode);
    device->SetRenderState(D3DRS_SRCBLEND, postRecordingState.srcBlend);
    device->SetRenderState(D3DRS_DESTBLEND, postRecordingState.destBlend);
    device->SetRenderState(D3DRS_FOGENABLE, postRecordingState.fogEnable);
    device->SetRenderState(D3DRS_SPECULARENABLE, postRecordingState.specularEnable);
    device->SetRenderState(D3DRS_LOCALVIEWER, postRecordingState.localViewer);
    device->SetRenderState(D3DRS_NORMALIZENORMALS, postRecordingState.normalizeNormals);
    device->SetRenderState(D3DRS_ZFUNC, postRecordingState.zFunc);
    device->SetRenderState(D3DRS_ALPHAFUNC, postRecordingState.alphaFunc);
    device->SetRenderState(D3DRS_ALPHAREF, postRecordingState.alphaRef);

    // Reset state to allow new recording sessions
    // Note: isRecording stays false until next startRecording() call
    isReplaying = false;
}

// Call this when HLSL rendering session is complete to trigger replay
void FixedFunctionShader::finalizeBatchAndReplay(int sceneCount) {
    // Handle dump request
    if (dumpRequested) {
        if (recordingEnabled) {
            // Recording ON: Batch dump
            LOG::logline("Frame dump: Dumping %d recorded calls (batch)", recordedCalls.size());
            StatusOverlay::setStatus("Frame dump: Batch complete");

            char logline[512];
            for (size_t i = 0; i < recordedCalls.size(); ++i) {
                const auto& call = recordedCalls[i];
                snprintf(logline, sizeof(logline), "Call %zu: texture=0x%p, vb=0x%p, ib=0x%p, hasShadows=%d, hasParamH=%d (batch)",
                         i, call.rs.texture, call.rs.vb, call.rs.ib,
                         call.sk.hasShadows, call.sk.hasParamH);
                LOG::logline(logline);

                if (call.rs.texture) {
                    D3DSURFACE_DESC desc;
                    if (SUCCEEDED(call.rs.texture->GetLevelDesc(0, &desc))) {
                        snprintf(logline, sizeof(logline), "  Texture: %dx%d, format=%d",
                                 desc.Width, desc.Height, desc.Format);
                        LOG::logline(logline);
                    }
                }
            }
        } else {
            // Recording OFF: Immediate dump was already done per-call
            LOG::logline("Frame dump: Immediate dump complete");
            StatusOverlay::setStatus("Frame dump: Immediate complete");
        }
        dumpRequested = false;
    }

    if (recordingEnabled) {
        // Only record/replay Scene 0 (world geometry)
        // Scene 1+ (hands, sunglare, UI) should render normally without recording
        if (sceneCount == 0) {
            // NOTE: bboxLookup is now populated during recording (in recordRenderCall)
            // so it's available for prepareOcclusionCullingForDepth which runs before this

            // Scene 0: record, replay, and reset for next cycle
            if (isRecording && !recordedCalls.empty()) {
                stopRecordingAndReplay();
            }

            // Ensure clean state for next scene - stop recording to exclude hands/UI
            isRecording = false;
            isReplaying = false;
            recordingCompletedThisFrame = true;  // Prevent restarting for Scene 1+
        } else {
            // Scene 1+: don't replay, just clear any stale recordings
            // This ensures hands/UI render normally without HLSL batching
            recordedCalls.clear();
        }
    } else {
        // When recording disabled, still clear calls after potential dump
        recordedCalls.clear();
    }

    // Note: Lights are cleared at start of new frame in startRecording(), not here
    // This allows lights to persist across all scenes in a frame (Scene 0, Scene 1 hands, etc.)

    // Always reset HLSL caches after rendering session completes
    resetHLSLCaches();
}

// Compare two LightStates for equality (to detect if we can reuse cached state)
bool FixedFunctionShader::compareLightStates(const LightState* a, const LightState* b) {
    // Fast path: check sizes first (cheapest comparison)
    if (a->lights.size() != b->lights.size() || a->active.size() != b->active.size()) {
        return false;
    }

    // Compare active array with memcmp (faster than loop for larger arrays)
    if (!a->active.empty() && memcmp(a->active.data(), b->active.data(), a->active.size() * sizeof(DWORD)) != 0) {
        return false;
    }

    // Compare globalAmbient as 4 DWORDs instead of 4 floats (avoids FP comparison)
    const DWORD* aAmb = reinterpret_cast<const DWORD*>(&a->globalAmbient);
    const DWORD* bAmb = reinterpret_cast<const DWORD*>(&b->globalAmbient);
    if (aAmb[0] != bAmb[0] || aAmb[1] != bAmb[1] || aAmb[2] != bAmb[2] || aAmb[3] != bAmb[3]) {
        return false;
    }

    return true;  // Same lighting state
}

void FixedFunctionShader::recordRenderCall(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, const ShaderKey& sk, int recordMWIdx) {
    {
        if (isReplaying) {
            return;  // Don't record during replay to avoid recursion
        }
    }

    // When recording is OFF and dump is requested, dump each call immediately
    if (!recordingEnabled && dumpRequested) {
        static int callIndex = 0;
        char logline[512];
        snprintf(logline, sizeof(logline), "Call %d: texture=0x%p, vb=0x%p, ib=0x%p (immediate)",
                 callIndex++, rs->texture, rs->vb, rs->ib);
        LOG::logline(logline);

        if (rs->texture) {
            D3DSURFACE_DESC desc;
            if (SUCCEEDED(rs->texture->GetLevelDesc(0, &desc))) {
                snprintf(logline, sizeof(logline), "  Texture: %dx%d, format=%d",
                         desc.Width, desc.Height, desc.Format);
                LOG::logline(logline);
            }
        }
    }

    // Extract lights for texture-based lighting system (all objects, even if culled)
    // Lights from culled objects can still illuminate visible geometry
    for (const auto& [id, light] : lightrs->lights) {
        // Check if light already exists using O(1) hash map lookup
        auto mapIt = DistantLand::sceneLightIndexMap.find(id);

        if (mapIt == DistantLand::sceneLightIndexMap.end()) {
            // New light - add to scene
            size_t newIndex = DistantLand::sceneLights.size();
            DistantLand::SceneLight sl;
            sl.id = id;
            sl.position = light.position;
            sl.diffuse = light.diffuse;
            sl.falloff = light.falloff;  // (constant, linear, quadratic)
            sl.radius = DistantLand::computeLightRadius(sl.falloff.x, sl.falloff.y, sl.falloff.z);
            sl.isVisible = false;  // Will be set during Hi-Z culling
            sl.lastSeenFrame = 0;  // Not used without persistence
            DistantLand::sceneLights.push_back(sl);
            DistantLand::sceneLightIndexMap[id] = newIndex;  // Add to index map
        } else {
            // Update dynamic properties (color may pulse)
            size_t index = mapIt->second;
            DistantLand::sceneLights[index].diffuse = light.diffuse;
        }
    }

    // Reuse last LightState if identical to avoid allocation overhead
    std::shared_ptr<LightState> sharedLightState;
    {
        // Ultra-fast path: pointer equality (O(1), no function call)
        if (lightrs == lastLightStatePtr) {
            sharedLightState = lastLightState;
        }
        // Fast path: value comparison only if pointer differs
        else if (lastLightState && compareLightStates(lastLightState.get(), lightrs)) {
            // Reuse existing shared_ptr (no allocation)
            sharedLightState = lastLightState;
        }
        // Slow path: create new copy
        else {
            sharedLightState = std::make_shared<LightState>(*lightrs);
            lastLightState = sharedLightState;
            lastLightStatePtr = lightrs;  // Cache raw pointer for next comparison
        }
    }

    {
        recordedCalls.emplace_back(rs, frs, sharedLightState, sk, recordMWIdx);

        // Immediately populate bboxLookup so depth pass can use current-frame bboxes
        // (prepareOcclusionCullingForDepth runs BEFORE finalizeBatchAndReplay)
        const auto& call = recordedCalls.back();
        if (call.hasBoundingBox) {
            VBIBKey key{call.rs.vb, call.rs.ib};
            bboxLookup[key] = {call.bboxMin, call.bboxMax};
        }
    }

    // Occluder selection and rasterization deferred to prepareRecordedCalls()
    // This removes the heaviest per-draw work from the recording hot path
}

void FixedFunctionShader::replayRecordedCalls(int sceneCount) {
    {
        if (recordedCalls.empty()) {
            return;
        }

        // Check if replay is disabled via ImGui
        if (!ImGuiManager::GetEnableReplay()) {
            return;
        }
    }

    isReplaying = true;

    // Get current matrices for culling
    D3DXMATRIX currentView, currentProj;
    D3DXMATRIX viewProj;
    D3DXMATRIX savedShadowViewproj[2];
    {
        ZoneScopedN("replay_GetTransforms");
        device->GetTransform(D3DTS_VIEW, &currentView);
        device->GetTransform(D3DTS_PROJECTION, &currentProj);
    }

    // Temporarily set shadow matrices to recording state
    savedShadowViewproj[0] = DistantLand::smViewproj[0];
    savedShadowViewproj[1] = DistantLand::smViewproj[1];
    DistantLand::smViewproj[0] = recordingShadowViewproj[0];
    DistantLand::smViewproj[1] = recordingShadowViewproj[1];

    // Calculate view-projection matrix for Hi-Z culling
    viewProj = currentView * currentProj;

    // Hi-Z pyramid generation moved to end of frame (Present) for better performance
    // We use previous frame's Hi-Z here for GPU culling (minimal 1-frame delay)
    // GPU-based culling: Hi-Z stays in VRAM, no CPU locking needed!

    // Hi-Z culling statistics
    int totalCalls = recordedCalls.size();
    int culledCalls = 0;
    int callsWithBBox = 0;
    int callsWithoutBBox = 0;
    int cacheHits = 0;    // Visibility reused from depth pass
    int cacheMisses = 0;  // New Hi-Z tests (alpha objects)

    // Check for debug key press (Y key)
    static bool debugHiZ = false;
    static int debugCallCount = 0;
    if (GetAsyncKeyState('Y') & 0x8000) {
        static bool wasPressed = false;
        if (!wasPressed) {
            debugHiZ = true;
            debugCallCount = 5; // Log next 5 culled calls
            LOG::logline(">> Hi-Z Debug: Enabled for next 5 CULLED calls");
            wasPressed = true;
        }
    } else {
        static bool wasPressed = false;
        wasPressed = false;
    }

    // Check for Hi-Z snapshot save key press (L key)
    if (GetAsyncKeyState('L') & 0x8000) {
        static bool wasPressed = false;
        if (!wasPressed) {
            LOG::logline(">> L key pressed: Saving Hi-Z snapshot...");
            DistantLand::saveHiZSnapshot();
            wasPressed = true;
        }
    } else {
        static bool wasPressed = false;
        wasPressed = false;
    }

    // Calculate camera velocity from previous frame (to compensate for one-frame-behind Hi-Z)
    D3DXVECTOR3 currentCameraPos = D3DXVECTOR3(DistantLand::eyePos.x, DistantLand::eyePos.y, DistantLand::eyePos.z);
    D3DXVECTOR3 cameraVelocity(0.0f, 0.0f, 0.0f);
    float cameraMovementMag = 0.0f;

    if (hasPrevCamera) {
        cameraVelocity = currentCameraPos - prevCameraPos;
        cameraMovementMag = D3DXVec3Length(&cameraVelocity);
    }

    // Store current camera for next frame
    prevCameraPos = currentCameraPos;
    prevCameraView = currentView;
    hasPrevCamera = true;

    // Light culling using Hi-Z occlusion (skip if disabled for profiling)
    if (ImGuiManager::GetEnableLightProcessing()) {
        DistantLand::cullSceneLights(viewProj);

        // Upload visible lights to GPU texture (may stall if GPU idle)
        DistantLand::uploadLightDataToTexture(currentView);
    }

    // Set light count parameter for shaders
    int numLights = (int)DistantLand::visibleLights.size();
    float lightParams[4] = {
        (float)numLights,                               // numLights
        numLights > 0 ? 1.0f / (numLights * 3) : 0.0f,  // texelSize
        0.0f, 0.0f
    };
    {
        ZoneScopedN("replay_SetLightParams");
        device->SetPixelShaderConstantF(50, lightParams, 1); // c50
    }

    // Inline Hi-Z culling using current matrices (same as bbox visualization)
    const size_t numCalls = recordedCalls.size();

    ZoneScopedN("replay_MainLoop");
    bool firstDrawDone = false;
    for (size_t i = 0; i < numCalls; i++) {
        auto& call = recordedCalls[i];  // Non-const to update shader key

        // Build MeshKey for this call
        MeshKey currentKey{call.rs.vb, call.rs.ib, call.rs.fvf, call.rs.baseIndex,
                          call.rs.vertCount, call.rs.startIndex, call.rs.primCount};

        // Visibility culling: reuse depth pass results when available
        bool shouldRender = true;
        if (call.hasBoundingBox) {
            callsWithBBox++;

            // Check if occluder (never cull occluders)
            bool isOccluder = (rasterizedOccluderMeshes.count(currentKey) > 0);

            if (!isOccluder) {
                // Use recordMWIndex to reuse visibility results from depth pass
                if (call.recordMWIndex >= 0 && call.recordMWIndex < (int)visibilityResults.size()) {
                    // Z-writing object tested in depth pass - reuse result
                    shouldRender = (visibilityResults[call.recordMWIndex] == 1);
                    cacheHits++;
                } else {
                    // Alpha object (not in recordMW) - test fresh
                    shouldRender = softwareOcclusionCuller.testBoundingBox(
                        call.bboxMin, call.bboxMax, currentView, currentProj);
                    cacheMisses++;
                }
                if (!shouldRender) {
                    culledCalls++;
                }
            }
        } else {
            callsWithoutBBox++;
        }

        if (shouldRender) {
            // Restore sampler states for this call (captured during recording)
            {
                ZoneScopedN("replay_SetSamplers");
                for (int stage = 0; stage < 8; ++stage) {
                    if (call.samplerStates[stage].captured) {
                        device->SetSamplerState(stage, D3DSAMP_ADDRESSU, call.samplerStates[stage].addressU);
                        device->SetSamplerState(stage, D3DSAMP_ADDRESSV, call.samplerStates[stage].addressV);
                    }
                }
            }

            // Occluder highlighting: tint occluders green for debug visualization
            if (ImGuiManager::GetHighlightOccluders()) {
                MeshKey highlightKey;
                highlightKey.vb = call.rs.vb;
                highlightKey.ib = call.rs.ib;
                highlightKey.fvf = call.rs.fvf;
                highlightKey.baseIndex = call.rs.baseIndex;  // Note: baseIndex, not ibBase!
                highlightKey.vertCount = call.rs.vertCount;
                highlightKey.startIndex = call.rs.startIndex;
                highlightKey.primCount = call.rs.primCount;

                bool isOccluder = (rasterizedOccluderMeshes.find(highlightKey) != rasterizedOccluderMeshes.end());
                if (isOccluder) {
                    // Create a modified FragmentState with green emissive tint
                    FragmentState tintedFrs = call.frs;
                    tintedFrs.material.emissive.r = 0.0f;
                    tintedFrs.material.emissive.g = 0.4f;
                    tintedFrs.material.emissive.b = 0.0f;
                    tintedFrs.material.emissive.a = 1.0f;
                    renderMorrowindHLSL_Internal(&call.rs, &tintedFrs, call.lightrs.get(), DIRTY_ALL);
                    continue;
                }
            }

            {
                ZoneScopedN("replay_RenderCall");
                // Restore Morrowind-recorded device state before each replay call
                // During recording, Morrowind sets these between calls; during replay we must do it
                // Debug mode: validate BEFORE pre-set to detect leaks from previous call
                if (i > 0 && ImGuiManager::GetStateLeakDetection() && call.expectedState.captured) {
                    validateDeviceState(call.expectedState, i);
                }
                if (call.expectedState.captured) {
                    device->SetRenderState(D3DRS_ALPHABLENDENABLE, call.expectedState.alphaBlendEnable);
                    device->SetRenderState(D3DRS_ALPHATESTENABLE, call.expectedState.alphaTestEnable);
                    device->SetRenderState(D3DRS_ZENABLE, call.expectedState.zEnable);
                    device->SetRenderState(D3DRS_ZWRITEENABLE, call.expectedState.zWriteEnable);
                    device->SetRenderState(D3DRS_CULLMODE, call.expectedState.cullMode);
                    device->SetRenderState(D3DRS_SRCBLEND, call.expectedState.srcBlend);
                    device->SetRenderState(D3DRS_DESTBLEND, call.expectedState.destBlend);
                    device->SetRenderState(D3DRS_FOGENABLE, call.expectedState.fogEnable);
                    // Invalidate cache since we bypassed it with raw SetRenderState
                    materialCache.reset();
                }
                renderMorrowindHLSL_Internal(&call.rs, &call.frs, call.lightrs.get(), call.dirtyFlags);
                if (!firstDrawDone) {
                    ZoneScopedN("replay_FirstDrawDone");
                    firstDrawDone = true;
                }
            }
        }
    }

    // Log culling statistics
    int renderedCalls = totalCalls - culledCalls;
    LOG::logline("Hi-Z Stats: %d total, %d bbox, %d culled (%.1f%%), cache: %d hits %d misses",
                 totalCalls, callsWithBBox, culledCalls,
                 totalCalls > 0 ? (culledCalls * 100.0f) / totalCalls : 0.0f,
                 cacheHits, cacheMisses);

    // Update ImGui debug stats
    ImGuiManager::UpdateDebugStats(
        totalCalls, renderedCalls, culledCalls,
        (int)DistantLand::sceneLights.size(), numLights,
        (int)DistantLand::recordMW.size(), 0  // Now shows filtered count
    );

    // Debug visualization: Render bounding boxes with color-coded status from ImGui
    int debugBBoxMode = ImGuiManager::GetBBoxVisualizationMode();

    if (debugBBoxMode > 0) {
        // Save render states
        IDirect3DStateBlock9* savedState;
        device->CreateStateBlock(D3DSBT_ALL, &savedState);

        // Setup for line rendering - disable depth test so boxes always render
        device->SetRenderState(D3DRS_ZENABLE, FALSE);
        device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
        device->SetRenderState(D3DRS_LIGHTING, FALSE);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
        device->SetRenderState(D3DRS_FOGENABLE, FALSE); // Disable fog
        device->SetRenderState(D3DRS_AMBIENT, 0xFFFFFFFF); // Full ambient
        device->SetRenderState(D3DRS_COLORVERTEX, TRUE);
        device->SetRenderState(D3DRS_DIFFUSEMATERIALSOURCE, D3DMCS_COLOR1);
        device->SetRenderState(D3DRS_EMISSIVEMATERIALSOURCE, D3DMCS_COLOR1);
        device->SetRenderState(D3DRS_TEXTUREFACTOR, 0xFFFFFFFF); // Full white texture factor
        device->SetFVF(D3DFVF_XYZ | D3DFVF_DIFFUSE);

        // Set up transforms for world-space rendering
        D3DXMATRIX identity;
        D3DXMatrixIdentity(&identity);
        device->SetTransform(D3DTS_WORLD, &identity);
        device->SetTransform(D3DTS_VIEW, &currentView);
        device->SetTransform(D3DTS_PROJECTION, &currentProj);

        // Helper to draw bbox edges
        auto drawBBox = [&](const D3DXVECTOR3& bmin, const D3DXVECTOR3& bmax, D3DCOLOR color) {
            struct Vertex { float x, y, z; D3DCOLOR color; };
            Vertex vertices[24] = {
                // Bottom face
                {bmin.x, bmin.y, bmin.z, color}, {bmax.x, bmin.y, bmin.z, color},
                {bmax.x, bmin.y, bmin.z, color}, {bmax.x, bmax.y, bmin.z, color},
                {bmax.x, bmax.y, bmin.z, color}, {bmin.x, bmax.y, bmin.z, color},
                {bmin.x, bmax.y, bmin.z, color}, {bmin.x, bmin.y, bmin.z, color},
                // Top face
                {bmin.x, bmin.y, bmax.z, color}, {bmax.x, bmin.y, bmax.z, color},
                {bmax.x, bmin.y, bmax.z, color}, {bmax.x, bmax.y, bmax.z, color},
                {bmax.x, bmax.y, bmax.z, color}, {bmin.x, bmax.y, bmax.z, color},
                {bmin.x, bmax.y, bmax.z, color}, {bmin.x, bmin.y, bmax.z, color},
                // Vertical edges
                {bmin.x, bmin.y, bmin.z, color}, {bmin.x, bmin.y, bmax.z, color},
                {bmax.x, bmin.y, bmin.z, color}, {bmax.x, bmin.y, bmax.z, color},
                {bmax.x, bmax.y, bmin.z, color}, {bmax.x, bmax.y, bmax.z, color},
                {bmin.x, bmax.y, bmin.z, color}, {bmin.x, bmax.y, bmax.z, color},
            };
            device->DrawPrimitiveUP(D3DPT_LINELIST, 12, vertices, sizeof(Vertex));
        };

        // Mode 1: Show culled objects only (red boxes) - test directly
        if (debugBBoxMode == 1) {
            for (size_t i = 0; i < recordedCalls.size(); i++) {
                const auto& call = recordedCalls[i];
                if (!call.hasBoundingBox) continue;

                // Direct occlusion test
                bool isVisible = softwareOcclusionCuller.testBoundingBox(
                    call.bboxMin,
                    call.bboxMax,
                    currentView,
                    currentProj
                );

                // Draw only culled objects in red
                if (!isVisible) {
                    drawBBox(call.bboxMin, call.bboxMax, D3DCOLOR_ARGB(255, 255, 0, 0));
                }
            }
        }

        // Mode 2: Show culled lights only (red boxes)
        if (debugBBoxMode == 2) {
            for (const auto& light : DistantLand::sceneLights) {
                if (!light.isVisible) {  // Only show culled lights
                    D3DXVECTOR3 bmin = light.position - D3DXVECTOR3(light.radius, light.radius, light.radius);
                    D3DXVECTOR3 bmax = light.position + D3DXVECTOR3(light.radius, light.radius, light.radius);

                    // Red color for culled lights
                    drawBBox(bmin, bmax, D3DCOLOR_ARGB(255, 255, 0, 0));
                }
            }
        }

        // Restore render states
        savedState->Apply();
        savedState->Release();
    }

    // Single Object Mode: Show selected object's bbox in CYAN (always visible, regardless of bbox mode)
    if (ImGuiManager::hiZSingleObjectMode) {
        // Save render states
        IDirect3DStateBlock9* savedState;
        device->CreateStateBlock(D3DSBT_ALL, &savedState);

        // Setup for line rendering - disable depth test so box always renders
        device->SetRenderState(D3DRS_ZENABLE, FALSE);
        device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
        device->SetRenderState(D3DRS_LIGHTING, FALSE);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
        device->SetRenderState(D3DRS_FOGENABLE, FALSE);
        device->SetRenderState(D3DRS_AMBIENT, 0xFFFFFFFF);
        device->SetRenderState(D3DRS_COLORVERTEX, TRUE);
        device->SetRenderState(D3DRS_DIFFUSEMATERIALSOURCE, D3DMCS_COLOR1);
        device->SetRenderState(D3DRS_EMISSIVEMATERIALSOURCE, D3DMCS_COLOR1);
        device->SetFVF(D3DFVF_XYZ | D3DFVF_DIFFUSE);

        // Set up transforms for world-space rendering
        D3DXMATRIX identity;
        D3DXMatrixIdentity(&identity);
        device->SetTransform(D3DTS_WORLD, &identity);
        device->SetTransform(D3DTS_VIEW, &currentView);
        device->SetTransform(D3DTS_PROJECTION, &currentProj);

        // Helper to draw bbox edges
        auto drawBBox = [&](const D3DXVECTOR3& bmin, const D3DXVECTOR3& bmax, D3DCOLOR color) {
            struct Vertex { float x, y, z; D3DCOLOR color; };
            Vertex vertices[24] = {
                // Bottom face
                {bmin.x, bmin.y, bmin.z, color}, {bmax.x, bmin.y, bmin.z, color},
                {bmax.x, bmin.y, bmin.z, color}, {bmax.x, bmax.y, bmin.z, color},
                {bmax.x, bmax.y, bmin.z, color}, {bmin.x, bmax.y, bmin.z, color},
                {bmin.x, bmax.y, bmin.z, color}, {bmin.x, bmin.y, bmin.z, color},
                // Top face
                {bmin.x, bmin.y, bmax.z, color}, {bmax.x, bmin.y, bmax.z, color},
                {bmax.x, bmin.y, bmax.z, color}, {bmax.x, bmax.y, bmax.z, color},
                {bmax.x, bmax.y, bmax.z, color}, {bmin.x, bmax.y, bmax.z, color},
                {bmin.x, bmax.y, bmax.z, color}, {bmin.x, bmin.y, bmax.z, color},
                // Vertical edges
                {bmin.x, bmin.y, bmin.z, color}, {bmin.x, bmin.y, bmax.z, color},
                {bmax.x, bmin.y, bmin.z, color}, {bmax.x, bmin.y, bmax.z, color},
                {bmax.x, bmax.y, bmin.z, color}, {bmax.x, bmax.y, bmax.z, color},
                {bmin.x, bmax.y, bmin.z, color}, {bmin.x, bmax.y, bmax.z, color},
            };
            device->DrawPrimitiveUP(D3DPT_LINELIST, 12, vertices, sizeof(Vertex));
        };

        // Find and draw the selected object's bbox
        int currentObjectIndex = 0;
        for (size_t i = 0; i < recordedCalls.size(); i++) {
            const auto& call = recordedCalls[i];

            // Count only opaque objects with >2 tris (matching selection logic)
            bool isOpaque = !call.rs.blendEnable && call.rs.zWrite;
            if (isOpaque && call.rs.primCount > 2) {
                if (currentObjectIndex == ImGuiManager::hiZSingleObjectIndex) {
                    if (call.hasBoundingBox) {
                        // Draw in CYAN
                        drawBBox(call.bboxMin, call.bboxMax, D3DCOLOR_ARGB(255, 0, 255, 255));
                    }
                    break;
                }
                currentObjectIndex++;
            }
        }

        // Restore render states
        savedState->Apply();
        savedState->Release();
    }

    // Restore current shadow matrices
    DistantLand::smViewproj[0] = savedShadowViewproj[0];
    DistantLand::smViewproj[1] = savedShadowViewproj[1];

    // Note: Camera position is already stored above for next frame's velocity calculation

    // Save current frame's calls for next frame's raycast targeting
    // Swap is efficient - avoids copying, just exchanges internal pointers
    std::swap(previousFrameCalls, recordedCalls);

    isReplaying = false;
}

// ------------------------------------
// FixedFunctionShader::RecordedRenderedState

FixedFunctionShader::RecordedRenderedState::RecordedRenderedState(const RenderedState& state)
    : RenderedState(state) {
    vb->AddRef();
    ib->AddRef();
    if (texture) {
        texture->AddRef();
    }
}

FixedFunctionShader::RecordedRenderedState::~RecordedRenderedState() {
    if (vb) {
        vb->Release();
    }
    if (ib) {
        ib->Release();
    }
    if (texture) {
        texture->Release();
    }
}

FixedFunctionShader::RecordedRenderedState::RecordedRenderedState(RecordedRenderedState&& source) noexcept
    : RenderedState(source) {
    source.vb = nullptr;
    source.ib = nullptr;
    source.texture = nullptr;
}

// ------------------------------------
// State Leak Detection

void FixedFunctionShader::validateDeviceState(const ExpectedDeviceState& expected, int callIndex) {
    if (!expected.captured) return;

    auto checkState = [&](D3DRENDERSTATETYPE state, const char* name, DWORD expectedVal) {
        DWORD actual;
        device->GetRenderState(state, &actual);
        if (actual != expectedVal) {
            LOG::logline("[LEAK] %s = %d but expected %d (call #%d)", name, actual, expectedVal, callIndex);
        }
    };

    // Skip render states covered by the pre-set (ALPHABLENDENABLE, ALPHATESTENABLE,
    // ZENABLE, ZWRITEENABLE, FOGENABLE, CULLMODE, SRCBLEND, DESTBLEND).
    // These are always restored before the next call, so leaks are benign.
    // Only check states that would escape the replay loop unhandled.

    // Check sampler states for stages 0-1
    for (int s = 0; s < 2; ++s) {
        DWORD addrU, addrV;
        device->GetSamplerState(s, D3DSAMP_ADDRESSU, &addrU);
        device->GetSamplerState(s, D3DSAMP_ADDRESSV, &addrV);
        if (addrU != expected.samplerAddressU[s]) {
            LOG::logline("[LEAK] Sampler%d ADDRESSU = %d but expected %d (call #%d)", s, addrU, expected.samplerAddressU[s], callIndex);
        }
        if (addrV != expected.samplerAddressV[s]) {
            LOG::logline("[LEAK] Sampler%d ADDRESSV = %d but expected %d (call #%d)", s, addrV, expected.samplerAddressV[s], callIndex);
        }
    }

    // Check for texture leaks on unused stages (stages 2-7 should be clean)
    for (int s = 2; s < 8; ++s) {
        IDirect3DBaseTexture9* tex = nullptr;
        device->GetTexture(s, &tex);
        if (tex) {
            tex->Release();
            if (!expected.textures[s]) {
                LOG::logline("[LEAK] Texture bound on stage %d but expected NULL (call #%d)", s, callIndex);
            }
        }
    }
}

// ------------------------------------
// FixedFunctionShader::HLSLRecordedCall

FixedFunctionShader::HLSLRecordedCall::HLSLRecordedCall(const RenderedState* rs_, const FragmentState* frs_, std::shared_ptr<LightState> lightrs_, const ShaderKey& sk_, int recordMWIdx)
    : rs(*rs_), frs(*frs_), lightrs(lightrs_), sk(sk_), hasBoundingBox(false), recordMWIndex(recordMWIdx), prepared(false), dirtyFlags(DIRTY_ALL) {
    // Lean recording: capture sampler states for stages 0-1 only (Morrowind-bound textures)
    // Stages 2+ are HLSL-specific textures bound by MGE XE with known sampler states
    for (int stage = 0; stage < 2; ++stage) {
        IDirect3DBaseTexture9* texture = nullptr;
        if (SUCCEEDED(device->GetTexture(stage, &texture)) && texture) {
            auto cacheIt = samplerCache.find(texture);
            if (cacheIt != samplerCache.end()) {
                samplerStates[stage].addressU = cacheIt->second.first;
                samplerStates[stage].addressV = cacheIt->second.second;
                samplerStates[stage].captured = true;
            } else {
                if (SUCCEEDED(device->GetSamplerState(stage, D3DSAMP_ADDRESSU, &samplerStates[stage].addressU)) &&
                    SUCCEEDED(device->GetSamplerState(stage, D3DSAMP_ADDRESSV, &samplerStates[stage].addressV))) {
                    samplerStates[stage].captured = true;
                    samplerCache[texture] = std::make_pair(samplerStates[stage].addressU, samplerStates[stage].addressV);
                }
            }
            texture->Release();
        } else {
            samplerStates[stage].captured = false;
        }
    }
    // Stages 2-7: mark as not captured (will use defaults during replay)
    for (int stage = 2; stage < 8; ++stage) {
        samplerStates[stage].captured = false;
    }

    // Defer bbox computation to prepare phase — only use cache hits during recording
    // This avoids VB/IB locks on cache miss (the expensive path)
    if (rs_->vb && (rs_->fvf & D3DFVF_POSITION_MASK) != 0) {
        MeshKey key;
        key.vb = rs_->vb;
        key.ib = rs_->ib;
        key.fvf = rs_->fvf;
        key.baseIndex = rs_->baseIndex;
        key.vertCount = rs_->vertCount;
        key.startIndex = rs_->startIndex;
        key.primCount = rs_->primCount;

        auto it = bboxCache.find(key);
        if (it != bboxCache.end()) {
            // Cache hit — cheap world-space transform
            const ObjectSpaceBBox& objBBox = it->second;
            D3DXVECTOR3 corners[8] = {
                D3DXVECTOR3(objBBox.bboxMin.x, objBBox.bboxMin.y, objBBox.bboxMin.z),
                D3DXVECTOR3(objBBox.bboxMax.x, objBBox.bboxMin.y, objBBox.bboxMin.z),
                D3DXVECTOR3(objBBox.bboxMin.x, objBBox.bboxMax.y, objBBox.bboxMin.z),
                D3DXVECTOR3(objBBox.bboxMax.x, objBBox.bboxMax.y, objBBox.bboxMin.z),
                D3DXVECTOR3(objBBox.bboxMin.x, objBBox.bboxMin.y, objBBox.bboxMax.z),
                D3DXVECTOR3(objBBox.bboxMax.x, objBBox.bboxMin.y, objBBox.bboxMax.z),
                D3DXVECTOR3(objBBox.bboxMin.x, objBBox.bboxMax.y, objBBox.bboxMax.z),
                D3DXVECTOR3(objBBox.bboxMax.x, objBBox.bboxMax.y, objBBox.bboxMax.z),
            };
            bboxMin = D3DXVECTOR3(1e10f, 1e10f, 1e10f);
            bboxMax = D3DXVECTOR3(-1e10f, -1e10f, -1e10f);
            for (int i = 0; i < 8; i++) {
                D3DXVECTOR3 worldCorner;
                D3DXVec3TransformCoord(&worldCorner, &corners[i], &rs_->worldTransforms[0]);
                bboxMin.x = std::min(bboxMin.x, worldCorner.x);
                bboxMin.y = std::min(bboxMin.y, worldCorner.y);
                bboxMin.z = std::min(bboxMin.z, worldCorner.z);
                bboxMax.x = std::max(bboxMax.x, worldCorner.x);
                bboxMax.y = std::max(bboxMax.y, worldCorner.y);
                bboxMax.z = std::max(bboxMax.z, worldCorner.z);
            }
            hasBoundingBox = true;
        }
        // Cache miss: hasBoundingBox stays false, will be computed in prepareRecordedCalls()
    }

    // Always capture device state for replay pre-set (prevents state leaks between calls)
    device->GetRenderState(D3DRS_ALPHABLENDENABLE, &expectedState.alphaBlendEnable);
    device->GetRenderState(D3DRS_ALPHATESTENABLE, &expectedState.alphaTestEnable);
    device->GetRenderState(D3DRS_ZENABLE, &expectedState.zEnable);
    device->GetRenderState(D3DRS_ZWRITEENABLE, &expectedState.zWriteEnable);
    device->GetRenderState(D3DRS_FOGENABLE, &expectedState.fogEnable);
    device->GetRenderState(D3DRS_CULLMODE, &expectedState.cullMode);
    device->GetRenderState(D3DRS_SRCBLEND, &expectedState.srcBlend);
    device->GetRenderState(D3DRS_DESTBLEND, &expectedState.destBlend);
    expectedState.captured = true;

    // Debug-only: capture additional state for validation
    if (ImGuiManager::GetStateLeakDetection()) {
        DWORD dbias, sbias;
        device->GetRenderState(D3DRS_DEPTHBIAS, &dbias);
        device->GetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, &sbias);
        expectedState.depthBias = *(float*)&dbias;
        expectedState.slopeScaledDepthBias = *(float*)&sbias;
        for (int s = 0; s < 2; ++s) {
            device->GetSamplerState(s, D3DSAMP_ADDRESSU, &expectedState.samplerAddressU[s]);
            device->GetSamplerState(s, D3DSAMP_ADDRESSV, &expectedState.samplerAddressV[s]);
        }
        for (int s = 0; s < 8; ++s) {
            expectedState.textures[s] = nullptr;
            device->GetTexture(s, &expectedState.textures[s]);
            if (expectedState.textures[s]) expectedState.textures[s]->Release();
        }
    }
}

bool FixedFunctionShader::computeBoundingBox(const RenderedState* rs, D3DXVECTOR3& bboxMin, D3DXVECTOR3& bboxMax) {
    static bool debugBBox = false;
    static int debugCount = 0;
    static bool keyCheckedThisRecording = false;

    // Check for U key only once per recording session (avoid GetAsyncKeyState in per-mesh hotpath)
    if (isRecording && !keyCheckedThisRecording) {
        if (GetAsyncKeyState('U') & 0x8000) {
            static bool wasPressed = false;
            if (!wasPressed) {
                debugBBox = true;
                debugCount = 10;
                LOG::logline(">> BBox Debug: Enabled for next 10 calls");
                wasPressed = true;
            }
        } else {
            static bool wasPressed = false;
            wasPressed = false;
        }
        keyCheckedThisRecording = true;
    }

    // Reset flag when not recording
    if (!isRecording) {
        keyCheckedThisRecording = false;
    }

    if (!rs->vb) {
        if (debugBBox && debugCount > 0) {
            LOG::logline("!! computeBBox: no VB");
            debugCount--;
        }
        return false;
    }

    // Determine position offset in vertex structure (FVF formats always have position first if XYZ is present)
    bool hasPosition = (rs->fvf & D3DFVF_POSITION_MASK) != 0;
    if (!hasPosition) {
        if (debugBBox && debugCount > 0) {
            LOG::logline("!! computeBBox: no position in FVF (fvf=0x%X, mask=0x%X)", rs->fvf, D3DFVF_POSITION_MASK);
            debugCount--;
        }
        return false;
    }

    // Create mesh key for cache lookup
    MeshKey key;
    key.vb = rs->vb;
    key.ib = rs->ib;
    key.fvf = rs->fvf;
    key.baseIndex = rs->baseIndex;
    key.vertCount = rs->vertCount;
    key.startIndex = rs->startIndex;
    key.primCount = rs->primCount;

    // Check cache first
    auto it = bboxCache.find(key);
    if (it != bboxCache.end()) {
        // Cache hit! Transform cached object-space bbox to world-space
        const ObjectSpaceBBox& objBBox = it->second;

        // Transform 8 corners of object-space bbox by world matrix
        D3DXVECTOR3 corners[8] = {
            D3DXVECTOR3(objBBox.bboxMin.x, objBBox.bboxMin.y, objBBox.bboxMin.z),
            D3DXVECTOR3(objBBox.bboxMax.x, objBBox.bboxMin.y, objBBox.bboxMin.z),
            D3DXVECTOR3(objBBox.bboxMin.x, objBBox.bboxMax.y, objBBox.bboxMin.z),
            D3DXVECTOR3(objBBox.bboxMax.x, objBBox.bboxMax.y, objBBox.bboxMin.z),
            D3DXVECTOR3(objBBox.bboxMin.x, objBBox.bboxMin.y, objBBox.bboxMax.z),
            D3DXVECTOR3(objBBox.bboxMax.x, objBBox.bboxMin.y, objBBox.bboxMax.z),
            D3DXVECTOR3(objBBox.bboxMin.x, objBBox.bboxMax.y, objBBox.bboxMax.z),
            D3DXVECTOR3(objBBox.bboxMax.x, objBBox.bboxMax.y, objBBox.bboxMax.z),
        };

        bboxMin = D3DXVECTOR3(1e10f, 1e10f, 1e10f);
        bboxMax = D3DXVECTOR3(-1e10f, -1e10f, -1e10f);

        for (int i = 0; i < 8; i++) {
            D3DXVECTOR3 worldCorner;
            D3DXVec3TransformCoord(&worldCorner, &corners[i], &rs->worldTransforms[0]);

            bboxMin.x = std::min(bboxMin.x, worldCorner.x);
            bboxMin.y = std::min(bboxMin.y, worldCorner.y);
            bboxMin.z = std::min(bboxMin.z, worldCorner.z);
            bboxMax.x = std::max(bboxMax.x, worldCorner.x);
            bboxMax.y = std::max(bboxMax.y, worldCorner.y);
            bboxMax.z = std::max(bboxMax.z, worldCorner.z);
        }

        if (debugBBox && debugCount > 0) {
            LOG::logline(">> computeBBox: CACHE HIT! world bbox=[%.2f,%.2f,%.2f] to [%.2f,%.2f,%.2f]",
                bboxMin.x, bboxMin.y, bboxMin.z, bboxMax.x, bboxMax.y, bboxMax.z);
            debugCount--;
        }

        return true;
    }

    // Cache miss - compute object-space bbox and cache it
    if (debugBBox && debugCount > 0) {
        LOG::logline(">> computeBBox: CACHE MISS - computing from vertices, VB=%p, fvf=0x%X, stride=%d", rs->vb, rs->fvf, rs->vbStride);
    }

    D3DXVECTOR3 objBBoxMin(1e10f, 1e10f, 1e10f);
    D3DXVECTOR3 objBBoxMax(-1e10f, -1e10f, -1e10f);

    // Lock vertex buffer to read position data (non-blocking to avoid GPU stalls)
    void* pVertices = nullptr;
    HRESULT hr = rs->vb->Lock(rs->vbOffset, 0, &pVertices, D3DLOCK_READONLY | D3DLOCK_DONOTWAIT);
    if (FAILED(hr)) {
        if (debugBBox && debugCount > 0) {
            LOG::logline("!! computeBBox: VB Lock failed hr=0x%X", hr);
            debugCount--;
        }
        return false;
    }

    // Get vertex stride from FVF
    UINT stride = rs->vbStride;
    if (stride == 0) {
        if (debugBBox && debugCount > 0) {
            LOG::logline("!! computeBBox: stride == 0");
            debugCount--;
        }
        rs->vb->Unlock();
        return false;
    }

    // For indexed primitives, we need to check all referenced vertices
    // Lock index buffer to determine which vertices to check (non-blocking)
    void* pIndices = nullptr;
    if (rs->ib) {
        hr = rs->ib->Lock(0, 0, &pIndices, D3DLOCK_READONLY | D3DLOCK_DONOTWAIT);
        if (FAILED(hr)) {
            if (debugBBox && debugCount > 0) {
                LOG::logline("!! computeBBox: IB lock failed, hr=0x%X", hr);
                debugCount--;
            }
            rs->vb->Unlock();
            return false;
        }

        // Determine index format (16-bit or 32-bit)
        D3DINDEXBUFFER_DESC ibDesc;
        rs->ib->GetDesc(&ibDesc);
        bool is16Bit = (ibDesc.Format == D3DFMT_INDEX16);

        // Process indexed vertices
        UINT indexCount = 0;
        switch (rs->primType) {
            case D3DPT_TRIANGLELIST: indexCount = rs->primCount * 3; break;
            case D3DPT_TRIANGLESTRIP: indexCount = rs->primCount + 2; break;
            case D3DPT_TRIANGLEFAN: indexCount = rs->primCount + 2; break;
            default:
                if (debugBBox && debugCount > 0) {
                    LOG::logline("!! computeBBox: unsupported primType=%d", rs->primType);
                    debugCount--;
                }
                rs->ib->Unlock();
                rs->vb->Unlock();
                return false;
        }

        for (UINT i = 0; i < indexCount; i++) {
            UINT vertexIndex;
            if (is16Bit) {
                vertexIndex = ((WORD*)pIndices)[rs->startIndex + i] + rs->baseIndex;
            } else {
                vertexIndex = ((DWORD*)pIndices)[rs->startIndex + i] + rs->baseIndex;
            }

            // Get vertex position (positions are always at offset 0 in FVF) - object space
            BYTE* vertexData = ((BYTE*)pVertices) + vertexIndex * stride;
            D3DXVECTOR3* pos = (D3DXVECTOR3*)vertexData;

            objBBoxMin.x = std::min(objBBoxMin.x, pos->x);
            objBBoxMin.y = std::min(objBBoxMin.y, pos->y);
            objBBoxMin.z = std::min(objBBoxMin.z, pos->z);
            objBBoxMax.x = std::max(objBBoxMax.x, pos->x);
            objBBoxMax.y = std::max(objBBoxMax.y, pos->y);
            objBBoxMax.z = std::max(objBBoxMax.z, pos->z);
        }

        rs->ib->Unlock();
    } else {
        // Non-indexed primitives - check sequential vertices
        UINT vertexCount = rs->vertCount;
        for (UINT i = 0; i < vertexCount; i++) {
            BYTE* vertexData = ((BYTE*)pVertices) + (rs->baseIndex + i) * stride;
            D3DXVECTOR3* pos = (D3DXVECTOR3*)vertexData;

            objBBoxMin.x = std::min(objBBoxMin.x, pos->x);
            objBBoxMin.y = std::min(objBBoxMin.y, pos->y);
            objBBoxMin.z = std::min(objBBoxMin.z, pos->z);
            objBBoxMax.x = std::max(objBBoxMax.x, pos->x);
            objBBoxMax.y = std::max(objBBoxMax.y, pos->y);
            objBBoxMax.z = std::max(objBBoxMax.z, pos->z);
        }
    }

    rs->vb->Unlock();

    // Validate object-space bounding box
    bool valid = (objBBoxMin.x <= objBBoxMax.x && objBBoxMin.y <= objBBoxMax.y && objBBoxMin.z <= objBBoxMax.z);
    if (!valid) {
        if (debugBBox && debugCount > 0) {
            LOG::logline("!! computeBBox: INVALID object bbox! min=[%.2f,%.2f,%.2f] max=[%.2f,%.2f,%.2f]",
                objBBoxMin.x, objBBoxMin.y, objBBoxMin.z, objBBoxMax.x, objBBoxMax.y, objBBoxMax.z);
            debugCount--;
        }
        return false;
    }

    // Cache the object-space bbox
    ObjectSpaceBBox cachedBBox;
    cachedBBox.bboxMin = objBBoxMin;
    cachedBBox.bboxMax = objBBoxMax;
    bboxCache[key] = cachedBBox;

    // Transform 8 corners to world space
    D3DXVECTOR3 corners[8] = {
        D3DXVECTOR3(objBBoxMin.x, objBBoxMin.y, objBBoxMin.z),
        D3DXVECTOR3(objBBoxMax.x, objBBoxMin.y, objBBoxMin.z),
        D3DXVECTOR3(objBBoxMin.x, objBBoxMax.y, objBBoxMin.z),
        D3DXVECTOR3(objBBoxMax.x, objBBoxMax.y, objBBoxMin.z),
        D3DXVECTOR3(objBBoxMin.x, objBBoxMin.y, objBBoxMax.z),
        D3DXVECTOR3(objBBoxMax.x, objBBoxMin.y, objBBoxMax.z),
        D3DXVECTOR3(objBBoxMin.x, objBBoxMax.y, objBBoxMax.z),
        D3DXVECTOR3(objBBoxMax.x, objBBoxMax.y, objBBoxMax.z),
    };

    bboxMin = D3DXVECTOR3(1e10f, 1e10f, 1e10f);
    bboxMax = D3DXVECTOR3(-1e10f, -1e10f, -1e10f);

    for (int i = 0; i < 8; i++) {
        D3DXVECTOR3 worldCorner;
        D3DXVec3TransformCoord(&worldCorner, &corners[i], &rs->worldTransforms[0]);

        bboxMin.x = std::min(bboxMin.x, worldCorner.x);
        bboxMin.y = std::min(bboxMin.y, worldCorner.y);
        bboxMin.z = std::min(bboxMin.z, worldCorner.z);
        bboxMax.x = std::max(bboxMax.x, worldCorner.x);
        bboxMax.y = std::max(bboxMax.y, worldCorner.y);
        bboxMax.z = std::max(bboxMax.z, worldCorner.z);
    }

    if (debugBBox && debugCount > 0) {
        LOG::logline(">> computeBBox: CACHED! obj=[%.2f,%.2f,%.2f] to [%.2f,%.2f,%.2f], world=[%.2f,%.2f,%.2f] to [%.2f,%.2f,%.2f]",
            objBBoxMin.x, objBBoxMin.y, objBBoxMin.z, objBBoxMax.x, objBBoxMax.y, objBBoxMax.z,
            bboxMin.x, bboxMin.y, bboxMin.z, bboxMax.x, bboxMax.y, bboxMax.z);
        debugCount--;
    }

    return true;
}

