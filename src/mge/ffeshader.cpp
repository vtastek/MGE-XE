
#include "ffeshader.h"
#include "cullthread.h"
#include "mge_tracy.h"
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
#include <climits>
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

// Triple-buffered FrameBuffer infrastructure
FixedFunctionShader::FrameBuffer FixedFunctionShader::frameBuffers[3];
int FixedFunctionShader::recordingBuffer = 0;

// HLSL Render Dispatch Recording System
bool FixedFunctionShader::isRecording = false;
bool FixedFunctionShader::isReplaying = false;
bool FixedFunctionShader::manualRecordingControl = false;
bool FixedFunctionShader::recordingEnabled = true;
bool FixedFunctionShader::recordingCompletedThisFrame = false;
bool FixedFunctionShader::hiZBuiltThisFrame = false;
bool FixedFunctionShader::dumpRequested = false;

// Slow frame detection: prepareMs stored by prepareRecordedCalls, read by replayRecordedCalls
static float lastPrepareMs = 0.0f;

// Flag: set to false by executeCullPass (cull thread) to prevent device calls in suffix fallback
// Main thread leaves this true (default). No overlap: main thread doesn't call computeShaderKeyWithSuffixes
// during cull window (recording is stopped, replay hasn't started).
bool deviceCallsSafeInPrepare = true;

// Diagnostic: cache hit/miss logging for first N frames (temporary)
static int hlslDiagFrameCounter = 0;
std::unordered_set<FixedFunctionShader::ShaderKey, FixedFunctionShader::ShaderKey::hasher> FixedFunctionShader::diagHitKeys;

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
FixedFunctionShader::SavedRenderStates FixedFunctionShader::postRecordingState = {};

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

// Per-object light packing for mode 3
std::vector<FixedFunctionShader::PerObjectLightInfo> FixedFunctionShader::perObjectLightInfo;
float FixedFunctionShader::perObjectTexelSize = 0.0f;
IDirect3DTexture9* FixedFunctionShader::texPerObjectLightData = nullptr;

std::unordered_map<std::string, FixedFunctionShader::CachedShaderSource> FixedFunctionShader::shaderSourceCache;
SRWLOCK FixedFunctionShader::hlslCacheLock = SRWLOCK_INIT;
HANDLE FixedFunctionShader::precacheThread = nullptr;

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
static SRWLOCK textureSuffixLock = SRWLOCK_INIT;

// Texture release callback (evicts from suffix cache on texture free)
void (*g_onTextureReleased)(IDirect3DTexture9* realTexture) = nullptr;

static void onTextureReleased(IDirect3DTexture9* tex) {
    AcquireSRWLockExclusive(&textureSuffixLock);
    textureSuffixResolutionCache.erase(tex);
    ReleaseSRWLockExclusive(&textureSuffixLock);
}

// Pre-populate suffix cache during recording (main thread) so that
// computeShaderKeyWithSuffixes() never needs device calls when called
// from the cull thread. Device calls (CreateTexture, GetRenderTargetData)
// are only safe on the main thread.
static void warmSuffixCache(IDirect3DDevice* dev, IDirect3DTexture9* texture) {
    if (!texture) return;

    AcquireSRWLockShared(&textureSuffixLock);
    bool found = textureSuffixResolutionCache.count(texture) > 0;
    ReleaseSRWLockShared(&textureSuffixLock);

    if (found) return;  // Already cached

    // Expensive path: device calls for hash computation (main thread only)
    TextureSuffixResolutionCache entry;
    entry.hash = BSA::calculateTextureHash((IDirect3DDevice9*)dev, texture, false);

    const std::string* textureName = BSA::resolveTextureNameFromHash(entry.hash);
    if (textureName && entry.hash.crc32 != 0) {
        entry.textureName = *textureName;
        entry.hasValidName = true;
        entry.variants = BSA::getTextureSuffixVariants(textureName->c_str());

        // Pre-load suffix textures so bindShaderTextures never stalls on disk I/O
        if (entry.variants) {
            if (entry.variants->hasDiffParamT()) {
                BSA::loadSuffixTexture((IDirect3DDevice9*)dev, *entry.variants, "diffparam_t");
            } else if (entry.variants->hasDiffParam()) {
                BSA::loadSuffixTexture((IDirect3DDevice9*)dev, *entry.variants, "diffparam");
            }
            if (entry.variants->hasParamH()) {
                BSA::loadSuffixTexture((IDirect3DDevice9*)dev, *entry.variants, "paramh");
            }
            if (entry.variants->hasParamX()) {
                BSA::loadSuffixTexture((IDirect3DDevice9*)dev, *entry.variants, "paramx");
            }
        }
    } else {
        entry.hasValidName = false;
        entry.variants = nullptr;
    }

    AcquireSRWLockExclusive(&textureSuffixLock);
    textureSuffixResolutionCache.emplace(texture, std::move(entry));
    ReleaseSRWLockExclusive(&textureSuffixLock);
}

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
    // Join precache thread — it ran during BSA/distant land init, should be nearly done
    if (precacheThread) {
        WaitForSingleObject(precacheThread, INFINITE);
        CloseHandle(precacheThread);
        precacheThread = nullptr;
        LOG::logline("-- Precache thread joined, %d shaders in cache", (int)cacheHLSLShaders.size());
    }

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

        // Disable Morrowind sunglare in HLSL mode (HLSL handles sun effects differently)
        Configuration.MGEFlags |= NO_MW_SUNGLARE;
        MWBridge::get()->disableSunglare();
        LOG::logline("-- Morrowind sunglare disabled for HLSL mode");

        // Reset LRU only — don't clear cacheHLSLShaders, precache thread already populated it
        hlslShaderLRU.shader = {};
        hlslShaderLRU.last_sk = ShaderKey();
        
        // Create default error shader for HLSL pipeline
        hlslShaderDefaultPurple = createPurpleErrorShader();

        // Start async compiler for on-demand compilation of remaining variants
        // (precache thread already populated the cache during splash screens)
        startAsyncCompiler();
    }

    // Register texture release callback for evict-on-release cache management
    g_onTextureReleased = onTextureReleased;

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
    // Set device early and start immediate precaching in a tracked thread
    if (!device && d) {
        device = d;
        LOG::logline("-- Starting immediate HLSL shader precaching (tracked thread)");

        // Launch a joinable Win32 thread (can be stopped on cell change / shutdown)
        precacheThread = CreateThread(nullptr, 0, [](LPVOID) -> DWORD {
            bool hlslMode = (Configuration.PerPixelLightFlags == 2);

            if (hlslMode) {
                LOG::logline("-- Precache thread started");

                int hlslVariants = 0;

                auto updateStatus = [&](int current, int total) {
                    char progressText[128];
                    std::snprintf(progressText, sizeof(progressText), "Compiling HLSL shaders: %d/%d", current, total);
                    StatusOverlay::setStatus(progressText);
                };

                // Variant struct: {lighting, lightMode, vertexCol, vertexMat, heavyLighting, skinning, dp, ph, px, fogMode, stages}
                // Derived from runtime hit data — only variants actually seen in gameplay
                struct ShaderVariant {
                    int lighting, lightMode, vertexCol, vertexMat, heavyLighting, skinning;
                    int hasDiffParam, hasParamH, hasParamX, fogMode, stages;
                };

                ShaderVariant variants[] = {
                    // lm=0 (sun only) — base, dp+ph, skinning, vm=1 skinning
                    {1,0, 0,1, 0,0, 0,0,0, 1,1}, {1,0, 1,2, 0,0, 0,0,0, 1,1},
                    {1,0, 0,1, 0,0, 1,1,0, 1,1}, {1,0, 1,2, 0,0, 1,1,0, 1,1},
                    {1,0, 0,1, 0,1, 0,0,0, 1,1}, {1,0, 1,2, 0,1, 0,0,0, 1,1},
                    {1,0, 1,1, 0,0, 0,0,0, 1,1}, {1,0, 1,1, 0,1, 0,0,0, 1,1},
                    {1,0, 1,1, 0,1, 1,1,0, 1,1},
                    {1,0, 0,1, 0,0, 0,0,0, 1,2},  // dual texture

                    // lm=1 (single point light) — base, dp+ph, skinning, vm=1 skinning
                    {1,1, 0,1, 0,0, 0,0,0, 1,1}, {1,1, 1,2, 0,0, 0,0,0, 1,1},
                    {1,1, 0,1, 0,0, 1,1,0, 1,1}, {1,1, 1,2, 0,0, 1,1,0, 1,1},
                    {1,1, 0,1, 0,1, 0,0,0, 1,1},
                    {1,1, 1,1, 0,1, 0,0,0, 1,1}, {1,1, 1,1, 0,1, 1,1,0, 1,1},
                    {1,1, 1,2, 0,0, 0,0,0, 2,1},  // fog=2

                    // lm=2 (few lights) — hl=0 and hl=1, base, dp+ph, skinning
                    {1,2, 0,1, 0,0, 0,0,0, 1,1}, {1,2, 0,1, 1,0, 0,0,0, 1,1},
                    {1,2, 1,2, 0,0, 0,0,0, 1,1}, {1,2, 1,2, 1,0, 0,0,0, 1,1},
                    {1,2, 0,1, 0,0, 1,1,0, 1,1}, {1,2, 0,1, 1,0, 1,1,0, 1,1},
                    {1,2, 1,2, 0,0, 1,1,0, 1,1}, {1,2, 1,2, 1,0, 1,1,0, 1,1},
                    {1,2, 0,1, 0,1, 0,0,0, 1,1}, {1,2, 0,1, 1,1, 0,0,0, 1,1},
                    {1,2, 1,2, 0,1, 0,0,0, 1,1}, {1,2, 1,2, 1,1, 0,0,0, 1,1},
                    {1,2, 1,1, 0,1, 1,1,0, 1,1},  // vm=1 skinning dp+ph
                    {1,2, 0,1, 0,0, 0,0,0, 1,2},  // dual texture
                    {1,2, 1,2, 0,0, 0,0,0, 2,1}, {1,2, 1,2, 1,0, 0,0,0, 2,1},  // fog=2

                    // lm=3 (texture lights, always hl=1) — base, dp+ph, fog=2
                    {1,3, 0,1, 1,0, 0,0,0, 1,1}, {1,3, 1,2, 1,0, 0,0,0, 1,1},
                    {1,3, 0,1, 1,0, 1,1,0, 1,1}, {1,3, 1,2, 1,0, 1,1,0, 1,1},
                    {1,3, 1,2, 1,0, 0,0,0, 2,1},  // fog=2
                };

                const int totalVariants = sizeof(variants) / sizeof(variants[0]);
                LOG::logline("-- Precaching %d HLSL shader variants (tracked thread)", totalVariants);

                for (int i = 0; i < totalVariants; i++) {
                    const auto& v = variants[i];

                    if (i % 5 == 0 || i == totalVariants - 1) {
                        updateStatus(i + 1, totalVariants);
                    }

                    {
                        ShaderKey sk;
                        memset(&sk, 0, sizeof(sk));
                        sk.uvSets = 1;
                        sk.useLighting = v.lighting;
                        sk.lightMode = v.lightMode;
                        sk.vertexColour = v.vertexCol;
                        sk.vertexMaterial = v.vertexMat;
                        sk.usesSkinning = v.skinning;
                        sk.heavyLighting = v.heavyLighting;
                        sk.hasDiffParam = v.hasDiffParam;
                        sk.hasParamH = v.hasParamH;
                        sk.hasParamX = v.hasParamX;
                        sk.hasGrass = 0;
                        sk.hasShadows = ((Configuration.MGEFlags & USE_SHADOWS) && (Configuration.MGEFlags & USE_DISTANT_LAND)) ? 1 : 0;
                        sk.fogMode = v.fogMode;
                        sk.activeStages = v.stages;

                        sk.stage[0] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 1, 0, 0, 0 };
                        if (v.stages > 1) {
                            sk.stage[1] = { D3DTOP_ADD, D3DTA_TEXTURE, D3DTA_CURRENT, D3DTA_CURRENT, 0, 0, 0, 0 };
                        }
                        memset(&sk.stage[v.stages], 0, sizeof(sk.stage[0]) * (8 - v.stages));

                        // Compile shader (no lock needed for generateMWShaderHLSL — it no longer writes cache)
                        HLSLShader shader = generateMWShaderHLSL(sk);

                        // Insert under exclusive lock
                        AcquireSRWLockExclusive(&hlslCacheLock);
                        if (cacheHLSLShaders.find(sk) == cacheHLSLShaders.end()) {
                            cacheHLSLShaders[sk] = shader;
                            hlslVariants++;
                        }
                        ReleaseSRWLockExclusive(&hlslCacheLock);
                    }
                }

                // Grass variants (hasGrass=1) — separate because main loop hardcodes hasGrass=0
                // From cache miss log: lm=0 vc=0/1, lm=1 vc=1
                struct GrassVariant { int lightMode, vertexCol, vertexMat; };
                GrassVariant grassVariants[] = {
                    {0, 0, 1}, {0, 1, 2},
                    {1, 0, 1}, {1, 1, 2},
                };
                for (const auto& gv : grassVariants) {
                    ShaderKey sk;
                    memset(&sk, 0, sizeof(sk));
                    sk.uvSets = 1;
                    sk.useLighting = 1;
                    sk.lightMode = gv.lightMode;
                    sk.vertexColour = gv.vertexCol;
                    sk.vertexMaterial = gv.vertexMat;
                    sk.fogMode = 1;
                    sk.activeStages = 1;
                    sk.hasGrass = 1;
                    sk.hasShadows = ((Configuration.MGEFlags & USE_SHADOWS) && (Configuration.MGEFlags & USE_DISTANT_LAND)) ? 1 : 0;
                    sk.stage[0] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 1, 0, 0, 0 };

                    HLSLShader shader = generateMWShaderHLSL(sk);
                    AcquireSRWLockExclusive(&hlslCacheLock);
                    if (cacheHLSLShaders.find(sk) == cacheHLSLShaders.end()) {
                        cacheHLSLShaders[sk] = shader;
                        hlslVariants++;
                    }
                    ReleaseSRWLockExclusive(&hlslCacheLock);
                }

                StatusOverlay::setStatus("HLSL shader precaching complete");
                LOG::logline("-- Precache thread completed: %d shaders compiled", hlslVariants);
            }
            return 0;
        }, nullptr, 0, nullptr);
    }
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
    // Per-object packed texture takes priority (mode 3 spatial query)
    if (texPerObjectLightData) {
        setCachedTexture(device, 5, texPerObjectLightData);
    } else if (DistantLand::texLightData) {
        setCachedTexture(device, 5, DistantLand::texLightData);
    }
}


// Helper function to compute ShaderKey with texture suffix detection
FixedFunctionShader::ShaderKey FixedFunctionShader::computeShaderKeyWithSuffixes(const RenderedState* rs, const FragmentState* frs, LightState* lightrs) {

    // Step 1: Determine texture suffix availability
    bool hasDiffParam = false, hasParamH = false, hasParamX = false, hasGrass = false;

    if (rs->texture) {
        // Check texture suffix resolution cache (protected by shared lock for thread safety)
        AcquireSRWLockShared(&textureSuffixLock);
        auto cacheIt = textureSuffixResolutionCache.find(rs->texture);
        if (cacheIt != textureSuffixResolutionCache.end()) {
            // Cache hit — read suffix flags under shared lock
            if (cacheIt->second.hasValidName && cacheIt->second.variants) {
                hasDiffParam = cacheIt->second.variants->hasDiffParam() || cacheIt->second.variants->hasDiffParamT();
                hasParamH = cacheIt->second.variants->hasParamH();
                hasParamX = cacheIt->second.variants->hasParamX();
                hasGrass = cacheIt->second.variants->hasGrass();
            }
            ReleaseSRWLockShared(&textureSuffixLock);
        } else {
            ReleaseSRWLockShared(&textureSuffixLock);

            if (deviceCallsSafeInPrepare) {
                // Main thread: perform expensive resolution (original fallback)
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

                if (entry.hasValidName && entry.variants) {
                    hasDiffParam = entry.variants->hasDiffParam() || entry.variants->hasDiffParamT();
                    hasParamH = entry.variants->hasParamH();
                    hasParamX = entry.variants->hasParamX();
                    hasGrass = entry.variants->hasGrass();
                }

                AcquireSRWLockExclusive(&textureSuffixLock);
                textureSuffixResolutionCache.emplace(rs->texture, std::move(entry));
                ReleaseSRWLockExclusive(&textureSuffixLock);
            } else {
                // Cull thread: device calls not safe. warmSuffixCache should have populated
                // this during recording. Skip suffix detection (default flags = no suffixes).
                LOG::logline("Warning: suffix cache miss for texture 0x%p on cull thread", rs->texture);
            }
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

    // Immediate path for hands (Scene 1+) and fallback when recording disabled.
    // Uses game's built-in 8 lights via lightrs, but needs shadow matrices computed.
    if (!ImGuiManager::GetEnableImmediateRendering()) {
        return;  // Skip immediate rendering if disabled
    }

    // Compute shadow world-view-projection matrices for this draw call
    // (rs from mged3d8device has zeros — compute from current shadow map VP)
    RenderedState rsWithShadows = *rs;
    rsWithShadows.shadowWorldViewProj[0] = rs->worldTransforms[0] * DistantLand::smViewproj[0];
    rsWithShadows.shadowWorldViewProj[1] = rs->worldTransforms[0] * DistantLand::smViewproj[1];

    renderMorrowindHLSL_Internal(&rsWithShadows, frs, lightrs);
}

// Internal rendering function that does the actual HLSL rendering
void FixedFunctionShader::renderMorrowindHLSL_Internal(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, DWORD dirtyFlags, int callIndex) {

    // Process any completed async shader compilations
    processAsyncCompletions();

    HLSLShader hlslShader;

    // Get ShaderKey with texture suffix detection
    ShaderKey sk;
    if (isReplaying) {
        // During replay, use the recorded ShaderKey with original suffix flags
        for (const auto& call : frameBuffers[recordingBuffer].recordedCalls) {
            if (&call.rs == rs) {
                sk = call.sk;
                break;
            }
        }
    } else {
        // During normal rendering, compute ShaderKey with texture suffix detection
        sk = computeShaderKeyWithSuffixes(rs, frs, lightrs);

        // Clamp lightMode to 2 (uniform-based) for immediate path.
        // Mode 3 (texture-based) requires per-object light packing from prepareRecordedCalls.
        if (sk.lightMode > 2) {
            sk.lightMode = 2;
        }
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
        bool exactHit = false;
        // Read from shader cache under shared lock (precache thread may be inserting)
        AcquireSRWLockShared(&hlslCacheLock);
        decltype(cacheHLSLShaders)::const_iterator iShader = cacheHLSLShaders.find(sk);

        if (iShader != cacheHLSLShaders.end()) {
            hlslShader = iShader->second;
            exactHit = true;
            if (hlslDiagFrameCounter <= 5) {
                diagHitKeys.insert(sk);
            }
        }

        if (!exactHit) {
            // Diagnostic: log cache misses on early frames to identify precache gaps
            if (hlslDiagFrameCounter <= 3) {
                char buf[512];
                snprintf(buf, sizeof(buf),
                    "CACHE MISS frame=%d: lm=%d lit=%d vc=%d vm=%d hl=%d skin=%d fog=%d uv=%d stages=%d shadow=%d detail=%d dp=%d ph=%d px=%d grass=%d bump=%d tg=%d",
                    hlslDiagFrameCounter,
                    (int)sk.lightMode, (int)sk.useLighting, (int)sk.vertexColour,
                    (int)sk.vertexMaterial, (int)sk.heavyLighting,
                    (int)sk.usesSkinning, (int)sk.fogMode, (int)sk.uvSets,
                    (int)sk.activeStages,
                    (int)sk.hasShadows, (int)sk.hasDetail, (int)sk.hasDiffParam,
                    (int)sk.hasParamH, (int)sk.hasParamX, (int)sk.hasGrass,
                    (int)sk.usesBumpmap, (int)sk.usesTexgen);
                LOG::logline("%s", buf);
                // Log per-stage details
                for (int s = 0; s < (int)sk.activeStages && s < 8; ++s) {
                    snprintf(buf, sizeof(buf),
                        "  stg%d=[op=%d a1=%d a2=%d a0=%d am=%d as=%d ti=%d tg=%d]",
                        s,
                        (int)sk.stage[s].colorOp, (int)sk.stage[s].colorArg1,
                        (int)sk.stage[s].colorArg2, (int)sk.stage[s].colorArg0,
                        (int)sk.stage[s].alphaOpMatched, (int)sk.stage[s].alphaOpSelect1,
                        (int)sk.stage[s].texcoordIndex, (int)sk.stage[s].texcoordGen);
                    LOG::logline("%s", buf);
                }
            }
            // Smart fallback hierarchy before using purple
            ShaderKey fallbackSk = sk;
            HLSLShader fallbackShader = {};
            bool foundFallback = false;

            // Normalize only alphaOpSelect1 (spurious variation); leave alphaOpMatched
            // untouched — it affects code generation and differs between stage[0] and stage[1+]
            auto normalizeAlpha = [](ShaderKey& key) {
                for (int s = 0; s < (int)key.activeStages; ++s) {
                    key.stage[s].alphaOpSelect1 = 0;
                }
            };

            // Normalize alpha bits on the base fallback key
            normalizeAlpha(fallbackSk);

            // lightMode + heavyLighting cascade: try different light modes AND hl=0
            int fallbackModes[] = {2, 1, 0};
            for (int hl = (int)sk.heavyLighting; hl >= 0 && !foundFallback; --hl) {
                fallbackSk.heavyLighting = hl;
                for (int i = 0; i < 3 && !foundFallback; ++i) {
                    if (fallbackModes[i] == (int)sk.lightMode && hl == (int)sk.heavyLighting) continue;
                    fallbackSk.lightMode = fallbackModes[i];
                    auto fallbackIter = cacheHLSLShaders.find(fallbackSk);
                    if (fallbackIter != cacheHLSLShaders.end()) {
                        fallbackShader = fallbackIter->second;
                        foundFallback = true;
                    }
                }
            }

            // If no point light fallback found, try texture suffix fallbacks
            if (!foundFallback && (sk.hasDiffParam || sk.hasParamH || sk.hasParamX)) {
                ShaderKey textureFallbackSk = sk;
                normalizeAlpha(textureFallbackSk);

                if (sk.hasDiffParam && !foundFallback) {
                    textureFallbackSk.hasDiffParam = 0;
                    auto fallbackIter = cacheHLSLShaders.find(textureFallbackSk);
                    if (fallbackIter != cacheHLSLShaders.end()) {
                        fallbackShader = fallbackIter->second;
                        foundFallback = true;
                    }
                }

                if (sk.hasParamH && !foundFallback) {
                    textureFallbackSk = sk;
                    normalizeAlpha(textureFallbackSk);
                    textureFallbackSk.hasDiffParam = 0;
                    textureFallbackSk.hasParamH = 0;
                    textureFallbackSk.hasParamX = 0;
                    auto fallbackIter = cacheHLSLShaders.find(textureFallbackSk);
                    if (fallbackIter != cacheHLSLShaders.end()) {
                        fallbackShader = fallbackIter->second;
                        foundFallback = true;
                    }
                }

                if (!foundFallback) {
                    textureFallbackSk = sk;
                    normalizeAlpha(textureFallbackSk);
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
                ShaderKey universalSk = sk;
                normalizeAlpha(universalSk);
                universalSk.heavyLighting = 0;
                universalSk.hasDiffParam = 0;
                universalSk.hasParamH = 0;
                universalSk.hasParamX = 0;

                for (int lm = 0; lm <= 2 && !foundFallback; ++lm) {
                    for (int vertCol = 0; vertCol <= 1 && !foundFallback; ++vertCol) {
                        for (int skinning = 0; skinning <= 1 && !foundFallback; ++skinning) {
                            universalSk.lightMode = lm;
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

            // Last resort: completely standardize key to match precache patterns
            // Resets fogMode, texgen, grass, bump, detail, and stage data
            if (!foundFallback) {
                ShaderKey stdSk;
                memset(&stdSk, 0, sizeof(stdSk));
                stdSk.uvSets = 1;
                stdSk.useLighting = sk.useLighting;
                stdSk.fogMode = 1;
                stdSk.activeStages = 1;
                stdSk.hasShadows = sk.hasShadows;
                stdSk.stage[0] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 1, 0, 0, 0 };

                for (int lm = 0; lm <= 2 && !foundFallback; ++lm) {
                    for (int vertCol = 0; vertCol <= 1 && !foundFallback; ++vertCol) {
                        for (int skinning = 0; skinning <= 1 && !foundFallback; ++skinning) {
                            stdSk.lightMode = lm;
                            stdSk.vertexColour = vertCol;
                            stdSk.vertexMaterial = vertCol + 1;
                            stdSk.usesSkinning = skinning;

                            auto fallbackIter = cacheHLSLShaders.find(stdSk);
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
        ReleaseSRWLockShared(&hlslCacheLock);

        // Queue async compilation for exact key on cache miss (outside lock)
        if (!exactHit) {
            queueShaderCompilation(sk);
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
        projMatrix = frameBuffers[recordingBuffer].proj;
        viewMatrix = frameBuffers[recordingBuffer].view;
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
        worldViewProj = rs->worldTransforms[0] * frameBuffers[recordingBuffer].view * frameBuffers[recordingBuffer].proj;
        worldView = rs->worldTransforms[0] * frameBuffers[recordingBuffer].view;
    } else {
        // Normal rendering - calculate from current matrices
        worldViewProj = worldMatrix * viewMatrix * projMatrix;
        worldView = worldMatrix * viewMatrix;
    }
    
    // Use constant tables to set matrices with cached handles (no per-draw string lookups)
    // Constant-setting failures are non-fatal: WorldViewProj is the critical transform,
    // other constants (View, Proj, World, etc.) use stale values if SetMatrix fails.
    if (hlslShader.vsConstantTable) {
        try {
            if (hlslShader.hWorldViewProj) {
                hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hWorldViewProj, &worldViewProj);
            }

            if (hlslShader.hView) {
                hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hView, &viewMatrix);
            }

            if (hlslShader.hProj) {
                HRESULT hr = hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hProj, &projMatrix);
                if (FAILED(hr)) {
                    // Constant table SetMatrix failed — bypass it with direct register write
                    // SetMatrix transposes internally, so we must transpose before SetVertexShaderConstantF
                    D3DXMATRIX projT;
                    D3DXMatrixTranspose(&projT, &projMatrix);
                    device->SetVertexShaderConstantF(hlslShader.projRegister, (float*)&projT, 4);
                    static bool projWarningLogged = false;
                    if (!projWarningLogged) {
                        LOG::logline("!! HLSL: Proj SetMatrix failed (hr=0x%08x), using direct register %d fallback", hr, hlslShader.projRegister);
                        projWarningLogged = true;
                    }
                }
            }

            if (hlslShader.hWorld) {
                hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hWorld, &rs->worldTransforms[0]);
            }

            if (hlslShader.hWorldView) {
                hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hWorldView, &worldView);
            }
            if (hlslShader.hVertexBlendPalette) {
                if (rs->vertexBlendState > 0) {
                    D3DXMATRIX currentWorldViewTransforms[4];
                    for (int i = 0; i < 4; i++) {
                        currentWorldViewTransforms[i] = rs->worldTransforms[i] * viewMatrix;
                    }
                    hlslShader.vsConstantTable->SetMatrixArray(device, hlslShader.hVertexBlendPalette, currentWorldViewTransforms, 4);
                } else {
                    D3DXMATRIX blendMatrices[4];
                    blendMatrices[0] = worldView;
                    memset(&blendMatrices[1], 0, sizeof(D3DXMATRIX) * 3);
                    hlslShader.vsConstantTable->SetMatrixArray(device, hlslShader.hVertexBlendPalette, blendMatrices, 4);
                }
            }

            if (hlslShader.hVertexBlendState) {
                D3DXVECTOR4 blendState((float)rs->vertexBlendState, 0, 0, 0);
                hlslShader.vsConstantTable->SetVector(device, hlslShader.hVertexBlendState, &blendState);
            }

            if (hlslShader.hShadowWorldViewProj) {
                hlslShader.vsConstantTable->SetMatrixArray(device, hlslShader.hShadowWorldViewProj, rs->shadowWorldViewProj, 2);
            }
        } catch (...) {
            LOG::logline("!! HLSL Vertex shader constant table access failed - shader may have been edited");
        }
    }
    // Set pixel shader constants using cached handles (no per-draw string lookups)
    // Constant-setting failures are non-fatal: stale material values are acceptable.
    if (hlslShader.psConstantTable) {
        try {
            if (hlslShader.hMaterialDiffuse) {
                hlslShader.psConstantTable->SetVector(device, hlslShader.hMaterialDiffuse, (D3DXVECTOR4*)&frs->material.diffuse);
            }

            if (hlslShader.hMaterialAmbient) {
                hlslShader.psConstantTable->SetVector(device, hlslShader.hMaterialAmbient, (D3DXVECTOR4*)&frs->material.ambient);
            }

            if (hlslShader.hMaterialEmissive) {
                hlslShader.psConstantTable->SetVector(device, hlslShader.hMaterialEmissive, (D3DXVECTOR4*)&frs->material.emissive);
            }
        // Set up lighting — extract sun + ambient for all modes, point lights only for modes 1-2
        const size_t MaxLights = 8;
        D3DXVECTOR4 bufferDiffuse[MaxLights];
        float bufferAmbient[MaxLights];
        float bufferPosition[3 * MaxLights];
        float bufferFalloffQuadratic[MaxLights], bufferFalloffLinear[MaxLights], bufferFalloffConstant;
        bool needPointLightBuffers = (sk.lightMode == 1 || sk.lightMode == 2);

        if (needPointLightBuffers) {
            memset(&bufferDiffuse, 0, sizeof(bufferDiffuse));
            memset(&bufferAmbient, 0, sizeof(bufferAmbient));
            memset(&bufferPosition, 0, sizeof(bufferPosition));
            memset(&bufferFalloffQuadratic, 0, sizeof(bufferFalloffQuadratic));
            memset(&bufferFalloffLinear, 0, sizeof(bufferFalloffLinear));
        }
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

            if (light->type == D3DLIGHT_POINT && needPointLightBuffers) {
                memcpy(&bufferDiffuse[pointLightCount], &light->diffuse, sizeof(light->diffuse));

                // Scatter position vectors for vectorization
                bufferPosition[pointLightCount] = light->viewspacePos.x;
                bufferPosition[pointLightCount + MaxLights] = light->viewspacePos.y;
                bufferPosition[pointLightCount + 2*MaxLights] = light->viewspacePos.z;

                // Scatter attenuation factors for vectorization (match Effect path)
                if (light->falloff.x > 0) {
                    bufferFalloffConstant = light->falloff.x;
                    bufferFalloffLinear[pointLightCount] = light->falloff.y;
                    bufferFalloffQuadratic[pointLightCount] = light->falloff.z;
                } else if (light->falloff.z > 0) {
                    bufferDiffuse[pointLightCount].x *= bufferFalloffConstant;
                    bufferDiffuse[pointLightCount].y *= bufferFalloffConstant;
                    bufferDiffuse[pointLightCount].z *= bufferFalloffConstant;
                    bufferAmbient[pointLightCount] = 1.0f + 1e-4f / sqrt(light->falloff.z);
                    bufferFalloffQuadratic[pointLightCount] = bufferFalloffConstant * light->falloff.z;
                } else if (light->falloff.y == 0.10000001f) {
                    bufferFalloffQuadratic[pointLightCount] = 5e-5;
                } else if (light->falloff.y > 0) {
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
            ambient.r = ambient.g = ambient.b = 1.25;
            sunDiffuse.r = sunDiffuse.g = sunDiffuse.b = 0.0;
        }

        // Sun + ambient constants (all modes)
        if (hlslShader.hLightSunDirection) {
            hlslShader.psConstantTable->SetFloatArray(device, hlslShader.hLightSunDirection, (const float*)&sunDirection, 3);
        }

        if (hlslShader.hLightSunDiffuse) {
            hlslShader.psConstantTable->SetFloatArray(device, hlslShader.hLightSunDiffuse, (const float*)&sunDiffuse, 3);
        }

        if (hlslShader.hLightSceneAmbient) {
            hlslShader.psConstantTable->SetFloatArray(device, hlslShader.hLightSceneAmbient, (const float*)&ambient, 3);
        }

        // Point light uniforms — only for lightMode 1 (single) and 2 (few loop)
        if (needPointLightBuffers) {
            if (hlslShader.hLightDiffuse) {
                hlslShader.psConstantTable->SetVectorArray(device, hlslShader.hLightDiffuse, bufferDiffuse, MaxLights);
            }

            if (hlslShader.hLightPosition) {
                D3DXVECTOR3 hlslLightPositions[MaxLights];
                for (int i = 0; i < (int)MaxLights; i++) {
                    hlslLightPositions[i].x = bufferPosition[i];
                    hlslLightPositions[i].y = bufferPosition[i + MaxLights];
                    hlslLightPositions[i].z = bufferPosition[i + 2*MaxLights];
                }
                hlslShader.psConstantTable->SetFloatArray(device, hlslShader.hLightPosition, (float*)hlslLightPositions, 3 * MaxLights);
            }

            if (hlslShader.hLightAmbient) {
                hlslShader.psConstantTable->SetFloatArray(device, hlslShader.hLightAmbient, bufferAmbient, MaxLights);
            }

            if (hlslShader.hPointLightCount) {
                hlslShader.psConstantTable->SetInt(device, hlslShader.hPointLightCount, (int)pointLightCount);
            }

            if (hlslShader.hLightFalloffQuadratic) {
                D3DXVECTOR4 quadraticData[2];
                for (int i = 0; i < 4; i++) {
                    quadraticData[0][i] = ((size_t)i < pointLightCount) ? bufferFalloffQuadratic[i] : 0.0f;
                    quadraticData[1][i] = ((size_t)(i + 4) < pointLightCount) ? bufferFalloffQuadratic[i + 4] : 0.0f;
                }
                hlslShader.psConstantTable->SetVectorArray(device, hlslShader.hLightFalloffQuadratic, quadraticData, 2);
            }

            if (hlslShader.hLightFalloffConstant) {
                hlslShader.psConstantTable->SetFloat(device, hlslShader.hLightFalloffConstant, bufferFalloffConstant);
            }
        }

        // Per-object light texture parameters for mode 3 (saturated objects)
        if (sk.lightMode == 3 && callIndex >= 0 && callIndex < (int)perObjectLightInfo.size()) {
            const auto& li = perObjectLightInfo[callIndex];
            float lightParams[4] = {
                (float)li.lightCount,
                perObjectTexelSize,
                (float)li.texelOffset,
                0.0f
            };
            device->SetPixelShaderConstantF(50, lightParams, 1);
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
            LOG::logline("!! HLSL Pixel shader constant table access failed");
        }
    }

    if (!rs->vb) {
        return;
    }

    HRESULT hr = device->SetFVF(rs->fvf);
    if (FAILED(hr)) {
        return;
    }

    hr = device->SetStreamSource(0, rs->vb, rs->vbOffset, rs->vbStride);
    if (FAILED(hr)) {
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
                // Cache will be cleared by invalidateShaderSourceCache() on main thread
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
    // 1. Join precache thread if still running
    if (precacheThread) {
        WaitForSingleObject(precacheThread, INFINITE);
        CloseHandle(precacheThread);
        precacheThread = nullptr;
    }

    // 2. Stop async compiler — joins thread and drains queue
    stopAsyncCompiler();

    // 3. Acquire exclusive lock and clear everything
    AcquireSRWLockExclusive(&hlslCacheLock);

    // Reset LRU to prevent use-after-free of released shader pointers
    hlslShaderLRU.shader = {};
    hlslShaderLRU.last_sk = ShaderKey();

    // Release COM objects and clear compiled shader cache
    for (auto& i : cacheHLSLShaders) {
        if (i.second.vertexShader) i.second.vertexShader->Release();
        if (i.second.pixelShader) i.second.pixelShader->Release();
        if (i.second.vsConstantTable) i.second.vsConstantTable->Release();
        if (i.second.psConstantTable) i.second.psConstantTable->Release();
    }
    cacheHLSLShaders.clear();

    // Clear shader source cache for hot reloading
    for (auto& i : shaderSourceCache) {
        delete[] i.second.source;
    }
    shaderSourceCache.clear();

    ReleaseSRWLockExclusive(&hlslCacheLock);

    // 4. Restart async compiler for future on-demand compilations
    startAsyncCompiler();
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
    bool anyCompleted = false;
    while (it != pendingCompilations.end()) {
        if (it->second->completed) {
            // Move completed shader to main cache under exclusive lock
            AcquireSRWLockExclusive(&hlslCacheLock);
            cacheHLSLShaders[it->first] = it->second->result;
            ReleaseSRWLockExclusive(&hlslCacheLock);
            anyCompleted = true;

            it = pendingCompilations.erase(it);
        } else {
            ++it;
        }
    }
    // Invalidate LRU so next draw picks up newly cached shaders
    if (anyCompleted) {
        hlslShaderLRU.last_sk = ShaderKey();
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
    D3D_SHADER_MACRO defines[10] = {};
    int defineCount = 0;

    // Light mode define: 0=sun only, 1=single, 2=few loop, 3=texture
    char lightModeStr[2] = {'0', '\0'};
    lightModeStr[0] = (char)('0' + sk.lightMode);
    defines[defineCount++] = {"LIGHT_MODE", lightModeStr};

    // Only emit USE_TEXTURE_LIGHTS for mode 3 (>8 lights, texture loop)
    if (sk.lightMode == 3) {
        defines[defineCount++] = {"USE_TEXTURE_LIGHTS", "1"};
    }

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
        hlslShader.projRegister = 0;
        if (hlslShader.hProj) {
            D3DXCONSTANT_DESC desc;
            UINT count = 1;
            if (SUCCEEDED(hlslShader.vsConstantTable->GetConstantDesc(hlslShader.hProj, &desc, &count))) {
                hlslShader.projRegister = desc.RegisterIndex;
            }
        }
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

        // Cache point light constant handles (only present for lightMode 1-2)
        hlslShader.hLightDiffuse = hlslShader.psConstantTable->GetConstantByName(NULL, "lightDiffuse");
        hlslShader.hLightPosition = hlslShader.psConstantTable->GetConstantByName(NULL, "lightPosition");
        hlslShader.hLightAmbient = hlslShader.psConstantTable->GetConstantByName(NULL, "lightAmbient");
        hlslShader.hPointLightCount = hlslShader.psConstantTable->GetConstantByName(NULL, "pointLightCount");
        hlslShader.hLightFalloffQuadratic = hlslShader.psConstantTable->GetConstantByName(NULL, "lightFalloffQuadratic");
        hlslShader.hLightFalloffConstant = hlslShader.psConstantTable->GetConstantByName(NULL, "lightFalloffConstant");
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
    
    return hlslShader;
}

void FixedFunctionShader::release() {
    // Join precache thread before cleaning up HLSL cache
    if (precacheThread) {
        WaitForSingleObject(precacheThread, INFINITE);
        CloseHandle(precacheThread);
        precacheThread = nullptr;
    }

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

    // Clean up per-object light texture
    if (texPerObjectLightData) {
        texPerObjectLightData->Release();
        texPerObjectLightData = nullptr;
    }
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
    if (!rs->useLighting || pointLightCount == 0) lightMode = 0;
    else if (pointLightCount == 1) lightMode = 1;
    else if (pointLightCount <= 6) lightMode = 2;
    else lightMode = 3;

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

// Helper: get current recording buffer's calls vector (eliminates static recordedCalls)
static auto& currentRecordedCalls() { return FixedFunctionShader::frameBuffers[FixedFunctionShader::recordingBuffer].recordedCalls; }

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

    currentRecordedCalls().reserve(4000);  // Pre-allocate (already cleared by rotateRecordingBuffer)
    samplerCache.clear();  // Clear sampler cache for new frame
    bboxLookup.clear();  // Clear for new frame (populated during recording)
    // NOTE: Do NOT clear recordMW here - it's populated by inspectIndexedPrimitive()
    // BEFORE startRecording() is called. recordMW is cleared at end of renderStage1/2.
    lastLightState.reset();  // Clear LightState cache for new recording

    // Clear pointer-keyed caches on cell transitions.
    // Morrowind reuses freed VB/IB/texture pointers for different data after cell changes,
    // so pointer-keyed caches (bboxCache, textureSuffixResolutionCache, blacklist) return stale data.
    {
        static bool lastWasExterior = false;
        static void* lastPlayerCell = nullptr;

        bool isExterior = MWBridge::get()->IsExterior();
        void* currentCell = MWBridge::get()->getPlayerCell();

        if (isExterior != lastWasExterior) {
            // Interior ↔ exterior transition: clear geometry caches
            // textureSuffixResolutionCache is now evict-on-release (no bulk clear needed)
            bboxCache.clear();
            softwareOcclusionCuller.clearBlacklist();
            lastWasExterior = isExterior;
        } else if (currentCell != lastPlayerCell) {
            // Any cell change (exterior-to-exterior, interior-to-interior): clear geometry caches
            bboxCache.clear();
            softwareOcclusionCuller.clearBlacklist();
        }

        lastPlayerCell = currentCell;
    }

    // Clear lights from previous frame (simple approach, no persistence)
    DistantLand::sceneLights.clear();
    DistantLand::sceneLightIndexMap.clear();
    isRecording = true;
    isReplaying = false;
    recordingCompletedThisFrame = false;  // Allow recording to proceed

    // Clear CPU depth buffer for new frame (replaces GPU Hi-Z readback)
    softwareOcclusionCuller.clear();

    // Capture view/projection matrices once at start of recording directly into FrameBuffer
    // Note: World transforms are captured per-call in each RenderedState
    auto& fb = frameBuffers[recordingBuffer];
    device->GetTransform(D3DTS_VIEW, &fb.view);
    device->GetTransform(D3DTS_PROJECTION, &fb.proj);
    fb.shadowViewproj[0] = DistantLand::smViewproj[0];
    fb.shadowViewproj[1] = DistantLand::smViewproj[1];
    fb.state = BufferState::Recording;
    fb.valid = true;

    LOG::logline("HLSL Recording: Started recording render dispatches");
}

// prepareOcclusionCullingForDepth - Build Hi-Z pyramid and filter recordMW before depth rendering
// Called at the start of renderStage1(), BEFORE renderDepth() executes
void FixedFunctionShader::prepareOcclusionCullingForDepth() {
    MGE_ZoneScopedN("Prepare Occlusion Culling for Depth");

    // Resize visibility results for this frame (indexed by draw order)
    visibilityResults.assign(DistantLand::recordMW.size(), -1);  // -1 = not yet tested

    // Phase 2a: Compute deferred bboxes and rasterize occluders from recorded HLSL calls
    // This must happen before Hi-Z build so the depth pass benefits from culling
    auto& recCalls = currentRecordedCalls();
    if (!recCalls.empty() && !hiZBuiltThisFrame) {
        // Compute bounding boxes for calls that missed the cache during recording
        {
            MGE_ZoneScopedN("Prepare: BBox Cache Misses");
            for (auto& call : recCalls) {
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
            MGE_ZoneScopedN("Prepare: Occluder Rasterization");
            int occluderCount = 0;
            const int MAX_OCCLUDERS = ImGuiManager::GetOccluderMaxCount();
            const float MIN_SCREEN_COVERAGE = 0.01f;
            rasterizedOccluderMeshes.clear();

            auto& fb = frameBuffers[recordingBuffer];
            D3DXMATRIX viewProj = fb.view * fb.proj;

            for (auto& call : recCalls) {
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
                    fb.view, fb.proj
                );
                occluderCount++;
            }
        }
    }

    // Build Hi-Z mipmap pyramid from rasterized occluders (only once per frame)
    if (!hiZBuiltThisFrame) {
        MGE_ZoneScopedN("Build Hi-Z Pyramid");
        softwareOcclusionCuller.buildHiZPyramid();
        if (ImGuiManager::GetShowHiZInterface()) {
            softwareOcclusionCuller.uploadHiZToTexture(reinterpret_cast<IDirect3DDevice9*>(device), ImGuiManager::GetHiZDisplayMip(), frameBuffers[recordingBuffer].proj, ImGuiManager::GetHiZInvert(), ImGuiManager::GetHiZShowRaycastGrid(), ImGuiManager::GetHiZRaycastStep());
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

    // Hi-Z bypass toggle for terrain hole diagnosis
    if (ImGuiManager::GetDisableHiZCulling()) {
        for (size_t i = 0; i < visibilityResults.size(); i++)
            visibilityResults[i] = 1;
        LOG::logline(">> Depth: %d objects (Hi-Z culling DISABLED by toggle)",
                     (int)DistantLand::recordMW.size());
        return;
    }

    // Direct Hi-Z culling for depth pass using bboxCache + current matrices
    {
        MGE_ZoneScopedN("Filter recordMW with Hi-Z Culling");

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
void FixedFunctionShader::matchPreviousFrameCalls(int bufferIndex) {
    MGE_ZoneScopedN("matchPreviousFrameCalls");

    auto& curCalls = frameBuffers[bufferIndex].recordedCalls;

    // Find previous frame's FrameBuffer (the one before this buffer)
    int prevBuf = (bufferIndex + 2) % 3;
    auto& prevCalls = frameBuffers[prevBuf].recordedCalls;

    if (prevCalls.empty()) {
        // First frame or no previous data — all dirty
        for (auto& call : curCalls) {
            call.dirtyFlags = DIRTY_ALL;
        }
        return;
    }

    // Build lookup from previous frame
    std::unordered_map<MeshKey, int, MeshKeyHash> prevLookup;
    prevLookup.reserve(prevCalls.size());
    for (int i = 0; i < (int)prevCalls.size(); ++i) {
        auto& prev = prevCalls[i];
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

    for (auto& call : curCalls) {
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

        auto& prev = prevCalls[it->second];
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
void FixedFunctionShader::prepareRecordedCalls(int bufferIndex) {
    MGE_ZoneScopedN("prepareRecordedCalls");

    auto& recCalls = frameBuffers[bufferIndex].recordedCalls;
    if (recCalls.empty()) {
        return;
    }

    // Compute shader keys and assign render bins (with slow-frame timing)
    LARGE_INTEGER freqQPC, prepStartQPC, prepEndQPC;
    QueryPerformanceFrequency(&freqQPC);
    QueryPerformanceCounter(&prepStartQPC);
    {
        MGE_ZoneScopedN("Prepare: Shader Keys + Bins");
        for (auto& call : recCalls) {
            call.sk = computeShaderKeyWithSuffixes(&call.rs, &call.frs, call.lightrs.get());
            call.prepared = true;

            // Classify into render bin
            if (call.sk.hasGrass)              call.bin = RenderBin::Grass;
            else if (call.sk.usesSkinning)     call.bin = RenderBin::Skinning;
            else if (call.rs.blendEnable)      call.bin = RenderBin::Blending;
            else if (call.rs.alphaTest)         call.bin = RenderBin::AlphaTested;
            else                                call.bin = RenderBin::Opaque;
        }
    }
    QueryPerformanceCounter(&prepEndQPC);
    lastPrepareMs = (prepEndQPC.QuadPart - prepStartQPC.QuadPart) * 1000.0f / freqQPC.QuadPart;
    if (lastPrepareMs > ImGuiManager::GetSlowFrameThreshold()) {
        LOG::logline("SLOW PREPARE: %.1fms for %d calls (Shader Keys + Bins)", lastPrepareMs, (int)recCalls.size());
    }

    // Performance mode: match against previous frame for dirty tracking
    if (ImGuiManager::GetPerformanceMode()) {
        matchPreviousFrameCalls(bufferIndex);
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

    // Batch-warm suffix cache — resolve all unique textures and pre-load suffix files
    // before prepare/replay, so neither stalls on hash computation or disk I/O.
    {
        MGE_ZoneScopedN("BatchWarmSuffixCache");
        auto& recCalls = frameBuffers[recordingBuffer].recordedCalls;
        std::unordered_set<IDirect3DTexture9*> seen;
        for (const auto& call : recCalls) {
            if (call.rs.texture && seen.insert(call.rs.texture).second) {
                AcquireSRWLockShared(&textureSuffixLock);
                bool found = textureSuffixResolutionCache.count(call.rs.texture) > 0;
                ReleaseSRWLockShared(&textureSuffixLock);
                if (!found) {
                    warmSuffixCache(device, call.rs.texture);
                }
            }
        }
    }

    // Phase 2b: Prepare shader keys (bbox + occluders already done in prepareOcclusionCullingForDepth)
    prepareRecordedCalls(recordingBuffer);

    // Phase 3: Replay all prepared calls (Hi-Z already built by prepareOcclusionCullingForDepth)
    replayRecordedCalls(0);

    // Data is already in frameBuffers[recordingBuffer].recordedCalls (recorded directly there)
    // No move or clear needed — buffer ownership transfers via rotateRecordingBuffer()

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

// Step 2: Submit recording to cull thread for async prepare (called before renderStage1)
void FixedFunctionShader::finalizeBatchAndSubmitCull() {
    // Handle dump request (same logic as finalizeBatchAndReplay)
    if (dumpRequested) {
        if (recordingEnabled) {
            auto& recCalls = currentRecordedCalls();
            LOG::logline("Frame dump: Dumping %d recorded calls (batch)", recCalls.size());
            StatusOverlay::setStatus("Frame dump: Batch complete");

            char logline[512];
            for (size_t i = 0; i < recCalls.size(); ++i) {
                const auto& call = recCalls[i];
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
            LOG::logline("Frame dump: Immediate dump complete");
            StatusOverlay::setStatus("Frame dump: Immediate complete");
        }
        dumpRequested = false;
    }

    if (!recordingEnabled || !isRecording || currentRecordedCalls().empty()) {
        // Nothing to cull — just mark Scene 0 as done
        isRecording = false;
        recordingCompletedThisFrame = true;
        return;
    }

    isRecording = false;

    // Capture Morrowind's end-of-Scene-0 device state for restoration after replay
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

    // Diagnostic: increment frame counter for cache miss/hit logging
    ++hlslDiagFrameCounter;

    // After frame 5, dump unused precached variants
    if (hlslDiagFrameCounter == 6) {
        LOG::logline("-- PRECACHE HIT REPORT: %d keys hit out of %d cached", (int)diagHitKeys.size(), (int)cacheHLSLShaders.size());
        AcquireSRWLockShared(&hlslCacheLock);
        for (const auto& entry : cacheHLSLShaders) {
            const auto& k = entry.first;
            bool hit = diagHitKeys.count(k) > 0;
            char buf[256];
            snprintf(buf, sizeof(buf),
                "%s lm=%d lit=%d vc=%d vm=%d hl=%d skin=%d fog=%d uv=%d stg=%d shd=%d det=%d dp=%d ph=%d px=%d gr=%d bm=%d tg=%d",
                hit ? "HIT " : "UNUSED",
                (int)k.lightMode, (int)k.useLighting, (int)k.vertexColour,
                (int)k.vertexMaterial, (int)k.heavyLighting,
                (int)k.usesSkinning, (int)k.fogMode, (int)k.uvSets,
                (int)k.activeStages,
                (int)k.hasShadows, (int)k.hasDetail, (int)k.hasDiffParam,
                (int)k.hasParamH, (int)k.hasParamX, (int)k.hasGrass,
                (int)k.usesBumpmap, (int)k.usesTexgen);
            LOG::logline("%s", buf);
        }
        ReleaseSRWLockShared(&hlslCacheLock);
        diagHitKeys.clear();
    }

    // Batch-warm suffix cache on main thread before cull thread gets the buffer.
    // Resolves all unique textures and pre-loads suffix files so that
    // computeShaderKeyWithSuffixes on the cull thread never hits expensive fallbacks,
    // and replay never stalls on hash computation or disk I/O.
    {
        MGE_ZoneScopedN("BatchWarmSuffixCache");
        auto& recCalls = frameBuffers[recordingBuffer].recordedCalls;
        std::unordered_set<IDirect3DTexture9*> seen;
        for (const auto& call : recCalls) {
            if (call.rs.texture && seen.insert(call.rs.texture).second) {
                AcquireSRWLockShared(&textureSuffixLock);
                bool found = textureSuffixResolutionCache.count(call.rs.texture) > 0;
                ReleaseSRWLockShared(&textureSuffixLock);
                if (!found) {
                    warmSuffixCache(device, call.rs.texture);
                }
            }
        }
    }

    // Submit to cull thread for async prepare
    int buf = recordingBuffer;
    frameBuffers[buf].state = BufferState::ReadyToCull;

    if (g_cullThread && g_cullThread->isRunning()) {
        g_cullThread->submitWork(buf, false);  // false = don't wait
    } else {
        // Fallback: no cull thread, run prepare inline
        executeCullPass(buf);
        frameBuffers[buf].state = BufferState::ReadyToRender;
    }

    recordingCompletedThisFrame = true;
}

// Step 2: Wait for cull completion and replay (called after renderStageBlend)
void FixedFunctionShader::waitCullAndReplay() {
    if (!recordingEnabled) return;

    auto& fb = frameBuffers[recordingBuffer];
    if (fb.state != BufferState::ReadyToCull && fb.state != BufferState::Culling
        && fb.state != BufferState::ReadyToRender) {
        return;  // Nothing was submitted
    }

    // Wait for cull thread to finish (should be done by now — renderStage1+Blend gave it time)
    if (g_cullThread && g_cullThread->isRunning() && fb.state != BufferState::ReadyToRender) {
        g_cullThread->waitForCompletion();
    }
    fb.state = BufferState::ReadyToRender;

    // Replay all prepared calls
    replayRecordedCalls(0);

    // Clean up HLSL-only texture slots to prevent DXVK descriptor bloat
    for (int i = 2; i < 6; i++) {
        device->SetTexture(i, NULL);
        textureCache.updateCache(i, nullptr);
        textureCache.textureValid[i] = false;
    }
    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);

    // Restore Morrowind's end-of-Scene-0 state
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

    isReplaying = false;

    // Reset HLSL caches after rendering session completes
    resetHLSLCaches();
}

// Call this when HLSL rendering session is complete to trigger replay
void FixedFunctionShader::finalizeBatchAndReplay(int sceneCount) {
    if (recordingEnabled) {
        if (sceneCount == 0) {
            // Scene 0: work already done by finalizeBatchAndSubmitCull() + waitCullAndReplay()
            // Just ensure clean state flags (should already be set, but defensive)
            isRecording = false;
            isReplaying = false;
            recordingCompletedThisFrame = true;
        } else {
            // Scene 1+: hands render via immediate path (with shadow matrices)
            currentRecordedCalls().clear();
            resetHLSLCaches();
        }
    } else {
        // When recording disabled, still clear calls after potential dump
        currentRecordedCalls().clear();
        resetHLSLCaches();
    }
}

// Scene lifecycle stubs for triple-buffered pipeline (Step 1: no-op, infrastructure only)
void FixedFunctionShader::markSceneStart(int sceneNum, bool isUI) {
    // Will be used in Step 2+ to track scene boundaries within a FrameBuffer
    // For now, processAsyncCompletions is called from renderMorrowindHLSL_Internal
    if (sceneNum == 0 && !isUI) {
        processAsyncCompletions();
    }
}

void FixedFunctionShader::markSceneEnd() {
    // Will be used in Step 2+ to finalize scene boundaries
}

// Triple-buffer pipeline: cull pass (called by CullThread or inline on main thread)
void FixedFunctionShader::executeCullPass(int bufferIndex) {
    // When called from CullThread, device calls are not safe (D3D9 is single-threaded).
    // When called inline from main thread (no cull thread fallback), device calls are OK.
    // CullThread::executeCull sets this to false before calling us.
    prepareRecordedCalls(bufferIndex);
}

void FixedFunctionShader::executeRenderPass(int bufferIndex) {
    // Will be called by RenderThread in Step 3
    // For now, replayRecordedCalls() is called inline from stopRecordingAndReplay()
    replayRecordedCalls(0);
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
    // Only process ACTIVE lights — lightrs->lights accumulates all SetLight() calls
    // across the session and never removes entries, so inactive/stale lights persist.
    for (DWORD id : lightrs->active) {
        auto lightIt = lightrs->lights.find(id);
        if (lightIt == lightrs->lights.end()) continue;
        const auto& light = lightIt->second;

        // Skip directional lights (sun) — only point lights for scene lighting
        if (light.type != D3DLIGHT_POINT) continue;

        // Distance filter: reject lights far from camera (stale interior lights after cell change)
        float dx = light.position.x - DistantLand::eyePos.x;
        float dy = light.position.y - DistantLand::eyePos.y;
        float dz = light.position.z - DistantLand::eyePos.z;
        float distSq = dx*dx + dy*dy + dz*dz;
        const float MAX_LIGHT_DIST_SQ = 8192.0f * 8192.0f;
        if (distSq > MAX_LIGHT_DIST_SQ) continue;

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
            sl.isVisible = false;
            sl.lastSeenFrame = 0;
            DistantLand::sceneLights.push_back(sl);
            DistantLand::sceneLightIndexMap[id] = newIndex;  // Add to index map
        } else {
            // Update dynamic properties (color may pulse)
            size_t index = mapIt->second;
            DistantLand::sceneLights[index].diffuse = light.diffuse;
        }
    }

    // Reuse last LightState if identical to avoid allocation overhead
    // NOTE: Cannot use raw pointer comparison here! lightrs is a file-scope static in
    // mged3d8device.cpp — its address never changes, but Morrowind mutates it in-place
    // between draw calls via LightEnable(). Pointer equality always returns true,
    // causing ALL calls to share the first call's light config.
    std::shared_ptr<LightState> sharedLightState;
    {
        MGE_ZoneScopedN("record_LightStateCache");
        // Content comparison: reuse if lights haven't changed since last call
        if (lastLightState && compareLightStates(lastLightState.get(), lightrs)) {
            // Reuse existing shared_ptr (no allocation)
            sharedLightState = lastLightState;
        }
        // Different lights: create new copy
        else {
            sharedLightState = std::make_shared<LightState>(*lightrs);
            lastLightState = sharedLightState;
        }
    }

    // Warm suffix cache on main thread (device calls not safe off main thread)
    warmSuffixCache(device, rs->texture);

    // Log per-bin frame event (we know the bin from sk and rs at record time)
    {
        FrameEvent::Type evType;
        if (sk.hasGrass)              evType = FrameEvent::DIP_Grass;
        else if (sk.usesSkinning)     evType = FrameEvent::DIP_Skinning;
        else if (rs->blendEnable)     evType = FrameEvent::DIP_Blending;
        else if (rs->alphaTest)       evType = FrameEvent::DIP_AlphaTested;
        else                          evType = FrameEvent::DIP_Opaque;
        ImGuiManager::LogFrameEvent(evType, 0, rs->primCount);
    }

    {
        currentRecordedCalls().emplace_back(rs, frs, sharedLightState, sk, recordMWIdx);

        // Immediately populate bboxLookup so depth pass can use current-frame bboxes
        // (prepareOcclusionCullingForDepth runs BEFORE finalizeBatchAndReplay)
        const auto& call = currentRecordedCalls().back();
        if (call.hasBoundingBox) {
            VBIBKey key{call.rs.vb, call.rs.ib};
            bboxLookup[key] = {call.bboxMin, call.bboxMax};
        }
    }

    // Occluder selection and rasterization deferred to prepareRecordedCalls()
    // This removes the heaviest per-draw work from the recording hot path
}

// ====== Full Pipeline A/B Diagnostic Snapshot System ======
// F5 captures snapshot A (saves to disk), F6 captures snapshot B (loads A from disk, diffs)
// Designed for cross-launch comparison: exterior-direct (A) vs interior→exterior (B)

#include <cstdio>

struct CallSnapshotInfo {
    // Geometry identity
    DWORD fvf;
    UINT primCount, vertCount;
    // Render state
    BYTE alphaTest, alphaFunc, alphaRef;
    BYTE blendEnable, srcBlend, destBlend;
    // Shader
    DWORD shaderKeyDword;  // first 32 bits of ShaderKey (bitfield)
    int lightMode;
    int activeStages;
    // World position (from worldTransform[0])
    float posX, posY, posZ;
    // Light state
    size_t activeCount;
    size_t pointLightCount;
    size_t lightsTransformedCount;
    // Flags
    bool hasBoundingBox;
    bool prepared;
    int8_t hiZVisible;  // 1=visible, 0=culled, -1=not tested
    // Point light details (first 3)
    struct PointLightDetail {
        DWORD id;
        float wx, wy, wz;
        float vx, vy, vz;
    };
    PointLightDetail pointLights[3];
    int numPointLightDetails;
};

struct FrameSnapshot {
    bool valid;
    int totalRecordedCalls;
    int sceneLightsTotal;
    float cameraX, cameraY, cameraZ;
    float eyePosX, eyePosY, eyePosZ;  // World position from DistantLand::eyePos
    float recordCameraX, recordCameraY, recordCameraZ;
    std::vector<CallSnapshotInfo> callInfos;  // ALL calls

    // Pipeline control-flow state (from g_pipelineDiag, filled at Present())
    PipelineDiag pipeline;

    // FFS-side recording state
    bool ffs_isRecording, ffs_isReplaying;
    bool ffs_recordingCompletedThisFrame, ffs_recordingEnabled, ffs_manualRecordingControl;
    int ffs_recordedCallCount;  // static recordedCalls.size()
    int ffs_recordingBuffer;    // which buffer index was recording
    // Per-buffer state
    struct BufferInfo {
        int callCount;
        bool valid;
        int state;  // BufferState enum as int
    };
    BufferInfo bufferInfos[3];

    FrameSnapshot() : valid(false), totalRecordedCalls(0), sceneLightsTotal(0),
        cameraX(0), cameraY(0), cameraZ(0),
        eyePosX(0), eyePosY(0), eyePosZ(0),
        recordCameraX(0), recordCameraY(0), recordCameraZ(0),
        pipeline{}, ffs_isRecording(false), ffs_isReplaying(false),
        ffs_recordingCompletedThisFrame(false), ffs_recordingEnabled(false),
        ffs_manualRecordingControl(false), ffs_recordedCallCount(0), ffs_recordingBuffer(0),
        bufferInfos{} {}

    bool saveToFile(const char* path) const {
        FILE* f = fopen(path, "w");
        if (!f) return false;

        fprintf(f, "[frame]\n");
        fprintf(f, "totalCalls=%d\n", totalRecordedCalls);
        fprintf(f, "sceneLights=%d\n", sceneLightsTotal);
        fprintf(f, "camera=%.2f,%.2f,%.2f\n", cameraX, cameraY, cameraZ);
        fprintf(f, "eyePos=%.2f,%.2f,%.2f\n", eyePosX, eyePosY, eyePosZ);
        fprintf(f, "recordCamera=%.2f,%.2f,%.2f\n", recordCameraX, recordCameraY, recordCameraZ);

        // Pipeline control-flow state
        fprintf(f, "[pipeline]\n");
        fprintf(f, "dip=%d,%d,%d,%d,%d,%d\n", pipeline.dipScene0, pipeline.dipScene1plus,
            pipeline.dipOffscreen, pipeline.dipUI, pipeline.dipStencilShadow, pipeline.dipUnknown);
        fprintf(f, "sceneCount=%d\n", pipeline.sceneCount);
        fprintf(f, "flags=%d,%d,%d,%d,%d,%d,%d,%d\n",
            pipeline.isMainView, pipeline.rendertargetNormal, pipeline.stage0Complete, pipeline.isFrameComplete,
            pipeline.isHUDComplete, pipeline.isHUDready, pipeline.isStencilScene, pipeline.isAmbientWhite);
        fprintf(f, "dl=%d,%d\n", pipeline.distantLandReady, pipeline.isPPLActive);
        fprintf(f, "viewMat=%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            pipeline.view_11, pipeline.view_12, pipeline.view_13,
            pipeline.view_41, pipeline.view_42, pipeline.view_43);
        fprintf(f, "mw=%d,%u\n", pipeline.mwLoaded, (unsigned)pipeline.mwCellAddr);

        // FFS recording state
        fprintf(f, "[ffs]\n");
        fprintf(f, "recording=%d,%d,%d,%d,%d\n",
            ffs_isRecording, ffs_isReplaying, ffs_recordingCompletedThisFrame,
            ffs_recordingEnabled, ffs_manualRecordingControl);
        fprintf(f, "recordedCalls=%d\n", ffs_recordedCallCount);
        fprintf(f, "recordingBuffer=%d\n", ffs_recordingBuffer);
        for (int i = 0; i < 3; i++) {
            fprintf(f, "buf%d=%d,%d,%d\n", i, bufferInfos[i].callCount, bufferInfos[i].valid, bufferInfos[i].state);
        }

        for (size_t i = 0; i < callInfos.size(); i++) {
            const auto& c = callInfos[i];
            fprintf(f, "[call %d]\n", (int)i);
            fprintf(f, "fvf=0x%X prim=%u vert=%u\n", c.fvf, c.primCount, c.vertCount);
            fprintf(f, "alpha=%d,%d,%d blend=%d,%d,%d\n",
                c.alphaTest, c.alphaFunc, c.alphaRef,
                c.blendEnable, c.srcBlend, c.destBlend);
            fprintf(f, "sk=0x%08X lightMode=%d stages=%d\n",
                c.shaderKeyDword, c.lightMode, c.activeStages);
            fprintf(f, "pos=%.2f,%.2f,%.2f\n", c.posX, c.posY, c.posZ);
            fprintf(f, "lights=%d pointLights=%d transformed=%d\n",
                (int)c.activeCount, (int)c.pointLightCount, (int)c.lightsTransformedCount);
            fprintf(f, "bbox=%d prepared=%d hiZ=%d\n", c.hasBoundingBox ? 1 : 0, c.prepared ? 1 : 0, (int)c.hiZVisible);
            for (int j = 0; j < c.numPointLightDetails; j++) {
                const auto& p = c.pointLights[j];
                fprintf(f, "light[%d] id=%u world=%.2f,%.2f,%.2f view=%.2f,%.2f,%.2f\n",
                    j, p.id, p.wx, p.wy, p.wz, p.vx, p.vy, p.vz);
            }
        }
        fclose(f);
        return true;
    }

    bool loadFromFile(const char* path) {
        FILE* f = fopen(path, "r");
        if (!f) return false;

        valid = false;
        callInfos.clear();
        char line[512];
        CallSnapshotInfo* currentCall = nullptr;

        enum Section { SEC_FRAME, SEC_PIPELINE, SEC_FFS, SEC_CALL } section = SEC_FRAME;

        while (fgets(line, sizeof(line), f)) {
            // Strip newline
            size_t len = strlen(line);
            while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = 0;

            if (strncmp(line, "[frame]", 7) == 0) { section = SEC_FRAME; continue; }
            if (strncmp(line, "[pipeline]", 10) == 0) { section = SEC_PIPELINE; continue; }
            if (strncmp(line, "[ffs]", 5) == 0) { section = SEC_FFS; continue; }
            if (strncmp(line, "[call ", 6) == 0) {
                section = SEC_CALL;
                callInfos.push_back(CallSnapshotInfo{});
                currentCall = &callInfos.back();
                memset(currentCall, 0, sizeof(CallSnapshotInfo));
            } else if (section == SEC_FRAME) {
                sscanf(line, "totalCalls=%d", &totalRecordedCalls);
                sscanf(line, "sceneLights=%d", &sceneLightsTotal);
                sscanf(line, "camera=%f,%f,%f", &cameraX, &cameraY, &cameraZ);
                sscanf(line, "eyePos=%f,%f,%f", &eyePosX, &eyePosY, &eyePosZ);
                sscanf(line, "recordCamera=%f,%f,%f", &recordCameraX, &recordCameraY, &recordCameraZ);
            } else if (section == SEC_PIPELINE) {
                int i0,i1,i2,i3,i4,i5,i6,i7;
                unsigned int u0;
                if (sscanf(line, "dip=%d,%d,%d,%d,%d,%d", &i0,&i1,&i2,&i3,&i4,&i5) == 6) {
                    pipeline.dipScene0=i0; pipeline.dipScene1plus=i1; pipeline.dipOffscreen=i2;
                    pipeline.dipUI=i3; pipeline.dipStencilShadow=i4; pipeline.dipUnknown=i5;
                } else if (sscanf(line, "sceneCount=%d", &pipeline.sceneCount) == 1) {
                } else if (sscanf(line, "flags=%d,%d,%d,%d,%d,%d,%d,%d", &i0,&i1,&i2,&i3,&i4,&i5,&i6,&i7) == 8) {
                    pipeline.isMainView=i0; pipeline.rendertargetNormal=i1; pipeline.stage0Complete=i2;
                    pipeline.isFrameComplete=i3; pipeline.isHUDComplete=i4; pipeline.isHUDready=i5;
                    pipeline.isStencilScene=i6; pipeline.isAmbientWhite=i7;
                } else if (sscanf(line, "dl=%d,%d", &i0,&i1) == 2) {
                    pipeline.distantLandReady=i0; pipeline.isPPLActive=i1;
                } else if (sscanf(line, "viewMat=%f,%f,%f,%f,%f,%f",
                    &pipeline.view_11, &pipeline.view_12, &pipeline.view_13,
                    &pipeline.view_41, &pipeline.view_42, &pipeline.view_43) == 6) {
                } else if (sscanf(line, "mw=%d,%u", &i0, &u0) == 2) {
                    pipeline.mwLoaded=i0; pipeline.mwCellAddr=(DWORD)u0;
                }
            } else if (section == SEC_FFS) {
                int i0,i1,i2,i3,i4;
                if (sscanf(line, "recording=%d,%d,%d,%d,%d", &i0,&i1,&i2,&i3,&i4) == 5) {
                    ffs_isRecording=i0; ffs_isReplaying=i1; ffs_recordingCompletedThisFrame=i2;
                    ffs_recordingEnabled=i3; ffs_manualRecordingControl=i4;
                } else if (sscanf(line, "recordedCalls=%d", &ffs_recordedCallCount) == 1) {
                } else if (sscanf(line, "recordingBuffer=%d", &ffs_recordingBuffer) == 1) {
                } else {
                    int bi, cc, v, s;
                    if (sscanf(line, "buf%d=%d,%d,%d", &bi, &cc, &v, &s) == 4 && bi >= 0 && bi < 3) {
                        bufferInfos[bi].callCount=cc; bufferInfos[bi].valid=v; bufferInfos[bi].state=s;
                    }
                }
            } else if (currentCall) {
                unsigned int fvfTmp;
                if (sscanf(line, "fvf=0x%X prim=%u vert=%u", &fvfTmp, &currentCall->primCount, &currentCall->vertCount) == 3) {
                    currentCall->fvf = (DWORD)fvfTmp;
                } else {
                    int at, af, ar, be, sb, db;
                    if (sscanf(line, "alpha=%d,%d,%d blend=%d,%d,%d", &at, &af, &ar, &be, &sb, &db) == 6) {
                        currentCall->alphaTest = (BYTE)at; currentCall->alphaFunc = (BYTE)af; currentCall->alphaRef = (BYTE)ar;
                        currentCall->blendEnable = (BYTE)be; currentCall->srcBlend = (BYTE)sb; currentCall->destBlend = (BYTE)db;
                    } else {
                        unsigned int skTmp;
                        if (sscanf(line, "sk=0x%X lightMode=%d stages=%d", &skTmp, &currentCall->lightMode, &currentCall->activeStages) == 3) {
                            currentCall->shaderKeyDword = (DWORD)skTmp;
                        } else if (sscanf(line, "pos=%f,%f,%f", &currentCall->posX, &currentCall->posY, &currentCall->posZ) == 3) {
                            // parsed
                        } else {
                            int lc, plc, tc;
                            if (sscanf(line, "lights=%d pointLights=%d transformed=%d", &lc, &plc, &tc) == 3) {
                                currentCall->activeCount = lc; currentCall->pointLightCount = plc; currentCall->lightsTransformedCount = tc;
                            } else {
                                int bb, pp, hz = -1;
                                if (sscanf(line, "bbox=%d prepared=%d hiZ=%d", &bb, &pp, &hz) >= 2) {
                                    currentCall->hasBoundingBox = (bb != 0); currentCall->prepared = (pp != 0);
                                    currentCall->hiZVisible = (int8_t)hz;
                                } else {
                                    int li; unsigned int lid;
                                    float lwx, lwy, lwz, lvx, lvy, lvz;
                                    if (sscanf(line, "light[%d] id=%u world=%f,%f,%f view=%f,%f,%f",
                                               &li, &lid, &lwx, &lwy, &lwz, &lvx, &lvy, &lvz) == 8) {
                                        if (currentCall->numPointLightDetails < 3) {
                                            auto& p = currentCall->pointLights[currentCall->numPointLightDetails];
                                            p.id = (DWORD)lid;
                                            p.wx = lwx; p.wy = lwy; p.wz = lwz;
                                            p.vx = lvx; p.vy = lvy; p.vz = lvz;
                                            currentCall->numPointLightDetails++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        fclose(f);
        valid = true;  // Pipeline state is valid even with 0 recorded calls (stuck state diagnostic)
        return valid;
    }

    // Structural key for position-based matching (quantized to 256-unit grid)
    struct StructuralKey {
        DWORD fvf;
        UINT primCount;
        int qx, qy;  // quantized position (nearest 256 units)

        bool operator==(const StructuralKey& other) const {
            return fvf == other.fvf && primCount == other.primCount && qx == other.qx && qy == other.qy;
        }
    };

    struct StructuralKeyHash {
        std::size_t operator()(const StructuralKey& k) const {
            std::size_t h = std::hash<DWORD>{}(k.fvf);
            h ^= std::hash<UINT>{}(k.primCount) << 1;
            h ^= std::hash<int>{}(k.qx) << 2;
            h ^= std::hash<int>{}(k.qy) << 3;
            return h;
        }
    };

    static StructuralKey makeKey(const CallSnapshotInfo& c) {
        return { c.fvf, c.primCount,
                 (int)floorf(c.posX / 256.0f + 0.5f),
                 (int)floorf(c.posY / 256.0f + 0.5f) };
    }

    // Write diff to a FILE* (used for both log and dedicated diff file)
    static void writeDiff(FILE* out, const FrameSnapshot& a, const FrameSnapshot& b) {
        fprintf(out, "=== Pipeline A/B DIFF (A=problem, B=reference) ===\n");

        // Frame-level
        if (a.totalRecordedCalls != b.totalRecordedCalls)
            fprintf(out, "DIFF totalRecordedCalls: A=%d B=%d\n", a.totalRecordedCalls, b.totalRecordedCalls);
        else
            fprintf(out, "SAME totalRecordedCalls: %d\n", a.totalRecordedCalls);

        if (a.sceneLightsTotal != b.sceneLightsTotal)
            fprintf(out, "DIFF sceneLightsTotal: A=%d B=%d\n", a.sceneLightsTotal, b.sceneLightsTotal);

        fprintf(out, "Camera A: (%.1f, %.1f, %.1f)  B: (%.1f, %.1f, %.1f)\n",
            a.cameraX, a.cameraY, a.cameraZ, b.cameraX, b.cameraY, b.cameraZ);
        fprintf(out, "EyePos A: (%.1f, %.1f, %.1f)  B: (%.1f, %.1f, %.1f)\n",
            a.eyePosX, a.eyePosY, a.eyePosZ, b.eyePosX, b.eyePosY, b.eyePosZ);
        fprintf(out, "RecordCamera A: (%.1f, %.1f, %.1f)  B: (%.1f, %.1f, %.1f)\n",
            a.recordCameraX, a.recordCameraY, a.recordCameraZ,
            b.recordCameraX, b.recordCameraY, b.recordCameraZ);

        // Pipeline control-flow state diff
        fprintf(out, "\n=== PIPELINE STATE ===\n");
        #define DIAG_DIFF_INT(field, name) \
            if (a.pipeline.field != b.pipeline.field) \
                fprintf(out, "DIFF " name ": A=%d B=%d\n", a.pipeline.field, b.pipeline.field); \
            else \
                fprintf(out, "SAME " name ": %d\n", a.pipeline.field);
        #define DIAG_DIFF_BOOL(field, name) \
            if (a.pipeline.field != b.pipeline.field) \
                fprintf(out, "DIFF " name ": A=%s B=%s\n", a.pipeline.field?"true":"false", b.pipeline.field?"true":"false"); \
            else \
                fprintf(out, "SAME " name ": %s\n", a.pipeline.field?"true":"false");

        DIAG_DIFF_INT(sceneCount, "sceneCount");
        DIAG_DIFF_BOOL(isMainView, "isMainView");
        DIAG_DIFF_BOOL(rendertargetNormal, "rendertargetNormal");
        DIAG_DIFF_BOOL(stage0Complete, "stage0Complete");
        DIAG_DIFF_BOOL(isFrameComplete, "isFrameComplete");
        DIAG_DIFF_BOOL(isHUDComplete, "isHUDComplete");
        DIAG_DIFF_BOOL(isHUDready, "isHUDready");
        DIAG_DIFF_BOOL(isStencilScene, "isStencilScene");
        DIAG_DIFF_BOOL(isAmbientWhite, "isAmbientWhite");
        DIAG_DIFF_BOOL(distantLandReady, "distantLandReady");
        DIAG_DIFF_BOOL(isPPLActive, "isPPLActive");
        DIAG_DIFF_BOOL(mwLoaded, "mwLoaded");
        DIAG_DIFF_INT(dipScene0, "dipScene0");
        DIAG_DIFF_INT(dipScene1plus, "dipScene1plus");
        DIAG_DIFF_INT(dipOffscreen, "dipOffscreen");
        DIAG_DIFF_INT(dipUI, "dipUI");
        DIAG_DIFF_INT(dipStencilShadow, "dipStencilShadow");
        DIAG_DIFF_INT(dipUnknown, "dipUnknown");

        fprintf(out, "ViewMatrix A: _11=%.4f _12=%.4f _13=%.4f _41=%.4f _42=%.4f _43=%.4f\n",
            a.pipeline.view_11, a.pipeline.view_12, a.pipeline.view_13,
            a.pipeline.view_41, a.pipeline.view_42, a.pipeline.view_43);
        fprintf(out, "ViewMatrix B: _11=%.4f _12=%.4f _13=%.4f _41=%.4f _42=%.4f _43=%.4f\n",
            b.pipeline.view_11, b.pipeline.view_12, b.pipeline.view_13,
            b.pipeline.view_41, b.pipeline.view_42, b.pipeline.view_43);
        fprintf(out, "MWCellAddr A: 0x%X  B: 0x%X\n", (unsigned)a.pipeline.mwCellAddr, (unsigned)b.pipeline.mwCellAddr);

        // FFS recording state diff
        fprintf(out, "\n=== FFS RECORDING STATE ===\n");
        #define FFS_DIFF_BOOL(field, name) \
            if (a.field != b.field) \
                fprintf(out, "DIFF " name ": A=%s B=%s\n", a.field?"true":"false", b.field?"true":"false"); \
            else \
                fprintf(out, "SAME " name ": %s\n", a.field?"true":"false");
        #define FFS_DIFF_INT(field, name) \
            if (a.field != b.field) \
                fprintf(out, "DIFF " name ": A=%d B=%d\n", a.field, b.field); \
            else \
                fprintf(out, "SAME " name ": %d\n", a.field);

        FFS_DIFF_BOOL(ffs_isRecording, "isRecording");
        FFS_DIFF_BOOL(ffs_isReplaying, "isReplaying");
        FFS_DIFF_BOOL(ffs_recordingCompletedThisFrame, "recordingCompletedThisFrame");
        FFS_DIFF_BOOL(ffs_recordingEnabled, "recordingEnabled");
        FFS_DIFF_BOOL(ffs_manualRecordingControl, "manualRecordingControl");
        FFS_DIFF_INT(ffs_recordedCallCount, "recordedCallCount");
        FFS_DIFF_INT(ffs_recordingBuffer, "recordingBuffer");

        for (int i = 0; i < 3; i++) {
            fprintf(out, "Buffer[%d] A: calls=%d valid=%d state=%d  B: calls=%d valid=%d state=%d\n",
                i, a.bufferInfos[i].callCount, a.bufferInfos[i].valid, a.bufferInfos[i].state,
                b.bufferInfos[i].callCount, b.bufferInfos[i].valid, b.bufferInfos[i].state);
        }

        #undef DIAG_DIFF_INT
        #undef DIAG_DIFF_BOOL
        #undef FFS_DIFF_BOOL
        #undef FFS_DIFF_INT

        // Build multimap from A calls by structural key
        std::unordered_multimap<StructuralKey, size_t, StructuralKeyHash> aLookup;
        for (size_t i = 0; i < a.callInfos.size(); i++) {
            aLookup.insert({makeKey(a.callInfos[i]), i});
        }

        // Track which A calls got matched
        std::vector<bool> aMatched(a.callInfos.size(), false);

        // Diff counters
        int shaderDiffs = 0, lightCountDiffs = 0, alphaDiffs = 0, blendDiffs = 0;
        int posDiffs = 0, bboxDiffs = 0, visDiffs = 0;
        // Visibility transition details for identifying false occlusion
        struct VisTransition {
            int aIdx, bIdx;
            int8_t aVis, bVis;
            DWORD fvf; UINT primCount;
            float posX, posY, posZ;
        };
        std::vector<VisTransition> visTransitions;
        int matchedCount = 0, onlyInA = 0, onlyInB = 0;

        // Grouped diff details
        std::unordered_map<DWORD, int> shaderXorHistogram;  // xor pattern -> count
        std::unordered_map<int, int> lightDeltaHistogram;   // (aCount - bCount) -> count
        // Alpha transition histogram: "A(test,func,ref) -> B(test,func,ref)" -> count
        struct AlphaTransition {
            BYTE aTest, aFunc, aRef, bTest, bFunc, bRef;
            bool operator==(const AlphaTransition& o) const {
                return aTest==o.aTest && aFunc==o.aFunc && aRef==o.aRef &&
                       bTest==o.bTest && bFunc==o.bFunc && bRef==o.bRef;
            }
        };
        struct AlphaTransitionHash {
            std::size_t operator()(const AlphaTransition& t) const {
                return std::hash<uint64_t>{}(
                    (uint64_t)t.aTest | ((uint64_t)t.aFunc<<8) | ((uint64_t)t.aRef<<16) |
                    ((uint64_t)t.bTest<<24) | ((uint64_t)t.bFunc<<32) | ((uint64_t)t.bRef<<40));
            }
        };
        std::unordered_map<AlphaTransition, int, AlphaTransitionHash> alphaHistogram;
        // Blend transition histogram
        struct BlendTransition {
            BYTE aEn, aSrc, aDst, bEn, bSrc, bDst;
            bool operator==(const BlendTransition& o) const {
                return aEn==o.aEn && aSrc==o.aSrc && aDst==o.aDst &&
                       bEn==o.bEn && bSrc==o.bSrc && bDst==o.bDst;
            }
        };
        struct BlendTransitionHash {
            std::size_t operator()(const BlendTransition& t) const {
                return std::hash<uint64_t>{}(
                    (uint64_t)t.aEn | ((uint64_t)t.aSrc<<8) | ((uint64_t)t.aDst<<16) |
                    ((uint64_t)t.bEn<<24) | ((uint64_t)t.bSrc<<32) | ((uint64_t)t.bDst<<40));
            }
        };
        std::unordered_map<BlendTransition, int, BlendTransitionHash> blendHistogram;

        // For each B call (reference), find matching A call
        for (size_t bi = 0; bi < b.callInfos.size(); bi++) {
            const auto& cb = b.callInfos[bi];
            StructuralKey bKey = makeKey(cb);

            // Find first unmatched A entry with same key
            auto range = aLookup.equal_range(bKey);
            size_t bestAi = SIZE_MAX;
            for (auto it = range.first; it != range.second; ++it) {
                if (!aMatched[it->second]) {
                    bestAi = it->second;
                    break;
                }
            }

            if (bestAi == SIZE_MAX) {
                onlyInB++;
                continue;
            }

            aMatched[bestAi] = true;
            matchedCount++;
            const auto& ca = a.callInfos[bestAi];

            // Diff matched pair
            if (ca.shaderKeyDword != cb.shaderKeyDword || ca.lightMode != cb.lightMode) {
                shaderDiffs++;
                DWORD xor_bits = ca.shaderKeyDword ^ cb.shaderKeyDword;
                shaderXorHistogram[xor_bits]++;
            }

            if (ca.activeCount != cb.activeCount || ca.pointLightCount != cb.pointLightCount) {
                lightCountDiffs++;
                int delta = (int)ca.pointLightCount - (int)cb.pointLightCount;
                lightDeltaHistogram[delta]++;
            }

            if (ca.alphaTest != cb.alphaTest || ca.alphaFunc != cb.alphaFunc || ca.alphaRef != cb.alphaRef) {
                alphaDiffs++;
                alphaHistogram[{ca.alphaTest, ca.alphaFunc, ca.alphaRef,
                                cb.alphaTest, cb.alphaFunc, cb.alphaRef}]++;
            }

            if (ca.blendEnable != cb.blendEnable || ca.srcBlend != cb.srcBlend || ca.destBlend != cb.destBlend) {
                blendDiffs++;
                blendHistogram[{ca.blendEnable, ca.srcBlend, ca.destBlend,
                                cb.blendEnable, cb.srcBlend, cb.destBlend}]++;
            }

            float dx = cb.posX - ca.posX;
            float dy = cb.posY - ca.posY;
            float dz = cb.posZ - ca.posZ;
            if (sqrtf(dx*dx + dy*dy + dz*dz) > 10.0f) {
                posDiffs++;
            }

            if (ca.hasBoundingBox != cb.hasBoundingBox) {
                bboxDiffs++;
            }

            // Visibility diff: detect false occlusion (A culled, B visible)
            if (ca.hiZVisible != cb.hiZVisible) {
                visDiffs++;
                // Track transition pattern with position for detailed output
                visTransitions.push_back({(int)bestAi, (int)bi, ca.hiZVisible, cb.hiZVisible,
                    ca.fvf, ca.primCount, ca.posX, ca.posY, ca.posZ});
            }
        }

        // Count unmatched A calls (stale calls in problem scene)
        for (size_t i = 0; i < a.callInfos.size(); i++) {
            if (!aMatched[i]) onlyInA++;
        }

        // --- Grouped output ---
        fprintf(out, "\n=== STRUCTURAL MATCHING: %d matched, %d ONLY IN A (stale), %d ONLY IN B (missing) ===\n",
            matchedCount, onlyInA, onlyInB);

        if (shaderDiffs > 0) {
            fprintf(out, "\nSHADER DIFFS: %d calls with changed shaderKey\n", shaderDiffs);
            // Known ShaderKey bit names for decoding XOR diffs
            const char* bitNames[] = {
                nullptr, nullptr, nullptr, nullptr,  // bits 0-3: uvSets
                "usesSkinning", "vertexColour", "heavyLighting", "useLighting",  // 4-7
                nullptr, nullptr,  // 8-9: lightMode
                nullptr, nullptr,  // 10-11: vertexMaterial
                nullptr, nullptr,  // 12-13: fogMode
                nullptr, nullptr, nullptr,  // 14-16: activeStages
                "usesBumpmap",  // 17
                nullptr, nullptr, nullptr,  // 18-20: bumpmapStage
                "usesTexgen", "projectiveTexgen",  // 21-22
                nullptr, nullptr, nullptr,  // 23-25: texgenStage
                "hasDiffParam", "hasParamH", "hasParamX",  // 26-28
                "hasShadows", "hasGrass", "hasDetail"  // 29-31
            };
            for (const auto& [xorBits, count] : shaderXorHistogram) {
                fprintf(out, "  xor=0x%08X: %d calls", xorBits, count);
                std::string decoded;
                for (int bit = 0; bit < 32; bit++) {
                    if ((xorBits & (1u << bit)) && bitNames[bit]) {
                        if (!decoded.empty()) decoded += "+";
                        decoded += bitNames[bit];
                    }
                }
                if (!decoded.empty()) {
                    fprintf(out, " [%s]", decoded.c_str());
                }
                fprintf(out, "\n");
            }
        }

        if (lightCountDiffs > 0) {
            fprintf(out, "\nLIGHT COUNT DIFFS: %d calls\n", lightCountDiffs);
            for (const auto& [delta, count] : lightDeltaHistogram) {
                fprintf(out, "  A has %+d more point lights: %d calls\n", delta, count);
            }
        }

        if (alphaDiffs > 0) {
            fprintf(out, "\nALPHA DIFFS: %d calls\n", alphaDiffs);
            for (const auto& [t, count] : alphaHistogram) {
                fprintf(out, "  A(test=%d func=%d ref=%d) -> B(test=%d func=%d ref=%d): %d calls\n",
                    t.aTest, t.aFunc, t.aRef, t.bTest, t.bFunc, t.bRef, count);
            }
        }
        if (blendDiffs > 0) {
            fprintf(out, "\nBLEND DIFFS: %d calls\n", blendDiffs);
            for (const auto& [t, count] : blendHistogram) {
                fprintf(out, "  A(en=%d src=%d dst=%d) -> B(en=%d src=%d dst=%d): %d calls\n",
                    t.aEn, t.aSrc, t.aDst, t.bEn, t.bSrc, t.bDst, count);
            }
        }
        if (posDiffs > 0) fprintf(out, "\nPOSITION DIFFS: %d calls moved >10 units\n", posDiffs);
        if (bboxDiffs > 0) fprintf(out, "\nBBOX DIFFS: %d calls\n", bboxDiffs);

        if (visDiffs > 0) {
            fprintf(out, "\nVISIBILITY DIFFS: %d calls with different Hi-Z culling\n", visDiffs);
            // Group by transition type
            int falseOcclusion = 0, falseMiss = 0, other = 0;
            for (const auto& v : visTransitions) {
                if (v.aVis == 0 && v.bVis == 1) falseOcclusion++;
                else if (v.aVis == 1 && v.bVis == 0) falseMiss++;
                else other++;
            }
            if (falseOcclusion > 0)
                fprintf(out, "  A=CULLED B=VISIBLE (FALSE OCCLUSION): %d calls\n", falseOcclusion);
            if (falseMiss > 0)
                fprintf(out, "  A=VISIBLE B=CULLED: %d calls\n", falseMiss);
            if (other > 0)
                fprintf(out, "  Other transitions: %d calls\n", other);

            // List false occlusion calls with details (the most important ones)
            if (falseOcclusion > 0) {
                fprintf(out, "  --- False occlusion details (A culled, B visible) ---\n");
                int printed = 0;
                for (const auto& v : visTransitions) {
                    if (v.aVis == 0 && v.bVis == 1 && printed < 30) {
                        fprintf(out, "    A[%d]/B[%d]: fvf=0x%X prim=%u pos=(%.1f,%.1f,%.1f)\n",
                            v.aIdx, v.bIdx, v.fvf, v.primCount, v.posX, v.posY, v.posZ);
                        printed++;
                    }
                }
                if (falseOcclusion > 30) fprintf(out, "    ... and %d more\n", falseOcclusion - 30);
            }
        }

        // List calls ONLY IN A (stale — present in problem scene, absent from reference)
        if (onlyInA > 0) {
            fprintf(out, "\n--- ONLY IN A (stale calls in problem scene): %d ---\n", onlyInA);
            int printed = 0;
            for (size_t i = 0; i < a.callInfos.size() && printed < 50; i++) {
                if (!aMatched[i]) {
                    const auto& c = a.callInfos[i];
                    fprintf(out, "  A[%d]: fvf=0x%X prim=%u pos=(%.1f,%.1f,%.1f) sk=0x%08X lm=%d lights=%d\n",
                        (int)i, c.fvf, c.primCount, c.posX, c.posY, c.posZ,
                        c.shaderKeyDword, c.lightMode, (int)c.pointLightCount);
                    printed++;
                }
            }
            if (onlyInA > 50) fprintf(out, "  ... and %d more\n", onlyInA - 50);
        }

        // List calls ONLY IN B (missing from problem scene — present in fresh reference)
        if (onlyInB > 0) {
            fprintf(out, "\n--- ONLY IN B (missing from problem scene): %d ---\n", onlyInB);
            // Re-scan to find unmatched B calls
            // Rebuild: mark B calls that found matches
            std::vector<bool> bMatched(b.callInfos.size(), false);
            // Reset A matched for re-matching
            std::fill(aMatched.begin(), aMatched.end(), false);
            for (size_t bi = 0; bi < b.callInfos.size(); bi++) {
                StructuralKey bKey = makeKey(b.callInfos[bi]);
                auto range = aLookup.equal_range(bKey);
                for (auto it = range.first; it != range.second; ++it) {
                    if (!aMatched[it->second]) {
                        aMatched[it->second] = true;
                        bMatched[bi] = true;
                        break;
                    }
                }
            }
            int printed = 0;
            for (size_t i = 0; i < b.callInfos.size() && printed < 50; i++) {
                if (!bMatched[i]) {
                    const auto& c = b.callInfos[i];
                    fprintf(out, "  B[%d]: fvf=0x%X prim=%u pos=(%.1f,%.1f,%.1f) sk=0x%08X lm=%d lights=%d\n",
                        (int)i, c.fvf, c.primCount, c.posX, c.posY, c.posZ,
                        c.shaderKeyDword, c.lightMode, (int)c.pointLightCount);
                    printed++;
                }
            }
            if (onlyInB > 50) fprintf(out, "  ... and %d more\n", onlyInB - 50);
        }

        fprintf(out, "\n=== SUMMARY ===\n");
        fprintf(out, "Calls: A=%d B=%d | Matched=%d | ONLY IN A (stale)=%d | ONLY IN B (missing)=%d\n",
            (int)a.callInfos.size(), (int)b.callInfos.size(), matchedCount, onlyInA, onlyInB);
        fprintf(out, "Diffs in matched: %d shader, %d lightCount, %d alpha, %d blend, %d position, %d bbox, %d visibility\n",
            shaderDiffs, lightCountDiffs, alphaDiffs, blendDiffs, posDiffs, bboxDiffs, visDiffs);
        fprintf(out, "=== END DIFF ===\n");
    }

    static void logDiff(const FrameSnapshot& a, const FrameSnapshot& b) {
        if (!a.valid || !b.valid) return;

        // Write to mgeXE.log via LOG
        LOG::logline("=== Pipeline A/B DIFF (summary) ===");
        LOG::logline("  totalCalls: A=%d B=%d", a.totalRecordedCalls, b.totalRecordedCalls);
        LOG::logline("  sceneLights: A=%d B=%d", a.sceneLightsTotal, b.sceneLightsTotal);
        LOG::logline("  eyePos A=(%.1f,%.1f,%.1f) B=(%.1f,%.1f,%.1f)",
            a.eyePosX, a.eyePosY, a.eyePosZ, b.eyePosX, b.eyePosY, b.eyePosZ);
        LOG::logline("  sceneCount: A=%d B=%d | isMainView: A=%d B=%d | dipScene0: A=%d B=%d | dipUI: A=%d B=%d",
            a.pipeline.sceneCount, b.pipeline.sceneCount,
            a.pipeline.isMainView, b.pipeline.isMainView,
            a.pipeline.dipScene0, b.pipeline.dipScene0,
            a.pipeline.dipUI, b.pipeline.dipUI);
        LOG::logline("  recordedCallCount: A=%d B=%d | recordingEnabled: A=%d B=%d",
            a.ffs_recordedCallCount, b.ffs_recordedCallCount,
            a.ffs_recordingEnabled, b.ffs_recordingEnabled);
        LOG::logline("  Full diff written to mge_snapshot_diff.log");

        // Write detailed diff to dedicated file
        FILE* diffFile = fopen("mge_snapshot_diff.log", "w");
        if (diffFile) {
            writeDiff(diffFile, a, b);
            fclose(diffFile);
        }
    }
};

static FrameSnapshot snapshotA;
static FrameSnapshot snapshotB;

// Helper: build CallSnapshotInfo vector from a FrameBuffer's recorded calls
static std::vector<CallSnapshotInfo> buildCallSnapshotsFromBuffer(const std::vector<FixedFunctionShader::HLSLRecordedCall>& calls) {
    std::vector<CallSnapshotInfo> infos;
    infos.reserve(calls.size());
    for (size_t i = 0; i < calls.size(); i++) {
        const auto& call = calls[i];
        CallSnapshotInfo info;
        memset(&info, 0, sizeof(info));

        info.fvf = call.rs.fvf;
        info.primCount = call.rs.primCount;
        info.vertCount = call.rs.vertCount;
        info.alphaTest = call.rs.alphaTest;
        info.alphaFunc = call.rs.alphaFunc;
        info.alphaRef = call.rs.alphaRef;
        info.blendEnable = call.rs.blendEnable;
        info.srcBlend = call.rs.srcBlend;
        info.destBlend = call.rs.destBlend;
        info.shaderKeyDword = *(const DWORD*)&call.sk;
        info.lightMode = call.sk.lightMode;
        info.activeStages = call.sk.activeStages;
        info.posX = call.rs.worldTransforms[0]._41;
        info.posY = call.rs.worldTransforms[0]._42;
        info.posZ = call.rs.worldTransforms[0]._43;
        info.activeCount = call.lightrs ? call.lightrs->active.size() : 0;
        info.lightsTransformedCount = call.lightrs ? call.lightrs->lightsTransformed.size() : 0;
        info.hasBoundingBox = call.hasBoundingBox;
        info.prepared = call.prepared;
        info.hiZVisible = -1;  // Not available at Present() time
        info.pointLightCount = 0;
        info.numPointLightDetails = 0;
        if (call.lightrs) {
            for (DWORD id : call.lightrs->active) {
                auto it = call.lightrs->lights.find(id);
                if (it != call.lightrs->lights.end() && it->second.type == D3DLIGHT_POINT) {
                    info.pointLightCount++;
                    if (info.numPointLightDetails < 3) {
                        auto& d = info.pointLights[info.numPointLightDetails];
                        d.id = id;
                        d.wx = it->second.position.x;
                        d.wy = it->second.position.y;
                        d.wz = it->second.position.z;
                        d.vx = it->second.viewspacePos.x;
                        d.vy = it->second.viewspacePos.y;
                        d.vz = it->second.viewspacePos.z;
                        info.numPointLightDetails++;
                    }
                }
            }
        }
        infos.push_back(info);
    }
    return infos;
}

// Fill common snapshot fields from current state
static void fillSnapshotState(FrameSnapshot& snap) {
    // Pipeline state from g_pipelineDiag (filled at top of Present())
    snap.pipeline = g_pipelineDiag;

    // FFS internal state
    snap.ffs_isRecording = FixedFunctionShader::getIsRecording();
    snap.ffs_isReplaying = FixedFunctionShader::getIsReplaying();
    snap.ffs_recordingCompletedThisFrame = false;  // Already reset by this point
    snap.ffs_recordingEnabled = FixedFunctionShader::getRecordingEnabled();
    snap.ffs_manualRecordingControl = FixedFunctionShader::getManualRecordingControl();
    snap.ffs_recordedCallCount = (int)FixedFunctionShader::getRecordedCallsCount();
    snap.ffs_recordingBuffer = FixedFunctionShader::getRecordingBufferIndex();

    // Per-buffer state
    for (int i = 0; i < 3; i++) {
        auto& fb = FixedFunctionShader::getFrameBuffer(i);
        snap.bufferInfos[i].callCount = (int)fb.recordedCalls.size();
        snap.bufferInfos[i].valid = fb.valid;
        snap.bufferInfos[i].state = (int)fb.state;
    }

    // Camera and eye position
    snap.eyePosX = DistantLand::eyePos.x;
    snap.eyePosY = DistantLand::eyePos.y;
    snap.eyePosZ = DistantLand::eyePos.z;
    snap.sceneLightsTotal = (int)DistantLand::sceneLights.size();

    // Read recorded calls from the recording buffer (already filled before rotation)
    auto& fb = FixedFunctionShader::getFrameBuffer(snap.ffs_recordingBuffer);
    snap.totalRecordedCalls = (int)fb.recordedCalls.size();
    snap.callInfos = buildCallSnapshotsFromBuffer(fb.recordedCalls);

    // Camera from the buffer's view matrix
    snap.cameraX = fb.view._41; snap.cameraY = fb.view._42; snap.cameraZ = fb.view._43;
    snap.recordCameraX = fb.view._41; snap.recordCameraY = fb.view._42; snap.recordCameraZ = fb.view._43;

    snap.valid = true;
}

void FixedFunctionShader::checkSnapshotHotkeys() {
    if (!ImGuiManager::GetDebugKeysEnabled()) return;

    // F5: Capture snapshot A and save to disk
    static bool f5WasPressed = false;
    bool f5State = (GetAsyncKeyState(VK_F5) & 0x8000) != 0;
    if (f5State && !f5WasPressed) {
        fillSnapshotState(snapshotA);

        if (snapshotA.saveToFile("mge_snapshot_A.log")) {
            LOG::logline("Snapshot A saved to mge_snapshot_A.log (%d calls, sceneCount=%d, isMainView=%d, dipScene0=%d)",
                snapshotA.totalRecordedCalls, snapshotA.pipeline.sceneCount,
                snapshotA.pipeline.isMainView, snapshotA.pipeline.dipScene0);
            StatusOverlay::setStatus("Snapshot A saved to mge_snapshot_A.log");
        } else {
            LOG::logline("!! Failed to write mge_snapshot_A.log");
            StatusOverlay::setStatus("Snapshot A capture failed (file write error)");
        }
    }
    f5WasPressed = f5State;

    // F6: Capture snapshot B, load A from disk, diff
    static bool f6WasPressed = false;
    bool f6State = (GetAsyncKeyState(VK_F6) & 0x8000) != 0;
    if (f6State && !f6WasPressed) {
        fillSnapshotState(snapshotB);

        LOG::logline("Snapshot B captured (%d calls, sceneCount=%d, isMainView=%d, dipScene0=%d)",
            snapshotB.totalRecordedCalls, snapshotB.pipeline.sceneCount,
            snapshotB.pipeline.isMainView, snapshotB.pipeline.dipScene0);

        // Load A from disk (supports cross-launch comparison)
        FrameSnapshot loadedA;
        if (loadedA.loadFromFile("mge_snapshot_A.log")) {
            LOG::logline("Loaded snapshot A from disk (%d calls)", loadedA.totalRecordedCalls);
            FrameSnapshot::logDiff(loadedA, snapshotB);
            StatusOverlay::setStatus("Snapshot B captured + diff written to mge_snapshot_diff.log");
        } else {
            LOG::logline(">> No mge_snapshot_A.log found. Press F5 first at good state, then relaunch.");
            StatusOverlay::setStatus("No snapshot A on disk. Press F5 first.");
        }
    }
    f6WasPressed = f6State;
}

void FixedFunctionShader::replayRecordedCalls(int sceneCount) {
    auto& recCalls = currentRecordedCalls();
    {
        if (recCalls.empty()) {
            return;
        }

        ImGuiManager::LogFrameEvent(FrameEvent::MGE_HLSLReplay, sceneCount, (int)recCalls.size());

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
        MGE_ZoneScopedN("replay_GetTransforms");
        device->GetTransform(D3DTS_VIEW, &currentView);
        device->GetTransform(D3DTS_PROJECTION, &currentProj);
    }

    // Temporarily set shadow matrices to recording state
    auto& fb = frameBuffers[recordingBuffer];
    savedShadowViewproj[0] = DistantLand::smViewproj[0];
    savedShadowViewproj[1] = DistantLand::smViewproj[1];
    DistantLand::smViewproj[0] = fb.shadowViewproj[0];
    DistantLand::smViewproj[1] = fb.shadowViewproj[1];

    // Calculate view-projection matrix for Hi-Z culling
    viewProj = currentView * currentProj;

    // Hi-Z culling statistics
    int totalCalls = recCalls.size();
    int culledCalls = 0;
    int callsWithBBox = 0;
    int callsWithoutBBox = 0;
    int cacheHits = 0;    // Visibility reused from depth pass
    int cacheMisses = 0;  // New Hi-Z tests (alpha objects)

    // Check for debug key press (Y key) - gated behind debug hotkeys toggle
    static bool debugHiZ = false;
    static int debugCallCount = 0;
    if (ImGuiManager::GetDebugKeysEnabled() && (GetAsyncKeyState('Y') & 0x8000)) {
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

    // Check for Hi-Z snapshot save key press (L key) - gated behind debug hotkeys toggle
    if (ImGuiManager::GetDebugKeysEnabled() && (GetAsyncKeyState('L') & 0x8000)) {
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

    // Per-object light packing for mode 3 (saturated Morrowind assignment)
    // Each mode 3 object gets only its spatially-nearby lights packed into the texture.
    int numSceneLights = (int)DistantLand::sceneLights.size();
    {
        MGE_ZoneScopedN("replay_PerObjectLightPack");
        const size_t numCallsForPack = recCalls.size();
        perObjectLightInfo.resize(numCallsForPack);
        memset(perObjectLightInfo.data(), 0, numCallsForPack * sizeof(PerObjectLightInfo));

        // Flat buffer: 12 floats per light (3 texels × 4 floats), packed contiguously
        std::vector<float> packedLightData;
        int currentTexelOffset = 0;
        int mode3Count = 0;
        int maxPerObjectLights = 0;

        for (size_t i = 0; i < numCallsForPack; i++) {
            if (recCalls[i].sk.lightMode != 3) continue;

            const auto& call = recCalls[i];

            // Bounding box for sphere-AABB intersection test
            // For large meshes, lights can be inside the bbox but far from center
            D3DXVECTOR3 bMin, bMax;
            if (call.hasBoundingBox) {
                bMin = call.bboxMin;
                bMax = call.bboxMax;
            } else {
                // No bbox: use object origin as a point
                float ox = call.rs.worldTransforms[0]._41;
                float oy = call.rs.worldTransforms[0]._42;
                float oz = call.rs.worldTransforms[0]._43;
                bMin = bMax = D3DXVECTOR3(ox, oy, oz);
            }

            int count = 0;

            for (const auto& light : DistantLand::sceneLights) {
                // Sphere-AABB intersection: closest point on bbox to light center
                float cx = (light.position.x < bMin.x) ? bMin.x : (light.position.x > bMax.x) ? bMax.x : light.position.x;
                float cy = (light.position.y < bMin.y) ? bMin.y : (light.position.y > bMax.y) ? bMax.y : light.position.y;
                float cz = (light.position.z < bMin.z) ? bMin.z : (light.position.z > bMax.z) ? bMax.z : light.position.z;
                float dx = light.position.x - cx;
                float dy = light.position.y - cy;
                float dz = light.position.z - cz;
                float dist2 = dx*dx + dy*dy + dz*dz;
                float lightRadius = light.radius;
                if (dist2 < lightRadius * lightRadius) {
                    // Transform to view-space and pack 12 floats (3 texels)
                    D3DXVECTOR4 worldPos4(light.position.x, light.position.y, light.position.z, 1.0f);
                    D3DXVECTOR4 viewPos4;
                    D3DXVec4Transform(&viewPos4, &worldPos4, &currentView);

                    // Texel 0: view-space position + radius
                    packedLightData.push_back(viewPos4.x);
                    packedLightData.push_back(viewPos4.y);
                    packedLightData.push_back(viewPos4.z);
                    packedLightData.push_back(light.radius);

                    // Texel 1: color
                    packedLightData.push_back(light.diffuse.r);
                    packedLightData.push_back(light.diffuse.g);
                    packedLightData.push_back(light.diffuse.b);
                    packedLightData.push_back(0.0f);

                    // Texel 2: falloff parameters
                    packedLightData.push_back(light.falloff.x);  // constant
                    packedLightData.push_back(light.falloff.y);  // linear
                    packedLightData.push_back(light.falloff.z);  // quadratic
                    packedLightData.push_back(0.0f);

                    count++;
                }
            }

            perObjectLightInfo[i] = {currentTexelOffset, count};
            currentTexelOffset += count * 3;  // 3 texels per light

            // Diagnostic: log first mode 3 object's bbox and nearest light (sphere-AABB distance)
            if (mode3Count == 0 && !DistantLand::sceneLights.empty()) {
                float nearestDist = FLT_MAX;
                int nearestIdx = -1;
                float nearestRadius = 0;
                for (int li = 0; li < (int)DistantLand::sceneLights.size(); li++) {
                    const auto& light = DistantLand::sceneLights[li];
                    float cx = (light.position.x < bMin.x) ? bMin.x : (light.position.x > bMax.x) ? bMax.x : light.position.x;
                    float cy = (light.position.y < bMin.y) ? bMin.y : (light.position.y > bMax.y) ? bMax.y : light.position.y;
                    float cz = (light.position.z < bMin.z) ? bMin.z : (light.position.z > bMax.z) ? bMax.z : light.position.z;
                    float dx = light.position.x - cx, dy = light.position.y - cy, dz = light.position.z - cz;
                    float d = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (d < nearestDist) { nearestDist = d; nearestIdx = li; nearestRadius = light.radius; }
                }
                LOG::logline("Mode3 diag: obj[%d] bbox=(%.0f,%.0f,%.0f)-(%.0f,%.0f,%.0f) nearest light[%d] dist=%.1f radius=%.1f found=%d",
                    (int)i, bMin.x, bMin.y, bMin.z, bMax.x, bMax.y, bMax.z,
                    nearestIdx, nearestDist, nearestRadius, count);
            }

            mode3Count++;
            if (count > maxPerObjectLights) maxPerObjectLights = count;
        }

        int totalTexels = currentTexelOffset;
        perObjectTexelSize = totalTexels > 0 ? 1.0f / totalTexels : 0.0f;

        // Upload packed light data to per-object texture
        if (totalTexels > 0) {
            // Create or resize texture if needed
            if (!texPerObjectLightData) {
                HRESULT hr = device->CreateTexture(
                    totalTexels, 1, 1, 0,
                    D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED,
                    &texPerObjectLightData, nullptr);
                if (FAILED(hr)) {
                    LOG::logline("!! Failed to create per-object light texture (hr=0x%X)", hr);
                    texPerObjectLightData = nullptr;
                }
            } else {
                D3DSURFACE_DESC desc;
                texPerObjectLightData->GetLevelDesc(0, &desc);
                if (desc.Width != (UINT)totalTexels) {
                    texPerObjectLightData->Release();
                    HRESULT hr = device->CreateTexture(
                        totalTexels, 1, 1, 0,
                        D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED,
                        &texPerObjectLightData, nullptr);
                    if (FAILED(hr)) {
                        LOG::logline("!! Failed to resize per-object light texture (hr=0x%X)", hr);
                        texPerObjectLightData = nullptr;
                    }
                }
            }

            if (texPerObjectLightData) {
                D3DLOCKED_RECT locked;
                if (SUCCEEDED(texPerObjectLightData->LockRect(0, &locked, nullptr, 0))) {
                    memcpy(locked.pBits, packedLightData.data(), totalTexels * 4 * sizeof(float));
                    texPerObjectLightData->UnlockRect(0);
                }
                device->SetTexture(5, texPerObjectLightData);
            }
        }

        if (mode3Count > 0) {
            LOG::logline("Mode3 packing: %d objects, maxLights/obj=%d, totalTexels=%d (from %d scene)",
                mode3Count, maxPerObjectLights, totalTexels, numSceneLights);
        }
    }

    // For mode 0-2 objects, also upload global light data for non-mode-3 texture access
    // Mode 2 uses uniform constants (not texture), mode 3 uses per-object texture via c50
    // Set default c50 for non-mode-3 objects (will be overridden per-draw for mode 3)
    {
        float lightParams[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        device->SetPixelShaderConstantF(50, lightParams, 1);
    }

    // Inline Hi-Z culling using current matrices (same as bbox visualization)
    const size_t numCalls = recCalls.size();

    // Bin statistics
    int binCounts[(int)RenderBin::Count] = {};
    static const char* binNames[] = { "Terrain", "Opaque", "Skinning", "Grass", "AlphaTested", "Blending" };

#ifdef TRACY_ENABLE
    // Tracy bin zone: manually managed ScopedZone for per-bin profiling regions
    static constexpr tracy::SourceLocationData binZoneSrcLoc { "RenderBin", TracyFunction, TracyFile, (uint32_t)__LINE__, 0 };
    tracy::ScopedZone* binZone = nullptr;
    alignas(tracy::ScopedZone) char binZoneBuf[sizeof(tracy::ScopedZone)];
    RenderBin currentBin = RenderBin::Count;
#endif

    // Slow frame detection: QPC timing for replay loop
    LARGE_INTEGER replayFreqQPC, replayStartQPC, replayEndQPC;
    QueryPerformanceFrequency(&replayFreqQPC);
    QueryPerformanceCounter(&replayStartQPC);
    float worstCallMs = 0.0f;
    int worstCallIndex = -1;
    int worstCallPrims = 0;
    int worstCallBin = 0;
    float slowCallThreshold = ImGuiManager::GetSlowCallThreshold();

    MGE_ZoneScopedN("replay_MainLoop");
    bool firstDrawDone = false;
    for (size_t i = 0; i < numCalls; i++) {
        auto& call = recCalls[i];  // Non-const to update shader key

        // Track bin statistics
        binCounts[(int)call.bin]++;

        // Per-bin suppress check
        switch (call.bin) {
            case RenderBin::Terrain:    if (ImGuiManager::GetSuppressTerrain()) continue; break;
            case RenderBin::Opaque:     if (ImGuiManager::GetSuppressOpaque()) continue; break;
            case RenderBin::Skinning:   if (ImGuiManager::GetSuppressSkinning()) continue; break;
            case RenderBin::Grass:      if (ImGuiManager::GetSuppressGrass()) continue; break;
            case RenderBin::AlphaTested:if (ImGuiManager::GetSuppressAlphaTested()) continue; break;
            case RenderBin::Blending:   if (ImGuiManager::GetSuppressBlending()) continue; break;
            default: break;
        }

#ifdef TRACY_ENABLE
        // Emit Tracy zone on bin transition
        if (g_tracyActive && call.bin != currentBin) {
            if (binZone) binZone->~ScopedZone();
            currentBin = call.bin;
            binZone = new (binZoneBuf) tracy::ScopedZone(&binZoneSrcLoc, TRACY_CALLSTACK, true);
            const char* name = binNames[(int)currentBin];
            binZone->Name(name, strlen(name));
        }
#endif

        // Build MeshKey for this call
        MeshKey currentKey{call.rs.vb, call.rs.ib, call.rs.fvf, call.rs.baseIndex,
                          call.rs.vertCount, call.rs.startIndex, call.rs.primCount};

        // Visibility culling: reuse depth pass results when available
        bool shouldRender = true;
        bool hiZDisabled = ImGuiManager::GetDisableHiZCulling();
        if (!hiZDisabled && call.hasBoundingBox) {
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
        } else if (!hiZDisabled) {
            callsWithoutBBox++;
        }

        if (shouldRender) {
            // Restore sampler states for this call (captured during recording)
            for (int stage = 0; stage < 8; ++stage) {
                if (call.samplerStates[stage].captured) {
                    device->SetSamplerState(stage, D3DSAMP_ADDRESSU, call.samplerStates[stage].addressU);
                    device->SetSamplerState(stage, D3DSAMP_ADDRESSV, call.samplerStates[stage].addressV);
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

            // Render bin highlighting: tint by bin category
            if (ImGuiManager::GetHighlightBins() && call.bin != RenderBin::Opaque) {
                FragmentState tintedFrs = call.frs;
                switch (call.bin) {
                    case RenderBin::Skinning:
                        tintedFrs.material.emissive = {0.0f, 0.0f, 1.0f, 1.0f}; break; // Blue
                    case RenderBin::Grass:
                        tintedFrs.material.emissive = {0.0f, 1.0f, 0.0f, 1.0f}; break; // Green
                    case RenderBin::AlphaTested:
                        tintedFrs.material.emissive = {1.0f, 1.0f, 0.0f, 1.0f}; break; // Yellow
                    case RenderBin::Blending:
                        tintedFrs.material.emissive = {1.0f, 0.0f, 1.0f, 1.0f}; break; // Magenta
                    default: break;
                }
                renderMorrowindHLSL_Internal(&call.rs, &tintedFrs, call.lightrs.get(), DIRTY_ALL);
                continue;
            }

            {
                MGE_ZoneScopedN("replay_RenderCall");
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
                LARGE_INTEGER callStartQPC, callEndQPC;
                QueryPerformanceCounter(&callStartQPC);
                renderMorrowindHLSL_Internal(&call.rs, &call.frs, call.lightrs.get(), call.dirtyFlags, (int)i);
                QueryPerformanceCounter(&callEndQPC);
                float callMs = (callEndQPC.QuadPart - callStartQPC.QuadPart) * 1000.0f / replayFreqQPC.QuadPart;
                if (callMs > worstCallMs) {
                    worstCallMs = callMs;
                    worstCallIndex = (int)i;
                    worstCallPrims = call.rs.primCount;
                    worstCallBin = (int)call.bin;
                }
                if (!firstDrawDone) {
                    MGE_ZoneScopedN("replay_FirstDrawDone");
                    firstDrawDone = true;
                }
            }
        }
    }

#ifdef TRACY_ENABLE
    // Close final bin zone
    if (binZone) binZone->~ScopedZone();
#endif

    // Slow frame detection: measure total replay time
    QueryPerformanceCounter(&replayEndQPC);
    float replayMs = (replayEndQPC.QuadPart - replayStartQPC.QuadPart) * 1000.0f / replayFreqQPC.QuadPart;
    float slowFrameThreshold = ImGuiManager::GetSlowFrameThreshold();
    if (replayMs > slowFrameThreshold) {
        LOG::logline("SLOW REPLAY: %.1fms for %d calls", replayMs, (int)numCalls);
    }

    // Auto-freeze on slow frame (prepare timing comes from prepareRecordedCalls via stored value)
    if (ImGuiManager::GetSlowFrameAutoFreeze() && !ImGuiManager::GetSlowFrameFrozen()) {
        if (lastPrepareMs > slowFrameThreshold || replayMs > slowFrameThreshold) {
            ImGuiManager::FreezeSlowFrame(lastPrepareMs, replayMs, worstCallIndex, worstCallMs, worstCallPrims, worstCallBin);
        }
    }

    // Log culling statistics
    int renderedCalls = totalCalls - culledCalls;
    LOG::logline("Hi-Z Stats: %d total, %d bbox, %d culled (%.1f%%), cache: %d hits %d misses",
                 totalCalls, callsWithBBox, culledCalls,
                 totalCalls > 0 ? (culledCalls * 100.0f) / totalCalls : 0.0f,
                 cacheHits, cacheMisses);

    // Log bin statistics
    LOG::logline("Bins: Terrain=%d Opaque=%d Skinning=%d Grass=%d AlphaTested=%d Blending=%d",
                 binCounts[(int)RenderBin::Terrain],
                 binCounts[(int)RenderBin::Opaque], binCounts[(int)RenderBin::Skinning],
                 binCounts[(int)RenderBin::Grass], binCounts[(int)RenderBin::AlphaTested],
                 binCounts[(int)RenderBin::Blending]);

    // Feed per-bin counts back to ImGui DIP stats (these replace the coarse Scene0 count)
    ImGuiManager::UpdateReplayBinCounts(
        binCounts[(int)RenderBin::Terrain],
        binCounts[(int)RenderBin::Opaque],
        binCounts[(int)RenderBin::Skinning],
        binCounts[(int)RenderBin::Grass],
        binCounts[(int)RenderBin::AlphaTested],
        binCounts[(int)RenderBin::Blending]);

    // Per-frame light summary: scan all calls for point light statistics and mode distribution
    {
        int minPL = INT_MAX, maxPL = 0;
        double sumPL = 0;
        int litCalls = 0;
        int modeCounts[4] = {0, 0, 0, 0};
        for (const auto& call : recCalls) {
            if (call.sk.lightMode < 4) modeCounts[call.sk.lightMode]++;
            if (!call.lightrs) continue;
            int pl = 0;
            for (DWORD id : call.lightrs->active) {
                auto it = call.lightrs->lights.find(id);
                if (it != call.lightrs->lights.end() && it->second.type == D3DLIGHT_POINT) pl++;
            }
            if (pl < minPL) minPL = pl;
            if (pl > maxPL) maxPL = pl;
            sumPL += pl;
            litCalls++;
        }
        if (litCalls == 0) minPL = 0;
        LOG::logline("Lights: %d scene, perObj: min=%d max=%d avg=%.1f (%d calls) modes:[%d,%d,%d,%d]",
            (int)DistantLand::sceneLights.size(), minPL, maxPL,
            litCalls > 0 ? sumPL / litCalls : 0.0, litCalls,
            modeCounts[0], modeCounts[1], modeCounts[2], modeCounts[3]);
    }

    // Update ImGui debug stats
    ImGuiManager::UpdateDebugStats(
        totalCalls, renderedCalls, culledCalls,
        (int)DistantLand::sceneLights.size(),
        (int)DistantLand::recordMW.size(), 0
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
            for (size_t i = 0; i < recCalls.size(); i++) {
                const auto& call = recCalls[i];
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
        for (size_t i = 0; i < recCalls.size(); i++) {
            const auto& call = recCalls[i];

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

    // Data is already in frameBuffers[recordingBuffer].recordedCalls (recorded directly there)
    frameBuffers[recordingBuffer].state = BufferState::Available;

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
    : rs(*rs_), frs(*frs_), lightrs(lightrs_), sk(sk_), hasBoundingBox(false), recordMWIndex(recordMWIdx), bin(RenderBin::Opaque), prepared(false), dirtyFlags(DIRTY_ALL) {
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

