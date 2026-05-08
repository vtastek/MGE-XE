
#include "ffeshader.h"
#include "texture_suffix.h"
#include "cullthread.h"
#include "renderthread.h"
#include "d3dcommandbuffer.h"
#include "mge_tracy.h"
#include "configuration.h"
#include "support/log.h"
#include "mwbridge.h"
#include "morrowindbsa.h"
#include "statusoverlay.h"
#include "distantland.h"
#include "distantlandhlsl.h"
#include "imgui_manager.h"
#include "hlsl_shader_manager.h"
#include "shader_utils.h"
#include "patch_displacement.h"
#include "mged3d8device.h"

#include <algorithm>
#include <sstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>
#include <climits>
#include <unordered_map>
#include <random>
#include <vector>

using std::string;
using std::stringstream;
using std::unordered_map;

// Per-stage command buffer set (defined in mged3d8device.cpp)
extern D3DCommandBufferSet g_cmdBufferSet;

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
D3DXHANDLE FixedFunctionShader::ehDebugMode;

float FixedFunctionShader::sunMultiplier, FixedFunctionShader::ambMultiplier;

// During replay, points to fb.shadowViewproj (recording-time matrices).
// Outside replay, nullptr — callers fall back to s_staging.smViewproj.

// HLSL Pipeline static variables
unordered_map<FixedFunctionShader::ShaderKey, FixedFunctionShader::HLSLShader, FixedFunctionShader::ShaderKey::hasher> FixedFunctionShader::cacheHLSLShaders;
FixedFunctionShader::HLSLShaderLRU FixedFunctionShader::hlslShaderLRU;
FixedFunctionShader::HLSLShader FixedFunctionShader::hlslShaderDefaultPurple;

// Triple buffered FrameBuffer infrastructure for N / N-1 / N-2 pipeline
// recordingBuffer: Frame N - main thread records draw calls
// prepBuffer:      Frame N-1 - CPU prep thread processes (shader keys, bins)
// renderBuffer:    Frame N-2 - GPU thread renders (what displays)
FixedFunctionShader::FrameBuffer FixedFunctionShader::frameBuffers[3];
int FixedFunctionShader::recordingBuffer = 0;
int FixedFunctionShader::prepBuffer = 1;
int FixedFunctionShader::renderBuffer = 2;
bool FixedFunctionShader::n1Ready = false;   // Set true after first frame swap
bool FixedFunctionShader::n2Ready = false;   // Set true after second frame swap
int FixedFunctionShader::swapCount = 0;      // Track swaps for warm-up

// Pipeline phase tracking for GPU call separation verification
FixedFunctionShader::PipelinePhase FixedFunctionShader::currentPhase = FixedFunctionShader::PipelinePhase::Idle;
FixedFunctionShader::PhaseCallCounts FixedFunctionShader::frameCaptureGpuCalls = {};
FixedFunctionShader::PhaseCallCounts FixedFunctionShader::recordingGpuCalls = {};

static int s_cellCrossDiagStartFrame = -1;
static int s_cellCrossDiagEndFrame = -1;
static void* s_cellCrossOldCell = nullptr;
static void* s_cellCrossNewCell = nullptr;
static bool s_cellCrossOldExterior = false;
static bool s_cellCrossNewExterior = false;

static FixedFunctionShader::PhaseCallCounts* getActiveCounters() {
    auto phase = FixedFunctionShader::getPhase();
    if (phase == FixedFunctionShader::PipelinePhase::FrameCapture)
        return &FixedFunctionShader::frameCaptureGpuCalls;
    if (phase == FixedFunctionShader::PipelinePhase::Recording)
        return &FixedFunctionShader::recordingGpuCalls;
    return nullptr;  // Idle/CpuPrepare/GpuRender — not tracked
}

void FixedFunctionShader::trackDeviceRead(const char* callName) {
    if (auto* c = getActiveCounters()) {
        c->deviceReads++;
    }
}

void FixedFunctionShader::trackDeviceWrite(const char* callName) {
    if (auto* c = getActiveCounters()) {
        c->deviceWrites++;
        // Log first few writes during recording as warnings
        if (currentPhase == PipelinePhase::Recording && c->deviceWrites <= 3) {
            LOG::logline("!! RECORD WRITE: device write '%s' during Recording (#%d)", callName, c->deviceWrites);
        }
    }
}

void FixedFunctionShader::trackDeviceSubmit(const char* callName) {
    if (auto* c = getActiveCounters()) {
        c->deviceSubmits++;
        // Always log submits — these are the worst violations
        LOG::logline("!! PHASE VIOLATION: GPU submit '%s' during %s phase (#%d)",
            callName,
            currentPhase == PipelinePhase::FrameCapture ? "FrameCapture" : "Recording",
            c->deviceSubmits);
    }
}

void FixedFunctionShader::beginCellCrossDiagnostics(void* oldCell, void* newCell, bool oldExterior, bool newExterior) {
    int frame = getFrameNumber();
    s_cellCrossDiagStartFrame = frame;
    // Covers recording at crossing and the N-1/N-2 prepare/replay frames that follow.
    s_cellCrossDiagEndFrame = frame + 8;
    s_cellCrossOldCell = oldCell;
    s_cellCrossNewCell = newCell;
    s_cellCrossOldExterior = oldExterior;
    s_cellCrossNewExterior = newExterior;

    LOG_CAT(LOG::Cat_Recording, "[CELLX] BEGIN frame=%d oldCell=%p newCell=%p oldExt=%d newExt=%d window=[%d,%d]",
                 frame, oldCell, newCell, oldExterior ? 1 : 0, newExterior ? 1 : 0,
                 s_cellCrossDiagStartFrame, s_cellCrossDiagEndFrame);
}

bool FixedFunctionShader::cellCrossDiagnosticsActive() {
    if (!LOG::catEnabled(LOG::Cat_Recording)) {
        return false;
    }
    int frame = getFrameNumber();
    return s_cellCrossDiagEndFrame >= 0 &&
           frame >= s_cellCrossDiagStartFrame &&
           frame <= s_cellCrossDiagEndFrame;
}

void FixedFunctionShader::logCellCrossFrame(const char* phase, const FrameBuffer& fb) {
    if (!cellCrossDiagnosticsActive()) return;

    int terrain = 0, terrainBlend = 0, opaque = 0, blending = 0;
    int disp = 0, renderableDisp = 0, baseH = 0, overlay = 0, overlayH = 0;
    int shouldRender = 0, absorbed = 0, bbox = 0;
    for (const auto& call : fb.recordedCalls) {
        if (call.shouldRender) ++shouldRender;
        if (call.absorbed) ++absorbed;
        if (call.hasBoundingBox) ++bbox;
        if (call.bin == RenderBin::Terrain) {
            ++terrain;
            if (call.baseParamHTexture) ++baseH;
            if (call.overlayTexture) ++overlay;
            if (call.overlayParamHTexture) ++overlayH;
            if (call.sk.hasDisplacement) {
                ++disp;
                if (call.shouldRender && !call.absorbed) ++renderableDisp;
            }
        } else if (call.bin == RenderBin::TerrainBlend) {
            ++terrainBlend;
        } else if (call.bin == RenderBin::Opaque) {
            ++opaque;
        } else if (call.bin == RenderBin::Blending) {
            ++blending;
        }
    }

    D3DXMATRIX invView, invCurrentView;
    D3DXVECTOR4 origin(0.0f, 0.0f, 0.0f, 1.0f), eye(0.0f, 0.0f, 0.0f, 1.0f), currentEye(0.0f, 0.0f, 0.0f, 1.0f);
    D3DXMatrixInverse(&invView, nullptr, &fb.view);
    D3DXVec4Transform(&eye, &origin, &invView);
    D3DXMatrixInverse(&invCurrentView, nullptr, &fb.currentView);
    D3DXVec4Transform(&currentEye, &origin, &invCurrentView);

    LOG::logline("[CELLX][%s] curFrame=%d fbFrame=%d oldCell=%p newCell=%p ext=%d->%d calls=%d should=%d bbox=%d terrain=%d blend=%d opaque=%d blending=%d disp=%d renderDisp=%d near=%u baseH=%d overlay=%d overlayH=%d cellBatch=%p/%Ix cached=%d",
                 phase ? phase : "?",
                 getFrameNumber(), fb.frameNumber, s_cellCrossOldCell, s_cellCrossNewCell,
                 s_cellCrossOldExterior ? 1 : 0, s_cellCrossNewExterior ? 1 : 0,
                 (int)fb.recordedCalls.size(), shouldRender, bbox, terrain, terrainBlend,
                 opaque, blending, disp, renderableDisp, fb.nearPatchCount,
                 baseH, overlay, overlayH, fb.cellBatchCacheKey, fb.cellBatchLayoutHash,
                 fb.useCachedMergedVB ? 1 : 0);
    LOG::logline("[CELLX][%s] eye(record)=%.1f,%.1f,%.1f eye(current)=%.1f,%.1f,%.1f deltaXY=%.1f,%.1f absorbed=%d",
                 phase ? phase : "?",
                 eye.x, eye.y, eye.z, currentEye.x, currentEye.y, currentEye.z,
                 currentEye.x - eye.x, currentEye.y - eye.y, absorbed);

    int emitted = 0;
    for (size_t i = 0; i < fb.recordedCalls.size() && emitted < 8; ++i) {
        const auto& call = fb.recordedCalls[i];
        if (call.bin != RenderBin::Terrain) continue;
        if (!call.sk.hasDisplacement && emitted >= 4) continue;

        float cx = 0.0f, cy = 0.0f, cz = 0.0f, d2 = -1.0f;
        if (call.hasBoundingBox) {
            cx = 0.5f * (call.bboxMin.x + call.bboxMax.x);
            cy = 0.5f * (call.bboxMin.y + call.bboxMax.y);
            cz = 0.5f * (call.bboxMin.z + call.bboxMax.z);
            float dx = cx - eye.x;
            float dy = cy - eye.y;
            d2 = dx * dx + dy * dy;
        }

        LOG::logline("[CELLX][%s] tile idx=%u disp=%d should=%d tier=%u neigh=%02x edgeHash=%08x vb=%p ib=%p tex=%p baseH=%p overlay=%p overlayH=%p center=%.1f,%.1f,%.1f d2=%.0f",
                     phase ? phase : "?",
                     (unsigned)i, call.sk.hasDisplacement ? 1 : 0, call.shouldRender ? 1 : 0,
                     (unsigned)call.subdivTier, (unsigned)call.subdivNeighborDirMask,
                     call.edgeContextHash, call.rs.vb, call.rs.ib, call.rs.texture,
                     call.baseParamHTexture, call.overlayTexture, call.overlayParamHTexture,
                     cx, cy, cz, d2);
        ++emitted;
    }
}

// HLSL Render Dispatch Recording System
std::atomic<bool> FixedFunctionShader::isRecording{false};
std::atomic<bool> FixedFunctionShader::isReplaying{false};
bool FixedFunctionShader::usingN1Buffer = false;
bool FixedFunctionShader::manualRecordingControl = false;
bool FixedFunctionShader::recordingEnabled = true;
bool FixedFunctionShader::recordingCompletedThisFrame = false;
bool FixedFunctionShader::recordingStartedThisFrame = false;
int FixedFunctionShader::currentRecordingScene = 0;
bool FixedFunctionShader::hiZBuiltThisFrame = false;
bool FixedFunctionShader::dumpRequested = false;

// Slow frame detection: prepareMs stored by prepareRecordedCalls, read by replayRecordedCalls
float lastPrepareMs = 0.0f;

// Flag: set to false by executeCullPass (cull thread) to prevent device calls in suffix fallback
// Main thread leaves this true (default). No overlap: main thread doesn't call computeShaderKeyWithSuffixes
// during cull window (recording is stopped, replay hasn't started).
bool deviceCallsSafeInPrepare = true;

// Global frame counter - incremented in Present(), used for debug logging in all modes
int g_diagFrameCounter = 0;
std::unordered_set<FixedFunctionShader::ShaderKey, FixedFunctionShader::ShaderKey::hasher> FixedFunctionShader::diagHitKeys;

// Bbox cache: persists across frames for fast object-space bbox lookup
std::unordered_map<FixedFunctionShader::MeshKey, FixedFunctionShader::ObjectSpaceBBox, FixedFunctionShader::MeshKeyHash> FixedFunctionShader::bboxCache;
std::mutex FixedFunctionShader::bboxCacheMutex;

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

// State contracts for phase transitions
StateContract FixedFunctionShader::preRecordingContract;
StateContract FixedFunctionShader::postRecordingContract;
StateContract FixedFunctionShader::transitionState[static_cast<int>(PhaseTransition::Count)];

// Phase transition name helper
const char* getPhaseTransitionName(PhaseTransition trans) {
    switch (trans) {
        // Main scene boundaries
        case PhaseTransition::RecordingEntry: return "RecordingEntry";
        case PhaseTransition::RecordingExit:  return "RecordingExit";

        // GPU phase sub-stages
        case PhaseTransition::Stage0Entry:    return "Stage0Entry";
        case PhaseTransition::ShadowEntry:    return "ShadowEntry";
        case PhaseTransition::ShadowExit:     return "ShadowExit";
        case PhaseTransition::SkyEntry:       return "SkyEntry";
        case PhaseTransition::SkyExit:        return "SkyExit";
        case PhaseTransition::WaterReflEntry: return "WaterReflEntry";
        case PhaseTransition::WaterReflExit:  return "WaterReflExit";
        case PhaseTransition::Stage0Exit:     return "Stage0Exit";

        // Offscreen
        case PhaseTransition::OffscreenEntry: return "OffscreenEntry";
        case PhaseTransition::OffscreenExit:  return "OffscreenExit";

        // Stage 1
        case PhaseTransition::Stage1Entry:    return "Stage1Entry";
        case PhaseTransition::DepthEntry:     return "DepthEntry";
        case PhaseTransition::DepthExit:      return "DepthExit";
        case PhaseTransition::Stage1Exit:     return "Stage1Exit";

        // Stage Blend
        case PhaseTransition::StageBlendEntry:  return "StageBlendEntry";
        case PhaseTransition::WaterPlaneEntry:  return "WaterPlaneEntry";
        case PhaseTransition::WaterPlaneExit:   return "WaterPlaneExit";
        case PhaseTransition::StageBlendExit:   return "StageBlendExit";

        // HLSL replay
        case PhaseTransition::ReplayEntry:    return "ReplayEntry";
        case PhaseTransition::ReplayExit:     return "ReplayExit";

        // Post-GPU
        case PhaseTransition::GpuExit:        return "GpuExit";
        case PhaseTransition::Scene1Entry:    return "Scene1Entry";
        case PhaseTransition::UIEntry:        return "UIEntry";

        default: return "Unknown";
    }
}

// Baseline file path helper
static std::string getBaselineFilePath(PhaseTransition trans) {
    char path[512];
    snprintf(path, sizeof(path), "Data Files/shaders/state_baselines/%s.txt", getPhaseTransitionName(trans));
    return std::string(path);
}

// Save state contract to file
void saveStateBaseline(PhaseTransition trans, const StateContract& state) {
    std::string path = getBaselineFilePath(trans);

    // Ensure directory exists
    CreateDirectoryA("Data Files/shaders/state_baselines", NULL);

    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        LOG::logline("Failed to save baseline: %s", path.c_str());
        return;
    }

    fprintf(f, "# State baseline for %s\n", getPhaseTransitionName(trans));
    fprintf(f, "zEnable=%lu\n", state.zEnable);
    fprintf(f, "zWriteEnable=%lu\n", state.zWriteEnable);
    fprintf(f, "zFunc=%lu\n", state.zFunc);
    fprintf(f, "alphaBlendEnable=%lu\n", state.alphaBlendEnable);
    fprintf(f, "srcBlend=%lu\n", state.srcBlend);
    fprintf(f, "destBlend=%lu\n", state.destBlend);
    fprintf(f, "alphaTestEnable=%lu\n", state.alphaTestEnable);
    fprintf(f, "alphaFunc=%lu\n", state.alphaFunc);
    fprintf(f, "alphaRef=%lu\n", state.alphaRef);
    fprintf(f, "cullMode=%lu\n", state.cullMode);
    fprintf(f, "fogEnable=%lu\n", state.fogEnable);
    fprintf(f, "specularEnable=%lu\n", state.specularEnable);
    fprintf(f, "localViewer=%lu\n", state.localViewer);
    fprintf(f, "normalizeNormals=%lu\n", state.normalizeNormals);

    fclose(f);
    LOG::logline("Saved baseline: %s", path.c_str());
}

// Load state contract from file
bool loadStateBaseline(PhaseTransition trans, StateContract* outState) {
    std::string path = getBaselineFilePath(trans);

    FILE* f = fopen(path.c_str(), "r");
    if (!f) {
        return false;
    }

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;

        char key[64];
        DWORD value;
        if (sscanf(line, "%63[^=]=%lu", key, &value) == 2) {
            if (strcmp(key, "zEnable") == 0) outState->zEnable = value;
            else if (strcmp(key, "zWriteEnable") == 0) outState->zWriteEnable = value;
            else if (strcmp(key, "zFunc") == 0) outState->zFunc = value;
            else if (strcmp(key, "alphaBlendEnable") == 0) outState->alphaBlendEnable = value;
            else if (strcmp(key, "srcBlend") == 0) outState->srcBlend = value;
            else if (strcmp(key, "destBlend") == 0) outState->destBlend = value;
            else if (strcmp(key, "alphaTestEnable") == 0) outState->alphaTestEnable = value;
            else if (strcmp(key, "alphaFunc") == 0) outState->alphaFunc = value;
            else if (strcmp(key, "alphaRef") == 0) outState->alphaRef = value;
            else if (strcmp(key, "cullMode") == 0) outState->cullMode = value;
            else if (strcmp(key, "fogEnable") == 0) outState->fogEnable = value;
            else if (strcmp(key, "specularEnable") == 0) outState->specularEnable = value;
            else if (strcmp(key, "localViewer") == 0) outState->localViewer = value;
            else if (strcmp(key, "normalizeNormals") == 0) outState->normalizeNormals = value;
        }
    }

    fclose(f);
    return true;
}

// Compare current state to baseline
bool compareToBaseline(PhaseTransition trans, const StateContract& current) {
    StateContract baseline;
    if (!loadStateBaseline(trans, &baseline)) {
        return true; // No baseline to compare against
    }

    bool match = true;
    const char* name = getPhaseTransitionName(trans);

    if (current.zEnable != baseline.zEnable) {
        LOG::logline("!! BASELINE DIFF [%s] zEnable: baseline=%lu current=%lu", name, baseline.zEnable, current.zEnable);
        match = false;
    }
    if (current.zWriteEnable != baseline.zWriteEnable) {
        LOG::logline("!! BASELINE DIFF [%s] zWriteEnable: baseline=%lu current=%lu", name, baseline.zWriteEnable, current.zWriteEnable);
        match = false;
    }
    if (current.alphaBlendEnable != baseline.alphaBlendEnable) {
        LOG::logline("!! BASELINE DIFF [%s] alphaBlendEnable: baseline=%lu current=%lu", name, baseline.alphaBlendEnable, current.alphaBlendEnable);
        match = false;
    }
    if (current.alphaTestEnable != baseline.alphaTestEnable) {
        LOG::logline("!! BASELINE DIFF [%s] alphaTestEnable: baseline=%lu current=%lu", name, baseline.alphaTestEnable, current.alphaTestEnable);
        match = false;
    }
    if (current.cullMode != baseline.cullMode) {
        LOG::logline("!! BASELINE DIFF [%s] cullMode: baseline=%lu current=%lu", name, baseline.cullMode, current.cullMode);
        match = false;
    }
    if (current.fogEnable != baseline.fogEnable) {
        LOG::logline("!! BASELINE DIFF [%s] fogEnable: baseline=%lu current=%lu", name, baseline.fogEnable, current.fogEnable);
        match = false;
    }

    return match;
}

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
IDirect3DTexture9* FixedFunctionShader::defaultParamHTexture = nullptr;

// Original detail texture storage
IDirect3DBaseTexture9* FixedFunctionShader::savedOriginalDetailTexture = nullptr;

// Per-object light packing for mode 3
std::vector<FixedFunctionShader::PerObjectLightInfo> FixedFunctionShader::perObjectLightInfo;
float FixedFunctionShader::perObjectTexelSize = 0.0f;
// Note: texPerObjectLightData moved to RenderThread for thread safety

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

// O3 recompilation system - upgrades O1 shaders to O3 after game starts
std::queue<FixedFunctionShader::ShaderKey> FixedFunctionShader::o3RecompileQueue;
std::mutex FixedFunctionShader::o3QueueMutex;
std::atomic<bool> FixedFunctionShader::o3RecompileActive{false};
std::vector<std::thread> FixedFunctionShader::o3RecompileWorkers;
std::atomic<bool> FixedFunctionShader::o3RecompileStarted{false};

// Precache progress counters for loading bar
std::atomic<int> FixedFunctionShader::precacheCompleted{0};
std::atomic<int> FixedFunctionShader::precacheTotal{0};

// Cache for texture resolutions to avoid repeated GetLevelDesc calls

// Texture release callback (evicts from suffix cache on texture free)
void (*g_onTextureReleased)(IDirect3DTexture9* realTexture) = nullptr;

// VB/IB release callbacks (evict bboxCache + PatchDisplacement on buffer free).
void (*g_onVertexBufferReleased)(IDirect3DVertexBuffer9* realBuffer) = nullptr;
void (*g_onIndexBufferReleased)(IDirect3DIndexBuffer9* realBuffer) = nullptr;

static string buildArgString(DWORD arg, const string& mask, const string& sampler);

bool FixedFunctionShader::init(IDirect3DDevice* d, ID3DXEffectPool* pool) {
    // Wait for precache thread with animated loading bar updates
    if (precacheThread) {
        auto mwBridge = MWBridge::get();
        char buffer[64];

        // Poll with timeout, updating loading bar each iteration
        while (WaitForSingleObject(precacheThread, 100) == WAIT_TIMEOUT) {
            int done = precacheCompleted.load();
            int total = precacheTotal.load();
            if (total > 0) {
                std::snprintf(buffer, sizeof(buffer), "Loading MGE XE... (%d/%d shaders)", done, total);
                mwBridge->showLoadingBar(buffer, 95.0f);
            }
        }

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
    ehDebugMode = effect->GetParameterByName(0, "debugMode");

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
    g_onTextureReleased = TextureSuffix::onTextureReleased;

    // Register VB/IB release callbacks for PatchDisplacement (keyed on VB/IB
    // pointers). bboxCache uses bulk-clear on cell transitions instead — see
    // recording_system.cpp startRecording.
    g_onVertexBufferReleased = [](IDirect3DVertexBuffer9* vb) {
        PatchDisplacement::onVertexBufferReleased(vb);
    };
    g_onIndexBufferReleased = [](IDirect3DIndexBuffer9* ib) {
        PatchDisplacement::onIndexBufferReleased(ib);
    };

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
        LOG::logline("-- Starting immediate HLSL shader precaching (multithreaded)");

        // Launch coordinator thread that spawns worker threads
        precacheThread = CreateThread(nullptr, 0, [](LPVOID) -> DWORD {
            bool hlslMode = (Configuration.PerPixelLightFlags == 2);

            if (hlslMode) {
                LOG::logline("-- Precache thread started");

                // Variant struct: {lighting, lightMode, vertexCol, vertexMat, heavyLighting, skinning, ph, px, fogMode, stages}
                struct ShaderVariant {
                    int lighting, lightMode, vertexCol, vertexMat, heavyLighting, skinning;
                    int hasParamH, hasParamX, fogMode, stages;
                };

                // Order: lm=3 first (most complex/likely needed), then lm=2, lm=1, lm=0, unlit last
                ShaderVariant variants[] = {
                    // lm=3 (texture lights) — FIRST: most complex, most likely to be needed
                    {1,3, 0,1, 0,0, 0,0, 1,1}, {1,3, 1,2, 0,0, 0,0, 1,1},  // hl=0 base
                    {1,3, 0,1, 0,0, 1,0, 1,1}, {1,3, 1,2, 0,0, 1,0, 1,1},  // hl=0 ph
                    {1,3, 0,1, 0,1, 0,0, 1,1}, {1,3, 1,2, 0,1, 0,0, 1,1},  // hl=0 skin
                    {1,3, 0,1, 1,0, 0,0, 1,1}, {1,3, 1,2, 1,0, 0,0, 1,1},  // hl=1 base
                    {1,3, 0,1, 1,0, 1,0, 1,1}, {1,3, 1,2, 1,0, 1,0, 1,1},  // hl=1 ph
                    {1,3, 0,1, 1,1, 0,0, 1,1}, {1,3, 1,2, 1,1, 0,0, 1,1},  // hl=1 skin
                    {1,3, 1,2, 0,0, 0,0, 2,1}, {1,3, 1,2, 1,0, 0,0, 2,1},  // fog=2

                    // lm=2 (few lights)
                    {1,2, 0,1, 0,0, 0,0, 1,1}, {1,2, 0,1, 1,0, 0,0, 1,1},
                    {1,2, 1,2, 0,0, 0,0, 1,1}, {1,2, 1,2, 1,0, 0,0, 1,1},
                    {1,2, 0,1, 0,0, 1,0, 1,1}, {1,2, 0,1, 1,0, 1,0, 1,1},
                    {1,2, 1,2, 0,0, 1,0, 1,1}, {1,2, 1,2, 1,0, 1,0, 1,1},
                    {1,2, 0,1, 0,1, 0,0, 1,1}, {1,2, 0,1, 1,1, 0,0, 1,1},
                    {1,2, 1,2, 0,1, 0,0, 1,1}, {1,2, 1,2, 1,1, 0,0, 1,1},
                    {1,2, 1,1, 0,1, 1,0, 1,1}, {1,2, 0,1, 0,0, 0,0, 1,2},
                    {1,2, 1,2, 0,0, 0,0, 2,1}, {1,2, 1,2, 1,0, 0,0, 2,1},

                    // lm=1 (single point light)
                    {1,1, 0,1, 0,0, 0,0, 1,1}, {1,1, 1,2, 0,0, 0,0, 1,1},
                    {1,1, 0,1, 0,0, 1,0, 1,1}, {1,1, 1,2, 0,0, 1,0, 1,1},
                    {1,1, 0,1, 0,1, 0,0, 1,1},
                    {1,1, 1,1, 0,1, 0,0, 1,1}, {1,1, 1,1, 0,1, 1,0, 1,1},
                    {1,1, 1,2, 0,0, 0,0, 2,1},

                    // lm=0 (sun only)
                    {1,0, 0,1, 0,0, 0,0, 1,1}, {1,0, 1,2, 0,0, 0,0, 1,1},
                    {1,0, 0,1, 0,0, 1,0, 1,1}, {1,0, 1,2, 0,0, 1,0, 1,1},
                    {1,0, 0,1, 0,1, 0,0, 1,1}, {1,0, 1,2, 0,1, 0,0, 1,1},
                    {1,0, 1,1, 0,0, 0,0, 1,1}, {1,0, 1,1, 0,1, 0,0, 1,1},
                    {1,0, 1,1, 0,1, 1,0, 1,1}, {1,0, 0,1, 0,0, 0,0, 1,2},

                    // UNLIT (lighting=0) — particles, UI, emissive-only objects (last)
                    {0,0, 0,0, 0,0, 0,0, 0,0}, {0,0, 0,0, 0,0, 0,0, 1,0},
                    {0,0, 0,1, 0,0, 0,0, 0,0}, {0,0, 0,1, 0,0, 0,0, 1,0},
                    {0,0, 1,2, 0,0, 0,0, 0,0}, {0,0, 1,2, 0,0, 0,0, 1,0},
                    {0,0, 0,1, 0,0, 0,0, 1,1}, {0,0, 1,2, 0,0, 0,0, 1,1},
                    {0,0, 0,1, 0,0, 0,0, 0,1}, {0,0, 1,2, 0,0, 0,0, 0,1},
                    {0,0, 0,0, 0,0, 0,0, 1,1}, {0,0, 0,0, 0,0, 0,0, 0,1},
                };
                const int numVariants = sizeof(variants) / sizeof(variants[0]);

                // Build vector of all ShaderKeys
                std::vector<ShaderKey> allKeys;
                allKeys.reserve(numVariants + 4);
                int hasShadows = ((Configuration.MGEFlags & USE_SHADOWS) && (Configuration.MGEFlags & USE_DISTANT_LAND)) ? 1 : 0;

                for (int i = 0; i < numVariants; i++) {
                    const auto& v = variants[i];
                    ShaderKey sk;
                    memset(&sk, 0, sizeof(sk));
                    sk.uvSets = (v.stages > 0) ? 1 : 0;
                    sk.useLighting = v.lighting;
                    sk.lightMode = v.lightMode;
                    sk.vertexColour = v.vertexCol;
                    sk.vertexMaterial = v.vertexMat;
                    sk.usesSkinning = v.skinning;
                    sk.heavyLighting = v.heavyLighting;
                    sk.hasParamH = v.hasParamH;
                    sk.hasParamX = v.hasParamX;
                    sk.hasGrass = 0;
                    sk.hasShadows = hasShadows;
                    sk.fogMode = v.fogMode;
                    sk.activeStages = v.stages;
                    sk.stage[0] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 1, 0, 0, 0 };
                    if (v.stages > 1) {
                        sk.stage[1] = { D3DTOP_ADD, D3DTA_TEXTURE, D3DTA_CURRENT, D3DTA_CURRENT, 0, 0, 0, 0 };
                    }
                    memset(&sk.stage[v.stages], 0, sizeof(sk.stage[0]) * (8 - v.stages));
                    allKeys.push_back(sk);
                }

                // Stateless batch variants (lm=3 priority, then lm=2, lm=1, lm=0)
                // These use useStatelessBatch=1 for batched rendering
                struct StatelessVariant { int lightMode, vertexCol, vertexMat, heavyLighting; };
                StatelessVariant statelessVariants[] = {
                    // lm=3 stateless (highest priority - texture lights)
                    {3, 0, 1, 0}, {3, 1, 2, 0}, {3, 0, 1, 1}, {3, 1, 2, 1},
                    // lm=2 stateless
                    {2, 0, 1, 0}, {2, 1, 2, 0}, {2, 0, 1, 1}, {2, 1, 2, 1},
                    // lm=1 stateless
                    {1, 0, 1, 0}, {1, 1, 2, 0},
                    // lm=0 stateless
                    {0, 0, 1, 0}, {0, 1, 2, 0},
                };
                for (const auto& sv : statelessVariants) {
                    ShaderKey sk;
                    memset(&sk, 0, sizeof(sk));
                    sk.uvSets = 1;
                    sk.useLighting = 1;
                    sk.lightMode = sv.lightMode;
                    sk.vertexColour = sv.vertexCol;
                    sk.vertexMaterial = sv.vertexMat;
                    sk.heavyLighting = sv.heavyLighting;
                    sk.fogMode = 1;
                    sk.activeStages = 1;
                    sk.hasShadows = hasShadows;
                    sk.useStatelessBatch = 1;  // Key: stateless batch mode
                    sk.stage[0] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 1, 0, 0, 0 };
                    allKeys.push_back(sk);
                }

                // Grass variants
                struct GrassVariant { int lightMode, vertexCol, vertexMat; };
                GrassVariant grassVariants[] = { {0, 0, 1}, {0, 1, 2}, {1, 0, 1}, {1, 1, 2} };
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
                    sk.hasShadows = hasShadows;
                    sk.stage[0] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 1, 0, 0, 0 };
                    allKeys.push_back(sk);
                }

                const int totalShaders = (int)allKeys.size();
                precacheTotal.store(totalShaders);
                precacheCompleted.store(0);

                LOG::logline("-- Precaching %d HLSL shader variants", totalShaders);

                // Atomic index for work distribution
                std::atomic<int> nextIndex(0);
                std::atomic<int> compiledCount(0);

                // Worker function - each thread pulls work via atomic index
                auto workerFunc = [&]() {
                    while (true) {
                        int idx = nextIndex.fetch_add(1);
                        if (idx >= totalShaders) break;

                        const ShaderKey& sk = allKeys[idx];
                        HLSLShader shader = generateMWShaderHLSL(sk);

                        AcquireSRWLockExclusive(&hlslCacheLock);
                        if (cacheHLSLShaders.find(sk) == cacheHLSLShaders.end()) {
                            cacheHLSLShaders[sk] = shader;
                        }
                        ReleaseSRWLockExclusive(&hlslCacheLock);

                        compiledCount.fetch_add(1);
                        precacheCompleted.store(compiledCount.load());
                    }
                };

                // Launch worker threads (cap at 4 to avoid device contention)
                int numThreads = std::min(4, (int)std::thread::hardware_concurrency());
                if (numThreads < 1) numThreads = 1;
                LOG::logline("-- Using %d compile threads", numThreads);

                std::vector<std::thread> workers;
                for (int t = 0; t < numThreads; t++) {
                    workers.emplace_back(workerFunc);
                }

                // Wait for all workers
                for (auto& w : workers) {
                    w.join();
                }

                StatusOverlay::setStatus("HLSL shader precaching complete");
                LOG::logline("-- Precache complete: %d shaders compiled", compiledCount.load());

                // Start O3 recompilation immediately after precache
                queueAllO3Recompiles();
                startO3RecompileThread();
                DistantLandHLSL::startO3RecompileThread();
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
    // Use HLSL pipeline if mode is set to HLSL
    if (isHLSLActive()) {
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

    // Debug visualization mode (shared with HLSL path)
    if (ehDebugMode) {
        effectFFE->SetInt(ehDebugMode, ImGuiManager::GetShaderDebugMode());
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
                actualTexture = defaultWhiteTexture;
                break;
            case 2: // ParamH slot (metallic/roughness) - use neutral PBR defaults
                actualTexture = defaultParamHTexture;
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

    // Create 1x1 paramH texture with neutral PBR values (new DXT5 packing semantics)
    // R=0 metalness, G=128 roughness(0.5), B=128 IOR(0.5), A=0 height (no parallax)
    // D3DFMT_A8R8G8B8 packs as 0xAARRGGBB. Defensive default only — with HAS_PARAMH
    // set, a real paramh is always bound.
    if (device->CreateTexture(1, 1, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &defaultParamHTexture, nullptr) == S_OK) {
        D3DLOCKED_RECT lockedRect;
        if (defaultParamHTexture->LockRect(0, &lockedRect, nullptr, 0) == S_OK) {
            *(DWORD*)lockedRect.pBits = 0x00008080; // A=0, R=0, G=128, B=128
            defaultParamHTexture->UnlockRect(0);
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
    DWORD vsCompileFlags = ShaderUtils::isDXVK() ? D3DCOMPILE_OPTIMIZATION_LEVEL1 : D3DCOMPILE_OPTIMIZATION_LEVEL3;
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

                // Resolve register offsets for error shader (inline since resolveConstReg is defined later)
                auto resolveReg = [&](D3DXHANDLE h) -> ConstReg {
                    ConstReg cr;
                    if (h) {
                        D3DXCONSTANT_DESC desc;
                        UINT count = 1;
                        if (SUCCEEDED(errorShader.vsConstantTable->GetConstantDesc(h, &desc, &count))) {
                            cr.reg = desc.RegisterIndex;
                            cr.count = desc.RegisterCount;
                            cr.regSet = (UINT)desc.RegisterSet;
                        }
                    }
                    return cr;
                };
                errorShader.regProj = resolveReg(errorShader.hProj);
                errorShader.regWorldView = resolveReg(errorShader.hWorldView);
                errorShader.regView = resolveReg(errorShader.hView);
                errorShader.regWorld = resolveReg(errorShader.hWorld);
            }
        }
        vsBlob->Release();
    }
    if (vsErrors) vsErrors->Release();
    
    // Compile pixel shader
    ID3DBlob* psBlob = nullptr;
    ID3DBlob* psErrors = nullptr;
    DWORD psCompileFlags = ShaderUtils::isDXVK() ? D3DCOMPILE_OPTIMIZATION_LEVEL1 : D3DCOMPILE_OPTIMIZATION_LEVEL3;
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

    ReleaseSRWLockExclusive(&hlslCacheLock);

    // Clear shader source cache for hot reloading (delegated to HLSLShaderManager)
    HLSLShaderManager::invalidateHLSLCache();

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
                
                // LOG::logline("-- HLSL Async compiled shader with flags paramh=%d paramx=%d",
                //            request->key.hasParamH, request->key.hasParamX);
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

// O3 Recompilation System - upgrades O1 shaders to O3 after game starts
void FixedFunctionShader::queueAllO3Recompiles() {
    // Scan cache under shared lock, queue all non-O3 shaders for O3 recompile
    std::lock_guard<std::mutex> queueLock(o3QueueMutex);
    AcquireSRWLockShared(&hlslCacheLock);

    int queued = 0;
    for (const auto& entry : cacheHLSLShaders) {
        if (entry.second.optimizationLevel < 3) {
            o3RecompileQueue.push(entry.first);
            queued++;
        }
    }

    ReleaseSRWLockShared(&hlslCacheLock);
    LOG::logline("-- O3 recompile: queued %d shaders for upgrade", queued);
}

void FixedFunctionShader::startO3RecompileThread() {
    // Only start once
    if (o3RecompileStarted.exchange(true)) {
        return;
    }

    o3RecompileActive = true;

    constexpr int numWorkers = 2;
    static std::atomic<int> totalCompiled{0};
    totalCompiled = 0;

    auto workerFunc = []() {
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);

        int compiled = 0;
        while (o3RecompileActive) {
            ShaderKey key;
            bool hasWork = false;

            // Pop from queue
            {
                std::lock_guard<std::mutex> lock(o3QueueMutex);
                if (!o3RecompileQueue.empty()) {
                    key = o3RecompileQueue.front();
                    o3RecompileQueue.pop();
                    hasWork = true;
                }
            }

            if (!hasWork) {
                break;  // Queue empty, done
            }

            // Compile at O3 (this is the slow part)
            HLSLShader newShader;
            {
                MGE_ZoneScopedN("FFE_HLSL_O3_Compile");
                newShader = generateMWShaderHLSL(key, 3);
            }

            // Replace O1 entry in cache (release old shaders first)
            AcquireSRWLockExclusive(&hlslCacheLock);
            auto it = cacheHLSLShaders.find(key);
            if (it != cacheHLSLShaders.end()) {
                // Release old O1 shaders
                if (it->second.vertexShader) it->second.vertexShader->Release();
                if (it->second.pixelShader) it->second.pixelShader->Release();
                if (it->second.vsConstantTable) it->second.vsConstantTable->Release();
                if (it->second.psConstantTable) it->second.psConstantTable->Release();
                // Replace with new O3 shader
                it->second = newShader;
            }
            ReleaseSRWLockExclusive(&hlslCacheLock);

            compiled++;
        }

        totalCompiled += compiled;
    };

    LOG::logline("-- O3 recompile: starting %d workers", numWorkers);
    for (int i = 0; i < numWorkers; i++) {
        o3RecompileWorkers.emplace_back(workerFunc);
    }

    // Detach a monitor thread to log completion and invalidate LRU
    std::thread([numWorkers]() {
        for (auto& w : o3RecompileWorkers) {
            if (w.joinable()) w.join();
        }
        o3RecompileWorkers.clear();

        // Invalidate LRU so next draw picks up O3 shaders
        hlslShaderLRU.last_sk = ShaderKey();

        LOG::logline("-- O3 recompile finished: %d shaders upgraded", totalCompiled.load());
        o3RecompileActive = false;
    }).detach();
}

void FixedFunctionShader::stopO3RecompileThread() {
    o3RecompileActive = false;

    for (auto& w : o3RecompileWorkers) {
        if (w.joinable()) w.join();
    }
    o3RecompileWorkers.clear();

    // Clear any remaining queue
    std::lock_guard<std::mutex> lock(o3QueueMutex);
    while (!o3RecompileQueue.empty()) {
        o3RecompileQueue.pop();
    }
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
    
    // LOG::logline("-- HLSL Queued async compilation for shader with flags paramh=%d paramx=%d",
    //            key.hasParamH, key.hasParamX);
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

// Resolve a D3DXHANDLE to register offset using GetConstantDesc
static FixedFunctionShader::ConstReg resolveConstReg(ID3DXConstantTable* table, D3DXHANDLE handle) {
    FixedFunctionShader::ConstReg cr;
    if (table && handle) {
        D3DXCONSTANT_DESC desc;
        UINT count = 1;
        if (SUCCEEDED(table->GetConstantDesc(handle, &desc, &count))) {
            cr.reg = desc.RegisterIndex;
            cr.count = desc.RegisterCount;
            // D3DXRS_BOOL=0, D3DXRS_INT4=1, D3DXRS_FLOAT4=2, D3DXRS_SAMPLER=3
            cr.regSet = (UINT)desc.RegisterSet;
        }
    }
    return cr;
}

FixedFunctionShader::HLSLShader FixedFunctionShader::generateMWShaderHLSL(const ShaderKey& sk, uint8_t optLevel) {
    HLSLShader hlslShader = {};
    
    // Shader entry points
    const char* vertexShaderName = "vs_main";
    const char* pixelShaderName = "ps_main";
    
    // Load separate vertex and pixel shader files
    DWORD vsFileSize = 0, psFileSize = 0;
    char* vertexShaderSource = HLSLShaderManager::loadHLSLShaderFile("Data Files\\shaders\\core-hlsl\\XE FixedFuncEmu_VS.hlsl", &vsFileSize);
    char* pixelShaderSource = HLSLShaderManager::loadHLSLShaderFile("Data Files\\shaders\\core-hlsl\\XE FixedFuncEmu_PS.hlsl", &psFileSize);
    
    if (!vertexShaderSource || !pixelShaderSource) {
        if (vertexShaderSource) delete[] vertexShaderSource;
        if (pixelShaderSource) delete[] pixelShaderSource;
        return hlslShaderDefaultPurple;
    }

    // Build shader defines based on ShaderKey
    D3D_SHADER_MACRO defines[14] = {};
    int defineCount = 0;

    // Light mode define: 0=sun only, 1=single, 2=few loop, 3=texture
    char lightModeStr[2] = {'0', '\0'};
    lightModeStr[0] = (char)('0' + sk.lightMode);
    defines[defineCount++] = {"LIGHT_MODE", lightModeStr};

    // Only emit USE_TEXTURE_LIGHTS for mode 3 (>8 lights, texture loop)
    if (sk.lightMode == 3) {
        defines[defineCount++] = {"USE_TEXTURE_LIGHTS", "1"};
    }

    if (sk.hasParamH) {
        defines[defineCount++] = {"HAS_PARAMH", "1"};
        // LOG::logline("HLSL: Compiling with HAS_PARAMH define");
    }
    if (sk.disableParallax) {
        defines[defineCount++] = {"SKIP_PARALLAX", "1"};
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
    if (sk.useInstancing) {
        defines[defineCount++] = {"USE_INSTANCING", "1"};
    }
    if (sk.useStatelessBatch) {
        defines[defineCount++] = {"USE_STATELESS_BATCH", "1"};
    }
    if (sk.hasOverlay) {
        defines[defineCount++] = {"HAS_OVERLAY", "1"};
    }
    if (sk.hasOverlayParamH) {
        defines[defineCount++] = {"HAS_OVERLAY_PARAMH", "1"};
    }
    if (sk.hasDisplacement) {
        defines[defineCount++] = {"HAS_DISPLACEMENT", "1"};
    }
    defines[defineCount] = {nullptr, nullptr}; // Null terminator
    
    // LOG::logline("HLSL: Compiling shader with %d defines", defineCount);
    
    // Compile vertex shader
    // TODO: Optimize - vertex shaders don't use texture suffix defines (HAS_PARAMH, HAS_PARAMX)
    // Many vertex shaders could be cached and reused across pixel shader variants
    ID3DBlob* vsBlob = nullptr;
    ID3DBlob* vsErrors = nullptr;

    // Use IEEE_STRICTNESS for consistent Z precision across shader permutations (stateless batch vs regular)
    DWORD vsCompileFlags = D3DCOMPILE_PREFER_FLOW_CONTROL | D3DCOMPILE_IEEE_STRICTNESS;
    // Use optimization level based on optLevel parameter (1=O1, 2=O2, 3=O3)
    if (optLevel == 1) {
        vsCompileFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL1;
    } else if (optLevel == 2) {
        vsCompileFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL2;
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

        // Resolve VS register offsets for command buffer path
        hlslShader.regWorldViewProj = resolveConstReg(hlslShader.vsConstantTable, hlslShader.hWorldViewProj);
        hlslShader.regView = resolveConstReg(hlslShader.vsConstantTable, hlslShader.hView);
        hlslShader.regProj = resolveConstReg(hlslShader.vsConstantTable, hlslShader.hProj);
        hlslShader.regWorld = resolveConstReg(hlslShader.vsConstantTable, hlslShader.hWorld);
        hlslShader.regWorldView = resolveConstReg(hlslShader.vsConstantTable, hlslShader.hWorldView);
        hlslShader.regVertexBlendPalette = resolveConstReg(hlslShader.vsConstantTable, hlslShader.hVertexBlendPalette);
        hlslShader.regVertexBlendState = resolveConstReg(hlslShader.vsConstantTable, hlslShader.hVertexBlendState);
        hlslShader.regShadowWorldViewProj = resolveConstReg(hlslShader.vsConstantTable, hlslShader.hShadowWorldViewProj);
    }

    // Log VS blob size before releasing
    // LOG::logline("-- HLSL VS blob size: %u bytes", vsBlob->GetBufferSize());

    vsBlob->Release();
    
    // Compile pixel shader using pixel shader source
    ID3DBlob* psBlob = nullptr;
    ID3DBlob* psErrors = nullptr;

    // Match D3DX9 effect compilation - no IEEE_STRICTNESS for invariance with depth pass
    DWORD psCompileFlags = D3DCOMPILE_PREFER_FLOW_CONTROL;
    // Use optimization level based on optLevel parameter (1=O1, 2=O2, 3=O3)
    if (optLevel == 1) {
        psCompileFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL1;
    } else if (optLevel == 2) {
        psCompileFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL2;
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

        // Resolve PS register offsets for command buffer path
        hlslShader.regMaterialDiffuse = resolveConstReg(hlslShader.psConstantTable, hlslShader.hMaterialDiffuse);
        hlslShader.regMaterialAmbient = resolveConstReg(hlslShader.psConstantTable, hlslShader.hMaterialAmbient);
        hlslShader.regMaterialEmissive = resolveConstReg(hlslShader.psConstantTable, hlslShader.hMaterialEmissive);
        hlslShader.regLightSunDirection = resolveConstReg(hlslShader.psConstantTable, hlslShader.hLightSunDirection);
        hlslShader.regLightSunDiffuse = resolveConstReg(hlslShader.psConstantTable, hlslShader.hLightSunDiffuse);
        hlslShader.regLightSceneAmbient = resolveConstReg(hlslShader.psConstantTable, hlslShader.hLightSceneAmbient);
        hlslShader.regShadowRcpRes = resolveConstReg(hlslShader.psConstantTable, hlslShader.hShadowRcpRes);
        hlslShader.regPCFFilterSize = resolveConstReg(hlslShader.psConstantTable, hlslShader.hPCFFilterSize);
        hlslShader.regLightDiffuse = resolveConstReg(hlslShader.psConstantTable, hlslShader.hLightDiffuse);
        hlslShader.regLightPosition = resolveConstReg(hlslShader.psConstantTable, hlslShader.hLightPosition);
        hlslShader.regLightAmbient = resolveConstReg(hlslShader.psConstantTable, hlslShader.hLightAmbient);
        hlslShader.regPointLightCount = resolveConstReg(hlslShader.psConstantTable, hlslShader.hPointLightCount);
        hlslShader.regLightFalloffQuadratic = resolveConstReg(hlslShader.psConstantTable, hlslShader.hLightFalloffQuadratic);
        hlslShader.regLightFalloffConstant = resolveConstReg(hlslShader.psConstantTable, hlslShader.hLightFalloffConstant);

        // Dynamic constants resolved on first use in renderMorrowindHLSL_Internal
        hlslShader.dynamicConstsResolved = false;
    }

    // Log compilation details for debugging (before releasing blobs)
    // LOG::logline("-- HLSL shader compiled successfully: VS=%s PS=%s", vertexShaderName, pixelShaderName);
    // LOG::logline("-- HLSL PS blob size: %u bytes", psBlob->GetBufferSize());
    
    psBlob->Release();
    
    // Clean up shader sources
    delete[] vertexShaderSource;
    delete[] pixelShaderSource;

    // Record optimization level used for this shader
    hlslShader.optimizationLevel = optLevel;

    return hlslShader;
}

// Triple buffer rotation at Present()
// Rotation: Recording(N) -> Prep(N-1) -> Render(N-2) -> Recording(cleared)
void FixedFunctionShader::swapBuffers() {
    // Snapshot final light state into recording buffer BEFORE swap.
    // After swap, this buffer becomes the prep buffer — prepareRecordedCalls() reads it
    // off the worker thread. Per-frame snapshot avoids main-thread/worker-thread races
    // and frame N vs N-1 pipeline mismatch (which caused light flicker on camera move).
    frameBuffers[recordingBuffer].lastLightState = lastLightState;

    // STRESS TEST: Poison old buffer BEFORE swap to catch N-1/N-2 confusion
    // If rendering uses stale data, it will produce obviously wrong results
    if (ImGuiManager::GetStressPoisonBuffers()) {
        auto& oldBuf = frameBuffers[renderBuffer];  // Just finished rendering (will become new recording)
        // Poison matrices with NaN to catch any stale usage
        memset(&oldBuf.view, 0xFF, sizeof(D3DXMATRIX));
        memset(&oldBuf.proj, 0xFF, sizeof(D3DXMATRIX));
        memset(&oldBuf.currentView, 0xFF, sizeof(D3DXMATRIX));
        memset(&oldBuf.currentProj, 0xFF, sizeof(D3DXMATRIX));
        memset(&oldBuf.shadowViewproj[0], 0xFF, sizeof(D3DXMATRIX));
        memset(&oldBuf.shadowViewproj[1], 0xFF, sizeof(D3DXMATRIX));
        memset(&oldBuf.currentShadowViewproj[0], 0xFF, sizeof(D3DXMATRIX));
        memset(&oldBuf.currentShadowViewproj[1], 0xFF, sizeof(D3DXMATRIX));
        oldBuf.frameNumber = -9999;  // Obvious sentinel value
    }

    // Rotate indices: render becomes new recording, prep becomes render, recording becomes prep
    int newRecording = renderBuffer;   // Was render (N-2), now recording (N)
    int newPrep = recordingBuffer;     // Was recording (N), now prep (N-1)
    int newRender = prepBuffer;        // Was prep (N-1), now render (N-2)

    recordingBuffer = newRecording;
    prepBuffer = newPrep;
    renderBuffer = newRender;

    // Clear the new recording buffer (was just rendered)
    frameBuffers[recordingBuffer].clear();

    // Reset prep buffer state: was recording, now needs prep work
    // Without this, stale ReadyToRender state from previous rotation causes CPU prep to skip
    frameBuffers[prepBuffer].state = BufferState::Recording;

    // After first swap, N-1 data is available in prep buffer
    n1Ready = true;
    // After second swap, N-2 data is available in render buffer
    if (++swapCount >= 2) {
        n2Ready = true;
    }

    // Reset Stage0Early flag for next frame's split render path
    resetStage0EarlyFlag();
}

void FixedFunctionShader::release() {
    // Clear all three frame buffers — releases COM refs before D3D device is destroyed
    // This prevents crashes during DLL unload when static destructors run
    for (int i = 0; i < 3; ++i) {
        frameBuffers[i].clear();

        // Release staging buffers (not released in clear() since they persist across frames)
        if (frameBuffers[i].particleStagingVB) {
            frameBuffers[i].particleStagingVB->Release();
            frameBuffers[i].particleStagingVB = nullptr;
        }
        if (frameBuffers[i].particleStagingIB) {
            frameBuffers[i].particleStagingIB->Release();
            frameBuffers[i].particleStagingIB = nullptr;
        }
        frameBuffers[i].stagingVBSize = 0;
        frameBuffers[i].stagingIBSize = 0;
    }

    // Join precache thread before cleaning up HLSL cache
    if (precacheThread) {
        WaitForSingleObject(precacheThread, INFINITE);
        CloseHandle(precacheThread);
        precacheThread = nullptr;
    }

    // Stop O3 recompile thread
    stopO3RecompileThread();

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

    // Clean up shader source cache (delegated to HLSLShaderManager)
    HLSLShaderManager::invalidateHLSLCache();

    // Reset material state cache
    materialCache.reset();

    // Note: texPerObjectLightData cleanup moved to RenderThread::stop()
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
    TextureSuffix::getBindingState().reset();
}



// ShaderKey - Captures a generatable shader configuration

FixedFunctionShader::ShaderKey::ShaderKey(const RenderedState* rs, const FragmentState* frs, const LightState* lightrs) {
    memset(this, 0, sizeof(ShaderKey));         // Clear padding bits for compares

    uvSets = (rs->fvf & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT;
    usesSkinning = rs->vertexBlendState ? 1 : 0;
    vertexColour = (rs->fvf & D3DFVF_DIFFUSE) ? 1 : 0;
    useLighting = rs->useLighting ? 1 : 0;
    
    // Count point lights specifically (exclude directional lights like sun)
    // Thread-local cache to skip iteration if lightrs pointer unchanged (Phase 3 optimization)
    static thread_local const LightState* s_cachedLightPtr = nullptr;
    static thread_local int s_cachedPointCount = 0;
    static thread_local int s_cachedDirCount = 0;

    int pointLightCount = 0;
    int directionalLightCount = 0;
    if (rs->useLighting && lightrs) {
        if (lightrs == s_cachedLightPtr) {
            // Fast path: same light state, reuse cached counts
            pointLightCount = s_cachedPointCount;
            directionalLightCount = s_cachedDirCount;
        } else {
            // Slow path: iterate and cache
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
            s_cachedLightPtr = lightrs;
            s_cachedPointCount = pointLightCount;
            s_cachedDirCount = directionalLightCount;
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
    static_assert(sizeof(ShaderKey) % sizeof(DWORD) == 0, "ShaderKey size must be DWORD-aligned");
    constexpr size_t N = sizeof(ShaderKey) / sizeof(DWORD);
    DWORD z[N];
    memcpy(&z, &k, sizeof(z));
    size_t h = z[0] << 16;
    for (size_t i = 1; i < N; ++i) h ^= z[i];
    return h;
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

// Recording system functions moved to recording_system.cpp:
// - startRecording(), stopRecordingAndReplay()
// - finalizeBatchAndSubmitCull(), waitCullAndReplay(), finalizeBatchAndReplay()
// - capturePostRecordingState(), finalizeAndRender(), executeGpuPhase()
// - markSceneStart(), markSceneEnd(), executeCullPass(), executeRenderPass()
// - compareLightStates(), recordRenderCall()

// ====== REMOVED: Recording system implementation (lines 3216-4011) ======
// See recording_system.cpp for the implementation

// Placeholder marker for code extraction verification
static_assert(true, "Recording system extracted to recording_system.cpp");

// ====== REMOVED: Recording system implementation (800 lines) ======
// Moved to recording_system.cpp
// Functions: startRecording, stopRecordingAndReplay, finalizeBatchAndSubmitCull,
//            waitCullAndReplay, finalizeBatchAndReplay, capturePostRecordingState,
//            finalizeAndRender, executeGpuPhase, markSceneStart, markSceneEnd,
//            executeCullPass, executeRenderPass, compareLightStates, recordRenderCall

// Helper functions still needed by remaining code in ffeshader.cpp

// Helper: get current recording buffer's calls vector
static auto& currentRecordedCalls() {
    return FixedFunctionShader::getRecordingBuffer().recordedCalls;
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
    float eyePosX, eyePosY, eyePosZ;  // World position from DistantLand::s_staging.eyePos
    float recordCameraX, recordCameraY, recordCameraZ;
    std::vector<CallSnapshotInfo> callInfos;  // ALL calls

    // Pipeline control-flow state (from g_pipelineDiag, filled at Present())
    PipelineDiag pipeline;

    // FFS-side recording state
    bool ffs_isRecording, ffs_isReplaying;
    bool ffs_recordingCompletedThisFrame, ffs_recordingEnabled, ffs_manualRecordingControl;
    int ffs_recordedCallCount;  // static recordedCalls.size()
    int ffs_bufferState;        // BufferState enum as int

    FrameSnapshot() : valid(false), totalRecordedCalls(0), sceneLightsTotal(0),
        cameraX(0), cameraY(0), cameraZ(0),
        eyePosX(0), eyePosY(0), eyePosZ(0),
        recordCameraX(0), recordCameraY(0), recordCameraZ(0),
        pipeline{}, ffs_isRecording(false), ffs_isReplaying(false),
        ffs_recordingCompletedThisFrame(false), ffs_recordingEnabled(false),
        ffs_manualRecordingControl(false), ffs_recordedCallCount(0), ffs_bufferState(0) {}

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
        fprintf(f, "dip=%d,%d,%d,%d,%d,%d\n", pipeline.dipScene0, pipeline.dipScene1,
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
        fprintf(f, "bufferState=%d\n", ffs_bufferState);

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
                    pipeline.dipScene0=i0; pipeline.dipScene1=i1; pipeline.dipOffscreen=i2;
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
                } else if (sscanf(line, "bufferState=%d", &ffs_bufferState) == 1) {
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
        DIAG_DIFF_INT(dipScene1, "dipScene1");
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
        FFS_DIFF_INT(ffs_bufferState, "bufferState");

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
                "hasParamH", "disableParallax", "hasParamX",  // 26-28
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

    // N-1: Read recording buffer for diagnostic snapshot
    auto& fb = FixedFunctionShader::getRecordingBuffer();
    snap.ffs_bufferState = (int)fb.state;

    // Camera and eye position
    snap.eyePosX = DistantLand::s_staging.eyePos.x;
    snap.eyePosY = DistantLand::s_staging.eyePos.y;
    snap.eyePosZ = DistantLand::s_staging.eyePos.z;
    snap.sceneLightsTotal = (int)fb.sceneLights.size();

    // Read recorded calls from the buffer
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
    // Capture complete device state snapshot for async replay (no assumptions about prior state)
    deviceState = g_deviceState;
    simulationTime = DistantLand::s_staging.simulationTime;
    windVec[0] = DistantLand::s_staging.windVec[0];
    windVec[1] = DistantLand::s_staging.windVec[1];

    // Capture sampler states for stages 0-1 (Morrowind-bound textures)
    // from proxy shadow state instead of per-draw GetSamplerState() calls.
    // Stages 2+ are HLSL-specific textures bound by MGE XE with known sampler states.
    for (int stage = 0; stage < 2; ++stage) {
        samplerStates[stage].addressU = deviceState.samplerAddressU[stage];
        samplerStates[stage].addressV = deviceState.samplerAddressV[stage];
        samplerStates[stage].captured = true;
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

        ObjectSpaceBBox objBBox;
        bool cacheHit = false;
        {
            std::lock_guard<std::mutex> lock(bboxCacheMutex);
            auto it = bboxCache.find(key);
            if (it != bboxCache.end()) {
                objBBox = it->second;
                cacheHit = true;
            }
        }
        if (cacheHit) {
            // Cache hit — cheap world-space transform
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
    expectedState.alphaBlendEnable = deviceState.alphaBlendEnable;
    expectedState.alphaTestEnable = deviceState.alphaTestEnable;
    expectedState.zEnable = deviceState.zEnable;
    expectedState.zWriteEnable = deviceState.zWriteEnable;
    expectedState.fogEnable = deviceState.fogEnable;
    expectedState.cullMode = deviceState.cullMode;
    expectedState.srcBlend = deviceState.srcBlend;
    expectedState.destBlend = deviceState.destBlend;
    expectedState.captured = true;

    // Debug-only: capture additional state for validation
    if (ImGuiManager::GetStateLeakDetection()) {
        expectedState.depthBias = deviceState.depthBias;
        expectedState.slopeScaledDepthBias = deviceState.slopeScaleDepthBias;
        for (int s = 0; s < 2; ++s) {
            expectedState.samplerAddressU[s] = deviceState.samplerAddressU[s];
            expectedState.samplerAddressV[s] = deviceState.samplerAddressV[s];
        }
        for (int s = 0; s < 8; ++s) {
            expectedState.textures[s] = nullptr;
            device->GetTexture(s, &expectedState.textures[s]);
            if (expectedState.textures[s]) expectedState.textures[s]->Release();
        }
    }
}

// computeBoundingBox moved to hiz_culling.cpp
