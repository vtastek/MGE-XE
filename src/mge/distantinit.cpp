
#include "proxydx/d3d8header.h"
#include "proxydx/devicelock.h"
#include "support/log.h"
#include "configuration.h"
#include "distantland.h"
#include "distantshader.h"
#include "dlformat.h"
#include "postshaders.h"
#include "morrowindbsa.h"
#include "msocclient.h"
#include "mwbridge.h"
#include "mgeversion.h"
#include "scenegraph_geometry_cache.h"
#include "statusoverlay.h"
#include "renderprocess.h"
#include "ipc/dlshare.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>



using std::string;
using std::string_view;
using std::vector;

bool DistantLand::ready = false;
bool DistantLand::isRenderCached = false;
bool DistantLand::isPPLActive = false;
bool DistantLand::cacheOpaqueMode = false;
bool DistantLand::cacheOnlyMode = false;
bool DistantLand::earlyWalkedCache = false;
bool DistantLand::renderThreadJobKicked = false;
bool DistantLand::earlyForgeKickoff = false;
bool DistantLand::earlyCulledGrass = false;
// Branch A (lead-in shrink): boot HOST-CULL-ONLY. The host's two-phase Hi-Z GPU cull owns
// occlusion, which lets frameSetupEarly skip the ~1.4ms main-thread MSOC classify entirely —
// the single largest serial item between MW's physics and the produce kick. VK_SCROLL flips
// back to the MSOC path for the A/B.
//
// ROUTE A MEASUREMENT (2026-07-20): booted OFF to price the other side of that trade. Skipping
// the classify does save its ~1.4ms, but the classify is also our DISCOVERY feed: without it
// liveDrawBuild is forced false (distantland.cpp) and onFrameReady runs the full refresh walk
// — measured at 7.05ms over 12,572 entries ([fse] walk=, [gc] visited=). We were re-walking the
// scene to rediscover what MW's own CullShow traversal already enumerated. Trading 1.4 for 7 is
// the wrong direction; this flip measures the other side. See tasks/cache-walk-offthread.md.
bool DistantLand::hostCullOnly = false;
// MW-ONLY-UI: boot OFF. Suppressing landscape has no visible cost, but its frame-time payoff was
// never isolated, and the whole suppression line measured only ~0.75ms while costing smoke and
// sorted alpha (tasks/forge-world-particles.md) — not enough to justify shipping an engine-state
// change by default. Opt in per root from the Forge Dev imgui panel.
int  DistantLand::mwWorldSuppress = 0;
std::vector<D3DXVECTOR4> DistantLand::reflectionWaterRects;
bool DistantLand::reflWaterCullActive = true;
bool DistantLand::reflGateWanted = false;
bool DistantLand::reflStaticsWanted = false;
bool DistantLand::boxOccluderDebug = false;
int  DistantLand::debugOverlayCycle = 0;
bool DistantLand::debugReflFrustum = false;
std::vector<DistantLand::ReflCacheDbgBox> DistantLand::reflCacheDbg;
D3DXMATRIX DistantLand::reflDbgViewProj;
bool DistantLand::reflDbgValid = false;
bool DistantLand::reflVisible = false;
D3DXMATRIX  DistantLand::reflCullViewProj;
D3DXMATRIX  DistantLand::reflCullProj;
D3DXVECTOR4 DistantLand::reflCullViewSphere;
VisibleSet<StlVector> DistantLand::reflectionSurvivors;
int DistantLand::numWaterVerts, DistantLand::numWaterTris;

IDirect3DDevice9* DistantLand::device;
ID3DXEffect* DistantLand::effect;
ID3DXEffect* DistantLand::effectShadow;
ID3DXEffect* DistantLand::effectDepth;
ID3DXEffectPool* DistantLand::effectPool;
IDirect3DVertexDeclaration9* DistantLand::LandDecl;
IDirect3DVertexDeclaration9* DistantLand::StaticDecl;
IDirect3DVertexDeclaration9* DistantLand::WaterDecl;
IDirect3DVertexDeclaration9* DistantLand::GrassDecl;

VendorSpecificRendering DistantLand::vsr;

IPC::Client DistantLand::ipcClient;
std::vector<DistantLand::DynamicVisGroup> DistantLand::dynamicVisGroups;
void* DistantLand::lastDistantVisCell;
bool DistantLand::isDistantLandLoaded = false;

VisibleSet<StlVector> DistantLand::visLand;
VisibleSet<StlVector> DistantLand::visDistant;
VisibleSet<StlVector> DistantLand::visGrass;
VisibleSet<StlVector> DistantLand::visDistantSurvivors;

VisibleSet<IpcClientVector> DistantLand::visLandShared;
VisibleSet<IpcClientVector> DistantLand::visDistantShared;
VisibleSet<IpcClientVector> DistantLand::visGrassShared;
VisibleSet<IpcClientVector> DistantLand::visExtraShared;
IPC::VecView<IPC::DynVisFlag> DistantLand::dynVisFlagsShared;
IPC::VecView<OcclusionMask::MaskChunk> DistantLand::maskBlobShared;

IPC::VecId DistantLand::visLandSharedId = IPC::InvalidVector;
IPC::VecId DistantLand::visDistantSharedId = IPC::InvalidVector;
IPC::VecId DistantLand::visGrassSharedId = IPC::InvalidVector;
IPC::VecId DistantLand::visExtraSharedId = IPC::InvalidVector;
IPC::VecId DistantLand::dynVisFlagsSharedId = IPC::InvalidVector;
IPC::VecId DistantLand::maskBlobSharedId = IPC::InvalidVector;

vector<DistantLand::RecordedState> DistantLand::recordMW;
vector<DistantLand::RecordedState> DistantLand::recordSky;
vector< std::pair<const RenderMesh*, int> > DistantLand::batchedGrass;
std::unordered_map<IDirect3DVertexBuffer9*, DistantLand::LandMeshCache> DistantLand::landMeshes;
std::vector<std::uint8_t> DistantLand::msocOccluded;

IDirect3DTexture9* DistantLand::texWorldColour, *DistantLand::texWorldNormals, *DistantLand::texWorldDetail;
IDirect3DTexture9* DistantLand::texDepthFrame;
IDirect3DSurface9* DistantLand::surfDepthDepth;
IDirect3DTexture9* DistantLand::texDistantBlend;
IDirect3DTexture9* DistantLand::texReflection;
IDirect3DSurface9* DistantLand::surfReflectionZ;
IDirect3DVolumeTexture9* DistantLand::texWater;
IDirect3DVertexBuffer9* DistantLand::vbWater;
IDirect3DIndexBuffer9* DistantLand::ibWater;
IDirect3DVertexBuffer9* DistantLand::vbGrassInstances;

IDirect3DVertexBuffer9* DistantLand::vbWaterLod;
IDirect3DIndexBuffer9* DistantLand::ibWaterLod;
int DistantLand::numWaterLodVerts;
std::vector<DistantLand::WaterLodLevel> DistantLand::waterLodLevels;
bool DistantLand::waterLodMeshOn = true;
float DistantLand::waterWaveAmp = 32.0f;
float DistantLand::waterWaveLen = 3000.0f;
float DistantLand::waterWaveSpeed = 1.0f;
float DistantLand::waterCrestSpread = 0.5f;

IDirect3DTexture9* DistantLand::texRain;
IDirect3DTexture9* DistantLand::texRipples;
IDirect3DTexture9* DistantLand::texRippleBuffer;
IDirect3DSurface9* DistantLand::surfRain;
IDirect3DSurface9* DistantLand::surfRipples;
IDirect3DSurface9* DistantLand::surfRippleBuffer;
IDirect3DVertexBuffer9* DistantLand::vbWaveSim;
IDirect3DVertexBuffer9* DistantLand::vbFoamSim;

IDirect3DTexture9* DistantLand::texFlow;
bool DistantLand::waterFlowDebugOn = true;
int  DistantLand::waterFlowDebugView = 0;
float DistantLand::waterFlowScroll = 0.4f;
float DistantLand::waterFlowSeaSpeed = 1.0f;
float DistantLand::waterFlowCycleUV = 4.0f;
float DistantLand::waterFlowSeaRefract = 1.0f;
float DistantLand::waterFlowWarp = 64.0f;

IDirect3DTexture9* DistantLand::texFoamP_A[DistantLand::foamCascades];
IDirect3DTexture9* DistantLand::texFoamP_B[DistantLand::foamCascades];
IDirect3DTexture9* DistantLand::texFoamField[DistantLand::foamCascades];
IDirect3DTexture9* DistantLand::texFoam[DistantLand::foamCascades];
IDirect3DSurface9* DistantLand::surfFoamP_A[DistantLand::foamCascades];
IDirect3DSurface9* DistantLand::surfFoamP_B[DistantLand::foamCascades];
IDirect3DSurface9* DistantLand::surfFoamField[DistantLand::foamCascades];
IDirect3DSurface9* DistantLand::surfFoam[DistantLand::foamCascades];
IDirect3DTexture9* DistantLand::texFoamUV_A[DistantLand::foamCascades];
IDirect3DTexture9* DistantLand::texFoamUV_B[DistantLand::foamCascades];
IDirect3DSurface9* DistantLand::surfFoamUV_A[DistantLand::foamCascades];
IDirect3DSurface9* DistantLand::surfFoamUV_B[DistantLand::foamCascades];
int   DistantLand::foamLastXpos[DistantLand::foamCascades];
int   DistantLand::foamLastYpos[DistantLand::foamCascades];
float DistantLand::foamOriginC[DistantLand::foamCascades][2];
bool DistantLand::foamSimReset = true;
bool DistantLand::waterFoamOn = true;
bool DistantLand::foamDebugView = false;
// Single-carrier foam tuning (cascades collapsed to one). Tune live via the NUMPAD8 cycle.
float DistantLand::foamFlowForce[foamCascades] = { 1.5f };
float DistantLand::foamDecay[foamCascades]     = { 0.94f };
float DistantLand::foamPressure[foamCascades]  = { 0.1f };
float DistantLand::foamScale[foamCascades]     = { 6.7f };
// Two-layer foam defaults: 32u fbm cells (fine streaks), FoamSpeed 1.0 (far-layer advect rate
// = ×river flow), erode threshold 0.35 (crisp edge), far-layer strength 1.0.
float DistantLand::foamDetailTile     = 160.0f;
float DistantLand::foamDetailSpeed    = 0.6f;
float DistantLand::foamErodeThreshold = 0.98f;
float DistantLand::foamFarAmount      = 1.0f;
float DistantLand::foamMix            = 0.5f;
float DistantLand::foamGaussRadius    = 4.0f;
float DistantLand::foamMinDensity     = 0.4f;
float DistantLand::foamUVDecay        = 0.97f;
float DistantLand::foamSimSpeed       = 1.0f;
float DistantLand::foamVortGain        = 4.0f;
float DistantLand::foamFineScale       = 5.0f;
float DistantLand::foamFineAmt         = 0.6f;
float DistantLand::foamCoarseScale     = 3.0f;
float DistantLand::foamCoarseAmt       = 0.5f;

IDirect3DTexture9* DistantLand::texShadow;
IDirect3DTexture9* DistantLand::texSoftShadow;
IDirect3DSurface9* DistantLand::surfShadowZ;
IDirect3DVertexBuffer9* DistantLand::vbFullFrame;
IDirect3DVertexBuffer9* DistantLand::vbClipCube;

D3DXMATRIX DistantLand::mwView, DistantLand::mwProj;
D3DXMATRIX DistantLand::smView[2], DistantLand::smProj[2];
D3DXMATRIX DistantLand::smViewproj[2];
D3DXVECTOR4 DistantLand::eyeVec, DistantLand::eyePos;
D3DXVECTOR4 DistantLand::sunVec, DistantLand::sunPos;
float DistantLand::sunVis;
RGBVECTOR DistantLand::sunCol, DistantLand::sunAmb, DistantLand::ambCol;
RGBVECTOR DistantLand::nearFogCol, DistantLand::horizonCol;
RGBVECTOR DistantLand::atmOutscatter(0.07, 0.36, 0.76);
RGBVECTOR DistantLand::atmInscatter(0.25, 0.38, 0.48);
D3DXVECTOR4 DistantLand::atmSkylightScatter(0.4456, 0.6194, 1.0, 0.44);
float DistantLand::fogStart, DistantLand::fogEnd;
float DistantLand::fogExpStart, DistantLand::fogExpDivisor;
float DistantLand::fogNearStart, DistantLand::fogNearEnd;
float DistantLand::nearViewRange;
float DistantLand::windScaling, DistantLand::niceWeather;
float DistantLand::lightSunMult, DistantLand::lightAmbMult;

D3DXHANDLE DistantLand::ehRcpRes;
D3DXHANDLE DistantLand::ehShadowRcpRes;
D3DXHANDLE DistantLand::ehWorld;
D3DXHANDLE DistantLand::ehView;
D3DXHANDLE DistantLand::ehProj;
D3DXHANDLE DistantLand::ehShadowViewproj;
D3DXHANDLE DistantLand::ehVertexBlendState;
D3DXHANDLE DistantLand::ehVertexBlendPalette;
D3DXHANDLE DistantLand::ehBoneMatrices;
D3DXHANDLE DistantLand::ehAlphaRef;
D3DXHANDLE DistantLand::ehMaterialAlpha;
D3DXHANDLE DistantLand::ehHasAlpha;
D3DXHANDLE DistantLand::ehHasBones;
D3DXHANDLE DistantLand::ehHasVCol;
D3DXHANDLE DistantLand::ehTex0;
D3DXHANDLE DistantLand::ehTex1;
D3DXHANDLE DistantLand::ehTex2;
D3DXHANDLE DistantLand::ehTex3;
D3DXHANDLE DistantLand::ehTex4;
D3DXHANDLE DistantLand::ehTex5;
D3DXHANDLE DistantLand::ehEyePos;
D3DXHANDLE DistantLand::ehFootPos;
D3DXHANDLE DistantLand::ehSunCol;
D3DXHANDLE DistantLand::ehSunAmb;
D3DXHANDLE DistantLand::ehSunVec;
D3DXHANDLE DistantLand::ehSunVecView;
D3DXHANDLE DistantLand::ehSunPos;
D3DXHANDLE DistantLand::ehSunVis;
D3DXHANDLE DistantLand::ehOutscatter;
D3DXHANDLE DistantLand::ehInscatter;
D3DXHANDLE DistantLand::ehSkyScatterFar;
D3DXHANDLE DistantLand::ehSkyCol;
D3DXHANDLE DistantLand::ehFogColNear;
D3DXHANDLE DistantLand::ehFogColFar;
D3DXHANDLE DistantLand::ehFogStart;
D3DXHANDLE DistantLand::ehFogRange;
D3DXHANDLE DistantLand::ehFogNearStart;
D3DXHANDLE DistantLand::ehFogNearRange;
D3DXHANDLE DistantLand::ehNearViewRange;
D3DXHANDLE DistantLand::ehStaticNearCull;
D3DXHANDLE DistantLand::ehShadowReflMult;
D3DXHANDLE DistantLand::ehLandNearCull;
D3DXHANDLE DistantLand::ehReflWaterClip;
D3DXHANDLE DistantLand::ehLightData, DistantLand::ehLightDataParams, DistantLand::ehLightIndices, DistantLand::ehTexLightView;
D3DXHANDLE DistantLand::ehWindVec;
D3DXHANDLE DistantLand::ehNiceWeather;
D3DXHANDLE DistantLand::ehTime;
D3DXHANDLE DistantLand::ehRippleOrigin;
D3DXHANDLE DistantLand::ehWaveHeight;
D3DXHANDLE DistantLand::ehFlow;
D3DXHANDLE DistantLand::ehFlowTransform;
D3DXHANDLE DistantLand::ehFlowWeight;
D3DXHANDLE DistantLand::ehFlowScroll;
D3DXHANDLE DistantLand::ehFlowSeaSpeed;
D3DXHANDLE DistantLand::ehFlowCycleUV;
D3DXHANDLE DistantLand::ehFlowSeaRefract;
D3DXHANDLE DistantLand::ehFlowDebugView;
D3DXHANDLE DistantLand::ehFlowWarp;
D3DXHANDLE DistantLand::ehWaveAmp;
D3DXHANDLE DistantLand::ehWaveLen;
D3DXHANDLE DistantLand::ehWaveSpeed;
D3DXHANDLE DistantLand::ehCrestSpread;
D3DXHANDLE DistantLand::ehFoamParticles;
D3DXHANDLE DistantLand::ehFoamFieldIn;
D3DXHANDLE DistantLand::ehFoamOrigin;
D3DXHANDLE DistantLand::ehFoamShift;
D3DXHANDLE DistantLand::ehFoamFieldShift;
D3DXHANDLE DistantLand::ehFoamPlayer;
D3DXHANDLE DistantLand::ehFoamParams;
D3DXHANDLE DistantLand::ehFoamWorldRes;
D3DXHANDLE DistantLand::ehFoamAdvance;
D3DXHANDLE DistantLand::ehFoam0;
D3DXHANDLE DistantLand::ehFoamOrigin0;
D3DXHANDLE DistantLand::ehFoamWeight;
D3DXHANDLE DistantLand::ehFoamDetail;
D3DXHANDLE DistantLand::ehFoamFarAmount;
D3DXHANDLE DistantLand::ehFoamUVIn;
D3DXHANDLE DistantLand::ehFoamUVRate;
D3DXHANDLE DistantLand::ehFoamUVDecay;
D3DXHANDLE DistantLand::ehFoamUVTex;
D3DXHANDLE DistantLand::ehFoamGaussRadius;
D3DXHANDLE DistantLand::ehFoamMinDensity;
D3DXHANDLE DistantLand::ehFoamVortGain;
D3DXHANDLE DistantLand::ehFoamFineScale;
D3DXHANDLE DistantLand::ehFoamFineAmt;
D3DXHANDLE DistantLand::ehFoamCoarseScale;
D3DXHANDLE DistantLand::ehFoamCoarseAmt;

std::function<void(IDirect3DSurface9*)> DistantLand::captureScreenHandler = nullptr;
bool DistantLand::captureScreenWithUI;


struct MeshResources {
    IDirect3DVertexBuffer9* vb;
    IDirect3DIndexBuffer9* ib;
    IDirect3DTexture9* tex;

    MeshResources(IDirect3DVertexBuffer9* _vb, IDirect3DIndexBuffer9* _ib, IDirect3DTexture9* _tex) : vb(_vb), ib(_ib), tex(_tex) {}
};
static vector<MeshResources> meshCollectionLand;
static vector<MeshResources> meshCollectionStatics;

// Capture the full triangle mesh of a distant-land tile so it can be
// re-used to build a horizon-curtain occluder for the MSOC mask.
// Subsampling fights MGE-XE's ROAM tessellator (irregular, cache-
// optimized meshes don't subsample cleanly), so we keep the native
// geometry; horizon construction reads it back at render time.
//
// Called inside the Lock/Unlock window during initLandscape /
// initLandscapeClient. Caller supplies CPU-side pointers to already-
// staged vertex and index bytes (we do NOT read from the WRITEONLY
// lock region — that's pathologically slow on DXVK).
//
//   vbBytes    : pointer to `verts` * SIZEOFLANDVERT bytes, each
//                vertex being POSITION float3 + TEXCOORD short2.
//   ibBytes    : pointer to `faces` * (large ? 12 : 6) bytes of
//                triangle indices.
//   large      : true if indices are uint32, false if uint16.
//
// We keep only the float3 position per vertex (UVs discarded) and
// promote all indices to uint32 so the runtime emit path has a
// single format to handle.
static void captureLandMesh(
    IDirect3DVertexBuffer9* vb,
    const void* vbBytes, unsigned verts,
    const void* ibBytes, unsigned faces, bool large)
{
    if (!vb || !vbBytes || !ibBytes || verts < 3 || faces == 0) return;

    DistantLand::LandMeshCache entry;
    entry.positions.resize(verts);
    entry.indices.resize(static_cast<size_t>(faces) * 3);

    // Extract POSITION float3 from each vertex. Stride is SIZEOFLANDVERT
    // (16 bytes: 12 position + 4 texcoord); position is at offset 0.
    const auto* vbytes = static_cast<const char*>(vbBytes);
    for (unsigned i = 0; i < verts; ++i) {
        const auto* pos = reinterpret_cast<const float*>(vbytes + i * SIZEOFLANDVERT);
        entry.positions[i] = D3DXVECTOR3(pos[0], pos[1], pos[2]);
    }

    // Promote indices to uint32. For the small-tile path (16-bit
    // indices), widen in-place as we copy.
    if (large) {
        std::memcpy(entry.indices.data(), ibBytes,
                    static_cast<size_t>(faces) * 3 * sizeof(std::uint32_t));
    } else {
        const auto* src = static_cast<const std::uint16_t*>(ibBytes);
        for (size_t i = 0; i < entry.indices.size(); ++i) {
            entry.indices[i] = src[i];
        }
    }

    DistantLand::landMeshes.emplace(vb, std::move(entry));

    // One-time log so we can tell capture is firing at all.
    static bool loggedOnce = false;
    if (!loggedOnce) {
        loggedOnce = true;
        LOG::logline("-- MSOC capture: first land tile captured (verts=%u tris=%u large=%s)",
                     verts, faces, large ? "u32" : "u16");
    }
}




// Water plane vertex declaration
const D3DVERTEXELEMENT9 WaterElem[] = {
    {0, 0,  D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
    D3DDECL_END()
};

// World mesh vertex declaration
const D3DVERTEXELEMENT9 LandElem[] = {
    {0, 0,  D3DDECLTYPE_FLOAT3,  D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
    {0, 12, D3DDECLTYPE_SHORT2N, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},
    D3DDECL_END()
};

// Distant static vertex declaration
const D3DVERTEXELEMENT9 StaticElem[] = {
    {0, 0,  D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
    {0, 8,  D3DDECLTYPE_UBYTE4N,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL,   0},
    {0, 12, D3DDECLTYPE_D3DCOLOR,  D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR,    0},
    {0, 16, D3DDECLTYPE_FLOAT16_2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},
    D3DDECL_END()
};

// Instanced grass vertex declaration
const D3DVERTEXELEMENT9 GrassElem[] = {
    {0, 0,  D3DDECLTYPE_FLOAT16_4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
    {0, 8,  D3DDECLTYPE_UBYTE4N,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL,   0},
    {0, 12, D3DDECLTYPE_D3DCOLOR,  D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR,    0},
    {0, 16, D3DDECLTYPE_FLOAT16_2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},
    {1, 0,  D3DDECLTYPE_FLOAT4,    D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 1},
    {1, 16, D3DDECLTYPE_FLOAT4,    D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 2},
    {1, 32, D3DDECLTYPE_FLOAT4,    D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 3},
    D3DDECL_END()
};



// Called from msoc.dll when MSOC finishes verdict classification, before any display() calls.
// Stores the MSOC-culled visible set for use in renderDepthFromCache this frame.
static void __cdecl onVisibleGeom(void* const* shapes, const float* /*boundsXYZR*/, int count) {
    DistantLand::updateVisibleSet(shapes, count);
}

bool DistantLand::init() {
    if (ready) {
        return true;
    }
    if (!device) {
        return false;
    }

    LOG::logline(">> Starting Distant Land init");
    vsr.init(device);
    MGE::GeometryCache::init(device);
    BSA::init();

    if (Configuration.UseSharedMemory && !initIpc()) {
        return false;
    }

    if (!initShader()) {
        return false;
    }

    if (!FixedFunctionShader::init(device, effectPool)) {
        return false;
    }

    if (!PostShaders::init(device)) {
        return false;
    }

    if (!initDepth()) {
        return false;
    }

    if (!initShadow()) {
        return false;
    }

    if (!initWater()) {
        return false;
    }

    if (!initLandscape()) {
        return false;
    }

    if (!initDistantStaticsClient()) {
        return false;
    }

    if (!initGrass()) {
        return false;
    }

    // Probe msoc.dll for the CPU occlusion mask. Soft dependency:
    // returns silently if the plugin isn't installed. Must run after
    // the rest of distant-land is wired up so the resulting log banner
    // lands next to other distant-land init lines in mgeXE.log.
    MSOCClient::init();
    // Visible-geom callback: with the early classify (mwse_classifyMainSceneNow) this
    // fires at BeginScene(0) with the engine's current-frame drawn set, which
    // buildFrustumVisibleSet consumes as the MSOC-culled cache set.
    MSOCClient::registerVisibleGeomCallback(onVisibleGeom);

    MWBridge::get()->patchResolveDuringInit(&resolveDynamicVisGroups);

    LOG::logline("<< Completed Distant Land init");
    ready = true;
    isRenderCached = false;
    return true;
}

bool DistantLand::initIpc() {
    if (!IPC::initImports()) {
        LOG::logline("!! Disabling shared memory because required memory mapping APIs are not available");
        Configuration.UseSharedMemory = false;
        // we'll return success so we can continue on the non-IPC path
        return true;
    }

    if (!ipcClient.startServer("mgeHost64.exe")) {
        return false;
    }

    // allocate shared vectors that will be reused for the duration of the program
    auto maybeLandVec = ipcClient.allocVecBlocking<RenderMesh>(1, 200000, 1);
    if (!maybeLandVec.has_value()) {
        return false;
    }
    auto& landVec = maybeLandVec.value();
    visLandSharedId = landVec.id();
    visLandShared.SetVector((IpcClientVector(landVec)));

    auto maybeDistantVec = ipcClient.allocVecBlocking<RenderMesh>(1, 200000, 1);
    if (!maybeDistantVec.has_value()) {
        return false;
    }
    auto& distantVec = maybeDistantVec.value();
    visDistantSharedId = distantVec.id();
    visDistantShared.SetVector((IpcClientVector(distantVec)));

    // we force the maximum number of grass elements to always be resident in memory. this currently equates to 704 KiB of
    // grass memory compared to the standard window size of 64 KiB, but it allows us to avoid a bunch of copying when
    // rendering grass.
    auto maybeGrassVec = ipcClient.allocVecBlocking<RenderMesh>(MaxGrassElements, MaxGrassElements, MaxGrassElements);
    if (!maybeGrassVec.has_value()) {
        return false;
    }
    auto& grassVec = maybeGrassVec.value();
    visGrassSharedId = grassVec.id();
    visGrassShared.SetVector((IpcClientVector(grassVec)));

    auto maybeExtraVec = ipcClient.allocVecBlocking<RenderMesh>(1, 200000, 1);
    if (!maybeExtraVec.has_value()) {
        return false;
    }
    auto& extraVec = maybeExtraVec.value();
    visExtraSharedId = extraVec.id();
    visExtraShared.SetVector((IpcClientVector(extraVec)));

    auto maybeDynVisVec = ipcClient.allocVecBlocking<IPC::DynVisFlag>(1, 1000, 1);
    if (!maybeDynVisVec.has_value()) {
        return false;
    }
    auto& dynVisVec = maybeDynVisVec.value();
    dynVisFlagsSharedId = dynVisVec.id();
    dynVisFlagsShared = dynVisVec;

    // Occlusion-mask transfer vec (host-side cull). One resident window of
    // kBlobChunks × 64KB chunks — large enough for the biggest blob (AVX512
    // ZTile buffer ~96KB + header) while keeping the Vec reservation tiny (the
    // element is a 64KB chunk, not a byte, so maxSize*windowBytes stays small;
    // see occlusionmask.h). Allocated unconditionally; written only when
    // host-cull is enabled + supported.
    auto maybeMaskVec = ipcClient.allocVecBlocking<OcclusionMask::MaskChunk>(
        OcclusionMask::kBlobChunks, OcclusionMask::kBlobChunks, OcclusionMask::kBlobChunks);
    if (!maybeMaskVec.has_value()) {
        return false;
    }
    auto& maskVec = maybeMaskVec.value();
    maskBlobSharedId = maskVec.id();
    maskBlobShared = maskVec;

    // Present-seam: bring up the out-of-process Forge D3D12 renderer + the 9On12 seam
    // (gated by Configuration.UseRenderProcess; no-op otherwise). Runs here, under the
    // "...MGE XE..." loading bar, so the cold bring-up hides behind the bar instead of
    // stalling the first menu present. DistantLand::device is set at startup.
    RenderProcess::init(&ipcClient, DistantLand::device);

    return true;
}

bool DistantLand::reloadShaders() {
    LOG::logline(">> Distant Land reloading");
    if (!initShader()) {
        return false;
    }

    FixedFunctionShader::release();
    if (!FixedFunctionShader::init(device, effectPool)) {
        return false;
    }

    return true;
}

static const string shaderCoreModPrefix = "XE Mod";
static const string pathCoreShaders = "Data Files\\shaders\\core\\";
static const string pathCoreMods = "Data Files\\shaders\\core-mods\\";

struct CoreModInclude : public ID3DXInclude {
    vector<string> modsFound;
    std::optional<string> testSingleMod;

    STDMETHOD(Open)(D3DXINCLUDE_TYPE IncludeType, LPCSTR pFileName, LPCVOID pParentData, LPCVOID *ppData, UINT *pBytes) {
        string filename(pFileName), shaderPath = filename;
        bool isMod = false;
        char *buffer = nullptr;
        HANDLE h;

        // Check if it uses the core shader path prefix, if not, add the prefix
        if (filename.compare(0, pathCoreShaders.length(), pathCoreShaders) != 0) {
            shaderPath = pathCoreShaders + filename;
        }

        if (!testSingleMod) {
            // Check if this file is moddable, and if a core-mod exists, use its path
            if (filename.substr(0, shaderCoreModPrefix.length()) == shaderCoreModPrefix) {
                string modShaderPath = pathCoreMods + filename;
                if (GetFileAttributes(modShaderPath.c_str()) != INVALID_FILE_ATTRIBUTES) {
                    isMod = true;
                    shaderPath = modShaderPath;
                }
            }
        }
        else {
            // Only load the specified mod for testing, ignoring others
            if (testSingleMod.value() == filename) {
                isMod = true;
                shaderPath = pathCoreMods + filename;
            }
        }

        // Read file contents for the effect compiler
        h = CreateFile(shaderPath.c_str(), GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
        if (h != INVALID_HANDLE_VALUE) {
            DWORD bytesRead, bufferSize = GetFileSize(h, NULL);

            buffer = new char[bufferSize];
            ReadFile(h, buffer, bufferSize, &bytesRead, 0);
            CloseHandle(h);

            if (isMod) {
                modsFound.push_back(filename);
            }

            *ppData = buffer;
            *pBytes = bufferSize;
            return S_OK;
        }
        return E_FAIL;
    }

    STDMETHOD(Close)(LPCVOID pData) {
        char *buffer = (char*)(pData);
        delete [] buffer;
        return S_OK;
    }
};

static void logShaderError(ID3DXBuffer* errors) {
    if (errors) {
        LOG::write("!! Shader compile errors:\n");
        LOG::write(reinterpret_cast<const char*>(errors->GetBufferPointer()));
        LOG::write("\n");
        errors->Release();
    }
    LOG::flush();
}

static bool createCoreEffectWithMods(const char *name, IDirect3DDevice9* device, vector<D3DXMACRO>& features, ID3DXEffectPool *effectPool, ID3DXEffect **pEffect, bool reportMods) {
    string path = pathCoreShaders + name;
    ID3DXBuffer* errors;
    CoreModInclude includer;
    HRESULT hr;

    // Attempt to compile with core mods first
    hr = D3DXCreateEffectFromFile(device, path.c_str(), &*features.begin(), &includer, D3DXSHADER_OPTIMIZATION_LEVEL3|D3DXFX_LARGEADDRESSAWARE, effectPool, pEffect, &errors);
    if (hr == D3D_OK) {
        if (reportMods) {
            for (auto& m : includer.modsFound) {
                LOG::logline("-- Using core mod %s", m.c_str());
            }
        }
        return true;
    } else {
        LOG::logline("!! Core shader %s failed to compile with core-mods. All core-mods are disabled. Checking for errors...", name);
        StatusOverlay::setStatus("Shader core mod error. Core mods are disabled for this session. Check mgeXE.log for error details.", StatusOverlay::PriorityError);
        if (errors) {
            errors->Release();
        }
    }

    // Individually test each core mod for errors
    auto modsFound = includer.modsFound;
    for(const auto& mod : modsFound) {
        ID3DXEffect *testEffect;
        includer.testSingleMod = mod;

        hr = D3DXCreateEffectFromFile(device, path.c_str(), &*features.begin(), &includer, D3DXSHADER_OPTIMIZATION_LEVEL0|D3DXFX_LARGEADDRESSAWARE, effectPool, &testEffect, &errors);
        if (hr == D3D_OK) {
            testEffect->Release();
        }
        else {
            LOG::logline("!! Shader core mod %s%s failed to compile. Disable or remove it until it is fixed.", pathCoreMods.c_str(), mod.c_str());
            logShaderError(errors);
        }
    }

    // Fallback to compiling without core mods
    hr = D3DXCreateEffectFromFile(device, path.c_str(), &*features.begin(), 0, D3DXSHADER_OPTIMIZATION_LEVEL3|D3DXFX_LARGEADDRESSAWARE, effectPool, pEffect, &errors);
    if (hr == D3D_OK) {
        return true;
    } else {
        LOG::logline("!! Core shader %s failed to compile. Do not replace core shaders. Reinstall MGE XE.", name);
        logShaderError(errors);
    }
    return false;
}

static const D3DXMACRO macroExpFog = { "USE_EXPFOG", "" };
static const D3DXMACRO macroScattering = { "USE_SCATTERING", "" };
static const D3DXMACRO macroFilterReflection = { "FILTER_WATER_REFLECTION", "" };
static const D3DXMACRO macroDynamicRipples = { "DYNAMIC_RIPPLES", "" };
static const D3DXMACRO macroWaterFlowMap = { "WATER_FLOW_MAP", "" };
static const D3DXMACRO macroWaterLodMesh = { "WATER_LOD_MESH", "" };
static const D3DXMACRO macroWaterFoam = { "WATER_FOAM", "" };
static const D3DXMACRO macroTerminator = { 0, 0 };

bool DistantLand::initShader() {
    vector<D3DXMACRO> features;
    HRESULT hr;

    // Disable exponential fog if distant land is initially off
    if (~Configuration.MGEFlags & USE_DISTANT_LAND) {
        Configuration.MGEFlags &= ~(EXP_FOG | USE_ATM_SCATTER);
    }

    // Set shader defines corresponding to required features
    if (Configuration.MGEFlags & EXP_FOG) {
        features.push_back(macroExpFog);

        // Requires exp. fog
        if (Configuration.MGEFlags & USE_ATM_SCATTER) {
            features.push_back(macroScattering);
        }
    }
    if (Configuration.MGEFlags & BLUR_REFLECTIONS) {
        features.push_back(macroFilterReflection);
    }
    if (Configuration.MGEFlags & DYNAMIC_RIPPLES) {
        features.push_back(macroDynamicRipples);
    }
    if (Configuration.UseWaterFlowMap) {
        features.push_back(macroWaterFlowMap);
        // 3D waves are meaningless without the flow direction, so the world-snapped
        // LOD mesh + flow-steered crest displacement rides the same gate.
        features.push_back(macroWaterLodMesh);
        // World-anchored hybrid particle foam: the flow map IS the foam sim's velocity
        // field, so the foam rides the same gate. The sim only runs when DYNAMIC_RIPPLES
        // is also active (it reuses the wave-sim fullscreen quad); the shader sample is a
        // harmless no-op (texFoam stays cleared) otherwise.
        features.push_back(macroWaterFoam);
    }
    features.push_back(macroTerminator);

    if (!effectPool) {
        hr = D3DXCreateEffectPool(&effectPool);
        if (hr != D3D_OK) {
            LOG::logline("!! Effect pool creation failure");
            return false;
        }
    }

    if (!createCoreEffectWithMods("XE Main.fx", device, features, effectPool, &effect, true)) {
        return false;
    }

    ehRcpRes = effect->GetParameterByName(0, "rcpRes");
    ehShadowRcpRes = effect->GetParameterByName(0, "shadowRcpRes");
    ehWorld = effect->GetParameterByName(0, "world");
    ehView = effect->GetParameterByName(0, "view");
    ehProj = effect->GetParameterByName(0, "proj");
    ehShadowViewproj = effect->GetParameterByName(0, "shadowViewProj");
    ehVertexBlendState = effect->GetParameterByName(0, "vertexBlendState");
    ehVertexBlendPalette = effect->GetParameterByName(0, "vertexBlendPalette");
    ehBoneMatrices = effect->GetParameterByName(0, "boneMatrices");
    ehAlphaRef = effect->GetParameterByName(0, "alphaRef");
    ehMaterialAlpha = effect->GetParameterByName(0, "materialAlpha");
    ehHasAlpha = effect->GetParameterByName(0, "hasAlpha");
    ehHasBones = effect->GetParameterByName(0, "hasBones");
    ehHasVCol = effect->GetParameterByName(0, "hasVCol");
    ehTex0 = effect->GetParameterByName(0, "tex0");
    ehTex1 = effect->GetParameterByName(0, "tex1");
    ehTex2 = effect->GetParameterByName(0, "tex2");
    ehTex3 = effect->GetParameterByName(0, "tex3");
    ehEyePos = effect->GetParameterByName(0, "eyePos");
    ehFootPos = effect->GetParameterByName(0, "footPos");
    ehSunCol = effect->GetParameterByName(0, "sunCol");
    ehSunAmb = effect->GetParameterByName(0, "sunAmb");
    ehSunVec = effect->GetParameterByName(0, "sunVec");
    ehSunVecView = effect->GetParameterByName(0, "sunVecView");
    ehSunPos = effect->GetParameterByName(0, "sunPos");
    ehSunVis = effect->GetParameterByName(0, "sunVis");
    ehSkyCol = effect->GetParameterByName(0, "skyCol");
    ehFogColNear = effect->GetParameterByName(0, "fogColNear");
    ehFogColFar = effect->GetParameterByName(0, "fogColFar");
    ehFogStart = effect->GetParameterByName(0, "fogStart");
    ehFogRange = effect->GetParameterByName(0, "fogRange");
    ehFogNearStart = effect->GetParameterByName(0, "nearFogStart");
    ehFogNearRange = effect->GetParameterByName(0, "nearFogRange");
    ehNearViewRange = effect->GetParameterByName(0, "nearViewRange");
    ehStaticNearCull = effect->GetParameterByName(0, "staticNearCull");
    ehShadowReflMult = effect->GetParameterByName(0, "shadowReflMult");
    ehLandNearCull = effect->GetParameterByName(0, "landNearCull");
    ehReflWaterClip = effect->GetParameterByName(0, "reflWaterClipPlane");
    ehLightData = effect->GetParameterByName(0, "texLightData");
    ehLightDataParams = effect->GetParameterByName(0, "lightDataParams");
    ehLightIndices = effect->GetParameterByName(0, "lightIndices");
    ehTexLightView = effect->GetParameterByName(0, "texLightView");
    ehWindVec = effect->GetParameterByName(0, "windVec");
    ehNiceWeather = effect->GetParameterByName(0, "niceWeather");
    ehTime = effect->GetParameterByName(0, "time");

    D3DVIEWPORT9 vp;
    device->GetViewport(&vp);
    float rcpres[2] = { 1.0f / vp.Width, 1.0f / vp.Height };
    effect->SetFloatArray(ehRcpRes, rcpres, 2);
    effect->SetFloat(ehShadowRcpRes, 1.0f / Configuration.DL.ShadowResolution);

    if (!createCoreEffectWithMods("XE Shadowmap.fx", device, features, effectPool, &effectShadow, false)) {
        return false;
    }
    if (!createCoreEffectWithMods("XE Depth.fx", device, features, effectPool, &effectDepth, false)) {
        return false;
    }

    // Atmosphere scattering specific parameters
    if (Configuration.MGEFlags & USE_ATM_SCATTER) {

        ehOutscatter = effect->GetParameterByName(0, "outscatter");
        ehInscatter = effect->GetParameterByName(0, "inscatter");
        ehSkyScatterFar = effect->GetParameterByName(0, "skyScatterColFar");

        // Mark moon geometry for detection
        MWBridge::get()->markMoonNodes(kMoonTag);
    }
    else {
        ehOutscatter = 0;
        ehInscatter = 0;
        ehSkyScatterFar = 0;
    }

    // Dynamic ripples specific parameters
    if (Configuration.MGEFlags & DYNAMIC_RIPPLES) {
        ehTex4 = effect->GetParameterByName(0, "tex4");
        ehTex5 = effect->GetParameterByName(0, "tex5");
        ehRippleOrigin = effect->GetParameterByName(0, "rippleOrigin");
        ehWaveHeight = effect->GetParameterByName(0, "waveHeight");
    }

    // Water flow map parameters
    if (Configuration.UseWaterFlowMap) {
        ehFlow = effect->GetParameterByName(0, "texFlow");
        ehFlowTransform = effect->GetParameterByName(0, "flowMapTransform");
        ehFlowWeight = effect->GetParameterByName(0, "flowMapWeight");
        ehFlowScroll = effect->GetParameterByName(0, "flowScrollSpeed");
        ehFlowSeaSpeed = effect->GetParameterByName(0, "flowSeaSpeed");
        ehFlowCycleUV = effect->GetParameterByName(0, "flowCycleUV");
        ehFlowSeaRefract = effect->GetParameterByName(0, "flowSeaRefract");
        ehFlowDebugView = effect->GetParameterByName(0, "flowDebugView");
        ehFlowWarp = effect->GetParameterByName(0, "flowMapWarp");
        // Flow-steered crest displacement uniforms (WATER_LOD_MESH).
        ehWaveAmp = effect->GetParameterByName(0, "waveAmp");
        ehWaveLen = effect->GetParameterByName(0, "waveLen");
        ehWaveSpeed = effect->GetParameterByName(0, "waveSpeed");
        ehCrestSpread = effect->GetParameterByName(0, "crestSpread");
        // World-anchored particle foam sim uniforms (WATER_FOAM).
        ehFoamParticles = effect->GetParameterByName(0, "texFoamParticles");
        ehFoamFieldIn = effect->GetParameterByName(0, "texFoamFieldIn");
        ehFoamOrigin = effect->GetParameterByName(0, "foamOrigin");
        ehFoamShift = effect->GetParameterByName(0, "foamShiftPx");
        ehFoamFieldShift = effect->GetParameterByName(0, "foamFieldShift");
        ehFoamPlayer = effect->GetParameterByName(0, "foamPlayer");
        ehFoamParams = effect->GetParameterByName(0, "foamParams");
        ehFoamWorldRes = effect->GetParameterByName(0, "foamWorldRes");   // sim-side, set per cascade
        ehFoamAdvance = effect->GetParameterByName(0, "foamAdvance");      // per-microstep texel advance (≤1)
        // Consume side: both cascades bound at once (fine + coarse).
        ehFoam0 = effect->GetParameterByName(0, "texFoam0");          // single carrier
        ehFoamOrigin0 = effect->GetParameterByName(0, "foamOrigin0");
        ehFoamWeight = effect->GetParameterByName(0, "foamWeight");
        ehFoamDetail = effect->GetParameterByName(0, "foamDetail");
        ehFoamFarAmount = effect->GetParameterByName(0, "foamFarAmount");
        ehFoamUVIn = effect->GetParameterByName(0, "texFoamUVIn");    // sim ping-pong source
        ehFoamUVRate = effect->GetParameterByName(0, "foamUVRate");
        ehFoamUVDecay = effect->GetParameterByName(0, "foamUVDecay");
        ehFoamUVTex = effect->GetParameterByName(0, "texFoamUV");     // consume: advected offset field
        ehFoamGaussRadius = effect->GetParameterByName(0, "foamGaussRadius");
        ehFoamMinDensity = effect->GetParameterByName(0, "foamMinDensity");
        ehFoamVortGain = effect->GetParameterByName(0, "foamVortGain");
        ehFoamFineScale = effect->GetParameterByName(0, "foamFineScale");
        ehFoamFineAmt = effect->GetParameterByName(0, "foamFineAmt");
        ehFoamCoarseScale = effect->GetParameterByName(0, "foamCoarseScale");
        ehFoamCoarseAmt = effect->GetParameterByName(0, "foamCoarseAmt");
    }

    return true;
}

bool DistantLand::initDepth() {
    HRESULT hr;
    D3DVIEWPORT9 vp;

    // Set up depth frame texture, requires its own z-buffer (my card fails to support INTZ/DF24)
    device->GetViewport(&vp);

    hr = device->CreateTexture(vp.Width, vp.Height, 1, D3DUSAGE_RENDERTARGET, D3DFMT_R32F, D3DPOOL_DEFAULT, &texDepthFrame, NULL);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create depth frame render target");
        return false;
    }

    hr = device->CreateDepthStencilSurface(vp.Width, vp.Height, D3DFMT_D24X8, D3DMULTISAMPLE_NONE, 0, FALSE, &surfDepthDepth, NULL);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create depth target z-buffer");
        return false;
    }

    return true;
}

bool DistantLand::initWater() {
    HRESULT hr;
    const UINT reflRes = 1024;

    // Reflection render target
    hr = device->CreateTexture(reflRes, reflRes, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &texReflection, NULL);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create reflection render target");
        return false;
    }

    // Reflection Z-buffer
    hr = device->CreateDepthStencilSurface(reflRes, reflRes, D3DFMT_D24X8, D3DMULTISAMPLE_NONE, 0, TRUE, &surfReflectionZ, NULL);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create reflection Z buffer");
        return false;
    }

    // Water normals and geometry
    const int resS = (Configuration.MGEFlags & DYNAMIC_RIPPLES) ? 150 : 16;
    const int resT = (Configuration.MGEFlags & DYNAMIC_RIPPLES) ? 120 : 15;
    numWaterVerts = resS * resT + 1;
    numWaterTris = 2 * resS * resT - resS;

    hr = D3DXCreateVolumeTextureFromFile(device, "Data Files\\textures\\MGE\\water_NRM.dds", &texWater);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to load water texture");
        return false;
    }
    hr = device->CreateVertexDeclaration(WaterElem, &WaterDecl);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create water decl");
        return false;
    }
    hr = device->CreateVertexBuffer(numWaterVerts * 12, 0, 0, g_spikeForceDefaultPool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &vbWater, 0);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create water verts");
        return false;
    }
    hr = device->CreateIndexBuffer(numWaterTris * 6, 0, D3DFMT_INDEX16, g_spikeForceDefaultPool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &ibWater, 0);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create water indices");
        return false;
    }

    // Build radial water mesh
    D3DXVECTOR3* v;
    vbWater->Lock(0, 0, (void**)&v, 0);

    // Water plane lies at water level - 1.0 (not -4.0, which is the fog transition)
    const float dS = float(6.28318530717958647692 / resS);
    int s, t;
    float r, w = -1.0f;

    *v++ = D3DXVECTOR3(0, 0, w);
    for (t = 0; t < resT; ++t) {
        if (Configuration.MGEFlags & DYNAMIC_RIPPLES) {
            // Higher mesh density near player
            // The mesh requires density past 8192 units to cover the z discontinuity at distant land
            r = float(t) / float(resT);
            r = 9600.0f * (0.9f * powf(r, 3) + 0.1f * r);
            // Extend last ring past horizon
            if ((t+1) == resT) {
                r = 500000.0f;
            }
        } else {
            r = 4096.0f * (1.0f + t * t);
        }

        for (s = 0; s < resS; ++s) {
            *v++ = D3DXVECTOR3(r * cos(dS * s), r * sin(dS * s), w);
        }
    }

    vbWater->Unlock();

    USHORT* i;
    ibWater->Lock(0, 0, (void**)&i, 0);

    // Centre triangles
    for (s = 0; s < resS; ++s) {
        *i++ = 0;
        *i++ = 1 + s;
        *i++ = 1 + (s+1) % resS;
    }
    // Rings
    for (t = 1; t < resT; ++t) {
        for (s = 0; s < resS; ++s) {
            USHORT tbase = 1 + resS*(t-1), s2 = (s+1) % resS;
            *i++ = tbase + s;
            *i++ = resS + tbase + s;
            *i++ = tbase + s2;
            *i++ = resS + tbase+ s;
            *i++ = resS + tbase + s2;
            *i++ = tbase + s2;
        }
    }

    ibWater->Unlock();

    // World-snapped LOD water mesh (gated by the flow map; A/B vs radial at runtime).
    if (Configuration.UseWaterFlowMap) {
        if (!initWaterLodMesh()) {
            return false;
        }
        // World-anchored hybrid particle foam sim RTs (WATER_FOAM).
        if (!initFoamSim()) {
            return false;
        }
    }

    if (Configuration.MGEFlags & DYNAMIC_RIPPLES) {
        // Setup water simulation
        if (!initDynamicWaves()) {
            return false;
        }

        // Disable Morrowind generated ripples
        MWBridge::get()->toggleRipples(false);
    }

    return true;
}

// Build the world-snapped nested-grid (geo-clipmap) water mesh. L concentric LOD
// levels: level 0 is a solid m x m patch of the finest cells; levels 1..L-1 are
// square annulus rings whose central hole is the footprint of the next-finer level.
// Vertices are stored in LOCAL integer grid units (centred on 0); renderWaterPlane
// supplies cell size + world snap per level, so one static VB/IB serves every frame.
//
// Two crack fixes vs the naive nested rings:
//  A. Flexible interior trim. Each finer level snaps in finer steps than the coarse
//     hole, so the hole must shift by e=(ex,ey) in {0,1} coarse cells to nest exactly.
//     We bake 4 hole variants per ring level; the draw picks one from the eye parity.
//  B. Stitched transition row. A level's outer edge is retriangulated at the coarser
//     neighbour's spacing (2*cellSize) so the shared boundary carries no un-shared
//     midpoint vertex -> no T-junction. The outer ring is hole-independent, so the
//     stitched annulus is emitted into every trim variant (one draw per level).
// Heights come from WaterVS (continuous in world XY), so coincident boundary verts
// evaluate the same height -> watertight. Geometry-only; the shader is untouched.
bool DistantLand::initWaterLodMesh() {
    const int   L  = 6;        // LOD levels
    const float c0 = 128.0f;   // finest cell size (world units) — ~4 verts across a 512u flow cell
    const int   m  = 64;       // grid cells per side per level (even)
    const int   verts1D = m + 1;
    const int   half = m / 2;

    HRESULT hr;
    waterLodLevels.clear();
    waterLodLevels.reserve(L);

    // Full grid per level; hole/ring verts kept for trivial indexing (16-bit safe).
    numWaterLodVerts = L * verts1D * verts1D;

    hr = device->CreateVertexBuffer(numWaterLodVerts * 12, 0, 0, g_spikeForceDefaultPool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &vbWaterLod, 0);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create LOD water verts");
        return false;
    }

    // Vertices: local integer lattice in [-half, half] on both axes, plane at z=-1.
    D3DXVECTOR3* v;
    vbWaterLod->Lock(0, 0, (void**)&v, 0);
    for (int k = 0; k < L; ++k) {
        for (int gy = 0; gy <= m; ++gy) {
            for (int gx = 0; gx <= m; ++gx) {
                *v++ = D3DXVECTOR3(float(gx - half), float(gy - half), -1.0f);
            }
        }
    }
    vbWaterLod->Unlock();

    // Indices grow with the stitch + 4 trim variants, so build into a list then size
    // the IB exactly. ~0.8 MB managed; built once, static thereafter.
    std::vector<USHORT> indices;
    indices.reserve(400000);

    int vertBase = 0;
    auto vidx = [&](int gx, int gy) -> USHORT {
        return USHORT(vertBase + gy * verts1D + gx);
    };
    // Emit one triangle, forcing CCW winding in local XY. Height is added by the VS,
    // so signed area on the z=-1 plane is constant and decides orientation; this lets
    // the corner/edge helpers ignore reflection-induced winding flips.
    auto addTri = [&](int ax, int ay, int bx, int by, int cx, int cy) {
        long cross = long(bx - ax) * (cy - ay) - long(by - ay) * (cx - ax);
        USHORT ia = vidx(ax, ay), ib = vidx(bx, by), ic = vidx(cx, cy);
        if (cross < 0) std::swap(ib, ic);
        indices.push_back(ia); indices.push_back(ib); indices.push_back(ic);
    };
    // Regular fine cell: two triangles.
    auto addCell = [&](int cx, int cy) {
        addTri(cx, cy, cx + 1, cy, cx + 1, cy + 1);
        addTri(cx, cy, cx + 1, cy + 1, cx, cy + 1);
    };
    // Transition edge block: two outer cells collapsed so the outer boundary spans
    // one coarse edge A-C (odd outer vertex B dropped). (ax,ay)=outer-left vertex,
    // (tx,ty)=+1 tangent step along the edge, (nx,ny)=+1 inward normal.
    auto addEdgeBlock = [&](int ax, int ay, int tx, int ty, int nx, int ny) {
        int Ax = ax,                Ay = ay;
        int Cx = ax + 2 * tx,       Cy = ay + 2 * ty;
        int Apx = ax + nx,          Apy = ay + ny;
        int Bpx = ax + tx + nx,     Bpy = ay + ty + ny;
        int Cpx = ax + 2 * tx + nx, Cpy = ay + 2 * ty + ny;
        addTri(Ax, Ay, Cx, Cy, Bpx, Bpy);
        addTri(Ax, Ay, Bpx, Bpy, Apx, Apy);
        addTri(Cx, Cy, Cpx, Cpy, Bpx, Bpy);
    };
    // Transition corner: 2x2 block whose two grid-boundary edges are coarse.
    // (cgx,cgy)=outer corner vertex, (dx,dy)=inward signs. Six triangles tile the
    // block; the two odd boundary midpoints are never referenced on the outer edges.
    auto addCorner = [&](int cgx, int cgy, int dx, int dy) {
        int Ox = cgx,          Oy = cgy;            // outer corner
        int Bx = cgx + 2 * dx, By = cgy;            // coarse edge 1 far end
        int Tx = cgx,          Ty = cgy + 2 * dy;   // coarse edge 2 far end
        int Mx = cgx + dx,     My = cgy + dy;       // centre
        int Rx = cgx + 2 * dx, Ry = cgy + dy;
        int Sx = cgx + 2 * dx, Sy = cgy + 2 * dy;   // inner corner
        int Ux = cgx + dx,     Uy = cgy + 2 * dy;
        addTri(Ox, Oy, Bx, By, Mx, My);
        addTri(Ox, Oy, Mx, My, Tx, Ty);
        addTri(Bx, By, Rx, Ry, Mx, My);
        addTri(Rx, Ry, Sx, Sy, Mx, My);
        addTri(Mx, My, Sx, Sy, Ux, Uy);
        addTri(Mx, My, Ux, Uy, Tx, Ty);
    };
    // Stitched outer annulus: 4 corners + 4 transition edges between them. Hole-
    // independent, so identical in every trim variant.
    auto addOuterStitch = [&]() {
        addCorner(0, 0, +1, +1);
        addCorner(m, 0, -1, +1);
        addCorner(0, m, +1, -1);
        addCorner(m, m, -1, -1);
        for (int c = 2; c <= m - 4; c += 2) {
            addEdgeBlock(c, 0, 1, 0,  0,  1);   // bottom
            addEdgeBlock(c, m, 1, 0,  0, -1);   // top
            addEdgeBlock(0, c, 0, 1,  1,  0);   // left
            addEdgeBlock(m, c, 0, 1, -1,  0);   // right
        }
    };

    for (int k = 0; k < L; ++k) {
        const bool stitch = (k < L - 1);          // every level but the outermost
        const int  numVar = (k == 0) ? 1 : 4;     // level 0 is solid (no hole)

        WaterLodLevel lvl;
        lvl.cellSize    = c0 * float(1 << k);
        lvl.vertBase    = vertBase;
        lvl.vertCount   = verts1D * verts1D;
        lvl.numVariants = numVar;
        for (int i = 0; i < 4; ++i) { lvl.ibStart[i] = 0; lvl.triCount[i] = 0; }

        for (int vrt = 0; vrt < numVar; ++vrt) {
            const int ex = vrt & 1;
            const int ey = (vrt >> 1) & 1;
            // Flexible-trim hole = next-finer level's footprint, shifted by parity so
            // it nests exactly. Always inside [2, m-2), away from the stitched ring.
            const int holeLoX = (m / 4) + ex, holeHiX = (3 * m / 4) + ex;
            const int holeLoY = (m / 4) + ey, holeHiY = (3 * m / 4) + ey;

            const size_t startIdx = indices.size();
            lvl.ibStart[vrt] = int(startIdx);

            // Interior fine cells. Skip the outer ring on stitched levels (emitted by
            // addOuterStitch) and the central hole on ring levels.
            for (int cy = 0; cy < m; ++cy) {
                for (int cx = 0; cx < m; ++cx) {
                    if (stitch) {
                        const bool corner = (cx <= 1 || cx >= m - 2) && (cy <= 1 || cy >= m - 2);
                        const bool bedge  = (cy == 0 || cy == m - 1) && (cx >= 2 && cx <= m - 3);
                        const bool vedge  = (cx == 0 || cx == m - 1) && (cy >= 2 && cy <= m - 3);
                        if (corner || bedge || vedge) continue;   // stitched ring
                    }
                    if (k > 0 && cx >= holeLoX && cx < holeHiX && cy >= holeLoY && cy < holeHiY) {
                        continue;   // hole = next-finer level's footprint
                    }
                    addCell(cx, cy);
                }
            }

            if (stitch) {
                addOuterStitch();
            }

            lvl.triCount[vrt] = int((indices.size() - startIdx) / 3);
        }

        waterLodLevels.push_back(lvl);
        vertBase += verts1D * verts1D;
    }

    const int totalTris = int(indices.size() / 3);
    hr = device->CreateIndexBuffer(int(indices.size()) * 2, 0, D3DFMT_INDEX16, g_spikeForceDefaultPool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &ibWaterLod, 0);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create LOD water indices");
        return false;
    }
    USHORT* idx;
    ibWaterLod->Lock(0, 0, (void**)&idx, 0);
    std::copy(indices.begin(), indices.end(), idx);
    ibWaterLod->Unlock();

    LOG::logline("-- Water LOD mesh: %d levels, %d verts, %d tris (finest cell %.0fu, reach %.0fu)",
                 L, numWaterLodVerts, totalTris, c0, m * c0 * float(1 << (L - 1)));
    return true;
}

bool DistantLand::initDynamicWaves() {
    HRESULT hr;

    hr = device->CreateTexture(waveTexResolution, waveTexResolution, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A16B16G16R16F, D3DPOOL_DEFAULT, &texRain, NULL);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create rain simulation texture");
        return false;
    }
    texRain->GetSurfaceLevel(0, &surfRain);
    device->ColorFill(surfRain, 0, 0);

    hr = device->CreateTexture(waveTexResolution, waveTexResolution, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A16B16G16R16F, D3DPOOL_DEFAULT, &texRipples, NULL);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create ripple simulation texture");
        return false;
    }
    texRipples->GetSurfaceLevel(0, &surfRipples);
    device->ColorFill(surfRipples, 0, 0);

    hr = device->CreateTexture(waveTexResolution, waveTexResolution, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A16B16G16R16F, D3DPOOL_DEFAULT, &texRippleBuffer, NULL);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create ripple simulation texture");
        return false;
    }
    texRippleBuffer->GetSurfaceLevel(0, &surfRippleBuffer);
    device->ColorFill(surfRippleBuffer, 0, 0);

    // Vertex buffer for wave texture
    static float waveVertices[] = {
        /*     -0.5f,                    -0.5f,                                               0,1,   0,0,0,0,
                -0.5f,                    waveTexResolution-0.5f,                 0,1,   0,1,0,1,
                waveTexResolution-0.5f,    -0.5f,                                  0,1,   1,0,1,0,
                waveTexResolution-0.5f,    waveTexResolution-0.5f,    0,1,   1,1,1,1 */

        // Use only one tri over the whole texture to prevent simulation seams at tri edges
        // Rendering to a surface that is bound as a source texture updates the texture after
        // each primitive, causing artifacts to appear at primitive boundaries
        -waveTexResolution/2  -0.5f,    waveTexResolution/2  -0.5f,  0,  1,     -0.5, 0.5,     0,0,
        waveTexResolution        -0.5f,    2*waveTexResolution  -0.5f,  0,  1,      1.0, 2.0,      0,1,
        waveTexResolution        -0.5f,    -waveTexResolution    -0.5f,  0,  1,     1.0, -1.0,     1,1
    };

    void* vp;
    hr = device->CreateVertexBuffer(3 * 32, D3DUSAGE_WRITEONLY, fvfWave, D3DPOOL_DEFAULT, &vbWaveSim, 0);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create wave simulation vb");
        return false;
    }
    if (vbWaveSim->Lock(0, 0, (void**)&vp, 0) != D3D_OK) {
        LOG::logline("!! Failed to lock wave simulation vb");
        return false;
    }
    memcpy(vp, waveVertices, sizeof(waveVertices));
    vbWaveSim->Unlock();

    return true;
}

// World-anchored hybrid particle foam sim resources (WATER_FOAM). Four fp16 RGBA RTs
// at foamTexResolution: two ping-pong particle buffers, one velocity/density field
// buffer, one foam-output buffer sampled by the water shader. The sim itself reuses
// the wave-sim fullscreen quad (vbWaveSim) + WaveVS, so no VB is created here.
bool DistantLand::initFoamSim() {
    HRESULT hr;
    const int res = foamTexResolution;

    // One full set of 4 RTs per cascade (fine + coarse); the resolution is shared.
    for (int c = 0; c < foamCascades; ++c) {
        struct { IDirect3DTexture9** tex; IDirect3DSurface9** surf; const char* name; } targets[] = {
            { &texFoamP_A[c],   &surfFoamP_A[c],   "foam particle A" },
            { &texFoamP_B[c],   &surfFoamP_B[c],   "foam particle B" },
            { &texFoamField[c], &surfFoamField[c], "foam field" },
            { &texFoam[c],      &surfFoam[c],      "foam output" },
            { &texFoamUV_A[c],  &surfFoamUV_A[c],  "foam UV A" },
            { &texFoamUV_B[c],  &surfFoamUV_B[c],  "foam UV B" },
        };

        for (auto& t : targets) {
            hr = device->CreateTexture(res, res, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A16B16G16R16F, D3DPOOL_DEFAULT, t.tex, NULL);
            if (hr != D3D_OK) {
                LOG::logline("!! Failed to create %s texture (cascade %d)", t.name, c);
                return false;
            }
            (*t.tex)->GetSurfaceLevel(0, t.surf);
            device->ColorFill(*t.surf, 0, 0);
        }
    }

    // Foam's own fullscreen triangle, scaled to foamTexResolution (mirrors the wave-sim
    // vbWaveSim triangle but at the foam RT size). The WaveVS geometry behaves in pixel
    // space, so a 512-sized triangle covers only a cropped corner of a larger foam RT;
    // sizing the triangle to the RT fills it exactly (see simulateFoam).
    const float r = (float)foamTexResolution;
    float foamVertices[] = {
        -r/2 - 0.5f,    r/2 - 0.5f,   0, 1,   -0.5f,  0.5f,   0, 0,
         r   - 0.5f,  2*r   - 0.5f,   0, 1,    1.0f,  2.0f,   0, 1,
         r   - 0.5f,   -r    - 0.5f,  0, 1,    1.0f, -1.0f,   1, 1
    };
    hr = device->CreateVertexBuffer(3 * 32, D3DUSAGE_WRITEONLY, fvfWave, D3DPOOL_DEFAULT, &vbFoamSim, 0);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create foam simulation vb");
        return false;
    }
    void* fvp;
    if (vbFoamSim->Lock(0, 0, &fvp, 0) != D3D_OK) {
        LOG::logline("!! Failed to lock foam simulation vb");
        return false;
    }
    memcpy(fvp, foamVertices, sizeof(foamVertices));
    vbFoamSim->Unlock();

    foamSimReset = true;
    return true;
}

bool DistantLand::initShadow() {
    const D3DFORMAT shadowFormat = D3DFMT_R16F, shadowZFormat = D3DFMT_D24S8;
    const UINT shadowSize = Configuration.DL.ShadowResolution, cascades = 2;
    HRESULT hr;

    // The shadow texture holds a horizontal-packed shadow atlas
    hr = device->CreateTexture(cascades * shadowSize, shadowSize, 1, D3DUSAGE_RENDERTARGET, shadowFormat, D3DPOOL_DEFAULT, &texShadow, NULL);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create shadow render target");
        return false;
    }
    hr = device->CreateTexture(cascades * shadowSize, shadowSize, 1, D3DUSAGE_RENDERTARGET, shadowFormat, D3DPOOL_DEFAULT, &texSoftShadow, NULL);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create shadow render target");
        return false;
    }
    hr = device->CreateDepthStencilSurface(cascades * shadowSize, shadowSize, shadowZFormat, D3DMULTISAMPLE_NONE, 0, TRUE, &surfShadowZ, NULL);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create shadow Z buffer");
        return false;
    }
    hr = device->CreateVertexBuffer(4 * 12, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &vbFullFrame, 0);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create shadow processing verts");
        return false;
    }
    hr = device->CreateVertexBuffer(14 * 12, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &vbClipCube, 0);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create shadow processing verts");
        return false;
    }

    // Used to cover an entire render target of any dimension
    D3DXVECTOR3* v;
    vbFullFrame->Lock(0, 0, (void**)&v, 0);
    v[0] = D3DXVECTOR3( -1.0f, 1.0f,  1.0f);
    v[1] = D3DXVECTOR3(-1.0f, -1.0f,  1.0f);
    v[2] = D3DXVECTOR3( 1.0f,  1.0f,  1.0f);
    v[3] = D3DXVECTOR3( 1.0f, -1.0f,  1.0f);
    vbFullFrame->Unlock();

    // Used to project the view frustum in world space
    // Slightly expanded from the canonical cube to allow for rasterization and filtering
    const float u = 1.01f;
    vbClipCube->Lock(0, 0, (void**)&v, 0);
    v[0] = D3DXVECTOR3(-u,  u, 0.0f);
    v[1] = D3DXVECTOR3(-u, -u, 0.0f);
    v[2] = D3DXVECTOR3( u,  u, 0.0f);
    v[3] = D3DXVECTOR3( u, -u, 0.0f);
    v[4] = D3DXVECTOR3( u, -u, 1.0f);
    v[5] = D3DXVECTOR3(-u, -u, 0.0f);
    v[6] = D3DXVECTOR3(-u, -u, 1.0f);
    v[7] = D3DXVECTOR3(-u,  u, 0.0f);
    v[8] = D3DXVECTOR3(-u,  u, 1.0f);
    v[9] = D3DXVECTOR3( u,  u, 0.0f);
    v[10] = D3DXVECTOR3( u,  u, 1.0f);
    v[11] = D3DXVECTOR3( u, -u, 1.0f);
    v[12] = D3DXVECTOR3(-u,  u, 1.0f);
    v[13] = D3DXVECTOR3(-u, -u, 1.0f);
    vbClipCube->Unlock();

    return true;
}

bool DistantLand::initDistantStaticsClient() {
    if (FAILED(device->CreateVertexDeclaration(StaticElem, &StaticDecl))) {
        LOG::logline("!! Failed to to create static vertex declaration");
        return false;
    }

    if (GetFileAttributes("Data Files\\distantland\\statics") == INVALID_FILE_ATTRIBUTES) {
        LOG::logline("!! Distant statics have not been generated");
        LOG::flush();
        return !(Configuration.MGEFlags & USE_DISTANT_LAND);
    }

    if (Configuration.UseSharedMemory) {
        auto staticsId = IPC::InvalidVector;
        auto subsetsId = IPC::InvalidVector;
        {
            auto maybeStatics = ipcClient.allocVecBlocking<DistantStatic>(1, 500000, 1);
            if (!maybeStatics.has_value()) {
                return false;
            }

            auto maybeSubsets = ipcClient.allocVecBlocking<DistantSubset>(1, 500000, 1);
            if (!maybeSubsets.has_value()) {
                return false;
            }

            auto& statics = maybeStatics.value();
            auto& subsets = maybeSubsets.value();
            if (!loadDistantStaticsClient(statics, subsets)) {
                return false;
            }

            staticsId = statics.id();
            subsetsId = subsets.id();
            if (!ipcClient.initDistantStatics(staticsId, subsetsId)) {
                return false;
            }

            // our views are destroyed
        }

        // free on server
        ipcClient.freeVecBlocking(staticsId);
        ipcClient.freeVec(subsetsId);
    } else {
        vector<DistantStatic> distantStatics;
        vector<DistantSubset> distantSubsets;
        if (!loadDistantStaticsClient(distantStatics, distantSubsets)) {
            return false;
        }
    }

    DistantLandShare::currentWorldSpace = nullptr;
    DistantLandShare::hasCurrentWorldSpace = false;
    isDistantLandLoaded = true;
    return true;
}

template<class T, class U>
bool DistantLand::loadStaticMeshes(HANDLE h, T& distantStatics, U& distantSubsets) {
    DWORD unused;

    size_t DistantStaticCount;
    ReadFile(h, &DistantStaticCount, 4, &unused, 0);
    distantStatics.reserve(DistantStaticCount);

    // we don't actually know yet how many subsets there will be, but it'll probably be at least this many
    distantSubsets.reserve(DistantStaticCount);
    
    HANDLE h2 = CreateFile("Data Files\\distantland\\statics\\static_meshes", GENERIC_READ, 0, 0, OPEN_EXISTING, 0, 0);
    if (h2 == INVALID_HANDLE_VALUE) {
        LOG::logline("!! Required distant statics files are missing, regeneration required - distantland/statics/static_meshes");
        LOG::flush();
        return false;
    }

    // Bright yellow error texture
    IDirect3DTexture9* errorTexture;
    device->CreateTexture(1, 1, 1, g_spikeForceDefaultPool ? D3DUSAGE_DYNAMIC : 0, D3DFMT_A8R8G8B8, g_spikeForceDefaultPool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &errorTexture, NULL);

    D3DLOCKED_RECT yellow;
    errorTexture->LockRect(0, &yellow, NULL, 0);
    *(DWORD*)yellow.pBits = 0xffffff00;
    errorTexture->UnlockRect(0);

    // Read entire file into one big memory buffer
    DWORD file_size = GetFileSize(h2, NULL);
    auto file_buffer = std::make_unique<char[]>(file_size);
    ReadFile(h2, file_buffer.get(), file_size, &unused, NULL);
    membuf_reader reader(file_buffer.get());
    CloseHandle(h2);

    for (DWORD distantStaticIndex = 0; distantStaticIndex < DistantStaticCount; distantStaticIndex++) {
        DistantStatic i = {};
        reader.read(&i.numSubsets, 4);
        reader.read(&i.sphere.radius, 4);
        reader.read(&i.sphere.center, 12);
        reader.read(&i.type, 1);

        i.aabbMin = D3DXVECTOR3(FLT_MAX, FLT_MAX, FLT_MAX);
        i.aabbMax = D3DXVECTOR3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        i.firstSubsetIndex = distantSubsets.size();
        for (size_t subsetIndex = 0; subsetIndex < i.numSubsets; subsetIndex++) {
            DistantSubset subset = {};

            // Get bounding sphere
            reader.read(&subset.sphere.radius, 4);
            reader.read(&subset.sphere.center, 12);

            // Get AABB min and max
            reader.read(&subset.aabbMin, 12);
            reader.read(&subset.aabbMax, 12);

            // Get vertex and face count
            reader.read(&subset.verts, 4);
            reader.read(&subset.faces, 4);

            // Update parent AABB
            i.aabbMin.x = std::min(i.aabbMin.x, subset.aabbMin.x);
            i.aabbMin.y = std::min(i.aabbMin.y, subset.aabbMin.y);
            i.aabbMin.z = std::min(i.aabbMin.z, subset.aabbMin.z);
            i.aabbMax.x = std::max(i.aabbMax.x, subset.aabbMax.x);
            i.aabbMax.y = std::max(i.aabbMax.y, subset.aabbMax.y);
            i.aabbMax.z = std::max(i.aabbMax.z, subset.aabbMax.z);

            // Load mesh data
            IDirect3DVertexBuffer9* vb;
            IDirect3DIndexBuffer9* ib;
            void* lockdata;

            device->CreateVertexBuffer(subset.verts * SIZEOFSTATICVERT, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &vb, 0);
            vb->Lock(0, 0, &lockdata, 0);
            reader.read(lockdata, subset.verts * SIZEOFSTATICVERT);
            vb->Unlock();

            device->CreateIndexBuffer(subset.faces * 6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT, &ib, 0);
            ib->Lock(0, 0, &lockdata, 0);
            reader.read(lockdata, subset.faces * 6); // Morrowind nifs don't support 32 bit indices?
            ib->Unlock();

            subset.vbuffer = vb;
            subset.ibuffer = ib;

            // Texturing flags
            bool texturingFlags[2];
            reader.read(&texturingFlags, 2);
            subset.hasAlpha = texturingFlags[0];
            subset.hasUVController = texturingFlags[1];

            // Load referenced texture
            unsigned short pathsize;
            reader.read(&pathsize, 2);
            const char* texname = reader.get();
            reader.advance(pathsize);

            IDirect3DTexture9* tex = BSA::loadTexture(device, texname);
            if (!tex) {
                LOG::logline("Cannot load texture %s", texname);
                errorTexture->AddRef();
                tex = errorTexture;
            }
            subset.tex = tex;

            // Keep resource pointers for deallocation
            meshCollectionStatics.push_back(MeshResources(vb, ib, tex));

            distantSubsets.push_back(subset);
        }

        distantStatics.push_back(i);
    }
    file_buffer.reset();
    errorTexture->Release();


    // Texture memory reporting
    int texturesLoaded, texMemUsage;
    BSA::cacheStats(&texturesLoaded, &texMemUsage);

    LOG::logline("-- Distant static geometry memory use: %d MB", file_size / (1 << 20));
    LOG::logline("-- Distant textures loaded, %d textures", texturesLoaded);
    LOG::logline("-- Distant texture memory use: %d MB", texMemUsage);
    LOG::flush();

    return true;
}

void DistantLand::loadVisGroupsClient(HANDLE h) {
    DWORD unused;

    // Load dynamic vis groups
    size_t dynamicVisGroupCount;
    ReadFile(h, &dynamicVisGroupCount, 4, &unused, 0);
    dynamicVisGroups.clear();

    if (dynamicVisGroupCount > 0) {
        const size_t visGroupRecordSize = 130;
        size_t visDataSize = visGroupRecordSize * dynamicVisGroupCount;
        auto visData = std::make_unique<char[]>(visDataSize);
        ReadFile(h, visData.get(), visDataSize, &unused, 0);
        membuf_reader visReader(visData.get());

        // VisGroup indexes use a 1-based index, group 0 is reserved for testing
        dynamicVisGroups.resize(dynamicVisGroupCount + 1);

        for (size_t nVisGroup = 1; nVisGroup <= dynamicVisGroupCount; ++nVisGroup) {
            DynamicVisGroup& dvg = dynamicVisGroups[nVisGroup];
            visReader.read(&dvg.source, 1);
            dvg.enabled = true;
            dvg.gameObject = nullptr;

            char id[64];
            visReader.read(&id, sizeof(id));
            dvg.id = id;

            uint8_t rangeCount;
            visReader.read(&rangeCount, sizeof(rangeCount));

            DynamicVisGroup::Range ranges[8];
            visReader.read(&ranges, sizeof(ranges));
            dvg.ranges.assign(ranges, ranges + rangeCount);
        }

        visData.reset();
    }
}

template<class T, class U>
bool DistantLand::loadDistantStaticsClient(T& distantStatics, U& distantSubsets) {
    auto h = DistantLandShare::beginReadStatics();
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }

    if (!loadStaticMeshes(h, distantStatics, distantSubsets)) {
        CloseHandle(h);
        return false;
    }
    loadVisGroupsClient(h);
    
    if (Configuration.UseSharedMemory) {
        CloseHandle(h);
        return true; // server will handle the rest of the logic
    }

    DistantLandShare::readDistantStatics(h, distantStatics, distantSubsets, dynamicVisGroups);
    CloseHandle(h);
    return true;
}

bool DistantLand::initLandscapeClient() {
    HANDLE file = CreateFile("Data Files\\distantland\\world", GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }

    DWORD file_size = GetFileSize(file, NULL);
    DWORD mesh_count, unused;
    ReadFile(file, &mesh_count, 4, &unused, 0);
    if (mesh_count == 0) {
        CloseHandle(file);
        return true;
    }

    auto id = IPC::InvalidVector;
    {
        // allocVecBlocking returns the optional by value; bind via auto&& so the
        // temporary's lifetime extends to this block (C++20 rejects auto& here).
        auto&& maybeBuffers = ipcClient.allocVecBlocking<IPC::LandscapeBuffers>(1, 200000, mesh_count);
        if (!maybeBuffers.has_value()) {
            return false;
        }

        auto& buffers = maybeBuffers.value();
        id = buffers.id();

        // the server will read data as we populate it
        if (!ipcClient.initLandscape(id)) {
            ipcClient.freeVecBlocking(id);
            return false;
        }

        buffers.start_write();
        for (DWORD i = 0; i < mesh_count; i++) {
            // skip info that will be handled by the server
            SetFilePointer(file, 40, NULL, FILE_CURRENT);

            DWORD verts = 0, faces = 0;
            IDirect3DVertexBuffer9* vb;
            IDirect3DIndexBuffer9* ib;
            void* lockdata;

            ReadFile(file, &verts, 4, &unused, 0);
            ReadFile(file, &faces, 4, &unused, 0);
            bool large = (verts > 0xFFFF || faces > 0xFFFF);

            // Stage both VB and IB bytes into CPU-side buffers so we
            // can (a) hand them to the land-mesh capture and (b) push
            // to the GPU via memcpy. Reading WRITEONLY locked memory
            // goes through uncached slow paths on DXVK — staging
            // avoids that entirely.
            device->CreateVertexBuffer(verts * SIZEOFLANDVERT, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &vb, 0);
            device->CreateIndexBuffer(faces * (large ? 12 : 6), D3DUSAGE_WRITEONLY, large ? D3DFMT_INDEX32 : D3DFMT_INDEX16, D3DPOOL_DEFAULT, &ib, 0);
            {
                std::vector<char> cpuVerts(static_cast<size_t>(verts) * SIZEOFLANDVERT);
                std::vector<char> cpuIndices(static_cast<size_t>(faces) * (large ? 12 : 6));
                ReadFile(file, cpuVerts.data(),   (DWORD)cpuVerts.size(),   &unused, 0);
                ReadFile(file, cpuIndices.data(), (DWORD)cpuIndices.size(), &unused, 0);

                captureLandMesh(vb, cpuVerts.data(), verts, cpuIndices.data(), faces, large);

                vb->Lock(0, 0, &lockdata, 0);
                std::memcpy(lockdata, cpuVerts.data(), cpuVerts.size());
                vb->Unlock();

                ib->Lock(0, 0, &lockdata, 0);
                std::memcpy(lockdata, cpuIndices.data(), cpuIndices.size());
                ib->Unlock();
            }

            buffers.push_back({ vb, ib });

            meshCollectionLand.push_back(MeshResources(vb, ib, 0));
        }
        buffers.end_write();

        // our views must be destroyed before we can free the vec
    }

    ipcClient.freeVec(id);
    CloseHandle(file);
    return true;
}

bool DistantLand::initLandscape() {
    HRESULT hr;

    hr = device->CreateVertexDeclaration(LandElem, &LandDecl);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to to create world vertex declaration");
        return false;
    }

    if (GetFileAttributes("Data Files\\distantland\\world") == INVALID_FILE_ATTRIBUTES) {
        LOG::logline("!! Distant land files have not been generated");
        LOG::flush();
        return !(Configuration.MGEFlags & USE_DISTANT_LAND);
    }

    hr = D3DXCreateTextureFromFileEx(device, "Data Files\\distantland\\world.dds", 0, 0, 0, 0, D3DFMT_UNKNOWN, D3DPOOL_DEFAULT, D3DX_DEFAULT, D3DX_DEFAULT, 0, 0, 0, &texWorldColour);
    if (hr != D3D_OK) {
        LOG::logline("!! Could not load world texture for distant land - distantland/world.dds");
        LOG::flush();
        return false;
    }

    hr = D3DXCreateTextureFromFileEx(device, "Data Files\\distantland\\world_n.dds", 0, 0, 0, 0, D3DFMT_UNKNOWN, D3DPOOL_DEFAULT, D3DX_DEFAULT, D3DX_DEFAULT, 0, 0, 0, &texWorldNormals);
    if (hr != D3D_OK) {
        LOG::logline("!! Could not load world normal map texture for distant land - distantland/world_n.dds");
        LOG::flush();
        return false;
    }

    hr = D3DXCreateTextureFromFileEx(device, "Data Files\\textures\\MGE\\world_detail.dds", 0, 0, 0, 0, D3DFMT_UNKNOWN, D3DPOOL_DEFAULT, D3DX_DEFAULT, D3DX_DEFAULT, 0, 0, 0, &texWorldDetail);
    if (hr != D3D_OK) {
        LOG::logline("!! Could not load world detail texture for distant land - textures/MGE/world_detail.dds");
        LOG::flush();
        return false;
    }

    LOG::logline("-- Landscape textures loaded");

    if (Configuration.UseSharedMemory) {
        return initLandscapeClient();
    }

    HANDLE file = CreateFile("Data Files\\distantland\\world", GENERIC_READ, 0, 0, OPEN_EXISTING, 0, 0);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }

    DWORD file_size = GetFileSize(file, NULL);
    DWORD mesh_count, unused;
    ReadFile(file, &mesh_count, 4, &unused, 0);

    vector<LandMesh> meshesLand;
    meshesLand.resize(mesh_count);

    if (!meshesLand.empty()) {
        D3DXVECTOR2 qtmin(FLT_MAX, FLT_MAX), qtmax(-FLT_MAX, -FLT_MAX);
        D3DXMATRIX world;
        D3DXMatrixIdentity(&world);

        // Load meshes and calculate max size of quadtree
        for (auto& i : meshesLand) {
            ReadFile(file, &i.sphere.radius, 4, &unused,0);
            ReadFile(file, &i.sphere.center, 12, &unused,0);

            D3DXVECTOR3 boxMin, boxMax;
            ReadFile(file, &boxMin, 12, &unused, 0);
            ReadFile(file, &boxMax, 12, &unused, 0);
            i.box.Set(boxMin, boxMax);

            ReadFile(file, &i.verts, 4, &unused, 0);
            ReadFile(file, &i.faces, 4, &unused, 0);

            bool large = (i.verts > 0xFFFF || i.faces > 0xFFFF);
            IDirect3DVertexBuffer9* vb;
            IDirect3DIndexBuffer9* ib;
            void* lockdata;

            // Same CPU-staging + full-mesh capture pattern as the IPC
            // path above. Stage vertex and index bytes into plain
            // std::vector buffers so we can hand them to the land-mesh
            // capture and then push to the GPU via memcpy — avoiding
            // any reads from WRITEONLY-locked driver memory.
            device->CreateVertexBuffer(i.verts * SIZEOFLANDVERT, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &vb, 0);
            device->CreateIndexBuffer(i.faces * (large ? 12 : 6), D3DUSAGE_WRITEONLY, large ? D3DFMT_INDEX32 : D3DFMT_INDEX16, D3DPOOL_DEFAULT, &ib, 0);
            {
                std::vector<char> cpuVerts(static_cast<size_t>(i.verts) * SIZEOFLANDVERT);
                std::vector<char> cpuIndices(static_cast<size_t>(i.faces) * (large ? 12 : 6));
                ReadFile(file, cpuVerts.data(),   (DWORD)cpuVerts.size(),   &unused, 0);
                ReadFile(file, cpuIndices.data(), (DWORD)cpuIndices.size(), &unused, 0);

                captureLandMesh(vb, cpuVerts.data(), i.verts, cpuIndices.data(), i.faces, large);

                vb->Lock(0, 0, &lockdata, 0);
                std::memcpy(lockdata, cpuVerts.data(), cpuVerts.size());
                vb->Unlock();

                ib->Lock(0, 0, &lockdata, 0);
                std::memcpy(lockdata, cpuIndices.data(), cpuIndices.size());
                ib->Unlock();
            }

            i.vbuffer = vb;
            i.ibuffer = ib;

            qtmin.x = std::min(qtmin.x, i.sphere.center.x - i.sphere.radius);
            qtmin.y = std::min(qtmin.y, i.sphere.center.y - i.sphere.radius);
            qtmax.x = std::max(qtmax.x, i.sphere.center.x + i.sphere.radius);
            qtmax.y = std::max(qtmax.y, i.sphere.center.y + i.sphere.radius);
        }

        DistantLandShare::LandQuadTree.SetBox(std::max(qtmax.x - qtmin.x, qtmax.y - qtmin.y), 0.5 * (qtmax + qtmin));

        // Add meshes to the quadtree
        for (auto& i : meshesLand) {
            meshCollectionLand.push_back(MeshResources(i.vbuffer, i.ibuffer, 0));
            DistantLandShare::LandQuadTree.AddMesh(i.sphere, i.box, world, false, false, texWorldColour, i.verts, i.vbuffer, i.faces, i.ibuffer);
        }
    }

    CloseHandle(file);
    DistantLandShare::LandQuadTree.CalcVolume();

    // Log approximate memory use
    LOG::logline("-- Distant landscape memory use: %d MB", file_size / (1 << 20));

    return true;
}

bool DistantLand::initGrass() {
    HRESULT hr;

    hr = device->CreateVertexDeclaration(GrassElem, &GrassDecl);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create grass decl");
        return false;
    }

    hr = device->CreateVertexBuffer(MaxGrassElements * GrassInstStride, D3DUSAGE_DYNAMIC|D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &vbGrassInstances, NULL);
    if (hr != D3D_OK) {
        LOG::logline("!! Failed to create grass instance buffer");
        return false;
    }

    return true;
}

void DistantLand::release() {
    if (!ready) {
        return;
    }

    LOG::logline("-- Renderer unloading");

    recordMW.clear();
    recordSky.clear();

    PostShaders::release();
    FixedFunctionShader::release();

    DistantLandShare::mapWorldSpaces.clear();

    for (auto& iM : meshCollectionStatics) {
        iM.vb->Release();
        iM.ib->Release();
        iM.tex->Release();
    }
    meshCollectionStatics.clear();

    DistantLandShare::LandQuadTree.Clear();
    for (auto& iM : meshCollectionLand) {
        iM.vb->Release();
        iM.ib->Release();
        // A shared texture is used for land, and is released below
    }
    meshCollectionLand.clear();

    // Drop the per-tile mesh cache so its position + index buffers
    // don't leak across release/init cycles. Map keys are the now-
    // released VB pointers, so any survivor would also be a dangling-
    // pointer hazard. Paired with the horizon-curtain workspace
    // teardown below.
    landMeshes.clear();
    shutdownHorizonWorkspace();

    // Tear down the dedicated MSOC cull worker so the thread doesn't outlive
    // the renderer across init/release cycles (mirrors the SceneGraph worker
    // shutdown). Safe to call when the worker was never spawned.
    joinCullWorker();

    if (texWorldColour) {
        texWorldColour->Release();
        texWorldColour = nullptr;
        texWorldNormals->Release();
        texWorldNormals = nullptr;
        texWorldDetail->Release();
        texWorldDetail = nullptr;
    }

    BSA::clearTextureCache();

    if (Configuration.MGEFlags & DYNAMIC_RIPPLES) {
        surfRain->Release();
        surfRain = nullptr;
        texRain->Release();
        texRain = nullptr;
        surfRipples->Release();
        surfRipples = nullptr;
        texRipples->Release();
        texRipples = nullptr;
        surfRippleBuffer->Release();
        surfRippleBuffer = nullptr;
        texRippleBuffer->Release();
        texRippleBuffer = nullptr;
        vbWaveSim->Release();
        vbWaveSim = nullptr;
    }

    LandDecl->Release();
    LandDecl = nullptr;
    StaticDecl->Release();
    StaticDecl = nullptr;
    WaterDecl->Release();
    WaterDecl = nullptr;
    GrassDecl->Release();
    GrassDecl = nullptr;

    texShadow->Release();
    texShadow = nullptr;
    texSoftShadow->Release();
    texSoftShadow = nullptr;
    surfShadowZ->Release();
    surfShadowZ = nullptr;

    if (texFlow) {
        texFlow->Release();
        texFlow = nullptr;
    }

    if (texFoam[0]) {
        for (int c = 0; c < foamCascades; ++c) {
            surfFoamP_A[c]->Release();   surfFoamP_A[c] = nullptr;
            texFoamP_A[c]->Release();    texFoamP_A[c] = nullptr;
            surfFoamP_B[c]->Release();   surfFoamP_B[c] = nullptr;
            texFoamP_B[c]->Release();    texFoamP_B[c] = nullptr;
            surfFoamField[c]->Release(); surfFoamField[c] = nullptr;
            texFoamField[c]->Release();  texFoamField[c] = nullptr;
            surfFoam[c]->Release();      surfFoam[c] = nullptr;
            texFoam[c]->Release();       texFoam[c] = nullptr;
            surfFoamUV_A[c]->Release();  surfFoamUV_A[c] = nullptr;
            texFoamUV_A[c]->Release();   texFoamUV_A[c] = nullptr;
            surfFoamUV_B[c]->Release();  surfFoamUV_B[c] = nullptr;
            texFoamUV_B[c]->Release();   texFoamUV_B[c] = nullptr;
        }
        if (vbFoamSim) { vbFoamSim->Release(); vbFoamSim = nullptr; }
    }

    texWater->Release();
    texWater = nullptr;
    texReflection->Release();
    texReflection = nullptr;
    surfReflectionZ->Release();
    surfReflectionZ = nullptr;
    vbWater->Release();
    vbWater = nullptr;
    ibWater->Release();
    ibWater = nullptr;
    if (vbWaterLod) {
        vbWaterLod->Release();
        vbWaterLod = nullptr;
    }
    if (ibWaterLod) {
        ibWaterLod->Release();
        ibWaterLod = nullptr;
    }
    vbGrassInstances->Release();
    vbGrassInstances = nullptr;
    vbFullFrame->Release();
    vbFullFrame = nullptr;
    vbClipCube->Release();
    vbClipCube = nullptr;

    texDepthFrame->Release();
    texDepthFrame = nullptr;
    surfDepthDepth->Release();
    surfDepthDepth = nullptr;

    effectPool->Release();
    effectPool = nullptr;
    effectShadow->Release();
    effectShadow = nullptr;
    effectDepth->Release();
    effectDepth = nullptr;
    effect->Release();
    effect = nullptr;

    LOG::logline("-- Renderer unloaded");
    LOG::flush();

    fogNearEnd = 0;
    device = nullptr;
    ready = false;
}
