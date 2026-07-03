#include "msocclient.h"
#include "configuration.h"
#include "support/log.h"

#include <Windows.h>

namespace {

using FnIsMaskReady = int (__cdecl*)();
using FnTestSphere  = int (__cdecl*)(float, float, float, float);
using FnTestOBB     = int (__cdecl*)(
    float, float, float,   // center
    float, float, float,   // vx
    float, float, float,   // vy
    float, float, float);  // vz
using FnDumpMask            = int  (__cdecl*)(const char* path);
using FnTestSphereBatch     = void (__cdecl*)(const float*, int, int*);
using FnGetSnapshotViewProj = void (__cdecl*)(float[16]);
using FnGetSnapshotAgeMs    = unsigned long long (__cdecl*)();
using FnGetMaskResolution   = void (__cdecl*)(int*, int*);
using FnAddOccluder         = int  (__cdecl*)(
    const float* verts, int vtxCount, int stride, int offY, int offW,
    const unsigned int* tris, int triCount,
    const float* modelMatrix16);
using FnAddPreTransformedOccluder = int  (__cdecl*)(
    const float* verts, int vtxCount, int stride, int offY, int offW,
    const unsigned int* tris, int triCount);
using FnRegisterVisGeom   = void (__cdecl*)(void(__cdecl*)(void* const*, const float*, int));
using FnUnregisterVisGeom = void (__cdecl*)(void(__cdecl*)(void* const*, const float*, int));
using FnRegisterOccGeom   = void (__cdecl*)(void(__cdecl*)(void* const*, const float*, int));
using FnUnregisterOccGeom = void (__cdecl*)(void(__cdecl*)(void* const*, const float*, int));
using FnCopyMask          = int  (__cdecl*)(void* dst, int dstBytes);
using FnClassifyNow       = int  (__cdecl*)(void* camera);
using FnSetOpaqueOwned    = void (__cdecl*)(int owned);

HMODULE       g_module            = nullptr;
FnIsMaskReady g_isMaskReady       = nullptr;
FnTestSphere  g_testSphere        = nullptr;
FnTestOBB     g_testOBB           = nullptr;  // optional; older plugins don't export it
FnDumpMask    g_dumpMask          = nullptr;  // optional; likewise
FnTestSphereBatch     g_testSphereBatch  = nullptr;  // optional
FnGetSnapshotViewProj g_getViewProj      = nullptr;  // optional
FnGetSnapshotAgeMs    g_getAgeMs         = nullptr;  // optional
FnGetMaskResolution   g_getMaskRes       = nullptr;  // optional
FnAddOccluder         g_addOccluder      = nullptr;  // optional
FnAddPreTransformedOccluder g_addPreTransformedOccluder = nullptr;  // optional
FnRegisterVisGeom     g_registerVisGeom   = nullptr;  // optional
FnUnregisterVisGeom   g_unregisterVisGeom = nullptr;  // optional
FnRegisterOccGeom     g_registerOccGeom   = nullptr;  // optional; occlusion refinement
FnUnregisterOccGeom   g_unregisterOccGeom = nullptr;  // optional
FnCopyMask            g_copyMask          = nullptr;  // optional; host-cull only
FnClassifyNow         g_classifyNow       = nullptr;  // optional; Stage 2 early classify
FnSetOpaqueOwned      g_setOpaqueOwned    = nullptr;  // optional; owned-opaque display skip
bool          g_probed             = false;

// Frozen ABI codes from the plugin. Match PatchOcclusionCulling.h.
constexpr int kRcVisible    = 0;
constexpr int kRcOccluded   = 1;
constexpr int kRcViewCulled = 2;
constexpr int kRcNotReady   = 3;

HMODULE tryLoadPlugin() {
    // First try the deterministic path — Morrowind's cwd is the install
    // root, so the MWSE plugin ships at this exact relative location.
    if (HMODULE h = LoadLibraryW(L"Data Files\\MWSE\\lib\\msoc.dll")) {
        return h;
    }
    // Fallback: plain name. Picked up if the user dropped msoc.dll next
    // to d3d8.dll (loader searches cwd + PATH before failing).
    return LoadLibraryW(L"msoc.dll");
}

} // namespace

void MSOCClient::init() {
    if (g_probed) return;
    g_probed = true;

    if (!Configuration.UseOcclusionCulling) {
        LOG::logline("-- MSOC: disabled via [Misc] Use Occlusion Culling=0");
        return;
    }

    g_module = tryLoadPlugin();
    if (!g_module) {
        LOG::logline("-- MSOC: msoc.dll not found; distant-statics occlusion disabled");
        return;
    }

    g_isMaskReady = reinterpret_cast<FnIsMaskReady>(
        GetProcAddress(g_module, "mwse_isOcclusionMaskReady"));
    g_testSphere = reinterpret_cast<FnTestSphere>(
        GetProcAddress(g_module, "mwse_testOcclusionSphere"));
    g_testOBB = reinterpret_cast<FnTestOBB>(
        GetProcAddress(g_module, "mwse_testOcclusionOBB"));
    g_dumpMask = reinterpret_cast<FnDumpMask>(
        GetProcAddress(g_module, "mwse_dumpOcclusionMask"));
    g_testSphereBatch = reinterpret_cast<FnTestSphereBatch>(
        GetProcAddress(g_module, "mwse_testOcclusionSphereBatch"));
    g_getViewProj = reinterpret_cast<FnGetSnapshotViewProj>(
        GetProcAddress(g_module, "mwse_getSnapshotViewProj"));
    g_getAgeMs = reinterpret_cast<FnGetSnapshotAgeMs>(
        GetProcAddress(g_module, "mwse_getSnapshotAgeMs"));
    g_getMaskRes = reinterpret_cast<FnGetMaskResolution>(
        GetProcAddress(g_module, "mwse_getMaskResolution"));
    g_addOccluder = reinterpret_cast<FnAddOccluder>(
        GetProcAddress(g_module, "mwse_addOccluder"));
    g_addPreTransformedOccluder = reinterpret_cast<FnAddPreTransformedOccluder>(
        GetProcAddress(g_module, "mwse_addPreTransformedOccluder"));
    g_registerVisGeom = reinterpret_cast<FnRegisterVisGeom>(
        GetProcAddress(g_module, "mwse_registerVisibleGeomCallback"));
    g_unregisterVisGeom = reinterpret_cast<FnUnregisterVisGeom>(
        GetProcAddress(g_module, "mwse_unregisterVisibleGeomCallback"));
    g_registerOccGeom = reinterpret_cast<FnRegisterOccGeom>(
        GetProcAddress(g_module, "mwse_registerOccludedGeomCallback"));
    g_unregisterOccGeom = reinterpret_cast<FnUnregisterOccGeom>(
        GetProcAddress(g_module, "mwse_unregisterOccludedGeomCallback"));
    g_copyMask = reinterpret_cast<FnCopyMask>(
        GetProcAddress(g_module, "mwse_copyOcclusionMask"));
    g_classifyNow = reinterpret_cast<FnClassifyNow>(
        GetProcAddress(g_module, "mwse_classifyMainSceneNow"));
    g_setOpaqueOwned = reinterpret_cast<FnSetOpaqueOwned>(
        GetProcAddress(g_module, "mwse_setOpaqueWorldOwned"));

    if (!g_isMaskReady || !g_testSphere) {
        LOG::logline("-- MSOC: msoc.dll loaded but required exports missing; disabling");
        FreeLibrary(g_module);
        g_module = nullptr;
        g_isMaskReady = nullptr;
        g_testSphere = nullptr;
        g_testOBB = nullptr;
        g_dumpMask = nullptr;
        g_testSphereBatch = nullptr;
        g_getViewProj = nullptr;
        g_getAgeMs = nullptr;
        g_getMaskRes = nullptr;
        g_addOccluder = nullptr;
        g_addPreTransformedOccluder = nullptr;
        g_registerVisGeom = nullptr;
        g_unregisterVisGeom = nullptr;
        g_registerOccGeom = nullptr;
        g_unregisterOccGeom = nullptr;
        g_copyMask = nullptr;
        g_classifyNow = nullptr;
        g_setOpaqueOwned = nullptr;
        return;
    }

    LOG::logline("-- MSOC: msoc.dll loaded, distant-statics occlusion active%s%s%s%s%s",
                 g_testOBB         ? " (OBB escalation available)" : " (sphere-only plugin)",
                 g_testSphereBatch ? " (batch query available)"    : "",
                 g_addOccluder     ? " (addOccluder available)"    : "",
                 g_addPreTransformedOccluder ? " (pre-transformed occluder available)" : "",
                 g_setOpaqueOwned  ? " (opaque-owned display skip available)"
                                   : " (NO opaque-owned skip - old msoc.dll)");
}

bool MSOCClient::isAvailable() {
    return g_testSphere != nullptr;
}

bool MSOCClient::isMaskReady() {
    return g_isMaskReady && g_isMaskReady() != 0;
}

bool MSOCClient::isSphereVisible(float worldX, float worldY, float worldZ, float radius) {
    return classifySphere(worldX, worldY, worldZ, radius) != ResultOccluded;
}

MSOCClient::TestResult MSOCClient::classifySphere(float worldX, float worldY, float worldZ, float radius) {
    if (!g_testSphere) {
        return ResultNotReady;
    }
    const int rc = g_testSphere(worldX, worldY, worldZ, radius);
    switch (rc) {
    case kRcVisible:    return ResultVisible;
    case kRcOccluded:   return ResultOccluded;
    case kRcViewCulled: return ResultViewCulled;
    default:            return ResultNotReady;
    }
}

MSOCClient::TestResult MSOCClient::classifyOBB(
    float cx, float cy, float cz,
    float vxX, float vxY, float vxZ,
    float vyX, float vyY, float vyZ,
    float vzX, float vzY, float vzZ)
{
    if (!g_testOBB) {
        return ResultNotReady;
    }
    const int rc = g_testOBB(cx, cy, cz, vxX, vxY, vxZ, vyX, vyY, vyZ, vzX, vzY, vzZ);
    switch (rc) {
    case kRcVisible:    return ResultVisible;
    case kRcOccluded:   return ResultOccluded;
    case kRcViewCulled: return ResultViewCulled;
    default:            return ResultNotReady;
    }
}

bool MSOCClient::dumpMask(const char* path) {
    if (!g_dumpMask || !path) {
        return false;
    }
    return g_dumpMask(path) != 0;
}

bool MSOCClient::classifySphereBatch(
    const float* centersAndRadii, int count, TestResult* outResults)
{
    if (!outResults || count <= 0) {
        return false;
    }
    // Fast path: plugin exports the batch entrypoint.
    if (g_testSphereBatch && centersAndRadii) {
        // Temp int buffer: the plugin writes kMaskQuery* ints (0..3);
        // we translate them into the C++ enum after the call so the
        // public wrapper hides the ABI detail.
        // Keep on-stack for small batches, heap for large — 2800 tests
        // means ~11 KB, safe on Windows main-thread stacks.
        int* tmp = reinterpret_cast<int*>(outResults); // results are 4 bytes each, enum layout matches
        static_assert(sizeof(TestResult) == sizeof(int),
            "classifySphereBatch relies on TestResult having the same size as int");
        g_testSphereBatch(centersAndRadii, count, tmp);
        // Coerce ints to enum values. Any out-of-range plugin return
        // collapses to ResultVisible (safe fallback — render it).
        for (int i = 0; i < count; ++i) {
            const int rc = tmp[i];
            switch (rc) {
            case kRcVisible:    outResults[i] = ResultVisible;    break;
            case kRcOccluded:   outResults[i] = ResultOccluded;   break;
            case kRcViewCulled: outResults[i] = ResultViewCulled; break;
            case kRcNotReady:   outResults[i] = ResultNotReady;   break;
            default:            outResults[i] = ResultVisible;    break;
            }
        }
        return true;
    }
    // Slow path: plugin is older. Fall back to per-sphere calls so
    // callers don't have to branch.
    if (!g_testSphere || !centersAndRadii) {
        return false;
    }
    for (int i = 0; i < count; ++i) {
        const float* s = centersAndRadii + i * 4;
        outResults[i] = classifySphere(s[0], s[1], s[2], s[3]);
    }
    return true;
}

bool MSOCClient::getSnapshotViewProj(float outMatrix[16]) {
    if (!outMatrix) return false;
    if (!g_getViewProj) {
        for (int i = 0; i < 16; ++i) outMatrix[i] = 0.0f;
        return false;
    }
    g_getViewProj(outMatrix);
    return true;
}

bool MSOCClient::getSnapshotAgeMs(unsigned long long* outMs) {
    if (!outMs) return false;
    if (!g_getAgeMs) {
        *outMs = 0;
        return false;
    }
    *outMs = g_getAgeMs();
    return true;
}

bool MSOCClient::getMaskResolution(int* outWidth, int* outHeight) {
    if (!g_getMaskRes) {
        if (outWidth)  *outWidth  = 0;
        if (outHeight) *outHeight = 0;
        return false;
    }
    g_getMaskRes(outWidth, outHeight);
    return true;
}

bool MSOCClient::addOccluder(
    const float* verts, int vtxCount, int stride, int offY, int offW,
    const unsigned int* tris, int triCount,
    const float* modelMatrix16)
{
    if (!g_addOccluder) {
        return false;
    }
    return g_addOccluder(
        verts, vtxCount, stride, offY, offW,
        tris, triCount,
        modelMatrix16) != 0;
}

bool MSOCClient::addPreTransformedOccluder(
    const float* verts, int vtxCount, int stride, int offY, int offW,
    const unsigned int* tris, int triCount)
{
    if (!g_addPreTransformedOccluder) {
        return false;
    }
    return g_addPreTransformedOccluder(
        verts, vtxCount, stride, offY, offW,
        tris, triCount) != 0;
}

bool MSOCClient::registerVisibleGeomCallback(FnVisibleGeomCallback cb) {
    if (!g_registerVisGeom || !cb) return false;
    g_registerVisGeom(cb);
    return true;
}

bool MSOCClient::unregisterVisibleGeomCallback(FnVisibleGeomCallback cb) {
    if (!g_unregisterVisGeom || !cb) return false;
    g_unregisterVisGeom(cb);
    return true;
}

bool MSOCClient::registerOccludedGeomCallback(FnVisibleGeomCallback cb) {
    if (!g_registerOccGeom || !cb) return false;
    g_registerOccGeom(cb);
    return true;
}

bool MSOCClient::unregisterOccludedGeomCallback(FnVisibleGeomCallback cb) {
    if (!g_unregisterOccGeom || !cb) return false;
    g_unregisterOccGeom(cb);
    return true;
}

bool MSOCClient::hasOccludedGeomCallback() {
    return g_registerOccGeom != nullptr;
}

int MSOCClient::classifyMainSceneNow(void* camera) {
    if (!g_classifyNow) return -1;
    return g_classifyNow(camera);
}

bool MSOCClient::hasEarlyClassify() {
    return g_classifyNow != nullptr;
}

bool MSOCClient::setOpaqueWorldOwned(bool owned) {
    if (!g_setOpaqueOwned) return false;
    g_setOpaqueOwned(owned ? 1 : 0);
    return true;
}

bool MSOCClient::hasOpaqueWorldOwned() {
    return g_setOpaqueOwned != nullptr;
}

bool MSOCClient::hasMaskExport() {
    return g_copyMask != nullptr;
}

int MSOCClient::copyMaskBlob(void* dst, int dstBytes) {
    if (!g_copyMask) return 0;
    return g_copyMask(dst, dstBytes);
}
