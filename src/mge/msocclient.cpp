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

HMODULE       g_module            = nullptr;
FnIsMaskReady g_isMaskReady       = nullptr;
FnTestSphere  g_testSphere        = nullptr;
FnTestOBB     g_testOBB           = nullptr;  // optional; older plugins don't export it
FnDumpMask    g_dumpMask          = nullptr;  // optional; likewise
FnTestSphereBatch     g_testSphereBatch  = nullptr;  // optional; Phase A expansion
FnGetSnapshotViewProj g_getViewProj      = nullptr;  // optional
FnGetSnapshotAgeMs    g_getAgeMs         = nullptr;  // optional
FnGetMaskResolution   g_getMaskRes       = nullptr;  // optional
FnAddOccluder         g_addOccluder      = nullptr;  // optional; Phase B expansion
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
        return;
    }

    LOG::logline("-- MSOC: msoc.dll loaded, distant-statics occlusion active%s%s%s",
                 g_testOBB         ? " (OBB escalation available)" : " (sphere-only plugin)",
                 g_testSphereBatch ? " (batch query available)"    : "",
                 g_addOccluder     ? " (addOccluder available)"    : "");
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
