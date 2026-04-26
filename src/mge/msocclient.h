#pragma once

// MOREFPS: client shim for msoc.dll (Masked Software Occlusion Culling
// plugin, ships with MWSE installs). The plugin builds a CPU occlusion
// mask from Morrowind's own near-scene occluders; we reuse that mask
// to cull distant statics before we submit them.
//
// This is a RUNTIME soft dependency. If msoc.dll is not present, not
// the expected version, or the user has disabled it in mge.ini, every
// query short-circuits to "visible" — no behavior change, no startup
// error, zero cost beyond one extra branch.
//
// Ownership: all exports live in msoc.dll; we LoadLibrary once at
// DistantLand::init() and keep the handle for process lifetime. The
// frozen C ABI is documented in the plugin's
// PatchOcclusionCulling.h (mwse_ prefix is historical — these exports
// used to live in MWSE.dll before the extraction).
class MSOCClient {
public:
    // One-time probe. Safe to call repeatedly; subsequent calls no-op.
    // Logs a single descriptive line to mgeXE.log either way.
    static void init();

    // True if msoc.dll was loaded and all required exports resolved.
    // Callers should early-out on false before iterating candidates.
    static bool isAvailable();

    // Wraps mwse_isOcclusionMaskReady. False if the plugin is absent
    // or the mask hasn't been populated this frame yet. Cheap enough
    // to call once per cull pass; callers should NOT call per-instance.
    static bool isMaskReady();

    // World-space sphere test. Returns true ("render it") on:
    //   - plugin absent
    //   - mask not ready this frame
    //   - plugin says VIEW_CULLED (sphere outside the mask's frustum —
    //     MGE-XE's distant frustum is wider, so this is common and
    //     does not imply we can skip the draw)
    // Returns false ONLY when the plugin positively reports OCCLUDED.
    static bool isSphereVisible(float worldX, float worldY, float worldZ, float radius);

    // Same test, but returns the raw plugin verdict for diagnostics.
    // Values match PatchOcclusionCulling.h (kMaskQuery*).
    enum TestResult {
        ResultVisible    = 0,
        ResultOccluded   = 1,
        ResultViewCulled = 2,
        ResultNotReady   = 3,
    };
    static TestResult classifySphere(float worldX, float worldY, float worldZ, float radius);

    // Tighter test for oriented bounding boxes. Expands (center +
    // (±vx ±vy ±vz)) into 8 corners plugin-side and runs MOC's
    // TestTriangles against the 12 box faces — catches architectural
    // giants whose bounding sphere is too loose for classifySphere to
    // ever return OCCLUDED. Roughly 10-50× more expensive per call
    // than classifySphere; gate use on a radius threshold.
    //
    // Returns ResultNotReady if the plugin is absent or doesn't export
    // mwse_testOcclusionOBB (older plugin version), so callers can
    // fall back to classifySphere transparently.
    static TestResult classifyOBB(
        float cx, float cy, float cz,
        float vxX, float vxY, float vxZ,
        float vyX, float vyY, float vyZ,
        float vzX, float vzY, float vzZ);

    // Diagnostic: write the plugin's current occlusion mask to disk as
    // a PFM (Portable FloatMap) image. Returns true on success. Silently
    // no-ops if the plugin is absent or doesn't export the dump hook.
    static bool dumpMask(const char* path);

    // Batch sphere test. Collapses N DLL-boundary crossings into one.
    //   centersAndRadii: N * 4 floats, (x, y, z, r) per sphere.
    //   count:           N
    //   outResults:      caller-allocated array of length N, filled
    //                    with TestResult values.
    // If the plugin doesn't export the batch symbol (older version),
    // falls back to per-sphere classifySphere calls so callers get
    // the same semantics either way. Returns true if any entry was
    // filled; false if plugin is absent.
    static bool classifySphereBatch(
        const float* centersAndRadii, int count, TestResult* outResults);

    // Snapshot metadata — returns true if plugin exports the accessor,
    // false otherwise (output params filled with safe zeros on false).
    static bool getSnapshotViewProj(float outMatrix[16]);
    static bool getSnapshotAgeMs(unsigned long long* outMs);
    static bool getMaskResolution(int* outWidth, int* outHeight);

    // Submit triangles that the plugin will rasterize into the NEXT
    // mask build, alongside Morrowind's native near-scene occluders.
    // Plugin copies the input arrays, so caller can free/reuse buffers
    // after this call returns.
    //
    // One-frame latency matches the double-buffered snapshot contract:
    // submissions in frame N appear in the snapshot MGE-XE queries in
    // frame N+2.
    //
    //   verts, tris           — occluder geometry (caller-owned)
    //   vtxCount, triCount    — element counts (triangles = triCount)
    //   stride, offY, offW    — MOC VertexLayout hints (bytes-per-vertex,
    //                            byte offset of .y and .w in the struct)
    //   modelMatrix16         — model-to-world column-major, or nullptr
    //                            when verts are already in world space
    //
    // Returns true if queued, false if the plugin rejected it (over
    // budget, invalid args, or addOccluder export absent).
    static bool addOccluder(
        const float* verts, int vtxCount, int stride, int offY, int offW,
        const unsigned int* tris, int triCount,
        const float* modelMatrix16 = nullptr);
};
