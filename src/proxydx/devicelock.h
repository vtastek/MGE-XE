#pragma once

#include <mutex>

// Single coarse device-submission mutex shared by the two submitters of the one
// real D3D9 device:
//   1. The engine, through the ProxyDevice / MGEProxyDevice forwarders.
//   2. The MGE render thread, which holds this lock for its ENTIRE pass
//      (state save -> RT switch -> draws -> restore -> unlock).
//
// D3DCREATE_MULTITHREADED makes the runtime's own per-call state safe; this
// mutex additionally serializes OUR two submitters at device-call granularity so
// neither interleaves a multi-call sequence (RT switch + state + draw, or an
// ID3DXEffect pass) with the other.
//
// Gated by g_deviceLockEnabled: only true while a second submitter is live. Nothing sets
// it since S5a deleted the MGE render thread, so the proxy forwarders skip the lock
// entirely and the single-threaded path pays nothing.
//
// NOTE: the lock is non-recursive. It must be taken at exactly one level per
// call chain. The MGEProxyDevice overrides that invoke MGE rendering
// (BeginScene/EndScene/DrawIndexedPrimitive) deliberately do NOT lock at entry —
// the lock lives in their bare ProxyDevice base forward, so the nested MGE
// rendering (which joins the render thread via the fence) runs unlocked and
// cannot self-deadlock against the worker.
extern std::mutex g_deviceMtx;
extern bool       g_deviceLockEnabled;

// Present seam: create MW's device as D3D9Ex (shared render-target HANDLE -> zero-copy
// Vulkan hand-off). On by default and no longer an ini setting; it self-clears at startup
// if Direct3DCreate9Ex or CreateDeviceEx fails, so this is the runtime answer to "did the
// Ex path actually take?" and must be read, not assumed, after device creation.
extern bool       g_useD3D9Ex;

// Set true only when MW's device really was created as D3D9Ex. D3D9Ex rejects
// D3DPOOL_MANAGED, so while this is set the proxy resource-creation forwarders translate
// MANAGED -> DEFAULT (textures also gain D3DUSAGE_DYNAMIC so they stay lockable), and
// MGE's own MANAGED allocations do the same. Off = every allocation stays as before.
extern bool       g_spikeForceDefaultPool;

struct MgeDeviceLock {
    bool held;
    MgeDeviceLock()  : held(g_deviceLockEnabled) { if (held) g_deviceMtx.lock(); }
    ~MgeDeviceLock() { if (held) g_deviceMtx.unlock(); }
    MgeDeviceLock(const MgeDeviceLock&) = delete;
    MgeDeviceLock& operator=(const MgeDeviceLock&) = delete;
};

#define MGE_DEVLOCK() MgeDeviceLock _mge_devlock_
