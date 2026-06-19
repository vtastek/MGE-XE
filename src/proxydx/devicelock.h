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
// Gated by g_deviceLockEnabled: only true while a render thread is live (set at
// device creation from Configuration.UseRenderThread). When the feature is off,
// the proxy forwarders skip the lock entirely, so the single-threaded path pays
// nothing and behaves byte-for-byte as before.
//
// NOTE: the lock is non-recursive. It must be taken at exactly one level per
// call chain. The MGEProxyDevice overrides that invoke MGE rendering
// (BeginScene/EndScene/DrawIndexedPrimitive) deliberately do NOT lock at entry —
// the lock lives in their bare ProxyDevice base forward, so the nested MGE
// rendering (which joins the render thread via the fence) runs unlocked and
// cannot self-deadlock against the worker.
extern std::mutex g_deviceMtx;
extern bool       g_deviceLockEnabled;

// Present-seam spike (Milestone B): set true only when MW's device was created as
// D3D9Ex (UseRenderProcessEx). D3D9Ex rejects D3DPOOL_MANAGED, so while this is set
// the proxy resource-creation forwarders translate MANAGED -> DEFAULT (textures also
// gain D3DUSAGE_DYNAMIC so they stay lockable), and MGE's own MANAGED allocations do
// the same. Off (the normal game) = every allocation stays exactly as before.
extern bool       g_spikeForceDefaultPool;

struct MgeDeviceLock {
    bool held;
    MgeDeviceLock()  : held(g_deviceLockEnabled) { if (held) g_deviceMtx.lock(); }
    ~MgeDeviceLock() { if (held) g_deviceMtx.unlock(); }
    MgeDeviceLock(const MgeDeviceLock&) = delete;
    MgeDeviceLock& operator=(const MgeDeviceLock&) = delete;
};

#define MGE_DEVLOCK() MgeDeviceLock _mge_devlock_
