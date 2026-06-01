#pragma once

#include <functional>

// MGE render thread — a single persistent worker that submits GPU work on a
// second core, overlapping our submission CPU with the engine's non-device CPU
// windows (frame-start + sky). Mirrors the SceneGraph worker's lifecycle: lazy
// spawn, condvar-signalled, drained/joined on device teardown.
//
// The submitted job runs UNLOCKED on the worker; the job itself takes the device
// lock (devicelock.h) for its whole body so it is atomic w.r.t. the engine's
// proxy forwarders. kick() enqueues one job; wait() is the same-frame fence the
// main thread blocks on before consuming the job's output; stop() joins.
namespace MGE::RenderThread {

    // Lazily spawn the worker (no-op if already running).
    void ensure();

    // Enqueue a single job and wake the worker. Spawns the worker if needed.
    // Only one job is in flight at a time; a kick while a prior job is still
    // running is not expected (the per-frame fence drains it first).
    void kick(std::function<void()> job);

    // Fence: block until the kicked job has finished (returns immediately if no
    // job is pending). Cheap in steady state — the job, kicked at BeginScene(0),
    // typically completes during the sky window before this is reached.
    void wait();

    // Drain any in-flight job, signal stop, and join the worker. Called from
    // MGEProxyDevice::Release() while the real device is still alive.
    void stop();

}
