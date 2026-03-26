#pragma once

#include "proxydx/d3d8header.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

/**
 * CpuPrepThread - Dedicated thread for CPU-intensive frame preparation.
 *
 * In the triple-buffered pipeline:
 *   Main thread records (N) -> CpuPrepThread prepares (N-1) -> RenderThread renders (N-2)
 *
 * The CPU prep thread performs (all pure CPU, no GPU device access):
 *   - prepareRecordedCalls: shader key compilation, state binning
 *   - executeHiZCulling: bbox computation, occluder rasterization, Hi-Z pyramid, visibility
 *   - applyVisibilityAndFilterRecordMW: filter recordMW using visibility results
 *
 * This runs in parallel with main thread recording and GPU thread rendering.
 */
class CpuPrepThread {
public:
    enum class WorkType {
        None,
        PrepareFrame,  // Full prep: prepareRecordedCalls + culling + visibility filter
        Shutdown
    };

    struct PrepWork {
        WorkType type = WorkType::None;
        D3DXMATRIX viewMatrix;
        D3DXMATRIX projMatrix;
    };

    enum class State {
        Idle,
        Working,
        Complete
    };

private:
    std::thread thread;
    mutable std::mutex mutex;
    std::condition_variable workAvailable;
    std::condition_variable workComplete;
    std::atomic<bool> shutdownRequested{false};
    State state{State::Idle};

    PrepWork pendingWork;
    bool hasPendingWork{false};

    void workerLoop();
    void executePrepareFrame(const D3DXMATRIX& view, const D3DXMATRIX& proj);

public:
    CpuPrepThread() = default;
    ~CpuPrepThread();

    CpuPrepThread(const CpuPrepThread&) = delete;
    CpuPrepThread& operator=(const CpuPrepThread&) = delete;
    CpuPrepThread(CpuPrepThread&&) = delete;
    CpuPrepThread& operator=(CpuPrepThread&&) = delete;

    void start();
    void stop();
    void submitWork(PrepWork&& work, bool waitNow = false);
    void waitForCompletion();
    bool isComplete() const;
    bool isPending() const { return !isComplete(); }
    bool isRunning() const { return thread.joinable(); }
};

extern CpuPrepThread* g_cpuPrepThread;
