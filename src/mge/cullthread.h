#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

/**
 * CullThread - Manages a dedicated thread for CPU-side occlusion culling.
 *
 * In the triple-buffered pipeline:
 *   Main thread records draw calls -> CullThread builds Hi-Z + sets shouldRender -> RenderThread submits GPU work
 *
 * The cull thread performs:
 *   - Shader key computation (computeShaderKeyWithSuffixes)
 *   - Render bin classification
 *   - Dirty tracking (matchPreviousFrameCalls)
 *   - Deferred bounding box computation
 *   - Occluder selection and rasterization
 *   - Hi-Z pyramid build
 *   - Per-call shouldRender flag computation
 */
class CullThread {
public:
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
    std::atomic<State> state{State::Idle};

    int pendingBufferIndex = -1;
    std::atomic<bool> hasPendingWork{false};

    void workerLoop();
    void executeCull(int bufferIndex);

public:
    CullThread() = default;
    ~CullThread();

    CullThread(const CullThread&) = delete;
    CullThread& operator=(const CullThread&) = delete;
    CullThread(CullThread&&) = delete;
    CullThread& operator=(CullThread&&) = delete;

    void start();
    void stop();

    /**
     * Submit a buffer for culling.
     * @param bufferIndex Which frame buffer to process
     * @param waitNow If true, block until cull completes
     */
    void submitWork(int bufferIndex, bool waitNow = false);

    void waitForCompletion();
    bool isComplete() const;
    bool isRunning() const { return thread.joinable(); }
};

extern CullThread* g_cullThread;
