#pragma once

#include "proxydx/d3d8header.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

/**
 * RenderThread - Manages a dedicated thread for GPU rendering operations.
 *
 * In the triple-buffered pipeline:
 *   Main thread records -> CullThread prepares -> RenderThread submits GPU work
 *
 * The render thread performs:
 *   - HLSL replay (replayRecordedCalls) from a prepared FrameBuffer
 *   - Distant land rendering stages (renderStage0/1/2)
 *   - Full frame rendering from triple buffer
 *
 * Phase 1 (current): Thread exists but sits idle. All rendering on main thread.
 * Phase 2 (Step 4): Main thread submits work, waits synchronously.
 * Phase 3 (future): Async overlap — main thread records while render thread draws.
 */
class RenderThread {
public:
    enum class WorkType {
        None,
        RenderStage0,
        RenderStage1,
        RenderStage2,
        ReplayHLSL,
        RenderFullFrame,
        Shutdown
    };

    struct SceneWork {
        WorkType type = WorkType::None;
        int sceneCount = 0;
        int bufferIndex = -1;
    };

    enum class State {
        Idle,
        Rendering,
        Complete
    };

private:
    std::thread thread;
    mutable std::mutex mutex;
    std::condition_variable workAvailable;
    std::condition_variable workComplete;
    std::atomic<bool> shutdownRequested{false};
    std::atomic<State> state{State::Idle};

    SceneWork pendingWork;
    std::atomic<bool> hasPendingWork{false};

    IDirect3DDevice9* device = nullptr;

    void workerLoop();

    void executeRenderStage0();
    void executeRenderStage1();
    void executeRenderStage2();
    void executeReplayHLSL(int sceneCount);
    void executeFullFrame(int bufferIndex);

public:
    RenderThread() = default;
    ~RenderThread();

    RenderThread(const RenderThread&) = delete;
    RenderThread& operator=(const RenderThread&) = delete;
    RenderThread(RenderThread&&) = delete;
    RenderThread& operator=(RenderThread&&) = delete;

    void start(IDirect3DDevice9* d3dDevice);
    void stop();
    void submitWork(SceneWork&& work, bool waitNow = true);
    void waitForCompletion();
    bool isComplete() const;
    bool isRunning() const { return thread.joinable(); }
};

extern RenderThread* g_renderThread;
