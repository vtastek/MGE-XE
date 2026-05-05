#pragma once

#include "proxydx/d3d8header.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

struct DLContext;

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
        RenderFullFrame,   // GPU work only - CPU prep done by CpuPrepThread
        Shutdown
    };

    struct SceneWork {
        WorkType type = WorkType::None;
        int sceneCount = 0;
        int bufferIndex = -1;
        DLContext* ctx = nullptr;
        bool useN1Buffer = false;  // N-1 mode: use prepBuffer instead of renderBuffer
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

    // Per-object light texture - owned by RenderThread to avoid race with main thread
    IDirect3DTexture9* texPerObjectLightData = nullptr;

    void workerLoop();

    void executeRenderStage0();
    void executeRenderStage1(DLContext* ctx);
    void executeRenderStage2(DLContext* ctx);
    void executeReplayHLSL(int sceneCount);
    void executeFullFrame(int bufferIndex, bool useN1Buffer);

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
    bool isPending() const { return !isComplete(); }
    bool isRunning() const { return thread.joinable(); }

    // Per-object light texture accessors - texture owned by RenderThread for thread safety
    IDirect3DTexture9* getPerObjectLightTexture() { return texPerObjectLightData; }
    void setPerObjectLightTexture(IDirect3DTexture9* tex) { texPerObjectLightData = tex; }
};

extern RenderThread* g_renderThread;
