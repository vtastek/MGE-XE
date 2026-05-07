
#include "renderthread.h"
#include "distantland.h"
#include "ffeshader.h"
#include "imgui_manager.h"
#include "support/log.h"
#include "mge_tracy.h"
#include <atomic>
#include <thread>
#include <chrono>

// Device ownership flag — set by render thread, checked by main thread for race detection
extern std::atomic<bool> g_renderThreadOwnsDevice;

RenderThread* g_renderThread = nullptr;

RenderThread::~RenderThread() {
    stop();
}

void RenderThread::start(IDirect3DDevice9* d3dDevice) {
    if (thread.joinable()) {
        LOG::logline("!! RenderThread::start called but thread already running");
        return;
    }

    device = d3dDevice;
    shutdownRequested = false;
    state = State::Idle;
    hasPendingWork = false;

    thread = std::thread(&RenderThread::workerLoop, this);

    #ifdef _WIN32
    SetThreadDescription(thread.native_handle(), L"MGE_RenderThread");
    #endif

    LOG::logline("-- RenderThread started");
}

void RenderThread::stop() {
    if (!thread.joinable()) {
        return;
    }

    LOG::logline("-- RenderThread stopping...");

    {
        std::lock_guard<std::mutex> lock(mutex);
        shutdownRequested = true;
        pendingWork.type = WorkType::Shutdown;
        hasPendingWork = true;
    }
    workAvailable.notify_one();

    thread.join();

    // Clean up per-object light texture
    if (texPerObjectLightData) {
        texPerObjectLightData->Release();
        texPerObjectLightData = nullptr;
    }

    device = nullptr;

    LOG::logline("-- RenderThread stopped");
}

void RenderThread::workerLoop() {
    threadId = std::this_thread::get_id();  // Store thread ID for identification
    LOG::logline("-- RenderThread worker started");

    while (true) {
        SceneWork work;

        {
            MGE_ZoneScopedN("RenderThread_Idle");
            std::unique_lock<std::mutex> lock(mutex);
            workAvailable.wait(lock, [this] {
                return hasPendingWork || shutdownRequested;
            });

            if (shutdownRequested && (!hasPendingWork || pendingWork.type == WorkType::Shutdown)) {
                LOG::logline("-- RenderThread worker exiting");
                break;
            }

            work = std::move(pendingWork);
            hasPendingWork = false;
            state = State::Rendering;
        }

        {
            MGE_ZoneScopedN("RenderThread_Execute");

            switch (work.type) {
            case WorkType::RenderStage0:
                executeRenderStage0();
                break;
            case WorkType::RenderStage0Early:
                LOG::logline("[RT] S0E_START");
                g_renderThreadOwnsDevice.store(true, std::memory_order_release);
                executeRenderStage0Early(work.useN1Buffer);
                g_renderThreadOwnsDevice.store(false, std::memory_order_release);
                LOG::logline("[RT] S0E_DONE");
                break;
            case WorkType::RenderStage1:
                executeRenderStage1(work.ctx);
                break;
            case WorkType::RenderStage2:
                executeRenderStage2(work.ctx);
                break;
            case WorkType::ReplayHLSL:
                executeReplayHLSL(work.sceneCount);
                break;
            case WorkType::RenderFullFrame:
                g_renderThreadOwnsDevice.store(true, std::memory_order_release);
                executeFullFrame(work.bufferIndex, work.useN1Buffer);
                g_renderThreadOwnsDevice.store(false, std::memory_order_release);
                // Stress test: simulate slow GPU to catch race conditions
                if (ImGuiManager::GetStressAsyncDelay()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                break;
            case WorkType::RenderRemaining:
                g_renderThreadOwnsDevice.store(true, std::memory_order_release);
                executeRenderRemaining(work.useN1Buffer);
                g_renderThreadOwnsDevice.store(false, std::memory_order_release);
                break;
            case WorkType::Shutdown:
            case WorkType::None:
                break;
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            state = State::Complete;
        }
        workComplete.notify_one();
    }
}

void RenderThread::submitWork(SceneWork&& work, bool waitNow) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        pendingWork = std::move(work);
        hasPendingWork = true;
        state = State::Idle;
    }
    workAvailable.notify_one();

    if (waitNow) {
        waitForCompletion();
    }
}

void RenderThread::waitForCompletion() {
    std::unique_lock<std::mutex> lock(mutex);
    workComplete.wait(lock, [this] {
        return state == State::Complete || (state == State::Idle && !hasPendingWork);
    });
    state = State::Idle;
}

bool RenderThread::isComplete() const {
    // Must read both state and hasPendingWork atomically to avoid TOCTOU race.
    // Worker thread sets hasPendingWork=false, then state=Rendering under lock.
    // Without lock here, main thread could see hasPendingWork=false before state=Rendering,
    // incorrectly concluding work is complete and skipping the wait.
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex));
    return state == State::Complete || (state == State::Idle && !hasPendingWork);
}

void RenderThread::executeRenderStage0() {
    MGE_ZoneScopedN("RT_RenderStage0");
    DistantLand::renderStage0();
}

void RenderThread::executeRenderStage1(DLContext* ctx) {
    MGE_ZoneScopedN("RT_RenderStage1");
    DistantLand::renderStage1(ctx);
}

void RenderThread::executeRenderStage2(DLContext* ctx) {
    MGE_ZoneScopedN("RT_RenderStage2");
    DistantLand::renderStage2(ctx);
}

void RenderThread::executeReplayHLSL(int sceneCount) {
    MGE_ZoneScopedN("RT_ReplayHLSL");
    FixedFunctionShader::finalizeBatchAndReplay(sceneCount);
}

void RenderThread::executeFullFrame(int /*bufferIndex*/, bool useN1Buffer) {
    MGE_ZoneScopedN("RT_FullFrame");
    static int fullFrameLogCount = 0;
    if (fullFrameLogCount++ < 20) {
        LOG::logline("[RT] executeFullFrame: starting renderFullFrameAsync (N1=%d)", useN1Buffer ? 1 : 0);
    }
    FixedFunctionShader::renderFullFrameAsync(useN1Buffer);
    if (fullFrameLogCount <= 20) {
        LOG::logline("[RT] executeFullFrame: completed");
    }
}

void RenderThread::executeRenderStage0Early(bool useN1Buffer) {
    MGE_ZoneScopedN("RT_Stage0Early");
    FixedFunctionShader::renderStage0Early(useN1Buffer);
}

void RenderThread::executeRenderRemaining(bool useN1Buffer) {
    MGE_ZoneScopedN("RT_RenderRemaining");
    FixedFunctionShader::renderRemainingStages(useN1Buffer);
}
