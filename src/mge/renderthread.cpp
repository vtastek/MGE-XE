
#include "renderthread.h"
#include "distantland.h"
#include "ffeshader.h"
#include "support/log.h"
#include "mge_tracy.h"

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
    device = nullptr;

    LOG::logline("-- RenderThread stopped");
}

void RenderThread::workerLoop() {
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
                executeFullFrame(work.bufferIndex);
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

void RenderThread::executeFullFrame(int bufferIndex) {
    MGE_ZoneScopedN("RT_FullFrame");
    FixedFunctionShader::executeRenderPass(bufferIndex);
}
