
#include "cpuprepthread.h"
#include "configuration.h"
#include "distantland.h"
#include "ffeshader.h"
#include "mged3d8device.h"
#include "renderthread.h"
#include "imgui_manager.h"
#include "support/log.h"
#include "mge_tracy.h"
#include <thread>
#include <chrono>

extern RenderThread* g_renderThread;
extern std::atomic<bool> g_skipSplitPathThisFrame;

CpuPrepThread* g_cpuPrepThread = nullptr;

CpuPrepThread::~CpuPrepThread() {
    stop();
}

void CpuPrepThread::start() {
    if (thread.joinable()) {
        LOG::logline("!! CpuPrepThread::start called but thread already running");
        return;
    }

    shutdownRequested = false;
    state = State::Idle;
    hasPendingWork = false;

    thread = std::thread(&CpuPrepThread::workerLoop, this);

    #ifdef _WIN32
    SetThreadDescription(thread.native_handle(), L"MGE_CpuPrepThread");
    #endif

    LOG::logline("-- CpuPrepThread started");
}

void CpuPrepThread::stop() {
    if (!thread.joinable()) {
        return;
    }

    LOG::logline("-- CpuPrepThread stopping...");

    {
        std::lock_guard<std::mutex> lock(mutex);
        shutdownRequested = true;
        pendingWork.type = WorkType::Shutdown;
        hasPendingWork = true;
    }
    workAvailable.notify_one();

    thread.join();

    LOG::logline("-- CpuPrepThread stopped");
}

void CpuPrepThread::workerLoop() {
    LOG::logline("-- CpuPrepThread worker started");

    while (true) {
        PrepWork work;

        {
            MGE_ZoneScopedN("CpuPrepThread_Idle");
            std::unique_lock<std::mutex> lock(mutex);
            workAvailable.wait(lock, [this] {
                return hasPendingWork || shutdownRequested;
            });

            if (shutdownRequested && (!hasPendingWork || pendingWork.type == WorkType::Shutdown)) {
                LOG::logline("-- CpuPrepThread worker exiting");
                break;
            }

            work = pendingWork;
            hasPendingWork = false;
            state = State::Working;
        }

        {
            MGE_ZoneScopedN("CpuPrepThread_Execute");

            switch (work.type) {
            case WorkType::PrepareFrame:
                executePrepareFrame(work.viewMatrix, work.projMatrix);
                // Stage0Early chaining DISABLED for sync mode
                // Reason: Stage0Early runs before Clear, which wipes depth buffer.
                // The fallback in renderRemainingStages runs after Clear, preserving depth.
                // Sync mode relies on the fallback path for correct depth ordering.
                // (Async mode doesn't use this code path - it submits GPU work at Present)
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

void CpuPrepThread::submitWork(PrepWork&& work, bool waitNow) {
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

void CpuPrepThread::waitForCompletion() {
    std::unique_lock<std::mutex> lock(mutex);
    workComplete.wait(lock, [this] {
        return state == State::Complete || (state == State::Idle && !hasPendingWork);
    });
    state = State::Idle;
}

bool CpuPrepThread::isComplete() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex));
    return state == State::Complete || (state == State::Idle && !hasPendingWork);
}

void CpuPrepThread::executePrepareFrame(const D3DXMATRIX& view, const D3DXMATRIX& proj) {
    MGE_ZoneScopedN("CPT_PrepareFrame");

    auto& fb = FixedFunctionShader::getPrepBuffer();
    int currentFrame = getFrameNumber();
    int prepFrame = fb.frameNumber;

    // Pipeline validation: prep should be N-1 (one frame behind current)
    int expectedPrepFrame = currentFrame - 1;
    if (prepFrame != expectedPrepFrame && prepFrame >= 0) {
        extern std::atomic<bool> g_skipSplitPathThisFrame;
        LOG::logline("!! CPT FRAME MISMATCH: current=%d prep=%d expected=%d (delta=%d) skipSplit=%d",
                     currentFrame, prepFrame, expectedPrepFrame, currentFrame - prepFrame,
                     g_skipSplitPathThisFrame.load(std::memory_order_acquire) ? 1 : 0);
    }

    // Skip if already prepared
    if (fb.state == FixedFunctionShader::BufferState::ReadyToRender) {
        LOG::logline("CPT: buffer already prepared (frame %d)", prepFrame);
        return;
    }

    // Phase 1: Prepare shader keys, state binning
    {
        MGE_ZoneScopedN("CPT_PrepareRecordedCalls");
        FixedFunctionShader::prepareRecordedCalls();
    }

    // Phase 2: Hi-Z culling (bbox, occluders, pyramid, visibility test)
    {
        MGE_ZoneScopedN("CPT_HiZCulling");
        FixedFunctionShader::executeHiZCulling(view, proj);
    }

    // Phase 3: Apply visibility and filter recordMW
    {
        MGE_ZoneScopedN("CPT_ApplyVisibility");
        FixedFunctionShader::applyVisibilityAndFilterRecordMW();
    }

    // Phase 4: Build merged batches (after culling so shouldRender is set)
    if (ImGuiManager::GetEnableStatelessBatch()) {
        MGE_ZoneScopedN("CPT_BuildStatelessBatches");
        FixedFunctionShader::buildStatelessBatches(fb);
    }

    // Mark buffer ready for GPU
    fb.state = FixedFunctionShader::BufferState::ReadyToRender;

    // Stress test: simulate slow CPU prep to catch race conditions
    if (ImGuiManager::GetStressAsyncDelay()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}
