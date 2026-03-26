
#include "cpuprepthread.h"
#include "ffeshader.h"
#include "mged3d8device.h"
#include "support/log.h"
#include "mge_tracy.h"

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
        LOG::logline("!! CPT FRAME MISMATCH: current=%d prep=%d expected=%d (delta=%d)",
                     currentFrame, prepFrame, expectedPrepFrame, currentFrame - prepFrame);
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

    // Mark buffer ready for GPU
    fb.state = FixedFunctionShader::BufferState::ReadyToRender;
}
