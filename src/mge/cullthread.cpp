
#include "cullthread.h"
#include "ffeshader.h"
#include "support/log.h"
#include "mge_tracy.h"

CullThread* g_cullThread = nullptr;

CullThread::~CullThread() {
    stop();
}

void CullThread::start() {
    if (thread.joinable()) {
        LOG::logline("!! CullThread::start called but thread already running");
        return;
    }

    shutdownRequested = false;
    state = State::Idle;
    hasPendingWork = false;

    thread = std::thread(&CullThread::workerLoop, this);

    #ifdef _WIN32
    SetThreadDescription(thread.native_handle(), L"MGE_CullThread");
    #endif

    LOG::logline("-- CullThread started");
}

void CullThread::stop() {
    if (!thread.joinable()) {
        return;
    }

    LOG::logline("-- CullThread stopping...");

    {
        std::lock_guard<std::mutex> lock(mutex);
        shutdownRequested = true;
        hasPendingWork = true;
    }
    workAvailable.notify_one();

    thread.join();

    LOG::logline("-- CullThread stopped");
}

void CullThread::workerLoop() {
    LOG::logline("-- CullThread worker started");

    while (true) {
        int bufferIndex;

        {
            MGE_ZoneScopedN("CullThread_Idle");
            std::unique_lock<std::mutex> lock(mutex);
            workAvailable.wait(lock, [this] {
                return hasPendingWork || shutdownRequested;
            });

            if (shutdownRequested) {
                LOG::logline("-- CullThread worker exiting");
                break;
            }

            bufferIndex = pendingBufferIndex;
            hasPendingWork = false;
            state = State::Working;
        }

        {
            MGE_ZoneScopedN("CullThread_Execute");
            executeCull(bufferIndex);
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            state = State::Complete;
        }
        workComplete.notify_one();
    }
}

// Flag defined in ffeshader.cpp — prevents device calls in suffix fallback on cull thread
extern bool deviceCallsSafeInPrepare;

void CullThread::executeCull(int bufferIndex) {
    MGE_ZoneScopedN("executeCull");
    (void)bufferIndex;  // Single buffer now
    auto& fb = FixedFunctionShader::currentFrameBuffer();
    fb.state = FixedFunctionShader::BufferState::Culling;
    deviceCallsSafeInPrepare = false;
    FixedFunctionShader::executeCullPass();
    deviceCallsSafeInPrepare = true;
    fb.state = FixedFunctionShader::BufferState::ReadyToRender;
}

void CullThread::submitWork(int bufferIndex, bool waitNow) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        pendingBufferIndex = bufferIndex;
        hasPendingWork = true;
        state = State::Idle;
    }
    workAvailable.notify_one();

    if (waitNow) {
        waitForCompletion();
    }
}

void CullThread::waitForCompletion() {
    std::unique_lock<std::mutex> lock(mutex);
    workComplete.wait(lock, [this] {
        return state == State::Complete || (state == State::Idle && !hasPendingWork);
    });
    state = State::Idle;
}

bool CullThread::isComplete() const {
    return state == State::Complete || (state == State::Idle && !hasPendingWork);
}
