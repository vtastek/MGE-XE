#include "renderthread.h"

#include <condition_variable>
#include <mutex>
#include <thread>

#include "support/log.h"
#include "mge_tracy.h"

namespace MGE::RenderThread {

    namespace {
        std::thread             g_thread;
        std::mutex              g_mtx;
        std::condition_variable g_cvWork;   // worker waits for a job here
        std::condition_variable g_cvDone;   // fence/stop wait for completion here
        std::function<void()>   g_job;
        bool                    g_hasJob  = false;  // job queued OR currently running
        bool                    g_stop    = false;
        bool                    g_started = false;

        void workerLoop() {
            for (;;) {
                std::function<void()> job;
                {
                    std::unique_lock<std::mutex> lk(g_mtx);
                    g_cvWork.wait(lk, [] { return g_stop || g_hasJob; });
                    if (g_stop) return;
                    job = std::move(g_job);
                    g_job = nullptr;
                    // g_hasJob stays true while the job runs so wait() blocks.
                }

                // Run unlocked — the job takes the device lock for its body.
                if (job) job();

                {
                    std::lock_guard<std::mutex> lk(g_mtx);
                    g_hasJob = false;
                }
                g_cvDone.notify_all();
            }
        }
    }

    void ensure() {
        if (!g_started) {
            g_started = true;
            g_stop = false;
            g_thread = std::thread(workerLoop);
            LOG::logline("-- [RENDERTHREAD] worker spawned");
        }
    }

    void kick(std::function<void()> job) {
        ensure();
        {
            std::unique_lock<std::mutex> lk(g_mtx);
            // Defensive drain: only one job may be in flight. In normal operation
            // the prior job was already fenced (wait()) before this kick, so this
            // returns immediately; it guards the abnormal path where renderStage0
            // never ran after a kick.
            g_cvDone.wait(lk, [] { return !g_hasJob; });
            g_job = std::move(job);
            g_hasJob = true;
        }
        g_cvWork.notify_one();
    }

    void wait() {
        MGE_ZoneScopedN("RenderThread:wait");
        std::unique_lock<std::mutex> lk(g_mtx);
        g_cvDone.wait(lk, [] { return !g_hasJob; });
    }

    void stop() {
        if (!g_started) return;

        // Drain any in-flight job first (it may still be touching the device /
        // render targets that are about to be released).
        {
            std::unique_lock<std::mutex> lk(g_mtx);
            g_cvDone.wait(lk, [] { return !g_hasJob; });
            g_stop = true;
        }
        g_cvWork.notify_one();
        if (g_thread.joinable()) g_thread.join();

        g_started = false;
        g_stop = false;
        LOG::logline("-- [RENDERTHREAD] worker joined");
    }

}
