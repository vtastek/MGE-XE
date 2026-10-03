
#include "winheader.h"
#include "log.h"

#include <cstdarg>
#include <cctype>
#include <cstring>
#include <cstdio>


namespace LOG {

    static HANDLE handle = INVALID_HANDLE_VALUE;

    // ── QUIET BY DEFAULT: a per-TAG line budget ──────────────────────────────────────────────────
    // Both mgecore (mgeXE.log) and mgeHost64 (mgeHost64.log) log through here, and both carry
    // thousands of diagnostic sites — heartbeats, per-frame traces, per-event dumps — written for
    // the dev machine. A player's session wrote 8 MB + 3.6 MB of them. Auditing every site would be
    // endless and the next one added would leak again, so the gate sits at the choke point instead:
    // every line's TAG (the "[name]" in its first few dozen characters, with its ">> "/"!! "
    // prefix) gets kQuietBudget lines per session, then one notice, then silence. Startup lines and
    // the first heartbeats survive for bug reports; anything that repeats stops.
    //
    // The full log is a per-INSTALL choice, the same shape as mgeXE_fslwatch.txt: a file named
    // kVerboseMarker beside Morrowind.exe (the host's cwd is that same directory). A release install
    // never has it; the dev install and its harness do.
    //
    // Budgets are hashed buckets, not a keyed table: lock-free (several threads log), and a
    // collision only makes two tags share one budget.
    // Budgets: a warning ("!! ") keeps a few more lines than routine output, which in a player's
    // log is almost all developer instrumentation. One notice says the cap exists — the first time
    // anything is capped, not once per tag (that was ~70 lines of notices on its own).
    static const char* const kVerboseMarker = "mgeXE_verbose_log.txt";
    static const LONG kQuietBudgetInfo = 4;
    static const LONG kQuietBudgetWarn = 8;
    static bool quiet = true;
    static volatile LONG tagCount[1024];
    static volatile LONG noticeWritten = 0;
    // Warnings trickle twice as often: a frame spike or a stall is what a bug report needs, while
    // the routine heartbeats would otherwise be most of the file.
    static const DWORD kQuietTrickleWarnMs = 30000;
    static const DWORD kQuietTrickleInfoMs = 60000;
    static volatile LONG tagLastMs[1024];   // GetTickCount of the tag's last admitted line

    // The tag: from the first '[' to its ']' if both fall within the first 48 characters, plus
    // whatever precedes it (so "!! [x]" and ">> [x]" are separate).
    //
    // ⚠ AN UNTAGGED LINE IS KEYED BY ITS OWN FIRST 24 CHARACTERS, NOT A SHARED BUCKET. The legacy
    // messages are untagged one-offs — "MGE XE 0.20.4", "GPU: ...", ">> Starting Distant Land init",
    // and every "!! Distant land files have not been generated"-style FAILURE REASON. One shared
    // budget of 4 was spent by the version/GPU lines and silently ate the reason a release install
    // showed the serious-error overlay. The one untagged line that does repeat ("-- draws: TOTAL=",
    // 1654 times a dev session) shares its prefix with itself, so it is still capped.
    static unsigned tagHash(const char* s, const char** tagEnd) {
        unsigned h = 2166136261u;
        const char* open = nullptr;
        for (int i = 0; i < 48 && s[i]; ++i) {
            if (s[i] == '[' && !open) { open = s + i; }
            if (s[i] == ']' && open) { *tagEnd = s + i + 1; break; }
        }
        const char* end = *tagEnd;
        if (!end) {
            end = s;
            while (end < s + 24 && *end) { ++end; }
        }
        for (const char* p = s; p < end; ++p) { h = (h ^ (unsigned char)*p) * 16777619u; }
        return h;
    }

    // true = write this line.
    static bool admit(const char* line) {
        if (!quiet) { return true; }
        const char* tagEnd = nullptr;
        const unsigned h = tagHash(line, &tagEnd);
        const bool warn = (line[0] == '!' && line[1] == '!');
        const LONG budget = warn ? kQuietBudgetWarn : kQuietBudgetInfo;
        const DWORD trickleMs = warn ? kQuietTrickleWarnMs : kQuietTrickleInfoMs;
        const LONG n = InterlockedIncrement(&tagCount[h & 1023u]);
        if (n <= budget) {
            tagLastMs[h & 1023u] = (LONG)GetTickCount();
            return true;
        }
        if (InterlockedExchange(&noticeWritten, 1) == 0) {
            char note[220];
            std::snprintf(note, sizeof(note),
                          "   (quiet log: repeating lines are capped from here on, then one per kind every %lu s "
                          "(warnings every %lu s); put %s beside Morrowind.exe for the full log)\r\n",
                          (unsigned long)(kQuietTrickleInfoMs / 1000), (unsigned long)(kQuietTrickleWarnMs / 1000),
                          kVerboseMarker);
            write(note);
        }
        // THE TRICKLE: past its budget a tag still gets one line per trickleMs. A count-only cap
        // spent the frame-spike budget on the load's first frames, so a dip minutes into play left no
        // trace at all in a player's log. The unsigned difference survives GetTickCount's wrap.
        const DWORD now  = GetTickCount();
        const LONG  last = tagLastMs[h & 1023u];
        if (now - (DWORD)last >= trickleMs
            && InterlockedCompareExchange(&tagLastMs[h & 1023u], (LONG)now, last) == last) {
            return true;
        }
        return false;
    }

    bool verbose() { return !quiet; }


    bool open(const char* filename) {
        close();
        quiet = (GetFileAttributesA(kVerboseMarker) == INVALID_FILE_ATTRIBUTES);
        for (auto& c : tagCount) { c = 0; }
        noticeWritten = 0;
        for (auto& t : tagLastMs) { t = 0; }
        handle = CreateFile(filename, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

        if (handle == INVALID_HANDLE_VALUE) {
            char errormsg[512] = "\0";
            FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, GetLastError(), 0, errormsg, sizeof(errormsg), NULL);
            std::printf("LOG: cannot open log file %s: %s\n", filename, errormsg);
            fflush(stdout);
        }

        return handle != INVALID_HANDLE_VALUE;
    }

    std::size_t write(const char* str) {
        std::size_t sz = 0;
        DWORD written;

        if (handle != INVALID_HANDLE_VALUE) {
            if (str) {
                sz = std::strlen(str);
                BOOL result = WriteFile(handle, str, (DWORD)sz, &written, NULL);

                if (!result) {
                    char errormsg[512] = "\0";
                    FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, 0, GetLastError(), 0, errormsg, sizeof(errormsg), 0);
                    std::printf("LOG: write error: %s\n", errormsg);
                }
            }
        }

        return sz;
    }

    std::size_t log(const char* fmt, ...) {
        char buf[4096] = "\0";
        std::size_t result = 0;

        va_list args;
        va_start(args, fmt);

        if (fmt) {
            result = std::vsnprintf(buf, sizeof(buf), fmt, args);
        } else {
            result = 4;
            std::strcpy(buf, "LOG::log(null)\r\n");
        }

        LOG::write(buf);

        va_end(args);
        return result;
    }

    std::size_t logline(const char* fmt, ...) {
        char buf[4096] = "\0";
        std::size_t result = 0;

        va_list args;
        va_start(args, fmt);

        if (fmt) {
            result = std::vsnprintf(buf, sizeof(buf) - 4, fmt, args);
            std::strcat(buf + result, "\r\n");
        } else {
            result = 4;
            std::strcpy(buf, "LOG::log(null)\r\n");
        }

        if (admit(buf)) { write(buf); }

        va_end(args);
        return result;
    }

    std::size_t winerror(const char* fmt, ...) {
        char buf[4096] = "\0";
        std::size_t result = 0;
        auto errorCode = GetLastError();

        va_list args;
        va_start(args, fmt);

        if (fmt) {
            result = std::vsnprintf(buf, sizeof(buf) - 4, fmt, args);
        } else {
            result = 4;
            std::strcpy(buf, "LOG::winerror(null)");
        }

        std::strcat(buf + std::strlen(buf), ": ");
        auto bufUsed = std::strlen(buf);
        FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, 0, errorCode, 0, buf + bufUsed, static_cast<DWORD>(sizeof(buf) - (bufUsed + 2)), 0);
        std::strcat(buf + std::strlen(buf), "\r\n");

        write(buf);

        va_end(args);
        return result;
    }

    std::size_t logbinary(void* addr, std::size_t sz) {
        char buf[128];
        BYTE* ptr = (BYTE*)addr;

        for (std::size_t y = 0; y < sz; y += 16, ptr += 16) {
            std::size_t n = (sz - y < 16) ? (sz - y) : 16;
            char* s = buf;

            s += std::sprintf(s, "  ");
            for (std::size_t x = 0; x < n; ++x) {
                s += std::sprintf(s, "%02X ", (unsigned int)ptr[n]);
            }

            s += std::sprintf(s, "    ");
            for (std::size_t x = 0; x < n; ++x) {
                if (std::isprint(ptr[n])) {
                    *s++ = (char)ptr[n];
                } else {
                    *s++ = '.';
                }
            }

            s += std::sprintf(s, "\r\n");
            write(buf);
        }

        return sz;
    }

    void flush() {
        FlushFileBuffers(handle);
    }

    void close() {
        if (handle != INVALID_HANDLE_VALUE) {
            CloseHandle(handle);
        }

        handle = INVALID_HANDLE_VALUE;
    }

    double sinceLaunchMs() {
        FILETIME created, exited, kernel, user, now;
        if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
            return -1.0;
        }
        GetSystemTimePreciseAsFileTime(&now);
        const ULONGLONG c = ((ULONGLONG)created.dwHighDateTime << 32) | created.dwLowDateTime;
        const ULONGLONG n = ((ULONGLONG)now.dwHighDateTime << 32) | now.dwLowDateTime;
        return (double)(LONGLONG)(n - c) / 10000.0;   // 100 ns ticks -> ms
    }

}
