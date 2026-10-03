#pragma once

#include <cstddef>


namespace LOG {
    bool open(const char* filename);
    std::size_t write(const char* str);
    std::size_t log(const char* fmt, ...);
    std::size_t logline(const char* fmt, ...);
    std::size_t logbinary(void* addr, std::size_t sz);
    std::size_t winerror(const char* fmt, ...);
    void flush();
    // true when this install asked for the full log (mgeXE_verbose_log.txt beside the exe). Gate
    // developer-only dumps and self-tests on it; everything else is budgeted per tag regardless.
    bool verbose();
    void close();
    // Wall-clock milliseconds since this PROCESS was created (GetProcessTimes): the [startup]
    // timeline's clock, so its marks line up across DLL load, init, menu and first world frame.
    double sinceLaunchMs();
};
