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
    void close();

    // Category bitmask for spammy diagnostic lines. Bit set = enabled.
    // Default 0 (all spam off). Bound to mge.ini [Logging] via t_bit entries.
    enum Category : unsigned {
        Cat_HashDB      = 1u << 0,  // texture hash DB build / collisions
        Cat_FrameStats  = 1u << 1,  // DXVK passes, DIPs, Scene totals, DW:, water material
        Cat_HiZ         = 1u << 2,  // Hi-Z occlusion, recordMW filter, SLOW PREPARE
        Cat_Recording   = 1u << 3,  // renderFullFrameAsync, [ORDER], [REC-P], HLSL Recording
        Cat_HLSLReplay  = 1u << 4,  // CACHE HIT, Scene N draw, Bins, Lights, bindShaderTextures, SLOW REPLAY
        Cat_Mode3       = 1u << 5,  // forced LightMode3 packing/diag
        Cat_DistantLand = 1u << 6,  // [WVT] [WATER] [NVR] [N1-STORE] [PPDCAP] particles draw
        Cat_SyncThread  = 1u << 7,  // [CPT] [SN1] [SYNC] [S0E] [S0GPU] sync mode threading
    };
    extern unsigned g_categoryMask;
    inline bool catEnabled(unsigned cat) { return (g_categoryMask & cat) != 0; }
};

#define LOG_CAT(cat, ...) do { if (LOG::catEnabled(cat)) LOG::logline(__VA_ARGS__); } while (0)
