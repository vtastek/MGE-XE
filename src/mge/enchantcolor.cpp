// MW's per-item enchanted-glow colour — see enchantcolor.h for why this reads the plugin files
// rather than the running engine, and for the whole node -> colour chain.

#include "enchantcolor.h"
#include "support/log.h"

#include <Windows.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace MGE::EnchantColor {

    namespace {

        // effectID -> 0..1 RGB, last plugin to define it wins (engine load-order precedence).
        std::unordered_map<int, std::array<float, 3>> g_effectColor;
        bool g_initDone = false;
        bool g_initOk   = false;

        std::string trim(const std::string& s) {
            size_t a = s.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) return {};
            size_t b = s.find_last_not_of(" \t\r\n");
            return s.substr(a, b - a + 1);
        }

        std::string lower(std::string s) {
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return (char)::tolower(c); });
            return s;
        }

        // Morrowind.ini [Game Files], in ini order — which IS the engine's load order (the launcher
        // writes GameFile0..N already sorted: masters by date, then plugins). Mirrors the host's
        // terrain.cpp::readGameFiles; kept as its own copy because that lives in the x64 host and
        // this is the x86 client, with no shared TU between them.
        bool readGameFiles(std::vector<std::string>& out) {
            HANDLE h = CreateFileA("Morrowind.ini", GENERIC_READ, FILE_SHARE_READ, nullptr,
                                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h == INVALID_HANDLE_VALUE) return false;
            const DWORD size = GetFileSize(h, nullptr);
            std::string text;
            if (size != INVALID_FILE_SIZE && size > 0) {
                text.resize(size);
                DWORD got = 0;
                if (!ReadFile(h, text.data(), size, &got, nullptr)) { got = 0; }
                text.resize(got);
            }
            CloseHandle(h);
            if (text.empty()) return false;

            bool inSection = false;
            size_t pos = 0;
            while (pos <= text.size()) {
                size_t nl = text.find('\n', pos);
                if (nl == std::string::npos) nl = text.size();
                std::string line = trim(text.substr(pos, nl - pos));
                pos = nl + 1;
                if (line.empty()) continue;
                if (line[0] == '[') {
                    inSection = (lower(line).rfind("[game files]", 0) == 0);
                    continue;
                }
                if (!inSection) continue;
                if (lower(line).rfind("gamefile", 0) != 0) continue;
                size_t semi = line.find(';');
                if (semi != std::string::npos) line = line.substr(0, semi);
                size_t eq = line.find('=');
                if (eq == std::string::npos) continue;
                std::string name = trim(line.substr(eq + 1));
                if (!name.empty()) out.push_back(name);
            }
            return true;
        }

        // Walk one plugin's top-level records, pulling MGEF colours.
        //
        // TES3 record framing: [4cc][uint32 dataSize][uint32 header1][uint32 flags][data...]. MGEF
        // carries INDX (int32 effect id) and MEDT (school:int32, baseCost:float, flags:int32, then
        // red/green/blue as int32 — the layout MGEgui's own reader and every TES3 tool agree on).
        // Streamed in 1 MB chunks so a 200 MB plugin never lands in memory whole.
        void scanPlugin(const std::string& path, unsigned& added) {
            HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                   OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
            if (h == INVALID_HANDLE_VALUE) return;

            LARGE_INTEGER fsize{};
            GetFileSizeEx(h, &fsize);
            uint64_t offset = 0;
            std::vector<uint8_t> buf;

            auto readAt = [&](uint64_t at, uint32_t len, std::vector<uint8_t>& dst) -> bool {
                if (at + len > (uint64_t)fsize.QuadPart) return false;
                dst.resize(len);
                LARGE_INTEGER li; li.QuadPart = (LONGLONG)at;
                if (!SetFilePointerEx(h, li, nullptr, FILE_BEGIN)) return false;
                DWORD got = 0;
                if (!ReadFile(h, dst.data(), len, &got, nullptr) || got != len) return false;
                return true;
            };

            std::vector<uint8_t> hdr;
            while (offset + 16 <= (uint64_t)fsize.QuadPart) {
                if (!readAt(offset, 16, hdr)) break;
                char tag[5] = { (char)hdr[0], (char)hdr[1], (char)hdr[2], (char)hdr[3], 0 };
                uint32_t dataSize = 0;
                std::memcpy(&dataSize, hdr.data() + 4, 4);
                const uint64_t bodyAt = offset + 16;
                offset = bodyAt + dataSize;

                if (std::strcmp(tag, "MGEF") != 0) continue;
                if (dataSize == 0 || dataSize > (1u << 20)) continue;
                if (!readAt(bodyAt, dataSize, buf)) break;

                int  effectId = -1;
                bool haveId = false, haveColor = false;
                float rgb[3] = { 1.0f, 1.0f, 1.0f };

                size_t b = 0;
                while (b + 8 <= buf.size()) {
                    char st[5] = { (char)buf[b], (char)buf[b + 1], (char)buf[b + 2], (char)buf[b + 3], 0 };
                    uint32_t ss = 0;
                    std::memcpy(&ss, buf.data() + b + 4, 4);
                    const size_t sub = b + 8;
                    if (sub + ss > buf.size()) break;
                    if (std::strcmp(st, "INDX") == 0 && ss >= 4) {
                        std::memcpy(&effectId, buf.data() + sub, 4);
                        haveId = true;
                    } else if (std::strcmp(st, "MEDT") == 0 && ss >= 24) {
                        // school(4) baseCost(4) flags(4) then r,g,b as int32
                        int32_t r = 0, g = 0, bl = 0;
                        std::memcpy(&r,  buf.data() + sub + 12, 4);
                        std::memcpy(&g,  buf.data() + sub + 16, 4);
                        std::memcpy(&bl, buf.data() + sub + 20, 4);
                        auto clamp255 = [](int32_t v) {
                            return (float)(v < 0 ? 0 : (v > 255 ? 255 : v)) * (1.0f / 255.0f);
                        };
                        rgb[0] = clamp255(r); rgb[1] = clamp255(g); rgb[2] = clamp255(bl);
                        haveColor = true;
                    }
                    b = sub + ss;
                }

                if (haveId && haveColor && effectId >= 0) {
                    g_effectColor[effectId] = { rgb[0], rgb[1], rgb[2] };   // later plugin wins
                    ++added;
                }
            }
            CloseHandle(h);
        }

    } // namespace

    bool init() {
        if (g_initDone) return g_initOk;
        g_initDone = true;

        std::vector<std::string> plugins;
        if (!readGameFiles(plugins) || plugins.empty()) {
            LOG::logline("!! [enchant] Morrowind.ini [Game Files] unreadable or empty — "
                         "enchanted-item glow will use the fallback tint");
            return (g_initOk = false);
        }

        unsigned records = 0;
        for (const auto& p : plugins) {
            scanPlugin("Data Files\\" + p, records);
        }

        g_initOk = !g_effectColor.empty();
        LOG::logline(">> [enchant] MGEF colours: %u effects from %u plugin(s), %u record(s) read",
                     (unsigned)g_effectColor.size(), (unsigned)plugins.size(), records);
        return g_initOk;
    }

    bool colorForObject(const void* tes3Object, float outRGB[3]) {
        if (!tes3Object) return false;
        if (!g_initDone) init();
        if (!g_initOk) return false;

        // Object::getEnchantment is VIRTUAL (vtable slot 0xD0, MWSE-documented). Taking it that way
        // rather than by per-type field offset is what makes one path cover WEAP/ARMO/CLOT/BOOK —
        // each keeps its enchantment pointer at a different offset (0x74 / 0xC0 / 0xB4 / ...), and
        // hardcoding four of them would silently mis-read the fifth.
        const void* vtable = *reinterpret_cast<void* const*>(tes3Object);
        if (!vtable) return false;
        using GetEnchantFn = const void* (__thiscall*)(const void*);
        auto getEnchantment = *reinterpret_cast<GetEnchantFn const*>(
            static_cast<const char*>(vtable) + 0xD0);
        if (!getEnchantment) return false;

        const void* ench = getEnchantment(tes3Object);
        if (!ench) return false;

        // Enchantment::effects[8] @ 0x34; Effect::effectID is its first field (int16). MW colours
        // the glow from the FIRST effect, which is also what OpenMW's getEnchantmentColor reads.
        const int16_t effectId = *reinterpret_cast<const int16_t*>(
            static_cast<const char*>(ench) + 0x34);
        if (effectId < 0) return false;

        auto it = g_effectColor.find((int)effectId);
        if (it == g_effectColor.end()) return false;
        outRGB[0] = it->second[0];
        outRGB[1] = it->second[1];
        outRGB[2] = it->second[2];
        return true;
    }

    unsigned effectCount() {
        return (unsigned)g_effectColor.size();
    }

}
