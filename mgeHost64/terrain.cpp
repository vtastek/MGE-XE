// mgeHost64 — host-owned terrain, T0: the LAND record loader. See terrain.h for the why.
//
// Ported from MGEgui/DistantLand/DistantLandForm.cs:496-573 (the plugin walk) — that parser is
// correct and battle-tested; everything DOWNSTREAM of it (decimated mesh, global atlas) is what
// this work replaces. Two traps in that port are load-bearing and easy to get silently wrong:
//
//   1. VTEX is stored INTERLEAVED as 4x4 blocks of 4x4, not row-major (see readVtex).
//   2. The LTEX index space is PER-PLUGIN. MGEgui rebuilds its `Textures` dictionary inside the
//      plugin loop and each LAND captures its own plugin's table. A single global index table
//      mis-textures the world the moment two plugins both define LTEX records.
//
// Both produce a plausible-looking world with the wrong textures rather than a crash, so the
// census log at the bottom of load() is the check that they are right.
//
// ABI note: built with the host's DEFAULT MSVC settings, deliberately NOT The Forge's
// (_HAS_EXCEPTIONS=0 + IMemory.h's new/delete override). Nothing allocated here is ever freed by
// forgerender.cpp — it sees POD through terrain.h only.

#include "terrain.h"

#include "support/winheader.h"
#include "support/log.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Terrain {

    namespace {

        // ---------------------------------------------------------------- small helpers

        bool tagIs(const uint8_t* p, const char (&t)[5]) {
            return p[0] == (uint8_t)t[0] && p[1] == (uint8_t)t[1]
                && p[2] == (uint8_t)t[2] && p[3] == (uint8_t)t[3];
        }

        std::string lower(std::string s) {
            for (char& c : s) {
                if (c >= 'A' && c <= 'Z') { c = (char)(c - 'A' + 'a'); }
                if (c == '/') { c = '\\'; }
            }
            return s;
        }

        std::string trim(const std::string& s) {
            size_t a = 0, b = s.size();
            while (a < b && (unsigned char)s[a] <= ' ') { ++a; }
            while (b > a && (unsigned char)s[b - 1] <= ' ') { --b; }
            return s.substr(a, b - a);
        }

        // Read-only whole-file mapping. The plugin set here is ~400 MB; mapping keeps it in the OS
        // page cache with no copy, and only the LAND/LTEX bodies are ever touched.
        struct MappedFile {
            HANDLE         file    = INVALID_HANDLE_VALUE;
            HANDLE         mapping = nullptr;
            const uint8_t* data    = nullptr;
            uint64_t       size    = 0;

            bool open(const char* path) {
                close();
                file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                if (file == INVALID_HANDLE_VALUE) { return false; }
                LARGE_INTEGER li = {};
                if (!GetFileSizeEx(file, &li) || li.QuadPart <= 0) { close(); return false; }
                size = (uint64_t)li.QuadPart;
                mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
                if (!mapping) { close(); return false; }
                data = (const uint8_t*)MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
                if (!data) { close(); return false; }
                return true;
            }
            void close() {
                if (data)    { UnmapViewOfFile(data); data = nullptr; }
                if (mapping) { CloseHandle(mapping);  mapping = nullptr; }
                if (file != INVALID_HANDLE_VALUE) { CloseHandle(file); file = INVALID_HANDLE_VALUE; }
                size = 0;
            }
            ~MappedFile() { close(); }
        };

        // ---------------------------------------------------------------- state

        std::mutex              g_mutex;
        std::condition_variable g_cv;
        bool                    g_started = false;
        bool                    g_loaded  = false;

        std::vector<LandCell>                    g_cells;
        std::unordered_map<uint64_t, uint32_t>   g_cellIndex;    // packed (x,y) -> index into g_cells
        std::vector<std::string>                 g_texNames;     // global land-texture table
        std::unordered_map<std::string, uint32_t> g_texIds;      // lowercased name -> global id
        int32_t  g_minX = 0, g_minY = 0, g_maxX = 0, g_maxY = 0;
        // The ONE synthetic cell every LAND-less grid position renders from (see defaultSlot()).
        // ~0u until load() appends it; g_cellIndex never points at it, so cellAt/slotAt keep
        // meaning "a real LAND record exists here".
        uint32_t g_defaultSlot = ~0u;

        std::vector<uint32_t>    g_packHeights;                  // GPU staging: 2 int16 per uint
        std::vector<uint32_t>    g_packColors;                   // GPU staging: 0x00BBGGRR per vertex
        std::vector<uint32_t>    g_packTex;                      // GPU staging: 2 uint16 tex ids per uint

        // --- land texture files (see readLandTextureFile) ---
        struct BsaEntry { uint32_t archive, offset, size; };
        std::unordered_map<std::string, std::string> g_looseTex;  // "sub\name.dds" -> real full path
        std::unordered_map<std::string, BsaEntry>    g_bsaTex;    // "textures\sub\name.dds" -> entry
        std::vector<HANDLE>      g_bsaHandles;
        std::vector<uint8_t>     g_texScratch;                    // readLandTextureFile's one buffer
        bool                     g_texIndexBuilt = false;

        uint64_t packKey(int32_t x, int32_t y) {
            return ((uint64_t)(uint32_t)x << 32) | (uint64_t)(uint32_t)y;
        }

        // ---------------------------------------------------------------- Morrowind.ini

        // Parse `[Game Files]` out of Morrowind.ini, preserving ini order — which IS the engine's
        // load order (the launcher writes GameFile0..N already sorted: masters by date, then plugins).
        // The ini is in the system codepage; plugin names are treated as opaque bytes.
        bool readGameFiles(std::vector<std::string>& out) {
            MappedFile ini;
            if (!ini.open("Morrowind.ini")) { return false; }
            const std::string text((const char*)ini.data, (size_t)ini.size);

            bool inSection = false;
            size_t pos = 0;
            while (pos <= text.size()) {
                size_t nl = text.find('\n', pos);
                if (nl == std::string::npos) { nl = text.size(); }
                std::string line = trim(text.substr(pos, nl - pos));
                pos = nl + 1;
                if (line.empty()) { continue; }
                if (line[0] == '[') {
                    inSection = (lower(line).rfind("[game files]", 0) == 0);
                    continue;
                }
                if (!inSection) { continue; }
                if (lower(line).rfind("gamefile", 0) != 0) { continue; }
                size_t semi = line.find(';');
                if (semi != std::string::npos) { line = line.substr(0, semi); }
                size_t eq = line.find('=');
                if (eq == std::string::npos) { continue; }
                std::string name = trim(line.substr(eq + 1));
                if (!name.empty()) { out.push_back(name); }
            }
            return true;
        }

        // ---------------------------------------------------------------- land texture files

        // Recursive case-insensitive index of Data Files\Textures\**. Keys are lowercased paths
        // RELATIVE to that root with backslashes, which is the shape LTEX names come in.
        void indexTextureDir(const std::string& absDir, const std::string& relPrefix, uint32_t depth) {
            if (depth > 8) { return; }                      // pathological symlink/junction guard
            WIN32_FIND_DATAA fd;
            HANDLE h = FindFirstFileA((absDir + "\\*").c_str(), &fd);
            if (h == INVALID_HANDLE_VALUE) { return; }
            do {
                const std::string name = fd.cFileName;
                if (name == "." || name == "..") { continue; }
                if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                    indexTextureDir(absDir + "\\" + name, relPrefix + lower(name) + "\\", depth + 1);
                } else {
                    g_looseTex.emplace(relPrefix + lower(name), absDir + "\\" + name);
                }
            } while (FindNextFileA(h, &fd));
            FindClose(h);
        }

        // Morrowind BSA directory. Layout per MGEgui/DistantLand/BSA.cs:
        //   u32 version, u32 hashOffset, u32 fileCount,
        //   fileCount * { u32 size, u32 offset }   at 12
        //   fileCount * u32 nameOffset             at 12 + count*8
        //   name table                             at 12 + count*12
        //   data starts at 12 + hashOffset + count*8   (the per-file offset is relative to that)
        void indexBsa(const std::string& path) {
            MappedFile mf;
            if (!mf.open(path.c_str()) || mf.size < 12) { return; }
            uint32_t hashOffset, count;
            std::memcpy(&hashOffset, mf.data + 4, 4);
            std::memcpy(&count,      mf.data + 8, 4);
            const uint64_t need = 12ull + (uint64_t)count * 12ull;
            if (!count || need > mf.size) { return; }

            // Keep the archive open for the life of the process — the reads are on demand.
            HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h == INVALID_HANDLE_VALUE) { return; }
            const uint32_t archive  = (uint32_t)g_bsaHandles.size();
            g_bsaHandles.push_back(h);

            const uint32_t dataBase  = 12u + hashOffset + count * 8u;
            const uint8_t* sizeTable = mf.data + 12;
            const uint8_t* nameOffs  = mf.data + 12 + (uint64_t)count * 8;
            const uint8_t* nameTable = mf.data + 12 + (uint64_t)count * 12;
            const uint8_t* end       = mf.data + mf.size;
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t sz, off, no;
                std::memcpy(&sz,  sizeTable + (uint64_t)i * 8,     4);
                std::memcpy(&off, sizeTable + (uint64_t)i * 8 + 4, 4);
                std::memcpy(&no,  nameOffs  + (uint64_t)i * 4,     4);
                const uint8_t* p = nameTable + no;
                if (p >= end) { continue; }
                uint32_t len = 0;
                while (p + len < end && p[len]) { ++len; }
                std::string nm = lower(std::string((const char*)p, len));
                if (nm.rfind("textures\\", 0) != 0) { continue; }   // only textures are ever wanted
                BsaEntry e; e.archive = archive; e.offset = dataBase + off; e.size = sz;
                g_bsaTex.emplace(std::move(nm), e);
            }
        }

        bool readLoose(const std::string& fullPath, std::vector<uint8_t>& out) {
            MappedFile mf;
            if (!mf.open(fullPath.c_str())) { return false; }
            out.assign(mf.data, mf.data + (size_t)mf.size);
            return true;
        }

        bool readBsa(const BsaEntry& e, std::vector<uint8_t>& out) {
            if (e.archive >= g_bsaHandles.size()) { return false; }
            HANDLE h = g_bsaHandles[e.archive];
            LARGE_INTEGER li; li.QuadPart = (LONGLONG)e.offset;
            if (!SetFilePointerEx(h, li, nullptr, FILE_BEGIN)) { return false; }
            out.resize(e.size);
            DWORD got = 0;
            if (!ReadFile(h, out.data(), e.size, &got, nullptr) || got != e.size) { return false; }
            return true;
        }

        // ---------------------------------------------------------------- LAND subrecords

        // VHGT: float row-0 offset + 65*65 int8 deltas. The accumulator is ROW-RELATIVE: at the end
        // of each row it resets to that row's column-0 value, so H[0,y] = H[0,y-1] + d and the rest
        // of the row runs along from there. (DistantLandForm.cs:515-524 — the `offset =
        // land.Heights[0, y]` line is the whole trick, and dropping it shears the world.)
        void readVhgt(const uint8_t* sd, uint32_t sz, LandCell& c) {
            if (sz < 4u + kCellVerts * kCellVerts) { return; }
            float f; std::memcpy(&f, sd, 4);
            int32_t offset = (int32_t)f;                     // C# (int) cast truncates toward zero
            const int8_t* d = (const int8_t*)(sd + 4);
            for (int y = 0; y < kCellVerts; ++y) {
                for (int x = 0; x < kCellVerts; ++x) {
                    offset += *d++;
                    int32_t h = offset;
                    if (h >  32767) { h =  32767; }
                    if (h < -32768) { h = -32768; }
                    c.height[y * kCellVerts + x] = (int16_t)h;
                }
                offset = c.height[y * kCellVerts];           // row-relative reset
            }
        }

        void readVclr(const uint8_t* sd, uint32_t sz, LandCell& c) {
            const uint32_t need = (uint32_t)(kCellVerts * kCellVerts * 3);
            if (sz < need) { return; }
            std::memcpy(c.color, sd, need);
            c.hasColor = 1;
        }

        // VTEX: 16x16 int16 texture indices stored as 4x4 BLOCKS of 4x4, read in (blockY, blockX,
        // inY, inX) order. Row-major reading here produces a fully plausible world with the
        // textures shuffled inside every 4x4 patch — silent corruption, hence the explicit loop.
        void readVtex(const uint8_t* sd, uint32_t sz, LandCell& c) {
            const uint32_t need = (uint32_t)(kCellTexRes * kCellTexRes * 2);
            if (sz < need) { return; }
            const uint8_t* p = sd;
            for (int y1 = 0; y1 < 4; ++y1) {
                for (int x1 = 0; x1 < 4; ++x1) {
                    for (int y2 = 0; y2 < 4; ++y2) {
                        for (int x2 = 0; x2 < 4; ++x2) {
                            uint16_t v; std::memcpy(&v, p, 2); p += 2;
                            c.tex[(y1 * 4 + y2) * kCellTexRes + (x1 * 4 + x2)] = v;
                        }
                    }
                }
            }
        }

        // ---------------------------------------------------------------- the load

        struct LoadStats {
            uint32_t plugins = 0, missingPlugins = 0;
            uint32_t landRecords = 0, cellsNew = 0, cellsOverridden = 0;
            uint32_t skippedNoHeights = 0, skippedOutOfRange = 0, missingVhgt = 0;
            uint32_t ltexRecords = 0, badTexIndices = 0;
            uint64_t bytesWalked = 0;
            // (x,y) of every LAND record dropped because DATA bit 0 was clear (no vertex heights).
            // Most are overrides of a cell that IS loaded elsewhere; the ones that are NOT are real
            // holes in the world, which is exactly what T3 must fail loudly on rather than draw
            // nothing once MW's own near land is gone.
            std::vector<uint64_t> heightlessKeys;
        };

        void load() {
            const auto t0 = std::chrono::steady_clock::now();
            LoadStats st;

            std::vector<std::string> plugins;
            if (!readGameFiles(plugins) || plugins.empty()) {
                LOG::logline("!! [terrain] Morrowind.ini [Game Files] unreadable or empty — no terrain");
                LOG::flush();
                std::lock_guard<std::mutex> lk(g_mutex);
                g_loaded = true;
                g_cv.notify_all();
                return;
            }

            g_cells.reserve(4096);
            g_texNames.clear();
            g_texIds.clear();
            g_texNames.push_back("_land_default.tga");        // global id 0 = MGE's default ground
            g_texIds[g_texNames[0]] = 0;

            // Per-cell stamp of the plugin ordinal that last wrote it, so a plugin that defines the
            // same cell twice doesn't get its VTEX remapped twice at plugin end.
            std::vector<uint32_t> cellStamp;
            cellStamp.reserve(4096);

            bool anyExtent = false;

            for (uint32_t pi = 0; pi < (uint32_t)plugins.size(); ++pi) {
                const std::string path = "Data Files\\" + plugins[pi];
                MappedFile mf;
                if (!mf.open(path.c_str())) {
                    LOG::logline("!! [terrain] plugin missing: %s", path.c_str());
                    ++st.missingPlugins;
                    continue;
                }
                ++st.plugins;
                st.bytesWalked += mf.size;

                // PER-PLUGIN LTEX index space (trap 2). local VTEX index -> global texture id.
                // Index 0 is always the default; MGEgui stores LTEX INTV + 1, so VTEX 0 means
                // "no LTEX", not "the first one".
                std::unordered_map<uint32_t, uint32_t> localToGlobal;
                localToGlobal[0] = 0;

                // Cells this plugin wrote, holding RAW LOCAL VTEX indices until the plugin ends.
                // Deferred so an LTEX record appearing after a LAND record still resolves — MGEgui
                // gets this for free by sharing one mutable dictionary per plugin.
                std::vector<uint32_t> pluginCells;

                const uint8_t* p   = mf.data;
                const uint8_t* end = mf.data + mf.size;
                while (p + 16 <= end) {
                    const uint8_t* tag = p;
                    uint32_t recSize; std::memcpy(&recSize, p + 4, 4);
                    const uint8_t* recData = p + 16;                 // tag(4) + size(4) + 8 skipped
                    if ((uint64_t)recSize > (uint64_t)(end - recData)) { break; }
                    const uint8_t* recEnd = recData + recSize;

                    if (tagIs(tag, "LAND")) {
                        ++st.landRecords;
                        LandCell c = {};
                        for (int i = 0; i < kCellVerts * kCellVerts; ++i) { c.height[i] = -256; }
                        std::memset(c.color, 0xFF, sizeof(c.color));   // LAND() ctor: white, up, -256
                        int32_t lx = 0, ly = 0;
                        bool haveIntv = false, usesVertexHeights = true, haveVhgt = false;

                        const uint8_t* q = recData;
                        while (q + 8 <= recEnd) {
                            const uint8_t* sub = q;
                            uint32_t subSize; std::memcpy(&subSize, q + 4, 4);
                            const uint8_t* sd = q + 8;
                            if (subSize > (uint32_t)(recEnd - sd)) { break; }

                            if (tagIs(sub, "INTV") && subSize >= 8) {
                                std::memcpy(&lx, sd, 4); std::memcpy(&ly, sd + 4, 4);
                                haveIntv = true;
                            } else if (tagIs(sub, "DATA") && subSize >= 4) {
                                int32_t flags; std::memcpy(&flags, sd, 4);
                                usesVertexHeights = (flags & 1) == 1;
                            } else if (tagIs(sub, "VHGT")) {
                                readVhgt(sd, subSize, c);
                                haveVhgt = true;
                            } else if (tagIs(sub, "VCLR")) {
                                readVclr(sd, subSize, c);
                            } else if (tagIs(sub, "VTEX")) {
                                readVtex(sd, subSize, c);
                            }
                            // VNML is deliberately skipped — normals come from the heightfield.
                            q = sd + subSize;
                        }

                        if (!usesVertexHeights || !haveIntv) {
                            ++st.skippedNoHeights;
                            if (haveIntv) { st.heightlessKeys.push_back(packKey(lx, ly)); }
                        } else if (lx < -4096 || lx > 4096 || ly < -4096 || ly > 4096) {
                            LOG::logline("!! [terrain] cell (%d,%d) in %s is absurdly far from origin — dropped",
                                         lx, ly, plugins[pi].c_str());
                            ++st.skippedOutOfRange;
                        } else {
                            if (!haveVhgt) { ++st.missingVhgt; }
                            c.cellX = lx; c.cellY = ly;
                            int16_t lo = c.height[0], hi = c.height[0];
                            for (int i = 1; i < kCellVerts * kCellVerts; ++i) {
                                if (c.height[i] < lo) { lo = c.height[i]; }
                                if (c.height[i] > hi) { hi = c.height[i]; }
                            }
                            c.minHeight = lo; c.maxHeight = hi;

                            const uint64_t key = packKey(lx, ly);
                            auto it = g_cellIndex.find(key);
                            uint32_t idx;
                            if (it == g_cellIndex.end()) {
                                idx = (uint32_t)g_cells.size();
                                g_cells.push_back(c);
                                cellStamp.push_back(0xFFFFFFFFu);
                                g_cellIndex[key] = idx;
                                ++st.cellsNew;
                            } else {
                                idx = it->second;
                                g_cells[idx] = c;               // later plugins win, whole record
                                ++st.cellsOverridden;
                            }
                            if (cellStamp[idx] != pi) { cellStamp[idx] = pi; pluginCells.push_back(idx); }

                            if (!anyExtent) {
                                g_minX = g_maxX = lx; g_minY = g_maxY = ly; anyExtent = true;
                            } else {
                                g_minX = std::min(g_minX, lx); g_maxX = std::max(g_maxX, lx);
                                g_minY = std::min(g_minY, ly); g_maxY = std::max(g_maxY, ly);
                            }
                        }
                    } else if (tagIs(tag, "LTEX")) {
                        ++st.ltexRecords;
                        uint32_t localIndex = 0;
                        std::string file;
                        bool haveIntv = false;

                        const uint8_t* q = recData;
                        while (q + 8 <= recEnd) {
                            const uint8_t* sub = q;
                            uint32_t subSize; std::memcpy(&subSize, q + 4, 4);
                            const uint8_t* sd = q + 8;
                            if (subSize > (uint32_t)(recEnd - sd)) { break; }

                            if (tagIs(sub, "INTV") && subSize >= 4) {
                                int32_t v; std::memcpy(&v, sd, 4);
                                localIndex = (uint32_t)(v + 1);          // VTEX indices are 1-based
                                haveIntv = true;
                            } else if (tagIs(sub, "DATA")) {
                                uint32_t len = 0;
                                while (len < subSize && sd[len] != 0) { ++len; }
                                file.assign((const char*)sd, len);
                            }
                            q = sd + subSize;
                        }
                        if (haveIntv && !file.empty()) {
                            const std::string key = lower(file);
                            auto tit = g_texIds.find(key);
                            uint32_t gid;
                            if (tit == g_texIds.end()) {
                                gid = (uint32_t)g_texNames.size();
                                g_texNames.push_back(key);
                                g_texIds[key] = gid;
                            } else {
                                gid = tit->second;
                            }
                            localToGlobal[localIndex] = gid;   // last LTEX at this index wins
                        }
                    }

                    p = recEnd;
                }

                // Plugin done — resolve this plugin's cells from ITS index space (trap 2).
                for (uint32_t idx : pluginCells) {
                    LandCell& c = g_cells[idx];
                    for (int i = 0; i < kCellTexRes * kCellTexRes; ++i) {
                        auto lit = localToGlobal.find((uint32_t)c.tex[i]);
                        if (lit == localToGlobal.end()) {
                            if (c.tex[i] != 0) { ++st.badTexIndices; }
                            c.tex[i] = 0;                        // MGEgui: missing index -> default
                        } else {
                            c.tex[i] = (uint16_t)lit->second;
                        }
                    }
                }
            }

            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0).count();

            uint32_t noColor = 0;
            int16_t worldLo = 0, worldHi = 0;
            for (size_t i = 0; i < g_cells.size(); ++i) {
                if (!g_cells[i].hasColor) { ++noColor; }
                if (i == 0) { worldLo = g_cells[i].minHeight; worldHi = g_cells[i].maxHeight; }
                worldLo = std::min(worldLo, g_cells[i].minHeight);
                worldHi = std::max(worldHi, g_cells[i].maxHeight);
            }
            const uint64_t bytes = (uint64_t)g_cells.size() * sizeof(LandCell);

            LOG::logline(">> [terrain] plugins: %u loaded, %u MISSING (%.1f MB walked)",
                         st.plugins, st.missingPlugins, (double)st.bytesWalked / (1024.0 * 1024.0));
            LOG::logline(">> [terrain] LAND: %u cells  x[%d..%d] y[%d..%d]  (%u records,"
                         " %u overrides, %u no-heights, %u out-of-range, %u no-VHGT, %u no-VCLR)",
                         (uint32_t)g_cells.size(), g_minX, g_maxX, g_minY, g_maxY,
                         st.landRecords, st.cellsOverridden, st.skippedNoHeights,
                         st.skippedOutOfRange, st.missingVhgt, noColor);
            LOG::logline(">> [terrain] LTEX: %u unique textures (%u records, %u unresolved VTEX indices)",
                         (uint32_t)g_texNames.size() - 1u, st.ltexRecords, st.badTexIndices);

            // Heightless LAND records that no OTHER plugin covers = cells with no terrain at all.
            std::sort(st.heightlessKeys.begin(), st.heightlessKeys.end());
            st.heightlessKeys.erase(std::unique(st.heightlessKeys.begin(), st.heightlessKeys.end()),
                                    st.heightlessKeys.end());
            uint32_t voids = 0;
            std::string voidList;
            for (uint64_t k : st.heightlessKeys) {
                if (g_cellIndex.find(k) != g_cellIndex.end()) { continue; }
                ++voids;
                if (voids <= 24) {
                    char b[32];
                    std::snprintf(b, sizeof(b), "%s(%d,%d)", voids > 1 ? " " : "",
                                  (int32_t)(k >> 32), (int32_t)(k & 0xFFFFFFFFu));
                    voidList += b;
                }
            }
            LOG::logline(">> [terrain] heightless LAND records: %u (%u are cells with NO terrain%s%s)",
                         (uint32_t)st.heightlessKeys.size(), voids,
                         voids ? ": " : "", voidList.c_str());
            LOG::logline(">> [terrain] height range %d..%d world units;  resident %llu MB;  parsed in %.0f ms",
                         (int)(worldLo * kHeightScale), (int)(worldHi * kHeightScale),
                         (unsigned long long)(bytes >> 20), ms);
            LOG::flush();
            std::printf("[forge][terrain] %u cells x[%d..%d] y[%d..%d], %u textures, %llu MB, %.0f ms\n",
                        (uint32_t)g_cells.size(), g_minX, g_maxX, g_minY, g_maxY,
                        (uint32_t)g_texNames.size() - 1u, (unsigned long long)(bytes >> 20), ms);

            // DEFAULT LAND — the one synthetic cell that fills every grid position with no LAND
            // record. MW draws those positions as a flat sheet at ESM::Land::DEFAULT_HEIGHT
            // (-2048 world units), textured _land_default, white vertex colour, normal +Z; we
            // drew nothing, which is the hole MW's own frame shows through out at sea and in the
            // gaps the "no terrain" list above names.
            //
            // Nothing has to be AUTHORED for that: the zero-initialised LandCell built at the top
            // of the LAND branch already IS it — height -256 VHGT × kHeightScale 8 = -2048, colour
            // memset white, and global texture id 0 is _land_default. The cell only has to exist.
            //
            // ONE cell, shared. A LandCell is ~21.6 KB and this install has ~42k empty positions
            // inside the padded grid: materialising one each would be ~900 MB of identical bytes.
            // The shaders index the world buffers by the instance's SLOT, so many instances can
            // point at one slot — see the cull-index/slot split in forgerender.cpp.
            //
            // Appended AFTER every stat and log line above, deliberately: the extent, the height
            // range, the cell count and the no-VCLR count all describe the WORLD, and a synthetic
            // cell at (0,0) with hasColor=0 would corrupt all four. buildGpuPack() picks it up for
            // free (it walks g_cells), costing one extra slot in heights/colour/VTEX ≈ 26 KB.
            {
                LandCell d = {};
                for (int i = 0; i < kCellVerts * kCellVerts; ++i) { d.height[i] = -256; }
                std::memset(d.color, 0xFF, sizeof(d.color));
                d.cellX = 0; d.cellY = 0;
                d.minHeight = d.maxHeight = -256;
                d.hasColor = 0;                        // tex[] stays 0 = _land_default
                g_defaultSlot = (uint32_t)g_cells.size();
                g_cells.push_back(d);
                LOG::logline(">> [terrain] default-land slot %u appended (flat %d world units, "
                             "texture id 0 '%s') — every LAND-less grid position renders from it",
                             g_defaultSlot, (int)(-256 * kHeightScale), g_texNames[0].c_str());
                LOG::flush();
            }

            {
                std::lock_guard<std::mutex> lk(g_mutex);
                g_loaded = true;
            }
            g_cv.notify_all();
        }

    }   // namespace

    // -------------------------------------------------------------------- public API

    void beginLoadAsync() {
        {
            std::lock_guard<std::mutex> lk(g_mutex);
            if (g_started) { return; }
            g_started = true;
        }
        std::thread(load).detach();
    }

    bool isLoaded() {
        std::lock_guard<std::mutex> lk(g_mutex);
        return g_loaded;
    }

    bool waitLoaded(uint32_t timeoutMs) {
        std::unique_lock<std::mutex> lk(g_mutex);
        if (!g_loaded) {
            g_cv.wait_for(lk, std::chrono::milliseconds(timeoutMs), [] { return g_loaded; });
        }
        return g_loaded;
    }

    uint32_t        cellCount() { return (uint32_t)g_cells.size(); }
    const LandCell* cells()     { return g_cells.empty() ? nullptr : g_cells.data(); }

    const LandCell* cellAt(int32_t x, int32_t y) {
        auto it = g_cellIndex.find(packKey(x, y));
        return (it == g_cellIndex.end()) ? nullptr : &g_cells[it->second];
    }

    void extent(int32_t& minX, int32_t& minY, int32_t& maxX, int32_t& maxY) {
        minX = g_minX; minY = g_minY; maxX = g_maxX; maxY = g_maxY;
    }

    uint32_t    texCount() { return (uint32_t)g_texNames.size(); }
    const char* texName(uint32_t id) {
        return (id < g_texNames.size()) ? g_texNames[id].c_str() : "";
    }

    int32_t slotAt(int32_t x, int32_t y) {
        auto it = g_cellIndex.find(packKey(x, y));
        return (it == g_cellIndex.end()) ? -1 : (int32_t)it->second;
    }

    uint32_t defaultSlot() { return g_defaultSlot; }

    uint64_t residentBytes() { return (uint64_t)g_cells.size() * sizeof(LandCell); }

    bool buildGpuPack() {
        if (!isLoaded() || g_cells.empty()) { return false; }
        if (!g_packHeights.empty()) { return true; }

        g_packHeights.assign((size_t)g_cells.size() * kHeightStrideUints, 0u);
        g_packColors.assign((size_t)g_cells.size() * kColorStrideUints, 0u);
        g_packTex.assign((size_t)g_cells.size() * kTexStrideUints, 0u);
        for (size_t s = 0; s < g_cells.size(); ++s) {
            const LandCell& c = g_cells[s];
            uint32_t* hp = &g_packHeights[s * kHeightStrideUints];
            uint32_t* cp = &g_packColors [s * kColorStrideUints];
            uint32_t* tp = &g_packTex    [s * kTexStrideUints];
            for (int v = 0; v < kCellVerts * kCellVerts; ++v) {
                const uint32_t h = (uint32_t)(uint16_t)c.height[v];
                hp[v >> 1] |= (v & 1) ? (h << 16) : h;                   // two int16 per uint
                cp[v] = (uint32_t)c.color[v * 3 + 0]                      // 0x00BBGGRR
                      | ((uint32_t)c.color[v * 3 + 1] << 8)
                      | ((uint32_t)c.color[v * 3 + 2] << 16);
            }
            for (int t = 0; t < kCellTexRes * kCellTexRes; ++t) {
                tp[t] = (uint32_t)c.tex[t];       // one id per uint — see kTexStrideUints
            }
        }
        return true;
    }

    void releaseGpuPack() {
        std::vector<uint32_t>().swap(g_packHeights);
        std::vector<uint32_t>().swap(g_packColors);
        std::vector<uint32_t>().swap(g_packTex);
    }

    const uint32_t* packedHeights() { return g_packHeights.empty() ? nullptr : g_packHeights.data(); }
    const uint32_t* packedColors()  { return g_packColors.empty()  ? nullptr : g_packColors.data();  }
    const uint32_t* packedTex()     { return g_packTex.empty()     ? nullptr : g_packTex.data();     }

    void buildTextureIndex() {
        if (g_texIndexBuilt) { return; }
        g_texIndexBuilt = true;
        const auto t0 = std::chrono::steady_clock::now();

        indexTextureDir("Data Files\\Textures", "", 0);

        // Every BSA beside the plugins. Later archives win on a key collision (emplace keeps the
        // first), which matches the loose-first-then-BSA precedence readLandTextureFile applies.
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA("Data Files\\*.bsa", &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) { continue; }
                indexBsa(std::string("Data Files\\") + fd.cFileName);
            } while (FindNextFileA(h, &fd));
            FindClose(h);
        }

        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        LOG::logline(">> [terrain] texture index: %zu loose under Data Files\\Textures, %zu in %zu BSA(s), %.0f ms",
                     g_looseTex.size(), g_bsaTex.size(), g_bsaHandles.size(), ms);
        LOG::flush();
    }

    uint32_t looseTextureCount() { return (uint32_t)g_looseTex.size(); }
    uint32_t bsaTextureCount()   { return (uint32_t)g_bsaTex.size(); }

    const uint8_t* readLandTextureFile(uint32_t texId, uint32_t* sizeOut) {
        if (sizeOut) { *sizeOut = 0; }
        buildTextureIndex();
        if (texId >= g_texNames.size()) { return nullptr; }

        // LTEX records the name the CS saw — almost always `.tga`, while what ships is `.dds`.
        const std::string name = lower(g_texNames[texId]);
        const size_t dot = name.find_last_of('.');
        const std::string stem = (dot == std::string::npos) ? name : name.substr(0, dot);
        const std::string dds  = stem + ".dds";

        // Same precedence as the DL generator: loose .dds, loose as-recorded, BSA .dds, BSA as-recorded.
        const std::string tries[2] = { dds, name };
        for (const std::string& key : tries) {
            auto it = g_looseTex.find(key);
            if (it != g_looseTex.end() && readLoose(it->second, g_texScratch)) {
                if (sizeOut) { *sizeOut = (uint32_t)g_texScratch.size(); }
                return g_texScratch.data();
            }
        }
        for (const std::string& key : tries) {
            auto it = g_bsaTex.find("textures\\" + key);
            if (it != g_bsaTex.end() && readBsa(it->second, g_texScratch)) {
                if (sizeOut) { *sizeOut = (uint32_t)g_texScratch.size(); }
                return g_texScratch.data();
            }
        }
        return nullptr;
    }

}
