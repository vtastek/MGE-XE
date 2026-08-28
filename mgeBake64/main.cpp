// mgeBake64 -- the distant-land baker MGE XE owns.
//
// First pass to land: --grass, which replaces millions of individually authored grass placements
// with a per-cell density field (see grassformat.h and tasks/dl-gen-ownership.md). The statics,
// mesh-library and LOD-texture passes follow; they still live in MGEgui today.

#include "grassbake.h"

#include <windows.h>
#include <psapi.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) { return {}; }
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// MGE.ini is a plain sectioned list; [DLWizard Plugins] holds the wizard's tick list in load order
// and [DLWizard Static Overrides] the .ovr paths. Reading it directly means the baker bakes exactly
// what the wizard would have baked.
bool readIni(const std::string& path, std::vector<std::string>& plugins,
             std::vector<std::string>& ovr) {
    FILE* fp = nullptr;
    if (fopen_s(&fp, path.c_str(), "rb") != 0 || !fp) { return false; }
    char line[2048];
    std::string section;
    while (std::fgets(line, sizeof(line), fp)) {
        std::string s = trim(line);
        if (s.empty()) { continue; }
        if (s.front() == '[' && s.back() == ']') { section = s; continue; }
        if (section == "[DLWizard Plugins]") { plugins.push_back(s); }
        else if (section == "[DLWizard Static Overrides]") { ovr.push_back(s); }
    }
    std::fclose(fp);
    return true;
}

void usage() {
    std::printf(
        "mgeBake64 -- MGE XE distant land baker\n"
        "\n"
        "  mgeBake64 --grass --mw <install dir> [options]\n"
        "\n"
        "    --mw <dir>     Morrowind install directory (the one holding 'Data Files')\n"
        "    --ini <path>   MGE.ini to read the plugin tick list from\n"
        "                   (default <mw>\\MGE3\\MGE.ini)\n"
        "    --out <path>   output file (default <mw>\\Data Files\\distantland\\statics\\grass.bin)\n"
        "    --ovr <path>   extra statics classifier override file; repeatable\n"
        "    --no-ini-ovr   ignore the override files listed in MGE.ini\n"
        "    -v             per-plugin blade counts\n");
}

}  // namespace

int main(int argc, char** argv) {
    bool doGrass = false, noIniOvr = false;
    std::string mw, ini, out;
    std::vector<std::string> extraOvr;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](std::string& dst) {
            if (i + 1 < argc) { dst = argv[++i]; }
            else { std::printf("missing value for %s\n", a.c_str()); }
        };
        if (a == "--grass") { doGrass = true; }
        else if (a == "--mw") { val(mw); }
        else if (a == "--ini") { val(ini); }
        else if (a == "--out") { val(out); }
        else if (a == "--ovr") { std::string p; val(p); if (!p.empty()) { extraOvr.push_back(p); } }
        else if (a == "--no-ini-ovr") { noIniOvr = true; }
        else if (a == "-v" || a == "--verbose") { verbose = true; }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::printf("unknown argument: %s\n\n", a.c_str()); usage(); return 2; }
    }

    if (!doGrass || mw.empty()) { usage(); return 2; }
    while (!mw.empty() && (mw.back() == '\\' || mw.back() == '/')) { mw.pop_back(); }
    if (ini.empty()) { ini = mw + "\\MGE3\\MGE.ini"; }
    if (out.empty()) { out = mw + "\\Data Files\\distantland\\statics\\grass.bin"; }

    bake::GrassSettings s;
    s.dataFiles = mw + "\\Data Files";
    s.outPath   = out;
    s.verbose   = verbose;

    std::vector<std::string> iniOvr;
    if (!readIni(ini, s.plugins, iniOvr)) {
        std::printf("[bake] ERROR: cannot read %s\n", ini.c_str());
        return 1;
    }
    if (!noIniOvr) { s.ovrFiles = iniOvr; }
    for (const std::string& p : extraOvr) { s.ovrFiles.push_back(p); }

    std::printf("[bake] install  : %s\n", mw.c_str());
    std::printf("[bake] ini      : %s (%zu plugins)\n", ini.c_str(), s.plugins.size());
    std::printf("[bake] out      : %s\n", out.c_str());

    LARGE_INTEGER f{}, t0{}, t1{};
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);

    bake::GrassStats st;
    bool ok = bake::bakeGrass(s, st);

    QueryPerformanceCounter(&t1);
    double secs = (double)(t1.QuadPart - t0.QuadPart) / (double)f.QuadPart;

    // Working set is NOT the number to watch here: the plugins are memory-mapped, so every page the
    // walk touches (1.2 GB of them) lands in the working set as reclaimable file cache. Private
    // commit is what actually has to stay flat as content grows.
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof(pmc);
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));

    std::printf("\n[bake][grass] %s in %.1f s\n", ok ? "done" : "FAILED", secs);
    std::printf("[bake][grass] refs scanned    %llu\n", (unsigned long long)st.refsScanned);
    std::printf("[bake][grass] blades          %llu\n", (unsigned long long)st.blades);
    std::printf("[bake][grass] cells / planes  %u / %u   palettes %u   meshes %u\n",
                st.cells, st.planes, st.palettes, st.grassModels);
    std::printf("[bake][grass] deletes         %u applied, %u unresolved   moved refs %u\n",
                st.deletesApplied, st.deletesUnresolved, st.movedRefs);
    std::printf("[bake][grass] out-of-cell     %u   clamped tiles %u\n",
                st.outOfCellRefs, st.clampedTiles);
    std::printf("[bake][grass] file            %.2f MB\n", st.bytesWritten / 1e6);
    // The number the whole rewrite exists for: this must stay flat as content grows.
    std::printf("[bake][grass] PEAK PRIVATE     %.1f MB   (peak working set %.1f MB, mostly mapped"
                " plugin pages)\n",
                pmc.PeakPagefileUsage / 1e6, pmc.PeakWorkingSetSize / 1e6);

    if (verbose) {
        std::printf("\n[bake][grass] per plugin:\n");
        for (const auto& kv : st.perPlugin) {
            if (kv.second) {
                std::printf("   %-36s %10llu\n", kv.first.c_str(), (unsigned long long)kv.second);
            }
        }
    }
    return ok ? 0 : 1;
}
