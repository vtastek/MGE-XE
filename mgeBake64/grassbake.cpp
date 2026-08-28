// The grass density pass.
//
// Two streaming walks over the selected plugins:
//   1. collect object definitions (NAME -> MODL) so a reference's mesh can be resolved;
//   2. walk exterior cells and, for every grass reference, do nothing but ++count[tile].
//
// Nothing is kept per reference. Peak memory is one accumulator per (cell, plugin) -- about 2 KB
// each, ~8000 of them on the measured install, so ~16 MB regardless of how many blades the content
// contains. That is the whole reason this exists: the C# path needed ~190 bytes per placement and
// died at 12M of them in a 2 GB process (tasks/dl-gen-ownership.md section 1).

#include "grassbake.h"
#include "espreader.h"
#include "grassformat.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <unordered_map>

namespace bake {

namespace {

constexpr uint32_t TD = grassfmt::kTileDim;
constexpr uint32_t TILES = TD * TD;

// --- statics classifier overrides -------------------------------------------------------------
// Same keywords MGEgui's StaticOverride parses, keyed by model path. This is also the modder
// escape hatch: a plant that must sit exactly where it was placed is classified as a normal static
// here, leaves the density field, and keeps its authored transform through the statics path.
enum class OvrType { Auto, Grass, OtherExplicit };

struct Override {
    bool     ignore  = false;
    OvrType  type    = OvrType::Auto;
    float    density = -1.0f;       // grass_NN, -1 = unset
};

std::string lower(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') { c = (char)(c - 'A' + 'a'); }
    }
    return s;
}

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) { return {}; }
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

void parseOverrides(const std::vector<std::string>& files,
                    std::unordered_map<std::string, Override>& out) {
    for (const std::string& f : files) {
        FILE* fp = nullptr;
        if (fopen_s(&fp, f.c_str(), "rb") != 0 || !fp) {
            std::printf("[bake][grass] WARNING: cannot open override list %s\n", f.c_str());
            continue;
        }
        char line[1024];
        bool inDefaultSection = true;
        while (std::fgets(line, sizeof(line), fp)) {
            std::string s = lower(trim(line));
            // strip an unescaped ':' comment, matching MGEgui's ParseOverrideFiles
            for (size_t i = 0; i < s.size(); ++i) {
                if (s[i] == ':' && (i == 0 || s[i - 1] != '\\')) { s = s.substr(0, i); break; }
            }
            s = trim(s);
            if (s.empty()) { continue; }
            if (s.front() == '[' && s.back() == ']') {
                // only the default section carries per-model type overrides
                inDefaultSection = false;
                continue;
            }
            if (!inDefaultSection) { continue; }
            size_t eq = s.rfind('=');
            if (eq == std::string::npos) { continue; }
            std::string key = trim(s.substr(0, eq));
            std::string val = trim(s.substr(eq + 1));
            if (key.empty()) { continue; }

            Override o;
            size_t pos = 0;
            while (pos < val.size()) {
                size_t sp = val.find(' ', pos);
                std::string kw = val.substr(pos, sp == std::string::npos ? std::string::npos : sp - pos);
                pos = (sp == std::string::npos) ? val.size() : sp + 1;
                if (kw.empty()) { continue; }
                if (kw == "ignore") {
                    o.ignore = true;
                } else if (kw.rfind("grass", 0) == 0) {
                    o.type = OvrType::Grass;
                    if (kw.size() > 6) {
                        float pct = (float)atof(kw.c_str() + 6);
                        if (pct >= 0.0f) { o.density = (pct > 100.0f) ? 1.0f : pct / 100.0f; }
                    }
                } else if (kw == "near" || kw == "far" || kw == "very_far" || kw == "tree" ||
                           kw == "building") {
                    o.type = OvrType::OtherExplicit;   // the precision opt-out
                } else if (kw == "auto") {
                    o.type = OvrType::Auto;
                }
            }
            out[key] = o;
        }
        std::fclose(fp);
    }
}

// --- accumulators -----------------------------------------------------------------------------

struct Accum {
    int16_t  cx = 0, cy = 0;
    uint32_t counts[TILES] = {};                       // per-tile blade count, pre-clamp
    std::unordered_map<uint16_t, uint32_t> meshHist;    // meshIdx -> blades, for the palette weights
    uint64_t blades = 0;
};

inline uint64_t accumKey(uint32_t plugin, int16_t cx, int16_t cy) {
    return ((uint64_t)plugin << 32) | ((uint64_t)(uint16_t)cx << 16) | (uint64_t)(uint16_t)cy;
}

}  // namespace

bool bakeGrass(const GrassSettings& s, GrassStats& st) {
    std::unordered_map<std::string, Override> overrides;
    parseOverrides(s.ovrFiles, overrides);
    std::printf("[bake][grass] %zu override entries from %zu file(s)\n",
                overrides.size(), s.ovrFiles.size());

    // ---- pass 1: object definitions -----------------------------------------------------------
    // NAME -> MODL across every plugin in load order; a later plugin redefining a name wins, which
    // is what MGEgui's StaticsList[name] = ... does.
    std::unordered_map<std::string, std::string> defs;
    uint32_t defsFromTag[3] = { 0, 0, 0 };   // STAT, ACTI, MISC -- reported so an odd pack shows up
    for (const std::string& p : s.plugins) {
        EspFile f;
        std::string path = s.dataFiles + "\\" + p;
        if (!f.open(path.c_str())) {
            std::printf("[bake][grass] WARNING: cannot open %s\n", path.c_str());
            continue;
        }
        RecordIter rec(f);
        while (rec.next()) {
            int which = -1;
            if (tagIs(rec.tag, "STAT")) { which = 0; }
            else if (tagIs(rec.tag, "ACTI")) { which = 1; }
            else if (tagIs(rec.tag, "MISC")) { which = 2; }
            if (which < 0) { continue; }
            std::string nm, md;
            SubIter sub(rec.body, rec.size);
            while (sub.next()) {
                if (tagIs(sub.tag, "NAME")) { nm = subString(sub.body, sub.size); }
                else if (tagIs(sub.tag, "MODL")) { md = subString(sub.body, sub.size); }
            }
            if (!nm.empty() && !md.empty()) { defs[nm] = md; ++defsFromTag[which]; }
        }
    }
    std::printf("[bake][grass] definitions: %zu (STAT %u, ACTI %u, MISC %u)\n",
                defs.size(), defsFromTag[0], defsFromTag[1], defsFromTag[2]);

    // ---- classify: which object ids are grass, and which mesh each uses -----------------------
    std::unordered_map<std::string, uint16_t> objToMesh;   // object id -> mesh table index
    std::vector<std::string> meshPaths;
    std::unordered_map<std::string, uint16_t> meshIndex;
    uint32_t optedOut = 0, ignored = 0;
    for (const auto& kv : defs) {
        const std::string& model = kv.second;
        auto it = overrides.find(model);
        bool isGrass;
        if (it != overrides.end()) {
            if (it->second.ignore) { ++ignored; continue; }
            if (it->second.type == OvrType::Grass) {
                isGrass = true;
            } else if (it->second.type == OvrType::OtherExplicit) {
                isGrass = false;                       // precision opt-out
                if (model.rfind("grass\\", 0) == 0) { ++optedOut; }
            } else {
                isGrass = (model.rfind("grass\\", 0) == 0);
            }
        } else {
            isGrass = (model.rfind("grass\\", 0) == 0);
        }
        if (!isGrass) { continue; }
        auto mi = meshIndex.find(model);
        uint16_t idx;
        if (mi == meshIndex.end()) {
            idx = (uint16_t)meshPaths.size();
            meshIndex[model] = idx;
            meshPaths.push_back(model);
        } else {
            idx = mi->second;
        }
        objToMesh[kv.first] = idx;
    }
    st.grassModels = (uint32_t)meshPaths.size();
    std::printf("[bake][grass] grass models %u, grass object ids %zu"
                " (%u opted out to precision placement, %u ignored)\n",
                st.grassModels, objToMesh.size(), optedOut, ignored);
    if (objToMesh.empty()) {
        std::printf("[bake][grass] nothing classified as grass -- writing no file\n");
        return false;
    }

    // ---- pass 2: accumulate density ------------------------------------------------------------
    std::unordered_map<uint64_t, Accum> accums;
    for (uint32_t pi = 0; pi < s.plugins.size(); ++pi) {
        EspFile f;
        std::string path = s.dataFiles + "\\" + s.plugins[pi];
        if (!f.open(path.c_str())) { continue; }
        uint64_t bladesHere = 0;

        RecordIter rec(f);
        while (rec.next()) {
            if (!tagIs(rec.tag, "CELL")) { continue; }

            bool     haveCell = false, interior = false;
            int16_t  cx = 0, cy = 0;

            // reference state, committed when the next FRMR (or the record) ends -- the same
            // shape MGEgui's parser uses, because a reference's subrecords follow its FRMR.
            bool     inRef = false, refDeleted = false, refHasPos = false, refMoved = false;
            float    rx = 0.0f, ry = 0.0f;
            uint16_t refMesh = 0xFFFF;

            auto commit = [&]() {
                if (!inRef) { return; }
                ++st.refsScanned;
                if (refMoved) { ++st.movedRefs; }
                if (refMesh == 0xFFFF) { return; }
                if (!refHasPos) {
                    if (refDeleted) { ++st.deletesUnresolved; }
                    return;
                }
                float lx = rx - (float)cx * grassfmt::kCellSize;
                float ly = ry - (float)cy * grassfmt::kCellSize;
                if (lx < 0.0f || lx >= grassfmt::kCellSize ||
                    ly < 0.0f || ly >= grassfmt::kCellSize) {
                    ++st.outOfCellRefs;
                    return;
                }
                uint32_t tx = (uint32_t)(lx * (TD / grassfmt::kCellSize));
                uint32_t ty = (uint32_t)(ly * (TD / grassfmt::kCellSize));
                uint32_t ti = ty * TD + tx;

                if (refDeleted) {
                    // A later plugin deleting an earlier plugin's blade: decrement whichever
                    // accumulator for this cell actually holds one of that mesh in that tile.
                    for (auto& kv : accums) {
                        if (kv.second.cx != cx || kv.second.cy != cy) { continue; }
                        auto h = kv.second.meshHist.find(refMesh);
                        if (h == kv.second.meshHist.end() || kv.second.counts[ti] == 0) { continue; }
                        --kv.second.counts[ti];
                        --kv.second.blades;
                        if (--h->second == 0) { kv.second.meshHist.erase(h); }
                        ++st.deletesApplied;
                        return;
                    }
                    ++st.deletesUnresolved;
                    return;
                }

                Accum& a = accums[accumKey(pi, cx, cy)];
                a.cx = cx; a.cy = cy;
                ++a.counts[ti];
                ++a.meshHist[refMesh];
                ++a.blades;
                ++bladesHere;
            };

            SubIter sub(rec.body, rec.size);
            while (sub.next()) {
                if (tagIs(sub.tag, "DATA") && sub.size == 12 && !haveCell) {
                    uint32_t flags = rdU32(sub.body);
                    cx = (int16_t)rdI32(sub.body + 4);
                    cy = (int16_t)rdI32(sub.body + 8);
                    interior = (flags & 1) != 0;
                    haveCell = true;
                } else if (tagIs(sub.tag, "FRMR")) {
                    commit();
                    inRef = true; refDeleted = false; refHasPos = false; refMoved = false;
                    refMesh = 0xFFFF;
                } else if (!inRef || interior || !haveCell) {
                    continue;
                } else if (tagIs(sub.tag, "NAME")) {
                    auto it = objToMesh.find(subString(sub.body, sub.size));
                    refMesh = (it == objToMesh.end()) ? (uint16_t)0xFFFF : it->second;
                } else if (tagIs(sub.tag, "DATA") && sub.size == 24) {
                    rx = rdF32(sub.body);
                    ry = rdF32(sub.body + 4);
                    refHasPos = true;
                } else if (tagIs(sub.tag, "DELE")) {
                    refDeleted = true;
                } else if (tagIs(sub.tag, "MVRF")) {
                    refMoved = true;
                }
            }
            if (!interior && haveCell) { commit(); }
        }
        st.perPlugin.emplace_back(s.plugins[pi], bladesHere);
        if (s.verbose && bladesHere) {
            std::printf("[bake][grass]   %-34s %10llu blades\n", s.plugins[pi].c_str(),
                        (unsigned long long)bladesHere);
        }
    }

    // ---- intern palettes -----------------------------------------------------------------------
    std::map<std::vector<uint16_t>, uint16_t> paletteIds;
    std::vector<std::vector<uint16_t>>        palettes;
    std::vector<std::unordered_map<uint16_t, uint64_t>> paletteWeights;

    struct Plane { int16_t cx, cy; uint16_t pal; std::vector<uint32_t> counts; uint64_t blades; };
    std::map<uint64_t, size_t> planeIndex;   // (cy,cx,pal) -> index in planes
    std::vector<Plane> planes;

    for (auto& kv : accums) {
        Accum& a = kv.second;
        if (a.blades == 0) { continue; }
        std::vector<uint16_t> set;
        set.reserve(a.meshHist.size());
        for (const auto& m : a.meshHist) { set.push_back(m.first); }
        std::sort(set.begin(), set.end());

        auto pit = paletteIds.find(set);
        uint16_t pal;
        if (pit == paletteIds.end()) {
            pal = (uint16_t)palettes.size();
            paletteIds[set] = pal;
            palettes.push_back(set);
            paletteWeights.emplace_back();
        } else {
            pal = pit->second;
        }
        for (const auto& m : a.meshHist) { paletteWeights[pal][m.first] += m.second; }

        uint64_t key = ((uint64_t)(uint16_t)a.cy << 32) | ((uint64_t)(uint16_t)a.cx << 16) | pal;
        auto pi2 = planeIndex.find(key);
        if (pi2 == planeIndex.end()) {
            planeIndex[key] = planes.size();
            planes.push_back(Plane{ a.cx, a.cy, pal,
                                    std::vector<uint32_t>(a.counts, a.counts + TILES), a.blades });
        } else {
            Plane& p = planes[pi2->second];
            for (uint32_t i = 0; i < TILES; ++i) { p.counts[i] += a.counts[i]; }
            p.blades += a.blades;
        }
    }

    // Drop meshes that are DEFINED as grass somewhere but never actually placed -- vanilla and TR
    // ship grass STATs the ticked plugins never use. Keeping them would hand the host names it
    // cannot resolve against static_meshes (nothing places them, so MGEgui never bakes them), and
    // "unresolved mesh" would stop meaning "something is wrong".
    {
        std::vector<uint32_t> remap(meshPaths.size(), 0xFFFFFFFFu);
        std::vector<std::string> kept;
        for (const std::vector<uint16_t>& set : palettes) {
            for (uint16_t m : set) {
                if (remap[m] == 0xFFFFFFFFu) {
                    remap[m] = (uint32_t)kept.size();
                    kept.push_back(meshPaths[m]);
                }
            }
        }
        if (kept.size() != meshPaths.size()) {
            std::printf("[bake][grass] pruned %zu defined-but-never-placed grass meshes\n",
                        meshPaths.size() - kept.size());
        }
        for (size_t i = 0; i < palettes.size(); ++i) {
            std::unordered_map<uint16_t, uint64_t> rw;
            for (uint16_t m : palettes[i]) { rw[(uint16_t)remap[m]] = paletteWeights[i][m]; }
            for (uint16_t& m : palettes[i]) { m = (uint16_t)remap[m]; }
            std::sort(palettes[i].begin(), palettes[i].end());
            paletteWeights[i].swap(rw);
        }
        meshPaths.swap(kept);
        st.grassModels = (uint32_t)meshPaths.size();
    }

    // planes were interned through an ordered map keyed by (cy, cx, pal), so this sort is the same
    // order the host will binary-search
    std::sort(planes.begin(), planes.end(), [](const Plane& a, const Plane& b) {
        if (a.cy != b.cy) { return a.cy < b.cy; }
        if (a.cx != b.cx) { return a.cx < b.cx; }
        return a.pal < b.pal;
    });

    std::vector<std::pair<int16_t, int16_t>> distinctCells;
    for (const Plane& p : planes) { distinctCells.emplace_back(p.cx, p.cy); }
    std::sort(distinctCells.begin(), distinctCells.end());
    distinctCells.erase(std::unique(distinctCells.begin(), distinctCells.end()), distinctCells.end());

    for (const Plane& p : planes) { st.blades += p.blades; }
    st.cells    = (uint32_t)distinctCells.size();
    st.planes   = (uint32_t)planes.size();
    st.palettes = (uint32_t)palettes.size();

    // ---- write ---------------------------------------------------------------------------------
    // Temp + rename: a failed or killed bake must leave the previous grass.bin intact. The C# path
    // truncates static_meshes before it validates anything, which is why a single OOM left the
    // install with no distant statics at all.
    std::string tmp = s.outPath + ".tmp";
    FILE* fp = nullptr;
    if (fopen_s(&fp, tmp.c_str(), "wb") != 0 || !fp) {
        std::printf("[bake][grass] ERROR: cannot write %s\n", tmp.c_str());
        return false;
    }

    std::vector<uint8_t> meshTable;
    for (const std::string& m : meshPaths) {
        uint16_t n = (uint16_t)m.size();
        meshTable.insert(meshTable.end(), (uint8_t*)&n, (uint8_t*)&n + 2);
        meshTable.insert(meshTable.end(), m.begin(), m.end());
    }

    std::vector<uint8_t> palTable;
    for (size_t i = 0; i < palettes.size(); ++i) {
        const std::vector<uint16_t>& set = palettes[i];
        uint16_t n = (uint16_t)set.size();
        palTable.insert(palTable.end(), (uint8_t*)&n, (uint8_t*)&n + 2);
        for (uint16_t m : set) { palTable.insert(palTable.end(), (uint8_t*)&m, (uint8_t*)&m + 2); }
        uint64_t total = 0;
        for (uint16_t m : set) { total += paletteWeights[i][m]; }
        for (uint16_t m : set) {
            // at least 1: a mesh in the palette must stay reachable, however rare it is
            uint64_t w = total ? (paletteWeights[i][m] * 255 + total / 2) / total : 255;
            palTable.push_back((uint8_t)(w < 1 ? 1 : (w > 255 ? 255 : w)));
        }
    }

    grassfmt::Header h{};
    std::memcpy(h.magic, grassfmt::kMagic, 4);
    h.version       = grassfmt::kVersion;
    h.tileDim       = TD;
    h.cellSize      = grassfmt::kCellSize;
    h.meshCount     = (uint32_t)meshPaths.size();
    h.paletteCount  = (uint32_t)palettes.size();
    h.planeCount    = (uint32_t)planes.size();
    h.meshTableOff  = (uint32_t)sizeof(grassfmt::Header);
    h.paletteOff    = h.meshTableOff + (uint32_t)meshTable.size();
    h.planeOff      = h.paletteOff + (uint32_t)palTable.size();
    h.sinkZ         = grassfmt::kDefaultSinkZ;
    h.scaleMin      = grassfmt::kDefaultScaleMin;
    h.scaleMax      = grassfmt::kDefaultScaleMax;
    h.tiltJitterDeg = grassfmt::kDefaultTiltJitterDeg;
    h.totalBlades   = st.blades;

    std::fwrite(&h, sizeof(h), 1, fp);
    std::fwrite(meshTable.data(), 1, meshTable.size(), fp);
    std::fwrite(palTable.data(), 1, palTable.size(), fp);

    std::vector<uint8_t> tile(TILES);
    for (const Plane& p : planes) {
        grassfmt::PlaneHeader ph{};
        ph.cx = p.cx; ph.cy = p.cy; ph.paletteId = p.pal; ph.blades = (uint32_t)p.blades;
        for (uint32_t i = 0; i < TILES; ++i) {
            uint32_t v = p.counts[i];
            if (v > 255) { v = 255; ++st.clampedTiles; }
            tile[i] = (uint8_t)v;
        }
        std::fwrite(&ph, sizeof(ph), 1, fp);
        std::fwrite(tile.data(), 1, tile.size(), fp);
    }
    // the clamp count is only known after the loop, so patch it in place
    h.clampedTiles = st.clampedTiles;
    std::fseek(fp, 0, SEEK_SET);
    std::fwrite(&h, sizeof(h), 1, fp);
    std::fseek(fp, 0, SEEK_END);
    st.bytesWritten = (uint64_t)std::ftell(fp);
    std::fclose(fp);

    std::remove(s.outPath.c_str());
    if (std::rename(tmp.c_str(), s.outPath.c_str()) != 0) {
        std::printf("[bake][grass] ERROR: cannot rename %s -> %s\n", tmp.c_str(), s.outPath.c_str());
        return false;
    }
    return true;
}

}  // namespace bake
