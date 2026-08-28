#pragma once

#include <string>
#include <vector>

namespace bake {

struct GrassSettings {
    std::string              dataFiles;    // "<install>\Data Files"
    std::vector<std::string> plugins;      // in load order
    std::vector<std::string> ovrFiles;     // statics classifier overrides
    std::string              outPath;      // written via .tmp + rename
    bool                     verbose = false;
};

struct GrassStats {
    uint64_t refsScanned      = 0;   // every exterior FRMR examined
    uint64_t blades           = 0;   // those that classified as grass
    uint32_t grassModels      = 0;
    uint32_t cells            = 0;
    uint32_t planes           = 0;
    uint32_t palettes         = 0;
    uint32_t clampedTiles     = 0;
    uint32_t deletesApplied   = 0;
    uint32_t deletesUnresolved = 0;
    uint32_t movedRefs        = 0;
    uint32_t outOfCellRefs    = 0;
    uint64_t bytesWritten     = 0;
    std::vector<std::pair<std::string, uint64_t>> perPlugin;  // for cross-checking the parse
};

bool bakeGrass(const GrassSettings& s, GrassStats& st);

}  // namespace bake
