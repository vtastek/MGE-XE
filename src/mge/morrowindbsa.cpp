
#include "morrowindbsa.h"
#include "proxydx/d3d8header.h"
#include "support/log.h"
#include "support/timing.h"
#include "configuration.h"

#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cctype>



namespace BSA {

using std::unordered_map;

struct CacheEntry {
    HANDLE file;
    DWORD position;
    DWORD size;
    std::string filename;  // Added for texture suffix support
};

struct BSAHash3 {
    union {
        struct {
            DWORD value1, value2;
        };
        __int64 LValue;
    };
};

struct EntryData {
    std::unique_ptr<char[]> data;
    unsigned int size;

    bool valid() const { return bool(data); }
};

// Note: loadedTextures would ideally store weakRefs, but COM doesn't support those.
static unordered_map<__int64, CacheEntry> cacheMap;
static unordered_map<__int64, IDirect3DTexture9*> loadedTextures;

// Texture suffix support data structures
static unordered_map<std::string, TextureSuffixVariants> textureSuffixDatabase;
static unordered_map<TextureRuntimeHash, std::string, TextureRuntimeHasher> textureHashToName;
static bool textureSuffixDatabaseBuilt = false;

// Runtime texture hash cache key (width, height, format, pool)
struct TextureCacheKey {
    UINT width;
    UINT height;
    D3DFORMAT format;
    D3DPOOL pool;
    
    bool operator==(const TextureCacheKey& other) const {
        return width == other.width && height == other.height && 
               format == other.format && pool == other.pool;
    }
};

// Hash function for TextureCacheKey
struct TextureCacheKeyHasher {
    size_t operator()(const TextureCacheKey& key) const {
        return ((size_t)key.width << 16) | ((size_t)key.height) | 
               ((size_t)key.format << 24) | ((size_t)key.pool << 28);
    }
};

// Runtime texture hash cache to avoid repeated calculations
static unordered_map<TextureCacheKey, TextureRuntimeHash, TextureCacheKeyHasher> runtimeTextureHashCache;



// CRC32 implementation for texture hashing
static unsigned int crc32_table[256];
static bool crc32_table_initialized = false;

// Helper function to get readable D3D format name
static const char* getD3DFormatName(D3DFORMAT format) {
    switch (format) {
        case D3DFMT_DXT1: return "DXT1";
        case D3DFMT_DXT3: return "DXT3";
        case D3DFMT_DXT5: return "DXT5";
        case D3DFMT_A8R8G8B8: return "A8R8G8B8(uncompressed)";
        case D3DFMT_X8R8G8B8: return "X8R8G8B8(uncompressed)";
        case D3DFMT_R5G6B5: return "R5G6B5(uncompressed)";
        case D3DFMT_A1R5G5B5: return "A1R5G5B5(uncompressed)";
        case D3DFMT_A8: return "A8(uncompressed)";
        default: {
            static char buffer[32];
            snprintf(buffer, sizeof(buffer), "Unknown(%u)", (unsigned int)format);
            return buffer;
        }
    }
}

static void init_crc32_table() {
    if (crc32_table_initialized) return;
    
    unsigned int c;
    for (int i = 0; i < 256; i++) {
        c = (unsigned int)i;
        for (int j = 0; j < 8; j++) {
            if (c & 1) {
                c = 0xedb88320L ^ (c >> 1);
            } else {
                c = c >> 1;
            }
        }
        crc32_table[i] = c;
    }
    crc32_table_initialized = true;
}

static unsigned int crc32(const unsigned char* buf, size_t len) {
    if (!buf || len == 0) {
        return 0;
    }
    
    // Sanity check on length to prevent excessive processing
    if (len > 0x10000000) { // 256MB limit
        LOG::logline("!! CRC32: Excessive length %u, truncating", (unsigned int)len);
        len = 0x10000000;
    }
    
    init_crc32_table();
    unsigned int c = 0xffffffffL;
    
    __try {
        for (size_t i = 0; i < len; i++) {
            c = crc32_table[(c ^ buf[i]) & 0xff] ^ (c >> 8);
        }
    }
    __except(EXCEPTION_EXECUTE_HANDLER) {
        LOG::logline("!! CRC32: Exception during calculation at offset %u", (unsigned int)c);
        return c ^ 0xffffffffL; // Return partial result
    }
    
    return c ^ 0xffffffffL;
}

// hashString - TES3 BSA Hash function
static BSAHash3 hashString(const char* str) {
    BSAHash3 result;

    unsigned int len = (unsigned int)strlen(str);

    // Use GhostWheel's code to hash the string
    unsigned int l = len >> 1;
    unsigned int sum, off, temp, i, n;

    for (sum = off = i = 0; i < l; i++) {
        sum ^= ((unsigned int)(str[i])) << (off & 0x1F);
        off += 8;
    }
    result.value1 = sum;

    for (sum = off = 0; i < len; i++) {
        temp = ((unsigned int)(str[i])) << (off & 0x1F);
        sum ^= temp;
        n = temp & 0x1F;
        sum = (sum << (32-n)) | (sum >> n);  // binary rotate right
        off += 8;
    }
    result.value2 = sum;
    return result;
}

// open - Read and store a BSA's index
static void open(const char* path) {
    HANDLE bsa = CreateFile(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    if (bsa == INVALID_HANDLE_VALUE) {
        return;
    }

    DWORD hashOffset, numFiles, bytesRead, unused;
    ReadFile(bsa, &hashOffset, 4, &bytesRead, 0);
    if (bytesRead != 4 || hashOffset != 0x100) {
        CloseHandle(bsa);
        return;
    }

    ReadFile(bsa, &hashOffset, 4, &unused, 0);
    ReadFile(bsa, &numFiles, 4, &unused, 0);
    
    for (DWORD i = 0; i < numFiles; i++) {
        CacheEntry entry;
        __int64 hash;

        entry.file = bsa;
        
        // Read file size and offset from file record table
        SetFilePointer(bsa, 12 + i*8, 0, FILE_BEGIN);
        ReadFile(bsa, &entry.size, 4, &unused, 0);
        ReadFile(bsa, &entry.position, 4, &unused, 0);
        entry.position += 12 + hashOffset + numFiles*8;

        // Read filename using the C# approach
        // 1. Read filename offset from filename offset table
        SetFilePointer(bsa, 12 + numFiles*8 + i*4, 0, FILE_BEGIN);
        DWORD filenameOffset;
        ReadFile(bsa, &filenameOffset, 4, &unused, 0);
        
        // 2. Seek to actual filename location
        SetFilePointer(bsa, filenameOffset + 12 + numFiles*12, 0, FILE_BEGIN);
        
        // 3. Read null-terminated filename string byte by byte
        std::string filename;
        char c;
        while (true) {
            ReadFile(bsa, &c, 1, &unused, 0);
            if (c == 0) break;
            filename += c;
        }
        entry.filename = filename;

        // Read hash
        SetFilePointer(bsa, 12 + hashOffset + i*8, 0, FILE_BEGIN);
        ReadFile(bsa, &hash, 8, &unused, 0);
        cacheMap[hash] = entry;
    }
    
    LOG::logline("BSA: Loaded %s with %d files", path, numFiles);
}

// init - Scan and index all BSA files
void init() {
    char path[MAX_PATH];
    WIN32_FIND_DATA data;

    HANDLE h = FindFirstFile("Data Files\\*.bsa", &data);
    if (h == INVALID_HANDLE_VALUE) {
        return;
    }

    do {
        std::snprintf(path, sizeof(path), "Data Files\\%s", data.cFileName);
        open(path);
    } while (FindNextFile(h, &data));

    FindClose(h);
    
    // Build texture suffix database after loading BSA files
    buildTextureSuffixDatabase();
}

// loadFile - Read a single file into memory, identified by hash only
static EntryData BSALoadFile(BSAHash3 hash) {
    auto it = cacheMap.find(hash.LValue);
    if (it == cacheMap.end()) {
        return EntryData();
    }

    const CacheEntry& entry = it->second;
    auto buf = std::make_unique<char[]>(entry.size);
    DWORD bytesRead;

    SetFilePointer(entry.file, entry.position, 0, FILE_BEGIN);
    ReadFile(entry.file, buf.get(), entry.size, &bytesRead, 0);

    if (bytesRead == entry.size) {
        return EntryData { std::move(buf), entry.size };
    } else {
        return EntryData();
    }
}

// loadTextureExact - Attempt to load a texture from a prioritized list of sources.
static IDirect3DTexture9* loadTextureExact(IDirect3DDevice9* dev, const char* filename) {
    char pathbuf[MAX_PATH];
    BSAHash3 hash = hashString(filename);
    IDirect3DTexture9* tex = nullptr;

    // First check if the texture is already loaded
    auto it = loadedTextures.find(hash.LValue);
    if (it != loadedTextures.end()) {
        it->second->AddRef();
        return it->second;
    }

    // Next check the distant land folder
    std::snprintf(pathbuf, sizeof(pathbuf), "Data Files\\distantland\\statics\\%s", filename);
    if (GetFileAttributes(pathbuf) != INVALID_FILE_ATTRIBUTES) {
        HRESULT hr = D3DXCreateTextureFromFileEx(dev, pathbuf, D3DX_FROM_FILE, D3DX_FROM_FILE, D3DX_FROM_FILE, 0, D3DFMT_UNKNOWN,
                     D3DPOOL_DEFAULT, D3DX_FILTER_NONE, D3DX_FILTER_NONE, 0, 0, 0, &tex);

        if (hr == D3D_OK) {
            loadedTextures[hash.LValue] = tex;
            return tex;
        }
    }

    // Then check the normal folder
    std::snprintf(pathbuf, sizeof(pathbuf), "Data Files\\%s", filename);
    if (GetFileAttributes(pathbuf) != INVALID_FILE_ATTRIBUTES) {
        HRESULT hr = D3DXCreateTextureFromFileEx(dev, pathbuf, D3DX_FROM_FILE, D3DX_FROM_FILE, D3DX_FROM_FILE, 0, D3DFMT_UNKNOWN,
                     D3DPOOL_DEFAULT, D3DX_FILTER_NONE, D3DX_FILTER_NONE, 0, 0, 0, &tex);

        if (hr == D3D_OK) {
            loadedTextures[hash.LValue] = tex;
            return tex;
        }
    }

    // Finally check the BSAs
    EntryData ed = BSALoadFile(hash);
    if (ed.valid()) {
        D3DXCreateTextureFromFileInMemoryEx(dev, ed.data.get(), ed.size, D3DX_FROM_FILE, D3DX_FROM_FILE, D3DX_FROM_FILE,
                                            0, D3DFMT_UNKNOWN, D3DPOOL_MANAGED, D3DX_DEFAULT, D3DX_DEFAULT, 0, 0, 0, &tex);

        // Cache even if the texture load failed
        loadedTextures[hash.LValue] = tex;
        return tex;
    }

    // File not found
    return nullptr;
}

// loadTexture -  Attempt to load a texture from a prioritized list of sources, with extension substitution.
IDirect3DTexture9* loadTexture(IDirect3DDevice9* dev, const char* filename) {
    char pathbuf[MAX_PATH];

    // Prefer loading file with DDS extension first
    std::snprintf(pathbuf, sizeof(pathbuf), "textures\\%s", filename);
    strcpy_s(pathbuf + strlen(pathbuf) - 3, 4, "dds");

    IDirect3DTexture9* tex = loadTextureExact(dev, pathbuf);
    if (tex) {
        return tex;
    }

    // Load file with original extension
    std::snprintf(pathbuf, sizeof(pathbuf), "textures\\%s", filename);
    return loadTextureExact(dev, pathbuf);
}

// clearTextureCache - Clear texture cache.
void clearTextureCache() {
    loadedTextures.clear();
}

// cacheStats - Returns number of textures cached, and approximate memory use in MB.
void cacheStats(int* total, int* memuse) {
    __int64 texMemUsage = 0;

    const auto& loadedTextures_const = loadedTextures;
    for (const auto& i : loadedTextures_const) {
        D3DSURFACE_DESC texdesc;
        i.second->GetLevelDesc(0, &texdesc);

        int bpp = 32;
        if (texdesc.Format == D3DFMT_DXT1) {
            bpp = 4;
        }
        if (texdesc.Format == D3DFMT_DXT3 || texdesc.Format == D3DFMT_DXT5) {
            bpp = 8;
        }

        texMemUsage += (texdesc.Width * texdesc.Height * bpp / 8) * 4 / 3;
    }

    *total = loadedTextures.size();
    *memuse = (int)(texMemUsage / 1048576.0);
}

// Helper function to normalize texture path for consistent lookup
static std::string normalizeTexturePath(const char* texPath) {
    std::string normalized = texPath;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](char c) {
        return (c == '\\') ? '/' : std::tolower(c);
    });
    
    // Remove textures/ prefix if present
    if (normalized.find("textures/") == 0) {
        normalized = normalized.substr(9);
    }
    
    return normalized;
}

// Extract base name from texture path (removes suffix like _diffparam, _nh, etc)
static std::string extractBaseName(const std::string& texPath) {
    std::string baseName = texPath;
    
    // Remove file extension
    size_t dotPos = baseName.find_last_of('.');
    if (dotPos != std::string::npos) {
        baseName = baseName.substr(0, dotPos);
    }
    
    // Check for known suffixes and remove them (order matters - check longer suffixes first)
    if (baseName.length() > 12 && baseName.substr(baseName.length() - 12) == "_diffparam_t") {
        return baseName.substr(0, baseName.length() - 12);
    }
    if (baseName.length() > 10 && baseName.substr(baseName.length() - 10) == "_diffparam") {
        return baseName.substr(0, baseName.length() - 10);
    }
    if (baseName.length() > 3 && baseName.substr(baseName.length() - 3) == "_nh") {
        return baseName.substr(0, baseName.length() - 3);
    }
    if (baseName.length() > 6 && baseName.substr(baseName.length() - 6) == "_param") {
        return baseName.substr(0, baseName.length() - 6);
    }
    
    return baseName;
}

// Check if texture name has a known suffix
static const char* getSuffixType(const std::string& texPath) {
    std::string baseName = texPath;
    
    // Remove file extension
    size_t dotPos = baseName.find_last_of('.');
    if (dotPos != std::string::npos) {
        baseName = baseName.substr(0, dotPos);
    }
    
    if (baseName.length() > 12 && baseName.substr(baseName.length() - 12) == "_diffparam_t") {
        return "diffparam_t";
    }
    if (baseName.length() > 10 && baseName.substr(baseName.length() - 10) == "_diffparam") {
        return "diffparam";
    }
    if (baseName.length() > 3 && baseName.substr(baseName.length() - 3) == "_nh") {
        return "normal";
    }
    if (baseName.length() > 6 && baseName.substr(baseName.length() - 6) == "_param") {
        return "param";
    }
    
    return nullptr; // Base texture
}

// Helper function to check if a texture path is in a grass folder
bool isInGrassFolder(const std::string& path) {
    std::string lowerPath = path;
    std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(), ::tolower);
    
    // Look for /grass/ or \grass\ anywhere in the path (folder separator on both sides)
    return (lowerPath.find("/grass/") != std::string::npos || 
            lowerPath.find("\\grass\\") != std::string::npos ||
            lowerPath.find("grass/") == 0 ||  // Starts with grass/
            lowerPath.find("grass\\") == 0 || // Starts with grass\ (backslash)
            lowerPath.find("/grass") == lowerPath.length() - 6 ||  // Ends with /grass
            lowerPath.find("\\grass") == lowerPath.length() - 6);  // Ends with \grass
}

// Helper function to recursively scan directories for suffix files and grass textures
static void scanDirectoryForSuffixes(const std::string& basePath, const std::string& relativePath, 
                                   std::unordered_map<std::string, TextureSuffixVariants>& suffixMap, int& suffixFilesFound) {
    std::string searchPath = basePath;
    if (!relativePath.empty()) {
        searchPath += "\\" + relativePath;
    }
    searchPath += "\\*";
    
    WIN32_FIND_DATA findFileData;
    HANDLE hFind = FindFirstFile(searchPath.c_str(), &findFileData);
    
    if (hFind == INVALID_HANDLE_VALUE) return;
    
    do {
        if (findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            // Skip . and .. directories
            if (strcmp(findFileData.cFileName, ".") == 0 || strcmp(findFileData.cFileName, "..") == 0) {
                continue;
            }
            
            // Recursively scan subdirectories
            std::string newRelativePath = relativePath.empty() ? findFileData.cFileName : 
                                        relativePath + "\\" + findFileData.cFileName;
            scanDirectoryForSuffixes(basePath, newRelativePath, suffixMap, suffixFilesFound);
        } else {
            std::string filename = findFileData.cFileName;
            std::string fullRelativePath = relativePath.empty() ? filename : relativePath + "\\" + filename;
            
            
            // Check for suffix patterns (check longer suffixes first)
            bool isSuffixFile = false;
            std::string suffixType;
            
            if (filename.length() > 15 && filename.substr(filename.length() - 16) == "_diffparam_t.dds") {
                isSuffixFile = true;
                suffixType = "diffparam_t";
            } else if (filename.length() > 13 && filename.substr(filename.length() - 14) == "_diffparam.dds") {
                isSuffixFile = true;
                suffixType = "diffparam";
            } else if (filename.length() > 7 && filename.substr(filename.length() - 7) == "_nh.dds") {
                isSuffixFile = true;
                suffixType = "normal";
            } else if (filename.length() > 10 && filename.substr(filename.length() - 10) == "_param.dds") {
                isSuffixFile = true;
                suffixType = "param";
            }
            
            if (isSuffixFile) {
                std::string normalizedPath = normalizeTexturePath(fullRelativePath.c_str());
                std::string baseName = extractBaseName(normalizedPath);
                
                TextureSuffixVariants& variants = suffixMap[baseName];
                variants.baseName = baseName;
                variants.isGrassTexture = false;
                
                if (suffixType == "diffparam") {
                    variants.diffparam = fullRelativePath;
                    LOG::logline("DEBUG: Storing _diffparam variant for base %s: %s", baseName.c_str(), variants.diffparam.c_str());
                } else if (suffixType == "diffparam_t") {
                    variants.diffparam_t = fullRelativePath;
                } else if (suffixType == "normal") {
                    variants.normal = fullRelativePath;
                } else if (suffixType == "param") {
                    variants.param = fullRelativePath;
                }
                
                suffixFilesFound++;
            }
            
            // Check if this texture is in grass folder (for any DDS file)
            if (isInGrassFolder(relativePath) && filename.length() > 4 && 
                filename.substr(filename.length() - 4) == ".dds") {
                
                std::string normalizedPath = normalizeTexturePath(fullRelativePath.c_str());
                std::string baseName = extractBaseName(normalizedPath);
                
                TextureSuffixVariants& variants = suffixMap[baseName];
                variants.baseName = baseName;
                variants.isGrassTexture = false;
                variants.isGrassTexture = true;
                // Check if this grass texture exists as a loose file
                std::string grassTexturePath = "Data Files\\textures\\" + fullRelativePath;
                std::replace(grassTexturePath.begin(), grassTexturePath.end(), '/', '\\');
                if (GetFileAttributes(grassTexturePath.c_str()) != INVALID_FILE_ATTRIBUTES) {
                    variants.baseTextureSource = "loose";
                    variants.baseTexturePath = grassTexturePath;
                }
                
            }
        }
    } while (FindNextFile(hFind, &findFileData));
    
    FindClose(hFind);
}

// buildTextureSuffixDatabase - New suffix-first approach: find loose suffixes, then their bases
void buildTextureSuffixDatabase() {
    if (textureSuffixDatabaseBuilt) return;
    
    LOG::logline("-- Building texture suffix database (recursive suffix-first approach)");
    
    // Phase 1: Discover all loose suffix files recursively
    std::unordered_map<std::string, TextureSuffixVariants> suffixMap;
    int suffixFilesFound = 0;
    
    // Recursively scan Data Files/textures/ and all subdirectories for suffix files
    scanDirectoryForSuffixes("Data Files\\textures", "", suffixMap, suffixFilesFound);
    
    // Phase 1.5: Discover grass textures from BSA files
    int grassTexturesFoundBSA = 0;
    for (const auto& entry : cacheMap) {
        const CacheEntry& cacheEntry = entry.second;
        const std::string& filename = cacheEntry.filename;
        
        // Check if it's a texture file in a grass folder
        if (filename.find("textures/") != std::string::npos || filename.find("textures\\") != std::string::npos) {
            if (isInGrassFolder(filename) && filename.length() > 4 && 
                filename.substr(filename.length() - 4) == ".dds") {
                
                std::string normalizedPath = normalizeTexturePath(filename.c_str());
                std::string baseName = extractBaseName(normalizedPath);
                
                TextureSuffixVariants& variants = suffixMap[baseName];
                variants.baseName = baseName;
                variants.isGrassTexture = true;
                variants.baseTextureSource = "bsa";
                variants.baseTexturePath = filename;
                
                grassTexturesFoundBSA++;
            }
        }
    }
    
    // Phase 2: For each base name, find the actual base texture (loose overrides BSA)
    int basesFound = 0;
    for (auto& pair : suffixMap) {
        const std::string& baseName = pair.first;
        TextureSuffixVariants& variants = pair.second;
        
        // Extract the directory path from one of the suffix files to know where to look for base texture
        std::string suffixPath;
        if (!variants.diffparam.empty()) {
            suffixPath = variants.diffparam;
        } else if (!variants.normal.empty()) {
            suffixPath = variants.normal;
        } else if (!variants.param.empty()) {
            suffixPath = variants.param;
        }
        
        // Determine the directory where the suffix was found
        std::string suffixDir = "";
        size_t lastSlash = suffixPath.find_last_of("/\\");
        if (lastSlash != std::string::npos) {
            suffixDir = suffixPath.substr(0, lastSlash + 1);  // Include the trailing slash
        }
        
        // Look for base texture in the same directory as the suffix
        std::string looseBasePath = "Data Files\\textures\\" + baseName + ".dds";
        
        // Replace forward slashes with backslashes for Windows file system
        std::replace(looseBasePath.begin(), looseBasePath.end(), '/', '\\');
        
        WIN32_FIND_DATA fileData;
        HANDLE hFile = FindFirstFile(looseBasePath.c_str(), &fileData);
        
        if (hFile != INVALID_HANDLE_VALUE) {
            // Found loose base texture in same directory as suffix
            variants.baseTextureSource = "loose";
            variants.baseTexturePath = looseBasePath;
            basesFound++;
            FindClose(hFile);
        } else {
            // Check BSA files for base texture
            std::string bsaPath = baseName + ".dds";
            bool foundInBSA = false;
            
            for (const auto& entry : cacheMap) {
                const std::string& filename = entry.second.filename;
                if (filename.find(bsaPath) != std::string::npos) {
                    variants.baseTextureSource = "bsa";
                    variants.baseTexturePath = filename;
                    basesFound++;
                    foundInBSA = true;
                    break;
                }
            }
            
            if (!foundInBSA) {
                LOG::logline("-- WARNING: Base texture not found for: %s (looked in %s)", baseName.c_str(), looseBasePath.c_str());
            }
        }
    }
    
    // Phase 3: Move to final database (entries with found base textures OR grass textures)
    for (const auto& pair : suffixMap) {
        const TextureSuffixVariants& variants = pair.second;
        if (!variants.baseTextureSource.empty() || variants.isGrassTexture) {
            textureSuffixDatabase[variants.baseName] = variants;
        }
    }
    
    textureSuffixDatabaseBuilt = true;
    LOG::logline("-- Suffix database built: %d suffix files, %d bases found, %d complete entries", 
                suffixFilesFound, basesFound, (int)textureSuffixDatabase.size());
}

// getTextureSuffixVariants - Get suffix variants for a base texture name
const TextureSuffixVariants* getTextureSuffixVariants(const char* baseTextureName) {
    if (!textureSuffixDatabaseBuilt) {
        buildTextureSuffixDatabase();
    }
    
    std::string normalized = normalizeTexturePath(baseTextureName);
    std::string baseName = extractBaseName(normalized);
    
    auto it = textureSuffixDatabase.find(baseName);
    return (it != textureSuffixDatabase.end()) ? &it->second : nullptr;
}

// calculateTextureHash - Calculate runtime hash of a texture for identification
TextureRuntimeHash calculateTextureHash(IDirect3DDevice9* device, IDirect3DTexture9* texture, bool useCache) {
    TextureRuntimeHash hash = {0, 0};
    
    if (!texture) {
        return hash;
    }
    
    // Get texture dimensions first
    D3DSURFACE_DESC desc;
    if (FAILED(texture->GetLevelDesc(0, &desc))) {
        return hash;
    }
    
    // Check cache first to avoid repeated calculations (only if useCache is true)
    TextureCacheKey cacheKey = {desc.Width, desc.Height, desc.Format, desc.Pool};
    if (useCache) {
        auto cacheIt = runtimeTextureHashCache.find(cacheKey);
        if (cacheIt != runtimeTextureHashCache.end()) {
            return cacheIt->second;
        }
    }
    
    // Filtering checks
    bool isDefaultPool = (desc.Pool == D3DPOOL_DEFAULT);
    bool isManagedPool = (desc.Pool == D3DPOOL_MANAGED);
    bool isRenderTarget = (desc.Usage & D3DUSAGE_RENDERTARGET) != 0;
    bool isDepthStencil = (desc.Usage & D3DUSAGE_DEPTHSTENCIL) != 0;
    
    // Dynamic pool filtering based on "Reduce Texture Memory Use" setting
    // When enabled: runtime textures use D3DPOOL_DEFAULT
    // When disabled: runtime textures use D3DPOOL_MANAGED
    D3DPOOL expectedPool = Configuration.UseDefaultTexturePool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED;
    bool isExpectedPool = (desc.Pool == expectedPool);
    
    if (!isExpectedPool) {
        return hash; // Return zero hash for filtered textures
    }
    if (isRenderTarget || isDepthStencil) {
        return hash; // Return zero hash for filtered textures
    }
    
    // Use staging texture approach for direct memory hashing (no temp files)
    bool hashSuccess = false;
    
    // Use staging texture for D3DPOOL_DEFAULT textures
    IDirect3DTexture9* stagingTexture = nullptr;
    HRESULT hr = device->CreateTexture(desc.Width, desc.Height, 1, 0, desc.Format, D3DPOOL_SYSTEMMEM, &stagingTexture, nullptr);
    
    if (SUCCEEDED(hr) && stagingTexture) {
        // Get surfaces for copy operation
        IDirect3DSurface9* srcSurface = nullptr;
        IDirect3DSurface9* dstSurface = nullptr;
        
        if (SUCCEEDED(texture->GetSurfaceLevel(0, &srcSurface)) &&
            SUCCEEDED(stagingTexture->GetSurfaceLevel(0, &dstSurface))) {
            
            // Use GetRenderTargetData - proven to work 100% of the time for Pool=0 textures
            hr = device->GetRenderTargetData(srcSurface, dstSurface);

            if (FAILED(hr)) {
                LOG::logline("CAPTURE: GetRenderTargetData FAILED (0x%08x) for %dx%d %s Pool=%d - unexpected!",
                           hr, desc.Width, desc.Height, getD3DFormatName(desc.Format), desc.Pool);
            }
            
            // If staging succeeded, hash directly from memory
            if (SUCCEEDED(hr)) {
                D3DLOCKED_RECT lockedRect;
                if (SUCCEEDED(dstSurface->LockRect(&lockedRect, nullptr, D3DLOCK_READONLY))) {
                    // Calculate the total size of pixel data more carefully
                    size_t dataSize = 0;
                    
                    // Sanity checks first
                    if (lockedRect.Pitch <= 0 || desc.Width == 0 || desc.Height == 0) {
                        LOG::logline("!! Invalid texture properties: Pitch=%d, Width=%d, Height=%d", 
                                   lockedRect.Pitch, desc.Width, desc.Height);
                        dstSurface->UnlockRect();
                        return hash;
                    }
                    
                    switch (desc.Format) {
                        case D3DFMT_DXT1:
                            // DXT1: 4x4 blocks, 8 bytes per block
                            dataSize = ((desc.Width + 3) / 4) * ((desc.Height + 3) / 4) * 8;
                            break;
                        case D3DFMT_DXT3:
                        case D3DFMT_DXT5:
                            // DXT3/5: 4x4 blocks, 16 bytes per block  
                            dataSize = ((desc.Width + 3) / 4) * ((desc.Height + 3) / 4) * 16;
                            break;
                        case D3DFMT_A8R8G8B8:
                        case D3DFMT_X8R8G8B8:
                            dataSize = (size_t)lockedRect.Pitch * desc.Height;
                            break;
                        case D3DFMT_R5G6B5:
                        case D3DFMT_A1R5G5B5:
                            dataSize = (size_t)lockedRect.Pitch * desc.Height;
                            break;
                        case D3DFMT_A8:
                            dataSize = (size_t)lockedRect.Pitch * desc.Height;
                            break;
                        default:
                            // Fallback: use pitch * height for unknown formats
                            dataSize = (size_t)lockedRect.Pitch * desc.Height;
                            break;
                    }
                    
                    // Additional sanity checks
                    if (dataSize == 0 || dataSize > 100 * 1024 * 1024) { // Max 100MB
                        LOG::logline("!! Invalid texture data size calculated: %zu bytes (Format=%d, %dx%d, Pitch=%d)", 
                                   dataSize, desc.Format, desc.Width, desc.Height, lockedRect.Pitch);
                        dstSurface->UnlockRect();
                        return hash;
                    }
                    
                    if (lockedRect.pBits == nullptr) {
                        LOG::logline("!! Null texture data pointer");
                        dstSurface->UnlockRect();
                        return hash;
                    }
                    
                    // Hybrid texture hashing: Multi-point sample + 8x8 mip + enhanced metadata
                    const unsigned char* dataPtr = reinterpret_cast<const unsigned char*>(lockedRect.pBits);

                    // Multi-point sampling for better discrimination
                    size_t pointSize = 64; // 64 bytes per sample point
                    size_t maxSampleSize = std::min(dataSize, (size_t)2048); // 2KB from main texture

                    // Enhanced metadata: include mip levels and texture size for uniqueness
                    DWORD mipLevels = texture->GetLevelCount();
                    size_t metadataSize = 20; // 5 x DWORD (Width, Height, Format, MipLevels, TotalSize)
                    size_t totalHashSize = metadataSize + maxSampleSize;
                    auto hashBuffer = std::make_unique<unsigned char[]>(totalHashSize);

                    // Pack enhanced metadata
                    DWORD* metadata = reinterpret_cast<DWORD*>(hashBuffer.get());
                    metadata[0] = desc.Width;
                    metadata[1] = desc.Height;
                    metadata[2] = (DWORD)desc.Format;
                    metadata[3] = mipLevels;
                    metadata[4] = (DWORD)dataSize;

                    // Multi-point sampling: beginning, middle, end points
                    unsigned char* sampleBuffer = hashBuffer.get() + metadataSize;

                    // Sample 1: Beginning (first 64 bytes)
                    size_t beginSample = std::min(pointSize, dataSize);
                    memcpy(sampleBuffer, dataPtr, beginSample);

                    // Sample 2: Middle point
                    if (dataSize > pointSize * 2) {
                        size_t midOffset = dataSize / 2;
                        size_t midSample = std::min(pointSize, dataSize - midOffset);
                        memcpy(sampleBuffer + pointSize, dataPtr + midOffset, midSample);
                    }

                    // Sample 3: End point
                    if (dataSize > pointSize * 3) {
                        size_t endOffset = dataSize - pointSize;
                        size_t endSample = std::min(pointSize, dataSize - endOffset);
                        memcpy(sampleBuffer + (2 * pointSize), dataPtr + endOffset, endSample);
                    }

                    // Add 8x8 mip level data for enhanced discrimination
                    size_t mipDataUsed = 0;
                    if (mipLevels > 1) {
                        // Find 8x8 mip level using proven method
                        DWORD targetMip = 0;
                        for (DWORD mip = 0; mip < mipLevels; mip++) {
                            DWORD mipWidth = std::max(1u, desc.Width >> mip);
                            DWORD mipHeight = std::max(1u, desc.Height >> mip);

                            if (mipWidth <= 8 && mipHeight <= 8) {
                                targetMip = mip;
                                break;
                            }
                            targetMip = mip;
                        }

                        // Capture 8x8 mip using proven GetRenderTargetData method
                        if (targetMip > 0) {
                            D3DSURFACE_DESC mipDesc;
                            if (SUCCEEDED(texture->GetLevelDesc(targetMip, &mipDesc))) {
                                IDirect3DTexture9* mipTexture = nullptr;
                                HRESULT mipHr = device->CreateTexture(mipDesc.Width, mipDesc.Height, 1, 0, mipDesc.Format, D3DPOOL_SYSTEMMEM, &mipTexture, nullptr);

                                if (SUCCEEDED(mipHr) && mipTexture) {
                                    IDirect3DSurface9* mipSrcSurface = nullptr;
                                    IDirect3DSurface9* mipDstSurface = nullptr;

                                    if (SUCCEEDED(texture->GetSurfaceLevel(targetMip, &mipSrcSurface)) &&
                                        SUCCEEDED(mipTexture->GetSurfaceLevel(0, &mipDstSurface))) {

                                        // Use proven capture method
                                        HRESULT mipCaptureHr = device->GetRenderTargetData(mipSrcSurface, mipDstSurface);

                                        if (SUCCEEDED(mipCaptureHr)) {
                                            D3DLOCKED_RECT mipLockedRect;
                                            if (SUCCEEDED(mipDstSurface->LockRect(&mipLockedRect, nullptr, D3DLOCK_READONLY))) {
                                                // Calculate 8x8 mip data size
                                                size_t mipDataSize = 0;
                                                switch (mipDesc.Format) {
                                                    case D3DFMT_DXT1:
                                                        mipDataSize = std::max(1U, (mipDesc.Width + 3) / 4) * std::max(1U, (mipDesc.Height + 3) / 4) * 8;
                                                        break;
                                                    case D3DFMT_DXT3:
                                                    case D3DFMT_DXT5:
                                                        mipDataSize = std::max(1U, (mipDesc.Width + 3) / 4) * std::max(1U, (mipDesc.Height + 3) / 4) * 16;
                                                        break;
                                                    default:
                                                        mipDataSize = (size_t)mipLockedRect.Pitch * mipDesc.Height;
                                                        break;
                                                }

                                                // Add 8x8 mip data to hash (up to 256 bytes)
                                                if (mipDataSize > 0 && mipDataSize <= 256) {
                                                    size_t availableSpace = maxSampleSize - (3 * pointSize);
                                                    mipDataUsed = std::min(mipDataSize, availableSpace);
                                                    if (mipDataUsed > 0) {
                                                        memcpy(sampleBuffer + (3 * pointSize), mipLockedRect.pBits, mipDataUsed);
                                                    }
                                                }

                                                mipDstSurface->UnlockRect();
                                            }
                                        }
                                    }

                                    if (mipSrcSurface) mipSrcSurface->Release();
                                    if (mipDstSurface) mipDstSurface->Release();
                                    mipTexture->Release();
                                }
                            }
                        }
                    }

                    // Update total hash size to include 8x8 mip data
                    totalHashSize = metadataSize + (3 * pointSize) + mipDataUsed;
                    
                    // Calculate hash
                    hash.size = desc.Width * desc.Height;
                    hash.crc32 = crc32(hashBuffer.get(), totalHashSize);
                    
                    if (hash.crc32 != 0) {
                        hashSuccess = true;
                    }
                    
                    dstSurface->UnlockRect();
                } else {
                    LOG::logline("!! Failed to lock staging texture surface");
                }
            }
        }
        
        if (srcSurface) srcSurface->Release();
        if (dstSurface) dstSurface->Release();
    }
    
    // Only release stagingTexture if it was successfully created
    if (stagingTexture) {
        stagingTexture->Release();
    }
    
    if (!hashSuccess) {
        return hash;
    }
    
    // Cache the calculated hash for future use (only if useCache is true)
    if (useCache) {
        runtimeTextureHashCache[cacheKey] = hash;
    }
    
    return hash;
}

// resolveTextureNameFromHash - Get texture name from runtime hash
const std::string* resolveTextureNameFromHash(const TextureRuntimeHash& hash) {
    auto it = textureHashToName.find(hash);
    return (it != textureHashToName.end()) ? &it->second : nullptr;
}

// loadSuffixTexture - Load a specific suffix variant of a texture
IDirect3DTexture9* loadSuffixTexture(IDirect3DDevice9* dev, const TextureSuffixVariants& variants, const char* suffixType) {
    const char* texturePath = nullptr;
    
    if (strcmp(suffixType, "diffparam") == 0 && variants.hasDiffParam()) {
        texturePath = variants.diffparam.c_str();
    } else if (strcmp(suffixType, "diffparam_t") == 0 && variants.hasDiffParamT()) {
        texturePath = variants.diffparam_t.c_str();
    } else if (strcmp(suffixType, "normal") == 0 && variants.hasNormal()) {
        texturePath = variants.normal.c_str();
    } else if (strcmp(suffixType, "param") == 0 && variants.hasParam()) {
        texturePath = variants.param.c_str();
    }
    
    if (texturePath) {
        // Add textures\\ prefix since loadTextureExact expects full path relative to Data Files
        std::string fullPath = "textures\\" + std::string(texturePath);
        return loadTextureExact(dev, fullPath.c_str());
    }
    
    return nullptr;
}

// dumpAllBSATextures - Dump all textures from BSA files to verify filename reading
void dumpAllBSATextures(const char* outputDir) {
    LOG::logline("-- Dumping all BSA textures to %s", outputDir);
    
    // Create output directory
    CreateDirectoryA(outputDir, NULL);
    
    int texturesDumped = 0;
    int totalTextures = 0;
    
    for (const auto& entry : cacheMap) {
        const CacheEntry& cacheEntry = entry.second;
        const std::string& filename = cacheEntry.filename;
        
        // Skip non-texture files
        if (filename.find("textures/") == std::string::npos && filename.find("textures\\") == std::string::npos) {
            continue;
        }
        
        // Check if it's a texture file
        size_t dotPos = filename.find_last_of('.');
        if (dotPos == std::string::npos) continue;
        
        std::string extension = filename.substr(dotPos + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        if (extension != "dds" && extension != "tga" && extension != "bmp") {
            continue;
        }
        
        totalTextures++;
        
        // Read data from BSA
        SetFilePointer(cacheEntry.file, cacheEntry.position, 0, FILE_BEGIN);
        
        auto buffer = std::make_unique<char[]>(cacheEntry.size);
        DWORD bytesRead;
        ReadFile(cacheEntry.file, buffer.get(), cacheEntry.size, &bytesRead, 0);
        
        if (bytesRead != cacheEntry.size) {
            LOG::logline("!! Failed to read BSA entry: %s", filename.c_str());
            continue;
        }
        
        // Create output file path
        std::string outputPath = std::string(outputDir) + "\\" + filename;
        
        // Create directory structure
        size_t slashPos = outputPath.find_last_of("/\\");
        if (slashPos != std::string::npos) {
            std::string dirPath = outputPath.substr(0, slashPos);
            
            // Create nested directories
            size_t pos = 0;
            while ((pos = dirPath.find_first_of("/\\", pos + 1)) != std::string::npos) {
                std::string subDir = dirPath.substr(0, pos);
                CreateDirectoryA(subDir.c_str(), NULL);
            }
            CreateDirectoryA(dirPath.c_str(), NULL);
        }
        
        // Write file
        HANDLE outputFile = CreateFileA(outputPath.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (outputFile != INVALID_HANDLE_VALUE) {
            DWORD bytesWritten;
            WriteFile(outputFile, buffer.get(), cacheEntry.size, &bytesWritten, NULL);
            CloseHandle(outputFile);
            
            if (bytesWritten == cacheEntry.size) {
                texturesDumped++;
            } else {
                LOG::logline("!! Failed to write file: %s", outputPath.c_str());
            }
        } else {
            LOG::logline("!! Failed to create file: %s", outputPath.c_str());
        }
        
        // Log progress every 100 files
        if (texturesDumped % 100 == 0) {
            LOG::logline("-- Dumped %d/%d textures so far", texturesDumped, totalTextures);
        }
    }
    
    LOG::logline("-- BSA texture dump complete: %d/%d textures dumped to %s", texturesDumped, totalTextures, outputDir);
}

// Device state diagnostic function
void logDeviceState(IDirect3DDevice9* device, const char* stage) {
    LOG::logline("=== DEVICE STATE DIAGNOSTIC: %s ===", stage);
    
    // Check device capabilities
    D3DCAPS9 caps;
    if (SUCCEEDED(device->GetDeviceCaps(&caps))) {
        LOG::logline("Device Caps: TextureCaps=0x%08x, MaxTextureWidth=%d, MaxTextureHeight=%d", 
                    caps.TextureCaps, caps.MaxTextureWidth, caps.MaxTextureHeight);
    }
    
    // Check current render target
    IDirect3DSurface9* renderTarget = nullptr;
    if (SUCCEEDED(device->GetRenderTarget(0, &renderTarget)) && renderTarget) {
        D3DSURFACE_DESC rtDesc;
        if (SUCCEEDED(renderTarget->GetDesc(&rtDesc))) {
            LOG::logline("Render Target: %dx%d, Format=%d, Pool=%d, Usage=0x%08x", 
                        rtDesc.Width, rtDesc.Height, rtDesc.Format, rtDesc.Pool, rtDesc.Usage);
        }
        renderTarget->Release();
    }
    
    // Check key render states
    DWORD alphaBlend, zWrite, lighting;
    device->GetRenderState(D3DRS_ALPHABLENDENABLE, &alphaBlend);
    device->GetRenderState(D3DRS_ZWRITEENABLE, &zWrite);
    device->GetRenderState(D3DRS_LIGHTING, &lighting);
    LOG::logline("Render States: AlphaBlend=%d, ZWrite=%d, Lighting=%d", alphaBlend, zWrite, lighting);
    
    // Test texture creation in different pools
    IDirect3DTexture9* testTextures[3] = {nullptr, nullptr, nullptr};
    D3DPOOL pools[3] = {D3DPOOL_DEFAULT, D3DPOOL_MANAGED, D3DPOOL_SYSTEMMEM};
    const char* poolNames[3] = {"DEFAULT", "MANAGED", "SYSTEMMEM"};
    
    for (int i = 0; i < 3; i++) {
        HRESULT hr = device->CreateTexture(64, 64, 1, 0, D3DFMT_A8R8G8B8, pools[i], &testTextures[i], nullptr);
        LOG::logline("Test Texture %s: %s", poolNames[i], SUCCEEDED(hr) ? "OK" : "FAILED");
        
        if (SUCCEEDED(hr) && testTextures[i]) {
            // Test save capability
            char testPath[256];
            snprintf(testPath, sizeof(testPath), "temp\\test_%s.dds", poolNames[i]);
            HRESULT saveHr = D3DXSaveTextureToFile(testPath, D3DXIFF_DDS, testTextures[i], nullptr);
            LOG::logline("Test Save %s: %s", poolNames[i], SUCCEEDED(saveHr) ? "OK" : "FAILED");
            DeleteFileA(testPath);
            
            testTextures[i]->Release();
        }
    }
    
    LOG::logline("=== END DEVICE STATE: %s ===", stage);
}

// calculateCRC32 - Calculate CRC32 hash of raw data
uint32_t calculateCRC32(const uint8_t* data, size_t length) {
    return crc32(data, length);
}

// addRuntimeTextureHash - Add a texture hash to the runtime database
void addRuntimeTextureHash(uint32_t crc32Hash, uint32_t size, const char* textureName) {
    TextureRuntimeHash hash;
    hash.crc32 = crc32Hash;
    hash.size = size;
    
    textureHashToName[hash] = std::string(textureName);
}


// buildBSATextureHashDatabase - Build hash database from base textures that have suffix variants
void buildBSATextureHashDatabase(IDirect3DDevice9* dev) {
    int databaseStartTime = HighResolutionTimer::getMicroseconds();
    LOG::logline("-- Starting texture hash database building with timing analysis");
    
    // Ensure suffix database is built first
    if (!textureSuffixDatabaseBuilt) {
        buildTextureSuffixDatabase();
    }
    
    // Clear existing hash database
    textureHashToName.clear();
    
    int texturesHashed = 0;
    int texturesMatched = 0;
    int hashCollisions = 0;
    
    // Hash base textures from suffix database (only textures with suffix variants)
    for (const auto& entry : textureSuffixDatabase) {
        const std::string& baseName = entry.first;
        const TextureSuffixVariants& variants = entry.second;
        
        IDirect3DTexture9* baseTexture = nullptr;
        HRESULT hr = E_FAIL;
        
        // Load base texture based on source (loose overrides BSA)
        if (variants.baseTextureSource == "loose") {
            // Load from loose file
            D3DPOOL runtimePool = Configuration.UseDefaultTexturePool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED;
            hr = D3DXCreateTextureFromFileEx(dev, variants.baseTexturePath.c_str(),
                D3DX_DEFAULT, D3DX_DEFAULT, D3DX_DEFAULT, 0, D3DFMT_UNKNOWN,
                runtimePool, D3DX_DEFAULT, D3DX_DEFAULT, 0, nullptr, nullptr, &baseTexture);
        }
        else if (variants.baseTextureSource == "bsa") {
            // Load from BSA file using proven BSALoadFile method
            BSAHash3 hash = hashString(variants.baseTexturePath.c_str());
            EntryData ed = BSALoadFile(hash);
            
            if (ed.valid()) {
                // Use DirectXTex staging pattern: SYSTEMMEM → DEFAULT via UpdateTexture
                if (Configuration.UseDefaultTexturePool) {
                    // Method 1: DirectXTex staging pattern for D3DPOOL_DEFAULT
                    IDirect3DTexture9* stagingTexture = nullptr;
                    
                    // Step 1: Load BSA data into D3DPOOL_SYSTEMMEM staging texture
                    HRESULT stagingHr = D3DXCreateTextureFromFileInMemoryEx(dev, ed.data.get(), ed.size,
                        D3DX_FROM_FILE, D3DX_FROM_FILE, D3DX_FROM_FILE,
                        0, D3DFMT_UNKNOWN, D3DPOOL_SYSTEMMEM, D3DX_DEFAULT, D3DX_DEFAULT, 0, nullptr, nullptr, &stagingTexture);
                    
                    if (SUCCEEDED(stagingHr) && stagingTexture) {
                        // Get staging texture properties
                        D3DSURFACE_DESC stagingDesc;
                        if (SUCCEEDED(stagingTexture->GetLevelDesc(0, &stagingDesc))) {
                            UINT mipLevels = stagingTexture->GetLevelCount();
                            
                            // Step 2: Create empty D3DPOOL_DEFAULT final texture
                            hr = dev->CreateTexture(stagingDesc.Width, stagingDesc.Height, mipLevels,
                                0, stagingDesc.Format, D3DPOOL_DEFAULT, &baseTexture, nullptr);
                            
                            if (SUCCEEDED(hr) && baseTexture) {
                                // Step 3: Use DirectXTex UpdateTexture transfer (SYSTEMMEM → DEFAULT)
                                HRESULT updateHr = dev->UpdateTexture(stagingTexture, baseTexture);
                                
                                if (!SUCCEEDED(updateHr)) {
                                    LOG::logline("!! DirectXTex UpdateTexture FAILED: %s (HRESULT: 0x%08x)", baseName.c_str(), updateHr);
                                    baseTexture->Release();
                                    baseTexture = nullptr;
                                }
                            } else {
                                LOG::logline("!! Failed to create D3DPOOL_DEFAULT texture: %s (HRESULT: 0x%08x)", baseName.c_str(), hr);
                            }
                        }
                        
                        stagingTexture->Release();
                    } else {
                        LOG::logline("!! Failed to create SYSTEMMEM staging texture: %s (HRESULT: 0x%08x)", baseName.c_str(), stagingHr);
                        hr = stagingHr;
                    }
                } else {
                    // Method 2: Direct loading for D3DPOOL_MANAGED (no staging needed)
                    hr = D3DXCreateTextureFromFileInMemoryEx(dev, ed.data.get(), ed.size, 
                        D3DX_FROM_FILE, D3DX_FROM_FILE, D3DX_FROM_FILE,
                        0, D3DFMT_UNKNOWN, D3DPOOL_MANAGED, D3DX_DEFAULT, D3DX_DEFAULT, 0, nullptr, nullptr, &baseTexture);
                    
                    if (!SUCCEEDED(hr)) {
                        LOG::logline("!! Direct MANAGED texture creation FAILED: %s (HRESULT: 0x%08x)", baseName.c_str(), hr);
                    }
                }
            }
        }
        
        if (SUCCEEDED(hr) && baseTexture) {
            
            // Calculate hash using same method as runtime textures (disable caching for BSA textures)
            TextureRuntimeHash texHash = calculateTextureHash(dev, baseTexture, false);
            
            if (texHash.crc32 != 0) {
                // Check for collision before adding
                auto existingEntry = textureHashToName.find(texHash);
                if (existingEntry != textureHashToName.end()) {
                    hashCollisions++;
                } else {
                    // Add to hash database
                    textureHashToName[texHash] = baseName;
                    texturesMatched++;
                }

            }
            
            baseTexture->Release();
        }
        
        texturesHashed++;
    }
    
    int databaseTotalTime = HighResolutionTimer::getMicroseconds() - databaseStartTime;
    float databaseTotalTimeMs = databaseTotalTime / 1000.0f;
    float avgTimePerTextureMs = texturesHashed > 0 ? databaseTotalTimeMs / texturesHashed : 0.0f;
    LOG::logline("Hash database complete: %d textures processed, %d matches, %.2f ms total",
               texturesHashed, texturesMatched, databaseTotalTimeMs);

    // Analyze hash collisions
    std::unordered_map<uint32_t, std::vector<std::string>> hashCollisionMap;
    for (const auto& entry : textureHashToName) {
        hashCollisionMap[entry.first.crc32].push_back(entry.second);
    }

    int collisionGroups = 0;
    int totalCollisions = 0;
    for (const auto& collision : hashCollisionMap) {
        if (collision.second.size() > 1) {
            collisionGroups++;
            totalCollisions += collision.second.size();

            LOG::logline("Collision Group #%d - Hash %08x (%d textures):",
                       collisionGroups, collision.first, (int)collision.second.size());
            for (const auto& texName : collision.second) {
                LOG::logline("  - %s", texName.c_str());
            }
        }
    }

    if (hashCollisions > 0) {
        LOG::logline("=== HASH COLLISION SUMMARY ===");
        LOG::logline("Real-time collisions detected: %d", hashCollisions);
        LOG::logline("Post-analysis collision groups: %d", collisionGroups);
        LOG::logline("Post-analysis textures with collisions: %d", totalCollisions);
        LOG::logline("Collision rate: %.2f%%", (float)hashCollisions / texturesMatched * 100.0f);
    } else {
        LOG::logline("=== NO HASH COLLISIONS DETECTED ===");
        LOG::logline("Perfect hash discrimination achieved!");
    }
}


}
