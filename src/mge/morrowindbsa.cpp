
#include "morrowindbsa.h"
#include "proxydx/d3d8header.h"
#include "support/log.h"

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



// CRC32 implementation for texture hashing (matches ReShade approach)
static unsigned int crc32_table[256];
static bool crc32_table_initialized = false;

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
    
    // Check for known suffixes and remove them
    if (baseName.length() > 10 && baseName.substr(baseName.length() - 10) == "_diffparam") {
        return baseName.substr(0, baseName.length() - 10);
    }
    if (baseName.length() > 3 && baseName.substr(baseName.length() - 3) == "_nh") {
        return baseName.substr(0, baseName.length() - 3);
    }
    if (baseName.length() > 2 && baseName.substr(baseName.length() - 2) == "_n") {
        return baseName.substr(0, baseName.length() - 2);
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
    
    if (baseName.length() > 10 && baseName.substr(baseName.length() - 10) == "_diffparam") {
        return "diffparam";
    }
    if (baseName.length() > 3 && baseName.substr(baseName.length() - 3) == "_nh") {
        return "normal";
    }
    if (baseName.length() > 2 && baseName.substr(baseName.length() - 2) == "_n") {
        return "normal";
    }
    if (baseName.length() > 6 && baseName.substr(baseName.length() - 6) == "_param") {
        return "diffparam";
    }
    
    return nullptr; // Base texture
}

// buildTextureSuffixDatabase - Analyze BSA contents to build texture suffix mapping
void buildTextureSuffixDatabase() {
    if (textureSuffixDatabaseBuilt) return;
    
    LOG::logline("-- Building texture suffix database from BSA files");
    int totalTextures = 0;
    int suffixTextures = 0;
    
    // Iterate through all cached BSA entries to find texture files
    for (const auto& entry : cacheMap) {
        const CacheEntry& cacheEntry = entry.second;
        const std::string& filename = cacheEntry.filename;
        
        // Skip non-texture files - check if filename contains "textures/"
        if (filename.find("textures/") == std::string::npos && filename.find("textures\\") == std::string::npos) {
            continue;
        }
        
        // Check if it's a texture file (dds, tga, bmp)
        size_t dotPos = filename.find_last_of('.');
        if (dotPos == std::string::npos) continue;
        
        std::string extension = filename.substr(dotPos + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        if (extension != "dds" && extension != "tga" && extension != "bmp") {
            continue;
        }
        
        totalTextures++;
        
        // Extract just the filename part (remove directory path)
        size_t slashPos = filename.find_last_of("/\\");
        std::string textureName = (slashPos != std::string::npos) ? filename.substr(slashPos + 1) : filename;
        std::string normalizedPath = normalizeTexturePath(textureName.c_str());
        
        const char* suffixType = getSuffixType(normalizedPath);
        if (suffixType) {
            suffixTextures++;
            std::string baseName = extractBaseName(normalizedPath);
            
            // Create or update suffix variants entry
            TextureSuffixVariants& variants = textureSuffixDatabase[baseName];
            variants.baseName = baseName;
            
            if (strcmp(suffixType, "diffparam") == 0) {
                variants.diffparam = filename;  // Store full BSA path
            } else if (strcmp(suffixType, "normal") == 0) {
                variants.normal = filename;  // Store full BSA path
            }
            
            LOG::logline("BSA: Found %s texture: %s -> base: %s", suffixType, filename.c_str(), baseName.c_str());
        }
        
        // Also store base textures for hash mapping
        if (!suffixType) {
            std::string baseName = extractBaseName(normalizedPath);
            TextureSuffixVariants& variants = textureSuffixDatabase[baseName];
            if (variants.baseName.empty()) {
                variants.baseName = baseName;
            }
        }
    }
    
    // Fallback: Also scan Data Files/textures for additional suffix textures
    WIN32_FIND_DATA findFileData;
    HANDLE hFind = FindFirstFile("Data Files\\textures\\*_diffparam.dds", &findFileData);
    
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            std::string filename = findFileData.cFileName;
            std::string normalizedPath = normalizeTexturePath(filename.c_str());
            
            const char* suffixType = getSuffixType(normalizedPath);
            if (suffixType) {
                std::string baseName = extractBaseName(normalizedPath);
                
                TextureSuffixVariants& variants = textureSuffixDatabase[baseName];
                variants.baseName = baseName;
                
                if (strcmp(suffixType, "diffparam") == 0 && variants.diffparam.empty()) {
                    variants.diffparam = "textures/" + filename;
                    suffixTextures++;
                    totalTextures++;
                }
            }
        } while (FindNextFile(hFind, &findFileData));
        FindClose(hFind);
    }
    
    hFind = FindFirstFile("Data Files\\textures\\*_nh.dds", &findFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            std::string filename = findFileData.cFileName;
            std::string normalizedPath = normalizeTexturePath(filename.c_str());
            
            const char* suffixType = getSuffixType(normalizedPath);
            if (suffixType) {
                std::string baseName = extractBaseName(normalizedPath);
                
                TextureSuffixVariants& variants = textureSuffixDatabase[baseName];
                variants.baseName = baseName;
                
                if (strcmp(suffixType, "normal") == 0 && variants.normal.empty()) {
                    variants.normal = "textures/" + filename;
                    suffixTextures++;
                    totalTextures++;
                }
            }
        } while (FindNextFile(hFind, &findFileData));
        FindClose(hFind);
    }
    
    // Hash database building is handled separately in ffeshader.cpp on first texture render
    
    textureSuffixDatabaseBuilt = true;
    LOG::logline("-- BSA: Built texture suffix database with %d total textures, %d suffix textures", totalTextures, suffixTextures);
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
TextureRuntimeHash calculateTextureHash(IDirect3DDevice9* device, IDirect3DTexture9* texture) {
    TextureRuntimeHash hash = {0, 0};
    
    if (!texture) {
        return hash;
    }
    
    // Get texture dimensions first
    D3DSURFACE_DESC desc;
    if (FAILED(texture->GetLevelDesc(0, &desc))) {
        return hash;
    }
    
    // Create system memory copy for hashing (compressed textures can't be locked directly)
    IDirect3DTexture9* systemTexture = nullptr;
    if (FAILED(D3DXCreateTexture(device, desc.Width, desc.Height, 1, 0, 
                                D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, &systemTexture))) {
        LOG::logline("HASH ERROR: Failed to create system memory texture for %dx%d", desc.Width, desc.Height);
        return hash;
    }
    
    // Get surfaces for copying
    IDirect3DSurface9* srcSurface = nullptr;
    IDirect3DSurface9* dstSurface = nullptr;
    
    if (FAILED(texture->GetSurfaceLevel(0, &srcSurface)) ||
        FAILED(systemTexture->GetSurfaceLevel(0, &dstSurface))) {
        LOG::logline("HASH ERROR: Failed to get surfaces for %dx%d", desc.Width, desc.Height);
        if (srcSurface) srcSurface->Release();
        if (dstSurface) dstSurface->Release();
        systemTexture->Release();
        return hash;
    }
    
    // Copy texture data to system memory format (decompresses if needed)
    if (FAILED(D3DXLoadSurfaceFromSurface(dstSurface, nullptr, nullptr,
                                         srcSurface, nullptr, nullptr, D3DX_FILTER_NONE, 0))) {
        LOG::logline("HASH ERROR: Failed to copy surface data for %dx%d", desc.Width, desc.Height);
        srcSurface->Release();
        dstSurface->Release();
        systemTexture->Release();
        return hash;
    }
    
    srcSurface->Release();
    dstSurface->Release();
    
    // Now we can lock the system memory texture
    IDirect3DSurface9* surface = nullptr;
    if (FAILED(systemTexture->GetSurfaceLevel(0, &surface))) {
        LOG::logline("HASH ERROR: Failed GetSurfaceLevel for system texture %dx%d", desc.Width, desc.Height);
        systemTexture->Release();
        return hash;
    }
    
    D3DLOCKED_RECT lockedRect;
    if (SUCCEEDED(surface->LockRect(&lockedRect, nullptr, D3DLOCK_READONLY))) {
        // Calculate hash of decompressed pixel data
        hash.size = desc.Width * desc.Height;
        hash.crc32 = crc32(static_cast<const unsigned char*>(lockedRect.pBits), 
                          lockedRect.Pitch * desc.Height);
        surface->UnlockRect();
        LOG::logline("HASH SUCCESS: %dx%d texture -> hash %08x, size %u", 
                    desc.Width, desc.Height, hash.crc32, hash.size);
    } else {
        LOG::logline("HASH ERROR: Failed LockRect for system texture %dx%d", desc.Width, desc.Height);
    }
    
    surface->Release();
    systemTexture->Release();
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
    } else if (strcmp(suffixType, "normal") == 0 && variants.hasNormal()) {
        texturePath = variants.normal.c_str();
    }
    
    if (texturePath) {
        return loadTextureExact(dev, texturePath);
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

// buildBSATextureHashDatabase - Build hash database from BSA textures that matches runtime format
void buildBSATextureHashDatabase(IDirect3DDevice9* dev) {
    int texturesHashed = 0;
    int texturesMatched = 0;
    
    // Clear existing hash database
    textureHashToName.clear();
    
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
        
        texturesHashed++;
        
        // Read texture data from BSA
        SetFilePointer(cacheEntry.file, cacheEntry.position, 0, FILE_BEGIN);
        
        auto buffer = std::make_unique<char[]>(cacheEntry.size);
        DWORD bytesRead;
        ReadFile(cacheEntry.file, buffer.get(), cacheEntry.size, &bytesRead, 0);
        
        if (bytesRead != cacheEntry.size) {
            continue;
        }
        
        // Try to load texture using D3DX to match runtime format
        IDirect3DTexture9* bsaTexture = nullptr;
        HRESULT hr = D3DXCreateTextureFromFileInMemory(
            dev, buffer.get(), cacheEntry.size, &bsaTexture);
            
        if (SUCCEEDED(hr) && bsaTexture) {
            // Calculate hash using same method as runtime textures
            TextureRuntimeHash texHash = calculateTextureHash(dev, bsaTexture);
            
            if (texHash.crc32 != 0) {
                // Extract just the filename for storage
                size_t slashPos = filename.find_last_of("/\\");
                std::string textureName = (slashPos != std::string::npos) ? 
                    filename.substr(slashPos + 1) : filename;
                
                // Store hash->name mapping
                textureHashToName[texHash] = textureName;
                texturesMatched++;
                
                // Log first few matches for verification
                if (texturesMatched <= 5) {
                    LOG::logline("BSA Hash: %08x -> %s (%dx%d)", 
                               texHash.crc32, textureName.c_str(),
                               texHash.size & 0xFFFF, (texHash.size >> 16) & 0xFFFF);
                }
            }
            
            bsaTexture->Release();
        }
        
        // Log progress every 100 textures
        if (texturesHashed % 100 == 0) {
            LOG::logline("-- BSA hash progress: %d textures processed, %d matched", 
                       texturesHashed, texturesMatched);
        }
    }
    
    LOG::logline("-- BSA texture hash database complete: %d textures processed, %d hash matches created", 
               texturesHashed, texturesMatched);
}

}
