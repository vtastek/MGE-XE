#pragma once

#include <string>
#include <unordered_map>
#include <vector>

struct IDirect3DDevice9;
struct IDirect3DTexture9;


namespace BSA {
    // Texture suffix variants structure
    struct TextureSuffixVariants {
        std::string baseName;
        std::string diffparam;         // _diffparam texture path
        std::string diffparam_t;       // _diffparam_t texture path (terrain)
        std::string paramh;            // _paramh texture path (metallic/roughness/IOR or metallic/height/IOR)
        std::string paramx;            // _paramx texture path (aniso rotation/strength/metallic)
        std::string baseTextureSource; // "loose" or "bsa"
        std::string baseTexturePath;   // Full path to base texture
        bool isGrassTexture;           // True if texture is in grass folder

        bool hasDiffParam() const { return !diffparam.empty(); }
        bool hasDiffParamT() const { return !diffparam_t.empty(); }
        bool hasParamH() const { return !paramh.empty(); }
        bool hasParamX() const { return !paramx.empty(); }
        bool hasGrass() const { return isGrassTexture; }
    };
    
    // Texture runtime hash for identification
    struct TextureRuntimeHash {
        unsigned int crc32;
        unsigned int size;
        
        bool operator==(const TextureRuntimeHash& other) const {
            return crc32 == other.crc32;  // Only compare CRC32, size can vary between BSA/runtime
        }
    };
    
    struct TextureRuntimeHasher {
        size_t operator()(const TextureRuntimeHash& hash) const {
            return static_cast<size_t>(hash.crc32);  // Only hash CRC32, ignore size
        }
    };

    void init();
    IDirect3DTexture9* loadTexture(IDirect3DDevice9* dev, const char* filename);
    void clearTextureCache();
    void cacheStats(int* total, int* memuse);
    
    // Texture suffix functionality
    void buildTextureSuffixDatabase();
    const TextureSuffixVariants* getTextureSuffixVariants(const char* baseTextureName);
    TextureRuntimeHash calculateTextureHash(IDirect3DDevice9* device, IDirect3DTexture9* texture, bool useCache = true);
    const std::string* resolveTextureNameFromHash(const TextureRuntimeHash& hash);
    IDirect3DTexture9* loadSuffixTexture(IDirect3DDevice9* dev, const TextureSuffixVariants& variants, const char* suffixType);
    
    // BSA verification functionality
    void dumpAllBSATextures(const char* outputDir);
    
    // Runtime texture hashing functionality
    uint32_t calculateCRC32(const uint8_t* data, size_t length);
    void addRuntimeTextureHash(uint32_t crc32Hash, uint32_t size, const char* textureName);
    void buildBSATextureHashDatabase(IDirect3DDevice9* dev);
    
    // Device state diagnostics
    void logDeviceState(IDirect3DDevice9* device, const char* stage);
    
}
