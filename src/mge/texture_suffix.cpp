// Texture suffix resolution and caching system implementation
#include "texture_suffix.h"
#include "configuration.h"
#include "support/log.h"

namespace TextureSuffix {

// Resolution cache: maps texture pointer to resolved suffix data
static std::unordered_map<IDirect3DTexture9*, ResolutionCache> s_resolutionCache;
static SRWLOCK s_cacheLock = SRWLOCK_INIT;

// Per-texture suffix flags cache (simpler, for quick lookups)
static std::unordered_map<IDirect3DTexture9*, SuffixTextureFlags> s_suffixFlagsCache;

// Binding state for caching suffix texture binds
static BindingState s_bindingState;

void init() {
    // Lock is zero-initialized (valid for SRWLOCK)
    // Clear caches on init
    clearCaches();
}

void warmCache(IDirect3DDevice9* device, IDirect3DTexture9* texture) {
    if (!texture) return;

    AcquireSRWLockShared(&s_cacheLock);
    bool found = s_resolutionCache.count(texture) > 0;
    ReleaseSRWLockShared(&s_cacheLock);

    if (found) return;  // Already cached

    // Expensive path: device calls for hash computation (main thread only)
    ResolutionCache entry;
    entry.hash = BSA::calculateTextureHash(device, texture, false);

    const std::string* textureName = BSA::resolveTextureNameFromHash(entry.hash);
    if (textureName && entry.hash.crc32 != 0) {
        entry.textureName = *textureName;
        entry.hasValidName = true;
        entry.variants = BSA::getTextureSuffixVariants(textureName->c_str());

        // Pre-load suffix textures so bindShaderTextures never stalls on disk I/O
        if (entry.variants) {
            if (entry.variants->hasDiffParamT()) {
                BSA::loadSuffixTexture(device, *entry.variants, "diffparam_t");
            } else if (entry.variants->hasDiffParam()) {
                BSA::loadSuffixTexture(device, *entry.variants, "diffparam");
            }
            if (entry.variants->hasParamH()) {
                BSA::loadSuffixTexture(device, *entry.variants, "paramh");
            }
            if (entry.variants->hasParamX()) {
                BSA::loadSuffixTexture(device, *entry.variants, "paramx");
            }
        }
    } else {
        entry.hasValidName = false;
        entry.variants = nullptr;
    }

    AcquireSRWLockExclusive(&s_cacheLock);
    s_resolutionCache.emplace(texture, std::move(entry));
    ReleaseSRWLockExclusive(&s_cacheLock);
}

SuffixTextureFlags getFlagsForTexture(IDirect3DDevice9* device, IDirect3DTexture9* texture) {
    SuffixTextureFlags flags = {false, false, false, false};

    if (!texture || Configuration.PerPixelLightFlags != 2) {
        return flags;
    }

    // Check simple cache first
    auto cacheIt = s_suffixFlagsCache.find(texture);
    if (cacheIt != s_suffixFlagsCache.end()) {
        return cacheIt->second;
    }

    // Not in cache - calculate hash and determine suffix flags
    BSA::TextureRuntimeHash texHash = BSA::calculateTextureHash(device, texture, false);

    if (texHash.crc32 != 0) {
        const std::string* textureName = BSA::resolveTextureNameFromHash(texHash);
        if (textureName) {
            LOG::logline("RUNTIME HASH MATCH: %08x -> %s", texHash.crc32, textureName->c_str());

            const BSA::TextureSuffixVariants* variants = BSA::getTextureSuffixVariants(textureName->c_str());
            if (variants && (variants->hasDiffParam() || variants->hasParamH() || variants->hasParamX() || variants->hasGrass())) {
                flags.hasDiffParam = variants->hasDiffParam() || variants->hasDiffParamT();
                flags.hasParamH = variants->hasParamH();
                flags.hasParamX = variants->hasParamX();
                flags.hasGrass = variants->hasGrass();
            }
        } else {
            LOG::logline("RUNTIME HASH FAILED: %08x -> NO MATCH FOUND", texHash.crc32);
        }
    }

    // Cache the result
    s_suffixFlagsCache[texture] = flags;

    return flags;
}

const ResolutionCache* getCachedResolution(IDirect3DTexture9* texture) {
    if (!texture) return nullptr;

    AcquireSRWLockShared(&s_cacheLock);
    auto it = s_resolutionCache.find(texture);
    const ResolutionCache* result = (it != s_resolutionCache.end()) ? &it->second : nullptr;
    ReleaseSRWLockShared(&s_cacheLock);
    return result;
}

const ResolutionCache* getOrCreateResolution(IDirect3DDevice9* device, IDirect3DTexture9* texture, bool allowDeviceCalls) {
    if (!texture) return nullptr;

    // Check cache first (shared lock)
    AcquireSRWLockShared(&s_cacheLock);
    auto it = s_resolutionCache.find(texture);
    if (it != s_resolutionCache.end()) {
        const ResolutionCache* result = &it->second;
        ReleaseSRWLockShared(&s_cacheLock);
        return result;
    }
    ReleaseSRWLockShared(&s_cacheLock);

    if (!allowDeviceCalls) {
        LOG::logline("Warning: suffix cache miss for texture 0x%p on cull thread", texture);
        return nullptr;
    }

    // Perform expensive resolution
    ResolutionCache entry;
    entry.hash = BSA::calculateTextureHash(device, texture, false);

    const std::string* textureName = BSA::resolveTextureNameFromHash(entry.hash);
    if (textureName && entry.hash.crc32 != 0) {
        entry.textureName = *textureName;
        entry.hasValidName = true;
        entry.variants = BSA::getTextureSuffixVariants(textureName->c_str());
    } else {
        entry.hasValidName = false;
        entry.variants = nullptr;
    }

    // Insert under exclusive lock
    AcquireSRWLockExclusive(&s_cacheLock);
    auto [insertIt, inserted] = s_resolutionCache.emplace(texture, std::move(entry));
    const ResolutionCache* result = &insertIt->second;
    ReleaseSRWLockExclusive(&s_cacheLock);

    return result;
}

void onTextureReleased(IDirect3DTexture9* texture) {
    AcquireSRWLockExclusive(&s_cacheLock);
    s_resolutionCache.erase(texture);
    ReleaseSRWLockExclusive(&s_cacheLock);

    // Also clear from simple flags cache
    s_suffixFlagsCache.erase(texture);
}

BindingState& getBindingState() {
    return s_bindingState;
}

SRWLOCK& getCacheLock() {
    return s_cacheLock;
}

void clearCaches() {
    AcquireSRWLockExclusive(&s_cacheLock);
    s_resolutionCache.clear();
    ReleaseSRWLockExclusive(&s_cacheLock);

    s_suffixFlagsCache.clear();
    s_bindingState.reset();
}

} // namespace TextureSuffix
