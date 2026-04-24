// Texture suffix resolution and caching system
// Handles BSA texture suffix detection (_paramh, _paramx, _grass)
#pragma once

#include "proxydx/d3d8header.h"
#include "morrowindbsa.h"
#include <unordered_map>
#include <string>

namespace TextureSuffix {

// Texture suffix flags for shader variant selection
struct SuffixTextureFlags {
    bool hasParamH = false;
    bool hasParamX = false;
    bool hasGrass = false;
    bool paramhNoParallax = false;  // paramh authored as _paramh_np: height-derived normal only, no parallax step
};

// Cached texture resolution data (avoids repeated hash calculations)
struct ResolutionCache {
    BSA::TextureRuntimeHash hash;
    std::string textureName;
    bool hasValidName;
    const BSA::TextureSuffixVariants* variants;
    float paramHWidth;   // paramH texture dims in texels; 0 if no paramH variant / not yet resolved
    float paramHHeight;

    ResolutionCache() : hasValidName(false), variants(nullptr), paramHWidth(0.0f), paramHHeight(0.0f) {}
};

// Suffix texture binding state (for caching bound textures)
struct BindingState {
    IDirect3DTexture9* lastBaseTexture;  // Texture pointer for fast comparison
    std::string currentBaseTextureName;
    IDirect3DTexture9* boundParamH;
    IDirect3DTexture9* boundParamX;

    BindingState() : lastBaseTexture(nullptr), boundParamH(nullptr), boundParamX(nullptr) {}

    void reset() {
        lastBaseTexture = nullptr;
        currentBaseTextureName.clear();
        boundParamH = nullptr;
        boundParamX = nullptr;
    }
};

// Initialize the suffix system
void init();

// Pre-populate suffix cache during recording (main thread only).
// Device calls (CreateTexture, GetRenderTargetData) are only safe on the main thread.
void warmCache(IDirect3DDevice9* device, IDirect3DTexture9* texture);

// Get suffix flags for a specific texture
SuffixTextureFlags getFlagsForTexture(IDirect3DDevice9* device, IDirect3DTexture9* texture);

// Look up cached resolution data for a texture (returns nullptr if not cached)
const ResolutionCache* getCachedResolution(IDirect3DTexture9* texture);

// Get or create resolution cache entry (may perform expensive hash calculation)
const ResolutionCache* getOrCreateResolution(IDirect3DDevice9* device, IDirect3DTexture9* texture, bool allowDeviceCalls = true);

// Texture release callback (evicts from cache on texture free)
void onTextureReleased(IDirect3DTexture9* texture);

// Get the binding state for caching suffix texture binds
BindingState& getBindingState();

// Access the resolution cache lock for thread-safe operations
SRWLOCK& getCacheLock();

// Clear all caches (for testing or reset)
void clearCaches();

} // namespace TextureSuffix
