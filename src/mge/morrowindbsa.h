#pragma once

struct IDirect3DDevice9;
struct IDirect3DTexture9;


namespace BSA {
    void init();
    IDirect3DTexture9* loadTexture(IDirect3DDevice9* dev, const char* filename);
    void clearTextureCache();
    void cacheStats(int* total, int* memuse);

    // Resolve a texture to its RAW FILE BYTES (DDS/TGA/etc.) using the same priority as
    // loadTexture (distantland\statics -> loose Data Files -> BSA) with .dds extension
    // substitution, but WITHOUT creating a D3D9 texture. `filename` is the bare texture
    // name (no "textures\" / "Data Files" prefix); this prepends "textures\" like loadTexture.
    // On success *outData is a malloc'd buffer the caller frees with std::free, *outSize its
    // length. Returns false if not found. Used by the Forge texture-residency path: ship the
    // raw DDS to the host process, which decodes it (mips + BC intact) — no GPU readback.
    // skipDistantStatics: when true, do NOT prefer the distantland\statics LOD folder (its
    // downscaled copies blur near geometry); resolve loose Data Files -> BSA like the engine's
    // near renderer. The Forge near-texturing path passes true.
    bool loadFileBytes(const char* filename, void** outData, unsigned* outSize, bool skipDistantStatics = false);
}
