#pragma once

#include <d3d9.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstdint>

// Resource tracking system for CPU-side data caching
// Eliminates GPU locks during recording and enables async bbox computation

class ResourceTracker {
public:
    using ResourceID = uint32_t;
    static constexpr ResourceID INVALID_ID = 0;

    struct VertexBufferData {
        ResourceID id;
        UINT length;
        DWORD usage;
        DWORD fvf;
        D3DPOOL pool;
        UINT stride;
        bool isDynamic;
        std::vector<uint8_t> cpuData;  // CPU-side copy for static buffers
        uint32_t lockCount;
    };

    struct IndexBufferData {
        ResourceID id;
        UINT length;
        DWORD usage;
        D3DFORMAT format;
        D3DPOOL pool;
        bool isDynamic;
        std::vector<uint8_t> cpuData;  // CPU-side copy for static buffers
        uint32_t lockCount;
    };

    struct TextureData {
        ResourceID id;
        UINT width;
        UINT height;
        UINT levels;
        DWORD usage;
        D3DFORMAT format;
        D3DPOOL pool;
        bool isDynamic;
        // For now, don't store texture pixels (too large)
        // Could store hash or small mips for debugging
    };

    // Singleton access
    static ResourceTracker& getInstance();

    // Resource creation tracking
    ResourceID trackVertexBuffer(IDirect3DVertexBuffer9* vb, UINT length, DWORD usage, DWORD fvf, D3DPOOL pool);
    ResourceID trackIndexBuffer(IDirect3DIndexBuffer9* ib, UINT length, DWORD usage, D3DFORMAT format, D3DPOOL pool);
    ResourceID trackTexture(IDirect3DTexture9* tex, UINT width, UINT height, UINT levels, DWORD usage, D3DFORMAT format, D3DPOOL pool);

    // Lock/Unlock tracking for CPU-side capture
    void onVertexBufferLock(IDirect3DVertexBuffer9* vb, void* data, UINT offset, UINT size, DWORD flags);
    void onVertexBufferUnlock(IDirect3DVertexBuffer9* vb);
    void onIndexBufferLock(IDirect3DIndexBuffer9* ib, void* data, UINT offset, UINT size, DWORD flags);
    void onIndexBufferUnlock(IDirect3DIndexBuffer9* ib);

    // Resource destruction
    void untrackVertexBuffer(IDirect3DVertexBuffer9* vb);
    void untrackIndexBuffer(IDirect3DIndexBuffer9* ib);
    void untrackTexture(IDirect3DTexture9* tex);

    // Data access (thread-safe)
    const VertexBufferData* getVertexBufferData(IDirect3DVertexBuffer9* vb) const;
    const IndexBufferData* getIndexBufferData(IDirect3DIndexBuffer9* ib) const;
    const TextureData* getTextureData(IDirect3DTexture9* tex) const;

    ResourceID getVertexBufferID(IDirect3DVertexBuffer9* vb) const;
    ResourceID getIndexBufferID(IDirect3DIndexBuffer9* ib) const;
    ResourceID getTextureID(IDirect3DTexture9* tex) const;

    // Access by ResourceID (returns nullptr if ID not found or no CPU data)
    const uint8_t* getVertexBufferData(ResourceID id) const;
    const uint8_t* getIndexBufferData(ResourceID id) const;
    bool isIndexBuffer16Bit(ResourceID id) const;

    // Clear all tracking data
    void clear();

private:
    ResourceTracker() : nextID(1) {}
    ResourceTracker(const ResourceTracker&) = delete;
    ResourceTracker& operator=(const ResourceTracker&) = delete;

    ResourceID allocateID() { return nextID++; }

    mutable std::mutex mutex;
    ResourceID nextID;

    std::unordered_map<IDirect3DVertexBuffer9*, VertexBufferData> vertexBuffers;
    std::unordered_map<IDirect3DIndexBuffer9*, IndexBufferData> indexBuffers;
    std::unordered_map<IDirect3DTexture9*, TextureData> textures;

    // Reverse lookup maps (ID -> pointer) for CPU-side data access
    std::unordered_map<ResourceID, IDirect3DVertexBuffer9*> vbIDToPointer;
    std::unordered_map<ResourceID, IDirect3DIndexBuffer9*> ibIDToPointer;

    // Temporary lock tracking
    struct LockData {
        void* lockedPtr;
        UINT offset;
        UINT size;
    };
    std::unordered_map<void*, LockData> activeLocks;
};
