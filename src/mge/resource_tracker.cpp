#include "resource_tracker.h"
#include "support/log.h"
#include <cstring>

ResourceTracker& ResourceTracker::getInstance() {
    static ResourceTracker instance;
    return instance;
}

ResourceTracker::ResourceID ResourceTracker::trackVertexBuffer(IDirect3DVertexBuffer9* vb, UINT length, DWORD usage, DWORD fvf, D3DPOOL pool) {
    std::lock_guard<std::mutex> lock(mutex);

    ResourceID id = allocateID();
    VertexBufferData data;
    data.id = id;
    data.length = length;
    data.usage = usage;
    data.fvf = fvf;
    data.pool = pool;
    data.isDynamic = (usage & D3DUSAGE_DYNAMIC) != 0;
    data.lockCount = 0;

    // Calculate stride from FVF
    data.stride = 0;
    if (fvf & D3DFVF_XYZ) data.stride += 12;
    if (fvf & D3DFVF_XYZRHW) data.stride += 16;
    if (fvf & D3DFVF_NORMAL) data.stride += 12;
    if (fvf & D3DFVF_DIFFUSE) data.stride += 4;
    if (fvf & D3DFVF_SPECULAR) data.stride += 4;
    DWORD texCount = (fvf & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT;
    data.stride += texCount * 8; // Assume 2D tex coords

    // For static buffers, pre-allocate CPU storage
    if (!data.isDynamic && pool != D3DPOOL_DEFAULT) {
        data.cpuData.reserve(length);
    }

    vertexBuffers[vb] = std::move(data);
    vbIDToPointer[id] = vb;  // Reverse lookup

    LOG::logline("ResourceTracker: VB created, ID=%u, length=%u, fvf=0x%X, stride=%u, dynamic=%d",
                 id, length, fvf, data.stride, data.isDynamic);

    return id;
}

ResourceTracker::ResourceID ResourceTracker::trackIndexBuffer(IDirect3DIndexBuffer9* ib, UINT length, DWORD usage, D3DFORMAT format, D3DPOOL pool) {
    std::lock_guard<std::mutex> lock(mutex);

    ResourceID id = allocateID();
    IndexBufferData data;
    data.id = id;
    data.length = length;
    data.usage = usage;
    data.format = format;
    data.pool = pool;
    data.isDynamic = (usage & D3DUSAGE_DYNAMIC) != 0;
    data.lockCount = 0;

    // For static buffers, pre-allocate CPU storage
    if (!data.isDynamic && pool != D3DPOOL_DEFAULT) {
        data.cpuData.reserve(length);
    }

    indexBuffers[ib] = std::move(data);
    ibIDToPointer[id] = ib;  // Reverse lookup

    LOG::logline("ResourceTracker: IB created, ID=%u, length=%u, format=%d, dynamic=%d",
                 id, length, format, data.isDynamic);

    return id;
}

ResourceTracker::ResourceID ResourceTracker::trackTexture(IDirect3DTexture9* tex, UINT width, UINT height, UINT levels, DWORD usage, D3DFORMAT format, D3DPOOL pool) {
    std::lock_guard<std::mutex> lock(mutex);

    ResourceID id = allocateID();
    TextureData data;
    data.id = id;
    data.width = width;
    data.height = height;
    data.levels = levels;
    data.usage = usage;
    data.format = format;
    data.pool = pool;
    data.isDynamic = (usage & D3DUSAGE_DYNAMIC) != 0;

    textures[tex] = std::move(data);

    LOG::logline("ResourceTracker: Texture created, ID=%u, %ux%u, levels=%u, dynamic=%d",
                 id, width, height, levels, data.isDynamic);

    return id;
}

void ResourceTracker::onVertexBufferLock(IDirect3DVertexBuffer9* vb, void* data, UINT offset, UINT size, DWORD flags) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = vertexBuffers.find(vb);
    if (it == vertexBuffers.end()) return;

    VertexBufferData& vbData = it->second;
    vbData.lockCount++;

    // Store lock info for unlock
    LockData lockData;
    lockData.lockedPtr = data;
    lockData.offset = offset;
    lockData.size = size == 0 ? vbData.length : size;
    activeLocks[vb] = lockData;
}

void ResourceTracker::onVertexBufferUnlock(IDirect3DVertexBuffer9* vb) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = vertexBuffers.find(vb);
    if (it == vertexBuffers.end()) return;

    VertexBufferData& vbData = it->second;

    // Find lock data
    auto lockIt = activeLocks.find(vb);
    if (lockIt == activeLocks.end()) return;

    const LockData& lockData = lockIt->second;

    // For static buffers, capture CPU-side copy on first unlock
    if (!vbData.isDynamic && vbData.cpuData.empty() && lockData.lockedPtr) {
        vbData.cpuData.resize(vbData.length);

        // Copy from locked memory
        if (lockData.offset == 0 && lockData.size >= vbData.length) {
            // Full buffer copy
            memcpy(vbData.cpuData.data(), lockData.lockedPtr, vbData.length);
            LOG::logline("ResourceTracker: VB ID=%u captured full CPU copy (%u bytes)", vbData.id, vbData.length);
        } else {
            // Partial copy - we need to lock again to get full buffer
            // For now, skip partial copies
            vbData.cpuData.clear();
        }
    }

    activeLocks.erase(lockIt);
}

void ResourceTracker::onIndexBufferLock(IDirect3DIndexBuffer9* ib, void* data, UINT offset, UINT size, DWORD flags) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = indexBuffers.find(ib);
    if (it == indexBuffers.end()) return;

    IndexBufferData& ibData = it->second;
    ibData.lockCount++;

    // Store lock info for unlock
    LockData lockData;
    lockData.lockedPtr = data;
    lockData.offset = offset;
    lockData.size = size == 0 ? ibData.length : size;
    activeLocks[ib] = lockData;
}

void ResourceTracker::onIndexBufferUnlock(IDirect3DIndexBuffer9* ib) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = indexBuffers.find(ib);
    if (it == indexBuffers.end()) return;

    IndexBufferData& ibData = it->second;

    // Find lock data
    auto lockIt = activeLocks.find(ib);
    if (lockIt == activeLocks.end()) return;

    const LockData& lockData = lockIt->second;

    // For static buffers, capture CPU-side copy on first unlock
    if (!ibData.isDynamic && ibData.cpuData.empty() && lockData.lockedPtr) {
        ibData.cpuData.resize(ibData.length);

        // Copy from locked memory
        if (lockData.offset == 0 && lockData.size >= ibData.length) {
            // Full buffer copy
            memcpy(ibData.cpuData.data(), lockData.lockedPtr, ibData.length);
            LOG::logline("ResourceTracker: IB ID=%u captured full CPU copy (%u bytes)", ibData.id, ibData.length);
        } else {
            // Partial copy - skip for now
            ibData.cpuData.clear();
        }
    }

    activeLocks.erase(lockIt);
}

void ResourceTracker::untrackVertexBuffer(IDirect3DVertexBuffer9* vb) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = vertexBuffers.find(vb);
    if (it != vertexBuffers.end()) {
        vbIDToPointer.erase(it->second.id);  // Remove reverse lookup
        vertexBuffers.erase(it);
    }
    activeLocks.erase(vb);
}

void ResourceTracker::untrackIndexBuffer(IDirect3DIndexBuffer9* ib) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = indexBuffers.find(ib);
    if (it != indexBuffers.end()) {
        ibIDToPointer.erase(it->second.id);  // Remove reverse lookup
        indexBuffers.erase(it);
    }
    activeLocks.erase(ib);
}

void ResourceTracker::untrackTexture(IDirect3DTexture9* tex) {
    std::lock_guard<std::mutex> lock(mutex);
    textures.erase(tex);
}

const ResourceTracker::VertexBufferData* ResourceTracker::getVertexBufferData(IDirect3DVertexBuffer9* vb) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = vertexBuffers.find(vb);
    return (it != vertexBuffers.end()) ? &it->second : nullptr;
}

const ResourceTracker::IndexBufferData* ResourceTracker::getIndexBufferData(IDirect3DIndexBuffer9* ib) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = indexBuffers.find(ib);
    return (it != indexBuffers.end()) ? &it->second : nullptr;
}

const ResourceTracker::TextureData* ResourceTracker::getTextureData(IDirect3DTexture9* tex) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = textures.find(tex);
    return (it != textures.end()) ? &it->second : nullptr;
}

ResourceTracker::ResourceID ResourceTracker::getVertexBufferID(IDirect3DVertexBuffer9* vb) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = vertexBuffers.find(vb);
    return (it != vertexBuffers.end()) ? it->second.id : INVALID_ID;
}

ResourceTracker::ResourceID ResourceTracker::getIndexBufferID(IDirect3DIndexBuffer9* ib) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = indexBuffers.find(ib);
    return (it != indexBuffers.end()) ? it->second.id : INVALID_ID;
}

ResourceTracker::ResourceID ResourceTracker::getTextureID(IDirect3DTexture9* tex) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = textures.find(tex);
    return (it != textures.end()) ? it->second.id : INVALID_ID;
}

void ResourceTracker::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    vertexBuffers.clear();
    indexBuffers.clear();
    textures.clear();
    vbIDToPointer.clear();
    ibIDToPointer.clear();
    activeLocks.clear();
    LOG::logline("ResourceTracker: Cleared all tracked resources");
}

// Access by ResourceID
const uint8_t* ResourceTracker::getVertexBufferData(ResourceID id) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto idIt = vbIDToPointer.find(id);
    if (idIt == vbIDToPointer.end()) return nullptr;

    auto vbIt = vertexBuffers.find(idIt->second);
    if (vbIt == vertexBuffers.end() || vbIt->second.cpuData.empty()) return nullptr;

    return vbIt->second.cpuData.data();
}

const uint8_t* ResourceTracker::getIndexBufferData(ResourceID id) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto idIt = ibIDToPointer.find(id);
    if (idIt == ibIDToPointer.end()) return nullptr;

    auto ibIt = indexBuffers.find(idIt->second);
    if (ibIt == indexBuffers.end() || ibIt->second.cpuData.empty()) return nullptr;

    return ibIt->second.cpuData.data();
}

bool ResourceTracker::isIndexBuffer16Bit(ResourceID id) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto idIt = ibIDToPointer.find(id);
    if (idIt == ibIDToPointer.end()) return false;

    auto ibIt = indexBuffers.find(idIt->second);
    if (ibIt == indexBuffers.end()) return false;

    return ibIt->second.format == D3DFMT_INDEX16;
}
