#pragma once

#include "proxydx/d3d9header.h"
#include <functional>

// Mesh identifier for bbox caching (VB + IB + FVF combo uniquely identifies object-space mesh)
struct MeshKey {
    IDirect3DVertexBuffer9* vb;
    IDirect3DIndexBuffer9* ib;
    DWORD fvf;
    UINT baseIndex;
    UINT vertCount;
    UINT startIndex;
    UINT primCount;

    bool operator==(const MeshKey& other) const {
        return vb == other.vb && ib == other.ib && fvf == other.fvf &&
               baseIndex == other.baseIndex && vertCount == other.vertCount &&
               startIndex == other.startIndex && primCount == other.primCount;
    }
};

// Hash function for MeshKey
struct MeshKeyHash {
    std::size_t operator()(const MeshKey& k) const {
        std::size_t h1 = std::hash<void*>{}(k.vb);
        std::size_t h2 = std::hash<void*>{}(k.ib);
        std::size_t h3 = std::hash<DWORD>{}(k.fvf);
        std::size_t h4 = std::hash<UINT>{}(k.baseIndex);
        std::size_t h5 = std::hash<UINT>{}(k.vertCount);
        std::size_t h6 = std::hash<UINT>{}(k.startIndex);
        std::size_t h7 = std::hash<UINT>{}(k.primCount);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5) ^ (h7 << 6);
    }
};

// Cached object-space bounding box
struct ObjectSpaceBBox {
    D3DXVECTOR3 bboxMin;
    D3DXVECTOR3 bboxMax;
};

// VB+IB key for bbox lookup between recordedCalls and recordMW
struct VBIBKey {
    IDirect3DVertexBuffer9* vb;
    IDirect3DIndexBuffer9* ib;

    bool operator==(const VBIBKey& other) const {
        return vb == other.vb && ib == other.ib;
    }
};

struct VBIBKeyHash {
    std::size_t operator()(const VBIBKey& k) const {
        std::size_t h1 = std::hash<void*>{}(k.vb);
        std::size_t h2 = std::hash<void*>{}(k.ib);
        return h1 ^ (h2 << 1);
    }
};
