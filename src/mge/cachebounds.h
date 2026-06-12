#pragma once

// cachebounds.h — world-space cull bounds for a GeometryCache entry.
//
// Shared by every consumer that frustum-tests the cache: the reflection /
// cache-terrain paths (rendercachedcolor.cpp), the main opaque pass, and the
// early deterministic visible-set build (buildFrustumVisibleSet, renderdepth.cpp).
// These two functions encode hard-won correctness fixes (terrain-origin offset,
// skinned bind-pose torso placement) and MUST be used verbatim — never re-derive
// a cull center from the object origin.

#include "dlmath.h"
#include "scenegraph_geometry_cache.h"

#include <algorithm>

// World-space bounding sphere for a cache entry. The cache stores the geometry's
// bound in MODEL space (boundsCenter/boundsRadius); the cull frustum is in world
// space. Using the object ORIGIN (worldTransformD3D translation) as the sphere
// center is wrong whenever the geometry is offset from its node origin — most
// dramatically for terrain patches, whose origin is the cell corner ~4096u from
// the real patch center, causing false culls when the camera is close/over the
// cell. Transform the model center by the world matrix and scale the radius by the
// largest axis scale so the sphere actually encloses the drawn geometry.
inline void cacheWorldBounds(const MGE::GeometryCache::CachedGeometry& e, D3DXVECTOR3& outCenter, float& outRadius) {
    const D3DXMATRIX& w = *reinterpret_cast<const D3DXMATRIX*>(e.worldTransformD3D);
    const D3DXVECTOR3 modelC(e.boundsCenter[0], e.boundsCenter[1], e.boundsCenter[2]);
    D3DXVec3TransformCoord(&outCenter, &modelC, &w);
    const float sx = D3DXVec3Length(reinterpret_cast<const D3DXVECTOR3*>(&w._11));
    const float sy = D3DXVec3Length(reinterpret_cast<const D3DXVECTOR3*>(&w._21));
    const float sz = D3DXVec3Length(reinterpret_cast<const D3DXVECTOR3*>(&w._31));
    outRadius = e.boundsRadius * std::max(sx, std::max(sy, sz));
}

// World-space tight AABB for a NON-SKINNED cache entry: the 8 corners of the stored
// model-space AABB transformed by the world matrix, then bounded. This mirrors
// FixedFunctionShader::computeBoundingBox (which walks the VB) so the cache color
// pass selects the SAME point lights as the reactive path — the cache VB is
// WRITEONLY, so renderMorrowind can't compute it and would otherwise get a fat
// sphere-cube box that over-selects lights (the "windows brighter in cache" delta).
inline void cacheWorldAABB(const MGE::GeometryCache::CachedGeometry& e,
                           D3DXVECTOR3& outMin, D3DXVECTOR3& outMax) {
    const D3DXMATRIX& w = *reinterpret_cast<const D3DXMATRIX*>(e.worldTransformD3D);
    const float* mn = e.aabbMin;
    const float* mx = e.aabbMax;
    bool first = true;
    for (int c = 0; c < 8; ++c) {
        const D3DXVECTOR3 corner((c & 1) ? mx[0] : mn[0],
                                 (c & 2) ? mx[1] : mn[1],
                                 (c & 4) ? mx[2] : mn[2]);
        D3DXVECTOR3 wc;
        D3DXVec3TransformCoord(&wc, &corner, &w);
        if (first) { outMin = outMax = wc; first = false; }
        else {
            outMin.x = std::min(outMin.x, wc.x); outMax.x = std::max(outMax.x, wc.x);
            outMin.y = std::min(outMin.y, wc.y); outMax.y = std::max(outMax.y, wc.y);
            outMin.z = std::min(outMin.z, wc.z); outMax.z = std::max(outMax.z, wc.z);
        }
    }
}

// World-space bounding sphere for a SKINNED cache entry, derived from the per-frame
// bone palette. The skinned VB is bind-pose; the posed geometry is placed entirely
// by the bones, so the geom's node origin (worldTransformD3D) is NOT where the posed
// part is. Using the origin as the cull center with the small bind-pose radius
// falsely culls close skinned parts whose skeleton root sits away from the part
// (e.g. a character's own torso at 1-2m: it reflects fine at 5-7m, then vanishes as
// the reflected frustum tightens around the misplaced origin sphere). Transform the
// bind-pose bounds center by each bone (model->world) and bound the resulting point
// set: every posed vertex is a weighted (convex) combination of bone-transformed
// bind positions, so a sphere over those centers plus the scaled bind radius
// conservatively encloses the posed part.
//
// Caller MUST exclude entries with numBones == 0 (div-by-zero) and
// skinnedUnsupported (numBones may exceed kMaxBones — the pts[] array overruns).
inline void cacheSkinnedWorldBounds(const MGE::GeometryCache::CachedGeometry& e,
                                    D3DXVECTOR3& outCenter, float& outRadius) {
    const int n = (int)e.numBones;
    const D3DXMATRIX* pal = reinterpret_cast<const D3DXMATRIX*>(e.bonePalette.data());
    const D3DXVECTOR3 modelC(e.boundsCenter[0], e.boundsCenter[1], e.boundsCenter[2]);

    D3DXVECTOR3 pts[MGE::GeometryCache::kMaxBones];
    D3DXVECTOR3 c(0, 0, 0);
    float maxScale = 0.0f;
    for (int i = 0; i < n; ++i) {
        D3DXVec3TransformCoord(&pts[i], &modelC, &pal[i]);
        c += pts[i];
        const float sx = D3DXVec3Length(reinterpret_cast<const D3DXVECTOR3*>(&pal[i]._11));
        const float sy = D3DXVec3Length(reinterpret_cast<const D3DXVECTOR3*>(&pal[i]._21));
        const float sz = D3DXVec3Length(reinterpret_cast<const D3DXVECTOR3*>(&pal[i]._31));
        maxScale = std::max(maxScale, std::max(sx, std::max(sy, sz)));
    }
    c /= (float)n;
    float r = 0.0f;
    for (int i = 0; i < n; ++i) {
        const D3DXVECTOR3 d = pts[i] - c;
        r = std::max(r, D3DXVec3Length(&d));
    }
    outCenter = c;
    outRadius = r + e.boundsRadius * maxScale;
}
