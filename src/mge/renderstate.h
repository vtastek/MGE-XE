#pragma once

#include "proxydx/d3d9header.h"
#include <unordered_map>
#include <vector>

// Rendered state captured per draw call
struct RenderedState {
    IDirect3DTexture9* texture;
    IDirect3DVertexBuffer9* vb;
    UINT vbOffset, vbStride;
    IDirect3DIndexBuffer9* ib;
    DWORD ibBase;
    DWORD fvf;
    DWORD zWrite, cullMode;
    DWORD vertexBlendState;
    D3DXMATRIX worldTransforms[4];
    D3DXMATRIX viewTransform;
    D3DXMATRIX worldViewTransforms[4];
    D3DXMATRIX shadowWorldViewProj[2];  // Complete shadow world-view-projection matrices at time of recording
    D3DCOLORVALUE diffuseMaterial;
    BYTE blendEnable, srcBlend, destBlend;
    BYTE alphaTest, alphaFunc, alphaRef;
    BYTE useLighting, useFog, matSrcDiffuse, matSrcEmissive;

    D3DPRIMITIVETYPE primType;
    UINT baseIndex, minIndex, vertCount, startIndex, primCount;

    // Bounding box for occlusion culling during depth rendering
    D3DXVECTOR3 bboxMin, bboxMax;
    bool hasBoundingBox;

    // Scene number (0=world, 1=particles, 2=hands)
    int sceneNum = 0;

    // Debug flags
    bool debugWireframe = false;  // Render as wireframe (debug visualization)
};

// RecordedMWState - RenderedState with COM reference management for deferred rendering.
// Used by both DistantLand::recordMW/recordSky and FrameBuffer per-frame isolation.
struct RecordedMWState : RenderedState {
    RecordedMWState(const RenderedState& state);
    ~RecordedMWState();
    RecordedMWState(const RecordedMWState&) = delete;
    RecordedMWState(RecordedMWState&& source) noexcept;
};

// Fragment/texture stage state
struct FragmentState {
    struct Stage {
        BYTE colorOp, colorArg1, colorArg2;
        BYTE alphaOp, alphaArg1, alphaArg2;
        BYTE colorArg0, alphaArg0, resultArg;
        DWORD texcoordIndex;
        DWORD texTransformFlags;
        float bumpEnvMat[2][2];
        float bumpLumiScale, bumpLumiBias;
    } stage[8];

    struct Material {
        D3DCOLORVALUE diffuse, ambient, emissive;
    } material;
};

// Light state for shader lighting calculations
struct LightState {
    struct Light {
        D3DLIGHTTYPE type;
        D3DCOLORVALUE diffuse;
        D3DVECTOR position;     // position / normalized direction
        D3DVECTOR viewspacePos;
        union {
            D3DVECTOR falloff;  // constant, linear, quadratic
            D3DVECTOR ambient;  // for directional lights
        };
    };

    D3DCOLORVALUE globalAmbient;
    std::unordered_map<DWORD, Light> lights;
    std::unordered_map<DWORD, bool> lightsTransformed;
    std::vector<DWORD> active;
};
