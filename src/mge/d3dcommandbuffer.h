#pragma once

#include "proxydx/d3d9header.h"
#include <vector>

struct D3DCmd {
    enum Type : uint8_t {
        Cmd_SetRenderState, Cmd_SetTextureStageState, Cmd_SetSamplerState,
        Cmd_SetTransform, Cmd_SetTexture, Cmd_SetMaterial,
        Cmd_SetLight, Cmd_LightEnable,
        Cmd_SetFVF, Cmd_SetStreamSource, Cmd_SetIndices,
        Cmd_DrawIndexedPrimitive, Cmd_DrawPrimitive,
        Cmd_Clear, Cmd_SetRenderTarget, Cmd_SetDepthStencilSurface,
        Cmd_BeginScene, Cmd_EndScene, Cmd_SetViewport
    };
    Type type;

    union {
        struct { DWORD state; DWORD value; } rs;
        struct { DWORD stage; DWORD state; DWORD value; } tss;
        struct { DWORD sampler; DWORD state; DWORD value; } ss;
        struct { DWORD transformType; D3DMATRIX matrix; } xform;
        struct { DWORD stage; IDirect3DBaseTexture9* tex; } tex;
        struct { D3DMATERIAL9 material; } mat;
        struct { DWORD index; D3DLIGHT9 light; } light;
        struct { DWORD index; BOOL enable; } lightEn;
        struct { DWORD fvf; } fvf;
        struct { UINT stream; IDirect3DVertexBuffer9* vb; UINT offset; UINT stride; } streamSrc;
        struct { IDirect3DIndexBuffer9* ib; } indices;
        struct { D3DPRIMITIVETYPE primType; INT baseVertex;
                 UINT minIndex; UINT numVerts; UINT startIndex; UINT primCount; } dip;
        struct { D3DPRIMITIVETYPE primType; UINT startVertex; UINT primCount; } dp;
        struct { DWORD count; DWORD flags; D3DCOLOR color; float z; DWORD stencil; } clear;
        struct { IDirect3DSurface9* surface; } rt;
        struct { IDirect3DSurface9* surface; } ds;
        struct { D3DVIEWPORT9 viewport; } vp;
    };
};

class D3DCommandBuffer {
public:
    D3DCommandBuffer() = default;
    ~D3DCommandBuffer() { clear(); }

    // Non-copyable
    D3DCommandBuffer(const D3DCommandBuffer&) = delete;
    D3DCommandBuffer& operator=(const D3DCommandBuffer&) = delete;

    // Movable
    D3DCommandBuffer(D3DCommandBuffer&& other) noexcept;
    D3DCommandBuffer& operator=(D3DCommandBuffer&& other) noexcept;

    void clear();
    void replay(IDirect3DDevice9* device) const;

    // Record methods
    void recordSetRenderState(DWORD state, DWORD value);
    void recordSetTextureStageState(DWORD stage, DWORD state, DWORD value);
    void recordSetSamplerState(DWORD sampler, DWORD state, DWORD value);
    void recordSetTransform(DWORD type, const D3DMATRIX* matrix);
    void recordSetTexture(DWORD stage, IDirect3DBaseTexture9* tex);
    void recordSetMaterial(const D3DMATERIAL9* mat);
    void recordSetLight(DWORD index, const D3DLIGHT9* light);
    void recordLightEnable(DWORD index, BOOL enable);
    void recordSetFVF(DWORD fvf);
    void recordSetStreamSource(UINT stream, IDirect3DVertexBuffer9* vb, UINT offset, UINT stride);
    void recordSetIndices(IDirect3DIndexBuffer9* ib);
    void recordDrawIndexedPrimitive(D3DPRIMITIVETYPE type, INT baseVertex,
                                     UINT minIdx, UINT numVerts, UINT startIdx, UINT primCount);
    void recordDrawPrimitive(D3DPRIMITIVETYPE type, UINT startVertex, UINT primCount);
    void recordClear(DWORD count, DWORD flags, D3DCOLOR color, float z, DWORD stencil);
    void recordSetRenderTarget(IDirect3DSurface9* surface);
    void recordSetDepthStencilSurface(IDirect3DSurface9* surface);
    void recordBeginScene();
    void recordEndScene();
    void recordSetViewport(const D3DVIEWPORT9* vp);

    int size() const { return (int)commands.size(); }
    size_t sizeBytes() const { return commands.size() * sizeof(D3DCmd); }

private:
    std::vector<D3DCmd> commands;
};
