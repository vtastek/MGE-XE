#pragma once

#include "proxydx/d3d9header.h"
#include <vector>
#include <unordered_map>

// Forward declaration
struct StateContract;

// Shadow state tracker for MW state during recording.
// Mirrors SetXXX calls so we can reconstruct MW state without GetXXX.
// This enables true async: device can be owned by render thread while main records.
struct MWStateTracker {
    // Render states (keyed by D3DRENDERSTATETYPE)
    std::unordered_map<DWORD, DWORD> renderStates;

    // Sampler states (keyed by sampler * 256 + state)
    std::unordered_map<DWORD, DWORD> samplerStates;

    // Transforms (keyed by D3DTRANSFORMSTATETYPE)
    std::unordered_map<DWORD, D3DMATRIX> transforms;

    // FVF (DX8 vertex shader = FVF code)
    DWORD fvf = 0;
    bool fvfTracked = false;

    void trackFVF(DWORD value) { fvf = value; fvfTracked = true; }
    bool getFVF(DWORD* outValue) const { if (fvfTracked) { *outValue = fvf; return true; } return false; }

    // Light enable states (keyed by light index)
    std::unordered_map<DWORD, BOOL> lightEnables;

    void trackLightEnable(DWORD index, BOOL enable) {
        lightEnables[index] = enable;
    }

    // Texture stage states (non-sampler: ColorOp, AlphaOp, etc.), keyed by stage * 256 + state
    std::unordered_map<DWORD, DWORD> textureStageStates;

    void trackTextureStageState(DWORD stage, DWORD state, DWORD value) {
        textureStageStates[stage * 256 + state] = value;
    }
    bool getTextureStageState(DWORD stage, DWORD state, DWORD* outValue) const {
        auto it = textureStageStates.find(stage * 256 + state);
        if (it != textureStageStates.end()) { *outValue = it->second; return true; }
        return false;
    }

    // Track a render state change
    void trackRenderState(DWORD state, DWORD value) {
        renderStates[state] = value;
    }

    // Track a sampler state change
    void trackSamplerState(DWORD sampler, DWORD state, DWORD value) {
        samplerStates[sampler * 256 + state] = value;
    }

    // Track a transform change
    void trackTransform(DWORD type, const D3DMATRIX& matrix) {
        transforms[type] = matrix;
    }

    // Get render state (returns false if not tracked)
    bool getRenderState(DWORD state, DWORD* outValue) const {
        auto it = renderStates.find(state);
        if (it != renderStates.end()) {
            *outValue = it->second;
            return true;
        }
        return false;
    }

    // Get sampler state (returns false if not tracked)
    bool getSamplerState(DWORD sampler, DWORD state, DWORD* outValue) const {
        auto it = samplerStates.find(sampler * 256 + state);
        if (it != samplerStates.end()) {
            *outValue = it->second;
            return true;
        }
        return false;
    }

    // Get transform (returns false if not tracked)
    bool getTransform(DWORD type, D3DMATRIX* outMatrix) const {
        auto it = transforms.find(type);
        if (it != transforms.end()) {
            *outMatrix = it->second;
            return true;
        }
        return false;
    }

    // Export tracked state to StateContract (defined in recording_system.cpp)
    void exportToStateContract(struct StateContract* out) const;

    void clear() {
        renderStates.clear();
        samplerStates.clear();
        transforms.clear();
        textureStageStates.clear();
        lightEnables.clear();
        fvf = 0;
        fvfTracked = false;
    }

    // Seed tracker with current device state (call at start of recording)
    void seedFromDevice(IDirect3DDevice9* dev);
};

struct D3DCmd {
    enum Type : uint8_t {
        Cmd_SetRenderState, Cmd_SetTextureStageState, Cmd_SetSamplerState,
        Cmd_SetTransform, Cmd_SetTexture, Cmd_SetMaterial,
        Cmd_SetLight, Cmd_LightEnable,
        Cmd_SetFVF, Cmd_SetStreamSource, Cmd_SetIndices,
        Cmd_DrawIndexedPrimitive, Cmd_DrawPrimitive,
        Cmd_Clear, Cmd_SetRenderTarget, Cmd_SetDepthStencilSurface,
        Cmd_BeginScene, Cmd_EndScene, Cmd_SetViewport,
        Cmd_SetVertexShader, Cmd_SetPixelShader,
        Cmd_SetVSConstantF, Cmd_SetPSConstantF,
        Cmd_SetVSConstantI, Cmd_SetPSConstantI,
        Cmd_GetRenderTargetData
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
        struct { IDirect3DVertexShader9* shader; } vs;
        struct { IDirect3DPixelShader9* shader; } ps;
        struct { UINT startReg; UINT count; UINT arenaOffset; } constF;
        struct { IDirect3DSurface9* source; IDirect3DSurface9* dest; } rtData;
    };
};

enum class CmdStage : uint8_t {
    Offscreen, PreScene, Scene0, InterScene, Scene1, Scene2, UI, Count
};

const char* CmdStageName(CmdStage s);

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
    void replayStateOnly(IDirect3DDevice9* device) const;  // Replay without draw calls (for state restoration)

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
    void recordSetVertexShader(IDirect3DVertexShader9* shader);
    void recordSetPixelShader(IDirect3DPixelShader9* shader);
    void recordSetVSConstantF(UINT startReg, const float* data, UINT count);
    void recordSetPSConstantF(UINT startReg, const float* data, UINT count);
    void recordSetVSConstantI(UINT startReg, const int* data, UINT count);
    void recordSetPSConstantI(UINT startReg, const int* data, UINT count);
    void recordGetRenderTargetData(IDirect3DSurface9* source, IDirect3DSurface9* dest);

    int size() const { return (int)commands.size(); }
    size_t sizeBytes() const { return commands.size() * sizeof(D3DCmd) + constantArena.size() * sizeof(float); }
    void dumpToFrameLog(int sceneNum) const;

private:
    std::vector<D3DCmd> commands;
    std::vector<float> constantArena;
};

class D3DCommandBufferSet {
public:
    D3DCommandBuffer& active() { return buffers[(int)activeStage]; }
    const D3DCommandBuffer& active() const { return buffers[(int)activeStage]; }
    D3DCommandBuffer& operator[](CmdStage s) { return buffers[(int)s]; }
    const D3DCommandBuffer& operator[](CmdStage s) const { return buffers[(int)s]; }
    void clearAll();
    int totalSize() const;
    size_t totalSizeBytes() const;
    void dumpToFrameLog() const;

    // State tracker for shadow state during recording
    MWStateTracker& stateTracker() { return tracker; }
    const MWStateTracker& stateTracker() const { return tracker; }

    // Convenience: record + track in one call
    void recordAndTrackRenderState(DWORD state, DWORD value) {
        active().recordSetRenderState(state, value);
        tracker.trackRenderState(state, value);
    }
    void recordAndTrackSamplerState(DWORD sampler, DWORD state, DWORD value) {
        active().recordSetSamplerState(sampler, state, value);
        tracker.trackSamplerState(sampler, state, value);
    }
    void recordAndTrackTransform(DWORD type, const D3DMATRIX* matrix) {
        active().recordSetTransform(type, matrix);
        tracker.trackTransform(type, *matrix);
    }

    CmdStage activeStage = CmdStage::PreScene;
private:
    D3DCommandBuffer buffers[(int)CmdStage::Count];
    MWStateTracker tracker;
};
