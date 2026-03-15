#include "d3dcommandbuffer.h"
#include "imgui_manager.h"

// --- MWStateTracker ---

void MWStateTracker::seedFromDevice(IDirect3DDevice9* dev) {
    DWORD val;

    // Depth state
    dev->GetRenderState(D3DRS_ZENABLE, &val); renderStates[D3DRS_ZENABLE] = val;
    dev->GetRenderState(D3DRS_ZWRITEENABLE, &val); renderStates[D3DRS_ZWRITEENABLE] = val;
    dev->GetRenderState(D3DRS_ZFUNC, &val); renderStates[D3DRS_ZFUNC] = val;

    // Blending state
    dev->GetRenderState(D3DRS_ALPHABLENDENABLE, &val); renderStates[D3DRS_ALPHABLENDENABLE] = val;
    dev->GetRenderState(D3DRS_SRCBLEND, &val); renderStates[D3DRS_SRCBLEND] = val;
    dev->GetRenderState(D3DRS_DESTBLEND, &val); renderStates[D3DRS_DESTBLEND] = val;

    // Alpha test state
    dev->GetRenderState(D3DRS_ALPHATESTENABLE, &val); renderStates[D3DRS_ALPHATESTENABLE] = val;
    dev->GetRenderState(D3DRS_ALPHAFUNC, &val); renderStates[D3DRS_ALPHAFUNC] = val;
    dev->GetRenderState(D3DRS_ALPHAREF, &val); renderStates[D3DRS_ALPHAREF] = val;

    // Culling and fog
    dev->GetRenderState(D3DRS_CULLMODE, &val); renderStates[D3DRS_CULLMODE] = val;
    dev->GetRenderState(D3DRS_FOGENABLE, &val); renderStates[D3DRS_FOGENABLE] = val;

    // Specular and lighting
    dev->GetRenderState(D3DRS_SPECULARENABLE, &val); renderStates[D3DRS_SPECULARENABLE] = val;
    dev->GetRenderState(D3DRS_LOCALVIEWER, &val); renderStates[D3DRS_LOCALVIEWER] = val;
    dev->GetRenderState(D3DRS_NORMALIZENORMALS, &val); renderStates[D3DRS_NORMALIZENORMALS] = val;

    // Sampler states for stages 0-1
    for (DWORD s = 0; s < 2; ++s) {
        dev->GetSamplerState(s, D3DSAMP_MINFILTER, &val); samplerStates[s * 256 + D3DSAMP_MINFILTER] = val;
        dev->GetSamplerState(s, D3DSAMP_MAGFILTER, &val); samplerStates[s * 256 + D3DSAMP_MAGFILTER] = val;
        dev->GetSamplerState(s, D3DSAMP_MIPFILTER, &val); samplerStates[s * 256 + D3DSAMP_MIPFILTER] = val;
        dev->GetSamplerState(s, D3DSAMP_ADDRESSU, &val); samplerStates[s * 256 + D3DSAMP_ADDRESSU] = val;
        dev->GetSamplerState(s, D3DSAMP_ADDRESSV, &val); samplerStates[s * 256 + D3DSAMP_ADDRESSV] = val;
    }

    // Transforms
    D3DMATRIX mat;
    dev->GetTransform(D3DTS_WORLD, &mat); transforms[D3DTS_WORLD] = mat;
    dev->GetTransform(D3DTS_VIEW, &mat); transforms[D3DTS_VIEW] = mat;
    dev->GetTransform(D3DTS_PROJECTION, &mat); transforms[D3DTS_PROJECTION] = mat;
}

// --- Move semantics ---

D3DCommandBuffer::D3DCommandBuffer(D3DCommandBuffer&& other) noexcept
    : commands(std::move(other.commands)), constantArena(std::move(other.constantArena)) {
}

D3DCommandBuffer& D3DCommandBuffer::operator=(D3DCommandBuffer&& other) noexcept {
    if (this != &other) {
        clear();
        commands = std::move(other.commands);
        constantArena = std::move(other.constantArena);
    }
    return *this;
}

// --- COM ref release ---

void D3DCommandBuffer::clear() {
    for (auto& cmd : commands) {
        switch (cmd.type) {
        case D3DCmd::Cmd_SetTexture:
            if (cmd.tex.tex) cmd.tex.tex->Release();
            break;
        case D3DCmd::Cmd_SetStreamSource:
            if (cmd.streamSrc.vb) cmd.streamSrc.vb->Release();
            break;
        case D3DCmd::Cmd_SetIndices:
            if (cmd.indices.ib) cmd.indices.ib->Release();
            break;
        case D3DCmd::Cmd_SetRenderTarget:
            if (cmd.rt.surface) cmd.rt.surface->Release();
            break;
        case D3DCmd::Cmd_SetDepthStencilSurface:
            if (cmd.ds.surface) cmd.ds.surface->Release();
            break;
        case D3DCmd::Cmd_SetVertexShader:
            if (cmd.vs.shader) cmd.vs.shader->Release();
            break;
        case D3DCmd::Cmd_SetPixelShader:
            if (cmd.ps.shader) cmd.ps.shader->Release();
            break;
        case D3DCmd::Cmd_GetRenderTargetData:
            if (cmd.rtData.source) cmd.rtData.source->Release();
            if (cmd.rtData.dest) cmd.rtData.dest->Release();
            break;
        default:
            break;
        }
    }
    commands.clear();
    constantArena.clear();
}

// --- Record methods ---

void D3DCommandBuffer::recordSetRenderState(DWORD state, DWORD value) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetRenderState;
    cmd.rs.state = state;
    cmd.rs.value = value;
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetTextureStageState(DWORD stage, DWORD state, DWORD value) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetTextureStageState;
    cmd.tss.stage = stage;
    cmd.tss.state = state;
    cmd.tss.value = value;
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetSamplerState(DWORD sampler, DWORD state, DWORD value) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetSamplerState;
    cmd.ss.sampler = sampler;
    cmd.ss.state = state;
    cmd.ss.value = value;
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetTransform(DWORD type, const D3DMATRIX* matrix) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetTransform;
    cmd.xform.transformType = type;
    cmd.xform.matrix = *matrix;
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetTexture(DWORD stage, IDirect3DBaseTexture9* tex) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetTexture;
    cmd.tex.stage = stage;
    cmd.tex.tex = tex;
    if (tex) tex->AddRef();
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetMaterial(const D3DMATERIAL9* mat) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetMaterial;
    cmd.mat.material = *mat;
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetLight(DWORD index, const D3DLIGHT9* light) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetLight;
    cmd.light.index = index;
    cmd.light.light = *light;
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordLightEnable(DWORD index, BOOL enable) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_LightEnable;
    cmd.lightEn.index = index;
    cmd.lightEn.enable = enable;
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetFVF(DWORD fvf) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetFVF;
    cmd.fvf.fvf = fvf;
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetStreamSource(UINT stream, IDirect3DVertexBuffer9* vb, UINT offset, UINT stride) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetStreamSource;
    cmd.streamSrc.stream = stream;
    cmd.streamSrc.vb = vb;
    cmd.streamSrc.offset = offset;
    cmd.streamSrc.stride = stride;
    if (vb) vb->AddRef();
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetIndices(IDirect3DIndexBuffer9* ib) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetIndices;
    cmd.indices.ib = ib;
    if (ib) ib->AddRef();
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordDrawIndexedPrimitive(D3DPRIMITIVETYPE type, INT baseVertex,
                                                    UINT minIdx, UINT numVerts, UINT startIdx, UINT primCount) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_DrawIndexedPrimitive;
    cmd.dip.primType = type;
    cmd.dip.baseVertex = baseVertex;
    cmd.dip.minIndex = minIdx;
    cmd.dip.numVerts = numVerts;
    cmd.dip.startIndex = startIdx;
    cmd.dip.primCount = primCount;
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordDrawPrimitive(D3DPRIMITIVETYPE type, UINT startVertex, UINT primCount) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_DrawPrimitive;
    cmd.dp.primType = type;
    cmd.dp.startVertex = startVertex;
    cmd.dp.primCount = primCount;
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordClear(DWORD count, DWORD flags, D3DCOLOR color, float z, DWORD stencil) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_Clear;
    cmd.clear.count = count;
    cmd.clear.flags = flags;
    cmd.clear.color = color;
    cmd.clear.z = z;
    cmd.clear.stencil = stencil;
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetRenderTarget(IDirect3DSurface9* surface) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetRenderTarget;
    cmd.rt.surface = surface;
    if (surface) surface->AddRef();
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetDepthStencilSurface(IDirect3DSurface9* surface) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetDepthStencilSurface;
    cmd.ds.surface = surface;
    if (surface) surface->AddRef();
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordBeginScene() {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_BeginScene;
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordEndScene() {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_EndScene;
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetViewport(const D3DVIEWPORT9* vp) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetViewport;
    cmd.vp.viewport = *vp;
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetVertexShader(IDirect3DVertexShader9* shader) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetVertexShader;
    cmd.vs.shader = shader;
    if (shader) shader->AddRef();
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetPixelShader(IDirect3DPixelShader9* shader) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetPixelShader;
    cmd.ps.shader = shader;
    if (shader) shader->AddRef();
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetVSConstantF(UINT startReg, const float* data, UINT count) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetVSConstantF;
    cmd.constF.startReg = startReg;
    cmd.constF.count = count;
    cmd.constF.arenaOffset = (UINT)constantArena.size();
    UINT numFloats = count * 4;
    constantArena.insert(constantArena.end(), data, data + numFloats);
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetPSConstantF(UINT startReg, const float* data, UINT count) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetPSConstantF;
    cmd.constF.startReg = startReg;
    cmd.constF.count = count;
    cmd.constF.arenaOffset = (UINT)constantArena.size();
    UINT numFloats = count * 4;
    constantArena.insert(constantArena.end(), data, data + numFloats);
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetVSConstantI(UINT startReg, const int* data, UINT count) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetVSConstantI;
    cmd.constF.startReg = startReg;
    cmd.constF.count = count;
    cmd.constF.arenaOffset = (UINT)constantArena.size();
    UINT numFloats = count * 4;  // reuse float arena, reinterpreted as int
    const float* asFloat = reinterpret_cast<const float*>(data);
    constantArena.insert(constantArena.end(), asFloat, asFloat + numFloats);
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordSetPSConstantI(UINT startReg, const int* data, UINT count) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_SetPSConstantI;
    cmd.constF.startReg = startReg;
    cmd.constF.count = count;
    cmd.constF.arenaOffset = (UINT)constantArena.size();
    UINT numFloats = count * 4;  // reuse float arena, reinterpreted as int
    const float* asFloat = reinterpret_cast<const float*>(data);
    constantArena.insert(constantArena.end(), asFloat, asFloat + numFloats);
    commands.push_back(cmd);
}

void D3DCommandBuffer::recordGetRenderTargetData(IDirect3DSurface9* source, IDirect3DSurface9* dest) {
    D3DCmd cmd;
    cmd.type = D3DCmd::Cmd_GetRenderTargetData;
    cmd.rtData.source = source;
    cmd.rtData.dest = dest;
    if (source) source->AddRef();
    if (dest) dest->AddRef();
    commands.push_back(cmd);
}

// --- Replay ---

void D3DCommandBuffer::replay(IDirect3DDevice9* device) const {
    for (const auto& cmd : commands) {
        switch (cmd.type) {
        case D3DCmd::Cmd_SetRenderState:
            device->SetRenderState((D3DRENDERSTATETYPE)cmd.rs.state, cmd.rs.value);
            break;
        case D3DCmd::Cmd_SetTextureStageState:
            device->SetTextureStageState(cmd.tss.stage, (D3DTEXTURESTAGESTATETYPE)cmd.tss.state, cmd.tss.value);
            break;
        case D3DCmd::Cmd_SetSamplerState:
            device->SetSamplerState(cmd.ss.sampler, (D3DSAMPLERSTATETYPE)cmd.ss.state, cmd.ss.value);
            break;
        case D3DCmd::Cmd_SetTransform:
            device->SetTransform((D3DTRANSFORMSTATETYPE)cmd.xform.transformType, &cmd.xform.matrix);
            break;
        case D3DCmd::Cmd_SetTexture:
            device->SetTexture(cmd.tex.stage, cmd.tex.tex);
            break;
        case D3DCmd::Cmd_SetMaterial:
            device->SetMaterial(&cmd.mat.material);
            break;
        case D3DCmd::Cmd_SetLight:
            device->SetLight(cmd.light.index, &cmd.light.light);
            break;
        case D3DCmd::Cmd_LightEnable:
            device->LightEnable(cmd.lightEn.index, cmd.lightEn.enable);
            break;
        case D3DCmd::Cmd_SetFVF:
            device->SetFVF(cmd.fvf.fvf);
            break;
        case D3DCmd::Cmd_SetStreamSource:
            device->SetStreamSource(cmd.streamSrc.stream, cmd.streamSrc.vb, cmd.streamSrc.offset, cmd.streamSrc.stride);
            break;
        case D3DCmd::Cmd_SetIndices:
            device->SetIndices(cmd.indices.ib);
            break;
        case D3DCmd::Cmd_DrawIndexedPrimitive:
            device->DrawIndexedPrimitive(cmd.dip.primType, cmd.dip.baseVertex,
                cmd.dip.minIndex, cmd.dip.numVerts, cmd.dip.startIndex, cmd.dip.primCount);
            break;
        case D3DCmd::Cmd_DrawPrimitive:
            device->DrawPrimitive(cmd.dp.primType, cmd.dp.startVertex, cmd.dp.primCount);
            break;
        case D3DCmd::Cmd_Clear:
            device->Clear(cmd.clear.count, NULL,
                cmd.clear.flags, cmd.clear.color, cmd.clear.z, cmd.clear.stencil);
            break;
        case D3DCmd::Cmd_SetRenderTarget:
            device->SetRenderTarget(0, cmd.rt.surface);
            break;
        case D3DCmd::Cmd_SetDepthStencilSurface:
            device->SetDepthStencilSurface(cmd.ds.surface);
            break;
        case D3DCmd::Cmd_BeginScene:
            device->BeginScene();
            break;
        case D3DCmd::Cmd_EndScene:
            device->EndScene();
            break;
        case D3DCmd::Cmd_SetViewport:
            device->SetViewport(&cmd.vp.viewport);
            break;
        case D3DCmd::Cmd_SetVertexShader:
            device->SetVertexShader(cmd.vs.shader);
            break;
        case D3DCmd::Cmd_SetPixelShader:
            device->SetPixelShader(cmd.ps.shader);
            break;
        case D3DCmd::Cmd_SetVSConstantF:
            device->SetVertexShaderConstantF(cmd.constF.startReg,
                &constantArena[cmd.constF.arenaOffset], cmd.constF.count);
            break;
        case D3DCmd::Cmd_SetPSConstantF:
            device->SetPixelShaderConstantF(cmd.constF.startReg,
                &constantArena[cmd.constF.arenaOffset], cmd.constF.count);
            break;
        case D3DCmd::Cmd_SetVSConstantI:
            device->SetVertexShaderConstantI(cmd.constF.startReg,
                reinterpret_cast<const int*>(&constantArena[cmd.constF.arenaOffset]), cmd.constF.count);
            break;
        case D3DCmd::Cmd_SetPSConstantI:
            device->SetPixelShaderConstantI(cmd.constF.startReg,
                reinterpret_cast<const int*>(&constantArena[cmd.constF.arenaOffset]), cmd.constF.count);
            break;
        case D3DCmd::Cmd_GetRenderTargetData:
            device->GetRenderTargetData(cmd.rtData.source, cmd.rtData.dest);
            break;
        }
    }
}

// Replay only state commands (skip draw calls) — for restoring MW device state
void D3DCommandBuffer::replayStateOnly(IDirect3DDevice9* device) const {
    for (const auto& cmd : commands) {
        switch (cmd.type) {
        // State commands — replay these
        case D3DCmd::Cmd_SetRenderState:
            device->SetRenderState((D3DRENDERSTATETYPE)cmd.rs.state, cmd.rs.value);
            break;
        case D3DCmd::Cmd_SetTextureStageState:
            device->SetTextureStageState(cmd.tss.stage, (D3DTEXTURESTAGESTATETYPE)cmd.tss.state, cmd.tss.value);
            break;
        case D3DCmd::Cmd_SetSamplerState:
            device->SetSamplerState(cmd.ss.sampler, (D3DSAMPLERSTATETYPE)cmd.ss.state, cmd.ss.value);
            break;
        case D3DCmd::Cmd_SetTransform:
            device->SetTransform((D3DTRANSFORMSTATETYPE)cmd.xform.transformType, &cmd.xform.matrix);
            break;
        case D3DCmd::Cmd_SetTexture:
            device->SetTexture(cmd.tex.stage, cmd.tex.tex);
            break;
        case D3DCmd::Cmd_SetMaterial:
            device->SetMaterial(&cmd.mat.material);
            break;
        case D3DCmd::Cmd_SetLight:
            device->SetLight(cmd.light.index, &cmd.light.light);
            break;
        case D3DCmd::Cmd_LightEnable:
            device->LightEnable(cmd.lightEn.index, cmd.lightEn.enable);
            break;
        case D3DCmd::Cmd_SetFVF:
            device->SetFVF(cmd.fvf.fvf);
            break;
        case D3DCmd::Cmd_SetStreamSource:
            device->SetStreamSource(cmd.streamSrc.stream, cmd.streamSrc.vb, cmd.streamSrc.offset, cmd.streamSrc.stride);
            break;
        case D3DCmd::Cmd_SetIndices:
            device->SetIndices(cmd.indices.ib);
            break;
        case D3DCmd::Cmd_SetViewport:
            device->SetViewport(&cmd.vp.viewport);
            break;

        // Skip draw commands
        case D3DCmd::Cmd_DrawIndexedPrimitive:
        case D3DCmd::Cmd_DrawPrimitive:
            break;

        // Skip scene/clear/RT commands — caller manages these
        case D3DCmd::Cmd_Clear:
        case D3DCmd::Cmd_SetRenderTarget:
        case D3DCmd::Cmd_SetDepthStencilSurface:
        case D3DCmd::Cmd_BeginScene:
        case D3DCmd::Cmd_EndScene:
            break;

        // Skip shader commands — HLSL-specific, not MW state
        case D3DCmd::Cmd_SetVertexShader:
        case D3DCmd::Cmd_SetPixelShader:
        case D3DCmd::Cmd_SetVSConstantF:
        case D3DCmd::Cmd_SetPSConstantF:
        case D3DCmd::Cmd_SetVSConstantI:
        case D3DCmd::Cmd_SetPSConstantI:
        case D3DCmd::Cmd_GetRenderTargetData:
            break;
        }
    }
}

// --- CmdStage names ---

const char* CmdStageName(CmdStage s) {
    static const char* names[] = { "Offscreen", "PreScene", "Scene0", "InterScene", "Scene1Plus", "UI" };
    return ((int)s < (int)CmdStage::Count) ? names[(int)s] : "Unknown";
}

// --- D3DCommandBufferSet ---

void D3DCommandBufferSet::clearAll() {
    for (int i = 0; i < (int)CmdStage::Count; i++) {
        buffers[i].clear();
    }
    tracker.clear();
    activeStage = CmdStage::PreScene;
}

int D3DCommandBufferSet::totalSize() const {
    int total = 0;
    for (int i = 0; i < (int)CmdStage::Count; i++) {
        total += buffers[i].size();
    }
    return total;
}

size_t D3DCommandBufferSet::totalSizeBytes() const {
    size_t total = 0;
    for (int i = 0; i < (int)CmdStage::Count; i++) {
        total += buffers[i].sizeBytes();
    }
    return total;
}

void D3DCommandBufferSet::dumpToFrameLog() const {
    for (int i = 0; i < (int)CmdStage::Count; i++) {
        if (buffers[i].size() > 0) {
            buffers[i].dumpToFrameLog(i);  // use stage index as sceneNum for identification
        }
    }
}

// --- D3DCommandBuffer dump ---

void D3DCommandBuffer::dumpToFrameLog(int sceneNum) const {
    for (const auto& cmd : commands) {
        switch (cmd.type) {
        case D3DCmd::Cmd_Clear: {
            StateDetail d;
            d.kind = StateDetail::ClearCall;
            d.clear.flags = cmd.clear.flags;
            d.clear.color = cmd.clear.color;
            d.clear.z = cmd.clear.z;
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_Clear, sceneNum, 0, d);
            break;
        }
        case D3DCmd::Cmd_BeginScene:
            ImGuiManager::LogFrameEvent(FrameEvent::Replay_BeginScene, sceneNum);
            break;
        case D3DCmd::Cmd_EndScene:
            ImGuiManager::LogFrameEvent(FrameEvent::Replay_EndScene, sceneNum);
            break;
        case D3DCmd::Cmd_SetRenderTarget: {
            StateDetail d;
            d.kind = StateDetail::RenderTarget;
            d.rt.color = (uintptr_t)cmd.rt.surface;
            d.rt.depth = 0;
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_SetRT, sceneNum, 0, d);
            break;
        }
        case D3DCmd::Cmd_SetDepthStencilSurface: {
            StateDetail d;
            d.kind = StateDetail::RenderTarget;
            d.rt.color = 0;
            d.rt.depth = (uintptr_t)cmd.ds.surface;
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_SetDS, sceneNum, 0, d);
            break;
        }
        case D3DCmd::Cmd_SetViewport: {
            StateDetail d;
            d.kind = StateDetail::Viewport;
            d.vp.x = cmd.vp.viewport.X;
            d.vp.y = cmd.vp.viewport.Y;
            d.vp.w = cmd.vp.viewport.Width;
            d.vp.h = cmd.vp.viewport.Height;
            d.vp.minZ = cmd.vp.viewport.MinZ;
            d.vp.maxZ = cmd.vp.viewport.MaxZ;
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_SetViewport, sceneNum, 0, d);
            break;
        }
        case D3DCmd::Cmd_SetRenderState: {
            StateDetail d;
            d.kind = StateDetail::RenderState;
            d.rs.state = cmd.rs.state;
            d.rs.value = cmd.rs.value;
            d.rs.prev = 0;
            d.rs.changed = true;
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_RS, sceneNum, 0, d);
            break;
        }
        case D3DCmd::Cmd_SetTextureStageState: {
            StateDetail d;
            d.kind = StateDetail::TextureStageState;
            d.tss.stage = cmd.tss.stage;
            d.tss.state = cmd.tss.state;
            d.tss.value = cmd.tss.value;
            d.tss.prev = 0;
            d.tss.changed = true;
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_TSS, sceneNum, 0, d);
            break;
        }
        case D3DCmd::Cmd_SetSamplerState: {
            StateDetail d;
            d.kind = StateDetail::None;
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_SS, sceneNum, cmd.ss.state, d);
            break;
        }
        case D3DCmd::Cmd_SetTransform: {
            StateDetail d;
            d.kind = StateDetail::Transform;
            d.xform.type = cmd.xform.transformType;
            memcpy(d.xform.m, &cmd.xform.matrix, 16 * sizeof(float));
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_Transform, sceneNum, 0, d);
            break;
        }
        case D3DCmd::Cmd_SetTexture: {
            StateDetail d;
            d.kind = StateDetail::Texture;
            d.tex.stage = cmd.tex.stage;
            d.tex.ptr = (uintptr_t)cmd.tex.tex;
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_Texture, sceneNum, 0, d);
            break;
        }
        case D3DCmd::Cmd_SetMaterial: {
            StateDetail d;
            d.kind = StateDetail::Material;
            d.mat.dr = cmd.mat.material.Diffuse.r;
            d.mat.dg = cmd.mat.material.Diffuse.g;
            d.mat.db = cmd.mat.material.Diffuse.b;
            d.mat.da = cmd.mat.material.Diffuse.a;
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_Material, sceneNum, 0, d);
            break;
        }
        case D3DCmd::Cmd_SetLight:
            ImGuiManager::LogFrameEvent(FrameEvent::Replay_Light, sceneNum, cmd.light.index);
            break;
        case D3DCmd::Cmd_LightEnable: {
            StateDetail d;
            d.kind = StateDetail::Light;
            d.light.index = cmd.lightEn.index;
            d.light.enable = cmd.lightEn.enable;
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_LightEnable, sceneNum, 0, d);
            break;
        }
        case D3DCmd::Cmd_SetFVF: {
            StateDetail d;
            d.kind = StateDetail::VertexShader;
            d.vs.fvf = cmd.fvf.fvf;
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_FVF, sceneNum, 0, d);
            break;
        }
        case D3DCmd::Cmd_SetStreamSource: {
            StateDetail d;
            d.kind = StateDetail::StreamSource;
            d.ss.stream = cmd.streamSrc.stream;
            d.ss.vb = (uintptr_t)cmd.streamSrc.vb;
            d.ss.stride = cmd.streamSrc.stride;
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_StreamSource, sceneNum, 0, d);
            break;
        }
        case D3DCmd::Cmd_SetIndices: {
            StateDetail d;
            d.kind = StateDetail::IndexBuffer;
            d.ib.ib = (uintptr_t)cmd.indices.ib;
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_Indices, sceneNum, 0, d);
            break;
        }
        case D3DCmd::Cmd_DrawIndexedPrimitive: {
            StateDetail d;
            d.kind = StateDetail::DrawCall;
            d.dip.primCount = cmd.dip.primCount;
            d.dip.vertCount = cmd.dip.numVerts;
            d.dip.fvf = 0;
            d.dip.vb = 0;
            d.dip.ib = 0;
            d.dip.tex0 = 0;
            d.dip.zWrite = 0;
            d.dip.cull = 0;
            d.dip.alphaBlend = 0;
            d.dip.alphaTest = 0;
            d.dip.srcBlend = 0;
            d.dip.destBlend = 0;
            d.dip.vertBlend = 0;
            ImGuiManager::LogFrameEventDetailed(FrameEvent::Replay_DIP, sceneNum, cmd.dip.primCount, d);
            break;
        }
        case D3DCmd::Cmd_DrawPrimitive:
            ImGuiManager::LogFrameEvent(FrameEvent::Replay_DP, sceneNum, cmd.dp.primCount);
            break;
        case D3DCmd::Cmd_SetVertexShader:
        case D3DCmd::Cmd_SetPixelShader:
        case D3DCmd::Cmd_SetVSConstantF:
        case D3DCmd::Cmd_SetPSConstantF:
        case D3DCmd::Cmd_SetVSConstantI:
        case D3DCmd::Cmd_SetPSConstantI:
        case D3DCmd::Cmd_GetRenderTargetData:
            break;
        }
    }
}
