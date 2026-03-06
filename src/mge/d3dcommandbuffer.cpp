#include "d3dcommandbuffer.h"

// --- Move semantics ---

D3DCommandBuffer::D3DCommandBuffer(D3DCommandBuffer&& other) noexcept
    : commands(std::move(other.commands)) {
}

D3DCommandBuffer& D3DCommandBuffer::operator=(D3DCommandBuffer&& other) noexcept {
    if (this != &other) {
        clear();
        commands = std::move(other.commands);
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
        default:
            break;
        }
    }
    commands.clear();
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
            device->Clear(cmd.clear.count, NULL, cmd.clear.flags, cmd.clear.color, cmd.clear.z, cmd.clear.stencil);
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
        }
    }
}
