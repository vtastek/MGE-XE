
#include "configuration.h"
#include "distantland.h"
#include "distantshader.h"
#include "mwbridge.h"
#include "proxydx/d3d8header.h"
#include "support/log.h"

#include <algorithm>
#include <cmath>



void DistantLand::renderDepth() {
    auto mwBridge = MWBridge::get();

    // Switch to render target
    RenderTargetSwitcher rtsw(surfDepthFrameMSAA, surfDepthDepth);
    device->Clear(0, 0, D3DCLEAR_ZBUFFER, 0, 1.0, 0);

    // Unbind depth sampler
    effect->SetTexture(ehTex3, NULL);

    // Projection should match main rendering
    // For Morrowind geometry, use Morrowind's native near plane to match recording
    D3DXMATRIX mwDepthProj = mwProj;
    // Morrowind uses near=1.0, but we extend far plane for better depth precision
    if (Configuration.MGEFlags & USE_DISTANT_LAND) {
        editProjectionZ(&mwDepthProj, 1.0f, Configuration.DL.DrawDist * kCellSize);
    }
    effect->SetMatrix(ehProj, &mwDepthProj);

    // Clear floating point buffer to far depth
    effectDepth->BeginPass(PASS_CLEARDEPTH);
    device->SetVertexDeclaration(WaterDecl);
    device->SetStreamSource(0, vbFullFrame, 0, 12);
    device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);
    effectDepth->EndPass();

    // Recorded draw calls with Morrowind near plane
    effectDepth->BeginPass(PASS_RENDERMWDEPTH);
    renderDepthRecorded();
    effectDepth->EndPass();

    // Copy recordMW depth to texCullDepth for Hi-Z culling (before distant land)
    if (texCullDepth) {
        IDirect3DSurface9* cullDepthSurface;
        texCullDepth->GetSurfaceLevel(0, &cullDepthSurface);
        device->StretchRect(surfDepthFrameMSAA, NULL, cullDepthSurface, NULL, D3DTEXF_NONE);
        cullDepthSurface->Release();
    }

    if (isDistantCell()) {
        if (!mwBridge->IsUnderwater(eyePos.z)) {
            // Distant land
            if (mwBridge->IsExterior()) {
                effectDepth->BeginPass(PASS_RENDERLANDDEPTH);
                renderDistantLandZ();
                effectDepth->EndPass();
            }

            // Distant statics
            effectDepth->BeginPass(PASS_RENDERSTATICSDEPTH);
            device->SetVertexDeclaration(StaticDecl);
            if (Configuration.UseSharedMemory) {
                visDistantShared.Render(device, effectDepth, effect, &ehTex0, &ehHasAlpha, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT);
            } else {
                visDistant.Render(device, effectDepth, effect, &ehTex0, &ehHasAlpha, &ehHasVCol, &ehWorld, SIZEOFSTATICVERT);
            }
            effectDepth->EndPass();
        }

        if (Configuration.MGEFlags & USE_GRASS) {
            // Grass
            effectDepth->BeginPass(PASS_RENDERGRASSDEPTHINST);
            renderGrassInstZ();
            effectDepth->EndPass();
        }
    }

    // Reset projection matrix
    effect->SetMatrix(ehProj, &mwProj);
}

void DistantLand::renderDepthAdditional() {
    // Switch to render target
    RenderTargetSwitcher rtsw(surfDepthFrameMSAA, surfDepthDepth);

    // Unbind depth sampler
    effect->SetTexture(ehTex3, NULL);

    // Projection should match main rendering
    // For Morrowind geometry, use Morrowind's native near plane to match recording
    D3DXMATRIX mwDepthProj = mwProj;
    if (Configuration.MGEFlags & USE_DISTANT_LAND) {
        editProjectionZ(&mwDepthProj, 1.0f, Configuration.DL.DrawDist * kCellSize);
    }
    effect->SetMatrix(ehProj, &mwDepthProj);

    // Recorded draw calls with Morrowind near plane
    effectDepth->BeginPass(PASS_RENDERMWDEPTH);
    renderDepthRecorded();
    effectDepth->EndPass();

    // Reset projection matrix
    effect->SetMatrix(ehProj, &mwProj);
}

void DistantLand::renderDepthRecorded() {
    // Use an alpha threshold for solidity that isn't precisely equal to a commonly used value (such as 0.5).
    // Vertex interpolators can be slightly inaccurate and cause a value that should be constant across a triangle
    // to have interpolated fragment values that vary either side of the threshold and cause noise.
    const float solidThreshold = 0.499f;

    // Recorded renders
    const auto& recordMW_const = recordMW;
    for (const auto& i : recordMW_const) {
        // Set variables in main effect; variables are shared via effect pool

        // Fragment colour routing
        bool alphaDependent = i.alphaTest || i.blendEnable;
        effect->SetBool(ehHasVCol, alphaDependent && (i.fvf & D3DFVF_DIFFUSE) != 0);
        effect->SetFloat(ehMaterialAlpha, alphaDependent ? i.diffuseMaterial.a : 1.0f);

        // Only bind texture for alphas
        if (alphaDependent && i.texture) {
            effect->SetTexture(ehTex0, i.texture);
            effect->SetBool(ehHasAlpha, true);
            effect->SetFloat(ehAlphaRef, i.alphaTest ? (i.alphaRef / 255.0f) : solidThreshold);
        } else {
            effect->SetTexture(ehTex0, 0);
            effect->SetBool(ehHasAlpha, false);
            effect->SetFloat(ehAlphaRef, -1.0f);
        }

        // Skin using worldview matrices for numerical accuracy
        effect->SetBool(ehHasBones, i.vertexBlendState != 0);
        effect->SetInt(ehVertexBlendState, i.vertexBlendState);

        // Phase A: Fix vertex animation timing - recalculate transforms with current view matrix
        if (i.vertexBlendState > 0) {
            // For skinned objects, recombine recorded world matrices with current view matrix
            // i.worldViewTransforms contains old view matrix data, causing animation mismatch
            D3DXMATRIX currentView, currentWorldViewTransforms[4];
            device->GetTransform(D3DTS_VIEW, &currentView);
            for (int j = 0; j < 4; j++) {
                currentWorldViewTransforms[j] = i.worldTransforms[j] * currentView;
            }
            effect->SetMatrixArray(ehVertexBlendPalette, currentWorldViewTransforms, 4);
        } else {
            effect->SetMatrixArray(ehVertexBlendPalette, i.worldViewTransforms, 4);
        }
        effectDepth->CommitChanges();

        // Phase A: Set render states to match HLSL rendering exactly
        // For alpha blended objects, disable backface culling to show both sides
        // In depth-only rendering, both front and back faces need proper depth values
        if (i.blendEnable) {
            device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
        } else {
            device->SetRenderState(D3DRS_CULLMODE, i.cullMode);
        }
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, i.blendEnable);
        if (i.blendEnable) {
            device->SetRenderState(D3DRS_SRCBLEND, i.srcBlend);
            device->SetRenderState(D3DRS_DESTBLEND, i.destBlend);
        }

        // Phase A: Set alpha test for depth-only rendering to match color rendering
        device->SetRenderState(D3DRS_ALPHATESTENABLE, i.alphaTest);
        if (i.alphaTest) {
            device->SetRenderState(D3DRS_ALPHAREF, i.alphaRef);
            device->SetRenderState(D3DRS_ALPHAFUNC, i.alphaFunc);
        }

        device->SetStreamSource(0, i.vb, i.vbOffset, i.vbStride);
        device->SetIndices(i.ib);
        device->SetFVF(i.fvf);
        device->DrawIndexedPrimitive(i.primType, i.baseIndex, i.minIndex, i.vertCount, i.startIndex, i.primCount);
    }
}

void DistantLand::generateHiZPyramid() {
    if (!texHiZ || !effectHiZ) {
        LOG::logline("Hi-Z: Skipping generation - texHiZ=%p effectHiZ=%p", texHiZ, effectHiZ);
        return;
    }

    LOG::logline("Hi-Z: Generating pyramid with %d mip levels (optimized mip chain method)", hiZLevels);

    // Save current render targets explicitly
    IDirect3DSurface9* savedRT0;
    IDirect3DSurface9* savedDepthStencil;
    device->GetRenderTarget(0, &savedRT0);
    device->GetDepthStencilSurface(&savedDepthStencil);

    // Step 1: Compile shaders for MAX downsampling
    IDirect3DSurface9* dstSurf;
    D3DXHANDLE hTechnique = effectHiZ->GetTechniqueByName("T0");
    D3DXHANDLE hPass = effectHiZ->GetPass(hTechnique, 0);

    D3DXPASS_DESC passDesc;
    effectHiZ->GetPassDesc(hPass, &passDesc);

    // Get vertex and pixel shaders from the pass
    IDirect3DVertexShader9* vs = nullptr;
    IDirect3DPixelShader9* ps = nullptr;

    // Compile shaders from effect
    ID3DXBuffer* vsCode = nullptr;
    ID3DXBuffer* psCode = nullptr;
    ID3DXBuffer* vsErrors = nullptr;
    ID3DXBuffer* psErrors = nullptr;

    const char* vsSource =
        "float4 main(float4 pos : POSITION, out float2 oTex : TEXCOORD0) : POSITION {\n"
        "    oTex = float2(pos.x * 0.5 + 0.5, -pos.y * 0.5 + 0.5);\n"
        "    return pos;\n"
        "}\n";

    HRESULT hrVS = D3DXCompileShader(
        vsSource, strlen(vsSource),
        NULL, NULL, "main", "vs_3_0", 0, &vsCode, &vsErrors, NULL
    );

    if (FAILED(hrVS) && vsErrors) {
        LOG::logline("!! Hi-Z VS compile error: %s", (char*)vsErrors->GetBufferPointer());
        vsErrors->Release();
    }

    const char* psSource =
        "sampler2D sampDepth : register(s0);\n"
        "float4 texelSize : register(c0);\n"
        "float4 main(float2 tex : TEXCOORD0) : COLOR0 {\n"
        "    float d0 = tex2D(sampDepth, tex).r;\n"
        "    float d1 = tex2D(sampDepth, tex + float2(texelSize.x, 0)).r;\n"
        "    float d2 = tex2D(sampDepth, tex + float2(0, texelSize.y)).r;\n"
        "    float d3 = tex2D(sampDepth, tex + texelSize.xy).r;\n"
        "    float maxDepth = max(max(d0, d1), max(d2, d3));\n"
        "    return float4(maxDepth, maxDepth, maxDepth, 1.0);\n"
        "}\n";

    HRESULT hrPS = D3DXCompileShader(psSource, strlen(psSource), NULL, NULL, "main", "ps_3_0", 0, &psCode, &psErrors, NULL);

    if (FAILED(hrPS) && psErrors) {
        LOG::logline("!! Hi-Z PS compile error: %s", (char*)psErrors->GetBufferPointer());
        psErrors->Release();
    }

    LOG::logline("Hi-Z shader compilation: VS=%s PS=%s", SUCCEEDED(hrVS) ? "OK" : "FAIL", SUCCEEDED(hrPS) ? "OK" : "FAIL");

    if (vsCode) device->CreateVertexShader((DWORD*)vsCode->GetBufferPointer(), &vs);
    if (psCode) device->CreatePixelShader((DWORD*)psCode->GetBufferPointer(), &ps);

    if (vsCode) vsCode->Release();
    if (psCode) psCode->Release();

    if (!vs || !ps) {
        LOG::logline("!! Hi-Z: Failed to compile shaders");
        if (vs) vs->Release();
        if (ps) ps->Release();
        device->SetRenderTarget(0, savedRT0);
        device->SetDepthStencilSurface(savedDepthStencil);
        savedRT0->Release();
        if (savedDepthStencil) savedDepthStencil->Release();
        return;
    }

    // Set shaders
    device->SetVertexShader(vs);
    device->SetPixelShader(ps);

    // Step 2: Generate all mip levels using MAX downsampling shader
    for (int mipLevel = 0; mipLevel < hiZLevels; mipLevel++) {
        // Get dimensions of destination mip level
        D3DSURFACE_DESC dstDesc;
        texHiZ->GetLevelDesc(mipLevel, &dstDesc);

        // Set render target to current mip level
        texHiZ->GetSurfaceLevel(mipLevel, &dstSurf);
        device->SetRenderTarget(0, dstSurf);
        device->SetDepthStencilSurface(NULL);

        // Set viewport to match the current mip size
        D3DVIEWPORT9 vp;
        vp.X = 0;
        vp.Y = 0;
        vp.Width = dstDesc.Width;
        vp.Height = dstDesc.Height;
        vp.MinZ = 0.0f;
        vp.MaxZ = 1.0f;
        device->SetViewport(&vp);

        // Bind input texture: texCullDepth for mip 0, previous mip level for others
        if (mipLevel == 0) {
            device->SetTexture(0, texCullDepth);
        } else {
            device->SetTexture(0, texHiZ);
            // Force sampling from previous mip level only
            device->SetSamplerState(0, D3DSAMP_MAXMIPLEVEL, mipLevel - 1);
        }

        // Configure sampler with POINT filtering
        device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
        device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
        device->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
        device->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
        device->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);

        // Set texel size as pixel shader constant (c0)
        // Use current output size (not previous) to map output pixels to 2x2 input regions
        float texelSize[4] = { 1.0f / dstDesc.Width, 1.0f / dstDesc.Height, 0.0f, 0.0f };
        device->SetPixelShaderConstantF(0, texelSize, 1);

        // Set render states
        device->SetRenderState(D3DRS_ZENABLE, FALSE);
        device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
        device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);

        // Render full-screen quad with MAX downsample shader
        device->SetVertexDeclaration(WaterDecl);
        device->SetStreamSource(0, vbFullFrame, 0, 12);
        device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);

        dstSurf->Release();

        LOG::logline("Hi-Z mip %d: %dx%d, texelSize=(%.6f, %.6f)", mipLevel, dstDesc.Width, dstDesc.Height, texelSize[0], texelSize[1]);
    }

    // Reset sampler state
    device->SetSamplerState(0, D3DSAMP_MAXMIPLEVEL, 0);

    // Cleanup
    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);
    vs->Release();
    ps->Release();

    // Restore render targets explicitly
    device->SetRenderTarget(0, savedRT0);
    device->SetDepthStencilSurface(savedDepthStencil);
    savedRT0->Release();
    if (savedDepthStencil) savedDepthStencil->Release();

    // Step 3: Copy entire Hi-Z mip chain to staging texture for CPU readback
    for (int mipLevel = 0; mipLevel < hiZLevels; mipLevel++) {
        IDirect3DSurface9* srcSurf = nullptr;
        IDirect3DSurface9* dstSurf = nullptr;

        texHiZ->GetSurfaceLevel(mipLevel, &srcSurf);
        texHiZStaging->GetSurfaceLevel(mipLevel, &dstSurf);
        HRESULT hr = device->GetRenderTargetData(srcSurf, dstSurf);
        srcSurf->Release();
        dstSurf->Release();

        if (FAILED(hr)) {
            static bool loggedOnce = false;
            if (!loggedOnce) {
                LOG::logline("!! Hi-Z: GetRenderTargetData failed for mip %d", mipLevel);
                loggedOnce = true;
            }
            return;
        }
    }

    // Force GPU to complete all operations before staging texture is accessed
    // This prevents heap corruption when cullAgainstHiZ() locks the staging texture
    IDirect3DQuery9* eventQuery = nullptr;
    if (SUCCEEDED(device->CreateQuery(D3DQUERYTYPE_EVENT, &eventQuery))) {
        eventQuery->Issue(D3DISSUE_END);
        // Wait for GPU to complete
        while (eventQuery->GetData(nullptr, 0, D3DGETDATA_FLUSH) == S_FALSE) {
            // Spin-wait for GPU completion
        }
        eventQuery->Release();
    }

    // Debug: Sample a few depth values from mip 0 to verify Hi-Z content
    static bool loggedOnce = false;
    if (!loggedOnce && texHiZStaging) {
        D3DLOCKED_RECT lr;
        if (SUCCEEDED(texHiZStaging->LockRect(0, &lr, NULL, D3DLOCK_READONLY))) {
            D3DSURFACE_DESC desc;
            texHiZStaging->GetLevelDesc(0, &desc);
            float* data = (float*)lr.pBits;
            int stride = lr.Pitch / sizeof(float);

            // Sample center and corners
            float center = data[(desc.Height/2) * stride + (desc.Width/2)];
            float tl = data[0];
            float tr = data[desc.Width-1];
            float bl = data[(desc.Height-1) * stride];
            float br = data[(desc.Height-1) * stride + (desc.Width-1)];

            LOG::logline("Hi-Z Mip0 samples: center=%.2f, TL=%.2f, TR=%.2f, BL=%.2f, BR=%.2f", center, tl, tr, bl, br);
            texHiZStaging->UnlockRect(0);
            loggedOnce = true;
        }
    }
}

bool DistantLand::cullAgainstHiZ(const D3DXVECTOR3& bboxMin, const D3DXVECTOR3& bboxMax, const D3DXMATRIX& worldViewProj, bool debugLog) {
    if (!texHiZStaging) return false;

    // Transform bounding box corners to clip space
    D3DXVECTOR3 corners[8] = {
        D3DXVECTOR3(bboxMin.x, bboxMin.y, bboxMin.z),
        D3DXVECTOR3(bboxMax.x, bboxMin.y, bboxMin.z),
        D3DXVECTOR3(bboxMin.x, bboxMax.y, bboxMin.z),
        D3DXVECTOR3(bboxMax.x, bboxMax.y, bboxMin.z),
        D3DXVECTOR3(bboxMin.x, bboxMin.y, bboxMax.z),
        D3DXVECTOR3(bboxMax.x, bboxMin.y, bboxMax.z),
        D3DXVECTOR3(bboxMin.x, bboxMax.y, bboxMax.z),
        D3DXVECTOR3(bboxMax.x, bboxMax.y, bboxMax.z)
    };

    // Transform corners and find screen space bounds
    float minX = 1e10f, maxX = -1e10f;
    float minY = 1e10f, maxY = -1e10f;
    float minLinearDepth = 1e10f;
    float maxLinearDepth = -1e10f;
    bool anyInFront = false;
    bool anyBehindCamera = false;
    float cornerDepths[8];  // Store all corner depths for detailed logging

    for (int i = 0; i < 8; i++) {
        D3DXVECTOR4 clipPos;
        D3DXVec3Transform(&clipPos, &corners[i], &worldViewProj);

        if (clipPos.w > 0.0f) {
            anyInFront = true;
            float invW = 1.0f / clipPos.w;
            float ndcX = clipPos.x * invW;
            float ndcY = clipPos.y * invW;

            // Store linear depth (clipPos.w is view-space Z, which is linear depth)
            float linearDepth = clipPos.w;
            cornerDepths[i] = linearDepth;

            float screenX = ndcX * 0.5f + 0.5f;
            float screenY = -ndcY * 0.5f + 0.5f;

            minX = std::min(minX, screenX);
            maxX = std::max(maxX, screenX);
            minY = std::min(minY, screenY);
            maxY = std::max(maxY, screenY);
            minLinearDepth = std::min(minLinearDepth, linearDepth);
            maxLinearDepth = std::max(maxLinearDepth, linearDepth);
        } else {
            anyBehindCamera = true;
            cornerDepths[i] = -1.0f;  // Mark as behind camera
        }
    }

    if (!anyInFront) return false;

    // If any corners are behind the camera, object intersects near plane - never cull
    if (anyBehindCamera) {
        if (debugLog) {
            LOG::logline("Hi-Z Debug: Object intersects near plane (corners behind camera) - forced visible");
        }
        return false;
    }

    // Clamp to screen bounds
    minX = std::max(0.0f, std::min(1.0f, minX));
    maxX = std::max(0.0f, std::min(1.0f, maxX));
    minY = std::max(0.0f, std::min(1.0f, minY));
    maxY = std::max(0.0f, std::min(1.0f, maxY));

    if (maxX <= 0.0f || minX >= 1.0f || maxY <= 0.0f || minY >= 1.0f) {
        return false;
    }

    // Calculate bbox screen size and select appropriate mip level
    D3DSURFACE_DESC desc;
    texHiZStaging->GetLevelDesc(0, &desc);
    float boxWidth = (maxX - minX) * desc.Width;
    float boxHeight = (maxY - minY) * desc.Height;
    float boxSize = std::max(boxWidth, boxHeight);

    // Select mip level based on bbox screen size
    // TEMPORARY FIX: Use mip 0 (full resolution) always for accuracy
    // The mip selection was causing false positives by sampling too coarse a region
    // TODO: Implement proper hierarchical sampling with correct mip selection
    int mipLevel = 0;

    // Lock the staging texture at selected mip level
    D3DLOCKED_RECT lr;
    HRESULT hr = texHiZStaging->LockRect(mipLevel, &lr, NULL, D3DLOCK_READONLY);
    if (FAILED(hr)) {
        return false;
    }

    // Get mip level dimensions
    texHiZStaging->GetLevelDesc(mipLevel, &desc);

    // Calculate pixel range covered by bbox at this mip level
    int pixelMinX = (int)(minX * desc.Width);
    int pixelMaxX = (int)(maxX * desc.Width);
    int pixelMinY = (int)(minY * desc.Height);
    int pixelMaxY = (int)(maxY * desc.Height);

    // Clamp to texture bounds
    pixelMinX = std::max(0, std::min((int)desc.Width - 1, pixelMinX));
    pixelMaxX = std::max(0, std::min((int)desc.Width - 1, pixelMaxX));
    pixelMinY = std::max(0, std::min((int)desc.Height - 1, pixelMinY));
    pixelMaxY = std::max(0, std::min((int)desc.Height - 1, pixelMaxY));

    // Sample ALL Hi-Z pixels covered by bbox and find maximum depth
    float* depthData = (float*)lr.pBits;
    int pitch = lr.Pitch / sizeof(float);
    float maxHiZDepth = 0.0f;
    float minHiZDepth = 1e10f;
    int pixelCount = 0;

    for (int y = pixelMinY; y <= pixelMaxY; y++) {
        for (int x = pixelMinX; x <= pixelMaxX; x++) {
            float depth = depthData[y * pitch + x];
            maxHiZDepth = std::max(maxHiZDepth, depth);
            minHiZDepth = std::min(minHiZDepth, depth);
            pixelCount++;
        }
    }

    texHiZStaging->UnlockRect(mipLevel);

    // Proper Hi-Z occlusion test with intersection detection:
    // Hi-Z stores MAXIMUM depth (furthest visible point per pixel)
    // We found maxHiZDepth = furthest visible point in bbox's screen region
    //
    // Three cases:
    // 1. All corners behind Hi-Z (minLinearDepth > maxHiZDepth) → completely occluded, CULL
    // 2. All corners in front of Hi-Z (maxLinearDepth < maxHiZDepth) → completely visible, RENDER
    // 3. Some corners in front, some behind → INTERSECTING Hi-Z surface, RENDER (partially visible)
    //
    // So we only cull if minLinearDepth > maxHiZDepth (all corners behind)
    float bias = 10.0f; // Conservative bias for depth precision and bbox padding
    bool culled = minLinearDepth > maxHiZDepth + bias;

    if (debugLog && culled) {
        // Only log when object is CULLED (false positive candidates)
        LOG::logline("Hi-Z Debug CULLED: minDepth=%.4f, maxDepth=%.4f, maxHiZ=%.4f, mip=%d, region=[%d,%d]->[%d,%d]",
                     minLinearDepth, maxLinearDepth, maxHiZDepth, mipLevel, pixelMinX, pixelMinY, pixelMaxX, pixelMaxY);

        // Log all 8 corner depths for detailed analysis
        LOG::logline("  Corner depths: [0]=%.2f [1]=%.2f [2]=%.2f [3]=%.2f [4]=%.2f [5]=%.2f [6]=%.2f [7]=%.2f",
                     cornerDepths[0], cornerDepths[1], cornerDepths[2], cornerDepths[3],
                     cornerDepths[4], cornerDepths[5], cornerDepths[6], cornerDepths[7]);

        // Log Hi-Z depth range and pixel count
        LOG::logline("  Hi-Z depth range: min=%.4f max=%.4f (%d pixels sampled)", minHiZDepth, maxHiZDepth, pixelCount);

        // Log screen-space bounds
        LOG::logline("  Screen bounds: X=[%.3f,%.3f] Y=[%.3f,%.3f] size=%.1fx%.1f pixels",
                     minX, maxX, minY, maxY, boxWidth, boxHeight);

        // Log bbox world coordinates
        LOG::logline("  BBox: min=(%.1f,%.1f,%.1f) max=(%.1f,%.1f,%.1f)",
                     bboxMin.x, bboxMin.y, bboxMin.z, bboxMax.x, bboxMax.y, bboxMax.z);
    }

    return culled;
}
