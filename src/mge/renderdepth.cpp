
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

    // Removed verbose logging - Hi-Z pyramid generation is silent unless errors occur

    // Save current render targets explicitly
    IDirect3DSurface9* savedRT0;
    IDirect3DSurface9* savedDepthStencil;
    device->GetRenderTarget(0, &savedRT0);
    device->GetDepthStencilSurface(&savedDepthStencil);

    // Use cached shaders (compiled once during initialization, not per-frame!)
    IDirect3DSurface9* dstSurf;

    if (!vsHiZ || !psHiZ) {
        LOG::logline("!! Hi-Z: Shaders not initialized (vsHiZ=%p psHiZ=%p)", vsHiZ, psHiZ);
        device->SetRenderTarget(0, savedRT0);
        device->SetDepthStencilSurface(savedDepthStencil);
        savedRT0->Release();
        if (savedDepthStencil) savedDepthStencil->Release();
        return;
    }

    // Set cached shaders
    device->SetVertexShader(vsHiZ);
    device->SetPixelShader(psHiZ);

    // Generate all mip levels using MAX downsampling shader
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
    }

    // Reset sampler state
    device->SetSamplerState(0, D3DSAMP_MAXMIPLEVEL, 0);

    // Cleanup (don't release cached shaders - they're reused every frame!)
    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);

    // Restore render targets explicitly
    device->SetRenderTarget(0, savedRT0);
    device->SetDepthStencilSurface(savedDepthStencil);
    savedRT0->Release();
    if (savedDepthStencil) savedDepthStencil->Release();

    // Step 3: Copy entire Hi-Z mip chain to staging texture for CPU readback
    // Note: GetRenderTargetData is synchronous - each call blocks until GPU completes that copy
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

            // Removed debug mip0 sampling logging
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
    float cornerDepths[8];

    for (int i = 0; i < 8; i++) {
        D3DXVECTOR4 clipPos;
        D3DXVec3Transform(&clipPos, &corners[i], &worldViewProj);

        if (clipPos.w > 0.0f) {
            anyInFront = true;
            float invW = 1.0f / clipPos.w;
            float ndcX = clipPos.x * invW;
            float ndcY = clipPos.y * invW;
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
            cornerDepths[i] = -1.0f;
        }
    }

    if (!anyInFront) return false;

    // If any corners behind camera, object intersects near plane - never cull
    if (anyBehindCamera) {
        if (debugLog) {
            LOG::logline("Hi-Z Debug: Object intersects near plane - forced visible");
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

    // Select mip level: target ~6 pixels at selected mip for good coverage
    // Add -1 mip bias to use higher resolution (one mip level lower/more detailed)
    int mipLevel = 0;
    if (boxSize > 8.0f) {
        mipLevel = (int)std::floor(std::log2(boxSize / 6.0f)) - 1;
        mipLevel = std::max(0, std::min(hiZLevels - 1, mipLevel));
    }

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

    // Sample ALL Hi-Z pixels in bbox region and find maximum depth
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

    // Occlusion test: if closest bbox corner is behind furthest visible depth, cull
    // Increased bias to handle depth precision errors and bbox expansion padding
    float bias = 20.0f;
    bool culled = minLinearDepth > maxHiZDepth + bias;

    if (debugLog && culled) {
        LOG::logline("Hi-Z Debug CULLED: minDepth=%.4f, maxDepth=%.4f, maxHiZ=%.4f, mip=%d, region=[%d,%d]->[%d,%d]",
                     minLinearDepth, maxLinearDepth, maxHiZDepth, mipLevel, pixelMinX, pixelMinY, pixelMaxX, pixelMaxY);
        LOG::logline("  Corner depths: [0]=%.2f [1]=%.2f [2]=%.2f [3]=%.2f [4]=%.2f [5]=%.2f [6]=%.2f [7]=%.2f",
                     cornerDepths[0], cornerDepths[1], cornerDepths[2], cornerDepths[3],
                     cornerDepths[4], cornerDepths[5], cornerDepths[6], cornerDepths[7]);
        LOG::logline("  Hi-Z depth range: min=%.4f max=%.4f (%d pixels sampled)", minHiZDepth, maxHiZDepth, pixelCount);
        LOG::logline("  Screen bounds: X=[%.3f,%.3f] Y=[%.3f,%.3f] size=%.1fx%.1f pixels",
                     minX, maxX, minY, maxY, boxWidth, boxHeight);
        LOG::logline("  BBox: min=(%.1f,%.1f,%.1f) max=(%.1f,%.1f,%.1f)",
                     bboxMin.x, bboxMin.y, bboxMin.z, bboxMax.x, bboxMax.y, bboxMax.z);
    }

    return culled;
}
