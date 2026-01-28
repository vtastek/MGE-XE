
#include "configuration.h"
#include "distantland.h"
#include "distantshader.h"
#include "ffeshader.h"
#include "mwbridge.h"
#include "proxydx/d3d8header.h"
#include "support/log.h"
#include "tracy/Tracy.hpp"

#include <algorithm>
#include <cmath>



void DistantLand::renderDepth() {
    auto mwBridge = MWBridge::get();

    // DEBUG: Log that renderDepth is being called
    static bool loggedOnce = false;
    if (!loggedOnce) {
        LOG::logline(">> renderDepth() is being called - depth rendering active");
        loggedOnce = true;
    }

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

    // Reset projection matrix
    effect->SetMatrix(ehProj, &mwProj);
}

void DistantLand::renderDepthDistantLand() {
    auto mwBridge = MWBridge::get();

    // Switch to render target (already set from previous pass, but be explicit)
    RenderTargetSwitcher rtsw(surfDepthFrameMSAA, surfDepthDepth);

    // Projection for distant land
    D3DXMATRIX mwDepthProj = mwProj;
    if (Configuration.MGEFlags & USE_DISTANT_LAND) {
        editProjectionZ(&mwDepthProj, 1.0f, Configuration.DL.DrawDist * kCellSize);
    }
    effect->SetMatrix(ehProj, &mwDepthProj);

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

    // DEBUG: Log renderDepthRecorded call and recordMW size
    static bool loggedOnce = false;
    if (!loggedOnce) {
        LOG::logline(">> renderDepthRecorded() called with %d recorded draws", (int)recordMW.size());
        loggedOnce = true;
    }

    // Note: Culling is already done by prepareOcclusionCullingForDepth() using Hi-Z.
    // recordMW is pre-filtered - only visible objects remain.

    // Recorded renders (pre-filtered by prepareOcclusionCullingForDepth)
    for (const auto& i : recordMW) {
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
    // Note: Culling stats are now logged in prepareOcclusionCullingForDepth()
}

// GPU-only Hi-Z mip generation (non-blocking, ~0.5ms)
// Called at end of Present() - generates mips on GPU, returns immediately
void DistantLand::generateHiZMipsGPU() {
    ZoneScopedN("HiZ_GenerateMipsGPU");

    static int callCount = 0;
    if (callCount < 5) {
        LOG::logline(">> Hi-Z: generateHiZMipsGPU() called (frame %d)", callCount);
        callCount++;
    }

    if (!texHiZ || !texHiZPrev || !effectHiZ) {
        LOG::logline("Hi-Z: Skipping GPU generation - texHiZ=%p texHiZPrev=%p effectHiZ=%p", texHiZ, texHiZPrev, effectHiZ);
        return;
    }

    // Save current render targets and states that we'll modify
    IDirect3DSurface9* savedRT0;
    IDirect3DSurface9* savedDepthStencil;
    D3DVIEWPORT9 savedViewport;
    IDirect3DVertexShader9* savedVS = nullptr;
    IDirect3DPixelShader9* savedPS = nullptr;
    IDirect3DVertexDeclaration9* savedDecl = nullptr;
    IDirect3DVertexBuffer9* savedVB = nullptr;
    UINT savedVBStride, savedVBOffset;

    // Save render states we'll modify
    DWORD savedZEnable, savedZWriteEnable, savedCullMode, savedAlphaBlendEnable;

    {
        device->GetRenderTarget(0, &savedRT0);
        device->GetDepthStencilSurface(&savedDepthStencil);
        device->GetViewport(&savedViewport);
        device->GetVertexShader(&savedVS);
        device->GetPixelShader(&savedPS);
        device->GetVertexDeclaration(&savedDecl);
        device->GetStreamSource(0, &savedVB, &savedVBOffset, &savedVBStride);

        device->GetRenderState(D3DRS_ZENABLE, &savedZEnable);
        device->GetRenderState(D3DRS_ZWRITEENABLE, &savedZWriteEnable);
        device->GetRenderState(D3DRS_CULLMODE, &savedCullMode);
        device->GetRenderState(D3DRS_ALPHABLENDENABLE, &savedAlphaBlendEnable);
    }

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

    // Generate all mip levels (0-N) using MAX downsampling shader
    // Mip 0: Downsample from texCullDepth (320x240) to texHiZ mip 0 (160x120) using MAX
    // Mips 1-N: Ping-pong downsample between texHiZ and texHiZPrev
    {
        int actualMipsGenerated = 0;
        for (int mipLevel = 0; mipLevel < hiZValidMips; mipLevel++) {
            // Determine which texture to render to and which to sample from (ping-pong)
            // Even mips (0, 2, 4...): render to texHiZ, sample from texHiZPrev
            // Odd mips  (1, 3, 5...): render to texHiZPrev, sample from texHiZ
            IDirect3DTexture9* dstTexture = (mipLevel % 2 == 0) ? texHiZ : texHiZPrev;
            IDirect3DTexture9* srcTexture;

            // Get dimensions of destination mip level
            D3DSURFACE_DESC dstDesc;
            dstTexture->GetLevelDesc(mipLevel, &dstDesc);

            // Double-check: Stop at 8x8 minimum - don't generate smaller mips
            if (dstDesc.Width < 8 || dstDesc.Height < 8) {
                break;
            }
            actualMipsGenerated++;

            // Set render target to current mip level of destination texture
            dstTexture->GetSurfaceLevel(mipLevel, &dstSurf);
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

            // Get SOURCE dimensions and mip level (texture we're sampling from)
            D3DSURFACE_DESC srcDesc;
            int sourceMipLevel;

            if (mipLevel == 0) {
                // Mip 0: Sample from texCullDepth (Morrowind geometry only, copied from surfDepthFrameMSAA)
                // texCullDepth is filled by StretchRect in distantland.cpp:206 after renderDepth()
                texCullDepth->GetLevelDesc(0, &srcDesc);
                sourceMipLevel = 0;
                srcTexture = texCullDepth;
            } else {
                // Mip N: Sample from OPPOSITE texture's mip N-1 (ping-pong to avoid read/write conflict!)
                srcTexture = (mipLevel % 2 == 0) ? texHiZPrev : texHiZ;
                srcTexture->GetLevelDesc(mipLevel - 1, &srcDesc);
                sourceMipLevel = mipLevel - 1;
            }

            // Bind source texture (will be different from destination!)
            device->SetTexture(0, srcTexture);

            // Configure sampler with POINT filtering (matches HizMipmap9.fx)
            // CRITICAL: MIPFILTER must be POINT (not NONE) for tex2Dlod to access mip levels!
            device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
            device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
            device->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_POINT);  // Was NONE - broke tex2Dlod!
            device->SetSamplerState(0, D3DSAMP_MAXMIPLEVEL, 0);  // Allow all mip levels
            device->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
            device->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);

            // Set SOURCE texture info as pixel shader constant (c0)
            // xy = source dimensions, z = source mip level, w = destination mip level
            // CRITICAL: tex2Dlod needs the SOURCE mip level, not destination!
            float sourceInfo[4] = { (float)srcDesc.Width, (float)srcDesc.Height, (float)sourceMipLevel, (float)mipLevel };
            device->SetPixelShaderConstantF(0, sourceInfo, 1);

            // Set render states
            device->SetRenderState(D3DRS_ZENABLE, FALSE);
            device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
            device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
            device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);

            // Render full-screen quad with MAX downsample shader
            device->SetVertexDeclaration(WaterDecl);
            device->SetStreamSource(0, vbFullFrame, 0, 12);
            device->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2);

            // CRITICAL: Unbind source texture BEFORE next iteration (matches HZBManagerD3D9.cpp:213)
            // D3D9 requires this so the texture can be sampled in the next ping-pong step
            // Do NOT unbind render target - let next iteration SetRenderTarget handle it
            device->SetTexture(0, nullptr);

            dstSurf->Release();
        }

        // No consolidation needed - GPU culling shader samples from both textures
        // Even mips (0,2,4...) are in texHiZ, odd mips (1,3,5...) are in texHiZPrev
        // The shader checks mip level % 2 and samples from appropriate texture

        // Depth texture logging removed - render targets can't be locked directly
    }

    // Reset sampler states we modified
    device->SetSamplerState(0, D3DSAMP_MAXMIPLEVEL, 0);
    device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
    device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
    device->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
    device->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
    device->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);

    // Restore all modified state
    {
        device->SetRenderTarget(0, savedRT0);
        device->SetDepthStencilSurface(savedDepthStencil);
        device->SetViewport(&savedViewport);
        device->SetVertexShader(savedVS);
        device->SetPixelShader(savedPS);
        device->SetVertexDeclaration(savedDecl);
        device->SetStreamSource(0, savedVB, savedVBOffset, savedVBStride);
        device->SetTexture(0, nullptr);  // Clear texture binding

        // Restore render states
        device->SetRenderState(D3DRS_ZENABLE, savedZEnable);
        device->SetRenderState(D3DRS_ZWRITEENABLE, savedZWriteEnable);
        device->SetRenderState(D3DRS_CULLMODE, savedCullMode);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, savedAlphaBlendEnable);

        // Release saved references
        savedRT0->Release();
        if (savedDepthStencil) savedDepthStencil->Release();
        if (savedVS) savedVS->Release();
        if (savedPS) savedPS->Release();
        if (savedDecl) savedDecl->Release();
        if (savedVB) savedVB->Release();
    }
}

// Consolidate ping-pong Hi-Z pyramid into single texture for GPU culling
// Called after generateHiZMipsGPU() - copies all mips from texHiZ/texHiZPrev → texHiZPrevFrame
void DistantLand::consolidateHiZPyramid() {
    ZoneScopedN("HiZ_Consolidate");

    if (!texHiZ || !texHiZPrev || !texHiZPrevFrame) {
        return;
    }

    // Copy all valid mip levels from ping-pong pair into consolidated previous frame texture
    // Even mips (0,2,4...) from texHiZ → texHiZPrevFrame
    // Odd mips  (1,3,5...) from texHiZPrev → texHiZPrevFrame
    for (int mipLevel = 0; mipLevel < hiZValidMips; mipLevel++) {
        IDirect3DSurface9* srcSurf = nullptr;
        IDirect3DSurface9* dstSurf = nullptr;

        // Determine source texture based on ping-pong pattern
        IDirect3DTexture9* srcTexture = (mipLevel % 2 == 0) ? texHiZ : texHiZPrev;

        // Get surfaces
        srcTexture->GetSurfaceLevel(mipLevel, &srcSurf);
        texHiZPrevFrame->GetSurfaceLevel(mipLevel, &dstSurf);

        // Copy mip level from ping-pong source to consolidated destination
        // Use StretchRect for fast GPU-to-GPU copy (no CPU involvement)
        HRESULT hr = device->StretchRect(srcSurf, NULL, dstSurf, NULL, D3DTEXF_NONE);

        if (FAILED(hr)) {
            static bool logged = false;
            if (!logged) {
                LOG::logline("!! Failed to consolidate Hi-Z mip %d (hr=0x%x)", mipLevel, hr);
                logged = true;
            }
        }

        srcSurf->Release();
        dstSurf->Release();
    }

    // texHiZPrevFrame now contains complete pyramid for next frame's GPU culling
    // This happens at the END of frame N, ready for GPU culling in frame N+1
}

// CPU-GPU sync copy from render target to staging texture (blocking, ~1.5ms)
// Called at beginning of Clear/BeginScene() - copies GPU-generated mips to CPU-readable staging
void DistantLand::copyHiZToStaging() {
    {
        // Unlock staging texture from 2 frames ago (all locked mips)
        if (hiZLockedMips > 0 && texHiZStagingPrev) {
            for (int mip = 0; mip < hiZLockedMips; mip++) {
                texHiZStagingPrev->UnlockRect(mip);
            }
            hiZLockedMips = 0;
        }
    }

    if (!texHiZ || !texHiZPrev) {
        return;
    }

    // Step 3: Copy valid Hi-Z mips to staging texture for CPU readback (only mips down to 8x8)
    // IMPORTANT: Check if previous GetRenderTargetData is complete BEFORE issuing a new one
    // This allows us to skip the copy if GPU is still busy, avoiding 6ms stalls
    bool canCopyThisFrame = false;

    {
        if (queryHiZCopy) {
            // Non-blocking query check: Is the previous GetRenderTargetData complete?
            HRESULT hr = queryHiZCopy->GetData(NULL, 0, D3DGETDATA_FLUSH);
            if (hr == S_OK) {
                // Previous copy finished - safe to issue new one
                canCopyThisFrame = true;
            } else {
                // GPU still busy with previous copy - skip this frame, use old Hi-Z data
                // This is the KEY to non-blocking behavior!
                canCopyThisFrame = false;
            }
        } else {
            // No query available - first frame, just try the copy
            canCopyThisFrame = true;
        }
    }

    if (!canCopyThisFrame) {
        // Skip GetRenderTargetData this frame - GPU busy, use previous Hi-Z data
        // This prevents the 6ms stall and maintains frame rate!
        // Don't rotate buffers or issue query - keep using current pyramid
        return;
    }

    // Only copy if GPU is ready (non-blocking async behavior)
    {
        for (int mipLevel = 0; mipLevel < hiZValidMips; mipLevel++) {
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

    {
        // Swap Hi-Z buffers: current becomes previous for next frame's culling
        std::swap(texHiZ, texHiZPrev);

        // Triple-buffer rotation: N → N-1 → N-2 → N (gives 2 frames = ~40ms for GetRenderTargetData to complete)
        IDirect3DTexture9* temp = texHiZStagingPrev;  // Save N-2 (oldest, about to be freed for reuse)
        texHiZStagingPrev = texHiZStaging2;           // N-1 becomes N-2 (safe to lock now)
        texHiZStaging2 = texHiZStaging;               // N becomes N-1 (still copying)
        texHiZStaging = temp;                         // N-2 becomes N (ready for fresh GetRenderTargetData)

        // Rotate queries along with buffers
        IDirect3DQuery9* tempQuery = queryHiZCopyPrev;
        queryHiZCopyPrev = queryHiZCopy2;
        queryHiZCopy2 = queryHiZCopy;
        queryHiZCopy = tempQuery;

        // Issue D3D9 Event Query AFTER rotation, so queryHiZCopy tracks the buffer we just copied to
        if (queryHiZCopy) {
            queryHiZCopy->Issue(D3DISSUE_END);
        }
    }

    {
        // Pre-lock FIRST HALF of mip levels from 2-frame-old staging (query-synced, minimal stall)
        // Wait for GPU copy to complete before locking (check query for N-2 frame)
        static int checkCount = 0;

        // Skip first 2 checks - triple buffer needs 2 frames to prime
        if (checkCount < 2) {
            checkCount++;
            return;
        }

        if (queryHiZCopyPrev) {
            // Poll query until GPU signals completion (should complete immediately after 2 frames)
            // Use minimal attempts since 2 frames is plenty of time for GetRenderTargetData
            HRESULT hr = queryHiZCopyPrev->GetData(NULL, 0, 0);

            if (hr != S_OK) {
                // Query not ready - very rare, skip locking this frame
                static int timeoutCount = 0;
                if (timeoutCount < 3) {
                    LOG::logline("!! Hi-Z: Query not ready after 2 frames (unusual) - skipping pre-lock");
                    timeoutCount++;
                }
                return;
            }
        }

        int mipsToLockNow = (hiZValidMips + 1) / 2; // Lock lower mips (0 to mid-1), most commonly used
        for (int mip = 0; mip < mipsToLockNow; mip++) {
            HRESULT hr = texHiZStagingPrev->LockRect(mip, &hiZLockedRects[mip], NULL, D3DLOCK_READONLY);
            if (FAILED(hr)) {
                LOG::logline("!! Hi-Z: Failed to pre-lock mip %d (2 frames old)", mip);
                // Unlock any that succeeded
                for (int j = 0; j < mip; j++) {
                    texHiZStagingPrev->UnlockRect(j);
                }
                hiZLockedMips = 0;
                return;
            }
        }
        hiZLockedMips = mipsToLockNow;
    }
}

void DistantLand::lockRemainingHiZMips() {

    if (!texHiZStagingPrev || hiZLockedMips >= hiZValidMips) {
        return; // Already all valid mips locked or no texture
    }

    // Lock SECOND HALF of mip levels (remaining valid mips from where we left off)
    int startMip = hiZLockedMips;
    for (int mip = startMip; mip < hiZValidMips; mip++) {
        HRESULT hr = texHiZStagingPrev->LockRect(mip, &hiZLockedRects[mip], NULL, D3DLOCK_READONLY);
        if (FAILED(hr)) {
            LOG::logline("!! Hi-Z: Failed to lock remaining mip %d", mip);
            // Unlock the ones we just tried to lock (keep the first half locked)
            for (int j = startMip; j < mip; j++) {
                texHiZStagingPrev->UnlockRect(j);
            }
            return;
        }
    }
    hiZLockedMips = hiZValidMips; // Now all valid mips are locked
}

// Save CPU Hi-Z pyramid snapshot to temp folder (all mip levels as DDS + PNG files)
// Called when L key is pressed for debugging/analysis
void DistantLand::saveHiZSnapshot() {
    // Create temp directory if it doesn't exist
    const char* tempDir = "temp";
    CreateDirectoryA(tempDir, NULL);

    // Generate timestamped folder name for this snapshot
    SYSTEMTIME st;
    GetLocalTime(&st);
    char snapshotDir[256];
    snprintf(snapshotDir, sizeof(snapshotDir), "%s/cpu_hiz_%04d%02d%02d_%02d%02d%02d",
             tempDir, st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    CreateDirectoryA(snapshotDir, NULL);

    // Save CPU Hi-Z buffers from software occlusion culler
    FixedFunctionShader::softwareOcclusionCuller.saveToDisk(device, snapshotDir);
}

bool DistantLand::cullAgainstHiZ(const D3DXVECTOR3& bboxMin, const D3DXVECTOR3& bboxMax, const D3DXMATRIX& worldViewProj, bool debugLog) {

    // TEMPORARY: Disable Hi-Z culling while debugging pyramid data issue
    return false;

    // Use previous frame's Hi-Z for async culling (no GPU stall waiting for current frame)
    if (!texHiZStagingPrev || hiZLockedMips == 0) return false;

    float minX, maxX, minY, maxY, minLinearDepth, maxLinearDepth;
    int mipLevel;
    D3DSURFACE_DESC desc;
    float cornerDepths[8];
    float boxWidth, boxHeight;
    int pixelMinX, pixelMaxX, pixelMinY, pixelMaxY;

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
    minX = 1e10f, maxX = -1e10f;
    minY = 1e10f, maxY = -1e10f;
    minLinearDepth = 1e10f;
    maxLinearDepth = -1e10f;
    bool anyInFront = false;
    bool anyBehindCamera = false;

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
    texHiZStagingPrev->GetLevelDesc(0, &desc);
    boxWidth = (maxX - minX) * desc.Width;
    boxHeight = (maxY - minY) * desc.Height;
    float boxSize = std::max(boxWidth, boxHeight);

    // Select mip level: target ~6 pixels at selected mip for good coverage
    // Add -1 mip bias to use higher resolution (one mip level lower/more detailed)
    mipLevel = 0;
    if (boxSize > 8.0f) {
        mipLevel = (int)std::floor(std::log2(boxSize / 6.0f)) - 1;
        mipLevel = std::max(0, std::min(hiZLevels - 1, mipLevel));
    }

    // Check if the required mip is locked yet (split locking strategy)
    if (mipLevel >= hiZLockedMips) {
        // This mip hasn't been locked yet - don't cull (conservative approach)
        return false;
    }

    // Use pre-locked staging texture (locked in Present, no stall here!)
    float maxHiZDepth, minHiZDepth;
    int pixelCount;

    // Use pre-locked rect data
    D3DLOCKED_RECT& lr = hiZLockedRects[mipLevel];

    // Get mip level dimensions
    texHiZStagingPrev->GetLevelDesc(mipLevel, &desc);

    // Calculate pixel range covered by bbox at this mip level
    pixelMinX = (int)(minX * desc.Width);
    pixelMaxX = (int)(maxX * desc.Width);
    pixelMinY = (int)(minY * desc.Height);
    pixelMaxY = (int)(maxY * desc.Height);

    // Clamp to texture bounds
    pixelMinX = std::max(0, std::min((int)desc.Width - 1, pixelMinX));
    pixelMaxX = std::max(0, std::min((int)desc.Width - 1, pixelMaxX));
    pixelMinY = std::max(0, std::min((int)desc.Height - 1, pixelMinY));
    pixelMaxY = std::max(0, std::min((int)desc.Height - 1, pixelMaxY));

    // Sample ALL Hi-Z pixels in bbox region and find maximum depth
    float* depthData = (float*)lr.pBits;
    int pitch = lr.Pitch / sizeof(float);
    maxHiZDepth = 0.0f;
    minHiZDepth = 1e10f;
    pixelCount = 0;

    for (int y = pixelMinY; y <= pixelMaxY; y++) {
        for (int x = pixelMinX; x <= pixelMaxX; x++) {
            float depth = depthData[y * pitch + x];
            maxHiZDepth = std::max(maxHiZDepth, depth);
            minHiZDepth = std::min(minHiZDepth, depth);
            pixelCount++;
        }
    }

    // No unlock needed - already unlocked at start of next frame's generateHiZPyramid

    // Detect depth discontinuities (gaps between buildings, fences, etc.)
    // If there's a large depth variation in the screen-space region, don't cull
    // because the object might be visible through the gap
    float depthRange = maxHiZDepth - minHiZDepth;

    // For large objects, always check at mip 0 for gap detection to avoid false culling
    // when coarse mips fill in thin gaps through dilation
    bool hasGap = false;
    if (mipLevel > 0 && depthRange > 50.0f) {
        // Re-check at mip 0 for more accurate gap detection using pre-locked data
        D3DLOCKED_RECT& lr0 = hiZLockedRects[0];
        D3DSURFACE_DESC desc0;
        texHiZStagingPrev->GetLevelDesc(0, &desc0);

        int pix0MinX = (int)(minX * desc0.Width);
        int pix0MaxX = (int)(maxX * desc0.Width);
        int pix0MinY = (int)(minY * desc0.Height);
        int pix0MaxY = (int)(maxY * desc0.Height);

        pix0MinX = std::max(0, std::min((int)desc0.Width - 1, pix0MinX));
        pix0MaxX = std::max(0, std::min((int)desc0.Width - 1, pix0MaxX));
        pix0MinY = std::max(0, std::min((int)desc0.Height - 1, pix0MinY));
        pix0MaxY = std::max(0, std::min((int)desc0.Height - 1, pix0MaxY));

        float* depth0 = (float*)lr0.pBits;
        int pitch0 = lr0.Pitch / sizeof(float);
        float max0 = 0.0f;
        float min0 = 1e10f;

        for (int y = pix0MinY; y <= pix0MaxY; y++) {
            for (int x = pix0MinX; x <= pix0MaxX; x++) {
                float d = depth0[y * pitch0 + x];
                max0 = std::max(max0, d);
                min0 = std::min(min0, d);
            }
        }

        float range0 = max0 - min0;
        if (range0 > 100.0f) {
            hasGap = true;
            if (debugLog) {
                LOG::logline("Hi-Z Debug: Gap detected at mip 0 (range=%.1f) - forced visible", range0);
            }
        }
    } else if (depthRange > 100.0f) {
        hasGap = true;
        if (debugLog) {
            LOG::logline("Hi-Z Debug: Depth discontinuity detected (range=%.1f) - forced visible", depthRange);
        }
    }

    if (hasGap) {
        return false; // Don't cull - might be visible through gap
    }

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

// --------------------------------------------------------
// Texture-Based Lighting System
// --------------------------------------------------------

float DistantLand::computeLightRadius(float constant, float linear, float quadratic) {
    // Match shader attenuation: (1 / (40*q*d² + c)) * (1 - (d/350)^4)
    // Use fixed 350 unit cutoff from shader, but extend slightly for culling safety margin
    // The shader's quartic cutoff at 350 units means lights are effectively invisible beyond that

    // Use 400 units as safe maximum (350 + margin for Hi-Z culling tolerance)
    return 400.0f;
}

bool DistantLand::cullLightAgainstHiZ(const D3DXVECTOR3& bboxMin, const D3DXVECTOR3& bboxMax, const D3DXMATRIX& worldViewProj) {
    // Treat lights like objects: use same culling logic
    // This ensures consistent behavior between object and light culling
    return cullAgainstHiZ(bboxMin, bboxMax, worldViewProj, false);
}

void DistantLand::cullSceneLights(const D3DXMATRIX& viewProj) {
    ZoneScopedN("cullSceneLights");
    visibleLights.clear();
    int culled = 0;

    // Debug: log first few lights
    static bool logged = false;
    if (!logged && sceneLights.size() > 0) {
        LOG::logline(">> Light Debug: Total lights = %d", sceneLights.size());
        for (int i = 0; i < std::min(3, (int)sceneLights.size()); i++) {
            const auto& light = sceneLights[i];
            LOG::logline("  Light %d: pos=(%.1f,%.1f,%.1f) radius=%.1f falloff=(%.3f,%.3f,%.3f) color=(%.2f,%.2f,%.2f)",
                i, light.position.x, light.position.y, light.position.z, light.radius,
                light.falloff.x, light.falloff.y, light.falloff.z,
                light.diffuse.r, light.diffuse.g, light.diffuse.b);
        }
        logged = true;
    }

    {
        ZoneScopedN("cullSceneLights_loop");
        for (auto& light : sceneLights) {
            // Create sphere bounding box for light volume
            D3DXVECTOR3 bboxMin = light.position - D3DXVECTOR3(light.radius, light.radius, light.radius);
            D3DXVECTOR3 bboxMax = light.position + D3DXVECTOR3(light.radius, light.radius, light.radius);

            // Hi-Z cull the light's influence volume (use light-specific function)
            bool isOccluded = cullLightAgainstHiZ(bboxMin, bboxMax, viewProj);

            if (!isOccluded) {
                light.isVisible = true;
                visibleLights.push_back(light);
            } else {
                light.isVisible = false;
                culled++;
            }
        }
    }

    static int lastLoggedTotal = -1;
    static int lastLoggedVisible = -1;

    // Log if light counts changed significantly
    if (lastLoggedTotal != (int)sceneLights.size() || lastLoggedVisible != (int)visibleLights.size()) {
        LOG::logline(">> Light Culling: %d total, %d visible, %d culled (%.1f%%)",
                     (int)sceneLights.size(), (int)visibleLights.size(), culled,
                     sceneLights.size() > 0 ? (culled * 100.0f) / sceneLights.size() : 0.0f);
        lastLoggedTotal = (int)sceneLights.size();
        lastLoggedVisible = (int)visibleLights.size();
    }
}

void DistantLand::uploadLightDataToTexture(const D3DXMATRIX& viewMatrix) {
    ZoneScopedN("uploadLightDataToTexture");
    int numLights = (int)visibleLights.size();

    if (numLights == 0) {
        // No lights - unbind texture
        if (texLightData) {
            device->SetTexture(5, nullptr);
        }
        return;
    }

    int texelsNeeded = numLights * 3;  // 3 texels per light

    // Create or resize texture if needed
    if (!texLightData) {
        ZoneScopedN("uploadLights_CreateTexture");
        HRESULT hr = device->CreateTexture(
            texelsNeeded, 1,        // 1D texture (width × 1)
            1,                      // No mipmaps
            0,                      // Not a render target
            D3DFMT_A32B32G32R32F,   // 128-bit float format
            D3DPOOL_MANAGED,
            &texLightData,
            nullptr
        );

        if (FAILED(hr)) {
            LOG::logline("!! Failed to create light data texture (hr=0x%X)", hr);
            return;
        }
    } else {
        // Check if we need to resize
        D3DSURFACE_DESC desc;
        texLightData->GetLevelDesc(0, &desc);

        if (desc.Width != (UINT)texelsNeeded) {
            ZoneScopedN("uploadLights_ResizeTexture");
            // Resize needed
            texLightData->Release();

            HRESULT hr = device->CreateTexture(
                texelsNeeded, 1,
                1,
                0,
                D3DFMT_A32B32G32R32F,
                D3DPOOL_MANAGED,
                &texLightData,
                nullptr
            );

            if (FAILED(hr)) {
                LOG::logline("!! Failed to resize light data texture (hr=0x%X)", hr);
                texLightData = nullptr;
                return;
            }
        }
    }

    // Lock and fill texture
    D3DLOCKED_RECT locked;
    {
        ZoneScopedN("uploadLights_LockRect");
        if (SUCCEEDED(texLightData->LockRect(0, &locked, nullptr, 0))) {
            float* data = (float*)locked.pBits;

            {
                ZoneScopedN("uploadLights_FillData");
                for (int i = 0; i < numLights; i++) {
                    const SceneLight& light = visibleLights[i];
                    int offset = i * 12;  // 3 texels × 4 floats per texel

                    // Transform light position to view-space (to match legacy system)
                    D3DXVECTOR4 worldPos4(light.position.x, light.position.y, light.position.z, 1.0f);
                    D3DXVECTOR4 viewPos4;
                    D3DXVec4Transform(&viewPos4, &worldPos4, &viewMatrix);

                    // Texel 0: view-space position + radius
                    data[offset + 0] = viewPos4.x;
                    data[offset + 1] = viewPos4.y;
                    data[offset + 2] = viewPos4.z;
                    data[offset + 3] = light.radius;

                    // Texel 1: color
                    data[offset + 4] = light.diffuse.r;
                    data[offset + 5] = light.diffuse.g;
                    data[offset + 6] = light.diffuse.b;
                    data[offset + 7] = 0.0f;

                    // Texel 2: falloff parameters
                    data[offset + 8]  = light.falloff.x;  // constant
                    data[offset + 9]  = light.falloff.y;  // linear
                    data[offset + 10] = light.falloff.z;  // quadratic
                    data[offset + 11] = 0.0f;
                }
            }

            {
                ZoneScopedN("uploadLights_UnlockRect");
                texLightData->UnlockRect(0);
            }
        }
    }

    // Bind to device (slot 5)
    {
        ZoneScopedN("uploadLights_SetTexture");
        device->SetTexture(5, texLightData);
    }

    LOG::logline(">> Uploaded %d lights to texture (slot 5)", numLights);
}

// -------- GPU-Based Hi-Z Occlusion Culling Implementation --------

void DistantLand::initGPUCulling() {
    LOG::logline(">> Initializing GPU-based Hi-Z occlusion culling");

    // Load GPU culling shader
    {
        ID3DXBuffer* errors = nullptr;
        const char* shaderPath = "Data Files\\shaders\\core-hlsl\\XE HiZCull.hlsl";

        HRESULT hr = D3DXCreateEffectFromFileA(
            device,
            shaderPath,
            nullptr,
            nullptr,
            D3DXSHADER_OPTIMIZATION_LEVEL3,
            effectPool,
            &effectGPUCull,
            &errors
        );

        if (FAILED(hr)) {
            if (errors) {
                LOG::logline("!! GPU culling shader compile error:\n%s", (char*)errors->GetBufferPointer());
                errors->Release();
            }
            LOG::logline("!! Failed to load GPU culling shader: %s", shaderPath);
            return;
        }

        if (errors) {
            LOG::logline("-- GPU culling shader warnings:\n%s", (char*)errors->GetBufferPointer());
            errors->Release();
        }

        LOG::logline(">> Loaded GPU culling shader: %s", shaderPath);
    }

    // Create vertex declaration for bounding box data
    {
        D3DVERTEXELEMENT9 elements[] = {
            { 0, 0,  D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 }, // BBoxMin
            { 0, 12, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 1 }, // BBoxMax
            { 0, 24, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 }, // ResultPixel
            D3DDECL_END()
        };

        HRESULT hr = device->CreateVertexDeclaration(elements, &declGPUCullBounds);
        if (FAILED(hr)) {
            LOG::logline("!! Failed to create GPU culling vertex declaration");
            return;
        }
    }

    // Allocate vertex buffer for bounding boxes (initial size: 4096 objects)
    const UINT initialMaxObjects = 4096;
    const UINT vertexSize = sizeof(float) * 8; // 3 + 3 + 2 = 8 floats per vertex
    {
        HRESULT hr = device->CreateVertexBuffer(
            initialMaxObjects * vertexSize,
            D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
            0,
            D3DPOOL_DEFAULT,
            &vbGPUCullBounds,
            nullptr
        );

        if (FAILED(hr)) {
            LOG::logline("!! Failed to create GPU culling vertex buffer");
            return;
        }
    }

    // Calculate results texture size (1 pixel per object, packed into square)
    gpuCullMaxObjects = initialMaxObjects;
    gpuCullResultsWidth = (UINT)ceil(sqrt((float)gpuCullMaxObjects));
    gpuCullResultsHeight = gpuCullResultsWidth;

    // Create results texture in GPU memory (render target)
    {
        HRESULT hr = device->CreateTexture(
            gpuCullResultsWidth,
            gpuCullResultsHeight,
            1, // Single mip level
            D3DUSAGE_RENDERTARGET,
            D3DFMT_A8R8G8B8, // Simple 32-bit format for visibility flags
            D3DPOOL_DEFAULT,
            &texGPUCullResults,
            nullptr
        );

        if (FAILED(hr)) {
            LOG::logline("!! Failed to create GPU culling results texture (GPU)");
            return;
        }

        hr = texGPUCullResults->GetSurfaceLevel(0, &surfGPUCullResults);
        if (FAILED(hr)) {
            LOG::logline("!! Failed to get GPU culling results surface (GPU)");
            return;
        }
    }

    // Create results texture in system memory (for readback)
    {
        HRESULT hr = device->CreateTexture(
            gpuCullResultsWidth,
            gpuCullResultsHeight,
            1,
            0, // No D3DUSAGE flags
            D3DFMT_A8R8G8B8,
            D3DPOOL_SYSTEMMEM,
            &texGPUCullResultsSys,
            nullptr
        );

        if (FAILED(hr)) {
            LOG::logline("!! Failed to create GPU culling results texture (system memory)");
            return;
        }

        hr = texGPUCullResultsSys->GetSurfaceLevel(0, &surfGPUCullResultsSys);
        if (FAILED(hr)) {
            LOG::logline("!! Failed to get GPU culling results surface (system memory)");
            return;
        }
    }

    LOG::logline(">> GPU culling initialized: max %d objects, results texture %dx%d",
                 gpuCullMaxObjects, gpuCullResultsWidth, gpuCullResultsHeight);
}

void DistantLand::shutdownGPUCulling() {
    if (surfGPUCullResultsSys) {
        surfGPUCullResultsSys->Release();
        surfGPUCullResultsSys = nullptr;
    }
    if (texGPUCullResultsSys) {
        texGPUCullResultsSys->Release();
        texGPUCullResultsSys = nullptr;
    }
    if (surfGPUCullResults) {
        surfGPUCullResults->Release();
        surfGPUCullResults = nullptr;
    }
    if (texGPUCullResults) {
        texGPUCullResults->Release();
        texGPUCullResults = nullptr;
    }
    if (declGPUCullBounds) {
        declGPUCullBounds->Release();
        declGPUCullBounds = nullptr;
    }
    if (vbGPUCullBounds) {
        vbGPUCullBounds->Release();
        vbGPUCullBounds = nullptr;
    }
    if (effectGPUCull) {
        effectGPUCull->Release();
        effectGPUCull = nullptr;
    }

    gpuCullMaxObjects = 0;
    gpuCullResultsWidth = 0;
    gpuCullResultsHeight = 0;
}

void DistantLand::beginGPUCullingQuery(
    int numObjects,
    const D3DXVECTOR3* bboxMins,
    const D3DXVECTOR3* bboxMaxs,
    const D3DXMATRIX& view,
    const D3DXMATRIX& proj
) {
    if (!effectGPUCull || numObjects <= 0) {
        return;
    }

    // Check if we need to resize buffers
    if ((UINT)numObjects > gpuCullMaxObjects) {
        LOG::logline(">> GPU culling: resizing buffers for %d objects (was %d)", numObjects, gpuCullMaxObjects);
        shutdownGPUCulling();

        // Reinitialize with larger size (round up to next power of 2)
        UINT newSize = 1;
        while (newSize < (UINT)numObjects) {
            newSize *= 2;
        }

        gpuCullMaxObjects = newSize;
        initGPUCulling();
    }

    // Upload bounding box data to vertex buffer
    {

        void* pData = nullptr;
        HRESULT hr = vbGPUCullBounds->Lock(0, 0, &pData, D3DLOCK_DISCARD);
        if (FAILED(hr)) {
            LOG::logline("!! Failed to lock GPU culling vertex buffer");
            return;
        }

        float* vertices = (float*)pData;
        for (int i = 0; i < numObjects; i++) {
            // Calculate pixel position in results texture
            int pixelX = i % gpuCullResultsWidth;
            int pixelY = i / gpuCullResultsWidth;

            // BBoxMin (3 floats)
            vertices[i * 8 + 0] = bboxMins[i].x;
            vertices[i * 8 + 1] = bboxMins[i].y;
            vertices[i * 8 + 2] = bboxMins[i].z;

            // BBoxMax (3 floats)
            vertices[i * 8 + 3] = bboxMaxs[i].x;
            vertices[i * 8 + 4] = bboxMaxs[i].y;
            vertices[i * 8 + 5] = bboxMaxs[i].z;

            // ResultPixel (2 floats) - center of pixel
            vertices[i * 8 + 6] = (float)pixelX + 0.5f;
            vertices[i * 8 + 7] = (float)pixelY + 0.5f;
        }

        vbGPUCullBounds->Unlock();
    }

    // Compute view-projection matrix
    D3DXMATRIX viewProj = view * proj;

    // Extract frustum planes from view-projection matrix
    D3DXVECTOR4 frustumPlanes[6];
    {
        // Left plane
        frustumPlanes[0].x = viewProj._14 + viewProj._11;
        frustumPlanes[0].y = viewProj._24 + viewProj._21;
        frustumPlanes[0].z = viewProj._34 + viewProj._31;
        frustumPlanes[0].w = viewProj._44 + viewProj._41;

        // Right plane
        frustumPlanes[1].x = viewProj._14 - viewProj._11;
        frustumPlanes[1].y = viewProj._24 - viewProj._21;
        frustumPlanes[1].z = viewProj._34 - viewProj._31;
        frustumPlanes[1].w = viewProj._44 - viewProj._41;

        // Bottom plane
        frustumPlanes[2].x = viewProj._14 + viewProj._12;
        frustumPlanes[2].y = viewProj._24 + viewProj._22;
        frustumPlanes[2].z = viewProj._34 + viewProj._32;
        frustumPlanes[2].w = viewProj._44 + viewProj._42;

        // Top plane
        frustumPlanes[3].x = viewProj._14 - viewProj._12;
        frustumPlanes[3].y = viewProj._24 - viewProj._22;
        frustumPlanes[3].z = viewProj._34 - viewProj._32;
        frustumPlanes[3].w = viewProj._44 - viewProj._42;

        // Near plane
        frustumPlanes[4].x = viewProj._13;
        frustumPlanes[4].y = viewProj._23;
        frustumPlanes[4].z = viewProj._33;
        frustumPlanes[4].w = viewProj._43;

        // Far plane
        frustumPlanes[5].x = viewProj._14 - viewProj._13;
        frustumPlanes[5].y = viewProj._24 - viewProj._23;
        frustumPlanes[5].z = viewProj._34 - viewProj._33;
        frustumPlanes[5].w = viewProj._44 - viewProj._43;

        // Normalize planes
        for (int i = 0; i < 6; i++) {
            float len = sqrt(frustumPlanes[i].x * frustumPlanes[i].x +
                           frustumPlanes[i].y * frustumPlanes[i].y +
                           frustumPlanes[i].z * frustumPlanes[i].z);
            if (len > 0.0f) {
                frustumPlanes[i] /= len;
            }
        }
    }

    // Get viewport dimensions from current viewport
    D3DVIEWPORT9 viewport;
    device->GetViewport(&viewport);
    D3DXVECTOR2 viewportSize((float)viewport.Width, (float)viewport.Height);
    D3DXVECTOR2 resultsSize((float)gpuCullResultsWidth, (float)gpuCullResultsHeight);

    // Set shader parameters
    {
        effectGPUCull->SetMatrix(effectGPUCull->GetParameterByName(NULL, "g_mView"), &view);
        effectGPUCull->SetMatrix(effectGPUCull->GetParameterByName(NULL, "g_mProjection"), &proj);
        effectGPUCull->SetMatrix(effectGPUCull->GetParameterByName(NULL, "g_mViewProjection"), &viewProj);
        effectGPUCull->SetVectorArray(effectGPUCull->GetParameterByName(NULL, "g_FrustumPlanes"), frustumPlanes, 6);
        effectGPUCull->SetValue(effectGPUCull->GetParameterByName(NULL, "g_ViewportSize"), &viewportSize, sizeof(D3DXVECTOR2));
        effectGPUCull->SetValue(effectGPUCull->GetParameterByName(NULL, "g_ResultsSize"), &resultsSize, sizeof(D3DXVECTOR2));

        // Bind consolidated previous frame Hi-Z pyramid texture
        // texHiZPrevFrame contains all mips from previous frame (consolidated after generation)
        // This provides true 1-frame latency for occlusion culling
        effectGPUCull->SetTexture(effectGPUCull->GetParameterByName(NULL, "g_texHiZPrevFrame"), texHiZPrevFrame);
    }

    // Save current render target
    IDirect3DSurface9* savedRT = nullptr;
    IDirect3DSurface9* savedDS = nullptr;
    device->GetRenderTarget(0, &savedRT);
    device->GetDepthStencilSurface(&savedDS);

    // Set results texture as render target
    device->SetRenderTarget(0, surfGPUCullResults);
    device->SetDepthStencilSurface(nullptr);

    // Clear results texture to black (all occluded)
    device->Clear(0, nullptr, D3DCLEAR_TARGET, D3DCOLOR_ARGB(0, 0, 0, 0), 1.0f, 0);

    // Render visibility test
    {
        device->SetVertexDeclaration(declGPUCullBounds);
        device->SetStreamSource(0, vbGPUCullBounds, 0, sizeof(float) * 8);

        // Set point size to 1.0 to ensure each point renders exactly 1 pixel
        float pointSize = 1.0f;
        device->SetRenderState(D3DRS_POINTSIZE, *((DWORD*)&pointSize));
        device->SetRenderState(D3DRS_POINTSPRITEENABLE, FALSE);
        device->SetRenderState(D3DRS_POINTSCALEENABLE, FALSE);

        UINT numPasses = 0;
        effectGPUCull->Begin(&numPasses, 0);

        for (UINT pass = 0; pass < numPasses; pass++) {
            effectGPUCull->BeginPass(pass);
            device->DrawPrimitive(D3DPT_POINTLIST, 0, numObjects);
            effectGPUCull->EndPass();
        }

        effectGPUCull->End();
    }

    // Restore render target
    device->SetRenderTarget(0, savedRT);
    device->SetDepthStencilSurface(savedDS);

    if (savedRT) savedRT->Release();
    if (savedDS) savedDS->Release();
}

void DistantLand::endGPUCullingQuery(int numObjects, bool* visibilityResults) {
    if (!texGPUCullResults || !texGPUCullResultsSys || numObjects <= 0) {
        return;
    }

    // Copy results from GPU to system memory
    {
        HRESULT hr = device->GetRenderTargetData(surfGPUCullResults, surfGPUCullResultsSys);
        if (FAILED(hr)) {
            LOG::logline("!! Failed to copy GPU culling results to system memory");
            return;
        }
    }

    // Lock and read results
    {
        D3DLOCKED_RECT lockedRect;
        HRESULT hr = texGPUCullResultsSys->LockRect(0, &lockedRect, nullptr, D3DLOCK_READONLY);
        if (FAILED(hr)) {
            LOG::logline("!! Failed to lock GPU culling results texture");
            return;
        }

        BYTE* pixels = (BYTE*)lockedRect.pBits;

        // DEBUG: Sample first few pixels at frames 500, 1000, 1500
        static int frameCount = 0;
        frameCount++;
        if (frameCount == 500 || frameCount == 1000 || frameCount == 1500) {
            LOG::logline(">> GPU Culling Debug Sample FRAME %d (first 5 objects):", frameCount);
            for (int i = 0; i < std::min(5, numObjects); i++) {
                int pixelX = i % gpuCullResultsWidth;
                int pixelY = i / gpuCullResultsWidth;
                int offset = pixelY * lockedRect.Pitch + pixelX * 4;

                // A8R8G8B8 format: [B, G, R, A]
                BYTE b = pixels[offset + 0];
                BYTE g = pixels[offset + 1];
                BYTE r = pixels[offset + 2];
                BYTE a = pixels[offset + 3];

                // Decode: R=closestDepth/1000, G=maxOccluderDepth/1000, B=mipLevel/10, A=isVisible
                float closestDepth = (r / 255.0f) * 1000.0f;
                float maxOccluderDepth = (g / 255.0f) * 1000.0f;
                float mipLevel = (b / 255.0f) * 10.0f;
                bool isVisible = (a > 127);

                LOG::logline("  [%d]: closestDepth=%.1f, maxOccluderDepth=%.1f, mipLevel=%.1f, isVisible=%d (RGBA=%d,%d,%d,%d)",
                    i, closestDepth, maxOccluderDepth, mipLevel, isVisible ? 1 : 0, r, g, b, a);
            }
        }

        for (int i = 0; i < numObjects; i++) {
            int pixelX = i % gpuCullResultsWidth;
            int pixelY = i / gpuCullResultsWidth;

            int offset = pixelY * lockedRect.Pitch + pixelX * 4; // 4 bytes per pixel (A8R8G8B8)

            // DEBUG: Read alpha channel (isVisible flag) - A8R8G8B8 format: [B, G, R, A]
            BYTE visibility = pixels[offset + 3]; // A channel in A8R8G8B8 format
            visibilityResults[i] = (visibility > 127);
        }

        texGPUCullResultsSys->UnlockRect(0);
    }
}
