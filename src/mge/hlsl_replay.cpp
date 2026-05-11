// HLSL Replay System - Draw dispatch and replay for HLSL rendering pipeline
// Extracted from ffeshader.cpp as part of modularization for threading
//
// This module contains:
// - bindShaderTextures: Smart texture slot binding
// - computeShaderKeyWithSuffixes: ShaderKey computation with texture suffix detection
// - renderMorrowindHLSL: Recording/immediate path dispatcher
// - renderMorrowindHLSL_Internal: Core HLSL rendering with shader/constant setup
// - replayRecordedCalls: Batched draw call replay with Hi-Z culling

#include "ffeshader.h"
#include "distantlandhlsl.h"
#include "texture_suffix.h"
#include "cullthread.h"
#include "renderthread.h"
#include "d3dcommandbuffer.h"
#include "mge_tracy.h"
#include "configuration.h"
#include "support/log.h"
#include "mwbridge.h"
#include "morrowindbsa.h"
#include "distantland.h"
#include "imgui_manager.h"
#include "hlsl_shader_manager.h"
#include "patch_displacement.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <climits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

// File-scope statics used by replay functions (duplicated from ffeshader.cpp)
// During replay, points to fb.shadowViewproj (recording-time matrices).
// Outside replay, nullptr - callers fall back to s_staging.smViewproj.
static const D3DXMATRIX* s_activeShadowVP = nullptr;

static void computeShadowCascadeDepths(float outDepths[4]) {
    const float nearClip = 1.0f;
    const float shadowDistance = ImGuiManager::GetShadowDistance();
    const float splitLambda = ImGuiManager::GetSplitLambda();
    const float oldNearFarSplitP = 0.5f;
    const float logSplit = nearClip * std::pow(shadowDistance / nearClip, oldNearFarSplitP);
    const float linearSplit = nearClip + (shadowDistance - nearClip) * oldNearFarSplitP;
    const float oldNearFarSplit = splitLambda * logSplit + (1.0f - splitLambda) * linearSplit;
    const float closeSplit = std::min(
        ImGuiManager::GetCloseCascadeDistance(),
        std::max(nearClip, oldNearFarSplit - 1.0f));

    outDepths[0] = closeSplit;
    outDepths[1] = oldNearFarSplit;
    outDepths[2] = shadowDistance;
    outDepths[3] = 0.0f;
}

// Slow frame detection: prepareMs stored by prepareRecordedCalls, read by replayRecordedCalls
extern float lastPrepareMs;

// Flag: set to false by executeCullPass (cull thread) to prevent device calls in suffix fallback
extern bool deviceCallsSafeInPrepare;

// Diagnostic: cache hit/miss logging for first N frames
static int hlslDiagFrameCounter = 0;

// Replay metrics for material sorting optimization analysis
ReplayMetrics g_replayMetrics = {};

// Sampler state cache for recording (duplicated from ffeshader.cpp for HLSLRecordedCall ctor)
static std::unordered_map<IDirect3DBaseTexture9*, std::pair<DWORD, DWORD>> samplerCache;

// Helper: get current buffer's recorded calls (respects N-1 mode)
static auto& currentRecordedCalls() {
    return FixedFunctionShader::isUsingN1Buffer()
        ? FixedFunctionShader::getPrepBuffer().recordedCalls
        : FixedFunctionShader::getRenderingBuffer().recordedCalls;
}

// Material state cache helper functions
static inline void setCachedRenderState(IDirect3DDevice9* device, D3DRENDERSTATETYPE state, DWORD value, DWORD& cachedValue, bool& cacheValid) {
    if (!cacheValid || cachedValue != value) {
        device->SetRenderState(state, value);
        cachedValue = value;
        cacheValid = true;
    }
}

static inline void setCachedFVF(IDirect3DDevice9* device, DWORD fvf, DWORD& cachedFVF, bool& cacheValid) {
    if (!cacheValid || cachedFVF != fvf) {
        device->SetFVF(fvf);
        cachedFVF = fvf;
        cacheValid = true;
    }
}

// Helper: resolve a D3DXHANDLE to register offset using GetConstantDesc
static FixedFunctionShader::ConstReg resolveConstReg(ID3DXConstantTable* table, D3DXHANDLE handle) {
    FixedFunctionShader::ConstReg cr;
    if (table && handle) {
        D3DXCONSTANT_DESC desc;
        UINT count = 1;
        if (SUCCEEDED(table->GetConstantDesc(handle, &desc, &count))) {
            cr.reg = desc.RegisterIndex;
            cr.count = desc.RegisterCount;
            // D3DXRS_BOOL=0, D3DXRS_INT4=1, D3DXRS_FLOAT4=2, D3DXRS_SAMPLER=3
            cr.regSet = (UINT)desc.RegisterSet;
        }
    }
    return cr;
}

// Helper: resolve a named constant to a register offset, storing result in cached
static void resolveAndCache(ID3DXConstantTable* table, const char* name, FixedFunctionShader::ConstReg& cached) {
    D3DXHANDLE h = table ? table->GetConstantByName(NULL, name) : (D3DXHANDLE)NULL;
    cached = resolveConstReg(table, h);
}

// Helper: write a transposed matrix to a command buffer or device
static void setMatrixConstantF(D3DCommandBuffer* cmdBuf, IDirect3DDevice9* device, bool isVS, UINT reg, const D3DXMATRIX& matrix) {
    D3DXMATRIX transposed;
    D3DXMatrixTranspose(&transposed, &matrix);
    if (cmdBuf) {
        if (isVS) cmdBuf->recordSetVSConstantF(reg, (float*)&transposed, 4);
        else cmdBuf->recordSetPSConstantF(reg, (float*)&transposed, 4);
    } else {
        if (isVS) device->SetVertexShaderConstantF(reg, (float*)&transposed, 4);
        else device->SetPixelShaderConstantF(reg, (float*)&transposed, 4);
    }
}

// Helper: write float4 constants to command buffer or device
static void setConstantF(D3DCommandBuffer* cmdBuf, IDirect3DDevice9* device, bool isVS, UINT reg, const float* data, UINT count) {
    if (cmdBuf) {
        if (isVS) cmdBuf->recordSetVSConstantF(reg, data, count);
        else cmdBuf->recordSetPSConstantF(reg, data, count);
    } else {
        if (isVS) device->SetVertexShaderConstantF(reg, data, count);
        else device->SetPixelShaderConstantF(reg, data, count);
    }
}

static void setCloseCascadeShadowParams(D3DCommandBuffer* cmdBuf, IDirect3DDevice9* device) {
    float closeBiasParams[4] = {
        ImGuiManager::GetClosePCFBias(),
        ImGuiManager::GetClosePCFBias2(),
        ImGuiManager::GetClosePCFSlopeBias(),
        ImGuiManager::GetClosePCFTerrainBias()
    };
    float closeFilterParams[4] = {
        ImGuiManager::GetClosePCFFilterSize(),
        0.0f,
        0.0f,
        0.0f
    };

    setConstantF(cmdBuf, device, false, 44, closeBiasParams, 1);
    setConstantF(cmdBuf, device, false, 45, closeFilterParams, 1);
}

static void setTerrainShadowParams(D3DCommandBuffer* cmdBuf, IDirect3DDevice9* device, FixedFunctionShader::ConstReg reg, float isTerrain) {
    float terrainParams[4] = { isTerrain, ImGuiManager::GetPCFTerrainBias(), 0.0f, 0.0f };
    UINT targetReg = reg.reg != FixedFunctionShader::REG_INVALID ? reg.reg : 25;
    setConstantF(cmdBuf, device, false, targetReg, terrainParams, 1);
}

// Apply complete device state from snapshot to command buffer or device directly.
// Used during async replay where device starts from UNKNOWN state (no synchronous assumptions).
static void applyDeviceState(D3DCommandBuffer* cmdBuf, IDirect3DDevice9* device, const DeviceStateSnapshot& s) {
    // Helper lambda for setting render state to either cmdBuf or device
    auto setRS = [cmdBuf, device](D3DRENDERSTATETYPE state, DWORD value) {
        if (cmdBuf) {
            cmdBuf->recordSetRenderState(state, value);
        } else {
            device->SetRenderState(state, value);
        }
    };

    // Depth states
    setRS(D3DRS_ZENABLE, s.zEnable);
    setRS(D3DRS_ZWRITEENABLE, s.zWriteEnable);
    setRS(D3DRS_ZFUNC, s.zFunc);
    // depthBias and slopeScaleDepthBias are float-encoded DWORDs
    setRS(D3DRS_DEPTHBIAS, *(DWORD*)&s.depthBias);
    setRS(D3DRS_SLOPESCALEDEPTHBIAS, *(DWORD*)&s.slopeScaleDepthBias);

    // Culling
    setRS(D3DRS_CULLMODE, s.cullMode);

    // Blending
    setRS(D3DRS_ALPHABLENDENABLE, s.alphaBlendEnable);
    if (s.alphaBlendEnable) {
        setRS(D3DRS_SRCBLEND, s.srcBlend);
        setRS(D3DRS_DESTBLEND, s.destBlend);
    }

    // Alpha test
    setRS(D3DRS_ALPHATESTENABLE, s.alphaTestEnable);
    if (s.alphaTestEnable) {
        setRS(D3DRS_ALPHAFUNC, s.alphaFunc);
        setRS(D3DRS_ALPHAREF, s.alphaRef);
    }

    // Lighting/Material (HLSL handles these internally, but set for completeness)
    setRS(D3DRS_LIGHTING, s.lighting);
    setRS(D3DRS_SPECULARENABLE, s.specularEnable);
    setRS(D3DRS_LOCALVIEWER, s.localViewer);
    setRS(D3DRS_NORMALIZENORMALS, s.normalizeNormals);
    setRS(D3DRS_DIFFUSEMATERIALSOURCE, s.diffuseMatSrc);
    setRS(D3DRS_EMISSIVEMATERIALSOURCE, s.emissiveMatSrc);
    setRS(D3DRS_AMBIENTMATERIALSOURCE, s.ambientMatSrc);
    setRS(D3DRS_COLORVERTEX, s.colorVertex);
    setRS(D3DRS_VERTEXBLEND, s.vertexBlend);

    // Fog
    setRS(D3DRS_FOGENABLE, s.fogEnable);

    // Output
    setRS(D3DRS_COLORWRITEENABLE, s.colorWriteEnable);

    // Stencil
    setRS(D3DRS_STENCILENABLE, s.stencilEnable);

    // UI-specific
    setRS(D3DRS_AMBIENT, s.ambient);
    setRS(D3DRS_TEXTUREFACTOR, s.textureFactor);

    // Clip planes
    setRS(D3DRS_CLIPPLANEENABLE, s.clipPlaneEnable);

    // Debug
    setRS(D3DRS_FILLMODE, s.fillMode);

    // Point sprites (particles) - ALWAYS restore ALL states to prevent leaking
    // Even when disabled, previous draw's scale values can affect next draw
    setRS(D3DRS_POINTSPRITEENABLE, s.pointSpriteEnable);
    setRS(D3DRS_POINTSCALEENABLE, s.pointScaleEnable);
    setRS(D3DRS_POINTSIZE, *(DWORD*)&s.pointSize);
    setRS(D3DRS_POINTSCALE_A, *(DWORD*)&s.pointScaleA);
    setRS(D3DRS_POINTSCALE_B, *(DWORD*)&s.pointScaleB);
    setRS(D3DRS_POINTSCALE_C, *(DWORD*)&s.pointScaleC);
}

// Set replay baseline — establishes known device state at start of async replay
static void setReplayBaseline(D3DCommandBuffer* cmdBuf, IDirect3DDevice9* device) {
    DeviceStateSnapshot baseline;  // Default-constructed = D3D9 defaults
    applyDeviceState(cmdBuf, device, baseline);

    // Also reset samplers to known state
    for (int i = 0; i < 8; i++) {
        if (cmdBuf) {
            cmdBuf->recordSetSamplerState(i, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
            cmdBuf->recordSetSamplerState(i, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);
        } else {
            device->SetSamplerState(i, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
            device->SetSamplerState(i, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);
        }
    }

    // Clear VS constants to prevent cross-draw leakage
    // Shaders may use registers we don't explicitly set — stale values cause rendering bugs
    static float zeros[32 * 4] = {0};  // c0-c31 (32 registers, 4 floats each)
    if (cmdBuf) {
        cmdBuf->recordSetVSConstantF(0, zeros, 32);
    } else {
        device->SetVertexShaderConstantF(0, zeros, 32);
    }

    // Clear PS constants to prevent cross-draw leakage (matching VS clear)
    // Without this, stale PS constants from previous draws can leak to current draw
    if (cmdBuf) {
        cmdBuf->recordSetPSConstantF(0, zeros, 32);
    } else {
        device->SetPixelShaderConstantF(0, zeros, 32);
    }
}

// Smart texture binding - only binds slots that the shader actually uses
void FixedFunctionShader::bindShaderTextures(const ShaderKey& sk, const RenderedState* rs) {
    // Original slot assignment (keeping existing layout):
    // Slot 0: Base texture (always bound)
    // Slot 1: Detail texture (conditional with ifdef)
    // Slot 2: ParamH metallic/roughness
    // Slot 3: ParamX anisotropic
    // Slot 4: Shadow map

    auto& bindState = TextureSuffix::getBindingState();

    // Slot 0: Base texture (always used by HLSL shaders)
    // Morrowind-mode packing: base color is the plain `<name>.dds` Morrowind binds.
    IDirect3DTexture9* baseTexture = rs->texture;
    setCachedTexture(device, 0, baseTexture);

    // Slot 1: Detail texture (conditional only - with ifdef support)
    if (sk.hasDetail && savedOriginalDetailTexture) {
        setCachedTexture(device, 1, static_cast<IDirect3DTexture9*>(savedOriginalDetailTexture));
    }

    // Slots 2-3: Suffix textures (only if shader has suffix support)
    if (sk.hasParamH || sk.hasParamX) {
        // Fast check: if same texture pointer, skip all expensive operations
        if (bindState.lastBaseTexture != rs->texture) {
            // Get or create resolution cache entry (may perform expensive hash calculation)
            const auto* cached = TextureSuffix::getOrCreateResolution((IDirect3DDevice9*)device, rs->texture, true);

            if (cached && cached->hasValidName) {
                if (bindState.currentBaseTextureName != cached->textureName) {
                    // Reset cache when texture changes
                    bindState.currentBaseTextureName = cached->textureName;
                    bindState.boundParamH = nullptr;
                    bindState.boundParamX = nullptr;

                    if (cached->variants) {
                        // Debug: log suffix binding attempts
                        static int bindLogCount = 0;
                        bool shouldLog = LOG::catEnabled(LOG::Cat_HLSLReplay) && (bindLogCount < 5);

                        // Slot 2: ParamH (metallic/roughness) - load once per texture change
                        if (sk.hasParamH) {
                            if (cached->variants->hasParamH()) {
                                if (!bindState.boundParamH) {
                                    bindState.boundParamH = BSA::loadSuffixTexture((IDirect3DDevice9*)device, *cached->variants, "paramh");
                                    if (shouldLog) LOG::logline("bindShaderTextures: loaded paramH=%p for %s", bindState.boundParamH, cached->textureName.c_str());
                                }
                                if (bindState.boundParamH) {
                                    setCachedTextureWithSamplerPreservation(device, 2, bindState.boundParamH);
                                    if (shouldLog) LOG::logline("bindShaderTextures: bound paramH to slot 2");
                                }
                            } else {
                                // Bind default paramH texture with neutral PBR values (metalness=0, roughness=0.9)
                                setCachedTextureWithSamplerPreservation(device, 2, defaultParamHTexture);
                            }
                        }

                        // Slot 3: ParamX (anisotropic) - load once per texture change
                        if (sk.hasParamX) {
                            if (cached->variants->hasParamX()) {
                                if (!bindState.boundParamX) {
                                    bindState.boundParamX = BSA::loadSuffixTexture((IDirect3DDevice9*)device, *cached->variants, "paramx");
                                    if (shouldLog) LOG::logline("bindShaderTextures: loaded paramX=%p for %s", bindState.boundParamX, cached->textureName.c_str());
                                }
                                if (bindState.boundParamX) {
                                    setCachedTextureWithSamplerPreservation(device, 3, bindState.boundParamX);
                                    if (shouldLog) LOG::logline("bindShaderTextures: bound paramX to slot 3");
                                }
                            } else {
                                // Bind default normal texture for missing paramX
                                setCachedTextureWithSamplerPreservation(device, 3, defaultNormalTexture);
                            }
                        }

                        if (shouldLog && (sk.hasParamH || sk.hasParamX)) bindLogCount++;
                    }
                } else if (sk.hasParamH || sk.hasParamX) {
                    // Same resolved name as the previous bind but a new base pointer
                    // (Morrowind reuses resource pointers across draws). The suffix
                    // textures loaded for this name are still the right content, so
                    // re-bind from bindState. Only fall back to defaults when no
                    // real suffix was ever loaded for this name — binding the
                    // default here unconditionally stomps the paramH slot on every
                    // pointer-churned draw and flattens height mapping on the tile.
                    if (sk.hasParamH) {
                        setCachedTextureWithSamplerPreservation(device, 2,
                            bindState.boundParamH ? bindState.boundParamH : defaultParamHTexture);
                    }
                    if (sk.hasParamX) {
                        setCachedTextureWithSamplerPreservation(device, 3,
                            bindState.boundParamX ? bindState.boundParamX : defaultNormalTexture);
                    }
                }
                bindState.lastBaseTexture = rs->texture;
            }
        }
    }

    // Slot 4: raw shadow map for HLSL depth compares.
    // Legacy effects sample texSoftShadow after the blur pass; HLSL does its own PCF/ESM compares.
    if (sk.hasShadows) {
        setCachedTexture(device, 4, DistantLand::texShadow);
        // Slot 7: Blue noise texture for shadow PCF
        if (DistantLand::texBlueNoise) {
            setCachedTexture(device, 7, DistantLand::texBlueNoise);
        }
    }

    // Slot 5: Light data texture (for texture-based point lighting)
    // Per-object packed texture takes priority (mode 3 spatial query)
    IDirect3DTexture9* perObjTex = g_renderThread ? g_renderThread->getPerObjectLightTexture() : nullptr;
    if (perObjTex) {
        setCachedTexture(device, 5, perObjTex);
    } else if (DistantLand::texLightData) {
        setCachedTexture(device, 5, DistantLand::texLightData);
    }
}

static void bindForwardSSAOSlot(IDirect3DDevice9* device, D3DCommandBuffer* cmdBuf, bool enabled) {
    IDirect3DBaseTexture9* tex = enabled ? DistantLand::texForwardSSAO : nullptr;
    device->SetTexture(10, tex);
    device->SetSamplerState(10, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
    device->SetSamplerState(10, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
    device->SetSamplerState(10, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
    device->SetSamplerState(10, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
    device->SetSamplerState(10, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);

    if (cmdBuf) {
        cmdBuf->recordSetTexture(10, tex);
        cmdBuf->recordSetSamplerState(10, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
        cmdBuf->recordSetSamplerState(10, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
        cmdBuf->recordSetSamplerState(10, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
        cmdBuf->recordSetSamplerState(10, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
        cmdBuf->recordSetSamplerState(10, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
    }
}

static void setForwardSSAOParams(IDirect3DDevice9* device, D3DCommandBuffer* cmdBuf, bool enabled) {
    D3DVIEWPORT9 vp{};
    if (FAILED(device->GetViewport(&vp))) {
        float v[4] = { enabled ? 1.0f : 0.0f, 0.0f, 0.0f, DistantLand::forwardSSAOBendNormals ? 1.0f : 0.0f };
        setConstantF(cmdBuf, device, false, 24, v, 1);
        return;
    }

    const float width = std::max(static_cast<float>(vp.Width), 1.0f);
    const float height = std::max(static_cast<float>(vp.Height), 1.0f);
    float v[4] = {
        enabled ? 1.0f : 0.0f,
        1.0f / width,
        1.0f / height,
        DistantLand::forwardSSAOBendNormals ? 1.0f : 0.0f
    };
    setConstantF(cmdBuf, device, false, 24, v, 1);
}


// Helper function to compute ShaderKey with texture suffix detection
FixedFunctionShader::ShaderKey FixedFunctionShader::computeShaderKeyWithSuffixes(const RenderedState* rs, const FragmentState* frs, LightState* lightrs) {

    // Step 1: Determine texture suffix availability
    bool hasParamH = false, hasParamX = false, hasGrass = false, disableParallax = false;

    if (rs->texture) {
        // Use TextureSuffix module for thread-safe cache lookup/creation
        // deviceCallsSafeInPrepare controls whether device calls are allowed
        const auto* cached = TextureSuffix::getOrCreateResolution(
            (IDirect3DDevice9*)device, rs->texture, deviceCallsSafeInPrepare);

        if (cached && cached->hasValidName && cached->variants) {
            hasParamH = cached->variants->hasParamH();
            hasParamX = cached->variants->hasParamX();
            hasGrass = cached->variants->hasGrass();
            disableParallax = cached->variants->paramhNoParallax;
        }
    }

    // Step 2: Create ShaderKey with suffix flags
    ShaderKey sk(rs, frs, lightrs);
    sk.hasParamH = hasParamH;
    sk.hasParamX = hasParamX;
    sk.hasGrass = hasGrass;
    sk.disableParallax = disableParallax;

    // Set shadow flag based on MGE configuration
    sk.hasShadows = ((Configuration.MGEFlags & USE_SHADOWS) && (Configuration.MGEFlags & USE_DISTANT_LAND)) ? 1 : 0;

    return sk;
}

// HLSL Pipeline Implementation
void FixedFunctionShader::renderMorrowindHLSL(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, int recordMWIdx) {
    // Skip if we're in replay mode to avoid recursion
    if (isReplaying.load(std::memory_order_acquire)) {
        // During replay mode, perform actual rendering with this specific call
        renderMorrowindHLSL_Internal(rs, frs, lightrs);
        return;
    }

    // Start recording at first HLSL call if not already recording (unless under manual control or disabled)
    // Recording happens for ALL modes (standard, PPL, HLSL) - renderStageBlend is designed to run
    // before replay, compositing distant land based on depth before the scene is drawn.
    if (!isRecording.load(std::memory_order_acquire) && !isReplaying.load(std::memory_order_acquire) && !manualRecordingControl && recordingEnabled && !recordingCompletedThisFrame && ImGuiManager::GetEnableRecording()) {
        startRecording();
    }

    // Debug: Log Scene 1/2 recording state (first few calls per frame)
    static int scene12RecLogCount = 0;
    static int lastRecFrame = -1;
    if (currentRecordingScene > 0 && lastRecFrame != hlslDiagFrameCounter) {
        scene12RecLogCount = 0;
        lastRecFrame = hlslDiagFrameCounter;
    }
    if (currentRecordingScene > 0 && scene12RecLogCount < 3) {
        LOG_CAT(LOG::Cat_HLSLReplay, "Scene %d draw: isRecording=%d, enableRec=%d",
            currentRecordingScene, isRecording.load(std::memory_order_relaxed), ImGuiManager::GetEnableRecording());
        scene12RecLogCount++;
    }

    // If recording is active, record the call for batched replay
    if (isRecording.load(std::memory_order_acquire) && ImGuiManager::GetEnableRecording()) {
        // Create a copy of rs and add CURRENT shadow world-view-projection matrices for this draw call
        // During recording, use current matrices; during replay, these will be the "recorded" matrices
        RenderedState rsWithShadows = *rs;

        // Use current shadow matrices for this specific draw call during recording
        for (int i = 0; i < kShadowCascadeCount; ++i) {
            rsWithShadows.shadowWorldViewProj[i] = rs->worldTransforms[0] * DistantLand::s_staging.smViewproj[i];
        }

        // Defer shader key computation to prepare phase (avoid texture hash lookups during recording)
        ShaderKey sk;
        memset(&sk, 0, sizeof(sk));  // Placeholder - computed in prepareRecordedCalls()
        recordRenderCall(&rsWithShadows, frs, lightrs, sk, recordMWIdx);
        return;
    }

    // Immediate path for Scene 1/2 (particles/hands) and fallback when recording disabled.
    // Uses game's built-in 8 lights via lightrs, but needs shadow matrices computed.
    // BLOCKED when async GPU is active - main thread must not touch device during recording
    if (!ImGuiManager::GetEnableImmediateRendering() || ImGuiManager::GetAsyncGpuThread()) {
        return;  // Skip immediate rendering if disabled or async mode active
    }

    // Debug: Log device state for Scene 1/2 immediate draws (first few per frame)
    static int scene12DrawLogCount = 0;
    static int lastFrameLogged = -1;
    int currentFrame = hlslDiagFrameCounter;
    if (currentFrame != lastFrameLogged) {
        scene12DrawLogCount = 0;
        lastFrameLogged = currentFrame;
    }
    if (scene12DrawLogCount < 3) {
        D3DXMATRIX proj, view, world;
        device->GetTransform(D3DTS_PROJECTION, &proj);
        device->GetTransform(D3DTS_VIEW, &view);
        device->GetTransform(D3DTS_WORLD, &world);

        DWORD pointSize, pointScaleEnable, pointScaleA, pointScaleB, pointScaleC;
        device->GetRenderState(D3DRS_POINTSIZE, &pointSize);
        device->GetRenderState(D3DRS_POINTSCALEENABLE, &pointScaleEnable);
        device->GetRenderState(D3DRS_POINTSCALE_A, &pointScaleA);
        device->GetRenderState(D3DRS_POINTSCALE_B, &pointScaleB);
        device->GetRenderState(D3DRS_POINTSCALE_C, &pointScaleC);

        LOG::logline(">> Scene %d Immediate #%d: proj[0][0]=%.3f proj[3][2]=%.3f view[3][2]=%.3f world[3][0]=%.1f",
            currentRecordingScene, scene12DrawLogCount, proj._11, proj._34, view._34, world._41);
        LOG::logline("   PointSprite: size=%08X scaleEnable=%d A=%08X B=%08X C=%08X",
            pointSize, pointScaleEnable, pointScaleA, pointScaleB, pointScaleC);
        scene12DrawLogCount++;
    }

    // Compute shadow world-view-projection matrices for this draw call
    // (rs from mged3d8device has zeros - compute from current shadow map VP)
    RenderedState rsWithShadows = *rs;
    for (int i = 0; i < kShadowCascadeCount; ++i) {
            rsWithShadows.shadowWorldViewProj[i] = rs->worldTransforms[0] * DistantLand::s_staging.smViewproj[i];
        }

    renderMorrowindHLSL_Internal(&rsWithShadows, frs, lightrs);
}

void FixedFunctionShader::renderMorrowindHLSL_Internal(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, DWORD dirtyFlags, int callIndex, D3DCommandBuffer* cmdBuf, const DeviceStateSnapshot* capturedState, const HLSLRecordedCall* replayCall) {

    // Process any completed async shader compilations
    // Skip when on GPU thread (cmdBuf != null) to avoid race on pendingCompilations
    if (!cmdBuf) {
        processAsyncCompletions();
    }

    // STRESS TEST: Corrupt device state before HLSL rendering
    // If visual output is unchanged, proves renderMorrowindHLSL_Internal properly sets all required states
    if (ImGuiManager::GetStressCorruptState() && !cmdBuf) {
        device->SetRenderState(D3DRS_ZENABLE, FALSE);
        device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
        device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
        device->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_ZERO);
        device->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_ZERO);
        device->SetRenderState(D3DRS_ALPHATESTENABLE, TRUE);
        device->SetRenderState(D3DRS_ALPHAREF, 255);
        device->SetRenderState(D3DRS_ALPHAFUNC, D3DCMP_GREATER);
        device->SetRenderState(D3DRS_FOGENABLE, TRUE);
        device->SetRenderState(D3DRS_LIGHTING, FALSE);
        device->SetRenderState(D3DRS_SPECULARENABLE, FALSE);
        device->SetRenderState(D3DRS_AMBIENTMATERIALSOURCE, D3DMCS_MATERIAL);
        device->SetRenderState(D3DRS_DIFFUSEMATERIALSOURCE, D3DMCS_MATERIAL);
        device->SetVertexShader(nullptr);
        device->SetPixelShader(nullptr);
        device->SetFVF(D3DFVF_XYZ);  // Wrong FVF
        // Set wrong texture stage states
        device->SetTextureStageState(0, D3DTSS_COLOROP, D3DTOP_DISABLE);
        device->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_DISABLE);
    }

    HLSLShader hlslShader;

    // Get ShaderKey with texture suffix detection
    ShaderKey sk;
    if (isReplaying.load(std::memory_order_acquire)) {
        // During replay, use the recorded ShaderKey with original suffix flags
        // Select buffer based on mode: N-1 uses prepBuffer, N-2 uses renderBuffer
        auto& fb = usingN1Buffer ? getPrepBuffer() : getRenderingBuffer();
        bool found = false;
        for (const auto& call : fb.recordedCalls) {
            if (&call.rs == rs) { sk = call.sk; found = true; break; }
        }
        if (!found) {
            for (const auto& call : fb.recordedCallsScene1) {
                if (&call.rs == rs) { sk = call.sk; found = true; break; }
            }
        }
        if (!found) {
            for (const auto& call : fb.recordedCallsScene2) {
                if (&call.rs == rs) { sk = call.sk; found = true; break; }
            }
        }
        // Phase 7: callers that pass a patched RenderedState (rsDisplaced) won't
        // match any &call.rs in the buffer. Fall back to the provided replayCall's
        // recorded ShaderKey so the shader variant (incl. HAS_DISPLACEMENT) is correct.
        if (!found && replayCall) { sk = replayCall->sk; found = true; }
    } else {
        // During normal rendering, compute ShaderKey with texture suffix detection
        sk = computeShaderKeyWithSuffixes(rs, frs, lightrs);

        // Clamp lightMode to 2 (uniform-based) for immediate path.
        // Mode 3 (texture-based) requires per-object light packing from prepareRecordedCalls.
        if (sk.lightMode > 2) {
            sk.lightMode = 2;
        }
    }

    // Check if Morrowind bound a detail texture to slot 1 (frequency optimization)
    // Save the original detail texture before we start binding our own textures
    if (savedOriginalDetailTexture) {
        savedOriginalDetailTexture->Release(); // Release previous frame's texture
        savedOriginalDetailTexture = nullptr;
    }
    device->GetTexture(1, &savedOriginalDetailTexture);
    sk.hasDetail = (savedOriginalDetailTexture != nullptr) ? 1 : 0;

    if (sk == hlslShaderLRU.last_sk) {
        hlslShader = hlslShaderLRU.shader;
    } else {
        bool exactHit = false;
        // Read from shader cache under shared lock (precache thread may be inserting)
        AcquireSRWLockShared(&hlslCacheLock);
        decltype(cacheHLSLShaders)::const_iterator iShader = cacheHLSLShaders.find(sk);

        if (iShader != cacheHLSLShaders.end()) {
            hlslShader = iShader->second;
            exactHit = true;
            if (hlslDiagFrameCounter <= 5) {
                diagHitKeys.insert(sk);
            }
            // Log cache hits on early frames for debugging
            if (hlslDiagFrameCounter <= 3 && LOG::catEnabled(LOG::Cat_HLSLReplay)) {
                char buf[512];
                snprintf(buf, sizeof(buf),
                    "CACHE HIT frame=%d: lm=%d lit=%d vc=%d vm=%d hl=%d skin=%d fog=%d uv=%d stages=%d shadow=%d detail=%d ph=%d px=%d grass=%d",
                    hlslDiagFrameCounter,
                    (int)sk.lightMode, (int)sk.useLighting, (int)sk.vertexColour,
                    (int)sk.vertexMaterial, (int)sk.heavyLighting,
                    (int)sk.usesSkinning, (int)sk.fogMode, (int)sk.uvSets,
                    (int)sk.activeStages,
                    (int)sk.hasShadows, (int)sk.hasDetail,
                    (int)sk.hasParamH, (int)sk.hasParamX, (int)sk.hasGrass);
                LOG::logline("%s", buf);
            }
        }

        if (!exactHit) {
            // Diagnostic: log cache misses on early frames to identify precache gaps
            if (hlslDiagFrameCounter <= 3 && LOG::catEnabled(LOG::Cat_HLSLReplay)) {
                char buf[512];
                snprintf(buf, sizeof(buf),
                    "CACHE MISS frame=%d: lm=%d lit=%d vc=%d vm=%d hl=%d skin=%d fog=%d uv=%d stages=%d shadow=%d detail=%d ph=%d px=%d grass=%d bump=%d tg=%d",
                    hlslDiagFrameCounter,
                    (int)sk.lightMode, (int)sk.useLighting, (int)sk.vertexColour,
                    (int)sk.vertexMaterial, (int)sk.heavyLighting,
                    (int)sk.usesSkinning, (int)sk.fogMode, (int)sk.uvSets,
                    (int)sk.activeStages,
                    (int)sk.hasShadows, (int)sk.hasDetail,
                    (int)sk.hasParamH, (int)sk.hasParamX, (int)sk.hasGrass,
                    (int)sk.usesBumpmap, (int)sk.usesTexgen);
                LOG::logline("%s", buf);
                // Log per-stage details
                for (int s = 0; s < (int)sk.activeStages && s < 8; ++s) {
                    snprintf(buf, sizeof(buf),
                        "  stg%d=[op=%d a1=%d a2=%d a0=%d am=%d as=%d ti=%d tg=%d]",
                        s,
                        (int)sk.stage[s].colorOp, (int)sk.stage[s].colorArg1,
                        (int)sk.stage[s].colorArg2, (int)sk.stage[s].colorArg0,
                        (int)sk.stage[s].alphaOpMatched, (int)sk.stage[s].alphaOpSelect1,
                        (int)sk.stage[s].texcoordIndex, (int)sk.stage[s].texcoordGen);
                    LOG::logline("%s", buf);
                }
            }
            // Smart fallback hierarchy before using purple
            ShaderKey fallbackSk = sk;
            HLSLShader fallbackShader = {};
            bool foundFallback = false;

            // Normalize only alphaOpSelect1 (spurious variation); leave alphaOpMatched
            // untouched — it affects code generation and differs between stage[0] and stage[1+]
            auto normalizeAlpha = [](ShaderKey& key) {
                for (int s = 0; s < (int)key.activeStages; ++s) {
                    key.stage[s].alphaOpSelect1 = 0;
                }
            };

            // Normalize alpha bits on the base fallback key
            normalizeAlpha(fallbackSk);

            // lightMode + heavyLighting cascade: try different light modes AND hl=0
            int fallbackModes[] = {2, 1, 0};
            for (int hl = (int)sk.heavyLighting; hl >= 0 && !foundFallback; --hl) {
                fallbackSk.heavyLighting = hl;
                for (int i = 0; i < 3 && !foundFallback; ++i) {
                    if (fallbackModes[i] == (int)sk.lightMode && hl == (int)sk.heavyLighting) continue;
                    fallbackSk.lightMode = fallbackModes[i];
                    auto fallbackIter = cacheHLSLShaders.find(fallbackSk);
                    if (fallbackIter != cacheHLSLShaders.end()) {
                        fallbackShader = fallbackIter->second;
                        foundFallback = true;
                    }
                }
            }

            // If no point light fallback found, try texture suffix fallbacks
            if (!foundFallback && (sk.hasParamH || sk.hasParamX)) {
                ShaderKey textureFallbackSk = sk;
                normalizeAlpha(textureFallbackSk);
                textureFallbackSk.hasParamH = 0;
                textureFallbackSk.hasParamX = 0;
                textureFallbackSk.disableParallax = 0;
                auto fallbackIter = cacheHLSLShaders.find(textureFallbackSk);
                if (fallbackIter != cacheHLSLShaders.end()) {
                    fallbackShader = fallbackIter->second;
                    foundFallback = true;
                }
            }

            // Final universal fallback: try the 8 guaranteed base combinations
            if (!foundFallback) {
                ShaderKey universalSk = sk;
                normalizeAlpha(universalSk);
                universalSk.heavyLighting = 0;
                universalSk.hasParamH = 0;
                universalSk.hasParamX = 0;
                universalSk.disableParallax = 0;

                for (int lm = 0; lm <= 2 && !foundFallback; ++lm) {
                    for (int vertCol = 0; vertCol <= 1 && !foundFallback; ++vertCol) {
                        for (int skinning = 0; skinning <= 1 && !foundFallback; ++skinning) {
                            universalSk.lightMode = lm;
                            universalSk.vertexColour = vertCol;
                            universalSk.vertexMaterial = vertCol + 1;
                            universalSk.usesSkinning = skinning;

                            auto fallbackIter = cacheHLSLShaders.find(universalSk);
                            if (fallbackIter != cacheHLSLShaders.end()) {
                                fallbackShader = fallbackIter->second;
                                foundFallback = true;
                            }
                        }
                    }
                }
            }

            // Last resort: completely standardize key to match precache patterns
            // Resets fogMode, texgen, grass, bump, detail, and stage data
            if (!foundFallback) {
                ShaderKey stdSk;
                memset(&stdSk, 0, sizeof(stdSk));
                stdSk.uvSets = 1;
                stdSk.useLighting = sk.useLighting;
                stdSk.fogMode = 1;
                stdSk.activeStages = 1;
                stdSk.hasShadows = sk.hasShadows;
                stdSk.stage[0] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 1, 0, 0, 0 };

                for (int lm = 0; lm <= 2 && !foundFallback; ++lm) {
                    for (int vertCol = 0; vertCol <= 1 && !foundFallback; ++vertCol) {
                        for (int skinning = 0; skinning <= 1 && !foundFallback; ++skinning) {
                            stdSk.lightMode = lm;
                            stdSk.vertexColour = vertCol;
                            stdSk.vertexMaterial = vertCol + 1;
                            stdSk.usesSkinning = skinning;

                            auto fallbackIter = cacheHLSLShaders.find(stdSk);
                            if (fallbackIter != cacheHLSLShaders.end()) {
                                fallbackShader = fallbackIter->second;
                                foundFallback = true;
                            }
                        }
                    }
                }
            }

            if (foundFallback) {
                hlslShader = fallbackShader;
            } else {
                hlslShader = hlslShaderDefaultPurple;
            }
        }
        ReleaseSRWLockShared(&hlslCacheLock);

        // Queue async compilation for exact key on cache miss (outside lock)
        if (!exactHit) {
            queueShaderCompilation(sk);
        }

        hlslShaderLRU.shader = hlslShader;
        hlslShaderLRU.last_sk = sk;
    }
    
    // DXVK-optimized texture binding - only touches slots the shader actually uses
    bindShaderTextures(sk, rs);

    // Phase 6A: bind absorbed TerrainBlend overlay to sampler s8 for the main-replay
    // (non-batched / singleton) path. Mirrors the merged-batch bind below.
    bool boundOverlay = false;
    if (replayCall && replayCall->overlayTexture) {
        IDirect3DTexture9* overlayBind = static_cast<IDirect3DTexture9*>(replayCall->overlayTexture);
        device->SetTexture(8, overlayBind);
        device->SetSamplerState(8, D3DSAMP_MINFILTER, D3DTEXF_ANISOTROPIC);
        device->SetSamplerState(8, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
        device->SetSamplerState(8, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
        device->SetSamplerState(8, D3DSAMP_MAXANISOTROPY, 16);
        device->SetSamplerState(8, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
        device->SetSamplerState(8, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);
        boundOverlay = true;
    }

    // Phase 8A: bind overlay's _paramh to sampler s9 so the PS can blend paramh
    // between base and overlay by the same AlphaGrid factor used for albedo.
    if (replayCall && replayCall->overlayParamHTexture) {
        device->SetTexture(9, replayCall->overlayParamHTexture);
        device->SetSamplerState(9, D3DSAMP_MINFILTER, D3DTEXF_ANISOTROPIC);
        device->SetSamplerState(9, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
        device->SetSamplerState(9, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
        device->SetSamplerState(9, D3DSAMP_MAXANISOTROPY, 16);
        device->SetSamplerState(9, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
        device->SetSamplerState(9, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);
    }

    const bool useForwardSSAO = DistantLand::forwardSSAOEnabled && DistantLand::forwardSSAOActive && replayCall &&
        (replayCall->sceneNum == 0 || replayCall->sceneNum == 2);
    bindForwardSSAOSlot(device, cmdBuf, useForwardSSAO);
    setForwardSSAOParams(device, cmdBuf, useForwardSSAO);

    // When building command buffer, snapshot bound textures into the buffer
    // (bindShaderTextures sets device textures; we need them in the command buffer
    //  so they replay correctly in order with other commands)
    if (cmdBuf) {
        for (DWORD slot = 0; slot < 6; slot++) {
            IDirect3DBaseTexture9* boundTex = nullptr;
            device->GetTexture(slot, &boundTex);
            cmdBuf->recordSetTexture(slot, boundTex);
            if (boundTex) boundTex->Release(); // GetTexture AddRef'd, recordSetTexture will AddRef again
        }
    }

    // Get current view matrix and compute inverse (needed for texture lights and shadows)
    // During replay, device may have UI view — use recorded game view instead
    // Select buffer based on mode: N-1 uses prepBuffer, N-2 uses renderBuffer
    D3DXMATRIX currentView;
    if (isReplaying.load(std::memory_order_acquire)) {
        auto& fb = usingN1Buffer ? getPrepBuffer() : getRenderingBuffer();
        currentView = fb.currentView;
    } else {
        device->GetTransform(D3DTS_VIEW, &currentView);
    }

    // Only recalculate inverse view when view changes
    if (!shadowMatricesValid || memcmp(&currentView, &cachedViewMatrix, sizeof(D3DXMATRIX)) != 0) {
        cachedViewMatrix = currentView;
        D3DXMatrixInverse(&cachedInverseView, NULL, &currentView);
        shadowMatricesValid = true;
    }

    // Set viewInverse matrix for texture-based lighting (always needed)
    setConstantF(cmdBuf, device, false, 18, (float*)&cachedInverseView, 4); // c18

    // Set shadow matrices if shadows are enabled
    if (sk.hasShadows) {
        // Compute shadow transform matrices using cached inverse view
        D3DXMATRIX cachedViewToShadowLocal[kShadowCascadeCount];
        const auto* svp = s_activeShadowVP ? s_activeShadowVP : DistantLand::s_staging.smViewproj;
        for (int i = 0; i < kShadowCascadeCount; ++i) {
            cachedViewToShadowLocal[i] = cachedInverseView * svp[i];
        }

        // VS keeps the first two legacy interpolators; PS gets all cascades from view-space position.
        setConstantF(cmdBuf, device, true, 60, (float*)&cachedViewToShadowLocal[0], 4); // c60-c63
        setConstantF(cmdBuf, device, true, 64, (float*)&cachedViewToShadowLocal[1], 4); // c64-c67
        for (int i = 0; i < kShadowCascadeCount; ++i) {
            setMatrixConstantF(cmdBuf, device, false, 31 + i * 4, cachedViewToShadowLocal[i]);
        }
        float shadowCascadeDepths[4];
        computeShadowCascadeDepths(shadowCascadeDepths);
        setConstantF(cmdBuf, device, false, 43, shadowCascadeDepths, 1);
        setCloseCascadeShadowParams(cmdBuf, device);

        // Set shadow resolution parameter
        float shadowRcpData[4] = { 1.0f / Configuration.DL.ShadowResolution, 0, 0, 0 };
        setConstantF(cmdBuf, device, false, 10, shadowRcpData, 1); // c10
    }


    // Save current render states before modifying them (only states that HLSL actually changes)
    DWORD savedAlphaBlendEnable = 0, savedAlphaTestEnable = 0;
    DWORD savedZEnable = 0, savedZWriteEnable = 0;
    DWORD savedSpecularEnable = 0, savedLocalViewer = 0, savedNormalizeNormals = 0;
    DWORD savedAmbientMatSrc = 0, savedDiffuseMatSrc = 0, savedEmissiveMatSrc = 0;
    if (!cmdBuf) {
        device->GetRenderState(D3DRS_ALPHABLENDENABLE, &savedAlphaBlendEnable);
        device->GetRenderState(D3DRS_ALPHATESTENABLE, &savedAlphaTestEnable);
        device->GetRenderState(D3DRS_ZENABLE, &savedZEnable);
        device->GetRenderState(D3DRS_ZWRITEENABLE, &savedZWriteEnable);
        device->GetRenderState(D3DRS_SPECULARENABLE, &savedSpecularEnable);
        device->GetRenderState(D3DRS_LOCALVIEWER, &savedLocalViewer);
        device->GetRenderState(D3DRS_NORMALIZENORMALS, &savedNormalizeNormals);
        device->GetRenderState(D3DRS_AMBIENTMATERIALSOURCE, &savedAmbientMatSrc);
        device->GetRenderState(D3DRS_DIFFUSEMATERIALSOURCE, &savedDiffuseMatSrc);
        device->GetRenderState(D3DRS_EMISSIVEMATERIALSOURCE, &savedEmissiveMatSrc);
    }

    // Set shaders
    if (cmdBuf) {
        cmdBuf->recordSetVertexShader(hlslShader.vertexShader);
        cmdBuf->recordSetPixelShader(hlslShader.pixelShader);
    } else {
        device->SetVertexShader(hlslShader.vertexShader);
        device->SetPixelShader(hlslShader.pixelShader);
    }
    
    // Use cached render state setting to minimize redundant SetRenderState calls
    // Depth and culling states
    DWORD zEnable, zWriteEnable;
    if (rs->blendEnable) {
        zEnable = TRUE;
        zWriteEnable = FALSE;
    } else {
        zEnable = rs->zWrite ? TRUE : FALSE;
        zWriteEnable = rs->zWrite ? TRUE : FALSE;
    }

    if (cmdBuf) {
        // Async replay path: apply full captured device state when available
        if (capturedState) {
            // Apply complete captured state (no assumptions about prior device state)
            applyDeviceState(cmdBuf, device, *capturedState);
        } else {
            // Fallback to computed state (legacy path, for non-recorded calls)
            cmdBuf->recordSetRenderState(D3DRS_ZENABLE, zEnable);
            cmdBuf->recordSetRenderState(D3DRS_ZWRITEENABLE, zWriteEnable);
            cmdBuf->recordSetRenderState(D3DRS_ZFUNC, D3DCMP_LESSEQUAL);
            cmdBuf->recordSetRenderState(D3DRS_CULLMODE, rs->cullMode);
            cmdBuf->recordSetRenderState(D3DRS_SPECULARENABLE, FALSE);
            cmdBuf->recordSetRenderState(D3DRS_LOCALVIEWER, FALSE);
            cmdBuf->recordSetRenderState(D3DRS_NORMALIZENORMALS, FALSE);
            cmdBuf->recordSetRenderState(D3DRS_ALPHABLENDENABLE, rs->blendEnable);
            if (rs->blendEnable) {
                cmdBuf->recordSetRenderState(D3DRS_SRCBLEND, rs->srcBlend);
                cmdBuf->recordSetRenderState(D3DRS_DESTBLEND, rs->destBlend);
            }
            cmdBuf->recordSetRenderState(D3DRS_ALPHATESTENABLE, rs->alphaTest);
            if (rs->alphaTest) {
                cmdBuf->recordSetRenderState(D3DRS_ALPHAFUNC, rs->alphaFunc);
                cmdBuf->recordSetRenderState(D3DRS_ALPHAREF, rs->alphaRef);
            }
        }
        cmdBuf->recordSetFVF(rs->fvf);
        cmdBuf->recordSetStreamSource(0, rs->vb, rs->vbOffset, rs->vbStride);
        if (rs->ib) {
            cmdBuf->recordSetIndices(rs->ib);
        }
    } else {
        setCachedRenderState(device, D3DRS_ZENABLE, zEnable, materialCache.zEnable, materialCache.zEnableValid);
        setCachedRenderState(device, D3DRS_ZWRITEENABLE, zWriteEnable, materialCache.zWriteEnable, materialCache.zWriteEnableValid);
        setCachedRenderState(device, D3DRS_ZFUNC, D3DCMP_LESSEQUAL, materialCache.zFunc, materialCache.zFuncValid);
        setCachedRenderState(device, D3DRS_CULLMODE, rs->cullMode, materialCache.cullMode, materialCache.cullModeValid);

        // Disable DX8 specular pipeline that Morrowind.exe might have enabled - HLSL handles specular internally
        setCachedRenderState(device, D3DRS_SPECULARENABLE, FALSE, materialCache.specularEnable, materialCache.specularEnableValid);
        setCachedRenderState(device, D3DRS_LOCALVIEWER, FALSE, materialCache.localViewer, materialCache.localViewerValid);
        setCachedRenderState(device, D3DRS_NORMALIZENORMALS, FALSE, materialCache.normalizeNormals, materialCache.normalizeNormalsValid);

        // Alpha blending states
        setCachedRenderState(device, D3DRS_ALPHABLENDENABLE, rs->blendEnable, materialCache.alphaBlendEnable, materialCache.alphaBlendEnableValid);
        if (rs->blendEnable) {
            setCachedRenderState(device, D3DRS_SRCBLEND, rs->srcBlend, materialCache.srcBlend, materialCache.srcBlendValid);
            setCachedRenderState(device, D3DRS_DESTBLEND, rs->destBlend, materialCache.destBlend, materialCache.destBlendValid);
        }

        // Alpha testing states
        setCachedRenderState(device, D3DRS_ALPHATESTENABLE, rs->alphaTest, materialCache.alphaTestEnable, materialCache.alphaTestEnableValid);
        if (rs->alphaTest) {
            setCachedRenderState(device, D3DRS_ALPHAFUNC, rs->alphaFunc, materialCache.alphaFunc, materialCache.alphaFuncValid);
            setCachedRenderState(device, D3DRS_ALPHAREF, rs->alphaRef, materialCache.alphaRef, materialCache.alphaRefValid);
        }

        // Fill mode: re-apply the captured value every draw so `twf` wireframe
        // survives the postshaders-forced SOLID between scenes. Falls back to
        // SOLID when we have no captured state (legacy/non-recorded calls).
        {
            DWORD fm = capturedState ? capturedState->fillMode : D3DFILL_SOLID;
            setCachedRenderState(device, D3DRS_FILLMODE, fm, materialCache.fillMode, materialCache.fillModeValid);
        }

        // Point sprites (particles) - ALWAYS restore ALL states from captured state
        // Even when disabled, previous draw's scale values can affect next draw
        if (capturedState) {
            device->SetRenderState(D3DRS_POINTSPRITEENABLE, capturedState->pointSpriteEnable);
            device->SetRenderState(D3DRS_POINTSCALEENABLE, capturedState->pointScaleEnable);
            device->SetRenderState(D3DRS_POINTSIZE, *(DWORD*)&capturedState->pointSize);
            device->SetRenderState(D3DRS_POINTSCALE_A, *(DWORD*)&capturedState->pointScaleA);
            device->SetRenderState(D3DRS_POINTSCALE_B, *(DWORD*)&capturedState->pointScaleB);
            device->SetRenderState(D3DRS_POINTSCALE_C, *(DWORD*)&capturedState->pointScaleC);
        }

        // Set vertex format (legacy DX8 FVF - HLSL input semantics handle layout internally)
        setCachedFVF(device, rs->fvf, materialCache.fvf, materialCache.fvfValid);

        // Set vertex and index buffers like the original system
        device->SetStreamSource(0, rs->vb, rs->vbOffset, rs->vbStride);
        if (rs->ib) {
            device->SetIndices(rs->ib);
        }
    }

    // Set up matrices using constant tables (like the Combined shader expects)
    D3DXMATRIX projMatrix, viewMatrix, worldMatrix;

    // During replay, use ALL recorded matrices to avoid stale matrix issues
    // During normal rendering, get them from the device
    // Select buffer based on mode: N-1 uses prepBuffer, N-2 uses renderBuffer
    if (isReplaying.load(std::memory_order_acquire)) {
        auto& fb = usingN1Buffer ? getPrepBuffer() : getRenderingBuffer();
        projMatrix = fb.currentProj;
        viewMatrix = fb.currentView;
        worldMatrix = rs->worldTransforms[0];
    } else {
        device->GetTransform(D3DTS_PROJECTION, &projMatrix);
        device->GetTransform(D3DTS_VIEW, &viewMatrix);
        device->GetTransform(D3DTS_WORLD, &worldMatrix);
    }

    // During replay, use recorded combined matrices; during normal rendering, calculate them
    D3DXMATRIX worldViewProj, worldView;
    if (isReplaying.load(std::memory_order_acquire)) {
        // Use precomputed worldViewTransforms from recording time - this has the CORRECT view matrix
        // that was active when MW issued this draw call. Recomputing with frameBuffer view would
        // use the wrong view if buffer indices rotated.
        worldView = rs->worldViewTransforms[0];
        worldViewProj = rs->worldViewTransforms[0] * projMatrix;
    } else {
        worldViewProj = worldMatrix * viewMatrix * projMatrix;
        worldView = worldMatrix * viewMatrix;
    }

    // Set vertex shader constants — command buffer path uses resolved registers,
    // direct path uses constant tables (SetMatrix transposes internally)

    // Debug: Log ALL blended draws (particles)
    if (rs->blendEnable && !rs->zWrite) {
        // Check point sprite state from captured state
        const char* psState = "no-cap";
        float psSize = 0;
        if (capturedState) {
            psState = capturedState->pointSpriteEnable ? "PS-ON" : "PS-off";
            psSize = capturedState->pointSize;
        }

    }

    if (cmdBuf) {
        if (hlslShader.regWorldViewProj.reg != REG_INVALID)
            setMatrixConstantF(cmdBuf, device, true, hlslShader.regWorldViewProj.reg, worldViewProj);
        if (hlslShader.regView.reg != REG_INVALID)
            setMatrixConstantF(cmdBuf, device, true, hlslShader.regView.reg, viewMatrix);
        if (hlslShader.regProj.reg != REG_INVALID)
            setMatrixConstantF(cmdBuf, device, true, hlslShader.regProj.reg, projMatrix);
        if (hlslShader.regWorld.reg != REG_INVALID)
            setMatrixConstantF(cmdBuf, device, true, hlslShader.regWorld.reg, rs->worldTransforms[0]);
        if (hlslShader.regWorldView.reg != REG_INVALID) {
            // For rigid draws (vbs=0), use precomputed worldViewTransforms[0] instead of replay-time worldView
            // The shader reads from worldview register (c4) for rigid path, not vertexBlendPalette
            if (rs->vertexBlendState == 0) {
                setMatrixConstantF(cmdBuf, device, true, hlslShader.regWorldView.reg, rs->worldViewTransforms[0]);
            } else {
                setMatrixConstantF(cmdBuf, device, true, hlslShader.regWorldView.reg, worldView);
            }
        }

        if (hlslShader.regVertexBlendPalette.reg != REG_INVALID) {
            D3DXMATRIX blendMatrices[4];
            if (rs->vertexBlendState > 0) {
                // Use precomputed worldViewTransforms from recording time
                for (int i = 0; i < 4; i++)
                    blendMatrices[i] = rs->worldViewTransforms[i];
            } else {
                // Rigid path: also use precomputed worldViewTransforms[0] for consistency
                // Using replay-time worldView would cause mismatch with recorded geometry
                blendMatrices[0] = rs->worldViewTransforms[0];
                memset(&blendMatrices[1], 0, sizeof(D3DXMATRIX) * 3);
            }
            // SetMatrixArray transposes each matrix — we must do the same
            D3DXMATRIX transposed[4];
            for (int i = 0; i < 4; i++)
                D3DXMatrixTranspose(&transposed[i], &blendMatrices[i]);
            cmdBuf->recordSetVSConstantF(hlslShader.regVertexBlendPalette.reg, (float*)transposed, 16);
        }

        if (hlslShader.regVertexBlendState.reg != REG_INVALID) {
            float blendState[4] = { (float)rs->vertexBlendState, 0, 0, 0 };
            cmdBuf->recordSetVSConstantF(hlslShader.regVertexBlendState.reg, blendState, 1);
        } else if (rs->blendEnable && !rs->zWrite) {
            // Particle without regVertexBlendState - state may be stale!
            static bool logged = false;
            if (!logged) {
                LOG::logline("!! PARTICLE: regVertexBlendState=INVALID, vbs=%d may be stale!", rs->vertexBlendState);
                logged = true;
            }
        }

        if (hlslShader.regShadowWorldViewProj.reg != REG_INVALID) {
            D3DXMATRIX shadowWVP[2];
            if (isReplaying.load(std::memory_order_acquire) && s_activeShadowVP) {
                shadowWVP[0] = rs->worldTransforms[0] * s_activeShadowVP[0];
                shadowWVP[1] = rs->worldTransforms[0] * s_activeShadowVP[1];
            } else {
                shadowWVP[0] = rs->shadowWorldViewProj[0];
                shadowWVP[1] = rs->shadowWorldViewProj[1];
            }
            D3DXMATRIX transposed[2];
            D3DXMatrixTranspose(&transposed[0], &shadowWVP[0]);
            D3DXMatrixTranspose(&transposed[1], &shadowWVP[1]);
            cmdBuf->recordSetVSConstantF(hlslShader.regShadowWorldViewProj.reg, (float*)transposed, 8);
        }
    } else if (hlslShader.vsConstantTable) {
        try {
            if (hlslShader.hWorldViewProj) {
                hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hWorldViewProj, &worldViewProj);
            }

            if (hlslShader.hView) {
                hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hView, &viewMatrix);
            }

            if (hlslShader.hProj) {
                HRESULT hr = hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hProj, &projMatrix);
                if (FAILED(hr)) {
                    D3DXMATRIX projT;
                    D3DXMatrixTranspose(&projT, &projMatrix);
                    device->SetVertexShaderConstantF(hlslShader.projRegister, (float*)&projT, 4);
                    static bool projWarningLogged = false;
                    if (!projWarningLogged) {
                        LOG::logline("!! HLSL: Proj SetMatrix failed (hr=0x%08x), using direct register %d fallback", hr, hlslShader.projRegister);
                        projWarningLogged = true;
                    }
                }
            }

            if (hlslShader.hWorld) {
                hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hWorld, &rs->worldTransforms[0]);
            }

            if (hlslShader.hWorldView) {
                // For rigid draws (vbs=0), use precomputed worldViewTransforms[0] instead of replay-time worldView
                // The shader reads from worldview register (c4) for rigid path, not vertexBlendPalette
                if (rs->vertexBlendState == 0) {
                    hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hWorldView, &rs->worldViewTransforms[0]);
                } else {
                    hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hWorldView, &worldView);
                }
            }
            if (hlslShader.hVertexBlendPalette) {
                if (rs->vertexBlendState > 0) {
                    // Use precomputed worldViewTransforms from recording time
                    hlslShader.vsConstantTable->SetMatrixArray(device, hlslShader.hVertexBlendPalette, rs->worldViewTransforms, 4);
                } else {
                    // Rigid path: also use precomputed worldViewTransforms[0] for consistency
                    D3DXMATRIX blendMatrices[4];
                    blendMatrices[0] = rs->worldViewTransforms[0];
                    memset(&blendMatrices[1], 0, sizeof(D3DXMATRIX) * 3);
                    hlslShader.vsConstantTable->SetMatrixArray(device, hlslShader.hVertexBlendPalette, blendMatrices, 4);
                }
            }

            if (hlslShader.hVertexBlendState) {
                D3DXVECTOR4 blendState((float)rs->vertexBlendState, 0, 0, 0);
                hlslShader.vsConstantTable->SetVector(device, hlslShader.hVertexBlendState, &blendState);
            }

            if (hlslShader.hShadowWorldViewProj) {
                if (isReplaying.load(std::memory_order_acquire) && s_activeShadowVP) {
                    D3DXMATRIX currentShadowWVP[2];
                    currentShadowWVP[0] = rs->worldTransforms[0] * s_activeShadowVP[0];
                    currentShadowWVP[1] = rs->worldTransforms[0] * s_activeShadowVP[1];
                    hlslShader.vsConstantTable->SetMatrixArray(device, hlslShader.hShadowWorldViewProj, currentShadowWVP, 2);
                } else {
                    hlslShader.vsConstantTable->SetMatrixArray(device, hlslShader.hShadowWorldViewProj, rs->shadowWorldViewProj, 2);
                }
            }
        } catch (...) {
            LOG::logline("!! HLSL Vertex shader constant table access failed - shader may have been edited");
        }
    }

    if (sk.hasShadows) {
        D3DXMATRIX viewInverse;
        D3DXMatrixInverse(&viewInverse, nullptr, &viewMatrix);
        const auto* svp = s_activeShadowVP ? s_activeShadowVP : DistantLand::s_staging.smViewproj;
        for (int i = 0; i < kShadowCascadeCount; ++i) {
            D3DXMATRIX viewToShadow = viewInverse * svp[i];
            setMatrixConstantF(cmdBuf, device, false, 31 + i * 4, viewToShadow);
        }
        float shadowCascadeDepths[4];
        computeShadowCascadeDepths(shadowCascadeDepths);
        setConstantF(cmdBuf, device, false, 43, shadowCascadeDepths, 1);
        setCloseCascadeShadowParams(cmdBuf, device);
    }

    // Compute lighting data (shared between cmdBuf and device paths)
    const size_t MaxLights = 8;
    D3DXVECTOR4 bufferDiffuse[MaxLights];
    float bufferAmbient[MaxLights];
    float bufferPosition[3 * MaxLights];
    float bufferFalloffQuadratic[MaxLights], bufferFalloffLinear[MaxLights], bufferFalloffConstant;
    bool needPointLightBuffers = (sk.lightMode == 1 || sk.lightMode == 2);

    if (needPointLightBuffers) {
        memset(&bufferDiffuse, 0, sizeof(bufferDiffuse));
        memset(&bufferAmbient, 0, sizeof(bufferAmbient));
        memset(&bufferPosition, 0, sizeof(bufferPosition));
        memset(&bufferFalloffQuadratic, 0, sizeof(bufferFalloffQuadratic));
        memset(&bufferFalloffLinear, 0, sizeof(bufferFalloffLinear));
    }
    bufferFalloffConstant = 0.33;

    RGBVECTOR sunDiffuse(0, 0, 0), ambient = lightrs->globalAmbient;
    D3DVECTOR sunDirection = {0, 0, 1};
    size_t n = std::min(lightrs->active.size(), MaxLights), pointLightCount = 0;
    for (; n --> 0; ) {
        DWORD i = lightrs->active[n];
        const LightState::Light* light = &lightrs->lights.find(i)->second;

        if (lightrs->lightsTransformed.find(i) == lightrs->lightsTransformed.end()) {
            if (light->type == D3DLIGHT_DIRECTIONAL) {
                D3DXVec3TransformNormal((D3DXVECTOR3*)&light->viewspacePos, (D3DXVECTOR3*)&light->position, &rs->viewTransform);
            } else {
                D3DXVec3TransformCoord((D3DXVECTOR3*)&light->viewspacePos, (D3DXVECTOR3*)&light->position, &rs->viewTransform);
            }
            lightrs->lightsTransformed[i] = true;
        }

        if (light->type == D3DLIGHT_POINT && needPointLightBuffers) {
            memcpy(&bufferDiffuse[pointLightCount], &light->diffuse, sizeof(light->diffuse));
            bufferPosition[pointLightCount] = light->viewspacePos.x;
            bufferPosition[pointLightCount + MaxLights] = light->viewspacePos.y;
            bufferPosition[pointLightCount + 2*MaxLights] = light->viewspacePos.z;

            if (light->falloff.x > 0) {
                bufferFalloffConstant = light->falloff.x;
                bufferFalloffLinear[pointLightCount] = light->falloff.y;
                bufferFalloffQuadratic[pointLightCount] = light->falloff.z;
            } else if (light->falloff.z > 0) {
                bufferDiffuse[pointLightCount].x *= bufferFalloffConstant;
                bufferDiffuse[pointLightCount].y *= bufferFalloffConstant;
                bufferDiffuse[pointLightCount].z *= bufferFalloffConstant;
                bufferAmbient[pointLightCount] = 1.0f + 1e-4f / sqrt(light->falloff.z);
                bufferFalloffQuadratic[pointLightCount] = bufferFalloffConstant * light->falloff.z;
            } else if (light->falloff.y == 0.10000001f) {
                bufferFalloffQuadratic[pointLightCount] = 5e-5;
            } else if (light->falloff.y > 0) {
                float brightness = 0.25f + 1e-4f / light->falloff.y;
                bufferDiffuse[pointLightCount].x = brightness;
                bufferDiffuse[pointLightCount].y = brightness;
                bufferDiffuse[pointLightCount].z = brightness;
                bufferAmbient[pointLightCount] = 1.0;
                bufferFalloffQuadratic[pointLightCount] = 0.5555f * light->falloff.y * light->falloff.y;
                bufferPosition[pointLightCount + 2*MaxLights] += 25.0;
            }
            ++pointLightCount;
        } else if (light->type == D3DLIGHT_DIRECTIONAL) {
            sunDiffuse = light->diffuse;
            sunDirection = light->viewspacePos;
            ambient.r += light->ambient.x;
            ambient.g += light->ambient.y;
            ambient.b += light->ambient.z;
        }
    }

    sunDiffuse *= sunMultiplier;
    ambient *= ambMultiplier;

    // Check full-bright ambient (Morrowind particle effect mode)
    // Use captured state to avoid reading stale device values during replay
    DWORD checkAmbient;
    if (capturedState) {
        checkAmbient = capturedState->ambient;
    } else {
        device->GetRenderState(D3DRS_AMBIENT, &checkAmbient);
    }
    if (checkAmbient == 0xffffffff) {
        ambient.r = ambient.g = ambient.b = 1.25;
        sunDiffuse.r = sunDiffuse.g = sunDiffuse.b = 0.0;
    }

    // Get fog color from staging (per-frame, not per-draw)
    // Using staged value avoids device read during replay
    float fogColor[4] = {
        DistantLand::s_staging.nearFogCol.r,
        DistantLand::s_staging.nearFogCol.g,
        DistantLand::s_staging.nearFogCol.b,
        1.0f
    };

    // Set pixel shader constants
    if (cmdBuf) {
        // Command buffer path: use resolved registers, write to cmdBuf
        // Resolve dynamic constants on first use for this shader variant
        if (!hlslShader.dynamicConstsResolved && hlslShader.psConstantTable && hlslShader.vsConstantTable) {
            resolveAndCache(hlslShader.psConstantTable, "shadingMode", hlslShader.regShadingMode);
            resolveAndCache(hlslShader.psConstantTable, "fogColNear", hlslShader.regFogColNear);
            resolveAndCache(hlslShader.psConstantTable, "materialAlpha", hlslShader.regMaterialAlpha);
            resolveAndCache(hlslShader.psConstantTable, "alphaRef", hlslShader.regAlphaRef);
            resolveAndCache(hlslShader.psConstantTable, "hasVCol", hlslShader.regHasVCol);
            resolveAndCache(hlslShader.psConstantTable, "hasAlpha", hlslShader.regHasAlpha);
            resolveAndCache(hlslShader.vsConstantTable, "hasBones", hlslShader.regHasBones);
            resolveAndCache(hlslShader.vsConstantTable, "hasAlpha", hlslShader.regHasAlphaVS);
            resolveAndCache(hlslShader.vsConstantTable, "texgenTransform", hlslShader.regTexgenTransform);
            resolveAndCache(hlslShader.psConstantTable, "bumpMatrix", hlslShader.regBumpMatrix);
            resolveAndCache(hlslShader.psConstantTable, "bumpLumiScaleBias", hlslShader.regBumpLumiScaleBias);
            resolveAndCache(hlslShader.psConstantTable, "PCF_penumbraScale", hlslShader.regPCFPenumbraScale);
            resolveAndCache(hlslShader.psConstantTable, "PCF_minPenumbra", hlslShader.regPCFMinPenumbra);
            resolveAndCache(hlslShader.psConstantTable, "PCF_maxPenumbra", hlslShader.regPCFMaxPenumbra);
            resolveAndCache(hlslShader.psConstantTable, "PCF_bias", hlslShader.regPCFBias);
            resolveAndCache(hlslShader.psConstantTable, "PCF_bias2", hlslShader.regPCFBias2);
            resolveAndCache(hlslShader.psConstantTable, "PCF_slopeBias", hlslShader.regPCFSlopeBias);
            resolveAndCache(hlslShader.psConstantTable, "terrainShadowParams", hlslShader.regTerrainShadowParams);
            resolveAndCache(hlslShader.vsConstantTable, "windVec", hlslShader.regWindVec);
            resolveAndCache(hlslShader.vsConstantTable, "time", hlslShader.regTime);
            resolveAndCache(hlslShader.psConstantTable, "normres", hlslShader.regNormres);
            resolveAndCache(hlslShader.psConstantTable, "debugMode", hlslShader.regDebugMode);
            resolveAndCache(hlslShader.psConstantTable, "intensityScalar", hlslShader.regIntensityScalar);
            resolveAndCache(hlslShader.psConstantTable, "attenuationMultiplier", hlslShader.regAttenuationMultiplier);
            resolveAndCache(hlslShader.psConstantTable, "attenuationCutoffDist", hlslShader.regAttenuationCutoffDist);
            resolveAndCache(hlslShader.psConstantTable, "parallaxScale", hlslShader.regParallaxScale);
            resolveAndCache(hlslShader.psConstantTable, "parallaxBias", hlslShader.regParallaxBias);
            hlslShader.dynamicConstsResolved = true;
        }

        // Material constants
        if (hlslShader.regMaterialDiffuse.reg != REG_INVALID)
            cmdBuf->recordSetPSConstantF(hlslShader.regMaterialDiffuse.reg, (float*)&frs->material.diffuse, 1);
        if (hlslShader.regMaterialAmbient.reg != REG_INVALID)
            cmdBuf->recordSetPSConstantF(hlslShader.regMaterialAmbient.reg, (float*)&frs->material.ambient, 1);
        if (hlslShader.regMaterialEmissive.reg != REG_INVALID)
            cmdBuf->recordSetPSConstantF(hlslShader.regMaterialEmissive.reg, (float*)&frs->material.emissive, 1);

        // Sun + ambient
        if (hlslShader.regLightSunDirection.reg != REG_INVALID) {
            float v[4] = { sunDirection.x, sunDirection.y, sunDirection.z, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regLightSunDirection.reg, v, 1);
        }
        if (hlslShader.regLightSunDiffuse.reg != REG_INVALID) {
            float v[4] = { sunDiffuse.r, sunDiffuse.g, sunDiffuse.b, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regLightSunDiffuse.reg, v, 1);
        }
        if (hlslShader.regLightSceneAmbient.reg != REG_INVALID) {
            float v[4] = { ambient.r, ambient.g, ambient.b, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regLightSceneAmbient.reg, v, 1);
        }
        // Point lights
        if (needPointLightBuffers) {
            if (hlslShader.regLightDiffuse.reg != REG_INVALID)
                cmdBuf->recordSetPSConstantF(hlslShader.regLightDiffuse.reg, (float*)bufferDiffuse, MaxLights);
            if (hlslShader.regLightPosition.reg != REG_INVALID) {
                // HLSL float3[8] = 8 registers, each float3 padded to float4
                float hlslLightPositions[MaxLights * 4];
                for (int i = 0; i < (int)MaxLights; i++) {
                    hlslLightPositions[i * 4 + 0] = bufferPosition[i];
                    hlslLightPositions[i * 4 + 1] = bufferPosition[i + MaxLights];
                    hlslLightPositions[i * 4 + 2] = bufferPosition[i + 2*MaxLights];
                    hlslLightPositions[i * 4 + 3] = 0.0f;
                }
                cmdBuf->recordSetPSConstantF(hlslShader.regLightPosition.reg, hlslLightPositions, MaxLights);
            }
            if (hlslShader.regLightAmbient.reg != REG_INVALID) {
                // HLSL float[8] = 8 registers, each scalar padded to float4
                float paddedAmbient[MaxLights * 4];
                for (int i = 0; i < (int)MaxLights; i++) {
                    paddedAmbient[i * 4 + 0] = bufferAmbient[i];
                    paddedAmbient[i * 4 + 1] = 0.0f;
                    paddedAmbient[i * 4 + 2] = 0.0f;
                    paddedAmbient[i * 4 + 3] = 0.0f;
                }
                cmdBuf->recordSetPSConstantF(hlslShader.regLightAmbient.reg, paddedAmbient, MaxLights);
            }
            if (hlslShader.regPointLightCount.reg != REG_INVALID) {
                // D3D9 loop instruction format: {count, initialValue, step, 0}
                // D3DXRS_INT4 = 1
                if (hlslShader.regPointLightCount.regSet == 1) {
                    int iv[4] = { (int)pointLightCount, 0, 1, 0 };
                    cmdBuf->recordSetPSConstantI(hlslShader.regPointLightCount.reg, iv, 1);
                } else {
                    float v[4] = { (float)(int)pointLightCount, 0, 0, 0 };
                    cmdBuf->recordSetPSConstantF(hlslShader.regPointLightCount.reg, v, 1);
                }
            }
            if (hlslShader.regLightFalloffQuadratic.reg != REG_INVALID) {
                // HLSL float[8] = 8 registers, each scalar padded to float4
                float paddedQuadratic[MaxLights * 4];
                for (int i = 0; i < (int)MaxLights; i++) {
                    paddedQuadratic[i * 4 + 0] = ((size_t)i < pointLightCount) ? bufferFalloffQuadratic[i] : 0.0f;
                    paddedQuadratic[i * 4 + 1] = 0.0f;
                    paddedQuadratic[i * 4 + 2] = 0.0f;
                    paddedQuadratic[i * 4 + 3] = 0.0f;
                }
                cmdBuf->recordSetPSConstantF(hlslShader.regLightFalloffQuadratic.reg, paddedQuadratic, MaxLights);
            }
            if (hlslShader.regLightFalloffConstant.reg != REG_INVALID) {
                float v[4] = { bufferFalloffConstant, 0, 0, 0 };
                cmdBuf->recordSetPSConstantF(hlslShader.regLightFalloffConstant.reg, v, 1);
            }
        }

        // Per-object light texture parameters for mode 3
        if (sk.lightMode == 3 && callIndex >= 0 && callIndex < (int)perObjectLightInfo.size()) {
            const auto& li = perObjectLightInfo[callIndex];
            float lightParams[4] = { (float)li.lightCount, perObjectTexelSize, (float)li.texelOffset, 0.0f };
            cmdBuf->recordSetPSConstantF(50, lightParams, 1);
        }

        // Dynamic PS constants
        if (hlslShader.regShadingMode.reg != REG_INVALID) {
            float v[4] = { 0, 0, (float)sk.vertexMaterial, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regShadingMode.reg, v, 1);
        }
        if (hlslShader.regShadowRcpRes.reg != REG_INVALID) {
            float v[4] = { 1.0f / Configuration.DL.ShadowResolution, 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regShadowRcpRes.reg, v, 1);
        }
        if (hlslShader.regPCFFilterSize.reg != REG_INVALID) {
            float v[4] = { ImGuiManager::GetPCFFilterSize(), 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regPCFFilterSize.reg, v, 1);
        }
        if (hlslShader.regPCFPenumbraScale.reg != REG_INVALID) {
            float v[4] = { ImGuiManager::GetPCFPenumbraScale(), 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regPCFPenumbraScale.reg, v, 1);
        }
        if (hlslShader.regPCFMinPenumbra.reg != REG_INVALID) {
            float v[4] = { ImGuiManager::GetPCFMinPenumbra(), 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regPCFMinPenumbra.reg, v, 1);
        }
        if (hlslShader.regPCFMaxPenumbra.reg != REG_INVALID) {
            float v[4] = { ImGuiManager::GetPCFMaxPenumbra(), 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regPCFMaxPenumbra.reg, v, 1);
        }
        if (hlslShader.regPCFBias.reg != REG_INVALID) {
            float v[4] = { ImGuiManager::GetPCFBias(), 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regPCFBias.reg, v, 1);
        }
        if (hlslShader.regPCFBias2.reg != REG_INVALID) {
            float v[4] = { ImGuiManager::GetPCFBias2(), 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regPCFBias2.reg, v, 1);
        }
        if (hlslShader.regPCFSlopeBias.reg != REG_INVALID) {
            float v[4] = { ImGuiManager::GetPCFSlopeBias(), 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regPCFSlopeBias.reg, v, 1);
        }
        // Terrain shadow params. terrainShadowParams is pinned to PS c25, so use
        // that as a fallback for paths whose dynamic register cache is still cold.
        if (sk.hasShadows) {
            float isTerrain = (replayCall && replayCall->bin == RenderBin::Terrain) ? 1.0f : 0.0f;
            setTerrainShadowParams(cmdBuf, device, hlslShader.regTerrainShadowParams, isTerrain);
        }
        if (hlslShader.regDebugMode.reg != REG_INVALID) {
            float v[4] = { (float)ImGuiManager::GetShaderDebugMode(), 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regDebugMode.reg, v, 1);
        }
        cmdBuf->recordSetPSConstantF(hlslShader.regFogColNear.reg != REG_INVALID ? hlslShader.regFogColNear.reg : 255, fogColor, 1);
        if (hlslShader.regIntensityScalar.reg != REG_INVALID) {
            float v[4] = { ImGuiManager::GetIntensityScalar(), 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regIntensityScalar.reg, v, 1);
        }
        if (hlslShader.regAttenuationMultiplier.reg != REG_INVALID) {
            float v[4] = { ImGuiManager::GetAttenuationMultiplier(), 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regAttenuationMultiplier.reg, v, 1);
        }
        if (hlslShader.regAttenuationCutoffDist.reg != REG_INVALID) {
            float v[4] = { ImGuiManager::GetAttenuationCutoffDist(), 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regAttenuationCutoffDist.reg, v, 1);
        }
        if (hlslShader.regParallaxScale.reg != REG_INVALID) {
            float v[4] = { ImGuiManager::GetParallaxScale(), 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regParallaxScale.reg, v, 1);
        }
        if (hlslShader.regParallaxBias.reg != REG_INVALID) {
            float v[4] = { ImGuiManager::GetParallaxBias(), 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regParallaxBias.reg, v, 1);
        }

        // VS dynamic constants
        if (hlslShader.regTexgenTransform.reg != REG_INVALID) {
            D3DXMATRIX identity;
            D3DXMatrixIdentity(&identity);
            setMatrixConstantF(cmdBuf, device, true, hlslShader.regTexgenTransform.reg, identity);
        }
        if (hlslShader.regHasBones.reg != REG_INVALID) {
            float v[4] = { rs->vertexBlendState > 0 ? 1.0f : 0.0f, 0, 0, 0 };
            cmdBuf->recordSetVSConstantF(hlslShader.regHasBones.reg, v, 1);
        }
        if (hlslShader.regHasAlphaVS.reg != REG_INVALID) {
            float v[4] = { rs->alphaTest ? 1.0f : 0.0f, 0, 0, 0 };
            cmdBuf->recordSetVSConstantF(hlslShader.regHasAlphaVS.reg, v, 1);
        }
        if (hlslShader.regWindVec.reg != REG_INVALID) {
            const float* wind = replayCall ? replayCall->windVec : DistantLand::s_staging.windVec;
            float v[4] = { wind[0], wind[1], 0, 0 };
            cmdBuf->recordSetVSConstantF(hlslShader.regWindVec.reg, v, 1);
        }
        if (hlslShader.regTime.reg != REG_INVALID) {
            float time = replayCall ? replayCall->simulationTime : DistantLand::s_staging.simulationTime;
            float v[4] = { time, 0, 0, 0 };
            cmdBuf->recordSetVSConstantF(hlslShader.regTime.reg, v, 1);
        }

        // PS misc constants
        if (hlslShader.regBumpMatrix.reg != REG_INVALID) {
            float v[4] = { 1, 0, 0, 1 };
            cmdBuf->recordSetPSConstantF(hlslShader.regBumpMatrix.reg, v, 1);
        }
        if (hlslShader.regBumpLumiScaleBias.reg != REG_INVALID) {
            float v[4] = { 1, 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regBumpLumiScaleBias.reg, v, 1);
        }
        if (hlslShader.regHasAlpha.reg != REG_INVALID) {
            float v[4] = { 0, 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regHasAlpha.reg, v, 1);
        }
        if (hlslShader.regHasVCol.reg != REG_INVALID) {
            float v[4] = { (rs->fvf & D3DFVF_DIFFUSE) ? 1.0f : 0.0f, 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regHasVCol.reg, v, 1);
        }
        if (hlslShader.regMaterialAlpha.reg != REG_INVALID) {
            float v[4] = { frs->material.diffuse.a, 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regMaterialAlpha.reg, v, 1);
        }
        if (hlslShader.regAlphaRef.reg != REG_INVALID) {
            float v[4] = { rs->alphaRef / 255.0f, 0, 0, 0 };
            cmdBuf->recordSetPSConstantF(hlslShader.regAlphaRef.reg, v, 1);
        }

        // Normres for paramH textures — read dims the suffix cache captured at load.
        // Reading slot 2 via GetTexture is unreliable: bindShaderTextures skips rebinding
        // when rs->texture is unchanged, so slot 2 can carry a foreign texture across draws.
        if (sk.hasParamH && hlslShader.regNormres.reg != REG_INVALID) {
            const auto* r = TextureSuffix::getCachedResolution(rs->texture);
            float v[4] = { r ? r->paramHWidth : 0.0f, r ? r->paramHHeight : 0.0f, 0.0f, 0.0f };
            cmdBuf->recordSetPSConstantF(hlslShader.regNormres.reg, v, 1);
        }
    } else if (hlslShader.psConstantTable) {
        // Device path: use constant tables (SetMatrix transposes internally)
        try {
            if (hlslShader.hMaterialDiffuse) {
                hlslShader.psConstantTable->SetVector(device, hlslShader.hMaterialDiffuse, (D3DXVECTOR4*)&frs->material.diffuse);
            }
            if (hlslShader.hMaterialAmbient) {
                hlslShader.psConstantTable->SetVector(device, hlslShader.hMaterialAmbient, (D3DXVECTOR4*)&frs->material.ambient);
            }
            if (hlslShader.hMaterialEmissive) {
                hlslShader.psConstantTable->SetVector(device, hlslShader.hMaterialEmissive, (D3DXVECTOR4*)&frs->material.emissive);
            }

        // Sun + ambient constants (all modes)
        if (hlslShader.hLightSunDirection) {
            hlslShader.psConstantTable->SetFloatArray(device, hlslShader.hLightSunDirection, (const float*)&sunDirection, 3);
        }
        if (hlslShader.hLightSunDiffuse) {
            hlslShader.psConstantTable->SetFloatArray(device, hlslShader.hLightSunDiffuse, (const float*)&sunDiffuse, 3);
        }
        if (hlslShader.hLightSceneAmbient) {
            hlslShader.psConstantTable->SetFloatArray(device, hlslShader.hLightSceneAmbient, (const float*)&ambient, 3);
        }
        // Point light uniforms — only for lightMode 1 (single) and 2 (few loop)
        if (needPointLightBuffers) {
            if (hlslShader.hLightDiffuse)
                hlslShader.psConstantTable->SetVectorArray(device, hlslShader.hLightDiffuse, bufferDiffuse, MaxLights);
            if (hlslShader.hLightPosition) {
                D3DXVECTOR3 hlslLightPositions[MaxLights];
                for (int i = 0; i < (int)MaxLights; i++) {
                    hlslLightPositions[i].x = bufferPosition[i];
                    hlslLightPositions[i].y = bufferPosition[i + MaxLights];
                    hlslLightPositions[i].z = bufferPosition[i + 2*MaxLights];
                }
                hlslShader.psConstantTable->SetFloatArray(device, hlslShader.hLightPosition, (float*)hlslLightPositions, 3 * MaxLights);
            }
            if (hlslShader.hLightAmbient)
                hlslShader.psConstantTable->SetFloatArray(device, hlslShader.hLightAmbient, bufferAmbient, MaxLights);
            if (hlslShader.hPointLightCount)
                hlslShader.psConstantTable->SetInt(device, hlslShader.hPointLightCount, (int)pointLightCount);
            if (hlslShader.hLightFalloffQuadratic) {
                D3DXVECTOR4 quadraticData[2];
                for (int i = 0; i < 4; i++) {
                    quadraticData[0][i] = ((size_t)i < pointLightCount) ? bufferFalloffQuadratic[i] : 0.0f;
                    quadraticData[1][i] = ((size_t)(i + 4) < pointLightCount) ? bufferFalloffQuadratic[i + 4] : 0.0f;
                }
                hlslShader.psConstantTable->SetVectorArray(device, hlslShader.hLightFalloffQuadratic, quadraticData, 2);
            }
            if (hlslShader.hLightFalloffConstant)
                hlslShader.psConstantTable->SetFloat(device, hlslShader.hLightFalloffConstant, bufferFalloffConstant);
        }

        // Per-object light texture parameters for mode 3 (saturated objects)
        if (sk.lightMode == 3 && callIndex >= 0 && callIndex < (int)perObjectLightInfo.size()) {
            const auto& li = perObjectLightInfo[callIndex];
            float lightParams[4] = { (float)li.lightCount, perObjectTexelSize, (float)li.texelOffset, 0.0f };
            device->SetPixelShaderConstantF(50, lightParams, 1);
        }

        D3DXHANDLE hShadingMode = hlslShader.psConstantTable->GetConstantByName(NULL, "shadingMode");
        if (hShadingMode) {
            float shadingModeData[4] = {0, 0, (float)sk.vertexMaterial, 0};
            hlslShader.psConstantTable->SetFloatArray(device, hShadingMode, shadingModeData, 4);
        }
        if (hlslShader.hShadowRcpRes) {
            float shadowRcp = 1.0f / Configuration.DL.ShadowResolution;
            hlslShader.psConstantTable->SetFloat(device, hlslShader.hShadowRcpRes, shadowRcp);
        }
        if (hlslShader.hPCFFilterSize)
            hlslShader.psConstantTable->SetFloat(device, hlslShader.hPCFFilterSize, ImGuiManager::GetPCFFilterSize());
        D3DXHANDLE hPCFPenumbraScale = hlslShader.psConstantTable->GetConstantByName(NULL, "PCF_penumbraScale");
        if (hPCFPenumbraScale) hlslShader.psConstantTable->SetFloat(device, hPCFPenumbraScale, ImGuiManager::GetPCFPenumbraScale());
        D3DXHANDLE hPCFMinPenumbra = hlslShader.psConstantTable->GetConstantByName(NULL, "PCF_minPenumbra");
        if (hPCFMinPenumbra) hlslShader.psConstantTable->SetFloat(device, hPCFMinPenumbra, ImGuiManager::GetPCFMinPenumbra());
        D3DXHANDLE hPCFMaxPenumbra = hlslShader.psConstantTable->GetConstantByName(NULL, "PCF_maxPenumbra");
        if (hPCFMaxPenumbra) hlslShader.psConstantTable->SetFloat(device, hPCFMaxPenumbra, ImGuiManager::GetPCFMaxPenumbra());
        D3DXHANDLE hPCFBias = hlslShader.psConstantTable->GetConstantByName(NULL, "PCF_bias");
        if (hPCFBias) hlslShader.psConstantTable->SetFloat(device, hPCFBias, ImGuiManager::GetPCFBias());
        D3DXHANDLE hPCFBias2 = hlslShader.psConstantTable->GetConstantByName(NULL, "PCF_bias2");
        if (hPCFBias2) hlslShader.psConstantTable->SetFloat(device, hPCFBias2, ImGuiManager::GetPCFBias2());
        D3DXHANDLE hPCFSlopeBias = hlslShader.psConstantTable->GetConstantByName(NULL, "PCF_slopeBias");
        if (hPCFSlopeBias) hlslShader.psConstantTable->SetFloat(device, hPCFSlopeBias, ImGuiManager::GetPCFSlopeBias());
        // Terrain shadow params. terrainShadowParams is pinned to PS c25, so use
        // that as a fallback for paths whose dynamic register cache is still cold.
        if (sk.hasShadows) {
            float isTerrain = (replayCall && replayCall->bin == RenderBin::Terrain) ? 1.0f : 0.0f;
            setTerrainShadowParams(nullptr, device, hlslShader.regTerrainShadowParams, isTerrain);
        }
        D3DXHANDLE hDebugMode = hlslShader.psConstantTable->GetConstantByName(NULL, "debugMode");
        if (hDebugMode) hlslShader.psConstantTable->SetInt(device, hDebugMode, ImGuiManager::GetShaderDebugMode());
        D3DXHANDLE hHeightBlendParams = hlslShader.psConstantTable->GetConstantByName(NULL, "heightBlendParams");
        if (hHeightBlendParams) {
            D3DXVECTOR4 hb(ImGuiManager::GetHeightBlendStrength(),
                           ImGuiManager::GetHeightBlendContrast(), 0.0f, 0.0f);
            hlslShader.psConstantTable->SetVector(device, hHeightBlendParams, &hb);
        }

        D3DXHANDLE hFogColNear = hlslShader.psConstantTable->GetConstantByName(NULL, "fogColNear");
        if (hFogColNear) hlslShader.psConstantTable->SetVector(device, hFogColNear, (D3DXVECTOR4*)fogColor);
        D3DXHANDLE hIntensityScalar = hlslShader.psConstantTable->GetConstantByName(NULL, "intensityScalar");
        if (hIntensityScalar) hlslShader.psConstantTable->SetFloat(device, hIntensityScalar, ImGuiManager::GetIntensityScalar());
        D3DXHANDLE hAttenuationMultiplier = hlslShader.psConstantTable->GetConstantByName(NULL, "attenuationMultiplier");
        if (hAttenuationMultiplier) hlslShader.psConstantTable->SetFloat(device, hAttenuationMultiplier, ImGuiManager::GetAttenuationMultiplier());
        D3DXHANDLE hAttenuationCutoffDist = hlslShader.psConstantTable->GetConstantByName(NULL, "attenuationCutoffDist");
        if (hAttenuationCutoffDist) hlslShader.psConstantTable->SetFloat(device, hAttenuationCutoffDist, ImGuiManager::GetAttenuationCutoffDist());
        D3DXHANDLE hParallaxScale = hlslShader.psConstantTable->GetConstantByName(NULL, "parallaxScale");
        if (hParallaxScale) hlslShader.psConstantTable->SetFloat(device, hParallaxScale, ImGuiManager::GetParallaxScale());
        D3DXHANDLE hParallaxBias = hlslShader.psConstantTable->GetConstantByName(NULL, "parallaxBias");
        if (hParallaxBias) hlslShader.psConstantTable->SetFloat(device, hParallaxBias, ImGuiManager::GetParallaxBias());

        D3DXHANDLE hTexgenTransform = hlslShader.vsConstantTable->GetConstantByName(NULL, "texgenTransform");
        if (hTexgenTransform) {
            D3DXMATRIX identity;
            D3DXMatrixIdentity(&identity);
            hlslShader.vsConstantTable->SetMatrix(device, hTexgenTransform, &identity);
        }
        D3DXHANDLE hBumpMatrix = hlslShader.psConstantTable->GetConstantByName(NULL, "bumpMatrix");
        if (hBumpMatrix) {
            D3DXVECTOR4 bumpMatrix(1, 0, 0, 1);
            hlslShader.psConstantTable->SetVector(device, hBumpMatrix, &bumpMatrix);
        }
        D3DXHANDLE hBumpLumiScaleBias = hlslShader.psConstantTable->GetConstantByName(NULL, "bumpLumiScaleBias");
        if (hBumpLumiScaleBias) {
            D3DXVECTOR2 scaleBias(1, 0);
            hlslShader.psConstantTable->SetFloatArray(device, hBumpLumiScaleBias, (float*)&scaleBias, 2);
        }
        D3DXHANDLE hHasAlpha = hlslShader.psConstantTable->GetConstantByName(NULL, "hasAlpha");
        if (hHasAlpha) hlslShader.psConstantTable->SetBool(device, hHasAlpha, false);
        D3DXHANDLE hHasBones = hlslShader.vsConstantTable->GetConstantByName(NULL, "hasBones");
        if (hHasBones) hlslShader.vsConstantTable->SetBool(device, hHasBones, rs->vertexBlendState > 0);
        D3DXHANDLE hHasAlphaVS = hlslShader.vsConstantTable->GetConstantByName(NULL, "hasAlpha");
        if (hHasAlphaVS) hlslShader.vsConstantTable->SetBool(device, hHasAlphaVS, rs->alphaTest);
        D3DXHANDLE hWindVec = hlslShader.vsConstantTable->GetConstantByName(NULL, "windVec");
        if (hWindVec) {
            const float* wind = replayCall ? replayCall->windVec : DistantLand::s_staging.windVec;
            hlslShader.vsConstantTable->SetFloatArray(device, hWindVec, wind, 2);
        }
        D3DXHANDLE hTime = hlslShader.vsConstantTable->GetConstantByName(NULL, "time");
        if (hTime) {
            float time = replayCall ? replayCall->simulationTime : DistantLand::s_staging.simulationTime;
            hlslShader.vsConstantTable->SetFloat(device, hTime, time);
        }
        D3DXHANDLE hHasVCol = hlslShader.psConstantTable->GetConstantByName(NULL, "hasVCol");
        if (hHasVCol) hlslShader.psConstantTable->SetBool(device, hHasVCol, (rs->fvf & D3DFVF_DIFFUSE) != 0);
        D3DXHANDLE hMaterialAlpha = hlslShader.psConstantTable->GetConstantByName(NULL, "materialAlpha");
        if (hMaterialAlpha) hlslShader.psConstantTable->SetFloat(device, hMaterialAlpha, frs->material.diffuse.a);
        D3DXHANDLE hAlphaRef = hlslShader.psConstantTable->GetConstantByName(NULL, "alphaRef");
        if (hAlphaRef) hlslShader.psConstantTable->SetFloat(device, hAlphaRef, rs->alphaRef / 255.0f);

        if (sk.hasParamH) {
            const auto* r = TextureSuffix::getCachedResolution(rs->texture);
            float v[2] = { r ? r->paramHWidth : 0.0f, r ? r->paramHHeight : 0.0f };
            D3DXHANDLE hNormres = hlslShader.psConstantTable->GetConstantByName(NULL, "normres");
            if (hNormres) hlslShader.psConstantTable->SetFloatArray(device, hNormres, v, 2);
        }
        } catch (...) {
            LOG::logline("!! HLSL Pixel shader constant table access failed");
        }
    }

    if (!rs->vb) {
        return;
    }

    if (cmdBuf) {
        // Depth bias
        cmdBuf->recordSetRenderState(D3DRS_DEPTHBIAS, *(DWORD*)&(const float&)-1e-6f);
        cmdBuf->recordSetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, *(DWORD*)&(const float&)-1e-6f);

        // Draw call
        if (rs->ib) {
            cmdBuf->recordDrawIndexedPrimitive(rs->primType, rs->baseIndex, rs->minIndex, rs->vertCount, rs->startIndex, rs->primCount);
        } else {
            cmdBuf->recordDrawPrimitive(rs->primType, rs->startIndex, rs->primCount);
        }

        // Reset depth bias
        cmdBuf->recordSetRenderState(D3DRS_DEPTHBIAS, 0);
        cmdBuf->recordSetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, 0);

        // Restore shaders (per-draw cleanup)
        cmdBuf->recordSetVertexShader(NULL);
        cmdBuf->recordSetPixelShader(NULL);
    } else {
        HRESULT hr = device->SetFVF(rs->fvf);
        if (FAILED(hr)) {
            return;
        }

        // Use staging buffer for Scene 1/2 snapshot draws, original VB for Scene 0
        // Select buffer based on mode: N-1 uses prepBuffer, N-2 uses renderBuffer
        auto& fbStaging = usingN1Buffer ? getPrepBuffer() : getRenderingBuffer();
        bool useStagingBuffer = replayCall && replayCall->usesSnapshot &&
                                fbStaging.particleStagingVB && fbStaging.particleStagingIB;

        if (useStagingBuffer) {
            hr = device->SetStreamSource(0, fbStaging.particleStagingVB, replayCall->stagingVBOffset, rs->vbStride);
        } else {
            hr = device->SetStreamSource(0, rs->vb, rs->vbOffset, rs->vbStride);
        }
        if (FAILED(hr)) {
            return;
        }

        // Phase A: Add small depth bias to resolve Z-fighting with depth prepass
        device->SetRenderState(D3DRS_DEPTHBIAS, *(DWORD*)&(const float&)-1e-6f);
        device->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, *(DWORD*)&(const float&)-1e-6f);

        if (rs->ib) {
            if (useStagingBuffer) {
                // Staging buffer path: IB offset is byte offset, convert to index offset
                D3DINDEXBUFFER_DESC ibDesc;
                fbStaging.particleStagingIB->GetDesc(&ibDesc);
                UINT indexSize = (ibDesc.Format == D3DFMT_INDEX32) ? 4 : 2;
                UINT startIndexOffset = replayCall->stagingIBOffset / indexSize;

                hr = device->SetIndices(fbStaging.particleStagingIB);
                if (FAILED(hr)) {
                    LOG::logline("!! HLSL pipeline: failed to set staging index buffer, hr=%x", hr);
                    device->SetRenderState(D3DRS_DEPTHBIAS, 0);
                    device->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, 0);
                    return;
                }
                // For staging buffer: baseIndex=0 (data is isolated), startIndex=offset into staging IB
                device->DrawIndexedPrimitive(rs->primType, 0, 0, rs->vertCount, startIndexOffset, rs->primCount);
            } else {
                hr = device->SetIndices(rs->ib);
                if (FAILED(hr)) {
                    LOG::logline("!! HLSL pipeline: failed to set index buffer, hr=%x", hr);
                    device->SetRenderState(D3DRS_DEPTHBIAS, 0);
                    device->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, 0);
                    return;
                }
                device->DrawIndexedPrimitive(rs->primType, rs->baseIndex, rs->minIndex, rs->vertCount, rs->startIndex, rs->primCount);
            }
        } else {
            device->DrawPrimitive(rs->primType, rs->startIndex, rs->primCount);
        }

        // Reset depth bias after drawing
        device->SetRenderState(D3DRS_DEPTHBIAS, 0);
        device->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, 0);

        // Restore device state after HLSL rendering
        device->SetVertexShader(NULL);
        device->SetPixelShader(NULL);

        // During replay, texture slots are managed by bindShaderTextures
        if (!isReplaying.load(std::memory_order_acquire)) {
            FixedFunctionShader::setCachedTexture(device, 2, nullptr);
            FixedFunctionShader::setCachedTexture(device, 3, nullptr);
            FixedFunctionShader::setCachedTexture(device, 4, nullptr);

            device->SetRenderState(D3DRS_ALPHABLENDENABLE, savedAlphaBlendEnable);
            device->SetRenderState(D3DRS_ALPHATESTENABLE, savedAlphaTestEnable);
            device->SetRenderState(D3DRS_ZENABLE, savedZEnable);
            device->SetRenderState(D3DRS_ZWRITEENABLE, savedZWriteEnable);
            device->SetRenderState(D3DRS_SPECULARENABLE, savedSpecularEnable);
            device->SetRenderState(D3DRS_LOCALVIEWER, savedLocalViewer);
            device->SetRenderState(D3DRS_NORMALIZENORMALS, savedNormalizeNormals);
            device->SetRenderState(D3DRS_AMBIENTMATERIALSOURCE, savedAmbientMatSrc);
            device->SetRenderState(D3DRS_DIFFUSEMATERIALSOURCE, savedDiffuseMatSrc);
            device->SetRenderState(D3DRS_EMISSIVEMATERIALSOURCE, savedEmissiveMatSrc);
        }
    }

    // Phase 6A: release sampler s8 if we bound an overlay on this call so the slot
    // doesn't leak into subsequent draws that don't have HAS_OVERLAY.
    if (boundOverlay) {
        device->SetTexture(8, nullptr);
    }
}

void FixedFunctionShader::replayRecordedCalls(int sceneCount, D3DCommandBuffer* cmdBuf) {
    // Increment frame counter on Scene 0 replay (once per frame)
    if (sceneCount == 0) {
        hlslDiagFrameCounter++;
    }

    // Select buffer based on mode: N-1 uses prepBuffer, N-2 uses renderBuffer
    auto& fb = usingN1Buffer ? getPrepBuffer() : getRenderingBuffer();
    std::vector<HLSLRecordedCall>* recCallsPtr;
    if (sceneCount == 1) {
        recCallsPtr = &fb.recordedCallsScene1;
    } else if (sceneCount >= 2) {
        recCallsPtr = &fb.recordedCallsScene2;
    } else {
        recCallsPtr = &fb.recordedCalls;
    }
    auto& recCalls = *recCallsPtr;

    {
        if (recCalls.empty()) {
            return;
        }

        ImGuiManager::LogFrameEvent(FrameEvent::MGE_HLSLReplay, sceneCount, (int)recCalls.size());

        // Check if replay is disabled via ImGui
        if (!ImGuiManager::GetEnableReplay()) {
            return;
        }
    }

    isReplaying.store(true, std::memory_order_release);

    // Save render states before replay to restore after (replay skips per-call restore)
    DWORD savedAlphaBlend = 0, savedAlphaTest = 0, savedZEnable = 0, savedZWrite = 0, savedZFunc = 0;
    DWORD savedAmbientMat = 0, savedDiffuseMat = 0, savedEmissiveMat = 0;
    if (!cmdBuf) {
        device->GetRenderState(D3DRS_ALPHABLENDENABLE, &savedAlphaBlend);
        device->GetRenderState(D3DRS_ALPHATESTENABLE, &savedAlphaTest);
        device->GetRenderState(D3DRS_ZENABLE, &savedZEnable);
        device->GetRenderState(D3DRS_ZWRITEENABLE, &savedZWrite);
        device->GetRenderState(D3DRS_ZFUNC, &savedZFunc);
        device->GetRenderState(D3DRS_AMBIENTMATERIALSOURCE, &savedAmbientMat);
        device->GetRenderState(D3DRS_DIFFUSEMATERIALSOURCE, &savedDiffuseMat);
        device->GetRenderState(D3DRS_EMISSIVEMATERIALSOURCE, &savedEmissiveMat);
    }

    // Reset to baseline state at start of each scene replay to prevent cross-scene leaks
    setReplayBaseline(cmdBuf, device);

    // Scene 0 only: Fill DL areas with dark ambient before MW geometry
    // Prevents bright sky clear color from bleeding through AA edges against DL
    // Uses depth test (GREATEREQUAL) to preserve DL grass already rendered
    if (sceneCount == 0 && !cmdBuf) {
        DistantLand::renderDepthBackfill(&fb.dlContext);
    }

    // N-1: Use rendering buffer's currentView/currentProj (stamped at previous Present())
    // These matrices represent the camera position when this frame was recorded
    D3DXMATRIX currentView = fb.currentView;
    D3DXMATRIX currentProj = fb.currentProj;

    // Set active shadow pointer to recording-time matrices for replay
    // (fb already declared at function start for buffer selection)
    s_activeShadowVP = fb.shadowViewproj;

    // Hi-Z culling statistics (shouldRender pre-set by executeHiZCulling)
    int totalCalls = recCalls.size();
    int culledCalls = 0;

    // Calculate camera velocity from previous frame (to compensate for one-frame-behind Hi-Z)
    D3DXVECTOR3 currentCameraPos = D3DXVECTOR3(DistantLand::s_staging.eyePos.x, DistantLand::s_staging.eyePos.y, DistantLand::s_staging.eyePos.z);
    D3DXVECTOR3 cameraVelocity(0.0f, 0.0f, 0.0f);
    float cameraMovementMag = 0.0f;

    if (hasPrevCamera) {
        cameraVelocity = currentCameraPos - prevCameraPos;
        cameraMovementMag = D3DXVec3Length(&cameraVelocity);
    }

    // Store current camera for next frame
    prevCameraPos = currentCameraPos;
    prevCameraView = currentView;
    hasPrevCamera = true;

    // Per-object light packing for mode 3 (saturated Morrowind assignment)
    // Each mode 3 object gets only its spatially-nearby lights packed into the texture.
    // NOTE: Only run for Scene 0 - runs once per frame, not 3x per scene.
    // Scene 1/2 use fixed-function 8-light path, not mode 3 per-object lights.
    const auto& sceneLights = fb.sceneLights;
    int numSceneLights = (int)sceneLights.size();
    if (sceneCount == 0) {
        MGE_ZoneScopedN("replay_PerObjectLightPack");
        const size_t numCallsForPack = recCalls.size();
        perObjectLightInfo.resize(numCallsForPack);
        memset(perObjectLightInfo.data(), 0, numCallsForPack * sizeof(PerObjectLightInfo));

        // Flat buffer: 12 floats per light (3 texels × 4 floats), packed contiguously
        std::vector<float> packedLightData;
        int currentTexelOffset = 0;
        int mode3Count = 0;
        int maxPerObjectLights = 0;

        for (size_t i = 0; i < numCallsForPack; i++) {
            // Pack lights for lightMode 3 OR if forced/batching is enabled (which forces mode 3)
            bool needsLightPack = (recCalls[i].sk.lightMode == 3);
            // Force lightMode 3 for all lit Opaque/Terrain when toggle is enabled
            if (ImGuiManager::GetForceLightMode3() && recCalls[i].sk.useLighting &&
                (recCalls[i].bin == RenderBin::Opaque ||
                 recCalls[i].bin == RenderBin::Terrain ||
                 recCalls[i].bin == RenderBin::TerrainBlend)) {
                needsLightPack = true;
            }
            // Also pack for stateless batching candidates
            if (ImGuiManager::GetEnableStatelessBatch() && recCalls[i].sk.useLighting &&
                (recCalls[i].bin == RenderBin::Opaque ||
                 recCalls[i].bin == RenderBin::Terrain ||
                 recCalls[i].bin == RenderBin::TerrainBlend)) {
                needsLightPack = true;
            }
            if (!needsLightPack) continue;

            const auto& call = recCalls[i];

            // Bounding box for sphere-AABB intersection test
            // For large meshes, lights can be inside the bbox but far from center
            D3DXVECTOR3 bMin, bMax;
            if (call.hasBoundingBox) {
                bMin = call.bboxMin;
                bMax = call.bboxMax;
            } else {
                // No bbox: use object origin as a point
                float ox = call.rs.worldTransforms[0]._41;
                float oy = call.rs.worldTransforms[0]._42;
                float oz = call.rs.worldTransforms[0]._43;
                bMin = bMax = D3DXVECTOR3(ox, oy, oz);
            }

            int count = 0;

            for (const auto& light : sceneLights) {
                // Skip lights culled by Hi-Z (but not during menu mode - Hi-Z uses stale matrices)
                if (!light.isVisible && !fb.dlContext.isRenderCached && !fb.postProcessData.isMenu) continue;

                // Sphere-AABB intersection: closest point on bbox to light center
                float cx = (light.position.x < bMin.x) ? bMin.x : (light.position.x > bMax.x) ? bMax.x : light.position.x;
                float cy = (light.position.y < bMin.y) ? bMin.y : (light.position.y > bMax.y) ? bMax.y : light.position.y;
                float cz = (light.position.z < bMin.z) ? bMin.z : (light.position.z > bMax.z) ? bMax.z : light.position.z;
                float dx = light.position.x - cx;
                float dy = light.position.y - cy;
                float dz = light.position.z - cz;
                float dist2 = dx*dx + dy*dy + dz*dz;
                float lightRadius = light.radius;
                if (dist2 < lightRadius * lightRadius) {
                    // Transform to view-space and pack 12 floats (3 texels)
                    D3DXVECTOR4 worldPos4(light.position.x, light.position.y, light.position.z, 1.0f);
                    D3DXVECTOR4 viewPos4;
                    D3DXVec4Transform(&viewPos4, &worldPos4, &currentView);

                    // Texel 0: view-space position + radius
                    packedLightData.push_back(viewPos4.x);
                    packedLightData.push_back(viewPos4.y);
                    packedLightData.push_back(viewPos4.z);
                    packedLightData.push_back(light.radius);

                    // Texel 1: color
                    packedLightData.push_back(light.diffuse.r);
                    packedLightData.push_back(light.diffuse.g);
                    packedLightData.push_back(light.diffuse.b);
                    packedLightData.push_back(0.0f);

                    // Texel 2: falloff parameters
                    packedLightData.push_back(light.falloff.x);  // constant
                    packedLightData.push_back(light.falloff.y);  // linear
                    packedLightData.push_back(light.falloff.z);  // quadratic
                    packedLightData.push_back(0.0f);

                    count++;
                }
            }

            perObjectLightInfo[i] = {currentTexelOffset, count};
            currentTexelOffset += count * 3;  // 3 texels per light

            // Diagnostic: log first mode 3 object's bbox and nearest light (sphere-AABB distance)
            if (mode3Count == 0 && !sceneLights.empty() && LOG::catEnabled(LOG::Cat_Mode3)) {
                float nearestDist = FLT_MAX;
                int nearestIdx = -1;
                float nearestRadius = 0;
                for (int li = 0; li < (int)sceneLights.size(); li++) {
                    const auto& light = sceneLights[li];
                    float cx = (light.position.x < bMin.x) ? bMin.x : (light.position.x > bMax.x) ? bMax.x : light.position.x;
                    float cy = (light.position.y < bMin.y) ? bMin.y : (light.position.y > bMax.y) ? bMax.y : light.position.y;
                    float cz = (light.position.z < bMin.z) ? bMin.z : (light.position.z > bMax.z) ? bMax.z : light.position.z;
                    float dx = light.position.x - cx, dy = light.position.y - cy, dz = light.position.z - cz;
                    float d = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (d < nearestDist) { nearestDist = d; nearestIdx = li; nearestRadius = light.radius; }
                }
                LOG::logline("Mode3 diag: obj[%d] bbox=(%.0f,%.0f,%.0f)-(%.0f,%.0f,%.0f) nearest light[%d] dist=%.1f radius=%.1f found=%d",
                    (int)i, bMin.x, bMin.y, bMin.z, bMax.x, bMax.y, bMax.z,
                    nearestIdx, nearestDist, nearestRadius, count);
            }

            mode3Count++;
            if (count > maxPerObjectLights) maxPerObjectLights = count;
        }

        int totalTexels = currentTexelOffset;
        perObjectTexelSize = totalTexels > 0 ? 1.0f / totalTexels : 0.0f;

        if (ImGuiManager::GetAndClearDumpLightSnapshot()) {
            static int s_lightDumpSeq = 0;
            ++s_lightDumpSeq;

            int rawActive = 0;
            int rawPoint = 0;
            int rawDirectional = 0;
            if (fb.lastLightState) {
                rawActive = (int)fb.lastLightState->active.size();
                for (DWORD id : fb.lastLightState->active) {
                    auto it = fb.lastLightState->lights.find(id);
                    if (it == fb.lastLightState->lights.end()) continue;
                    if (it->second.type == D3DLIGHT_POINT) ++rawPoint;
                    else if (it->second.type == D3DLIGHT_DIRECTIONAL) ++rawDirectional;
                }
            }

            int packedObjects = 0;
            int packedZero = 0;
            int packedMin = INT_MAX;
            int packedMax = 0;
            int packedSum = 0;
            int lightModeCounts[4] = {0, 0, 0, 0};
            for (size_t i = 0; i < numCallsForPack; ++i) {
                const auto& call = recCalls[i];
                if (call.sk.lightMode < 4) ++lightModeCounts[call.sk.lightMode];
                bool needsLightPack = (call.sk.lightMode == 3);
                if (ImGuiManager::GetForceLightMode3() && call.sk.useLighting &&
                    (call.bin == RenderBin::Opaque ||
                     call.bin == RenderBin::Terrain ||
                     call.bin == RenderBin::TerrainBlend)) {
                    needsLightPack = true;
                }
                if (ImGuiManager::GetEnableStatelessBatch() && call.sk.useLighting &&
                    (call.bin == RenderBin::Opaque ||
                     call.bin == RenderBin::Terrain ||
                     call.bin == RenderBin::TerrainBlend)) {
                    needsLightPack = true;
                }
                if (!needsLightPack) continue;

                int count = perObjectLightInfo[i].lightCount;
                ++packedObjects;
                if (count == 0) ++packedZero;
                if (count < packedMin) packedMin = count;
                if (count > packedMax) packedMax = count;
                packedSum += count;
            }
            if (packedObjects == 0) packedMin = 0;

            LOG::logline("=== LIGHT SNAPSHOT %d BEGIN ===", s_lightDumpSeq);
            LOG::logline("LightSnapshot frame=%d scene=%d calls=%d renderedFb=%p camera=(%.2f,%.2f,%.2f) rawActive=%d rawPoint=%d rawDirectional=%d sceneLights=%d mode3Objects=%d zeroPacked=%d packedMin=%d packedMax=%d packedAvg=%.2f texels=%d texelSize=%.8f lightModes=[%d,%d,%d,%d]",
                fb.frameNumber, sceneCount, (int)recCalls.size(), &fb,
                currentCameraPos.x, currentCameraPos.y, currentCameraPos.z,
                rawActive, rawPoint, rawDirectional, numSceneLights,
                packedObjects, packedZero, packedMin, packedMax,
                packedObjects > 0 ? (float)packedSum / packedObjects : 0.0f,
                totalTexels, perObjectTexelSize,
                lightModeCounts[0], lightModeCounts[1], lightModeCounts[2], lightModeCounts[3]);

            if (fb.lastLightState) {
                int rawLogged = 0;
                for (DWORD id : fb.lastLightState->active) {
                    auto it = fb.lastLightState->lights.find(id);
                    if (it == fb.lastLightState->lights.end()) {
                        LOG::logline("LightSnapshot raw[%d] id=%u missing", rawLogged, id);
                    } else {
                        const auto& light = it->second;
                        LOG::logline("LightSnapshot raw[%d] id=%u type=%d pos=(%.2f,%.2f,%.2f) diffuse=(%.4f,%.4f,%.4f) falloff=(%.6f,%.6f,%.6f)",
                            rawLogged, id, (int)light.type,
                            light.position.x, light.position.y, light.position.z,
                            light.diffuse.r, light.diffuse.g, light.diffuse.b,
                            light.falloff.x, light.falloff.y, light.falloff.z);
                    }
                    if (++rawLogged >= 64) {
                        LOG::logline("LightSnapshot raw: truncated after 64 active lights");
                        break;
                    }
                }
            } else {
                LOG::logline("LightSnapshot raw: no fb.lastLightState");
            }

            for (int li = 0; li < (int)sceneLights.size() && li < 64; ++li) {
                const auto& light = sceneLights[li];
                float dx = light.position.x - currentCameraPos.x;
                float dy = light.position.y - currentCameraPos.y;
                float dz = light.position.z - currentCameraPos.z;
                float dist = sqrtf(dx * dx + dy * dy + dz * dz);
                LOG::logline("LightSnapshot scene[%d] id=%u pos=(%.2f,%.2f,%.2f) camDist=%.2f radius=%.2f diffuse=(%.4f,%.4f,%.4f) falloff=(%.6f,%.6f,%.6f)",
                    li, light.id, light.position.x, light.position.y, light.position.z,
                    dist, light.radius, light.diffuse.r, light.diffuse.g, light.diffuse.b,
                    light.falloff.x, light.falloff.y, light.falloff.z);
            }
            if (sceneLights.size() > 64) {
                LOG::logline("LightSnapshot scene: truncated after 64 of %d lights", (int)sceneLights.size());
            }

            int objectLogged = 0;
            for (size_t i = 0; i < numCallsForPack && objectLogged < 96; ++i) {
                const auto& call = recCalls[i];
                bool needsLightPack = (call.sk.lightMode == 3);
                if (ImGuiManager::GetForceLightMode3() && call.sk.useLighting &&
                    (call.bin == RenderBin::Opaque ||
                     call.bin == RenderBin::Terrain ||
                     call.bin == RenderBin::TerrainBlend)) {
                    needsLightPack = true;
                }
                if (ImGuiManager::GetEnableStatelessBatch() && call.sk.useLighting &&
                    (call.bin == RenderBin::Opaque ||
                     call.bin == RenderBin::Terrain ||
                     call.bin == RenderBin::TerrainBlend)) {
                    needsLightPack = true;
                }
                if (!needsLightPack) continue;

                D3DXVECTOR3 bMin, bMax;
                if (call.hasBoundingBox) {
                    bMin = call.bboxMin;
                    bMax = call.bboxMax;
                } else {
                    float ox = call.rs.worldTransforms[0]._41;
                    float oy = call.rs.worldTransforms[0]._42;
                    float oz = call.rs.worldTransforms[0]._43;
                    bMin = bMax = D3DXVECTOR3(ox, oy, oz);
                }

                float nearestDist = FLT_MAX;
                DWORD nearestId = 0;
                float nearestRadius = 0.0f;
                for (const auto& light : sceneLights) {
                    float cx = (light.position.x < bMin.x) ? bMin.x : (light.position.x > bMax.x) ? bMax.x : light.position.x;
                    float cy = (light.position.y < bMin.y) ? bMin.y : (light.position.y > bMax.y) ? bMax.y : light.position.y;
                    float cz = (light.position.z < bMin.z) ? bMin.z : (light.position.z > bMax.z) ? bMax.z : light.position.z;
                    float dx = light.position.x - cx;
                    float dy = light.position.y - cy;
                    float dz = light.position.z - cz;
                    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
                    if (dist < nearestDist) {
                        nearestDist = dist;
                        nearestId = light.id;
                        nearestRadius = light.radius;
                    }
                }

                LOG::logline("LightSnapshot obj[%d] call=%d bin=%d lm=%d lighting=%d count=%d texelOffset=%d bbox=(%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f) nearestId=%u nearestDist=%.2f nearestRadius=%.2f tex=%p vb=%p ib=%p",
                    objectLogged, (int)i, (int)call.bin, (int)call.sk.lightMode, (int)call.sk.useLighting,
                    perObjectLightInfo[i].lightCount, perObjectLightInfo[i].texelOffset,
                    bMin.x, bMin.y, bMin.z, bMax.x, bMax.y, bMax.z,
                    nearestId, nearestDist, nearestRadius,
                    call.rs.texture, call.rs.vb, call.rs.ib);
                ++objectLogged;
            }
            if (packedObjects > objectLogged) {
                LOG::logline("LightSnapshot obj: truncated after %d of %d packable objects", objectLogged, packedObjects);
            }
            LOG::logline("=== LIGHT SNAPSHOT %d END ===", s_lightDumpSeq);
        }

        // Upload packed light data to per-object texture (owned by RenderThread for thread safety)
        if (totalTexels > 0 && g_renderThread) {
            IDirect3DTexture9* tex = g_renderThread->getPerObjectLightTexture();

            // Create or resize texture if needed
            if (!tex) {
                HRESULT hr = device->CreateTexture(
                    totalTexels, 1, 1, 0,
                    D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED,
                    &tex, nullptr);
                if (FAILED(hr)) {
                    LOG::logline("!! Failed to create per-object light texture (hr=0x%X)", hr);
                    tex = nullptr;
                }
                g_renderThread->setPerObjectLightTexture(tex);
            } else {
                D3DSURFACE_DESC desc;
                HRESULT hr = tex->GetLevelDesc(0, &desc);
                if (FAILED(hr)) {
                    // Texture became invalid (D3D threading issue) - release and recreate
                    LOG::logline("!! texPerObjectLightData->GetLevelDesc failed (hr=0x%X), recreating", hr);
                    tex->Release();
                    tex = nullptr;
                    hr = device->CreateTexture(
                        totalTexels, 1, 1, 0,
                        D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED,
                        &tex, nullptr);
                    if (FAILED(hr)) {
                        LOG::logline("!! Failed to recreate per-object light texture (hr=0x%X)", hr);
                    }
                    g_renderThread->setPerObjectLightTexture(tex);
                } else if (desc.Width != (UINT)totalTexels) {
                    tex->Release();
                    hr = device->CreateTexture(
                        totalTexels, 1, 1, 0,
                        D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED,
                        &tex, nullptr);
                    if (FAILED(hr)) {
                        LOG::logline("!! Failed to resize per-object light texture (hr=0x%X)", hr);
                        tex = nullptr;
                    }
                    g_renderThread->setPerObjectLightTexture(tex);
                }
            }

            if (tex) {
                D3DLOCKED_RECT locked;
                if (SUCCEEDED(tex->LockRect(0, &locked, nullptr, 0))) {
                    memcpy(locked.pBits, packedLightData.data(), totalTexels * 4 * sizeof(float));
                    tex->UnlockRect(0);
                }
                device->SetTexture(5, tex);
            }
        }

        if (mode3Count > 0) {
            LOG_CAT(LOG::Cat_Mode3, "Mode3 packing: %d objects, maxLights/obj=%d, totalTexels=%d (from %d scene)",
                mode3Count, maxPerObjectLights, totalTexels, numSceneLights);
        }
    }

    // For mode 0-2 objects, also upload global light data for non-mode-3 texture access
    // Mode 2 uses uniform constants (not texture), mode 3 uses per-object texture via c50
    // Set default c50 for non-mode-3 objects (will be overridden per-draw for mode 3)
    {
        float lightParams[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        device->SetPixelShaderConstantF(50, lightParams, 1);
    }

    // Inline Hi-Z culling using current matrices (same as bbox visualization)
    const size_t numCalls = recCalls.size();

    // Bin statistics
    int binCounts[(int)RenderBin::Count] = {};
    static const char* binNames[] = { "Terrain", "TerrainBlend", "Opaque", "Skinning", "Grass", "AlphaTested", "Blending" };

    // Material sorting metrics collection
    std::unordered_set<IDirect3DTexture9*> seenTextures;
    std::unordered_set<MaterialKey, MaterialKey::Hasher> seenMaterials;
    MaterialKey prevMaterial = {};
    bool hasPrevMaterial = false;
    int materialTransitions = 0;
    int drawCallCount = 0;

#ifdef TRACY_ENABLE
    // Tracy bin zone: manually managed ScopedZone for per-bin profiling regions
    static constexpr tracy::SourceLocationData binZoneSrcLoc { "RenderBin", TracyFunction, TracyFile, (uint32_t)__LINE__, 0 };
    tracy::ScopedZone* binZone = nullptr;
    alignas(tracy::ScopedZone) char binZoneBuf[sizeof(tracy::ScopedZone)];
    RenderBin currentBin = RenderBin::Count;
#endif

    // Slow frame detection: QPC timing for replay loop
    LARGE_INTEGER replayFreqQPC, replayStartQPC, replayEndQPC;
    QueryPerformanceFrequency(&replayFreqQPC);
    QueryPerformanceCounter(&replayStartQPC);
    float worstCallMs = 0.0f;
    int worstCallIndex = -1;
    int worstCallPrims = 0;
    int worstCallBin = 0;
    float slowCallThreshold = ImGuiManager::GetSlowCallThreshold();

    // Track merged batch draws (Scene 0 only) - per-draw data in texture
    std::unordered_set<size_t> mergedBatchIndices;
    int mergedDrawCalls = 0;

    // Merged batch texture setup block - creates and fills draw data texture
    if (sceneCount == 0 && ImGuiManager::GetEnableStatelessBatch() &&
        !fb.mergedBatches.empty() && !fb.drawDataStaging.empty()) {
        MGE_ZoneScopedN("replay_BatchedDraws");

        // Reset suffix texture binding cache - ensures fresh binds for batched draws
        TextureSuffix::getBindingState().reset();

        UINT totalDraws = (UINT)fb.drawDataStaging.size();
        const UINT DRAW_DATA_WIDTH = 16;  // 16 texels per draw (256 bytes / 16 bytes per texel)

        // Create or resize draw data texture as needed
        if (!fb.texDrawData || fb.texDrawDataHeight < totalDraws) {
            if (fb.texDrawData) {
                fb.texDrawData->Release();
                fb.texDrawData = nullptr;
            }
            // Round up height to power of 2 for better texture handling
            UINT newHeight = 1;
            while (newHeight < totalDraws) newHeight *= 2;
            if (newHeight < 64) newHeight = 64;  // Minimum 64 draws

            HRESULT hr = device->CreateTexture(
                DRAW_DATA_WIDTH, newHeight, 1, 0,
                D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED,
                &fb.texDrawData, nullptr);
            if (FAILED(hr)) {
                LOG::logline("!! Failed to create draw data texture (hr=0x%X)", hr);
                fb.texDrawData = nullptr;
            } else {
                fb.texDrawDataHeight = newHeight;
                LOG::logline("-- Created draw data texture: %dx%d", DRAW_DATA_WIDTH, newHeight);
            }
        }

        if (fb.texDrawData) {
            // Fill per-object light params from perObjectLightInfo (populated earlier in this function)
            // drawDataStaging is indexed sequentially; we need to map back to original call indices
            // Order: {lightCount, texelSize, texelOffset, 0} to match lightDataParams in shader
            UINT stagingIdx = 0;
            static bool loggedLightFill = false;
            int logCount = 0;

            // Far-cascade shadow matrix: fb.shadowViewproj[1] is updated by Stage0GPU's
            // renderShadowMap pass each frame, so we bake (world × shadowVP1) here on
            // the render thread instead of in CPT (where shadowVP1 would be the previous
            // frame's value, producing visible biasing in the far cascade).
            const D3DXMATRIX& shadowVP1 = fb.shadowViewproj[1];

            // Fill light params + far-shadow matrix for merged batches
            for (const auto& mb : fb.mergedBatches) {
                for (size_t callIdx : mb.callIndices) {
                    if (stagingIdx < fb.drawDataStaging.size()) {
                        StatelessDrawData& dd = fb.drawDataStaging[stagingIdx];

                        // Bake far-cascade shadow matrix unconditionally — it depends only
                        // on the call's world transform and this frame's shadowVP1.
                        const D3DXMATRIX m = recCalls[callIdx].rs.worldTransforms[0] * shadowVP1;
                        dd.worldShadow1_0[0] = m._11; dd.worldShadow1_0[1] = m._21; dd.worldShadow1_0[2] = m._31; dd.worldShadow1_0[3] = m._41;
                        dd.worldShadow1_1[0] = m._12; dd.worldShadow1_1[1] = m._22; dd.worldShadow1_1[2] = m._32; dd.worldShadow1_1[3] = m._42;
                        dd.worldShadow1_2[0] = m._13; dd.worldShadow1_2[1] = m._23; dd.worldShadow1_2[2] = m._33; dd.worldShadow1_2[3] = m._43;
                        dd.worldShadow1_3[0] = m._14; dd.worldShadow1_3[1] = m._24; dd.worldShadow1_3[2] = m._34; dd.worldShadow1_3[3] = m._44;

                        if (callIdx < perObjectLightInfo.size()) {
                            const auto& li = perObjectLightInfo[callIdx];
                            dd.lightParams[0] = (float)li.lightCount;
                            dd.lightParams[1] = perObjectTexelSize;
                            dd.lightParams[2] = (float)li.texelOffset;
                            // lightParams[3] = vertexMaterial, already set in hiz_culling.cpp - don't overwrite

                            if (!loggedLightFill && logCount < 5 && li.lightCount > 0) {
                                LOG::logline("MergedBatch lightFill: stagingIdx=%d, callIdx=%d, lightCount=%d, texelOffset=%d, texelSize=%.6f",
                                    stagingIdx, (int)callIdx, li.lightCount, li.texelOffset, perObjectTexelSize);
                                logCount++;
                            }
                        }
                    }
                    stagingIdx++;
                }
            }
            if (logCount > 0) loggedLightFill = true;

            // Upload draw data to texture
            D3DLOCKED_RECT locked;
            if (SUCCEEDED(fb.texDrawData->LockRect(0, &locked, nullptr, 0))) {
                // Copy each draw's data as one 16-texel row.
                float* texData = (float*)locked.pBits;
                UINT pitch = locked.Pitch / sizeof(float);  // floats per row

                for (UINT d = 0; d < totalDraws; d++) {
                    const StatelessDrawData& src = fb.drawDataStaging[d];
                    float* row = texData + d * pitch;
                    memcpy(row, &src, sizeof(StatelessDrawData));
                }
                fb.texDrawData->UnlockRect(0);

                // Bind draw data texture - use vertex texture sampler for VS, regular for PS
                // D3DVERTEXTEXTURESAMPLER0 = 256 for vertex shader texture sampling
                device->SetTexture(D3DVERTEXTEXTURESAMPLER0, fb.texDrawData);
                device->SetSamplerState(D3DVERTEXTEXTURESAMPLER0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
                device->SetSamplerState(D3DVERTEXTEXTURESAMPLER0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
                device->SetSamplerState(D3DVERTEXTEXTURESAMPLER0, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
                device->SetSamplerState(D3DVERTEXTEXTURESAMPLER0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
                device->SetSamplerState(D3DVERTEXTEXTURESAMPLER0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
                // Bind to PS slot 6 only (shader samples from s6)
                device->SetTexture(6, fb.texDrawData);
                device->SetSamplerState(6, D3DSAMP_MINFILTER, D3DTEXF_POINT);
                device->SetSamplerState(6, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
                device->SetSamplerState(6, D3DSAMP_MIPFILTER, D3DTEXF_NONE);

                // Ensure light texture (slot 5) is bound for texture-based point lighting
                IDirect3DTexture9* lightTex = g_renderThread ? g_renderThread->getPerObjectLightTexture() : nullptr;
                if (lightTex) {
                    device->SetTexture(5, lightTex);
                }
                // Texture setup complete - merged batch draws happen below
            }
        }
    }

    // Issue merged batch draws (Scene 0 only) - collapses different geometries sharing same texture into mega-draws
    if (sceneCount == 0 && ImGuiManager::GetEnableStatelessBatch() && !fb.mergedBatches.empty()) {
        MGE_ZoneScopedN("replay_MergedBatchDraws");

        IDirect3DVertexBuffer9* useVB = nullptr;
        IDirect3DIndexBuffer9* useIB = nullptr;
        const std::vector<CachedDrawInfo>* drawInfos = nullptr;
        std::vector<CachedDrawInfo> builtDrawInfos;

        if (fb.useCachedMergedVB && fb.cachedMergedVB && fb.cachedMergedIB &&
            fb.cachedDrawInfos.size() == fb.mergedBatches.size()) {
            MGE_ZoneScopedN("MergedBatch_UseCachedGeometry");
            useVB = fb.cachedMergedVB;
            useIB = fb.cachedMergedIB;
            drawInfos = &fb.cachedDrawInfos;
        } else {
            UINT totalMergedVerts = 0;
            UINT totalMergedIndices = 0;
            UINT vbSizeNeeded = 0;

            {
                MGE_ZoneScopedN("MergedBatch_CountGeometry");
                for (const auto& mb : fb.mergedBatches) {
                    totalMergedVerts += mb.totalVertices;
                    totalMergedIndices += mb.totalIndices;
                    if (!mb.callIndices.empty()) {
                        const auto& call = recCalls[mb.callIndices[0]];
                        UINT batchExpandedStride = call.rs.vbStride + sizeof(float);
                        vbSizeNeeded += mb.totalVertices * batchExpandedStride;
                        vbSizeNeeded += batchExpandedStride;
                    }
                }
            }
            UINT ibSizeNeeded = totalMergedIndices * sizeof(DWORD);  // 32-bit indices for >64k verts

            // Track the last layout each frame buffer wrote into its dynamic VB/IB. Cached layouts
            // keep AddRef'd references to those buffers, so a different layout needs a fresh object.
            static std::unordered_map<const void*, std::pair<void*, size_t>> s_fbLastBuiltLayout;
            bool layoutChanged = false;
            {
                MGE_ZoneScopedN("MergedBatch_CellCacheCheck");
                auto lastLayoutIt = s_fbLastBuiltLayout.find(&fb);
                if (lastLayoutIt != s_fbLastBuiltLayout.end() &&
                    (lastLayoutIt->second.first != fb.cellBatchCacheKey ||
                     lastLayoutIt->second.second != fb.cellBatchLayoutHash)) {
                    layoutChanged = true;
                    LOG_CAT(LOG::Cat_HLSLReplay, "MergedBatch: Layout changed for fb=%p, forcing VB/IB recreation (oldCell=%p oldHash=%Ix, newCell=%p newHash=%Ix)",
                                 &fb, lastLayoutIt->second.first, lastLayoutIt->second.second,
                                 fb.cellBatchCacheKey, fb.cellBatchLayoutHash);
                }
                s_fbLastBuiltLayout[&fb] = std::make_pair(fb.cellBatchCacheKey, fb.cellBatchLayoutHash);
            }

            {
                MGE_ZoneScopedN("MergedBatch_EnsureBuffers");
                if (fb.mergedVB == nullptr || fb.mergedVBSize < vbSizeNeeded || layoutChanged) {
                    if (fb.mergedVB) {
                        fb.mergedVB->Release();
                        fb.mergedVB = nullptr;
                        fb.mergedVBSize = 0;
                    }
                    UINT newSize = std::max(vbSizeNeeded, 1024u * 1024u);
                    if (SUCCEEDED(device->CreateVertexBuffer(newSize, D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                                                             0, D3DPOOL_DEFAULT, &fb.mergedVB, nullptr))) {
                        fb.mergedVBSize = newSize;
                        LOG_CAT(LOG::Cat_HLSLReplay, "MergedBatch: Created dynamic VB, %d bytes", newSize);
                    }
                }

                if (fb.mergedIB == nullptr || fb.mergedIBSize < totalMergedIndices || layoutChanged) {
                    if (fb.mergedIB) {
                        fb.mergedIB->Release();
                        fb.mergedIB = nullptr;
                        fb.mergedIBSize = 0;
                    }
                    UINT newCount = std::max(totalMergedIndices, 256u * 1024u);
                    if (SUCCEEDED(device->CreateIndexBuffer(newCount * sizeof(DWORD), D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY,
                                                            D3DFMT_INDEX32, D3DPOOL_DEFAULT, &fb.mergedIB, nullptr))) {
                        fb.mergedIBSize = newCount;
                        LOG_CAT(LOG::Cat_HLSLReplay, "MergedBatch: Created dynamic IB (32-bit), %d indices", newCount);
                    }
                }
            }

            if (fb.mergedVB && fb.mergedIB && vbSizeNeeded > 0 && ibSizeNeeded > 0) {
                MGE_ZoneScopedN("MergedBatch_RebuildGeometry");
                void* vbData = nullptr;
                void* ibData = nullptr;
                bool vbLocked = false;
                bool ibLocked = false;

                HRESULT hrVB;
                {
                    MGE_ZoneScopedN("MergedBatch_LockTargetVB");
                    hrVB = fb.mergedVB->Lock(0, vbSizeNeeded, &vbData, D3DLOCK_DISCARD);
                }
                if (SUCCEEDED(hrVB)) {
                    vbLocked = true;
                    HRESULT hrIB;
                    {
                        MGE_ZoneScopedN("MergedBatch_LockTargetIB");
                        hrIB = fb.mergedIB->Lock(0, ibSizeNeeded, &ibData, D3DLOCK_DISCARD);
                    }
                    if (SUCCEEDED(hrIB)) {
                        ibLocked = true;

                        BYTE* vbBase = static_cast<BYTE*>(vbData);
                        DWORD* ibDst = static_cast<DWORD*>(ibData);
                        UINT vbByteOffset = 0;
                        UINT ibOffset = 0;

                        builtDrawInfos.reserve(fb.mergedBatches.size());

                        static bool loggedMergedDebug = false;
                        {
                            MGE_ZoneScopedN("MergedBatch_CopySourceGeometry");
                            for (const auto& mb : fb.mergedBatches) {
                                if (mb.callIndices.empty()) continue;

                                const auto& firstCall = recCalls[mb.callIndices[0]];
                                UINT batchStride = firstCall.rs.vbStride;
                                UINT batchExpandedStride = batchStride + sizeof(float);

                                UINT remainder = vbByteOffset % batchExpandedStride;
                                if (remainder != 0) {
                                    vbByteOffset += (batchExpandedStride - remainder);
                                }

                                BYTE* vbDst = vbBase + vbByteOffset;

                                if (!loggedMergedDebug) {
                                    LOG::logline("MergedBatch DEBUG: calls=%d, batchStride=%d, expandedStride=%d, FVF=0x%X, drawDataOffset=%d",
                                        (int)mb.callIndices.size(), batchStride, batchExpandedStride, mb.key.fvf, mb.drawDataOffset);
                                }

                                CachedDrawInfo info = {};
                                info.vbByteOffset = vbByteOffset;
                                info.ibStartIndex = ibOffset;
                                info.expandedStride = batchExpandedStride;

                                UINT localVertOffset = 0;

                                for (size_t i = 0; i < mb.callIndices.size(); i++) {
                                    size_t callIdx = mb.callIndices[i];
                                    const auto& call = recCalls[callIdx];
                                    UINT drawIndex = mb.drawDataOffset + (UINT)i;
                                    UINT srcStride = call.rs.vbStride;

                                    if (!loggedMergedDebug && i < 3) {
                                        LOG::logline("  Call %d: drawIndex=%d, srcStride=%d, vertCount=%d",
                                            (int)i, drawIndex, srcStride, call.rs.vertCount);
                                    }

                                    D3DINDEXBUFFER_DESC ibDesc = {};
                                    call.rs.ib->GetDesc(&ibDesc);
                                    bool is32Bit = (ibDesc.Format == D3DFMT_INDEX32);
                                    UINT idxSize = is32Bit ? 4 : 2;

                                    UINT indexCount = call.rs.primCount * 3;
                                    void* srcIndicesRaw = nullptr;

                                    if (call.rs.ib && SUCCEEDED(call.rs.ib->Lock(call.rs.startIndex * idxSize,
                                                                                indexCount * idxSize,
                                                                                &srcIndicesRaw, D3DLOCK_READONLY))) {
                                        UINT realMaxIdx = 0;
                                        UINT realMinIdx = 0xFFFFFFFF;

                                        if (is32Bit) {
                                            const DWORD* srcIdx32 = static_cast<const DWORD*>(srcIndicesRaw);
                                            for (UINT idx = 0; idx < indexCount; idx++) {
                                                if (srcIdx32[idx] > realMaxIdx) realMaxIdx = srcIdx32[idx];
                                            }

                                            UINT absoluteFloor = (realMaxIdx > call.rs.vertCount) ? (realMaxIdx - call.rs.vertCount) : 0;
                                            for (UINT idx = 0; idx < indexCount; idx++) {
                                                if (srcIdx32[idx] >= absoluteFloor && srcIdx32[idx] < realMinIdx) {
                                                    realMinIdx = srcIdx32[idx];
                                                }
                                            }

                                            for (UINT idx = 0; idx < indexCount; idx++) {
                                                ibDst[idx] = (srcIdx32[idx] - realMinIdx) + localVertOffset;
                                            }
                                        } else {
                                            const WORD* srcIdx16 = static_cast<const WORD*>(srcIndicesRaw);
                                            for (UINT idx = 0; idx < indexCount; idx++) {
                                                if (srcIdx16[idx] > realMaxIdx) realMaxIdx = srcIdx16[idx];
                                            }

                                            UINT absoluteFloor = (realMaxIdx > call.rs.vertCount) ? (realMaxIdx - call.rs.vertCount) : 0;
                                            for (UINT idx = 0; idx < indexCount; idx++) {
                                                if (srcIdx16[idx] >= absoluteFloor && srcIdx16[idx] < realMinIdx) {
                                                    realMinIdx = srcIdx16[idx];
                                                }
                                            }

                                            for (UINT idx = 0; idx < indexCount; idx++) {
                                                ibDst[idx] = (DWORD)(srcIdx16[idx] - realMinIdx) + localVertOffset;
                                            }
                                        }

                                        ibDst += indexCount;
                                        call.rs.ib->Unlock();

                                        UINT srcVertexStart = call.rs.baseIndex + realMinIdx;
                                        UINT srcLockOffset = call.rs.vbOffset + (srcVertexStart * srcStride);

                                        void* srcVerts = nullptr;
                                        if (call.rs.vb && SUCCEEDED(call.rs.vb->Lock(srcLockOffset,
                                                                                    call.rs.vertCount * srcStride,
                                                                                    &srcVerts, D3DLOCK_READONLY))) {
                                            const BYTE* src = static_cast<const BYTE*>(srcVerts);
                                            for (UINT v = 0; v < call.rs.vertCount; v++) {
                                                UINT copySize = std::min(srcStride, batchStride);
                                                memcpy(vbDst, src, copySize);
                                                float* drawIdxPtr = reinterpret_cast<float*>(vbDst + batchStride);
                                                *drawIdxPtr = (float)drawIndex;
                                                if (!loggedMergedDebug && v == 0 && i == 0) {
                                                    LOG::logline("  DrawIndex write: drawIndex=%d at offset %d, expandedStride=%d, wrote %.1f",
                                                        drawIndex, batchStride, batchExpandedStride, *drawIdxPtr);
                                                }
                                                vbDst += batchExpandedStride;
                                                src += srcStride;
                                            }
                                            call.rs.vb->Unlock();
                                        }
                                    }

                                    localVertOffset += call.rs.vertCount;
                                    info.vertCount += call.rs.vertCount;
                                    info.primCount += call.rs.primCount;
                                    vbByteOffset += call.rs.vertCount * batchExpandedStride;
                                    ibOffset += indexCount;
                                }

                                builtDrawInfos.push_back(info);
                                loggedMergedDebug = true;
                            }
                        }
                    }
                }

                {
                    MGE_ZoneScopedN("MergedBatch_UnlockTargets");
                    if (ibLocked) {
                        fb.mergedIB->Unlock();
                    }
                    if (vbLocked) {
                        fb.mergedVB->Unlock();
                    }
                }

                if (ibLocked && vbLocked && builtDrawInfos.size() == fb.mergedBatches.size()) {
                    useVB = fb.mergedVB;
                    useIB = fb.mergedIB;
                    drawInfos = &builtDrawInfos;

                    {
                        MGE_ZoneScopedN("MergedBatch_StoreCellCache");
                        FixedFunctionShader::storeCellBatchCacheVB(fb.cellBatchCacheKey, fb.cellBatchLayoutHash, fb.mergedVB, fb.mergedIB, builtDrawInfos);
                    }

                    static bool loggedMerge = false;
                    if (!loggedMerge && LOG::catEnabled(LOG::Cat_HLSLReplay)) {
                        loggedMerge = true;
                        LOG::logline("MergedBatch: %d batches ready, %d total verts, %d total indices (stored in cache)",
                                     (int)fb.mergedBatches.size(), totalMergedVerts, totalMergedIndices);
                    }
                }
            }
        }

        if (useVB && useIB && drawInfos && drawInfos->size() == fb.mergedBatches.size()) {
            {
                MGE_ZoneScopedN("MergedBatch_MarkMergedIndices");
                for (const auto& mb : fb.mergedBatches) {
                    for (size_t callIdx : mb.callIndices) {
                        mergedBatchIndices.insert(callIdx);
                    }
                }
            }

                // Cache for merged vertex declarations (FVF+stride -> decl with appended drawIndex)
                // Key combines FVF (low 32 bits) with stride (high 32 bits) to handle edge cases
                static std::unordered_map<uint64_t, IDirect3DVertexDeclaration9*> mergedDeclCache;

                // Helper to get/create merged vertex declaration
                // Must match FVF layout exactly, including variable-size texture coordinates
                auto getMergedDecl = [&](DWORD fvf, UINT originalStride) -> IDirect3DVertexDeclaration9* {
                    uint64_t cacheKey = ((uint64_t)originalStride << 32) | fvf;
                    auto it = mergedDeclCache.find(cacheKey);
                    if (it != mergedDeclCache.end()) return it->second;

                    // Build declaration from FVF + drawIndex at end
                    std::vector<D3DVERTEXELEMENT9> elements;
                    WORD offset = 0;

                    // Position (handle XYZRHW for transformed vertices)
                    if (fvf & D3DFVF_XYZRHW) {
                        elements.push_back({0, offset, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITIONT, 0});
                        offset += 16;
                    } else if (fvf & D3DFVF_XYZ) {
                        elements.push_back({0, offset, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0});
                        offset += 12;
                    }

                    // Blend weights (must come after position in FVF order)
                    int posType = (fvf >> 1) & 0x7;
                    int blendWeights = (posType >= 3) ? (posType - 2) : 0;
                    if (blendWeights > 0 && blendWeights <= 4) {
                        BYTE types[] = {D3DDECLTYPE_FLOAT1, D3DDECLTYPE_FLOAT2, D3DDECLTYPE_FLOAT3, D3DDECLTYPE_FLOAT4};
                        elements.push_back({0, offset, types[blendWeights-1], D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDWEIGHT, 0});
                        offset += blendWeights * 4;
                    }

                    // Normal
                    if (fvf & D3DFVF_NORMAL) {
                        elements.push_back({0, offset, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL, 0});
                        offset += 12;
                    }

                    // Diffuse color
                    if (fvf & D3DFVF_DIFFUSE) {
                        elements.push_back({0, offset, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0});
                        offset += 4;
                    }

                    // Specular color
                    if (fvf & D3DFVF_SPECULAR) {
                        elements.push_back({0, offset, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 1});
                        offset += 4;
                    }

                    // Texture coordinates with proper size decoding
                    int numTex = (fvf & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT;
                    for (int t = 0; t < numTex; t++) {
                        // Decode texture coordinate format from FVF (2 bits per texcoord starting at bit 16)
                        int fmt = (fvf >> (16 + t * 2)) & 0x3;
                        BYTE type = D3DDECLTYPE_FLOAT2;
                        int size = 8;
                        switch (fmt) {
                            case 0: type = D3DDECLTYPE_FLOAT2; size = 8; break;  // D3DFVF_TEXTUREFORMAT2 (default)
                            case 1: type = D3DDECLTYPE_FLOAT3; size = 12; break; // D3DFVF_TEXTUREFORMAT3
                            case 2: type = D3DDECLTYPE_FLOAT4; size = 16; break; // D3DFVF_TEXTUREFORMAT4
                            case 3: type = D3DDECLTYPE_FLOAT1; size = 4; break;  // D3DFVF_TEXTUREFORMAT1
                        }
                        elements.push_back({0, offset, type, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, (BYTE)t});
                        offset += size;
                    }

                    // DrawIndex at the end (at originalStride offset)
                    elements.push_back({0, (WORD)originalStride, D3DDECLTYPE_FLOAT1, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 7});

                    // End marker
                    elements.push_back(D3DDECL_END());

                    IDirect3DVertexDeclaration9* decl = nullptr;
                    if (SUCCEEDED(device->CreateVertexDeclaration(elements.data(), &decl))) {
                        mergedDeclCache[cacheKey] = decl;
                        LOG_CAT(LOG::Cat_HLSLReplay, "MergedBatch: Created vertex decl for FVF 0x%X, stride %d->%d",
                                     fvf, originalStride, originalStride + 4);
                    }
                    return decl;
                };

            D3DXMATRIX viewT, projT;
            D3DXVECTOR3 sunDirView;
            D3DXMATRIX shadowViewToClip[kShadowCascadeCount];
            RGBVECTOR sunDiffuse;
            RGBVECTOR sceneAmbient;

            {
                MGE_ZoneScopedN("MergedBatch_DrawSetup");
                // Draw each merged batch
                device->SetIndices(useIB);

                // Hoist loop-invariant calculations (Phase 2 optimization)
                D3DXMatrixTranspose(&viewT, &fb.currentView);
                D3DXMatrixTranspose(&projT, &fb.currentProj);

                // Pre-calculate sun direction in view space
                D3DXVec3TransformNormal(&sunDirView, (const D3DXVECTOR3*)&DistantLand::s_staging.sunVec, &fb.currentView);

                // Pre-calculate lighting values
                sunDiffuse = DistantLand::s_staging.lightSunMult * DistantLand::s_staging.sunCol;
                sceneAmbient = DistantLand::s_staging.lightAmbMult *
                    (DistantLand::s_staging.sunAmb + DistantLand::s_staging.ambCol);

                // Pre-calculate shadow view-to-clip matrices
                D3DXMATRIX viewInverse;
                D3DXMatrixInverse(&viewInverse, nullptr, &fb.currentView);
                for (int i = 0; i < kShadowCascadeCount; ++i) {
                    shadowViewToClip[i] = viewInverse * fb.shadowViewproj[i];
                }
            }

            // Local shader cache to avoid SRW locks in loop
            std::unordered_map<ShaderKey, HLSLShader, ShaderKey::hasher> localShaderCache;

            {
                MGE_ZoneScopedN("MergedBatch_DrawLoop");
                for (size_t bi = 0; bi < fb.mergedBatches.size(); bi++) {
                    const auto& mb = fb.mergedBatches[bi];
                    const auto& di = (*drawInfos)[bi];
                    if (di.vertCount == 0 || di.primCount == 0) continue;

                    // Get first call for shader setup
                    const auto& firstCall = recCalls[mb.callIndices[0]];

                    // Check suppress flags based on bin (same logic as main loop)
                    bool suppressed = false;
                    switch (firstCall.bin) {
                        case RenderBin::Terrain:
                        case RenderBin::TerrainBlend:
                                                    suppressed = ImGuiManager::GetSuppressTerrain(); break;
                        case RenderBin::Opaque:     suppressed = ImGuiManager::GetSuppressOpaque(); break;
                        case RenderBin::Skinning:   suppressed = ImGuiManager::GetSuppressSkinning(); break;
                        case RenderBin::Grass:      suppressed = ImGuiManager::GetSuppressGrass(); break;
                        case RenderBin::AlphaTested:suppressed = ImGuiManager::GetSuppressAlphaTested(); break;
                        case RenderBin::Blending:   suppressed = ImGuiManager::GetSuppressBlending(); break;
                        default: break;
                    }
                    if (suppressed) continue;

                    // Use stride from batch key - guaranteed to match all calls in this batch
                    UINT originalStride = mb.key.stride;

                    // Get or create vertex declaration
                    IDirect3DVertexDeclaration9* mergedDecl = nullptr;
                    {
                        MGE_ZoneScopedN("MergedBatch_GetDecl");
                        mergedDecl = getMergedDecl(mb.key.fvf, originalStride);
                    }
                    if (!mergedDecl) continue;

                    // Build shader key (same as stateless batch)
                    ShaderKey sk = firstCall.sk;
                    sk.useStatelessBatch = true;
                    sk.useInstancing = false;
                    // Force LightMode 3 for all batches - required for texture-based lighting
                    // Batches use per-draw light params from draw data texture, which requires
                    // USE_TEXTURE_LIGHTS to be defined (only true when lightMode == 3)
                    if (sk.useLighting && sk.lightMode < 3) {
                        sk.lightMode = 3;
                    }
                    // Phase 5B: Terrain tiles that absorbed a TerrainBlend overlay need the
                    // HAS_OVERLAY variant — PS samples s8 and composites over the base color.
                    sk.hasOverlay = (mb.key.overlayTexture != nullptr) ? 1 : 0;
                    // Phase 8A: if the overlay carries its own _paramh, PS samples s9 and
                    // blends paramh with the base by the same AlphaGrid factor.
                    sk.hasOverlayParamH = (mb.key.overlayParamHTexture != nullptr) ? 1 : 0;

                    // Look up shader variant - check local cache first to avoid SRW lock
                    HLSLShader hlslShader = {};
                    {
                        MGE_ZoneScopedN("MergedBatch_ShaderLookup");
                        auto localIt = localShaderCache.find(sk);
                        if (localIt != localShaderCache.end()) {
                            hlslShader = localIt->second;
                        } else {
                            AcquireSRWLockShared(&hlslCacheLock);
                            auto iShader = cacheHLSLShaders.find(sk);
                            if (iShader != cacheHLSLShaders.end()) {
                                hlslShader = iShader->second;
                                localShaderCache[sk] = hlslShader;
                            }
                            ReleaseSRWLockShared(&hlslCacheLock);
                        }
                    }

                    // If shader not found, queue compilation and skip this batch
                    if (!hlslShader.vertexShader || !hlslShader.pixelShader) {
                        queueShaderCompilation(sk);
                        continue;
                    }

                    {
                        MGE_ZoneScopedN("MergedBatch_BindShadersConstants");
                        // Bind shaders
                        device->SetVertexShader(hlslShader.vertexShader);
                        device->SetPixelShader(hlslShader.pixelShader);

                        // Set view/proj matrices using hoisted transposes
                        device->SetVertexShaderConstantF(12, (float*)&viewT, 4);  // view at c12
                        device->SetVertexShaderConstantF(0, (float*)&projT, 4);   // proj at c0
                    }

                    {
                        MGE_ZoneScopedN("MergedBatch_BindDrawDataTextures");
                        // Set draw data texture params: {1/width, 1/height, 0, 0}
                        // DRAW_DATA_WIDTH = 16 (16 texels per draw)
                        float drawDataParams[4] = { 1.0f / 16.0f, 1.0f / fb.texDrawDataHeight, 0.0f, 0.0f };
                        device->SetVertexShaderConstantF(70, drawDataParams, 1);  // VS: c70
                        device->SetPixelShaderConstantF(20, drawDataParams, 1);   // PS: c20

                        // Bind draw data texture with explicit VTF sampler states
                        device->SetTexture(D3DVERTEXTEXTURESAMPLER0, fb.texDrawData);
                        device->SetSamplerState(D3DVERTEXTEXTURESAMPLER0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
                        device->SetSamplerState(D3DVERTEXTEXTURESAMPLER0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
                        device->SetSamplerState(D3DVERTEXTEXTURESAMPLER0, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
                        device->SetSamplerState(D3DVERTEXTEXTURESAMPLER0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
                        device->SetSamplerState(D3DVERTEXTEXTURESAMPLER0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
                        // Bind to PS slot 6 only (shader samples from s6)
                        device->SetTexture(6, fb.texDrawData);
                        device->SetSamplerState(6, D3DSAMP_MINFILTER, D3DTEXF_POINT);
                        device->SetSamplerState(6, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
                        device->SetSamplerState(6, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
                    }

                    {
                        MGE_ZoneScopedN("MergedBatch_BindMaterialTextures");
                        // Bind textures using smart binding function
                        bindShaderTextures(sk, &firstCall.rs);

                        // Ensure light texture (slot 5) is bound after bindShaderTextures
                        // to prevent it from being overwritten
                        IDirect3DTexture9* lightTex = g_renderThread ? g_renderThread->getPerObjectLightTexture() : DistantLand::texLightData;
                        if (lightTex) {
                            device->SetTexture(5, lightTex);
                        }

                        // Phase 5B: bind absorbed TerrainBlend overlay texture to sampler s8.
                        // The PS samples s8 under HAS_OVERLAY and composites overlay over base.
                        if (mb.key.overlayTexture) {
                            IDirect3DTexture9* overlayBind = mb.key.overlayTexture;
                            device->SetTexture(8, overlayBind);
                            device->SetSamplerState(8, D3DSAMP_MINFILTER, D3DTEXF_ANISOTROPIC);
                            device->SetSamplerState(8, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
                            device->SetSamplerState(8, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
                            device->SetSamplerState(8, D3DSAMP_MAXANISOTROPY, 16);
                            device->SetSamplerState(8, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
                            device->SetSamplerState(8, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);
                        }
                        // Phase 8A: bind overlay's _paramh to sampler s9 so PS can blend paramh.
                        if (mb.key.overlayParamHTexture) {
                            device->SetTexture(9, mb.key.overlayParamHTexture);
                            device->SetSamplerState(9, D3DSAMP_MINFILTER, D3DTEXF_ANISOTROPIC);
                            device->SetSamplerState(9, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
                            device->SetSamplerState(9, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
                            device->SetSamplerState(9, D3DSAMP_MAXANISOTROPY, 16);
                            device->SetSamplerState(9, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
                            device->SetSamplerState(9, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);
                        }

                        const bool useForwardSSAO = DistantLand::forwardSSAOEnabled && DistantLand::forwardSSAOActive &&
                            (firstCall.sceneNum == 0 || firstCall.sceneNum == 2);
                        bindForwardSSAOSlot(device, nullptr, useForwardSSAO);
                        setForwardSSAOParams(device, nullptr, useForwardSSAO);
                    }

                    {
                        MGE_ZoneScopedN("MergedBatch_SetLightConstants");
                        // Set sun/ambient lighting constants using hoisted values
                        if (hlslShader.regLightSunDirection.reg != REG_INVALID) {
                            float v[4] = { sunDirView.x, sunDirView.y, sunDirView.z, 0 };
                            device->SetPixelShaderConstantF(hlslShader.regLightSunDirection.reg, v, 1);
                        }
                        if (hlslShader.regLightSunDiffuse.reg != REG_INVALID) {
                            float v[4] = { sunDiffuse.r, sunDiffuse.g, sunDiffuse.b, 0 };
                            device->SetPixelShaderConstantF(hlslShader.regLightSunDiffuse.reg, v, 1);
                        }
                        if (hlslShader.regLightSceneAmbient.reg != REG_INVALID) {
                            float v[4] = { sceneAmbient.r, sceneAmbient.g, sceneAmbient.b, 0 };
                            device->SetPixelShaderConstantF(hlslShader.regLightSceneAmbient.reg, v, 1);
                        }
                        // Set debug mode for shader visualization (hardcoded c22 - batch shaders skip dynamic resolution)
                        {
                            float v[4] = { (float)ImGuiManager::GetShaderDebugMode(), 0, 0, 0 };
                            device->SetPixelShaderConstantF(22, v, 1);
                        }
                        {
                            float v[4] = { ImGuiManager::GetHeightBlendStrength(), ImGuiManager::GetHeightBlendContrast(), 0, 0 };
                            device->SetPixelShaderConstantF(23, v, 1);
                        }
                        // c26 intensityScalar, c27 attenuationMultiplier, c28 attenuationCutoffDist
                        // c29 parallaxScale, c30 parallaxBias
                        // (statically declared in common.hlsl). Batch shaders skip dynamic resolution.
                        {
                            float v[4] = { ImGuiManager::GetIntensityScalar(), 0, 0, 0 };
                            device->SetPixelShaderConstantF(26, v, 1);
                        }
                        {
                            float v[4] = { ImGuiManager::GetAttenuationMultiplier(), 0, 0, 0 };
                            device->SetPixelShaderConstantF(27, v, 1);
                        }
                        {
                            float v[4] = { ImGuiManager::GetAttenuationCutoffDist(), 0, 0, 0 };
                            device->SetPixelShaderConstantF(28, v, 1);
                        }
                        {
                            float v[4] = { ImGuiManager::GetParallaxScale(), 0, 0, 0 };
                            device->SetPixelShaderConstantF(29, v, 1);
                        }
                        {
                            float v[4] = { ImGuiManager::GetParallaxBias(), 0, 0, 0 };
                            device->SetPixelShaderConstantF(30, v, 1);
                        }
                        // Set shadow matrices using hoisted view-to-shadow transforms
                        if (hlslShader.hShadowWorldViewProj) {
                            hlslShader.vsConstantTable->SetMatrixArray(device, hlslShader.hShadowWorldViewProj, shadowViewToClip, 2);
                        }
                        for (int i = 0; i < kShadowCascadeCount; ++i) {
                            setMatrixConstantF(nullptr, device, false, 31 + i * 4, shadowViewToClip[i]);
                        }
                        {
                            float shadowCascadeDepths[4];
                            computeShadowCascadeDepths(shadowCascadeDepths);
                            device->SetPixelShaderConstantF(43, shadowCascadeDepths, 1);
                            setCloseCascadeShadowParams(nullptr, device);
                        }

                        // Shadow PS constants - use resolved registers to avoid conflicts.
                        // Only set if shader has shadow constants (sk.hasShadows matches HAS_SHADOWS).
                        if (sk.hasShadows) {
                            // Use resolved registers when available, hardcoded as fallback
                            auto setIfValid = [&](ConstReg reg, int fallback, float val) {
                                float v[4] = { val, 0, 0, 0 };
                                device->SetPixelShaderConstantF(reg.reg != REG_INVALID ? reg.reg : fallback, v, 1);
                            };
                            setIfValid(hlslShader.regShadowRcpRes, 10, 1.0f / Configuration.DL.ShadowResolution);
                            setIfValid(hlslShader.regPCFBias, 11, ImGuiManager::GetPCFBias());
                            setIfValid(hlslShader.regPCFBias2, 12, ImGuiManager::GetPCFBias2());
                            setIfValid(hlslShader.regPCFFilterSize, 13, ImGuiManager::GetPCFFilterSize());
                            setIfValid(hlslShader.regPCFPenumbraScale, 14, ImGuiManager::GetPCFPenumbraScale());
                            setIfValid(hlslShader.regPCFMinPenumbra, 15, ImGuiManager::GetPCFMinPenumbra());
                            setIfValid(hlslShader.regPCFMaxPenumbra, 16, ImGuiManager::GetPCFMaxPenumbra());
                            setIfValid(hlslShader.regPCFSlopeBias, 17, ImGuiManager::GetPCFSlopeBias());

                            // Terrain shadow params. terrainShadowParams is pinned
                            // to PS c25; merged batches may not have resolved the
                            // dynamic register cache before reaching this path.
                            float isTerrain = (mb.key.bin == (uint8_t)RenderBin::Terrain) ? 1.0f : 0.0f;
                            setTerrainShadowParams(nullptr, device, hlslShader.regTerrainShadowParams, isTerrain);
                        }
                    }

                    {
                        MGE_ZoneScopedN("MergedBatch_SetRenderState");
                        // Set render state
                        device->SetRenderState(D3DRS_ALPHABLENDENABLE, mb.key.blendState & 0x1);
                        device->SetRenderState(D3DRS_SRCBLEND, (mb.key.blendState >> 4) & 0xF);
                        device->SetRenderState(D3DRS_DESTBLEND, (mb.key.blendState >> 8) & 0xF);
                        device->SetRenderState(D3DRS_ZENABLE, mb.key.zState & 0x3);
                        device->SetRenderState(D3DRS_ZWRITEENABLE, (mb.key.zState >> 2) & 0x1);
                        device->SetRenderState(D3DRS_CULLMODE, mb.key.cullMode);
                        // Match non-batch path's state surface: without these, stale ALPHATESTENABLE
                        // from a preceding alpha-tested draw kills partial-alpha TerrainBlend overlays.
                        if (firstCall.expectedState.captured) {
                            device->SetRenderState(D3DRS_ALPHATESTENABLE, firstCall.expectedState.alphaTestEnable);
                            device->SetRenderState(D3DRS_FOGENABLE, firstCall.expectedState.fogEnable);
                        }
                        // Fill mode: `twf` flips this to WIREFRAME globally, but postshaders
                        // force SOLID between scenes so we re-apply from the captured state.
                        device->SetRenderState(D3DRS_FILLMODE, firstCall.deviceState.fillMode);
                    }

                    {
                        MGE_ZoneScopedN("MergedBatch_SetStreamsAndSubmit");
                        // Set up vertex stream - offset 0, use BaseVertexIndex for batch positioning
                        device->SetVertexDeclaration(mergedDecl);
                        device->SetStreamSource(0, useVB, 0, di.expandedStride);
                        device->SetStreamSourceFreq(0, 1);  // Not instanced
                        device->SetStreamSource(1, nullptr, 0, 0);

                        // Draw merged geometry
                        // Use BaseVertexIndex to offset into merged VB (avoids double-offset with stream offset)
                        // Indices are local to batch (rebased during copy)
                        INT baseVertex = (INT)(di.vbByteOffset / di.expandedStride);
                        device->DrawIndexedPrimitive(
                            D3DPT_TRIANGLELIST,
                            baseVertex,           // BaseVertexIndex - added to each index by GPU
                            0,                    // MinIndex
                            di.vertCount,
                            di.ibStartIndex,
                            di.primCount
                        );
                    }

                    mergedDrawCalls++;
                }
            }
            // NOTE: Texture cleanup moved to consolidated section after both batch types
        }
    }

    // Clean up draw data texture after merged batch draws
    if (sceneCount == 0 && !fb.mergedBatches.empty()) {
        device->SetTexture(D3DVERTEXTEXTURESAMPLER0, nullptr);
        device->SetTexture(6, nullptr);
        device->SetTexture(8, nullptr);
    }

    MGE_ZoneScopedN("replay_MainLoop");

    // Phase 8.7: per-frame edge-height coalescing and SubdivPatch builds are now
    // done once in renderStage1 via PatchDisplacement::prebuildNearPatches(),
    // before both the depth prepass and this color replay. This loop only does
    // findCached lookups so depth and color cannot race the cache builder.

    // Phase 8.4: push displacement falloff (c73) once per replay invocation.
    // XY = (R_outer, R_inner), ZW = world-space camera XY. VS samples this only
    // under HAS_DISPLACEMENT, so pushing unconditionally is safe; non-displaced
    // draws ignore it. Eye XY derived from fb.currentView for N-1 consistency,
    // matching the selector in hiz_culling.cpp.
    {
        D3DXMATRIX invView;
        D3DXMatrixInverse(&invView, nullptr, &fb.currentView);
        D3DXVECTOR4 origin(0.0f, 0.0f, 0.0f, 1.0f);
        D3DXVECTOR4 eyeW;
        D3DXVec4Transform(&eyeW, &origin, &invView);
        float falloff[4] = { 2560.0f, 1280.0f, eyeW.x, eyeW.y };
        device->SetVertexShaderConstantF(73, falloff, 1);
    }

    // Push HLSL FFE near-fog params (c49=nearFogStart, c50=nearFogRange) per
    // replay so XE FixedFuncEmu_VS::fogMWScalar reaches 0 at MW view distance
    // — same point DL begins blending. Without this push the registers stay
    // stale and input.fog falls apart, mismatching the DL horizon.
    {
        float fogParams[8] = {
            DistantLand::s_staging.fogNearStart, 0, 0, 0,
            DistantLand::s_staging.fogNearEnd,   0, 0, 0,
        };
        device->SetVertexShaderConstantF(49, fogParams, 2);
    }

    FixedFunctionShader::logCellCrossFrame(sceneCount == 0 ? "REPLAY0-BEGIN" : "REPLAY-BEGIN", fb);

    bool firstDrawDone = false;
    int cellxRenderedDisplaced = 0;
    int cellxDisplacementNotInSet = 0;
    int cellxDisplacementBuildNull = 0;
    int cellxLoweredTerrain = 0;
    int cellxFlatTerrain = 0;
    for (size_t i = 0; i < numCalls; i++) {
        // Skip calls that were handled by merged batches
        if (mergedBatchIndices.count(i)) continue;
        auto& call = recCalls[i];  // Non-const to update shader key

        // Phase 5A: TerrainBlend overlays are absorbed into their paired Terrain draws;
        // the blend color is composited in the pixel shader via sampler s8. Never draw
        // the blend tile as a separate geometry pass.
        if (call.bin == RenderBin::TerrainBlend) continue;

        // Force lightMode 3 for lit Opaque/Terrain when toggle enabled (before rendering)
        if (ImGuiManager::GetForceLightMode3() && call.sk.useLighting && call.sk.lightMode < 3 &&
            (call.bin == RenderBin::Opaque ||
             call.bin == RenderBin::Terrain ||
             call.bin == RenderBin::TerrainBlend)) {
            call.sk.lightMode = 3;
        }

        // No Z-clear between scenes: Scene 1/2 depth-tests against Scene 0.
        // Alpha-sorted particles (Scene 1) properly occlude behind world geometry.

        // Track bin statistics
        binCounts[(int)call.bin]++;

        // Per-bin suppress check (Scene 0 only - Scene 2 hands have separate 1P checkboxes)
        switch (call.bin) {
            case RenderBin::Terrain:
            case RenderBin::TerrainBlend:
                                        if (sceneCount == 0 && ImGuiManager::GetSuppressTerrain()) continue; break;
            case RenderBin::Opaque:     if (sceneCount == 0 && ImGuiManager::GetSuppressOpaque()) continue; break;
            case RenderBin::Skinning:   if (sceneCount == 0 && ImGuiManager::GetSuppressSkinning()) continue; break;
            case RenderBin::Grass:      if (sceneCount == 0 && ImGuiManager::GetSuppressGrass()) continue; break;
            case RenderBin::AlphaTested:if (sceneCount == 0 && ImGuiManager::GetSuppressAlphaTested()) continue; break;
            case RenderBin::Blending:   if (sceneCount == 0 && ImGuiManager::GetSuppressBlending()) continue; break;
            default: break;
        }

#ifdef TRACY_ENABLE
        // Emit Tracy zone on bin transition
        if (g_tracyActive && call.bin != currentBin) {
            if (binZone) binZone->~ScopedZone();
            currentBin = call.bin;
            binZone = new (binZoneBuf) tracy::ScopedZone(&binZoneSrcLoc, TRACY_CALLSTACK, true);
            const char* name = binNames[(int)currentBin];
            binZone->Name(name, strlen(name));
        }
#endif

        // Visibility culling: shouldRender pre-set by executeHiZCulling()
        if (!call.shouldRender) {
            culledCalls++;
            continue;
        }

        // Material metrics tracking (for sorting optimization analysis) - Scene 0 only
        if (sceneCount == 0) {
            MaterialKey curMaterial;
            curMaterial.texture = call.rs.texture;
            curMaterial.blendState = (uint16_t)((call.expectedState.captured ? call.expectedState.alphaBlendEnable : 0) |
                                     ((call.expectedState.captured ? call.expectedState.srcBlend : 0) << 4) |
                                     ((call.expectedState.captured ? call.expectedState.destBlend : 0) << 8));
            curMaterial.alphaState = (uint16_t)(call.expectedState.captured ? call.expectedState.alphaTestEnable : 0);
            curMaterial.zState = (uint8_t)((call.expectedState.captured ? call.expectedState.zEnable : 1) |
                                 ((call.expectedState.captured ? call.expectedState.zWriteEnable : 1) << 2));
            curMaterial.cullMode = call.expectedState.captured ? (uint8_t)call.expectedState.cullMode : D3DCULL_CW;
            curMaterial.bin = call.bin;

            seenTextures.insert(call.rs.texture);
            seenMaterials.insert(curMaterial);
            if (hasPrevMaterial && !(curMaterial == prevMaterial)) {
                materialTransitions++;
            }
            prevMaterial = curMaterial;
            hasPrevMaterial = true;
            drawCallCount++;
        }

        {
            // Restore sampler states for this call - ALWAYS set to prevent leaking
            // Use captured values if available, otherwise default to WRAP
            for (int stage = 0; stage < 2; ++stage) {  // Only stages 0-1 are MW textures
                DWORD addrU = call.samplerStates[stage].captured ? call.samplerStates[stage].addressU : D3DTADDRESS_WRAP;
                DWORD addrV = call.samplerStates[stage].captured ? call.samplerStates[stage].addressV : D3DTADDRESS_WRAP;
                if (cmdBuf) {
                    cmdBuf->recordSetSamplerState(stage, D3DSAMP_ADDRESSU, addrU);
                    cmdBuf->recordSetSamplerState(stage, D3DSAMP_ADDRESSV, addrV);
                } else {
                    device->SetSamplerState(stage, D3DSAMP_ADDRESSU, addrU);
                    device->SetSamplerState(stage, D3DSAMP_ADDRESSV, addrV);
                }
            }

            // Occluder highlighting: tint occluders green for debug visualization
            if (ImGuiManager::GetHighlightOccluders()) {
                MeshKey highlightKey;
                highlightKey.vb = call.rs.vb;
                highlightKey.ib = call.rs.ib;
                highlightKey.fvf = call.rs.fvf;
                highlightKey.baseIndex = call.rs.baseIndex;  // Note: baseIndex, not ibBase!
                highlightKey.vertCount = call.rs.vertCount;
                highlightKey.startIndex = call.rs.startIndex;
                highlightKey.primCount = call.rs.primCount;

                bool isOccluder = (rasterizedOccluderMeshes.find(highlightKey) != rasterizedOccluderMeshes.end());
                if (isOccluder) {
                    // Create a modified FragmentState with green emissive tint
                    FragmentState tintedFrs = call.frs;
                    tintedFrs.material.emissive.r = 0.0f;
                    tintedFrs.material.emissive.g = 0.4f;
                    tintedFrs.material.emissive.b = 0.0f;
                    tintedFrs.material.emissive.a = 1.0f;
                    renderMorrowindHLSL_Internal(&call.rs, &tintedFrs, call.lightrs.get(), DIRTY_ALL, -1, cmdBuf, &call.deviceState, &call);
                    continue;
                }
            }

            // Render bin highlighting: tint by bin category
            if (ImGuiManager::GetHighlightBins() && call.bin != RenderBin::Opaque) {
                FragmentState tintedFrs = call.frs;
                switch (call.bin) {
                    case RenderBin::Skinning:
                        tintedFrs.material.emissive = {0.0f, 0.0f, 1.0f, 1.0f}; break; // Blue
                    case RenderBin::Grass:
                        tintedFrs.material.emissive = {0.0f, 1.0f, 0.0f, 1.0f}; break; // Green
                    case RenderBin::AlphaTested:
                        tintedFrs.material.emissive = {1.0f, 1.0f, 0.0f, 1.0f}; break; // Yellow
                    case RenderBin::Blending:
                        tintedFrs.material.emissive = {1.0f, 0.0f, 1.0f, 1.0f}; break; // Magenta
                    default: break;
                }
                renderMorrowindHLSL_Internal(&call.rs, &tintedFrs, call.lightrs.get(), DIRTY_ALL, -1, cmdBuf, &call.deviceState, &call);
                continue;
            }

            {
                MGE_ZoneScopedN("replay_RenderCall");
                // Restore Morrowind-recorded device state before each replay call
                if (i > 0 && !cmdBuf && ImGuiManager::GetStateLeakDetection() && call.expectedState.captured) {
                    validateDeviceState(call.expectedState, i);
                }
                if (call.expectedState.captured) {
                    if (cmdBuf) {
                        cmdBuf->recordSetRenderState(D3DRS_ALPHABLENDENABLE, call.expectedState.alphaBlendEnable);
                        cmdBuf->recordSetRenderState(D3DRS_ALPHATESTENABLE, call.expectedState.alphaTestEnable);
                        cmdBuf->recordSetRenderState(D3DRS_ZENABLE, call.expectedState.zEnable);
                        cmdBuf->recordSetRenderState(D3DRS_ZWRITEENABLE, call.expectedState.zWriteEnable);
                        cmdBuf->recordSetRenderState(D3DRS_CULLMODE, call.expectedState.cullMode);
                        cmdBuf->recordSetRenderState(D3DRS_SRCBLEND, call.expectedState.srcBlend);
                        cmdBuf->recordSetRenderState(D3DRS_DESTBLEND, call.expectedState.destBlend);
                        cmdBuf->recordSetRenderState(D3DRS_FOGENABLE, call.expectedState.fogEnable);
                    } else {
                        device->SetRenderState(D3DRS_ALPHABLENDENABLE, call.expectedState.alphaBlendEnable);
                        device->SetRenderState(D3DRS_ALPHATESTENABLE, call.expectedState.alphaTestEnable);
                        device->SetRenderState(D3DRS_ZENABLE, call.expectedState.zEnable);
                        device->SetRenderState(D3DRS_ZWRITEENABLE, call.expectedState.zWriteEnable);
                        device->SetRenderState(D3DRS_CULLMODE, call.expectedState.cullMode);
                        device->SetRenderState(D3DRS_SRCBLEND, call.expectedState.srcBlend);
                        device->SetRenderState(D3DRS_DESTBLEND, call.expectedState.destBlend);
                        device->SetRenderState(D3DRS_FOGENABLE, call.expectedState.fogEnable);
                        materialCache.reset();
                    }
                }
                LARGE_INTEGER callStartQPC, callEndQPC;
                QueryPerformanceCounter(&callStartQPC);

                // Phase 7: near-camera landscape patch — swap in the dense subdivided
                // VB/IB and use the HAS_DISPLACEMENT shader variant. The
                // subdivided perimeter carries zero heights, so shared edges with
                // non-subdivided neighbors still sit on the original Z — no cracks.
                // Only Terrain calls flagged hasDisplacement during prepare land here.
                bool renderedDisplaced = false;
                if (call.sk.hasDisplacement && !cmdBuf) {
                    FixedFunctionShader::TerrainPatchKey pk{call.rs.vb, call.rs.ib};
                    bool inSet = false;
                    for (uint32_t s = 0; s < fb.nearPatchCount; ++s) {
                        if (fb.nearPatches[s] == pk) { inSet = true; break; }
                    }
                    if (inSet) {
                        PatchDisplacement::SubdivPatch* sp = PatchDisplacement::findCached(
                            pk, call, call.overlayTexture,
                            ImGuiManager::GetDisplacementScale(),
                            call.subdivTier,
                            ImGuiManager::GetDisplacementGamma(),
                            ImGuiManager::GetDisplacementPivot());
                        if (!sp) {
                            ++cellxDisplacementBuildNull;
                            // Cache miss in color path. Prebuild in renderStage1
                            // is the sole builder, so a miss here means key
                            // params changed between prebuild and replay (or
                            // the tile entered the near-set after prebuild).
                            LOG::logline("[DISPCACHE][COLOR-MISS] idx=%u vb=%p ib=%p tier=%u neigh=%02x ctxHash=%08x scale=%.2f gamma=%.2f pivot=%.2f overlay=%p",
                                         (unsigned)i, call.rs.vb, call.rs.ib,
                                         (unsigned)call.subdivTier,
                                         (unsigned)call.subdivNeighborDirMask,
                                         (unsigned)call.edgeContextHash,
                                         ImGuiManager::GetDisplacementScale(),
                                         ImGuiManager::GetDisplacementGamma(),
                                         ImGuiManager::GetDisplacementPivot(),
                                         call.overlayTexture);
                        }
                        if (sp) {
                            // Optional debug tint: yellow emissive so the 4 near patches are
                            // visually obvious and we can confirm selection tracks the camera.
                            FragmentState tintedFrs = call.frs;
                            if (ImGuiManager::GetDebugHighlightNearPatches()) {
                                tintedFrs.material.emissive = {1.0f, 1.0f, 0.0f, 1.0f};
                            }
                            RenderedState rsDisp = call.rs;
                            rsDisp.vb = sp->vb;
                            rsDisp.vbOffset = 0;
                            rsDisp.vbStride = sp->stride;
                            rsDisp.ib = sp->ib;
                            rsDisp.fvf = sp->fvf;
                            rsDisp.ibBase = 0;
                            rsDisp.baseIndex = 0;
                            rsDisp.minIndex = 0;
                            rsDisp.vertCount = sp->vertCount;
                            rsDisp.startIndex = 0;
                            rsDisp.primCount = sp->primCount;
                            D3DXMatrixMultiply(&rsDisp.worldViewTransforms[0],
                                               &rsDisp.worldTransforms[0], &fb.currentView);
                            for (int i = 0; i < kShadowCascadeCount; ++i) {
                            rsDisp.shadowWorldViewProj[i] = rsDisp.worldTransforms[0] * DistantLand::s_staging.smViewproj[i];
                        }
                            renderMorrowindHLSL_Internal(&rsDisp, &tintedFrs, call.lightrs.get(),
                                                         DIRTY_ALL, (int)i, cmdBuf, &call.deviceState, &call);
                            renderedDisplaced = true;
                            ++cellxRenderedDisplaced;
                        }
                        // Cache-miss tally + diagnostics now handled in the !sp
                        // block above (always-on, not gated on cellCross).
                    } else {
                        ++cellxDisplacementNotInSet;
                        if (FixedFunctionShader::cellCrossDiagnosticsActive()) {
                            LOG::logline("[CELLX][REPLAY0] disp-not-in-near-set idx=%u vb=%p ib=%p near=%u",
                                         (unsigned)i, call.rs.vb, call.rs.ib, fb.nearPatchCount);
                        }
                    }
                }
                if (!renderedDisplaced) {
                    renderMorrowindHLSL_Internal(&call.rs, &call.frs, call.lightrs.get(), call.dirtyFlags, (int)i, cmdBuf, &call.deviceState, &call);
                    if (call.bin == RenderBin::Terrain) ++cellxFlatTerrain;
                }
                QueryPerformanceCounter(&callEndQPC);
                float callMs = (callEndQPC.QuadPart - callStartQPC.QuadPart) * 1000.0f / replayFreqQPC.QuadPart;
                if (callMs > worstCallMs) {
                    worstCallMs = callMs;
                    worstCallIndex = (int)i;
                    worstCallPrims = call.rs.primCount;
                    worstCallBin = (int)call.bin;
                }
                if (!firstDrawDone) {
                    MGE_ZoneScopedN("replay_FirstDrawDone");
                    firstDrawDone = true;
                }
            }
        }
    }

#ifdef TRACY_ENABLE
    // Close final bin zone
    if (binZone) binZone->~ScopedZone();
#endif

    if (sceneCount == 0 && FixedFunctionShader::cellCrossDiagnosticsActive()) {
        LOG::logline("[CELLX][REPLAY0-END] renderedDisp=%d buildNull=%d notInNearSet=%d loweredTerrain=%d flatTerrain=%d near=%u",
                     cellxRenderedDisplaced, cellxDisplacementBuildNull,
                     cellxDisplacementNotInSet, cellxLoweredTerrain,
                     cellxFlatTerrain, fb.nearPatchCount);
    }


    // Store material sorting metrics for ImGui display (Scene 0 only)
    if (sceneCount == 0) {
        g_replayMetrics.uniqueTextures = (int)seenTextures.size();
        g_replayMetrics.uniqueMaterialKeys = (int)seenMaterials.size();
        g_replayMetrics.materialTransitions = materialTransitions;
        g_replayMetrics.totalDrawCalls = drawCallCount;
        // Instancing metrics removed - stateless batching doesn't use InstanceKey
        g_replayMetrics.uniqueInstanceKeys = 0;
        g_replayMetrics.potentialBatches = 0;
        g_replayMetrics.totalBatchableDraws = 0;
    }

    // Slow frame detection: measure total replay time
    QueryPerformanceCounter(&replayEndQPC);
    float replayMs = (replayEndQPC.QuadPart - replayStartQPC.QuadPart) * 1000.0f / replayFreqQPC.QuadPart;
    float slowFrameThreshold = ImGuiManager::GetSlowFrameThreshold();
    if (replayMs > slowFrameThreshold) {
        LOG::logline("SLOW REPLAY: %.1fms for %d calls", replayMs, (int)numCalls);
    }

    // Auto-freeze on slow frame (prepare timing comes from prepareRecordedCalls via stored value)
    if (ImGuiManager::GetSlowFrameAutoFreeze() && !ImGuiManager::GetSlowFrameFrozen()) {
        if (lastPrepareMs > slowFrameThreshold || replayMs > slowFrameThreshold) {
            ImGuiManager::FreezeSlowFrame(lastPrepareMs, replayMs, worstCallIndex, worstCallMs, worstCallPrims, worstCallBin);
        }
    }

    // Log culling statistics (shouldRender pre-set by executeHiZCulling)
    int renderedCalls = totalCalls - culledCalls;
    LOG_CAT(LOG::Cat_HLSLReplay, "Hi-Z Replay: %d total, %d culled (%.1f%%), %d rendered",
                 totalCalls, culledCalls,
                 totalCalls > 0 ? (culledCalls * 100.0f) / totalCalls : 0.0f,
                 renderedCalls);

    // Log bin statistics
    LOG_CAT(LOG::Cat_HLSLReplay, "Bins: Terrain=%d TerrainBlend=%d Opaque=%d Skinning=%d Grass=%d AlphaTested=%d Blending=%d",
                 binCounts[(int)RenderBin::Terrain],
                 binCounts[(int)RenderBin::TerrainBlend],
                 binCounts[(int)RenderBin::Opaque], binCounts[(int)RenderBin::Skinning],
                 binCounts[(int)RenderBin::Grass], binCounts[(int)RenderBin::AlphaTested],
                 binCounts[(int)RenderBin::Blending]);

    // Feed per-bin counts back to ImGui DIP stats (TerrainBlend rolls into the terrain bucket — same suppression toggle)
    ImGuiManager::UpdateReplayBinCounts(
        binCounts[(int)RenderBin::Terrain] + binCounts[(int)RenderBin::TerrainBlend],
        binCounts[(int)RenderBin::Opaque],
        binCounts[(int)RenderBin::Skinning],
        binCounts[(int)RenderBin::Grass],
        binCounts[(int)RenderBin::AlphaTested],
        binCounts[(int)RenderBin::Blending]);

    // Per-frame light summary: scan all calls for point light statistics and mode distribution
    if (LOG::catEnabled(LOG::Cat_HLSLReplay)) {
        int minPL = INT_MAX, maxPL = 0;
        double sumPL = 0;
        int litCalls = 0;
        int modeCounts[4] = {0, 0, 0, 0};
        for (const auto& call : recCalls) {
            if (call.sk.lightMode < 4) modeCounts[call.sk.lightMode]++;
            if (!call.lightrs) continue;
            int pl = 0;
            for (DWORD id : call.lightrs->active) {
                auto it = call.lightrs->lights.find(id);
                if (it != call.lightrs->lights.end() && it->second.type == D3DLIGHT_POINT) pl++;
            }
            if (pl < minPL) minPL = pl;
            if (pl > maxPL) maxPL = pl;
            sumPL += pl;
            litCalls++;
        }
        if (litCalls == 0) minPL = 0;
        LOG::logline("Lights: %d scene, perObj: min=%d max=%d avg=%.1f (%d calls) modes:[%d,%d,%d,%d]",
            (int)sceneLights.size(), minPL, maxPL,
            litCalls > 0 ? sumPL / litCalls : 0.0, litCalls,
            modeCounts[0], modeCounts[1], modeCounts[2], modeCounts[3]);
    }

    // Update ImGui debug stats (Scene 0 only)
    if (sceneCount == 0) {
        int visibleLights = 0;
        // During menu mode, Hi-Z culling uses stale matrices - treat all lights as visible for display
        if (fb.dlContext.isRenderCached || fb.postProcessData.isMenu) {
            visibleLights = (int)sceneLights.size();
        } else {
            for (const auto& light : sceneLights) {
                if (light.isVisible) visibleLights++;
            }
        }
        int culledLights = (int)sceneLights.size() - visibleLights;
        ImGuiManager::UpdateDebugStats(
            totalCalls, renderedCalls, culledCalls,
            (int)sceneLights.size(), culledLights,
            (int)DistantLand::recordMW.size(), 0
        );
    }

    // Debug visualization: Render bounding boxes with color-coded status from ImGui
    int debugBBoxMode = ImGuiManager::GetBBoxVisualizationMode();

    if (debugBBoxMode > 0) {
        // Save render states
        IDirect3DStateBlock9* savedState;
        device->CreateStateBlock(D3DSBT_ALL, &savedState);

        // Setup for line rendering - disable depth test so boxes always render
        device->SetRenderState(D3DRS_ZENABLE, FALSE);
        device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
        device->SetRenderState(D3DRS_LIGHTING, FALSE);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
        device->SetRenderState(D3DRS_FOGENABLE, FALSE); // Disable fog
        device->SetRenderState(D3DRS_AMBIENT, 0xFFFFFFFF); // Full ambient
        device->SetRenderState(D3DRS_COLORVERTEX, TRUE);
        device->SetRenderState(D3DRS_DIFFUSEMATERIALSOURCE, D3DMCS_COLOR1);
        device->SetRenderState(D3DRS_EMISSIVEMATERIALSOURCE, D3DMCS_COLOR1);
        device->SetRenderState(D3DRS_TEXTUREFACTOR, 0xFFFFFFFF); // Full white texture factor
        device->SetFVF(D3DFVF_XYZ | D3DFVF_DIFFUSE);

        // Set up transforms for world-space rendering
        D3DXMATRIX identity;
        D3DXMatrixIdentity(&identity);
        device->SetTransform(D3DTS_WORLD, &identity);
        device->SetTransform(D3DTS_VIEW, &currentView);
        device->SetTransform(D3DTS_PROJECTION, &currentProj);

        // Helper to draw bbox edges
        auto drawBBox = [&](const D3DXVECTOR3& bmin, const D3DXVECTOR3& bmax, D3DCOLOR color) {
            struct Vertex { float x, y, z; D3DCOLOR color; };
            Vertex vertices[24] = {
                // Bottom face
                {bmin.x, bmin.y, bmin.z, color}, {bmax.x, bmin.y, bmin.z, color},
                {bmax.x, bmin.y, bmin.z, color}, {bmax.x, bmax.y, bmin.z, color},
                {bmax.x, bmax.y, bmin.z, color}, {bmin.x, bmax.y, bmin.z, color},
                {bmin.x, bmax.y, bmin.z, color}, {bmin.x, bmin.y, bmin.z, color},
                // Top face
                {bmin.x, bmin.y, bmax.z, color}, {bmax.x, bmin.y, bmax.z, color},
                {bmax.x, bmin.y, bmax.z, color}, {bmax.x, bmax.y, bmax.z, color},
                {bmax.x, bmax.y, bmax.z, color}, {bmin.x, bmax.y, bmax.z, color},
                {bmin.x, bmax.y, bmax.z, color}, {bmin.x, bmin.y, bmax.z, color},
                // Vertical edges
                {bmin.x, bmin.y, bmin.z, color}, {bmin.x, bmin.y, bmax.z, color},
                {bmax.x, bmin.y, bmin.z, color}, {bmax.x, bmin.y, bmax.z, color},
                {bmax.x, bmax.y, bmin.z, color}, {bmax.x, bmax.y, bmax.z, color},
                {bmin.x, bmax.y, bmin.z, color}, {bmin.x, bmax.y, bmax.z, color},
            };
            device->DrawPrimitiveUP(D3DPT_LINELIST, 12, vertices, sizeof(Vertex));
        };

        // Mode 1: Show culled objects only (red boxes) - test directly
        if (debugBBoxMode == 1) {
            for (size_t i = 0; i < recCalls.size(); i++) {
                const auto& call = recCalls[i];
                if (!call.hasBoundingBox) continue;

                // Direct occlusion test
                bool isVisible = softwareOcclusionCuller.testBoundingBox(
                    call.bboxMin,
                    call.bboxMax,
                    currentView,
                    currentProj
                );

                // Draw only culled objects in red
                if (!isVisible) {
                    drawBBox(call.bboxMin, call.bboxMax, D3DCOLOR_ARGB(255, 255, 0, 0));
                }
            }
        }

        // Mode 2: Show culled lights only (red boxes)
        if (debugBBoxMode == 2) {
            for (const auto& light : sceneLights) {
                if (!light.isVisible) {  // Only show culled lights
                    D3DXVECTOR3 bmin = light.position - D3DXVECTOR3(light.radius, light.radius, light.radius);
                    D3DXVECTOR3 bmax = light.position + D3DXVECTOR3(light.radius, light.radius, light.radius);

                    // Red color for culled lights
                    drawBBox(bmin, bmax, D3DCOLOR_ARGB(255, 255, 0, 0));
                }
            }
        }

        // Restore render states
        savedState->Apply();
        savedState->Release();
    }

    // Single Object Mode: Show selected object's bbox in CYAN (always visible, regardless of bbox mode)
    if (ImGuiManager::hiZSingleObjectMode) {
        // Save render states
        IDirect3DStateBlock9* savedState;
        device->CreateStateBlock(D3DSBT_ALL, &savedState);

        // Setup for line rendering - disable depth test so box always renders
        device->SetRenderState(D3DRS_ZENABLE, FALSE);
        device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
        device->SetRenderState(D3DRS_LIGHTING, FALSE);
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
        device->SetRenderState(D3DRS_FOGENABLE, FALSE);
        device->SetRenderState(D3DRS_AMBIENT, 0xFFFFFFFF);
        device->SetRenderState(D3DRS_COLORVERTEX, TRUE);
        device->SetRenderState(D3DRS_DIFFUSEMATERIALSOURCE, D3DMCS_COLOR1);
        device->SetRenderState(D3DRS_EMISSIVEMATERIALSOURCE, D3DMCS_COLOR1);
        device->SetFVF(D3DFVF_XYZ | D3DFVF_DIFFUSE);

        // Set up transforms for world-space rendering
        D3DXMATRIX identity;
        D3DXMatrixIdentity(&identity);
        device->SetTransform(D3DTS_WORLD, &identity);
        device->SetTransform(D3DTS_VIEW, &currentView);
        device->SetTransform(D3DTS_PROJECTION, &currentProj);

        // Helper to draw bbox edges
        auto drawBBox = [&](const D3DXVECTOR3& bmin, const D3DXVECTOR3& bmax, D3DCOLOR color) {
            struct Vertex { float x, y, z; D3DCOLOR color; };
            Vertex vertices[24] = {
                // Bottom face
                {bmin.x, bmin.y, bmin.z, color}, {bmax.x, bmin.y, bmin.z, color},
                {bmax.x, bmin.y, bmin.z, color}, {bmax.x, bmax.y, bmin.z, color},
                {bmax.x, bmax.y, bmin.z, color}, {bmin.x, bmax.y, bmin.z, color},
                {bmin.x, bmax.y, bmin.z, color}, {bmin.x, bmin.y, bmin.z, color},
                // Top face
                {bmin.x, bmin.y, bmax.z, color}, {bmax.x, bmin.y, bmax.z, color},
                {bmax.x, bmin.y, bmax.z, color}, {bmax.x, bmax.y, bmax.z, color},
                {bmax.x, bmax.y, bmax.z, color}, {bmin.x, bmax.y, bmax.z, color},
                {bmin.x, bmax.y, bmax.z, color}, {bmin.x, bmin.y, bmax.z, color},
                // Vertical edges
                {bmin.x, bmin.y, bmin.z, color}, {bmin.x, bmin.y, bmax.z, color},
                {bmax.x, bmin.y, bmin.z, color}, {bmax.x, bmin.y, bmax.z, color},
                {bmax.x, bmax.y, bmin.z, color}, {bmax.x, bmax.y, bmax.z, color},
                {bmin.x, bmax.y, bmin.z, color}, {bmin.x, bmax.y, bmax.z, color},
            };
            device->DrawPrimitiveUP(D3DPT_LINELIST, 12, vertices, sizeof(Vertex));
        };

        // Find and draw the selected object's bbox
        int currentObjectIndex = 0;
        for (size_t i = 0; i < recCalls.size(); i++) {
            const auto& call = recCalls[i];

            // Count only opaque objects with >2 tris (matching selection logic)
            bool isOpaque = !call.rs.blendEnable && call.rs.zWrite;
            if (isOpaque && call.rs.primCount > 2) {
                if (currentObjectIndex == ImGuiManager::hiZSingleObjectIndex) {
                    if (call.hasBoundingBox) {
                        // Draw in CYAN
                        drawBBox(call.bboxMin, call.bboxMax, D3DCOLOR_ARGB(255, 0, 255, 255));
                    }
                    break;
                }
                currentObjectIndex++;
            }
        }

        // Restore render states
        savedState->Apply();
        savedState->Release();
    }

    // Clear replay shadow pointer — callers revert to s_staging
    s_activeShadowVP = nullptr;

    // Note: Camera position is already stored above for next frame's velocity calculation

    // Mark buffer as available after replay (correct buffer based on mode)
    auto& completedFb = usingN1Buffer ? getPrepBuffer() : getRenderingBuffer();
    completedFb.state = BufferState::Available;

    // Restore render states after replay completes
    if (!cmdBuf) {
        device->SetRenderState(D3DRS_ALPHABLENDENABLE, savedAlphaBlend);
        device->SetRenderState(D3DRS_ALPHATESTENABLE, savedAlphaTest);
        device->SetRenderState(D3DRS_ZENABLE, savedZEnable);
        device->SetRenderState(D3DRS_ZWRITEENABLE, savedZWrite);
        device->SetRenderState(D3DRS_ZFUNC, savedZFunc);
        device->SetRenderState(D3DRS_AMBIENTMATERIALSOURCE, savedAmbientMat);
        device->SetRenderState(D3DRS_DIFFUSEMATERIALSOURCE, savedDiffuseMat);
        device->SetRenderState(D3DRS_EMISSIVEMATERIALSOURCE, savedEmissiveMat);
    }

    isReplaying.store(false, std::memory_order_release);
}
