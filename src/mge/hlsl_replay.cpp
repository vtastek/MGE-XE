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

#include <algorithm>
#include <cstring>
#include <climits>
#include <unordered_map>

// File-scope statics used by replay functions (duplicated from ffeshader.cpp)
// During replay, points to fb.shadowViewproj (recording-time matrices).
// Outside replay, nullptr - callers fall back to s_staging.smViewproj.
static const D3DXMATRIX* s_activeShadowVP = nullptr;

// Slow frame detection: prepareMs stored by prepareRecordedCalls, read by replayRecordedCalls
extern float lastPrepareMs;

// Flag: set to false by executeCullPass (cull thread) to prevent device calls in suffix fallback
extern bool deviceCallsSafeInPrepare;

// Diagnostic: cache hit/miss logging for first N frames
static int hlslDiagFrameCounter = 0;

// Cache for texture resolutions to avoid repeated GetLevelDesc calls
static std::unordered_map<IDirect3DTexture9*, D3DXVECTOR2> textureResolutionCache;

// Sampler state cache for recording (duplicated from ffeshader.cpp for HLSLRecordedCall ctor)
static std::unordered_map<IDirect3DBaseTexture9*, std::pair<DWORD, DWORD>> samplerCache;

// Helper: get current frame's recorded calls
static auto& currentRecordedCalls() {
    return FixedFunctionShader::frameBuffers[FixedFunctionShader::recordingBuffer].recordedCalls;
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

    // Slot 0: Base texture or diffparam replacement (always used by HLSL shaders)
    IDirect3DTexture9* baseTexture = rs->texture;

    // Check for diffparam replacement if shader supports it
    if (sk.hasDiffParam) {
        const auto* cached = TextureSuffix::getCachedResolution(rs->texture);
        if (cached && cached->hasValidName && cached->variants) {
            // Use cached diffparam replacement or load once per texture change
            if (bindState.currentBaseTextureName == cached->textureName &&
                bindState.boundDiffParam) {
                // Use cached replacement
                baseTexture = bindState.boundDiffParam;
            } else {
                // Load replacement texture once for this base texture
                IDirect3DTexture9* replacementTexture = nullptr;
                if (cached->variants->hasDiffParamT()) {
                    replacementTexture = BSA::loadSuffixTexture((IDirect3DDevice9*)device,
                                                              *cached->variants, "diffparam_t");
                }
                if (!replacementTexture && cached->variants->hasDiffParam()) {
                    replacementTexture = BSA::loadSuffixTexture((IDirect3DDevice9*)device,
                                                              *cached->variants, "diffparam");
                }

                if (replacementTexture) {
                    baseTexture = replacementTexture;
                    // Cache this replacement for future use with same base texture
                    bindState.boundDiffParam = replacementTexture;
                }
            }
        }
    }

    // Use sampler preservation if base texture is a diffparam replacement
    if (sk.hasDiffParam && baseTexture != rs->texture) {
        setCachedTextureWithSamplerPreservation(device, 0, baseTexture);
    } else {
        setCachedTexture(device, 0, baseTexture);
    }

    // Slot 1: Detail texture (conditional only - with ifdef support)
    if (sk.hasDetail && savedOriginalDetailTexture) {
        setCachedTexture(device, 1, static_cast<IDirect3DTexture9*>(savedOriginalDetailTexture));
    }

    // Slots 2-3: Suffix textures (only if shader has suffix support)
    if (sk.hasDiffParam || sk.hasParamH || sk.hasParamX) {
        // Fast check: if same texture pointer, skip all expensive operations
        if (bindState.lastBaseTexture != rs->texture) {
            // Get or create resolution cache entry (may perform expensive hash calculation)
            const auto* cached = TextureSuffix::getOrCreateResolution((IDirect3DDevice9*)device, rs->texture, true);

            if (cached && cached->hasValidName) {
                if (bindState.currentBaseTextureName != cached->textureName) {
                    // Reset cache when texture changes
                    bindState.currentBaseTextureName = cached->textureName;
                    bindState.boundDiffParam = nullptr;
                    bindState.boundParamH = nullptr;
                    bindState.boundParamX = nullptr;

                    if (cached->variants) {
                        // Slot 2: ParamH (metallic/roughness) - load once per texture change
                        if (sk.hasParamH && cached->variants->hasParamH()) {
                            if (!bindState.boundParamH) {
                                bindState.boundParamH = BSA::loadSuffixTexture((IDirect3DDevice9*)device, *cached->variants, "paramh");
                            }
                            if (bindState.boundParamH) {
                                setCachedTextureWithSamplerPreservation(device, 2, bindState.boundParamH);
                            }
                        }

                        // Slot 3: ParamX (anisotropic) - load once per texture change
                        if (sk.hasParamX && cached->variants->hasParamX()) {
                            if (!bindState.boundParamX) {
                                bindState.boundParamX = BSA::loadSuffixTexture((IDirect3DDevice9*)device, *cached->variants, "paramx");
                            }
                            if (bindState.boundParamX) {
                                setCachedTextureWithSamplerPreservation(device, 3, bindState.boundParamX);
                            }
                        }
                    }
                }
                bindState.lastBaseTexture = rs->texture;
            }
        }
    }

    // Slot 4: Shadow map
    if (sk.hasShadows) {
        setCachedTexture(device, 4, DistantLand::texSoftShadow);
    }

    // Slot 5: Light data texture (for texture-based point lighting)
    // Per-object packed texture takes priority (mode 3 spatial query)
    if (texPerObjectLightData) {
        setCachedTexture(device, 5, texPerObjectLightData);
    } else if (DistantLand::texLightData) {
        setCachedTexture(device, 5, DistantLand::texLightData);
    }
}


// Helper function to compute ShaderKey with texture suffix detection
FixedFunctionShader::ShaderKey FixedFunctionShader::computeShaderKeyWithSuffixes(const RenderedState* rs, const FragmentState* frs, LightState* lightrs) {

    // Step 1: Determine texture suffix availability
    bool hasDiffParam = false, hasParamH = false, hasParamX = false, hasGrass = false;

    if (rs->texture) {
        // Use TextureSuffix module for thread-safe cache lookup/creation
        // deviceCallsSafeInPrepare controls whether device calls are allowed
        const auto* cached = TextureSuffix::getOrCreateResolution(
            (IDirect3DDevice9*)device, rs->texture, deviceCallsSafeInPrepare);

        if (cached && cached->hasValidName && cached->variants) {
            hasDiffParam = cached->variants->hasDiffParam() || cached->variants->hasDiffParamT();
            hasParamH = cached->variants->hasParamH();
            hasParamX = cached->variants->hasParamX();
            hasGrass = cached->variants->hasGrass();
        }
    }

    // Step 2: Create ShaderKey with suffix flags
    ShaderKey sk(rs, frs, lightrs);
    sk.hasDiffParam = hasDiffParam;
    sk.hasParamH = hasParamH;
    sk.hasParamX = hasParamX;
    sk.hasGrass = hasGrass;

    // Set shadow flag based on MGE configuration
    sk.hasShadows = ((Configuration.MGEFlags & USE_SHADOWS) && (Configuration.MGEFlags & USE_DISTANT_LAND)) ? 1 : 0;

    return sk;
}

// HLSL Pipeline Implementation
void FixedFunctionShader::renderMorrowindHLSL(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, int recordMWIdx) {
    // Skip if we're in replay mode to avoid recursion
    if (isReplaying) {
        // During replay mode, perform actual rendering with this specific call
        renderMorrowindHLSL_Internal(rs, frs, lightrs);
        return;
    }

    // Start recording at first HLSL call if not already recording (unless under manual control or disabled)
    // Recording happens for ALL modes (standard, PPL, HLSL) - renderStageBlend is designed to run
    // before replay, compositing distant land based on depth before the scene is drawn.
    if (!isRecording && !isReplaying && !manualRecordingControl && recordingEnabled && !recordingCompletedThisFrame && ImGuiManager::GetEnableRecording()) {
        startRecording();
    }

    // If recording is active, record the call for batched replay
    if (isRecording && ImGuiManager::GetEnableRecording()) {
        // Create a copy of rs and add CURRENT shadow world-view-projection matrices for this draw call
        // During recording, use current matrices; during replay, these will be the "recorded" matrices
        RenderedState rsWithShadows = *rs;

        // Use current shadow matrices for this specific draw call during recording
        rsWithShadows.shadowWorldViewProj[0] = rs->worldTransforms[0] * DistantLand::s_staging.smViewproj[0];
        rsWithShadows.shadowWorldViewProj[1] = rs->worldTransforms[0] * DistantLand::s_staging.smViewproj[1];

        // Defer shader key computation to prepare phase (avoid texture hash lookups during recording)
        ShaderKey sk;
        memset(&sk, 0, sizeof(sk));  // Placeholder - computed in prepareRecordedCalls()
        recordRenderCall(&rsWithShadows, frs, lightrs, sk, recordMWIdx);
        return;
    }

    // Immediate path for hands (Scene 1+) and fallback when recording disabled.
    // Uses game's built-in 8 lights via lightrs, but needs shadow matrices computed.
    if (!ImGuiManager::GetEnableImmediateRendering()) {
        return;  // Skip immediate rendering if disabled
    }

    // Compute shadow world-view-projection matrices for this draw call
    // (rs from mged3d8device has zeros - compute from current shadow map VP)
    RenderedState rsWithShadows = *rs;
    rsWithShadows.shadowWorldViewProj[0] = rs->worldTransforms[0] * DistantLand::s_staging.smViewproj[0];
    rsWithShadows.shadowWorldViewProj[1] = rs->worldTransforms[0] * DistantLand::s_staging.smViewproj[1];

    renderMorrowindHLSL_Internal(&rsWithShadows, frs, lightrs);
}

void FixedFunctionShader::renderMorrowindHLSL_Internal(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, DWORD dirtyFlags, int callIndex, D3DCommandBuffer* cmdBuf, const DeviceStateSnapshot* capturedState) {

    // Process any completed async shader compilations
    processAsyncCompletions();

    HLSLShader hlslShader;

    // Get ShaderKey with texture suffix detection
    ShaderKey sk;
    if (isReplaying) {
        // During replay, use the recorded ShaderKey with original suffix flags
        for (const auto& call : frameBuffers[recordingBuffer].recordedCalls) {
            if (&call.rs == rs) {
                sk = call.sk;
                break;
            }
        }
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
        }

        if (!exactHit) {
            // Diagnostic: log cache misses on early frames to identify precache gaps
            if (hlslDiagFrameCounter <= 3) {
                char buf[512];
                snprintf(buf, sizeof(buf),
                    "CACHE MISS frame=%d: lm=%d lit=%d vc=%d vm=%d hl=%d skin=%d fog=%d uv=%d stages=%d shadow=%d detail=%d dp=%d ph=%d px=%d grass=%d bump=%d tg=%d",
                    hlslDiagFrameCounter,
                    (int)sk.lightMode, (int)sk.useLighting, (int)sk.vertexColour,
                    (int)sk.vertexMaterial, (int)sk.heavyLighting,
                    (int)sk.usesSkinning, (int)sk.fogMode, (int)sk.uvSets,
                    (int)sk.activeStages,
                    (int)sk.hasShadows, (int)sk.hasDetail, (int)sk.hasDiffParam,
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
            if (!foundFallback && (sk.hasDiffParam || sk.hasParamH || sk.hasParamX)) {
                ShaderKey textureFallbackSk = sk;
                normalizeAlpha(textureFallbackSk);

                if (sk.hasDiffParam && !foundFallback) {
                    textureFallbackSk.hasDiffParam = 0;
                    auto fallbackIter = cacheHLSLShaders.find(textureFallbackSk);
                    if (fallbackIter != cacheHLSLShaders.end()) {
                        fallbackShader = fallbackIter->second;
                        foundFallback = true;
                    }
                }

                if (sk.hasParamH && !foundFallback) {
                    textureFallbackSk = sk;
                    normalizeAlpha(textureFallbackSk);
                    textureFallbackSk.hasDiffParam = 0;
                    textureFallbackSk.hasParamH = 0;
                    textureFallbackSk.hasParamX = 0;
                    auto fallbackIter = cacheHLSLShaders.find(textureFallbackSk);
                    if (fallbackIter != cacheHLSLShaders.end()) {
                        fallbackShader = fallbackIter->second;
                        foundFallback = true;
                    }
                }

                if (!foundFallback) {
                    textureFallbackSk = sk;
                    normalizeAlpha(textureFallbackSk);
                    textureFallbackSk.hasDiffParam = 0;
                    textureFallbackSk.hasParamH = 0;
                    textureFallbackSk.hasParamX = 0;
                    auto fallbackIter = cacheHLSLShaders.find(textureFallbackSk);
                    if (fallbackIter != cacheHLSLShaders.end()) {
                        fallbackShader = fallbackIter->second;
                        foundFallback = true;
                    }
                }
            }

            // Final universal fallback: try the 8 guaranteed base combinations
            if (!foundFallback) {
                ShaderKey universalSk = sk;
                normalizeAlpha(universalSk);
                universalSk.heavyLighting = 0;
                universalSk.hasDiffParam = 0;
                universalSk.hasParamH = 0;
                universalSk.hasParamX = 0;

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
    D3DXMATRIX currentView;
    if (isReplaying) {
        currentView = frameBuffers[recordingBuffer].view;
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
        static D3DXMATRIX cachedViewToShadowLocal[2];
        static bool shadowTransformValid = false;

        if (!shadowTransformValid || !shadowMatricesValid) {
            const auto* svp = s_activeShadowVP ? s_activeShadowVP : DistantLand::s_staging.smViewproj;
            cachedViewToShadowLocal[0] = cachedInverseView * svp[0];
            cachedViewToShadowLocal[1] = cachedInverseView * svp[1];
            shadowTransformValid = shadowMatricesValid;
        }

        setConstantF(cmdBuf, device, true, 20, (float*)&cachedViewToShadowLocal[0], 4); // c20-c23
        setConstantF(cmdBuf, device, true, 24, (float*)&cachedViewToShadowLocal[1], 4); // c24-c27

        // Set shadow resolution parameter
        float shadowRcpData[4] = { 1.0f / Configuration.DL.ShadowResolution, 0, 0, 0 };
        setConstantF(cmdBuf, device, false, 10, shadowRcpData, 1); // c10
    }


    // Save current render states before modifying them (only states that HLSL actually changes)
    DWORD savedAlphaBlendEnable = 0, savedAlphaTestEnable = 0;
    DWORD savedZEnable = 0, savedZWriteEnable = 0;
    DWORD savedSpecularEnable = 0, savedLocalViewer = 0, savedNormalizeNormals = 0;
    if (!cmdBuf) {
        device->GetRenderState(D3DRS_ALPHABLENDENABLE, &savedAlphaBlendEnable);
        device->GetRenderState(D3DRS_ALPHATESTENABLE, &savedAlphaTestEnable);
        device->GetRenderState(D3DRS_ZENABLE, &savedZEnable);
        device->GetRenderState(D3DRS_ZWRITEENABLE, &savedZWriteEnable);
        device->GetRenderState(D3DRS_SPECULARENABLE, &savedSpecularEnable);
        device->GetRenderState(D3DRS_LOCALVIEWER, &savedLocalViewer);
        device->GetRenderState(D3DRS_NORMALIZENORMALS, &savedNormalizeNormals);
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
    if (isReplaying) {
        projMatrix = frameBuffers[recordingBuffer].proj;
        viewMatrix = frameBuffers[recordingBuffer].view;
        worldMatrix = rs->worldTransforms[0];
    } else {
        device->GetTransform(D3DTS_PROJECTION, &projMatrix);
        device->GetTransform(D3DTS_VIEW, &viewMatrix);
        device->GetTransform(D3DTS_WORLD, &worldMatrix);
    }

    // During replay, use recorded combined matrices; during normal rendering, calculate them
    D3DXMATRIX worldViewProj, worldView;
    if (isReplaying) {
        worldViewProj = rs->worldTransforms[0] * frameBuffers[recordingBuffer].view * frameBuffers[recordingBuffer].proj;
        worldView = rs->worldTransforms[0] * frameBuffers[recordingBuffer].view;
    } else {
        worldViewProj = worldMatrix * viewMatrix * projMatrix;
        worldView = worldMatrix * viewMatrix;
    }

    // Set vertex shader constants — command buffer path uses resolved registers,
    // direct path uses constant tables (SetMatrix transposes internally)
    if (cmdBuf) {
        if (hlslShader.regWorldViewProj.reg != REG_INVALID)
            setMatrixConstantF(cmdBuf, device, true, hlslShader.regWorldViewProj.reg, worldViewProj);
        if (hlslShader.regView.reg != REG_INVALID)
            setMatrixConstantF(cmdBuf, device, true, hlslShader.regView.reg, viewMatrix);
        if (hlslShader.regProj.reg != REG_INVALID)
            setMatrixConstantF(cmdBuf, device, true, hlslShader.regProj.reg, projMatrix);
        if (hlslShader.regWorld.reg != REG_INVALID)
            setMatrixConstantF(cmdBuf, device, true, hlslShader.regWorld.reg, rs->worldTransforms[0]);
        if (hlslShader.regWorldView.reg != REG_INVALID)
            setMatrixConstantF(cmdBuf, device, true, hlslShader.regWorldView.reg, worldView);

        if (hlslShader.regVertexBlendPalette.reg != REG_INVALID) {
            D3DXMATRIX blendMatrices[4];
            if (rs->vertexBlendState > 0) {
                for (int i = 0; i < 4; i++)
                    blendMatrices[i] = rs->worldTransforms[i] * viewMatrix;
            } else {
                blendMatrices[0] = worldView;
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
        }

        if (hlslShader.regShadowWorldViewProj.reg != REG_INVALID) {
            D3DXMATRIX shadowWVP[2];
            if (isReplaying && s_activeShadowVP) {
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
                hlslShader.vsConstantTable->SetMatrix(device, hlslShader.hWorldView, &worldView);
            }
            if (hlslShader.hVertexBlendPalette) {
                if (rs->vertexBlendState > 0) {
                    D3DXMATRIX currentWorldViewTransforms[4];
                    for (int i = 0; i < 4; i++) {
                        currentWorldViewTransforms[i] = rs->worldTransforms[i] * viewMatrix;
                    }
                    hlslShader.vsConstantTable->SetMatrixArray(device, hlslShader.hVertexBlendPalette, currentWorldViewTransforms, 4);
                } else {
                    D3DXMATRIX blendMatrices[4];
                    blendMatrices[0] = worldView;
                    memset(&blendMatrices[1], 0, sizeof(D3DXMATRIX) * 3);
                    hlslShader.vsConstantTable->SetMatrixArray(device, hlslShader.hVertexBlendPalette, blendMatrices, 4);
                }
            }

            if (hlslShader.hVertexBlendState) {
                D3DXVECTOR4 blendState((float)rs->vertexBlendState, 0, 0, 0);
                hlslShader.vsConstantTable->SetVector(device, hlslShader.hVertexBlendState, &blendState);
            }

            if (hlslShader.hShadowWorldViewProj) {
                if (isReplaying && s_activeShadowVP) {
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
    DWORD checkAmbient;
    device->GetRenderState(D3DRS_AMBIENT, &checkAmbient);
    if (checkAmbient == 0xffffffff) {
        ambient.r = ambient.g = ambient.b = 1.25;
        sunDiffuse.r = sunDiffuse.g = sunDiffuse.b = 0.0;
    }

    // Get fog color (needed by both paths)
    DWORD fogColorDword = 0x808080FF;
    device->GetRenderState(D3DRS_FOGCOLOR, &fogColorDword);
    float fogColor[4] = {
        ((fogColorDword >> 16) & 0xFF) / 255.0f,
        ((fogColorDword >> 8) & 0xFF) / 255.0f,
        (fogColorDword & 0xFF) / 255.0f,
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
            resolveAndCache(hlslShader.vsConstantTable, "windVec", hlslShader.regWindVec);
            resolveAndCache(hlslShader.vsConstantTable, "time", hlslShader.regTime);
            resolveAndCache(hlslShader.psConstantTable, "normres", hlslShader.regNormres);
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
        cmdBuf->recordSetPSConstantF(hlslShader.regFogColNear.reg != REG_INVALID ? hlslShader.regFogColNear.reg : 255, fogColor, 1);

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
            static float smoothWind[2] = {0, 0};
            if (!MWBridge::get()->IsMenu()) {
                const float f = 0.02f;
                const float* wind = MWBridge::get()->GetWindVector();
                smoothWind[0] += f * (1.0f * wind[0] - smoothWind[0]);
                smoothWind[1] += f * (1.0f * wind[1] - smoothWind[1]);
            }
            float v[4] = { smoothWind[0], smoothWind[1], 0, 0 };
            cmdBuf->recordSetVSConstantF(hlslShader.regWindVec.reg, v, 1);
        }
        if (hlslShader.regTime.reg != REG_INVALID) {
            float v[4] = { MWBridge::get()->simulationTime(), 0, 0, 0 };
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

        // Normres for paramH textures
        if (sk.hasParamH && hlslShader.regNormres.reg != REG_INVALID) {
            IDirect3DBaseTexture9* normalTexture;
            device->GetTexture(2, &normalTexture);
            if (normalTexture && normalTexture->GetType() == D3DRTYPE_TEXTURE) {
                IDirect3DTexture9* tex = static_cast<IDirect3DTexture9*>(normalTexture);
                D3DXVECTOR2 normres;
                auto cacheIt = textureResolutionCache.find(tex);
                if (cacheIt != textureResolutionCache.end()) {
                    normres = cacheIt->second;
                } else {
                    D3DSURFACE_DESC desc;
                    if (SUCCEEDED(tex->GetLevelDesc(0, &desc))) {
                        normres = D3DXVECTOR2((float)desc.Width, (float)desc.Height);
                        textureResolutionCache[tex] = normres;
                    } else {
                        normres = D3DXVECTOR2(1.0f, 1.0f);
                    }
                }
                float v[4] = { normres.x, normres.y, 0, 0 };
                cmdBuf->recordSetPSConstantF(hlslShader.regNormres.reg, v, 1);
                normalTexture->Release();
            }
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

        D3DXHANDLE hFogColNear = hlslShader.psConstantTable->GetConstantByName(NULL, "fogColNear");
        if (hFogColNear) hlslShader.psConstantTable->SetVector(device, hFogColNear, (D3DXVECTOR4*)fogColor);

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
            static float smoothWind[2] = {0, 0};
            if (!MWBridge::get()->IsMenu()) {
                const float f = 0.02f;
                const float* wind = MWBridge::get()->GetWindVector();
                smoothWind[0] += f * (1.0f * wind[0] - smoothWind[0]);
                smoothWind[1] += f * (1.0f * wind[1] - smoothWind[1]);
            }
            hlslShader.vsConstantTable->SetFloatArray(device, hWindVec, smoothWind, 2);
        }
        D3DXHANDLE hTime = hlslShader.vsConstantTable->GetConstantByName(NULL, "time");
        if (hTime) hlslShader.vsConstantTable->SetFloat(device, hTime, MWBridge::get()->simulationTime());
        D3DXHANDLE hHasVCol = hlslShader.psConstantTable->GetConstantByName(NULL, "hasVCol");
        if (hHasVCol) hlslShader.psConstantTable->SetBool(device, hHasVCol, (rs->fvf & D3DFVF_DIFFUSE) != 0);
        D3DXHANDLE hMaterialAlpha = hlslShader.psConstantTable->GetConstantByName(NULL, "materialAlpha");
        if (hMaterialAlpha) hlslShader.psConstantTable->SetFloat(device, hMaterialAlpha, frs->material.diffuse.a);
        D3DXHANDLE hAlphaRef = hlslShader.psConstantTable->GetConstantByName(NULL, "alphaRef");
        if (hAlphaRef) hlslShader.psConstantTable->SetFloat(device, hAlphaRef, rs->alphaRef / 255.0f);

        if (sk.hasParamH) {
            IDirect3DBaseTexture9* normalTexture;
            device->GetTexture(2, &normalTexture);
            if (normalTexture && normalTexture->GetType() == D3DRTYPE_TEXTURE) {
                IDirect3DTexture9* tex = static_cast<IDirect3DTexture9*>(normalTexture);
                D3DXVECTOR2 normres;
                auto cacheIt = textureResolutionCache.find(tex);
                if (cacheIt != textureResolutionCache.end()) {
                    normres = cacheIt->second;
                } else {
                    D3DSURFACE_DESC desc;
                    if (SUCCEEDED(tex->GetLevelDesc(0, &desc))) {
                        normres = D3DXVECTOR2((float)desc.Width, (float)desc.Height);
                        textureResolutionCache[tex] = normres;
                    } else {
                        normres = D3DXVECTOR2(1.0f, 1.0f);
                    }
                }
                D3DXHANDLE hNormres = hlslShader.psConstantTable->GetConstantByName(NULL, "normres");
                if (hNormres) hlslShader.psConstantTable->SetFloatArray(device, hNormres, (float*)&normres, 2);
                normalTexture->Release();
            }
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

        hr = device->SetStreamSource(0, rs->vb, rs->vbOffset, rs->vbStride);
        if (FAILED(hr)) {
            return;
        }

        // Phase A: Add small depth bias to resolve Z-fighting with depth prepass
        device->SetRenderState(D3DRS_DEPTHBIAS, *(DWORD*)&(const float&)-1e-6f);
        device->SetRenderState(D3DRS_SLOPESCALEDEPTHBIAS, *(DWORD*)&(const float&)-1e-6f);

        if (rs->ib) {
            hr = device->SetIndices(rs->ib);
            if (FAILED(hr)) {
                LOG::logline("!! HLSL pipeline: failed to set index buffer, hr=%x", hr);
                return;
            }
            device->DrawIndexedPrimitive(rs->primType, rs->baseIndex, rs->minIndex, rs->vertCount, rs->startIndex, rs->primCount);
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
        if (!isReplaying) {
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
        }
    }

}

void FixedFunctionShader::replayRecordedCalls(int sceneCount, D3DCommandBuffer* cmdBuf) {
    auto& recCalls = currentRecordedCalls();
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

    isReplaying = true;

    // Use game view/proj for light transforms and debug visualization
    // In deferred pipeline, device may have UI view — use s_staging which has the game matrices
    D3DXMATRIX currentView = DistantLand::s_staging.mwView;
    D3DXMATRIX currentProj = DistantLand::s_staging.mwProj;

    // Set active shadow pointer to recording-time matrices for replay
    auto& fb = frameBuffers[recordingBuffer];
    s_activeShadowVP = fb.shadowViewproj;

    // Hi-Z culling statistics (shouldRender pre-set by executeHiZCulling)
    int totalCalls = recCalls.size();
    int culledCalls = 0;

    // Check for debug key press (Y key) - gated behind debug hotkeys toggle
    static bool debugHiZ = false;
    static int debugCallCount = 0;
    if (ImGuiManager::GetDebugKeysEnabled() && (GetAsyncKeyState('Y') & 0x8000)) {
        static bool wasPressed = false;
        if (!wasPressed) {
            debugHiZ = true;
            debugCallCount = 5; // Log next 5 culled calls
            LOG::logline(">> Hi-Z Debug: Enabled for next 5 CULLED calls");
            wasPressed = true;
        }
    } else {
        static bool wasPressed = false;
        wasPressed = false;
    }

    // Check for Hi-Z snapshot save key press (L key) - gated behind debug hotkeys toggle
    if (ImGuiManager::GetDebugKeysEnabled() && (GetAsyncKeyState('L') & 0x8000)) {
        static bool wasPressed = false;
        if (!wasPressed) {
            LOG::logline(">> L key pressed: Saving Hi-Z snapshot...");
            DistantLand::saveHiZSnapshot();
            wasPressed = true;
        }
    } else {
        static bool wasPressed = false;
        wasPressed = false;
    }

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
    int numSceneLights = (int)DistantLand::sceneLights.size();
    {
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
            if (recCalls[i].sk.lightMode != 3) continue;

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

            for (const auto& light : DistantLand::sceneLights) {
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
            if (mode3Count == 0 && !DistantLand::sceneLights.empty()) {
                float nearestDist = FLT_MAX;
                int nearestIdx = -1;
                float nearestRadius = 0;
                for (int li = 0; li < (int)DistantLand::sceneLights.size(); li++) {
                    const auto& light = DistantLand::sceneLights[li];
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

        // Upload packed light data to per-object texture
        if (totalTexels > 0) {
            // Create or resize texture if needed
            if (!texPerObjectLightData) {
                HRESULT hr = device->CreateTexture(
                    totalTexels, 1, 1, 0,
                    D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED,
                    &texPerObjectLightData, nullptr);
                if (FAILED(hr)) {
                    LOG::logline("!! Failed to create per-object light texture (hr=0x%X)", hr);
                    texPerObjectLightData = nullptr;
                }
            } else {
                D3DSURFACE_DESC desc;
                texPerObjectLightData->GetLevelDesc(0, &desc);
                if (desc.Width != (UINT)totalTexels) {
                    texPerObjectLightData->Release();
                    HRESULT hr = device->CreateTexture(
                        totalTexels, 1, 1, 0,
                        D3DFMT_A32B32G32R32F, D3DPOOL_MANAGED,
                        &texPerObjectLightData, nullptr);
                    if (FAILED(hr)) {
                        LOG::logline("!! Failed to resize per-object light texture (hr=0x%X)", hr);
                        texPerObjectLightData = nullptr;
                    }
                }
            }

            if (texPerObjectLightData) {
                D3DLOCKED_RECT locked;
                if (SUCCEEDED(texPerObjectLightData->LockRect(0, &locked, nullptr, 0))) {
                    memcpy(locked.pBits, packedLightData.data(), totalTexels * 4 * sizeof(float));
                    texPerObjectLightData->UnlockRect(0);
                }
                device->SetTexture(5, texPerObjectLightData);
            }
        }

        if (mode3Count > 0) {
            LOG::logline("Mode3 packing: %d objects, maxLights/obj=%d, totalTexels=%d (from %d scene)",
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
    static const char* binNames[] = { "Terrain", "Opaque", "Skinning", "Grass", "AlphaTested", "Blending" };

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

    MGE_ZoneScopedN("replay_MainLoop");
    bool firstDrawDone = false;
    for (size_t i = 0; i < numCalls; i++) {
        auto& call = recCalls[i];  // Non-const to update shader key
        // No Z-clear between scenes: Scene 1+ depth-tests against Scene 0.
        // Alpha-sorted objects properly occlude behind world geometry.

        // Track bin statistics
        binCounts[(int)call.bin]++;

        // Per-bin suppress check
        switch (call.bin) {
            case RenderBin::Terrain:    if (ImGuiManager::GetSuppressTerrain()) continue; break;
            case RenderBin::Opaque:     if (ImGuiManager::GetSuppressOpaque()) continue; break;
            case RenderBin::Skinning:   if (ImGuiManager::GetSuppressSkinning()) continue; break;
            case RenderBin::Grass:      if (ImGuiManager::GetSuppressGrass()) continue; break;
            case RenderBin::AlphaTested:if (ImGuiManager::GetSuppressAlphaTested()) continue; break;
            case RenderBin::Blending:   if (ImGuiManager::GetSuppressBlending()) continue; break;
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

        {
            // Restore sampler states for this call (captured during recording)
            for (int stage = 0; stage < 8; ++stage) {
                if (call.samplerStates[stage].captured) {
                    if (cmdBuf) {
                        cmdBuf->recordSetSamplerState(stage, D3DSAMP_ADDRESSU, call.samplerStates[stage].addressU);
                        cmdBuf->recordSetSamplerState(stage, D3DSAMP_ADDRESSV, call.samplerStates[stage].addressV);
                    } else {
                        device->SetSamplerState(stage, D3DSAMP_ADDRESSU, call.samplerStates[stage].addressU);
                        device->SetSamplerState(stage, D3DSAMP_ADDRESSV, call.samplerStates[stage].addressV);
                    }
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
                    renderMorrowindHLSL_Internal(&call.rs, &tintedFrs, call.lightrs.get(), DIRTY_ALL, -1, cmdBuf, &call.deviceState);
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
                renderMorrowindHLSL_Internal(&call.rs, &tintedFrs, call.lightrs.get(), DIRTY_ALL, -1, cmdBuf, &call.deviceState);
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
                renderMorrowindHLSL_Internal(&call.rs, &call.frs, call.lightrs.get(), call.dirtyFlags, (int)i, cmdBuf, &call.deviceState);
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
    LOG::logline("Hi-Z Replay: %d total, %d culled (%.1f%%), %d rendered",
                 totalCalls, culledCalls,
                 totalCalls > 0 ? (culledCalls * 100.0f) / totalCalls : 0.0f,
                 renderedCalls);

    // Log bin statistics
    LOG::logline("Bins: Terrain=%d Opaque=%d Skinning=%d Grass=%d AlphaTested=%d Blending=%d",
                 binCounts[(int)RenderBin::Terrain],
                 binCounts[(int)RenderBin::Opaque], binCounts[(int)RenderBin::Skinning],
                 binCounts[(int)RenderBin::Grass], binCounts[(int)RenderBin::AlphaTested],
                 binCounts[(int)RenderBin::Blending]);

    // Feed per-bin counts back to ImGui DIP stats (these replace the coarse Scene0 count)
    ImGuiManager::UpdateReplayBinCounts(
        binCounts[(int)RenderBin::Terrain],
        binCounts[(int)RenderBin::Opaque],
        binCounts[(int)RenderBin::Skinning],
        binCounts[(int)RenderBin::Grass],
        binCounts[(int)RenderBin::AlphaTested],
        binCounts[(int)RenderBin::Blending]);

    // Per-frame light summary: scan all calls for point light statistics and mode distribution
    {
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
            (int)DistantLand::sceneLights.size(), minPL, maxPL,
            litCalls > 0 ? sumPL / litCalls : 0.0, litCalls,
            modeCounts[0], modeCounts[1], modeCounts[2], modeCounts[3]);
    }

    // Update ImGui debug stats
    ImGuiManager::UpdateDebugStats(
        totalCalls, renderedCalls, culledCalls,
        (int)DistantLand::sceneLights.size(),
        (int)DistantLand::recordMW.size(), 0
    );

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
            for (const auto& light : DistantLand::sceneLights) {
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

    // Data is already in frameBuffers[recordingBuffer].recordedCalls (recorded directly there)
    frameBuffers[recordingBuffer].state = BufferState::Available;

    isReplaying = false;
}
