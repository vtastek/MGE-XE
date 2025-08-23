
#include "ffeshader.h"
#include "configuration.h"
#include "support/log.h"
#include "mwbridge.h"

#include <algorithm>
#include <sstream>
#include <thread>
#include <chrono>
#include <atomic>

using std::string;
using std::stringstream;
using std::unordered_map;

IDirect3DDevice* FixedFunctionShader::device;
ID3DXEffectPool* FixedFunctionShader::constantPool;
unordered_map<FixedFunctionShader::ShaderKey, ID3DXEffect*, FixedFunctionShader::ShaderKey::hasher> FixedFunctionShader::cacheEffects;
FixedFunctionShader::ShaderLRU FixedFunctionShader::shaderLRU;
ID3DXEffect* FixedFunctionShader::effectDefaultPurple;

D3DXHANDLE FixedFunctionShader::ehWorld, FixedFunctionShader::ehWorldView;
D3DXHANDLE FixedFunctionShader::ehVertexBlendState, FixedFunctionShader::ehVertexBlendPalette;
D3DXHANDLE FixedFunctionShader::ehTex0, FixedFunctionShader::ehTex1, FixedFunctionShader::ehTex2, FixedFunctionShader::ehTex3, FixedFunctionShader::ehTex4, FixedFunctionShader::ehTex5;
D3DXHANDLE FixedFunctionShader::ehMaterialDiffuse, FixedFunctionShader::ehMaterialAmbient, FixedFunctionShader::ehMaterialEmissive;
D3DXHANDLE FixedFunctionShader::ehLightSceneAmbient, FixedFunctionShader::ehLightSunDiffuse, FixedFunctionShader::ehLightDiffuse;
D3DXHANDLE FixedFunctionShader::ehLightSunDirection, FixedFunctionShader::ehLightPosition, FixedFunctionShader::ehLightAmbient;
D3DXHANDLE FixedFunctionShader::ehLightFalloffQuadratic, FixedFunctionShader::ehLightFalloffLinear, FixedFunctionShader::ehLightFalloffConstant;
D3DXHANDLE FixedFunctionShader::ehTexgenTransform, FixedFunctionShader::ehBumpMatrix, FixedFunctionShader::ehBumpLumiScaleBias;

float FixedFunctionShader::sunMultiplier, FixedFunctionShader::ambMultiplier;

// HLSL Pipeline static variables
unordered_map<FixedFunctionShader::ShaderKey, FixedFunctionShader::HLSLShader, FixedFunctionShader::ShaderKey::hasher> FixedFunctionShader::cacheHLSLShaders;
FixedFunctionShader::HLSLShaderLRU FixedFunctionShader::hlslShaderLRU;
FixedFunctionShader::HLSLShader FixedFunctionShader::hlslShaderDefaultPurple;

static string buildArgString(DWORD arg, const string& mask, const string& sampler);



bool FixedFunctionShader::init(IDirect3DDevice* d, ID3DXEffectPool* pool) {
    device = d;
    constantPool = pool;

    // Create last resort shader when a generated shader fails somehow
    const D3DXMACRO generateDefault[] = { "FFE_ERROR_MATERIAL", "", 0, 0 };
    ID3DXEffect* effect;
    ID3DXBuffer* errors;

    HRESULT hr = D3DXCreateEffectFromFile(device, "Data Files\\shaders\\core\\XE FixedFuncEmu.fx", generateDefault, 0, D3DXSHADER_OPTIMIZATION_LEVEL3|D3DXFX_LARGEADDRESSAWARE, constantPool, &effect, &errors);
    if (hr != D3D_OK) {
        if (errors) {
            LOG::write("!! Shader compile errors:\n");
            LOG::write(reinterpret_cast<const char*>(errors->GetBufferPointer()));
            LOG::write("\n");
            errors->Release();
        }
        return false;
    }

    // Use it to bind shared parameters too
    ehWorld = effect->GetParameterByName(0, "world");
    ehVertexBlendState = effect->GetParameterByName(0, "vertexBlendState");
    ehVertexBlendPalette = effect->GetParameterByName(0, "vertexBlendPalette");
    ehTex0 = effect->GetParameterByName(0, "tex0");
    ehTex1 = effect->GetParameterByName(0, "tex1");
    ehTex2 = effect->GetParameterByName(0, "tex2");
    ehTex3 = effect->GetParameterByName(0, "tex3");
    ehTex4 = effect->GetParameterByName(0, "tex4");
    ehTex5 = effect->GetParameterByName(0, "tex5");

    ehWorldView = effect->GetParameterByName(0, "worldview");
    ehMaterialDiffuse = effect->GetParameterByName(0, "materialDiffuse");
    ehMaterialAmbient = effect->GetParameterByName(0, "materialAmbient");
    ehMaterialEmissive = effect->GetParameterByName(0, "materialEmissive");
    ehLightSceneAmbient = effect->GetParameterByName(0, "lightSceneAmbient");
    ehLightSunDiffuse = effect->GetParameterByName(0, "lightSunDiffuse");
    ehLightSunDirection = effect->GetParameterByName(0, "lightSunDirection");
    ehLightDiffuse = effect->GetParameterByName(0, "lightDiffuse");
    ehLightAmbient = effect->GetParameterByName(0, "lightAmbient");
    ehLightPosition = effect->GetParameterByName(0, "lightPosition");
    ehLightFalloffQuadratic = effect->GetParameterByName(0, "lightFalloffQuadratic");
    ehLightFalloffLinear = effect->GetParameterByName(0, "lightFalloffLinear");
    ehLightFalloffConstant = effect->GetParameterByName(0, "lightFalloffConstant");
    ehTexgenTransform = effect->GetParameterByName(0, "texgenTransform");
    ehBumpMatrix = effect->GetParameterByName(0, "bumpMatrix");
    ehBumpLumiScaleBias = effect->GetParameterByName(0, "bumpLumiScaleBias");

    effectDefaultPurple = effect;
    sunMultiplier = ambMultiplier = 1.0;

    // Clear cache and LRU, important if the renderer resets
    shaderLRU.effect = nullptr;
    shaderLRU.last_sk = ShaderKey();
    cacheEffects.clear();

    // Initialize HLSL pipeline if enabled
    if (Configuration.UseHLSLPipeline) {
        LOG::logline("-- Initializing HLSL compilation pipeline");
        
        // Clear HLSL cache and LRU
        hlslShaderLRU.shader = {};
        hlslShaderLRU.last_sk = ShaderKey();
        cacheHLSLShaders.clear();
        
        // Create default error shader for HLSL pipeline
        // TODO: Load and compile default HLSL error shader
        hlslShaderDefaultPurple = {};
    }

    // Pre-warm cache if any per-pixel mode is active
    if (Configuration.MGEFlags & USE_FFESHADER) {
        LOG::logline("-- Per-pixel shader precaching");
        precacheAsync();
    }

    return true;
}

void FixedFunctionShader::precacheAsync() {
    // Move precaching to a separate thread - essential variants to prevent stuttering
    std::thread precacheThread([]() {
        LOG::logline("-- Starting async per-pixel shader precaching (essential variants)");

        ShaderKey skCommon;
        memset(&skCommon, 0, sizeof skCommon);
        skCommon.uvSets = 1;

        int compiledVariants = 0;

        for (int vertexCol = 0; vertexCol <= 1; ++vertexCol) {
            skCommon.vertexColour = vertexCol;
            skCommon.vertexMaterial = vertexCol + 1;

            for (int heavyLighting = 0; heavyLighting <= 1; ++heavyLighting) {
                skCommon.heavyLighting = heavyLighting;

                for (int skinning = 0; skinning <= 1; ++skinning) {
                    skCommon.usesSkinning = skinning;

                    // Standard diffuse texturing (most common)
                    skCommon.activeStages = 1;
                    skCommon.fogMode = 1;
                    skCommon.usesTexgen = 0;
                    skCommon.stage[0] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 1, 0, 0, 0 };
                    memset(&skCommon.stage[1], 0, sizeof skCommon.stage[1]);
                    generateMWShader(skCommon);
                    compiledVariants++;

                    // Dual texture (common for details)
                    skCommon.activeStages = 2;
                    skCommon.fogMode = 1;
                    skCommon.usesTexgen = 0;
                    skCommon.stage[0] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 1, 0, 0, 0 };
                    skCommon.stage[1] = { D3DTOP_ADD, D3DTA_TEXTURE, D3DTA_CURRENT, D3DTA_CURRENT, 0, 0, 0, 0 };
                    generateMWShader(skCommon);
                    compiledVariants++;

                    // Particle effects (additive blend)
                    skCommon.activeStages = 1;
                    skCommon.fogMode = 2;
                    skCommon.usesTexgen = 0;
                    skCommon.stage[0] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 1, 0, 0, 0 };
                    memset(&skCommon.stage[1], 0, sizeof skCommon.stage[1]);
                    generateMWShader(skCommon);
                    compiledVariants++;

                    // Enchantment effects
                    skCommon.activeStages = 2;
                    skCommon.fogMode = 0;
                    skCommon.usesTexgen = 1;
                    skCommon.stage[0] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 0, 1, 0, 3 };
                    skCommon.stage[1] = { D3DTOP_MODULATE, D3DTA_TEXTURE, D3DTA_CURRENT, D3DTA_CURRENT, 1, 0, 0, 0 };
                    generateMWShader(skCommon);
                    compiledVariants++;
                }

                // Untextured surfaces
                skCommon.usesSkinning = 0;
                skCommon.fogMode = 1;
                skCommon.usesTexgen = 0;
                skCommon.activeStages = 1;
                skCommon.stage[0] = { D3DTOP_SELECTARG2, D3DTA_TEXTURE, D3DTA_DIFFUSE, D3DTA_CURRENT, 1, 0, 0, 0 };
                memset(&skCommon.stage[1], 0, sizeof skCommon.stage[1]);
                generateMWShader(skCommon);
                compiledVariants++;

                // Progress logging
                if (compiledVariants % 4 == 0) {
                    LOG::logline("-- Precaching progress: %d shaders", compiledVariants);
                }
            }
        }

        LOG::logline("-- Async precaching completed: %d essential shaders compiled", compiledVariants);
        });

    precacheThread.detach();
}

void FixedFunctionShader::updateLighting(float sunMult, float ambMult) {
    sunMultiplier = sunMult;
    ambMultiplier = ambMult;
}

void FixedFunctionShader::renderMorrowind(const RenderedState* rs, const FragmentState* frs, LightState* lightrs) {
    // Use HLSL pipeline if mode is set to HLSL (PerPixelLightFlags == 2)
    if (Configuration.PerPixelLightFlags == 2) {
        renderMorrowindHLSL(rs, frs, lightrs);
        return;
    }
    
    ID3DXEffect* effectFFE;

    // Check if state matches last used effect
    ShaderKey sk(rs, frs, lightrs);

    if (sk == shaderLRU.last_sk) {
        effectFFE = shaderLRU.effect;
    } else {
        // Read from shader cache / generate
        decltype(cacheEffects)::const_iterator iEffect = cacheEffects.find(sk);

        if (iEffect != cacheEffects.end()) {
            effectFFE = iEffect->second;
        } else {
            effectFFE = generateMWShader(sk);
        }

        shaderLRU.effect = effectFFE;
        shaderLRU.last_sk = sk;
    }

    // Set up material
    effectFFE->SetVector(ehMaterialDiffuse, (D3DXVECTOR4*)&frs->material.diffuse);
    effectFFE->SetVector(ehMaterialAmbient, (D3DXVECTOR4*)&frs->material.ambient);
    effectFFE->SetVector(ehMaterialEmissive, (D3DXVECTOR4*)&frs->material.emissive);

    // Set up lighting
    const size_t MaxLights = 8;
    D3DXVECTOR4 bufferDiffuse[MaxLights];
    float bufferAmbient[MaxLights];
    float bufferPosition[3 * MaxLights];
    float bufferFalloffQuadratic[MaxLights], bufferFalloffLinear[MaxLights], bufferFalloffConstant;

    memset(&bufferDiffuse, 0, sizeof(bufferDiffuse));
    memset(&bufferAmbient, 0, sizeof(bufferAmbient));
    memset(&bufferPosition, 0, sizeof(bufferPosition));
    memset(&bufferFalloffQuadratic, 0, sizeof(bufferFalloffQuadratic));
    memset(&bufferFalloffLinear, 0, sizeof(bufferFalloffLinear));
    bufferFalloffConstant = 0.33;

    // Check each active light
    RGBVECTOR sunDiffuse(0, 0, 0), ambient = lightrs->globalAmbient;
    size_t n = std::min(lightrs->active.size(), MaxLights), pointLightCount = 0;
    for (; n --> 0; ) {
        DWORD i = lightrs->active[n];
        const LightState::Light* light = &lightrs->lights.find(i)->second;

        // Transform to view space if not transformed this frame
        if (lightrs->lightsTransformed.find(i) == lightrs->lightsTransformed.end()) {
            if (light->type == D3DLIGHT_DIRECTIONAL) {
                D3DXVec3TransformNormal((D3DXVECTOR3*)&light->viewspacePos, (D3DXVECTOR3*)&light->position, &rs->viewTransform);
            } else {
                D3DXVec3TransformCoord((D3DXVECTOR3*)&light->viewspacePos, (D3DXVECTOR3*)&light->position, &rs->viewTransform);
            }

            lightrs->lightsTransformed[i] = true;
        }

        if (light->type == D3DLIGHT_POINT) {
            memcpy(&bufferDiffuse[pointLightCount], &light->diffuse, sizeof(light->diffuse));

            // Scatter position vectors for vectorization
            bufferPosition[pointLightCount] = light->viewspacePos.x;
            bufferPosition[pointLightCount + MaxLights] = light->viewspacePos.y;
            bufferPosition[pointLightCount + 2*MaxLights] = light->viewspacePos.z;

            // Scatter attenuation factors for vectorization
            if (light->falloff.x > 0) {
                // Standard point light source (falloffConstant doesn't vary per light)
                bufferFalloffConstant = light->falloff.x;
                bufferFalloffLinear[pointLightCount] = light->falloff.y;
                bufferFalloffQuadratic[pointLightCount] = light->falloff.z;
            } else if (light->falloff.z > 0) {
                // Probably a magic light source patched by Morrowind Code Patch
                // Patched falloff calculation is quadratic only, which needs to be
                // modified to account for the standard falloffConstant
                // Diffuse colour is correctly specified with the patch
                // Some overbrightness is applied to diffuse to cause glowing
                bufferDiffuse[pointLightCount].x *= bufferFalloffConstant;
                bufferDiffuse[pointLightCount].y *= bufferFalloffConstant;
                bufferDiffuse[pointLightCount].z *= bufferFalloffConstant;
                bufferAmbient[pointLightCount] = 1.0f + 1e-4f / sqrt(light->falloff.z);
                bufferFalloffQuadratic[pointLightCount] = bufferFalloffConstant * light->falloff.z;
            } else if (light->falloff.y == 0.10000001f) {
                // Projectile light source, normally hard coded by Morrowind to { 0, 3 * (1/30), 0 }
                // This falloff value cannot be produced by other magic effects
                // Replacement falloff is significantly brighter to look cool
                // Avoids modifying colour or position
                bufferFalloffQuadratic[pointLightCount] = 5e-5;
            } else if (light->falloff.y > 0) {
                // Light magic effect, falloffs calculated by { 0, 3 / (22 * spell magnitude), 0 }
                // A mix of ambient (falloff but no N.L component) and over-bright diffuse lighting
                // It is approximated with a half-lambert weight + quadratic falloff
                // Light colour is altered to avoid variable brightness from Morrowind bugs
                // The point source is moved up slightly as it is often embedded in the ground
                float brightness = 0.25f + 1e-4f / light->falloff.y;
                bufferDiffuse[pointLightCount].x = brightness;
                bufferDiffuse[pointLightCount].y = brightness;
                bufferDiffuse[pointLightCount].z = brightness;
                bufferAmbient[pointLightCount] = 1.0;
                bufferFalloffQuadratic[pointLightCount] = 0.5555f * light->falloff.y * light->falloff.y;
                bufferPosition[pointLightCount + 2*MaxLights] += 25.0;
            }
            
            // Debug: Log first light falloff values for Effect shader path (every 60 frames)
            static int effectDebugFrameCount = 0;
            effectDebugFrameCount++;
            if (pointLightCount == 0 && effectDebugFrameCount % 60 == 0) {
                LOG::logline("=== EFFECT SHADER Light[0] Falloff Values ===");
                LOG::logline("light->falloff: (%f, %f, %f)", light->falloff.x, light->falloff.y, light->falloff.z);
                LOG::logline("bufferFalloffQuadratic[0]: %f", bufferFalloffQuadratic[0]);
                LOG::logline("bufferFalloffConstant: %f", bufferFalloffConstant);
                
                // Calculate distance for reference
                float dist = sqrt(light->viewspacePos.x * light->viewspacePos.x + 
                                 light->viewspacePos.y * light->viewspacePos.y + 
                                 light->viewspacePos.z * light->viewspacePos.z);
                LOG::logline("Light[0] distance: %f", dist);
                
                // Calculate Effect shader attenuation
                float effectFalloff = bufferFalloffQuadratic[0] * dist * dist + bufferFalloffConstant;
                float effectAttenuation = (effectFalloff > 0) ? (1.0f / effectFalloff) : 0.0f;
                LOG::logline("Effect falloff: %f, Effect atten: %f", effectFalloff, effectAttenuation);
            }
            
            ++pointLightCount;
        } else if (light->type == D3DLIGHT_DIRECTIONAL) {
            effectFFE->SetFloatArray(ehLightSunDirection, (const float*)&light->viewspacePos, 3);

            sunDiffuse = light->diffuse;
            ambient.r += light->ambient.x;
            ambient.g += light->ambient.y;
            ambient.b += light->ambient.z;
        }
    }

    // Apply light multipliers, for HDR light levels
    sunDiffuse *= sunMultiplier;
    ambient *= ambMultiplier;

    // Special case, check if ambient state is pure white (distant land does not record this for a reason)
    // Morrowind temporarily sets this for full-bright particle effects, but just adding it
    // to other ambient sources above would cause over-brightness
    DWORD checkAmbient;
    device->GetRenderState(D3DRS_AMBIENT, &checkAmbient);
    if (checkAmbient == 0xffffffff) {
        // Set lighting to result in full-bright equivalent after tonemapping
        ambient.r = ambient.g = ambient.b = 1.25;
        sunDiffuse.r = sunDiffuse.g = sunDiffuse.b = 0.0;
    }

    effectFFE->SetFloatArray(ehLightSceneAmbient, ambient, 3);
    effectFFE->SetFloatArray(ehLightSunDiffuse, sunDiffuse, 3);
    effectFFE->SetVectorArray(ehLightDiffuse, bufferDiffuse, MaxLights);
    
    // Debug: Log Effect shader light colors every 120 frames
    static int effectLightDebugCounter = 0;
    effectLightDebugCounter++;
    if (effectLightDebugCounter % 120 == 0) {
        LOG::logline("=== EFFECT Light Colors ===");
        for (int i = 0; i < pointLightCount; i++) {
            LOG::logline("Effect Light[%d]: diffuse=(%.3f,%.3f,%.3f)", i,
                         bufferDiffuse[i].x, bufferDiffuse[i].y, bufferDiffuse[i].z);
        }
    }
    effectFFE->SetFloatArray(ehLightAmbient, bufferAmbient, MaxLights);
    effectFFE->SetFloatArray(ehLightPosition, bufferPosition, 3 * MaxLights);
    effectFFE->SetFloatArray(ehLightFalloffQuadratic, bufferFalloffQuadratic, MaxLights);
    effectFFE->SetFloatArray(ehLightFalloffLinear, bufferFalloffLinear, MaxLights);
    effectFFE->SetFloat(ehLightFalloffConstant, bufferFalloffConstant);

    // Bump mapping state
    if (sk.usesBumpmap) {
        const FragmentState::Stage& bumpStage = frs->stage[sk.bumpmapStage];
        effectFFE->SetFloatArray(ehBumpMatrix, &bumpStage.bumpEnvMat[0][0], 4);
        effectFFE->SetFloatArray(ehBumpLumiScaleBias, &bumpStage.bumpLumiScale, 2);
    }

    // Texgen texture matrix
    if (sk.usesTexgen) {
        D3DXMATRIX m;
        device->GetTransform((D3DTRANSFORMSTATETYPE)(D3DTS_TEXTURE0 + sk.texgenStage), &m);
        effectFFE->SetMatrix(ehTexgenTransform, &m);
    }

    // Copy texture bindings from fixed function pipe
    const D3DXHANDLE ehIndex[] = { ehTex0, ehTex1, ehTex2, ehTex3, ehTex4, ehTex5 };
    for (n = 0; n != std::min((int)sk.activeStages, 6); ++n) {
        IDirect3DBaseTexture9* tex;
        device->GetTexture(n, &tex);
        effectFFE->SetTexture(ehIndex[n], tex);
        if (tex) {
            tex->Release();
        }
    }

    // Set common state and render
    effectFFE->SetInt(ehVertexBlendState, rs->vertexBlendState);
    if (rs->vertexBlendState) {
        effectFFE->SetMatrixArray(ehVertexBlendPalette, rs->worldViewTransforms, 4);
    } else {
        effectFFE->SetMatrix(ehWorld, &rs->worldTransforms[0]);
        effectFFE->SetMatrix(ehWorldView, &rs->worldViewTransforms[0]);
    }

    UINT passes;
    effectFFE->Begin(&passes, D3DXFX_DONOTSAVESTATE);
    effectFFE->BeginPass(0);
    device->DrawIndexedPrimitive(rs->primType, rs->baseIndex, rs->minIndex, rs->vertCount, rs->startIndex, rs->primCount);
    effectFFE->EndPass();
    effectFFE->End();

    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);
}

ID3DXEffect* FixedFunctionShader::generateMWShader(const ShaderKey& sk) {
    string genVBCoupling, genPSCoupling, genTransform, genTexcoords, genVertexColour, genLightCount, genMaterial, genTexturing, genFog;
    stringstream buf;

    // Identify output texcoords and check for texgen; supports max. one per shader
    int texGen = 0, texGenSrcIndex = 0, texGenOutputIndex = sk.uvSets, totalOutputCoords = sk.uvSets;
    if (sk.usesTexgen) {
        texGen = sk.stage[sk.texgenStage].texcoordGen;
        texGenSrcIndex = sk.stage[sk.texgenStage].texcoordIndex;

        ++totalOutputCoords;
        if (sk.projectiveTexgen) {
            ++totalOutputCoords;
        }
    }

    if (totalOutputCoords > 4) {
        LOG::logline("!! Shader generator error: excessive texcoord usage (%d).", totalOutputCoords);
        sk.log();
        LOG::flush();

        effectDefaultPurple->AddRef();
        cacheEffects[sk] = effectDefaultPurple;
        return effectDefaultPurple;
    }

    // Pack 2d texcoords into interpolators and map to stages
    const char* strInterpolators[] = { "01", "23" };
    const char* strTexcoordPacking[] = { ".xy", ".zw" };
    string texcoordNames[8], texSamplers[8];

    for (int i = 0; i != sk.activeStages; ++i) {
        bool isTexGen = bool(sk.stage[i].texcoordGen);
        int x = isTexGen ? texGenOutputIndex : sk.stage[i].texcoordIndex;

        buf.str(string());
        buf << "IN.texcoord" << strInterpolators[x >> 1] << strTexcoordPacking[x & 1];
        if (isTexGen && sk.projectiveTexgen) {
            buf << " / IN.texcoord" << strInterpolators[(x+1) >> 1] << strTexcoordPacking[(x+1) & 1];
        }
        texcoordNames[i] = buf.str();
        buf.str(string());
        buf << "tex2D(sampFFE" << i << ", " << texcoordNames[i] << ")";
        texSamplers[i] = buf.str();
    }

    // Vertex format coupling, generate equivalent struct to input FVF
    buf.str(string());

    if (sk.usesSkinning) {
        buf << "float4 blendweights : BLENDWEIGHT; ";
    }
    if (sk.vertexColour) {
        buf << "float4 col : COLOR; ";
    }
    for (int i = 0; i != sk.uvSets; ++i) {
        buf << "float2 texcoord" << i << " : TEXCOORD" << i << "; ";
    }

    genVBCoupling = buf.str();

    // Pixel shader coupling, passes texcoords and colours
    buf.str(string());

    if (sk.vertexColour) {
        buf << "centroid float4 col : COLOR; ";
    }
    if (totalOutputCoords == 1) {
        buf << "float2 texcoord01 : TEXCOORD0; ";
    } else if (totalOutputCoords > 1) {
        buf << "float4 texcoord01 : TEXCOORD0; ";
    }
    if (totalOutputCoords == 3) {
        buf << "float2 texcoord23 : TEXCOORD1; ";
    } else if (totalOutputCoords == 4) {
        buf << "float4 texcoord23 : TEXCOORD1; ";
    }

    genPSCoupling = buf.str();

    // Transform / skinning
    buf.str(string());

    if (sk.usesSkinning) {
        buf << "viewpos = skinnedVertex(IN.pos, IN.blendweights); normal = skinnedNormal(IN.nrm, IN.blendweights);";
    } else {
        buf << "viewpos = rigidVertex(IN.pos); normal = rigidNormal(IN.nrm);";
    }

    genTransform = buf.str();

    // Texcoord routing and texgen
    string texRouting[4];
    for (int i = 0; i != sk.uvSets; ++i) {
        buf.str(string());
        buf << "IN.texcoord" << i;
        texRouting[i] = buf.str();
    }

    buf.str(string());

    if (texGen) {
        buf << "float3 texgen = ";
        switch (texGen) {
        case D3DTSS_TCI_CAMERASPACENORMAL >> 16:
            buf << "texgenNormal(normal); ";
            break;
        case D3DTSS_TCI_CAMERASPACEPOSITION >> 16:
            buf << "texgenPosition(viewpos); ";
            break;
        case D3DTSS_TCI_CAMERASPACEREFLECTIONVECTOR >> 16:
            buf << "texgenReflection(viewpos, normal); ";
            break;
        case D3DTSS_TCI_SPHEREMAP >> 16:
            buf << "texgenSphere(" << texRouting[texGenSrcIndex] << "); ";
            break;
        }
        buf << "texgen = mul(float4(texgen, 1), texgenTransform).xyz; ";
        texRouting[texGenOutputIndex] = "texgen.xy";
        if (sk.projectiveTexgen) {
            texRouting[texGenOutputIndex + 1] = "texgen.zz";
        }
    }

    if (totalOutputCoords == 1) {
        buf << "OUT.texcoord01 = " << texRouting[0] << ";";
    } else if (totalOutputCoords > 1) {
        buf << "OUT.texcoord01 = float4(" << texRouting[0] << ", " << texRouting[1] << "); ";
    }
    if (totalOutputCoords == 3) {
        buf << "OUT.texcoord23 = " << texRouting[2] << ";";
    } else if (totalOutputCoords == 4) {
        buf << "OUT.texcoord23 = float4(" << texRouting[2] << ", " << texRouting[3] << ");";
    }

    genTexcoords = buf.str();

    // Vertex colour routing
    buf.str(string());
    if (sk.vertexColour) {
        buf << "OUT.col = IN.col;";
    }
    genVertexColour = buf.str();

    // Lighting
    if (sk.vertexMaterial == 0) {
        genLightCount = "0";
    } else {
        genLightCount = sk.heavyLighting ? "8" : "4";
    }

    // Vertex material
    buf.str(string());
    switch (sk.vertexMaterial) {
    case 0:
        buf << "diffuse = " << (sk.vertexColour ? "IN.col;" : "1.0;");
        break;
    case 1:
        buf << "diffuse = vertexMaterialNone(d, a);";
        break;
    case 2:
        buf << "diffuse = vertexMaterialDiffAmb(d, a, IN.col);";
        break;
    case 3:
        buf << "diffuse = vertexMaterialEmissive(d, a, IN.col);";
        break;
    }
    genMaterial = buf.str();

    // Texture and shading operations
    buf.str(string());
    string arg1, arg2, arg3;

    for (int i = 0; i != sk.activeStages; ++i) {
        const ShaderKey::Stage& s = sk.stage[i];
        const string dest = s.alphaOpMatched ? "c = " : "c.rgb = ";
        const string mask = s.alphaOpMatched ? "" : ".rgb";

        arg1 = buildArgString(s.colorArg1, mask, texSamplers[i]);
        arg2 = buildArgString(s.colorArg2, mask, texSamplers[i]);

        switch (s.colorOp) {
        case D3DTOP_SELECTARG1:
            buf << dest << arg1 << ";";
            break;

        case D3DTOP_SELECTARG2:
            buf << dest << arg2 << ";";
            break;

        case D3DTOP_MODULATE:
            buf << dest << arg1 << " * " << arg2 << ";";
            break;

        case D3DTOP_MODULATE2X:
            buf << dest << "2 * " << arg1 << " * " << arg2 << ";";
            break;

        case D3DTOP_MODULATE4X:
            buf << dest << "4 * " << arg1 << " * " << arg2 << ";";
            break;

        case D3DTOP_ADD:
            buf << dest << arg1 << " + " << arg2 << ";";
            break;

        case D3DTOP_ADDSIGNED:
            buf << dest << arg1 << " + " << arg2 << " - 0.5;";
            break;

        case D3DTOP_ADDSIGNED2X:
            buf << dest << "2 * (" << arg1 << "+" << arg2 << ") - 1;";
            break;

        case D3DTOP_SUBTRACT:
            buf << dest << arg1 << " - " << arg2 << ";";
            break;

        case D3DTOP_BLENDDIFFUSEALPHA:
            buf << dest << "lerp(" << arg1 << ", " << arg2 << ", diffuse.a);";
            break;

        case D3DTOP_BLENDTEXTUREALPHA:
            arg3 = buildArgString(D3DTA_TEXTURE, "", texSamplers[i]);
            buf << "float4 temp" << i << " = " << arg3 << "; lerp(" << arg1 << ", " << arg1 << ", temp" << i <<".a);";
            break;

        case D3DTOP_BUMPENVMAP:
            arg3 = buildArgString(D3DTA_TEXTURE, "", texSamplers[i]);
            buf << "float4 bump = bumpmapStage(sampFFE" << i+1 << ", " << texcoordNames[i+1] << ", " << arg3 << ");";
            texSamplers[i+1] = "bump";
            break;

        case D3DTOP_BUMPENVMAPLUMINANCE:
            arg3 = buildArgString(D3DTA_TEXTURE, "", texSamplers[i]);
            buf << "float4 bump = bumpmapLumiStage(sampFFE" << i+1 << ", " << texcoordNames[i+1] << ", " << arg3 << ");";
            texSamplers[i+1] = "bump";
            break;

        case D3DTOP_DOTPRODUCT3:
            arg1 = buildArgString(s.colorArg1, ".rgb", texSamplers[i]);
            arg2 = buildArgString(s.colorArg2, ".rgb", texSamplers[i]);
            buf << "c.rgb = dot(" << arg1 << ", " << arg2 << ");";
            break;

        case D3DTOP_MULTIPLYADD:
            arg1 = buildArgString(s.colorArg1, ".rgb", texSamplers[i]);
            arg2 = buildArgString(s.colorArg2, ".rgb", texSamplers[i]);
            arg3 = buildArgString(s.colorArg0, ".rgb", texSamplers[i]);
            buf << "c.rgb = " << arg1 << " * " << arg2 << " + " << arg3 << ";";
            break;

        default:
            buf << "unsupported";
            break;
        }

        if (s.alphaOpSelect1) {
            // Alpha Select1 op, assumes alpha args are the same as color args
            switch (s.colorArg1) {
            case D3DTA_DIFFUSE:
                buf << "c.a = diffuse.a";
                break;

            case D3DTA_TEXTURE:
                // The HLSL compiler is able to optimize this repeated sampler use and does not generate an extra texld.
                buf << "c.a = " << texSamplers[i] << ".a;";
                break;
            }
        }

        buf << " \\\n";
    }

    genTexturing = buf.str();

    // Final fog application
    buf.str(string());

    switch (sk.fogMode) {
    case 0:     // Fog disabled
        break;
    case 1:     // Standard fog mode
        buf << "c.rgb = lerp(fogColNear, c.rgb, fog); ";
        break;
    case 2:     // Additive objects should fog towards black, which preserves the destination correctly
        buf << "c.rgb *= fog; ";
        break;
    }

    genFog = buf.str();

    // Compile HLSL through insertions into a template file
    const D3DXMACRO generatedCode[] = {
        "FFE_VB_COUPLING", genVBCoupling.c_str(),
        "FFE_SHADER_COUPLING", genPSCoupling.c_str(),
        "FFE_TRANSFORM_SKIN", genTransform.c_str(),
        "FFE_TEXCOORDS_TEXGEN", genTexcoords.c_str(),
        "FFE_VERTEX_COLOUR", genVertexColour.c_str(),
        "FFE_LIGHTS_ACTIVE", genLightCount.c_str(),
        "FFE_VERTEX_MATERIAL", genMaterial.c_str(),
        "FFE_TEXTURING", genTexturing.c_str(),
        "FFE_FOG_APPLICATION", genFog.c_str(),
        0, 0
    };

    // Create effect while pooling constants with everything else
    ID3DXEffect* effectFFE;
    ID3DXBuffer* errors;

    //LOG::logline("-- Generating replacement fixed function shader");
    //sk.log();

    HRESULT hr = D3DXCreateEffectFromFile(device, "Data Files\\shaders\\core\\XE FixedFuncEmu.fx", generatedCode, 0, D3DXSHADER_OPTIMIZATION_LEVEL3|D3DXFX_LARGEADDRESSAWARE, constantPool, &effectFFE, &errors);

    if (hr != D3D_OK) {
        LOG::logline("!! Generating FFE shader: compile error %xh", hr);
        if (errors) {
            LOG::write("!! Shader compile errors:\n");
            LOG::write(reinterpret_cast<const char*>(errors->GetBufferPointer()));
            LOG::write("\n");
            errors->Release();
        }
        LOG::write("\n");
        effectDefaultPurple->AddRef();
        effectFFE = effectDefaultPurple;
    }

    cacheEffects[sk] = effectFFE;
    return effectFFE;
}

string buildArgString(DWORD arg, const string& mask, const string& sampler) {
    stringstream s;

    switch (arg) {
    case D3DTA_DIFFUSE:
        s << "diffuse" << mask;
        break;
    case D3DTA_CURRENT:
        s << "c" << mask;
        break;
    case D3DTA_TEXTURE:
        s << sampler << mask;
        break;
    default:
        s << "unsupported";
        break;
    }

    return s.str();
}

// HLSL Pipeline Implementation
void FixedFunctionShader::renderMorrowindHLSL(const RenderedState* rs, const FragmentState* frs, LightState* lightrs) {
    HLSLShader hlslShader;

    // Check if state matches last used shader
    ShaderKey sk(rs, frs, lightrs);

    if (sk == hlslShaderLRU.last_sk) {
        hlslShader = hlslShaderLRU.shader;
    } else {
        // Read from shader cache / generate
        decltype(cacheHLSLShaders)::const_iterator iShader = cacheHLSLShaders.find(sk);

        if (iShader != cacheHLSLShaders.end()) {
            hlslShader = iShader->second;
        } else {
            hlslShader = generateMWShaderHLSL(sk);
        }

        hlslShaderLRU.shader = hlslShader;
        hlslShaderLRU.last_sk = sk;
    }

    // Save current render states before modifying them
    DWORD savedLighting, savedFogEnable, savedAlphaBlendEnable, savedAlphaTestEnable;
    DWORD savedZEnable, savedZWriteEnable;
    device->GetRenderState(D3DRS_LIGHTING, &savedLighting);
    device->GetRenderState(D3DRS_FOGENABLE, &savedFogEnable);
    device->GetRenderState(D3DRS_ALPHABLENDENABLE, &savedAlphaBlendEnable);
    device->GetRenderState(D3DRS_ALPHATESTENABLE, &savedAlphaTestEnable);
    device->GetRenderState(D3DRS_ZENABLE, &savedZEnable);
    device->GetRenderState(D3DRS_ZWRITEENABLE, &savedZWriteEnable);
    
    // Set shaders
    device->SetVertexShader(hlslShader.vertexShader);
    device->SetPixelShader(hlslShader.pixelShader);
    
    // Set up render states to match what D3DXEffect was doing
    // For alpha blended objects, enable depth testing but disable depth writing
    if (rs->blendEnable) {
        device->SetRenderState(D3DRS_ZENABLE, TRUE);
        device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
    } else {
        device->SetRenderState(D3DRS_ZENABLE, rs->zWrite ? TRUE : FALSE);
        device->SetRenderState(D3DRS_ZWRITEENABLE, rs->zWrite ? TRUE : FALSE);
    }
    device->SetRenderState(D3DRS_ZFUNC, D3DCMP_LESSEQUAL);
    device->SetRenderState(D3DRS_CULLMODE, rs->cullMode);
    device->SetRenderState(D3DRS_LIGHTING, FALSE);
    device->SetRenderState(D3DRS_FOGENABLE, FALSE);
    
    // Alpha blending states
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, rs->blendEnable);
    if (rs->blendEnable) {
        device->SetRenderState(D3DRS_SRCBLEND, rs->srcBlend);
        device->SetRenderState(D3DRS_DESTBLEND, rs->destBlend);
    }
    
    // Alpha testing states  
    device->SetRenderState(D3DRS_ALPHATESTENABLE, rs->alphaTest);
    if (rs->alphaTest) {
        device->SetRenderState(D3DRS_ALPHAFUNC, rs->alphaFunc);
        device->SetRenderState(D3DRS_ALPHAREF, rs->alphaRef);
    }
    
    // Set vertex declaration for the FVF
    device->SetFVF(rs->fvf);
    
    // Set vertex and index buffers like the original system
    device->SetStreamSource(0, rs->vb, rs->vbOffset, rs->vbStride);
    if (rs->ib) {
        device->SetIndices(rs->ib);
    }

    // Set up matrices using constant tables (like the Combined shader expects)
    D3DXMATRIX projMatrix, viewMatrix, worldMatrix;
    device->GetTransform(D3DTS_PROJECTION, &projMatrix);
    device->GetTransform(D3DTS_VIEW, &viewMatrix);
    device->GetTransform(D3DTS_WORLD, &worldMatrix);
    
    D3DXMATRIX worldViewProj = worldMatrix * viewMatrix * projMatrix;
    D3DXMATRIX worldView = worldMatrix * viewMatrix;
    
    // Debug: Log matrix values once per frame to avoid spam
    static int debugFrameCount = 0;
    debugFrameCount++;
    if (debugFrameCount % 60 == 0) {  // Log every 60 frames
        LOG::logline("=== HLSL Matrix Debug Frame %d ===", debugFrameCount);
        LOG::logline("viewMatrix: [%f,%f,%f,%f] [%f,%f,%f,%f] [%f,%f,%f,%f] [%f,%f,%f,%f]",
            viewMatrix._11, viewMatrix._12, viewMatrix._13, viewMatrix._14,
            viewMatrix._21, viewMatrix._22, viewMatrix._23, viewMatrix._24,
            viewMatrix._31, viewMatrix._32, viewMatrix._33, viewMatrix._34,
            viewMatrix._41, viewMatrix._42, viewMatrix._43, viewMatrix._44);
        LOG::logline("worldView: [%f,%f,%f,%f] [%f,%f,%f,%f] [%f,%f,%f,%f] [%f,%f,%f,%f]",
            worldView._11, worldView._12, worldView._13, worldView._14,
            worldView._21, worldView._22, worldView._23, worldView._24,
            worldView._31, worldView._32, worldView._33, worldView._34,
            worldView._41, worldView._42, worldView._43, worldView._44);
    }
    
    // Use constant tables to set matrices
    if (hlslShader.vsConstantTable) {
        D3DXHANDLE hWorldViewProj = hlslShader.vsConstantTable->GetConstantByName(NULL, "worldViewProj");
        if (hWorldViewProj) {
            hlslShader.vsConstantTable->SetMatrix(device, hWorldViewProj, &worldViewProj);
        }
        
        D3DXHANDLE hView = hlslShader.vsConstantTable->GetConstantByName(NULL, "view");
        if (hView) {
            hlslShader.vsConstantTable->SetMatrix(device, hView, &viewMatrix);
        }
        
        D3DXHANDLE hProj = hlslShader.vsConstantTable->GetConstantByName(NULL, "proj");
        if (hProj) {
            hlslShader.vsConstantTable->SetMatrix(device, hProj, &projMatrix);
        }
        
        D3DXHANDLE hWorld = hlslShader.vsConstantTable->GetConstantByName(NULL, "world");
        if (hWorld) {
            hlslShader.vsConstantTable->SetMatrix(device, hWorld, &worldMatrix);
        }
        
        D3DXHANDLE hWorldView = hlslShader.vsConstantTable->GetConstantByName(NULL, "worldview");
        if (hWorldView) {
            hlslShader.vsConstantTable->SetMatrix(device, hWorldView, &worldView);
        }
        
        // Set up vertex blend palette for skinning using Morrowind's actual data
        D3DXHANDLE hVertexBlendPalette = hlslShader.vsConstantTable->GetConstantByName(NULL, "vertexBlendPalette");
        if (hVertexBlendPalette) {
            if (rs->vertexBlendState > 0) {
                // For skinned objects, use the bone matrices from Morrowind
                hlslShader.vsConstantTable->SetMatrixArray(device, hVertexBlendPalette, rs->worldViewTransforms, 4);
            } else {
                // For rigid objects, set first matrix to worldview and clear others
                D3DXMATRIX blendMatrices[4];
                blendMatrices[0] = worldView;
                memset(&blendMatrices[1], 0, sizeof(D3DXMATRIX) * 3);
                hlslShader.vsConstantTable->SetMatrixArray(device, hVertexBlendPalette, blendMatrices, 4);
            }
        }
        
        D3DXHANDLE hVertexBlendState = hlslShader.vsConstantTable->GetConstantByName(NULL, "vertexBlendState");
        if (hVertexBlendState) {
            D3DXVECTOR4 blendState((float)rs->vertexBlendState, 0, 0, 0);
            hlslShader.vsConstantTable->SetVector(device, hVertexBlendState, &blendState);
        }
    }
    
    // Set pixel shader constants using constant tables (like Combined shader expects)
    if (hlslShader.psConstantTable) {
        D3DXHANDLE hMaterialDiffuse = hlslShader.psConstantTable->GetConstantByName(NULL, "materialDiffuse");
        if (hMaterialDiffuse) {
            hlslShader.psConstantTable->SetVector(device, hMaterialDiffuse, (D3DXVECTOR4*)&frs->material.diffuse);
        }
        
        D3DXHANDLE hMaterialAmbient = hlslShader.psConstantTable->GetConstantByName(NULL, "materialAmbient");
        if (hMaterialAmbient) {
            hlslShader.psConstantTable->SetVector(device, hMaterialAmbient, (D3DXVECTOR4*)&frs->material.ambient);
        }
        
        D3DXHANDLE hMaterialEmissive = hlslShader.psConstantTable->GetConstantByName(NULL, "materialEmissive");
        if (hMaterialEmissive) {
            hlslShader.psConstantTable->SetVector(device, hMaterialEmissive, (D3DXVECTOR4*)&frs->material.emissive);
        }
        
        // Set up lighting using the same logic as the original renderMorrowind
        const size_t MaxLights = 8;
        D3DXVECTOR4 bufferDiffuse[MaxLights];
        float bufferAmbient[MaxLights];
        D3DXVECTOR3 bufferPosition[MaxLights];  // Proper float3 positions
        float bufferFalloffQuadratic[MaxLights], bufferFalloffLinear[MaxLights], bufferFalloffConstant;

        memset(&bufferDiffuse, 0, sizeof(bufferDiffuse));
        memset(&bufferAmbient, 0, sizeof(bufferAmbient));
        memset(&bufferPosition, 0, sizeof(bufferPosition));
        memset(&bufferFalloffQuadratic, 0, sizeof(bufferFalloffQuadratic));
        memset(&bufferFalloffLinear, 0, sizeof(bufferFalloffLinear));
        bufferFalloffConstant = 0.33;

        // Check each active light
        RGBVECTOR sunDiffuse(0, 0, 0), ambient = lightrs->globalAmbient;
        D3DVECTOR sunDirection = {0, 0, 1};
        size_t n = std::min(lightrs->active.size(), MaxLights), pointLightCount = 0;
        for (; n --> 0; ) {
            DWORD i = lightrs->active[n];
            const LightState::Light* light = &lightrs->lights.find(i)->second;

            // Transform to view space if not transformed this frame
            if (lightrs->lightsTransformed.find(i) == lightrs->lightsTransformed.end()) {
                if (light->type == D3DLIGHT_DIRECTIONAL) {
                    D3DXVec3TransformNormal((D3DXVECTOR3*)&light->viewspacePos, (D3DXVECTOR3*)&light->position, &rs->viewTransform);
                } else {
                    D3DXVec3TransformCoord((D3DXVECTOR3*)&light->viewspacePos, (D3DXVECTOR3*)&light->position, &rs->viewTransform);
                }

                lightrs->lightsTransformed[i] = true;
            }

            if (light->type == D3DLIGHT_POINT) {
                memcpy(&bufferDiffuse[pointLightCount], &light->diffuse, sizeof(light->diffuse));

                // Set position as proper float3
                bufferPosition[pointLightCount] = D3DXVECTOR3(light->viewspacePos.x, light->viewspacePos.y, light->viewspacePos.z);
                
                // Debug: Log first light position data and falloff values
                if (pointLightCount == 0 && debugFrameCount % 60 == 0) {
                    LOG::logline("Light[0] original pos: (%f, %f, %f)", light->position.x, light->position.y, light->position.z);
                    LOG::logline("Light[0] viewspace pos: (%f, %f, %f)", light->viewspacePos.x, light->viewspacePos.y, light->viewspacePos.z);
                    LOG::logline("Light[0] buffer pos: (%f, %f, %f)", bufferPosition[0].x, bufferPosition[0].y, bufferPosition[0].z);
                    
                    // Calculate distance for reference
                    float dist = sqrt(light->viewspacePos.x * light->viewspacePos.x + 
                                     light->viewspacePos.y * light->viewspacePos.y + 
                                     light->viewspacePos.z * light->viewspacePos.z);
                    LOG::logline("Light[0] distance: %f", dist);
                    
                    // Log CPU-side falloff values being set
                    LOG::logline("=== CPU Falloff Values ===");
                    LOG::logline("bufferFalloffQuadratic[0]: %f", bufferFalloffQuadratic[0]);
                    LOG::logline("bufferFalloffConstant: %f", bufferFalloffConstant);
                    LOG::logline("light->falloff: (%f, %f, %f)", light->falloff.x, light->falloff.y, light->falloff.z);
                    
                    // Calculate Effect shader result
                    float effectFalloff = bufferFalloffQuadratic[0] * dist * dist + bufferFalloffConstant;
                    float effectAttenuation = (effectFalloff > 0) ? (1.0f / effectFalloff) : 0.0f;
                    float hlslScaled = 15000.0f / (dist * dist);
                    LOG::logline("Effect falloff: %f, Effect atten: %f", effectFalloff, effectAttenuation);
                    LOG::logline("HLSL 15000 scaled: %f, Ratio: %f", hlslScaled, hlslScaled / (effectAttenuation > 0.0001f ? effectAttenuation : 0.0001f));
                }

                // Scatter attenuation factors for vectorization
                if (light->falloff.x > 0) {
                    // Standard point light source (falloffConstant doesn't vary per light)
                    bufferFalloffConstant = light->falloff.x;
                    bufferFalloffLinear[pointLightCount] = light->falloff.y;
                    bufferFalloffQuadratic[pointLightCount] = light->falloff.z;
                } else if (light->falloff.z > 0) {
                    // Probably a magic light source patched by Morrowind Code Patch
                    bufferDiffuse[pointLightCount].x *= bufferFalloffConstant;
                    bufferDiffuse[pointLightCount].y *= bufferFalloffConstant;
                    bufferDiffuse[pointLightCount].z *= bufferFalloffConstant;
                    bufferAmbient[pointLightCount] = 1.0f + 1e-4f / sqrt(light->falloff.z);
                    bufferFalloffQuadratic[pointLightCount] = bufferFalloffConstant * light->falloff.z;
                } else if (light->falloff.y == 0.10000001f) {
                    // Projectile light source, normally hard coded by Morrowind to { 0, 3 * (1/30), 0 }
                    // This falloff value cannot be produced by other magic effects
                    // Replacement falloff is significantly brighter to look cool
                    // Avoids modifying colour or position
                    bufferFalloffQuadratic[pointLightCount] = 5e-5;
                } else if (light->falloff.y > 0) {
                    // Light magic effect, falloffs calculated by { 0, 3 / (22 * spell magnitude), 0 }
                    // A mix of ambient (falloff but no N.L component) and over-bright diffuse lighting
                    // It is approximated with a half-lambert weight + quadratic falloff
                    // Preserve original light color instead of overwriting with white brightness
                    // The point source is moved up slightly as it is often embedded in the ground
                    bufferAmbient[pointLightCount] = 1.0;
                    bufferFalloffQuadratic[pointLightCount] = 0.5555f * light->falloff.y * light->falloff.y;
                    bufferPosition[pointLightCount].z += 25.0;
                }

                ++pointLightCount;
            } else if (light->type == D3DLIGHT_DIRECTIONAL) {
                sunDiffuse = light->diffuse;
                sunDirection = light->viewspacePos;  // Already transformed to view space
                // Add directional light ambient to global ambient like the original
                ambient.r += light->ambient.x;
                ambient.g += light->ambient.y;
                ambient.b += light->ambient.z;
            }
        }
        
        // Apply light multipliers, for HDR light levels
        sunDiffuse *= sunMultiplier;
        ambient *= ambMultiplier;
        
        // Special case, check if ambient state is pure white (distant land does not record this for a reason)
        // Morrowind temporarily sets this for full-bright particle effects
        DWORD checkAmbient;
        device->GetRenderState(D3DRS_AMBIENT, &checkAmbient);
        if (checkAmbient == 0xffffffff) {
            // Set lighting to result in full-bright equivalent after tonemapping
            ambient.r = ambient.g = ambient.b = 1.25;
            sunDiffuse.r = sunDiffuse.g = sunDiffuse.b = 0.0;
        }
        
        // Set lighting constants using the same format as the original system
        D3DXHANDLE hLightSunDirection = hlslShader.psConstantTable->GetConstantByName(NULL, "lightSunDirection");
        if (hLightSunDirection) {
            hlslShader.psConstantTable->SetFloatArray(device, hLightSunDirection, (const float*)&sunDirection, 3);
        } else {
            LOG::logline("!! lightSunDirection constant not found in pixel shader");
        }
        
        D3DXHANDLE hLightSunDiffuse = hlslShader.psConstantTable->GetConstantByName(NULL, "lightSunDiffuse");
        if (hLightSunDiffuse) {
            hlslShader.psConstantTable->SetFloatArray(device, hLightSunDiffuse, (const float*)&sunDiffuse, 3);
        } else {
            LOG::logline("!! lightSunDiffuse constant not found in pixel shader");
        }
        
        D3DXHANDLE hLightSceneAmbient = hlslShader.psConstantTable->GetConstantByName(NULL, "lightSceneAmbient");
        if (hLightSceneAmbient) {
            hlslShader.psConstantTable->SetFloatArray(device, hLightSceneAmbient, (const float*)&ambient, 3);
        } else {
            LOG::logline("!! lightSceneAmbient constant not found in pixel shader");
        }
        
        // Set light arrays in pixel shader (same as Effect shader approach)
        D3DXHANDLE hLightDiffuse = hlslShader.psConstantTable->GetConstantByName(NULL, "lightDiffuse");
        if (hLightDiffuse) {
            hlslShader.psConstantTable->SetVectorArray(device, hLightDiffuse, bufferDiffuse, MaxLights);
        } else {
            LOG::logline("!! lightDiffuse array constant not found in pixel shader");
        }
        
        D3DXHANDLE hLightPosition = hlslShader.psConstantTable->GetConstantByName(NULL, "lightPosition");
        if (hLightPosition) {
            hlslShader.psConstantTable->SetVectorArray(device, hLightPosition, (D3DXVECTOR4*)bufferPosition, MaxLights);
        } else {
            LOG::logline("!! lightPosition array constant not found in pixel shader");
        }
        
        // Set light ambient array
        D3DXHANDLE hLightAmbient = hlslShader.psConstantTable->GetConstantByName(NULL, "lightAmbient");
        if (hLightAmbient) {
            hlslShader.psConstantTable->SetFloatArray(device, hLightAmbient, bufferAmbient, MaxLights);
        } else {
            LOG::logline("!! lightAmbient array constant not found in pixel shader");
        }
        
        // Debug: List all available constants in pixel shader
        if (debugFrameCount % 60 == 0) {
            D3DXCONSTANTTABLE_DESC desc;
            if (SUCCEEDED(hlslShader.psConstantTable->GetDesc(&desc))) {
                LOG::logline("=== Available PS Constants ===");
                LOG::logline("Total constants: %d", desc.Constants);
                for (UINT i = 0; i < desc.Constants && i < 20; i++) {  // Limit to first 20
                    D3DXHANDLE handle = hlslShader.psConstantTable->GetConstant(NULL, i);
                    if (handle) {
                        D3DXCONSTANT_DESC constDesc;
                        UINT count = 1;
                        if (SUCCEEDED(hlslShader.psConstantTable->GetConstantDesc(handle, &constDesc, &count))) {
                            LOG::logline("Constant[%d]: %s", i, constDesc.Name ? constDesc.Name : "NULL");
                        }
                    }
                }
            }
        }
        
        // Set falloff constants exactly like Effect shader
        D3DXHANDLE hLightFalloffQuadratic = hlslShader.psConstantTable->GetConstantByName(NULL, "lightFalloffQuadratic");
        if (hLightFalloffQuadratic) {
            hlslShader.psConstantTable->SetFloatArray(device, hLightFalloffQuadratic, bufferFalloffQuadratic, MaxLights);
            
            // Debug: Log falloff values every 60 frames
            if (debugFrameCount % 60 == 0) {
                LOG::logline("=== Falloff Constants Debug ===");
                LOG::logline("bufferFalloffQuadratic[0]: %f", bufferFalloffQuadratic[0]);
                LOG::logline("bufferFalloffConstant: %f", bufferFalloffConstant);
                
                // Calculate what Effect shader attenuation would be with logged distance
                if (pointLightCount > 0) {
                    float dist = sqrt(bufferPosition[0].x * bufferPosition[0].x + 
                                     bufferPosition[0].y * bufferPosition[0].y + 
                                     bufferPosition[0].z * bufferPosition[0].z);
                    float effectFalloff = bufferFalloffQuadratic[0] * dist * dist + bufferFalloffConstant;
                    float effectAttenuation = (effectFalloff > 0) ? (1.0f / effectFalloff) : 0.0f;
                    float hlslScaled = 15000.0f / (dist * dist);
                    
                    LOG::logline("Distance: %f, Effect falloff: %f, Effect atten: %f", dist, effectFalloff, effectAttenuation);
                    LOG::logline("HLSL 15000 scaled atten: %f, Ratio: %f", hlslScaled, hlslScaled / effectAttenuation);
                }
            }
        } else {
            LOG::logline("!! lightFalloffQuadratic array constant not found in pixel shader");
        }
        
        D3DXHANDLE hLightFalloffConstant = hlslShader.psConstantTable->GetConstantByName(NULL, "lightFalloffConstant");
        if (hLightFalloffConstant) {
            hlslShader.psConstantTable->SetFloat(device, hLightFalloffConstant, bufferFalloffConstant);
        } else {
            LOG::logline("!! lightFalloffConstant constant not found in pixel shader");
        }
        
        // Set shading mode from actual material mode calculation
        D3DXHANDLE hShadingMode = hlslShader.psConstantTable->GetConstantByName(NULL, "shadingMode");
        if (hShadingMode) {
            float shadingModeData[4] = {0, 0, (float)sk.vertexMaterial, 0};
            hlslShader.psConstantTable->SetFloatArray(device, hShadingMode, shadingModeData, 4);
        } else {
            LOG::logline("!! shadingMode constant not found in pixel shader");
        }
        
        // Set fog color
        DWORD fogColorDword = 0x808080FF;
        device->GetRenderState(D3DRS_FOGCOLOR, &fogColorDword);
        D3DXVECTOR4 fogColor(
            ((fogColorDword >> 16) & 0xFF) / 255.0f,
            ((fogColorDword >> 8) & 0xFF) / 255.0f,
            (fogColorDword & 0xFF) / 255.0f,
            1.0f
        );
        
        D3DXHANDLE hFogColNear = hlslShader.psConstantTable->GetConstantByName(NULL, "fogColNear");
        if (hFogColNear) {
            hlslShader.psConstantTable->SetVector(device, hFogColNear, &fogColor);
        }
        
        // Copy texture bindings from device like ID3DXEffect system (supports Combined shader)
        for (int i = 0; i < 6; ++i) {
            IDirect3DBaseTexture9* tex;
            device->GetTexture(i, &tex);
            if (tex) {
                device->SetTexture(i, tex);
                tex->Release();
            }
        }
        
        // Set proper sampler states for texturing
        device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
        device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
        device->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
        device->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
        device->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);
        
        // Set missing constants that the Combined shader expects
        D3DXHANDLE hTexgenTransform = hlslShader.vsConstantTable->GetConstantByName(NULL, "texgenTransform");
        if (hTexgenTransform) {
            D3DXMATRIX identity;
            D3DXMatrixIdentity(&identity);
            hlslShader.vsConstantTable->SetMatrix(device, hTexgenTransform, &identity);
        }
        
        D3DXHANDLE hBumpMatrix = hlslShader.psConstantTable->GetConstantByName(NULL, "bumpMatrix");
        if (hBumpMatrix) {
            D3DXVECTOR4 bumpMatrix(1, 0, 0, 1);  // Identity 2x2 matrix
            hlslShader.psConstantTable->SetVector(device, hBumpMatrix, &bumpMatrix);
        }
        
        D3DXHANDLE hBumpLumiScaleBias = hlslShader.psConstantTable->GetConstantByName(NULL, "bumpLumiScaleBias");
        if (hBumpLumiScaleBias) {
            D3DXVECTOR2 scaleBias(1, 0);  // Scale=1, Bias=0
            hlslShader.psConstantTable->SetFloatArray(device, hBumpLumiScaleBias, (float*)&scaleBias, 2);
        }
        
        // Set critical shared variables that the Combined shader needs
        D3DXHANDLE hHasAlpha = hlslShader.psConstantTable->GetConstantByName(NULL, "hasAlpha");
        if (hHasAlpha) {
            hlslShader.psConstantTable->SetBool(device, hHasAlpha, false);
        }
        
        D3DXHANDLE hHasBones = hlslShader.vsConstantTable->GetConstantByName(NULL, "hasBones");
        if (hHasBones) {
            hlslShader.vsConstantTable->SetBool(device, hHasBones, rs->vertexBlendState > 0);
        }
        
        D3DXHANDLE hHasVCol = hlslShader.psConstantTable->GetConstantByName(NULL, "hasVCol");
        if (hHasVCol) {
            hlslShader.psConstantTable->SetBool(device, hHasVCol, (rs->fvf & D3DFVF_DIFFUSE) != 0);
        }
        
        D3DXHANDLE hMaterialAlpha = hlslShader.psConstantTable->GetConstantByName(NULL, "materialAlpha");
        if (hMaterialAlpha) {
            hlslShader.psConstantTable->SetFloat(device, hMaterialAlpha, frs->material.diffuse.a);
        }
        
        D3DXHANDLE hAlphaRef = hlslShader.psConstantTable->GetConstantByName(NULL, "alphaRef");
        if (hAlphaRef) {
            hlslShader.psConstantTable->SetFloat(device, hAlphaRef, rs->alphaRef / 255.0f);
        }
    }
    
    // Error checking for vertex/index buffers
    if (!rs->vb) {
        LOG::logline("!! HLSL pipeline: null vertex buffer, skipping draw call");
        return;
    }
    
    // Set vertex declaration and stream sources with error checking
    HRESULT hr = device->SetFVF(rs->fvf);
    if (FAILED(hr)) {
        LOG::logline("!! HLSL pipeline: failed to set FVF %x, hr=%x", rs->fvf, hr);
        return;
    }
    
    hr = device->SetStreamSource(0, rs->vb, rs->vbOffset, rs->vbStride);
    if (FAILED(hr)) {
        LOG::logline("!! HLSL pipeline: failed to set vertex buffer, hr=%x", hr);
        return;
    }
    
    // Execute the draw call with proper error checking
    if (rs->ib) {
        hr = device->SetIndices(rs->ib);
        if (FAILED(hr)) {
            LOG::logline("!! HLSL pipeline: failed to set index buffer, hr=%x", hr);
            return;
        }
        device->DrawIndexedPrimitive(rs->primType, rs->ibBase, rs->minIndex, rs->vertCount, rs->startIndex, rs->primCount);
    } else {
        device->DrawPrimitive(rs->primType, rs->startIndex, rs->primCount);
    }
    
    // Restore device state after HLSL rendering (like the original system does)
    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);
    
    // Restore critical render states that affect other rendering modes
    device->SetRenderState(D3DRS_LIGHTING, savedLighting);
    device->SetRenderState(D3DRS_FOGENABLE, savedFogEnable);
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, savedAlphaBlendEnable);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, savedAlphaTestEnable);
    device->SetRenderState(D3DRS_ZENABLE, savedZEnable);
    device->SetRenderState(D3DRS_ZWRITEENABLE, savedZWriteEnable);
}

FixedFunctionShader::HLSLShader FixedFunctionShader::generateMWShaderHLSL(const ShaderKey& sk) {
    HLSLShader hlslShader = {};
    
    // Use Simple shader with fixed positioning
    const char* vertexShaderName = "vs_main";
    const char* pixelShaderName = "ps_main";
    
    // Determine light count from shader key (matching Effect shader logic)
    int lightCount = 0;
    if (sk.vertexMaterial != 0) {
        lightCount = sk.heavyLighting ? 8 : 4;
    }
    
    // Load Simple shader source from file 
    HANDLE hFile = CreateFileA("Data Files\\shaders\\core-hlsl\\XE FixedFuncEmu.hlsl", 
                               GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        LOG::logline("!! HLSL file not found: XE FixedFuncEmu.hlsl");
        return hlslShaderDefaultPurple;
    }
    
    DWORD fileSize = GetFileSize(hFile, nullptr);
    char* shaderSource = new char[fileSize + 1];
    DWORD bytesRead;
    ReadFile(hFile, shaderSource, fileSize, &bytesRead, nullptr);
    shaderSource[fileSize] = '\0';
    CloseHandle(hFile);
    
    // Create shader defines for light count (like Effect shader FFE_LIGHTS_ACTIVE)
    char lightCountStr[16];
    sprintf(lightCountStr, "%d", lightCount);
    D3D_SHADER_MACRO defines[] = {
        { "LIGHT_COUNT", lightCountStr },
        { nullptr, nullptr }
    };
    
    LOG::logline("-- Generating HLSL shader with %d lights (heavyLighting=%d, vertexMaterial=%d)", 
                 lightCount, sk.heavyLighting, sk.vertexMaterial);
    if (lightCount == 0) {
        LOG::logline("!! WARNING: Compiling shader with NO LIGHTING (lightCount=0)");
    }

    // Compile vertex shader
    ID3DBlob* vsBlob = nullptr;
    ID3DBlob* vsErrors = nullptr;
    
    HRESULT hr = D3DCompile(
        shaderSource,
        fileSize,
        "XE FixedFuncEmu.hlsl",
        defines, // Shader defines for light count
        nullptr, // Include handler
        vertexShaderName,
        "vs_3_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &vsBlob,
        &vsErrors
    );
    
    if (FAILED(hr)) {
        if (vsErrors) {
            LOG::write("!! HLSL Vertex Shader compile errors:\n");
            LOG::write(reinterpret_cast<const char*>(vsErrors->GetBufferPointer()));
            LOG::write("\n");
            vsErrors->Release();
        }
        LOG::logline("!! HLSL Vertex Shader compilation failed, using default");
        delete[] shaderSource;
        return hlslShaderDefaultPurple;
    }
    
    // Create vertex shader
    hr = device->CreateVertexShader(
        reinterpret_cast<DWORD*>(vsBlob->GetBufferPointer()),
        &hlslShader.vertexShader
    );
    
    if (FAILED(hr)) {
        LOG::logline("!! Failed to create HLSL vertex shader");
        vsBlob->Release();
        delete[] shaderSource;
        return hlslShaderDefaultPurple;
    }
    
    // Get constant table for vertex shader
    hr = D3DXGetShaderConstantTable(
        reinterpret_cast<DWORD*>(vsBlob->GetBufferPointer()),
        &hlslShader.vsConstantTable
    );
    
    vsBlob->Release();
    
    // Compile pixel shader using same source
    ID3DBlob* psBlob = nullptr;
    ID3DBlob* psErrors = nullptr;
    
    hr = D3DCompile(
        shaderSource,
        fileSize,
        "XE FixedFuncEmu.hlsl",
        defines, // Shader defines for light count
        nullptr, // Include handler
        pixelShaderName,
        "ps_3_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &psBlob,
        &psErrors
    );
    
    if (FAILED(hr)) {
        if (psErrors) {
            LOG::write("!! HLSL Pixel Shader compile errors:\n");
            LOG::write(reinterpret_cast<const char*>(psErrors->GetBufferPointer()));
            LOG::write("\n");
            psErrors->Release();
        }
        LOG::logline("!! HLSL Pixel Shader compilation failed, using default");
        // Clean up vertex shader
        if (hlslShader.vertexShader) hlslShader.vertexShader->Release();
        if (hlslShader.vsConstantTable) hlslShader.vsConstantTable->Release();
        delete[] shaderSource;
        return hlslShaderDefaultPurple;
    }
    
    // Create pixel shader
    hr = device->CreatePixelShader(
        reinterpret_cast<DWORD*>(psBlob->GetBufferPointer()),
        &hlslShader.pixelShader
    );
    
    if (FAILED(hr)) {
        LOG::logline("!! Failed to create HLSL pixel shader");
        psBlob->Release();
        // Clean up vertex shader
        if (hlslShader.vertexShader) hlslShader.vertexShader->Release();
        if (hlslShader.vsConstantTable) hlslShader.vsConstantTable->Release();
        delete[] shaderSource;
        return hlslShaderDefaultPurple;
    }
    
    // Get constant table for pixel shader
    hr = D3DXGetShaderConstantTable(
        reinterpret_cast<DWORD*>(psBlob->GetBufferPointer()),
        &hlslShader.psConstantTable
    );
    
    psBlob->Release();
    
    // Clean up shader source
    delete[] shaderSource;
    
    // Cache the compiled shader
    cacheHLSLShaders[sk] = hlslShader;
    
    LOG::logline("-- HLSL shader compiled successfully: VS=%s PS=%s", vertexShaderName, pixelShaderName);
    return hlslShader;
}

void FixedFunctionShader::release() {
    // Clean up D3DXEffect cache
    for (auto& i : cacheEffects) {
        if (i.second) {
            i.second->Release();
        }
    }

    shaderLRU.effect = nullptr;
    shaderLRU.last_sk = ShaderKey();
    cacheEffects.clear();
    effectDefaultPurple->Release();
    
    // Clean up HLSL cache
    for (auto& i : cacheHLSLShaders) {
        if (i.second.vertexShader) i.second.vertexShader->Release();
        if (i.second.pixelShader) i.second.pixelShader->Release();
        if (i.second.vsConstantTable) i.second.vsConstantTable->Release();
        if (i.second.psConstantTable) i.second.psConstantTable->Release();
    }
    hlslShaderLRU.shader = {};
    hlslShaderLRU.last_sk = ShaderKey();
    cacheHLSLShaders.clear();
    
    // Clean up default HLSL shader
    if (hlslShaderDefaultPurple.vertexShader) hlslShaderDefaultPurple.vertexShader->Release();
    if (hlslShaderDefaultPurple.pixelShader) hlslShaderDefaultPurple.pixelShader->Release();
    if (hlslShaderDefaultPurple.vsConstantTable) hlslShaderDefaultPurple.vsConstantTable->Release();
    if (hlslShaderDefaultPurple.psConstantTable) hlslShaderDefaultPurple.psConstantTable->Release();
}



// ShaderKey - Captures a generatable shader configuration

FixedFunctionShader::ShaderKey::ShaderKey(const RenderedState* rs, const FragmentState* frs, const LightState* lightrs) {
    memset(this, 0, sizeof(ShaderKey));         // Clear padding bits for compares

    uvSets = (rs->fvf & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT;
    usesSkinning = rs->vertexBlendState ? 1 : 0;
    vertexColour = (rs->fvf & D3DFVF_DIFFUSE) ? 1 : 0;

    // Match constant material, diffuse+ambient vcol, or emissive vcol
    static int debugLogCounter = 0;
    debugLogCounter++;
    
    if (rs->useLighting) {
        heavyLighting = (lightrs->active.size() > 4) ? 1 : 0;
        vertexMaterial = 1;

        if (vertexColour) {
            if (rs->matSrcDiffuse == D3DMCS_COLOR1) {
                vertexMaterial = 2;
            } else if (rs->matSrcEmissive == D3DMCS_COLOR1) {
                vertexMaterial = 3;
            }
        }
        
        // Log lighting decisions every 60 objects
        if (debugLogCounter % 60 == 0) {
            LOG::logline("=== ShaderKey Lighting Debug ===");
            LOG::logline("useLighting=true, activeLights=%d, heavyLighting=%d, vertexMaterial=%d", 
                         lightrs->active.size(), heavyLighting, vertexMaterial);
            LOG::logline("vertexColour=%d, matSrcDiffuse=0x%x, matSrcEmissive=0x%x", 
                         vertexColour, rs->matSrcDiffuse, rs->matSrcEmissive);
        }
    } else {
        // Object with no lighting - this could be the problem
        if (debugLogCounter % 60 == 0) {
            LOG::logline("=== ShaderKey NO LIGHTING ===");
            LOG::logline("useLighting=FALSE -> vertexMaterial=0 (no lights)");
            LOG::logline("activeLights=%d, vertexColour=%d", lightrs->active.size(), vertexColour);
        }
    }

    if (rs->useFog) {
        // Match premultipled alpha or additive blending
        if (rs->blendEnable && (rs->srcBlend == D3DBLEND_ONE || rs->destBlend == D3DBLEND_ONE)) {
            fogMode = 2;
        } else {
            fogMode = 1;
        }
    }

    DWORD maxTexcoordIndex = 0;
    bool bumpStageFixup = false;

    for (int i = 0; i != 8; ++i) {
        const FragmentState::Stage& s = frs->stage[i];

        if (s.colorOp == D3DTOP_DISABLE) {
            activeStages = i;
            break;
        }

        stage[i].colorOp = s.colorOp;
        stage[i].colorArg1 = s.colorArg1;
        stage[i].colorArg2 = s.colorArg2;
        stage[i].colorArg0 = s.colorArg0;
        stage[i].alphaOpMatched = (s.alphaOp == s.colorOp);
        stage[i].alphaOpSelect1 = (s.alphaOp == D3DTOP_SELECTARG1 && s.alphaArg1 == s.colorArg1);
        stage[i].texcoordIndex = s.texcoordIndex & 3;
        stage[i].texcoordGen = s.texcoordIndex >> 16;
        maxTexcoordIndex = std::max(maxTexcoordIndex, (DWORD)stage[i].texcoordIndex);

        if (s.colorOp == D3DTOP_BUMPENVMAP || s.colorOp == D3DTOP_BUMPENVMAPLUMINANCE) {
            usesBumpmap = 1;
            bumpmapStage = i;
            stage[i].alphaOpMatched = false;
            stage[i].alphaOpSelect1 = false;
            bumpStageFixup = true;
        } else if (bumpStageFixup) {
            stage[i].alphaOpMatched = false;
            stage[i].alphaOpSelect1 = false;
            bumpStageFixup = false;
        }

        if (stage[i].texcoordGen) {
            usesTexgen = 1;
            projectiveTexgen = (s.texTransformFlags == (D3DTTFF_COUNT3 | D3DTTFF_PROJECTED)) ? 1 : 0;
            texgenStage = i;
        }
    }

    // Generate based on actual UV sets available and used
    DWORD usedUVSets = maxTexcoordIndex + 1;
    uvSets = std::min((DWORD)uvSets, usedUVSets);
}

bool FixedFunctionShader::ShaderKey::operator<(const ShaderKey& other) const {
    return memcmp(this, &other, sizeof(ShaderKey)) < 0;
}

bool FixedFunctionShader::ShaderKey::operator==(const ShaderKey& other) const {
    return memcmp(this, &other, sizeof(ShaderKey)) == 0;
}

std::size_t FixedFunctionShader::ShaderKey::hasher::operator()(const ShaderKey& k) const {
    DWORD z[9];
    memcpy(&z, &k, sizeof(z));
    return (z[0] << 16) ^ z[1] ^ z[2] ^ z[3] ^ z[4] ^ z[5] ^ z[6] ^ z[7] ^ z[8];
}

void FixedFunctionShader::ShaderKey::log() const {
    const char* opSymbols[] = { "?", "disable", "select1", "select2", "mul", "mul2x", "mul4x", "add", "addsigned", "addsigned2x", "sub", "?", "blend.diffuse", "blend.texture", "?", "?", "?", "?", "?", "?", "?", "?", "bump", "bump.l", "dp3", "mad", "?" };
    const char* argSymbols[] = { "diffuse", "current", "texture", "tfactor", "specular", "temp", "constant" };
    const char* texgenSymbols[] = { "none", "normal", "position", "reflection", "sphere" };

    const unsigned char *dump = (const unsigned char*)this;
    stringstream stream;
    stream << "   Hex: ";
    for(int i = 0; i < sizeof *this; ++i) {
        char hex[4];
        snprintf(hex, sizeof hex, "%02x ", dump[i]);
        stream << hex;
    }
    LOG::logline("%s", stream.str().c_str());

    LOG::logline("   Input state: UVs:%d skin:%d vcol:%d lights:%d vmat:%d fogm:%d", uvSets, usesSkinning, vertexColour, vertexMaterial ? (heavyLighting ? 8 : 4) : 0, vertexMaterial, fogMode);
    LOG::logline("   Texture stages:");
    for (int i = 0; i != activeStages; ++i) {
        const auto& s = stage[i];
        if (s.colorOp != D3DTOP_MULTIPLYADD) { // or D3DTOP_LERP (unused)
            LOG::logline("    [%d] %s % 12s    %s, %s            uv %d texgen %s", i,
                         s.alphaOpMatched ? "RGBA" : "RGB ",
                         opSymbols[s.colorOp], argSymbols[s.colorArg1], argSymbols[s.colorArg2],
                         s.texcoordIndex, texgenSymbols[s.texcoordGen]);
        } else {
            LOG::logline("    [%d] %s % 12s    %s, %s, %s   uv %d texgen %s", i,
                         s.alphaOpMatched ? "RGBA" : "RGB ",
                         opSymbols[s.colorOp], argSymbols[s.colorArg1], argSymbols[s.colorArg2], argSymbols[s.colorArg0],
                         s.texcoordIndex, texgenSymbols[s.texcoordGen]);
        }
        if (s.alphaOpSelect1) {
            LOG::logline("           A % 12s    %s", opSymbols[D3DTOP_SELECTARG1], argSymbols[s.colorArg1]);
        }
    }
    LOG::logline("");
}
