
#include "ffeshader.h"
#include "configuration.h"
#include "drawstats.h"
#include "scenegraph.h"
#include "support/log.h"

#include <Windows.h>
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
IDirect3DTexture9* FixedFunctionShader::texLightData = nullptr;
uint64_t FixedFunctionShader::lastUploadedRevision = (uint64_t)-1;
unsigned int FixedFunctionShader::lastUploadedPointCount = 0;
std::unordered_map<FixedFunctionShader::MeshKey, FixedFunctionShader::ObjectSpaceBBox,
                   FixedFunctionShader::MeshKeyHash> FixedFunctionShader::bboxCache;

D3DXHANDLE FixedFunctionShader::ehWorld, FixedFunctionShader::ehWorldView;
D3DXHANDLE FixedFunctionShader::ehView, FixedFunctionShader::ehBoneMatrices;
D3DXHANDLE FixedFunctionShader::ehVertexBlendState, FixedFunctionShader::ehVertexBlendPalette;
D3DXHANDLE FixedFunctionShader::ehTex0, FixedFunctionShader::ehTex1, FixedFunctionShader::ehTex2, FixedFunctionShader::ehTex3, FixedFunctionShader::ehTex4, FixedFunctionShader::ehTex5;
D3DXHANDLE FixedFunctionShader::ehMaterialDiffuse, FixedFunctionShader::ehMaterialAmbient, FixedFunctionShader::ehMaterialEmissive;
D3DXHANDLE FixedFunctionShader::ehLightSceneAmbient, FixedFunctionShader::ehLightSunDiffuse, FixedFunctionShader::ehLightDiffuse;
D3DXHANDLE FixedFunctionShader::ehLightSunDirection, FixedFunctionShader::ehLightPosition, FixedFunctionShader::ehLightAmbient;
D3DXHANDLE FixedFunctionShader::ehLightFalloffQuadratic, FixedFunctionShader::ehLightFalloffLinear, FixedFunctionShader::ehLightFalloffConstant;
D3DXHANDLE FixedFunctionShader::ehTexLightData, FixedFunctionShader::ehLightDataParams, FixedFunctionShader::ehLightIndices, FixedFunctionShader::ehTexLightView;
D3DXHANDLE FixedFunctionShader::ehTexgenTransform, FixedFunctionShader::ehBumpMatrix, FixedFunctionShader::ehBumpLumiScaleBias;
D3DXHANDLE FixedFunctionShader::ehPointLightMult;
D3DXHANDLE FixedFunctionShader::ehShadowAtlas, FixedFunctionShader::ehApplyCacheShadow;

float FixedFunctionShader::sunMultiplier, FixedFunctionShader::ambMultiplier;

// Texture-light selection instrumentation accumulators. Moved out of
// renderMorrowind's local statics so the extracted selectTextureLights() (shared
// with the cache terrain pass) can increment them while renderMorrowind's periodic
// [SELECTION-TIMING]/[UPLOAD-TIMING]/[PRECULL]/[SELECTED-DIST] log still reads
// them. Cumulative (never reset — reported as totals / frameCount).
namespace {
    unsigned long long s_selectionCalls   = 0;
    unsigned long long s_selectionTotalNs = 0;
    unsigned long long s_uploads          = 0;
    unsigned long long s_uploadsTotalNs   = 0;
    unsigned long long s_preCullAliveSum  = 0;
    unsigned long long s_preCullTotalSum  = 0;
    unsigned long long s_selectedBuckets[6] = {0};
    LARGE_INTEGER      s_qpcFreq = {};
}

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
    ehView = effect->GetParameterByName(0, "view");
    ehBoneMatrices = effect->GetParameterByName(0, "boneMatrices");
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
    ehTexLightData = effect->GetParameterByName(0, "texLightData");
    ehLightDataParams = effect->GetParameterByName(0, "lightDataParams");
    ehLightIndices = effect->GetParameterByName(0, "lightIndices");
    ehTexLightView = effect->GetParameterByName(0, "texLightView");
    ehPointLightMult = effect->GetParameterByName(0, "pointLightMult");
    ehTexgenTransform = effect->GetParameterByName(0, "texgenTransform");
    ehBumpMatrix = effect->GetParameterByName(0, "bumpMatrix");
    ehBumpLumiScaleBias = effect->GetParameterByName(0, "bumpLumiScaleBias");
    ehShadowAtlas = effect->GetParameterByName(0, "texShadowAtlas");
    ehApplyCacheShadow = effect->GetParameterByName(0, "applyCacheShadow");
    // Explicit off at init so the main reactive scene never takes the shadow-fold
    // branch until the cache reflection pass enables it (shared-pool param).
    if (ehApplyCacheShadow) effect->SetBool(ehApplyCacheShadow, FALSE);

    effectDefaultPurple = effect;
    sunMultiplier = ambMultiplier = 1.0;

    // Phase 2: allocate the dynamic light texture for the
    // USE_TEXTURE_LIGHTS shader path. Width = kTexelsPerLight * kMaxTexLights
    // = 192 texels at default; height = 1; R32G32B32A32F so each texel
    // holds 4 floats. D3DUSAGE_DYNAMIC + D3DPOOL_DEFAULT = update via
    // LockRect with LOCKED_DISCARD on the render thread without driver
    // flush stalls. Sampler binding happens in renderMorrowind per draw
    // when sk.useTextureLightVariant is set.
    if (texLightData) {
        texLightData->Release();
        texLightData = nullptr;
    }
    {
        const UINT texW = kTexelsPerLight * kMaxTexLights;
        HRESULT thr = device->CreateTexture(texW, 1, 1,
            D3DUSAGE_DYNAMIC, D3DFMT_A32B32G32R32F,
            D3DPOOL_DEFAULT, &texLightData, nullptr);
        if (thr != D3D_OK) {
            LOG::logline("!! FFE light texture create failed: 0x%08x", (unsigned)thr);
            texLightData = nullptr;
        } else {
            LOG::logline("-- FFE light texture created: %ux1, %u lights * %u texels",
                         texW, kMaxTexLights, kTexelsPerLight);
        }
    }
    lastUploadedRevision = (unsigned int)-1;
    lastUploadedPointCount = 0;
    // Bbox cache is keyed by D3D resource pointers; pointers
    // become invalid after device reset, so wipe.
    bboxCache.clear();

    // Clear cache and LRU, important if the renderer resets
    shaderLRU.effect = nullptr;
    shaderLRU.last_sk = ShaderKey();
    cacheEffects.clear();

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

            // Light bucket sweep: 0 = 4 lights (heavyLighting=0),
            // 1 = 8 lights (heavyLighting=1), 2 = 64 lights (useTextureLightVariant=1).
            // The 64-light variant only fires when msoc emits lights, but we
            // pre-compile it so the first dense-interior frame doesn't stutter.
            for (int lightBucket = 0; lightBucket <= 2; ++lightBucket) {
                skCommon.heavyLighting = (lightBucket == 1) ? 1 : 0;
                skCommon.useTextureLightVariant = (lightBucket == 2) ? 1 : 0;

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

void FixedFunctionShader::setCacheShadow(IDirect3DTexture9* atlas, bool enable) {
    // Set on any pooled FFE effect; `shared` params propagate to every variant.
    if (!effectDefaultPurple) return;
    if (ehShadowAtlas) effectDefaultPurple->SetTexture(ehShadowAtlas, atlas);
    if (ehApplyCacheShadow) effectDefaultPurple->SetBool(ehApplyCacheShadow, enable ? TRUE : FALSE);
}

// selectTextureLights — revision-keyed texLightData upload + view-keyed frustum
// precull + per-mesh sphere-AABB nearest-kMaxIndicesPerMesh selection. Extracted
// from renderMorrowind so the cache terrain pass (renderCachedTerrain) selects
// lights with byte-identical logic — parity with the reactive path (the A/B
// light-seam fix). The caller must hold MGE::SceneGraph::SnapshotReadLock and pass
// the locked snapshot + its kMaxTexLights-clamped count, the draw's view matrix,
// and the draw's world-space AABB. device's D3DTS_PROJECTION must be the matching
// projection (drives the precull frustum). Fills idxFloats (>= kMaxIndicesPerMesh)
// and returns the selected light count (0 if no snapshot / no texture / none in range).
int FixedFunctionShader::selectTextureLights(
        const std::vector<MGE::SceneGraph::PointLight>& snapshotLights,
        unsigned int snapshotCount, const D3DXMATRIX& view,
        const D3DXVECTOR3& bMin, const D3DXVECTOR3& bMax,
        float* idxFloats, bool logPerf) {
    if (snapshotCount == 0 || !texLightData) return 0;
    if (s_qpcFreq.QuadPart == 0) QueryPerformanceFrequency(&s_qpcFreq);

    // Per-draw selection scratch (persistent statics; no hot-path allocation).
    // s_lightWorldPos caches world positions for the sphere-AABB test; s_lightAlive
    // flags lights whose 2*radius sphere intersects the camera frustum (precull
    // recomputed only when the view matrix changes); s_lastPrecullView caches that
    // view. s_msocToTex maps snapshot index → texture row (identity here).
    static unsigned int s_msocToTex[kMaxTexLights];
    static D3DXVECTOR3 s_lightWorldPos[kMaxTexLights];
    static bool        s_lightAlive[kMaxTexLights];
    static D3DXMATRIX  s_lastPrecullView = {};

    const uint64_t currentRev = MGE::SceneGraph::frameRevision();
    const bool needUpload = (currentRev != lastUploadedRevision);

    if (needUpload && texLightData) {
        LARGE_INTEGER tsUploadBegin{};
        if (logPerf) QueryPerformanceCounter(&tsUploadBegin);
        D3DLOCKED_RECT locked;
        if (SUCCEEDED(texLightData->LockRect(0, &locked, nullptr, D3DLOCK_DISCARD))) {
            float* dst = (float*)locked.pBits;
            memset(dst, 0, kTexelsPerLight * kMaxTexLights * 4 * sizeof(float));

            for (unsigned int i = 0; i < snapshotCount; ++i) {
                const auto& pl = snapshotLights[i];
                s_lightWorldPos[i] = D3DXVECTOR3(pl.worldPos[0], pl.worldPos[1], pl.worldPos[2]);

                const unsigned int t = i * kTexelsPerLight * 4;
                dst[t + 0] = pl.worldPos[0];
                dst[t + 1] = pl.worldPos[1];
                dst[t + 2] = pl.worldPos[2];
                dst[t + 4] = pl.diffuse[0];
                dst[t + 5] = pl.diffuse[1];
                dst[t + 6] = pl.diffuse[2];
                dst[t + 8]  = pl.falloff[0];
                dst[t + 9]  = pl.falloff[1];
                dst[t + 10] = pl.falloff[2];
                dst[t + 11] = pl.radius;

                s_msocToTex[i] = i;
            }
            texLightData->UnlockRect(0);

            lastUploadedRevision = currentRev;
            lastUploadedPointCount = snapshotCount;
            memset(&s_lastPrecullView, 0, sizeof(s_lastPrecullView));
        }
        if (logPerf) {
            LARGE_INTEGER tsUploadEnd;
            QueryPerformanceCounter(&tsUploadEnd);
            const unsigned long long uploadTicks =
                static_cast<unsigned long long>(tsUploadEnd.QuadPart - tsUploadBegin.QuadPart);
            if (s_qpcFreq.QuadPart > 0) {
                s_uploadsTotalNs +=
                    uploadTicks * 1000000000ULL / static_cast<unsigned long long>(s_qpcFreq.QuadPart);
            }
            ++s_uploads;
        }
    }

    // Frustum precull. Recomputed whenever the view matrix changes OR a fresh upload
    // reset s_lastPrecullView. Marks each light "alive" if its 2*radius influence
    // sphere intersects the camera frustum.
    if (lastUploadedPointCount > 0) {
        const bool precullStale = (memcmp(&s_lastPrecullView, &view, sizeof(D3DXMATRIX)) != 0);
        if (precullStale) {
            D3DMATRIX projTransform;
            device->GetTransform(D3DTS_PROJECTION, &projTransform);
            D3DXMATRIX viewProj;
            D3DXMatrixMultiply(&viewProj, &view, (const D3DXMATRIX*)&projTransform);
            D3DXPLANE planes[6] = {
                D3DXPLANE(viewProj._14 + viewProj._11, viewProj._24 + viewProj._21,
                          viewProj._34 + viewProj._31, viewProj._44 + viewProj._41),
                D3DXPLANE(viewProj._14 - viewProj._11, viewProj._24 - viewProj._21,
                          viewProj._34 - viewProj._31, viewProj._44 - viewProj._41),
                D3DXPLANE(viewProj._14 - viewProj._12, viewProj._24 - viewProj._22,
                          viewProj._34 - viewProj._32, viewProj._44 - viewProj._42),
                D3DXPLANE(viewProj._14 + viewProj._12, viewProj._24 + viewProj._22,
                          viewProj._34 + viewProj._32, viewProj._44 + viewProj._42),
                D3DXPLANE(viewProj._13, viewProj._23, viewProj._33, viewProj._43),
                D3DXPLANE(viewProj._14 - viewProj._13, viewProj._24 - viewProj._23,
                          viewProj._34 - viewProj._33, viewProj._44 - viewProj._43),
            };
            for (int p = 0; p < 6; ++p) D3DXPlaneNormalize(&planes[p], &planes[p]);

            unsigned int aliveCount = 0;
            for (unsigned int i = 0; i < lastUploadedPointCount; ++i) {
                const float r = snapshotLights[i].radius * 2.0f;
                const D3DXVECTOR3& lp = s_lightWorldPos[i];
                bool alive = true;
                for (int p = 0; p < 6; ++p) {
                    const float d = D3DXPlaneDotCoord(&planes[p], &lp);
                    if (d < -r) { alive = false; break; }
                }
                s_lightAlive[i] = alive;
                if (alive) ++aliveCount;
            }
            if (logPerf) {
                s_preCullAliveSum += aliveCount;
                s_preCullTotalSum += lastUploadedPointCount;
            }
            s_lastPrecullView = view;
        }
    }

    unsigned int candidateCount = 0;
    if (lastUploadedPointCount > 0) {
        LARGE_INTEGER selBegin{};
        if (logPerf) QueryPerformanceCounter(&selBegin);

        struct Cand { unsigned int texIdx; float dist2OverR2; };
        Cand candidates[kMaxTexLights];

        for (unsigned int i = 0; i < snapshotCount; ++i) {
            if (!s_lightAlive[i]) continue;
            const auto& pl = snapshotLights[i];
            if (pl.radius <= 0.0f) continue;
            const float effR  = pl.radius * 2.0f;
            const float effR2 = effR * effR;

            const D3DXVECTOR3& lp = s_lightWorldPos[i];
            const float cx = std::max(bMin.x, std::min(bMax.x, lp.x));
            const float cy = std::max(bMin.y, std::min(bMax.y, lp.y));
            const float cz = std::max(bMin.z, std::min(bMax.z, lp.z));
            const float dx = lp.x - cx, dy = lp.y - cy, dz = lp.z - cz;
            const float dist2 = dx*dx + dy*dy + dz*dz;

            if (dist2 < effR2) {
                candidates[candidateCount].texIdx = i;
                const float r2 = pl.radius * pl.radius;
                candidates[candidateCount].dist2OverR2 = dist2 / r2;
                ++candidateCount;
            }
        }

        if (candidateCount > kMaxIndicesPerMesh) {
            std::partial_sort(
                candidates, candidates + kMaxIndicesPerMesh,
                candidates + candidateCount,
                [](const Cand& a, const Cand& b) {
                    return a.dist2OverR2 < b.dist2OverR2;
                });
            candidateCount = kMaxIndicesPerMesh;
        }

        if (logPerf) {
            if (candidateCount == 0)                       ++s_selectedBuckets[0];
            else if (candidateCount <= 4)                  ++s_selectedBuckets[1];
            else if (candidateCount <= 8)                  ++s_selectedBuckets[2];
            else if (candidateCount <= 16)                 ++s_selectedBuckets[3];
            else if (candidateCount < kMaxIndicesPerMesh)  ++s_selectedBuckets[4];
            else                                           ++s_selectedBuckets[5];
        }

        for (unsigned int j = 0; j < candidateCount; ++j) {
            idxFloats[j] = (float)candidates[j].texIdx;
        }

        if (logPerf) {
            LARGE_INTEGER selEnd;
            QueryPerformanceCounter(&selEnd);
            const unsigned long long selDeltaTicks =
                (unsigned long long)(selEnd.QuadPart - selBegin.QuadPart);
            if (s_qpcFreq.QuadPart > 0) {
                s_selectionTotalNs +=
                    selDeltaTicks * 1000000000ULL / (unsigned long long)s_qpcFreq.QuadPart;
            }
            ++s_selectionCalls;
        }
    }
    return (int)candidateCount;
}

void FixedFunctionShader::renderMorrowind(const RenderedState* rs, const FragmentState* frs, LightState* lightrs, float pointLightMult,
                                          const D3DXMATRIX* cacheBonePalette, int cacheNumBones, const D3DXMATRIX* cacheView,
                                          const D3DXVECTOR3* cacheWorldBoundsMin, const D3DXVECTOR3* cacheWorldBoundsMax) {
    ID3DXEffect* effectFFE;

    // Instrument: per-draw stats — engine lightrs.active.size(), msoc snapshot
    // size, and per-variant timing (QPC delta around the function body).
    // Variant buckets: 0 = no point lights, 1 = 4-light shader, 2 = 8-light,
    // 3 = 64-light (msoc-emit). Per-variant ns totals + call counts let us
    // measure the cost delta when msoc-emit fires the 64-light path vs the
    // existing 4/8 buckets. QPC overhead is ~80 ns/call * 2 calls/draw =
    // ~240 us/frame at 1500 draws — diagnostic-grade, not free.
    static size_t s_peakLights = 0;
    static size_t s_peakMsoc = 0;
    static unsigned long long s_callCount = 0;
    static unsigned long long s_lastReportCall = 0;
    static unsigned long long s_engineBuckets[6] = {0};
    static unsigned long long s_msocBuckets[8] = {0};
    static unsigned long long s_variantCalls[4] = {0};
    static unsigned long long s_variantTotalNs[4] = {0};
    // Per-mesh selection / upload / precull / selected-dist accumulators moved to
    // file scope (s_selectionCalls, s_selectionTotalNs, s_uploads, s_uploadsTotalNs,
    // s_preCullAliveSum, s_preCullTotalSum, s_selectedBuckets, s_qpcFreq) so the
    // extracted selectTextureLights() — shared with the cache terrain pass — feeds
    // the same periodic histograms this function logs below.
    if (s_qpcFreq.QuadPart == 0) QueryPerformanceFrequency(&s_qpcFreq);

    // Cache the LogDistantPipeline flag once per draw. All per-draw
    // instrumentation (QPC pairs, counter buckets, peak-tracking, periodic
    // dumps) gates on this. Off = ~100 us/frame saved in dense PPL scenes
    // (mostly from skipped QPC pairs; ~30 ns each × ~4 sites × ~900 draws).
    const bool logPerf = Configuration.LogDistantPipeline;

    LARGE_INTEGER tsBegin{};
    if (logPerf) QueryPerformanceCounter(&tsBegin);

    if (logPerf) {
        const size_t activeSize = lightrs->active.size();
        ++s_callCount;
        if (activeSize <= 4) ++s_engineBuckets[0];
        else if (activeSize <= 7) ++s_engineBuckets[1];
        else if (activeSize == 8) ++s_engineBuckets[2];
        else if (activeSize <= 12) ++s_engineBuckets[3];
        else if (activeSize <= 16) ++s_engineBuckets[4];
        else ++s_engineBuckets[5];

        if (activeSize > s_peakLights) {
            s_peakLights = activeSize;
            LOG::logline("-- [LIGHTS-INSTR] new peak lightrs.active.size = %zu", activeSize);
        }
    }

    // Many-lights consumer (texture-backed). MGE::SceneGraph captures every
    // NiLight in worldObjectRoot + worldPickObjectRoot + sgSunlight each
    // throttled rebuild and pre-extracts point-light fields into a POD
    // vector (engine branching for standard / MCP magic / projectile /
    // spell-effect lights applied SceneGraph-side). When non-empty, we
    // pack the POD entries into texLightData (3 texels per light) once
    // per revision, bind that texture to sampler 7 + push lightDataParams,
    // and let the shader's USE_TEXTURE_LIGHTS variant iterate
    // runtime-bounded lights with per-pixel attenuation.
    //
    // Directional sun handling stays via the engine loop below — the sun
    // is set up via MGEProxyDevice::SetLight intercept and threaded
    // through lightrs->active. Only point-light fill is replaced.
    // The async scene-graph walk worker (scenegraph.cpp rebuildAsync)
    // swaps g_pointLights under SnapshotReadLock; a swap landing mid-read frees
    // the buffer this reference indexes (the c0000005 in renderMorrowind+0x42e:
    // stale base + clamped count walked off the reallocated array). Hold the
    // lock across the count sample AND every snapshotLights read so the buffer
    // can't move and the loop bound can't outlive it. useTextureLights /
    // candidateCount / idxFloats are declared above the block so they survive
    // to the shader-param pushes below; the lock releases at the block's close,
    // before the caller (inspectIndexedPrimitive) issues DrawIndexedPrimitive.
    // In synchronous mode the lock is a documented no-op.
    bool useTextureLights = false;
    unsigned int candidateCount = 0;
    float idxFloats[8 * 4] = { 0 };
    {
    MGE::SceneGraph::SnapshotReadLock snapshotLock;
    const auto& snapshotLights = MGE::SceneGraph::pointLights();
    const unsigned int snapshotCount =
        std::min<unsigned int>(static_cast<unsigned int>(snapshotLights.size()), kMaxTexLights);
    useTextureLights = (snapshotCount > 0) && (texLightData != nullptr);

    // Instrument: snapshot size distribution. Gated by logPerf so the bucket
    // increments + peak-tracking + log all skip when LogDistantPipeline is off.
    if (logPerf) {
        if (snapshotCount == 0)         ++s_msocBuckets[0];
        else if (snapshotCount <= 8)    ++s_msocBuckets[1];
        else if (snapshotCount <= 16)   ++s_msocBuckets[2];
        else if (snapshotCount <= 32)   ++s_msocBuckets[3];
        else if (snapshotCount <= 48)   ++s_msocBuckets[4];
        else if (snapshotCount <= 64)   ++s_msocBuckets[5];
        else if (snapshotCount <= 128)  ++s_msocBuckets[6];
        else                            ++s_msocBuckets[7];
        if (snapshotCount > s_peakMsoc) {
            s_peakMsoc = snapshotCount;
            LOG::logline("-- [LIGHTS-INSTR] new peak snapshotCount = %zu", (size_t)snapshotCount);
        }
    }

    // Per-mesh light selection — hoisted above ShaderKey so we know candidateCount
    // before picking the variant. The revision-keyed texLightData upload + view-keyed
    // frustum precull + per-mesh sphere-AABB cull are extracted into
    // selectTextureLights() so the cache terrain pass (renderCachedTerrain) selects
    // lights with BYTE-IDENTICAL logic — that's what keeps cache color in lockstep
    // with the reactive path (the A/B light-seam fix). Bounds: cache draws pass an
    // explicit world AABB (their VBs are D3DUSAGE_WRITEONLY, so computeBoundingBox
    // can't read them and would fall back to the object origin — the object seam);
    // reactive draws use computeBoundingBox, falling back to origin on lock failure.
    if (useTextureLights) {
        D3DXVECTOR3 bMin, bMax;
        if (cacheWorldBoundsMin) {
            bMin = *cacheWorldBoundsMin; bMax = *cacheWorldBoundsMax;
        } else if (!computeBoundingBox(rs, bMin, bMax)) {
            const D3DXMATRIX& wt = rs->worldTransforms[0];
            bMin = bMax = D3DXVECTOR3(wt._41, wt._42, wt._43);
        }
        candidateCount = selectTextureLights(snapshotLights, snapshotCount,
                                             rs->viewTransform, bMin, bMax, idxFloats, logPerf);
    }
    }   // release SnapshotReadLock — all snapshotLights reads complete

    // Fast-path gate: snapshot is active AND this mesh is in range of
    // at least one snapshot light. False here means use the engine 8-light
    // shader path with all-zero point buffers — only the directional sun
    // contributes (still correct since the snapshot says no point lights
    // reach this mesh).
    const bool actuallyUseTextureLights = useTextureLights && (candidateCount > 0);

    // Check if state matches last used effect
    ShaderKey sk(rs, frs, lightrs);
    if (cacheBonePalette) {
        // Cache 32-bone indexed skinning (mutually exclusive with the reactive
        // 4-matrix path, which the cache feed never uses — rs->vertexBlendState 0).
        sk.usesCacheSkin = 1;
    }
    if (actuallyUseTextureLights) {
        // Select the 64-light shader variant; per-pixel attenuation
        // will dim out lights that don't affect a given fragment.
        sk.useTextureLightVariant = 1;
    }

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

    // Per-draw point-light scale (1 in the reactive path; faded by the cache
    // reflection pass at the distant-land handover). Always pushed so it never
    // leaks a faded value into the next draw.
    if (ehPointLightMult) effectFFE->SetFloat(ehPointLightMult, pointLightMult);

    // Set up material
    effectFFE->SetVector(ehMaterialDiffuse, (D3DXVECTOR4*)&frs->material.diffuse);
    effectFFE->SetVector(ehMaterialAmbient, (D3DXVECTOR4*)&frs->material.ambient);
    effectFFE->SetVector(ehMaterialEmissive, (D3DXVECTOR4*)&frs->material.emissive);

    // Set up lighting (constant-array path — used when msoc-emit is off,
    // unchanged from upstream MGE-XE).
    const size_t MaxLights = 8;
    // Morrowind's conventional constant attenuation, used as the per-slot default
    // and as the multiplier for MCP-patched magic lights.
    const float kDefaultFalloffConstant = 0.33f;
    D3DXVECTOR4 bufferDiffuse[MaxLights];
    float bufferAmbient[MaxLights];
    float bufferPosition[3 * MaxLights];
    float bufferFalloffQuadratic[MaxLights], bufferFalloffLinear[MaxLights], bufferFalloffConstant[MaxLights];

    // The 8-slot constant arrays below are read only by the
    // engine-emulating shader paths (4-light / 8-light). When the
    // texture-light variant (useTextureLights) is active, the compiled
    // shader doesn't reference them, and the matching SetFloatArray
    // pushes near the bottom of this function are skipped — so the
    // zero-init is dead work too. Keep the buffers declared so the
    // skipped SetFloatArray block is the only conditional, but skip
    // the memsets to save ~50 ns/draw on the v3 path.
    if (!actuallyUseTextureLights) {
        memset(&bufferDiffuse, 0, sizeof(bufferDiffuse));
        memset(&bufferAmbient, 0, sizeof(bufferAmbient));
        memset(&bufferPosition, 0, sizeof(bufferPosition));
        memset(&bufferFalloffQuadratic, 0, sizeof(bufferFalloffQuadratic));
        memset(&bufferFalloffLinear, 0, sizeof(bufferFalloffLinear));
        for (size_t i = 0; i != MaxLights; ++i) bufferFalloffConstant[i] = kDefaultFalloffConstant;
    }

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
            // Phase 2 wire-in: skip engine-emit point lights when
            // msoc is feeding us the snapshot. Directional handling below
            // still runs (sun must come from the engine — it's set up via
            // MGEProxyDevice::SetLight intercept).
            if (useTextureLights) continue;

            memcpy(&bufferDiffuse[pointLightCount], &light->diffuse, sizeof(light->diffuse));

            // Scatter position vectors for vectorization
            bufferPosition[pointLightCount] = light->viewspacePos.x;
            bufferPosition[pointLightCount + MaxLights] = light->viewspacePos.y;
            bufferPosition[pointLightCount + 2*MaxLights] = light->viewspacePos.z;

            // Scatter attenuation factors for vectorization
            if (light->falloff.x > 0) {
                // Standard point light source
                bufferFalloffConstant[pointLightCount] = light->falloff.x;
                bufferFalloffLinear[pointLightCount] = light->falloff.y;
                bufferFalloffQuadratic[pointLightCount] = light->falloff.z;
            } else if (light->falloff.z > 0) {
                // Probably a magic light source patched by Morrowind Code Patch
                // Patched falloff calculation is quadratic only, which needs to be
                // modified to account for the standard falloffConstant
                // Diffuse colour is correctly specified with the patch
                // Some overbrightness is applied to diffuse to cause glowing
                bufferDiffuse[pointLightCount].x *= kDefaultFalloffConstant;
                bufferDiffuse[pointLightCount].y *= kDefaultFalloffConstant;
                bufferDiffuse[pointLightCount].z *= kDefaultFalloffConstant;
                bufferAmbient[pointLightCount] = 1.0f + 1e-4f / sqrt(light->falloff.z);
                bufferFalloffQuadratic[pointLightCount] = kDefaultFalloffConstant * light->falloff.z;
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
            ++pointLightCount;
        } else if (light->type == D3DLIGHT_DIRECTIONAL) {
            effectFFE->SetFloatArray(ehLightSunDirection, (const float*)&light->viewspacePos, 3);

            sunDiffuse = light->diffuse;
            ambient.r += light->ambient.x;
            ambient.g += light->ambient.y;
            ambient.b += light->ambient.z;
        }
    }

    // Texture-light parameter pushes. The cull and selection
    // already ran above (hoisted block) so candidateCount and idxFloats
    // are populated. Gated on actuallyUseTextureLights so demoted draws
    // (candidateCount == 0) skip the 4 pushes AND get the v1/v2 shader
    // variant instead of v3 — main payoff of the fast-path.
    if (actuallyUseTextureLights) {
        const float texelSize = 1.0f / (float)(kTexelsPerLight * kMaxTexLights);
        D3DXVECTOR4 lightDataParams_v(
            (float)candidateCount,  // .x = runtime loop count
            texelSize,              // .y = 1/textureWidth (texel U stride)
            0.0f,                   // .z = unused (we use indices, not slices)
            0.0f);                  // .w = unused

        if (ehTexLightData) effectFFE->SetTexture(ehTexLightData, texLightData);
        if (ehLightDataParams) effectFFE->SetVector(ehLightDataParams, &lightDataParams_v);
        if (ehLightIndices) effectFFE->SetVectorArray(ehLightIndices, (D3DXVECTOR4*)idxFloats, 8);
        if (ehTexLightView) effectFFE->SetMatrix(ehTexLightView, &rs->viewTransform);
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

    // Sun + scene-ambient are read by every shader variant (texture-light
    // path uses them too — sun direction and global ambient are mixed in
    // before the per-light loop in the pixel shader). Always push.
    effectFFE->SetFloatArray(ehLightSceneAmbient, ambient, 3);
    effectFFE->SetFloatArray(ehLightSunDiffuse, sunDiffuse, 3);
    // Engine 8-light constant arrays. Only read by the
    // engine-emulating shader paths (calcLighting4 / calcLighting8).
    // The texture-light variant ignores them — skipping these 6 D3DX9
    // parameter pushes saves ~600 ns/draw on the v3 path. The
    // SetFloatArray boundary itself is the cost (~100 ns each, includes
    // parameter handle resolution + constant-table write); the
    // compiled shader's eliminated-as-dead status doesn't make the
    // framework call free.
    if (!actuallyUseTextureLights) {
        effectFFE->SetVectorArray(ehLightDiffuse, bufferDiffuse, MaxLights);
        effectFFE->SetFloatArray(ehLightAmbient, bufferAmbient, MaxLights);
        effectFFE->SetFloatArray(ehLightPosition, bufferPosition, 3 * MaxLights);
        effectFFE->SetFloatArray(ehLightFalloffQuadratic, bufferFalloffQuadratic, MaxLights);
        effectFFE->SetFloatArray(ehLightFalloffLinear, bufferFalloffLinear, MaxLights);
        effectFFE->SetFloatArray(ehLightFalloffConstant, bufferFalloffConstant, MaxLights);
    }

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
    if (sk.usesCacheSkin) {
        // boneMatrices (model->world) + view; the VS skins via skinIndexed then
        // applies view. No rigid world/worldview needed.
        effectFFE->SetMatrixArray(ehBoneMatrices, cacheBonePalette, cacheNumBones);
        if (cacheView) effectFFE->SetMatrix(ehView, cacheView);
    } else if (rs->vertexBlendState) {
        effectFFE->SetMatrixArray(ehVertexBlendPalette, rs->worldViewTransforms, 4);
    } else {
        effectFFE->SetMatrix(ehWorld, &rs->worldTransforms[0]);
        effectFFE->SetMatrix(ehWorldView, &rs->worldViewTransforms[0]);
    }

    UINT passes;
    effectFFE->Begin(&passes, D3DXFX_DONOTSAVESTATE);
    effectFFE->BeginPass(0);
    DrawStats::count(rs->primCount);
    device->DrawIndexedPrimitive(rs->primType, rs->baseIndex, rs->minIndex, rs->vertCount, rs->startIndex, rs->primCount);
    effectFFE->EndPass();
    effectFFE->End();

    device->SetVertexShader(NULL);
    device->SetPixelShader(NULL);

    // Instrument: end QPC, attribute the elapsed time to the variant bucket
    // selected for this draw, and emit a periodic histogram + per-variant
    // averages. variantIdx mirrors the precache labelling: 0=no point lights,
    // 1=4 lights, 2=8 lights, 3=64 (msoc-emit). Entire block gated by
    // logPerf — when off, the per-draw QPC + accumulators + log all skip,
    // which is the bulk of the per-frame instrumentation overhead.
    if (logPerf) {
        LARGE_INTEGER tsEnd;
        QueryPerformanceCounter(&tsEnd);
        const unsigned long long deltaTicks =
            (unsigned long long)(tsEnd.QuadPart - tsBegin.QuadPart);
        const unsigned long long deltaNs =
            (s_qpcFreq.QuadPart > 0)
                ? (deltaTicks * 1000000000ULL / (unsigned long long)s_qpcFreq.QuadPart)
                : 0ULL;
        unsigned int variantIdx;
        if (sk.vertexMaterial == 0)        variantIdx = 0;
        else if (sk.useTextureLightVariant)   variantIdx = 3;
        else if (sk.heavyLighting)         variantIdx = 2;
        else                               variantIdx = 1;
        ++s_variantCalls[variantIdx];
        s_variantTotalNs[variantIdx] += deltaNs;

        if (s_callCount - s_lastReportCall >= 100000) {
            LOG::logline(
                "-- [LIGHTS-INSTR] calls=%llu peak=%zu peakMsoc=%zu engine[0-4|5-7|8|9-12|13-16|17+]=%llu|%llu|%llu|%llu|%llu|%llu msoc[0|1-8|9-16|17-32|33-48|49-64|65-128|129+]=%llu|%llu|%llu|%llu|%llu|%llu|%llu|%llu",
                s_callCount, s_peakLights, s_peakMsoc,
                s_engineBuckets[0], s_engineBuckets[1], s_engineBuckets[2],
                s_engineBuckets[3], s_engineBuckets[4], s_engineBuckets[5],
                s_msocBuckets[0], s_msocBuckets[1], s_msocBuckets[2], s_msocBuckets[3],
                s_msocBuckets[4], s_msocBuckets[5], s_msocBuckets[6], s_msocBuckets[7]);
            LOG::logline(
                "-- [VARIANT-TIMING] v0(none)=%llu/%lluns(%.0f) v1(4)=%llu/%lluns(%.0f) v2(8)=%llu/%lluns(%.0f) v3(64)=%llu/%lluns(%.0f)",
                s_variantCalls[0], s_variantTotalNs[0],
                s_variantCalls[0] ? (double)s_variantTotalNs[0] / s_variantCalls[0] : 0.0,
                s_variantCalls[1], s_variantTotalNs[1],
                s_variantCalls[1] ? (double)s_variantTotalNs[1] / s_variantCalls[1] : 0.0,
                s_variantCalls[2], s_variantTotalNs[2],
                s_variantCalls[2] ? (double)s_variantTotalNs[2] / s_variantCalls[2] : 0.0,
                s_variantCalls[3], s_variantTotalNs[3],
                s_variantCalls[3] ? (double)s_variantTotalNs[3] / s_variantCalls[3] : 0.0);
            LOG::logline(
                "-- [SELECTION-TIMING] calls=%llu totalNs=%llu avgNs=%.0f",
                s_selectionCalls, s_selectionTotalNs,
                s_selectionCalls ? (double)s_selectionTotalNs / s_selectionCalls : 0.0);
            // Texture-upload cost (LockRect + memset + pack + frustum precull).
            // uploads/calls ratio = how often we actually touch the texture
            // (lower with the SceneGraph change-detect; vanishes for static scenes).
            const double uploadAvg = s_uploads ? (double)s_uploadsTotalNs / s_uploads : 0.0;
            const double uploadFrac = s_callCount ? 100.0 * (double)s_uploads / s_callCount : 0.0;
            LOG::logline(
                "-- [UPLOAD-TIMING] uploads=%llu (%.2f%% of draws) totalNs=%llu avgNs=%.0f",
                s_uploads, uploadFrac, s_uploadsTotalNs, uploadAvg);
            // Frustum precull alive ratio. High = many lights survived the
            // 2*radius frustum check (poor cull, big candidate set per mesh).
            // Low = camera barely sees any lights (dense interior with most
            // lights behind walls / off-camera).
            const double aliveRatio = s_preCullTotalSum
                ? 100.0 * (double)s_preCullAliveSum / s_preCullTotalSum : 0.0;
            LOG::logline(
                "-- [PRECULL] aliveRatio=%.0f%% (%llu alive / %llu total across uploads)",
                aliveRatio, s_preCullAliveSum, s_preCullTotalSum);
            // Per-mesh selected-count histogram. If the rightmost bucket
            // (capped at 32) is > 0, kMaxIndicesPerMesh is binding for
            // some meshes — bumping it would let more lights through.
            // If most draws sit in [0, 1-4, 5-8], the cap is fine.
            LOG::logline(
                "-- [SELECTED-DIST] [0|1-4|5-8|9-16|17-31|=32]=%llu|%llu|%llu|%llu|%llu|%llu",
                s_selectedBuckets[0], s_selectedBuckets[1], s_selectedBuckets[2],
                s_selectedBuckets[3], s_selectedBuckets[4], s_selectedBuckets[5]);
            // Per-frame averages. SceneGraph::frameCount() ticks once per
            // bridge call, so cumulative-counter / frameCount = per-frame.
            // Skips emission if frame count is zero (bridge hasn't fired)
            // or 1 (avoids meaningless single-frame divides).
            const unsigned long long frames = MGE::SceneGraph::frameCount();
            if (frames > 1) {
                const double v3PerFrame  = (double)s_variantCalls[3] / frames;
                const double v3NsFrame   = (double)s_variantTotalNs[3] / frames;
                const double selNsFrame  = (double)s_selectionTotalNs / frames;
                const double upNsFrame   = (double)s_uploadsTotalNs   / frames;
                const double upPerFrame  = (double)s_uploads          / frames;
                LOG::logline(
                    "-- [PER-FRAME] frames=%llu v3draws=%.1f v3ns=%.0f selns=%.0f upns=%.0f uploads=%.2f",
                    frames, v3PerFrame, v3NsFrame, selNsFrame, upNsFrame, upPerFrame);
            }
            s_lastReportCall = s_callCount;
        }
    }
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
    if (sk.usesCacheSkin) {
        buf << "float4 blendweights : BLENDWEIGHT; float4 blendindices : BLENDINDICES; ";
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
    } else if (sk.usesCacheSkin) {
        buf << "viewpos = cacheSkinnedVertex(IN.pos, IN.blendweights, IN.blendindices); normal = cacheSkinnedNormal(IN.nrm, IN.blendweights, IN.blendindices);";
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
    // msoc-emit path: USE_TEXTURE_LIGHTS macro switches the
    // shader to read lights from a 1D texture (3 texels per light) with
    // a runtime loop count. Non-msoc draws still pick the 4/8 buckets
    // via heavyLighting below. The FFE_LIGHTS_ACTIVE value is moot when
    // USE_TEXTURE_LIGHTS is defined (the constant-array path is #ifdef'd
    // out), but pick "0" so the unused per-vertex/per-light loops fold
    // away cleanly during compile.
    if (sk.vertexMaterial == 0) {
        genLightCount = "0";
    } else if (sk.useTextureLightVariant) {
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
    // Phase 2: USE_TEXTURE_LIGHTS macro present (with value "1")
    // only for the msoc-emit variant. For all other variants we omit it
    // so the shader's #ifdef branch falls through to the constant-array
    // path. Using #ifdef rather than #if 0/1 because that's the convention
    // already used in this file (FFE_ERROR_MATERIAL, VERIFY).
    const char* useTextureLightsValue = sk.useTextureLightVariant ? "1" : nullptr;
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
        // Conditional macro — appears with value "1" iff the texture-
        // light variant is being generated. Pre-terminator entry; the
        // real terminator is the {0, 0} below.
        useTextureLightsValue ? "USE_TEXTURE_LIGHTS" : nullptr, useTextureLightsValue,
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

void FixedFunctionShader::release() {
    for (auto& i : cacheEffects) {
        if (i.second) {
            i.second->Release();
        }
    }

    shaderLRU.effect = nullptr;
    shaderLRU.last_sk = ShaderKey();
    cacheEffects.clear();
    effectDefaultPurple->Release();
    if (texLightData) {
        texLightData->Release();
        texLightData = nullptr;
    }
    lastUploadedRevision = (unsigned int)-1;
    lastUploadedPointCount = 0;
    bboxCache.clear();
}

// Vertex-buffer-derived per-mesh world-space AABB. Ported from
// vtastek's hiz_culling.cpp::computeBoundingBox with simplifications
// (no debug-key gating, no recording-thread coordination — we run on
// the render thread). Cache hit: 8-corner transform by worldTransforms[0]
// (~50 ALU). Cache miss: lock VB (READONLY+DONOTWAIT), walk verts/
// indices, find object-space min/max, cache, then transform.
//
// Lock failures fall through gracefully — we return false and the
// caller falls back to scene-wide flat (treats every msoc light as
// affecting this draw).
bool FixedFunctionShader::computeBoundingBox(const RenderedState* rs,
    D3DXVECTOR3& outMin, D3DXVECTOR3& outMax)
{
    if (!rs->vb) return false;
    if ((rs->fvf & D3DFVF_POSITION_MASK) == 0) return false;

    MeshKey key{ rs->vb, rs->ib, rs->fvf,
                 rs->baseIndex, rs->vertCount, rs->startIndex, rs->primCount };

    auto transformBoxToWorld = [&](const ObjectSpaceBBox& b) {
        const D3DXVECTOR3 corners[8] = {
            { b.minX, b.minY, b.minZ }, { b.maxX, b.minY, b.minZ },
            { b.minX, b.maxY, b.minZ }, { b.maxX, b.maxY, b.minZ },
            { b.minX, b.minY, b.maxZ }, { b.maxX, b.minY, b.maxZ },
            { b.minX, b.maxY, b.maxZ }, { b.maxX, b.maxY, b.maxZ },
        };
        outMin = D3DXVECTOR3( 1e30f,  1e30f,  1e30f);
        outMax = D3DXVECTOR3(-1e30f, -1e30f, -1e30f);
        for (int i = 0; i != 8; ++i) {
            D3DXVECTOR3 wc;
            D3DXVec3TransformCoord(&wc, &corners[i], &rs->worldTransforms[0]);
            outMin.x = std::min(outMin.x, wc.x);
            outMin.y = std::min(outMin.y, wc.y);
            outMin.z = std::min(outMin.z, wc.z);
            outMax.x = std::max(outMax.x, wc.x);
            outMax.y = std::max(outMax.y, wc.y);
            outMax.z = std::max(outMax.z, wc.z);
        }
    };

    auto it = bboxCache.find(key);
    if (it != bboxCache.end()) {
        transformBoxToWorld(it->second);
        return true;
    }

    // Cache miss — lock VB and (if indexed) IB, find min/max.
    UINT stride = rs->vbStride;
    if (stride == 0) return false;

    void* pVerts = nullptr;
    if (FAILED(rs->vb->Lock(rs->vbOffset, 0, &pVerts,
                            D3DLOCK_READONLY | D3DLOCK_NOSYSLOCK))) {
        return false;
    }

    ObjectSpaceBBox bb;
    bb.minX = bb.minY = bb.minZ =  1e30f;
    bb.maxX = bb.maxY = bb.maxZ = -1e30f;

    if (rs->ib) {
        void* pIdx = nullptr;
        if (FAILED(rs->ib->Lock(0, 0, &pIdx,
                                D3DLOCK_READONLY | D3DLOCK_NOSYSLOCK))) {
            rs->vb->Unlock();
            return false;
        }
        D3DINDEXBUFFER_DESC ibd;
        rs->ib->GetDesc(&ibd);
        const bool is16 = (ibd.Format == D3DFMT_INDEX16);

        UINT idxCount = 0;
        switch (rs->primType) {
        case D3DPT_TRIANGLELIST:  idxCount = rs->primCount * 3; break;
        case D3DPT_TRIANGLESTRIP: idxCount = rs->primCount + 2; break;
        case D3DPT_TRIANGLEFAN:   idxCount = rs->primCount + 2; break;
        default:
            rs->ib->Unlock();
            rs->vb->Unlock();
            return false;
        }

        for (UINT i = 0; i != idxCount; ++i) {
            UINT vi;
            if (is16) vi = ((WORD*)pIdx)[rs->startIndex + i] + rs->baseIndex;
            else      vi = ((DWORD*)pIdx)[rs->startIndex + i] + rs->baseIndex;
            const float* p = (const float*)((BYTE*)pVerts + vi * stride);
            bb.minX = std::min(bb.minX, p[0]);
            bb.minY = std::min(bb.minY, p[1]);
            bb.minZ = std::min(bb.minZ, p[2]);
            bb.maxX = std::max(bb.maxX, p[0]);
            bb.maxY = std::max(bb.maxY, p[1]);
            bb.maxZ = std::max(bb.maxZ, p[2]);
        }
        rs->ib->Unlock();
    } else {
        for (UINT i = 0; i != rs->vertCount; ++i) {
            const float* p = (const float*)((BYTE*)pVerts + (rs->baseIndex + i) * stride);
            bb.minX = std::min(bb.minX, p[0]);
            bb.minY = std::min(bb.minY, p[1]);
            bb.minZ = std::min(bb.minZ, p[2]);
            bb.maxX = std::max(bb.maxX, p[0]);
            bb.maxY = std::max(bb.maxY, p[1]);
            bb.maxZ = std::max(bb.maxZ, p[2]);
        }
    }
    rs->vb->Unlock();

    if (bb.minX > bb.maxX) return false;  // never wrote (zero verts)
    bboxCache[key] = bb;
    transformBoxToWorld(bb);
    return true;
}



// ShaderKey - Captures a generatable shader configuration

FixedFunctionShader::ShaderKey::ShaderKey(const RenderedState* rs, const FragmentState* frs, const LightState* lightrs) {
    memset(this, 0, sizeof(ShaderKey));         // Clear padding bits for compares

    uvSets = (rs->fvf & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT;
    usesSkinning = rs->vertexBlendState ? 1 : 0;
    vertexColour = (rs->fvf & D3DFVF_DIFFUSE) ? 1 : 0;

    // Match constant material, diffuse+ambient vcol, or emissive vcol
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

    LOG::logline("   Input state: UVs:%d skin:%d vcol:%d lights:%d vmat:%d fogm:%d", uvSets, usesSkinning, vertexColour, vertexMaterial ? (useTextureLightVariant ? 64 : (heavyLighting ? 8 : 4)) : 0, vertexMaterial, fogMode);
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
