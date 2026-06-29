#include "mge_se_prelude.h"

#include "NIAVObject.h"
#include "NIGeometry.h"
#include "NIGeometryData.h"
#include "NINode.h"
#include "NISwitchNode.h"
#include "NIProperty.h"
#include "NIRTTIDefines.h"
#include "NISourceTexture.h"
#include "NIDX8TextureData.h"
#include "NITriBasedGeometry.h"
#include "NITriBasedGeometryData.h"
#include "NISkinInstance.h"

#include "configuration.h"
#include "datahandler_view.h"
#include "mge_tracy.h"
#include "proxydx/d3d8texture.h"
#include "proxydx/devicelock.h"
#include "scenegraph_geometry_cache.h"
#include "renderprocess.h"
#include "ipc/geomwire.h"
#include "support/log.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace MGE::GeometryCache {

    namespace {

        IDirect3DDevice9* g_device          = nullptr;
        IDirect3DVertexDeclaration9* g_skinnedDecl = nullptr;
        uint64_t          g_frame          = 0;
        // Set while walking the landscape (terrain) root so visitGeometry can tag
        // entries (CachedGeometry::isLandscape). Terrain is excluded from the
        // cache color pass — it stays on the distant-land path.
        bool              g_walkingLandscape = false;
        // Set while walking worldPickObjectRoot so visitGeometry can tag entries
        // (CachedGeometry::isPickRoot) — that root is outside the engine's world-camera
        // occlusion classify; the Stage 2 engine-set cull keeps these via frustum.
        bool              g_walkingPick = false;
        // Set while walking skyRoot (SK1 sky takeover) so visitGeometry tags entries
        // (CachedGeometry::isSky) and forces a per-frame re-upload — the dome's vertex-colour
        // gradient changes every frame (sun angle / weather) without bumping revisionID.
        bool              g_walkingSky = false;
        // SK2: monotonic counter assigned to each isSky entry's skyOrder during the skyRoot
        // walk (reset to 0 at the start of each sky walk). Encodes back-to-front subtree order
        // so the Forge sky pass can sort its alpha-blended draws (dome → stars → sun → moons).
        uint16_t          g_skyVisitCounter = 0;

        // NiAlphaProperty blend-function index (Gamebryo order) -> D3DBLEND. Defined with the
        // moon support below; forward-declared so extractMaterial can translate sky blend modes.
        D3DBLEND niBlendToD3D(unsigned int ni);
        uint32_t          g_uploadedThisFrame  = 0;
        uint64_t          g_uploadedInterval   = 0; // cumulative over log interval
        // Phase 0 diagnostic: null-bone influence accounting (the suspected NPC
        // "explosion" source — a null SkinInstance::bones[b] currently falls back
        // to identity, parking influenced verts at the model/cell origin).
        uint32_t          g_nullBoneHitsInterval  = 0; // total null bones seen / interval
        uint32_t          g_nullBonePartsInterval = 0; // distinct parts with >=1 null bone
        const char*       g_nullBoneSampleTex     = nullptr; // a sample part's texture

        std::unordered_map<uint32_t, CachedGeometry>      g_cache;
        // Reverse map GPU texture -> SourceTexture::fileName, for resolveTextureName().
        // DORMANT: nothing populates this currently. The per-frame rebuild (walking
        // NI property state per node) cost ~1.5ms and had no consumer, so it was
        // removed. When a feature needs GPU-texture -> source-path resolution, populate
        // incrementally from extractMaterial (create/material-change only), not per frame.
        std::unordered_map<IDirect3DTexture9*, const char*> g_textureNameMap;

        void releaseEntry(CachedGeometry& e) {
            if (e.vb[0]) { e.vb[0]->Release(); e.vb[0] = nullptr; }
            if (e.vb[1]) { e.vb[1]->Release(); e.vb[1] = nullptr; }
            if (e.ib)    { e.ib->Release();    e.ib    = nullptr; }
        }

        IDirect3DTexture9* getDX9Texture(NI::Texture* tex) {
            if (!tex || !tex->rendererData) return nullptr;
            auto* srd = static_cast<NI::DX8SourceTextureData*>(
                static_cast<void*>(tex->rendererData));
            if (!srd->d3dTexture) return nullptr;
            return static_cast<ProxyTexture*>(srd->d3dTexture)->realTexture;
        }

        void extractMaterial(CachedGeometry& e, NI::Geometry* geom) {
            e.d3dTexture  = nullptr;
            e.d3dOverlay  = nullptr;
            e.d3dDark     = nullptr;
            e.d3dDetail   = nullptr;
            e.d3dGlow     = nullptr;
            e.baseUV = e.darkUV = e.detailUV = e.glowUV = 0;
            e.textureName = nullptr;
            e.overlayTextureName = nullptr;
            e.darkTextureName = e.detailTextureName = e.glowTextureName = nullptr;
            e.alphaRef    = 0.0f;
            e.alphaTest   = false;
            e.blendEnable = false;
            // SK1 sky: default to the standard transparency blend; overwritten below from the
            // NiAlphaProperty flags when present. Only consumed for isSky entries (the Forge
            // sky pass); opaque/alpha-test draws ignore these.
            e.srcBlend    = static_cast<unsigned char>(D3DBLEND_SRCALPHA);
            e.destBlend   = static_cast<unsigned char>(D3DBLEND_INVSRCALPHA);
            // Default material: white diffuse/ambient, no emissive (texture-only
            // opaque). Overwritten below when the shape carries a MaterialProperty.
            e.matDiffuse[0]  = e.matDiffuse[1]  = e.matDiffuse[2]  = e.matDiffuse[3]  = 1.0f;
            e.matAmbient[0]  = e.matAmbient[1]  = e.matAmbient[2]  = e.matAmbient[3]  = 1.0f;
            e.matEmissive[0] = e.matEmissive[1] = e.matEmissive[2] = e.matEmissive[3] = 0.0f;
            e.vColSource = 0;   // SOURCE_IGNORE until a VertexColorProperty says otherwise

            auto* ps = reinterpret_cast<NI::PropertyState*>(geom->propertyState);
            if (!ps) return;

            if (ps->vertexColor) {
                e.vColSource = static_cast<uint8_t>(ps->vertexColor->source);  // 0 ignore, 1 emissive, 2 amb+diff
            }

            if (ps->material) {
                const auto* mp = ps->material;
                e.matDiffuse[0]  = mp->diffuse.r;  e.matDiffuse[1]  = mp->diffuse.g;
                e.matDiffuse[2]  = mp->diffuse.b;  e.matDiffuse[3]  = mp->alpha;
                e.matAmbient[0]  = mp->ambient.r;  e.matAmbient[1]  = mp->ambient.g;
                e.matAmbient[2]  = mp->ambient.b;  e.matAmbient[3]  = 1.0f;
                e.matEmissive[0] = mp->emissive.r; e.matEmissive[1] = mp->emissive.g;
                e.matEmissive[2] = mp->emissive.b; e.matEmissive[3] = 0.0f;
            }

            if (ps->alpha) {
                const auto* ap = ps->alpha;
                e.alphaTest   = (ap->flags & NI::AlphaProperty::TEST_ENABLE_MASK) != 0;
                e.blendEnable = (ap->flags & NI::AlphaProperty::ALPHA_MASK) != 0;
                e.alphaRef    = ap->alphaTestRef / 255.0f;
                // Sky blend factors (same translation the moon path uses).
                e.srcBlend  = static_cast<unsigned char>(niBlendToD3D(
                    (ap->flags & NI::AlphaProperty::SRC_BLEND_MASK)  >> NI::AlphaProperty::SRC_BLEND_POS));
                e.destBlend = static_cast<unsigned char>(niBlendToD3D(
                    (ap->flags & NI::AlphaProperty::DEST_BLEND_MASK) >> NI::AlphaProperty::DEST_BLEND_POS));
            }

            if (ps->texture) {
                const auto* baseMap = ps->texture->getBaseMap();
                if (baseMap && baseMap->texture) {
                    auto* tex = baseMap->texture.get();
                    if (tex->isInstanceOfType(NI::RTTIStaticPtr::NiSourceTexture)) {
                        auto* st = static_cast<NI::SourceTexture*>(tex);
                        e.textureName = st->fileName;
                        e.d3dTexture  = getDX9Texture(tex);
                        e.baseUV = baseMap->texCoordSet >= 3u ? 3u : static_cast<uint8_t>(baseMap->texCoordSet);
                    }
                }
                // Multi-map siblings (dark/detail/glow) on the same property — the
                // PPL fixed-function blend the cache color pass reconstructs. Store
                // the D3D9 texture and the UV set it samples (its true texCoordSet,
                // clamped to 0..3). uploadEntry sizes the VB so every used set is
                // carried (e.g. the glow-mod detail map on set 2).
                auto captureMap = [&](NI::TexturingProperty::Map* map,
                                      IDirect3DTexture9*& outTex, uint8_t& outUV,
                                      const char*& outName) {
                    if (!map || !map->texture) return;
                    auto* mtex = map->texture.get();
                    if (!mtex->isInstanceOfType(NI::RTTIStaticPtr::NiSourceTexture)) return;
                    IDirect3DTexture9* d3d = getDX9Texture(mtex);
                    if (!d3d) return;
                    outTex = d3d;
                    // The map's texture is a confirmed NiSourceTexture — also record its
                    // source filename so the Forge path can resolve it to a bindless slot.
                    outName = static_cast<NI::SourceTexture*>(mtex)->fileName;
                    // Store the map's TRUE UV set (clamped to 3 — FFE texcoordIndex is
                    // 2-bit / FVF carries <=4 sets). uploadEntry sizes the VB to cover it.
                    outUV  = map->texCoordSet >= 3u ? 3u : static_cast<uint8_t>(map->texCoordSet);
                };
                captureMap(ps->texture->getDarkMap(),   e.d3dDark,   e.darkUV,   e.darkTextureName);
                captureMap(ps->texture->getDetailMap(), e.d3dDetail, e.detailUV, e.detailTextureName);
                captureMap(ps->texture->getGlowMap(),   e.d3dGlow,   e.glowUV,   e.glowTextureName);

                // Terrain decal overlay: maps[6] = DECAL_1 (the second land texture
                // for splat blending). Present on multi-texture terrain patches.
                if (ps->texture->maps.getEndIndex() > 6u) {
                    const auto* decalMap = ps->texture->maps.at(6);
                    if (decalMap && decalMap->texture) {
                        auto* dtex = decalMap->texture.get();
                        if (dtex->isInstanceOfType(NI::RTTIStaticPtr::NiSourceTexture)) {
                            e.d3dOverlay = getDX9Texture(dtex);
                            // Record the overlay source filename so the Forge path can
                            // resolve it to a bindless slot (same cast captureMap uses).
                            e.overlayTextureName = static_cast<NI::SourceTexture*>(dtex)->fileName;
                        }
                    }
                }
            }
        }

        // Vertex layout matching MorrowindVertIn (depth/shadow VS input).
        // D3DFVF_XYZ | D3DFVF_NORMAL | D3DFVF_DIFFUSE | D3DFVF_TEX1, stride 36.
        struct DepthVertex {
            float x, y, z;    // POSITION  (12)
            float nx, ny, nz; // NORMAL    (12, model-space; depth/shadow ignore, color pass lights)
            DWORD color;       // DIFFUSE   ( 4, 0xFFFFFFFF — hasVCol=false, unused)
            float u, v;        // TEXCOORD0 ( 8, from UV set 0)
        };
        static_assert(sizeof(DepthVertex) == 36, "DepthVertex size mismatch");
        // Multi-map shapes append extra UV sets after the TEXCOORD0 of DepthVertex
        // (stride = kVBStridePos + 8*uvSetCount). uploadEntry writes them with a
        // generic byte-offset writer rather than a fixed struct, so 1..4 UV sets
        // share one code path; the depth/shadow VS read only TEXCOORD0.

        // Skinned vertex layout matching SkinnedVertIn (VS palette skinning input).
        // Drawn with g_skinnedDecl; stride 56.
        struct SkinnedVertex {
            float x, y, z;          // POSITION     (12) bind-pose
            float nx, ny, nz;       // NORMAL       (12) bind-pose model-space
            float w0, w1, w2, w3;   // BLENDWEIGHT  (16) top-4 influences, normalized
            DWORD indices;          // BLENDINDICES ( 4) UBYTE4 bone palette indices
            float u, v;             // TEXCOORD0    ( 8)
            DWORD color;            // COLOR        ( 4) Phase 2: per-vertex colour
        };
        static_assert(sizeof(SkinnedVertex) == 56, "SkinnedVertex size mismatch");

        void uploadEntry(CachedGeometry& e, NI::TriBasedGeometry* geom,
                         NI::TriBasedGeometryData* data, uint32_t key) {
            const auto vertexCount = static_cast<uint32_t>(data->getActiveVertexCount());
            const auto triCount    = static_cast<uint32_t>(data->getActiveTriangleCount());

            const auto* mv = data->vertex;          // model-space, always present

            // Invalid geometry — drop any stale buffers so the entry is skipped.
            if (!vertexCount || !triCount || !mv) {
                releaseEntry(e);
                return;
            }

            // UV-set count: carry as many sets as the maps actually use (the glow-mod
            // windows put base/dark on sets 0/1 and the detail map on set 2). The maps'
            // texCoordSets were captured in extractMaterial (which runs first). Bound by
            // the mesh's own set count and 4 (FFE texcoordIndex is 2-bit). Per-set blocks
            // are contiguous in textureCoords — set s starts at +s*storedVerts, sized by
            // the data's stored vertexCount. Single-UV geometry resolves to count 1.
            // Landscape excluded: terrain splats via d3dOverlay and its passes bind the
            // single-UV stride unconditionally.
            const uint16_t storedVerts = data->vertexCount;
            uint8_t maxMapUV = e.baseUV;
            if (e.d3dDark)   maxMapUV = std::max(maxMapUV, e.darkUV);
            if (e.d3dDetail) maxMapUV = std::max(maxMapUV, e.detailUV);
            if (e.d3dGlow)   maxMapUV = std::max(maxMapUV, e.glowUV);
            const uint8_t availSets = data->textureCoords
                ? static_cast<uint8_t>(std::min<unsigned>(data->textureSets, 4u)) : 1u;
            uint8_t uvSetCount = std::min<uint8_t>(static_cast<uint8_t>(maxMapUV + 1), availSets);
            if (uvSetCount < 1 || g_walkingLandscape) uvSetCount = 1;
            const unsigned int stride = MGE::GeometryCache::kVBStridePos + 8u * uvSetCount;
            const DWORD vbFVF = MGE::GeometryCache::kVBFVFBase
                              | (static_cast<DWORD>(uvSetCount) << D3DFVF_TEXCOUNT_SHIFT);

            // On a size OR UV-layout change, drop both slots + IB so they repopulate
            // at the new size/stride.
            const bool sizeChanged = (e.vertexCount != vertexCount)
                || (e.triangleCount != triCount) || (e.uvSetCount != uvSetCount);
            if (sizeChanged) {
                releaseEntry(e);
            }

            const uint8_t slot = 1u - e.writeSlot;

            // Model-space VB (XYZ|NORMAL|DIFFUSE|TEXn); per-draw world*view palette.
            if (!e.vb[slot]) {
                HRESULT hr = g_device->CreateVertexBuffer(
                    vertexCount * stride,
                    D3DUSAGE_WRITEONLY,
                    vbFVF,
                    g_spikeForceDefaultPool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &e.vb[slot], nullptr);
                if (FAILED(hr)) { e.vb[slot] = nullptr; return; }
            }

            const bool createdIB = (e.ib == nullptr);
            if (createdIB) {
                g_device->CreateIndexBuffer(
                    triCount * 6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16,
                    g_spikeForceDefaultPool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &e.ib, nullptr);
            }

            void* vbData = nullptr;
            if (SUCCEEDED(e.vb[slot]->Lock(0, 0, &vbData, 0))) {
                const auto* uvs = data->textureCoords;  // NI::Point2*, set-major, nullptr if no UVs
                const auto* nrm = data->normal;         // NI::Point3*, nullptr if no normals
                const auto* vcol = data->color;         // NI::PackedColor*(b,g,r,a)=D3DCOLOR, null if none
                // Generic writer: DepthVertex prefix (pos+normal+color+UV0) then 8 bytes
                // per additional UV set. Set s for vertex i lives at uvs[s*storedVerts+i]
                // (set-major, confirmed by the [MULTIMAP] raw dump). Walk by byte offset
                // so 1..4 UV sets share one path.
                auto* base = static_cast<uint8_t*>(vbData);
                // Tight model-space AABB over the verts (for world-AABB light
                // selection that matches the reactive computeBoundingBox).
                float mn[3] = { mv[0].x, mv[0].y, mv[0].z };
                float mx[3] = { mv[0].x, mv[0].y, mv[0].z };
                for (uint32_t i = 0; i < vertexCount; ++i) {
                    auto* v = reinterpret_cast<DepthVertex*>(base + i * stride);
                    v->x = mv[i].x; v->y = mv[i].y; v->z = mv[i].z;
                    if (mv[i].x < mn[0]) mn[0] = mv[i].x; if (mv[i].x > mx[0]) mx[0] = mv[i].x;
                    if (mv[i].y < mn[1]) mn[1] = mv[i].y; if (mv[i].y > mx[1]) mx[1] = mv[i].y;
                    if (mv[i].z < mn[2]) mn[2] = mv[i].z; if (mv[i].z > mx[2]) mx[2] = mv[i].z;
                    // Model-space normals; lit by renderMorrowind in the cache color pass
                    // (Phase 0.5). Depth/shadow ignore them. Up if the mesh has none.
                    if (nrm) { v->nx = nrm[i].x; v->ny = nrm[i].y; v->nz = nrm[i].z; }
                    else     { v->nx = 0.0f; v->ny = 0.0f; v->nz = 1.0f; }
                    // PackedColor byte order (b,g,r,a) is exactly D3DCOLOR, copy straight.
                    v->color = vcol ? *reinterpret_cast<const DWORD*>(&vcol[i]) : 0xFFFFFFFF;
                    v->u = uvs ? uvs[i].x : 0.0f;          // UV set 0
                    v->v = uvs ? uvs[i].y : 0.0f;
                    // Extra UV sets 1..uvSetCount-1, appended after TEXCOORD0.
                    auto* extra = reinterpret_cast<float*>(base + i * stride + 36);
                    for (uint8_t s = 1; s < uvSetCount; ++s) {
                        const auto& p = uvs[s * storedVerts + i];
                        *extra++ = p.x;
                        *extra++ = p.y;
                    }
                }
                e.aabbMin[0] = mn[0]; e.aabbMin[1] = mn[1]; e.aabbMin[2] = mn[2];
                e.aabbMax[0] = mx[0]; e.aabbMax[1] = mx[1]; e.aabbMax[2] = mx[2];
                e.vb[slot]->Unlock();
            }

            // Non-skinned topology may change on a revision bump, so rewrite the IB
            // whenever uploadEntry runs (rare — only first upload or revision change).
            if (e.ib) {
                const auto* triList = data->getTriList();
                if (triList) {
                    void* ibData = nullptr;
                    if (SUCCEEDED(e.ib->Lock(0, 0, &ibData, 0))) {
                        memcpy(ibData, triList, triCount * 6);
                        e.ib->Unlock();
                    }
                }
            }

            e.writeSlot          = slot;
            e.vertexCount        = vertexCount;
            e.triangleCount      = triCount;
            e.revisionID         = data->revisionID;
            e.isSkinned          = false;
            e.numBones           = 0;
            e.skinnedUnsupported = false;
            e.uvSetCount         = uvSetCount;
            e.vbStride           = static_cast<uint16_t>(stride);
            e.vbFVF              = vbFVF;
            e.hasVertexColor     = (data->color != nullptr);

            const auto& b = data->bounds;
            e.boundsCenter[0] = b.center.x;
            e.boundsCenter[1] = b.center.y;
            e.boundsCenter[2] = b.center.z;
            e.boundsRadius    = b.radius;

            // M1: ship model-space pos+normal+indices to the Forge host (non-skinned
            // opaques AND near terrain — worldLandscapeRoot patches carry model-space
            // vertex/normal/triList just like objects, and buildD3DTransform already set
            // worldTransformD3D for them above). Flat-shaded for now (texturing is the
            // next milestone, shared by objects+terrain). Re-uploads only on revision change.
            if (RenderProcess::wantsGeometryCapture()) {
                const auto* nrm = data->normal;
                const auto* capUvs = data->textureCoords;
                // Tier 2a lighting: ship the real per-vertex colour ONLY when the mesh uses
                // VertexColorProperty source 2 (ambient+diffuse / DiffAmb) — the case where MW
                // actually folds vcol into lighting. Otherwise ship white (0xFFFFFFFF) so the
                // host's universal col*(d+a) path reduces to the white-material (d+a) case.
                // (Emissive routing / non-white material constants are Tier 2b.)
                const auto* vcol = (e.hasVertexColor && e.vColSource == 2) ? data->color : nullptr;
                const auto* triList = data->getTriList();

                // Tier 4 multi-map: a part with dark/detail/glow siblings rides the SEPARATE
                // wide vertex format (GeomVertexWireMM, 4 UV sets) + its own host pipeline.
                // Single-map parts (the 99% case) keep the lean GeomVertexWire path. Landscape
                // is excluded (uvSetCount forced to 1 above; terrain splats via d3dOverlay).
                const bool isMultiMap = !g_walkingLandscape
                    && (e.d3dDark || e.d3dDetail || e.d3dGlow);

                if (isMultiMap && triList) {
                    static std::vector<IPC::GeomVertexWireMM> mmScratch;  // single-threaded cache walk
                    mmScratch.resize(vertexCount);
                    for (uint32_t i = 0; i < vertexCount; ++i) {
                        auto& w = mmScratch[i];
                        w.px = mv[i].x; w.py = mv[i].y; w.pz = mv[i].z;
                        if (nrm) { w.nx = nrm[i].x; w.ny = nrm[i].y; w.nz = nrm[i].z; }
                        else     { w.nx = 0.0f;    w.ny = 0.0f;    w.nz = 1.0f; }
                        w.color = vcol ? *reinterpret_cast<const DWORD*>(&vcol[i]) : 0xFFFFFFFFu;
                        // UV sets 0..3, read set-major (uvs[set*storedVerts + i]). Sets the VB
                        // doesn't carry (>= uvSetCount) duplicate set 0 — those stages are
                        // dropped client-side (cacheMapActive: uv < uvSetCount) so never read.
                        for (uint8_t s = 0; s < 4; ++s) {
                            const uint8_t src = (capUvs && s < uvSetCount) ? s : 0u;
                            if (capUvs) {
                                const auto& p = capUvs[(uint32_t)src * storedVerts + i];
                                w.uv[s][0] = p.x; w.uv[s][1] = p.y;
                            } else {
                                w.uv[s][0] = 0.0f; w.uv[s][1] = 0.0f;
                            }
                        }
                    }
                    RenderProcess::captureMultiMapGeometry(key, data->revisionID,
                        reinterpret_cast<uint32_t>(data),   // object identity (recycled-key guard)
                        mmScratch.data(), vertexCount,
                        reinterpret_cast<const uint16_t*>(triList), triCount * 3u);
                } else {
                    static std::vector<IPC::GeomVertexWire> scratch;  // single-threaded cache walk
                    scratch.resize(vertexCount);
                    // Base-map UV: set e.baseUV (set-major, uvs[set*storedVerts + i]). Most static
                    // meshes use set 0; honour the captured base map's true set for correctness.
                    const uint32_t uvBase = (uint32_t)e.baseUV * storedVerts;
                    for (uint32_t i = 0; i < vertexCount; ++i) {
                        auto& w = scratch[i];
                        w.px = mv[i].x; w.py = mv[i].y; w.pz = mv[i].z;
                        if (nrm) { w.nx = nrm[i].x; w.ny = nrm[i].y; w.nz = nrm[i].z; }
                        else     { w.nx = 0.0f;    w.ny = 0.0f;    w.nz = 1.0f; }
                        if (capUvs) { w.u = capUvs[uvBase + i].x; w.v = capUvs[uvBase + i].y; }
                        else        { w.u = 0.0f;                 w.v = 0.0f; }
                        w.color = vcol ? *reinterpret_cast<const DWORD*>(&vcol[i]) : 0xFFFFFFFFu;
                    }
                    if (triList) {
                        // NI::Triangle is 3 packed uint16 indices (== the IB byte layout
                        // used above via memcpy(.., triCount*6)). SK1 sky: force a re-upload every
                        // frame (g_walkingSky) so the dome's per-frame vertex-colour gradient
                        // bypasses the (modelId,vc,rev) dedup, which would otherwise skip it.
                        RenderProcess::captureGeometry(key, data->revisionID,
                            reinterpret_cast<uint32_t>(data),   // object identity (recycled-key guard)
                            scratch.data(), vertexCount,
                            reinterpret_cast<const uint16_t*>(triList), triCount * 3u,
                            g_walkingSky);
                    }
                }
            }

            ++g_uploadedThisFrame;
        }

        void buildD3DFromTransform(float out[16], const NI::Transform& t);

        // Static skinned VB: bind-pose positions + per-vertex top-4 bone influences
        // (weights + palette indices). Built once (or on revision change); the bone
        // matrices update per frame via buildBonePalette. Sets skinnedUnsupported
        // when numBones exceeds the shader palette (kMaxBones) — no CPU fallback.
        void buildSkinnedVB(CachedGeometry& e, NI::TriBasedGeometry* geom,
                            NI::TriBasedGeometryData* data,
                            NI::SkinInstance* si, NI::SkinData* sd) {
            const auto vertexCount = static_cast<uint32_t>(data->getActiveVertexCount());
            const auto triCount    = static_cast<uint32_t>(data->getActiveTriangleCount());
            const auto* mv = data->vertex;
            if (!vertexCount || !triCount || !mv) { releaseEntry(e); return; }

            const uint32_t key      = reinterpret_cast<uint32_t>(geom);
            const uint32_t numBones = sd->numBones;

            releaseEntry(e);
            e.writeSlot          = 0;
            e.vertexCount        = vertexCount;
            e.triangleCount      = triCount;
            e.revisionID         = data->revisionID;
            e.isSkinned          = true;
            e.numBones           = numBones;
            e.uvSetCount         = 1;       // skinnedDecl carries one UV set; multi-map is non-skinned only
            e.vbStride           = MGE::GeometryCache::kSkinnedVBStride;
            e.vbFVF              = 0;        // skinned draws use skinnedDecl, not an FVF
            e.hasVertexColor     = (data->color != nullptr);   // Phase 2: skinned VB now carries colour

            if (numBones > MGE::GeometryCache::kMaxBones) {
                // Too many bones for the VS palette — skip this caster, report once.
                e.skinnedUnsupported = true;
                static bool warnOnce = true;
                if (warnOnce) {
                    LOG::logline("!! [GEOM CACHE] skinned mesh has %u bones (> kMaxBones %u); skipped",
                                 numBones, MGE::GeometryCache::kMaxBones);
                    warnOnce = false;
                }
                return;
            }
            e.skinnedUnsupported = false;

            // Invert per-bone weight lists into per-vertex influences.
            struct Inf { float w; uint8_t b; };
            std::vector<std::vector<Inf>> perVert(vertexCount);
            for (uint32_t b = 0; b < numBones; ++b) {
                const auto& bd = sd->boneData[b];
                if (!bd.weights) continue;
                for (uint32_t k = 0; k < bd.weightCount; ++k) {
                    const uint32_t vi = bd.weights[k].index;
                    if (vi >= vertexCount) continue;
                    perVert[vi].push_back({ bd.weights[k].weight, static_cast<uint8_t>(b) });
                }
            }

            HRESULT hr = g_device->CreateVertexBuffer(
                vertexCount * MGE::GeometryCache::kSkinnedVBStride, D3DUSAGE_WRITEONLY,
                0, g_spikeForceDefaultPool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &e.vb[0], nullptr);
            if (FAILED(hr)) { e.vb[0] = nullptr; return; }

            // M-Skinning: also capture a model-space SkinnedVertexWire stream for the
            // Forge host (GPU palette skinning). Filled from the same inverted influences
            // we write into the D3D9 VB below; shipped once per (key,revision), re-uploaded
            // on revision change. Carries base-map UV (skinned texturing); per-vertex colour
            // (DiffAmb) is still omitted — a later fidelity tier.
            const bool wantCapture = RenderProcess::wantsGeometryCapture();
            static std::vector<IPC::SkinnedVertexWire> skScratch;  // single-threaded cache walk
            if (wantCapture) skScratch.resize(vertexCount);

            void* vbData = nullptr;
            if (SUCCEEDED(e.vb[0]->Lock(0, 0, &vbData, 0))) {
                auto* verts = static_cast<SkinnedVertex*>(vbData);
                const auto* uvs = data->textureCoords;
                const auto* nrm = data->normal;         // bind-pose model-space normals
                const auto* vcol = data->color;         // NI::PackedColor*(b,g,r,a)=D3DCOLOR, null if none
                for (uint32_t i = 0; i < vertexCount; ++i) {
                    auto& infs = perVert[i];
                    std::sort(infs.begin(), infs.end(),
                              [](const Inf& a, const Inf& b) { return a.w > b.w; });
                    float w[4] = {0,0,0,0};
                    uint8_t idx[4] = {0,0,0,0};
                    float sum = 0.0f;
                    const size_t n = infs.size() < 4 ? infs.size() : 4;
                    for (size_t j = 0; j < n; ++j) { w[j] = infs[j].w; idx[j] = infs[j].b; sum += infs[j].w; }
                    if (sum > 1e-6f) { for (int j = 0; j < 4; ++j) w[j] /= sum; }
                    else             { w[0] = 1.0f; }

                    verts[i].x = mv[i].x; verts[i].y = mv[i].y; verts[i].z = mv[i].z;
                    // Bind-pose normals; the VS skins them by the bone palette (same as
                    // position) for the cache color pass. Up if the mesh has none.
                    if (nrm) { verts[i].nx = nrm[i].x; verts[i].ny = nrm[i].y; verts[i].nz = nrm[i].z; }
                    else     { verts[i].nx = 0.0f; verts[i].ny = 0.0f; verts[i].nz = 1.0f; }
                    verts[i].w0 = w[0]; verts[i].w1 = w[1]; verts[i].w2 = w[2]; verts[i].w3 = w[3];
                    verts[i].indices = static_cast<DWORD>(idx[0])
                                     | (static_cast<DWORD>(idx[1]) << 8)
                                     | (static_cast<DWORD>(idx[2]) << 16)
                                     | (static_cast<DWORD>(idx[3]) << 24);
                    verts[i].u = uvs ? uvs[i].x : 0.0f;
                    verts[i].v = uvs ? uvs[i].y : 0.0f;
                    // PackedColor byte order (b,g,r,a) is exactly D3DCOLOR, copy straight.
                    verts[i].color = vcol ? *reinterpret_cast<const DWORD*>(&vcol[i]) : 0xFFFFFFFF;
                    // Mirror the same pos/normal/weights/indices into the host wire stream.
                    if (wantCapture) {
                        auto& sw = skScratch[i];
                        sw.px = verts[i].x;  sw.py = verts[i].y;  sw.pz = verts[i].z;
                        sw.nx = verts[i].nx; sw.ny = verts[i].ny; sw.nz = verts[i].nz;
                        sw.w0 = verts[i].w0; sw.w1 = verts[i].w1;
                        sw.w2 = verts[i].w2; sw.w3 = verts[i].w3;
                        sw.indices = verts[i].indices;
                        sw.u = verts[i].u;   sw.v = verts[i].v;   // base-map UV for the host
                    }
                }
                e.vb[0]->Unlock();
            }

            hr = g_device->CreateIndexBuffer(
                triCount * 6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16,
                g_spikeForceDefaultPool ? D3DPOOL_DEFAULT : D3DPOOL_MANAGED, &e.ib, nullptr);
            if (SUCCEEDED(hr)) {
                const auto* triList = data->getTriList();
                if (triList) {
                    void* ibData = nullptr;
                    if (SUCCEEDED(e.ib->Lock(0, 0, &ibData, 0))) {
                        memcpy(ibData, triList, triCount * 6);
                        e.ib->Unlock();
                    }
                }
            }

            const auto& b = data->bounds;
            e.boundsCenter[0] = b.center.x;
            e.boundsCenter[1] = b.center.y;
            e.boundsCenter[2] = b.center.z;
            e.boundsRadius    = b.radius;

            // Ship the captured skinned VB to the Forge host (one part, SKINNED flag +
            // numBones). The per-frame bone palette ships separately from buildDrawList.
            if (wantCapture) {
                const auto* triList = data->getTriList();
                if (triList) {
                    RenderProcess::captureSkinnedGeometry(key, data->revisionID,
                        reinterpret_cast<uint32_t>(data),   // object identity (recycled-key guard)
                        skScratch.data(), vertexCount,
                        reinterpret_cast<const uint16_t*>(triList), triCount * 3u, numBones);
                }
            }

            ++g_uploadedThisFrame;
        }

        // Per-frame: fill bonePalette with each bone's model->world matrix
        // (D3D row-vector form, pos*M = world). Cheap: numBones matrices, no
        // per-vertex work. Assumes numBones <= kMaxBones (guarded at VB build).
        void buildD3DTransform(float out[16], const NI::TriBasedGeometry* geom);

        void buildBonePalette(CachedGeometry& e, const NI::TriBasedGeometry* geom,
                              NI::SkinInstance* si, NI::SkinData* sd) {
            const uint32_t numBones = sd->numBones;
            e.bonePalette.resize(numBones * 16);
            bool partHadNullBone = false;
            for (uint32_t b = 0; b < numBones; ++b) {
                float* m = &e.bonePalette[b * 16];
                NI::AVObject* boneNode = si->bones[b];
                if (!boneNode) {
                    // Phase 0: a null bone influence. Fall back to the geometry's own
                    // world transform (keeps influenced verts attached to the object)
                    // instead of identity, which parked them at the model/cell origin
                    // and stretched origin->NPC triangles into the depth buffer. The
                    // diagnostic counters stay on permanently to catch any recurrence.
                    ++g_nullBoneHitsInterval;
                    if (!partHadNullBone) {
                        partHadNullBone = true;
                        ++g_nullBonePartsInterval;
                        if (e.textureName) g_nullBoneSampleTex = e.textureName;
                    }
                    buildD3DTransform(m, geom);
                    continue;
                }
                // Compose: apply bone offset, then bone world (matches CPU-skin math).
                const NI::Transform composed = boneNode->worldTransform * sd->boneData[b].transform;
                buildD3DFromTransform(m, composed);
            }
            e.numBones = numBones;
        }

        // Build a D3D9 row-major matrix (pos*M form) from an NI::Transform.
        void buildD3DFromTransform(float out[16], const NI::Transform& t) {
            const float s = t.scale;
            const auto& R = t.rotation;
            const auto& T = t.translation;
            out[0]  = s * R.m0.x; out[1]  = s * R.m1.x; out[2]  = s * R.m2.x; out[3]  = 0;
            out[4]  = s * R.m0.y; out[5]  = s * R.m1.y; out[6]  = s * R.m2.y; out[7]  = 0;
            out[8]  = s * R.m0.z; out[9]  = s * R.m1.z; out[10] = s * R.m2.z; out[11] = 0;
            out[12] = T.x;        out[13] = T.y;         out[14] = T.z;         out[15] = 1;
        }

        void buildD3DTransform(float out[16], const NI::TriBasedGeometry* geom) {
            buildD3DFromTransform(out, geom->worldTransform);
        }

        // Sign of the upper-left 3x3 determinant of a row-major affine matrix.
        // Negative => the transform mirrors (reflects) the mesh, flipping clip-space
        // triangle winding. Left-side body parts / armor reuse the right mesh via a
        // negative-scale node, so they hit this. The depth/shadow cache must cull the
        // OPPOSITE face for these, else it records the inner surface (SSAO shows
        // "inside-out" left limbs). Sign is layout-invariant (det A == det Aᵀ).
        bool isMirroredMatrix(const float m[16]) {
            const float det = m[0] * (m[5] * m[10] - m[6] * m[9])
                            - m[1] * (m[4] * m[10] - m[6] * m[8])
                            + m[2] * (m[4] * m[9]  - m[5] * m[8]);
            return det < 0.0f;
        }

        // Pick the transform that maps this part's mesh into world space: the bone
        // palette root for skinned parts (the limb's bones share the reflection sign),
        // the geometry's world transform otherwise.
        bool computeMirrored(const CachedGeometry& e) {
            const float* m = (e.isSkinned && e.numBones > 0 && !e.skinnedUnsupported)
                             ? e.bonePalette.data() : e.worldTransformD3D;
            return isMirroredMatrix(m);
        }

        void visitGeometry(NI::TriBasedGeometry* geom, bool inCharacter) {
            auto* data = geom->getModelData().get();
            if (!data) return;

            const uint32_t key = reinterpret_cast<uint32_t>(geom);

            // Skin state: a valid SkinInstance with SkinData + bone array.
            NI::SkinInstance* si = geom->skinInstance.get();
            NI::SkinData*     sd = si ? si->skinData.get() : nullptr;
            const bool sk = (si && sd && si->bones);

            auto it = g_cache.find(key);
            if (it == g_cache.end()) {
                auto& e = g_cache[key];
                e.vb[0] = e.vb[1] = nullptr; e.ib = nullptr; e.writeSlot = 0;
                e.numBones = 0; e.skinnedUnsupported = false;
                // Material first: uploadEntry reads the captured map UV sets (baseUV/
                // darkUV/detailUV/glowUV, set here) to size the VB's UV-set count.
                extractMaterial(e, geom);
                if (sk) {
                    buildSkinnedVB(e, geom, data, si, sd);          // static
                    if (!e.skinnedUnsupported) buildBonePalette(e, geom, si, sd);
                } else {
                    uploadEntry(e, geom, data, key);
                }
                buildD3DTransform(e.worldTransformD3D, geom);       // bounds center
                e.dynamicHint = (sk || inCharacter) ? 4 : 0;
                e.lastFrame = g_frame;
                e.isLandscape = g_walkingLandscape;
                e.isPickRoot = g_walkingPick;
                e.isSky = g_walkingSky;
                if (g_walkingSky) e.skyOrder = g_skyVisitCounter++;  // SK2 back-to-front key
                e.mirrored = computeMirrored(e);                    // winding flip for depth/shadow
            } else {
                auto& e = it->second;
                e.lastFrame = g_frame;
                e.isLandscape = g_walkingLandscape;
                e.isPickRoot = g_walkingPick;
                e.isSky = g_walkingSky;
                if (g_walkingSky) e.skyOrder = g_skyVisitCounter++;  // SK2 back-to-front key
                if (sk) {
                    // Static skinned VB: rebuild only on revision / skin-state change.
                    if (data->revisionID != e.revisionID || !e.isSkinned) {
                        extractMaterial(e, geom);
                        buildSkinnedVB(e, geom, data, si, sd);
                    }
                    if (!e.skinnedUnsupported) buildBonePalette(e, geom, si, sd);  // per frame
                    buildD3DTransform(e.worldTransformD3D, geom);            // bounds center
                    e.dynamicHint = 4;
                } else {
                    // SK1 sky: force re-extract + re-upload every frame (the dome's vertex
                    // colours change without a revisionID bump) — see g_walkingSky.
                    const bool changed = (data->revisionID != e.revisionID) || e.isSkinned || g_walkingSky;
                    if (changed) {
                        extractMaterial(e, geom);
                        uploadEntry(e, geom, data, key);
                    }
                    if (inCharacter) {
                        buildD3DTransform(e.worldTransformD3D, geom);
                        e.dynamicHint = 4;
                    } else {
                        float newTransform[16];
                        buildD3DTransform(newTransform, geom);
                        if (memcmp(newTransform, e.worldTransformD3D, sizeof(newTransform)) != 0) {
                            e.dynamicHint = 4;
                        } else if (e.dynamicHint > 0) {
                            --e.dynamicHint;
                        }
                        memcpy(e.worldTransformD3D, newTransform, sizeof(newTransform));
                    }
                }
                e.mirrored = computeMirrored(e);                    // winding flip for depth/shadow
            }
        }

        // bypassCull skips the entry's own app-cull check. Used for a NiSwitchNode's
        // active child: switchIndex already selected it, and a menu-frame cull pass may
        // have transiently app-culled it; the cache frustum-culls later anyway.
        void walk(NI::AVObject* av, bool inCharacter = false, bool bypassCull = false) {
            if (!av) return;
            if (!bypassCull && av->getAppCulled()) return;

            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiTriBasedGeom)) {
                visitGeometry(static_cast<NI::TriBasedGeometry*>(av), inCharacter);
                return;
            }

            // NiSwitchNode (e.g. "Glow in the Dark"'s NightDaySwitch): only the child
            // at switchIndex is displayed. Walk that child EXPLICITLY rather than
            // iterating all children and trusting per-child app-cull — in menu frames
            // the engine's cull pass can leave the inactive (day) variant un-culled and
            // the active (night) one culled, so the generic NiNode path below would
            // capture the wrong variant (day window showing while a menu is open).
            // switchIndex < 0 means no active child. Bypass the active child's own
            // app-cull so a transient menu cull can't drop it.
            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiSwitchNode)) {
                auto* sw = static_cast<NI::SwitchNode*>(av);
                const int idx = sw->switchIndex;
                if (idx >= 0 && (size_t)idx < sw->children.getEndIndex()) {
                    walk(sw->children.at(idx).get(), inCharacter, true);
                }
                return;
            }

            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) {
                auto* node = static_cast<NI::Node*>(av);
                const auto count = node->children.getEndIndex();
                // If not already flagged, check whether any direct child is a skinned
                // mesh — if so, the whole subtree is a character (NPC/creature) and all
                // non-skinned geometry within it (bone-attached equipment, head, etc.)
                // must be treated as dynamic regardless of per-frame transform delta.
                bool isCharNode = inCharacter;
                if (!isCharNode) {
                    for (size_t i = 0; i < count; ++i) {
                        NI::AVObject* child = node->children.at(i).get();
                        if (!child) continue;
                        if (child->isInstanceOfType(NI::RTTIStaticPtr::NiTriBasedGeom)) {
                            if (static_cast<NI::TriBasedGeometry*>(child)->skinInstance.get()) {
                                isCharNode = true;
                                break;
                            }
                        }
                    }
                }
                for (size_t i = 0; i < count; ++i)
                    walk(node->children.at(i), isCharNode);
            }
        }

    // ---- Reflection moon support (scene-graph-sourced) -----------------------------
    // The engine records a moon in recordSky only when it draws it for the MAIN camera,
    // so a moon up but outside the main frustum (looking away/down) never reaches the
    // water reflection that way. These helpers materialize the moon billboards straight
    // from the live scene graph, gated by the engine's own appCulled, so the reflection
    // can draw them frustum-independently.

    // Per-moon-shape D3D9 geometry keyed on NiGeometry*. Created once; vertex data is
    // refreshed each call (4 verts — negligible) so phase/fade/orientation updates track.
    // D3DPOOL_MANAGED so the buffers survive device reset without an explicit release.
    struct MoonGeom {
        IDirect3DVertexBuffer9* vb;
        IDirect3DIndexBuffer9*  ib;
        uint32_t vertCount, triCount;
    };
    std::unordered_map<uint32_t, MoonGeom> g_moonGeom;

    // NiAlphaProperty blend-function index (Gamebryo order) -> D3DBLEND.
    D3DBLEND niBlendToD3D(unsigned int ni) {
        switch (ni) {
            case 0:  return D3DBLEND_ONE;
            case 1:  return D3DBLEND_ZERO;
            case 2:  return D3DBLEND_SRCCOLOR;
            case 3:  return D3DBLEND_INVSRCCOLOR;
            case 4:  return D3DBLEND_DESTCOLOR;
            case 5:  return D3DBLEND_INVDESTCOLOR;
            case 6:  return D3DBLEND_SRCALPHA;
            case 7:  return D3DBLEND_INVSRCALPHA;
            case 8:  return D3DBLEND_DESTALPHA;
            case 9:  return D3DBLEND_INVDESTALPHA;
            case 10: return D3DBLEND_SRCALPHASAT;
            default: return D3DBLEND_ONE;
        }
    }

    // (Re)create + refresh the DepthVertex-layout VB/IB for one moon shape. Returns
    // false (shape skipped) on degenerate geometry or allocation failure.
    bool materializeMoonShape(NI::TriBasedGeometry* geom, MGE::GeometryCache::MoonShapeDraw& out) {
        auto* data = static_cast<NI::TriBasedGeometryData*>(geom->getModelData().get());
        if (!data) return false;
        const uint32_t vc  = static_cast<uint32_t>(data->getActiveVertexCount());
        const uint32_t tc  = static_cast<uint32_t>(data->getActiveTriangleCount());
        const auto*    mv  = data->vertex;
        const auto*    tri = data->getTriList();
        if (!vc || !tc || !mv || !tri) return false;

        const uint32_t key = reinterpret_cast<uint32_t>(geom);
        auto& g = g_moonGeom[key];

        if (!g.vb || g.vertCount != vc) {
            if (g.vb) { g.vb->Release(); g.vb = nullptr; }
            if (FAILED(g_device->CreateVertexBuffer(vc * MGE::GeometryCache::kVBStride, 0,
                    MGE::GeometryCache::kVBFVF, D3DPOOL_MANAGED, &g.vb, nullptr)))
                return false;
        }
        if (!g.ib || g.triCount != tc) {
            if (g.ib) { g.ib->Release(); g.ib = nullptr; }
            if (FAILED(g_device->CreateIndexBuffer(tc * 6, 0, D3DFMT_INDEX16,
                    D3DPOOL_MANAGED, &g.ib, nullptr)))
                return false;
        }
        g.vertCount = vc;
        g.triCount  = tc;

        const auto* nrm = data->normal;
        const auto* col = data->color;
        const auto* uv  = data->textureCoords;   // set 0 (moons are single-UV)
        void* vbData = nullptr;
        if (FAILED(g.vb->Lock(0, 0, &vbData, 0))) return false;
        auto* dst = static_cast<DepthVertex*>(vbData);
        for (uint32_t i = 0; i < vc; ++i) {
            dst[i].x = mv[i].x; dst[i].y = mv[i].y; dst[i].z = mv[i].z;
            if (nrm) { dst[i].nx = nrm[i].x; dst[i].ny = nrm[i].y; dst[i].nz = nrm[i].z; }
            else     { dst[i].nx = 0.0f;     dst[i].ny = 0.0f;     dst[i].nz = 1.0f; }
            dst[i].color = col ? *reinterpret_cast<const DWORD*>(&col[i]) : 0xFFFFFFFFu;
            if (uv) { dst[i].u = uv[i].x; dst[i].v = uv[i].y; }
            else    { dst[i].u = 0.0f;    dst[i].v = 0.0f; }
        }
        g.vb->Unlock();

        void* ibData = nullptr;
        if (SUCCEEDED(g.ib->Lock(0, 0, &ibData, 0))) {
            memcpy(ibData, tri, tc * 6);
            g.ib->Unlock();
        }

        out.vb        = g.vb;
        out.ib        = g.ib;
        out.vertCount = vc;
        out.triCount  = tc;
        buildD3DTransform(out.worldTransform, geom);
        return true;
    }

    // Recursively collect up to 2 drawable moon shapes under av, respecting per-node
    // appCulled (a full moon has no dark-side cutout, so its Shadow Node is culled).
    void collectMoonShapes(NI::AVObject* av, MGE::GeometryCache::MoonShapeDraw* out, int& count) {
        if (!av || count >= 2) return;
        if (av->getAppCulled()) return;

        if (av->isInstanceOfType(NI::RTTIStaticPtr::NiTriBasedGeom)) {
            auto* geom = static_cast<NI::TriBasedGeometry*>(av);
            MGE::GeometryCache::MoonShapeDraw d = {};
            if (!materializeMoonShape(geom, d)) return;

            d.texture   = nullptr;
            d.srcBlend  = D3DBLEND_SRCALPHA;
            d.destBlend = D3DBLEND_INVSRCALPHA;
            auto* ps = reinterpret_cast<NI::PropertyState*>(geom->propertyState);
            if (ps) {
                if (ps->texture) {
                    auto* baseMap = ps->texture->getBaseMap();
                    if (baseMap && baseMap->texture) {
                        auto* tex = baseMap->texture.get();
                        if (tex->isInstanceOfType(NI::RTTIStaticPtr::NiSourceTexture))
                            d.texture = getDX9Texture(tex);
                    }
                }
                if (ps->alpha) {
                    const unsigned short f = ps->alpha->flags;
                    d.srcBlend  = static_cast<unsigned char>(niBlendToD3D(
                        (f & NI::AlphaProperty::SRC_BLEND_MASK)  >> NI::AlphaProperty::SRC_BLEND_POS));
                    d.destBlend = static_cast<unsigned char>(niBlendToD3D(
                        (f & NI::AlphaProperty::DEST_BLEND_MASK) >> NI::AlphaProperty::DEST_BLEND_POS));
                }
            }
            if (!d.texture) return;   // no base map -> nothing to draw
            // The dark-side cutout lives under the moon root's 'Shadow Node'; the lit disc
            // under 'Moon Node'. Discriminate by that parent name — robust, since the disc
            // can share the cutout's alpha-blend mode (blend alone is ambiguous).
            d.isMoonShadow = false;
            if (av->parentNode) {
                const char* pn = av->parentNode->getName();
                if (pn && std::strcmp(pn, "Shadow Node") == 0) d.isMoonShadow = true;
            }
            out[count++] = d;
            return;
        }

        if (av->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) {
            auto* node = static_cast<NI::Node*>(av);
            const auto n = node->children.getEndIndex();
            for (size_t i = 0; i < n && count < 2; ++i)
                collectMoonShapes(node->children.at(i).get(), out, count);
        }
    }

    int buildMoonDrawListImpl(NI::Node* root, MGE::GeometryCache::MoonShapeDraw out[2]) {
        if (!g_device || !root) return 0;
        if (root->getAppCulled()) return 0;   // moon is down / hidden by phase
        int count = 0;
        collectMoonShapes(root, out, count);
        // Draw the dark-side cutout before the disc.
        if (count == 2 && !out[0].isMoonShadow && out[1].isMoonShadow) {
            MGE::GeometryCache::MoonShapeDraw t = out[0]; out[0] = out[1]; out[1] = t;
        }
        return count;
    }

    }

    void init(IDirect3DDevice9* device) {
        g_device = device;

        // Vertex declaration for skinned VBs (SkinnedVertex, stride 56). The COLOR
        // element (offset 52) is read only by the FFE cache-skin color path when the
        // part uses vertex colour; the depth/shadow skinned VS don't declare it and
        // D3D9 ignores unread elements, so those passes are unaffected.
        if (!g_skinnedDecl && g_device) {
            static const D3DVERTEXELEMENT9 elems[] = {
                {0, 0,  D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION,     0},
                {0, 12, D3DDECLTYPE_FLOAT3,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL,       0},
                {0, 24, D3DDECLTYPE_FLOAT4,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDWEIGHT,  0},
                {0, 40, D3DDECLTYPE_UBYTE4,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDINDICES, 0},
                {0, 44, D3DDECLTYPE_FLOAT2,   D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD,     0},
                {0, 52, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR,        0},
                D3DDECL_END()
            };
            g_device->CreateVertexDeclaration(elems, &g_skinnedDecl);
        }
    }

    IDirect3DVertexDeclaration9* skinnedDecl() {
        return g_skinnedDecl;
    }

    // ---- SK0: sky-takeover diagnostic (scene-graph inspect, NO capture) ----------
    // The sky is a proper NiNode subtree "skyRoot" — a sibling of worldRoot under the
    // World Scene Graph Root (the node that parents the cell roots + the camera root).
    // Reach it by climbing worldObjectRoot to the topmost ancestor and finding the
    // "skyRoot" child by name. SK0 only LOGS the subtree (name / cull / verts / blend /
    // alpha-test / texture) so we can confirm what the real walk would capture before
    // any host draw. No g_cache writes, no side effects. Gated + throttled by the caller.
    static NI::Node* findSkyRoot(void* dataHandler) {
        NI::AVObject* n = MGE::DataHandlerView::worldObjectRoot(dataHandler);
        if (!n) return nullptr;
        while (n->parentNode) n = n->parentNode;          // climb to World Scene Graph Root
        if (!n->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) return nullptr;
        auto* root = static_cast<NI::Node*>(n);
        const auto count = root->children.getEndIndex();
        for (size_t i = 0; i < count; ++i) {
            NI::AVObject* c = root->children.at(i).get();
            if (!c) continue;
            const char* nm = c->getName();
            if (nm && std::strcmp(nm, "skyRoot") == 0
                   && c->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) {
                return static_cast<NI::Node*>(c);
            }
        }
        return nullptr;
    }

    static void dumpSkyNode(NI::AVObject* av, int depth) {
        if (!av) return;
        const char* nm     = av->getName();
        const bool  culled = av->getAppCulled();
        if (av->isInstanceOfType(NI::RTTIStaticPtr::NiTriBasedGeom)) {
            auto* geom = static_cast<NI::TriBasedGeometry*>(av);
            auto* data = geom->getModelData().get();
            const uint32_t vc = data ? static_cast<uint32_t>(data->getActiveVertexCount()) : 0u;
            bool blend = false, atest = false; const char* texName = nullptr;
            auto* ps = reinterpret_cast<NI::PropertyState*>(geom->propertyState);
            if (ps) {
                if (ps->alpha) {
                    blend = (ps->alpha->flags & NI::AlphaProperty::ALPHA_MASK) != 0;
                    atest = (ps->alpha->flags & NI::AlphaProperty::TEST_ENABLE_MASK) != 0;
                }
                if (ps->texture) {
                    auto* baseMap = ps->texture->getBaseMap();
                    if (baseMap && baseMap->texture) {
                        auto* tex = baseMap->texture.get();
                        if (tex->isInstanceOfType(NI::RTTIStaticPtr::NiSourceTexture))
                            texName = static_cast<NI::SourceTexture*>(tex)->fileName;
                    }
                }
            }
            LOG::logline("[SKY] %*sGEOM '%s' culled=%d verts=%u blend=%d atest=%d tex=%s",
                depth * 2, "", nm ? nm : "(null)", culled ? 1 : 0, vc,
                blend ? 1 : 0, atest ? 1 : 0, texName ? texName : "(none)");
            return;
        }
        LOG::logline("[SKY] %*sNODE '%s' culled=%d", depth * 2, "", nm ? nm : "(null)", culled ? 1 : 0);
        if (av->isInstanceOfType(NI::RTTIStaticPtr::NiNode)) {
            auto* node = static_cast<NI::Node*>(av);
            const auto count = node->children.getEndIndex();
            for (size_t i = 0; i < count; ++i)
                dumpSkyNode(node->children.at(i).get(), depth + 1);
        }
    }

    void onFrameReady(void* dataHandler) {
        if (!g_device || !dataHandler) return;
        if (!Configuration.UseSceneGraphSnapshot) return;
        MGE_ZoneScopedN("GeometryCache::onFrameReady");

        ++g_frame;
        g_uploadedThisFrame = 0;

        {
            MGE_ZoneScopedN("GeomCache:walkObjects");
            walk(MGE::DataHandlerView::worldObjectRoot(dataHandler));
        }
        {
            MGE_ZoneScopedN("GeomCache:walkPickObjects");
            g_walkingPick = true;
            walk(MGE::DataHandlerView::worldPickObjectRoot(dataHandler));
            g_walkingPick = false;
        }
        {
            MGE_ZoneScopedN("GeomCache:walkLandscape");
            g_walkingLandscape = true;
            walk(MGE::DataHandlerView::worldLandscapeRoot(dataHandler));
            g_walkingLandscape = false;
        }
        // SK1 sky takeover: walk skyRoot only when the Forge sky pass is live (F7 toggle ON +
        // seam compositing). walk()'s getAppCulled() early-return gives free day/night/phase/
        // weather selection; captured shapes ride the same capture/IPC seam as opaques but are
        // tagged isSky → the Forge host's dedicated alpha-blend sky pass draws them.
        if (RenderProcess::wantsSkyCapture()) {
            MGE_ZoneScopedN("GeomCache:walkSky");
            g_walkingSky = true;
            g_skyVisitCounter = 0;   // SK2: restart back-to-front ordering each sky walk
            walk(findSkyRoot(dataHandler));
            g_walkingSky = false;
        }

        // SK0 (sky takeover, diagnostic): periodically dump the skyRoot subtree so we can
        // confirm the shapes/materials the real walk will capture. No capture, no draw.
        if (Configuration.LogDistantPipeline) {
            static uint64_t s_lastSkyLog = 0;
            if (g_frame - s_lastSkyLog >= 300) {
                s_lastSkyLog = g_frame;
                if (NI::Node* skyRoot = findSkyRoot(dataHandler)) {
                    LOG::logline("== [SKY DUMP] frame %llu ==", (unsigned long long)g_frame);
                    dumpSkyNode(skyRoot, 0);
                } else {
                    LOG::logline("== [SKY DUMP] skyRoot NOT found (frame %llu) ==", (unsigned long long)g_frame);
                }
            }
        }

        // Evict entries not seen this frame
        {
            MGE_ZoneScopedN("GeomCache:evict");
            for (auto it = g_cache.begin(); it != g_cache.end(); ) {
                if (it->second.lastFrame != g_frame) {
                    releaseEntry(it->second);
                    it = g_cache.erase(it);
                } else {
                    ++it;
                }
            }
        }

        g_uploadedInterval += g_uploadedThisFrame;

        if (Configuration.LogDistantPipeline) {
            static uint64_t s_lastLog = 0;
            if (g_frame - s_lastLog >= 1800) {
                uint32_t skinnedCount = 0, namedTexCount = 0, nullTexCount = 0, nullNameCount = 0;
                uint32_t mirroredCount = 0;
                // Terrain (isLandscape) characterization: how the splat passes land
                // in the cache. landOpaque = base layers (no blend), landAlphaTest,
                // landBlend = alpha-splat layers (separate trishapes if >0 here).
                // landVCol = carry vertex colours. Samples a few texture names.
                uint32_t landCount = 0, landOpaque = 0, landAlphaTest = 0, landBlend = 0, landVCol = 0;
                const char* landTexA = nullptr; const char* landTexB = nullptr;
                for (const auto& kv : g_cache) {
                    if (kv.second.isSkinned) ++skinnedCount;
                    if (kv.second.mirrored) ++mirroredCount;
                    if (kv.second.d3dTexture && kv.second.textureName) ++namedTexCount;
                    else if (!kv.second.d3dTexture) ++nullTexCount;
                    else ++nullNameCount;
                    if (kv.second.isLandscape) {
                        ++landCount;
                        if (kv.second.blendEnable) ++landBlend;
                        else if (kv.second.alphaTest) ++landAlphaTest;
                        else ++landOpaque;
                        if (kv.second.hasVertexColor) ++landVCol;
                        if (kv.second.textureName) {
                            if (!landTexA) landTexA = kv.second.textureName;
                            else if (!landTexB && kv.second.textureName != landTexA) landTexB = kv.second.textureName;
                        }
                    }
                }
                LOG::logline("-- [GEOM CACHE] frame=%llu cached=%zu skinned=%u named=%u nulltex=%u nullname=%u mapSize=%zu uploads/interval=%llu",
                    g_frame, g_cache.size(), skinnedCount, namedTexCount, nullTexCount, nullNameCount, g_textureNameMap.size(), g_uploadedInterval);
                LOG::logline("-- [GEOM CACHE] nullbone hits/interval=%u parts/interval=%u sampleTex=%s mirrored=%u",
                    g_nullBoneHitsInterval, g_nullBonePartsInterval,
                    g_nullBoneSampleTex ? g_nullBoneSampleTex : "(none)", mirroredCount);
                LOG::logline("-- [GEOM CACHE] landscape=%u opaque=%u alphatest=%u blend=%u vcol=%u texA=%s texB=%s",
                    landCount, landOpaque, landAlphaTest, landBlend, landVCol,
                    landTexA ? landTexA : "(none)", landTexB ? landTexB : "(none)");
                g_uploadedInterval = 0;
                g_nullBoneHitsInterval = 0;
                g_nullBonePartsInterval = 0;
                g_nullBoneSampleTex = nullptr;
                s_lastLog = g_frame;
            }
        }
    }

    const std::unordered_map<uint32_t, CachedGeometry>& cache() {
        return g_cache;
    }

    int buildMoonDrawList(void* moonRoot, MoonShapeDraw out[2]) {
        return buildMoonDrawListImpl(static_cast<NI::Node*>(moonRoot), out);
    }

    const char* resolveTextureName(IDirect3DTexture9* tex) {
        if (!tex) return nullptr;
        auto it = g_textureNameMap.find(tex);
        return (it != g_textureNameMap.end()) ? it->second : nullptr;
    }

}
