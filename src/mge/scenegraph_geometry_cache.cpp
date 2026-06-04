#include "mge_se_prelude.h"

#include "NIAVObject.h"
#include "NIGeometry.h"
#include "NIGeometryData.h"
#include "NINode.h"
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
#include "scenegraph_geometry_cache.h"
#include "support/log.h"

#include <algorithm>

namespace MGE::GeometryCache {

    namespace {

        IDirect3DDevice9* g_device          = nullptr;
        IDirect3DVertexDeclaration9* g_skinnedDecl = nullptr;
        uint64_t          g_frame          = 0;
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
            e.textureName = nullptr;
            e.alphaRef    = 0.0f;
            e.alphaTest   = false;
            e.blendEnable = false;

            auto* ps = reinterpret_cast<NI::PropertyState*>(geom->propertyState);
            if (!ps) return;

            if (ps->alpha) {
                const auto* ap = ps->alpha;
                e.alphaTest   = (ap->flags & NI::AlphaProperty::TEST_ENABLE_MASK) != 0;
                e.blendEnable = (ap->flags & NI::AlphaProperty::ALPHA_MASK) != 0;
                e.alphaRef    = ap->alphaTestRef / 255.0f;
            }

            if (ps->texture) {
                const auto* baseMap = ps->texture->getBaseMap();
                if (baseMap && baseMap->texture) {
                    auto* tex = baseMap->texture.get();
                    if (tex->isInstanceOfType(NI::RTTIStaticPtr::NiSourceTexture)) {
                        auto* st = static_cast<NI::SourceTexture*>(tex);
                        e.textureName = st->fileName;
                        e.d3dTexture  = getDX9Texture(tex);
                    }
                }
            }
        }

        // Vertex layout matching MorrowindVertIn (depth/shadow VS input).
        // D3DFVF_XYZ | D3DFVF_NORMAL | D3DFVF_DIFFUSE | D3DFVF_TEX1, stride 36.
        struct DepthVertex {
            float x, y, z;    // POSITION  (12)
            float nx, ny, nz; // NORMAL    (12, zeros — depth/shadow don't use normals)
            DWORD color;       // DIFFUSE   ( 4, 0xFFFFFFFF — hasVCol=false, unused)
            float u, v;        // TEXCOORD0 ( 8, from UV set 0)
        };
        static_assert(sizeof(DepthVertex) == 36, "DepthVertex size mismatch");

        // Skinned vertex layout matching SkinnedVertIn (VS palette skinning input).
        // Drawn with g_skinnedDecl; stride 40.
        struct SkinnedVertex {
            float x, y, z;          // POSITION     (12) bind-pose
            float w0, w1, w2, w3;   // BLENDWEIGHT  (16) top-4 influences, normalized
            DWORD indices;          // BLENDINDICES ( 4) UBYTE4 bone palette indices
            float u, v;             // TEXCOORD0    ( 8)
        };
        static_assert(sizeof(SkinnedVertex) == 40, "SkinnedVertex size mismatch");

        void uploadEntry(CachedGeometry& e, NI::TriBasedGeometry* geom,
                         NI::TriBasedGeometryData* data) {
            const auto vertexCount = static_cast<uint32_t>(data->getActiveVertexCount());
            const auto triCount    = static_cast<uint32_t>(data->getActiveTriangleCount());

            const auto* mv = data->vertex;          // model-space, always present

            // Invalid geometry — drop any stale buffers so the entry is skipped.
            if (!vertexCount || !triCount || !mv) {
                releaseEntry(e);
                return;
            }

            // On a size change, drop both slots + IB so they repopulate at the new size.
            const bool sizeChanged = (e.vertexCount != vertexCount) || (e.triangleCount != triCount);
            if (sizeChanged) {
                releaseEntry(e);
            }

            const uint8_t slot = 1u - e.writeSlot;

            // Model-space VB (XYZ|NORMAL|DIFFUSE|TEX1); per-draw world*view palette.
            if (!e.vb[slot]) {
                HRESULT hr = g_device->CreateVertexBuffer(
                    vertexCount * sizeof(DepthVertex),
                    D3DUSAGE_WRITEONLY,
                    MGE::GeometryCache::kVBFVF,
                    D3DPOOL_MANAGED, &e.vb[slot], nullptr);
                if (FAILED(hr)) { e.vb[slot] = nullptr; return; }
            }

            const bool createdIB = (e.ib == nullptr);
            if (createdIB) {
                g_device->CreateIndexBuffer(
                    triCount * 6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16,
                    D3DPOOL_MANAGED, &e.ib, nullptr);
            }

            void* vbData = nullptr;
            if (SUCCEEDED(e.vb[slot]->Lock(0, 0, &vbData, 0))) {
                auto* verts = static_cast<DepthVertex*>(vbData);
                const auto* uvs = data->textureCoords;  // NI::Point2*, nullptr if no UVs
                for (uint32_t i = 0; i < vertexCount; ++i) {
                    verts[i].x = mv[i].x; verts[i].y = mv[i].y; verts[i].z = mv[i].z;
                    verts[i].nx = 0.0f; verts[i].ny = 0.0f; verts[i].nz = 0.0f;
                    verts[i].color = 0xFFFFFFFF;
                    verts[i].u = uvs ? uvs[i].x : 0.0f;
                    verts[i].v = uvs ? uvs[i].y : 0.0f;
                }
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

            const auto& b = data->bounds;
            e.boundsCenter[0] = b.center.x;
            e.boundsCenter[1] = b.center.y;
            e.boundsCenter[2] = b.center.z;
            e.boundsRadius    = b.radius;

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

            const uint32_t numBones = sd->numBones;

            releaseEntry(e);
            e.writeSlot          = 0;
            e.vertexCount        = vertexCount;
            e.triangleCount      = triCount;
            e.revisionID         = data->revisionID;
            e.isSkinned          = true;
            e.numBones           = numBones;

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
                0, D3DPOOL_MANAGED, &e.vb[0], nullptr);
            if (FAILED(hr)) { e.vb[0] = nullptr; return; }

            void* vbData = nullptr;
            if (SUCCEEDED(e.vb[0]->Lock(0, 0, &vbData, 0))) {
                auto* verts = static_cast<SkinnedVertex*>(vbData);
                const auto* uvs = data->textureCoords;
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
                    verts[i].w0 = w[0]; verts[i].w1 = w[1]; verts[i].w2 = w[2]; verts[i].w3 = w[3];
                    verts[i].indices = static_cast<DWORD>(idx[0])
                                     | (static_cast<DWORD>(idx[1]) << 8)
                                     | (static_cast<DWORD>(idx[2]) << 16)
                                     | (static_cast<DWORD>(idx[3]) << 24);
                    verts[i].u = uvs ? uvs[i].x : 0.0f;
                    verts[i].v = uvs ? uvs[i].y : 0.0f;
                }
                e.vb[0]->Unlock();
            }

            hr = g_device->CreateIndexBuffer(
                triCount * 6, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16,
                D3DPOOL_MANAGED, &e.ib, nullptr);
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
                if (sk) {
                    buildSkinnedVB(e, geom, data, si, sd);          // static
                    if (!e.skinnedUnsupported) buildBonePalette(e, geom, si, sd);
                } else {
                    uploadEntry(e, geom, data);
                }
                extractMaterial(e, geom);
                buildD3DTransform(e.worldTransformD3D, geom);       // bounds center
                e.dynamicHint = (sk || inCharacter) ? 4 : 0;
                e.lastFrame = g_frame;
                e.mirrored = computeMirrored(e);                    // winding flip for depth/shadow
            } else {
                auto& e = it->second;
                e.lastFrame = g_frame;
                if (sk) {
                    // Static skinned VB: rebuild only on revision / skin-state change.
                    if (data->revisionID != e.revisionID || !e.isSkinned) {
                        buildSkinnedVB(e, geom, data, si, sd);
                        extractMaterial(e, geom);
                    }
                    if (!e.skinnedUnsupported) buildBonePalette(e, geom, si, sd);  // per frame
                    buildD3DTransform(e.worldTransformD3D, geom);            // bounds center
                    e.dynamicHint = 4;
                } else {
                    const bool changed = (data->revisionID != e.revisionID) || e.isSkinned;
                    if (changed) {
                        uploadEntry(e, geom, data);
                        extractMaterial(e, geom);
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

        void walk(NI::AVObject* av, bool inCharacter = false) {
            if (!av || av->getAppCulled()) return;

            if (av->isInstanceOfType(NI::RTTIStaticPtr::NiTriBasedGeom)) {
                visitGeometry(static_cast<NI::TriBasedGeometry*>(av), inCharacter);
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

    }

    void init(IDirect3DDevice9* device) {
        g_device = device;

        // Vertex declaration for skinned VBs (SkinnedVertex, stride 40).
        if (!g_skinnedDecl && g_device) {
            static const D3DVERTEXELEMENT9 elems[] = {
                {0, 0,  D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION,     0},
                {0, 12, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDWEIGHT,  0},
                {0, 28, D3DDECLTYPE_UBYTE4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_BLENDINDICES, 0},
                {0, 32, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD,     0},
                D3DDECL_END()
            };
            g_device->CreateVertexDeclaration(elems, &g_skinnedDecl);
        }
    }

    IDirect3DVertexDeclaration9* skinnedDecl() {
        return g_skinnedDecl;
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
            walk(MGE::DataHandlerView::worldPickObjectRoot(dataHandler));
        }
        {
            MGE_ZoneScopedN("GeomCache:walkLandscape");
            walk(MGE::DataHandlerView::worldLandscapeRoot(dataHandler));
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
                for (const auto& kv : g_cache) {
                    if (kv.second.isSkinned) ++skinnedCount;
                    if (kv.second.mirrored) ++mirroredCount;
                    if (kv.second.d3dTexture && kv.second.textureName) ++namedTexCount;
                    else if (!kv.second.d3dTexture) ++nullTexCount;
                    else ++nullNameCount;
                }
                LOG::logline("-- [GEOM CACHE] frame=%llu cached=%zu skinned=%u named=%u nulltex=%u nullname=%u mapSize=%zu uploads/interval=%llu",
                    g_frame, g_cache.size(), skinnedCount, namedTexCount, nullTexCount, nullNameCount, g_textureNameMap.size(), g_uploadedInterval);
                LOG::logline("-- [GEOM CACHE] nullbone hits/interval=%u parts/interval=%u sampleTex=%s mirrored=%u",
                    g_nullBoneHitsInterval, g_nullBonePartsInterval,
                    g_nullBoneSampleTex ? g_nullBoneSampleTex : "(none)", mirroredCount);
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

    const char* resolveTextureName(IDirect3DTexture9* tex) {
        if (!tex) return nullptr;
        auto it = g_textureNameMap.find(tex);
        return (it != g_textureNameMap.end()) ? it->second : nullptr;
    }

}
