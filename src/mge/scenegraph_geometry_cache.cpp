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

namespace MGE::GeometryCache {

    namespace {

        IDirect3DDevice9* g_device          = nullptr;
        uint64_t          g_frame          = 0;
        uint32_t          g_uploadedThisFrame  = 0;
        uint64_t          g_uploadedInterval   = 0; // cumulative over log interval

        std::unordered_map<uint32_t, CachedGeometry>      g_cache;
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

        void registerTexture(NI::Texture* tex) {
            if (!tex || !tex->isInstanceOfType(NI::RTTIStaticPtr::NiSourceTexture)) return;
            auto* st = static_cast<NI::SourceTexture*>(tex);
            if (!st->fileName) return;
            auto* dx9 = getDX9Texture(tex);
            if (dx9) g_textureNameMap[dx9] = st->fileName;
        }

        void registerAllMaps(NI::Geometry* geom) {
            auto* ps = reinterpret_cast<NI::PropertyState*>(geom->propertyState);
            if (!ps || !ps->texture) return;
            auto* tp = ps->texture;
            // Slot 0: base texture (all geometry)
            const auto* bm = tp->getBaseMap();
            if (bm && bm->texture) registerTexture(bm->texture.get());
            // Slot 6: terrain decal (blend overlay from adjacent patch)
            if (tp->maps.getEndIndex() > 6u) {
                const auto* dm = tp->maps.at(6);
                if (dm && dm->texture) registerTexture(dm->texture.get());
            }
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

        void uploadEntry(CachedGeometry& e, NI::TriBasedGeometry* geom,
                         NI::TriBasedGeometryData* data) {
            const auto vertexCount = static_cast<uint32_t>(data->getActiveVertexCount());
            const auto triCount    = static_cast<uint32_t>(data->getActiveTriangleCount());

            const auto* mv = data->vertex;          // model-space, always present
            const bool  sk = (geom->skinInstance.get() != nullptr);

            // Invalid geometry — drop any stale buffers so the entry is skipped.
            if (!vertexCount || !triCount || !mv) {
                releaseEntry(e);
                return;
            }

            // Validate skin data before allocating VB (avoids allocate-then-release).
            NI::SkinInstance* si = nullptr;
            NI::SkinData*     sd = nullptr;
            if (sk) {
                si = geom->skinInstance.get();
                sd = si ? si->skinData.get() : nullptr;
                if (!sd || !si->bones) {
                    releaseEntry(e);
                    return;
                }
            }

            // On a size change, drop both slots + IB so they repopulate at the
            // new size; the unused slot rebuilds lazily next time it's written.
            const bool sizeChanged = (e.vertexCount != vertexCount) || (e.triangleCount != triCount);
            if (sizeChanged) {
                if (e.vb[0]) { e.vb[0]->Release(); e.vb[0] = nullptr; }
                if (e.vb[1]) { e.vb[1]->Release(); e.vb[1] = nullptr; }
                if (e.ib)    { e.ib->Release();    e.ib    = nullptr; }
            }

            // Write the slot NOT being read by the in-flight depth pass (which
            // reads vb[writeSlot] from last frame); flip writeSlot at the end so
            // the shadow pass reads the fresh slot. Avoids per-frame VB churn
            // while keeping the depth/shadow CPU-GPU overlap hazard-free.
            const uint8_t slot = 1u - e.writeSlot;

            // VB: world-space pos + zero normal + white color + UV (set 0).
            // Matches MorrowindVertIn so the depth/shadow VS declaration is satisfied.
            if (!e.vb[slot]) {
                HRESULT hr = g_device->CreateVertexBuffer(
                    vertexCount * sizeof(DepthVertex),
                    D3DUSAGE_WRITEONLY,
                    MGE::GeometryCache::kVBFVF,
                    D3DPOOL_MANAGED, &e.vb[slot], nullptr);
                if (FAILED(hr)) { e.vb[slot] = nullptr; return; }
            }

            // IB: triangle indices (uint16, 6 bytes per triangle), shared across
            // slots. Topology is constant for skinned meshes, so the IB is written
            // only when it is (re)created; non-skinned re-uploads (revision bump)
            // rewrite it too.
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

                if (sk) {
                    // CPU skinning: accumulate weighted bone contributions into world-space.
                    for (uint32_t i = 0; i < vertexCount; ++i) {
                        verts[i].x = verts[i].y = verts[i].z = 0.0f;
                        verts[i].nx = 0.0f; verts[i].ny = 0.0f; verts[i].nz = 0.0f;
                        verts[i].color = 0xFFFFFFFF;
                        verts[i].u = uvs ? uvs[i].x : 0.0f;
                        verts[i].v = uvs ? uvs[i].y : 0.0f;
                    }
                    const uint32_t numBones = sd->numBones;
                    for (uint32_t b = 0; b < numBones; ++b) {
                        auto* boneNode = si->bones[b];
                        if (!boneNode) continue;
                        const auto& bdata = sd->boneData[b];
                        if (!bdata.weights) continue;
                        const NI::Transform& offset = bdata.transform;
                        const NI::Transform& bworld = boneNode->worldTransform;
                        for (uint32_t k = 0; k < bdata.weightCount; ++k) {
                            const uint32_t vi = bdata.weights[k].index;
                            if (vi >= vertexCount) continue;
                            const float w = bdata.weights[k].weight;
                            const NI::Point3 v_bone  = offset * NI::Point3{mv[vi].x, mv[vi].y, mv[vi].z};
                            const NI::Point3 v_world = bworld * v_bone;
                            verts[vi].x += w * v_world.x;
                            verts[vi].y += w * v_world.y;
                            verts[vi].z += w * v_world.z;
                        }
                    }
                } else {
                    // Non-skinned: store model-space positions.
                    // worldTransformD3D is set per-frame and applied per-draw.
                    for (uint32_t i = 0; i < vertexCount; ++i) {
                        verts[i].x = mv[i].x; verts[i].y = mv[i].y; verts[i].z = mv[i].z;
                        verts[i].nx = 0.0f; verts[i].ny = 0.0f; verts[i].nz = 0.0f;
                        verts[i].color = 0xFFFFFFFF;
                        verts[i].u = uvs ? uvs[i].x : 0.0f;
                        verts[i].v = uvs ? uvs[i].y : 0.0f;
                    }
                }
                e.vb[slot]->Unlock();
            }

            if (e.ib && (createdIB || !sk)) {
                const auto* triList = data->getTriList();
                if (triList) {
                    void* ibData = nullptr;
                    if (SUCCEEDED(e.ib->Lock(0, 0, &ibData, 0))) {
                        memcpy(ibData, triList, triCount * 6);
                        e.ib->Unlock();
                    }
                }
            }

            e.writeSlot     = slot;   // flip: shadow pass now reads the fresh slot
            e.vertexCount   = vertexCount;
            e.triangleCount = triCount;
            e.revisionID    = data->revisionID;
            e.isSkinned     = (geom->skinInstance.get() != nullptr);

            const auto& b = data->bounds;
            e.boundsCenter[0] = b.center.x;
            e.boundsCenter[1] = b.center.y;
            e.boundsCenter[2] = b.center.z;
            e.boundsRadius    = b.radius;

            ++g_uploadedThisFrame;
        }

        // Build a D3D9 row-major world matrix from the NI::Transform.
        void buildD3DTransform(float out[16], const NI::TriBasedGeometry* geom) {
            const float s = geom->worldTransform.scale;
            const auto& R = geom->worldTransform.rotation;
            const auto& T = geom->worldTransform.translation;
            out[0]  = s * R.m0.x; out[1]  = s * R.m1.x; out[2]  = s * R.m2.x; out[3]  = 0;
            out[4]  = s * R.m0.y; out[5]  = s * R.m1.y; out[6]  = s * R.m2.y; out[7]  = 0;
            out[8]  = s * R.m0.z; out[9]  = s * R.m1.z; out[10] = s * R.m2.z; out[11] = 0;
            out[12] = T.x;        out[13] = T.y;         out[14] = T.z;         out[15] = 1;
        }

        void visitGeometry(NI::TriBasedGeometry* geom, bool inCharacter) {
            auto* data = geom->getModelData().get();
            if (!data) return;

            registerAllMaps(geom);

            const uint32_t key = reinterpret_cast<uint32_t>(geom);

            auto it = g_cache.find(key);
            if (it == g_cache.end()) {
                auto& e = g_cache[key];
                e.vb[0] = e.vb[1] = nullptr; e.ib = nullptr; e.writeSlot = 0;
                uploadEntry(e, geom, data);
                extractMaterial(e, geom);
                buildD3DTransform(e.worldTransformD3D, geom);
                e.dynamicHint = inCharacter ? 4 : 0;
                e.lastFrame = g_frame;
            } else {
                auto& e = it->second;
                e.lastFrame = g_frame;
                const bool changed = (data->revisionID != e.revisionID) || e.isSkinned;
                if (changed) {
                    uploadEntry(e, geom, data);
                    extractMaterial(e, geom);
                }
                if (inCharacter) {
                    // Character parts: always live, no need to diff the transform.
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
    }

    void onFrameReady(void* dataHandler) {
        if (!g_device || !dataHandler) return;
        if (!Configuration.UseSceneGraphSnapshot) return;
        MGE_ZoneScopedN("GeometryCache::onFrameReady");

        ++g_frame;
        g_uploadedThisFrame = 0;

        g_textureNameMap.clear();
        walk(MGE::DataHandlerView::worldObjectRoot(dataHandler));
        walk(MGE::DataHandlerView::worldPickObjectRoot(dataHandler));
        walk(MGE::DataHandlerView::worldLandscapeRoot(dataHandler));

        // Evict entries not seen this frame
        for (auto it = g_cache.begin(); it != g_cache.end(); ) {
            if (it->second.lastFrame != g_frame) {
                releaseEntry(it->second);
                it = g_cache.erase(it);
            } else {
                ++it;
            }
        }

        g_uploadedInterval += g_uploadedThisFrame;

        if (Configuration.LogDistantPipeline) {
            static uint64_t s_lastLog = 0;
            if (g_frame - s_lastLog >= 1800) {
                uint32_t skinnedCount = 0, namedTexCount = 0, nullTexCount = 0, nullNameCount = 0;
                for (const auto& kv : g_cache) {
                    if (kv.second.isSkinned) ++skinnedCount;
                    if (kv.second.d3dTexture && kv.second.textureName) ++namedTexCount;
                    else if (!kv.second.d3dTexture) ++nullTexCount;
                    else ++nullNameCount;
                }
                LOG::logline("-- [GEOM CACHE] frame=%llu cached=%zu skinned=%u named=%u nulltex=%u nullname=%u mapSize=%zu uploads/interval=%llu",
                    g_frame, g_cache.size(), skinnedCount, namedTexCount, nullTexCount, nullNameCount, g_textureNameMap.size(), g_uploadedInterval);
                g_uploadedInterval = 0;
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
