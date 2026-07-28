#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "../3rdparty/niflib/include/niflib.h"
#include "../3rdparty/niflib/include/obj/NiObject.h"
#include "../3rdparty/niflib/include/obj/NiAVObject.h"
#include "../3rdparty/niflib/include/obj/NiNode.h"
#include "../3rdparty/niflib/include/obj/NiSwitchNode.h"
#include "../3rdparty/niflib/include/obj/NiLODNode.h"
#include "../3rdparty/niflib/include/obj/NiProperty.h"
#include "../3rdparty/niflib/include/obj/NiAlphaProperty.h"
#include "../3rdparty/niflib/include/obj/NiMaterialProperty.h"
#include "../3rdparty/niflib/include/obj/NiTexturingProperty.h"
#include "../3rdparty/niflib/include/obj/NiSourceTexture.h"
#include "../3rdparty/niflib/include/obj/NiStringExtraData.h"
#include "../3rdparty/niflib/include/obj/NiTriBasedGeom.h"
#include "../3rdparty/niflib/include/obj/NiTriBasedGeomData.h"
#include "../3rdparty/niflib/include/obj/NiTriStripsData.h"
#include "../3rdparty/niflib/include/obj/NiUVController.h"
#include "../3rdparty/niflib/include/obj/NiUVData.h"
#include "../3rdparty/niflib/include/obj/RootCollisionNode.h"

#include <assert.h>
#include <fstream>
#include <strstream>
#include <vector>
#include <float.h>
#include <map>
#include <memory>
#include <cstdint>
#include <utility>

#include <d3d9.h>
#include <d3dx9.h>
#include "DXVertex.h"

#include "progmesh/ProgMesh.h"

// Vertex Cache Optimizer
#include "../3rdparty/tootle/src/TootleLib/include/tootlelib.h"

using namespace Niflib;
using std::vector;


static IDirect3DDevice9* device;
static HANDLE staticFile;

// --- Hero distant-statics animation capture (MGE XE mod: ghostfence + lava) ---
// When g_heroMode is set (C# loaded a <model>_herodist.nif variant), ProcessNif captures the REAL
// per-subset NiUVData key groups + NiAlphaProperty blend/test state and appends them to a side file
// (hero_anim.data) opened by BeginHeroAnim. The side file is additive: absent => no hero anim, and
// every existing bake keeps loading byte-identically. g_staticOrdinal tracks the distant-static
// record index (one per successful nif.Save) so hero records join back to the host statics library.
static bool     g_heroMode = false;
static HANDLE   g_heroFile = 0;
static uint32_t g_staticOrdinal = 0;   // next static_meshes record index (reset in BeginStaticCreation)
static uint32_t g_heroRecordCount = 0; // backpatched into the hero_anim.data header at EndHeroAnim

// Sky-bake only (default false = real distant-land statics gen behaviour is byte-for-byte unchanged):
// when set, ExportShape accepts textureless / UV-less shapes (MW's vertex/material-coloured sky dome)
// instead of dropping them. Toggled by the SetAllowTexturelessShapes export around a sky bake.
static bool g_allowTextureless = false;

// --- Glow in the Dahrk day/night window variants ---
// GitD ships each window mesh as a NiSwitchNode with children named "off" / "on" / "int-day" and
// bakes indexActive = 0 = OFF into the file. SearchShapes used to resolve that switch with
// GetActiveChild(), so distant land froze on the UNLIT variant forever and a lit town dissolved
// into dark windows at the DL handoff. A subset now carries which variant it is; the host draws
// exactly one of the pair and clips the other (static_meshes flags[1] bits 1/2 -> subset flags
// bits 3/4). kVariantAll is every other subset in the game: no bits, drawn unconditionally.
enum SubsetVariant : unsigned char {
    kVariantAll   = 0,   // not part of a day/night pair — always drawn
    kVariantNight = 1,   // GitD "on" child: lit windows
    kVariantDay   = 2,   // the switch's active child: today's (unlit) appearance
};

// One shape found by SearchShapes, plus the variant of the subtree it was found in.
struct SubsetRef {
    NiTriBasedGeomRef geom;
    unsigned char     variant;
};

// Functions from OpenEXR to convert a float to a half float
static inline unsigned short FloatToHalfI(unsigned int i) {
    int s =  (i >> 16) & 0x00008000;
    int e = ((i >> 23) & 0x000000ff) - (127 - 15);
    int m =   i        & 0x007fffff;

    if (e <= 0) {
        if (e < -10) {
            return 0;
        }
        m = (m | 0x00800000) >> (1 - e);

        return s | (m >> 13);
    } else if (e == 0xff - (127 - 15)) {
        if (m == 0) { // Inf
            return s | 0x7c00;
        } else { // NAN
            m >>= 13;
            return s | 0x7c00 | m | (m == 0);
        }
    } else {
        if (e > 30) { // Overflow
            return s | 0x7c00;
        }

        return s | (e << 10) | (m >> 13);
    }
}

static inline unsigned short FloatToHalf(float i) {
    union {
        float f;
        unsigned int i;
    } v;
    v.f = i;
    return FloatToHalfI(v.i);
}

struct DXMatrix {
    float data[4*4];
};

// Per-subset hero animation, captured only in hero mode. Default-copyable (vectors), so ExportedNode
// operator= just assigns it. src/dst default to SRCALPHA/INVSRCALPHA, the MW ghostfence/lava blend.
struct HeroAnim {
    bool    hasAnim    = false;              // subset carries a NiUVController with keys
    bool    blend      = false;              // NiAlphaProperty blend enabled
    uint8_t src        = 6;                  // BlendFunc BF_SRC_ALPHA
    uint8_t dst        = 7;                  // BlendFunc BF_ONE_MINUS_SRC_ALPHA
    bool    test       = false;              // alpha test enabled
    uint8_t func       = 4;                  // TestFunc TF_GREATER
    uint8_t threshold  = 128;
    float   cycleStart = 0.0f, cycleStop = 0.0f;  // NiTimeController LOOP window
    std::vector<std::pair<float, float>> uKeys;   // (time, U offset)
    std::vector<std::pair<float, float>> vKeys;   // (time, V offset)
};

struct ExportedNode {
    Vector3 center;
    float radius;
    Vector3 max;
    Vector3 min;
    int verts;
    int faces;
    std::unique_ptr<DXVertex[]> vBuffer;
    std::unique_ptr<unsigned short[]> iBuffer;
    string tex;
    float emissive;
    bool alphaTestEnabled;
    bool alphaBlendEnabled;
    bool hasUVController;
    unsigned char variant;   // SubsetVariant — GitD day/night window pair member, or kVariantAll
    // Fixed-function stage op for a multi-map layer above the base: 0 = not a layer (the base, or
    // any ordinary single-texture subset), 1 = MOD, 2 = MOD2X, 3 = ADD. Mirrors the near path's
    // reconstruction exactly (scenegraph_geometry_cache.h: "MODULATE dark / MODULATE2X detail /
    // ADD glow") — using plain MOD for a grey-centred detail map halves the whole window, and
    // dropping the ADD glow layer is what made lit windows read as barely emissive.
    unsigned char layerOp;
    HeroAnim hero;

    ExportedNode() :
        center(0,0,0), radius(0), verts(0), faces(0), emissive(0),
        alphaTestEnabled(false), alphaBlendEnabled(false), hasUVController(false),
        variant(kVariantAll), layerOp(0) {
    }

    ExportedNode(const ExportedNode& src) :
        center(0,0,0), radius(0), verts(0), faces(0), emissive(0),
        alphaTestEnabled(false), alphaBlendEnabled(false), hasUVController(false),
        variant(kVariantAll), layerOp(0) {

        *this = src;
    }

    ExportedNode& operator=(const ExportedNode& src) {
        vBuffer.reset();
        iBuffer.reset();

        verts = src.verts;
        faces = src.faces;
        tex = src.tex;
        emissive = src.emissive;
        alphaTestEnabled = src.alphaTestEnabled;
        alphaBlendEnabled = src.alphaBlendEnabled;
        hasUVController = src.hasUVController;
        variant = src.variant;
        layerOp = src.layerOp;
        hero = src.hero;

        if (verts) {
            vBuffer = std::make_unique<DXVertex[]>(verts);
            memcpy(vBuffer.get(), src.vBuffer.get(), verts * sizeof(DXVertex));
        }

        if (faces) {
            iBuffer = std::make_unique<unsigned short[]>(faces * 3);
            memcpy(iBuffer.get(), src.iBuffer.get(), faces * 3 * sizeof(unsigned short));
        }

        return *this;
    }

    void CalcBounds() {
        // If the node has no vertices, give it default bounds
        if (verts == 0) {
            center.x = 0.0f;
            center.y = 0.0f;
            center.z = 0.0f;
            radius = 0.0f;
            return;
        }

        max = Vector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        min = Vector3(FLT_MAX, FLT_MAX, FLT_MAX);

        for (int v = 0; v < verts; ++v) {
            float x = vBuffer[v].Position.x;
            float y = vBuffer[v].Position.y;
            float z = vBuffer[v].Position.z;

            if (x > max.x) { max.x = x; }
            if (y > max.y) { max.y = y; }
            if (z > max.z) { max.z = z; }

            if (x < min.x) { min.x = x; }
            if (y < min.y) { min.y = y; }
            if (z < min.z) { min.z = z; }
        }

        // Store center of this node
        center = (min + max) / 2;

        // Find the furthest point from the center to get the radius
        float radius_squared = 0.0f;
        for (int v = 0; v < verts; ++v) {
            float x = vBuffer[v].Position.x;
            float y = vBuffer[v].Position.y;
            float z = vBuffer[v].Position.z;

            float dist_squared = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y) + (z-center.z)*(z-center.z);

            if (dist_squared > radius_squared) {
                radius_squared = dist_squared;
            }

        }

        // Store local radius of this node
        radius = sqrt(radius_squared);
    }

    void Optimize(unsigned int cache_size, float simplify) {
        const unsigned int stride = 36;

        // Reduce vertex count by scaling factor simplify
        if (simplify < 1 && faces > 8) {
            ProgMesh pmesh(verts, faces, vBuffer.get(), iBuffer.get());
            pmesh.ComputeProgressiveMesh();

            DXVertex* newVerts;
            WORD* newFaces;

            if (pmesh.DoProgressiveMesh(simplify, (DWORD*)&verts, (DWORD*)&faces, &newVerts, &newFaces) > 0) {
                vBuffer.reset(newVerts);
                iBuffer.reset(newFaces);

                /*char buf[260];
                sprintf(buf, "Mesh simplified from %d to %d", debugVerts, verts);
                OutputDebugStringA(buf);*/
            }
        }

        // Create temporary 32-bit index buffer
        size_t iBufferSize = faces * 3;
        vector<unsigned int> iBuffer32(iBufferSize);
        for (size_t j = 0; j < iBufferSize; ++j) {
            iBuffer32[j] = iBuffer[j];
        }

        TootleResult result = TootleOptimizeVCache(&*iBuffer32.begin(), faces, verts, cache_size, &*iBuffer32.begin(), NULL, TOOTLE_VCACHE_AUTO);

        if (result != TOOTLE_OK) {
            // log_file << "TootleOptimizeVCache returned an error" << endl;
            return;
        }

        result = TootleOptimizeVertexMemory(vBuffer.get(), &*iBuffer32.begin(), verts, faces, stride, vBuffer.get(), &*iBuffer32.begin(), NULL);

        if (result != TOOTLE_OK) {
            // log_file << "TootleOptimizeVertexMemory returned an error" << endl;
            return;
        }

        // Copy 32-bit index buffer back into 16-bit indices
        for (size_t j = 0; j < iBufferSize; ++j) {
            iBuffer[j] = (unsigned short)iBuffer32[j];
        }
    }


    void Save(HANDLE& file) {
        DWORD unused;

        // Write radius and center
        WriteFile(file, &radius, 4, &unused, 0);

        WriteFile(file, &center.x, 4, &unused, 0);
        WriteFile(file, &center.y, 4, &unused, 0);
        WriteFile(file, &center.z, 4, &unused, 0);

        // Write min and max (bounding box)
        WriteFile(file, &min.x, 4, &unused, 0);
        WriteFile(file, &min.y, 4, &unused, 0);
        WriteFile(file, &min.z, 4, &unused, 0);

        WriteFile(file, &max.x, 4, &unused, 0);
        WriteFile(file, &max.y, 4, &unused, 0);
        WriteFile(file, &max.z, 4, &unused, 0);

        // Write vert and face counts
        WriteFile(file, &verts, 4, &unused, 0);
        WriteFile(file, &faces, 4, &unused, 0);

        // Compress vertex buffer
        vector<DXCompressedVertex> compVBuf(verts);

        for (int i = 0; i < verts; ++i) {
            DXCompressedVertex& cv = compVBuf[i];
            DXVertex& v = vBuffer[i];

            // Copy uncompressed values
            cv.Diffuse[0] = v.Diffuse[0];
            cv.Diffuse[1] = v.Diffuse[1];
            cv.Diffuse[2] = v.Diffuse[2];
            cv.Diffuse[3] = v.Diffuse[3];

            // Compress position
            cv.Position[0] = FloatToHalf(v.Position.x);
            cv.Position[1] = FloatToHalf(v.Position.y);
            cv.Position[2] = FloatToHalf(v.Position.z);
            cv.Position[3] = FloatToHalf(1.0f);

            // Compress texcoords
            cv.texCoord[0] = FloatToHalf(v.texCoord.u);
            cv.texCoord[1] = FloatToHalf(v.texCoord.v);

            // Compress normals + emissive
            cv.Normal[0] = (unsigned char)(255.0f * (v.Normal.x * 0.5 + 0.5));
            cv.Normal[1] = (unsigned char)(255.0f * (v.Normal.y * 0.5 + 0.5));
            cv.Normal[2] = (unsigned char)(255.0f * (v.Normal.z * 0.5 + 0.5));
            cv.Normal[3] = (unsigned char)(255.0f * emissive);
        }

        // Write vertex and index buffers
        WriteFile(file, &*compVBuf.begin(), verts * sizeof(DXCompressedVertex), &unused, 0);
        WriteFile(file, iBuffer.get(), faces * 3 * sizeof(unsigned short), &unused, 0);

        // Write texturing flags. static_meshes has NO version or magic header — it is parsed
        // positionally — so the GitD day/night variant is encoded by WIDENING flags[1] (only ever
        // 0 or 1 before) instead of appending a field. An existing distant-land install therefore
        // reads bits 1/2 as 0 on every subset, i.e. unconditional, i.e. exactly today's behaviour;
        // regenerating is what opts in. flags[0] stays a plain bool.
        unsigned char flags[2];
        flags[0] = (alphaTestEnabled || alphaBlendEnabled) ? 1 : 0;
        flags[1] = (unsigned char)((hasUVController          ? 0x1 : 0)    // bit0: NiUVController
                                 | (variant == kVariantNight ? 0x2 : 0)    // bit1: night-only variant
                                 | (variant == kVariantDay   ? 0x4 : 0)    // bit2: day-only variant
                                 | ((layerOp & 0x3)         << 3));      // bits3-4: layer op (1 MOD, 2 MOD2X, 3 ADD)
        WriteFile(file, &flags, 2, &unused, 0);

        // Write texture name
        unsigned short slen = (unsigned short)tex.size() + 1;
        WriteFile(file, &slen, 2, &unused, 0);
        WriteFile(file, tex.c_str(), slen, &unused, 0);
    }
};

enum StaticType {
    STATIC_AUTO = 0,
    STATIC_NEAR = 1,
    STATIC_FAR = 2,
    STATIC_VERY_FAR = 3,
    STATIC_GRASS = 4,
    STATIC_TREE = 5,
    STATIC_BUILDING = 6
};

class ExportedNif {
public:
    Vector3 center;
    float radius;
    unsigned char static_type;
    vector<ExportedNode> nodes;

    void CalcBounds() {
        // Calculate the total bounds of all nodes
        Vector3 max = Vector3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        Vector3 min = Vector3(FLT_MAX, FLT_MAX, FLT_MAX);

        // Find minimum and maximum x, y, and z positions
        for (size_t n = 0; n < nodes.size(); ++n) {
            // If the node has no vertices, give it default bounds
            if (nodes[n].verts == 0) {
                nodes[n].center.x = 0.0f;
                nodes[n].center.y = 0.0f;
                nodes[n].center.z = 0.0f;
                nodes[n].radius = 0.0f;
                continue;
            }

            for (int v = 0; v < nodes[n].verts; ++v) {
                float x, y, z;
                x = nodes[n].vBuffer[v].Position.x;
                y = nodes[n].vBuffer[v].Position.y;
                z = nodes[n].vBuffer[v].Position.z;

                if (x > max.x) { max.x = x; }
                if (y > max.y) { max.y = y; }
                if (z > max.z) { max.z = z; }

                if (x < min.x) { min.x = x; }
                if (y < min.y) { min.y = y; }
                if (z < min.z) { min.z = z; }
            }
        }

        // Average min/max positions to get center
        center = (min + max) / 2;

        // Find the furthest point from the center to get the radius
        float radius_squared = 0.0f;
        for (size_t n = 0; n < nodes.size(); ++n) {
            for (int v = 0; v < nodes[n].verts; ++v) {
                float x = nodes[n].vBuffer[v].Position.x;
                float y = nodes[n].vBuffer[v].Position.y;
                float z = nodes[n].vBuffer[v].Position.z;

                float dist_squared = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y) + (z-center.z)*(z-center.z);

                if (dist_squared > radius_squared) {
                    radius_squared = dist_squared;
                }
            }
        }

        radius = sqrt(radius_squared);
    }

    void CalcNodeBounds() {
        for (size_t i = 0; i < nodes.size(); ++i) {
            nodes[i].CalcBounds();
        }
    }

private:

    bool MergeShape(ExportedNode* dst, ExportedNode* src) {
        // Sum vert and face counts
        int verts = src->verts + dst->verts;
        int faces = src->faces + dst->faces;

        // Create new buffers large enough to hold all vertices from original ones
        auto v_buf = std::make_unique<DXVertex[]>(verts);
        auto i_buf = std::make_unique<unsigned short[]>(faces*3);

        // Copy data from previous buffers into new ones
        memcpy(v_buf.get(), dst->vBuffer.get(), dst->verts * sizeof(DXVertex));
        memcpy(v_buf.get() + dst->verts, src->vBuffer.get(), src->verts * sizeof(DXVertex));

        memcpy(i_buf.get(), dst->iBuffer.get(), dst->faces * 3 * sizeof(unsigned short));
        memcpy(i_buf.get() + dst->faces * 3, src->iBuffer.get(), src->faces * 3 * sizeof(unsigned short));

        // Account for the offset in the indices copied from src
        for (int i = dst->faces * 3; i < faces * 3; ++i) {
            i_buf[i] += dst->verts;
        }

        // Set new values in dst
        dst->vBuffer.swap(v_buf);
        dst->iBuffer.swap(i_buf);
        dst->verts = verts;
        dst->faces = faces;

        return true;
    }

    void SearchShapes(NiAVObjectRef rootObj, vector<SubsetRef>* SubsetNodes, unsigned char variant) {
        // Exclude hidden objects
        if (!rootObj->GetVisibility()) {
            return;
        }

        // Check if this object is derived from NiTriBasedGeom
        NiTriBasedGeomRef niGeom = DynamicCast<NiTriBasedGeom>(rootObj);
        if (niGeom) {
            SubsetRef ref = { niGeom, variant };
            SubsetNodes->push_back(ref);
            return;
        }

        // Check if this object derives from NiNode and, thus, may have children
        // Select appropriate LOD for NiLODNodes, switch index for NiSwitchNodes, ignore RootCollisionNodes
        NiNodeRef niNode = DynamicCast<NiNode>(rootObj);
        if (niNode) {
            const auto children = niNode->GetChildren();
            NiSwitchNodeRef niSwitch = DynamicCast<NiSwitchNode>(rootObj);
            NiLODNodeRef lod = DynamicCast<NiLODNode>(rootObj);
            RootCollisionNodeRef collision = DynamicCast<RootCollisionNode>(rootObj);

            if (lod) {
                // Pick LOD level with 1 cell equivalent distance, which may result in no node selected
                const float lodDist = 8192.0f;
                const auto levels = lod->GetLODLevels();
                int index = -1;

                for (int i = 0; i < levels.size(); ++i) {
                    if (lodDist >= levels[i].nearExtent && lodDist < levels[i].farExtent) {
                        index = i;
                        break;
                    }
                }

                if (index >= 0 && index < children.size()) {
                    SearchShapes(children[index], SubsetNodes, variant);
                }
            } else if (niSwitch) {
                // Glow in the Dahrk day/night windows. GetActiveChild() is children[indexActive]
                // straight from the file, and every GitD mesh ships indexActive = 0 = "off", so the
                // bake used to freeze distant land on the unlit variant permanently — the lit
                // geometry and its glow-atlas UVs were never in the LOD at all.
                //
                // When a child named "on" exists (GitD's own convention, the same key the mod keys
                // on), export BOTH: the active child tagged day-only and "on" tagged night-only.
                // They share vertex POSITIONS but not UVs ("on" indexes the glow atlas), so they
                // cannot share a vertex buffer — the lit variant is its own subset, ~1 KB per mesh
                // DEFINITION shared by every instance of it.
                //
                // Any other use of NiSwitchNode, and any switch nested inside an already-tagged
                // subtree, keeps the old single-active-child behaviour (fail-safe direction: no
                // night variant, behaves exactly as today).
                //
                // NEVER split a HERO mesh. The ghostfence _herodist.nif carries its own
                // NightDaySwitch with OFF/ON children, but there the two sides are not the
                // day/night alternatives this split assumes — hero subsets are captured
                // as-is and joined back to hero_anim.data BY ORDINAL, so splitting both
                // shifted every anim slot and clipped half the layers, and the fence lost
                // its glow. Hero mode is the whole "capture every layer exactly" contract;
                // it opts out of this feature entirely.
                auto child = niSwitch->GetActiveChild();
                NiAVObjectRef onChild;
                if (variant == kVariantAll && !g_heroMode) {
                    for (auto c : children) {
                        if (c && _stricmp(c->GetName().c_str(), "on") == 0) {
                            onChild = c;
                            break;
                        }
                    }
                }

                if (onChild && onChild != child) {
                    const size_t dayFirst = SubsetNodes->size();
                    if (child) {
                        SearchShapes(child, SubsetNodes, kVariantDay);
                    }
                    const size_t nightFirst = SubsetNodes->size();
                    SearchShapes(onChild, SubsetNodes, kVariantNight);

                    // A day/night split is only safe if BOTH halves exist: the host clips the day
                    // subsets after dark, so tagging them while the "on" subtree yielded nothing
                    // (hidden, or no geometry under it) would make the windows VANISH at night
                    // rather than merely not glow. Demote the pair back to unconditional.
                    if (SubsetNodes->size() == nightFirst) {
                        for (size_t i = dayFirst; i < nightFirst; ++i) {
                            (*SubsetNodes)[i].variant = variant;
                        }
                    }
                } else if (child) {
                    SearchShapes(child, SubsetNodes, variant);
                }
            } else if (!collision) {
                // Call this function for any children
                for (auto child : children) {
                    SearchShapes(child, SubsetNodes, variant);
                }
            }
        }
    }

    static NiPropertyRef ResolveProperty(NiAVObjectRef obj, const Niflib::Type& type) {
        // Check immediate object for the property
        NiPropertyRef prop = obj->GetPropertyByType(type);
        if (prop) {
            return prop;
        }

        // Check up the hierarchy to see if any property is inherited
        auto parent = DynamicCast<NiAVObject>(obj->GetParent());
        return parent ? ResolveProperty(parent, type) : nullptr;
    }

    // The populated texture layers of a shape's material, in fixed-function stage order, with the
    // stage op each one combines with. This is the SAME reconstruction the near path uses
    // (scenegraph_geometry_cache.h: "MODULATE dark / MODULATE2X detail / ADD glow"), so a lit window
    // reads identically either side of the DL handoff — which is the entire point. Getting these
    // wrong is visible: MOD on the grey-centred detail map halves the window, and omitting the
    // additive glow layer is what left lit windows looking barely emissive.
    // Returns the layer count; out[] = TexType slot, ops[] = 0 base / 1 MOD / 2 MOD2X / 3 ADD.
    static const int kMaxLayers = 4;
    int LayerSlots(NiTriBasedGeomRef niGeom, int out[kMaxLayers], unsigned char ops[kMaxLayers]) {
        NiAVObjectRef asAVObject = DynamicCast<NiAVObject>(niGeom);
        NiTexturingPropertyRef tp =
            DynamicCast<NiTexturingProperty>(ResolveProperty(asAVObject, NiTexturingProperty::TYPE));
        int n = 0;
        if (!tp) { return 0; }
        const int           order[kMaxLayers] = { BASE_MAP, DARK_MAP, DETAIL_MAP, GLOW_MAP };
        const unsigned char op[kMaxLayers]    = { 0,        1,        2,          3        };
        for (int i = 0; i < kMaxLayers; ++i) {
            const int s = order[i];
            if (s < tp->GetTextureCount() && tp->HasTexture(s)
                && tp->GetTexture(s).source && tp->GetTexture(s).source->IsTextureExternal()) {
                out[n] = s;
                ops[n] = op[i];
                ++n;
            }
        }
        // A material with no BASE but e.g. a glow map would make the first layer an ADD onto nothing.
        // Demote whatever comes first to the base so there is always something to combine onto.
        if (n > 0) { ops[0] = 0; }
        return n;
    }

    bool ExportShape(NiTriBasedGeomRef niGeom, ExportedNode* node, int forceSlot = -1) {
        // Resolve property inheritance
        NiAVObjectRef asAVObject = DynamicCast<NiAVObject>(niGeom);
        NiTexturingPropertyRef niTexProp = DynamicCast<NiTexturingProperty>(ResolveProperty(asAVObject, NiTexturingProperty::TYPE));
        NiAlphaPropertyRef niAlphaProp = DynamicCast<NiAlphaProperty>(ResolveProperty(asAVObject, NiAlphaProperty::TYPE));
        NiMaterialPropertyRef niMatProp = DynamicCast<NiMaterialProperty>(ResolveProperty(asAVObject, NiMaterialProperty::TYPE));

        // Which texture slot to bake. Distant land is single-texture, single-UV-set, so it takes
        // the BASE map — except on a Glow in the Dahrk LIT variant, which is a MULTI-MAP material
        // whose three layers each carry a different part of the look, on their own UV set:
        //   in_redoran_window_01 "on": base=glow\tex02_dark1 (uv0)  — a near-neutral glow sheet
        //                              dark=glow\tex03_2      (uv1)  — THE COLOUR (the orange)
        //                              detail=tx_glass_amber_02 (uv2) — the glass artwork
        // No ONE layer is enough — measured on real scenes, each carries a different part:
        //   base   -> "just white windows"                (the sheet is nearly colourless)
        //   detail -> "less white, but still white"; ex_vivec_c_04 "very close, slight hue change"
        //   dark   -> the tint the other two are missing
        // and per the author, "sometimes just dark is enough, sometimes you need 2, sometimes all 3".
        // So the bake replicates the material: LayerSlots() enumerates the populated layers and the
        // caller exports ONE SUBSET PER LAYER, each with its own texture AND its own UV set (the sets
        // differ on purpose — verified max delta 0.68 between sets on in_redoran_window_01, so they
        // cannot be composited offline). forceSlot picks which layer THIS node bakes; layers after
        // the first carry multiplyLayer, and the host multiplies them onto the one beneath.
        // forceSlot < 0 = the default single-texture behaviour every other static in the game gets.
        const int texSlot = (forceSlot >= 0) ? forceSlot : 0;

        // Check that an external texture exists. Textureless shapes are dropped, EXCEPT during a sky
        // bake (g_allowTextureless) where the MW atmosphere dome is deliberately texture-free.
        NiSourceTextureRef niSrcTex;
        bool hasTexture = false;
        unsigned int texUVSet = 0;
        if (niTexProp && niTexProp->GetTextureCount() > texSlot) {
            TexDesc texDesc = niTexProp->GetTexture(texSlot);
            niSrcTex = texDesc.source;
            texUVSet = texDesc.uvSet;
            hasTexture = (niSrcTex && niSrcTex->IsTextureExternal());
        }
        if (!hasTexture && !g_allowTextureless) {
            // log_file << "External texture does not exist" << endl;
            return false;
        }

        // Get data object (NiTriBasedGeomData) from geometry node
        NiTriBasedGeomDataRef niGeomData = DynamicCast<NiTriBasedGeomData>(niGeom->GetData());
        if (!niGeomData) {
            // log_file << "There is no Geometry data on this mesh." << endl;
            return false;
        }

        // Indices
        vector<Triangle> tris = niGeomData->GetTriangles();
        node->faces = tris.size();
        if (node->faces == 0) {
            // log_file << "This mesh has no triangles." << endl;
            return false;
        }

        // Check that there is at least one set of texture coords available. As with the texture check,
        // the sky dome legitimately has none, so tolerate that during a sky bake (UVs default to 0).
        // The baked UV set follows the baked texture SLOT (see texSlot above): a lit GitD window
        // indexes its artwork through uv set 2, so taking set 0 there would map the glow sheet's
        // coordinates onto the window art. Falls back to set 0 if the shape doesn't carry that set.
        bool hasUV = niGeomData->GetUVSetCount() > 0;
        if (texUVSet >= (unsigned int)niGeomData->GetUVSetCount()) {
            texUVSet = 0;
        }
        if (!hasUV && !g_allowTextureless) {
            // log_file << "There are no texture coordinates on this mesh." << endl;
            return false;
        }

        // alpha prop -> flag alpha test, alpha blend
        if (niAlphaProp) {
            node->alphaTestEnabled = niAlphaProp->GetTestState();
            node->alphaBlendEnabled = niAlphaProp->GetBlendState();
            // Hero mode captures the full blend/test state per subset (the legacy bake collapses both
            // to one bool; the host would then draw a translucent fence as an opaque cutout).
            if (g_heroMode) {
                node->hero.blend     = niAlphaProp->GetBlendState();
                node->hero.src       = (uint8_t)niAlphaProp->GetSourceBlendFunc();
                node->hero.dst       = (uint8_t)niAlphaProp->GetDestBlendFunc();
                node->hero.test      = niAlphaProp->GetTestState();
                node->hero.func      = (uint8_t)niAlphaProp->GetTestFunc();
                node->hero.threshold = niAlphaProp->GetTestThreshold();
            }
        }

        // Get diffuse color (will be baked into vertices)
        // Get the emissive color (will be averaged and stored in the 4th channel of normals)
        // Get the alpha of the material
        Color3 diffuse(1.0f, 1.0f, 1.0f);
        Color3 emissive(0.0f, 0.0f, 0.0f);
        float alpha = 1.0f;

        if (niMatProp) {
            diffuse = niMatProp->GetDiffuseColor();
            emissive = niMatProp->GetEmissiveColor();
            alpha = niMatProp->GetTransparency();
        }
        node->emissive = (emissive.r + emissive.b + emissive.g) / 3.0f;

        // Check for UV controller and extra data, to flag for special rendering
        const char *specialTag = "mge.distant.scroll";
        bool detectedUVAnim = false;
        NiUVControllerRef uvCtrl = NULL;

        if (niGeom->IsAnimated()) {
            for (auto& c : niGeom->GetControllers()) {
                NiUVControllerRef uvc = DynamicCast<NiUVController>(c);
                if (uvc) {
                    detectedUVAnim = true;
                    uvCtrl = uvc;
                    break;
                }
            }
        }
        if (g_heroMode) {
            // Hero path: capture the REAL NiUVData key groups (0=U off, 1=V off) + LOOP window. No tag
            // required (the _herodist copy IS the opt-in). hasUVController stays FALSE so the host does
            // NOT also apply the legacy 0.08 V-scroll; hero anim rides its own slot from hero_anim.data.
            if (uvCtrl) {
                NiUVDataRef uvData = uvCtrl->GetData();
                if (uvData) {
                    vector<KeyGroup<float>> groups = uvData->GetUVGroups();
                    for (const auto& k : groups[0].keys) { node->hero.uKeys.push_back(std::make_pair(k.time, k.data)); }
                    for (const auto& k : groups[1].keys) { node->hero.vKeys.push_back(std::make_pair(k.time, k.data)); }
                }
                node->hero.cycleStart = uvCtrl->GetStartTime();
                node->hero.cycleStop  = uvCtrl->GetStopTime();
                node->hero.hasAnim    = !node->hero.uKeys.empty() || !node->hero.vKeys.empty();
            }
        } else if (detectedUVAnim) {
            // Legacy path: only the hand-tagged _dist stand-ins get the single-layer V scroll.
            for (auto& extra : niGeom->GetExtraData()) {
                auto extraString = DynamicCast<NiStringExtraData>(extra);
                if (extraString && extraString->GetData() == specialTag) {
                    node->hasUVController = true;
                }
            }
        }

        // Now that we're sure this mesh is valid, start the conversion

        // Get transformation of mesh as 4x4 matrix
        Matrix44 transform = niGeom->GetWorldTransform();

        // Get a matrix that only contains the world rotation matrix
        // This will be used to transform normals
        Matrix44 rotation(transform.GetRotation());

        // Vertex data
        vector<Vector3> positions;
        vector<Vector3> normals;
        if (niGeom->IsSkin()) {
            niGeom->GetSkinDeformation(positions, normals);
        } else {
            positions = niGeomData->GetVertices();
            normals = niGeomData->GetNormals();
        }
        vector<Color4> colors = niGeomData->GetColors();
        vector<TexCoord> texCoords;
        if (hasUV) {
            texCoords = niGeomData->GetUVSet(texUVSet);
        }

        // Vertices
        bool hasNormals = normals.size() > 0;
        bool hasColors = colors.size() > 0;
        node->verts = niGeomData->GetVertexCount();
        node->vBuffer = std::make_unique<DXVertex[]>(node->verts);

        for (int i = 0; i < node->verts; i++) {
            // Push the world transform into the vertices
            // Apply the world transform's rotation to the normals
            node->vBuffer[i].Position = transform * positions[i];

            if (hasNormals) {
                node->vBuffer[i].Normal = rotation * normals[i];
            } else {
                node->vBuffer[i].Normal.x = 0;
                node->vBuffer[i].Normal.y = 0;
                node->vBuffer[i].Normal.z = 1;
            }
            if (hasColors) {
                // Diffuse/ambient vertex material source
                node->vBuffer[i].Diffuse[0] = (unsigned char)(255.0f * colors[i].b);
                node->vBuffer[i].Diffuse[1] = (unsigned char)(255.0f * colors[i].g);
                node->vBuffer[i].Diffuse[2] = (unsigned char)(255.0f * colors[i].r);
                node->vBuffer[i].Diffuse[3] = (unsigned char)(255.0f * colors[i].a);
            } else {
                // Use material property
                node->vBuffer[i].Diffuse[0] = (unsigned char)(255.0f * diffuse.b);
                node->vBuffer[i].Diffuse[1] = (unsigned char)(255.0f * diffuse.g);
                node->vBuffer[i].Diffuse[2] = (unsigned char)(255.0f * diffuse.r);
                node->vBuffer[i].Diffuse[3] = (unsigned char)(255.0f * alpha);
            }

            if (hasUV) {
                node->vBuffer[i].texCoord = texCoords[i];
            } else {
                node->vBuffer[i].texCoord.u = 0.0f;
                node->vBuffer[i].texCoord.v = 0.0f;
            }
        }

        // Write index buffer
        node->iBuffer = std::make_unique<unsigned short[]>(node->faces*3);
        for (int i = 0; i < node->faces; i++) {
            node->iBuffer[i*3+0] = tris[i].v1;
            node->iBuffer[i*3+1] = tris[i].v2;
            node->iBuffer[i*3+2] = tris[i].v3;
        }

        // Get texture file path (textureless sky dome keeps an empty path)
        string s;
        if (hasTexture) {
            s = niSrcTex->GetTextureFileName();

            // Make texture path all lowercase
            for (size_t i = 0; i < s.size(); ++i) {
                if (s[i] >= 'A' && s[i] <= 'Z') {
                    s[i] += 32;
                }
            }

            // If the path starts with "textures" or "\textures" remove it
            size_t pos = s.find("textures");
            if (pos == 0 || (pos == 1 && s[0] == '\\')) {
                s = s.substr(pos + 8, string::npos);
            }

            // Remove any leading backslashes that remain
            if (!s.empty() && s[0] == '\\') {
                s = s.substr(1, string::npos);
            }
        }

        node->tex = s;

        return true;
    }


public:

    void Optimize(unsigned int cache_size, float simplify) {
        // Try to combine nodes that have the same texture path
        map<string, ExportedNode*> node_tex;

        // Hero meshes must NOT merge by texture: the ghostfence layers a "faster" and a "slower"
        // NiTriShape over the SAME tx_gg_fence_01 texture, and the interference between them at 2:1
        // scroll speeds IS the effect. Merging would collapse them to one subset (one UV offset) and
        // kill it. Keep every subset distinct so each carries its own controller + alpha state.
        if (!g_heroMode) {
        for (size_t i = 0; i < nodes.size(); ++i) {
            // GitD day/night window variants are ALTERNATIVES, not layers, and they can share a
            // texture (the "on" child often just indexes a glow region of the same atlas). Merging
            // them would collapse the pair into one always-drawn subset and lose the whole feature,
            // so the merge key carries the variant as well as the texture path.
            //
            // NIGHT variants are never merged at all: each is one LAYER of a multi-map material,
            // carrying its own UV set, and two layers can even share a texture (ray_alpha appears as
            // both base and glow). Merging would concatenate two different UV parameterisations
            // under one texture. The key is a zero-padded ordinal so it is unique AND sorts in
            // emission order — the merged-node rebuild below iterates the map by key, so a
            // non-ordered key would reshuffle layers away from their base.
            char okey[24];
            string key;
            if (nodes[i].variant == kVariantNight) {
                sprintf_s(okey, "1%08zu", i);
                key = okey;
            } else {
                key = string(1, (char)('0' + nodes[i].variant)) + nodes[i].tex;
            }

            // Check if this node has already been found
            map<string, ExportedNode*>::iterator it = node_tex.find(key);

            if (it == node_tex.end()) {
                // Nothing with this texture has been found yet.  Store the node's pointer in the map
                node_tex[key] = &nodes[i];
            } else {
                // A shape with this texture has been found already.  Merge this one into it.
                MergeShape(it->second, &nodes[i]);
            }
        }
        }

        size_t count = 0;
        if (node_tex.size() < nodes.size() && node_tex.size() != 0) {
            // We reduced the number of nodes, so create a new list to save
            vector<ExportedNode> merged_nodes(node_tex.size());

            for (map<string, ExportedNode*>::iterator it = node_tex.begin(); it != node_tex.end(); ++it) {
                merged_nodes[count] = *(it->second);
                ++count;
            }

            nodes = merged_nodes;
        }


        // Now optimize each node
        for (size_t i = 0; i < nodes.size(); ++i) {
            nodes[i].Optimize(cache_size, simplify);
        }
    }

    bool LoadNifFromStream(const char* data, int size) {
        istrstream s(data, size);
        NiAVObjectRef rootObj;

        try {
            rootObj = DynamicCast<NiAVObject>(ReadNifTree(s, 0));
        } catch (std::runtime_error& e) {
            std::fstream error_log("mge3\\distant-land-niflib-error.log", std::ios_base::out | std::ios_base::app);
            error_log << e.what() << std::endl;
            return false;
        }

        if (!rootObj) {
            // log_file << "Root object was null." << endl;
            return false;
        }

        // Object root transform should not affect results
        rootObj->SetLocalTransform(Matrix44::IDENTITY);

        vector<SubsetRef> SubsetNodes;
        SearchShapes(rootObj, &SubsetNodes, kVariantAll);

        if (SubsetNodes.size() == 0) {
            // log_file << "SubsetNodes size is zero." << endl;
            return false;
        }

        for (size_t i = 0; i < SubsetNodes.size(); ++i) {
            const unsigned char variant = SubsetNodes[i].variant;

            // A GitD LIT window is a multi-map material and no single layer reproduces it, so
            // replicate it: one subset per populated layer, each carrying its own texture and its
            // own UV set, multiplied back together at draw time. Every other subset in the game
            // (including the unlit DAY variant, which is single-map) takes the one-texture path.
            int           layers[kMaxLayers];
            unsigned char layerOps[kMaxLayers] = { 0, 0, 0, 0 };
            int layerCount = (variant == kVariantNight)
                           ? LayerSlots(SubsetNodes[i].geom, layers, layerOps) : 0;
            if (layerCount <= 1) {
                layers[0] = -1;       // default: base map, ExportShape's own resolution
                layerOps[0] = 0;
                layerCount = 1;
            }

            for (int L = 0; L < layerCount; ++L) {
                ExportedNode tmp_node;
                // Set BEFORE the export: ExportShape reads variant for the layer semantics and
                // never writes the field back.
                tmp_node.variant = variant;
                if (!ExportShape(SubsetNodes[i].geom, &tmp_node, layers[L])) { continue; }
                // Animations are the one thing distant land gives up here (the author's call:
                // "only sacrifice the animations. herodists keep the animations"). Hero meshes
                // never reach this path at all — g_heroMode opts out of the variant split — so
                // the ghostfence/lava NiUVController replay is untouched.
                if (variant == kVariantNight) {
                    tmp_node.hasUVController = false;
                }
                // Layer 0 is the base (op 0); each layer above it folds onto what is already there
                // with its own fixed-function op — MOD for the dark map, MOD2X for the detail map,
                // ADD for the glow map.
                tmp_node.layerOp = layerOps[L];
                nodes.push_back(tmp_node);
            }
        }

        if (nodes.size() == 0) {
            // log_file << "nodes size is zero." << endl;
            return false;
        }

        // The other half of the "both halves or neither" rule above. SearchShapes only collects
        // CANDIDATE shapes; ExportShape still drops any that have no external texture, no triangles
        // or no UVs, so the night variant can disappear after the switch was already split. If none
        // survived, no subset may stay day-only — the windows would go missing after dark. (Per NIF
        // rather than per switch node: GitD meshes carry exactly one day/night switch.)
        bool haveNight = false;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (nodes[i].variant == kVariantNight) { haveNight = true; break; }
        }
        if (!haveNight) {
            for (size_t i = 0; i < nodes.size(); ++i) {
                if (nodes[i].variant == kVariantDay) { nodes[i].variant = kVariantAll; }
            }
        }

        // Success
        return true;
    }

    bool Save() {
        // HANDLE h = CreateFileA(outpath, GENERIC_WRITE, 0, 0, CREATE_ALWAYS, 0, 0);
        HANDLE h = staticFile;

        DWORD unused;
        if (h == INVALID_HANDLE_VALUE) {
            // log_file << "File handle is invalid." << endl;
            return false;
        }
        int nodeCount = (int)nodes.size();
        SetFilePointer(h, 0, NULL, FILE_END);

        WriteFile(h, &nodeCount, 4, &unused, 0);
        WriteFile(h, &radius, 4, &unused, 0);
        WriteFile(h, &center.x, 4, &unused, 0 );
        WriteFile(h, &center.y, 4, &unused, 0 );
        WriteFile(h, &center.z, 4, &unused, 0 );
        WriteFile(h, &static_type, 1, &unused, 0);

        for (size_t i = 0; i < nodes.size(); i++) {
            nodes[i].Save(h);
        }

        return true;
    }
};



extern "C" int __stdcall GetVertSize() {
    return (int)sizeof(DXVertex);
}

extern "C" int __stdcall GetCompressedVertSize() {
    return (int)sizeof(DXCompressedVertex);
}

extern "C" int __stdcall GetLandVertSize() {
    return (int)sizeof(DXCompressedLandVertex);
}

extern "C" float __stdcall ProcessNif(char* data, int datasize, float simplify, float cutoff, BYTE static_type) {

    // Load the NIF data into our DirectX-friendly format
    ExportedNif nif;
    if (!nif.LoadNifFromStream(data, datasize)) {
        // log_file << "LoadNifFromStream failed." << endl;
        return -1;
    }

    // Calculate the bounds of the NIF to determine whether it exceeds our cutoff value
    nif.CalcBounds();

    if (static_type == STATIC_AUTO && nif.radius < cutoff) {
        // log_file << "Radius was below cutoff value." << endl;
        return -2;
    }

    // Buildings are treated as if they are twice their actual size.
    if (static_type == STATIC_BUILDING && nif.radius * 2.0f < cutoff) {
        // log_file << "Radius was below cutoff value." << endl;
        return -2;
    }

    if (!staticFile) {
        return nif.radius;
    }

    // Optimize NIF and calculate node bounds
    nif.Optimize(16, simplify);
    nif.CalcNodeBounds();

    // Determine whether this will be a near or far distant static based on size
    nif.static_type = static_type;

    // Save NIF to new format
    if (!nif.Save()) {
        // log_file << "NIF Save failed." << endl;
        return -3;
    }

    // This static's index in static_meshes == the host statics-library index. Increment for EVERY
    // successful Save (hero or not) so hero records join back to the right library static.
    uint32_t thisStatic = g_staticOrdinal++;

    // Hero mode: append this static's animated / blended subsets to hero_anim.data. Subset index i
    // matches the host's subset order because hero mode skips the texture merge (subsets stay 1:1).
    if (g_heroMode && g_heroFile) {
        DWORD unused;
        for (size_t i = 0; i < nif.nodes.size(); ++i) {
            const HeroAnim& h = nif.nodes[i].hero;
            if (!h.hasAnim && !h.blend) { continue; }   // only subsets the host needs to know about
            uint16_t subset = (uint16_t)i;
            uint8_t  blend = h.blend ? 1 : 0;
            uint8_t  test  = h.test  ? 1 : 0;
            uint8_t  nU = h.uKeys.size() > 255 ? 255 : (uint8_t)h.uKeys.size();
            uint8_t  nV = h.vKeys.size() > 255 ? 255 : (uint8_t)h.vKeys.size();
            WriteFile(g_heroFile, &thisStatic, 4, &unused, 0);
            WriteFile(g_heroFile, &subset, 2, &unused, 0);
            WriteFile(g_heroFile, &blend, 1, &unused, 0);
            WriteFile(g_heroFile, &h.src, 1, &unused, 0);
            WriteFile(g_heroFile, &h.dst, 1, &unused, 0);
            WriteFile(g_heroFile, &test, 1, &unused, 0);
            WriteFile(g_heroFile, &h.func, 1, &unused, 0);
            WriteFile(g_heroFile, &h.threshold, 1, &unused, 0);
            WriteFile(g_heroFile, &h.cycleStart, 4, &unused, 0);
            WriteFile(g_heroFile, &h.cycleStop, 4, &unused, 0);
            WriteFile(g_heroFile, &nU, 1, &unused, 0);
            WriteFile(g_heroFile, &nV, 1, &unused, 0);
            for (uint8_t k = 0; k < nU; ++k) {
                float t = h.uKeys[k].first, v = h.uKeys[k].second;
                WriteFile(g_heroFile, &t, 4, &unused, 0);
                WriteFile(g_heroFile, &v, 4, &unused, 0);
            }
            for (uint8_t k = 0; k < nV; ++k) {
                float t = h.vKeys[k].first, v = h.vKeys[k].second;
                WriteFile(g_heroFile, &t, 4, &unused, 0);
                WriteFile(g_heroFile, &v, 4, &unused, 0);
            }
            ++g_heroRecordCount;
        }
    }

    return nif.radius;
}

extern "C" void __stdcall BeginStaticCreation(IDirect3DDevice9* _device, char* outpath) {
    device = _device;
    g_staticOrdinal = 0;   // fresh library index run; hero records join against it
    if (outpath) {
        staticFile = CreateFileA(outpath, FILE_GENERIC_WRITE, 0, 0, CREATE_ALWAYS, 0, 0);
    } else {
        staticFile = 0;
    }
}

// --- Hero animation side file (MGE XE mod) ---------------------------------------------------
// SetHeroMode toggles per-NIF capture; the C# driver sets it right before ProcessNif when it loaded
// a <model>_herodist.nif variant, and clears it after. BeginHeroAnim/EndHeroAnim bracket the whole
// statics run (call once, around BeginStaticCreation/EndStaticCreation).
extern "C" void __stdcall SetHeroMode(int on) {
    g_heroMode = (on != 0);
}

extern "C" void __stdcall BeginHeroAnim(char* outpath) {
    g_heroRecordCount = 0;
    g_heroFile = 0;
    if (outpath) {
        HANDLE h = CreateFileA(outpath, FILE_GENERIC_WRITE, 0, 0, CREATE_ALWAYS, 0, 0);
        if (h != INVALID_HANDLE_VALUE) {
            g_heroFile = h;
            DWORD unused;
            char     magic[4] = { 'M', 'G', 'H', 'A' };
            uint32_t version  = 1;
            uint32_t count    = 0;   // placeholder, backpatched in EndHeroAnim
            WriteFile(g_heroFile, magic, 4, &unused, 0);
            WriteFile(g_heroFile, &version, 4, &unused, 0);
            WriteFile(g_heroFile, &count, 4, &unused, 0);
        }
    }
}

extern "C" void __stdcall EndHeroAnim() {
    if (g_heroFile) {
        SetFilePointer(g_heroFile, 8, NULL, FILE_BEGIN);   // recordCount field (after magic+version)
        DWORD unused;
        WriteFile(g_heroFile, &g_heroRecordCount, 4, &unused, 0);
        CloseHandle(g_heroFile);
    }
    g_heroFile = 0;
}

extern "C" void __stdcall EndStaticCreation() {
    CloseHandle(staticFile);
}

// Sky-bake toggle: when on, ProcessNif accepts textureless / UV-less shapes (the MW atmosphere dome).
// Default off, so ordinary distant-land statics generation is completely unaffected.
extern "C" void __stdcall SetAllowTexturelessShapes(int on) {
    g_allowTextureless = (on != 0);
}
