#pragma once

// Wire format for the M1 opaque-geometry seam (32-bit MW cache -> 64-bit Forge host).
// Deliberately free of STL and D3D9/D3D12 headers so BOTH the host TU (forgerender.cpp,
// which keeps STL/D3D out for Forge ABI reasons) and the 32-bit client can include it.
//
// A geometry upload batch is a flat byte blob shipped through a shared byte vector:
//   [GeomPartWire][verts: GeomVertexWire * vertexCount][indices: uint16 * indexCount]
//   ... repeated partCount times (partCount carried in the RPC params).
// Parts are slot-indexed: the client assigns each cached NiTriShape* a dense slot, and
// the host stores meshes in a flat array indexed by that slot (no host-side hashing).
// Per-frame draw lists (M1c) reference the same slots.

#include <cstdint>

namespace IPC {

#pragma pack(push, 4)

    // Model-space vertex: position + normal + base-map UV (set e.baseUV) + per-vertex
    // colour. 36 bytes. (Texturing Phase 1 added u,v; Tier 2a lighting added color.)
    // `color` is a packed D3DCOLOR (B,G,R,A byte order, == NI::PackedColor) read by the
    // host as B8G8R8A8_UNORM. The client writes the real vertex colour ONLY when the mesh
    // uses VertexColorProperty source 2 (ambient+diffuse / DiffAmb); otherwise it writes
    // 0xFFFFFFFF (white), which makes the shader's col*(d+a) reduce to the white-material
    // (d+a) case — so the host always runs the DiffAmb path with no per-draw routing flag.
    struct GeomVertexWire {
        float px, py, pz;
        float nx, ny, nz;
        float u, v;
        std::uint32_t color;         // packed D3DCOLOR (B,G,R,A); white when vColSource != DiffAmb
    };

    // M-Skinning: bind-pose skinned vertex — position + normal + top-4 bone influences
    // (weights + packed UBYTE4 palette indices) + base-map UV. The GPU palette-skins this
    // with the per-frame bone window. `indices` packs idx0 in the low byte, matching D3D9
    // SkinnedVertex.indices so the FSL R8G8B8A8_UINT unpacks in the same order. 52 bytes.
    // (UV added for skinned texturing; colour still omitted — DiffAmb vcol is a later tier.)
    struct SkinnedVertexWire {
        float px, py, pz;
        float nx, ny, nz;
        float w0, w1, w2, w3;
        std::uint32_t indices;       // packed UBYTE4 bone palette indices (idx0 = low byte)
        float u, v;                  // base-map UV (set 0); skinned meshes are single-UV
    };

    // Tier 4 multi-map: a STATIC opaque part carrying dark/detail/glow sibling maps on the
    // same NiTexturingProperty (e.g. "Glow in the Dark" night windows). It needs up to 4 UV
    // sets, so it rides a SEPARATE wide vertex format + its own host pipeline (single-UV
    // geometry — the 99% case — stays lean at 36 B GeomVertexWire). pos+normal+color +
    // 4 UV sets (absent sets duplicate set 0). 60 bytes. Sampled set-major from the cache VB.
    struct GeomVertexWireMM {
        float px, py, pz;            // 0
        float nx, ny, nz;            // 12
        std::uint32_t color;         // 24  packed D3DCOLOR (B,G,R,A); white when vColSource != DiffAmb
        float uv[4][2];              // 28..59  UV sets 0..3 (dup of set 0 for absent sets)
    };

    // GeomPartWire::flags bits.
    constexpr std::uint16_t kGeomFlagSkinned  = 0x1;   // part vertices are SkinnedVertexWire (stride 56)
    constexpr std::uint16_t kGeomFlagMultiMap = 0x2;   // part vertices are GeomVertexWireMM (stride 60)

    // Per-part header preceding the part's vertex+index data in the batch blob. When
    // (flags & kGeomFlagSkinned), the part's vertices are SkinnedVertexWire (stride 44)
    // and numBones is the part's bone count; otherwise GeomVertexWire (stride 36). The
    // index-buffer layout is unchanged either way.
    struct GeomPartWire {
        std::uint32_t slot;          // dense host array index assigned by the client
        std::uint16_t revisionID;    // cache revisionID at capture (host re-uploads on change)
        std::uint16_t flags;         // kGeomFlag* bits (was pad)
        std::uint32_t vertexCount;   // GeomVertexWire / SkinnedVertexWire count
        std::uint32_t indexCount;    // uint16 index count (= triangleCount * 3)
        std::uint16_t numBones;      // skinned parts: bone count (<= kMaxBones); 0 otherwise
        std::uint16_t pad2;
    };

    // M1c per-frame draw item: which uploaded mesh (slot) to draw, with its current
    // model->world transform (D3DXMATRIX bytes, row-major — uploaded straight into the
    // host's gObject cbuffer; see opaque.srt.h for the no-transpose convention). The
    // per-frame draw list is an array of these in a chunked byte vec, with the camera
    // view*proj carried inline in the RenderFrame RPC params. 108 bytes.
    struct DrawItemWire {
        std::uint32_t slot;
        float         world[16];
        std::uint32_t texIndex;   // bindless gTextures[] slot for the base map (0 = default white)
        float         alphaRef;   // alpha-test reference 0..1 (0 = no alpha test; frag discards a < ref)
        // Tier 2b material (FFE PerPixelPS vertexMaterial routing). RGB only — the opaque
        // output alpha is forced to 1 (coverage mask) and alpha test uses texture alpha vs
        // alphaRef, so material alpha is unused. Mirrors CachedGeometry.matDiffuse/Ambient/Emissive.
        float         matDiffuse[3];
        float         matAmbient[3];
        float         matEmissive[3];
        std::uint32_t vColSource;  // 0 none (const material), 1 emissive (vcol->emissive), 2 diffamb (vcol->d+a)
        // Terrain DECAL_1 overlay: bindless gTextures[] slot for the second land texture,
        // blended over the base by vcol ALPHA (the AlphaGrid). 0 = no decal / non-terrain
        // (the frag's splat is gated off when this is 0, so every other draw is unchanged).
        std::uint32_t overlayTexIndex;
    };

    // Texture-residency upload (Phase 2 bindless texturing). The client resolves each unique
    // texture to a dense slot, reads its RAW DDS bytes via BSA::loadFileBytes, and ships
    // [TexUploadWire][dds bytes] entries through the geometry channel's chunked vec. The host
    // parses the DDS (BCn/uncompressed + mip chain) into gTextures[slot]. slot 0 is the host's
    // default white texture (never uploaded). Sent once per unique texture (no re-upload).
    struct TexUploadWire {
        std::uint32_t slot;
        std::uint32_t byteLen;    // length of the DDS blob that follows inline
    };

    // Bindless texture-array capacity (client residency cap == host gTextures[] size; the host
    // mirrors this as MAX_TEXTURES in opaque.srt.h / kMaxTextures in forgerender.cpp).
    // MUST stay in lock-step with host MAX_TEXTURES. The Persistent descriptor TABLE (gTextures +
    // gStaticsArrays) crashes Forge's addDescriptorSet >1024 entries on this stack (2048 faults in
    // consume_descriptor_handles; 1024 verified OK). gTextures(896) + gStaticsArrays(128) = 1024,
    // exactly the proven-OK boundary. Distant STATICS no longer live in gTextures — they moved to
    // gStaticsArrays (descriptor-array of Texture2DArrays, bucketed by format/size; see
    // forgerender.cpp). Only the 3 distant-land ATLAS slots remain host-reserved in gTextures.
    constexpr std::uint32_t kMaxTextures = 896;

    // Host-owned distant LAND atlas reserves the TOP kDlReserve slots of the shared bindless
    // gTextures[] array (base/normal/detail — 3 slots; the rest is headroom). Distant statics left
    // gTextures for gStaticsArrays, so this shrank 320 -> 8. The client caps its own bottom-up
    // residency below it: client slots [1, kMaxTextures-kDlReserve); land atlas [kMaxTextures-3,
    // kMaxTextures). Client over-cap falls back to white (client-side LRU recycles its range).
    constexpr std::uint32_t kDlReserve = 8;

    // M-Skinning per-frame draw item: which uploaded skinned mesh (slot) to draw, the
    // part's bone count, and its mirror flag (left-side parts reuse the right mesh via a
    // negative-scale bone → inside-out without the mirror pipeline). There is NO per-draw
    // world matrix — the bone palette is already world-space. A SkinnedDrawWire is
    // followed INLINE by numBones * 64 palette bytes (each bone a model->world D3DXMATRIX,
    // row-major). A skinned-draw blob = repeated [SkinnedDrawWire][palette]. 16 bytes + palette.
    struct SkinnedDrawWire {
        std::uint32_t slot;
        std::uint32_t numBones;
        std::uint32_t mirror;        // 1 = mirrored (negative-determinant); pick CW pipeline
        std::uint32_t texIndex;      // bindless gTextures[] slot for the base map (0 = default white)
        float         alphaRef;      // alpha-test reference 0..1 (0 = no alpha test; frag discards a < ref)
    };

    // Tier 4 multi-map per-frame draw item: a STATIC opaque part with up to 4 ORDERED texture
    // stages (built CLIENT-side by replicating rendercachedcolor.cpp::buildCacheStages — present
    // maps pushed with their op + UV set, stable-sorted by texCoordSet ascending). The host runs
    // a fixed-function stage loop replicating the cache color pass:
    //   BASE  : c = tex.rgb * lit;  baseA = tex.a   (arg2 = DIFFUSE even when not stage 0)
    //   MOD   : c *= tex.rgb        MOD2X : c *= tex.rgb*2        ADD : c += tex.rgb
    // alphaRef applies to the BASE stage only (other stages' alpha is keep-prev / irrelevant to
    // opaque coverage). Material + vColSource feed the SAME Tier 2b/3a lit term as opaque.frag.
    struct MultiMapDrawWire {
        std::uint32_t slot;
        float         world[16];
        float         matDiffuse[3];
        float         matAmbient[3];
        float         matEmissive[3];
        std::uint32_t vColSource;    // 0 none (const material), 1 emissive, 2 diffamb
        float         alphaRef;      // base-stage alpha test 0..1 (0 = no test)
        std::uint32_t stageCount;    // 1..4 (ordered by texCoordSet)
        // Per stage: texIndex (low 16) | uvSet (bits 16-17) | op (bits 18-19).
        // op: 0 BASE (MOD x DIFFUSE), 1 MOD, 2 MOD2X, 3 ADD.
        std::uint32_t stages[4];
    };

    // Per-stage word packers (client builds, host/shader unpack).
    constexpr std::uint32_t kMMOpBase  = 0u;
    constexpr std::uint32_t kMMOpMod   = 1u;
    constexpr std::uint32_t kMMOpMod2X = 2u;
    constexpr std::uint32_t kMMOpAdd   = 3u;
    inline std::uint32_t packMMStage(std::uint32_t texIndex, std::uint32_t uvSet, std::uint32_t op) {
        return (texIndex & 0xFFFFu) | ((uvSet & 0x3u) << 16) | ((op & 0x3u) << 18);
    }

    // Tier 3a point light (per-frame, world-space). One entry == three float4, so the host
    // memcpy's the received light array straight into its light cbuffer with no repacking.
    // Mirrors MGE::SceneGraph::PointLight (diffuse pre-multiplied by dimmer; falloff = the
    // raw 1/(k0+k1·d+k2·d²) coefficients; radius = the engine's specular.r fade). The frag
    // replicates XE FixedFuncEmu.fx evalOnePointLight EXACTLY (full k0/k1/k2 attenuation +
    // smoothstep(radius, 2·radius) soft cutoff) in WORLD space. pointLightMult is baked into
    // color client-side (the main cache path uses 1.0, so this is identity today).
    struct PointLightWire {
        float posRadius[4];   // xyz = world position, w = soft-cutoff radius (specular.r)
        float color[4];       // xyz = diffuse rgb (dimmer- and pointLightMult-scaled); w unused
        float falloff[4];     // x = k0 const, y = k1 linear, z = k2 quad; w unused
    };

    // Per-frame point-light cap. The frag loops a bounded working set (Tier 3a is the
    // correctness-first, no-cull step); Tier 3b clustered culling lifts this. MUST match the
    // host MAX_POINT_LIGHTS (opaque.srt.h) and kMaxPointLights (forgerender.cpp).
    constexpr std::uint32_t kMaxPointLights = 128;

    // SK1 sky takeover: per-frame sky draw item — an alpha-blended sky shape (SK1 = the gradient
    // atmosphere dome). Like DrawItemWire it references an uploaded mesh slot + its camera-relative
    // world (D3DXMATRIX bytes, row-major). texIndex 0 = vertex-colour-only (the dome); SK2 textured
    // shapes carry a real bindless slot. srcBlend/destBlend are D3DBLEND_* (translated from
    // NiAlphaProperty); SK1 draws SRCALPHA/INVSRCALPHA, so the host ignores them for now and SK2
    // buckets draws by blend-pair. Drawn FIRST in the host colour pass (depth off) so it sits behind
    // the opaque world. 108 bytes.
    struct SkyDrawWire {
        std::uint32_t slot;
        float         world[16];
        std::uint32_t texIndex;    // bindless gTextures[] slot (0 = vertex-colour-only dome)
        std::uint32_t srcBlend;    // D3DBLEND_* source factor (SK2: bucket draws by blend-pair)
        std::uint32_t destBlend;   // D3DBLEND_* dest factor
        float         alphaRef;    // alpha-test ref 0..1 (0 = no test)
        // SK2 FFP modulation: the captured MaterialProperty diffuse rgb + per-element alpha
        // fade (= matDiffuse[3]: star night-fade, cloud cross-fade). vColSource routes the
        // frag between vertex colour (dome/stars) and the constant material (moon disc/shadow),
        // mirroring opaque.frag. The frag does c = tex * base; c.a *= matAlpha.
        float         matColor[3]; // material diffuse rgb
        float         matAlpha;    // material alpha (weather/night fade; 1 = opaque)
        std::uint32_t vColSource;  // 0 none (const material), 1 emissive, 2 diffamb
        // WT2 reflection: this shape is the SUN DISC (client re-billboards it to face the MAIN
        // camera, buildSkyDrawList). The Forge reflection pass needs to flip its basis-z before the
        // mirror so it stays round in the mirrored view (the moons arrive camera-faced too but look
        // fine, so only the sun is flagged). 0 for every other sky shape. Appended (offset-stable).
        std::uint32_t isSunDisc;
    };

    // Per-frame sky draw cap. SK1 draws only the dome; the full sky subtree is ~15 shapes (SK2).
    // MUST match the host kMaxSkyDraws (forgerender.cpp).
    constexpr std::uint32_t kMaxSkyDraws = 64;

    // AT1 sorted-alpha takeover: per-frame alpha-BLENDED world draw item (banners, tapestries,
    // foliage, window glass — the scene-1 sorted set MW draws over the empty z-buffer when Forge
    // owns the opaques). Like DrawItemWire it references an uploaded mesh slot + its camera-
    // relative world; srcBlend/destBlend are the captured NiAlphaProperty D3DBLEND_* factors
    // (SkyDrawWire's pattern — the host buckets alpha-over vs additive per draw). The CLIENT
    // sorts the list back-to-front (MW's sorter criterion: bound-center view depth) and the host
    // draws in received order, depth-tested against the opaque prepass but never writing.
    // matAlpha = MaterialProperty alpha (the FFE per-draw fade); the frag does
    // a = tex.a * vcolA * matAlpha. 144 bytes (128 + AT3 captured-geometry locators + cullFlags).
    struct AlphaDrawWire {
        std::uint32_t slot;
        float         world[16];
        std::uint32_t texIndex;    // bindless gTextures[] slot (0 = default white)
        std::uint32_t srcBlend;    // D3DBLEND_* source factor
        std::uint32_t destBlend;   // D3DBLEND_* dest factor
        float         alphaRef;    // alpha-test ref 0..1 (0 = no test)
        float         matDiffuse[3];
        float         matAlpha;    // material alpha (MaterialProperty::alpha)
        float         matAmbient[3];
        float         matEmissive[3];
        std::uint32_t vColSource;  // 0 none (const material), 1 emissive, 2 diffamb
        // AT3 captured-alpha: when slot == kAlphaSlotCaptured this item does NOT reference an
        // uploaded mesh slot — its geometry lives in the shared captured VB/IB (see kMaxCaptured*
        // below), and these three fields locate it: [vertexBase, vertexBase+?) verts,
        // indexCount indices starting at indexBase (index values already rebased to 0). The
        // host binds pCapAlphaVB/pCapAlphaIB and draws cmdDrawIndexedInstanced(indexCount,
        // indexBase, 1, vertexBase, idx). Zeroed for cached-mesh (slot) items. 12 bytes.
        std::uint32_t vertexBase;  // first captured vertex (index into the shared captured VB)
        std::uint32_t indexBase;   // first captured index (into the shared captured IB)
        std::uint32_t indexCount;  // captured index count (triangleCount * 3)
        // Cull selection (mirrors MW's per-shape cull mode). bit0 = twoSided (NiStencilProperty
        // DRAW_BOTH → CULL_NONE); bit1 = mirrored (negative-determinant world → reversed winding,
        // like the opaque mirror PSO). Single-sided non-mirrored (flags==0) → CULL_BACK.
        std::uint32_t cullFlags;
    };
    constexpr std::uint32_t kAlphaCullTwoSided = 1u;   // bit0
    constexpr std::uint32_t kAlphaCullMirrored = 2u;   // bit1

    // AT3 sentinel slot: an AlphaDrawWire whose geometry is CAPTURED (shared VB/IB), not a
    // cached mesh slot. Chosen 0xFFFFFFFF so it can never collide with a real dense slot.
    constexpr std::uint32_t kAlphaSlotCaptured = 0xFFFFFFFFu;

    // Per-frame alpha draw cap. Dense cities run a few hundred blended shapes; 1024 gives 4x
    // headroom while keeping the world window at exactly 64KB (1024 x 64B, one CBV window).
    // MUST match the host kMaxAlphaDraws (forgerender.cpp).
    constexpr std::uint32_t kMaxAlphaDraws = 1024;

    // AT3 captured-alpha caps (within the shared kMaxAlphaDraws sort). The captured VB/IB are
    // single-buffered persistent-mapped host resources filled per frame from the client's copy;
    // 20000 verts (36B GeomVertexWire = 720KB) + 60000 uint16 indices (120KB) = 840KB fits one
    // GeomChunk (1MB), so a 1-chunk vec sidesteps the IPC uint32-reservation hazard.
    constexpr std::uint32_t kMaxCapturedAlphaDraws   = 256;
    constexpr std::uint32_t kMaxCapturedAlphaVerts   = 20000;
    constexpr std::uint32_t kMaxCapturedAlphaIndices = 60000;

#pragma pack(pop)

    // Transport chunk for the shared upload vector. The IPC Vec reserves
    // maxSize * windowBytes bytes (vec.cpp), so a byte-element vector with a multi-MB
    // window overflows uint32 — exactly the trap the occlusion-mask path avoids by
    // using a big chunk element (few elements, large each). We mirror that: the window
    // is kGeomChunks chunks of 1MB, so reservation = maxSize(=window) * 1MB stays well
    // under uint32. Both sides alloc / read the vector as GeomChunk so size() and
    // &vec[0] stay consistent; the exact byte length travels in the RPC params.
    // assign_bytes flat-copies the blob in.
    //
    // Geometry and textures have SEPARATE window sizes (both use the GeomChunk element type,
    // just different chunk counts). Geometry batches are small (vertex+index data, packed per
    // window) — 8MB is ample. Textures need a big window: a single DDS must fit in ONE window
    // or resolveTextureSlot/flushTextures drop it to white. 4096x4096 replacers (DXT1 ~11MB,
    // DXT5/BC7+mips ~22MB) need ~32MB. Keeping them separate avoids paying the 32MB reservation
    // on the geom vec too. NOTE the d3d8.dll client is 32-bit: each vec RESERVES (not commits) a
    // window of address space, so geom(8) + tex(32) = 40MB reserved — budget the 2-4GB space.
    constexpr unsigned kGeomChunkBytes = 1u << 20;             // 1 MB (shared element size)
    constexpr unsigned kGeomChunks     = 8;                    // 8 MB geometry window
    constexpr unsigned kTexChunks      = 32;                   // 32 MB texture window
    constexpr unsigned kGeomWindowBytes = kGeomChunkBytes * kGeomChunks;
    constexpr unsigned kTexWindowBytes  = kGeomChunkBytes * kTexChunks;
    struct GeomChunk { char b[kGeomChunkBytes]; };

}
