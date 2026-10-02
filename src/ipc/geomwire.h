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
#include <cmath>     // std::sqrt / std::floor — packQuatSmallest3 (G3 gobo side array)
#include <cstring>   // std::memcpy — bit-copy for the packed light-identity lane

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
    // RELEASE sentinel: the client's geometry cache evicted this slot's object (picked up into
    // inventory, despawned, MWSE-disabled — gone from the world WITHIN a cell, not a cell change).
    // The record is header-only (vertexCount = indexCount = 0, no payload). The host forgets the
    // slot's shadow-caster record so its shadow stops ghosting in place. See renderprocess release
    // emit + forgerender release handler.
    constexpr std::uint16_t kGeomFlagRelease  = 0x4;
    // NiUVController takeover: the part carries a GeomUVAnimWire key track appended AFTER its
    // indices (uvAnimBytes in the header). The host evaluates the track from MW sim time and
    // scrolls the UVs in-shader — the engine's per-tick vertex-UV rewrites no longer reship.
    constexpr std::uint16_t kGeomFlagUVAnim   = 0x8;
    // GEOMETRY DEDUP (tasks/forge-geometry-dedup.md): this slot is another INSTANCE of a mesh whose
    // bytes are already in the host arena as BLOCK `GeomPartWire::block`. Header-only (vertexCount =
    // indexCount = 0, no payload), like Release. The host points the slot at the block's VB/IB range
    // and refcounts it; the range is parked only when its last holder releases. Vertices are
    // model-space, so every placed copy of a rock/tree/wall used to ship identical bytes (85% of the
    // geometry shipped in an AutoZip stress run was such duplicates).
    constexpr std::uint16_t kGeomFlagAlias    = 0x10;

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
        // kGeomFlagUVAnim: byte size of the GeomUVAnimWire payload (header + keys) appended
        // after this part's indices. 0 for every other part (was pad2 — always zero on the
        // wire, so old blobs parse identically). Both part-boundary walkers (client chunker
        // flushGeometry, host parser uploadGeometry) add it to the part size.
        std::uint16_t uvAnimBytes;
        // GEOMETRY DEDUP block id (client-assigned, monotonic, never reused; 0 = none). On a plain
        // static arena upload: nonzero = the host registers this part's arena range as that block,
        // so later kGeomFlagAlias records can share it. On an alias: the block to share. 0 for every
        // part that is not a block (skinned, multimap, uvAnim, forced and re-uploads).
        std::uint32_t block;
    };
    static_assert(sizeof(GeomPartWire) == 24, "GeomPartWire changed size: client and host must ship together");

    // NiUVController key track, shipped ONCE with the mesh (appended after the part's indices;
    // size in GeomPartWire::uvAnimBytes). Followed inline by (keyCountU + keyCountV +
    // keyCountSU + keyCountSV) x {float time, float value} linear keys, in that track order:
    // U-offset, V-offset, U-tiling, V-tiling. Bezier/TBC source keys are resampled to linear
    // CLIENT-side at capture. The host evaluates each track at t = MW sim time cycled into
    // [keyMin, keyMax] per cycleType and rebuilds MW's rewrite from the ORIGINAL UVs the client
    // uploads: u' = (u-0.5)*tileU + 0.5 - offU, v' = (v-0.5)*tileV + 0.5 + offV (scale about the
    // texture centre, then offset — OpenMW's NiUVController matrix, in D3D's V-down space). An
    // absent tiling track means tile 1. The r0 candle flames animate tiling (a breathing flame),
    // which is why tiling is carried at all: rejecting it left every flame on the per-frame engine
    // reship path, hundreds of reships a frame, and the flames blinked while the game ran.
    struct GeomUVAnimWire {
        std::uint8_t  setIndex;      // UVController::textureSet, clamped to the captured uvSetCount-1
        std::uint8_t  cycleType;     // 0 loop / 1 reverse (ping-pong) / 2 clamp
        std::uint16_t keyCountU;     // U-offset linear keys following this header
        std::uint16_t keyCountV;     // V-offset linear keys (after the U keys)
        std::uint16_t keyCountSU;    // U-tiling linear keys (after the V keys); 0 = tile 1
        float         frequency;     // TimeController frequency/phase: keyTime = t*frequency + phase
        float         phase;
        float         keyMin, keyMax;// TimeController low/highKeyFrame (the cycle window)
        float         baseU, baseV;  // controller currentU/VOffset AT CAPTURE (embedded in the verts)
        std::uint16_t keyCountSV;    // V-tiling linear keys (after the U-tiling keys); 0 = tile 1
        std::uint16_t pad;
    };
    static_assert(sizeof(GeomUVAnimWire) == 36, "GeomUVAnimWire wire size");

    // DrawItemWire::casterFlags bits (C4d categorical shadow casters).
    // LIVE = the game says this part moves: geometry under a character subtree (NPC/creature
    // body parts + bone-attached equipment/weapons) or owned by an Activator/Door reference
    // (silt strider, steam machinery, doors). The host keeps LIVE casters OUT of the cached
    // static shadow tiles and re-renders them in the per-frame dynamic tile instead — no
    // motion heuristics, no bake/rebake epoch churn. Everything else (statics, clutter,
    // containers, light fixtures) stays on the cached static path (move/appear/release
    // bumps the caster epoch; clutter delay is acceptable).
    constexpr std::uint32_t kDrawCasterLive = 0x1;

    // ANIMATED (Deliverable A, source-mover): the LIVE part actually MOVES — it (or an
    // ancestor in its own object hierarchy) carries an ACTIVE transform-animating NI
    // controller (Keyframe/Path/LookAt/Roll). LIVE is a coarse TES3 record-TYPE flag (every
    // Activator/Door), but most fixtures — a still hammock, a fixed lantern whose only
    // controller animates its flame texture — never move. With the host g_shadowSourceMover
    // toggle on, only LIVE|ANIMATED casters take the per-frame dynamic tile; a LIVE-but-static
    // activator falls back to the cached static path (no dyn-atlas pollution, no 6-face churn).
    // Resolved once at capture (hasTransformAnim), like the LIVE category itself.
    constexpr std::uint32_t kDrawCasterAnimated = 0x2;

    // STENCIL "FAKE HOLE" PORTALS (comGravePit and its ref-65 family). A handful of MW meshes fake
    // an opening in a solid surface with the stencil buffer: a flat MASK quad over the opening
    // (stencil ALWAYS -> REPLACE) records where the opening is VISIBLE, then a HULL volume drawn
    // with the depth TEST OFF (NiZBufferProperty) and stencil EQUAL overwrites the occluder's depth
    // — but ONLY inside the mask — so the geometry BEHIND the surface (a grave pit below the LAND
    // heightfield, a room behind a Dwemer grate) is no longer depth-killed by it. The host's pDepth
    // is D32_SFLOAT with no stencil plane, so the trick is reproduced with a 1-bit gate in an R8
    // render target: the mask's own depth test writes the gate, the hull's frag discards where the
    // gate is 0. That is the same two-step, one plane over.
    //
    // These three bits mark the members of one portal OBJECT so the host can pull it out of the
    // GPU-driven indirect groups and draw it INLINE, in role order (masks -> hulls -> rest), which
    // is the ordering the trick needs and the thing cmdExecuteIndirect cannot express. Being inline
    // also takes the draws out of the two-phase Hi-Z occlusion test — needed, not incidental: the
    // hull and the pit interior live BEHIND the very surface they are erasing, so the occlusion test
    // culls them every time.
    //
    // MASK|HULL are the roled helpers; MEMBER is everything else in the same object (the pit dirt,
    // the surrounding mound). A mask never draws colour — the meshes carry editor colours on them
    // (comGravePit's is blue, dwrvgratepipe's is literally stencilerror.dds) and they are authored
    // to be invisible. MASK|HULL are also barred from the shadow-caster lists: a helper volume
    // casting its own shadow into the pit it opens would be nonsense.
    //
    // Riding casterFlags' spare bits — no wire size change (DrawItemWire is a shipped-together 144).
    constexpr std::uint32_t kDrawPortalMask   = 0x4;
    constexpr std::uint32_t kDrawPortalHull   = 0x8;
    constexpr std::uint32_t kDrawPortalMember = 0x10;
    constexpr std::uint32_t kDrawPortalAny    = kDrawPortalMask | kDrawPortalHull | kDrawPortalMember;

    // M1c per-frame draw item: which uploaded mesh (slot) to draw, with its current
    // model->world transform (D3DXMATRIX bytes, row-major — uploaded straight into the
    // host's gObject cbuffer; see opaque.srt.h for the no-transpose convention). The
    // per-frame draw list is an array of these in a chunked byte vec, with the camera
    // view*proj carried inline in the RenderFrame RPC params. 148 bytes — client and
    // host MUST ship together on any size change (AT3 precedent).
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
        std::uint32_t casterFlags; // kDrawCasterLive bit (C4d shadow-caster category) + kDrawPortal* role
        std::uint32_t clampMode;   // NiTexturingProperty::Map::clampMode, RAW (see kTexClamp*)
        // EMISSIVE GAIN — the flux/area boost, shipped as its OWN lane and NOT folded into
        // matEmissive. It is a per-channel RATIO (kEmissiveFlux * ownLight.diffuse / fixtureArea,
        // see computeEmissiveGain), and matEmissive is an AUTHORED colour that the host's vert puts
        // through decodeAuthored(). Folding the two together put the ratio inside srgbToLinear,
        // which is unclamped past 1.0, so a gain g reached the frag as ~g^2.4 (2.11 -> 5.61,
        // 25.95 -> 2189, a candle flame's 401 -> 1.56e6). Separately: on vColSource 1 the emissive
        // IS the vertex colour and the frags never read matEmissive at all, so the folded gain was
        // discarded outright for flames. Both halves are fixed by keeping the ratio in its own
        // lane and letting opaque.vert / multimap.vert apply it AFTER the decode, to whichever
        // source vColSource selected. 1,1,1 = no boost (not a lit fixture).
        //
        // ⚠ THE IDENTITY IS 1, NOT 0 — this field is NOT zero-init-safe. The host's vert
        // MULTIPLIES the selected emissive by it, and on vColSource 1 the selected emissive is the
        // vertex colour, so a `Wire item{}` that never assigns this renders that draw's emissive
        // BLACK rather than merely un-boosted. Every writer must set it; the two paths with no
        // fixture to derive a gain from (captured DIPs, FP particle quads) set 1.0 explicitly.
        float         emissiveGain[3];
        // PBR MATERIAL (tasks/forge-pbr-materials.md, Track C). Bindless gTextures[] slot of the
        // base texture's `<base>_paramh.dds` — DXT5, R = metal, G = roughness, B = IOR/spec,
        // A = HEIGHT — or 0 when the base texture has none. The same pattern as texIndex and
        // overlayTexIndex a third time: a slot on the draw item, resolved (and LRU-refreshed) through
        // the client's own residency, so an in-use param map can never be recycled out from under a
        // draw that still names it.
        //
        // ⚠ 0 IS "NO PBR MATERIAL", AND THAT IS WHAT KEEPS MIXED COVERAGE BIT-IDENTICAL. opaque.frag
        // runs its PBR block only for a nonzero slot, on a flat per-draw branch, so every material
        // with no _paramh takes exactly today's arithmetic — one shader, no permutation, and coverage
        // grows by dropping files in. The host also zeroes it when the slot's upload did not land
        // (and when the pbrEnable knob is off), so the failure mode is the OLD image, never a white
        // param map read as metal = 1.
        std::uint32_t paramTexIndex;
    };
    // ⚠ This comment block said "112 bytes" for three field-additions after it stopped being true
    // (emissiveGain and the lanes before it took it to 140; paramTexIndex made 144). A size stated in
    // prose is a claim nothing checks, so the number now lives where the compiler reads it.
    // (derivTexIndex, the baked `_paramd` slot, took it to 148 until that arm was RETIRED 2026-09-18.)
    static_assert(sizeof(DrawItemWire) == 144, "DrawItemWire changed size: client and host must ship together");

    // MW's texture address mode, shipped raw (the shader's TEX_* defines use the same 4 values in
    // the same order, so nothing translates it anywhere along the way).
    //   0 CLAMP_S_CLAMP_T   1 CLAMP_S_WRAP_T   2 WRAP_S_CLAMP_T   3 WRAP_S_WRAP_T (MW's default)
    // Before this existed the host sampled EVERYTHING through one anisotropic REPEAT sampler, so any
    // mesh with UVs outside [0,1] that relied on CLAMP tiled instead of holding its edge texel.
    constexpr std::uint32_t kTexClampSClampT = 0u;
    constexpr std::uint32_t kTexClampSWrapT  = 1u;
    constexpr std::uint32_t kTexWrapSClampT  = 2u;
    constexpr std::uint32_t kTexWrapSWrapT   = 3u;   // default when a mesh has no texturing property

    // ENCHANTED-ITEM GLOW, riding the clampMode lane's spare bits (DrawItemWire / SkinnedDrawWire /
    // AlphaDrawWire all carry clampMode as a full uint32 for a 2-bit value). Set = MW has attached
    // its shared enchant NiTextureEffect to this shape's node, so the host must lay the caustic
    // environment map over it (enchantglow.h.fsl).
    //
    // A spare bit here rather than a new wire field on purpose: clampMode is ALREADY forwarded from
    // every one of those three wires into packTexAlpha, which is the single choke point every draw
    // path goes through — so the host decodes it in ONE place and every path (near/skinned/alpha,
    // main and first-person) inherits it with no per-path plumbing and no wire size change.
    // packTexAlpha masks the address mode to & 3 before packing, so this bit can never be mistaken
    // for a sampler mode.
    constexpr std::uint32_t kTexFlagEnchantGlow = 4u;

    // Texture-residency upload (Phase 2 bindless texturing). The client resolves each unique
    // texture to a dense slot, reads its RAW DDS bytes via BSA::loadFileBytes, and ships
    // [TexUploadWire][dds bytes] entries through the geometry channel's chunked vec. The host
    // parses the DDS (BCn/uncompressed + mip chain) into gTextures[slot]. slot 0 is the host's
    // default white texture (never uploaded). Sent once per unique texture (no re-upload).
    // TexUploadWire::slot bit 31: this texture is DATA, not colour — upload it under its stored
    // UNORM format and never through the scene's sRGB view. Set by the client on `_paramh` maps.
    //
    // Needed because the host's scene decode (toSceneTextureFormat) re-views every BC1/2/3 as
    // _SRGB in the linear scene, and a _paramh is BC3: its RGB would come back sRGB-DECODED —
    // roughness 0.5 read as 0.21 — while its alpha (height) survived, since sRGB never touches
    // alpha. The host cannot tell a param map from albedo by its bytes, so the client says so.
    // Bit 31 and not a new field: plain slots are < kMaxTextures and flip slots use bit 15 with
    // their bucket/layer in bits 0-14, so bits 16-31 of `slot` are free on every upload.
    constexpr std::uint32_t kTexUploadData = 0x80000000u;
    // TexUploadWire::slot bit 30: RELEASE this plain slot (byteLen 0, no DDS follows). The client
    // evicted it (evictStaleTextures: no live part names it, unsampled for a while); the host
    // retires the texture after the frame fence and points the slot back at the default white.
    constexpr std::uint32_t kTexUploadRelease = 0x40000000u;
    // TexUploadWire::slot bit 29: the slot is COLD — no draw list the client shipped in the last
    // kTexColdFrames frames names it (a never-used slot, an evicted one, or a streamed texture's
    // fresh slot). The host is 2-deep (tasks/forge-pipeline-depth.md), so a frame is usually executing
    // when an upload lands; the bindless range is DESCRIPTORS_VOLATILE, and a descriptor write to a
    // slot no in-flight frame reads needs no wait. Without the bit the host waits the frames in flight
    // first (descWriteGuard) — the safe default for a slot something may still be sampling.
    constexpr std::uint32_t kTexUploadCold = 0x20000000u;
    // Client frames (g_frame) after which a slot's last reference can no longer be in flight: a park
    // build stamps frame f, fires at f+1, and the host holds at most two frames — so f+4 is clear.
    // Twice that, for margin; it only delays when a streamed texture's old slot returns to the pool.
    constexpr std::uint32_t kTexColdFrames = 8;
    // Async stream lane (Command::StreamUpload, tasks/forge-async-texture-stream.md): entries per
    // batch, capped so the receipt's failedMask has one exact bit per entry.
    constexpr std::uint32_t kMaxStreamBatch = 32;

    struct TexUploadWire {
        std::uint32_t slot;       // bindless slot (or encoded flip slot); bit 31 = kTexUploadData, bit 30 = kTexUploadRelease, bit 29 = kTexUploadCold
        std::uint32_t byteLen;    // length of the DDS blob that follows inline
        // FLIP-BOOK ARRAY SLICES ONLY (slot & kFlipSlotFlag): total slice count of the
        // Texture2DArray this slice belongs to. Every slice of a bucket carries the SAME value, so
        // the host can create the array from whichever slice arrives first and validate the rest —
        // no separate declaration record, no ordering requirement, no multi-batch state machine.
        // 0 for ordinary 2D uploads.
        std::uint32_t arraySize;
    };

    // Bindless texture-array capacity (client residency cap == host gTextures[] size; the host
    // mirrors this as MAX_TEXTURES in opaque.srt.h / kMaxTextures in forgerender.cpp).
    // MUST stay in lock-step with host MAX_TEXTURES.
    //
    // SIZED BY THE ACTIVE GRID (2026-09-30). At 880 a dense exterior did not fit: the MWSE census
    // (tools/mwse-dev-mods/MGEProbe texcensus) counts 1079 unique texture files in Dragonstar East's
    // 9 loaded cells (Seyda Neen: 475), so every fast turn LRU-recycled what the last turn loaded —
    // 1400 recycles over a 720 deg spin, 25-87 ms frames on BOTH laps. MW loads a cell's textures
    // with the cell, so the grid's set is fixed and a cap above it makes turning free after the
    // first sight. Upper bound: < 4096, because enchantglow.h.fsl packs a slot into 12 bits
    // (ENCHANT_REFLECT_BIT). Plain slots also stay clear of kFlipSlotFlag (0x8000).
    //
    // The old ceiling ("the Persistent table crashes addDescriptorSet above 1024 entries") dated from
    // the June DIAG8 cap-mismatch era; nothing in Forge's D3D12 path limits it today (1M-entry
    // shader-visible heap, unbounded rootsig range, one set instance). Table = gTextures +
    // gStaticsArrays(128) + gFlipArrays(16). Only the 3 distant-land ATLAS slots are host-reserved in
    // gTextures (kDlReserve, at the top).
    constexpr std::uint32_t kMaxTextures = 4064;

    // ---- Flip-book texture arrays -----------------------------------------------------------
    // A NiFlipController flip book used to claim ONE BINDLESS SLOT PER FRAME — Enhanced Light's
    // magelight is 300 frames, i.e. a third of the whole residency for one spell, and a rich scene
    // then sat at 888/888 recycling. A book is uniform by construction (every frame the same format
    // and size), which makes it the ideal Texture2DArray: one descriptor for the whole book.
    //
    // Same shape as gStaticsArrays: a descriptor-array of Texture2DArrays bucketed by (format,
    // width, height); books sharing a bucket occupy disjoint layer ranges. 16 buckets is generous —
    // a bucket is a FORMAT+SIZE class, not a book — and books that overflow it (or that aren't
    // uniform) simply stay on the per-slot path, which still works.
    constexpr std::uint32_t kMaxFlipBuckets = 16;
    constexpr std::uint32_t kMaxFlipLayers  = 2048;   // per bucket; also the 11-bit field limit

    // The ENCODING is shared by the wire slot, the client's g_texSlot values, the per-draw
    // packTexAlpha field and the shader — ONE representation end to end, which is why nothing
    // between them needs a second lookup or an extra per-draw lane. It has to survive
    // packTexAlpha's 16-bit texIndex field, hence the tight layout:
    //   bit 15      : set = this is a flip-array slice, not a gTextures[] slot
    //   bits 11..14 : bucket  (0..kMaxFlipBuckets-1)
    //   bits 0..10  : layer   (0..kMaxFlipLayers-1)
    // Plain slots are < kMaxTextures (4064) so they never collide with the flag.
    constexpr std::uint32_t kFlipSlotFlag   = 0x8000u;
    constexpr std::uint32_t kFlipBucketShift = 11u;
    constexpr std::uint32_t kFlipLayerMask   = 0x7FFu;

    constexpr std::uint32_t makeFlipSlot(std::uint32_t bucket, std::uint32_t layer) {
        return kFlipSlotFlag | (bucket << kFlipBucketShift) | (layer & kFlipLayerMask);
    }
    constexpr bool          isFlipSlot(std::uint32_t s)     { return (s & kFlipSlotFlag) != 0; }
    constexpr std::uint32_t flipSlotBucket(std::uint32_t s) {
        return (s >> kFlipBucketShift) & (kMaxFlipBuckets - 1u);
    }
    constexpr std::uint32_t flipSlotLayer(std::uint32_t s)  { return s & kFlipLayerMask; }

    // Host-owned distant LAND atlas reserves the TOP kDlReserve slots of the shared bindless
    // gTextures[] array (base/normal/detail — 3 slots; the rest is headroom). Distant statics left
    // gTextures for gStaticsArrays, so this shrank 320 -> 8. The client caps its own bottom-up
    // residency below it: client slots [1, kMaxTextures-kDlReserve); land atlas [kMaxTextures-3,
    // kMaxTextures). Client over-cap falls back to white (client-side LRU recycles its range).
    constexpr std::uint32_t kDlReserve = 8;

    // SkinnedDrawWire::blendFlags — a skinned part's NiAlphaProperty state. Alpha-TEST rides
    // alphaRef (and always has); these are the alpha-BLEND bits, which the wire carried no room
    // for until ghosts/manes turned up rendering solid. A blended skinned part stays on the
    // skinned list (it needs the bone palette, which only that list carries) and is merely
    // TAGGED — the host packs it in every walk exactly as before and moves only its DRAW into
    // the alpha stage. See [[project_forge_alpha_composite_gap]].
    constexpr std::uint32_t kSkinFlagBlended  = 1u << 0;   // NiAlphaProperty alpha blend enabled
    constexpr std::uint32_t kSkinFlagTwoSided = 1u << 1;   // NiStencilProperty DRAW_BOTH → CULL_NONE
    // Together these two answer "is this blended part actually a SOLID body?", which alphaRef alone
    // cannot: the wire encodes "no alpha test" as alphaRef == 0, and a cutout card is free to test
    // at ref 0 (GREATER 0 = "discard only fully transparent texels"). The velk's mane does exactly
    // that — NiAlphaProperty 0x12ED, test ON, ref 0 — so it arrives indistinguishable from Dagoth
    // Ur's 0x00ED, test OFF. Census over 3058 skinned NIFs: 901 blend-without-test properties vs
    // 449 blend-with-test, so the split is real and worth a bit each.
    //
    //   kSkinFlagAlphaTest — NiAlphaProperty TEST_ENABLE. The shape leans on its texture's alpha to
    //                        carve holes (hair/mane/foliage cards). NEVER treat as solid.
    //   kSkinFlagAlphaAnim — the NiMaterialProperty carries a NiAlphaController, i.e. this shape's
    //                        alpha is DRIVEN and is normally 1.0. That is what makes Dagoth Ur
    //                        blended at all: his controller sits at 1.0 for the whole fight and only
    //                        runs 1->0 for the death dissolve. matAlpha below is re-read every frame
    //                        (refreshAnimatedMaterial), so the host can simply watch it fall.
    constexpr std::uint32_t kSkinFlagAlphaTest = 1u << 2;
    constexpr std::uint32_t kSkinFlagAlphaAnim = 1u << 3;

    // M-Skinning per-frame draw item: which uploaded skinned mesh (slot) to draw, the
    // part's bone count, and its mirror flag (left-side parts reuse the right mesh via a
    // negative-scale bone → inside-out without the mirror pipeline). There is NO per-draw
    // world matrix — the bone palette is already world-space. A SkinnedDrawWire is
    // followed INLINE by numBones * 64 palette bytes (each bone a model->world D3DXMATRIX,
    // row-major). A skinned-draw blob = repeated [SkinnedDrawWire][palette]. 44 bytes + palette.
    struct SkinnedDrawWire {
        std::uint32_t slot;
        std::uint32_t numBones;
        std::uint32_t mirror;        // 1 = mirrored (negative-determinant); pick CW pipeline
        std::uint32_t texIndex;      // bindless gTextures[] slot for the base map (0 = default white)
        float         alphaRef;      // alpha-test reference 0..1 (0 = no alpha test; frag discards a < ref)
        std::uint32_t clampMode;     // NiTexturingProperty::Map::clampMode, RAW (see kTexClamp*)
        std::uint32_t blendFlags;    // kSkinFlag* (0 for the opaque majority)
        float         matAlpha;      // MaterialProperty::alpha — the FFE per-draw fade (AlphaDrawWire::matAlpha)
        std::uint32_t srcBlend;      // D3DBLEND_* (translated from NiAlphaProperty); read only when BLENDED
        std::uint32_t destBlend;     // D3DBLEND_*
        // Back-to-front sort key for the host's skinned-alpha draw list: BONE 0's translation
        // dotted with the view forward. A skinned part's worldTransformD3D is not a meaningful
        // position (the bones carry it), so the static path's bound-centre expression does not
        // transfer. Eye-relative, so it is invariant to the camera-relative palette shift below.
        float         viewDepth;
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
        // Per stage: texIndex (low 16) | uvSet (bits 16-17) | op (bits 18-19) | clampMode (bits 20-21).
        // op: 0 BASE (MOD x DIFFUSE), 1 MOD, 2 MOD2X, 3 ADD.
        // clampMode is PER STAGE (each NiTexturingProperty::Map carries its own) — a glow map can
        // clamp while the base map wraps, so it cannot live once per draw like the other paths.
        std::uint32_t stages[4];
        // Route C (glow-window multimap-blend): bit0 = BLENDED. An alpha-BLEND multi-map shape
        // (e.g. Glow-in-the-Dahrk night windows: base orange x dark shape) rides THIS same list but
        // the host skips it in the opaque MM color/Z-prepass/shadow loops and draws it in the alpha
        // stage with a blend PSO (SRCALPHA/INVSRCALPHA, depth no-write) using multimap_alpha.frag.
        // 0 for every opaque multi-map draw.
        std::uint32_t drawFlags;
        // MaterialProperty::alpha — the FFE per-draw fade, identical in meaning to
        // AlphaDrawWire::matAlpha. Route C originally shipped no such field because the SEVEN
        // Glow-in-the-Dahrk window meshes it was measured on all had matAlpha 1.0 AND fully opaque
        // base maps, which made outA collapse to the vertex alpha. That sample was not the Route C
        // SET: any blended shape whose glow map is its own base map lands here too — e.g. the light
        // rays in in_c_rich_r_swin_bay_01.nif (base = glow = textures\glow\ray_alpha.dds, the old
        // MODULATE-then-ADD double-brightness trick), whose transparency lives ENTIRELY in the base
        // map's alpha channel. Route C must therefore reproduce MW's full FFE alpha,
        // texA * vcolA * matAlpha, exactly as alpha.frag does. Read only for BLENDED draws.
        float         matAlpha;
        // EMISSIVE GAIN — the flux/area boost, shipped as its OWN lane and NOT folded into
        // matEmissive. It is a per-channel RATIO (kEmissiveFlux * ownLight.diffuse / fixtureArea,
        // see computeEmissiveGain), and matEmissive is an AUTHORED colour that the host's vert puts
        // through decodeAuthored(). Folding the two together put the ratio inside srgbToLinear,
        // which is unclamped past 1.0, so a gain g reached the frag as ~g^2.4 (2.11 -> 5.61,
        // 25.95 -> 2189, a candle flame's 401 -> 1.56e6). Separately: on vColSource 1 the emissive
        // IS the vertex colour and the frags never read matEmissive at all, so the folded gain was
        // discarded outright for flames. Both halves are fixed by keeping the ratio in its own
        // lane and letting opaque.vert / multimap.vert apply it AFTER the decode, to whichever
        // source vColSource selected. 1,1,1 = no boost (not a lit fixture).
        //
        // ⚠ THE IDENTITY IS 1, NOT 0 — this field is NOT zero-init-safe. The host's vert
        // MULTIPLIES the selected emissive by it, and on vColSource 1 the selected emissive is the
        // vertex colour, so a `Wire item{}` that never assigns this renders that draw's emissive
        // BLACK rather than merely un-boosted. Every writer must set it; the two paths with no
        // fixture to derive a gain from (captured DIPs, FP particle quads) set 1.0 explicitly.
        float         emissiveGain[3];
    };
    constexpr std::uint32_t kMMDrawFlagBlended = 1u;   // MultiMapDrawWire::drawFlags bit0
    // Enchanted-item glow, Route C's copy of kTexFlagEnchantGlow. Multi-map has no clampMode lane
    // to hide in (its address mode is PER STAGE, packed into the stage words), so the bit rides
    // drawFlags instead; the host relays it into the multimap instance Meta word.
    constexpr std::uint32_t kMMDrawFlagEnchantGlow = 2u;   // drawFlags bit1

    // Per-stage word packers (client builds, host/shader unpack).
    constexpr std::uint32_t kMMOpBase  = 0u;
    constexpr std::uint32_t kMMOpMod   = 1u;
    constexpr std::uint32_t kMMOpMod2X = 2u;
    constexpr std::uint32_t kMMOpAdd   = 3u;
    inline std::uint32_t packMMStage(std::uint32_t texIndex, std::uint32_t uvSet, std::uint32_t op,
                                     std::uint32_t clampMode = kTexWrapSWrapT) {
        return (texIndex & 0xFFFFu) | ((uvSet & 0x3u) << 16) | ((op & 0x3u) << 18)
             | ((clampMode & 0x3u) << 20);
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
        float color[4];       // xyz = diffuse rgb (dimmer- and pointLightMult-scaled);
                              // w = P2 identity lane: packLightIdFlags() bits (id<<8 | flags).
                              // The frag reads only .rgb, so the bit pattern is inert on the GPU.
        float falloff[4];     // x = k0 const, y = k1 linear, z = k2 quad; w = shadow slot+1
                              // (host-patched by the shadow manager; 0 = unshadowed)
    };

    // P2 light identity/change flags. Packed by the client into PointLightWire::color[3] (an
    // unused-by-shader lane) and read back by the host shadow manager (P2 = log-only; P3 uses
    // it to hold a slot across frames without the position-tolerance hack). 24-bit id + 8-bit
    // flags carried as the float's raw bits — NOT a numeric float value; always bit-copy.
    constexpr std::uint32_t kLightFlagMoved   = 1u << 0;   // moved > 0.5u since last seen
    constexpr std::uint32_t kLightFlagNew     = 1u << 1;   // fresh id this frame (spawn / recycle)
    constexpr std::uint32_t kLightFlagFixture = 1u << 2;   // ESM fixture light (name light*/torch*/furn*) →
                                                           // shadow-priority BOOST so real fixtures win slots
                                                           // over nameless injected window/ambient fill.
    constexpr std::uint32_t kLightFlagCarried = 1u << 3;   // the PLAYER's held light (rides the camera). Host
                                                           // forces its shadow slot to re-render every frame —
                                                           // never caches a baked tile (detaches on slow rotate).

    inline float packLightIdFlags(std::uint32_t id, std::uint32_t flags) {
        const std::uint32_t bits = ((id & 0x00FFFFFFu) << 8) | (flags & 0xFFu);
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
    inline void unpackLightIdFlags(float lane, std::uint32_t& id, std::uint32_t& flags) {
        std::uint32_t bits;
        std::memcpy(&bits, &lane, sizeof(bits));
        id    = bits >> 8;
        flags = bits & 0xFFu;
    }

    // G3 fixture gobos (tasks/forge-light-gobo.md): a SIDE ARRAY appended to the light blob, one
    // entry per PointLightWire in the same order — [PointLightWire x n][LightGoboWire x n], so
    // lightBytes == n * 56 when it is present. PointLightWire has no free lane left (colour.w is the
    // identity, falloff.w the shadow slot), and a side array keeps its 48-byte memcpy intact.
    //   idHash    = fixtureIdHash(the owning reference's LIGH object id); 0 = not a LIGH reference
    //               (a carried torch resolves to its actor, a spell light to nothing);
    //   rotPacked = packQuatSmallest3(the reference's MODEL->WORLD rotation) — the frame the gobo
    //               was baked in (NOT the NiLight's own: an AttachLight node may be rotated).
    // The host maps idHash through fixtures.data to a gobo layer.
    struct LightGoboWire {
        std::uint32_t idHash;
        std::uint32_t rotPacked;
    };

    // 32-bit FNV-1a over the LOWERCASE object id — MGEgui DistantLandForm.cs FixtureIdHash, the
    // fixtures.data key. ASCII lowercase only (ids are ASCII in practice; the C# side lowercases
    // with the culture, which agrees on ASCII).
    inline std::uint32_t fixtureIdHash(const char* id) {
        std::uint32_t h = 2166136261u;
        for (; id && *id; ++id) {
            std::uint8_t b = (std::uint8_t)*id;
            if (b >= 'A' && b <= 'Z') { b = (std::uint8_t)(b + 32); }
            h ^= b;
            h *= 16777619u;
        }
        return h;
    }

    // Unit quaternion (x,y,z,w) -> 32 bits, smallest-three: 2 bits for the largest |component| (made
    // positive; q and -q are one rotation), then the other three in x,y,z,w order at 10 bits over
    // [-1/sqrt2, 1/sqrt2]. ~0.24 deg worst case. The shaders read it with asuint and gobo.h.fsl's
    // goboQuatUnpack — MUST match that. Used by the host for baked lights too, so there is one packer.
    inline std::uint32_t packQuatSmallest3(const float qIn[4]) {
        float q[4] = { qIn[0], qIn[1], qIn[2], qIn[3] };
        float n = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        if (!(n > 1e-12f)) { q[0] = q[1] = q[2] = 0.0f; q[3] = 1.0f; n = 1.0f; }
        n = 1.0f / std::sqrt(n);
        int big = 0;
        for (int k = 0; k < 4; ++k) { q[k] *= n; if (std::fabs(q[k]) > std::fabs(q[big])) { big = k; } }
        const float sgn = (q[big] < 0.0f) ? -1.0f : 1.0f;
        std::uint32_t p = (std::uint32_t)big << 30;
        int shift = 20;
        for (int k = 0; k < 4; ++k) {
            if (k == big) { continue; }
            const float t = (q[k] * sgn + 0.70710678f) * (1023.0f / 1.41421356f) + 0.5f;
            const int   v = (int)std::floor(t);
            p |= (std::uint32_t)(v < 0 ? 0 : (v > 1023 ? 1023 : v)) << shift;
            shift -= 10;
        }
        return p;
    }

    // Rotation matrix in COLUMN-vector form (world = R * v; R[r][c]) -> unit quaternion (x,y,z,w).
    inline void quatFromRotation(const float R[3][3], float q[4]) {
        const float tr = R[0][0] + R[1][1] + R[2][2];
        if (tr > 0.0f) {
            const float s = std::sqrt(tr + 1.0f) * 2.0f;
            q[3] = 0.25f * s;
            q[0] = (R[2][1] - R[1][2]) / s; q[1] = (R[0][2] - R[2][0]) / s; q[2] = (R[1][0] - R[0][1]) / s;
        } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
            const float s = std::sqrt(1.0f + R[0][0] - R[1][1] - R[2][2]) * 2.0f;
            q[3] = (R[2][1] - R[1][2]) / s;
            q[0] = 0.25f * s; q[1] = (R[0][1] + R[1][0]) / s; q[2] = (R[0][2] + R[2][0]) / s;
        } else if (R[1][1] > R[2][2]) {
            const float s = std::sqrt(1.0f + R[1][1] - R[0][0] - R[2][2]) * 2.0f;
            q[3] = (R[0][2] - R[2][0]) / s;
            q[0] = (R[0][1] + R[1][0]) / s; q[1] = 0.25f * s; q[2] = (R[1][2] + R[2][1]) / s;
        } else {
            const float s = std::sqrt(1.0f + R[2][2] - R[0][0] - R[1][1]) * 2.0f;
            q[3] = (R[1][0] - R[0][1]) / s;
            q[0] = (R[0][2] + R[2][0]) / s; q[1] = (R[1][2] + R[2][1]) / s; q[2] = 0.25f * s;
        }
    }

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
    // the opaque world. 120 bytes.
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
        // SK3 cloud scroll: MW scrolls the cloud layer by REWRITING the shape's UVs in the mesh
        // data every frame (the per-frame sky revisionID bump), but sky VBs ship ONCE — the host
        // clouds froze at their capture-time scroll position (F11 re-toggle "fixed" it by
        // re-capturing). The client diffs the live vertex-0 UV against the uploaded baseline and
        // ships the uniform offset; sky.vert adds it to the baked UV (wrap sampler handles the
        // modulo). Zero for every non-UV-animated sky shape. Appended (offset-stable).
        float         uvOffset[2];
        // P2b: WHAT THIS SHAPE IS (kSkyClass* below). The sky list is not one thing — it is five
        // elements wanting three different treatments once the Hosek-Wilkie field owns the
        // atmosphere, and nothing already on the wire can tell them apart:
        //   * vColSource answers "where does the colour come from", not "what is this": the dome
        //     and the stars are both 2, and the moon disc and its shadow layer are both 0.
        //   * isSunDisc is a single flag with two live consumers (the reflection re-face and the
        //     proxy's sun reject) and must NOT be widened into this enum — `if (it.isSunDisc)`
        //     would then fire for every moon.
        // Set by the client from the base-map name (renderprocess.cpp::classifySky), because that
        // name is the only thing MW's sky subtree carries that identifies an element. Appended
        // (offset-stable), and kSkyClassOther is 0 so an unset field means "draw it as authored",
        // i.e. exactly today's behaviour.
        std::uint32_t skyClass;
    };

    // The five sky elements, and OTHER for anything a mod adds. Only three treatments exist host
    // side (authored / pinned / radiant), but the CLASSES stay distinct because the host's CPU loop
    // needs DOME apart from CLOUD — the physical sky retires the dome and nothing else.
    enum SkyClass : std::uint32_t {
        kSkyClassOther = 0,   // unrecognised: drawn exactly as authored, whatever the sky does
        kSkyClassDome  = 1,   // the untextured atmosphere gradient — the ONE shape H-W replaces
        kSkyClassCloud = 2,   // Tx_Sky_<weather> — the layer that makes weather READ as weather
        kSkyClassSun   = 3,   // tx_sun_05 — an already-exposed glare sprite, not a photosphere
        kSkyClassMoon  = 4,   // tx_masser_* / tx_secunda_* — the LIT disc, drawn ADDITIVELY
        kSkyClassStars = 5,   // Tx_Stars*.tga + Tx_Stars_Nebula*.tga
        // tx_mooncircle_full_M|S — the full circle drawn alpha-OVER UNDER each lit disc, painted
        // with MW's SKY colour. It is NOT a moon and must not be treated as one: its job is to
        // OCCLUDE THE STARS behind the moon's dark limb (the stars draw at order 1-7, this at 9/11)
        // and to make that limb blend into the sky. Pinning it to its authored colour under a
        // physical sky is exactly wrong — MW's authored night sky is a dark blue and the H-W night
        // sky is black, so the pinned circle reads as a lit dark side. Verified in play, 2026-08-21:
        // *"moons look good, dark side is a tad brighter."*
        kSkyClassMoonShadow = 6,
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
    // a = tex.a * vcolA * matAlpha. 128 + AT3 captured-geometry locators + cullFlags/clampMode +
    // the AT3 extra-stage words (all APPENDED, so every existing field offset is stable).
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
        std::uint32_t clampMode;   // NiTexturingProperty::Map::clampMode, RAW (see kTexClamp*)
        // AT3 multi-stage: the FFE texture stages BEYOND the base map (dark/detail/glow), which MW
        // folds into the same DIP. Without them a base x dark shape renders at base brightness —
        // kurp's Enhanced Light VFX pair every base map with a blackmip*/darkmap* MODULATE layer.
        // Cached multi-map blends never need this (they ride Route C's MultiMapDrawWire); this is
        // for the shapes that only ever reach us as a captured DIP.
        //
        // Packed with packMMStage() — the SAME encoding Route C uses, so the shader decode is
        // shared: texIndex (low 16) | uvSet (bits 16-17) | op (bits 18-19) | clampMode (bits 20-21).
        // op is kMMOpMod / kMMOpMod2X / kMMOpAdd (never kMMOpBase — stage 0 IS texIndex above).
        //
        // uvSet is ALWAYS 0 here. The captured VB is GeomVertexWire (one UV set, 36 B) shared with
        // every cached mesh and with the alpha PSO's input layout, so a stage sampling UV set 1+
        // cannot be honoured; the client DROPS such a stage rather than sample the wrong
        // coordinates (that shape then renders exactly as it did before this field existed). The
        // Enhanced Light census says the uvSet-1 shapes are all cached/Route-C-owned anyway.
        std::uint32_t stageCount;  // extra stages actually present, 0..3 (0 = base map only)
        std::uint32_t stages[3];
        // EMISSIVE GAIN — the flux/area boost, shipped as its OWN lane and NOT folded into
        // matEmissive. It is a per-channel RATIO (kEmissiveFlux * ownLight.diffuse / fixtureArea,
        // see computeEmissiveGain), and matEmissive is an AUTHORED colour that the host's vert puts
        // through decodeAuthored(). Folding the two together put the ratio inside srgbToLinear,
        // which is unclamped past 1.0, so a gain g reached the frag as ~g^2.4 (2.11 -> 5.61,
        // 25.95 -> 2189, a candle flame's 401 -> 1.56e6). Separately: on vColSource 1 the emissive
        // IS the vertex colour and the frags never read matEmissive at all, so the folded gain was
        // discarded outright for flames. Both halves are fixed by keeping the ratio in its own
        // lane and letting opaque.vert / multimap.vert apply it AFTER the decode, to whichever
        // source vColSource selected. 1,1,1 = no boost (not a lit fixture).
        //
        // ⚠ THE IDENTITY IS 1, NOT 0 — this field is NOT zero-init-safe. The host's vert
        // MULTIPLIES the selected emissive by it, and on vColSource 1 the selected emissive is the
        // vertex colour, so a `Wire item{}` that never assigns this renders that draw's emissive
        // BLACK rather than merely un-boosted. Every writer must set it; the two paths with no
        // fixture to derive a gain from (captured DIPs, FP particle quads) set 1.0 explicitly.
        float         emissiveGain[3];
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
