#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct IDirect3DDevice9;
struct IDirect3DVertexBuffer9;
struct IDirect3DIndexBuffer9;
struct IDirect3DTexture9;
struct IDirect3DVertexDeclaration9;

namespace MGE::GeometryCache {

    // Per-geometry entry keyed on NiTriShape* (cast to uint32_t on x86).
    // Non-skinned objects: worldTransformD3D updated each frame.
    // Skinned objects: bind-pose vertices with per-vertex weights + bone indices ship once;
    //   the bone matrices update per frame into bonePalette and the host's vertex shader
    //   skins (no per-frame CPU skinning, no per-frame re-ship).
    // Entries are created on first visit and evicted when the object is no longer
    // in the scene (skipped by a complete onFrameReady walk).
    // S5b: entries used to carry a DX9 mirror of the geometry — a double-buffered vertex
    // buffer (vb[2]/writeSlot/readVB), an index buffer, and the stride/FVF to bind them —
    // for MGE's own cache draws. S5a deleted the last of those draws (the depth pre-pass),
    // so the cache is a pure producer now: CPU-side bounds/material data plus the wire
    // stream it ships to the Forge host. Nothing here touches the D3D9 device.
    struct CachedGeometry {
        uint32_t vertexCount;
        uint32_t triangleCount;
        float    boundsCenter[3];       // model-space bound center (for culling)
        float    boundsRadius;          // model-space bound radius
        // Tight model-space AABB (min/max over the mesh verts), captured at upload.
        // Transformed (8 corners) to a world AABB for point-light selection, matching
        // FixedFunctionShader::computeBoundingBox on the reactive path — the cache VB
        // is WRITEONLY so renderMorrowind can't walk it. Non-skinned only; skinned
        // entries keep the bone-derived sphere bound for light selection.
        float    aabbMin[3];
        float    aabbMax[3];
        uint16_t revisionID;            // GeometryData::revisionID at last upload
        // Forge-mode "already shipped to host" sentinel. In default Forge play the dead DX9
        // mirror VB (e.vb) is no longer built (needMirror() false), so its pointer can no
        // longer serve as uploadEntry's content-identity gate "already uploaded" proxy. Set
        // true at the end of a capture (uploadEntry/buildSkinnedVB); cleared in releaseEntry
        // (topo change / eviction => must re-process + re-ship).
        bool     hostUploaded = false;
        bool     isSkinned;
        // Skinned: per-frame bone palette (model->world, 16 floats per bone) and
        // bone count. skinnedUnsupported set when numBones exceeds the shader palette
        // (kMaxBones) — such entries are skipped (reported), no CPU fallback.
        std::vector<float> bonePalette;
        uint32_t numBones;
        bool     skinnedUnsupported;
        uint8_t  dynamicHint;           // counts down from N when transform moves; 0 = static
        // True when the part's world/bone transform has negative determinant (a
        // mirrored left-side part). Clip-space winding is flipped, so the cache
        // depth/shadow draws must cull the opposite face for these.
        bool     mirrored;
        // True for entries walked from the world landscape (terrain) root. The
        // cache-driven color pass excludes these — terrain needs vertex colours +
        // texture splatting we don't synthesize yet (Phase 2); it stays on the
        // distant-land reflection path. Depth/shadow ignore this flag.
        bool     isLandscape;
        // True for entries walked from worldPickObjectRoot (dropped items, projectiles
        // etc.). That root is NOT traversed by the engine's world-camera occlusion
        // classify, so the Stage 2 engine-set cache cull keeps these via frustum
        // instead of dropping them for absence from the world classify.
        bool     isPickRoot;
        // C4d shadow-caster category: true = LIVE (the game says this part moves) —
        // geometry inside a character subtree (NPC/creature body parts + bone-attached
        // equipment/weapons, via the inCharacter walk verdict) or owned by an Activator/
        // Door reference (silt strider, steam machinery, doors; resolved once at capture
        // through the node's TES3 extra data). Ships in DrawItemWire.casterFlags: the
        // Forge host keeps LIVE casters out of the cached static shadow tiles and
        // re-renders them in the per-frame dynamic tile instead.
        bool     isLive;
        // Deliverable A source-mover: true when this LIVE part actually moves — it or an
        // ancestor in its own object hierarchy carries an ACTIVE transform-animating controller
        // (Keyframe/Path/LookAt/Roll). Distinguishes a swinging lantern from a fixed one (both
        // Activator = isLive). Ships in DrawItemWire.casterFlags as kDrawCasterAnimated; the host
        // (g_shadowSourceMover on) keeps only isLive && animated rigid casters in the dynamic
        // tile, baking still activators into the cached static tile. Set once at capture.
        bool     animated;
        // SK1 sky takeover: true for entries walked from skyRoot (atmosphere dome, stars,
        // sun, moons, clouds). Sky is alpha-blended and drawn by the Forge host's dedicated
        // sky pass (depth off, behind the opaque world) — it's EXCLUDED from every opaque
        // draw list and re-uploaded every frame (the dome's vertex-colour gradient changes
        // with sun angle / weather without a revisionID bump). srcBlend/destBlend are the
        // NiAlphaProperty blend factors (D3DBLEND_*), translated in extractMaterial.
        bool     isSky;
        // FP1a first-person takeover: true for entries walked from the WorldController
        // armCamera root (the first-person arms/weapon subtree, rendered by MW in its
        // own post-z-clear scene with its own camera). FP entries are EXCLUDED from
        // every main-scene draw list (dispatch, caster re-emit, frustum fallback) and
        // drawn only by the host's dedicated FP pass under the arm camera's viewProj.
        bool     isFP;
        // Cell-grid eviction (engine-authoritative gone-signal). Tagged at capture from MW's
        // DataHandler: null => this entry belongs to the EXTERIOR (its owning cell is derived
        // from worldTransformD3D at sweep time); non-null => the interior Cell* it was captured
        // under. The sweep evicts any entry whose home cell has left MW's live grid — exterior:
        // |cell - centralGrid| > kCellGridRadius; interior: homeInteriorCell != currentInteriorCell.
        // This replaces the parent-chain climb / age rule with the grid MW itself streams on.
        const void* homeInteriorCell = nullptr;
        unsigned char srcBlend;   // D3DBLEND_* (sky source blend factor)
        unsigned char destBlend;  // D3DBLEND_* (sky dest blend factor)
        // SK2: subtree visit order within the skyRoot walk (0 = first child visited).
        // walk() descends skyRoot children in deterministic array order each frame, so this
        // == MW's back-to-front sky draw order (atmosphere dome → stars → sun → moons). The
        // Forge sky pass sorts its draw list by this so multiple alpha-blended shapes layer
        // correctly. Only meaningful for isSky entries; stale (but harmless) otherwise.
        uint16_t skyOrder;
        // SK3 cloud scroll: MW scrolls the cloud layer by rewriting the shape's UVs in the
        // mesh data every frame, but sky VBs ship to the host ONCE — without this the host
        // clouds freeze at their capture-time scroll position. skyBaseUV/skyBaseUVLast =
        // vertex 0 / last-vertex UV at upload (the baseline baked into the shipped VB);
        // skyUVOffset = live vertex-0 UV − baseline, refreshed by the per-frame sky walk and
        // shipped in SkyDrawWire.uvOffset (sky.vert adds it; wrap sampler handles the modulo).
        // skyBaseUVLast exists only for the one-shot uniformity check (MW is expected to
        // shift ALL verts by the same delta). Only meaningful for isSky entries; zero otherwise.
        float    skyBaseUV[2];
        float    skyBaseUVLast[2];
        float    skyUVOffset[2];
        // SK4 sky vertex-colour tracking: FNV-1a hash of the vertex-colour array at last
        // upload. MW rebakes sky vcols in place (cloud weather tint, star fade) without our
        // walk seeing a re-upload trigger — sky skips the revisionID rule because MW bumps
        // it EVERY frame (UV scroll). The per-frame sky walk hashes the live vcols of
        // vColSource==2 shapes (the only ones whose real vcol ships to the host) and
        // re-runs uploadEntry on mismatch, so tint changes ship as they happen (byte-
        // quantized colour interpolation → a few re-uploads/sec during transitions).
        // Only meaningful for isSky entries; zero otherwise.
        uint32_t skyVcolHash;
        // Material (pointers into NI memory — valid for the session)
        IDirect3DTexture9* d3dTexture;  // null if no base texture
        // Terrain decal overlay (TexturingProperty maps[6] = DECAL_1): the second
        // land texture, blended over the base by the AlphaGrid in vertex-colour
        // alpha. Null for non-terrain / single-texture tiles. Drives the cache
        // terrain reflection's two-texture splat.
        IDirect3DTexture9* d3dOverlay;
        // DECAL_1 overlay source filename (SourceTexture::fileName), for resolving the
        // overlay to a Forge bindless slot. Null when no overlay. The D3D9 cache terrain
        // pass uses d3dOverlay above; the Forge path needs the name to upload/resolve.
        const char*        overlayTextureName;
        // Multi-map texturing (e.g. "Glow in the Dark" night windows): the base
        // map's DARK/DETAIL/GLOW siblings on the same NiTexturingProperty. Each is
        // null when absent; *UV is the cached UV set the map samples (clamped to
        // {0,1} — the VB carries at most a second UV set). The cache color pass
        // reconstructs the PPL fixed-function multi-stage blend from these
        // (MODULATE dark / MODULATE2X detail / ADD glow), matching the FFE JIT.
        // BUMP/GLOSS are env-map effects MW disables in fixed function — out of
        // scope; terrain DECAL_1 stays on d3dOverlay.
        IDirect3DTexture9* d3dDark;
        IDirect3DTexture9* d3dDetail;
        IDirect3DTexture9* d3dGlow;
        // Each map's true UV set (NI texCoordSet, clamped 0..3 — the FFE shader's
        // texcoordIndex is 2-bit and an FVF carries at most 4 sets). The cache color
        // pass sets each stage's texcoordIndex to these so a map samples its OWN set
        // (e.g. the "Glow in the Dark" detail map on set 2), matching PPL. Glow-mod
        // windows carry 3 UV sets: base(0), dark(1), detail(2).
        uint8_t baseUV, darkUV, detailUV, glowUV;
        // Each map's NiTexturingProperty::Map::clampMode, RAW (0 CLAMP_S_CLAMP_T, 1 CLAMP_S_WRAP_T,
        // 2 WRAP_S_CLAMP_T, 3 WRAP_S_WRAP_T). PER MAP, not per shape: a glow map can clamp while the
        // base map wraps. The Forge host used to ignore this entirely and sampled everything through
        // one anisotropic REPEAT sampler, so any mesh with UVs outside [0,1] that relied on CLAMP
        // tiled instead of holding its edge texel (the texture "kept going" past the artist's edge —
        // glaring on alpha-tested cutouts).
        //
        // These MUST default to 3 (WRAP_S_WRAP_T), and that is why they carry explicit initializers
        // while every field around them does not: entries are created by `g_cache[key]`, which
        // value-initializes, and a zeroed clampMode would read as CLAMP_S_CLAMP_T — the exact
        // OPPOSITE of MW's default. A shape with no texturing property would clamp instead of wrap.
        uint8_t baseClamp = 3, darkClamp = 3, detailClamp = 3, glowClamp = 3;
        // Non-skinned VB UV-set count: the highest UV set any present map uses + 1,
        // bounded by the mesh's set count and 4 (1 = ordinary single-UV geometry, the 99%
        // case). Shipped to the host, which sizes its own vertex layout from it. Skinned
        // entries keep uvSetCount=1.
        // (S5b: the derived vbStride/vbFVF went with the DX9 mirror they bound.)
        uint8_t  uvSetCount;
        const char*        textureName; // SourceTexture::fileName, null if none
        // Multi-map sibling source filenames (SourceTexture::fileName), for resolving the
        // dark/detail/glow maps to Forge bindless slots. Null when the map is absent. The
        // D3D9 cache color pass uses the d3d* textures above; the Forge path needs the names.
        const char*        darkTextureName;
        const char*        detailTextureName;
        const char*        glowTextureName;
        float alphaRef;
        bool  alphaTest;
        bool  blendEnable;
        // A NiFlipController is attached to this shape's NiTexturingProperty: the bound base
        // NiSourceTexture is swapped over time (flip-book animation — fires, water, magic VFX;
        // 135 controllers across 131 NIFs in a modded install, tools/nif-controller-census.csv).
        // The swap never touches NiGeometryData, so revisionID does NOT move and the material
        // re-extract that every other texture change rides is never triggered. Entries carrying
        // this flag get their bound texture re-read each frame they are visited
        // (refreshAnimatedTexture); everything else pays nothing.
        bool  texAnimated;
        // NiStencilProperty DRAW_BOTH: the shape is authored two-sided (window panes,
        // waterfalls, thin cloth) and MW draws it with culling OFF. Single-sided shapes
        // (no stencil / not DRAW_BOTH) MW draws CULL_BACK — the Forge alpha pass must
        // honour this so a solid alpha mesh (draped altar cloth) doesn't show its back/
        // interior faces through the front. Default false = single-sided (CULL_BACK).
        bool  twoSided;
        // Material colours (RGBA) captured from the NI MaterialProperty on the
        // create/material-change path, for the cache-driven color pass
        // (Phase 0.5). Default to white diffuse/ambient, zero emissive when the
        // shape has no material. Depth/shadow ignore these.
        float matDiffuse[4];
        float matAmbient[4];
        float matEmissive[4];
        // Per-channel emissive-boost gain derived from this fixture's OWN light at capture
        // (see computeEmissiveGain). 1,1,1 = not a lit fixture / no boost. Applied by
        // emissiveForDraw(), NOT folded into matEmissive, so a capture-once material can
        // still pick up a re-derived gain: the draw list is rebuilt every frame.
        float emissiveGain[3];
        // Vertex colour usage. hasVertexColor: the mesh carries per-vertex colours
        // (filled into the non-skinned VB's DIFFUSE slot). vColSource: NI
        // VertexColorProperty::source — 0 ignore (vcol unused, constant material),
        // 1 emissive, 2 ambient+diffuse. The color pass uses vcol only when both say
        // so, else real material colours win (don't white-wash them).
        bool    hasVertexColor;
        uint8_t vColSource;
        // Lifecycle
        uint64_t lastFrame;             // frame counter from most recent visit
        // FP0: frame stamp set by markSubtreeSuppressed on entries under a subtree the
        // engine has appCulled this frame (the inactive-POV player body). The offscreen
        // shadow-caster re-emit loop sweeps the whole cache with no appCulled knowledge
        // (walk() early-returns on culled nodes without touching entries), so without
        // this the freshly-culled 3rd-person body keeps rendering until the eviction
        // sweep. Consumers skip entries with suppressedFrame == currentFrame().
        uint64_t suppressedFrame;
        // Nearest NiSwitchNode ancestor at capture (null for the overwhelming majority of
        // entries), plus the index of the switch child this shape hangs under. A NiSwitchNode
        // displays ONLY the child at switchIndex — Glow in the Dahrk's "NightDaySwitch" flips
        // between coincident OFF / ON / INT-DAY window variants that differ only in material
        // (ON/INT-DAY are emissive white, OFF is not).
        //
        // walk() honours switchIndex (it descends the active child only), but it is the ONLY
        // thing that did: ensureLive()'s first-sight capture off the engine classify feed does
        // not, and the eviction sweep's parentVerdict only asks whether the chain still reaches
        // a live root — an INACTIVE switch child is still fully parented, so a variant captured
        // while it was active never left the cache. Both variants then sat in the host's draw
        // list at identical transforms and the winner was decided by draw order, not by the flag
        // that is supposed to decide it (the day/night window glow bug: MW drew the lit variant,
        // we drew whichever landed last).
        //
        // switchOwner is a raw engine node, so it is vtable-validated before every deref (same
        // guard the eviction climb uses) and cleared by purgeAll.
        const void* switchOwner = nullptr;
        int         switchChild = -1;
        // The GeometryData* this entry was built from. A NiTriShape address can be
        // recycled onto a NEW shape (cell transitions) while the entry survives —
        // with deferred eviction + the far-keep hysteresis that window is real, so
        // identity is checked on every visit: mismatch rebuilds from scratch (the
        // same recycled-key guard the host-side captureGeometry dedup uses).
        const void* dataPtr;
        // D3D row-major world transform for non-skinned objects (model-space VBs).
        // Cast to D3DXMATRIX* for use with D3DXMatrixMultiply.
        // Skinned objects are CPU-skinned to world-space; worldTransformD3D is unused for them.
        float worldTransformD3D[16];
    };

    // Must be called once before onFrameReady, with the D3D9 device.
    void init(IDirect3DDevice9* device);

    // Called once per frame from renderStage0. Walks the scenegraph
    // from the two world roots, creates/updates VBs for new/changed geometry,
    // and evicts entries not seen this frame. dataHandler is the
    // TES3::DataHandler* (typed void* to avoid TES3 header deps).
    //
    // W1.5 active-cell gate: when gateRadius > 0, whole NiNode subtrees whose world
    // bound lies entirely beyond gateRadius of gateEye (xyz) are SKIPPED — MW's own
    // view-distance cull means nothing there can enter any drawn set, so refreshing
    // those entries is dead per-frame work (the Forge host draws the far world from
    // its own data). Callers must size gateRadius to cover the frustum's far CORNERS
    // (view-z far plane * slope factor), not just the view distance. gateRadius <= 0
    // (or null eye) disables the gate — the full-walk behavior is unchanged. Skipped
    // entries go stale but are kept by the eviction sweep's hysteresis (see .cpp)
    // so returning to an area doesn't re-capture/re-upload it.
    // W3 live-read at build: when liveDrawBuild is true the per-frame refresh walk is
    // SKIPPED — the classify-visible keys are freshened individually via ensureLive()
    // in buildFrustumVisibleSet (lazy capture on first sight), and a frame with no
    // classify result pulls the full walk in via ensureFullWalk(). Sky walk, eviction
    // sweep (age-rule on live frames) and heartbeat still run here every frame.
    void onFrameReady(void* dataHandler, const float* gateEye = nullptr, float gateRadius = 0.0f,
                      bool liveDrawBuild = false);

    // Drop EVERY cached entry, routing each key through the normal eviction channel so the
    // Forge feed releases the matching host mesh slots. Call on a cell teardown (load door,
    // teleport, save load): the cache keys on shape addresses and cannot otherwise tell that the
    // whole scene graph was destroyed, so old-cell entries survive into the new cell and get
    // emitted for a frame — the one-frame flash of the previous cell's objects/NPCs. Survivors
    // are re-captured lazily on first sight.
    void purgeAll();

    // Arm the post-load residency window: a few frames of forced full-cell capture, so the whole
    // active cell (interior) / active-cell grid (exterior) is captured and host-resident before the
    // player can turn around. Without it the cache fills only from the engine's FRUSTUM-limited
    // classify set, so geometry behind the camera pops in (or lands as one build-time burst) on the
    // first rotation after a load. purgeAll() calls this itself; call it directly for a load that
    // does NOT purge — the very first evaluation, where there is no old cell to flush.
    void armPostLoadWalk();

    // Refresh (or lazily capture) ONE entry straight off its live NiTriShape*. Only
    // valid for keys the engine drew THIS frame (classify visible set) — that is what
    // guarantees the pointer is alive. Refreshes exactly the per-frame-varying fields
    // the draw paths read (worldTransformD3D, bonePalette, mirrored, lastFrame) plus
    // the revision-gated re-upload; on a cache miss it captures with root-derived
    // context (object/pick/landscape). Returns null if the shape has no model data or
    // capture failed. A no-op lookup on entries the full walk already stamped.
    const CachedGeometry* ensureLive(uint32_t key);

    // Run the full refresh walk NOW if this frame hasn't run one — the fail-safe for
    // live-draw-build frames where no classify result exists (the frustum fallback
    // iterates the whole cache and needs current-frame freshness). Idempotent.
    void ensureFullWalk();

    // FP0: stamp suppressedFrame = currentFrame() on every cached TriBasedGeom entry in
    // the subtree rooted at avObject (a NI::AVObject*, passed as void* to keep NI deps
    // out of headers), traversing WITHOUT the appCulled early-out — the whole point is
    // marking a subtree the engine just culled. Derefs only the live node the caller
    // fetched from the engine THIS frame; never touches cached keys. Returns the number
    // of entries stamped. Called from onFrameReady for the player's inactive-POV body.
    uint32_t markSubtreeSuppressed(void* avObject);

    // MW-ONLY-UI world suppression. Drives appCulled on MW's world roots so the ENGINE stops
    // traversing them (Phase 2 stopped MGE drawing, but MW still walked the graph and issued
    // every draw for the proxy to reject). Applied in onFrameReady AFTER our capture walks.
    //
    // INDEPENDENT BITS, not a ladder. A cumulative ladder cannot attribute a symptom to a root:
    // "smoke vanished at level 2" only ever implied pick, because level 2 culled landscape AND
    // pick together. Each root must be switchable alone for the observation to mean anything.
    //
    // Suppressing a root also removes the blended DIPs captureAlphaDraw consumes from it (that
    // content reaches the host ONLY via MW actually issuing the draw). Weather/VFX/projectile/
    // spell roots are worldRoot siblings and are never suppressed.
    enum SuppressBits { kSuppressLand = 1, kSuppressPick = 2, kSuppressObjects = 4 };
    void applyWorldSuppression(int mask);
    // The mask currently APPLIED to the engine's roots (not the requested one). Consumers that
    // walk those roots themselves must consult this and bypass the root's appCulled flag, or they
    // read an empty scene — see SceneGraph::runWalk.
    int  worldSuppressionApplied();
    // Clear every flag we set. Called at the UI transition and at Present: the roots are shared
    // with MW's off-screen targets (local map), which must never render culled.
    void restoreWorldSuppression();

    const std::unordered_map<uint32_t, CachedGeometry>& cache();

    // Keys whose cached entry passes the offscreen shadow-caster PRE-DISTANCE mover filter
    // (skinned / multimap head / rigid LIVE — everything the Forge feed's offscreen re-emit loop
    // in buildGeometryDrawLists would consider before the per-frame distance cull). Maintained
    // incrementally at capture / reclassify / evict, so that loop iterates ~dozens of candidate
    // movers instead of scanning the WHOLE cache every frame (the fixed ~1ms tail probe pinned to
    // the full-map walk). Superset of what any want-flags admit — the consumer still applies the
    // per-frame distance / visible-set / suppressedFrame / want-flag checks unchanged.
    const std::unordered_set<uint32_t>& moverCandidates();

    // Sky / first-person membership sets — the same incremental-maintenance pattern as
    // moverCandidates, for the OTHER two full-cache scans the Forge feed used to pay every
    // frame (buildSkyDrawList's main + [sk-diag] loops, buildFPFrame's gather). Maintained
    // at capture / reclassify / evict alongside the mover set; every per-frame filter
    // (lastFrame freshness, host slot, classification re-check) stays with the consumer.
    // Invariant: both sets ⊆ keys(cache()).
    const std::unordered_set<uint32_t>& skyKeys();
    const std::unordered_set<uint32_t>& fpKeys();

    // Near-eye PLAIN-STATIC shadow-caster keys, rebuilt each eviction sweep (30-frame cadence).
    // Movers re-emit offscreen via moverCandidates(); a plain static (lantern, wall fixture) is
    // not a mover candidate, so without this its host caster record is never seeded until it is
    // drawn once — a fixture behind the camera on first cell load casts no shadow until looked at.
    // The Forge feed's offscreen re-emit iterates this set and re-applies the precise per-frame
    // filter (distance / visible-set / suppressedFrame / MM / collision-proxy). A distance-culled
    // snapshot, not a ⊆-maintained set — cleared+rebuilt wholesale each sweep. Stale keys are
    // harmless (the consumer guards with cache().find).
    const std::vector<uint32_t>& nearStaticCasters();

    // Near-eye ALPHA (blended alpha-over cutout) shadow-caster keys, same snapshot discipline as
    // nearStaticCasters(). Blended fixtures (lanterns/banners/foliage) cast via the host alpha
    // shadow path, which is fed by the VISIBLE alpha draw list — so an off-screen blended fixture
    // casts nothing until looked at. The Forge feed re-emits these into alphaCands so their caster
    // records seed regardless of view. Distance-culled snapshot; stale keys guarded by cache().find.
    const std::vector<uint32_t>& nearAlphaCasters();

    // Frames elapsed since the eviction sweep last ran (0 on a sweep frame). Build-spike
    // observability: tests whether build spikes align with the ~30-frame sweep cadence.
    uint32_t framesSinceEvictSweep();

    // First-sight capture budget. setCaptureBudget(-1) = unlimited (the default);
    // setCaptureBudget(n >= 0) lets ensureLive() do at most n first-sight lazy captures
    // before deferring the rest (returns null — the classify set regenerates every frame,
    // so a deferred key re-arrives and captures next frame; a 1-frame delay, never a loss).
    // NEVER gates the refresh path — pose/palette freshness of cached entries is
    // correctness. Each call also resets the deferred counter; captureDeferredLastBuild()
    // reads how many first-sight keys were deferred since the last setCaptureBudget call.
    void setCaptureBudget(int budget);
    uint32_t captureDeferredLastBuild();

    // Running first-sight lazy-capture count for the current cache frame (reset by
    // onFrameReady). Snapshot before/after a build loop to count captures-this-build.
    uint32_t liveCaptureCount();

    // Drain the keys the eviction sweep dropped since the last call (objects that left the world
    // within a cell — picked up, despawned, disabled). The Forge feed maps each to its host mesh
    // slot and ships a release sentinel so the host forgets the slot's shadow-caster record.
    // Moves the internal list into `out` and clears it (so each key is delivered exactly once).
    void takeEvictedKeys(std::vector<uint32_t>& out);

    // The cache's frame counter (incremented by each onFrameReady). Eviction is a
    // periodic sweep, not per-frame, so consumers that scan the WHOLE cache must
    // treat entries with lastFrame != currentFrame() as stale (not in the scene this
    // frame). Consumers driven by a current-frame visible set don't need this.
    uint64_t currentFrame();

    // The emissive triple a draw should ship: matEmissive times emissiveGain. Every Forge
    // draw-list builder goes through this so the boost has exactly one definition; the DX9
    // baseline path reads matEmissive directly and stays vanilla (it is the A/B reference).
    // `out` receives 3 floats.
    void emissiveForDraw(const CachedGeometry& e, float* out);

    // Vertex buffer format used by the reflection-moon shapes below (and, until S5b, by
    // each CachedGeometry's DX9 mirror VB).
    // D3DFVF_XYZ | D3DFVF_NORMAL | D3DFVF_DIFFUSE | D3DFVF_TEX1
    // Layout: float3 pos, float3 normal, DWORD color(0xFFFFFFFF), float2 uv
    static constexpr unsigned int kVBStride = 36;
    static constexpr unsigned int kVBFVF    = 0x152; // XYZ|NORMAL|DIFFUSE|TEX1

    // S5b: kVBStridePos / kVBFVFBase sized the multi-map mirror VB, and kSkinnedVBStride +
    // skinnedDecl() described the skinned one. Both mirrors are gone; uvSetCount is shipped
    // to the host, which sizes its own layout.

    // kMaxBones must match MAX_BONES in "XE Common.fx".
    static constexpr unsigned int kMaxBones = 32;

    // ---- Reflection moon support -------------------------------------------------
    // One drawable billboard shape of a moon (its Shadow Node cutout or Moon Node disc),
    // materialized from the live scene graph so the water reflection can draw moons that
    // are up but outside the main-camera frustum (recordSky only captures what the main
    // view drew). VB/IB are in kVBFVF/kVBStride layout (matches the sky StatVertIn).
    struct MoonShapeDraw {
        IDirect3DVertexBuffer9* vb;       // kVBFVF, kVBStride
        IDirect3DIndexBuffer9*  ib;       // D3DFMT_INDEX16 triangle list
        IDirect3DTexture9*      texture;  // base map (null draws skipped by caller)
        unsigned int vertCount;
        unsigned int triCount;
        unsigned char srcBlend;           // D3DBLEND_* (translated from NiAlphaProperty)
        unsigned char destBlend;          // D3DBLEND_*
        bool  isMoonShadow;               // dark-side cutout (destBlend == INVSRCALPHA)
        float worldTransform[16];         // D3D row-major model->world
    };

    // Materialize the drawable shapes (Shadow Node + Moon Node, up to 2) of the moon
    // rooted at moonRoot (a NI::Node*, passed as void* to keep NI deps out of headers).
    // Returns 0 when the moon's root node is app-culled (down/hidden by phase — the
    // engine's own "is this moon up" oracle) or has no drawable shape; else the shape
    // count, shadow-cutout first. The VB/IB are owned by an internal per-NiGeometry
    // cache (created once, vertex data refreshed each call).
    int buildMoonDrawList(void* moonRoot, MoonShapeDraw out[2]);

    // Reverse map: IDirect3DTexture9* → SourceTexture::fileName.
    // DORMANT — the reverse map is currently unpopulated (the per-frame rebuild was
    // removed as unused; see scenegraph_geometry_cache.cpp). Always returns null until
    // a consumer repopulates it incrementally from extractMaterial.
    const char* resolveTextureName(IDirect3DTexture9* tex);

    // ---- FP particle billboarding (torch flame, enchant sparks) --------------------
    // buildFPParticleQuads (run during the first-person walk) billboards each FP particle
    // system's live particles against the arm camera into camera-facing quads. MW's particle
    // renderer does this at draw time; under FP suppression that never runs, and visitGeometry
    // captured only particle CENTERS (a degenerate mesh). The client's buildFPFrame appends
    // these quads to the shared captured-alpha VB/IB and emits kAlphaSlotCaptured FP alpha
    // items. Vert data is IPC::GeomVertexWire[]; index data is uint16[] (rebased per system to
    // 0 at vertexBase). See tasks/forge-fp-particles.md.
    struct FPParticleRec {
        uint32_t vertexBase;   // first vertex in fpParticleVerts()
        uint32_t vertexCount;
        uint32_t indexBase;    // first index in fpParticleIndices()
        uint32_t indexCount;
        IDirect3DTexture9* texture;   // base map GPU texture (resolve to a bindless slot)
        uint32_t srcBlend;            // D3DBLEND_* (from NiAlphaProperty)
        uint32_t destBlend;           // D3DBLEND_*
        float    matEmissive[3];      // NiMaterialProperty emissive (flame glow; smoke = 0)
    };
    const void* fpParticleVerts(uint32_t& countOut);    // GeomVertexWire[countOut]
    const void* fpParticleIndices(uint32_t& countOut);  // uint16[countOut]
    const std::vector<FPParticleRec>& fpParticleRecs();

}
