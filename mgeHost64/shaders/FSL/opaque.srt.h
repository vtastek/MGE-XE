// mgeHost64 — M1c opaque scene SRT (batched cbuffer, scales past the 64KB cap).
//
// A single 64KB cbuffer holds at most 1024 float4x4 (the D3D12 cbuffer limit). To draw
// more than that per frame we BATCH: the host keeps a big world buffer of N 64KB windows
// and a PerBatch descriptor set whose instance b points at window b (via pRanges). Draws
// are issued in batches of 1024; within a batch the per-INSTANCE DrawIndex attribute
// (0..1023) selects the matrix from the bound window.
//
//   PerFrame  gFrameData : camera view*proj (bound once).
//   PerBatch  gBatch     : worlds[1024] for the current 1024-draw window (rebound per batch).
//
// Two cbuffers in SEPARATE sets get different register spaces (no overlap — two cbuffers
// in ONE set both land on b0 and collide). Matrix convention (PROVEN, column-major
// cbuffer): host uploads D3DXMATRIX bytes as-is, mul(M, v) reproduces D3DX's row-vector v*M.
#pragma once

// Point-light shadow SLOT params (slotPosRad/slotTile/bias/flicker) — the first-person forward
// path (opaque.frag's fpShadowVisibility) samples the cube atlas directly from these; the host
// binds the SAME pShadowMaskParamsCbv the compute mask uses into PerFrame gShadowParams.
#include "shadowparams.h.fsl"

#define OPAQUE_BATCH 1024   // matrices per 64KB cbuffer window; must match host kBatchSize
#define MAX_TEXTURES 880    // bindless gTextures[] array size; MUST match IPC::kMaxTextures (geomwire.h)
// NiFlipController flip books: a descriptor-array of Texture2DArrays, one element per (format,
// size) bucket, one LAYER per book frame. A 300-frame book used to claim 300 gTextures slots; it
// now claims one descriptor. Same structure as MAX_STATICS_BUCKETS below and the same reason.
// MAX_TEXTURES + MAX_STATICS_BUCKETS + MAX_FLIP_BUCKETS = 1024, the proven-OK Persistent-table
// size — these 16 came OUT of gTextures, which is why that dropped 896 -> 880.
// MUST match IPC::kMaxFlipBuckets (geomwire.h). Slot encoding: see texsample.h.fsl::isFlipSlot.
#define MAX_FLIP_BUCKETS 16
// Distant-statics texture residency: a descriptor-array of Texture2DArrays, one element per
// (format, capped-size) bucket. Each element is ONE descriptor (format/size are runtime resource
// props; HLSL sees only Texture2DArray<float4>), so this holds arrays of DIFFERENT formats+sizes
// indexed bindlessly — no switch, no per-texture descriptor. MAX_TEXTURES + MAX_STATICS_BUCKETS =
// 1024, the proven-OK Persistent-table size (see geomwire.h). statics texSlot = (bucket<<16)|layer.
#define MAX_STATICS_BUCKETS 128
// Host-owned terrain land textures: same bucketed-Texture2DArray residency as gStaticsArrays, in
// its OWN declaration (see gTerrainArrays). 499 unique LTEX over a handful of (format, capped size)
// combinations, so this is generously sized; the residency log reports actual occupancy.
#define MAX_TERRAIN_BUCKETS 32
#define MAX_POINT_LIGHTS 128 // per-frame point-light cap; MUST match IPC::kMaxPointLights (geomwire.h)
// NiUVController takeover: per-frame UV-animation table (gUVAnim). Entry id = (du, dv, setIndex, 0);
// id 0 is reserved = "no animation" (entry 0 stays zero). MUST match host kMaxUVAnim (forgerender.cpp).
#define MAX_UV_ANIM 256

STRUCT(FrameData)
{
    DATA(float4x4, viewProj, None);
    // Tier 1 lighting (per-frame, from DistantLand). All float4 for clean 16-byte cbuffer
    // packing; shaders read .xyz. sunDir = WORLD-space sun TRAVEL direction (normalized) —
    // to-sun is -sunDir, matching FFE's saturate(dot(N, -lightSunDirection)).
    DATA(float4, sunDir,     None);   // xyz = world sun travel dir
    DATA(float4, sunCol,     None);   // xyz = sun diffuse color
    DATA(float4, ambCol,     None);   // xyz = scene ambient color
    DATA(float4, fogColNear, None);   // xyz = near fog color (lerp target)
    DATA(float4, fogParams,  None);   // x = fogNearStart, y = fogNearEnd
    DATA(float4, eyePos,     None);   // xyz = world camera position (fog distance)
    // F12 debug view (offset 160B, float index 40). x: 0=normal, 1=depth world-distance grayscale.
    // Appended after eyePos so every existing field keeps its offset; 176B total < 256B CBV min.
    DATA(float4, debugParams, None);  // x = debug mode, yz = invScreen, w = toggle bits (1 AO, 2 bent-N, 4 amb=white)
    // Dev panel intensity modifiers (offset 176B, float index 44). All default 1.0 (no-op). They
    // scale per-component output of the Forge passes only, so surfaces Forge does NOT draw stay put
    // — cranking one isolates what's still on MW's own path. 192B total < 256B CBV min.
    DATA(float4, dbgScales,   None);  // x = ambient, y = diffuse, z = albedo, w = overall
    // Phase 1a distant-land LOD (host-owned DL). Appended so every field above keeps its offset.
    // lodParams.x = the near↔far POINT-LIGHT handoff radius (Phase E's `nearOwn`, camera-relative
    // world units; 0 = the dedup is off). Only terrain.frag reads it, to choose between gLightsNear
    // and gLights per fragment — see the pick there for why this exact radius makes the seam invisible.
    // lodParams.yz WERE the DL world bake's normal/detail atlas slots; the bake and its
    // distantland.vert/.frag were deleted in T4 (tasks/forge-terrain.md), so yz are now written 0
    // and read by nothing. Kept as padding rather than repacked — every field below would shift.
    // lodParams.w = nearViewRange, still LIVE: statics.vert gates the hero near-cut on it. 208B.
    DATA(float4, lodParams,   None);
    // lodSunAmb: xyz = distant-land sun ambient (XE Mod Landscape.fx sunAmb). 224B < 256B CBV min.
    DATA(float4, lodSunAmb,   None);
    // Phase 1a/1b LIVE: the REAL camera eye (absolute world). The live frame is camera-relative
    // (near worlds pre-shifted by -eye, eyePos = 0), but resident DL geometry is in ABSOLUTE world
    // coords — distantland.vert subtracts lodEye to match; statics are host-shifted so statics.vert
    // never reads it. Near/opaque/skinned/multimap paths leave it 0 and never touch it. 240B < 256B.
    DATA(float4, lodEye,      None);
    // SK2 dev "ownership tell": tint the Forge sky toward magenta so it's unmistakable that the
    // Forge host (not MW) is drawing the sky — the takeover is otherwise byte-identical. Only
    // sky.frag reads these; every other pass ignores them (0 = no tint = clean A/B). 256B == min CBV.
    //   x = tint amount 0..1, y = host time (seconds, for the pulse), z = pulse flag (0/1), w unused.
    DATA(float4, skyParams,   None);
    // Real water reflection (WV2): camera-relative below-water CLIP plane (nx,ny,nz,d) for the
    // reflect-geo pass. distantland.vert/statics.vert emit SV_ClipDistance0 = dot(plane.xyz,worldPos)+plane.w
    // so only ABOVE-water geometry reflects (seabed clipped). Pass-all (0,0,0,1) in every non-reflect
    // cbuffer (Clip = 1 >= 0 → nothing clipped); the real plane lives ONLY in pReflectFrameCbvGeo.
    // 272B struct → host frame cbuffers bumped to 512B (256B min CBV, 256B-aligned). Only these two
    // verts read it; no frag change (SV_ClipDistance is a rasterizer system-value). 272B.
    DATA(float4, gReflWaterClip, None);
    // C2 host-computed sky dome: the current interpolated ZENITH sky colour (client ships MW's
    // per-frame getCurrentWeatherSkyCol via lighting[28..31]). sky.frag's dome branch builds a
    // vertical gradient fogColNear(horizon) -> skyZenith(zenith), so the atmosphere dome no longer
    // needs its baked per-vertex gradient re-uploaded each frame. Only sky.frag reads it. 288B < 512B.
    DATA(float4, skyZenith, None);
    // Shadow-atlas debug view (F12 mode 11/12): per-slot state bitmasks as REAL uints (asuint —
    // float lanes drop bits >= 24). x = ACTIVE-slot mask (static atlas view), y = DYNAMIC-slot mask
    // (dynamic atlas view). Only shadowatlasview.frag reads these; 0 elsewhere = every tile dim. 304B.
    DATA(float4, atlasDbg, None);
    // AT1 near-opaque alpha depth prepass. x = the opacity at or above which an alpha pixel is
    // allowed to WRITE DEPTH (live knob "ALPHA: depth-write opacity"). Only alphadepth.frag reads
    // it; the alpha COLOUR pass never writes depth at all. 320B < 512B. See alphadepth.frag.fsl.
    DATA(float4, alphaParams, None);
    // UV-animated distant statics (ghostfence). x = MW SIMULATION time in seconds, pre-wrapped by the
    // host to [0, 12.5) — exactly one V cycle of the 0.08/s scroll, so the wrap is seamless and t stays
    // small enough for float32 to hold sub-frame precision. Only statics.vert reads it (a subset scrolls
    // only if its flags carry bit2, set from the NIF's NiUVController at bake time). 336B < 512B.
    // y = hero blend-pass enabled. z = Glow in the Dahrk night signal: the client's signed margin in
    // GAME HOURS into the period where GitD lights a window mesh (>0 lit, <0 dark), consumed only by
    // statics.vert's day/night variant clip (flags bits 3/4).
    // w = "fog samples the sky" silhouette SHAPE — the exponent on the extinction's fog ramp
    // (skydome.h.fsl). < 1 front-loads the darkening so ridges finish going dark BEFORE the melt
    // washes them out, which is what separates them into distinct shades; 1 = a linear ramp. Read
    // only by applyFog(); host clamps it away from 0.
    DATA(float4, timeParams, None);
    // Hero distant statics UV animation (Phase 3). The host evaluates the real NiUVController keys
    // for up to 8 UNIQUE (deduped) animations — the fence fasterA/fasterB/slower layers + lava
    // base/crust/third — into these per frame. A subset carries its 1-based slot in flags bits
    // 16-23 (statics.vert); uvOffsets[slot-1].xy is ADDED to the base UV. Appended after timeParams
    // so every existing field keeps its offset. 336 + 128 = 464B < 512B CBV. Only statics.vert reads
    // it; 0 elsewhere is a harmless null tail. Slot count MUST match kMaxHeroAnimSlots host-side.
    DATA(float4, uvOffsets[8], None);
    // Clustered forward lighting (froxel grid) for the DISTANT baked point lights. distantland.frag /
    // statics.frag read these to find their froxel (screen tile x radial-distance slice) and loop only
    // that froxel's lights from gFroxelMask, instead of the whole streamed <=128 set. Appended after
    // uvOffsets so every field above keeps its offset (464 -> 496B < 512B CBV). froxelDims.x == 0 =>
    // clustering OFF => both frags fall back to the brute gLights loop (near/opaque paths never read
    // these). Slice metric = length(worldPosRel), matching froxelassign.comp exactly.
    DATA(float4, froxelDims, None);   // x=tilesX, y=tilesY, z=NZslices, w=tileSize(px); x<=0 => brute loop
    DATA(float4, froxelZ,    None);   // x=log(d0), y=invLogRange (1/log(d1/d0)); zw unused
    // Alpha SHADOW-RECEIVE threshold (496B, float index 124). x = the opacity at/above which an alpha
    // sheet writes into the dedicated shadow-receive depth (alphashadowdepth.frag) so it receives its
    // own point-light shadow — SEPARATE from alphaParams.x (the fold-fix depth-write threshold). Only
    // alphashadowdepth.frag reads it; 0 elsewhere is inert. 512B == the 512B host CBV.
    DATA(float4, alphaShadowParams, None);
};

STRUCT(BatchData)
{
    DATA(float4x4, worlds[OPAQUE_BATCH], None);
};

// Tier 3a point lights (per-frame). One light = 3 float4 packed exactly like
// IPC::PointLightWire so the host memcpy's the wire array straight in:
//   lights[i*3+0] = float4(worldPos.xyz, radius)
//   lights[i*3+1] = float4(diffuse.rgb,  unused)   // dimmer- & pointLightMult-scaled
//   lights[i*3+2] = float4(k0, k1, k2,   unused)   // 1/(k0+k1·d+k2·d²) attenuation
// lightParams.x = active light count (host writes the actual uploaded count).
// lightParams.y = point-light REACH in radii. Attenuation is driven to exactly zero at
//   reach*radius. The host slaves this to the shadow test range (g_shadowRangeK * g_lightReachFrac,
//   frac<=1), so the lit region is always strictly INSIDE the shadow-tested region and a light can
//   never spill past the shadow it should be casting. Every frag that evaluates point lights
//   (opaque/alpha/multimap) must use it — one of them keeping a hardcoded reach = a leak on that
//   material only.
// 16 + 128*3*16 = 6160 B < the 64KB cbuffer limit.
//
// POINT_LIGHT_TAIL: where the 1->0 ramp to the reach begins, as a fraction of reach. The light is
// unchanged inside it and fades over the remainder, so shortening the reach trims the faint tail
// rather than dimming the light's core. Shared by all three point-light frags — keep it here, not
// copied into each: a per-shader reach/ramp is exactly how the leak got in.
#define POINT_LIGHT_TAIL 0.75f
STRUCT(LightData)
{
    DATA(float4, lightParams, None);                 // x = count, y = light reach in radii
    DATA(float4, lights[MAX_POINT_LIGHTS * 3], None);
    // Near clustered forward lighting (froxel grid) for the NEAR live point lights. opaque.frag /
    // multimap.frag read these to find their froxel (screen tile x radial-distance slice) and loop
    // only that froxel's lights from gFroxelMaskNear, instead of the whole uploaded <=128 set. Parked
    // in the NEAR light cbuffer (pLightCbv) — NOT gFrameData — because the distant path owns
    // gFrameData.froxelDims for the same frame; the near phase is temporally separate so its grid
    // params ride the near-specific gLights. Appended after lights[] (harmless null tail for the
    // distant/FP paths that bind a different gLights buffer). froxelDimsNear.x == 0 => clustering OFF
    // => the near frags fall back to the brute gLights loop. Slice metric = length(In.WorldPos),
    // matching froxelassign.comp exactly (near lights are already camera-relative, like In.WorldPos).
    DATA(float4, froxelDimsNear, None);   // x=tilesX, y=tilesY, z=NZslices, w=tileSize(px); x<=0 => brute loop
    DATA(float4, froxelZNear,    None);   // x=log(d0), y=invLogRange (1/log(d1/d0)); zw unused
};

BEGIN_SRT_NO_AB(SrtData)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(FrameData), gFrameData)
        // Tier 2 AO buffer (rgb = view-space bent normal, a = visibility). Read ONLY by the F12
        // debug branches (debugParams.x == 3 AO / == 4 bent normal); normal shading never touches
        // it, so modes 0/1 stay byte-for-byte unchanged. A CBV + SRV coexist in one set fine — the
        // compute SRTs already pair SRV+UAV in a single set (LinDepth/AO PerBatch). Tier 3 will read
        // this unconditionally to modulate ambient.
        DECL_TEXTURE(PerFrame, Tex2D(float4), gAO)
        // Forge water takeover (WT1). Appended AFTER gAO so gFrameData(CBV)/gAO(t0) keep their
        // offsets — opaque/skinned/multimap/sky frags never sample these (harmless null tail). Bound
        // once into pPerFrameSet (stable descriptors; the refraction/reflect CONTENTS change per
        // frame, the views don't). Only water.frag reads them.
        //   gWaterNormalVol = water_NRM.dds as a 3D animated-normal volume (rg = normal, a = height).
        //   gRefractColor   = host copy of the pre-water colour target (refraction source).
        //   gSceneLinDepth  = pLinearDepth (RAW reverse-Z DEVICE depth; near=1 far=0) for shoreline.
        //                     Filled TWICE per frame: once after the Z-prepass (near opaque + TERRAIN
        //                     — what GTAO and the point-light shadow mask read, both running before
        //                     the colour pass) and again at the colour->water seam,
        //                     which adds the DL statics that draw only in the colour pass. The second
        //                     fill is the version water.frag and volfog.frag sample.
        //   gReflectColor   = reflection RT (WT1 = stand-in / unused; WT2 fills it with the mirror pass).
        DECL_TEXTURE(PerFrame, Tex3D(float4), gWaterNormalVol)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gRefractColor)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSceneLinDepth)
        DECL_TEXTURE(PerFrame, Tex2D(float4), gReflectColor)
        // P1 point-light shadows: the screen-space visibility mask written by shadowmask.comp
        // (R32G32B32A32_UINT; 4 bits per shadow slot — uint4 lanes of 8: x = 0-7, y = 8-15,
        // z = 16-23, w = 24-31, 15 = fully lit).
        // Appended AFTER gReflectColor so every existing PerFrame offset stays stable. Read by
        // opaque.frag's light loop when a light's falloff.w lane carries slot+1 (host-patched);
        // multimap.frag joins in P4. All other frags ignore it (harmless null tail elsewhere).
        DECL_TEXTURE(PerFrame, Tex2D(uint4), gShadowMask)
        // Shadow-atlas debug view (F12 mode 11/12): the two D32 shadow atlases sampled read-only
        // by shadowatlasview.frag. Appended AFTER gShadowMask so every existing PerFrame offset
        // stays stable; only the debug-view frag references them (harmless null tail elsewhere —
        // they are bound ONLY into the main pPerFrameSet, like the water/mask SRVs).
        DECL_TEXTURE(PerFrame, Tex2D(float), gShadowAtlas)
        DECL_TEXTURE(PerFrame, Tex2D(float), gShadowAtlasDyn)
        // SUN (directional) shadow moments map — the DL-statics-primary MSM map, RGBA16_UNORM
        // (packed 4 moments). Phase A: read only by sunshadowview.frag (F12 mode 13 blit). Phase B:
        // the opaque/alpha receiver samples it for sun shadows. Bound into pPerFrameSet like the
        // atlas views; harmless null tail elsewhere.
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSunMoments)
        // ...and the SAME atlas's raw D32 depth — the depth buffer the caster pass already z-tests
        // against, which up to now was written and thrown away. The near cascade's PCSS/PCF reads it
        // directly (msmrecv.h.fsl), so hard-surface softness costs no extra pass and no extra VRAM.
        // Declared Tex2D(float) exactly like gShadowAtlas (also a D32 render target sampled as an
        // SRV). Rests in SHADER_RESOURCE like gSunMoments; only DEPTH_WRITE inside the caster pass.
        DECL_TEXTURE(PerFrame, Tex2D(float), gSunDepth)
        // Clustered forward lighting: the froxel light-mask (128 bits/froxel = 4 uint), written by
        // froxelassign.comp (UAV) and read here as an SRV. Appended AFTER gShadowAtlasDyn so every
        // existing PerFrame offset stays stable. Bound once into pPerFrameSet (like gShadowMask); only
        // distantland.frag / statics.frag index it, and only when gFrameData.froxelDims.x > 0 (else the
        // brute loop). All other frags ignore it (harmless null tail).
        DECL_BUFFER(PerFrame, Buffer(uint), gFroxelMask)
        // Near clustered forward lighting: a SECOND froxel light-mask (128 bits/froxel = 4 uint) for
        // the NEAR live point lights, written by froxelassign.comp into the near mask (UAV) and read
        // here as an SRV. Kept separate from gFroxelMask (which the distant path builds AFTER the near
        // colour phase within the same frame) so the near and distant grids never entangle their
        // buffer state. Appended AFTER gFroxelMask so every existing PerFrame offset stays stable; only
        // opaque.frag / multimap.frag index it, and only when gLights.froxelDimsNear.x > 0 (else the
        // brute loop). All other frags ignore it (harmless null tail).
        DECL_BUFFER(PerFrame, Buffer(uint), gFroxelMaskNear)
        // NiUVController takeover: the per-frame UV-animation table — float4[MAX_UV_ANIM] of
        // (du, dv, setIndex, 0), evaluated host-side from MW sim time (one entry per animated
        // draw; id rides the instance stream — opaque word[0] bits 16+, multimap Meta bits
        // 24-31; id 0 = no animation). Appended AFTER gFroxelMaskNear so every existing
        // PerFrame offset stays stable. Read by opaque.vert / multimap.vert only when the
        // stamped id != 0 (harmless null tail everywhere else).
        DECL_BUFFER(PerFrame, Buffer(float4), gUVAnim)
        // First-person shadow reception (direct cube-atlas test). The FP arms are a post-composite
        // overlay drawn under the ARM camera — the screen-space gShadowMask reconstructs the WORLD
        // behind each arm pixel, not the arm, so opaque.frag's FP path (gFrameData.alphaShadowParams.y
        // set) instead samples gShadowAtlas/gShadowAtlasDyn DIRECTLY using the interpolated world pos +
        // vertex normal, driven by these per-slot params. A 2nd CBV in the set → register b1 (gFrameData
        // is b0); textures/buffers keep their registers. Host binds the SAME pShadowMaskParamsCbv the
        // compute mask uses (slots are camera-relative → identical for the arm view). Read only when the
        // FP flag is set (harmless everywhere else). MUST be bound into every SrtData PerFrame instance.
        DECL_CBUFFER(PerFrame, CBUFFER(ShadowMaskParams), gShadowParams)
        // AT3 multi-stage: per-draw FFE texture stages BEYOND the base map, indexed by the alpha
        // draw's own instance index (opaque.vert forwards it as DrawIdx). xyz = packMMStage words
        // (texIndex | uvSet | op | clampMode — the SAME encoding Route C's MultiMapDrawWire uses,
        // so the decode below is shared with multimap.frag), w = how many of them are live.
        // w == 0 means "base map only" and is the case for every cached alpha draw, so an unbound
        // or stale table degrades to the pre-existing single-map look rather than to garbage.
        // Only alpha.frag reads it (harmless null tail everywhere else). Appended LAST so every
        // existing PerFrame offset stays stable.
        DECL_BUFFER(PerFrame, Buffer(uint4), gAlphaStages)
        // Host-owned TERRAIN residency (tasks/forge-terrain.md). The whole world's LAND heightfield
        // and hand-painted vertex colour, uploaded ONCE at first exterior and never streamed — 3910
        // cells is ~99 MB, which simply fits, so there is no residency scheme to get wrong.
        //   gTerrainHeights: two int16 (VHGT units) per uint, cell stride 2113 uints.
        //   gTerrainColor  : one 0x00BBGGRR per vertex,       cell stride 4225 uints.
        // Both index as slot*stride + (y*65 + x). Read by terrain.vert ONLY (a null tail for every
        // other shader on this root signature); appended LAST so every existing PerFrame offset
        // stays stable. In the PerFrame set rather than Persistent because that set's SRV table is
        // already at its proven-OK 1024 entries (gTextures + gStaticsArrays + gFlipArrays).
        DECL_BUFFER(PerFrame, Buffer(uint), gTerrainHeights)
        DECL_BUFFER(PerFrame, Buffer(uint), gTerrainColor)
        // ...and the LAND texture layout: gTerrainTex holds each cell's 16x16 VTEX as ONE texture
        // SLOT per uint (cell stride 256) — the LTEX ids are resolved to (bucket<<16)|layer once at
        // residency build, so the frag's hot path is one load, not a load plus an id->slot
        // indirection. One slot per uint, not two packed: a slot needs the full 32 bits, and
        // packing two per uint truncated every real texture to 0 (white).
        // gTerrainCellGrid maps a WORLD grid coord to slot+1 (0 = no cell); it is what
        // lets the frag reach into a NEIGHBOUR cell's VTEX, which Morrowind's own vertex->texture-
        // square rounding requires at every cell edge — without it the world gets a texture seam
        // every 8192 units.
        DECL_BUFFER(PerFrame, Buffer(uint), gTerrainTex)
        DECL_BUFFER(PerFrame, Buffer(uint), gTerrainCellGrid)
        // Land textures: array of Texture2DArrays bucketed by (format, capped size), exactly the
        // gStaticsArrays shape and for the same reason — 499 unique LTEX will not fit in individual
        // bindless slots (MAX_TEXTURES is 880 with the near scene already contending). Its own
        // declaration, NOT gStaticsArrays: the statics bake is itself on the way out.
        // Slot encoding is the same (bucket<<16)|layer. Declared LAST in the set.
        DECL_ARRAY_TEXTURES(PerFrame, Tex2DArray(float4), gTerrainArrays, MAX_TERRAIN_BUCKETS)
        // SH2 top-down world HEIGHT map (skyamb.h.fsl / skyheight.comp.fsl): one R16F world height
        // per texel — max(terrain, statics) — over a 65536-unit window that follows the camera.
        // Read by every lit path through skyAmbFactor(), which marches a few taps of it to get the
        // sky occlusion GTAO cannot reach (it only sees what is on screen) and the SH does not model
        // (it assumes the whole hemisphere is visible). Its world mapping rides
        // gShadowParams.skyAOMap, so this texture carries no state of its own.
        //
        // DECLARED LAST, and that is load-bearing, not tidiness: FSL assigns every resource in a set
        // its mOffset from ONE running per-set counter, so inserting a declaration anywhere above
        // silently re-points every bind after it. Append only.
        DECL_TEXTURE(PerFrame, Tex2D(float), gSkyHeight)
        // LONG-RANGE sun occlusion (sunocc.comp.fsl): the sun-BLOCKED world Z per texel, derived
        // from gSkyHeight over the SAME window. One sample says whether a point — on the ground or
        // in the air — is in the shadow of anything up-sun of it, out to the whole 65536-unit map
        // rather than the cascades' one cell. Read by msmrecv.h.fsl's two receivers (the colour pass
        // past the last cascade, and the volumetric march everywhere). Its mapping rides
        // gShadowParams.skyAOMap + sunOcc, so this texture carries no state of its own.
        // Appended after gSkyHeight — append only, see the note above it.
        DECL_TEXTURE(PerFrame, Tex2D(float), gSunOcc)
        // "Fog samples the sky" (skydome.h.fsl): a copy of the colour target taken RIGHT AFTER the
        // SK1 sky pass — which is drawn FIRST in the colour pass, depth off — so this texture is
        // literally "what is behind this fragment", clouds/moons/sun glare and all. Fog lerps toward
        // it at the fragment's own screen position, which is what makes a distant surface converge on
        // the exact pixel it is covering instead of on an analytic dome that has to be kept in sync.
        //
        // PREMULTIPLIED (rgb·a, a): the sky blends SRCALPHA/INVSRCALPHA into a transparent-cleared
        // target, so a == sky coverage and a == 0 means the dome does not reach that pixel (the band
        // MW's own flat fog owns). An UNBOUND or stale binding therefore reads (0,0,0,0) → a = 0 →
        // the helper returns fogColNear → exactly the flat-fog behaviour this replaced. The failure
        // mode is the old image, not a black one.
        //
        // Per-pass: the main sets get the screen copy, pPerFrameSetReflectGeo gets the MIRROR's own
        // sky copy, so both sides of the water melt into their own sky. The texture's inverse extent
        // rides gFrameData.fogParams.zw (per-pass, because gFrameData is the per-pass cbuffer while
        // gShadowParams is shared) — see skydome.h.fsl.
        //
        // Appended AFTER gSunOcc — append only, see the note above gSkyHeight.
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSkyColor)
    END_SRT_SET(PerFrame)
    // Point-light cbuffer — rides the otherwise-unused PerDraw set (FSL has exactly four
    // fixed update frequencies: Persistent/PerFrame/PerBatch/PerDraw; a custom set name has
    // no register space). Its own set ⇒ b0 in spaceSET_PerDraw, no collision with PerFrame's
    // b0. Read by opaque.frag for both the static and skinned paths.
    BEGIN_SRT_SET(PerDraw)
        DECL_CBUFFER(PerDraw, CBUFFER(LightData), gLights)
        // The NEAR live light list, bound ALONGSIDE gLights on EVERY PerDraw instance. Every other
        // path picks its list at bind time — near geometry takes the live set, distant geometry the
        // baked one — because every other path lives on one side of the handoff. Terrain doesn't: the
        // ground runs from underfoot to the horizon in a single draw, so it is the one surface that
        // spans both, and it needs both lists resident to pick per fragment (terrain.frag).
        //
        // Declared LAST, so gLights keeps offset 0 and every existing bind is untouched (FSL assigns
        // offsets from ONE running per-set counter — inserting mid-set silently re-points the rest).
        // The host points this at pLightCbv in all three instances, so the near/dist/FP choice for
        // gLights is unaffected; only terrain.frag reads it.
        DECL_CBUFFER(PerDraw, CBUFFER(LightData), gLightsNear)
    END_SRT_SET(PerDraw)
    // Bindless base-map textures only — NO dynamic sampler here. The frag samples with the FSL
    // built-in STATIC sampler gSamplerAnisotropic (anisotropic 8x, WRAP, baked into the root sig).
    //
    // Why no dynamic sampler: FSL assigns each resource's reflected mOffset from a SINGLE running
    // per-set counter, but textures (SRV heap) and samplers (sampler heap) live in SEPARATE D3D12
    // descriptor tables. With a 1024-entry array declared first, gTextures gets mOffset 0 (correct,
    // register t0) but a sampler declared after it gets mOffset 1024 — the host's bind then lands at
    // sampler-table slot 1024 while the shader reads s0 (slot 0 = null sampler => POINT filter +
    // non-REPEAT address => tiled UVs sample black). The two can't BOTH sit at offset 0 in one set,
    // so the sampler is hoisted to a static sampler instead. (The mirror of the original
    // sampler-before-array bug; see [[project_forge_bindless_textures]].)
    BEGIN_SRT_SET(Persistent)
        DECL_ARRAY_TEXTURES(Persistent, Tex2D(float4), gTextures, MAX_TEXTURES)
        // Distant-statics: array of Texture2DArrays (bucketed by format/size). Declared AFTER
        // gTextures so it stacks at SRV offset MAX_TEXTURES (FSL single per-set counter); the host
        // binds it via SRT_RES_IDX(...,gStaticsArrays). statics.frag samples gStaticsArrays[bucket].
        DECL_ARRAY_TEXTURES(Persistent, Tex2DArray(float4), gStaticsArrays, MAX_STATICS_BUCKETS)
        // Flip-book frames (NiFlipController). Declared LAST so it stacks at SRV offset
        // MAX_TEXTURES + MAX_STATICS_BUCKETS. sampleBase() routes here on the slot's flag bit, so
        // every world path (opaque/alpha/depth/shadow/multimap) picks it up from that one decode.
        DECL_ARRAY_TEXTURES(Persistent, Tex2DArray(float4), gFlipArrays, MAX_FLIP_BUCKETS)
    END_SRT_SET(Persistent)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER(PerBatch, CBUFFER(BatchData), gBatch)
    END_SRT_SET(PerBatch)
END_SRT(SrtData)
