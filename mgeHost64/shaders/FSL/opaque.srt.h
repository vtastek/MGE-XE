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

// PER-VIEW sky data for the physical (Hosek-Wilkie) sky pass — a STRUCT header, no SRT of its own.
// It has to be a second cbuffer rather than more lanes in gShadowParams for exactly one reason: it
// carries THIS view's invViewProj, and gShadowParams is bound by POINTER into every PerFrame set
// (which is what let P1's anchor reach the mirror for free, and is what makes it useless here — the
// mirror would reconstruct main-view rays). See skyview.h.fsl.
#include "skyview.h.fsl"

// THE ATMOSPHERE's parameter block (tasks/forge-atmosphere.md S2c). The STRUCT only — atmosphere.h.fsl
// carries the medium itself and is HLSL bodies, while this header is compiled as C++ by the host.
// skyhw.frag needs both: the params to deparameterise a direction into a sky-view uv, and the medium
// to do it. Sited here rather than on the sky pass's own set because S3's applyFog() reads the same
// two through the SAME PerFrame set — aerial perspective is the same integral as the sky, which is
// the whole reason the fog and the horizon stop being two colours.
#include "atmosparams.h.fsl"

// Bent-normal strength used to live here as AO_BENT_INTENSITY, a compile-time gain on gAO.rgb. It
// is gone: the strength is a PRODUCER-side knob now (aocommon.h.fsl's aoBentStrength, a live slider
// in the AO panel), which is strictly better placed. Compute shaders hot-reload on F8, so it can be
// tuned in the running game — whereas this file's consumers are graphics shaders and retuning here
// cost an FSL recompile AND a host restart. It could not have moved into FrameData either: that
// struct is exactly 512 B against a 512 B CBV.
//
// The consumers now reconstruct rather than gain: `normalize(N * gAO.a + gAO.rgb)`, which is exact.
// See aocommon.h.fsl for what gAO.rgb means and opaque.frag for why our normal carries the gAO.a
// weight. opaque.frag and multimap.frag must stay in lockstep; alpha.frag / multimap_alpha.frag do
// not apply the bend at all (they are not in the Z-prepass gAO is built from).

#define OPAQUE_BATCH 1024   // matrices per 64KB cbuffer window; must match host kBatchSize
#define MAX_TEXTURES 4064   // bindless gTextures[] array size; MUST match IPC::kMaxTextures (geomwire.h),
                            // which says why (a dense exterior grid is ~1100 textures) and why < 4096
// NiFlipController flip books: a descriptor-array of Texture2DArrays, one element per (format,
// size) bucket, one LAYER per book frame. A 300-frame book used to claim 300 gTextures slots; it
// now claims one descriptor. Same structure as MAX_STATICS_BUCKETS below and the same reason.
// MUST match IPC::kMaxFlipBuckets (geomwire.h). Slot encoding: see texsample.h.fsl::isFlipSlot.
#define MAX_FLIP_BUCKETS 16
// Distant-statics texture residency: a descriptor-array of Texture2DArrays, one element per
// (format, capped-size) bucket. Each element is ONE descriptor (format/size are runtime resource
// props; HLSL sees only Texture2DArray<float4>), so this holds arrays of DIFFERENT formats+sizes
// indexed bindlessly — no switch, no per-texture descriptor. Shares the Persistent table with
// gTextures (see geomwire.h kMaxTextures). statics texSlot = (bucket<<16)|layer.
#define MAX_STATICS_BUCKETS 128
// Host-owned terrain land textures: same bucketed-Texture2DArray residency as gStaticsArrays, in
// its OWN declaration (see gTerrainArrays). 499 unique LTEX over a handful of (format, capped size)
// combinations, so this is generously sized; the residency log reports actual occupancy.
#define MAX_TERRAIN_BUCKETS 32
// DISTANT-STATICS PBR: the `_paramh` companions of the statics library, in their own bucket set
// (gStaticsParamArrays, PerFrame). SIZED BY COVERAGE, not by the albedo's 128: only the LOD textures
// that ship a companion are planned here — 116 of 2008 in the shipped bake — so this needs far fewer
// (format, capped-size) combinations than the albedo does. The residency log reports occupancy and
// says so loudly if it overflows.
#define MAX_STATICS_PARAM_BUCKETS 48
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
    // ⚠ THE NAME IS HISTORICAL (S2j, tasks/forge-atmosphere.md). This carried C2's zenith sky colour
    // for sky.frag's host-gradient dome, which SK4 retired; S2j deleted that branch and re-used the
    // lanes. The host keeps MW's zenith colour for the legacy SH projection in a host global.
    //   x = the physical sky's native -> scene scale for the FOG's haze target (skydome.h.fsl's
    //       fogHazeTarget); 0 = no physical sky this frame -> fog targets fogColNear (pre-S2j image).
    //   y = this view's eye z relative to the worldPosRel origin: 0 main, 2*dRel in the water mirror.
    //   z = sin(near-field haze lift): the minimum elevation a NEAR fragment's haze is read at,
    //       ramped to 0 at the knee in-shader (knob fogHazeLiftDeg; 0 = unlifted).
    //   w = the enchanted-item glow word (enchantglow.h.fsl) — unchanged.
    // 288B < 512B.
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
    // ⚠ .zw ARE THE TWO FOG LOOK LANES, AND THEY LIVE IN A FROXEL ROW BECAUSE FrameData IS EXACTLY
    // FULL — alphaShadowParams ends at 512B, which IS the host CBV, so there is no room to append and
    // these two zeroed spares are what is left. Named here rather than smuggled: both are ARTIST
    // knobs, both default to a value that reproduces the previous image exactly, and both exist
    // because the constants they replace were `#define`s in skydome.h.fsl / fog.h.fsl that reach
    // opaque/statics/terrain/alpha/multimap — none of which hot-reload — so dialling either one cost
    // a HOST RESTART. skydome.h.fsl's own comment asked for this ("move it to a lane if it needs
    // dialling in play"); it needs dialling in play.
    //
    //   z = THE SKY-TARGET SATURATION KNEE (skydome.h.fsl's fogSkyShare). Below it the fog target is
    //       flat fogColNear; above it the sampled sky fades in. 0.85 = the shipped look. LOWER brings
    //       the sky in earlier and closes the gap between a fogged static (which melts to MW's flat
    //       colour) and the water's reflection-hole fill (fogSkyColor, share 1.0, no knee) sitting
    //       right beside it in the mirror — reported as "static reflections are fogged brighter blue
    //       while statics assume MW fog color". 1.0 = never sample the sky at all.
    //   w = NEAR-FIELD HAZE DENSITY, per world unit (fog.h.fsl's mwFogRamp). MW's ramp is exactly 1
    //       (clear) for everything closer than fogStart — ~490 m in clear weather at 16 cells — so
    //       there is a large dead zone with NO fog at all, reported as "close fog being ignored".
    //       This is a Beer-Lambert term that starts at the EYE and multiplies the ramp, so it fills
    //       the dead zone without moving the fog wall (where the ramp is already ~0). 0 = off, and
    //       off is bit-identical.
    DATA(float4, froxelZ,    None);   // x=log(d0), y=invLogRange (1/log(d1/d0)); z=sky knee, w=near haze
    // 496B, float indices 124..127. The LAST float4 in FrameData — 512B == the 512B host CBV.
    //   x (124) alpha SHADOW-RECEIVE threshold: the opacity at/above which an alpha sheet writes into
    //           the dedicated shadow-receive depth (alphashadowdepth.frag) so it receives its own
    //           point-light shadow. SEPARATE from alphaParams.x (the fold-fix depth-write threshold).
    //           Only alphashadowdepth.frag reads it; 0 elsewhere is inert.
    //   y (125) FP direct-atlas shadow flag = "first person AND receiving shadows". Drives
    //           opaque.frag's fpShadowVisibility. NOT an "am I the arm?" test — it is 0 in the FP
    //           pass whenever FP shadow reception is off or the atlas is not resident.
    //   z (126) FIRST-PERSON pass, unconditionally. That IS the "am I the arm?" test, and gAO needs
    //           it: the arms are never in the world Z-prepass, so every screen-space buffer built
    //           from that depth describes the world behind them (opaque.frag gates the gAO read on
    //           it). Two flags because the two questions came apart the first time .y was reused.
    //   w (127) spare.
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
    // G3 fixture gobos for the NEAR list (tasks/forge-light-gobo.md): light 2k in .xy, 2k+1 in .zw,
    // each (gobo layer+1, rotation bits as asfloat — gobosample.h.fsl). The near entries' own spare
    // lanes are taken (colour.w identity, falloff.w shadow slot), hence a tail. Zero on the baked
    // (distant) list, which carries its gobo in-entry instead. lightParams.z = goboForceNear.
    DATA(float4, goboNear[MAX_POINT_LIGHTS / 2], None);
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
        //   gWaterNormalVol = the W3 wave field: water_NRM.dds BOMBED at launch into a 1024²x32 RG8
        //                     Texture2DArray of SLOPES (see forgerender.cpp::bakeWaterField). It is an
        //                     ARRAY, not a Texture3D, and that is load-bearing twice over:
        //                       - FSL ships NO SampleTex3D for D3D at all (d3d.h:621 is commented out),
        //                         so a 3D resource can only be sampled at an EXPLICIT LOD, which
        //                         bypasses anisotropy by definition. SampleTex2DArray exists, so the
        //                         array gets real hardware AF through gSamplerAnisotropic for free —
        //                         9 taps per water pixel becomes 4.
        //                       - a Texture3D's mips reduce the SLICE axis, i.e. they low-pass the
        //                         ANIMATION and bank temporal variance as if it were spatial roughness.
        //                         An array's cannot.
        //                     ⚠ An array does NOT filter across slices; water.frag lerps the two time
        //                     slices by hand.
        //   gRefractColor   = host copy of the pre-water colour target (refraction source).
        //   gSceneLinDepth  = pLinearDepth (RAW reverse-Z DEVICE depth; near=1 far=0) for shoreline.
        //                     Filled TWICE per frame: once after the Z-prepass (near opaque + TERRAIN
        //                     — what GTAO and the point-light shadow mask read, both running before
        //                     the colour pass) and again at the colour->water seam,
        //                     which adds the DL statics that draw only in the colour pass. The second
        //                     fill is the version water.frag and volfog.frag sample.
        //   gReflectColor   = reflection RT (WT1 = stand-in / unused; WT2 fills it with the mirror pass).
        DECL_TEXTURE(PerFrame, Tex2DArray(float4), gWaterNormalVol)
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
        // stays stable. In the PerFrame set rather than Persistent because that set's SRV table was
        // then capped at 1024 entries (gTextures + gStaticsArrays + gFlipArrays; lifted 2026-09-30).
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
        // bindless slots (MAX_TEXTURES was 880 with the near scene already contending). Its own
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
        // WT4d/P2: the PRE-FILTERED planar reflection — pReflectColor box-averaged into a 6-level
        // mip chain by reflectmip.comp. Water is the host's first PBR material, and roughness is
        // only meaningful if the reflection LOD and the specular lobe width are the SAME number;
        // this is the reflection half of that. Sampled with gSamplerTrilinearClamp (s4) at
        // log2(alpha * kReflectSize * scale), so calm water still resolves to LOD 0 = a sharp
        // mirror. A SEPARATE texture from gReflectColor because that RT's format and PSO are shared
        // with the main colour pass (see reflectmip.srt.h); it falls back to gReflectColor's own
        // texture if the pyramid failed to build, which is the pre-WT4d image.
        //
        // Read by water.frag ONLY, bound ONLY into the main pPerFrameSet — the same arrangement as
        // the four WT1 water SRVs above, and a harmless null tail on every other instance.
        // Appended AFTER gSkyColor — append only, see the note above gSkyHeight.
        DECL_TEXTURE(PerFrame, Tex2D(float4), gReflectMips)
        // ...and the ROUGHNESS half of the same feature: the sub-texel slope VARIANCE of the wave
        // field, per texel per mip, over a box chain built alongside it (Olano-Baker LEAN/CLEAN).
        // One scalar per texel (R16F), so it is declared float, not float4.
        //
        // ⚠ HALF RESOLUTION AND ONE LEVEL SHORT, deliberately. sigma²(0) is identically ZERO by
        // construction (a single texel has no internal variance), so storing it is pure waste: this
        // array's levels 0..7 ARE the 1024² field's levels 1..8, and water.frag reconstructs the whole
        // sub-level-1 range from the known V(0) = 0 with NO extra sample. 179 MB -> 111 MB.
        //
        // ⚠ THE FINISHED VARIANCE, not the second moment. sigma² = E[|s|²] - |E[s]|² is evaluated on
        // the CPU at load, where both operands are exact. It must NOT be reassembled here from two
        // sampled moments: bilerp(|s|²) - |bilerp(s)|² is the bilinear-weighted variance (Jensen),
        // which is nonzero even at mip 0, oscillates with sub-texel position, and vanishes exactly
        // at texel centres — so it reads as blur-up-close, pulsing, and a texel-aligned GRID of
        // sharper lines. All three were observed. See the long note in water.frag.fsl.
        //
        // Read by water.frag ONLY; bound ONLY into the main pPerFrameSet. If the companion volume
        // failed to build this slot is left UNBOUND, which reads zero — and zero variance degrades
        // to exactly the base roughness. No flag and no fallback binding: deliberately NOT a
        // stand-in texture, since a 2D or float4 resource in a Tex2DArray(float) slot is the type
        // mismatch the gWaterNormalVol comment above already calls illegal.
        // Appended AFTER gReflectMips — append only, see the note above gSkyHeight.
        DECL_TEXTURE(PerFrame, Tex2DArray(float), gWaterSlopeVar)
        // W4c: the mirror pass's own depth (pReflectDepth, 1024² D32 reverse-Z), so water can ask
        // HOW FAR AWAY the thing it is reflecting actually is.
        //
        // The reflection blur is `2*alpha * L/(L+d)`, L = water->reflected object, d = camera->water:
        // a ray deflected at the surface sweeps L before it lands, and that sweep subtends less angle
        // the closer the object is. Assuming L -> inf (the shipped model) makes the blur constant
        // along a reflection, when in reality it GROWS from zero at the waterline to full at the tip —
        // the single most recognisable thing about a reflection in water.
        // Under the mirror the reflected object sits at `d + L` from the camera, so this depth IS
        // `L + d` and the factor is just `1 - dist/D_refl`. Sky (un-drawn, reverse-Z 0 = far) gives 1,
        // the shoreline touching the water gives 0. Both correct, neither special-cased.
        //
        // ⚠ Read by water.frag ONLY, bound ONLY into the main pPerFrameSet, and left UNBOUND when the
        // reflect pass failed to build — which reads ZERO = reverse-Z far = factor 1 = exactly the
        // pre-W4c behaviour. Same deliberate no-fallback arrangement as gWaterSlopeVar above.
        // Appended AFTER gWaterSlopeVar — append only, see the note above gSkyHeight.
        DECL_TEXTURE(PerFrame, Tex2D(float), gReflectDepth)
        // R2b: the actor-ripple wave field (ripplesim.srt.h). .x height .y velocity .zw slope; the
        // frag reads only .zw. A world-space grid around the camera, 1 unit/texel, advanced by two
        // compute dispatches per frame — which is the whole point of it existing, since evaluating
        // the same wakes analytically cost the water pass 0.31 -> 32.81 ms.
        //
        // ⚠ Read by water.frag ONLY and bound ONLY into the main pPerFrameSet. Left UNBOUND when
        // the sim failed to build, which reads ZERO = flat water = exactly the pre-R2b surface.
        // Same deliberate no-fallback arrangement as gWaterSlopeVar and gReflectDepth above.
        // Appended AFTER gReflectDepth — append only, see the note above gSkyHeight.
        DECL_TEXTURE(PerFrame, Tex2D(float4), gRippleField)
        // R2c: the DISPERSIVE wake field — the same layout as gRippleField above and read the same
        // way, but a coarser, longer-ranged grid (512² @ 8 units/texel) stepped by ripplewave.comp.
        // Two fields and not one because the Kelvin wedge needs omega^2 = g|k|, whose kernel cannot
        // reach a 90-200 unit wavelength at 1 unit/texel, while 8 units/texel puts the near-field
        // splash below Nyquist. Neither grid can do the other's job; the frag sums their slopes.
        // Same unbound-reads-zero-is-flat-water arrangement. Appended AFTER gRippleField.
        DECL_TEXTURE(PerFrame, Tex2D(float4), gWakeField)
        // STENCIL "FAKE HOLE" PORTAL gate (see IPC::kDrawPortalMask). One bit per pixel: 1 where a
        // portal MASK quad passed its own depth test this frame, i.e. where the opening is actually
        // VISIBLE. portalhull.frag discards on it, which is how the hull's depth-test-off write gets
        // confined to the opening — the job MW gives the stencil buffer, which pDepth (D32_SFLOAT,
        // no stencil plane) cannot do.
        //
        // The gate RT must share pDepth's sample count to be bound alongside it, so the type forks
        // on PORTAL_GATE_SAMPLES exactly like linearizedepth.srt.h's gSceneDepth — its own macro
        // rather than SAMPLE_COUNT so that nothing else on this root signature can ever flip it by
        // accident. Both branches are ONE SRV slot, so the merged root signature is identical
        // across the variants.
        //
        // ⚠ Read by portalhull.frag ONLY and bound ONLY into the main pPerFrameSet, the same
        // arrangement as gRippleField/gWakeField above. Unbound reads ZERO — gate 0 = the hull
        // discards everywhere = no punch at all = exactly the pre-portal image. The fail-safe
        // direction, and the reason there is no fallback path. Appended AFTER gWakeField.
#ifndef PORTAL_GATE_SAMPLES
#define PORTAL_GATE_SAMPLES 1
#endif
#if PORTAL_GATE_SAMPLES > 1
        DECL_TEXTURE(PerFrame, Tex2DMS(float4, PORTAL_GATE_SAMPLES), gPortalGate)
#else
        DECL_TEXTURE(PerFrame, Tex2D(float4), gPortalGate)
#endif
        // W23: the TILING CAUSTIC map (caustic.srt.h), one array slice per depth. Read by
        // waterLightTransmit() in waterfog.h.fsl, which is the single place the sun is attenuated on
        // submerged geometry — so folding it in there reaches opaque/terrain/alpha/multimap/statics
        // with no call-site churn at all.
        //
        // ⚠⚠ STORED AS (gain - 1): ZERO IS THE IDENTITY HERE, AND THAT INVERTS THE HOUSE RULE.
        // Everywhere else on this set the convention is "unbound reads zero = the pre-feature image",
        // and it works because those terms are ADDITIVE or are slopes. This one MULTIPLIES sunlight,
        // so an unbound read of 0 would not mean "no caustics", it would mean "no sun" — every
        // submerged surface black. There are seven PerFrame sets that shade submerged geometry and
        // this is bound into a subset of them, so that is not a hypothetical. Subtracting 1 at the
        // resolve and adding it back at the consumer keeps the convention pointing the right way:
        // unbound -> 0 -> gain 1.0 -> exactly the pre-W23 image.
        //
        // Sampled with gSamplerBilinearWrap: the map TILES by construction (integer wavevectors on
        // the 2*pi/tile lattice), so REPEAT addressing is exact rather than a papered seam. The array
        // index is not filtered by the sampler, so the consumer lerps two slices by hand.
        // Appended AFTER gPortalGate — append only, FSL assigns descriptor offsets from one running
        // counter and an insertion silently re-points every later binding.
        DECL_TEXTURE(PerFrame, Tex2DArray(float), gCausticField)
        // G7: the GRASS CRUSH field (grasscrush.srt.h) — a world-locked clearance-height map that
        // says, per texel, how low the lowest occupant surface over it is. .x = clearance relative to
        // a snapped crushOriginZ, .yz = lay direction, .w = validity.
        //
        // ⚠ Read by grass.vert ONLY, and bound into exactly the two PerFrame sets a grass draw ever
        // uses: the main set and every cascade instance of pPerFrameSetSun (the caster shares the
        // vertex shader, which is what keeps the shadow attached to the folded blade). Left UNBOUND
        // everywhere else, which reads ZERO — and zero is .w = 0 = no crush = exactly the pre-G7
        // image. The convention points the right way here without the gain-1 inversion gCausticField
        // needed, because validity is its own channel rather than inferred from a magnitude; see
        // grasscrush.srt.h for why that had to be so.
        // Appended AFTER gCausticField — append only, FSL assigns descriptor offsets from one running
        // counter and an insertion silently re-points every later binding.
        DECL_TEXTURE(PerFrame, Tex2D(float4), gGrassCrush)
        // ─── THE ATMOSPHERE (tasks/forge-atmosphere.md S2c) ──────────────────────────────────────
        // The SKY-VIEW LUT — the sky's radiance in every direction from the camera, rebuilt every
        // frame by atmos_skyview.comp. skyhw.frag samples this instead of evaluating a closed form,
        // and atmos_sh.comp projects THE SAME TEXTURE into the SH that lights the world: the sky you
        // see and the light it casts are one object, so they cannot drift apart the way two
        // evaluations of one model could. That is the whole of what S2 buys structurally.
        //
        // An unbound read is (0,0,0,0) = a black sky. Distinguishable from a legitimately black sky
        // only through gSkyView.params.x, which is the hard "the pass is armed" gate and the reason
        // that lane exists.
        // Appended AFTER gGrassCrush — append only, FSL assigns descriptor offsets from one running
        // counter and an insertion silently re-points every later binding.
        DECL_TEXTURE(PerFrame, Tex2D(float4), gAtmosSkyView)
        // ...and the TRANSMITTANCE LUT. Read by the F12 15/16 debug view, and — from S3 — by every
        // consumer that needs the sun's colour AT A POINT rather than at the camera: the aerial
        // perspective march, and S4's cloud lighting, which needs the beam at cloud altitude. It
        // rides the same set as the sky-view LUT because those two are always wanted together and a
        // second binding site for one of them is a second place to get the pair half-right.
        DECL_TEXTURE(PerFrame, Tex2D(float4), gAtmosTransmittance)
        // ...and the medium's own parameters, needed to turn a world direction into a uv in it (the
        // LUT is parameterised by zenith angle and azimuth-relative-to-the-sun, not by direction).
        // The SAME cbuffer the LUT cooks read, bound here as well rather than copied — for the reason
        // hosekcheck.srt.h gave about sharing gSkyView: a consumer reading its own private copy would
        // prove the arithmetic and say nothing about whether the parameters reaching the frag are the
        // ones the LUTs were built from, which is half of what can go wrong.
        DECL_CBUFFER(PerFrame, CBUFFER(AtmosphereParams), gAtmosParams)
        // ─── M1 MOTION VECTORS (tasks/forge-upscale.md) ──────────────────────────────────────────
        // Camera-only screen-space motion, in RENDER-rect PIXELS, previous-minus-current. Written by
        // motionvectors.comp at the colour->water seam.
        //
        // ⚠ READ BY THE F12 DEBUG VIEW (mvview.frag) AND BY NOTHING ELSE — for now. It is bound here
        // rather than given the debug view a private SRT because that is the pattern every other
        // looked-at buffer in this renderer already follows (gShadowMask, gAtmosSkyView, gSkyHeight),
        // and a debug view on a private set would be a second place the resource has to be wired.
        //
        // ⚠ ONE FRAME'S CONTENT IS ONE FRAME'S CONTENT. The colour frags run BEFORE the dispatch that
        // fills this, so any colour-pass reader would see LAST frame's vectors. mvview.frag is drawn
        // in the post-everything overlay block, after the dispatch, and is therefore current — do not
        // move a reader of this slot earlier without re-checking that.
        //
        // Appended AFTER gAtmosParams — append only, FSL assigns descriptor offsets from one running
        // counter and an insertion silently re-points every later binding.
        // Unbound in the sets that never display it, which reads (0,0) = "nothing moved" — the same
        // benign default gGrassCrush documents, and the correct one for a vector field.
        DECL_TEXTURE(PerFrame, Tex2D(float2), gMotionVectors)
        // ...and the REACTIVE MASK beside it (F12 mode 18). Same append-only rule, same reason, and
        // it rides the same set as the vectors because the two are always wanted together: the mask
        // says which of those vectors to disbelieve, and a second binding site for one of them is a
        // second place to get the pair half-right. Unbound reads 0 = "trust every vector", which is
        // the pre-mask behaviour and the correct benign default.
        DECL_TEXTURE(PerFrame, Tex2D(float),  gReactiveMask)
        // ─── S2i: THE GAP SKY — the clear arm of the deck mixture, for the DOME only ─────────────
        // gAtmosSkyView above is the cover-MIXED field and stays exactly what it was: the SH light
        // pass integrates it, and every aerial-perspective consumer reads it. This is its clear arm
        // — what the sky looks like BETWEEN the clouds — and skyhw.frag blends the two by
        // gAtmosParams.deckMix.z so a BROKEN deck stops veiling its own gaps.
        //
        // Appended AFTER gReactiveMask — append only, FSL assigns descriptor offsets from one
        // running counter and an insertion silently re-points every later binding.
        DECL_TEXTURE(PerFrame, Tex2D(float4), gAtmosSkyViewClear)
        // ─── TERRAIN PBR: the _paramh material on the ground ───────────────────────────────────
        // The SAME bucketed-Texture2DArray residency as gTerrainArrays above, in one more
        // declaration — and one more VTEX pack, which is what makes this cost one load per tap
        // instead of a per-pixel id->slot indirection. gTerrainParamTex has the identical shape to
        // gTerrainTex (TERRAIN_TSTRIDE uints per cell, one per texture square), built by the SAME
        // remap, so terrain.frag reaches both through one square INDEX.
        //
        // (gTerrainDerivTex / gTerrainDerivArrays — the baked `_paramd` derivative — were REMOVED with
        // that arm, 2026-09-18. Removing a declaration re-points every later binding exactly as an
        // insertion would, which is safe ONLY because every shader including this header and the
        // host are rebuilt together from it: fsl.py --compile over the whole shaders.list, then the
        // whole bin/DIRECT3D12 deployed. A partial deploy after this change is a mis-bound frame.)
        //
        // ⚠ SIZED BY COVERAGE, NOT BY LTEX COUNT. Only ~96 of 551 land textures ship a _paramh, so
        // the buckets are planned over the textures that HAVE a companion file — 551 slices of
        // mostly-default would be ~450 MB of nothing. A land texture with no companion gets slot 0
        // and is shaded exactly as it was before this existed.
        //
        // Slot 0 = "this land texture has no map of this kind", so these arrays are never sampled at
        // bucket 0 and an UNBOUND set reads 0 = no PBR on the ground = the pre-feature image. The
        // benign default points the right way with no fallback binding, like gGrassCrush.
        //
        // Appended AFTER gAtmosSkyViewClear — append only, FSL assigns descriptor offsets from one
        // running counter and an insertion silently re-points every later binding. (The note on
        // gTerrainArrays calling itself "declared LAST in the set" is stale: a dozen resources have
        // been appended after it since. The rule is append at the END, which is here.)
        DECL_BUFFER(PerFrame, Buffer(uint), gTerrainParamTex)
        DECL_ARRAY_TEXTURES(PerFrame, Tex2DArray(float4), gTerrainParamArrays, MAX_TERRAIN_BUCKETS)
        // ─── DISTANT-STATICS PBR: the `_paramh` material past the handover ─────────────────────────
        // The near mesh path shades a full material and the distant statics pass shaded albedo only,
        // so a big object crossing the handover DROPPED its roughness and relief in one frame. These
        // two are the other half of it; statics.frag reads them.
        //
        // ⚠ PerFrame AND NOT Persistent, although gStaticsArrays — the albedo these belong to — is
        // Persistent. That table was then EXACTLY at its 1024 cap (gTextures 880 + gStaticsArrays 128
        // + gFlipArrays 16; lifted 2026-09-30) with no room at all, so the companion set lives here, as
        // gTerrainParamArrays does for the same reason. It costs the statics pass nothing: it already
        // binds a PerFrame set for gFrameData/gShadowParams/gFroxelMask.
        //
        // gStaticsParamSlot is the albedo slot -> param slot map, and it is a BUFFER rather than a
        // second lane in the instance stream because the statics wire has no spare lane: InstParams
        // is full (texSlot, flags, glow stagger, near-cut plane) and widening the stride would touch
        // cullscatter.comp, the CPU cull, the probe and the vertex layout to carry 4 bytes per
        // INSTANCE for something that is a property of the SUBSET. Its layout is a 128-entry header
        // (one base per albedo bucket) followed by the per-(bucket, layer) runs:
        //     base  = gStaticsParamSlot[bucket]
        //     pslot = gStaticsParamSlot[base + layer]      // 0 = this texture ships no _paramh
        // Both loads are scalar — TexIndex is FLAT and one EI subset is one texture — so the whole
        // lookup is uniform per draw, and so is the branch it gates.
        //
        // Slot 0 = "no material", so an UNBOUND buffer reads 0 everywhere = no PBR on distant statics
        // = the pre-feature image. The benign default points the right way with no fallback binding,
        // like gGrassCrush and the terrain pair above.
        //
        // Appended AFTER gTerrainParamArrays — append only, FSL assigns descriptor offsets from one
        // running counter and an insertion silently re-points every later binding.
        DECL_BUFFER(PerFrame, Buffer(uint), gStaticsParamSlot)
        DECL_ARRAY_TEXTURES(PerFrame, Tex2DArray(float4), gStaticsParamArrays, MAX_STATICS_PARAM_BUCKETS)
        // Fixture GOBOS (tasks/forge-light-gobo.md G2): one octahedral 128^2 R8 layer per unique LIGH
        // model, the fixture's self-shadow as seen from its light, stored as OCCLUSION so an unbound
        // slot reads "open" (gobosample.h.fsl). Read only by the BAKED light loops (statics.frag,
        // terrain.frag's distant arm, pointlights.h.fsl's distant arm), which draw under pPerFrameSet
        // and pPerFrameSetReflectGeo — goboBake binds exactly those two. Appended last (one counter).
        DECL_TEXTURE(PerFrame, Tex2DArray(float), gGoboArray)
        // ─── SKY-VISIBILITY MAPS (skyamb.h.fsl skyVisMaps; gShadowParams.skyVis) ───────────────────
        // skyvis.comp's half-res (vis, distance, coverage, 0) over the MAIN camera's prepass depth.
        // ⚠ Bound ONLY into the main pPerFrameSet: it describes that camera's pixels and no other's.
        // Unbound reads 0, and distance 0 matches no fragment, so every other set takes the march.
        // Appended AFTER gGoboArray — append only, FSL assigns descriptor offsets from one running
        // counter and an insertion silently re-points every later binding.
        DECL_TEXTURE(PerFrame, Tex2D(float4), gSkyVisScreen)
        // pDepth ITSELF, for the water surface (knob waterDepthDirect, water_dd1/water_dd4.frag): sample
        // 0 read in place while the draw depth-tests through a READ-ONLY DSV, instead of the seam's
        // R32F copy (gSceneLinDepth). The type forks on its OWN macro, exactly like gPortalGate, so
        // nothing else on this root signature can flip it; both branches are one SRV slot. Bound ONLY
        // into the main pPerFrameSet and read only by the direct water variants; unbound elsewhere.
        // Appended AFTER gSkyVisScreen — append only (one running descriptor counter).
#ifndef WATER_DEPTH_SAMPLES
#define WATER_DEPTH_SAMPLES 1
#endif
#if WATER_DEPTH_SAMPLES > 1
        DECL_TEXTURE(PerFrame, Depth2DMS(float, WATER_DEPTH_SAMPLES), gSceneDepthMS)
#else
        DECL_TEXTURE(PerFrame, Depth2D(float), gSceneDepthMS)
#endif
        // GI field (gifield.frag.fsl, tasks/forge-gi.md v2): low-frequency RGB surface radiance over
        // gSkyHeight's window (same mapping, gShadowParams.skyAOMap); a = settled-ness (0 = no data).
        // Read by skyamb.h.fsl in place of the constant occluded-share floor when skyAOFloor.z > 0.
        // Unbound reads (0,0,0,0) -> a = 0 -> the receiver keeps the floor. Appended last.
        DECL_TEXTURE(PerFrame, Tex2D(float4), gGiField)
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
        // The physical sky's PER-VIEW cbuffer. It rides PerBatch, and that is not an arbitrary
        // parking spot: PerBatch is the set the sky pass ALREADY switches between main and mirror
        // (pPerBatchSetSky / pPerBatchSetReflectSky), so "one resource, two views" is the frequency
        // this belongs at and no new bind point appears in any draw loop.
        //
        // Declared AFTER gBatch so gBatch keeps offset 0 and every existing PerBatch bind is
        // untouched — FSL assigns offsets from ONE running per-set counter, and inserting ahead of
        // an existing resource silently re-points it. ⚠ Every OTHER PerBatch instance (the opaque
        // batch windows, the shadow caster pool, water) leaves this slot at its null descriptor,
        // which is correct and harmless: only skyhw.frag reads it, and it is bound in exactly the
        // two instances that draw it.
        DECL_CBUFFER(PerBatch, CBUFFER(SkyViewData), gSkyView)
    END_SRT_SET(PerBatch)
END_SRT(SrtData)
