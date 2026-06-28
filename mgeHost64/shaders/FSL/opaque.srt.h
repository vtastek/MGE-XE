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

#define OPAQUE_BATCH 1024   // matrices per 64KB cbuffer window; must match host kBatchSize
#define MAX_TEXTURES 896    // bindless gTextures[] array size; MUST match IPC::kMaxTextures (geomwire.h)
// Distant-statics texture residency: a descriptor-array of Texture2DArrays, one element per
// (format, capped-size) bucket. Each element is ONE descriptor (format/size are runtime resource
// props; HLSL sees only Texture2DArray<float4>), so this holds arrays of DIFFERENT formats+sizes
// indexed bindlessly — no switch, no per-texture descriptor. MAX_TEXTURES + MAX_STATICS_BUCKETS =
// 1024, the proven-OK Persistent-table size (see geomwire.h). statics texSlot = (bucket<<16)|layer.
#define MAX_STATICS_BUCKETS 128
#define MAX_POINT_LIGHTS 128 // per-frame point-light cap; MUST match IPC::kMaxPointLights (geomwire.h)

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
    // distantland.vert/.frag read these; opaque/skinned/multimap never touch them (harmless when 0).
    // lodParams: x = base-atlas bindless slot, y = normal-atlas slot, z = detail-atlas slot,
    //            w = nearViewRange (for the landBias z-sink). 208B.
    DATA(float4, lodParams,   None);
    // lodSunAmb: xyz = distant-land sun ambient (XE Mod Landscape.fx sunAmb). 224B < 256B CBV min.
    DATA(float4, lodSunAmb,   None);
    // Phase 1a/1b LIVE: the REAL camera eye (absolute world). The live frame is camera-relative
    // (near worlds pre-shifted by -eye, eyePos = 0), but resident DL geometry is in ABSOLUTE world
    // coords — distantland.vert subtracts lodEye to match; statics are host-shifted so statics.vert
    // never reads it. Near/opaque/skinned/multimap paths leave it 0 and never touch it. 240B < 256B.
    DATA(float4, lodEye,      None);
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
// 16 + 128*3*16 = 6160 B < the 64KB cbuffer limit.
STRUCT(LightData)
{
    DATA(float4, lightParams, None);                 // x = count
    DATA(float4, lights[MAX_POINT_LIGHTS * 3], None);
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
    END_SRT_SET(PerFrame)
    // Point-light cbuffer — rides the otherwise-unused PerDraw set (FSL has exactly four
    // fixed update frequencies: Persistent/PerFrame/PerBatch/PerDraw; a custom set name has
    // no register space). Its own set ⇒ b0 in spaceSET_PerDraw, no collision with PerFrame's
    // b0. Read by opaque.frag for both the static and skinned paths.
    BEGIN_SRT_SET(PerDraw)
        DECL_CBUFFER(PerDraw, CBUFFER(LightData), gLights)
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
    END_SRT_SET(Persistent)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER(PerBatch, CBUFFER(BatchData), gBatch)
    END_SRT_SET(PerBatch)
END_SRT(SrtData)
