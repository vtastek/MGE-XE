// mgeHost64 — planar-reflection mip-pyramid build compute SRT (WT4d / PBR water, P2).
//
// The water surface has a ROUGHNESS now (water.frag's slope-variance alpha), and a BRDF cone of
// half-angle ~alpha at the surface subtends ~alpha in the mirrored image, so the reflection has to
// be pre-filtered and sampled at a matching LOD. pReflectColor cannot carry those mips itself: its
// format is tied to g_live.sceneColorFormat and its PSO is SHARED with the main colour pass
// (pSkyPipeline spans both), so adding mips + a UAV to it puts the reflect pass's RTV/clear
// semantics at risk for no gain. A separate pReflectMips texture mirrors the Hi-Z shape instead,
// which is this host's working precedent for a compute-built pyramid.
//
// Two pipelines share this ONE set (hizreduce.srt.h's exact pattern):
//   reflectmipfirst.comp — copy pReflectColor (SRV) -> pReflectMips mip 0. A dispatch rather than a
//                          CopyResource because the two resources have DIFFERENT mip counts, which
//                          CopyResource forbids; it also format-converts for free (see below).
//   reflectmip.comp      — 2x2 BOX-AVERAGE mip i-1 -> mip i. Not min, not max: this is radiance.
//
// ⚠⚠ pReflectColor IS PREMULTIPLIED (rgb already scaled by coverage, a = coverage; water.frag
// resolves it as reflSample.rgb + fogCol*(1-a)). The reduce therefore averages ALL FOUR CHANNELS
// WITH IDENTICAL WEIGHTS and nothing cleverer. An alpha-weighted downsample would rebuild the
// premultiplication seam bug one mip level down, where it reads as a DARK FRINGE along the
// reflected horizon — the horizon fog band is the only place coverage is neither 0 nor 1, and so
// the only witness. Third instance of this bug class ([[project_forge_premultiplied_pair]]).
//
// pReflectMips is R16G16B16A16_SFLOAT REGARDLESS of sceneColorFormat, and that is deliberate on two
// counts. (1) B8G8R8A8_UNORM is NOT in D3D12's TypedUAVLoadAdditionalFormats list, so the reduce's
// UAV load of the source mip would not be legal against the LDR scene format; R16G16B16A16_FLOAT
// is. (2) It decouples the pyramid from the HDR flip entirely — step 6 of forge-postprocess.md
// moves sceneColorFormat and this file does not care. The first pass reads pReflectColor through an
// SRV, so the conversion is free.
//
// Update frequency = Persistent, same as hizreduce. Two different Persistent-frequency sets in one
// cmd is safe: cmdBindDescriptorSet's rebind cache is keyed on the GPU DESCRIPTOR HANDLE, not on
// the set object, so distinct sets at the same root index always rebind (Direct3D12.c:4685).
// Resource names are unique across all merged compute SRTs (house rule since the d3d.py aliasing
// bug), and there is exactly ONE SRT per header ([[project_forge_srt_one_per_header]]).
#pragma once

BEGIN_SRT(ReflectMipSrtData)
    BEGIN_SRT_SET(Persistent)
        DECL_TEXTURE  (Persistent, Tex2D(float4),  gReflMipSrcTex)  // pReflectColor (first pass only)
        DECL_RWTEXTURE(Persistent, RTex2D(float4), gReflMipSrc)     // pyramid mip i-1 (reduce only)
        DECL_RWTEXTURE(Persistent, WTex2D(float4), gReflMipDst)     // pyramid mip i (both passes)
    END_SRT_SET(Persistent)
END_SRT(ReflectMipSrtData)
