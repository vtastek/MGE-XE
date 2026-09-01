// mgeHost64 — the ATMOSPHERE LUT chain's SRT (tasks/forge-atmosphere.md S2a).
//
// ONE SRT for all four atmosphere compute passes, on the shared ComputeRootSignature. They differ
// only in which slots they actually touch, which is exactly the sunocc/skyheight arrangement and is
// what keeps this to one header — a second SRT in the same file would alias this one's registers and
// compile clean anyway, because DXC dead-strips the unused one. [[project_forge_srt_one_per_header]]
//
//   atmos_transmittance.comp   writes gAtmosOutA (256x64)   reads nothing
//   atmos_multiscatter.comp    writes gAtmosOutA (32x32)    reads gAtmosTransmittance
//   atmos_skyview.comp         writes gAtmosOutA (192x108)  reads both
//   atmos_sh.comp              writes gAtmosShOut (buffer)  reads all three
//
// ⚠ THE OUTPUT IS ONE SLOT, NOT THREE, AND THAT IS DELIBERATE. Three UAV slots would mean every
// pass declaring two it does not write, each of which is a descriptor that has to be bound to
// something type-valid or the write vanishes silently — the failure this file's neighbours all carry
// a warning about. One slot bound per-instance to whichever LUT that dispatch owns makes the
// binding impossible to get half-right: it is either the LUT being written or nothing.
//
// FOUR SET INSTANCES over the one PerBatch layout, one per pass. That is the aoblur/sunblur
// ping-pong idiom and it rebinds correctly because the host's bind cache compares the SET's GPU
// handle rather than the root index.
#pragma once

#include "atmosparams.h.fsl"   // AtmosphereParams. The STRUCT only — the medium (atmosphere.h.fsl)
                              // is HLSL bodies and this header is compiled as C++ by the host.

BEGIN_SRT(AtmosphereSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER  (PerBatch, CBUFFER(AtmosphereParams), gAtmosParams)
        // The two LUTs a pass may READ. Every instance binds BOTH, type-valid, even where the pass
        // ignores them (the transmittance pass reads neither): an unwritten SRV descriptor is
        // undefined heap memory rather than a null read, and the one thing worse than a wrong LUT is
        // a LUT-shaped view of somebody else's texture.
        DECL_TEXTURE  (PerBatch, Tex2D(float4),             gAtmosTransmittance)
        DECL_TEXTURE  (PerBatch, Tex2D(float4),             gAtmosMultiScatter)
        // ...and the ONE output this dispatch owns.
        DECL_RWTEXTURE(PerBatch, WTex2D(float4),            gAtmosOutA)
        // The measurement's landing zone (atmos_sh.comp only). RW so the SH pass can write it and
        // every other pass leaves it alone; bound to the same buffer in all four instances because a
        // UAV slot has to point somewhere legal even when nothing stores through it.
        DECL_RWBUFFER (PerBatch, RWBuffer(uint),            gAtmosShOut)
        // The sky-view LUT, read by the SH pass (which cannot reach it through gAtmosOutA — that is
        // its own output). Bound type-valid everywhere else, same rule as the two above.
        DECL_TEXTURE  (PerBatch, Tex2D(float4),             gAtmosSkyView)
    END_SRT_SET(PerBatch)
END_SRT(AtmosphereSrtData)
