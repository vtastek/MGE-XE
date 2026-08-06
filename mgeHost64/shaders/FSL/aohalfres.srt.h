// mgeHost64 — half-res AO chain: the constants BOTH ends share.
//
// The chain (the "AO half-res" panel toggle; default OFF, so the full-res path is byte-identical
// until it is ticked):
//
//   linearize -> pLinearDepth (full)
//   aodepthdown  : pLinearDepth        -> pLinearDepthHalf      (aodepthdown.srt.h)
//   AO           : pLinearDepthHalf    -> pAOHalf               (unchanged shader, half-res set)
//   aoblur       : pAOHalf             -> pAOBlurHalf           (unchanged shader, half-res set)
//   aoupscale    : pAOBlurHalf + both depths -> pAOBlur (FULL)  (aoupscale.srt.h)
//
// pAOBlur stays full res and stays exactly what the colour frags sample as gAO, so the frag-side
// contract does not change at all — no edit to opaque/alpha/multimap/terrain .frag. pLinearDepth
// also stays full res: shadowmask.comp and water.frag's gSceneLinDepth both read it. Only the AO
// chain goes half.
//
// A SEPARATE cbuffer from the AO pass's, deliberately. The upscale needs full AND half dimensions,
// and gtao.srt.h:17 / aoblur.srt.h:21 both warn that ONE host buffer backs gAOParams and
// gBlurParams — growing one struct without the other corrupts the blur's knobs SILENTLY rather
// than failing to compile. Its own buffer sidesteps that trap entirely. (The AO pass's shared
// screenParams lane carries HALF dims while the toggle is on: the AO pass and the blur both run at
// half, so one value serves both.)
//
// Its own FILE, and one SRT per file, for a second reason: fsl.py emits every SRT in an included
// header into the shader, and two SRTs at the same update frequency alias each other's registers
// (both start at b0/t1/... in that frequency's space). Two shaders that each declared both survived
// only because DXC dead-strips the unreferenced one — an accident, not a contract.
#pragma once

STRUCT(AOUpParams)
{
    DATA(float4x4, invViewProj, None);   //  0..15 depth -> world (both passes' range weight)
    DATA(float4,   fullDims,    None);   // 16..19 xy = full w,h ; zw = 1/w, 1/h
    DATA(float4,   halfDims,    None);   // 20..23 xy = half w,h ; zw = 1/w, 1/h
    DATA(float4,   upParams,    None);   // 24..27 x = range sigma (world u), y = plane-distance sine
                                         //        sigma for the BEND channel (<= 0 disables); zw spare
};
