// mgeHost64 — SUN-OCCLUSION height map: the long-range sun shadow (tasks/lighting.md).
//
// The sun cascades reach one MW cell (g_sunShadowRange = 8192) and return "fully lit" for every
// point outside them, so the entire distant world was unshadowed by construction: a ridge three
// cells away shadowed nothing, and the volumetric march — which reads the same cascades — lit up
// air the ridge should have blacked out. That is not something a bigger cascade fixes; a cascade
// that covered the fog wall would have texels tens of units wide.
//
// So this map answers the same question a completely different way. For each texel it stores the
// SUN-BLOCKED HEIGHT: the world Z below which a point at that XY is in shadow.
//
//     blockZ(p) = max over d > 0 of ( H(p + d * sunXY) - d * tan(sunElevation) )
//
// A point (x, y, z) is lit iff z >= blockZ(x, y). That is ONE texture sample, correct at ANY
// altitude — which is the whole reason for storing a HEIGHT rather than a horizon ANGLE. An angle is
// only meaningful for points standing on the ground; the volumetric march queries points in the air,
// and it needs an answer per sample at the per-sample budget the fog was designed around (one
// texture instruction, volfog.frag.fsl).
//
// The source is gSkyHeight, SH2's top-down max(terrain, statics) map — already resident, already
// rebuilt on its own snap trigger, and already sharing this map's window, so the two need exactly
// one published origin between them. Nothing new is traversed and no geometry is drawn: this pass
// is a pure function of a texture that exists.
//
// It is a DERIVATIVE, so it must be rebuilt whenever either of its two inputs moves — the height map
// (snap crossing) or the sun. The host owns both triggers; this shader has no state.
#pragma once

// "No occluder anywhere along the ray." Same sentinel and the same reasoning as SKY_HEIGHT_NONE:
// emphatically not 0, because 0 is Morrowind's sea level, and a zero-cleared map would put the
// shadow line at the waterline over every stretch of open sea. Large and negative loses every max().
#define SUN_OCC_NONE (-30000.0f)

STRUCT(SunOccParams)
{
    // xy = ABSOLUTE world XY of the OUTPUT map's (0,0) texel CORNER — the same corner gSkyHeight
    //      uses, because the two share a window and therefore a published origin,
    // z  = world units per texel of the OUTPUT map, w = output resolution in texels (square).
    DATA(float4, mapOrigin, None);
    // xy = unit horizontal direction TOWARD the sun, z = tan(sun elevation), w = march step count.
    DATA(float4, sunMarch,  None);
    // x = inner march radius (world u — where the first tap lands), y = outer radius. The host
    // SHORTENS y as the sun climbs: past a certain distance H - d*tan can no longer beat what the
    // march already holds, so at noon the far taps are provably dead weight. zw spare.
    DATA(float4, marchDist, None);
    // x = SOURCE (gSkyHeight) resolution in texels, y = its world units per texel. Its origin is
    // ours — see mapOrigin. zw spare.
    DATA(float4, srcMap,    None);
};

BEGIN_SRT(SunOccSrtData)
    // ONE set holding CBV + SRV + UAV, the shape skyheight.srt.h settled on and for the same reason
    // (gtao.srt.h's split-set UAV write vanished silently on this host). PerBatch is the frequency
    // every data-driven compute pass here uses; they are neighbours in one command list and rebind
    // correctly because the host's cache compares the SET's gpu handle, not just the root index.
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER  (PerBatch, CBUFFER(SunOccParams), gSunOccParams)
        DECL_TEXTURE  (PerBatch, Tex2D(float),          gSunOccHeightIn)   // = gSkyHeight (SH2)
        DECL_RWTEXTURE(PerBatch, WTex2D(float),         gSunOccOut)        // blockZ per texel
    END_SRT_SET(PerBatch)
END_SRT(SunOccSrtData)
