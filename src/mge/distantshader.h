#pragma once

enum RenderPassID {
    PASS_SETUP = 0,
    PASS_RENDERGRASSINST,
    PASS_RENDERSHADOW,
    PASS_RENDERSHADOWFFE,
    PASS_RENDERLAND,
    PASS_RENDERLANDREFL,
    PASS_RENDERSTATICSEXTERIOR,
    PASS_RENDERSTATICSINTERIOR,
    PASS_RENDERSKY,
    PASS_RENDERCLOUDS,
    PASS_RENDERWATER,
    PASS_RENDERUNDERWATER,
    PASS_RENDERCAUSTICS,
    PASS_BLENDMGE,
    PASS_DEBUGSHADOW,
    PASS_PLAYERWAVE,
    PASS_WAVESTEP,
    PASS_WORKAROUND,
    PASS_RENDERCACHETERRAIN,  // appended pass P13; cache near terrain in reflections
    PASS_RENDERSHADOWFFE_SKINNED,  // appended pass P14; cache-skinned shadow receiver (reflections)
    PASS_RENDERCACHETERRAINLIT,   // appended pass P15; cache near terrain with point lights (main view)
    PASS_RENDERCACHETERRAINREFLLIT,  // appended pass P16; cache near terrain with point lights + below-water clip (reflections)
    PASS_FOAM_ADVECT,   // appended pass P17; foam sim Voronoi particle advection (WATER_FOAM only)
    PASS_FOAM_FIELD,    // appended pass P18; foam sim velocity/density field smoothing
    PASS_FOAM_EXTRACT,  // appended pass P19; foam sim vorticity → texFoam extraction
    PASS_FOAM_UV        // appended pass P20; advect the foam-detail UV offset field through velocity
};

enum RenderShadowMapID {
    PASS_CLEARSHADOWMAP = 0,
    PASS_SHADOWSTENCIL,
    PASS_RENDERSHADOWMAP,
    PASS_SOFTENSHADOWMAP,
    PASS_RENDERSHADOWMAP_MW,
    PASS_RENDERSHADOWMAP_MW_SKINNED   // appended pass P4s; VS palette skinning
};

// S5a: enum DepthPassID (PASS_CLEARDEPTH / PASS_RENDERMWDEPTH / PASS_RENDERLANDDEPTH /
// PASS_RENDERSTATICSDEPTH / PASS_RENDERGRASSDEPTHINST / PASS_RENDERMWDEPTH_SKINNED) indexed
// the passes of "XE Depth.fx". Both the effect and every pass that used it are gone.

static const int SIZEOFSTATICVERT = 20;
static const int SIZEOFLANDVERT = 16;
