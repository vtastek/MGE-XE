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
    PASS_RENDERCACHETERRAINREFLLIT  // appended pass P16; cache near terrain with point lights + below-water clip (reflections)
};

enum RenderShadowMapID {
    PASS_CLEARSHADOWMAP = 0,
    PASS_SHADOWSTENCIL,
    PASS_RENDERSHADOWMAP,
    PASS_SOFTENSHADOWMAP,
    PASS_RENDERSHADOWMAP_MW,
    PASS_RENDERSHADOWMAP_MW_SKINNED   // appended pass P4s; VS palette skinning
};

enum RenderDepthID {
    PASS_CLEARDEPTH = 0,
    PASS_RENDERMWDEPTH,
    PASS_RENDERLANDDEPTH,
    PASS_RENDERSTATICSDEPTH,
    PASS_RENDERGRASSDEPTHINST,
    PASS_RENDERMWDEPTH_SKINNED         // appended pass D1s; VS palette skinning
};

static const int SIZEOFSTATICVERT = 20;
static const int SIZEOFLANDVERT = 16;
