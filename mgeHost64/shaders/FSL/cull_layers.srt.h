// mgeHost64 — statics LAYER draws (cull_layers.comp): its own small set, so the shared CullSrtData
// (whose every slot all six cull lanes must bind) does not grow for a camera-lane-only pass.
#pragma once

#include "cullparams.h.fsl"   // StaticsSubset

BEGIN_SRT(CullLayersSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_BUFFER  (PerBatch, Buffer(StaticsSubset), gStaticsSubsets)
        DECL_BUFFER  (PerBatch, Buffer(uint),          gLayerList)     // [0] = entry count, then sids
        DECL_RWBUFFER(PerBatch, RWBuffer(uint),        gSubsetCount)   // the camera lane's, after COUNT
        DECL_RWBUFFER(PerBatch, RWBuffer(uint),        gSubsetOffset)  // ...and its prefix, after SCAN
        DECL_RWBUFFER(PerBatch, RWBuffer(uint),        gLayerArgs)     // IndirectDrawIndexArguments[entry]
    END_SRT_SET(PerBatch)
END_SRT(CullLayersSrtData)
