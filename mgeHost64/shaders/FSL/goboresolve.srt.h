// mgeHost64 — fixture GOBO resolve SRT (tasks/forge-light-gobo.md G1). One SRT per header
// (project_forge_srt_one_per_header). Shares the merged ComputeRootSignature.
//
// The bake rasterises a BATCH of fixtures into one R8 atlas: row s = fixture (firstLayer + s), six
// faceRes^2 cube faces left to right (+X -X +Y -Y +Z -Z, gobo.h.fsl). This pass turns each row into
// that fixture's octahedral gobo layer.
#pragma once

STRUCT(GoboResolveParams)
{
    // x = first gobo layer of this batch, y = fixtures in the batch, z = face resolution (px),
    // w = octahedral resolution (px).
    DATA(uint4, dims, None);
};

BEGIN_SRT(GoboResolveSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER  (PerBatch, CBUFFER(GoboResolveParams), gGoboResolveParams)
        DECL_TEXTURE  (PerBatch, Tex2D(float), gGoboCube)
        DECL_RWTEXTURE(PerBatch, RWTex2DArray(float), gGoboOut)
    END_SRT_SET(PerBatch)
END_SRT(GoboResolveSrtData)
