// mgeHost64 — M1c opaque scene SRT (shader resource table).
//
// Two update frequencies:
//   PerFrame  gFrameData : the camera view*proj (one cbuffer, bound once per frame).
//   PerDraw   gObject    : the part's model->world transform (one instance per visible
//                          part; the host binds instance i before each part's draw,
//                          06_MaterialPlayground pattern).
// Matrix convention: the host uploads D3DXMATRIX bytes (row-major) straight into these
// cbuffers; HLSL reads them column-major (= transpose), so mul(M, v) in the shaders
// reproduces D3DX's row-vector v*M with NO CPU transpose.
#pragma once

STRUCT(FrameData)
{
    DATA(float4x4, viewProj, None);
};

STRUCT(Object)
{
    DATA(float4x4, world, None);
};

BEGIN_SRT_NO_AB(SrtData)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(FrameData), gFrameData)
    END_SRT_SET(PerFrame)
    BEGIN_SRT_SET(PerDraw)
        DECL_CBUFFER(PerDraw, CBUFFER(Object), gObject)
    END_SRT_SET(PerDraw)
END_SRT(SrtData)
