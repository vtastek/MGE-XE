// mgeHost64 — M1c opaque scene SRT (cbuffer-array, scalable to 1023 draws/frame).
//
// One PerFrame descriptor set, one cbuffer (gScene), bound once — no per-draw descriptor
// churn: gScene = the camera view*proj + an array of up to 1023 visible part world
// matrices (a 64KB cbuffer: 1 + 1023 = 1024 float4x4 = the D3D12 cbuffer max). Each draw
// selects its matrix with a per-INSTANCE vertex attribute (DrawIndex), fed by an identity
// instance-index buffer + cmdDrawIndexedInstanced(firstInstance=i).
// Matrix convention (PROVEN): the host uploads D3DXMATRIX bytes (row-major) straight in;
// HLSL reads cbuffer matrices column-major (= transpose), so mul(M, v) reproduces D3DX's
// v*M with NO CPU transpose.
#pragma once

#ifndef OPAQUE_MAX_DRAWS
#define OPAQUE_MAX_DRAWS 1023
#endif

STRUCT(SceneData)
{
    DATA(float4x4, viewProj, None);
    DATA(float4x4, worlds[OPAQUE_MAX_DRAWS], None);
};

BEGIN_SRT_NO_AB(SrtData)
    BEGIN_SRT_SET(PerFrame)
        DECL_CBUFFER(PerFrame, CBUFFER(SceneData), gScene)
    END_SRT_SET(PerFrame)
END_SRT(SrtData)
