//--------------------------------------
// Generated from Forge Shading Language
//--------------------------------------

#define DIRECT3D12
#define DIRECT3D12
#define STAGE_FRAG
/*
* Copyright (c) 2017-2025 The Forge Interactive Inc.
*
* This file is part of The-Forge
* (see https://github.com/ConfettiFX/The-Forge).
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

#ifndef _D3D_H
#define _D3D_H

#define UINT_MAX 4294967295
#define FLT_MAX  3.402823466e+38F

inline float2 f2(float x) { return float2(x, x); }
#if !defined(DIRECT3D12)
inline float2 f2(bool x) { return float2(x, x); }
inline float2 f2(int x) { return float2(x, x); }
inline float2 f2(uint x) { return float2(x, x); }
#endif

#define packed_float3 float3

#define f4(X) float4(X,X,X,X)
#define f3(X) float3(X,X,X)
// #define f2(X) float2(X,X)
#define u4(X)  uint4(X,X,X,X)
#define u3(X)  uint3(X,X,X)
#define u2(X)  uint2(X,X)
#define i4(X)   int4(X,X,X,X)
#define i3(X)   int3(X,X,X)
#define i2(X)   int2(X,X)

#define h4(X)  half4(X,X,X,X)

#define short4 int4
#define short3 int3
#define short2 int2
#define short  int

#define ushort4 uint4
#define ushort3 uint3
#define ushort2 uint2
#define ushort  uint

#if !defined(DIRECT3D12) 
#define min16float half
#define min16float2 half2
#define min16float3 half3
#define min16float4 half4
#endif


/* Matrix */

// float3x3 f3x3(float4x4 X) { return (float3x3)X; }

#define to_f3x3(M) ((float3x3)M)

// #define f2x3 float3x2
// #define f3x2 float2x3

#define f2x2 float2x2
#define f2x3 float3x2
#define f2x4 float4x2
#define f3x2 float2x3
#define f3x3 float3x3
#define f3x4 float4x3
#define f4x2 float2x4
#define f4x3 float3x4
#define f4x4 float4x4

#define make_f2x2_cols(C0, C1) transpose(f2x2(C0, C1))
#define make_f2x2_rows(R0, R1) f2x2(R0, R1)
#define make_f2x2_col_elems(E00, E01, E10, E11) f2x2(E00, E10, E01, E11)
#define make_f2x3_cols(C0, C1) transpose(f3x2(C0, C1))
#define make_f2x3_rows(R0, R1, R2) f2x3(R0, R1, R2)
#define make_f2x3_col_elems(E00, E01, E10, E11, E20, E21) f2x3(E00, E10, E20, E01, E11, E21)

#define make_f3x3_row_elems  f3x3

inline f3x2 make_f3x2_cols(float2 c0, float2 c1, float2 c2)
{ return transpose(f2x3(c0, c1, c2)); }
// TODO: add all the others

#define make_f3x3_cols(C0, C1, C2) transpose(float3x3(C0, C1, C2))
inline f3x3 make_f3x3_rows(float3 r0, float3 r1, float3 r2)
{ return f3x3(r0, r1, r2); }

#define make_f4x4_col_elems(E00, E01, E02, E03, E10, E11, E12, E13, E20, E21, E22, E23, E30, E31, E32, E33) \
    f4x4(E00, E10, E20, E30, E01, E11, E21, E31, E02, E12, E22, E32, E03, E13, E23, E33)
#define make_f4x4_row_elems f4x4
#define make_f4x4_cols(C0, C1, C2, C3) transpose(f4x4(C0, C1, C2, C3))

inline f4x4 Identity()
{
    return make_f4x4_row_elems(
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
}

inline void setElem(inout f4x4 M, int i, int j, float val) { M[j][i] = val; }
inline float getElem(f4x4 M, int i, int j) { return M[j][i]; }

inline float4 getCol(in f4x4 M, const uint i) { return float4(M[0][i], M[1][i], M[2][i], M[3][i]); }
inline float3 getCol(in f4x3 M, const uint i) { return float3(M[0][i], M[1][i], M[2][i]); }
inline float2 getCol(in f4x2 M, const uint i) { return float2(M[0][i], M[1][i]); }

inline float4 getCol(in f3x4 M, const uint i) { return float4(M[0][i], M[1][i], M[2][i], M[3][i]); }
inline float3 getCol(in f3x3 M, const uint i) { return float3(M[0][i], M[1][i], M[2][i]); }
inline float2 getCol(in f3x2 M, const uint i) { return float2(M[0][i], M[1][i]); }

inline float4 getCol(in f2x4 M, const uint i) { return float4(M[0][i], M[1][i], M[2][i], M[3][i]); }
inline float3 getCol(in f2x3 M, const uint i) { return float3(M[0][i], M[1][i], M[2][i]); }
inline float2 getCol(in f2x2 M, const uint i) { return float2(M[0][i], M[1][i]); }

#define getCol0(M) getCol(M, 0)
#define getCol1(M) getCol(M, 1)
#define getCol2(M) getCol(M, 2)
#define getCol3(M) getCol(M, 3)

#define getRow(M, I) (M)[I]
#define getRow0(M) getRow(M, 0)
#define getRow1(M) getRow(M, 1)
#define getRow2(M) getRow(M, 2)
#define getRow3(M) getRow(M, 3)


inline f4x4 setCol(inout f4x4 M, in float4 col, const uint i) { M[0][i] = col[0]; M[1][i] = col[1]; M[2][i] = col[2]; M[3][i] = col[3]; return M; }
inline f4x3 setCol(inout f4x3 M, in float3 col, const uint i) { M[0][i] = col[0]; M[1][i] = col[1]; M[2][i] = col[2];; return M; }
inline f4x2 setCol(inout f4x2 M, in float2 col, const uint i) { M[0][i] = col[0]; M[1][i] = col[1]; return M; }

inline f3x4 setCol(inout f3x4 M, in float4 col, const uint i) { M[0][i] = col[0]; M[1][i] = col[1]; M[2][i] = col[2]; M[3][i] = col[3]; return M; }
inline f3x3 setCol(inout f3x3 M, in float3 col, const uint i) { M[0][i] = col[0]; M[1][i] = col[1]; M[2][i] = col[2];; return M; }
inline f3x2 setCol(inout f3x2 M, in float2 col, const uint i) { M[0][i] = col[0]; M[1][i] = col[1]; return M; }

inline f2x4 setCol(inout f2x4 M, in float4 col, const uint i) { M[0][i] = col[0]; M[1][i] = col[1]; M[2][i] = col[2]; M[3][i] = col[3]; return M; }
inline f2x3 setCol(inout f2x3 M, in float3 col, const uint i) { M[0][i] = col[0]; M[1][i] = col[1]; M[2][i] = col[2];; return M; }
inline f2x2 setCol(inout f2x2 M, in float2 col, const uint i) { M[0][i] = col[0]; M[1][i] = col[1]; return M; }

#define setCol0(M, C) setCol(M, C, 0)
#define setCol1(M, C) setCol(M, C, 1)
#define setCol2(M, C) setCol(M, C, 2)
#define setCol3(M, C) setCol(M, C, 3)

f4x4 setRow(inout f4x4 M, in float4 row, const uint i) { M[i] = row; return M; }
f4x3 setRow(inout f4x3 M, in float4 row, const uint i) { M[i] = row; return M; }
f4x2 setRow(inout f4x2 M, in float4 row, const uint i) { M[i] = row; return M; }

f3x4 setRow(inout f3x4 M, in float3 row, const uint i) { M[i] = row; return M; }
f3x3 setRow(inout f3x3 M, in float3 row, const uint i) { M[i] = row; return M; }
f3x2 setRow(inout f3x2 M, in float3 row, const uint i) { M[i] = row; return M; }

f2x4 setRow(inout f2x4 M, in float2 row, const uint i) { M[i] = row; return M; }
f2x3 setRow(inout f2x3 M, in float2 row, const uint i) { M[i] = row; return M; }
f2x2 setRow(inout f2x2 M, in float2 row, const uint i) { M[i] = row; return M; }

#define setRow0(M, R) setRow(M, R, 0)
#define setRow1(M, R) setRow(M, R, 1)
#define setRow2(M, R) setRow(M, R, 2)
#define setRow3(M, R) setRow(M, R, 3)


// mapping of glsl format qualifiers
#define rgba8 float4

#define VS_MAIN main
#define PS_MAIN main
#define CS_MAIN main
#define TC_MAIN main
#define TE_MAIN main

#ifdef DIRECT3D12
#define FSL_REG(REG_0, REG_1) REG_1
#else
#define FSL_REG(REG_0, REG_1) REG_0
#endif

#ifdef RETURN_TYPE
    #define INIT_MAIN RETURN_TYPE Out
    #define RETURN return Out
#else
    #define INIT_MAIN
    #define RETURN return
#endif

// #if defined(DIRECT3D12) 
//     #define FSL_VertexID(NAME)         uint  NAME : SV_VertexID
//     #define SV_InstanceID(NAME)        uint  NAME : SV_InstanceID
//     #define FSL_GroupID(NAME)          uint3 NAME : SV_GroupID
//     #define FSL_DispatchThreadID(NAME) uint3 NAME : SV_DispatchThreadID
//     #define FSL_GroupThreadID(NAME)    uint3 NAME : SV_GroupThreadID
//     #define FSL_GroupIndex(NAME)       uint  NAME : SV_GroupIndex
//     #define FSL_SampleIndex(NAME)      uint  NAME : SV_SampleIndex
//     #define FSL_PrimitiveID(NAME)      uint  NAME : SV_PrimitiveID
// #endif

#define SV_PointSize NONE

#define packed_float3 float3

#define out_coverage uint

// #define DDX ddx
// #define DDY ddy

// bool greaterThanEqual(float2 a, float b)
// {
//     return all(a >= float2(b, b));
// }
// bool greaterThan(float2 a, float b)
// {
//     return all(a > float2(b, b));
// }

#if defined( ORBIS )
bool2 And(const bool2 a, const bool2 b)
{ return a && b; }
#else
bool2 And(const bool2 a, const bool2 b)
{ return and(a, b); }
#endif

#define _GREATER_THAN(TYPE) \
// bool2 GreaterThan(const TYPE##2 a, const TYPE b) { return a > b; } \
// bool3 GreaterThan(const TYPE##3 a, const TYPE b) { return a > b; } \
// bool4 GreaterThan(const TYPE##4 a, const TYPE b) { return a > b; } \
// bool2 GreaterThan(const TYPE a, const TYPE##2 b) { return a > b; } \
// bool3 GreaterThan(const TYPE a, const TYPE##3 b) { return a > b; } \
// bool4 GreaterThan(const TYPE a, const TYPE##4 b) { return a > b; } 
// _GREATER_THAN(float)
// _GREATER_THAN(int)
// bool4 GreaterThan(const float4 a, const float4 b) { return a > b; }

#define GreaterThan(A, B)      ((A) > (B))
#define GreaterThanEqual(A, B) ((A) >= (B))
#define LessThan(A, B)         ((A) < (B))
#define LessThanEqual(A, B)    ((A) <= (B))

#define AllGreaterThan(X, Y)      all(GreaterThan(X, Y))
#define AllGreaterThanEqual(X, Y) all(GreaterThanEqual(X, Y))
#define AllLessThan(X, Y)         all(LessThan(X, Y))
#define AllLessThanEqual(X, Y)    all(LessThanEqual((X), (Y)))

#define AnyGreaterThan(X, Y)      any(GreaterThan(X, Y))
#define AnyGreaterThanEqual(X, Y) any(GreaterThanEqual(X, Y))
#define AnyLessThan(X, Y)         any(LessThan(X, Y))
#define AnyLessThanEqual(X, Y)    any(LessThanEqual((X), (Y)))

#define select lerp

#define fast_min min
#define fast_max max
#define isordered(X, Y) ( ((X)==(X)) && ((Y)==(Y)) )
#define isunordered(X, Y) (isnan(X) || isnan(Y))

uint extract_bits(uint src, uint off, uint bits) { uint mask = (1u << bits) - 1; return (src >> off) & mask; } // ABfe
// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/bfi---sm5---asm-
uint insert_bits(uint src, uint ins, uint off, uint bits)
{ 
    uint bitmask = (((1u << bits)-1) << off) & 0xffffffff;
    return ((ins << off) & bitmask) | (src & ~bitmask);
} // ABfiM

#define Equal(X, Y) ((X) == (Y))

#define row_major(X) X

#if defined(ORBIS)
#define NonUniformResourceIndex(X) (X)
#endif

// #if defined(DIRECT3D12)
//     #define GroupMemoryBarrier GroupMemoryBarrierWithGroupSync
//     #define AllMemoryBarrier AllMemoryBarrierWithGroupSync
// #else
#undef GroupMemoryBarrier
#define GroupMemoryBarrier GroupMemoryBarrierWithGroupSync
#undef AllMemoryBarrier
#define AllMemoryBarrier AllMemoryBarrierWithGroupSync
#undef MemoryBarrier
#define MemoryBarrier DeviceMemoryBarrier
// #endif


#define _DECL_FLOAT_TYPES(EXPR) \
EXPR(half) \
EXPR(half2) \
EXPR(half3) \
EXPR(half4) \
EXPR(float) \
EXPR(float2) \
EXPR(float3) \
EXPR(float4)

#ifndef _DECL_TYPES
#define _DECL_TYPES(EXPR) \
EXPR(int) \
EXPR(int2) \
EXPR(int3) \
EXPR(int4) \
EXPR(uint) \
EXPR(uint2) \
EXPR(uint3) \
EXPR(uint4) \
EXPR(half) \
EXPR(half2) \
EXPR(half3) \
EXPR(half4) \
EXPR(float) \
EXPR(float2) \
EXPR(float3) \
EXPR(float4)
#endif

#define _DECL_SCALAR_TYPES(EXPR) \
EXPR(int) \
EXPR(uint) \
EXPR(half) \
EXPR(float)

#if defined(DIRECT3D12) 
// #define _DECL_AtomicAdd(TYPE) \
// inline void AtomicAdd(inout TYPE dst, TYPE value, out TYPE original_val) \
// { InterlockedAdd(dst, value, original_val); }
// _DECL_AtomicAdd(int)
// _DECL_AtomicAdd(uint)
#define AtomicAdd(DEST, VALUE, ORIGINAL_VALUE) \
    InterlockedAdd(DEST, VALUE, ORIGINAL_VALUE)
	
#define AtomicOr(DEST, VALUE, ORIGINAL_VALUE) \
    InterlockedOr(DEST, VALUE, ORIGINAL_VALUE)

#define AtomicAnd(DEST, VALUE, ORIGINAL_VALUE) \
    InterlockedAnd(DEST, VALUE, ORIGINAL_VALUE)

#define AtomicXor(DEST, VALUE, ORIGINAL_VALUE) \
    InterlockedXor(DEST, VALUE, ORIGINAL_VALUE)
#endif


// #define AtomicStore(DEST, VALUE) \
//     (DEST) = (VALUE)

#define _DECL_AtomicStore(TYPE) \
inline void AtomicStore(inout TYPE dst, TYPE value) \
{ dst = value; }
_DECL_TYPES(_DECL_AtomicStore)

#define AtomicLoad(SRC) \
    SRC

#define AtomicExchange(DEST, VALUE, ORIGINAL_VALUE) \
    InterlockedExchange((DEST), (VALUE), (ORIGINAL_VALUE))
	
#define AtomicCompareExchange(DEST, COMPARE_VALUE, VALUE, ORIGINAL_VALUE) \
    InterlockedCompareExchange((DEST), (COMPARE_VALUE), (VALUE), (ORIGINAL_VALUE))

#if defined(DIRECT3D12) || defined(ORBIS) || defined(PROSPERO)
    #define inout(T) inout T
    #define out(T) out T
    #define in(T) in T
    #define inout_array(T, X) inout T X
    #define out_array(T, X)   out T X
    #define in_array(T, X)    in T X
    #undef groupshared
    #define groupshared(T) inout T
#else
    // fxc macro expansion workaround
    #define inout_float  inout float
    #define inout_float2 inout float2
    #define inout_float3 inout float3
    #define inout_float4 inout float4
    #define inout_uint   inout uint
    #define inout_uint2  inout uint2
    #define inout_uint3  inout uint3
    #define inout_uint4  inout uint4
    #define inout_int    inout int
    #define inout_int2   inout int2
    #define inout_int3   inout int3
    #define inout_int4   inout int4
    #define inout(T)     inout_ ## T

    #define out_float  out float
    #define out_float2 out float2
    #define out_float3 out float3
    #define out_float4 out float4
    #define out_uint   out uint
    #define out_uint2  out uint2
    #define out_uint3  out uint3
    #define out_uint4  out uint4
    #define out_int    out int
    #define out_int2   out int2
    #define out_int3   out int3
    #define out_int4   out int4
    #define out(T)     out_ ## T

    #define in_float  in float
    #define in_float2 in float2
    #define in_float3 in float3
    #define in_float4 in float4
    #define in_uint   in uint
    #define in_uint2  in uint2
    #define in_uint3  in uint3
    #define in_uint4  in uint4
    #define in_int    in int
    #define in_int2   in int2
    #define in_int3   in int3
    #define in_int4   in int4
    #define in(T)     in_ ## T
    
    #define groupshared(T)     inout_ ## T

#endif

#define NUM_THREADS(X, Y, Z) [numthreads(X, Y, Z)]

#define FLAT(X) nointerpolation X
#define CENTROID(X) centroid X

#define STRUCT(NAME) struct NAME

#define DATA(TYPE, NAME, SEM) TYPE NAME : SEM

#define ByteBuffer ByteAddressBuffer
#define RWByteBuffer RWByteAddressBuffer
#define WByteBuffer RWByteAddressBuffer

inline uint LoadByte(ByteBuffer buff, uint address)   { return buff.Load(address);  }
inline uint2 LoadByte2(ByteBuffer buff, uint address) { return buff.Load2(address); }
inline uint3 LoadByte3(ByteBuffer buff, uint address) { return buff.Load3(address); }
inline uint4 LoadByte4(ByteBuffer buff, uint address) { return buff.Load4(address); }

inline uint LoadByte(RWByteBuffer buff, uint address)   { return buff.Load(address);  }
inline uint2 LoadByte2(RWByteBuffer buff, uint address) { return buff.Load2(address); }
inline uint3 LoadByte3(RWByteBuffer buff, uint address) { return buff.Load3(address); }
inline uint4 LoadByte4(RWByteBuffer buff, uint address) { return buff.Load4(address); }

inline void StoreByte(RWByteBuffer buff, uint address, uint val)   { buff.Store(address, val);  }
inline void StoreByte2(RWByteBuffer buff, uint address, uint2 val) { buff.Store2(address, val); }
inline void StoreByte3(RWByteBuffer buff, uint address, uint3 val) { buff.Store3(address, val); }
inline void StoreByte4(RWByteBuffer buff, uint address, uint4 val) { buff.Store4(address, val); }

#define _DECL_SampleLvlTexCube(TYPE) \
inline TYPE SampleLvlTexCube(TextureCube<TYPE> tex, SamplerState smp, float3 p, float l) \
{ return tex.SampleLevel(smp, p, l); }
_DECL_FLOAT_TYPES(_DECL_SampleLvlTexCube)

#define _DECL_SampleLvlTexCubeArray(TYPE) \
inline TYPE SampleLvlTexCubeArray(TextureCubeArray<TYPE> tex, SamplerState smp, float3 p, float l) \
{ return tex.SampleLevel(smp, float4(p, l), 0); }
_DECL_FLOAT_TYPES(_DECL_SampleLvlTexCubeArray)
// _DECL_SampleLvlTexCube(float)
// _DECL_SampleLvlTexCube(float2)
// _DECL_SampleLvlTexCube(float3)
// _DECL_SampleLvlTexCube(float4)

// #define SampleLvlTexCube(NAME, SAMPLER, COORD, LEVEL) NAME.SampleLevel(SAMPLER, COORD, LEVEL)
// #define SampleLvlTex2D(NAME, SAMPLER, COORD, LEVEL) NAME.SampleLevel(SAMPLER, COORD, LEVEL)
#define _DECL_SampleLvlTex2D(TYPE) \
inline TYPE SampleLvlTex2D(Texture2D<TYPE> tex, SamplerState smp, float2 p, float l) \
{ return tex.SampleLevel(smp, p, l); }
_DECL_FLOAT_TYPES(_DECL_SampleLvlTex2D)

// #define SampleLvlTex3D(NAME, SAMPLER, COORD, LEVEL) NAME.SampleLevel(SAMPLER, COORD, LEVEL)
#define _DECL_SampleLvlTex3D(TYPE) \
inline TYPE SampleLvlTex3D(Texture3D<TYPE> tex, SamplerState smp, float3 p, float l) \
{ return tex.SampleLevel(smp, p, l); }
_DECL_FLOAT_TYPES(_DECL_SampleLvlTex3D)

#define _DECL_SampleLvlTex2DArray(TYPE) \
inline TYPE SampleLvlTex2DArray(Texture3D<TYPE> tex, SamplerState smp, float3 p, float l) \
{ return tex.SampleLevel(smp, p, l); }
_DECL_FLOAT_TYPES(_DECL_SampleLvlTex2DArray)

// #define SampleLvlOffsetTex2D(NAME, SAMPLER, COORD, LEVEL, OFFSET) NAME.SampleLevel(SAMPLER, COORD, LEVEL, OFFSET)
#define SampleLvlOffsetTex3D(NAME, SAMPLER, COORD, LEVEL, OFFSET) NAME.SampleLevel(SAMPLER, COORD, LEVEL, OFFSET)

#define _DECL_SampleLvlOffsetTex2D(TYPE) \
inline TYPE SampleLvlOffsetTex2D(Texture2D<TYPE> tex, SamplerState smp, float2 p, float l, int2 o) \
{ return tex.SampleLevel(smp, p, l, o); }
_DECL_FLOAT_TYPES(_DECL_SampleLvlOffsetTex2D)
// _DECL_SampleLvlOffsetTex2D(float)
// _DECL_SampleLvlOffsetTex2D(float2)
// _DECL_SampleLvlOffsetTex2D(float3)
// _DECL_SampleLvlOffsetTex2D(float4)

float4  _to4(in float4  x)  { return x; }
float4  _to4(in float3  x)  { return float4(x, 0); }
float4  _to4(in float2  x)  { return float4(x, 0, 0); }
float4  _to4(in float x)    { return float4(x, 0, 0, 0); }

// #ifdef ORBIS
// #define LoadTex2D(TEX, SMP, P) ((TEX)[P])
// #else
// inline TYPE LoadTex2D(RWTexture2D<TYPE> tex, SamplerState smp, int2 p) { return tex.Load(p); }
// #if 0 && (defined(DIRECT3D12) 
// #define _DECL_LoadTex2D(TYPE) \
// inline TYPE LoadTex2D(Texture2D<TYPE>   tex, SamplerState smp, int2 p) { return tex.Load(int3(p, 0)); } \
// inline TYPE LoadTex2D(RWTexture2D<TYPE> tex, SamplerState smp, int2 p) { return tex[p]; } \
// inline TYPE LoadTex2D(Texture2D<TYPE>   tex, int _, int2 p) { return tex.Load(int3(p, 0)); } \
// inline TYPE LoadTex2D(RWTexture2D<TYPE> tex, int _, int2 p) { return tex[p]; }
// #else
// #define _DECL_LoadTex2D(TYPE) \
// inline TYPE LoadTex2D(Texture2D<TYPE>   tex, SamplerState smp, int2 p) { return tex[p]; } \
// inline TYPE LoadTex2D(RWTexture2D<TYPE> tex, SamplerState smp, int2 p) { return tex[p]; } \
// inline TYPE LoadTex2D(Texture2D<TYPE>   tex, int _, int2 p) { return tex[p]; } \
// inline TYPE LoadTex2D(RWTexture2D<TYPE> tex, int _, int2 p) { return tex[p]; }
// #endif

#define _DECL_LoadTex1D(TYPE) \
inline TYPE LoadTex1D(Texture1D<TYPE>   tex, SamplerState smp, int p, int lod) { return tex.Load(int2(p, lod)); } \
inline TYPE LoadTex1D(Texture1D<TYPE>   tex, int _, int p, int lod) { return tex.Load(int2(p, lod)); } \
inline TYPE LoadRWTex1D(RWTexture1D<TYPE> tex, int p) { return tex[p]; }
_DECL_TYPES(_DECL_LoadTex1D)

#define _DECL_LoadTex2D(TYPE) \
inline TYPE LoadTex2D(Texture2D<TYPE>   tex, SamplerState smp, int2 p, int lod) { return tex.Load(int3(p, lod)); } \
inline TYPE LoadTex2D(Texture2D<TYPE>   tex, int _, int2 p, int lod) { return tex.Load(int3(p, lod)); } \
inline TYPE LoadRWTex2D(RWTexture2D<TYPE> tex, int2 p) { return tex[p]; }
_DECL_TYPES(_DECL_LoadTex2D)

#if defined(DIRECT3D12)
#define _DECL_LoadRasterizerOrderedTexture2D(TYPE) \
inline TYPE LoadRWTex2D(RasterizerOrderedTexture2D<TYPE> tex, int2 p) \
{ return tex[p]; }
_DECL_TYPES(_DECL_LoadRasterizerOrderedTexture2D)
#endif

#define _DECL_LoadTex3D(TYPE) \
inline TYPE LoadTex3D(Texture2DArray<TYPE> tex,     SamplerState smp, int3 p, int lod) { return tex.Load(int4(p, lod)); } \
inline TYPE LoadTex3D(Texture3D<TYPE> tex,          SamplerState smp, int3 p, int lod) { return tex.Load(int4(p, lod)); } \
inline TYPE LoadTex3D(Texture2DArray<TYPE> tex,     int _, int3 p, int lod) { return tex.Load(int4(p, lod)); } \
inline TYPE LoadTex3D(Texture3D<TYPE> tex,          int _, int3 p, int lod) { return tex.Load(int4(p, lod)); } \
inline TYPE LoadRWTex3D(RWTexture3D<TYPE> tex,      int3 p) { return tex[p]; } \
inline TYPE LoadRWTex3D(RWTexture2DArray<TYPE> tex, int3 p) { return tex[p]; }
_DECL_TYPES(_DECL_LoadTex3D)

#define LoadTex2DMS(NAME, SAMPLER, COORD, SMP) NAME.Load(COORD, SMP)
#define LoadTex2DArrayMS(NAME, SAMPLER, COORD, SMP) NAME.Load(COORD, SMP)


#define LoadLvlOffsetTex2D(TEX, SMP, P, L, O) (TEX).Load( int3((P).xy, L), O )

// #define SampleGradTex2D(NAME, SAMPLER, COORD, DX, DY) (NAME).SampleGrad( (SAMPLER), (COORD), (DX), (DY) )
#define _DECL_SampleGradTex2D(TYPE) \
inline TYPE SampleGradTex2D(Texture2D<TYPE> tex, SamplerState smp, float2 p, float2 dx, float2 dy) \
{ return tex.SampleGrad(smp, p, dx, dy); }
_DECL_FLOAT_TYPES(_DECL_SampleGradTex2D)

#define GatherRedTex2D(NAME, SAMPLER, COORD) (NAME).GatherRed( (SAMPLER), (COORD) )
#define GatherRedOffsetTex2D(NAME, SAMPLER, COORD, OFFSET) (NAME).GatherRed( (SAMPLER), (COORD), (OFFSET) )

// #define SampleTexCube(NAME, SAMPLER, COORD) (NAME).Sample( (SAMPLER), (COORD) )

#define _DECL_SampleTexCube(TYPE) \
inline TYPE SampleTexCube(TextureCube<TYPE> tex, SamplerState smp, float3 p) \
{ return tex.Sample(smp, p); }
_DECL_FLOAT_TYPES(_DECL_SampleTexCube)

#define _DECL_SampleTexCubeArray(TYPE) \
inline TYPE SampleTexCubeArray(TextureCubeArray<TYPE> tex, SamplerState smp, float4 p) \
{ return tex.Sample(smp, p); }
_DECL_FLOAT_TYPES(_DECL_SampleTexCubeArray)

#define _DECL_SampleUTexCube(TYPE) \
inline TYPE SampleUTexCube(TextureCube<TYPE> tex, SamplerState smp, float3 p) \
{ return tex.Sample(smp, p); }
_DECL_FLOAT_TYPES(_DECL_SampleUTexCube)

#define _DECL_SampleITexCube(TYPE) \
inline TYPE SampleITexCube(TextureCube<TYPE> tex, SamplerState smp, float3 p) \
{ return tex.Sample(smp, p); }
_DECL_FLOAT_TYPES(_DECL_SampleITexCube)

#define _DECL_SampleTex2D(TYPE) \
inline TYPE SampleTex2D(Texture2D<TYPE> tex, SamplerState smp, float2 p) \
{ return tex.Sample(smp, p); }
_DECL_FLOAT_TYPES(_DECL_SampleTex2D)

#define _DECL_SampleTex1D(TYPE) \
inline TYPE SampleTex1D(Texture1D<TYPE> tex, SamplerState smp, float p) \
{ return tex.Sample(smp, p); }
_DECL_FLOAT_TYPES(_DECL_SampleTex1D)

#define _DECL_SampleTex2DProj(TYPE) \
inline TYPE SampleTex2DProj(Texture2D<TYPE> tex, SamplerState smp, float4 p) \
{ return tex.Sample(smp, p.xy/p.w); }
_DECL_FLOAT_TYPES(_DECL_SampleTex2DProj)

#define _DECL_SampleTex2DArray(TYPE) \
inline TYPE SampleTex2DArray(Texture2DArray<TYPE> tex, SamplerState smp, float3 p) \
{ return tex.Sample(smp, p); }
_DECL_FLOAT_TYPES(_DECL_SampleTex2DArray)


// inline float4 SampleTex2D(Texture2D<float4> tex, SamplerState smp, float2 p)
// { return tex.Sample(smp, p); }

// #define SampleTex1D(NAME, SAMPLER, COORD) NAME.Sample(SAMPLER, COORD)
// #define SampleTex2D SampleTex1D
// #define SampleTex3D SampleTex1D
// #define SampleTex2DArray SampleTex1D

// #define CmpLvl0Tex2D(TEX, SMP, PC) (TEX).SampleCmpLevelZero(SMP, PC.xy, PC.z)

#define _DECL_CompareTex2D(TYPE) \
inline TYPE CompareTex2D(Texture2D<TYPE> tex, SamplerComparisonState smp, float3 p) \
{ return tex.SampleCmpLevelZero(smp, p.xy, p.z); }
_DECL_CompareTex2D(float)

#define _DECL_CompareTex2DProj(TYPE) \
inline float CompareTex2DProj(Texture2D<TYPE> tex, SamplerComparisonState smp, float4 p) \
{ return tex.SampleCmpLevelZero(smp, p.xy/p.w, p.z/p.w); }
_DECL_FLOAT_TYPES(_DECL_CompareTex2DProj)

// inline void Write2D(RWTexture2D<float4> tex, int2 p, float4 val)
// { tex[p] = val; }

#define _DECL_Write2D(TYPE) \
inline void Write2D(RWTexture2D<TYPE> tex, int2 p, TYPE val) \
{ tex[p] = val; }
_DECL_TYPES(_DECL_Write2D)

#if defined(DIRECT3D12)
#define _DECL_WriteRasterizerOrderedTexture2D(TYPE) \
inline void Write2D(RasterizerOrderedTexture2D<TYPE> tex, int2 p, TYPE val) \
{ tex[p] = val; }
_DECL_TYPES(_DECL_WriteRasterizerOrderedTexture2D)
#endif

#define _DECL_Write3D(TYPE) \
inline void Write3D(RWTexture3D<TYPE> tex, int3 p, TYPE val)      { tex[p] = val; } \
inline void Write3D(RWTexture2DArray<TYPE> tex, int3 p, TYPE val) { tex[p] = val; }
_DECL_TYPES(_DECL_Write3D)


#define Load2D(TEX, P) (TEX[(P)])
#define Load3D(TEX, P) (TEX[(P)])


// #define _DECL_Load2D(TYPE) \
// inline TYPE Load2D(Texture2D<TYPE> tex,   SamplerState smp, int2 p) { return tex[p]; }
// _DECL_TYPES(_DECL_Load2D)

// void Write2D(RWTexture2D<float>  dst, int2 coord, float4 val) { dst[coord] = val.x;    }
// void Write2D(RWTexture2D<float2> dst, int2 coord, float4 val) { dst[coord] = val.xy;   }
// void Write2D(RWTexture2D<float3> dst, int2 coord, float4 val) { dst[coord] = val.xyz;  }
// void Write2D(RWTexture2D<float4> dst, int2 coord, float4 val) { dst[coord] = val.xyzw; }

// void Write3D(RWTexture3D<float>  dst, int3 coord, float4 val) { dst[coord] = val.x;    }
// void Write3D(RWTexture3D<float2> dst, int3 coord, float4 val) { dst[coord] = val.xy;   }
// void Write3D(RWTexture3D<float3> dst, int3 coord, float4 val) { dst[coord] = val.xyz;  }
// void Write3D(RWTexture3D<float4> dst, int3 coord, float4 val) { dst[coord] = val.xyzw; }

// void Write3D(RWTexture2DArray<float>  dst, int3 coord, float4 val) { dst[coord] = val.x;    }
// void Write3D(RWTexture2DArray<float2> dst, int3 coord, float4 val) { dst[coord] = val.xy;   }
// void Write3D(RWTexture2DArray<float3> dst, int3 coord, float4 val) { dst[coord] = val.xyz;  }
// void Write3D(RWTexture2DArray<float4> dst, int3 coord, float4 val) { dst[coord] = val.xyzw; }
// void Write3D(RWTexture2DArray<uint>   dst, int3 coord, uint4  val) { dst[coord] = val.x;    }

// #define AtomicMin3D( DST, COORD, VALUE, ORIGINAL_VALUE ) (InterlockedMin((DST)[uint3((COORD).xyz)], (VALUE), (ORIGINAL_VALUE)))
#define _DECL_AtomicMin3D(TYPE) \
inline void AtomicMin3D(RWTexture3D<TYPE> tex, int3 p, TYPE val, out TYPE original_val) \
{ InterlockedMin(tex[p], val, original_val); } \
inline void AtomicMin3D(RWTexture2DArray<TYPE> tex, int3 p, TYPE val, out TYPE original_val) \
{ InterlockedMin(tex[p], val, original_val); }
_DECL_AtomicMin3D(uint)

// #define AtomicMax3D( DST, COORD, VALUE, ORIGINAL_VALUE ) (InterlockedMax((DST)[uint3((COORD).xyz)], (VALUE), (ORIGINAL_VALUE)))
#define _DECL_AtomicMax3D(TYPE) \
inline void AtomicMax3D(RWTexture3D<TYPE> tex, int3 p, TYPE val, out TYPE original_val) \
{ InterlockedMax(tex[p], val, original_val); } \
inline void AtomicMax3D(RWTexture2DArray<TYPE> tex, int3 p, TYPE val, out TYPE original_val) \
{ InterlockedMax(tex[p], val, original_val); }
_DECL_AtomicMax3D(uint)

#define _DECL_AtomicMin2D(TYPE) \
inline void AtomicMin2D(RWTexture2D<TYPE> tex, int2 p, TYPE val, out TYPE original_val) \
{ InterlockedMin(tex[p], val, original_val); }
_DECL_AtomicMin2D(uint)

#define _DECL_AtomicMax2D(TYPE) \
inline void AtomicMax2D(RWTexture2D<TYPE> tex, int2 p, TYPE val, out TYPE original_val) \
{ InterlockedMax(tex[p], val, original_val); }
_DECL_AtomicMax2D(uint)

#if defined(FT_ATOMICS_64)
#define AtomicMinU64(DST, VALUE) InterlockedMin(DST, VALUE)
#define AtomicMaxU64(DST, VALUE) InterlockedMax(DST, VALUE)
#endif

// #define _DECL_AtomicMin2DArray(TYPE) \
// inline void AtomicMin2DArray(RWTexture2DArray<TYPE> tex, int2 p, int layer, TYPE val, out TYPE original_val) \
// { InterlockedMin(tex[int3(p, layer)], val, original_val); }
// _DECL_AtomicMin2DArray(uint)

#if defined(DIRECT3D12)
    #define AtomicMin InterlockedMin
    #define AtomicMax InterlockedMax
#endif

// #ifndef ORBIS
// #if !defined(ORBIS) && !defined(PROSPERO)
// #endif

// #define WRITE2D(NAME, COORD, VAL) NAME[int2(COORD.xy)] = VAL
#define WRITE2D(NAME, COORD, VAL) write2D(NAME, COORD.xy, VAL)
#define WRITE3D(NAME, COORD, VAL) write3D(NAME, COORD.xyz, VAL)

#define UNROLL_N(X) [unroll(X)]
#define UNROLL [unroll]
#define LOOP [loop]
#define FLATTEN [flatten]



#if defined(ORBIS) || defined(PROSPERO)
    #undef Buffer
    #undef RWBuffer
#endif

#define Buffer(TYPE) StructuredBuffer<TYPE>
#define RWBuffer(TYPE) RWStructuredBuffer<TYPE>
#define WBuffer(TYPE) RWStructuredBuffer<TYPE>

#if defined(DIRECT3D12)
#define RWCoherentBuffer(TYPE) globallycoherent RWBuffer(TYPE)
#elif !defined(PROSPERO)
#define RWCoherentBuffer(TYPE) RWBuffer(TYPE)
#endif


#define NO_SAMPLER 0u
#if defined(DIRECT3D12)
inline int2 GetDimensions(RasterizerOrderedTexture2D<uint> t, int _) { uint2 d; t.GetDimensions(d.x, d.y); return d; }
#endif
inline int2 GetDimensions(Texture2D t, int _) { uint2 d; t.GetDimensions(d.x, d.y); return d; }
inline int2 GetDimensions(Texture2D t, SamplerState smp) { return GetDimensions(t, NO_SAMPLER); }
inline int2 GetDimensions(Texture2D<uint> t, int _) { uint2 d; t.GetDimensions(d.x, d.y); return d; }
inline int2 GetDimensions(Texture2D<uint> t, SamplerState smp) { return GetDimensions(t, NO_SAMPLER); }
inline int2 GetDimensions(Texture2D<float> t, int _) { uint2 d; t.GetDimensions(d.x, d.y); return d; }
inline int2 GetDimensions(Texture2D<float> t, SamplerState smp) { return GetDimensions(t, NO_SAMPLER); }
inline int2 GetDimensions(RWTexture2D<uint> t, int _) { uint2 d; t.GetDimensions(d.x, d.y); return d; }
inline int2 GetDimensions(RWTexture2D<uint> t, SamplerState smp) { return GetDimensions(t, NO_SAMPLER); }
inline int2 GetDimensions(RWTexture2D<float> t, int _) { uint2 d; t.GetDimensions(d.x, d.y); return d; }
inline int2 GetDimensions(RWTexture2D<float> t, SamplerState smp) { return GetDimensions(t, NO_SAMPLER); }
inline int2 GetDimensions(RWTexture2D<float4> t, int _) { uint2 d; t.GetDimensions(d.x, d.y); return d; }
inline int2 GetDimensions(RWTexture2D<float4> t, SamplerState smp) { return GetDimensions(t, NO_SAMPLER); }
inline int2 GetDimensions(TextureCube t, int _) { uint2 d; t.GetDimensions(d.x, d.y); return d; }
inline int2 GetDimensions(TextureCube t, SamplerState smp) { return GetDimensions(t, NO_SAMPLER); }

#define GetDimensionsMS(tex, dim) int2 dim; { uint3 d; tex.GetDimensions(d.x, d.y, d.z); dim = int2(d.xy); }
// #define GetDimensions(TEX, SAMPLER) _GetDimensions(TEX)

// int2 GetDimensions(Texture2D t, SamplerState s)
// { uint2 d; t.GetDimensions(d[0], d[1]); return d; }
// int2 GetDimensions(TextureCube t, SamplerState s)
// { uint2 d; t.GetDimensions(d[0], d[1]); return d; }

#define TexCube(ELEM_TYPE) TextureCube<ELEM_TYPE>
#define TexCubeArray(ELEM_TYPE) TextureCubeArray<ELEM_TYPE>

#define Tex1D(ELEM_TYPE) Texture1D<ELEM_TYPE>
#define Tex2D(ELEM_TYPE) Texture2D<ELEM_TYPE>
#define Tex3D(ELEM_TYPE) Texture3D<ELEM_TYPE>

#define Tex2DMS(ELEM_TYPE, SMP_CNT) Texture2DMS<ELEM_TYPE, SMP_CNT>

#define Tex1DArray(ELEM_TYPE) Texture1DArray<ELEM_TYPE>
#define Tex2DArray(ELEM_TYPE) Texture2DArray<ELEM_TYPE>

#define RWTex1D(ELEM_TYPE) RWTexture1D<ELEM_TYPE>
#define RWTex2D(ELEM_TYPE) RWTexture2D<ELEM_TYPE>
#define RWTex3D(ELEM_TYPE) RWTexture3D<ELEM_TYPE>

#define RWTex1DArray(ELEM_TYPE) RWTexture1DArray<ELEM_TYPE>
#define RWTex2DArray(ELEM_TYPE) RWTexture2DArray<ELEM_TYPE>

#define WTex1D RWTex1D
#define WTex2D RWTex2D
#define WTex3D RWTex3D
#define WTex1DArray RWTex1DArray
#define WTex2DArray RWTex2DArray

#define RTex1D RWTex1D
#define RTex2D RWTex2D
#define RTex3D RWTex3D
#define RTex1DArray RWTex1DArray
#define RTex2DArray RWTex2DArray

#ifdef DIRECT3D12
    #define RasterizerOrderedTex2D(ELEM_TYPE, GROUP_INDEX) RasterizerOrderedTexture2D<ELEM_TYPE>
    #define RasterizerOrderedTex2DArray(ELEM_TYPE, GROUP_INDEX) RasterizerOrderedTexture2DArray<ELEM_TYPE>
#elif !defined(PROSPERO)
    #define RasterizerOrderedTex2D(ELEM_TYPE, GROUP_INDEX) RWTex2D(ELEM_TYPE)
    #define RasterizerOrderedTex2DArray(ELEM_TYPE, GROUP_INDEX) RWTex2DArray(ELEM_TYPE)
#endif

#define Depth2D Tex2D
#define Depth2DMS Tex2DMS

#define SHADER_CONSTANT(INDEX, TYPE, NAME, VALUE) const TYPE NAME = VALUE

#define FSL_CONST(TYPE, NAME) static const TYPE NAME
#define STATIC static
#define INLINE inline

#define ToFloat3x3(NAME) ((float3x3) NAME )

#if defined(DIRECT3D12)
    #define CBUFFER(T) ConstantBuffer<T>
#elif defined(ORBIS) || defined(PROSPERO)
    #define CBUFFER(T) T
#else
    #define CBUFFER(T) cbuffer
#endif


#define EARLY_FRAGMENT_TESTS [earlydepthstencil]


// tesselation
#define TESS_VS_SHADER(X)

#define PCF_INIT
#define INPUT_PATCH(T, NC) InputPatch<T, NC>
#define OUTPUT_PATCH(T, NC) OutputPatch<T, NC>
#define FSL_OutputControlPointID(N) uint N : SV_OutputControlPointID
#define PATCH_CONSTANT_FUNC(F) [patchconstantfunc(F)]
#define OUTPUT_CONTROL_POINTS(P) [outputcontrolpoints(P)]
#define MAX_TESS_FACTOR(F) [maxtessfactor(F)]
#define FSL_DomainPartitioning(X, Y) [domain(X)] [partitioning(Y)]
#define FSL_OutputTopology(T) [outputtopology(T)]

#ifdef STAGE_TESE
#define TESS_LAYOUT(D, P, T) \
    [domain(D)]
#endif

#ifdef STAGE_TESC
#define TESS_LAYOUT(D, P, T) \
    [domain(D)] \
    [partitioning(P)] \
    [outputtopology(T)]
#endif

#if defined(DIRECT3D12) 
#define FSL_DomainLocation(N) float3 N : SV_DomainLocation
#else
#define FSL_DomainLocation(N) float2 N : SV_DomainLocation
#endif

#ifdef ENABLE_WAVEOPS

    #define  WaveGetMaxActiveIndex() WaveActiveMax(WaveGetLaneIndex())
	#define	 WaveIsHelperLane()		 IsHelperLane()

    #if !defined(ballot_t)
    #define ballot_t uint4
    #endif

    #if !defined(CountBallot)
    #define CountBallot(B) (countbits((B).x) + countbits((B).y) + countbits((B).z) + countbits((B).w))
    #endif

#endif


#define EnablePSInterlock()
#define BeginPSInterlock()
#define EndPSInterlock()


#ifndef STAGE_VERT
    #define VR_VIEW_ID(VID) (0)
#else
    #define VR_VIEW_ID 0
#endif
#define VR_MULTIVIEW_COUNT 1

#if defined(DIRECT3D12)
#define ADDRESS_MODE_REPEAT TEXTURE_ADDRESS_MODE_WRAP
#define ADDRESS_MODE_MIRROR TEXTURE_ADDRESS_MODE_MIRROR
#define ADDRESS_MODE_CLAMP_TO_EDGE TEXTURE_ADDRESS_MODE_CLAMP
#define ADDRESS_MODE_CLAMP_TO_BORDER TEXTURE_ADDRESS_MODE_BORDER

#define CMP_NEVER COMPARISON_FUNC_NONE
#define CMP_LESS COMPARISON_FUNC_LESS
#define CMP_EQUAL COMPARISON_FUNC_EQUAL
#define CMP_LEQUAL COMPARISON_FUNC_LESS_EQUAL
#define CMP_GREATER COMPARISON_FUNC_GREATER
#define CMP_NOTEQUAL COMPARISON_FUNC_NOT_EQUAL
#define CMP_GEQUAL COMPARISON_FUNC_GREATER_EQUAL
#define CMP_ALWAYS COMPARISON_FUNC_ALWAYS
#endif

#endif // _D3D_H

#line 1 "FSL/shaders.list"
#line 10 "FSL/shaders.list"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
#line 25 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fsl_srt.h"
#line 40 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fsl_srt.h"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/d3d12_srt.h"
#line 41 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fsl_srt.h"
#line 26 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
#line 185 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"

SamplerState gSamplerPointClamp : register( s0 , space100 ) ;
#line 188 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerPointWrap : register( s1 , space100 ) ;
#line 190 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerBilinearClamp : register( s2 , space100 ) ;
#line 192 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerBilinearWrap : register( s3 , space100 ) ;
#line 194 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerTrilinearClamp : register( s4 , space100 ) ;
#line 196 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerTrilinearWrap : register( s5 , space100 ) ;
#line 198 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerPointMirror : register( s6 , space100 ) ;
#line 200 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerPointBorder : register( s7 , space100 ) ;
#line 202 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerTrilinearMirror : register( s8 , space100 ) ;
#line 204 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerTrilinearBorder : register( s9 , space100 ) ;
#line 206 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerAnisotropic : register( s10 , space100 ) ;
#line 226 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerAnisoClampClamp : register( s11 , space100 ) ;
#line 228 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerAnisoClampWrap : register( s12 , space100 ) ;
#line 230 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerAnisoWrapClamp : register( s13 , space100 ) ;
#line 238 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSampler2xWrapWrap : register( s14 , space100 ) ;
#line 240 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSampler2xClampClamp : register( s15 , space100 ) ;
#line 242 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSampler2xClampWrap : register( s16 , space100 ) ;
#line 244 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSampler2xWrapClamp : register( s17 , space100 ) ;
#line 247 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"

#line 11 "FSL/shaders.list"
#line 177 "FSL/shaders.list"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
#line 31 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
#line 20 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
#line 26 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
STRUCT(ShadowMaskParams)
{
    float4x4 invViewProj;






    float4 screenParams;
    float4 maskParams;
    float4 slotPosRad[ 32 ];
    float4 slotTile[ 32 ];





    float4 biasParams;



    uint4 slotBits;
#line 60 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 slotFlick[ 32 ];










    float4x4 sunViewProj[ 2 ];
#line 85 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 sunParams;


    float4 sunCascadeTexel;
#line 104 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 sunPcf0;






    float4 sunPcf1;







    float4 volFog0;






    float4 volFog1;




    float4 volFog2;






    float4 volFog3;
#line 161 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 volFog4;







    float4 screenAlloc;
#line 181 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 shAr;
    float4 shAg;
    float4 shAb;
#line 196 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 skyParams;










    float4 skyAOMap;
#line 229 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 sunOcc;
#line 245 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 skyAO2;
#line 294 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 toneParams;
#line 308 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterFogCol;
#line 327 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterFogPlane;
#line 344 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterFogLight;
#line 361 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterFogExt;








    float4 waterFogScatter;

    float4 waterFogPhase;









    float4 waterFogPhase2;





    float4 waterFogKd;
#line 425 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 calParams;
#line 444 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 agxCurve;
    float4 agxCurveScale;
    float4 agxLookParams;








    float4 waterFogSun;
#line 471 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterDistort;
#line 485 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterCaustic;
#line 524 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterCaustic2;
#line 541 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterCausticRip;
    float4 waterCausticWake;
#line 562 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterCausticDyn;
#line 583 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterCaustic3;
#line 596 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterCut;
#line 627 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterCausticProj;
#line 645 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 grassParams;
#line 661 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 grassParams2;
#line 674 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 grassParams3;
#line 689 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 grassParams4;
#line 705 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 grassParams5;










    float4 grassParams6;










    float4 grassParams7;
#line 742 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 grassParams8;
#line 763 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 maskProf;
#line 780 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 aoBounce;
#line 781
};
#line 21 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
#line 27 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skyview.h.fsl"
#line 35 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skyview.h.fsl"
STRUCT(SkyViewData)
{



    float4x4 invViewProj;
#line 53 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skyview.h.fsl"
    float4 radiance;










    float4 sunDirW;




    float4 params;
#line 84 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skyview.h.fsl"
    float4 sunDisc;
#line 106 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skyview.h.fsl"
    float4 elemRadiance;
#line 107
};
#line 28 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
#line 35 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosparams.h.fsl"
#line 16 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosparams.h.fsl"
STRUCT(AtmosphereParams)
{
    float4 rayleigh;
    float4 mieSca;
    float4 mieExt;
    float4 ozone;
    float4 profGeom;


    float4 planet;
    float4 sunDir;
    float4 solar;
    float4 moonDir;
    float4 nightP;
    float4 marchP;

    float4 lutDims;
    float4 lutDims2;
#line 52 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosparams.h.fsl"
    float4 cloud;
#line 69 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosparams.h.fsl"
    float4 deckMsA;
    float4 deckMsB;
#line 101 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosparams.h.fsl"
    float4 deckMix;
#line 102
};
#line 36 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
#line 73 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
STRUCT(FrameData)
{
    float4x4 viewProj;



    float4 sunDir;
    float4 sunCol;
    float4 ambCol;
    float4 fogColNear;
    float4 fogParams;
    float4 eyePos;


    float4 debugParams;



    float4 dbgScales;








    float4 lodParams;

    float4 lodSunAmb;




    float4 lodEye;




    float4 skyParams;






    float4 gReflWaterClip;




    float4 skyZenith;



    float4 atlasDbg;



    float4 alphaParams;
#line 144 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
    float4 timeParams;






    float4 uvOffsets[8];






    float4 froxelDims;
#line 180 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
    float4 froxelZ;
#line 194 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
    float4 alphaShadowParams;
#line 195
};

STRUCT(BatchData)
{
    float4x4 worlds[ 1024 ];
#line 200
};
#line 221 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
STRUCT(LightData)
{
    float4 lightParams;
    float4 lights[ 128  * 3];









    float4 froxelDimsNear;
    float4 froxelZNear;
#line 236
};

        CBUFFER(FrameData) gFrameData :  register(b0,space1);





        Tex2D(float4) gAO :  register(t1,space1);
#line 272 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2DArray(float4) gWaterNormalVol :  register(t2,space1);
        Tex2D(float4) gRefractColor :  register(t3,space1);
        Tex2D(float4) gSceneLinDepth :  register(t4,space1);
        Tex2D(float4) gReflectColor :  register(t5,space1);






        Tex2D(uint4) gShadowMask :  register(t6,space1);




        Tex2D(float) gShadowAtlas :  register(t7,space1);
        Tex2D(float) gShadowAtlasDyn :  register(t8,space1);




        Tex2D(float4) gSunMoments :  register(t9,space1);





        Tex2D(float) gSunDepth :  register(t10,space1);





        Buffer(uint) gFroxelMask :  register(t11,space1);







        Buffer(uint) gFroxelMaskNear :  register(t12,space1);






        Buffer(float4) gUVAnim :  register(t13,space1);








        CBUFFER(ShadowMaskParams) gShadowParams :  register(b14,space1);








        Buffer(uint4) gAlphaStages :  register(t15,space1);









        Buffer(uint) gTerrainHeights :  register(t16,space1);
        Buffer(uint) gTerrainColor :  register(t17,space1);









        Buffer(uint) gTerrainTex :  register(t18,space1);
        Buffer(uint) gTerrainCellGrid :  register(t19,space1);





        Tex2DArray(float4) gTerrainArrays[ 32 ] :  register(t20,space1);










        Tex2D(float) gSkyHeight :  register(t52,space1);







        Tex2D(float) gSunOcc :  register(t53,space1);
#line 404 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gSkyColor :  register(t54,space1);
#line 417 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gReflectMips :  register(t55,space1);
#line 440 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2DArray(float) gWaterSlopeVar :  register(t56,space1);
#line 457 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float) gReflectDepth :  register(t57,space1);









        Tex2D(float4) gRippleField :  register(t58,space1);






        Tex2D(float4) gWakeField :  register(t59,space1);
#line 497 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gPortalGate :  register(t60,space1);
#line 518 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2DArray(float) gCausticField :  register(t61,space1);
#line 532 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gGrassCrush :  register(t62,space1);
#line 545 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gAtmosSkyView :  register(t63,space1);





        Tex2D(float4) gAtmosTransmittance :  register(t64,space1);






        CBUFFER(AtmosphereParams) gAtmosParams :  register(b65,space1);
#line 577 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float2) gMotionVectors :  register(t66,space1);





        Tex2D(float) gReactiveMask :  register(t67,space1);








        Tex2D(float4) gAtmosSkyViewClear :  register(t68,space1);




        CBUFFER(LightData) gLights :  register(b0,space3);










        CBUFFER(LightData) gLightsNear :  register(b1,space3);
#line 623 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gTextures[ 880 ] :  register(t0,space0);



        Tex2DArray(float4) gStaticsArrays[ 128 ] :  register(t880,space0);



        Tex2DArray(float4) gFlipArrays[ 16 ] :  register(t1008,space0);
        CBUFFER(BatchData) gBatch :  register(b0,space2);
#line 647 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        CBUFFER(SkyViewData) gSkyView :  register(b1,space2);
#line 32 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/tonemap.h.fsl"
#line 32 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/tonemap.h.fsl"
float3 tonemap(float3 c)
{
    c = clamp(c, 0.0f, 2.2f);
    c = (((0.0548303f * c - 0.189786f) * c - 0.154732f) * c + 1.12969f) * c;
    return c;
}
#line 64 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/tonemap.h.fsl"
float3 inverseTonemap(float3 d)
{
    float3 s = sqrt(max(1.0f - clamp(d, 0.0f, 1.0f), 0.0f));
    return (1.0f - s) * (1.7636304f + s * (-0.4027722f + s * 0.4084603f));
}
#line 33 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/linearize.h.fsl"
#line 39 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/linearize.h.fsl"
float3 srgbToLinear(float3 c)
{
    c = max(c, float3(0.0f, 0.0f, 0.0f));
    float3 lo = c * (1.0f / 12.92f);
    float3 hi = pow(c * (1.0f / 1.055f) + (0.055f / 1.055f), 2.4f);
    return lerp(lo, hi, step(float3(0.04045f, 0.04045f, 0.04045f), c));
}

float srgbToLinear1(float c)
{
    c = max(c, 0.0f);
    float lo = c * (1.0f / 12.92f);
    float hi = pow(c * (1.0f / 1.055f) + (0.055f / 1.055f), 2.4f);
    return (c >= 0.04045f) ? hi : lo;
}



float3 linearToSrgb(float3 c)
{
    c = max(c, float3(0.0f, 0.0f, 0.0f));
    float3 lo = c * 12.92f;
    float3 hi = 1.055f * pow(c, 1.0f / 2.4f) - 0.055f;
    return lerp(lo, hi, step(float3(0.0031308f, 0.0031308f, 0.0031308f), c));
}
#line 82 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/linearize.h.fsl"
float3 mod2xLinear(float3 tLinear)
{
    return srgbToLinear(2.0f * linearToSrgb(tLinear));
}
#line 34 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/scenecolor.h.fsl"
#line 113 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/scenecolor.h.fsl"
float3 tonemapInPass(float3 c)
{
    return (gShadowParams.toneParams.x > 0.5f) ? c : tonemap(c);
}







float3 liftInPass(float3 c)
{
    if (gShadowParams.toneParams.x <= 0.5f) { return c; }







    if (gShadowParams.toneParams.w > 0.5f) { return c; }






    if (gShadowParams.toneParams.y > 0.5f) {
        return srgbToLinear(inverseTonemap(linearToSrgb(c)));
    }
    return inverseTonemap(c);
}






bool sceneIsLinear()
{
    return gShadowParams.toneParams.y > 0.5f;
}





float3 decodeAuthored(float3 c)
{
    return (gShadowParams.toneParams.y > 0.5f) ? srgbToLinear(c) : c;
}




float decodeAuthored1(float c)
{
    return (gShadowParams.toneParams.y > 0.5f) ? srgbToLinear1(c) : c;
}




float3 mod2xStage(float3 t)
{
    return (gShadowParams.toneParams.y > 0.5f) ? mod2xLinear(t) : (t * 2.0f);
}
#line 282 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/scenecolor.h.fsl"
float3 expandExposedEmissiveP(float3 emis, float3 albedoRgb, float cov, float p)
{
    if (!(p > 0.0f)) { return emis; }





    float3 a = max(albedoRgb, float3(0.0f, 0.0f, 0.0f));
    float w = saturate(max(max(a.r, a.g), a.b) * max(cov, 0.0f));
    float lumaE = dot(max(emis, float3(0.0f, 0.0f, 0.0f)), float3(0.2126f, 0.7152f, 0.0722f));
    float fMin = 1.0f / max(lumaE, 1.0f);
    float f = fMin + (1.0f - fMin) * pow(w, p);
    return emis * f;
}



float3 expandExposedEmissive(float3 emis, float3 albedoRgb, float cov)
{
    return expandExposedEmissiveP(emis, albedoRgb, cov, gShadowParams.skyAO2.w);
}






float3 expandExposedEmissiveDelta(float3 emis, float3 albedoRgb, float cov)
{
    return expandExposedEmissive(emis, albedoRgb, cov) - emis;
}
#line 35 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/agx.h.fsl"
#line 90 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/agx.h.fsl"
float3 agxInset(float3 v)
{
    return float3(
        dot(v, float3(0.842479062253094f, 0.0784335999999992f, 0.0792237451477643f)),
        dot(v, float3(0.0423282422610123f, 0.878468636469772f, 0.0791661274605434f)),
        dot(v, float3(0.0423756549057051f, 0.0784336f, 0.879142973793104f)));
}


float3 agxOutsetMat(float3 v)
{
    return float3(
        dot(v, float3( 1.19687900512017f, -0.0980208811401368f, -0.0990297440797205f)),
        dot(v, float3(-0.0528968517574562f, 1.15190312990417f, -0.0989611768448433f)),
        dot(v, float3(-0.0529716355144438f, -0.0980434501171241f, 1.15107367264116f)));
}
#line 151 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/agx.h.fsl"
float3 agxContrast(float3 x, float4 curve, float4 curveScale)
{
    float3 px = float3(curveScale.z, curveScale.z, curveScale.z);


    float3 hi = step(px, x);
    float3 S = lerp(float3(-curveScale.x, -curveScale.x, -curveScale.x),
                     float3( curveScale.y, curveScale.y, curveScale.y), hi);
    float3 p = lerp(float3(curve.y, curve.y, curve.y),
                     float3(curve.z, curve.z, curve.z), hi);
    float3 t = max(curve.x * (x - px) / S, float3(0.0f, 0.0f, 0.0f));
    return (t / pow(1.0f + pow(t, p), 1.0f / p)) * S + float3(curveScale.w, curveScale.w, curveScale.w);
}
#line 211 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/agx.h.fsl"
float3 agxInverseNeutral(float3 d, float4 curve, float4 curveScale, float4 look)
{
    float3 v = pow(max(d, float3(0.0f, 0.0f, 0.0f)), float3(1.0f / 2.2f, 1.0f / 2.2f, 1.0f / 2.2f));



    float invPow = 1.0f / max(look.y, 1.0e-4f);
    v = (pow(max(v, float3(0.0f, 0.0f, 0.0f)), float3(invPow, invPow, invPow)) - look.w)
      / max(look.x, 1.0e-4f);




    float3 py = float3(curveScale.w, curveScale.w, curveScale.w);
    float3 hi = step(py, v);
    float3 S = lerp(float3(-curveScale.x, -curveScale.x, -curveScale.x),
                     float3( curveScale.y, curveScale.y, curveScale.y), hi);
    float3 p = lerp(float3(curve.y, curve.y, curve.y),
                     float3(curve.z, curve.z, curve.z), hi);
    float3 y = clamp((v - py) / S, float3(0.0f, 0.0f, 0.0f),
                      float3(1.0f - 1.0e-5f, 1.0f - 1.0e-5f, 1.0f - 1.0e-5f));
    float3 t = y / pow(max(1.0f - pow(y, p), float3(1.0e-12f, 1.0e-12f, 1.0e-12f)),
                        float3(1.0f, 1.0f, 1.0f) / p);
    float3 x = clamp(t * S / max(curve.x, 1.0e-4f) + float3(curveScale.z, curveScale.z, curveScale.z),
                      float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f));

    return exp2(x * ( (4.026069f)  -  (-12.47393f) ) +  (-12.47393f) );
}







float3 agx(float3 v, float4 curve, float4 curveScale)
{
    v = agxInset(max(v, float3(0.0f, 0.0f, 0.0f)));


    v = clamp(log2(max(v, float3(1.0e-10f, 1.0e-10f, 1.0e-10f))),
              float3( (-12.47393f) ,  (-12.47393f) ,  (-12.47393f) ),
              float3( (4.026069f) ,  (4.026069f) ,  (4.026069f) ));
    v = (v -  (-12.47393f) ) * (1.0f / ( (4.026069f)  -  (-12.47393f) ));
    return agxContrast(v, curve, curveScale);
}
#line 277 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/agx.h.fsl"
float3 agxLook(float3 v, float slope, float power, float sat, float offset)
{
    v = pow(max(v * slope + offset, float3(0.0f, 0.0f, 0.0f)),
            float3(power, power, power));
    float luma = dot(v, float3(0.2126f, 0.7152f, 0.0722f));
    return luma + sat * (v - luma);
}


float3 agxOutset(float3 v)
{
    v = agxOutsetMat(v);
    return pow(max(v, float3(0.0f, 0.0f, 0.0f)), float3(2.2f, 2.2f, 2.2f));
}
#line 36 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosphere.h.fsl"
#line 85 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosphere.h.fsl"
float atmosCloudProfile(AtmosphereParams P, float h)
{


    float x = (h - P.profGeom.z) / max(P.profGeom.w, 1.0f);
    float ax = min(abs(x), 1.0f);





    float ts = saturate((1.0f - ax) / (1.0f -  0.60f ));
    float flat = ts * ts * (3.0f - 2.0f * ts);
    float bell = 0.5f + 0.5f * cos( 3.14159265358979323846f  * ax);
    return lerp(flat, bell, saturate(P.cloud.w));
}




void atmosCloudAt(AtmosphereParams P, float h, out(float3) sca, out(float3) ext)
{
    float dC = atmosCloudProfile(P, h);
    sca = f3(P.cloud.x * dC);
    ext = f3(P.cloud.y * dC);
}
#line 124 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosphere.h.fsl"
float atmosDeckMs(AtmosphereParams P, float xNorm)
{
    float4 A = P.deckMsA;
    float4 B = P.deckMsB;
    float u = saturate(xNorm) * 7.0f;
    float r = A.x;
    r += (A.y - A.x) * saturate(u - 0.0f);
    r += (A.z - A.y) * saturate(u - 1.0f);
    r += (A.w - A.z) * saturate(u - 2.0f);
    r += (B.x - A.w) * saturate(u - 3.0f);
    r += (B.y - B.x) * saturate(u - 4.0f);
    r += (B.z - B.y) * saturate(u - 5.0f);
    r += (B.w - B.z) * saturate(u - 6.0f);
    return r;
}





float atmosDeckDepth01(AtmosphereParams P, float h)
{
    return saturate((P.profGeom.z + P.profGeom.w - h) / max(2.0f * P.profGeom.w, 1.0f));
}
#line 162 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosphere.h.fsl"
void atmosMediumAt(AtmosphereParams P, float h,
                   out(float3) scaR, out(float3) scaM, out(float3) scaC,
                   out(float3) extAir, out(float3) extCloud)
{
    float dR = exp(-max(h, 0.0f) / max(P.rayleigh.w, 1.0f));
    float dM = exp(-max(h, 0.0f) / max(P.mieSca.w, 1.0f));


    float dO = max(0.0f, 1.0f - abs(h - P.profGeom.x) / max(P.profGeom.y, 1.0f));

    scaR = P.rayleigh.xyz * dR;
    scaM = P.mieSca.xyz * dM;



    extAir = scaR + P.mieExt.xyz * dM + P.ozone.xyz * dO;
    atmosCloudAt(P, h, scaC, extCloud);
}


float atmosPhaseRayleigh(float cosT)
{
    return (3.0f / (16.0f *  3.14159265358979323846f )) * (1.0f + cosT * cosT);
}



float atmosPhaseMie(float cosT, float g)
{
    float g2 = g * g;
    float den = 1.0f + g2 - 2.0f * g * cosT;
    den = max(den, 1.0e-6f);
    return ((1.0f - g2) / (4.0f *  3.14159265358979323846f )) / (den * sqrt(den));
}





float atmosRaySphere(float3 ro, float3 rd, float rad)
{
    float b = dot(ro, rd);
    float c = dot(ro, ro) - rad * rad;
    float disc = b * b - c;
    if (disc < 0.0f) { return -1.0f; }
    float s = sqrt(disc);
    float t0 = -b - s;
    float t1 = -b + s;
    if (t1 < 0.0f) { return -1.0f; }
    return (t0 < 0.0f) ? t1 : t0;
}


float atmosHorizonCos(float r, float Rg)
{
    float s = Rg / max(r, 1.0f);
    return -sqrt(max(0.0f, 1.0f - s * s));
}
#line 249 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosphere.h.fsl"
bool atmosDeckSpan(AtmosphereParams P, float3 ro, float3 rd, float tMax,
                   out(float) tIn, out(float) tOut)
{
    tIn = 0.0f;
    tOut = -1.0f;
    float hw = P.profGeom.w;
    if (P.cloud.y <= 0.0f || hw <= 0.0f || tMax <= 0.0f) { return false; }

    float rHi = P.planet.x + P.profGeom.z + hw;
    float rLo = P.planet.x + P.profGeom.z - hw;
    float b = dot(ro, rd);
    float cH = dot(ro, ro) - rHi * rHi;
    float dH = b * b - cH;


    if (dH <= 0.0f) { return false; }
    float sH = sqrt(dH);
    float t0 = max(0.0f, -b - sH);
    float t1 = min(tMax, -b + sH);






    float cL = dot(ro, ro) - rLo * rLo;
    if (cL < 0.0f)
    {
        float sL = sqrt(max(0.0f, b * b - cL));
        t0 = max(t0, -b + sL);
    }
    if (t1 <= t0) { return false; }
    tIn = t0;
    tOut = t1;
    return true;
}




int atmosMarchPlan(AtmosphereParams P, float3 ro, float3 rd, float tMax,
                   int baseSteps, int deckSteps, int mode,
                   out(float4) planT, out(float3) planN)
{
    planT = f4(0.0f);
    planN = float3(float(baseSteps), 0.0f, 0.0f);

    float tIn, tOut;
    if (deckSteps <= 0 || !atmosDeckSpan(P, ro, rd, tMax, tIn, tOut))
    {
        return baseSteps;
    }




    float uIn = saturate(tIn / tMax);
    float uOut = saturate(tOut / tMax);
    if (mode ==  1 )
    {
        uIn = sqrt(uIn);
        uOut = sqrt(uOut);
    }

    int n0 = 0;
    int n2 = 0;
    if (baseSteps > 0)
    {




        float outside = max(uIn + (1.0f - uOut), 1.0e-6f);
        n0 = (uIn > 1.0e-5f) ? max(1, int(float(baseSteps) * uIn / outside + 0.5f)) : 0;
        n2 = (uOut < 1.0f - 1.0e-5f) ? max(1, baseSteps - n0) : 0;
    }

    planT = float4(tIn, tOut, uIn, uOut);
    planN = float3(float(n0), float(deckSteps), float(n2));
    return n0 + deckSteps + n2;
}



void atmosMarchStep(float4 planT, float3 planN, int s, float tMax, int mode,
                    out(float) t0, out(float) t1)
{
    int n0 = int(planN.x);
    int nD = int(planN.y);
    float u0, u1;

    if (nD <= 0)
    {



        u0 = float(s) / float(max(n0, 1));
        u1 = float(s + 1) / float(max(n0, 1));
    }
    else if (s < n0)
    {
        u0 = planT.z * (float(s) / float(n0));
        u1 = planT.z * (float(s + 1) / float(n0));
    }
    else if (s < n0 + nD)
    {



        int k = s - n0;
        t0 = planT.x + (planT.y - planT.x) * (float(k) / float(nD));
        t1 = planT.x + (planT.y - planT.x) * (float(k + 1) / float(nD));
        return;
    }
    else
    {
        int n2 = max(int(planN.z), 1);
        int k = s - n0 - nD;
        u0 = planT.w + (1.0f - planT.w) * (float(k) / float(n2));
        u1 = planT.w + (1.0f - planT.w) * (float(k + 1) / float(n2));
    }

    t0 = (mode ==  1 ) ? (u0 * u0 * tMax) : (u0 * tMax);
    t1 = (mode ==  1 ) ? (u1 * u1 * tMax) : (u1 * tMax);
}





float atmosUnitFromUv(float u, float n) { return (u - 0.5f / n) / max(1.0f - 1.0f / n, 1.0e-6f); }
float atmosUvFromUnit(float x, float n) { return 0.5f / n + x * (1.0f - 1.0f / n); }





float2 atmosTransmittanceUv(AtmosphereParams P, float r, float mu)
{
    float Rg = P.planet.x;
    float Rt = P.planet.y;
    float H = sqrt(max(0.0f, Rt * Rt - Rg * Rg));
    float rho = sqrt(max(0.0f, r * r - Rg * Rg));
    float d = max(0.0f, -r * mu + sqrt(max(0.0f, r * r * (mu * mu - 1.0f) + Rt * Rt)));
    float dMin = Rt - r;
    float dMax = rho + H;
    float xMu = (dMax - dMin > 1.0e-3f) ? saturate((d - dMin) / (dMax - dMin)) : 0.0f;
    float xR = saturate(rho / max(H, 1.0e-3f));
    return float2(atmosUvFromUnit(xMu, P.lutDims.z), atmosUvFromUnit(xR, P.lutDims.w));
}

void atmosTransmittanceParams(AtmosphereParams P, float2 uv, out(float) r, out(float) mu)
{
    float Rg = P.planet.x;
    float Rt = P.planet.y;
    float xMu = atmosUnitFromUv(uv.x, P.lutDims.z);
    float xR = atmosUnitFromUv(uv.y, P.lutDims.w);
    float H = sqrt(max(0.0f, Rt * Rt - Rg * Rg));
    float rho = H * saturate(xR);
    r = sqrt(rho * rho + Rg * Rg);
    float dMin = Rt - r;
    float dMax = rho + H;
    float d = dMin + saturate(xMu) * (dMax - dMin);
    mu = (d < 1.0e-3f) ? 1.0f : clamp((H * H - rho * rho - d * d) / (2.0f * r * d), -1.0f, 1.0f);
}





float2 atmosMultiScatterUv(AtmosphereParams P, float r, float muSun)
{
    float xS = saturate(muSun * 0.5f + 0.5f);
    float xR = saturate((r - P.planet.x) / max(P.planet.y - P.planet.x, 1.0f));
    return float2(atmosUvFromUnit(xS, P.lutDims2.x), atmosUvFromUnit(xR, P.lutDims2.y));
}

void atmosMultiScatterParams(AtmosphereParams P, float2 uv, out(float) r, out(float) muSun)
{
    float xS = atmosUnitFromUv(uv.x, P.lutDims2.x);
    float xR = atmosUnitFromUv(uv.y, P.lutDims2.y);
    muSun = clamp(saturate(xS) * 2.0f - 1.0f, -1.0f, 1.0f);
#line 455 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosphere.h.fsl"
    float rGround = P.planet.x +  10.0f ;
    r = max(rGround, P.planet.x + saturate(xR) * (P.planet.y - P.planet.x));
}
#line 470 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosphere.h.fsl"
void atmosSkyViewParams(AtmosphereParams P, float2 uv, float r,
                        out(float) viewZenithCos, out(float) lightViewCos)
{
    float u = atmosUnitFromUv(uv.x, P.lutDims.x);
    float v = atmosUnitFromUv(uv.y, P.lutDims.y);
    u = saturate(u);
    v = saturate(v);

    float Rg = P.planet.x;
    float cosBeta = sqrt(max(0.0f, r * r - Rg * Rg)) / max(r, 1.0f);
    float beta = acos(clamp(cosBeta, -1.0f, 1.0f));
    float zenithHorizonAngle =  3.14159265358979323846f  - beta;

    if (v < 0.5f)
    {
        float c = 1.0f - 2.0f * v;
        c = 1.0f - c * c;
        viewZenithCos = cos(zenithHorizonAngle * c);
    }
    else
    {
        float c = v * 2.0f - 1.0f;
        c = c * c;
        viewZenithCos = cos(zenithHorizonAngle + beta * c);
    }
    float c2 = u * u;
    lightViewCos = -(c2 * 2.0f - 1.0f);
}

float2 atmosSkyViewUv(AtmosphereParams P, float r, float viewZenithCos, float lightViewCos)
{
    float Rg = P.planet.x;
    float cosBeta = sqrt(max(0.0f, r * r - Rg * Rg)) / max(r, 1.0f);
    float beta = acos(clamp(cosBeta, -1.0f, 1.0f));
    float zenithHorizonAngle =  3.14159265358979323846f  - beta;
    float viewZenithAngle = acos(clamp(viewZenithCos, -1.0f, 1.0f));

    float v;
    if (viewZenithAngle < zenithHorizonAngle)
    {
        float c = (zenithHorizonAngle > 1.0e-4f) ? (viewZenithAngle / zenithHorizonAngle) : 0.0f;
        c = sqrt(max(0.0f, 1.0f - c));
        v = (1.0f - c) * 0.5f;
    }
    else
    {
        float c = (beta > 1.0e-4f) ? ((viewZenithAngle - zenithHorizonAngle) / beta) : 0.0f;
        v = sqrt(saturate(c)) * 0.5f + 0.5f;
    }
    float u = sqrt(saturate(-lightViewCos * 0.5f + 0.5f));
    return float2(atmosUvFromUnit(saturate(u), P.lutDims.x),
                  atmosUvFromUnit(saturate(v), P.lutDims.y));
}




float2 atmosSkyViewUvFromDir(AtmosphereParams P, float r, float3 dir)
{
    float3 toSun = P.sunDir.xyz;


    float3 sunH = toSun - float3(0.0f, 0.0f, 1.0f) * toSun.z;
    float sunHLen = length(sunH);
    float3 dirH = dir - float3(0.0f, 0.0f, 1.0f) * dir.z;
    float dirHLen = length(dirH);
    float lightViewCos = 1.0f;
    if (sunHLen > 1.0e-4f && dirHLen > 1.0e-4f)
    {
        lightViewCos = clamp(dot(sunH / sunHLen, dirH / dirHLen), -1.0f, 1.0f);
    }
    return atmosSkyViewUv(P, r, clamp(dir.z, -1.0f, 1.0f), lightViewCos);
}
#line 37 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
#line 40 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/phase.h.fsl"
#line 17 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/phase.h.fsl"
float hgNorm(float cosT, float g)
{
    float g2 = g * g;
    float d = 1.0f + g2 - 2.0f * g * cosT;
    return (1.0f - g2) / max(d * sqrt(max(d, 1.0e-4f)), 1.0e-4f);
}
#line 39 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/phase.h.fsl"
float phaseDualLobe(float cosT, float iso,
                    float gFwd, float gainFwd, float gBack, float gainBack, float softCeil)
{
    float lobes = gainFwd * hgNorm(cosT, gFwd) + gainBack * hgNorm(cosT, gBack);
    if (softCeil > 0.0f) { lobes = lobes / (1.0f + lobes / softCeil); }
    return iso + lobes;
}
#line 41 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterplane.h.fsl"
#line 18 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterplane.h.fsl"
bool waterFogActive()
{
    return gShadowParams.waterFogPlane.y >= 0.5f;
}
#line 36 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterplane.h.fsl"
bool waterCameraSubmerged()
{
    return gShadowParams.waterFogPlane.z > 0.5f;
}





bool waterPlaneValid()
{
    return gShadowParams.waterFogPlane.w > 0.5f;
}




float waterPlaneRelZ()
{
    return gShadowParams.waterFogPlane.x - gFrameData.lodEye.z;
}


bool waterFogCameraSubmerged()
{
    return waterFogActive() && waterCameraSubmerged();
}
#line 42 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"




float waterFogPlaneRelZ()
{
    return gShadowParams.waterFogPlane.x - gFrameData.lodEye.z;
}
#line 71 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/caustic.h.fsl"
#line 72 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
#line 83 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
float causticFetch(float2 uv, float s0, float s1, float sf)
{
    float a0 =  gCausticField.SampleLevel(gSamplerBilinearWrap, float3(uv, s0), 0.0f) ;
    float a1 =  gCausticField.SampleLevel(gSamplerBilinearWrap, float3(uv, s1), 0.0f) ;
    return lerp(a0, a1, sf);
}






float causticDynFetch(float2 uv, float slice)
{
    return  gCausticField.SampleLevel(gSamplerBilinearClamp, float3(uv, slice), 0.0f) ;
}






float causticDynEdge(float2 uv)
{
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) { return 0.0f; }
    float2 e = min(uv, 1.0f - uv);
    return saturate(min(e.x, e.y) * 8.0f);
}
#line 124 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
float waterFogDepthBelow(float3 worldPosRel)
{
    return max(waterFogPlaneRelZ() - worldPosRel.z, 0.0f);
}


float waterFogEyeDepth()
{
    return max(waterFogPlaneRelZ(), 0.0f);
}
#line 173 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
float waterPathLength(float3 worldPosRel, float waterRelZ, bool camUnder)
{
    float d = length(worldPosRel);
    float posToSurf = waterRelZ - worldPosRel.z;

    if (camUnder) {
        if (posToSurf >= 0.0f) { return d; }









        float camToSurf = max(waterRelZ, 0.0f);
        float share = (worldPosRel.z > 0.0f) ? (camToSurf / worldPosRel.z) : 1.0f;
        return d * saturate(share);
    }

    if (posToSurf <= 0.0f) { return 0.0f; }
    return d * (posToSurf / max(-worldPosRel.z, posToSurf));
}
#line 213 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
float2 waterFogSample(float3 worldPosRel)
{
    if (!waterFogActive()) { return float2(0.0f, 0.0f); }

    float d = length(worldPosRel);
    float len = waterPathLength(worldPosRel, waterFogPlaneRelZ(),
                                gShadowParams.waterFogPlane.z > 0.5f);
    float frac = (d > 1.0e-4f) ? saturate(len / d) : 0.0f;
    return float2(1.0f - exp(-gShadowParams.waterFogCol.w * len), frac);
}



float waterFogFactor(float3 worldPosRel)
{
    return waterFogSample(worldPosRel).x;
}
#line 262 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
float2 waterFogSampleSubmerged(float3 worldPosRel)
{
    if (!waterFogActive()) { return float2(0.0f, 0.0f); }
    float d = length(worldPosRel);
    return float2(1.0f - exp(-gShadowParams.waterFogCol.w * d), 1.0f);
}
#line 294 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
float3 waterFogColor(float3 worldPosRel)
{
    float3 c = gShadowParams.waterFogCol.rgb;

    float s = gShadowParams.waterFogLight.w;
    if (s <= 0.0f) { return c; }

    float depthRep = 0.5f * (waterFogEyeDepth() + waterFogDepthBelow(worldPosRel));
    float3 t = exp(-gShadowParams.waterFogLight.rgb * depthRep *  1.2039f );
    return c * lerp(float3(1.0f, 1.0f, 1.0f), t, s);
}
#line 338 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
void waterLightTransmit(float3 worldPosRel, out float3 tSun, out float3 tAmb)
{
    tSun = float3(1.0f, 1.0f, 1.0f);
    tAmb = tSun;

    float strength = gShadowParams.waterFogLight.w;
    if (!waterFogActive() || strength <= 0.0f) { return; }

    float depth = waterFogDepthBelow(worldPosRel);
    if (depth <= 0.0f) { return; }
#line 361 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float3 k = lerp(gShadowParams.waterFogLight.rgb, gShadowParams.waterFogKd.rgb,
                    gShadowParams.waterFogExt.w);



    float cosAir = abs(gFrameData.sunDir.z);
    float cosWater = sqrt(max(1.0f - (1.0f - cosAir * cosAir) / ( 1.333f  *  1.333f ), 0.0f));
    float pathSun = depth / max(cosWater,  0.6612f );

    tSun = lerp(tSun, exp(-k * pathSun), strength);
    tAmb = lerp(tAmb, exp(-k * depth *  1.2039f ), strength);
#line 412 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float causStr = gShadowParams.waterCaustic.x;
    float dFar = gShadowParams.waterCaustic2.y * max(gFrameData.lodParams.w, 4096.0f);
    float dfade = (dFar > 0.0f)
                        ? 1.0f - smoothstep(0.5f * dFar, dFar, length(worldPosRel))
                        : 1.0f;
#line 436 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float cd0 = gShadowParams.waterCaustic3.x;



    float cdecay = (cd0 > 0.0f) ? exp(-pow(max(depth, 0.0f) / cd0, 1.27f)) : 1.0f;
    if (causStr * dfade * cdecay > 0.0f) {



        float2 worldXY = gFrameData.lodEye.xy + worldPosRel.xy;
#line 470 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
        float tanW = sqrt(max(1.0f - cosWater * cosWater, 0.0f)) / max(cosWater,  0.6612f );
        float2 sunXY = gFrameData.sunDir.xy;
        float sunL = length(sunXY);
        float2 bearing = (sunL > 1e-4f) ? (sunXY / sunL) : float2(1.0f, 0.0f);
        float2 shift = (sunL > 1e-4f) ? bearing * (-depth * tanW) : float2(0.0f, 0.0f);

        float2 uv = (worldXY + shift) * gShadowParams.waterCaustic.y;






        float3 caustic = float3(1.0f, 1.0f, 1.0f);








        float sIdx = gShadowParams.waterCaustic.z * log2(max(depth, 1.0f))
                    + gShadowParams.waterCaustic.w;
#line 507 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
        float fade = saturate(sIdx + 1.0f)
                    * (1.0f - saturate(sIdx - (float)( 5  - 1)));
#line 521 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
        if (fade > 0.0f && gShadowParams.waterCaustic3.y > 0.0f) {
            float s0 = clamp(floor(sIdx), 0.0f, (float)( 5  - 1));
            float s1 = min(s0 + 1.0f, (float)( 5  - 1));
            float sf = saturate(sIdx - s0);
#line 577 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
            float2 duv = bearing * (gShadowParams.waterCaustic2.x * depth);
            float want = length(duv);
            float3 dev;
            if (want > 1.0e-9f) {



                float2 pv = (duv / want) * (1.0f / (float) 512 );
                float g0 = causticFetch(uv, s0, s1, sf);
                float gp = causticFetch(uv + pv, s0, s1, sf);
                float gm = causticFetch(uv - pv, s0, s1, sf);


                float sw = 0.5f * (gp - gm) * (want * (float) 512 );



                float maxSw = 0.9f * (1.0f + g0);
                sw = clamp(sw, -maxSw, maxSw);
                dev.g = g0;
                dev.b = g0 + sw;
                dev.r = g0 - 0.4409f * sw;
            } else {


                dev = causticFetch(uv, s0, s1, sf).xxx;
            }



            caustic = 1.0f + dev * fade;
        }
#line 624 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
        float2 dynPos = worldPosRel.xy + shift;



        float ripStr = gShadowParams.waterCausticDyn.x;
        if (ripStr > 0.0f) {
            float2 ruv = (dynPos - gShadowParams.waterCausticRip.xy)
                       * (gShadowParams.waterCausticRip.z / max(gShadowParams.waterCausticRip.w, 1.0f));
            float redge = causticDynEdge(ruv);
            if (redge > 0.0f) {
                float rIdx = gShadowParams.waterCausticDyn.z * log2(max(depth, 1.0f))
                           + gShadowParams.waterCausticDyn.w;
#line 651 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
                float rMax = gShadowParams.waterCaustic2.w;
                float rf = saturate(rIdx + 1.0f)
                         * (1.0f - smoothstep(rMax, 1.5f * rMax, depth));
                if (rf > 0.0f) {
                    float r0 = clamp(floor(rIdx), 0.0f, (float)( 2  - 1));
                    float r1 = min(r0 + 1.0f, (float)( 2  - 1));
                    float rd = lerp(causticDynFetch(ruv, (float) 5  + r0),
                                    causticDynFetch(ruv, (float) 5  + r1),
                                    saturate(rIdx - r0));










                    caustic *= max(1.0f + rd * (rf * redge * ripStr), 0.0f);
                }
            }
        }
#line 690 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
        float wakeStr = gShadowParams.waterCausticDyn.y;
        if (wakeStr > 0.0f) {
            float2 wuv = (dynPos - gShadowParams.waterCausticWake.xy)
                       * (gShadowParams.waterCausticWake.z / max(gShadowParams.waterCausticWake.w, 1.0f));
            float wedge = causticDynEdge(wuv);
            if (wedge > 0.0f) {
                float wd = causticDynFetch(wuv, (float) 7 );
                float wMax = gShadowParams.waterCaustic2.z;
                float wscale = min(depth, wMax) * (1.0f /  256.0f )
                             * (1.0f - smoothstep(wMax, 1.5f * wMax, depth));




                wscale = lerp(wscale, 1.0f, saturate(gShadowParams.waterCaustic3.z));






                caustic *= max(1.0f + wd * (wedge * wakeStr * wscale), 0.0f);
            }
        }






        tSun *= lerp(float3(1.0f, 1.0f, 1.0f), caustic, saturate(causStr) * dfade * cdecay);
    }
}
#line 776 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
bool causticProjectorActive(float3 worldPosRel)
{
    if (!waterFogActive()) { return false; }





    if (waterFogDepthBelow(worldPosRel) > 0.0f) {
        return gShadowParams.waterCausticProj.x > 0.0f
            || gShadowParams.waterCausticProj.z > 0.0f;
    }
    return gShadowParams.waterCausticProj.y > 0.0f;
}
#line 855 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
float causticProjector(float3 worldPosRel, float3 lightPosRel)
{
    float w = waterFogPlaneRelZ();
    bool fragBelow = waterFogDepthBelow(worldPosRel) > 0.0f;
    bool lightBelow = lightPosRel.z < w;




    if (!fragBelow && !lightBelow) { return 1.0f; }




    float lampOff = abs(lightPosRel.z - w);
    if (lampOff <= 1.0e-4f) { return 1.0f; }

    float strength = fragBelow ? (lightBelow ? gShadowParams.waterCausticProj.x
                                             : gShadowParams.waterCausticProj.z)
                               : gShadowParams.waterCausticProj.y;
    if (strength <= 0.0f) { return 1.0f; }




    float virtZ;
    if (lightBelow) {
        virtZ = fragBelow ? (w + lampOff)
                          : (w - lampOff /  1.333f );
    } else {
        virtZ = w + lampOff *  1.333f ;
    }
    float3 virtPos = float3(lightPosRel.xy, virtZ);





    float3 toFrag = worldPosRel - virtPos;
    float t = (w - virtPos.z) / (worldPosRel.z - virtPos.z);
    float2 surfXY = virtPos.xy + t * toFrag.xy;




    float segLen = length(toFrag);
    float pathL = (1.0f - t) * segLen;
    if (pathL <= 0.0f) { return 1.0f; }
#line 920 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float tir = 1.0f;
    if (fragBelow && lightBelow) {
        float cosInc = abs(toFrag.z) / max(segLen, 1.0e-4f);
        tir = 1.0f - smoothstep( 0.6612f  * 0.85f,  0.6612f , cosInc);
        if (tir <= 0.0f) { return 1.0f; }
    }
#line 938 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float devPerSlope = fragBelow ? (lightBelow ? 2.0f :  (1.0f - 1.0f / 1.333f ) ) :  ( 1.333f - 1.0f) ;
    float dEff = (devPerSlope /  (1.0f - 1.0f / 1.333f ) ) * pathL;
    float sIdx = gShadowParams.waterCaustic.z * log2(max(dEff, 1.0f))
                + gShadowParams.waterCaustic.w;
    float fade = saturate(sIdx + 1.0f)
                * (1.0f - saturate(sIdx - (float)( 5  - 1)));
#line 956 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float cdecay = 1.0f;
    if (fragBelow) {
        float cd0 = gShadowParams.waterCaustic3.x;



        cdecay = (cd0 > 0.0f) ? exp(-pow(max(pathL, 0.0f) / cd0, 1.27f)) : 1.0f;
    }

    float k = saturate(strength) * tir * cdecay;
    if (k <= 0.0f) { return 1.0f; }




    float caustic = 1.0f;





    if (fade > 0.0f && gShadowParams.waterCaustic3.y > 0.0f) {
        float2 uv = (gFrameData.lodEye.xy + surfXY) * gShadowParams.waterCaustic.y;
        float s0 = clamp(floor(sIdx), 0.0f, (float)( 5  - 1));
        float s1 = min(s0 + 1.0f, (float)( 5  - 1));
        float sf = saturate(sIdx - s0);





        caustic = 1.0f + causticFetch(uv, s0, s1, sf) * fade;
    }







    float ripStr = gShadowParams.waterCausticDyn.x;
    if (ripStr > 0.0f) {
        float2 ruv = (surfXY - gShadowParams.waterCausticRip.xy)
                   * (gShadowParams.waterCausticRip.z / max(gShadowParams.waterCausticRip.w, 1.0f));
        float redge = causticDynEdge(ruv);
        if (redge > 0.0f) {
            float rIdx = gShadowParams.waterCausticDyn.z * log2(max(dEff, 1.0f))
                       + gShadowParams.waterCausticDyn.w;
            float rMax = gShadowParams.waterCaustic2.w;
            float rf = saturate(rIdx + 1.0f)
                     * (1.0f - smoothstep(rMax, 1.5f * rMax, dEff));
            if (rf > 0.0f) {
                float r0 = clamp(floor(rIdx), 0.0f, (float)( 2  - 1));
                float r1 = min(r0 + 1.0f, (float)( 2  - 1));
                float rd = lerp(causticDynFetch(ruv, (float) 5  + r0),
                                causticDynFetch(ruv, (float) 5  + r1),
                                saturate(rIdx - r0));



                caustic *= max(1.0f + rd * (rf * redge * ripStr), 0.0f);
            }
        }
    }

    float wakeStr = gShadowParams.waterCausticDyn.y;
    if (wakeStr > 0.0f) {
        float2 wuv = (surfXY - gShadowParams.waterCausticWake.xy)
                   * (gShadowParams.waterCausticWake.z / max(gShadowParams.waterCausticWake.w, 1.0f));
        float wedge = causticDynEdge(wuv);
        if (wedge > 0.0f) {
            float wd = causticDynFetch(wuv, (float) 7 );
            float wMax = gShadowParams.waterCaustic2.z;
            float wscale = min(dEff, wMax) * (1.0f /  256.0f )
                         * (1.0f - smoothstep(wMax, 1.5f * wMax, dEff));
            wscale = lerp(wscale, 1.0f, saturate(gShadowParams.waterCaustic3.z));
            caustic *= max(1.0f + wd * (wedge * wakeStr * wscale), 0.0f);
        }
    }





    return lerp(1.0f, caustic, k);
}
#line 1112 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
float3 waterInscatterSeg(float3 sigT, float3 kLight, float3 tauView, float L,
                         float dNear, float dFar, float mEff, float slant, bool openEnded)
{
    float3 e0 = exp(-kLight * (slant * dNear));


    float3 r = sigT + kLight * (slant * mEff);

    if (openEnded) {





        return e0 / max(r, max(0.05f * sigT, float3(1.0e-9f, 1.0e-9f, 1.0e-9f)));
    }

    float3 e1 = exp(-kLight * (slant * dFar) - tauView);




    float3 sgn = lerp(float3(-1.0f, -1.0f, -1.0f), float3(1.0f, 1.0f, 1.0f), step(0.0f, r));
    float3 den = sgn * max(abs(r), float3(1.0e-30f, 1.0e-30f, 1.0e-30f));
    float3 lim = (0.5f * L) * (e0 + e1);
    return lerp((e0 - e1) / den, lim, step(abs(r * L), float3(1.0e-4f, 1.0e-4f, 1.0e-4f)));
}




float waterPhase(float cosT)
{
    return phaseDualLobe(cosT,
                         gShadowParams.waterFogPhase2.x,
                         gShadowParams.waterFogPhase.x, gShadowParams.waterFogPhase.y,
                         gShadowParams.waterFogPhase.z, gShadowParams.waterFogPhase.w,
                         gShadowParams.waterFogPhase2.y);
}
#line 1167 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
void waterColumn(float3 worldPosRel, float L, bool openEnded, out float3 inscat, out float3 trans)
{
    float3 sigT = gShadowParams.waterFogExt.rgb;
    float3 sigS = gShadowParams.waterFogScatter.rgb;



    float3 kLgt = gShadowParams.waterFogKd.rgb;

    float d = max(length(worldPosRel), 1.0e-4f);




    float mEff = -worldPosRel.z / d;
    float dNear = waterFogEyeDepth();
    float dFar = max(dNear + mEff * L, 0.0f);
#line 1211 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float mEffSeg = (L > 0.0f) ? ((dFar - dNear) / L) : mEff;

    float3 tauView = sigT * L;
    trans = openEnded ? float3(0.0f, 0.0f, 0.0f) : exp(-tauView);
#line 1289 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float3 sigA = max(sigT - sigS, float3(0.0f, 0.0f, 0.0f));
    float wIso = saturate(gShadowParams.waterFogPhase2.z);
    float3 sigSi = sigS * (1.0f - gShadowParams.waterFogPhase2.w);





    float3 sigSm = lerp(sigS, sigSi, wIso);
    float3 sigTm = lerp(sigT, sigA + sigSi, wIso);
    float3 tauViewM = sigTm * L;





    float cosAir = abs(gFrameData.sunDir.z);
    float cosWater = sqrt(max(1.0f - (1.0f - cosAir * cosAir) / ( 1.333f  *  1.333f ), 0.0f));
    float cosW = max(cosWater,  0.6612f );
    float slantSun = 1.0f / cosW;
#line 1370 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float cosEntry = (gShadowParams.waterFogSun.y >= 0.0f) ? gShadowParams.waterFogSun.y : cosAir;
    float cosWEnt = max(sqrt(max(1.0f - (1.0f - cosEntry * cosEntry) / ( 1.333f  *  1.333f ), 0.0f)),
                         0.6612f );
    float rs = (cosEntry -  1.333f  * cosWEnt) / (cosEntry +  1.333f  * cosWEnt);
    float rp = ( 1.333f  * cosEntry - cosWEnt) / ( 1.333f  * cosEntry + cosWEnt);
    float sunEnter = (cosEntry / cosWEnt) * (1.0f - 0.5f * (rs * rs + rp * rp));
    sunEnter = lerp(1.0f, sunEnter, saturate(gShadowParams.waterFogSun.x));

    float3 iSun = waterInscatterSeg(sigTm, kLgt, tauViewM, L, dNear, dFar, mEffSeg, slantSun, openEnded);
    float3 iAmb = waterInscatterSeg(sigTm, kLgt, tauViewM, L, dNear, dFar, mEffSeg,  1.2039f , openEnded);

    float ph = waterPhase(dot(worldPosRel / d, -gFrameData.sunDir.xyz));
#line 1416 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    ph = lerp(ph, 1.0f, wIso);
#line 1452 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    inscat = sigSm * ( 0.25f  * gFrameData.sunCol.rgb * ph * iSun * sunEnter
                    +  (0.5f * (1.0f - 0.6612f ) * 1.333f * 1.333f )  * gFrameData.ambCol.rgb * iAmb)
           * gShadowParams.waterFogScatter.w;
}








float3 waterFogBlend(float3 legacy, float3 behind, float3 worldPosRel, float2 wf)
{
    float volS = gShadowParams.waterFogExt.w;
    if (volS <= 0.0f) { return legacy; }


    float L = length(worldPosRel) * wf.y;
    if (L <= 0.0f) { return legacy; }

    float3 inscat, trans;
    waterColumn(worldPosRel, L, false, inscat, trans);
    return lerp(legacy, behind * trans + inscat, volS);
}


float3 waterFogComposite(float3 behind, float3 worldPosRel, float2 wf)
{
    return waterFogBlend(lerp(behind, waterFogColor(worldPosRel), wf.x), behind, worldPosRel, wf);
}
#line 1496 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
float3 waterFogVeil(float3 worldPosRel)
{
    float3 legacy = waterFogColor(worldPosRel);
    float volS = gShadowParams.waterFogExt.w;
    if (volS <= 0.0f) { return legacy; }

    float3 inscat, trans;
    waterColumn(worldPosRel, 0.0f, true, inscat, trans);
    return lerp(legacy, inscat, volS);
}



float waterFogVolStrength()
{
    return gShadowParams.waterFogExt.w;
}
#line 38 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"

STRUCT(VSOutput)
{
    DATA(float4, Position, SV_Position);
    DATA(float2, Uv, TEXCOORD0);
    DATA(float4, Color, COLOR);
    DATA(FLAT(uint), TexIndex, TEXCOORD1);
    DATA(FLAT(float3),MatDiffuse, TEXCOORD2);
    DATA(FLAT(uint), VColSource, TEXCOORD3);
    DATA(FLAT(float), MatAlpha, TEXCOORD4);
    DATA(float3, WorldDir, TEXCOORD5);
    DATA(FLAT(uint), SkyClass, TEXCOORD6);
#line 50
};
#line 64 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
[RootSignature( "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "3" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "3" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "3" ", offset = 0))," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "2" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "2" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "2" ", offset = 0))," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "1" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "1" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "1" ", offset = 0))," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "0" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "0" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "0" ", offset = 0))," "DescriptorTable(" "SAMPLER(s0, numDescriptors = unbounded, space = " "0" ", offset = 0))," "StaticSampler(s0, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s1, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s2, space = 100," "filter = FILTER_MIN_MAG_LINEAR_MIP_POINT," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s3, space = 100," "filter = FILTER_MIN_MAG_LINEAR_MIP_POINT," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s4, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s5, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s6, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT," "addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR, addressW = TEXTURE_ADDRESS_MIRROR)," "StaticSampler(s7, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT, borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK," "addressU = TEXTURE_ADDRESS_BORDER, addressV = TEXTURE_ADDRESS_BORDER, addressW = TEXTURE_ADDRESS_BORDER)," "StaticSampler(s8, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR," "addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR, addressW = TEXTURE_ADDRESS_MIRROR)," "StaticSampler(s9, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR, borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK," "addressU = TEXTURE_ADDRESS_BORDER, addressV = TEXTURE_ADDRESS_BORDER, addressW = TEXTURE_ADDRESS_BORDER)," "StaticSampler(s10, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s11, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s12, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s13, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s14, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s15, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s16, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s17, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_WRAP)" )]
float4 PS_MAIN( VSOutput In ): SV_TARGET
{
    //INIT_MAIN;
    float4 tex = SampleTex2D(gTextures[In.TexIndex], gSamplerAnisotropic, In.Uv);







    float4 base;
    if (In.VColSource == 3u) {







        float3 dir = normalize(In.WorldDir);
        float t = saturate(dir.z);
        base = float4(lerp(gFrameData.fogColNear.rgb, gFrameData.skyZenith.rgb, t), 1.0f);
    } else {
        base = (In.VColSource != 0u) ? In.Color : float4(In.MatDiffuse, 1.0f);
    }
    float4 c = tex * base;
    c.a *= In.MatAlpha;




    float amt = gFrameData.skyParams.x;
    if (gFrameData.skyParams.z > 0.5f) { amt *= 0.5f + 0.5f * sin(gFrameData.skyParams.y * 3.0f); }
    c.rgb = lerp(c.rgb, float3(1.0f, 0.0f, 1.0f), amt);
#line 122 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
    const uint cls = In.SkyClass;

    if (cls ==  6u  && gShadowParams.toneParams.y > 0.5f) {
#line 149 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
        float2 mUv = atmosSkyViewUvFromDir(gAtmosParams, gSkyView.sunDirW.w,
                                           normalize(In.WorldDir));
        c.rgb = max(SampleLvlTex2D(gAtmosSkyView, gSamplerBilinearClamp, mUv, 0).rgb, f3(0.0f))
              * gSkyView.radiance.w;
    } else if (cls ==  4u  && gShadowParams.agxCurve.w > 0.5f) {
#line 188 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
        float E = max(gShadowParams.calParams.y, 1.0e-4f);
        c.rgb = agxInverseNeutral(c.rgb, gShadowParams.agxCurve, gShadowParams.agxCurveScale,
                                  gShadowParams.agxLookParams) / E;
    } else if (cls ==  3u  && gShadowParams.toneParams.y > 0.5f
               && dot(gSkyView.sunDisc.xyz, float3(1.0f, 1.0f, 1.0f)) > 0.0f) {
#line 223 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
        float3 emis = c.rgb * gSkyView.sunDisc.xyz;
        c.rgb = expandExposedEmissiveP(emis, tex.rgb, c.a, gSkyView.sunDisc.w);
    } else if (cls ==  5u  && gShadowParams.toneParams.y > 0.5f) {
#line 249 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
        c.rgb *= gSkyView.elemRadiance.x;
    } else if (cls ==  2u  && gShadowParams.toneParams.y > 0.5f) {
#line 267 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
        c.rgb *= gSkyView.elemRadiance.y;
    } else {







        c.rgb = liftInPass(c.rgb);
    }
#line 320 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/sky.frag.fsl"
    if (waterFogCameraSubmerged()) {
        float3 skyRel = normalize(In.WorldDir) * 1.0e5f;
        c.rgb = waterFogBlend(c.rgb, c.rgb, skyRel, waterFogSample(skyRel));
    }
    return (c);
}
#line 178 "FSL/shaders.list"
