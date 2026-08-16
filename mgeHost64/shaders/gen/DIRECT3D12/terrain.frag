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
#line 215 "FSL/shaders.list"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/terrain.frag.fsl"
#line 23 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/terrain.frag.fsl"
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


    float4 slotFlick[ 32 ];










    float4x4 sunViewProj[ 2 ];
#line 75 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 sunParams;


    float4 sunCascadeTexel;
#line 94 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 sunPcf0;






    float4 sunPcf1;







    float4 volFog0;






    float4 volFog1;




    float4 volFog2;






    float4 volFog3;
#line 151 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 volFog4;







    float4 screenAlloc;
#line 171 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 shAr;
    float4 shAg;
    float4 shAb;
#line 186 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 skyParams;










    float4 skyAOMap;
#line 219 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 sunOcc;
#line 235 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 skyAO2;
#line 284 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 toneParams;
#line 298 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterFogCol;
#line 317 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterFogPlane;
#line 334 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterFogLight;
#line 351 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 waterFogExt;








    float4 waterFogScatter;

    float4 waterFogPhase;









    float4 waterFogPhase2;





    float4 waterFogKd;
#line 399 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 calParams;
#line 400
};
#line 21 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
#line 58 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
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
#line 129 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
    float4 timeParams;






    float4 uvOffsets[8];






    float4 froxelDims;
    float4 froxelZ;
#line 158 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
    float4 alphaShadowParams;
#line 159
};

STRUCT(BatchData)
{
    float4x4 worlds[ 1024 ];
#line 164
};
#line 185 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
STRUCT(LightData)
{
    float4 lightParams;
    float4 lights[ 128  * 3];









    float4 froxelDimsNear;
    float4 froxelZNear;
#line 200
};

        CBUFFER(FrameData) gFrameData :  register(b0,space1);





        Tex2D(float4) gAO :  register(t1,space1);
#line 236 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
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
#line 368 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gSkyColor :  register(t54,space1);
#line 381 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gReflectMips :  register(t55,space1);
#line 404 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2DArray(float) gWaterSlopeVar :  register(t56,space1);
#line 421 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float) gReflectDepth :  register(t57,space1);









        Tex2D(float4) gRippleField :  register(t58,space1);






        Tex2D(float4) gWakeField :  register(t59,space1);
#line 461 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gPortalGate :  register(t60,space1);





        CBUFFER(LightData) gLights :  register(b0,space3);










        CBUFFER(LightData) gLightsNear :  register(b1,space3);
#line 493 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gTextures[ 880 ] :  register(t0,space0);



        Tex2DArray(float4) gStaticsArrays[ 128 ] :  register(t880,space0);



        Tex2DArray(float4) gFlipArrays[ 16 ] :  register(t1008,space0);
        CBUFFER(BatchData) gBatch :  register(b0,space2);
#line 24 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/terrain.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/msmrecv.h.fsl"
#line 45 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/msmrecv.h.fsl"
float4 UnpackMoments(float4 packedMoments)
{
    packedMoments -= float4(0.5f, 0.0f, 0.5f, 0.0f);

    float4x4 inv = make_f4x4_row_elems(-1.0f/3.0f, 0.0f, sqrt(3.0f), 0.0f,
                                        0.0f, 0.125f, 0.0f, 1.0f,
                                       -0.75f, 0.0f, 0.75f * sqrt(3.0f), 0.0f,
                                        0.0f, -1.125f, 0.0f, 1.0f);

    float4 unpackedMoments = mul(inv, packedMoments);
    return lerp(unpackedMoments, float4(0.0f, 0.63f, 0.0f, 0.63f),  6.0e-5f );
}

float3x3 msmInverse3x3(float3x3 A)
{
    float3x3 result;
    float detA = dot(A[0], cross(A[1], A[2]));
    float invDetA = 1.0f / detA;
    result[0] = invDetA * cross(A[1], A[2]);
    result[1] = invDetA * cross(A[2], A[0]);
    result[2] = invDetA * cross(A[0], A[1]);
    result = transpose(result);
    return result;
}


float ComputeMSMShadowIntensity(float4 b, float zf)
{

    float3x3 B = float3x3(float3(1.0f, b.x, b.y),
                          float3(b.x, b.y, b.z),
                          float3(b.y, b.z, b.w));

    float3 c = mul(msmInverse3x3(B), float3(1.0f, zf, zf*zf));


    float discriminant = sqrt(max(c.y * c.y - 4.0f * c.z * c.x, 0.0f));

    float z2 = (-c.y - discriminant) / (2.0f * c.z);
    float z3 = (-c.y + discriminant) / (2.0f * c.z);

    if (z3 < z2) { float temp = z2; z2 = z3; z3 = temp; }

    float case2 = step(z2, zf) * step(zf, z3);
    float case3 = step(z3, zf);

    float result2 = (zf * z3 - b.x * (zf + z3) + b.y) / ((z3 - z2) * (zf - z2));
    float result3 = 1.0f - (z2 * z3 - b.x * (z2 + z3) + b.y) / ((zf - z2) * (zf - z3));

    return saturate(case2 * result2 + case3 * result3);
}






float sunCascadeOcclusion(int c, float3 p, float zBias)
{
    float4 posLS = mul(gShadowParams.sunViewProj[c], float4(p, 1.0f));


    posLS.xyz /= posLS.w;

    float2 uvTile = saturate(posLS.xy * float2(0.5f, -0.5f) + float2(0.5f, 0.5f));
    float2 uv = float2((float(c) + uvTile.x) * (1.0f / float( 2 )), uvTile.y);
    float zf = (1.0f - posLS.z) - zBias;

    float4 moments = UnpackMoments(SampleLvlTex2D(gSunMoments, gSamplerBilinearClamp, uv, 0));
    return ComputeMSMShadowIntensity(moments, zf);
}
#line 148 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/msmrecv.h.fsl"
float2 sunDiscTap(int i, int n, float rot)
{
    float r = sqrt((float(i) + 0.5f) / float(n));
    float th = float(i) * 2.39996323f + rot;
    return float2(r * cos(th), r * sin(th));
}
#line 170 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/msmrecv.h.fsl"
float sunDiscRotation(float3 worldPosRel)
{
    float2 h = float2(dot(worldPosRel, float3(0.7391f, 0.3179f, 0.5107f)),
                      dot(worldPosRel, float3(0.2311f, 0.6203f, 0.1157f)));
    return frac(sin(dot(h, float2(12.9898f, 78.233f))) * 43758.5453f) * 6.2831853f;
}







float sunPcfTap(float2 uv, float ref)
{
    float4 g = GatherRedTex2D(gSunDepth, gSamplerPointClamp, uv);

    float4 occ = step(ref, g);

    float2 t = frac(uv *  float2(float( 2048 * 2 ), float( 2048 ))  - 0.5f);
    return lerp(lerp(occ.w, occ.z, t.x), lerp(occ.x, occ.y, t.x), t.y);
}


float sunPcssOcclusion(int c, float3 p, float3 worldPosRel, float slopeBias)
{
    float4 posLS = mul(gShadowParams.sunViewProj[c], float4(p, 1.0f));
    posLS.xyz /= posLS.w;

    float2 uvTile = saturate(posLS.xy * float2(0.5f, -0.5f) + float2(0.5f, 0.5f));
    float2 uvC = float2((float(c) + uvTile.x) * (1.0f / float( 2 )), uvTile.y);

    float ref = posLS.z + gShadowParams.sunPcf1.x + slopeBias;
    float rot = sunDiscRotation(worldPosRel);
    float srch = gShadowParams.sunPcf0.x;




    float2 lo = float2((float(c) ) * (1.0f / float( 2 )), 0.0f) +  float2(1.0f / float( 2048 * 2 ), 1.0f / float( 2048 )) ;
    float2 hi = float2((float(c) + 1.0f) * (1.0f / float( 2 )), 1.0f) -  float2(1.0f / float( 2048 * 2 ), 1.0f / float( 2048 )) ;


    float blockerSum = 0.0f;
    float blockerCnt = 0.0f;
    UNROLL for (int i = 0; i <  12 ; ++i)
    {
        float2 sp = clamp(uvC + sunDiscTap(i,  12 , rot) * srch *  float2(1.0f / float( 2048 * 2 ), 1.0f / float( 2048 )) , lo, hi);
        float d = SampleLvlTex2D(gSunDepth, gSamplerPointClamp, sp, 0).r;
        float hit = step(ref, d);
        blockerSum += d * hit;
        blockerCnt += hit;
    }

    if (blockerCnt < 0.5f) { return 0.0f; }






    float dzNorm = max(blockerSum / blockerCnt - posLS.z, 0.0f);
    float texelW = max(gShadowParams.sunCascadeTexel[c], 1.0e-3f);
    float radius = clamp(gShadowParams.sunPcf0.y * dzNorm / texelW,
                          gShadowParams.sunPcf0.z, gShadowParams.sunPcf0.w);


    float occlusion = 0.0f;
    UNROLL for (int j = 0; j <  16 ; ++j)
    {
        float2 sp = clamp(uvC + sunDiscTap(j,  16 , rot) * radius *  float2(1.0f / float( 2048 * 2 ), 1.0f / float( 2048 )) , lo, hi);
        occlusion += sunPcfTap(sp, ref);
    }
    return occlusion * (1.0f / float( 16 ));
}
#line 261 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/msmrecv.h.fsl"
float sunOccMapOcclusion(float3 worldAbs)
{
    float4 so = gShadowParams.sunOcc;
    if (so.x <= 0.0f) { return 0.0f; }


    float4 m = gShadowParams.skyAOMap;
    float2 uv = (worldAbs.xy - m.xy) * m.z;



    float2 e = abs(uv - 0.5f) * 2.0f;
    float edge = saturate((1.0f - max(e.x, e.y)) * 8.0f);
    if (edge <= 0.0f) { return 0.0f; }

    float blockZ = SampleLvlTex2D(gSunOcc, gSamplerBilinearClamp, uv, 0).r;




    float t = saturate(((blockZ - so.z) - worldAbs.z) / max(so.y, 1.0e-3f));
    t = t * t * (3.0f - 2.0f * t);
    return t * so.x * edge;
}
#line 315 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/msmrecv.h.fsl"
float sunShadowVolumetric(float3 p)
{
    if (gShadowParams.sunParams.x <= 0.0f) { return 1.0f; }


    int sel = -1;
    float m = 0.0f;
    UNROLL for (int i = 0; i <  2 ; ++i)
    {
        if (sel < 0)
        {
            float4 q = mul(gShadowParams.sunViewProj[i], float4(p, 1.0f));
            float mm = max(abs(q.x), abs(q.y));
            if (mm < 1.0f -  0.008f  && q.z > 0.0f && q.z < 1.0f) { sel = i; m = mm; }
        }
    }

    float mapOcc = sunOccMapOcclusion(p + gFrameData.lodEye.xyz);


    if (sel < 0) { return 1.0f - mapOcc; }

    float occ = sunCascadeOcclusion(sel, p, gShadowParams.sunParams.y);







    float band = 1.0f -  0.008f  -  0.12f ;
    if (m > band && sel + 1 >=  2 )
    {
        occ = lerp(occ, mapOcc, saturate((m - band) /  0.12f ));
    }
    return 1.0f - occ;
}








float sunCascadeShadow(int c, float3 p, float3 worldPosRel, float zBias, float slopeBias)
{
    if (gShadowParams.sunPcf1.z > 0.5f) { return sunPcssOcclusion(c, p, worldPosRel, slopeBias); }
    return sunCascadeOcclusion(c, p, zBias);
}
#line 376 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/msmrecv.h.fsl"
float sunShadowVisibility(float3 worldPosRel, float3 N)
{
    float strength = gShadowParams.sunParams.x;
    if (strength <= 0.0f) { return 1.0f; }





    float mapOcc = sunOccMapOcclusion(worldPosRel + gFrameData.lodEye.xyz);




    int sel = -1;
    float m = 0.0f;
    UNROLL for (int i = 0; i <  2 ; ++i)
    {
        if (sel < 0)
        {
            float4 q = mul(gShadowParams.sunViewProj[i], float4(worldPosRel, 1.0f));
            float mm = max(abs(q.x), abs(q.y));


            if (mm < 1.0f -  0.008f  && q.z > 0.0f && q.z < 1.0f) { sel = i; m = mm; }
        }
    }


    if (sel < 0) { return 1.0f - mapOcc * strength; }

    float bias = gShadowParams.sunParams.y;
    float noff = gShadowParams.sunParams.w;
    float4 texel = gShadowParams.sunCascadeTexel;






    float NdotL = saturate(dot(N, -gFrameData.sunDir.xyz));
    float slopeBias = gShadowParams.sunPcf1.y * min(sqrt(1.0f - NdotL * NdotL) / max(NdotL, 0.1f), 10.0f);

    float occlusion = sunCascadeShadow(sel, worldPosRel + N * (noff * texel[sel]), worldPosRel,
                                       bias, slopeBias);






    float band = 1.0f -  0.008f  -  0.12f ;
    float t = (m > band) ? saturate((m - band) /  0.12f ) : 0.0f;
    bool toMap = (sel + 1 >=  2 );
    if (t > 0.0f && !toMap)
    {
        float occNext = sunCascadeShadow(sel + 1, worldPosRel + N * (noff * texel[sel + 1]),
                                         worldPosRel, bias, slopeBias);
        occlusion = lerp(occlusion, occNext, t);
    }
#line 450 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/msmrecv.h.fsl"
    float lbr = gShadowParams.sunParams.z;
    occlusion = saturate(occlusion / max(1.0f - lbr, 1.0e-4f));

    if (t > 0.0f && toMap) { occlusion = lerp(occlusion, mapOcc, t); }

    return 1.0f - occlusion * strength;
}
#line 25 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/terrain.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowreceive.h.fsl"
#line 47 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowreceive.h.fsl"
int2 shadowMaskPixel(float4 svPos)
{
    int2 px = int2(svPos.xy);
    float dzx = ddx(svPos.z);
    float dzy = ddy(svPos.z);





    float tol = 0.5f * (abs(dzx) + abs(dzy)) + 1e-6f;
    if (abs(LoadTex2D(gSceneLinDepth, NO_SAMPLER, px, 0).r - svPos.z) <= tol) { return px; }






    float ntol = 1.5f * (abs(dzx) + abs(dzy)) + 1e-6f;
    UNROLL
    for (int i = 0; i < 4; ++i)
    {
        int2 o = (i == 0) ? int2(-1, 0)
               : (i == 1) ? int2( 1, 0)
               : (i == 2) ? int2( 0,-1)
                          : int2( 0, 1);
        int2 q = px + o;
        float zq = svPos.z + dzx * float(o.x) + dzy * float(o.y);
        if (abs(LoadTex2D(gSceneLinDepth, NO_SAMPLER, q, 0).r - zq) <= ntol) { return q; }
    }
    return px;
}
#line 26 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/terrain.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skyamb.h.fsl"
#line 83 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skyamb.h.fsl"
float skyAOVisibility(float3 worldAbs)
{
    float4 m = gShadowParams.skyAOMap;





    float2 uv = (worldAbs.xy - m.xy) * m.z;
    float2 e = abs(uv - 0.5f) * 2.0f;
    float edge = saturate((1.0f - max(e.x, e.y)) * 8.0f);
    if (edge <= 0.0f) { return 1.0f; }

    float inner = gShadowParams.skyParams.z;
    float outer = gShadowParams.skyParams.w;
    int steps = (int)max(m.w, 1.0f);




    float logRatio = log2(max(outer / max(inner, 1.0f), 1.0f));
    float myH = worldAbs.z;
#line 137 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skyamb.h.fsl"
    float selfH = SampleLvlTex2D(gSkyHeight, gSamplerPointClamp, uv, 0).r;
    float above = selfH - myH;
    float trust = 1.0f - (1.0f - gShadowParams.skyAO2.z)
                       * saturate((above - gShadowParams.skyAO2.x) / max(gShadowParams.skyAO2.y, 1.0f));





    float4 maxS = f4(0.0f);
    float maxS4 = 0.0f;

    for (int k = 0; k < steps; ++k)
    {
        float t = ((float)k + 1.0f) / (float)steps;
        float d = inner * exp2(t * logRatio);

        float r = d * m.z;
        float2 o0 = float2( 0.98769f, 0.15643f) * r;
        float2 o1 = float2( 0.15643f, 0.98769f) * r;
        float2 o2 = float2(-0.89101f, 0.45399f) * r;
        float2 o3 = float2(-0.70711f, -0.70711f) * r;
        float2 o4 = float2( 0.45399f, -0.89101f) * r;



        float4 h;
        h.x = SampleLvlTex2D(gSkyHeight, gSamplerBilinearClamp, uv + o0, 0).r;
        h.y = SampleLvlTex2D(gSkyHeight, gSamplerBilinearClamp, uv + o1, 0).r;
        h.z = SampleLvlTex2D(gSkyHeight, gSamplerBilinearClamp, uv + o2, 0).r;
        h.w = SampleLvlTex2D(gSkyHeight, gSamplerBilinearClamp, uv + o3, 0).r;
        float h4 = SampleLvlTex2D(gSkyHeight, gSamplerBilinearClamp, uv + o4, 0).r;




        float4 dh = h - f4(myH);
        float dh4 = h4 - myH;
        maxS = max(maxS, dh * rsqrt(dh * dh + f4(d * d)));
        maxS4 = max(maxS4, dh4 * rsqrt(dh4 * dh4 + d * d));
    }

    float4 s = saturate(maxS);
    float s4 = saturate(maxS4);
    float4 vis = f4(1.0f) - s * s;
    float vis4 = 1.0f - s4 * s4;
    float ao = (dot(vis, f4(1.0f)) + vis4) * 0.2f;





    ao = lerp(1.0f, ao, trust);


    return lerp(1.0f, ao, gShadowParams.skyParams.y * edge);
}





float3 skyAmbFactor(float3 N, float3 worldPosRel)
{





    float ao = (gShadowParams.skyParams.y > 0.0f) ? skyAOVisibility(worldPosRel + gFrameData.lodEye.xyz)
                                                  : 1.0f;


    float s = gShadowParams.skyParams.x;
    if (s <= 0.0f) { return float3(ao, ao, ao); }

    float4 n = float4(normalize(N), 1.0f);
    float3 f = float3(dot(gShadowParams.shAr, n),
                      dot(gShadowParams.shAg, n),
                      dot(gShadowParams.shAb, n));








    return lerp(float3(1.0f, 1.0f, 1.0f), max(f, float3(0.0f, 0.0f, 0.0f)), s) * ao;
}
#line 27 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/terrain.frag.fsl"
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
#line 28 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/terrain.frag.fsl"
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
#line 29 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/terrain.frag.fsl"
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
#line 30 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/terrain.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
#line 54 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
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
float waterFogDepthBelow(float3 worldPosRel)
{
    return max(waterFogPlaneRelZ() - worldPosRel.z, 0.0f);
}


float waterFogEyeDepth()
{
    return max(waterFogPlaneRelZ(), 0.0f);
}
#line 120 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
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
#line 160 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
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
#line 203 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
float3 waterFogColor(float3 worldPosRel)
{
    float3 c = gShadowParams.waterFogCol.rgb;

    float s = gShadowParams.waterFogLight.w;
    if (s <= 0.0f) { return c; }

    float depthRep = 0.5f * (waterFogEyeDepth() + waterFogDepthBelow(worldPosRel));
    float3 t = exp(-gShadowParams.waterFogLight.rgb * depthRep *  1.2039f );
    return c * lerp(float3(1.0f, 1.0f, 1.0f), t, s);
}
#line 247 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
void waterLightTransmit(float3 worldPosRel, out float3 tSun, out float3 tAmb)
{
    tSun = float3(1.0f, 1.0f, 1.0f);
    tAmb = tSun;

    float strength = gShadowParams.waterFogLight.w;
    if (!waterFogActive() || strength <= 0.0f) { return; }

    float depth = waterFogDepthBelow(worldPosRel);
    if (depth <= 0.0f) { return; }
#line 270 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float3 k = lerp(gShadowParams.waterFogLight.rgb, gShadowParams.waterFogKd.rgb,
                    gShadowParams.waterFogExt.w);



    float cosAir = abs(gFrameData.sunDir.z);
    float cosWater = sqrt(max(1.0f - (1.0f - cosAir * cosAir) / ( 1.333f  *  1.333f ), 0.0f));
    float pathSun = depth / max(cosWater,  0.6612f );

    tSun = lerp(tSun, exp(-k * pathSun), strength);
    tAmb = lerp(tAmb, exp(-k * depth *  1.2039f ), strength);
}
#line 346 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
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
#line 401 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
void waterColumn(float3 worldPosRel, float L, bool openEnded, out float3 inscat, out float3 trans)
{
    float3 sigT = gShadowParams.waterFogExt.rgb;
    float3 sigS = gShadowParams.waterFogScatter.rgb;



    float3 kLgt = gShadowParams.waterFogKd.rgb;

    float d = max(length(worldPosRel), 1.0e-4f);




    float mEff = -worldPosRel.z / d;
    float dNear = waterFogEyeDepth();
    float dFar = max(dNear + mEff * L, 0.0f);

    float3 tauView = sigT * L;
    trans = openEnded ? float3(0.0f, 0.0f, 0.0f) : exp(-tauView);





    float cosAir = abs(gFrameData.sunDir.z);
    float cosWater = sqrt(max(1.0f - (1.0f - cosAir * cosAir) / ( 1.333f  *  1.333f ), 0.0f));
    float slantSun = 1.0f / max(cosWater,  0.6612f );

    float3 iSun = waterInscatterSeg(sigT, kLgt, tauView, L, dNear, dFar, mEff, slantSun, openEnded);
    float3 iAmb = waterInscatterSeg(sigT, kLgt, tauView, L, dNear, dFar, mEff,  1.2039f , openEnded);

    float ph = waterPhase(dot(worldPosRel / d, -gFrameData.sunDir.xyz));
#line 461 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float tauS = ((sigS.r + sigS.g + sigS.b) * (1.0f / 3.0f)) * L;
    float wSingle = openEnded ? 0.0f : exp(-tauS);
    ph = lerp(ph, 1.0f, saturate(gShadowParams.waterFogPhase2.z) * (1.0f - wSingle));
#line 493 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    inscat = sigS * ( 0.25f  * gFrameData.sunCol.rgb * ph * iSun
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
#line 537 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
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
#line 55 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
#line 78 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterplane.h.fsl"
#line 79 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
#line 80 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
#line 102 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
float mwFogRamp(float dist)
{
    float fogStart = gFrameData.fogParams.x;
    float fogEnd = gFrameData.fogParams.y;
    float S = gShadowParams.toneParams.z;






    float span = max(fogEnd - fogStart, 1.0f);
    float t = saturate((dist - fogStart) / span);

    if (S > 0.0f)
    {
#line 141 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
        float x = S * t * t;
        float e = exp(-S);
        return saturate((exp(-x) - e) / (1.0f - e));
    }


    return 1.0f - t;
}
#line 173 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
float mwFogFactor(float dist)
{
    if (waterCameraSubmerged()) { return 1.0f; }
    return mwFogRamp(dist);
}
#line 236 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
float mwFogAirShare(float fog, float waterShare, float dist)
{

    if (waterShare <= 0.0f) { return fog; }

    float oTot = 1.0f - mwFogRamp(dist);
    if (oTot <= 0.0f) { return 1.0f; }

    float oAir = 1.0f - mwFogRamp(dist * (1.0f - waterShare));
    return 1.0f - (1.0f - fog) * saturate(oAir / oTot);
}



float mwFogAirShareAt(float fog, float3 worldPosRel)
{
    return mwFogAirShare(fog, waterFogSample(worldPosRel).y, length(worldPosRel));
}
#line 56 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
#line 84 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
float4 fogSkySample(float2 pixelXy)
{
    return SampleLvlTex2D(gSkyColor, gSamplerBilinearClamp, pixelXy * gFrameData.fogParams.zw, 0);
}







float3 fogSkyTarget(float4 s, float skyShare)
{
    float3 sky = s.rgb + gFrameData.fogColNear.rgb * (1.0f - s.a);
    return lerp(gFrameData.fogColNear.rgb, sky, gFrameData.fogColNear.w * skyShare);
}
#line 131 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
float fogSkyShare(float fogAir)
{
    return smoothstep( 0.85f , 1.0f, 1.0f - fogAir);
}




float3 fogSkyColor(float2 pixelXy)
{
    return fogSkyTarget(fogSkySample(pixelXy), 1.0f);
}



float3 fogSkyColorAt(float2 pixelXy, float fogAir)
{
    return fogSkyTarget(fogSkySample(pixelXy), fogSkyShare(fogAir));
}
#line 174 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
float fogExtinction(float fogAir, float skyBehind)
{
    float ramp = pow(1.0f - fogAir, gFrameData.timeParams.w);
    return 1.0f - gFrameData.skyParams.w * ramp * skyBehind;
}
#line 191 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
float3 applyFog(float3 lit, float3 worldPosRel, float2 pixelXy, float fog)
{
    float4 s = fogSkySample(pixelXy);




    float2 wf = waterFogSample(worldPosRel);









    fog = saturate(fog);
#line 279 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
    float fogAir = mwFogAirShare(fog, wf.y, length(worldPosRel));
#line 305 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
    if (waterCameraSubmerged()) { fogAir = 1.0f; }
#line 339 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
    float upness = worldPosRel.z * rsqrt(max(dot(worldPosRel, worldPosRel), 1.0e-12f));
    float skyBehind = max(s.a, saturate(upness *  38.0f ));
#line 396 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
    float3 behind = waterFogComposite(lit, worldPosRel, wf);





    behind *= fogExtinction(fogAir, skyBehind);





    return lerp(fogSkyTarget(s, fogSkyShare(fogAir)), behind, fogAir);
}
#line 31 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/terrain.frag.fsl"






STRUCT(VSOutput)
{
    DATA(float4, Position, SV_Position);
    DATA(float3, WorldPos, TEXCOORD0);
    DATA(float3, Normal, TEXCOORD1);
    DATA(CENTROID(float), Fog, TEXCOORD2);
    DATA(float2, Lattice, TEXCOORD3);
    DATA(FLAT(uint4), Cell, TEXCOORD4);
#line 45
};


float3 loadVertexColor(uint slot, int x, int y)
{
    uint c = gTerrainColor[slot *  4225u  + (uint)(y *  65  + x)];
    return float3(float(c & 0xFFu), float((c >> 8u) & 0xFFu), float((c >> 16u) & 0xFFu))
         * (1.0f / 255.0f);
}

uint terrainCellAt(int lx, int ly, uint spanX, uint spanY)
{
    if (lx < 0 || ly < 0 || lx >= (int)spanX || ly >= (int)spanY) { return 0u; }
    return gTerrainCellGrid[(uint)ly * spanX + (uint)lx];
}






uint landTexSlot(int lx, int ly, uint spanX, uint spanY, int tx, int ty)
{
    if (tx < 0) { --lx; tx += 16; } else if (tx > 15) { ++lx; tx -= 16; }
    if (ty < 0) { --ly; ty += 16; } else if (ty > 15) { ++ly; ty -= 16; }
    uint cell = terrainCellAt(lx, ly, spanX, spanY);
    if (cell == 0u) { return 0u; }
    return gTerrainTex[(cell - 1u) *  256u  + (uint)(ty * 16 + tx)];
}



int squareX(int x) { return (int)floor((float(x) - 2.0f) * 0.25f); }
int squareY(int y) { return (int)ceil ((float(y) - 2.0f) * 0.25f); }




float3 sampleLand(uint slot, float2 uv)
{
    uint bucket = slot >> 16u;
    uint layer = slot & 0xFFFFu;
    return SampleTex2DArray(gTerrainArrays[bucket], gSamplerAnisotropic,
                            float3(uv, float(layer))).rgb;
}

[RootSignature( "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "3" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "3" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "3" ", offset = 0))," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "2" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "2" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "2" ", offset = 0))," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "1" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "1" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "1" ", offset = 0))," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "0" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "0" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "0" ", offset = 0))," "DescriptorTable(" "SAMPLER(s0, numDescriptors = unbounded, space = " "0" ", offset = 0))," "StaticSampler(s0, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s1, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s2, space = 100," "filter = FILTER_MIN_MAG_LINEAR_MIP_POINT," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s3, space = 100," "filter = FILTER_MIN_MAG_LINEAR_MIP_POINT," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s4, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s5, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s6, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT," "addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR, addressW = TEXTURE_ADDRESS_MIRROR)," "StaticSampler(s7, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT, borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK," "addressU = TEXTURE_ADDRESS_BORDER, addressV = TEXTURE_ADDRESS_BORDER, addressW = TEXTURE_ADDRESS_BORDER)," "StaticSampler(s8, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR," "addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR, addressW = TEXTURE_ADDRESS_MIRROR)," "StaticSampler(s9, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR, borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK," "addressU = TEXTURE_ADDRESS_BORDER, addressV = TEXTURE_ADDRESS_BORDER, addressW = TEXTURE_ADDRESS_BORDER)," "StaticSampler(s10, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s11, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s12, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s13, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s14, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s15, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s16, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s17, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_WRAP)" )]
float4 PS_MAIN( VSOutput In ): SV_TARGET
{
    //INIT_MAIN;

    const int lx = (int)In.Cell.y;
    const int ly = (int)In.Cell.z;
    const uint spanX = In.Cell.w & 0xFFFFu;
    const uint spanY = In.Cell.w >> 16u;



    float2 lat = clamp(In.Lattice, float2(0.0f, 0.0f), float2(float( 64 ), float( 64 )));
    int x0 = (int)floor(lat.x), y0 = (int)floor(lat.y);
    int x1 = min(x0 + 1,  64 ), y1 = min(y0 + 1,  64 );
    float fx = lat.x - float(x0), fy = lat.y - float(y0);

    uint id00 = landTexSlot(lx, ly, spanX, spanY, squareX(x0), squareY(y0));
    uint id10 = landTexSlot(lx, ly, spanX, spanY, squareX(x1), squareY(y0));
    uint id01 = landTexSlot(lx, ly, spanX, spanY, squareX(x0), squareY(y1));
    uint id11 = landTexSlot(lx, ly, spanX, spanY, squareX(x1), squareY(y1));


    float2 uv = lat * 0.25f;

    float3 albedo;
    if (id00 == id10 && id00 == id01 && id00 == id11) {
        albedo = sampleLand(id00, uv);
    } else {
        float w00 = (1.0f - fx) * (1.0f - fy);
        float w10 = fx * (1.0f - fy);
        float w01 = (1.0f - fx) * fy;
        float w11 = fx * fy;
        albedo = sampleLand(id00, uv) * w00;
        albedo += sampleLand(id10, uv) * w10;
        albedo += sampleLand(id01, uv) * w01;
        albedo += sampleLand(id11, uv) * w11;
    }
#line 144 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/terrain.frag.fsl"
    {
        const uint cslot = In.Cell.x;
        float3 c00 = loadVertexColor(cslot, x0, y0);
        float3 c10 = loadVertexColor(cslot, x1, y0);
        float3 c01 = loadVertexColor(cslot, x0, y1);
        float3 c11 = loadVertexColor(cslot, x1, y1);
        albedo *= lerp(lerp(c00, c10, fx), lerp(c01, c11, fx), fy);
    }

    float3 normal = normalize(In.Normal);
    float sunVis = sunShadowVisibility(In.WorldPos, normal);
#line 186 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/terrain.frag.fsl"
    bool inReflect = (gFrameData.gReflWaterClip.z != 0.0f);
    uint aoFlags = (uint)(gFrameData.debugParams.w + 0.5f);






    uint dbgMode = (uint)(gFrameData.debugParams.x + 0.5f);
    bool aoUsed = ((aoFlags & 3u) != 0u) || dbgMode == 3u || dbgMode == 4u;
    int2 srcPx = int2(In.Position.xy);
    if (!inReflect && (aoUsed || gShadowParams.slotBits.x != 0u)) {
        srcPx = shadowMaskPixel(In.Position);
    }
    float4 aoSample = inReflect ? float4(0.0f, 0.0f, 0.0f, 1.0f)
                                 : LoadTex2D(gAO, NO_SAMPLER, srcPx, 0);



    float3 amb = gFrameData.lodSunAmb.rgb * skyAmbFactor(normal, In.WorldPos);
    if ((aoFlags & 1u) != 0u) { amb *= aoSample.a; }



    float3 tSunW, tAmbW;
    waterLightTransmit(In.WorldPos, tSunW, tAmbW);
    float3 result = albedo
                  * (gFrameData.sunCol.rgb * saturate(dot(-gFrameData.sunDir.xyz, normal)) * sunVis
                        * tSunW
                     + amb * tAmbW);
#line 238 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/terrain.frag.fsl"
    float3 pointDiffuse = float3(0.0f, 0.0f, 0.0f);






    int2 maskPx = srcPx;
    if (length(In.WorldPos) < gFrameData.lodParams.x)
    {


        uint nL = (uint)gLightsNear.lightParams.x;
        float reachK = gLightsNear.lightParams.y;
        int tilesX = (int)gLightsNear.froxelDimsNear.x;



        bool clustered = (tilesX > 0) && !inReflect;
        uint base = 0u;
        if (clustered)
        {
            int tilesY = (int)gLightsNear.froxelDimsNear.y;
            int NZ = (int)gLightsNear.froxelDimsNear.z;
            float tile = gLightsNear.froxelDimsNear.w;
            int tx = clamp((int)(In.Position.x / tile), 0, tilesX - 1);
            int ty = clamp((int)(In.Position.y / tile), 0, tilesY - 1);
            float dd = length(In.WorldPos);
            int zs = clamp((int)((log(max(dd, 1.0f)) - gLightsNear.froxelZNear.x) * gLightsNear.froxelZNear.y * (float)NZ), 0, NZ - 1);
            base = (((uint)ty * (uint)tilesX + (uint)tx) * (uint)NZ + (uint)zs) * 4u;
        }
        for (uint wi = 0u; wi < 4u; ++wi)
        {
            uint bits;
            if (clustered) {
                bits = gFroxelMaskNear[base + wi];
            } else {
                uint lo = wi * 32u;
                uint cnt = (lo >= nL) ? 0u : min(32u, nL - lo);
                bits = (cnt >= 32u) ? 0xFFFFFFFFu : ((1u << cnt) - 1u);
            }
            while (bits != 0u)
            {
                uint i = wi * 32u + firstbitlow(bits);
                bits = bits & (bits - 1u);
                if (i >= nL) { continue; }
                float4 posR = gLightsNear.lights[i * 3u + 0u];
                float3 lcol = gLightsNear.lights[i * 3u + 1u].rgb;
                float4 fo = gLightsNear.lights[i * 3u + 2u];
                float reach = posR.w * reachK;
                float3 toL = posR.xyz - In.WorldPos;
                float d2 = dot(toL, toL);
                if (d2 >= reach * reach) { continue; }
                float invD = rsqrt(max(d2, 1e-8f));
                float d = d2 * invD;
                float att = 1.0f / max(fo.z * d2 + fo.y * d + fo.x, 1e-4f);
                att *= 1.0f - smoothstep( 0.75f  * reach, reach, d);






                uint slotP1 = (uint)fo.w;
                if (slotP1 != 0u && !inReflect)
                {


                    uint4 mw = LoadTex2D(gShadowMask, NO_SAMPLER, maskPx, 0).xyzw;
                    uint s = slotP1 - 1u;
                    uint lane = s >> 3u;
                    uint word = lane == 0u ? mw.x : (lane == 1u ? mw.y : (lane == 2u ? mw.z : mw.w));
                    uint nib = (word >> ((s & 7u) * 4u)) & 0xFu;
                    att *= float(nib) * (1.0f / 15.0f);
                }
                float lambert = saturate(dot(normal, toL) * invD);
                pointDiffuse += lambert * att * lcol;
            }
        }
    }
    else
    {


        uint nL = (uint)gLights.lightParams.x;
        float reachK = gLights.lightParams.y;
        int tilesX = (int)gFrameData.froxelDims.x;
        bool clustered = (tilesX > 0);
        uint base = 0u;
        if (clustered)
        {
            int tilesY = (int)gFrameData.froxelDims.y;
            int NZ = (int)gFrameData.froxelDims.z;
            float tile = gFrameData.froxelDims.w;
            int tx = clamp((int)(In.Position.x / tile), 0, tilesX - 1);
            int ty = clamp((int)(In.Position.y / tile), 0, tilesY - 1);
            float dd = length(In.WorldPos);
            int zs = clamp((int)((log(max(dd, 1.0f)) - gFrameData.froxelZ.x) * gFrameData.froxelZ.y * (float)NZ), 0, NZ - 1);
            base = (((uint)ty * (uint)tilesX + (uint)tx) * (uint)NZ + (uint)zs) * 4u;
        }
        for (uint wi = 0u; wi < 4u; ++wi)
        {
            uint bits;
            if (clustered) {
                bits = gFroxelMask[base + wi];
            } else {
                uint lo = wi * 32u;
                uint cnt = (lo >= nL) ? 0u : min(32u, nL - lo);
                bits = (cnt >= 32u) ? 0xFFFFFFFFu : ((1u << cnt) - 1u);
            }
            while (bits != 0u)
            {
                uint i = wi * 32u + firstbitlow(bits);
                bits = bits & (bits - 1u);
                if (i >= nL) { continue; }
                float4 posR = gLights.lights[i * 3u + 0u];
                float3 lcol = gLights.lights[i * 3u + 1u].rgb;
                float3 fo = gLights.lights[i * 3u + 2u].xyz;
                float reach = posR.w * reachK;
                float3 toL = posR.xyz - In.WorldPos;
                float d2 = dot(toL, toL);
                if (d2 >= reach * reach) { continue; }
                float invD = rsqrt(max(d2, 1e-8f));
                float d = d2 * invD;
                float att = 1.0f / max(fo.z * d2 + fo.y * d + fo.x, 1e-4f);
                att *= 1.0f - smoothstep( 0.75f  * reach, reach, d);
                float lambert = saturate(dot(normal, toL) * invD);
                pointDiffuse += lambert * att * lcol;
            }
        }
    }
    result += albedo * pointDiffuse;

    result = tonemapInPass(result);
    result = applyFog(result, In.WorldPos, In.Position.xy, In.Fog);


    uint dbg = (uint)(gFrameData.debugParams.x + 0.5f);





    if (dbg == 3u || dbg == 4u) {





        if (dbg == 4u) { RETURN(float4(aoSample.rgb * 0.5f + 0.5f, 1.0f)); }
        float v = aoSample.a; RETURN(float4(v, v, v, 1.0f));
    }
    if (dbg == 1u || dbg == 2u) {
        float dist = length(In.WorldPos - gFrameData.eyePos.xyz);
        float g = saturate(dist * (1.0f / 8192.0f));
        return (float4(g, g, g, 1.0f));
    }
    return (float4(result, 1.0f));
}
#line 216 "FSL/shaders.list"
