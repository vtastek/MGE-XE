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
#line 196 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"

SamplerState gSamplerPointClamp : register( s0 , space100 ) ;
#line 199 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerPointWrap : register( s1 , space100 ) ;
#line 201 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerBilinearClamp : register( s2 , space100 ) ;
#line 203 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerBilinearWrap : register( s3 , space100 ) ;
#line 205 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerTrilinearClamp : register( s4 , space100 ) ;
#line 207 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerTrilinearWrap : register( s5 , space100 ) ;
#line 209 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerPointMirror : register( s6 , space100 ) ;
#line 211 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerPointBorder : register( s7 , space100 ) ;
#line 213 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerTrilinearMirror : register( s8 , space100 ) ;
#line 215 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerTrilinearBorder : register( s9 , space100 ) ;
#line 217 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerAnisotropic : register( s10 , space100 ) ;
#line 237 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerAnisoClampClamp : register( s11 , space100 ) ;
#line 239 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerAnisoClampWrap : register( s12 , space100 ) ;
#line 241 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSamplerAnisoWrapClamp : register( s13 , space100 ) ;
#line 249 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSampler2xWrapWrap : register( s14 , space100 ) ;
#line 251 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSampler2xClampClamp : register( s15 , space100 ) ;
#line 253 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSampler2xClampWrap : register( s16 , space100 ) ;
#line 255 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"
SamplerState gSampler2xWrapClamp : register( s17 , space100 ) ;
#line 258 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/../../../3rdparty/The-Forge/Common_3/Graphics/FSL/defaults.h"

#line 11 "FSL/shaders.list"
#line 408 "FSL/shaders.list"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
#line 60 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
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
#line 792 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 pbrParams;
#line 820 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 pbrTerrain;
#line 832 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 pbrTerrainAO;
#line 847 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 pbrStatics;
#line 860 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 terrainTex;
#line 884 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 parallax;
#line 898 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 parallax2;
#line 915 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 parallax3;
#line 932 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 terrainDisp;
#line 945 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 terrainDisp2;





    float4 terrainDisp3;
#line 974 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 terrainMacro;
#line 986 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 terrainMacro2;








    float4 terrainMacro3;
#line 1015 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 pbrShade;









    float4 pbrShade2;
#line 1037 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/shadowparams.h.fsl"
    float4 sunny16;







    float4 skyVis;










    float4 skyAOFloor;






    float4 sunNoise;


    float4 weatherState;


    float4 waterFlowMap;
    float4 waterBodyAmp;

    float4 waterFlowParams;

    float4 weatherState2;

    float4 skyAOGround;
#line 1077
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
#line 112 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skyview.h.fsl"
    float4 elemRadiance;
#line 113
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
#line 58 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosparams.h.fsl"
    float4 cloud;
#line 75 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosparams.h.fsl"
    float4 deckMsA;
    float4 deckMsB;
#line 107 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosparams.h.fsl"
    float4 deckMix;
#line 108
};
#line 36 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
#line 78 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
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
#line 155 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
    float4 timeParams;






    float4 uvOffsets[8];






    float4 froxelDims;
#line 191 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
    float4 froxelZ;
#line 205 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
    float4 alphaShadowParams;
#line 206
};

STRUCT(BatchData)
{
    float4x4 worlds[ 1024 ];
#line 211
};
#line 232 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
STRUCT(LightData)
{
    float4 lightParams;
    float4 lights[ 128  * 3];









    float4 froxelDimsNear;
    float4 froxelZNear;




    float4 goboNear[ 128  / 2];
#line 252
};

        CBUFFER(FrameData) gFrameData :  register(b0,space1);





        Tex2D(float4) gAO :  register(t1,space1);
#line 288 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
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
#line 420 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gSkyColor :  register(t54,space1);
#line 433 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gReflectMips :  register(t55,space1);
#line 456 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2DArray(float) gWaterSlopeVar :  register(t56,space1);
#line 473 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float) gReflectDepth :  register(t57,space1);









        Tex2D(float4) gRippleField :  register(t58,space1);






        Tex2D(float4) gWakeField :  register(t59,space1);
#line 513 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gPortalGate :  register(t60,space1);
#line 534 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2DArray(float) gCausticField :  register(t61,space1);
#line 548 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gGrassCrush :  register(t62,space1);
#line 561 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gAtmosSkyView :  register(t63,space1);





        Tex2D(float4) gAtmosTransmittance :  register(t64,space1);






        CBUFFER(AtmosphereParams) gAtmosParams :  register(b65,space1);
#line 593 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float2) gMotionVectors :  register(t66,space1);





        Tex2D(float) gReactiveMask :  register(t67,space1);








        Tex2D(float4) gAtmosSkyViewClear :  register(t68,space1);
#line 635 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Buffer(uint) gTerrainParamTex :  register(t69,space1);
        Tex2DArray(float4) gTerrainParamArrays[ 32 ] :  register(t70,space1);
#line 665 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Buffer(uint) gStaticsParamSlot :  register(t102,space1);
        Tex2DArray(float4) gStaticsParamArrays[ 48 ] :  register(t103,space1);





        Tex2DArray(float) gGoboArray :  register(t151,space1);






        Tex2D(float4) gSkyVisScreen :  register(t152,space1);
#line 692 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Depth2D(float) gSceneDepthMS :  register(t153,space1);





        Tex2D(float4) gGiField :  register(t154,space1);



        Tex2D(float4) gWaterBodies :  register(t155,space1);

        Tex2D(float4) gWaterFlow :  register(t156,space1);




        Tex2D(float4) gSwField :  register(t157,space1);




        CBUFFER(LightData) gLights :  register(b0,space3);










        CBUFFER(LightData) gLightsNear :  register(b1,space3);
#line 740 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gTextures[ 4064 ] :  register(t0,space0);



        Tex2DArray(float4) gStaticsArrays[ 128 ] :  register(t4064,space0);



        Tex2DArray(float4) gFlipArrays[ 16 ] :  register(t4192,space0);
        CBUFFER(BatchData) gBatch :  register(b0,space2);
#line 764 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        CBUFFER(SkyViewData) gSkyView :  register(b1,space2);
#line 61 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/swswell.h.fsl"
#line 16 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/swswell.h.fsl"
float swSwellPer(int k) { return (k == 0) ? 1.0f : (k == 1) ? 0.75f : 0.58f; }
float swSwellAng(int k) { return (k == 0) ? 0.0f : (k == 1) ? 0.42f : -0.35f; }
float swSwellAmp(int k) { return (k == 0) ? 1.0f : (k == 1) ? 0.6f : 0.4f; }

float2 swSwellDir(float2 wind, int k)
{
    const float ca = cos(swSwellAng(k));
    const float sa = sin(swSwellAng(k));
    return float2(wind.x * ca - wind.y * sa, wind.x * sa + wind.y * ca);
}




float3 swSwellEval(float2 w, float2 wind, float A, float per, float c, float t, float footprint)
{
    float3 r = float3(0.0f, 0.0f, 0.0f);
    UNROLL for (int k = 0; k < 3; ++k)
    {
        const float lk = max(c, 1.0f) * per * swSwellPer(k);
        const float fp = (footprint > 0.0f) ? (1.0f - smoothstep(0.125f * lk, 0.25f * lk, footprint)) : 1.0f;
        const float2 d = swSwellDir(wind, k);
        const float kk = 6.28318531f / lk;
        const float ph = kk * (dot(w, d) - c * t) + 1.7f * float(k);
        const float a = A * swSwellAmp(k) * fp;
        r.x += a * cos(ph);
        r.yz -= d * (a * kk * sin(ph));
    }
    return r;
}




float2 swSwellDisp(float2 w, float2 wind, float A, float per, float c, float t, float r)
{
    float2 res = float2(0.0f, 0.0f);
    UNROLL for (int k = 0; k < 3; ++k)
    {
        const float Tk = max(per * swSwellPer(k), 0.1f);
        const float lk = max(c, 1.0f) * Tk;
        const float2 d = swSwellDir(wind, k);
        const float kk = 6.28318531f / lk;
        const float om = 6.28318531f / Tk;
        const float ph = kk * (dot(w, d) - c * t) + 1.7f * float(k);
        const float U =  689.69f  * A * swSwellAmp(k) / max(c, 1.0f);
        res += d * (U * (r * cos(ph) - om * sin(ph)) / (om * om + r * r));
    }
    return res;
}


float3 swSwellState(float2 w, float2 wind, float A, float per, float c, float t)
{
    float3 r = float3(0.0f, 0.0f, 0.0f);
    UNROLL for (int k = 0; k < 3; ++k)
    {
        const float lk = max(c, 1.0f) * per * swSwellPer(k);
        const float2 d = swSwellDir(wind, k);
        const float kk = 6.28318531f / lk;
        const float e = A * swSwellAmp(k) * cos(kk * (dot(w, d) - c * t) + 1.7f * float(k));
        r.x += e;
        r.yz += d * ( 689.69f  * e / max(c, 1.0f));
    }
    return r;
}
#line 62 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
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
#line 63 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
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
#line 64 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/scenecolor.h.fsl"
#line 112 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/scenecolor.h.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/scenemax.h.fsl"
#line 25 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/scenemax.h.fsl"
float3 clampSceneTerm(float3 c)
{
    return min(c, float3( 32768.0f ,  32768.0f ,  32768.0f ));
}
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
#line 288 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/scenecolor.h.fsl"
float3 expandExposedEmissiveP(float3 emis, float3 albedoRgb, float cov, float p, float unit)
{
    if (!(p > 0.0f)) { return emis; }





    float3 a = max(albedoRgb, float3(0.0f, 0.0f, 0.0f));
    float w = saturate(max(max(a.r, a.g), a.b) * max(cov, 0.0f));
    float lumaE = dot(max(emis, float3(0.0f, 0.0f, 0.0f)), float3(0.2126f, 0.7152f, 0.0722f));
    float fMin = unit / max(lumaE, unit);
    float f = fMin + (1.0f - fMin) * pow(w, p);
    return emis * f;
}



float3 expandExposedEmissive(float3 emis, float3 albedoRgb, float cov)
{
    return expandExposedEmissiveP(emis, albedoRgb, cov, gShadowParams.skyAO2.w, gShadowParams.sunny16.y);
}






float3 expandExposedEmissiveDelta(float3 emis, float3 albedoRgb, float cov)
{
    return expandExposedEmissive(emis, albedoRgb, cov) - emis;
}
#line 65 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
#line 78 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterplane.h.fsl"
#line 38 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterplane.h.fsl"
bool waterFogActive()
{
    return gShadowParams.waterFogPlane.y >= 0.5f;
}
#line 56 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterplane.h.fsl"
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
#line 79 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
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
#line 349 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
bool submergedColumn(float3 worldPosRel, out float depth, out float3 k, out float strength)
{
    depth = 0.0f;
    k = float3(0.0f, 0.0f, 0.0f);
    strength = gShadowParams.waterFogLight.w;
    if (!waterFogActive() || strength <= 0.0f) { return false; }

    depth = waterFogDepthBelow(worldPosRel);
    if (depth <= 0.0f) { return false; }


    k = lerp(gShadowParams.waterFogLight.rgb, gShadowParams.waterFogKd.rgb,
             gShadowParams.waterFogExt.w);
    return true;
}





float3 submergedAmbT(float depth, float3 k, float strength)
{
    float entry = lerp(1.0f,  0.934f , saturate(depth /  32.0f ));
    return lerp(float3(1.0f, 1.0f, 1.0f), exp(-k * depth *  1.2039f ) * entry, strength);
}




float3 waterAmbTransmit(float3 worldPosRel)
{
    float depth, strength; float3 k;
    if (!submergedColumn(worldPosRel, depth, k, strength)) { return float3(1.0f, 1.0f, 1.0f); }
    return submergedAmbT(depth, k, strength);
}
#line 405 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
float3 waterImmersedF0(float3 f0, float3 worldPosRel)
{
    if (!waterFogActive()) { return f0; }
    float w = saturate(waterFogDepthBelow(worldPosRel) /  32.0f );
    if (w <= 0.0f) { return f0; }
    float3 s = min(sqrt(saturate(f0)), f3(0.999f));
    float3 n = (f3(1.0f) + s) / (f3(1.0f) - s);
    float3 r = (n - f3( 1.333f )) / (n + f3( 1.333f ));
    return lerp(f0, r * r, w);
}

void waterLightTransmit(float3 worldPosRel, out float3 tSun, out float3 tAmb)
{
    tSun = float3(1.0f, 1.0f, 1.0f);
    tAmb = tSun;

    float depth, strength; float3 k;
    if (!submergedColumn(worldPosRel, depth, k, strength)) { return; }
#line 440 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float cosAir = abs(gFrameData.sunDir.z);
    float cosWater = sqrt(max(1.0f - (1.0f - cosAir * cosAir) / ( 1.333f  *  1.333f ), 0.0f));
    float pathSun = depth / max(cosWater,  0.6612f );

    tSun = lerp(tSun, exp(-k * pathSun), strength);
    tAmb = submergedAmbT(depth, k, strength);
#line 486 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float causStr = gShadowParams.waterCaustic.x;
    float dFar = gShadowParams.waterCaustic2.y * max(gFrameData.lodParams.w, 4096.0f);
    float dfade = (dFar > 0.0f)
                        ? 1.0f - smoothstep(0.5f * dFar, dFar, length(worldPosRel))
                        : 1.0f;
#line 510 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float cd0 = gShadowParams.waterCaustic3.x;



    float cdecay = (cd0 > 0.0f) ? exp(-pow(max(depth, 0.0f) / cd0, 1.27f)) : 1.0f;
    if (causStr * dfade * cdecay > 0.0f) {



        float2 worldXY = gFrameData.lodEye.xy + worldPosRel.xy;
#line 544 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
        float tanW = sqrt(max(1.0f - cosWater * cosWater, 0.0f)) / max(cosWater,  0.6612f );
        float2 sunXY = gFrameData.sunDir.xy;
        float sunL = length(sunXY);
        float2 bearing = (sunL > 1e-4f) ? (sunXY / sunL) : float2(1.0f, 0.0f);
        float2 shift = (sunL > 1e-4f) ? bearing * (-depth * tanW) : float2(0.0f, 0.0f);

        float2 uv = (worldXY + shift) * gShadowParams.waterCaustic.y;






        float3 caustic = float3(1.0f, 1.0f, 1.0f);








        float sIdx = gShadowParams.waterCaustic.z * log2(max(depth, 1.0f))
                    + gShadowParams.waterCaustic.w;
#line 581 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
        float fade = saturate(sIdx + 1.0f)
                    * (1.0f - saturate(sIdx - (float)( 5  - 1)));
#line 595 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
        if (fade > 0.0f && gShadowParams.waterCaustic3.y > 0.0f) {
            float s0 = clamp(floor(sIdx), 0.0f, (float)( 5  - 1));
            float s1 = min(s0 + 1.0f, (float)( 5  - 1));
            float sf = saturate(sIdx - s0);
#line 651 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
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
#line 698 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
        float2 dynPos = worldPosRel.xy + shift;



        float ripStr = gShadowParams.waterCausticDyn.x;
        if (ripStr > 0.0f) {
            float2 ruv = (dynPos - gShadowParams.waterCausticRip.xy)
                       * (gShadowParams.waterCausticRip.z / max(gShadowParams.waterCausticRip.w, 1.0f));
            float redge = causticDynEdge(ruv);
            if (redge > 0.0f) {
                float rIdx = gShadowParams.waterCausticDyn.z * log2(max(depth, 1.0f))
                           + gShadowParams.waterCausticDyn.w;
#line 725 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
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
#line 764 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
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
#line 850 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
bool causticProjectorActive(float3 worldPosRel)
{
    if (!waterFogActive()) { return false; }





    if (waterFogDepthBelow(worldPosRel) > 0.0f) {
        return gShadowParams.waterCausticProj.x > 0.0f
            || gShadowParams.waterCausticProj.z > 0.0f;
    }
    return gShadowParams.waterCausticProj.y > 0.0f;
}
#line 929 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
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
#line 994 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float tir = 1.0f;
    if (fragBelow && lightBelow) {
        float cosInc = abs(toFrag.z) / max(segLen, 1.0e-4f);
        tir = 1.0f - smoothstep( 0.6612f  * 0.85f,  0.6612f , cosInc);
        if (tir <= 0.0f) { return 1.0f; }
    }
#line 1012 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float devPerSlope = fragBelow ? (lightBelow ? 2.0f :  (1.0f - 1.0f / 1.333f ) ) :  ( 1.333f - 1.0f) ;
    float dEff = (devPerSlope /  (1.0f - 1.0f / 1.333f ) ) * pathL;
    float sIdx = gShadowParams.waterCaustic.z * log2(max(dEff, 1.0f))
                + gShadowParams.waterCaustic.w;
    float fade = saturate(sIdx + 1.0f)
                * (1.0f - saturate(sIdx - (float)( 5  - 1)));
#line 1030 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
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
#line 1186 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
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
#line 1241 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
void waterColumn(float3 worldPosRel, float L, bool openEnded, out float3 inscat, out float3 trans)
{
    float3 sigT = gShadowParams.waterFogExt.rgb;
    float3 sigS = gShadowParams.waterFogScatter.rgb;



    float3 kLgt = gShadowParams.waterFogKd.rgb;

    float d = max(length(worldPosRel), 1.0e-4f);




    float mEff = -worldPosRel.z / d;
    float dNear = waterFogEyeDepth();
    float dFar = max(dNear + mEff * L, 0.0f);
#line 1285 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    float mEffSeg = (L > 0.0f) ? ((dFar - dNear) / L) : mEff;

    float3 tauView = sigT * L;
    trans = openEnded ? float3(0.0f, 0.0f, 0.0f) : exp(-tauView);
#line 1363 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
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
#line 1444 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
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
#line 1490 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
    ph = lerp(ph, 1.0f, wIso);
#line 1526 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
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
#line 1570 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
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
#line 80 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
#line 122 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
float mwFogNearHaze(float dist)
{




    float k = max(gFrameData.froxelZ.w, 0.0f);
    return (k > 0.0f) ? exp(-k * max(dist, 0.0f)) : 1.0f;
}

float mwFogRamp(float dist)
{
    float fogStart = gFrameData.fogParams.x;
    float fogEnd = gFrameData.fogParams.y;
    float S = gShadowParams.toneParams.z;






    float span = max(fogEnd - fogStart, 1.0f);
    float t = saturate((dist - fogStart) / span);

    if (S > 0.0f)
    {
#line 171 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
        float x = S * t * t;
        float e = exp(-S);
        return saturate(mwFogNearHaze(dist) * (exp(-x) - e) / (1.0f - e));
    }


    return saturate(mwFogNearHaze(dist) * (1.0f - t));
}
#line 203 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
float mwFogFactor(float dist)
{
    if (waterCameraSubmerged()) { return 1.0f; }
    return mwFogRamp(dist);
}
#line 225 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
float mwFogFactorAt(float3 worldPos)
{
    return mwFogFactor(length(worldPos - gFrameData.eyePos.xyz));
}
#line 288 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
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
#line 66 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
#line 54 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterfog.h.fsl"
#line 55 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/fog.h.fsl"
#line 56 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
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


float atmosPhaseCornetteShanks(float cosT, float g)
{
    float g2 = g * g;
    float den = 1.0f + g2 - 2.0f * g * cosT;
    den = max(den, 1.0e-6f);
    return (3.0f / (8.0f *  3.14159265358979323846f )) * (1.0f - g2) * (1.0f + cosT * cosT)
         / ((2.0f + g2) * den * sqrt(den));
}
#line 219 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosphere.h.fsl"
float atmosPhaseAerosol(AtmosphereParams P, float cosT)
{
    if (P.nightP.w < 0.5f)
        return atmosPhaseMie(cosT, P.mieExt.w);
    return P.nightP.z * atmosPhaseMie(cosT, P.ozone.w)
         + (1.0f - P.nightP.z) * atmosPhaseCornetteShanks(cosT, P.nightP.y);
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
#line 279 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosphere.h.fsl"
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
#line 485 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosphere.h.fsl"
    float rGround = P.planet.x +  10.0f ;
    r = max(rGround, P.planet.x + saturate(xR) * (P.planet.y - P.planet.x));
}
#line 500 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/atmosphere.h.fsl"
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
#line 57 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skyradiance.h.fsl"
#line 52 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skyradiance.h.fsl"
float3 atmosSkyRadianceNative(float3 dir, float r)
{
    float2 uv = atmosSkyViewUvFromDir(gAtmosParams, r, dir);
    float vRow = 0.5f - 0.5f / max(gAtmosParams.lutDims.y, 2.0f);
    if (uv.y > vRow)
    {
        uv.y = vRow;
        float muRow, unusedLvc;
        atmosSkyViewParams(gAtmosParams, float2(0.5f, vRow), r, muRow, unusedLvc);
        float3 toSun = gAtmosParams.sunDir.xyz;
        float3 d = dir * rsqrt(max(dot(dir, dir), 1.0e-12f));
        float den = sqrt(max(0.0f, 1.0f - muRow * muRow))
                     * sqrt(max(0.0f, 1.0f - toSun.z * toSun.z));
        float lvc = (den > 1.0e-6f)
                     ? clamp((dot(d, toSun) - muRow * toSun.z) / den, -1.0f, 1.0f) : 1.0f;
        uv.x = atmosSkyViewUv(gAtmosParams, r, muRow, lvc).x;
    }
    float3 Lgap = SampleLvlTex2D(gAtmosSkyViewClear, gSamplerBilinearClamp, uv, 0).rgb;
    float3 Lmix = SampleLvlTex2D(gAtmosSkyView, gSamplerBilinearClamp, uv, 0).rgb;
    return max(lerp(Lgap, Lmix, saturate(gAtmosParams.deckMix.z)), f3(0.0f));
}








float3 atmosHorizonDirSameAngle(float3 d)
{
    float3 s = gAtmosParams.sunDir.xyz;
    float sH = sqrt(max(0.0f, 1.0f - s.z * s.z));
    float3 dH = float3(d.x, d.y, 0.0f);
    float dHl = length(dH);
    if (!(sH > 1.0e-4f))
        return (dHl > 1.0e-6f) ? (dH / dHl) : float3(1.0f, 0.0f, 0.0f);
    float3 sh = float3(s.x, s.y, 0.0f) / sH;
    float3 side = float3(-sh.y, sh.x, 0.0f);
    float cphi = clamp(dot(d, s) / sH, -1.0f, 1.0f);
    float sphi = sqrt(max(0.0f, 1.0f - cphi * cphi)) * ((dot(dH, side) < 0.0f) ? -1.0f : 1.0f);
    return sh * cphi + side * sphi;
}
#line 112 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skyradiance.h.fsl"
float3 atmosSunFloorDir(float3 d, float thetaF)
{
    if (!(thetaF > 0.0f)) { return d; }
    float3 s = gAtmosParams.sunDir.xyz;
    float sH = sqrt(max(0.0f, 1.0f - s.z * s.z));
    float dHl = sqrt(max(0.0f, 1.0f - d.z * d.z));
    float den = sH * dHl;
    if (!(den > 1.0e-4f)) { return d; }
    float th = acos(clamp(dot(d, s), -1.0f, 1.0f));
    float thF = min(sqrt(th * th + thetaF * thetaF), 3.14159265f);
    float cphi = clamp((cos(thF) - d.z * s.z) / den, -1.0f, 1.0f);
    float sphi = sqrt(max(0.0f, 1.0f - cphi * cphi));
    float3 sh = float3(s.x, s.y, 0.0f) / sH;
    float3 side = float3(-sh.y, sh.x, 0.0f);
    if (dot(float3(d.x, d.y, 0.0f), side) < 0.0f) { sphi = -sphi; }
    return float3((sh * cphi + side * sphi).xy * dHl, d.z);
}
#line 143 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skyradiance.h.fsl"
float3 atmosSunBeamAt(float r)
{
    float mu = gAtmosParams.sunDir.w;
    float muH = atmosHorizonCos(r, gAtmosParams.planet.x);
    if (!(mu > muH)) { return f3(0.0f); }
    float4 tr = SampleLvlTex2D(gAtmosTransmittance, gSamplerBilinearClamp,
                               atmosTransmittanceUv(gAtmosParams, r, mu), 0);
    float cosF = (mu - muH) / max(1.0f - muH, 1.0e-6f);
    return tr.rgb * lerp(1.0f, tr.a, saturate(gAtmosParams.deckMix.x)) * gAtmosParams.solar.xyz * cosF;
}
#line 58 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
#line 86 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
float4 fogSkySample(float2 pixelXy)
{
    return SampleLvlTex2D(gSkyColor, gSamplerBilinearClamp, pixelXy * gFrameData.fogParams.zw, 0);
}
#line 132 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
float3 fogHazeTarget(float3 worldPosRel, float fogAir)
{
    float scale = gFrameData.skyZenith.x;
    if (!(scale > 0.0f)) { return gFrameData.fogColNear.rgb; }
    float3 ray = worldPosRel - float3(0.0f, 0.0f, gFrameData.skyZenith.y);
    float3 dir = ray * rsqrt(max(dot(ray, ray), 1.0e-12f));
    float knee = clamp(gFrameData.froxelZ.z, 1.0e-3f, 0.999f);
    float nearW = 1.0f - saturate((1.0f - fogAir) / knee);




    if (dir.z < 0.0f) { dir = atmosHorizonDirSameAngle(dir); }


    float floorDeg = floor(gFrameData.skyZenith.z);
    float sinLift = gFrameData.skyZenith.z - floorDeg;
    dir.z = max(dir.z, sinLift * nearW);







    if (floorDeg > 0.0f)
    {
        float mu = clamp(dir.z, -1.0f, 1.0f);
        float2 hz = dir.xy * rsqrt(max(dot(dir.xy, dir.xy), 1.0e-12f));
        float3 dU = float3(hz * sqrt(max(0.0f, 1.0f - mu * mu)), mu);
        dir = atmosSunFloorDir(dU, radians(floorDeg));
    }
    return atmosSkyRadianceNative(dir, gAtmosParams.planet.w) * scale;
}
#line 181 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
float3 fogSkyTarget(float4 s, float skyShare, float3 haze)
{
    float3 mw = gFrameData.fogColNear.rgb;
    float w = gFrameData.fogColNear.w;
    float3 sky = s.rgb + haze * (1.0f - s.a);
    return lerp(mw, sky, w * skyShare) + (haze - mw) * (w * (1.0f - skyShare));
}
#line 231 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
float fogSkyShare(float fogAir)
{


    float knee = clamp(gFrameData.froxelZ.z, 0.0f, 0.999f);
    return smoothstep(knee, 1.0f, 1.0f - fogAir);
}





float3 fogSkyColor(float2 pixelXy)
{
    return fogSkyTarget(fogSkySample(pixelXy), 1.0f, gFrameData.fogColNear.rgb);
}





float3 fogSkyColorAt(float2 pixelXy, float fogAir, float3 worldPosRel)
{
    return fogSkyTarget(fogSkySample(pixelXy), fogSkyShare(fogAir), fogHazeTarget(worldPosRel, fogAir));
}
#line 280 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
float fogExtinction(float fogAir, float skyBehind)
{
    float ramp = pow(1.0f - fogAir, gFrameData.timeParams.w);
    return 1.0f - gFrameData.skyParams.w * ramp * skyBehind;
}
#line 298 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
float3 applyFog(float3 lit, float3 worldPosRel, float2 pixelXy)
{


    float fog = mwFogFactorAt(worldPosRel);




    float2 wf = waterFogSample(worldPosRel);









    fog = saturate(fog);






    if (fog >= 1.0f && wf.x == 0.0f && wf.y == 0.0f && !waterCameraSubmerged()) { return lit; }

    float4 s = fogSkySample(pixelXy);
#line 397 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
    float fogAir = mwFogAirShare(fog, wf.y, length(worldPosRel));
#line 423 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
    if (waterCameraSubmerged()) { fogAir = 1.0f; }
#line 457 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
    float upness = worldPosRel.z * rsqrt(max(dot(worldPosRel, worldPosRel), 1.0e-12f));
    float skyBehind = max(s.a, saturate(upness *  38.0f ));
#line 514 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
    float3 behind = waterFogComposite(lit, worldPosRel, wf);





    behind *= fogExtinction(fogAir, skyBehind);





    return lerp(fogSkyTarget(s, fogSkyShare(fogAir), fogHazeTarget(worldPosRel, fogAir)), behind, fogAir);
}
#line 67 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
#line 68 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/wavefield.h.fsl"
#line 111 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/wavefield.h.fsl"
float4 footprintEllipse(float2 dWdx, float2 dWdy)
{

    float axx = dWdx.x * dWdx.x + dWdy.x * dWdy.x;
    float ayy = dWdx.y * dWdx.y + dWdy.y * dWdy.y;
    float axy = dWdx.x * dWdx.y + dWdy.x * dWdy.y;

    float tr = axx + ayy;
    float cr = dWdx.x * dWdy.y - dWdx.y * dWdy.x;
    float disc = sqrt(max(tr * tr - 4.0f * cr * cr, 0.0f));
    float lmax = max(0.5f * (tr + disc), 1e-12f);
    float lmin = max(0.5f * (tr - disc), 0.0f);




    float2 v = float2(axy, lmax - axx);
    float vl = dot(v, v);
    float2 dir = (vl > 1e-20f) ? v * rsqrt(vl)
                               : ((axx >= ayy) ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f));
    return float4(lmin, lmax, dir);
}





uint noiseMix(uint h, uint k, uint m, uint sh) { h ^= k; h *= m; h ^= h >> sh; return h; }




float perlinGrad3(uint h32, float dx, float dy, float dw, out float gx, out float gy)
{
    uint h = h32 & 15u;
    bool uIsX = (h < 8u);
    bool vIsY = (h < 4u);
    bool vIsX = (h == 12u || h == 14u);
    float u = uIsX ? dx : dy;
    float v = vIsY ? dy : (vIsX ? dx : dw);
    float su = ((h & 1u) != 0u) ? -1.0f : 1.0f;
    float sv = ((h & 2u) != 0u) ? -1.0f : 1.0f;
    gx = (uIsX ? su : 0.0f) + ((!vIsY && vIsX) ? sv : 0.0f);
    gy = (uIsX ? 0.0f : su) + (vIsY ? sv : 0.0f);
    return su * u + sv * v;
}
#line 175 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/wavefield.h.fsl"
float perlinXYW(float3 pf, int3 ioff, uint zSeed, int wPeriod, out float2 dxy)
{
    float3 fl = floor(pf);
    float3 f = pf - fl;
    int3 i = int3(fl) + ioff;



    float3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
    float3 du = 30.0f * f * f * (f * (f - 2.0f) + 1.0f);


    int iw0 = i.z, iw1 = i.z + 1;
    if (wPeriod > 0) {
        iw0 = ((iw0 % wPeriod) + wPeriod) % wPeriod;
        iw1 = ((iw1 % wPeriod) + wPeriod) % wPeriod;
    }


    const uint seed0 = noiseMix(0x9E3779B9u, zSeed, 0x27D4EB2Fu, 15);
    uint hx[2];
    UNROLL
    for (int a = 0; a < 2; ++a) { hx[a] = noiseMix(seed0, asuint(i.x + a), 0x85EBCA6Bu, 13); }
    uint hxy[4];
    UNROLL
    for (int b = 0; b < 4; ++b) { hxy[b] = noiseMix(hx[b & 1], asuint(i.y + (b >> 1)), 0xC2B2AE35u, 16); }

    float val = 0.0f;
    float2 g = float2(0.0f, 0.0f);




    UNROLL
    for (int c = 0; c < 8; ++c)
    {
        int cx = c & 1, cy = (c >> 1) & 1, cw = (c >> 2) & 1;

        float ax = (cx != 0) ? u.x : 1.0f - u.x;
        float bx = (cx != 0) ? du.x : -du.x;
        float ay = (cy != 0) ? u.y : 1.0f - u.y;
        float by = (cy != 0) ? du.y : -du.y;
        float aw = (cw != 0) ? u.z : 1.0f - u.z;

        uint h = noiseMix(hxy[cy * 2 + cx], asuint((cw != 0) ? iw1 : iw0), 0x165667B1u, 13);

        float gx, gy;
        float dc = perlinGrad3(h, f.x - (float)cx, f.y - (float)cy, f.z - (float)cw, gx, gy);

        float wgt = ax * ay * aw;
        val += wgt * dc;
        g.x += (bx * ay * aw) * dc + wgt * gx;
        g.y += (ax * by * aw) * dc + wgt * gy;
    }

    dxy = g *  0.9820f ;
    return val *  0.9820f ;
}






float noiseFbm(float2 relXY, float2 eyeXY, float tt, float4 cfgA, float4 cfgB,
               float lac, float distortion, bool doNormalize, float4 fpEll,
               out float2 dv, out float s2)
{
    const float scale = max(cfgA.x, 1e-7f);
    const float detail = clamp(cfgA.y, 0.0f, (float)( 8  - 1));
    const float rough = max(cfgA.z, 0.0f);
    const float lacun = max(lac, 1.0f);







    const int wCells = (int)(cfgB.z + 0.5f);
    const float w0 = tt * ((float)wCells /  20.0f );
    const float z0 = cfgB.w;


    float2 pe = eyeXY * scale;
    float2 base = relXY * scale + (pe - floor(pe));
    float2 eoff = floor(pe);





    float4 JD = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (distortion != 0.0f)
    {
        float2 gdx, gdy;
        uint dseed = (uint)((int)(z0 * 64.0f));
        float dx = perlinXYW(float3(base + 13.5f, w0 + 13.5f), int3(eoff, 0), dseed ^ 0x51ED2701u, 0, gdx);
        float dy = perlinXYW(float3(base - 7.3f, w0 - 7.3f), int3(eoff, 0), dseed ^ 0x9E2A7C15u, 0, gdy);
        base += distortion * float2(dx, dy);
        JD = distortion * float4(gdx.x, gdx.y, gdy.x, gdy.y);
    }

    const int octInt = (int)detail;
    const float octFrc = detail - (float)octInt;
    const int octLast = octInt + ((octFrc != 0.0f) ? 1 : 0);

    float sum = 0.0f, maxamp = 0.0f, amp = 1.0f, fs = 1.0f;
    float2 gsum = float2(0.0f, 0.0f);
    float var = 0.0f;
    float lastVal = 0.0f;
    float2 lastGrad = float2(0.0f, 0.0f);
    float lastAmp = 0.0f;

    LOOP
    for (int o = 0; o <=  8 ; ++o)
    {
        if (o > octLast) { break; }





        float2 qo = eoff * fs;
        float2 qi = floor(qo);
        float3 P = float3(base * fs + (qo - qi), w0 * fs);



        uint zSeed = (uint)((int)(z0 * 64.0f)) * 0x9E3779B9u + (uint)o * 0x7F4A7C15u;




        int wp = (wCells > 0) ? max(1, (int)(((float)wCells * fs) + 0.5f)) : 0;
#line 324 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/wavefield.h.fsl"
        float sMin = sqrt(fpEll.x);
        float sMax = sqrt(fpEll.y);
        float ratio = sMax / max(sMin, 1e-6f);
        int nTap = (int)clamp(ceil(ratio), 1.0f, (float) 4 );
        float wTap = max(sMax / (float)nTap, sMin);

        float sfreq = scale * fs;
        float kk =  3.14159265f  *  3.14159265f  * sfreq * sfreq;
        float att = exp(-0.5f * kk * wTap * wTap);
        float att2 = att * att;
#line 358 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/wavefield.h.fsl"
        if (att < 0.01f) {
            float rr = (rough * lacun) * (rough * lacun);
            float tailAmp = amp * amp * sfreq * sfreq;
            float nLeft = (float)(octLast - o + 1);
            float tail = (abs(1.0f - rr) > 1e-3f)
                          ? tailAmp * (1.0f - pow(rr, nLeft)) / (1.0f - rr)
                          : tailAmp * nLeft;
            var += 2.0f * tail *  0.35f ;
            break;
        }



        float2 gOct = float2(0.0f, 0.0f);
        float vOct = 0.0f;
        float2 tapStep = fpEll.zw * (sfreq * 2.0f * sMax);
        LOOP
        for (int t = 0; t < nTap; ++t)
        {
            float u = ((2.0f * (float)t + 1.0f) / (float)nTap) - 1.0f;
            float2 gT;
            float vT = perlinXYW(float3(P.xy + tapStep * (0.5f * u), P.z),
                                  int3((int2)qi, 0), zSeed, wp, gT);
            vOct += vT;
            gOct += gT;
        }
        float invTap = 1.0f / (float)nTap;
        vOct *= invTap;
        gOct *= invTap;


        gOct *= sfreq;




        float bank = 2.0f * (1.0f - att2) * amp * amp * sfreq * sfreq *  0.35f ;

        if (o <= octInt) {
            sum += vOct * att * amp;
            gsum += gOct * att * amp;
            maxamp += amp;
            var += bank;
        } else {
            lastVal = vOct * att;
            lastGrad = gOct * att;
            lastAmp = amp;
            var += bank * octFrc * octFrc;
        }

        amp *= rough;
        fs *= lacun;
    }




    float sum2 = sum + lastVal * lastAmp;
    float2 g2 = gsum + lastGrad * lastAmp;
    float maxa2 = maxamp + lastAmp;

    float vA, vB;
    float2 gA, gB;
    if (doNormalize) {
        vA = 0.5f * sum / max(maxamp, 1e-8f) + 0.5f;
        vB = 0.5f * sum2 / max(maxa2, 1e-8f) + 0.5f;
        gA = 0.5f * gsum / max(maxamp, 1e-8f);
        gB = 0.5f * g2 / max(maxa2, 1e-8f);
        var *= 0.25f / max(maxamp * maxamp, 1e-16f);
    } else {
        vA = sum; vB = sum2; gA = gsum; gB = g2;
    }

    float mixw = (octFrc != 0.0f) ? octFrc : 0.0f;
    dv = lerp(gA, gB, mixw);
    s2 = var;



    if (distortion != 0.0f) {
        dv = float2(dv.x + dv.x * JD.x + dv.y * JD.z,
                    dv.y + dv.x * JD.y + dv.y * JD.w);
    }
    return lerp(vA, vB, mixw);
}
#line 467 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/wavefield.h.fsl"
float3 waveFieldGrad(float2 relXY, float2 eyeXY, float tt, float4 fpEll, out float sigma2)
{
    float4x4 A = transpose(gBatch.worlds[8]);
    float4x4 B = transpose(gBatch.worlds[9]);
    float4x4 G = transpose(gBatch.worlds[10]);

    const float lac = G[0].x;
    const float distort = G[0].y;
    const bool normalize = G[0].z > 0.5f;



    const int isolate = (int)(G[0].w + 0.5f);

    float h = 0.0f;
    float2 dh = float2(0.0f, 0.0f);
    sigma2 = 0.0f;

    UNROLL
    for (int n = 0; n <  4 ; ++n)
    {
        float4 cfgA = A[n];
        float4 cfgB = B[n];
        if (cfgA.w == 0.0f) { continue; }
        if (isolate > 0 && n != isolate - 1) { continue; }

        float2 dv; float s2;
        float v = noiseFbm(relXY, eyeXY, tt, cfgA, cfgB, lac, distort, normalize, fpEll, dv, s2);


        float span = cfgB.y - cfgB.x;
        float inv = (abs(span) > 1e-6f) ? 1.0f / span : 0.0f;
        float t = saturate((v - cfgB.x) * inv);
        float dt = (t > 0.0f && t < 1.0f) ? inv : 0.0f;
        float vMid = normalize ? 0.5f : 0.0f;
        float tMid = saturate((vMid - cfgB.x) * inv);

        float k = cfgA.w * dt;
        h += cfgA.w * (t - tMid);
        dh += k * dv;
        sigma2 += k * k * s2;
    }
    return float3(h, dh);
}
#line 535 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/wavefield.h.fsl"
float3 waveHeightDebug(float3 hg, float gain, float2 dWdx, float2 dWdy)
{
    float hv = hg.x * gain;
    float hw = abs(gain) * (abs(dot(hg.yz, dWdx)) + abs(dot(hg.yz, dWdy))) + 1e-8f;




    float3 col = (hv >= 0.0f)
               ? lerp(float3(0.06f, 0.06f, 0.07f), float3(1.00f, 0.58f, 0.10f), saturate(hv))
               : lerp(float3(0.06f, 0.06f, 0.07f), float3(0.10f, 0.42f, 1.00f), saturate(-hv));


    float ch = hv /  0.25f ;
    float d = abs(ch - round(ch)) *  0.25f  / hw;

    float lines = saturate(1.5f - d) * (1.0f - smoothstep(0.25f, 1.0f, hw /  0.25f ));
    col = lerp(col, float3(0.02f, 0.02f, 0.02f), 0.6f * lines);



    col = lerp(float3(1.0f, 1.0f, 1.0f), col, smoothstep(0.0f, 1.5f * hw, abs(hv)));
    return col;
}
#line 69 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterripple.h.fsl"
#line 68 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterripple.h.fsl"
void ripplePacket(float r, float rc, float W, float k, float slopeAmp, float4 fade, float fp,
                  out float dhdrOut, out float varOut)
{
    dhdrOut = 0.0f; varOut = 0.0f;
    float s = r - rc;
    float u = r * fade.x;
    if (abs(s) >= W || u >= 1.0f || W <= 0.0f || k <= 0.0f) { return; }







    float e = exp(-fade.y * u);
    float env = (e - fade.z) * fade.w;
    float dEnv = -fade.y * fade.x * e * fade.w;

    float x = 3.14159265f * s / W;
    float win = 0.5f * (1.0f + cos(x));
    float dwin = -0.5f * (3.14159265f / W) * sin(x);
    float ks = k * s;
    float c = cos(ks);
    float sn = sin(ks);

    float hA = slopeAmp / k;



    float dhdr = hA * (env * (dwin * c - k * win * sn) + dEnv * win * c);





    float res = 1.0f - smoothstep(1.5f, 3.14159265f, k * fp);
    dhdrOut = dhdr * res;
    float sAmp = slopeAmp * win * env;
    varOut = 0.5f * sAmp * sAmp * (1.0f - res * res);
}









float4 rippleHash4(int2 cell, int cyc)
{
    uint3 p = uint3(uint(cell.x + 32768), uint(cell.y + 32768), uint(cyc + 1));
    uint n = p.x * 1597334677u ^ p.y * 3812015801u ^ p.z * 2654435761u;
    n ^= n >> 15; n *= 2246822519u;
    n ^= n >> 13; n *= 3266489917u;
    n ^= n >> 16;
    uint4 q = uint4(n, n * 1664525u + 1013904223u,
                    n * 22695477u + 1u, n * 134775813u + 1u);
    q.y ^= q.y >> 15; q.z ^= q.z >> 15; q.w ^= q.w >> 15;
    return float4(q & uint4(0xFFFFFFu, 0xFFFFFFu, 0xFFFFFFu, 0xFFFFFFu)) * (1.0f / 16777216.0f);
}
#line 163 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterripple.h.fsl"
float3 rainRipples(float2 p, float phase, float cellSize, float period, float life,
                   float density, float slots, float slope, float lamMin, float lamMax,
                   float radiusMax, float decay, float cyclesPerWrap, float4 fpEll)
{
    float3 acc = float3(0.0f, 0.0f, 0.0f);
    if (density <= 0.0f || slope <= 0.0f || cellSize <= 0.0f || period <= 0.0f || life <= 0.0f) {
        return acc;
    }

    float inv = 1.0f / cellSize;
    int2 base = int2(floor(p * inv));
    float lifeFrac = min(life / period, 1.0f);
    float N = max(cyclesPerWrap, 1.0f);
    int slotsI = max(int(slots + 0.5f), 1);




    float rDie = max(min(radiusMax, cellSize), 1e-3f);
    float dcy = max(decay, 0.01f);
    float eEnd = exp(-dcy);
    float4 fade = float4(1.0f / rDie, dcy, eEnd, 1.0f / (1.0f - eEnd));

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 cell = base + int2(i, j);




            float4 hc = rippleHash4(cell, -1);










            for (int m = 0; m < slotsI; ++m) {










                float ph = phase + hc.z + float(m) / float(slotsI);
                float fl = floor(ph);
                float u = ph - fl;







                float cf = fl - N * floor(fl / N);
                int cyc = int(cf + 0.5f);

                int key = cyc * slotsI + m;
                float4 h = rippleHash4(cell, key);




                if (h.w > density) { continue; }





                float uStart = h.z * (1.0f - lifeFrac);
                float t = (u - uStart) * period;
                if (t <= 0.0f || t >= life) { continue; }





                float4 hp = rippleHash4(cell, key + 4096);
                float lam = lerp(lamMin, lamMax, hp.x);
                float n = floor(1.0f + hp.y * 2.999f);
                float k = 6.2831853f / max(lam, 0.05f);





                float W = 0.5f * (n + 1.0f) * lam;




                float rc = ((rDie + W) / life) * t;

                float2 d = p - (float2(cell) + h.xy) * cellSize;
                float r = length(d);
                if (r <= 1e-3f) { continue; }
                float2 dir = d / r;
#line 282 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/waterripple.h.fsl"
                float a2 = dot(dir, fpEll.zw); a2 *= a2;
                float fpd = 0.5f * sqrt(max(a2 * fpEll.y + (1.0f - a2) * fpEll.x, 1e-12f));

                float dd, vv;
                ripplePacket(r, rc, W, k, slope, fade, fpd, dd, vv);
                acc.xy += dd * dir;
                acc.z += vv;
            }
        }
    }
    return acc;
}
#line 70 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
#line 90 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
float waterSceneDevZ(float2 uvAlloc)
{
    return SampleLvlTex2D(gSceneLinDepth, gSamplerPointClamp, uvAlloc, 0.0f).r;
}
#line 149 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
float fresnelDielectric(float cosI, float eta)
{
    cosI = saturate(cosI);
    float sin2T = eta * eta * (1.0f - cosI * cosI);
    float cosT = sqrt(max(1.0f - sin2T, 0.0f));
    float a = eta * cosI;
    float rs = (a - cosT) / max(a + cosT, 1.0e-6f);
    float rp = (cosI - eta * cosT) / max(cosI + eta * cosT, 1.0e-6f);
    return (sin2T >= 1.0f) ? 1.0f : saturate(0.5f * (rs * rs + rp * rp));
}
#line 180 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
float fresnelDielectricRough(float cosI, float alpha, float eta)
{
#line 213 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float sigA = min(atan(alpha * 0.70710678f), 0.9069f);
    float d = 1.7320508f * sigA;
    float sinI = sqrt(saturate(1.0f - cosI * cosI));
    float cd = cos(d);
    float sd = sin(d);
    return (2.0f / 3.0f) * fresnelDielectric(cosI, eta)
         + (1.0f / 6.0f) * fresnelDielectric(cosI * cd + sinI * sd, eta)
         + (1.0f / 6.0f) * fresnelDielectric(cosI * cd - sinI * sd, eta);
}
#line 274 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
STRUCT(VSOutput)
{
    DATA(float4, Position, SV_Position);
    DATA(float3, WorldPos, TEXCOORD0);
    DATA(float2, RestXY, TEXCOORD1);
#line 279
};



float3 reconstructWorld(float4x4 invVP, float2 uv, float deviceZ)
{
    float ndcX = uv.x * 2.0f - 1.0f;
    float ndcY = 1.0f - uv.y * 2.0f;
    float4 p = mul(invVP, float4(ndcX, ndcY, deviceZ, 1.0f));
    return p.xyz / p.w;
}
#line 331 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
float occlusionFrom(float waterDist, float sceneDist, float nearRange)
{
    float valid = step(sceneDist, nearRange + 64.0f);
    return valid * smoothstep(-128.0f, 384.0f, waterDist - sceneDist);
}





float sceneOcclusionAt(float4x4 invVP, float2 uvVp, float2 texToVp, float waterDist, float nearRange)
{
    float devZ = waterSceneDevZ(uvVp / texToVp);
    float sceneDist = length(reconstructWorld(invVP, uvVp, devZ));
    return occlusionFrom(waterDist, sceneDist, nearRange);
}
#line 382 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
float2 clampAllocUV(float2 uv)
{
    float2 ia = gShadowParams.screenAlloc.zw;
    float2 h = 0.5f * ia;
    return clamp(uv, h, gShadowParams.screenParams.xy * ia - h);
}

float4 sampleReflectSmear(float2 uv, float lod, float2 stepUV, uint taps)
{
    if (taps < 2u) { return SampleLvlTex2D(gReflectMips, gSamplerTrilinearClamp, uv, lod); }
#line 414 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float2 lim = min(uv, 1.0f - uv);
    float2 sa = abs(stepUV);
    float kx = (sa.x > 1.0e-8f) ? (lim.x / sa.x) : 1.0e30f;
    float ky = (sa.y > 1.0e-8f) ? (lim.y / sa.y) : 1.0e30f;
    stepUV *= saturate(min(kx, ky));

    float4 acc = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float wsum = 0.0f;
    float inv = 2.0f / (float)(taps - 1u);
    for (uint i = 0u; i < taps; ++i) {
        float t = (float)i * inv - 1.0f;
        float w = exp(-t * t);
        acc += w * SampleLvlTex2D(gReflectMips, gSamplerTrilinearClamp, uv + t * stepUV, lod);
        wsum += w;
    }
    return acc / wsum;
}

[RootSignature( "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "3" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "3" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "3" ", offset = 0))," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "2" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "2" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "2" ", offset = 0))," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "1" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "1" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "1" ", offset = 0))," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "0" ", offset = 0, flags = DESCRIPTORS_VOLATILE)," "CBV(b0, numDescriptors = unbounded, space = " "0" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "0" ", offset = 0))," "DescriptorTable(" "SAMPLER(s0, numDescriptors = unbounded, space = " "0" ", offset = 0))," "StaticSampler(s0, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s1, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s2, space = 100," "filter = FILTER_MIN_MAG_LINEAR_MIP_POINT," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s3, space = 100," "filter = FILTER_MIN_MAG_LINEAR_MIP_POINT," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s4, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s5, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s6, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT," "addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR, addressW = TEXTURE_ADDRESS_MIRROR)," "StaticSampler(s7, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT, borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK," "addressU = TEXTURE_ADDRESS_BORDER, addressV = TEXTURE_ADDRESS_BORDER, addressW = TEXTURE_ADDRESS_BORDER)," "StaticSampler(s8, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR," "addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR, addressW = TEXTURE_ADDRESS_MIRROR)," "StaticSampler(s9, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR, borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK," "addressU = TEXTURE_ADDRESS_BORDER, addressV = TEXTURE_ADDRESS_BORDER, addressW = TEXTURE_ADDRESS_BORDER)," "StaticSampler(s10, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s11, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s12, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s13, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s14, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s15, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s16, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s17, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_WRAP)" )]


float foamHash(float2 p)
{
    p = frac(p * float2(127.1f, 311.7f));
    p += dot(p, p + 34.23f);
    return frac(p.x * p.y);
}
float foamVnoise(float2 p)
{
    const float2 i = floor(p);
    const float2 f = frac(p);
    const float2 u = f * f * (3.0f - 2.0f * f);
    const float a = foamHash(i), b = foamHash(i + float2(1.0f, 0.0f));
    const float c = foamHash(i + float2(0.0f, 1.0f)), d = foamHash(i + float2(1.0f, 1.0f));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}
float foamFbm2(float2 p)
{
    float v = 0.0f, a = 0.5f, fr = 1.0f;
    for (int i = 0; i < 4; ++i)
    {
        v += a * (1.0f - abs(2.0f * foamVnoise(p * fr) - 1.0f));
        fr *= 2.07f;
        a *= 0.5f;
    }
    return v;
}

float4 PS_MAIN( VSOutput In ): SV_TARGET
{
    //INIT_MAIN;


    float4x4 P = transpose(gBatch.worlds[6]);
    float4x4 invVP = gBatch.worlds[7];
    float waterLevelRel = P[0].x;
    float windFactor = P[0].y;
    float shoreDepthBias= P[0].z;
    float time = P[0].w;
    float3 depthBaseColor= P[1].xyz;
    bool underwater = P[1].w > 0.5f;
    float3 camFwd = P[2].xyz;
#line 490 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float distortFar = P[2].w * max(gFrameData.lodParams.w, 4096.0f);



    uint waterFlags = (uint)(P[3].x + 0.5f);
    uint waterDbg = waterFlags & 3u;
    bool useSchlick = (waterFlags & 4u) != 0u;



    bool noFogMelt = (waterFlags & 8u) != 0u;

    bool proceduralWaves = (waterFlags & 16u) != 0u;



    bool heightView = (waterFlags & 32u) != 0u;


    bool normalView = (waterFlags & 4096u) != 0u;

    bool truePlaneOn = (waterFlags & 8192u) != 0u;

    bool flatWater = (waterFlags & 16384u) != 0u;



    bool physFresnel = (waterFlags & 32768u) != 0u;



    bool snellGain = (waterFlags & 65536u) != 0u;










    bool hdrView = (waterFlags & 131072u) != 0u;
#line 544 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    bool noReflect = (waterFlags & 262144u) != 0u;

    bool destProbe = (waterFlags & 524288u) != 0u;

    bool flatProbe = (waterFlags & 1048576u) != 0u;





    waterDbg = 0u; heightView = false; normalView = false; hdrView = false;
    destProbe = false; flatProbe = false;






    uint smearTaps = (waterFlags >> 6) & 63u;


    float4 waveG1 = transpose(gBatch.worlds[10])[1];
    float heightGain = waveG1.x;


    float waveAmp = waveG1.y;
    float2 flowDir = float2(0.0f, 0.0f);
    float flowStr = 0.0f;
    float4 bodyW = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float2 bodyUV = float2(0.0f, 0.0f);



    {
        const float4 xf = gShadowParams.waterFlowMap;
        if (xf.z > 0.0f)
        {
            const float2 buv = (In.WorldPos.xy + gFrameData.lodEye.xy - xf.xy) * xf.zw;
            const float4 bw = SampleLvlTex2D(gWaterBodies, gSamplerBilinearClamp, buv, 0.0f);
            const float ws = bw.r + bw.g + bw.b + bw.a;
            bodyUV = buv;
            if (ws > 1.0e-3f)
            {
                bodyW = bw / ws;
                const float amp = dot(bw, gShadowParams.waterBodyAmp) / ws;
                waveAmp *= lerp(1.0f, amp, saturate(ws * 4.0f));
            }
            const float3 fl = SampleLvlTex2D(gWaterFlow, gSamplerBilinearClamp, buv, 0.0f).rgb;
            flowDir = fl.rg * 2.0f - 1.0f;
            flowStr = fl.b * saturate(length(flowDir) * 2.0f);
            flowDir = (dot(flowDir, flowDir) > 1.0e-6f) ? normalize(flowDir) : float2(0.0f, 0.0f);
        }
    }



    float waveSlices = max(waveG1.z, 1.0f);
    float waveTile = max(waveG1.w, 1.0f);



    float4 ripG2 = transpose(gBatch.worlds[10])[2];
    float4 ripG3 = transpose(gBatch.worlds[10])[3];
    float rippleAmp = ripG2.x;
    float rainCell = max(ripG2.y, 1.0f);
    float rainPeriod = max(ripG2.z, 0.01f);
    float rainDensity = ripG2.w;
    float rippleLamMin = max(ripG3.x, 0.1f);
    float rippleLamMax = max(ripG3.y, rippleLamMin);
    float rippleRadius = max(ripG3.z, 0.0f);
    float rippleLife = max(ripG3.w, 0.01f);




    float4 shoreG = transpose(gBatch.worlds[11])[1];
    float shoreDepthMax = shoreG.x;
    float shoreFadeNear = shoreG.y;
    float shoreFadeFar = max(shoreG.z, shoreG.y + 1.0f);
    float shoreBreath = shoreG.w;
    float4 ripG4 = transpose(gBatch.worlds[11])[0];
    float rippleCycles = max(ripG4.x, 1.0f);
    float rippleDecay = max(ripG4.y, 0.0f);
    float rippleSlots = max(ripG4.z, 1.0f);
    float ripplePhase = ripG4.w;




    float4 simG0 = transpose(gBatch.worlds[12])[0];
    float4 simG1 = transpose(gBatch.worlds[12])[1];

    float4 wakeG0 = transpose(gBatch.worlds[12])[2];
    float4 wakeG1 = transpose(gBatch.worlds[12])[3];









    float alphaBase = P[3].y;
    float reflSize = P[3].z;
    float reflBlurGain = P[3].w;
#line 664 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float2 invAlloc = gShadowParams.screenAlloc.zw;
    float2 invScreen = gShadowParams.screenParams.zw;
    float2 texToVp = gShadowParams.screenAlloc.xy * gShadowParams.screenParams.zw;
    float3 fogCol = gFrameData.fogColNear.rgb;
#line 723 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float trueRel = waterLevelRel - 1.0f;
    float2 rayUv = In.Position.xy * invScreen;
    float3 rayA = reconstructWorld(invVP, rayUv, 0.25f);
    float3 rayB = reconstructWorld(invVP, rayUv, 0.75f);
    float3 rayD = rayA - rayB;
    rayD = (dot(rayD, In.WorldPos) < 0.0f) ? -rayD : rayD;
    float3 rayDir = normalize(rayD);
#line 747 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float invFocal = length(ddy(rayDir));
    float focalPx = (invFocal > 1e-7f) ? (1.0f / invFocal) : 1000.0f;
#line 761 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float3 surfPos = In.WorldPos;
    float dist = length(surfPos);
    float3 EyeVec = truePlaneOn ? rayDir
                                 : ((dist > 1e-4f) ? surfPos / dist : float3(0, 0, 1));
#line 832 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float wcutR = max(gShadowParams.waterDistort.w, 0.0f);
    float wcutH = max(gShadowParams.waterCut.x, 0.0f);
    float wcutAbove = max(-waterLevelRel, 0.0f);
    float wcut = 1.0f;
    float wcutdark = 0.0f;
    if (wcutR > 0.0f && wcutAbove < wcutH) {
        float wd = smoothstep(0.889f * wcutR, 1.010f * wcutR, dist);
        wcut = smoothstep(0.900f * wcutR, 1.000f * wcutR, dist);
        wcutdark = saturate(wd * (1.0f - wd) * 3.0f);
    }
#line 884 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float fog = mwFogRamp(dist);
#line 931 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float distortFarL =  0.5f  * distortFar;
    float distortMag = (distortFarL > 0.0f) ? min(dist, 0.5f * distortFarL) : dist;
    float distortFadeL = (distortFarL > 0.0f)
                            ? (1.0f - smoothstep(0.5f * distortFarL, distortFarL, dist))
                            : 1.0f;

    float distortFade = (distortFar > 0.0f)
                            ? (1.0f - smoothstep(0.5f * distortFar, distortFar, dist))
                            : 1.0f;



    float distortAmpLegacy = (windFactor * distortMag + 0.1f) * distortFadeL;
#line 983 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float angMix = saturate(gShadowParams.waterDistort.x);
    float distortGain = max(gShadowParams.waterDistort.y, 0.0f);
    float distortAmpAng =  (1.0f - 1.0f / 1.333f )  * focalPx * distortFade;
    float distortAmp = lerp(distortAmpLegacy, distortAmpAng, angMix) * distortGain;



    float distortUWFactor = lerp(2.0f,  1.333f , angMix);




    float2 worldXY = surfPos.xy + gFrameData.lodEye.xy;









    float t = 0.4f * time;
    float2 tc2 = worldXY / waveTile;
#line 1042 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float2 dWdx = ddx(surfPos.xy);
    float2 dWdy = ddy(surfPos.xy);








    float4 fpEll = footprintEllipse(dWdx, dWdy);









    float3 ddxEye = ddx(EyeVec);
    float3 ddyEye = ddy(EyeVec);
    float pixAngle = max(length(ddxEye), length(ddyEye));
#line 1077 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float3 normal = float3(0.0f, 0.0f, 1.0f);
#line 1107 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float afSigma2 = 0.0f;
    float afAniso = 1.0f;
    {



        float texelWorld = waveTile /  1024.0f ;
        float sMinW = max(sqrt(fpEll.x), texelWorld);
        float sMaxW = max(sqrt(fpEll.y), sMinW);
        afAniso = sMaxW / sMinW;



        float zf = frac(t) * waveSlices;
        float z0 = floor(zf);
        float fz = zf - z0;
        float z1 = (z0 + 1.0f < waveSlices) ? (z0 + 1.0f) : 0.0f;





        float2 dTdx = dWdx / waveTile;
        float2 dTdy = dWdy / waveTile;
        float2 s0 = 2.0f *  gWaterNormalVol.SampleGrad(gSamplerAnisotropic, float3(tc2, z0), dTdx, dTdy) .
#line 1132 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
rg - 1.0f;
        float2 s1 = 2.0f *  gWaterNormalVol.SampleGrad(gSamplerAnisotropic, float3(tc2, z1), dTdx, dTdy) .
#line 1134 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
rg - 1.0f;




        const float flowSpeed = gShadowParams.waterFlowParams.x * flowStr;
        if (flowSpeed > 0.5f)
        {
            const float per = max(gShadowParams.waterFlowParams.y, 0.1f);
            const float ph0 = frac(time / per);
            const float ph1 = frac(time / per + 0.5f);
            const float2 d0 = flowDir * (flowSpeed * per * (ph0 - 0.5f) / waveTile);
            const float2 d1 = flowDir * (flowSpeed * per * (ph1 - 0.5f) / waveTile);
            const float wB = abs(2.0f * ph0 - 1.0f);
            float2 a0 = 2.0f *  gWaterNormalVol.SampleGrad(gSamplerAnisotropic, float3(tc2 - d0, z0), dTdx, dTdy) .rg - 1.0f;
            float2 a1 = 2.0f *  gWaterNormalVol.SampleGrad(gSamplerAnisotropic, float3(tc2 - d0, z1), dTdx, dTdy) .rg - 1.0f;
            float2 b0 = 2.0f *  gWaterNormalVol.SampleGrad(gSamplerAnisotropic, float3(tc2 - d1 + 0.37f, z0), dTdx, dTdy) .rg - 1.0f;
            float2 b1 = 2.0f *  gWaterNormalVol.SampleGrad(gSamplerAnisotropic, float3(tc2 - d1 + 0.37f, z1), dTdx, dTdy) .rg - 1.0f;
            s0 = lerp(a0, b0, wB);
            s1 = lerp(a1, b1, wB);
        }
        normal = normalize(float3(lerp(s0, s1, fz) * waveAmp, 1.0f));
#line 1194 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
        float wTap = max(sMinW, sMaxW / (float) 8 );
        float lodA = log2(wTap / texelWorld);
        float lodV = 0.5f * (lodA + log2(sMaxW / texelWorld));
#line 1208 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
        float lodS = max(lodV - 1.0f, 0.0f);
        float vr =  gWaterSlopeVar.SampleLevel(gSamplerTrilinearWrap, float3(tc2, z0), lodS) .r;
        afSigma2 = max(0.0f, vr) * min(lodV, 1.0f) * (waveAmp * waveAmp);
    }
#line 1225 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float waveSigma2 = 0.0f;
    float3 waveHG = float3(0.0f, 0.0f, 0.0f);
    if (proceduralWaves || heightView) {
        waveHG = waveFieldGrad(In.WorldPos.xy, gFrameData.lodEye.xy, time, fpEll, waveSigma2);
    }




    if (heightView) { RETURN(float4(waveHeightDebug(waveHG, heightGain, dWdx, dWdy), 1.0f)); }

    if (proceduralWaves) {



        normal = normalize(float3(-waveHG.yz, 1.0f));
    }










    {
        const float swellAmp = gShadowParams.waterFlowParams.w * bodyW.r;
        if (swellAmp > 0.01f)
        {
            float2 wd = float2(0.8f, 0.6f);
            if (flowStr > 0.01f) { wd = normalize(lerp(wd, flowDir, saturate(flowStr * 6.0f))); }
            const float sMaxW = sqrt(max(fpEll.y, 0.0f));
            const float4 lam = float4(6000.0f, 4000.0f, 2500.0f, 1600.0f);
            const float4 cyc = float4(2.0f, 3.0f, 4.0f, 5.0f);
            const float4 amp = float4(90.0f, 70.0f, 40.0f, 20.0f);
            const float4 ang = float4(0.0f, 0.35f, -0.3f, 0.6f);
            float2 sl = float2(0.0f, 0.0f);
            for (int i = 0; i < 4; ++i)
            {
                const float ca = cos(ang[i]), sa = sin(ang[i]);
                const float2 d = float2(wd.x * ca - wd.y * sa, wd.x * sa + wd.y * ca);
                const float k = 6.2831853f / lam[i];
                const float ph = k * dot(d, worldXY) - 6.2831853f * cyc[i] * time / 20.0f;
                const float fd = saturate(1.0f - 4.0f * sMaxW / lam[i]);
                sl -= d * (amp[i] * k * cos(ph) * fd);
            }
            const float2 base = normal.xy / max(normal.z, 1e-4f);
            normal = normalize(float3(base + sl * swellAmp, 1.0f));
        }
    }
#line 1312 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float sigma2 = proceduralWaves ? waveSigma2 : afSigma2;
#line 1338 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    if (rippleAmp > 0.0f && !flatWater) {
        float3 rip = rainRipples(worldXY, ripplePhase, rainCell, rainPeriod, rippleLife,
                                  rainDensity, rippleSlots, rippleAmp, rippleLamMin, rippleLamMax,
                                  rippleRadius, rippleDecay, rippleCycles, fpEll);
        float2 base = normal.xy / max(normal.z, 1e-4f);
        normal = normalize(float3(base + rip.xy, 1.0f));
        sigma2 += rip.z;
    }
#line 1360 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    if (simG1.y > 0.5f && !flatWater) {
        float2 texel = (surfPos.xy - simG0.xy) * simG0.z;
        float2 uv = texel / simG0.w;



        float2 e = min(uv, 1.0f - uv);
        float edge = saturate(min(e.x, e.y) * 8.0f);
        if (edge > 0.0f)
        {






            float4 fld = SampleLvlTex2D(gRippleField, gSamplerTrilinearClamp, uv, 0);
            float2 slope = fld.zw * simG0.z * simG1.x * edge;
            float2 base3 = normal.xy / max(normal.z, 1e-4f);
            normal = normalize(float3(base3 + slope, 1.0f));
        }
    }








    if (wakeG1.y > 0.5f && !flatWater) {
        float2 wtex = (surfPos.xy - wakeG0.xy) * wakeG0.z;
        float2 wuv = wtex / wakeG0.w;
        float2 we = min(wuv, 1.0f - wuv);
        float wfade = saturate(min(we.x, we.y) * 8.0f);
        if (wfade > 0.0f)
        {
            float4 wfld = SampleLvlTex2D(gWakeField, gSamplerTrilinearClamp, wuv, 0);
            float2 wslope = wfld.zw * wakeG0.z * wakeG1.x * wfade;
            float2 base4 = normal.xy / max(normal.z, 1e-4f);
            normal = normalize(float3(base4 + wslope, 1.0f));
        }
    }





    float swWW = 0.0f;
    {
        const float4x4 SWP = transpose(gBatch.worlds[15]);
        const float4x4 SWQ = transpose(gBatch.worlds[17]);
        const float4 s0 = SWP[0];
        const float4 s1 = SWP[1];
        const float4 s2 = SWP[2];
        const float4 s3 = SWP[3];
        if (s3.z > 0.0f && !flatWater)
        {

            float sw = 0.0f;
            const float2 suv = (In.RestXY - s0.xy) * s0.z;
            if (s0.w > 0.5f)
            {
                const float2 se = min(suv, 1.0f - suv);
                sw = smoothstep(0.0f, max(s2.x, 1.0e-3f), min(se.x, se.y))
                   * (1.0f - smoothstep(s3.x, max(s3.y, s3.x + 1.0f), abs(s2.w)));
            }
            float2 sslope = float2(0.0f, 0.0f);
            if (sw < 1.0f)
            {


                const float bAmp = (dot(bodyW, float4(1.0f, 1.0f, 1.0f, 1.0f)) > 0.5f)
                                 ? (bodyW.r * gShadowParams.waterBodyAmp.x + bodyW.a * gShadowParams.waterBodyAmp.w)
                                 : 1.0f;
                const float expo = (gShadowParams.waterFlowMap.z > 0.0f)
                                 ? SampleLvlTex2D(gWaterFlow, gSamplerBilinearClamp, bodyUV, 0.0f).a : 1.0f;
                const float fpW = dist / (focalPx * max(abs(EyeVec.z), 0.1f));
                const float3 an = swSwellEval(In.RestXY + gFrameData.lodEye.xy, SWQ[0].xy, s3.z * bAmp * expo,
                                               s3.w, SWQ[0].w, SWQ[0].z, fpW);
                sslope = -an.yz * (1.0f - sw);
            }
            if (sw > 0.0f)
            {
                const float tx = 16.0f * s0.z;
                const float hR = SampleLvlTex2D(gSwField, gSamplerBilinearClamp, suv + float2(tx, 0.0f), 0.0f).z;
                const float hL = SampleLvlTex2D(gSwField, gSamplerBilinearClamp, suv - float2(tx, 0.0f), 0.0f).z;
                const float hU = SampleLvlTex2D(gSwField, gSamplerBilinearClamp, suv + float2(0.0f, tx), 0.0f).z;
                const float hD = SampleLvlTex2D(gSwField, gSamplerBilinearClamp, suv - float2(0.0f, tx), 0.0f).z;

                sslope += -float2(hR - hL, hU - hD) / 32.0f * sw;
                swWW = SampleLvlTex2D(gSwField, gSamplerBilinearClamp, suv, 0.0f).w * s1.w * sw;
            }
            sslope *= s1.x * s1.z;
            const float2 base5 = normal.xy / max(normal.z, 1e-4f);
            normal = normalize(float3(base5 + sslope, 1.0f));
        }
    }





    float alpha = sqrt(alphaBase * alphaBase + sigma2);
#line 1511 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float reflTexelRad = max(pixAngle * (gShadowParams.screenParams.x / max(reflSize, 1.0f)), 1e-7f);
    float cosI = abs(EyeVec.z);
#line 1531 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float reflDevZ = SampleLvlTex2D(gReflectDepth, gSamplerPointClamp,
                                      In.Position.xy * invScreen, 0.0f).r;
    float reflDist = length(reconstructWorld(invVP, In.Position.xy * invScreen, reflDevZ));
    float hitFrac = saturate(1.0f - dist / max(reflDist, dist + 1e-3f));

    float lobeMajor = 2.0f * alpha * hitFrac;
    float lobeMinor = lobeMajor * cosI;
#line 1562 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float tapEff = max(lobeMinor, reflTexelRad);
    float anisoReq = lobeMajor / max(tapEff, 1e-9f);
    uint smearN = (smearTaps > 1u)
                   ? (uint)clamp(ceil(anisoReq), 1.0f, (float)smearTaps) : 1u;
    float tapRad = (smearN > 1u) ? max(tapEff, lobeMajor / (float)smearN) : lobeMinor;
    float reflLodRaw = log2(max(reflBlurGain * tapRad / reflTexelRad, 1.0f));
#line 1582 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float nz = (EyeVec.z < 0.0f) ? 1.0f : -1.0f;
    float sinI = max(length(EyeVec.xy), 1e-4f);
    float3 smearDir = float3(cosI * EyeVec.xy / sinI, sinI * nz);
    float2 smearPx = float2(dot(smearDir, ddxEye) / max(dot(ddxEye, ddxEye), 1e-12f),
                             dot(smearDir, ddyEye) / max(dot(ddyEye, ddyEye), 1e-12f));


    float2 smearUV = max(lobeMajor - tapRad, 0.0f) * smearPx * invScreen;
#line 1612 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float reflLod = min(reflLodRaw, (float) 3 );
#line 1634 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float over = max(reflLodRaw - (float) 3 , 0.0f);
    float reflSkyMix = 1.0f - exp2(-over);

    float2 baseUV = In.Position.xy * invAlloc;
#line 1651 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    if (underwater) {
#line 1671 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
        bool uwUnified = waterFogCameraSubmerged();



        float2 uwWf = waterFogSampleSubmerged(In.WorldPos);







        float3 uwFogCol = uwUnified ? waterFogColor(In.WorldPos) : fogCol;
        float uwTrans = uwUnified ? (1.0f - uwWf.x)
                                     : saturate(exp(-dist / 500.0f));
        float uwFog = uwUnified ? uwTrans : saturate(exp(-dist / 4096.0f));
        float3 nrm = -normal;
#line 1768 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
        float3 uwDeep = uwFogCol;
        {
            float volS = waterFogVolStrength();
            if (volS > 0.0f) {
                float3 rdir = float3(EyeVec.x, EyeVec.y, -abs(EyeVec.z));
                float3 dIns, dTr;
                waterColumn(rdir, 0.0f, true, dIns, dTr);
                uwDeep = lerp(uwFogCol, dIns, volS);
            }
        }
#line 1795 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
        float cosU = dot(-EyeVec, nrm);
        float fresnelU = physFresnel ? fresnelDielectricRough(cosU, alpha,  1.333f )
                                      : pow(saturate(1.12f - 0.65f * cosU), 8.0f);





        if (noReflect) { fresnelU = 0.0f; }





        float2 reffactorU = distortUWFactor * distortAmp * nrm.xy;
#line 1833 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
        float2 reffactorUR = (1.0f - fresnelU) * reffactorU;


        float2 ruvU = clampAllocUV(baseUV + (-2.0f * reffactorUR) * invAlloc);
        float3 refractedU = SampleLvlTex2D(gRefractColor, gSamplerBilinearClamp, ruvU, 0.0f).rgb;
#line 1868 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
        refractedU *= snellGain ? ( 1.333f  *  1.333f ) : 1.0f;









        float3 refrLegacy = lerp(uwFogCol, refractedU, uwTrans);
        refractedU = lerp(refrLegacy, refractedU, waterFogVolStrength());







        float2 reflUVU = In.Position.xy * invScreen
                       + float2(-2.1f * reffactorU.x, abs(reffactorU.y)) * invScreen;
        float4 reflSampleU = sampleReflectSmear(reflUVU, reflLod, smearUV, smearN);
        float3 reflectedU = reflSampleU.rgb + uwDeep * (1.0f - reflSampleU.a);
#line 1922 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
        float4 reflAvgU = SampleLvlTex2D(gReflectMips, gSamplerTrilinearClamp, reflUVU,
                                         (float) 5 );
        float3 uwRough = reflAvgU.rgb + uwDeep * (1.0f - reflAvgU.a);
        if (reflSkyMix > 0.0f) { reflectedU = lerp(reflectedU, uwRough, reflSkyMix); }
#line 1944 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
        reflectedU = waterFogBlend(reflectedU, reflectedU, In.WorldPos, uwWf);






        if (waterDbg == 3u) { RETURN(float4(reflectedU, 1.0f)); }

        if (waterDbg == 1u) {
            float4 rawReflU = SampleLvlTex2D(gReflectColor, gSamplerBilinearClamp,
                                             In.Position.xy * invScreen, 0.0f);
            return (float4(rawReflU.rgb, 1.0f));
        }
        if (waterDbg == 2u) { RETURN(float4(refractedU, 1.0f)); }





        float3 resultU = lerp(refractedU, reflectedU, fresnelU);
#line 1982 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
        if (!physFresnel) {
            float3 sunPosU = -gFrameData.sunDir.xyz;
            float refractsun = saturate(dot(-EyeVec, normalize(-sunPosU + nrm)));
            resultU += clampSceneTerm(gFrameData.sunCol.rgb * pow(refractsun, 6.0f) * uwFog);
        }

        return (float4(resultU, 1.0f));
    }
#line 2044 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    if (flatWater) { normal = float3(0.0f, 0.0f, 1.0f); }
#line 2058 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float NoV = saturate(dot(-EyeVec, normal));
    float fresnel;
    if (physFresnel) {




        fresnel = fresnelDielectricRough(NoV, alpha, 1.0f /  1.333f );
    } else if (useSchlick) {







        float f90 = max(1.0f - alpha,  0.02f );
        float m = 1.0f - NoV;
        float m5 = m * m; m5 = m5 * m5 * m;
        fresnel =  0.02f  + (f90 -  0.02f ) * m5;
    } else {



        fresnel =  0.02f  + pow(saturate(0.9988f - 0.28f * NoV), 16.0f);
    }

    if (noReflect) { fresnel = 0.0f; }


    float2 reffactor = distortAmp * normal.xy;
#line 2117 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float shorePre = 0.0f;
    if (shoreDepthMax > 0.0f) {
        float shoreDevZ = waterSceneDevZ(baseUV);
        float shoreDist = length(reconstructWorld(invVP, baseUV * texToVp, shoreDevZ));
        float shoreDepth = max(shoreDist - dist, 0.0f) + shoreBreath * (0.95f - normal.z);
        shorePre = (1.0f - smoothstep(0.0f, shoreDepthMax, shoreDepth))
                 * (1.0f - smoothstep(shoreFadeNear, shoreFadeFar, dist))
                 * (1.0f - occlusionFrom(dist, shoreDist, gFrameData.lodParams.w));
    }
    float shoreCancel = 1.0f - shorePre;
#line 2154 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float2 reffactorR = (1.0f - fresnel) * shoreCancel * wcut * reffactor;


    float2 distUV = clampAllocUV(baseUV + reffactorR.yx * invAlloc);
    float sceneDevZ= waterSceneDevZ(distUV);
    float3 sceneW = reconstructWorld(invVP, distUV * texToVp, sceneDevZ);
    float sceneDist= length(sceneW);
#line 2229 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float refrNear = gFrameData.lodParams.w;
    float refrOccl = (1.0f - smoothstep(refrNear, refrNear + 128.0f, sceneDist))
                      * smoothstep(0.0f, 384.0f, dist - sceneDist);


    float3 refrDist = float3(0.0f, 0.0f, 0.0f);
    float3 refrFlat = float3(0.0f, 0.0f, 0.0f);
    float shorefactor = 0.0f;
#line 2297 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    {





        float2 ruv = clampAllocUV(baseUV + (1.0f - refrOccl) * reffactorR.yx * invAlloc);
        refrDist = SampleLvlTex2D(gRefractColor, gSamplerBilinearClamp, ruv, 0.0f).rgb;
        refrFlat = SampleLvlTex2D(gRefractColor, gSamplerBilinearClamp, baseUV, 0.0f).rgb;
#line 2347 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
        shorefactor = shorePre;





    }


	float3 sunPos = -gFrameData.sunDir.xyz;
    float3 L = normalize(sunPos);
    float3 V = -EyeVec;
    float3 H = normalize(L + V);
    float NoL = saturate(dot(normal, L));
    float NoH = saturate(dot(normal, H));
    float VoH = saturate(dot(V, H));
#line 2397 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float2 reflBase = In.Position.xy * invScreen;
    float2 reflOff = shoreCancel * float2(-2.1f * reffactor.x, abs(reffactor.y)) * invScreen;
    float reflOccl = sceneOcclusionAt(invVP, reflBase + reflOff, texToVp, dist,
                                       gFrameData.lodParams.w);
    float2 reflUV = reflBase + (1.0f - reflOccl) * reflOff;
    float4 reflSample = sampleReflectSmear(reflUV, reflLod, smearUV, smearN);
#line 2427 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float3 reflHole = fogSkyColor(In.Position.xy);
    float3 reflected = reflSample.rgb + reflHole * (1.0f - reflSample.a);



    if (reflSkyMix > 0.0f) {
        reflected = lerp(reflected, reflHole, reflSkyMix);
    }
#line 2526 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    if (waterDbg == 1u) {
        float4 rawRefl = SampleLvlTex2D(gReflectColor, gSamplerBilinearClamp,
                                        In.Position.xy * invScreen, 0.0f);
        return (float4(rawRefl.rgb, 1.0f));
    }
#line 2543 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    if (hdrView) {
        float4 raw = SampleLvlTex2D(gReflectColor, gSamplerPointClamp,
                                    In.Position.xy * invScreen, 0.0f);
        float3 rad = (raw.a > 0.001f) ? (raw.rgb / raw.a) : float3(0.0f, 0.0f, 0.0f);
        float m = max(rad.r, max(rad.g, rad.b));
        float3 c;
        if (m >= 0.998f && m <= 1.002f) { c = float3(1.0f, 0.0f, 1.0f); }
        else if (m < 1.0f) { c = float3(0.25f, 0.25f, 0.25f) * m; }
        else if (m < 2.0f) { c = float3(0.0f, 1.0f, 0.0f); }
        else if (m < 4.0f) { c = float3(1.0f, 1.0f, 0.0f); }
        else if (m < 8.0f) { c = float3(1.0f, 0.5f, 0.0f); }
        else { c = float3(1.0f, 0.0f, 0.0f); }
        return (float4(c, 1.0f));
    }






    if (waterDbg == 2u) { RETURN(float4(refrDist, 1.0f)); }
#line 2582 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    if (waterDbg == 3u) {
        return (float4(saturate(reflLod / (float) 3 ),
                      saturate(alpha * 20.0f),
                      saturate(log2(afAniso) / log2((float) 8 )), 1.0f));
    }
#line 2603 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    if (normalView) {
        float3 mirrorDir = reflect(EyeVec, normal);
        float glint = saturate(dot(mirrorDir, -gFrameData.sunDir.xyz));
        return (float4(-EyeVec.x,
                      EyeVec.y,
                      -EyeVec.z, 1.0f));
    }
#line 2638 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float3 result = reflected * fresnel;
    float kDst = 1.0f - fresnel;
#line 2660 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float aSpec = max(alpha,  0.00465f );
    float aEff = saturate(aSpec +  0.00465f );
    float a2 = aEff * aEff;


    float dDen = NoH * NoH * (a2 - 1.0f) + 1.0f;
    float D = a2 / max( 3.14159265359f  * dDen * dDen, 1e-8f);


    float lv = NoL * sqrt(NoV * NoV * (1.0f - a2) + a2);
    float ll = NoV * sqrt(NoL * NoL * (1.0f - a2) + a2);
    float Vis = 0.5f / max(lv + ll, 1e-8f);








    float fm = 1.0f - VoH;
    float fm5 = fm * fm; fm5 = fm5 * fm5 * fm;
    float F = physFresnel ? fresnelDielectric(VoH, 1.0f /  1.333f )
                             :  0.02f  + (1.0f -  0.02f ) * fm5;



    float sphereNorm = (aSpec * aSpec) / max(a2, 1e-8f);
    float3 spec = gFrameData.sunCol.rgb * (D * Vis * F * NoL * min(sphereNorm, 1.0f));
#line 2714 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    if (!noReflect) { result += tonemapInPass(clampSceneTerm(spec)); }







    result = result * wcut;
    kDst = kDst * wcut + (1.0f - wcut);


    float wcutDim = 1.0f - 0.9f * wcutdark;
    result = result * wcutDim;
    kDst = kDst * wcutDim;
#line 2748 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    result = result * (1.0f - shorefactor);
    kDst = kDst * (1.0f - shorefactor) + shorefactor;
#line 2763 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    {
        const float foamAmt = gShadowParams.waterFlowParams.z;
        const float riverC = bodyW.g;
        const float shoreC = bodyW.r + bodyW.a;
        if (foamAmt > 0.0f && (riverC + shoreC) > 0.01f && dist < 12000.0f)
        {
            const float per = max(gShadowParams.waterFlowParams.y, 0.1f);
            const float ph0 = frac(time / per);
            const float ph1 = frac(time / per + 0.5f);
            const float fbl = abs(1.0f - 2.0f * ph0);
            const float spd = gShadowParams.waterFlowParams.x * max(flowStr, 0.25f);
            const float kTile = 160.0f;

            const float2 eyeT = frac(gFrameData.lodEye.xy / (kTile * 64.0f)) * 64.0f;
            const float2 fP = surfPos.xy / kTile + eyeT;
            const float2 fdisp = flowDir * (spd * per / kTile);
            float nf = lerp(foamFbm2(fP - fdisp * ph0), foamFbm2(fP - fdisp * ph1 + 7.3f), fbl);
            float ncr = lerp(foamFbm2(fP / 3.0f - fdisp / 3.0f * ph0), foamFbm2(fP / 3.0f - fdisp / 3.0f * ph1 + 7.3f), fbl);
            float nfi = lerp(foamFbm2(fP * 5.0f - fdisp * 5.0f * ph0), foamFbm2(fP * 5.0f - fdisp * 5.0f * ph1 + 7.3f), fbl);


            const float4 xf = gShadowParams.waterFlowMap;
            const float2 cs = float2(512.0f * xf.z, 512.0f * xf.w);
            const float gR = SampleLvlTex2D(gWaterFlow, gSamplerBilinearClamp, bodyUV + float2(cs.x, 0.0f), 0.0f).g;
            const float gL = SampleLvlTex2D(gWaterFlow, gSamplerBilinearClamp, bodyUV - float2(cs.x, 0.0f), 0.0f).g;
            const float rU = SampleLvlTex2D(gWaterFlow, gSamplerBilinearClamp, bodyUV + float2(0.0f, cs.y), 0.0f).r;
            const float rD = SampleLvlTex2D(gWaterFlow, gSamplerBilinearClamp, bodyUV - float2(0.0f, cs.y), 0.0f).r;
            const float vort = abs((gR - gL) - (rU - rD)) * 4.0f;



            const float fDevZ = waterSceneDevZ(baseUV);
            const float fDist = length(reconstructWorld(invVP, baseUV * texToVp, fDevZ));
            const float vDepth = max(fDist - dist, 0.0f) * abs(EyeVec.z) + 40.0f * (0.95f - normal.z);



            float band = 1.0f - saturate(vDepth / 120.0f);
            band = band * band * (1.0f - occlusionFrom(dist, fDist, gFrameData.lodParams.w));

            float cover = saturate(riverC * (1.0f + vort) + shoreC * band);
            cover *= lerp(1.0f, saturate(0.4f + ncr), 0.5f);
            float foam = saturate((cover * (0.5f + nf) - 0.98f) * 3.0f);
            foam *= lerp(1.0f, saturate(0.5f + nfi), 0.6f);
            foam *= foamAmt * (1.0f - smoothstep(6000.0f, 12000.0f, dist));

            float sunVis = 1.0f;
            const float4 so = gShadowParams.sunOcc;
            const float4 sm = gShadowParams.skyAOMap;
            if (so.x > 0.0f && sm.z > 0.0f)
            {
                const float2 ouv = (surfPos.xy + gFrameData.lodEye.xy - sm.xy) * sm.z;
                const float blockZ = SampleLvlTex2D(gSunOcc, gSamplerBilinearClamp, ouv, 0.0f).r;
                float t = saturate(((blockZ - so.z) - (surfPos.z + gFrameData.lodEye.z + 8.0f)) / max(so.y, 1.0e-3f));
                t = t * t * (3.0f - 2.0f * t);
                sunVis = 1.0f - t * so.x;
            }
            const float3 Lf = -normalize(gFrameData.sunDir.xyz);
            const float3 foamCol = 0.6f * (gFrameData.sunCol.rgb * saturate(Lf.z) * sunVis + gFrameData.ambCol.rgb);
            foam *= 0.85f;
            result = lerp(result, foamCol, foam);
            kDst = kDst * (1.0f - foam);
        }
    }





    if (swWW > 0.002f)
    {
        const float kT = 48.0f;
        const float2 eyeT = frac(gFrameData.lodEye.xy / (kT * 64.0f)) * 64.0f;
        const float2 fP = In.RestXY / kT + eyeT;
        const float nf = foamFbm2(fP);
        const float nfi = foamFbm2(fP * 4.0f + 3.1f);


        float wf = saturate((saturate(swWW) * nf - 0.25f) * 6.0f);
        wf *= saturate(nfi * 1.6f - 0.2f);
        float sunVisW = 1.0f;
        const float4 so = gShadowParams.sunOcc;
        const float4 sm = gShadowParams.skyAOMap;
        if (so.x > 0.0f && sm.z > 0.0f)
        {
            const float2 ouv = (surfPos.xy + gFrameData.lodEye.xy - sm.xy) * sm.z;
            const float blockZ = SampleLvlTex2D(gSunOcc, gSamplerBilinearClamp, ouv, 0.0f).r;
            float t = saturate(((blockZ - so.z) - (surfPos.z + gFrameData.lodEye.z + 8.0f)) / max(so.y, 1.0e-3f));
            t = t * t * (3.0f - 2.0f * t);
            sunVisW = 1.0f - t * so.x;
        }
        const float3 Lw = -normalize(gFrameData.sunDir.xyz);
        const float3 wwCol = 0.6f * (gFrameData.sunCol.rgb * saturate(Lw.z) * sunVisW + gFrameData.ambCol.rgb);
        wf *= 0.9f;
        result = lerp(result, wwCol, wf);
        kDst = kDst * (1.0f - wf);
    }






    {
        const float bodyView = transpose(gBatch.worlds[17])[2].x;
        if (bodyView > 0.5f)
        {
            float3 dbg;
            if (bodyView < 1.5f)
            {
                dbg = bodyW.r * float3(0.10f, 0.25f, 1.0f) + bodyW.g * float3(0.10f, 0.90f, 0.15f)
                    + bodyW.b * float3(1.0f, 0.12f, 0.10f) + bodyW.a * float3(1.0f, 0.90f, 0.10f);
                if (dot(bodyW, float4(1.0f, 1.0f, 1.0f, 1.0f)) < 0.5f) { dbg = float3(0.3f, 0.3f, 0.3f); }
            }
            else
            {
                const float h = atan2(flowDir.y, flowDir.x) * 0.1591549f + 0.5f;
                const float3 hue = saturate(float3(abs(h * 6.0f - 3.0f) - 1.0f, 2.0f - abs(h * 6.0f - 2.0f),
                                                   2.0f - abs(h * 6.0f - 4.0f)));
                const float str = saturate(flowStr * 2.0f);
                const float2 wxy = In.RestXY + gFrameData.lodEye.xy;
                const float stripe = smoothstep(0.35f, 0.5f, abs(frac(dot(wxy, flowDir) / 256.0f - time * 0.5f) - 0.5f) * 2.0f);
                dbg = (dot(flowDir, flowDir) > 1.0e-4f) ? hue * (0.25f + 0.75f * str) * (0.6f + 0.4f * stripe)
                                                        : float3(0.25f, 0.25f, 0.25f);
            }
            result = dbg * (0.75f + 0.25f * saturate(normal.z * 8.0f - 7.0f));
            kDst = 0.0f;
        }
    }
#line 2960 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
	if (!noFogMelt) {
	    float E = fogExtinction(fog, 1.0f);
	    result *= E;
	    kDst *= E;
	    result = result * fog
	            + fogSkyColorAt(In.Position.xy, fog, In.WorldPos) * ((1.0f - fog) * (1.0f - kDst));
	}
#line 3020 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
	float2 offPx = (1.0f - refrOccl) * reffactorR;
	float wOwn = saturate(gShadowParams.waterDistort.z)
	              * smoothstep(0.5f, 2.0f, length(offPx));
	float kDstBl = kDst * (1.0f - wOwn);
	result += kDst * (refrDist - (1.0f - wOwn) * refrFlat);
#line 3043 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
	if (destProbe) { RETURN(float4(-kDst * refrFlat, 1.0f - kDst)); }
#line 3055 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
	if (flatProbe) { RETURN(float4(refrFlat, 1.0f)); }





	return (float4(result, 1.0f - kDstBl));
}
#line 409 "FSL/shaders.list"
