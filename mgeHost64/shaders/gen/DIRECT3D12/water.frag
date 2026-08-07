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
#line 244 "FSL/shaders.list"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
#line 49 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
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
#line 236
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




        CBUFFER(LightData) gLights :  register(b0,space3);










        CBUFFER(LightData) gLightsNear :  register(b1,space3);
#line 435 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/opaque.srt.h"
        Tex2D(float4) gTextures[ 880 ] :  register(t0,space0);



        Tex2DArray(float4) gStaticsArrays[ 128 ] :  register(t880,space0);



        Tex2DArray(float4) gFlipArrays[ 16 ] :  register(t1008,space0);
        CBUFFER(BatchData) gBatch :  register(b0,space2);
#line 50 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
#line 69 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
float4 fogSkySample(float2 pixelXy)
{
    return SampleLvlTex2D(gSkyColor, gSamplerBilinearClamp, pixelXy * gFrameData.fogParams.zw, 0);
}







float3 fogSkyTarget(float4 s)
{
    float3 sky = s.rgb + gFrameData.fogColNear.rgb * (1.0f - s.a);
    return lerp(gFrameData.fogColNear.rgb, sky, gFrameData.fogColNear.w);
}


float3 fogSkyColor(float2 pixelXy)
{
    return fogSkyTarget(fogSkySample(pixelXy));
}
#line 103 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
float3 applyFog(float3 lit, float3 worldPosRel, float2 pixelXy, float fog)
{
    float4 s = fogSkySample(pixelXy);









    fog = saturate(fog);
#line 149 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
    float upness = worldPosRel.z * rsqrt(max(dot(worldPosRel, worldPosRel), 1.0e-12f));
    float skyBehind = max(s.a, saturate(upness *  38.0f ));
#line 171 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/skydome.h.fsl"
    float ramp = pow(1.0f - fog, gFrameData.timeParams.w);
    lit *= 1.0f - gFrameData.skyParams.w * ramp * skyBehind;

    return lerp(fogSkyTarget(s), lit, fog);
}
#line 51 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
#line 1 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/tonemap.h.fsl"
#line 32 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/tonemap.h.fsl"
float3 tonemap(float3 c)
{
    c = clamp(c, 0.0f, 2.2f);
    c = (((0.0548303f * c - 0.189786f) * c - 0.154732f) * c + 1.12969f) * c;
    return c;
}
#line 52 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
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
#line 53 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
#line 99 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
STRUCT(VSOutput)
{
    DATA(float4, Position, SV_Position);
    DATA(float3, WorldPos, TEXCOORD0);
#line 103
};



float3 reconstructWorld(float4x4 invVP, float2 uv, float deviceZ)
{
    float ndcX = uv.x * 2.0f - 1.0f;
    float ndcY = 1.0f - uv.y * 2.0f;
    float4 p = mul(invVP, float4(ndcX, ndcY, deviceZ, 1.0f));
    return p.xyz / p.w;
}

[RootSignature( "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "3" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "3" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "3" ", offset = 0))," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "2" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "2" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "2" ", offset = 0))," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "1" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "1" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "1" ", offset = 0))," "DescriptorTable(" "SRV(t0, numDescriptors = unbounded, space = " "0" ", offset = 0)," "CBV(b0, numDescriptors = unbounded, space = " "0" ", offset = 0)," "UAV(u0, numDescriptors = unbounded, space = " "0" ", offset = 0))," "DescriptorTable(" "SAMPLER(s0, numDescriptors = unbounded, space = " "0" ", offset = 0))," "StaticSampler(s0, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s1, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s2, space = 100," "filter = FILTER_MIN_MAG_LINEAR_MIP_POINT," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s3, space = 100," "filter = FILTER_MIN_MAG_LINEAR_MIP_POINT," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s4, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s5, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s6, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT," "addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR, addressW = TEXTURE_ADDRESS_MIRROR)," "StaticSampler(s7, space = 100," "filter = FILTER_MIN_MAG_MIP_POINT, borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK," "addressU = TEXTURE_ADDRESS_BORDER, addressV = TEXTURE_ADDRESS_BORDER, addressW = TEXTURE_ADDRESS_BORDER)," "StaticSampler(s8, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR," "addressU = TEXTURE_ADDRESS_MIRROR, addressV = TEXTURE_ADDRESS_MIRROR, addressW = TEXTURE_ADDRESS_MIRROR)," "StaticSampler(s9, space = 100," "filter = FILTER_MIN_MAG_MIP_LINEAR, borderColor = STATIC_BORDER_COLOR_TRANSPARENT_BLACK," "addressU = TEXTURE_ADDRESS_BORDER, addressV = TEXTURE_ADDRESS_BORDER, addressW = TEXTURE_ADDRESS_BORDER)," "StaticSampler(s10, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s11, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s12, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s13, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 8," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s14, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s15, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_CLAMP)," "StaticSampler(s16, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_WRAP, addressW = TEXTURE_ADDRESS_WRAP)," "StaticSampler(s17, space = 100," "filter = FILTER_ANISOTROPIC, maxAnisotropy = 2," "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_CLAMP, addressW = TEXTURE_ADDRESS_WRAP)" )]
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



    uint waterFlags = (uint)(P[3].x + 0.5f);
    uint waterDbg = waterFlags & 3u;
    bool useSchlick = (waterFlags & 4u) != 0u;



    bool noFogMelt = (waterFlags & 8u) != 0u;

    bool proceduralWaves = (waterFlags & 16u) != 0u;



    bool heightView = (waterFlags & 32u) != 0u;


    float4 waveG1 = transpose(gBatch.worlds[10])[1];
    float heightGain = waveG1.x;


    float waveAmp = waveG1.y;



    float waveSlices = max(waveG1.z, 1.0f);
    float waveTile = max(waveG1.w, 1.0f);









    float alphaBase = P[3].y;
    float reflSize = P[3].z;
    float reflBlurGain = P[3].w;
#line 184 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float2 invAlloc = gShadowParams.screenAlloc.zw;
    float2 invScreen = gShadowParams.screenParams.zw;
    float2 texToVp = gShadowParams.screenAlloc.xy * gShadowParams.screenParams.zw;
    float3 fogCol = gFrameData.fogColNear.rgb;


    float3 EyeVec = In.WorldPos;
    float dist = length(EyeVec);
    EyeVec = (dist > 1e-4f) ? EyeVec / dist : float3(0, 0, 1);


    float fog = saturate((gFrameData.fogParams.y - dist)
                       / (gFrameData.fogParams.y - gFrameData.fogParams.x));



    float2 worldXY = In.WorldPos.xy + gFrameData.lodEye.xy;









    float t = 0.4f * time;
    float2 tc2 = worldXY / waveTile;
#line 244 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float2 dWdx = ddx(In.WorldPos.xy);
    float2 dWdy = ddy(In.WorldPos.xy);








    float4 fpEll = footprintEllipse(dWdx, dWdy);




    float pixAngle = max(length(ddx(EyeVec)), length(ddy(EyeVec)));
#line 272 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float3 normal = float3(0.0f, 0.0f, 1.0f);
#line 302 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
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
#line 327 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
rg - 1.0f;
        float2 s1 = 2.0f *  gWaterNormalVol.SampleGrad(gSamplerAnisotropic, float3(tc2, z1), dTdx, dTdy) .
#line 329 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
rg - 1.0f;
        normal = normalize(float3(lerp(s0, s1, fz) * waveAmp, 1.0f));
#line 369 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
        float wTap = max(sMinW, sMaxW / (float) 8 );
        float lodA = log2(wTap / texelWorld);
        float lodV = 0.5f * (lodA + log2(sMaxW / texelWorld));
#line 383 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
        float lodS = max(lodV - 1.0f, 0.0f);
        float vr =  gWaterSlopeVar.SampleLevel(gSamplerTrilinearWrap, float3(tc2, z0), lodS) .r;
        afSigma2 = max(0.0f, vr) * min(lodV, 1.0f) * (waveAmp * waveAmp);
    }
#line 400 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float waveSigma2 = 0.0f;
    float3 waveHG = float3(0.0f, 0.0f, 0.0f);
    if (proceduralWaves || heightView) {
        waveHG = waveFieldGrad(In.WorldPos.xy, gFrameData.lodEye.xy, time, fpEll, waveSigma2);
    }




    if (heightView) { RETURN(float4(waveHeightDebug(waveHG, heightGain, dWdx, dWdy), 1.0f)); }

    if (proceduralWaves) {



        normal = normalize(float3(-waveHG.yz, 1.0f));
    }
#line 452 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float sigma2 = proceduralWaves ? waveSigma2 : afSigma2;





    float alpha = sqrt(alphaBase * alphaBase + sigma2);
#line 475 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float reflTexelRad = max(pixAngle * (gShadowParams.screenParams.x / max(reflSize, 1.0f)), 1e-7f);
    float reflLodRaw = log2(max(reflBlurGain * 2.0f * alpha / reflTexelRad, 1.0f));
#line 492 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float reflLod = min(reflLodRaw, (float) 3 );
#line 510 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    float over = max(reflLodRaw - (float) 3 , 0.0f);
    float reflSkyMix = 1.0f - exp2(-over);

    float2 baseUV = In.Position.xy * invAlloc;








    if (underwater) {
        float uwFog = saturate(exp(-dist / 4096.0f));
        float3 nrm = -normal;


        float2 reffactorU = 2.0f * (windFactor * dist + 0.1f) * nrm.xy;


        float2 ruvU = baseUV + (-2.0f * reffactorU) * invAlloc;
        float3 refractedU = SampleLvlTex2D(gRefractColor, gSamplerBilinearClamp, ruvU, 0.0f).rgb;
        refractedU = lerp(fogCol, refractedU, exp(-dist / 500.0f));







        float2 reflUVU = In.Position.xy * invScreen
                       + float2(-2.1f * reffactorU.x, abs(reffactorU.y)) * invScreen;
        float4 reflSampleU = SampleLvlTex2D(gReflectMips, gSamplerTrilinearClamp, reflUVU, reflLod);
        float3 reflectedU = reflSampleU.rgb + fogCol * (1.0f - reflSampleU.a);




        if (reflSkyMix > 0.0f) { reflectedU = lerp(reflectedU, fogCol, reflSkyMix); }

        if (waterDbg == 1u) {
            float4 rawReflU = SampleLvlTex2D(gReflectColor, gSamplerBilinearClamp,
                                             In.Position.xy * invScreen, 0.0f);
            return (float4(rawReflU.rgb, 1.0f));
        }
        if (waterDbg == 2u) { RETURN(float4(refractedU, 1.0f)); }



        float fresnelU = pow(saturate(1.12f - 0.65f * dot(-EyeVec, nrm)), 8.0f);
        float3 resultU = lerp(refractedU, reflectedU, fresnelU);


        float3 sunPosU = -gFrameData.sunDir.xyz;
        float refractsun = saturate(dot(-EyeVec, normalize(-sunPosU + nrm)));
        resultU += gFrameData.sunCol.rgb * pow(refractsun, 6.0f) * uwFog;

        return (float4(resultU, 1.0f));
    }


    float3 depthColor = depthBaseColor * float3(0.1, 0.3, 1.0) * 0.5;


    float2 reffactor = (windFactor * dist + 0.1f) * normal.xy;


    float2 distUV = baseUV + reffactor.yx * invAlloc;
    float sceneDevZ= SampleLvlTex2D(gSceneLinDepth, gSamplerPointClamp, distUV, 0.0f).r;
    float3 sceneW = reconstructWorld(invVP, distUV * texToVp, sceneDevZ);
    float sceneDist= length(sceneW);
    float aboveWater = step(sceneDist + shoreDepthBias, dist);
    float depth = max(shoreDepthBias, sceneDist - dist);

    float3 refracted = depthColor;
    float shorefactor = 0.0f;
    if (depth < 4000.0f && aboveWater < 0.5f) {
        float2 ruv = baseUV + saturate(depth / 100.0f) * reffactor.yx * invAlloc;
        refracted = SampleLvlTex2D(gRefractColor, gSamplerBilinearClamp, ruv, 0.0f).rgb;


        sceneDevZ = SampleLvlTex2D(gSceneLinDepth, gSamplerPointClamp, ruv, 0.0f).r;
        sceneW = reconstructWorld(invVP, ruv * texToVp, sceneDevZ);
        sceneDist = length(sceneW);
        depth = max(shoreDepthBias, sceneDist - dist);
        float denom = max(abs(dot(EyeVec, camFwd)), 0.25f);
        depth /= denom;
        depth += 300.0f * (0.95f - normal.z);

        float depthscale = saturate(exp(-depth / 500.0f));
        shorefactor = pow(depthscale, 5.0f);
        refracted = lerp(depthColor, refracted, 0.8f * depthscale + 0.2f * shorefactor);
    }


	float3 sunPos = -gFrameData.sunDir.xyz;
    float3 L = normalize(sunPos);
    float3 V = -EyeVec;
    float3 H = normalize(L + V);
    float NoL = saturate(dot(normal, L));
    float NoH = saturate(dot(normal, H));
    float VoH = saturate(dot(V, H));

	refracted = lerp(refracted, ((1.0 - VoH) * 0.3 + 0.2) * gFrameData.sunCol.r * float3(0.2, 0.9, 0.7) * 0.5, 0.5);









    float2 reflUV = In.Position.xy * invScreen
                  + float2(-2.1f * reffactor.x, abs(reffactor.y)) * invScreen;
    float4 reflSample = SampleLvlTex2D(gReflectMips, gSamplerTrilinearClamp, reflUV, reflLod);
    float3 reflected = reflSample.rgb + fogCol * (1.0f - reflSample.a);



    if (reflSkyMix > 0.0f) {
        reflected = lerp(reflected, fogSkyColor(In.Position.xy), reflSkyMix);
    }
#line 657 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    if (!noFogMelt) { reflected = lerp(fogSkyColor(In.Position.xy), reflected, fog); }
#line 669 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    if (waterDbg == 1u) {
        float4 rawRefl = SampleLvlTex2D(gReflectColor, gSamplerBilinearClamp,
                                        In.Position.xy * invScreen, 0.0f);
        return (float4(rawRefl.rgb, 1.0f));
    }
    if (waterDbg == 2u) { RETURN(float4(refracted, 1.0f)); }
#line 693 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
    if (waterDbg == 3u) {
        return (float4(saturate(reflLod / (float) 3 ),
                      saturate(alpha * 20.0f),
                      saturate(log2(afAniso) / log2((float) 8 )), 1.0f));
    }







    float NoV = saturate(dot(-EyeVec, normal));
    float fresnel;
    if (useSchlick) {







        float f90 = max(1.0f - alpha,  0.02f );
        float m = 1.0f - NoV;
        float m5 = m * m; m5 = m5 * m5 * m;
        fresnel =  0.02f  + (f90 -  0.02f ) * m5;
    } else {



        fresnel =  0.02f  + pow(saturate(0.9988f - 0.28f * NoV), 16.0f);
    }
    float3 result = lerp(refracted, reflected, fresnel);
#line 746 "C:/projects/mgexe/MGE-XE/mgeHost64/shaders/FSL/water.frag.fsl"
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
    float F =  0.02f  + (1.0f -  0.02f ) * fm5;



    float sphereNorm = (aSpec * aSpec) / max(a2, 1e-8f);
    float3 spec = gFrameData.sunCol.rgb * (D * Vis * F * NoL * min(sphereNorm, 1.0f));






    result += tonemap(spec) * fog;


    float wdist = dist / lerp(1200.0f, 0.0f, saturate((-waterLevelRel) / 7.0f));
    float wcut = smoothstep(0.09f, 0.1f, wdist);
    float wcutdark = smoothstep(0.0889f, 0.101f, wdist);
    wcutdark = wcutdark * (1.0f - wcutdark);
    wcutdark = saturate(wcutdark * 3.0f);
    result = lerp(refracted, result, wcut);
    result = lerp(result, result * 0.1f, wcutdark);




	result = lerp(fogCol, result, fog);


    result = lerp(result, refracted, shorefactor * fog);

    return (float4(result, 1.0f));
}
#line 245 "FSL/shaders.list"
