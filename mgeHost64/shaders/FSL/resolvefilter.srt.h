// mgeHost64 — MSAA resolve FILTER, compute stage. SRT for resolvefilter.comp.fsl.
//
// This is the load-heavy half of resolve.frag lifted into compute so it can use LDS. It does the
// Catmull-Rom reconstruction and NOTHING else: no exposure, no bloom, no tonemap, no encode, no
// dither. Its output is exactly the value resolve.frag used to hold in `outRgba` right after
// `sum / totalWeight` — the raw filtered PREMULTIPLIED pair, still carrying Catmull-Rom's
// overshoot and undershoot, which is why the target is fp16 and not a UNORM.
//
// WHY IT IS A SEPARATE PASS. LDS is compute-only, and the pass's output cannot be pRT: the shared
// cross-process render target is created with D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET and no UAV
// (forgerender.cpp, the CreateCommittedResource block), on a B8G8R8A8_UNORM whose typed-UAV store
// is not base-guaranteed. So the filter writes an fp16 1x intermediate and resolve.frag reads it
// as gResolveFiltered, keeping every downstream stage — and the premultiplication rules that took
// three attempts to get right — exactly where they are.
//
// The extra intermediate costs one 4.1 MP fp16 write (~33 MB). resolve.frag's read SHRINKS by more
// than that: it was 64 MSAA Loads per pixel and is now one.
//
// Its own SRT rather than a set inside ResolveSrtData: this is a COMPUTE pass on the shared
// ComputeRootSignature, and ResolveSrtData is a graphics SRT on default.rootsig.
#pragma once

// MIRROR of resolve.srt.h's ResolveParams prefix, float for float — the host writes ONE buffer and
// binds it to both. Only the lanes this pass reads are named; the rest are padding here and live
// knobs there. Edit the two together.
STRUCT(RFParams)
{
    DATA(float4, dims, None);   // xy = render w,h ; z = filter diameter ; w = integer sample radius
    DATA(float4, opts, None);   // x = inverse-luminance (firefly) on/off ; z = Mitchell C
};

BEGIN_SRT(ResolveFilterSrtData)
    BEGIN_SRT_SET(PerBatch)
        DECL_CBUFFER(PerBatch, CBUFFER(RFParams), gRFParams)
#if SAMPLE_COUNT > 1
        DECL_TEXTURE(PerBatch, Tex2DMS(float4, SAMPLE_COUNT), gRFSource)
#else
        DECL_TEXTURE(PerBatch, Tex2D(float4), gRFSource)
#endif
        DECL_RWTEXTURE(PerBatch, WTex2D(float4), gRFOut)
    END_SRT_SET(PerBatch)
END_SRT(ResolveFilterSrtData)
