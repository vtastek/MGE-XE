// mgeHost64 — the IUpscaler seam's first backend.  tasks/forge-upscale.md M1 step 4b.
//
// Why this file exists at all, what the seam's contract is, and why the unread fields on
// UpscaleInputs are deliberately there: upscale.h. This file is only the PASSTHROUGH backend.
//
// ⚠ IT OWNS NOTHING THE RENDERER OWNS. No `g_live`, no renderer globals, no knobs — every input
// arrives as a parameter. That is the property that makes 4d a drop-in: the NGX backend is another
// implementation of the same six calls, not a second set of hooks into forgerender.cpp.
//
// ABI: this TU is built with The Forge's settings (no exceptions / no RTTI / _HAS_EXCEPTIONS=0), the
// same as forgerender.cpp, because it talks to IGraphics/IResourceLoader directly. Keep host STL out
// of it for the same reason.

#include "upscale.h"

#include <cstdio>
#include <cstring>

#include "support/log.h"   // LOG::logline -> mgeHost64.log

#include "OS/Interfaces/IOperatingSystem.h"
#include "Graphics/GraphicsConfig.h"
#include "Graphics/Interfaces/IGraphics.h"
#include "Resources/ResourceLoader/Interfaces/IResourceLoader.h"
// IMemory.h overrides new/delete/malloc — Forge convention: include it LAST, and it is what makes
// destroyUpscaler() a function in this file rather than a `delete` at the call site.
#include "Utilities/Interfaces/IMemory.h"
// defaults.h provides the C++ definitions of the FSL macros (STRUCT / DATA / BEGIN_SRT /
// DECL_CBUFFER / SRT_SET_DESC / SRT_RES_IDX); the .srt.h then declares SRT_UpscaleSrtData and the
// descriptor indices. Per the Forge convention (06_MaterialPlayground) these come AFTER IMemory.h.
#include "Graphics/FSL/defaults.h"
#include "shaders/FSL/upscale.srt.h"

namespace
{
    // ⚠ THE TWO FILTER PARAMETERS MOVED ONTO UpscaleInputs (M1 4c follow-up) and are no longer
    // constants here. The note they used to carry said a filter parameter "is a thing to walk once
    // the seam is known-correct"; 4b closed and 4c verified, and bloom now reads THIS PASS'S OUTPUT,
    // so the anti-ringing clamp became a question rather than a setting. They arrive as parameters
    // because this file owns nothing the renderer owns — see upscale.h.

    class PassthroughUpscaler final : public IUpscaler
    {
    public:
        bool init(Renderer* pRenderer, uint32_t allocW, uint32_t allocH,
                  TinyImageFormat colorFmt) override
        {
            if (!pRenderer || allocW == 0u || allocH == 0u)
            {
                return false;
            }
            mAllocW = allocW;
            mAllocH = allocH;

            // THE OUTPUT TARGET. ALLOC-sized, following pAO / pLinearDepth rather than a
            // RenderTarget, because the pass is COMPUTE — it needs a UAV, not an RTV, and a Forge
            // RenderTarget would drag an RTV and a render-pass bind it never uses.
            //
            // ⚠ ALLOC-SIZED FOR THE REASON EVERY OTHER SCREEN TARGET IS: a live render-scale change
            // must need no reallocation. It costs ~56 MB at the 3360x2100 allocation, which is why
            // the host only creates this backend when the feature is actually enabled — an
            // allocation for a path that is gated off is pure waste (the same call the bloom pyramid
            // makes for its ~33 MB).
            //
            // ⚠⚠ ALWAYS fp16, NEVER `colorFmt` blindly — the argument is taken so a future backend
            // CAN match the scene target, but this one must not: the source is scene-referred and
            // unbounded, so a UNORM destination would clamp every value above 1.0 and the upscaled
            // frame would arrive at the resolve already tonemapped-by-clipping. Same two reasons
            // reflectmip.srt.h and bloom.srt.h give — plus B8G8R8A8_UNORM is not in D3D12's
            // TypedUAVLoadAdditionalFormats set, so it could not be a UAV destination at LDR anyway.
            (void)colorFmt;

            TextureDesc td = {};
            td.mWidth = allocW;
            td.mHeight = allocH;
            td.mDepth = 1;
            td.mArraySize = 1;
            td.mMipLevels = 1;
            td.mSampleCount = SAMPLE_COUNT_1;   // an upscaler REPLACES MSAA; the host refuses above 1x
            td.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
            // SHADER_RESOURCE at rest, pHiz's convention: evaluate() does SR -> UAV, dispatch,
            // UAV -> SR, so the state is balanced every frame and the FIRST frame needs no special
            // case. Starting in UNORDERED_ACCESS instead would make frame 0's pre-dispatch barrier a
            // lie, and D3D12 rejects a transition whose before-state does not match.
            td.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
            td.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
            td.pName = "upscaleOutput";
            TextureLoadDesc tld = {};
            tld.ppTexture = &mOutput;
            tld.pDesc = &td;
            addResource(&tld, nullptr);
            waitForAllResourceLoads();
            if (!mOutput)
            {
                LOG::logline("!! [upscale] output target %ux%u RGBA16F FAILED to allocate — backend DECLINED",
                             allocW, allocH);
                shutdown(pRenderer);
                return false;
            }

            BufferLoadDesc cb = {};
            cb.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            cb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            cb.mDesc.mFlags       = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            cb.mDesc.mSize        = 256;   // cbuffer alignment, not sizeof(UpscaleParams)
            cb.mDesc.pName        = "upscaleParams";
            cb.ppBuffer           = &mParamsCbv;
            addResource(&cb, nullptr);
            waitForAllResourceLoads();
            if (!mParamsCbv || !mParamsCbv->pCpuMappedAddress)
            {
                LOG::logline("!! [upscale] params cbuffer FAILED — backend DECLINED");
                shutdown(pRenderer);
                return false;
            }

            if (!buildPipeline(pRenderer))
            {
                LOG::logline("!! [upscale] upscale.comp FAILED to load — backend DECLINED"
                             " (dxil missing on disk?)");
                shutdown(pRenderer);
                return false;
            }

            DescriptorSetDesc sd = SRT_SET_DESC(UpscaleSrtData, Persistent, 1, 0);
            addDescriptorSet(pRenderer, &sd, &mSet);
            if (!mSet)
            {
                LOG::logline("!! [upscale] descriptor set FAILED — backend DECLINED");
                shutdown(pRenderer);
                return false;
            }
            return true;
        }

        // ONE descriptor write, at path-build time, on the host thread, outside command recording —
        // where every other bind in this renderer happens. The colour texture is REMEMBERED so
        // evaluate() can refuse a mismatch rather than sample a stale descriptor; see upscale.h.
        bool bindInputs(Renderer* pRenderer, const UpscaleInputs& inputs) override
        {
            if (!pRenderer || !mSet || !mOutput || !mParamsCbv || !inputs.pColor)
            {
                return false;
            }
            // A LOCAL COPY of the source pointer, because `inputs` is const and ppTextures is a
            // non-const Texture**. updateDescriptorSet reads through it immediately, so a stack
            // temporary is correct — the same shape `bloomTex` uses at the resolve's own bind site.
            Texture* srcTex = inputs.pColor;
            DescriptorData d[3] = {};
            d[0].mIndex     = SRT_RES_IDX(UpscaleSrtData, Persistent, gUpscaleParams);
            d[0].ppBuffers  = &mParamsCbv;
            d[1].mIndex     = SRT_RES_IDX(UpscaleSrtData, Persistent, gUpscaleSrc);
            d[1].mCount     = 1;   // REQUIRED for a single texture — an unset count binds nothing
            d[1].ppTextures = &srcTex;
            d[2].mIndex     = SRT_RES_IDX(UpscaleSrtData, Persistent, gUpscaleDst);
            d[2].mCount     = 1;
            d[2].ppTextures = &mOutput;
            updateDescriptorSet(pRenderer, 0, mSet, 3, d);
            mBoundColor = inputs.pColor;
            return true;
        }

        // 4b: nothing to do. The target is ALLOC-sized and the output rect rides the cbuffer, so a
        // live render-scale change costs one float. 4d is where the NGX feature is torn down and
        // rebuilt against the new output extent, which is why this is a call and not a field.
        void resize(uint32_t outW, uint32_t outH) override
        {
            (void)outW;
            (void)outH;
        }

        Texture* evaluate(Cmd* pCmd, const UpscaleInputs& inputs) override
        {
            // ⚠ THE IDENTITY FAST PATH, AND IT IS THE WHOLE REASON SCALE 1.0 IS FREE. Returning the
            // source having recorded NOTHING makes `in == out` bit-identical by CONSTRUCTION rather
            // than by measurement — no dispatch, no barrier, no filter that has to be proven to be
            // an identity. The consequence, spelled out because forgetting it makes a 4b test
            // meaningless: **scale 1.0 proves nothing about the seam.** Test at 0.5.
            if (!ready() || !pCmd || !inputs.pColor)
            {
                return inputs.pColor;
            }
            if (inputs.inW == inputs.outW && inputs.inH == inputs.outH)
            {
                return inputs.pColor;
            }
            // A colour texture other than the one bindInputs() wrote into the set would be sampled
            // through a stale descriptor — silently, and as a plausible-looking image. Refuse.
            if (inputs.pColor != mBoundColor)
            {
                LOG::logline("!! [upscale] colour texture changed since bindInputs — pass SKIPPED"
                             " this frame (rebind required)");
                return inputs.pColor;
            }
            if (inputs.inW == 0u || inputs.inH == 0u || inputs.outW == 0u || inputs.outH == 0u)
            {
                return inputs.pColor;
            }
            // alloc >= out >= in is the model, and the target really is only allocW x allocH — a
            // dispatch past that would write outside the surface. The host's setRenderSize already
            // clamps to the allocation, so reaching this is a contract violation rather than a
            // configuration: say so once and decline, instead of scribbling.
            if (inputs.outW > mAllocW || inputs.outH > mAllocH || inputs.inW > inputs.outW
                || inputs.inH > inputs.outH)
            {
                if (!mRectComplained)
                {
                    mRectComplained = true;
                    LOG::logline("!! [upscale] rect violates alloc >= out >= in "
                                 "(in=%ux%u out=%ux%u alloc=%ux%u) — pass DECLINED",
                                 inputs.inW, inputs.inH, inputs.outW, inputs.outH,
                                 mAllocW, mAllocH);
                    LOG::flush();
                }
                return inputs.pColor;
            }

            {
                float* p = (float*)mParamsCbv->pCpuMappedAddress;
                p[0]  = (float)inputs.inW;
                p[1]  = (float)inputs.inH;
                p[2]  = 1.0f / (float)inputs.inW;
                p[3]  = 1.0f / (float)inputs.inH;
                p[4]  = (float)inputs.outW;
                p[5]  = (float)inputs.outH;
                p[6]  = 1.0f / (float)inputs.outW;
                p[7]  = 1.0f / (float)inputs.outH;
                p[8]  = inputs.sharpness;
                p[9]  = inputs.antiRing;
                p[10] = 0.0f;
                p[11] = 0.0f;
            }

            // ⚠ THE SOURCE'S STATE IS THE HOST'S PROBLEM, NOT THIS BACKEND'S. pSceneColor is already
            // in SHADER_RESOURCE when this runs — the host's `sceneColorToSR()` hoist put it there,
            // once, for the three readers (upscale, bloom, resolve). A barrier here would be a
            // second, redundant transition on a very large surface, and on hardware that
            // decompresses that is not free.
            cmdBeginDebugMarker(pCmd, 0.4f, 0.8f, 1.0f, "UPSCALE (Catmull-Rom + anti-ring)");
            TextureBarrier tb = {};
            tb.pTexture      = mOutput;
            tb.mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
            tb.mNewState     = RESOURCE_STATE_UNORDERED_ACCESS;
            cmdResourceBarrier(pCmd, 0, nullptr, 1, &tb, 0, nullptr);

            cmdBindPipeline(pCmd, mPipeline);
            cmdBindDescriptorSet(pCmd, 0, mSet);
            cmdDispatch(pCmd, (inputs.outW + 7u) / 8u, (inputs.outH + 7u) / 8u, 1);

            tb.mCurrentState = RESOURCE_STATE_UNORDERED_ACCESS;
            tb.mNewState     = RESOURCE_STATE_SHADER_RESOURCE;   // the resolve reads it as an SRV
            cmdResourceBarrier(pCmd, 0, nullptr, 1, &tb, 0, nullptr);
            cmdEndDebugMarker(pCmd);

            return mOutput;
        }

        Texture* outputTexture() const override { return mOutput; }

        // F8. The TARGET and the descriptor set survive — only the shader and the pipeline are
        // rebuilt — so a failed reload leaves the pass inert rather than dangling, and the host's
        // pointer comparison then simply never sees an upscaled frame. Same shape as bloom /
        // reflectmip / motionvectors.
        bool reload(Renderer* pRenderer) override
        {
            if (!pRenderer)
            {
                return false;
            }
            if (mPipeline) { removePipeline(pRenderer, mPipeline); mPipeline = nullptr; }
            if (mShader)   { removeShader(pRenderer, mShader);     mShader = nullptr; }
            const bool ok = buildPipeline(pRenderer);
            LOG::logline(ok ? ">> [upscale] upscale.comp hot-reloaded"
                            : "!! [upscale] hot-reload FAILED — pass DISABLED (dxil missing on disk?)");
            LOG::flush();
            return ok && ready();
        }

        // ⚠ SET -> PIPELINE -> SHADER -> BUFFER -> TEXTURE, the order the rest of this renderer tears
        // down in, and the texture goes LAST because the descriptor set references it.
        void shutdown(Renderer* pRenderer) override
        {
            if (!pRenderer)
            {
                return;
            }
            if (mSet)       { removeDescriptorSet(pRenderer, mSet); mSet = nullptr; }
            if (mPipeline)  { removePipeline(pRenderer, mPipeline); mPipeline = nullptr; }
            if (mShader)    { removeShader(pRenderer, mShader);     mShader = nullptr; }
            if (mParamsCbv) { removeResource(mParamsCbv);           mParamsCbv = nullptr; }
            if (mOutput)    { removeResource(mOutput);              mOutput = nullptr; }
            mBoundColor = nullptr;
        }

        const char* name() const override { return "passthrough"; }

    private:
        bool ready() const
        {
            return mOutput && mParamsCbv && mParamsCbv->pCpuMappedAddress && mPipeline && mSet
                && mBoundColor;
        }

        bool buildPipeline(Renderer* pRenderer)
        {
            ShaderLoadDesc sd = {};
            sd.mComp.pFileName = "upscale.comp";
            addShader(pRenderer, &sd, &mShader);
            if (!mShader)
            {
                return false;
            }
            PipelineDesc pd = {};
            pd.mType = PIPELINE_TYPE_COMPUTE;
            pd.mComputeDesc.pShaderProgram = mShader;
            addPipeline(pRenderer, &pd, &mPipeline);
            return mPipeline != nullptr;
        }

        Texture*       mOutput = nullptr;
        Buffer*        mParamsCbv = nullptr;
        Shader*        mShader = nullptr;
        Pipeline*      mPipeline = nullptr;
        DescriptorSet* mSet = nullptr;
        // What bindInputs() actually wrote into the set. evaluate() compares against it rather than
        // trusting the caller — see the refusal there.
        Texture*       mBoundColor = nullptr;
        uint32_t       mAllocW = 0, mAllocH = 0;
        // The rect check above fires per FRAME if it fires at all, and a per-frame log line is how a
        // 300-frame heartbeat gets buried. Once is enough to find it.
        bool           mRectComplained = false;
    };
}

IUpscaler* createPassthroughUpscaler()
{
    // tf_new, not new: IMemory.h poisons `new` in this TU, and the object has to come off the same
    // allocator destroyUpscaler() gives it back to.
    return tf_new(PassthroughUpscaler);
}

void destroyUpscaler(IUpscaler* pUpscaler)
{
    if (pUpscaler)
    {
        // Virtual destructor, so this dispatches to the concrete backend before freeing. The
        // allocation and the free live in one file for exactly that reason — a `delete` at the call
        // site would not even compile past IMemory.h, and one that did would be on the wrong heap.
        tf_delete(pUpscaler);
    }
}
