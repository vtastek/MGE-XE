// mgeHost64 — the IUpscaler seam's SECOND backend: NVIDIA DLSS via NGX.  tasks/forge-upscale.md
// M1 step 4d.
//
// Why the seam exists, what its contract is, and why UpscaleInputs carries fields the passthrough
// never read: upscale.h. This file is only the NGX backend, and it is a different implementation of
// the same six calls — NOT a second set of hooks into forgerender.cpp. It owns nothing the renderer
// owns; every input arrives as a parameter.
//
// ─── THE FOUR THINGS 4a-4c ALREADY SETTLED, SO THIS FILE DOES NOT HAVE TO ────────────────────────
//
//  * The motion vectors are ALREADY in DLSS's convention (motionvectors.comp.fsl:84 —
//    `previous - current`, in pixels, +x right / +y down). No conversion pass, so InMVScale stays
//    (1,1) and the only open question is which RESOLUTION the vectors are in — see mvAtInputRect.
//  * Every input already RESTS in RESOURCE_STATE_SHADER_RESOURCE at the upscale point: pSceneColor
//    via the host's `sceneColorToSR()` hoist, pLinearDepth from the seam re-linearize's tail
//    barrier, pMotionVectors / pMvReactive from the MV dispatch's tail. Forge's SHADER_RESOURCE is
//    `NON_PIXEL | PIXEL` (IGraphics.h:164), which is what NGX wants its inputs in. So this file
//    adds exactly ONE pair of barriers, around its own output.
//  * The output is bloom-safe by construction: 4c's prefilter reads the RAW scene, so a temporal
//    reconstruction can never feed an ENERGY operation
//    ([[feedback_energy_op_takes_the_energy_honest_source]]).
//  * The vs2013 static lib links and runs under MSVC 14.44 — checked with a probe TU, not assumed
//    ([[project_dlss_sdk_vendored]]).
//
// ─── ⚠⚠ NO IDENTITY FAST PATH. THIS IS THE ONE PLACE THE TWO BACKENDS MUST DIFFER ───────────────
//
// `PassthroughUpscaler::evaluate` returns the source unchanged when `in == out`, and upscale.h
// presents that as a property of the SEAM. It is not — it is a property of a SPATIAL filter, whose
// output at 1:1 is its input. **DLAA is `in == out` and must still run.** Inheriting the early-out
// would make the very first thing this milestone tries do nothing at all, and it would look exactly
// like a working identity, because a working identity is what it would be.
//
// ─── ⚠⚠ NGX CLOBBERS THE COMMAND LIST'S DESCRIPTOR HEAPS AND ROOT SIGNATURES ────────────────────
//
// This is not in the plan and it is the single thing most likely to turn a working DLSS frame into
// a corrupted rest-of-frame. `NVSDK_NGX_D3D12_EvaluateFeature` runs NGX's own compute work on the
// command list we hand it: it calls SetDescriptorHeaps and SetComputeRootSignature with its own,
// and it does not restore ours. The Forge sets BOTH exactly once, in `beginCmd`
// (Direct3D12.c:5267), and never again — and `cmdBindDescriptorSet` skips a root-table set whose
// handle matches its per-command-list cache (`mBoundDescriptorSets`, Direct3D12.c:4686). So after
// NGX runs, every later pass in the frame binds through the WRONG root signature, against NGX's
// heaps, with Forge convinced it has already bound them.
//
// `restoreForgeCmdState()` below puts all three back, replicating `beginCmd`'s three statements
// exactly. It needs `DescriptorHeap::pHeap`, and `DescriptorHeap` is private to Direct3D12.c — so
// the layout assumption is VERIFIED AT RUNTIME rather than trusted: Forge cached each heap's GPU
// start handle in `mBoundHeapStartHandles` at beginCmd, so reading `pHeap` back through the first
// member and asking the API for its start handle has to reproduce that number. If it does not, this
// backend refuses to run at all rather than corrupting the frame after it
// ([[feedback_guard_fallback_is_the_bug]] — a guard whose fallback is "carry on" is the bug).
//
// ABI: built with The Forge's settings (no exceptions / no RTTI / _HAS_EXCEPTIONS=0), same as
// upscale.cpp and forgerender.cpp. Host STL stays out of it.

#include "upscale.h"

#include <cstdio>
#include <cstdarg>
#include <cstring>

#include "support/log.h"   // LOG::logline -> mgeHost64.log

#include "OS/Interfaces/IOperatingSystem.h"
#include "Graphics/GraphicsConfig.h"
#include "Graphics/Interfaces/IGraphics.h"   // ...and, under DIRECT3D12, d3d12.h
#include "Resources/ResourceLoader/Interfaces/IResourceLoader.h"

// The vendored D3D12 Super Resolution subset (3rdparty/DLSS, v310.7.0). AFTER IGraphics.h so
// d3d12.h is already in — nvsdk_ngx.h forward-declares ID3D12Device/ID3D12GraphicsCommandList and
// the redundant typedefs are only legal against the real ones. BEFORE IMemory.h, which poisons
// new/delete/malloc for everything that follows.
#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers.h"

#include "Utilities/Interfaces/IMemory.h"

namespace
{
    // ─── IDENTITY ────────────────────────────────────────────────────────────────────────────────
    // A project GUID plus an engine version, which is the route NVIDIA documents for an application
    // that has no NVIDIA-issued Application ID (nvsdk_ngx_defs.h, NVSDK_NGX_Application_Identifier:
    // "If your NVIDIA contact did not provide you an ID for this purpose, use ProjectDesc with
    // NVSDK_NGX_ENGINE_TYPE_CUSTOM"). Fixed, not generated per run: it is how NGX recognises this
    // application across sessions for its own driver-side per-app settings.
    const char* const kNgxProjectId    = "a2f0a4c1-7f3e-4d59-9b6b-2c9d1f5e83b4";
    const char* const kNgxEngineVersion = "MGE XE mgeHost64 (The-Forge D3D12)";

    // NGX's own diagnostics, piped into mgeHost64.log. The whole 4b/4d instrument argument in one
    // line: when DLSS declines, the reason exists — it is just written somewhere nobody reads. The
    // callback may be invoked from any thread and must not retain `message`; LOG::logline copies.
    //
    // ⚠ FILE ONLY, unlike this backend's OWN lines below. NVIDIA's runtime is extremely chatty
    // (hundreds of lines per init) and putting that on the console would bury the host's own
    // startup report, which is the thing a person is actually reading.
    void NVSDK_CONV ngxLogCallback(const char* message, NVSDK_NGX_Logging_Level level,
                                   NVSDK_NGX_Feature sourceComponent)
    {
        (void)sourceComponent;
        (void)level;
        if (!message) { return; }
        LOG::logline(">> [ngx] %s", message);
    }

    // ─── THIS BACKEND'S OWN LINES GO TO **BOTH** SINKS ───────────────────────────────────────────
    // ⚠ AND THAT IS NOT BELT-AND-BRACES, IT IS THE ONLY WAY THEY SURVIVE THE ONE RUN THAT MATTERS
    // MOST. `LOG::open` uses CREATE_ALWAYS and The Forge's own LOGF holds a SECOND handle on the
    // same `mgeHost64.log` with an independent file pointer ([[project_forge_scene_probe_truncates_log]]).
    // Under `--forge-scene` the two interleave such that the LOG::logline half of the file is
    // overwritten — measured here: a probe run that reached NGX's CreateFeature left NOT ONE
    // `[upscale-ngx]` line in the file, while NVIDIA's own runtime chatter came through on stdout.
    //
    // A backend whose refusals are invisible in the harness that runs it is the exact defect 4b
    // spent a step fixing at the seam level ("a silently-absent upscaler is indistinguishable from
    // a working one"), so it does not get to reappear one layer down. stdout is what the scene
    // probe and the console both read, and every other bring-up report in this host — the arm line,
    // the AO gate, the format caps — is a `std::printf` for the same reason.
    void ngxLog(const char* fmt, ...)
    {
        char buf[1024];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        LOG::logline("%s", buf);
        LOG::flush();
        std::printf("%s\n", buf);
        std::fflush(stdout);
    }

    const char* ngxResultStr(NVSDK_NGX_Result r)
    {
        switch (r)
        {
        case NVSDK_NGX_Result_Success:                          return "Success";
        case NVSDK_NGX_Result_Fail:                             return "Fail";
        case NVSDK_NGX_Result_FAIL_FeatureNotSupported:         return "FeatureNotSupported";
        case NVSDK_NGX_Result_FAIL_PlatformError:               return "PlatformError";
        case NVSDK_NGX_Result_FAIL_FeatureAlreadyExists:        return "FeatureAlreadyExists";
        case NVSDK_NGX_Result_FAIL_FeatureNotFound:             return "FeatureNotFound";
        case NVSDK_NGX_Result_FAIL_InvalidParameter:            return "InvalidParameter";
        case NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall:       return "ScratchBufferTooSmall";
        case NVSDK_NGX_Result_FAIL_NotInitialized:              return "NotInitialized";
        case NVSDK_NGX_Result_FAIL_UnsupportedInputFormat:      return "UnsupportedInputFormat";
        case NVSDK_NGX_Result_FAIL_RWFlagMissing:               return "RWFlagMissing (the output "
                                                                       "texture has no UAV)";
        case NVSDK_NGX_Result_FAIL_MissingInput:                return "MissingInput";
        case NVSDK_NGX_Result_FAIL_UnableToInitializeFeature:   return "UnableToInitializeFeature";
        case NVSDK_NGX_Result_FAIL_OutOfDate:                   return "OutOfDate (nvngx_dlss.dll "
                                                                       "older than this SDK)";
        case NVSDK_NGX_Result_FAIL_OutOfGPUMemory:              return "OutOfGPUMemory";
        case NVSDK_NGX_Result_FAIL_UnsupportedFormat:           return "UnsupportedFormat";
        case NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath:  return "UnableToWriteToAppDataPath";
        case NVSDK_NGX_Result_FAIL_UnsupportedParameter:        return "UnsupportedParameter";
        case NVSDK_NGX_Result_FAIL_Denied:                      return "Denied (NGX is disabled for "
                                                                       "this app or driver)";
        case NVSDK_NGX_Result_FAIL_NotImplemented:              return "NotImplemented";
        default:                                                return "unknown";
        }
    }

    // ─── THE QUALITY MODE IS A LABEL ON A RECT, AND ONLY THE SDK KNOWS WHICH LABEL FITS ─────────
    // The host authors ONE rect, through `upscaleScale()` -> `setRenderSize()`, and it is the only
    // rect anything renders into. This backend does not get to author a second one; what it does is
    // tell NGX which of its modes the rect it was handed CORRESPONDS to, because DLSS tunes itself
    // per mode.
    //
    // ⚠⚠ THE FIRST VERSION GUESSED THAT FROM DLSS'S PUBLISHED LINEAR RATIOS (DLAA 1.0, Quality
    // 0.667, Balanced 0.58, Performance 0.5, UltraPerformance 0.333) AND PICKED THE NEAREST BY
    // MIDPOINT. In-game 2026-09-02 that produced ELEVEN `OUTSIDE the range DLSS reports` warnings
    // from the check below, and the three shapes it caught say exactly why a ratio table cannot do
    // this job:
    //
    //   DLAA             range=[1663x1040 .. 1680x1050]   ours=1410x882  — DLAA is NOT A BAND. It is
    //                    in == out plus about one percent, so a "> 0.834 is DLAA" threshold is
    //                    wrong for every value it accepts except 1.0.
    //   UltraPerformance range=[ 560x350  ..  560x350 ]   ours=554x346   — NO RANGE AT ALL. min ==
    //                    max == optimal, a FIXED rect; only an exact match is legal.
    //   Performance      range=[ 840x525  .. 1680x1050]   ours=840x524   — MISSES BY ONE PIXEL,
    //                    because setRenderSize rounds to EVEN and DLSS's minimum height here is 525.
    //
    // So the published ratios are the modes' OPTIMAL points, and what a mode will actually ACCEPT is
    // a separate, driver- and version-dependent range that only the SDK can report.
    // [[feedback_prior_art_constants_dont_transfer]] — the warning was written into this file's own
    // 4b-era comment and then walked into anyway.
    //
    // ⚠ THE ANSWER IS NOT TO LET THE SDK AUTHOR THE RECT. It still cannot:
    // `NGX_DLSS_GET_OPTIMAL_SETTINGS` has to be called from the render thread (`setRenderSize` runs
    // on the CLIENT/IPC thread and nvsdk_ngx.h opens with "Methods in this library are NOT thread
    // safe"), so the host stays the single author of the rect. What changes is that the MODE is now
    // read out of the SDK instead of guessed: ask every mode what it accepts, and take the
    // highest-quality one whose range actually CONTAINS the rect we rendered.
    //
    // Called once per feature build — i.e. only when the rects change — not per frame.
    struct QualityPick
    {
        NVSDK_NGX_PerfQuality_Value mode = NVSDK_NGX_PerfQuality_Value_MaxQuality;
        bool                        contained = false;   // did any mode's range accept the rect
    };

    // Highest quality first, so the first containing mode wins. UltraQuality is deliberately absent:
    // it is not implemented by the shipping DLSS runtime and querying it returns Quality's numbers.
    const NVSDK_NGX_PerfQuality_Value kQualityLadder[] = {
        NVSDK_NGX_PerfQuality_Value_DLAA,
        NVSDK_NGX_PerfQuality_Value_MaxQuality,
        NVSDK_NGX_PerfQuality_Value_Balanced,
        NVSDK_NGX_PerfQuality_Value_MaxPerf,
        NVSDK_NGX_PerfQuality_Value_UltraPerformance,
    };

    const char* qualityName(NVSDK_NGX_PerfQuality_Value q)
    {
        switch (q)
        {
        case NVSDK_NGX_PerfQuality_Value_DLAA:             return "DLAA";
        case NVSDK_NGX_PerfQuality_Value_MaxQuality:       return "Quality";
        case NVSDK_NGX_PerfQuality_Value_Balanced:         return "Balanced";
        case NVSDK_NGX_PerfQuality_Value_MaxPerf:          return "Performance";
        case NVSDK_NGX_PerfQuality_Value_UltraPerformance: return "UltraPerformance";
        case NVSDK_NGX_PerfQuality_Value_UltraQuality:     return "UltraQuality";
        default:                                           return "?";
        }
    }

    NVSDK_NGX_DLSS_Hint_Render_Preset ngxPreset(uint32_t p)
    {
        switch (p)
        {
        case kUpscalePresetK: return NVSDK_NGX_DLSS_Hint_Render_Preset_K;
        case kUpscalePresetJ: return NVSDK_NGX_DLSS_Hint_Render_Preset_J;
        case kUpscalePresetL: return NVSDK_NGX_DLSS_Hint_Render_Preset_L;
        case kUpscalePresetM: return NVSDK_NGX_DLSS_Hint_Render_Preset_M;
        default:              return NVSDK_NGX_DLSS_Hint_Render_Preset_Default;
        }
    }

    class NgxUpscaler final : public IUpscaler
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
            mDevice = pRenderer->mDx.pDevice;
            if (!mDevice)
            {
                ngxLog("!! [upscale-ngx] no ID3D12Device on the renderer — backend DECLINED");
                return false;
            }

            // ─── THE OUTPUT TARGET ───────────────────────────────────────────────────────────────
            // Same shape and the same reasoning as the passthrough's (upscale.cpp): ALLOC-sized so a
            // live render-scale change needs no reallocation, ALWAYS fp16 rather than `colorFmt`
            // because the source is scene-referred and unbounded, SHADER_RESOURCE at rest so
            // evaluate's SR -> UAV -> SR is balanced on frame 0 as well as every frame after.
            //
            // ⚠ THE UAV FLAG IS NOT OPTIONAL HERE THE WAY IT IS THERE. NGX writes this resource
            // through an unordered-access view and returns FAIL_RWFlagMissing if the resource was
            // not created with D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS — which is what
            // DESCRIPTOR_TYPE_RW_TEXTURE gets us.
            (void)colorFmt;
            TextureDesc td = {};
            td.mWidth = allocW;
            td.mHeight = allocH;
            td.mDepth = 1;
            td.mArraySize = 1;
            td.mMipLevels = 1;
            td.mSampleCount = SAMPLE_COUNT_1;   // an upscaler REPLACES MSAA; the host refuses above 1x
            td.mFormat = TinyImageFormat_R16G16B16A16_SFLOAT;
            td.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
            td.mDescriptors = (DescriptorType)(DESCRIPTOR_TYPE_TEXTURE | DESCRIPTOR_TYPE_RW_TEXTURE);
            td.pName = "upscaleOutputNgx";

            // ─── ⚠⚠ THE RESOURCE IS CREATED **TYPED**, BY HAND, AND NOT BY THE FORGE ─────────────
            // The Forge makes every texture's D3D12 resource TYPELESS and puts typed views on it
            // (`InitializeTextureDesc`: `desc->Format = DXGI_FORMATToTypeless(dxFormat)`, with the
            // typed format used only when no typeless equivalent exists). That is the right default
            // for a renderer that owns both ends — but this resource is handed to code that does
            // NOT own the views: NGX writes it, and an injected NGX overlay inspects it.
            //
            // Measured 2026-09-02, from the RenoDX DLSS5 addon's own log once it was hooking us:
            //     DLSS output format 9 is not a supported typed codec format
            //     (requires shader sampling and typed UAV support)
            // DXGI format 9 is R16G16B16A16_TYPELESS — a typeless resource carries no format for a
            // third party to build an SRV or a typed UAV from, so the addon can see our output and
            // still refuse to touch it.
            //
            // So the resource is created here as a TYPED R16G16B16A16_FLOAT and imported into a
            // Forge Texture via `pNativeHandle` (Direct3D12.c:3663 takes that path and sets
            // mOwnsImage = false, so Forge builds our SRV/UAV on it and never frees it). Everything
            // downstream — the resolve's descriptor set, the barriers, outputTexture() — is
            // unchanged, because a Forge Texture wrapping an imported resource is still a Forge
            // Texture. ⚠ The consequence of mOwnsImage = false is that removeResource does NOT
            // release the resource, which is why shutdown() releases mOutputRes explicitly.
            {
                D3D12_HEAP_PROPERTIES hp = {};
                hp.Type = D3D12_HEAP_TYPE_DEFAULT;
                D3D12_RESOURCE_DESC rd = {};
                rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                rd.Width = allocW;
                rd.Height = allocH;
                rd.DepthOrArraySize = 1;
                rd.MipLevels = 1;
                rd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;   // ⚠ TYPED, that is the whole point
                rd.SampleDesc.Count = 1;
                rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
                // UAV because NGX writes through one (a missing flag is FAIL_RWFlagMissing); the
                // resource is also sampled by the resolve, which needs no flag of its own.
                rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                const HRESULT hr = mDevice->CreateCommittedResource(
                    &hp, D3D12_HEAP_FLAG_NONE, &rd,
                    // Matches td.mStartState below, so the first frame's SR -> UAV barrier is
                    // truthful. D3D12 rejects a transition whose before-state is a lie.
                    D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE, nullptr,
                    IID_PPV_ARGS(&mOutputRes));
                if (FAILED(hr) || !mOutputRes) {
                    ngxLog("!! [upscale-ngx] typed output CreateCommittedResource FAILED (0x%08lX) "
                           "— backend DECLINED", (unsigned long)hr);
                    shutdown(pRenderer);
                    return false;
                }
                mOutputRes->SetName(L"upscaleOutputNgx (typed RGBA16F)");
                td.pNativeHandle = mOutputRes;
            }

            TextureLoadDesc tld = {};
            tld.ppTexture = &mOutput;
            tld.pDesc = &td;
            addResource(&tld, nullptr);

            // ─── THE EXPOSURE TEXTURE ────────────────────────────────────────────────────────────
            // 1x1 R32_FLOAT holding the scalar E, which is the form DLSS takes it in. It has to be a
            // TEXTURE rather than a float because DLSS reads it on the GPU, inside its own
            // evaluation, and the alternative — NVSDK_NGX_DLSS_Feature_Flags_AutoExposure — is DLSS
            // metering the frame itself. We already have a servo that meters the DELIVERED frame
            // against an authored setpoint (scenecal), and two adaptation loops on one image is one
            // more than the picture can have.
            TextureDesc ed = {};
            ed.mWidth = 1; ed.mHeight = 1; ed.mDepth = 1;
            ed.mArraySize = 1; ed.mMipLevels = 1;
            ed.mSampleCount = SAMPLE_COUNT_1;
            ed.mFormat = TinyImageFormat_R32_SFLOAT;
            ed.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
            ed.mDescriptors = DESCRIPTOR_TYPE_TEXTURE;
            ed.pName = "upscaleExposure";
            TextureLoadDesc eld = {};
            eld.ppTexture = &mExposureTex;
            eld.pDesc = &ed;
            addResource(&eld, nullptr);

            // The staging row it is filled from. An upload-heap buffer is permanently in
            // GENERIC_READ (which includes COPY_SOURCE), so it needs no barrier of its own — the
            // same arrangement `mvStatsReset` uses. 256 bytes because that is
            // D3D12_TEXTURE_DATA_PITCH_ALIGNMENT and therefore the minimum legal row pitch for the
            // placed footprint below, not because four bytes are not enough.
            //
            // ⚠ SINGLE-BUFFERED, and that is safe for the same reason pIndirectArgs is: the host
            // render is lockstep (waitForFences gates the next frame's record against the last
            // frame's completion), so the CPU cannot overwrite a row the GPU has not read.
            BufferLoadDesc eb = {};
            eb.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
            eb.mDesc.mFlags       = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
            eb.mDesc.mSize        = 256;
            eb.mDesc.mStartState  = RESOURCE_STATE_GENERIC_READ;
            eb.mDesc.pName        = "upscaleExposureUpload";
            eb.ppBuffer           = &mExposureUpload;
            addResource(&eb, nullptr);
            waitForAllResourceLoads();

            if (!mOutput || !mExposureTex || !mExposureUpload
                || !mExposureUpload->pCpuMappedAddress)
            {
                ngxLog("!! [upscale-ngx] resource allocation FAILED (out=%d exposure=%d "
                             "upload=%d) — backend DECLINED",
                             mOutput ? 1 : 0, mExposureTex ? 1 : 0, mExposureUpload ? 1 : 0);
                shutdown(pRenderer);
                return false;
            }

            // ─── NGX INIT ────────────────────────────────────────────────────────────────────────
            // ⚠ EVERY REFUSAL BELOW LOGS AND RETURNS FALSE, and a false init leaves the host on the
            // passthrough with not one pixel changed — the guard shape 4b established, and the only
            // shape that is honest here: at DLAA the absent backend and the working one both deliver
            // an image at the output rect, so the log is the ONLY instrument that can tell them
            // apart.
            // ⚠⚠ A MEMBER, NOT A LOCAL, AND THAT IS A LIFETIME REQUIREMENT RATHER THAN A STYLE
            // CHOICE. NGX RETAINS this pointer — the struct carries an `InternalData` field
            // nvsdk_ngx_defs.h labels "Used internally by NGX", i.e. the SDK writes into the caller's
            // storage and reads it back later — so a stack local leaves NGX dereferencing a dead
            // frame from `CreateFeature` onward. That is exactly the shape of the first 4d bring-up
            // failure: init clean, feature created clean, then DXGI_ERROR_DEVICE_HUNG inside the
            // first evaluate, with nothing wrong in any D3D12 call the debug layer could see.
            mCommonInfo = {};
            mCommonInfo.LoggingInfo.LoggingCallback = ngxLogCallback;
            mCommonInfo.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
            mCommonInfo.LoggingInfo.DisableOtherLoggingSinks = false;

            // The application data path is where NGX writes its own logs and caches; it must be
            // WRITABLE. `.` is the Morrowind install directory (main.cpp sets cwd), which is also
            // where mgeHost64.log lives and where nvngx_dlss.dll has to be deployed — NGX's default
            // search path for a feature dll is the application folder.
            NVSDK_NGX_Result r = NVSDK_NGX_D3D12_Init_with_ProjectID(
                kNgxProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, kNgxEngineVersion,
                L".", mDevice, &mCommonInfo);
            if (NVSDK_NGX_FAILED(r))
            {
                ngxLog("!! [upscale-ngx] NVSDK_NGX_D3D12_Init_with_ProjectID FAILED: 0x%08X "
                             "(%s) — backend DECLINED. The usual cause is nvngx_dlss.dll missing "
                             "beside mgeHost64.exe, or a non-NVIDIA adapter.",
                             (unsigned)r, ngxResultStr(r));
                shutdown(pRenderer);
                return false;
            }
            mNgxInited = true;

            // ⚠ GetCapabilityParameters, NOT AllocateParameters. The optimal-settings callback is
            // only published on the capability block (nvsdk_ngx_helpers.h:80 says so in as many
            // words), and the availability flags live nowhere else.
            r = NVSDK_NGX_D3D12_GetCapabilityParameters(&mParams);
            if (NVSDK_NGX_FAILED(r) || !mParams)
            {
                ngxLog("!! [upscale-ngx] GetCapabilityParameters FAILED: 0x%08X (%s) — "
                             "backend DECLINED", (unsigned)r, ngxResultStr(r));
                shutdown(pRenderer);
                return false;
            }

            int available = 0;
            NVSDK_NGX_Parameter_GetI(mParams, NVSDK_NGX_Parameter_SuperSampling_Available,
                                     &available);
            if (!available)
            {
                // The reason, not just the verdict. A "DLSS unavailable" line with no cause is the
                // same non-instrument as a silently-absent upscaler.
                int          initResult = 0;
                int          needsDriver = 0;
                unsigned int major = 0, minor = 0;
                NVSDK_NGX_Parameter_GetI(mParams,
                                         NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult,
                                         &initResult);
                NVSDK_NGX_Parameter_GetI(mParams,
                                         NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver,
                                         &needsDriver);
                NVSDK_NGX_Parameter_GetUI(mParams,
                                          NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor,
                                          &major);
                NVSDK_NGX_Parameter_GetUI(mParams,
                                          NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor,
                                          &minor);
                ngxLog("!! [upscale-ngx] DLSS Super Resolution NOT AVAILABLE on this system — "
                             "backend DECLINED. FeatureInitResult=0x%08X (%s) needsUpdatedDriver=%d "
                             "minDriver=%u.%u",
                             (unsigned)initResult, ngxResultStr((NVSDK_NGX_Result)initResult),
                             needsDriver, major, minor);
                shutdown(pRenderer);
                return false;
            }

            ngxLog(">> [upscale-ngx] NGX initialised, DLSS Super Resolution AVAILABLE. Output "
                         "target %ux%u RGBA16F (alloc-sized, SRV+UAV). ⚠ THE FEATURE ITSELF IS NOT "
                         "CREATED YET — it needs a command list, so it is built at the first "
                         "evaluate() and rebuilt whenever the rects change.",
                         allocW, allocH);
            return true;
        }

        // ─── bindInputs: A RECORD, NOT A BIND ────────────────────────────────────────────────────
        // ⚠ AND THIS IS THE ONE PLACE WHERE THE TWO BACKENDS' CONTRACTS GENUINELY DIFFER, so it is
        // said out loud rather than left to be inferred from an empty function body. The passthrough
        // writes a descriptor set here and REFUSES in evaluate() if it is later handed a different
        // colour texture, because sampling a stale descriptor is silent and looks plausible. NGX has
        // no descriptor set of ours: it takes raw ID3D12Resource pointers per evaluation, so a
        // changed pointer is not a hazard and refusing one would be theatre.
        //
        // What this call is FOR, then, is the ARM LOG — the claim that the resources the backend
        // needs actually exist at the moment it says it is ready. That is why the host had to move
        // the call: pMotionVectors / pMvReactive are created ~1400 lines BELOW where 4b bound the
        // colour, so binding at the old site would have this function announce "motion vectors:
        // NULL" for a backend that cannot run without them.
        bool bindInputs(Renderer* pRenderer, const UpscaleInputs& inputs) override
        {
            (void)pRenderer;
            if (!mOutput || !mExposureTex)
            {
                return false;
            }
            mBoundColor = inputs.pColor;
            mBoundDepth = inputs.pDepth;
            mBoundMv    = inputs.pMotionVectors;
            mBoundMask  = inputs.pReactive;
            ngxLog(">> [upscale-ngx] inputs: colour=%p depth=%p motionVectors=%p reactive=%p",
                         (void*)inputs.pColor, (void*)inputs.pDepth,
                         (void*)inputs.pMotionVectors, (void*)inputs.pReactive);
            // ⚠ COLOUR, DEPTH AND MOTION VECTORS ARE REQUIRED; the reactive mask is not. DLSS
            // treats a null bias mask as "no pixel is reactive", which is a real configuration
            // (it is the mask's own gain-0 arm) rather than a failure.
            if (!inputs.pColor || !inputs.pDepth || !inputs.pMotionVectors)
            {
                ngxLog("!! [upscale-ngx] a REQUIRED input is null (colour/depth/motion "
                             "vectors) — backend DECLINED. A temporal upscaler cannot run without "
                             "all three.");
                return false;
            }
            return true;
        }

        // The output rect changed. 4b's passthrough had nothing to do here; this backend's NGX
        // feature is created AGAINST a pair of rects, so a change means a teardown and a rebuild.
        //
        // ⚠ THIS IS A HINT, NOT THE AUTHORITY. evaluate() re-checks the rects it is actually handed
        // every frame and rebuilds on a mismatch, because it is the only caller that sees the INPUT
        // rect as well as the output one and the only one that runs on the render thread with a
        // command list in hand. The host does not currently call this at all, and does not need to.
        void resize(uint32_t outW, uint32_t outH) override
        {
            if (outW != mFeatOutW || outH != mFeatOutH)
            {
                mFeatureStale = true;
            }
        }

        Texture* evaluate(Cmd* pCmd, const UpscaleInputs& inputs) override
        {
            if (!mParams || !mOutput || !pCmd || !inputs.pColor)
            {
                return inputs.pColor;
            }
            // A hard NGX failure is permanent for the session: it means the driver or the runtime
            // refused, and retrying it 165 times a second would fill the log with the same line and
            // pay the failure's cost every frame.
            if (mDisabled)
            {
                return inputs.pColor;
            }
            // ⚠⚠ THERE IS DELIBERATELY NO `in == out` EARLY-OUT HERE. See the file header: DLAA is
            // in == out, and it is the FIRST configuration this backend is meant to run in.
            if (!inputs.pDepth || !inputs.pMotionVectors)
            {
                complainOnce(&mMissingComplained,
                             "!! [upscale-ngx] depth or motion vectors are null at evaluate — pass "
                             "SKIPPED. The frame is delivered un-upscaled.");
                return inputs.pColor;
            }
            if (inputs.inW == 0u || inputs.inH == 0u || inputs.outW == 0u || inputs.outH == 0u)
            {
                return inputs.pColor;
            }
            if (inputs.outW > mAllocW || inputs.outH > mAllocH || inputs.inW > inputs.outW
                || inputs.inH > inputs.outH)
            {
                char msg[256];
                std::snprintf(msg, sizeof(msg),
                              "!! [upscale-ngx] rect violates alloc >= out >= in (in=%ux%u "
                              "out=%ux%u alloc=%ux%u) — pass DECLINED",
                              inputs.inW, inputs.inH, inputs.outW, inputs.outH, mAllocW, mAllocH);
                complainOnce(&mRectComplained, msg);
                return inputs.pColor;
            }
            // The Forge state restore (see the file header) has to be possible before NGX is allowed
            // to run, not discovered to be impossible afterwards.
            if (!probeForgeCmdState(pCmd))
            {
                mDisabled = true;
                return inputs.pColor;
            }

            const int flags = featureFlags(inputs);
            if (!ensureFeature(pCmd, inputs, flags))
            {
                mDisabled = true;
                return inputs.pColor;
            }

            cmdBeginDebugMarker(pCmd, 0.2f, 0.9f, 0.4f, "UPSCALE (NGX / DLSS)");

            if (inputs.useSuppliedExposure)
            {
                uploadExposure(pCmd, inputs.exposure);
            }

            TextureBarrier tb = {};
            tb.pTexture      = mOutput;
            tb.mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
            tb.mNewState     = RESOURCE_STATE_UNORDERED_ACCESS;
            cmdResourceBarrier(pCmd, 0, nullptr, 1, &tb, 0, nullptr);

            NVSDK_NGX_D3D12_DLSS_Eval_Params ep = {};
            ep.Feature.pInColor  = inputs.pColor->mDx.pResource;
            ep.Feature.pInOutput = mOutput->mDx.pResource;
            // ⚠ SHARPNESS IS NOT PASSED, AND THAT IS SETTLED BY READING RATHER THAN BY TASTE:
            // nvsdk_ngx_defs.h:296 marks NVSDK_NGX_DLSS_Feature_Flags_DoSharpening
            // `SR_DEPRECATED_SHARPENING` with the comment "Sharpness is not supported". So
            // `inputs.sharpness` / `inputs.antiRing` stay what upscale.h says they are — the
            // PASSTHROUGH's spatial filter — and this backend ignores both, the way the passthrough
            // ignores `reset`.
            ep.Feature.InSharpness = 0.0f;
            ep.pInDepth          = inputs.pDepth->mDx.pResource;
            ep.pInMotionVectors  = inputs.pMotionVectors->mDx.pResource;
            ep.pInExposureTexture = inputs.useSuppliedExposure ? mExposureTex->mDx.pResource
                                                               : nullptr;
            // The reprojection-consistency mask (M1 step 3). DLSS's "bias current colour" mask is
            // exactly what it is for: where the mask is 1 the accumulator leans on THIS frame rather
            // than on a history the reprojection cannot justify. Null when the host did not produce
            // one this frame, which DLSS reads as "nothing is reactive" — the mask's own gain-0 arm.
            ep.pInBiasCurrentColorMask = (inputs.pReactive && inputs.useReactiveMask)
                                             ? inputs.pReactive->mDx.pResource : nullptr;
            // ⚠ IN **INPUT** PIXELS, +x right / +y down, and it is the offset that was actually
            // applied to the matrices this frame (g_jitterPx, drawn once per RENDERED frame). Not
            // the sequence index, not an NDC value.
            ep.InJitterOffsetX = inputs.jitterX;
            ep.InJitterOffsetY = inputs.jitterY;
            // ⚠⚠ THE SUB-RECT, AND IT IS WHAT MAKES ALLOC-SIZED TARGETS LEGAL. Every screen resource
            // in this renderer is allocated at the CEILING rect and written only inside the render
            // rect ([[project_forge_alloc_vs_render_uv]]); outside it holds LAST frame's texels.
            // InRenderSubrectDimensions tells DLSS the valid extent, so it never reads the stale
            // border. The bases are all (0,0) because every one of those writes starts at the
            // origin — stated rather than omitted, because a non-zero base here is the difference
            // between a correct frame and a plausible-looking mis-registered one.
            ep.InRenderSubrectDimensions.Width  = inputs.inW;
            ep.InRenderSubrectDimensions.Height = inputs.inH;
            ep.InColorSubrectBase  = { 0, 0 };
            ep.InDepthSubrectBase  = { 0, 0 };
            ep.InMVSubrectBase     = { 0, 0 };
            ep.InBiasCurrentColorSubrectBase = { 0, 0 };
            ep.InOutputSubrectBase = { 0, 0 };
            // 1:1. The vectors are already in the pixel units DLSS wants (motionvectors.comp.fsl:84)
            // — this is the field that would carry a conversion, and there is none to carry.
            ep.InMVScaleX = 1.0f;
            ep.InMVScaleY = 1.0f;
            // ⚠ THE COLOUR IS NOT PRE-EXPOSED. pSceneColor holds scene-referred radiance and E is
            // applied downstream, in resolve.frag. InPreExposure describes a multiplier ALREADY
            // BAKED INTO the colour so DLSS can divide it out of its history; ours is 1 for exactly
            // as long as that stays true. The exposure DLSS should expect is the separate texture
            // above, not this.
            ep.InPreExposure = 1.0f;
            ep.InReset = inputs.reset ? 1 : 0;

            // ⚠ THE FIRST EVALUATE SAYS WHAT IT IS ABOUT TO HAND OVER, once. A device hang inside
            // NGX names no resource, so the only way to attribute one is for the run to have
            // already stated which inputs were live in it — the bisection is worthless if the log
            // cannot say which arm produced it.
            if (!mEvalAnnounced)
            {
                mEvalAnnounced = true;
                ngxLog(">> [upscale-ngx] first evaluate: colour=%p depth=%p mv=%p bias=%p "
                       "exposureTex=%p jitter=(%+.3f, %+.3f) subrect=%ux%u reset=%d "
                       "preExposure=1.0 mvScale=(1,1)",
                       (void*)ep.Feature.pInColor, (void*)ep.pInDepth,
                       (void*)ep.pInMotionVectors, (void*)ep.pInBiasCurrentColorMask,
                       (void*)ep.pInExposureTexture,
                       (double)ep.InJitterOffsetX, (double)ep.InJitterOffsetY,
                       ep.InRenderSubrectDimensions.Width, ep.InRenderSubrectDimensions.Height,
                       ep.InReset);
            }
            const NVSDK_NGX_Result r =
                NGX_D3D12_EVALUATE_DLSS_EXT(pCmd->mDx.pCmdList, mFeature, mParams, &ep);

            // ⚠⚠ BEFORE ANY OTHER FORGE CALL ON THIS COMMAND LIST. NGX left its own descriptor heaps
            // and root signatures bound; the barrier below is a Forge call and every pass after this
            // one is too. See the file header.
            restoreForgeCmdState(pCmd);

            tb.mCurrentState = RESOURCE_STATE_UNORDERED_ACCESS;
            tb.mNewState     = RESOURCE_STATE_SHADER_RESOURCE;   // the resolve reads it as an SRV
            cmdResourceBarrier(pCmd, 0, nullptr, 1, &tb, 0, nullptr);
            cmdEndDebugMarker(pCmd);

            deviceRemovedAt("NGX_D3D12_EVALUATE_DLSS_EXT");
            if (NVSDK_NGX_FAILED(r))
            {
                ngxLog("!! [upscale-ngx] EvaluateFeature FAILED: 0x%08X (%s) — upscaling "
                             "DISABLED for this session; the frame falls back to the raster rect. "
                             "⚠ The input rect is still narrowed, so the picture will be a small "
                             "one in the corner until upscaleInputScale is returned to 1.0.",
                             (unsigned)r, ngxResultStr(r));
                mDisabled = true;
                return inputs.pColor;
            }
            return mOutput;
        }

        Texture* outputTexture() const override { return mOutput; }

        // F8. There is no shader of ours to reload — DLSS's weights live in nvngx_dlss.dll and are
        // loaded by the driver. Returning `ready()` keeps the host's "is this backend still able to
        // run" contract honest without pretending to have rebuilt anything.
        bool reload(Renderer* pRenderer) override
        {
            (void)pRenderer;
            return mParams != nullptr && mOutput != nullptr && !mDisabled;
        }

        // FEATURE -> PARAMETERS -> NGX -> our resources, which is NGX's own documented order (the
        // parameter block is allocated and released by the SDK, so it must outlive the feature that
        // was created through it) followed by this renderer's usual buffer-then-texture teardown.
        void shutdown(Renderer* pRenderer) override
        {
            (void)pRenderer;
            if (mFeature)
            {
                NVSDK_NGX_D3D12_ReleaseFeature(mFeature);
                mFeature = nullptr;
            }
            if (mParams)
            {
                NVSDK_NGX_D3D12_DestroyParameters(mParams);
                mParams = nullptr;
            }
            if (mNgxInited)
            {
                NVSDK_NGX_D3D12_Shutdown1(mDevice);
                mNgxInited = false;
            }
            if (mExposureUpload) { removeResource(mExposureUpload); mExposureUpload = nullptr; }
            if (mExposureTex)    { removeResource(mExposureTex);    mExposureTex = nullptr; }
            if (mOutput)         { removeResource(mOutput);         mOutput = nullptr; }
            // ⚠ EXPLICIT, because the Forge Texture above imported this resource rather than
            // creating it (mOwnsImage = false), so removeResource released the VIEWS and not the
            // memory. Released AFTER the texture, so no descriptor outlives the resource it names.
            if (mOutputRes)      { mOutputRes->Release();               mOutputRes = nullptr; }
            mDevice = nullptr;
            mBoundColor = mBoundDepth = mBoundMv = mBoundMask = nullptr;
        }

        const char* name() const override { return "ngx-dlss"; }

        // ─── WHAT RECT EACH MODE WANTS, STRAIGHT FROM THE SDK ────────────────────────────────────
        // This is the other half of retiring the free slider. The host used to author a rect from a
        // ratio and this file then reported whether DLSS would accept it; now the host asks first
        // and renders a rect DLSS has already said it wants, so the `OUTSIDE the range` warning
        // becomes unreachable through the panel rather than merely visible.
        //
        // ⚠ THE **OPTIMAL** RECT, NOT THE RANGE'S EDGE. Every mode reports optimal/min/max and the
        // optimal is the one its network was tuned at; the range exists for dynamic-resolution
        // engines that need to move within a mode without a rebuild, which this host does not do.
        // Taking min or max would be legal and worse.
        //
        // ⚠ ODD RECTS ARE KEPT, and that is a reversal of the host's own rounding rule. setRenderSize
        // rounds to EVEN so the half-res AO and bloom chains never see a fractional column — but
        // DLSS's optimal height here is 525, and rounding that to 524 is exactly the one-pixel miss
        // that produced a range violation in the first in-game session. Between "the half-res chains
        // handle an odd rect" (they have since 4a, it is stated there) and "DLSS runs outside its
        // stated range", the odd rect is plainly the lesser problem — so a rect that came from the
        // SDK is passed through untouched and the host's rounding applies only to the fallback.
        bool queryModeRects(uint32_t outW, uint32_t outH,
                            uint32_t* inW, uint32_t* inH, uint32_t count) override
        {
            if (!mParams || outW == 0u || outH == 0u || count < (uint32_t)kUpscaleModeCount)
            {
                return false;
            }
            // Off and DLAA both rasterise at the output rect. DLAA is NOT queried for it: the SDK
            // reports a small band around the output rect and its optimal IS the output rect, so
            // asking would risk taking a rect one pixel off native for a mode whose entire meaning
            // is "native".
            inW[kUpscaleModeOff]  = outW;  inH[kUpscaleModeOff]  = outH;
            inW[kUpscaleModeDLAA] = outW;  inH[kUpscaleModeDLAA] = outH;

            static const struct { UpscaleMode mode; NVSDK_NGX_PerfQuality_Value q; } kMap[] = {
                { kUpscaleModeQuality,     NVSDK_NGX_PerfQuality_Value_MaxQuality       },
                { kUpscaleModeBalanced,    NVSDK_NGX_PerfQuality_Value_Balanced         },
                { kUpscaleModePerformance, NVSDK_NGX_PerfQuality_Value_MaxPerf          },
                { kUpscaleModeUltraPerf,   NVSDK_NGX_PerfQuality_Value_UltraPerformance },
            };
            bool any = false;
            for (const auto& m : kMap)
            {
                unsigned int optW = 0, optH = 0, maxW = 0, maxH = 0, minW = 0, minH = 0;
                float        sharpness = 0.0f;
                const NVSDK_NGX_Result r = NGX_DLSS_GET_OPTIMAL_SETTINGS(
                    mParams, outW, outH, m.q,
                    &optW, &optH, &maxW, &maxH, &minW, &minH, &sharpness);
                if (NVSDK_NGX_FAILED(r) || optW == 0u || optH == 0u)
                {
                    // ⚠ 0x0 MEANS "THIS BACKEND CANNOT SERVE THIS MODE" and the host greys it out.
                    // Echoing the output rect instead would offer the player a mode that silently
                    // rendered at native.
                    inW[m.mode] = 0u; inH[m.mode] = 0u;
                    continue;
                }
                inW[m.mode] = optW; inH[m.mode] = optH;
                any = true;
            }
            if (any && (outW != mModeTableOutW || outH != mModeTableOutH))
            {
                mModeTableOutW = outW;
                mModeTableOutH = outH;
                ngxLog(">> [upscale-ngx] mode rects for out=%ux%u (from the SDK, not from ratios): "
                       "DLAA %ux%u | Quality %ux%u | Balanced %ux%u | Performance %ux%u | "
                       "UltraPerf %ux%u",
                       outW, outH, outW, outH,
                       inW[kUpscaleModeQuality],     inH[kUpscaleModeQuality],
                       inW[kUpscaleModeBalanced],    inH[kUpscaleModeBalanced],
                       inW[kUpscaleModePerformance], inH[kUpscaleModePerformance],
                       inW[kUpscaleModeUltraPerf],   inH[kUpscaleModeUltraPerf]);
            }
            return any;
        }

    private:
        // ⚠ A DEVICE REMOVAL IS DETECTED WHEREVER IT NEXT MATTERS, NOT WHERE IT HAPPENED — the first
        // 4d bring-up run reported it as a failed `Map` inside an unrelated `addBuffer`, three
        // subsystems away from the call that caused it. NGX's two GPU-recording calls are the two
        // places this backend can be the cause, so it asks after each of them by name. Same
        // instrument the host's own `logDeviceRemoved` is, scoped to this file because the backend
        // is where the answer is actionable.
        bool deviceRemovedAt(const char* where)
        {
            if (!mDevice) { return false; }
            const HRESULT reason = mDevice->GetDeviceRemovedReason();
            if (reason == S_OK) { return false; }
            ngxLog("!! [upscale-ngx] DEVICE REMOVED at %s — reason 0x%08lX. Upscaling DISABLED for "
                   "this session.", where, (unsigned long)reason);
            mDisabled = true;
            return true;
        }

        void complainOnce(bool* pFlag, const char* msg)
        {
            if (*pFlag) { return; }
            *pFlag = true;
            ngxLog("%s", msg);
        }

        // ⚠ MVJittered IS NOT OPTIONAL AND upscale.h FORWARD-REFERENCES THIS LINE FOR IT. Both
        // matrices in the reprojection carry their OWN frame's jitter, so the vectors this renderer
        // produces are jittered ("where on the previous frame's screen was this surface", literally).
        // DLSS's DEFAULT is jitter-EXCLUDED. Getting it wrong costs a sub-pixel error that reads as
        // "slightly soft" rather than as a convention mismatch, which is the worst kind of wrong.
        //
        // DepthInverted: pLinearDepth's name lies in our favour — it holds the RAW reverse-Z DEVICE
        // depth (near = 1, far = 0); what the linearize pass does is the MSAA resolve plus SRV
        // exposure.
        //
        // IsHDR: the scene target is scene-referred fp16 and unbounded. Not a display-referred
        // signal, so DLSS must not assume 0..1.
        //
        // NOT AutoExposure: we supply the exposure texture instead. See its allocation.
        int featureFlags(const UpscaleInputs& inputs) const
        {
            int f = NVSDK_NGX_DLSS_Feature_Flags_MVJittered
                  | NVSDK_NGX_DLSS_Feature_Flags_DepthInverted
                  | NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
            // ⚠ AutoExposure IS A **CREATE** FLAG, so flipping the host's `useSuppliedExposure`
            // rebuilds the feature — which is why `featureFlags()` feeds ensureFeature's comparison
            // rather than being recomputed at evaluate. A flag change that did not rebuild would
            // leave DLSS metering one way while we fed it the other.
            if (!inputs.useSuppliedExposure)
            {
                f |= NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
            }
            // ⚠ MVLowRes SAYS "THE MOTION VECTORS ARE AT THE **INPUT** RECT", which ours are:
            // motionVectors is dispatched over the render rect and written at input resolution, like
            // every other screen target here. At DLAA (in == out) the flag cannot matter, which is
            // exactly why it is the first suspect if Quality smears while DLAA is clean — and why
            // the host carries it as a field rather than this file asserting it. The SDK headers
            // never state which sense is the default; `upscaleMvLowRes=0` is the one-token A/B.
            if (inputs.mvAtInputRect)
            {
                f |= NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
            }
            return f;
        }

        // Create the DLSS feature, or re-create it because something it was created AGAINST changed.
        // ⚠ IT RECORDS ONTO THE FRAME'S COMMAND LIST, which is why it cannot live in init(): NGX
        // needs a list to upload its weights on, and the only list this backend ever sees is the one
        // evaluate() is handed.
        //
        // ⚠ THE QUALITY MODE IS DERIVED **HERE**, NOT PASSED IN, and that is not a tidy-up: picking
        // it means five `NGX_DLSS_GET_OPTIMAL_SETTINGS` calls (see pickQuality), and the mode is a
        // pure function of the rects — so if the rects are unchanged the mode cannot have changed
        // either, and the query belongs on the rebuild path rather than on every frame.
        bool ensureFeature(Cmd* pCmd, const UpscaleInputs& inputs, int flags)
        {
            if (mFeature && !mFeatureStale && mFeatInW == inputs.inW && mFeatInH == inputs.inH
                && mFeatOutW == inputs.outW && mFeatOutH == inputs.outH && mFeatFlags == flags
                && mFeatPreset == inputs.preset)
            {
                return true;
            }
            if (mFeature)
            {
                // ⚠ THE OLD FEATURE'S GPU WORK MAY STILL BE IN FLIGHT. ReleaseFeature is documented
                // as invalidating the handle immediately, so this is only safe because the host
                // render is lockstep: the frame that last evaluated has completed before this frame
                // records. If the host ever overlaps frames, this needs the same deferred-destroy
                // treatment the texture slots have ([[project_forge_host_overlap_hazards]]).
                NVSDK_NGX_D3D12_ReleaseFeature(mFeature);
                mFeature = nullptr;
            }

            const NVSDK_NGX_PerfQuality_Value quality = pickQuality(inputs);

            // ─── THE MODEL HINT, SET BEFORE CreateFeature READS THE PARAMETER BLOCK ──────────────
            // ⚠ ALL SIX MODES ARE SET, not just the one being created, and that is deliberate: the
            // hints live on the parameter block rather than on the feature, and the block outlives
            // every feature built through it. Setting only the current mode's would leave the other
            // five carrying whatever a previous build put there, so a mode change would silently
            // pick up a stale model — a look change with no visible cause, which is the worst kind.
            {
                const NVSDK_NGX_DLSS_Hint_Render_Preset hint = ngxPreset(inputs.preset);
                NVSDK_NGX_Parameter_SetUI(mParams,
                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, (unsigned int)hint);
                NVSDK_NGX_Parameter_SetUI(mParams,
                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, (unsigned int)hint);
                NVSDK_NGX_Parameter_SetUI(mParams,
                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, (unsigned int)hint);
                NVSDK_NGX_Parameter_SetUI(mParams,
                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, (unsigned int)hint);
                NVSDK_NGX_Parameter_SetUI(mParams,
                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, (unsigned int)hint);
                NVSDK_NGX_Parameter_SetUI(mParams,
                    NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality, (unsigned int)hint);
            }

            NVSDK_NGX_DLSS_Create_Params cp = {};
            cp.Feature.InWidth            = inputs.inW;
            cp.Feature.InHeight           = inputs.inH;
            cp.Feature.InTargetWidth      = inputs.outW;
            cp.Feature.InTargetHeight     = inputs.outH;
            cp.Feature.InPerfQualityValue = quality;
            cp.InFeatureCreateFlags       = flags;
            // Our output subrect base is (0,0), so DLSS never needs to write anywhere but the
            // origin of an alloc-sized target. Left false rather than set defensively: enabling it
            // asks DLSS for a capability we do not use.
            cp.InEnableOutputSubrects     = false;

            const NVSDK_NGX_Result r = NGX_D3D12_CREATE_DLSS_EXT(pCmd->mDx.pCmdList, 1, 1, &mFeature,
                                                                 mParams, &cp);
            // ⚠ CREATE_DLSS RECORDS AND BINDS TOO. Same restore as after an evaluate.
            restoreForgeCmdState(pCmd);
            if (NVSDK_NGX_FAILED(r) || !mFeature)
            {
                ngxLog("!! [upscale-ngx] CreateFeature FAILED: 0x%08X (%s) for in=%ux%u "
                             "out=%ux%u mode=%s flags=0x%X — upscaling DISABLED for this session.",
                             (unsigned)r, ngxResultStr(r), inputs.inW, inputs.inH,
                             inputs.outW, inputs.outH, qualityName(quality), (unsigned)flags);
                mFeature = nullptr;
                return false;
            }

            mFeatInW = inputs.inW;   mFeatInH = inputs.inH;
            mFeatOutW = inputs.outW; mFeatOutH = inputs.outH;
            mFeatQuality = quality;  mFeatFlags = flags;
            mFeatPreset = inputs.preset;
            mFeatureStale = false;
            if (deviceRemovedAt("NGX_D3D12_CREATE_DLSS_EXT")) { return false; }
            unsigned long long vram = 0ull;
            unsigned int optLevel = 0u;
            NGX_DLSS_GET_STATS_1(mParams, &vram, &optLevel);
            // ⚠ THE FLAGS ARE DECODED, NOT JUST PRINTED AS A HEX WORD. Every one of them is a
            // CONVENTION this integration had to get right, and three of the four are invisible in
            // the picture when wrong — MVJittered wrong reads as "slightly soft", DepthInverted
            // wrong as "ghosting", AutoExposure as a subtly different response. A hex value nobody
            // decodes is not an instrument.
            ngxLog(">> [upscale-ngx] feature CREATED: in=%ux%u out=%ux%u mode=%s flags=0x%X "
                         "preset=%s [MVJittered%s DepthInverted IsHDR, exposure %s] vram=%.1f MB",
                         inputs.inW, inputs.inH, inputs.outW, inputs.outH, qualityName(quality),
                         (unsigned)flags,
                         kUpscalePresetNames[inputs.preset < (uint32_t)kUpscalePresetCount
                                             ? inputs.preset : 0u],
                         (flags & NVSDK_NGX_DLSS_Feature_Flags_MVLowRes) ? " MVLowRes" : "",
                         (flags & NVSDK_NGX_DLSS_Feature_Flags_AutoExposure)
                             ? "AUTO (DLSS meters it; the supplied-texture path HANGS the GPU — "
                               "see tasks/forge-upscale.md 4d)"
                             : "SUPPLIED (1x1 R32F)",
                         (double)vram / (1024.0 * 1024.0));
            return true;
        }

        // ─── PICK THE MODE BY ASKING, NOT BY GUESSING ────────────────────────────────────────────
        // Walk the ladder highest-quality-first and take the first mode whose reported [min,max]
        // CONTAINS the rect the host actually rendered. See the note beside kQualityLadder for the
        // three in-game failures that retired the ratio table this replaces.
        //
        // ⚠ THE WHOLE TABLE IS LOGGED, not just the winner. A mode choice that looks wrong in the
        // picture is otherwise a number with no derivation behind it — and since these ranges are
        // driver- and version-dependent, the row that will matter on a future driver is the one
        // nobody thought to print. It is five lines once per rect change, not per frame.
        NVSDK_NGX_PerfQuality_Value pickQuality(const UpscaleInputs& inputs)
        {
            QualityPick pick;
            // The fallback if NOTHING contains the rect: the mode whose OPTIMAL is nearest in area,
            // so a rect in one of DLSS's dead zones still gets the closest-tuned network rather than
            // an arbitrary one.
            double bestErr = 1.0e300;
            const double ourArea = (double)inputs.inW * (double)inputs.inH;

            for (NVSDK_NGX_PerfQuality_Value q : kQualityLadder)
            {
                unsigned int optW = 0, optH = 0, maxW = 0, maxH = 0, minW = 0, minH = 0;
                float        sharpness = 0.0f;
                const NVSDK_NGX_Result r = NGX_DLSS_GET_OPTIMAL_SETTINGS(
                    mParams, inputs.outW, inputs.outH, q,
                    &optW, &optH, &maxW, &maxH, &minW, &minH, &sharpness);
                if (NVSDK_NGX_FAILED(r))
                {
                    ngxLog("!! [upscale-ngx]   %-16s GET_OPTIMAL_SETTINGS FAILED: 0x%08X (%s)",
                           qualityName(q), (unsigned)r, ngxResultStr(r));
                    continue;
                }
                // ⚠ min/max CAN BE EQUAL — UltraPerformance reports min == max == optimal, i.e. a
                // FIXED rect with no dynamic range, which is one of the three things the ratio table
                // could not express. The containment test handles that case for free; a ratio
                // threshold never could.
                const bool contains = (minW != 0u && maxW != 0u)
                                   && inputs.inW >= minW && inputs.inW <= maxW
                                   && inputs.inH >= minH && inputs.inH <= maxH;
                if (contains && !pick.contained)
                {
                    pick.mode = q;
                    pick.contained = true;
                }
                if (!pick.contained)
                {
                    const double err = (double)optW * (double)optH - ourArea;
                    const double abserr = (err < 0.0) ? -err : err;
                    if (abserr < bestErr) { bestErr = abserr; pick.mode = q; }
                }
                ngxLog("%s [upscale-ngx]   %-16s optimal=%ux%u range=[%ux%u .. %ux%u]%s",
                       contains ? ">>" : "  ", qualityName(q), optW, optH, minW, minH, maxW, maxH,
                       contains ? "  <- accepts our rect" : "");
            }

            if (!pick.contained)
            {
                // ⚠ IT STILL RUNS — DLSS did not fault at any of the eleven out-of-range rects the
                // first in-game session produced, so refusing here would trade a working (if
                // off-nominal) frame for none. But it is a REAL gap and it is named rather than
                // smoothed over: the host's input-scale slider is FREE and floors at 0.33, while
                // DLSS's usable rects are a set of bands with holes between them — most sharply
                // below Performance's 0.5 floor, where the only thing left is UltraPerformance's
                // single fixed rect. Presenting the modes as discrete choices instead of a free
                // slider is what closes it, and that is 4d-3.
                ngxLog("!! [upscale-ngx] NO DLSS mode accepts in=%ux%u for out=%ux%u — the host's "
                       "input scale has landed in a gap between this driver's mode ranges. Falling "
                       "back to '%s' (nearest optimal). The frame renders and DLSS does not fault, "
                       "but it is running outside its stated dynamic range: distrust the picture "
                       "before distrusting anything else.",
                       inputs.inW, inputs.inH, inputs.outW, inputs.outH, qualityName(pick.mode));
            }
            return pick.mode;
        }

        // One float into a 1x1 R32_FLOAT, through the upload row. Cheap enough not to be worth
        // conditionalising on the value having changed — the servo moves E every frame it steps.
        void uploadExposure(Cmd* pCmd, float e)
        {
            if (!mExposureTex || !mExposureUpload || !mExposureUpload->pCpuMappedAddress)
            {
                return;
            }
            const float v = (e > 0.0f) ? e : 1.0f;
            // ⚠ ONLY WHEN IT MOVED. The servo changes E on most frames but not all — it HOLDS on a
            // frame that produced no measurement (a paused host, a menu, the scene probe, which pins
            // E to exactly 1.0) — and a copy plus two transitions on a 1x1 texture is pure cost on
            // those. It also makes the steady-state frame recording one item shorter, which is the
            // shape every other per-frame update in this renderer already has.
            if (mExposureValid && v == mLastExposure)
            {
                return;
            }
            mLastExposure = v;
            mExposureValid = true;
            std::memcpy(mExposureUpload->pCpuMappedAddress, &v, sizeof(float));

            TextureBarrier tb = {};
            tb.pTexture      = mExposureTex;
            tb.mCurrentState = RESOURCE_STATE_SHADER_RESOURCE;
            tb.mNewState     = RESOURCE_STATE_COPY_DEST;
            cmdResourceBarrier(pCmd, 0, nullptr, 1, &tb, 0, nullptr);

            D3D12_TEXTURE_COPY_LOCATION dst = {};
            dst.pResource        = mExposureTex->mDx.pResource;
            dst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.SubresourceIndex = 0;
            D3D12_TEXTURE_COPY_LOCATION src = {};
            src.pResource = mExposureUpload->mDx.pResource;
            src.Type      = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint.Offset             = 0;
            src.PlacedFootprint.Footprint.Format   = DXGI_FORMAT_R32_FLOAT;
            src.PlacedFootprint.Footprint.Width    = 1;
            src.PlacedFootprint.Footprint.Height   = 1;
            src.PlacedFootprint.Footprint.Depth    = 1;
            // D3D12_TEXTURE_DATA_PITCH_ALIGNMENT. Four bytes of payload, but the row pitch is a
            // hardware constraint and D3D12 rejects anything smaller.
            src.PlacedFootprint.Footprint.RowPitch = 256;
            pCmd->mDx.pCmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

            tb.mCurrentState = RESOURCE_STATE_COPY_DEST;
            tb.mNewState     = RESOURCE_STATE_SHADER_RESOURCE;
            cmdResourceBarrier(pCmd, 0, nullptr, 1, &tb, 0, nullptr);
        }

        // ─── THE FORGE STATE RESTORE ─────────────────────────────────────────────────────────────
        // See the file header for why this has to exist at all. `DescriptorHeap` is private to
        // Direct3D12.c and its first member is `ID3D12DescriptorHeap* pHeap` (Direct3D12.c:341) —
        // which is an assumption, so it is CHECKED rather than trusted: Forge cached each heap's GPU
        // start handle in `mBoundHeapStartHandles` during beginCmd, from exactly these heaps, so
        // asking the API for the start handle again has to reproduce that number. A mismatch means
        // the layout moved under us, and the honest answer is then to refuse to run rather than to
        // scribble on the rest of the frame.
        bool probeForgeCmdState(Cmd* pCmd)
        {
            if (mProbeDone)
            {
                return mProbeOk;
            }
            mProbeDone = true;
            mProbeOk = false;
            if (!pCmd->mDx.pBoundHeaps[0] || !pCmd->mDx.pBoundHeaps[1] || !pCmd->pRenderer)
            {
                ngxLog("!! [upscale-ngx] the command list has no bound descriptor heaps — "
                             "NGX's clobber of them could not be undone. Backend DISABLED.");
                return false;
            }
            for (int i = 0; i < 2; ++i)
            {
                ID3D12DescriptorHeap* heap = heapOf(pCmd, i);
                if (!heap)
                {
                    ngxLog("!! [upscale-ngx] bound heap %d resolved to null — backend "
                                 "DISABLED", i);
                    return false;
                }
                D3D12_GPU_DESCRIPTOR_HANDLE h = heap->GetGPUDescriptorHandleForHeapStart();
                if (h.ptr != pCmd->mDx.mBoundHeapStartHandles[i].ptr)
                {
                    ngxLog("!! [upscale-ngx] descriptor-heap layout probe FAILED on heap %d "
                                 "(got 0x%llX, The Forge cached 0x%llX). DescriptorHeap::pHeap is "
                                 "no longer the struct's first member, so NGX's clobber of the "
                                 "command list's heaps cannot be undone. Backend DISABLED — the "
                                 "frame renders exactly as it does without DLSS.",
                                 i, (unsigned long long)h.ptr,
                                 (unsigned long long)pCmd->mDx.mBoundHeapStartHandles[i].ptr);
                    return false;
                }
            }
            if (!pCmd->pRenderer->mDx.pGraphicsRootSignature
                || !pCmd->pRenderer->mDx.pComputeRootSignature)
            {
                ngxLog("!! [upscale-ngx] the renderer has no root signatures to restore — "
                             "backend DISABLED");
                return false;
            }
            mProbeOk = true;
            ngxLog(">> [upscale-ngx] Forge command-list state probe OK — NGX's descriptor-heap "
                         "and root-signature clobber will be undone after every evaluate.");
            return true;
        }

        static ID3D12DescriptorHeap* heapOf(Cmd* pCmd, int i)
        {
            // The verified reinterpretation. Kept in one function so there is exactly one place the
            // assumption lives, and probeForgeCmdState() is the thing that licenses it.
            return *(ID3D12DescriptorHeap* const*)pCmd->mDx.pBoundHeaps[i];
        }

        // Replicates the three statements beginCmd makes (Direct3D12.c:5273-5292) and nothing else.
        void restoreForgeCmdState(Cmd* pCmd)
        {
            ID3D12DescriptorHeap* heaps[2] = { heapOf(pCmd, 0), heapOf(pCmd, 1) };
            pCmd->mDx.pCmdList->SetDescriptorHeaps(2, heaps);
            if (pCmd->mDx.mType == QUEUE_TYPE_GRAPHICS)
            {
                pCmd->mDx.pCmdList->SetGraphicsRootSignature(
                    pCmd->pRenderer->mDx.pGraphicsRootSignature);
            }
            pCmd->mDx.pCmdList->SetComputeRootSignature(pCmd->pRenderer->mDx.pComputeRootSignature);
            // ⚠ AND THE CACHE, WHICH IS THE HALF THAT IS EASY TO MISS. cmdBindDescriptorSet SKIPS a
            // SetRootDescriptorTable whose handle matches this cache (Direct3D12.c:4686). Setting a
            // root signature resets the root arguments, so every cached handle is now a lie — leave
            // it standing and the next pass to re-bind a set it "already had" binds nothing at all.
            std::memset(pCmd->mDx.mBoundDescriptorSets, 0, sizeof(pCmd->mDx.mBoundDescriptorSets));
        }

        Texture*        mOutput = nullptr;
        // The typed D3D12 resource behind mOutput. Owned by THIS class, not by The Forge — see the
        // note at its creation for why it is not a plain Forge texture.
        ID3D12Resource* mOutputRes = nullptr;
        Texture*      mExposureTex = nullptr;
        Buffer*       mExposureUpload = nullptr;
        ID3D12Device* mDevice = nullptr;

        // ⚠ OUTLIVES init() ON PURPOSE — NGX keeps the pointer. See its fill site.
        NVSDK_NGX_FeatureCommonInfo mCommonInfo = {};
        NVSDK_NGX_Parameter* mParams = nullptr;
        NVSDK_NGX_Handle*    mFeature = nullptr;
        bool                 mNgxInited = false;

        // What the live feature was CREATED against. evaluate() compares the rects it is handed
        // against these every frame — the one authority for "does the feature still match".
        uint32_t mFeatInW = 0, mFeatInH = 0, mFeatOutW = 0, mFeatOutH = 0;
        NVSDK_NGX_PerfQuality_Value mFeatQuality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
        int      mFeatFlags = 0;
        uint32_t mFeatPreset = kUpscalePresetDefault;
        bool     mFeatureStale = false;

        // Recorded by bindInputs for the arm log; see the note there on why this backend does not
        // refuse a mismatch the way the passthrough does.
        Texture* mBoundColor = nullptr;
        Texture* mBoundDepth = nullptr;
        Texture* mBoundMv    = nullptr;
        Texture* mBoundMask  = nullptr;

        uint32_t mAllocW = 0, mAllocH = 0;
        // What output rect the published mode table was last LOGGED for — so a per-frame republish
        // costs no log line, while a resolution change still says what it changed to.
        uint32_t mModeTableOutW = 0, mModeTableOutH = 0;
        bool     mDisabled = false;          // a hard NGX failure: permanent for the session
        bool     mProbeDone = false;
        bool     mProbeOk = false;
        // What the exposure texture currently HOLDS, so a frame whose E did not move records no
        // copy at all. See uploadExposure.
        float    mLastExposure = 0.0f;
        bool     mExposureValid = false;
        bool     mEvalAnnounced = false;
        bool     mRectComplained = false;
        bool     mMissingComplained = false;
    };
}

IUpscaler* createNgxUpscaler()
{
    // tf_new, not new, for the same reason the passthrough uses it: IMemory.h poisons `new` in this
    // TU and the object has to come off the allocator destroyUpscaler() gives it back to.
    return tf_new(NgxUpscaler);
}
