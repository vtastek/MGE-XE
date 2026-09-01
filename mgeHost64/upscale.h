// mgeHost64 — THE UPSCALER SEAM.  tasks/forge-upscale.md M1 step 4b.
//
// ─── WHAT THIS FILE IS FOR, AND WHY IT LANDS BEFORE NGX ──────────────────────────────────────────
//
// The point of 4b is NOT the picture a spatial upscaler makes. It is that a real upscaler is a pass
// that reads an INPUT-res image and writes a SEPARATE OUTPUT-res resource, and every part of that —
// the target's lifetime, who owns it, which descriptor the resolve reads, where the pass sits in the
// frame, how the input rect is driven — has to exist before a single NVIDIA symbol is linked. Build
// it with DLSS present and every mistake in the seam looks like a DLSS artefact.
//
// So the seam lands first with a backend that has no NVIDIA dependency, and that backend is **not
// throwaway scaffolding**: `createPassthroughUpscaler()` is the PERMANENT fallback for the
// AMD / Intel / GTX-1080 tiers this project promises to keep running. A bilinear blit would be
// permanent softness for those users, which is why the passthrough is a Catmull-Rom with an
// anti-ringing clamp rather than a copy.
//
// ─── THE CONTRACT ────────────────────────────────────────────────────────────────────────────────
//
// NO ACCESS TO `g_live`. Everything the backend needs arrives as parameters, and that is the single
// property that makes the backend swappable: 4d's NGX backend is a different implementation of the
// same six calls, not a second set of hooks into the renderer. It is also what lets the passthrough
// be TESTED — a backend that reads renderer globals can only be run inside a frame.
//
// ⚠ `evaluate()` RETURNS `inputs.pColor` UNCHANGED WHEN `in == out`, HAVING RECORDED NOTHING.
// Identity is then bit-identical and free BY CONSTRUCTION rather than by measurement, which is the
// only kind of identity worth having here. The consequence has to be said out loud because it is the
// thing that makes a 4b test meaningless if it is forgotten: **scale 1.0 proves nothing about the
// seam.** The real test is run at 0.5.
//
// ⚠ THE HOST DECIDES WHETHER THE FRAME WAS UPSCALED BY COMPARING POINTERS, not by re-deriving the
// rects. `evaluate()` returns "what the frame should READ", so `returned != inputs.pColor` is the
// one true answer to "did a pass run", and every downstream decision (which descriptor set instance
// the resolve binds, what its source clamp is, what the bloom tap is scaled by) hangs off that one
// comparison. A second copy of the in==out test somewhere else is a second place to get it wrong.
//
// ⚠ NO MID-FRAME `updateDescriptorSet`. `bindInputs()` exists precisely so the backend's descriptor
// writes happen ONCE, at path-build time, on the host thread, outside command recording — which is
// where every other bind in this renderer happens. `evaluate()` then REFUSES (returns pColor) if it
// is ever handed a colour texture other than the one it was bound to, rather than silently sampling
// a stale descriptor. A backend with no descriptor set of its own (NGX) returns true and does
// nothing.
//
// ─── WHAT IS DELIBERATELY DECLARED BUT UNREAD IN 4b ──────────────────────────────────────────────
//
// `pDepth`, `pMotionVectors`, `pReactive`, `jitter*`, `exposure` and `reset` are all on the input
// struct and none of them is read by the passthrough. That is on purpose: 4d needs the SHAPE to
// already be right, and a field added later is a field whose plumbing has never been exercised. The
// host fills every one of them today, so the day the NGX backend is dropped in behind this interface
// there is no new wiring to get wrong — only the backend.
//
// ⚠⚠ ONE NOTE FOR 4d THAT BELONGS HERE RATHER THAN IN THE BACKEND, because motionvectors.srt.h
// already forward-references this file for it: the motion vectors this renderer produces are
// **JITTERED** ("where on the previous frame's screen was this surface", literally, both matrices
// carrying their own frame's jitter). DLSS's DEFAULT is jitter-EXCLUDED vectors, so the NGX backend
// MUST set `NVSDK_NGX_DLSS_Feature_Flags_MVJittered`. Getting that wrong costs a sub-pixel error
// that reads as "slightly soft" rather than as a convention mismatch.
#pragma once

#include <cstdint>
// TinyImageFormat, for `init`'s scene-colour format argument. Pulled in directly rather than relying
// on the includer having reached IGraphics.h first — a header that only compiles in one include
// order is a trap for the next TU that wants it.
#include "Resources/ResourceLoader/ThirdParty/OpenSource/tinyimageformat/tinyimageformat_base.h"

// The Forge's own C-style types. Forward-declared rather than included: this header is pulled into
// forgerender.cpp AFTER IMemory.h has poisoned `new`/`delete`, and the fewer Forge headers it drags
// through that door the better.
struct Renderer;
struct Cmd;
struct Texture;

// Everything a temporal upscaler can want, filled by the host every frame whether the current
// backend reads it or not (see the note above on why the unread fields are here).
struct UpscaleInputs
{
    // The scene colour. ⚠ It is an ALLOC-sized surface with the INPUT rect written inside it — the
    // rest holds LAST FRAME'S texels ([[project_forge_alloc_vs_render_uv]]). Every tap a backend
    // takes must clamp to `inW x inH`, never to the surface extent.
    //
    // ⚠⚠ AND ITS rgb IS PREMULTIPLIED BY ITS ALPHA. A resample is a FILTER, so every tap weight has
    // to hit all four channels identically; split them and the witness is a darker seam along the
    // horizon fog band, the only place in the frame where coverage is neither 0 nor 1. That rule has
    // now been stated in resolve.srt.h, bloom.srt.h, reflectmip.srt.h and upscale.comp.fsl, and it
    // has been got wrong at least twice.
    Texture* pColor = nullptr;
    // Declared now, unread by the passthrough — 4d needs the shape to already be right.
    // ⚠ pDepth is the RAW reverse-Z DEVICE depth (pLinearDepth's name lies in our favour: what the
    // linearize pass actually does is the MSAA resolve + SRV exposure).
    Texture* pDepth = nullptr;
    Texture* pMotionVectors = nullptr;   // R16G16_SFLOAT, screen-space pixels, previous-minus-current
    Texture* pReactive = nullptr;        // R8, the reprojection-consistency mask (M1 step 3)

    // alloc >= out >= in, the three rects 4a split apart. `in` is where the scene RASTERISED;
    // `out` is what the client COMPOSITES; `alloc` is what every screen target was allocated at.
    uint32_t inW = 0, inH = 0;
    uint32_t outW = 0, outH = 0;
    uint32_t allocW = 0, allocH = 0;

    // This frame's sub-pixel jitter in pixels, +x right / +y down — the same sense the half-pixel
    // term uses, and the same value the matrices were actually rendered with.
    float jitterX = 0.0f, jitterY = 0.0f;
    // The exposure servo's current E. ⚠ 4d must feed the PREVIOUS frame's value: the servo meters
    // the DELIVERED image, so handing DLSS an exposure derived from DLSS output closes a loop around
    // the upscaler ([[project_forge_exposure_rail_instrument]]).
    float exposure = 1.0f;
    // History discontinuity — a camera cut, a cell change, a device reset. Declared, unread in 4b
    // (the passthrough has no history to reset).
    bool reset = false;
    // --- SPATIAL FILTER PARAMETERS (M1 4c follow-up) ---------------------------------------------
    // They arrive HERE, as parameters, rather than as globals in the backend, because that is this
    // header's whole contract: "it owns nothing the renderer owns — every input arrives as a
    // parameter", which is the property that makes 4d a drop-in. They were `constexpr` in
    // upscale.cpp under a note saying a filter parameter "is a thing to walk once the seam is
    // known-correct". 4b closed and 4c verified, so it is now that time — and the thing forcing it
    // is a real question they are the instrument for: **bloom loses impact at scale < 1 on a
    // candle-lit interior**, and bloom's source is now this pass's output.
    //
    // A backend that has no spatial filter (NGX, 4d) simply ignores both, the way the passthrough
    // ignores `reset`.
    //
    // Mitchell C at B = 0. 0.5 = Catmull-Rom.
    float sharpness = 0.5f;
    // ⚠ THE ANTI-RINGING CLAMP, AND IT IS THE ONE SUSPECT WORTH A LIVE KNOB. It is REQUIRED for
    // correctness — Catmull-Rom's outer lobe is negative and this source is scene-referred and
    // unbounded — but it is also a NON-LINEAR, NON-ENERGY-CONSERVING operator, and since 4c it sits
    // directly upstream of an ENERGY operation (the bloom pyramid). This tree has already made that
    // exact distinction twice, in the same direction: the HDR EXR dump takes a hardware BOX resolve
    // because the shader resolve "is a display reconstruction and is deliberately not
    // energy-conserving", and `g_bloomInvLuma` ships OFF for the same reason. 0 is the diagnostic
    // arm — expect visible ringing around bright edges while it is off; it is a measurement, not a
    // setting.
    float antiRing = 1.0f;
};

// The seam. Six calls, and a backend is nothing but an implementation of them.
class IUpscaler
{
public:
    virtual ~IUpscaler() {}

    // Create everything that is sized off the ALLOCATION rect — for the passthrough that is one
    // fp16 target, one cbuffer, one shader, one pipeline and one descriptor set. Alloc-sized for the
    // same reason every other screen target in this renderer is: a live render-scale change must
    // need no reallocation. Returns false on ANY failure, having cleaned up after itself; the host
    // then forces the feature off and renders exactly as it did before this file existed.
    virtual bool init(Renderer* pRenderer, uint32_t allocW, uint32_t allocH,
                      TinyImageFormat colorFmt) = 0;

    // Bind the INPUT resources into the backend's own descriptor set. Called ONCE, at path-build
    // time, outside command recording. See the header note on why this is not folded into evaluate.
    virtual bool bindInputs(Renderer* pRenderer, const UpscaleInputs& inputs) = 0;

    // The OUTPUT rect changed. 4b: a no-op — the target is alloc-sized and the rect rides the
    // cbuffer. 4d: this is where the NGX feature gets torn down and rebuilt, which is why it is a
    // call and not a field.
    virtual void resize(uint32_t outW, uint32_t outH) = 0;

    // Record the pass. Returns WHAT THE FRAME SHOULD READ — its own output when it did something,
    // and `inputs.pColor` unchanged when it did not (in == out, or any internal refusal). The
    // returned texture is left in RESOURCE_STATE_SHADER_RESOURCE.
    virtual Texture* evaluate(Cmd* pCmd, const UpscaleInputs& inputs) = 0;

    // The resource the host binds into the resolve's second descriptor-set instance, once, at
    // path-build time. Never null after a successful init.
    virtual Texture* outputTexture() const = 0;

    // Rebuild the shader + pipeline from disk (F8). The TARGET and the descriptor set survive, so a
    // failed reload leaves the pass inert rather than dangling — same shape as bloom / reflectmip /
    // motionvectors. Returns false if the backend is no longer able to run.
    virtual bool reload(Renderer* pRenderer) = 0;

    // Release GPU resources. Does NOT free the object — see destroyUpscaler.
    virtual void shutdown(Renderer* pRenderer) = 0;

    // For the log lines and the `[rect]` heartbeat. A silently-absent upscaler is indistinguishable
    // from a working one at scale 1.0, which is exactly why the name is on the wire.
    virtual const char* name() const = 0;
};

// The 4b backend: a Catmull-Rom resample with an anti-ringing clamp, and the permanent fallback for
// every GPU that will never run NGX.
IUpscaler* createPassthroughUpscaler();

// Free the object itself. Separate from shutdown() because IMemory.h poisons `delete` in every TU
// that includes it — the allocation and the free both have to live on the tf_ allocator, and keeping
// them in one file is how that stays true.
void destroyUpscaler(IUpscaler* pUpscaler);
