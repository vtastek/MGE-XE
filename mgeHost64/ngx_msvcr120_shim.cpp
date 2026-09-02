// mgeHost64 — THE THREE CRT SYMBOLS NVIDIA'S vs2013 NGX LIBRARY WANTS AND THE UCRT NO LONGER HAS.
// tasks/forge-upscale.md M1 step 4d.
//
// ─── WHY THIS FILE EXISTS, AND WHY THE 4d PLAN SAID IT WOULD NOT NEED TO ─────────────────────────
//
// The plan opened with "the static lib risk is dead — checked, not assumed": a probe TU including
// `d3d12.h` + `nvsdk_ngx.h` + `nvsdk_ngx_helpers.h` compiled, linked and RAN against
// `nvsdk_ngx_s.lib` under MSVC 14.44 ([[project_dlss_sdk_vendored]]). That was a true measurement of
// the wrong configuration, and it is worth naming the gap rather than quietly patching over it,
// because the shape recurs: **the probe was built `cl /EHsc`, whose default CRT is `/MT`; this host
// builds `/MD`.** The probe therefore linked the static-CRT objects of a static-CRT library and
// nothing was mixed. The host mixes.
//
// What the real link says:
//
//   nvsdk_ngx_s.lib  — `/MT`. Against this host: RuntimeLibrary MT_StaticRelease vs MD_DynamicRelease
//                      (two CRTs, two heaps in one process — not a warning to defeat, a defect),
//                      plus the three symbols below.
//   nvsdk_ngx_d.lib  — the SAME objects built `/MD`. ⚠ The `_d` is the CRT LINKAGE, not "dynamic
//                      import library", which is what VENDORED.md's escalation list assumed it was
//                      (`dumpbin /ARCHIVEMEMBERS` says `.../release_dynamic_release_vs2013/...`).
//                      This is the right library and it is what the project links. The RuntimeLibrary
//                      and _ITERATOR_DEBUG_LEVEL directives now MATCH; only `_MSC_VER` and the three
//                      symbols remain.
//
// `_MSC_VER` is handled where it has to be — `_ALLOW_MSC_VER_MISMATCH` on the x64 configurations,
// because `#pragma detect_mismatch` is collected across the WHOLE image and every one of our objects
// emits 1900 (`yvals.h:140`). It cannot be confined to one TU. The comment in the .vcxproj says the
// same thing from the build's side.
//
// ⚠ AND THE MIX IS NOW A NARROW, STATED ONE RATHER THAN A BLIND ONE. There is ONE CRT in the process
// (the UCRT, via /MD). What the vs2013 objects want is not a second CRT, it is three symbols that
// VS2015's UCRT rewrite removed or made inline. All three are used ONLY by NGX's own logging path
// (`util-log.obj`), which is why getting one subtly wrong degrades a log line rather than a frame.
//
// ─── WHY NOT THE OTHER TWO ROUTES ────────────────────────────────────────────────────────────────
//
// VENDORED.md listed two escalations. Both were read before this file was written:
//
//  1. "the dynamic import lib" — does not exist. See above: `_d` is the CRT flavour. Taken anyway,
//     because it removes the RuntimeLibrary conflict, which is the half of this that WOULD have been
//     a real two-heap defect.
//  2. GetProcAddress against the driver's NGX core. Checked: `_nvngx.dll` (in the driver store, not
//     System32) does export `NVSDK_NGX_D3D12_Init_ProjectID`, `GetCapabilityParameters`,
//     `CreateFeature`, `EvaluateFeature`, `ReleaseFeature`, `DestroyParameters` and `Shutdown1`, so
//     it is genuinely possible. It is NOT taken, because it means re-implementing the part of the
//     SDK that does version negotiation and snippet discovery against a driver export surface NVIDIA
//     documents nowhere — trading three symbols in a logging path for the whole loader. The static
//     lib IS the supported route; this file is the price of taking it.
//
// ⚠ IT DOES NOT COST THE NON-NVIDIA FALLBACK, which was the one thing worth checking before
// committing to a static link. The SDK `LoadLibrary`s `_nvngx.dll` inside `NVSDK_NGX_D3D12_Init*` —
// there is no import-table entry for it (`dumpbin /DIRECTIVES` on the lib names only MSVCRT,
// OLDNAMES, msvcprt and uuid) — so mgeHost64.exe still LOADS on an AMD or Intel machine and the
// backend declines at init with a named reason, which is exactly the contract upscale.h promises.
//
// ─── THE MECHANISM ───────────────────────────────────────────────────────────────────────────────
//
// The references are `__imp_`-prefixed because the objects were built `/MD` against msvcr120.dll,
// so each one is a POINTER VARIABLE the call sites indirect through, not a function. So this file
// defines the pointers, and `/alternatename` gives them the decorated names the linker is looking
// for — used for all three rather than only for the C++-mangled one, so there is one mechanism here
// instead of two.

#include <cstdio>
#include <cstdarg>
#include <cstring>

// The three names the vs2013 objects import. `/alternatename:undefined=defined` is resolved only for
// symbols nothing else provides, which is precisely the case for all three.
#pragma comment(linker, "/alternatename:__imp___iob_func=mgeNgxImp_iob_func")
#pragma comment(linker, "/alternatename:__imp_vfprintf_s=mgeNgxImp_vfprintf_s")
#pragma comment(linker, "/alternatename:__imp_?_Winerror_map@std@@YAPEBDH@Z=mgeNgxImp_Winerror_map")

namespace
{
    // ─── THE FAKE `_iobuf` TABLE ─────────────────────────────────────────────────────────────────
    // In msvcr120, `__iob_func()` returned the base of a THREE-ELEMENT ARRAY of `struct _iobuf` and
    // the macros were `stdin = &__iob_func()[0]`, `stdout = [1]`, `stderr = [2]`. The UCRT replaced
    // that with `__acrt_iob_func(i)` and made `FILE` opaque, so there is no array to hand back and
    // no way to make one that the UCRT would accept.
    //
    // What NGX actually does with the result is index it and pass the element to `vfprintf_s` —
    // which is also ours. So the array does not have to BE anything: it only has to be an address
    // space we can recognise on the way back in. Three inert slots, and the shim below maps an
    // incoming pointer to the real UCRT stream by its index.
    //
    // ⚠ 48 BYTES IS msvcr120's x64 `sizeof(FILE)` (_ptr, _cnt+pad, _base, _flag, _file, _charbuf,
    // _bufsiz, _tmpfname), and it is what makes `&base[2]` land on slot 2 rather than in the middle
    // of slot 1. It is DERIVED rather than trusted: if it were wrong the incoming pointer would not
    // be a whole number of slots from the base, and `streamFor()` falls back to stderr instead of
    // computing a nonsense index. The cost of being wrong is therefore a log line going to stderr
    // instead of stdout — which is why this is a stated assumption with a guard and not a risk.
    constexpr size_t kMsvcr120FileSize = 48;
    alignas(16) unsigned char gFakeIob[3 * kMsvcr120FileSize] = {};

    // Map one of the fake slots back onto a real UCRT stream. Anything unrecognised goes to stderr:
    // this is a diagnostic path, and a log line on the wrong stream is strictly better than a write
    // through a pointer the UCRT never handed out.
    std::FILE* streamFor(void* p)
    {
        unsigned char* q = (unsigned char*)p;
        if (q < gFakeIob || q >= gFakeIob + sizeof(gFakeIob))
        {
            return stderr;
        }
        const size_t off = (size_t)(q - gFakeIob);
        if ((off % kMsvcr120FileSize) != 0)
        {
            return stderr;   // the size assumption above is wrong; say so by behaving safely
        }
        switch (off / kMsvcr120FileSize)
        {
        case 0:  return stdin;
        case 1:  return stdout;
        default: return stderr;
        }
    }

    void* NgxIobFunc()
    {
        return gFakeIob;
    }

    int NgxVfprintfS(void* stream, const char* format, va_list args)
    {
        if (!format) { return -1; }
        return std::vfprintf(streamFor(stream), format, args);
    }

    // `std::_Winerror_map(int)` was VS2013's internal Windows-error-code -> message-text helper,
    // reached through `std::system_category().message()`. Later STLs renamed and reworked it, so
    // there is nothing to forward to.
    //
    // ⚠ IT RETURNS A CONSTANT STRING RATHER THAN nullptr, and that is the whole content of this
    // shim: the caller's next move is a `strlen` or a `std::string` construction, so a null would
    // turn NGX's error-reporting path — the one that runs when something has ALREADY gone wrong —
    // into an access violation. Losing the text of an error message is a cost worth paying; losing
    // the frame while NGX is trying to tell us why it failed is not. The code itself is still on
    // NGX's own log line via the callback in upscale_ngx.cpp, so nothing diagnosable is lost.
    const char* NgxWinerrorMap(int)
    {
        return "unknown error (msvcp120 std::_Winerror_map is not available in the UCRT; see "
               "mgeHost64.log's [ngx] lines for the code)";
    }
}

// The pointer variables themselves. `extern "C"` so the names are undecorated and the
// `/alternatename` directives above can find them.
extern "C" {
    void* mgeNgxImp_iob_func      = (void*)&NgxIobFunc;
    void* mgeNgxImp_vfprintf_s    = (void*)&NgxVfprintfS;
    void* mgeNgxImp_Winerror_map  = (void*)&NgxWinerrorMap;
}
