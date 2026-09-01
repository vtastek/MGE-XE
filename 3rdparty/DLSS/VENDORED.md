# NVIDIA DLSS SDK — vendored subset

**Source:** https://github.com/NVIDIA/DLSS, tag **v310.7.0** (commit `a291cc7`), fetched 2026-08-31.
**Licence:** `LICENSE.txt` (NVIDIA RTX SDKs License) — travels with these files, do not separate it.

Consumed by `tasks/forge-upscale.md` M1: the DLSS Super Resolution backend behind `IUpscaler`.

## What is here, and what is deliberately NOT

This is the **D3D12 Super Resolution** subset and nothing else. The upstream repo also ships Vulkan
headers, Ray Reconstruction (`dlssd`) and Frame Generation (`dlssg`); all three are omitted.

⚠⚠ **THE FRAME-GENERATION HEADERS AND `nvngx_dlssg.dll` ARE EXCLUDED ON PURPOSE, NOT BY OVERSIGHT.**
DLSS Frame Generation requires owning Present, and this architecture deliberately does not: the host
renders offscreen and hands a shared NT handle to the client, which imports it into DXVK and presents
(`tasks/forge-upscale.md` opens with why). FG cannot be made to work here at any effort. Not shipping
its headers is the cheapest way to stop someone spending a day discovering that.

Vulkan and Ray Reconstruction are omitted for the ordinary reason: nothing uses them.

```
include/nvsdk_ngx.h          the C API
include/nvsdk_ngx_defs.h     enums/typedefs           (nvsdk_ngx.h -> defs + params)
include/nvsdk_ngx_params.h   parameter block          (params -> defs)
include/nvsdk_ngx_helpers.h  D3D12 DLSS helpers       (helpers -> nvsdk_ngx.h + defs)
```
That is the complete dependency closure — verified by reading the `#include` lines, not assumed.

## The libraries, and a warning about the toolset

```
lib/Windows_x86_64/vs2013/x64/nvsdk_ngx_s.lib       static, release
lib/Windows_x86_64/vs2013/x64/nvsdk_ngx_s_dbg.lib   static, debug
lib/Windows_x86_64/vs2013/x64/nvsdk_ngx_d.lib       dynamic import, release
```

⚠ **NVIDIA SHIPS NOTHING NEWER THAN vs2013.** The upstream `lib/Windows_x86_64/` has exactly
`vs2010`, `vs2012` and `vs2013` (plus `khr`/`uwp` variants) — there is no vs2015+ build, and this
host is MSVC 14.4x. MSVC 2015 moved to the Universal CRT, so a vs2013 static library *can* drag in
legacy CRT symbols.

✅ **IT DOES NOT — CHECKED, NOT ASSUMED.** A probe TU that includes `d3d12.h` +
`nvsdk_ngx.h` + `nvsdk_ngx_helpers.h`, instantiates `NVSDK_NGX_D3D12_DLSS_Eval_Params` and
`NVSDK_NGX_DLSS_Create_Params`, and links `nvsdk_ngx_s.lib` compiles and links clean under
`cl /std:c++17 /EHsc` x64 (MSVC 14.44) and the resulting exe RUNS:

```
ngx headers ok: q=2 flags=5 evalsz=368 createsz=28
```

`flags=5` is `IsHDR | MVJittered` — confirming the flag M1's jittered motion vectors require
actually exists in this SDK version. So the static lib is the route; the escalations below are
recorded only in case a future SDK bump reopens the question.

1. `nvsdk_ngx_d.lib` (dynamic import) instead of the static one;
2. bind the handful of NGX entry points by `GetProcAddress` — which this codebase already does for
   PIX (`g_pixBegin`) and RenderDoc, and which has the independent virtue of degrading gracefully on
   a machine with no NVIDIA driver. That is a hard requirement anyway: the `IUpscaler` seam exists so
   AMD, Intel and the GTX-1080 tier are not left with nothing.

## The runtime

```
lib/Windows_x86_64/rel/nvngx_dlss.dll    58.9 MB — the RUNTIME, loaded BY the SDK
```
⚠ Not a substitute for the SDK and not interchangeable with it — the confusion that made this
milestone look unblocked when it was not. `mwdlss/nvngx_dlss.dll` was always the runtime only.
It has to sit where NGX is told to look (beside `mgeHost64.exe`) before DLSS will initialise;
it is NOT deployed there yet, because nothing loads it until the backend exists.

## Licence notes that bear on shipping

* §2a — the application must have material additional functionality beyond the SDK. MGE XE plainly does.
* §2b — **modifications and derivative works of SDK *source code* that are distributed must carry
  "This software contains source code provided by NVIDIA Corporation."** Nothing here is modified;
  if that ever changes, the notice goes with it.
* §4b — the SDK may not be distributed as a stand-alone product. It travels inside MGE XE or not at all.

⚠ `LICENSE.txt` is cp1252, not UTF-8 (smart quotes show as `?` in a UTF-8 reader). Copy it as BYTES;
re-encoding it is a modification of a licence file. [[project_cp1252_edit_tool_corruption]]
