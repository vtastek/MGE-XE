# Third-party notices

MGE XE itself is GPLv2 (see `license.txt`). This file lists third-party material redistributed in
the source tree or in a shipped binary, together with the notices those licences require us to
carry. Anything added here must state **what** was taken, **where it lives**, and **which clause**
obliges the notice — a list of names without those three is not a notice, it is a bibliography.

---

## Hosek-Wilkie analytic sky-dome model — coefficient dataset

**What:** the RGB coefficient dataset for the analytic skylight model of

> Lukas Hosek and Alexander Wilkie, *"An Analytic Model for Full Spectral Sky-Dome Radiance"*,
> ACM SIGGRAPH 2012 — and *"Adding a Solar Radiance Function to the Hosek Skylight Model"*,
> IEEE Computer Graphics and Applications, 2013.
> <http://cgg.mff.cuni.cz/projects/SkylightModelling/>

**Where:** `mgeHost64/hosek_data.h` — a copy of `ArHosekSkyModelData_RGB.h` from the authors'
reference release 1.4a, values verbatim (the file's own preamble lists the four mechanical changes:
an include guard, `static const` linkage under a namespace, our array names, and CRLF → LF). The
model is *evaluated* by our own code — `mgeHost64/hosek.h` (host) and
`mgeHost64/shaders/FSL/hosek.h.fsl` (shader) — which is why only the data file is redistributed and
none of the reference implementation. `tools/hosek_reference.c` regenerates the values that gate it.

**Licence:** 3-clause BSD. Clause 1 requires the copyright notice and disclaimer to be retained in
redistributed **source**, which `mgeHost64/hosek_data.h` does verbatim at the top of the file.
Clause 2 requires them to be reproduced in the documentation accompanying a redistributed
**binary**, which is what this entry is for. Clause 3 forbids using the authors' names to endorse
this project, and we do not.

```
Copyright (c) 2012 - 2013, Lukas Hosek and Alexander Wilkie
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * None of the names of the contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

*Other vendored dependencies (The Forge, MWSE/SharedSE, masked-occlusion-culling, imgui, niflib, …)
carry their own licence files inside `3rdparty/` and in their own repositories; this file records
only material whose licence obliges a notice to travel with the shipped binary.*
