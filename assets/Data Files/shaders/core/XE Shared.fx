// XE Shared.fx
// The shared-uniform carrier (tasks/startup-time.md S2).
//
// DistantLand::setupCommonEffect writes the per-frame shared uniforms (view, proj, sun, fog, wind,
// time...) through ONE effect in the pool, and every effect compiled into that pool (XE
// FixedFuncEmu.fx) reads them. That carrier used to be XE Main.fx, the DX9 distant-land, water,
// grass and sky renderer. None of its techniques is drawn any more, and it cost ~2.9 s of D3DX
// compile under the "Loading MGE XE..." bar. The declarations live in XE Common.fx, so this
// effect is that include plus the one empty technique D3DX insists on ("There were no
// techniques" fails the compile). Nothing ever draws it.

#include "XE Common.fx"

technique SharedParamsOnly {
    pass P0 {
    }
}
