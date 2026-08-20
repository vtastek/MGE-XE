/* mgeHost64 — regenerate the reference values that gate mgeHost64/hosek.h.
 *
 * WHY THIS FILE EXISTS. hosek.h contains three cooked Hosek-Wilkie configurations and twelve
 * radiances that its init self-test compares against, and those numbers were NOT computed by us —
 * they came out of the authors' own reference implementation. A gate whose expected values cannot
 * be regenerated is not a gate; it is a second set of constants to mistrust. This is how to
 * regenerate them.
 *
 * WHAT YOU NEED. The Hosek & Wilkie release 1.4a (ArHosekSkyModel.{c,h} +
 * ArHosekSkyModelData_{Spectral,CIEXYZ,RGB}.h), from
 *     http://cgg.mff.cuni.cz/projects/SkylightModelling/
 * It is 3-clause BSD; mgeHost64/hosek_data.h is a copy of the RGB data file from it (see the notice
 * there and THIRD-PARTY-NOTICES.md). The .c is NOT vendored — only the data is — because the host
 * has its own evaluation and vendoring a second one is precisely the drift this whole step is about.
 *
 * BUILD AND RUN (from a directory holding the release's files plus this one):
 *     gcc -std=c99 -O2 -o hosek_reference hosek_reference.c ArHosekSkyModel.c -lm
 *     ./hosek_reference
 * Paste the output over kRefCases[] in mgeHost64/hosek.h. The host's own gate then reports the
 * agreement on startup as `[forge-hosek] coefficient gate OK`, which should read ~2e-6 (the
 * reference is double precision and the host is float; anything larger is a real difference).
 *
 * ⚠ THE PROBE POINTS ARE CHOSEN, NOT ARBITRARY. Between the three cases they exercise every path in
 * hosek.h's cook(): both albedo halves, integer AND fractional turbidity, the elevation warp near
 * both ends of its domain, and the `turbidity == 10` early-out's neighbour. Adding a case is fine;
 * removing one costs coverage that is not obvious from the numbers.
 */
#include "ArHosekSkyModel.h"
#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846

static void one(double T, double A, double elevDeg, const char* tag)
{
    double elev = elevDeg * PI / 180.0;
    ArHosekSkyModelState* s = arhosek_rgb_skymodelstate_alloc_init(T, A, elev);
    printf("    { %.6ff, %.6ff, %.9ff,   // %s\n", T, A, elev, tag);
    printf("      {\n");
    for (int c = 0; c < 3; ++c) {
        printf("        {");
        for (int i = 0; i < 9; ++i) { printf(" %.9ef,", s->configs[c][i]); }
        printf(" },\n");
    }
    printf("      },\n");
    printf("      { %.9ef, %.9ef, %.9ef },\n", s->radiances[0], s->radiances[1], s->radiances[2]);
    /* The four probes: zenith, a near-horizon point 90 degrees from the sun, the circumsolar peak,
     * and a mid-sky point on the far side. Stored as (cos(theta), gamma) because that is exactly
     * what Hosek::evalNative takes — no conversion between the generator and the gate. */
    double th[4] = { 0.0, PI * 0.5 * 0.99, PI * 0.5 - elev, PI * 0.25 };
    double gm[4] = { PI * 0.5 - elev, PI * 0.5, 0.02, PI * 0.75 };
    printf("      {\n");
    for (int p = 0; p < 4; ++p) {
        printf("        { %.9ef, %.9ef, { ", cos(th[p]), gm[p]);
        for (int c = 0; c < 3; ++c) {
            printf("%.9ef, ", arhosek_tristim_skymodel_radiance(s, th[p], gm[p], c));
        }
        printf("} },\n");
    }
    printf("      },\n");
    printf("    },\n");
    arhosekskymodelstate_free(s);
}

int main(void)
{
    one(3.0, 0.10, 41.34, "the reference clear day: T 3, ash albedo, MW's sunDir.z = -0.66");
    one(6.4, 0.35, 8.00,  "hazy + bright ground + low sun: exercises BOTH interpolations at once");
    one(1.0, 0.00, 89.90, "the corner of the domain: T and albedo at their minima, sun overhead");
    return 0;
}
