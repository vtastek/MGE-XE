/*
 * terrain_horizon_occluder.c
 *
 * Reference implementation. See terrain_horizon_occluder.h for API and
 * coordinate conventions, and DESIGN_NOTE.md for rationale and integration.
 */

#include "terrain_horizon_occluder.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Helpers
 * ========================================================================= */

static inline float thc_minf(float a, float b) { return a < b ? a : b; }
static inline float thc_maxf(float a, float b) { return a > b ? a : b; }
static inline int   thc_mini(int a, int b)     { return a < b ? a : b; }
static inline int   thc_maxi(int a, int b)     { return a > b ? a : b; }

/*
 * Linear clip-space x for column c with resolution R:
 *   ndc_x(c) = -1 + 2 * c / (R - 1)
 *
 * When R is small this can be slightly coarser than 1 pixel; for the
 * curtain's purposes this is fine (the curtain only needs to stay below
 * the true silhouette, not match it to pixel precision).
 */
static inline float thc_col_to_ndc(int c, int resolution) {
    return -1.0f + 2.0f * (float)c / (float)(resolution - 1);
}

/* =========================================================================
 * Lifecycle
 * ========================================================================= */

int thc_horizon_init(thc_horizon_t *hz,
                     int resolution,
                     void *(*alloc_fn)(size_t),
                     void (*free_fn_unused)(void *))
{
    (void)free_fn_unused;
    if (!hz || resolution < 2) return 1;
    if (!alloc_fn) alloc_fn = malloc;

    hz->resolution = resolution;
    hz->h = (float *)alloc_fn(sizeof(float) * (size_t)resolution);
    hz->d = (float *)alloc_fn(sizeof(float) * (size_t)resolution);
    if (!hz->h || !hz->d) return 2;

    thc_horizon_reset(hz);
    return 0;
}

void thc_horizon_free(thc_horizon_t *hz, void (*free_fn)(void *))
{
    if (!hz) return;
    if (!free_fn) free_fn = free;
    free_fn(hz->h); hz->h = NULL;
    free_fn(hz->d); hz->d = NULL;
    hz->resolution = 0;
}

void thc_horizon_reset(thc_horizon_t *hz)
{
    if (!hz) return;
    for (int c = 0; c < hz->resolution; ++c) {
        hz->h[c] = THC_Y_BELOW;
        hz->d[c] = 0.0f;
    }
}

/* =========================================================================
 * Build: test + update one bintree-node projection against the horizon
 * ========================================================================= */

int thc_horizon_test_and_update(thc_horizon_t *hz,
                                const thc_node_projection_t *p)
{
    if (!hz || !p) return 0;

    const int c0 = thc_maxi(p->c0, 0);
    const int c1 = thc_mini(p->c1, hz->resolution - 1);
    if (c0 > c1) return 1;   /* degenerate / off-screen -> treat as culled */

    /* Phase 1: fully-occluded test.
     *
     * The node is strictly below the horizon at every column it spans if
     * p->y_upper < h[c] for all c in [c0, c1]. One pass over the span.
     */
    int fully_below = 1;
    for (int c = c0; c <= c1; ++c) {
        if (p->y_upper >= hz->h[c]) {
            fully_below = 0;
            break;
        }
    }
    if (fully_below) return 1;

    /* Phase 2: update.
     *
     * Where this node's upper silhouette exceeds the current horizon, push
     * h[c] up and record the node's far_depth for use in curtain building.
     * Using the node's y_upper here (a single value across the span) is a
     * conservative upper bound on the true silhouette inside the span.
     */
    for (int c = c0; c <= c1; ++c) {
        if (p->y_upper > hz->h[c]) {
            hz->h[c] = p->y_upper;
            hz->d[c] = p->far_depth;
        }
    }
    return 0;
}

/* =========================================================================
 * Simplify: budget-bounded Douglas-Peucker on (h, d) with a max-heap
 *
 * Work layout inside the workspace:
 *   [char keep[resolution]]
 *   [heap_item items[max_heap_capacity]]
 *
 * The heap is a binary max-heap keyed on normalized error. Each item is
 * an interval [a, b] along with the interior column with the largest
 * error relative to the (a, b) linear segment.
 * ========================================================================= */

typedef struct {
    int   a;
    int   b;
    int   split;       /* interior column with max error, or -1 if none */
    float err;         /* normalized error of split point */
} thc_heap_item_t;

typedef struct {
    thc_heap_item_t *items;
    int              size;
    int              capacity;
} thc_heap_t;

static void thc_heap_sift_up(thc_heap_t *hp, int i) {
    while (i > 0) {
        int parent = (i - 1) >> 1;
        if (hp->items[parent].err >= hp->items[i].err) break;
        thc_heap_item_t t = hp->items[parent];
        hp->items[parent] = hp->items[i];
        hp->items[i] = t;
        i = parent;
    }
}

static void thc_heap_sift_down(thc_heap_t *hp, int i) {
    for (;;) {
        int l = (i << 1) + 1;
        int r = l + 1;
        int best = i;
        if (l < hp->size && hp->items[l].err > hp->items[best].err) best = l;
        if (r < hp->size && hp->items[r].err > hp->items[best].err) best = r;
        if (best == i) break;
        thc_heap_item_t t = hp->items[best];
        hp->items[best] = hp->items[i];
        hp->items[i] = t;
        i = best;
    }
}

static int thc_heap_push(thc_heap_t *hp, thc_heap_item_t item) {
    if (hp->size >= hp->capacity) return 0;
    hp->items[hp->size] = item;
    thc_heap_sift_up(hp, hp->size);
    ++hp->size;
    return 1;
}

static int thc_heap_pop(thc_heap_t *hp, thc_heap_item_t *out) {
    if (hp->size == 0) return 0;
    *out = hp->items[0];
    --hp->size;
    if (hp->size > 0) {
        hp->items[0] = hp->items[hp->size];
        thc_heap_sift_down(hp, 0);
    }
    return 1;
}

/*
 * Scan (a, b) exclusive, find the interior column with the largest
 * normalized error from the linear interpolant between endpoints.
 * Normalization: error_x = |v_i - lerp(v_a, v_b, t)| / eps_x, score is
 * the max across (h, d) axes. If eps_x <= 0, that axis does not
 * contribute to scoring (effectively infinite tolerance).
 */
static thc_heap_item_t thc_compute_worst(const thc_horizon_t *hz,
                                         int a, int b,
                                         float eps_h, float eps_d)
{
    thc_heap_item_t item;
    item.a = a;
    item.b = b;
    item.split = -1;
    item.err = 0.0f;

    if (b - a < 2) return item;

    const float h_a = hz->h[a], h_b = hz->h[b];
    const float d_a = hz->d[a], d_b = hz->d[b];
    const float inv_span = 1.0f / (float)(b - a);
    const float inv_eps_h = (eps_h > 0.0f) ? 1.0f / eps_h : 0.0f;
    const float inv_eps_d = (eps_d > 0.0f) ? 1.0f / eps_d : 0.0f;

    int   best_idx = -1;
    float best_err = 0.0f;

    for (int i = a + 1; i < b; ++i) {
        float t = (float)(i - a) * inv_span;
        float h_lerp = h_a + t * (h_b - h_a);
        float d_lerp = d_a + t * (d_b - d_a);
        float e_h = fabsf(hz->h[i] - h_lerp) * inv_eps_h;
        float e_d = fabsf(hz->d[i] - d_lerp) * inv_eps_d;
        float e = (e_h > e_d) ? e_h : e_d;
        if (e > best_err) { best_err = e; best_idx = i; }
    }

    item.split = best_idx;
    item.err = best_err;
    return item;
}

size_t thc_simplify_workspace_required_bytes(int resolution, int max_samples)
{
    /*
     * keep[] bytes: resolution
     * heap capacity: up to 2 * max_samples intervals in the worst case
     *   (each split produces two new intervals; bounded by max_samples-1
     *   splits because every split adds exactly one "keep" slot). We size
     *   generously: 2 * max_samples.
     */
    size_t keep_bytes = (size_t)resolution;
    size_t heap_bytes = (size_t)(2 * max_samples) * sizeof(thc_heap_item_t);
    /* Align to 16 for clean boundaries. */
    keep_bytes = (keep_bytes + 15u) & ~(size_t)15u;
    return keep_bytes + heap_bytes;
}

int thc_horizon_simplify(const thc_horizon_t *hz,
                         thc_simplify_workspace_t *ws,
                         thc_sample_t *out,
                         int max_samples,
                         float eps_h,
                         float eps_d,
                         int tile_align)
{
    if (!hz || !ws || !out || max_samples < 2) return 0;

    const int R = hz->resolution;
    if (R < 2) return 0;

    const size_t need = thc_simplify_workspace_required_bytes(R, max_samples);
    if (!ws->memory || ws->size_bytes < need) return 0;

    /* Carve workspace. */
    char *base = (char *)ws->memory;
    char *keep = base;
    size_t keep_aligned = ((size_t)R + 15u) & ~(size_t)15u;
    thc_heap_t heap;
    heap.items = (thc_heap_item_t *)(base + keep_aligned);
    heap.size = 0;
    heap.capacity = 2 * max_samples;

    memset(keep, 0, (size_t)R);
    keep[0] = 1;
    keep[R - 1] = 1;
    int kept = 2;

    /* Seed with the whole-range interval. */
    thc_heap_item_t seed = thc_compute_worst(hz, 0, R - 1, eps_h, eps_d);
    if (seed.split >= 0 && seed.err > 1.0f) {
        thc_heap_push(&heap, seed);
    }

    while (kept < max_samples) {
        thc_heap_item_t top;
        if (!thc_heap_pop(&heap, &top)) break;
        if (top.err <= 1.0f) break;           /* below threshold -> done */
        if (top.split <= top.a || top.split >= top.b) continue;

        keep[top.split] = 1;
        ++kept;

        thc_heap_item_t L = thc_compute_worst(hz, top.a, top.split, eps_h, eps_d);
        thc_heap_item_t Rg = thc_compute_worst(hz, top.split, top.b, eps_h, eps_d);
        if (L.split >= 0 && L.err > 1.0f)  thc_heap_push(&heap, L);
        if (Rg.split >= 0 && Rg.err > 1.0f) thc_heap_push(&heap, Rg);
    }

    /*
     * Optional tile alignment: snap non-endpoint kept columns to the
     * nearest multiple of tile_align. We walk left-to-right, snap, and
     * de-duplicate any collisions.
     *
     * The snap may move a column up or down by up to tile_align/2. Since
     * the curtain building later takes per-segment conservative values,
     * misalignment by half a tile is absorbed harmlessly.
     */
    if (tile_align > 1) {
        for (int c = 1; c < R - 1; ++c) {
            if (!keep[c]) continue;
            int snapped = ((c + tile_align / 2) / tile_align) * tile_align;
            if (snapped <= 0) snapped = 1;
            if (snapped >= R - 1) snapped = R - 2;
            if (snapped != c) {
                keep[c] = 0;
                if (!keep[snapped]) keep[snapped] = 1;
            }
        }
    }

    /* Gather kept columns into the output array, sorted. */
    int n = 0;
    for (int c = 0; c < R && n < max_samples; ++c) {
        if (!keep[c]) continue;
        out[n].col   = c;
        out[n].ndc_x = thc_col_to_ndc(c, R);
        out[n].h     = hz->h[c];
        out[n].d     = hz->d[c];
        ++n;
    }

    /* If simplify happened to return fewer than 2 (pathological R), force 2. */
    if (n < 2) {
        out[0].col = 0;
        out[0].ndc_x = -1.0f;
        out[0].h = hz->h[0];
        out[0].d = hz->d[0];
        out[1].col = R - 1;
        out[1].ndc_x = 1.0f;
        out[1].h = hz->h[R - 1];
        out[1].d = hz->d[R - 1];
        n = 2;
    }
    return n;
}

/* =========================================================================
 * Emit curtains as clip-space triangles
 *
 * Per segment [s_i, s_{i+1}]:
 *
 *      TL ----- TR          TL = (s_i.ndc_x,   y_top, z)
 *      |\       |           TR = (s_{i+1}.ndc_x, y_top, z)
 *      |  \     |           BL = (s_i.ndc_x,   y_bottom, z)
 *      |    \   |           BR = (s_{i+1}.ndc_x, y_bottom, z)
 *      |      \ |
 *      BL ----- BR
 *
 *      y_top = min(h_i, h_{i+1})   (conservative lower bound on silhouette)
 *      z     = max(d_i, d_{i+1})   (conservative farther bound on depth)
 *
 * Triangle 1: TL, BL, BR  (ccw, y-up, looking down -z from camera)
 * Triangle 2: TL, BR, TR
 *
 * y_bottom typically just below -1 (e.g. -1.1) so the strip extends past
 * the screen. Any polygon fully off-screen is cheap for MSOC to reject at
 * the clip stage, so slight overhang is free.
 *
 * All vertices have w = 1.0 to signal "pre-transformed" when MSOC is
 * invoked with an identity transform.
 * ========================================================================= */

static inline thc_curtain_vertex_t thc_v(float x, float y, float z) {
    thc_curtain_vertex_t v = { x, y, z, 1.0f };
    return v;
}

int thc_emit_curtains(const thc_sample_t *samples,
                      int n_samples,
                      float ndc_y_bottom,
                      thc_curtain_vertex_t *out_verts,
                      int out_verts_capacity)
{
    if (!samples || !out_verts || n_samples < 2) return 0;

    int segs = n_samples - 1;
    int need_verts = 6 * segs;
    if (out_verts_capacity < need_verts) return 0;

    int w = 0;
    for (int i = 0; i < segs; ++i) {
        const thc_sample_t *s0 = &samples[i];
        const thc_sample_t *s1 = &samples[i + 1];

        float y_top = thc_minf(s0->h, s1->h);
        float z     = thc_maxf(s0->d, s1->d);

        /* If neither column has been touched (h == THC_Y_BELOW), skip the
         * segment entirely - there is no silhouette here. */
        if (y_top <= THC_Y_BELOW * 0.5f) continue;

        thc_curtain_vertex_t TL = thc_v(s0->ndc_x, y_top,        z);
        thc_curtain_vertex_t TR = thc_v(s1->ndc_x, y_top,        z);
        thc_curtain_vertex_t BL = thc_v(s0->ndc_x, ndc_y_bottom, z);
        thc_curtain_vertex_t BR = thc_v(s1->ndc_x, ndc_y_bottom, z);

        /* Triangle 1 */
        out_verts[w++] = TL;
        out_verts[w++] = BL;
        out_verts[w++] = BR;
        /* Triangle 2 */
        out_verts[w++] = TL;
        out_verts[w++] = BR;
        out_verts[w++] = TR;
    }

    return w / 3;
}

/* =========================================================================
 * Utility: project a world-space AABB to a conservative horizon node
 * projection. Implementation detail: we transform the 8 corners with the
 * view-projection matrix, divide by w for on-screen corners, and take the
 * bbox of the resulting clip-space points.
 *
 * Caller is responsible for frustum culling before invoking this; behind-
 * plane corners are handled conservatively (clamped to near plane) but
 * that is a degenerate case and should not dominate in a well-behaved
 * renderer.
 * ========================================================================= */

static inline void thc_mat4_mul_vec4(const float m[16],
                                     float x, float y, float z, float w,
                                     float out[4])
{
    /* Row-major * column vector. */
    out[0] = m[0]*x  + m[1]*y  + m[2]*z  + m[3]*w;
    out[1] = m[4]*x  + m[5]*y  + m[6]*z  + m[7]*w;
    out[2] = m[8]*x  + m[9]*y  + m[10]*z + m[11]*w;
    out[3] = m[12]*x + m[13]*y + m[14]*z + m[15]*w;
}

int thc_project_aabb(const float vp[16],
                     const float aabb_min[3],
                     const float aabb_max[3],
                     int resolution,
                     float z_far,
                     thc_node_projection_t *out)
{
    if (!vp || !aabb_min || !aabb_max || !out) return 0;
    if (resolution < 2) return 0;

    float x_min = +1e30f, x_max = -1e30f;
    float y_min = +1e30f, y_max = -1e30f;
    float z_max = -1e30f;
    int   any_in_front = 0;

    for (int i = 0; i < 8; ++i) {
        float x = (i & 1) ? aabb_max[0] : aabb_min[0];
        float y = (i & 2) ? aabb_max[1] : aabb_min[1];
        float z = (i & 4) ? aabb_max[2] : aabb_min[2];
        float v[4];
        thc_mat4_mul_vec4(vp, x, y, z, 1.0f, v);

        /* Guard against w <= 0 (behind camera). Snap to a small positive w
         * so we get a conservative, albeit imprecise, screen-space extent.
         * If _no_ corner is in front, the AABB is entirely behind the camera
         * and we bail. */
        if (v[3] > 1e-6f) {
            any_in_front = 1;
            float inv_w = 1.0f / v[3];
            float nx = v[0] * inv_w;
            float ny = v[1] * inv_w;
            float nz = v[2] * inv_w;
            if (nx < x_min) x_min = nx;
            if (nx > x_max) x_max = nx;
            if (ny < y_min) y_min = ny;
            if (ny > y_max) y_max = ny;
            if (nz > z_max) z_max = nz;
        }
    }

    if (!any_in_front) return 0;

    /* Clamp clip-x to [-1, 1] and map to column indices. */
    if (x_min < -1.0f) x_min = -1.0f;
    if (x_max > +1.0f) x_max = +1.0f;
    if (x_min > x_max) return 0;

    const float half_span = 0.5f * (float)(resolution - 1);
    int c0 = (int)floorf((x_min + 1.0f) * half_span);
    int c1 = (int)ceilf ((x_max + 1.0f) * half_span);
    if (c0 < 0) c0 = 0;
    if (c1 >= resolution) c1 = resolution - 1;

    /* If clamped clip-y is entirely below -1 the AABB is off the bottom
     * and will not contribute meaningfully to the horizon; still valid
     * to return the projection and let the caller filter. */
    if (y_max < -1.0f) y_max = -1.0f;
    if (y_max > +1.0f) y_max = +1.0f;

    /* Clamp far_depth by caller-provided z_far (pre-projection linear far
     * or NDC z=1 depending on convention). The caller chooses the scale. */
    if (z_max > z_far) z_max = z_far;

    out->c0 = c0;
    out->c1 = c1;
    out->y_upper = y_max;
    out->far_depth = z_max;
    return 1;
}
