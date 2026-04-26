/*
 * terrain_horizon_occluder.h
 *
 * Conservative screen-space terrain occluder from a 1D horizon buffer.
 *
 * Purpose
 * -------
 * Given a ROAM-tessellated terrain traversed front-to-back, build a 1D
 * horizon buffer h(x) + depth buffer d(x), simplify to a small number of
 * adaptive samples, and emit a tiny set of screen-space "curtain" triangles
 * (in clip space) that can be submitted to an existing occlusion-culling
 * system (e.g. Intel Masked Occlusion Culling) alongside static occluders.
 *
 * The resulting unified occlusion buffer can then be queried in a single
 * pass by the scene graph for AABB visibility tests.
 *
 * Coordinate conventions
 * ----------------------
 *   - Clip-space x, y follow OpenGL NDC after perspective divide:
 *     x in [-1, +1] left -> right
 *     y in [-1, +1] bottom -> top  (y-UP)
 *   - Depth (d) is a scalar where LARGER = FARTHER. Callers using a reverse-Z
 *     pipeline should negate or invert d before passing it in.
 *   - Column indices run [0, resolution-1], mapped linearly to clip-space x.
 *
 * Thread-safety
 * -------------
 * All functions are thread-compatible, not thread-safe. One horizon per
 * frame-worker is the intended usage.
 *
 * Units / precision
 * -----------------
 * All state is float32. Resolutions are typically 512 or 1024 columns.
 * Memory footprint: ~8 * resolution bytes + simplify workspace.
 */

#ifndef TERRAIN_HORIZON_OCCLUDER_H
#define TERRAIN_HORIZON_OCCLUDER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ----- Types ----- */

/*
 * The horizon buffer. h[c] is the maximum clip-space y observed for any
 * terrain fragment whose projected bbox spans column c. d[c] is the
 * corresponding far-depth contributed by the bintree node that set h[c].
 *
 * Initial state (after thc_horizon_reset): h[c] = THC_Y_BELOW, d[c] = 0.
 */
typedef struct thc_horizon {
    int    resolution;
    float *h;                 /* [resolution] */
    float *d;                 /* [resolution] */
} thc_horizon_t;

/*
 * Projection of a single ROAM bintree node in clip space. Produced by the
 * caller's ROAM traversal. Column range is inclusive on both ends.
 *
 *   c0, c1     : clip-x column range, 0 <= c0 <= c1 < resolution
 *   y_upper    : clip-space y of the node's upper silhouette
 *                (the highest point in the sub-patch that could occlude)
 *   far_depth  : the deepest (farthest) clip-space z contributed by the
 *                node. Conservative choice: the max of all vertex depths
 *                in the sub-patch.
 */
typedef struct thc_node_projection {
    int   c0;
    int   c1;
    float y_upper;
    float far_depth;
} thc_node_projection_t;

/*
 * A simplified sample chosen from the horizon buffer. col is the source
 * column index; ndc_x, h, d are the corresponding clip-space values,
 * pre-computed by the simplifier.
 */
typedef struct thc_sample {
    int   col;
    float ndc_x;
    float h;
    float d;
} thc_sample_t;

/*
 * One clip-space vertex of the curtain output, suitable for direct submit
 * to MSOC's RenderTriangles with an identity matrix. w is always 1.0.
 */
typedef struct thc_curtain_vertex {
    float x, y, z, w;
} thc_curtain_vertex_t;

/*
 * Preallocated scratch for the DP simplifier. Owned by the caller, lives
 * for at least one frame. Size proportional to horizon resolution + sample
 * budget. See thc_simplify_workspace_required_bytes().
 */
typedef struct thc_simplify_workspace {
    void  *memory;
    size_t size_bytes;
} thc_simplify_workspace_t;

/* ----- Constants ----- */

/*
 * Sentinel for "no terrain seen in this column yet". Chosen well below any
 * plausible clip-space y to guarantee that h[c] = THC_Y_BELOW is reliably
 * below any real silhouette and never causes a false cull.
 */
#define THC_Y_BELOW  (-1e30f)

/* ----- Lifecycle ----- */

/*
 * Initialize an empty horizon of the given column resolution. Allocates
 * h and d arrays on the provided allocator. Returns 0 on success, nonzero
 * on allocation failure.
 *
 * The allocator callbacks may be NULL to use malloc/free.
 */
int  thc_horizon_init (thc_horizon_t *hz,
                       int resolution,
                       void *(*alloc_fn)(size_t),
                       void  (*free_fn)(void *));

void thc_horizon_free (thc_horizon_t *hz,
                       void (*free_fn)(void *));

/*
 * Reset all columns to the initial state. Call at the start of every frame
 * before the ROAM traversal.
 */
void thc_horizon_reset(thc_horizon_t *hz);

/* ----- Build phase: invoked during ROAM traversal ----- */

/*
 * Test a bintree node's projection against the current horizon and, if not
 * fully occluded, update the horizon in-place.
 *
 * Returns:
 *   1  if the node (and every point it contains) is strictly below the
 *      current horizon at every column it spans. Caller should prune this
 *      subtree and NOT recurse; nothing inside it can ever rise above the
 *      horizon.
 *   0  otherwise. The horizon has been updated where p->y_upper exceeded
 *      the prior value.
 *
 * Semantics rely on strict front-to-back traversal order. A node that
 * returns 0 and gets recursed into will have its children tested against
 * the updated horizon, which now includes contributions from this node
 * and anything nearer.
 */
int  thc_horizon_test_and_update(thc_horizon_t *hz,
                                 const thc_node_projection_t *p);

/* ----- Simplify phase: invoked once per frame after ROAM walk ----- */

/*
 * Required workspace size for thc_horizon_simplify given a horizon
 * resolution and a sample budget. Call once at startup to allocate.
 */
size_t thc_simplify_workspace_required_bytes(int resolution, int max_samples);

/*
 * Approximate the horizon with up to max_samples piecewise-linear samples
 * using a budget-bounded Douglas-Peucker over a joint (h, d) error metric.
 *
 * Output samples are sorted by column. Endpoints (col 0 and resolution-1)
 * are always included; the remainder are chosen by greedy max-error.
 *
 * Parameters
 *   eps_h, eps_d : split thresholds in clip-space units. A subinterval is
 *                  split when max(|h_err|/eps_h, |d_err|/eps_d) > 1.
 *                  Set eps_d very large if you do not want depth-driven
 *                  splitting. Setting either to 0 or negative disables
 *                  threshold-based early stop (runs to max_samples).
 *   tile_align   : if > 1, snap each chosen column to the nearest multiple
 *                  of tile_align (preserving endpoint columns). Used to
 *                  align curtain quads to MSOC's tile column grid, e.g. 32.
 *                  Pass 0 or 1 to disable alignment.
 *
 * Returns the number of samples written, always >= 2.
 */
int thc_horizon_simplify(const thc_horizon_t *hz,
                         thc_simplify_workspace_t *ws,
                         thc_sample_t *out,
                         int max_samples,
                         float eps_h,
                         float eps_d,
                         int tile_align);

/* ----- Emit phase: invoked once per frame after simplify ----- */

/*
 * Emit clip-space triangles representing the curtain strip. For n samples
 * this produces (n - 1) quads = 2 * (n - 1) triangles = 6 * (n - 1) verts.
 *
 * Per curtain segment [s_i, s_{i+1}]:
 *   y_top = min(s_i.h, s_{i+1}.h)    (conservative: stays below true horizon)
 *   z     = max(s_i.d, s_{i+1}.d)    (conservative: stays at/behind terrain)
 *
 * ndc_y_bottom is the clip-space y for the bottom edge of the strip.
 * Recommend -1.1f (just below screen) to avoid edge precision issues.
 *
 * Winding is counter-clockwise. For MSOC, use BACKFACE_NONE to be robust
 * to orientation.
 *
 * Returns the triangle count written, or 0 if n_samples < 2.
 */
int thc_emit_curtains(const thc_sample_t *samples,
                      int n_samples,
                      float ndc_y_bottom,
                      thc_curtain_vertex_t *out_verts,
                      int out_verts_capacity);

/* ----- Utility: map a node's world-space bbox to thc_node_projection ----- */

/*
 * Given a 4x4 view-projection matrix (row-major, multiplied on the right
 * with column-vector positions) and an axis-aligned bounding box in world
 * space, compute a conservative node projection.
 *
 * The projection encloses all 8 bbox corners after clip + perspective
 * divide. Behind-plane corners are clamped to the near plane (the caller
 * is responsible for frustum-culling before calling this).
 *
 * Returns 1 if the projection is non-degenerate, 0 if the bbox is behind
 * the camera or entirely off-screen (caller should skip the subtree).
 */
int thc_project_aabb(const float vp[16],
                     const float aabb_min[3],
                     const float aabb_max[3],
                     int resolution,
                     float z_far,
                     thc_node_projection_t *out);

#ifdef __cplusplus
}
#endif

#endif /* TERRAIN_HORIZON_OCCLUDER_H */
