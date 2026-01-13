/*
 * omega_raytrace - BVH-accelerated raytracing for CS2 visibility checks
 * 
 * Uses Bounding Volume Hierarchy for O(log n) ray queries instead of O(n)
 * Möller-Trumbore algorithm for ray-triangle intersection
 * Beer-Lambert law for smoke density accumulation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

/* ========== Triangle/BVH Structures ========== */

typedef struct {
    float v0[3];
    float v1[3];
    float v2[3];
} Triangle;

typedef struct {
    float min[3];
    float max[3];
} AABB;

typedef struct {
    AABB bounds;
    int left;
    int right;
    int tri_count;
    int _pad;
} BVHNode;

/* ========== Smoke Voxel Structures ========== */

typedef struct {
    float pos[3];       /* World position (center of voxel) */
    float opacity;      /* 0.0 - 1.0 */
} SmokeVoxel;

typedef struct {
    SmokeVoxel* voxels;
    int count;
    int capacity;
    float half_size;    /* Half the voxel cube size */
} SmokeCloud;

/* ========== Global State ========== */

/* World geometry */
static Triangle* g_triangles = NULL;
static int g_tri_count = 0;
static BVHNode* g_bvh_nodes = NULL;
static int g_bvh_node_count = 0;
static int* g_tri_indices = NULL;

/* Smoke voxels */
static SmokeCloud* g_smokes = NULL;
static int g_smoke_count = 0;
static int g_smoke_capacity = 0;

/* Constants */
#define MAX_LEAF_TRIS 3
#define EPSILON 1e-6f
#define MAX_SMOKES 16
#define DEFAULT_VOXEL_HALF_SIZE 10.0f
#define DEFAULT_EXTINCTION 6.0f
#define DEFAULT_SMOKE_THRESHOLD 0.7f

/* ========== Math Helpers ========== */

static inline float minf(float a, float b) { return a < b ? a : b; }
static inline float maxf(float a, float b) { return a > b ? a : b; }
static inline float clampf(float v, float lo, float hi) { return minf(maxf(v, lo), hi); }

/* ========== AABB Helpers ========== */

static inline void aabb_init(AABB* box) {
    box->min[0] = box->min[1] = box->min[2] = FLT_MAX;
    box->max[0] = box->max[1] = box->max[2] = -FLT_MAX;
}

static inline void aabb_expand_point(AABB* box, const float* p) {
    for (int i = 0; i < 3; i++) {
        if (p[i] < box->min[i]) box->min[i] = p[i];
        if (p[i] > box->max[i]) box->max[i] = p[i];
    }
}

static inline void aabb_expand_triangle(AABB* box, const Triangle* tri) {
    aabb_expand_point(box, tri->v0);
    aabb_expand_point(box, tri->v1);
    aabb_expand_point(box, tri->v2);
}

static inline void aabb_get_centroid(const Triangle* tri, float* centroid) {
    centroid[0] = (tri->v0[0] + tri->v1[0] + tri->v2[0]) / 3.0f;
    centroid[1] = (tri->v0[1] + tri->v1[1] + tri->v2[1]) / 3.0f;
    centroid[2] = (tri->v0[2] + tri->v1[2] + tri->v2[2]) / 3.0f;
}

/* Ray-AABB intersection (slab method) - returns 1 if intersect, 0 otherwise */
static inline int ray_aabb_intersect(
    const float* origin, const float* inv_dir,
    const AABB* box, float t_max
) {
    float t1, t2, t_min = 0.0f;
    
    for (int i = 0; i < 3; i++) {
        t1 = (box->min[i] - origin[i]) * inv_dir[i];
        t2 = (box->max[i] - origin[i]) * inv_dir[i];
        
        if (t1 > t2) {
            float tmp = t1; t1 = t2; t2 = tmp;
        }
        
        if (t1 > t_min) t_min = t1;
        if (t2 < t_max) t_max = t2;
        
        if (t_min > t_max) return 0;
    }
    
    return 1;
}

/*
 * Ray-AABB intersection with t-values for smoke density calculation
 * Returns: 1 if intersects, 0 otherwise
 * out_t0, out_t1: entry and exit t-values (clamped to [0, 1] for segment)
 */
static int segment_aabb_intersect(
    const float* p0, const float* dir, float seg_len,
    const float* box_min, const float* box_max,
    float* out_t0, float* out_t1
) {
    float t0 = 0.0f, t1 = 1.0f;
    
    for (int i = 0; i < 3; i++) {
        float d = dir[i];
        
        if (fabsf(d) < EPSILON) {
            /* Ray parallel to slab */
            if (p0[i] < box_min[i] || p0[i] > box_max[i]) {
                return 0;
            }
        } else {
            float inv = 1.0f / d;
            float t_near = (box_min[i] - p0[i]) * inv;
            float t_far = (box_max[i] - p0[i]) * inv;
            
            if (t_near > t_far) {
                float tmp = t_near;
                t_near = t_far;
                t_far = tmp;
            }
            
            t0 = maxf(t0, t_near);
            t1 = minf(t1, t_far);
            
            if (t0 > t1) return 0;
        }
    }
    
    *out_t0 = clampf(t0, 0.0f, 1.0f);
    *out_t1 = clampf(t1, 0.0f, 1.0f);
    return *out_t0 <= *out_t1;
}

/* ========== BVH Construction ========== */

static int* g_sort_axis_ptr;
static Triangle* g_sort_tris_ptr;

static int compare_centroid(const void* a, const void* b) {
    int ia = *(const int*)a;
    int ib = *(const int*)b;
    int axis = *g_sort_axis_ptr;
    
    float ca[3], cb[3];
    aabb_get_centroid(&g_sort_tris_ptr[ia], ca);
    aabb_get_centroid(&g_sort_tris_ptr[ib], cb);
    
    if (ca[axis] < cb[axis]) return -1;
    if (ca[axis] > cb[axis]) return 1;
    return 0;
}

static int bvh_build_recursive(int start, int count) {
    if (g_bvh_node_count >= g_tri_count * 2) {
        return -1;
    }
    
    int node_idx = g_bvh_node_count++;
    BVHNode* node = &g_bvh_nodes[node_idx];
    
    aabb_init(&node->bounds);
    for (int i = 0; i < count; i++) {
        aabb_expand_triangle(&node->bounds, &g_triangles[g_tri_indices[start + i]]);
    }
    
    if (count <= MAX_LEAF_TRIS) {
        node->left = -1;
        node->right = start;
        node->tri_count = count;
        return node_idx;
    }
    
    float extent[3];
    int axis = 0;
    for (int i = 0; i < 3; i++) {
        extent[i] = node->bounds.max[i] - node->bounds.min[i];
        if (extent[i] > extent[axis]) axis = i;
    }
    
    g_sort_axis_ptr = &axis;
    g_sort_tris_ptr = g_triangles;
    qsort(&g_tri_indices[start], count, sizeof(int), compare_centroid);
    
    int mid = count / 2;
    
    node->tri_count = 0;
    node->left = bvh_build_recursive(start, mid);
    node->right = bvh_build_recursive(start + mid, count - mid);
    
    return node_idx;
}

static void bvh_build(void) {
    if (g_bvh_nodes) {
        free(g_bvh_nodes);
        g_bvh_nodes = NULL;
    }
    if (g_tri_indices) {
        free(g_tri_indices);
        g_tri_indices = NULL;
    }
    g_bvh_node_count = 0;
    
    if (g_tri_count == 0) return;
    
    g_bvh_nodes = (BVHNode*)malloc(sizeof(BVHNode) * g_tri_count * 2);
    if (!g_bvh_nodes) return;
    
    g_tri_indices = (int*)malloc(sizeof(int) * g_tri_count);
    if (!g_tri_indices) {
        free(g_bvh_nodes);
        g_bvh_nodes = NULL;
        return;
    }
    
    for (int i = 0; i < g_tri_count; i++) {
        g_tri_indices[i] = i;
    }
    
    bvh_build_recursive(0, g_tri_count);
}

/* ========== Ray-Triangle Intersection (Möller-Trumbore) ========== */

static inline int ray_triangle_intersect(
    const float* origin,
    const float* dir,
    const Triangle* tri,
    float* t_out
) {
    float e1[3], e2[3], h[3], s[3], q[3];
    float a, f, u, v, t;
    
    e1[0] = tri->v1[0] - tri->v0[0];
    e1[1] = tri->v1[1] - tri->v0[1];
    e1[2] = tri->v1[2] - tri->v0[2];
    
    e2[0] = tri->v2[0] - tri->v0[0];
    e2[1] = tri->v2[1] - tri->v0[1];
    e2[2] = tri->v2[2] - tri->v0[2];
    
    h[0] = dir[1] * e2[2] - dir[2] * e2[1];
    h[1] = dir[2] * e2[0] - dir[0] * e2[2];
    h[2] = dir[0] * e2[1] - dir[1] * e2[0];
    
    a = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
    
    if (a > -EPSILON && a < EPSILON) return 0;
    
    f = 1.0f / a;
    
    s[0] = origin[0] - tri->v0[0];
    s[1] = origin[1] - tri->v0[1];
    s[2] = origin[2] - tri->v0[2];
    
    u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
    if (u < 0.0f || u > 1.0f) return 0;
    
    q[0] = s[1] * e1[2] - s[2] * e1[1];
    q[1] = s[2] * e1[0] - s[0] * e1[2];
    q[2] = s[0] * e1[1] - s[1] * e1[0];
    
    v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);
    if (v < 0.0f || u + v > 1.0f) return 0;
    
    t = f * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]);
    
    if (t > EPSILON) {
        *t_out = t;
        return 1;
    }
    
    return 0;
}

/* ========== BVH Traversal ========== */

static int bvh_trace_ray(
    const float* origin,
    const float* dir,
    float max_dist
) {
    if (!g_bvh_nodes || g_bvh_node_count == 0) return 0;
    
    float inv_dir[3];
    inv_dir[0] = 1.0f / (fabsf(dir[0]) > EPSILON ? dir[0] : EPSILON);
    inv_dir[1] = 1.0f / (fabsf(dir[1]) > EPSILON ? dir[1] : EPSILON);
    inv_dir[2] = 1.0f / (fabsf(dir[2]) > EPSILON ? dir[2] : EPSILON);
    
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        BVHNode* node = &g_bvh_nodes[node_idx];
        
        if (!ray_aabb_intersect(origin, inv_dir, &node->bounds, max_dist)) {
            continue;
        }
        
        if (node->tri_count > 0) {
            for (int i = 0; i < node->tri_count; i++) {
                int tri_idx = g_tri_indices[node->right + i];
                float t;
                if (ray_triangle_intersect(origin, dir, &g_triangles[tri_idx], &t)) {
                    if (t > EPSILON && t < max_dist - EPSILON) {
                        return 1;
                    }
                }
            }
        } else {
            if (node->left >= 0 && stack_ptr < 63) {
                stack[stack_ptr++] = node->left;
            }
            if (node->right >= 0 && stack_ptr < 63) {
                stack[stack_ptr++] = node->right;
            }
        }
    }
    
    return 0;
}

/* ========== Smoke Voxel Functions ========== */

/*
 * Trace ray through smoke voxels using Beer-Lambert law
 * Returns optical depth (tau) accumulated along ray
 */
static float trace_smoke_cloud(
    const SmokeCloud* cloud,
    const float* origin,
    const float* dir,
    float seg_len,
    float extinction
) {
    if (!cloud || !cloud->voxels || cloud->count == 0) return 0.0f;
    
    float tau = 0.0f;
    const float half = cloud->half_size;
    const float inv_half = 1.0f / (half * 2.0f);
    
    for (int i = 0; i < cloud->count; i++) {
        const SmokeVoxel* v = &cloud->voxels[i];
        
        /* Skip nearly transparent voxels */
        if (v->opacity < 0.02f) continue;
        
        /* Build AABB for this voxel */
        float box_min[3] = {
            v->pos[0] - half,
            v->pos[1] - half,
            v->pos[2] - half
        };
        float box_max[3] = {
            v->pos[0] + half,
            v->pos[1] + half,
            v->pos[2] + half
        };
        
        /* Test ray-AABB intersection */
        float t0, t1;
        if (!segment_aabb_intersect(origin, dir, seg_len, box_min, box_max, &t0, &t1)) {
            continue;
        }
        
        /* Calculate path length through voxel */
        float inside_len = (t1 - t0) * seg_len;
        if (inside_len <= 0.0f) continue;
        
        /* Accumulate optical depth using Beer-Lambert */
        float contrib = extinction * v->opacity * (inside_len * inv_half);
        tau += contrib;
        
        /* Early out if fully occluded */
        if (tau > 12.0f) {
            return 12.0f;
        }
    }
    
    return tau;
}

/*
 * Trace ray through all smoke clouds
 * Returns occlusion value: 0.0 = clear, 1.0 = fully blocked
 */
static float trace_all_smokes(
    const float* origin,
    const float* dir,
    float seg_len,
    float extinction
) {
    if (g_smoke_count == 0) return 0.0f;
    
    float total_tau = 0.0f;
    
    for (int i = 0; i < g_smoke_count; i++) {
        total_tau += trace_smoke_cloud(&g_smokes[i], origin, dir, seg_len, extinction);
        
        if (total_tau > 12.0f) {
            return 1.0f;
        }
    }
    
    /* Convert optical depth to transmittance: T = e^(-tau) */
    float transmittance = expf(-total_tau);
    
    /* Return occlusion (1 - transmittance) */
    return 1.0f - transmittance;
}

/* ========== Public API: World Geometry ========== */

EXPORT int rt_load(const char* filepath) {
    if (g_triangles) {
        free(g_triangles);
        g_triangles = NULL;
    }
    if (g_bvh_nodes) {
        free(g_bvh_nodes);
        g_bvh_nodes = NULL;
    }
    if (g_tri_indices) {
        free(g_tri_indices);
        g_tri_indices = NULL;
    }
    g_tri_count = 0;
    g_bvh_node_count = 0;
    
    FILE* f = fopen(filepath, "rb");
    if (!f) return -1;
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (size % 36 != 0) {
        fclose(f);
        return -1;
    }
    
    int count = size / 36;
    
    g_triangles = (Triangle*)malloc(sizeof(Triangle) * count);
    if (!g_triangles) {
        fclose(f);
        return -1;
    }
    
    size_t read = fread(g_triangles, sizeof(Triangle), count, f);
    fclose(f);
    
    if ((int)read != count) {
        free(g_triangles);
        g_triangles = NULL;
        return -1;
    }
    
    g_tri_count = count;
    bvh_build();
    
    return count;
}

EXPORT int rt_is_visible(
    float x1, float y1, float z1,
    float x2, float y2, float z2
) {
    if (!g_triangles || g_tri_count == 0) return 1;
    
    float origin[3] = {x1, y1, z1};
    float target[3] = {x2, y2, z2};
    
    float dir[3];
    dir[0] = target[0] - origin[0];
    dir[1] = target[1] - origin[1];
    dir[2] = target[2] - origin[2];
    
    float dist = sqrtf(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
    if (dist < EPSILON) return 1;
    
    dir[0] /= dist;
    dir[1] /= dist;
    dir[2] /= dist;
    
    if (g_bvh_nodes && g_bvh_node_count > 0) {
        return bvh_trace_ray(origin, dir, dist) ? 0 : 1;
    }
    
    for (int i = 0; i < g_tri_count; i++) {
        float t;
        if (ray_triangle_intersect(origin, dir, &g_triangles[i], &t)) {
            if (t > EPSILON && t < dist - EPSILON) {
                return 0;
            }
        }
    }
    
    return 1;
}

EXPORT void rt_unload(void) {
    if (g_triangles) {
        free(g_triangles);
        g_triangles = NULL;
    }
    if (g_bvh_nodes) {
        free(g_bvh_nodes);
        g_bvh_nodes = NULL;
    }
    if (g_tri_indices) {
        free(g_tri_indices);
        g_tri_indices = NULL;
    }
    g_tri_count = 0;
    g_bvh_node_count = 0;
    
    /* Also clear smokes */
    if (g_smokes) {
        for (int i = 0; i < g_smoke_capacity; i++) {
            if (g_smokes[i].voxels) {
                free(g_smokes[i].voxels);
            }
        }
        free(g_smokes);
        g_smokes = NULL;
    }
    g_smoke_count = 0;
    g_smoke_capacity = 0;
}

EXPORT int rt_get_triangle_count(void) {
    return g_tri_count;
}

EXPORT int rt_get_bvh_node_count(void) {
    return g_bvh_node_count;
}

/* ========== Public API: Smoke Voxels ========== */

/*
 * Clear all smoke voxel data (call at start of each frame)
 */
EXPORT void rt_clear_smokes(void) {
    for (int i = 0; i < g_smoke_count; i++) {
        g_smokes[i].count = 0;
    }
    g_smoke_count = 0;
}

/*
 * Add smoke voxels for a single smoke grenade
 * voxel_data: flat array of [x, y, z, opacity] * count
 * Returns: smoke cloud index, or -1 on error
 */
EXPORT int rt_add_smoke_voxels(
    float* voxel_data,
    int count,
    float half_size
) {
    if (!voxel_data || count <= 0) return -1;
    
    /* Initialize smoke array if needed */
    if (!g_smokes) {
        g_smoke_capacity = MAX_SMOKES;
        g_smokes = (SmokeCloud*)calloc(g_smoke_capacity, sizeof(SmokeCloud));
        if (!g_smokes) return -1;
    }
    
    if (g_smoke_count >= g_smoke_capacity) {
        return -1;  /* Max smokes reached */
    }
    
    SmokeCloud* cloud = &g_smokes[g_smoke_count];
    
    /* Allocate/reallocate voxel array */
    if (cloud->capacity < count) {
        if (cloud->voxels) free(cloud->voxels);
        cloud->voxels = (SmokeVoxel*)malloc(sizeof(SmokeVoxel) * count);
        if (!cloud->voxels) {
            cloud->capacity = 0;
            return -1;
        }
        cloud->capacity = count;
    }
    
    /* Copy voxel data */
    for (int i = 0; i < count; i++) {
        cloud->voxels[i].pos[0] = voxel_data[i * 4 + 0];
        cloud->voxels[i].pos[1] = voxel_data[i * 4 + 1];
        cloud->voxels[i].pos[2] = voxel_data[i * 4 + 2];
        cloud->voxels[i].opacity = voxel_data[i * 4 + 3];
    }
    
    cloud->count = count;
    cloud->half_size = half_size > 0 ? half_size : DEFAULT_VOXEL_HALF_SIZE;
    
    return g_smoke_count++;
}

/*
 * Get smoke occlusion along a ray
 * Returns: 0.0 = clear, 1.0 = fully blocked
 */
EXPORT float rt_get_smoke_occlusion(
    float x1, float y1, float z1,
    float x2, float y2, float z2,
    float extinction
) {
    if (g_smoke_count == 0) return 0.0f;
    
    float origin[3] = {x1, y1, z1};
    
    float dir[3];
    dir[0] = x2 - x1;
    dir[1] = y2 - y1;
    dir[2] = z2 - z1;
    
    float dist = sqrtf(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
    if (dist < EPSILON) return 0.0f;
    
    /* Normalize */
    dir[0] /= dist;
    dir[1] /= dist;
    dir[2] /= dist;
    
    if (extinction <= 0.0f) extinction = DEFAULT_EXTINCTION;
    
    return trace_all_smokes(origin, dir, dist, extinction);
}

/*
 * Full visibility check: world geometry + smoke voxels
 * Returns: 1 if visible, 0 if blocked
 */
EXPORT int rt_is_visible_with_smokes(
    float x1, float y1, float z1,
    float x2, float y2, float z2,
    float smoke_threshold,
    float extinction
) {
    /* Check world geometry first (fastest rejection) */
    if (!rt_is_visible(x1, y1, z1, x2, y2, z2)) {
        return 0;
    }
    
    /* Check smoke occlusion */
    if (g_smoke_count > 0) {
        if (smoke_threshold <= 0.0f) smoke_threshold = DEFAULT_SMOKE_THRESHOLD;
        if (extinction <= 0.0f) extinction = DEFAULT_EXTINCTION;
        
        float occlusion = rt_get_smoke_occlusion(x1, y1, z1, x2, y2, z2, extinction);
        if (occlusion >= smoke_threshold) {
            return 0;
        }
    }
    
    return 1;
}

/*
 * Get number of active smoke clouds
 */
EXPORT int rt_get_smoke_count(void) {
    return g_smoke_count;
}

/*
 * Get total voxel count across all smokes
 */
EXPORT int rt_get_total_voxel_count(void) {
    int total = 0;
    for (int i = 0; i < g_smoke_count; i++) {
        total += g_smokes[i].count;
    }
    return total;
}
