/*
 * omega-raytrace - Fast raytracing library for Omega
 * Compile: Windows DLL / Linux SO
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

#define EPSILON 0.0000001f
#define MAX_DEPTH 24
#define MAX_LEAF_TRIS 8

/* -------------------------------------------------------------------------- */
/* Types                                                                      */
/* -------------------------------------------------------------------------- */

typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    Vec3 p1, p2, p3;
} Triangle;

typedef struct {
    Vec3 min, max;
} AABB;

typedef struct KDNode {
    AABB bbox;
    struct KDNode* left;
    struct KDNode* right;
    Triangle* triangles;
    int tri_count;
} KDNode;

/* -------------------------------------------------------------------------- */
/* Globals                                                                    */
/* -------------------------------------------------------------------------- */

static Triangle* g_triangles = NULL;
static int g_tri_count = 0;
static KDNode* g_tree = NULL;

/* -------------------------------------------------------------------------- */
/* Vector Math                                                                */
/* -------------------------------------------------------------------------- */

static inline Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return (Vec3){ a.x - b.x, a.y - b.y, a.z - b.z };
}

static inline float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return (Vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static inline float vec3_length(Vec3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

static inline Vec3 vec3_normalize(Vec3 v) {
    float len = vec3_length(v);
    if (len < EPSILON) return (Vec3){0, 0, 0};
    return (Vec3){ v.x / len, v.y / len, v.z / len };
}

static inline float minf(float a, float b) { return a < b ? a : b; }
static inline float maxf(float a, float b) { return a > b ? a : b; }

/* -------------------------------------------------------------------------- */
/* Ray-Triangle Intersection (Möller-Trumbore)                                */
/* -------------------------------------------------------------------------- */

static int ray_hits_triangle(Vec3 origin, Vec3 target, Triangle* tri) {
    Vec3 edge1 = vec3_sub(tri->p2, tri->p1);
    Vec3 edge2 = vec3_sub(tri->p3, tri->p1);
    Vec3 ray_dir = vec3_sub(target, origin);
    
    Vec3 h = vec3_cross(ray_dir, edge2);
    float a = vec3_dot(edge1, h);
    
    if (a > -EPSILON && a < EPSILON) return 0;
    
    float f = 1.0f / a;
    Vec3 s = vec3_sub(origin, tri->p1);
    float u = f * vec3_dot(s, h);
    
    if (u < 0.0f || u > 1.0f) return 0;
    
    Vec3 q = vec3_cross(s, edge1);
    float v = f * vec3_dot(ray_dir, q);
    
    if (v < 0.0f || u + v > 1.0f) return 0;
    
    float t = f * vec3_dot(edge2, q);
    return (t > EPSILON && t < 1.0f);
}

/* -------------------------------------------------------------------------- */
/* Ray-AABB Intersection (Slabs)                                              */
/* -------------------------------------------------------------------------- */

static int ray_hits_aabb(Vec3 origin, Vec3 target, AABB* bbox) {
    Vec3 dir = vec3_normalize(vec3_sub(target, origin));
    
    float inv_x = fabsf(dir.x) < EPSILON ? 1e30f : 1.0f / dir.x;
    float inv_y = fabsf(dir.y) < EPSILON ? 1e30f : 1.0f / dir.y;
    float inv_z = fabsf(dir.z) < EPSILON ? 1e30f : 1.0f / dir.z;
    
    float t1 = (bbox->min.x - origin.x) * inv_x;
    float t2 = (bbox->max.x - origin.x) * inv_x;
    float t3 = (bbox->min.y - origin.y) * inv_y;
    float t4 = (bbox->max.y - origin.y) * inv_y;
    float t5 = (bbox->min.z - origin.z) * inv_z;
    float t6 = (bbox->max.z - origin.z) * inv_z;
    
    float tmin = maxf(maxf(minf(t1, t2), minf(t3, t4)), minf(t5, t6));
    float tmax = minf(minf(maxf(t1, t2), maxf(t3, t4)), maxf(t5, t6));
    
    return (tmax >= 0 && tmin <= tmax);
}

/* -------------------------------------------------------------------------- */
/* KD-Tree                                                                    */
/* -------------------------------------------------------------------------- */

static AABB calc_bbox(Triangle* tris, int count) {
    AABB bbox = {
        .min = { 1e30f, 1e30f, 1e30f },
        .max = { -1e30f, -1e30f, -1e30f }
    };
    
    for (int i = 0; i < count; i++) {
        Vec3* pts[3] = { &tris[i].p1, &tris[i].p2, &tris[i].p3 };
        for (int j = 0; j < 3; j++) {
            bbox.min.x = minf(bbox.min.x, pts[j]->x);
            bbox.min.y = minf(bbox.min.y, pts[j]->y);
            bbox.min.z = minf(bbox.min.z, pts[j]->z);
            bbox.max.x = maxf(bbox.max.x, pts[j]->x);
            bbox.max.y = maxf(bbox.max.y, pts[j]->y);
            bbox.max.z = maxf(bbox.max.z, pts[j]->z);
        }
    }
    return bbox;
}

static float get_centroid(Triangle* tri, int axis) {
    if (axis == 0) return (tri->p1.x + tri->p2.x + tri->p3.x) / 3.0f;
    if (axis == 1) return (tri->p1.y + tri->p2.y + tri->p3.y) / 3.0f;
    return (tri->p1.z + tri->p2.z + tri->p3.z) / 3.0f;
}

static int compare_tris_x(const void* a, const void* b) {
    float ca = get_centroid((Triangle*)a, 0);
    float cb = get_centroid((Triangle*)b, 0);
    return (ca > cb) - (ca < cb);
}

static int compare_tris_y(const void* a, const void* b) {
    float ca = get_centroid((Triangle*)a, 1);
    float cb = get_centroid((Triangle*)b, 1);
    return (ca > cb) - (ca < cb);
}

static int compare_tris_z(const void* a, const void* b) {
    float ca = get_centroid((Triangle*)a, 2);
    float cb = get_centroid((Triangle*)b, 2);
    return (ca > cb) - (ca < cb);
}

static KDNode* build_tree(Triangle* tris, int count, int depth) {
    if (count <= 0) return NULL;
    
    KDNode* node = (KDNode*)calloc(1, sizeof(KDNode));
    node->bbox = calc_bbox(tris, count);
    
    if (count <= MAX_LEAF_TRIS || depth >= MAX_DEPTH) {
        node->triangles = (Triangle*)malloc(count * sizeof(Triangle));
        memcpy(node->triangles, tris, count * sizeof(Triangle));
        node->tri_count = count;
        return node;
    }
    
    int axis = depth % 3;
    if (axis == 0) qsort(tris, count, sizeof(Triangle), compare_tris_x);
    else if (axis == 1) qsort(tris, count, sizeof(Triangle), compare_tris_y);
    else qsort(tris, count, sizeof(Triangle), compare_tris_z);
    
    int mid = count / 2;
    node->left = build_tree(tris, mid, depth + 1);
    node->right = build_tree(tris + mid, count - mid, depth + 1);
    
    return node;
}

static void free_tree(KDNode* node) {
    if (!node) return;
    free_tree(node->left);
    free_tree(node->right);
    free(node->triangles);
    free(node);
}

static int trace_tree(KDNode* node, Vec3 origin, Vec3 target) {
    if (!node) return 0;
    if (!ray_hits_aabb(origin, target, &node->bbox)) return 0;
    
    if (node->triangles) {
        for (int i = 0; i < node->tri_count; i++) {
            if (ray_hits_triangle(origin, target, &node->triangles[i])) {
                return 1;
            }
        }
        return 0;
    }
    
    return trace_tree(node->left, origin, target) || 
           trace_tree(node->right, origin, target);
}

/* -------------------------------------------------------------------------- */
/* File Loading                                                               */
/* -------------------------------------------------------------------------- */

static int is_hex_format(const char* data, size_t len) {
    if (len < 50) return 0;
    int hex = 0, spaces = 0;
    for (int i = 0; i < 50 && i < (int)len; i++) {
        char c = data[i];
        if (c == ' ') spaces++;
        else if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) hex++;
        else return 0;
    }
    return (spaces > 5 && hex > spaces);
}

static unsigned char* hex_to_binary(const char* hex_data, size_t hex_len, size_t* out_len) {
    size_t max_bytes = hex_len / 2;
    unsigned char* binary = (unsigned char*)malloc(max_bytes);
    size_t bi = 0;
    
    for (size_t i = 0; i < hex_len - 1; i++) {
        char c1 = hex_data[i];
        char c2 = hex_data[i + 1];
        
        int is_hex1 = (c1 >= '0' && c1 <= '9') || (c1 >= 'a' && c1 <= 'f') || (c1 >= 'A' && c1 <= 'F');
        int is_hex2 = (c2 >= '0' && c2 <= '9') || (c2 >= 'a' && c2 <= 'f') || (c2 >= 'A' && c2 <= 'F');
        
        if (is_hex1 && is_hex2) {
            unsigned int byte;
            char temp[3] = { c1, c2, 0 };
            sscanf(temp, "%x", &byte);
            binary[bi++] = (unsigned char)byte;
            i++;
        }
    }
    
    *out_len = bi;
    return binary;
}

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

EXPORT int rt_load_file(const char* path) {
    /* Clean up previous data */
    if (g_tree) {
        free_tree(g_tree);
        g_tree = NULL;
    }
    if (g_triangles) {
        free(g_triangles);
        g_triangles = NULL;
    }
    g_tri_count = 0;
    
    /* Read file */
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char* data = (char*)malloc(file_size);
    if (!data) {
        fclose(f);
        return -2;
    }
    
    fread(data, 1, file_size, f);
    fclose(f);
    
    /* Convert hex if needed */
    unsigned char* binary_data = (unsigned char*)data;
    size_t binary_len = file_size;
    int was_hex = 0;
    
    if (is_hex_format(data, file_size)) {
        binary_data = hex_to_binary(data, file_size, &binary_len);
        free(data);
        data = NULL;
        was_hex = 1;
    }
    
    /* Parse triangles */
    size_t tri_size = sizeof(Triangle);
    int num_tris = (int)(binary_len / tri_size);
    
    if (num_tris == 0) {
        if (was_hex) free(binary_data);
        else free(data);
        return -3;
    }
    
    g_triangles = (Triangle*)malloc(num_tris * tri_size);
    memcpy(g_triangles, binary_data, num_tris * tri_size);
    g_tri_count = num_tris;
    
    if (was_hex) free(binary_data);
    else free(data);
    
    /* Validate */
    if (g_triangles[0].p1.x != g_triangles[0].p1.x || 
        fabsf(g_triangles[0].p1.x) > 50000.0f) {
        free(g_triangles);
        g_triangles = NULL;
        g_tri_count = 0;
        return -4;
    }
    
    /* Build tree */
    g_tree = build_tree(g_triangles, g_tri_count, 0);
    
    return g_tri_count;
}

EXPORT int rt_is_visible(float start_x, float start_y, float start_z,
                         float end_x, float end_y, float end_z) {
    if (!g_tree) return 1;  /* No map = visible */
    
    Vec3 origin = { start_x, start_y, start_z };
    Vec3 target = { end_x, end_y, end_z };
    
    /* Returns 1 if visible (no hit), 0 if blocked */
    return !trace_tree(g_tree, origin, target);
}

EXPORT void rt_unload(void) {
    if (g_tree) {
        free_tree(g_tree);
        g_tree = NULL;
    }
    if (g_triangles) {
        free(g_triangles);
        g_triangles = NULL;
    }
    g_tri_count = 0;
}

EXPORT int rt_get_triangle_count(void) {
    return g_tri_count;
}

EXPORT int rt_is_loaded(void) {
    return g_tree != NULL;
}
