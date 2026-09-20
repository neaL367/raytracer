#pragma once
// Host-side GPU scene: std430 mirrors of common.glsl structs + camera.
// Prims come from flatten_scene (scene_data -> typed arrays + BVH);
// only the camera math lives here (mirrors scene/scene.h framing).
#include <cmath>
#include <string>
#include <vector>

struct GPUSphere {
    float c[4], c1[4], tm[4], alb[4], alb2[4], emit[4], prm[4]; // c.w=radius
    // prm=(type,rough,ir,motionflag); tm=(t0,t1,0,0); type 6 = fog volume
    // (boundary in c, density in prm.y, albedo in alb)
};
struct GPUQuad {
    float Q[4], u[4], v[4], alb[4], alb2[4], emit[4], prm[4];
};
struct GPUTri {
    float a[4], b[4], c[4], n0[4], n1[4], n2[4], tuvA[4], tuvB[4], alb[4], alb2[4], emit[4], prm[4];
};
struct GPUCam {
    float o[4], ll[4], h[4], v[4];
};

struct gpu_scene {
    GPUCam cam;
    std::vector<GPUSphere> spheres;
    std::vector<GPUQuad> quads; // emissive first (NEE indexes leading quads)
    std::vector<GPUTri> tris;
};

// Default pinhole (vfov 90, focus 1): matches CPU default camera.
inline GPUCam gpu_default_cam(int W, int H) {
    const float aspect = (float)((double)W / (double)H);
    GPUCam cam = {{0, 0, 0, 0}, {-aspect, -1, -1, 0}, {2 * aspect, 0, 0, 0}, {0, 2, 0, 0}};
    return cam;
}

// Cornell (278,278,-800) vfov 40: mirrors CPU cornell camera exactly.
inline GPUCam gpu_cornell_cam(int W, int H) {
    const float aspect = (float)((double)W / (double)H);
    const float th = 40.0f * 3.14159265f / 180.0f;
    const float hh = 2.0f * tanf(th * 0.5f) * 800.0f;
    const float ww = hh * aspect;
    GPUCam cam = {{278, 278, -800, 0},
                  {278 + ww * 0.5f, 278 - hh * 0.5f, 0, 0},
                  {-ww, 0, 0, 0},
                  {0, hh, 0, 0}};
    return cam;
}

inline GPUCam build_gpu_camera(int W, int H, const std::string &name) {
    if (name == "cornell")
        return gpu_cornell_cam(W, H);
    return gpu_default_cam(W, H);
}
