#pragma once
// Host-side GPU scene: std430 mirrors of common.glsl structs + camera.
// Prims come from flatten_scene (scene_data -> typed arrays + BVH);
// only the camera math lives here (mirrors scene/scene.h framing).
#include <cmath>
#include <string>
#include <vector>

#include "../core/vec3.h"

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
    float a1[4], b1[4], c1[4], tm[4]; // motion endpoints + range (moving when tm.y > tm.x)
};
struct GPUCam {
    float o[4], ll[4], h[4], v[4];
    float lens[4]; // x = thin-lens radius (aperture/2, 0 = pinhole)
};

// std430 and UBO layout guarantees between C++ host and GLSL compute kernels
static_assert(sizeof(GPUSphere) == 112, "GPUSphere must be 112 bytes to match GLSL std430 layout");
static_assert(sizeof(GPUQuad)   == 112, "GPUQuad must be 112 bytes to match GLSL std430 layout");
static_assert(sizeof(GPUTri)    == 256, "GPUTri must be 256 bytes to match GLSL std430 layout");
static_assert(sizeof(GPUCam)    == 80,  "GPUCam must be 80 bytes to match GLSL UBO layout");

struct gpu_scene {    GPUCam cam;
    std::vector<GPUSphere> spheres;
    std::vector<GPUQuad> quads; // emissive first (NEE indexes leading quads)
    std::vector<GPUTri> tris;
};

// Default pinhole (vfov 90, focus 1): matches CPU default camera.
inline GPUCam gpu_default_cam(int W, int H, double aperture = 0.0) {
    const float aspect = (float)((double)W / (double)H);
    GPUCam cam = {};
    cam.o[0] = 0;
    cam.o[1] = 0;
    cam.o[2] = 0;
    cam.ll[0] = -aspect;
    cam.ll[1] = -1;
    cam.ll[2] = -1;
    cam.h[0] = 2 * aspect;
    cam.v[1] = 2;
    cam.lens[0] = (float)(aperture * 0.5);
    return cam;
}


// Generic lookat camera: mirrors CPU camera math in fp32 (viewport scaled
// by focus_dist once, no second multiply; see camera.h comment).
inline GPUCam gpu_lookat_cam(float fx, float fy, float fz, float tx, float ty, float tz,
                             float vfov_deg, int W, int H, double aperture,
                             double focus_dist) {
    const float aspect = (float)((double)W / (double)H);
    const float th = vfov_deg * 3.14159265f / 180.0f;
    const float hh = 2.0f * tanf(th * 0.5f) * (float)focus_dist;
    const float ww = hh * aspect;
    // w = unit(from-at), u = unit(up x w), v = w x u.
    float wx = fx - tx, wy = fy - ty, wz = fz - tz;
    float wl = sqrtf(wx * wx + wy * wy + wz * wz);
    wx /= wl;
    wy /= wl;
    wz /= wl;
    float ux = wz, uy = 0.0f, uz = -wx; // cross((0,1,0), w), vup is +y
    float ul = sqrtf(ux * ux + uy * uy + uz * uz);
    ux /= ul;
    uy /= ul;
    uz /= ul;
    float vx = wy * uz - wz * uy, vy = wz * ux - wx * uz, vz = wx * uy - wy * ux;
    GPUCam cam = {};
    cam.o[0] = fx;
    cam.o[1] = fy;
    cam.o[2] = fz;
    cam.ll[0] = fx - 0.5f * ww * ux - 0.5f * hh * vx - (float)focus_dist * wx;
    cam.ll[1] = fy - 0.5f * ww * uy - 0.5f * hh * vy - (float)focus_dist * wy;
    cam.ll[2] = fz - 0.5f * ww * uz - 0.5f * hh * vz - (float)focus_dist * wz;
    cam.h[0] = ww * ux;
    cam.h[1] = ww * uy;
    cam.h[2] = ww * uz;
    cam.v[0] = hh * vx;
    cam.v[1] = hh * vy;
    cam.v[2] = hh * vz;
    cam.lens[0] = (float)(aperture * 0.5);
    return cam;
}

inline GPUCam build_gpu_camera(int W, int H, const std::string &name,
                               double aperture = 0.0) {
    (void)name;
    return gpu_default_cam(W, H, aperture);
}

// Mirror of a CPU camera (origin/lower_left/horiz/vert/lens): any scene or
// tune matches automatically, no per-scene vfov to keep in sync (M53: the
// hardcoded default cam stayed vfov-90 after the tune moved CPU to 75).
// Needs camera/camera.h (included by the caller, not here).
inline GPUCam gpu_cam_from_cpu(const vec3 &origin, const vec3 &lower_left,
                               const vec3 &horiz, const vec3 &vert, double lens_radius,
                               int blades = 0, double anamorphic = 1.0, double distortion = 0.0) {
    GPUCam cam = {};
    cam.o[0] = (float)origin.x();
    cam.o[1] = (float)origin.y();
    cam.o[2] = (float)origin.z();
    cam.ll[0] = (float)lower_left.x();
    cam.ll[1] = (float)lower_left.y();
    cam.ll[2] = (float)lower_left.z();
    cam.h[0] = (float)horiz.x();
    cam.h[1] = (float)horiz.y();
    cam.h[2] = (float)horiz.z();
    cam.v[0] = (float)vert.x();
    cam.v[1] = (float)vert.y();
    cam.v[2] = (float)vert.z();
    cam.lens[0] = (float)lens_radius;
    cam.lens[1] = (float)blades;
    cam.lens[2] = (float)anamorphic;
    cam.lens[3] = (float)distortion;
    return cam;
}
