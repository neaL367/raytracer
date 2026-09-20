#pragma once
// Host-side scene: std430 mirrors of common.glsl structs + builder.
// Pure data, zero Vulkan: the seam is bytes in, bytes out.
#include <cstring>
#include <vector>

struct GPUSphere {
    float c[4], alb[4], emit[4], prm[4]; // c.w=radius; prm=(type,fuzz,ir,0)
};
struct GPUQuad {
    float Q[4], u[4], v[4], alb[4], emit[4], prm[4];
};
struct GPUTri {
    float a[4], b[4], c[4], alb[4], emit[4], prm[4];
};
struct GPUCam {
    float o[4], ll[4], h[4], v[4];
};

// mat types: 0 lambertian, 1 metal, 2 dielectric, 3 emissive.
// Mirrors CPU main scene numbers exactly (parity depends on it).
struct gpu_scene {
    GPUCam cam;
    std::vector<GPUSphere> spheres;
    std::vector<GPUQuad> quads; // emissive first (NEE indexes leading quads)
    std::vector<GPUTri> tris;
};

inline gpu_scene build_gpu_scene(int W, int H) {
    gpu_scene s;
    const float aspect = (float)((double)W / (double)H);
    GPUCam cam = {{0, 0, 0, 0}, {-aspect, -1, -1, 0}, {2 * aspect, 0, 0, 0}, {0, 2, 0, 0}};
    s.cam = cam;

    auto sph = [](float x, float y, float z, float r, float ar, float ag, float ab,
                  float er, float eg, float eb, float type, float fuzz, float ir) {
        GPUSphere s{};
        s.c[0] = x;
        s.c[1] = y;
        s.c[2] = z;
        s.c[3] = r;
        s.alb[0] = ar;
        s.alb[1] = ag;
        s.alb[2] = ab;
        s.emit[0] = er;
        s.emit[1] = eg;
        s.emit[2] = eb;
        s.prm[0] = type;
        s.prm[1] = fuzz;
        s.prm[2] = ir;
        return s;
    };
    s.spheres = {
        sph(0, -100.5f, -1, 100, 0.5f, 0.5f, 0.5f, 0, 0, 0, 0, 0, 0),
        sph(0, 0, -1, 0.5f, 0.7f, 0.3f, 0.3f, 0, 0, 0, 0, 0, 0),
        sph(-1, 0, -1, 0.5f, 0.8f, 0.8f, 0.8f, 0, 0, 0, 1, 0.3f, 0),
        sph(1, 0, -1, 0.5f, 0, 0, 0, 0, 0, 0, 2, 0, 1.5f),
    };
    GPUQuad light{};
    {
        float Q[4] = {-1, 1.9f, -2, 0}, u[4] = {2, 0, 0, 0}, v[4] = {0, 0, 2, 0};
        std::memcpy(light.Q, Q, sizeof Q);
        std::memcpy(light.u, u, sizeof u);
        std::memcpy(light.v, v, sizeof v);
        light.emit[0] = light.emit[1] = light.emit[2] = 4.0f;
        light.prm[0] = 3;
    }
    s.quads = {light};
    GPUTri tri{};
    {
        float a[4] = {-0.3f, -0.35f, -0.6f, 0}, b[4] = {0.3f, -0.35f, -0.6f, 0},
              c[4] = {0, 0.1f, -0.6f, 0}, alb[4] = {0.7f, 0.3f, 0.3f, 0};
        std::memcpy(tri.a, a, sizeof a);
        std::memcpy(tri.b, b, sizeof b);
        std::memcpy(tri.c, c, sizeof c);
        std::memcpy(tri.alb, alb, sizeof alb);
    }
    s.tris = {tri};
    return s;
}
