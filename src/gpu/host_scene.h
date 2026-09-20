#pragma once
// Host-side scene: std430 mirrors of common.glsl structs + builder.
// Pure data, zero Vulkan: the seam is bytes in, bytes out.
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

struct GPUSphere {
    float c[4], alb[4], alb2[4], emit[4], prm[4]; // c.w=radius; prm=(type,fuzz,ir,0)
};
struct GPUQuad {
    float Q[4], u[4], v[4], alb[4], alb2[4], emit[4], prm[4];
};
struct GPUTri {
    float a[4], b[4], c[4], alb[4], alb2[4], emit[4], prm[4];
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

inline gpu_scene build_gpu_cornell(int W, int H);
inline gpu_scene build_gpu_default(int W, int H);

inline gpu_scene build_gpu_scene(int W, int H, const std::string &name = "default") {
    if (name == "cornell")
        return build_gpu_cornell(W, H);
    return build_gpu_default(W, H);
}
// Cornell numbers mirror scene/scene.h (CPU) exactly: walls, boxes,
// ceiling light first (NEE indexes leading emissive quads).
inline gpu_scene build_gpu_cornell(int W, int H) {
    gpu_scene s;
    const float aspect = (float)((double)W / (double)H);
    // Camera (278,278,-800) vfov 40: u=(-1,0,0), v=(0,1,0), w=(0,0,-1).
    const float th = 40.0f * 3.14159265f / 180.0f;
    const float hh = 2.0f * tanf(th * 0.5f) * 800.0f;
    const float ww = hh * aspect;
    GPUCam cam = {{278, 278, -800, 0},
                  {278 + ww * 0.5f, 278 - hh * 0.5f, 0, 0},
                  {-ww, 0, 0, 0},
                  {0, hh, 0, 0}};
    s.cam = cam;
    auto quad = [](float qx, float qy, float qz, float ux, float uy, float uz, float vx,
                   float vy, float vz, float ar, float ag, float ab, float er, float eg,
                   float eb, float type) {
        GPUQuad q{};
        q.Q[0] = qx;
        q.Q[1] = qy;
        q.Q[2] = qz;
        q.u[0] = ux;
        q.u[1] = uy;
        q.u[2] = uz;
        q.v[0] = vx;
        q.v[1] = vy;
        q.v[2] = vz;
        q.alb[0] = ar;
        q.alb[1] = ag;
        q.alb[2] = ab;
        q.emit[0] = er;
        q.emit[1] = eg;
        q.emit[2] = eb;
        q.prm[0] = type;
        return q;
    };
    // Light FIRST (NEE), then shell + boxes. Red x=555 (image-left).
    s.quads.push_back(quad(213, 554, 227, 130, 0, 0, 0, 0, 105, 0, 0, 0, 7, 7, 7, 3));
    const float R[3] = {0.63f, 0.065f, 0.05f}, G[3] = {0.14f, 0.45f, 0.15f},
                Wt[3] = {0.725f, 0.71f, 0.68f};
    s.quads.push_back(quad(0, 0, 0, 555, 0, 0, 0, 0, 555, Wt[0], Wt[1], Wt[2], 0, 0, 0, 0));
    s.quads.push_back(quad(0, 555, 0, 555, 0, 0, 0, 0, 555, Wt[0], Wt[1], Wt[2], 0, 0, 0, 0));
    s.quads.push_back(quad(0, 0, 555, 555, 0, 0, 0, 555, 0, Wt[0], Wt[1], Wt[2], 0, 0, 0, 0));
    s.quads.push_back(
        quad(555, 0, 0, 0, 0, 555, 0, 555, 0, R[0], R[1], R[2], 0, 0, 0, 0));
    s.quads.push_back(
        quad(0, 0, 0, 0, 0, 555, 0, 555, 0, G[0], G[1], G[2], 0, 0, 0, 0));
    auto box = [&](float x0, float y0, float z0, float x1, float y1, float z1) {
        float dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
        s.quads.push_back(quad(x0, y0, z0, dx, 0, 0, 0, 0, dz, Wt[0], Wt[1], Wt[2], 0, 0, 0, 0));
        s.quads.push_back(quad(x0, y1, z0, dx, 0, 0, 0, 0, dz, Wt[0], Wt[1], Wt[2], 0, 0, 0, 0));
        s.quads.push_back(quad(x0, y0, z0, 0, 0, dz, 0, dy, 0, Wt[0], Wt[1], Wt[2], 0, 0, 0, 0));
        s.quads.push_back(quad(x1, y0, z0, 0, 0, dz, 0, dy, 0, Wt[0], Wt[1], Wt[2], 0, 0, 0, 0));
        s.quads.push_back(quad(x0, y0, z1, dx, 0, 0, 0, dy, 0, Wt[0], Wt[1], Wt[2], 0, 0, 0, 0));
        s.quads.push_back(quad(x0, y0, z0, dx, 0, 0, 0, dy, 0, Wt[0], Wt[1], Wt[2], 0, 0, 0, 0));
    };
    box(265, 0, 295, 430, 330, 460);
    box(130, 0, 65, 295, 165, 230);
    return s;
}

inline gpu_scene build_gpu_default(int W, int H) {
    gpu_scene s;
    const float aspect = (float)((double)W / (double)H);
    GPUCam cam = {{0, 0, 0, 0}, {-aspect, -1, -1, 0}, {2 * aspect, 0, 0, 0}, {0, 2, 0, 0}};
    s.cam = cam;

    auto sph = [](float x, float y, float z, float r, float ar, float ag, float ab,
                   float br, float bg, float bb, float er, float eg, float eb, float type,
                   float fuzz, float ir) {
        GPUSphere s{};
        s.c[0] = x;
        s.c[1] = y;
        s.c[2] = z;
        s.c[3] = r;
        s.alb[0] = ar;
        s.alb[1] = ag;
        s.alb[2] = ab;
        s.alb2[0] = br;
        s.alb2[1] = bg;
        s.alb2[2] = bb;
        s.emit[0] = er;
        s.emit[1] = eg;
        s.emit[2] = eb;
        s.prm[0] = type;
        s.prm[1] = fuzz;
        s.prm[2] = ir;
        return s;
    };
    // type 4 = checker diffuse (alb even, alb2 odd, prm.y scale).
    s.spheres = {
        sph(0, -100.5f, -1, 100, 0.8f, 0.8f, 0.8f, 0.3f, 0.3f, 0.3f, 0, 0, 0, 4, 4.0f, 0),
        sph(-1, 0, -1, 0.5f, 0.8f, 0.8f, 0.8f, 0, 0, 0, 0, 0, 0, 1, 0.3f, 0),
        sph(1, 0, -1, 0.5f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1.5f),
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
    // Cube mesh baked at scene position (matches assets/cube.obj):
    // center (0,-0.15,-1), size 0.7. Checker red like CPU.
    {
        const float V[8][3] = {{-0.35f, -0.5f, -1.35f}, {0.35f, -0.5f, -1.35f},
                               {0.35f, -0.5f, -0.65f}, {-0.35f, -0.5f, -0.65f},
                               {-0.35f, 0.2f, -1.35f}, {0.35f, 0.2f, -1.35f},
                               {0.35f, 0.2f, -0.65f}, {-0.35f, 0.2f, -0.65f}};
        const int F[6][4] = {{0, 1, 2, 3}, {4, 7, 6, 5}, {0, 4, 5, 1},
                             {1, 5, 6, 2}, {2, 6, 7, 3}, {4, 0, 3, 7}};
        for (auto &f : F)
            for (int k = 0; k < 2; ++k) {
                int ia = f[0], ib = f[1 + k], ic = f[2 + k];
                GPUTri tri{};
                tri.a[0] = V[ia][0];
                tri.a[1] = V[ia][1];
                tri.a[2] = V[ia][2];
                tri.b[0] = V[ib][0];
                tri.b[1] = V[ib][1];
                tri.b[2] = V[ib][2];
                tri.c[0] = V[ic][0];
                tri.c[1] = V[ic][1];
                tri.c[2] = V[ic][2];
                tri.alb[0] = 0.7f;
                tri.alb[1] = 0.3f;
                tri.alb[2] = 0.3f;
                tri.alb2[0] = tri.alb2[1] = tri.alb2[2] = 0.9f;
                tri.prm[0] = 4;
                tri.prm[1] = 3.0f;
                s.tris.push_back(tri);
            }
    }
    return s;
}
