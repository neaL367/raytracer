#pragma once
// HDR bilateral denoiser: spatial gaussian x luminance-range gaussian.
// Runs on linear HDR pre-tonemap (range weight tracks scene ratios,
// no halo on the bright quad). Radius 2, single-thread post pass:
// 90k px is trivial vs render time. No external deps (OIDN ruled out).
#include "../core/vec3.h"

#include <cmath>
#include <vector>

inline double denoise_luminance(const vec3 &c) {
    return 0.2126 * c.x() + 0.7152 * c.y() + 0.0722 * c.z();
}

inline std::vector<vec3> bilateral_denoise(const std::vector<vec3> &fb, int W, int H,
                                           double sigma_s = 1.5, double sigma_r = 0.15,
                                           int iters = 1) {
    const int R = 2;
    std::vector<vec3> src = fb, dst(fb.size());
    // Spatial kernel precomputed (range part is per-pixel).
    double spat[2 * R + 1][2 * R + 1];
    for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx)
            spat[dy + R][dx + R] = std::exp(-(dx * dx + dy * dy) / (2 * sigma_s * sigma_s));
    for (int it = 0; it < iters; ++it) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const vec3 &c = src[(size_t)y * W + x];
                double lc = denoise_luminance(c);
                // Range width tracks local level (+floor for darks).
                double sr = sigma_r * (lc + 0.1);
                double norm = 2 * sr * sr;
                vec3 acc(0, 0, 0);
                double wsum = 0;
                for (int dy = -R; dy <= R; ++dy) {
                    int ny = y + dy;
                    if (ny < 0 || ny >= H)
                        continue;
                    for (int dx = -R; dx <= R; ++dx) {
                        int nx = x + dx;
                        if (nx < 0 || nx >= W)
                            continue;
                        const vec3 &n = src[(size_t)ny * W + nx];
                        double dl = denoise_luminance(n) - lc;
                        double w = spat[dy + R][dx + R] * std::exp(-(dl * dl) / norm);
                        acc += w * n;
                        wsum += w;
                    }
                }
                dst[(size_t)y * W + x] = acc / wsum;
            }
        }
        src = dst;
    }
    return dst;
}
