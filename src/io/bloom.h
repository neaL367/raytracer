#pragma once
// Fast HDR multi-scale bloom filter (downsample / upsample tent pyramid).
// Runs on linear HDR pre-tonemap buffers to simulate camera lens glare,
// emissive surface halos, and specular glow.
#include "../core/vec3.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace bloom {

inline double luminance(const vec3 &c) {
    return 0.2126 * c.x() + 0.7152 * c.y() + 0.0722 * c.z();
}

// Prefilter highlights with a soft knee threshold to prevent hard cutoff speckle
inline vec3 threshold_highlight(const vec3 &c, double threshold = 1.0, double knee = 0.5) {
    double l = luminance(c);
    double soft = l - threshold + knee;
    soft = std::clamp(soft, 0.0, 2.0 * knee);
    soft = (soft * soft) / (4.0 * knee + 1e-5);
    double contrib = std::max(soft, l - threshold);
    contrib /= std::max(l, 1e-4);
    if (contrib <= 0.0) return vec3(0, 0, 0);
    return c * contrib;
}

// 13-tap bilinear tent downsample (Karis / Jimenez style)
inline std::vector<vec3> downsample(const std::vector<vec3> &src, int src_w, int src_h, int dst_w, int dst_h) {
    std::vector<vec3> dst((size_t)dst_w * dst_h);
    double x_scale = (double)src_w / dst_w;
    double y_scale = (double)src_h / dst_h;

    for (int y = 0; y < dst_h; ++y) {
        double center_y = (y + 0.5) * y_scale - 0.5;
        for (int x = 0; x < dst_w; ++x) {
            double center_x = (x + 0.5) * x_scale - 0.5;

            // 13-tap sample weights: center 4x, diagonals 1x, axes 2x
            vec3 accum(0, 0, 0);
            double total_w = 0.0;

            static const struct { double dx, dy, w; } taps[] = {
                // Inner 4
                {-0.5, -0.5, 4.0}, { 0.5, -0.5, 4.0}, {-0.5,  0.5, 4.0}, { 0.5,  0.5, 4.0},
                // Diagonal outer 4
                {-1.0, -1.0, 1.0}, { 1.0, -1.0, 1.0}, {-1.0,  1.0, 1.0}, { 1.0,  1.0, 1.0},
                // Axis outer 4
                { 0.0, -1.0, 2.0}, { 0.0,  1.0, 2.0}, {-1.0,  0.0, 2.0}, { 1.0,  0.0, 2.0},
                // Center
                { 0.0,  0.0, 4.0}
            };

            for (const auto &tap : taps) {
                int sx = std::clamp((int)std::round(center_x + tap.dx * x_scale * 0.5), 0, src_w - 1);
                int sy = std::clamp((int)std::round(center_y + tap.dy * y_scale * 0.5), 0, src_h - 1);
                accum += src[(size_t)sy * src_w + sx] * tap.w;
                total_w += tap.w;
            }

            dst[(size_t)y * dst_w + x] = accum / total_w;
        }
    }
    return dst;
}

// 9-tap 3x3 tent upsample with radius scale
inline std::vector<vec3> upsample(const std::vector<vec3> &src, int src_w, int src_h,
                                  int dst_w, int dst_h, double filter_radius = 1.0) {
    std::vector<vec3> dst((size_t)dst_w * dst_h);
    double x_scale = (double)src_w / dst_w;
    double y_scale = (double)src_h / dst_h;

    for (int y = 0; y < dst_h; ++y) {
        double center_y = (y + 0.5) * y_scale - 0.5;
        for (int x = 0; x < dst_w; ++x) {
            double center_x = (x + 0.5) * x_scale - 0.5;

            vec3 accum(0, 0, 0);
            double total_w = 0.0;

            for (int dy = -1; dy <= 1; ++dy) {
                double wy = (dy == 0) ? 2.0 : 1.0;
                int sy = std::clamp((int)std::round(center_y + dy * filter_radius), 0, src_h - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    double wx = (dx == 0) ? 2.0 : 1.0;
                    int sx = std::clamp((int)std::round(center_x + dx * filter_radius), 0, src_w - 1);
                    double w = wx * wy;
                    accum += src[(size_t)sy * src_w + sx] * w;
                    total_w += w;
                }
            }

            dst[(size_t)y * dst_w + x] = accum / total_w;
        }
    }
    return dst;
}

// Apply multi-scale pyramid bloom to linear HDR image
inline std::vector<vec3> apply_bloom(const std::vector<vec3> &hdr, int W, int H,
                                     double threshold = 1.0, double intensity = 0.08) {
    if (W < 4 || H < 4 || intensity <= 0.001)
        return hdr;

    // 1. Prefilter threshold
    std::vector<vec3> pref(hdr.size());
    for (size_t i = 0; i < hdr.size(); ++i) {
        pref[i] = threshold_highlight(hdr[i], threshold, 0.5);
    }

    // 2. Downsample pyramid (Level 1: 1/2, Level 2: 1/4)
    int w1 = std::max(1, W / 2), h1 = std::max(1, H / 2);
    int w2 = std::max(1, W / 4), h2 = std::max(1, H / 4);

    std::vector<vec3> down1 = downsample(pref, W, H, w1, h1);
    std::vector<vec3> down2 = downsample(down1, w1, h1, w2, h2);

    // 3. Upsample pyramid
    std::vector<vec3> up1 = upsample(down2, w2, h2, w1, h1, 1.0);
    for (size_t i = 0; i < up1.size(); ++i)
        up1[i] = down1[i] * 0.5 + up1[i] * 0.5;

    std::vector<vec3> bloom_out = upsample(up1, w1, h1, W, H, 1.0);

    // 4. Composite bloom onto HDR buffer
    std::vector<vec3> result(hdr.size());
    for (size_t i = 0; i < hdr.size(); ++i) {
        result[i] = hdr[i] + bloom_out[i] * intensity;
    }

    return result;
}

} // namespace bloom
