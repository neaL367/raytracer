#pragma once
// Pure image-compare math for the viewer + --stats CLI.
// u8 RGB top-first rows (ppm_io order); viewer stays thin, tests own this.
#include <cmath>
#include <cstdint>
#include <vector>

struct diff_stats {
    double mean_abs = 0; // mean |a-b| over all bytes (0..255)
    double max_abs = 0;  // worst byte
    double frac_over = 0; // fraction of bytes above threshold
    int w = 0, h = 0;
};

inline diff_stats compare_images(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b,
                                 int w, int h, int thresh = 8) {
    diff_stats s;
    s.w = w;
    s.h = h;
    if (a.size() != b.size() || a.empty())
        return s;
    double sum = 0;
    size_t over = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        sum += d;
        if (d > s.max_abs)
            s.max_abs = d;
        if (d > thresh)
            over++;
    }
    s.mean_abs = sum / a.size();
    s.frac_over = (double)over / a.size();
    return s;
}

// Grayscale heatmap: per-pixel mean diff x gain, clamped.
inline std::vector<uint8_t> diff_heatmap(const std::vector<uint8_t> &a,
                                         const std::vector<uint8_t> &b, int w, int h,
                                         double gain = 4.0) {
    std::vector<uint8_t> out(a.size(), 0);
    if (a.size() != b.size() || a.size() != (size_t)w * h * 3)
        return out;
    for (size_t i = 0; i + 2 < a.size(); i += 3) {
        double d = (std::fabs((double)a[i] - (double)b[i]) +
                    std::fabs((double)a[i + 1] - (double)b[i + 1]) +
                    std::fabs((double)a[i + 2] - (double)b[i + 2])) /
                   3.0;
        uint8_t v = (uint8_t)(d * gain > 255 ? 255 : d * gain);
        out[i] = out[i + 1] = out[i + 2] = v;
    }
    return out;
}
