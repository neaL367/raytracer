#pragma once
// Edge-aware bilateral denoise over the linear HDR framebuffer, applied
// after rendering and before tonemapping (denoising display-referred data
// crushes shadows — the filter must see linear light). Each pixel becomes a
// weighted average of its neighborhood: gaussian falloff in space times
// gaussian falloff in luminance difference, so flat noise averages away
// while true edges (large luminance steps) keep their weight near zero and
// survive. Pure function of the framebuffer: same input, same output.
#include "../core/vec3.h"

#include <cmath>
#include <vector>

inline double luminance_of(const vec3 &c)
{
    return 0.2126 * c.x() + 0.7152 * c.y() + 0.0722 * c.z();
}

inline std::vector<vec3> bilateral_denoise(const std::vector<vec3> &src, int width, int height,
                                           int radius = 2, double sigma_space = 1.0,
                                           double sigma_range = 0.1)
{
    std::vector<vec3> dst(src.size());
    const double space_denom = 2.0 * sigma_space * sigma_space;
    const double range_denom = 2.0 * sigma_range * sigma_range;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            vec3 center = src[static_cast<size_t>(y) * width + x];
            double center_lum = luminance_of(center);
            vec3 acc(0, 0, 0);
            double wsum = 0.0;
            for (int dy = -radius; dy <= radius; ++dy)
            {
                int ny = y + dy;
                if (ny < 0)
                    ny = 0;
                if (ny > height - 1)
                    ny = height - 1;
                for (int dx = -radius; dx <= radius; ++dx)
                {
                    int nx = x + dx;
                    if (nx < 0)
                        nx = 0;
                    if (nx > width - 1)
                        nx = width - 1;
                    vec3 c = src[static_cast<size_t>(ny) * width + nx];
                    double dl = luminance_of(c) - center_lum;
                    double w = std::exp(-(dx * dx + dy * dy) / space_denom - dl * dl / range_denom);
                    acc += c * w;
                    wsum += w;
                }
            }
            // wsum >= 1: the center tap itself (d = 0, dl = 0) weighs exactly
            // 1, so the quotient is always defined for finite inputs.
            dst[static_cast<size_t>(y) * width + x] = acc * (1.0 / wsum);
        }
    }
    return dst;
}
