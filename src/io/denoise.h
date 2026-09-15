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

// Joint bilateral: the range gate splits into color, albedo, and normal
// terms from guide buffers. A neighbor contributes only if it is nearby,
// similarly bright, the same material, AND facing the same way — texture
// edges (albedo differs) and silhouettes (normal differs) stop the blur
// even where color alone would bleed across. Guides are linear HDR albedo
// and mapped normals (n*0.5+0.5) from first-hit passes.
inline std::vector<vec3> joint_bilateral_denoise(const std::vector<vec3> &src,
                                                 const std::vector<vec3> &albedo,
                                                 const std::vector<vec3> &normal, int width, int height,
                                                 int radius = 2, double sigma_space = 1.0,
                                                 double sigma_range = 0.1, double sigma_albedo = 0.15,
                                                 double sigma_normal = 0.25)
{
    std::vector<vec3> dst(src.size());
    const double space_denom = 2.0 * sigma_space * sigma_space;
    const double range_denom = 2.0 * sigma_range * sigma_range;
    const double albedo_denom = 2.0 * sigma_albedo * sigma_albedo;
    const double normal_denom = 2.0 * sigma_normal * sigma_normal;
    auto decode_normal = [](const vec3 &m) {
        return vec3(m.x() * 2.0 - 1.0, m.y() * 2.0 - 1.0, m.z() * 2.0 - 1.0);
    };
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            size_t c0 = static_cast<size_t>(y) * width + x;
            vec3 center = src[c0];
            double center_lum = luminance_of(center);
            vec3 center_alb = albedo[c0];
            vec3 center_nrm = decode_normal(normal[c0]);
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
                    size_t ci = static_cast<size_t>(ny) * width + nx;
                    vec3 c = src[ci];
                    double dl = luminance_of(c) - center_lum;
                    vec3 da = albedo[ci] - center_alb;
                    vec3 nn = decode_normal(normal[ci]);
                    double dn = 1.0 - (center_nrm.x() * nn.x() + center_nrm.y() * nn.y() +
                                       center_nrm.z() * nn.z());
                    double w = std::exp(-(dx * dx + dy * dy) / space_denom - dl * dl / range_denom -
                                        (da.x() * da.x() + da.y() * da.y() + da.z() * da.z()) / albedo_denom -
                                        dn * dn / normal_denom);
                    acc += c * w;
                    wsum += w;
                }
            }
            // Same guarantee as plain bilateral: the center tap weighs 1.
            dst[c0] = acc * (1.0 / wsum);
        }
    }
    return dst;
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
