#pragma once
// Output transform: exposure, tonemap in linear, then gamma 2.2.
// Applied per pixel between the linear framebuffer and quantization;
// the framebuffer itself always stays linear HDR.
#include "../core/vec3.h"

#include <cmath>

// ACES fitted approximation (Narkowicz): maps unbounded linear HDR into
// displayable range with a filmic shoulder — highlights roll off instead of
// clipping. Per-channel fit; slight hue shifts in extreme highlights are the
// known price of the cheap version.
inline double aces_fit(double x)
{
    return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
}

inline vec3 tonemap(const vec3 &linear_hdr, double exposure)
{
    vec3 c = linear_hdr * exposure;
    c = vec3(aces_fit(c.x()), aces_fit(c.y()), aces_fit(c.z()));
    const double inv_gamma = 1.0 / 2.2;
    return vec3(std::pow(c.x(), inv_gamma), std::pow(c.y(), inv_gamma), std::pow(c.z(), inv_gamma));
}
