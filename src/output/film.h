#pragma once
#include "../core/vec3.h"
#include <cmath>

// Film: linear HDR -> exposure -> ACES -> sRGB LDR.
// Narkowicz ACES fit: cheap, monotonic, unbounded-in/[0,1]-out.
// sRGB piecewise (toe + power). Shared by CPU + GPU writers,
// so parity holds by construction.
inline vec3 aces_approx(const vec3 &x) {
    auto curve = [](double v) {
        v = v < 0 ? 0 : v;
        return (v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14);
    };
    double r = curve(x.x()), g = curve(x.y()), b = curve(x.z());
    r = r > 1 ? 1 : r; // fit overshoots 1 slightly at extremes
    g = g > 1 ? 1 : g;
    b = b > 1 ? 1 : b;
    return vec3(r, g, b);
}

inline double srgb_encode(double v) {
    v = v < 0 ? 0 : (v > 1 ? 1 : v);
    return (v <= 0.0031308) ? 12.92 * v : 1.055 * std::pow(v, 1.0 / 2.4) - 0.055;
}

inline vec3 tonemap(const vec3 &hdr, double exposure) {
    vec3 e = hdr * exposure;
    vec3 a = aces_approx(e);
    return vec3(srgb_encode(a.x()), srgb_encode(a.y()), srgb_encode(a.z()));
}
