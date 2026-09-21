#pragma once
// Table-free value noise, NTW Ch.5 flavor: integer lattice hash +
// Hermite-smoothed trilinear + fBm turbulence. One formula on CPU and GPU
// (mirrored in path.comp as vnoise/vturb); the probe values below are
// bit-exact on CPU, statistical parity across backends.
#include "vec3.h"

#include <cmath>
#include <cstdint>

namespace value_noise {

// Wrapping multiply hash; uint32 casts keep negative lattice coords defined
// (mod 2^32), and GLSL uint arithmetic matches bit-for-bit.
inline uint32_t lattice_hash(int x, int y, int z) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u +
                 (uint32_t)z * 1440662683u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}

inline double lattice_unit(int x, int y, int z) {
    return (lattice_hash(x, y, z) & 0xffffu) / 65535.0;
}

inline double fade(double t) { return t * t * (3.0 - 2.0 * t); }

// Smoothed lattice noise in [0,1]. Point is pre-scaled by the caller.
inline double at(const vec3 &p) {
    int xi = (int)std::floor(p.x()), yi = (int)std::floor(p.y()),
        zi = (int)std::floor(p.z());
    double xf = p.x() - xi, yf = p.y() - yi, zf = p.z() - zi;
    double u = fade(xf), v = fade(yf), w = fade(zf);
    double c000 = lattice_unit(xi, yi, zi), c100 = lattice_unit(xi + 1, yi, zi);
    double c010 = lattice_unit(xi, yi + 1, zi), c110 = lattice_unit(xi + 1, yi + 1, zi);
    double c001 = lattice_unit(xi, yi, zi + 1), c101 = lattice_unit(xi + 1, yi, zi + 1);
    double c011 = lattice_unit(xi, yi + 1, zi + 1),
           c111 = lattice_unit(xi + 1, yi + 1, zi + 1);
    double x00 = c000 + u * (c100 - c000), x10 = c010 + u * (c110 - c010);
    double x01 = c001 + u * (c101 - c001), x11 = c011 + u * (c111 - c011);
    double y0 = x00 + v * (x10 - x00), y1 = x01 + v * (x11 - x01);
    return y0 + w * (y1 - y0);
}

// fBm turbulence: octave-weighted |noise|, normalized to [0,1] by weights.
inline double turb(const vec3 &p, int depth) {
    double sum = 0.0, weight = 0.0, amp = 1.0;
    vec3 q = p;
    for (int i = 0; i < depth; ++i) {
        sum += amp * std::fabs(at(q));
        weight += amp;
        amp *= 0.5;
        q = q * 2.0;
    }
    return weight > 0 ? sum / weight : 0.0;
}

} // namespace value_noise
