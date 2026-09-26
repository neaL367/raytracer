#pragma once
#include "random.h"
#include <utility>
#include <vector>

// One sampler seam. Offsets in [0,1)^2 added to integer pixel coords.
// Stratified: n*n cells, one jittered sample each -> variance falls
// faster on edges than pure jitter. Non-squares fall back to jitter.
using sample_offset = std::pair<double, double>;

inline std::vector<sample_offset> jitter_offsets(int n) {
    std::vector<sample_offset> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i)
        out.emplace_back(random_double(), random_double());
    return out;
}

inline std::vector<sample_offset> stratified_offsets(int n) {
    std::vector<sample_offset> out;
    out.reserve(n * n);
    for (int iy = 0; iy < n; ++iy)
        for (int ix = 0; ix < n; ++ix)
            out.emplace_back((ix + random_double()) / n, (iy + random_double()) / n);
    // Shuffle cell order: avoids directional correlation in accumulation.
    for (int i = (int)out.size() - 1; i > 0; --i) {
        int j = (int)(random_double() * (i + 1));
        if (j > i)
            j = i;
        auto tmp = out[i];
        out[i] = out[j];
        out[j] = tmp;
    }
    return out;
}

// Cosine-weighted hemisphere dir (local frame, +z up). pdf = cos/PI.
// Used with ONB to align +z to surface normal.
inline vec3 random_cosine_direction() {
    double r1 = random_double();
    double r2 = random_double();
    double phi = 2 * 3.1415926535897932385 * r1;
    double x = std::cos(phi) * std::sqrt(r2);
    double y = std::sin(phi) * std::sqrt(r2);
    double z = std::sqrt(1 - r2);
    return vec3(x, y, z);
}
inline double cosine_pdf(double cos_theta) {
    return cos_theta <= 0 ? 0 : cos_theta / 3.1415926535897932385;
}

// Sobol-2D (Bratley-Fox, Gray order): dim0 = van der Corput, dim1 Joe-Kuo
// s=2/a=1/m=(1,3). Goldens: i=1 (1/2,1/2), i=2 (3/4,1/4), i=3 (1/4,3/4).
// First 4^m points tile every 2^-m grid cell exactly once: perfect
// stratification where jitter only hopes for it.
inline double sobol_dim0(unsigned i) {
    unsigned g = i ^ (i >> 1); // Gray code
    unsigned x = 0;
    for (int j = 0; j < 32; ++j)
        if (g & (1u << (unsigned)j))
            x ^= 1u << (31 - (unsigned)j);
    return x / 4294967296.0;
}
inline double sobol_dim1(unsigned i) {
    static unsigned V[32];
    static bool init = false;
    if (!init) {
        V[0] = 1u << 31;
        V[1] = 3u << 30;
        for (int j = 2; j < 32; ++j)
            V[j] = V[j - 2] ^ V[j - 1] ^ (V[j - 2] >> 2);
        init = true;
    }
    unsigned g = i ^ (i >> 1);
    unsigned x = 0;
    for (int j = 0; j < 32; ++j)
        if (g & (1u << (unsigned)j))
            x ^= V[j];
    return x / 4294967296.0;
}

// Cranley-Patterson rotated Sobol: one random shift mod 1 per pixel keeps
// the set unbiased while preserving the stratification. Same 2-draw cost
// profile as jitter (shift only), points deterministic given the shift.
inline std::vector<sample_offset> sobol_offsets(int n) {
    double sx = random_double(), sy = random_double();
    std::vector<sample_offset> out;
    out.reserve(n > 0 ? (size_t)n : 0);
    for (int k = 0; k < n; ++k) {
        double x = sobol_dim0((unsigned)k) + sx;
        double y = sobol_dim1((unsigned)k) + sy;
        out.emplace_back(x >= 1.0 ? x - 1.0 : x, y >= 1.0 ? y - 1.0 : y);
    }
    return out;
}
inline void fill_pixel_samples(int n, std::vector<sample_offset> &out) {
    if (n <= 1) {
        out.resize(1);
        out[0] = {0.5, 0.5};
        return;
    }
    int s = 0;
    for (; (s + 1) * (s + 1) <= n; ++s)
        ;
    if (s * s == n) {
        out.resize((size_t)(s * s));
        size_t idx = 0;
        for (int iy = 0; iy < s; ++iy)
            for (int ix = 0; ix < s; ++ix)
                out[idx++] = {(ix + random_double()) / s, (iy + random_double()) / s};
        for (int i = (int)out.size() - 1; i > 0; --i) {
            int j = (int)(random_double() * (i + 1));
            if (j > i)
                j = i;
            std::swap(out[(size_t)i], out[(size_t)j]);
        }
        return;
    }
    out.resize((size_t)n);
    for (int i = 0; i < n; ++i)
        out[(size_t)i] = {random_double(), random_double()};
}

// Perfect square -> strata, else jitter. Sobol utils remain for tests;
// the renderer uses the stratified path only.
inline std::vector<sample_offset> pixel_samples(int n) {
    std::vector<sample_offset> out;
    fill_pixel_samples(n, out);
    return out;
}
