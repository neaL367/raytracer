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

// Perfect square -> strata, else jitter. n=1 returns center (M2 path).
inline std::vector<sample_offset> pixel_samples(int n) {
    if (n <= 1)
        return {{0.5, 0.5}};
    int s = 0;
    for (; (s + 1) * (s + 1) <= n; ++s)
        ;
    if (s * s == n)
        return stratified_offsets(s);
    return jitter_offsets(n);
}
