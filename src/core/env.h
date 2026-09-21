#pragma once
// Analytic environment light (M38 v1): legacy sky gradient + a small sun
// disc, sampled uniformly on the sphere (pdf 1/4PI) with power-heuristic MIS
// against the BSDF. Opt-in via scene flag; off = legacy gradient miss,
// byte-exact. Mirrored in path.comp (env_radiance/env_sample). A textured
// HDRI with luminance-CDF sampling is the documented next step, not this one.
#include "vec3.h"

#include <cmath>

namespace env_light {

inline vec3 sun_dir() { return unit_vector(vec3(0.5, 0.8, 0.35)); }

// Legacy background gradient, now the single source for both paths.
inline vec3 sky(const vec3 &d) {
    vec3 unit = unit_vector(d);
    double t = 0.5 * (unit.y() + 1.0);
    return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

// Sky plus sun disc (quadratic ramp across the rim, ~30x white at center).
inline vec3 radiance(const vec3 &d) {
    vec3 ud = unit_vector(d);
    vec3 L = sky(ud);
    double s = dot(ud, sun_dir());
    if (s > 0.9993) {
        double k = (s - 0.9993) / (1.0 - 0.9993);
        L = L + vec3(30.0, 25.0, 18.0) * (k * k);
    }
    return L;
}

inline double sample_pdf() { return 1.0 / (4.0 * 3.1415926535897932385); }

// Uniform sphere direction from 2 draws (mirrors light_point sphere branch).
inline vec3 sample_dir(double u1, double u2) {
    double z = 1.0 - 2.0 * u1;
    double phi = 2.0 * 3.1415926535897932385 * u2;
    double r = std::sqrt(std::max(1.0 - z * z, 0.0));
    return vec3(r * std::cos(phi), r * std::sin(phi), z);
}

} // namespace env_light
