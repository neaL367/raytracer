#pragma once
#include "vec3.h"
#include <random>

// Single RNG seam. Fixed seed for tests, reseeded per render.
// Reason: deterministic tests, decorrelated pixels later.
inline std::mt19937 &rng_engine() {
    static thread_local std::mt19937 eng{42};
    return eng;
}
inline void rng_seed(unsigned s) { rng_engine().seed(s); }
inline double random_double() {
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_engine());
}
inline double random_double(double lo, double hi) {
    return lo + (hi - lo) * random_double();
}
inline vec3 vec3::random() {
    return vec3(random_double(), random_double(), random_double());
}
inline vec3 vec3::random(double min, double max) {
    return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
}
inline vec3 random_in_unit_sphere() {
    for (;;) {
        vec3 p(random_double(-1, 1), random_double(-1, 1), random_double(-1, 1));
        if (p.length_squared() < 1)
            return p;
    }
}
inline vec3 random_unit_vector() { return unit_vector(random_in_unit_sphere()); }
// Uniform disk (z=0) for thin-lens aperture sampling.
inline vec3 random_in_unit_disk() {
    for (;;) {
        vec3 p(random_double(-1, 1), random_double(-1, 1), 0);
        if (p.length_squared() < 1)
            return p;
    }
}
