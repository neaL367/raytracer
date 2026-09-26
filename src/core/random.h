#pragma once
#include "vec3.h"
#include <cstdint>

inline uint64_t &rng_state() {
    static thread_local uint64_t s{0x9e3779b97f4a7c15ULL};
    return s;
}
inline uint64_t rng_next() {
    uint64_t z = (rng_state() += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
inline void rng_seed(unsigned s) {
    uint64_t z = static_cast<uint64_t>(s) + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    rng_state() = z ^ (z >> 31);
}
inline bool &rng_fixed_flag() {
    static thread_local bool f = false;
    return f;
}
inline double random_double() {
    if (rng_fixed_flag())
        return 0.5;
    return (rng_next() >> 11) * 1.1102230246251565e-16;
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
