#pragma once
#include "vec3.h"

// Ray pure data: origin, direction, shutter time, and cached inverse direction.
// Caching inv_dir upon construction eliminates millions of floating-point divisions
// in AABB slab and QBVH node traversal without any change in bit-exactness.
class ray {
public:
    ray() : orig{0,0,0}, dir{0,0,0}, inv_dir{0,0,0}, tm(0) {}
    ray(const vec3 &o, const vec3 &d)
        : orig(o), dir(d),
          inv_dir(1.0 / d.x(), 1.0 / d.y(), 1.0 / d.z()),
          tm(0) {}
    ray(const vec3 &o, const vec3 &d, double time)
        : orig(o), dir(d),
          inv_dir(1.0 / d.x(), 1.0 / d.y(), 1.0 / d.z()),
          tm(time) {}

    vec3 origin() const { return orig; }
    vec3 direction() const { return dir; }
    vec3 inv_direction() const { return inv_dir; }
    vec3 at(double t) const { return orig + t * dir; }
    double time() const { return tm; }
    // Path time inheritance: scattered/shadow rays keep the primary's
    // time (motion consistency down the whole path, not just primaries).
    void set_time(double t) { tm = t; }

private:
    vec3 orig;
    vec3 dir;
    vec3 inv_dir;
    double tm = 0;
};
