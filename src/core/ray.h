#pragma once
#include "vec3.h"

// Ray pure data: origin, direction, shutter time. Time rides along
// untouched by integrator/materials; moving geometry reads it.
class ray {
public:
    ray() {}
    ray(const vec3 &o, const vec3 &d) : orig(o), dir(d), tm(0) {}
    ray(const vec3 &o, const vec3 &d, double time) : orig(o), dir(d), tm(time) {}

    vec3 origin() const { return orig; }
    vec3 direction() const { return dir; }
    vec3 at(double t) const { return orig + t * dir; }
    double time() const { return tm; }
    // Path time inheritance: scattered/shadow rays keep the primary's
    // time (motion consistency down the whole path, not just primaries).
    void set_time(double t) { tm = t; }

private:
    vec3 orig;
    vec3 dir;
    double tm = 0;
};
