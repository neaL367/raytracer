#pragma once
#include "vec3.h"

// Ray pure data, no behavior beyond p(t). Reason: keeps sampling
// and intersection decoupled; integrator owns meaning of t.
class ray {
public:
    ray() {}
    ray(const vec3 &o, const vec3 &d) : orig(o), dir(d) {}

    vec3 origin() const { return orig; }
    vec3 direction() const { return dir; }
    vec3 at(double t) const { return orig + t * dir; }

private:
    vec3 orig;
    vec3 dir;
};
