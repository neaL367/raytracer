#pragma once
#include "vec3.h"

// Orthonormal basis from a normal. Builds stable frame for cosine
// sampling: local (x,y,z) -> x*u + y*v + z*w aligns +z with normal.
class onb {
public:
    vec3 u, v, w;

    void build_from_w(const vec3 &n) {
        w = unit_vector(n);
        vec3 a = fabs(w.x()) > 0.9 ? vec3(0, 1, 0) : vec3(1, 0, 0);
        v = unit_vector(cross(w, a));
        u = cross(v, w);
    }
    vec3 local(double a, double b, double c) const {
        return a * u + b * v + c * w;
    }
    vec3 local(const vec3 &p) const {
        return p.x() * u + p.y() * v + p.z() * w;
    }
};
