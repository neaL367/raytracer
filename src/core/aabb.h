#pragma once
#include "vec3.h"
#include "ray.h"
#include <algorithm>

// Axis-aligned box. Slab test per axis narrows [t_min,t_max]; parallel
// rays survive only if origin inside slab. Padded by builders to kill
// degenerate zero-thickness false misses.
class aabb {
public:
    vec3 minimum;
    vec3 maximum;

    aabb() {}
    aabb(const vec3 &a, const vec3 &b) : minimum(a), maximum(b) {}

    bool hit(const ray &r, double t_min, double t_max) const {
        for (int a = 0; a < 3; ++a) {
            double invD = 1.0 / r.direction().e[a];
            double t0 = (minimum.e[a] - r.origin().e[a]) * invD;
            double t1 = (maximum.e[a] - r.origin().e[a]) * invD;
            if (invD < 0.0)
                std::swap(t0, t1);
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min)
                return false;
        }
        return true;
    }

    static aabb surrounding(const aabb &a, const aabb &b) {
        vec3 lo(fmin(a.minimum.x(), b.minimum.x()),
                fmin(a.minimum.y(), b.minimum.y()),
                fmin(a.minimum.z(), b.minimum.z()));
        vec3 hi(fmax(a.maximum.x(), b.maximum.x()),
                fmax(a.maximum.y(), b.maximum.y()),
                fmax(a.maximum.z(), b.maximum.z()));
        return aabb(lo, hi);
    }

    int longest_axis() const {
        vec3 d = maximum - minimum;
        if (d.x() > d.y() && d.x() > d.z())
            return 0;
        return (d.y() > d.z()) ? 1 : 2;
    }

    double surface_area() const {
        vec3 d = maximum - minimum;
        return 2.0 * (d.x() * d.y() + d.y() * d.z() + d.z() * d.x());
    }
};
