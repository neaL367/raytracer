#pragma once
#include "vec3.h"
#include "ray.h"
#include "bench_stats.h"
#include <algorithm>

class aabb // axis-aligned bounding box: represents a rectangular prism in 3D space, defined by its minimum and maximum corner points
{
public:
    aabb() {}
    aabb(const vec3 &a, const vec3 &b) : minimum(a), maximum(b) {}

    const vec3 &min() const { return minimum; }
    const vec3 &max() const { return maximum; }

    // Surface area: 2(xy + yz + zx) of the box sides. Used to weigh split
    // candidates — a ray hits a box with probability proportional to this.
    double surface_area() const
    {
        vec3 d = maximum - minimum;
        return 2.0 * (d.x() * d.y() + d.y() * d.z() + d.z() * d.x());
    }

    bool hit(const ray &r, double t_min, double t_max) const
    {
        count_box_test();
        for (int a = 0; a < 3; a++)
        {
            double origin_a = (a == 0) ? r.origin().x() : (a == 1) ? r.origin().y()
                                                                   : r.origin().z();
            double inv_d = (a == 0) ? r.inv_direction().x() : (a == 1) ? r.inv_direction().y()
                                                                       : r.inv_direction().z();
            double min_a = (a == 0) ? minimum.x() : (a == 1) ? minimum.y()
                                                             : minimum.z();
            double max_a = (a == 0) ? maximum.x() : (a == 1) ? maximum.y()
                                                             : maximum.z();

            double t0 = (min_a - origin_a) * inv_d;
            double t1 = (max_a - origin_a) * inv_d;

            if (inv_d < 0.0)
                std::swap(t0, t1);

            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;

            if (t_max <= t_min)
                return false;
        }
        return true;
    }

private:
    vec3 minimum, maximum;
};

inline aabb surrounding_box(const aabb &box0, const aabb &box1)
{
    vec3 small(std::fmin(box0.min().x(), box1.min().x()),
               std::fmin(box0.min().y(), box1.min().y()),
               std::fmin(box0.min().z(), box1.min().z()));
    vec3 big(std::fmax(box0.max().x(), box1.max().x()),
             std::fmax(box0.max().y(), box1.max().y()),
             std::fmax(box0.max().z(), box1.max().z()));
    return aabb(small, big);
}