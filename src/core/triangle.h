#pragma once
#include "hittable.h"
#include "vec3.h"
#include "material.h"
#include <memory>

class triangle : public hittable
{
public:
    triangle(const vec3 &v0, const vec3 &v1, const vec3 &v2, std::shared_ptr<material> mat)
        : v0(v0), v1(v1), v2(v2), mat(mat) {}

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override
    {
        const double epsilon = 1e-8;

        vec3 edge1 = v1 - v0;
        vec3 edge2 = v2 - v0;
        vec3 h = cross(r.direction(), edge2);
        double a = dot(edge1, h);

        if (std::fabs(a) < epsilon)
            return false; // ray is parallel to the triangle's plane

        double f = 1.0 / a;
        vec3 s = r.origin() - v0;
        double u = f * dot(s, h);

        if (u < 0.0 || u > 1.0)
            return false;

        vec3 q = cross(s, edge1);
        double v = f * dot(r.direction(), q);

        if (v < 0.0 || u + v > 1.0)
            return false;

        double t = f * dot(edge2, q);

        if (t < t_min || t > t_max)
            return false;

        rec.t = t;
        rec.point = r.at(t);
        vec3 outward_normal = unit_vector(cross(edge1, edge2));
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;

        return true;
    }

private:
    vec3 v0, v1, v2;
    std::shared_ptr<material> mat;
};