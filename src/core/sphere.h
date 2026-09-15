#pragma once

#include "hittable.h"
#include "vec3.h"
#include "material.h"

#include <cmath>
#include <memory>

class sphere : public hittable
{
public:
    sphere(const vec3 &center, double radius, std::shared_ptr<material> mat)
        : center(center), radius(radius), mat(mat) {}

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override
    {
        vec3 oc = r.origin() - center;
        double a = r.direction().length_squared();
        double half_b = dot(oc, r.direction());
        double c = oc.length_squared() - radius * radius;

        double discriminant = half_b * half_b - a * c;
        if (discriminant < 0)
            return false;

        double sqrt_d = std::sqrt(discriminant);

        double root = (-half_b - sqrt_d) / a;
        if (root < t_min || root > t_max)
        {
            root = (-half_b + sqrt_d) / a;
            if (root < t_min || root > t_max)
                return false;
        }

        rec.t = root;
        rec.point = r.at(rec.t);
        vec3 outward_normal = (rec.point - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;

        return true;
    }

private:
    vec3 center;
    double radius;
    std::shared_ptr<material> mat;
};