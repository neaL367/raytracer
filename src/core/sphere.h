#pragma once
#include "hittable.h"
#include "vec3.h"

class sphere : public hittable
{
public:
    sphere(const vec3 &center, double radius) : center(center), radius(radius) {}

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
        rec.normal = (rec.point - center) / radius;

        return true;
    }

private:
    vec3 center;
    double radius;
};