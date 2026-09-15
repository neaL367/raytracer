#pragma once

#include "hittable.h"
#include "vec3.h"
#include "material.h"

#include <cmath>
#include <memory>
#include <numbers>

class sphere : public hittable
{
public:
    sphere(const vec3 &center, double radius, std::shared_ptr<material> mat)
        : center(center), radius(radius), mat(mat) {}

    // Read access for scene upload to GPU buffers (SoA flattening).
    const vec3 &position() const { return center; }
    double size() const { return radius; }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override
    {
        count_prim_test();
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
        if (mat->needs_uv())
            get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.mat = mat;

        return true;
    }

    bool bounding_box(aabb &output_box) const override
    {
        output_box = aabb(
            center - vec3(radius, radius, radius),
            center + vec3(radius, radius, radius));
        return true;
    }

private:
    // Unit-sphere parametrization: latitude from the pole, longitude around
    // the equator. Stable everywhere except exactly at the poles.
    static void get_sphere_uv(const vec3 &p, double &u, double &v)
    {
        double theta = std::acos(-p.y());
        double phi = std::atan2(-p.z(), p.x()) + std::numbers::pi;
        u = phi / (2.0 * std::numbers::pi);
        v = theta / std::numbers::pi;
    }

    vec3 center;
    double radius;
    std::shared_ptr<material> mat;
};