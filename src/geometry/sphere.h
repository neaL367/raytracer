#pragma once
#include "hittable.h"
#include <memory>

// Sphere now a hittable carrying material. hit_record shared, no dup.
class sphere : public hittable {
public:
    sphere() {}
    sphere(const vec3 &c, double r, std::shared_ptr<material> m)
        : center(c), radius(r), mat(m) {}

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        vec3 oc = r.origin() - center;
        double a = dot(r.direction(), r.direction());
        double half_b = dot(oc, r.direction());
        double c = dot(oc, oc) - radius * radius;
        double discriminant = half_b * half_b - a * c;
        if (discriminant < 0)
            return false;
        double sqrtd = std::sqrt(discriminant);

        double root = (-half_b - sqrtd) / a;
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || root > t_max)
                return false;
        }
        rec.t = root;
        rec.point = r.at(root);
        vec3 outward = (rec.point - center) / radius;
        rec.set_face_normal(r, outward);
        rec.mat = mat;
        return true;
    }

    bool bounding_box(aabb &box) const override {
        box = aabb(center - vec3(radius, radius, radius),
                   center + vec3(radius, radius, radius));
        return true;
    }

private:
    vec3 center;
    double radius = 0;
    std::shared_ptr<material> mat;
};
