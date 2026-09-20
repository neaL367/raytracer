#pragma once
#include "../core/vec3.h"
#include "../core/ray.h"

struct hit_record {
    double t = 0;
    vec3 point;
    vec3 normal; // always outward for M1 (single sphere, no inside case yet)
};

// Sphere analytic hit. half_b form used (b=-2h): fewer ops, stable.
// Interval [t_min,t_max] culls self-hit acne and far occluders.
class sphere {
public:
    sphere() {}
    sphere(const vec3 &c, double r) : center(c), radius(r) {}

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const {
        vec3 oc = r.origin() - center;
        double a = dot(r.direction(), r.direction());
        double half_b = dot(oc, r.direction());
        double c = dot(oc, oc) - radius * radius;
        double discriminant = half_b * half_b - a * c;
        if (discriminant < 0)
            return false;
        double sqrtd = std::sqrt(discriminant);

        double root = (-half_b - sqrtd) / a; // nearest first
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || root > t_max)
                return false;
        }
        rec.t = root;
        rec.point = r.at(root);
        rec.normal = (rec.point - center) / radius;
        return true;
    }

private:
    vec3 center;
    double radius = 0;
};
