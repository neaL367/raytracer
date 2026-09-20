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
        // Spherical UVs: azimuth -> u, polar -> v. Seam at -x, poles pinch.
        {
            vec3 op = (rec.point - center) / radius;
            double theta = std::acos(op.y() < -1 ? -1 : (op.y() > 1 ? 1 : op.y()));
            double phi = std::atan2(-op.z(), op.x()) + 3.1415926535897932385;
            rec.u = phi / (2 * 3.1415926535897932385);
            rec.v = theta / 3.1415926535897932385;
        }
        return true;
    }

    bool bounding_box(aabb &box) const override {
        box = aabb(center - vec3(radius, radius, radius),
                   center + vec3(radius, radius, radius));
        return true;
    }

    // GPU flatten accessors (pure data out, no traversal knowledge).
    const vec3 &center_ref() const { return center; }
    double radius_val() const { return radius; }
    std::shared_ptr<material> mat_ptr() const { return mat; }

private:
    vec3 center;
    double radius = 0;
    std::shared_ptr<material> mat;
};
