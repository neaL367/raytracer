#pragma once
#include "hittable.h"
#include <memory>

// Sphere now a hittable carrying material. hit_record shared, no dup.
// Moving variant lerps centers over [t0,t1]; static when c0==c1.
class sphere : public hittable {
public:
    sphere() {}
    sphere(const vec3 &c, double r, std::shared_ptr<material> m)
        : c0(c), c1(c), radius(r), mat(m), tm0(0), tm1(1) {}
    sphere(const vec3 &a, const vec3 &b, double t0, double t1, double r,
           std::shared_ptr<material> m)
        : c0(a), c1(b), radius(r), mat(m), tm0(t0), tm1(t1) {}
    sphere(const vec3 &a, const vec3 &b, double r, std::shared_ptr<material> m)
        : c0(a), c1(b), radius(r), mat(m), tm0(0), tm1(1) {}

    vec3 center(double time) const {
        if (tm1 <= tm0)
            return c0;
        double f = (time - tm0) / (tm1 - tm0);
        f = f < 0 ? 0 : (f > 1 ? 1 : f);
        return c0 + f * (c1 - c0);
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        vec3 cen = center(r.time());
        vec3 oc = r.origin() - cen;
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
        vec3 outward = (rec.point - cen) / radius;
        rec.set_face_normal(r, outward);
        rec.mat = mat;
        rec.hit_obj = nullptr; // clear medium tag: shared tmp reuse must not
                               // misclassify solids in the shadow march (M48)
        rec.hit_prim = this; // nesting identity (M57)
        // Spherical UVs: azimuth -> u, polar -> v. Seam at -x, poles pinch.
        {
            vec3 op = (rec.point - cen) / radius;
            double theta = std::acos(op.y() < -1 ? -1 : (op.y() > 1 ? 1 : op.y()));
            double phi = std::atan2(-op.z(), op.x()) + 3.1415926535897932385;
            rec.u = phi / (2 * 3.1415926535897932385);
            rec.v = theta / 3.1415926535897932385;
            // Tangent along increasing azimuth; pole fallback.
            vec3 t(-op.z(), 0, op.x());
            if (t.length_squared() <= 1e-12)
                t = vec3(1, 0, 0);
            rec.tangent = unit_vector(t);
            rec.has_tangent = true;
        }
        return true;
    }

    bool bounding_box(aabb &box) const override {
        // Union of both endpoints: loose under motion, always correct.
        aabb b0(c0 - vec3(radius, radius, radius), c0 + vec3(radius, radius, radius));
        aabb b1(c1 - vec3(radius, radius, radius), c1 + vec3(radius, radius, radius));
        box = aabb::surrounding(b0, b1);
        return true;
    }

    // GPU flatten accessors (pure data out, no traversal knowledge).
    // Static-scene upload uses t=tm0 endpoint (motion mirror flagged).
    const vec3 &center_ref() const { return c0; }
    const vec3 &center1_ref() const { return c1; }
    void time_range(double &t0, double &t1) const {
        t0 = tm0;
        t1 = tm1;
    }
    double radius_val() const { return radius; }
    std::shared_ptr<material> mat_ptr() const { return mat; }

private:
    vec3 c0, c1;
    double radius = 0;
    std::shared_ptr<material> mat;
    double tm0 = 0, tm1 = 1;
};
