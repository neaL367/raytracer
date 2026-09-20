#pragma once
#include "hittable.h"
#include "../core/random.h"
#include <memory>

// Parallelogram from corner Q + edges u,v. Ray-plane then alpha/beta
// window check. Same interface, renderer never special-cases shape.
class quad : public hittable {
public:
    quad() {}
    quad(const vec3 &q, const vec3 &a, const vec3 &b, std::shared_ptr<material> m)
        : Q(q), u(a), v(b), mat(m) {
        vec3 n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q);
        w = n / dot(n, n);
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {        double denom = dot(normal, r.direction());
        if (fabs(denom) < 1e-8)
            return false; // parallel
        double t = (D - dot(normal, r.origin())) / denom;
        if (t < t_min || t > t_max)
            return false;
        vec3 p = r.at(t);
        vec3 pq = p - Q;
        double alpha = dot(w, cross(pq, v));
        double beta = dot(w, cross(u, pq));
        if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1)
            return false;
        rec.t = t;
        rec.point = p;
        rec.set_face_normal(r, normal);
        rec.mat = mat;
        rec.u = alpha; // parametric coords double as UVs
        rec.v = beta;
        return true;
    }

    bool bounding_box(aabb &box) const override {
        const double pad = 1e-4;
        vec3 c1 = Q + u, c2 = Q + v, c3 = Q + u + v;
        vec3 lo(fmin(Q.x(), fmin(c1.x(), fmin(c2.x(), c3.x()))) - pad,
                fmin(Q.y(), fmin(c1.y(), fmin(c2.y(), c3.y()))) - pad,
                fmin(Q.z(), fmin(c1.z(), fmin(c2.z(), c3.z()))) - pad);
        vec3 hi(fmax(Q.x(), fmax(c1.x(), fmax(c2.x(), c3.x()))) + pad,
                fmax(Q.y(), fmax(c1.y(), fmax(c2.y(), c3.y()))) + pad,
                fmax(Q.z(), fmax(c1.z(), fmax(c2.z(), c3.z()))) + pad);
        box = aabb(lo, hi);
        return true;
    }

    // Light-sampling support: uniform point + area for NEE pdf.
    double area() const { return cross(u, v).length(); }
    vec3 sample_point() const {
        return Q + random_double() * u + random_double() * v;
    }
    vec3 light_normal() const { return normal; }
    std::shared_ptr<material> mat_ptr() const { return mat; }
    // GPU flatten accessors.
    const vec3 &corner() const { return Q; }
    const vec3 &edge_u() const { return u; }
    const vec3 &edge_v() const { return v; }

private:
    vec3 Q, u, v, w, normal;
    double D = 0;
    std::shared_ptr<material> mat;
};
