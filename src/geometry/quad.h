#pragma once
#include "hittable.h"
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

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        double denom = dot(normal, r.direction());
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
        return true;
    }

private:
    vec3 Q, u, v, w, normal;
    double D = 0;
    std::shared_ptr<material> mat;
};
