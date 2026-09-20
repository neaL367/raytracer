#pragma once
#include "hittable.h"
#include <memory>

// Möller–Trumbore, double-sided via set_face_normal. Determinant eps
// rejects parallel rays. Optional vertex normals: barycentric blend +
// renormalize gives smooth shading; unset = flat face normal.
class triangle : public hittable {
public:
    triangle() {}
    triangle(const vec3 &a, const vec3 &b, const vec3 &c, std::shared_ptr<material> m)
        : v0(a), v1(b), v2(c), mat(m), smooth(false) {}
    triangle(const vec3 &a, const vec3 &b, const vec3 &c, const vec3 &na,
             const vec3 &nb, const vec3 &nc, std::shared_ptr<material> m)
        : v0(a), v1(b), v2(c), n0(na), n1(nb), n2(nc), mat(m), smooth(true) {}

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        const double eps = 1e-8;
        vec3 e1 = v1 - v0, e2 = v2 - v0;
        vec3 pvec = cross(r.direction(), e2);
        double det = dot(e1, pvec);
        if (fabs(det) < eps)
            return false; // parallel
        double inv = 1.0 / det;
        vec3 tvec = r.origin() - v0;
        double u = dot(tvec, pvec) * inv;
        if (u < 0 || u > 1)
            return false;
        vec3 qvec = cross(tvec, e1);
        double v = dot(r.direction(), qvec) * inv;
        if (v < 0 || u + v > 1)
            return false;
        double t = dot(e2, qvec) * inv;
        if (t < t_min || t > t_max)
            return false;
        rec.t = t;
        rec.point = r.at(t);
        vec3 flat = unit_vector(cross(e1, e2));
        if (smooth) {
            // Barycentric blend of vertex normals, renormalized. Face
            // flip preserved (double-sided): blend first, orient after.
            double w0 = 1 - u - v;
            vec3 blended = unit_vector(n0 * w0 + n1 * u + n2 * v);
            // Orient toward ray via the face side (blended ~= face dir).
            rec.set_face_normal(r, dot(blended, flat) < 0 ? -blended : blended);
        } else {
            rec.set_face_normal(r, flat);
        }
        rec.mat = mat;
        rec.u = u; // barycentric weights as UVs (sum <= 1)
        rec.v = v;
        return true;
    }

    bool bounding_box(aabb &box) const override {        const double pad = 1e-4; // zero-thickness plane needs slab volume
        vec3 lo(fmin(v0.x(), fmin(v1.x(), v2.x())) - pad,
                fmin(v0.y(), fmin(v1.y(), v2.y())) - pad,
                fmin(v0.z(), fmin(v1.z(), v2.z())) - pad);
        vec3 hi(fmax(v0.x(), fmax(v1.x(), v2.x())) + pad,
                fmax(v0.y(), fmax(v1.y(), v2.y())) + pad,
                fmax(v0.z(), fmax(v1.z(), v2.z())) + pad);
        box = aabb(lo, hi);
        return true;
    }

    // GPU flatten accessors.
    const vec3 &vert(int i) const { return (i == 0) ? v0 : ((i == 1) ? v1 : v2); }
    // Smooth normals for upload; flat tris report the face normal x3
    // (same pixels, small upload cost, one shader path).
    vec3 norm_vert(int i) const {
        if (!smooth) {
            vec3 e1 = v1 - v0, e2 = v2 - v0;
            return unit_vector(cross(e1, e2));
        }
        return (i == 0) ? n0 : ((i == 1) ? n1 : n2);
    }
    std::shared_ptr<material> mat_ptr() const { return mat; }

private:
    vec3 v0, v1, v2;
    vec3 n0, n1, n2; // valid only when smooth
    bool smooth = false;
    std::shared_ptr<material> mat;
};
