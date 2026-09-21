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
        : v0(a), v1(b), v2(c), v0b(a), v1b(b), v2b(c), mat(m), smooth(false) {}
    triangle(const vec3 &a, const vec3 &b, const vec3 &c, const vec3 &na,
             const vec3 &nb, const vec3 &nc, std::shared_ptr<material> m)
        : v0(a), v1(b), v2(c), v0b(a), v1b(b), v2b(c), n0(na), n1(nb), n2(nc), mat(m),
          smooth(true), has_uv(false) {}
    // UV variant: corner UVs interpolate to rec.u/v; combines with smooth.
    triangle(const vec3 &a, const vec3 &b, const vec3 &c, double uu0, double vv0,
             double uu1, double vv1, double uu2, double vv2,
             std::shared_ptr<material> m)
        : v0(a), v1(b), v2(c), v0b(a), v1b(b), v2b(c), mat(m), smooth(false),
          has_uv(true), t0(uu0, vv0, 0), t1(uu1, vv1, 0), t2(uu2, vv2, 0) {}
    // Full variant: smooth normals plus corner UVs.
    triangle(const vec3 &a, const vec3 &b, const vec3 &c, const vec3 &na,
             const vec3 &nb, const vec3 &nc, double uu0, double vv0, double uu1,
             double vv1, double uu2, double vv2, std::shared_ptr<material> m)
        : v0(a), v1(b), v2(c), v0b(a), v1b(b), v2b(c), n0(na), n1(nb), n2(nc), mat(m),
          smooth(true), has_uv(true), t0(uu0, vv0, 0), t1(uu1, vv1, 0), t2(uu2, vv2, 0) {}

    // Motion endpoints (scene-applied; OBJ carries no motion). Smooth
    // normals stay static — exact for rigid translation.
    void set_motion(const vec3 &a, const vec3 &b, const vec3 &c, double t0, double t1) {
        v0b = a;
        v1b = b;
        v2b = c;
        tm0 = t0;
        tm1 = t1;
    }
    vec3 vert_at(int i, double time) const {
        const vec3 &a = (i == 0) ? v0 : ((i == 1) ? v1 : v2);
        const vec3 &b = (i == 0) ? v0b : ((i == 1) ? v1b : v2b);
        if (tm1 <= tm0)
            return a;
        double f = (time - tm0) / (tm1 - tm0);
        f = f < 0 ? 0 : (f > 1 ? 1 : f);
        return a + f * (b - a);
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        const double eps = 1e-8;
        vec3 p0 = vert_at(0, r.time()), p1 = vert_at(1, r.time()),
             p2 = vert_at(2, r.time());
        vec3 e1 = p1 - p0, e2 = p2 - p0;
        vec3 pvec = cross(r.direction(), e2);
        double det = dot(e1, pvec);
        if (fabs(det) < eps)
            return false; // parallel
        double inv = 1.0 / det;
        vec3 tvec = r.origin() - p0;
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
        if (has_uv) {
            // Corner-UV blend (image textures consume it directly).
            double w0 = 1 - u - v;
            rec.u = t0.x() * w0 + t1.x() * u + t2.x() * v;
            rec.v = t0.y() * w0 + t1.y() * u + t2.y() * v;
        } else {
            rec.u = u; // barycentric weights as UVs (sum <= 1)
            rec.v = v;
        }
        // Tangent from UV derivatives (Mikkelsen-lite) on lerped verts;
        // barycentric UVs reduce to e1. Degenerate -> fallback flag off.
        {
            vec3 duv1, duv2;
            if (has_uv) {
                duv1 = vec3(t1.x() - t0.x(), t1.y() - t0.y(), 0);
                duv2 = vec3(t2.x() - t0.x(), t2.y() - t0.y(), 0);
            } else {
                duv1 = vec3(1, 0, 0);
                duv2 = vec3(0, 1, 0);
            }
            double det = duv1.x() * duv2.y() - duv2.x() * duv1.y();
            vec3 flat = unit_vector(cross(e1, e2));
            if (fabs(det) > 1e-12) {
                vec3 t = (e1 * duv2.y() - e2 * duv1.y()) / det;
                t = t - flat * dot(t, flat);
                if (t.length_squared() > 1e-12) {
                    rec.tangent = unit_vector(t);
                    rec.has_tangent = true;
                }
            }
        }
        return true;
    }

    bool bounding_box(aabb &box) const override {        const double pad = 1e-4; // zero-thickness plane needs slab volume
        // Union of both motion endpoints: loose under motion, always correct.
        vec3 ps[6] = {v0, v1, v2, v0b, v1b, v2b};
        vec3 lo = ps[0], hi = ps[0];
        for (int i = 1; i < 6; ++i) {
            lo = vec3(fmin(lo.x(), ps[i].x()), fmin(lo.y(), ps[i].y()),
                      fmin(lo.z(), ps[i].z()));
            hi = vec3(fmax(hi.x(), ps[i].x()), fmax(hi.y(), ps[i].y()),
                      fmax(hi.z(), ps[i].z()));
        }
        box = aabb(lo - vec3(pad, pad, pad), hi + vec3(pad, pad, pad));
        return true;
    }

    // GPU flatten accessors.
    const vec3 &vert(int i) const { return (i == 0) ? v0 : ((i == 1) ? v1 : v2); }
    const vec3 &vert1(int i) const { return (i == 0) ? v0b : ((i == 1) ? v1b : v2b); }
    void time_range(double &t0, double &t1) const {
        t0 = tm0;
        t1 = tm1;
    }
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
    bool uv_present() const { return has_uv; }
    vec3 uv_vert(int i) const { return (i == 0) ? t0 : ((i == 1) ? t1 : t2); }

private:
    vec3 v0, v1, v2;
    vec3 v0b, v1b, v2b; // motion endpoints (== base when static)
    double tm0 = 0, tm1 = 1;
    vec3 n0, n1, n2; // valid only when smooth
    bool smooth = false;
    vec3 t0, t1, t2; // corner UVs in x/y; valid only when has_uv
    bool has_uv = false;
    std::shared_ptr<material> mat;
};
