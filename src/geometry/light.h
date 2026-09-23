#pragma once
// Unified area-light view over quad/sphere/triangle for NEE.
// Same math every shape: uniform point (exactly 2 RNG draws), geometric
// area, geometric normal; pdf_area = d^2/(n*A*cosA). Sphere honors
// center(time), so moving lights sample correctly. Sampling pdfs are
// geometric even for smooth-shaded tris (MIS reverse uses these too).
#include "hittable.h"
#include "quad.h"
#include "sphere.h"
#include "triangle.h"
#include "../material/material.h"

#include <cmath>
#include <memory>

struct light {
    light() {}
    light(const std::shared_ptr<quad> &q) : shape(q) {}
    light(const std::shared_ptr<sphere> &s) : shape(s) {}
    light(const std::shared_ptr<triangle> &t) : shape(t) {}
    std::shared_ptr<hittable> shape;
};

inline vec3 light_point(const light &lt, double u1, double u2, double time) {
    if (auto q = std::dynamic_pointer_cast<quad>(lt.shape))
        return q->corner() + u1 * q->edge_u() + u2 * q->edge_v();
    if (auto s = std::dynamic_pointer_cast<sphere>(lt.shape)) {
        // Exact uniform on sphere (no rejection, fixed draw count).
        double z = 1.0 - 2.0 * u1;
        double phi = 2.0 * 3.1415926535897932385 * u2;
        double r = std::sqrt(std::max(1.0 - z * z, 0.0));
        vec3 off(r * std::cos(phi), r * std::sin(phi), z);
        return s->center(time) + s->radius_val() * off;
    }
    auto t = std::dynamic_pointer_cast<triangle>(lt.shape);
    // Sqrt-barycentric: uniform over the triangle, fixed draw count.
    // Moving tris sample lerped verts (mirrors the hit path).
    double su = std::sqrt(u1);
    return t->vert_at(0, time) * (1.0 - su) + t->vert_at(1, time) * (su * (1.0 - u2)) +
           t->vert_at(2, time) * (su * u2);
}

inline vec3 light_normal_at(const light &lt, const vec3 &p, double time) {
    // Rigid-motion exact (translation preserves normals); deforming tris
    // approximate with base-pose geometry, like static smooth normals.
    if (auto q = std::dynamic_pointer_cast<quad>(lt.shape))
        return q->light_normal();
    if (auto s = std::dynamic_pointer_cast<sphere>(lt.shape))
        return unit_vector(p - s->center(time));
    auto t = std::dynamic_pointer_cast<triangle>(lt.shape);
    vec3 e1 = t->vert(1) - t->vert(0), e2 = t->vert(2) - t->vert(0);
    return unit_vector(cross(e1, e2));
}

inline double light_area(const light &lt) {
    if (auto q = std::dynamic_pointer_cast<quad>(lt.shape))
        return q->area();
    if (auto s = std::dynamic_pointer_cast<sphere>(lt.shape)) {
        double r = s->radius_val();
        return 4.0 * 3.1415926535897932385 * r * r;
    }
    auto t = std::dynamic_pointer_cast<triangle>(lt.shape);
    vec3 e1 = t->vert(1) - t->vert(0), e2 = t->vert(2) - t->vert(0);
    return 0.5 * cross(e1, e2).length();
}

inline std::shared_ptr<material> light_mat(const light &lt) {
    if (auto q = std::dynamic_pointer_cast<quad>(lt.shape))
        return q->mat_ptr();
    if (auto s = std::dynamic_pointer_cast<sphere>(lt.shape))
        return s->mat_ptr();
    return std::dynamic_pointer_cast<triangle>(lt.shape)->mat_ptr();
}

// UVs for an NEE light sample drawn with (u1, u2): the same coordinates
// the shape hit path would report for that point, so textured emission
// matches direct camera hits. Returns false for unknown shapes.
inline bool light_uv(const light &lt, double u1, double u2, double time, double &u,
                     double &v) {
    if (std::dynamic_pointer_cast<quad>(lt.shape)) {
        u = u1;
        v = u2;
        return true;
    }
    if (auto s = std::dynamic_pointer_cast<sphere>(lt.shape)) {
        (void)s;
        (void)time; // UVs use the sampled offset direction, not the center
        const double pi = 3.1415926535897932385;
        double z = 1.0 - 2.0 * u1;
        double phi = 2.0 * pi * u2;
        double r = std::sqrt(std::max(1.0 - z * z, 0.0));
        vec3 off(r * std::cos(phi), r * std::sin(phi), z);
        double oy = off.y() < -1 ? -1 : (off.y() > 1 ? 1 : off.y());
        double theta = std::acos(oy);
        double a = std::atan2(-off.z(), off.x()) + pi;
        u = a / (2.0 * pi);
        v = theta / pi;
        return true;
    }
    if (auto t = std::dynamic_pointer_cast<triangle>(lt.shape)) {
        (void)time; // barycentric weights are motion-independent
        double su = std::sqrt(u1);
        double w0 = 1.0 - su;
        double w1 = su * (1.0 - u2);
        double w2 = su * u2;
        if (t->uv_present()) {
            vec3 c0 = t->uv_vert(0), c1 = t->uv_vert(1), c2 = t->uv_vert(2);
            u = c0.x() * w0 + c1.x() * w1 + c2.x() * w2;
            v = c0.y() * w0 + c1.y() * w1 + c2.y() * w2;
        } else {
            u = w1;
            v = w2;
        }
        return true;
    }
    return false;
}

// Emission at an NEE sample: synthesize the hit record the material needs.
// `dist` doubles as the texture LOD distance, mirroring direct hits.
inline vec3 light_emission(const light &lt, const vec3 &lp, double u, double v,
                           double dist) {
    hit_record rec;
    rec.point = lp;
    rec.u = u;
    rec.v = v;
    rec.t = dist;
    return light_mat(lt)->emitted(rec);
}
