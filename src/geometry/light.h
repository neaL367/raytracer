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
    double su = std::sqrt(u1);
    return t->vert(0) * (1.0 - su) + t->vert(1) * (su * (1.0 - u2)) +
           t->vert(2) * (su * u2);
}

inline vec3 light_normal_at(const light &lt, const vec3 &p, double time) {
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
