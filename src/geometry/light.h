#pragma once
// Unified area-light view over quad/sphere/triangle/disk/cylinder/capsule/cone for NEE.
// Same math every shape: uniform point (exactly 2 RNG draws), geometric
// area, geometric normal; pdf_area = d^2/(n*A*cosA). Sphere honors
// center(time), so moving lights sample correctly. Sampling pdfs are
// geometric even for smooth-shaded tris (MIS reverse uses these too).
#include "hittable.h"
#include "quad.h"
#include "sphere.h"
#include "triangle.h"
#include "disk.h"
#include "cylinder.h"
#include "capsule.h"
#include "cone.h"
#include "../material/material.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

struct light {
    light() {}
    light(const std::shared_ptr<quad> &q) : shape(q) {}
    light(const std::shared_ptr<sphere> &s) : shape(s) {}
    light(const std::shared_ptr<triangle> &t) : shape(t) {}
    light(const std::shared_ptr<disk> &d) : shape(d) {}
    light(const std::shared_ptr<cylinder> &c) : shape(c) {}
    light(const std::shared_ptr<capsule> &cap) : shape(cap) {}
    light(const std::shared_ptr<cone> &cn) : shape(cn) {}
    std::shared_ptr<hittable> shape;
};

inline vec3 light_point(const light &lt, double u1, double u2, double time) {
    if (auto q = std::dynamic_pointer_cast<quad>(lt.shape))
        return q->corner() + u1 * q->edge_u() + u2 * q->edge_v();
    if (auto d = std::dynamic_pointer_cast<disk>(lt.shape))
        return d->sample_point(u1, u2);
    if (auto s = std::dynamic_pointer_cast<sphere>(lt.shape)) {
        // Exact uniform on sphere (no rejection, fixed draw count).
        double z = 1.0 - 2.0 * u1;
        double phi = 2.0 * 3.1415926535897932385 * u2;
        double r = std::sqrt(std::max(1.0 - z * z, 0.0));
        vec3 off(r * std::cos(phi), r * std::sin(phi), z);
        return s->center(time) + s->radius_val() * off;
    }
    if (auto cyl = std::dynamic_pointer_cast<cylinder>(lt.shape)) {
        vec3 ax = cyl->get_top() - cyl->get_base();
        double len = ax.length();
        vec3 ax_u = (len > 1e-8) ? ax / len : vec3(0, 1, 0);
        vec3 up = (std::abs(ax_u.x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        vec3 u_ax = unit_vector(cross(ax_u, up));
        vec3 v_ax = cross(ax_u, u_ax);
        double theta = 2.0 * 3.1415926535897932385 * u2;
        vec3 rad = std::cos(theta) * u_ax + std::sin(theta) * v_ax;
        return cyl->get_base() + (u1 * len) * ax_u + cyl->get_radius() * rad;
    }
    if (auto cap = std::dynamic_pointer_cast<capsule>(lt.shape)) {
        vec3 ax = cap->get_b() - cap->get_a();
        double len = ax.length();
        double r = cap->get_radius();
        double a_cyl = 2.0 * 3.1415926535897932385 * r * len;
        double a_sph = 4.0 * 3.1415926535897932385 * r * r;
        double frac_cyl = a_cyl / (a_cyl + a_sph + 1e-8);
        if (u1 < frac_cyl) {
            double u1_cyl = u1 / frac_cyl;
            vec3 ax_u = (len > 1e-8) ? ax / len : vec3(0, 1, 0);
            vec3 up = (std::abs(ax_u.x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
            vec3 u_ax = unit_vector(cross(ax_u, up));
            vec3 v_ax = cross(ax_u, u_ax);
            double theta = 2.0 * 3.1415926535897932385 * u2;
            vec3 rad = std::cos(theta) * u_ax + std::sin(theta) * v_ax;
            return cap->get_a() + (u1_cyl * len) * ax_u + r * rad;
        } else {
            double u1_sph = (u1 - frac_cyl) / (1.0 - frac_cyl);
            double z = 1.0 - 2.0 * u1_sph;
            double phi = 2.0 * 3.1415926535897932385 * u2;
            double r_xy = std::sqrt(std::max(0.0, 1.0 - z * z));
            vec3 off(r_xy * std::cos(phi), r_xy * std::sin(phi), z);
            vec3 center = (z >= 0.0) ? cap->get_b() : cap->get_a();
            return center + r * off;
        }
    }
    if (auto cn = std::dynamic_pointer_cast<cone>(lt.shape)) {
        vec3 ax = cn->get_top() - cn->get_base();
        double len = ax.length();
        vec3 ax_u = (len > 1e-8) ? ax / len : vec3(0, 1, 0);
        vec3 up = (std::abs(ax_u.x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        vec3 u_ax = unit_vector(cross(ax_u, up));
        vec3 v_ax = cross(ax_u, u_ax);
        double r0 = cn->get_r0(), r1 = cn->get_r1();
        double s = u1;
        if (std::abs(r1 - r0) > 1e-6) {
            double term = r0 * r0 + u1 * (r1 * r1 - r0 * r0);
            s = (std::sqrt(std::max(0.0, term)) - r0) / (r1 - r0);
        }
        double r = r0 + s * (r1 - r0);
        double theta = 2.0 * 3.1415926535897932385 * u2;
        vec3 rad = std::cos(theta) * u_ax + std::sin(theta) * v_ax;
        return cn->get_base() + (s * len) * ax_u + r * rad;
    }
    auto t = std::dynamic_pointer_cast<triangle>(lt.shape);
    // Sqrt-barycentric: uniform over the triangle, fixed draw count.
    // Moving tris sample lerped verts (mirrors the hit path).
    double su = std::sqrt(u1);
    return t->vert_at(0, time) * (1.0 - su) + t->vert_at(1, time) * (su * (1.0 - u2)) +
           t->vert_at(2, time) * (su * u2);
}

inline vec3 light_normal_at(const light &lt, const vec3 &p, double time) {
    if (auto q = std::dynamic_pointer_cast<quad>(lt.shape))
        return q->light_normal();
    if (auto d = std::dynamic_pointer_cast<disk>(lt.shape))
        return d->light_normal();
    if (auto s = std::dynamic_pointer_cast<sphere>(lt.shape))
        return unit_vector(p - s->center(time));
    if (auto cyl = std::dynamic_pointer_cast<cylinder>(lt.shape)) {
        vec3 ax = cyl->get_top() - cyl->get_base();
        double len = ax.length();
        vec3 ax_u = (len > 1e-8) ? ax / len : vec3(0, 1, 0);
        vec3 to_p = p - cyl->get_base();
        vec3 radial = to_p - dot(to_p, ax_u) * ax_u;
        return unit_vector(radial);
    }
    if (auto cap = std::dynamic_pointer_cast<capsule>(lt.shape)) {
        vec3 ax = cap->get_b() - cap->get_a();
        double len = ax.length();
        vec3 ax_u = (len > 1e-8) ? ax / len : vec3(0, 1, 0);
        double s = dot(p - cap->get_a(), ax_u);
        if (s <= 0.0)
            return unit_vector(p - cap->get_a());
        if (s >= len)
            return unit_vector(p - cap->get_b());
        vec3 axis_pt = cap->get_a() + s * ax_u;
        return unit_vector(p - axis_pt);
    }
    if (auto cn = std::dynamic_pointer_cast<cone>(lt.shape)) {
        vec3 ax = cn->get_top() - cn->get_base();
        double len = ax.length();
        vec3 ax_u = (len > 1e-8) ? ax / len : vec3(0, 1, 0);
        double s = dot(p - cn->get_base(), ax_u);
        vec3 axis_pt = cn->get_base() + s * ax_u;
        vec3 rad_dir = unit_vector(p - axis_pt);
        double dr = (cn->get_r1() - cn->get_r0()) / (len > 1e-8 ? len : 1.0);
        return unit_vector(rad_dir - dr * ax_u);
    }
    auto t = std::dynamic_pointer_cast<triangle>(lt.shape);
    vec3 e1 = t->vert(1) - t->vert(0), e2 = t->vert(2) - t->vert(0);
    return unit_vector(cross(e1, e2));
}

inline double light_area(const light &lt) {
    if (auto q = std::dynamic_pointer_cast<quad>(lt.shape))
        return q->area();
    if (auto d = std::dynamic_pointer_cast<disk>(lt.shape))
        return d->area();
    if (auto s = std::dynamic_pointer_cast<sphere>(lt.shape)) {
        double r = s->radius_val();
        return 4.0 * 3.1415926535897932385 * r * r;
    }
    if (auto cyl = std::dynamic_pointer_cast<cylinder>(lt.shape)) {
        double r = cyl->get_radius();
        double l = (cyl->get_top() - cyl->get_base()).length();
        return 2.0 * 3.1415926535897932385 * r * l;
    }
    if (auto cap = std::dynamic_pointer_cast<capsule>(lt.shape)) {
        double r = cap->get_radius();
        double l = (cap->get_b() - cap->get_a()).length();
        return 2.0 * 3.1415926535897932385 * r * l + 4.0 * 3.1415926535897932385 * r * r;
    }
    if (auto cn = std::dynamic_pointer_cast<cone>(lt.shape)) {
        double r0 = cn->get_r0(), r1 = cn->get_r1();
        double l = (cn->get_top() - cn->get_base()).length();
        double slant = std::sqrt((r1 - r0) * (r1 - r0) + l * l);
        return 3.1415926535897932385 * (r0 + r1) * slant;
    }
    auto t = std::dynamic_pointer_cast<triangle>(lt.shape);
    vec3 e1 = t->vert(1) - t->vert(0), e2 = t->vert(2) - t->vert(0);
    return 0.5 * cross(e1, e2).length();
}

inline std::shared_ptr<material> light_mat(const light &lt) {
    if (auto q = std::dynamic_pointer_cast<quad>(lt.shape))
        return q->mat_ptr();
    if (auto d = std::dynamic_pointer_cast<disk>(lt.shape))
        return d->mat_ptr();
    if (auto s = std::dynamic_pointer_cast<sphere>(lt.shape))
        return s->mat_ptr();
    if (auto cyl = std::dynamic_pointer_cast<cylinder>(lt.shape))
        return cyl->mat_ptr();
    if (auto cap = std::dynamic_pointer_cast<capsule>(lt.shape))
        return cap->mat_ptr();
    if (auto cn = std::dynamic_pointer_cast<cone>(lt.shape))
        return cn->mat_ptr();
    return std::dynamic_pointer_cast<triangle>(lt.shape)->mat_ptr();
}

// UVs for an NEE light sample drawn with (u1, u2): the same coordinates
// the shape hit path would report for that point, so textured emission
// matches direct camera hits.
inline bool light_uv(const light &lt, double u1, double u2, double time, double &u,
                     double &v) {
    if (std::dynamic_pointer_cast<quad>(lt.shape)) {
        u = u1;
        v = u2;
        return true;
    }
    if (std::dynamic_pointer_cast<disk>(lt.shape)) {
        u = u2;
        v = std::sqrt(u1);
        return true;
    }
    if (std::dynamic_pointer_cast<cylinder>(lt.shape) ||
        std::dynamic_pointer_cast<capsule>(lt.shape) ||
        std::dynamic_pointer_cast<cone>(lt.shape)) {
        u = u2;
        v = u1;
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

// Power approx for importance picking: area * luminance at center sample.
// Single NEE strategy: forward picks and reverse densities both use it.
inline double light_power_approx(const light &lt) {
    double area = light_area(lt);
    if (area <= 0)
        return 0.0;
    vec3 lp = light_point(lt, 0.5, 0.5, 0.0);
    double lu = 0.5, lv = 0.5;
    light_uv(lt, 0.5, 0.5, 0.0, lu, lv);
    vec3 Le = light_emission(lt, lp, lu, lv, 1.0);
    double lum = 0.2126 * Le.x() + 0.7152 * Le.y() + 0.0722 * Le.z();
    return lum > 0 ? area * lum : 0.0;
}

// CDF over light powers (cumulative, last entry = total). Falls back to
// uniform (equal weights) when total power is zero.
inline void build_light_cdf(const std::vector<light> &lights, std::vector<double> &cdf,
                             double &total) {
    cdf.assign(lights.size(), 0.0);
    total = 0.0;
    for (size_t i = 0; i < lights.size(); ++i) {
        total += light_power_approx(lights[i]);
        cdf[i] = total;
    }
    if (total <= 0) {
        for (size_t i = 0; i < lights.size(); ++i)
            cdf[i] = (double)(i + 1);
        total = (double)lights.size();
    }
}

// Pick light index by power CDF. Returns index + pick probability.
inline size_t pick_light_power(const std::vector<double> &cdf, double total, double u,
                                double &pick_p) {
    size_t n = cdf.size();
    double x = u * total;
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (cdf[mid] < x)
            lo = mid + 1;
        else
            hi = mid;
    }
    size_t idx = lo < n ? lo : n - 1;
    double prev = idx > 0 ? cdf[idx - 1] : 0.0;
    pick_p = (cdf[idx] - prev) / total;
    if (pick_p <= 0)
        pick_p = 1.0 / (double)n;
    return idx;
}
