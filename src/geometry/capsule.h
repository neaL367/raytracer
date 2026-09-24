#pragma once
#include "hittable.h"
#include <algorithm>
#include <cmath>
#include <memory>

// Analytical 3D capsule primitive: segment from a to b with spherical end caps of radius r.
class capsule : public hittable {
public:
    capsule() : radius(0.5), length(1.0) { init_static(); }
    capsule(const vec3 &p0, const vec3 &p1, double r, std::shared_ptr<material> m)
        : a(p0), b(p1), radius(r), mat(m) {
        vec3 axis_vec = b - a;
        length = axis_vec.length();
        axis = (length > 1e-8) ? axis_vec / length : vec3(0, 1, 0);

        vec3 up = (std::abs(axis.x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        u_axis = unit_vector(cross(axis, up));
        v_axis = cross(axis, u_axis);
        init_static();
    }

    void init_static() {
        radius_sq = radius * radius;
        inv_radius = (radius != 0.0) ? (1.0 / radius) : 0.0;
        double total_len = length + 2.0 * radius;
        inv_total_len = (total_len > 1e-8) ? (1.0 / total_len) : 0.0;
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        double nearest_t = t_max + 1.0;
        int hit_type = 0; // 1 = cylinder body, 2 = sphere cap A, 3 = sphere cap B
        double best_s = 0.0;

        // 1. Test infinite cylinder section
        vec3 dp = r.origin() - a;
        vec3 d_proj = r.direction() - dot(r.direction(), axis) * axis;
        vec3 dp_proj = dp - dot(dp, axis) * axis;

        double c_a = dot(d_proj, d_proj);
        double half_b = dot(d_proj, dp_proj);
        double c_c = dot(dp_proj, dp_proj) - radius_sq;
        double disc = half_b * half_b - c_a * c_c;

        if (disc >= 0.0 && c_a > 1e-12) {
            double sqrtd = std::sqrt(disc);
            double root = (-half_b - sqrtd) / c_a;
            if (root > t_min && root < t_max) {
                double s = dot(r.at(root) - a, axis);
                if (s >= 0.0 && s <= length) {
                    nearest_t = root;
                    hit_type = 1;
                    best_s = s;
                }
            }
            if (hit_type == 0) {
                root = (-half_b + sqrtd) / c_a;
                if (root > t_min && root < t_max) {
                    double s = dot(r.at(root) - a, axis);
                    if (s >= 0.0 && s <= length) {
                        nearest_t = root;
                        hit_type = 1;
                        best_s = s;
                    }
                }
            }
        }

        // 2. Test Sphere Cap A (at point a)
        vec3 oc_a = r.origin() - a;
        double sa_a = dot(r.direction(), r.direction());
        double sa_hb = dot(oc_a, r.direction());
        double sa_c = dot(oc_a, oc_a) - radius_sq;
        double sa_disc = sa_hb * sa_hb - sa_a * sa_c;
        if (sa_disc >= 0.0) {
            double sqrtd = std::sqrt(sa_disc);
            double root = (-sa_hb - sqrtd) / sa_a;
            if (root > t_min && root < nearest_t) {
                vec3 p = r.at(root);
                if (dot(p - a, axis) <= 0.0) {
                    nearest_t = root;
                    hit_type = 2;
                    best_s = 0.0;
                }
            }
            if (hit_type != 2) {
                root = (-sa_hb + sqrtd) / sa_a;
                if (root > t_min && root < nearest_t) {
                    vec3 p = r.at(root);
                    if (dot(p - a, axis) <= 0.0) {
                        nearest_t = root;
                        hit_type = 2;
                        best_s = 0.0;
                    }
                }
            }
        }

        // 3. Test Sphere Cap B (at point b)
        vec3 oc_b = r.origin() - b;
        double sb_hb = dot(oc_b, r.direction());
        double sb_c = dot(oc_b, oc_b) - radius_sq;
        double sb_disc = sb_hb * sb_hb - sa_a * sb_c;
        if (sb_disc >= 0.0) {
            double sqrtd = std::sqrt(sb_disc);
            double root = (-sb_hb - sqrtd) / sa_a;
            if (root > t_min && root < nearest_t) {
                vec3 p = r.at(root);
                if (dot(p - b, axis) >= 0.0) {
                    nearest_t = root;
                    hit_type = 3;
                    best_s = length;
                }
            }
            if (hit_type != 3) {
                root = (-sb_hb + sqrtd) / sa_a;
                if (root > t_min && root < nearest_t) {
                    vec3 p = r.at(root);
                    if (dot(p - b, axis) >= 0.0) {
                        nearest_t = root;
                        hit_type = 3;
                        best_s = length;
                    }
                }
            }
        }

        if (hit_type == 0)
            return false;

        rec.t = nearest_t;
        rec.point = r.at(nearest_t);

        vec3 center_on_axis = a + best_s * axis;
        vec3 normal_vec = unit_vector(rec.point - center_on_axis);
        rec.set_face_normal(r, normal_vec);

        double phi = std::atan2(dot(normal_vec, v_axis), dot(normal_vec, u_axis)) + 3.1415926535897932385;
        rec.u = phi / (2.0 * 3.1415926535897932385);
        rec.v = (best_s + radius) * inv_total_len;
        rec.mat = mat;
        rec.hit_obj = nullptr;
        rec.hit_prim = this;
        rec.tangent = u_axis;
        rec.has_tangent = true;

        return true;
    }

    bool hit_any(const ray &r, double t_min, double t_max) const override {
        // 1. Test infinite cylinder section
        vec3 dp = r.origin() - a;
        vec3 d_proj = r.direction() - dot(r.direction(), axis) * axis;
        vec3 dp_proj = dp - dot(dp, axis) * axis;

        double a_cyl = d_proj.length_squared();
        double half_b = dot(d_proj, dp_proj);
        double c_cyl = dp_proj.length_squared() - radius_sq;
        double disc = half_b * half_b - a_cyl * c_cyl;

        if (disc >= 0.0 && a_cyl > 1e-12) {
            double sqrtd = std::sqrt(disc);
            double root = (-half_b - sqrtd) / a_cyl;
            if (root >= t_min && root <= t_max) {
                double s = dot(r.at(root) - a, axis);
                if (s >= 0.0 && s <= length)
                    return true;
            }
            root = (-half_b + sqrtd) / a_cyl;
            if (root >= t_min && root <= t_max) {
                double s = dot(r.at(root) - a, axis);
                if (s >= 0.0 && s <= length)
                    return true;
            }
        }

        // 2. Test spherical end cap at a
        vec3 oc_a = r.origin() - a;
        double a_ray = r.direction().length_squared();
        double hb_a = dot(oc_a, r.direction());
        double c_a = oc_a.length_squared() - radius_sq;
        double disc_a = hb_a * hb_a - a_ray * c_a;
        if (disc_a >= 0.0 && a_ray > 1e-12) {
            double sqrtd_a = std::sqrt(disc_a);
            double root = (-hb_a - sqrtd_a) / a_ray;
            if (root >= t_min && root <= t_max) {
                if (dot(r.at(root) - a, axis) < 0.0)
                    return true;
            }
            root = (-hb_a + sqrtd_a) / a_ray;
            if (root >= t_min && root <= t_max) {
                if (dot(r.at(root) - a, axis) < 0.0)
                    return true;
            }
        }

        // 3. Test spherical end cap at b
        vec3 oc_b = r.origin() - b;
        double hb_b = dot(oc_b, r.direction());
        double c_b = oc_b.length_squared() - radius_sq;
        double disc_b = hb_b * hb_b - a_ray * c_b;
        if (disc_b >= 0.0 && a_ray > 1e-12) {
            double sqrtd_b = std::sqrt(disc_b);
            double root = (-hb_b - sqrtd_b) / a_ray;
            if (root >= t_min && root <= t_max) {
                if (dot(r.at(root) - b, axis) > 0.0)
                    return true;
            }
            root = (-hb_b + sqrtd_b) / a_ray;
            if (root >= t_min && root <= t_max) {
                if (dot(r.at(root) - b, axis) > 0.0)
                    return true;
            }
        }
        return false;
    }

    bool bounding_box(aabb &box) const override {
        const double pad = 1e-4;
        vec3 lo(std::min(a.x(), b.x()) - radius - pad,
                std::min(a.y(), b.y()) - radius - pad,
                std::min(a.z(), b.z()) - radius - pad);
        vec3 hi(std::max(a.x(), b.x()) + radius + pad,
                std::max(a.y(), b.y()) + radius + pad,
                std::max(a.z(), b.z()) + radius + pad);
        box = aabb(lo, hi);
        return true;
    }

    const vec3 &get_a() const { return a; }
    const vec3 &get_b() const { return b; }
    double get_radius() const { return radius; }
    double get_length() const { return length; }
    std::shared_ptr<material> mat_ptr() const { return mat; }

private:
    vec3 a, b;
    double radius;
    double length;
    vec3 axis;
    vec3 u_axis, v_axis;
    std::shared_ptr<material> mat;
    double radius_sq = 0.25;
    double inv_radius = 2.0;
    double inv_total_len = 0.5;
};
