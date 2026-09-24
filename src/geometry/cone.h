#pragma once
#include "hittable.h"
#include <algorithm>
#include <cmath>
#include <memory>

// Analytical truncated conical frustum between p0 (radius r0) and p1 (radius r1).
class cone : public hittable {
public:
    cone() : r0(1.0), r1(0.0), length(1.0), capped(true) { init_static(); }
    cone(const vec3 &p0, const vec3 &p1, double radius0, double radius1,
         std::shared_ptr<material> m, bool is_capped = true)
        : base(p0), top(p1), r0(radius0), r1(radius1), capped(is_capped), mat(m) {
        vec3 axis_vec = top - base;
        length = axis_vec.length();
        axis = (length > 1e-8) ? axis_vec / length : vec3(0, 1, 0);

        vec3 up = (std::abs(axis.x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        u_axis = unit_vector(cross(axis, up));
        v_axis = cross(axis, u_axis);
        init_static();
    }

    void init_static() {
        dr = (r1 - r0) / (length > 1e-8 ? length : 1.0);
        dr_sq = dr * dr;
        r0_sq = r0 * r0;
        r1_sq = r1 * r1;
        inv_length = (length > 1e-8) ? (1.0 / length) : 0.0;
        inv_r0 = (r0 > 0.0) ? (1.0 / r0) : 1.0;
        inv_r1 = (r1 > 0.0) ? (1.0 / r1) : 1.0;
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        double nearest_t = t_max + 1.0;
        int hit_type = 0; // 1 = conical mantle, 2 = base cap, 3 = top cap
        double best_s = 0.0;
        vec3 best_normal;

        vec3 dp = r.origin() - base;

        // Decompose ray into axial and radial components
        double d_dot_a = dot(r.direction(), axis);
        double dp_dot_a = dot(dp, axis);
        vec3 d_rad = r.direction() - d_dot_a * axis;
        vec3 dp_rad = dp - dp_dot_a * axis;

        // Quadratic coefficients for conical surface: ||P_rad||^2 = (r0 + dr * s)^2
        double a = dot(d_rad, d_rad) - dr_sq * d_dot_a * d_dot_a;
        double half_b = dot(d_rad, dp_rad) - dr * (r0 + dr * dp_dot_a) * d_dot_a;
        double c = dot(dp_rad, dp_rad) - (r0 + dr * dp_dot_a) * (r0 + dr * dp_dot_a);
        double disc = half_b * half_b - a * c;

        if (disc >= 0.0 && std::abs(a) > 1e-12) {
            double sqrtd = std::sqrt(disc);
            double root = (-half_b - sqrtd) / a;
            if (root > t_min && root < t_max) {
                double s = dp_dot_a + root * d_dot_a;
                if (s >= 0.0 && s <= length) {
                    nearest_t = root;
                    hit_type = 1;
                    best_s = s;
                }
            }
            if (hit_type == 0) {
                root = (-half_b + sqrtd) / a;
                if (root > t_min && root < t_max) {
                    double s = dp_dot_a + root * d_dot_a;
                    if (s >= 0.0 && s <= length) {
                        nearest_t = root;
                        hit_type = 1;
                        best_s = s;
                    }
                }
            }
        }

        // Check planar end caps if enabled
        if (capped) {
            // Base cap at s = 0 (normal = -axis)
            if (r0 > 1e-6) {
                double denom = dot(r.direction(), -axis);
                if (std::abs(denom) > 1e-8) {
                    double t_cap = dot(base - r.origin(), -axis) / denom;
                    if (t_cap > t_min && t_cap < nearest_t) {
                        vec3 p = r.at(t_cap);
                        if ((p - base).length_squared() <= r0 * r0) {
                            nearest_t = t_cap;
                            hit_type = 2;
                            best_normal = -axis;
                        }
                    }
                }
            }
            // Top cap at s = length (normal = axis)
            if (r1 > 1e-6) {
                double denom = dot(r.direction(), axis);
                if (std::abs(denom) > 1e-8) {
                    double t_cap = dot(top - r.origin(), axis) / denom;
                    if (t_cap > t_min && t_cap < nearest_t) {
                        vec3 p = r.at(t_cap);
                        if ((p - top).length_squared() <= r1 * r1) {
                            nearest_t = t_cap;
                            hit_type = 3;
                            best_normal = axis;
                        }
                    }
                }
            }
        }

        if (hit_type == 0)
            return false;

        rec.t = nearest_t;
        rec.point = r.at(nearest_t);

        if (hit_type == 1) {
            vec3 axis_pt = base + best_s * axis;
            vec3 radial = rec.point - axis_pt;
            double r_at_s = r0 + dr * best_s;
            vec3 outward_rad = (r_at_s > 1e-6) ? radial / r_at_s : u_axis;
            // Surface normal slanted by dr
            vec3 norm = unit_vector(outward_rad - dr * axis);
            rec.set_face_normal(r, norm);
            double phi = std::atan2(dot(radial, v_axis), dot(radial, u_axis)) + 3.1415926535897932385;
            rec.u = phi / (2.0 * 3.1415926535897932385);
            rec.v = best_s * inv_length;
        } else {
            rec.set_face_normal(r, best_normal);
            vec3 cap_p = (hit_type == 2) ? rec.point - base : rec.point - top;
            double inv_cap_r = (hit_type == 2) ? inv_r0 : inv_r1;
            rec.u = 0.5 + 0.5 * (dot(cap_p, u_axis) * inv_cap_r);
            rec.v = 0.5 + 0.5 * (dot(cap_p, v_axis) * inv_cap_r);
        }

        rec.mat = mat;
        rec.hit_obj = nullptr;
        rec.hit_prim = this;
        rec.tangent = u_axis;
        rec.has_tangent = true;

        return true;
    }

    bool hit_any(const ray &r, double t_min, double t_max) const override {
        vec3 dp = r.origin() - base;
        double d_dot_a = dot(r.direction(), axis);
        double dp_dot_a = dot(dp, axis);
        vec3 d_rad = r.direction() - d_dot_a * axis;
        vec3 dp_rad = dp - dp_dot_a * axis;

        double a = dot(d_rad, d_rad) - dr_sq * d_dot_a * d_dot_a;
        double half_b = dot(d_rad, dp_rad) - dr * (r0 + dr * dp_dot_a) * d_dot_a;
        double c = dot(dp_rad, dp_rad) - (r0 + dr * dp_dot_a) * (r0 + dr * dp_dot_a);
        double disc = half_b * half_b - a * c;

        if (disc >= 0.0 && std::abs(a) > 1e-12) {
            double sqrtd = std::sqrt(disc);
            double root = (-half_b - sqrtd) / a;
            if (root >= t_min && root <= t_max) {
                double s = dp_dot_a + root * d_dot_a;
                if (s >= 0.0 && s <= length)
                    return true;
            }
            root = (-half_b + sqrtd) / a;
            if (root >= t_min && root <= t_max) {
                double s = dp_dot_a + root * d_dot_a;
                if (s >= 0.0 && s <= length)
                    return true;
            }
        }

        if (capped) {
            double denom = dot(r.direction(), axis);
            if (std::abs(denom) > 1e-8) {
                double t_base = dot(base - r.origin(), axis) / denom;
                if (t_base >= t_min && t_base <= t_max) {
                    if ((r.at(t_base) - base).length_squared() <= r0_sq)
                        return true;
                }
                double t_top = dot(top - r.origin(), axis) / denom;
                if (t_top >= t_min && t_top <= t_max) {
                    if ((r.at(t_top) - top).length_squared() <= r1_sq)
                        return true;
                }
            }
        }
        return false;
    }

    bool bounding_box(aabb &box) const override {
        const double pad = 1e-4;
        double max_r = std::max(r0, r1);
        vec3 lo(std::min(base.x(), top.x()) - max_r - pad,
                std::min(base.y(), top.y()) - max_r - pad,
                std::min(base.z(), top.z()) - max_r - pad);
        vec3 hi(std::max(base.x(), top.x()) + max_r + pad,
                std::max(base.y(), top.y()) + max_r + pad,
                std::max(base.z(), top.z()) + max_r + pad);
        box = aabb(lo, hi);
        return true;
    }

    const vec3 &get_base() const { return base; }
    const vec3 &get_top() const { return top; }
    double get_r0() const { return r0; }
    double get_r1() const { return r1; }
    double get_length() const { return length; }
    std::shared_ptr<material> mat_ptr() const { return mat; }

private:
    vec3 base, top;
    double r0, r1;
    double length;
    vec3 axis;
    vec3 u_axis, v_axis;
    bool capped;
    std::shared_ptr<material> mat;
    double dr = 0.0;
    double dr_sq = 0.0;
    double r0_sq = 1.0;
    double r1_sq = 0.0;
    double inv_length = 1.0;
    double inv_r0 = 1.0;
    double inv_r1 = 1.0;
};
