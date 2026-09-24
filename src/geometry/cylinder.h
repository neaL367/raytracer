#pragma once
#include "hittable.h"
#include <algorithm>
#include <cmath>
#include <memory>

// Finite capped 3D cylinder defined between points p0 and p1 with radius r.
class cylinder : public hittable {
public:
    cylinder() : radius(1.0), length(1.0), capped(true) {}
    cylinder(const vec3 &p0, const vec3 &p1, double r, std::shared_ptr<material> m, bool is_capped = true)
        : base(p0), top(p1), radius(r), capped(is_capped), mat(m) {
        vec3 axis_vec = top - base;
        length = axis_vec.length();
        axis = (length > 1e-8) ? axis_vec / length : vec3(0, 1, 0);

        // Orthonormal basis perpendicular to cylinder axis
        vec3 up = (std::abs(axis.x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        u_axis = unit_vector(cross(axis, up));
        v_axis = cross(axis, u_axis);
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        vec3 dp = r.origin() - base;
        vec3 d_proj = r.direction() - dot(r.direction(), axis) * axis;
        vec3 dp_proj = dp - dot(dp, axis) * axis;

        double a = dot(d_proj, d_proj);
        double half_b = dot(d_proj, dp_proj);
        double c = dot(dp_proj, dp_proj) - radius * radius;
        double disc = half_b * half_b - a * c;

        double nearest_t = t_max + 1.0;
        int hit_type = 0; // 1 = body, 2 = bottom cap, 3 = top cap
        vec3 best_normal;
        double best_h = 0.0;

        // 1. Check curved cylindrical body
        if (disc >= 0.0 && a > 1e-12) {
            double sqrtd = std::sqrt(disc);
            double root = (-half_b - sqrtd) / a;
            if (root > t_min && root < t_max) {
                double h = dot(r.at(root) - base, axis);
                if (h >= 0.0 && h <= length) {
                    nearest_t = root;
                    hit_type = 1;
                    best_h = h;
                }
            }
            if (hit_type == 0) {
                root = (-half_b + sqrtd) / a;
                if (root > t_min && root < t_max) {
                    double h = dot(r.at(root) - base, axis);
                    if (h >= 0.0 && h <= length) {
                        nearest_t = root;
                        hit_type = 1;
                        best_h = h;
                    }
                }
            }
        }

        // 2. Check flat caps if enabled
        if (capped) {
            // Bottom cap (at base, normal = -axis)
            double denom = dot(r.direction(), -axis);
            if (std::abs(denom) > 1e-8) {
                double t_cap = dot(base - r.origin(), -axis) / denom;
                if (t_cap > t_min && t_cap < nearest_t) {
                    vec3 p_cap = r.at(t_cap);
                    if ((p_cap - base).length_squared() <= radius * radius) {
                        nearest_t = t_cap;
                        hit_type = 2;
                        best_normal = -axis;
                    }
                }
            }

            // Top cap (at top, normal = axis)
            denom = dot(r.direction(), axis);
            if (std::abs(denom) > 1e-8) {
                double t_cap = dot(top - r.origin(), axis) / denom;
                if (t_cap > t_min && t_cap < nearest_t) {
                    vec3 p_cap = r.at(t_cap);
                    if ((p_cap - top).length_squared() <= radius * radius) {
                        nearest_t = t_cap;
                        hit_type = 3;
                        best_normal = axis;
                    }
                }
            }
        }

        if (hit_type == 0)
            return false;

        rec.t = nearest_t;
        rec.point = r.at(nearest_t);

        if (hit_type == 1) {
            vec3 radial = rec.point - (base + best_h * axis);
            rec.set_face_normal(r, unit_vector(radial));
            double phi = std::atan2(dot(radial, v_axis), dot(radial, u_axis)) + 3.1415926535897932385;
            rec.u = phi / (2.0 * 3.1415926535897932385);
            rec.v = (length > 1e-8) ? best_h / length : 0.0;
        } else {
            rec.set_face_normal(r, best_normal);
            vec3 cap_p = (hit_type == 2) ? rec.point - base : rec.point - top;
            rec.u = 0.5 + 0.5 * (dot(cap_p, u_axis) / radius);
            rec.v = 0.5 + 0.5 * (dot(cap_p, v_axis) / radius);
        }

        rec.mat = mat;
        rec.hit_obj = nullptr;
        rec.hit_prim = this;
        rec.tangent = u_axis;
        rec.has_tangent = true;

        return true;
    }

    bool bounding_box(aabb &box) const override {
        const double pad = 1e-4;
        // Bounding box encompassing two spherical end caps of radius r at base and top
        vec3 lo(std::min(base.x(), top.x()) - radius - pad,
                std::min(base.y(), top.y()) - radius - pad,
                std::min(base.z(), top.z()) - radius - pad);
        vec3 hi(std::max(base.x(), top.x()) + radius + pad,
                std::max(base.y(), top.y()) + radius + pad,
                std::max(base.z(), top.z()) + radius + pad);
        box = aabb(lo, hi);
        return true;
    }

    const vec3 &get_base() const { return base; }
    const vec3 &get_top() const { return top; }
    double get_radius() const { return radius; }
    double get_length() const { return length; }

private:
    vec3 base;
    vec3 top;
    double radius;
    double length;
    vec3 axis;
    vec3 u_axis, v_axis;
    bool capped;
    std::shared_ptr<material> mat;
};
