#pragma once
#include "hittable.h"
#include "../core/random.h"
#include <cmath>
#include <memory>

// Planar circular disk / annulus primitive defined by center, normal, and outer/inner radius.
class disk : public hittable {
public:
    disk() : radius(1.0), inner_radius(0.0) {}
    disk(const vec3 &c, const vec3 &n, double r, std::shared_ptr<material> m, double r_in = 0.0)
        : center(c), normal(unit_vector(n)), radius(r), inner_radius(r_in), mat(m) {
        // Construct orthonormal tangent frame for UV mapping and sampling
        vec3 up = (std::abs(normal.x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        u_axis = unit_vector(cross(normal, up));
        v_axis = cross(normal, u_axis);
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        double denom = dot(normal, r.direction());
        if (std::abs(denom) < 1e-8)
            return false;

        double t = dot(center - r.origin(), normal) / denom;
        if (t < t_min || t > t_max)
            return false;

        vec3 p = r.at(t);
        vec3 v = p - center;
        double dist2 = v.length_squared();
        if (dist2 > radius * radius || dist2 < inner_radius * inner_radius)
            return false;

        rec.t = t;
        rec.point = p;
        rec.set_face_normal(r, normal);
        rec.mat = mat;
        rec.hit_obj = nullptr;
        rec.hit_prim = this;

        // Polar UV coordinates
        double dist = std::sqrt(dist2);
        double phi = std::atan2(dot(v, v_axis), dot(v, u_axis)) + 3.1415926535897932385;
        rec.u = phi / (2.0 * 3.1415926535897932385);
        rec.v = (radius > inner_radius) ? (dist - inner_radius) / (radius - inner_radius) : 0.0;
        rec.tangent = u_axis;
        rec.has_tangent = true;

        return true;
    }

    bool bounding_box(aabb &box) const override {
        const double pad = 1e-4;
        // Exact half-extent of planar disk projected onto axis i: R * sqrt(1 - n_i^2)
        double ex = radius * std::sqrt(std::max(0.0, 1.0 - normal.x() * normal.x())) + pad;
        double ey = radius * std::sqrt(std::max(0.0, 1.0 - normal.y() * normal.y())) + pad;
        double ez = radius * std::sqrt(std::max(0.0, 1.0 - normal.z() * normal.z())) + pad;

        box = aabb(vec3(center.x() - ex, center.y() - ey, center.z() - ez),
                   vec3(center.x() + ex, center.y() + ey, center.z() + ez));
        return true;
    }

    double area() const {
        return 3.1415926535897932385 * (radius * radius - inner_radius * inner_radius);
    }

    vec3 sample_point() const {
        double r1 = random_double();
        double r2 = random_double();
        double r = std::sqrt(inner_radius * inner_radius + r1 * (radius * radius - inner_radius * inner_radius));
        double theta = 2.0 * 3.1415926535897932385 * r2;
        return center + r * (std::cos(theta) * u_axis + std::sin(theta) * v_axis);
    }

    vec3 light_normal() const { return normal; }
    std::shared_ptr<material> mat_ptr() const { return mat; }

    const vec3 &get_center() const { return center; }
    const vec3 &get_normal() const { return normal; }
    double get_radius() const { return radius; }
    double get_inner_radius() const { return inner_radius; }

private:
    vec3 center;
    vec3 normal;
    double radius;
    double inner_radius;
    std::shared_ptr<material> mat;
    vec3 u_axis, v_axis;
};
