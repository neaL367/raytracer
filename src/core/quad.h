#pragma once
#include "hittable.h"
#include "vec3.h"
#include "material.h"
#include "random.h"
#include <memory>

// Quadrilateral: origin corner Q plus edge vectors u and v. Hit test solves
// the ray-plane intersection, then checks planar barycentric coordinates via
// the precomputed w vector (RTOW quad formulation). Sampling draws uniform
// points for direct-light estimation; pdf_value converts that area measure
// to a solid-angle measure from the shading point.
class quad : public hittable
{
public:
    quad(const vec3 &Q, const vec3 &u, const vec3 &v, std::shared_ptr<material> mat)
        : Q(Q), u(u), v(v), mat(mat)
    {
        vec3 n = cross(u, v);
        normal_vec = unit_vector(n);
        area_val = n.length();
        w_vec = n / n.length_squared();
        set_bounding_box();
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override
    {
        double denom = dot(normal_vec, r.direction());
        if (std::fabs(denom) < 1e-8)
            return false;

        double t = (plane_d - dot(normal_vec, r.origin())) / denom;
        if (t < t_min || t > t_max)
            return false;

        vec3 p = r.at(t);
        vec3 planar = p - Q;
        double alpha = dot(w_vec, cross(planar, v));
        double beta = dot(w_vec, cross(u, planar));
        if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0)
            return false;

        rec.t = t;
        rec.point = p;
        rec.set_face_normal(r, normal_vec);
        rec.mat = mat;
        return true;
    }

    bool bounding_box(aabb &output_box) const override
    {
        output_box = bbox;
        return true;
    }

    // Uniform point on the surface, for direct-light sampling.
    vec3 sample() const
    {
        return Q + random_double() * u + random_double() * v;
    }

    // Solid-angle pdf of sampling direction dir from origin: area pdf
    // (1/area) converted by dist^2/cos. Zero when the ray misses the quad.
    double pdf_value(const vec3 &origin, const vec3 &direction) const
    {
        hit_record rec;
        if (!hit(ray(origin, direction), 0.001, 1000.0, rec))
            return 0.0;
        double dist2 = (rec.point - origin).length_squared();
        double cos_light = std::fabs(dot(direction, normal_vec));
        if (cos_light < 1e-8)
            return 0.0;
        return dist2 / (cos_light * area_val);
    }

    const vec3 &normal() const { return normal_vec; }
    const std::shared_ptr<material> &mat_ptr() const { return mat; }
    double area() const { return area_val; }

private:
    void set_bounding_box()
    {
        vec3 p0 = Q, p1 = Q + u, p2 = Q + v, p3 = Q + u + v;
        vec3 small(std::fmin(std::fmin(p0.x(), p1.x()), std::fmin(p2.x(), p3.x())),
                   std::fmin(std::fmin(p0.y(), p1.y()), std::fmin(p2.y(), p3.y())),
                   std::fmin(std::fmin(p0.z(), p1.z()), std::fmin(p2.z(), p3.z())));
        vec3 big(std::fmax(std::fmax(p0.x(), p1.x()), std::fmax(p2.x(), p3.x())),
                 std::fmax(std::fmax(p0.y(), p1.y()), std::fmax(p2.y(), p3.y())),
                 std::fmax(std::fmax(p0.z(), p1.z()), std::fmax(p2.z(), p3.z())));
        const double padding = 0.0001;
        vec3 pad(padding, padding, padding);
        bbox = aabb(small - pad, big + pad);
        plane_d = dot(normal_vec, Q);
    }

    vec3 Q, u, v;
    std::shared_ptr<material> mat;
    vec3 normal_vec;
    vec3 w_vec;
    aabb bbox;
    double area_val;
    double plane_d;
};
