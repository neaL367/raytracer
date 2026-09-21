#pragma once
// Geometric instances (NTW Ch.7 pattern): move the ray, never the geometry.
// translate shifts the ray origin; rotate_y spins ray about Y by -theta and
// spins the hit point/normal back by +theta. Boxes are built axis-aligned at
// the origin, then rotated, then translated — the canonical Cornell pose.
// GPU flatten bakes instances to world-space quads (collect_baked_quads),
// so shaders never learn about this module.
#include "hittable.h"
#include "quad.h"

#include <cmath>
#include <memory>
#include <vector>

class translate : public hittable {
public:
    translate(std::shared_ptr<hittable> inner, const vec3 &offset)
        : obj(inner), off(offset) {}

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        ray moved(r.origin() - off, r.direction(), r.time());
        if (!obj->hit(moved, t_min, t_max, rec))
            return false;
        rec.point = rec.point + off;
        return true; // normals rotation-free: translation preserves them
    }

    bool bounding_box(aabb &box) const override {
        aabb inner;
        if (!obj->bounding_box(inner))
            return false;
        box = aabb(inner.minimum + off, inner.maximum + off);
        return true;
    }

    const std::shared_ptr<hittable> &inner_ref() const { return obj; }
    const vec3 &offset() const { return off; }

private:
    std::shared_ptr<hittable> obj;
    vec3 off;
};

class rotate_y : public hittable {
public:
    rotate_y(std::shared_ptr<hittable> inner, double angle_degrees)
        : obj(inner), theta(angle_degrees * 3.1415926535897932385 / 180.0),
          sin_t(std::sin(angle_degrees * 3.1415926535897932385 / 180.0)),
          cos_t(std::cos(angle_degrees * 3.1415926535897932385 / 180.0)) {}

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        // Inverse-rotate the ray (-theta), hit local space, rotate back (+theta).
        vec3 o = r.origin(), d = r.direction();
        vec3 o_loc(cos_t * o.x() - sin_t * o.z(), o.y(),
                   sin_t * o.x() + cos_t * o.z());
        vec3 d_loc(cos_t * d.x() - sin_t * d.z(), d.y(),
                   sin_t * d.x() + cos_t * d.z());
        ray local(o_loc, d_loc, r.time());
        if (!obj->hit(local, t_min, t_max, rec))
            return false;
        vec3 p = rec.point, n = rec.normal;
        rec.point = vec3(cos_t * p.x() + sin_t * p.z(), p.y(),
                         -sin_t * p.x() + cos_t * p.z());
        rec.normal = vec3(cos_t * n.x() + sin_t * n.z(), n.y(),
                          -sin_t * n.x() + cos_t * n.z());
        return true;
    }

    bool bounding_box(aabb &box) const override {
        aabb inner;
        if (!obj->bounding_box(inner))
            return false;
        // Rotate all 8 corners: tight bound of the swept volume.
        vec3 lo(1e30, 1e30, 1e30), hi(-1e30, -1e30, -1e30);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int k = 0; k < 2; ++k) {
                    double x = i ? inner.maximum.x() : inner.minimum.x();
                    double y = j ? inner.maximum.y() : inner.minimum.y();
                    double z = k ? inner.maximum.z() : inner.minimum.z();
                    double rx = cos_t * x + sin_t * z;
                    double rz = -sin_t * x + cos_t * z;
                    lo = vec3(fmin(lo.x(), rx), fmin(lo.y(), y), fmin(lo.z(), rz));
                    hi = vec3(fmax(hi.x(), rx), fmax(hi.y(), y), fmax(hi.z(), rz));
                }
        const double pad = 1e-4;
        box = aabb(lo - vec3(pad, pad, pad), hi + vec3(pad, pad, pad));
        return true;
    }

    const std::shared_ptr<hittable> &inner_ref() const { return obj; }
    double sin_theta() const { return sin_t; }
    double cos_theta() const { return cos_t; }

private:
    std::shared_ptr<hittable> obj;
    double theta, sin_t, cos_t;
};

// Rigid bake for the GPU path: flatten instances into world-space quads.
// Accumulator is world = R(c,s) * p + T. Returns false on non-quad leaves
// (volumes, spheres) — flatten fails loudly.
namespace instance_detail {

inline vec3 rot_point(const vec3 &p, double c, double s) {
    return vec3(c * p.x() + s * p.z(), p.y(), -s * p.x() + c * p.z());
}
inline vec3 rot_dir(const vec3 &d, double c, double s) {
    return vec3(c * d.x() + s * d.z(), d.y(), -s * d.x() + c * d.z());
}

inline bool collect_baked_quads(const std::shared_ptr<hittable> &o, double c,
                                double s, const vec3 &T, std::vector<quad> &out) {
    if (auto q = std::dynamic_pointer_cast<quad>(o)) {
        // Q is a point: rotate then translate. u/v are directions: rotate only.
        vec3 Qw = rot_point(q->corner(), c, s) + T;
        vec3 uw = rot_dir(q->edge_u(), c, s);
        vec3 vw = rot_dir(q->edge_v(), c, s);
        out.emplace_back(Qw, uw, vw, q->mat_ptr());
        return true;
    }
    if (auto list = std::dynamic_pointer_cast<hittable_list>(o)) {
        for (const auto &child : list->children())
            if (!collect_baked_quads(child, c, s, T, out))
                return false;
        return true;
    }
    if (auto tr = std::dynamic_pointer_cast<translate>(o)) {
        vec3 T2 = rot_point(tr->offset(), c, s) + T;
        return collect_baked_quads(tr->inner_ref(), c, s, T2, out);
    }
    if (auto ry = std::dynamic_pointer_cast<rotate_y>(o)) {
        double ci = ry->cos_theta(), si = ry->sin_theta();
        double c2 = c * ci - s * si;
        double s2 = s * ci + c * si;
        return collect_baked_quads(ry->inner_ref(), c2, s2, T, out);
    }
    return false;
}

} // namespace instance_detail
