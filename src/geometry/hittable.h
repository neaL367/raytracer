#pragma once
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/aabb.h"
#include <memory>
#include <vector>

class material; // fwd: geometry owns shape, material owns scatter
class hittable; // fwd: hit_record links the owning shape (volumes set it)

struct hit_record {
    double t = 0;
    vec3 point;
    vec3 normal;
    bool front_face = true;
    std::shared_ptr<material> mat;
    double u = 0, v = 0; // shape UVs for textures (sphere/quad/tri fill)
    // Shading tangent for anisotropy (shapes with UVs fill it; default off).
    vec3 tangent{0, 0, 0};
    bool has_tangent = false;
    // Geometric (flat) face normal for smooth meshes (M72). Protects against
    // shading-normal inconsistency where specular reflections plunge into mesh interiors.
    vec3 geo_normal{0, 0, 0};
    bool has_geo_normal = false;
    // Owning shape for volume transmittance march (media set this, shapes
    // leave null). Raw pointer: lifetime owned by the scene, never stored.
    const hittable *hit_obj = nullptr;
    // Hit prim identity for nesting (M57): shapes stamp `this`, media leave
    // whatever the boundary set. Fresh per query (traversals overwrite rec
    // wholesale on hit); never consulted after a miss.
    const hittable *hit_prim = nullptr;
    // Nesting interface ratio (M57): the integrator resolves the medium
    // stack before scatter() and stamps eta here; dielectric uses it when
    // set, else legacy front_face behavior (unit tests hit the legacy path).
    double nest_eta = 0;
    bool nest_set = false;

    // Outward vs inward decided by ray dir. Glass needs true normal side.
    inline void set_face_normal(const ray &r, const vec3 &outward) {
        front_face = dot(r.direction(), outward) < 0;
        normal = front_face ? outward : -outward;
    }
};

// Shape contract. Only interface renderer knows. BVH later same base.
class hittable {
public:
    virtual ~hittable() = default;
    virtual bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const = 0;
    virtual bool hit_any(const ray &r, double t_min, double t_max) const {
        hit_record rec;
        return hit(r, t_min, t_max, rec);
    }
    virtual bool bounding_box(aabb &box) const = 0;
};

// Flat list, closest-hit scan. O(N) flagged: BVH replaces in M5.
class hittable_list : public hittable {
public:
    void add(std::shared_ptr<hittable> o) { objects.push_back(o); }
    void clear() { objects.clear(); }
    // Instance baking (GPU flatten) walks box children without RTTI soup.
    const std::vector<std::shared_ptr<hittable>> &children() const { return objects; }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        hit_record tmp;
        bool any = false;
        double closest = t_max;
        for (const auto &o : objects) {
            tmp.hit_obj = nullptr; // stale medium tags must not survive
                                   // across children sharing tmp (M48)
            if (o->hit(r, t_min, closest, tmp)) {
                any = true;
                closest = tmp.t;
                rec = tmp;
            }
        }
        return any;
    }

    bool hit_any(const ray &r, double t_min, double t_max) const override {
        for (const auto &o : objects) {
            if (o->hit_any(r, t_min, t_max))
                return true;
        }
        return false;
    }

    bool bounding_box(aabb &box) const override {
        if (objects.empty())
            return false;
        aabb tmp;
        bool first = true;
        for (const auto &o : objects) {
            aabb b;
            if (!o->bounding_box(b))
                return false;
            tmp = first ? b : aabb::surrounding(tmp, b);
            first = false;
        }
        box = tmp;
        return true;
    }

private:
    std::vector<std::shared_ptr<hittable>> objects;
};
