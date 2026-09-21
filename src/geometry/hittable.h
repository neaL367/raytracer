#pragma once
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/aabb.h"
#include <memory>
#include <vector>

class material; // fwd: geometry owns shape, material owns scatter

struct hit_record {
    double t = 0;
    vec3 point;
    vec3 normal;
    bool front_face = true;
    std::shared_ptr<material> mat;
    double u = 0, v = 0; // shape UVs for textures (sphere/quad/tri fill)

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
            if (o->hit(r, t_min, closest, tmp)) {
                any = true;
                closest = tmp.t;
                rec = tmp;
            }
        }
        return any;
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
