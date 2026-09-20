#pragma once
#include "../geometry/hittable.h"
#include "../core/bench_stats.h"
#include <algorithm>
#include <memory>
#include <vector>

// Median-split BVH on the hittable interface. Longest-axis partition,
// leaf cap 4, near-first traversal. Integrator never names it: one-line
// swap list->BVH in main. Binned SAH later rebuilds only this file.
class bvh_node : public hittable {
public:
    bvh_node() {}
    bvh_node(std::vector<std::shared_ptr<hittable>> &objs, size_t start, size_t end) {
        aabb bounds;
        bool first = true;
        for (size_t i = start; i < end; ++i) {
            aabb b;
            if (!objs[i]->bounding_box(b))
                continue; // all current shapes bound; guard anyway
            bounds = first ? b : aabb::surrounding(bounds, b);
            first = false;
        }
        box = bounds;

        size_t span = end - start;
        if (span <= 4) {
            for (size_t i = start; i < end; ++i)
                prims.push_back(objs[i]);
            return;
        }
        int axis = bounds.longest_axis();
        auto cmp = [axis](const std::shared_ptr<hittable> &a,
                          const std::shared_ptr<hittable> &b) {
            aabb ba, bb;
            a->bounding_box(ba);
            b->bounding_box(bb);
            double ca = (ba.minimum.e[axis] + ba.maximum.e[axis]) * 0.5;
            double cb = (bb.minimum.e[axis] + bb.maximum.e[axis]) * 0.5;
            return ca < cb;
        };
        std::sort(objs.begin() + (long)start, objs.begin() + (long)end, cmp);
        size_t mid = start + span / 2;
        left = std::make_shared<bvh_node>(objs, start, mid);
        right = std::make_shared<bvh_node>(objs, mid, end);
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        ::count_box();
        if (!box.hit(r, t_min, t_max))
            return false;
        if (!prims.empty()) {
            hit_record tmp;
            bool any = false;
            double closest = t_max;
            for (const auto &p : prims) {
                ::count_prim();
                if (p->hit(r, t_min, closest, tmp)) {
                    any = true;
                    closest = tmp.t;
                    rec = tmp;
                }
            }
            return any;
        }
        // Near-first: test the child whose box opens earlier.
        bool hl = left && left->box.hit(r, t_min, t_max);
        bool hr = right && right->box.hit(r, t_min, t_max);
        hit_record lrec, rrec;
        bool gl = false, gr = false;
        // Order by entry distance would need slab t; two extra box tests
        // are cheap vs primitive tests, so probe-then-order.
        if (hl && hr) {
            // Both open: go left first, narrow t_max for right.
            gl = left->hit(r, t_min, t_max, lrec);
            double tmax2 = gl ? lrec.t : t_max;
            gr = right->hit(r, t_min, tmax2, rrec);
        } else if (hl) {
            gl = left->hit(r, t_min, t_max, lrec);
        } else if (hr) {
            gr = right->hit(r, t_min, t_max, rrec);
        } else {
            return false;
        }
        if (gl && gr) {
            rec = (lrec.t < rrec.t) ? lrec : rrec;
            return true;
        }
        if (gl) {
            rec = lrec;
            return true;
        }
        if (gr) {
            rec = rrec;
            return true;
        }
        return false;
    }

    bool bounding_box(aabb &b) const override {
        b = box;
        return true;
    }

private:
    aabb box;
    std::shared_ptr<bvh_node> left, right;
    std::vector<std::shared_ptr<hittable>> prims; // non-empty = leaf
};
