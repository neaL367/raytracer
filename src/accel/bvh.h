#pragma once
#include "../geometry/hittable.h"
#include "../core/bench_stats.h"
#include <algorithm>
#include <memory>
#include <vector>

// BVH on the hittable interface. Binned SAH by default (16 bins/axis,
// Ct=1 box, Ci=1 prim, leaf cap 4); median split kept for A/B via flag.
// Integrator never names it. Zero interface change for future rebuilds.
class bvh_node : public hittable {
public:
    bvh_node() {}
    bvh_node(std::vector<std::shared_ptr<hittable>> &objs, size_t start, size_t end,
             bool use_sah = true) {
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
        if (use_sah && try_sah_split(objs, start, end, bounds))
            return;
        median_split(objs, start, end, bounds, use_sah);
    }
    explicit bvh_node(const hittable_list &list, bool use_sah = true) {
        auto objs = list.children();
        *this = bvh_node(objs, 0, objs.size(), use_sah);
    }

    bool is_leaf() const { return !prims.empty(); }
    // GPU flatten accessors (read-only view of the built tree).
    const aabb &node_box() const { return box; }
    const std::shared_ptr<bvh_node> &child(bool to_right) const {
        return to_right ? right : left;
    }
    const std::vector<std::shared_ptr<hittable>> &leaf_prims() const { return prims; }
    size_t count_prims() const {
        if (is_leaf())
            return prims.size();
        size_t n = 0;
        if (left)
            n += left->count_prims();
        if (right)
            n += right->count_prims();
        return n;
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
                tmp.hit_obj = nullptr; // stale medium tags (M48)
                if (p->hit(r, t_min, closest, tmp)) {
                    any = true;
                    closest = tmp.t;
                    rec = tmp;
                }
            }
            return any;
        }
        // Near-first: test the child whose box opens earlier.
        double entry_l = t_min, entry_r = t_min;
        bool hl = left && left->box.hit_entry(r, t_min, t_max, entry_l);
        bool hr = right && right->box.hit_entry(r, t_min, t_max, entry_r);
        hit_record lrec, rrec;
        bool gl = false, gr = false;
        // Entries come from the slab probes above; only two scalar compares
        // order the children. Ties stay left-first, preserving legacy order.
        if (hl && hr) {
            if (entry_l <= entry_r) {
                // Near child first, narrow t_max for the far child.
                gl = left->hit(r, t_min, t_max, lrec);
                double tmax2 = gl ? lrec.t : t_max;
                gr = right->hit(r, t_min, tmax2, rrec);
            } else {
                gr = right->hit(r, t_min, t_max, rrec);
                double tmax2 = gr ? rrec.t : t_max;
                gl = left->hit(r, t_min, tmax2, lrec);
            }
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

    bool hit_any(const ray &r, double t_min, double t_max) const override {
        ::count_box();
        if (!box.hit(r, t_min, t_max))
            return false;
        if (!prims.empty()) {
            for (const auto &p : prims) {
                ::count_prim();
                if (p->hit_any(r, t_min, t_max))
                    return true;
            }
            return false;
        }
        if (left && left->hit_any(r, t_min, t_max))
            return true;
        if (right && right->hit_any(r, t_min, t_max))
            return true;
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

    static double centroid(const std::shared_ptr<hittable> &o, int axis) {
        aabb b;
        o->bounding_box(b);
        return (b.minimum.e[axis] + b.maximum.e[axis]) * 0.5;
    }

    void median_split(std::vector<std::shared_ptr<hittable>> &objs, size_t start,
                      size_t end, const aabb &bounds, bool sub_sah) {
        int axis = bounds.longest_axis();
        std::sort(objs.begin() + (long)start, objs.begin() + (long)end,
                  [axis](const std::shared_ptr<hittable> &a,
                         const std::shared_ptr<hittable> &b) {
                      aabb ba, bb;
                      a->bounding_box(ba);
                      b->bounding_box(bb);
                      double ca = (ba.minimum.e[axis] + ba.maximum.e[axis]) * 0.5;
                      double cb = (bb.minimum.e[axis] + bb.maximum.e[axis]) * 0.5;
                      return ca < cb;
                  });
        size_t mid = start + (end - start) / 2;
        left = std::make_shared<bvh_node>(objs, start, mid, sub_sah);
        right = std::make_shared<bvh_node>(objs, mid, end, sub_sah);
    }

    // Binned SAH: 16 bins/axis over centroid bounds, sweep both sides,
    // cheapest split wins if it beats leaf cost. Returns false when no
    // split pays (caller falls back to median).
    bool try_sah_split(std::vector<std::shared_ptr<hittable>> &objs, size_t start,
                       size_t end, const aabb &bounds) {
        const int BINS = 16;
        const double Ct = 1.0, Ci = 1.0;
        double span = (double)(end - start);
        double leaf_cost = Ci * span;
        double node_area = bounds.surface_area();
        if (node_area <= 0)
            return false;

        // Centroid bounds: zero extent on every axis = degenerate.
        vec3 cmin(1e30, 1e30, 1e30), cmax(-1e30, -1e30, -1e30);
        for (size_t i = start; i < end; ++i) {
            aabb b;
            objs[i]->bounding_box(b);
            vec3 c = (b.minimum + b.maximum) * 0.5;
            for (int a = 0; a < 3; ++a) {
                if (c.e[a] < cmin.e[a])
                    cmin.e[a] = c.e[a];
                if (c.e[a] > cmax.e[a])
                    cmax.e[a] = c.e[a];
            }
        }

        int best_axis = -1, best_bin = -1;
        double best_cost = leaf_cost;
        for (int axis = 0; axis < 3; ++axis) {
            double lo = cmin.e[axis], hi = cmax.e[axis];
            if (hi - lo < 1e-12)
                continue; // all centroids coincide on this axis
            int count[BINS] = {};
            aabb bnds[BINS];
            bool used[BINS] = {};
            for (size_t i = start; i < end; ++i) {
                aabb b;
                objs[i]->bounding_box(b);
                double c = ((b.minimum.e[axis] + b.maximum.e[axis]) * 0.5 - lo) / (hi - lo);
                int bin = (int)(c * BINS);
                if (bin < 0)
                    bin = 0;
                if (bin >= BINS)
                    bin = BINS - 1;
                if (!used[bin]) {
                    bnds[bin] = b;
                    used[bin] = true;
                } else {
                    bnds[bin] = aabb::surrounding(bnds[bin], b);
                }
                count[bin]++;
            }
            // Prefix/suffix sweeps: bounds + counts each side of each cut.
            aabb left_box[BINS];
            int left_n[BINS] = {};
            aabb acc;
            int n = 0;
            bool have = false;
            for (int b = 0; b < BINS; ++b) {
                if (used[b]) {
                    acc = have ? aabb::surrounding(acc, bnds[b]) : bnds[b];
                    have = true;
                    n += count[b];
                }
                left_box[b] = acc;
                left_n[b] = n;
            }
            aabb right_box[BINS];
            int right_n[BINS] = {};
            have = false;
            n = 0;
            for (int b = BINS - 1; b >= 0; --b) {
                if (used[b]) {
                    acc = have ? aabb::surrounding(acc, bnds[b]) : bnds[b];
                    have = true;
                    n += count[b];
                }
                right_box[b] = acc;
                right_n[b] = n;
            }
            for (int b = 0; b < BINS - 1; ++b) {
                if (left_n[b] == 0 || right_n[b + 1] == 0)
                    continue;
                double cost = Ct + Ci * (left_box[b].surface_area() * left_n[b] +
                                         right_box[b + 1].surface_area() * right_n[b + 1]) /
                                        node_area;
                if (cost < best_cost) {
                    best_cost = cost;
                    best_axis = axis;
                    best_bin = b;
                }
            }
        }
        if (best_axis < 0)
            return false; // no paying split: median fallback

        // Partition in place by centroid vs winning bin boundary.
        double lo = cmin.e[best_axis], hi = cmax.e[best_axis];
        auto mid_it = std::partition(
            objs.begin() + (long)start, objs.begin() + (long)end,
            [&](const std::shared_ptr<hittable> &o) {
                double c = (centroid(o, best_axis) - lo) / (hi - lo);
                int bin = (int)(c * BINS);
                if (bin >= BINS)
                    bin = BINS - 1;
                return bin <= best_bin;
            });
        size_t mid = (size_t)(mid_it - objs.begin());
        if (mid == start || mid == end)
            return false; // degenerate partition: median fallback
        left = std::make_shared<bvh_node>(objs, start, mid, true);
        right = std::make_shared<bvh_node>(objs, mid, end, true);
        return true;
    }
};
