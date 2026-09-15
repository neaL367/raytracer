#pragma once
#include "hittable.h"
#include "hittable_list.h"
#include "aabb.h"

#include <algorithm>
#include <vector>
#include <iostream>

class bvh_node : public hittable // hierarchy of bounding boxes; inner nodes route rays, leaves hold the primitives
{
public:
    // Copies the list's pointers once, then partitions that single vector in
    // place. Recursion works on index ranges, so building costs comparisons
    // only — no per-node allocation the way a copy-per-level build does.
    bvh_node(const hittable_list &list, size_t max_leaf_size = 2)
        : leaf_size(max_leaf_size == 0 ? 1 : max_leaf_size)
    {
        auto objects = list.objects_ref();
        build(objects, 0, objects.size());
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override
    {
        if (!box.hit(r, t_min, t_max))
            return false;

        if (is_leaf)
        {
            // Same closest-hit scan as a flat list, but over a handful of
            // primitives instead of the whole scene.
            hit_record temp_rec;
            bool hit_anything = false;
            double closest_so_far = t_max;
            for (const auto &prim : prims)
            {
                if (prim->hit(r, t_min, closest_so_far, temp_rec))
                {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }
            return hit_anything;
        }

        bool hit_left = left->hit(r, t_min, t_max, rec);
        bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

        return hit_left || hit_right;
    }

    bool bounding_box(aabb &output_box) const override
    {
        output_box = box;
        return true;
    }

    // Count nodes, leaves, and maximum depth of a built tree (for reporting).
    // Runs once per build, never per ray.
    void census(size_t &nodes, size_t &leaves, size_t &max_depth, size_t depth = 0) const
    {
        ++nodes;
        if (depth > max_depth)
            max_depth = depth;
        if (is_leaf)
        {
            ++leaves;
            return;
        }
        left->census(nodes, leaves, max_depth, depth + 1);
        right->census(nodes, leaves, max_depth, depth + 1);
    }

private:
    // Child constructor: shares the root's vector by reference, owns nothing.
    bvh_node(std::vector<std::shared_ptr<hittable>> &objects, size_t start, size_t end, size_t max_leaf)
        : leaf_size(max_leaf)
    {
        build(objects, start, end);
    }

    // Splits at the cheapest binned-SAH position, falling back to a median
    // split when binning finds nothing usable (tiny or degenerate ranges).
    // SAH models expected cost as: one box test plus, for each side, the
    // primitive count weighted by that side's surface area relative to the
    // parent — a ray that hits the parent hits a child with probability
    // proportional to the child's area. A leaf holding every primitive costs
    // span, so any split costing more than that is rejected in favor of a
    // leaf. Binning (12 bins over centroid bounds, swept once per axis)
    // approximates the optimal split for the price of one linear pass.
    void build(std::vector<std::shared_ptr<hittable>> &objects, size_t start, size_t end)
    {
        size_t span = end - start;
        if (span == 0)
        {
            is_leaf = true;
            return;
        }

        aabb node_box;
        bool first = true;
        for (size_t i = start; i < end; ++i)
        {
            aabb b;
            if (!objects[i]->bounding_box(b))
                std::cerr << "No bounding box in bvh_node build.\n";
            node_box = first ? b : surrounding_box(node_box, b);
            first = false;
        }

        if (span <= leaf_size)
        {
            make_leaf(objects, start, end, node_box);
            return;
        }

        size_t mid = sah_split(objects, start, end, node_box, span);
        if (mid == start || mid == end)
            mid = median_split(objects, start, end, node_box);

        // Direct new (not make_shared): only member code can reach the private
        // range constructor below.
        left.reset(new bvh_node(objects, start, mid, leaf_size));
        right.reset(new bvh_node(objects, mid, end, leaf_size));

        box = surrounding_box(left->box, right->box);
    }

    void make_leaf(std::vector<std::shared_ptr<hittable>> &objects, size_t start, size_t end,
                   const aabb &node_box)
    {
        is_leaf = true;
        prims.assign(objects.begin() + start, objects.begin() + end);
        box = node_box;
    }

    // Median split along the range's longest axis. Always leaves both sides
    // non-empty, so recursion terminates. nth_element (not sort) puts the
    // median in place in linear time; order inside each half is unspecified
    // but deterministic for a given input.
    static size_t median_split(std::vector<std::shared_ptr<hittable>> &objects, size_t start, size_t end,
                               const aabb &node_box)
    {
        vec3 extent = node_box.max() - node_box.min();
        int axis = (extent.x() > extent.y())
                       ? ((extent.x() > extent.z()) ? 0 : 2)
                       : ((extent.y() > extent.z()) ? 1 : 2);

        size_t mid = start + (end - start) / 2;
        std::nth_element(objects.begin() + start, objects.begin() + mid, objects.begin() + end,
                         [axis](const std::shared_ptr<hittable> &a, const std::shared_ptr<hittable> &b) {
                             return centroid_axis(a, axis) < centroid_axis(b, axis);
                         });
        return mid;
    }

    // Cheapest split over 12 centroid bins on each axis, or start when none
    // pays (caller falls back to median). Never returns the edges: candidate
    // splits with an empty side are skipped, and a partition that still lands
    // everything on one side (identical centroids) is rejected the same way.
    static size_t sah_split(std::vector<std::shared_ptr<hittable>> &objects, size_t start, size_t end,
                            const aabb &node_box, size_t span)
    {
        constexpr int bins = 12;
        const double parent_area = node_box.surface_area();
        const double leaf_cost = static_cast<double>(span);
        const double traversal_cost = 1.0;

        int best_axis = -1;
        double best_pos = 0.0;
        double best_cost = leaf_cost;

        for (int axis = 0; axis < 3; ++axis)
        {
            // Centroid bounds, not node bounds: bins must span where the
            // primitives' centers are, or every centroid lands in one bin.
            double cmin = centroid_axis(objects[start], axis);
            double cmax = cmin;
            for (size_t i = start + 1; i < end; ++i)
            {
                double c = centroid_axis(objects[i], axis);
                if (c < cmin)
                    cmin = c;
                if (c > cmax)
                    cmax = c;
            }
            if (cmax - cmin < 1e-12 || parent_area < 1e-12)
                continue;

            aabb bin_box[bins];
            size_t bin_count[bins] = {};
            for (size_t i = start; i < end; ++i)
            {
                double c = centroid_axis(objects[i], axis);
                int b = static_cast<int>(bins * (c - cmin) / (cmax - cmin));
                if (b == bins)
                    b = bins - 1;
                aabb prim_box;
                objects[i]->bounding_box(prim_box);
                bin_box[b] = (bin_count[b] == 0) ? prim_box : surrounding_box(bin_box[b], prim_box);
                ++bin_count[b];
            }

            // Prefix/suffix accumulate box + count so each of the 11 split
            // positions is evaluated in O(1) from its neighbors.
            aabb left_box[bins];
            size_t left_count[bins] = {};
            {
                aabb acc;
                size_t n = 0;
                for (int i = 0; i < bins; ++i)
                {
                    if (bin_count[i] > 0)
                        acc = (n == 0) ? bin_box[i] : surrounding_box(acc, bin_box[i]);
                    n += bin_count[i];
                    left_box[i] = acc;
                    left_count[i] = n;
                }
            }

            aabb right_box[bins];
            size_t right_count[bins] = {};
            {
                aabb racc;
                size_t n = 0;
                for (int i = bins - 1; i >= 0; --i)
                {
                    if (bin_count[i] > 0)
                        racc = (n == 0) ? bin_box[i] : surrounding_box(racc, bin_box[i]);
                    n += bin_count[i];
                    right_box[i] = racc;
                    right_count[i] = n;
                }
            }

            for (int i = 0; i < bins - 1; ++i)
            {
                if (left_count[i] == 0 || right_count[i + 1] == 0)
                    continue;
                double cost = traversal_cost + (left_count[i] * left_box[i].surface_area() +
                                                right_count[i + 1] * right_box[i + 1].surface_area()) /
                                                   parent_area;
                if (cost < best_cost)
                {
                    best_cost = cost;
                    best_axis = axis;
                    best_pos = cmin + (cmax - cmin) * (i + 1) / bins;
                }
            }
        }

        if (best_axis < 0)
            return start;

        auto mid_it = std::partition(objects.begin() + start, objects.begin() + end,
                                     [best_axis, best_pos](const std::shared_ptr<hittable> &h) {
                                         return centroid_axis(h, best_axis) < best_pos;
                                     });
        size_t mid = static_cast<size_t>(mid_it - objects.begin());
        return (mid == start || mid == end) ? start : mid;
    }

    // Mid-point between a primitive's box corners on one axis. Centroids
    // split mixed-size scenes (tiny spheres next to a huge ground sphere)
    // far better than box corners: a giant box's corner can sit anywhere.
    static double centroid_axis(const std::shared_ptr<hittable> &h, int axis)
    {
        aabb b;
        if (!h->bounding_box(b))
            std::cerr << "No bounding box in bvh_node build.\n";
        vec3 c = (b.min() + b.max()) * 0.5;
        return (axis == 0) ? c.x() : (axis == 1) ? c.y() : c.z();
    }

    std::shared_ptr<bvh_node> left;
    std::shared_ptr<bvh_node> right;
    std::vector<std::shared_ptr<hittable>> prims; // leaves only
    aabb box;
    size_t leaf_size;
    bool is_leaf = false;
};
