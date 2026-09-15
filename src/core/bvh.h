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

    // Splits along the longest axis of this range's bounds, at the median
    // centroid. The median always leaves both sides non-empty, so recursion
    // terminates with no degenerate-split guard. nth_element (not sort) puts
    // the median in place in linear time; order inside each half is
    // unspecified but deterministic for a given input.
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
            is_leaf = true;
            prims.assign(objects.begin() + start, objects.begin() + end);
            box = node_box;
            return;
        }

        vec3 extent = node_box.max() - node_box.min();
        int axis = (extent.x() > extent.y())
                       ? ((extent.x() > extent.z()) ? 0 : 2)
                       : ((extent.y() > extent.z()) ? 1 : 2);

        size_t mid = start + span / 2;
        std::nth_element(objects.begin() + start, objects.begin() + mid, objects.begin() + end,
                         [axis](const std::shared_ptr<hittable> &a, const std::shared_ptr<hittable> &b) {
                             return centroid_axis(a, axis) < centroid_axis(b, axis);
                         });

        // Direct new (not make_shared): only member code can reach the private
        // range constructor below.
        left.reset(new bvh_node(objects, start, mid, leaf_size));
        right.reset(new bvh_node(objects, mid, end, leaf_size));

        box = surrounding_box(left->box, right->box);
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
