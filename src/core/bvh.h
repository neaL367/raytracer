#pragma once
#include "hittable.h"
#include "hittable_list.h"
#include "aabb.h"
#include "random.h"

#include <algorithm>
#include <vector>
#include <iostream>

class bvh_node : public hittable // bounding volume hierarchy node: a spatial data structure that organizes hittable objects for efficient ray intersection tests
{
public:
    bvh_node(const hittable_list &list) : bvh_node(list.objects_ref(), 0, list.objects_ref().size()) {}

    bvh_node(const std::vector<std::shared_ptr<hittable>> &src_objects, size_t start, size_t end)
    {
        auto objects = src_objects; // mutable copy — we're about to sort it

        int axis = static_cast<int>(random_double(0, 3));
        auto comparator = (axis == 0)   ? box_x_compare
                          : (axis == 1) ? box_y_compare
                                        : box_z_compare;

        size_t object_span = end - start;

        if (object_span == 1)
        {
            left = right = objects[start];
        }
        else if (object_span == 2)
        {
            if (comparator(objects[start], objects[start + 1]))
            {
                left = objects[start];
                right = objects[start + 1];
            }
            else
            {
                left = objects[start + 1];
                right = objects[start];
            }
        }
        else
        {
            std::sort(objects.begin() + start, objects.begin() + end, comparator);

            size_t mid = start + object_span / 2;
            left = std::make_shared<bvh_node>(objects, start, mid);
            right = std::make_shared<bvh_node>(objects, mid, end);
        }

        aabb box_left, box_right;

        if (!left->bounding_box(box_left) || !right->bounding_box(box_right))
        {
            std::cerr << "No bounding box in bvh_node constructor.\n";
        }

        box = surrounding_box(box_left, box_right);
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override
    {
        if (!box.hit(r, t_min, t_max))
            return false;

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
    // Uses dynamic_cast to tell inner nodes from leaf primitives; that is
    // fine here because a census runs once per build, never per ray. The
    // single-object case stores the same child on both sides, so it is
    // counted exactly once.
    void census(size_t &nodes, size_t &leaves, size_t &max_depth, size_t depth = 0) const
    {
        ++nodes;
        if (depth > max_depth)
            max_depth = depth;
        if (left == right)
        {
            ++leaves;
            return;
        }
        if (const auto *ln = dynamic_cast<const bvh_node *>(left.get()))
            ln->census(nodes, leaves, max_depth, depth + 1);
        else
            ++leaves;
        if (const auto *rn = dynamic_cast<const bvh_node *>(right.get()))
            rn->census(nodes, leaves, max_depth, depth + 1);
        else
            ++leaves;
    }

private:
    std::shared_ptr<hittable> left;
    std::shared_ptr<hittable> right;
    aabb box;

    static bool box_compare(const std::shared_ptr<hittable> &a, const std::shared_ptr<hittable> &b, int axis)
    {
        aabb box_a, box_b;
        if (!a->bounding_box(box_a) || !b->bounding_box(box_b))
            std::cerr << "No bounding box in bvh_node constructor.\n";

        double a_val = (axis == 0) ? box_a.min().x() : (axis == 1) ? box_a.min().y()
                                                                   : box_a.min().z();
        double b_val = (axis == 0) ? box_b.min().x() : (axis == 1) ? box_b.min().y()
                                                                   : box_b.min().z();
        return a_val < b_val;
    }

    static bool box_x_compare(const std::shared_ptr<hittable> &a, const std::shared_ptr<hittable> &b)
    {
        return box_compare(a, b, 0);
    }
    static bool box_y_compare(const std::shared_ptr<hittable> &a, const std::shared_ptr<hittable> &b)
    {
        return box_compare(a, b, 1);
    }
    static bool box_z_compare(const std::shared_ptr<hittable> &a, const std::shared_ptr<hittable> &b)
    {
        return box_compare(a, b, 2);
    }
};