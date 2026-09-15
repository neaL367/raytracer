#pragma once

#include "hittable.h"
#include <vector>

#include <memory>

class hittable_list : public hittable
{
public:
    hittable_list() {}

    size_t size() const { return objects.size(); }
    const std::vector<std::shared_ptr<hittable>> &objects_ref() const { return objects; }

    void add(std::shared_ptr<hittable> object)
    {
        objects.push_back(object);
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override
    {
        hit_record temp_rec;
        bool hit_anything = false;
        double closest_so_far = t_max;

        for (const auto &object : objects)
        {
            if (object->hit(r, t_min, closest_so_far, temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    bool bounding_box(aabb &output_box) const override
    {
        if (objects.empty())
            return false;

        aabb temp_box;
        bool first_box = true;

        for (const auto &object : objects)
        {
            if (!object->bounding_box(temp_box))
                return false;
            output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
            first_box = false;
        }

        return true;
    }

private:
    std::vector<std::shared_ptr<hittable>> objects;
};