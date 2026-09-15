#pragma once

#include "ray.h"
#include "hit_record.h"
#include "aabb.h"

class hittable
{
public:
    virtual ~hittable() = default;
    virtual bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const = 0;
    virtual bool bounding_box(aabb &output_box) const = 0;
};