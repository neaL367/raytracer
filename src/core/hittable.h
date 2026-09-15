#pragma once
#include "ray.h"
#include "hit_record.h"

class hittable
{
public:
    virtual ~hittable() = default;
    virtual bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const = 0;
};