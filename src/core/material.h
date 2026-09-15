#pragma once

#include "ray.h"
#include "vec3.h"

struct hit_record; // forward declaration — explained below

class material
{
public:
    virtual ~material() = default;
    virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered) const = 0;
};