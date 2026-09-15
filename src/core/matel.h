#pragma once

#include "material.h"
#include "hit_record.h"
#include "random.h"

class metal : public material
{
public:
    metal(const vec3 &albedo) : albedo(albedo) {}

    bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered) const override
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.point, reflected);
        attenuation = albedo;
        return dot(scattered.direction(), rec.normal) > 0;
    }

private:
    vec3 albedo;
};