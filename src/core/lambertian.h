#pragma once

#include "material.h"
#include "hit_record.h"
#include "random.h"

class lambertian : public material
{
public:
    lambertian(const vec3 &albedo) : albedo(albedo) {}

    bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered) const override
    {
        vec3 scatter_direction = rec.normal + random_unit_vector();

        if (scatter_direction.length_squared() < 1e-8)
            scatter_direction = rec.normal;

        scattered = ray(rec.point, scatter_direction);
        attenuation = albedo;
        return true;
    }

private:
    vec3 albedo;
};