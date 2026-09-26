#pragma once
#include "material_base.h"

// Isotropic volume scatter: uniform sphere direction, albedo attenuates.
// Not diffuse (no NEE into volumes: direct sampling of media flagged).
class isotropic : public material {
public:
    isotropic(const vec3 &a) : albedo(a) {}
    bool scatter(const ray &, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
        vec3 dir = random_unit_vector();
        if (near_zero(dir))
            dir = rec.normal; // degenerate guard (fixed 0.5 draws hit exact zero)
        scattered = ray(rec.point, dir);
        attenuation = albedo;
        return true;
    }
    // Uniform sphere: every direction has density 1/4PI, so found lights
    // MIS-weight instead of counting full (they used to, as "specular").
    double direction_pdf(const vec3 &, const hit_record &) const override {
        return 1.0 / (4.0 * 3.1415926535897932385);
    }
    bool is_volume() const override { return true; }
    vec3 surface_albedo(const hit_record &) const override { return albedo; }

private:
    vec3 albedo;
};
