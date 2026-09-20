#pragma once
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/random.h"
#include "../geometry/hittable.h"
#include <memory>

// Material answers scatter only. No light/traversal knowledge.
// Returns false = ray absorbed (killed, contributes black).
class material {
public:
    virtual ~material() = default;
    virtual bool scatter(const ray &in, const hit_record &rec,
                         vec3 &attenuation, ray &scattered) const = 0;
};

class lambertian : public material {
public:
    lambertian(const vec3 &a) : albedo(a) {}
    bool scatter(const ray &, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
        vec3 dir = rec.normal + random_unit_vector();
        if (near_zero(dir))
            dir = rec.normal; // degenerate guard
        scattered = ray(rec.point, dir);
        attenuation = albedo;
        return true;
    }

private:
    vec3 albedo;
};

class metal : public material {
public:
    metal(const vec3 &a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}
    bool scatter(const ray &in, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
        vec3 refl = reflect(unit_vector(in.direction()), rec.normal);
        scattered = ray(rec.point, refl + fuzz * random_in_unit_sphere());
        attenuation = albedo;
        return dot(scattered.direction(), rec.normal) > 0;
    }

private:
    vec3 albedo;
    double fuzz;
};

class dielectric : public material {
public:
    dielectric(double ri) : ir(ri) {}
    bool scatter(const ray &in, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
        attenuation = vec3(1, 1, 1); // glass absorbs nothing
        double ratio = rec.front_face ? (1.0 / ir) : ir;
        vec3 unit = unit_vector(in.direction());
        double cos_t = fmin(dot(-unit, rec.normal), 1.0);
        double sin_t = std::sqrt(1.0 - cos_t * cos_t);
        bool cannot_refract = ratio * sin_t > 1.0;
        vec3 dir = (cannot_refract || reflectance(cos_t, ratio) > random_double())
                       ? reflect(unit, rec.normal)
                       : refract(unit, rec.normal, ratio);
        scattered = ray(rec.point, dir);
        return true;
    }

private:
    double ir;
    // Schlick approx: grazing -> mirror, normal -> ~4% for glass.
    static double reflectance(double cos, double ref_idx) {
        double r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow(1 - cos, 5);
    }
};
