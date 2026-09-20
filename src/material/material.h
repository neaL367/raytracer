#pragma once
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/random.h"
#include "../core/sampler.h"
#include "../core/onb.h"
#include "../core/texture.h"
#include "../geometry/hittable.h"
#include <memory>

// Material answers scatter only. No light/traversal knowledge.
// Returns false = ray absorbed (killed, contributes black).
// emitted() default black; diffuse_light overrides (no scatter).
class material {
public:
    virtual ~material() = default;
    virtual bool scatter(const ray &in, const hit_record &rec,
                         vec3 &attenuation, ray &scattered) const = 0;
    virtual vec3 emitted() const { return vec3(0, 0, 0); }
    // NEE applies to diffuse only; specular paths skip explicit lights.
    virtual bool is_diffuse() const { return false; }
};

class lambertian : public material {
public:
    lambertian(const vec3 &a) : tex(std::make_shared<solid_color>(a)) {}
    lambertian(std::shared_ptr<texture> t) : tex(t) {}
    bool scatter(const ray &, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
        // Cosine-weighted: pdf cos/PI cancels f*cos term, throughput *= albedo exact.
        onb frame;
        frame.build_from_w(rec.normal);
        vec3 dir = frame.local(random_cosine_direction());
        if (near_zero(dir))
            dir = rec.normal; // degenerate guard
        scattered = ray(rec.point, dir);
        attenuation = tex->value(rec.u, rec.v, rec.point);
        return true;
    }
    bool is_diffuse() const override { return true; }

private:
    std::shared_ptr<texture> tex;
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

// Pure emitter: never scatters, integrator reads emitted() on hit.
class diffuse_light : public material {
public:
    diffuse_light(const vec3 &c) : emit_color(c) {}
    bool scatter(const ray &, const hit_record &,
                 vec3 &, ray &) const override {
        return false;
    }
    vec3 emitted() const override { return emit_color; }

private:
    vec3 emit_color;
};
