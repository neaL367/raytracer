#pragma once

#include "ray.h"
#include "vec3.h"
#include "hit_record.h"
#include "random.h"
#include "onb.h"
#include "texture.h"

#include <cmath>
#include <memory>
#include <numbers>

class material
{
public:
    virtual ~material() = default;
    virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered) const = 0;
    // Emitted radiance; black for everything except lights.
    virtual vec3 emitted() const { return vec3(0, 0, 0); }
    // Solid-angle pdf of having sampled direction scattered from rec. Zero
    // default: delta materials have no density to report (see specular()).
    virtual double scattering_pdf(const ray &, const hit_record &, const ray &) const
    {
        return 0.0;
    }
    // True when shading needs surface parametrization (image maps). Lets
    // primitives skip UV math — acos/atan2 per hit — nobody will read.
    virtual bool needs_uv() const { return false; }
    // Diffuse reflectance for denoiser guides (AOVs). Lights report black:
    // emission is not reflectance.
    virtual vec3 surface_albedo(const hit_record &) const { return vec3(0, 0, 0); }
    // True for delta distributions (mirror, glass): no finite sampling
    // density exists, so the direct-light estimator skips them and only the
    // bounce is integrated. Temporary seam — the BRDF/pdf refactor replaces
    // this flag with real pdfs.
    virtual bool specular() const { return false; }
};

class lambertian : public material
{
public:
    lambertian(const vec3 &albedo) : lambertian(std::make_shared<solid_color>(albedo)) {}
    lambertian(std::shared_ptr<texture> albedo) : albedo(albedo) {}

    bool scatter(const ray &, const hit_record &rec, vec3 &attenuation, ray &scattered) const override
    {
        // Cosine-weighted hemisphere sample around the normal. Replaces the
        // old normal-plus-random-sphere hack: that distribution has no clean
        // closed-form pdf, and MIS needs sampler and pdf to agree exactly.
        onb basis(rec.normal);
        vec3 d = random_cosine_direction();
        vec3 direction = basis.local(d.x(), d.y(), d.z());
        scattered = ray(rec.point, direction);
        attenuation = albedo->value(rec.u, rec.v, rec.point);
        return true;
    }

    double scattering_pdf(const ray &, const hit_record &rec, const ray &scattered) const override
    {
        double cos_theta = dot(rec.normal, unit_vector(scattered.direction()));
        return (cos_theta < 0.0) ? 0.0 : cos_theta / std::numbers::pi;
    }

    bool needs_uv() const override { return albedo->needs_uv(); }

    vec3 surface_albedo(const hit_record &rec) const override
    {
        return albedo->value(rec.u, rec.v, rec.point);
    }

    // Read access for GPU upload (SoA material flattening).
    const std::shared_ptr<texture> &tex() const { return albedo; }

private:
    std::shared_ptr<texture> albedo;
};

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

    bool specular() const override { return true; }

    // Read access for GPU upload.
    const vec3 &tint() const { return albedo; }

    vec3 surface_albedo(const hit_record &) const override { return albedo; }

private:
    vec3 albedo;
};

double reflectance(double cosine, double ref_idx)
{
    double r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * std::pow(1 - cosine, 5);
}

class dielectric : public material
{
public:
    dielectric(double refraction_index) : ir(refraction_index) {}

    bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered) const override
    {
        attenuation = vec3(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double())
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.point, direction);
        return true;
    }

    bool specular() const override { return true; }

    // Read access for GPU upload.
    double index() const { return ir; }

    vec3 surface_albedo(const hit_record &) const override { return vec3(1, 1, 1); }

private:
    double ir;
};

class diffuse_light : public material
{
public:
    diffuse_light(const vec3 &emit_color) : emit_color(emit_color) {}

    bool scatter(const ray &, const hit_record &, vec3 &, ray &) const override
    {
        return false; // lights emit; they never bounce
    }

    vec3 emitted() const override { return emit_color; }

    // Read access for GPU upload.
    const vec3 &emission() const { return emit_color; }

private:
    vec3 emit_color;
};