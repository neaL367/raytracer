#pragma once
#include "vec3.h"
#include <memory>
#include <cmath>

// Texture: color as a function of surface position. Materials sample this
// instead of holding a flat albedo, so one lambertian covers solids,
// checkers, and (next) image maps. u/v come from the hit record; p is the
// 3D hit point for procedural patterns that ignore parametrization.
class texture
{
public:
    virtual ~texture() = default;
    virtual vec3 value(double u, double v, const vec3 &p) const = 0;
    // True for parametrized maps (image textures); procedural patterns work
    // from the 3D point alone and cost no UV computation.
    virtual bool needs_uv() const { return false; }
};

class solid_color : public texture
{
public:
    solid_color(const vec3 &albedo) : albedo(albedo) {}

    vec3 value(double u, double v, const vec3 &p) const override
    {
        return albedo;
    }

private:
    vec3 albedo;
};

class checker_texture : public texture
{
public:
    checker_texture(double scale, std::shared_ptr<texture> even, std::shared_ptr<texture> odd)
        : inv_scale(1.0 / scale), even(even), odd(odd) {}

    checker_texture(double scale, const vec3 &c1, const vec3 &c2)
        : checker_texture(scale, std::make_shared<solid_color>(c1), std::make_shared<solid_color>(c2)) {}

    vec3 value(double u, double v, const vec3 &p) const override
    {
        // 3D checker: product of sines flips sign per cell along every axis,
        // so cells alternate in all three dimensions, not just on a plane.
        double sines = std::sin(inv_scale * p.x()) * std::sin(inv_scale * p.y()) * std::sin(inv_scale * p.z());
        return (sines < 0.0) ? odd->value(u, v, p) : even->value(u, v, p);
    }

private:
    double inv_scale;
    std::shared_ptr<texture> even;
    std::shared_ptr<texture> odd;
};
