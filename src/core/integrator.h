#pragma once
#include "vec3.h"
#include "ray.h"
#include "hittable.h"
#include "quad.h"
#include "random.h"
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>

// Integrator: turns rays into radiance. Geometry answers "what did I hit",
// materials answer "how does it scatter" — only this class knows what an
// image means: emission plus estimated incoming light, recursed to a depth
// limit. Alternative strategies (direct lighting only, bidirectional, GPU
// wavefront) implement the same interface over the same scene.
class integrator
{
public:
    virtual ~integrator() = default;
    virtual vec3 Li(const ray &r, const hittable &world,
                    const std::vector<std::shared_ptr<quad>> &lights, int depth) const = 0;
};

class path_tracer : public integrator
{
public:
    path_tracer(bool do_nee) : nee(do_nee) {}

    vec3 Li(const ray &r, const hittable &world,
            const std::vector<std::shared_ptr<quad>> &lights, int depth) const override
    {
        if (depth <= 0)
            return vec3(0, 0, 0);

        hit_record rec;
        if (!world.hit(r, 0.001, 1000.0, rec))
            return sky_color(r);

        vec3 color = rec.mat->emitted();

        ray scattered;
        vec3 attenuation;
        if (!rec.mat->scatter(r, rec, attenuation, scattered))
            return color; // emissive surface: no bounce

        if (nee && !rec.mat->specular() && !lights.empty())
            color += direct_light(rec, attenuation, world, lights);

        return color + attenuation * Li(scattered, world, lights, depth - 1);
    }

private:
    static vec3 sky_color(const ray &r)
    {
        vec3 unit_direction = unit_vector(r.direction());
        double a = 0.5 * (unit_direction.y() + 1.0);
        return (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0);
    }

    // Next-event estimation: pick one light uniformly, sample a point on it,
    // and test the shadow ray. The uniform area sample (pdf 1/area, times 1/n
    // for the light choice) is converted to solid angle at the shading point:
    // pdf_dir = dist^2 / (cos_light * area * n).
    static vec3 direct_light(const hit_record &rec, const vec3 &albedo,
                             const hittable &world,
                             const std::vector<std::shared_ptr<quad>> &lights)
    {
        size_t n = lights.size();
        size_t idx = (n == 1) ? 0 : std::min(n - 1, static_cast<size_t>(random_double(0, n)));
        const auto &light = lights[idx];

        vec3 to_light = light->sample() - rec.point;
        double dist2 = to_light.length_squared();
        double dist = std::sqrt(dist2);
        vec3 dir = to_light / dist;

        double cos_surface = dot(rec.normal, dir);
        double cos_light = dot(-dir, light->normal());
        if (cos_surface <= 0.0 || cos_light <= 0.0)
            return vec3(0, 0, 0);

        hit_record tmp;
        if (world.hit(ray(rec.point, dir), 0.001, dist - 0.001, tmp))
            return vec3(0, 0, 0); // occluded

        double pdf_dir = dist2 / (cos_light * light->area() * static_cast<double>(n));
        vec3 emission = light->mat_ptr()->emitted();
        return albedo * emission * cos_surface / pdf_dir;
    }

    bool nee;
};
