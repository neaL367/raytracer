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

// Transmittance along a shadow ray through homogeneous fog: blocked reads 0,
// clear reads exp(-sigma * dist). Free function so unit tests can check it
// without an integrator instance.
inline double shadow_transmittance(const hittable &world, double sigma,
                                   const vec3 &origin, const vec3 &direction, double dist)
{
    hit_record tmp;
    if (world.hit(ray(origin, direction), 0.001, dist - 0.001, tmp))
        return 0.0;
    return std::exp(-sigma * dist);
}

class path_tracer : public integrator
{
public:
    // Homogeneous fog: extinction sigma_t per unit distance, single-scatter
    // albedo fog_albedo, isotropic phase. Zero density disables everything.
    static constexpr double fog_albedo = 0.85;
    static constexpr double inv_4pi = 0.07957747154594767; // 1/(4*pi)

    path_tracer(bool do_nee, bool do_rr = true, double fog_density = 0.0)
        : nee(do_nee), roulette(do_rr), sigma_t(fog_density) {}

    vec3 Li(const ray &r, const hittable &world,
            const std::vector<std::shared_ptr<quad>> &lights, int depth) const override
    {
        return Li_weighted(r, world, lights, depth, 1.0, 1.0);
    }

private:
    // Balance heuristic: every emission event is weighted by the probability
    // that the strategy which found it would find it, over all strategies
    // that could have. Light-sampled emission gets pl/(pl+ps), BSDF-sampled
    // emission ps/(pl+ps) — the weights sum to 1, so direct light is counted
    // exactly once instead of twice (explicit ray plus lucky bounce).
    // Throughput itself stays unweighted: with cosine sampling the
    // albedo*cos/pdf quotient collapses back to albedo.
    vec3 Li_weighted(const ray &r, const hittable &world,
                     const std::vector<std::shared_ptr<quad>> &lights,
                     int depth, double emission_weight, double throughput) const
    {
        if (depth <= 0)
            return vec3(0, 0, 0);

        hit_record rec;
        bool has_hit = world.hit(r, 0.001, 1000.0, rec);

        // Homogeneous fog: free-flight distance tm ~ sigma_t * exp(-sigma_t).
        // Reaching the surface (tm beyond it) conditions away its own
        // transmittance, so the surface branch below is unchanged; scattering
        // first weighs albedo. Gated on density so the default path draws no
        // extra numbers and its streams never shift.
        if (sigma_t > 0.0)
        {
            double xi = random_double();
            if (xi < 1e-12)
                xi = 1e-12; // log(0) is not a scattering distance
            double tm = -std::log(xi) / sigma_t;
            if (!has_hit || tm < rec.t)
            {
                vec3 mp = r.at(tm);
                vec3 color = vec3(0, 0, 0); // the medium itself emits nothing
                if (nee && !lights.empty())
                    color += direct_medium(mp, world, lights);

                vec3 ndir = random_unit_vector(); // isotropic phase
                double bounce_weight = 1.0;
                if (nee && !lights.empty())
                {
                    double ps = inv_4pi;
                    double pl = light_pdf(mp, unit_vector(ndir), lights);
                    bounce_weight = (ps + pl > 0.0) ? ps / (ps + pl) : 1.0;
                }

                double rr_scale = 1.0;
                double path_throughput = throughput * fog_albedo;
                if (roulette && path_throughput < 0.9)
                {
                    double q = std::max(0.05, path_throughput);
                    if (random_double() >= q)
                        return color;
                    rr_scale = 1.0 / q;
                }
                ray next(mp, ndir);
                return color + vec3(fog_albedo, fog_albedo, fog_albedo) * rr_scale *
                                   Li_weighted(next, world, lights, depth - 1, bounce_weight, path_throughput);
            }
        }

        if (!has_hit)
            return sky_color(r); // background is not an emitter: never weighted

        vec3 color = rec.mat->emitted() * emission_weight;

        ray scattered;
        vec3 attenuation;
        if (!rec.mat->scatter(r, rec, attenuation, scattered))
            return color; // emissive surface: no bounce

        if (nee && !rec.mat->specular() && !lights.empty())
            color += direct_light(r, rec, attenuation, world, lights);

        double bounce_weight = 1.0;
        if (nee && !rec.mat->specular() && !lights.empty())
        {
            double ps = rec.mat->scattering_pdf(r, rec, scattered);
            double pl = light_pdf(rec.point, unit_vector(scattered.direction()), lights);
            bounce_weight = (ps + pl > 0.0) ? ps / (ps + pl) : 1.0;
        }

        // Russian roulette: terminate dim paths with probability 1-q, scale
        // survivors by 1/q — expectation unchanged, deep chains stop burning
        // full depth-50 walks for thousandths of radiance. Survival tracks the
        // strongest surviving channel, floored so near-black paths still end
        // occasionally instead of never.
        double rr_scale = 1.0;
        double path_throughput = throughput * max_channel(attenuation);
        if (roulette && path_throughput < 0.9)
        {
            double q = std::max(0.05, path_throughput);
            if (random_double() >= q)
                return color; // terminated: emission + direct kept, bounce dropped
            rr_scale = 1.0 / q;
        }

        return color + attenuation * rr_scale *
                          Li_weighted(scattered, world, lights, depth - 1, bounce_weight, path_throughput);
    }

    static double max_channel(const vec3 &v)
    {
        return std::max(v.x(), std::max(v.y(), v.z()));
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
    // pdf_dir = dist^2 / (cos_light * area * n). Weighted by the balance
    // heuristic against the BSDF sampling the bounce would have used.
    // Shadow transmittance (not just occlusion) folds fog in: blocked reads
    // 0, clear reads exp(-sigma * dist). With sigma = 0 that is exactly the
    // old occluded-or-1 behavior, bit for bit.
    vec3 direct_light(const ray &r_in, const hit_record &rec, const vec3 &albedo,
                      const hittable &world,
                      const std::vector<std::shared_ptr<quad>> &lights) const
    {
        size_t n = lights.size();
        size_t idx = (n == 1) ? 0 : std::min(n - 1, static_cast<size_t>(random_double(0.0, static_cast<double>(n))));
        const auto &light = lights[idx];

        vec3 to_light = light->sample() - rec.point;
        double dist2 = to_light.length_squared();
        double dist = std::sqrt(dist2);
        vec3 dir = to_light / dist;

        double cos_surface = dot(rec.normal, dir);
        double cos_light = dot(-dir, light->normal());
        if (cos_surface <= 0.0 || cos_light <= 0.0)
            return vec3(0, 0, 0);

        double trans = shadow_transmittance(world, sigma_t, rec.point, dir, dist);
        if (trans <= 0.0)
            return vec3(0, 0, 0); // occluded

        double pdf_light = dist2 / (cos_light * light->area() * static_cast<double>(n));
        double pdf_bsdf = rec.mat->scattering_pdf(r_in, rec, ray(rec.point, dir));
        double weight = (pdf_light + pdf_bsdf > 0.0) ? pdf_light / (pdf_light + pdf_bsdf) : 0.0;
        vec3 emission = light->mat_ptr()->emitted();
        return albedo * emission * cos_surface * trans / pdf_light * weight;
    }

    // In-scatter NEE at a medium event: isotropic phase (1/4pi) replaces the
    // cosine lobe, and there is no surface orientation to project.
    vec3 direct_medium(const vec3 &p, const hittable &world,
                       const std::vector<std::shared_ptr<quad>> &lights) const
    {
        size_t n = lights.size();
        size_t idx = (n == 1) ? 0 : std::min(n - 1, static_cast<size_t>(random_double(0.0, static_cast<double>(n))));
        const auto &light = lights[idx];

        vec3 to_light = light->sample() - p;
        double dist2 = to_light.length_squared();
        double dist = std::sqrt(dist2);
        vec3 dir = to_light / dist;

        double cos_light = dot(-dir, light->normal());
        if (cos_light <= 0.0)
            return vec3(0, 0, 0);

        double trans = shadow_transmittance(world, sigma_t, p, dir, dist);
        if (trans <= 0.0)
            return vec3(0, 0, 0);

        double pdf_light = dist2 / (cos_light * light->area() * static_cast<double>(n));
        double weight = (pdf_light + inv_4pi > 0.0) ? pdf_light / (pdf_light + inv_4pi) : 0.0;
        vec3 emission = light->mat_ptr()->emitted();
        return vec3(fog_albedo, fog_albedo, fog_albedo) * emission * inv_4pi * trans / pdf_light * weight;
    }

    // Mixture pdf of the light-sampling strategy: uniform light choice over
    // per-light solid-angle densities.
    static double light_pdf(const vec3 &origin, const vec3 &direction,
                            const std::vector<std::shared_ptr<quad>> &lights)
    {
        if (lights.empty())
            return 0.0;
        double sum = 0.0;
        for (const auto &light : lights)
            sum += light->pdf_value(origin, direction);
        return sum / static_cast<double>(lights.size());
    }

    bool nee;
    bool roulette;
    double sigma_t;
};

// Reference integrator for GPU parity: closest hit mapped to color, no
// materials, no lights, no RNG. Any backend tracing the same scene must
// produce the same normals (within float precision) — that is the whole
// test.
class normal_integrator : public integrator
{
public:
    vec3 Li(const ray &r, const hittable &world,
            const std::vector<std::shared_ptr<quad>> &, int) const override
    {
        hit_record rec;
        if (!world.hit(r, 0.001, 1000.0, rec))
        {
            vec3 unit_direction = unit_vector(r.direction());
            double a = 0.5 * (unit_direction.y() + 1.0);
            return (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0);
        }
        return rec.normal * 0.5 + vec3(0.5, 0.5, 0.5);
    }
};
