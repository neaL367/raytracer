#pragma once
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/random.h"
#include "../core/sampler.h"
#include "../core/bench_stats.h"
#include "../core/env.h"
#include "../geometry/hittable.h"
#include "../geometry/light.h"
#include "../geometry/quad.h"
#include "../geometry/volume.h"
#include "../material/material.h"
#include "pdf.h"
#include "nesting.h"
#include <cmath>
#include <memory>
#include <vector>

// Path integrator: NEE + power-heuristic MIS + Russian roulette.
// Loop (not recursion): throughput accumulates, L sums weighted emission.
// Delta bounces skip NEE and count light hits full; sampled materials
// (cosine, isotropic) weight BSDF-found lights by MIS. Camera-ray light
// hits count full. Densities live in pdf.h (single source of truth).
// First-hit AOV query: albedo + world normal, no bounce, no RNG.
// Guide draws never perturb the beauty stream (pure function of ray).
inline void first_hit_aov(const ray &r, const hittable &world, vec3 &albedo, vec3 &normal,
                           bool &hit, double *depth = nullptr) {
    hit_record rec;
    hit = world.hit(r, 0.001, 1e30, rec);
    if (!hit)
        return;
    albedo = rec.mat->surface_albedo(rec);
    normal = rec.mat->resolve_normal(rec);
    if (depth)
        *depth = rec.t;
}

// Render context for Li(): one const-ref replaces the growing parameter
// list. New switches (HDRI pointers, LOD bias, extra flags) extend this
// struct instead of changing the integrator signature again.
struct render_params {
    const hittable &world;
    const std::vector<light> &lights;
    int max_depth = 50;
    const std::vector<std::shared_ptr<hittable>> &media;
    bool use_env = false;
    bool black_bg = false;
    // Power-CDF light importance (always on): forward NEE picks by power,
    // reverse densities match. Built once per render, read-only in Li().
    const std::vector<double> *light_cdf = nullptr;
    double light_power_total = 0.0;
};

class integrator {
public:
    vec3 Li(const ray &r, const render_params &p) const {
        count_ray(); // primary
        vec3 throughput(1, 1, 1);
        vec3 L(0, 0, 0);
        ray cur = r;
        vec3 prev_point = r.origin();
        bool specular = true; // camera path: direct light views count full
        double pdf_b_last = 0;
        medium_stack nest; // nested-dielectric path state (M57; air bottom)
        const double pi = 3.1415926535897932385;

        for (int bounce = 0; bounce < p.max_depth; ++bounce) {
            hit_record rec;
            if (!p.world.hit(cur, 0.001, 1e30, rec)) {
                if (p.black_bg)
                    break;
                // Miss: legacy sky, or the sun+sky environment. Bounced
                // misses MIS-weight against the env strategy (primary misses
                // keep full count: specular path, same as legacy).
                vec3 miss_L = p.use_env ? env_light::radiance(cur.direction())
                                      : env_light::sky(cur.direction());
                if (p.use_env && !specular) {
                    // MIS weight vs the sun-cone mixture forward sampler.
                    double w = direction_pdf::power_weight(
                        pdf_b_last,
                        env_light::sample_pdf_mix_for(cur.direction()));
                    L += throughput * miss_L * w;
                } else {
                    L += throughput * miss_L;
                }
                break;
            }
            vec3 Le = rec.mat->emitted(rec);
            if (Le.length_squared() > 0) {
                if (specular) {
                    L += throughput * Le;
                } else {
                    // MIS weight for BSDF-found light: power heuristic over
                    // the CDF-matched NEE density (single source: pdf.h).
                    double pdf_l = direction_pdf::nee_value_for_hit_power(
                        p.lights, *p.light_cdf, p.light_power_total, rec.mat,
                        rec.point, prev_point, cur.time());
                    double w = (pdf_l <= 0) ? 1.0
                                            : direction_pdf::power_weight(pdf_b_last, pdf_l);
                    L += throughput * Le * w;
                }
                break; // emission terminates path
            }

            ray scattered;
            vec3 attenuation;
            // Nested interfaces (M57): resolve the medium stack for
            // transmissive hits; dielectric::scatter consumes nest_eta when
            // set, else legacy front_face. Non-dielectrics skip (the virtuals
            // default to vacuum, so only true glass routes here).
            if (rec.mat->ior() != 1.0 || rec.mat->priority() != 0) {
                double eta = 0, chord = 0;
                vec3 exit_absorb;
                medium_stack::event ev =
                    nest.resolve(rec.mat->ior(), rec.mat->priority(), rec.hit_prim,
                                 rec.point, rec.mat->absorb(), eta, chord, exit_absorb);
                if (ev != medium_stack::PASS) {
                    rec.nest_eta = eta;
                    rec.nest_set = true;
                    // Beer's law on the exit chord (M58): ENTER only stamps.
                    if (ev == medium_stack::EXIT)
                        throughput = throughput * beer_transmittance(exit_absorb, chord);
                }
            }
            if (!rec.mat->scatter(cur, rec, attenuation, scattered))
                break; // absorbed
            bool diffuse = rec.mat->is_diffuse();
            bool vol = rec.mat->is_volume(); // scattering event in media
            // Density gate (M60): thin media skip explicit NEE (back to
            // walks-only); hit_obj is the firing medium here.
            bool vol_nee = false;
            if (vol) {
                double dens = 0;
                if (auto cm = dynamic_cast<const constant_medium *>(rec.hit_obj))
                    dens = cm->density_val();
                else if (auto hm = dynamic_cast<const heterogeneous_medium *>(rec.hit_obj))
                    dens = hm->density_val();
                vol_nee = volume_nee_fires(dens);
            }

            if (diffuse && !p.lights.empty()) {
                // Next-event estimation: power-CDF light + uniform point.
                double pick_p = 1.0;
                size_t li = pick_light_power(*p.light_cdf, p.light_power_total,
                                             random_double(), pick_p);
                const auto &light = p.lights[li];
                double eu1 = random_double(), eu2 = random_double();
                vec3 lp = light_point(light, eu1, eu2, cur.time());
                vec3 toL = lp - rec.point;
                double dist = toL.length();
                vec3 wi = toL / dist;
                vec3 ln = light_normal_at(light, lp, cur.time());
                double cosS = dot(rec.normal, wi);
                double cosA = fabs(dot(ln, -wi));
                double area = light_area(light);
                if (cosS > 0 && cosA > 0 && area > 0) {
                    count_ray(); // shadow ray
                    // Transmittance-weighted NEE: smoke attenuates instead
                    // of binary-blocking (march returns 1.0 with no media).
                    double Tr = shadow_transmittance(p.world, p.media, rec.point, wi, dist,
                                                     cur.time());
                    if (Tr > 0) {
                        double lu = 0, lv = 0;
                        light_uv(light, eu1, eu2, cur.time(), lu, lv);
                        vec3 light_Le = light_emission(light, lp, lu, lv, dist);
                        // Forward pdf matches the pick: uniform 1/N or
                        // power pick_p. Weight factor carries 1/pick_p.
                        double pdf_l = pick_p * dist * dist / (area * cosA);
                        double pdf_b = cosine_pdf(cosS);
                        double w = direction_pdf::power_weight(pdf_l, pdf_b);
                        // f*G/pdf_area: rho*Le*cosS*cosA*A/(PI*dist^2*pick_p)
                        L += throughput * attenuation * light_Le *
                             (cosS * cosA * area / (pick_p * pi * dist * dist)) *
                             w * Tr;
                    }
                }
            }

            if (p.use_env && diffuse) {
                // Environment NEE: 50/50 sun-cone leg (analytic only; HDRI
                // keeps its own CDF), shadow probe to infinity, power MIS
                // against the cosine strategy.
                double pdf_e = 0.0;
                vec3 edir;
                if (env_light::g_hdri() == nullptr && random_double() < 0.5) {
                    double u1 = random_double(), u2 = random_double();
                    double c = env_light::sun_cos_thresh();
                    double z = 1.0 - (1.0 - c) * u1;
                    double phi = 2.0 * pi * u2;
                    double rr = std::sqrt(std::max(1.0 - z * z, 0.0));
                    vec3 local(rr * std::cos(phi), rr * std::sin(phi), z);
                    onb sun_frame;
                    sun_frame.build_from_w(env_light::sun_dir());
                    edir = sun_frame.local(local);
                    double cone_pdf = env_light::sun_cone_pdf();
                    double sph_pdf = env_light::analytic_sample_pdf();
                    pdf_e = 0.5 * cone_pdf + 0.5 * sph_pdf;
                } else {
                    edir = env_light::sample_dir(random_double(), random_double(),
                                                 pdf_e);
                    if (env_light::g_hdri() == nullptr) {
                        double cone_pdf =
                            (dot(unit_vector(edir), env_light::sun_dir()) >=
                             env_light::sun_cos_thresh())
                                ? env_light::sun_cone_pdf()
                                : 0.0;
                        pdf_e = 0.5 * cone_pdf + 0.5 * pdf_e;
                    }
                }
                double cosS = dot(rec.normal, edir);
                if (cosS > 0 && pdf_e > 0) {
                    count_ray(); // env shadow ray
                    double Tr = shadow_transmittance(p.world, p.media, rec.point, edir, 1e30,
                                                     cur.time());
                    if (Tr > 0) {
                        vec3 env_Le = env_light::radiance(edir);
                        double pdf_b = cosine_pdf(cosS);
                        double w = direction_pdf::power_weight(pdf_e, pdf_b);
                        L += throughput * attenuation * env_Le *
                             (cosS / (pi * pdf_e)) * w * Tr;
                    }
                }
            }

            if (vol_nee && !p.lights.empty()) {
                // Volume NEE (M59): direct-light in-scattering at the event.
                // Phase is uniform (no cosS gate, no cosS in the weight);
                // MIS against the 1/4PI continuation, like surface NEE.
                // Draws only on events, so event-free scenes stay byte-exact.
                double vpick_p = 1.0;
                size_t vli = pick_light_power(*p.light_cdf, p.light_power_total,
                                              random_double(), vpick_p);
                const auto &light = p.lights[vli];
                double eu1 = random_double(), eu2 = random_double();
                vec3 lp = light_point(light, eu1, eu2, cur.time());
                vec3 toL = lp - rec.point;
                double dist = toL.length();
                vec3 wi = toL / dist;
                vec3 ln = light_normal_at(light, lp, cur.time());
                double cosA = fabs(dot(ln, -wi));
                double area = light_area(light);
                if (cosA > 0 && area > 0) {
                    count_ray(); // shadow ray
                    double Tr = shadow_transmittance(p.world, p.media, rec.point, wi, dist,
                                                     cur.time());
                    if (Tr > 0) {
                        double lu = 0, lv = 0;
                        light_uv(light, eu1, eu2, cur.time(), lu, lv);
                        vec3 light_Le = light_emission(light, lp, lu, lv, dist);
                        double pdf_l = vpick_p * dist * dist / (area * cosA);
                        double pdf_b = 1.0 / (4.0 * pi);
                        double w = direction_pdf::power_weight(pdf_l, pdf_b);
                        // f*G/pdf: (rho/4PI)*Le*A*cosA/(dist^2*pick_p).
                        L += throughput * attenuation * light_Le *
                             (cosA * area / (vpick_p * 4.0 * pi * dist * dist)) *
                             w * Tr;
                    }
                }
            }

            if (p.use_env && vol_nee) {
                // Environment in-scattering: sample from active env, probe to
                // infinity, power MIS against the uniform phase continuation.
                double pdf_e = 0.0;
                vec3 edir = env_light::sample_dir(random_double(), random_double(), pdf_e);
                count_ray(); // env shadow ray
                if (pdf_e > 0) {
                    double Tr = shadow_transmittance(p.world, p.media, rec.point, edir, 1e30,
                                                     cur.time());
                    if (Tr > 0) {
                        vec3 env_Le = env_light::radiance(edir);
                        double pdf_b = 1.0 / (4.0 * pi);
                        double w = direction_pdf::power_weight(pdf_e, pdf_b);
                        L += throughput * attenuation * env_Le * (1.0 / (4.0 * pi * pdf_e)) *
                             w * Tr;
                    }
                }
            }

            if (diffuse) {
                // Cosine sampling: f*cos/pdf = rho exact, no division.
                vec3 wi = unit_vector(scattered.direction());
                pdf_b_last = cosine_pdf(dot(wi, rec.normal));
                throughput = throughput * attenuation;
                specular = false;
            } else {
                // First-class sampling density: delta materials report 0 and
                // keep full light counts; isotropic now MIS-weights (1/4PI).
                pdf_b_last = rec.mat->direction_pdf(scattered.direction(), rec);
                throughput = throughput * attenuation;
                specular = (pdf_b_last <= 0);
            }
            prev_point = rec.point;
            scattered.set_time(cur.time()); // path shares primary time
            cur = scattered;
            count_ray(); // bounce ray

            // Russian roulette past depth 3: luminance survival probability
            // (kills single-channel stragglers max-channel keeps), unbiased
            // 1/q rescale. Mirrored in the GPU shader.
            if (bounce >= 3) {
                double q = 0.2126 * throughput.x() + 0.7152 * throughput.y() +
                           0.0722 * throughput.z();
                q = q < 0.95 ? q : 0.95;
                if (q <= 0 || random_double() > q)
                    break;
                throughput = throughput / q;
            }
        }
        return L;
    }
};
