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
    normal = rec.normal;
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
    bool mix_pdf = false;
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
                    // MIS weight vs active env PDF (HDRI or analytic 1/4PI).
                    double w = direction_pdf::power_weight(
                        pdf_b_last, env_light::sample_pdf_for(cur.direction()));
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
                    // the NEE density of the same path (single source: pdf.h).
                    double pdf_l = direction_pdf::nee_value_for_hit(
                        p.lights, rec.mat, rec.point, prev_point, cur.time());
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

            if (diffuse && !p.lights.empty() && !p.mix_pdf) {
                // Next-event estimation: uniform light + uniform point.
                int li = (int)(random_double() * p.lights.size());
                if (li >= (int)p.lights.size())
                    li = (int)p.lights.size() - 1;
                const auto &light = p.lights[(size_t)li];
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
                        double pdf_l = dist * dist /
                                       ((double)p.lights.size() * area * cosA);
                        double pdf_b = cosine_pdf(cosS);
                        double w = direction_pdf::power_weight(pdf_l, pdf_b);
                        // f*G/pdf_area: rho*Le*cosS*cosA*A*L/(PI*dist^2)
                        L += throughput * attenuation * light_Le *
                             (cosS * cosA * (double)p.lights.size() * area /
                              (pi * dist * dist)) * w * Tr;
                    }
                }
            }

            if (p.use_env && diffuse && !p.mix_pdf) {
                // Environment NEE: sample from the active env (analytic uniform
                // sphere or HDRI CDF), shadow probe to infinity, power MIS
                // against the cosine strategy. Draws RNG only when opted in.
                double pdf_e = 0.0;
                vec3 edir = env_light::sample_dir(random_double(), random_double(), pdf_e);
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

            if (vol_nee && !p.lights.empty() && !p.mix_pdf) {
                // Volume NEE (M59): direct-light in-scattering at the event.
                // Phase is uniform (no cosS gate, no cosS in the weight);
                // MIS against the 1/4PI continuation, like surface NEE.
                // Draws only on events, so event-free scenes stay byte-exact.
                int li = (int)(random_double() * p.lights.size());
                if (li >= (int)p.lights.size())
                    li = (int)p.lights.size() - 1;
                const auto &light = p.lights[(size_t)li];
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
                        double pdf_l = dist * dist /
                                       ((double)p.lights.size() * area * cosA);
                        double pdf_b = 1.0 / (4.0 * pi);
                        double w = direction_pdf::power_weight(pdf_l, pdf_b);
                        // f*G/pdf: (rho/4PI)*Le*A*cosA/dist^2, N lights.
                        L += throughput * attenuation * light_Le *
                             (cosA * (double)p.lights.size() * area /
                              (4.0 * pi * dist * dist)) * w * Tr;
                    }
                }
            }

            if (p.use_env && vol_nee && !p.mix_pdf) {
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
                if (p.mix_pdf) {
                    // Book mixture path (M61): direction + density from the
                    // 50/50 blend, weight by cosine/mixture, no shadow rays.
                    // Emission counts full (specular=true below): the blend
                    // already importance-samples lights, no MIS weights.
                    double pdf_mix = 0;
                    vec3 mdir = direction_pdf::sample_mixture(p.world, p.lights, rec.point,
                                                              rec.normal, cur.time(),
                                                              pdf_mix);
                    double cval =
                        direction_pdf::cosine_value(unit_vector(mdir), rec.normal);
                    if (pdf_mix <= 0)
                        break; // degenerate: absorbed
                    throughput = throughput * attenuation * (cval / pdf_mix);
                    scattered = ray(rec.point, mdir);
                } else {
                    // Cosine sampling: f*cos/pdf = rho exact, no division.
                    vec3 wi = unit_vector(scattered.direction());
                    pdf_b_last = cosine_pdf(dot(wi, rec.normal));
                    throughput = throughput * attenuation;
                }
                specular = p.mix_pdf; // mixture: full counts, no MIS anywhere
            } else {
                // First-class sampling density: delta materials report 0 and
                // keep full light counts; isotropic now MIS-weights (1/4PI).
                pdf_b_last = rec.mat->direction_pdf(scattered.direction(), rec);
                throughput = throughput * attenuation;
                specular = (pdf_b_last <= 0);
                // Mixture mode: volumes continue pure, found lights full.
                if (p.mix_pdf)
                    specular = true;
            }
            prev_point = rec.point;
            scattered.set_time(cur.time()); // path shares primary time
            cur = scattered;
            count_ray(); // bounce ray

            // Russian roulette past depth 3: unbiased, scales by 1/q.
            if (bounce >= 3) {
                double q = throughput.x();
                if (throughput.y() > q)
                    q = throughput.y();
                if (throughput.z() > q)
                    q = throughput.z();
                q = q < 0.95 ? q : 0.95;
                if (q <= 0 || random_double() > q)
                    break;
                throughput = throughput / q;
            }
        }
        return L;
    }
};
