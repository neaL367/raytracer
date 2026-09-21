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
#include "../material/material.h"
#include "pdf.h"
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

class integrator {
public:
    vec3 Li(const ray &r, const hittable &world,
            const std::vector<light> &lights, int max_depth,
            bool use_env = false) const {
        count_ray(); // primary
        vec3 throughput(1, 1, 1);
        vec3 L(0, 0, 0);
        ray cur = r;
        vec3 prev_point = r.origin();
        bool specular = true; // camera path: direct light views count full
        double pdf_b_last = 0;
        const double pi = 3.1415926535897932385;

        for (int bounce = 0; bounce < max_depth; ++bounce) {
            hit_record rec;
            if (!world.hit(cur, 0.001, 1e30, rec)) {
                // Miss: legacy sky, or the sun+sky environment. Bounced
                // misses MIS-weight against the env strategy (primary misses
                // keep full count: specular path, same as legacy).
                vec3 miss_L = use_env ? env_light::radiance(cur.direction())
                                      : env_light::sky(cur.direction());
                if (use_env && !specular) {
                    double w = direction_pdf::power_weight(
                        pdf_b_last, env_light::sample_pdf());
                    L += throughput * miss_L * w;
                } else {
                    L += throughput * miss_L;
                }
                break;
            }
            vec3 Le = rec.mat->emitted();
            if (Le.length_squared() > 0) {
                if (specular) {
                    L += throughput * Le;
                } else {
                    // MIS weight for BSDF-found light: power heuristic over
                    // the NEE density of the same path (single source: pdf.h).
                    double pdf_l = direction_pdf::nee_value_for_hit(
                        lights, rec.mat, rec.point, prev_point, cur.time());
                    double w = (pdf_l <= 0) ? 1.0
                                            : direction_pdf::power_weight(pdf_b_last, pdf_l);
                    L += throughput * Le * w;
                }
                break; // emission terminates path
            }

            ray scattered;
            vec3 attenuation;
            if (!rec.mat->scatter(cur, rec, attenuation, scattered))
                break; // absorbed
            bool diffuse = rec.mat->is_diffuse();

            if (diffuse && !lights.empty()) {
                // Next-event estimation: uniform light + uniform point.
                int li = (int)(random_double() * lights.size());
                if (li >= (int)lights.size())
                    li = (int)lights.size() - 1;
                const auto &light = lights[(size_t)li];
                vec3 lp = light_point(light, random_double(), random_double(), cur.time());
                vec3 toL = lp - rec.point;
                double dist = toL.length();
                vec3 wi = toL / dist;
                vec3 ln = light_normal_at(light, lp, cur.time());
                double cosS = dot(rec.normal, wi);
                double cosA = fabs(dot(ln, -wi));
                double area = light_area(light);
                if (cosS > 0 && cosA > 0 && area > 0) {
                    hit_record tmp;
                    count_ray(); // shadow ray
                    ray shadow(rec.point, wi, cur.time());
                    bool blocked = world.hit(shadow, 0.001, dist - 0.001, tmp);
                    if (!blocked) {
                        vec3 light_Le = light_mat(light)->emitted();
                        double pdf_l = dist * dist /
                                       ((double)lights.size() * area * cosA);
                        double pdf_b = cosine_pdf(cosS);
                        double w = direction_pdf::power_weight(pdf_l, pdf_b);
                        // f*G/pdf_area: rho*Le*cosS*cosA*A*L/(PI*dist^2)
                        L += throughput * attenuation * light_Le *
                             (cosS * cosA * (double)lights.size() * area /
                              (pi * dist * dist)) * w;
                    }
                }
            }

            if (use_env && diffuse) {
                // Environment NEE: uniform-sphere sample, shadow probe to
                // infinity, power MIS against the cosine strategy. Draws RNG
                // only when opted in, so env-off streams stay byte-exact.
                vec3 edir = env_light::sample_dir(random_double(), random_double());
                double cosS = dot(rec.normal, edir);
                if (cosS > 0) {
                    hit_record etmp;
                    count_ray(); // env shadow ray
                    ray eshadow(rec.point, edir, cur.time());
                    if (!world.hit(eshadow, 0.001, 1e30, etmp)) {
                        vec3 env_Le = env_light::radiance(edir);
                        double pdf_e = env_light::sample_pdf();
                        double pdf_b = cosine_pdf(cosS);
                        double w = direction_pdf::power_weight(pdf_e, pdf_b);
                        L += throughput * attenuation * env_Le *
                             (cosS / (pi * pdf_e)) * w;
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
