#pragma once
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/random.h"
#include "../core/sampler.h"
#include "../geometry/hittable.h"
#include "../geometry/quad.h"
#include "../material/material.h"
#include <cmath>
#include <memory>
#include <vector>

// Path integrator: NEE + balance-heuristic MIS + Russian roulette.
// Loop (not recursion): throughput accumulates, L sums weighted emission.
// Specular (delta) bounces skip NEE and count light hits full; diffuse
// BSDF hits weight by MIS. Camera-ray light hits count full.
class integrator {
public:
    vec3 Li(const ray &r, const hittable &world,
            const std::vector<std::shared_ptr<quad>> &lights, int max_depth) const {
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
                vec3 unit = unit_vector(cur.direction());
                double t = 0.5 * (unit.y() + 1.0);
                L += throughput * ((1.0 - t) * vec3(1, 1, 1) + t * vec3(0.5, 0.7, 1.0));
                break;
            }
            vec3 Le = rec.mat->emitted();
            if (Le.length_squared() > 0) {
                if (specular) {
                    L += throughput * Le;
                } else {
                    // MIS weight for BSDF-found light: needs direction pdf
                    // from prev vertex to this hit point.
                    vec3 to_hit = rec.point - prev_point;
                    double dist = to_hit.length();
                    vec3 wi = to_hit / dist;
                    double pdf_l = 0;
                    for (const auto &lt : lights) {
                        if (lt->mat_ptr() == rec.mat) {
                            double cosA = fabs(dot(lt->light_normal(), -wi));
                            if (cosA > 0 && lt->area() > 0)
                                pdf_l = dist * dist /
                                        ((double)lights.size() * lt->area() * cosA);
                            break;
                        }
                    }
                    double w = (pdf_l <= 0) ? 1.0
                                            : pdf_b_last / (pdf_b_last + pdf_l);
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
                vec3 lp = light->sample_point();
                vec3 toL = lp - rec.point;
                double dist = toL.length();
                vec3 wi = toL / dist;
                double cosS = dot(rec.normal, wi);
                double cosA = fabs(dot(light->light_normal(), -wi));
                if (cosS > 0 && cosA > 0 && light->area() > 0) {
                    hit_record tmp;
                    bool blocked = world.hit(ray(rec.point, wi), 0.001,
                                             dist - 0.001, tmp);
                    if (!blocked) {
                        vec3 light_Le = light->mat_ptr()->emitted();
                        double pdf_l = dist * dist /
                                       ((double)lights.size() * light->area() * cosA);
                        double pdf_b = cosine_pdf(cosS);
                        double w = pdf_l / (pdf_l + pdf_b);
                        // f*G/pdf_area: rho*Le*cosS*cosA*A*L/(PI*dist^2)
                        L += throughput * attenuation * light_Le *
                             (cosS * cosA * (double)lights.size() * light->area() /
                              (pi * dist * dist)) * w;
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
                throughput = throughput * attenuation;
                specular = true; // delta bounce: light hits count full
            }
            prev_point = rec.point;
            cur = scattered;

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
