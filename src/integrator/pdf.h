#pragma once
// Direction-sampling densities (RTOYL Ch.9 flavor), added beside the current
// NEE path: nothing includes this yet, so M32 pixels are untouched.
// cosine branch: pdf = cos/PI. Light branch: uniform light + uniform point,
// pdf = dist^2/(N*A*cos) when the ray strikes an emissive shape, else 0.
// Mixture: 50/50 blend, one interface for both sampling and evaluation.
#include "../core/onb.h"
#include "../core/random.h"
#include "../core/ray.h"
#include "../core/sampler.h"
#include "../core/vec3.h"
#include "../geometry/hittable.h"
#include "../geometry/light.h"

#include <cmath>
#include <memory>
#include <vector>

namespace direction_pdf {

// Cosine lobe value for a (unit) direction about the shading normal.
inline double cosine_value(const vec3 &dir, const vec3 &normal) {
    return cosine_pdf(dot(dir, normal));
}

// Power-pick reverse density: pick_p(i) * dist^2/(A*cosA) for the struck
// light i (matched to the forward power sampler).
inline double nee_value_for_hit_power(const std::vector<light> &lights,
                                       const std::vector<double> &cdf, double total,
                                       const std::shared_ptr<material> &mat,
                                       const vec3 &hit_point, const vec3 &prev_point,
                                       double time) {
    vec3 to_hit = hit_point - prev_point;
    double dist = to_hit.length();
    if (dist <= 0 || cdf.size() != lights.size() || total <= 0)
        return 0.0;
    vec3 wi = to_hit / dist;
    for (size_t i = 0; i < lights.size(); ++i) {
        if (light_mat(lights[i]) == mat) {
            double cosA =
                fabs(dot(light_normal_at(lights[i], hit_point, time), -wi));
            double A = light_area(lights[i]);
            if (cosA > 0 && A > 0) {
                double prev = i > 0 ? cdf[i - 1] : 0.0;
                double pick_p = (cdf[i] - prev) / total;
                if (pick_p <= 0)
                    return 0.0;
                return pick_p * dist * dist / (A * cosA);
            }
            return 0.0;
        }
    }
    return 0.0;
}

// Power heuristic for one sample each (PBRT PathIntegrator): squares the
// densities before the balance ratio, cutting fireflies from rare bright hits.
inline double power_weight(double pdf_a, double pdf_b) {
    double a2 = pdf_a * pdf_a, b2 = pdf_b * pdf_b;
    double denom = a2 + b2;
    return denom > 0 ? a2 / denom : 0.0;
}

} // namespace direction_pdf
