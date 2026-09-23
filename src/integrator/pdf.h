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

// NEE-strategy pdf for firing origin along dir: strike an emissive shape and
// report its sampling density, else 0.
inline double nee_value_for_hit(const std::vector<light> &lights,
                                const std::shared_ptr<material> &mat,
                                const vec3 &hit_point, const vec3 &prev_point,
                                double time) {
    vec3 to_hit = hit_point - prev_point;
    double dist = to_hit.length();
    if (dist <= 0)
        return 0.0;
    vec3 wi = to_hit / dist;
    for (const auto &lt : lights) {
        if (light_mat(lt) == mat) {
            double cosA =
                fabs(dot(light_normal_at(lt, hit_point, time), -wi));
            double A = light_area(lt);
            if (cosA > 0 && A > 0)
                return dist * dist / ((double)lights.size() * A * cosA);
            return 0.0;
        }
    }
    return 0.0;
}

inline double nee_value(const hittable &world, const std::vector<light> &lights,
                        const vec3 &origin, const vec3 &dir, double time) {
    hit_record rec;
    if (!world.hit(ray(origin, dir, time), 0.001, 1e30, rec))
        return 0.0;
    if (rec.mat->emitted().length_squared() <= 0)
        return 0.0;
    return nee_value_for_hit(lights, rec.mat, rec.point, origin, time);
}

// Power heuristic for one sample each (PBRT PathIntegrator): squares the
// densities before the balance ratio, cutting fireflies from rare bright hits.
inline double power_weight(double pdf_a, double pdf_b) {
    double a2 = pdf_a * pdf_a, b2 = pdf_b * pdf_b;
    double denom = a2 + b2;
    return denom > 0 ? a2 / denom : 0.0;
}

// 50/50 mixture of the two strategies, evaluated for one direction.
// Empty lights degenerate to pure cosine (not half-cosine: the sampler
// below always takes the cosine branch then, so the pdf must match).
inline double mixture_value(const hittable &world, const std::vector<light> &lights,
                            const vec3 &origin, const vec3 &dir, const vec3 &normal,
                            double time) {
    double c = cosine_value(unit_vector(dir), normal);
    if (lights.empty())
        return c;
    double l = nee_value(world, lights, origin, unit_vector(dir), time);
    return 0.5 * c + 0.5 * l;
}

// One mixture draw: direction + its mixture density. Cosine branch builds an
// ONB around the shading normal; light branch aims at a uniform light point.
inline vec3 sample_mixture(const hittable &world, const std::vector<light> &lights,
                           const vec3 &origin, const vec3 &normal, double time,
                           double &pdf_out) {
    vec3 dir;
    if (lights.empty() || random_double() < 0.5) {
        onb frame;
        frame.build_from_w(normal);
        dir = frame.local(random_cosine_direction());
    } else {
        int li = (int)(random_double() * lights.size());
        if (li >= (int)lights.size())
            li = (int)lights.size() - 1;
        const auto &light = lights[(size_t)li];
        vec3 lp = light_point(light, random_double(), random_double(), time);
        dir = unit_vector(lp - origin);
    }
    pdf_out = mixture_value(world, lights, origin, dir, normal, time);
    return dir;
}

} // namespace direction_pdf
