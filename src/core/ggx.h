#pragma once
#include "vec3.h"

#include <cmath>

// Trowbridge-Reitz (GGX) microfacet core: NDF + height-correlated Smith
// + Schlick Fresnel + Heitz VNDF sampling. RNG-explicit (u1, u2 in),
// so unit tests own the math and CPU/GPU mirrors stay honest.
// All vectors in the shading frame (+z = normal), V/L above surface.
namespace ggx {

inline double alpha_of(double roughness) {
    double r = roughness < 0 ? 0 : (roughness > 1 ? 1 : roughness);
    return r * r; // perceptual (Disney): alpha = r^2
}

// NDF: a^2 / (PI * ((cos_h^2)(a^2 - 1) + 1)^2).
inline double D(double alpha, double cos_h) {
    double a2 = alpha * alpha;
    double d = cos_h * cos_h * (a2 - 1.0) + 1.0;
    return a2 / (3.1415926535897932385 * d * d);
}

// Smith single-direction lambda. Stable at alpha=0 (returns 0) and
// cos=1 (tan=0); caller keeps cos > 0.
inline double lambda(double alpha, double cos_w) {
    double t = std::sqrt(std::max(1.0 - cos_w * cos_w, 0.0)) / cos_w;
    double a = alpha * t;
    return (std::sqrt(1.0 + a * a) - 1.0) * 0.5;
}

// Height-correlated G2 over G1(V): (1+L(V)) / (1+L(V)+L(L)).
// The VNDF pdf cancels the NDF, leaving weight = F * this ratio.
inline double weight_ratio(double alpha, double cos_v, double cos_l) {
    double lv = lambda(alpha, cos_v);
    double ll = lambda(alpha, cos_l);
    return (1.0 + lv) / (1.0 + lv + ll);
}

inline vec3 fresnel_schlick(const vec3 &F0, double cos_vh) {
    double f = 1.0 - cos_vh;
    double f5 = f * f * f * f * f;
    return F0 + (vec3(1, 1, 1) - F0) * f5;
}

// Heitz VNDF: visible-normal sampling, 2 uniforms. Degenerates cleanly
// at alpha=0 (Ne collapses to +z regardless of the disk draw).
inline vec3 vndf_sample(double alpha, const vec3 &V, double u1, double u2,
                        vec3 &H) {
    const double pi = 3.1415926535897932385;
    vec3 Vh = unit_vector(vec3(alpha * V.x(), alpha * V.y(), V.z()));
    double lensq = Vh.x() * Vh.x() + Vh.y() * Vh.y();
    vec3 T1 = lensq > 0 ? vec3(-Vh.y(), Vh.x(), 0.0) / std::sqrt(lensq)
                        : vec3(1, 0, 0);
    vec3 T2 = cross(Vh, T1);
    double r = std::sqrt(u1);
    double phi = 2.0 * pi * u2;
    double t1 = r * std::cos(phi);
    double t2 = r * std::sin(phi);
    double s = 0.5 * (1.0 + Vh.z());
    t2 = (1.0 - s) * std::sqrt(std::max(1.0 - t1 * t1, 0.0)) + s * t2;
    vec3 Nh = T1 * t1 + T2 * t2 + Vh * std::sqrt(std::max(1.0 - t1 * t1 - t2 * t2, 0.0));
    H = unit_vector(vec3(alpha * Nh.x(), alpha * Nh.y(), std::max(Nh.z(), 0.0)));
    vec3 L = H * (2.0 * dot(V, H)) - V; // reflect incident (-V) about H
    return L;
}

} // namespace ggx
