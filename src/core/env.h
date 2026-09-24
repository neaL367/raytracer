#pragma once
// Analytic environment light (M38 v1): legacy sky gradient + a small sun
// disc, sampled uniformly on the sphere (pdf 1/4PI) with power-heuristic MIS
// against the BSDF. Opt-in via scene flag; off = legacy gradient miss,
// byte-exact. Mirrored in path.comp (env_radiance/env_sample).
//
// M63: extended to support an equirectangular HDRI map (hdri_env). When
// g_hdri is set, radiance/sample/pdf delegate to it; nenv carries the mode:
//   nenv == 0 → off (legacy gradient, no MIS)
//   nenv == 1 → analytic sun+sky (original behaviour, byte-exact)
//   nenv == 2 → HDRI importance sampling (new path)
// The analytic path is untouched; all anchors remain valid.
#include "vec3.h"
#include "hdri.h"

#include <cmath>
#include <algorithm>

namespace env_light {

// Thread-local pointer set by the application before rendering.
// Null = analytic mode. The object's lifetime must exceed the render.
inline hdri_env *&g_hdri() {
    static thread_local hdri_env *p = nullptr;
    return p;
}

inline vec3 &custom_sun_dir() {
    static vec3 s_sun = unit_vector(vec3(0.5, 0.8, 0.35));
    return s_sun;
}
inline vec3 sun_dir() { return custom_sun_dir(); }
inline void set_sun_dir(const vec3 &d) { custom_sun_dir() = unit_vector(d); }
inline void get_sun_angles(double &az_deg, double &el_deg) {
    vec3 d = custom_sun_dir();
    double el = std::asin(std::clamp(d.y(), -1.0, 1.0));
    double az = std::atan2(d.x(), d.z());
    el_deg = el * (180.0 / 3.1415926535897932385);
    az_deg = az * (180.0 / 3.1415926535897932385);
    if (az_deg < 0.0) az_deg += 360.0;
}
inline void set_sun_angles(double az_deg, double el_deg) {
    double az = az_deg * (3.1415926535897932385 / 180.0);
    double el = el_deg * (3.1415926535897932385 / 180.0);
    double cos_el = std::cos(el);
    custom_sun_dir() = unit_vector(vec3(cos_el * std::sin(az), std::sin(el), cos_el * std::cos(az)));
}

// Legacy background gradient, now the single source for both paths.
inline vec3 sky(const vec3 &d) {
    vec3 unit = unit_vector(d);
    double t = 0.5 * (unit.y() + 1.0);
    return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

// Sky plus sun disc (quadratic ramp across the rim, ~30x white at center).
inline vec3 analytic_radiance(const vec3 &d) {
    vec3 ud = unit_vector(d);
    vec3 L = sky(ud);
    double s = dot(ud, sun_dir());
    if (s > 0.9993) {
        double k = (s - 0.9993) / (1.0 - 0.9993);
        L = L + vec3(30.0, 25.0, 18.0) * (k * k);
    }
    return L;
}

// Dispatch: HDRI if loaded, else analytic sky+sun.
inline vec3 radiance(const vec3 &d) {
    hdri_env *h = g_hdri();
    if (h && !h->empty())
        return h->radiance(d);
    return analytic_radiance(d);
}

// Analytic-only alias (kept for legacy sky miss branch without --env).
inline vec3 legacy_sky(const vec3 &d) { return sky(d); }

// Uniform-sphere sample PDF (1/4π) — used by the analytic path.
inline double analytic_sample_pdf() { return 1.0 / (4.0 * 3.1415926535897932385); }

// Effective PDF for a given direction (used by the MIS miss weight).
// Returns the HDRI pdf when active, else 1/4π.
inline double sample_pdf_for(const vec3 &d) {
    hdri_env *h = g_hdri();
    if (h && !h->empty())
        return h->pdf(d);
    return analytic_sample_pdf();
}

// Legacy scalar accessor kept for existing call sites that don't need
// direction-dependent pdf (analytic-only).
inline double sample_pdf() { return analytic_sample_pdf(); }

// Uniform sphere direction from 2 draws (mirrors light_point sphere branch).
inline vec3 analytic_sample_dir(double u1, double u2) {
    double z   = 1.0 - 2.0 * u1;
    double phi = 2.0 * 3.1415926535897932385 * u2;
    double r   = std::sqrt(std::max(1.0 - z * z, 0.0));
    return vec3(r * std::cos(phi), r * std::sin(phi), z);
}

// Sample a direction from the active env (HDRI or analytic sphere).
// Returns the direction; sets pdf_out to the solid-angle pdf.
inline vec3 sample_dir(double u1, double u2, double &pdf_out) {
    hdri_env *h = g_hdri();
    if (h && !h->empty()) {
        auto [d, p] = h->sample(u1, u2);
        pdf_out = p;
        return d;
    }
    pdf_out = analytic_sample_pdf();
    return analytic_sample_dir(u1, u2);
}

// Legacy two-arg overload (analytic-only callers; pdf discarded).
inline vec3 sample_dir(double u1, double u2) {
    double dummy;
    return sample_dir(u1, u2, dummy);
}

} // namespace env_light
