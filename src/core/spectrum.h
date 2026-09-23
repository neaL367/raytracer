#pragma once
// Hero-wavelength spectral transport (M67): discrete 3-channel heroes.
//
// Instead of RGB throughput with one shared IOR, a spectral path carries a
// single hero wavelength (one of 650/550/450nm, drawn uniformly per path)
// and evaluates all wavelength-dependent physics (Cauchy dispersion,
// complex conductor Fresnel, thin-film phase) at that wavelength. RGB
// material colors are transported channel-wise: sampling hero channel c
// with probability 1/3 and accumulating 3x into channel c is an unbiased
// estimator of the legacy RGB image when dispersion is zero (B=0, no n/k,
// no film). Legacy mode (hero off) is bit-exact: no new RNG draws, no
// changed code paths.
//
// The discrete heroes keep film/GPU mirrors cheap (3 evaluations) while
// showing real prismatic splits across paths. Continuous-lambda + CMF
// accumulation is the documented follow-up, not this milestone.
#include "vec3.h"

#include <cmath>
#include <complex>

namespace spectrum {

// Hero wavelengths in nm, aligned to R/G/B channels 0/1/2.
inline constexpr double kHeroLambda[3] = {650.0, 550.0, 450.0};

// Active hero channel for the current path, or -1 when spectral mode is
// off. Set by the integrator around Li(); materials read it in scatter().
// Thread-local like rng_fixed_flag/mip_render_h (same lifetime rules).
inline int &hero_channel() {
    static thread_local int c = -1;
    return c;
}

// Channel-pick: identity when spectral is off, else the hero channel
// value with other channels zeroed. NO sampling weight here: the 1/3
// hero probability is compensated once, by scaling the finished path
// contribution x3 at the end of Li(). (Weighting per quantity would
// compound to 9x in two-factor terms like throughput x emission.)
inline vec3 pick(const vec3 &v, int hero, bool spectral) {
    if (!spectral || hero < 0 || hero > 2)
        return v;
    vec3 z(0, 0, 0);
    z.e[hero] = v.e[hero];
    return z;
}

// Cauchy dispersion: n(l) = n_ref + B*(1/l^2 - 1/l_ref^2), l in um.
// B = 0 (default) reproduces the legacy constant IOR exactly.
inline double cauchy_ior(double n_ref, double B, double lambda_nm,
                         double ref_nm = 587.6) {
    if (B == 0.0)
        return n_ref;
    double l = lambda_nm / 1000.0; // nm -> um
    double lr = ref_nm / 1000.0;
    return n_ref + B * (1.0 / (l * l) - 1.0 / (lr * lr));
}

// ---- Tiny complex core (mirrored in GLSL with vec2) ----
// Used by exact conductor Fresnel and the thin-film Airy formula so both
// share one tested arithmetic. std::complex here; GLSL mirrors it by hand.

inline double cabs2(const std::complex<double> &z) { return std::norm(z); }

// Exact unpolarized Fresnel reflectance for dielectric(n0) -> conductor
// (n,k) at cos_theta_i. n0 = 1 for air; film stacks call this per interface
// via the amplitude version below.
inline double conductor_R(double n0, double n, double k, double cos_ti) {
    double c = cos_ti < 0 ? 0 : (cos_ti > 1 ? 1 : cos_ti);
    double s2 = 1.0 - c * c;
    std::complex<double> N(n, k);
    std::complex<double> N2 = N * N;
    // cos_tt^2 = 1 - (n0^2 s2)/N^2 (complex Snell).
    std::complex<double> cos2t = 1.0 - (n0 * n0 * s2) / N2;
    std::complex<double> cost = std::sqrt(cos2t);
    std::complex<double> rs = (n0 * c - N * cost) / (n0 * c + N * cost);
    std::complex<double> rp = (N * c - n0 * cost) / (N * c + n0 * cost);
    return 0.5 * (cabs2(rs) + cabs2(rp));
}

// Amplitude (complex) Fresnel pair for one interface, for film stacks.
inline void fresnel_ri(double n0_re, double n0_im, double n1_re, double n1_im,
                       double cos_ti, std::complex<double> &rs,
                       std::complex<double> &rp) {
    double c = cos_ti < 0 ? 0 : (cos_ti > 1 ? 1 : cos_ti);
    double s2 = 1.0 - c * c;
    std::complex<double> N0(n0_re, n0_im), N1(n1_re, n1_im);
    std::complex<double> cos2t = 1.0 - (N0 * N0 * s2) / (N1 * N1);
    std::complex<double> cost = std::sqrt(cos2t);
    rs = (N0 * c - N1 * cost) / (N0 * c + N1 * cost);
    rp = (N1 * c - N0 * cost) / (N1 * c + N0 * cost);
}

// ---- Measured conductor data (approximate educational values) ----
// n/k at the three hero wavelengths (650/550/450nm), after Johnson &
// Christy / Palik trends. Approximate: good for beetle-wing hues, not
// for metrology. Preset 0 = none (legacy Schlick-from-albedo).
inline bool conductor_nk(int preset, int channel, double &n, double &k) {
    if (channel < 0 || channel > 2)
        return false;
    // Rows: Au, Ag, Cu, Al. Columns: 650nm, 550nm, 450nm.
    static const double nn[4][3] = {
        {0.17, 0.35, 1.45}, // Au
        {0.07, 0.13, 0.16}, // Ag
        {0.28, 0.95, 1.25}, // Cu
        {1.45, 0.95, 0.65}, // Al
    };
    static const double kk[4][3] = {
        {3.80, 2.75, 1.95}, // Au
        {4.20, 4.00, 2.60}, // Ag
        {3.40, 2.65, 2.40}, // Cu
        {7.60, 6.00, 5.30}, // Al
    };
    if (preset < 1 || preset > 4)
        return false;
    n = nn[preset - 1][channel];
    k = kk[preset - 1][channel];
    return true;
}

} // namespace spectrum
