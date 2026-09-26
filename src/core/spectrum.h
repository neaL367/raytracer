#pragma once
// Spectral utilities: channel wavelengths (thin-film RGB + n/k presets),
// Cauchy dispersion, and the tiny complex Fresnel core. Transport is RGB
// only (no hero-wavelength mode): materials evaluate wavelength-dependent
// physics per channel or at luminance mean.
#include "vec3.h"

#include <cmath>
#include <complex>
#include <algorithm>

namespace spectrum {

// Channel wavelengths in nm, aligned to R/G/B channels 0/1/2.
inline constexpr double kChannelLambda[3] = {650.0, 550.0, 450.0};

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

// Tanner-Helland / Mitchell Charity blackbody radiation approximation to linear RGB
inline vec3 blackbody_to_rgb(double kelvin) {
    kelvin = std::clamp(kelvin, 1000.0, 40000.0);
    double temp = kelvin / 100.0;
    double red, green, blue;

    // Red
    if (temp <= 66.0) {
        red = 255.0;
    } else {
        red = temp - 60.0;
        red = 329.698727446 * std::pow(red, -0.1332047592);
        red = std::clamp(red, 0.0, 255.0);
    }

    // Green
    if (temp <= 66.0) {
        green = temp;
        green = 99.4708025861 * std::log(green) - 161.1195681661;
        green = std::clamp(green, 0.0, 255.0);
    } else {
        green = temp - 60.0;
        green = 288.1221695283 * std::pow(green, -0.0755148492);
        green = std::clamp(green, 0.0, 255.0);
    }

    // Blue
    if (temp >= 66.0) {
        blue = 255.0;
    } else if (temp <= 19.0) {
        blue = 0.0;
    } else {
        blue = temp - 10.0;
        blue = 138.5177312231 * std::log(blue) - 305.0447927307;
        blue = std::clamp(blue, 0.0, 255.0);
    }

    // Convert sRGB [0..1] to linear RGB
    auto to_lin = [](double c) {
        return (c <= 0.04045) ? (c / 12.92) : std::pow((c + 0.055) / 1.055, 2.4);
    };
    vec3 col(to_lin(red / 255.0), to_lin(green / 255.0), to_lin(blue / 255.0));
    double mx = std::max({col.x(), col.y(), col.z()});
    return (mx > 0.0) ? (col / mx) : vec3(1, 1, 1);
}

} // namespace spectrum
