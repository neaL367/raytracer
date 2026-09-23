#pragma once
// Thin-film interference (M68): single lossless film on a substrate,
// unpolarized Airy formula. Evaluated at the three hero wavelengths so
// both RGB and spectral-hero paths share one routine (spectral picks the
// hero channel downstream). d = 0 reproduces the bare interface exactly.
//
// Substrate may be dielectric (k=0) or conductor (n,k). The film itself
// is lossless (real n_film). Incident medium n0 is real (air or glass).
#include "spectrum.h"
#include "vec3.h"

#include <cmath>
#include <complex>

namespace thinfilm {

// Amplitude reflectance of one film-covered interface at one wavelength.
// n0 = incident IOR (real), nf = film IOR (real), d_nm = film thickness,
// (ns_n, ns_k) = substrate, cos_ti = incident cos, lambda_nm = wavelength.
inline double film_R(double n0, double nf, double d_nm, double ns_n, double ns_k,
                     double cos_ti, double lambda_nm) {
    double c = cos_ti < 0 ? 0 : (cos_ti > 1 ? 1 : cos_ti);
    if (d_nm <= 0.0) {
        // Bare interface: reduces to plain Fresnel (exact legacy limit).
        if (ns_k == 0.0) {
            std::complex<double> as, ap;
            spectrum::fresnel_ri(n0, 0.0, ns_n, 0.0, c, as, ap);
            return 0.5 * (std::norm(as) + std::norm(ap));
        }
        return spectrum::conductor_R(n0, ns_n, ns_k, c);
    }
    double s2 = 1.0 - c * c;
    // cos in film via real Snell (film lossless, n0 <= nf assumed).
    double sin2t1 = (n0 * n0 / (nf * nf)) * s2;
    if (sin2t1 >= 1.0)
        return 1.0; // evanescent in film (exit-side edge): total reflection
    double cost1 = std::sqrt(1.0 - sin2t1);
    // r12: real Fresnel air/film at (c, cost1).
    double rs12 = (n0 * c - nf * cost1) / (n0 * c + nf * cost1);
    double rp12 = (nf * c - n0 * cost1) / (nf * c + n0 * cost1);
    // r23: film -> substrate (complex) at cos cost1.
    std::complex<double> rs23, rp23;
    spectrum::fresnel_ri(nf, 0.0, ns_n, ns_k, cost1, rs23, rp23);
    // Round-trip phase: delta = 2*pi*nf*d*cos1/lambda.
    double delta = 2.0 * 3.1415926535897932385 * nf * d_nm * cost1 / lambda_nm;
    std::complex<double> e(std::cos(2.0 * delta), std::sin(2.0 * delta));
    auto airy = [&](double r12, const std::complex<double> &r23) {
        std::complex<double> num = r12 + r23 * e;
        std::complex<double> den = 1.0 + r12 * r23 * e;
        return std::norm(num / den);
    };
    return 0.5 * (airy(rs12, rs23) + airy(rp12, rp23));
}

// RGB evaluation at the three hero wavelengths (650/550/450nm).
inline vec3 film_R_rgb(double n0, double nf, double d_nm, double ns_n, double ns_k,
                       double cos_ti) {
    vec3 R(0, 0, 0);
    for (int c = 0; c < 3; ++c)
        R.e[c] = film_R(n0, nf, d_nm, ns_n, ns_k, cos_ti, spectrum::kHeroLambda[c]);
    return R;
}

} // namespace thinfilm
