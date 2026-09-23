#pragma once
#include "material_base.h"

class metal : public material {
public:
    // GGX conductor: F0 = albedo, perceptual roughness (alpha = r^2).
    // Roughness 0 = delta mirror. VNDF sampling, weight F*G2/G1(V).
    metal(const vec3 &a, double r) : albedo(a), roughness(r < 0 ? 0 : (r > 1 ? 1 : r)) {}
    // Measured-data conductor (M67): n/k preset 1..4 (Au/Ag/Cu/Al, approx),
    // exact complex Fresnel per hero channel instead of Schlick-from-F0.
    // Works in RGB mode too (full 3-channel evaluation, no stream change).
    metal(int nk_preset, double r)
        : roughness(r < 0 ? 0 : (r > 1 ? 1 : r)), nk_id(nk_preset) {}
    // Anisotropic variant: per-axis roughness, TBN frame from hit tangent.
    // Falls back to the ONB-u frame when the tangent is missing/degenerate.
    metal(const vec3 &a, double rx, double ry)
        : albedo(a), roughness(-1), rough_x(rx < 0 ? 0 : (rx > 1 ? 1 : rx)),
          rough_y(ry < 0 ? 0 : (ry > 1 ? 1 : ry)) {}
    metal(int nk_preset, double rx, double ry)
        : roughness(-1), rough_x(rx < 0 ? 0 : (rx > 1 ? 1 : rx)),
          rough_y(ry < 0 ? 0 : (ry > 1 ? 1 : ry)), nk_id(nk_preset) {}
    bool is_aniso() const { return roughness < 0; }
    int nk_preset() const { return nk_id; }
    // Thin-film overcoat (M68): d_nm = 0 (default) disables. Set before
    // sharing; draw-free so RGB streams stay byte-exact.
    void set_film(double d_nm, double n_film = 1.5) {
        film_d = d_nm < 0 ? 0 : d_nm;
        film_n = n_film;
    }
    double film_thickness() const { return film_d; }
    double film_ior() const { return film_n; }
    // Exact spectral reflectance at the three hero wavelengths for n/k
    // metals (draw-free, so RGB-mode streams stay byte-exact).
    vec3 spectral_reflectance(double cos_vh) const {
        vec3 R(0, 0, 0);
        for (int c = 0; c < 3; ++c) {
            double n = 0, k = 0;
            if (spectrum::conductor_nk(nk_id, c, n, k))
                R.e[c] = spectrum::conductor_R(1.0, n, k, cos_vh);
        }
        return R;
    }
    // Film-modulated reflectance (M68): Airy overcoat on the substrate,
    // evaluated per hero channel. Substrate = measured n/k when present,
    // else F0-as-dielectric-n (documented approximation).
    vec3 film_reflectance(double cos_vh) const {
        if (nk_id > 0) {
            vec3 R(0, 0, 0);
            for (int c = 0; c < 3; ++c) {
                double n = 0, k = 0;
                spectrum::conductor_nk(nk_id, c, n, k);
                R.e[c] = thinfilm::film_R(1.0, film_n, film_d, n, k, cos_vh,
                                          spectrum::kHeroLambda[c]);
            }
            return R;
        }
        // Legacy F0 metal: substrate n from R0, k = 0 (documented approx).
        double r0 = (albedo.x() + albedo.y() + albedo.z()) / 3.0;
        double sq = r0 < 0 ? 0 : std::sqrt(r0 > 1 ? 1 : r0);
        double sub_n = (1.0 + sq) / (1.0 - sq + 1e-6);
        return thinfilm::film_R_rgb(1.0, film_n, film_d, sub_n, 0.0, cos_vh);
    }
    bool scatter(const ray &in, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
        if (is_aniso())
            return scatter_aniso(in, rec, attenuation, scattered);
        if (roughness <= 0) {
            // Delta mirror path: Schlick or measured n/k along the single
            // reflection direction. Film wraps both (M68).
            vec3 dir = reflect(unit_vector(in.direction()), rec.normal);
            if (dot(dir, rec.normal) <= 0)
                return false;
            double cos_v = std::max(dot(-unit_vector(in.direction()), rec.normal), 0.0);
            vec3 F = (film_d > 0) ? film_reflectance(cos_v)
                     : (nk_id > 0) ? spectral_reflectance(cos_v)
                                   : ggx::fresnel_schlick(albedo, cos_v);
            attenuation = F;
            scattered = ray(rec.point, dir);
            return true;
        }
        onb frame;
        frame.build_from_w(rec.normal);
        vec3 V = unit_vector(-in.direction());
        vec3 Vl(dot(V, frame.u), dot(V, frame.v), dot(V, frame.w));
        double alpha = ggx::alpha_of(roughness);
        vec3 H;
        vec3 Ll = ggx::vndf_sample(alpha, Vl, random_double(), random_double(), H);
        if (Ll.z() <= 0)
            return false; // below-surface lobe: absorbed (as before)
        double cos_vh = std::max(dot(Vl, H), 0.0);
        double ratio = ggx::weight_ratio(alpha, Vl.z(), Ll.z());
        vec3 F = (film_d > 0) ? film_reflectance(cos_vh)
                 : (nk_id > 0) ? spectral_reflectance(cos_vh)
                               : ggx::fresnel_schlick(albedo, cos_vh);
        attenuation = F * ratio;
        scattered = ray(rec.point, frame.local(Ll));
        return true;
    }
    // Anisotropic scatter: T from the hit tangent (orthonormalized vs N),
    // ONB-u fallback by the same rule the GPU mirrors.
    bool scatter_aniso(const ray &in, const hit_record &rec, vec3 &attenuation,
                       ray &scattered) const {
        vec3 N = rec.normal;
        vec3 T = rec.has_tangent ? rec.tangent - N * dot(rec.tangent, N) : vec3(0, 0, 0);
        if (T.length_squared() <= 1e-12) {
            onb frame;
            frame.build_from_w(N);
            T = frame.u;
        } else {
            T = unit_vector(T);
        }
        vec3 B = cross(N, T);
        vec3 V = unit_vector(-in.direction());
        vec3 Vl(dot(V, T), dot(V, B), dot(V, N));
        double ax = ggx::alpha_of(rough_x), ay = ggx::alpha_of(rough_y);
        vec3 H;
        vec3 Ll = ggx::vndf_aniso(ax, ay, Vl, random_double(), random_double(), H);
        if (Ll.z() <= 0)
            return false;
        double cos_vh = std::max(dot(Vl, H), 0.0);
        double ratio = ggx::weight_ratio_aniso(ax, ay, Vl, Ll);
        vec3 F = (film_d > 0) ? film_reflectance(cos_vh)
                 : (nk_id > 0) ? spectral_reflectance(cos_vh)
                               : ggx::fresnel_schlick(albedo, cos_vh);
        attenuation = F * ratio;
        scattered = ray(rec.point, T * Ll.x() + B * Ll.y() + N * Ll.z());
        return true;
    }
    vec3 surface_albedo(const hit_record &) const override {
        return (nk_id > 0) ? spectral_reflectance(1.0) : albedo;
    }
    bool export_gpu(float alb[4], float alb2[4], float emit[4],
                    float prm[4]) const override {
        alb[0] = (float)albedo.x();
        alb[1] = (float)albedo.y();
        alb[2] = (float)albedo.z();
        alb2[0] = (float)film_d; // thin-film thickness nm (M68); 0 = off
        alb2[1] = (float)film_n; // film IOR (M68)
        alb2[2] = (float)nk_id; // measured n/k preset (M67); 0 = legacy F0
        emit[0] = emit[1] = emit[2] = 0;
        if (is_aniso()) {
            prm[0] = static_cast<float>(MatType::ANISO); // anisotropic conductor (ax, ay)
            prm[1] = (float)rough_x;
            prm[2] = (float)rough_y;
        } else {
            prm[0] = static_cast<float>(MatType::CONDUCTOR); // GGX conductor (fuzz-era type 1 deleted)
            prm[1] = (float)roughness;
            prm[2] = 0;
        }
        prm[3] = 0;
        return true;
    }

private:
    vec3 albedo{1, 1, 1};
    double roughness = 0; // >=0 for isotropic; <0 flags anisotropic variant
    double rough_x = 0;
    double rough_y = 0;
    int nk_id = 0; // 0 = albedo-based Schlick, 1..4 = Au/Ag/Cu/Al measured n/k
    double film_d = 0.0; // thin-film thickness in nm (M68); 0 = off
    double film_n = 1.5; // thin-film IOR (M68)
};
