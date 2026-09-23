#pragma once
#include "material_base.h"
#include "../core/onb.h"
#include "../core/ggx.h"
#include <algorithm>
#include <cmath>

// Disney Principled BRDF (Brent Burley / SIGGRAPH 2012 / Frostbite PBR)
// Unifies:
// - Diffuse retro-reflection + optional subsurface flat falloff
// - Metallic & dielectric specular with height-correlated Smith GGX VNDF
// - Specular tint towards base color
// - Velvet / cloth grazing sheen
// - Secondary clearcoat reflection layer with GTR1 distribution
class disney_material : public material {
public:
    vec3 base_color{0.8, 0.8, 0.8};
    double metallic = 0.0;
    double roughness = 0.5;
    double specular = 0.5;
    double specular_tint = 0.0;
    double sheen = 0.0;
    double sheen_tint = 0.5;
    double clearcoat = 0.0;
    double clearcoat_gloss = 1.0;
    double subsurface = 0.0;

    disney_material() = default;
    explicit disney_material(const vec3 &color, double metal = 0.0, double rough = 0.5)
        : base_color(color), metallic(metal), roughness(rough) {}

    // Builder pattern methods for clean ergonomics
    disney_material &set_metallic(double m) { metallic = m; return *this; }
    disney_material &set_roughness(double r) { roughness = r; return *this; }
    disney_material &set_specular(double s) { specular = s; return *this; }
    disney_material &set_specular_tint(double st) { specular_tint = st; return *this; }
    disney_material &set_sheen(double sh, double sht = 0.5) { sheen = sh; sheen_tint = sht; return *this; }
    disney_material &set_clearcoat(double cc, double gloss = 1.0) { clearcoat = cc; clearcoat_gloss = gloss; return *this; }
    disney_material &set_subsurface(double ss) { subsurface = ss; return *this; }

    bool is_diffuse() const override {
        // True if there is a non-negligible diffuse component for NEE
        return metallic < 0.99;
    }

    vec3 surface_albedo(const hit_record &rec) const override {
        (void)rec;
        return base_color;
    }

    double direction_pdf(const vec3 &scattered_dir, const hit_record &rec) const override {
        const double pi_val = 3.1415926535897932385;
        vec3 n = rec.normal;
        vec3 wo = unit_vector(scattered_dir);
        double cos_theta = dot(n, wo);
        if (cos_theta <= 0.0) return 0.0;
        return cos_theta / pi_val;
    }

    bool scatter(const ray &in, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
        const double pi_val = 3.1415926535897932385;
        vec3 n = rec.normal;
        vec3 V = -unit_vector(in.direction());
        if (dot(n, V) < 0.0)
            n = -n;

        onb uvw;
        uvw.build_from_w(n);
        vec3 Vl = vec3(dot(V, uvw.u), dot(V, uvw.v), dot(V, uvw.w));
        if (Vl.z() <= 1e-6)
            return false;

        // Luminance and tint calculation
        double lum = 0.3 * base_color.x() + 0.6 * base_color.y() + 0.1 * base_color.z();
        vec3 c_tint = lum > 0.0 ? base_color / lum : vec3(1.0, 1.0, 1.0);
        vec3 c_spec0 = (1.0 - metallic) * (0.08 * specular * ((1.0 - specular_tint) * vec3(1.0, 1.0, 1.0) + specular_tint * c_tint)) +
                       metallic * base_color;
        vec3 c_sheen = (1.0 - sheen_tint) * vec3(1.0, 1.0, 1.0) + sheen_tint * c_tint;

        // Lobe selection weights
        double w_diff = (1.0 - metallic) * 0.5;
        double w_spec = metallic + (1.0 - metallic) * 0.5;
        double w_coat = 0.25 * clearcoat;
        double w_sum = w_diff + w_spec + w_coat;
        if (w_sum <= 1e-8)
            return false;

        double xi = random_double() * w_sum;
        vec3 L;

        if (xi < w_diff) {
            // Diffuse lobe (Burley retro-reflective + optional subsurface + sheen)
            vec3 cd = random_cosine_direction();
            vec3 Ll = cd;
            L = uvw.local(Ll);

            vec3 H = unit_vector(V + L);
            double cos_theta_d = std::max(dot(V, H), 0.0);
            double cos_theta_l = std::max(Ll.z(), 0.0);
            double cos_theta_v = std::max(Vl.z(), 0.0);

            double fd90 = 0.5 + 2.0 * roughness * cos_theta_d * cos_theta_d;
            double fl = 1.0 + (fd90 - 1.0) * std::pow(std::max(1.0 - cos_theta_l, 0.0), 5.0);
            double fv = 1.0 + (fd90 - 1.0) * std::pow(std::max(1.0 - cos_theta_v, 0.0), 5.0);
            vec3 fd = base_color * (fl * fv);

            // Optional Subsurface approximation blend
            if (subsurface > 0.0) {
                double fss90 = roughness * cos_theta_d * cos_theta_d;
                double fss_l = 1.0 + (fss90 - 1.0) * std::pow(std::max(1.0 - cos_theta_l, 0.0), 5.0);
                double fss_v = 1.0 + (fss90 - 1.0) * std::pow(std::max(1.0 - cos_theta_v, 0.0), 5.0);
                vec3 fss = base_color * (1.25 * (fss_l * fss_v * (1.0 / (cos_theta_l + cos_theta_v + 1e-5) - 0.5) + 0.5));
                fd = (1.0 - subsurface) * fd + subsurface * fss;
            }

            // Sheen term
            vec3 f_sheen = sheen * c_sheen * std::pow(std::max(1.0 - cos_theta_d, 0.0), 5.0);

            attenuation = (fd * (1.0 - metallic) + f_sheen) * (w_sum / w_diff);
        } else if (xi < w_diff + w_spec) {
            // Specular GGX lobe (VNDF sampling)
            double alpha = roughness * roughness;
            alpha = std::clamp(alpha, 0.001, 1.0);

            vec3 H_local;
            vec3 Ll = ggx::vndf_sample(alpha, Vl, random_double(), random_double(), H_local);
            if (Ll.z() <= 0.0)
                return false;

            L = uvw.local(Ll);
            double cos_vh = std::max(dot(Vl, H_local), 0.0);
            vec3 F = c_spec0 + (vec3(1.0, 1.0, 1.0) - c_spec0) * std::pow(std::max(1.0 - cos_vh, 0.0), 5.0);

            double ratio = ggx::weight_ratio(alpha, Vl.z(), Ll.z());
            attenuation = F * ratio * (w_sum / w_spec);
        } else {
            // Clearcoat lobe (GTR1 distribution)
            double gloss = std::clamp(clearcoat_gloss, 0.0, 1.0);
            double alpha_g = (1.0 - gloss) * 0.1 + gloss * 0.001;
            double a2 = alpha_g * alpha_g;

            // GTR1 inversion
            double u1 = random_double(), u2 = random_double();
            double cos_theta_h = std::sqrt(std::max(0.0, (1.0 - std::pow(a2, 1.0 - u1)) / (1.0 - a2)));
            double sin_theta_h = std::sqrt(std::max(0.0, 1.0 - cos_theta_h * cos_theta_h));
            double phi_h = 2.0 * pi_val * u2;

            vec3 H_local(sin_theta_h * std::cos(phi_h), sin_theta_h * std::sin(phi_h), cos_theta_h);
            vec3 Ll = 2.0 * dot(Vl, H_local) * H_local - Vl;
            if (Ll.z() <= 0.0)
                return false;

            L = uvw.local(Ll);
            double cos_vh = std::max(dot(Vl, H_local), 0.0);
            double F_coat = 0.04 + (1.0 - 0.04) * std::pow(std::max(1.0 - cos_vh, 0.0), 5.0);

            attenuation = vec3(1.0, 1.0, 1.0) * (0.25 * clearcoat * F_coat * (w_sum / w_coat));
        }

        scattered = ray(rec.point, L, in.time());
        return true;
    }

    bool export_gpu(float alb[4], float alb2[4], float emit[4],
                    float prm[4]) const override {
        alb[0] = static_cast<float>(base_color.x());
        alb[1] = static_cast<float>(base_color.y());
        alb[2] = static_cast<float>(base_color.z());
        alb[3] = static_cast<float>(metallic);

        alb2[0] = static_cast<float>(specular);
        alb2[1] = static_cast<float>(specular_tint);
        alb2[2] = static_cast<float>(sheen);
        alb2[3] = static_cast<float>(sheen_tint);

        emit[0] = static_cast<float>(clearcoat);
        emit[1] = static_cast<float>(clearcoat_gloss);
        emit[2] = static_cast<float>(subsurface);
        emit[3] = 0.0f;

        prm[0] = static_cast<float>(MatType::DISNEY);
        prm[1] = static_cast<float>(roughness);
        prm[2] = 1.5f; // default IOR
        prm[3] = 0.0f;
        return true;
    }
};
