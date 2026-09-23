#pragma once
#include "material_base.h"

class dielectric : public material {
public:
    // Legacy pinhole ctor: smooth (r=0), unnested (prio=0).
    dielectric(double ri) : ir(ri), roughness(0), prio(0) {}
    // Rough Walter BTDF variant (M43): GGX roughness.
    dielectric(double ri, double r)
        : ir(ri), roughness(r < 0 ? 0 : (r > 1 ? 1 : r)), prio(0) {}
    // Nested interface (M57): priority resolves overlapping media.
    dielectric(double ri, double r, int pri)
        : ir(ri), roughness(r < 0 ? 0 : (r > 1 ? 1 : r)), prio(pri) {}
    // Absorbing glass (M58): Beer's law over the nested exit chord.
    // Cauchy B (M67): dispersion for spectral hero paths, um^2; 0 = legacy.
    dielectric(double ri, double r, int pri, const vec3 &sigma, double cauchyB = 0.0)
        : ir(ri), roughness(r < 0 ? 0 : (r > 1 ? 1 : r)), prio(pri), sigma(sigma),
          cauchyB(cauchyB < 0 ? 0 : cauchyB) {}
    int priority() const override { return prio; }
    double ior() const override { return ir; }
    double ior_at(int channel) const override {
        if (channel < 0 || channel > 2)
            return ir;
        return spectrum::cauchy_ior(ir, cauchyB, spectrum::kHeroLambda[channel]);
    }
    double dispersion() const { return cauchyB; }
    // Thin-film overcoat, entry side (M68): d_nm = 0 (default) disables.
    void set_film(double d_nm, double n_film = 1.5) {
        film_d = d_nm < 0 ? 0 : d_nm;
        film_n = n_film;
    }
    double film_thickness() const { return film_d; }
    double film_ior() const { return film_n; }
    // Film reflect prob at cos incidence, hero-resolved in spectral mode
    // (else luminance-mean). n0 = air on entry, glass on exit (approx).
    double film_prob(double cos_ti, double iri, bool front_face) const {
        double n0 = front_face ? 1.0 : iri;
        vec3 R = thinfilm::film_R_rgb(n0, film_n, film_d, iri, 0.0, cos_ti);
        int hero = spectrum::hero_channel();
        if (hero >= 0 && hero <= 2)
            return R.e[hero];
        return (R.x() + R.y() + R.z()) / 3.0;
    }
    vec3 absorb() const override { return sigma; }
    bool scatter(const ray &in, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
        if (roughness <= 0)
            return scatter_smooth(in, rec, attenuation, scattered);
        // Walter microfacet BTDF: H ~ VNDF, reflect w.p. F else refract.
        // Choice probs cancel the Fresnel: both lobes weigh G2/G1(V) <= 1.
        onb frame;
        frame.build_from_w(rec.normal);
        vec3 V = unit_vector(-in.direction());
        vec3 Vl(dot(V, frame.u), dot(V, frame.v), dot(V, frame.w));
        double alpha = ggx::alpha_of(roughness);
        vec3 H;
        ggx::vndf_sample(alpha, Vl, random_double(), random_double(), H);
        double cosVH = dot(Vl, H);
        if (cosVH <= 0)
            return false; // degenerate microfacet: absorbed
        // Hero-resolved IOR (M67): legacy ir when spectral mode is off.
        double iri = ior_at(spectrum::hero_channel());
        double eta = rec.nest_set ? rec.nest_eta : (rec.front_face ? (1.0 / iri) : iri); // n_i/n_o
        double sinT2 = eta * eta * (1.0 - cosVH * cosVH);
        double F = 0;
        if (film_d > 0) {
            F = film_prob(cosVH < 0 ? 0 : (cosVH > 1 ? 1 : cosVH), iri,
                          rec.front_face);
        } else {
            F = (sinT2 > 1.0) ? 1.0 : reflectance(fmin(cosVH, 1.0), eta);
        }
        vec3 Ll;
        if (sinT2 > 1.0 || random_double() < F) {
            Ll = H * (2.0 * cosVH) - Vl; // reflect incident (-V) about H
            if (Ll.z() <= 0)
                return false;
        } else {
            // Refract: L = eta*I + H*(eta*cosI - sqrt(k)), I = -V.
            double k = 1.0 - sinT2;
            double cosI = cosVH;
            Ll = Vl * (-eta) + H * (eta * cosI - std::sqrt(k));
            double Llen = Ll.length();
            if (Llen <= 0)
                return false;
            Ll = Ll / Llen;
            if (Ll.z() >= 0)
                return false; // transmitted lobe lives below
        }
        double w = ggx::weight_ratio(alpha, Vl.z(), fabs(Ll.z()));
        attenuation = vec3(w, w, w); // glass absorbs nothing; weight <= 1
        scattered = ray(rec.point, frame.local(Ll));
        return true;
    }
    // Legacy delta path: bit-exact pre-roughness behavior (roughness 0).
    bool scatter_smooth(const ray &in, const hit_record &rec, vec3 &attenuation,
                        ray &scattered) const {
        attenuation = vec3(1, 1, 1); // glass absorbs nothing
        double iri = ior_at(spectrum::hero_channel());
        double ratio = rec.nest_set ? rec.nest_eta : (rec.front_face ? (1.0 / iri) : iri);
        vec3 unit = unit_vector(in.direction());
        double cos_t = fmin(dot(-unit, rec.normal), 1.0);
        double sin_t = std::sqrt(1.0 - cos_t * cos_t);
        bool cannot_refract = ratio * sin_t > 1.0;
        double frefl = cannot_refract ? 1.0
                     : (film_d > 0) ? film_prob(cos_t, iri, rec.front_face)
                                    : reflectance(cos_t, ratio);
        vec3 dir = (cannot_refract || frefl > random_double())
                        ? reflect(unit, rec.normal)
                        : refract(unit, rec.normal, ratio);
        scattered = ray(rec.point, dir);
        return true;
    }
    bool export_gpu(float alb[4], float alb2[4], float emit[4],
                    float prm[4]) const override {
        alb[0] = alb[1] = alb[2] = 0;
        alb2[0] = alb2[1] = alb2[2] = 0;
        emit[0] = emit[1] = emit[2] = 0;
        prm[0] = static_cast<float>(MatType::GLASS);
        prm[1] = (float)roughness; // 0 = legacy delta path, bit-exact
        prm[2] = (float)ir;
        prm[3] = (float)prio; // nesting priority (M57); 0 = legacy
        alb[0] = (float)sigma.x(); // absorption sigma (M58); 0 = clear
        alb[1] = (float)sigma.y();
        alb[2] = (float)sigma.z();
        alb2[0] = (float)cauchyB; // Cauchy B in um^2 (M67); 0 = no dispersion
        alb2[1] = (float)film_d; // thin-film thickness nm (M68); 0 = off
        alb2[2] = (float)film_n; // film IOR (M68)
        return true;
    }

private:
    double ir;
    double roughness;
    int prio = 0; // nesting priority (M57); 0 = legacy unnested
    vec3 sigma{0, 0, 0}; // absorption (M58); 0 = clear glass
    double cauchyB = 0.0; // dispersion (M67); 0 = constant IOR
    double film_d = 0.0; // thin-film thickness in nm (M68); 0 = off
    double film_n = 1.5; // thin-film IOR (M68)
    // Schlick approx: grazing -> mirror, normal -> ~4% for glass.
    static double reflectance(double cos, double ref_idx) {
        double r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow(1 - cos, 5);
    }
};
