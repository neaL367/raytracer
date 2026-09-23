#pragma once
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/random.h"
#include "../core/sampler.h"
#include "../core/onb.h"
#include "../core/ggx.h"
#include "../core/spectrum.h"
#include "../core/thinfilm.h"
#include "../core/texture.h"
#include "../geometry/hittable.h"
#include <memory>

enum class MatType : int {
    SOLID     = 0,
    RESERVED  = 1, // fuzz-era deleted; do not renumber
    GLASS     = 2,
    EMIT      = 3,
    CHECKER   = 4,
    IMAGE     = 5,
    FOG       = 6,
    CONDUCTOR = 7,
    HET       = 8,
    NOISE     = 9,
    ANISO     = 10
};

inline constexpr MatType kAllActiveMatTypes[] = {
    MatType::SOLID,
    MatType::GLASS,
    MatType::EMIT,
    MatType::CHECKER,
    MatType::IMAGE,
    MatType::FOG,
    MatType::CONDUCTOR,
    MatType::HET,
    MatType::NOISE,
    MatType::ANISO
};

inline const char *mat_type_name(MatType t) {
    switch (t) {
        case MatType::SOLID:     return "SOLID";
        case MatType::RESERVED:  return "RESERVED";
        case MatType::GLASS:     return "GLASS";
        case MatType::EMIT:      return "EMIT";
        case MatType::CHECKER:   return "CHECKER";
        case MatType::IMAGE:     return "IMAGE";
        case MatType::FOG:       return "FOG";
        case MatType::CONDUCTOR: return "CONDUCTOR";
        case MatType::HET:       return "HET";
        case MatType::NOISE:     return "NOISE";
        case MatType::ANISO:     return "ANISO";
    }
    return "UNKNOWN";
}

// Material answers scatter only. No light/traversal knowledge.
// Returns false = ray absorbed (killed, contributes black).
// emitted() default black; diffuse_light overrides (no scatter).
class material {
public:
    virtual ~material() = default;
    virtual bool scatter(const ray &in, const hit_record &rec,
                         vec3 &attenuation, ray &scattered) const = 0;
    virtual vec3 emitted() const { return vec3(0, 0, 0); }
    // UV-aware emission: textured emitters override. The no-arg form is for
    // flat emitters and legacy callers; textured emission needs a hit.
    virtual vec3 emitted(const hit_record &rec) const {
        (void)rec;
        return emitted();
    }
    virtual bool is_emissive() const { return false; }
    // NEE applies to diffuse only; specular paths skip explicit lights.
    virtual bool is_diffuse() const { return false; }
    // Volume scatter (isotropic phase): fires phase-sampled NEE (M59).
    virtual bool is_volume() const { return false; }
    // Nesting priority + IOR for transmissive interfaces (M57): default 0
    // = unnested legacy behavior. Only dielectric overrides.
    virtual int priority() const { return 0; }
    virtual double ior() const { return 1.0; }
    // Wavelength-resolved IOR (M67): channel 0/1/2 = 650/550/450nm hero,
    // -1 = legacy constant. Only dispersive dielectrics override.
    virtual double ior_at(int channel) const {
        (void)channel;
        return ior();
    }
    // Absorption sigma for Beer's law (M58): default black = clear.
    virtual vec3 absorb() const { return vec3(0, 0, 0); }
    // Sampling density of the scattered direction (solid angle). Delta
    // materials (mirror, glass) return 0: the integrator counts their light
    // hits full instead of MIS-weighting them.
    virtual double direction_pdf(const vec3 &, const hit_record &) const { return 0.0; }
    // First-hit albedo for AOV guides. Speculars report neutral
    // (weak guidance there: flagged limit, not wrong pixels).
    virtual vec3 surface_albedo(const hit_record &rec) const {
        (void)rec;
        return vec3(1, 1, 1);
    }
    // GPU params export: alb, alb2, emit, prm=(type,rough,ir,0).
    // False = non-exportable (host substitutes loud magenta).
    virtual bool export_gpu(float alb[4], float alb2[4], float emit[4],
                            float prm[4]) const {
        (void)alb;
        (void)alb2;
        (void)emit;
        (void)prm;
        return false;
    }
};

class lambertian : public material {
public:
    lambertian(const vec3 &a) : tex(std::make_shared<solid_color>(a)) {}
    lambertian(std::shared_ptr<texture> t) : tex(t) {}
    // GPU flatten reads the pattern for image textures.
    const std::shared_ptr<texture> &tex_ref() const { return tex; }
    bool scatter(const ray &, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
        // Cosine-weighted: pdf cos/PI cancels f*cos term, throughput *= albedo exact.
        onb frame;
        frame.build_from_w(rec.normal);
        vec3 dir = frame.local(random_cosine_direction());
        if (near_zero(dir))
            dir = rec.normal; // degenerate guard
        scattered = ray(rec.point, dir);
        attenuation = tex->sample(rec.u, rec.v, rec.point, rec.t);
        return true;
    }
    bool is_diffuse() const override { return true; }
    double direction_pdf(const vec3 &wi, const hit_record &rec) const override {
        return cosine_pdf(dot(unit_vector(wi), rec.normal));
    }
    vec3 surface_albedo(const hit_record &rec) const override {
        return tex->value(rec.u, rec.v, rec.point);
    }
    bool export_gpu(float alb[4], float alb2[4], float emit[4],
                    float prm[4]) const override {
        // Solid -> MatType::SOLID; solid-checker -> MatType::CHECKER (even/odd/scale).
        // Nested textures refuse (magenta fallback, loud not silent).
        if (auto s = dynamic_cast<const solid_color *>(tex.get())) {
            alb[0] = (float)s->rgb().x();
            alb[1] = (float)s->rgb().y();
            alb[2] = (float)s->rgb().z();
            alb2[0] = alb2[1] = alb2[2] = 0;
            emit[0] = emit[1] = emit[2] = 0;
            prm[0] = static_cast<float>(MatType::SOLID);
            prm[1] = prm[2] = prm[3] = 0;
            return true;
        }
        if (auto c = dynamic_cast<const checker *>(tex.get())) {
            auto e = dynamic_cast<const solid_color *>(c->tex_even().get());
            auto o = dynamic_cast<const solid_color *>(c->tex_odd().get());
            if (!e || !o)
                return false;
            alb[0] = (float)e->rgb().x();
            alb[1] = (float)e->rgb().y();
            alb[2] = (float)e->rgb().z();
            alb2[0] = (float)o->rgb().x();
            alb2[1] = (float)o->rgb().y();
            alb2[2] = (float)o->rgb().z();
            emit[0] = emit[1] = emit[2] = 0;
            prm[0] = static_cast<float>(MatType::CHECKER);
            prm[1] = (float)c->tex_scale();
            prm[2] = prm[3] = 0;
            return true;
        }
        // Procedural marble family -> MatType::NOISE (freq, depth, mode in prm).
        if (auto n = dynamic_cast<const noise_texture *>(tex.get())) {
            alb[0] = (float)n->color0().x();
            alb[1] = (float)n->color0().y();
            alb[2] = (float)n->color0().z();
            alb2[0] = (float)n->color1().x();
            alb2[1] = (float)n->color1().y();
            alb2[2] = (float)n->color1().z();
            emit[0] = emit[1] = emit[2] = 0;
            prm[0] = static_cast<float>(MatType::NOISE);
            prm[1] = (float)n->freq();
            prm[2] = (float)n->depth();
            prm[3] = (float)n->mode();
            return true;
        }
        return false;
    }

private:
    std::shared_ptr<texture> tex;
};

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
    vec3 albedo{0, 0, 0};
    double roughness; // <0 = anisotropic (uses rough_x/y)
    double rough_x = 0, rough_y = 0;
    int nk_id = 0; // measured n/k preset (M67); 0 = Schlick-from-albedo
    double film_d = 0.0; // thin-film thickness in nm (M68); 0 = off
    double film_n = 1.5; // thin-film IOR (M68)
};

class dielectric : public material {
public:
    dielectric(double ri, double r = 0) : ir(ri), roughness(r < 0 ? 0 : (r > 1 ? 1 : r)) {}
    // Nesting priority (M57): contained shells order by pri (outer < inner).
    // Default 0 reproduces legacy single-level front_face behavior exactly.
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
        double frefl = (film_d > 0) ? film_prob(cos_t, iri, rec.front_face)
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

// Isotropic volume scatter: uniform sphere direction, albedo attenuates.
// Not diffuse (no NEE into volumes: direct sampling of media flagged).
class isotropic : public material {
public:
    isotropic(const vec3 &a) : albedo(a) {}
    bool scatter(const ray &, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
        scattered = ray(rec.point, random_unit_vector());
        attenuation = albedo;
        return true;
    }
    // Uniform sphere: every direction has density 1/4PI, so found lights
    // MIS-weight instead of counting full (they used to, as "specular").
    double direction_pdf(const vec3 &, const hit_record &) const override {
        return 1.0 / (4.0 * 3.1415926535897932385);
    }
    bool is_volume() const override { return true; }
    vec3 surface_albedo(const hit_record &) const override { return albedo; }

private:
    vec3 albedo;
};

// Pure emitter: never scatters, integrator reads emitted() on hit.
// A texture makes emission spatially varying (photo area light, gobo);
// the hit record supplies UVs, like lambertian albedo.
class diffuse_light : public material {
public:
    diffuse_light(const vec3 &c)
        : tex(std::make_shared<solid_color>(c)) {}
    diffuse_light(std::shared_ptr<texture> t) : tex(t) {}
    // GPU flatten reads the pattern for image textures.
    const std::shared_ptr<texture> &tex_ref() const { return tex; }
    bool scatter(const ray &, const hit_record &,
                 vec3 &, ray &) const override {
        return false;
    }
    vec3 emitted() const override {
        if (auto s = dynamic_cast<const solid_color *>(tex.get()))
            return s->rgb();
        return vec3(0, 0, 0); // textured emission needs hit UVs
    }
    vec3 emitted(const hit_record &rec) const override {
        if (auto s = dynamic_cast<const solid_color *>(tex.get()))
            return s->rgb();
        if (tex)
            return tex->sample(rec.u, rec.v, rec.point, rec.t);
        return vec3(0, 0, 0);
    }
    bool is_emissive() const override { return true; }
    vec3 surface_albedo(const hit_record &rec) const override {
        if (auto s = dynamic_cast<const solid_color *>(tex.get()))
            return s->rgb();
        if (tex)
            return tex->value(rec.u, rec.v, rec.point);
        return vec3(0, 0, 0);
    }
    bool export_gpu(float alb[4], float alb2[4], float emit[4],
                    float prm[4]) const override {
        if (auto s = dynamic_cast<const solid_color *>(tex.get())) {
            alb[0] = alb[1] = alb[2] = 0;
            alb2[0] = alb2[1] = alb2[2] = 0;
            emit[0] = (float)s->rgb().x();
            emit[1] = (float)s->rgb().y();
            emit[2] = (float)s->rgb().z();
            prm[0] = static_cast<float>(MatType::EMIT);
            prm[1] = prm[2] = prm[3] = 0;
            return true;
        }
        // Image textures export through the flatten image path; other
        // patterns stay host-only and fall back to loud magenta on device.
        return false;
    }

private:
    std::shared_ptr<texture> tex;
};
