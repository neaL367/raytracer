#pragma once
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/random.h"
#include "../core/sampler.h"
#include "../core/onb.h"
#include "../core/ggx.h"
#include "../core/texture.h"
#include "../geometry/hittable.h"
#include <memory>

// Material answers scatter only. No light/traversal knowledge.
// Returns false = ray absorbed (killed, contributes black).
// emitted() default black; diffuse_light overrides (no scatter).
class material {
public:
    virtual ~material() = default;
    virtual bool scatter(const ray &in, const hit_record &rec,
                         vec3 &attenuation, ray &scattered) const = 0;
    virtual vec3 emitted() const { return vec3(0, 0, 0); }
    // NEE applies to diffuse only; specular paths skip explicit lights.
    virtual bool is_diffuse() const { return false; }
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
        // Solid -> type 0; solid-checker -> type 4 (even/odd/scale).
        // Nested textures refuse (magenta fallback, loud not silent).
        if (auto s = dynamic_cast<const solid_color *>(tex.get())) {
            alb[0] = (float)s->rgb().x();
            alb[1] = (float)s->rgb().y();
            alb[2] = (float)s->rgb().z();
            alb2[0] = alb2[1] = alb2[2] = 0;
            emit[0] = emit[1] = emit[2] = 0;
            prm[0] = 0;
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
            prm[0] = 4;
            prm[1] = (float)c->tex_scale();
            prm[2] = prm[3] = 0;
            return true;
        }
        // Procedural marble family -> type 9 (freq, depth, mode in prm).
        if (auto n = dynamic_cast<const noise_texture *>(tex.get())) {
            alb[0] = (float)n->color0().x();
            alb[1] = (float)n->color0().y();
            alb[2] = (float)n->color0().z();
            alb2[0] = (float)n->color1().x();
            alb2[1] = (float)n->color1().y();
            alb2[2] = (float)n->color1().z();
            emit[0] = emit[1] = emit[2] = 0;
            prm[0] = 9;
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
    bool scatter(const ray &in, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
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
        attenuation = ggx::fresnel_schlick(albedo, cos_vh) * ratio;
        scattered = ray(rec.point, frame.local(Ll));
        return true;
    }
    vec3 surface_albedo(const hit_record &) const override { return albedo; }
    bool export_gpu(float alb[4], float alb2[4], float emit[4],
                    float prm[4]) const override {
        alb[0] = (float)albedo.x();
        alb[1] = (float)albedo.y();
        alb[2] = (float)albedo.z();
        alb2[0] = alb2[1] = alb2[2] = 0;
        emit[0] = emit[1] = emit[2] = 0;
        prm[0] = 7; // GGX conductor (fuzz-era type 1 deleted)
        prm[1] = (float)roughness;
        prm[2] = prm[3] = 0;
        return true;
    }

private:
    vec3 albedo;
    double roughness;
};

class dielectric : public material {
public:
    dielectric(double ri) : ir(ri) {}
    bool scatter(const ray &in, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
        attenuation = vec3(1, 1, 1); // glass absorbs nothing
        double ratio = rec.front_face ? (1.0 / ir) : ir;
        vec3 unit = unit_vector(in.direction());
        double cos_t = fmin(dot(-unit, rec.normal), 1.0);
        double sin_t = std::sqrt(1.0 - cos_t * cos_t);
        bool cannot_refract = ratio * sin_t > 1.0;
        vec3 dir = (cannot_refract || reflectance(cos_t, ratio) > random_double())
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
        prm[0] = 2;
        prm[1] = 0;
        prm[2] = (float)ir;
        prm[3] = 0;
        return true;
    }

private:
    double ir;
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
    vec3 surface_albedo(const hit_record &) const override { return albedo; }

private:
    vec3 albedo;
};

// Pure emitter: never scatters, integrator reads emitted() on hit.
class diffuse_light : public material {
public:
    diffuse_light(const vec3 &c) : emit_color(c) {}
    bool scatter(const ray &, const hit_record &,
                 vec3 &, ray &) const override {
        return false;
    }
    vec3 emitted() const override { return emit_color; }
    vec3 surface_albedo(const hit_record &) const override { return emit_color; }
    bool export_gpu(float alb[4], float alb2[4], float emit[4],
                    float prm[4]) const override {
        alb[0] = alb[1] = alb[2] = 0;
        alb2[0] = alb2[1] = alb2[2] = 0;
        emit[0] = (float)emit_color.x();
        emit[1] = (float)emit_color.y();
        emit[2] = (float)emit_color.z();
        prm[0] = 3;
        prm[1] = prm[2] = prm[3] = 0;
        return true;
    }

private:
    vec3 emit_color;
};
