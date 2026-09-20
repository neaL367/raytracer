#pragma once
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/random.h"
#include "../core/sampler.h"
#include "../core/onb.h"
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
    // First-hit albedo for AOV guides. Speculars report neutral
    // (weak guidance there: flagged limit, not wrong pixels).
    virtual vec3 surface_albedo(const hit_record &rec) const {
        (void)rec;
        return vec3(1, 1, 1);
    }
    // GPU params export: alb, alb2, emit, prm=(type,fuzz,ir,0).
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
    bool scatter(const ray &, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
        // Cosine-weighted: pdf cos/PI cancels f*cos term, throughput *= albedo exact.
        onb frame;
        frame.build_from_w(rec.normal);
        vec3 dir = frame.local(random_cosine_direction());
        if (near_zero(dir))
            dir = rec.normal; // degenerate guard
        scattered = ray(rec.point, dir);
        attenuation = tex->value(rec.u, rec.v, rec.point);
        return true;
    }
    bool is_diffuse() const override { return true; }
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
        return false;
    }

private:
    std::shared_ptr<texture> tex;
};

class metal : public material {
public:
    metal(const vec3 &a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}
    bool scatter(const ray &in, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
        vec3 refl = reflect(unit_vector(in.direction()), rec.normal);
        scattered = ray(rec.point, refl + fuzz * random_in_unit_sphere());
        attenuation = albedo;
        return dot(scattered.direction(), rec.normal) > 0;
    }
    vec3 surface_albedo(const hit_record &) const override { return albedo; }
    bool export_gpu(float alb[4], float alb2[4], float emit[4],
                    float prm[4]) const override {
        alb[0] = (float)albedo.x();
        alb[1] = (float)albedo.y();
        alb[2] = (float)albedo.z();
        alb2[0] = alb2[1] = alb2[2] = 0;
        emit[0] = emit[1] = emit[2] = 0;
        prm[0] = 1;
        prm[1] = (float)fuzz;
        prm[2] = prm[3] = 0;
        return true;
    }

private:
    vec3 albedo;
    double fuzz;
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
