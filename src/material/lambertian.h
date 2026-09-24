#pragma once
#include "material_base.h"

class lambertian : public material {
public:
    lambertian(const vec3 &a) : tex(std::make_shared<solid_color>(a)) {}
    lambertian(std::shared_ptr<texture> t) : tex(t) {}
    // GPU flatten reads the pattern for image textures.
    const std::shared_ptr<texture> &tex_ref() const { return tex; }
    bool scatter(const ray &, const hit_record &rec,
                 vec3 &attenuation, ray &scattered) const override {
        // Cosine-weighted: pdf cos/PI cancels f*cos term, throughput *= albedo exact.
        vec3 eff_n = resolve_normal(rec);
        onb frame;
        frame.build_from_w(eff_n);
        vec3 dir = frame.local(random_cosine_direction());
        if (near_zero(dir))
            dir = eff_n; // degenerate guard
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
