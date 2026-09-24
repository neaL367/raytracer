#pragma once
#include "material_base.h"

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
    vec3 get_emit() const {
        if (auto s = dynamic_cast<const solid_color *>(tex.get()))
            return s->rgb();
        return vec3(0, 0, 0);
    }
    void set_emit(const vec3 &c) {
        tex = std::make_shared<solid_color>(c);
    }
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
