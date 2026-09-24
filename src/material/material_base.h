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

#include "../gpu/shaders/material_types.inc"

enum class MatType : int {
    SOLID     = MAT_SOLID,
    RESERVED  = MAT_RESERVED, // fuzz-era deleted; do not renumber
    GLASS     = MAT_GLASS,
    EMIT      = MAT_EMIT,
    CHECKER   = MAT_CHECKER,
    IMAGE     = MAT_IMAGE,
    FOG       = MAT_FOG,
    CONDUCTOR = MAT_CONDUCTOR,
    HET       = MAT_HET,
    NOISE     = MAT_NOISE,
    ANISO     = MAT_ANISO,
    DISNEY    = MAT_DISNEY
};

// Static assertions ensuring C++ MatType enum exactly mirrors GLSL shader constants
static_assert(static_cast<int>(MatType::SOLID) == MAT_SOLID, "MatType::SOLID mismatch");
static_assert(static_cast<int>(MatType::GLASS) == MAT_GLASS, "MatType::GLASS mismatch");
static_assert(static_cast<int>(MatType::EMIT) == MAT_EMIT, "MatType::EMIT mismatch");
static_assert(static_cast<int>(MatType::CHECKER) == MAT_CHECKER, "MatType::CHECKER mismatch");
static_assert(static_cast<int>(MatType::IMAGE) == MAT_IMAGE, "MatType::IMAGE mismatch");
static_assert(static_cast<int>(MatType::FOG) == MAT_FOG, "MatType::FOG mismatch");
static_assert(static_cast<int>(MatType::CONDUCTOR) == MAT_CONDUCTOR, "MatType::CONDUCTOR mismatch");
static_assert(static_cast<int>(MatType::HET) == MAT_HET, "MatType::HET mismatch");
static_assert(static_cast<int>(MatType::NOISE) == MAT_NOISE, "MatType::NOISE mismatch");
static_assert(static_cast<int>(MatType::ANISO) == MAT_ANISO, "MatType::ANISO mismatch");
static_assert(static_cast<int>(MatType::DISNEY) == MAT_DISNEY, "MatType::DISNEY mismatch");

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
    MatType::ANISO,
    MatType::DISNEY
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
        case MatType::DISNEY:    return "DISNEY";
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

    // Tangent-space normal mapping (M73)
    std::shared_ptr<texture> normal_map = nullptr;
    void set_normal_map(std::shared_ptr<texture> nm) { normal_map = nm; }
    const std::shared_ptr<texture> &get_normal_map() const { return normal_map; }

    vec3 resolve_normal(const hit_record &rec) const {
        if (!normal_map) return rec.normal;
        vec3 t_norm = normal_map->sample(rec.u, rec.v, rec.point, rec.t);
        vec3 n_tan = 2.0 * t_norm - vec3(1.0, 1.0, 1.0);
        if (n_tan.length_squared() <= 1e-12) return rec.normal;
        n_tan = unit_vector(n_tan);

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
        vec3 perturbed = unit_vector(T * n_tan.x() + B * n_tan.y() + N * n_tan.z());
        return (dot(perturbed, rec.normal) < 0.0) ? rec.normal : perturbed;
    }
};
