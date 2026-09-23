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
