#pragma once
#include "hittable.h"
#include <cmath>
#include <memory>

enum class BumpPattern {
    RIPPLE,    // Concentric wave rings
    BRUSHED,   // Micro-grooved concentric/tangent streaks
    HAMMERED,  // Voronoi / cellular dimples
    STIPPLE    // High frequency sand/stipple
};

class bump_map : public hittable {
public:
    bump_map(std::shared_ptr<hittable> inner, BumpPattern pattern, double scale = 1.0, double strength = 0.25)
        : obj(inner), pat(pattern), sc(scale), str(strength) {}

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        if (!obj->hit(r, t_min, t_max, rec))
            return false;

        // Perturb rec.normal based on pattern
        vec3 n = rec.normal;
        vec3 p = rec.point * sc;
        vec3 dN(0, 0, 0);

        if (pat == BumpPattern::RIPPLE) {
            double dist = std::sqrt(p.x() * p.x() + p.z() * p.z());
            double wave = std::sin(dist * 20.0);
            if (dist > 1e-4) {
                dN = vec3((p.x() / dist) * wave * str, 0.0, (p.z() / dist) * wave * str);
            }
        } else if (pat == BumpPattern::BRUSHED) {
            // High frequency micro-streaks along tangent
            double freq = std::sin(p.y() * 120.0 + std::sin(p.x() * 40.0));
            dN = vec3(freq * str * 0.5, 0.0, freq * str * 0.5);
        } else if (pat == BumpPattern::HAMMERED) {
            // Cellular/faceted dimples using sin/cos lattice
            double s1 = std::sin(p.x() * 15.0) * std::cos(p.y() * 15.0);
            double s2 = std::sin(p.y() * 15.0) * std::cos(p.z() * 15.0);
            double s3 = std::sin(p.z() * 15.0) * std::cos(p.x() * 15.0);
            dN = vec3(s1, s2, s3) * str;
        } else if (pat == BumpPattern::STIPPLE) {
            // High frequency noise
            double n1 = std::sin(p.x() * 50.0 + p.y() * 37.0);
            double n2 = std::cos(p.y() * 43.0 + p.z() * 53.0);
            dN = vec3(n1, 0.0, n2) * str;
        }

        rec.normal = unit_vector(n + dN);
        return true;
    }

    bool hit_any(const ray &r, double t_min, double t_max) const override {
        return obj->hit_any(r, t_min, t_max);
    }

    bool bounding_box(aabb &box) const override {
        return obj->bounding_box(box);
    }

    const std::shared_ptr<hittable> &inner_ref() const { return obj; }
    BumpPattern pattern() const { return pat; }
    double scale() const { return sc; }
    double strength() const { return str; }

private:
    std::shared_ptr<hittable> obj;
    BumpPattern pat;
    double sc;
    double str;
};
