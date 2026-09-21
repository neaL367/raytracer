#pragma once
// Constant-density medium inside a boundary shape. Samples free-flight
// distance s = -ln(1-xi)/density along the ray; scatter wins when it
// lands inside the boundary chord. Density per unit ray-t (matches our
// non-normalized rays; same convention throughout).
// Bounds = boundary bounds: BVH needs no changes. Integrator needs no
// changes (rides the scatter interface; NEE correctly skips: not diffuse).
#include "hittable.h"
#include "../core/random.h"
#include "../material/material.h"

#include <cmath>
#include <memory>

class constant_medium : public hittable {
public:
    constant_medium(std::shared_ptr<hittable> boundary, double density,
                    std::shared_ptr<material> phase)
        : border(boundary), neg_inv_density(-1.0 / density), mat(phase) {}
    constant_medium(std::shared_ptr<hittable> boundary, double density, const vec3 &c)
        : border(boundary), neg_inv_density(-1.0 / density),
          mat(std::make_shared<isotropic>(c)) {}

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        hit_record b0, b1;
        if (!border->hit(r, -1e30, 1e30, b0))
            return false;
        if (b0.t > t_max)
            return false;
        if (!border->hit(r, b0.t + 1e-4, 1e30, b1))
            return false;
        if (b0.t < t_min)
            b0.t = t_min;
        if (b1.t > t_max)
            b1.t = t_max;
        if (b0.t >= b1.t)
            return false;
        if (b0.t < 0)
            b0.t = 0;
        double chord = b1.t - b0.t;
        double scatter_dist = neg_inv_density * std::log(random_double());
        if (scatter_dist > chord)
            return false; // ray escapes: surface behind wins
        rec.t = b0.t + scatter_dist;
        rec.point = r.at(rec.t);
        rec.normal = vec3(0, 1, 0); // arbitrary: isotropic ignores it
        rec.front_face = true;
        rec.mat = mat;
        rec.u = rec.v = 0;
        return true;
    }

    bool bounding_box(aabb &box) const override { return border->bounding_box(box); }

    // GPU flatten accessors (fog uploads as a type-6 sphere slot).
    const std::shared_ptr<hittable> &border_ref() const { return border; }
    double density_val() const { return -1.0 / neg_inv_density; }
    std::shared_ptr<material> phase_ref() const { return mat; }

private:
    std::shared_ptr<hittable> border;
    double neg_inv_density;
    std::shared_ptr<material> mat;
};

// Heterogeneous medium: procedural sinusoidal modulation over a constant
// majorant (sigma). Delta (Woodcock) tracking: tentative free-flights at
// the majorant, accepted with p = density(x)/sigma (null scatters advance).
// Same boundary-chord + clamp contract as constant_medium; shadow rays
// treat accepted events as blocking on both backends.
class heterogeneous_medium : public hittable {
public:
    heterogeneous_medium(std::shared_ptr<hittable> boundary, double density,
                         std::shared_ptr<material> phase, double fx = 5.0,
                         double fy = 4.0, double fz = 6.0)
        : border(boundary), sigma(density), mat(phase), fx(fx), fy(fy), fz(fz) {}

    // Modulation in [0,1]: majorant sigma is exact, tracking unbiased.
    double modulation(const vec3 &p) const {
        double m = 0.5 + 0.5 * std::sin(fx * p.x()) * std::sin(fy * p.y()) *
                                 std::sin(fz * p.z());
        return m < 0 ? 0 : (m > 1 ? 1 : m);
    }

    bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override {
        hit_record b0, b1;
        if (!border->hit(r, -1e30, 1e30, b0))
            return false;
        if (b0.t > t_max)
            return false;
        if (!border->hit(r, b0.t + 1e-4, 1e30, b1))
            return false;
        if (b0.t < t_min)
            b0.t = t_min;
        if (b1.t > t_max)
            b1.t = t_max;
        if (b0.t >= b1.t)
            return false;
        if (b0.t < 0)
            b0.t = 0;
        double neg_inv = -1.0 / sigma;
        double cursor = b0.t;
        for (int i = 0; i < 1024; ++i) {
            double s = neg_inv * std::log(random_double());
            double x = cursor + s;
            if (x > b1.t)
                return false; // escapes: surface behind wins
            if (modulation(r.at(x)) > random_double()) {
                rec.t = x;
                rec.point = r.at(x);
                rec.normal = vec3(0, 1, 0); // arbitrary: isotropic ignores it
                rec.front_face = true;
                rec.mat = mat;
                rec.u = rec.v = 0;
                return true;
            }
            cursor = x; // null scatter: advance, throughput untouched
        }
        return false; // null budget exhausted: escapes (P ~ 2^-1024)
    }

    bool bounding_box(aabb &box) const override { return border->bounding_box(box); }

    // GPU flatten accessors (hetero uploads as a type-8 sphere slot).
    const std::shared_ptr<hittable> &border_ref() const { return border; }
    double density_val() const { return sigma; }
    std::shared_ptr<material> phase_ref() const { return mat; }
    vec3 freqs() const { return vec3(fx, fy, fz); }

private:
    std::shared_ptr<hittable> border;
    double sigma;
    std::shared_ptr<material> mat;
    double fx, fy, fz;
};
