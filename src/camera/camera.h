#pragma once
#include "../core/vec3.h"
#include "../core/ray.h"
#include "../core/random.h"
#include <cmath>

// Thin-lens camera. aperture 0 = pinhole exactly (same math, zero disk).
// focus_dist scales viewport: rays converge at focal plane, blur elsewhere.
// Default ctor reproduces M1-M3 pinhole (vfov 90, focus 1) bit-exact.
class camera {
public:
    camera(double aspect_ratio = 16.0 / 9.0)
        : camera(vec3(0, 0, 0), vec3(0, 0, -1), vec3(0, 1, 0),
                 90.0, aspect_ratio, 0.0, 1.0) {}

    camera(const vec3 &lookfrom, const vec3 &lookat, const vec3 &vup,
           double vfov_deg, double aspect_ratio, double aperture, double focus_dist) {
        double theta = vfov_deg * 3.1415926535897932385 / 180.0;
        double h = 2.0 * std::tan(theta / 2.0);
        double viewport_height = h * focus_dist;
        double viewport_width = viewport_height * aspect_ratio;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        // Viewport already scaled by focus_dist above: no second multiply
        // (an extra focus factor here is invisible at focus=1 but blows up
        // the FOV for telephoto-style distances like Cornell's 800).
        horizontal = viewport_width * u;
        vertical = viewport_height * v;
        lower_left = origin - horizontal / 2 - vertical / 2 - focus_dist * w;
        lens_radius = aperture / 2;
    }

    ray get_ray(double s, double t) const {
        vec3 rd = lens_radius * random_in_unit_disk();
        vec3 offset = u * rd.x() + v * rd.y();
        // Closed shutter draws no RNG: default stream bit-exact.
        double tm = (shutter1 > shutter0) ? shutter0 + random_double() * (shutter1 - shutter0)
                                          : shutter0;
        return ray(origin + offset,
                   lower_left + s * horizontal + t * vertical - origin - offset, tm);
    }

    void set_shutter(double t0, double t1) {
        shutter0 = t0;
        shutter1 = t1;
    }

    vec3 lens_origin() const { return origin; }
    double lens_r() const { return lens_radius; }
    // GPU mirror accessors (M53: exact viewport copy, no vfov resync).
    vec3 eye() const { return origin; }
    vec3 corner() const { return lower_left; }
    vec3 span_u() const { return horizontal; }
    vec3 span_v() const { return vertical; }
    vec3 dir() const { return -w; }
    vec3 up_dir() const { return v; }
    vec3 right_dir() const { return u; }

private:
    vec3 origin, lower_left, horizontal, vertical, u, v, w;
    double lens_radius = 0;
    double shutter0 = 0, shutter1 = 0;
};
