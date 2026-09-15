#pragma once
#include "vec3.h"
#include "random.h"
#include <cmath>
#include <numbers>

// Orthonormal basis around a normal: maps hemisphere samples drawn around
// +z into the frame (u, v, w=n). Built from an arbitrary helper axis that is
// nowhere near parallel to n, so the cross products stay well-conditioned.
class onb
{
public:
    onb() {}
    onb(const vec3 &n)
    {
        w = n;
        vec3 a = (std::fabs(w.x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        v = unit_vector(cross(w, a));
        u = cross(w, v);
    }

    vec3 local(double a, double b, double c) const
    {
        return a * u + b * v + c * w;
    }

private:
    vec3 u, v, w;
};

// Cosine-weighted hemisphere direction: uniform randoms (r1, r2) map to a
// direction whose solid-angle pdf is cos(theta)/pi. Matches
// lambertian::scattering_pdf below — sampler and pdf must always agree.
inline vec3 random_cosine_direction()
{
    double r1 = random_double();
    double r2 = random_double();
    double phi = 2.0 * std::numbers::pi * r1;
    double x = std::cos(phi) * std::sqrt(r2);
    double y = std::sin(phi) * std::sqrt(r2);
    double z = std::sqrt(1.0 - r2);
    return vec3(x, y, z);
}
