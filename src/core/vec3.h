#pragma once
#include <cmath>
#include <iostream>

// vec3 value type, not hierarchy. Reason: cache-friendly, inlinable,
// no vtable in hot loop. Millions of ops per image, indirection kills.
class vec3 {
public:
    double e[3];

    vec3() : e{0, 0, 0} {}
    vec3(double x, double y, double z) : e{x, y, z} {}

    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }

    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    vec3 &operator+=(const vec3 &v) {
        e[0] += v.e[0]; e[1] += v.e[1]; e[2] += v.e[2];
        return *this;
    }
    vec3 &operator-=(const vec3 &v) {
        e[0] -= v.e[0]; e[1] -= v.e[1]; e[2] -= v.e[2];
        return *this;
    }
    vec3 &operator*=(double t) {
        e[0] *= t; e[1] *= t; e[2] *= t;
        return *this;
    }
    vec3 &operator/=(double t) { return *this *= 1 / t; }

    double length_squared() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    double length() const { return std::sqrt(length_squared()); }

    static vec3 random();
    static vec3 random(double min, double max);
};

inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0]+v.e[0], u.e[1]+v.e[1], u.e[2]+v.e[2]);
}
inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0]-v.e[0], u.e[1]-v.e[1], u.e[2]-v.e[2]);
}
inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0]*v.e[0], u.e[1]*v.e[1], u.e[2]*v.e[2]);
}
inline vec3 operator*(double t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}
inline vec3 operator*(const vec3 &v, double t) { return t * v; }
inline vec3 operator/(const vec3 &v, double t) { return (1 / t) * v; }

inline double dot(const vec3 &u, const vec3 &v) {
    return u.e[0]*v.e[0] + u.e[1]*v.e[1] + u.e[2]*v.e[2];
}
inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1]*v.e[2] - u.e[2]*v.e[1],
                u.e[2]*v.e[0] - u.e[0]*v.e[2],
                u.e[0]*v.e[1] - u.e[1]*v.e[0]);
}
inline vec3 unit_vector(const vec3 &v) {
    double len = v.length();
    return (len > 0.0) ? (v * (1.0 / len)) : vec3(0, 0, 0);
}

// reflect: v mirrored about n. Roughness-0 conductors use it exact.
inline vec3 reflect(const vec3 &v, const vec3 &n) { return v - 2 * dot(v, n) * n; }
// refract per Snell. eta = n1/n2. TIR handled by caller via discriminant.
inline vec3 refract(const vec3 &uv, const vec3 &n, double eta) {
    double cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = eta * (uv + cos_theta * n);
    vec3 r_out_parallel = -std::sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}
// Degenerate scatter guard: random fallback avoids NaN normals.
inline bool near_zero(const vec3 &v) {
    const double s = 1e-8;
    return fabs(v.e[0]) < s && fabs(v.e[1]) < s && fabs(v.e[2]) < s;
}

inline std::ostream &operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

using point3 = vec3;
using color = vec3;
