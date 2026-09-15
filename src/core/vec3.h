#pragma once       // include guard: prevents this header's contents being seen twice in one translation unit
#include <cmath>   // for std::sqrt, used in vec3::length()
#include <ostream> // for std::ostream, used in operator<< overload

class vec3 // 3D vector: represents a point, direction, or color depending on context
{
public:
    vec3() : e{0, 0, 0} {}                             // default constructor: zero vector
    vec3(double x, double y, double z) : e{x, y, z} {} // parameterized constructor

    double x() const { return e[0]; } // read-only accessor for x component
    double y() const { return e[1]; } // read-only accessor for y component
    double z() const { return e[2]; } // read-only accessor for z component

    vec3 &operator+=(const vec3 &other) // compound assignment: modifies *this in place, unlike operator+
    {
        e[0] += other.x();
        e[1] += other.y();
        e[2] += other.z();
        return *this; // return a reference to the modified object, enabling chaining (a += b += c)
    }

    double length_squared() const // returns the squared length of the vector, which is faster to compute than the actual length and often sufficient for comparisons
    {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    double length() const // returns the actual length of the vector, which is the square root of the squared length
    {
        return std::sqrt(length_squared());
    }

private:
    double e[3]; // x, y, z stored contiguously (matters later for cache/SIMD behavior)
};

vec3 operator+(const vec3 &lhs, const vec3 &rhs) // free function: produces a new vec3, leaves lhs/rhs unchanged
{
    return vec3(lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z());
}

vec3 operator-(const vec3 &lhs, const vec3 &rhs) // free function: component-wise subtraction, new vec3
{
    return vec3(lhs.x() - rhs.x(), lhs.y() - rhs.y(), lhs.z() - rhs.z());
}

vec3 operator-(const vec3 &v) // unary negation: returns a new vec3 with all components negated
{
    return vec3(-v.x(), -v.y(), -v.z());
}

vec3 operator*(const vec3 &v, double t) // vector * scalar: does the actual component-wise multiplication
{
    return vec3(v.x() * t, v.y() * t, v.z() * t);
}

vec3 operator*(double t, const vec3 &v) // scalar * vector: reorders arguments and delegates to the overload above
{
    return v * t;
}

vec3 operator*(const vec3 &u, const vec3 &v) // vector * vector: component-wise multiplication, not a dot or cross product
{
    return vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

vec3 operator/(const vec3 &v, double t) // vector / scalar: implemented as multiply-by-reciprocal, not direct division
{
    return v * (1.0 / t);
}

std::ostream &operator<<(std::ostream &out, const vec3 &v) // overloads the << operator for easy printing of vec3 objects, e.g., std::cout << v;
{
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

double dot(const vec3 &u, const vec3 &v) // dot product: returns a scalar, not a vector
{
    return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

vec3 cross(const vec3 &u, const vec3 &v) // cross product: returns a vector perpendicular to both u and v, following the right-hand rule
{
    return vec3(u.y() * v.z() - u.z() * v.y(),
                u.z() * v.x() - u.x() * v.z(),
                u.x() * v.y() - u.y() * v.x());
}

vec3 unit_vector(const vec3 &v) // returns a new vector in the same direction as v but with length 1, useful for normalization
{
    return v / v.length();
}

vec3 reflect(const vec3 &v, const vec3 &n) // reflection formula: reflects vector v around normal n, assuming n is a unit vector
{
    return v - 2 * dot(v, n) * n;
}

vec3 refract(const vec3 &uv, const vec3 &n, double etai_over_etat) // Snell's law: computes the refracted ray direction given an incident unit vector uv, a normal n, and the ratio of indices of refraction etai_over_etat
{
    double cos_theta = std::fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}