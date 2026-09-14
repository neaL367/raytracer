#pragma once // include guard: prevents this header's contents being seen twice in one translation unit

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

vec3 operator*(const vec3 &v, double t) // vector * scalar: does the actual component-wise multiplication
{
    return vec3(v.x() * t, v.y() * t, v.z() * t);
}

vec3 operator*(double t, const vec3 &v) // scalar * vector: reorders arguments and delegates to the overload above
{
    return v * t;
}

vec3 operator/(const vec3 &v, double t) // vector / scalar: implemented as multiply-by-reciprocal, not direct division
{
    return v * (1.0 / t);
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