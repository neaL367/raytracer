#pragma once

class vec3
{
public:
    vec3() : e{0, 0, 0} {}
    vec3(double x, double y, double z) : e{x, y, z} {}

    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }

    vec3 &operator+=(const vec3 &other)
    {
        e[0] += other.x();
        e[1] += other.y();
        e[2] += other.z();
        return *this;
    }

private:
    double e[3];
};

vec3 operator+(const vec3 &lhs, const vec3 &rhs)
{
    return vec3(lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z());
}