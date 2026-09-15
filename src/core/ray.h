#pragma once      // include guard: prevents this header's contents being seen twice in one translation unit

#include "vec3.h" // include the vec3 class definition, as rays are defined by a point and a direction in 3D space

class ray // represents a ray in 3D space, defined by an origin point and a direction vector
{
public:
    ray() {}                                                                         // default constructor: does not initialize origin or direction, which may lead to undefined behavior if used before setting them
    ray(const vec3 &origin, const vec3 &direction) : orig(origin), dir(direction) {} // parameterized constructor: initializes the ray with a specific origin and direction

    const vec3 &origin() const { return orig; }   // read-only accessor for the origin point of the ray
    const vec3 &direction() const { return dir; } // read-only accessor for the direction vector of the ray

    vec3 at(double t) const // computes a point along the ray at a given parameter t, where t is a scalar that scales the direction vector from the origin
    {
        return orig + t * dir;
    }

private:
    vec3 orig; // the origin point of the ray, representing where the ray starts in 3D space
    vec3 dir;  // the direction vector of the ray, representing the direction in which the ray is pointing; it is not necessarily normalized
};