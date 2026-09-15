#pragma once      // include guard: prevents this header's contents being seen twice in one translation unit

#include "vec3.h" // include the vec3 class definition, as rays are defined by a point and a direction in 3D space

class ray // represents a ray in 3D space, defined by an origin point and a direction vector
{
public:
    ray() {} // default constructor: leaves all members uninitialized, same as before — assign before use
    ray(const vec3 &origin, const vec3 &direction) : orig(origin), dir(direction),
                                                     inv_dir(1.0 / direction.x(), 1.0 / direction.y(), 1.0 / direction.z()) {}

    const vec3 &origin() const { return orig; }   // read-only accessor for the origin point of the ray
    const vec3 &direction() const { return dir; } // read-only accessor for the direction vector of the ray
    // Component-wise reciprocal of direction, computed once at construction.
    // The slab test needs it per axis per box; recomputing it there costs a
    // division (15+ cycles) where a multiply (3-5 cycles) does. A zero
    // direction component yields infinity, which the slab math already
    // handles — identical values to computing 1.0/dir per test.
    const vec3 &inv_direction() const { return inv_dir; }

    vec3 at(double t) const // computes a point along the ray at a given parameter t, where t is a scalar that scales the direction vector from the origin
    {
        return orig + t * dir;
    }

private:
    vec3 orig; // the origin point of the ray, representing where the ray starts in 3D space
    vec3 dir;  // the direction vector of the ray, representing the direction in which the ray is pointing; it is not necessarily normalized
    vec3 inv_dir; // per-component reciprocal of dir; see inv_direction()
};