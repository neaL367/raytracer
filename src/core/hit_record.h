#pragma once

#include "vec3.h"

#include <memory>

class material; // forward declaration, mirroring the one in material.h

struct hit_record
{
    vec3 point;
    vec3 normal;
    std::shared_ptr<material> mat;
    double t;
    // Surface parametrization for textures. Sphere fills both; primitives
    // without a natural mapping (triangle, quad) leave zeros.
    double u = 0.0;
    double v = 0.0;
    bool front_face;

    void set_face_normal(const ray &r, const vec3 &outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};