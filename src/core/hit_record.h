#pragma once

#include "vec3.h"

#include <memory>

class material; // forward declaration, mirroring the one in material.h

struct hit_record
{
    vec3 point;
    vec3 normal;
    double t;
    std::shared_ptr<material> mat;
};