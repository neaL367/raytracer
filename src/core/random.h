#pragma once

#include "vec3.h"

#include <cstdlib>

inline double random_double()
{
    // returns a random real in [0,1)
    return std::rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max)
{
    return min + (max - min) * random_double();
}

inline vec3 random_vec3()
{
    return vec3(random_double(), random_double(), random_double());
}

inline vec3 random_vec3(double min, double max)
{
    return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
}

inline vec3 random_in_unit_sphere()
{
    while (true)
    {
        vec3 p = random_vec3(-1, 1);
        if (p.length_squared() < 1)
            return p;
    }
}

inline vec3 random_unit_vector()
{
    return unit_vector(random_in_unit_sphere());
}