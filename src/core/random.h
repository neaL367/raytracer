#pragma once
#include <cstdlib>

inline double random_double()
{
    // returns a random real in [0,1)
    return std::rand() / (RAND_MAX + 1.0);
}