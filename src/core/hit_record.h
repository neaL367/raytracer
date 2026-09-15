#pragma once
#include "vec3.h"

struct hit_record
{
    vec3 point;
    vec3 normal;
    double t;
};