#pragma once

#include "vec3.h"

#include <cstdlib>
#include <numbers>
#include <random>
#include <atomic>

// --- deterministic bench mode ------------------------------------------------
// Set BEFORE any RNG use (and before spawning render threads). Each thread's
// generator is seeded lazily on first use, so every worker gets the same
// fixed seed on every run -> identical image for a given seed.
// Why per-thread fixed seeds instead of one shared generator? A shared
// generator needs a mutex per random number (millions of calls) and makes
// output depend on thread scheduling order. Per-thread seeds cost nothing
// and stay deterministic because each thread always renders the same rows.
inline std::atomic<bool> &rng_deterministic_flag()
{
    static std::atomic<bool> flag{false};
    return flag;
}

inline std::atomic<unsigned> &rng_bench_seed()
{
    static std::atomic<unsigned> seed{42u};
    return seed;
}

inline void set_deterministic_rng(bool on, unsigned seed = 42u)
{
    rng_bench_seed().store(seed, std::memory_order_relaxed);
    rng_deterministic_flag().store(on, std::memory_order_relaxed);
}

inline double random_double()
{
    thread_local std::mt19937 generator = [] {
        if (rng_deterministic_flag().load(std::memory_order_relaxed))
            return std::mt19937(rng_bench_seed().load(std::memory_order_relaxed));
        return std::mt19937(std::random_device{}());
    }();
    thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
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

inline double degrees_to_radians(double degrees)
{
    return degrees * std::numbers::pi / 180.0;
}

inline vec3 random_in_unit_disk()
{
    while (true)
    {
        vec3 p = vec3(random_double(-1, 1), random_double(-1, 1), 0);
        if (p.length_squared() < 1)
            return p;
    }
}