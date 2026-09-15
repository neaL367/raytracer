#pragma once
// Optional ray-statistics counters: how many bounding-box and primitive
// intersection tests a render performs. Disabled by default, in which case
// each counting call costs a single predictable branch and nothing else.
//
// The counters are thread_local rather than shared atomics: a shared atomic
// increment on every box test measured about a 5x slowdown from contention,
// while a thread-local increment costs roughly a cycle. Each worker reads out
// its own totals when finished; the main thread adds them up after joining,
// so there are no locks and no cross-thread traffic during the render.
#include <atomic>
#include <cstdint>

inline std::atomic<bool> &bench_enabled_flag()
{
    static std::atomic<bool> flag{false};
    return flag;
}

inline std::uint64_t &thread_box_tests()
{
    thread_local std::uint64_t n{0};
    return n;
}

inline std::uint64_t &thread_prim_tests()
{
    thread_local std::uint64_t n{0};
    return n;
}

inline void count_box_test()
{
    if (bench_enabled_flag().load(std::memory_order_relaxed))
        ++thread_box_tests();
}

inline void count_prim_test()
{
    if (bench_enabled_flag().load(std::memory_order_relaxed))
        ++thread_prim_tests();
}
