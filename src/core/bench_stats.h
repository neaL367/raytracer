#pragma once
// Bench-only counters for --bench runs.
//
// Disabled by default: the normal render path pays exactly one predictable
// branch per test and nothing else.
//
// Design note: counters are thread_local, NOT shared atomics. A shared
// atomic fetch_add on every box test measured ~5x slowdown from contention
// alone — it would pollute the timings this harness exists to capture.
// thread_local ++ costs ~1 cycle. Each worker reads out its own deltas at
// the end of its strip (see main.cpp render_rows); the main thread sums
// them after join. No locks, no fences beyond the join that already exists.
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
