#pragma once
#include <atomic>
#include <cstdint>

// Bench counters: atomics skipped unless --bench enabled, so normal
// renders pay one predictable branch, not cache-line ping-pong.
inline std::atomic<bool> &bench_enabled_flag() {
    static std::atomic<bool> f{false};
    return f;
}
inline std::atomic<std::uint64_t> &bench_rays() {
    static std::atomic<std::uint64_t> c{0};
    return c;
}
inline std::atomic<std::uint64_t> &bench_boxes() {
    static std::atomic<std::uint64_t> c{0};
    return c;
}
inline std::atomic<std::uint64_t> &bench_prims() {
    static std::atomic<std::uint64_t> c{0};
    return c;
}
inline void count_ray() {
    if (bench_enabled_flag().load(std::memory_order_relaxed))
        bench_rays().fetch_add(1, std::memory_order_relaxed);
}
inline void count_box() {
    if (bench_enabled_flag().load(std::memory_order_relaxed))
        bench_boxes().fetch_add(1, std::memory_order_relaxed);
}
inline void count_prim() {
    if (bench_enabled_flag().load(std::memory_order_relaxed))
        bench_prims().fetch_add(1, std::memory_order_relaxed);
}
