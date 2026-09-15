#pragma once
// Tile-scheduled multithreaded render loop. Owns no scene state: it borrows
// camera, world, lights, and integrator for the duration of the call and
// fills the caller's framebuffer with linear HDR radiance (tonemapping
// happens at output, in io/ppm.h). Per-thread timings and ray counters come
// back in render_stats for the bench report.
#include "../app/config.h"
#include "../core/bench_stats.h"
#include "../core/camera.h"
#include "../core/hittable.h"
#include "../core/integrator.h"
#include "../core/quad.h"
#include "../core/random.h"
#include "../core/ray.h"
#include "../core/vec3.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

struct render_stats
{
    std::vector<double> thread_seconds;
    std::vector<std::uint64_t> thread_box;
    std::vector<std::uint64_t> thread_prim;
    double render_seconds = 0.0;
    unsigned num_threads = 0;
};

inline render_stats render_framebuffer(const camera &cam, const hittable &world,
                                       const std::vector<std::shared_ptr<quad>> &lights,
                                       const integrator &tracer, const render_config &cfg,
                                       int image_width, int image_height,
                                       std::vector<vec3> &framebuffer)
{
    render_stats stats;
    stats.num_threads = std::thread::hardware_concurrency();
    if (stats.num_threads == 0)
        stats.num_threads = 4; // fallback if the system can't report a count
    stats.thread_seconds.assign(stats.num_threads, 0.0);
    stats.thread_box.assign(stats.num_threads, 0);
    stats.thread_prim.assign(stats.num_threads, 0);

    bench_enabled_flag().store(cfg.bench, std::memory_order_relaxed);

    auto render_start = std::chrono::high_resolution_clock::now();

    // Workers grab row-strips from a shared atomic counter until none remain.
    // Relaxed ordering suffices: the counter only mints unique indices, it
    // publishes no data and enforces no ordering. Larger strips mean fewer
    // atomic grabs but coarser balancing; smaller strips balance better at
    // the cost of more grabs and worse cache reuse.
    const int num_tiles = (image_height + cfg.tile_rows - 1) / cfg.tile_rows;
    std::atomic<int> next_tile{0};

    std::vector<std::thread> threads;
    auto render_rows = [&](unsigned thread_idx)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        // Snapshot this worker's own thread-local counters; subtracting at the
        // end yields this thread's totals with no synchronization.
        std::uint64_t b0 = thread_box_tests(), p0 = thread_prim_tests();
        for (;;)
        {
            int tile = next_tile.fetch_add(1, std::memory_order_relaxed);
            if (tile >= num_tiles)
                break;
            int row_start = tile * cfg.tile_rows;
            int row_end = std::min(row_start + cfg.tile_rows, image_height);
            // Reseed per tile so a given tile always draws the same random
            // stream regardless of which worker thread renders it.
            if (cfg.bench)
                reseed_thread_rng(cfg.bench_seed + static_cast<unsigned>(tile));
            for (int j = row_start; j < row_end; ++j)
            {
                for (int i = 0; i < image_width; ++i)
                {
                    vec3 pixel_color(0, 0, 0);

                    for (int sample = 0; sample < cfg.samples_per_pixel; ++sample)
                    {
                        // Single sample means pixel center: the deterministic
                        // reference. Anything else draws two numbers and maps
                        // them through the stratum (or the whole pixel).
                        double ox, oy;
                        if (cfg.samples_per_pixel == 1)
                        {
                            ox = 0.5;
                            oy = 0.5;
                        }
                        else
                        {
                            ox = random_double();
                            oy = random_double();
                            if (cfg.stratified)
                            {
                                ox = ((sample % cfg.strat_n) + ox) / cfg.strat_n;
                                oy = ((sample / cfg.strat_n) + oy) / cfg.strat_n;
                            }
                        }
                        double s = (i + ox) / (image_width - 1);
                        double t = (j + oy) / (image_height - 1);

                        ray r = cam.get_ray(s, t);
                        pixel_color += tracer.Li(r, world, lights, cfg.max_depth);
                    }

                    double scale = 1.0 / cfg.samples_per_pixel;
                    framebuffer[j * image_width + i] = pixel_color * scale;
                }
            }
        } // end tile-grab loop
        auto t1 = std::chrono::high_resolution_clock::now();
        // Disjoint indices: no lock needed, each thread writes only its slots.
        stats.thread_seconds[thread_idx] = std::chrono::duration<double>(t1 - t0).count();
        stats.thread_box[thread_idx] = thread_box_tests() - b0;
        stats.thread_prim[thread_idx] = thread_prim_tests() - p0;
    };

    for (unsigned int t = 0; t < stats.num_threads; ++t)
        threads.emplace_back(render_rows, t);

    for (auto &th : threads)
        th.join();

    auto render_end = std::chrono::high_resolution_clock::now();
    stats.render_seconds = std::chrono::duration<double>(render_end - render_start).count();
    bench_enabled_flag().store(false, std::memory_order_relaxed);
    return stats;
}
