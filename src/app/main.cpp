#include "core/vec3.h"
#include "core/ray.h"
#include "core/random.h"
#include "core/sampler.h"
#include "core/bench_stats.h"
#include "core/texture.h"
#include "core/obj_loader.h"
#include "camera/camera.h"
#include "geometry/hittable.h"
#include "geometry/sphere.h"
#include "geometry/triangle.h"
#include "geometry/quad.h"
#include "accel/bvh.h"
#include "material/material.h"
#include "integrator/integrator.h"
#include "io/denoise.h"
#include "output/ppm.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char **argv) {
    int spp = 16; // 4x4 strata default; perfect squares stratify
    double aperture = 0.0; // 0 = pinhole; >0 thin-lens blur
    double exposure = 1.0; // linear HDR scale before ACES
    unsigned num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 4;
    int tile_rows = 8;
    bool bench = false;
    bool use_sah = true;
    bool do_denoise = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--samples" && i + 1 < argc)
            spp = std::max(1, std::atoi(argv[++i]));
        else if (a == "--aperture" && i + 1 < argc)
            aperture = std::max(0.0, std::atof(argv[++i]));
        else if (a == "--threads" && i + 1 < argc)
            num_threads = (unsigned)std::max(1, std::atoi(argv[++i]));
        else if (a == "--tile" && i + 1 < argc)
            tile_rows = std::max(1, std::atoi(argv[++i]));
        else if (a == "--split" && i + 1 < argc)
            use_sah = (std::string(argv[++i]) != "median");
        else if (a == "--exposure" && i + 1 < argc)
            exposure = std::max(0.0, std::atof(argv[++i]));
        else if (a == "--denoise")
            do_denoise = true;
        else if (a == "--bench")
            bench = true;
    }
    // Legacy M2 stream needs one global RNG in pixel order: single thread.
    bool legacy = (spp == 1);
    if (legacy && num_threads != 1) {
        std::cout << "note: spp=1 legacy path forces 1 thread (M2 stream)\n";
        num_threads = 1;
    }

    const int W = 400;
    const int H = static_cast<int>(W / (16.0 / 9.0));
    const int max_depth = 50; // RR handles termination; depth is backstop
    const unsigned base_seed = 42;

    std::vector<std::shared_ptr<hittable>> objs;
    std::vector<std::shared_ptr<quad>> lights;
    // Checker ground (world-pos parity) + cube mesh replaces center sphere.
    auto ground_mat =
        std::make_shared<lambertian>(std::make_shared<checker>(4.0, vec3(0.8, 0.8, 0.8),
                                                               vec3(0.3, 0.3, 0.3)));
    auto cube_mat =
        std::make_shared<lambertian>(std::make_shared<checker>(3.0, vec3(0.7, 0.3, 0.3),
                                                               vec3(0.9, 0.9, 0.9)));
    auto left_mat = std::make_shared<metal>(vec3(0.8, 0.8, 0.8), 0.3);
    auto right_mat = std::make_shared<dielectric>(1.5);
    auto light_mat = std::make_shared<diffuse_light>(vec3(4, 4, 4));

    objs.push_back(std::make_shared<sphere>(vec3(0, -100.5, -1), 100, ground_mat));
    std::vector<std::shared_ptr<triangle>> mesh;
    if (!obj_loader::load_obj("assets/cube.obj", mesh, cube_mat)) {
        // No asset (pared checkout): center sphere keeps binary working.
        std::cerr << "assets/cube.obj missing: falling back to sphere\n";
        objs.push_back(std::make_shared<sphere>(
            vec3(0, 0, -1), 0.5,
            std::make_shared<lambertian>(vec3(0.7, 0.3, 0.3))));
    } else {
        for (auto &t : mesh)
            objs.push_back(t);
    }
    objs.push_back(std::make_shared<sphere>(vec3(-1, 0, -1), 0.5, left_mat));
    objs.push_back(std::make_shared<sphere>(vec3(1, 0, -1), 0.5, right_mat));
    auto light = std::make_shared<quad>(vec3(-1, 1.9, -2), vec3(2, 0, 0),
                                        vec3(0, 0, 2), light_mat);
    objs.push_back(light);
    lights.push_back(light);

    // BVH over everything incl. light quad: shadow + NEE rays traverse it.
    bvh_node world(objs, 0, objs.size(), use_sah);

    camera cam(vec3(0, 0, 0), vec3(0, 0, -1), vec3(0, 1, 0),
               90.0, double(W) / double(H), aperture, 1.0);
    integrator tracer;
    std::vector<vec3> fb(W * H);
    bench_enabled_flag().store(bench, std::memory_order_relaxed);

    auto render_pixel = [&](int i, int j) {
        vec3 acc(0, 0, 0);
        if (legacy) {
            double u = double(i) / (W - 1);
            double v = double(j) / (H - 1);
            acc = tracer.Li(cam.get_ray(u, v), world, lights, max_depth);
        } else {
            rng_seed(base_seed + (unsigned)(j * W + i));
            auto offs = pixel_samples(spp);
            for (auto [ox, oy] : offs) {
                double u = (i + ox) / W;
                double v = (j + oy) / H;
                acc += tracer.Li(cam.get_ray(u, v), world, lights, max_depth);
            }
            acc /= (double)offs.size();
        }
        fb[(size_t)j * W + i] = acc;
    };

    auto t_start = std::chrono::high_resolution_clock::now();
    if (legacy)
        rng_seed(base_seed);
    const int num_tiles = (H + tile_rows - 1) / tile_rows;
    std::atomic<int> next_tile{0};
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < num_threads; ++t) {
        workers.emplace_back([&] {
            for (;;) {
                int ti = next_tile.fetch_add(1, std::memory_order_relaxed);
                if (ti >= num_tiles)
                    break;
                int j0 = ti * tile_rows;
                int j1 = std::min(j0 + tile_rows, H);
                for (int j = j0; j < j1; ++j)
                    for (int i = 0; i < W; ++i)
                        render_pixel(i, j);
            }
        });
    }
    for (auto &th : workers)
        th.join();
    auto t_end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();

    if (do_denoise) {
        auto d0 = std::chrono::high_resolution_clock::now();
        fb = bilateral_denoise(fb, W, H);
        secs += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - d0)
                    .count();
    }

    std::filesystem::create_directories("out");
    if (!write_ppm("out/image.ppm", fb, W, H, exposure)) {
        std::cerr << "write failed\n";
        return 1;
    }
    std::uint64_t rays = bench_rays().load();
    std::cout << "wrote out/image.ppm " << W << "x" << H << " spp=" << spp
              << " threads=" << num_threads << " tile=" << tile_rows
              << " split=" << (use_sah ? "sah" : "median")
              << " exposure=" << exposure << "\n";
    std::cout << "render " << secs << "s";
    if (bench) {
        std::cout << " rays=" << rays << " (" << (rays / 1e6 / secs) << " Mrays/s)"
                  << " box=" << bench_boxes().load()
                  << " prim=" << bench_prims().load();
    }
    std::cout << "\n";
    return 0;
}
