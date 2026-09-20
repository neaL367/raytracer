#include "core/vec3.h"
#include "core/ray.h"
#include "core/random.h"
#include "core/sampler.h"
#include "core/bench_stats.h"
#include "camera/camera.h"
#include "geometry/hittable.h"
#include "geometry/quad.h"
#include "accel/bvh.h"
#include "integrator/integrator.h"
#include "scene/scene.h"
#include "io/denoise.h"
#include "output/ppm.h"
#include "output/pfm.h"

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
    std::string scene_name = "default";
    double shutter0 = 0, shutter1 = 0;
    double fog_density = 0;
    double het_density = 0;
    unsigned seed = 42; // base RNG seed; per-pixel stream = seed + pixel index
    int W = 400, H = -1; // H defaults to 16:9 unless --height given
    std::string hdr_path; // empty = no float dump
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
        else if (a == "--scene" && i + 1 < argc)
            scene_name = argv[++i];
        else if (a == "--shutter" && i + 2 < argc) {
            shutter0 = std::atof(argv[++i]);
            shutter1 = std::atof(argv[++i]);
        } else if (a == "--fog" && i + 1 < argc) {
            fog_density = std::max(0.0, std::atof(argv[++i]));
        } else if (a == "--het" && i + 1 < argc) {
            het_density = std::max(0.0, std::atof(argv[++i]));
        }
        else if (a == "--seed" && i + 1 < argc)
            seed = (unsigned)std::max(0, std::atoi(argv[++i]));
        else if (a == "--width" && i + 1 < argc)
            W = std::max(8, std::atoi(argv[++i]));
        else if (a == "--height" && i + 1 < argc)
            H = std::max(8, std::atoi(argv[++i]));
        else if (a == "--bench")
            bench = true;
        else if (a == "--hdr" && i + 1 < argc)
            hdr_path = argv[++i];
    }
    // Legacy M2 stream needs one global RNG in pixel order: single thread.
    bool legacy = (spp == 1);
    if (legacy && num_threads != 1) {
        std::cout << "note: spp=1 legacy path forces 1 thread (M2 stream)\n";
        num_threads = 1;
    }

    // Square-ish scenes (cornell) pass --height explicitly; default 16:9.
    if (H <= 0)
        H = static_cast<int>(W / (16.0 / 9.0));
    const int max_depth = 50; // RR handles termination; depth is backstop
    const unsigned base_seed = seed;
    mip_render_h() = H; // LOD seam: texture minification follows output height

    // One construction order shared with the GPU uploader (scene/scene.h).
    scene_data scene = build_scene(scene_name, double(W) / double(H), aperture, shutter0,
                                   shutter1, fog_density, het_density);
    std::vector<std::shared_ptr<hittable>> &objs = scene.objs;
    std::vector<light> &lights = scene.lights;
    camera &cam = scene.cam;

    // BVH over everything incl. light quad: shadow + NEE rays traverse it.
    bvh_node world(objs, 0, objs.size(), use_sah);

    integrator tracer;
    std::vector<vec3> fb((size_t)W * H);
    // Guide buffers only under --denoise: default pixels byte-exact.
    std::vector<vec3> albedo_fb((size_t)W * H), normal_fb((size_t)W * H);
    bench_enabled_flag().store(bench, std::memory_order_relaxed);

    auto render_pixel = [&](int i, int j) {
        vec3 acc(0, 0, 0), alb(0, 0, 0), nrm(0, 0, 0);
        if (legacy) {
            double u = double(i) / (W - 1);
            double v = double(j) / (H - 1);
            ray primary = cam.get_ray(u, v);
            acc = tracer.Li(primary, world, lights, max_depth);
            if (do_denoise) {
                vec3 a, n;
                bool hit = false;
                first_hit_aov(primary, world, a, n, hit);
                if (hit) {
                    alb = a;
                    nrm = n;
                }
            }
        } else {
            rng_seed(base_seed + (unsigned)(j * W + i));
            auto offs = pixel_samples(spp);
            for (auto [ox, oy] : offs) {
                double u = (i + ox) / W;
                double v = (j + oy) / H;
                ray primary = cam.get_ray(u, v);
                acc += tracer.Li(primary, world, lights, max_depth);
                // Guides appended after beauty: deterministic order, and
                // AOV uses no RNG so the beauty stream never shifts.
                if (do_denoise) {
                    vec3 a, n;
                    bool hit = false;
                    first_hit_aov(primary, world, a, n, hit);
                    if (hit) {
                        alb += a;
                        nrm += n;
                    }
                }
            }
            acc /= (double)offs.size();
            if (do_denoise) {
                alb /= (double)offs.size();
                if (nrm.length_squared() > 0) // all-miss pixels keep zero guide
                    nrm = unit_vector(nrm);
            }
        }
        fb[(size_t)j * W + i] = acc;
        if (do_denoise) {
            albedo_fb[(size_t)j * W + i] = alb;
            normal_fb[(size_t)j * W + i] = nrm;
        }
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
        // Joint bilateral on guides; plain filter kept for no-guide use.
        fb = joint_bilateral_denoise(fb, albedo_fb, normal_fb, W, H);
        secs += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - d0)
                    .count();
    }

    std::filesystem::create_directories("out");
    if (!write_ppm("out/image.ppm", fb, W, H, exposure)) {
        std::cerr << "write failed\n";
        return 1;
    }
    if (!hdr_path.empty() && !write_pfm(hdr_path.c_str(), fb, W, H)) {
        std::cerr << "hdr dump failed\n";
        return 1;
    }
    std::uint64_t rays = bench_rays().load();
    std::cout << "wrote out/image.ppm " << W << "x" << H << " spp=" << spp
              << " threads=" << num_threads << " tile=" << tile_rows
              << " split=" << (use_sah ? "sah" : "median")
              << " exposure=" << exposure << " scene=" << scene_name
              << " denoise=" << (do_denoise ? "joint" : "off")
              << " shutter=[" << shutter0 << "," << shutter1 << "]"
              << " fog=" << fog_density << " het=" << het_density << " seed=" << base_seed << "\n";
    std::cout << "render " << secs << "s";
    if (bench) {
        std::cout << " rays=" << rays << " (" << (rays / 1e6 / secs) << " Mrays/s)"
                  << " box=" << bench_boxes().load()
                  << " prim=" << bench_prims().load();
    }
    std::cout << "\n";
    return 0;
}
