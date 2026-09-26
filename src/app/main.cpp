#include "core/vec3.h"
#include "core/ray.h"
#include "core/random.h"
#include "core/sampler.h"
#include "core/bench_stats.h"
#include "camera/camera.h"
#include "geometry/hittable.h"
#include "geometry/quad.h"
#include "accel/bvh.h"
#include "accel/qbvh.h"
#include "accel/qbvh8.h"
#include "integrator/integrator.h"
#include "scene/scene.h"
#include "io/denoise.h"
#include "output/ppm.h"
#include "output/pfm.h"

#include <algorithm>
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
    int tile_rows = 32;
    bool bench = false;
    bool do_denoise = false;
    std::string scene_name = "showcase";
    double shutter0 = 0, shutter1 = 0;
    double fog_density = 0;
    double het_density = 0;
    bool marble_demo = false;
    bool env_demo = false;
    bool dump_aov = false; // --aov: albedo/normal/depth PFM trio next to PPM
    bool fixed_rng = false; // --fixed-rng: deterministic 0.5 stream (M54)
    std::string hdri_env_path; // --hdri-env <file.hdr>
    unsigned seed = 42; // base RNG seed; per-pixel stream = seed + pixel index
    int W = 400, H = -1; // H defaults to 16:9 unless --height given
    int max_depth = 50; // RR handles termination; depth is backstop
    bool spp_set = false;
    bool width_set = false;
    std::string hdr_path; // empty = no float dump
    std::string out_path = "out/image.ppm";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--samples" || a == "--spp") && i + 1 < argc) {
            spp = std::max(1, std::atoi(argv[++i]));
            spp_set = true;
        } else if (a == "--aperture" && i + 1 < argc)
            aperture = std::max(0.0, std::atof(argv[++i]));
        else if (a == "--threads" && i + 1 < argc)
            num_threads = (unsigned)std::max(1, std::atoi(argv[++i]));
        else if (a == "--tile" && i + 1 < argc)
            tile_rows = std::max(1, std::atoi(argv[++i]));
        else if (a == "--exposure" && i + 1 < argc)
            exposure = std::max(0.0, std::atof(argv[++i]));
        else if (a == "--denoise")
            do_denoise = true;
        else if (a == "--scene" && i + 1 < argc)
            scene_name = argv[++i];
        else if (a == "--list-scenes") {
            print_scenes();
            return 0;
        }
        else if (a == "--shutter" && i + 2 < argc) {
            shutter0 = std::atof(argv[++i]);
            shutter1 = std::atof(argv[++i]);
        } else if (a == "--fog" && i + 1 < argc) {
            fog_density = std::max(0.0, std::atof(argv[++i]));
        } else if (a == "--het" && i + 1 < argc) {
            het_density = std::max(0.0, std::atof(argv[++i]));
        } else if (a == "--noise")
            marble_demo = true;
        else if (a == "--env")
            env_demo = true;
        else if (a == "--seed" && i + 1 < argc)
            seed = (unsigned)std::max(0, std::atoi(argv[++i]));
        else if (a == "--width" && i + 1 < argc) {
            W = std::max(8, std::atoi(argv[++i]));
            width_set = true;
        } else if (a == "--height" && i + 1 < argc)
            H = std::max(8, std::atoi(argv[++i]));
        else if (a == "--bench")
            bench = true;
        else if (a == "--hdr" && i + 1 < argc)
            hdr_path = argv[++i];
        else if (a == "--aov")
            dump_aov = true;
        else if (a == "--fixed-rng")
            fixed_rng = true;
        else if (a == "--maxdepth" && i + 1 < argc)
            max_depth = std::max(1, std::atoi(argv[++i]));
        else if (a == "--hdri-env" && i + 1 < argc)
            hdri_env_path = argv[++i];
        else if (a.ends_with(".ppm"))
            out_path = a;
    }
    // "showcase" defaults: 1200 width, 675 height, 512 spp (unless explicitly overridden).
    if (scene_name == "showcase" || scene_name == "show" || scene_name == "demo") {
        if (!width_set)
            W = 1200;
        if (H <= 0)
            H = 675;
        if (!spp_set)
            spp = 512;
    }
    if (H <= 0)
        H = static_cast<int>(W / (16.0 / 9.0));
    const unsigned base_seed = seed;
    mip_render_h() = H; // LOD seam: texture minification follows output height

    // One construction order shared with the GPU uploader (scene/scene.h).
    scene_data scene = build_scene(scene_name, double(W) / double(H), aperture, shutter0,
                                    shutter1, fog_density, het_density, marble_demo,
                                    env_demo);
    // M63: load HDRI if requested. This also activates the env path (nenv_mode==2).
    if (!hdri_env_path.empty()) {
        scene.hdri = std::make_shared<hdri_env>();
        if (!scene.hdri->load(hdri_env_path)) {
            std::cerr << "hdri-env: could not load '" << hdri_env_path << "'\n";
            scene.hdri.reset();
        } else {
            scene.env_light = true; // activate env MIS path
            scene.black_bg = false; // HDRI replaces the studio-void background
            env_demo = true; // report the active environment in render metadata
        }
    }
    std::vector<std::shared_ptr<hittable>> &objs = scene.objs;
    std::vector<light> &lights = scene.lights;
    camera &cam = scene.cam;

    // SAH QBVH over everything incl. lights: fp32 8-wide traversal.
    qbvh8_node world(objs, 0, objs.size());

    // Power CDF for NEE (built once, read-only during render).
    std::vector<double> light_cdf;
    double light_total = 0.0;
    if (!lights.empty())
        build_light_cdf(lights, light_cdf, light_total);

    integrator tracer;
    render_params params{world,   lights,    max_depth, scene.media,
                         scene.env_light, scene.black_bg};
    params.light_cdf = light_cdf.empty() ? nullptr : &light_cdf;
    params.light_power_total = light_total;
    std::vector<vec3> fb((size_t)W * H);
    // Guide buffers only under --denoise/--aov: default pixels byte-exact.
    std::vector<vec3> albedo_fb((size_t)W * H), normal_fb((size_t)W * H);
    std::vector<vec3> depth_fb((size_t)W * H); // x = first-hit t, -1 on miss
    const bool want_guides = do_denoise || dump_aov;
    bench_enabled_flag().store(bench, std::memory_order_relaxed);

    // Two-wave adaptive sampling, budget-neutral: wave A covers all tiles
    // at sppA, then the noisier half of tiles gets topped to spp. Combined
    // mean weighted by sample counts stays unbiased; streams decorrelate
    // by seed xor. Deterministic: same pixels, same seeds, same hot set.
    // Fixed-RNG stays single-wave uniform: it is the cross-backend
    // metrology instrument, and adaptive counts would break exactness.
    // The GPU stays uniform (no two-wave mirror): cross-backend parity is
    // statistical by design, and fixed metrology needs matched counts.
    const int sppA = fixed_rng ? spp : std::max(1, spp / 2);
    const int sppB = spp - sppA;
    const unsigned seedB = base_seed ^ 0x9E3779B9u;
    std::vector<double> sumsq((size_t)W * H, 0.0); // luminance sum of squares
    std::vector<int> cnt((size_t)W * H, 0); // samples per pixel
    std::vector<double> dep_sum((size_t)W * H, 0.0);
    std::vector<int> dep_cnt((size_t)W * H, 0);

    auto render_pixel = [&](int i, int j, int wave_spp, unsigned wave_seed) {
        vec3 acc(0, 0, 0), alb(0, 0, 0), nrm(0, 0, 0);
        rng_seed(wave_seed + (unsigned)(j * W + i));
        static thread_local std::vector<sample_offset> offs;
        fill_pixel_samples(wave_spp, offs);
        size_t p = (size_t)j * W + i;
        for (auto [ox, oy] : offs) {
            double u = (i + ox) / W;
            double v = (j + oy) / H;
            ray primary = cam.get_ray(u, v);
            vec3 col = tracer.Li(primary, params);
            acc += col;
            sumsq[p] += denoise_luminance(col) * denoise_luminance(col);
            // Guides appended after beauty: deterministic order, and
            // AOV uses no RNG so the beauty stream never shifts.
            if (want_guides) {
                vec3 a, n;
                bool hit = false;
                double t = -1;
                first_hit_aov(primary, world, a, n, hit, &t);
                if (hit) {
                    alb += a;
                    nrm += n;
                    dep_sum[p] += t;
                    dep_cnt[p] += 1;
                }
            }
        }
        fb[p] += acc;
        cnt[p] += (int)offs.size();
        if (want_guides) {
            albedo_fb[p] += alb;
            normal_fb[p] += nrm;
        }
    };

    auto t_start = std::chrono::high_resolution_clock::now();
    const int tile_w = 32;
    const int tile_h = tile_rows;
    const int tiles_x = (W + tile_w - 1) / tile_w;
    const int tiles_y = (H + tile_h - 1) / tile_h;
    const int num_tiles = tiles_x * tiles_y;
    std::atomic<int> completed_tiles{0};
    int total_work = num_tiles;
    auto run_wave = [&](const std::vector<int> &tiles, int wave_spp, unsigned wave_seed,
                        const char *tag) {
        std::atomic<int> next_tile{0};
        const int nw = (int)tiles.size();
        std::vector<std::thread> workers;
        for (unsigned t = 0; t < num_threads; ++t) {
            workers.emplace_back([&] {
                if (fixed_rng)
                    rng_fixed_flag() = true; // thread-local: enable per worker
                // M63: bind HDRI pointer thread-locally so env_light dispatches
                // to it. Null when no HDRI loaded (analytic path unchanged).
                env_light::g_hdri() = scene.hdri ? scene.hdri.get() : nullptr;
                for (;;) {
                    int k = next_tile.fetch_add(1, std::memory_order_relaxed);
                    if (k >= nw)
                        break;
                    int ti = tiles[(size_t)k];
                    int tx = ti % tiles_x;
                    int ty = ti / tiles_x;
                    int x0 = tx * tile_w;
                    int x1 = std::min(x0 + tile_w, W);
                    int y0 = ty * tile_h;
                    int y1 = std::min(y0 + tile_h, H);
                    for (int j = y0; j < y1; ++j)
                        for (int i = x0; i < x1; ++i)
                            render_pixel(i, j, wave_spp, wave_seed);
                    int done = completed_tiles.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (done % 5 == 0 || done == total_work) {
                        std::cerr << "\rRendering " << tag << " [" << done * 100 / total_work
                                  << "%] (" << done << "/" << total_work << " tiles)"
                                  << std::flush;
                    }
                }
            });
        }
        for (auto &th : workers)
            th.join();
    };
    std::vector<int> all_tiles((size_t)num_tiles);
    for (int ti = 0; ti < num_tiles; ++ti)
        all_tiles[(size_t)ti] = ti;
    run_wave(all_tiles, sppA, base_seed, "A");
    // Hot half by luminance variance; wave B tops them to full spp.
    std::vector<int> hot_tiles;
    if (sppB > 0 && num_tiles > 1) {
        std::vector<double> score((size_t)num_tiles, 0.0);
        for (int ti = 0; ti < num_tiles; ++ti) {
            int tx = ti % tiles_x, ty = ti / tiles_x;
            int x0 = tx * tile_w, x1 = std::min(x0 + tile_w, W);
            int y0 = ty * tile_h, y1 = std::min(y0 + tile_h, H);
            double m1 = 0, m2 = 0;
            int npx = 0;
            for (int j = y0; j < y1; ++j)
                for (int i = x0; i < x1; ++i) {
                    size_t q = (size_t)j * W + i;
                    double c = cnt[q] > 0 ? 1.0 / cnt[q] : 0.0;
                    m1 += denoise_luminance(fb[q]) * c;
                    m2 += sumsq[q] * c * c;
                    ++npx;
                }
            m1 /= npx;
            m2 /= npx;
            score[(size_t)ti] = std::max(m2 - m1 * m1, 0.0);
        }
        std::vector<int> order = all_tiles;
        std::nth_element(order.begin(), order.begin() + order.size() / 2, order.end(),
                         [&](int a, int b) { return score[(size_t)a] > score[(size_t)b]; });
        hot_tiles.assign(order.begin(), order.begin() + order.size() / 2);
        total_work = num_tiles + (int)hot_tiles.size();
        run_wave(hot_tiles, sppB, seedB, "B");
    }
    std::cerr << "\n";
    // Finalize sums to means.
    for (size_t p = 0; p < fb.size(); ++p) {
        if (cnt[p] > 0)
            fb[p] = fb[p] / (double)cnt[p];
        if (want_guides) {
            if (cnt[p] > 0)
                albedo_fb[p] = albedo_fb[p] / (double)cnt[p];
            if (normal_fb[p].length_squared() > 0)
                normal_fb[p] = unit_vector(normal_fb[p]);
            depth_fb[p] = vec3(dep_cnt[p] > 0 ? dep_sum[p] / dep_cnt[p] : -1.0,
                               dep_cnt[p] > 0 ? dep_sum[p] / dep_cnt[p] : -1.0,
                               dep_cnt[p] > 0 ? dep_sum[p] / dep_cnt[p] : -1.0);
        }
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();
    double eff_spp = spp;
    {
        double csum = 0;
        for (int c : cnt)
            csum += c;
        if (!cnt.empty())
            eff_spp = csum / cnt.size();
    }

    if (do_denoise) {
        auto d0 = std::chrono::high_resolution_clock::now();
        // Joint bilateral on guides; plain filter kept for no-guide use.
        fb = joint_bilateral_denoise(fb, albedo_fb, normal_fb, W, H);
        secs += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - d0)
                    .count();
    }

    if (auto p = std::filesystem::path(out_path).parent_path(); !p.empty())
        std::filesystem::create_directories(p);
    else
        std::filesystem::create_directories("out");
    if (!write_ppm(out_path.c_str(), fb, W, H, exposure)) {
        std::cerr << "write failed\n";
        return 1;
    }
    if (!hdr_path.empty() && !write_pfm(hdr_path.c_str(), fb, W, H)) {
        std::cerr << "hdr dump failed\n";
        return 1;
    }
    if (dump_aov) {
        // Linear PFM trio for denoise/ML workflows (no film curve applied).
        if (!write_pfm("out/aov_albedo.pfm", albedo_fb, W, H) ||
            !write_pfm("out/aov_normal.pfm", normal_fb, W, H) ||
            !write_pfm("out/aov_depth.pfm", depth_fb, W, H)) {
            std::cerr << "aov dump failed\n";
            return 1;
        }
    }
    std::uint64_t rays = bench_rays().load();
    std::cout << "wrote " << out_path << " " << W << "x" << H << " spp=" << spp
              << " (eff " << eff_spp << ")"
              << " threads=" << num_threads << " tile=" << tile_rows
              << " exposure=" << exposure << " scene=" << scene_name
              << " denoise=" << (do_denoise ? "joint" : "off")
              << " shutter=[" << shutter0 << "," << shutter1 << "]"
              << " fog=" << fog_density << " het=" << het_density
              << " noise=" << (marble_demo ? "on" : "off")
              << " env=" << (env_demo ? "on" : "off")
              << " aov=" << (dump_aov ? "on" : "off") << " seed=" << base_seed << "\n";
    std::cout << "render " << secs << "s";
    if (bench) {
        std::cout << " rays=" << rays << " (" << (rays / 1e6 / secs) << " Mrays/s)"
                  << " box=" << bench_boxes().load()
                  << " prim=" << bench_prims().load();
    }
    std::cout << "\n";
    return 0;
}
