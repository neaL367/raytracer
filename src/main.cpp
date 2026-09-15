#include "core/vec3.h"
#include "core/ray.h"
#include "core/sphere.h"
#include "core/hittable_list.h"
#include "core/camera.h"
#include "core/random.h"
#include "core/triangle.h"
#include "core/quad.h"
#include "core/integrator.h"
#include "core/obj_loader.h"
#include "core/bvh.h"
#include "core/bench_stats.h"
#include "scene/scene.h"
#include "app/config.h"
#include "render/renderer.h"
#include "io/denoise.h"
#include "io/oidn_denoise.h"
#include "io/ppm.h"

#include <iostream>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>

int main(int argc, char **argv)
{
    render_config cfg;
    switch (parse_cli(argc, argv, cfg))
    {
    case cli_result::help:
        return 0;
    case cli_result::error:
        return 1;
    case cli_result::run:
        break;
    }
    // Seed before the first random draw so every later one — scene scatter
    // and BVH splits included — follows the fixed sequence.
    if (cfg.bench)
        set_deterministic_rng(true, cfg.bench_seed);

    auto total_start = std::chrono::high_resolution_clock::now();

    const int image_width = 800;
    const int image_height = static_cast<int>(image_width / (16.0 / 9.0));

    // One construction order shared with the GPU uploader (scene/scene.h).
    scene_data scene = build_scene(cfg);
    hittable_list &flat_objects = scene.objects;

    // --- build the BVH once, from the fully flattened list ---
    auto bvh_start = std::chrono::high_resolution_clock::now();
    hittable_list bvh_world;
    auto bvh_root = std::make_shared<bvh_node>(flat_objects, cfg.max_leaf_size);
    bvh_world.add(bvh_root);
    if (scene.ground_outside)
        bvh_world.add(scene.ground_outside);
    auto bvh_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> bvh_elapsed = bvh_end - bvh_start;

    size_t bvh_nodes = 0, bvh_leaves = 0, bvh_depth = 0;
    if (cfg.bench)
        bvh_root->census(bvh_nodes, bvh_leaves, bvh_depth);

    camera cam = default_camera(cfg.aperture);

    // Sole owner, called through the base interface: the render loop never
    // names the strategy, so swapping touches the lines below only. One
    // virtual call per primary ray — inaudible next to the millions of
    // virtual hit() calls inside each path.
    std::unique_ptr<integrator> tracer;
    if (cfg.shade_mode == "normal")
        tracer = std::make_unique<normal_integrator>();
    else
        tracer = std::make_unique<path_tracer>(cfg.do_nee, cfg.do_rr, cfg.fog_density);

    std::vector<vec3> framebuffer(image_width * image_height);
    render_stats stats = render_framebuffer(cam, bvh_world, scene.lights, *tracer, cfg,
                                            image_width, image_height, framebuffer);

    double denoise_seconds = 0.0;
    std::string denoise_used = "off";

    // Guide buffers for denoisers: first-hit albedo + normals under the same
    // sampling, so every pixel lines up with the beauty pass. Separate
    // single-purpose integrators — no interface change, negligible cost
    // (primary rays only, no bounces). OIDN forces them on: unguided OIDN
    // is strictly worse than guided, so there is no unguided code path.
    std::vector<vec3> albedo_fb(image_width * image_height);
    std::vector<vec3> normal_fb(image_width * image_height);
    bool need_aov = cfg.do_aov || cfg.do_oidn;
    if (need_aov)
    {
        albedo_integrator albedo_tracer;
        normal_integrator normal_tracer;
        render_framebuffer(cam, bvh_world, scene.lights, albedo_tracer, cfg,
                           image_width, image_height, albedo_fb);
        render_framebuffer(cam, bvh_world, scene.lights, normal_tracer, cfg,
                           image_width, image_height, normal_fb);
    }
    if (cfg.do_aov)
    {
        write_ppm("albedo.ppm", albedo_fb, image_width, image_height, cfg.exposure, false);
        write_ppm("normal.ppm", normal_fb, image_width, image_height, cfg.exposure, false);
        std::cout << "Wrote albedo.ppm + normal.ppm\n";
    }

    if (cfg.do_oidn)
    {
        auto denoise_start = std::chrono::high_resolution_clock::now();
        // OIDN speaks float triplets; the framebuffer is double.
        size_t px = static_cast<size_t>(image_width) * image_height;
        std::vector<float> color(px * 3), albedo(px * 3), normal(px * 3);
        for (size_t i = 0; i < px; ++i)
        {
            color[i * 3] = static_cast<float>(framebuffer[i].x());
            color[i * 3 + 1] = static_cast<float>(framebuffer[i].y());
            color[i * 3 + 2] = static_cast<float>(framebuffer[i].z());
            albedo[i * 3] = static_cast<float>(albedo_fb[i].x());
            albedo[i * 3 + 1] = static_cast<float>(albedo_fb[i].y());
            albedo[i * 3 + 2] = static_cast<float>(albedo_fb[i].z());
            normal[i * 3] = static_cast<float>(normal_fb[i].x());
            normal[i * 3 + 1] = static_cast<float>(normal_fb[i].y());
            normal[i * 3 + 2] = static_cast<float>(normal_fb[i].z());
        }
        if (oidn_denoise_rt(color.data(), albedo.data(), normal.data(), image_width, image_height))
        {
            for (size_t i = 0; i < px; ++i)
                framebuffer[i] = vec3(color[i * 3], color[i * 3 + 1], color[i * 3 + 2]);
            denoise_used = "oidn";
        }
        else if (cfg.do_denoise)
        {
            // OIDN unavailable or failed: bilateral instead of nothing.
            std::cout << "oidn failed: falling back to bilateral\n";
        }
        denoise_seconds = std::chrono::duration<double>(
                              std::chrono::high_resolution_clock::now() - denoise_start)
                              .count();
    }
    if (cfg.do_denoise && denoise_used != "oidn")
    {
        auto denoise_start = std::chrono::high_resolution_clock::now();
        framebuffer = bilateral_denoise(framebuffer, image_width, image_height);
        denoise_seconds = std::chrono::duration<double>(
                              std::chrono::high_resolution_clock::now() - denoise_start)
                              .count();
        denoise_used = "bilateral";
    }

    std::uint64_t image_hash = write_ppm("output.ppm", framebuffer, image_width, image_height,
                                         cfg.exposure, cfg.bench);
    std::cout << "Wrote output.ppm\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;
    std::cout << "Render time: " << stats.render_seconds << " seconds\n";

    if (cfg.bench)
    {
        double t_min = stats.thread_seconds[0], t_max = stats.thread_seconds[0], t_sum = 0.0;
        for (double t : stats.thread_seconds)
        {
            t_min = std::min(t_min, t);
            t_max = std::max(t_max, t);
            t_sum += t;
        }
        double t_mean = t_sum / stats.thread_seconds.size();
        std::uint64_t box_sum = 0, prim_sum = 0;
        for (size_t k = 0; k < stats.thread_seconds.size(); ++k)
        {
            box_sum += stats.thread_box[k];
            prim_sum += stats.thread_prim[k];
        }
        std::cout << "[bench] seed=" << cfg.bench_seed
                  << " spheres=" << cfg.extra_spheres
                  << " samples=" << cfg.samples_per_pixel
                  << " depth=" << cfg.max_depth
                  << " tile_rows=" << cfg.tile_rows
                  << " aperture=" << cfg.aperture
                  << " leaf=" << cfg.max_leaf_size
                  << " ground=" << (cfg.ground_in_bvh ? "in" : "out")
                  << " nee=" << (cfg.do_nee ? "on" : "off")
                  << " glass=" << (cfg.use_glass ? "on" : "off")
                  << " fog=" << cfg.fog_density
                  << " denoise=" << denoise_used
                  << " aov=" << (cfg.do_aov ? "on" : "off")
                  << " shade=" << cfg.shade_mode
                  << " rr=" << (cfg.do_rr ? "on" : "off")
                  << " strat=" << (cfg.stratified ? "on" : "off")
                  << " bilinear=" << (cfg.do_bilinear ? "on" : "off")
                  << " exposure=" << cfg.exposure
                  << " threads=" << stats.num_threads << "\n";
        std::cout << "[bench] bvh_build=" << bvh_elapsed.count() << "s"
                  << " nodes=" << bvh_nodes
                  << " leaves=" << bvh_leaves
                  << " max_depth=" << bvh_depth << "\n";
        std::cout << "[bench] render=" << stats.render_seconds << "s"
                  << " denoise=" << denoise_seconds << "s"
                  << " total=" << total_elapsed.count() << "s\n";
        std::cout << "[bench] per_thread=[";
        for (size_t k = 0; k < stats.thread_seconds.size(); ++k)
            std::cout << (k ? "," : "") << stats.thread_seconds[k];
        std::cout << "] mean=" << t_mean
                  << " straggler_gap=" << (t_max - t_mean) << "\n";
        std::cout << "[bench] box_tests=" << box_sum << " prim_tests=" << prim_sum << "\n";
        std::cout << "[bench] image_hash=" << image_hash << "\n";
    }

    return 0;
}
