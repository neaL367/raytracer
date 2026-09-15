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
#include "app/config.h"
#include "render/renderer.h"
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

    // --- build a FLAT list of every individual primitive, no nested hittable_lists ---
    hittable_list flat_objects;

    auto material_ground = std::make_shared<lambertian>(
        std::make_shared<checker_texture>(0.32, vec3(0.8, 0.8, 0.8), vec3(0.2, 0.2, 0.2)));
    auto material_center = std::make_shared<lambertian>(
        std::make_shared<image_texture>("assets/uv_check.ppm", cfg.do_bilinear));
    auto material_right = std::make_shared<metal>(vec3(0.8, 0.6, 0.2));
    auto material_triangle = std::make_shared<lambertian>(vec3(0.2, 0.8, 0.2));
    auto material_mesh = std::make_shared<lambertian>(vec3(0.6, 0.6, 0.6));

    auto mesh = load_obj("assets/model.obj", material_mesh);
    std::cout << "Loaded " << mesh->size() << " triangles\n";

    // flatten the mesh's triangles directly into flat_objects, instead of nesting the list
    for (const auto &tri : mesh->objects_ref())
        flat_objects.add(tri);

    flat_objects.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, material_center));
    flat_objects.add(std::make_shared<sphere>(vec3(1, 0, -1), 0.5, material_right));
    flat_objects.add(std::make_shared<triangle>(
        vec3(-1, -1, -2), vec3(1, -1, -2), vec3(0, 1, -2),
        material_triangle));

    // Overhead area light. Lives outside the BVH input unless NEE is on, so
    // the default path renders the exact historical scene (anchor hash holds).
    std::vector<std::shared_ptr<quad>> lights;
    if (cfg.do_nee)
    {
        auto light_mat = std::make_shared<diffuse_light>(vec3(3, 3, 3));
        auto area_light = std::make_shared<quad>(
            vec3(-2, 5, -2), vec3(4, 0, 0), vec3(0, 0, 4), light_mat);
        lights.push_back(area_light);
        flat_objects.add(area_light);
    }

    // --- scale the scene up to a size where a BVH actually pays off ---
    for (int i = 0; i < cfg.extra_spheres; ++i)
    {
        vec3 center(random_double(-10, 10), random_double(-10, 10), random_double(-15, -5));
        flat_objects.add(std::make_shared<sphere>(center, 0.2, material_mesh));
    }

    // The ground is a radius-100 sphere whose box overlaps nearly everything.
    // Inside the tree it poisons every ancestor box it touches; tested
    // separately the tree only holds finite objects. Closest-hit logic makes
    // both placements render identically — this flag measures the cost gap.
    auto ground = std::make_shared<sphere>(vec3(0, -100.5, -1), 100, material_ground);
    if (cfg.ground_in_bvh)
        flat_objects.add(ground);

    std::cout << "Total flat primitives: " << flat_objects.size() << "\n";

    // --- build the BVH once, from the fully flattened list ---
    auto bvh_start = std::chrono::high_resolution_clock::now();
    hittable_list bvh_world;
    auto bvh_root = std::make_shared<bvh_node>(flat_objects, cfg.max_leaf_size);
    bvh_world.add(bvh_root);
    if (!cfg.ground_in_bvh)
        bvh_world.add(ground);
    auto bvh_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> bvh_elapsed = bvh_end - bvh_start;

    size_t bvh_nodes = 0, bvh_leaves = 0, bvh_depth = 0;
    if (cfg.bench)
        bvh_root->census(bvh_nodes, bvh_leaves, bvh_depth);

    vec3 lookfrom(4, 3, 5);
    vec3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    double dist_to_focus = (lookfrom - lookat).length();
    double aperture = 0.05;

    camera cam(lookfrom, lookat, vup, 25, 16.0 / 9.0, aperture, dist_to_focus);

    // Sole owner, called through the base interface: the render loop never
    // names path_tracer, so swapping strategies touches one line. One virtual
    // call per primary ray — inaudible next to the millions of virtual hit()
    // calls inside each path.
    std::unique_ptr<integrator> tracer = std::make_unique<path_tracer>(cfg.do_nee, cfg.do_rr);

    std::vector<vec3> framebuffer(image_width * image_height);
    render_stats stats = render_framebuffer(cam, bvh_world, lights, *tracer, cfg,
                                            image_width, image_height, framebuffer);

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
                  << " leaf=" << cfg.max_leaf_size
                  << " ground=" << (cfg.ground_in_bvh ? "in" : "out")
                  << " nee=" << (cfg.do_nee ? "on" : "off")
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
