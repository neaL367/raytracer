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

#include <fstream>
#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cmath>

int main(int argc, char **argv)
{
    // Optional flags: --bench prints timing/counter report with fixed RNG seed,
    // --spheres N sets added random spheres (default 300), --samples N sets
    // samples per pixel (default 196 = 14x14 strata; must be a perfect square
    // for stratification, otherwise plain jitter is used), --tile N sets
    // scheduling strip height in rows (default 8), --leaf N sets BVH leaf
    // capacity (default 2), --ground outside tests the ground sphere
    // separately from the tree, --nee enables next-event estimation
    // against an overhead area light, --norr disables russian roulette
    // (paths always run to full depth), --nostrat disables stratified
    // pixel sampling (pure jitter instead), --seed N sets the fixed
    // RNG seed (default 42).
    bool bench = false;
    int extra_spheres = 300;
    int samples_per_pixel = 196;
    int tile_rows = 8;
    unsigned bench_seed = 42u;
    size_t max_leaf_size = 2;
    bool ground_in_bvh = true;
    bool do_nee = false;
    bool do_rr = true;
    bool do_strat = true;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--bench")
            bench = true;
        else if (arg == "--spheres" && i + 1 < argc)
            extra_spheres = std::stoi(argv[++i]);
        else if (arg == "--samples" && i + 1 < argc)
            samples_per_pixel = std::stoi(argv[++i]);
        else if (arg == "--tile" && i + 1 < argc)
            tile_rows = std::stoi(argv[++i]);
        else if (arg == "--leaf" && i + 1 < argc)
            max_leaf_size = static_cast<size_t>(std::stoul(argv[++i]));
        else if (arg == "--ground" && i + 1 < argc)
            ground_in_bvh = (std::string(argv[++i]) != "outside");
        else if (arg == "--nee")
            do_nee = true;
        else if (arg == "--norr")
            do_rr = false;
        else if (arg == "--nostrat")
            do_strat = false;
        else if (arg == "--seed" && i + 1 < argc)
            bench_seed = static_cast<unsigned>(std::stoul(argv[++i]));
    }
    // Seed before the first random draw so every later one — scene scatter
    // and BVH splits included — follows the fixed sequence.
    if (bench)
        set_deterministic_rng(true, bench_seed);

    auto total_start = std::chrono::high_resolution_clock::now();

    // Stratified pixel sampling: the pixel is an n x n grid of strata with
    // one jittered sample each, so samples spread evenly instead of clumping.
    // Needs a perfect square count; anything else (or --nostrat) falls back
    // to plain jitter, which draws every sample over the whole pixel.
    int strat_n = static_cast<int>(std::sqrt(samples_per_pixel + 0.5));
    bool stratified = do_strat && strat_n * strat_n == samples_per_pixel && strat_n > 0;
    if (bench && do_strat && !stratified)
        std::cout << "[bench] samples=" << samples_per_pixel << " not square: jitter fallback\n";

    const int image_width = 800;
    const int image_height = static_cast<int>(image_width / (16.0 / 9.0));
    const int max_depth = 50;

    // --- build a FLAT list of every individual primitive, no nested hittable_lists ---
    hittable_list flat_objects;

    auto material_ground = std::make_shared<lambertian>(
        std::make_shared<checker_texture>(0.32, vec3(0.8, 0.8, 0.8), vec3(0.2, 0.2, 0.2)));
    auto material_center = std::make_shared<lambertian>(
        std::make_shared<image_texture>("assets/uv_check.ppm"));
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
    if (do_nee)
    {
        auto light_mat = std::make_shared<diffuse_light>(vec3(3, 3, 3));
        auto area_light = std::make_shared<quad>(
            vec3(-2, 5, -2), vec3(4, 0, 0), vec3(0, 0, 4), light_mat);
        lights.push_back(area_light);
        flat_objects.add(area_light);
    }

    // --- scale the scene up to a size where a BVH actually pays off ---
    for (int i = 0; i < extra_spheres; ++i)
    {
        vec3 center(random_double(-10, 10), random_double(-10, 10), random_double(-15, -5));
        flat_objects.add(std::make_shared<sphere>(center, 0.2, material_mesh));
    }

    // The ground is a radius-100 sphere whose box overlaps nearly everything.
    // Inside the tree it poisons every ancestor box it touches; tested
    // separately the tree only holds finite objects. Closest-hit logic makes
    // both placements render identically — this flag measures the cost gap.
    auto ground = std::make_shared<sphere>(vec3(0, -100.5, -1), 100, material_ground);
    if (ground_in_bvh)
        flat_objects.add(ground);

    std::cout << "Total flat primitives: " << flat_objects.size() << "\n";

    // --- build the BVH once, from the fully flattened list ---
    auto bvh_start = std::chrono::high_resolution_clock::now();
    hittable_list bvh_world;
    auto bvh_root = std::make_shared<bvh_node>(flat_objects, max_leaf_size);
    bvh_world.add(bvh_root);
    if (!ground_in_bvh)
        bvh_world.add(ground);
    auto bvh_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> bvh_elapsed = bvh_end - bvh_start;

    size_t bvh_nodes = 0, bvh_leaves = 0, bvh_depth = 0;
    if (bench)
        bvh_root->census(bvh_nodes, bvh_leaves, bvh_depth);

    vec3 lookfrom(4, 3, 5);
    vec3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    double dist_to_focus = (lookfrom - lookat).length();
    double aperture = 0.05;

    camera cam(lookfrom, lookat, vup, 25, 16.0 / 9.0, aperture, dist_to_focus);

    // Sole owner, called through the base interface: the render loop below
    // never names path_tracer, so swapping strategies touches one line.
    // One virtual call per primary ray — inaudible next to the millions of
    // virtual hit() calls inside each path.
    std::unique_ptr<integrator> tracer = std::make_unique<path_tracer>(do_nee, do_rr);

    std::vector<vec3> framebuffer(image_width * image_height);

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 4; // fallback if the system can't report a count

    std::vector<std::thread> threads;
    std::vector<double> thread_seconds(num_threads, 0.0);
    std::vector<std::uint64_t> thread_box(num_threads, 0);
    std::vector<std::uint64_t> thread_prim(num_threads, 0);

    bench_enabled_flag().store(bench, std::memory_order_relaxed);

    auto render_start = std::chrono::high_resolution_clock::now();

    // Workers grab row-strips from a shared atomic counter until none remain.
    // Relaxed ordering suffices: the counter only mints unique indices, it
    // publishes no data and enforces no ordering. Larger strips mean fewer
    // atomic grabs but coarser balancing; smaller strips balance better at
    // the cost of more grabs and worse cache reuse.
    const int num_tiles = (image_height + tile_rows - 1) / tile_rows;
    std::atomic<int> next_tile{0};

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
            int row_start = tile * tile_rows;
            int row_end = std::min(row_start + tile_rows, image_height);
            // Reseed per tile so a given tile always draws the same random
            // stream regardless of which worker thread renders it.
            if (bench)
                reseed_thread_rng(bench_seed + static_cast<unsigned>(tile));
            for (int j = row_start; j < row_end; ++j)
        {
            for (int i = 0; i < image_width; ++i)
            {
                vec3 pixel_color(0, 0, 0);

                for (int sample = 0; sample < samples_per_pixel; ++sample)
                {
                    // Stratum (sx, sy) from the sample index; the jitter stays
                    // inside it. Same two random draws per sample as jitter,
                    // only the mapping from draw to pixel position changes.
                    double ox = random_double();
                    double oy = random_double();
                    if (stratified)
                    {
                        ox = ((sample % strat_n) + ox) / strat_n;
                        oy = ((sample / strat_n) + oy) / strat_n;
                    }
                    double s = (i + ox) / (image_width - 1);
                    double t = (j + oy) / (image_height - 1);

                    ray r = cam.get_ray(s, t);
                    pixel_color += tracer->Li(r, bvh_world, lights, max_depth);
                    // pixel_color += ray_color(r, flat_objects, max_depth);
                }

                double scale = 1.0 / samples_per_pixel;
                framebuffer[j * image_width + i] = vec3(
                    std::sqrt(pixel_color.x() * scale),
                    std::sqrt(pixel_color.y() * scale),
                    std::sqrt(pixel_color.z() * scale));
            }
        }
        } // end tile-grab loop
        auto t1 = std::chrono::high_resolution_clock::now();
        // Disjoint indices: no lock needed, each thread writes only its slots.
        thread_seconds[thread_idx] = std::chrono::duration<double>(t1 - t0).count();
        thread_box[thread_idx] = thread_box_tests() - b0;
        thread_prim[thread_idx] = thread_prim_tests() - p0;
    };

    for (unsigned int t = 0; t < num_threads; ++t)
        threads.emplace_back(render_rows, t);

    for (auto &th : threads)
        th.join();

    auto render_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> render_elapsed = render_end - render_start;
    bench_enabled_flag().store(false, std::memory_order_relaxed);

    // FNV-1a 64 over the exact quantized bytes written to the PPM, computed
    // inline with the write loop so hashing costs one pass, not two.
    std::uint64_t image_hash = 1469598103934665603ULL;
    auto hash_byte = [&](unsigned char b)
    {
        image_hash ^= b;
        image_hash *= 1099511628211ULL;
    };

    // single-threaded: write the completed framebuffer out to the PPM file
    std::ofstream out("output.ppm");
    out << "P3\n"
        << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j)
    {
        for (int i = 0; i < image_width; ++i)
        {
            const vec3 &c = framebuffer[j * image_width + i];
            // Clamp: lit values can exceed 1 (no tonemapper yet); raw >255
            // bytes are malformed P3, so clip at the displayable range.
            int ir = std::min(255, static_cast<int>(255.999 * c.x()));
            int ig = std::min(255, static_cast<int>(255.999 * c.y()));
            int ib = std::min(255, static_cast<int>(255.999 * c.z()));
            out << ir << ' ' << ig << ' ' << ib << '\n';
            if (bench)
            {
                hash_byte(static_cast<unsigned char>(ir));
                hash_byte(static_cast<unsigned char>(ig));
                hash_byte(static_cast<unsigned char>(ib));
            }
        }
    }

    std::cout << "Wrote output.ppm\n";

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;
    std::cout << "Render time: " << render_elapsed.count() << " seconds\n";

    if (bench)
    {
        double t_min = thread_seconds[0], t_max = thread_seconds[0], t_sum = 0.0;
        for (double t : thread_seconds)
        {
            t_min = std::min(t_min, t);
            t_max = std::max(t_max, t);
            t_sum += t;
        }
        double t_mean = t_sum / thread_seconds.size();
        std::cout << "[bench] seed=" << bench_seed
                  << " spheres=" << extra_spheres
                  << " samples=" << samples_per_pixel
                  << " tile_rows=" << tile_rows
                  << " leaf=" << max_leaf_size
                  << " ground=" << (ground_in_bvh ? "in" : "out")
                  << " nee=" << (do_nee ? "on" : "off")
                  << " rr=" << (do_rr ? "on" : "off")
                  << " strat=" << (stratified ? "on" : "off")
                  << " threads=" << num_threads << "\n";
        std::cout << "[bench] bvh_build=" << bvh_elapsed.count() << "s"
                  << " nodes=" << bvh_nodes
                  << " leaves=" << bvh_leaves
                  << " max_depth=" << bvh_depth << "\n";
        std::cout << "[bench] render=" << render_elapsed.count() << "s"
                  << " total=" << total_elapsed.count() << "s\n";
        std::cout << "[bench] per_thread=[";
        for (size_t k = 0; k < thread_seconds.size(); ++k)
            std::cout << (k ? "," : "") << thread_seconds[k];
        std::cout << "] mean=" << t_mean
                  << " straggler_gap=" << (t_max - t_mean) << "\n";
        std::cout << "[bench] box_tests=" << [&] {
            std::uint64_t s = 0;
            for (auto n : thread_box)
                s += n;
            return s;
        }() << " prim_tests=" << [&] {
            std::uint64_t s = 0;
            for (auto n : thread_prim)
                s += n;
            return s;
        }() << "\n";
        std::cout << "[bench] image_hash=" << image_hash << "\n";
    }

    return 0;
}