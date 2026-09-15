#include "core/vec3.h"
#include "core/ray.h"
#include "core/sphere.h"
#include "core/hittable_list.h"
#include "core/camera.h"
#include "core/random.h"
#include "core/triangle.h"
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

vec3 ray_color(const ray &r, const hittable &world, int depth)
{
    if (depth <= 0)
        return vec3(0, 0, 0);

    hit_record rec;
    if (world.hit(r, 0.001, 1000.0, rec))
    {
        ray scattered;
        vec3 attenuation;
        if (rec.mat->scatter(r, rec, attenuation, scattered))
        {
            return attenuation * ray_color(scattered, world, depth - 1);
        }
        return vec3(0, 0, 0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    double a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.5, 0.7, 1.0);
}

int main(int argc, char **argv)
{
    // --- CLI: --bench enables deterministic RNG + stats + report ------------
    // --spheres N scales the scene (default 300), --samples N overrides
    // samples/pixel (default 200), --seed N sets the bench seed (default 42).
    bool bench = false;
    int extra_spheres = 300;
    int samples_per_pixel = 200;
    unsigned bench_seed = 42u;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--bench")
            bench = true;
        else if (arg == "--spheres" && i + 1 < argc)
            extra_spheres = std::stoi(argv[++i]);
        else if (arg == "--samples" && i + 1 < argc)
            samples_per_pixel = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc)
            bench_seed = static_cast<unsigned>(std::stoul(argv[++i]));
    }
    // Must precede ANY random_double use (scene scatter + BVH axis picks).
    if (bench)
        set_deterministic_rng(true, bench_seed);

    auto total_start = std::chrono::high_resolution_clock::now();

    const int image_width = 800;
    const int image_height = static_cast<int>(image_width / (16.0 / 9.0));
    const int max_depth = 50;

    // --- build a FLAT list of every individual primitive, no nested hittable_lists ---
    hittable_list flat_objects;

    auto material_ground = std::make_shared<lambertian>(vec3(0.8, 0.8, 0.0));
    auto material_center = std::make_shared<lambertian>(vec3(1.0, 0.0, 0.0));
    auto material_right = std::make_shared<metal>(vec3(0.8, 0.6, 0.2));
    auto material_triangle = std::make_shared<lambertian>(vec3(0.2, 0.8, 0.2));
    auto material_mesh = std::make_shared<lambertian>(vec3(0.6, 0.6, 0.6));

    auto mesh = load_obj("assets/model.obj", material_mesh);
    std::cout << "Loaded " << mesh->size() << " triangles\n";

    // flatten the mesh's triangles directly into flat_objects, instead of nesting the list
    for (const auto &tri : mesh->objects_ref())
        flat_objects.add(tri);

    flat_objects.add(std::make_shared<sphere>(vec3(0, -100.5, -1), 100, material_ground));
    flat_objects.add(std::make_shared<sphere>(vec3(0, 0, -1), 0.5, material_center));
    flat_objects.add(std::make_shared<sphere>(vec3(1, 0, -1), 0.5, material_right));
    flat_objects.add(std::make_shared<triangle>(
        vec3(-1, -1, -2), vec3(1, -1, -2), vec3(0, 1, -2),
        material_triangle));

    // --- scale the scene up to a size where a BVH actually pays off ---
    for (int i = 0; i < extra_spheres; ++i)
    {
        vec3 center(random_double(-10, 10), random_double(-10, 10), random_double(-15, -5));
        flat_objects.add(std::make_shared<sphere>(center, 0.2, material_mesh));
    }

    std::cout << "Total flat primitives: " << flat_objects.size() << "\n";

    // --- build the BVH once, from the fully flattened list ---
    auto bvh_start = std::chrono::high_resolution_clock::now();
    hittable_list bvh_world;
    auto bvh_root = std::make_shared<bvh_node>(flat_objects);
    bvh_world.add(bvh_root);
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

    // NOTE: static strip assignment (kept for D0 baseline; D1 replaces this
    // with a tile queue). The lambda takes its thread index by value so it
    // can record its own busy time without any shared-state writes.
    auto render_rows = [&](unsigned thread_idx, int row_start, int row_end)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        // Snapshot this worker's own thread_local counters; the deltas are
        // this strip's true cost, readable without any synchronization.
        std::uint64_t b0 = thread_box_tests(), p0 = thread_prim_tests();
        for (int j = row_start; j < row_end; ++j)
        {
            for (int i = 0; i < image_width; ++i)
            {
                vec3 pixel_color(0, 0, 0);

                for (int sample = 0; sample < samples_per_pixel; ++sample)
                {
                    double s = (i + random_double()) / (image_width - 1);
                    double t = (j + random_double()) / (image_height - 1);

                    ray r = cam.get_ray(s, t);
                    pixel_color += ray_color(r, bvh_world, max_depth);
                    // pixel_color += ray_color(r, flat_objects, max_depth);
                }

                double scale = 1.0 / samples_per_pixel;
                framebuffer[j * image_width + i] = vec3(
                    std::sqrt(pixel_color.x() * scale),
                    std::sqrt(pixel_color.y() * scale),
                    std::sqrt(pixel_color.z() * scale));
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        // Disjoint indices: no lock needed, each thread writes only its slots.
        thread_seconds[thread_idx] = std::chrono::duration<double>(t1 - t0).count();
        thread_box[thread_idx] = thread_box_tests() - b0;
        thread_prim[thread_idx] = thread_prim_tests() - p0;
    };

    int rows_per_thread = image_height / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t)
    {
        int row_start = t * rows_per_thread;
        int row_end = (t == num_threads - 1) ? image_height : row_start + rows_per_thread;
        threads.emplace_back(render_rows, t, row_start, row_end);
    }

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
            int ir = static_cast<int>(255.999 * c.x());
            int ig = static_cast<int>(255.999 * c.y());
            int ib = static_cast<int>(255.999 * c.z());
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